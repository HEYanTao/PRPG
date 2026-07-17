"""Lean deterministic calendar-month generation and CSV delivery for PRPG v5.

The scientific operation is intentionally small: a K4 state chain selects one
complete synchronized historical month for every model month.  Daily vectors
are copied unchanged, and every coarser frequency is a stable sum of those
daily log returns.  Random streams are keyed by global path identity, so
process scheduling and shard layout cannot change consumer bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from prpg.errors import GenerationError, IntegrityError
from prpg.execution import deterministic_process_map
from prpg.model.artifact import ModelArtifactStore
from prpg.model.hmm import deterministic_hmm_restart_seeds
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    Domain,
    Family,
    ModelFitScope,
    RNGKey,
    Stage,
    categorical_from_uniform,
    uniforms,
)
from prpg.storage.csv_v1 import format_csv_float64
from prpg.v5.config import ResolvedV5Inputs, load_v5_inputs
from prpg.v5.data import load_v5_month_library
from prpg.v5.hmm import K4_REVIEW_SCHEMA

V5_GENERATOR_VERSION: Final = "prpg-v5-calendar-month-generator-v1"
V5_CSV_SCHEMA_VERSION: Final = 2
V5_CSV_SERIALIZER_VERSION: Final = "prpg-v5-csv-v2-cpython-17g"
V5_MANIFEST_SCHEMA: Final = "prpg-v5-generation-manifest-v1"

DAILY_HEADER: Final = (
    "path_id,period_index,simulation_year,model_month,session_in_month,"
    "equity_log_return,muni_bond_log_return,taxable_bond_log_return"
)
COARSE_HEADER: Final = (
    "path_id,period_index,simulation_year,period_in_year,"
    "equity_log_return,muni_bond_log_return,taxable_bond_log_return"
)

V5Role = Literal["training", "validation", "test"]
V5Frequency = Literal["daily", "monthly", "quarterly", "annual"]
V5GenerationProfile = Literal["smoke", "parity", "pilot", "configured", "canonical"]
V5Profile = V5GenerationProfile

_ROLES: Final[tuple[V5Role, ...]] = ("training", "validation", "test")
_FREQUENCIES: Final[tuple[V5Frequency, ...]] = (
    "daily",
    "monthly",
    "quarterly",
    "annual",
)
_PROFILE_NAMES: Final = frozenset(
    {"smoke", "parity", "pilot", "configured", "canonical"}
)
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40,64}\Z")
_PATH_ID = re.compile(r"P[0-9]{6}\Z")
_WORKER_CACHE: tuple[str, V5GenerationInput] | None = None
_CSV_BYTES_PER_ROW_ESTIMATE: Final = 128
_CANONICAL_MINIMUM_FREE_DISK_BYTES: Final = 15 * 1024**3


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise GenerationError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationError(f"{label} must be a positive integer")
    return value


def _uint128(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= (1 << 128) - 1
    ):
        raise GenerationError(f"{label} must be an unsigned 128-bit integer")
    return value


def _frozen_array(
    value: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike,
    label: str,
) -> np.ndarray[Any, Any]:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if result.dtype.kind == "f" and not bool(np.isfinite(result).all()):
        raise IntegrityError(f"{label} contains non-finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class V5GenerationInput:
    """Verified numerical input for serial and multicore v5 generation."""

    master_seed: int
    scientific_version: int
    maximum_path_entity: int
    config_fingerprint: str
    seed_registry_fingerprint: str
    data_fingerprint: str
    model_fingerprint: str
    model_config_fingerprint: str
    model_fit_seed_fingerprint: str
    feature_names: tuple[str, str, str]
    daily_returns: np.ndarray[Any, Any]
    month_start_offsets: np.ndarray[Any, Any]
    monthly_period_ordinals: np.ndarray[Any, Any]
    decoded_states: np.ndarray[Any, Any]
    transition_matrix: np.ndarray[Any, Any]
    stationary_distribution: np.ndarray[Any, Any]
    state_pools: tuple[np.ndarray[Any, Any], ...] = field(init=False, repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.scientific_version, "scientific version")
        maximum = _positive_int(self.maximum_path_entity, "maximum path entity")
        _uint128(self.master_seed, "master seed")
        for value, label in (
            (self.config_fingerprint, "config fingerprint"),
            (self.seed_registry_fingerprint, "seed-registry fingerprint"),
            (self.data_fingerprint, "data fingerprint"),
            (self.model_fingerprint, "model fingerprint"),
            (self.model_config_fingerprint, "model-config fingerprint"),
            (self.model_fit_seed_fingerprint, "model-fit-seed fingerprint"),
        ):
            _fingerprint(value, label)
        if (
            len(self.feature_names) != 3
            or len(set(self.feature_names)) != 3
            or any(not isinstance(name, str) or not name for name in self.feature_names)
        ):
            raise IntegrityError(
                "v5 generation requires three distinct configured feature tickers"
            )

        daily = _frozen_array(
            self.daily_returns, dtype=np.float64, label="daily return library"
        )
        offsets = _frozen_array(
            self.month_start_offsets, dtype=np.int64, label="month offsets"
        )
        ordinals = _frozen_array(
            self.monthly_period_ordinals, dtype=np.int64, label="month ordinals"
        )
        states = _frozen_array(
            self.decoded_states, dtype=np.int64, label="decoded states"
        )
        transition = _frozen_array(
            self.transition_matrix, dtype=np.float64, label="transition matrix"
        )
        stationary = _frozen_array(
            self.stationary_distribution,
            dtype=np.float64,
            label="stationary distribution",
        )
        month_count = len(ordinals)
        if daily.ndim != 2 or daily.shape[1] != 3 or len(daily) == 0:
            raise IntegrityError("v5 daily return library has invalid geometry")
        if (
            offsets.ndim != 1
            or len(offsets) != month_count + 1
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(daily)
            or bool((np.diff(offsets) <= 0).any())
        ):
            raise IntegrityError("v5 month offsets have invalid geometry")
        if (
            ordinals.ndim != 1
            or states.shape != (month_count,)
            or month_count == 0
            or bool((np.diff(ordinals) <= 0).any())
            or bool(((states < 0) | (states >= 4)).any())
        ):
            raise IntegrityError("v5 model month alignment is invalid")
        if transition.shape != (4, 4) or bool((transition < 0.0).any()):
            raise IntegrityError("v5 transition matrix is invalid")
        if not np.allclose(transition.sum(axis=1), np.ones(4), rtol=0.0, atol=1e-12):
            raise IntegrityError("v5 transition rows do not sum to one")
        if stationary.shape != (4,) or bool((stationary < 0.0).any()):
            raise IntegrityError("v5 stationary distribution is invalid")
        if not math.isclose(
            float(stationary.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ) or not np.allclose(stationary @ transition, stationary, rtol=0.0, atol=1e-12):
            raise IntegrityError("v5 stationary distribution does not match K4")
        pools = tuple(
            _frozen_array(
                np.flatnonzero(states == state),
                dtype=np.int64,
                label=f"state-{state} source pool",
            )
            for state in range(4)
        )
        if any(pool.ndim != 1 or len(pool) == 0 for pool in pools):
            raise IntegrityError("v5 generation requires four non-empty source pools")

        assignments: tuple[tuple[str, object], ...] = (
            ("daily_returns", daily),
            ("month_start_offsets", offsets),
            ("monthly_period_ordinals", ordinals),
            ("decoded_states", states),
            ("transition_matrix", transition),
            ("stationary_distribution", stationary),
            ("state_pools", pools),
        )
        for name, item in assignments:
            object.__setattr__(self, name, item)
        identity = {
            "schema": "prpg-v5-generation-input-v1",
            "generator_version": V5_GENERATOR_VERSION,
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "master_seed": self.master_seed,
            "scientific_version": self.scientific_version,
            "maximum_path_entity": maximum,
            "config_fingerprint": self.config_fingerprint,
            "seed_registry_fingerprint": self.seed_registry_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "model_fit_seed_fingerprint": self.model_fit_seed_fingerprint,
            "feature_names": list(self.feature_names),
            "state_pool_sizes": [len(pool) for pool in pools],
        }
        object.__setattr__(self, "fingerprint", _sha256_json(identity))


def load_v5_generation_input(
    inputs: ResolvedV5Inputs,
    processed_data_fingerprint: str,
    model_fingerprint: str,
    *,
    path_generation_seed: int | None = None,
) -> V5GenerationInput:
    """Verify model/data and construct input for the requested path RNG seed."""

    _fingerprint(processed_data_fingerprint, "processed-data fingerprint")
    _fingerprint(model_fingerprint, "approved K4 fingerprint")
    library = load_v5_month_library(inputs, processed_data_fingerprint)
    store = ModelArtifactStore(inputs.artifact_root)
    reference = store.verify(model_fingerprint)
    identity = reference.manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise IntegrityError("v5 model artifact identity is missing")
    if identity.get("data_fingerprint") != processed_data_fingerprint:
        raise IntegrityError("v5 model was not fitted to the selected month library")
    metadata = identity.get("model_metadata")
    provenance = identity.get("provenance")
    if not isinstance(metadata, Mapping) or not isinstance(provenance, Mapping):
        raise IntegrityError("v5 model metadata or provenance is missing")
    if (
        metadata.get("schema") != K4_REVIEW_SCHEMA
        or tuple(cast(Sequence[object], metadata.get("feature_names", ())))
        != library.feature_names
        or metadata.get("pool_floor_passed") is not True
        or metadata.get("state_counts") is None
    ):
        raise IntegrityError("v5 model is not the reviewed usable return-K4 artifact")
    if provenance.get("processed_data_fingerprint") != processed_data_fingerprint:
        raise IntegrityError("v5 model provenance differs from the processed data")
    model_config_fingerprint, model_fit_seed_fingerprint = (
        _verify_scoped_model_fit_provenance(inputs, store, model_fingerprint)
    )
    arrays = store.load_arrays(model_fingerprint)
    required = {
        "decoded_states",
        "monthly_period_ordinals",
        "transition_matrix",
        "stationary_distribution",
    }
    if not required.issubset(arrays):
        raise IntegrityError("v5 model artifact is missing generation arrays")
    if not np.array_equal(
        arrays["monthly_period_ordinals"], library.monthly_period_ordinals
    ):
        raise IntegrityError("v5 model months are misaligned with the source library")
    result = V5GenerationInput(
        master_seed=(
            inputs.config.run.master_seed
            if path_generation_seed is None
            else _uint128(path_generation_seed, "path-generation seed")
        ),
        scientific_version=inputs.config.model.scientific_version,
        maximum_path_entity=inputs.config.simulation.paths,
        config_fingerprint=inputs.config_fingerprint,
        seed_registry_fingerprint=inputs.seed_registry_fingerprint,
        data_fingerprint=processed_data_fingerprint,
        model_fingerprint=model_fingerprint,
        model_config_fingerprint=model_config_fingerprint,
        model_fit_seed_fingerprint=model_fit_seed_fingerprint,
        feature_names=library.feature_names,
        daily_returns=library.daily_returns,
        month_start_offsets=library.month_start_offsets,
        monthly_period_ordinals=library.monthly_period_ordinals,
        decoded_states=arrays["decoded_states"],
        transition_matrix=arrays["transition_matrix"],
        stationary_distribution=arrays["stationary_distribution"],
    )
    expected_counts = metadata.get("state_counts")
    if [len(pool) for pool in result.state_pools] != list(
        cast(Sequence[object], expected_counts)
    ):
        raise IntegrityError("v5 source pools differ from reviewed K4 state counts")
    if any(
        len(pool) < inputs.config.model.minimum_state_months
        for pool in result.state_pools
    ):
        raise IntegrityError("v5 source pool fell below the approved support floor")
    return result


def _verify_scoped_model_fit_provenance(
    inputs: ResolvedV5Inputs,
    store: ModelArtifactStore,
    model_fingerprint: str,
) -> tuple[str, str]:
    """Bind a fitted model only to inputs that can change its calibration.

    The historical v5 artifact recorded the whole application configuration.
    Its verified review document also records the actual estimator settings and
    every deterministic restart seed.  Comparing those scoped values preserves
    the original artifact while allowing output geometry and execution settings
    to change without an unnecessary HMM refit.
    """

    review = store.read_document(model_fingerprint, "k4-calibration-review.json")
    expected_model_python = inputs.config.model.model_dump(mode="python")
    if review.get("estimator_settings") != expected_model_python:
        raise IntegrityError("v5 model estimator settings differ from active inputs")
    expected_model = inputs.config.model.model_dump(mode="json")
    diagnostics = review.get("restart_diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(
        diagnostics, str | bytes | bytearray
    ):
        raise IntegrityError("v5 model restart provenance is missing")
    observed: list[int] = []
    for index, value in enumerate(diagnostics):
        if (
            not isinstance(value, Mapping)
            or value.get("restart_index") != index
            or isinstance(value.get("seed"), bool)
            or not isinstance(value.get("seed"), int)
        ):
            raise IntegrityError("v5 model restart provenance is invalid")
        observed.append(cast(int, value["seed"]))
    expected = deterministic_hmm_restart_seeds(
        master_seed=inputs.config.run.master_seed,
        scientific_version=inputs.config.model.scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=inputs.config.model.states,
        restart_count=inputs.config.model.deterministic_restarts,
    )
    if tuple(observed) != expected:
        raise IntegrityError("v5 model-fit seed scope differs from active inputs")
    return (
        _sha256_json(
            {
                "schema": "prpg-v5-model-config-scope-v1",
                "model": expected_model,
            }
        ),
        _sha256_json(
            {
                "schema": "prpg-v5-model-fit-seed-scope-v1",
                "restart_seeds": list(expected),
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class V5PathSelection:
    """The deterministic hidden state and source-month sequence for one path."""

    path_entity: int
    states: np.ndarray[Any, Any]
    source_month_indices: np.ndarray[Any, Any]


def _path_entity(value: object, maximum: int) -> int:
    entity = _positive_int(value, "path entity")
    if entity > maximum:
        raise GenerationError("path entity exceeds the configured v5 namespace")
    return entity


def _horizon_months(value: object) -> int:
    months = _positive_int(value, "horizon months")
    if months % 12:
        raise GenerationError("v5 horizon must contain complete model years")
    return months


def select_v5_path(
    generation_input: V5GenerationInput,
    path_entity: int,
    horizon_months: int,
) -> V5PathSelection:
    """Select exactly one K4 state and one independent source month per month."""

    if not isinstance(generation_input, V5GenerationInput):
        raise GenerationError("v5 path selection requires a generation input")
    entity = _path_entity(path_entity, generation_input.maximum_path_entity)
    months = _horizon_months(horizon_months)
    state_key = RNGKey(
        master_seed=generation_input.master_seed,
        scientific_version=generation_input.scientific_version,
        domain=Domain.PRODUCTION,
        family=Family.NOT_APPLICABLE,
        entity=entity,
        stage=Stage.STATE_CHAIN,
    )
    source_key = RNGKey(
        master_seed=generation_input.master_seed,
        scientific_version=generation_input.scientific_version,
        domain=Domain.PRODUCTION,
        family=Family.NOT_APPLICABLE,
        entity=entity,
        stage=Stage.SOURCE_RANK_UNIFORMS,
    )
    state_draws = uniforms(state_key, months)
    source_draws = uniforms(source_key, months)
    states = np.empty(months, dtype=np.int64)
    sources = np.empty(months, dtype=np.int64)
    states[0] = categorical_from_uniform(
        cast(Sequence[float], generation_input.stationary_distribution),
        float(state_draws[0]),
    )
    for month in range(1, months):
        states[month] = categorical_from_uniform(
            cast(
                Sequence[float],
                generation_input.transition_matrix[int(states[month - 1])],
            ),
            float(state_draws[month]),
        )
    for month, (state, draw) in enumerate(zip(states, source_draws, strict=True)):
        pool = generation_input.state_pools[int(state)]
        rank = min(math.floor(float(draw) * len(pool)), len(pool) - 1)
        sources[month] = int(pool[rank])
    states.setflags(write=False)
    sources.setflags(write=False)
    return V5PathSelection(entity, states, sources)


@dataclass(frozen=True, slots=True)
class V5MonthBlock:
    """One selected synchronized source month in consumer asset order."""

    model_month_index: int
    state: int
    source_month_index: int
    daily_log_returns: np.ndarray[Any, Any]


def iter_v5_month_blocks(
    generation_input: V5GenerationInput,
    selection: V5PathSelection,
) -> Iterator[V5MonthBlock]:
    """Yield whole selected blocks without source-date or successor logic."""

    if not isinstance(selection, V5PathSelection):
        raise GenerationError("v5 block iteration requires a path selection")
    if selection.states.shape != selection.source_month_indices.shape:
        raise IntegrityError("v5 path selection arrays are misaligned")
    offsets = generation_input.month_start_offsets
    for month, (state, source) in enumerate(
        zip(selection.states, selection.source_month_indices, strict=True), start=1
    ):
        source_index = int(source)
        if source_index not in generation_input.state_pools[int(state)]:
            raise IntegrityError("v5 selected source month is outside its state pool")
        start = int(offsets[source_index])
        stop = int(offsets[source_index + 1])
        # The immutable library follows configured feature roles
        # equity/taxable/muni; consumers require equity/muni/taxable, hence the
        # single explicit [0,2,1] reorder independent of ticker symbols.
        block = np.ascontiguousarray(
            generation_input.daily_returns[start:stop][:, [0, 2, 1]],
            dtype=np.float64,
        )
        block.setflags(write=False)
        yield V5MonthBlock(month, int(state), source_index, block)


def _stable_column_sum(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result = np.asarray(
        [math.fsum(float(value) for value in values[:, column]) for column in range(3)],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class V5MaterializedPath:
    """One path and all daily-authoritative derived frequency arrays."""

    path_entity: int
    selection: V5PathSelection
    month_lengths: np.ndarray[Any, Any]
    daily_log_returns: np.ndarray[Any, Any]
    monthly_log_returns: np.ndarray[Any, Any]
    quarterly_log_returns: np.ndarray[Any, Any]
    annual_log_returns: np.ndarray[Any, Any]


def materialize_v5_path(
    generation_input: V5GenerationInput,
    path_entity: int,
    horizon_months: int,
) -> V5MaterializedPath:
    """Materialize one path, deriving all coarse logs with ``math.fsum``."""

    selection = select_v5_path(generation_input, path_entity, horizon_months)
    blocks = tuple(iter_v5_month_blocks(generation_input, selection))
    if len(blocks) != horizon_months:
        raise IntegrityError("v5 path did not produce its complete month horizon")
    month_lengths = _frozen_array(
        [len(block.daily_log_returns) for block in blocks],
        dtype=np.int64,
        label="generated month lengths",
    )
    daily = _frozen_array(
        np.concatenate([block.daily_log_returns for block in blocks], axis=0),
        dtype=np.float64,
        label="generated daily path",
    )
    monthly = _frozen_array(
        np.vstack([_stable_column_sum(block.daily_log_returns) for block in blocks]),
        dtype=np.float64,
        label="generated monthly path",
    )
    quarterly = _frozen_array(
        np.vstack(
            [
                _stable_column_sum(monthly[start : start + 3])
                for start in range(0, horizon_months, 3)
            ]
        ),
        dtype=np.float64,
        label="generated quarterly path",
    )
    annual = _frozen_array(
        np.vstack(
            [
                _stable_column_sum(monthly[start : start + 12])
                for start in range(0, horizon_months, 12)
            ]
        ),
        dtype=np.float64,
        label="generated annual path",
    )
    return V5MaterializedPath(
        path_entity=selection.path_entity,
        selection=selection,
        month_lengths=month_lengths,
        daily_log_returns=daily,
        monthly_log_returns=monthly,
        quarterly_log_returns=quarterly,
        annual_log_returns=annual,
    )


@dataclass(frozen=True, slots=True)
class V5PathRange:
    """One explicit, role-labelled inclusive range of global path entities."""

    role: V5Role
    first_path: int
    last_path: int

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise GenerationError("v5 path range role is invalid")
        first = _positive_int(self.first_path, "first path")
        last = _positive_int(self.last_path, "last path")
        if first > last:
            raise GenerationError("v5 path range is reversed")

    @property
    def path_count(self) -> int:
        return self.last_path - self.first_path + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "first_path": self.first_path,
            "last_path": self.last_path,
            "path_count": self.path_count,
        }


@dataclass(frozen=True, slots=True)
class V5WorkUnit:
    """One role-pure contiguous shard owned by exactly one worker."""

    work_unit_id: str
    role: V5Role
    shard_index: int
    first_path: int
    last_path: int

    @property
    def path_count(self) -> int:
        return self.last_path - self.first_path + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "work_unit_id": self.work_unit_id,
            "role": self.role,
            "shard_index": self.shard_index,
            "first_path": self.first_path,
            "last_path": self.last_path,
            "path_count": self.path_count,
        }


@dataclass(frozen=True, slots=True)
class V5GenerationPlan:
    """Exact selected path geometry and deterministic role-aligned sharding."""

    profile: V5GenerationProfile
    horizon_months: int
    paths_per_shard: int
    path_ranges: tuple[V5PathRange, ...]
    work_units: tuple[V5WorkUnit, ...] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_NAMES:
            raise GenerationError("v5 generation profile is invalid")
        _horizon_months(self.horizon_months)
        shard_size = _positive_int(self.paths_per_shard, "paths per shard")
        if not self.path_ranges:
            raise GenerationError("v5 generation plan has no paths")
        ordered = tuple(
            sorted(
                self.path_ranges,
                key=lambda item: (_ROLES.index(item.role), item.first_path),
            )
        )
        if ordered != self.path_ranges:
            raise GenerationError("v5 path ranges are not in canonical role order")
        occupied: set[int] = set()
        units: list[V5WorkUnit] = []
        shard_index = 0
        for path_range in ordered:
            values = set(range(path_range.first_path, path_range.last_path + 1))
            if occupied.intersection(values):
                raise GenerationError("v5 path ranges overlap")
            occupied.update(values)
            for first in range(
                path_range.first_path, path_range.last_path + 1, shard_size
            ):
                last = min(path_range.last_path, first + shard_size - 1)
                identifier = (
                    f"{path_range.role}-{shard_index:05d}-p{first:06d}-p{last:06d}"
                )
                units.append(
                    V5WorkUnit(
                        work_unit_id=identifier,
                        role=path_range.role,
                        shard_index=shard_index,
                        first_path=first,
                        last_path=last,
                    )
                )
                shard_index += 1
        object.__setattr__(self, "work_units", tuple(units))
        object.__setattr__(self, "fingerprint", _sha256_json(self.as_dict()))

    @property
    def path_count(self) -> int:
        return sum(value.path_count for value in self.path_ranges)

    @property
    def csv_file_count(self) -> int:
        return len(self.work_units) * len(_FREQUENCIES)

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "horizon_months": self.horizon_months,
            "paths_per_shard": self.paths_per_shard,
            "path_count": self.path_count,
            "path_ranges": [value.as_dict() for value in self.path_ranges],
            "work_unit_count": len(self.work_units),
            "csv_file_count": self.csv_file_count,
        }


def _configured_role_ranges(
    inputs: ResolvedV5Inputs,
) -> Mapping[V5Role, tuple[int, int]]:
    split = inputs.config.simulation.split
    training_last = split.training
    validation_last = training_last + split.validation
    total = validation_last + split.test
    result: dict[V5Role, tuple[int, int]] = {
        "training": (1, training_last),
        "validation": (training_last + 1, validation_last),
        "test": (validation_last + 1, total),
    }
    return MappingProxyType(result)


def _validate_plan_against_inputs(
    inputs: ResolvedV5Inputs, plan: V5GenerationPlan
) -> None:
    boundaries = _configured_role_ranges(inputs)
    for selected in plan.path_ranges:
        first, last = boundaries[selected.role]
        if not first <= selected.first_path <= selected.last_path <= last:
            raise GenerationError(
                "v5 selected path range is outside its configured role namespace"
            )


def build_v5_generation_plan(
    inputs: ResolvedV5Inputs,
    *,
    profile: V5GenerationProfile,
    path_ranges: Sequence[V5PathRange] | None = None,
    horizon_months: int | None = None,
    paths_per_shard: int | None = None,
) -> V5GenerationPlan:
    """Build a registered profile or an explicit role-aligned path plan."""

    if profile not in _PROFILE_NAMES:
        raise GenerationError("v5 generation profile is invalid")
    boundaries = _configured_role_ranges(inputs)
    selected: tuple[V5PathRange, ...]
    if path_ranges is None:
        if profile == "smoke":
            selected = (V5PathRange("training", 1, 1),)
            default_shard = 1
        elif profile == "parity":
            selected = (
                V5PathRange("training", 1, 70),
                V5PathRange(
                    "validation",
                    boundaries["validation"][0],
                    boundaries["validation"][0] + 9,
                ),
                V5PathRange("test", boundaries["test"][0], boundaries["test"][0] + 9),
            )
            default_shard = 10
        elif profile == "pilot":
            selected = (
                V5PathRange("training", 1, 400),
                V5PathRange(
                    "validation",
                    boundaries["validation"][0],
                    boundaries["validation"][0] + 49,
                ),
                V5PathRange("test", boundaries["test"][0], boundaries["test"][0] + 49),
            )
            default_shard = 50
        else:  # configured and canonical both select the full configured geometry
            selected = tuple(V5PathRange(role, *boundaries[role]) for role in _ROLES)
            default_shard = inputs.config.simulation.paths_per_shard
    else:
        selected = tuple(path_ranges)
        default_shard = inputs.config.simulation.paths_per_shard
    plan = V5GenerationPlan(
        profile=profile,
        horizon_months=(
            inputs.config.simulation.horizon_months
            if horizon_months is None
            else horizon_months
        ),
        paths_per_shard=(default_shard if paths_per_shard is None else paths_per_shard),
        path_ranges=selected,
    )
    _validate_plan_against_inputs(inputs, plan)
    if (
        profile == "canonical"
        and path_ranges is None
        and (
            plan.path_count != inputs.config.simulation.paths
            or len(plan.work_units) != 100
            or plan.csv_file_count != 400
        )
    ):
        raise IntegrityError("canonical v5 plan geometry changed")
    return plan


def _path_id_text(entity: int) -> str:
    value = f"P{entity:06d}"
    if _PATH_ID.fullmatch(value) is None:  # pragma: no cover - namespace bounds
        raise GenerationError("v5 path ID is outside the six-digit namespace")
    return value


@dataclass(frozen=True, slots=True)
class V5CSVFileRecord:
    frequency: V5Frequency
    relative_path: str
    rows: int
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frequency": self.frequency,
            "relative_path": self.relative_path,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class V5WorkResult:
    work_unit_id: str
    role: V5Role
    shard_index: int
    first_path: int
    last_path: int
    path_count: int
    daily_rows_minimum: int
    daily_rows_maximum: int
    daily_rows_sum: int
    files: tuple[V5CSVFileRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "work_unit_id": self.work_unit_id,
            "role": self.role,
            "shard_index": self.shard_index,
            "first_path": self.first_path,
            "last_path": self.last_path,
            "path_count": self.path_count,
            "daily_rows_minimum": self.daily_rows_minimum,
            "daily_rows_maximum": self.daily_rows_maximum,
            "daily_rows_sum": self.daily_rows_sum,
            "files": [value.as_dict() for value in self.files],
        }


class _DigestingCSV:
    def __init__(
        self,
        destination: Path,
        relative_path: str,
        frequency: V5Frequency,
        header: str,
    ) -> None:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        self.destination = destination
        self.relative_path = relative_path
        self.frequency = frequency
        self.temporary = Path(name)
        self.handle = os.fdopen(descriptor, "wb", closefd=True)
        self.digest = hashlib.sha256()
        self.byte_count = 0
        self.rows = 0
        self.closed = False
        self.write((header + "\n").encode("utf-8"), row=False)

    def write(self, content: bytes, *, row: bool = True) -> None:
        written = self.handle.write(content)
        if written != len(content):
            raise IntegrityError("v5 CSV temporary write was incomplete")
        self.digest.update(content)
        self.byte_count += len(content)
        if row:
            self.rows += 1

    def publish(self) -> V5CSVFileRecord:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.closed = True
        os.chmod(self.temporary, 0o600)
        try:
            os.link(self.temporary, self.destination)
        except FileExistsError as error:
            raise IntegrityError("v5 CSV destination already exists") from error
        except OSError as error:
            raise IntegrityError("v5 CSV could not be atomically published") from error
        _fsync_directory(self.destination.parent)
        self.temporary.unlink()
        return V5CSVFileRecord(
            frequency=self.frequency,
            relative_path=self.relative_path,
            rows=self.rows,
            bytes=self.byte_count,
            sha256=self.digest.hexdigest(),
        )

    def abort(self) -> None:
        if not self.closed:
            self.handle.close()
            self.closed = True
        with suppress(FileNotFoundError):
            self.temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise IntegrityError("v5 output directory fsync failed") from error


def _safe_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."} or "/" in part or "\\" in part:
            raise IntegrityError("v5 output relative path is unsafe")
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as error:
            raise IntegrityError("v5 output directory could not be created") from error
        if current.is_symlink() or not current.is_dir():
            raise IntegrityError("v5 output directory is unsafe")
    return current


def _csv_relative_path(unit: V5WorkUnit, frequency: V5Frequency) -> str:
    filename = (
        f"part-{unit.shard_index:05d}-path-"
        f"{_path_id_text(unit.first_path)}-{_path_id_text(unit.last_path)}.csv"
    )
    return PurePosixPath(
        "consumer", "returns", unit.role, frequency, filename
    ).as_posix()


def _encode_vector(vector: np.ndarray[Any, Any]) -> tuple[str, str, str]:
    if vector.shape != (3,) or vector.dtype != np.dtype(np.float64):
        raise IntegrityError("v5 output vector has invalid geometry")
    return cast(
        tuple[str, str, str],
        tuple(format_csv_float64(value) for value in vector),
    )


def _daily_row(
    path_id: str,
    period_index: int,
    simulation_year: int,
    model_month: int,
    session_in_month: int,
    vector: np.ndarray[Any, Any],
) -> bytes:
    fields = (
        path_id,
        str(period_index),
        str(simulation_year),
        str(model_month),
        str(session_in_month),
        *_encode_vector(vector),
    )
    return (",".join(fields) + "\n").encode("utf-8")


def _coarse_row(
    path_id: str,
    period_index: int,
    periods_per_year: int,
    vector: np.ndarray[Any, Any],
) -> bytes:
    zero = period_index - 1
    fields = (
        path_id,
        str(period_index),
        str(zero // periods_per_year + 1),
        str(zero % periods_per_year + 1),
        *_encode_vector(vector),
    )
    return (",".join(fields) + "\n").encode("utf-8")


def write_v5_work_unit(
    generation_input: V5GenerationInput,
    *,
    output_root: str | Path,
    work_unit: V5WorkUnit,
    horizon_months: int,
) -> V5WorkResult:
    """Write four atomic role-aligned CSV siblings for one deterministic shard."""

    root = Path(output_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("v5 output root is unsafe")
    targets: dict[V5Frequency, _DigestingCSV] = {}
    try:
        for frequency in _FREQUENCIES:
            relative = _csv_relative_path(work_unit, frequency)
            pure = PurePosixPath(relative)
            parent = _safe_directory(root, pure.parent)
            destination = parent / pure.name
            if destination.exists() or destination.is_symlink():
                raise IntegrityError("v5 CSV destination is not fresh")
            targets[frequency] = _DigestingCSV(
                destination,
                relative,
                frequency,
                DAILY_HEADER if frequency == "daily" else COARSE_HEADER,
            )
        daily_counts: list[int] = []
        for entity in range(work_unit.first_path, work_unit.last_path + 1):
            path = materialize_v5_path(generation_input, entity, horizon_months)
            path_id = _path_id_text(entity)
            daily_offset = 0
            period_index = 1
            for month_zero, length_value in enumerate(path.month_lengths):
                length = int(length_value)
                simulation_year = month_zero // 12 + 1
                model_month = month_zero % 12 + 1
                for session_zero in range(length):
                    targets["daily"].write(
                        _daily_row(
                            path_id,
                            period_index,
                            simulation_year,
                            model_month,
                            session_zero + 1,
                            path.daily_log_returns[daily_offset + session_zero],
                        )
                    )
                    period_index += 1
                daily_offset += length
            if daily_offset != len(path.daily_log_returns):
                raise IntegrityError("v5 daily month lengths did not consume the path")
            daily_counts.append(daily_offset)
            coarse_values: tuple[tuple[V5Frequency, np.ndarray[Any, Any], int], ...] = (
                ("monthly", path.monthly_log_returns, 12),
                ("quarterly", path.quarterly_log_returns, 4),
                ("annual", path.annual_log_returns, 1),
            )
            for frequency, values, periods_per_year in coarse_values:
                sink = targets[frequency]
                for zero, vector in enumerate(values):
                    sink.write(_coarse_row(path_id, zero + 1, periods_per_year, vector))
        records = tuple(targets[frequency].publish() for frequency in _FREQUENCIES)
    except BaseException:
        for target in targets.values():
            target.abort()
        raise
    return V5WorkResult(
        work_unit_id=work_unit.work_unit_id,
        role=work_unit.role,
        shard_index=work_unit.shard_index,
        first_path=work_unit.first_path,
        last_path=work_unit.last_path,
        path_count=work_unit.path_count,
        daily_rows_minimum=min(daily_counts),
        daily_rows_maximum=max(daily_counts),
        daily_rows_sum=sum(daily_counts),
        files=records,
    )


@dataclass(frozen=True, slots=True)
class V5GenerationLoader:
    """Small pickle-safe loader reverified once in every worker process."""

    config_path: str
    processed_data_fingerprint: str
    model_fingerprint: str
    config_fingerprint: str
    seed_registry_fingerprint: str
    path_generation_seed: int | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.config_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise GenerationError("v5 loader configuration path is unsafe")
        for value, label in (
            (self.processed_data_fingerprint, "processed-data fingerprint"),
            (self.model_fingerprint, "model fingerprint"),
            (self.config_fingerprint, "config fingerprint"),
            (self.seed_registry_fingerprint, "seed-registry fingerprint"),
        ):
            _fingerprint(value, label)
        if self.path_generation_seed is not None:
            _uint128(self.path_generation_seed, "path-generation seed")
        object.__setattr__(self, "config_path", str(path))
        object.__setattr__(self, "fingerprint", _sha256_json(self.as_dict()))

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "prpg-v5-generation-loader-v1",
            "config_path": self.config_path,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "seed_registry_fingerprint": self.seed_registry_fingerprint,
        }
        # Omitting the field for the default preserves the canonical loader
        # identity and all historical generation behavior.
        if self.path_generation_seed is not None:
            result["path_generation_seed"] = self.path_generation_seed
        return result


def build_v5_generation_loader(
    inputs: ResolvedV5Inputs,
    *,
    processed_data_fingerprint: str,
    model_fingerprint: str,
    path_generation_seed: int | None = None,
) -> V5GenerationLoader:
    """Bind a worker loader to the exact approved config/data/model/seed inputs."""

    # Verify once in the parent before any run root is created.
    load_v5_generation_input(
        inputs,
        processed_data_fingerprint,
        model_fingerprint,
        path_generation_seed=path_generation_seed,
    )
    return V5GenerationLoader(
        config_path=str(inputs.config_path),
        processed_data_fingerprint=processed_data_fingerprint,
        model_fingerprint=model_fingerprint,
        config_fingerprint=inputs.config_fingerprint,
        seed_registry_fingerprint=inputs.seed_registry_fingerprint,
        path_generation_seed=path_generation_seed,
    )


def _load_from_loader(loader: V5GenerationLoader) -> V5GenerationInput:
    if loader.fingerprint != _sha256_json(loader.as_dict()):
        raise IntegrityError("v5 generation loader fingerprint changed")
    inputs = load_v5_inputs(loader.config_path)
    if (
        inputs.config_fingerprint != loader.config_fingerprint
        or inputs.seed_registry_fingerprint != loader.seed_registry_fingerprint
    ):
        raise IntegrityError("v5 config or seed registry changed after loader creation")
    return load_v5_generation_input(
        inputs,
        loader.processed_data_fingerprint,
        loader.model_fingerprint,
        path_generation_seed=loader.path_generation_seed,
    )


@dataclass(frozen=True, slots=True)
class _WorkerTask:
    loader: V5GenerationLoader
    output_root: str
    work_unit: V5WorkUnit
    horizon_months: int


def _run_worker(task: _WorkerTask) -> V5WorkResult:
    global _WORKER_CACHE
    if _WORKER_CACHE is None or _WORKER_CACHE[0] != task.loader.fingerprint:
        _WORKER_CACHE = (task.loader.fingerprint, _load_from_loader(task.loader))
    return write_v5_work_unit(
        _WORKER_CACHE[1],
        output_root=task.output_root,
        work_unit=task.work_unit,
        horizon_months=task.horizon_months,
    )


@dataclass(frozen=True, slots=True)
class V5GenerationResult:
    """Parent-verified completed generation and its final publications."""

    run_root: Path
    manifest_path: Path
    checksum_path: Path
    manifest_fingerprint: str
    simulation_fingerprint: str
    materialization_fingerprint: str
    profile: V5GenerationProfile
    workers: int
    path_count: int
    work_units: int
    csv_files: int
    total_rows: int
    total_bytes: int
    wall_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "run_root": str(self.run_root),
            "manifest_path": str(self.manifest_path),
            "checksum_path": str(self.checksum_path),
            "manifest_fingerprint": self.manifest_fingerprint,
            "simulation_fingerprint": self.simulation_fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "profile": self.profile,
            "workers": self.workers,
            "path_count": self.path_count,
            "work_units": self.work_units,
            "csv_files": self.csv_files,
            "total_rows": self.total_rows,
            "total_bytes": self.total_bytes,
            "wall_seconds": self.wall_seconds,
        }


def _git_output(root: Path, *arguments: str, allow_empty: bool = False) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GenerationError("v5 source provenance is unavailable") from error
    output = completed.stdout.strip()
    if not output and not allow_empty:
        raise GenerationError("v5 source provenance is unavailable")
    return output


def _source_provenance(config_path: Path) -> dict[str, object]:
    project_root = config_path.parent.parent
    try:
        lock_bytes = (project_root / "requirements.lock").read_bytes()
    except OSError as error:
        raise GenerationError("v5 dependency lock is unavailable") from error
    commit = _git_output(project_root, "rev-parse", "--verify", "HEAD")
    if _COMMIT.fullmatch(commit) is None:
        raise GenerationError("v5 source commit is invalid")
    status = _git_output(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        allow_empty=True,
    )
    return {
        "git_commit": commit,
        "dirty_worktree": bool(status),
        "requirements_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }


def _disk_preflight(
    root: Path,
    generation_input: V5GenerationInput,
    plan: V5GenerationPlan,
    *,
    canonical: bool,
) -> dict[str, object]:
    month_lengths = np.diff(generation_input.month_start_offsets)
    maximum_daily_rows = (
        plan.path_count * plan.horizon_months * int(month_lengths.max())
    )
    coarse_rows = plan.path_count * (
        plan.horizon_months + plan.horizon_months // 3 + plan.horizon_months // 12
    )
    estimated_output_bytes = (
        maximum_daily_rows + coarse_rows
    ) * _CSV_BYTES_PER_ROW_ESTIMATE
    required = max(
        _CANONICAL_MINIMUM_FREE_DISK_BYTES,
        2 * estimated_output_bytes,
    )
    try:
        available = shutil.disk_usage(root.parent).free
    except OSError as error:
        raise GenerationError("v5 free disk capacity is unavailable") from error
    passed = available >= required
    if canonical and not passed:
        raise GenerationError(
            "insufficient free disk for canonical v5 production",
            details={
                "available_bytes": available,
                "required_bytes": required,
                "estimated_output_bytes": estimated_output_bytes,
            },
        )
    return {
        "method": "maximum_source_month_rows_times_128_bytes",
        "available_bytes": available,
        "estimated_output_bytes": estimated_output_bytes,
        "required_bytes": required,
        "canonical_gate_applied": canonical,
        "passed": passed,
    }


def _simulation_identity(
    generation_input: V5GenerationInput,
    plan: V5GenerationPlan,
    *,
    path_generation_seed_overridden: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "prpg-v5-simulation-identity-v1",
        "data_fingerprint": generation_input.data_fingerprint,
        "model_fingerprint": generation_input.model_fingerprint,
        "seed_registry_fingerprint": generation_input.seed_registry_fingerprint,
        "generator_version": V5_GENERATOR_VERSION,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "scientific_version": generation_input.scientific_version,
        "feature_order_internal": list(generation_input.feature_names),
        "consumer_asset_order": ["equity", "muni_bond", "taxable_bond"],
        "initial_state": "stationary",
        "source_selection": "uniform_floor_rank_with_replacement_per_model_month",
        "horizon_months": plan.horizon_months,
        "path_ranges": [value.as_dict() for value in plan.path_ranges],
    }
    if path_generation_seed_overridden:
        result["path_generation_seed"] = generation_input.master_seed
    return result


def _materialization_identity(
    simulation_fingerprint: str, plan: V5GenerationPlan
) -> dict[str, object]:
    return {
        "schema": "prpg-v5-materialization-identity-v1",
        "simulation_fingerprint": simulation_fingerprint,
        "csv_schema_version": V5_CSV_SCHEMA_VERSION,
        "csv_serializer_version": V5_CSV_SERIALIZER_VERSION,
        "return_representation": "log_total_decimal",
        "newline": "LF",
        "compression": "none",
        "frequencies": list(_FREQUENCIES),
        "plan": plan.as_dict(),
    }


def _consumer_assets(inputs: ResolvedV5Inputs) -> dict[str, str]:
    assets = inputs.config.data.assets
    return {
        "equity": assets.equity,
        "muni_bond": assets.muni_bond,
        "taxable_bond": assets.taxable_bond,
    }


def _toolchain_identity(source: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "prpg-v5-toolchain-identity-v1",
        "git_commit": source["git_commit"],
        "requirements_lock_sha256": source["requirements_lock_sha256"],
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "generator_version": V5_GENERATOR_VERSION,
        "csv_serializer_version": V5_CSV_SERIALIZER_VERSION,
    }


def _regular_file_hash(path: Path) -> tuple[int, str, int]:
    if not path.is_file() or path.is_symlink():
        raise IntegrityError("v5 consumer CSV is missing or unsafe")
    digest = hashlib.sha256()
    byte_count = 0
    newline_count = 0
    try:
        with path.open("rb") as stream:
            while content := stream.read(1024 * 1024):
                digest.update(content)
                byte_count += len(content)
                newline_count += content.count(b"\n")
    except OSError as error:
        raise IntegrityError("v5 consumer CSV could not be rehashed") from error
    if newline_count <= 1:
        raise IntegrityError("v5 consumer CSV has no data rows")
    return byte_count, digest.hexdigest(), newline_count - 1


def _verify_results(
    root: Path,
    plan: V5GenerationPlan,
    results: tuple[V5WorkResult, ...],
) -> tuple[V5WorkResult, ...]:
    if len(results) != len(plan.work_units):
        raise IntegrityError("v5 worker result set is incomplete")
    verified: list[V5WorkResult] = []
    for scope, result in zip(plan.work_units, results, strict=True):
        if (
            result.work_unit_id != scope.work_unit_id
            or result.role != scope.role
            or result.shard_index != scope.shard_index
            or result.first_path != scope.first_path
            or result.last_path != scope.last_path
            or result.path_count != scope.path_count
            or tuple(value.frequency for value in result.files) != _FREQUENCIES
        ):
            raise IntegrityError("v5 worker result scope or sibling order changed")
        for record in result.files:
            expected = _csv_relative_path(scope, record.frequency)
            if record.relative_path != expected:
                raise IntegrityError("v5 worker returned the wrong CSV path")
            byte_count, sha256, rows = _regular_file_hash(root / expected)
            if (
                byte_count != record.bytes
                or sha256 != record.sha256
                or rows != record.rows
            ):
                raise IntegrityError("v5 parent CSV rehash differs from worker receipt")
            if record.frequency != "daily":
                per_path = {
                    "monthly": plan.horizon_months,
                    "quarterly": plan.horizon_months // 3,
                    "annual": plan.horizon_months // 12,
                }[record.frequency]
                if record.rows != scope.path_count * per_path:
                    raise IntegrityError("v5 coarse CSV row count differs from plan")
        daily_record = result.files[0]
        if (
            daily_record.rows != result.daily_rows_sum
            or result.daily_rows_minimum <= 0
            or result.daily_rows_maximum < result.daily_rows_minimum
        ):
            raise IntegrityError("v5 daily row summary is invalid")
        verified.append(result)
    return tuple(verified)


def _totals(
    plan: V5GenerationPlan, results: tuple[V5WorkResult, ...]
) -> dict[str, object]:
    by_frequency: dict[str, dict[str, int]] = {
        frequency: {"files": 0, "rows": 0, "bytes": 0} for frequency in _FREQUENCIES
    }
    total_rows = 0
    total_bytes = 0
    for result in results:
        for record in result.files:
            item = by_frequency[record.frequency]
            item["files"] += 1
            item["rows"] += record.rows
            item["bytes"] += record.bytes
            total_rows += record.rows
            total_bytes += record.bytes
    daily_sum = sum(value.daily_rows_sum for value in results)
    daily_min = min(value.daily_rows_minimum for value in results)
    daily_max = max(value.daily_rows_maximum for value in results)
    expected = {
        "monthly": plan.path_count * plan.horizon_months,
        "quarterly": plan.path_count * plan.horizon_months // 3,
        "annual": plan.path_count * plan.horizon_months // 12,
    }
    if any(by_frequency[name]["rows"] != rows for name, rows in expected.items()):
        raise IntegrityError("v5 final coarse row totals differ from plan")
    if by_frequency["daily"]["rows"] != daily_sum:
        raise IntegrityError("v5 final daily row total differs from worker summaries")
    return {
        "paths": plan.path_count,
        "work_units": len(results),
        "csv_files": plan.csv_file_count,
        "rows": total_rows,
        "bytes": total_bytes,
        "by_frequency": by_frequency,
        "daily_rows_per_path": {
            "minimum": daily_min,
            "maximum": daily_max,
            "mean": daily_sum / plan.path_count,
        },
    }


def _write_new_file(path: Path, content: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            written = stream.write(content)
            if written != len(content):
                raise IntegrityError("v5 final publication write was incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise IntegrityError("v5 final publication already exists") from error
    except OSError as error:
        raise IntegrityError("v5 final publication failed") from error


def run_v5_generation(
    inputs: ResolvedV5Inputs,
    *,
    model_fingerprint: str,
    processed_data_fingerprint: str,
    output_root: str | Path,
    run_name: str,
    profile: V5GenerationProfile,
    workers: int,
    plan: V5GenerationPlan | None = None,
    path_generation_seed: int | None = None,
) -> V5GenerationResult:
    """Generate one registered v5 profile and publish final manifest/checksums."""

    if not isinstance(run_name, str) or _RUN_NAME.fullmatch(run_name) is None:
        raise GenerationError("v5 run name is invalid")
    worker_count = _positive_int(workers, "worker count")
    if worker_count > 9:
        raise GenerationError("v5 generation supports at most nine workers")
    selected_plan = (
        build_v5_generation_plan(inputs, profile=profile) if plan is None else plan
    )
    if selected_plan.profile != profile:
        raise GenerationError("v5 explicit plan profile differs from the run profile")
    _validate_plan_against_inputs(inputs, selected_plan)
    if profile == "canonical" and path_generation_seed is not None:
        raise GenerationError(
            "canonical v5 generation does not permit a path-seed override"
        )
    loader = build_v5_generation_loader(
        inputs,
        processed_data_fingerprint=processed_data_fingerprint,
        model_fingerprint=model_fingerprint,
        path_generation_seed=path_generation_seed,
    )
    generation_input = _load_from_loader(loader)
    simulation_identity = _simulation_identity(
        generation_input,
        selected_plan,
        path_generation_seed_overridden=path_generation_seed is not None,
    )
    simulation_fingerprint = _sha256_json(simulation_identity)
    materialization_identity = _materialization_identity(
        simulation_fingerprint, selected_plan
    )
    materialization_fingerprint = _sha256_json(materialization_identity)
    source = _source_provenance(inputs.config_path)
    canonical = profile == "canonical"
    if canonical:
        canonical_plan = build_v5_generation_plan(inputs, profile="canonical")
        if selected_plan.fingerprint != canonical_plan.fingerprint:
            raise GenerationError(
                "canonical v5 run requires the exact frozen full plan"
            )
        if cast(bool, source["dirty_worktree"]):
            raise GenerationError("canonical v5 production requires a clean worktree")
    toolchain = _toolchain_identity(source)
    toolchain_fingerprint = _sha256_json(toolchain)
    run_identity: dict[str, object] = {
        "schema": "prpg-v5-run-identity-v1",
        "run_name": run_name,
        "simulation_fingerprint": simulation_fingerprint,
        "materialization_fingerprint": materialization_fingerprint,
        "toolchain_fingerprint": toolchain_fingerprint,
    }
    if path_generation_seed is not None:
        run_identity["path_generation_seed"] = generation_input.master_seed
    run_fingerprint = _sha256_json(run_identity)
    root = Path(output_root).expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise GenerationError("v5 generation requires a fresh run root")
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        disk_preflight = _disk_preflight(
            root, generation_input, selected_plan, canonical=canonical
        )
        root.mkdir(mode=0o700)
    except OSError as error:
        raise GenerationError("v5 run root could not be created") from error
    tasks = tuple(
        _WorkerTask(loader, str(root), unit, selected_plan.horizon_months)
        for unit in selected_plan.work_units
    )
    started = time.monotonic()
    results = deterministic_process_map(
        _run_worker, tasks, workers=worker_count, chunksize=1
    )
    verified = _verify_results(root, selected_plan, results)
    totals = _totals(selected_plan, verified)
    checksum_content = "".join(
        f"{record.sha256}  {record.relative_path}\n"
        for record in sorted(
            (item for result in verified for item in result.files),
            key=lambda value: value.relative_path,
        )
    ).encode("utf-8")
    checksum_sha256 = hashlib.sha256(checksum_content).hexdigest()
    manifest_identity = {
        "schema_id": V5_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "run_name": run_name,
        "source": source,
        "config_fingerprint": inputs.config_fingerprint,
        "seed_registry_fingerprint": inputs.seed_registry_fingerprint,
        "processed_data_fingerprint": processed_data_fingerprint,
        "model_fingerprint": model_fingerprint,
        "approved_model_fingerprint": model_fingerprint,
        "simulation_fingerprint": simulation_fingerprint,
        "materialization_fingerprint": materialization_fingerprint,
        "toolchain_fingerprint": toolchain_fingerprint,
        "run_fingerprint": run_fingerprint,
        "generator_version": V5_GENERATOR_VERSION,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "csv_schema_version": V5_CSV_SCHEMA_VERSION,
        "csv_serializer_version": V5_CSV_SERIALIZER_VERSION,
        "return_representation": "log_total_decimal",
        "assets": _consumer_assets(inputs),
        "plan": selected_plan.as_dict(),
        "disk_preflight": disk_preflight,
        "workers": worker_count,
        "work_units": [value.as_dict() for value in verified],
        "totals": totals,
        "checksums": {
            "relative_path": "checksums.sha256",
            "entries": selected_plan.csv_file_count,
            "sha256": checksum_sha256,
        },
    }
    if path_generation_seed is not None:
        manifest_identity["path_generation_seed"] = generation_input.master_seed
    manifest_fingerprint = _sha256_json(manifest_identity)
    manifest = {
        **manifest_identity,
        "manifest_fingerprint": manifest_fingerprint,
    }
    checksum_path = root / "checksums.sha256"
    manifest_path = root / "manifest.json"
    _write_new_file(checksum_path, checksum_content)
    _write_new_file(manifest_path, _canonical_bytes(manifest))
    elapsed = time.monotonic() - started
    return V5GenerationResult(
        run_root=root,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        manifest_fingerprint=manifest_fingerprint,
        simulation_fingerprint=simulation_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
        profile=profile,
        workers=worker_count,
        path_count=selected_plan.path_count,
        work_units=len(selected_plan.work_units),
        csv_files=selected_plan.csv_file_count,
        total_rows=cast(int, totals["rows"]),
        total_bytes=cast(int, totals["bytes"]),
        wall_seconds=elapsed,
    )
