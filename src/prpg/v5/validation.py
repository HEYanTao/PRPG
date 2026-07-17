"""Lean full-row validation for delivery-version-5 consumer returns.

The validator has one responsibility: prove that the published CSV bundle is
the deterministic, structurally valid materialization of the frozen v5 model.
It deliberately does not reuse the legacy LF/HF G5/G7 qualification stack.
Distribution, tail, drawdown, ACF, source-use, and trading-strategy diagnostics
remain report-first work after the hard production path is operational.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from prpg.errors import IntegrityError
from prpg.model.artifact import ModelArtifactStore
from prpg.simulation.rng import Domain, Family, RNGKey, Stage, uniforms
from prpg.v5.config import ResolvedV5Inputs
from prpg.v5.data import load_v5_month_library

VALIDATION_SCHEMA: Final = "prpg-v5-validation-report-v1"
GENERATION_MANIFEST_SCHEMA: Final = "prpg-v5-generation-manifest-v1"
MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
MAX_CSV_ROW_BYTES: Final = 1024
ASSET_COUNT: Final = 3
STATE_COUNT: Final = 4
FREQUENCIES: Final = ("daily", "monthly", "quarterly", "annual")
ROLES: Final = ("training", "validation", "test")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

DAILY_HEADER: Final = (
    b"path_id,period_index,simulation_year,model_month,session_in_month,"
    b"equity_log_return,muni_bond_log_return,taxable_bond_log_return\n"
)
COARSE_HEADER: Final = (
    b"path_id,period_index,simulation_year,period_in_year,"
    b"equity_log_return,muni_bond_log_return,taxable_bond_log_return\n"
)

Frequency: TypeAlias = Literal["daily", "monthly", "quarterly", "annual"]
Role: TypeAlias = Literal["training", "validation", "test"]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


class _Selection(Protocol):
    @property
    def states(self) -> npt.ArrayLike: ...

    @property
    def source_month_indices(self) -> npt.ArrayLike: ...


class V5ReplayPath(Protocol):
    """Minimal generation result consumed by independent validation."""

    @property
    def selection(self) -> _Selection: ...

    @property
    def month_lengths(self) -> npt.ArrayLike: ...

    @property
    def daily_log_returns(self) -> npt.ArrayLike: ...

    @property
    def monthly_log_returns(self) -> npt.ArrayLike: ...

    @property
    def quarterly_log_returns(self) -> npt.ArrayLike: ...

    @property
    def annual_log_returns(self) -> npt.ArrayLike: ...


ReplayPath = Callable[[int], V5ReplayPath]


@dataclass(frozen=True, slots=True)
class _ValidationReference:
    historical_daily_consumer_order: FloatArray
    transition_matrix: FloatArray
    stationary_distribution: FloatArray
    state_pools: tuple[IntArray, ...]


@dataclass(frozen=True, slots=True)
class V5ValidationReport:
    """Compact pass report returned only after every hard check succeeds."""

    schema_id: str
    status: Literal["passed"]
    profile: str
    paths: int
    horizon_months: int
    work_units: int
    csv_files: int
    rows: int
    bytes: int
    checksums_verified: int
    maximum_source_replay_error: float
    maximum_log_compounding_error: float
    complete_path_duplicates: int
    historical_covariance: tuple[tuple[float, ...], ...]
    simulated_covariance: tuple[tuple[float, ...], ...]
    volatility_ratios: tuple[float, float, float]
    maximum_correlation_error: float
    relative_covariance_error: float
    occupancy: tuple[float, float, float, float]
    occupancy_limits: tuple[float, float, float, float]
    transition_probabilities: tuple[tuple[float, ...], ...]
    transition_limits: tuple[tuple[float, ...], ...]
    elapsed_seconds: float
    manifest_sha256: str
    report_only_deferred: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result without location-dependent paths."""

        return {
            "schema_id": self.schema_id,
            "status": self.status,
            "profile": self.profile,
            "paths": self.paths,
            "horizon_months": self.horizon_months,
            "work_units": self.work_units,
            "csv_files": self.csv_files,
            "rows": self.rows,
            "bytes": self.bytes,
            "checksums_verified": self.checksums_verified,
            "maximum_source_replay_error": self.maximum_source_replay_error,
            "maximum_log_compounding_error": self.maximum_log_compounding_error,
            "complete_path_duplicates": self.complete_path_duplicates,
            "historical_covariance": [list(row) for row in self.historical_covariance],
            "simulated_covariance": [list(row) for row in self.simulated_covariance],
            "volatility_ratios": list(self.volatility_ratios),
            "maximum_correlation_error": self.maximum_correlation_error,
            "relative_covariance_error": self.relative_covariance_error,
            "occupancy": list(self.occupancy),
            "occupancy_limits": list(self.occupancy_limits),
            "transition_probabilities": [
                list(row) for row in self.transition_probabilities
            ],
            "transition_limits": [list(row) for row in self.transition_limits],
            "elapsed_seconds": self.elapsed_seconds,
            "manifest_sha256": self.manifest_sha256,
            "report_only_deferred": list(self.report_only_deferred),
        }


@dataclass(frozen=True, slots=True)
class _FileRecord:
    frequency: Frequency
    relative_path: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _WorkUnit:
    work_unit_id: str
    role: Role
    first_path: int
    last_path: int
    files: tuple[_FileRecord, ...]


@dataclass(frozen=True, slots=True)
class _Manifest:
    profile: str
    paths: int
    horizon_months: int
    paths_per_shard: int
    data_fingerprint: str
    model_fingerprint: str
    path_generation_seed: int
    work_units: tuple[_WorkUnit, ...]
    total_rows: int
    total_bytes: int
    checksum_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _RiskResult:
    historical_covariance: FloatArray
    volatility_ratios: FloatArray
    correlation_error: float
    covariance_error: float


@dataclass(frozen=True, slots=True)
class _StateResult:
    occupancy: FloatArray
    occupancy_limits: FloatArray
    transition: FloatArray
    transition_limits: FloatArray


def validate_v5_run(
    inputs: ResolvedV5Inputs,
    run_root: str | Path,
    *,
    expected_paths: int | None = None,
    _replay_path: ReplayPath | None = None,
    _reference: _ValidationReference | None = None,
) -> V5ValidationReport:
    """Stream every v5 CSV and apply only the frozen Section-30 hard gates.

    ``expected_paths`` supports the named smoke/parity/pilot profiles.  The two
    underscore-prefixed injections exist solely for small tests; ordinary and
    CLI callers always use verified data/model artifacts plus the production
    generator's deterministic replay API.
    """

    if not isinstance(inputs, ResolvedV5Inputs):
        raise IntegrityError("version-5 validation inputs have the wrong type")
    root = _safe_run_root(run_root)
    started = time.monotonic()
    manifest = _load_manifest(inputs, root, expected_paths)
    reference = _reference or _load_reference(
        inputs, manifest.data_fingerprint, manifest.model_fingerprint
    )
    replay = _replay_path or _default_replay(
        inputs,
        manifest.data_fingerprint,
        manifest.model_fingerprint,
        manifest.horizon_months,
        manifest.path_generation_seed,
    )
    _validate_reference(reference)
    records = tuple(record for unit in manifest.work_units for record in unit.files)
    _validate_checksum_list(root, records, manifest.checksum_sha256)
    _validate_consumer_allowlist(root, records)

    tolerance = inputs.config.validation.csv_log_sum_absolute_tolerance
    covariance_sum = np.zeros((ASSET_COUNT, ASSET_COUNT), dtype=np.float64)
    occupancy_rows = np.empty((manifest.paths, STATE_COUNT), dtype=np.float64)
    transition_counts = np.zeros((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    path_digests: set[bytes] = set()
    maximum_replay_error = 0.0
    maximum_compounding_error = 0.0
    total_rows = 0
    total_bytes = 0

    observed_path = 0
    for unit in manifest.work_units:
        readers = {record.frequency: _CSVReader(root, record) for record in unit.files}
        try:
            for path_entity in range(unit.first_path, unit.last_path + 1):
                observed_path += 1
                materialized = replay(path_entity)
                expected = _validated_replay(
                    materialized,
                    path_entity=path_entity,
                    horizon_months=manifest.horizon_months,
                )
                _validate_source_selection(
                    reference,
                    scientific_version=inputs.seed_registry.scientific_version,
                    path_generation_seed=manifest.path_generation_seed,
                    path_entity=path_entity,
                    states=expected.states,
                    observed_sources=expected.sources,
                )
                daily = readers["daily"].read_path(path_entity, expected.month_lengths)
                monthly = readers["monthly"].read_path(
                    path_entity, manifest.horizon_months
                )
                quarterly = readers["quarterly"].read_path(
                    path_entity, manifest.horizon_months // 3
                )
                annual = readers["annual"].read_path(
                    path_entity, manifest.horizon_months // 12
                )
                maximum_replay_error = max(
                    maximum_replay_error,
                    _maximum_error(daily, expected.daily),
                    _maximum_error(monthly, expected.monthly),
                    _maximum_error(quarterly, expected.quarterly),
                    _maximum_error(annual, expected.annual),
                )
                if maximum_replay_error > tolerance:
                    raise IntegrityError(
                        "version-5 consumer values differ from deterministic replay"
                    )
                compounding_error = _compounding_error(
                    daily,
                    expected.month_lengths,
                    monthly,
                    quarterly,
                    annual,
                )
                maximum_compounding_error = max(
                    maximum_compounding_error, compounding_error
                )
                if compounding_error > tolerance:
                    raise IntegrityError(
                        "version-5 cross-frequency log compounding failed"
                    )
                covariance = np.asarray(
                    np.cov(daily, rowvar=False, ddof=1), dtype=np.float64
                )
                if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
                    raise IntegrityError("version-5 path covariance is invalid")
                covariance_sum += covariance
                states = expected.states
                occupancy_rows[observed_path - 1] = np.bincount(
                    states, minlength=STATE_COUNT
                ) / len(states)
                np.add.at(transition_counts, (states[:-1], states[1:]), 1)
                digest = _path_digest(expected.month_lengths, daily)
                if digest in path_digests:
                    raise IntegrityError(
                        "version-5 run contains a duplicate complete daily path"
                    )
                path_digests.add(digest)
            for reader in readers.values():
                reader.finish()
                total_rows += reader.rows
                total_bytes += reader.bytes
        finally:
            for reader in readers.values():
                reader.close()
    if observed_path != manifest.paths:
        raise IntegrityError("version-5 validated path count differs from its plan")
    _validate_consumer_allowlist(root, records)
    if total_rows != manifest.total_rows or total_bytes != manifest.total_bytes:
        raise IntegrityError("version-5 observed totals differ from its manifest")

    simulated_covariance = covariance_sum / manifest.paths
    enforce_science = manifest.profile in ("pilot", "canonical")
    risk = _risk_gate(
        inputs,
        reference.historical_daily_consumer_order,
        simulated_covariance,
        enforce=enforce_science,
    )
    state = _state_gate(
        inputs,
        occupancy_rows,
        transition_counts,
        reference.stationary_distribution,
        reference.transition_matrix,
        enforce=enforce_science,
    )
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise IntegrityError("version-5 validation elapsed time is invalid")
    return V5ValidationReport(
        schema_id=VALIDATION_SCHEMA,
        status="passed",
        profile=manifest.profile,
        paths=manifest.paths,
        horizon_months=manifest.horizon_months,
        work_units=len(manifest.work_units),
        csv_files=len(records),
        rows=total_rows,
        bytes=total_bytes,
        checksums_verified=len(records),
        maximum_source_replay_error=maximum_replay_error,
        maximum_log_compounding_error=maximum_compounding_error,
        complete_path_duplicates=0,
        historical_covariance=_matrix_tuple(risk.historical_covariance),
        simulated_covariance=_matrix_tuple(simulated_covariance),
        volatility_ratios=_vector3(risk.volatility_ratios),
        maximum_correlation_error=risk.correlation_error,
        relative_covariance_error=risk.covariance_error,
        occupancy=_vector4(state.occupancy),
        occupancy_limits=_vector4(state.occupancy_limits),
        transition_probabilities=_matrix_tuple(state.transition),
        transition_limits=_matrix_tuple(state.transition_limits),
        elapsed_seconds=elapsed,
        manifest_sha256=manifest.manifest_sha256,
        report_only_deferred=(
            "tails_drawdowns_acf",
            "source_use_and_repeated_subsequences",
            "quarterly_price_only_strategy_diagnostics",
        ),
    )


@dataclass(frozen=True, slots=True)
class _Replay:
    states: IntArray
    sources: IntArray
    month_lengths: IntArray
    daily: FloatArray
    monthly: FloatArray
    quarterly: FloatArray
    annual: FloatArray


def _validated_replay(
    value: V5ReplayPath, *, path_entity: int, horizon_months: int
) -> _Replay:
    try:
        states = np.asarray(value.selection.states, dtype=np.int64)
        sources = np.asarray(value.selection.source_month_indices, dtype=np.int64)
        lengths = np.asarray(value.month_lengths, dtype=np.int64)
        daily = np.asarray(value.daily_log_returns, dtype=np.float64)
        monthly = np.asarray(value.monthly_log_returns, dtype=np.float64)
        quarterly = np.asarray(value.quarterly_log_returns, dtype=np.float64)
        annual = np.asarray(value.annual_log_returns, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise IntegrityError("version-5 deterministic replay is invalid") from error
    if (
        states.shape != (horizon_months,)
        or sources.shape != (horizon_months,)
        or lengths.shape != (horizon_months,)
        or bool((states < 0).any())
        or bool((states >= STATE_COUNT).any())
        or bool((sources < 0).any())
        or bool((lengths <= 0).any())
        or daily.shape != (int(lengths.sum()), ASSET_COUNT)
        or monthly.shape != (horizon_months, ASSET_COUNT)
        or quarterly.shape != (horizon_months // 3, ASSET_COUNT)
        or annual.shape != (horizon_months // 12, ASSET_COUNT)
        or any(
            not np.isfinite(array).all()
            for array in (daily, monthly, quarterly, annual)
        )
    ):
        raise IntegrityError(
            "version-5 deterministic replay has invalid geometry",
            details={"path_entity": path_entity},
        )
    return _Replay(states, sources, lengths, daily, monthly, quarterly, annual)


def _default_replay(
    inputs: ResolvedV5Inputs,
    data_fingerprint: str,
    model_fingerprint: str,
    horizon_months: int,
    path_generation_seed: int,
) -> ReplayPath:
    from prpg.v5.generation import (  # imported lazily to avoid a module cycle
        load_v5_generation_input,
        materialize_v5_path,
    )

    generation_input = load_v5_generation_input(
        inputs,
        data_fingerprint,
        model_fingerprint,
        path_generation_seed=path_generation_seed,
    )

    def replay(path_entity: int) -> V5ReplayPath:
        return materialize_v5_path(
            generation_input, path_entity, horizon_months=horizon_months
        )

    return replay


def _validate_source_selection(
    reference: _ValidationReference,
    *,
    scientific_version: int,
    path_generation_seed: int,
    path_entity: int,
    states: IntArray,
    observed_sources: IntArray,
) -> None:
    """Check source ranks independently from the production selector."""

    source_key = RNGKey(
        master_seed=path_generation_seed,
        scientific_version=scientific_version,
        domain=Domain.PRODUCTION,
        family=Family.NOT_APPLICABLE,
        entity=path_entity,
        stage=Stage.SOURCE_RANK_UNIFORMS,
    )
    draws = uniforms(source_key, len(states))
    expected_sources = np.empty_like(observed_sources)
    for month, (state, draw) in enumerate(zip(states, draws, strict=True)):
        pool = reference.state_pools[int(state)]
        rank = min(math.floor(float(draw) * len(pool)), len(pool) - 1)
        expected_sources[month] = pool[rank]
    if not np.array_equal(observed_sources, expected_sources):
        raise IntegrityError(
            "version-5 source-month selection violates the conditional uniform "
            "RNG contract",
            details={"path_entity": path_entity},
        )


def _load_reference(
    inputs: ResolvedV5Inputs, data_fingerprint: str, model_fingerprint: str
) -> _ValidationReference:
    library = load_v5_month_library(inputs, data_fingerprint)
    arrays = ModelArtifactStore(inputs.artifact_root).load_arrays(model_fingerprint)
    try:
        decoded = np.asarray(arrays["decoded_states"], dtype=np.int64)
        transition = np.asarray(arrays["transition_matrix"], dtype=np.float64)
        stationary = np.asarray(arrays["stationary_distribution"], dtype=np.float64)
    except KeyError as error:
        raise IntegrityError("version-5 model lacks validation arrays") from error
    # Stored feature order is equity, taxable, muni; consumer order is equity,
    # muni, taxable.  Make this semantic reorder explicit at the boundary.
    historical = np.asarray(library.daily_returns[:, (0, 2, 1)], dtype=np.float64)
    pools = tuple(
        np.asarray(np.flatnonzero(decoded == state), dtype=np.int64)
        for state in range(STATE_COUNT)
    )
    return _ValidationReference(historical, transition, stationary, pools)


def _validate_reference(value: _ValidationReference) -> None:
    if (
        not isinstance(value, _ValidationReference)
        or value.historical_daily_consumer_order.ndim != 2
        or value.historical_daily_consumer_order.shape[1] != ASSET_COUNT
        or value.historical_daily_consumer_order.shape[0] < 2
        or value.transition_matrix.shape != (STATE_COUNT, STATE_COUNT)
        or value.stationary_distribution.shape != (STATE_COUNT,)
        or len(value.state_pools) != STATE_COUNT
        or any(
            pool.ndim != 1 or len(pool) == 0 or bool((pool < 0).any())
            for pool in value.state_pools
        )
        or any(
            not np.isfinite(array).all()
            for array in (
                value.historical_daily_consumer_order,
                value.transition_matrix,
                value.stationary_distribution,
            )
        )
        or bool((value.transition_matrix < 0.0).any())
        or bool((value.stationary_distribution < 0.0).any())
        or not np.allclose(value.transition_matrix.sum(axis=1), 1.0, atol=1e-12)
        or not np.isclose(value.stationary_distribution.sum(), 1.0, atol=1e-12)
    ):
        raise IntegrityError("version-5 validation reference is invalid")


def _risk_gate(
    inputs: ResolvedV5Inputs,
    historical: FloatArray,
    simulated: FloatArray,
    *,
    enforce: bool,
) -> _RiskResult:
    historical_covariance = np.asarray(
        np.cov(historical, rowvar=False, ddof=1), dtype=np.float64
    )
    historical_diagonal = np.diag(historical_covariance)
    simulated_diagonal = np.diag(simulated)
    if bool((historical_diagonal <= 0.0).any()) or bool(
        (simulated_diagonal < 0.0).any()
    ):
        raise IntegrityError("version-5 covariance diagonal is invalid")
    volatility_ratios = np.sqrt(simulated_diagonal / historical_diagonal)
    historical_correlation = _correlation(historical_covariance)
    simulated_correlation = _correlation(simulated)
    off_diagonal = np.triu_indices(ASSET_COUNT, k=1)
    correlation_error = float(
        np.max(np.abs(simulated_correlation - historical_correlation)[off_diagonal])
    )
    denominator = float(np.linalg.norm(historical_covariance, ord="fro"))
    covariance_error = float(
        np.linalg.norm(simulated - historical_covariance, ord="fro") / denominator
    )
    config = inputs.config.validation
    if enforce and (
        bool((volatility_ratios < config.volatility_ratio_minimum).any())
        or bool((volatility_ratios > config.volatility_ratio_maximum).any())
        or correlation_error >= config.correlation_error_maximum_exclusive
        or covariance_error > config.covariance_relative_error_maximum
    ):
        raise IntegrityError(
            "version-5 core daily covariance plausibility gate failed",
            details={
                "volatility_ratios": volatility_ratios.tolist(),
                "maximum_correlation_error": correlation_error,
                "relative_covariance_error": covariance_error,
            },
        )
    return _RiskResult(
        historical_covariance,
        volatility_ratios,
        correlation_error,
        covariance_error,
    )


def _state_gate(
    inputs: ResolvedV5Inputs,
    occupancy_rows: FloatArray,
    counts: IntArray,
    stationary: FloatArray,
    transition: FloatArray,
    *,
    enforce: bool,
) -> _StateResult:
    paths = occupancy_rows.shape[0]
    occupancy = np.asarray(
        np.mean(occupancy_rows, axis=0, dtype=np.float64), dtype=np.float64
    )
    occupancy_se = (
        np.zeros(STATE_COUNT, dtype=np.float64)
        if paths == 1
        else np.std(occupancy_rows, axis=0, ddof=1) / math.sqrt(paths)
    )
    occupancy_limits = np.maximum(
        5.0 * occupancy_se, inputs.config.validation.occupancy_absolute_floor
    )
    if enforce and bool((np.abs(occupancy - stationary) > occupancy_limits).any()):
        raise IntegrityError("version-5 state occupancy gate failed")
    departures = counts.sum(axis=1)
    if enforce and bool((departures <= 0).any()):
        raise IntegrityError("version-5 state transition support is empty")
    observed = transition.copy()
    np.divide(
        counts,
        departures[:, np.newaxis],
        out=observed,
        where=departures[:, np.newaxis] > 0,
    )
    transition_se = np.sqrt(
        np.divide(
            transition * (1.0 - transition),
            departures[:, np.newaxis],
            out=np.zeros_like(transition),
            where=departures[:, np.newaxis] > 0,
        )
    )
    transition_limits = np.maximum(
        5.0 * transition_se,
        inputs.config.validation.transition_cell_absolute_floor,
    )
    if enforce and bool((np.abs(observed - transition) > transition_limits).any()):
        raise IntegrityError("version-5 state transition gate failed")
    return _StateResult(occupancy, occupancy_limits, observed, transition_limits)


def _compounding_error(
    daily: FloatArray,
    month_lengths: IntArray,
    monthly: FloatArray,
    quarterly: FloatArray,
    annual: FloatArray,
) -> float:
    offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(month_lengths, dtype=np.int64))
    )
    expected_monthly = np.vstack(
        [
            np.sum(daily[int(start) : int(stop)], axis=0, dtype=np.float64)
            for start, stop in pairwise(offsets)
        ]
    )
    expected_quarterly = monthly.reshape(-1, 3, ASSET_COUNT).sum(axis=1)
    expected_annual = monthly.reshape(-1, 12, ASSET_COUNT).sum(axis=1)
    return max(
        _maximum_error(expected_monthly, monthly),
        _maximum_error(expected_quarterly, quarterly),
        _maximum_error(expected_annual, annual),
    )


class _CSVReader:
    def __init__(self, root: Path, record: _FileRecord) -> None:
        self.record = record
        self.path = root.joinpath(*PurePosixPath(record.relative_path).parts)
        descriptor, initial = _open_regular(self.path, "version-5 CSV")
        self._initial = initial
        self._handle = os.fdopen(descriptor, "rb", closefd=True)
        self._digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0
        self._finished = False
        header = self._handle.readline(MAX_CSV_ROW_BYTES + 1)
        self._digest.update(header)
        self.bytes += len(header)
        expected = DAILY_HEADER if record.frequency == "daily" else COARSE_HEADER
        if header != expected:
            self._handle.close()
            raise IntegrityError("version-5 consumer CSV header is invalid")

    def read_path(self, path_entity: int, geometry: int | IntArray) -> FloatArray:
        if isinstance(geometry, np.ndarray):
            rows = int(geometry.sum())
            month_by_row = np.repeat(np.arange(len(geometry)), geometry)
            session_by_row = np.concatenate(
                [np.arange(1, int(length) + 1) for length in geometry]
            )
        else:
            rows = geometry
            month_by_row = np.empty(0, dtype=np.int64)
            session_by_row = np.empty(0, dtype=np.int64)
        values = np.empty((rows, ASSET_COUNT), dtype=np.float64)
        path_id = f"P{path_entity:06d}"
        periods_per_year = {"monthly": 12, "quarterly": 4, "annual": 1}
        for index in range(rows):
            line = self._handle.readline(MAX_CSV_ROW_BYTES + 1)
            if not line:
                raise IntegrityError("version-5 consumer CSV is truncated")
            if (
                len(line) > MAX_CSV_ROW_BYTES
                or not line.endswith(b"\n")
                or b"\r" in line
            ):
                raise IntegrityError("version-5 consumer CSV row encoding is invalid")
            self._digest.update(line)
            self.bytes += len(line)
            try:
                fields = line[:-1].decode("utf-8").split(",")
            except UnicodeDecodeError as error:
                raise IntegrityError("version-5 consumer CSV is not UTF-8") from error
            expected_keys: tuple[str, ...]
            if self.record.frequency == "daily":
                month = int(month_by_row[index])
                expected_keys = (
                    path_id,
                    str(index + 1),
                    str(month // 12 + 1),
                    str(month % 12 + 1),
                    str(int(session_by_row[index])),
                )
                return_fields = fields[5:]
                valid_length = len(fields) == 8
            else:
                per_year = periods_per_year[self.record.frequency]
                expected_keys = (
                    path_id,
                    str(index + 1),
                    str(index // per_year + 1),
                    str(index % per_year + 1),
                )
                return_fields = fields[4:]
                valid_length = len(fields) == 7
            if not valid_length or tuple(fields[: len(expected_keys)]) != expected_keys:
                raise IntegrityError(
                    "version-5 consumer path or period keys are noncanonical"
                )
            try:
                values[index] = tuple(float(item) for item in return_fields)
            except (TypeError, ValueError) as error:
                raise IntegrityError(
                    "version-5 consumer return is not numeric"
                ) from error
            if not np.isfinite(values[index]).all():
                raise IntegrityError("version-5 consumer return is not finite")
            self.rows += 1
        return values

    def finish(self) -> None:
        if self._finished:
            raise IntegrityError("version-5 CSV reader finalized twice")
        self._finished = True
        if self._handle.read(1):
            raise IntegrityError("version-5 consumer CSV contains excess rows")
        final = os.fstat(self._handle.fileno())
        current = self.path.stat(follow_symlinks=False)
        if any(
            (
                final.st_dev != self._initial.st_dev,
                final.st_ino != self._initial.st_ino,
                final.st_size != self._initial.st_size,
                final.st_mtime_ns != self._initial.st_mtime_ns,
                current.st_dev != self._initial.st_dev,
                current.st_ino != self._initial.st_ino,
                current.st_size != self._initial.st_size,
                current.st_mtime_ns != self._initial.st_mtime_ns,
            )
        ):
            raise IntegrityError("version-5 consumer CSV changed during validation")
        if (
            self.rows != self.record.rows
            or self.bytes != self.record.bytes
            or self._digest.hexdigest() != self.record.sha256
        ):
            raise IntegrityError("version-5 CSV row, byte, or checksum result differs")

    def close(self) -> None:
        self._handle.close()


def _load_manifest(
    inputs: ResolvedV5Inputs, root: Path, expected_paths: int | None
) -> _Manifest:
    content = _read_small(root / "manifest.json", MAX_MANIFEST_BYTES, "manifest")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("version-5 manifest is not valid JSON") from error
    raw = _mapping(value, "manifest")
    if (
        raw.get("schema_id") != GENERATION_MANIFEST_SCHEMA
        or raw.get("status") != "complete"
    ):
        raise IntegrityError("version-5 generation manifest is not complete")
    manifest_fingerprint = _fingerprint(raw.get("manifest_fingerprint"), "manifest")
    identity = dict(raw)
    identity.pop("manifest_fingerprint", None)
    if manifest_fingerprint != _json_fingerprint(identity):
        raise IntegrityError("version-5 manifest fingerprint differs")
    if (
        raw.get("config_fingerprint") != inputs.config_fingerprint
        or raw.get("seed_registry_fingerprint") != inputs.seed_registry_fingerprint
    ):
        raise IntegrityError("version-5 manifest input fingerprints differ")
    data_fingerprint = _fingerprint(raw.get("processed_data_fingerprint"), "data")
    model_fingerprint = _fingerprint(raw.get("model_fingerprint"), "model")
    if raw.get("approved_model_fingerprint") != model_fingerprint:
        raise IntegrityError("version-5 approved model binding differs")
    path_generation_seed = _uint128(
        raw.get("path_generation_seed", inputs.seed_registry.master_seed),
        "path-generation seed",
    )
    plan = _mapping(raw.get("plan"), "plan")
    profile = plan.get("profile")
    if not isinstance(profile, str) or not profile:
        raise IntegrityError("version-5 profile is invalid")
    paths = _positive_int(plan.get("path_count"), "path count")
    expected = (
        inputs.config.simulation.paths if expected_paths is None else expected_paths
    )
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or paths != expected
    ):
        raise IntegrityError(
            "version-5 manifest path count differs from requested profile"
        )
    horizon = _positive_int(plan.get("horizon_months"), "horizon")
    if horizon != inputs.config.simulation.horizon_months:
        raise IntegrityError("version-5 manifest horizon differs from configuration")
    paths_per_shard = _positive_int(plan.get("paths_per_shard"), "paths per shard")
    ranges = _path_ranges(inputs, plan.get("path_ranges"), paths)
    units = _work_units(raw.get("work_units"), ranges, paths, horizon)
    if plan.get("work_unit_count") != len(units) or plan.get(
        "csv_file_count"
    ) != 4 * len(units):
        raise IntegrityError("version-5 manifest plan totals differ")
    totals = _mapping(raw.get("totals"), "totals")
    total_rows = sum(record.rows for unit in units for record in unit.files)
    total_bytes = sum(record.bytes for unit in units for record in unit.files)
    if (
        totals.get("paths") != paths
        or totals.get("work_units") != len(units)
        or totals.get("csv_files") != 4 * len(units)
        or totals.get("rows") != total_rows
        or totals.get("bytes") != total_bytes
    ):
        raise IntegrityError("version-5 manifest observed totals differ")
    _validate_frequency_totals(totals, units)
    checksums = _mapping(raw.get("checksums"), "checksums")
    if checksums.get("relative_path") != "checksums.sha256" or checksums.get(
        "entries"
    ) != 4 * len(units):
        raise IntegrityError("version-5 manifest checksum summary is invalid")
    checksum_sha256 = _fingerprint(checksums.get("sha256"), "checksum-list")
    return _Manifest(
        profile,
        paths,
        horizon,
        paths_per_shard,
        data_fingerprint,
        model_fingerprint,
        path_generation_seed,
        units,
        total_rows,
        total_bytes,
        checksum_sha256,
        hashlib.sha256(content).hexdigest(),
    )


def _path_ranges(
    inputs: ResolvedV5Inputs, value: object, paths: int
) -> tuple[tuple[Role, int, int], ...]:
    raw = _sequence(value, "path ranges")
    result: list[tuple[Role, int, int]] = []
    total = 0
    previous_role = -1
    occupied: set[int] = set()
    split = inputs.config.simulation.split
    bounds = {
        "training": (1, split.training),
        "validation": (split.training + 1, split.training + split.validation),
        "test": (
            split.training + split.validation + 1,
            inputs.config.simulation.paths,
        ),
    }
    for item_value in raw:
        item = _mapping(item_value, "path range")
        role = item.get("role")
        if role not in ROLES or any(existing[0] == role for existing in result):
            raise IntegrityError("version-5 path role is invalid or duplicated")
        first = _positive_int(item.get("first_path"), "first path")
        last = _positive_int(item.get("last_path"), "last path")
        role_index = ROLES.index(cast(Role, role))
        lower, upper = bounds[cast(Role, role)]
        values = set(range(first, last + 1))
        if (
            role_index <= previous_role
            or last < first
            or not lower <= first <= last <= upper
            or item.get("path_count") != last - first + 1
            or occupied.intersection(values)
        ):
            raise IntegrityError("version-5 path ranges are invalid")
        result.append((cast(Role, role), first, last))
        occupied.update(values)
        total += len(values)
        previous_role = role_index
    if total != paths:
        raise IntegrityError("version-5 path ranges do not cover the profile")
    return tuple(result)


def _work_units(
    value: object,
    ranges: tuple[tuple[Role, int, int], ...],
    paths: int,
    horizon: int,
) -> tuple[_WorkUnit, ...]:
    raw = _sequence(value, "work units")
    units: list[_WorkUnit] = []
    expected_entities = tuple(
        entity for _role, first, last in ranges for entity in range(first, last + 1)
    )
    observed_entities: list[int] = []
    for raw_item in raw:
        item = _mapping(raw_item, "work unit")
        first = _positive_int(item.get("first_path"), "work-unit first path")
        last = _positive_int(item.get("last_path"), "work-unit last path")
        role = item.get("role")
        if last < first or item.get("path_count") != last - first + 1:
            raise IntegrityError("version-5 work-unit path geometry is invalid")
        expected_role = next(
            (r for r, start, stop in ranges if start <= first <= last <= stop), None
        )
        if role != expected_role:
            raise IntegrityError("version-5 work unit crosses or mislabels a path role")
        identifier = item.get("work_unit_id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or any(unit.work_unit_id == identifier for unit in units)
        ):
            raise IntegrityError(
                "version-5 work-unit identity is invalid or duplicated"
            )
        files = _files(item.get("files"), cast(Role, role), first, last, horizon)
        units.append(_WorkUnit(identifier, cast(Role, role), first, last, files))
        observed_entities.extend(range(first, last + 1))
    if len(observed_entities) != paths or tuple(observed_entities) != expected_entities:
        raise IntegrityError("version-5 work units do not cover all paths")
    return tuple(units)


def _files(
    value: object,
    role: Role,
    first: int,
    last: int,
    horizon: int,
) -> tuple[_FileRecord, ...]:
    raw = _sequence(value, "work-unit files")
    if len(raw) != 4:
        raise IntegrityError("version-5 work unit must contain four CSV files")
    records: list[_FileRecord] = []
    for raw_record in raw:
        record = _mapping(raw_record, "file record")
        frequency = record.get("frequency")
        if frequency not in FREQUENCIES or any(
            item.frequency == frequency for item in records
        ):
            raise IntegrityError("version-5 CSV frequency is invalid or duplicated")
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise IntegrityError("version-5 CSV relative path is invalid")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".csv":
            raise IntegrityError("version-5 CSV relative path is unsafe")
        expected_prefix = PurePosixPath(
            "consumer", "returns", role, cast(str, frequency)
        )
        if pure.parent != expected_prefix:
            raise IntegrityError(
                "version-5 CSV is outside its role/frequency directory"
            )
        rows = _positive_int(record.get("rows"), "CSV rows")
        count = last - first + 1
        fixed = {"monthly": horizon, "quarterly": horizon // 3, "annual": horizon // 12}
        if frequency != "daily" and rows != count * fixed[cast(str, frequency)]:
            raise IntegrityError("version-5 coarse CSV row count is invalid")
        records.append(
            _FileRecord(
                cast(Frequency, frequency),
                relative,
                rows,
                _positive_int(record.get("bytes"), "CSV bytes"),
                _fingerprint(record.get("sha256"), "CSV"),
            )
        )
    order = {frequency: index for index, frequency in enumerate(FREQUENCIES)}
    return tuple(sorted(records, key=lambda item: order[item.frequency]))


def _validate_frequency_totals(
    totals: Mapping[str, object], units: Sequence[_WorkUnit]
) -> None:
    raw = _mapping(totals.get("by_frequency"), "frequency totals")
    if set(raw) != set(FREQUENCIES):
        raise IntegrityError("version-5 frequency-total keys are invalid")
    for frequency in FREQUENCIES:
        records = [
            record
            for unit in units
            for record in unit.files
            if record.frequency == frequency
        ]
        observed = _mapping(raw.get(frequency), f"{frequency} totals")
        expected = {
            "files": len(records),
            "rows": sum(record.rows for record in records),
            "bytes": sum(record.bytes for record in records),
        }
        if dict(observed) != expected:
            raise IntegrityError("version-5 frequency totals differ from CSV records")
    daily = _mapping(totals.get("daily_rows_per_path"), "daily path totals")
    if set(daily) != {"minimum", "maximum", "mean"}:
        raise IntegrityError("version-5 daily path summary keys are invalid")
    minimum = daily.get("minimum")
    maximum = daily.get("maximum")
    mean = daily.get("mean")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum <= 0
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < minimum
        or isinstance(mean, bool)
        or not isinstance(mean, int | float)
        or not math.isfinite(float(mean))
        or not minimum <= float(mean) <= maximum
    ):
        raise IntegrityError("version-5 daily path summary is invalid")


def _validate_checksum_list(
    root: Path, records: Sequence[_FileRecord], expected_sha256: str
) -> None:
    expected = "".join(
        f"{record.sha256}  {record.relative_path}\n"
        for record in sorted(records, key=lambda item: item.relative_path)
    ).encode("utf-8")
    observed = _read_small(
        root / "checksums.sha256", len(expected) + 1, "checksum list"
    )
    if observed != expected:
        raise IntegrityError("version-5 checksum list differs from its manifest")
    if hashlib.sha256(observed).hexdigest() != expected_sha256:
        raise IntegrityError("version-5 checksum-list hash differs from its manifest")


def _validate_consumer_allowlist(root: Path, records: Sequence[_FileRecord]) -> None:
    expected = {record.relative_path for record in records}
    returns_root = root / "consumer" / "returns"
    if returns_root.is_symlink() or not returns_root.is_dir():
        raise IntegrityError("version-5 consumer return tree is missing or unsafe")
    actual: set[str] = set()
    for child in returns_root.rglob("*"):
        metadata = child.lstat()
        if stat.S_ISDIR(metadata.st_mode) and not child.is_symlink():
            continue
        if stat.S_ISREG(metadata.st_mode) and not child.is_symlink():
            actual.add(child.relative_to(root).as_posix())
            continue
        raise IntegrityError("version-5 consumer tree contains an unsafe entry")
    if actual != expected:
        raise IntegrityError("version-5 consumer CSV allowlist differs")


def _safe_run_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("version-5 run root is missing or unsafe")
    return root


def _read_small(path: Path, maximum: int, label: str) -> bytes:
    descriptor, metadata = _open_regular(path, f"version-5 {label}")
    try:
        if metadata.st_size > maximum:
            raise IntegrityError(f"version-5 {label} exceeds its size bound")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            content = handle.read(maximum + 1)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
    if len(content) > maximum:
        raise IntegrityError(f"version-5 {label} exceeds its size bound")
    return content


def _open_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise IntegrityError(f"{label} cannot be opened") from error
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise IntegrityError(f"{label} is not a regular file")
    return descriptor, metadata


def _maximum_error(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape:
        raise IntegrityError("version-5 compared return geometries differ")
    return float(np.max(np.abs(left - right), initial=0.0))


def _path_digest(lengths: IntArray, daily: FloatArray) -> bytes:
    normalized = np.asarray(daily, dtype="<f8", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    digest = hashlib.sha256(np.asarray(lengths, dtype="<i8").tobytes(order="C"))
    digest.update(normalized.tobytes(order="C"))
    return digest.digest()


def _correlation(covariance: FloatArray) -> FloatArray:
    diagonal = np.diag(covariance)
    if bool((diagonal <= 0.0).any()):
        raise IntegrityError("version-5 correlation covariance is degenerate")
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    result = covariance / denominator
    if not np.isfinite(result).all():
        raise IntegrityError("version-5 correlation is non-finite")
    return np.asarray(result, dtype=np.float64)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"version-5 {label} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise IntegrityError(f"version-5 {label} must be a sequence")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntegrityError(f"version-5 {label} must be a positive integer")
    return value


def _uint128(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= (1 << 128) - 1
    ):
        raise IntegrityError(f"version-5 {label} must be an unsigned 128-bit integer")
    return value


def _fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IntegrityError(f"version-5 {label} fingerprint is invalid")
    return value


def _json_fingerprint(value: object) -> str:
    content = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _matrix_tuple(value: npt.ArrayLike) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(value, dtype=np.float64)
    return tuple(tuple(float(item) for item in row) for row in array)


def _vector3(value: npt.ArrayLike) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    return float(array[0]), float(array[1]), float(array[2])


def _vector4(value: npt.ArrayLike) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    return float(array[0]), float(array[1]), float(array[2]), float(array[3])
