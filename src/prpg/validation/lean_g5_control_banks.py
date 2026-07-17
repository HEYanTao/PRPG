"""Deterministic historical-control banks for the lean G5 alpha check.

This module does one bounded job: bootstrap synchronized historical return
rows, score each pseudo-history with the exact six price-only strategies, and
retain one compact six-alpha row per path.  It deliberately does not choose
thresholds, make a G5 decision, publish evidence, or authorize generation.

The three bank roles use disjoint keys in the registered ``G5_OUTER_NULL``
namespace.  A and B use the two registered artifact contexts at replicate
zero, while C uses the control context beginning at replicate 10,000.  LF and
HF remain separated by the registered RNG family.  Scheduling and worker
count never enter the scientific identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError
from prpg.execution import deterministic_process_map
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)
from prpg.validation.lean_g5_strategies import (
    LEAN_G5_STRATEGY_NAMES,
    Frequency,
    LeanG5StrategyModel,
    score_lean_g5_path,
)
from prpg.validation.outer_null import unconditioned_stationary_bootstrap_trace

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
BankRole: TypeAlias = Literal["A", "B", "C"]
NoncanonicalUse: TypeAlias = Literal["pilot", "test"]

LEAN_G5_CONTROL_BANK_SCHEMA_ID: Final = "prpg-g5-lean-control-bank-v1"
LEAN_G5_CONTROL_BANK_CONTRACT_ID: Final = "g5-lean-six-family-three-bank-v1"
LEAN_G5_CONTROL_BANK_ROLES: Final[tuple[BankRole, ...]] = ("A", "B", "C")
LEAN_G5_LF_PATHS_PER_BANK: Final = 500
LEAN_G5_LF_HISTORY_LENGTH: Final = 217
LEAN_G5_HF_PATHS_PER_BANK: Final = 200
LEAN_G5_HF_HISTORY_LENGTH: Final = 4_547

_ROLE_RNG_COORDINATES: Final = {
    "A": (CalibrationArtifact.DESIGN_OR_CONTROL, 0),
    "B": (CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0),
    "C": (CalibrationArtifact.DESIGN_OR_CONTROL, 10_000),
}
_MAXIMUM_WORKERS: Final = 9
_DEFAULT_BATCH_SIZE: Final = 16
_ASSET_COUNT: Final = 3
_STRATEGY_COUNT: Final = 6


@dataclass(frozen=True, slots=True)
class LeanG5ControlBankGeometry:
    """Exact path geometry for one frequency and one bank role."""

    frequency: Frequency
    path_count: int
    history_length: int


@dataclass(frozen=True, slots=True)
class LeanG5ControlRngIdentity:
    """One path's complete registered RNG identity."""

    bank_role: BankRole
    frequency: Frequency
    path_index: int
    master_seed: int
    scientific_version: int
    rng_contract: int
    family: int
    artifact: int
    replicate: int
    spawn_key: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class LeanG5HistoricalControlDraw:
    """One auditable pseudo-history; production bank building discards it."""

    bank_role: BankRole
    frequency: Frequency
    path_index: int
    history_length: int
    mean_block_length: float
    source_fingerprint: str
    rng_identity: LeanG5ControlRngIdentity
    source_indices: IntArray
    restart_flags: BoolArray
    log_returns: FloatArray
    fingerprint: str


@dataclass(frozen=True, slots=True)
class LeanG5ScoredControlBank:
    """Compact historical-control result with exactly six alpha columns."""

    schema_id: str
    contract_id: str
    bank_role: BankRole
    frequency: Frequency
    path_count: int
    history_length: int
    score_observations: int
    production_geometry: bool
    noncanonical_use: NoncanonicalUse | None
    mean_block_length: float
    source_observations: int
    source_fingerprint: str
    model_fingerprint: str
    master_seed: int
    scientific_version: int
    rng_contract: int
    rng_artifact: int
    rng_replicate_start: int
    rng_identity_fingerprint: str
    strategy_names: tuple[str, ...]
    alpha_rows: FloatArray
    alpha_rows_sha256: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _ScoredControlWork:
    source: FloatArray
    continuation_links: BoolArray
    source_fingerprint: str
    model: LeanG5StrategyModel
    bank_role: BankRole
    frequency: Frequency
    history_length: int
    mean_block_length: float
    master_seed: int
    scientific_version: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _ScoredAlphaChunk:
    start: int
    stop: int
    score_observations: int
    alpha_rows: FloatArray


def lean_g5_control_bank_geometry(
    frequency: Frequency,
) -> LeanG5ControlBankGeometry:
    """Return the immutable production geometry for one A/B/C bank."""

    checked = _frequency(frequency)
    if checked == "monthly":
        return LeanG5ControlBankGeometry(
            checked,
            LEAN_G5_LF_PATHS_PER_BANK,
            LEAN_G5_LF_HISTORY_LENGTH,
        )
    return LeanG5ControlBankGeometry(
        checked,
        LEAN_G5_HF_PATHS_PER_BANK,
        LEAN_G5_HF_HISTORY_LENGTH,
    )


def lean_g5_control_rng_identity(
    *,
    bank_role: BankRole,
    frequency: Frequency,
    path_index: int,
    master_seed: int,
    scientific_version: int,
) -> LeanG5ControlRngIdentity:
    """Resolve one path to a unique key in the closed RNG registry."""

    role = _bank_role(bank_role)
    checked_frequency = _frequency(frequency)
    geometry = lean_g5_control_bank_geometry(checked_frequency)
    index = _bounded_integer(path_index, "control-bank path index", minimum=0)
    if index >= geometry.path_count:
        raise ValidationError(
            "control-bank path index is outside the production bank geometry"
        )
    artifact, replicate_start = _ROLE_RNG_COORDINATES[role]
    family = Family.LF if checked_frequency == "monthly" else Family.HF
    try:
        key = calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.G5_OUTER_NULL,
            artifact=artifact,
            variant=0,
            replicate=replicate_start + index,
            domain=Domain.THRESHOLD_CALIBRATION,
            family=family,
            stage=Stage.REFERENCE_RESAMPLING,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("control-bank RNG identity is invalid") from error
    identity = {
        "bank_role": role,
        "frequency": checked_frequency,
        "path_index": index,
        "master_seed": key.master_seed,
        "spawn_key": list(key.spawn_key),
    }
    return LeanG5ControlRngIdentity(
        bank_role=role,
        frequency=checked_frequency,
        path_index=index,
        master_seed=key.master_seed,
        scientific_version=key.scientific_version,
        rng_contract=key.contract,
        family=int(key.family),
        artifact=int(artifact),
        replicate=replicate_start + index,
        spawn_key=key.spawn_key,
        fingerprint=_fingerprint(identity),
    )


def draw_lean_g5_historical_control_path(
    historical_log_returns: ArrayLike,
    source_continuation_links: ArrayLike,
    *,
    bank_role: BankRole,
    frequency: Frequency,
    path_index: int,
    history_length: int | None = None,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
    noncanonical_use: NoncanonicalUse | None = None,
) -> LeanG5HistoricalControlDraw:
    """Materialize one path for audit or focused tests.

    Production bank construction calls the same private draw and immediately
    discards its full history after scoring.  A reduced history is permitted
    only when explicitly labelled ``pilot`` or ``test``.
    """

    source, links, source_fingerprint = _historical_source(
        historical_log_returns, source_continuation_links
    )
    role = _bank_role(bank_role)
    checked_frequency = _frequency(frequency)
    geometry = lean_g5_control_bank_geometry(checked_frequency)
    length = geometry.history_length if history_length is None else history_length
    _geometry(
        checked_frequency,
        path_count=geometry.path_count,
        history_length=length,
        noncanonical_use=noncanonical_use,
        path_count_is_production=True,
    )
    return _draw_checked_control_path(
        source,
        links,
        source_fingerprint=source_fingerprint,
        bank_role=role,
        frequency=checked_frequency,
        path_index=path_index,
        history_length=length,
        mean_block_length=_mean_block_length(mean_block_length),
        master_seed=master_seed,
        scientific_version=scientific_version,
    )


def build_scored_lean_g5_control_bank(
    historical_log_returns: ArrayLike,
    source_continuation_links: ArrayLike,
    model: LeanG5StrategyModel,
    *,
    bank_role: BankRole,
    frequency: Frequency,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
    path_count: int | None = None,
    history_length: int | None = None,
    noncanonical_use: NoncanonicalUse | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> LeanG5ScoredControlBank:
    """Build and score one A/B/C bank in deterministic input order.

    Omitting ``path_count`` and ``history_length`` selects the exact public
    production defaults.  Overrides may only reduce geometry and require an
    explicit ``pilot`` or ``test`` label.  Worker count and batch size are
    execution details and do not alter the returned scientific fingerprint.
    """

    source, links, source_fingerprint = _historical_source(
        historical_log_returns, source_continuation_links
    )
    role = _bank_role(bank_role)
    checked_frequency = _frequency(frequency)
    if not isinstance(model, LeanG5StrategyModel):
        raise ValidationError("control-bank scoring requires a lean G5 model")
    if model.frequency != checked_frequency:
        raise ValidationError("control-bank model and frequency do not match")
    production = lean_g5_control_bank_geometry(checked_frequency)
    count = production.path_count if path_count is None else path_count
    length = production.history_length if history_length is None else history_length
    canonical = _geometry(
        checked_frequency,
        path_count=count,
        history_length=length,
        noncanonical_use=noncanonical_use,
        path_count_is_production=False,
    )
    worker_count = _bounded_integer(workers, "control-bank workers", minimum=1)
    if worker_count > _MAXIMUM_WORKERS:
        raise ValidationError("control-bank workers cannot exceed nine")
    batch = _bounded_integer(batch_size, "control-bank batch size", minimum=1)
    block_length = _mean_block_length(mean_block_length)

    jobs = tuple(
        _ScoredControlWork(
            source=source,
            continuation_links=links,
            source_fingerprint=source_fingerprint,
            model=model,
            bank_role=role,
            frequency=checked_frequency,
            history_length=length,
            mean_block_length=block_length,
            master_seed=master_seed,
            scientific_version=scientific_version,
            start=start,
            stop=min(start + batch, count),
        )
        for start in range(0, count, batch)
    )
    chunks = deterministic_process_map(
        _score_control_chunk,
        jobs,
        workers=min(worker_count, len(jobs)),
        chunksize=1,
    )
    alpha_rows = np.empty((count, _STRATEGY_COUNT), dtype=np.float64)
    expected_start = 0
    score_observations: int | None = None
    for chunk in chunks:
        if chunk.start != expected_start or chunk.stop <= chunk.start:
            raise ValidationError("control-bank worker chunks are missing or unordered")
        if chunk.alpha_rows.shape != (chunk.stop - chunk.start, _STRATEGY_COUNT):
            raise ValidationError("control-bank worker returned the wrong score shape")
        if score_observations is None:
            score_observations = chunk.score_observations
        elif score_observations != chunk.score_observations:
            raise ValidationError("control-bank score histories are inconsistent")
        alpha_rows[chunk.start : chunk.stop] = chunk.alpha_rows
        expected_start = chunk.stop
    if (
        expected_start != count
        or score_observations is None
        or not bool(np.isfinite(alpha_rows).all())
    ):
        raise ValidationError("control-bank scoring is incomplete or non-finite")
    frozen_rows = _freeze_float(alpha_rows)
    rows_sha256 = hashlib.sha256(frozen_rows.tobytes(order="C")).hexdigest()
    artifact, replicate_start = _ROLE_RNG_COORDINATES[role]
    rng_fingerprint = _rng_bank_fingerprint(
        bank_role=role,
        frequency=checked_frequency,
        path_count=count,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    identity = {
        "schema_id": LEAN_G5_CONTROL_BANK_SCHEMA_ID,
        "contract_id": LEAN_G5_CONTROL_BANK_CONTRACT_ID,
        "bank_role": role,
        "frequency": checked_frequency,
        "path_count": count,
        "history_length": length,
        "score_observations": score_observations,
        "production_geometry": canonical,
        "noncanonical_use": noncanonical_use,
        "mean_block_length": block_length,
        "source_observations": int(source.shape[0]),
        "source_fingerprint": source_fingerprint,
        "model_fingerprint": model.fingerprint,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_contract": RNG_CONTRACT_VERSION,
        "rng_artifact": int(artifact),
        "rng_replicate_start": replicate_start,
        "rng_identity_fingerprint": rng_fingerprint,
        "strategy_names": LEAN_G5_STRATEGY_NAMES,
        "alpha_rows_sha256": rows_sha256,
    }
    return LeanG5ScoredControlBank(
        schema_id=LEAN_G5_CONTROL_BANK_SCHEMA_ID,
        contract_id=LEAN_G5_CONTROL_BANK_CONTRACT_ID,
        bank_role=role,
        frequency=checked_frequency,
        path_count=count,
        history_length=length,
        score_observations=score_observations,
        production_geometry=canonical,
        noncanonical_use=noncanonical_use,
        mean_block_length=block_length,
        source_observations=int(source.shape[0]),
        source_fingerprint=source_fingerprint,
        model_fingerprint=model.fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_contract=RNG_CONTRACT_VERSION,
        rng_artifact=int(artifact),
        rng_replicate_start=replicate_start,
        rng_identity_fingerprint=rng_fingerprint,
        strategy_names=LEAN_G5_STRATEGY_NAMES,
        alpha_rows=frozen_rows,
        alpha_rows_sha256=rows_sha256,
        fingerprint=_fingerprint(identity),
    )


def build_scored_lean_g5_control_banks(
    historical_log_returns: ArrayLike,
    source_continuation_links: ArrayLike,
    model: LeanG5StrategyModel,
    *,
    frequency: Frequency,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
    path_count: int | None = None,
    history_length: int | None = None,
    noncanonical_use: NoncanonicalUse | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> tuple[LeanG5ScoredControlBank, ...]:
    """Build A, B, then C, running at most one parallel bank at a time."""

    return tuple(
        build_scored_lean_g5_control_bank(
            historical_log_returns,
            source_continuation_links,
            model,
            bank_role=role,
            frequency=frequency,
            mean_block_length=mean_block_length,
            master_seed=master_seed,
            scientific_version=scientific_version,
            workers=workers,
            path_count=path_count,
            history_length=history_length,
            noncanonical_use=noncanonical_use,
            batch_size=batch_size,
        )
        for role in LEAN_G5_CONTROL_BANK_ROLES
    )


def _score_control_chunk(work: _ScoredControlWork) -> _ScoredAlphaChunk:
    rows = np.empty((work.stop - work.start, _STRATEGY_COUNT), dtype=np.float64)
    observations: int | None = None
    for offset, path_index in enumerate(range(work.start, work.stop)):
        draw = _draw_checked_control_path(
            work.source,
            work.continuation_links,
            source_fingerprint=work.source_fingerprint,
            bank_role=work.bank_role,
            frequency=work.frequency,
            path_index=path_index,
            history_length=work.history_length,
            mean_block_length=work.mean_block_length,
            master_seed=work.master_seed,
            scientific_version=work.scientific_version,
        )
        score = score_lean_g5_path(work.model, draw.log_returns)
        if observations is None:
            observations = score.observations
        elif observations != score.observations:
            raise ValidationError("control-bank path scores have unequal history")
        rows[offset] = score.alpha_sharpes
    if observations is None:  # pragma: no cover - work ranges are nonempty by design
        raise AssertionError("empty control-bank worker range")
    return _ScoredAlphaChunk(
        start=work.start,
        stop=work.stop,
        score_observations=observations,
        alpha_rows=_freeze_float(rows),
    )


def _draw_checked_control_path(
    source: FloatArray,
    continuation_links: BoolArray,
    *,
    source_fingerprint: str,
    bank_role: BankRole,
    frequency: Frequency,
    path_index: int,
    history_length: int,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
) -> LeanG5HistoricalControlDraw:
    rng_identity = lean_g5_control_rng_identity(
        bank_role=bank_role,
        frequency=frequency,
        path_index=path_index,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    key = calibration_rng_key(
        master_seed=master_seed,
        scientific_version=scientific_version,
        purpose=CalibrationPurpose.G5_OUTER_NULL,
        artifact=CalibrationArtifact(rng_identity.artifact),
        variant=0,
        replicate=rng_identity.replicate,
        domain=Domain.THRESHOLD_CALIBRATION,
        family=Family(rng_identity.family),
        stage=Stage.REFERENCE_RESAMPLING,
    )
    draws = np.asarray(key.generator().random((history_length, 2)), dtype=np.float64)
    trace = unconditioned_stationary_bootstrap_trace(
        int(source.shape[0]),
        continuation_links,
        draws[:, 0],
        draws[:, 1],
        mean_block_length=mean_block_length,
    )
    path = _freeze_float(source[trace.source_indices])
    source_indices = _freeze_int(trace.source_indices)
    restart_flags = _freeze_bool(trace.restart_flags)
    identity = {
        "bank_role": bank_role,
        "frequency": frequency,
        "path_index": path_index,
        "history_length": history_length,
        "mean_block_length": mean_block_length,
        "source_fingerprint": source_fingerprint,
        "rng_identity_fingerprint": rng_identity.fingerprint,
        "source_indices_sha256": hashlib.sha256(
            source_indices.tobytes(order="C")
        ).hexdigest(),
        "log_returns_sha256": hashlib.sha256(path.tobytes(order="C")).hexdigest(),
    }
    return LeanG5HistoricalControlDraw(
        bank_role=bank_role,
        frequency=frequency,
        path_index=path_index,
        history_length=history_length,
        mean_block_length=mean_block_length,
        source_fingerprint=source_fingerprint,
        rng_identity=rng_identity,
        source_indices=source_indices,
        restart_flags=restart_flags,
        log_returns=path,
        fingerprint=_fingerprint(identity),
    )


def _historical_source(
    log_returns: ArrayLike, continuation_links: ArrayLike
) -> tuple[FloatArray, BoolArray, str]:
    try:
        source = np.asarray(log_returns, dtype="<f8", order="C").copy()
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "control-bank historical returns must be numeric"
        ) from error
    if source.ndim != 2 or source.shape[1:] != (_ASSET_COUNT,) or source.shape[0] < 2:
        raise ValidationError(
            "control-bank historical returns must have shape (observations, 3)"
        )
    if not bool(np.isfinite(source).all()):
        raise ValidationError("control-bank historical returns must be finite")
    raw_links = np.asarray(continuation_links)
    if (
        raw_links.ndim != 1
        or raw_links.dtype.kind != "b"
        or raw_links.size != source.shape[0] - 1
    ):
        raise ValidationError(
            "control-bank continuation links must be boolean and one row shorter"
        )
    links = raw_links.astype(np.bool_, copy=True)
    frozen_source = _freeze_float(source)
    frozen_links = _freeze_bool(links)
    identity = {
        "asset_order": ("equity", "muni_bond", "taxable_bond"),
        "observations": int(frozen_source.shape[0]),
        "log_returns_sha256": hashlib.sha256(
            frozen_source.tobytes(order="C")
        ).hexdigest(),
        "continuation_links_sha256": hashlib.sha256(
            frozen_links.tobytes(order="C")
        ).hexdigest(),
    }
    return frozen_source, frozen_links, _fingerprint(identity)


def _geometry(
    frequency: Frequency,
    *,
    path_count: object,
    history_length: object,
    noncanonical_use: object,
    path_count_is_production: bool,
) -> bool:
    production = lean_g5_control_bank_geometry(frequency)
    count = _bounded_integer(path_count, "control-bank path count", minimum=1)
    length = _bounded_integer(history_length, "control-bank history length", minimum=1)
    minimum_length = 17 if frequency == "monthly" else 257
    if length < minimum_length:
        raise ValidationError("control-bank history is too short for alpha scoring")
    if count > production.path_count or length > production.history_length:
        raise ValidationError(
            "control-bank overrides may only reduce production geometry"
        )
    exact = count == production.path_count and length == production.history_length
    if path_count_is_production:
        exact = length == production.history_length
    if exact and noncanonical_use is not None:
        raise ValidationError("production control-bank geometry cannot be relabelled")
    if not exact and noncanonical_use not in {"pilot", "test"}:
        raise ValidationError(
            "reduced control-bank geometry must be labelled pilot or test"
        )
    return exact


def _rng_bank_fingerprint(
    *,
    bank_role: BankRole,
    frequency: Frequency,
    path_count: int,
    master_seed: int,
    scientific_version: int,
) -> str:
    fingerprints = tuple(
        lean_g5_control_rng_identity(
            bank_role=bank_role,
            frequency=frequency,
            path_index=index,
            master_seed=master_seed,
            scientific_version=scientific_version,
        ).fingerprint
        for index in range(path_count)
    )
    return _fingerprint(
        {
            "bank_role": bank_role,
            "frequency": frequency,
            "path_count": path_count,
            "path_key_fingerprints": fingerprints,
        }
    )


def _bank_role(value: object) -> BankRole:
    if value == "A":
        return "A"
    if value == "B":
        return "B"
    if value == "C":
        return "C"
    raise ValidationError("control-bank role must be A, B, or C")


def _frequency(value: object) -> Frequency:
    if value == "monthly":
        return "monthly"
    if value == "daily":
        return "daily"
    raise ValidationError("control-bank frequency must be monthly or daily")


def _mean_block_length(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValidationError("control-bank mean block length must be at least one")
    result = float(value)
    if not math.isfinite(result) or result < 1.0:
        raise ValidationError("control-bank mean block length must be at least one")
    return result


def _bounded_integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{label} must be an integer at least {minimum}")
    return value


def _freeze_float(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _freeze_int(value: ArrayLike) -> IntArray:
    array = np.asarray(value, dtype="<i8", order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype="<i8").reshape(array.shape)


def _freeze_bool(value: ArrayLike) -> BoolArray:
    array = np.asarray(value, dtype=np.bool_, order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)


def _fingerprint(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
