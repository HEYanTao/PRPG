"""Deterministic G5 raw-return outer-null calibration.

This module implements the independent raw-history reference described in
design Section 14.1.  It deliberately does *not* run the candidate HMM or the
conditioned production kernel.  Every replicate draws two independent,
equal-length pseudo-histories from synchronized three-asset historical rows by
an unconditioned, non-circular stationary bootstrap.

The result is a bounded foundation for the complete G5 suite.  It calibrates
the base-frequency location/scale, covariance/correlation, marginal,
taxable-minus-municipal spread, serial-dependence, and drawdown max statistics
available from :mod:`prpg.validation.metrics`.  Later G5 components add the
pre-registered adversarial, joint-distance, state, and policy-specific
families.  Consequently, even an exact 50,000-replicate result here is never a
generation authority; only the separately sealed G5 evidence coordinator can
create that authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError, ValidationError
from prpg.execution import deterministic_process_map, resolve_worker_count
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)
from prpg.validation.metrics import (
    BaseFrequency,
    BaseFrequencyMetrics,
    compute_base_frequency_metrics,
)
from prpg.validation.records import CanonicalThresholds, build_canonical_thresholds
from prpg.validation.statistics import (
    OUTER_NULL_REPLICATES,
    empirical_higher_critical_value,
    pointwise_outer_null_audit,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

OUTER_NULL_SCHEMA_ID: Final = "prpg-g5-raw-outer-null-v1"
OUTER_NULL_SCHEMA_VERSION: Final = 1
OUTER_NULL_CRITICAL_PROBABILITY: Final = 0.95
OUTER_NULL_STATISTIC_NAMES: Final = (
    "location_scale_cap_ratio_max",
    "covariance_correlation_cap_ratio_max",
    "marginal_quantile_cap_ratio_max",
    "spread_cap_ratio_max",
    "serial_dependence_max",
    "drawdown_max",
)

# The first four statistics are ratios to owner-approved practical caps.  A
# fixed exceedance of one is therefore eligible for the pointwise precision
# audit; it is not a second multiplicity procedure.
_PRACTICAL_CAP_STATISTIC_COUNT: Final = 4
_MEAN_OVER_VOL_CAP: Final = 0.05
_ABS_ANNUAL_MEAN_CAP: Final = 0.01
_RELATIVE_VOL_CAP: Final = 0.05
_COVARIANCE_CAP: Final = 0.10
_CORRELATION_CAP: Final = 0.05
_QUANTILE_OVER_SIGMA_CAP: Final = 0.15
_SPREAD_MEAN_OVER_VOL_CAP: Final = 0.05
_SPREAD_ABS_ANNUAL_MEAN_CAP: Final = 0.005
_SPREAD_RELATIVE_VOL_CAP: Final = 0.05
_SPREAD_QUANTILE_OVER_SIGMA_CAP: Final = 0.15
_WORKER_BASELINE_BYTES: Final = 128 * 1024 * 1024
_PARENT_BASELINE_BYTES: Final = 128 * 1024 * 1024
_MAXIMUM_MEMORY_FRACTION: Final = 0.70
_FINGERPRINT: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")


class OuterBootstrapReason(StrEnum):
    """Mutually exclusive restart outcome for one unconditioned trace row."""

    FIRST_OBSERVATION = "first_observation"
    SOURCE_END = "source_end"
    SOURCE_GAP = "source_gap"
    RANDOM_RESTART = "random_restart"
    CONTINUATION = "continuation"


@dataclass(frozen=True, slots=True)
class OuterBootstrapTrace:
    """One auditable non-circular stationary-bootstrap index trace."""

    source_indices: IntArray
    restart_flags: BoolArray
    reasons: tuple[OuterBootstrapReason, ...]
    restart_uniforms_consumed: int
    source_uniforms_consumed: int


@dataclass(frozen=True, slots=True)
class RegisteredOuterNullPair:
    """The two independently keyed pseudo-histories for one null replicate."""

    replicate: int
    first_trace: OuterBootstrapTrace
    second_trace: OuterBootstrapTrace


@dataclass(frozen=True, slots=True)
class OuterNullBaseStatistics:
    """One immutable candidate/reference base-statistic vector."""

    statistic_names: tuple[str, ...]
    values: tuple[float, ...]

    def as_mapping(self) -> dict[str, float]:
        """Return name/value pairs in the frozen registry order."""

        return dict(zip(self.statistic_names, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class OuterNullCriticalValue:
    """A higher empirical quantile for canonical or reduced runs."""

    name: str
    probability: float
    sample_count: int
    method: str
    zero_based_index: int
    one_based_rank: int
    value: float

    def as_dict(self) -> dict[str, str | float | int]:
        """Return a stable JSON-compatible representation."""

        return {
            "name": self.name,
            "probability": self.probability,
            "sample_count": self.sample_count,
            "method": self.method,
            "zero_based_index": self.zero_based_index,
            "one_based_rank": self.one_based_rank,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class OuterNullCalibrationResult:
    """Complete base-metric calibration outcome, never generation authority."""

    schema_id: str
    schema_version: int
    frequency: BaseFrequency
    observations: int
    mean_block_length: float
    restart_probability: float
    master_seed: int
    scientific_version: int
    replicate_count: int
    workers: int
    batch_size: int
    statistic_names: tuple[str, ...]
    replicate_statistics: FloatArray
    critical_values: tuple[OuterNullCriticalValue, ...]
    statistics_sha256: str
    source_sha256: str
    wall_seconds: float
    fixed_sample_canonical: bool
    canonical_thresholds: CanonicalThresholds | None
    sealed_g5_publication: bool
    generation_authorized: bool
    fingerprint: str


@dataclass(frozen=True, slots=True)
class OuterNullResourceProjection:
    """Conservative memory and linear-throughput projection for a target run."""

    schema_id: str
    observations: int
    target_replicates: int
    workers: int
    statistic_count: int
    estimated_peak_memory_bytes: int
    physical_memory_bytes: int
    maximum_memory_fraction: float
    estimated_memory_fraction: float
    memory_passed: bool
    pilot_replicates: int
    pilot_wall_seconds: float
    projected_wall_seconds: float
    projected_wall_hours: float
    target_is_canonical_count: bool
    generation_authorized: bool


@dataclass(frozen=True, slots=True)
class _OuterNullWork:
    source: FloatArray
    continuation_links: BoolArray
    frequency: BaseFrequency
    mean_block_length: float
    master_seed: int
    scientific_version: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _OuterNullChunk:
    start: int
    stop: int
    statistics: FloatArray


def unconditioned_stationary_bootstrap_trace(
    source_observations: int,
    source_continuation_links: ArrayLike,
    restart_uniforms: ArrayLike,
    source_uniforms: ArrayLike,
    *,
    mean_block_length: float,
) -> OuterBootstrapTrace:
    """Build one exact unconditioned non-circular source-index trace.

    The first row, source end, and source gap force a restart.  Otherwise the
    independent Bernoulli draw restarts with probability ``1 / L``.  Both
    supplied uniform vectors contain one pre-draw per output row, so branch
    outcomes never alter later random-number consumption.  A restart chooses
    uniformly from every valid synchronized source row.
    """

    observations = _positive_integer(source_observations, "source observations")
    links = _continuation_links(source_continuation_links, observations)
    restart = _uniform_vector(restart_uniforms, "restart uniforms")
    source = _uniform_vector(source_uniforms, "source uniforms")
    if restart.size != source.size or restart.size == 0:
        raise ValidationError(
            "outer-bootstrap uniform vectors must have the same positive length"
        )
    probability = _restart_probability(mean_block_length)
    indices, flags, reasons = _bootstrap_indices(
        observations,
        links,
        restart,
        source,
        probability,
        retain_reasons=True,
    )
    assert reasons is not None
    return OuterBootstrapTrace(
        source_indices=indices,
        restart_flags=flags,
        reasons=reasons,
        restart_uniforms_consumed=int(restart.size),
        source_uniforms_consumed=int(source.size),
    )


def registered_outer_null_pair(
    source_observations: int,
    source_continuation_links: ArrayLike,
    *,
    output_observations: int | None = None,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
    family: Family,
    replicate: int,
) -> RegisteredOuterNullPair:
    """Draw a replicate's independent artifact-0/artifact-1 registered pair."""

    observations = _positive_integer(source_observations, "source observations")
    output = (
        observations
        if output_observations is None
        else _positive_integer(output_observations, "output observations")
    )
    links = _continuation_links(source_continuation_links, observations)
    checked_family = _outer_family(family)
    checked_replicate = _replicate(replicate)
    traces: list[OuterBootstrapTrace] = []
    for artifact in (
        CalibrationArtifact.DESIGN_OR_CONTROL,
        CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    ):
        key = calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.G5_OUTER_NULL,
            artifact=artifact,
            variant=0,
            replicate=checked_replicate,
            domain=Domain.THRESHOLD_CALIBRATION,
            family=checked_family,
            stage=Stage.REFERENCE_RESAMPLING,
        )
        draws = np.asarray(key.generator().random((output, 2)), dtype=np.float64)
        traces.append(
            unconditioned_stationary_bootstrap_trace(
                observations,
                links,
                draws[:, 0],
                draws[:, 1],
                mean_block_length=mean_block_length,
            )
        )
    return RegisteredOuterNullPair(checked_replicate, traces[0], traces[1])


def calibrate_outer_null(
    historical_log_returns: ArrayLike,
    source_continuation_links: ArrayLike,
    *,
    frequency: BaseFrequency,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
    replicate_count: int = OUTER_NULL_REPLICATES,
    workers: int = 1,
    batch_size: int = 512,
    threshold_set_id: str | None = None,
    binding_fingerprints: Mapping[str, str] | None = None,
    sobol_direction_fingerprint: str | None = None,
) -> OuterNullCalibrationResult:
    """Run exact registered outer-null replicates in bounded ordered batches.

    Reduced counts are explicitly noncanonical rehearsals.  At exactly 50,000
    replicates the canonical threshold metadata is mandatory and the existing
    strict G5 threshold record builder is used.  Regardless of count, this
    function never seals G5 evidence or authorizes generation.
    """

    source = _source_matrix(historical_log_returns)
    checked_frequency = _frequency(frequency)
    _frequency_support(source.shape[0], checked_frequency)
    links = _continuation_links(source_continuation_links, source.shape[0])
    block_length = _mean_block_length(mean_block_length)
    count = _replicate_count(replicate_count)
    batch = _positive_integer(batch_size, "outer-null batch size")
    worker_count = _workers(workers, count)
    _scientific_identity(master_seed, scientific_version)
    canonical = count == OUTER_NULL_REPLICATES
    _threshold_metadata(
        canonical=canonical,
        threshold_set_id=threshold_set_id,
        binding_fingerprints=binding_fingerprints,
        sobol_direction_fingerprint=sobol_direction_fingerprint,
    )

    started = time.monotonic()
    jobs = tuple(
        _OuterNullWork(
            source=source,
            continuation_links=links,
            frequency=checked_frequency,
            mean_block_length=block_length,
            master_seed=master_seed,
            scientific_version=scientific_version,
            start=start,
            stop=min(start + batch, count),
        )
        for start in range(0, count, batch)
    )
    chunks = deterministic_process_map(
        _run_outer_null_chunk,
        jobs,
        workers=worker_count,
        chunksize=1,
    )
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed < 0.0:  # pragma: no cover
        raise AssertionError("monotonic outer-null timer is invalid")

    statistics = np.empty((count, len(OUTER_NULL_STATISTIC_NAMES)), dtype=np.float64)
    expected_start = 0
    for chunk in chunks:
        if chunk.start != expected_start or chunk.stop <= chunk.start:
            raise ValidationError(
                "outer-null worker chunks are missing or out of order"
            )
        if chunk.statistics.shape != (
            chunk.stop - chunk.start,
            len(OUTER_NULL_STATISTIC_NAMES),
        ):
            raise ValidationError(
                "outer-null worker returned the wrong statistic shape"
            )
        statistics[chunk.start : chunk.stop] = chunk.statistics
        expected_start = chunk.stop
    if expected_start != count or not bool(np.isfinite(statistics).all()):
        raise ValidationError("outer-null run is incomplete or non-finite")
    statistics.setflags(write=False)
    return summarize_outer_null_statistics(
        statistics,
        frequency=checked_frequency,
        observations=int(source.shape[0]),
        mean_block_length=block_length,
        workers=worker_count,
        batch_size=batch,
        source_sha256=_source_sha256(source, links),
        wall_seconds=elapsed,
        master_seed=master_seed,
        scientific_version=scientific_version,
        threshold_set_id=threshold_set_id,
        binding_fingerprints=binding_fingerprints,
        sobol_direction_fingerprint=sobol_direction_fingerprint,
    )


def compute_outer_null_base_max_statistics(
    first_log_returns: ArrayLike,
    reference_log_returns: ArrayLike,
    *,
    frequency: BaseFrequency,
) -> OuterNullBaseStatistics:
    """Compute the exact candidate/reference vector used by calibration.

    The first matrix is the candidate-like sample and the second matrix is the
    raw historical reference; practical-cap denominators therefore come only
    from the second matrix, matching later qualification use.
    """

    first = _source_matrix(first_log_returns)
    reference = _source_matrix(reference_log_returns)
    checked_frequency = _frequency(frequency)
    if first.shape != reference.shape:
        raise ValidationError(
            "outer-null candidate and reference histories must have equal shape"
        )
    _frequency_support(int(first.shape[0]), checked_frequency)
    values = _base_max_statistics(first, reference, frequency=checked_frequency)
    return OuterNullBaseStatistics(
        statistic_names=OUTER_NULL_STATISTIC_NAMES,
        values=tuple(float(value) for value in values),
    )


def summarize_outer_null_statistics(
    replicate_statistics: ArrayLike,
    *,
    frequency: BaseFrequency,
    observations: int,
    mean_block_length: float,
    workers: int,
    batch_size: int,
    source_sha256: str,
    wall_seconds: float,
    master_seed: int,
    scientific_version: int,
    threshold_set_id: str | None = None,
    binding_fingerprints: Mapping[str, str] | None = None,
    sobol_direction_fingerprint: str | None = None,
) -> OuterNullCalibrationResult:
    """Validate statistics, select exact higher ranks, and build canonical records."""

    try:
        values = np.asarray(replicate_statistics, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise ValidationError("outer-null statistics must be numeric") from error
    if values.ndim != 2 or values.shape[1] != len(OUTER_NULL_STATISTIC_NAMES):
        raise ValidationError("outer-null statistics have the wrong matrix shape")
    count = _replicate_count(int(values.shape[0]))
    if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValidationError("outer-null statistics must be finite and non-negative")
    checked_frequency = _frequency(frequency)
    checked_observations = _positive_integer(observations, "observations")
    _frequency_support(checked_observations, checked_frequency)
    block_length = _mean_block_length(mean_block_length)
    worker_count = _positive_integer(workers, "workers")
    batch = _positive_integer(batch_size, "outer-null batch size")
    source_fingerprint = _sha256(source_sha256, "source SHA-256")
    elapsed = _nonnegative_finite(wall_seconds, "outer-null wall seconds")
    _scientific_identity(master_seed, scientific_version)
    canonical = count == OUTER_NULL_REPLICATES
    _threshold_metadata(
        canonical=canonical,
        threshold_set_id=threshold_set_id,
        binding_fingerprints=binding_fingerprints,
        sobol_direction_fingerprint=sobol_direction_fingerprint,
    )

    critical_values = tuple(
        _higher_critical(name, values[:, index], OUTER_NULL_CRITICAL_PROBABILITY)
        for index, name in enumerate(OUTER_NULL_STATISTIC_NAMES)
    )
    thresholds: CanonicalThresholds | None = None
    if canonical:
        assert threshold_set_id is not None
        assert binding_fingerprints is not None
        assert sobol_direction_fingerprint is not None
        empirical = {
            _qualified_name(checked_frequency, name): empirical_higher_critical_value(
                values[:, index], OUTER_NULL_CRITICAL_PROBABILITY
            )
            for index, name in enumerate(OUTER_NULL_STATISTIC_NAMES)
        }
        audits = {
            f"{_qualified_name(checked_frequency, name)}.practical_cap": (
                pointwise_outer_null_audit(
                    int(np.count_nonzero(values[:, index] > 1.0))
                )
            )
            for index, name in enumerate(
                OUTER_NULL_STATISTIC_NAMES[:_PRACTICAL_CAP_STATISTIC_COUNT]
            )
        }
        thresholds = build_canonical_thresholds(
            threshold_set_id=threshold_set_id,
            binding_fingerprints=binding_fingerprints,
            sobol_direction_fingerprint=sobol_direction_fingerprint,
            critical_values=empirical,
            audit_intervals=audits,
        )

    normalized = np.asarray(values, dtype="<f8", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    statistics_sha256 = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    identity: dict[str, object] = {
        "schema_id": OUTER_NULL_SCHEMA_ID,
        "schema_version": OUTER_NULL_SCHEMA_VERSION,
        "frequency": checked_frequency,
        "observations": checked_observations,
        "mean_block_length": block_length,
        "restart_probability": 1.0 / block_length,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "replicate_count": count,
        "statistic_names": list(OUTER_NULL_STATISTIC_NAMES),
        "critical_values": [item.as_dict() for item in critical_values],
        "statistics_sha256": statistics_sha256,
        "source_sha256": source_fingerprint,
        "fixed_sample_canonical": canonical,
        "canonical_threshold_fingerprint": (
            None if thresholds is None else thresholds.fingerprint
        ),
        "sealed_g5_publication": False,
        "generation_authorized": False,
    }
    values.setflags(write=False)
    return OuterNullCalibrationResult(
        schema_id=OUTER_NULL_SCHEMA_ID,
        schema_version=OUTER_NULL_SCHEMA_VERSION,
        frequency=checked_frequency,
        observations=checked_observations,
        mean_block_length=block_length,
        restart_probability=1.0 / block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
        replicate_count=count,
        workers=worker_count,
        batch_size=batch,
        statistic_names=OUTER_NULL_STATISTIC_NAMES,
        replicate_statistics=values,
        critical_values=critical_values,
        statistics_sha256=statistics_sha256,
        source_sha256=source_fingerprint,
        wall_seconds=elapsed,
        fixed_sample_canonical=canonical,
        canonical_thresholds=thresholds,
        sealed_g5_publication=False,
        generation_authorized=False,
        fingerprint=_fingerprint(identity),
    )


def project_outer_null_resources(
    pilot: OuterNullCalibrationResult,
    *,
    target_replicates: int = OUTER_NULL_REPLICATES,
    physical_memory_bytes: int,
) -> OuterNullResourceProjection:
    """Project a same-host/same-worker target from one completed pilot."""

    if not isinstance(pilot, OuterNullCalibrationResult):
        raise ValidationError("outer-null projection requires a completed pilot")
    target = _replicate_count(target_replicates)
    memory = _positive_integer(physical_memory_bytes, "physical memory bytes")
    if pilot.replicate_count <= 0 or pilot.wall_seconds <= 0.0:
        raise ValidationError("outer-null pilot must have positive measured duration")
    estimated_peak = estimate_outer_null_peak_memory_bytes(
        observations=pilot.observations,
        replicate_count=target,
        workers=pilot.workers,
        statistic_count=len(pilot.statistic_names),
    )
    fraction = estimated_peak / memory
    projected_seconds = pilot.wall_seconds * target / pilot.replicate_count
    return OuterNullResourceProjection(
        schema_id="prpg-g5-outer-null-resource-projection-v1",
        observations=pilot.observations,
        target_replicates=target,
        workers=pilot.workers,
        statistic_count=len(pilot.statistic_names),
        estimated_peak_memory_bytes=estimated_peak,
        physical_memory_bytes=memory,
        maximum_memory_fraction=_MAXIMUM_MEMORY_FRACTION,
        estimated_memory_fraction=fraction,
        memory_passed=fraction <= _MAXIMUM_MEMORY_FRACTION,
        pilot_replicates=pilot.replicate_count,
        pilot_wall_seconds=pilot.wall_seconds,
        projected_wall_seconds=projected_seconds,
        projected_wall_hours=projected_seconds / 3600.0,
        target_is_canonical_count=target == OUTER_NULL_REPLICATES,
        generation_authorized=False,
    )


def estimate_outer_null_peak_memory_bytes(
    *,
    observations: int,
    replicate_count: int,
    workers: int,
    statistic_count: int = len(OUTER_NULL_STATISTIC_NAMES),
) -> int:
    """Return a conservative resident-memory estimate without executing work."""

    rows = _positive_integer(observations, "observations")
    replicates = _replicate_count(replicate_count)
    worker_count = _positive_integer(workers, "workers")
    statistics = _positive_integer(statistic_count, "statistic count")
    source_and_links = rows * (3 * 8 + 1)
    # Two source-index arrays, two copied histories, and one stream's pair of
    # pre-drawn uniform arrays at a time, plus NumPy intermediate headroom.
    per_worker_arrays = rows * (2 * 8 + 2 * 3 * 8 + 2 * 8 + 64)
    parent_statistics = replicates * statistics * 8 * 2
    return int(
        _PARENT_BASELINE_BYTES
        + source_and_links
        + parent_statistics
        + worker_count * (_WORKER_BASELINE_BYTES + source_and_links + per_worker_arrays)
    )


def _run_outer_null_chunk(work: _OuterNullWork) -> _OuterNullChunk:
    result = np.empty(
        (work.stop - work.start, len(OUTER_NULL_STATISTIC_NAMES)),
        dtype=np.float64,
    )
    family = Family.LF if work.frequency == "monthly" else Family.HF
    for output_row, replicate in enumerate(range(work.start, work.stop)):
        first_indices, second_indices = _registered_pair_indices(
            source_observations=int(work.source.shape[0]),
            continuation_links=work.continuation_links,
            output_observations=int(work.source.shape[0]),
            mean_block_length=work.mean_block_length,
            master_seed=work.master_seed,
            scientific_version=work.scientific_version,
            family=family,
            replicate=replicate,
        )
        first = np.asarray(work.source[first_indices], dtype=np.float64)
        second = np.asarray(work.source[second_indices], dtype=np.float64)
        result[output_row] = _base_max_statistics(
            first,
            second,
            frequency=work.frequency,
        )
    if not bool(np.isfinite(result).all()) or bool((result < 0.0).any()):
        raise ValidationError("outer-null worker produced invalid statistics")
    result.setflags(write=False)
    return _OuterNullChunk(work.start, work.stop, result)


def _registered_pair_indices(
    *,
    source_observations: int,
    continuation_links: BoolArray,
    output_observations: int,
    mean_block_length: float,
    master_seed: int,
    scientific_version: int,
    family: Family,
    replicate: int,
) -> tuple[IntArray, IntArray]:
    probability = _restart_probability(mean_block_length)
    output: list[IntArray] = []
    for artifact in (
        CalibrationArtifact.DESIGN_OR_CONTROL,
        CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    ):
        key = calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.G5_OUTER_NULL,
            artifact=artifact,
            variant=0,
            replicate=replicate,
            domain=Domain.THRESHOLD_CALIBRATION,
            family=family,
            stage=Stage.REFERENCE_RESAMPLING,
        )
        draws = np.asarray(
            key.generator().random((output_observations, 2)), dtype=np.float64
        )
        indices, _, _ = _bootstrap_indices(
            source_observations,
            continuation_links,
            draws[:, 0],
            draws[:, 1],
            probability,
            retain_reasons=False,
        )
        output.append(indices)
    return output[0], output[1]


def _bootstrap_indices(
    source_observations: int,
    links: BoolArray,
    restart_uniforms: FloatArray,
    source_uniforms: FloatArray,
    restart_probability: float,
    *,
    retain_reasons: bool,
) -> tuple[IntArray, BoolArray, tuple[OuterBootstrapReason, ...] | None]:
    indices = np.empty(restart_uniforms.size, dtype=np.int64)
    flags = np.zeros(restart_uniforms.size, dtype=np.bool_)
    reasons: list[OuterBootstrapReason] | None = [] if retain_reasons else None
    for position in range(restart_uniforms.size):
        if position == 0:
            reason = OuterBootstrapReason.FIRST_OBSERVATION
        else:
            next_source = int(indices[position - 1]) + 1
            if next_source >= source_observations:
                reason = OuterBootstrapReason.SOURCE_END
            elif not links[next_source - 1]:
                reason = OuterBootstrapReason.SOURCE_GAP
            elif restart_uniforms[position] < restart_probability:
                reason = OuterBootstrapReason.RANDOM_RESTART
            else:
                reason = OuterBootstrapReason.CONTINUATION
        restarted = reason is not OuterBootstrapReason.CONTINUATION
        flags[position] = restarted
        if restarted:
            indices[position] = math.floor(
                float(source_uniforms[position]) * source_observations
            )
        else:
            indices[position] = indices[position - 1] + 1
        if reasons is not None:
            reasons.append(reason)
    indices.setflags(write=False)
    flags.setflags(write=False)
    return indices, flags, None if reasons is None else tuple(reasons)


def _base_max_statistics(
    first: FloatArray,
    second: FloatArray,
    *,
    frequency: BaseFrequency,
) -> FloatArray:
    probabilities = (
        (0.10, 0.50, 0.90) if frequency == "monthly" else (0.01, 0.05, 0.50, 0.95, 0.99)
    )
    first_metrics = compute_base_frequency_metrics(
        first,
        frequency=frequency,
        quantile_probabilities=probabilities,
    )
    second_metrics = compute_base_frequency_metrics(
        second,
        frequency=frequency,
        quantile_probabilities=probabilities,
    )
    output = np.asarray(
        [
            _location_scale_statistic(first_metrics, second_metrics),
            _covariance_correlation_statistic(first_metrics, second_metrics),
            _marginal_quantile_statistic(first_metrics, second_metrics),
            _spread_statistic(first, second, frequency=frequency),
            _serial_dependence_statistic(first_metrics, second_metrics),
            _drawdown_statistic(first_metrics, second_metrics),
        ],
        dtype=np.float64,
    )
    if not bool(np.isfinite(output).all()) or bool((output < 0.0).any()):
        raise ValidationError("outer-null base max statistics are invalid")
    return output


def _location_scale_statistic(
    first: BaseFrequencyMetrics, second: BaseFrequencyMetrics
) -> float:
    mean_first = np.asarray(first.annualized_log_mean)
    mean_second = np.asarray(second.annualized_log_mean)
    vol_first = np.asarray(first.annualized_volatility)
    vol_second = np.asarray(second.annualized_volatility)
    reference_vol = _positive_denominator(vol_second)
    components = (
        np.max(np.abs(mean_first - mean_second) / reference_vol) / _MEAN_OVER_VOL_CAP,
        np.max(np.abs(mean_first - mean_second)) / _ABS_ANNUAL_MEAN_CAP,
        np.max(np.abs(vol_first - vol_second) / reference_vol) / _RELATIVE_VOL_CAP,
    )
    return float(max(components))


def _covariance_correlation_statistic(
    first: BaseFrequencyMetrics, second: BaseFrequencyMetrics
) -> float:
    covariance_first = np.asarray(first.base_frequency_covariance)
    covariance_second = np.asarray(second.base_frequency_covariance)
    denominator = np.linalg.norm(covariance_second, ord="fro")
    if not math.isfinite(float(denominator)) or denominator <= 0.0:
        raise ValidationError("outer-null covariance reference is degenerate")
    covariance = float(
        np.linalg.norm(covariance_first - covariance_second, ord="fro")
        / denominator
        / _COVARIANCE_CAP
    )
    correlation_difference = np.abs(
        np.asarray(first.correlation) - np.asarray(second.correlation)
    )
    correlation = (
        float(np.max(correlation_difference[np.triu_indices(3, k=1)]))
        / _CORRELATION_CAP
    )
    return max(covariance, correlation)


def _marginal_quantile_statistic(
    first: BaseFrequencyMetrics, second: BaseFrequencyMetrics
) -> float:
    annualization = float(first.annualization_factor)
    reference_base_sigma = _positive_denominator(
        np.asarray(second.annualized_volatility) / math.sqrt(annualization)
    )
    differences = np.abs(
        np.asarray(first.marginal_quantiles) - np.asarray(second.marginal_quantiles)
    )
    return float(np.max(differences / reference_base_sigma) / _QUANTILE_OVER_SIGMA_CAP)


def _spread_statistic(
    first: FloatArray,
    second: FloatArray,
    *,
    frequency: BaseFrequency,
) -> float:
    annualization = 12 if frequency == "monthly" else 252
    spread_first = first[:, 2] - first[:, 1]
    spread_second = second[:, 2] - second[:, 1]
    base_vol_first = float(np.std(spread_first, ddof=1))
    base_vol_second = float(np.std(spread_second, ddof=1))
    reference_base_vol = base_vol_second
    if not math.isfinite(reference_base_vol) or reference_base_vol <= 0.0:
        raise ValidationError("outer-null taxable-minus-municipal spread is degenerate")
    reference_annual_vol = reference_base_vol * math.sqrt(annualization)
    annual_mean_difference = annualization * abs(
        float(np.mean(spread_first) - np.mean(spread_second))
    )
    relative_volatility = abs(base_vol_first - base_vol_second) / reference_base_vol
    probabilities: Sequence[float] = (
        (0.10, 0.50, 0.90) if frequency == "monthly" else (0.01, 0.05, 0.50, 0.95, 0.99)
    )
    quantile_difference = float(
        np.max(
            np.abs(
                np.quantile(spread_first, probabilities, method="higher")
                - np.quantile(spread_second, probabilities, method="higher")
            )
        )
    )
    return float(
        max(
            annual_mean_difference / reference_annual_vol / _SPREAD_MEAN_OVER_VOL_CAP,
            annual_mean_difference / _SPREAD_ABS_ANNUAL_MEAN_CAP,
            relative_volatility / _SPREAD_RELATIVE_VOL_CAP,
            quantile_difference / reference_base_vol / _SPREAD_QUANTILE_OVER_SIGMA_CAP,
        )
    )


def _serial_dependence_statistic(
    first: BaseFrequencyMetrics, second: BaseFrequencyMetrics
) -> float:
    return float(
        max(
            np.max(np.abs(np.asarray(first.raw_acf) - np.asarray(second.raw_acf))),
            np.max(
                np.abs(np.asarray(first.absolute_acf) - np.asarray(second.absolute_acf))
            ),
            np.max(
                np.abs(np.asarray(first.squared_acf) - np.asarray(second.squared_acf))
            ),
        )
    )


def _drawdown_statistic(
    first: BaseFrequencyMetrics, second: BaseFrequencyMetrics
) -> float:
    depth = float(
        np.max(
            np.abs(
                np.asarray(first.maximum_drawdown) - np.asarray(second.maximum_drawdown)
            )
        )
    )
    duration = float(
        np.max(
            np.abs(
                np.asarray(first.maximum_drawdown_duration, dtype=np.float64)
                - np.asarray(second.maximum_drawdown_duration, dtype=np.float64)
            )
        )
        / first.observations
    )
    return max(depth, duration)


def _higher_critical(
    name: str, values: FloatArray, probability: float
) -> OuterNullCriticalValue:
    zero_based = math.ceil(probability * (values.size - 1))
    selected = float(np.partition(values.copy(), zero_based)[zero_based])
    return OuterNullCriticalValue(
        name=name,
        probability=probability,
        sample_count=int(values.size),
        method="higher",
        zero_based_index=zero_based,
        one_based_rank=zero_based + 1,
        value=selected,
    )


def _source_matrix(values: ArrayLike) -> FloatArray:
    try:
        source = np.asarray(values, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise ValidationError("historical log returns must be numeric") from error
    if source.ndim != 2 or source.shape[1:] != (3,) or source.shape[0] < 2:
        raise ValidationError(
            "historical log returns must have shape (observations, 3)"
        )
    if not bool(np.isfinite(source).all()):
        raise ValidationError("historical log returns must all be finite")
    if bool((np.ptp(source, axis=0) == 0.0).any()):
        raise ValidationError("historical outer-null source has a constant asset")
    spread = source[:, 2] - source[:, 1]
    if np.ptp(spread) == 0.0:
        raise ValidationError("historical outer-null spread is constant")
    source.setflags(write=False)
    return source


def _continuation_links(values: ArrayLike, observations: int) -> BoolArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != observations - 1 or raw.dtype.kind != "b":
        raise ValidationError(
            "outer-null continuation links must be boolean with source length minus one"
        )
    links = raw.astype(np.bool_, copy=True)
    links.setflags(write=False)
    return links


def _uniform_vector(values: ArrayLike, label: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind == "b":
        raise ValidationError(f"{label} must be a one-dimensional numeric vector")
    try:
        result = np.asarray(values, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} must be numeric") from error
    if not bool(np.isfinite(result).all()) or bool(
        ((result < 0.0) | (result >= 1.0)).any()
    ):
        raise ValidationError(f"{label} must be finite and in [0, 1)")
    return result


def _mean_block_length(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValidationError("mean block length must be finite and at least one")
    result = float(value)
    if not math.isfinite(result) or result < 1.0:
        raise ValidationError("mean block length must be finite and at least one")
    return result


def _restart_probability(mean_block_length: object) -> float:
    return 1.0 / _mean_block_length(mean_block_length)


def _replicate(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (0 <= value < OUTER_NULL_REPLICATES)
    ):
        raise ValidationError("outer-null replicate must be in [0, 49,999]")
    return value


def _replicate_count(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (1 <= value <= OUTER_NULL_REPLICATES)
    ):
        raise ValidationError("outer-null replicate count must be in [1, 50,000]")
    return value


def _workers(value: object, replicate_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("outer-null workers must be a positive integer")
    if value > replicate_count:
        raise ValidationError("outer-null workers cannot exceed replicate count")
    try:
        return resolve_worker_count(value, reserve_cores=1)
    except ModelError as error:
        raise ValidationError(
            "outer-null workers exceed available host capacity"
        ) from error


def _outer_family(value: object) -> Family:
    if value is Family.LF:
        return Family.LF
    if value is Family.HF:
        return Family.HF
    raise ValidationError("outer-null family must be LF or HF")


def _frequency(value: object) -> BaseFrequency:
    if value == "daily":
        return "daily"
    if value == "monthly":
        return "monthly"
    raise ValidationError("outer-null frequency must be daily or monthly")


def _frequency_support(observations: int, frequency: BaseFrequency) -> None:
    minimum = 61 if frequency == "daily" else 13
    if observations < minimum:
        raise ValidationError(
            f"outer-null {frequency} source requires at least {minimum} observations"
        )


def _scientific_identity(master_seed: object, scientific_version: object) -> None:
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or not 0 <= master_seed < 1 << 128
    ):
        raise ValidationError(
            "outer-null master seed must be an unsigned 128-bit integer"
        )
    if (
        isinstance(scientific_version, bool)
        or not isinstance(scientific_version, int)
        or not 0 <= scientific_version < 1 << 32
    ):
        raise ValidationError(
            "outer-null scientific version must be an unsigned 32-bit integer"
        )


def _threshold_metadata(
    *,
    canonical: bool,
    threshold_set_id: str | None,
    binding_fingerprints: Mapping[str, str] | None,
    sobol_direction_fingerprint: str | None,
) -> None:
    supplied = (
        threshold_set_id is not None,
        binding_fingerprints is not None,
        sobol_direction_fingerprint is not None,
    )
    if canonical and not all(supplied):
        raise ValidationError(
            "canonical outer-null run requires threshold ID and binding fingerprints"
        )
    if not canonical and any(supplied):
        raise ValidationError(
            "reduced outer-null run must not claim canonical threshold metadata"
        )
    if canonical:
        assert threshold_set_id is not None
        assert binding_fingerprints is not None
        assert sobol_direction_fingerprint is not None
        if _IDENTIFIER.fullmatch(threshold_set_id) is None:
            raise ValidationError("outer-null threshold set ID is invalid")
        if not binding_fingerprints:
            raise ValidationError("outer-null binding fingerprints must be non-empty")
        for name, fingerprint in binding_fingerprints.items():
            if not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None:
                raise ValidationError("outer-null binding name is invalid")
            _sha256(fingerprint, "outer-null binding fingerprint")
        _sha256(sobol_direction_fingerprint, "Sobol direction fingerprint")


def _positive_denominator(values: FloatArray) -> FloatArray:
    if not bool(np.isfinite(values).all()) or bool((values <= 0.0).any()):
        raise ValidationError("outer-null scale denominator is degenerate")
    return values


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{label} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{label} must be finite and non-negative")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValidationError(f"{label} is invalid")
    return value


def _qualified_name(frequency: BaseFrequency, name: str) -> str:
    return f"{frequency}.{name}"


def _source_sha256(source: FloatArray, links: BoolArray) -> str:
    normalized = np.asarray(source, dtype="<f8", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(normalized.tobytes(order="C"))
    digest.update(np.asarray(links, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
