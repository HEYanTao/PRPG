"""Registered execution for kernel-mean calibration and verification.

The statistical estimators live in :mod:`prpg.model.alpha`; this module owns
the Section 4.5/4.6 execution contract around them.  Every history-length
window has one purpose-7, stage-6 stream.  That stream is consumed in the
fixed order ``monthly state, restart, source rank`` and is never shared with
the purpose-4 block/dependence experiment.  Purpose 8 is deliberately
streamless: verification reconstructs the same purpose-7 windows and applies
the frozen offset grid directly.

Worker jobs return only compact sufficient statistics.  The coordinator sorts
their zero-based window ranges before merging, so process completion order and
worker count cannot change the scientific reduction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from itertools import pairwise
from typing import Any, TypeAlias, TypeGuard, cast

import numpy as np
import numpy.typing as npt

from prpg.errors import ModelError
from prpg.execution import deterministic_process_map
from prpg.model.alpha import (
    AlphaOffsetGrid,
    BaseFrequency,
    KernelMeanPrecision,
    KernelWindowStatistics,
    alpha_offset_grid,
    finalize_kernel_mean_precision,
    historical_target_mean,
    kernel_window_statistics,
    merge_kernel_window_statistics,
)
from prpg.model.block_execution import (
    Artifact,
    BlockStateModel,
    CalibrationGeometry,
    RegisteredUniformChunk,
    iter_paired_trace_chunks,
    resolve_calibration_geometry,
)
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.bootstrap import expand_monthly_states_to_daily
from prpg.model.hmm import GaussianHMMFit
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_array_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_entity,
    calibration_rng_key,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
ProcessMap: TypeAlias = Callable[..., tuple[object, ...]]

CANONICAL_PREFIX_WINDOW_COUNTS = (
    10_000,
    20_000,
    40_000,
    80_000,
    160_000,
    320_000,
)
CANONICAL_CONFIDENCE = 0.95
CANONICAL_ANNUALIZED_HALF_WIDTH_MAX = 0.001
CANONICAL_LAMBDAS = (0.25, 0.50, 0.75, 1.00)
CANONICAL_STATISTICS_BATCH_WINDOWS = 128
DEFAULT_PROCESS_CHUNKSIZE = 1
VERIFICATION_ABSOLUTE_TOLERANCE = 1e-12
PURPOSE_7_DRAW_ORDER = (
    "monthly_state_uniforms",
    "restart_uniforms",
    "source_rank_uniforms",
)
_CANONICAL_CELL_ORDER = (
    ("design", "monthly"),
    ("design", "daily"),
    ("production", "monthly"),
    ("production", "daily"),
)
_KERNEL_EXECUTION_CAPABILITY = object()
_MINTED_KERNEL_EXECUTIONS: dict[int, KernelCalibrationExecution] = {}
_KERNEL_IDENTITY_FIELDS = frozenset(
    {
        "parent_model_fingerprint",
        "source_materialization_fingerprint",
        "rng_execution_fingerprint",
        "execution_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class RegisteredKernelWindowBatch:
    """Compact purpose-7 evidence for one contiguous window range."""

    artifact: Artifact
    frequency: BaseFrequency
    first_window_index: int
    stop_window_index: int
    first_stream_entity: int
    last_stream_entity: int
    draws_per_window: int
    draw_order: tuple[str, str, str]
    statistics: KernelWindowStatistics

    @property
    def windows(self) -> int:
        return self.stop_window_index - self.first_window_index


@dataclass(frozen=True, slots=True)
class KernelPrecisionLook:
    """One preregistered cumulative precision decision."""

    look_index: int
    windows: int
    precision: KernelMeanPrecision
    maximum_annualized_half_width: float
    passed: bool


@dataclass(frozen=True, slots=True)
class KernelVerificationResult:
    """Purpose-8 direct replay evidence for every lambda/state/asset offset."""

    purpose: CalibrationPurpose
    stream_count: int
    windows: int
    grid_fingerprint: str
    replay_kernel_sample_fingerprint: str
    observation_counts: IntArray
    directly_adjusted_means: FloatArray
    algebraic_adjusted_means: FloatArray
    algebraic_residual_means: FloatArray
    maximum_absolute_mean_error: float
    maximum_annualized_mean_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class KernelMemoryEstimate:
    """Conservative in-process array allocation for resource preflight."""

    artifact: Artifact
    frequency: BaseFrequency
    workers: int
    batch_windows: int
    verification: bool
    per_worker_bytes: int
    all_workers_bytes: int


@dataclass(frozen=True, slots=True)
class KernelCalibrationExecution:
    """Compact result for one artifact-frequency kernel calibration cell."""

    artifact: Artifact
    frequency: BaseFrequency
    geometry: CalibrationGeometry
    n_states: int
    selected_length: int
    model_artifact_fingerprint: str
    target_history_fingerprint: str
    calibration_stream_fingerprint: str
    draw_order: tuple[str, str, str]
    draws_per_window: int
    planned_looks: tuple[int, ...]
    completed_looks: tuple[KernelPrecisionLook, ...]
    stopped_windows: int
    precision_passed: bool
    verification_passed: bool
    passed: bool
    final_precision: KernelMeanPrecision
    offset_grid: AlphaOffsetGrid | None
    verification: KernelVerificationResult | None
    failure_reason: str | None
    purpose_7_streams: int
    purpose_8_streams: int
    first_purpose_7_entity: int
    last_purpose_7_entity: int
    workers: int
    statistics_batch_windows: int
    process_chunksize: int
    strict_canonical: bool
    parent_model_fingerprint: str | None = None
    source_materialization_fingerprint: str | None = None
    rng_execution_fingerprint: str | None = None
    execution_fingerprint: str | None = None
    _source_model: BlockStateModel | GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_historical_returns: FloatArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_historical_states: IntArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_continuation_links: BoolArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _master_seed: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _scientific_version: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class KernelCalibrationExecutionIdentity:
    """Reverified parent, materialization, RNG, and result identity."""

    master_seed: int
    scientific_version: int
    artifact: Artifact
    frequency: BaseFrequency
    n_states: int
    selected_length: int
    parent_model_fingerprint: str
    source_materialization_fingerprint: str
    rng_execution_fingerprint: str
    execution_fingerprint: str


@dataclass(frozen=True, slots=True)
class KernelCalibrationReplaySource:
    """Exact live parents retained by one registered kernel execution.

    The returned arrays remain immutable bytes-backed views.  This is not an
    authorization token; callers must continue to reverify the parent
    :class:`KernelCalibrationExecution` before and after consuming it.
    """

    identity: KernelCalibrationExecutionIdentity
    model: BlockStateModel | GaussianHMMFit
    historical_returns: FloatArray
    historical_states: IntArray
    continuation_links: BoolArray


def kernel_calibration_execution_identity(
    value: KernelCalibrationExecution,
) -> KernelCalibrationExecutionIdentity:
    """Re-hash one executor-minted kernel result and all retained parents."""

    if (
        not isinstance(value, KernelCalibrationExecution)
        or getattr(value, "_execution_seal", None) is not _KERNEL_EXECUTION_CAPABILITY
        or _MINTED_KERNEL_EXECUTIONS.get(id(value)) is not value
    ):
        raise ModelError("kernel calibration lacks executor authority")
    model = value._source_model
    returns = value._source_historical_returns
    states = value._source_historical_states
    links = value._source_continuation_links
    master_seed = value._master_seed
    scientific_version = value._scientific_version
    if (
        model is None
        or returns is None
        or states is None
        or links is None
        or master_seed is None
        or scientific_version is None
    ):
        raise ModelError("kernel calibration retained source identity is incomplete")
    _require_immutable_bytes_array(returns, "kernel historical returns")
    _require_immutable_bytes_array(states, "kernel historical states")
    _require_immutable_bytes_array(links, "kernel continuation links")
    _require_immutable_public_arrays(value)
    parent_fingerprint = _kernel_parent_model_fingerprint(model)
    materialization_fingerprint = _kernel_source_materialization_fingerprint(
        value=value,
        parent_model_fingerprint=parent_fingerprint,
        historical_returns=returns,
        historical_states=states,
        continuation_links=links,
    )
    rng_fingerprint = _stream_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=_registered_artifact(value.artifact),
        family=_registered_family(value.frequency),
        geometry=value.geometry,
        selected_length=value.selected_length,
    )
    execution_fingerprint = _kernel_execution_fingerprint(value)
    state_model = _coerce_state_model(model)
    if (
        value.n_states != state_model.n_states
        or value.geometry.artifact != value.artifact
        or value.geometry.frequency != value.frequency
        or value.calibration_stream_fingerprint != rng_fingerprint
        or value.parent_model_fingerprint != parent_fingerprint
        or value.source_materialization_fingerprint != materialization_fingerprint
        or value.rng_execution_fingerprint != rng_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("kernel calibration execution identity changed")
    return KernelCalibrationExecutionIdentity(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=value.artifact,
        frequency=value.frequency,
        n_states=value.n_states,
        selected_length=value.selected_length,
        parent_model_fingerprint=parent_fingerprint,
        source_materialization_fingerprint=materialization_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def kernel_calibration_replay_source(
    value: KernelCalibrationExecution,
) -> KernelCalibrationReplaySource:
    """Return the exact immutable sources after full producer reverification."""

    identity = kernel_calibration_execution_identity(value)
    model = value._source_model
    returns = value._source_historical_returns
    states = value._source_historical_states
    links = value._source_continuation_links
    if model is None or returns is None or states is None or links is None:
        raise ModelError("kernel calibration replay source is incomplete")
    return KernelCalibrationReplaySource(identity, model, returns, states, links)


def is_registered_kernel_calibration_execution(
    value: object,
) -> TypeGuard[KernelCalibrationExecution]:
    """Return whether ``value`` is an unchanged executor-minted result."""

    if not isinstance(value, KernelCalibrationExecution):
        return False
    try:
        kernel_calibration_execution_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class KernelCalibrationCellSpec:
    """Caller-owned inputs for one cell in the exact four-cell suite."""

    artifact: Artifact
    frequency: BaseFrequency
    model: BlockStateModel | GaussianHMMFit
    selected_length: int
    historical_returns: npt.ArrayLike
    historical_source_states: npt.ArrayLike
    source_continuation_links: npt.ArrayLike
    model_artifact_fingerprint: str
    target_history_fingerprint: str
    monthly_segment_lengths: Sequence[int] | None = None
    daily_segment_lengths: Sequence[int] | None = None


@dataclass(frozen=True, slots=True)
class KernelCalibrationSuite:
    """Exact design/production by monthly/daily kernel-calibration suite."""

    cells: tuple[KernelCalibrationExecution, ...]
    all_precision_passed: bool
    all_verification_passed: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class _KernelContext:
    model: BlockStateModel
    source_model: BlockStateModel | GaussianHMMFit
    master_seed: int
    scientific_version: int
    artifact: Artifact
    registered_artifact: CalibrationArtifact
    frequency: BaseFrequency
    family: Family
    geometry: CalibrationGeometry
    selected_length: int
    historical_returns: FloatArray
    historical_source_states: IntArray
    source_continuation_links: BoolArray
    model_artifact_fingerprint: str
    calibration_stream_fingerprint: str


@dataclass(frozen=True, slots=True)
class _BatchRange:
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _KernelWorkerJob:
    context: _KernelContext
    ranges: tuple[_BatchRange, ...]
    verification_offsets: FloatArray | None


@dataclass(frozen=True, slots=True)
class _KernelWorkerOutput:
    batch: RegisteredKernelWindowBatch
    directly_adjusted_sums: FloatArray | None


def registered_kernel_window_batch(
    *,
    model: BlockStateModel | GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    artifact: Artifact,
    frequency: BaseFrequency,
    selected_length: int,
    historical_returns: npt.ArrayLike,
    historical_source_states: npt.ArrayLike,
    source_continuation_links: npt.ArrayLike,
    model_artifact_fingerprint: str,
    first_window_index: int,
    window_count: int,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
) -> RegisteredKernelWindowBatch:
    """Generate one independently replayable purpose-7 statistics batch."""

    context = _kernel_context(
        model=model,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        frequency=frequency,
        selected_length=selected_length,
        historical_returns=historical_returns,
        historical_source_states=historical_source_states,
        source_continuation_links=source_continuation_links,
        model_artifact_fingerprint=model_artifact_fingerprint,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    first = _nonnegative_integer(first_window_index, "first_window_index")
    windows = _positive_integer(window_count, "window_count")
    if first + windows > CANONICAL_PREFIX_WINDOW_COUNTS[-1]:
        raise ModelError("kernel batch exceeds the registered purpose-7 range")
    return _generate_batch(context, _BatchRange(first, first + windows), None).batch


def registered_kernel_memory_estimate(
    *,
    artifact: Artifact,
    frequency: BaseFrequency,
    workers: int,
    batch_windows: int = CANONICAL_STATISTICS_BATCH_WINDOWS,
    verification: bool = True,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
) -> KernelMemoryEstimate:
    """Estimate peak NumPy arrays without materializing a purpose-7 stream.

    The estimate includes uniforms, monthly/target states, source indices,
    restart flags, sampled vectors, and two additional sampled-vector-sized
    workspaces during direct purpose-8 verification.  Python/process runtime
    overhead remains the coordinator's separate safety margin.
    """

    worker_count = _positive_integer(workers, "workers")
    windows = _positive_integer(batch_windows, "batch_windows")
    if not isinstance(verification, bool):
        raise ModelError("verification must be boolean")
    geometry = resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    monthly = geometry.monthly_observations
    target = geometry.target_observations
    float_values = windows * (monthly + 2 * target)
    integer_values = windows * (monthly + 2 * target)
    boolean_values = windows * 2 * target
    sampled_values = windows * target * 3
    verification_values = 2 * sampled_values if verification else 0
    per_worker = (
        float_values + integer_values + sampled_values + verification_values
    ) * np.dtype(np.float64).itemsize + boolean_values * np.dtype(np.bool_).itemsize
    return KernelMemoryEstimate(
        artifact=artifact,
        frequency=frequency,
        workers=worker_count,
        batch_windows=windows,
        verification=verification,
        per_worker_bytes=per_worker,
        all_workers_bytes=per_worker * worker_count,
    )


def merge_registered_kernel_batches(
    batches: Sequence[RegisteredKernelWindowBatch],
) -> KernelWindowStatistics:
    """Sort and merge disjoint purpose-7 batches by global window index."""

    if not batches:
        raise ModelError("at least one registered kernel batch is required")
    validated = tuple(_validated_batch(item) for item in batches)
    ordered = tuple(sorted(validated, key=lambda item: item.first_window_index))
    first = ordered[0]
    if first.first_window_index != 0:
        raise ModelError("registered kernel batches must begin at window zero")
    identity = (
        first.artifact,
        first.frequency,
        first.draws_per_window,
        first.draw_order,
        first.statistics.model_artifact_fingerprint,
        first.statistics.calibration_stream_fingerprint,
    )
    previous_stop = 0
    statistics: list[KernelWindowStatistics] = []
    for item in ordered:
        observed_identity = (
            item.artifact,
            item.frequency,
            item.draws_per_window,
            item.draw_order,
            item.statistics.model_artifact_fingerprint,
            item.statistics.calibration_stream_fingerprint,
        )
        if observed_identity != identity:
            raise ModelError("registered kernel batches have inconsistent identity")
        if item.first_window_index != previous_stop:
            raise ModelError(
                "registered kernel batches must be contiguous and disjoint",
                details={
                    "expected_start": previous_stop,
                    "actual_start": item.first_window_index,
                },
            )
        previous_stop = item.stop_window_index
        statistics.append(item.statistics)
    return merge_kernel_window_statistics(statistics)


def run_registered_kernel_calibration_cell(
    *,
    model: BlockStateModel | GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    artifact: Artifact,
    frequency: BaseFrequency,
    selected_length: int,
    historical_returns: npt.ArrayLike,
    historical_source_states: npt.ArrayLike,
    source_continuation_links: npt.ArrayLike,
    model_artifact_fingerprint: str,
    target_history_fingerprint: str,
    workers: int = 1,
    prefix_window_counts: Sequence[int] = CANONICAL_PREFIX_WINDOW_COUNTS,
    annualized_half_width_max: float = CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
    confidence: float = CANONICAL_CONFIDENCE,
    lambdas: Sequence[float] = CANONICAL_LAMBDAS,
    statistics_batch_windows: int = CANONICAL_STATISTICS_BATCH_WINDOWS,
    process_chunksize: int = DEFAULT_PROCESS_CHUNKSIZE,
    strict_canonical: bool = True,
    verify: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
    process_map: ProcessMap = deterministic_process_map,
) -> KernelCalibrationExecution:
    """Run the six-look prefix-stable calibration for one exact cell.

    A precision miss is returned as fail-closed evidence rather than raised;
    malformed inputs, missing batches, or verification disagreement raise
    :class:`ModelError`.  No offset grid is created unless a planned look
    passes the annualized simultaneous half-width gate.
    """

    if strict_canonical and process_map is not deterministic_process_map:
        raise ModelError(
            "strict canonical kernel calibration does not accept an injected "
            "process map"
        )
    _validate_execution_controls(
        workers=workers,
        prefix_window_counts=prefix_window_counts,
        annualized_half_width_max=annualized_half_width_max,
        confidence=confidence,
        lambdas=lambdas,
        statistics_batch_windows=statistics_batch_windows,
        process_chunksize=process_chunksize,
        strict_canonical=strict_canonical,
        verify=verify,
    )
    _validate_fingerprint(target_history_fingerprint, "target_history_fingerprint")
    context = _kernel_context(
        model=model,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        frequency=frequency,
        selected_length=selected_length,
        historical_returns=historical_returns,
        historical_source_states=historical_source_states,
        source_continuation_links=source_continuation_links,
        model_artifact_fingerprint=model_artifact_fingerprint,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    looks = tuple(int(value) for value in prefix_window_counts)
    lambda_values = tuple(float(value) for value in lambdas)
    batch_size = int(statistics_batch_windows)
    worker_count = int(workers)
    map_chunksize = int(process_chunksize)
    accumulated: list[RegisteredKernelWindowBatch] = []
    completed: list[KernelPrecisionLook] = []
    previous = 0
    final_precision: KernelMeanPrecision | None = None
    precision_passed = False

    for look_index, stop in enumerate(looks):
        outputs = _execute_window_range(
            context,
            start=previous,
            stop=stop,
            batch_windows=batch_size,
            workers=worker_count,
            process_chunksize=map_chunksize,
            process_map=process_map,
            verification_offsets=None,
        )
        accumulated.extend(output.batch for output in outputs)
        merged = merge_registered_kernel_batches(accumulated)
        if merged.windows != stop:
            raise ModelError("kernel prefix merge produced the wrong look size")
        final_precision = finalize_kernel_mean_precision(
            merged,
            periods_per_year=_periods_per_year(frequency),
            confidence=float(confidence),
        )
        maximum = float(np.max(final_precision.annualized_simultaneous_half_widths))
        passed = maximum <= float(annualized_half_width_max)
        completed.append(
            KernelPrecisionLook(
                look_index=look_index,
                windows=stop,
                precision=final_precision,
                maximum_annualized_half_width=maximum,
                passed=passed,
            )
        )
        previous = stop
        if passed:
            precision_passed = True
            break

    if final_precision is None:  # pragma: no cover - controls reject no looks
        raise AssertionError("kernel calibration completed no precision look")

    grid: AlphaOffsetGrid | None = None
    verification: KernelVerificationResult | None = None
    verification_passed = False
    failure_reason: str | None = None
    if precision_passed:
        target_mean = historical_target_mean(context.historical_returns)
        grid = alpha_offset_grid(
            final_precision,
            target_mean,
            np.asarray(lambda_values, dtype=np.float64),
            target_history_fingerprint=target_history_fingerprint,
        )
        if verify:
            verification = _verify_registered_kernel_calibration(
                context=context,
                precision=final_precision,
                offset_grid=grid,
                target_mean=target_mean,
                target_history_fingerprint=target_history_fingerprint,
                workers=worker_count,
                statistics_batch_windows=batch_size,
                process_chunksize=map_chunksize,
                process_map=process_map,
            )
            verification_passed = verification.passed
            if not verification_passed:
                failure_reason = "purpose_8_offset_verification_failed"
        else:
            failure_reason = "purpose_8_offset_verification_not_run"
    else:
        failure_reason = "precision_target_not_met_at_maximum_registered_look"

    passed = precision_passed and verification_passed
    result = KernelCalibrationExecution(
        artifact=artifact,
        frequency=frequency,
        geometry=context.geometry,
        n_states=context.model.n_states,
        selected_length=selected_length,
        model_artifact_fingerprint=model_artifact_fingerprint,
        target_history_fingerprint=target_history_fingerprint,
        calibration_stream_fingerprint=context.calibration_stream_fingerprint,
        draw_order=PURPOSE_7_DRAW_ORDER,
        draws_per_window=_draws_per_window(context.geometry),
        planned_looks=looks,
        completed_looks=tuple(completed),
        stopped_windows=completed[-1].windows,
        precision_passed=precision_passed,
        verification_passed=verification_passed,
        passed=passed,
        final_precision=final_precision,
        offset_grid=grid,
        verification=verification,
        failure_reason=failure_reason,
        purpose_7_streams=completed[-1].windows,
        purpose_8_streams=0,
        first_purpose_7_entity=calibration_entity(
            CalibrationPurpose.KERNEL_ESTIMATION,
            context.registered_artifact,
            0,
            0,
        ),
        last_purpose_7_entity=calibration_entity(
            CalibrationPurpose.KERNEL_ESTIMATION,
            context.registered_artifact,
            0,
            completed[-1].windows - 1,
        ),
        workers=worker_count,
        statistics_batch_windows=batch_size,
        process_chunksize=map_chunksize,
        strict_canonical=strict_canonical,
    )
    result = cast(KernelCalibrationExecution, _immutable_public_tree(result))
    parent_fingerprint = _kernel_parent_model_fingerprint(context.source_model)
    materialization_fingerprint = _kernel_source_materialization_fingerprint(
        value=result,
        parent_model_fingerprint=parent_fingerprint,
        historical_returns=context.historical_returns,
        historical_states=context.historical_source_states,
        continuation_links=context.source_continuation_links,
    )
    execution_fingerprint = _kernel_execution_fingerprint(result)
    for name, value in (
        ("parent_model_fingerprint", parent_fingerprint),
        ("source_materialization_fingerprint", materialization_fingerprint),
        ("rng_execution_fingerprint", context.calibration_stream_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("_source_model", context.source_model),
        ("_source_historical_returns", context.historical_returns),
        ("_source_historical_states", context.historical_source_states),
        ("_source_continuation_links", context.source_continuation_links),
        ("_master_seed", context.master_seed),
        ("_scientific_version", context.scientific_version),
        ("_execution_seal", _KERNEL_EXECUTION_CAPABILITY),
    ):
        object.__setattr__(result, name, value)
    _MINTED_KERNEL_EXECUTIONS[id(result)] = result
    # Exercise the same public verifier used at the G3 boundary before return.
    kernel_calibration_execution_identity(result)
    return result


def replay_registered_kernel_verification(
    calibration: KernelCalibrationExecution,
    *,
    model: BlockStateModel | GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    historical_returns: npt.ArrayLike,
    historical_source_states: npt.ArrayLike,
    source_continuation_links: npt.ArrayLike,
    workers: int = 1,
    process_chunksize: int = DEFAULT_PROCESS_CHUNKSIZE,
    process_map: ProcessMap = deterministic_process_map,
) -> KernelVerificationResult:
    """Independently replay an already-passed cell's streamless verification."""

    if not isinstance(calibration, KernelCalibrationExecution):
        raise ModelError("kernel verification requires a calibration execution")
    kernel_identity = kernel_calibration_execution_identity(calibration)
    if not calibration.precision_passed or calibration.offset_grid is None:
        raise ModelError("kernel verification requires a precision-passed offset grid")
    context = _kernel_context(
        model=model,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=calibration.artifact,
        frequency=calibration.frequency,
        selected_length=calibration.selected_length,
        historical_returns=historical_returns,
        historical_source_states=historical_source_states,
        source_continuation_links=source_continuation_links,
        model_artifact_fingerprint=calibration.model_artifact_fingerprint,
        strict_canonical=calibration.strict_canonical,
        monthly_segment_lengths=calibration.geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            calibration.geometry.target_segment_lengths
            if calibration.frequency == "daily"
            else None
        ),
    )
    if (
        kernel_identity.master_seed != master_seed
        or kernel_identity.scientific_version != scientific_version
        or context.calibration_stream_fingerprint
        != calibration.calibration_stream_fingerprint
    ):
        raise ModelError("kernel verification replay stream identity changed")
    target_mean = historical_target_mean(context.historical_returns)
    return _verify_registered_kernel_calibration(
        context=context,
        precision=calibration.final_precision,
        offset_grid=calibration.offset_grid,
        target_mean=target_mean,
        target_history_fingerprint=calibration.target_history_fingerprint,
        workers=_positive_integer(workers, "workers"),
        statistics_batch_windows=calibration.statistics_batch_windows,
        process_chunksize=_positive_integer(process_chunksize, "process_chunksize"),
        process_map=process_map,
    )


def _verify_registered_kernel_calibration(
    *,
    context: _KernelContext,
    precision: KernelMeanPrecision,
    offset_grid: AlphaOffsetGrid,
    target_mean: npt.ArrayLike,
    target_history_fingerprint: str,
    workers: int,
    statistics_batch_windows: int,
    process_chunksize: int,
    process_map: ProcessMap = deterministic_process_map,
) -> KernelVerificationResult:
    """Replay purpose-7 inputs and directly verify the purpose-8 offset cube.

    The function never constructs a purpose-8 RNG key.  The replay must
    reproduce the final unadjusted kernel fingerprint before any adjusted
    mean is accepted.
    """

    if not isinstance(context, _KernelContext):
        raise ModelError("kernel verification requires an execution context")
    _validate_fingerprint(target_history_fingerprint, "target_history_fingerprint")
    target = _asset_vector(target_mean, "target_mean")
    expected_grid = alpha_offset_grid(
        precision,
        target,
        offset_grid.lambdas,
        target_history_fingerprint=target_history_fingerprint,
    )
    if (
        offset_grid.grid_fingerprint != expected_grid.grid_fingerprint
        or offset_grid.model_artifact_fingerprint
        != expected_grid.model_artifact_fingerprint
        or offset_grid.kernel_sample_fingerprint
        != expected_grid.kernel_sample_fingerprint
        or not np.array_equal(offset_grid.lambdas, expected_grid.lambdas)
        or not np.array_equal(offset_grid.offsets, expected_grid.offsets)
    ):
        raise ModelError("kernel verification offset grid is inconsistent")
    if precision.windows <= 0 or precision.windows > CANONICAL_PREFIX_WINDOW_COUNTS[-1]:
        raise ModelError("kernel verification window count is outside purpose 7")
    offsets = np.asarray(offset_grid.offsets, dtype=np.float64).copy()
    offsets.setflags(write=False)
    outputs = _execute_window_range(
        context,
        start=0,
        stop=precision.windows,
        batch_windows=_positive_integer(
            statistics_batch_windows, "statistics_batch_windows"
        ),
        workers=_positive_integer(workers, "workers"),
        process_chunksize=_positive_integer(process_chunksize, "process_chunksize"),
        process_map=process_map,
        verification_offsets=offsets,
    )
    replay = merge_registered_kernel_batches([output.batch for output in outputs])
    replay_precision = finalize_kernel_mean_precision(
        replay,
        periods_per_year=precision.periods_per_year,
        confidence=precision.confidence,
    )
    if (
        replay_precision.kernel_sample_fingerprint
        != precision.kernel_sample_fingerprint
    ):
        raise ModelError("purpose-8 replay did not reproduce purpose-7 inputs")

    adjusted_sums = np.zeros(offsets.shape, dtype=np.float64)
    for output in outputs:
        if output.directly_adjusted_sums is None:
            raise ModelError("purpose-8 replay omitted direct adjusted sums")
        adjusted_sums += output.directly_adjusted_sums
    counts = replay_precision.state_observation_counts.astype(np.int64, copy=True)
    directly_adjusted = adjusted_sums / counts[None, :, None]
    algebraic_adjusted = replay_precision.means[None, :, :] - offsets
    residual = algebraic_adjusted - target[None, None, :]
    difference = directly_adjusted - algebraic_adjusted
    maximum_error = float(np.max(np.abs(difference)))
    annualized_error = maximum_error * float(precision.periods_per_year)
    passed = maximum_error <= VERIFICATION_ABSOLUTE_TOLERANCE
    for array in (counts, directly_adjusted, algebraic_adjusted, residual):
        array.setflags(write=False)
    return KernelVerificationResult(
        purpose=CalibrationPurpose.KERNEL_VERIFICATION,
        stream_count=0,
        windows=precision.windows,
        grid_fingerprint=offset_grid.grid_fingerprint,
        replay_kernel_sample_fingerprint=replay_precision.kernel_sample_fingerprint,
        observation_counts=counts,
        directly_adjusted_means=directly_adjusted,
        algebraic_adjusted_means=algebraic_adjusted,
        algebraic_residual_means=residual,
        maximum_absolute_mean_error=maximum_error,
        maximum_annualized_mean_error=annualized_error,
        tolerance=VERIFICATION_ABSOLUTE_TOLERANCE,
        passed=passed,
    )


def run_registered_kernel_calibration_suite(
    specs: Sequence[KernelCalibrationCellSpec],
    *,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
    prefix_window_counts: Sequence[int] = CANONICAL_PREFIX_WINDOW_COUNTS,
    annualized_half_width_max: float = CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
    confidence: float = CANONICAL_CONFIDENCE,
    lambdas: Sequence[float] = CANONICAL_LAMBDAS,
    statistics_batch_windows: int = CANONICAL_STATISTICS_BATCH_WINDOWS,
    process_chunksize: int = DEFAULT_PROCESS_CHUNKSIZE,
    strict_canonical: bool = True,
    process_map: ProcessMap = deterministic_process_map,
) -> KernelCalibrationSuite:
    """Run exactly the four design/production by monthly/daily cells."""

    supplied = tuple(specs)
    observed_order = tuple((item.artifact, item.frequency) for item in supplied)
    if observed_order != _CANONICAL_CELL_ORDER:
        raise ModelError(
            "kernel suite requires exact design-monthly, design-daily, "
            "production-monthly, production-daily order"
        )
    state_counts = tuple(_coerce_state_model(item.model).n_states for item in supplied)
    if strict_canonical and len(set(state_counts)) != 1:
        raise ModelError(
            "strict canonical kernel suite requires the same K in all cells"
        )
    if strict_canonical:
        for left, right in ((0, 1), (2, 3)):
            left_model = _coerce_state_model(supplied[left].model)
            right_model = _coerce_state_model(supplied[right].model)
            if (
                supplied[left].model_artifact_fingerprint
                != supplied[right].model_artifact_fingerprint
                or not np.array_equal(
                    left_model.stationary_distribution,
                    right_model.stationary_distribution,
                )
                or not np.array_equal(
                    left_model.transition_matrix,
                    right_model.transition_matrix,
                )
            ):
                raise ModelError(
                    "strict canonical monthly/daily cells must share one artifact model"
                )
            monthly_geometry = resolve_calibration_geometry(
                artifact=supplied[left].artifact,
                frequency="monthly",
                strict_canonical=True,
                monthly_segment_lengths=supplied[left].monthly_segment_lengths,
            )
            daily_geometry = resolve_calibration_geometry(
                artifact=supplied[right].artifact,
                frequency="daily",
                strict_canonical=True,
                monthly_segment_lengths=supplied[right].monthly_segment_lengths,
                daily_segment_lengths=supplied[right].daily_segment_lengths,
            )
            expanded = expand_monthly_states_to_daily(
                supplied[left].historical_source_states,
                monthly_segment_lengths=monthly_geometry.monthly_segment_lengths,
                daily_segment_lengths=daily_geometry.target_segment_lengths,
            )
            daily_states = np.asarray(supplied[right].historical_source_states)
            if not np.array_equal(expanded.states, daily_states):
                raise ModelError(
                    "strict canonical daily source states must be the exact "
                    "21-session expansion of monthly source states"
                )
    results = tuple(
        run_registered_kernel_calibration_cell(
            model=spec.model,
            master_seed=master_seed,
            scientific_version=scientific_version,
            artifact=spec.artifact,
            frequency=spec.frequency,
            selected_length=spec.selected_length,
            historical_returns=spec.historical_returns,
            historical_source_states=spec.historical_source_states,
            source_continuation_links=spec.source_continuation_links,
            model_artifact_fingerprint=spec.model_artifact_fingerprint,
            target_history_fingerprint=spec.target_history_fingerprint,
            workers=workers,
            prefix_window_counts=prefix_window_counts,
            annualized_half_width_max=annualized_half_width_max,
            confidence=confidence,
            lambdas=lambdas,
            statistics_batch_windows=statistics_batch_windows,
            process_chunksize=process_chunksize,
            strict_canonical=strict_canonical,
            verify=True,
            monthly_segment_lengths=spec.monthly_segment_lengths,
            daily_segment_lengths=spec.daily_segment_lengths,
            process_map=process_map,
        )
        for spec in supplied
    )
    all_precision = all(item.precision_passed for item in results)
    all_verification = all(item.verification_passed for item in results)
    return KernelCalibrationSuite(
        cells=results,
        all_precision_passed=all_precision,
        all_verification_passed=all_verification,
        passed=all_precision and all_verification,
    )


def _execute_window_range(
    context: _KernelContext,
    *,
    start: int,
    stop: int,
    batch_windows: int,
    workers: int,
    process_chunksize: int,
    process_map: ProcessMap,
    verification_offsets: FloatArray | None,
) -> tuple[_KernelWorkerOutput, ...]:
    ranges = tuple(
        _BatchRange(first, min(first + batch_windows, stop))
        for first in range(start, stop, batch_windows)
    )
    if not ranges:
        raise ModelError("kernel execution range must contain at least one window")
    group_count = min(workers, len(ranges))
    ranges_per_group = math.ceil(len(ranges) / group_count)
    jobs = tuple(
        _KernelWorkerJob(
            context=context,
            ranges=ranges[index : index + ranges_per_group],
            verification_offsets=verification_offsets,
        )
        for index in range(0, len(ranges), ranges_per_group)
    )
    mapped = process_map(
        _run_kernel_worker,
        jobs,
        workers=workers,
        chunksize=process_chunksize,
    )
    flattened: list[_KernelWorkerOutput] = []
    try:
        for group in mapped:
            if not isinstance(group, tuple):
                raise ModelError("kernel process map returned a malformed worker group")
            for output in group:
                if not isinstance(output, _KernelWorkerOutput):
                    raise ModelError("kernel process map returned a malformed batch")
                flattened.append(output)
    except TypeError as error:
        raise ModelError("kernel process map returned malformed results") from error
    ordered = tuple(sorted(flattened, key=lambda item: item.batch.first_window_index))
    expected = ranges
    observed = tuple(
        _BatchRange(item.batch.first_window_index, item.batch.stop_window_index)
        for item in ordered
    )
    if observed != expected:
        raise ModelError("kernel process results are missing, duplicated, or misranged")
    return ordered


def _run_kernel_worker(job: _KernelWorkerJob) -> tuple[_KernelWorkerOutput, ...]:
    return tuple(
        _generate_batch(job.context, item, job.verification_offsets)
        for item in job.ranges
    )


def _generate_batch(
    context: _KernelContext,
    batch_range: _BatchRange,
    verification_offsets: FloatArray | None,
) -> _KernelWorkerOutput:
    windows = batch_range.stop - batch_range.start
    if windows <= 0:
        raise ModelError("kernel batch range must be positive")
    geometry = context.geometry
    monthly = geometry.monthly_observations
    target = geometry.target_observations
    draws_per_window = _draws_per_window(geometry)
    state_uniforms = np.empty((windows, monthly), dtype=np.float64)
    restart_uniforms = np.empty((windows, target), dtype=np.float64)
    rank_uniforms = np.empty((windows, target), dtype=np.float64)
    for row, replicate in enumerate(range(batch_range.start, batch_range.stop)):
        key = calibration_rng_key(
            master_seed=context.master_seed,
            scientific_version=context.scientific_version,
            purpose=CalibrationPurpose.KERNEL_ESTIMATION,
            artifact=context.registered_artifact,
            variant=0,
            replicate=replicate,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=context.family,
            stage=Stage.REFERENCE_RESAMPLING,
        )
        draws = key.generator().random(draws_per_window)
        state_uniforms[row] = draws[:monthly]
        restart_uniforms[row] = draws[monthly : monthly + target]
        rank_uniforms[row] = draws[monthly + target :]
    for array in (state_uniforms, restart_uniforms, rank_uniforms):
        array.setflags(write=False)
    traces = iter_paired_trace_chunks(
        model=context.model,
        geometry=geometry,
        historical_source_states=context.historical_source_states,
        source_continuation_links=context.source_continuation_links,
        intended_lengths=(context.selected_length,),
        uniform_chunks=(
            RegisteredUniformChunk(
                history_start=0,
                monthly_state_uniforms=state_uniforms,
                restart_uniforms=restart_uniforms,
                source_rank_uniforms=rank_uniforms,
            ),
        ),
        expected_history_count=windows,
    )
    trace = next(traces)
    try:
        next(traces)
    except StopIteration:
        pass
    else:  # pragma: no cover - one range and one L are structurally singular
        raise AssertionError("kernel trace generator returned extra chunks")
    sampled_returns = np.asarray(
        context.historical_returns[trace.source_indices], dtype=np.float64
    )
    statistics = kernel_window_statistics(
        sampled_returns,
        trace.target_states,
        n_states=context.model.n_states,
        first_window_index=batch_range.start,
        frequency=context.frequency,
        model_artifact_fingerprint=context.model_artifact_fingerprint,
        calibration_stream_fingerprint=context.calibration_stream_fingerprint,
    )
    first_entity = calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION,
        context.registered_artifact,
        0,
        batch_range.start,
    )
    last_entity = calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION,
        context.registered_artifact,
        0,
        batch_range.stop - 1,
    )
    batch = RegisteredKernelWindowBatch(
        artifact=context.artifact,
        frequency=context.frequency,
        first_window_index=batch_range.start,
        stop_window_index=batch_range.stop,
        first_stream_entity=first_entity,
        last_stream_entity=last_entity,
        draws_per_window=draws_per_window,
        draw_order=PURPOSE_7_DRAW_ORDER,
        statistics=statistics,
    )
    directly_adjusted: FloatArray | None = None
    if verification_offsets is not None:
        expected_shape = (
            len(CANONICAL_LAMBDAS),
            context.model.n_states,
            3,
        )
        if verification_offsets.shape != expected_shape:
            raise ModelError("purpose-8 offsets have the wrong 4 by K by 3 shape")
        directly_adjusted = np.zeros(expected_shape, dtype=np.float64)
        for lambda_index in range(expected_shape[0]):
            for state in range(context.model.n_states):
                selected = sampled_returns[trace.target_states == state]
                directly_adjusted[lambda_index, state] = np.sum(
                    selected - verification_offsets[lambda_index, state],
                    axis=0,
                    dtype=np.float64,
                )
        directly_adjusted.setflags(write=False)
    return _KernelWorkerOutput(batch, directly_adjusted)


def _kernel_context(
    *,
    model: BlockStateModel | GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    artifact: Artifact,
    frequency: BaseFrequency,
    selected_length: int,
    historical_returns: npt.ArrayLike,
    historical_source_states: npt.ArrayLike,
    source_continuation_links: npt.ArrayLike,
    model_artifact_fingerprint: str,
    strict_canonical: bool,
    monthly_segment_lengths: Sequence[int] | None,
    daily_segment_lengths: Sequence[int] | None,
) -> _KernelContext:
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if not isinstance(model, BlockStateModel | GaussianHMMFit):
        raise ModelError(
            "kernel calibration requires a fitted HMM or block state model"
        )
    source_model = model
    state_model = _coerce_state_model(source_model)
    if strict_canonical and state_model.n_states not in {2, 3, 4, 5}:
        raise ModelError("strict canonical kernel calibration requires K in 2..5")
    _validate_fingerprint(model_artifact_fingerprint, "model_artifact_fingerprint")
    geometry = resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    length = _positive_integer(selected_length, "selected_length")
    lower, upper = (
        MONTHLY_BLOCK_LENGTH_BOUNDS
        if frequency == "monthly"
        else DAILY_BLOCK_LENGTH_BOUNDS
    )
    if not lower <= length <= upper:
        raise ModelError("selected kernel block length is outside its hard bounds")
    returns = _immutable_bytes_array(
        _return_matrix(historical_returns),
        dtype=np.dtype("<f8"),
        label="kernel historical returns",
    )
    raw_states = np.asarray(historical_source_states)
    if (
        raw_states.ndim != 1
        or raw_states.size != returns.shape[0]
        or raw_states.dtype.kind not in {"i", "u"}
    ):
        raise ModelError("kernel source states must be integer and match returns")
    states = raw_states.astype(np.int64, copy=True)
    if bool(((states < 0) | (states >= state_model.n_states)).any()):
        raise ModelError("kernel source states contain an invalid state")
    missing = [
        state
        for state in range(state_model.n_states)
        if not bool(np.any(states == state))
    ]
    if missing:
        raise ModelError("kernel source states omit fitted states")
    raw_links = np.asarray(source_continuation_links)
    if (
        raw_links.ndim != 1
        or raw_links.size != returns.shape[0] - 1
        or raw_links.dtype.kind != "b"
    ):
        raise ModelError("kernel source links must be Boolean with n-1 rows")
    links = raw_links.astype(np.bool_, copy=True)
    if strict_canonical:
        if returns.shape[0] != geometry.target_observations:
            raise ModelError("strict canonical kernel source has the wrong row count")
        if not np.array_equal(links, geometry.target_continuation_links):
            raise ModelError("strict canonical kernel source has the wrong segments")
        # ``model_artifact_fingerprint`` is the pre-kernel scientific-core
        # identity.  It intentionally includes decoded return pools and PC1
        # references in addition to the fitted state model, so it cannot equal
        # the narrower live-parent fingerprint revalidated below.  The final
        # scientific-artifact builder recomputes and matches this core identity.
    registered_artifact = _registered_artifact(artifact)
    family = _registered_family(frequency)
    stream_fingerprint = _stream_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=registered_artifact,
        family=family,
        geometry=geometry,
        selected_length=length,
    )
    # Validate both ends of the complete registered prefix before any worker starts.
    for replicate in (0, CANONICAL_PREFIX_WINDOW_COUNTS[-1] - 1):
        calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.KERNEL_ESTIMATION,
            artifact=registered_artifact,
            variant=0,
            replicate=replicate,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=family,
            stage=Stage.REFERENCE_RESAMPLING,
        )
    states = cast(
        IntArray,
        _immutable_bytes_array(
            states,
            dtype=np.dtype("<i8"),
            label="kernel historical states",
        ),
    )
    links = cast(
        BoolArray,
        _immutable_bytes_array(
            links,
            dtype=np.dtype("?"),
            label="kernel continuation links",
        ),
    )
    return _KernelContext(
        model=state_model,
        source_model=source_model,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        registered_artifact=registered_artifact,
        frequency=frequency,
        family=family,
        geometry=geometry,
        selected_length=length,
        historical_returns=returns,
        historical_source_states=states,
        source_continuation_links=links,
        model_artifact_fingerprint=model_artifact_fingerprint,
        calibration_stream_fingerprint=stream_fingerprint,
    )


def _validate_execution_controls(
    *,
    workers: int,
    prefix_window_counts: Sequence[int],
    annualized_half_width_max: float,
    confidence: float,
    lambdas: Sequence[float],
    statistics_batch_windows: int,
    process_chunksize: int,
    strict_canonical: bool,
    verify: bool,
) -> None:
    _positive_integer(workers, "workers")
    _positive_integer(statistics_batch_windows, "statistics_batch_windows")
    _positive_integer(process_chunksize, "process_chunksize")
    if not isinstance(strict_canonical, bool) or not isinstance(verify, bool):
        raise ModelError("strict_canonical and verify must be boolean")
    looks = _positive_increasing_integers(prefix_window_counts, "prefix_window_counts")
    if looks[-1] > CANONICAL_PREFIX_WINDOW_COUNTS[-1]:
        raise ModelError("kernel looks exceed the registered purpose-7 range")
    if isinstance(annualized_half_width_max, bool) or not isinstance(
        annualized_half_width_max, float | np.floating
    ):
        raise ModelError("annualized_half_width_max must be a positive float")
    if not np.isfinite(annualized_half_width_max) or annualized_half_width_max <= 0.0:
        raise ModelError("annualized_half_width_max must be a positive float")
    if isinstance(confidence, bool) or not isinstance(confidence, float | np.floating):
        raise ModelError("confidence must be a float in (0, 1)")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ModelError("confidence must be a float in (0, 1)")
    lambda_values = _lambda_tuple(lambdas)
    if lambda_values != CANONICAL_LAMBDAS:
        raise ModelError(
            "kernel calibration requires exactly lambdas 0.25, 0.50, 0.75, 1.00"
        )
    if strict_canonical and (
        looks != CANONICAL_PREFIX_WINDOW_COUNTS
        or float(annualized_half_width_max) != CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
        or float(confidence) != CANONICAL_CONFIDENCE
        or int(statistics_batch_windows) != CANONICAL_STATISTICS_BATCH_WINDOWS
        or not verify
    ):
        raise ModelError("strict canonical kernel controls do not match registration")


def _validated_batch(
    batch: RegisteredKernelWindowBatch,
) -> RegisteredKernelWindowBatch:
    if not isinstance(batch, RegisteredKernelWindowBatch):
        raise ModelError("registered kernel batch has the wrong type")
    first = _nonnegative_integer(batch.first_window_index, "first_window_index")
    stop = _positive_integer(batch.stop_window_index, "stop_window_index")
    if stop <= first or batch.statistics.first_window_index != first:
        raise ModelError("registered kernel batch has an invalid window range")
    if (
        batch.statistics.stop_window_index != stop
        or batch.statistics.windows != stop - first
    ):
        raise ModelError("registered kernel batch statistics range is inconsistent")
    registered_artifact = _registered_artifact(batch.artifact)
    expected_first = calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION, registered_artifact, 0, first
    )
    expected_last = calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION, registered_artifact, 0, stop - 1
    )
    if (
        batch.first_stream_entity != expected_first
        or batch.last_stream_entity != expected_last
    ):
        raise ModelError("registered kernel batch stream entities are inconsistent")
    if batch.draw_order != PURPOSE_7_DRAW_ORDER or batch.draws_per_window <= 0:
        raise ModelError("registered kernel batch draw contract is inconsistent")
    if batch.frequency != batch.statistics.frequency:
        raise ModelError("registered kernel batch frequency is inconsistent")
    return batch


def _stream_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    family: Family,
    geometry: CalibrationGeometry,
    selected_length: int,
) -> str:
    values = (
        "prpg-purpose-7-kernel-estimation-stream-v1",
        str(RNG_CONTRACT_VERSION),
        str(master_seed),
        str(scientific_version),
        str(int(Domain.BLOCK_ALPHA_CALIBRATION)),
        str(int(family)),
        str(int(CalibrationPurpose.KERNEL_ESTIMATION)),
        str(int(artifact)),
        "0",
        str(int(Stage.REFERENCE_RESAMPLING)),
        repr(geometry.monthly_segment_lengths),
        repr(geometry.target_segment_lengths),
        str(selected_length),
        *PURPOSE_7_DRAW_ORDER,
    )
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _kernel_parent_model_fingerprint(
    model: BlockStateModel | GaussianHMMFit,
) -> str:
    if isinstance(model, GaussianHMMFit):
        return gaussian_hmm_fit_fingerprint(model)
    if not isinstance(model, BlockStateModel):
        raise ModelError("kernel calibration retained an invalid parent model")
    return scientific_materialization_fingerprint(
        # Match the registered block executor's parent identity exactly so a
        # state-model source cannot drift merely by crossing task boundaries.
        schema_id="registered_block_state_model",
        metadata={"n_states": model.n_states},
        arrays={
            "stationary_distribution": model.stationary_distribution,
            "transition_matrix": model.transition_matrix,
        },
    )


def _kernel_source_materialization_fingerprint(
    *,
    value: KernelCalibrationExecution,
    parent_model_fingerprint: str,
    historical_returns: FloatArray,
    historical_states: IntArray,
    continuation_links: BoolArray,
) -> str:
    _validate_fingerprint(parent_model_fingerprint, "parent_model_fingerprint")
    return scientific_materialization_fingerprint(
        schema_id="kernel_calibration_source_materialization",
        metadata={
            "artifact": value.artifact,
            "frequency": value.frequency,
            "geometry": {
                "artifact": value.geometry.artifact,
                "frequency": value.geometry.frequency,
                "monthly_segment_lengths": list(value.geometry.monthly_segment_lengths),
                "strict_canonical": value.geometry.strict_canonical,
                "target_segment_lengths": list(value.geometry.target_segment_lengths),
            },
            "model_artifact_fingerprint": value.model_artifact_fingerprint,
            "n_states": value.n_states,
            "parent_model_fingerprint": parent_model_fingerprint,
            "selected_length": value.selected_length,
            "target_history_fingerprint": value.target_history_fingerprint,
        },
        arrays={
            "continuation_links": continuation_links,
            "historical_returns": historical_returns,
            "historical_states": historical_states,
        },
    )


def _kernel_execution_fingerprint(value: KernelCalibrationExecution) -> str:
    normalized = _normalize_scientific_content(
        value,
        excluded_fields=(
            _KERNEL_IDENTITY_FIELDS
            | {
                "workers",
                "statistics_batch_windows",
                "process_chunksize",
            }
        ),
    )
    return hashlib.sha256(
        (
            json.dumps(
                {
                    "schema_id": "prpg-kernel-calibration-execution-v1",
                    "result": normalized,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _normalize_scientific_content(
    value: object,
    *,
    excluded_fields: frozenset[str] | set[str] = frozenset(),
) -> object:
    if isinstance(value, np.ndarray):
        return {
            "array_fingerprint": scientific_array_fingerprint(
                value,
                role="kernel_execution_array",
            ),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _normalize_scientific_content(value.item())
    if isinstance(value, Enum):
        return _normalize_scientific_content(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _normalize_scientific_content(
                    getattr(value, item.name),
                    excluded_fields=excluded_fields,
                )
                for item in fields(value)
                if not item.name.startswith("_") and item.name not in excluded_fields
            },
        }
    if isinstance(value, tuple | list):
        return [
            _normalize_scientific_content(item, excluded_fields=excluded_fields)
            for item in value
        ]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("kernel execution contains a non-finite scalar")
        return value
    raise ModelError(
        "kernel execution contains an unsupported identity value",
        details={"type": type(value).__name__},
    )


def _immutable_public_tree(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _immutable_bytes_array(
            value,
            dtype=value.dtype,
            label="kernel execution output",
        )
    if isinstance(value, tuple):
        return tuple(_immutable_public_tree(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: _immutable_public_tree(getattr(value, item.name))
            for item in fields(value)
            if item.init and not item.name.startswith("_")
        }
        return replace(value, **updates)
    return value


def _require_immutable_public_arrays(value: object) -> None:
    if isinstance(value, np.ndarray):
        _require_immutable_bytes_array(value, "kernel execution output")
        return
    if isinstance(value, tuple | list):
        for item in value:
            _require_immutable_public_arrays(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if not item.name.startswith("_"):
                _require_immutable_public_arrays(getattr(value, item.name))


def _immutable_bytes_array(
    value: npt.ArrayLike,
    *,
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    source = np.asarray(value)
    canonical = np.ascontiguousarray(source.astype(dtype, copy=False))
    immutable = np.frombuffer(canonical.tobytes(order="C"), dtype=dtype).reshape(
        canonical.shape
    )
    _require_immutable_bytes_array(immutable, label)
    return immutable


def _require_immutable_bytes_array(value: np.ndarray, label: str) -> None:
    array = np.asarray(value)
    if array.flags.writeable or not array.flags.c_contiguous:
        raise ModelError(f"{label} must be a read-only contiguous array")
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        raise ModelError(f"{label} must be backed by immutable bytes")


def _coerce_state_model(
    model: BlockStateModel | GaussianHMMFit,
) -> BlockStateModel:
    if isinstance(model, BlockStateModel):
        return model
    if isinstance(model, GaussianHMMFit):
        return BlockStateModel.from_fitted_hmm(model)
    raise ModelError("kernel calibration requires a fitted HMM or block state model")


def _registered_artifact(artifact: Artifact) -> CalibrationArtifact:
    if artifact == "design":
        return CalibrationArtifact.DESIGN_OR_CONTROL
    if artifact == "production":
        return CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    raise ModelError("kernel artifact must be design or production")


def _registered_family(frequency: BaseFrequency) -> Family:
    if frequency == "monthly":
        return Family.LF
    if frequency == "daily":
        return Family.HF
    raise ModelError("kernel frequency must be monthly or daily")


def _draws_per_window(geometry: CalibrationGeometry) -> int:
    return geometry.monthly_observations + 2 * geometry.target_observations


def _periods_per_year(frequency: BaseFrequency) -> int:
    return 12 if frequency == "monthly" else 252


def _return_matrix(values: npt.ArrayLike) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("historical kernel returns must be numeric") from error
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3:
        raise ModelError("historical kernel returns must have shape (n, 3)")
    if not np.isfinite(result).all():
        raise ModelError("historical kernel returns must be finite")
    return result.copy()


def _asset_vector(values: npt.ArrayLike, label: str) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ModelError(f"{label} must contain three finite values")
    return result.copy()


def _lambda_tuple(values: Sequence[float]) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ModelError("lambda grid must be numeric") from error
    if (
        not result
        or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in result)
        or any(right <= left for left, right in pairwise(result))
    ):
        raise ModelError("lambda grid must be strictly increasing in (0, 1]")
    return result


def _positive_increasing_integers(values: Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(values, str | bytes):
        raise ModelError(f"{label} must contain positive increasing integers")
    raw = tuple(values)
    if not raw:
        raise ModelError(f"{label} must contain positive increasing integers")
    result = tuple(_positive_integer(value, label) for value in raw)
    if any(right <= left for left, right in pairwise(result)):
        raise ModelError(f"{label} must contain positive increasing integers")
    return result


def _validate_fingerprint(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelError(f"{label} must be a lowercase SHA-256 digest")


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ModelError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return result


def _nonnegative_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ModelError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ModelError(f"{label} must be a non-negative integer")
    return result
