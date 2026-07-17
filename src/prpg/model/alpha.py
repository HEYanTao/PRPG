"""Approval-independent alpha-policy calibration primitives.

The project design declares an identity historical-vector policy and a
one-shot, state-specific kernel-mean translation.  This module implements only
that deterministic arithmetic and mergeable sufficient statistics.  It does
not simulate calibration paths, choose a production policy, or embed any
unapproved G3/G5 threshold.

All vectors use the fixed semantic order ``(equity, muni_bond,
taxable_bond)``.  Callers remain responsible for keeping monthly and daily
statistics in separate artifacts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from prpg.errors import ModelError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BaseFrequency: TypeAlias = Literal["monthly", "daily"]

ASSET_COUNT = 3
KERNEL_CONTROLLED_FREQUENCIES = 2
KERNEL_PREFIX_LOOKS = 6


class AlphaPolicy(StrEnum):
    """Version-1 candidate return transformations."""

    HISTORICAL_VECTOR = "historical_vector"
    KERNEL_MEAN_NEUTRAL = "kernel_mean_neutral"


@dataclass(frozen=True, slots=True)
class KernelMeanStatistics:
    """Mergeable per-state vector statistics using Chan's stable formula."""

    counts: IntArray
    means: FloatArray
    centered_sum_squares: FloatArray
    observations: int


@dataclass(frozen=True, slots=True)
class KernelMeanEstimate:
    """Final per-state kernel means and per-vector descriptive diagnostics."""

    counts: IntArray
    means: FloatArray
    sample_variances: FloatArray
    independent_vector_standard_errors: FloatArray


@dataclass(frozen=True, slots=True)
class KernelWindowStatistics:
    """Mergeable cluster sufficient statistics with one cluster per window."""

    frequency: BaseFrequency
    model_artifact_fingerprint: str
    calibration_stream_fingerprint: str
    windows: int
    first_window_index: int
    stop_window_index: int
    source_window_fingerprint: str
    window_fingerprint: str
    state_window_counts: IntArray
    state_observation_counts: IntArray
    state_return_sums: FloatArray
    cluster_return_sum_squares: FloatArray
    cluster_count_return_products: FloatArray
    cluster_count_squares: FloatArray


@dataclass(frozen=True, slots=True)
class KernelMeanPrecision:
    """Window-clustered kernel means and simultaneous confidence half-widths."""

    frequency: BaseFrequency
    model_artifact_fingerprint: str
    calibration_stream_fingerprint: str
    source_window_fingerprint: str
    window_statistics_fingerprint: str
    kernel_sample_fingerprint: str
    sample_stage: Literal["unadjusted_kernel"]
    windows: int
    periods_per_year: int
    confidence: float
    degrees_freedom: int
    critical_value: float
    simultaneous_comparisons: int
    state_window_counts: IntArray
    state_observation_counts: IntArray
    means: FloatArray
    cluster_standard_errors: FloatArray
    annualized_means: FloatArray
    annualized_simultaneous_half_widths: FloatArray


@dataclass(frozen=True, slots=True)
class AlphaOffsetGrid:
    """One-shot state/asset translations for an ordered lambda grid."""

    frequency: BaseFrequency
    model_artifact_fingerprint: str
    kernel_sample_fingerprint: str
    target_history_fingerprint: str
    grid_fingerprint: str
    lambdas: FloatArray
    offsets: FloatArray


@dataclass(frozen=True, slots=True)
class SelectedAlphaOffsets:
    """The smallest externally qualified offset with bound provenance."""

    frequency: BaseFrequency
    model_artifact_fingerprint: str
    grid_fingerprint: str
    lambda_value: float
    state_offsets: FloatArray
    selection_fingerprint: str


@dataclass(frozen=True, slots=True)
class ReturnVectorBatch:
    """Synchronized vectors bound to frequency, artifact, and policy stage."""

    values: FloatArray
    target_states: IntArray
    frequency: BaseFrequency
    model_artifact_fingerprint: str
    source_batch_fingerprint: str
    batch_fingerprint: str
    applied_policy: AlphaPolicy | None
    policy_parameter_fingerprint: str | None


def kernel_mean_statistics(
    sampled_returns: ArrayLike,
    target_states: ArrayLike,
    *,
    n_states: int,
) -> KernelMeanStatistics:
    """Summarize unadjusted kernel output without retaining every vector.

    A batch may omit a state so independently generated batches can be merged.
    Empty-state rows use canonical zero means and zero centered sums; finalizing
    an estimate still requires at least two observations in every state.
    """

    state_count = _positive_integer(n_states, "n_states")
    returns = _return_matrix(sampled_returns, label="sampled returns")
    states = _state_vector(target_states, expected_size=returns.shape[0])
    if states.size and int(np.max(states)) >= state_count:
        raise ModelError(
            "target state is outside the declared state count",
            details={"maximum_state": int(np.max(states)), "n_states": state_count},
        )

    counts = np.bincount(states, minlength=state_count).astype(np.int64, copy=False)
    means = np.zeros((state_count, ASSET_COUNT), dtype=np.float64)
    centered = np.zeros_like(means)
    for state in range(state_count):
        selected = returns[states == state]
        if selected.size == 0:
            continue
        state_mean = np.mean(selected, axis=0, dtype=np.float64)
        means[state] = state_mean
        residuals = selected - state_mean
        centered[state] = np.sum(residuals * residuals, axis=0, dtype=np.float64)

    _make_read_only(counts, means, centered)
    return KernelMeanStatistics(
        counts=counts,
        means=means,
        centered_sum_squares=centered,
        observations=int(returns.shape[0]),
    )


def merge_kernel_mean_statistics(
    statistics: Sequence[KernelMeanStatistics],
) -> KernelMeanStatistics:
    """Merge batch statistics deterministically in caller-supplied order."""

    if not statistics:
        raise ModelError("at least one kernel-mean statistics batch is required")
    validated = tuple(_validated_statistics(item) for item in statistics)
    n_states = validated[0].counts.size
    if any(item.counts.size != n_states for item in validated[1:]):
        raise ModelError("kernel-mean batches use different state counts")

    counts = np.zeros(n_states, dtype=np.int64)
    means = np.zeros((n_states, ASSET_COUNT), dtype=np.float64)
    centered = np.zeros_like(means)
    observations = 0

    for batch in validated:
        for state in range(n_states):
            left = int(counts[state])
            right = int(batch.counts[state])
            if right == 0:
                continue
            if left == 0:
                counts[state] = right
                means[state] = batch.means[state]
                centered[state] = batch.centered_sum_squares[state]
                continue
            total = left + right
            delta = batch.means[state] - means[state]
            centered[state] += batch.centered_sum_squares[state] + delta * delta * (
                float(left) * float(right) / float(total)
            )
            means[state] += delta * (float(right) / float(total))
            counts[state] = total
        observations += batch.observations

    if observations != int(np.sum(counts, dtype=np.int64)):
        raise ModelError("merged kernel-mean observation count is inconsistent")
    _make_read_only(counts, means, centered)
    return KernelMeanStatistics(
        counts=counts,
        means=means,
        centered_sum_squares=centered,
        observations=observations,
    )


def finalize_kernel_means(statistics: KernelMeanStatistics) -> KernelMeanEstimate:
    """Create per-state estimates after enforcing usable Monte Carlo support."""

    item = _validated_statistics(statistics)
    sparse = np.flatnonzero(item.counts < 2)
    if sparse.size:
        raise ModelError(
            "every state needs at least two kernel observations",
            details={
                "states": sparse.astype(int).tolist(),
                "counts": item.counts[sparse].astype(int).tolist(),
            },
        )
    denominator = (item.counts - 1).astype(np.float64)[:, np.newaxis]
    variances = item.centered_sum_squares / denominator
    variances = np.maximum(variances, 0.0)
    standard_errors = np.sqrt(variances / item.counts.astype(np.float64)[:, np.newaxis])
    means = item.means.copy()
    counts = item.counts.copy()
    _make_read_only(counts, means, variances, standard_errors)
    return KernelMeanEstimate(
        counts=counts,
        means=means,
        sample_variances=variances,
        independent_vector_standard_errors=standard_errors,
    )


def kernel_window_statistics(
    sampled_return_windows: ArrayLike,
    target_state_windows: ArrayLike,
    *,
    n_states: int,
    first_window_index: int,
    frequency: BaseFrequency,
    model_artifact_fingerprint: str,
    calibration_stream_fingerprint: str,
) -> KernelWindowStatistics:
    """Summarize independent history-length windows for clustered inference.

    Returns have shape ``(windows, periods, 3)`` and states have shape
    ``(windows, periods)``. A window contributes a state-specific return sum
    and count; the retained cross-products are sufficient for the clustered
    ratio-of-sums variance without treating serial periods as independent.
    """

    state_count = _positive_integer(n_states, "n_states")
    first = _nonnegative_integer(first_window_index, "first_window_index")
    checked_frequency = _frequency(frequency)
    _validate_fingerprint(model_artifact_fingerprint, "model_artifact_fingerprint")
    _validate_fingerprint(
        calibration_stream_fingerprint, "calibration_stream_fingerprint"
    )
    returns = _return_cube(sampled_return_windows, label="sampled return windows")
    states = _state_matrix(target_state_windows, expected_shape=returns.shape[:2])
    if int(np.max(states)) >= state_count:
        raise ModelError(
            "target state is outside the declared state count",
            details={"maximum_state": int(np.max(states)), "n_states": state_count},
        )

    window_presence = np.zeros(state_count, dtype=np.int64)
    observation_counts = np.zeros(state_count, dtype=np.int64)
    return_sums = np.zeros((state_count, ASSET_COUNT), dtype=np.float64)
    return_sum_squares = np.zeros_like(return_sums)
    count_return_products = np.zeros_like(return_sums)
    count_squares = np.zeros(state_count, dtype=np.float64)

    for window_returns, window_states in zip(returns, states, strict=True):
        for state in range(state_count):
            selected = window_returns[window_states == state]
            count = int(selected.shape[0])
            if count == 0:
                continue
            vector_sum = np.asarray(
                np.sum(selected, axis=0, dtype=np.float64), dtype=np.float64
            )
            window_presence[state] += 1
            observation_counts[state] += count
            return_sums[state] += vector_sum
            return_sum_squares[state] += vector_sum * vector_sum
            count_return_products[state] += float(count) * vector_sum
            count_squares[state] += float(count * count)

    _make_read_only(
        window_presence,
        observation_counts,
        return_sums,
        return_sum_squares,
        count_return_products,
        count_squares,
    )
    stop = first + int(returns.shape[0])
    source_fingerprint = _window_source_fingerprint(
        first,
        checked_frequency,
        model_artifact_fingerprint,
        calibration_stream_fingerprint,
        returns,
        states,
    )
    window_fingerprint = _window_statistics_fingerprint(
        frequency=checked_frequency,
        model_artifact_fingerprint=model_artifact_fingerprint,
        calibration_stream_fingerprint=calibration_stream_fingerprint,
        first_window_index=first,
        stop_window_index=stop,
        source_window_fingerprint=source_fingerprint,
        arrays=(
            window_presence,
            observation_counts,
            return_sums,
            return_sum_squares,
            count_return_products,
            count_squares,
        ),
    )
    return KernelWindowStatistics(
        frequency=checked_frequency,
        model_artifact_fingerprint=model_artifact_fingerprint,
        calibration_stream_fingerprint=calibration_stream_fingerprint,
        windows=int(returns.shape[0]),
        first_window_index=first,
        stop_window_index=stop,
        source_window_fingerprint=source_fingerprint,
        window_fingerprint=window_fingerprint,
        state_window_counts=window_presence,
        state_observation_counts=observation_counts,
        state_return_sums=return_sums,
        cluster_return_sum_squares=return_sum_squares,
        cluster_count_return_products=count_return_products,
        cluster_count_squares=count_squares,
    )


def merge_kernel_window_statistics(
    statistics: Sequence[KernelWindowStatistics],
) -> KernelWindowStatistics:
    """Merge independent window batches without changing cluster identity."""

    if not statistics:
        raise ModelError("at least one kernel-window statistics batch is required")
    validated = tuple(_validated_window_statistics(item) for item in statistics)
    n_states = validated[0].state_observation_counts.size
    if any(item.state_observation_counts.size != n_states for item in validated[1:]):
        raise ModelError("kernel-window batches use different state counts")
    identity = (
        validated[0].frequency,
        validated[0].model_artifact_fingerprint,
        validated[0].calibration_stream_fingerprint,
    )
    if any(
        (
            item.frequency,
            item.model_artifact_fingerprint,
            item.calibration_stream_fingerprint,
        )
        != identity
        for item in validated[1:]
    ):
        raise ModelError(
            "kernel-window batches use different frequency, artifact, "
            "or stream identity"
        )
    if validated[0].first_window_index != 0:
        raise ModelError("kernel-window merge must begin with window index zero")
    for previous, current in pairwise(validated):
        if current.first_window_index != previous.stop_window_index:
            raise ModelError(
                "kernel-window batches must be contiguous, ordered, and disjoint",
                details={
                    "previous_stop": previous.stop_window_index,
                    "current_start": current.first_window_index,
                },
            )
    state_window_counts = np.zeros(n_states, dtype=np.int64)
    state_observation_counts = np.zeros(n_states, dtype=np.int64)
    state_return_sums = np.zeros((n_states, ASSET_COUNT), dtype=np.float64)
    cluster_return_sum_squares = np.zeros_like(state_return_sums)
    cluster_count_return_products = np.zeros_like(state_return_sums)
    cluster_count_squares = np.zeros(n_states, dtype=np.float64)
    windows = 0
    for item in validated:
        windows += item.windows
        state_window_counts += item.state_window_counts
        state_observation_counts += item.state_observation_counts
        state_return_sums += item.state_return_sums
        cluster_return_sum_squares += item.cluster_return_sum_squares
        cluster_count_return_products += item.cluster_count_return_products
        cluster_count_squares += item.cluster_count_squares

    _make_read_only(
        state_window_counts,
        state_observation_counts,
        state_return_sums,
        cluster_return_sum_squares,
        cluster_count_return_products,
        cluster_count_squares,
    )
    source_fingerprint = _merged_window_source_fingerprint(validated)
    window_fingerprint = _window_statistics_fingerprint(
        frequency=identity[0],
        model_artifact_fingerprint=identity[1],
        calibration_stream_fingerprint=identity[2],
        first_window_index=0,
        stop_window_index=windows,
        source_window_fingerprint=source_fingerprint,
        arrays=(
            state_window_counts,
            state_observation_counts,
            state_return_sums,
            cluster_return_sum_squares,
            cluster_count_return_products,
            cluster_count_squares,
        ),
    )
    return KernelWindowStatistics(
        frequency=identity[0],
        model_artifact_fingerprint=identity[1],
        calibration_stream_fingerprint=identity[2],
        windows=windows,
        first_window_index=0,
        stop_window_index=windows,
        source_window_fingerprint=source_fingerprint,
        window_fingerprint=window_fingerprint,
        state_window_counts=state_window_counts,
        state_observation_counts=state_observation_counts,
        state_return_sums=state_return_sums,
        cluster_return_sum_squares=cluster_return_sum_squares,
        cluster_count_return_products=cluster_count_return_products,
        cluster_count_squares=cluster_count_squares,
    )


def finalize_kernel_mean_precision(
    statistics: KernelWindowStatistics,
    *,
    periods_per_year: int,
    confidence: float,
) -> KernelMeanPrecision:
    """Compute window-clustered simultaneous precision evidence.

    Version 1 mechanically controls all three assets, every selected state,
    both base frequencies, and all six registered cumulative looks.
    """

    item = _validated_window_statistics(statistics)
    annualization = _positive_integer(periods_per_year, "periods_per_year")
    expected_annualization = 12 if item.frequency == "monthly" else 252
    if annualization != expected_annualization:
        raise ModelError(
            "periods_per_year does not match the kernel frequency",
            details={
                "frequency": item.frequency,
                "expected": expected_annualization,
                "observed": annualization,
            },
        )
    comparisons = (
        ASSET_COUNT
        * int(item.state_observation_counts.size)
        * KERNEL_CONTROLLED_FREQUENCIES
        * KERNEL_PREFIX_LOOKS
    )
    if isinstance(confidence, bool) or not isinstance(confidence, float | np.floating):
        raise ModelError("confidence must be a floating-point value in (0, 1)")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ModelError("confidence must be a floating-point value in (0, 1)")
    if item.windows < 2:
        raise ModelError("clustered kernel precision needs at least two windows")
    if item.first_window_index != 0 or item.stop_window_index != item.windows:
        raise ModelError(
            "kernel precision requires a contiguous prefix from window zero"
        )
    sparse = np.flatnonzero(item.state_window_counts < 2)
    if sparse.size:
        raise ModelError(
            "every state needs observations in at least two independent windows",
            details={
                "states": sparse.astype(int).tolist(),
                "window_counts": item.state_window_counts[sparse].astype(int).tolist(),
            },
        )

    counts = item.state_observation_counts.astype(np.float64)[:, np.newaxis]
    means = item.state_return_sums / counts
    score_squares = (
        item.cluster_return_sum_squares
        - 2.0 * means * item.cluster_count_return_products
        + means * means * item.cluster_count_squares[:, np.newaxis]
    )
    scale = np.maximum(item.cluster_return_sum_squares, 1.0)
    if np.any(score_squares < -1e-12 * scale):
        raise ModelError("clustered kernel variance is numerically negative")
    score_squares = np.maximum(score_squares, 0.0)
    standard_errors = (
        np.sqrt((float(item.windows) / float(item.windows - 1)) * score_squares)
        / counts
    )
    tail_probability = (1.0 - level) / (2.0 * float(comparisons))
    critical = float(student_t.ppf(1.0 - tail_probability, df=item.windows - 1))
    if not np.isfinite(critical) or critical <= 0.0:
        raise ModelError("simultaneous Student-t critical value is invalid")
    annualized_means = means * float(annualization)
    half_widths = critical * standard_errors * float(annualization)

    state_window_counts = item.state_window_counts.copy()
    state_observation_counts = item.state_observation_counts.copy()
    _make_read_only(
        state_window_counts,
        state_observation_counts,
        means,
        standard_errors,
        annualized_means,
        half_widths,
    )
    sample_fingerprint = _kernel_precision_fingerprint(
        frequency=item.frequency,
        model_artifact_fingerprint=item.model_artifact_fingerprint,
        calibration_stream_fingerprint=item.calibration_stream_fingerprint,
        source_window_fingerprint=item.source_window_fingerprint,
        window_statistics_fingerprint=item.window_fingerprint,
        windows=item.windows,
        periods_per_year=annualization,
        confidence=level,
        degrees_freedom=item.windows - 1,
        critical_value=critical,
        simultaneous_comparisons=comparisons,
        arrays=(
            state_window_counts,
            state_observation_counts,
            means,
            standard_errors,
            annualized_means,
            half_widths,
        ),
    )
    return KernelMeanPrecision(
        frequency=item.frequency,
        model_artifact_fingerprint=item.model_artifact_fingerprint,
        calibration_stream_fingerprint=item.calibration_stream_fingerprint,
        source_window_fingerprint=item.source_window_fingerprint,
        window_statistics_fingerprint=item.window_fingerprint,
        kernel_sample_fingerprint=sample_fingerprint,
        sample_stage="unadjusted_kernel",
        windows=item.windows,
        periods_per_year=annualization,
        confidence=level,
        degrees_freedom=item.windows - 1,
        critical_value=critical,
        simultaneous_comparisons=comparisons,
        state_window_counts=state_window_counts,
        state_observation_counts=state_observation_counts,
        means=means,
        cluster_standard_errors=standard_errors,
        annualized_means=annualized_means,
        annualized_simultaneous_half_widths=half_widths,
    )


def historical_target_mean(historical_returns: ArrayLike) -> FloatArray:
    """Return the component-wise arithmetic mean of synchronized log vectors."""

    history = _return_matrix(historical_returns, label="historical returns")
    mean = np.asarray(np.mean(history, axis=0, dtype=np.float64), dtype=np.float64)
    _make_read_only(mean)
    return mean


def alpha_offset_grid(
    kernel_precision: KernelMeanPrecision,
    target_mean: ArrayLike,
    lambdas: ArrayLike,
    *,
    target_history_fingerprint: str,
) -> AlphaOffsetGrid:
    """Compute ``lambda * (kernel_mean - target_mean)`` exactly once.

    Lambda values must be strictly increasing in ``(0, 1]``.  The function is
    deliberately agnostic about which grid values are approved or qualified.
    """

    precision = _validated_kernel_precision(kernel_precision)
    kernel = precision.means
    target = _asset_vector(target_mean, label="target mean")
    values = _lambda_vector(lambdas)
    _validate_fingerprint(target_history_fingerprint, "target_history_fingerprint")
    offsets = values[:, np.newaxis, np.newaxis] * (
        kernel[np.newaxis, :, :] - target[np.newaxis, np.newaxis, :]
    )
    values = values.copy()
    _make_read_only(values, offsets)
    fingerprint = _tagged_array_fingerprint(
        "prpg-alpha-offset-grid-v1",
        (
            precision.frequency,
            precision.model_artifact_fingerprint,
            precision.kernel_sample_fingerprint,
            target_history_fingerprint,
        ),
        (values, offsets),
    )
    return AlphaOffsetGrid(
        frequency=precision.frequency,
        model_artifact_fingerprint=precision.model_artifact_fingerprint,
        kernel_sample_fingerprint=precision.kernel_sample_fingerprint,
        target_history_fingerprint=target_history_fingerprint,
        grid_fingerprint=fingerprint,
        lambdas=values,
        offsets=offsets,
    )


def make_return_vector_batch(
    sampled_returns: ArrayLike,
    target_states: ArrayLike,
    *,
    frequency: BaseFrequency,
    model_artifact_fingerprint: str,
) -> ReturnVectorBatch:
    """Create an unadjusted, provenance-bound batch for one policy application."""

    returns = _return_matrix(sampled_returns, label="sampled returns").copy()
    states = _state_vector(target_states, expected_size=returns.shape[0]).copy()
    checked_frequency = _frequency(frequency)
    _validate_fingerprint(model_artifact_fingerprint, "model_artifact_fingerprint")
    _make_read_only(returns, states)
    fingerprint = _tagged_array_fingerprint(
        "prpg-unadjusted-return-batch-v1",
        (checked_frequency, model_artifact_fingerprint),
        (returns, states),
    )
    return ReturnVectorBatch(
        values=returns,
        target_states=states,
        frequency=checked_frequency,
        model_artifact_fingerprint=model_artifact_fingerprint,
        source_batch_fingerprint=fingerprint,
        batch_fingerprint=fingerprint,
        applied_policy=None,
        policy_parameter_fingerprint=None,
    )


def apply_alpha_policy(
    batch: ReturnVectorBatch,
    *,
    policy: AlphaPolicy | str,
    selected_offsets: SelectedAlphaOffsets | None = None,
) -> ReturnVectorBatch:
    """Apply one candidate policy to synchronized sampled return vectors.

    The historical-vector policy is an exact copy.  The neutral policy applies
    one supplied offset by target state.  This API never estimates or updates
    offsets, preventing calibration output from being fed back iteratively.
    """

    source = _validated_return_batch(batch)
    if source.applied_policy is not None:
        raise ModelError("alpha policy has already been applied to this return batch")
    try:
        selected_policy = AlphaPolicy(policy)
    except (TypeError, ValueError) as error:
        raise ModelError(
            "unknown alpha policy", details={"policy": str(policy)}
        ) from error

    if selected_policy is AlphaPolicy.HISTORICAL_VECTOR:
        if selected_offsets is not None:
            raise ModelError(
                "historical-vector policy does not accept selected offsets"
            )
        transformed = source.values.copy()
        offset_fingerprint = "identity"
    else:
        if selected_offsets is None:
            raise ModelError("kernel-mean-neutral policy requires selected offsets")
        offsets = _validated_selected_offsets(selected_offsets)
        if offsets.frequency != source.frequency:
            raise ModelError("alpha offset frequency does not match return batch")
        if offsets.model_artifact_fingerprint != source.model_artifact_fingerprint:
            raise ModelError("alpha offset artifact does not match return batch")
        if int(np.max(source.target_states)) >= offsets.state_offsets.shape[0]:
            raise ModelError(
                "target state has no corresponding alpha offset",
                details={
                    "maximum_state": int(np.max(source.target_states)),
                    "offset_states": int(offsets.state_offsets.shape[0]),
                },
            )
        transformed = source.values - offsets.state_offsets[source.target_states]
        offset_fingerprint = offsets.selection_fingerprint

    if not np.all(np.isfinite(transformed)):
        raise ModelError("alpha-policy transformation produced non-finite returns")
    states = source.target_states.copy()
    _make_read_only(transformed, states)
    fingerprint = _tagged_array_fingerprint(
        "prpg-policy-applied-return-batch-v1",
        (
            source.source_batch_fingerprint,
            selected_policy.value,
            offset_fingerprint,
        ),
        (transformed, states),
    )
    return ReturnVectorBatch(
        values=transformed,
        target_states=states,
        frequency=source.frequency,
        model_artifact_fingerprint=source.model_artifact_fingerprint,
        source_batch_fingerprint=source.source_batch_fingerprint,
        batch_fingerprint=fingerprint,
        applied_policy=selected_policy,
        policy_parameter_fingerprint=offset_fingerprint,
    )


def select_smallest_qualified_lambda(
    grid: AlphaOffsetGrid,
    qualified: ArrayLike,
) -> SelectedAlphaOffsets:
    """Return the smallest preordered offset passing every external gate."""

    checked = _validated_offset_grid(grid)
    mask = np.asarray(qualified)
    if mask.ndim != 1 or mask.size != checked.lambdas.size or mask.dtype.kind != "b":
        raise ModelError("qualified mask must be Boolean and match the lambda grid")
    passing = np.flatnonzero(mask)
    if passing.size == 0:
        raise ModelError("no alpha lambda passed every required gate")
    index = int(passing[0])
    state_offsets = checked.offsets[index].copy()
    _make_read_only(state_offsets)
    selection_fingerprint = _tagged_array_fingerprint(
        "prpg-selected-alpha-offset-v1",
        (
            checked.frequency,
            checked.model_artifact_fingerprint,
            checked.grid_fingerprint,
            repr(float(checked.lambdas[index])),
        ),
        (state_offsets,),
    )
    return SelectedAlphaOffsets(
        frequency=checked.frequency,
        model_artifact_fingerprint=checked.model_artifact_fingerprint,
        grid_fingerprint=checked.grid_fingerprint,
        lambda_value=float(checked.lambdas[index]),
        state_offsets=state_offsets,
        selection_fingerprint=selection_fingerprint,
    )


def _validated_offset_grid(grid: AlphaOffsetGrid) -> AlphaOffsetGrid:
    if not isinstance(grid, AlphaOffsetGrid):
        raise ModelError("alpha offset grid has the wrong type")
    frequency = _frequency(grid.frequency)
    for label, value in (
        ("model_artifact_fingerprint", grid.model_artifact_fingerprint),
        ("kernel_sample_fingerprint", grid.kernel_sample_fingerprint),
        ("target_history_fingerprint", grid.target_history_fingerprint),
        ("grid_fingerprint", grid.grid_fingerprint),
    ):
        _validate_fingerprint(value, label)
    lambdas = _lambda_vector(grid.lambdas).copy()
    offsets = np.asarray(grid.offsets, dtype=np.float64)
    if offsets.ndim != 3 or offsets.shape != (
        lambdas.size,
        offsets.shape[1] if offsets.ndim == 3 else 0,
        ASSET_COUNT,
    ):
        raise ModelError("alpha offset grid array shape is invalid")
    if offsets.shape[1] == 0 or not np.all(np.isfinite(offsets)):
        raise ModelError("alpha offset grid must contain finite state offsets")
    offsets = offsets.copy()
    expected = _tagged_array_fingerprint(
        "prpg-alpha-offset-grid-v1",
        (
            frequency,
            grid.model_artifact_fingerprint,
            grid.kernel_sample_fingerprint,
            grid.target_history_fingerprint,
        ),
        (lambdas, offsets),
    )
    if expected != grid.grid_fingerprint:
        raise ModelError("alpha offset grid fingerprint does not match its content")
    _make_read_only(lambdas, offsets)
    return AlphaOffsetGrid(
        frequency=frequency,
        model_artifact_fingerprint=grid.model_artifact_fingerprint,
        kernel_sample_fingerprint=grid.kernel_sample_fingerprint,
        target_history_fingerprint=grid.target_history_fingerprint,
        grid_fingerprint=grid.grid_fingerprint,
        lambdas=lambdas,
        offsets=offsets,
    )


def _validated_selected_offsets(
    selected: SelectedAlphaOffsets,
) -> SelectedAlphaOffsets:
    if not isinstance(selected, SelectedAlphaOffsets):
        raise ModelError("selected alpha offsets have the wrong type")
    frequency = _frequency(selected.frequency)
    for label, value in (
        ("model_artifact_fingerprint", selected.model_artifact_fingerprint),
        ("grid_fingerprint", selected.grid_fingerprint),
        ("selection_fingerprint", selected.selection_fingerprint),
    ):
        _validate_fingerprint(value, label)
    if not np.isfinite(selected.lambda_value) or not 0.0 < selected.lambda_value <= 1.0:
        raise ModelError("selected alpha lambda must be finite and in (0, 1]")
    offsets = _state_asset_matrix(
        selected.state_offsets, label="selected state offsets"
    ).copy()
    expected = _tagged_array_fingerprint(
        "prpg-selected-alpha-offset-v1",
        (
            frequency,
            selected.model_artifact_fingerprint,
            selected.grid_fingerprint,
            repr(float(selected.lambda_value)),
        ),
        (offsets,),
    )
    if expected != selected.selection_fingerprint:
        raise ModelError("selected alpha offset fingerprint does not match its content")
    _make_read_only(offsets)
    return SelectedAlphaOffsets(
        frequency=frequency,
        model_artifact_fingerprint=selected.model_artifact_fingerprint,
        grid_fingerprint=selected.grid_fingerprint,
        lambda_value=float(selected.lambda_value),
        state_offsets=offsets,
        selection_fingerprint=selected.selection_fingerprint,
    )


def _validated_return_batch(batch: ReturnVectorBatch) -> ReturnVectorBatch:
    if not isinstance(batch, ReturnVectorBatch):
        raise ModelError("return vector batch has the wrong type")
    values = _return_matrix(batch.values, label="return batch values").copy()
    states = _state_vector(batch.target_states, expected_size=values.shape[0]).copy()
    frequency = _frequency(batch.frequency)
    _validate_fingerprint(
        batch.model_artifact_fingerprint, "model_artifact_fingerprint"
    )
    _validate_fingerprint(batch.source_batch_fingerprint, "source_batch_fingerprint")
    _validate_fingerprint(batch.batch_fingerprint, "batch_fingerprint")
    if batch.applied_policy is None:
        if batch.policy_parameter_fingerprint is not None:
            raise ModelError("unadjusted return batch cannot have policy parameters")
        expected_source = _tagged_array_fingerprint(
            "prpg-unadjusted-return-batch-v1",
            (frequency, batch.model_artifact_fingerprint),
            (values, states),
        )
        if (
            expected_source != batch.source_batch_fingerprint
            or expected_source != batch.batch_fingerprint
        ):
            raise ModelError("unadjusted return batch fingerprint is invalid")
        applied_policy = None
        parameter_fingerprint = None
    else:
        try:
            applied_policy = AlphaPolicy(batch.applied_policy)
        except (TypeError, ValueError) as error:
            raise ModelError("return batch has an unknown applied policy") from error
        parameter_fingerprint = batch.policy_parameter_fingerprint
        if parameter_fingerprint == "identity":
            if applied_policy is not AlphaPolicy.HISTORICAL_VECTOR:
                raise ModelError("identity policy parameters require historical-vector")
        else:
            if applied_policy is not AlphaPolicy.KERNEL_MEAN_NEUTRAL:
                raise ModelError("selected offsets require kernel-mean-neutral policy")
            if not isinstance(parameter_fingerprint, str):
                raise ModelError("kernel-mean-neutral policy parameters are missing")
            _validate_fingerprint(parameter_fingerprint, "policy_parameter_fingerprint")
        expected_batch = _tagged_array_fingerprint(
            "prpg-policy-applied-return-batch-v1",
            (
                batch.source_batch_fingerprint,
                applied_policy.value,
                str(parameter_fingerprint),
            ),
            (values, states),
        )
        if expected_batch != batch.batch_fingerprint:
            raise ModelError("policy-applied return batch fingerprint is invalid")
    _make_read_only(values, states)
    return ReturnVectorBatch(
        values=values,
        target_states=states,
        frequency=frequency,
        model_artifact_fingerprint=batch.model_artifact_fingerprint,
        source_batch_fingerprint=batch.source_batch_fingerprint,
        batch_fingerprint=batch.batch_fingerprint,
        applied_policy=applied_policy,
        policy_parameter_fingerprint=parameter_fingerprint,
    )


def _validated_statistics(item: KernelMeanStatistics) -> KernelMeanStatistics:
    if not isinstance(item, KernelMeanStatistics):
        raise ModelError("kernel-mean batch has the wrong type")
    try:
        counts = np.asarray(item.counts)
        means = np.asarray(item.means, dtype=np.float64)
        centered = np.asarray(item.centered_sum_squares, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("kernel-mean batch arrays are invalid") from error
    if counts.ndim != 1 or counts.size == 0 or counts.dtype.kind not in "iu":
        raise ModelError("kernel-mean counts must be a non-empty integer vector")
    expected = (counts.size, ASSET_COUNT)
    if means.shape != expected or centered.shape != expected:
        raise ModelError("kernel-mean batch array shapes are inconsistent")
    if np.any(counts < 0):
        raise ModelError("kernel-mean counts cannot be negative")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(centered)):
        raise ModelError("kernel-mean batch contains non-finite values")
    if np.any(centered < -1e-12):
        raise ModelError("kernel centered sums of squares cannot be negative")
    if not isinstance(item.observations, int) or isinstance(item.observations, bool):
        raise ModelError("kernel observation count must be an integer")
    if item.observations != int(np.sum(counts, dtype=np.int64)):
        raise ModelError("kernel observation count does not match state counts")
    empty = counts == 0
    if np.any(means[empty] != 0.0) or np.any(centered[empty] != 0.0):
        raise ModelError("empty kernel states must use canonical zero statistics")
    normalized_counts = counts.astype(np.int64, copy=True)
    normalized_means = means.copy()
    normalized_centered = np.maximum(centered, 0.0)
    _make_read_only(normalized_counts, normalized_means, normalized_centered)
    return KernelMeanStatistics(
        counts=normalized_counts,
        means=normalized_means,
        centered_sum_squares=normalized_centered,
        observations=item.observations,
    )


def _validated_window_statistics(
    item: KernelWindowStatistics,
) -> KernelWindowStatistics:
    if not isinstance(item, KernelWindowStatistics):
        raise ModelError("kernel-window batch has the wrong type")
    frequency = _frequency(item.frequency)
    _validate_fingerprint(item.model_artifact_fingerprint, "model_artifact_fingerprint")
    _validate_fingerprint(
        item.calibration_stream_fingerprint, "calibration_stream_fingerprint"
    )
    if isinstance(item.windows, bool) or not isinstance(item.windows, int):
        raise ModelError("kernel-window count must be a positive integer")
    if item.windows <= 0:
        raise ModelError("kernel-window count must be a positive integer")
    first = _nonnegative_integer(item.first_window_index, "first_window_index")
    stop = _positive_integer(item.stop_window_index, "stop_window_index")
    if stop - first != item.windows:
        raise ModelError("kernel-window range does not match its window count")
    _validate_fingerprint(item.source_window_fingerprint, "source_window_fingerprint")
    _validate_fingerprint(item.window_fingerprint, "window_fingerprint")

    try:
        state_windows = np.asarray(item.state_window_counts)
        observations = np.asarray(item.state_observation_counts)
        return_sums = np.asarray(item.state_return_sums, dtype=np.float64)
        return_squares = np.asarray(item.cluster_return_sum_squares, dtype=np.float64)
        count_return = np.asarray(item.cluster_count_return_products, dtype=np.float64)
        count_squares = np.asarray(item.cluster_count_squares, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("kernel-window batch arrays are invalid") from error
    if (
        state_windows.ndim != 1
        or state_windows.size == 0
        or state_windows.dtype.kind not in "iu"
        or observations.shape != state_windows.shape
        or observations.dtype.kind not in "iu"
    ):
        raise ModelError("kernel-window state counts must be non-empty integer vectors")
    expected = (state_windows.size, ASSET_COUNT)
    if (
        return_sums.shape != expected
        or return_squares.shape != expected
        or count_return.shape != expected
        or count_squares.shape != state_windows.shape
    ):
        raise ModelError("kernel-window batch array shapes are inconsistent")
    numeric_arrays = (return_sums, return_squares, count_return, count_squares)
    if any(not np.all(np.isfinite(array)) for array in numeric_arrays):
        raise ModelError("kernel-window batch contains non-finite values")
    if (
        np.any(state_windows < 0)
        or np.any(observations < 0)
        or np.any(observations < state_windows)
        or np.any(state_windows > item.windows)
        or np.any(return_squares < -1e-12)
        or np.any(count_squares < -1e-12)
        or np.any(count_squares + 1e-12 < observations)
    ):
        raise ModelError("kernel-window batch contains invalid counts or squares")
    if np.any((state_windows == 0) != (observations == 0)):
        raise ModelError("kernel-window presence and observation counts disagree")
    empty = observations == 0
    if any(np.any(array[empty] != 0.0) for array in numeric_arrays):
        raise ModelError(
            "empty kernel-window states must use canonical zero statistics"
        )
    if not np.allclose(count_squares, np.rint(count_squares), rtol=0.0, atol=1e-9):
        raise ModelError("kernel-window squared counts must be exact integers")
    window_count = float(item.windows)
    centered_count = count_squares - observations.astype(np.float64) ** 2 / window_count
    centered_return = return_squares - return_sums * return_sums / window_count
    centered_cross = (
        count_return
        - observations.astype(np.float64)[:, np.newaxis] * return_sums / window_count
    )
    diagonal_scale = np.maximum(
        np.maximum(np.abs(count_squares)[:, np.newaxis], np.abs(return_squares)),
        1.0,
    )
    if np.any(centered_count < -1e-10 * diagonal_scale[:, 0]) or np.any(
        centered_return < -1e-10 * diagonal_scale
    ):
        raise ModelError("kernel-window centered moments are inconsistent")
    centered_count = np.maximum(centered_count, 0.0)
    centered_return = np.maximum(centered_return, 0.0)
    determinants = centered_count[:, np.newaxis] * centered_return - (
        centered_cross * centered_cross
    )
    determinant_scale = np.maximum(
        np.abs(centered_count[:, np.newaxis] * centered_return)
        + centered_cross * centered_cross,
        1.0,
    )
    if np.any(determinants < -1e-10 * determinant_scale):
        raise ModelError("kernel-window count/return moment matrix is not PSD")

    normalized_state_windows = state_windows.astype(np.int64, copy=True)
    normalized_observations = observations.astype(np.int64, copy=True)
    normalized_return_sums = return_sums.copy()
    normalized_return_squares = np.maximum(return_squares, 0.0)
    normalized_count_return = count_return.copy()
    normalized_count_squares = np.maximum(count_squares, 0.0)
    expected_fingerprint = _window_statistics_fingerprint(
        frequency=frequency,
        model_artifact_fingerprint=item.model_artifact_fingerprint,
        calibration_stream_fingerprint=item.calibration_stream_fingerprint,
        first_window_index=first,
        stop_window_index=stop,
        source_window_fingerprint=item.source_window_fingerprint,
        arrays=(
            normalized_state_windows,
            normalized_observations,
            normalized_return_sums,
            normalized_return_squares,
            normalized_count_return,
            normalized_count_squares,
        ),
    )
    if expected_fingerprint != item.window_fingerprint:
        raise ModelError("kernel-window fingerprint does not match its content")
    _make_read_only(
        normalized_state_windows,
        normalized_observations,
        normalized_return_sums,
        normalized_return_squares,
        normalized_count_return,
        normalized_count_squares,
    )
    return KernelWindowStatistics(
        frequency=frequency,
        model_artifact_fingerprint=item.model_artifact_fingerprint,
        calibration_stream_fingerprint=item.calibration_stream_fingerprint,
        windows=item.windows,
        first_window_index=first,
        stop_window_index=stop,
        source_window_fingerprint=item.source_window_fingerprint,
        window_fingerprint=item.window_fingerprint,
        state_window_counts=normalized_state_windows,
        state_observation_counts=normalized_observations,
        state_return_sums=normalized_return_sums,
        cluster_return_sum_squares=normalized_return_squares,
        cluster_count_return_products=normalized_count_return,
        cluster_count_squares=normalized_count_squares,
    )


def _validated_kernel_precision(
    item: KernelMeanPrecision,
) -> KernelMeanPrecision:
    if not isinstance(item, KernelMeanPrecision):
        raise ModelError("kernel precision has the wrong type")
    frequency = _frequency(item.frequency)
    for label, value in (
        ("model_artifact_fingerprint", item.model_artifact_fingerprint),
        ("calibration_stream_fingerprint", item.calibration_stream_fingerprint),
        ("source_window_fingerprint", item.source_window_fingerprint),
        ("window_statistics_fingerprint", item.window_statistics_fingerprint),
        ("kernel_sample_fingerprint", item.kernel_sample_fingerprint),
    ):
        _validate_fingerprint(value, label)
    if item.sample_stage != "unadjusted_kernel":
        raise ModelError(
            "kernel precision is not from the unadjusted calibration stage"
        )
    windows = _positive_integer(item.windows, "windows")
    periods_per_year = _positive_integer(item.periods_per_year, "periods_per_year")
    expected_annualization = 12 if frequency == "monthly" else 252
    if periods_per_year != expected_annualization:
        raise ModelError("kernel precision annualization does not match its frequency")
    if item.degrees_freedom != windows - 1 or windows < 2:
        raise ModelError("kernel precision degrees of freedom are invalid")
    if (
        isinstance(item.confidence, bool)
        or not isinstance(item.confidence, float | np.floating)
        or not np.isfinite(item.confidence)
        or not 0.0 < float(item.confidence) < 1.0
    ):
        raise ModelError("kernel precision confidence is invalid")
    if not np.isfinite(item.critical_value) or item.critical_value <= 0.0:
        raise ModelError("kernel precision critical value is invalid")

    try:
        state_windows = np.asarray(item.state_window_counts)
        observations = np.asarray(item.state_observation_counts)
        means = np.asarray(item.means, dtype=np.float64)
        standard_errors = np.asarray(item.cluster_standard_errors, dtype=np.float64)
        annualized_means = np.asarray(item.annualized_means, dtype=np.float64)
        half_widths = np.asarray(
            item.annualized_simultaneous_half_widths, dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ModelError("kernel precision arrays are invalid") from error
    if (
        state_windows.ndim != 1
        or state_windows.size == 0
        or state_windows.dtype.kind not in "iu"
        or observations.shape != state_windows.shape
        or observations.dtype.kind not in "iu"
    ):
        raise ModelError("kernel precision counts must be non-empty integer vectors")
    expected_shape = (state_windows.size, ASSET_COUNT)
    numeric = (means, standard_errors, annualized_means, half_widths)
    if any(array.shape != expected_shape for array in numeric):
        raise ModelError("kernel precision array shapes are inconsistent")
    if any(not np.all(np.isfinite(array)) for array in numeric):
        raise ModelError("kernel precision contains non-finite values")
    if (
        np.any(state_windows < 2)
        or np.any(state_windows > windows)
        or np.any(observations < state_windows)
        or np.any(standard_errors < 0.0)
        or np.any(half_widths < 0.0)
    ):
        raise ModelError("kernel precision counts or uncertainties are invalid")
    comparisons = (
        ASSET_COUNT
        * int(state_windows.size)
        * KERNEL_CONTROLLED_FREQUENCIES
        * KERNEL_PREFIX_LOOKS
    )
    if item.simultaneous_comparisons != comparisons:
        raise ModelError("kernel precision comparison count is invalid")
    tail_probability = (1.0 - float(item.confidence)) / (2.0 * comparisons)
    expected_critical = float(
        student_t.ppf(1.0 - tail_probability, df=item.degrees_freedom)
    )
    if not np.isclose(item.critical_value, expected_critical, rtol=1e-13, atol=0.0):
        raise ModelError("kernel precision critical value is inconsistent")
    if not np.allclose(
        annualized_means,
        means * float(periods_per_year),
        rtol=1e-13,
        atol=0.0,
    ) or not np.allclose(
        half_widths,
        item.critical_value * standard_errors * float(periods_per_year),
        rtol=1e-13,
        atol=0.0,
    ):
        raise ModelError("kernel precision annualized values are inconsistent")

    normalized_state_windows = state_windows.astype(np.int64, copy=True)
    normalized_observations = observations.astype(np.int64, copy=True)
    normalized_means = means.copy()
    normalized_standard_errors = standard_errors.copy()
    normalized_annualized_means = annualized_means.copy()
    normalized_half_widths = half_widths.copy()
    expected_fingerprint = _kernel_precision_fingerprint(
        frequency=frequency,
        model_artifact_fingerprint=item.model_artifact_fingerprint,
        calibration_stream_fingerprint=item.calibration_stream_fingerprint,
        source_window_fingerprint=item.source_window_fingerprint,
        window_statistics_fingerprint=item.window_statistics_fingerprint,
        windows=windows,
        periods_per_year=periods_per_year,
        confidence=float(item.confidence),
        degrees_freedom=item.degrees_freedom,
        critical_value=float(item.critical_value),
        simultaneous_comparisons=comparisons,
        arrays=(
            normalized_state_windows,
            normalized_observations,
            normalized_means,
            normalized_standard_errors,
            normalized_annualized_means,
            normalized_half_widths,
        ),
    )
    if expected_fingerprint != item.kernel_sample_fingerprint:
        raise ModelError(
            "kernel sample fingerprint does not match its precision record"
        )
    _make_read_only(
        normalized_state_windows,
        normalized_observations,
        normalized_means,
        normalized_standard_errors,
        normalized_annualized_means,
        normalized_half_widths,
    )
    return KernelMeanPrecision(
        frequency=frequency,
        model_artifact_fingerprint=item.model_artifact_fingerprint,
        calibration_stream_fingerprint=item.calibration_stream_fingerprint,
        source_window_fingerprint=item.source_window_fingerprint,
        window_statistics_fingerprint=item.window_statistics_fingerprint,
        kernel_sample_fingerprint=item.kernel_sample_fingerprint,
        sample_stage="unadjusted_kernel",
        windows=windows,
        periods_per_year=periods_per_year,
        confidence=float(item.confidence),
        degrees_freedom=item.degrees_freedom,
        critical_value=float(item.critical_value),
        simultaneous_comparisons=comparisons,
        state_window_counts=normalized_state_windows,
        state_observation_counts=normalized_observations,
        means=normalized_means,
        cluster_standard_errors=normalized_standard_errors,
        annualized_means=normalized_annualized_means,
        annualized_simultaneous_half_widths=normalized_half_widths,
    )


def _return_matrix(values: ArrayLike, *, label: str) -> FloatArray:
    matrix = _numeric_array(values, label=label)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != ASSET_COUNT:
        raise ModelError(f"{label} must have shape (n, {ASSET_COUNT}) with n positive")
    return matrix


def _return_cube(values: ArrayLike, *, label: str) -> FloatArray:
    cube = _numeric_array(values, label=label)
    if (
        cube.ndim != 3
        or cube.shape[0] == 0
        or cube.shape[1] == 0
        or cube.shape[2] != ASSET_COUNT
    ):
        raise ModelError(
            f"{label} must have shape (windows, periods, {ASSET_COUNT}) "
            "with positive dimensions"
        )
    return cube


def _state_asset_matrix(values: ArrayLike, *, label: str) -> FloatArray:
    matrix = _numeric_array(values, label=label)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != ASSET_COUNT:
        raise ModelError(
            f"{label} must have shape (states, {ASSET_COUNT}) with states positive"
        )
    return matrix


def _asset_vector(values: ArrayLike, *, label: str) -> FloatArray:
    vector = _numeric_array(values, label=label)
    if vector.shape != (ASSET_COUNT,):
        raise ModelError(f"{label} must contain exactly {ASSET_COUNT} values")
    return vector


def _numeric_array(values: ArrayLike, *, label: str) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if not np.all(np.isfinite(result)):
        raise ModelError(f"{label} must contain only finite values")
    return result


def _state_vector(values: ArrayLike, *, expected_size: int) -> IntArray:
    states = np.asarray(values)
    if states.ndim != 1 or states.size != expected_size:
        raise ModelError("target states must be one-dimensional and match returns")
    if states.dtype.kind not in "iu":
        raise ModelError("target states must have integer dtype")
    if np.any(states < 0):
        raise ModelError("target states cannot be negative")
    return states.astype(np.int64, copy=False)


def _state_matrix(values: ArrayLike, *, expected_shape: tuple[int, int]) -> IntArray:
    states = np.asarray(values)
    if states.shape != expected_shape:
        raise ModelError("target-state windows must match return-window geometry")
    if states.dtype.kind not in "iu":
        raise ModelError("target-state windows must have integer dtype")
    if np.any(states < 0):
        raise ModelError("target-state windows cannot be negative")
    return states.astype(np.int64, copy=False)


def _lambda_vector(values: ArrayLike) -> FloatArray:
    vector = _numeric_array(values, label="lambda grid")
    if vector.ndim != 1 or vector.size == 0:
        raise ModelError("lambda grid must be a non-empty vector")
    if np.any(vector <= 0.0) or np.any(vector > 1.0):
        raise ModelError("lambda values must lie in (0, 1]")
    if vector.size > 1 and np.any(np.diff(vector) <= 0.0):
        raise ModelError("lambda values must be strictly increasing")
    return vector


def _window_source_fingerprint(
    first_window_index: int,
    frequency: BaseFrequency,
    model_artifact_fingerprint: str,
    calibration_stream_fingerprint: str,
    returns: FloatArray,
    states: IntArray,
) -> str:
    return _tagged_array_fingerprint(
        "prpg-unadjusted-kernel-window-source-v1",
        (
            frequency,
            model_artifact_fingerprint,
            calibration_stream_fingerprint,
            str(first_window_index),
        ),
        (returns, states),
    )


def _window_statistics_fingerprint(
    *,
    frequency: BaseFrequency,
    model_artifact_fingerprint: str,
    calibration_stream_fingerprint: str,
    first_window_index: int,
    stop_window_index: int,
    source_window_fingerprint: str,
    arrays: Sequence[NDArray[np.generic]],
) -> str:
    return _tagged_array_fingerprint(
        "prpg-unadjusted-kernel-window-statistics-v1",
        (
            frequency,
            model_artifact_fingerprint,
            calibration_stream_fingerprint,
            str(first_window_index),
            str(stop_window_index),
            source_window_fingerprint,
        ),
        arrays,
    )


def _tagged_array_fingerprint(
    namespace: str,
    tags: Sequence[str],
    arrays: Sequence[NDArray[np.generic]],
) -> str:
    digest = hashlib.sha256()
    for value in (namespace, *tags):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    for array in arrays:
        source = np.asarray(array)
        if source.dtype.kind == "f":
            canonical = np.ascontiguousarray(source, dtype=np.dtype("<f8"))
        elif source.dtype.kind in "iu":
            canonical = np.ascontiguousarray(source, dtype=np.dtype("<i8"))
        elif source.dtype.kind == "b":
            canonical = np.ascontiguousarray(source, dtype=np.dtype("?"))
        else:  # pragma: no cover - private callers validate numeric arrays
            raise ModelError("fingerprinted array has an unsupported dtype")
        dtype_name = canonical.dtype.str.encode("ascii")
        digest.update(len(dtype_name).to_bytes(1, "big", signed=False))
        digest.update(dtype_name)
        digest.update(canonical.ndim.to_bytes(1, "big", signed=False))
        for dimension in canonical.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _merged_window_source_fingerprint(
    statistics: Sequence[KernelWindowStatistics],
) -> str:
    digest = hashlib.sha256(b"prpg-unadjusted-kernel-window-source-merge-v1\0")
    for item in statistics:
        digest.update(item.first_window_index.to_bytes(8, "big", signed=False))
        digest.update(item.stop_window_index.to_bytes(8, "big", signed=False))
        digest.update(bytes.fromhex(item.source_window_fingerprint))
    return digest.hexdigest()


def _kernel_precision_fingerprint(
    *,
    frequency: BaseFrequency,
    model_artifact_fingerprint: str,
    calibration_stream_fingerprint: str,
    source_window_fingerprint: str,
    window_statistics_fingerprint: str,
    windows: int,
    periods_per_year: int,
    confidence: float,
    degrees_freedom: int,
    critical_value: float,
    simultaneous_comparisons: int,
    arrays: Sequence[NDArray[np.generic]],
) -> str:
    return _tagged_array_fingerprint(
        "prpg-unadjusted-kernel-mean-precision-v1",
        (
            frequency,
            model_artifact_fingerprint,
            calibration_stream_fingerprint,
            source_window_fingerprint,
            window_statistics_fingerprint,
            str(windows),
            str(periods_per_year),
            repr(float(confidence)),
            str(degrees_freedom),
            repr(float(critical_value)),
            str(simultaneous_comparisons),
            "unadjusted_kernel",
        ),
        arrays,
    )


def _validate_fingerprint(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelError(f"{label} must be a lowercase SHA-256 hex digest")


def _frequency(value: str) -> BaseFrequency:
    if value == "monthly":
        return "monthly"
    if value == "daily":
        return "daily"
    raise ModelError("base frequency must be 'monthly' or 'daily'")


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


def _make_read_only(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
