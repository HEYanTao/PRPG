"""Exact conditioned stationary-bootstrap index-trace primitives.

The routines in this module implement the branch and draw contract frozen in
design Section 8.5.  They choose historical row indices only: copying or
adjusting return vectors belongs to a later generation layer.  Both routines
consume caller-supplied uniform vectors so their results are independent of
execution order, worker count, and conditional branches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.hmm import (
    PROBABILITY_ATOL,
    STATIONARY_RESIDUAL_ATOL,
    validate_markov_chain,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class BootstrapTraceReason(StrEnum):
    """Mutually exclusive reason for each source-index trace step.

    The enum order mirrors the deterministic restart precedence.  A
    continuation-eligible step ends as either ``RANDOM_RESTART`` or
    ``CONTINUATION``.
    """

    FIRST_OBSERVATION = "first_observation"
    TARGET_SEGMENT_START = "target_segment_start"
    TARGET_STATE_TRANSITION = "target_state_transition"
    SOURCE_END = "source_end"
    SOURCE_GAP = "source_gap"
    SOURCE_STATE_MISMATCH = "source_state_mismatch"
    RANDOM_RESTART = "random_restart"
    CONTINUATION = "continuation"


@dataclass(frozen=True, slots=True)
class SourceIndexTrace:
    """Auditable outcome of conditioned stationary-bootstrap index sampling."""

    source_indices: IntArray
    restart_flags: BoolArray
    reasons: tuple[BootstrapTraceReason, ...]
    restart_uniform_used: BoolArray
    source_rank_uniform_used: BoolArray
    restart_uniforms_consumed: int
    source_rank_uniforms_consumed: int


@dataclass(frozen=True, slots=True)
class TargetStateTrace:
    """Target states drawn with one pre-drawn uniform per observation."""

    states: IntArray
    uniforms_consumed: int


@dataclass(frozen=True, slots=True)
class SegmentedStateTrace:
    """State vector plus explicit segment geometry and forward links."""

    states: IntArray
    segment_lengths: tuple[int, ...]
    continuation_links: BoolArray
    uniforms_consumed: int


@dataclass(frozen=True, slots=True)
class ExpandedStateTrace:
    """Monthly states expanded and terminally truncated within each segment."""

    states: IntArray
    segment_lengths: tuple[int, ...]
    continuation_links: BoolArray


@dataclass(frozen=True, slots=True)
class IdealRestartTrace:
    """Counterfactual restart trace without source fragmentation."""

    restart_flags: BoolArray
    reasons: tuple[BootstrapTraceReason, ...]
    restart_uniform_used: BoolArray
    restart_uniforms_consumed: int


def simulate_target_states_from_uniforms(
    stationary_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    uniforms: ArrayLike,
) -> TargetStateTrace:
    """Simulate a stationary Markov path using the exact categorical rule.

    ``uniforms[0]`` selects the initial state from the supplied stationary
    distribution.  Every later uniform selects from the transition row of the
    previously selected state.  Categorical selection uses cumulative
    probability search with an explicit last-bin fallback for accepted
    floating-point row-sum residuals.
    """

    transition, stationary = _validated_markov_inputs(
        stationary_distribution, transition_matrix
    )
    draws = _uniform_vector(uniforms, expected_size=None, label="state uniforms")
    if draws.size == 0:
        raise ModelError("state uniforms must be non-empty")
    states = np.empty(draws.size, dtype=np.int64)
    states[0] = _categorical_index(stationary, float(draws[0]))
    for position in range(1, draws.size):
        states[position] = _categorical_index(
            transition[states[position - 1]], float(draws[position])
        )
    _make_read_only(states)
    return TargetStateTrace(states=states, uniforms_consumed=int(draws.size))


def simulate_segmented_target_states_from_uniforms(
    stationary_distribution: ArrayLike,
    transition_matrix: ArrayLike,
    uniforms: ArrayLike,
    *,
    segment_lengths: Sequence[int],
) -> SegmentedStateTrace:
    """Simulate each target segment independently, resetting its first draw."""

    draws = _uniform_vector(uniforms, expected_size=None, label="state uniforms")
    lengths = _validated_segment_lengths(
        segment_lengths,
        expected_total=int(draws.size),
        label="target segment lengths",
    )
    transition, stationary = _validated_markov_inputs(
        stationary_distribution, transition_matrix
    )
    states = np.empty(draws.size, dtype=np.int64)
    offset = 0
    for length in lengths:
        stop = offset + length
        segment_draws = draws[offset:stop]
        states[offset] = _categorical_index(stationary, float(segment_draws[0]))
        for local_position in range(1, length):
            position = offset + local_position
            states[position] = _categorical_index(
                transition[states[position - 1]],
                float(segment_draws[local_position]),
            )
        offset = stop
    links = _continuation_from_segment_lengths(lengths)
    _make_read_only(states)
    return SegmentedStateTrace(
        states=states,
        segment_lengths=lengths,
        continuation_links=links,
        uniforms_consumed=int(draws.size),
    )


def expand_monthly_states_to_daily(
    monthly_states: ArrayLike,
    *,
    monthly_segment_lengths: Sequence[int],
    daily_segment_lengths: Sequence[int],
) -> ExpandedStateTrace:
    """Repeat states 21 sessions and terminally truncate each segment.

    Truncation is local to each segment and may remove only part of its final
    repeated month.  This prevents the production gap from being erased by a
    global repeat-and-truncate operation.
    """

    monthly = _state_vector(monthly_states, "monthly target states")
    monthly_lengths = _validated_segment_lengths(
        monthly_segment_lengths,
        expected_total=int(monthly.size),
        label="monthly segment lengths",
    )
    daily_lengths = _validated_segment_lengths(
        daily_segment_lengths,
        expected_total=None,
        label="daily segment lengths",
    )
    if len(monthly_lengths) != len(daily_lengths):
        raise ModelError(
            "monthly and daily geometry must contain the same segment count",
            details={
                "monthly_segments": len(monthly_lengths),
                "daily_segments": len(daily_lengths),
            },
        )

    expanded: list[IntArray] = []
    offset = 0
    for segment, (monthly_length, daily_length) in enumerate(
        zip(monthly_lengths, daily_lengths, strict=True)
    ):
        minimum = (monthly_length - 1) * 21 + 1
        maximum = monthly_length * 21
        if not minimum <= daily_length <= maximum:
            raise ModelError(
                "daily segment is not a terminal truncation of 21-session months",
                details={
                    "segment": segment,
                    "monthly_length": monthly_length,
                    "daily_length": daily_length,
                    "allowed": [minimum, maximum],
                },
            )
        source = monthly[offset : offset + monthly_length]
        expanded.append(np.repeat(source, 21)[:daily_length])
        offset += monthly_length
    states = np.concatenate(expanded).astype(np.int64, copy=False)
    links = _continuation_from_segment_lengths(daily_lengths)
    _make_read_only(states)
    return ExpandedStateTrace(
        states=states,
        segment_lengths=daily_lengths,
        continuation_links=links,
    )


def ideal_restart_trace(
    target_states: ArrayLike,
    restart_uniforms: ArrayLike,
    *,
    restart_probability: float,
    target_continuation_links: ArrayLike | None = None,
) -> IdealRestartTrace:
    """Build the paired ideal restart trace from the shared Bernoulli draws."""

    target = _state_vector(target_states, "target states")
    draws = _uniform_vector(
        restart_uniforms,
        expected_size=target.size,
        label="restart uniforms",
    )
    probability = _restart_probability(restart_probability)
    target_links = _optional_target_links(target_continuation_links, target.size)
    restarts = np.zeros(target.size, dtype=np.bool_)
    restart_used = np.zeros(target.size, dtype=np.bool_)
    reasons: list[BootstrapTraceReason] = []
    for position in range(target.size):
        reason = _target_forced_reason(
            position=position,
            target_states=target,
            target_links=target_links,
        )
        if reason is None:
            restart_used[position] = True
            reason = (
                BootstrapTraceReason.RANDOM_RESTART
                if draws[position] < probability
                else BootstrapTraceReason.CONTINUATION
            )
        restarts[position] = reason is not BootstrapTraceReason.CONTINUATION
        reasons.append(reason)
    _make_read_only(restarts)
    _make_read_only(restart_used)
    return IdealRestartTrace(
        restart_flags=restarts,
        reasons=tuple(reasons),
        restart_uniform_used=restart_used,
        restart_uniforms_consumed=int(target.size),
    )


def conditioned_source_index_trace(
    target_states: ArrayLike,
    historical_source_states: ArrayLike,
    source_continuation_links: ArrayLike,
    restart_uniforms: ArrayLike,
    source_rank_uniforms: ArrayLike,
    *,
    restart_probability: float,
    target_continuation_links: ArrayLike | None = None,
) -> SourceIndexTrace:
    """Build the exact non-circular conditioned-bootstrap source-index trace.

    ``source_continuation_links[i]`` declares whether historical source row
    ``i + 1`` is a valid chronological continuation of row ``i``.  The vector
    therefore has exactly one fewer element than ``historical_source_states``.
    All uniform vectors must already contain one draw per target observation;
    every draw is counted as consumed even when its branch-specific value is
    deliberately ignored.
    """

    target = _state_vector(target_states, "target states")
    source = _state_vector(historical_source_states, "historical source states")
    links = _continuation_links(source_continuation_links, source.size)
    restart_draws = _uniform_vector(
        restart_uniforms,
        expected_size=target.size,
        label="restart uniforms",
    )
    rank_draws = _uniform_vector(
        source_rank_uniforms,
        expected_size=target.size,
        label="source-rank uniforms",
    )
    probability = _restart_probability(restart_probability)
    target_links = _optional_target_links(target_continuation_links, target.size)

    eligible: dict[int, IntArray] = {}
    for state in np.unique(target):
        indices = np.flatnonzero(source == state).astype(np.int64, copy=False)
        if indices.size == 0:
            raise ModelError(
                "target state has no eligible historical source index",
                details={"state": int(state)},
            )
        eligible[int(state)] = indices

    indices = np.empty(target.size, dtype=np.int64)
    restarts = np.zeros(target.size, dtype=np.bool_)
    restart_used = np.zeros(target.size, dtype=np.bool_)
    rank_used = np.zeros(target.size, dtype=np.bool_)
    reasons: list[BootstrapTraceReason] = []

    for position, state_value in enumerate(target):
        state = int(state_value)
        reason = _deterministic_reason(
            position=position,
            target_states=target,
            source_states=source,
            source_links=links,
            target_links=target_links,
            selected_indices=indices,
        )
        if reason is None:
            restart_used[position] = True
            if restart_draws[position] < probability:
                reason = BootstrapTraceReason.RANDOM_RESTART
            else:
                reason = BootstrapTraceReason.CONTINUATION

        restarted = reason is not BootstrapTraceReason.CONTINUATION
        restarts[position] = restarted
        if restarted:
            rank_used[position] = True
            candidates = eligible[state]
            rank = math.floor(float(rank_draws[position]) * candidates.size)
            indices[position] = candidates[rank]
        else:
            indices[position] = indices[position - 1] + 1
        reasons.append(reason)

    for array in (indices, restarts, restart_used, rank_used):
        _make_read_only(array)
    return SourceIndexTrace(
        source_indices=indices,
        restart_flags=restarts,
        reasons=tuple(reasons),
        restart_uniform_used=restart_used,
        source_rank_uniform_used=rank_used,
        restart_uniforms_consumed=int(target.size),
        source_rank_uniforms_consumed=int(target.size),
    )


def _deterministic_reason(
    *,
    position: int,
    target_states: IntArray,
    source_states: IntArray,
    source_links: BoolArray,
    target_links: BoolArray,
    selected_indices: IntArray,
) -> BootstrapTraceReason | None:
    """Return the first applicable forced-restart reason in policy order."""

    target_reason = _target_forced_reason(
        position=position,
        target_states=target_states,
        target_links=target_links,
    )
    if target_reason is not None:
        return target_reason
    next_source = int(selected_indices[position - 1]) + 1
    if next_source >= source_states.size:
        return BootstrapTraceReason.SOURCE_END
    if not source_links[next_source - 1]:
        return BootstrapTraceReason.SOURCE_GAP
    if source_states[next_source] != target_states[position]:
        return BootstrapTraceReason.SOURCE_STATE_MISMATCH
    return None


def _target_forced_reason(
    *,
    position: int,
    target_states: IntArray,
    target_links: BoolArray,
) -> BootstrapTraceReason | None:
    if position == 0:
        return BootstrapTraceReason.FIRST_OBSERVATION
    if not target_links[position - 1]:
        return BootstrapTraceReason.TARGET_SEGMENT_START
    if target_states[position] != target_states[position - 1]:
        return BootstrapTraceReason.TARGET_STATE_TRANSITION
    return None


def _categorical_index(probabilities: FloatArray, uniform: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if uniform < cumulative:
            return index
    return len(probabilities) - 1


def _validated_markov_inputs(
    stationary_distribution: ArrayLike,
    transition_matrix: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    try:
        transition = np.asarray(transition_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("transition matrix must be numeric") from error
    chain = validate_markov_chain(transition)
    stationary = _probability_vector(
        stationary_distribution,
        expected_size=transition.shape[0],
        label="stationary distribution",
    )
    residual = float(np.max(np.abs(stationary @ transition - stationary)))
    if residual > STATIONARY_RESIDUAL_ATOL:
        raise ModelError(
            "supplied stationary distribution is inconsistent with the chain",
            details={
                "residual": residual,
                "tolerance": STATIONARY_RESIDUAL_ATOL,
            },
        )
    difference = float(np.max(np.abs(stationary - chain.stationary_distribution)))
    if difference > PROBABILITY_ATOL:
        raise ModelError(
            "supplied stationary distribution differs from the validated chain",
            details={"maximum_absolute_difference": difference},
        )
    return transition, stationary


def _state_vector(values: ArrayLike, label: str) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ModelError(f"{label} must be a non-empty one-dimensional vector")
    if raw.dtype.kind not in {"i", "u"}:
        raise ModelError(f"{label} must have integer dtype")
    result = raw.astype(np.int64, copy=True)
    if bool((result < 0).any()):
        raise ModelError(f"{label} must contain non-negative labels")
    return result


def _continuation_links(values: ArrayLike, source_size: int) -> BoolArray:
    raw = np.asarray(values)
    expected = source_size - 1
    if raw.ndim != 1 or raw.size != expected:
        raise ModelError(
            "source continuation links must have historical length minus one",
            details={"actual": int(raw.size), "expected": expected},
        )
    if raw.size and raw.dtype.kind != "b":
        raise ModelError("source continuation links must have boolean dtype")
    return raw.astype(np.bool_, copy=True)


def _optional_target_links(values: ArrayLike | None, target_size: int) -> BoolArray:
    if values is None:
        return np.ones(max(0, target_size - 1), dtype=np.bool_)
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != max(0, target_size - 1):
        raise ModelError(
            "target continuation links must have target length minus one",
            details={"actual": int(raw.size), "expected": max(0, target_size - 1)},
        )
    if raw.size and raw.dtype.kind != "b":
        raise ModelError("target continuation links must have boolean dtype")
    return raw.astype(np.bool_, copy=True)


def _validated_segment_lengths(
    values: Sequence[int],
    *,
    expected_total: int | None,
    label: str,
) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ModelError(f"{label} must be a sequence of positive integers") from error
    if not raw:
        raise ModelError(f"{label} must be non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int | np.integer)
        for value in raw
    ):
        raise ModelError(f"{label} must contain integers")
    result = tuple(int(value) for value in raw)
    if any(value <= 0 for value in result):
        raise ModelError(f"{label} must contain positive lengths")
    total = sum(result)
    if expected_total is not None and total != expected_total:
        raise ModelError(
            f"{label} must sum to the state-vector length",
            details={"actual": total, "expected": expected_total},
        )
    return result


def _continuation_from_segment_lengths(lengths: tuple[int, ...]) -> BoolArray:
    total = sum(lengths)
    links = np.ones(max(0, total - 1), dtype=np.bool_)
    boundary = 0
    for length in lengths[:-1]:
        boundary += length
        links[boundary - 1] = False
    _make_read_only(links)
    return links


def _uniform_vector(
    values: ArrayLike, *, expected_size: int | None, label: str
) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ModelError(f"{label} must be one-dimensional")
    if raw.dtype.kind == "b":
        raise ModelError(f"{label} must be numeric, not boolean")
    try:
        result = np.asarray(values, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if expected_size is not None and result.size != expected_size:
        raise ModelError(
            f"{label} must contain one draw per target observation",
            details={"actual": int(result.size), "expected": expected_size},
        )
    if not np.isfinite(result).all() or bool(((result < 0.0) | (result >= 1.0)).any()):
        raise ModelError(f"{label} must be finite and in [0, 1)")
    return result


def _probability_vector(
    values: ArrayLike, *, expected_size: int, label: str
) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if result.ndim != 1 or result.size != expected_size:
        raise ModelError(
            f"{label} has incompatible shape",
            details={"actual": list(result.shape), "expected": [expected_size]},
        )
    if not np.isfinite(result).all():
        raise ModelError(f"{label} must be finite")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise ModelError(f"{label} must be in [0, 1]")
    probability_error = abs(float(result.sum()) - 1.0)
    if probability_error > PROBABILITY_ATOL:
        raise ModelError(
            f"{label} must sum to one",
            details={"absolute_sum_error": probability_error},
        )
    return result.astype(np.float64, copy=True)


def _restart_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ModelError("restart probability must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ModelError("restart probability must be finite and in (0, 1]")
    return result


def _make_read_only(array: NDArray[np.generic]) -> None:
    array.setflags(write=False)
