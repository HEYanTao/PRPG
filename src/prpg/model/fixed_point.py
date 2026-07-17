"""Fail-closed design HMM/block fixed-point state machine.

The binding Phase-3 order is structural here rather than a caller-supplied
flag: preliminary PPW starts are support-filtered and selected, block lengths
are finalized only for that result, and every support filter plus selection is
then rerun at the finalized lengths.  The resulting version-3 tuple

``(fixed K=4, monthly L, daily L)``

must be identical in two consecutive iterations.  A repeated non-adjacent
state is an oscillation, and failure to converge by iteration five is fatal.
The K=2,3,5 viability family remains report evidence and cannot alter state
equality.  No API exposes a way to choose a favorable prior iteration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from prpg.errors import ModelError
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.selection import OWNER_FIXED_N_STATES

VERSION_1_CANDIDATES = (2, 3, 4, 5)
MAXIMUM_FIXED_POINT_ITERATIONS = 5
OWNER_FIXED_K4_FIXED_POINT_STATE_CONTRACT = "owner-fixed-k4-lmonthly-ldaily-v1"


@dataclass(frozen=True, slots=True)
class CandidateSelectionSnapshot:
    """Compact result of one complete support-filter and selection pass."""

    admissible_n_states: tuple[int, ...]
    selected_n_states: int


@dataclass(frozen=True, slots=True)
class FixedPointState:
    """The exact owner-fixed K4/block-length convergence tuple."""

    selected_n_states: int
    monthly_length: int
    daily_length: int


@dataclass(frozen=True, slots=True)
class FixedPointIteration:
    """One fully ordered preliminary/finalized calibration iteration."""

    iteration: int
    input_state: FixedPointState | None
    preliminary_monthly_length: int
    preliminary_daily_length: int
    preliminary_selection: CandidateSelectionSnapshot
    finalized_monthly_length: int
    finalized_daily_length: int
    final_selection: CandidateSelectionSnapshot
    state: FixedPointState


@dataclass(frozen=True, slots=True)
class DesignFixedPointResult:
    """Converged consecutive state and every retained iteration."""

    state: FixedPointState
    iterations: tuple[FixedPointIteration, ...]
    converged_at_iteration: int
    maximum_iterations: int


SupportRunner: TypeAlias = Callable[[int, int], CandidateSelectionSnapshot]
LengthFinalizer: TypeAlias = Callable[[int, int, int], tuple[int, int]]
IterationRunner: TypeAlias = Callable[
    [int, FixedPointState | None], FixedPointIteration
]


def execute_fixed_point_iteration(
    *,
    iteration: int,
    input_state: FixedPointState | None,
    preliminary_monthly_length: int,
    preliminary_daily_length: int,
    preliminary_support_and_selection: SupportRunner,
    finalize_selected_lengths: LengthFinalizer,
    final_support_and_selection: SupportRunner,
) -> FixedPointIteration:
    """Execute the binding order once using caller-owned scientific workers."""

    checked_iteration = _iteration(iteration)
    if input_state is not None:
        _state(input_state)
    monthly_start = _length(preliminary_monthly_length, "monthly")
    daily_start = _length(preliminary_daily_length, "daily")
    preliminary = _snapshot(
        preliminary_support_and_selection(monthly_start, daily_start)
    )
    finalized = finalize_selected_lengths(
        preliminary.selected_n_states,
        monthly_start,
        daily_start,
    )
    if not isinstance(finalized, tuple) or len(finalized) != 2:
        raise ModelError("fixed-point length finalizer must return a two-tuple")
    monthly_final = _length(finalized[0], "monthly")
    daily_final = _length(finalized[1], "daily")
    if monthly_final > monthly_start or daily_final > daily_start:
        raise ModelError(
            "finalized block lengths may not exceed preliminary PPW starts"
        )
    final = _snapshot(final_support_and_selection(monthly_final, daily_final))
    state = FixedPointState(
        final.selected_n_states,
        monthly_final,
        daily_final,
    )
    _state(state)
    return FixedPointIteration(
        checked_iteration,
        input_state,
        monthly_start,
        daily_start,
        preliminary,
        monthly_final,
        daily_final,
        final,
        state,
    )


def run_design_fixed_point(
    iteration_runner: IterationRunner,
) -> DesignFixedPointResult:
    """Require consecutive equality, rejecting oscillation or five-look drift."""

    if not callable(iteration_runner):
        raise ModelError("fixed-point iteration runner must be callable")
    records: list[FixedPointIteration] = []
    states: list[FixedPointState] = []
    previous: FixedPointState | None = None
    seen: dict[FixedPointState, int] = {}
    for iteration in range(1, MAXIMUM_FIXED_POINT_ITERATIONS + 1):
        record = iteration_runner(iteration, previous)
        _record(record, iteration=iteration, expected_input=previous)
        current = record.state
        records.append(record)
        states.append(current)
        if previous is not None and current == previous:
            return DesignFixedPointResult(
                current,
                tuple(records),
                iteration,
                MAXIMUM_FIXED_POINT_ITERATIONS,
            )
        if current in seen:
            raise ModelError(
                "design HMM/block fixed point oscillated",
                details={
                    "first_iteration": seen[current],
                    "repeated_iteration": iteration,
                    "states": [_state_dict(value) for value in states],
                },
            )
        seen[current] = iteration
        previous = current
    raise ModelError(
        "design HMM/block fixed point did not converge within five iterations",
        details={"states": [_state_dict(value) for value in states]},
    )


def _record(
    value: FixedPointIteration,
    *,
    iteration: int,
    expected_input: FixedPointState | None,
) -> None:
    if not isinstance(value, FixedPointIteration):
        raise ModelError("fixed-point runner returned the wrong record type")
    if value.iteration != iteration or value.input_state != expected_input:
        raise ModelError("fixed-point iteration identity/input state is inconsistent")
    preliminary = _snapshot(value.preliminary_selection)
    final = _snapshot(value.final_selection)
    monthly_start = _length(value.preliminary_monthly_length, "monthly")
    daily_start = _length(value.preliminary_daily_length, "daily")
    monthly_final = _length(value.finalized_monthly_length, "monthly")
    daily_final = _length(value.finalized_daily_length, "daily")
    if monthly_final > monthly_start or daily_final > daily_start:
        raise ModelError("fixed-point record increased a preliminary block length")
    expected = FixedPointState(
        final.selected_n_states,
        monthly_final,
        daily_final,
    )
    if value.state != expected:
        raise ModelError("fixed-point state does not match its final support rerun")
    if preliminary != value.preliminary_selection:
        raise ModelError("fixed-point preliminary selection is not canonical")
    _state(value.state)


def _snapshot(value: CandidateSelectionSnapshot) -> CandidateSelectionSnapshot:
    if not isinstance(value, CandidateSelectionSnapshot):
        raise ModelError("support/selection runner returned the wrong record type")
    admissible = _admissible(value.admissible_n_states)
    if value.selected_n_states not in admissible:
        raise ModelError("selected K must be in the admissible fixed-point set")
    if value.selected_n_states != OWNER_FIXED_N_STATES:
        raise ModelError("fixed-point selection must select owner-fixed K=4")
    return CandidateSelectionSnapshot(admissible, value.selected_n_states)


def _state(value: FixedPointState) -> FixedPointState:
    if not isinstance(value, FixedPointState):
        raise ModelError("fixed-point state has the wrong record type")
    if value.selected_n_states != OWNER_FIXED_N_STATES:
        raise ModelError("fixed-point state must retain owner-fixed K=4")
    monthly = _length(value.monthly_length, "monthly")
    daily = _length(value.daily_length, "daily")
    return FixedPointState(
        value.selected_n_states,
        monthly,
        daily,
    )


def _admissible(values: Sequence[int]) -> tuple[int, ...]:
    supplied = tuple(values)
    if (
        not supplied
        or tuple(sorted(set(supplied))) != supplied
        or any(value not in VERSION_1_CANDIDATES for value in supplied)
    ):
        raise ModelError(
            "fixed-point admissible K set must be a sorted non-empty subset of 2..5"
        )
    return supplied


def _length(value: int, frequency: str) -> int:
    bounds = (
        MONTHLY_BLOCK_LENGTH_BOUNDS
        if frequency == "monthly"
        else DAILY_BLOCK_LENGTH_BOUNDS
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not bounds[0] <= value <= bounds[1]
    ):
        raise ModelError(f"fixed-point {frequency} length is outside {bounds}")
    return value


def _iteration(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAXIMUM_FIXED_POINT_ITERATIONS
    ):
        raise ModelError("fixed-point iteration must be in [1, 5]")
    return value


def _state_dict(value: FixedPointState) -> dict[str, object]:
    return {
        "selected_n_states": value.selected_n_states,
        "monthly_length": value.monthly_length,
        "daily_length": value.daily_length,
    }
