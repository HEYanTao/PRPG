"""Tests for the binding design HMM/block fixed-point state machine."""

from __future__ import annotations

from dataclasses import replace

import pytest

from prpg.errors import ModelError
from prpg.model.fixed_point import (
    CandidateSelectionSnapshot,
    FixedPointIteration,
    FixedPointState,
    execute_fixed_point_iteration,
    run_design_fixed_point,
)


def _record(
    iteration: int,
    input_state: FixedPointState | None,
    state: FixedPointState,
    admissible: tuple[int, ...] = (4,),
) -> FixedPointIteration:
    snapshot = CandidateSelectionSnapshot(
        admissible,
        state.selected_n_states,
    )
    return FixedPointIteration(
        iteration,
        input_state,
        state.monthly_length,
        state.daily_length,
        snapshot,
        state.monthly_length,
        state.daily_length,
        snapshot,
        state,
    )


def test_iteration_structurally_orders_preliminary_finalize_and_rerun() -> None:
    calls: list[str] = []

    def preliminary(monthly: int, daily: int) -> CandidateSelectionSnapshot:
        calls.append(f"preliminary:{monthly}:{daily}")
        return CandidateSelectionSnapshot((2, 4), 4)

    def finalize(selected: int, monthly: int, daily: int) -> tuple[int, int]:
        calls.append(f"finalize:{selected}:{monthly}:{daily}")
        return 5, 20

    def final(monthly: int, daily: int) -> CandidateSelectionSnapshot:
        calls.append(f"final:{monthly}:{daily}")
        return CandidateSelectionSnapshot((4,), 4)

    result = execute_fixed_point_iteration(
        iteration=1,
        input_state=None,
        preliminary_monthly_length=6,
        preliminary_daily_length=24,
        preliminary_support_and_selection=preliminary,
        finalize_selected_lengths=finalize,
        final_support_and_selection=final,
    )
    assert calls == ["preliminary:6:24", "finalize:4:6:24", "final:5:20"]
    assert result.state == FixedPointState(4, 5, 20)


def test_two_consecutive_identical_states_converge() -> None:
    state = FixedPointState(4, 5, 20)
    result = run_design_fixed_point(
        lambda iteration, previous: _record(iteration, previous, state)
    )
    assert result.converged_at_iteration == 2
    assert len(result.iterations) == 2
    assert result.state == state


def test_report_only_non_k_admissibility_does_not_affect_convergence() -> None:
    state = FixedPointState(4, 5, 20)
    report_sets = ((2, 4), (3, 4, 5))

    result = run_design_fixed_point(
        lambda iteration, previous: _record(
            iteration,
            previous,
            state,
            report_sets[iteration - 1],
        )
    )

    assert result.converged_at_iteration == 2
    assert result.state == FixedPointState(4, 5, 20)


def test_nonadjacent_repeat_is_oscillation_not_favorable_selection() -> None:
    states = (
        FixedPointState(4, 5, 20),
        FixedPointState(4, 4, 18),
        FixedPointState(4, 5, 20),
    )

    def runner(iteration: int, previous: FixedPointState | None) -> FixedPointIteration:
        return _record(iteration, previous, states[iteration - 1])

    with pytest.raises(ModelError, match="oscillated"):
        run_design_fixed_point(runner)


def test_five_distinct_iterations_fail_closed() -> None:
    states = (
        FixedPointState(4, 2, 10),
        FixedPointState(4, 3, 11),
        FixedPointState(4, 4, 12),
        FixedPointState(4, 5, 13),
        FixedPointState(4, 6, 14),
    )
    with pytest.raises(ModelError, match="five iterations"):
        run_design_fixed_point(
            lambda iteration, previous: _record(
                iteration,
                previous,
                states[iteration - 1],
            )
        )


def test_runner_identity_and_state_cross_checks_fail_closed() -> None:
    state = FixedPointState(4, 4, 12)
    with pytest.raises(ModelError, match="identity/input"):
        run_design_fixed_point(
            lambda iteration, _previous: _record(iteration, state, state)
        )
    malformed = replace(_record(1, None, state), finalized_monthly_length=3)
    with pytest.raises(ModelError, match="does not match"):
        run_design_fixed_point(lambda _iteration, _previous: malformed)


def test_finalizer_cannot_increase_or_return_invalid_length() -> None:
    snapshot = CandidateSelectionSnapshot((4,), 4)
    with pytest.raises(ModelError, match="may not exceed"):
        execute_fixed_point_iteration(
            iteration=1,
            input_state=None,
            preliminary_monthly_length=5,
            preliminary_daily_length=20,
            preliminary_support_and_selection=lambda *_: snapshot,
            finalize_selected_lengths=lambda *_: (6, 20),
            final_support_and_selection=lambda *_: snapshot,
        )
    with pytest.raises(ModelError, match="outside"):
        execute_fixed_point_iteration(
            iteration=1,
            input_state=None,
            preliminary_monthly_length=5,
            preliminary_daily_length=20,
            preliminary_support_and_selection=lambda *_: snapshot,
            finalize_selected_lengths=lambda *_: (1, 20),
            final_support_and_selection=lambda *_: snapshot,
        )
