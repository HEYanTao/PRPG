"""Closed numerical-driver tests for the registered design fixed point."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import cast

import pytest
from test_g3_calibration_views import (
    _MASTER_SEED,
    _SCIENTIFIC_VERSION,
    _registered_iteration_suite,
    _registered_selection_bootstrap,
)

import prpg.model.candidate_fixed_point_execution as execution_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.candidate_fixed_point_execution import (
    execute_registered_design_fixed_point,
)
from prpg.model.candidate_suite import RegisteredDesignCandidateSuite
from prpg.model.fixed_point import FixedPointState
from prpg.model.input import CalibrationInput
from prpg.model.registered_fixed_point import (
    IterationSource,
    RegisteredDesignFixedPoint,
    is_registered_design_fixed_point,
    registered_design_fixed_point_sources,
)


def _common_from_suite(
    suite: RegisteredDesignCandidateSuite,
) -> execution_module._RegisteredCommonSources:
    return execution_module._RegisteredCommonSources(
        design_slice=suite._design_slice,
        design_features=suite._design_features,
        preliminary_outcome=suite._preliminary_outcome,
        preliminary=suite.preliminary,
        fit_set=suite._fit_set,
        rolling_bootstrap=_registered_selection_bootstrap(),
        macro_materialization=suite._macro_materialization,
        monthly_materialization=suite._monthly_materialization,
        daily_materialization=suite._daily_materialization,
        decoded_pools_by_k=tuple(row.decoded_return_pools for row in suite.rows),
        rolling_by_k=tuple(row.rolling for row in suite.rows),
        macro_by_k=tuple(row.macro_stability for row in suite.rows),
        master_seed=suite.master_seed,
        scientific_version=suite.scientific_version,
    )


def _placeholder_input() -> CalibrationInput:
    # Common-stage replacement means no field is read.  Keeping the exact type
    # still exercises the public boundary without constructing fake provenance.
    return object.__new__(CalibrationInput)


def test_driver_reuses_common_parents_and_converges_on_first_adjacent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_suite = _registered_iteration_suite(monkeypatch, fixed_k4=True)
    common = _common_from_suite(seed_suite)
    preparation_calls = 0
    starts: list[tuple[int, int]] = []
    original_blocks = execution_module._execute_registered_block_outcomes

    def prepare(
        _calibration_input: CalibrationInput,
        *,
        master_seed: int,
        scientific_version: int,
        workers: int,
    ) -> execution_module._RegisteredCommonSources:
        nonlocal preparation_calls
        preparation_calls += 1
        assert master_seed == _MASTER_SEED
        assert scientific_version == _SCIENTIFIC_VERSION
        assert workers == 1
        return common

    def blocks(
        supplied: execution_module._RegisteredCommonSources,
        *,
        monthly_start: int,
        daily_start: int,
    ) -> object:
        starts.append((monthly_start, daily_start))
        return original_blocks(
            supplied,
            monthly_start=monthly_start,
            daily_start=daily_start,
        )

    monkeypatch.setattr(
        execution_module,
        "_prepare_common_registered_sources",
        prepare,
    )
    monkeypatch.setattr(
        execution_module,
        "_execute_registered_block_outcomes",
        blocks,
    )

    result = execute_registered_design_fixed_point(
        _placeholder_input(),
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    sources, _monthly, _daily = registered_design_fixed_point_sources(result)

    assert is_registered_design_fixed_point(result)
    assert result.outcome == "converged"
    assert result.converged_at_iteration == 2
    assert preparation_calls == 1
    assert starts[0] == (
        common.preliminary.monthly.combined.starting_length,
        common.preliminary.daily.combined.starting_length,
    )
    first_state = result.iterations[0].state
    assert first_state is not None
    assert starts[1] == (first_state.monthly_length, first_state.daily_length)

    first_suite = sources[0][0]
    second_suite = sources[1][0]
    for suite, _viability, selection in sources:
        assert selection.selection.selection_policy == "owner_fixed_k4_v3"
        assert selection.selection.selected_n_states == 4
        assert suite._design_slice is common.design_slice
        assert suite._design_features is common.design_features
        assert suite._preliminary_outcome is common.preliminary_outcome
        assert suite._fit_set is common.fit_set
        assert suite._macro_materialization is common.macro_materialization
        assert suite._monthly_materialization is common.monthly_materialization
        assert suite._daily_materialization is common.daily_materialization
        assert (
            selection._registered_bootstrap_materialization is common.rolling_bootstrap
        )
        assert all(
            row.rolling is parent
            for row, parent in zip(suite.rows, common.rolling_by_k, strict=True)
        )
        assert all(
            row.macro_stability is parent
            for row, parent in zip(suite.rows, common.macro_by_k, strict=True)
        )
    assert first_suite.rows[2].monthly_block is not second_suite.rows[2].monthly_block
    assert first_suite.rows[2].daily_block is not second_suite.rows[2].daily_block


def _controlled_common(
    monthly_start: int = 7,
    daily_start: int = 30,
) -> execution_module._RegisteredCommonSources:
    preliminary = SimpleNamespace(
        monthly=SimpleNamespace(
            combined=SimpleNamespace(starting_length=monthly_start)
        ),
        daily=SimpleNamespace(combined=SimpleNamespace(starting_length=daily_start)),
    )
    value = object.__new__(execution_module._RegisteredCommonSources)
    object.__setattr__(value, "preliminary", preliminary)
    return value


def _controlled_source(iteration: int) -> IterationSource:
    # The registered assembler is replaced in state-machine-only tests; unique
    # placeholders let those tests inspect exact history length and ordering.
    return cast(
        IterationSource,
        (object(), object(), SimpleNamespace(iteration=iteration)),
    )


@pytest.mark.parametrize(
    ("states", "expected_iterations"),
    [
        (
            (
                FixedPointState(4, 6, 24),
                FixedPointState(4, 6, 24),
            ),
            2,
        ),
        (
            (
                FixedPointState(4, 6, 24),
                FixedPointState(4, 5, 20),
                FixedPointState(4, 6, 24),
            ),
            3,
        ),
        ((None,), 1),
    ],
    ids=("adjacent_convergence", "oscillation", "no_admissible_terminal"),
)
def test_loop_stops_at_first_binding_terminal(
    monkeypatch: pytest.MonkeyPatch,
    states: tuple[FixedPointState | None, ...],
    expected_iterations: int,
) -> None:
    calls: list[tuple[int, int, int]] = []
    assembled: list[tuple[IterationSource, ...]] = []
    sentinel = object.__new__(RegisteredDesignFixedPoint)

    def iteration(
        _common: execution_module._RegisteredCommonSources,
        *,
        iteration: int,
        monthly_start: int,
        daily_start: int,
    ) -> execution_module._RegisteredIterationExecution:
        calls.append((iteration, monthly_start, daily_start))
        return execution_module._RegisteredIterationExecution(
            _controlled_source(iteration),
            states[iteration - 1],
        )

    def assemble(
        sources: tuple[IterationSource, ...],
    ) -> RegisteredDesignFixedPoint:
        assembled.append(sources)
        return sentinel

    monkeypatch.setattr(execution_module, "_execute_registered_iteration", iteration)
    monkeypatch.setattr(
        execution_module,
        "assemble_registered_design_fixed_point",
        assemble,
    )

    assert (
        execution_module._execute_registered_iteration_loop(_controlled_common())
        is sentinel
    )
    assert len(calls) == expected_iterations
    assert len(assembled) == 1
    assert len(assembled[0]) == expected_iterations
    if expected_iterations > 1:
        first = states[0]
        assert first is not None
        assert calls[1][1:] == (first.monthly_length, first.daily_length)


def test_loop_runs_all_five_distinct_states_without_favorable_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = tuple(FixedPointState(4, monthly, 20 + monthly) for monthly in range(2, 7))
    calls: list[int] = []
    assembled: list[tuple[IterationSource, ...]] = []
    sentinel = object.__new__(RegisteredDesignFixedPoint)

    def iteration(
        _common: execution_module._RegisteredCommonSources,
        *,
        iteration: int,
        monthly_start: int,
        daily_start: int,
    ) -> execution_module._RegisteredIterationExecution:
        del monthly_start, daily_start
        calls.append(iteration)
        return execution_module._RegisteredIterationExecution(
            _controlled_source(iteration),
            states[iteration - 1],
        )

    def assemble(
        sources: tuple[IterationSource, ...],
    ) -> RegisteredDesignFixedPoint:
        assembled.append(sources)
        return sentinel

    monkeypatch.setattr(execution_module, "_execute_registered_iteration", iteration)
    monkeypatch.setattr(
        execution_module,
        "assemble_registered_design_fixed_point",
        assemble,
    )

    assert (
        execution_module._execute_registered_iteration_loop(_controlled_common())
        is sentinel
    )
    assert calls == [1, 2, 3, 4, 5]
    assert len(assembled[0]) == 5


@pytest.mark.parametrize(
    "error",
    (
        ModelError("generic model fault"),
        IntegrityError("integrity fault"),
        RuntimeError("runtime fault"),
        OSError("filesystem fault"),
        MemoryError("memory fault"),
    ),
    ids=("model", "integrity", "runtime", "filesystem", "memory"),
)
def test_public_driver_propagates_generic_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(
        execution_module,
        "_prepare_common_registered_sources",
        fail,
    )
    with pytest.raises(type(error), match=str(error)):
        execute_registered_design_fixed_point(
            _placeholder_input(),
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )


def test_iteration_fault_propagates_without_minting_partial_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembled = False

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("iteration implementation fault")

    def assemble(*_args: object, **_kwargs: object) -> object:
        nonlocal assembled
        assembled = True
        return object()

    monkeypatch.setattr(execution_module, "_execute_registered_iteration", fail)
    monkeypatch.setattr(
        execution_module,
        "assemble_registered_design_fixed_point",
        assemble,
    )

    with pytest.raises(RuntimeError, match="iteration implementation fault"):
        execution_module._execute_registered_iteration_loop(_controlled_common())
    assert not assembled


def test_public_driver_exposes_no_scientific_injection_or_reduced_geometry() -> None:
    signature = inspect.signature(execute_registered_design_fixed_point)

    assert tuple(signature.parameters) == (
        "calibration_input",
        "master_seed",
        "scientific_version",
        "workers",
    )
    assert all(
        name not in signature.parameters
        for name in (
            "fit_function",
            "handler_map",
            "history_count",
            "strict_canonical",
            "maximum_iterations",
        )
    )


@pytest.mark.parametrize("workers", (0, -1, True, 1.5))
def test_public_driver_rejects_invalid_execution_workers(workers: object) -> None:
    with pytest.raises(ModelError, match="workers must be a positive integer"):
        execute_registered_design_fixed_point(
            _placeholder_input(),
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
            workers=workers,  # type: ignore[arg-type]
        )
