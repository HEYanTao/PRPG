from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from tests.unit.test_g3_calibration_views import _candidate_views, _slices

import prpg.model.benchmarks as benchmark_module
import prpg.model.g3_calibration_views as views_module
import prpg.model.registered_holdout_execution as execution_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.benchmarks import (
    InvalidHoldoutBenchmark,
    InvalidHoldoutBenchmarkEvidence,
    fit_ridge_var1_benchmark,
)
from prpg.model.registered_holdout_execution import (
    RegisteredSealedHoldoutEvaluation,
    RegisteredSealedHoldoutInvalidBenchmark,
    execute_registered_sealed_holdout,
    is_registered_sealed_holdout_evaluation,
    is_registered_sealed_holdout_invalid_benchmark,
    registered_sealed_holdout_evaluation_identity,
    registered_sealed_holdout_evaluation_sources,
    registered_sealed_holdout_invalid_benchmark_identity,
    registered_sealed_holdout_invalid_benchmark_sources,
)


def _authorize_reduced_selection_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in only for the fixed-point wrapper in focused producer tests."""

    monkeypatch.setattr(
        views_module,
        "is_registered_candidate_selection_task_view",
        views_module.is_candidate_selection_task_view,
    )


def _sources(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    _authorize_reduced_selection_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    design, holdout, _ = _slices()
    return selection, design, holdout


def _invalid_benchmark_signal() -> InvalidHoldoutBenchmark:
    return InvalidHoldoutBenchmark(
        InvalidHoldoutBenchmarkEvidence(
            failure_code="var_spectral_radius_invalid",
            failure_message="VAR benchmark spectral radius is not below 0.995",
            failure_details=(("spectral_radius", 0.999),),
        )
    )


def test_registered_holdout_retains_exact_sources_and_read_only_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, design, holdout = _sources(monkeypatch)
    result = execute_registered_sealed_holdout(selection, design, holdout)

    assert isinstance(result, RegisteredSealedHoldoutEvaluation)
    assert is_registered_sealed_holdout_evaluation(result)
    identity = registered_sealed_holdout_evaluation_identity(result)
    assert identity.evidence_fingerprint == result.evidence_fingerprint
    assert registered_sealed_holdout_evaluation_sources(result) == (
        selection,
        result._selected_fit_source,
        design,
        holdout,
        result.evidence,
    )
    assert result.generation_authorized is False
    assert not result.evidence.hmm_log_scores.flags.writeable
    assert not result.evidence.benchmark_scores.gaussian.flags.writeable
    assert not result.evidence.benchmark_scores.ridge_var1.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.evidence.hmm_log_scores[0] = 0.0


def test_valid_benchmark_veto_loss_is_not_an_invalid_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, design, holdout = _sources(monkeypatch)
    result = execute_registered_sealed_holdout(selection, design, holdout)

    assert isinstance(result, RegisteredSealedHoldoutEvaluation)
    assert result.evidence.decision.passed is False
    assert not is_registered_sealed_holdout_invalid_benchmark(result)


def test_invalid_benchmark_is_the_only_caught_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, design, holdout = _sources(monkeypatch)
    calls: list[tuple[object, object, object]] = []

    def invalid(fit: object, design_slice: object, holdout_slice: object) -> None:
        calls.append((fit, design_slice, holdout_slice))
        raise _invalid_benchmark_signal()

    monkeypatch.setattr(execution_module, "evaluate_sealed_holdout", invalid)
    result = execute_registered_sealed_holdout(selection, design, holdout)

    assert isinstance(result, RegisteredSealedHoldoutInvalidBenchmark)
    assert len(calls) == 1
    assert calls[0][1:] == (design, holdout)
    assert result.failure_code == "var_spectral_radius_invalid"
    assert result.failure_details == (("spectral_radius", 0.999),)
    assert result.generation_authorized is False
    assert is_registered_sealed_holdout_invalid_benchmark(result)
    identity = registered_sealed_holdout_invalid_benchmark_identity(result)
    assert identity.failure_evidence_fingerprint == result.failure_evidence_fingerprint
    assert registered_sealed_holdout_invalid_benchmark_sources(result) == (
        selection,
        result._selected_fit_source,
        design,
        holdout,
    )


@pytest.mark.parametrize(
    "error",
    (
        ModelError("generic model fault"),
        IntegrityError("integrity fault"),
        RuntimeError("programming fault"),
        OSError("filesystem fault"),
        MemoryError("memory fault"),
    ),
)
def test_generic_and_operational_faults_propagate_without_scientific_outcome(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    selection, design, holdout = _sources(monkeypatch)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(execution_module, "evaluate_sealed_holdout", fail)
    with pytest.raises(type(error), match=str(error)):
        execute_registered_sealed_holdout(selection, design, holdout)


def test_legacy_selection_is_rejected_before_holdout_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    design, holdout, _ = _slices()
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("evaluation must remain locked")

    monkeypatch.setattr(execution_module, "evaluate_sealed_holdout", forbidden)
    with pytest.raises(ModelError, match="final fixed-point"):
        execute_registered_sealed_holdout(selection, design, holdout)
    assert called is False


def test_holdout_outcome_copy_and_public_tampering_cannot_mint_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, design, holdout = _sources(monkeypatch)
    result = execute_registered_sealed_holdout(selection, design, holdout)
    assert isinstance(result, RegisteredSealedHoldoutEvaluation)

    copied = copy.copy(result)
    assert not is_registered_sealed_holdout_evaluation(copied)
    object.__setattr__(result, "evidence_fingerprint", "0" * 64)
    assert not is_registered_sealed_holdout_evaluation(result)


def test_invalid_outcome_copy_and_public_tampering_cannot_mint_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, design, holdout = _sources(monkeypatch)

    def invalid(*_args: object, **_kwargs: object) -> None:
        raise _invalid_benchmark_signal()

    monkeypatch.setattr(execution_module, "evaluate_sealed_holdout", invalid)
    result = execute_registered_sealed_holdout(selection, design, holdout)
    assert isinstance(result, RegisteredSealedHoldoutInvalidBenchmark)

    copied = copy.copy(result)
    assert not is_registered_sealed_holdout_invalid_benchmark(copied)
    object.__setattr__(result, "failure_code", "var_score_invalid")
    assert not is_registered_sealed_holdout_invalid_benchmark(result)


def test_ridge_linear_algebra_failure_is_narrowly_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(17).normal(size=(193, 4))
    continuation = np.r_[False, np.ones(192, dtype=np.bool_)]

    def fail_solve(*_args: object, **_kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("synthetic numerical solve failure")

    monkeypatch.setattr(benchmark_module.np.linalg, "solve", fail_solve)
    with pytest.raises(InvalidHoldoutBenchmark) as caught:
        fit_ridge_var1_benchmark(values, continuation)
    assert caught.value.evidence.failure_code == "var_ridge_solve_failed"


def test_ridge_generic_fault_and_bad_input_are_not_scientific_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(18).normal(size=(193, 4))
    continuation = np.r_[False, np.ones(192, dtype=np.bool_)]

    def generic(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("synthetic implementation fault")

    monkeypatch.setattr(benchmark_module.np.linalg, "solve", generic)
    with pytest.raises(RuntimeError, match="implementation fault"):
        fit_ridge_var1_benchmark(values, continuation)

    with pytest.raises(ModelError) as bad_input:
        fit_ridge_var1_benchmark(np.ones((1, 4)), np.asarray([False]))
    assert not isinstance(bad_input.value, InvalidHoldoutBenchmark)


def test_ridge_eigenvalue_numerical_failure_is_narrowly_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(19).normal(size=(193, 4))
    continuation = np.r_[False, np.ones(192, dtype=np.bool_)]

    def fail_eigenvalues(*_args: object, **_kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("synthetic eigenvalue failure")

    monkeypatch.setattr(benchmark_module.np.linalg, "eigvals", fail_eigenvalues)
    with pytest.raises(InvalidHoldoutBenchmark) as caught:
        fit_ridge_var1_benchmark(values, continuation)
    assert caught.value.evidence.failure_code == "var_eigenvalues_nonfinite"
