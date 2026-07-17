from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.benchmarks import (
    HoldoutBenchmarkScores,
    fit_gaussian_benchmark,
    fit_ridge_var1_benchmark,
    holdout_veto_decision,
    score_holdout_benchmarks,
)


def _stable_history(rows: int = 120) -> np.ndarray:
    rng = np.random.default_rng(20260714)
    values = np.empty((rows, 2), dtype=np.float64)
    values[0] = [0.2, -0.1]
    coefficients = np.asarray([[0.55, 0.10], [-0.05, 0.40]])
    intercept = np.asarray([0.03, -0.02])
    for row in range(1, rows):
        values[row] = (
            intercept + coefficients @ values[row - 1] + rng.normal(0.0, 0.15, 2)
        )
    return values


def test_gaussian_benchmark_uses_population_covariance_and_floor() -> None:
    values = np.asarray([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])

    result = fit_gaussian_benchmark(values)

    np.testing.assert_allclose(result.mean, [1.0, 2.0])
    np.testing.assert_allclose(result.covariance[0, 0], 2.0 / 3.0)
    assert np.linalg.eigvalsh(result.covariance).min() == pytest.approx(1e-6)
    assert result.condition_number <= 1e6
    assert not result.covariance.flags.writeable


def test_ridge_var_uses_only_valid_links_and_solves_stationary_law() -> None:
    values = _stable_history()
    continuation = np.ones(len(values), dtype=np.bool_)
    continuation[[0, 60]] = False

    result = fit_ridge_var1_benchmark(values, continuation)

    assert result.training_links == len(values) - 2
    assert result.spectral_radius < 0.995
    assert result.residual_condition_number <= 1e6
    assert result.stationary_condition_number <= 1e6
    np.testing.assert_allclose(
        result.stationary_mean,
        np.linalg.solve(np.eye(2) - result.coefficients, result.intercept),
    )
    np.testing.assert_allclose(
        result.stationary_covariance,
        result.coefficients @ result.stationary_covariance @ result.coefficients.T
        + result.residual_covariance,
        atol=1e-12,
    )


def test_holdout_scoring_uses_design_terminal_and_stationary_gap_reset() -> None:
    training = _stable_history()
    holdout = np.asarray([[0.10, -0.05], [0.15, -0.02], [1.00, 0.50], [0.75, 0.20]])
    continuation = np.asarray([False, True, False, True])

    first = score_holdout_benchmarks(training, holdout, continuation)
    changed_training = training.copy()
    changed_training[-1] += 5.0
    second = score_holdout_benchmarks(changed_training, holdout, continuation)
    assert first.ridge_var1[0] != second.ridge_var1[0]

    changed_holdout = holdout.copy()
    changed_holdout[1] += 100.0
    third = score_holdout_benchmarks(training, changed_holdout, continuation)
    assert first.ridge_var1[2] == pytest.approx(third.ridge_var1[2])
    assert first.gaussian.shape == first.ridge_var1.shape == (4,)
    assert not first.ridge_var1.flags.writeable


@pytest.mark.parametrize(
    ("hmm", "expected"),
    [
        ([0.0, 0.0], True),
        ([-0.1, 0.0], False),
        ([0.0, -0.1], False),
    ],
)
def test_holdout_veto_requires_both_nonnegative_differences(
    hmm: list[float], expected: bool
) -> None:
    benchmarks = HoldoutBenchmarkScores(
        gaussian=np.asarray([0.0, 0.0]),
        ridge_var1=np.asarray([0.0, 0.0]),
    )

    result = holdout_veto_decision(hmm, benchmarks)

    assert result.passed is expected


def test_holdout_veto_accepts_exact_equality_and_rejects_shape_mismatch() -> None:
    benchmarks = HoldoutBenchmarkScores(
        gaussian=np.asarray([1.0, 1.0]),
        ridge_var1=np.asarray([0.5, 0.5]),
    )
    result = holdout_veto_decision([1.0, 1.0], benchmarks)
    assert result.passed
    assert result.hmm_minus_gaussian == 0.0
    with pytest.raises(ModelError, match="identical shapes"):
        holdout_veto_decision([1.0], benchmarks)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fit_gaussian_benchmark([[1.0]]),
        lambda: fit_gaussian_benchmark([[1.0], [np.nan]]),
        lambda: fit_ridge_var1_benchmark([[0.0], [1.0]], np.asarray([False, False])),
        lambda: score_holdout_benchmarks(
            [[0.0], [1.0]], [[0.0], [1.0]], np.asarray([True, True])
        ),
    ],
)
def test_benchmark_inputs_fail_closed(call: object) -> None:
    with pytest.raises(ModelError):
        call()  # type: ignore[operator]
