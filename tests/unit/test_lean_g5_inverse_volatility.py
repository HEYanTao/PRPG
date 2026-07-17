from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from prpg.errors import ValidationError
from prpg.validation.lean_g5_inverse_volatility import (
    fit_lean_g5_inverse_volatility_model,
    lean_g5_inverse_volatility_weights,
    score_lean_g5_inverse_volatility_path,
    score_lean_g5_inverse_volatility_paths,
)

FloatArray = NDArray[np.float64]


def _return_paths(
    seed: int,
    *,
    paths: int,
    observations: int,
) -> tuple[FloatArray, ...]:
    generator = np.random.default_rng(seed)
    result: list[FloatArray] = []
    for path_index in range(paths):
        innovations = generator.normal(
            loc=np.asarray((0.0007, 0.00025, 0.00035)),
            scale=np.asarray((0.018, 0.006, 0.008)),
            size=(observations, 3),
        )
        simple = np.empty_like(innovations)
        simple[0] = innovations[0]
        for row in range(1, observations):
            simple[row] = (
                innovations[row]
                + 0.15 * simple[row - 1]
                + math.sin((row + path_index) / 9.0)
                * np.asarray((0.001, -0.0004, 0.0006))
            )
        result.append(np.log1p(simple))
    return tuple(result)


@pytest.fixture(scope="module")
def monthly_development() -> tuple[FloatArray, ...]:
    return _return_paths(910, paths=100, observations=36)


@pytest.fixture(scope="module")
def monthly_model(monthly_development):  # type: ignore[no-untyped-def]
    return fit_lean_g5_inverse_volatility_model(
        monthly_development, frequency="monthly"
    )


@pytest.fixture(scope="module")
def daily_development() -> tuple[FloatArray, ...]:
    return _return_paths(911, paths=20, observations=110)


@pytest.fixture(scope="module")
def daily_model(daily_development):  # type: ignore[no-untyped-def]
    return fit_lean_g5_inverse_volatility_model(daily_development, frequency="daily")


def test_monthly_fit_is_only_the_development_average_comparator(
    monthly_model,  # type: ignore[no-untyped-def]
    monthly_development: tuple[FloatArray, ...],
) -> None:
    all_weights = np.concatenate(
        tuple(
            lean_g5_inverse_volatility_weights(monthly_model, path).values
            for path in monthly_development
        ),
        axis=0,
    )
    expected = np.mean(all_weights, axis=0, dtype=np.float64)
    expected /= np.sum(expected)

    assert monthly_model.frequency == "monthly"
    assert monthly_model.lookback_periods == 6
    assert monthly_model.rebalance_periods == 1
    assert monthly_model.annualization == 12
    assert monthly_model.development_paths == 100
    assert monthly_model.development_weight_rows == 3_000
    np.testing.assert_allclose(
        monthly_model.comparator_weights, expected, rtol=0.0, atol=2.0e-15
    )
    assert np.all(monthly_model.comparator_weights > 0.0)
    assert np.sum(monthly_model.comparator_weights) == pytest.approx(1.0)
    assert not monthly_model.comparator_weights.flags.writeable


def test_first_timer_decision_is_hand_calculated_from_prior_simple_returns(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    signs = np.asarray((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    simple = np.zeros((12, 3), dtype=np.float64)
    simple[:6] = signs[:, np.newaxis] * np.asarray((0.01, 0.02, 0.04))
    simple[6:] = _return_paths(12, paths=1, observations=6)[0]
    # The helper returned log returns; convert only the appended rows back to
    # simple before forming one consistent log-return path.
    simple[6:] = np.expm1(simple[6:])

    weights = lean_g5_inverse_volatility_weights(monthly_model, np.log1p(simple))

    assert weights.first_return_index == 6
    np.testing.assert_allclose(
        weights.values[0],
        np.asarray((4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0)),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(weights.values, axis=1), np.ones(weights.values.shape[0]), atol=1.0e-15
    )
    assert np.all(weights.values > 0.0)


def test_timer_never_observes_the_return_on_which_its_weight_is_used(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    original = _return_paths(913, paths=1, observations=30)[0]
    changed = original.copy()
    changed[15:] += np.asarray((0.7, -0.5, 0.4))

    original_weights = lean_g5_inverse_volatility_weights(monthly_model, original)
    changed_weights = lean_g5_inverse_volatility_weights(monthly_model, changed)

    # The weight applied to row 15 sees rows 9..14.  Row 15 can first affect
    # the weight applied to row 16.
    through_changed_row = 15 - original_weights.first_return_index + 1
    np.testing.assert_array_equal(
        original_weights.values[:through_changed_row],
        changed_weights.values[:through_changed_row],
    )


def test_daily_timer_uses_63_sessions_and_holds_for_21(
    daily_model,  # type: ignore[no-untyped-def]
    daily_development: tuple[FloatArray, ...],
) -> None:
    weights = lean_g5_inverse_volatility_weights(daily_model, daily_development[0])

    assert daily_model.lookback_periods == 63
    assert daily_model.rebalance_periods == 21
    assert daily_model.annualization == 252
    assert daily_model.development_paths == 20
    assert weights.first_return_index == 63
    for start in range(0, weights.values.shape[0], 21):
        held = weights.values[start : start + 21]
        np.testing.assert_array_equal(held, np.repeat(held[:1], held.shape[0], axis=0))


def test_path_score_matches_two_hand_calculated_zero_cash_sharpes(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    path = _return_paths(914, paths=1, observations=80)[0]
    score = score_lean_g5_inverse_volatility_path(monthly_model, path)
    weights = lean_g5_inverse_volatility_weights(monthly_model, path)
    simple = np.expm1(path[weights.first_return_index :])
    timer_returns = np.einsum("ta,ta->t", weights.values, simple)
    comparator_returns = simple @ monthly_model.comparator_weights
    timer_expected = (
        math.sqrt(12.0)
        * float(np.mean(timer_returns))
        / float(np.std(timer_returns, ddof=1))
    )
    comparator_expected = (
        math.sqrt(12.0)
        * float(np.mean(comparator_returns))
        / float(np.std(comparator_returns, ddof=1))
    )

    assert score.frequency == "monthly"
    assert score.observations == 74
    assert score.timer_sharpe == pytest.approx(timer_expected)
    assert score.comparator_sharpe == pytest.approx(comparator_expected)
    assert score.improvement == pytest.approx(timer_expected - comparator_expected)


def test_identical_scorer_retains_every_historical_or_simulated_path_row(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    paths = _return_paths(915, paths=7, observations=72)

    rows = score_lean_g5_inverse_volatility_paths(monthly_model, paths)
    expected = np.asarray(
        tuple(
            score_lean_g5_inverse_volatility_path(monthly_model, path).improvement
            for path in paths
        )
    )

    assert rows.frequency == "monthly"
    assert rows.paths == 7
    np.testing.assert_array_equal(rows.improvements, expected)
    np.testing.assert_array_equal(
        rows.improvements,
        score_lean_g5_inverse_volatility_paths(monthly_model, paths).improvements,
    )
    assert np.any(rows.improvements != 0.0)
    assert not rows.improvements.flags.writeable


def test_registered_development_counts_and_equal_history_are_binding() -> None:
    with pytest.raises(ValidationError, match="path count"):
        fit_lean_g5_inverse_volatility_model(
            _return_paths(1, paths=99, observations=24), frequency="monthly"
        )

    unequal = list(_return_paths(2, paths=100, observations=24))
    unequal[-1] = unequal[-1][:-1]
    with pytest.raises(ValidationError, match="equal history"):
        fit_lean_g5_inverse_volatility_model(unequal, frequency="monthly")


def test_degenerate_or_nonfinite_inputs_fail_instead_of_fabricating_a_score(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    constant = tuple(np.zeros((20, 3), dtype=np.float64) for _ in range(100))
    with pytest.raises(ValidationError, match="volatility is degenerate"):
        fit_lean_g5_inverse_volatility_model(constant, frequency="monthly")

    nonfinite = _return_paths(3, paths=1, observations=20)[0]
    nonfinite[10, 1] = np.nan
    with pytest.raises(ValidationError, match="must be finite"):
        score_lean_g5_inverse_volatility_path(monthly_model, nonfinite)

    overflow = _return_paths(4, paths=1, observations=20)[0]
    overflow[10, 0] = 1_000.0
    with pytest.raises(ValidationError, match="simple returns are non-finite"):
        score_lean_g5_inverse_volatility_path(monthly_model, overflow)


def test_degenerate_sharpe_and_invalid_model_fail_closed(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    signs = np.asarray((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    simple = np.empty((8, 3), dtype=np.float64)
    simple[:6] = signs[:, np.newaxis] * np.asarray((0.01, 0.02, 0.04))
    simple[6:] = np.asarray((0.005, 0.01, 0.02))
    with pytest.raises(ValidationError, match="Sharpe denominator is degenerate"):
        score_lean_g5_inverse_volatility_path(monthly_model, np.log1p(simple))

    invalid = replace(monthly_model, comparator_weights=np.asarray((1.0, 0.0, 0.0)))
    with pytest.raises(ValidationError, match="model comparator weights are invalid"):
        score_lean_g5_inverse_volatility_path(
            invalid, _return_paths(5, paths=1, observations=20)[0]
        )


def test_invalid_frequency_shape_and_empty_batch_fail() -> None:
    with pytest.raises(ValidationError, match="frequency"):
        fit_lean_g5_inverse_volatility_model(
            _return_paths(6, paths=100, observations=20),
            frequency="weekly",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="wrong shape"):
        fit_lean_g5_inverse_volatility_model(
            tuple(np.zeros((20, 2)) for _ in range(100)), frequency="monthly"
        )

    model = fit_lean_g5_inverse_volatility_model(
        _return_paths(7, paths=100, observations=20), frequency="monthly"
    )
    with pytest.raises(ValidationError, match="score paths are required"):
        score_lean_g5_inverse_volatility_paths(model, ())
