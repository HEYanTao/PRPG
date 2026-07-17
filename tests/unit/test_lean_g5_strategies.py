from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from prpg.errors import ValidationError
from prpg.validation.lean_g5_strategies import (
    LEAN_G5_FEATURE_NAMES,
    LEAN_G5_STRATEGY_NAMES,
    fit_lean_g5_strategy_model,
    lean_g5_strategy_weights,
    score_lean_g5_path,
)

FloatArray = NDArray[np.float64]


def _return_paths(
    seed: int,
    *,
    paths: int = 6,
    observations: int = 96,
) -> tuple[FloatArray, ...]:
    generator = np.random.default_rng(seed)
    result: list[FloatArray] = []
    for path_index in range(paths):
        innovations = generator.normal(
            loc=np.asarray((0.0008, 0.0003, 0.0004)),
            scale=np.asarray((0.018, 0.007, 0.008)),
            size=(observations, 3),
        )
        values = np.empty_like(innovations)
        values[0] = innovations[0]
        for row in range(1, observations):
            values[row] = (
                innovations[row]
                + 0.22 * values[row - 1]
                + 0.002
                * math.sin((row + 3 * path_index) / 7.0)
                * np.asarray((1.0, -0.4, 0.6))
            )
        result.append(values)
    return tuple(result)


@pytest.fixture(scope="module")
def monthly_model():  # type: ignore[no-untyped-def]
    return fit_lean_g5_strategy_model(_return_paths(100), frequency="monthly")


def test_fit_freezes_exact_ten_feature_and_six_strategy_geometry(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    assert len(LEAN_G5_FEATURE_NAMES) == 10
    assert len(LEAN_G5_STRATEGY_NAMES) == 6
    assert monthly_model.frequency == "monthly"
    assert monthly_model.periods_per_month == 1
    assert monthly_model.rebalance_periods == 1
    assert monthly_model.warmup_periods == 12
    assert monthly_model.feature_center.shape == (10,)
    assert monthly_model.feature_scale.shape == (10,)
    assert monthly_model.linear_coefficients.shape == (11, 3)
    assert monthly_model.quadratic_coefficients.shape == (66, 3)
    assert monthly_model.development_average_weights.shape == (6, 3)
    assert np.all(monthly_model.development_average_weights >= 0.0)
    np.testing.assert_array_equal(
        np.sum(monthly_model.development_average_weights, axis=1), np.ones(6)
    )
    for array in (
        monthly_model.feature_center,
        monthly_model.feature_scale,
        monthly_model.linear_coefficients,
        monthly_model.quadratic_coefficients,
        monthly_model.development_average_weights,
    ):
        assert not array.flags.writeable
    assert len(monthly_model.fingerprint) == 64


def test_traditional_rules_match_hand_computed_first_decision(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    path = np.zeros((30, 3), dtype=np.float64)
    path[:11, 0] = 0.02
    path[11, 0] = -0.01
    path[:12, 1] = 0.005
    path[:12, 2] = 0.004
    path[12:] = _return_paths(901, paths=1, observations=18)[0]

    weights = lean_g5_strategy_weights(monthly_model, path)

    assert weights.first_return_index == 12
    np.testing.assert_array_equal(
        weights.values[0, :4],
        np.asarray(
            (
                (1.0, 0.0, 0.0),  # greatest six-month cumulative return
                (1.0, 0.0, 0.0),  # furthest above 12-month moving average
                (1.0, 0.0, 0.0),  # worst preceding one-month return
                (0.0, 1.0, 0.0),  # muni beat taxable over six months
            )
        ),
    )


def test_exact_ties_split_long_only_weights(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    common = np.linspace(-0.01, 0.01, 40)
    path = np.repeat(common[:, np.newaxis], 3, axis=1)

    first = lean_g5_strategy_weights(monthly_model, path).values[0]

    thirds = np.full(3, 1.0 / 3.0)
    np.testing.assert_array_equal(first[0], thirds)
    np.testing.assert_array_equal(first[1], thirds)
    np.testing.assert_array_equal(first[2], thirds)
    np.testing.assert_array_equal(first[3], np.asarray((0.0, 0.5, 0.5)))


def test_weights_do_not_look_at_current_or_future_returns(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    original = _return_paths(333, paths=1, observations=80)[0]
    changed = original.copy()
    changed[45:] += np.asarray((0.8, -0.7, 0.6))

    original_weights = lean_g5_strategy_weights(monthly_model, original)
    changed_weights = lean_g5_strategy_weights(monthly_model, changed)

    # Weight row 45 sees rows 0..44.  The first changed return can affect only
    # the following decision, never the weight applied to itself.
    through_changed_row = 45 - original_weights.first_return_index + 1
    np.testing.assert_array_equal(
        original_weights.values[:through_changed_row],
        changed_weights.values[:through_changed_row],
    )


def test_development_comparator_is_the_mean_of_frozen_strategy_weights() -> None:
    development = _return_paths(444, paths=5, observations=84)
    model = fit_lean_g5_strategy_model(development, frequency="monthly")
    weights = np.concatenate(
        tuple(lean_g5_strategy_weights(model, path).values for path in development),
        axis=0,
    )
    np.testing.assert_allclose(
        model.development_average_weights,
        np.mean(weights, axis=0),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_hf_uses_month_equivalent_lookbacks_and_holds_for_21_sessions() -> None:
    development = _return_paths(555, paths=4, observations=600)
    model = fit_lean_g5_strategy_model(development, frequency="daily")
    path_weights = lean_g5_strategy_weights(model, development[0])

    assert model.periods_per_month == 21
    assert model.rebalance_periods == 21
    assert model.warmup_periods == 252
    assert path_weights.first_return_index == 252
    for offset in range(0, path_weights.values.shape[0], 21):
        held = path_weights.values[offset : offset + 21]
        np.testing.assert_array_equal(
            held,
            np.repeat(held[:1], held.shape[0], axis=0),
        )


def test_alpha_score_matches_hand_calculation_and_is_not_zero_filled(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    historical_control = _return_paths(777, paths=1, observations=217)[0]

    score = score_lean_g5_path(monthly_model, historical_control)
    weights = lean_g5_strategy_weights(monthly_model, historical_control)
    simple = np.expm1(historical_control[weights.first_return_index :])
    design = np.column_stack((np.ones(simple.shape[0]), simple))
    active = np.einsum(
        "ta,ta->t",
        weights.values[:, 0] - monthly_model.development_average_weights[0],
        simple,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, active, rcond=None)
    residual = active - design @ coefficients
    residual_sd = math.sqrt(
        float(residual @ residual) / (design.shape[0] - design.shape[1])
    )
    expected = math.sqrt(12.0) * float(coefficients[0]) / residual_sd

    assert score.frequency == "monthly"
    assert score.observations == 205
    assert score.strategy_names == LEAN_G5_STRATEGY_NAMES
    np.testing.assert_allclose(score.alpha_sharpes[0], expected, rtol=1.0e-12)
    assert np.isfinite(score.alpha_sharpes).all()
    assert not np.all(score.alpha_sharpes == 0.0)
    assert score.static_exposure_only == (False,) * len(LEAN_G5_STRATEGY_NAMES)
    np.testing.assert_array_equal(
        score.alpha_sharpes,
        score_lean_g5_path(monthly_model, historical_control).alpha_sharpes,
    )


def test_exactly_static_exposure_reports_zero_timing_alpha(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    coefficients = np.zeros_like(monthly_model.linear_coefficients)
    coefficients[0] = (1.0, 0.0, 0.0)
    frozen = np.frombuffer(coefficients.tobytes(), dtype=np.float64).reshape(
        coefficients.shape
    )
    static_linear = replace(monthly_model, linear_coefficients=frozen)

    score = score_lean_g5_path(
        static_linear,
        _return_paths(778, paths=1, observations=217)[0],
    )

    assert score.static_exposure_only[4] is True
    assert score.alpha_sharpes[4] == 0.0
    assert np.isfinite(score.alpha_sharpes).all()


def test_rank_degenerate_alpha_control_fails_instead_of_fabricating_zero(
    monthly_model,  # type: ignore[no-untyped-def]
) -> None:
    common = np.sin(np.arange(80, dtype=np.float64) / 5.0) * 0.01
    rank_degenerate = np.repeat(common[:, np.newaxis], 3, axis=1)

    with pytest.raises(ValidationError, match="rank-degenerate"):
        score_lean_g5_path(monthly_model, rank_degenerate)


@pytest.mark.parametrize(
    "bad_path",
    (
        np.zeros((40, 2)),
        np.zeros((12, 3)),
        np.column_stack((np.zeros((40, 2)), np.full(40, np.nan))),
    ),
)
def test_invalid_score_path_shape_or_values_fail(
    monthly_model,  # type: ignore[no-untyped-def]
    bad_path: FloatArray,
) -> None:
    with pytest.raises(ValidationError):
        lean_g5_strategy_weights(monthly_model, bad_path)


def test_equal_history_and_nondegenerate_features_are_required() -> None:
    unequal = (
        _return_paths(12, paths=1, observations=80)[0],
        *_return_paths(13, paths=1, observations=81),
    )
    with pytest.raises(ValidationError, match="equal history"):
        fit_lean_g5_strategy_model(unequal, frequency="monthly")

    constant = tuple(np.zeros((40, 3), dtype=np.float64) for _ in range(3))
    with pytest.raises(ValidationError, match="features are degenerate"):
        fit_lean_g5_strategy_model(constant, frequency="monthly")


def test_invalid_frequency_and_development_shape_fail() -> None:
    with pytest.raises(ValidationError, match="frequency"):
        fit_lean_g5_strategy_model(_return_paths(4), frequency="weekly")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="wrong shape"):
        fit_lean_g5_strategy_model((np.zeros((30, 2)),), frequency="monthly")
