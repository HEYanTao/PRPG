from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from prpg.errors import ValidationError
from prpg.validation.lean_g5_inverse_volatility import LeanG5InverseVolatilityRows
from prpg.validation.lean_g5_statistics import (
    LEAN_G5_COMPONENT_NAMES,
    LEAN_G5_HOLM_FAMILIES,
    LEAN_G5_PILOT_GEOMETRY,
    LEAN_G5_PRODUCTION_GEOMETRY,
    LeanG5AlphaRows,
    LeanG5Geometry,
    LeanG5RegisteredResampleSeeds,
    LeanG5ResampleIndices,
    evaluate_lean_g5_alpha,
    fixed_clopper_pearson_precision_audit,
    four_family_holm,
    higher_empirical_quantile,
    plus_one_upper_tail_p_value,
    resample_indices_from_registered_seeds,
    simultaneous_alpha_equivalence,
    simultaneous_inverse_volatility_gap,
    simultaneous_leakage_gap,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _small_geometry() -> LeanG5Geometry:
    return LeanG5Geometry.test(monthly_paths=5, daily_paths=4, resamples=6)


def _small_indices() -> LeanG5ResampleIndices:
    monthly = np.asarray(
        (
            (0, 1, 2, 3, 4),
            (0, 0, 1, 3, 4),
            (0, 1, 1, 2, 4),
            (0, 2, 2, 3, 4),
            (0, 1, 3, 3, 4),
            (0, 1, 2, 4, 4),
        ),
        dtype=np.int64,
    )
    daily = np.asarray(
        (
            (0, 1, 2, 3),
            (0, 0, 2, 3),
            (0, 1, 1, 3),
            (0, 1, 2, 2),
            (0, 2, 2, 3),
            (0, 1, 3, 3),
        ),
        dtype=np.int64,
    )
    return LeanG5ResampleIndices(
        simulated_monthly=monthly,
        simulated_daily=daily,
        historical_monthly=monthly,
        historical_daily=daily,
    )


def _small_scores() -> LeanG5AlphaRows:
    monthly = (
        np.arange(5, dtype=np.float64)[:, np.newaxis] * 0.002
        + np.arange(6, dtype=np.float64)[np.newaxis, :] * 0.001
        - 0.006
    )
    daily = (
        np.arange(4, dtype=np.float64)[:, np.newaxis] * 0.002
        + np.arange(6, dtype=np.float64)[np.newaxis, :] * 0.001
        - 0.005
    )
    return LeanG5AlphaRows(monthly=monthly, daily=daily)


def _shift(rows: LeanG5AlphaRows, amount: float) -> LeanG5AlphaRows:
    return LeanG5AlphaRows(
        monthly=np.asarray(rows.monthly, dtype=np.float64) + amount,
        daily=np.asarray(rows.daily, dtype=np.float64) + amount,
    )


def _inverse_rows(amount: float = 0.0) -> dict[str, LeanG5InverseVolatilityRows]:
    return {
        "monthly": LeanG5InverseVolatilityRows(
            "monthly", np.arange(5, dtype=np.float64) * 0.01 + amount
        ),
        "daily": LeanG5InverseVolatilityRows(
            "daily", np.arange(4, dtype=np.float64) * 0.012 + amount
        ),
    }


def _manual_resampled_mean_and_se(
    values: FloatArray, indices: IntArray
) -> tuple[FloatArray, FloatArray]:
    selected = values[indices]
    return (
        np.mean(selected, axis=1),
        np.std(selected, axis=1, ddof=1) / math.sqrt(values.shape[0]),
    )


def test_production_and_pilot_geometry_are_exact_and_distinguishable() -> None:
    assert (
        LEAN_G5_PRODUCTION_GEOMETRY.monthly_paths,
        LEAN_G5_PRODUCTION_GEOMETRY.daily_paths,
        LEAN_G5_PRODUCTION_GEOMETRY.resamples,
        LEAN_G5_PRODUCTION_GEOMETRY.canonical,
    ) == (500, 200, 10_000, True)
    assert (
        LEAN_G5_PILOT_GEOMETRY.monthly_paths,
        LEAN_G5_PILOT_GEOMETRY.daily_paths,
        LEAN_G5_PILOT_GEOMETRY.resamples,
        LEAN_G5_PILOT_GEOMETRY.canonical,
    ) == (100, 40, 1_000, False)

    with pytest.raises(ValidationError, match="exactly 500/200/10,000"):
        LeanG5Geometry(499, 200, 10_000, "production")
    with pytest.raises(ValidationError, match="exactly 100/40/1,000"):
        LeanG5Geometry(100, 41, 1_000, "pilot")


def test_higher_quantile_and_plus_one_p_value_match_hand_counts() -> None:
    geometry = LeanG5Geometry.test(monthly_paths=2, daily_paths=2, resamples=4)
    values = np.asarray((4.0, 1.0, 3.0, 2.0))

    quantile = higher_empirical_quantile(values, 0.5, geometry=geometry)
    p_value = plus_one_upper_tail_p_value(values, 2.0, geometry=geometry)

    # ceil(0.5 * (4 - 1)) = 2, so the zero-based sorted value is 3.
    assert quantile.zero_based_index == 2
    assert quantile.one_based_rank == 3
    assert quantile.method == "higher"
    assert quantile.value == 3.0
    # Three null values tie or exceed two; the plus-one result is 4 / 5.
    assert p_value.exceedances == 3
    assert p_value.p_value == 0.8

    with pytest.raises(ValidationError, match="resample count"):
        higher_empirical_quantile(values[:-1], 0.5, geometry=geometry)


def test_four_family_holm_uses_exact_registry_and_stable_step_down() -> None:
    audit = four_family_holm(
        {"G5-D": 0.040, "G5-B": 0.010, "G5-A": 0.010, "G5-C": 0.030}
    )

    assert tuple(item.name for item in audit.decisions) == LEAN_G5_HOLM_FAMILIES
    assert tuple(item.rejected for item in audit.decisions) == (
        True,
        True,
        False,
        False,
    )
    assert tuple(item.step_rank for item in audit.decisions) == (1, 2, 3, 4)

    with pytest.raises(ValidationError, match="exactly G5-A/B/C/D"):
        four_family_holm({"G5-A": 0.1, "G5-B": 0.1, "G5-C": 0.1})


def test_fixed_clopper_pearson_precision_matches_registered_arithmetic() -> None:
    audit = fixed_clopper_pearson_precision_audit()

    assert audit.exceedances == 500
    assert audit.trials == 10_000
    assert audit.estimate == 0.05
    assert audit.confidence == 0.95
    assert audit.lower == pytest.approx(0.04580994219877061)
    assert audit.upper == pytest.approx(0.05445452702835893)
    assert audit.maximum_distance == pytest.approx(0.004454527028358926)
    assert audit.permitted_distance == 0.005
    assert audit.passed


def test_registered_seeds_are_deterministic_and_do_not_change_geometry() -> None:
    geometry = _small_geometry()
    seeds = LeanG5RegisteredResampleSeeds(11, 12, 13, 14)

    first = resample_indices_from_registered_seeds(seeds, geometry=geometry)
    second = resample_indices_from_registered_seeds(seeds, geometry=geometry)

    for left, right, paths in (
        (first.simulated_monthly, second.simulated_monthly, 5),
        (first.simulated_daily, second.simulated_daily, 4),
        (first.historical_monthly, second.historical_monthly, 5),
        (first.historical_daily, second.historical_daily, 4),
    ):
        left_array = np.asarray(left)
        np.testing.assert_array_equal(left_array, right)
        assert left_array.shape == (6, paths)
        assert int(np.min(left_array)) >= 0
        assert int(np.max(left_array)) < paths
        assert not left_array.flags.writeable

    with pytest.raises(ValidationError, match="unsigned 64-bit"):
        resample_indices_from_registered_seeds(
            LeanG5RegisteredResampleSeeds(-1, 12, 13, 14), geometry=geometry
        )
    with pytest.raises(ValidationError, match="must be distinct"):
        resample_indices_from_registered_seeds(
            LeanG5RegisteredResampleSeeds(11, 11, 13, 14), geometry=geometry
        )


def test_twelve_component_max_t_interval_matches_direct_hand_calculation() -> None:
    geometry = _small_geometry()
    indices = _small_indices()
    scores = _small_scores()

    audit = simultaneous_alpha_equivalence(scores, indices, geometry=geometry)

    monthly = np.asarray(scores.monthly, dtype=np.float64)
    daily = np.asarray(scores.daily, dtype=np.float64)
    monthly_boot_mean, monthly_boot_se = _manual_resampled_mean_and_se(
        monthly, np.asarray(indices.simulated_monthly, dtype=np.int64)
    )
    daily_boot_mean, daily_boot_se = _manual_resampled_mean_and_se(
        daily, np.asarray(indices.simulated_daily, dtype=np.int64)
    )
    studentized = np.concatenate(
        (
            (monthly_boot_mean - np.mean(monthly, axis=0)) / monthly_boot_se,
            (daily_boot_mean - np.mean(daily, axis=0)) / daily_boot_se,
        ),
        axis=1,
    )
    maxima = np.max(np.abs(studentized), axis=1)
    expected_critical = float(np.quantile(maxima, 0.90, method="higher"))
    expected_first_se = float(np.std(monthly[:, 0], ddof=1) / math.sqrt(5))
    expected_first_mean = float(np.mean(monthly[:, 0]))

    assert len(audit.names) == 12
    assert audit.names == LEAN_G5_COMPONENT_NAMES
    assert audit.critical_value.value == pytest.approx(expected_critical)
    assert audit.estimates[0] == pytest.approx(expected_first_mean)
    assert audit.standard_errors[0] == pytest.approx(expected_first_se)
    assert audit.lower[0] == pytest.approx(
        expected_first_mean - expected_critical * expected_first_se
    )
    assert audit.confidence == 0.90
    assert audit.margin == 0.10
    assert audit.passed
    assert not audit.geometry.canonical


def test_nonzero_historical_control_is_compared_instead_of_zero_filled() -> None:
    geometry = _small_geometry()
    indices = _small_indices()
    historical = _shift(_small_scores(), 0.11)
    simulated = _shift(historical, 0.01)

    audit = simultaneous_leakage_gap(simulated, historical, indices, geometry=geometry)

    assert all(abs(value) > 0.05 for value in audit.historical_estimates)
    assert all(value == pytest.approx(0.01) for value in audit.gaps)
    assert audit.maximum_upper == pytest.approx(0.01, abs=1.0e-14)
    assert audit.confidence == 0.95
    assert audit.cap == 0.10
    assert audit.passed


def test_injected_alpha_leakage_and_equivalence_defects_fail_grouped_gate() -> None:
    geometry = _small_geometry()
    indices = _small_indices()
    clean = _small_scores()

    leakage = simultaneous_leakage_gap(
        _shift(clean, 0.20), clean, indices, geometry=geometry
    )
    assert not leakage.passed
    assert leakage.maximum_upper == pytest.approx(0.20, abs=1.0e-14)
    assert all(not passed for passed in leakage.component_passed)

    monthly_defect = np.asarray(clean.monthly, dtype=np.float64).copy()
    monthly_defect[:, 0] += 0.20
    equivalence = simultaneous_alpha_equivalence(
        LeanG5AlphaRows(monthly_defect, clean.daily),
        indices,
        geometry=geometry,
    )
    assert not equivalence.passed
    assert not equivalence.component_passed[0]


def test_inverse_volatility_uses_one_historical_control_max_t_family() -> None:
    geometry = _small_geometry()
    indices = _small_indices()
    simulated = _small_scores()
    historical = _shift(simulated, -0.01)
    historical_inverse = _inverse_rows()
    simulated_inverse = _inverse_rows(0.01)

    inverse = simultaneous_inverse_volatility_gap(
        simulated_inverse,
        historical_inverse,
        indices,
        geometry=geometry,
    )

    passed = evaluate_lean_g5_alpha(
        simulated,
        historical,
        indices,
        simulated_inverse,
        historical_inverse,
        geometry=geometry,
    )
    defect = evaluate_lean_g5_alpha(
        simulated,
        historical,
        indices,
        _inverse_rows(0.201),
        historical_inverse,
        geometry=geometry,
    )

    assert inverse.confidence == 0.95
    assert inverse.cap == 0.20
    assert inverse.frequencies == ("monthly", "daily")
    assert inverse.gaps == pytest.approx((0.01, 0.01))
    assert inverse.maximum_upper == pytest.approx(0.01, abs=1.0e-14)
    assert inverse.critical_value.sample_count == 6
    assert inverse.passed
    assert passed.passed
    assert not defect.passed
    assert not defect.inverse_volatility.passed
    assert all(not value for value in defect.inverse_volatility.component_passed)


def test_bad_rows_indices_and_degenerate_scores_fail_closed() -> None:
    geometry = _small_geometry()
    indices = _small_indices()
    scores = _small_scores()

    with pytest.raises(ValidationError, match="wrong shape"):
        simultaneous_alpha_equivalence(
            LeanG5AlphaRows(np.zeros((5, 5)), scores.daily),
            indices,
            geometry=geometry,
        )

    bad_monthly_indices = np.asarray(indices.simulated_monthly).copy()
    bad_monthly_indices[0, 0] = 5
    with pytest.raises(ValidationError, match="out of range"):
        simultaneous_alpha_equivalence(
            scores,
            LeanG5ResampleIndices(
                bad_monthly_indices,
                indices.simulated_daily,
                indices.historical_monthly,
                indices.historical_daily,
            ),
            geometry=geometry,
        )

    with pytest.raises(ValidationError, match="standard error is degenerate"):
        simultaneous_alpha_equivalence(
            LeanG5AlphaRows(np.zeros((5, 6)), np.zeros((4, 6))),
            indices,
            geometry=geometry,
        )

    with pytest.raises(ValidationError, match="monthly and daily"):
        evaluate_lean_g5_alpha(
            scores,
            _shift(scores, -0.01),
            indices,
            {"monthly": _inverse_rows()["monthly"]},
            _inverse_rows(),
            geometry=geometry,
        )
