from __future__ import annotations

import math

import numpy as np
import pytest

from prpg.errors import ValidationError
from prpg.validation.metrics import (
    PathDiversityAccumulator,
    compute_base_frequency_metrics,
    compute_path_diversity,
    cross_frequency_log_compounding_identity,
    fixed_group_boundaries,
)


def test_base_frequency_metrics_match_direct_three_asset_calculations() -> None:
    time = np.arange(1.0, 9.0)
    values = np.column_stack((time / 100.0, -time / 200.0, (time % 3) / 300.0))

    result = compute_base_frequency_metrics(
        values,
        frequency="monthly",
        quantile_probabilities=(0.25, 0.50, 0.75),
        raw_acf_lags=(1, 2),
        transformed_acf_lags=(1, 2),
    )

    direct_covariance = np.cov(values, rowvar=False, ddof=1)
    assert result.annualized_log_mean == pytest.approx(
        tuple(12.0 * values.mean(axis=0))
    )
    assert result.annualized_volatility == pytest.approx(
        tuple(np.sqrt(12.0 * np.diag(direct_covariance)))
    )
    assert np.asarray(result.base_frequency_covariance) == pytest.approx(
        direct_covariance
    )
    assert np.asarray(result.annualized_covariance) == pytest.approx(
        12.0 * direct_covariance
    )
    assert np.asarray(result.marginal_quantiles) == pytest.approx(
        np.quantile(values, (0.25, 0.50, 0.75), axis=0, method="higher")
    )
    assert result.lower_tail_probabilities == (0.25,)
    assert result.upper_tail_probabilities == (0.75,)
    assert (
        result.fingerprint
        == compute_base_frequency_metrics(
            values.copy(),
            frequency="monthly",
            quantile_probabilities=(0.25, 0.50, 0.75),
            raw_acf_lags=(1, 2),
            transformed_acf_lags=(1, 2),
        ).fingerprint
    )


def test_constant_series_metrics_remain_finite_and_are_disclosed() -> None:
    values = np.column_stack(
        (
            np.linspace(-0.02, 0.03, 10),
            np.zeros(10),
            np.full(10, 0.001),
        )
    )

    result = compute_base_frequency_metrics(
        values,
        frequency="monthly",
        raw_acf_lags=(1,),
        transformed_acf_lags=(1,),
    )

    assert result.constant_assets == (False, True, True)
    flattened = np.asarray(
        [
            *result.annualized_volatility,
            *np.asarray(result.correlation).ravel(),
            *np.asarray(result.raw_acf).ravel(),
            *np.asarray(result.absolute_acf).ravel(),
            *np.asarray(result.squared_acf).ravel(),
        ]
    )
    assert np.isfinite(flattened).all()
    assert result.correlation[1] == (0.0, 1.0, 0.0)
    assert result.raw_acf[0][1:] == (0.0, 0.0)


def test_drawdown_depth_and_duration_use_log_wealth_without_exponentiating_wealth() -> (
    None
):
    values = np.zeros((4, 3), dtype=np.float64)
    values[:, 0] = (math.log(2.0), math.log(0.5), 0.0, 0.0)

    result = compute_base_frequency_metrics(
        values,
        frequency="monthly",
        quantile_probabilities=(0.25, 0.75),
        raw_acf_lags=(1,),
        transformed_acf_lags=(1,),
    )

    assert result.maximum_drawdown[0] == pytest.approx(0.5)
    assert result.maximum_drawdown_duration[0] == 3
    assert result.maximum_drawdown[1:] == (0.0, 0.0)


def test_cross_frequency_identity_handles_explicit_irregular_boundaries() -> None:
    base = np.arange(30, dtype=np.float64).reshape(10, 3) / 1_000.0
    boundaries = (0, 2, 5, 10)
    aggregate = np.vstack(
        [base[start:stop].sum(axis=0) for start, stop in ((0, 2), (2, 5), (5, 10))]
    )

    exact = cross_frequency_log_compounding_identity(base, aggregate, boundaries)
    changed = aggregate.copy()
    changed[1, 0] += 1e-8
    failed = cross_frequency_log_compounding_identity(base, changed, boundaries)
    tolerant = cross_frequency_log_compounding_identity(
        base,
        changed,
        boundaries,
        absolute_tolerance=1e-7,
    )

    assert exact.exact_match and exact.passed
    assert exact.maximum_absolute_error == (0.0, 0.0, 0.0)
    assert not failed.exact_match and not failed.passed
    assert not tolerant.exact_match and tolerant.passed


def test_fixed_group_boundaries_require_an_exact_partition() -> None:
    assert fixed_group_boundaries(12, 3) == (0, 3, 6, 9, 12)
    with pytest.raises(ValidationError, match="not divisible"):
        fixed_group_boundaries(10, 3)


def test_streaming_path_diversity_canonicalizes_signed_zero() -> None:
    first = np.arange(18, dtype=np.float64).reshape(6, 3)
    second = first.copy()
    first[0, 0] = -0.0
    second[0, 0] = 0.0
    third = first.copy()
    third[-1, -1] += 1.0

    result = compute_path_diversity(iter((first, second, third)))

    assert result.path_count == 3
    assert result.unique_path_count == 2
    assert result.duplicate_copies == 1
    assert result.duplicate_group_count == 1
    assert result.paths_in_duplicate_groups == 2
    assert result.maximum_multiplicity == 2
    assert result.unique_fraction == pytest.approx(2.0 / 3.0)
    assert len(result.duplicate_hashes) == 1


def test_path_diversity_rejects_mixed_horizons() -> None:
    accumulator = PathDiversityAccumulator()
    accumulator.update(np.zeros((2, 3)))

    with pytest.raises(ValidationError, match="different horizons"):
        accumulator.update(np.zeros((3, 3)))


def test_metrics_reject_nan_and_acf_lag_without_support() -> None:
    values = np.zeros((3, 3))
    values[0, 0] = np.nan
    with pytest.raises(ValidationError, match="finite"):
        compute_base_frequency_metrics(
            values,
            frequency="monthly",
            raw_acf_lags=(1,),
            transformed_acf_lags=(1,),
        )

    with pytest.raises(ValidationError, match="smaller than observations"):
        compute_base_frequency_metrics(
            np.zeros((3, 3)),
            frequency="monthly",
            raw_acf_lags=(3,),
            transformed_acf_lags=(1,),
        )
