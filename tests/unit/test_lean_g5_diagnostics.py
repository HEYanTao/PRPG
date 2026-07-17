from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ValidationError
from prpg.validation.lean_g5_diagnostics import (
    LEAN_G5_DIAGNOSTICS_SCHEMA_ID,
    compute_lean_g5_path_diagnostics,
    compute_lean_g5_return_diversity,
    compute_lean_g5_state_behavior,
)


def _monthly_path(seed: int = 123, observations: int = 72) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    scale = np.asarray([0.035, 0.012, 0.017])
    return np.asarray(rng.normal(size=(observations, 3)) * scale, dtype=np.float64)


def test_path_diagnostics_are_compact_deterministic_and_directly_calculated() -> None:
    values = _monthly_path()

    first = compute_lean_g5_path_diagnostics(values, frequency="monthly")
    second = compute_lean_g5_path_diagnostics(values.copy(), frequency="monthly")

    assert first == second
    assert first.schema_id == LEAN_G5_DIAGNOSTICS_SCHEMA_ID
    assert first.observations == 72
    assert len(first.names) == len(first.values) == len(set(first.names))
    assert first.value("covariance.0.2") == pytest.approx(
        np.cov(values, rowvar=False, ddof=1)[0, 2]
    )
    assert first.value("taxable_muni_spread.mean") == pytest.approx(
        np.mean(values[:, 2] - values[:, 1])
    )
    assert all(np.isfinite(first.values))
    with pytest.raises(ValidationError, match="unknown"):
        first.value("not-an-endpoint")


def test_daily_diagnostics_retain_registered_autocorrelation_horizons() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(44))
    values = rng.normal(scale=0.01, size=(300, 3))

    result = compute_lean_g5_path_diagnostics(values, frequency="daily")

    assert "raw_acf.20.equity" in result.names
    assert "absolute_acf.60.muni_bond" in result.names
    assert "squared_acf.60.taxable_bond" in result.names
    assert "raw_acf.21.equity" not in result.names


def test_injected_covariance_correlation_and_desynchronization_are_observable() -> None:
    pristine = _monthly_path(seed=7)
    covariance_defect = pristine.copy()
    covariance_defect[:, 0] *= 2.0
    correlation_defect = pristine.copy()
    correlation_defect[:, 1] = correlation_defect[:, 0]
    desynchronized = pristine.copy()
    desynchronized[1:, 1] = desynchronized[:-1, 0]
    desynchronized[0, 1] = 0.0

    baseline = compute_lean_g5_path_diagnostics(pristine, frequency="monthly")
    covariance = compute_lean_g5_path_diagnostics(
        covariance_defect, frequency="monthly"
    )
    correlation = compute_lean_g5_path_diagnostics(
        correlation_defect, frequency="monthly"
    )
    lagged = compute_lean_g5_path_diagnostics(desynchronized, frequency="monthly")

    assert covariance.value("covariance.0.0") == pytest.approx(
        4.0 * baseline.value("covariance.0.0")
    )
    assert correlation.value("correlation.0.1") == pytest.approx(1.0)
    assert lagged.value("cross_lag.0.1") > 0.99
    assert abs(baseline.value("cross_lag.0.1")) < 0.4


def test_injected_serial_tail_drawdown_and_spread_defects_are_observable() -> None:
    pristine = _monthly_path(seed=91)
    serial = pristine.copy()
    serial[:, 0] = np.sort(serial[:, 0])
    clipped = np.clip(pristine, -0.01, 0.01)
    drawdown = pristine.copy()
    drawdown[30, 0] = -1.0
    spread = pristine.copy()
    spread[:, 2] += 0.01

    baseline = compute_lean_g5_path_diagnostics(pristine, frequency="monthly")
    serial_result = compute_lean_g5_path_diagnostics(serial, frequency="monthly")
    clipped_result = compute_lean_g5_path_diagnostics(clipped, frequency="monthly")
    drawdown_result = compute_lean_g5_path_diagnostics(drawdown, frequency="monthly")
    spread_result = compute_lean_g5_path_diagnostics(spread, frequency="monthly")

    assert serial_result.value("raw_acf.1.equity") > 0.8
    assert abs(baseline.value("raw_acf.1.equity")) < 0.4
    assert clipped_result.value("quantile.0.01.equity") >= -0.01
    assert clipped_result.value("quantile.0.99.equity") <= 0.01
    assert drawdown_result.value("drawdown_depth.equity") > 0.6
    assert spread_result.value("taxable_muni_spread.mean") == pytest.approx(
        baseline.value("taxable_muni_spread.mean") + 0.01
    )


def test_joint_downside_tail_defect_is_observable() -> None:
    pristine = _monthly_path(seed=308, observations=240)
    synchronized = pristine.copy()
    synchronized[:, 1] = synchronized[:, 0]

    baseline = compute_lean_g5_path_diagnostics(pristine, frequency="monthly")
    defective = compute_lean_g5_path_diagnostics(synchronized, frequency="monthly")

    assert defective.value("downside_coexceedance.0.1") >= 0.10
    assert defective.value("downside_coexceedance.0.1") > 3.0 * baseline.value(
        "downside_coexceedance.0.1"
    )


def test_state_behavior_preserves_absent_state_as_unsupported() -> None:
    returns = np.arange(36, dtype=np.float64).reshape(12, 3) / 10_000.0
    states = np.asarray([0, 0, 1, 1, 1, 2, 0, 2, 2, 2, 2, 0], dtype=np.int64)

    result = compute_lean_g5_state_behavior([returns], [states])

    assert result.path_count == 1
    assert result.observations == 12
    assert result.occupancy_counts == (4, 3, 5, 0)
    assert result.supported_states == (True, True, True, False)
    assert result.occupancy == pytest.approx((4 / 12, 3 / 12, 5 / 12, 0.0))
    assert result.transition_probabilities[3] == (None, None, None, None)
    assert result.mean_dwell[3] is None
    assert result.state_conditional_log_mean[3] is None


def test_transition_diagonal_and_dwell_defect_are_visible_without_path_crossing() -> (
    None
):
    returns = np.zeros((40, 3), dtype=np.float64)
    cycling = np.tile(np.arange(4, dtype=np.int64), 10)
    sticky = np.repeat(np.arange(4, dtype=np.int64), 10)

    baseline = compute_lean_g5_state_behavior([returns], [cycling])
    defective = compute_lean_g5_state_behavior([returns], [sticky])

    assert baseline.transition_probabilities[0][0] == 0.0
    assert defective.transition_probabilities[0][0] == pytest.approx(0.9)
    assert baseline.mean_dwell == (1.0, 1.0, 1.0, 1.0)
    assert defective.mean_dwell == (10.0, 10.0, 10.0, 10.0)

    split = compute_lean_g5_state_behavior(
        [returns[:20], returns[20:]], [cycling[:20], cycling[20:]]
    )
    assert sum(split.departure_counts) == 38


def test_complete_path_copy_and_signed_zero_are_detected_exactly() -> None:
    first = _monthly_path(seed=10, observations=18)
    second = first.copy()
    first[0, 0] = -0.0
    second[0, 0] = 0.0
    third = _monthly_path(seed=11, observations=18)

    result = compute_lean_g5_return_diversity(
        [first, second, third], frequency="monthly"
    )

    assert result.unique_complete_paths == 2
    assert result.complete_path_duplicate_copies == 1
    assert result.complete_path_duplicate_groups == 1
    assert result.maximum_complete_path_multiplicity == 2
    assert result.subsequence_window == 6
    assert result.repeated_subsequence_copies >= 13
    assert result.maximum_subsequence_multiplicity >= 2


def test_direct_return_subsequence_copy_is_detected_without_source_indices() -> None:
    first = _monthly_path(seed=21, observations=20)
    second = _monthly_path(seed=22, observations=20)
    pristine = compute_lean_g5_return_diversity(
        [first, second], frequency="monthly", subsequence_window=4
    )
    second[8:12] = first[3:7]

    copied = compute_lean_g5_return_diversity(
        [first, second], frequency="monthly", subsequence_window=4
    )
    repeated = compute_lean_g5_return_diversity(
        [first.copy(), second.copy()],
        frequency="monthly",
        subsequence_window=4,
    )

    assert pristine.repeated_subsequence_copies == 0
    assert copied.complete_path_duplicate_copies == 0
    assert copied.repeated_subsequence_copies >= 1
    assert copied.repeated_subsequence_groups >= 1
    assert copied == repeated


def test_diagnostics_reject_malformed_or_unsupported_inputs() -> None:
    with pytest.raises(ValidationError, match="at least 13"):
        compute_lean_g5_path_diagnostics(np.zeros((12, 3)), frequency="monthly")
    with pytest.raises(ValidationError, match="finite"):
        invalid = np.zeros((20, 3))
        invalid[0, 0] = np.nan
        compute_lean_g5_path_diagnostics(invalid, frequency="monthly")
    with pytest.raises(ValidationError, match="aligned K=4"):
        compute_lean_g5_state_behavior(
            [np.zeros((4, 3))], [np.asarray([0.0, 1.0, 2.0, 3.0])]
        )
    with pytest.raises(ValidationError, match="different horizons"):
        compute_lean_g5_return_diversity(
            [np.zeros((5, 3)), np.zeros((6, 3))],
            frequency="monthly",
            subsequence_window=2,
        )
    with pytest.raises(ValidationError, match="exceeds"):
        compute_lean_g5_return_diversity(
            [np.zeros((5, 3))], frequency="monthly", subsequence_window=6
        )
