from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.dependence import (
    dependence_holm_gate,
    dependence_max_statistic_cell,
    evaluate_dependence_vector,
    feasible_sensitivity_lengths,
    fit_dependence_reference,
)


def _returns(rows: int, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(size=rows)
    return np.column_stack(
        (
            common + 0.3 * rng.normal(size=rows),
            0.4 * common + rng.normal(size=rows),
            -0.2 * common + rng.normal(size=rows),
        )
    )


def test_monthly_and_daily_endpoint_dimensions_and_frozen_order() -> None:
    monthly = _returns(80)
    monthly_mask = np.ones(80, dtype=np.bool_)
    monthly_mask[0] = False
    reference = fit_dependence_reference(monthly, monthly_mask, frequency="monthly")

    assert len(reference.labels) == 87
    assert reference.labels[:4] == (
        "corr_equity_muni_bond",
        "corr_equity_taxable_bond",
        "corr_muni_bond_taxable_bond",
        "acf_equity_lag_1",
    )
    assert reference.labels[-1] == "acf_squared_pc1_lag_12"
    evaluated = evaluate_dependence_vector(monthly, monthly_mask, reference)
    np.testing.assert_array_equal(
        evaluated.transformed_values, reference.transformed_values
    )

    daily = _returns(140)
    daily_mask = np.ones(140, dtype=np.bool_)
    daily_mask[0] = False
    daily_reference = fit_dependence_reference(daily, daily_mask, frequency="daily")
    assert len(daily_reference.labels) == 31
    assert daily_reference.labels[-1] == "acf_squared_pc1_lag_63"
    assert not reference.transformed_values.flags.writeable


def test_gap_aware_acf_excludes_cross_segment_pair() -> None:
    values = _returns(90)
    mask = np.ones(90, dtype=np.bool_)
    mask[[0, 45]] = False
    reference = fit_dependence_reference(values, mask, frequency="monthly")

    centered = values[:, 0] - values[:, 0].mean()
    valid = np.ones(89, dtype=np.bool_)
    valid[44] = False
    expected = np.dot(centered[1:][valid], centered[:-1][valid]) / np.sqrt(
        np.dot(centered[1:][valid], centered[1:][valid])
        * np.dot(centered[:-1][valid], centered[:-1][valid])
    )
    assert reference.raw_values[3] == pytest.approx(expected)


def test_simulation_uses_frozen_pc1_transform() -> None:
    values = _returns(100)
    mask = np.r_[False, np.ones(99, dtype=np.bool_)]
    reference = fit_dependence_reference(values, mask, frequency="monthly")
    shifted = values.copy()
    shifted[:, 0] = shifted[:, 0] * 4.0 + 20.0

    result = evaluate_dependence_vector(shifted, mask, reference)
    adaptive = fit_dependence_reference(shifted, mask, frequency="monthly")

    assert not np.array_equal(result.raw_values, adaptive.raw_values)
    np.testing.assert_array_equal(result.labels, reference.labels)


def test_two_sided_max_stat_has_plus_one_pvalue_and_exact_count() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(10_000, 3))
    cell = dependence_max_statistic_cell([0.0, 0.0, 0.0], samples)

    expected = (
        1 + np.count_nonzero(cell.replicate_statistics >= cell.observed_statistic)
    ) / 10_001
    assert cell.p_value == expected
    assert cell.repetitions == 10_000
    assert cell.critical_value_95 == np.quantile(
        cell.replicate_statistics, 0.95, method="higher"
    )
    with pytest.raises(ModelError, match="exact shape"):
        dependence_max_statistic_cell([0.0, 0.0, 0.0], samples[:-1])


def test_constant_components_are_exact_and_all_constant_is_valid() -> None:
    samples = np.ones((10_000, 2))
    cell = dependence_max_statistic_cell([1.0, 1.0], samples)
    assert cell.observed_statistic == 0.0
    assert cell.p_value == 1.0
    assert not cell.active_components.any()
    with pytest.raises(ModelError, match="constant component"):
        dependence_max_statistic_cell([1.0, 1.1], samples)


def test_holm_gate_uses_exact_four_cell_order() -> None:
    samples = np.arange(10_000, dtype=np.float64)[:, None]
    cells = tuple(dependence_max_statistic_cell([5_000.0], samples) for _ in range(4))
    gate = dependence_holm_gate(cells)  # type: ignore[arg-type]
    assert gate.cell_order == (
        "design_monthly",
        "design_daily",
        "production_monthly",
        "production_daily",
    )
    assert gate.passed
    with pytest.raises(ModelError, match="exactly four"):
        dependence_holm_gate(cells[:3])  # type: ignore[arg-type]


def test_feasible_neighbor_lengths_respect_hard_bounds() -> None:
    assert feasible_sensitivity_lengths(2, frequency="monthly") == (2, 3)
    assert feasible_sensitivity_lengths(12, frequency="monthly") == (11, 12)
    assert feasible_sensitivity_lengths(59, frequency="daily") == (58, 59, 60)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: fit_dependence_reference(
            np.ones((80, 3)),
            np.r_[False, np.ones(79, dtype=np.bool_)],
            frequency="monthly",
        ),
        lambda: fit_dependence_reference(
            _returns(80), np.ones(80, dtype=np.bool_), frequency="monthly"
        ),
        lambda: fit_dependence_reference(
            _returns(80), np.r_[False, np.ones(79, dtype=np.bool_)], frequency="weekly"
        ),
        lambda: feasible_sensitivity_lengths(1, frequency="daily"),
    ],
)
def test_dependence_inputs_fail_closed(operation: object) -> None:
    with pytest.raises(ModelError):
        operation()  # type: ignore[operator]
