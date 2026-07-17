from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.bridge import (
    accelerator_eligibility,
    actual_bridge_gate,
    bridge_resource_preflight,
    fixed_k_bridge_statistic,
    tier_a_selection_gate,
    tier_b_tail_gate,
)


def _fixed_k(value: float = 0.5):
    return fixed_k_bridge_statistic(
        centroid_shifts=[0.75 * value],
        transition_cell_shifts=[0.10 * value],
        occupancy_shifts=[0.10 * value],
        relative_median_dwell_shifts=[0.25 * value],
        dwell_survival_shifts=[0.10 * value],
    )


def test_tier_a_uses_exact_250_and_one_sided_lower_bound() -> None:
    passed = tier_a_selection_gate(np.ones(250, dtype=np.bool_))
    failed = tier_a_selection_gate(
        np.r_[np.ones(225, dtype=np.bool_), np.zeros(25, dtype=np.bool_)]
    )

    assert passed.one_sided_lower_95 > 0.90
    assert passed.passed
    assert not failed.passed
    with pytest.raises(ModelError, match="exactly 250"):
        tier_a_selection_gate(np.ones(249, dtype=np.bool_))


def test_fixed_k_statistic_normalizes_every_component_cap() -> None:
    result = _fixed_k(1.0)
    np.testing.assert_allclose(result.normalized_components, 1.0)
    assert result.maximum == 1.0
    assert not result.normalized_components.flags.writeable


def test_tier_b_passes_or_fails_only_at_planned_looks() -> None:
    passing = tier_b_tail_gate(0.5, np.ones(1_000))
    failing = tier_b_tail_gate(0.5, np.zeros(1_000))

    assert passing.decision == "pass_above_tail"
    assert passing.passed
    assert passing.pairs_used == 1_000
    assert failing.decision == "fail_below_tail"
    assert not failing.passed
    with pytest.raises(ModelError, match="before a decision"):
        tier_b_tail_gate(0.5, np.r_[np.ones(25), np.zeros(975)])


def test_tier_b_crossing_at_ten_thousand_fails_closed() -> None:
    # 250/10,000 estimates exactly 0.025; planned CP intervals keep crossing.
    pseudo = np.zeros(10_000)
    pseudo[::40] = 1.0
    result = tier_b_tail_gate(0.5, pseudo)

    assert result.decision == "fail_closed_at_maximum"
    assert result.pairs_used == 10_000
    assert len(result.looks) == 10
    assert not result.stopped_early


def test_accelerator_requires_both_members_in_99_pairs() -> None:
    shape = (100, 2)
    valid = np.ones(shape, dtype=np.bool_)
    likelihood = np.zeros(shape)
    component = np.zeros(shape)
    statistic = np.zeros(shape)
    decisions = np.ones(shape, dtype=np.bool_)
    likelihood[0, 1] = 2e-6

    eligible = accelerator_eligibility(
        accelerated_valid=valid,
        ordinary_valid=valid,
        likelihood_difference_per_observation=likelihood,
        maximum_component_difference=component,
        statistic_difference=statistic,
        component_decisions_identical=decisions,
    )
    assert eligible.agreeing_pairs == 99
    assert eligible.eligible
    likelihood[1, 0] = 2e-6
    ineligible = accelerator_eligibility(
        accelerated_valid=valid,
        ordinary_valid=valid,
        likelihood_difference_per_observation=likelihood,
        maximum_component_difference=component,
        statistic_difference=statistic,
        component_decisions_identical=decisions,
    )
    assert not ineligible.eligible


def test_actual_bridge_applies_downstream_and_continuous_caps() -> None:
    gate = actual_bridge_gate(
        design_n_states=2,
        production_n_states=2,
        same_aligned_order=True,
        fixed_k=_fixed_k(1.0),
        design_monthly_block_length=5,
        production_monthly_block_length=7,
        design_daily_block_length=30,
        production_daily_block_length=40,
        design_monthly_state_counts=[100, 100],
        production_monthly_state_counts=[80, 125],
        design_daily_state_counts=[1000, 1000],
        production_daily_state_counts=[800, 1250],
        monthly_kernel_offset_differences=[0.005 / 12],
        daily_kernel_offset_differences=[0.005 / 252],
    )

    assert gate.passed
    np.testing.assert_allclose(gate.monthly_support_ratios, [0.8, 1.25])
    failed = actual_bridge_gate(
        design_n_states=2,
        production_n_states=2,
        same_aligned_order=False,
        fixed_k=_fixed_k(1.01),
        design_monthly_block_length=5,
        production_monthly_block_length=8,
        design_daily_block_length=30,
        production_daily_block_length=41,
        design_monthly_state_counts=[100, 100],
        production_monthly_state_counts=[79, 126],
        design_daily_state_counts=[1000, 1000],
        production_daily_state_counts=[799, 1251],
        monthly_kernel_offset_differences=[0.006 / 12],
        daily_kernel_offset_differences=[0.006 / 252],
    )
    assert not failed.passed
    assert "canonical_order_changed" in failed.reasons
    assert "centroid_hard_cap_exceeded" in failed.reasons


def test_actual_bridge_enforces_v2_absolute_block_bounds() -> None:
    common = {
        "design_n_states": 2,
        "production_n_states": 2,
        "same_aligned_order": True,
        "fixed_k": _fixed_k(1.0),
        "design_monthly_block_length": 12,
        "production_monthly_block_length": 12,
        "design_daily_block_length": 126,
        "production_daily_block_length": 126,
        "design_monthly_state_counts": [100, 100],
        "production_monthly_state_counts": [100, 100],
        "design_daily_state_counts": [1_000, 1_000],
        "production_daily_state_counts": [1_000, 1_000],
        "monthly_kernel_offset_differences": [0.0],
        "daily_kernel_offset_differences": [0.0],
    }

    assert actual_bridge_gate(**common).passed  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="outside the approved bounds"):
        actual_bridge_gate(  # type: ignore[arg-type]
            **{**common, "design_daily_block_length": 127}
        )


def test_resource_preflight_requires_binding_nine_workers_and_limits() -> None:
    passed = bridge_resource_preflight(
        benchmark_pair_seconds=np.full(100, 30.0),
        projected_pair_fits=20_500,
        workers=9,
        peak_memory_fraction=0.69,
    )
    assert passed.passed
    failed = bridge_resource_preflight(
        benchmark_pair_seconds=np.full(100, 200.0),
        projected_pair_fits=20_500,
        workers=8,
        peak_memory_fraction=0.71,
    )
    assert not failed.passed
    assert "worker_count_is_not_binding_nine" in failed.reasons
    assert "peak_memory_exceeds_limit" in failed.reasons


@pytest.mark.parametrize(
    "operation",
    [
        lambda: fixed_k_bridge_statistic(
            centroid_shifts=[],
            transition_cell_shifts=[0.0],
            occupancy_shifts=[0.0],
            relative_median_dwell_shifts=[0.0],
            dwell_survival_shifts=[0.0],
        ),
        lambda: tier_b_tail_gate(-1.0, np.ones(1_000)),
        lambda: tier_b_tail_gate(1.0, np.ones(500)),
        lambda: bridge_resource_preflight(
            benchmark_pair_seconds=np.ones(99),
            projected_pair_fits=1,
            workers=9,
            peak_memory_fraction=0.1,
        ),
    ],
)
def test_bridge_inputs_fail_closed(operation: object) -> None:
    with pytest.raises(ModelError):
        operation()  # type: ignore[operator]
