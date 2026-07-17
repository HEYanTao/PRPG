"""Pure decision primitives for the dual-reference Phase-3 bridge.

History generation and HMM fitting remain orchestration concerns.  This module
freezes the exact Tier-A binomial gate, Tier-B sequential decision, accelerator
audit, actual-artifact hard caps, and resource preflight from Sections 4.5 and
4.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from prpg.errors import ModelError
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.inference import BinomialInterval, clopper_pearson_interval

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
TierBDecision: TypeAlias = Literal[
    "pass_above_tail",
    "fail_below_tail",
    "fail_closed_at_maximum",
]

TIER_A_PAIRS = 250
TIER_A_CONFIDENCE = 0.95
TIER_A_MINIMUM_LOWER_BOUND = 0.90
TIER_B_BATCH_SIZE = 1_000
TIER_B_MAXIMUM_PAIRS = 10_000
TIER_B_TAIL_PROBABILITY = 0.025
TIER_B_INTERVAL_ERROR = 0.0025
ACCELERATOR_AUDIT_PAIRS = 100
ACCELERATOR_MINIMUM_AGREEMENTS = 99


@dataclass(frozen=True, slots=True)
class TierAResult:
    """One 250-pair order-selection stress result for one DGP."""

    successes: int
    pairs: int
    estimate: float
    one_sided_lower_95: float
    required_lower_bound: float
    passed: bool


@dataclass(frozen=True, slots=True)
class FixedKBridgeStatistic:
    """Hard-cap-normalized fixed-K component family and its maximum."""

    component_names: tuple[str, ...]
    raw_maxima: FloatArray
    hard_caps: FloatArray
    normalized_components: FloatArray
    maximum: float


@dataclass(frozen=True, slots=True)
class TierBLook:
    """One planned 1,000-pair Tier-B look."""

    look: int
    pairs: int
    exceedances: int
    interval: BinomialInterval


@dataclass(frozen=True, slots=True)
class TierBResult:
    """Frozen sequential Tier-B tail decision for one DGP."""

    actual_statistic: float
    decision: TierBDecision
    passed: bool
    stopped_early: bool
    looks: tuple[TierBLook, ...]
    pairs_used: int
    exceedances: int
    estimate: float
    empirical_quantile_975: float


@dataclass(frozen=True, slots=True)
class AcceleratorResult:
    """First-100-pair comparison of accelerated and ordinary fits."""

    agreeing_pairs: int
    audited_pairs: int
    eligible: bool
    pair_agreement: BoolArray


@dataclass(frozen=True, slots=True)
class ActualBridgeGate:
    """Actual design-to-production hard-cap and downstream comparison."""

    fixed_k: FixedKBridgeStatistic
    same_k: bool
    same_order: bool
    monthly_block_change: int
    daily_block_change: int
    monthly_support_ratios: FloatArray
    daily_support_ratios: FloatArray
    maximum_annualized_monthly_kernel_change: float
    maximum_annualized_daily_kernel_change: float
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class BridgeResourcePreflight:
    """Projected resource decision before either bridge tier is launched."""

    benchmark_pairs: int
    workers: int
    projected_worst_case_hours: float
    peak_memory_fraction: float
    maximum_hours: float
    maximum_memory_fraction: float
    passed: bool
    reasons: tuple[str, ...]


def tier_a_selection_gate(successful_pairs: ArrayLike) -> TierAResult:
    """Apply the exact one-sided 95% Clopper-Pearson lower-bound gate."""

    successful = np.asarray(successful_pairs)
    if successful.shape != (TIER_A_PAIRS,) or successful.dtype.kind != "b":
        raise ModelError("Tier A requires exactly 250 boolean pseudo-pair outcomes")
    successes = int(np.count_nonzero(successful))
    lower = _one_sided_binomial_lower(
        successes, TIER_A_PAIRS, confidence=TIER_A_CONFIDENCE
    )
    return TierAResult(
        successes,
        TIER_A_PAIRS,
        successes / TIER_A_PAIRS,
        lower,
        TIER_A_MINIMUM_LOWER_BOUND,
        lower >= TIER_A_MINIMUM_LOWER_BOUND,
    )


def fixed_k_bridge_statistic(
    *,
    centroid_shifts: ArrayLike,
    transition_cell_shifts: ArrayLike,
    occupancy_shifts: ArrayLike,
    relative_median_dwell_shifts: ArrayLike,
    dwell_survival_shifts: ArrayLike,
) -> FixedKBridgeStatistic:
    """Normalize the five binding continuous families by their hard caps."""

    names = (
        "centroid",
        "transition_cell",
        "occupancy",
        "relative_median_dwell",
        "dwell_survival",
    )
    arrays = (
        _absolute_finite(centroid_shifts, names[0]),
        _absolute_finite(transition_cell_shifts, names[1]),
        _absolute_finite(occupancy_shifts, names[2]),
        _absolute_finite(relative_median_dwell_shifts, names[3]),
        _absolute_finite(dwell_survival_shifts, names[4]),
    )
    maxima = np.asarray([float(np.max(values)) for values in arrays])
    caps = np.asarray([0.75, 0.10, 0.10, 0.25, 0.10], dtype=np.float64)
    normalized = maxima / caps
    maximum = float(np.max(normalized))
    _readonly(maxima, caps, normalized)
    return FixedKBridgeStatistic(names, maxima, caps, normalized, maximum)


def tier_b_tail_gate(
    actual_statistic: float,
    pseudo_statistics: ArrayLike,
) -> TierBResult:
    """Apply ten planned two-sided CP looks at most, failing closed at 10,000."""

    if not np.isfinite(actual_statistic) or actual_statistic < 0.0:
        raise ModelError("actual Tier-B statistic must be finite and non-negative")
    pseudo = np.asarray(pseudo_statistics, dtype=np.float64)
    if (
        pseudo.ndim != 1
        or pseudo.size == 0
        or pseudo.size > TIER_B_MAXIMUM_PAIRS
        or pseudo.size % TIER_B_BATCH_SIZE != 0
        or not np.isfinite(pseudo).all()
        or bool((pseudo < 0.0).any())
    ):
        raise ModelError(
            "Tier-B pseudo statistics require finite 1,000-pair batches through 10,000"
        )
    looks: list[TierBLook] = []
    decision: TierBDecision | None = None
    used = 0
    for end in range(TIER_B_BATCH_SIZE, len(pseudo) + 1, TIER_B_BATCH_SIZE):
        exceedances = int(np.count_nonzero(pseudo[:end] >= actual_statistic))
        interval = clopper_pearson_interval(
            exceedances,
            end,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        looks.append(TierBLook(len(looks) + 1, end, exceedances, interval))
        used = end
        if interval.lower > TIER_B_TAIL_PROBABILITY:
            decision = "pass_above_tail"
            break
        if interval.upper < TIER_B_TAIL_PROBABILITY:
            decision = "fail_below_tail"
            break
    if decision is None:
        if len(pseudo) != TIER_B_MAXIMUM_PAIRS:
            raise ModelError(
                "Tier-B evidence ended before a decision or the fixed 10,000-pair cap"
            )
        decision = "fail_closed_at_maximum"
    used_values = pseudo[:used]
    exceedances = int(np.count_nonzero(used_values >= actual_statistic))
    quantile = float(np.quantile(used_values, 0.975, method="higher"))
    passed = decision == "pass_above_tail"
    return TierBResult(
        float(actual_statistic),
        decision,
        passed,
        used < TIER_B_MAXIMUM_PAIRS,
        tuple(looks),
        used,
        exceedances,
        exceedances / used,
        quantile,
    )


def accelerator_eligibility(
    *,
    accelerated_valid: ArrayLike,
    ordinary_valid: ArrayLike,
    likelihood_difference_per_observation: ArrayLike,
    maximum_component_difference: ArrayLike,
    statistic_difference: ArrayLike,
    component_decisions_identical: ArrayLike,
) -> AcceleratorResult:
    """Audit both prefix/full members for exactly the first 100 pseudo-pairs."""

    shape = (ACCELERATOR_AUDIT_PAIRS, 2)
    accelerated = _boolean_matrix(accelerated_valid, shape, "accelerated validity")
    ordinary = _boolean_matrix(ordinary_valid, shape, "ordinary validity")
    likelihood = _finite_matrix(
        likelihood_difference_per_observation, shape, "likelihood difference"
    )
    component = _finite_matrix(
        maximum_component_difference, shape, "component difference"
    )
    statistic = _finite_matrix(statistic_difference, shape, "statistic difference")
    decisions = _boolean_matrix(
        component_decisions_identical, shape, "component decisions"
    )
    same_validity = accelerated == ordinary
    comparable = accelerated & ordinary
    member_agreement = (
        same_validity
        & (~comparable | (np.abs(likelihood) <= 1e-6))
        & (~comparable | (np.abs(component) <= 1e-3))
        & (~comparable | (np.abs(statistic) <= 1e-3))
        & (~comparable | decisions)
    )
    pair_agreement = member_agreement.all(axis=1).astype(np.bool_)
    agreeing = int(np.count_nonzero(pair_agreement))
    _readonly(pair_agreement)
    return AcceleratorResult(
        agreeing,
        ACCELERATOR_AUDIT_PAIRS,
        agreeing >= ACCELERATOR_MINIMUM_AGREEMENTS,
        pair_agreement,
    )


def actual_bridge_gate(
    *,
    design_n_states: int,
    production_n_states: int,
    same_aligned_order: bool,
    fixed_k: FixedKBridgeStatistic,
    design_monthly_block_length: int,
    production_monthly_block_length: int,
    design_daily_block_length: int,
    production_daily_block_length: int,
    design_monthly_state_counts: ArrayLike,
    production_monthly_state_counts: ArrayLike,
    design_daily_state_counts: ArrayLike,
    production_daily_state_counts: ArrayLike,
    monthly_kernel_offset_differences: ArrayLike,
    daily_kernel_offset_differences: ArrayLike,
) -> ActualBridgeGate:
    """Apply actual-artifact K/order, cap, block, support, and kernel gates."""

    _positive_integer(design_n_states, "design_n_states")
    _positive_integer(production_n_states, "production_n_states")
    if not isinstance(same_aligned_order, bool):
        raise ModelError("same_aligned_order must be boolean")
    design_monthly = _counts(design_monthly_state_counts, design_n_states, "monthly")
    production_monthly = _counts(
        production_monthly_state_counts, production_n_states, "monthly"
    )
    design_daily = _counts(design_daily_state_counts, design_n_states, "daily")
    production_daily = _counts(
        production_daily_state_counts, production_n_states, "daily"
    )
    same_k = design_n_states == production_n_states
    if same_k:
        monthly_ratios = production_monthly / design_monthly
        daily_ratios = production_daily / design_daily
    else:
        monthly_ratios = np.empty(0, dtype=np.float64)
        daily_ratios = np.empty(0, dtype=np.float64)
    for value, label, bounds in (
        (
            design_monthly_block_length,
            "design monthly block length",
            MONTHLY_BLOCK_LENGTH_BOUNDS,
        ),
        (
            production_monthly_block_length,
            "production monthly block length",
            MONTHLY_BLOCK_LENGTH_BOUNDS,
        ),
        (
            design_daily_block_length,
            "design daily block length",
            DAILY_BLOCK_LENGTH_BOUNDS,
        ),
        (
            production_daily_block_length,
            "production daily block length",
            DAILY_BLOCK_LENGTH_BOUNDS,
        ),
    ):
        _bounded_block_length(value, label, bounds)
    monthly_block_change = abs(
        production_monthly_block_length - design_monthly_block_length
    )
    daily_block_change = abs(production_daily_block_length - design_daily_block_length)
    monthly_kernel = _absolute_finite(
        monthly_kernel_offset_differences, "monthly kernel offset difference"
    )
    daily_kernel = _absolute_finite(
        daily_kernel_offset_differences, "daily kernel offset difference"
    )
    annual_monthly = 12.0 * float(np.max(monthly_kernel))
    annual_daily = 252.0 * float(np.max(daily_kernel))
    reasons: list[str] = []
    if not same_k:
        reasons.append("selected_k_changed")
    if not same_aligned_order:
        reasons.append("canonical_order_changed")
    for index, name in enumerate(fixed_k.component_names):
        if fixed_k.normalized_components[index] > 1.0:
            reasons.append(f"{name}_hard_cap_exceeded")
    if monthly_block_change > 2:
        reasons.append("monthly_block_change_exceeded")
    if daily_block_change > 10:
        reasons.append("daily_block_change_exceeded")
    if same_k and bool(((monthly_ratios < 0.80) | (monthly_ratios > 1.25)).any()):
        reasons.append("monthly_state_support_ratio_outside_range")
    if same_k and bool(((daily_ratios < 0.80) | (daily_ratios > 1.25)).any()):
        reasons.append("daily_state_support_ratio_outside_range")
    if annual_monthly > 0.005:
        reasons.append("monthly_kernel_offset_change_exceeded")
    if annual_daily > 0.005:
        reasons.append("daily_kernel_offset_change_exceeded")
    monthly_ratios = monthly_ratios.astype(np.float64, copy=True)
    daily_ratios = daily_ratios.astype(np.float64, copy=True)
    _readonly(monthly_ratios, daily_ratios)
    return ActualBridgeGate(
        fixed_k,
        same_k,
        same_aligned_order,
        monthly_block_change,
        daily_block_change,
        monthly_ratios,
        daily_ratios,
        annual_monthly,
        annual_daily,
        tuple(reasons),
    )


def bridge_resource_preflight(
    *,
    benchmark_pair_seconds: ArrayLike,
    projected_pair_fits: int,
    workers: int,
    peak_memory_fraction: float,
    maximum_hours: float = 72.0,
    maximum_memory_fraction: float = 0.70,
) -> BridgeResourcePreflight:
    """Project conservatively from 100 representative pair timings."""

    timings = np.asarray(benchmark_pair_seconds, dtype=np.float64)
    if (
        timings.shape != (100,)
        or not np.isfinite(timings).all()
        or bool((timings <= 0).any())
    ):
        raise ModelError("bridge preflight requires 100 positive finite pair timings")
    _positive_integer(projected_pair_fits, "projected_pair_fits")
    _positive_integer(workers, "workers")
    if not np.isfinite(peak_memory_fraction) or peak_memory_fraction < 0.0:
        raise ModelError("peak memory fraction must be finite and non-negative")
    if not np.isfinite(maximum_hours) or maximum_hours <= 0.0:
        raise ModelError("maximum hours must be finite and positive")
    if (
        not np.isfinite(maximum_memory_fraction)
        or not 0.0 < maximum_memory_fraction <= 1.0
    ):
        raise ModelError("maximum memory fraction must be finite and in (0, 1]")
    worst = float(np.max(timings))
    projected_hours = worst * projected_pair_fits / workers / 3_600.0
    reasons: list[str] = []
    if workers != 9:
        reasons.append("worker_count_is_not_binding_nine")
    if projected_hours > maximum_hours:
        reasons.append("projected_wall_clock_exceeds_limit")
    if peak_memory_fraction > maximum_memory_fraction:
        reasons.append("peak_memory_exceeds_limit")
    return BridgeResourcePreflight(
        100,
        workers,
        projected_hours,
        float(peak_memory_fraction),
        float(maximum_hours),
        float(maximum_memory_fraction),
        not reasons,
        tuple(reasons),
    )


def _one_sided_binomial_lower(
    successes: int, trials: int, *, confidence: float
) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def _absolute_finite(values: ArrayLike, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ModelError(f"{label} must be non-empty and finite")
    return np.abs(result)


def _counts(values: ArrayLike, states: int, frequency: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (states,)
        or not np.isfinite(result).all()
        or bool((result <= 0).any())
    ):
        raise ModelError(
            f"design/production {frequency} state counts must be positive and match K"
        )
    return result


def _boolean_matrix(values: ArrayLike, shape: tuple[int, int], label: str) -> BoolArray:
    result = np.asarray(values)
    if result.shape != shape or result.dtype.kind != "b":
        raise ModelError(
            f"{label} must be an exact {shape[0]}-by-{shape[1]} boolean matrix"
        )
    return result.astype(np.bool_, copy=False)


def _finite_matrix(values: ArrayLike, shape: tuple[int, int], label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ModelError(
            f"{label} must be an exact finite {shape[0]}-by-{shape[1]} matrix"
        )
    return result


def _positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")


def _bounded_block_length(
    value: int,
    label: str,
    bounds: tuple[int, int],
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not bounds[0] <= value <= bounds[1]
    ):
        raise ModelError(
            f"{label} is outside the approved bounds",
            details={
                "lower_bound": bounds[0],
                "upper_bound": bounds[1],
                "value": value,
            },
        )


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
