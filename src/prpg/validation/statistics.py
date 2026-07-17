"""Frozen G5 Monte Carlo and multiplicity calculations.

The functions here consume statistics or counts produced elsewhere.  They do
not own a random stream and therefore remain cheap to unit test and reusable
by both calibration and later production validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

from prpg.errors import ValidationError

OUTER_NULL_REPLICATES: Final = 50_000
OUTER_NULL_AUDIT_CONFIDENCE: Final = 0.95
OUTER_NULL_MAX_HALF_WIDTH: Final = 0.005
HOLM_FAMILYWISE_ALPHA: Final = 0.05
META_REPLICATES: Final = 1_000
META_TYPE_I_MIN_FALSE_REJECTIONS: Final = 37
META_TYPE_I_MAX_FALSE_REJECTIONS: Final = 64
META_REQUIRED_POWER: Final = 0.90
META_MIN_DETECTIONS: Final = 925

# The approved twelve-claim registry.  A mapping with a missing, extra, or
# misspelled claim fails before any confidence bound is reported.
POWER_CLAIM_NAMES: Final = (
    "linear_alpha_sharpe",
    "normalized_covariance",
    "maximum_correlation",
    "one_session_desynchronization",
    "tail_clipping",
    "transition_perturbation",
    "hmm_merged_states",
    "hmm_overlapping_states",
    "nonlinear_daily",
    "nonlinear_monthly",
    "subsequence_daily",
    "subsequence_monthly",
)
POWER_CLAIM_COUNT: Final = len(POWER_CLAIM_NAMES)


@dataclass(frozen=True, slots=True)
class EmpiricalCriticalValue:
    """An exact NumPy ``method='higher'`` order statistic."""

    probability: float
    sample_count: int
    method: Literal["higher"]
    zero_based_index: int
    one_based_rank: int
    value: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the canonical JSON-compatible representation."""

        return {
            "probability": self.probability,
            "sample_count": self.sample_count,
            "method": self.method,
            "zero_based_index": self.zero_based_index,
            "one_based_rank": self.one_based_rank,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class PointwiseAuditInterval:
    """Ordinary pointwise two-sided 95% Clopper-Pearson audit."""

    exceedances: int
    trials: int
    estimate: float
    confidence: float
    lower: float
    upper: float
    conservative_half_width: float
    maximum_half_width: float
    precision_passed: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return the canonical JSON-compatible representation."""

        return {
            "exceedances": self.exceedances,
            "trials": self.trials,
            "estimate": self.estimate,
            "confidence": self.confidence,
            "lower": self.lower,
            "upper": self.upper,
            "conservative_half_width": self.conservative_half_width,
            "maximum_half_width": self.maximum_half_width,
            "precision_passed": self.precision_passed,
        }


@dataclass(frozen=True, slots=True)
class HolmDecision:
    """One original family member after deterministic Holm step-down."""

    name: str
    raw_p_value: float
    adjusted_p_value: float
    step_rank: int
    comparison_alpha: float
    rejected: bool

    def as_dict(self) -> dict[str, float | int | str | bool]:
        """Return the canonical JSON-compatible representation."""

        return {
            "name": self.name,
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "step_rank": self.step_rank,
            "comparison_alpha": self.comparison_alpha,
            "rejected": self.rejected,
        }


@dataclass(frozen=True, slots=True)
class HolmAudit:
    """Complete alpha-0.05 Holm result in canonical name order."""

    familywise_alpha: float
    decisions: tuple[HolmDecision, ...]
    any_rejected: bool


@dataclass(frozen=True, slots=True)
class MetaTypeIAudit:
    """The approved 1,000-null complete-suite false-rejection gate."""

    false_rejections: int
    trials: int
    estimate: float
    lower: float
    upper: float
    target: float
    accepted_count_range: tuple[int, int]
    passed: bool

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""

        return {
            "false_rejections": self.false_rejections,
            "trials": self.trials,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "target": self.target,
            "accepted_count_range": list(self.accepted_count_range),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PowerClaimAudit:
    """One Bonferroni-adjusted one-sided Clopper-Pearson power bound."""

    name: str
    detections: int
    trials: int
    estimate: float
    lower_confidence_bound: float
    per_claim_alpha: float
    required_power: float
    minimum_detections: int
    passed: bool

    def as_dict(self) -> dict[str, float | int | str | bool]:
        """Return the canonical JSON-compatible representation."""

        return {
            "name": self.name,
            "detections": self.detections,
            "trials": self.trials,
            "estimate": self.estimate,
            "lower_confidence_bound": self.lower_confidence_bound,
            "per_claim_alpha": self.per_claim_alpha,
            "required_power": self.required_power,
            "minimum_detections": self.minimum_detections,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class MetaPowerAudit:
    """All twelve registered power claims under one Bonferroni family."""

    familywise_alpha: float
    per_claim_alpha: float
    claims: tuple[PowerClaimAudit, ...]
    passed: bool


def empirical_higher_critical_value(
    outer_null_statistics: ArrayLike,
    probability: float,
) -> EmpiricalCriticalValue:
    """Select the exact higher empirical quantile from exactly 50,000 values.

    NumPy's ``higher`` method selects sorted index
    ``ceil(probability * (n - 1))``.  Both its zero-based index and human-
    readable one-based rank are retained as evidence; a binomial interval is
    intentionally not attached to a data-selected critical quantile.
    """

    values = np.asarray(outer_null_statistics, dtype=np.float64)
    if values.shape != (OUTER_NULL_REPLICATES,):
        raise ValidationError(
            "outer-null critical values require exactly 50,000 statistics",
            details={"shape": list(values.shape)},
        )
    if not bool(np.isfinite(values).all()):
        raise ValidationError("outer-null statistics must all be finite")
    checked_probability = _probability(probability, "critical probability")
    zero_based = math.ceil(checked_probability * (values.size - 1))
    critical = float(np.quantile(values, checked_probability, method="higher"))
    # Assert the documented rank and library selection remain identical.
    ranked_value = float(np.partition(values.copy(), zero_based)[zero_based])
    if critical != ranked_value:  # pragma: no cover - pinned NumPy invariant
        raise ValidationError("higher-quantile rank does not match NumPy")
    return EmpiricalCriticalValue(
        probability=checked_probability,
        sample_count=OUTER_NULL_REPLICATES,
        method="higher",
        zero_based_index=zero_based,
        one_based_rank=zero_based + 1,
        value=critical,
    )


def pointwise_outer_null_audit(exceedances: int) -> PointwiseAuditInterval:
    """Audit one fixed exceedance event over exactly 50,000 IID replicates."""

    count = _count(exceedances, OUTER_NULL_REPLICATES, "outer-null exceedances")
    estimate, lower, upper = _two_sided_clopper_pearson(
        count,
        OUTER_NULL_REPLICATES,
        confidence=OUTER_NULL_AUDIT_CONFIDENCE,
    )
    half_width = max(estimate - lower, upper - estimate)
    return PointwiseAuditInterval(
        exceedances=count,
        trials=OUTER_NULL_REPLICATES,
        estimate=estimate,
        confidence=OUTER_NULL_AUDIT_CONFIDENCE,
        lower=lower,
        upper=upper,
        conservative_half_width=half_width,
        maximum_half_width=OUTER_NULL_MAX_HALF_WIDTH,
        precision_passed=half_width <= OUTER_NULL_MAX_HALF_WIDTH,
    )


def holm_decisions(p_values: Mapping[str, float]) -> HolmAudit:
    """Apply alpha-0.05 Holm correction with stable ``(p, name)`` ordering."""

    if not isinstance(p_values, Mapping) or not p_values:
        raise ValidationError("Holm p-values must be a non-empty mapping")
    checked: dict[str, float] = {}
    for name, value in p_values.items():
        checked_name = _name(name, "Holm family name")
        if checked_name in checked:
            raise ValidationError("Holm family names must be unique")
        checked[checked_name] = _probability(value, f"Holm p-value {checked_name}")

    ordered = sorted(checked, key=lambda name: (checked[name], name))
    size = len(ordered)
    adjusted_by_name: dict[str, float] = {}
    rejected_by_name: dict[str, bool] = {}
    rank_by_name: dict[str, int] = {}
    alpha_by_name: dict[str, float] = {}
    running_adjusted = 0.0
    still_rejecting = True
    for zero_rank, name in enumerate(ordered):
        remaining = size - zero_rank
        comparison_alpha = HOLM_FAMILYWISE_ALPHA / remaining
        candidate_adjusted = min(1.0, remaining * checked[name])
        running_adjusted = max(running_adjusted, candidate_adjusted)
        rejected = still_rejecting and checked[name] <= comparison_alpha
        if not rejected:
            still_rejecting = False
        adjusted_by_name[name] = running_adjusted
        rejected_by_name[name] = rejected
        rank_by_name[name] = zero_rank + 1
        alpha_by_name[name] = comparison_alpha

    decisions = tuple(
        HolmDecision(
            name=name,
            raw_p_value=checked[name],
            adjusted_p_value=adjusted_by_name[name],
            step_rank=rank_by_name[name],
            comparison_alpha=alpha_by_name[name],
            rejected=rejected_by_name[name],
        )
        for name in sorted(checked)
    )
    return HolmAudit(
        familywise_alpha=HOLM_FAMILYWISE_ALPHA,
        decisions=decisions,
        any_rejected=any(item.rejected for item in decisions),
    )


def meta_type_i_audit(false_rejections: int) -> MetaTypeIAudit:
    """Evaluate the exact 37..64 acceptance rule for 1,000 null runs."""

    count = _count(false_rejections, META_REPLICATES, "meta type-I rejections")
    estimate, lower, upper = _two_sided_clopper_pearson(
        count,
        META_REPLICATES,
        confidence=0.95,
    )
    passed = (
        META_TYPE_I_MIN_FALSE_REJECTIONS <= count <= META_TYPE_I_MAX_FALSE_REJECTIONS
    )
    # This is an approved exact-count restatement of interval containment.
    if passed != (lower <= HOLM_FAMILYWISE_ALPHA <= upper):
        raise ValidationError("type-I count rule and exact interval diverged")
    return MetaTypeIAudit(
        false_rejections=count,
        trials=META_REPLICATES,
        estimate=estimate,
        lower=lower,
        upper=upper,
        target=HOLM_FAMILYWISE_ALPHA,
        accepted_count_range=(
            META_TYPE_I_MIN_FALSE_REJECTIONS,
            META_TYPE_I_MAX_FALSE_REJECTIONS,
        ),
        passed=passed,
    )


def bonferroni_power_audit(
    detections: Mapping[str, int],
) -> MetaPowerAudit:
    """Evaluate all twelve registered 1,000-run material-power claims.

    Each claim receives one-sided tail probability ``0.05 / 12``.  A claim
    passes only when its exact lower bound is at least 0.90; under the frozen
    sample size this is equivalent to at least 925 detections.
    """

    if not isinstance(detections, Mapping):
        raise ValidationError("power detections must be a claim mapping")
    if any(not isinstance(name, str) for name in detections):
        raise ValidationError("power claim names must be strings")
    observed_names = set(detections)
    required_names = set(POWER_CLAIM_NAMES)
    if observed_names != required_names:
        raise ValidationError(
            "power detections do not match the twelve-claim registry",
            details={
                "missing": sorted(required_names - observed_names),
                "extra": sorted(observed_names - required_names),
            },
        )
    per_claim_alpha = HOLM_FAMILYWISE_ALPHA / POWER_CLAIM_COUNT
    claims: list[PowerClaimAudit] = []
    for name in POWER_CLAIM_NAMES:
        count = _count(detections[name], META_REPLICATES, f"power detections {name}")
        lower = _one_sided_clopper_pearson_lower(
            count,
            META_REPLICATES,
            alpha=per_claim_alpha,
        )
        bound_passed = lower >= META_REQUIRED_POWER
        count_passed = count >= META_MIN_DETECTIONS
        if bound_passed != count_passed:
            raise ValidationError("power count rule and exact bound diverged")
        claims.append(
            PowerClaimAudit(
                name=name,
                detections=count,
                trials=META_REPLICATES,
                estimate=count / META_REPLICATES,
                lower_confidence_bound=lower,
                per_claim_alpha=per_claim_alpha,
                required_power=META_REQUIRED_POWER,
                minimum_detections=META_MIN_DETECTIONS,
                passed=bound_passed,
            )
        )
    result = tuple(claims)
    return MetaPowerAudit(
        familywise_alpha=HOLM_FAMILYWISE_ALPHA,
        per_claim_alpha=per_claim_alpha,
        claims=result,
        passed=all(item.passed for item in result),
    )


def _two_sided_clopper_pearson(
    successes: int,
    trials: int,
    *,
    confidence: float,
) -> tuple[float, float, float]:
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    estimate = successes / trials
    if not all(math.isfinite(value) for value in (estimate, lower, upper)):
        raise ValidationError("Clopper-Pearson interval is non-finite")
    return float(estimate), lower, upper


def _one_sided_clopper_pearson_lower(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    if successes == 0:
        return 0.0
    lower = float(beta.ppf(alpha, successes, trials - successes + 1))
    if not math.isfinite(lower):
        raise ValidationError("one-sided Clopper-Pearson bound is non-finite")
    return lower


def _count(value: object, maximum: int, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValidationError(f"{label} must be an integer in [0, {maximum}]")
    return value


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{label} must be a finite value in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValidationError(f"{label} must be a finite value in [0, 1]")
    return result


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationError(f"{label} must be a non-empty canonical string")
    return value
