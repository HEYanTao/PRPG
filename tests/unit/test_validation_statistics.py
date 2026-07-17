from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ValidationError
from prpg.validation.statistics import (
    OUTER_NULL_REPLICATES,
    POWER_CLAIM_NAMES,
    bonferroni_power_audit,
    empirical_higher_critical_value,
    holm_decisions,
    meta_type_i_audit,
    pointwise_outer_null_audit,
)


def test_empirical_higher_critical_value_records_exact_numpy_rank() -> None:
    values = np.arange(OUTER_NULL_REPLICATES, dtype=np.float64)[::-1]

    result = empirical_higher_critical_value(values, 0.95)

    assert result.method == "higher"
    assert result.zero_based_index == 47_500
    assert result.one_based_rank == 47_501
    assert result.value == np.quantile(values, 0.95, method="higher")


@pytest.mark.parametrize("size", [49_999, 50_001])
def test_empirical_critical_value_rejects_any_other_run_count(size: int) -> None:
    with pytest.raises(ValidationError, match="exactly 50,000"):
        empirical_higher_critical_value(np.zeros(size), 0.95)


def test_pointwise_outer_null_precision_is_attainable_for_every_count() -> None:
    maximum = -1.0
    maximum_counts: list[int] = []
    for exceedances in range(OUTER_NULL_REPLICATES + 1):
        result = pointwise_outer_null_audit(exceedances)
        assert result.precision_passed
        assert result.lower <= result.estimate <= result.upper
        if result.conservative_half_width > maximum:
            maximum = result.conservative_half_width
            maximum_counts = [exceedances]
        elif result.conservative_half_width == maximum:
            maximum_counts.append(exceedances)

    assert maximum == pytest.approx(0.004392602182149319)
    assert maximum_counts == [24_834, 25_166]


def test_holm_is_deterministic_in_name_order_and_stops_after_first_failure() -> None:
    first = holm_decisions({"d": 0.040, "b": 0.010, "a": 0.010, "c": 0.030})
    second = holm_decisions({"c": 0.030, "a": 0.010, "d": 0.040, "b": 0.010})

    assert first == second
    assert tuple(item.name for item in first.decisions) == ("a", "b", "c", "d")
    assert tuple(item.rejected for item in first.decisions) == (
        True,
        True,
        False,
        False,
    )
    assert tuple(item.step_rank for item in first.decisions) == (1, 2, 3, 4)


@pytest.mark.parametrize(
    ("false_rejections", "passed"),
    [(36, False), (37, True), (64, True), (65, False)],
)
def test_meta_type_i_exact_acceptance_boundaries(
    false_rejections: int,
    passed: bool,
) -> None:
    result = meta_type_i_audit(false_rejections)

    assert result.passed is passed
    assert result.accepted_count_range == (37, 64)
    assert (result.lower <= 0.05 <= result.upper) is passed


def test_bonferroni_power_exact_924_925_boundary() -> None:
    detections = dict.fromkeys(POWER_CLAIM_NAMES, 925)
    detections[POWER_CLAIM_NAMES[0]] = 924

    failed = bonferroni_power_audit(detections)
    passed = bonferroni_power_audit(dict.fromkeys(POWER_CLAIM_NAMES, 925))

    assert not failed.passed
    assert failed.claims[0].lower_confidence_bound == pytest.approx(0.8992139172061988)
    assert failed.claims[1].lower_confidence_bound == pytest.approx(0.9003416097679645)
    assert passed.passed
    assert all(item.minimum_detections == 925 for item in passed.claims)


def test_bonferroni_power_requires_exact_twelve_claim_registry() -> None:
    detections = dict.fromkeys(POWER_CLAIM_NAMES[:-1], 1_000)

    with pytest.raises(ValidationError, match="twelve-claim registry"):
        bonferroni_power_audit(detections)
