from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from prpg.errors import IntegrityError
from prpg.validation.records import (
    CanonicalThresholds,
    ValidationDecision,
    build_canonical_results,
    build_canonical_thresholds,
    canonical_record_bytes,
)
from prpg.validation.statistics import (
    empirical_higher_critical_value,
    pointwise_outer_null_audit,
)
from prpg.validation.store import ValidationArtifactStore

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


def _thresholds() -> CanonicalThresholds:
    values = np.arange(50_000, dtype=np.float64)
    return build_canonical_thresholds(
        threshold_set_id="g5.production.v1",
        binding_fingerprints={"model": _A, "config": _B},
        sobol_direction_fingerprint=_C,
        critical_values={
            "joint": empirical_higher_critical_value(values, 0.95),
            "tails": empirical_higher_critical_value(values[::-1], 0.99),
        },
        audit_intervals={
            "joint_tail": pointwise_outer_null_audit(2_500),
            "tail_tail": pointwise_outer_null_audit(500),
        },
    )


def test_threshold_value_is_order_independent_fingerprinted_and_frozen() -> None:
    first = _thresholds()
    values = np.arange(50_000, dtype=np.float64)
    second = build_canonical_thresholds(
        threshold_set_id="g5.production.v1",
        binding_fingerprints={"config": _B, "model": _A},
        sobol_direction_fingerprint=_C,
        critical_values={
            "tails": empirical_higher_critical_value(values[::-1], 0.99),
            "joint": empirical_higher_critical_value(values, 0.95),
        },
        audit_intervals={
            "tail_tail": pointwise_outer_null_audit(500),
            "joint_tail": pointwise_outer_null_audit(2_500),
        },
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.precision_passed
    assert canonical_record_bytes(first) == canonical_record_bytes(second)
    with pytest.raises(FrozenInstanceError):
        first.threshold_set_id = "changed"  # type: ignore[misc]


def test_results_required_failures_cannot_be_rescued_by_diagnostics() -> None:
    thresholds = _thresholds()
    passing = build_canonical_results(
        result_id="g5.candidate.historical_vector",
        threshold_fingerprint=thresholds.fingerprint,
        subject_fingerprint=_D,
        metric_fingerprints={"daily": _A, "monthly": _B},
        decisions=(
            ValidationDecision("diagnostic", False, False, 1.2, "report only"),
            ValidationDecision("required", True, True, 0.02, "<= 0.05"),
        ),
    )
    failing = build_canonical_results(
        result_id="g5.candidate.historical_vector",
        threshold_fingerprint=thresholds.fingerprint,
        subject_fingerprint=_D,
        metric_fingerprints={"monthly": _B, "daily": _A},
        decisions=(
            ValidationDecision("required", True, False, 0.08, "<= 0.05"),
            ValidationDecision("diagnostic", False, True, "good", "report only"),
        ),
    )

    assert passing.passed
    assert not failing.passed
    assert tuple(item.name for item in passing.decisions) == ("diagnostic", "required")


def test_validation_store_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    thresholds = _thresholds()
    results = build_canonical_results(
        result_id="g5.candidate.historical_vector",
        threshold_fingerprint=thresholds.fingerprint,
        subject_fingerprint=_D,
        metric_fingerprints={"base": _A},
        decisions=(ValidationDecision("required", True, True, True, "must pass"),),
    )
    store = ValidationArtifactStore(tmp_path)

    threshold_record = store.publish_thresholds(thresholds)
    result_record = store.publish_results(results)

    assert threshold_record.path.read_bytes() == canonical_record_bytes(thresholds)
    assert result_record.path.read_bytes() == canonical_record_bytes(results)
    assert store.load_thresholds(thresholds.fingerprint) == thresholds
    assert store.load_results(results.fingerprint) == results
    with pytest.raises(IntegrityError, match="overwrite refused"):
        store.publish_thresholds(thresholds)
    with pytest.raises(IntegrityError, match="overwrite refused"):
        store.publish_results(results)


def test_validation_store_fails_closed_on_changed_bytes(tmp_path: Path) -> None:
    thresholds = _thresholds()
    store = ValidationArtifactStore(tmp_path)
    record = store.publish_thresholds(thresholds)
    changed = record.path.read_bytes().replace(
        b'"precision_passed":true', b'"precision_passed":false'
    )
    record.path.write_bytes(changed)

    with pytest.raises(IntegrityError, match="stored threshold"):
        store.load_thresholds(thresholds.fingerprint)
