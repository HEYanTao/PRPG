"""Immutable canonical G5 threshold and validation-result value objects."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, cast

from prpg.errors import IntegrityError, ValidationError
from prpg.validation.statistics import (
    OUTER_NULL_AUDIT_CONFIDENCE,
    OUTER_NULL_MAX_HALF_WIDTH,
    OUTER_NULL_REPLICATES,
    EmpiricalCriticalValue,
    PointwiseAuditInterval,
    pointwise_outer_null_audit,
)

THRESHOLD_SCHEMA_ID: Final = "prpg-g5-canonical-thresholds-v1"
RESULT_SCHEMA_ID: Final = "prpg-g5-canonical-results-v1"
VALIDATION_RECORD_SCHEMA_VERSION: Final = 1

ObservedValue: TypeAlias = str | float | int | bool | None

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class ThresholdValue:
    """One named fixed-sample empirical critical value."""

    name: str
    probability: float
    sample_count: int
    quantile_method: str
    one_based_rank: int
    critical_value: float

    def as_dict(self) -> dict[str, str | float | int]:
        return {
            "name": self.name,
            "probability": self.probability,
            "sample_count": self.sample_count,
            "quantile_method": self.quantile_method,
            "one_based_rank": self.one_based_rank,
            "critical_value": self.critical_value,
        }


@dataclass(frozen=True, slots=True)
class AuditIntervalValue:
    """One named fixed-event pointwise precision audit."""

    name: str
    exceedances: int
    trials: int
    estimate: float
    confidence: float
    lower: float
    upper: float
    conservative_half_width: float
    maximum_half_width: float
    precision_passed: bool

    def as_dict(self) -> dict[str, str | float | int | bool]:
        return {
            "name": self.name,
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
class CanonicalThresholds:
    """Canonical thresholds bound to exact upstream scientific identities."""

    schema_id: str
    schema_version: int
    threshold_set_id: str
    binding_fingerprints: tuple[tuple[str, str], ...]
    sobol_direction_fingerprint: str
    thresholds: tuple[ThresholdValue, ...]
    audit_intervals: tuple[AuditIntervalValue, ...]
    precision_passed: bool
    fingerprint: str

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "threshold_set_id": self.threshold_set_id,
            "binding_fingerprints": dict(self.binding_fingerprints),
            "sobol_direction_fingerprint": self.sobol_direction_fingerprint,
            "thresholds": [item.as_dict() for item in self.thresholds],
            "audit_intervals": [item.as_dict() for item in self.audit_intervals],
            "precision_passed": self.precision_passed,
        }

    def as_dict(self) -> dict[str, object]:
        result = self.identity_dict()
        result["fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    """One named required or diagnostic observable outcome."""

    name: str
    required: bool
    passed: bool
    observed: ObservedValue
    criterion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
            "passed": self.passed,
            "observed": self.observed,
            "criterion": self.criterion,
        }


@dataclass(frozen=True, slots=True)
class CanonicalValidationResults:
    """Final immutable decisions bound to thresholds, subject, and metrics."""

    schema_id: str
    schema_version: int
    result_id: str
    threshold_fingerprint: str
    subject_fingerprint: str
    metric_fingerprints: tuple[tuple[str, str], ...]
    decisions: tuple[ValidationDecision, ...]
    passed: bool
    fingerprint: str

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "threshold_fingerprint": self.threshold_fingerprint,
            "subject_fingerprint": self.subject_fingerprint,
            "metric_fingerprints": dict(self.metric_fingerprints),
            "decisions": [item.as_dict() for item in self.decisions],
            "passed": self.passed,
        }

    def as_dict(self) -> dict[str, object]:
        result = self.identity_dict()
        result["fingerprint"] = self.fingerprint
        return result


def build_canonical_thresholds(
    *,
    threshold_set_id: str,
    binding_fingerprints: Mapping[str, str],
    sobol_direction_fingerprint: str,
    critical_values: Mapping[str, EmpiricalCriticalValue],
    audit_intervals: Mapping[str, PointwiseAuditInterval],
) -> CanonicalThresholds:
    """Build a name-order-independent canonical G5 threshold artifact."""

    identifier = _identifier(threshold_set_id, "threshold set ID")
    bindings = _fingerprint_mapping(binding_fingerprints, "threshold binding")
    sobol = _fingerprint(sobol_direction_fingerprint, "Sobol direction")
    if not isinstance(critical_values, Mapping) or not critical_values:
        raise ValidationError("at least one empirical critical value is required")
    thresholds: list[ThresholdValue] = []
    for name in sorted(critical_values):
        checked_name = _identifier(name, "threshold name")
        critical = critical_values[name]
        if not isinstance(critical, EmpiricalCriticalValue):
            raise ValidationError("critical value has the wrong type")
        expected_index = math.ceil(critical.probability * (OUTER_NULL_REPLICATES - 1))
        if (
            not 0.0 <= critical.probability <= 1.0
            or critical.sample_count != OUTER_NULL_REPLICATES
            or critical.method != "higher"
            or critical.zero_based_index != expected_index
            or critical.one_based_rank != expected_index + 1
            or not math.isfinite(critical.value)
        ):
            raise ValidationError("empirical critical value is not canonical")
        thresholds.append(
            ThresholdValue(
                name=checked_name,
                probability=critical.probability,
                sample_count=critical.sample_count,
                quantile_method=critical.method,
                one_based_rank=critical.one_based_rank,
                critical_value=_zero(critical.value),
            )
        )
    audits: list[AuditIntervalValue] = []
    if not isinstance(audit_intervals, Mapping):
        raise ValidationError("audit intervals must be a mapping")
    for name in sorted(audit_intervals):
        checked_name = _identifier(name, "audit interval name")
        audit = audit_intervals[name]
        if not isinstance(audit, PointwiseAuditInterval):
            raise ValidationError("audit interval has the wrong type")
        expected = pointwise_outer_null_audit(audit.exceedances)
        if audit != expected:
            raise ValidationError("pointwise audit interval is not canonical")
        audits.append(
            AuditIntervalValue(
                name=checked_name,
                exceedances=audit.exceedances,
                trials=audit.trials,
                estimate=audit.estimate,
                confidence=audit.confidence,
                lower=audit.lower,
                upper=audit.upper,
                conservative_half_width=audit.conservative_half_width,
                maximum_half_width=audit.maximum_half_width,
                precision_passed=audit.precision_passed,
            )
        )
    threshold_tuple = tuple(thresholds)
    audit_tuple = tuple(audits)
    precision_passed = all(item.precision_passed for item in audit_tuple)
    identity: dict[str, object] = {
        "schema_id": THRESHOLD_SCHEMA_ID,
        "schema_version": VALIDATION_RECORD_SCHEMA_VERSION,
        "threshold_set_id": identifier,
        "binding_fingerprints": dict(bindings),
        "sobol_direction_fingerprint": sobol,
        "thresholds": [item.as_dict() for item in threshold_tuple],
        "audit_intervals": [item.as_dict() for item in audit_tuple],
        "precision_passed": precision_passed,
    }
    return CanonicalThresholds(
        schema_id=THRESHOLD_SCHEMA_ID,
        schema_version=VALIDATION_RECORD_SCHEMA_VERSION,
        threshold_set_id=identifier,
        binding_fingerprints=bindings,
        sobol_direction_fingerprint=sobol,
        thresholds=threshold_tuple,
        audit_intervals=audit_tuple,
        precision_passed=precision_passed,
        fingerprint=_sha256(identity),
    )


def build_canonical_results(
    *,
    result_id: str,
    threshold_fingerprint: str,
    subject_fingerprint: str,
    metric_fingerprints: Mapping[str, str],
    decisions: Sequence[ValidationDecision],
) -> CanonicalValidationResults:
    """Build canonical decisions; diagnostics never rescue a required failure."""

    identifier = _identifier(result_id, "validation result ID")
    threshold = _fingerprint(threshold_fingerprint, "threshold artifact")
    subject = _fingerprint(subject_fingerprint, "validation subject")
    metrics = _fingerprint_mapping(metric_fingerprints, "metric")
    if not isinstance(decisions, Sequence) or not decisions:
        raise ValidationError("validation results require at least one decision")
    by_name: dict[str, ValidationDecision] = {}
    for value in decisions:
        if not isinstance(value, ValidationDecision):
            raise ValidationError("validation decision has the wrong type")
        name = _identifier(value.name, "validation decision name")
        if name in by_name:
            raise ValidationError("validation decision names must be unique")
        if type(value.required) is not bool or type(value.passed) is not bool:
            raise ValidationError("validation decision flags must be booleans")
        observed = _observed(value.observed)
        criterion = _criterion(value.criterion)
        by_name[name] = ValidationDecision(
            name=name,
            required=value.required,
            passed=value.passed,
            observed=observed,
            criterion=criterion,
        )
    ordered = tuple(by_name[name] for name in sorted(by_name))
    passed = all(item.passed for item in ordered if item.required)
    identity: dict[str, object] = {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": VALIDATION_RECORD_SCHEMA_VERSION,
        "result_id": identifier,
        "threshold_fingerprint": threshold,
        "subject_fingerprint": subject,
        "metric_fingerprints": dict(metrics),
        "decisions": [item.as_dict() for item in ordered],
        "passed": passed,
    }
    return CanonicalValidationResults(
        schema_id=RESULT_SCHEMA_ID,
        schema_version=VALIDATION_RECORD_SCHEMA_VERSION,
        result_id=identifier,
        threshold_fingerprint=threshold,
        subject_fingerprint=subject,
        metric_fingerprints=metrics,
        decisions=ordered,
        passed=passed,
        fingerprint=_sha256(identity),
    )


def canonical_thresholds_from_dict(value: Mapping[str, object]) -> CanonicalThresholds:
    """Rebuild and revalidate a serialized threshold artifact."""

    _exact_keys(
        value,
        {
            "schema_id",
            "schema_version",
            "threshold_set_id",
            "binding_fingerprints",
            "sobol_direction_fingerprint",
            "thresholds",
            "audit_intervals",
            "precision_passed",
            "fingerprint",
        },
        "threshold artifact",
    )
    if (
        value["schema_id"] != THRESHOLD_SCHEMA_ID
        or value["schema_version"] != VALIDATION_RECORD_SCHEMA_VERSION
    ):
        raise IntegrityError("threshold artifact schema is unsupported")
    critical_values: dict[str, EmpiricalCriticalValue] = {}
    for raw in _object_list(value["thresholds"], "thresholds"):
        _exact_keys(
            raw,
            {
                "name",
                "probability",
                "sample_count",
                "quantile_method",
                "one_based_rank",
                "critical_value",
            },
            "threshold value",
        )
        probability = _float(raw["probability"], "threshold probability")
        sample_count = _int(raw["sample_count"], "threshold sample count")
        rank = _int(raw["one_based_rank"], "threshold rank")
        method = raw["quantile_method"]
        if method != "higher":
            raise IntegrityError("threshold quantile method is invalid")
        name = _string(raw["name"], "threshold name")
        if name in critical_values:
            raise IntegrityError("threshold names are duplicated")
        critical_values[name] = EmpiricalCriticalValue(
            probability=probability,
            sample_count=sample_count,
            method="higher",
            zero_based_index=rank - 1,
            one_based_rank=rank,
            value=_float(raw["critical_value"], "critical value"),
        )
    audits: dict[str, PointwiseAuditInterval] = {}
    for raw in _object_list(value["audit_intervals"], "audit intervals"):
        _exact_keys(
            raw,
            {
                "name",
                "exceedances",
                "trials",
                "estimate",
                "confidence",
                "lower",
                "upper",
                "conservative_half_width",
                "maximum_half_width",
                "precision_passed",
            },
            "audit interval",
        )
        name = _string(raw["name"], "audit interval name")
        if name in audits:
            raise IntegrityError("audit interval names are duplicated")
        audits[name] = PointwiseAuditInterval(
            exceedances=_int(raw["exceedances"], "audit exceedances"),
            trials=_int(raw["trials"], "audit trials"),
            estimate=_float(raw["estimate"], "audit estimate"),
            confidence=_float(raw["confidence"], "audit confidence"),
            lower=_float(raw["lower"], "audit lower"),
            upper=_float(raw["upper"], "audit upper"),
            conservative_half_width=_float(
                raw["conservative_half_width"], "audit half-width"
            ),
            maximum_half_width=_float(
                raw["maximum_half_width"], "audit maximum half-width"
            ),
            precision_passed=_bool(raw["precision_passed"], "audit precision flag"),
        )
    bindings = _string_mapping(value["binding_fingerprints"], "threshold bindings")
    rebuilt = build_canonical_thresholds(
        threshold_set_id=_string(value["threshold_set_id"], "threshold set ID"),
        binding_fingerprints=bindings,
        sobol_direction_fingerprint=_string(
            value["sobol_direction_fingerprint"], "Sobol fingerprint"
        ),
        critical_values=critical_values,
        audit_intervals=audits,
    )
    if (
        value["precision_passed"] is not rebuilt.precision_passed
        or value["fingerprint"] != rebuilt.fingerprint
    ):
        raise IntegrityError("threshold artifact identity is inconsistent")
    return rebuilt


def canonical_results_from_dict(
    value: Mapping[str, object],
) -> CanonicalValidationResults:
    """Rebuild and revalidate a serialized validation-results artifact."""

    _exact_keys(
        value,
        {
            "schema_id",
            "schema_version",
            "result_id",
            "threshold_fingerprint",
            "subject_fingerprint",
            "metric_fingerprints",
            "decisions",
            "passed",
            "fingerprint",
        },
        "validation results",
    )
    if (
        value["schema_id"] != RESULT_SCHEMA_ID
        or value["schema_version"] != VALIDATION_RECORD_SCHEMA_VERSION
    ):
        raise IntegrityError("validation result schema is unsupported")
    decisions: list[ValidationDecision] = []
    for raw in _object_list(value["decisions"], "validation decisions"):
        _exact_keys(
            raw,
            {"name", "required", "passed", "observed", "criterion"},
            "validation decision",
        )
        decisions.append(
            ValidationDecision(
                name=_string(raw["name"], "decision name"),
                required=_bool(raw["required"], "decision required flag"),
                passed=_bool(raw["passed"], "decision passed flag"),
                observed=_observed(raw["observed"]),
                criterion=_string(raw["criterion"], "decision criterion"),
            )
        )
    rebuilt = build_canonical_results(
        result_id=_string(value["result_id"], "validation result ID"),
        threshold_fingerprint=_string(
            value["threshold_fingerprint"], "threshold fingerprint"
        ),
        subject_fingerprint=_string(
            value["subject_fingerprint"], "subject fingerprint"
        ),
        metric_fingerprints=_string_mapping(
            value["metric_fingerprints"], "metric fingerprints"
        ),
        decisions=decisions,
    )
    if (
        value["passed"] is not rebuilt.passed
        or value["fingerprint"] != rebuilt.fingerprint
    ):
        raise IntegrityError("validation result identity is inconsistent")
    return rebuilt


def canonical_record_bytes(
    value: CanonicalThresholds | CanonicalValidationResults,
) -> bytes:
    """Serialize one supported record as sorted finite JSON plus newline."""

    if isinstance(value, CanonicalThresholds):
        if canonical_thresholds_from_dict(value.as_dict()) != value:
            raise ValidationError("threshold record is not canonical")
    elif isinstance(value, CanonicalValidationResults):
        if canonical_results_from_dict(value.as_dict()) != value:
            raise ValidationError("validation result record is not canonical")
    else:
        raise ValidationError("unsupported validation record type")
    return _canonical_bytes(value.as_dict())


def _fingerprint_mapping(
    values: Mapping[str, str], label: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise ValidationError(f"{label} fingerprints must be a non-empty mapping")
    result: list[tuple[str, str]] = []
    for name in sorted(values):
        result.append(
            (
                _identifier(name, f"{label} name"),
                _fingerprint(values[name], label),
            )
        )
    return tuple(result)


def _observed(value: object) -> ObservedValue:
    if value is None or type(value) in {str, bool, int}:
        return cast(ObservedValue, value)
    if isinstance(value, float) and math.isfinite(value):
        return _zero(value)
    raise ValidationError("decision observed value must be a finite JSON scalar")


def _criterion(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationError("decision criterion must be a non-empty canonical string")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a canonical identifier")
    return value


def _fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValidationError(f"{label} fingerprint is invalid")
    return value


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise IntegrityError(f"{label} keys are invalid")


def _object_list(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise IntegrityError(f"{label} must be a JSON array")
    result: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise IntegrityError(f"{label} entries must be JSON objects")
        result.append(cast(Mapping[str, object], item))
    return result


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} must be a JSON object")
    result: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not isinstance(item, str):
            raise IntegrityError(f"{label} entries must be strings")
        result[name] = item
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise IntegrityError(f"{label} must be a string")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IntegrityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise IntegrityError(f"{label} must be finite")
    return result


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{label} must be an integer")
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool) or type(value) is not bool:
        raise IntegrityError(f"{label} must be a boolean")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


# These constants are imported above so serialized audit validation remains
# visibly tied to the approved precision contract, even when refactoring.
assert OUTER_NULL_AUDIT_CONFIDENCE == 0.95
assert OUTER_NULL_MAX_HALF_WIDTH == 0.005
