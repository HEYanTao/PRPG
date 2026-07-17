"""Fail-closed evidence contract for the Phase-3 quality gate.

The statistical modules calculate individual results.  This module defines the
single immutable checklist that records whether G3 passed.  G3 is deliberately
evidence-only: even a complete passing decision never authorizes generation.
Generation authority belongs to a later, separately sealed G5 qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from prpg.errors import ModelError
from prpg.model.g3_registry import G3_GATE_ORDER

GateStatus: TypeAlias = Literal["passed", "failed"]

G3_SCHEMA_VERSION = 2
G3_CONTRACT_VERSION = "section-4.10-owner-approved-fixed-k4-20260715-v3"
G3_RESOURCE_PROJECTION_VERSION = "g3-rehearsal-resource-projection-v3"
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GATE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class G3GateRecord:
    """One required scientific or integrity decision and its evidence hash."""

    gate_id: str
    status: GateStatus
    evidence_fingerprint: str
    summary: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "evidence_fingerprint": self.evidence_fingerprint,
            "summary": dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class G3EvidenceBundle:
    """Complete owner-approved G3 decision, suitable for canonical JSON.

    ``passed`` records the G3 outcome.  ``generation_authorized`` is a fixed
    false contract field so persisted readers cannot confuse a scientific
    gate decision with the later G5 generation capability.
    """

    schema_version: int
    contract_version: str
    config_fingerprint: str
    processed_data_fingerprint: str
    calibration_input_fingerprint: str
    design_artifact_fingerprint: str | None
    production_artifact_fingerprint: str | None
    records: tuple[G3GateRecord, ...]
    passed: bool
    generation_authorized: Literal[False]
    fingerprint: str

    def identity_dict(self) -> dict[str, Any]:
        """Return the exact identity whose canonical bytes form the hash."""

        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "config_fingerprint": self.config_fingerprint,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "calibration_input_fingerprint": self.calibration_input_fingerprint,
            "design_artifact_fingerprint": self.design_artifact_fingerprint,
            "production_artifact_fingerprint": self.production_artifact_fingerprint,
            "records": [record.as_dict() for record in self.records],
            "passed": self.passed,
            "generation_authorized": self.generation_authorized,
        }

    def canonical_json(self) -> str:
        """Serialize the bundle identity with deterministic JSON spelling."""

        return _canonical_json(self.identity_dict())


def gate_record(
    gate_id: str,
    *,
    passed: bool,
    evidence: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> G3GateRecord:
    """Create a record whose hash binds the complete supporting evidence."""

    if gate_id not in G3_GATE_ORDER or not _GATE_ID_PATTERN.fullmatch(gate_id):
        raise ModelError("G3 gate ID is not in the closed registry")
    if not isinstance(passed, bool):
        raise ModelError("G3 gate decision must be boolean")
    normalized_evidence = _normalized_mapping(evidence, "G3 gate evidence")
    normalized_summary = _normalized_mapping(summary, "G3 gate summary")
    evidence_identity = {
        "schema_version": G3_SCHEMA_VERSION,
        "contract_version": G3_CONTRACT_VERSION,
        "gate_id": gate_id,
        "passed": passed,
        "evidence": normalized_evidence,
    }
    fingerprint = _sha256(_canonical_json(evidence_identity).encode())
    return G3GateRecord(
        gate_id,
        "passed" if passed else "failed",
        fingerprint,
        MappingProxyType(normalized_summary),
    )


def build_g3_evidence_bundle(
    *,
    config_fingerprint: str,
    processed_data_fingerprint: str,
    calibration_input_fingerprint: str,
    design_artifact_fingerprint: str | None,
    production_artifact_fingerprint: str | None,
    records: Sequence[G3GateRecord],
) -> G3EvidenceBundle:
    """Build an evidence-only G3 decision bundle."""

    return _build_g3_evidence_bundle(
        config_fingerprint=config_fingerprint,
        processed_data_fingerprint=processed_data_fingerprint,
        calibration_input_fingerprint=calibration_input_fingerprint,
        design_artifact_fingerprint=design_artifact_fingerprint,
        production_artifact_fingerprint=production_artifact_fingerprint,
        records=records,
    )


def _build_g3_evidence_bundle(
    *,
    config_fingerprint: str,
    processed_data_fingerprint: str,
    calibration_input_fingerprint: str,
    design_artifact_fingerprint: str | None,
    production_artifact_fingerprint: str | None,
    records: Sequence[G3GateRecord],
) -> G3EvidenceBundle:
    """Build a bundle whose only positive outcome is ``passed``."""

    _fingerprint(config_fingerprint, "config fingerprint")
    _fingerprint(processed_data_fingerprint, "processed data fingerprint")
    _fingerprint(calibration_input_fingerprint, "calibration input fingerprint")
    if design_artifact_fingerprint is not None:
        _fingerprint(design_artifact_fingerprint, "design artifact fingerprint")
    if production_artifact_fingerprint is not None:
        _fingerprint(production_artifact_fingerprint, "production artifact fingerprint")
    supplied = tuple(records)
    actual_order = tuple(record.gate_id for record in supplied)
    if actual_order != G3_GATE_ORDER:
        raise ModelError(
            "G3 records must contain the exact closed gate order",
            details={
                "expected": list(G3_GATE_ORDER),
                "actual": list(actual_order),
            },
        )
    for record in supplied:
        _validate_record(record)
    all_passed = all(record.passed for record in supplied)
    artifacts_present = (
        design_artifact_fingerprint is not None
        and production_artifact_fingerprint is not None
    )
    passed = all_passed and artifacts_present
    identity: dict[str, Any] = {
        "schema_version": G3_SCHEMA_VERSION,
        "contract_version": G3_CONTRACT_VERSION,
        "config_fingerprint": config_fingerprint,
        "processed_data_fingerprint": processed_data_fingerprint,
        "calibration_input_fingerprint": calibration_input_fingerprint,
        "design_artifact_fingerprint": design_artifact_fingerprint,
        "production_artifact_fingerprint": production_artifact_fingerprint,
        "records": [record.as_dict() for record in supplied],
        "passed": passed,
        "generation_authorized": False,
    }
    fingerprint = _sha256(_canonical_json(identity).encode())
    bundle = G3EvidenceBundle(
        G3_SCHEMA_VERSION,
        G3_CONTRACT_VERSION,
        config_fingerprint,
        processed_data_fingerprint,
        calibration_input_fingerprint,
        design_artifact_fingerprint,
        production_artifact_fingerprint,
        supplied,
        passed,
        False,
        fingerprint,
    )
    return bundle


def verify_g3_evidence_bundle(bundle: G3EvidenceBundle) -> None:
    """Rebuild and compare an evidence-only G3 bundle."""

    if not isinstance(bundle, G3EvidenceBundle):
        raise ModelError("G3 evidence bundle has the wrong type")
    rebuilt = _build_g3_evidence_bundle(
        config_fingerprint=bundle.config_fingerprint,
        processed_data_fingerprint=bundle.processed_data_fingerprint,
        calibration_input_fingerprint=bundle.calibration_input_fingerprint,
        design_artifact_fingerprint=bundle.design_artifact_fingerprint,
        production_artifact_fingerprint=bundle.production_artifact_fingerprint,
        records=bundle.records,
    )
    if bundle.schema_version != G3_SCHEMA_VERSION:
        raise ModelError("G3 bundle schema version is unsupported")
    if bundle.contract_version != G3_CONTRACT_VERSION:
        raise ModelError("G3 bundle contract version is unsupported")
    if bundle.generation_authorized is not False:
        raise ModelError("G3 bundle must be evidence-only")
    if bundle.identity_dict() != rebuilt.identity_dict():
        raise ModelError("G3 bundle decision fields are inconsistent")
    if bundle.fingerprint != rebuilt.fingerprint:
        raise ModelError("G3 bundle fingerprint is inconsistent")


def _validate_record(record: G3GateRecord) -> None:
    if not isinstance(record, G3GateRecord):
        raise ModelError("G3 records must use the immutable gate-record type")
    if record.gate_id not in G3_GATE_ORDER:
        raise ModelError("G3 record contains an unregistered gate")
    if record.status not in {"passed", "failed"}:
        raise ModelError("G3 record status is invalid")
    _fingerprint(record.evidence_fingerprint, "gate evidence fingerprint")
    normalized = _normalized_mapping(record.summary, "G3 gate summary")
    if dict(record.summary) != normalized:
        raise ModelError("G3 record summary is not canonical")


def _normalized_mapping(values: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(values, Mapping) or not values:
        raise ModelError(f"{label} must be a non-empty mapping")
    normalized = _normalize_json(dict(values), path=label)
    if not isinstance(normalized, dict):
        raise AssertionError("mapping normalization did not return a dictionary")
    # The canonical serialization is also the definitive JSON encodability test.
    _canonical_json(normalized)
    return normalized


def _normalize_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError(f"{path} contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ModelError(f"{path} contains an invalid JSON key")
            lowered = key.casefold()
            if any(marker in lowered for marker in _SECRET_MARKERS):
                raise ModelError(f"{path} contains a secret-like key")
            result[key] = _normalize_json(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_normalize_json(item, path=f"{path}[]") for item in value]
    raise ModelError(
        f"{path} contains an unsupported JSON value",
        details={"type": type(value).__name__},
    )


def _fingerprint(value: str, label: str) -> None:
    if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise ModelError(f"{label} must be a lowercase SHA-256 fingerprint")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
