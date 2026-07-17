from __future__ import annotations

from dataclasses import replace

import pytest

from prpg.errors import ModelError
from prpg.model.g3 import (
    G3_CONTRACT_VERSION,
    G3_GATE_ORDER,
    build_g3_evidence_bundle,
    gate_record,
    verify_g3_evidence_bundle,
)


def _records(*, failing: str | None = None):
    return tuple(
        gate_record(
            gate_id,
            passed=gate_id != failing,
            evidence={"metric": 1.0, "rows": [1, 2, 3]},
            summary={"passed": gate_id != failing},
        )
        for gate_id in G3_GATE_ORDER
    )


def test_legacy_boolean_bundle_is_complete_but_never_generation_authority() -> None:
    bundle = build_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        design_artifact_fingerprint="d" * 64,
        production_artifact_fingerprint="e" * 64,
        records=_records(),
    )

    assert bundle.passed
    assert not bundle.generation_authorized
    assert bundle.contract_version == G3_CONTRACT_VERSION
    assert len(bundle.records) == 26
    assert len(bundle.fingerprint) == 64
    verify_g3_evidence_bundle(bundle)


def test_any_failed_gate_or_missing_artifact_fails_closed() -> None:
    failed = build_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        design_artifact_fingerprint="d" * 64,
        production_artifact_fingerprint="e" * 64,
        records=_records(failing="sealed_holdout_veto"),
    )
    missing = build_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        design_artifact_fingerprint="d" * 64,
        production_artifact_fingerprint=None,
        records=_records(),
    )

    assert not failed.passed
    assert not failed.generation_authorized
    assert not missing.passed


def test_gate_evidence_fingerprint_is_content_sensitive() -> None:
    first = gate_record(
        G3_GATE_ORDER[0],
        passed=True,
        evidence={"value": 1.0},
        summary={"ok": True},
    )
    second = gate_record(
        G3_GATE_ORDER[0],
        passed=True,
        evidence={"value": 2.0},
        summary={"ok": True},
    )
    assert first.evidence_fingerprint != second.evidence_fingerprint


def test_wrong_gate_order_and_secret_or_nonfinite_evidence_fail() -> None:
    with pytest.raises(ModelError, match="exact closed gate order"):
        build_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            design_artifact_fingerprint="d" * 64,
            production_artifact_fingerprint="e" * 64,
            records=tuple(reversed(_records())),
        )
    with pytest.raises(ModelError, match="secret-like"):
        gate_record(
            G3_GATE_ORDER[0],
            passed=True,
            evidence={"api_token": "no"},
            summary={"ok": True},
        )
    with pytest.raises(ModelError, match="non-finite"):
        gate_record(
            G3_GATE_ORDER[0],
            passed=True,
            evidence={"value": float("nan")},
            summary={"ok": True},
        )


def test_bundle_verification_detects_tampering() -> None:
    bundle = build_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        design_artifact_fingerprint="d" * 64,
        production_artifact_fingerprint="e" * 64,
        records=_records(),
    )
    with pytest.raises(ModelError, match="fingerprint"):
        verify_g3_evidence_bundle(replace(bundle, fingerprint="0" * 64))


def test_prior_contract_cannot_be_verified_as_version_3_evidence() -> None:
    bundle = build_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        design_artifact_fingerprint="d" * 64,
        production_artifact_fingerprint="e" * 64,
        records=_records(),
    )

    assert G3_CONTRACT_VERSION == ("section-4.10-owner-approved-fixed-k4-20260715-v3")
    with pytest.raises(ModelError, match="contract version is unsupported"):
        verify_g3_evidence_bundle(
            replace(
                bundle,
                contract_version="sections-4.5-4.6-owner-approved-20260714-v1",
            )
        )


@pytest.mark.parametrize(
    "fingerprint",
    ["A" * 64, "a" * 63, "not-a-hash"],
)
def test_bundle_rejects_noncanonical_fingerprints(fingerprint: str) -> None:
    with pytest.raises(ModelError, match="lowercase SHA-256"):
        build_g3_evidence_bundle(
            config_fingerprint=fingerprint,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            design_artifact_fingerprint="d" * 64,
            production_artifact_fingerprint="e" * 64,
            records=_records(),
        )
