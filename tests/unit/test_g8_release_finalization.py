from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import prpg.validation.release as release_module
from prpg.data.calendar import CALENDAR_ID
from prpg.errors import GenerationError, IntegrityError, ReleaseError
from prpg.storage.csv_v1 import CSV_SCHEMA_VERSION, CSV_SERIALIZER_VERSION
from prpg.validation.release import (
    ReleaseEvidence,
    ReleaseFinalizationPlan,
    build_external_release_attestation,
    finalize_complete,
    publish_released,
    verify_complete,
    verify_released,
)
from prpg.validation.structural import (
    CONSUMER_MANIFEST_SCHEMA_ID,
    CONSUMER_MANIFEST_SCHEMA_VERSION,
    DATA_VALIDATED_SCHEMA_ID,
    DATA_VALIDATED_SCHEMA_VERSION,
    STRUCTURAL_REPORT_SCHEMA_ID,
    STRUCTURAL_REPORT_SCHEMA_VERSION,
    StructuralValidationPlan,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_SMALL_PLAN = StructuralValidationPlan(
    profile_id="g8-test-1y",
    years=1,
    lf_paths=1,
    hf_paths=1,
    lf_paths_per_shard=1,
    hf_paths_per_shard=1,
)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _stage_g7_root(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> ReleaseFinalizationPlan:
    monkeypatch.setattr(release_module, "CANONICAL_STRUCTURAL_PLAN", _SMALL_PLAN)
    root.mkdir()
    records: list[dict[str, Any]] = []
    for family, directory, frequency, periods in (
        ("LF", "low", "monthly", 12),
        ("LF", "low", "quarterly", 4),
        ("LF", "low", "annual", 1),
        ("HF", "high", "daily", 252),
        ("HF", "high", "weekly", 52),
    ):
        path_id = f"{family}000001"
        relative = (
            f"consumer/returns/{directory}/{frequency}/"
            f"part-00000-path-{path_id}-{path_id}.csv"
        )
        content = f"fixture-{family}-{frequency}\n".encode()
        _write(root / relative, content)
        records.append(
            {
                "work_unit_id": f"{family.lower()}-00000-path-000001-000001",
                "work_unit_receipt_fingerprint": "4" * 64,
                "family": family,
                "frequency": frequency,
                "shard_index": 0,
                "first_path_entity": 1,
                "last_path_entity": 1,
                "first_path_id": path_id,
                "last_path_id": path_id,
                "path_count": 1,
                "years": 1,
                "periods_per_path": periods,
                "rows": periods,
                "relative_path": relative,
                "bytes": len(content),
                "sha256": _sha(content),
                "csv_receipt_fingerprint": "5" * 64,
            }
        )
    manifest = {
        "schema_id": CONSUMER_MANIFEST_SCHEMA_ID,
        "schema_version": CONSUMER_MANIFEST_SCHEMA_VERSION,
        "calendar_id": CALENDAR_ID,
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "csv_serializer_version": CSV_SERIALIZER_VERSION,
        "plan_fingerprint": _SMALL_PLAN.fingerprint,
        "storage_binding": {
            "simulation_fingerprint": "0" * 64,
            "materialization_fingerprint": "1" * 64,
            "toolchain_fingerprint": "2" * 64,
            "run_fingerprint": "3" * 64,
        },
        "datasets": release_module._expected_datasets(_SMALL_PLAN),
        "work_unit_count": _SMALL_PLAN.work_unit_count,
        "csv_shard_count": _SMALL_PLAN.csv_shard_count,
        "total_rows": _SMALL_PLAN.total_rows,
        "total_values": _SMALL_PLAN.total_values,
        "shards": records,
    }
    manifest_content = _canonical(manifest)
    _write(root / "consumer" / "consumer-manifest.json", manifest_content)

    report_identity = {
        "schema_id": STRUCTURAL_REPORT_SCHEMA_ID,
        "schema_version": STRUCTURAL_REPORT_SCHEMA_VERSION,
        "plan_fingerprint": _SMALL_PLAN.fingerprint,
        "canonical_profile": True,
        "manifest_sha256": _sha(manifest_content),
        "total_rows": _SMALL_PLAN.total_rows,
        "total_values": _SMALL_PLAN.total_values,
        "work_units": _SMALL_PLAN.work_unit_count,
        "csv_shards": _SMALL_PLAN.csv_shard_count,
        "aggregation_values_checked": 348,
        "roundtrip_values_checked": _SMALL_PLAN.total_values,
        "max_abs_log_aggregation_error": 0.0,
        "max_abs_simple_compounding_error": 0.0,
        "max_abs_log_simple_roundtrip_error": 0.0,
        "elapsed_seconds": 1.0,
        "rows_per_second": float(_SMALL_PLAN.total_rows),
        "datasets": release_module._expected_datasets(_SMALL_PLAN),
        "shards": [
            {
                "relative_path": record["relative_path"],
                "rows": record["rows"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for record in records
        ],
        "hook_metrics": {},
    }
    report_fingerprint = _sha(_canonical(report_identity))
    report_content = _canonical(
        {
            "schema_version": STRUCTURAL_REPORT_SCHEMA_VERSION,
            "report_fingerprint": report_fingerprint,
            "identity": report_identity,
        }
    )
    _write(root / "validation" / "structural-report.json", report_content)
    data_validated = _canonical(
        {
            "schema_id": DATA_VALIDATED_SCHEMA_ID,
            "schema_version": DATA_VALIDATED_SCHEMA_VERSION,
            "scientific_pass_fingerprint": "e" * 64,
            "structural_report_fingerprint": report_fingerprint,
            "structural_report_sha256": _sha(report_content),
        }
    )
    _write(root / "DATA_VALIDATED", data_validated)
    _write(root / "run-state.json", _canonical({"state": "FINALIZATION_PENDING"}))
    clean_replay = _canonical({"gate": "clean-replay", "passed": True})
    release_authorization = _canonical(
        {"gate": "release-authorization", "passed": True}
    )
    _write(root / "validation" / "clean-replay.json", clean_replay)
    _write(
        root / "validation" / "release-authorization.json",
        release_authorization,
    )
    evidence = (
        ReleaseEvidence(
            evidence_id="EV-014",
            purpose="clean_replay",
            relative_path="validation/clean-replay.json",
            sha256=_sha(clean_replay),
        ),
        ReleaseEvidence(
            evidence_id="EV-G8-AUTH",
            purpose="release_authorization",
            relative_path="validation/release-authorization.json",
            sha256=_sha(release_authorization),
        ),
    )
    return ReleaseFinalizationPlan(
        run_id="canonical-test-run",
        generation_commit="1" * 40,
        release_commit="2" * 40,
        release_tag="v0.1.0-test",
        finalizer_identity="finalizer-a",
        immutable_audit_files=tuple(
            sorted(
                {
                    "run-state.json",
                    "validation/clean-replay.json",
                    "validation/release-authorization.json",
                    "validation/structural-report.json",
                }
            )
        ),
        evidence=evidence,
    )


def test_complete_and_released_happy_path_are_one_way_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)

    complete = finalize_complete(run_root, plan=plan)

    assert complete.run_id == plan.run_id
    assert complete.generation_commit == plan.generation_commit
    assert complete.release_commit == plan.release_commit
    assert complete.covered_file_count == 13
    assert verify_complete(run_root) == complete
    assert not (run_root / "RELEASED").exists()
    root_checksum_before = (run_root / "SHA256SUMS").read_bytes()

    consumer_lines = (run_root / "consumer" / "SHA256SUMS").read_text().splitlines()
    assert consumer_lines == sorted(consumer_lines, key=lambda line: line[66:])
    assert all("DATA_VALIDATED" not in line for line in consumer_lines)
    root_lines = (run_root / "SHA256SUMS").read_text().splitlines()
    assert root_lines == sorted(root_lines, key=lambda line: line[66:])
    assert any(line.endswith("  consumer/SHA256SUMS") for line in root_lines)
    assert not any(line.endswith("  SHA256SUMS") for line in root_lines)
    assert not any("DATA_VALIDATED" in line for line in root_lines)
    assert not any("COMPLETE" in line for line in root_lines)

    attestation = build_external_release_attestation(
        run_root, verifier_identity="independent-verifier"
    )
    attestation_path = tmp_path / "attestations" / f"{plan.run_id}-g8.json"
    _write(attestation_path, attestation)
    released = publish_released(run_root, attestation_path=attestation_path)

    assert released.complete == complete
    assert released.verifier_identity == "independent-verifier"
    assert released.attestation_uri == attestation_path.as_uri()
    assert verify_released(run_root) == released
    assert verify_released(run_root, attestation_path=attestation_path) == released
    assert verify_complete(run_root) == complete
    assert (run_root / "SHA256SUMS").read_bytes() == root_checksum_before

    with pytest.raises(ReleaseError):
        finalize_complete(run_root, plan=plan)
    with pytest.raises(ReleaseError):
        publish_released(run_root, attestation_path=attestation_path)


def test_finalization_is_location_independent_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    first_plan = _stage_g7_root(first, monkeypatch)
    finalize_complete(first, plan=first_plan)
    first_root = (first / "SHA256SUMS").read_bytes()
    first_complete = (first / "COMPLETE").read_bytes()

    second = tmp_path / "second"
    second_plan = _stage_g7_root(second, monkeypatch)
    finalize_complete(second, plan=second_plan)

    assert second_plan.fingerprint == first_plan.fingerprint
    assert (second / "consumer" / "SHA256SUMS").read_bytes() == (
        first / "consumer" / "SHA256SUMS"
    ).read_bytes()
    assert (second / "SHA256SUMS").read_bytes() == first_root
    assert (second / "COMPLETE").read_bytes() == first_complete


@pytest.mark.parametrize("unexpected", [".DS_Store", "validation/debug.tmp"])
def test_strict_allowlist_rejects_unexpected_files_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unexpected: str
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    _write(run_root / unexpected, b"unexpected")

    with pytest.raises(IntegrityError, match="allowlist"):
        finalize_complete(run_root, plan=plan)

    assert not (run_root / "consumer" / "SHA256SUMS").exists()
    assert not (run_root / "run-manifest.json").exists()
    assert not (run_root / "COMPLETE").exists()


def test_symlink_and_invalid_lifecycle_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    os.symlink(run_root / "run-state.json", run_root / "validation" / "alias.json")
    with pytest.raises(IntegrityError, match="unsafe file"):
        finalize_complete(run_root, plan=plan)

    (run_root / "validation" / "alias.json").unlink()
    _write(run_root / "RUNNING", b"active\n")
    with pytest.raises(ReleaseError, match="lifecycle"):
        finalize_complete(run_root, plan=plan)
    assert not (run_root / "COMPLETE").exists()


def test_g7_binding_or_covered_file_tamper_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    finalize_complete(run_root, plan=plan)

    state = run_root / "run-state.json"
    state.write_bytes(state.read_bytes() + b"tamper")
    with pytest.raises(IntegrityError, match="hash differs"):
        verify_complete(run_root)

    second = tmp_path / "second"
    second_plan = _stage_g7_root(second, monkeypatch)
    marker = json.loads((second / "DATA_VALIDATED").read_bytes())
    marker["structural_report_sha256"] = "f" * 64
    (second / "DATA_VALIDATED").write_bytes(_canonical(marker))
    with pytest.raises(IntegrityError, match="structural report differs"):
        finalize_complete(second, plan=second_plan)
    assert not (second / "COMPLETE").exists()


def test_self_consistent_but_impossible_g7_report_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    report_path = run_root / "validation" / "structural-report.json"
    report = json.loads(report_path.read_bytes())
    del report["identity"]["total_rows"]
    report["report_fingerprint"] = _sha(_canonical(report["identity"]))
    report_content = _canonical(report)
    report_path.write_bytes(report_content)
    marker_path = run_root / "DATA_VALIDATED"
    marker = json.loads(marker_path.read_bytes())
    marker["structural_report_fingerprint"] = report["report_fingerprint"]
    marker["structural_report_sha256"] = _sha(report_content)
    marker_path.write_bytes(_canonical(marker))

    with pytest.raises(IntegrityError, match="identity keys"):
        finalize_complete(run_root, plan=plan)

    assert not (run_root / "COMPLETE").exists()


def test_verifier_rejects_lifecycle_marker_inside_root_sha256sums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    finalize_complete(run_root, plan=plan)

    manifest_path = run_root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["covered_files"] = sorted([*manifest["covered_files"], "DATA_VALIDATED"])
    manifest_content = _canonical(manifest)
    manifest_path.write_bytes(manifest_content)

    entries = {
        line[66:]: line[:64]
        for line in (run_root / "SHA256SUMS").read_text().splitlines()
    }
    entries["run-manifest.json"] = _sha(manifest_content)
    entries["DATA_VALIDATED"] = _sha((run_root / "DATA_VALIDATED").read_bytes())
    checksum_content = "".join(
        f"{entries[path]}  {path}\n" for path in sorted(entries)
    ).encode()
    (run_root / "SHA256SUMS").write_bytes(checksum_content)
    complete_path = run_root / "COMPLETE"
    complete = json.loads(complete_path.read_bytes())
    complete["root_sha256sums_sha256"] = _sha(checksum_content)
    complete["run_manifest_sha256"] = _sha(manifest_content)
    complete_path.write_bytes(_canonical(complete))

    with pytest.raises(IntegrityError, match="excluded lifecycle"):
        verify_complete(run_root)


def test_attestation_requires_external_path_and_independent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    finalize_complete(run_root, plan=plan)

    with pytest.raises(ReleaseError, match="independent"):
        build_external_release_attestation(
            run_root, verifier_identity=plan.finalizer_identity
        )

    inside = run_root / "validation" / "clean-replay.json"
    with pytest.raises(ReleaseError, match="outside"):
        publish_released(run_root, attestation_path=inside)
    assert not (run_root / "RELEASED").exists()


def test_attestation_and_released_binding_detect_mismatch_or_later_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan = _stage_g7_root(run_root, monkeypatch)
    finalize_complete(run_root, plan=plan)
    attestation_path = tmp_path / "attestation.json"
    content = build_external_release_attestation(
        run_root, verifier_identity="independent-verifier"
    )
    document = json.loads(content)
    document["release_tag"] = "wrong-tag"
    _write(attestation_path, _canonical(document))
    with pytest.raises(IntegrityError, match="differs from COMPLETE"):
        publish_released(run_root, attestation_path=attestation_path)
    assert not (run_root / "RELEASED").exists()

    attestation_path.write_bytes(content)
    publish_released(run_root, attestation_path=attestation_path)
    attestation_path.write_bytes(content + b"tamper")
    with pytest.raises(IntegrityError):
        verify_released(run_root, attestation_path=attestation_path)


def test_plan_rejects_unsafe_or_incomplete_prerequisite_contract() -> None:
    clean = ReleaseEvidence(
        evidence_id="EV-014",
        purpose="clean_replay",
        relative_path="validation/clean.json",
        sha256="a" * 64,
    )
    authorization = ReleaseEvidence(
        evidence_id="EV-015",
        purpose="release_authorization",
        relative_path="validation/authorization.json",
        sha256="b" * 64,
    )
    with pytest.raises(GenerationError, match="required frozen files"):
        ReleaseFinalizationPlan(
            run_id="run",
            generation_commit="1" * 40,
            release_commit="2" * 40,
            release_tag="v1",
            finalizer_identity="finalizer",
            immutable_audit_files=(
                "validation/authorization.json",
                "validation/clean.json",
            ),
            evidence=(clean, authorization),
        )
    with pytest.raises(GenerationError, match="unsafe"):
        ReleaseEvidence(
            evidence_id="EV-X",
            purpose="additional",
            relative_path="../escape.json",
            sha256="c" * 64,
        )
