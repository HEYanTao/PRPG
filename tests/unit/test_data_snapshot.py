from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from prpg.data.providers import AcquisitionAttempt, ProviderPayload
from prpg.data.snapshot import MANIFEST_NAME, SnapshotStore
from prpg.errors import DataValidationError, IntegrityError


def _payload(
    *,
    provider: str = "fred",
    identifier: str = "INDPRO",
    content: bytes = b"DATE,INDPRO\n2026-01-01,100\n",
    retrieved_utc: str = "2026-07-14T12:00:00+00:00",
) -> ProviderPayload:
    frame = pd.DataFrame(
        {identifier: [100.0]}, index=pd.DatetimeIndex(["2026-01-01"], name="DATE")
    )
    attempt = AcquisitionAttempt(
        number=1,
        started_utc=retrieved_utc,
        finished_utc=retrieved_utc,
        outcome="success",
        http_status=200,
    )
    return ProviderPayload(
        provider=provider,
        identifier=identifier,
        frame=frame,
        raw_bytes=content,
        retrieved_utc=retrieved_utc,
        request={
            "start": "2007-01-01",
            "end": "2026-06-30",
            "retrieved_utc": retrieved_utc,
        },
        library_version="test-1",
        attempts=(attempt,),
        metadata={"current_vintage_retrospective": True},
    )


def _contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "cutoff": "2026-06-30",
        "macro_vintage_policy": "current_vintage_retrospective",
        "created_utc": "excluded-from-identity",
    }


def test_create_is_atomic_and_offline_readable(tmp_path: Path) -> None:
    store = SnapshotStore(
        tmp_path / "artifacts",
        now=lambda: datetime(2026, 7, 14, 13, 0, tzinfo=UTC),
    )
    payload = _payload()

    reference = store.create(
        [payload], acquisition_contract=_contract(), acquisition_group_id="group-001"
    )

    assert not reference.reused
    assert reference.path == (
        tmp_path / "artifacts" / "data" / "sha256" / reference.fingerprint
    )
    assert len(reference.fingerprint) == 64
    assert store.read_payload(reference.fingerprint, "fred", "INDPRO") == (
        payload.raw_bytes
    )
    entry = reference.manifest["files"][0]
    assert entry["provider"] == "fred"
    assert entry["identifier"] == "INDPRO"
    assert entry["request"]["start"] == "2007-01-01"
    assert entry["sha256"] == payload.sha256
    assert entry["bytes"] == len(payload.raw_bytes)
    assert entry["retrieved_utc"] == payload.retrieved_utc
    assert not tuple(reference.path.parent.glob(".stage-*"))


def test_identity_excludes_retrieval_and_manifest_timestamps(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    first = _payload(retrieved_utc="2026-07-14T12:00:00+00:00")
    second = _payload(retrieved_utc="2026-07-14T14:00:00+00:00")
    first_contract = _contract()
    second_contract = {**_contract(), "created_utc": "another-audit-time"}

    created = store.create(
        [first], acquisition_contract=first_contract, acquisition_group_id="first"
    )
    reused = store.create(
        [second], acquisition_contract=second_contract, acquisition_group_id="second"
    )

    assert reused.reused
    assert reused.fingerprint == created.fingerprint
    assert reused.path == created.path
    # Idempotence never mutates the original immutable audit manifest.
    assert reused.manifest["acquisition_group_id"] == "first"
    assert reused.manifest["files"][0]["retrieved_utc"] == first.retrieved_utc


def test_contract_or_payload_change_creates_a_new_fingerprint(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    baseline = store.create(
        [_payload()], acquisition_contract=_contract(), acquisition_group_id="one"
    )
    changed_payload = store.create(
        [_payload(content=b"DATE,INDPRO\n2026-01-01,101\n")],
        acquisition_contract=_contract(),
        acquisition_group_id="two",
    )
    changed_contract = store.create(
        [_payload()],
        acquisition_contract={**_contract(), "cutoff": "2026-05-31"},
        acquisition_group_id="three",
    )

    assert (
        len(
            {
                baseline.fingerprint,
                changed_payload.fingerprint,
                changed_contract.fingerprint,
            }
        )
        == 3
    )


def test_tamper_blocks_verification_and_idempotent_reuse(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    payload = _payload()
    created = store.create(
        [payload], acquisition_contract=_contract(), acquisition_group_id="original"
    )
    raw_path = created.path / created.manifest["files"][0]["relative_path"]
    raw_path.write_bytes(b"tampered")

    with pytest.raises(IntegrityError, match="integrity check failed"):
        store.verify(created.fingerprint)
    with pytest.raises(IntegrityError, match="integrity check failed"):
        store.create(
            [payload], acquisition_contract=_contract(), acquisition_group_id="retry"
        )
    assert raw_path.read_bytes() == b"tampered"


def test_manifest_path_traversal_is_rejected_before_read(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    created = store.create(
        [_payload()], acquisition_contract=_contract(), acquisition_group_id="safe"
    )
    manifest_path = created.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["relative_path"] = "../outside.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IntegrityError, match="relative path is unsafe"):
        store.verify(created.fingerprint)


def test_unmanifested_file_and_symlink_fail_closed(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    created = store.create(
        [_payload()], acquisition_contract=_contract(), acquisition_group_id="safe"
    )
    (created.path / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(IntegrityError, match="unmanifested"):
        store.verify(created.fingerprint)

    (created.path / "unexpected.txt").unlink()
    raw_path = created.path / created.manifest["files"][0]["relative_path"]
    raw_path.unlink()
    raw_path.symlink_to(tmp_path / "outside.csv")
    with pytest.raises(IntegrityError, match="unsafe|cannot be resolved"):
        store.verify(created.fingerprint)


def test_duplicate_and_unsafe_provider_identifiers_are_rejected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    payload = _payload()

    with pytest.raises(DataValidationError, match="duplicate"):
        store.create(
            [payload, payload],
            acquisition_contract=_contract(),
            acquisition_group_id="duplicate",
        )
    with pytest.raises(DataValidationError, match="safe path component"):
        store.create(
            [_payload(identifier="../INDPRO")],
            acquisition_contract=_contract(),
            acquisition_group_id="unsafe",
        )


@pytest.mark.parametrize("fingerprint", ["../escape", "A" * 64, "abc"])
def test_snapshot_lookup_rejects_noncanonical_fingerprint(
    tmp_path: Path, fingerprint: str
) -> None:
    with pytest.raises(IntegrityError, match="fingerprint is invalid"):
        SnapshotStore(tmp_path).verify(fingerprint)
