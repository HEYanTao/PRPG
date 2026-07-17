from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import prpg.data.snapshot as snapshot_module
from prpg.data.providers import AcquisitionAttempt, ProviderPayload
from prpg.data.snapshot import MANIFEST_NAME, SnapshotReference, SnapshotStore
from prpg.errors import DataValidationError, IntegrityError


class _ContractValue(Enum):
    RETROSPECTIVE = "current-vintage-retrospective"


class _Dumpable:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"from_model": True}


def _payload(
    *,
    provider: str = "fred",
    identifier: str = "INDPRO",
    content: bytes = b"DATE,INDPRO\n2026-01-01,100\n",
) -> ProviderPayload:
    retrieved = "2026-07-14T12:00:00+00:00"
    return ProviderPayload(
        provider=provider,
        identifier=identifier,
        frame=pd.DataFrame(
            {identifier: [100.0]},
            index=pd.DatetimeIndex(["2026-01-01"], name="DATE"),
        ),
        raw_bytes=content,
        retrieved_utc=retrieved,
        request={"start": "2007-01-01", "end": "2026-06-30"},
        library_version="test-1",
        attempts=(
            AcquisitionAttempt(
                number=1,
                started_utc=retrieved,
                finished_utc=retrieved,
                outcome="success",
                http_status=200,
            ),
        ),
    )


def _create(store: SnapshotStore) -> SnapshotReference:
    return store.create(
        [_payload()],
        acquisition_contract={"cutoff": "2026-06-30"},
        acquisition_group_id="negative-tests",
    )


def _read_manifest(reference: SnapshotReference) -> dict[str, Any]:
    value = json.loads((reference.path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(reference: SnapshotReference, manifest: object) -> None:
    (reference.path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_missing_snapshot_directory_and_manifest_fail_closed(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    fingerprint = "0" * 64

    with pytest.raises(IntegrityError, match="directory is missing"):
        store.verify(fingerprint)

    snapshot_path = store.store_root / fingerprint
    snapshot_path.mkdir(parents=True)
    with pytest.raises(IntegrityError, match="manifest is missing"):
        store.verify(fingerprint)


def test_snapshot_and_manifest_symlinks_are_rejected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.store_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    fingerprint = "1" * 64
    (store.store_root / fingerprint).symlink_to(outside, target_is_directory=True)

    with pytest.raises(IntegrityError, match="directory is missing or unsafe"):
        store.verify(fingerprint)

    (store.store_root / fingerprint).unlink()
    snapshot_path = store.store_root / fingerprint
    snapshot_path.mkdir()
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_text("{}", encoding="utf-8")
    (snapshot_path / MANIFEST_NAME).symlink_to(outside_manifest)
    with pytest.raises(IntegrityError, match="manifest is missing or unsafe"):
        store.verify(fingerprint)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("root", "root is not an object"),
        ("schema", "schema is unsupported"),
        ("fingerprint", "fingerprint does not match"),
        ("identity_type", "identity is missing"),
        ("identity_hash", "identity hash mismatch"),
        ("files_empty", "file list is empty"),
        ("file_record", "file record is invalid"),
        ("path_type", "invalid or duplicate relative path"),
        ("path_duplicate", "invalid or duplicate relative path"),
    ],
)
def test_corrupt_manifest_structures_are_rejected(
    tmp_path: Path, case: str, message: str
) -> None:
    store = SnapshotStore(tmp_path)
    reference = _create(store)
    manifest: object = _read_manifest(reference)
    assert isinstance(manifest, dict)

    if case == "root":
        manifest = []
    elif case == "schema":
        manifest["schema_version"] = 999
    elif case == "fingerprint":
        manifest["snapshot_fingerprint"] = "f" * 64
    elif case == "identity_type":
        manifest["identity"] = None
    elif case == "identity_hash":
        manifest["identity"]["acquisition_contract"]["cutoff"] = "2026-05-31"
    elif case == "files_empty":
        manifest["files"] = []
    elif case == "file_record":
        manifest["files"] = [42]
    elif case == "path_type":
        manifest["files"][0]["relative_path"] = 42
    elif case == "path_duplicate":
        manifest["files"].append(dict(manifest["files"][0]))
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)
    _write_manifest(reference, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


def test_malformed_manifest_json_is_an_integrity_failure(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    reference = _create(store)
    (reference.path / MANIFEST_NAME).write_text("{broken", encoding="utf-8")

    with pytest.raises(IntegrityError, match="manifest cannot be read") as captured:
        store.verify(reference.fingerprint)
    assert captured.value.details["error_type"] == "JSONDecodeError"


@pytest.mark.parametrize("relative_path", ["/absolute.csv", "raw\\escape.csv", ""])
def test_manifest_unsafe_path_forms_fail_before_payload_read(
    tmp_path: Path, relative_path: str
) -> None:
    store = SnapshotStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    manifest["files"][0]["relative_path"] = relative_path
    _write_manifest(reference, manifest)

    with pytest.raises(IntegrityError, match="relative path is unsafe"):
        store.verify(reference.fingerprint)


def test_payload_path_that_is_a_directory_is_not_accepted(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    reference = _create(store)
    payload_path = reference.path / reference.manifest["files"][0]["relative_path"]
    payload_path.unlink()
    payload_path.mkdir()

    with pytest.raises(IntegrityError, match="payload is missing or unsafe"):
        store.verify(reference.fingerprint)


def test_payload_symlink_escape_and_internal_link_are_rejected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    original_path = reference.path / manifest["files"][0]["relative_path"]
    original_path.unlink()

    outside = tmp_path / "outside.csv"
    outside.write_bytes(_payload().raw_bytes)
    original_path.symlink_to(outside)
    with pytest.raises(IntegrityError, match="escapes its root"):
        store.verify(reference.fingerprint)

    original_path.unlink()
    internal = reference.path / "internal.csv"
    internal.write_bytes(_payload().raw_bytes)
    original_path.symlink_to(internal)
    with pytest.raises(IntegrityError, match="symbolic link"):
        store.verify(reference.fingerprint)


def test_missing_payload_lookup_is_a_validation_error(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    reference = _create(store)

    with pytest.raises(DataValidationError, match="lookup is not unique") as captured:
        store.read_payload(reference.fingerprint, "fred", "NOT_A_SERIES")
    assert captured.value.details["matches"] == 0


def test_empty_payload_collection_and_bytes_are_rejected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(DataValidationError, match="at least one payload"):
        store.create([], acquisition_contract={}, acquisition_group_id="no-payloads")
    with pytest.raises(DataValidationError, match="payload is empty"):
        store.create(
            [_payload(content=b"")],
            acquisition_contract={},
            acquisition_group_id="empty-payload",
        )


def test_identity_normalizes_supported_contract_value_types(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    contract: Any = _Dumpable()
    # Exercise the dumpable branch through public creation first.
    dumped = store.create(
        [_payload()], acquisition_contract=contract, acquisition_group_id="dumpable"
    )
    assert dumped.manifest["identity"]["acquisition_contract"] == {"from_model": True}

    rich_contract = {
        "items": ("a", "b"),
        "unordered": {2, 1},
        "policy": _ContractValue.RETROSPECTIVE,
        "as_of": datetime(2026, 6, 30, 12, tzinfo=UTC),
        "day": date(2026, 6, 30),
        "path": Path("relative/data"),
        "optional": None,
        "enabled": True,
        "coverage": 0.8,
    }
    reference = store.create(
        [_payload()],
        acquisition_contract=rich_contract,
        acquisition_group_id="rich-contract",
    )
    normalized = reference.manifest["identity"]["acquisition_contract"]
    assert normalized["unordered"] == [1, 2]
    assert normalized["policy"] == "current-vintage-retrospective"
    assert normalized["path"] == "relative/data"


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_identity_values_fail_before_publication(
    tmp_path: Path, value: float
) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(DataValidationError, match="non-finite"):
        store.create(
            [_payload()],
            acquisition_contract={"invalid": value},
            acquisition_group_id="nonfinite",
        )
    assert not store.store_root.exists()


def test_unsupported_identity_value_fails_before_publication(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(DataValidationError, match="unsupported value") as captured:
        store.create(
            [_payload()],
            acquisition_contract={"invalid": object()},
            acquisition_group_id="unsupported",
        )
    assert captured.value.details["value_type"] == "object"


def test_audit_metadata_is_json_normalized_but_not_part_of_identity(
    tmp_path: Path,
) -> None:
    store = SnapshotStore(tmp_path)
    audit = {
        "dumpable": _Dumpable(),
        "sequence": (1, 2),
        "set": {1},
        "policy": _ContractValue.RETROSPECTIVE,
        "when": datetime(2026, 7, 14, tzinfo=UTC),
        "day": date(2026, 7, 14),
        "path": Path("local/raw"),
        "none": None,
        "ratio": 0.8,
        "display_only": object(),
    }
    first = store.create(
        [_payload()],
        acquisition_contract={"cutoff": "2026-06-30"},
        acquisition_group_id="audit-one",
        audit_metadata=audit,
    )
    second = store.create(
        [_payload()],
        acquisition_contract={"cutoff": "2026-06-30"},
        acquisition_group_id="audit-two",
        audit_metadata={"different": "ignored for identity"},
    )

    assert second.fingerprint == first.fingerprint
    assert second.reused
    assert first.manifest["audit_metadata"]["dumpable"] == {"from_model": True}
    assert first.manifest["audit_metadata"]["path"] == "local/raw"


def test_nonfinite_audit_metadata_cleans_staging_directory(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(DataValidationError, match="manifest contains a non-finite"):
        store.create(
            [_payload()],
            acquisition_contract={},
            acquisition_group_id="bad-audit",
            audit_metadata={"bad": float("nan")},
        )
    assert not tuple(store.store_root.glob(".stage-*"))


@pytest.mark.parametrize("generic", [False, True])
def test_content_addressed_publish_handles_a_winning_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generic: bool
) -> None:
    store = SnapshotStore(tmp_path)
    real_rename = os.rename

    def concurrent_winner(source: str | Path, destination: str | Path) -> None:
        real_rename(source, destination)
        if generic:
            raise OSError("simulated EEXIST variant")
        raise FileExistsError("simulated race")

    monkeypatch.setattr(snapshot_module.os, "rename", concurrent_winner)
    reference = _create(store)

    assert reference.reused
    assert store.verify(reference.fingerprint).fingerprint == reference.fingerprint
    assert not tuple(store.store_root.glob(".stage-*"))


def test_publish_oserror_without_winner_preserves_no_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)

    def fail_rename(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("simulated disk error")

    monkeypatch.setattr(snapshot_module.os, "rename", fail_rename)
    with pytest.raises(OSError, match="disk error"):
        _create(store)
    assert not tuple(store.store_root.glob(".stage-*"))
    assert not tuple(path for path in store.store_root.iterdir() if path.is_dir())


def test_write_failure_removes_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(snapshot_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        _create(store)
    assert not tuple(store.store_root.glob(".stage-*"))


def test_verified_reuse_rejects_identity_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)
    _create(store)
    original_verify = store.verify

    def mismatched_verify(
        fingerprint: str, *, reused: bool = True
    ) -> SnapshotReference:
        verified = original_verify(fingerprint, reused=reused)
        manifest = dict(verified.manifest)
        manifest["identity"] = {"different": True}
        return SnapshotReference(
            fingerprint=verified.fingerprint,
            path=verified.path,
            manifest=manifest,
            reused=verified.reused,
        )

    monkeypatch.setattr(store, "verify", mismatched_verify)
    with pytest.raises(IntegrityError, match="fingerprint collision"):
        _create(store)


def test_canonical_json_encoding_failure_is_a_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path)

    def fail_json(*_args: object, **_kwargs: object) -> str:
        raise ValueError("simulated encoder failure")

    monkeypatch.setattr(snapshot_module.json, "dumps", fail_json)
    with pytest.raises(DataValidationError, match="encoded canonically"):
        _create(store)
