from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prpg.model.artifact as artifact_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import (
    MODEL_MANIFEST_NAME,
    MODEL_PARAMETERS_NAME,
    ModelArtifactReference,
    ModelArtifactStore,
)


class _Policy(Enum):
    BASELINE = "historical_vector"


class _Dumpable:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"candidate": "k3", "passed": True}


class _BadDumpable:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        raise RuntimeError(f"cannot dump in {mode}")


class _BadArray:
    def __array__(self) -> np.ndarray[Any, Any]:
        raise ValueError("cannot convert")


def _kwargs() -> dict[str, Any]:
    return {
        "artifact_role": "design",
        "data_fingerprint": "a" * 64,
        "model_metadata": {
            "candidate": "k3",
            "feature_names": ("ip", "inflation", "curve", "credit"),
            "policy": _Policy.BASELINE,
            "selected": np.bool_(True),
            "parameter_count": np.int64(38),
        },
        "arrays": {
            "decoded_states": np.array([0, 1, 1, 0], dtype=np.int64),
            "transition_matrix": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64),
        },
        "calibration_documents": {
            "block-calibration.json": {
                "policy": "stationary-frequency-wide-v1",
                "passed": True,
                "lengths": {"monthly": 3, "daily": 11},
            },
            "evidence/duration-report.json": {
                "status": "passed",
                "replicates": 17,
            },
        },
        "provenance": {
            "producer": "prpg model calibrate",
            "source_code_hash": "b" * 64,
            "resolved_config_fingerprint": "c" * 64,
            "produced_for": date(2026, 7, 14),
        },
    }


def _create(store: ModelArtifactStore, **updates: Any) -> ModelArtifactReference:
    values = _kwargs()
    values.update(updates)
    return store.create(**values)


def _read_manifest(reference: ModelArtifactReference) -> dict[str, Any]:
    value = json.loads(
        (reference.path / MODEL_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, value: object, *, canonical: bool = True) -> None:
    if canonical:
        content = artifact_module._canonical_json_bytes(value)
    else:
        content = json.dumps(value, indent=2).encode("utf-8")
    path.write_bytes(content)


def _readdress(
    store: ModelArtifactStore,
    reference: ModelArtifactReference,
    identity: dict[str, Any],
) -> tuple[str, Path]:
    manifest = _read_manifest(reference)
    fingerprint = hashlib.sha256(
        artifact_module._canonical_json_bytes(identity)
    ).hexdigest()
    manifest["identity"] = identity
    manifest["model_artifact_fingerprint"] = fingerprint
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)
    destination = store.store_root / fingerprint
    reference.path.rename(destination)
    return fingerprint, destination


def _replace_parameter_archive(
    store: ModelArtifactStore,
    reference: ModelArtifactReference,
    content: bytes,
    *,
    identity_mutator: Any | None = None,
) -> str:
    identity = dict(reference.manifest["identity"])
    parameters = dict(identity["parameters"])
    parameters["bytes"] = len(content)
    parameters["sha256"] = hashlib.sha256(content).hexdigest()
    if identity_mutator is not None:
        identity_mutator(parameters)
    identity["parameters"] = parameters
    (reference.path / MODEL_PARAMETERS_NAME).write_bytes(content)
    fingerprint, _ = _readdress(store, reference, identity)
    return fingerprint


def _archive_members(content: bytes) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _zip_bytes(
    members: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
    timestamp: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0),
    archive_comment: bytes = b"",
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        archive.comment = archive_comment
        for name, content in members:
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = compression
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def test_model_artifact_round_trip_is_canonical_and_pickle_free(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)

    reference = _create(store)
    arrays = store.load_arrays(reference.fingerprint)
    block = store.read_document(reference.fingerprint, "block-calibration.json")

    assert not reference.reused
    assert reference.path == tmp_path / "model" / "sha256" / reference.fingerprint
    assert {
        path.relative_to(reference.path).as_posix()
        for path in reference.path.rglob("*")
        if path.is_file()
    } == {
        MODEL_MANIFEST_NAME,
        MODEL_PARAMETERS_NAME,
        "block-calibration.json",
        "evidence/duration-report.json",
    }
    manifest = reference.manifest
    assert set(manifest) == {
        "schema_version",
        "model_artifact_fingerprint",
        "identity",
    }
    assert manifest["identity"]["artifact_role"] == "design"
    assert manifest["identity"]["data_fingerprint"] == "a" * 64
    assert manifest["identity"]["model_metadata"]["policy"] == "historical_vector"
    assert manifest["identity"]["provenance"]["produced_for"] == "2026-07-14"
    assert manifest["identity"]["parameters"]["allow_pickle"] is False
    assert (
        hashlib.sha256(
            artifact_module._canonical_json_bytes(manifest["identity"])
        ).hexdigest()
        == reference.fingerprint
    )
    assert (reference.path / MODEL_MANIFEST_NAME).read_bytes() == (
        artifact_module._canonical_json_bytes(manifest)
    )
    assert list(arrays) == ["decoded_states", "transition_matrix"]
    np.testing.assert_array_equal(arrays["decoded_states"], [0, 1, 1, 0])
    np.testing.assert_allclose(arrays["transition_matrix"], [[0.9, 0.1], [0.2, 0.8]])
    assert not arrays["decoded_states"].flags.writeable
    assert not arrays["transition_matrix"].flags.writeable
    assert block["lengths"] == {"daily": 11, "monthly": 3}
    assert (
        stat.S_IMODE((reference.path / MODEL_PARAMETERS_NAME).stat().st_mode) == 0o600
    )

    with zipfile.ZipFile(reference.path / MODEL_PARAMETERS_NAME) as archive:
        assert archive.namelist() == ["decoded_states.npy", "transition_matrix.npy"]
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )


def test_manifest_is_observation_free_and_array_layout_independent(
    tmp_path: Path,
) -> None:
    first = ModelArtifactStore(tmp_path)
    reference = _create(first)
    manifest_bytes = (reference.path / MODEL_MANIFEST_NAME).read_bytes()
    values = _kwargs()
    arrays = values["arrays"]
    arrays["decoded_states"] = np.array([0, 1, 1, 0], dtype=">i8")
    arrays["transition_matrix"] = np.asfortranarray(
        np.array([[0.9, 0.1], [0.2, 0.8]], dtype=">f8")
    )
    second = ModelArtifactStore(tmp_path)
    replay = second.create(**values)

    assert replay.reused
    assert replay.fingerprint == reference.fingerprint
    assert "created_utc" not in replay.manifest
    assert (replay.path / MODEL_MANIFEST_NAME).read_bytes() == manifest_bytes


def test_returned_artifact_values_are_recursively_and_irreversibly_immutable(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    arrays = store.load_arrays(reference.fingerprint)
    document = store.read_document(reference.fingerprint, "block-calibration.json")

    with pytest.raises(TypeError):
        reference.manifest["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        reference.manifest["identity"]["model_metadata"]["candidate"] = "k4"
    assert isinstance(reference.manifest["identity"]["documents"], tuple)
    with pytest.raises(TypeError):
        arrays["new"] = np.array([1])  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        arrays["decoded_states"][0] = 9
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        arrays["decoded_states"].setflags(write=True)
    with pytest.raises(TypeError):
        document["lengths"]["monthly"] = 99


def test_production_role_and_integrity_verification_never_authorize_g3(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store, artifact_role="production")
    verified = store.verify(reference.fingerprint)

    assert verified.manifest["identity"]["artifact_role"] == "production"
    assert verified.verification_scope == "integrity_only"
    assert verified.generation_authorized is False
    assert "generation_authorized" not in verified.manifest["identity"]
    assert "never authorizes G3" in (ModelArtifactStore.verify.__doc__ or "")


@pytest.mark.parametrize(
    "update",
    [
        {"artifact_role": "production"},
        {"data_fingerprint": "d" * 64},
        {"model_metadata": {"candidate": "k4"}},
        {"provenance": {"producer": "different"}},
        {"arrays": {"state": np.array([1, 2], dtype=np.int64)}},
        {"calibration_documents": {"block-calibration.json": {"length": 5}}},
    ],
)
def test_every_scientific_component_changes_the_identity(
    tmp_path: Path, update: dict[str, Any]
) -> None:
    store = ModelArtifactStore(tmp_path)
    baseline = _create(store)
    changed = _create(store, **update)

    assert changed.fingerprint != baseline.fingerprint


@pytest.mark.parametrize("role", ["", "candidate", "Production", 1, None])
def test_invalid_artifact_role_is_rejected(tmp_path: Path, role: object) -> None:
    with pytest.raises(ModelError, match="role is invalid"):
        _create(ModelArtifactStore(tmp_path), artifact_role=role)


@pytest.mark.parametrize("fingerprint", ["", "a" * 63, "A" * 64, "g" * 64, 4])
def test_invalid_data_fingerprint_is_rejected(
    tmp_path: Path, fingerprint: object
) -> None:
    with pytest.raises(ModelError, match="data fingerprint is invalid"):
        _create(ModelArtifactStore(tmp_path), data_fingerprint=fingerprint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_metadata", {}, "non-empty JSON object"),
        ("provenance", {}, "non-empty JSON object"),
        ("model_metadata", {1: "bad"}, "non-string object key"),
        ("model_metadata", {"x": object()}, "unsupported value"),
        ("model_metadata", {"x": _BadDumpable()}, "cannot be converted to JSON"),
        ("model_metadata", {"x": float("nan")}, "non-finite number"),
        ("model_metadata", {"api_key": "value"}, "secret-like key"),
        ("provenance", {"nested": [{"password": "value"}]}, "secret-like key"),
    ],
)
def test_invalid_identity_json_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        _create(ModelArtifactStore(tmp_path), **{field: value})


@pytest.mark.parametrize(
    "key",
    [
        "API-Key",
        "api key",
        "api.key",
        "access.token",
        "pass-word",
        "auth-orization",
        "client credential",
    ],
)
def test_secret_key_markers_are_separator_and_case_insensitive(
    tmp_path: Path, key: str
) -> None:
    with pytest.raises(ModelError, match="secret-like key or value"):
        _create(ModelArtifactStore(tmp_path), model_metadata={key: "redacted"})


@pytest.mark.parametrize(
    "value",
    [
        "AKIA1234567890ABCDEF",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "eyJabcdefghijk.abcdefghijk.abcdefghijk",
        "-----BEGIN PRIVATE KEY-----\nnot-real\n-----END PRIVATE KEY-----",
        "Bearer abcdefghijklmnopqrstuvwxyz123456",
        "password=abcdefghijklmnopqrstuvwxyz",
        "https://audit-user:long-password@example.test/path",
    ],
)
def test_high_confidence_secret_values_are_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ModelError, match="secret-like key or value"):
        _create(
            ModelArtifactStore(tmp_path),
            model_metadata={"audit_note": value},
        )


def test_benign_hashes_and_secret_related_documentation_are_not_false_positives(
    tmp_path: Path,
) -> None:
    reference = _create(
        ModelArtifactStore(tmp_path),
        model_metadata={
            "source_code_hash": "d" * 64,
            "description": "tokenized observations contain no credentials",
            "command_help": "configure --api-key-file outside the artifact",
        },
    )

    assert reference.manifest["identity"]["model_metadata"]["source_code_hash"] == (
        "d" * 64
    )


def test_json_normalization_accepts_dumpable_paths_dates_and_scalars(
    tmp_path: Path,
) -> None:
    metadata = {
        "model": _Dumpable(),
        "path": Path("relative/evidence"),
        "at": datetime(2026, 7, 14, 1, tzinfo=timezone(timedelta(hours=-6))),
        "float": np.float64(1.5),
        "items": (1, None, True),
    }

    reference = _create(ModelArtifactStore(tmp_path), model_metadata=metadata)
    stored = reference.manifest["identity"]["model_metadata"]

    assert stored["model"] == {"candidate": "k3", "passed": True}
    assert stored["path"] == "relative/evidence"
    assert stored["at"] == "2026-07-14T01:00:00-06:00"
    assert stored["items"] == (1, None, True)


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({}, "at least one"),
        ({"BadName": np.ones(2)}, "name is invalid"),
        ({"bad-name": np.ones(2)}, "name is invalid"),
        ({"scalar": np.array(1.0)}, "non-empty and non-scalar"),
        ({"empty": np.array([], dtype=np.float64)}, "non-empty and non-scalar"),
        ({"object": np.array([object()], dtype=object)}, "dtype is not safely"),
        ({"complex": np.array([1 + 2j])}, "dtype is not safely"),
        ({"bad": np.array([np.inf])}, "non-finite"),
        ({"bad": _BadArray()}, "cannot be converted"),
    ],
)
def test_unsafe_parameter_arrays_are_rejected(
    tmp_path: Path, arrays: dict[str, np.ndarray[Any, Any]], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        _create(ModelArtifactStore(tmp_path), arrays=arrays)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        "/absolute.json",
        r"bad\windows.json",
        "UPPER.json",
        "not-json.txt",
        MODEL_MANIFEST_NAME,
        MODEL_PARAMETERS_NAME,
        ".hidden.json",
    ],
)
def test_unsafe_calibration_document_paths_are_rejected(
    tmp_path: Path, path: str
) -> None:
    with pytest.raises(ModelError, match="path is unsafe or invalid"):
        _create(
            ModelArtifactStore(tmp_path),
            calibration_documents={path: {"status": "passed"}},
        )


def test_file_directory_collision_and_invalid_document_json_are_rejected(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    with pytest.raises(ModelError, match="file/directory collision"):
        _create(
            store,
            calibration_documents={
                "report.json": {"status": "passed"},
                "report.json/detail.json": {"status": "passed"},
            },
        )
    with pytest.raises(ModelError, match="non-empty JSON object"):
        _create(store, calibration_documents={"report.json": {}})
    with pytest.raises(ModelError, match="secret-like key"):
        _create(store, calibration_documents={"report.json": {"cookie": "x"}})
    with pytest.raises(ModelError, match="must be a mapping"):
        _create(store, calibration_documents=[])  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="document path is invalid"):
        _create(store, calibration_documents={1: {"status": "passed"}})


def test_empty_calibration_document_set_is_supported(tmp_path: Path) -> None:
    reference = _create(ModelArtifactStore(tmp_path), calibration_documents={})

    assert reference.manifest["identity"]["documents"] == ()
    assert {path.name for path in reference.path.iterdir()} == {
        MODEL_MANIFEST_NAME,
        MODEL_PARAMETERS_NAME,
    }


def test_invalid_lookup_and_fingerprint_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)

    with pytest.raises(ModelError, match="lookup is not unique") as captured:
        store.read_document(reference.fingerprint, "missing.json")
    assert captured.value.details["matches"] == 0
    with pytest.raises(ModelError, match="unsafe"):
        store.read_document(reference.fingerprint, "../escape.json")
    with pytest.raises(IntegrityError, match="fingerprint is invalid"):
        store.verify("bad")


def test_parameter_and_document_tamper_are_detected(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    parameter_reference = _create(store)
    (parameter_reference.path / MODEL_PARAMETERS_NAME).write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="parameters integrity check failed"):
        store.verify(parameter_reference.fingerprint)

    document_reference = _create(
        store, model_metadata={"candidate": "different-for-second-artifact"}
    )
    (document_reference.path / "block-calibration.json").write_text(
        '{"status":"tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(IntegrityError, match="document integrity check failed"):
        store.verify(document_reference.fingerprint)


def test_load_apis_rehash_files_reopened_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    array_store = ModelArtifactStore(tmp_path / "arrays")
    array_reference = _create(array_store)
    real_array_verify = array_store.verify

    def verify_then_tamper_arrays(
        fingerprint: str, *, reused: bool = True
    ) -> ModelArtifactReference:
        reference = real_array_verify(fingerprint, reused=reused)
        (reference.path / MODEL_PARAMETERS_NAME).write_bytes(b"post-verify tamper")
        return reference

    monkeypatch.setattr(array_store, "verify", verify_then_tamper_arrays)
    with pytest.raises(IntegrityError, match="parameters integrity check failed"):
        array_store.load_arrays(array_reference.fingerprint)

    document_store = ModelArtifactStore(tmp_path / "documents")
    document_reference = _create(document_store)
    real_document_verify = document_store.verify

    def verify_then_tamper_document(
        fingerprint: str, *, reused: bool = True
    ) -> ModelArtifactReference:
        reference = real_document_verify(fingerprint, reused=reused)
        (reference.path / "block-calibration.json").write_bytes(
            b'{"post_verify":"tamper"}\n'
        )
        return reference

    monkeypatch.setattr(document_store, "verify", verify_then_tamper_document)
    with pytest.raises(IntegrityError, match="document integrity check failed"):
        document_store.read_document(
            document_reference.fingerprint, "block-calibration.json"
        )


@pytest.mark.parametrize("extra_kind", ["file", "directory", "symlink"])
def test_unmanifested_tree_nodes_fail_closed(tmp_path: Path, extra_kind: str) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    extra = reference.path / "unexpected"
    if extra_kind == "file":
        extra.write_text("x", encoding="utf-8")
    elif extra_kind == "directory":
        extra.mkdir()
    else:
        outside = tmp_path / "outside"
        outside.write_text("x", encoding="utf-8")
        extra.symlink_to(outside)

    with pytest.raises(IntegrityError, match="tree does not match"):
        store.verify(reference.fingerprint)


def test_manifest_and_payload_symlinks_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    manifest_reference = _create(store)
    manifest = manifest_reference.path / MODEL_MANIFEST_NAME
    content = manifest.read_bytes()
    manifest.unlink()
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_bytes(content)
    manifest.symlink_to(outside_manifest)
    with pytest.raises(IntegrityError, match="escapes its root|symbolic link"):
        store.verify(manifest_reference.fingerprint)

    parameter_reference = _create(
        store, model_metadata={"candidate": "parameter-symlink"}
    )
    parameters = parameter_reference.path / MODEL_PARAMETERS_NAME
    parameter_content = parameters.read_bytes()
    parameters.unlink()
    outside_parameters = tmp_path / "outside-parameters.npz"
    outside_parameters.write_bytes(parameter_content)
    parameters.symlink_to(outside_parameters)
    with pytest.raises(IntegrityError, match="tree does not match|escapes its root"):
        store.verify(parameter_reference.fingerprint)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: [], "root is not an object"),
        (
            lambda value: {**value, "unexpected": True},
            "manifest fields are invalid",
        ),
        (
            lambda value: {**value, "schema_version": 999},
            "schema is unsupported",
        ),
        (
            lambda value: {**value, "model_artifact_fingerprint": "f" * 64},
            "does not match its directory",
        ),
        (lambda value: {**value, "identity": None}, "identity is missing"),
    ],
)
def test_manifest_schema_corruption_fails_closed(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, mutation(manifest))

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


def test_manifest_malformed_missing_and_noncanonical_fail_closed(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    malformed = _create(store)
    (malformed.path / MODEL_MANIFEST_NAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(IntegrityError, match="cannot be read"):
        store.verify(malformed.fingerprint)

    missing = _create(store, model_metadata={"candidate": "missing-manifest"})
    (missing.path / MODEL_MANIFEST_NAME).unlink()
    with pytest.raises(IntegrityError, match="manifest is missing"):
        store.verify(missing.fingerprint)

    noncanonical = _create(store, model_metadata={"candidate": "pretty-manifest"})
    value = _read_manifest(noncanonical)
    _write_manifest(noncanonical.path / MODEL_MANIFEST_NAME, value, canonical=False)
    with pytest.raises(IntegrityError, match="not canonical JSON"):
        store.verify(noncanonical.fingerprint)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda identity: identity.update({"unexpected": True}), "identity fields"),
        (
            lambda identity: identity.update({"schema_version": 999}),
            "identity schema",
        ),
        (
            lambda identity: identity.update({"artifact_role": "candidate"}),
            "role is invalid",
        ),
        (
            lambda identity: identity.update({"data_fingerprint": "bad"}),
            "data fingerprint is invalid",
        ),
        (
            lambda identity: identity.update({"model_metadata": {}}),
            "non-empty JSON object",
        ),
        (
            lambda identity: identity.update({"provenance": {"credential": "leaked"}}),
            "secret-like key",
        ),
    ],
)
def test_identity_schema_corruption_fails_before_hash_comparison(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    identity = dict(manifest["identity"])
    mutate(identity)
    manifest["identity"] = identity
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


def test_valid_identity_edit_without_readdressing_is_hash_mismatch(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    manifest["identity"]["model_metadata"]["candidate"] = "k4"
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match="identity hash mismatch"):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "pickle", "format fields"),
        ("allow_pickle", True, "format fields"),
        ("bytes", 0, "byte size"),
        ("sha256", "bad", "SHA-256"),
        ("arrays", [], "records are empty"),
    ],
)
def test_parameter_record_schema_is_fail_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    parameters = dict(manifest["identity"]["parameters"])
    parameters[field] = value
    manifest["identity"]["parameters"] = parameters
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "record is missing or invalid"),
        ({"unexpected": True}, "record fields are invalid"),
    ],
)
def test_parameter_record_type_and_exact_fields_are_enforced(
    tmp_path: Path, value: object, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    manifest["identity"]["parameters"] = value
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda record: record.update({"name": "Bad"}),
            "record name is invalid",
        ),
        (
            lambda record: record.update({"member_name": "wrong.npy"}),
            "member name is invalid",
        ),
        (
            lambda record: record.update({"dtype": ">i8"}),
            "unsupported or noncanonical",
        ),
        (lambda record: record.update({"shape": []}), "shape is invalid"),
        (lambda record: record.update({"order": "F"}), "order is invalid"),
        (lambda record: record.update({"bytes": False}), "byte size is invalid"),
    ],
)
def test_array_record_schema_is_fail_closed(
    tmp_path: Path, mutator: Any, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    parameters = dict(manifest["identity"]["parameters"])
    records = [dict(record) for record in parameters["arrays"]]
    mutator(records[0])
    parameters["arrays"] = records
    manifest["identity"]["parameters"] = parameters
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-record", "array record is invalid"),
        (
            {
                "name": "state",
                "member_name": "state.npy",
                "dtype": 4,
                "shape": [1],
                "order": "C",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "array dtype is invalid",
        ),
        (
            {
                "name": "state",
                "member_name": "state.npy",
                "dtype": "not-a-dtype",
                "shape": [1],
                "order": "C",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "array dtype is invalid",
        ),
    ],
)
def test_array_record_type_and_dtype_parser_fail_closed(
    tmp_path: Path, value: object, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    manifest["identity"]["parameters"]["arrays"] = [value]
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


def test_unsorted_and_duplicate_array_records_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    parameters = dict(manifest["identity"]["parameters"])
    parameters["arrays"] = list(reversed(parameters["arrays"]))
    manifest["identity"]["parameters"] = parameters
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)
    with pytest.raises(IntegrityError, match="unsorted or duplicated"):
        store.verify(reference.fingerprint)


def test_document_record_schema_and_path_are_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    document = dict(manifest["identity"]["documents"][0])
    document["relative_path"] = "../escape.json"
    manifest["identity"]["documents"] = [document]
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)
    with pytest.raises(IntegrityError, match="unsafe or invalid"):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        (None, "document records are invalid"),
        (["not-a-record"], "document record is invalid"),
        ([{"relative_path": 4}], "document record fields are invalid"),
    ],
)
def test_document_record_types_and_exact_fields_are_enforced(
    tmp_path: Path, documents: object, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    manifest["identity"]["documents"] = documents
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("relative_path", 4, "document path is invalid"),
        ("media_type", "text/plain", "media type is invalid"),
        ("bytes", 0, "byte size is invalid"),
        ("sha256", "bad", "SHA-256 is invalid"),
    ],
)
def test_document_record_field_values_are_enforced(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    documents = [dict(record) for record in manifest["identity"]["documents"]]
    documents[0][field] = value
    manifest["identity"]["documents"] = documents
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


@pytest.mark.parametrize("mode", ["duplicate", "unsorted"])
def test_duplicate_or_unsorted_document_records_fail_closed(
    tmp_path: Path, mode: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    manifest = _read_manifest(reference)
    documents = [dict(record) for record in manifest["identity"]["documents"]]
    manifest["identity"]["documents"] = (
        [documents[0], documents[0]] if mode == "duplicate" else documents[::-1]
    )
    _write_manifest(reference.path / MODEL_MANIFEST_NAME, manifest)

    expected = "paths are duplicated" if mode == "duplicate" else "records are unsorted"
    with pytest.raises(IntegrityError, match=expected):
        store.verify(reference.fingerprint)


def test_npz_extra_duplicate_and_reordered_members_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)

    extra = _create(store, model_metadata={"case": "extra"})
    members = _archive_members((extra.path / MODEL_PARAMETERS_NAME).read_bytes())
    fingerprint = _replace_parameter_archive(
        store, extra, _zip_bytes([*members, ("extra.npy", members[0][1])])
    )
    with pytest.raises(IntegrityError, match="order or allowlist"):
        store.verify(fingerprint)

    duplicate = _create(store, model_metadata={"case": "duplicate"})
    members = _archive_members((duplicate.path / MODEL_PARAMETERS_NAME).read_bytes())
    with pytest.warns(UserWarning, match="Duplicate name"):
        archive = _zip_bytes([members[0], members[0], members[1]])
    fingerprint = _replace_parameter_archive(store, duplicate, archive)
    with pytest.raises(IntegrityError, match="duplicate members"):
        store.verify(fingerprint)

    reordered = _create(store, model_metadata={"case": "reordered"})
    members = _archive_members((reordered.path / MODEL_PARAMETERS_NAME).read_bytes())
    fingerprint = _replace_parameter_archive(
        store, reordered, _zip_bytes(members[::-1])
    )
    with pytest.raises(IntegrityError, match="order or allowlist"):
        store.verify(fingerprint)


@pytest.mark.parametrize(
    "archive_builder",
    [
        lambda members: _zip_bytes(members, compression=zipfile.ZIP_DEFLATED),
        lambda members: _zip_bytes(members, timestamp=(2026, 7, 14, 0, 0, 0)),
        lambda members: _zip_bytes(members, archive_comment=b"not-canonical"),
    ],
)
def test_noncanonical_npz_metadata_fails_closed(
    tmp_path: Path, archive_builder: Any
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    members = _archive_members((reference.path / MODEL_PARAMETERS_NAME).read_bytes())
    fingerprint = _replace_parameter_archive(store, reference, archive_builder(members))

    with pytest.raises(IntegrityError, match="non-canonical|non-empty comment"):
        store.verify(fingerprint)


def test_npz_member_hash_size_dtype_shape_and_canonical_bytes_are_verified(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)

    bad_hash = _create(store, model_metadata={"case": "member-hash"})
    content = (bad_hash.path / MODEL_PARAMETERS_NAME).read_bytes()

    def change_hash(parameters: dict[str, Any]) -> None:
        records = [dict(record) for record in parameters["arrays"]]
        records[0]["sha256"] = "f" * 64
        parameters["arrays"] = records

    fingerprint = _replace_parameter_archive(
        store, bad_hash, content, identity_mutator=change_hash
    )
    with pytest.raises(IntegrityError, match="member hash"):
        store.verify(fingerprint)

    bad_size = _create(store, model_metadata={"case": "member-size"})
    content = (bad_size.path / MODEL_PARAMETERS_NAME).read_bytes()

    def change_size(parameters: dict[str, Any]) -> None:
        records = [dict(record) for record in parameters["arrays"]]
        records[0]["bytes"] += 1
        parameters["arrays"] = records

    fingerprint = _replace_parameter_archive(
        store, bad_size, content, identity_mutator=change_size
    )
    with pytest.raises(IntegrityError, match="member size"):
        store.verify(fingerprint)

    bad_dtype = _create(store, model_metadata={"case": "dtype"})
    content = (bad_dtype.path / MODEL_PARAMETERS_NAME).read_bytes()

    def change_dtype(parameters: dict[str, Any]) -> None:
        records = [dict(record) for record in parameters["arrays"]]
        records[0]["dtype"] = "<i4"
        parameters["arrays"] = records

    fingerprint = _replace_parameter_archive(
        store, bad_dtype, content, identity_mutator=change_dtype
    )
    with pytest.raises(IntegrityError, match="dtype or shape"):
        store.verify(fingerprint)

    noncanonical = _create(store, model_metadata={"case": "npy-trailing"})
    members = _archive_members((noncanonical.path / MODEL_PARAMETERS_NAME).read_bytes())
    changed_members = [(members[0][0], members[0][1] + b"trailing"), members[1]]
    archive = _zip_bytes(changed_members)

    def update_first_member(parameters: dict[str, Any]) -> None:
        records = [dict(record) for record in parameters["arrays"]]
        records[0]["bytes"] = len(changed_members[0][1])
        records[0]["sha256"] = hashlib.sha256(changed_members[0][1]).hexdigest()
        parameters["arrays"] = records

    fingerprint = _replace_parameter_archive(
        store, noncanonical, archive, identity_mutator=update_first_member
    )
    with pytest.raises(IntegrityError, match="bytes are non-canonical"):
        store.verify(fingerprint)


def test_malformed_npz_is_an_integrity_failure(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    fingerprint = _replace_parameter_archive(store, reference, b"not-a-zip")

    with pytest.raises(IntegrityError, match="cannot be read safely"):
        store.verify(fingerprint)


@pytest.mark.parametrize("placement", ["prefix", "suffix"])
def test_npz_must_equal_exact_regenerated_canonical_archive(
    tmp_path: Path, placement: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    canonical = (reference.path / MODEL_PARAMETERS_NAME).read_bytes()
    altered = (
        b"self-extracting-prefix" + canonical
        if placement == "prefix"
        else canonical + b"trailing-after-end-of-central-directory"
    )
    fingerprint = _replace_parameter_archive(store, reference, altered)

    with pytest.raises(IntegrityError, match="bytes or records are not canonical"):
        store.verify(fingerprint)


def test_invalid_and_nonfinite_npy_members_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)

    invalid = _create(store, model_metadata={"case": "invalid-npy"})
    members = _archive_members((invalid.path / MODEL_PARAMETERS_NAME).read_bytes())
    invalid_member = b"not-an-npy"
    archive = _zip_bytes([(members[0][0], invalid_member), members[1]])

    def update_invalid(parameters: dict[str, Any]) -> None:
        records = [dict(record) for record in parameters["arrays"]]
        records[0]["bytes"] = len(invalid_member)
        records[0]["sha256"] = hashlib.sha256(invalid_member).hexdigest()
        parameters["arrays"] = records

    fingerprint = _replace_parameter_archive(
        store, invalid, archive, identity_mutator=update_invalid
    )
    with pytest.raises(IntegrityError, match="cannot be loaded without pickle"):
        store.verify(fingerprint)

    nonfinite = _create(store, model_metadata={"case": "nonfinite-npy"})
    members = _archive_members((nonfinite.path / MODEL_PARAMETERS_NAME).read_bytes())
    buffer = io.BytesIO()
    np.save(
        buffer,
        np.array([[np.nan, 0.1], [0.2, 0.8]], dtype=np.float64),
        allow_pickle=False,
    )
    nonfinite_member = buffer.getvalue()
    archive = _zip_bytes([members[0], (members[1][0], nonfinite_member)])

    def update_nonfinite(parameters: dict[str, Any]) -> None:
        records = [dict(record) for record in parameters["arrays"]]
        records[1]["bytes"] = len(nonfinite_member)
        records[1]["sha256"] = hashlib.sha256(nonfinite_member).hexdigest()
        parameters["arrays"] = records

    fingerprint = _replace_parameter_archive(
        store, nonfinite, archive, identity_mutator=update_nonfinite
    )
    with pytest.raises(IntegrityError, match="contains a non-finite"):
        store.verify(fingerprint)


def test_npy_loader_defenses_reject_nonarrays_unsafe_dtypes_and_fortran_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record: dict[str, Any] = {
        "name": "value",
        "member_name": "value.npy",
        "dtype": "<f8",
        "shape": [2, 2],
        "order": "C",
        "bytes": 1,
        "sha256": "a" * 64,
    }
    monkeypatch.setattr(artifact_module.np, "load", lambda *_args, **_kwargs: 4)
    with pytest.raises(IntegrityError, match="is not an array"):
        artifact_module._load_npy_member(b"x", record)

    monkeypatch.setattr(
        artifact_module.np,
        "load",
        lambda *_args, **_kwargs: np.array([object()], dtype=object),
    )
    unsafe_record = {**record, "dtype": "|O", "shape": [1]}
    with pytest.raises(IntegrityError, match="dtype is unsafe"):
        artifact_module._load_npy_member(b"x", unsafe_record)

    monkeypatch.undo()
    fortran = np.asfortranarray(np.arange(4.0).reshape(2, 2))
    buffer = io.BytesIO()
    np.save(buffer, fortran, allow_pickle=False)
    with pytest.raises(IntegrityError, match="bytes are non-canonical"):
        artifact_module._load_npy_member(buffer.getvalue(), record)


def test_np_load_is_always_called_with_pickle_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    real_load = artifact_module.np.load
    calls: list[bool] = []

    def guarded_load(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        calls.append(kwargs.get("allow_pickle"))
        assert kwargs.get("allow_pickle") is False
        return real_load(*args, **kwargs)

    monkeypatch.setattr(artifact_module.np, "load", guarded_load)
    arrays = store.load_arrays(reference.fingerprint)

    assert set(arrays) == {"decoded_states", "transition_matrix"}
    assert calls and all(value is False for value in calls)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"status": "passed"}\n', "not canonical JSON"),
        (b"[]\n", "root is not an object"),
        (b"{broken", "cannot be read"),
        (b'{"api_key":"leak"}\n', "secret-like key"),
        (b'{"value":NaN}\n', "non-finite number"),
    ],
)
def test_readdressed_invalid_json_document_still_fails_closed(
    tmp_path: Path, content: bytes, message: str
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    identity = dict(reference.manifest["identity"])
    documents = [dict(record) for record in identity["documents"]]
    path = documents[0]["relative_path"]
    documents[0]["bytes"] = len(content)
    documents[0]["sha256"] = hashlib.sha256(content).hexdigest()
    identity["documents"] = documents
    (reference.path / path).write_bytes(content)
    fingerprint, _ = _readdress(store, reference, identity)

    with pytest.raises(IntegrityError, match=message):
        store.verify(fingerprint)


def test_low_level_path_and_read_failures_are_stable_integrity_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(IntegrityError, match="relative path is unsafe"):
        artifact_module._safe_artifact_path(root, "../escape", must_exist=False)

    loop = root / "loop"
    loop.symlink_to(loop)
    with pytest.raises(IntegrityError, match="path cannot be resolved") as loop_error:
        artifact_module._safe_artifact_path(root, "loop", must_exist=True)
    assert loop_error.value.details["error_type"] == "RuntimeError"

    target = root / "target"
    target.write_text("value", encoding="utf-8")
    internal = root / "internal"
    internal.symlink_to(target)
    with pytest.raises(IntegrityError, match="symbolic link"):
        artifact_module._safe_artifact_path(root, "internal", must_exist=True)

    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise PermissionError("simulated unreadable file")

    monkeypatch.setattr(artifact_module.os, "read", fail_read)
    with pytest.raises(IntegrityError, match="cannot be read") as read_error:
        artifact_module._read_regular_file(root, "target", label="test payload")
    assert read_error.value.details["error_type"] == "PermissionError"


def test_descriptor_reader_detects_in_place_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "payload"
    target.write_bytes(b"immutable payload")
    original_mtime = target.stat().st_mtime_ns
    real_read = artifact_module.os.read
    touched = False

    def read_then_touch(descriptor: int, size: int) -> bytes:
        nonlocal touched
        content = real_read(descriptor, size)
        if not touched:
            touched = True
            os.utime(
                target,
                ns=(original_mtime + 1_000_000_000, original_mtime + 1_000_000_000),
            )
        return content

    monkeypatch.setattr(artifact_module.os, "read", read_then_touch)
    with pytest.raises(IntegrityError, match="changed while it was read"):
        artifact_module._read_regular_file(root, "payload", label="test payload")


def test_symlink_ancestor_inspection_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    real_lstat = Path.lstat

    def fail_target(path: Path) -> os.stat_result:
        if path == target:
            raise PermissionError("simulated lstat denial")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_target)
    with pytest.raises(IntegrityError, match="cannot be inspected") as captured:
        artifact_module._reject_symlink_ancestors(target, error_type=IntegrityError)
    assert captured.value.details["error_type"] == "PermissionError"


def test_special_files_and_canonical_encoder_fail_closed(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    fifo = reference.path / "unexpected-fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(IntegrityError, match="tree does not match") as captured:
            store.verify(reference.fingerprint)
        assert "unexpected-fifo" in captured.value.details["unsafe"]
    finally:
        fifo.unlink()

    with pytest.raises(ModelError, match="encoded canonically"):
        artifact_module._canonical_json_bytes({"unsupported": {1, 2}})


@pytest.mark.parametrize("generic", [False, True])
def test_atomic_publish_handles_a_winning_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generic: bool
) -> None:
    store = ModelArtifactStore(tmp_path)
    real_rename = os.rename

    def concurrent_winner(source: str | Path, destination: str | Path) -> None:
        real_rename(source, destination)
        if generic:
            raise OSError("simulated EEXIST variant")
        raise FileExistsError("simulated race")

    monkeypatch.setattr(artifact_module.os, "rename", concurrent_winner)
    reference = _create(store)

    assert reference.reused
    assert store.verify(reference.fingerprint).fingerprint == reference.fingerprint
    assert not tuple(store.store_root.glob(".stage-*"))


def test_publish_and_write_failures_leave_no_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelArtifactStore(tmp_path / "rename")

    def fail_rename(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("simulated disk error")

    monkeypatch.setattr(artifact_module.os, "rename", fail_rename)
    with pytest.raises(OSError, match="disk error"):
        _create(store)
    assert not tuple(store.store_root.glob(".stage-*"))
    assert not tuple(
        path for path in store.store_root.iterdir() if len(path.name) == 64
    )

    monkeypatch.undo()
    fsync_store = ModelArtifactStore(tmp_path / "fsync")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(artifact_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        _create(fsync_store)
    assert not tuple(fsync_store.store_root.glob(".stage-*"))


def test_post_rename_parent_fsync_failure_is_not_misclassified_as_a_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelArtifactStore(tmp_path)
    real_fsync_directory = artifact_module._fsync_directory

    def fail_only_published_parent(path: Path) -> None:
        if path == store.store_root:
            raise OSError("post-rename parent fsync failed")
        real_fsync_directory(path)

    monkeypatch.setattr(artifact_module, "_fsync_directory", fail_only_published_parent)
    with pytest.raises(OSError, match="post-rename parent fsync failed"):
        _create(store)

    published = tuple(
        path for path in store.store_root.iterdir() if len(path.name) == 64
    )
    assert len(published) == 1
    assert not tuple(store.store_root.glob(".stage-*"))

    monkeypatch.setattr(artifact_module, "_fsync_directory", real_fsync_directory)
    reused = _create(store)
    assert reused.reused


def test_unsafe_store_ancestors_and_missing_artifact_fail_closed(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "model").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError, match="store path is unsafe"):
        _create(ModelArtifactStore(artifact_root))

    symlink_root = tmp_path / "artifact-root-link"
    symlink_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symbolic-link ancestor"):
        _create(ModelArtifactStore(symlink_root))

    file_root = tmp_path / "artifact-root-file"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(IntegrityError, match="root is not a safe directory"):
        _create(ModelArtifactStore(file_root))

    unsafe_sha_root = tmp_path / "unsafe-sha"
    (unsafe_sha_root / "model").mkdir(parents=True)
    (unsafe_sha_root / "model" / "sha256").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError, match="store path is unsafe"):
        _create(ModelArtifactStore(unsafe_sha_root))

    absent_store = ModelArtifactStore(tmp_path / "absent-store")
    with pytest.raises(IntegrityError, match="store path is missing or unsafe"):
        absent_store.verify("e" * 64)

    clean = ModelArtifactStore(tmp_path / "clean")
    clean._prepare_store_root()
    with pytest.raises(IntegrityError, match="directory is missing"):
        clean.verify("f" * 64)


def test_indirect_symlink_ancestor_is_rejected_before_store_creation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-parent"
    (outside / "nested").mkdir(parents=True)
    link = tmp_path / "linked-parent"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(IntegrityError, match="symbolic-link ancestor") as captured:
        _create(ModelArtifactStore(link / "nested" / "artifacts"))

    assert captured.value.details["path"] == str(link)
    assert not (outside / "nested" / "artifacts").exists()


def test_descriptor_reader_rejects_hardlinks_and_uses_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelArtifactStore(tmp_path / "hardlink")
    reference = _create(store)
    document = reference.path / "block-calibration.json"
    outside = tmp_path / "outside-document.json"
    document.rename(outside)
    os.link(outside, document)
    with pytest.raises(IntegrityError, match="not a private regular file"):
        store.verify(reference.fingerprint)

    nofollow_store = ModelArtifactStore(tmp_path / "nofollow")
    nofollow_reference = _create(nofollow_store)
    real_open = artifact_module.os.open
    observed_flags: list[int] = []

    def recording_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        observed_flags.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifact_module.os, "open", recording_open)
    nofollow_store.verify(nofollow_reference.fingerprint)
    if hasattr(os, "O_NOFOLLOW"):
        assert observed_flags
        assert all(flags & os.O_NOFOLLOW for flags in observed_flags)


def test_verified_reuse_rejects_an_identity_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ModelArtifactStore(tmp_path)
    original = _create(store)
    real_verify = store.verify

    def mismatched_verify(
        fingerprint: str, *, reused: bool = True
    ) -> ModelArtifactReference:
        reference = real_verify(fingerprint, reused=reused)
        manifest = dict(reference.manifest)
        manifest["identity"] = {"different": True}
        return ModelArtifactReference(
            reference.fingerprint,
            reference.path,
            manifest,
            reference.reused,
        )

    monkeypatch.setattr(store, "verify", mismatched_verify)
    with pytest.raises(IntegrityError, match="fingerprint collision"):
        _create(store)
    assert original.path.exists()


def test_verified_reuse_preserves_race_failure_as_exception_cause(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    reference = _create(store)
    race_error = OSError("simulated race")

    with pytest.raises(IntegrityError, match="fingerprint collision") as collision:
        store._verified_reuse(
            reference.path,
            reference.fingerprint,
            {"different": True},
            cause=race_error,
        )
    assert collision.value.__cause__ is race_error

    missing = "f" * 64
    with pytest.raises(IntegrityError, match="directory is missing") as missing_error:
        store._verified_reuse(
            store.store_root / missing,
            missing,
            {"different": True},
            cause=race_error,
        )
    assert missing_error.value.__cause__ is race_error

    with pytest.raises(IntegrityError, match="directory is missing") as plain_error:
        store._verified_reuse(
            store.store_root / ("e" * 64),
            "e" * 64,
            {"different": True},
        )
    assert plain_error.value.__cause__ is None
