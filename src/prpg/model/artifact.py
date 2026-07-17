"""Immutable, content-addressed storage for Phase 3 model artifacts.

The statistical primitives intentionally return ordinary arrays and scalars.
This module is the serialization boundary that turns those values into an
auditable artifact without ever serializing a fitted Python object.  A model
artifact contains exactly:

* ``model-manifest.json`` -- canonical JSON identity and provenance;
* ``model-parameters.npz`` -- a deterministic ZIP of non-pickle ``.npy``
  members; and
* zero or more explicitly allowlisted canonical JSON calibration reports.

The artifact fingerprint is the SHA-256 of its canonical scientific identity.
That identity includes the processed-data fingerprint, artifact role, model
metadata, producer provenance, the exact NPZ hash and per-array records, and
every calibration-document hash.  The manifest contains no unauthenticated
observational fields or timestamps: operational timing belongs in the run
ledger.  Existing content is never overwritten and is reused only after
complete offline verification.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from prpg.errors import IntegrityError, ModelError, PRPGError

MODEL_ARTIFACT_SCHEMA_VERSION = 1
MODEL_MANIFEST_NAME = "model-manifest.json"
MODEL_PARAMETERS_NAME = "model-parameters.npz"
NPZ_FORMAT_VERSION = "deterministic-npz-v1"

ModelArtifactRole: TypeAlias = Literal["design", "production"]
_ARTIFACT_ROLES = frozenset({"design", "production"})
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_PATH_COMPONENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_SHA256_PATTERN = _FINGERPRINT_PATTERN
_SUPPORTED_DTYPE_KINDS = frozenset("biuf")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTRIBUTES = 0o600 << 16
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[\s_.-]*key|password|secret|access[\s_.-]*token)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9._~+/-]{12,}"
    ),
    re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]{8,}@"),
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "model_artifact_fingerprint",
        "identity",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "artifact_role",
        "data_fingerprint",
        "model_metadata",
        "provenance",
        "parameters",
        "documents",
    }
)
_PARAMETER_KEYS = frozenset(
    {
        "relative_path",
        "format",
        "allow_pickle",
        "bytes",
        "sha256",
        "arrays",
    }
)
_ARRAY_RECORD_KEYS = frozenset(
    {"name", "member_name", "dtype", "shape", "order", "bytes", "sha256"}
)
_DOCUMENT_RECORD_KEYS = frozenset({"relative_path", "media_type", "bytes", "sha256"})

ErrorType: TypeAlias = type[ModelError] | type[IntegrityError]


@dataclass(frozen=True, slots=True)
class ModelArtifactReference:
    """A verified integrity reference, never a scientific gate authorization.

    ``artifact_role=production`` describes the full-data fit scope only.
    Neither construction nor :meth:`ModelArtifactStore.verify` proves G3 or
    authorizes generation; a separate gate orchestrator must do that.
    """

    verification_scope: ClassVar[Literal["integrity_only"]] = "integrity_only"
    generation_authorized: ClassVar[Literal[False]] = False

    fingerprint: str
    path: Path
    manifest: Mapping[str, Any]
    reused: bool


class ModelArtifactStore:
    """Publish and verify artifacts below ``artifact_root/model/sha256``."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store_root = self.artifact_root / "model" / "sha256"

    def create(
        self,
        *,
        artifact_role: ModelArtifactRole,
        data_fingerprint: str,
        model_metadata: Mapping[str, Any],
        arrays: Mapping[str, npt.ArrayLike],
        calibration_documents: Mapping[str, Mapping[str, Any]],
        provenance: Mapping[str, Any],
    ) -> ModelArtifactReference:
        """Atomically create or safely reuse one scientific model artifact.

        ``model_metadata`` and ``provenance`` are identity-bearing JSON
        objects.  Calibration-document keys are safe relative ``.json`` paths
        such as ``block-calibration.json``.  Array keys are lower-case
        snake-case NPZ member names.  Arrays are canonicalized to contiguous
        little-endian numeric storage and must be finite and non-empty.

        The role describes fit scope only.  In particular, creating a
        ``production``-role artifact does not assert G3 passage or authorize
        generation; those decisions belong to the calibration orchestrator
        and gate evidence.
        """

        _validate_role(artifact_role, error_type=ModelError)
        _validate_fingerprint(
            data_fingerprint, label="data fingerprint", error_type=ModelError
        )
        normalized_metadata = _normalize_json_object(
            model_metadata,
            label="model metadata",
            error_type=ModelError,
            require_nonempty=True,
        )
        normalized_provenance = _normalize_json_object(
            provenance,
            label="model provenance",
            error_type=ModelError,
            require_nonempty=True,
        )
        canonical_arrays = _canonical_arrays(arrays)
        parameter_bytes, array_records = _npz_bytes(canonical_arrays)
        parameter_record: dict[str, Any] = {
            "relative_path": MODEL_PARAMETERS_NAME,
            "format": NPZ_FORMAT_VERSION,
            "allow_pickle": False,
            "bytes": len(parameter_bytes),
            "sha256": _sha256_bytes(parameter_bytes),
            "arrays": array_records,
        }
        document_bytes, document_records = _canonical_documents(calibration_documents)
        identity: dict[str, Any] = {
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "artifact_role": artifact_role,
            "data_fingerprint": data_fingerprint,
            "model_metadata": normalized_metadata,
            "provenance": normalized_provenance,
            "parameters": parameter_record,
            "documents": document_records,
        }
        fingerprint = _sha256_bytes(_canonical_json_bytes(identity))

        self._prepare_store_root()
        destination = self.store_root / fingerprint
        if destination.exists() or destination.is_symlink():
            return self._verified_reuse(destination, fingerprint, identity)

        stage = Path(
            tempfile.mkdtemp(prefix=f".stage-{fingerprint[:12]}-", dir=self.store_root)
        )
        try:
            _write_new_file(stage / MODEL_PARAMETERS_NAME, parameter_bytes)
            for relative_path, content in document_bytes.items():
                target = stage.joinpath(*PurePosixPath(relative_path).parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                _write_new_file(target, content)

            manifest: dict[str, Any] = {
                "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
                "model_artifact_fingerprint": fingerprint,
                "identity": identity,
            }
            _write_new_file(
                stage / MODEL_MANIFEST_NAME, _canonical_json_bytes(manifest)
            )
            _fsync_tree(stage)

            try:
                os.rename(stage, destination)
            except OSError as error:
                if destination.exists() or destination.is_symlink():
                    return self._verified_reuse(
                        destination, fingerprint, identity, cause=error
                    )
                raise
            # This fsync is deliberately outside the rename race handler.  A
            # durability failure after a successful rename must propagate; it
            # is not evidence that a concurrent writer won the content race.
            _fsync_directory(self.store_root)
            return self.verify(fingerprint, reused=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def verify(
        self, fingerprint: str, *, reused: bool = True
    ) -> ModelArtifactReference:
        """Verify stored integrity only; this never authorizes G3/generation."""

        _validate_fingerprint(
            fingerprint, label="model artifact fingerprint", error_type=IntegrityError
        )
        self._validate_store_path()
        root = self.store_root / fingerprint
        _reject_symlink_ancestors(root, error_type=IntegrityError)
        if not root.is_dir() or root.is_symlink():
            raise IntegrityError(
                "model artifact directory is missing or unsafe",
                details={"fingerprint": fingerprint},
            )

        manifest_bytes = _read_regular_file(
            root,
            MODEL_MANIFEST_NAME,
            label="model artifact manifest",
            details={"fingerprint": fingerprint},
        )
        manifest = _parse_json_object(manifest_bytes, label="model artifact manifest")
        if manifest_bytes != _canonical_json_bytes(manifest):
            raise IntegrityError("model artifact manifest is not canonical JSON")
        _require_exact_keys(manifest, _MANIFEST_KEYS, label="model artifact manifest")
        if manifest.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise IntegrityError("model artifact manifest schema is unsupported")
        if manifest.get("model_artifact_fingerprint") != fingerprint:
            raise IntegrityError(
                "model artifact manifest fingerprint does not match its directory"
            )

        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise IntegrityError("model artifact identity is missing or invalid")
        _validate_identity(identity)
        computed = _sha256_bytes(_canonical_json_bytes(identity))
        if computed != fingerprint:
            raise IntegrityError(
                "model artifact identity hash mismatch",
                details={"expected": fingerprint, "computed": computed},
            )

        parameters = identity["parameters"]
        documents = identity["documents"]
        expected_files = {
            MODEL_MANIFEST_NAME,
            parameters["relative_path"],
            *(record["relative_path"] for record in documents),
        }
        _verify_tree_allowlist(root, expected_files)

        parameter_bytes = _read_regular_file(
            root, parameters["relative_path"], label="model parameters"
        )
        _verify_file_record(parameter_bytes, parameters, label="model parameters")
        _read_and_verify_npz(parameter_bytes, parameters["arrays"])

        for record in documents:
            content = _read_regular_file(
                root,
                record["relative_path"],
                label="model calibration document",
                details={"relative_path": record["relative_path"]},
            )
            _verify_file_record(
                content,
                record,
                label="model calibration document",
                details={"relative_path": record["relative_path"]},
            )
            _verify_json_document(content, record["relative_path"])

        frozen_manifest = _deep_freeze(manifest)
        if not isinstance(frozen_manifest, Mapping):  # pragma: no cover - invariant
            raise AssertionError("frozen manifest is not a mapping")
        return ModelArtifactReference(fingerprint, root, frozen_manifest, reused)

    def load_arrays(self, fingerprint: str) -> Mapping[str, np.ndarray[Any, Any]]:
        """Return verified, read-only model arrays without enabling pickle."""

        reference = self.verify(fingerprint)
        parameters = reference.manifest["identity"]["parameters"]
        content = _read_regular_file(
            reference.path,
            parameters["relative_path"],
            label="model parameters",
        )
        _verify_file_record(content, parameters, label="model parameters")
        arrays = _read_and_verify_npz(content, parameters["arrays"])
        return MappingProxyType(arrays)

    def read_document(self, fingerprint: str, relative_path: str) -> Mapping[str, Any]:
        """Return one verified calibration JSON object by its allowlisted path."""

        _validate_document_path(relative_path, error_type=ModelError)
        reference = self.verify(fingerprint)
        records = [
            record
            for record in reference.manifest["identity"]["documents"]
            if record["relative_path"] == relative_path
        ]
        if len(records) != 1:
            raise ModelError(
                "model calibration document lookup is not unique",
                details={
                    "fingerprint": fingerprint,
                    "relative_path": relative_path,
                    "matches": len(records),
                },
            )
        content = _read_regular_file(
            reference.path,
            relative_path,
            label="model calibration document",
        )
        _verify_file_record(
            content,
            records[0],
            label="model calibration document",
            details={"relative_path": relative_path},
        )
        value = _deep_freeze(_verify_json_document(content, relative_path))
        if not isinstance(value, Mapping):  # pragma: no cover - invariant
            raise AssertionError("frozen calibration document is not a mapping")
        return value

    def _prepare_store_root(self) -> None:
        _reject_symlink_ancestors(self.artifact_root, error_type=IntegrityError)
        if self.artifact_root.exists() or self.artifact_root.is_symlink():
            if not self.artifact_root.is_dir() or self.artifact_root.is_symlink():
                raise IntegrityError("model artifact root is not a safe directory")
        else:
            self.artifact_root.mkdir(parents=True, mode=0o700)
        if not self.artifact_root.is_dir() or self.artifact_root.is_symlink():
            raise IntegrityError("model artifact root is not a safe directory")
        model_root = self.artifact_root / "model"
        if model_root.exists() and (not model_root.is_dir() or model_root.is_symlink()):
            raise IntegrityError("model artifact store path is unsafe")
        model_root.mkdir(exist_ok=True, mode=0o700)
        if self.store_root.exists() and (
            not self.store_root.is_dir() or self.store_root.is_symlink()
        ):
            raise IntegrityError("model artifact store path is unsafe")
        self.store_root.mkdir(exist_ok=True, mode=0o700)
        _reject_symlink_ancestors(self.store_root, error_type=IntegrityError)

    def _validate_store_path(self) -> None:
        model_root = self.artifact_root / "model"
        _reject_symlink_ancestors(self.store_root, error_type=IntegrityError)
        if (
            not self.artifact_root.is_dir()
            or self.artifact_root.is_symlink()
            or not model_root.is_dir()
            or model_root.is_symlink()
            or not self.store_root.is_dir()
            or self.store_root.is_symlink()
        ):
            raise IntegrityError("model artifact store path is missing or unsafe")

    def _verified_reuse(
        self,
        path: Path,
        fingerprint: str,
        expected_identity: Mapping[str, Any],
        *,
        cause: OSError | None = None,
    ) -> ModelArtifactReference:
        try:
            reference = self.verify(fingerprint, reused=True)
        except PRPGError as error:
            if cause is not None:
                raise error from cause
            raise
        if _canonical_json_bytes(reference.manifest["identity"]) != (
            _canonical_json_bytes(expected_identity)
        ):
            collision = IntegrityError(
                "immutable model artifact fingerprint collision",
                details={"fingerprint": fingerprint, "path": str(path)},
            )
            if cause is not None:
                raise collision from cause
            raise collision
        return reference


def _canonical_arrays(
    arrays: Mapping[str, npt.ArrayLike],
) -> dict[str, np.ndarray[Any, Any]]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ModelError("model artifact requires at least one parameter array")
    result: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in sorted(arrays.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not _ARRAY_NAME_PATTERN.fullmatch(name):
            raise ModelError(
                "model parameter array name is invalid", details={"name": str(name)}
            )
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ModelError(
                "model parameter array cannot be converted",
                details={"name": name, "error_type": type(error).__name__},
            ) from error
        if array.ndim == 0 or array.size == 0:
            raise ModelError(
                "model parameter array must be non-empty and non-scalar",
                details={"name": name, "shape": list(array.shape)},
            )
        if array.dtype.fields is not None or array.dtype.kind not in (
            _SUPPORTED_DTYPE_KINDS
        ):
            raise ModelError(
                "model parameter array dtype is not safely serializable",
                details={"name": name, "dtype": array.dtype.str},
            )
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise ModelError(
                "model parameter array contains a non-finite value",
                details={"name": name},
            )
        target_dtype = (
            array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
        )
        result[name] = np.ascontiguousarray(array.astype(target_dtype, copy=False))
    return result


def _npz_bytes(
    arrays: Mapping[str, np.ndarray[Any, Any]],
) -> tuple[bytes, list[dict[str, Any]]]:
    archive_buffer = io.BytesIO()
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name, array in sorted(arrays.items()):
            member_name = f"{name}.npy"
            content = _npy_bytes(array)
            info = zipfile.ZipInfo(member_name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _ZIP_EXTERNAL_ATTRIBUTES
            archive.writestr(info, content)
            records.append(
                {
                    "name": name,
                    "member_name": member_name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "order": "C",
                    "bytes": len(content),
                    "sha256": _sha256_bytes(content),
                }
            )
    return archive_buffer.getvalue(), records


def _canonical_documents(
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    if not isinstance(documents, Mapping):
        raise ModelError("model calibration documents must be a mapping")
    _validate_document_paths(tuple(documents), error_type=ModelError)
    encoded: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for relative_path, value in sorted(documents.items()):
        normalized = _normalize_json_object(
            value,
            label=f"model calibration document {relative_path}",
            error_type=ModelError,
            require_nonempty=True,
        )
        content = _canonical_json_bytes(normalized)
        encoded[relative_path] = content
        records.append(
            {
                "relative_path": relative_path,
                "media_type": "application/json",
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
    return encoded, records


def _read_and_verify_npz(
    content: bytes, records: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray[Any, Any]]:
    expected_members = [record["member_name"] for record in records]
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            if archive.comment:
                raise IntegrityError("model parameter NPZ has a non-empty comment")
            infos = archive.infolist()
            actual_members = [info.filename for info in infos]
            if len(actual_members) != len(set(actual_members)):
                raise IntegrityError("model parameter NPZ has duplicate members")
            if actual_members != expected_members:
                raise IntegrityError(
                    "model parameter NPZ member order or allowlist is invalid",
                    details={
                        "expected": expected_members,
                        "actual": actual_members,
                    },
                )
            for info, record in zip(infos, records, strict=True):
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != _ZIP_TIMESTAMP
                    or info.extra
                    or info.comment
                    or info.create_system != 3
                    or info.external_attr != _ZIP_EXTERNAL_ATTRIBUTES
                ):
                    raise IntegrityError(
                        "model parameter NPZ member metadata is non-canonical",
                        details={"member_name": info.filename},
                    )
                if info.file_size != record["bytes"]:
                    raise IntegrityError(
                        "model parameter NPZ member size does not match its record",
                        details={"member_name": info.filename},
                    )
                member = archive.read(info)
                if _sha256_bytes(member) != record["sha256"]:
                    raise IntegrityError(
                        "model parameter NPZ member hash does not match its record",
                        details={"member_name": info.filename},
                    )
                array = _load_npy_member(member, record)
                arrays[record["name"]] = array
    except IntegrityError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise IntegrityError(
            "model parameter NPZ cannot be read safely",
            details={"error_type": type(error).__name__},
        ) from error
    canonical_content, canonical_records = _npz_bytes(arrays)
    if canonical_content != content or canonical_records != _mutable_json(records):
        raise IntegrityError("model parameter NPZ bytes or records are not canonical")
    return arrays


def _load_npy_member(content: bytes, record: Mapping[str, Any]) -> np.ndarray[Any, Any]:
    try:
        value = np.load(io.BytesIO(content), allow_pickle=False)
    except (OSError, TypeError, ValueError) as error:
        raise IntegrityError(
            "model parameter NPY member cannot be loaded without pickle",
            details={
                "member_name": record["member_name"],
                "error_type": type(error).__name__,
            },
        ) from error
    if not isinstance(value, np.ndarray):
        raise IntegrityError("model parameter NPY member is not an array")
    if value.dtype.str != record["dtype"] or tuple(value.shape) != tuple(
        record["shape"]
    ):
        raise IntegrityError(
            "model parameter NPY dtype or shape does not match its record",
            details={"member_name": record["member_name"]},
        )
    if value.dtype.fields is not None or value.dtype.kind not in _SUPPORTED_DTYPE_KINDS:
        raise IntegrityError(
            "model parameter NPY dtype is unsafe",
            details={"member_name": record["member_name"]},
        )
    if value.dtype.kind == "f" and not bool(np.isfinite(value).all()):
        raise IntegrityError(
            "model parameter NPY contains a non-finite value",
            details={"member_name": record["member_name"]},
        )
    canonical = np.ascontiguousarray(value)
    if not value.flags.c_contiguous or _npy_bytes(canonical) != content:
        raise IntegrityError(
            "model parameter NPY member bytes are non-canonical",
            details={"member_name": record["member_name"]},
        )
    # ``setflags(write=False)`` on an owning ndarray can be reversed by the
    # caller.  Rebuild over an immutable ``bytes`` owner so attempting to set
    # WRITEABLE=True fails at the NumPy boundary.
    immutable_data = canonical.tobytes(order="C")
    immutable = np.frombuffer(
        immutable_data, dtype=canonical.dtype, count=canonical.size
    ).reshape(canonical.shape)
    return immutable


def _validate_identity(identity: dict[str, Any]) -> None:
    _require_exact_keys(identity, _IDENTITY_KEYS, label="model artifact identity")
    if identity.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise IntegrityError("model artifact identity schema is unsupported")
    _validate_role(identity.get("artifact_role"), error_type=IntegrityError)
    _validate_fingerprint(
        identity.get("data_fingerprint"),
        label="data fingerprint",
        error_type=IntegrityError,
    )
    _normalize_json_object(
        identity.get("model_metadata"),
        label="model metadata",
        error_type=IntegrityError,
        require_nonempty=True,
    )
    _normalize_json_object(
        identity.get("provenance"),
        label="model provenance",
        error_type=IntegrityError,
        require_nonempty=True,
    )

    parameters = identity.get("parameters")
    if not isinstance(parameters, dict):
        raise IntegrityError("model parameter record is missing or invalid")
    _require_exact_keys(parameters, _PARAMETER_KEYS, label="model parameter record")
    if (
        parameters.get("relative_path") != MODEL_PARAMETERS_NAME
        or parameters.get("format") != NPZ_FORMAT_VERSION
        or parameters.get("allow_pickle") is not False
    ):
        raise IntegrityError("model parameter record format fields are invalid")
    _validate_positive_size(parameters.get("bytes"), label="model parameter file")
    _validate_sha256(parameters.get("sha256"), label="model parameter file")
    array_records = parameters.get("arrays")
    if not isinstance(array_records, list) or not array_records:
        raise IntegrityError("model parameter array records are empty or invalid")
    names: list[str] = []
    for record in array_records:
        _validate_array_record(record)
        names.append(record["name"])
    if names != sorted(names) or len(names) != len(set(names)):
        raise IntegrityError("model parameter array records are unsorted or duplicated")

    documents = identity.get("documents")
    if not isinstance(documents, list):
        raise IntegrityError("model calibration document records are invalid")
    paths: list[str] = []
    for record in documents:
        if not isinstance(record, dict):
            raise IntegrityError("model calibration document record is invalid")
        _require_exact_keys(
            record, _DOCUMENT_RECORD_KEYS, label="model calibration document record"
        )
        path = record.get("relative_path")
        if not isinstance(path, str):
            raise IntegrityError("model calibration document path is invalid")
        _validate_document_path(path, error_type=IntegrityError)
        if record.get("media_type") != "application/json":
            raise IntegrityError("model calibration document media type is invalid")
        _validate_positive_size(record.get("bytes"), label="model calibration document")
        _validate_sha256(record.get("sha256"), label="model calibration document")
        paths.append(path)
    _validate_document_paths(tuple(paths), error_type=IntegrityError)
    if paths != sorted(paths):
        raise IntegrityError("model calibration document records are unsorted")


def _validate_array_record(value: Any) -> None:
    if not isinstance(value, dict):
        raise IntegrityError("model parameter array record is invalid")
    _require_exact_keys(value, _ARRAY_RECORD_KEYS, label="model parameter array record")
    name = value.get("name")
    if not isinstance(name, str) or not _ARRAY_NAME_PATTERN.fullmatch(name):
        raise IntegrityError("model parameter array record name is invalid")
    if value.get("member_name") != f"{name}.npy":
        raise IntegrityError("model parameter array member name is invalid")
    dtype_text = value.get("dtype")
    if not isinstance(dtype_text, str):
        raise IntegrityError("model parameter array dtype is invalid")
    try:
        dtype = np.dtype(dtype_text)
    except TypeError as error:
        raise IntegrityError("model parameter array dtype is invalid") from error
    if (
        dtype.str != dtype_text
        or dtype.fields is not None
        or dtype.kind not in _SUPPORTED_DTYPE_KINDS
        or (dtype.itemsize > 1 and dtype.byteorder not in {"<", "="})
    ):
        raise IntegrityError(
            "model parameter array dtype is unsupported or noncanonical"
        )
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item < 0 for item in shape)
        or math.prod(shape) <= 0
    ):
        raise IntegrityError("model parameter array shape is invalid")
    if value.get("order") != "C":
        raise IntegrityError("model parameter array order is invalid")
    _validate_positive_size(value.get("bytes"), label="model parameter NPY member")
    _validate_sha256(value.get("sha256"), label="model parameter NPY member")


def _normalize_json_object(
    value: Any,
    *,
    label: str,
    error_type: ErrorType,
    require_nonempty: bool,
) -> dict[str, Any]:
    normalized = _normalize_json(value, label=label, error_type=error_type)
    if not isinstance(normalized, dict) or (require_nonempty and not normalized):
        raise error_type(f"{label} must be a non-empty JSON object")
    if _contains_secret(normalized):
        raise error_type(f"{label} contains a secret-like key or value")
    return normalized


def _normalize_json(value: Any, *, label: str, error_type: ErrorType) -> Any:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except Exception as error:
            raise error_type(
                f"{label} model value cannot be converted to JSON",
                details={"error_type": type(error).__name__},
            ) from error
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise error_type(f"{label} contains a non-string object key")
        return {
            key: _normalize_json(item, label=label, error_type=error_type)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list | tuple):
        return [
            _normalize_json(item, label=label, error_type=error_type) for item in value
        ]
    if isinstance(value, Enum):
        return _normalize_json(value.value, label=label, error_type=error_type)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{label} contains a non-finite number")
        return value
    raise error_type(
        f"{label} contains an unsupported value",
        details={"value_type": type(value).__name__},
    )


def _contains_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            compact = normalized.replace("_", "")
            if any(
                marker in normalized or marker.replace("_", "") in compact
                for marker in _SECRET_KEY_MARKERS
            ):
                return True
            if _contains_secret(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    elif isinstance(value, str):
        return any(
            pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS
        )
    return False


def _validate_document_paths(paths: tuple[Any, ...], *, error_type: ErrorType) -> None:
    normalized: list[str] = []
    for value in paths:
        if not isinstance(value, str):
            raise error_type("model calibration document path is invalid")
        _validate_document_path(value, error_type=error_type)
        normalized.append(value)
    if len(normalized) != len(set(normalized)) or len(normalized) != len(
        {path.casefold() for path in normalized}
    ):
        raise error_type("model calibration document paths are duplicated")
    all_files = {MODEL_MANIFEST_NAME, MODEL_PARAMETERS_NAME, *normalized}
    for path in all_files:
        prefix = f"{path}/"
        if any(other.startswith(prefix) for other in all_files if other != path):
            raise error_type(
                "model artifact file paths have a file/directory collision",
                details={"relative_path": path},
            )


def _validate_document_path(value: str, *, error_type: ErrorType) -> None:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or value in {MODEL_MANIFEST_NAME, MODEL_PARAMETERS_NAME}
        or pure.suffix != ".json"
        or "\\" in value
        or any(
            part in {"", ".", ".."} or not _PATH_COMPONENT_PATTERN.fullmatch(part)
            for part in pure.parts
        )
    ):
        raise error_type(
            "model calibration document path is unsafe or invalid",
            details={"relative_path": value},
        )


def _validate_role(value: Any, *, error_type: ErrorType) -> None:
    if not isinstance(value, str) or value not in _ARTIFACT_ROLES:
        raise error_type(
            "model artifact role is invalid",
            details={"artifact_role": value},
        )


def _validate_fingerprint(value: Any, *, label: str, error_type: ErrorType) -> None:
    if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise error_type(f"{label} is invalid", details={"fingerprint": value})


def _validate_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise IntegrityError(f"{label} SHA-256 is invalid")


def _validate_positive_size(value: Any, *, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise IntegrityError(f"{label} byte size is invalid")


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise IntegrityError(
            f"{label} fields are invalid",
            details={
                "unexpected": sorted(actual - expected),
                "missing": sorted(expected - actual),
            },
        )


def _verify_file_record(
    content: bytes,
    record: Mapping[str, Any],
    *,
    label: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if len(content) != record["bytes"] or _sha256_bytes(content) != record["sha256"]:
        raise IntegrityError(f"{label} integrity check failed", details=details)


def _parse_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            f"{label} cannot be read",
            details={"error_type": type(error).__name__},
        ) from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} root is not an object")
    return value


def _verify_json_document(content: bytes, relative_path: str) -> dict[str, Any]:
    value = _parse_json_object(content, label="model calibration document")
    normalized = _normalize_json_object(
        value,
        label="model calibration document",
        error_type=IntegrityError,
        require_nonempty=True,
    )
    if content != _canonical_json_bytes(normalized):
        raise IntegrityError(
            "model calibration document is not canonical JSON",
            details={"relative_path": relative_path},
        )
    return normalized


def _safe_artifact_path(root: Path, relative_path: str, *, must_exist: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise IntegrityError(
            "model artifact relative path is unsafe",
            details={"relative_path": relative_path},
        )
    root_resolved = root.resolve()
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise IntegrityError(
            "model artifact path cannot be resolved",
            details={
                "relative_path": relative_path,
                "error_type": type(error).__name__,
            },
        ) from error
    if not resolved.is_relative_to(root_resolved):
        raise IntegrityError(
            "model artifact relative path escapes its root",
            details={"relative_path": relative_path},
        )
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise IntegrityError(
                "model artifact path contains a symbolic link",
                details={"relative_path": relative_path},
            )
    return candidate


def _read_regular_file(
    root: Path,
    relative_path: str,
    *,
    label: str,
    details: Mapping[str, Any] | None = None,
) -> bytes:
    """Read one confined regular file through no-follow directory descriptors."""

    _reject_symlink_ancestors(root, error_type=IntegrityError)
    _safe_artifact_path(root, relative_path, must_exist=False)
    parts = PurePosixPath(relative_path).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise IntegrityError(
                f"{label} is not a private regular file", details=details
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_state != after_state or len(content) != after.st_size:
            raise IntegrityError(f"{label} changed while it was read", details=details)
        return content
    except IntegrityError:
        raise
    except OSError as error:
        missing_or_unsafe = error.errno in {
            errno.ENOENT,
            errno.ENOTDIR,
            errno.ELOOP,
        }
        raise IntegrityError(
            (
                f"{label} is missing or unsafe"
                if missing_or_unsafe
                else f"{label} cannot be read"
            ),
            details={**dict(details or {}), "error_type": type(error).__name__},
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _reject_symlink_ancestors(path: Path, *, error_type: ErrorType) -> None:
    """Reject every existing symlink component from the filesystem root down."""

    absolute = path.absolute()
    chain = [absolute, *absolute.parents]
    for candidate in reversed(chain):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise error_type(
                "model artifact path ancestors cannot be inspected",
                details={
                    "path": str(candidate),
                    "error_type": type(error).__name__,
                },
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise error_type(
                "model artifact path contains a symbolic-link ancestor",
                details={"path": str(candidate)},
            )


def _verify_tree_allowlist(root: Path, expected_files: set[str]) -> None:
    expected_directories: set[str] = set()
    for relative_path in expected_files:
        parent = PurePosixPath(relative_path).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    unsafe: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            unsafe.append(relative)
        elif path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            unsafe.append(relative)
    if (
        unsafe
        or actual_files != expected_files
        or actual_directories != expected_directories
    ):
        raise IntegrityError(
            "model artifact tree does not match its allowlist",
            details={
                "unsafe": sorted(unsafe),
                "unexpected_files": sorted(actual_files - expected_files),
                "missing_files": sorted(expected_files - actual_files),
                "unexpected_directories": sorted(
                    actual_directories - expected_directories
                ),
                "missing_directories": sorted(
                    expected_directories - actual_directories
                ),
            },
        )


def _npy_bytes(array: np.ndarray[Any, Any]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _mutable_json(value: Any) -> Any:
    """Return ordinary dict/list containers suitable for canonical encoding."""

    if isinstance(value, Mapping):
        return {str(key): _mutable_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_mutable_json(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                _mutable_json(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ModelError(
            "model artifact metadata cannot be encoded canonically",
            details={"error_type": type(error).__name__},
        ) from error


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
