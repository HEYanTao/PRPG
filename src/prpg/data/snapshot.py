"""Immutable, content-addressed storage for provider payloads.

The snapshot identity is deliberately narrower than the audit manifest.  It is
derived from the frozen acquisition contract, provider request semantics, and
payload content hashes.  Retrieval/attempt timestamps are retained for audit
but cannot create a second identity for identical scientific inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from prpg.data.providers import ProviderPayload
from prpg.errors import DataValidationError, IntegrityError

SNAPSHOT_SCHEMA_VERSION = 1
MANIFEST_NAME = "snapshot-manifest.json"
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_TIMESTAMP_KEYS = {
    "created_at",
    "created_utc",
    "finished_at",
    "finished_utc",
    "retrieval_timestamp",
    "retrieved_at",
    "retrieved_utc",
    "started_at",
    "started_utc",
    "timestamp",
}


@dataclass(frozen=True, slots=True)
class SnapshotReference:
    """A verified local snapshot location."""

    fingerprint: str
    path: Path
    manifest: Mapping[str, Any]
    reused: bool


class SnapshotStore:
    """Publish and verify snapshots below ``artifact_root/data/sha256``."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.store_root = self.artifact_root / "data" / "sha256"
        self._now = now or (lambda: datetime.now(UTC))

    def create(
        self,
        payloads: Iterable[ProviderPayload],
        *,
        acquisition_contract: Mapping[str, Any],
        acquisition_group_id: str,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> SnapshotReference:
        """Atomically publish payloads, or safely reuse an identical snapshot.

        Existing content is never replaced.  A directory at the expected
        fingerprint is reusable only after full offline verification and an
        exact identity comparison.
        """

        ordered_payloads = self._validated_payloads(payloads)
        contract = _identity_value(acquisition_contract)
        identity = self._identity(ordered_payloads, contract)
        fingerprint = _sha256_bytes(_canonical_json_bytes(identity))
        self.store_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self.store_root / fingerprint

        if destination.exists():
            return self._verified_reuse(destination, fingerprint, identity)

        stage = Path(
            tempfile.mkdtemp(prefix=f".stage-{fingerprint[:12]}-", dir=self.store_root)
        )
        try:
            entries: list[dict[str, Any]] = []
            for payload in ordered_payloads:
                relative_path = self._payload_relative_path(payload)
                target = _safe_join(stage, relative_path, must_exist=False)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                _write_new_file(target, payload.raw_bytes)
                entry = payload.manifest_record(relative_path=relative_path)
                # Do not trust a provider object to override integrity fields.
                entry["sha256"] = _sha256_bytes(payload.raw_bytes)
                entry["bytes"] = len(payload.raw_bytes)
                entries.append(_manifest_value(entry))

            manifest: dict[str, Any] = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "snapshot_fingerprint": fingerprint,
                "created_utc": self._now().astimezone(UTC).isoformat(),
                "acquisition_group_id": acquisition_group_id,
                "identity": identity,
                "audit_metadata": _manifest_value(dict(audit_metadata or {})),
                "files": entries,
            }
            _write_new_file(stage / MANIFEST_NAME, _canonical_json_bytes(manifest))
            _fsync_directory(stage)

            try:
                os.rename(stage, destination)
                _fsync_directory(self.store_root)
            except FileExistsError:
                # Another coordinator won the same content-addressed race.
                return self._verified_reuse(destination, fingerprint, identity)
            except OSError:
                # POSIX may report ENOTEMPTY/EEXIST as a generic OSError.
                if destination.exists():
                    return self._verified_reuse(destination, fingerprint, identity)
                raise

            return self.verify(fingerprint, reused=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def verify(self, fingerprint: str, *, reused: bool = True) -> SnapshotReference:
        """Verify manifest identity, allowlisted paths, sizes, and SHA-256s."""

        snapshot_path = self._snapshot_path(fingerprint)
        if not snapshot_path.is_dir() or snapshot_path.is_symlink():
            raise IntegrityError(
                "snapshot directory is missing or unsafe",
                details={"fingerprint": fingerprint},
            )
        manifest_path = snapshot_path / MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise IntegrityError(
                "snapshot manifest is missing or unsafe",
                details={"fingerprint": fingerprint},
            )
        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "snapshot manifest cannot be read",
                details={
                    "fingerprint": fingerprint,
                    "error_type": type(error).__name__,
                },
            ) from error
        if not isinstance(manifest_value, dict):
            raise IntegrityError(
                "snapshot manifest root is not an object",
                details={"fingerprint": fingerprint},
            )
        manifest: dict[str, Any] = manifest_value
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise IntegrityError(
                "snapshot manifest schema is unsupported",
                details={"fingerprint": fingerprint},
            )
        if manifest.get("snapshot_fingerprint") != fingerprint:
            raise IntegrityError(
                "snapshot manifest fingerprint does not match its directory",
                details={"fingerprint": fingerprint},
            )
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise IntegrityError(
                "snapshot identity is missing",
                details={"fingerprint": fingerprint},
            )
        computed = _sha256_bytes(_canonical_json_bytes(_identity_value(identity)))
        if computed != fingerprint:
            raise IntegrityError(
                "snapshot identity hash mismatch",
                details={"fingerprint": fingerprint, "computed": computed},
            )

        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise IntegrityError(
                "snapshot file list is empty or invalid",
                details={"fingerprint": fingerprint},
            )
        allowed = {MANIFEST_NAME}
        seen: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise IntegrityError(
                    "snapshot file record is invalid",
                    details={"fingerprint": fingerprint},
                )
            relative_path = item.get("relative_path")
            expected_hash = item.get("sha256")
            expected_size = item.get("bytes")
            if not isinstance(relative_path, str) or relative_path in seen:
                raise IntegrityError(
                    "snapshot contains an invalid or duplicate relative path",
                    details={"fingerprint": fingerprint},
                )
            seen.add(relative_path)
            candidate = _safe_join(snapshot_path, relative_path, must_exist=True)
            if not candidate.is_file() or candidate.is_symlink():
                raise IntegrityError(
                    "snapshot payload is missing or unsafe",
                    details={
                        "fingerprint": fingerprint,
                        "relative_path": relative_path,
                    },
                )
            actual_size = candidate.stat().st_size
            actual_hash = _sha256_file(candidate)
            if expected_size != actual_size or expected_hash != actual_hash:
                raise IntegrityError(
                    "snapshot payload integrity check failed",
                    details={
                        "fingerprint": fingerprint,
                        "relative_path": relative_path,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                        "expected_bytes": expected_size,
                        "actual_bytes": actual_size,
                    },
                )
            allowed.add(relative_path)

        actual_files = {
            path.relative_to(snapshot_path).as_posix()
            for path in snapshot_path.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_files != allowed:
            raise IntegrityError(
                "snapshot contains unmanifested or missing files",
                details={
                    "fingerprint": fingerprint,
                    "unexpected": sorted(actual_files - allowed),
                    "missing": sorted(allowed - actual_files),
                },
            )
        return SnapshotReference(
            fingerprint=fingerprint,
            path=snapshot_path,
            manifest=manifest,
            reused=reused,
        )

    def read_payload(self, fingerprint: str, provider: str, identifier: str) -> bytes:
        """Verify a snapshot and return one exact provider payload offline."""

        reference = self.verify(fingerprint)
        matches = [
            entry
            for entry in reference.manifest["files"]
            if entry.get("provider") == provider
            and entry.get("identifier") == identifier
        ]
        if len(matches) != 1:
            raise DataValidationError(
                "snapshot payload lookup is not unique",
                details={
                    "fingerprint": fingerprint,
                    "provider": provider,
                    "identifier": identifier,
                    "matches": len(matches),
                },
            )
        relative_path = matches[0]["relative_path"]
        return _safe_join(reference.path, relative_path, must_exist=True).read_bytes()

    def _verified_reuse(
        self,
        path: Path,
        fingerprint: str,
        expected_identity: Mapping[str, Any],
    ) -> SnapshotReference:
        reference = self.verify(fingerprint, reused=True)
        actual_identity = _canonical_json_bytes(reference.manifest["identity"])
        if actual_identity != _canonical_json_bytes(expected_identity):
            raise IntegrityError(
                "immutable snapshot fingerprint collision",
                details={"fingerprint": fingerprint, "path": str(path)},
            )
        return reference

    def _snapshot_path(self, fingerprint: str) -> Path:
        if not _FINGERPRINT_PATTERN.fullmatch(fingerprint):
            raise IntegrityError(
                "snapshot fingerprint is invalid",
                details={"fingerprint": fingerprint},
            )
        return self.store_root / fingerprint

    @staticmethod
    def _validated_payloads(
        payloads: Iterable[ProviderPayload],
    ) -> tuple[ProviderPayload, ...]:
        values = tuple(payloads)
        if not values:
            raise DataValidationError("a snapshot requires at least one payload")
        keys: set[tuple[str, str]] = set()
        for payload in values:
            _safe_component(payload.provider, "provider")
            _safe_component(payload.identifier, "identifier")
            key = (payload.provider, payload.identifier)
            if key in keys:
                raise DataValidationError(
                    "snapshot contains a duplicate provider identifier",
                    details={"provider": key[0], "identifier": key[1]},
                )
            keys.add(key)
            if not payload.raw_bytes:
                raise DataValidationError(
                    "snapshot payload is empty",
                    details={"provider": key[0], "identifier": key[1]},
                )
        return tuple(sorted(values, key=lambda item: (item.provider, item.identifier)))

    @staticmethod
    def _payload_relative_path(payload: ProviderPayload) -> str:
        provider = _safe_component(payload.provider, "provider")
        identifier = _safe_component(payload.identifier, "identifier")
        return PurePosixPath("raw", provider, f"{identifier}.csv").as_posix()

    @staticmethod
    def _identity(
        payloads: tuple[ProviderPayload, ...], contract: Any
    ) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "acquisition_contract": contract,
            "payloads": [
                {
                    "provider": payload.provider,
                    "identifier": payload.identifier,
                    "request": _identity_value(payload.request),
                    "library_version": payload.library_version,
                    "raw_kind": payload.raw_kind,
                    "sha256": _sha256_bytes(payload.raw_bytes),
                    "bytes": len(payload.raw_bytes),
                }
                for payload in payloads
            ],
        }


def _safe_component(value: str, label: str) -> str:
    if not _SAFE_COMPONENT_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise DataValidationError(
            f"snapshot {label} is not a safe path component",
            details={label: value},
        )
    return value


def _safe_join(root: Path, relative_path: str, *, must_exist: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative_path
    ):
        raise IntegrityError(
            "snapshot relative path is unsafe",
            details={"relative_path": relative_path},
        )
    root_resolved = root.resolve()
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as error:
        raise IntegrityError(
            "snapshot path cannot be resolved",
            details={
                "relative_path": relative_path,
                "error_type": type(error).__name__,
            },
        ) from error
    if not resolved.is_relative_to(root_resolved):
        raise IntegrityError(
            "snapshot relative path escapes its root",
            details={"relative_path": relative_path},
        )
    if must_exist:
        current = root_resolved
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise IntegrityError(
                    "snapshot path contains a symbolic link",
                    details={"relative_path": relative_path},
                )
    return candidate


def _identity_value(value: Any) -> Any:
    """Return canonical identity data while removing observational timestamps."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _TIMESTAMP_KEYS
        }
    if isinstance(value, list | tuple):
        return [_identity_value(item) for item in value]
    if isinstance(value, set):
        normalized = [_identity_value(item) for item in value]
        return sorted(normalized, key=lambda item: _canonical_json_bytes(item))
    if isinstance(value, Enum):
        return _identity_value(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataValidationError("snapshot identity contains a non-finite number")
        return value
    raise DataValidationError(
        "snapshot identity contains an unsupported value",
        details={"value_type": type(value).__name__},
    )


def _manifest_value(value: Any) -> Any:
    """Convert audit metadata to strict JSON without dropping timestamps."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple | set):
        return [_manifest_value(item) for item in value]
    if isinstance(value, Enum):
        return _manifest_value(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataValidationError("snapshot manifest contains a non-finite number")
        return value
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            "snapshot metadata cannot be encoded canonically",
            details={"error_type": type(error).__name__},
        ) from error
    return (text + "\n").encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # fdopen owns the descriptor once constructed.
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
