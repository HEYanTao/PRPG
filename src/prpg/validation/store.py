"""Small durable no-overwrite stores for canonical G5 JSON records."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from prpg.errors import IntegrityError, ValidationError
from prpg.validation.records import (
    CanonicalThresholds,
    CanonicalValidationResults,
    canonical_record_bytes,
    canonical_results_from_dict,
    canonical_thresholds_from_dict,
)

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
ArtifactKind = Literal["thresholds", "results"]


@dataclass(frozen=True, slots=True)
class StoredValidationRecord:
    """Filesystem location of one newly published immutable JSON record."""

    kind: ArtifactKind
    fingerprint: str
    path: Path
    bytes: int


class ValidationArtifactStore:
    """Publish and revalidate content-addressed threshold/result JSON files.

    A same-directory temporary file is flushed and fsynced before publication
    by a no-replace hard link.  Existing destinations are never overwritten or
    silently reused.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "validation"

    def publish_thresholds(self, value: CanonicalThresholds) -> StoredValidationRecord:
        """Publish one new canonical threshold record."""

        if not isinstance(value, CanonicalThresholds):
            raise ValidationError("threshold store requires canonical thresholds")
        return self._publish(
            "thresholds", value.fingerprint, canonical_record_bytes(value)
        )

    def publish_results(
        self, value: CanonicalValidationResults
    ) -> StoredValidationRecord:
        """Publish one new canonical validation-result record."""

        if not isinstance(value, CanonicalValidationResults):
            raise ValidationError("result store requires canonical validation results")
        return self._publish(
            "results", value.fingerprint, canonical_record_bytes(value)
        )

    def load_thresholds(self, fingerprint: str) -> CanonicalThresholds:
        """Load and fully revalidate one canonical threshold record."""

        raw = self._load("thresholds", fingerprint)
        try:
            rebuilt = canonical_thresholds_from_dict(_json_object(raw))
        except ValidationError as error:
            raise IntegrityError("stored threshold content is invalid") from error
        if rebuilt.fingerprint != fingerprint or canonical_record_bytes(rebuilt) != raw:
            raise IntegrityError("stored threshold bytes are noncanonical")
        return rebuilt

    def load_results(self, fingerprint: str) -> CanonicalValidationResults:
        """Load and fully revalidate one canonical validation-result record."""

        raw = self._load("results", fingerprint)
        try:
            rebuilt = canonical_results_from_dict(_json_object(raw))
        except ValidationError as error:
            raise IntegrityError(
                "stored validation-result content is invalid"
            ) from error
        if rebuilt.fingerprint != fingerprint or canonical_record_bytes(rebuilt) != raw:
            raise IntegrityError("stored validation-result bytes are noncanonical")
        return rebuilt

    def _publish(
        self,
        kind: ArtifactKind,
        fingerprint: str,
        content: bytes,
    ) -> StoredValidationRecord:
        checked = _required_fingerprint(fingerprint)
        directory = self._directory(kind, create=True)
        destination = directory / f"{checked}.json"
        if destination.exists() or destination.is_symlink():
            raise IntegrityError(
                "validation artifact already exists; overwrite refused"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checked}.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                written = handle.write(content)
                if written != len(content):
                    raise IntegrityError("validation artifact write was incomplete")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise IntegrityError(
                    "validation artifact already exists; overwrite refused"
                ) from error
            _fsync_directory(directory)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        return StoredValidationRecord(kind, checked, destination, len(content))

    def _load(self, kind: ArtifactKind, fingerprint: str) -> bytes:
        checked = _required_fingerprint(fingerprint)
        path = self._directory(kind, create=False) / f"{checked}.json"
        try:
            metadata = path.lstat()
        except OSError as error:
            raise IntegrityError("validation artifact is missing") from error
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise IntegrityError("validation artifact is not a regular file")
        try:
            return path.read_bytes()
        except OSError as error:
            raise IntegrityError("validation artifact cannot be read") from error

    def _directory(self, kind: ArtifactKind, *, create: bool) -> Path:
        directory = self.root / kind
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory.is_dir() or directory.is_symlink():
            raise IntegrityError("validation artifact directory is missing or unsafe")
        return directory


def _required_fingerprint(value: object) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValidationError("validation artifact fingerprint is invalid")
    return value


def _json_object(content: bytes) -> dict[str, object]:
    try:
        value: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("validation artifact is not valid JSON") from error
    if not isinstance(value, dict):
        raise IntegrityError("validation artifact root must be a JSON object")
    return cast(dict[str, object], value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
