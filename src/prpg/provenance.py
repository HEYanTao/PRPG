"""Race-resistant executable-source and dependency-lock provenance.

Scientific receipts must identify the bytes that executed, not merely the
installed library versions.  This module hashes the complete ``src/prpg``
package plus the build/seed-registry inputs and binds the dependency lock
separately.  Tests, documentation, caches, and workspace location are not part
of the executable identity.

Canonical G3 uses a narrower, explicitly ordered execution closure.  That
closure is intentionally separate from the generic whole-package identity so
later G4--G8 work cannot stale an already frozen G3 result.  The manifest is
itself identity-bearing: changing path membership or order changes the G3
source fingerprint even when every file byte is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from prpg.errors import IntegrityError, ModelError

EXECUTABLE_SOURCE_SCHEMA: Final = "prpg-executable-source-v1"
G3_EXECUTION_CLOSURE_SCHEMA: Final = "prpg-g3-execution-closure-v2"
G3_EXECUTION_CLOSURE_MANIFEST_SCHEMA: Final = "prpg-g3-execution-closure-manifest-v2"
G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT: Final = (
    "f5e0afa8e444b6a4aefaf4bbc33bd022926ee8c2865121a8b7c6e3da1253e436"
)
DEPENDENCY_LOCK_PATH: Final = "requirements.lock"
_STATIC_SOURCE_PATHS: Final = (
    ".python-version",
    "pyproject.toml",
    "configs/seed-registry.yaml",
)
_READ_CHUNK_BYTES: Final = 1 << 20
_LOCKED_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"=="
    r"(?P<version>[A-Za-z0-9](?:[A-Za-z0-9.!+_-]*[A-Za-z0-9])?)\Z"
)

G3_EXECUTION_MODULE_PATHS: Final = (
    "src/prpg/__init__.py",
    "src/prpg/config.py",
    "src/prpg/data/__init__.py",
    "src/prpg/data/processed.py",
    "src/prpg/data/providers.py",
    "src/prpg/data/snapshot.py",
    "src/prpg/data/transform.py",
    "src/prpg/errors.py",
    "src/prpg/execution.py",
    "src/prpg/g3_launch.py",
    "src/prpg/model/__init__.py",
    "src/prpg/model/adequacy.py",
    "src/prpg/model/adequacy_execution.py",
    "src/prpg/model/adequacy_gate.py",
    "src/prpg/model/alpha.py",
    "src/prpg/model/artifact.py",
    "src/prpg/model/benchmarks.py",
    "src/prpg/model/block_execution.py",
    "src/prpg/model/block_execution_registered.py",
    "src/prpg/model/block_length.py",
    "src/prpg/model/bootstrap.py",
    "src/prpg/model/bridge.py",
    "src/prpg/model/bridge_driver.py",
    "src/prpg/model/bridge_fixed_k.py",
    "src/prpg/model/bridge_history.py",
    "src/prpg/model/bridge_selection.py",
    "src/prpg/model/calibration.py",
    "src/prpg/model/calibration_identity.py",
    "src/prpg/model/candidate_fixed_point_execution.py",
    "src/prpg/model/candidate_gate.py",
    "src/prpg/model/candidate_suite.py",
    "src/prpg/model/dependence.py",
    "src/prpg/model/dependence_execution.py",
    "src/prpg/model/dependence_gate_execution.py",
    "src/prpg/model/fixed_point.py",
    "src/prpg/model/g3.py",
    "src/prpg/model/g3_assembly.py",
    "src/prpg/model/g3_boundary_views.py",
    "src/prpg/model/g3_calibration_views.py",
    "src/prpg/model/g3_checkpoint.py",
    "src/prpg/model/g3_evidence_store.py",
    "src/prpg/model/g3_exact_v2_finalization.py",
    "src/prpg/model/g3_planned_look_execution.py",
    "src/prpg/model/g3_preflight.py",
    "src/prpg/model/g3_registry.py",
    "src/prpg/model/g3_rehearsal.py",
    "src/prpg/model/g3_resource_projection.py",
    "src/prpg/model/g3_science_handlers.py",
    "src/prpg/model/g3_success_bridge_stage.py",
    "src/prpg/model/g3_success_execution.py",
    "src/prpg/model/g3_success_production_stage.py",
    "src/prpg/model/g3_success_runner.py",
    "src/prpg/model/g3_task_binding.py",
    "src/prpg/model/g3_task_publication.py",
    "src/prpg/model/hmm.py",
    "src/prpg/model/inference.py",
    "src/prpg/model/input.py",
    "src/prpg/model/kernel_execution.py",
    "src/prpg/model/materialization_identity.py",
    "src/prpg/model/ppw_reference.py",
    "src/prpg/model/production_fit_execution.py",
    "src/prpg/model/refit_adequacy.py",
    "src/prpg/model/refit_adequacy_execution.py",
    "src/prpg/model/regime_policy.py",
    "src/prpg/model/registered_fixed_point.py",
    "src/prpg/model/registered_holdout_execution.py",
    "src/prpg/model/run_store.py",
    "src/prpg/model/scientific_artifact.py",
    "src/prpg/model/scientific_artifact_pair.py",
    "src/prpg/model/scientific_policy.py",
    "src/prpg/model/selection.py",
    "src/prpg/model/stability.py",
    "src/prpg/model/transition_bootstrap.py",
    "src/prpg/provenance.py",
    "src/prpg/simulation/rng.py",
)

G3_EXECUTION_RESOURCE_PATHS: Final = (
    ".python-version",
    "configs/seed-registry.yaml",
    "scripts/run_v3_g3_rehearsal.py",
    "src/prpg/vendor/__init__.py",
    "src/prpg/vendor/patton_politis_white/LICENSE",
    "src/prpg/vendor/patton_politis_white/SOURCE.json",
    "src/prpg/vendor/patton_politis_white/__init__.py",
    "src/prpg/vendor/patton_politis_white/opt_block_length_REV_dec07.txt",
)


@dataclass(frozen=True, slots=True)
class ExecutableSourceFile:
    """One relative regular-file record in canonical path order."""

    relative_path: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ExecutableSourceIdentity:
    """Exact source/lock identities used by preflight and artifact provenance."""

    schema_id: str
    source_files: tuple[ExecutableSourceFile, ...]
    source_code_fingerprint: str
    dependency_lock_relative_path: str
    dependency_lock_bytes: int
    dependency_lock_fingerprint: str

    def source_identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "source_files": [item.as_dict() for item in self.source_files],
        }


@dataclass(frozen=True, slots=True)
class G3ExecutionClosureManifest:
    """The ordered files that may influence the dedicated canonical G3 path."""

    schema_id: str
    module_paths: tuple[str, ...]
    resource_static_paths: tuple[str, ...]
    dependency_lock_path: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "module_paths": list(self.module_paths),
            "resource_static_paths": list(self.resource_static_paths),
            "dependency_lock_path": self.dependency_lock_path,
        }


G3_EXECUTION_CLOSURE_MANIFEST: Final = G3ExecutionClosureManifest(
    schema_id=G3_EXECUTION_CLOSURE_MANIFEST_SCHEMA,
    module_paths=G3_EXECUTION_MODULE_PATHS,
    resource_static_paths=G3_EXECUTION_RESOURCE_PATHS,
    dependency_lock_path=DEPENDENCY_LOCK_PATH,
)


@dataclass(frozen=True, slots=True)
class G3ExecutionClosureIdentity:
    """Exact manifest, file records, and lock used by canonical G3."""

    schema_id: str
    manifest: G3ExecutionClosureManifest
    manifest_fingerprint: str
    source_files: tuple[ExecutableSourceFile, ...]
    source_code_fingerprint: str
    dependency_lock_relative_path: str
    dependency_lock_bytes: int
    dependency_lock_fingerprint: str
    dependency_versions: tuple[tuple[str, str], ...]

    def source_identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "manifest": self.manifest.as_dict(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "source_files": [item.as_dict() for item in self.source_files],
        }


def inspect_executable_source(
    project_root: str | Path | None = None,
) -> ExecutableSourceIdentity:
    """Hash the exact executable/build bytes without mutating the workspace."""

    root = _project_root(project_root)
    package_root = root / "src" / "prpg"
    if not package_root.is_dir() or package_root.is_symlink():
        raise IntegrityError("PRPG package source directory is missing or unsafe")

    relative_paths = list(_STATIC_SOURCE_PATHS)
    for path in package_root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise IntegrityError(
                "executable source tree contains a symbolic link",
                details={"relative_path": relative.as_posix()},
            )
        if path.is_file():
            relative_paths.append(relative.as_posix())

    if len(relative_paths) != len(set(relative_paths)):
        raise IntegrityError("executable source allowlist contains duplicate paths")
    records: list[ExecutableSourceFile] = []
    for relative_path in sorted(relative_paths):
        content = _read_stable_regular(root, relative_path)
        records.append(
            ExecutableSourceFile(
                relative_path,
                len(content),
                hashlib.sha256(content).hexdigest(),
            )
        )
    if not records:
        raise IntegrityError("executable source allowlist is empty")
    provisional = {
        "schema_id": EXECUTABLE_SOURCE_SCHEMA,
        "source_files": [item.as_dict() for item in records],
    }
    source_fingerprint = hashlib.sha256(_canonical_bytes(provisional)).hexdigest()
    lock_bytes = _read_stable_regular(root, DEPENDENCY_LOCK_PATH)
    return ExecutableSourceIdentity(
        EXECUTABLE_SOURCE_SCHEMA,
        tuple(records),
        source_fingerprint,
        DEPENDENCY_LOCK_PATH,
        len(lock_bytes),
        hashlib.sha256(lock_bytes).hexdigest(),
    )


def inspect_g3_execution_closure(
    project_root: str | Path | None = None,
) -> G3ExecutionClosureIdentity:
    """Hash only the frozen, ordered execution closure for canonical G3."""

    root = _project_root(project_root)
    manifest = G3_EXECUTION_CLOSURE_MANIFEST
    relative_paths = manifest.module_paths + manifest.resource_static_paths
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise IntegrityError("G3 execution-closure manifest is empty or duplicated")
    records = tuple(
        _source_file_record(root, relative_path) for relative_path in relative_paths
    )
    manifest_fingerprint = hashlib.sha256(
        _canonical_bytes(manifest.as_dict())
    ).hexdigest()
    if manifest_fingerprint != G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT:
        raise IntegrityError("G3 execution-closure manifest fingerprint changed")
    provisional = {
        "schema_id": G3_EXECUTION_CLOSURE_SCHEMA,
        "manifest": manifest.as_dict(),
        "manifest_fingerprint": manifest_fingerprint,
        "source_files": [item.as_dict() for item in records],
    }
    lock_bytes = _read_stable_regular(root, manifest.dependency_lock_path)
    dependency_versions = parse_dependency_lock_versions(lock_bytes)
    return G3ExecutionClosureIdentity(
        schema_id=G3_EXECUTION_CLOSURE_SCHEMA,
        manifest=manifest,
        manifest_fingerprint=manifest_fingerprint,
        source_files=records,
        source_code_fingerprint=hashlib.sha256(
            _canonical_bytes(provisional)
        ).hexdigest(),
        dependency_lock_relative_path=manifest.dependency_lock_path,
        dependency_lock_bytes=len(lock_bytes),
        dependency_lock_fingerprint=hashlib.sha256(lock_bytes).hexdigest(),
        dependency_versions=dependency_versions,
    )


def inspect_dependency_lock_versions(
    project_root: str | Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Read and strictly parse the exact project dependency lock."""

    root = _project_root(project_root)
    return parse_dependency_lock_versions(
        _read_stable_regular(root, DEPENDENCY_LOCK_PATH)
    )


def parse_dependency_lock_versions(content: bytes) -> tuple[tuple[str, str], ...]:
    """Return the complete ordered ``name==version`` inventory.

    Canonical G3 accepts only blank lines, whole-line comments, and exact pins.
    Names are normalized with the standard ``[-_.]+ -> -`` rule so spelling
    variants cannot conceal duplicate distributions.
    """

    if not isinstance(content, bytes):
        raise ModelError("dependency lock content must be bytes")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IntegrityError("dependency lock is not UTF-8") from error
    versions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if line != line.strip():
            raise IntegrityError(
                "dependency lock entry must be an exact name==version pin",
                details={"line": line_number},
            )
        match = _LOCKED_REQUIREMENT.fullmatch(line)
        if match is None:
            raise IntegrityError(
                "dependency lock entry must be an exact name==version pin",
                details={"line": line_number},
            )
        name = re.sub(r"[-_.]+", "-", match.group("name")).lower()
        version_value = match.group("version")
        if name in seen:
            raise IntegrityError(
                "dependency lock contains a duplicate distribution",
                details={"distribution": name, "line": line_number},
            )
        seen.add(name)
        versions.append((name, version_value))
    if not versions:
        raise IntegrityError("dependency lock contains no exact pins")
    return tuple(versions)


def verify_executable_source_identity(
    value: ExecutableSourceIdentity,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Require byte-for-byte equality with a fresh source-tree inspection."""

    if not isinstance(value, ExecutableSourceIdentity):
        raise ModelError("executable source identity has the wrong type")
    observed = inspect_executable_source(project_root)
    if value != observed:
        raise IntegrityError("executable source or dependency lock changed")


def verify_g3_execution_closure_identity(
    value: G3ExecutionClosureIdentity,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Require a fresh byte-for-byte match to the scoped G3 closure."""

    if not isinstance(value, G3ExecutionClosureIdentity):
        raise ModelError("G3 execution-closure identity has the wrong type")
    observed = inspect_g3_execution_closure(project_root)
    if value != observed:
        raise IntegrityError("G3 execution closure or dependency lock changed")


def _source_file_record(root: Path, relative_path: str) -> ExecutableSourceFile:
    content = _read_stable_regular(root, relative_path)
    return ExecutableSourceFile(
        relative_path=relative_path,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _project_root(value: str | Path | None) -> Path:
    if value is None:
        root = Path(__file__).resolve().parents[2]
        # In an installed wheel, ``__file__`` is below site-packages rather
        # than the delivered project root. Commands are documented to run from
        # that root, which contains the lock and complete source tree.
        if not (root / DEPENDENCY_LOCK_PATH).is_file():
            working_root = Path.cwd().resolve()
            if (
                (working_root / DEPENDENCY_LOCK_PATH).is_file()
                and (working_root / "pyproject.toml").is_file()
                and (working_root / "src" / "prpg").is_dir()
            ):
                root = working_root
    else:
        root = Path(value).absolute()
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError("project root is missing or unsafe")
    return root


def _read_stable_regular(root: Path, relative_path: str) -> bytes:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise IntegrityError("executable source relative path is unsafe")
    path = root / relative
    try:
        before = path.lstat()
    except OSError as error:
        raise IntegrityError(
            "executable source file is missing",
            details={"relative_path": relative.as_posix()},
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise IntegrityError(
            "executable source file is not a private regular file",
            details={"relative_path": relative.as_posix()},
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError(
            "executable source file cannot be opened safely",
            details={"relative_path": relative.as_posix()},
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise IntegrityError("executable source changed before it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise IntegrityError("executable source disappeared while reading") from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(opened, name) != getattr(after, name)
        or getattr(after, name) != getattr(current, name)
        for name in stable_fields
    ):
        raise IntegrityError(
            "executable source changed while reading",
            details={"relative_path": relative.as_posix()},
        )
    content = b"".join(chunks)
    if len(content) != after.st_size:
        raise IntegrityError("executable source byte count changed while reading")
    return content


def _canonical_bytes(value: object) -> bytes:
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
