"""Copy one completed G3 result into a clean downstream runtime.

This is a deliberately small operational boundary.  It copies only the
completed run ledger, its 26 checkpoints, the design and production model
artifacts, the scientific pair receipt, and the durable G3 evidence manifest.
The existing G3 evidence loader verifies the complete scientific result both
before and after the copy; this module adds no new scientific gate or
generation authority.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_checkpoint import (
    CHECKPOINT_STORAGE_DIRECTORY,
    CalibrationCheckpointStore,
)
from prpg.model.g3_evidence_store import (
    G3_EVIDENCE_DIRECTORY,
    G3EvidenceStore,
    LoadedG3EvidenceAuthority,
)
from prpg.model.run_store import CalibrationRunStore
from prpg.model.scientific_artifact_pair import (
    SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY,
)

_SHA256_CHUNK_BYTES: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class G3ArtifactHandoffReceipt:
    """Concise identity and copy-size receipt for one verified handoff."""

    run_id: str
    evidence_fingerprint: str
    completion_fingerprint: str
    design_artifact_fingerprint: str
    production_artifact_fingerprint: str
    pair_receipt_fingerprint: str
    checkpoint_count: int
    copied_file_count: int
    copied_bytes: int


def handoff_completed_g3(
    source_runtime_root: str | Path,
    destination_runtime_root: str | Path,
    *,
    evidence_fingerprint: str,
) -> G3ArtifactHandoffReceipt:
    """Copy one passed G3 result into a new, otherwise absent runtime.

    The conventional runtime layout is ``artifacts/`` plus ``runs/`` under
    each supplied root.  The destination must not exist: an existing file,
    directory, or symbolic link is never reused or overwritten.
    """

    source = Path(source_runtime_root).absolute()
    destination = Path(destination_runtime_root).absolute()
    _require_source_root(source)
    if destination.exists() or destination.is_symlink():
        raise ModelError("G3 handoff destination already exists; overwrite refused")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlink_components(destination.parent, "G3 handoff destination parent")
    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.parent.resolve(strict=True) / destination.name
    if source_resolved == destination_resolved or source_resolved in (
        destination_resolved.parents
    ):
        raise ModelError("G3 handoff destination cannot be inside its source runtime")

    source_loaded = _load_completed_runtime(source, evidence_fingerprint)
    relative_roots = _required_relative_roots(source_loaded)
    source_inventory = _inventory(source, relative_roots)

    destination.mkdir(mode=0o700)
    try:
        for relative_root in relative_roots:
            source_tree = source / relative_root
            destination_tree = destination / relative_root
            destination_tree.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copytree(
                source_tree,
                destination_tree,
                symlinks=True,
                copy_function=shutil.copyfile,
            )

        if _inventory(source, relative_roots) != source_inventory:
            raise IntegrityError("G3 handoff source changed while it was copied")
        destination_inventory = _inventory(destination, (Path("."),))
        if destination_inventory != source_inventory:
            raise IntegrityError("G3 handoff destination bytes differ from source")

        destination_loaded = _load_completed_runtime(destination, evidence_fingerprint)
        if _loaded_identity(destination_loaded) != _loaded_identity(source_loaded):
            raise IntegrityError("G3 handoff changed the loaded scientific identity")
        if _inventory(destination, (Path("."),)) != source_inventory:
            raise IntegrityError("G3 handoff bytes changed during destination loading")
    except Exception:
        _remove_created_destination(destination)
        raise

    return G3ArtifactHandoffReceipt(
        run_id=source_loaded.run_id,
        evidence_fingerprint=source_loaded.reference.fingerprint,
        completion_fingerprint=source_loaded.completion_fingerprint,
        design_artifact_fingerprint=(
            source_loaded.design_artifact.reference.fingerprint
        ),
        production_artifact_fingerprint=(
            source_loaded.production_artifact.reference.fingerprint
        ),
        pair_receipt_fingerprint=source_loaded.pair_receipt_fingerprint,
        checkpoint_count=len(source_loaded.checkpoint_fingerprints),
        copied_file_count=len(source_inventory),
        copied_bytes=sum(size for size, _ in source_inventory.values()),
    )


def _load_completed_runtime(
    runtime_root: Path, evidence_fingerprint: str
) -> LoadedG3EvidenceAuthority:
    artifact_root = runtime_root / "artifacts"
    return G3EvidenceStore(artifact_root).load(
        evidence_fingerprint,
        run_store=CalibrationRunStore(runtime_root / "runs"),
        checkpoint_store=CalibrationCheckpointStore(artifact_root),
        model_store=ModelArtifactStore(artifact_root),
    )


def _required_relative_roots(
    loaded: LoadedG3EvidenceAuthority,
) -> tuple[Path, ...]:
    checkpoint_fingerprints = sorted(set(loaded.checkpoint_fingerprints.values()))
    model_fingerprints = sorted(
        {
            loaded.design_artifact.reference.fingerprint,
            loaded.production_artifact.reference.fingerprint,
        }
    )
    roots = [
        Path("runs") / "calibration" / loaded.run_id,
        Path("artifacts")
        / G3_EVIDENCE_DIRECTORY
        / "sha256"
        / loaded.reference.fingerprint,
        Path("artifacts")
        / SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY
        / "sha256"
        / loaded.pair_receipt_fingerprint,
    ]
    roots.extend(
        Path("artifacts") / "model" / "sha256" / fingerprint
        for fingerprint in model_fingerprints
    )
    roots.extend(
        Path("artifacts")
        / CHECKPOINT_STORAGE_DIRECTORY
        / "model"
        / "sha256"
        / fingerprint
        for fingerprint in checkpoint_fingerprints
    )
    return tuple(roots)


def _loaded_identity(loaded: LoadedG3EvidenceAuthority) -> tuple[object, ...]:
    return (
        loaded.reference.fingerprint,
        loaded.evidence_bundle.fingerprint,
        loaded.run_id,
        loaded.completion_fingerprint,
        tuple(loaded.task_result_fingerprints.items()),
        tuple(loaded.checkpoint_fingerprints.items()),
        tuple(loaded.planned_look_fingerprints.items()),
        loaded.design_artifact.reference.fingerprint,
        loaded.production_artifact.reference.fingerprint,
        loaded.pair_receipt_fingerprint,
        loaded.g3_source_code_fingerprint,
        loaded.g3_dependency_lock_fingerprint,
    )


def _inventory(
    runtime_root: Path, relative_roots: tuple[Path, ...]
) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    for relative_root in relative_roots:
        tree_root = runtime_root / relative_root
        _walk_tree(runtime_root, tree_root, records)
    return records


def _walk_tree(
    runtime_root: Path,
    directory: Path,
    records: dict[str, tuple[int, str]],
) -> None:
    try:
        information = directory.lstat()
    except OSError as error:
        raise IntegrityError("required G3 handoff directory is missing") from error
    if not stat.S_ISDIR(information.st_mode) or stat.S_ISLNK(information.st_mode):
        raise IntegrityError("required G3 handoff path is not a private directory")
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise IntegrityError("required G3 handoff directory cannot be read") from error
    for entry in entries:
        try:
            entry_information = entry.lstat()
        except OSError as error:
            raise IntegrityError("G3 handoff tree changed while inspected") from error
        if stat.S_ISDIR(entry_information.st_mode):
            _walk_tree(runtime_root, entry, records)
            continue
        if not stat.S_ISREG(entry_information.st_mode):
            raise IntegrityError("G3 handoff tree contains a symbolic or special file")
        content = _read_private_regular(entry)
        relative_path = entry.relative_to(runtime_root).as_posix()
        if relative_path in records:
            raise IntegrityError("G3 handoff file selection overlaps")
        records[relative_path] = (len(content), hashlib.sha256(content).hexdigest())


def _read_private_regular(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise IntegrityError("G3 handoff file is missing") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise IntegrityError("G3 handoff file is not private regular storage")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError("G3 handoff file cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _SHA256_CHUNK_BYTES):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise IntegrityError("G3 handoff file disappeared while read") from error
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(opened, field) != getattr(after, field)
        or getattr(after, field) != getattr(current, field)
        for field in fields
    ):
        raise IntegrityError("G3 handoff file changed while read")
    return b"".join(chunks)


def _require_source_root(source: Path) -> None:
    _reject_symlink_components(source, "G3 handoff source runtime")
    if not source.is_dir() or source.is_symlink():
        raise IntegrityError("G3 handoff source runtime is missing or unsafe")


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            information = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise IntegrityError(f"{label} cannot be inspected") from error
        if stat.S_ISLNK(information.st_mode):
            raise IntegrityError(f"{label} contains a symbolic link")


def _remove_created_destination(destination: Path) -> None:
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
