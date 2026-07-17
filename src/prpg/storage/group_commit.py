"""Serial LF/HF sibling group commit and durable receipt verification.

A work unit is complete only when all of its frequency siblings exist and a
single canonical receipt binds their bytes, ranges, source lineage, worker,
duration, and the caller's simulation/materialization/toolchain/run identities.
All siblings are written and verified in one same-filesystem stage tree before
the first canonical publication.  Valid existing groups are never reused by
the commit API.  Orphaned or invalid partial groups can be moved as a whole to
an operational quarantine below the same root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, TypeGuard, TypeVar, cast

from prpg.errors import GenerationError, IntegrityError
from prpg.simulation.aggregation import (
    HFWeeklyReturns,
    LFAggregatedReturns,
    is_sealed_hf_weekly_returns,
    is_sealed_lf_aggregated_returns,
)
from prpg.simulation.serial import SerialBasePath, is_sealed_serial_base_path
from prpg.storage.csv_v1 import (
    CSV_HEADER_BYTES,
    CSV_SCHEMA_VERSION,
    CSV_SERIALIZER_VERSION,
    CSVFamily,
    CSVFrequency,
    CSVShardReceipt,
    _atomic_publish_no_replace,
    _fsync_directory,
    _path_id,
    _prepare_output_root,
    _prepare_relative_directory,
    _receipt_identity,
    _sha256_json,
    format_csv_float64,
    is_sealed_csv_shard_receipt,
    write_serial_csv_shard,
)

GROUP_RECEIPT_SCHEMA_VERSION: Final = 1
GROUP_RECEIPT_SCHEMA_ID: Final = "prpg-serial-csv-work-unit-v1"
GROUP_RECEIPT_DIRECTORY: Final = PurePosixPath("receipts", "storage")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_WORK_UNIT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_WORKER_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}\Z")
_STAGE_SUFFIX = re.compile(r"[a-z0-9_]{8}\Z")
_RECEIPT_SEAL = object()

WorkUnitFamily: TypeAlias = Literal["LF", "HF"]
_SourceT = TypeVar("_SourceT")


@dataclass(frozen=True, slots=True)
class StorageCommitBinding:
    """Validated external identities inherited by one storage work unit."""

    simulation_fingerprint: str
    materialization_fingerprint: str
    toolchain_fingerprint: str
    run_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "simulation_fingerprint",
            "materialization_fingerprint",
            "toolchain_fingerprint",
            "run_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
                raise GenerationError(f"storage binding {name} is invalid")

    def as_dict(self) -> dict[str, str]:
        return {
            "simulation_fingerprint": self.simulation_fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "run_fingerprint": self.run_fingerprint,
        }


@dataclass(frozen=True, slots=True, init=False)
class LoadedWorkUnitReceipt:
    """A canonical durable receipt whose complete sibling set was reverified."""

    fingerprint: str
    path: Path
    identity: Mapping[str, Any]
    work_unit_id: str
    family: WorkUnitFamily
    first_path_entity: int
    last_path_entity: int
    worker_identity: str
    duration_monotonic_ns: int
    sibling_paths: tuple[str, ...]
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class QuarantinedWorkUnit:
    """Operational record of one safely moved orphaned/invalid sibling set."""

    work_unit_id: str
    quarantine_relative_path: str
    moved_relative_paths: tuple[str, ...]


def serial_work_unit_id(
    family: WorkUnitFamily,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
) -> str:
    """Return the one registered ID for an exact serial work-unit scope."""

    checked_family = _family(family)
    index = _nonnegative_integer(shard_index, "shard index", maximum=99_999)
    first, last = _scope_range(first_path_entity, last_path_entity)
    identifier = (
        f"{checked_family.lower()}-{index:05d}-"
        f"{_path_id(checked_family, first).lower()}-"
        f"{_path_id(checked_family, last).lower()}"
    )
    if _WORK_UNIT_ID.fullmatch(identifier) is None:  # pragma: no cover - bounds prove
        raise AssertionError("canonical work-unit ID violates its registry")
    return identifier


def commit_lf_serial_work_unit(
    output_root: str | Path,
    *,
    binding: StorageCommitBinding,
    work_unit_id: str,
    worker_identity: str,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
    expected_path_count: int,
    base_paths: Sequence[SerialBasePath],
    aggregates: Sequence[LFAggregatedReturns],
) -> LoadedWorkUnitReceipt:
    """Commit monthly/quarterly/annual siblings as one LF work unit.

    ``worker_identity`` is a stable serial-executor audit label.  The receipt's
    ``duration_monotonic_ns`` is measured internally, in nanoseconds, from this
    API entry through sibling publication and pre-receipt verification.
    """

    started_ns = time.monotonic_ns()
    bases, derived, years = _validated_lf_sources(
        first_path_entity=first_path_entity,
        last_path_entity=last_path_entity,
        expected_path_count=expected_path_count,
        base_paths=base_paths,
        aggregates=aggregates,
    )
    return _commit_serial_work_unit(
        output_root,
        binding=binding,
        work_unit_id=work_unit_id,
        worker_identity=worker_identity,
        started_ns=started_ns,
        family="LF",
        shard_index=shard_index,
        first_path_entity=first_path_entity,
        last_path_entity=last_path_entity,
        expected_path_count=expected_path_count,
        years=years,
        bases=bases,
        derived=derived,
    )


def commit_hf_serial_work_unit(
    output_root: str | Path,
    *,
    binding: StorageCommitBinding,
    work_unit_id: str,
    worker_identity: str,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
    expected_path_count: int,
    base_paths: Sequence[SerialBasePath],
    weekly_returns: Sequence[HFWeeklyReturns],
) -> LoadedWorkUnitReceipt:
    """Commit daily/weekly siblings as one HF work unit.

    ``worker_identity`` is a stable serial-executor audit label.  Duration is
    measured internally with ``time.monotonic_ns`` and bound into the receipt.
    """

    started_ns = time.monotonic_ns()
    bases, derived, years = _validated_hf_sources(
        first_path_entity=first_path_entity,
        last_path_entity=last_path_entity,
        expected_path_count=expected_path_count,
        base_paths=base_paths,
        weekly_returns=weekly_returns,
    )
    return _commit_serial_work_unit(
        output_root,
        binding=binding,
        work_unit_id=work_unit_id,
        worker_identity=worker_identity,
        started_ns=started_ns,
        family="HF",
        shard_index=shard_index,
        first_path_entity=first_path_entity,
        last_path_entity=last_path_entity,
        expected_path_count=expected_path_count,
        years=years,
        bases=bases,
        derived=derived,
    )


def verify_serial_work_unit(
    output_root: str | Path,
    *,
    binding: StorageCommitBinding,
    work_unit_id: str,
) -> LoadedWorkUnitReceipt:
    """Fail closed unless the durable receipt and every sibling reverify."""

    root = _existing_safe_root(output_root)
    checked_binding = _binding(binding)
    identifier = _work_unit_id(work_unit_id)
    receipt_path = root / _receipt_relative_path(identifier)
    return _load_and_verify_receipt(
        root,
        receipt_path,
        expected_binding=checked_binding,
    )


def is_verified_work_unit_receipt(
    value: object,
) -> TypeGuard[LoadedWorkUnitReceipt]:
    """Return whether ``value`` is the original loaded receipt instance."""

    try:
        _verify_loaded_receipt(value)
    except (AttributeError, GenerationError, IntegrityError, TypeError, ValueError):
        return False
    return True


def quarantine_serial_work_unit(
    output_root: str | Path,
    *,
    work_unit_id: str,
    family: WorkUnitFamily,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
) -> QuarantinedWorkUnit:
    """Move every known regular single-link sibling/receipt out of canonical paths."""

    root = _existing_safe_root(output_root)
    identifier = _work_unit_id(work_unit_id)
    checked_family = _family(family)
    index = _nonnegative_integer(shard_index, "shard index", maximum=99_999)
    first, last = _scope_range(first_path_entity, last_path_entity)
    _require_canonical_work_unit_id(identifier, checked_family, index, first, last)
    candidates = [
        *(
            root / relative
            for relative in _sibling_relative_paths(checked_family, index, first, last)
        ),
        root / _receipt_relative_path(identifier),
    ]
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if not existing:
        raise IntegrityError("work unit has no canonical files to quarantine")
    for path in existing:
        _require_regular_single_link(path, "quarantine source")

    parent = _prepare_relative_directory(root, ("quarantine", "storage", identifier))
    quarantine = _new_quarantine_directory(parent)
    moved: list[str] = []
    try:
        for source in existing:
            relative = source.relative_to(root)
            destination_parent = quarantine / relative.parent
            destination_parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            destination = destination_parent / relative.name
            if destination.exists() or destination.is_symlink():
                raise IntegrityError("quarantine destination already exists")
            os.rename(source, destination)
            _fsync_directory(source.parent)
            _fsync_directory(destination.parent)
            moved.append(relative.as_posix())
    except OSError as error:
        raise IntegrityError("work-unit quarantine move failed") from error
    return QuarantinedWorkUnit(
        work_unit_id=identifier,
        quarantine_relative_path=quarantine.relative_to(root).as_posix(),
        moved_relative_paths=tuple(sorted(moved)),
    )


def _commit_serial_work_unit(
    output_root: str | Path,
    *,
    binding: StorageCommitBinding,
    work_unit_id: str,
    worker_identity: str,
    started_ns: int,
    family: WorkUnitFamily,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
    expected_path_count: int,
    years: int,
    bases: tuple[SerialBasePath, ...],
    derived: tuple[LFAggregatedReturns, ...] | tuple[HFWeeklyReturns, ...],
) -> LoadedWorkUnitReceipt:
    checked_binding = _binding(binding)
    identifier = _work_unit_id(work_unit_id)
    worker = _worker_identity(worker_identity)
    started = _nonnegative_integer_unbounded(started_ns, "work-unit start monotonic ns")
    checked_family = _family(family)
    index = _nonnegative_integer(shard_index, "shard index", maximum=99_999)
    first, last = _scope_range(first_path_entity, last_path_entity)
    _require_canonical_work_unit_id(identifier, checked_family, index, first, last)
    count = _positive_integer(expected_path_count, "expected path count")
    if count != last - first + 1:
        raise GenerationError("work-unit expected count differs from its path range")
    root = _prepare_output_root(output_root)
    receipt_path = root / _receipt_relative_path(identifier)
    sibling_relatives = _sibling_relative_paths(checked_family, index, first, last)
    siblings = [root / relative for relative in sibling_relatives]

    if receipt_path.exists() or receipt_path.is_symlink():
        try:
            _load_and_verify_receipt(root, receipt_path, expected_binding=None)
        except IntegrityError:
            _quarantine_scope(root, identifier, checked_family, index, first, last)
            raise
        raise GenerationError(
            "valid work-unit receipt already exists; reuse is forbidden"
        )
    _reconcile_stale_stage_roots(
        root,
        work_unit_id=identifier,
        sibling_relative_paths=sibling_relatives,
    )
    if any(path.exists() or path.is_symlink() for path in siblings):
        _quarantine_scope(root, identifier, checked_family, index, first, last)
        raise IntegrityError(
            "canonical siblings existed without a valid work-unit receipt"
        )

    stage_path = _create_stage_root(root, identifier)
    stage_root: Path | None = stage_path
    csv_receipts: list[CSVShardReceipt] = []
    try:
        for frequency, source in _frequency_sources(checked_family, bases, derived):
            receipt = write_serial_csv_shard(
                stage_path,
                frequency=frequency,
                shard_index=index,
                first_path_entity=first,
                last_path_entity=last,
                values=source,
            )
            if not is_sealed_csv_shard_receipt(receipt):
                raise IntegrityError("CSV writer returned an unsealed sibling receipt")
            csv_receipts.append(receipt)
        if tuple(item.relative_path for item in csv_receipts) != tuple(
            sibling_relatives
        ):
            raise IntegrityError("CSV sibling order/path differs from work-unit scope")
        for receipt in csv_receipts:
            _verify_csv_sibling(stage_path, receipt)
        _publish_staged_siblings(root, stage_path, tuple(csv_receipts))
        _discard_stage_root(stage_path)
        stage_root = None
        for receipt in csv_receipts:
            _verify_csv_sibling(root, receipt)
        duration = _duration_monotonic_ns(started)
        identity = _group_identity(
            binding=checked_binding,
            work_unit_id=identifier,
            worker_identity=worker,
            duration_monotonic_ns=duration,
            family=checked_family,
            shard_index=index,
            first_path_entity=first,
            last_path_entity=last,
            expected_path_count=count,
            years=years,
            bases=bases,
            derived=derived,
            receipts=tuple(csv_receipts),
        )
        _publish_group_receipt(receipt_path, identity)
        return _load_and_verify_receipt(
            root,
            receipt_path,
            expected_binding=checked_binding,
        )
    except BaseException:
        if stage_root is not None:
            _discard_stage_root(stage_root)
        if any(
            path.exists() or path.is_symlink() for path in (*siblings, receipt_path)
        ):
            _quarantine_scope(root, identifier, checked_family, index, first, last)
        raise


def _validated_lf_sources(
    *,
    first_path_entity: int,
    last_path_entity: int,
    expected_path_count: int,
    base_paths: Sequence[SerialBasePath],
    aggregates: Sequence[LFAggregatedReturns],
) -> tuple[tuple[SerialBasePath, ...], tuple[LFAggregatedReturns, ...], int]:
    bases = _source_tuple(base_paths, "LF base paths")
    derived = _source_tuple(aggregates, "LF aggregates")
    first, last = _scope_range(first_path_entity, last_path_entity)
    count = _positive_integer(expected_path_count, "expected path count")
    _source_lengths(count, first, last, bases, derived)
    years: int | None = None
    for entity, base, aggregate in zip(
        range(first, last + 1), bases, derived, strict=True
    ):
        if (
            not is_sealed_serial_base_path(base)
            or base.family != "LF"
            or base.base_frequency != "monthly"
            or not is_sealed_lf_aggregated_returns(aggregate)
            or base.path_entity != entity
            or aggregate.path_entity != entity
            or aggregate.base_path_fingerprint != base.fingerprint
            or aggregate.years != base.years
        ):
            raise IntegrityError("LF work-unit source binding is invalid")
        years = _common_years(years, base.years)
    if years is None:  # pragma: no cover - positive count invariant
        raise AssertionError("LF source validation produced no horizon")
    return bases, derived, years


def _validated_hf_sources(
    *,
    first_path_entity: int,
    last_path_entity: int,
    expected_path_count: int,
    base_paths: Sequence[SerialBasePath],
    weekly_returns: Sequence[HFWeeklyReturns],
) -> tuple[tuple[SerialBasePath, ...], tuple[HFWeeklyReturns, ...], int]:
    bases = _source_tuple(base_paths, "HF base paths")
    derived = _source_tuple(weekly_returns, "HF weekly returns")
    first, last = _scope_range(first_path_entity, last_path_entity)
    count = _positive_integer(expected_path_count, "expected path count")
    _source_lengths(count, first, last, bases, derived)
    years: int | None = None
    for entity, base, weekly in zip(
        range(first, last + 1), bases, derived, strict=True
    ):
        if (
            not is_sealed_serial_base_path(base)
            or base.family != "HF"
            or base.base_frequency != "daily"
            or not is_sealed_hf_weekly_returns(weekly)
            or base.path_entity != entity
            or weekly.path_entity != entity
            or weekly.base_path_fingerprint != base.fingerprint
            or weekly.years != base.years
        ):
            raise IntegrityError("HF work-unit source binding is invalid")
        years = _common_years(years, base.years)
    if years is None:  # pragma: no cover - positive count invariant
        raise AssertionError("HF source validation produced no horizon")
    return bases, derived, years


def _frequency_sources(
    family: WorkUnitFamily,
    bases: tuple[SerialBasePath, ...],
    derived: tuple[LFAggregatedReturns, ...] | tuple[HFWeeklyReturns, ...],
) -> tuple[tuple[CSVFrequency, Sequence[Any]], ...]:
    if family == "LF":
        return (
            ("monthly", bases),
            ("quarterly", derived),
            ("annual", derived),
        )
    return (("daily", bases), ("weekly", derived))


def _create_stage_root(root: Path, work_unit_id: str) -> Path:
    try:
        path = Path(tempfile.mkdtemp(prefix=f".prpg-stage-{work_unit_id}-", dir=root))
        path.chmod(0o700)
        _fsync_directory(root)
    except OSError as error:
        raise IntegrityError("work-unit staging directory cannot be created") from error
    if path.is_symlink() or not path.is_dir():  # pragma: no cover - mkdtemp invariant
        raise IntegrityError("work-unit staging directory is unsafe")
    return path


def _publish_staged_siblings(
    root: Path,
    stage_root: Path,
    receipts: tuple[CSVShardReceipt, ...],
) -> None:
    publications: list[tuple[Path, Path]] = []
    for receipt in receipts:
        relative = PurePosixPath(receipt.relative_path)
        parent = _prepare_relative_directory(root, relative.parent.parts)
        source = stage_root / relative
        destination = parent / relative.name
        _require_regular_single_link(source, "staged CSV sibling")
        if destination.exists() or destination.is_symlink():
            raise IntegrityError("CSV sibling destination appeared before publication")
        publications.append((source, destination))
    for source, destination in publications:
        _atomic_publish_no_replace(source, destination)


def _discard_stage_root(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError("work-unit staging root changed before cleanup")
    for descendant in path.rglob("*"):
        if descendant.is_symlink():
            raise IntegrityError("work-unit staging tree contains a symbolic link")
    parent = path.parent
    try:
        shutil.rmtree(path)
    except OSError as error:
        raise IntegrityError("work-unit staging cleanup failed") from error
    _fsync_directory(parent)


def _reconcile_stale_stage_roots(
    root: Path,
    *,
    work_unit_id: str,
    sibling_relative_paths: tuple[str, ...],
) -> None:
    prefix = f".prpg-stage-{work_unit_id}-"
    try:
        candidates = tuple(
            sorted(
                (child for child in root.iterdir() if child.name.startswith(prefix)),
                key=lambda item: item.name,
            )
        )
    except OSError as error:
        raise IntegrityError("work-unit staging roots cannot be enumerated") from error
    expected = {relative: root / relative for relative in sibling_relative_paths}
    for stage_root in candidates:
        suffix = stage_root.name.removeprefix(prefix)
        if (
            _STAGE_SUFFIX.fullmatch(suffix) is None
            or stage_root.is_symlink()
            or not stage_root.is_dir()
        ):
            raise IntegrityError("stale work-unit staging root is unsafe")
        aliases: list[tuple[Path, Path]] = []
        try:
            descendants = tuple(stage_root.rglob("*"))
        except OSError as error:
            raise IntegrityError("stale staging tree cannot be enumerated") from error
        for descendant in descendants:
            try:
                metadata = descendant.lstat()
            except OSError as error:
                raise IntegrityError(
                    "stale staging entry cannot be inspected"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise IntegrityError("stale staging tree contains a symbolic link")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise IntegrityError("stale staging tree contains a non-regular file")
            relative = descendant.relative_to(stage_root).as_posix()
            canonical = expected.get(relative)
            if canonical is not None and (canonical.exists() or canonical.is_symlink()):
                if metadata.st_nlink != 2:
                    raise IntegrityError(
                        "stale staged sibling has an unexpected hard-link count"
                    )
                staged_identity = _linked_file_identity(
                    descendant, expected_links=2, label="stale staged sibling"
                )
                canonical_identity = _linked_file_identity(
                    canonical, expected_links=2, label="stale canonical sibling"
                )
                if staged_identity != canonical_identity:
                    raise IntegrityError(
                        "stale staged/canonical sibling ownership or content differs"
                    )
                aliases.append((descendant, canonical))
            elif metadata.st_nlink != 1:
                raise IntegrityError(
                    "stale staging file has an unknown hard-link alias"
                )
        for staged, canonical in aliases:
            try:
                staged.unlink()
            except OSError as error:
                raise IntegrityError(
                    "stale staged hard link cannot be removed"
                ) from error
            _fsync_directory(staged.parent)
            _require_regular_single_link(canonical, "reconciled canonical sibling")
        _discard_stage_root(stage_root)


def _linked_file_identity(
    path: Path,
    *,
    expected_links: int,
    label: str,
) -> tuple[int, int, int, str]:
    _reject_symlink_path(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError(f"{label} cannot be opened") from error
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            initial = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != expected_links:
                raise IntegrityError(f"{label} link identity is invalid")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            final = os.fstat(handle.fileno())
            try:
                current = path.stat(follow_symlinks=False)
            except OSError as error:
                raise IntegrityError(f"{label} path changed during read") from error
            if (
                final.st_dev != initial.st_dev
                or final.st_ino != initial.st_ino
                or final.st_size != initial.st_size
                or final.st_nlink != expected_links
                or current.st_dev != initial.st_dev
                or current.st_ino != initial.st_ino
                or current.st_size != initial.st_size
                or current.st_nlink != expected_links
            ):
                raise IntegrityError(f"{label} changed during read")
            return initial.st_dev, initial.st_ino, initial.st_size, digest.hexdigest()
    except OSError as error:
        raise IntegrityError(f"{label} cannot be read") from error


def _duration_monotonic_ns(started_ns: int) -> int:
    finished = time.monotonic_ns()
    if finished < started_ns:
        raise IntegrityError("monotonic clock moved backwards during work unit")
    return finished - started_ns


def _group_identity(
    *,
    binding: StorageCommitBinding,
    work_unit_id: str,
    worker_identity: str,
    duration_monotonic_ns: int,
    family: WorkUnitFamily,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
    expected_path_count: int,
    years: int,
    bases: tuple[SerialBasePath, ...],
    derived: tuple[LFAggregatedReturns, ...] | tuple[HFWeeklyReturns, ...],
    receipts: tuple[CSVShardReceipt, ...],
) -> dict[str, object]:
    return {
        "schema_id": GROUP_RECEIPT_SCHEMA_ID,
        "schema_version": GROUP_RECEIPT_SCHEMA_VERSION,
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "csv_serializer_version": CSV_SERIALIZER_VERSION,
        "binding": binding.as_dict(),
        "work_unit_id": work_unit_id,
        "worker_identity": worker_identity,
        "duration_monotonic_ns": duration_monotonic_ns,
        "family": family,
        "shard_index": shard_index,
        "first_path_entity": first_path_entity,
        "last_path_entity": last_path_entity,
        "expected_path_count": expected_path_count,
        "years": years,
        "base_path_fingerprints": [item.fingerprint for item in bases],
        "derived_fingerprints": [item.fingerprint for item in derived],
        "derived_base_path_fingerprints": [
            item.base_path_fingerprint for item in derived
        ],
        "siblings": [_sibling_identity(item) for item in receipts],
    }


def _sibling_identity(receipt: CSVShardReceipt) -> dict[str, object]:
    return {
        "frequency": receipt.frequency,
        "relative_path": receipt.relative_path,
        "rows": receipt.rows,
        "bytes": receipt.bytes,
        "sha256": receipt.sha256,
        "source_fingerprints": list(receipt.source_fingerprints),
        "csv_receipt_fingerprint": receipt.fingerprint,
    }


def _publish_group_receipt(path: Path, identity: Mapping[str, object]) -> None:
    fingerprint = _sha256_canonical(identity)
    document = {
        "schema_version": GROUP_RECEIPT_SCHEMA_VERSION,
        "receipt_fingerprint": fingerprint,
        "identity": identity,
    }
    content = _canonical_bytes(document)
    parent = _prepare_relative_directory(path.parents[2], GROUP_RECEIPT_DIRECTORY.parts)
    if parent != path.parent:  # pragma: no cover - fixed path invariant
        raise AssertionError("group receipt parent differs from canonical directory")
    if path.exists() or path.is_symlink():
        raise IntegrityError("group receipt destination already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    removed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            written = handle.write(content)
            if written != len(content):
                raise IntegrityError("group receipt temporary write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        _atomic_publish_no_replace(temporary, path)
    finally:
        try:
            temporary.unlink()
            removed = True
        except FileNotFoundError:
            pass
        if removed:
            _fsync_directory(parent)


def _load_and_verify_receipt(
    root: Path,
    path: Path,
    *,
    expected_binding: StorageCommitBinding | None,
) -> LoadedWorkUnitReceipt:
    content = _read_regular_single_link(path, "work-unit receipt")
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("work-unit receipt is invalid JSON") from error
    if not isinstance(document, dict) or content != _canonical_bytes(document):
        raise IntegrityError("work-unit receipt is not canonical JSON")
    if set(document) != {"schema_version", "receipt_fingerprint", "identity"}:
        raise IntegrityError("work-unit receipt document keys are invalid")
    if document["schema_version"] != GROUP_RECEIPT_SCHEMA_VERSION:
        raise IntegrityError("work-unit receipt schema is unsupported")
    identity = document["identity"]
    if not isinstance(identity, dict):
        raise IntegrityError("work-unit receipt identity is invalid")
    fingerprint = _required_fingerprint(
        document["receipt_fingerprint"], "work-unit receipt fingerprint"
    )
    if fingerprint != _sha256_canonical(identity):
        raise IntegrityError("work-unit receipt content hash is invalid")
    try:
        normalized = _validated_group_identity(identity)
        binding = StorageCommitBinding(**normalized["binding"])
    except (GenerationError, TypeError) as error:
        raise IntegrityError("work-unit receipt semantic fields are invalid") from error
    expected_path = root / _receipt_relative_path(normalized["work_unit_id"])
    if path != expected_path:
        raise IntegrityError("work-unit receipt path differs from its identity")
    if expected_binding is not None and binding != expected_binding:
        raise IntegrityError("work-unit receipt storage binding differs")
    _verify_group_siblings(root, normalized)
    frozen = _deep_freeze(normalized)
    if not isinstance(frozen, Mapping):  # pragma: no cover - mapping invariant
        raise AssertionError("frozen group identity is not a mapping")
    result = object.__new__(LoadedWorkUnitReceipt)
    for name, value in (
        ("fingerprint", fingerprint),
        ("path", path),
        ("identity", frozen),
        ("work_unit_id", normalized["work_unit_id"]),
        ("family", normalized["family"]),
        ("first_path_entity", normalized["first_path_entity"]),
        ("last_path_entity", normalized["last_path_entity"]),
        ("worker_identity", normalized["worker_identity"]),
        ("duration_monotonic_ns", normalized["duration_monotonic_ns"]),
        (
            "sibling_paths",
            tuple(item["relative_path"] for item in normalized["siblings"]),
        ),
        ("_seal", _RECEIPT_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    return result


def _verify_loaded_receipt(value: object) -> None:
    if type(value) is not LoadedWorkUnitReceipt:
        raise GenerationError("work-unit receipt was not loaded by the verifier")
    receipt = value
    if receipt._seal is not _RECEIPT_SEAL or receipt._owner_id != id(receipt):
        raise IntegrityError("loaded work-unit receipt process seal is invalid")
    if (
        not isinstance(receipt.fingerprint, str)
        or _FINGERPRINT.fullmatch(receipt.fingerprint) is None
        or not isinstance(receipt.identity, Mapping)
        or not isinstance(receipt.path, Path)
        or not receipt.path.is_absolute()
    ):
        raise IntegrityError("loaded work-unit receipt fields are invalid")
    thawed = _json_thaw(receipt.identity)
    if not isinstance(thawed, dict):  # pragma: no cover - mapping handled above
        raise IntegrityError("loaded work-unit identity is not a mapping")
    normalized = _validated_group_identity(thawed)
    if (
        receipt.fingerprint != _sha256_canonical(normalized)
        or receipt.identity != _deep_freeze(normalized)
        or receipt.work_unit_id != normalized["work_unit_id"]
        or receipt.family != normalized["family"]
        or receipt.first_path_entity != normalized["first_path_entity"]
        or receipt.last_path_entity != normalized["last_path_entity"]
        or receipt.worker_identity != normalized["worker_identity"]
        or receipt.duration_monotonic_ns != normalized["duration_monotonic_ns"]
        or receipt.sibling_paths
        != tuple(item["relative_path"] for item in normalized["siblings"])
        or receipt.path.parts[-3:]
        != ("receipts", "storage", f"{receipt.work_unit_id}.json")
    ):
        raise IntegrityError("loaded work-unit receipt identity fields differ")


def _validated_group_identity(value: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_id",
        "schema_version",
        "csv_schema_version",
        "csv_serializer_version",
        "binding",
        "work_unit_id",
        "worker_identity",
        "duration_monotonic_ns",
        "family",
        "shard_index",
        "first_path_entity",
        "last_path_entity",
        "expected_path_count",
        "years",
        "base_path_fingerprints",
        "derived_fingerprints",
        "derived_base_path_fingerprints",
        "siblings",
    }
    if set(value) != expected_keys:
        raise IntegrityError("work-unit receipt identity keys are invalid")
    if (
        value["schema_id"] != GROUP_RECEIPT_SCHEMA_ID
        or value["schema_version"] != GROUP_RECEIPT_SCHEMA_VERSION
        or value["csv_schema_version"] != CSV_SCHEMA_VERSION
        or value["csv_serializer_version"] != CSV_SERIALIZER_VERSION
    ):
        raise IntegrityError("work-unit receipt identity schema is unsupported")
    binding = value["binding"]
    if not isinstance(binding, dict) or set(binding) != {
        "simulation_fingerprint",
        "materialization_fingerprint",
        "toolchain_fingerprint",
        "run_fingerprint",
    }:
        raise IntegrityError("work-unit receipt binding is invalid")
    StorageCommitBinding(**binding)
    identifier = _work_unit_id(value["work_unit_id"])
    worker = _worker_identity(value["worker_identity"])
    duration = _nonnegative_integer_unbounded(
        value["duration_monotonic_ns"], "work-unit duration monotonic ns"
    )
    family = _family(value["family"])
    index = _nonnegative_integer(value["shard_index"], "shard index", maximum=99_999)
    first, last = _scope_range(value["first_path_entity"], value["last_path_entity"])
    _require_canonical_work_unit_id(identifier, family, index, first, last)
    count = _positive_integer(value["expected_path_count"], "expected path count")
    years = _positive_integer(value["years"], "years")
    if count != last - first + 1:
        raise IntegrityError("work-unit receipt path count/range differs")
    base = _fingerprint_list(value["base_path_fingerprints"], count, "base")
    derived = _fingerprint_list(value["derived_fingerprints"], count, "derived")
    derived_base = _fingerprint_list(
        value["derived_base_path_fingerprints"], count, "derived-base"
    )
    if derived_base != base:
        raise IntegrityError("work-unit derived/base lineage differs")
    siblings = value["siblings"]
    frequencies = _frequencies(family)
    if not isinstance(siblings, list) or len(siblings) != len(frequencies):
        raise IntegrityError("work-unit sibling receipt count is invalid")
    expected_paths = _sibling_relative_paths(family, index, first, last)
    normalized_siblings: list[dict[str, Any]] = []
    for position, (item, frequency, relative) in enumerate(
        zip(siblings, frequencies, expected_paths, strict=True)
    ):
        normalized_siblings.append(
            _validated_sibling_identity(
                item,
                frequency=frequency,
                relative_path=relative,
                family=family,
                shard_index=index,
                first=first,
                last=last,
                path_count=count,
                years=years,
                expected_sources=base if position == 0 else derived,
            )
        )
    return {
        **value,
        "work_unit_id": identifier,
        "worker_identity": worker,
        "duration_monotonic_ns": duration,
        "family": family,
        "shard_index": index,
        "first_path_entity": first,
        "last_path_entity": last,
        "expected_path_count": count,
        "years": years,
        "base_path_fingerprints": base,
        "derived_fingerprints": derived,
        "derived_base_path_fingerprints": derived_base,
        "siblings": normalized_siblings,
    }


def _validated_sibling_identity(
    value: object,
    *,
    frequency: CSVFrequency,
    relative_path: str,
    family: WorkUnitFamily,
    shard_index: int,
    first: int,
    last: int,
    path_count: int,
    years: int,
    expected_sources: tuple[str, ...],
) -> dict[str, Any]:
    expected_keys = {
        "frequency",
        "relative_path",
        "rows",
        "bytes",
        "sha256",
        "source_fingerprints",
        "csv_receipt_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise IntegrityError("work-unit sibling identity keys are invalid")
    if value["frequency"] != frequency or value["relative_path"] != relative_path:
        raise IntegrityError("work-unit sibling frequency/path is invalid")
    rows = _positive_integer(value["rows"], "sibling rows")
    bytes_count = _positive_integer(value["bytes"], "sibling bytes")
    sha256 = _required_fingerprint(value["sha256"], "sibling sha256")
    sources = _fingerprint_list(
        value["source_fingerprints"], path_count, "sibling source"
    )
    if sources != expected_sources:
        raise IntegrityError("work-unit sibling source lineage differs")
    receipt_fingerprint = _required_fingerprint(
        value["csv_receipt_fingerprint"], "CSV receipt fingerprint"
    )
    periods = {"monthly": 12, "quarterly": 4, "annual": 1, "daily": 252, "weekly": 52}[
        frequency
    ]
    if rows != path_count * years * periods:
        raise IntegrityError("work-unit sibling row count is invalid")
    csv_identity = _receipt_identity(
        relative_path=relative_path,
        frequency=frequency,
        family=family,
        shard_index=shard_index,
        first_path_entity=first,
        last_path_entity=last,
        first_path_id=_path_id(family, first),
        last_path_id=_path_id(family, last),
        path_count=path_count,
        years=years,
        rows=rows,
        bytes_count=bytes_count,
        sha256=sha256,
        source_fingerprints=sources,
    )
    if receipt_fingerprint != _sha256_json(csv_identity):
        raise IntegrityError("embedded CSV receipt fingerprint is invalid")
    return {
        "frequency": frequency,
        "relative_path": relative_path,
        "rows": rows,
        "bytes": bytes_count,
        "sha256": sha256,
        "source_fingerprints": sources,
        "csv_receipt_fingerprint": receipt_fingerprint,
    }


def _verify_group_siblings(root: Path, identity: Mapping[str, Any]) -> None:
    for sibling in identity["siblings"]:
        _verify_csv_record(
            root / sibling["relative_path"],
            family=identity["family"],
            frequency=sibling["frequency"],
            first=identity["first_path_entity"],
            last=identity["last_path_entity"],
            years=identity["years"],
            expected_rows=sibling["rows"],
            expected_bytes=sibling["bytes"],
            expected_sha256=sibling["sha256"],
        )


def _verify_csv_sibling(root: Path, receipt: CSVShardReceipt) -> None:
    _verify_csv_record(
        root / receipt.relative_path,
        family=receipt.family,
        frequency=receipt.frequency,
        first=receipt.first_path_entity,
        last=receipt.last_path_entity,
        years=receipt.years,
        expected_rows=receipt.rows,
        expected_bytes=receipt.bytes,
        expected_sha256=receipt.sha256,
    )


def _verify_csv_record(
    path: Path,
    *,
    family: CSVFamily,
    frequency: CSVFrequency,
    first: int,
    last: int,
    years: int,
    expected_rows: int,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    descriptor = _open_regular_single_link(path, "CSV sibling")
    digest = hashlib.sha256()
    bytes_count = 0
    rows = 0
    periods = {"monthly": 12, "quarterly": 4, "annual": 1, "daily": 252, "weekly": 52}[
        frequency
    ]
    rows_per_path = years * periods
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            header = handle.readline()
            digest.update(header)
            bytes_count += len(header)
            if header != CSV_HEADER_BYTES:
                raise IntegrityError("CSV sibling header is invalid")
            for line in handle:
                digest.update(line)
                bytes_count += len(line)
                if not line.endswith(b"\n") or b"\r" in line:
                    raise IntegrityError("CSV sibling line ending is invalid")
                zero_index = rows % rows_per_path
                entity = first + rows // rows_per_path
                if entity > last:
                    raise IntegrityError("CSV sibling contains excess path rows")
                period_index = zero_index + 1
                simulation_year = zero_index // periods + 1
                period_in_year = zero_index % periods + 1
                try:
                    fields = line[:-1].decode("utf-8").split(",")
                except UnicodeDecodeError as error:
                    raise IntegrityError("CSV sibling row is not UTF-8") from error
                if len(fields) != 7 or fields[:4] != [
                    _path_id(family, entity),
                    str(period_index),
                    str(simulation_year),
                    str(period_in_year),
                ]:
                    raise IntegrityError(
                        "CSV sibling row key/field contract is invalid"
                    )
                for text in fields[4:]:
                    try:
                        parsed = float(text)
                    except ValueError as error:
                        raise IntegrityError(
                            "CSV sibling return is not decimal"
                        ) from error
                    if format_csv_float64(parsed) != text:
                        raise IntegrityError(
                            "CSV sibling return grammar is noncanonical"
                        )
                rows += 1
            final_stat = os.fstat(handle.fileno())
            if final_stat.st_nlink != 1 or not stat.S_ISREG(final_stat.st_mode):
                raise IntegrityError("CSV sibling file identity changed during read")
            _require_path_matches_descriptor(path, final_stat, "CSV sibling")
    except GenerationError as error:
        raise IntegrityError("CSV sibling contains a non-finite return") from error
    if (
        rows != expected_rows
        or rows != (last - first + 1) * rows_per_path
        or bytes_count != expected_bytes
        or digest.hexdigest() != expected_sha256
    ):
        raise IntegrityError("CSV sibling hash/size/row contract is invalid")


def _scope_range(first: object, last: object) -> tuple[int, int]:
    checked_first = _positive_integer(first, "first path entity", maximum=999_999)
    checked_last = _positive_integer(last, "last path entity", maximum=999_999)
    if checked_first > checked_last:
        raise GenerationError("work-unit first path entity exceeds last")
    return checked_first, checked_last


def _source_tuple(value: Sequence[_SourceT], label: str) -> tuple[_SourceT, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise GenerationError(f"{label} must be a finite sequence")
    return tuple(value)


def _source_lengths(
    count: int,
    first: int,
    last: int,
    bases: tuple[Any, ...],
    derived: tuple[Any, ...],
) -> None:
    if count != last - first + 1:
        raise GenerationError("expected path count differs from work-unit range")
    if len(bases) != count or len(derived) != count:
        raise GenerationError("work-unit source count differs from expected range")


def _common_years(current: int | None, value: int) -> int:
    if current is not None and current != value:
        raise GenerationError("work-unit sources cannot mix horizons")
    return value


def _binding(value: object) -> StorageCommitBinding:
    if type(value) is not StorageCommitBinding:
        raise GenerationError("work-unit storage binding must be typed")
    return value


def _work_unit_id(value: object) -> str:
    if not isinstance(value, str) or _WORK_UNIT_ID.fullmatch(value) is None:
        raise GenerationError("work-unit ID is not a safe registered identifier")
    return value


def _require_canonical_work_unit_id(
    value: str,
    family: WorkUnitFamily,
    shard_index: int,
    first: int,
    last: int,
) -> None:
    expected = serial_work_unit_id(family, shard_index, first, last)
    if value != expected:
        raise GenerationError(
            "work-unit ID does not match its canonical family/shard/range scope",
            details={"expected": expected, "actual": value},
        )


def _worker_identity(value: object) -> str:
    if not isinstance(value, str) or _WORKER_IDENTITY.fullmatch(value) is None:
        raise GenerationError("work-unit worker identity is invalid")
    return value


def _family(value: object) -> WorkUnitFamily:
    if not isinstance(value, str) or value not in {"LF", "HF"}:
        raise GenerationError("work-unit family must be LF or HF")
    return cast(WorkUnitFamily, value)


def _positive_integer(value: object, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise GenerationError(f"{label} exceeds {maximum}")
    return value


def _nonnegative_integer(value: object, label: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise GenerationError(f"{label} must be in [0, {maximum}]")
    return value


def _nonnegative_integer_unbounded(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationError(f"{label} must be a nonnegative integer")
    return value


def _frequencies(family: WorkUnitFamily) -> tuple[CSVFrequency, ...]:
    return ("monthly", "quarterly", "annual") if family == "LF" else ("daily", "weekly")


def _sibling_relative_paths(
    family: WorkUnitFamily, shard_index: int, first: int, last: int
) -> tuple[str, ...]:
    directory = "low" if family == "LF" else "high"
    first_id = _path_id(family, first)
    last_id = _path_id(family, last)
    filename = f"part-{shard_index:05d}-path-{first_id}-{last_id}.csv"
    return tuple(
        PurePosixPath("consumer", "returns", directory, frequency, filename).as_posix()
        for frequency in _frequencies(family)
    )


def _receipt_relative_path(work_unit_id: str) -> PurePosixPath:
    return GROUP_RECEIPT_DIRECTORY / f"{work_unit_id}.json"


def _existing_safe_root(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("work-unit output root must be a path")
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise GenerationError("work-unit root must be absolute and traversal-free")
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError("work-unit root is missing or unsafe")
    for component in reversed((root, *root.parents)):
        if component.is_symlink():
            raise IntegrityError("work-unit root contains a symbolic link")
    return root


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} is invalid")
    return value


def _fingerprint_list(value: object, size: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or len(value) != size:
        raise IntegrityError(f"work-unit {label} fingerprint count is invalid")
    return tuple(_required_fingerprint(item, f"{label} fingerprint") for item in value)


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _json_thaw(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise IntegrityError("work-unit receipt contains noncanonical JSON") from error


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_thaw(item) for item in value]
    return value


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _open_regular_single_link(path: Path, label: str) -> int:
    _reject_symlink_path(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError(f"{label} cannot be opened") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise IntegrityError(f"{label} is not a regular single-link file")
    return descriptor


def _read_regular_single_link(path: Path, label: str) -> bytes:
    descriptor = _open_regular_single_link(path, label)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            content = handle.read()
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise IntegrityError(f"{label} changed during read")
            _require_path_matches_descriptor(path, metadata, label)
            return content
    except OSError as error:
        raise IntegrityError(f"{label} cannot be read") from error


def _require_regular_single_link(path: Path, label: str) -> None:
    descriptor = _open_regular_single_link(path, label)
    os.close(descriptor)


def _reject_symlink_path(path: Path, label: str) -> None:
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise IntegrityError(f"{label} path contains a symbolic link")


def _require_path_matches_descriptor(
    path: Path, metadata: os.stat_result, label: str
) -> None:
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise IntegrityError(f"{label} path changed during read") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or current.st_dev != metadata.st_dev
        or current.st_ino != metadata.st_ino
    ):
        raise IntegrityError(f"{label} path changed during read")


def _new_quarantine_directory(parent: Path) -> Path:
    for attempt in range(100_000):
        candidate = parent / f"attempt-{attempt:05d}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        _fsync_directory(parent)
        return candidate
    raise IntegrityError("work-unit quarantine namespace is exhausted")


def _quarantine_scope(
    root: Path,
    work_unit_id: str,
    family: WorkUnitFamily,
    shard_index: int,
    first: int,
    last: int,
) -> QuarantinedWorkUnit:
    return quarantine_serial_work_unit(
        root,
        work_unit_id=work_unit_id,
        family=family,
        shard_index=shard_index,
        first_path_entity=first,
        last_path_entity=last,
    )
