"""Streaming structural validation for one practical K=4 delivery run.

The validator consumes the concise practical production manifest and the five
consumer CSV datasets.  It checks only observable delivery outcomes: exact
geometry, checksums, row keys, finite returns, cross-frequency log sums, and
duplicate complete base paths.  At most one path from each sibling frequency
is retained in memory, so the canonical 19.45-million-row delivery can be
validated without loading a dataset or shard in full.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from prpg.data.calendar import WEEK_BOUNDARIES
from prpg.errors import GenerationError, IntegrityError
from prpg.simulation.production import (
    ProductionPlan,
    ProductionWorkUnit,
    production_work_units,
)
from prpg.storage.csv_v1 import CSV_HEADER_BYTES

PRACTICAL_DELIVERY_REPORT_SCHEMA_ID: Final = "prpg-practical-delivery-report-v1"
PRACTICAL_MANIFEST_SCHEMA_ID: Final = "prpg-practical-manifest-v1"
MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_CSV_ROW_BYTES: Final = 1024
ASSET_COUNT: Final = 3
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")

Family: TypeAlias = Literal["LF", "HF"]
Frequency: TypeAlias = Literal["monthly", "quarterly", "annual", "daily", "weekly"]
FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PracticalDatasetSummary:
    """Plan-derived and fully observed geometry for one CSV frequency."""

    family: Family
    frequency: Frequency
    paths: int
    periods_per_path: int
    shards: int
    rows: int


@dataclass(frozen=True, slots=True)
class PracticalDeliveryReport:
    """Compact result returned only after every structural check passes."""

    schema_id: str
    status: Literal["passed"]
    profile_id: str
    years: int
    lf_paths: int
    hf_paths: int
    work_units: int
    csv_shards: int
    total_rows: int
    total_bytes: int
    checksums_verified: int
    aggregation_paths_checked: int
    aggregation_values_checked: int
    duplicate_base_paths: int
    manifest_sha256: str
    elapsed_seconds: float
    rows_per_second: float
    datasets: tuple[PracticalDatasetSummary, ...]


@dataclass(frozen=True, slots=True)
class _FrequencySpec:
    family: Family
    family_directory: Literal["low", "high"]
    periods_per_year: int


_FREQUENCY_SPECS: Final[Mapping[Frequency, _FrequencySpec]] = {
    "monthly": _FrequencySpec("LF", "low", 12),
    "quarterly": _FrequencySpec("LF", "low", 4),
    "annual": _FrequencySpec("LF", "low", 1),
    "daily": _FrequencySpec("HF", "high", 252),
    "weekly": _FrequencySpec("HF", "high", 52),
}
_FAMILY_FREQUENCIES: Final[Mapping[Family, tuple[Frequency, ...]]] = {
    "LF": ("monthly", "quarterly", "annual"),
    "HF": ("daily", "weekly"),
}
_ORDERED_FREQUENCIES: Final[tuple[Frequency, ...]] = (
    "monthly",
    "quarterly",
    "annual",
    "daily",
    "weekly",
)


@dataclass(frozen=True, slots=True)
class _ShardRecord:
    work_unit_id: str
    family: Family
    frequency: Frequency
    relative_path: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedManifest:
    plan: ProductionPlan
    manifest_sha256: str
    scopes: tuple[ProductionWorkUnit, ...]
    records: tuple[tuple[_ShardRecord, ...], ...]
    total_bytes: int


def validate_practical_delivery(
    run_root: str | Path,
) -> PracticalDeliveryReport:
    """Validate one complete practical run using bounded streaming reads.

    The function does not publish, repair, or modify any run artifact.  Any
    unexpected manifest, file, row, checksum, aggregation, or duplicate-path
    condition aborts with :class:`IntegrityError`; otherwise a small in-memory
    report is returned.
    """

    root = _existing_run_root(run_root)
    started = time.monotonic()
    manifest = _load_and_validate_manifest(root)
    flat_records = tuple(record for group in manifest.records for record in group)
    _validate_checksum_summary(root, flat_records)
    _validate_csv_allowlist(root, flat_records)

    duplicate_digests: dict[Family, set[bytes]] = {"LF": set(), "HF": set()}
    aggregation_values = 0
    for scope, records in zip(manifest.scopes, manifest.records, strict=True):
        streams = tuple(
            _CSVShardStream(root, record, manifest.plan) for record in records
        )
        try:
            for entity in range(scope.first_path_entity, scope.last_path_entity + 1):
                paths = tuple(stream.read_path(entity) for stream in streams)
                if scope.family == "LF":
                    aggregation_values += _validate_lf_aggregates(
                        paths[0], paths[1], paths[2], manifest.plan.years
                    )
                    base = paths[0]
                else:
                    aggregation_values += _validate_hf_aggregates(
                        paths[0], paths[1], manifest.plan.years
                    )
                    base = paths[0]
                digest = hashlib.sha256(base.tobytes(order="C")).digest()
                family_digests = duplicate_digests[scope.family]
                if digest in family_digests:
                    raise IntegrityError(
                        "practical delivery contains a duplicate complete base path",
                        details={"family": scope.family, "path_entity": entity},
                    )
                family_digests.add(digest)
            for stream in streams:
                stream.finish()
        finally:
            for stream in streams:
                stream.close()

    elapsed = max(time.monotonic() - started, 1e-12)
    plan = manifest.plan
    return PracticalDeliveryReport(
        schema_id=PRACTICAL_DELIVERY_REPORT_SCHEMA_ID,
        status="passed",
        profile_id=plan.profile_id,
        years=plan.years,
        lf_paths=plan.lf_paths,
        hf_paths=plan.hf_paths,
        work_units=plan.work_unit_count,
        csv_shards=plan.csv_shard_count,
        total_rows=plan.total_rows,
        total_bytes=manifest.total_bytes,
        checksums_verified=plan.csv_shard_count,
        aggregation_paths_checked=plan.lf_paths + plan.hf_paths,
        aggregation_values_checked=aggregation_values,
        duplicate_base_paths=0,
        manifest_sha256=manifest.manifest_sha256,
        elapsed_seconds=elapsed,
        rows_per_second=plan.total_rows / elapsed,
        datasets=_dataset_summaries(plan),
    )


def _load_and_validate_manifest(root: Path) -> _ValidatedManifest:
    manifest_path = root / "manifest.json"
    content = _read_regular_file_limited(
        manifest_path, maximum=MAX_MANIFEST_BYTES, label="practical manifest"
    )
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("practical manifest is not valid JSON") from error
    manifest = _mapping(value, "practical manifest")
    if manifest.get("schema_id") != PRACTICAL_MANIFEST_SCHEMA_ID:
        raise IntegrityError("practical manifest schema is invalid")
    if manifest.get("status") != "complete":
        raise IntegrityError("practical production run is not complete")
    fingerprint = _fingerprint_value(
        manifest.get("manifest_fingerprint"), "practical manifest fingerprint"
    )
    identity = dict(manifest)
    identity.pop("manifest_fingerprint", None)
    if fingerprint != _sha256_json(identity):
        raise IntegrityError("practical manifest fingerprint differs")

    plan = _plan_from_manifest(manifest.get("plan"))
    scopes = production_work_units(plan)
    records = _records_from_manifest(manifest.get("work_units"), plan, scopes)
    flat = tuple(record for group in records for record in group)
    total_bytes = sum(record.bytes for record in flat)
    totals = _mapping(manifest.get("totals"), "practical manifest totals")
    expected_totals = {
        "paths": plan.lf_paths + plan.hf_paths,
        "work_units": plan.work_unit_count,
        "csv_shards": plan.csv_shard_count,
        "rows": plan.total_rows,
        "bytes": total_bytes,
    }
    if any(totals.get(key) != expected for key, expected in expected_totals.items()):
        raise IntegrityError("practical manifest totals differ from its plan")
    workers = manifest.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise IntegrityError("practical manifest worker count is invalid")
    return _ValidatedManifest(
        plan=plan,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        scopes=scopes,
        records=records,
        total_bytes=total_bytes,
    )


def _plan_from_manifest(value: object) -> ProductionPlan:
    raw = _mapping(value, "practical production plan")
    profile_id = raw.get("profile_id")
    if not isinstance(profile_id, str):
        raise IntegrityError("practical production profile is invalid")
    try:
        plan = ProductionPlan(
            profile_id=profile_id,
            years=_manifest_int(raw, "years"),
            lf_paths=_manifest_int(raw, "lf_paths"),
            hf_paths=_manifest_int(raw, "hf_paths"),
            lf_paths_per_shard=_manifest_int(raw, "lf_paths_per_shard"),
            hf_paths_per_shard=_manifest_int(raw, "hf_paths_per_shard"),
        )
    except GenerationError as error:
        raise IntegrityError("practical production plan is invalid") from error
    if dict(raw) != plan.identity():
        raise IntegrityError("practical production plan identity differs")
    return plan


def _records_from_manifest(
    value: object,
    plan: ProductionPlan,
    scopes: tuple[ProductionWorkUnit, ...],
) -> tuple[tuple[_ShardRecord, ...], ...]:
    raw_units = _sequence(value, "practical work units")
    by_id: dict[str, Mapping[str, object]] = {}
    for value_item in raw_units:
        item = _mapping(value_item, "practical work unit")
        identifier = item.get("work_unit_id")
        if not isinstance(identifier, str) or identifier in by_id:
            raise IntegrityError(
                "practical work-unit identity is invalid or duplicated"
            )
        _fingerprint_value(
            item.get("receipt_fingerprint"), "practical work-unit receipt"
        )
        by_id[identifier] = item
    if len(by_id) != plan.work_unit_count:
        raise IntegrityError("practical work-unit count differs from its plan")

    result: list[tuple[_ShardRecord, ...]] = []
    for scope in scopes:
        if scope.work_unit_id not in by_id:
            raise IntegrityError("practical manifest is missing a planned work unit")
        item = by_id.pop(scope.work_unit_id)
        siblings = _sequence(item.get("siblings"), "practical sibling records")
        sibling_by_frequency: dict[str, Mapping[str, object]] = {}
        for sibling_value in siblings:
            sibling = _mapping(sibling_value, "practical sibling record")
            frequency = sibling.get("frequency")
            if not isinstance(frequency, str) or frequency in sibling_by_frequency:
                raise IntegrityError(
                    "practical sibling frequency is invalid or duplicated"
                )
            sibling_by_frequency[frequency] = sibling
        expected_frequencies = _FAMILY_FREQUENCIES[scope.family]
        if set(sibling_by_frequency) != set(expected_frequencies):
            raise IntegrityError(
                "practical sibling frequency set differs from its plan"
            )

        count = scope.last_path_entity - scope.first_path_entity + 1
        scope_records: list[_ShardRecord] = []
        for frequency in expected_frequencies:
            sibling = sibling_by_frequency[frequency]
            spec = _FREQUENCY_SPECS[frequency]
            relative = _expected_relative_path(scope, frequency, spec)
            rows = count * plan.years * spec.periods_per_year
            if sibling.get("relative_path") != relative or sibling.get("rows") != rows:
                raise IntegrityError("practical sibling geometry differs from its plan")
            bytes_count = _manifest_int(sibling, "bytes")
            if bytes_count <= len(CSV_HEADER_BYTES):
                raise IntegrityError("practical sibling byte count is invalid")
            sha256 = _fingerprint_value(sibling.get("sha256"), "practical CSV")
            scope_records.append(
                _ShardRecord(
                    work_unit_id=scope.work_unit_id,
                    family=scope.family,
                    frequency=frequency,
                    relative_path=relative,
                    rows=rows,
                    bytes=bytes_count,
                    sha256=sha256,
                )
            )
        result.append(tuple(scope_records))
    if by_id:
        raise IntegrityError("practical manifest contains unplanned work units")
    return tuple(result)


def _expected_relative_path(
    scope: ProductionWorkUnit,
    frequency: Frequency,
    spec: _FrequencySpec,
) -> str:
    first = f"{scope.family}{scope.first_path_entity:06d}"
    last = f"{scope.family}{scope.last_path_entity:06d}"
    filename = f"part-{scope.shard_index:05d}-path-{first}-{last}.csv"
    return PurePosixPath(
        "consumer", "returns", spec.family_directory, frequency, filename
    ).as_posix()


def _validate_checksum_summary(root: Path, records: tuple[_ShardRecord, ...]) -> None:
    expected = "".join(
        f"{record.sha256}  {record.relative_path}\n"
        for record in sorted(records, key=lambda item: item.relative_path)
    ).encode("utf-8")
    observed = _read_regular_file_limited(
        root / "checksums.sha256",
        maximum=len(expected) + 1,
        label="practical checksum summary",
    )
    if observed != expected:
        raise IntegrityError("practical checksum summary differs from its manifest")


def _validate_csv_allowlist(root: Path, records: tuple[_ShardRecord, ...]) -> None:
    expected = {record.relative_path for record in records}
    returns_root = root / "consumer" / "returns"
    if returns_root.is_symlink() or not returns_root.is_dir():
        raise IntegrityError("practical consumer return directory is missing or unsafe")
    actual: set[str] = set()
    try:
        for child in returns_root.rglob("*"):
            metadata = child.lstat()
            if stat.S_ISDIR(metadata.st_mode) and not child.is_symlink():
                continue
            if stat.S_ISREG(metadata.st_mode) and not child.is_symlink():
                actual.add(child.relative_to(root).as_posix())
                continue
            raise IntegrityError("practical consumer return tree has an unsafe entry")
    except OSError as error:
        raise IntegrityError(
            "practical consumer return tree cannot be scanned"
        ) from error
    if actual != expected:
        raise IntegrityError("practical CSV file set differs from its plan")


class _CSVShardStream:
    """Hashing line reader retaining only one planned path at a time."""

    def __init__(self, root: Path, record: _ShardRecord, plan: ProductionPlan) -> None:
        self.record = record
        self.periods_per_year = _FREQUENCY_SPECS[record.frequency].periods_per_year
        self.periods_per_path = plan.years * self.periods_per_year
        self.path = root.joinpath(*PurePosixPath(record.relative_path).parts)
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._rows = 0
        self._finished = False
        descriptor, metadata = _open_regular_file(self.path, "practical CSV shard")
        self._initial_stat = metadata
        self._handle = os.fdopen(descriptor, "rb", closefd=True)
        try:
            header = self._handle.readline(len(CSV_HEADER_BYTES) + 1)
            self._digest.update(header)
            self._bytes += len(header)
            if header != CSV_HEADER_BYTES:
                raise IntegrityError("practical CSV-v1 header is invalid")
        except BaseException:
            self._handle.close()
            raise

    def read_path(self, path_entity: int) -> FloatArray:
        if self._finished:
            raise IntegrityError("practical CSV shard was read after finalization")
        values = np.empty((self.periods_per_path, ASSET_COUNT), dtype=np.float64)
        path_id = f"{self.record.family}{path_entity:06d}"
        for zero_index in range(self.periods_per_path):
            line = self._handle.readline(MAX_CSV_ROW_BYTES + 1)
            if not line:
                raise IntegrityError("practical CSV shard is truncated")
            if len(line) > MAX_CSV_ROW_BYTES:
                raise IntegrityError("practical CSV row exceeds its size bound")
            self._digest.update(line)
            self._bytes += len(line)
            if not line.endswith(b"\n") or b"\r" in line:
                raise IntegrityError("practical CSV row does not use exact LF endings")
            try:
                fields = line[:-1].decode("utf-8").split(",")
            except UnicodeDecodeError as error:
                raise IntegrityError("practical CSV row is not UTF-8") from error
            expected_keys = [
                path_id,
                str(zero_index + 1),
                str(zero_index // self.periods_per_year + 1),
                str(zero_index % self.periods_per_year + 1),
            ]
            if len(fields) != 7 or fields[:4] != expected_keys:
                raise IntegrityError(
                    "practical CSV path or period sequence is not contiguous"
                )
            for asset, text in enumerate(fields[4:]):
                try:
                    parsed = float(text)
                except ValueError as error:
                    raise IntegrityError(
                        "practical CSV return is not numeric"
                    ) from error
                if not math.isfinite(parsed):
                    raise IntegrityError("practical CSV return is not finite")
                values[zero_index, asset] = parsed
            self._rows += 1
        values.flags.writeable = False
        return values

    def finish(self) -> None:
        if self._finished:
            raise IntegrityError("practical CSV shard was finalized twice")
        self._finished = True
        if self._handle.read(1):
            raise IntegrityError("practical CSV shard contains excess rows")
        final = os.fstat(self._handle.fileno())
        current = self.path.stat(follow_symlinks=False)
        initial = self._initial_stat
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_dev != initial.st_dev
            or final.st_ino != initial.st_ino
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or current.st_dev != initial.st_dev
            or current.st_ino != initial.st_ino
            or current.st_size != initial.st_size
            or current.st_mtime_ns != initial.st_mtime_ns
        ):
            raise IntegrityError("practical CSV shard changed during validation")
        if (
            self._rows != self.record.rows
            or self._bytes != self.record.bytes
            or self._digest.hexdigest() != self.record.sha256
        ):
            raise IntegrityError("practical CSV row, byte, or checksum result differs")

    def close(self) -> None:
        self._handle.close()


def _validate_lf_aggregates(
    monthly: FloatArray,
    quarterly: FloatArray,
    annual: FloatArray,
    years: int,
) -> int:
    monthly_years = monthly.reshape(years, 12, ASSET_COUNT)
    expected_quarterly = (
        monthly_years.reshape(years, 4, 3, ASSET_COUNT)
        .sum(axis=2)
        .reshape(-1, ASSET_COUNT)
    )
    expected_annual = monthly_years.sum(axis=1)
    if not np.array_equal(expected_quarterly, quarterly) or not np.array_equal(
        expected_annual, annual
    ):
        raise IntegrityError("practical LF cross-frequency log sums are not exact")
    return quarterly.size + annual.size


def _validate_hf_aggregates(daily: FloatArray, weekly: FloatArray, years: int) -> int:
    expected = np.empty((years * 52, ASSET_COUNT), dtype=np.float64)
    output = 0
    for year in range(years):
        offset = year * 252
        for week in range(52):
            start = offset + WEEK_BOUNDARIES[week]
            stop = offset + WEEK_BOUNDARIES[week + 1]
            expected[output] = daily[start:stop].sum(axis=0)
            output += 1
    if not np.array_equal(expected, weekly):
        raise IntegrityError("practical HF daily-to-weekly log sums are not exact")
    return weekly.size


def _dataset_summaries(plan: ProductionPlan) -> tuple[PracticalDatasetSummary, ...]:
    result: list[PracticalDatasetSummary] = []
    for frequency in _ORDERED_FREQUENCIES:
        spec = _FREQUENCY_SPECS[frequency]
        paths = plan.lf_paths if spec.family == "LF" else plan.hf_paths
        shard_size = (
            plan.lf_paths_per_shard if spec.family == "LF" else plan.hf_paths_per_shard
        )
        periods = plan.years * spec.periods_per_year
        result.append(
            PracticalDatasetSummary(
                family=spec.family,
                frequency=frequency,
                paths=paths,
                periods_per_path=periods,
                shards=(paths + shard_size - 1) // shard_size,
                rows=paths * periods,
            )
        )
    return tuple(result)


def _existing_run_root(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("practical delivery root must be a path")
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise IntegrityError("practical delivery root cannot be a symbolic link")
    try:
        root = candidate.resolve(strict=True)
    except OSError as error:
        raise IntegrityError("practical delivery root does not exist") from error
    if not root.is_dir():
        raise IntegrityError("practical delivery root is not a directory")
    return root


def _read_regular_file_limited(path: Path, *, maximum: int, label: str) -> bytes:
    descriptor, metadata = _open_regular_file(path, label)
    try:
        if metadata.st_size > maximum:
            raise IntegrityError(f"{label} exceeds its size bound")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            content = handle.read(maximum + 1)
        if len(content) > maximum:
            raise IntegrityError(f"{label} exceeds its size bound")
        return content
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_regular_file(path: Path, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise IntegrityError(f"{label} is missing or cannot be opened") from error
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise IntegrityError(f"{label} is not a regular file")
    return descriptor, metadata


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise IntegrityError(f"{label} is not an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise IntegrityError(f"{label} is not an array")
    return cast(Sequence[object], value)


def _manifest_int(value: Mapping[str, object], key: str) -> int:
    observed = value.get(key)
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        raise IntegrityError(f"practical manifest {key} is not a positive integer")
    return observed


def _fingerprint_value(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} is not a lowercase SHA-256 fingerprint")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IntegrityError("practical manifest is not canonical JSON data") from error


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()
