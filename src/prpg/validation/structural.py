"""Streaming G7 validation of the complete consumer CSV bundle.

This module deliberately has no CLI, scheduling, generation, qualification,
or release-finalization responsibilities.  It consumes the exact durable
storage receipts, a minimal strict consumer manifest, and the published CSV
tree.  Every row is parsed independently while retaining at most one path from
each sibling frequency in memory.

``DATA_VALIDATED`` is the only lifecycle marker this module can create.  Its
high-level publication function is restricted to the canonical 50-year
5,000-LF/1,000-HF plan, requires a separate scientific-pass fingerprint, and
publishes nothing when any structural check fails.  ``COMPLETE`` and
``RELEASED`` remain later, independent release concerns.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, TypeAlias, TypeGuard, cast

import numpy as np
import numpy.typing as npt

from prpg.data.calendar import CALENDAR_ID, WEEK_BOUNDARIES
from prpg.errors import GenerationError, IntegrityError
from prpg.storage.csv_v1 import (
    CSV_HEADER_BYTES,
    CSV_SCHEMA_VERSION,
    CSV_SERIALIZER_VERSION,
    CSVFamily,
    CSVFrequency,
    format_csv_float64,
)
from prpg.storage.group_commit import (
    LoadedWorkUnitReceipt,
    StorageCommitBinding,
    serial_work_unit_id,
    verify_serial_work_unit,
)

STRUCTURAL_PLAN_SCHEMA_ID: Final = "prpg-g7-structural-plan-v1"
CONSUMER_MANIFEST_SCHEMA_ID: Final = "prpg-consumer-structural-manifest-v1"
CONSUMER_MANIFEST_SCHEMA_VERSION: Final = 1
STRUCTURAL_REPORT_SCHEMA_ID: Final = "prpg-g7-structural-report-v1"
STRUCTURAL_REPORT_SCHEMA_VERSION: Final = 1
DATA_VALIDATED_SCHEMA_ID: Final = "prpg-data-validated-v1"
DATA_VALIDATED_SCHEMA_VERSION: Final = 1
LOG_AGGREGATION_TOLERANCE: Final = 1e-12
SIMPLE_RETURN_TOLERANCE: Final = 1e-10
MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
MAX_CSV_ROW_BYTES: Final = 1024
ASSET_COUNT: Final = 3
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_HOOK_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_REPORT_SEAL = object()

Family: TypeAlias = Literal["LF", "HF"]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
FloatArray: TypeAlias = npt.NDArray[np.float64]


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
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class _DatasetSpec:
    family: Family
    family_directory: Literal["low", "high"]
    frequency: CSVFrequency
    periods_per_year: int


_DATASETS: Final = (
    _DatasetSpec("LF", "low", "monthly", 12),
    _DatasetSpec("LF", "low", "quarterly", 4),
    _DatasetSpec("LF", "low", "annual", 1),
    _DatasetSpec("HF", "high", "daily", 252),
    _DatasetSpec("HF", "high", "weekly", 52),
)
_DATASET_BY_FREQUENCY: Final = MappingProxyType(
    {item.frequency: item for item in _DATASETS}
)
_FAMILY_FREQUENCIES: Final = MappingProxyType(
    {
        "LF": ("monthly", "quarterly", "annual"),
        "HF": ("daily", "weekly"),
    }
)


@dataclass(frozen=True, slots=True)
class StructuralValidationPlan:
    """Exact expected output geometry for one structural scan.

    Noncanonical geometries exist only so the same production parser can be
    exercised on small fixtures.  Only ``CANONICAL_STRUCTURAL_PLAN`` may reach
    the ``DATA_VALIDATED`` publication boundary.
    """

    profile_id: str
    years: int
    lf_paths: int
    hf_paths: int
    lf_paths_per_shard: int
    hf_paths_per_shard: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", self.profile_id) is None
        ):
            raise GenerationError("structural plan profile_id is invalid")
        for name in (
            "years",
            "lf_paths",
            "hf_paths",
            "lf_paths_per_shard",
            "hf_paths_per_shard",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GenerationError(f"structural plan {name} must be positive")
        if self.lf_paths > 999_999 or self.hf_paths > 999_999:
            raise GenerationError("structural plan path count exceeds CSV namespace")
        identity = self.identity()
        object.__setattr__(self, "fingerprint", _sha256_json(identity))

    def identity(self) -> dict[str, JSONValue]:
        """Return the location-independent frozen geometry identity."""

        return {
            "schema_id": STRUCTURAL_PLAN_SCHEMA_ID,
            "calendar_id": CALENDAR_ID,
            "csv_schema_version": CSV_SCHEMA_VERSION,
            "csv_serializer_version": CSV_SERIALIZER_VERSION,
            "profile_id": self.profile_id,
            "years": self.years,
            "lf_paths": self.lf_paths,
            "hf_paths": self.hf_paths,
            "lf_paths_per_shard": self.lf_paths_per_shard,
            "hf_paths_per_shard": self.hf_paths_per_shard,
        }

    @property
    def total_rows(self) -> int:
        """Return the exact row total across all five frequencies."""

        lf_periods = self.years * (12 + 4 + 1)
        hf_periods = self.years * (252 + 52)
        return self.lf_paths * lf_periods + self.hf_paths * hf_periods

    @property
    def total_values(self) -> int:
        return self.total_rows * ASSET_COUNT

    @property
    def work_unit_count(self) -> int:
        return _ceiling_divide(
            self.lf_paths, self.lf_paths_per_shard
        ) + _ceiling_divide(self.hf_paths, self.hf_paths_per_shard)

    @property
    def csv_shard_count(self) -> int:
        return 3 * _ceiling_divide(
            self.lf_paths, self.lf_paths_per_shard
        ) + 2 * _ceiling_divide(self.hf_paths, self.hf_paths_per_shard)


CANONICAL_STRUCTURAL_PLAN: Final = StructuralValidationPlan(
    profile_id="canonical-50y-v1",
    years=50,
    lf_paths=5_000,
    hf_paths=1_000,
    lf_paths_per_shard=100,
    hf_paths_per_shard=50,
)


@dataclass(frozen=True, slots=True)
class StructuralPathView:
    """One already-validated path delivered to an optional metric hook."""

    family: Family
    frequency: CSVFrequency
    path_entity: int
    path_id: str
    log_returns: FloatArray = field(repr=False)


class StructuralMetricHook(Protocol):
    """Bounded-memory extension point for later G5/G7 scientific metrics."""

    name: str

    def observe_path(self, path: StructuralPathView) -> None:
        """Consume one path after its structural checks have passed."""

    def finalize(self) -> Mapping[str, JSONValue]:
        """Return a finite canonical-JSON metric payload."""


@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    family: Family
    frequency: CSVFrequency
    paths: int
    periods_per_path: int
    rows: int
    values: int
    shards: int


@dataclass(frozen=True, slots=True)
class ShardValidationSummary:
    relative_path: str
    rows: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True, init=False)
class StructuralValidationReport:
    """Immutable all-row structural pass report."""

    schema_id: str
    schema_version: int
    plan_fingerprint: str
    canonical_profile: bool
    manifest_sha256: str
    total_rows: int
    total_values: int
    work_units: int
    csv_shards: int
    aggregation_values_checked: int
    roundtrip_values_checked: int
    max_abs_log_aggregation_error: float
    max_abs_simple_compounding_error: float
    max_abs_log_simple_roundtrip_error: float
    elapsed_seconds: float
    rows_per_second: float
    datasets: tuple[DatasetValidationSummary, ...]
    shards: tuple[ShardValidationSummary, ...]
    hook_metrics: Mapping[str, JSONValue]
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def identity(self) -> dict[str, JSONValue]:
        """Return the exact JSON identity covered by ``fingerprint``."""

        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "plan_fingerprint": self.plan_fingerprint,
            "canonical_profile": self.canonical_profile,
            "manifest_sha256": self.manifest_sha256,
            "total_rows": self.total_rows,
            "total_values": self.total_values,
            "work_units": self.work_units,
            "csv_shards": self.csv_shards,
            "aggregation_values_checked": self.aggregation_values_checked,
            "roundtrip_values_checked": self.roundtrip_values_checked,
            "max_abs_log_aggregation_error": self.max_abs_log_aggregation_error,
            "max_abs_simple_compounding_error": (self.max_abs_simple_compounding_error),
            "max_abs_log_simple_roundtrip_error": (
                self.max_abs_log_simple_roundtrip_error
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "rows_per_second": self.rows_per_second,
            "datasets": [
                {
                    "family": item.family,
                    "frequency": item.frequency,
                    "paths": item.paths,
                    "periods_per_path": item.periods_per_path,
                    "rows": item.rows,
                    "values": item.values,
                    "shards": item.shards,
                }
                for item in self.datasets
            ],
            "shards": [
                {
                    "relative_path": item.relative_path,
                    "rows": item.rows,
                    "bytes": item.bytes,
                    "sha256": item.sha256,
                }
                for item in self.shards
            ],
            "hook_metrics": _json_thaw(self.hook_metrics),
        }


@dataclass(frozen=True, slots=True)
class _WorkUnitScope:
    family: Family
    shard_index: int
    first: int
    last: int
    work_unit_id: str


@dataclass(slots=True)
class _ErrorAccumulator:
    aggregation_values: int = 0
    max_log: float = 0.0
    max_simple: float = 0.0
    roundtrip_values: int = 0
    max_roundtrip: float = 0.0


def publish_consumer_manifest_from_receipts(
    run_root: str | Path,
    *,
    plan: StructuralValidationPlan,
    binding: StorageCommitBinding,
) -> str:
    """Publish the minimal strict consumer manifest from verified receipts.

    The returned value is the SHA-256 of the exact manifest bytes.  Existing
    manifests are never overwritten.  No CSV content or receipt is created or
    repaired here.
    """

    root = _existing_safe_root(run_root)
    _require_plan(plan)
    if not isinstance(binding, StorageCommitBinding):
        raise GenerationError("consumer manifest requires a storage binding")
    _verify_receipt_allowlist(root, plan)
    records = _verified_manifest_records(root, plan, binding)
    document = _manifest_document(plan, binding, records)
    content = _canonical_bytes(document)
    path = root / "consumer" / "consumer-manifest.json"
    _publish_no_replace(path, content)
    return hashlib.sha256(content).hexdigest()


def validate_consumer_structure(
    run_root: str | Path,
    *,
    plan: StructuralValidationPlan = CANONICAL_STRUCTURAL_PLAN,
    hooks: Sequence[StructuralMetricHook] = (),
) -> StructuralValidationReport:
    """Stream and independently validate every consumer return row."""

    started = time.monotonic()
    root = _existing_safe_root(run_root)
    _require_plan(plan)
    checked_hooks = _validated_hooks(hooks)
    manifest, manifest_sha256 = _load_manifest(root, plan)
    binding = _manifest_binding(manifest)
    records = _manifest_records(manifest)
    _verify_receipt_allowlist(root, plan)
    verified_records = _verified_manifest_records(root, plan, binding)
    if records != verified_records:
        raise IntegrityError("consumer manifest differs from durable shard receipts")
    _verify_consumer_allowlist(root, records)

    errors = _ErrorAccumulator()
    shard_summaries: list[ShardValidationSummary] = []
    record_offset = 0
    for scope in _work_unit_scopes(plan):
        frequencies = _FAMILY_FREQUENCIES[scope.family]
        count = len(frequencies)
        scope_records = records[record_offset : record_offset + count]
        record_offset += count
        if tuple(item["frequency"] for item in scope_records) != frequencies:
            raise IntegrityError("consumer manifest work-unit frequency order differs")
        summaries = _scan_work_unit(
            root,
            plan=plan,
            scope=scope,
            records=scope_records,
            hooks=checked_hooks,
            errors=errors,
        )
        shard_summaries.extend(summaries)
    if record_offset != len(records):  # pragma: no cover - plan proves
        raise AssertionError("consumer manifest record traversal is incomplete")
    _verify_consumer_allowlist(root, records)

    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise IntegrityError("structural validation clock result is invalid")
    hook_metrics = _finalize_hooks(checked_hooks)
    report = _mint_report(
        plan=plan,
        manifest_sha256=manifest_sha256,
        shard_summaries=tuple(shard_summaries),
        errors=errors,
        elapsed_seconds=elapsed,
        hook_metrics=hook_metrics,
    )
    _verify_report(report)
    return report


def validate_and_publish_data_validated(
    run_root: str | Path,
    *,
    scientific_pass_fingerprint: str,
    hooks: Sequence[StructuralMetricHook] = (),
    plan: StructuralValidationPlan = CANONICAL_STRUCTURAL_PLAN,
) -> StructuralValidationReport:
    """Reject the obsolete untyped DATA_VALIDATED publication boundary.

    A bare fingerprint cannot prove that science consumed the same structural
    scan.  Call ``validate_consumer_structure(..., hooks=(g7_hook,))``, build
    the typed G7 scientific pass from that returned report, then use
    :func:`prpg.validation.scientific.publish_g7_validated_pair`.
    """

    _ = run_root, scientific_pass_fingerprint, hooks, plan
    raise GenerationError(
        "DATA_VALIDATED requires the coordinated typed G7 report pair"
    )


def estimate_canonical_scan_seconds(report: StructuralValidationReport) -> float:
    """Project canonical scan time from an observed structural scan rate."""

    _verify_report(report)
    if report.rows_per_second <= 0.0:
        raise IntegrityError("structural throughput must be positive")
    estimate = CANONICAL_STRUCTURAL_PLAN.total_rows / report.rows_per_second
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise IntegrityError("canonical structural scan estimate is invalid")
    return estimate


def is_sealed_structural_validation_report(
    value: object,
) -> TypeGuard[StructuralValidationReport]:
    try:
        _verify_report(value)
    except (AttributeError, GenerationError, IntegrityError, TypeError, ValueError):
        return False
    return True


def _work_unit_scopes(plan: StructuralValidationPlan) -> tuple[_WorkUnitScope, ...]:
    scopes: list[_WorkUnitScope] = []
    for family, paths, paths_per_shard in (
        ("LF", plan.lf_paths, plan.lf_paths_per_shard),
        ("HF", plan.hf_paths, plan.hf_paths_per_shard),
    ):
        for shard_index, first in enumerate(range(1, paths + 1, paths_per_shard)):
            last = min(paths, first + paths_per_shard - 1)
            scopes.append(
                _WorkUnitScope(
                    family=cast(Family, family),
                    shard_index=shard_index,
                    first=first,
                    last=last,
                    work_unit_id=serial_work_unit_id(
                        cast(Family, family), shard_index, first, last
                    ),
                )
            )
    return tuple(scopes)


def _dataset_summaries(plan: StructuralValidationPlan) -> list[dict[str, JSONValue]]:
    summaries: list[dict[str, JSONValue]] = []
    for spec in _DATASETS:
        paths = plan.lf_paths if spec.family == "LF" else plan.hf_paths
        paths_per_shard = (
            plan.lf_paths_per_shard if spec.family == "LF" else plan.hf_paths_per_shard
        )
        periods = plan.years * spec.periods_per_year
        rows = paths * periods
        summaries.append(
            {
                "family": spec.family,
                "frequency": spec.frequency,
                "paths": paths,
                "periods_per_path": periods,
                "rows": rows,
                "values": rows * ASSET_COUNT,
                "shards": _ceiling_divide(paths, paths_per_shard),
            }
        )
    return summaries


def _manifest_document(
    plan: StructuralValidationPlan,
    binding: StorageCommitBinding,
    records: list[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    return {
        "schema_id": CONSUMER_MANIFEST_SCHEMA_ID,
        "schema_version": CONSUMER_MANIFEST_SCHEMA_VERSION,
        "calendar_id": CALENDAR_ID,
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "csv_serializer_version": CSV_SERIALIZER_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "storage_binding": cast(dict[str, JSONValue], binding.as_dict()),
        "datasets": cast(JSONValue, _dataset_summaries(plan)),
        "work_unit_count": plan.work_unit_count,
        "csv_shard_count": plan.csv_shard_count,
        "total_rows": plan.total_rows,
        "total_values": plan.total_values,
        "shards": cast(JSONValue, records),
    }


def _verified_manifest_records(
    root: Path,
    plan: StructuralValidationPlan,
    binding: StorageCommitBinding,
) -> list[dict[str, JSONValue]]:
    records: list[dict[str, JSONValue]] = []
    for scope in _work_unit_scopes(plan):
        receipt = verify_serial_work_unit(
            root,
            binding=binding,
            work_unit_id=scope.work_unit_id,
        )
        records.extend(_records_from_receipt(receipt, plan, scope))
    if len(records) != plan.csv_shard_count:
        raise IntegrityError("verified receipt shard count differs from plan")
    return records


def _records_from_receipt(
    receipt: LoadedWorkUnitReceipt,
    plan: StructuralValidationPlan,
    scope: _WorkUnitScope,
) -> list[dict[str, JSONValue]]:
    identity = receipt.identity
    if (
        receipt.work_unit_id != scope.work_unit_id
        or receipt.family != scope.family
        or receipt.first_path_entity != scope.first
        or receipt.last_path_entity != scope.last
        or identity.get("shard_index") != scope.shard_index
        or identity.get("years") != plan.years
        or identity.get("expected_path_count") != scope.last - scope.first + 1
    ):
        raise IntegrityError("work-unit receipt differs from structural plan")
    siblings = identity.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, str | bytes):
        raise IntegrityError("work-unit receipt siblings are invalid")
    frequencies = _FAMILY_FREQUENCIES[scope.family]
    if len(siblings) != len(frequencies):
        raise IntegrityError("work-unit receipt sibling count differs from plan")
    records: list[dict[str, JSONValue]] = []
    for sibling, frequency in zip(siblings, frequencies, strict=True):
        if not isinstance(sibling, Mapping):
            raise IntegrityError("work-unit sibling receipt is invalid")
        checked_frequency = cast(CSVFrequency, frequency)
        spec = _DATASET_BY_FREQUENCY[checked_frequency]
        path_count = scope.last - scope.first + 1
        expected_rows = path_count * plan.years * spec.periods_per_year
        relative_path = _expected_csv_relative(scope, spec)
        expected_static: dict[str, JSONValue] = {
            "work_unit_id": scope.work_unit_id,
            "work_unit_receipt_fingerprint": receipt.fingerprint,
            "family": scope.family,
            "frequency": checked_frequency,
            "shard_index": scope.shard_index,
            "first_path_entity": scope.first,
            "last_path_entity": scope.last,
            "first_path_id": _path_id(scope.family, scope.first),
            "last_path_id": _path_id(scope.family, scope.last),
            "path_count": path_count,
            "years": plan.years,
            "periods_per_path": plan.years * spec.periods_per_year,
            "rows": expected_rows,
            "relative_path": relative_path,
        }
        for key, expected in (
            ("frequency", checked_frequency),
            ("relative_path", relative_path),
            ("rows", expected_rows),
        ):
            if sibling.get(key) != expected:
                raise IntegrityError(
                    f"work-unit sibling {key} differs from structural plan"
                )
        bytes_count = _positive_int(sibling.get("bytes"), "sibling bytes")
        sha256 = _require_fingerprint(sibling.get("sha256"), "sibling SHA-256")
        csv_receipt = _require_fingerprint(
            sibling.get("csv_receipt_fingerprint"), "CSV receipt"
        )
        records.append(
            {
                **expected_static,
                "bytes": bytes_count,
                "sha256": sha256,
                "csv_receipt_fingerprint": csv_receipt,
            }
        )
    return records


def _expected_csv_relative(scope: _WorkUnitScope, spec: _DatasetSpec) -> str:
    filename = (
        f"part-{scope.shard_index:05d}-path-"
        f"{_path_id(scope.family, scope.first)}-"
        f"{_path_id(scope.family, scope.last)}.csv"
    )
    return PurePosixPath(
        "consumer", "returns", spec.family_directory, spec.frequency, filename
    ).as_posix()


def _load_manifest(
    root: Path, plan: StructuralValidationPlan
) -> tuple[dict[str, Any], str]:
    path = root / "consumer" / "consumer-manifest.json"
    content = _read_small_regular_file(path, "consumer manifest", MAX_MANIFEST_BYTES)
    document = _parse_canonical_json(content, "consumer manifest")
    expected_keys = {
        "schema_id",
        "schema_version",
        "calendar_id",
        "csv_schema_version",
        "csv_serializer_version",
        "plan_fingerprint",
        "storage_binding",
        "datasets",
        "work_unit_count",
        "csv_shard_count",
        "total_rows",
        "total_values",
        "shards",
    }
    if set(document) != expected_keys:
        raise IntegrityError("consumer manifest keys are invalid")
    if (
        document["schema_id"] != CONSUMER_MANIFEST_SCHEMA_ID
        or document["schema_version"] != CONSUMER_MANIFEST_SCHEMA_VERSION
        or document["calendar_id"] != CALENDAR_ID
        or document["csv_schema_version"] != CSV_SCHEMA_VERSION
        or document["csv_serializer_version"] != CSV_SERIALIZER_VERSION
        or document["plan_fingerprint"] != plan.fingerprint
        or document["datasets"] != _dataset_summaries(plan)
        or document["work_unit_count"] != plan.work_unit_count
        or document["csv_shard_count"] != plan.csv_shard_count
        or document["total_rows"] != plan.total_rows
        or document["total_values"] != plan.total_values
    ):
        raise IntegrityError("consumer manifest geometry or schema differs")
    _manifest_binding(document)
    records = _manifest_records(document)
    if len(records) != plan.csv_shard_count:
        raise IntegrityError("consumer manifest shard count differs from plan")
    _validate_manifest_record_shapes(records, plan)
    return document, hashlib.sha256(content).hexdigest()


def _manifest_binding(document: Mapping[str, Any]) -> StorageCommitBinding:
    value = document.get("storage_binding")
    if not isinstance(value, dict) or set(value) != {
        "simulation_fingerprint",
        "materialization_fingerprint",
        "toolchain_fingerprint",
        "run_fingerprint",
    }:
        raise IntegrityError("consumer manifest storage binding is invalid")
    try:
        return StorageCommitBinding(**value)
    except (GenerationError, TypeError) as error:
        raise IntegrityError("consumer manifest storage binding is invalid") from error


def _manifest_records(document: Mapping[str, Any]) -> list[dict[str, JSONValue]]:
    value = document.get("shards")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise IntegrityError("consumer manifest shards are invalid")
    return cast(list[dict[str, JSONValue]], value)


def _validate_manifest_record_shapes(
    records: list[dict[str, JSONValue]], plan: StructuralValidationPlan
) -> None:
    expected_keys = {
        "work_unit_id",
        "work_unit_receipt_fingerprint",
        "family",
        "frequency",
        "shard_index",
        "first_path_entity",
        "last_path_entity",
        "first_path_id",
        "last_path_id",
        "path_count",
        "years",
        "periods_per_path",
        "rows",
        "relative_path",
        "bytes",
        "sha256",
        "csv_receipt_fingerprint",
    }
    offset = 0
    for scope in _work_unit_scopes(plan):
        for frequency in _FAMILY_FREQUENCIES[scope.family]:
            record = records[offset]
            offset += 1
            if set(record) != expected_keys:
                raise IntegrityError("consumer manifest shard record keys are invalid")
            spec = _DATASET_BY_FREQUENCY[frequency]
            count = scope.last - scope.first + 1
            expected_static: Mapping[str, JSONValue] = {
                "work_unit_id": scope.work_unit_id,
                "family": scope.family,
                "frequency": frequency,
                "shard_index": scope.shard_index,
                "first_path_entity": scope.first,
                "last_path_entity": scope.last,
                "first_path_id": _path_id(scope.family, scope.first),
                "last_path_id": _path_id(scope.family, scope.last),
                "path_count": count,
                "years": plan.years,
                "periods_per_path": plan.years * spec.periods_per_year,
                "rows": count * plan.years * spec.periods_per_year,
                "relative_path": _expected_csv_relative(scope, spec),
            }
            if any(record.get(key) != value for key, value in expected_static.items()):
                raise IntegrityError("consumer manifest shard geometry differs")
            _positive_int(record.get("bytes"), "manifest shard bytes")
            _require_fingerprint(record.get("sha256"), "manifest shard SHA-256")
            _require_fingerprint(
                record.get("csv_receipt_fingerprint"), "manifest CSV receipt"
            )
            _require_fingerprint(
                record.get("work_unit_receipt_fingerprint"),
                "manifest work-unit receipt",
            )


def _verify_receipt_allowlist(root: Path, plan: StructuralValidationPlan) -> None:
    directory = root / "receipts" / "storage"
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityError("storage receipt directory is missing or unsafe")
    expected = {f"{scope.work_unit_id}.json" for scope in _work_unit_scopes(plan)}
    try:
        actual: set[str] = set()
        for child in directory.iterdir():
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise IntegrityError("storage receipt tree has a non-regular entry")
            actual.add(child.name)
    except OSError as error:
        raise IntegrityError("storage receipt directory cannot be scanned") from error
    if actual != expected:
        raise IntegrityError("storage receipt file allowlist differs from plan")


def _verify_consumer_allowlist(root: Path, records_value: object) -> None:
    if not isinstance(records_value, Sequence) or isinstance(
        records_value, str | bytes
    ):
        raise IntegrityError("consumer shard records are invalid")
    records = cast(Sequence[Mapping[str, Any]], records_value)
    expected_files = {"consumer/consumer-manifest.json"}
    for record in records:
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise IntegrityError("consumer manifest relative path is invalid")
        expected_files.add(relative)
    expected_directories = {"consumer"}
    for relative in expected_files:
        current = PurePosixPath(relative).parent
        while current.as_posix() not in {".", ""}:
            expected_directories.add(current.as_posix())
            current = current.parent
    consumer = root / "consumer"
    if consumer.is_symlink() or not consumer.is_dir():
        raise IntegrityError("consumer directory is missing or unsafe")
    actual_files: set[str] = set()
    actual_directories = {"consumer"}
    try:
        for child in consumer.rglob("*"):
            relative = child.relative_to(root).as_posix()
            metadata = child.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                if child.is_symlink():
                    raise IntegrityError("consumer tree contains a symbolic link")
                actual_directories.add(relative)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                actual_files.add(relative)
            else:
                raise IntegrityError("consumer tree contains an unsafe entry")
    except OSError as error:
        raise IntegrityError("consumer tree cannot be scanned") from error
    if actual_files != expected_files or actual_directories != expected_directories:
        raise IntegrityError("consumer file or directory allowlist differs from plan")


class _CSVShardStream:
    """Bounded line reader for one exact receipt-bound CSV shard."""

    def __init__(self, root: Path, record: Mapping[str, JSONValue]) -> None:
        relative = record["relative_path"]
        if not isinstance(relative, str):  # pragma: no cover - manifest normalized
            raise IntegrityError("CSV shard relative path is invalid")
        self.record = record
        self.path = root / relative
        self.family = cast(Family, record["family"])
        self.frequency = cast(CSVFrequency, record["frequency"])
        self.periods_per_path = cast(int, record["periods_per_path"])
        self.expected_rows = cast(int, record["rows"])
        self.expected_bytes = cast(int, record["bytes"])
        self.expected_sha256 = cast(str, record["sha256"])
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._rows = 0
        self._finished = False
        descriptor, metadata = _open_regular_file(self.path, "CSV shard")
        self._initial_stat = metadata
        self._handle = os.fdopen(descriptor, "rb", closefd=True)
        try:
            header = self._handle.readline(len(CSV_HEADER_BYTES) + 1)
            self._digest.update(header)
            self._bytes += len(header)
            if header != CSV_HEADER_BYTES:
                raise IntegrityError("CSV shard header or line ending is invalid")
        except BaseException:
            self._handle.close()
            raise

    def read_path(self, path_entity: int) -> FloatArray:
        if self._finished:
            raise IntegrityError("CSV shard was read after finalization")
        values = np.empty((self.periods_per_path, ASSET_COUNT), dtype=np.float64)
        path_id = _path_id(self.family, path_entity)
        periods_per_year = _DATASET_BY_FREQUENCY[self.frequency].periods_per_year
        for zero_index in range(self.periods_per_path):
            line = self._handle.readline(MAX_CSV_ROW_BYTES + 1)
            if not line:
                raise IntegrityError("CSV shard is truncated")
            if len(line) > MAX_CSV_ROW_BYTES:
                raise IntegrityError("CSV shard row exceeds its grammar bound")
            self._digest.update(line)
            self._bytes += len(line)
            if not line.endswith(b"\n") or b"\r" in line:
                raise IntegrityError("CSV shard row line ending is invalid")
            try:
                fields = line[:-1].decode("utf-8").split(",")
            except UnicodeDecodeError as error:
                raise IntegrityError("CSV shard row is not UTF-8") from error
            period_index = zero_index + 1
            simulation_year = zero_index // periods_per_year + 1
            period_in_year = zero_index % periods_per_year + 1
            if len(fields) != 7 or fields[:4] != [
                path_id,
                str(period_index),
                str(simulation_year),
                str(period_in_year),
            ]:
                raise IntegrityError(
                    "CSV shard row key is duplicated, missing, or noncanonical"
                )
            for asset, text in enumerate(fields[4:]):
                try:
                    parsed = float(text)
                    canonical = format_csv_float64(parsed)
                except (GenerationError, ValueError) as error:
                    raise IntegrityError(
                        "CSV shard return is not finite decimal"
                    ) from error
                if canonical != text:
                    raise IntegrityError(
                        "CSV shard return has signed zero or noncanonical text"
                    )
                values[zero_index, asset] = parsed
            self._rows += 1
        values.flags.writeable = False
        return values

    def finish(self) -> ShardValidationSummary:
        if self._finished:
            raise IntegrityError("CSV shard was finalized twice")
        self._finished = True
        try:
            if self._handle.read(1):
                raise IntegrityError("CSV shard contains excess rows or bytes")
            final = os.fstat(self._handle.fileno())
            current = self.path.stat(follow_symlinks=False)
            initial = self._initial_stat
            if (
                not stat.S_ISREG(final.st_mode)
                or final.st_nlink != 1
                or current.st_dev != initial.st_dev
                or current.st_ino != initial.st_ino
                or current.st_size != initial.st_size
                or current.st_mtime_ns != initial.st_mtime_ns
                or final.st_dev != initial.st_dev
                or final.st_ino != initial.st_ino
                or final.st_size != initial.st_size
                or final.st_mtime_ns != initial.st_mtime_ns
            ):
                raise IntegrityError("CSV shard changed during structural scan")
            digest = self._digest.hexdigest()
            if (
                self._rows != self.expected_rows
                or self._bytes != self.expected_bytes
                or digest != self.expected_sha256
            ):
                raise IntegrityError("CSV shard row/byte/hash receipt differs")
            return ShardValidationSummary(
                relative_path=cast(str, self.record["relative_path"]),
                rows=self._rows,
                bytes=self._bytes,
                sha256=digest,
            )
        finally:
            self._handle.close()

    def close(self) -> None:
        self._handle.close()


def _scan_work_unit(
    root: Path,
    *,
    plan: StructuralValidationPlan,
    scope: _WorkUnitScope,
    records: Sequence[Mapping[str, JSONValue]],
    hooks: tuple[StructuralMetricHook, ...],
    errors: _ErrorAccumulator,
) -> tuple[ShardValidationSummary, ...]:
    streams = tuple(_CSVShardStream(root, record) for record in records)
    summaries: list[ShardValidationSummary] = []
    try:
        for entity in range(scope.first, scope.last + 1):
            values = tuple(stream.read_path(entity) for stream in streams)
            if scope.family == "LF":
                if len(values) != 3:  # pragma: no cover - registry invariant
                    raise AssertionError("LF structural scan has wrong siblings")
                _validate_lf_path(values[0], values[1], values[2], plan.years, errors)
            else:
                if len(values) != 2:  # pragma: no cover - registry invariant
                    raise AssertionError("HF structural scan has wrong siblings")
                _validate_hf_path(values[0], values[1], plan.years, errors)
            for stream, path_values in zip(streams, values, strict=True):
                _observe_hooks(
                    hooks,
                    family=scope.family,
                    frequency=stream.frequency,
                    entity=entity,
                    values=path_values,
                )
        for stream in streams:
            summaries.append(stream.finish())
        return tuple(summaries)
    finally:
        for stream in streams:
            stream.close()


def _validate_lf_path(
    monthly: FloatArray,
    quarterly: FloatArray,
    annual: FloatArray,
    years: int,
    errors: _ErrorAccumulator,
) -> None:
    _validate_path_numerics(monthly, errors)
    _validate_path_numerics(quarterly, errors)
    _validate_path_numerics(annual, errors)
    monthly_years = monthly.reshape(years, 12, ASSET_COUNT)
    quarters_grouped = monthly_years.reshape(years, 4, 3, ASSET_COUNT)
    expected_quarterly = quarters_grouped.sum(axis=2).reshape(-1, ASSET_COUNT)
    quarterly_years = quarterly.reshape(years, 4, ASSET_COUNT)
    expected_annual_from_quarters = quarterly_years.sum(axis=1)
    expected_annual_from_months = monthly_years.sum(axis=1)
    _compare_log_aggregates(expected_quarterly, quarterly, errors)
    _compare_log_aggregates(expected_annual_from_quarters, annual, errors)
    _compare_log_aggregates(expected_annual_from_months, annual, errors)
    _compare_simple_compounding(
        quarters_grouped.reshape(-1, 3, ASSET_COUNT), quarterly, errors
    )
    _compare_simple_compounding(monthly_years, annual, errors)
    _compare_simple_compounding(quarterly_years, annual, errors)


def _validate_hf_path(
    daily: FloatArray,
    weekly: FloatArray,
    years: int,
    errors: _ErrorAccumulator,
) -> None:
    _validate_path_numerics(daily, errors)
    _validate_path_numerics(weekly, errors)
    starts = np.fromiter(
        (
            year * 252 + WEEK_BOUNDARIES[week]
            for year in range(years)
            for week in range(52)
        ),
        dtype=np.int64,
        count=years * 52,
    )
    weekly_expected = np.add.reduceat(daily, starts, axis=0)
    _compare_log_aggregates(weekly_expected, weekly, errors)
    with np.errstate(over="ignore", invalid="ignore"):
        simple_factors = 1.0 + np.expm1(daily)
        compounded = np.multiply.reduceat(simple_factors, starts, axis=0) - 1.0
    _compare_compounded_simple(compounded, weekly, errors)


def _validate_path_numerics(values: FloatArray, errors: _ErrorAccumulator) -> None:
    if values.dtype != np.dtype(np.float64) or not bool(np.isfinite(values).all()):
        raise IntegrityError("CSV path is not finite float64")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        simple = np.expm1(values)
        recovered = np.log1p(simple)
        wealth = np.cumsum(values, axis=0)
    if (
        not bool(np.isfinite(simple).all())
        or bool(np.any(simple <= -1.0))
        or not bool(np.isfinite(recovered).all())
        or not bool(np.isfinite(wealth).all())
    ):
        raise IntegrityError("CSV path simple returns or cumulative wealth are invalid")
    difference = np.abs(recovered - values)
    maximum = float(np.max(difference, initial=0.0))
    if maximum > SIMPLE_RETURN_TOLERANCE:
        raise IntegrityError("CSV log/simple round-trip exceeds tolerance")
    errors.roundtrip_values += values.size
    errors.max_roundtrip = max(errors.max_roundtrip, maximum)


def _compare_log_aggregates(
    expected: FloatArray, actual: FloatArray, errors: _ErrorAccumulator
) -> None:
    difference = np.abs(expected - actual)
    maximum = float(np.max(difference, initial=0.0))
    if not math.isfinite(maximum) or maximum > LOG_AGGREGATION_TOLERANCE:
        raise IntegrityError("cross-frequency log sum exceeds tolerance")
    errors.aggregation_values += actual.size
    errors.max_log = max(errors.max_log, maximum)


def _compare_simple_compounding(
    constituents: FloatArray,
    actual_log: FloatArray,
    errors: _ErrorAccumulator,
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        compounded = np.prod(1.0 + np.expm1(constituents), axis=1) - 1.0
    _compare_compounded_simple(compounded, actual_log, errors)


def _compare_compounded_simple(
    compounded: FloatArray,
    actual_log: FloatArray,
    errors: _ErrorAccumulator,
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        actual_simple = np.expm1(actual_log)
    difference = np.abs(compounded - actual_simple)
    maximum = float(np.max(difference, initial=0.0))
    if (
        not bool(np.isfinite(compounded).all())
        or not bool(np.isfinite(actual_simple).all())
        or not math.isfinite(maximum)
        or maximum > SIMPLE_RETURN_TOLERANCE
    ):
        raise IntegrityError("cross-frequency simple compounding exceeds tolerance")
    errors.aggregation_values += actual_log.size
    errors.max_simple = max(errors.max_simple, maximum)


def _validated_hooks(
    hooks: Sequence[StructuralMetricHook],
) -> tuple[StructuralMetricHook, ...]:
    result = tuple(hooks)
    names: set[str] = set()
    for hook in result:
        name = getattr(hook, "name", None)
        observe = getattr(hook, "observe_path", None)
        finalize = getattr(hook, "finalize", None)
        if (
            not isinstance(name, str)
            or _HOOK_NAME.fullmatch(name) is None
            or name in names
            or not callable(observe)
            or not callable(finalize)
        ):
            raise GenerationError("structural metric hook contract is invalid")
        names.add(name)
    return result


def _observe_hooks(
    hooks: tuple[StructuralMetricHook, ...],
    *,
    family: Family,
    frequency: CSVFrequency,
    entity: int,
    values: FloatArray,
) -> None:
    if not hooks:
        return
    view = StructuralPathView(
        family=family,
        frequency=frequency,
        path_entity=entity,
        path_id=_path_id(family, entity),
        log_returns=values,
    )
    for hook in hooks:
        hook.observe_path(view)


def _finalize_hooks(
    hooks: tuple[StructuralMetricHook, ...],
) -> Mapping[str, JSONValue]:
    metrics: dict[str, JSONValue] = {}
    for hook in hooks:
        payload = hook.finalize()
        if not isinstance(payload, Mapping):
            raise IntegrityError("structural metric hook payload is not a mapping")
        checked = _validated_json_value(dict(payload), f"hook {hook.name}")
        if not isinstance(checked, dict):  # pragma: no cover - mapping input
            raise AssertionError("validated hook payload is not a dictionary")
        metrics[hook.name] = checked
    return cast(Mapping[str, JSONValue], _json_freeze(metrics))


def _mint_report(
    *,
    plan: StructuralValidationPlan,
    manifest_sha256: str,
    shard_summaries: tuple[ShardValidationSummary, ...],
    errors: _ErrorAccumulator,
    elapsed_seconds: float,
    hook_metrics: Mapping[str, JSONValue],
) -> StructuralValidationReport:
    datasets = tuple(
        DatasetValidationSummary(
            family=cast(Family, item["family"]),
            frequency=cast(CSVFrequency, item["frequency"]),
            paths=cast(int, item["paths"]),
            periods_per_path=cast(int, item["periods_per_path"]),
            rows=cast(int, item["rows"]),
            values=cast(int, item["values"]),
            shards=cast(int, item["shards"]),
        )
        for item in _dataset_summaries(plan)
    )
    rows_per_second = plan.total_rows / elapsed_seconds
    result = object.__new__(StructuralValidationReport)
    for name, value in (
        ("schema_id", STRUCTURAL_REPORT_SCHEMA_ID),
        ("schema_version", STRUCTURAL_REPORT_SCHEMA_VERSION),
        ("plan_fingerprint", plan.fingerprint),
        ("canonical_profile", plan == CANONICAL_STRUCTURAL_PLAN),
        ("manifest_sha256", manifest_sha256),
        ("total_rows", plan.total_rows),
        ("total_values", plan.total_values),
        ("work_units", plan.work_unit_count),
        ("csv_shards", plan.csv_shard_count),
        ("aggregation_values_checked", errors.aggregation_values),
        ("roundtrip_values_checked", errors.roundtrip_values),
        ("max_abs_log_aggregation_error", errors.max_log),
        ("max_abs_simple_compounding_error", errors.max_simple),
        ("max_abs_log_simple_roundtrip_error", errors.max_roundtrip),
        ("elapsed_seconds", elapsed_seconds),
        ("rows_per_second", rows_per_second),
        ("datasets", datasets),
        ("shards", shard_summaries),
        ("hook_metrics", hook_metrics),
        ("_seal", _REPORT_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    object.__setattr__(result, "fingerprint", _sha256_json(result.identity()))
    return result


def _verify_report(value: object) -> None:
    if type(value) is not StructuralValidationReport:
        raise GenerationError("structural report was not minted by the validator")
    report = value
    if (
        report._seal is not _REPORT_SEAL
        or report._owner_id != id(report)
        or report.schema_id != STRUCTURAL_REPORT_SCHEMA_ID
        or report.schema_version != STRUCTURAL_REPORT_SCHEMA_VERSION
        or _FINGERPRINT.fullmatch(report.plan_fingerprint) is None
        or _FINGERPRINT.fullmatch(report.manifest_sha256) is None
        or not isinstance(report.canonical_profile, bool)
        or report.total_rows <= 0
        or report.total_values != report.total_rows * ASSET_COUNT
        or report.work_units <= 0
        or report.csv_shards <= 0
        or len(report.shards) != report.csv_shards
        or len(report.datasets) != len(_DATASETS)
        or sum(item.rows for item in report.datasets) != report.total_rows
        or sum(item.values for item in report.datasets) != report.total_values
        or sum(item.shards for item in report.datasets) != report.csv_shards
        or report.roundtrip_values_checked != report.total_values
        or report.aggregation_values_checked <= 0
        or not math.isfinite(report.elapsed_seconds)
        or report.elapsed_seconds <= 0.0
        or not math.isfinite(report.rows_per_second)
        or report.rows_per_second <= 0.0
    ):
        raise IntegrityError("structural report fields are invalid")
    for number, tolerance in (
        (report.max_abs_log_aggregation_error, LOG_AGGREGATION_TOLERANCE),
        (report.max_abs_simple_compounding_error, SIMPLE_RETURN_TOLERANCE),
        (report.max_abs_log_simple_roundtrip_error, SIMPLE_RETURN_TOLERANCE),
    ):
        if not math.isfinite(number) or not 0.0 <= number <= tolerance:
            raise IntegrityError("structural report tolerance result is invalid")
    for shard in report.shards:
        if (
            not isinstance(shard.relative_path, str)
            or shard.rows <= 0
            or shard.bytes <= len(CSV_HEADER_BYTES)
            or _FINGERPRINT.fullmatch(shard.sha256) is None
        ):
            raise IntegrityError("structural report shard result is invalid")
    if report.hook_metrics != _json_freeze(
        _validated_json_value(
            _json_thaw(report.hook_metrics), "structural report hook metrics"
        )
    ):
        raise IntegrityError("structural report hook metrics are invalid")
    if report.fingerprint != _sha256_json(report.identity()):
        raise IntegrityError("structural report fingerprint is invalid")


def _report_document_bytes(report: StructuralValidationReport) -> bytes:
    _verify_report(report)
    return _canonical_bytes(
        {
            "schema_version": STRUCTURAL_REPORT_SCHEMA_VERSION,
            "report_fingerprint": report.fingerprint,
            "identity": report.identity(),
        }
    )


def _require_clean_validation_boundary(root: Path) -> None:
    forbidden = (
        "RUNNING",
        "INTERRUPTED",
        "FAILED",
        "DATA_VALIDATED",
        "COMPLETE",
        "RELEASED",
    )
    if any((root / name).exists() or (root / name).is_symlink() for name in forbidden):
        raise IntegrityError("run lifecycle is not clean for DATA_VALIDATED")
    report = root / "validation" / "structural-report.json"
    if report.exists() or report.is_symlink():
        raise IntegrityError("structural report destination already exists")


def _existing_safe_root(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("run root must be a path")
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise GenerationError("run root must be absolute and traversal-free")
    try:
        if root.is_symlink() or not root.is_dir():
            raise IntegrityError("run root is missing or unsafe")
        for component in (root, *root.parents):
            if component.is_symlink():
                raise IntegrityError("run root contains a symbolic-link component")
    except OSError as error:
        raise IntegrityError("run root cannot be inspected") from error
    return root


def _open_regular_file(path: Path, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise IntegrityError(f"{label} cannot be opened") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise IntegrityError(f"{label} is not a regular single-link file")
    return descriptor, metadata


def _read_small_regular_file(path: Path, label: str, maximum: int) -> bytes:
    descriptor, initial = _open_regular_file(path, label)
    try:
        if initial.st_size > maximum:
            raise IntegrityError(f"{label} exceeds its size bound")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            content = handle.read(maximum + 1)
            if len(content) > maximum:
                raise IntegrityError(f"{label} exceeds its size bound")
            final = os.fstat(handle.fileno())
            current = path.stat(follow_symlinks=False)
            if (
                final.st_dev != initial.st_dev
                or final.st_ino != initial.st_ino
                or final.st_size != initial.st_size
                or final.st_mtime_ns != initial.st_mtime_ns
                or current.st_dev != initial.st_dev
                or current.st_ino != initial.st_ino
                or current.st_size != initial.st_size
                or current.st_mtime_ns != initial.st_mtime_ns
                or current.st_nlink != 1
            ):
                raise IntegrityError(f"{label} changed during read")
            return content
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def _parse_canonical_json(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} is invalid JSON") from error
    try:
        canonical = _canonical_bytes(value)
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"{label} contains noncanonical JSON") from error
    if not isinstance(value, dict) or content != canonical:
        raise IntegrityError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], value)


def _publish_no_replace(path: Path, content: bytes) -> None:
    parent = path.parent
    if parent.exists() or parent.is_symlink():
        if parent.is_symlink() or not parent.is_dir():
            raise IntegrityError("publication directory is unsafe")
    else:
        try:
            parent.mkdir(parents=True, mode=0o700)
        except OSError as error:
            raise IntegrityError("publication directory cannot be created") from error
    if path.exists() or path.is_symlink():
        raise IntegrityError("publication destination already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            written = handle.write(content)
            if written != len(content):
                raise IntegrityError("publication write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise IntegrityError(
                "publication destination appeared concurrently"
            ) from error
        except OSError as error:
            raise IntegrityError("publication cannot be linked atomically") from error
        _fsync_directory(parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise IntegrityError("publication directory cannot be fsynced") from error


def _require_plan(value: object) -> StructuralValidationPlan:
    if not isinstance(value, StructuralValidationPlan):
        raise GenerationError("structural validation plan is invalid")
    if value.fingerprint != _sha256_json(value.identity()):
        raise IntegrityError("structural validation plan fingerprint is invalid")
    return value


def _require_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise GenerationError(f"{label} fingerprint is invalid")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntegrityError(f"{label} must be a positive integer")
    return value


def _path_id(family: Family | CSVFamily, entity: int) -> str:
    return f"{family}{entity:06d}"


def _ceiling_divide(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _validated_json_value(value: object, label: str) -> JSONValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IntegrityError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_validated_json_value(item, label) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise IntegrityError(f"{label} contains a non-string key")
        return {
            key: _validated_json_value(item, label)
            for key, item in sorted(value.items())
        }
    raise IntegrityError(f"{label} is not canonical-JSON compatible")


def _json_freeze(value: JSONValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _json_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_json_freeze(item) for item in value)
    return value


def _json_thaw(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {
            str(key): _json_thaw(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_json_thaw(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise IntegrityError("frozen JSON value contains an unsupported type")
