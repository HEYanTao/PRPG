"""Deterministic multicore generation of canonical LF/HF CSV work units.

The numerical generator remains :mod:`prpg.simulation.serial`; this module is
the operational Phase-6 coordinator around it.  Each spawned worker reloads
the exact durable G3/G5 authority once, constructs a process-local sealed
serial input, owns complete LF or HF sibling shards, and publishes only through
the existing group-commit boundary.  The parent receives small receipt
summaries and independently re-verifies every durable receipt.

The coordinator intentionally keeps the lifecycle small.  A new run has one
``RUNNING`` marker and an atomic ``run-state.json``.  A clean interruption is
``INTERRUPTED`` and may resume only with the identical identity.  Unexpected
errors are ``FAILED`` and terminal.  Valid completed work-unit receipts are
the resume checkpoints; partial sibling sets are quarantined before the whole
work unit is regenerated.  Scientific failures and G5 policy selection are
outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import stat
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
import psutil
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from prpg.config import PRPGConfig, load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.errors import GenerationError, IntegrityError
from prpg.execution import resource_snapshot
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_evidence_store import G3EvidenceStore
from prpg.model.g5_evidence_store import G5EvidenceStore
from prpg.model.g5_generation_authority import G5QualifiedProductionAuthority
from prpg.model.input import load_calibration_input
from prpg.model.lean_g5_generation_authority import (
    LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID,
    LeanG5ProductionAuthority,
    LeanG5ProductionAuthorityStore,
)
from prpg.model.run_store import CalibrationRunStore
from prpg.provenance import inspect_executable_source
from prpg.simulation.aggregation import (
    aggregate_hf_weekly_log_returns,
    aggregate_lf_log_returns,
)
from prpg.simulation.serial import (
    SERIAL_GENERATOR_VERSION,
    SerialGenerationInput,
    build_serial_generation_input,
    generate_configured_serial_base_path,
    is_sealed_serial_generation_input,
)
from prpg.storage.csv_v1 import CSV_SCHEMA_VERSION, CSV_SERIALIZER_VERSION
from prpg.storage.group_commit import (
    LoadedWorkUnitReceipt,
    StorageCommitBinding,
    commit_hf_serial_work_unit,
    commit_lf_serial_work_unit,
    quarantine_serial_work_unit,
    serial_work_unit_id,
    verify_serial_work_unit,
)

PRODUCTION_PLAN_SCHEMA_ID: Final = "prpg-production-plan-v1"
PRODUCTION_LOADER_SCHEMA_ID: Final = "prpg-production-loader-v1"
PRODUCTION_BINDING_SCHEMA_ID: Final = "prpg-production-binding-v1"
PRODUCTION_RUN_STATE_SCHEMA_ID: Final = "prpg-production-run-state-v1"
PRODUCTION_RUN_STATE_SCHEMA_VERSION: Final = 1
PRODUCTION_RESULT_SCHEMA_ID: Final = "prpg-production-result-v1"
CSV_ROW_ESTIMATE_BYTES: Final = 128
RECEIPT_OVERHEAD_BYTES_PER_WORK_UNIT: Final = 64 * 1024
WORKER_MEMORY_ESTIMATE_BYTES: Final = 512 * 1024 * 1024
PARENT_MEMORY_ESTIMATE_BYTES: Final = 1024 * 1024 * 1024
DISK_SAFETY_MULTIPLIER: Final = 3
MAX_MEMORY_FRACTION: Final = 0.70
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_WORKER_CACHE: tuple[str, SerialGenerationInput] | None = None

Family: TypeAlias = Literal["LF", "HF"]
RunDisposition: TypeAlias = Literal[
    "not_started", "running", "interrupted", "failed", "generated"
]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


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
class ProductionPlan:
    """Exact horizon, path counts, and complete-sibling shard ownership."""

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
            raise GenerationError("production plan profile_id is invalid")
        for name in (
            "years",
            "lf_paths",
            "hf_paths",
            "lf_paths_per_shard",
            "hf_paths_per_shard",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GenerationError(f"production plan {name} must be positive")
        if self.lf_paths > 999_999 or self.hf_paths > 999_999:
            raise GenerationError("production path count exceeds the CSV namespace")
        if self.lf_paths_per_shard > self.lf_paths:
            raise GenerationError("LF shard size exceeds its path count")
        if self.hf_paths_per_shard > self.hf_paths:
            raise GenerationError("HF shard size exceeds its path count")
        object.__setattr__(self, "fingerprint", _sha256_json(self.identity()))

    @classmethod
    def from_config(cls, config: PRPGConfig) -> ProductionPlan:
        """Resolve the production geometry from one validated configuration."""

        if not isinstance(config, PRPGConfig):
            raise GenerationError("production plan requires a validated config")
        return cls(
            profile_id=(
                "canonical-50y-v1"
                if (
                    config.simulation.low_frequency.paths == 5_000
                    and config.simulation.high_frequency.paths == 1_000
                    and config.simulation.low_frequency.years == 50
                    and config.export.low_paths_per_shard == 100
                    and config.export.high_paths_per_shard == 50
                )
                else "configured-noncanonical-v1"
            ),
            years=config.simulation.low_frequency.years,
            lf_paths=config.simulation.low_frequency.paths,
            hf_paths=config.simulation.high_frequency.paths,
            lf_paths_per_shard=config.export.low_paths_per_shard,
            hf_paths_per_shard=config.export.high_paths_per_shard,
        )

    def identity(self) -> dict[str, JSONValue]:
        return {
            "schema_id": PRODUCTION_PLAN_SCHEMA_ID,
            "profile_id": self.profile_id,
            "years": self.years,
            "lf_paths": self.lf_paths,
            "hf_paths": self.hf_paths,
            "lf_paths_per_shard": self.lf_paths_per_shard,
            "hf_paths_per_shard": self.hf_paths_per_shard,
        }

    @property
    def canonical(self) -> bool:
        return self == CANONICAL_PRODUCTION_PLAN

    @property
    def total_rows(self) -> int:
        return self.lf_paths * self.years * 17 + self.hf_paths * self.years * 304

    @property
    def total_values(self) -> int:
        return self.total_rows * 3

    @property
    def work_unit_count(self) -> int:
        return _ceil_div(self.lf_paths, self.lf_paths_per_shard) + _ceil_div(
            self.hf_paths, self.hf_paths_per_shard
        )

    @property
    def csv_shard_count(self) -> int:
        return 3 * _ceil_div(self.lf_paths, self.lf_paths_per_shard) + 2 * _ceil_div(
            self.hf_paths, self.hf_paths_per_shard
        )


CANONICAL_PRODUCTION_PLAN: Final = ProductionPlan(
    profile_id="canonical-50y-v1",
    years=50,
    lf_paths=5_000,
    hf_paths=1_000,
    lf_paths_per_shard=100,
    hf_paths_per_shard=50,
)


@dataclass(frozen=True, slots=True)
class ProductionWorkUnit:
    """One complete sibling-set scope assigned to exactly one worker."""

    family: Family
    shard_index: int
    first_path_entity: int
    last_path_entity: int
    work_unit_id: str

    @property
    def path_count(self) -> int:
        return self.last_path_entity - self.first_path_entity + 1


@dataclass(frozen=True, slots=True)
class ProductionLoaderSpec:
    """Pickle-safe identity from which each worker reloads sealed authority."""

    project_root: str
    config_path: str
    processed_data_fingerprint: str
    g3_evidence_fingerprint: str
    g5_authority_fingerprint: str
    config_fingerprint: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        project = _absolute_directory(self.project_root, "project root")
        config = _absolute_regular_file(self.config_path, "configuration")
        if config.parent != project / "configs" and project not in config.parents:
            raise GenerationError("configuration must be inside the project root")
        for name in (
            "processed_data_fingerprint",
            "g3_evidence_fingerprint",
            "g5_authority_fingerprint",
            "config_fingerprint",
        ):
            _required_fingerprint(getattr(self, name), name.replace("_", " "))
        object.__setattr__(self, "project_root", str(project))
        object.__setattr__(self, "config_path", str(config))
        object.__setattr__(self, "fingerprint", _sha256_json(self.identity()))

    def identity(self) -> dict[str, JSONValue]:
        return {
            "schema_id": PRODUCTION_LOADER_SCHEMA_ID,
            "project_root": self.project_root,
            "config_path": self.config_path,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "g3_evidence_fingerprint": self.g3_evidence_fingerprint,
            "g5_authority_fingerprint": self.g5_authority_fingerprint,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ProductionBinding:
    """Four public content identities plus their storage binding."""

    data_fingerprint: str
    simulation_fingerprint: str
    qualification_fingerprint: str
    materialization_fingerprint: str
    toolchain_fingerprint: str
    run_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "data_fingerprint",
            "simulation_fingerprint",
            "qualification_fingerprint",
            "materialization_fingerprint",
            "toolchain_fingerprint",
            "run_fingerprint",
        ):
            _required_fingerprint(getattr(self, name), name.replace("_", " "))

    @property
    def storage(self) -> StorageCommitBinding:
        return StorageCommitBinding(
            simulation_fingerprint=self.simulation_fingerprint,
            materialization_fingerprint=self.materialization_fingerprint,
            toolchain_fingerprint=self.toolchain_fingerprint,
            run_fingerprint=self.run_fingerprint,
        )

    def identity(self) -> dict[str, JSONValue]:
        return {
            "schema_id": PRODUCTION_BINDING_SCHEMA_ID,
            "data_fingerprint": self.data_fingerprint,
            "simulation_fingerprint": self.simulation_fingerprint,
            "qualification_fingerprint": self.qualification_fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "run_fingerprint": self.run_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ProductionResourceEstimate:
    """Conservative preflight and measured-throughput projection."""

    workers: int
    logical_cpu_count: int
    physical_memory_bytes: int
    estimated_peak_memory_bytes: int
    estimated_memory_fraction: float
    final_output_bytes_estimate: int
    required_free_disk_bytes: int
    disk_free_bytes: int
    projected_wall_seconds: float | None
    memory_passed: bool
    disk_passed: bool
    time_projection_available: bool

    @property
    def passed(self) -> bool:
        return self.memory_passed and self.disk_passed


@dataclass(frozen=True, slots=True)
class ProductionWorkResult:
    """Small primitive result returned from a worker to the coordinator."""

    work_unit_id: str
    family: Family
    shard_index: int
    first_path_entity: int
    last_path_entity: int
    receipt_fingerprint: str
    sibling_sha256: tuple[str, ...]
    receipt_duration_seconds: float
    worker_identity: str


@dataclass(frozen=True, slots=True)
class ProductionExecutionResult:
    """Parent-verified generated result; this is not a G7 or release pass."""

    run_root: Path
    plan_fingerprint: str
    binding: ProductionBinding
    workers: int
    generated_work_units: int
    reused_work_units: int
    work_unit_results: tuple[ProductionWorkResult, ...]
    wall_seconds: float
    rows_per_second: float
    peak_process_tree_rss_bytes: int
    peak_memory_fraction: float
    swap_used_before_bytes: int | None
    swap_used_after_bytes: int | None
    source_code_fingerprint: str
    dependency_lock_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class LoadedProductionRun:
    """Verified durable identity of a completed generation-stage run.

    This is the small fresh-process hand-off from G6 generation to G7
    validation.  It deliberately does not reconstruct every worker result in
    memory; instead it rechecks the frozen run documents and every durable
    work-unit receipt before returning the loader, plan, and storage binding
    needed by the validators.
    """

    run_root: Path
    loader: ProductionLoaderSpec
    plan: ProductionPlan
    binding: ProductionBinding
    workers: int
    result_fingerprint: str
    source_code_fingerprint: str
    dependency_lock_fingerprint: str


@dataclass(frozen=True, slots=True)
class ProductionReplayReport:
    """Byte-parity result for one representative LF and HF clean replay."""

    source_run_fingerprint: str
    replay_run_fingerprint: str
    source_result_fingerprint: str
    work_unit_ids: tuple[str, str]
    csv_shards_compared: int
    source_code_fingerprint: str
    dependency_lock_fingerprint: str
    elapsed_seconds: float
    identical: bool
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ProductionDeterminismReport:
    """Observable CSV-byte comparison for identical work-unit scopes."""

    left_run_fingerprint: str
    right_run_fingerprint: str
    work_units_compared: int
    csv_shards_compared: int
    identical: bool
    mismatches: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkObservation:
    """One same-scope worker-count timing and resource observation."""

    workers: int
    run_root: Path
    run_fingerprint: str
    result_fingerprint: str
    wall_seconds: float
    rows_per_second: float
    parallel_efficiency: float
    peak_process_tree_rss_bytes: int
    csv_parity_with_one_worker: bool


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkReport:
    """Measured G6 worker-count comparison and canonical wall projection."""

    benchmark_plan_fingerprint: str
    worker_counts: tuple[int, ...]
    observations: tuple[ProductionBenchmarkObservation, ...]
    selected_workers: int
    selected_rows_per_second: float
    canonical_projected_wall_seconds: float
    canonical_projected_wall_hours: float
    all_csv_parity_passed: bool
    minimum_parallel_efficiency: float
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _WorkerTask:
    loader: ProductionLoaderSpec
    output_root: str
    binding: StorageCommitBinding
    scope: ProductionWorkUnit


def production_work_units(plan: ProductionPlan) -> tuple[ProductionWorkUnit, ...]:
    """Return every LF then HF scope in canonical deterministic order."""

    if not isinstance(plan, ProductionPlan):
        raise GenerationError("production work units require a ProductionPlan")
    result: list[ProductionWorkUnit] = []
    for family, count, shard_size in (
        ("LF", plan.lf_paths, plan.lf_paths_per_shard),
        ("HF", plan.hf_paths, plan.hf_paths_per_shard),
    ):
        for shard_index, first in enumerate(range(1, count + 1, shard_size)):
            last = min(count, first + shard_size - 1)
            checked_family = cast(Family, family)
            result.append(
                ProductionWorkUnit(
                    family=checked_family,
                    shard_index=shard_index,
                    first_path_entity=first,
                    last_path_entity=last,
                    work_unit_id=serial_work_unit_id(
                        checked_family, shard_index, first, last
                    ),
                )
            )
    if len(result) != plan.work_unit_count:
        raise AssertionError("production work-unit arithmetic is inconsistent")
    return tuple(result)


def build_production_loader_spec(
    *,
    project_root: str | Path,
    config_path: str | Path,
    processed_data_fingerprint: str,
    g3_evidence_fingerprint: str,
    g5_authority_fingerprint: str,
) -> ProductionLoaderSpec:
    """Build and verify the portable durable-loader identity."""

    project = Path(project_root).expanduser().resolve(strict=True)
    config_file = Path(config_path).expanduser().resolve(strict=True)
    config = load_config(config_file)
    return ProductionLoaderSpec(
        project_root=str(project),
        config_path=str(config_file),
        processed_data_fingerprint=processed_data_fingerprint,
        g3_evidence_fingerprint=g3_evidence_fingerprint,
        g5_authority_fingerprint=g5_authority_fingerprint,
        config_fingerprint=config.fingerprint(),
    )


def load_production_generation_input(
    loader: ProductionLoaderSpec,
) -> SerialGenerationInput:
    """Reload all durable parents and mint one process-local serial input."""

    if not isinstance(loader, ProductionLoaderSpec):
        raise GenerationError("production loader has the wrong type")
    if loader.fingerprint != _sha256_json(loader.identity()):
        raise IntegrityError("production loader fingerprint is invalid")
    project_root = _absolute_directory(loader.project_root, "project root")
    config_path = _absolute_regular_file(loader.config_path, "configuration")
    config = load_config(config_path)
    if config.fingerprint() != loader.config_fingerprint:
        raise IntegrityError("production configuration changed after loader freeze")
    artifact_root = _resolve_project_path(project_root, config.data.artifact_root)
    calibration_run_root = _resolve_project_path(project_root, config.run.output_root)
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(artifact_root),
        loader.processed_data_fingerprint,
        calibration_config_fingerprint=loader.config_fingerprint,
        holdout_months=config.model.design_holdout_months,
    )
    g3_evidence = G3EvidenceStore(artifact_root).load(
        loader.g3_evidence_fingerprint,
        run_store=CalibrationRunStore(calibration_run_root),
        checkpoint_store=CalibrationCheckpointStore(artifact_root),
        model_store=ModelArtifactStore(artifact_root),
    )
    lean_store = LeanG5ProductionAuthorityStore(artifact_root)
    authority: G5QualifiedProductionAuthority | LeanG5ProductionAuthority
    if lean_store.contains(loader.g5_authority_fingerprint):
        authority = lean_store.load(
            loader.g5_authority_fingerprint,
            g3_evidence=g3_evidence,
        )
    else:
        authority = G5EvidenceStore(artifact_root).load_final_authority(
            loader.g5_authority_fingerprint,
            g3_evidence=g3_evidence,
        )
    result = build_serial_generation_input(
        authority=authority,
        config=config,
        calibration_input=calibration_input,
    )
    if not is_sealed_serial_generation_input(result):
        raise IntegrityError("production serial mint returned an invalid input")
    return result


def build_production_binding(
    loader: ProductionLoaderSpec,
    *,
    plan: ProductionPlan,
    run_name: str,
) -> ProductionBinding:
    """Derive all generation/materialization identities before any CSV write."""

    if not isinstance(plan, ProductionPlan):
        raise GenerationError("production binding requires a ProductionPlan")
    name = _run_name(run_name)
    generation_input = load_production_generation_input(loader)
    source = inspect_executable_source(loader.project_root)
    toolchain_identity = {
        "schema_id": "prpg-production-toolchain-v1",
        "source_code_fingerprint": source.source_code_fingerprint,
        "dependency_lock_fingerprint": source.dependency_lock_fingerprint,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "csv_serializer_version": CSV_SERIALIZER_VERSION,
    }
    toolchain_fingerprint = _sha256_json(toolchain_identity)
    simulation_identity = {
        "schema_id": "prpg-simulation-fingerprint-v1",
        "data_fingerprint": loader.processed_data_fingerprint,
        "g3_evidence_fingerprint": loader.g3_evidence_fingerprint,
        "g5_authority_fingerprint": loader.g5_authority_fingerprint,
        "generation_input_fingerprint": generation_input.fingerprint,
        "production_artifact_fingerprint": (
            generation_input.production_artifact_fingerprint
        ),
        "generation_policy": generation_input.generation_policy.as_dict(),
        "master_seed": generation_input.master_seed,
        "scientific_version": generation_input.scientific_version,
        "serial_generator_version": SERIAL_GENERATOR_VERSION,
        "toolchain_fingerprint": toolchain_fingerprint,
    }
    simulation_fingerprint = _sha256_json(simulation_identity)
    if (
        generation_input.g5_authority_schema_id
        == LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID
    ):
        qualification_identity = {
            "schema_id": "prpg-lean-qualification-fingerprint-v1",
            "simulation_fingerprint": simulation_fingerprint,
            "g5_authority_fingerprint": loader.g5_authority_fingerprint,
            "qualification_permit_fingerprint": (
                generation_input.g5_threshold_registry_fingerprint
            ),
            "assessment_fingerprint": (
                generation_input.g5_qualification_results_fingerprint
            ),
        }
    else:
        qualification_identity = {
            "schema_id": "prpg-qualification-fingerprint-v1",
            "simulation_fingerprint": simulation_fingerprint,
            "g5_authority_fingerprint": loader.g5_authority_fingerprint,
            "threshold_registry_fingerprint": (
                generation_input.g5_threshold_registry_fingerprint
            ),
            "qualification_results_fingerprint": (
                generation_input.g5_qualification_results_fingerprint
            ),
            "meta_validation_fingerprint": (
                generation_input.g5_meta_validation_fingerprint
            ),
        }
    qualification_fingerprint = _sha256_json(qualification_identity)
    materialization_fingerprint = _sha256_json(
        {
            "schema_id": "prpg-materialization-fingerprint-v1",
            "simulation_fingerprint": simulation_fingerprint,
            "plan": plan.identity(),
            "csv_schema_version": CSV_SCHEMA_VERSION,
            "csv_serializer_version": CSV_SERIALIZER_VERSION,
            "compression": "none",
            "newline": "LF",
        }
    )
    run_fingerprint = _sha256_json(
        {
            "schema_id": "prpg-production-run-fingerprint-v1",
            "run_name": name,
            "simulation_fingerprint": simulation_fingerprint,
            "qualification_fingerprint": qualification_fingerprint,
            "materialization_fingerprint": materialization_fingerprint,
            "plan_fingerprint": plan.fingerprint,
        }
    )
    return ProductionBinding(
        data_fingerprint=loader.processed_data_fingerprint,
        simulation_fingerprint=simulation_fingerprint,
        qualification_fingerprint=qualification_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
        toolchain_fingerprint=toolchain_fingerprint,
        run_fingerprint=run_fingerprint,
    )


def production_resource_estimate(
    *,
    plan: ProductionPlan,
    config: PRPGConfig,
    output_root: str | Path,
    measured_rows_per_second: float | None = None,
    requested_workers: int | None = None,
) -> ProductionResourceEstimate:
    """Fail-closed RAM/disk preflight and optional wall-time projection."""

    if not isinstance(plan, ProductionPlan) or not isinstance(config, PRPGConfig):
        raise GenerationError("production preflight requires typed plan and config")
    resources = resource_snapshot(
        output_root,
        requested_workers=(
            config.execution.workers if requested_workers is None else requested_workers
        ),
        reserve_cores=config.execution.reserve_physical_cores,
    )
    peak = PARENT_MEMORY_ESTIMATE_BYTES + (
        resources.resolved_workers * WORKER_MEMORY_ESTIMATE_BYTES
    )
    memory_fraction = peak / resources.physical_memory_bytes
    final_bytes = (
        plan.total_rows * CSV_ROW_ESTIMATE_BYTES
        + plan.work_unit_count * RECEIPT_OVERHEAD_BYTES_PER_WORK_UNIT
    )
    required_disk = DISK_SAFETY_MULTIPLIER * final_bytes
    projection: float | None = None
    if measured_rows_per_second is not None:
        if (
            isinstance(measured_rows_per_second, bool)
            or not isinstance(measured_rows_per_second, int | float)
            or not math.isfinite(float(measured_rows_per_second))
            or float(measured_rows_per_second) <= 0.0
        ):
            raise GenerationError("measured rows/second must be finite and positive")
        projection = plan.total_rows / float(measured_rows_per_second)
    return ProductionResourceEstimate(
        workers=resources.resolved_workers,
        logical_cpu_count=resources.logical_cpu_count,
        physical_memory_bytes=resources.physical_memory_bytes,
        estimated_peak_memory_bytes=peak,
        estimated_memory_fraction=memory_fraction,
        final_output_bytes_estimate=final_bytes,
        required_free_disk_bytes=required_disk,
        disk_free_bytes=resources.disk_free_bytes,
        projected_wall_seconds=projection,
        memory_passed=(
            memory_fraction
            <= min(config.execution.memory_fraction_limit, MAX_MEMORY_FRACTION)
        ),
        disk_passed=resources.disk_free_bytes >= required_disk,
        time_projection_available=projection is not None,
    )


def execute_serial_work_unit(
    generation_input: SerialGenerationInput,
    *,
    output_root: str | Path,
    binding: StorageCommitBinding,
    scope: ProductionWorkUnit,
    worker_identity: str,
) -> ProductionWorkResult:
    """Generate, aggregate, commit, and reverify one complete sibling group."""

    if not is_sealed_serial_generation_input(generation_input):
        raise GenerationError("work unit requires a sealed production input")
    if not isinstance(binding, StorageCommitBinding):
        raise GenerationError("work unit requires a storage binding")
    _validate_scope(scope, generation_input)
    bases = tuple(
        generate_configured_serial_base_path(
            generation_input,
            family=scope.family,
            path_entity=entity,
        )
        for entity in range(scope.first_path_entity, scope.last_path_entity + 1)
    )
    if scope.family == "LF":
        aggregates = tuple(aggregate_lf_log_returns(path) for path in bases)
        receipt = commit_lf_serial_work_unit(
            output_root,
            binding=binding,
            work_unit_id=scope.work_unit_id,
            worker_identity=worker_identity,
            shard_index=scope.shard_index,
            first_path_entity=scope.first_path_entity,
            last_path_entity=scope.last_path_entity,
            expected_path_count=scope.path_count,
            base_paths=bases,
            aggregates=aggregates,
        )
    else:
        weekly = tuple(aggregate_hf_weekly_log_returns(path) for path in bases)
        receipt = commit_hf_serial_work_unit(
            output_root,
            binding=binding,
            work_unit_id=scope.work_unit_id,
            worker_identity=worker_identity,
            shard_index=scope.shard_index,
            first_path_entity=scope.first_path_entity,
            last_path_entity=scope.last_path_entity,
            expected_path_count=scope.path_count,
            base_paths=bases,
            weekly_returns=weekly,
        )
    verified = verify_serial_work_unit(
        Path(output_root), binding=binding, work_unit_id=scope.work_unit_id
    )
    if verified.fingerprint != receipt.fingerprint:
        raise IntegrityError("worker receipt changed during immediate verification")
    return _work_result_from_receipt(verified, scope)


def _production_worker(task: _WorkerTask) -> ProductionWorkResult:
    global _WORKER_CACHE
    if not isinstance(task, _WorkerTask):
        raise GenerationError("production worker task has the wrong type")
    if _WORKER_CACHE is None or _WORKER_CACHE[0] != task.loader.fingerprint:
        _WORKER_CACHE = (
            task.loader.fingerprint,
            load_production_generation_input(task.loader),
        )
    generation_input = _WORKER_CACHE[1]
    return execute_serial_work_unit(
        generation_input,
        output_root=task.output_root,
        binding=task.binding,
        scope=task.scope,
        worker_identity=f"spawn-{os.getpid()}",
    )


def run_production_generation(
    *,
    loader: ProductionLoaderSpec,
    plan: ProductionPlan,
    binding: ProductionBinding,
    output_root: str | Path,
    run_name: str,
    workers: int,
    resume: bool = False,
    measured_rows_per_second: float | None = None,
) -> ProductionExecutionResult:
    """Generate every planned work unit and independently verify its receipt.

    This operation stops at the generated staging boundary.  It does not write
    a consumer manifest, ``DATA_VALIDATED``, ``COMPLETE``, or ``RELEASED``.
    """

    if not isinstance(loader, ProductionLoaderSpec):
        raise GenerationError("production run requires a loader spec")
    if not isinstance(plan, ProductionPlan) or not isinstance(
        binding, ProductionBinding
    ):
        raise GenerationError("production run requires typed plan and binding")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise GenerationError("production workers must be a positive integer")
    checked_run_name = _run_name(run_name)
    root = _absolute_output_path(output_root)
    config = load_config(loader.config_path)
    _validate_plan_against_config(plan, config)
    estimate = production_resource_estimate(
        plan=plan,
        config=config,
        output_root=root,
        measured_rows_per_second=measured_rows_per_second,
        requested_workers=workers,
    )
    if estimate.workers != workers:
        raise GenerationError(
            "production worker count exceeds the resource-safe resolved count",
            details={
                "requested_workers": workers,
                "resolved_workers": estimate.workers,
                "logical_cpu_count": estimate.logical_cpu_count,
            },
        )
    if not estimate.passed:
        raise GenerationError(
            "production resource preflight failed",
            details={
                "workers": workers,
                "estimated_peak_memory_bytes": estimate.estimated_peak_memory_bytes,
                "physical_memory_bytes": estimate.physical_memory_bytes,
                "estimated_memory_fraction": estimate.estimated_memory_fraction,
                "required_free_disk_bytes": estimate.required_free_disk_bytes,
                "disk_free_bytes": estimate.disk_free_bytes,
            },
        )
    _verify_production_binding(
        loader,
        plan=plan,
        binding=binding,
        run_name=checked_run_name,
    )
    source_before = inspect_executable_source(loader.project_root)
    scopes = production_work_units(plan)
    started = time.monotonic()
    with _exclusive_run_lock(root, create=not resume):
        state = _prepare_run_state(
            root,
            loader=loader,
            plan=plan,
            binding=binding,
            workers=workers,
            resume=resume,
            estimate=estimate,
            source_code_fingerprint=source_before.source_code_fingerprint,
            dependency_lock_fingerprint=source_before.dependency_lock_fingerprint,
        )
        reused: list[ProductionWorkResult] = []
        pending: list[ProductionWorkUnit] = []
        try:
            for scope in scopes:
                receipt_path = (
                    root / "receipts" / "storage" / (f"{scope.work_unit_id}.json")
                )
                if receipt_path.exists() or receipt_path.is_symlink():
                    if not resume:
                        raise IntegrityError(
                            "fresh production root contains an existing receipt"
                        )
                    receipt = verify_serial_work_unit(
                        root,
                        binding=binding.storage,
                        work_unit_id=scope.work_unit_id,
                    )
                    reused.append(_work_result_from_receipt(receipt, scope))
                    continue
                if _scope_has_canonical_sibling(root, scope):
                    quarantine_serial_work_unit(
                        root,
                        work_unit_id=scope.work_unit_id,
                        family=scope.family,
                        shard_index=scope.shard_index,
                        first_path_entity=scope.first_path_entity,
                        last_path_entity=scope.last_path_entity,
                    )
                pending.append(scope)
            tasks = tuple(
                _WorkerTask(loader, str(root), binding.storage, scope)
                for scope in pending
            )
            sampler = _ProductionResourceSampler(0.10)
            sampler.start()
            try:
                generated = _run_worker_tasks(tasks, workers=workers)
            finally:
                sampler.stop()
            ordered = {item.work_unit_id: item for item in (*reused, *generated)}
            if len(ordered) != len(scopes):
                raise IntegrityError("production results are missing or duplicated")
            verified_results: list[ProductionWorkResult] = []
            for scope in scopes:
                receipt = verify_serial_work_unit(
                    root,
                    binding=binding.storage,
                    work_unit_id=scope.work_unit_id,
                )
                verified = _work_result_from_receipt(receipt, scope)
                observed = ordered.get(scope.work_unit_id)
                if observed is None or (
                    observed.receipt_fingerprint != verified.receipt_fingerprint
                    or observed.sibling_sha256 != verified.sibling_sha256
                ):
                    raise IntegrityError(
                        "parent receipt verification differs from worker result"
                    )
                verified_results.append(verified)
            source_after = inspect_executable_source(loader.project_root)
            if source_after != source_before:
                raise IntegrityError("executable source changed during production")
            wall_seconds = time.monotonic() - started
            if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
                raise IntegrityError("production wall-clock measurement is invalid")
            rows_per_second = plan.total_rows / wall_seconds
            result = _mint_execution_result(
                root=root,
                plan=plan,
                binding=binding,
                workers=workers,
                generated_count=len(generated),
                reused_count=len(reused),
                work_results=tuple(verified_results),
                wall_seconds=wall_seconds,
                rows_per_second=rows_per_second,
                sampler=sampler,
                source_code_fingerprint=source_before.source_code_fingerprint,
                dependency_lock_fingerprint=(source_before.dependency_lock_fingerprint),
            )
            measured_memory_limit = min(
                config.execution.memory_fraction_limit, MAX_MEMORY_FRACTION
            )
            if (
                result.peak_process_tree_rss_bytes <= 0
                or not math.isfinite(result.peak_memory_fraction)
                or result.peak_memory_fraction > measured_memory_limit
            ):
                raise GenerationError(
                    "production measured-memory gate failed",
                    details={
                        "peak_process_tree_rss_bytes": (
                            result.peak_process_tree_rss_bytes
                        ),
                        "peak_memory_fraction": result.peak_memory_fraction,
                        "maximum_memory_fraction": measured_memory_limit,
                    },
                )
            _finish_generated_state(root, state, result)
            return result
        except KeyboardInterrupt:
            _transition_run_state(
                root,
                state,
                disposition="interrupted",
                failure={"type": "KeyboardInterrupt", "message": "interrupted"},
            )
            raise
        except BaseException as error:
            _transition_run_state(
                root,
                state,
                disposition="failed",
                failure={"type": type(error).__name__, "message": str(error)[:1000]},
            )
            raise


def load_generated_production_run(output_root: str | Path) -> LoadedProductionRun:
    """Reload and verify a completed generation run for downstream gates."""

    root = _absolute_existing_directory(output_root, "production output root")
    state = _read_canonical_json(root / "run-state.json", "production run state")
    if (
        state.get("schema_id") != PRODUCTION_RUN_STATE_SCHEMA_ID
        or state.get("schema_version") != PRODUCTION_RUN_STATE_SCHEMA_VERSION
        or state.get("disposition") != "generated"
        or state.get("failure") is not None
    ):
        raise IntegrityError("production run is not in the generated disposition")

    loader_value = state.get("loader")
    plan_value = state.get("plan")
    binding_value = state.get("binding")
    if not isinstance(loader_value, Mapping):
        raise IntegrityError("production run loader identity is invalid")
    if not isinstance(plan_value, Mapping):
        raise IntegrityError("production run plan identity is invalid")
    if not isinstance(binding_value, Mapping):
        raise IntegrityError("production run binding identity is invalid")

    loader_keys = {
        "schema_id",
        "project_root",
        "config_path",
        "processed_data_fingerprint",
        "g3_evidence_fingerprint",
        "g5_authority_fingerprint",
        "config_fingerprint",
    }
    if set(loader_value) != loader_keys or loader_value.get("schema_id") != (
        PRODUCTION_LOADER_SCHEMA_ID
    ):
        raise IntegrityError("production run loader identity is invalid")
    loader = ProductionLoaderSpec(
        project_root=cast(str, loader_value["project_root"]),
        config_path=cast(str, loader_value["config_path"]),
        processed_data_fingerprint=cast(
            str, loader_value["processed_data_fingerprint"]
        ),
        g3_evidence_fingerprint=cast(str, loader_value["g3_evidence_fingerprint"]),
        g5_authority_fingerprint=cast(str, loader_value["g5_authority_fingerprint"]),
        config_fingerprint=cast(str, loader_value["config_fingerprint"]),
    )
    if loader.identity() != dict(loader_value):
        raise IntegrityError("production run loader identity changed")

    plan_keys = {
        "schema_id",
        "profile_id",
        "years",
        "lf_paths",
        "hf_paths",
        "lf_paths_per_shard",
        "hf_paths_per_shard",
    }
    if set(plan_value) != plan_keys or plan_value.get("schema_id") != (
        PRODUCTION_PLAN_SCHEMA_ID
    ):
        raise IntegrityError("production run plan identity is invalid")
    plan = ProductionPlan(
        profile_id=cast(str, plan_value["profile_id"]),
        years=cast(int, plan_value["years"]),
        lf_paths=cast(int, plan_value["lf_paths"]),
        hf_paths=cast(int, plan_value["hf_paths"]),
        lf_paths_per_shard=cast(int, plan_value["lf_paths_per_shard"]),
        hf_paths_per_shard=cast(int, plan_value["hf_paths_per_shard"]),
    )
    if plan.identity() != dict(plan_value):
        raise IntegrityError("production run plan identity changed")

    binding_keys = {
        "schema_id",
        "data_fingerprint",
        "simulation_fingerprint",
        "qualification_fingerprint",
        "materialization_fingerprint",
        "toolchain_fingerprint",
        "run_fingerprint",
    }
    if set(binding_value) != binding_keys or binding_value.get("schema_id") != (
        PRODUCTION_BINDING_SCHEMA_ID
    ):
        raise IntegrityError("production run binding identity is invalid")
    binding = ProductionBinding(
        data_fingerprint=cast(str, binding_value["data_fingerprint"]),
        simulation_fingerprint=cast(str, binding_value["simulation_fingerprint"]),
        qualification_fingerprint=cast(str, binding_value["qualification_fingerprint"]),
        materialization_fingerprint=cast(
            str, binding_value["materialization_fingerprint"]
        ),
        toolchain_fingerprint=cast(str, binding_value["toolchain_fingerprint"]),
        run_fingerprint=cast(str, binding_value["run_fingerprint"]),
    )
    if binding.identity() != dict(binding_value):
        raise IntegrityError("production run binding identity changed")

    result_fingerprint = _required_fingerprint(
        state.get("result_fingerprint"), "production result"
    )
    result_document = _read_canonical_json(
        root / "logs" / "generation-result.json", "production generation result"
    )
    identity = result_document.get("identity")
    if (
        result_document.get("schema_id") != PRODUCTION_RESULT_SCHEMA_ID
        or result_document.get("result_fingerprint") != result_fingerprint
        or not isinstance(identity, Mapping)
        or identity.get("schema_id") != PRODUCTION_RESULT_SCHEMA_ID
        or identity.get("plan_fingerprint") != plan.fingerprint
        or identity.get("binding") != binding.identity()
    ):
        raise IntegrityError("production generation result identity is invalid")
    if _sha256_json(dict(identity)) != result_fingerprint:
        raise IntegrityError("production generation result fingerprint changed")
    workers = identity.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise IntegrityError("production generation worker count is invalid")
    source_code_fingerprint = _required_fingerprint(
        identity.get("source_code_fingerprint"), "production source code"
    )
    dependency_lock_fingerprint = _required_fingerprint(
        identity.get("dependency_lock_fingerprint"), "production dependency lock"
    )

    _write_or_verify_frozen_run_inputs(
        root,
        loader=loader,
        binding=binding,
        create=False,
    )
    for scope in production_work_units(plan):
        verify_serial_work_unit(
            root,
            binding=binding.storage,
            work_unit_id=scope.work_unit_id,
        )
    return LoadedProductionRun(
        run_root=root,
        loader=loader,
        plan=plan,
        binding=binding,
        workers=workers,
        result_fingerprint=result_fingerprint,
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
    )


def replay_representative_work_units(
    source_run_root: str | Path,
    *,
    output_root: str | Path,
    run_name: str = "g8-clean-replay",
) -> ProductionReplayReport:
    """Regenerate the first complete LF/HF shards and require CSV byte parity."""

    source_run = load_generated_production_run(source_run_root)
    current_source = inspect_executable_source(source_run.loader.project_root)
    if (
        current_source.source_code_fingerprint != source_run.source_code_fingerprint
        or current_source.dependency_lock_fingerprint
        != source_run.dependency_lock_fingerprint
    ):
        raise IntegrityError("executable source changed since production generation")
    replay_root = _absolute_output_path(output_root)
    if replay_root.exists() or replay_root.is_symlink():
        raise GenerationError("clean replay output root already exists")
    replay_root.mkdir(parents=True, mode=0o700)
    replay_binding = rebind_production_run(
        source_run.binding,
        plan=source_run.plan,
        run_name=run_name,
    )
    generation_input = load_production_generation_input(source_run.loader)
    scopes = production_work_units(source_run.plan)
    lf_scope = next(item for item in scopes if item.family == "LF")
    hf_scope = next(item for item in scopes if item.family == "HF")
    selected = (lf_scope, hf_scope)
    started = time.monotonic()
    shard_count = 0
    for scope in selected:
        execute_serial_work_unit(
            generation_input,
            output_root=replay_root,
            binding=replay_binding.storage,
            scope=scope,
            worker_identity="clean-replay-parent",
        )
        source_receipt = verify_serial_work_unit(
            source_run.run_root,
            binding=source_run.binding.storage,
            work_unit_id=scope.work_unit_id,
        )
        replay_receipt = verify_serial_work_unit(
            replay_root,
            binding=replay_binding.storage,
            work_unit_id=scope.work_unit_id,
        )
        source_hashes = _receipt_sibling_hashes(source_receipt)
        replay_hashes = _receipt_sibling_hashes(replay_receipt)
        shard_count += len(source_hashes)
        if source_hashes != replay_hashes:
            raise IntegrityError("clean replay CSV hashes differ from production")
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise IntegrityError("clean replay wall-clock measurement is invalid")
    identity = {
        "schema_id": "prpg-production-clean-replay-v1",
        "source_run_fingerprint": source_run.binding.run_fingerprint,
        "replay_run_fingerprint": replay_binding.run_fingerprint,
        "source_result_fingerprint": source_run.result_fingerprint,
        "work_unit_ids": [item.work_unit_id for item in selected],
        "csv_shards_compared": shard_count,
        "source_code_fingerprint": source_run.source_code_fingerprint,
        "dependency_lock_fingerprint": source_run.dependency_lock_fingerprint,
        "elapsed_seconds": elapsed,
        "identical": True,
    }
    fingerprint = _sha256_json(identity)
    report = {**identity, "fingerprint": fingerprint}
    _publish_no_replace(
        replay_root / "clean-replay-report.json", _canonical_bytes(report)
    )
    return ProductionReplayReport(
        source_run_fingerprint=source_run.binding.run_fingerprint,
        replay_run_fingerprint=replay_binding.run_fingerprint,
        source_result_fingerprint=source_run.result_fingerprint,
        work_unit_ids=(lf_scope.work_unit_id, hf_scope.work_unit_id),
        csv_shards_compared=shard_count,
        source_code_fingerprint=source_run.source_code_fingerprint,
        dependency_lock_fingerprint=source_run.dependency_lock_fingerprint,
        elapsed_seconds=elapsed,
        identical=True,
        fingerprint=fingerprint,
    )


def compare_production_outputs(
    *,
    left_root: str | Path,
    right_root: str | Path,
    plan: ProductionPlan,
    left_binding: ProductionBinding,
    right_binding: ProductionBinding,
) -> ProductionDeterminismReport:
    """Compare exact CSV hashes while allowing operational receipt differences."""

    left = _absolute_existing_directory(left_root, "left production root")
    right = _absolute_existing_directory(right_root, "right production root")
    if (
        left_binding.simulation_fingerprint != right_binding.simulation_fingerprint
        or left_binding.materialization_fingerprint
        != right_binding.materialization_fingerprint
    ):
        raise GenerationError(
            "determinism comparison requires equal simulation/materialization"
        )
    mismatches: list[str] = []
    shard_count = 0
    for scope in production_work_units(plan):
        left_receipt = verify_serial_work_unit(
            left,
            binding=left_binding.storage,
            work_unit_id=scope.work_unit_id,
        )
        right_receipt = verify_serial_work_unit(
            right,
            binding=right_binding.storage,
            work_unit_id=scope.work_unit_id,
        )
        left_hashes = _receipt_sibling_hashes(left_receipt)
        right_hashes = _receipt_sibling_hashes(right_receipt)
        shard_count += len(left_hashes)
        if left_hashes != right_hashes:
            mismatches.append(scope.work_unit_id)
    identity = {
        "schema_id": "prpg-production-determinism-report-v1",
        "left_run_fingerprint": left_binding.run_fingerprint,
        "right_run_fingerprint": right_binding.run_fingerprint,
        "plan_fingerprint": plan.fingerprint,
        "work_units_compared": plan.work_unit_count,
        "csv_shards_compared": shard_count,
        "identical": not mismatches,
        "mismatches": mismatches,
    }
    return ProductionDeterminismReport(
        left_run_fingerprint=left_binding.run_fingerprint,
        right_run_fingerprint=right_binding.run_fingerprint,
        work_units_compared=plan.work_unit_count,
        csv_shards_compared=shard_count,
        identical=not mismatches,
        mismatches=tuple(mismatches),
        fingerprint=_sha256_json(identity),
    )


def benchmark_production_worker_counts(
    *,
    loader: ProductionLoaderSpec,
    plan: ProductionPlan,
    base_binding: ProductionBinding,
    output_parent: str | Path,
    worker_counts: Sequence[int] = (1, 2, 4, 9),
    run_name_prefix: str = "g6-benchmark",
) -> ProductionBenchmarkReport:
    """Run identical full-horizon scopes and compare exact CSV bytes."""

    counts = _validated_worker_counts(worker_counts)
    prefix = _run_name(run_name_prefix)
    parent = _absolute_output_path(output_parent)
    if parent.exists() or parent.is_symlink():
        raise GenerationError("benchmark output parent already exists")
    parent.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    results: list[tuple[ProductionExecutionResult, ProductionBinding]] = []
    for workers in counts:
        run_name = f"{prefix}-w{workers}"
        binding = rebind_production_run(
            base_binding,
            plan=plan,
            run_name=run_name,
        )
        root = parent / f"workers-{workers}"
        result = run_production_generation(
            loader=loader,
            plan=plan,
            binding=binding,
            output_root=root,
            run_name=run_name,
            workers=workers,
        )
        results.append((result, binding))
    baseline_result, baseline_binding = results[0]
    observations: list[ProductionBenchmarkObservation] = []
    for result, binding in results:
        parity = True
        if result is not baseline_result:
            parity = compare_production_outputs(
                left_root=baseline_result.run_root,
                right_root=result.run_root,
                plan=plan,
                left_binding=baseline_binding,
                right_binding=binding,
            ).identical
        efficiency = result.rows_per_second / (
            result.workers * baseline_result.rows_per_second
        )
        observations.append(
            ProductionBenchmarkObservation(
                workers=result.workers,
                run_root=result.run_root,
                run_fingerprint=binding.run_fingerprint,
                result_fingerprint=result.fingerprint,
                wall_seconds=result.wall_seconds,
                rows_per_second=result.rows_per_second,
                parallel_efficiency=efficiency,
                peak_process_tree_rss_bytes=result.peak_process_tree_rss_bytes,
                csv_parity_with_one_worker=parity,
            )
        )
    if not all(item.csv_parity_with_one_worker for item in observations):
        raise IntegrityError(
            "production benchmark CSV parity failed; worker selection withheld"
        )
    selected = max(observations, key=lambda item: item.rows_per_second)
    projection = CANONICAL_PRODUCTION_PLAN.total_rows / selected.rows_per_second
    identity = {
        "schema_id": "prpg-g6-production-benchmark-v1",
        "benchmark_plan_fingerprint": plan.fingerprint,
        "worker_counts": list(counts),
        "observations": [
            {
                "workers": item.workers,
                "run_fingerprint": item.run_fingerprint,
                "result_fingerprint": item.result_fingerprint,
                "wall_seconds": item.wall_seconds,
                "rows_per_second": item.rows_per_second,
                "parallel_efficiency": item.parallel_efficiency,
                "peak_process_tree_rss_bytes": item.peak_process_tree_rss_bytes,
                "csv_parity_with_one_worker": item.csv_parity_with_one_worker,
            }
            for item in observations
        ],
        "selected_workers": selected.workers,
        "selected_rows_per_second": selected.rows_per_second,
        "canonical_projected_wall_seconds": projection,
        "all_csv_parity_passed": all(
            item.csv_parity_with_one_worker for item in observations
        ),
        "minimum_parallel_efficiency": min(
            item.parallel_efficiency for item in observations
        ),
    }
    report = ProductionBenchmarkReport(
        benchmark_plan_fingerprint=plan.fingerprint,
        worker_counts=counts,
        observations=tuple(observations),
        selected_workers=selected.workers,
        selected_rows_per_second=selected.rows_per_second,
        canonical_projected_wall_seconds=projection,
        canonical_projected_wall_hours=projection / 3600.0,
        all_csv_parity_passed=all(
            item.csv_parity_with_one_worker for item in observations
        ),
        minimum_parallel_efficiency=min(
            item.parallel_efficiency for item in observations
        ),
        fingerprint=_sha256_json(identity),
    )
    _publish_no_replace(
        parent / "benchmark-report.json",
        _canonical_bytes({**identity, "fingerprint": report.fingerprint}),
    )
    return report


def rebind_production_run(
    binding: ProductionBinding, *, plan: ProductionPlan, run_name: str
) -> ProductionBinding:
    """Change only the operational run identity, never generated CSV bytes."""

    if not isinstance(binding, ProductionBinding) or not isinstance(
        plan, ProductionPlan
    ):
        raise GenerationError("production run rebinding requires typed inputs")
    name = _run_name(run_name)
    run_fingerprint = _sha256_json(
        {
            "schema_id": "prpg-production-run-fingerprint-v1",
            "run_name": name,
            "simulation_fingerprint": binding.simulation_fingerprint,
            "qualification_fingerprint": binding.qualification_fingerprint,
            "materialization_fingerprint": binding.materialization_fingerprint,
            "plan_fingerprint": plan.fingerprint,
        }
    )
    return ProductionBinding(
        data_fingerprint=binding.data_fingerprint,
        simulation_fingerprint=binding.simulation_fingerprint,
        qualification_fingerprint=binding.qualification_fingerprint,
        materialization_fingerprint=binding.materialization_fingerprint,
        toolchain_fingerprint=binding.toolchain_fingerprint,
        run_fingerprint=run_fingerprint,
    )


def _verify_production_binding(
    loader: ProductionLoaderSpec,
    *,
    plan: ProductionPlan,
    binding: ProductionBinding,
    run_name: str,
) -> None:
    """Rederive the complete provenance binding before any production write."""

    expected = build_production_binding(loader, plan=plan, run_name=run_name)
    if binding != expected:
        raise IntegrityError(
            "production binding differs from the frozen loader, plan, or run name"
        )


def _run_worker_tasks(
    tasks: tuple[_WorkerTask, ...], *, workers: int
) -> tuple[ProductionWorkResult, ...]:
    if not tasks:
        return ()
    if workers == 1:
        with threadpool_limits(limits=1):
            return tuple(_production_worker(task) for task in tasks)
    context = mp.get_context("spawn")
    with (
        _single_thread_environment(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_limit_worker_native_threads,
        ) as executor,
    ):
        return tuple(executor.map(_production_worker, tasks, chunksize=1))


class _ProductionResourceSampler:
    """Sparse parent-plus-child RSS and swap sampler for one generation stage."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes = 0
        self.swap_before_bytes = _swap_used_bytes()
        self.swap_after_bytes = self.swap_before_bytes

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
        self.swap_after_bytes = _swap_used_bytes()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            root = psutil.Process(os.getpid())
            processes = (root, *root.children(recursive=True))
        except (OSError, psutil.Error):
            return
        rss = 0
        for process in processes:
            try:
                rss += int(process.memory_info().rss)
            except (OSError, psutil.Error):
                continue
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)


def _prepare_run_state(
    root: Path,
    *,
    loader: ProductionLoaderSpec,
    plan: ProductionPlan,
    binding: ProductionBinding,
    workers: int,
    resume: bool,
    estimate: ProductionResourceEstimate,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
) -> dict[str, Any]:
    path = root / "run-state.json"
    if not resume:
        unexpected = tuple(
            child.name for child in root.iterdir() if child.name != ".run.lock"
        )
        if unexpected:
            raise GenerationError("fresh production root is not empty")
    _write_or_verify_frozen_run_inputs(
        root,
        loader=loader,
        binding=binding,
        create=not resume,
    )
    if resume:
        state = _read_canonical_json(path, "production run state")
        if (
            state.get("schema_id") != PRODUCTION_RUN_STATE_SCHEMA_ID
            or state.get("schema_version") != PRODUCTION_RUN_STATE_SCHEMA_VERSION
            or state.get("loader") != loader.identity()
            or state.get("plan") != plan.identity()
            or state.get("binding") != binding.identity()
            or state.get("source_code_fingerprint") != source_code_fingerprint
            or state.get("dependency_lock_fingerprint") != dependency_lock_fingerprint
        ):
            raise IntegrityError("resume state differs from frozen production inputs")
        disposition = state.get("disposition")
        if disposition not in {"running", "interrupted"}:
            raise GenerationError(
                "production resume requires interrupted or stale-running state"
            )
        for marker in ("RUNNING", "INTERRUPTED"):
            with suppress(FileNotFoundError):
                (root / marker).unlink()
        attempts = state.get("attempts")
        if not isinstance(attempts, list):
            raise IntegrityError("production run attempts are invalid")
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "workers": workers,
                "started_unix_ns": time.time_ns(),
                "resume": True,
            }
        )
        state["disposition"] = "running"
        state["failure"] = None
        state["resource_estimate"] = _resource_estimate_identity(estimate)
    else:
        state = {
            "schema_id": PRODUCTION_RUN_STATE_SCHEMA_ID,
            "schema_version": PRODUCTION_RUN_STATE_SCHEMA_VERSION,
            "disposition": "running",
            "loader": loader.identity(),
            "plan": plan.identity(),
            "binding": binding.identity(),
            "source_code_fingerprint": source_code_fingerprint,
            "dependency_lock_fingerprint": dependency_lock_fingerprint,
            "resource_estimate": _resource_estimate_identity(estimate),
            "attempts": [
                {
                    "attempt": 1,
                    "workers": workers,
                    "started_unix_ns": time.time_ns(),
                    "resume": False,
                }
            ],
            "result_fingerprint": None,
            "failure": None,
        }
    _atomic_replace_json(path, state)
    _publish_no_replace(
        root / "RUNNING",
        _canonical_bytes(
            {
                "schema_id": "prpg-production-running-v1",
                "run_fingerprint": binding.run_fingerprint,
            }
        ),
    )
    return state


def _finish_generated_state(
    root: Path, state: dict[str, Any], result: ProductionExecutionResult
) -> None:
    report = _execution_result_identity(result)
    _publish_no_replace(
        root / "logs" / "generation-result.json",
        _canonical_bytes(
            {
                "schema_id": PRODUCTION_RESULT_SCHEMA_ID,
                "result_fingerprint": result.fingerprint,
                "identity": report,
            }
        ),
    )
    with suppress(FileNotFoundError):
        (root / "RUNNING").unlink()
    state["disposition"] = "generated"
    state["result_fingerprint"] = result.fingerprint
    state["failure"] = None
    attempts = state.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        attempts[-1]["finished_unix_ns"] = time.time_ns()
        attempts[-1]["outcome"] = "generated"
    _atomic_replace_json(root / "run-state.json", state)


def _write_or_verify_frozen_run_inputs(
    root: Path,
    *,
    loader: ProductionLoaderSpec,
    binding: ProductionBinding,
    create: bool,
) -> None:
    config = load_config(loader.config_path)
    documents = {
        "config/resolved-config.json": {
            "schema_id": "prpg-resolved-production-config-v1",
            "config_fingerprint": loader.config_fingerprint,
            "config": config.model_dump(mode="json", exclude_none=False),
        },
        "data/data-manifest.json": {
            "schema_id": "prpg-production-data-binding-v1",
            "processed_data_fingerprint": loader.processed_data_fingerprint,
        },
        "model/model-manifest.json": {
            "schema_id": "prpg-production-model-binding-v1",
            "g3_evidence_fingerprint": loader.g3_evidence_fingerprint,
            "g5_authority_fingerprint": loader.g5_authority_fingerprint,
            "simulation_fingerprint": binding.simulation_fingerprint,
            "qualification_fingerprint": binding.qualification_fingerprint,
            "materialization_fingerprint": binding.materialization_fingerprint,
            "toolchain_fingerprint": binding.toolchain_fingerprint,
        },
    }
    for relative, document in documents.items():
        path = root / relative
        content = _canonical_bytes(document)
        if create:
            _publish_no_replace(path, content)
        else:
            try:
                observed = path.read_bytes()
            except OSError as error:
                raise IntegrityError("frozen production input is missing") from error
            if observed != content:
                raise IntegrityError("frozen production input changed before resume")


def _transition_run_state(
    root: Path,
    state: dict[str, Any],
    *,
    disposition: Literal["interrupted", "failed"],
    failure: dict[str, str],
) -> None:
    with suppress(FileNotFoundError):
        (root / "RUNNING").unlink()
    state["disposition"] = disposition
    state["failure"] = failure
    attempts = state.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        attempts[-1]["finished_unix_ns"] = time.time_ns()
        attempts[-1]["outcome"] = disposition
    with suppress(OSError, TypeError, ValueError):
        _atomic_replace_json(root / "run-state.json", state)
        _publish_no_replace(
            root / disposition.upper(),
            _canonical_bytes(
                {
                    "schema_id": f"prpg-production-{disposition}-v1",
                    "run_fingerprint": cast(Mapping[str, Any], state["binding"])[
                        "run_fingerprint"
                    ],
                    "failure": failure,
                }
            ),
        )


def _mint_execution_result(
    *,
    root: Path,
    plan: ProductionPlan,
    binding: ProductionBinding,
    workers: int,
    generated_count: int,
    reused_count: int,
    work_results: tuple[ProductionWorkResult, ...],
    wall_seconds: float,
    rows_per_second: float,
    sampler: _ProductionResourceSampler,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
) -> ProductionExecutionResult:
    physical_memory = int(psutil.virtual_memory().total)
    peak_fraction = sampler.peak_rss_bytes / physical_memory
    provisional = ProductionExecutionResult(
        run_root=root,
        plan_fingerprint=plan.fingerprint,
        binding=binding,
        workers=workers,
        generated_work_units=generated_count,
        reused_work_units=reused_count,
        work_unit_results=work_results,
        wall_seconds=wall_seconds,
        rows_per_second=rows_per_second,
        peak_process_tree_rss_bytes=sampler.peak_rss_bytes,
        peak_memory_fraction=peak_fraction,
        swap_used_before_bytes=sampler.swap_before_bytes,
        swap_used_after_bytes=sampler.swap_after_bytes,
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
        fingerprint="0" * 64,
    )
    fingerprint = _sha256_json(_execution_result_identity(provisional))
    return ProductionExecutionResult(
        run_root=provisional.run_root,
        plan_fingerprint=provisional.plan_fingerprint,
        binding=provisional.binding,
        workers=provisional.workers,
        generated_work_units=provisional.generated_work_units,
        reused_work_units=provisional.reused_work_units,
        work_unit_results=provisional.work_unit_results,
        wall_seconds=provisional.wall_seconds,
        rows_per_second=provisional.rows_per_second,
        peak_process_tree_rss_bytes=provisional.peak_process_tree_rss_bytes,
        peak_memory_fraction=provisional.peak_memory_fraction,
        swap_used_before_bytes=provisional.swap_used_before_bytes,
        swap_used_after_bytes=provisional.swap_used_after_bytes,
        source_code_fingerprint=provisional.source_code_fingerprint,
        dependency_lock_fingerprint=provisional.dependency_lock_fingerprint,
        fingerprint=fingerprint,
    )


def _execution_result_identity(
    result: ProductionExecutionResult,
) -> dict[str, JSONValue]:
    return {
        "schema_id": PRODUCTION_RESULT_SCHEMA_ID,
        "plan_fingerprint": result.plan_fingerprint,
        "binding": result.binding.identity(),
        "workers": result.workers,
        "generated_work_units": result.generated_work_units,
        "reused_work_units": result.reused_work_units,
        "wall_seconds": result.wall_seconds,
        "rows_per_second": result.rows_per_second,
        "peak_process_tree_rss_bytes": result.peak_process_tree_rss_bytes,
        "peak_memory_fraction": result.peak_memory_fraction,
        "swap_used_before_bytes": result.swap_used_before_bytes,
        "swap_used_after_bytes": result.swap_used_after_bytes,
        "source_code_fingerprint": result.source_code_fingerprint,
        "dependency_lock_fingerprint": result.dependency_lock_fingerprint,
        "work_units": [
            {
                "work_unit_id": item.work_unit_id,
                "family": item.family,
                "shard_index": item.shard_index,
                "first_path_entity": item.first_path_entity,
                "last_path_entity": item.last_path_entity,
                "receipt_fingerprint": item.receipt_fingerprint,
                "sibling_sha256": list(item.sibling_sha256),
                "receipt_duration_seconds": item.receipt_duration_seconds,
                "worker_identity": item.worker_identity,
            }
            for item in result.work_unit_results
        ],
    }


def _work_result_from_receipt(
    receipt: LoadedWorkUnitReceipt, scope: ProductionWorkUnit
) -> ProductionWorkResult:
    if (
        receipt.work_unit_id != scope.work_unit_id
        or receipt.family != scope.family
        or receipt.first_path_entity != scope.first_path_entity
        or receipt.last_path_entity != scope.last_path_entity
    ):
        raise IntegrityError("work-unit receipt differs from assigned scope")
    duration = receipt.duration_monotonic_ns / 1_000_000_000.0
    if not math.isfinite(duration) or duration < 0.0:
        raise IntegrityError("work-unit receipt duration is invalid")
    return ProductionWorkResult(
        work_unit_id=scope.work_unit_id,
        family=scope.family,
        shard_index=scope.shard_index,
        first_path_entity=scope.first_path_entity,
        last_path_entity=scope.last_path_entity,
        receipt_fingerprint=receipt.fingerprint,
        sibling_sha256=_receipt_sibling_hashes(receipt),
        receipt_duration_seconds=duration,
        worker_identity=receipt.worker_identity,
    )


def _receipt_sibling_hashes(receipt: LoadedWorkUnitReceipt) -> tuple[str, ...]:
    siblings = receipt.identity.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, str | bytes):
        raise IntegrityError("work-unit receipt sibling list is invalid")
    hashes: list[str] = []
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise IntegrityError("work-unit receipt sibling is invalid")
        hashes.append(_required_fingerprint(sibling.get("sha256"), "CSV SHA-256"))
    return tuple(hashes)


def _scope_has_canonical_sibling(root: Path, scope: ProductionWorkUnit) -> bool:
    return any(
        path.exists() or path.is_symlink() for path in _scope_sibling_paths(root, scope)
    )


def _scope_sibling_paths(root: Path, scope: ProductionWorkUnit) -> tuple[Path, ...]:
    items: tuple[tuple[str, str], ...]
    if scope.family == "LF":
        items = (("low", "monthly"), ("low", "quarterly"), ("low", "annual"))
    else:
        items = (("high", "daily"), ("high", "weekly"))
    first = f"{scope.family}{scope.first_path_entity:06d}"
    last = f"{scope.family}{scope.last_path_entity:06d}"
    filename = f"part-{scope.shard_index:05d}-path-{first}-{last}.csv"
    return tuple(
        root / "consumer" / "returns" / family_dir / frequency / filename
        for family_dir, frequency in items
    )


def _validate_scope(
    scope: ProductionWorkUnit, generation_input: SerialGenerationInput
) -> None:
    if not isinstance(scope, ProductionWorkUnit):
        raise GenerationError("production work-unit scope has the wrong type")
    maximum = (
        generation_input.configured_lf_paths
        if scope.family == "LF"
        else generation_input.configured_hf_paths
    )
    if (
        scope.shard_index < 0
        or scope.first_path_entity <= 0
        or scope.last_path_entity < scope.first_path_entity
        or scope.last_path_entity > maximum
        or scope.work_unit_id
        != serial_work_unit_id(
            scope.family,
            scope.shard_index,
            scope.first_path_entity,
            scope.last_path_entity,
        )
    ):
        raise GenerationError("production work-unit scope is invalid")


def _validate_plan_against_config(plan: ProductionPlan, config: PRPGConfig) -> None:
    if (
        plan.years != config.simulation.low_frequency.years
        or plan.lf_paths > config.simulation.low_frequency.paths
        or plan.hf_paths > config.simulation.high_frequency.paths
    ):
        raise GenerationError("production plan exceeds configured generation scope")
    if plan.canonical and plan != ProductionPlan.from_config(config):
        raise GenerationError("canonical production plan differs from configuration")


def _resource_estimate_identity(
    value: ProductionResourceEstimate,
) -> dict[str, JSONValue]:
    return {
        "workers": value.workers,
        "logical_cpu_count": value.logical_cpu_count,
        "physical_memory_bytes": value.physical_memory_bytes,
        "estimated_peak_memory_bytes": value.estimated_peak_memory_bytes,
        "estimated_memory_fraction": value.estimated_memory_fraction,
        "final_output_bytes_estimate": value.final_output_bytes_estimate,
        "required_free_disk_bytes": value.required_free_disk_bytes,
        "disk_free_bytes": value.disk_free_bytes,
        "projected_wall_seconds": value.projected_wall_seconds,
        "memory_passed": value.memory_passed,
        "disk_passed": value.disk_passed,
        "time_projection_available": value.time_projection_available,
    }


_NATIVE_THREAD_VARIABLES: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _limit_worker_native_threads() -> None:
    for name in _NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"
    threadpool_limits(limits=1)


@contextmanager
def _single_thread_environment() -> Any:
    previous = {name: os.environ.get(name) for name in _NATIVE_THREAD_VARIABLES}
    for name in _NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _exclusive_run_lock(root: Path, *, create: bool) -> Any:
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - supported macOS has fcntl
        raise GenerationError("production run locking is unavailable") from error
    if create:
        if root.exists() or root.is_symlink():
            raise GenerationError("fresh production output root already exists")
        try:
            root.mkdir(parents=True, mode=0o700)
        except OSError as error:
            raise GenerationError("production output root cannot be created") from error
    else:
        _absolute_existing_directory(root, "production output root")
    lock_path = root / ".run.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise IntegrityError("production run lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise GenerationError("production run is already locked") from error
        yield
    finally:
        if descriptor is not None:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(descriptor)
        with suppress(FileNotFoundError):
            lock_path.unlink()


def _publish_no_replace(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError(f"publication destination already exists: {path.name}")
    try:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                written = handle.write(content)
                if written != len(content):
                    raise IntegrityError("publication write was incomplete")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with suppress(FileNotFoundError):
                path.unlink()
            raise
        _fsync_directory(path.parent)
    except OSError as error:
        raise IntegrityError("publication failed") from error


def _atomic_replace_json(path: Path, value: object) -> None:
    content = _canonical_bytes(value)
    if path.is_symlink():
        raise IntegrityError("run-state destination is a symbolic link")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            written = handle.write(content)
            if written != len(content):
                raise IntegrityError("run-state write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise IntegrityError("run-state atomic update failed") from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > 16 * 1024 * 1024
        ):
            raise IntegrityError(f"{label} file is unsafe")
        content = path.read_bytes()
    except OSError as error:
        raise IntegrityError(f"{label} cannot be read") from error
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} JSON is invalid") from error
    if not isinstance(decoded, dict) or _canonical_bytes(decoded) != content:
        raise IntegrityError(f"{label} is not canonical JSON")
    return decoded


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _absolute_output_path(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("production output root must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise GenerationError("production output root must be absolute and safe")
    resolved = path.resolve(strict=False)
    probe = resolved.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.is_symlink() or not probe.is_dir():
        raise IntegrityError("production output parent is unsafe")
    return resolved


def _absolute_directory(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise GenerationError(f"{label} must be an absolute safe path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GenerationError(f"{label} is unavailable") from error
    if path.is_symlink() or not resolved.is_dir():
        raise IntegrityError(f"{label} is unsafe")
    return resolved


def _absolute_existing_directory(value: str | Path, label: str) -> Path:
    return _absolute_directory(value, label)


def _absolute_regular_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise GenerationError(f"{label} must be an absolute safe path")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise GenerationError(f"{label} is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not resolved.is_file()
    ):
        raise IntegrityError(f"{label} is unsafe")
    return resolved


def _resolve_project_path(project_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    resolved = candidate.expanduser().resolve(strict=False)
    if project_root != resolved and project_root not in resolved.parents:
        raise GenerationError("configured project path escapes the project root")
    return resolved


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} is not a SHA-256 fingerprint")
    return value


def _run_name(value: object) -> str:
    if not isinstance(value, str) or _RUN_NAME.fullmatch(value) is None:
        raise GenerationError("production run name is invalid")
    return value


def _validated_worker_counts(values: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise GenerationError("benchmark worker counts must be a sequence")
    counts = tuple(values)
    if (
        not counts
        or counts[0] != 1
        or tuple(sorted(set(counts))) != counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in counts
        )
    ):
        raise GenerationError(
            "benchmark worker counts must be sorted unique positives starting at 1"
        )
    return counts


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _swap_used_bytes() -> int | None:
    try:
        return int(psutil.swap_memory().used)
    except (OSError, NotImplementedError, psutil.Error):
        return None
