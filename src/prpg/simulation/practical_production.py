"""Lean multicore delivery runner for the owner-approved practical K=4 model.

This module deliberately reuses the project's tested numerical generator,
frequency aggregation, CSV writer, and work-unit receipts.  Its job is only to
load the compact K=4 artifact in each process, distribute complete CSV shards,
verify the finished receipts in the parent, and publish one concise manifest.

There is no scientific gate graph, intermediate evidence publication, or
crash-recovery protocol here.  A failed run has no completion manifest and is
discarded before retrying at a new output path.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from prpg.config import PRPGConfig, load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.errors import GenerationError, IntegrityError
from prpg.execution import resource_snapshot
from prpg.model.input import load_calibration_input
from prpg.model.practical_k4 import PracticalK4ArtifactStore
from prpg.simulation.production import (
    ProductionPlan,
    ProductionWorkResult,
    ProductionWorkUnit,
    execute_serial_work_unit,
    production_work_units,
)
from prpg.simulation.serial import (
    SERIAL_GENERATOR_VERSION,
    SerialGenerationInput,
    build_practical_k4_serial_generation_input,
)
from prpg.storage.csv_v1 import CSV_SCHEMA_VERSION, CSV_SERIALIZER_VERSION
from prpg.storage.group_commit import (
    LoadedWorkUnitReceipt,
    StorageCommitBinding,
    verify_serial_work_unit,
)

PRACTICAL_PRODUCTION_LOADER_SCHEMA_ID: Final = "prpg-practical-loader-v1"
PRACTICAL_PRODUCTION_MANIFEST_SCHEMA_ID: Final = "prpg-practical-manifest-v1"
PRACTICAL_PRODUCTION_BINDING_SCHEMA_ID: Final = "prpg-practical-binding-v1"
MAX_PRACTICAL_WORKERS: Final = 9
ESTIMATED_CSV_BYTES_PER_ROW: Final = 128
MINIMUM_FREE_DISK_MULTIPLIER: Final = 2

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40,64}\Z")
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_WORKER_CACHE: tuple[str, SerialGenerationInput] | None = None


def _canonical_bytes(value: object) -> bytes:
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


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise GenerationError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


@dataclass(frozen=True, slots=True)
class PracticalProductionLoader:
    """Small pickle-safe specification reloaded once in each worker process."""

    project_root: str
    config_path: str
    artifact_root: str
    artifact_fingerprint: str
    processed_data_fingerprint: str
    config_fingerprint: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        project = Path(self.project_root).expanduser().resolve(strict=True)
        config = Path(self.config_path).expanduser().resolve(strict=True)
        artifact_root = Path(self.artifact_root).expanduser().resolve(strict=True)
        if not project.is_dir() or not artifact_root.is_dir() or not config.is_file():
            raise GenerationError("practical loader paths are invalid")
        if project not in config.parents:
            raise GenerationError("practical configuration must be inside project root")
        _required_fingerprint(self.artifact_fingerprint, "K4 artifact")
        _required_fingerprint(self.processed_data_fingerprint, "processed data")
        _required_fingerprint(self.config_fingerprint, "configuration")
        object.__setattr__(self, "project_root", str(project))
        object.__setattr__(self, "config_path", str(config))
        object.__setattr__(self, "artifact_root", str(artifact_root))
        object.__setattr__(self, "fingerprint", _fingerprint(self.identity()))

    def identity(self) -> dict[str, str]:
        return {
            "schema_id": PRACTICAL_PRODUCTION_LOADER_SCHEMA_ID,
            "project_root": self.project_root,
            "config_path": self.config_path,
            "artifact_root": self.artifact_root,
            "artifact_fingerprint": self.artifact_fingerprint,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PracticalProductionBinding:
    """Only the four identities required by the existing CSV receipt API."""

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
            _required_fingerprint(getattr(self, name), name.replace("_", " "))

    @property
    def storage(self) -> StorageCommitBinding:
        return StorageCommitBinding(
            simulation_fingerprint=self.simulation_fingerprint,
            materialization_fingerprint=self.materialization_fingerprint,
            toolchain_fingerprint=self.toolchain_fingerprint,
            run_fingerprint=self.run_fingerprint,
        )

    def identity(self) -> dict[str, str]:
        return {
            "schema_id": PRACTICAL_PRODUCTION_BINDING_SCHEMA_ID,
            **self.storage.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PracticalProductionResult:
    """Useful final facts from a completed and parent-verified generation run."""

    run_root: Path
    manifest_path: Path
    checksum_path: Path
    manifest_fingerprint: str
    workers: int
    work_units: int
    csv_shards: int
    total_rows: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class _WorkerTask:
    loader: PracticalProductionLoader
    output_root: str
    binding: StorageCommitBinding
    scope: ProductionWorkUnit


def build_practical_production_loader(
    *,
    project_root: str | Path,
    config_path: str | Path,
    artifact_root: str | Path,
    artifact_fingerprint: str,
    processed_data_fingerprint: str,
) -> PracticalProductionLoader:
    """Resolve paths and bind the exact practical artifact, data, and config."""

    try:
        config_file = Path(config_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise GenerationError("practical configuration path is invalid") from error
    config = load_config(config_file)
    return PracticalProductionLoader(
        project_root=str(Path(project_root).expanduser().resolve(strict=True)),
        config_path=str(config_file),
        artifact_root=str(Path(artifact_root).expanduser().resolve(strict=True)),
        artifact_fingerprint=artifact_fingerprint,
        processed_data_fingerprint=processed_data_fingerprint,
        config_fingerprint=config.fingerprint(),
    )


def load_practical_generation_input(
    loader: PracticalProductionLoader,
) -> SerialGenerationInput:
    """Reload durable inputs and mint an independent process-local serial input."""

    if not isinstance(loader, PracticalProductionLoader):
        raise GenerationError("practical generation loader has the wrong type")
    if loader.fingerprint != _fingerprint(loader.identity()):
        raise IntegrityError("practical generation loader fingerprint changed")
    config = load_config(loader.config_path)
    if config.fingerprint() != loader.config_fingerprint:
        raise IntegrityError("practical configuration changed after loader creation")
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(loader.artifact_root),
        loader.processed_data_fingerprint,
        calibration_config_fingerprint=loader.config_fingerprint,
        holdout_months=config.model.design_holdout_months,
    )
    artifact = PracticalK4ArtifactStore(Path(loader.artifact_root)).load(
        loader.artifact_fingerprint
    )
    return build_practical_k4_serial_generation_input(
        authority=artifact,
        config=config,
        calibration_input=calibration_input,
    )


def build_practical_production_binding(
    loader: PracticalProductionLoader,
    *,
    plan: ProductionPlan,
    run_name: str,
) -> PracticalProductionBinding:
    """Derive the small set of identities needed for final CSV receipts."""

    if not isinstance(plan, ProductionPlan):
        raise GenerationError("practical production binding requires a plan")
    if loader.fingerprint != _fingerprint(loader.identity()):
        raise IntegrityError("practical generation loader fingerprint changed")
    if not isinstance(run_name, str) or _RUN_NAME.fullmatch(run_name) is None:
        raise GenerationError("practical run name is invalid")
    config = load_config(loader.config_path)
    if config.fingerprint() != loader.config_fingerprint:
        raise IntegrityError("practical configuration changed after loader creation")
    toolchain = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "serial_generator": SERIAL_GENERATOR_VERSION,
        "csv_schema": CSV_SCHEMA_VERSION,
        "csv_serializer": CSV_SERIALIZER_VERSION,
    }
    toolchain_fingerprint = _fingerprint(toolchain)
    simulation_fingerprint = _fingerprint(
        {
            "artifact_fingerprint": loader.artifact_fingerprint,
            "processed_data_fingerprint": loader.processed_data_fingerprint,
            "config_fingerprint": loader.config_fingerprint,
            "master_seed": config.run.master_seed,
            "scientific_version": config.simulation.scientific_version,
            "serial_generator": SERIAL_GENERATOR_VERSION,
        }
    )
    materialization_fingerprint = _fingerprint(
        {
            "simulation_fingerprint": simulation_fingerprint,
            "plan": plan.identity(),
            "csv_schema": CSV_SCHEMA_VERSION,
            "csv_serializer": CSV_SERIALIZER_VERSION,
        }
    )
    run_fingerprint = _fingerprint(
        {
            "run_name": run_name,
            "simulation_fingerprint": simulation_fingerprint,
            "materialization_fingerprint": materialization_fingerprint,
            "toolchain_fingerprint": toolchain_fingerprint,
        }
    )
    return PracticalProductionBinding(
        simulation_fingerprint=simulation_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
        toolchain_fingerprint=toolchain_fingerprint,
        run_fingerprint=run_fingerprint,
    )


def resolve_practical_workers(
    config: PRPGConfig, *, requested_workers: int | None = None
) -> int:
    """Use available CPU capacity but cap production at the approved nine workers."""

    if not isinstance(config, PRPGConfig):
        raise GenerationError("worker resolution requires a validated config")
    requested: str | int = (
        config.execution.workers if requested_workers is None else requested_workers
    )
    resources = resource_snapshot(
        Path(config.run.output_root),
        requested_workers=requested,
        reserve_cores=config.execution.reserve_physical_cores,
    )
    return min(MAX_PRACTICAL_WORKERS, resources.resolved_workers)


def run_practical_production(
    *,
    loader: PracticalProductionLoader,
    plan: ProductionPlan,
    output_root: str | Path,
    run_name: str,
    workers: int,
) -> PracticalProductionResult:
    """Generate all planned CSVs, verify receipts, then publish final checksums."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise GenerationError("practical worker count must be a positive integer")
    if workers > MAX_PRACTICAL_WORKERS:
        raise GenerationError("practical production supports at most nine workers")
    config = load_config(loader.config_path)
    _validate_plan(plan, config)
    source_provenance = _practical_source_provenance(loader.project_root)
    if _requires_clean_worktree(plan=plan, config=config, run_name=run_name) and cast(
        bool, source_provenance["dirty_worktree"]
    ):
        raise GenerationError(
            "canonical practical production requires a clean committed worktree",
            details={
                "git_commit": source_provenance["git_commit"],
                "run_name": run_name,
            },
        )
    binding = build_practical_production_binding(loader, plan=plan, run_name=run_name)
    root = Path(output_root).expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise GenerationError("practical production output path already exists")
    parent = root.parent
    if not parent.exists() or not parent.is_dir():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise GenerationError(
                "practical output parent cannot be created"
            ) from error
    free_disk = resource_snapshot(
        parent,
        requested_workers=workers,
        reserve_cores=0,
    ).disk_free_bytes
    estimated_output = plan.total_rows * ESTIMATED_CSV_BYTES_PER_ROW
    if free_disk < MINIMUM_FREE_DISK_MULTIPLIER * estimated_output:
        raise GenerationError(
            "insufficient free disk for practical production",
            details={
                "estimated_output_bytes": estimated_output,
                "available_bytes": free_disk,
            },
        )
    try:
        root.mkdir(mode=0o700)
    except OSError as error:
        raise GenerationError("practical output directory cannot be created") from error
    scopes = production_work_units(plan)
    tasks = tuple(
        _WorkerTask(loader, str(root), binding.storage, scope) for scope in scopes
    )
    started = time.monotonic()
    _run_worker_tasks(tasks, workers=workers)

    receipts = tuple(
        verify_serial_work_unit(
            root,
            binding=binding.storage,
            work_unit_id=scope.work_unit_id,
        )
        for scope in scopes
    )
    manifest_identity = _manifest_identity(
        loader=loader,
        plan=plan,
        binding=binding,
        run_name=run_name,
        source_provenance=source_provenance,
        workers=workers,
        scopes=scopes,
        receipts=receipts,
    )
    totals = cast(Mapping[str, object], manifest_identity["totals"])
    observed_rows = cast(int, totals["rows"])
    if observed_rows != plan.total_rows:
        raise IntegrityError(
            "verified CSV row total differs from the production plan",
            details={"expected": plan.total_rows, "observed": observed_rows},
        )
    manifest_fingerprint = _fingerprint(manifest_identity)
    manifest = {
        **manifest_identity,
        "manifest_fingerprint": manifest_fingerprint,
    }
    checksum_lines = _checksum_lines(receipts)
    manifest_path = root / "manifest.json"
    checksum_path = root / "checksums.sha256"
    _write_new_file(checksum_path, checksum_lines.encode("utf-8"))
    _write_new_file(manifest_path, _canonical_bytes(manifest))
    wall_seconds = time.monotonic() - started
    return PracticalProductionResult(
        run_root=root,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        manifest_fingerprint=manifest_fingerprint,
        workers=workers,
        work_units=len(receipts),
        csv_shards=plan.csv_shard_count,
        total_rows=observed_rows,
        wall_seconds=wall_seconds,
    )


def _practical_worker(task: _WorkerTask) -> ProductionWorkResult:
    global _WORKER_CACHE
    if _WORKER_CACHE is None or _WORKER_CACHE[0] != task.loader.fingerprint:
        _WORKER_CACHE = (
            task.loader.fingerprint,
            load_practical_generation_input(task.loader),
        )
    return execute_serial_work_unit(
        _WORKER_CACHE[1],
        output_root=task.output_root,
        binding=task.binding,
        scope=task.scope,
        worker_identity=f"spawn-{os.getpid()}",
    )


def _run_worker_tasks(
    tasks: tuple[_WorkerTask, ...], *, workers: int
) -> tuple[ProductionWorkResult, ...]:
    if workers == 1:
        with threadpool_limits(limits=1):
            return tuple(_practical_worker(task) for task in tasks)
    context = mp.get_context("spawn")
    with (
        _single_thread_environment(),
        ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor,
    ):
        return tuple(executor.map(_practical_worker, tasks, chunksize=1))


def _manifest_identity(
    *,
    loader: PracticalProductionLoader,
    plan: ProductionPlan,
    binding: PracticalProductionBinding,
    run_name: str,
    source_provenance: Mapping[str, object],
    workers: int,
    scopes: tuple[ProductionWorkUnit, ...],
    receipts: tuple[LoadedWorkUnitReceipt, ...],
) -> dict[str, object]:
    if len(scopes) != len(receipts):
        raise IntegrityError("practical production receipt count is incomplete")
    work_units: list[dict[str, object]] = []
    total_rows = 0
    total_bytes = 0
    for scope, receipt in zip(scopes, receipts, strict=True):
        if (
            receipt.work_unit_id != scope.work_unit_id
            or receipt.family != scope.family
            or receipt.first_path_entity != scope.first_path_entity
            or receipt.last_path_entity != scope.last_path_entity
        ):
            raise IntegrityError("practical production receipt scope changed")
        siblings = _manifest_siblings(receipt)
        total_rows += sum(cast(int, item["rows"]) for item in siblings)
        total_bytes += sum(cast(int, item["bytes"]) for item in siblings)
        work_units.append(
            {
                "work_unit_id": receipt.work_unit_id,
                "receipt_fingerprint": receipt.fingerprint,
                "siblings": siblings,
            }
        )
    return {
        "schema_id": PRACTICAL_PRODUCTION_MANIFEST_SCHEMA_ID,
        "status": "complete",
        "run_name": run_name,
        "git_commit": source_provenance["git_commit"],
        "dirty_worktree": source_provenance["dirty_worktree"],
        "requirements_lock_sha256": source_provenance["requirements_lock_sha256"],
        "artifact_fingerprint": loader.artifact_fingerprint,
        "processed_data_fingerprint": loader.processed_data_fingerprint,
        "config_fingerprint": loader.config_fingerprint,
        "plan": plan.identity(),
        "binding": binding.identity(),
        "workers": workers,
        "totals": {
            "paths": plan.lf_paths + plan.hf_paths,
            "work_units": len(receipts),
            "csv_shards": plan.csv_shard_count,
            "rows": total_rows,
            "bytes": total_bytes,
        },
        "work_units": work_units,
    }


def _manifest_siblings(receipt: LoadedWorkUnitReceipt) -> list[dict[str, object]]:
    raw = receipt.identity.get("siblings")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise IntegrityError("practical production sibling receipt is invalid")
    result: list[dict[str, object]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise IntegrityError("practical production sibling receipt is invalid")
        frequency = value.get("frequency")
        relative_path = value.get("relative_path")
        rows = value.get("rows")
        bytes_count = value.get("bytes")
        sha256 = value.get("sha256")
        if (
            not isinstance(frequency, str)
            or not isinstance(relative_path, str)
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(bytes_count, bool)
            or not isinstance(bytes_count, int)
            or bytes_count <= 0
        ):
            raise IntegrityError("practical production sibling metadata is invalid")
        _required_fingerprint(sha256, "CSV")
        result.append(
            {
                "frequency": frequency,
                "relative_path": relative_path,
                "rows": rows,
                "bytes": bytes_count,
                "sha256": sha256,
            }
        )
    return result


def _checksum_lines(receipts: tuple[LoadedWorkUnitReceipt, ...]) -> str:
    entries: list[tuple[str, str]] = []
    for receipt in receipts:
        for sibling in _manifest_siblings(receipt):
            entries.append(
                (
                    cast(str, sibling["relative_path"]),
                    cast(str, sibling["sha256"]),
                )
            )
    return "".join(f"{sha256}  {path}\n" for path, sha256 in sorted(entries))


def _validate_plan(plan: ProductionPlan, config: PRPGConfig) -> None:
    if not isinstance(plan, ProductionPlan):
        raise GenerationError("practical production requires a ProductionPlan")
    if (
        plan.years != config.simulation.low_frequency.years
        or plan.lf_paths > config.simulation.low_frequency.paths
        or plan.hf_paths > config.simulation.high_frequency.paths
    ):
        raise GenerationError("practical production plan exceeds its configuration")


def _practical_source_provenance(project_root: str | Path) -> dict[str, object]:
    """Record the committed revision, tree state, and dependency-lock bytes."""

    try:
        root = Path(project_root).expanduser().resolve(strict=True)
        lock_content = (root / "requirements.lock").read_bytes()
    except OSError as error:
        raise GenerationError("practical source provenance is unavailable") from error
    commit = _git_output(root, "rev-parse", "--verify", "HEAD")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise GenerationError("practical Git commit identity is invalid")
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        allow_empty=True,
    )
    return {
        "git_commit": commit,
        "dirty_worktree": bool(status),
        "requirements_lock_sha256": hashlib.sha256(lock_content).hexdigest(),
    }


def _git_output(project_root: Path, *arguments: str, allow_empty: bool = False) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GenerationError("practical Git provenance is unavailable") from error
    output = completed.stdout.strip()
    if not output and not allow_empty:
        raise GenerationError("practical Git provenance is unavailable")
    return output


def _requires_clean_worktree(
    *, plan: ProductionPlan, config: PRPGConfig, run_name: str
) -> bool:
    """Treat a full-size run as canonical unless explicitly named pilot/smoke."""

    noncanonical_tokens = {"pilot", "smoke"}
    explicitly_noncanonical = bool(
        noncanonical_tokens.intersection(run_name.split("-"))
    )
    full_size = (
        plan.lf_paths == config.simulation.low_frequency.paths
        and plan.hf_paths == config.simulation.high_frequency.paths
    )
    return full_size and not explicitly_noncanonical


def _write_new_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise IntegrityError(f"cannot publish {path.name}") from error


@contextmanager
def _single_thread_environment() -> Iterator[None]:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
