"""Explicitly noncanonical reduced rehearsal for the closed G3 task graph.

The canonical G3 composition root deliberately accepts only exact scientific
geometry and writes task, model, pair, and evidence stores.  A prelaunch
rehearsal must do neither.  This module therefore owns a separate, one-way
boundary: a code-owned reduced workload is walked through the same closed
26-task registry, its resource measurements are passed to the pure projection
module, and the only return value is an unreleasable report.

No canonical store or authority type is imported here.  In particular, a
rehearsal cannot initialize a launch, publish a scientific task result, mint a
model/artifact pair, publish G3 evidence, or authorize generation.  Task
observations say ``graph_traversed`` rather than ``exercised`` or ``passed``:
real numerical-family coverage is carried separately by the measurement
bundle, while reduced geometry is never a scientific G3 decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, cast

import numpy as np
import psutil
from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

from prpg.config import PRPGConfig
from prpg.errors import IntegrityError, ModelError
from prpg.execution import deterministic_process_map
from prpg.model.adequacy import (
    LATENT_EXTENSION_MONTHS,
    LATENT_MEASURED_MONTHS,
    LATENT_SURVIVAL_MONTHS,
    generate_latent_chains,
    latent_cluster_bootstrap,
)
from prpg.model.block_execution import Artifact
from prpg.model.block_execution_registered import (
    RegisteredBlockUniformMaterialization,
    iter_materialized_block_trace_chunks,
    materialize_registered_block_uniforms,
    run_materialized_block_calibration,
)
from prpg.model.block_length import (
    BLOCK_POLICY_VERSION,
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.bridge import (
    bridge_resource_preflight,
)
from prpg.model.bridge_driver import (
    BRIDGE_DGPS,
    BridgeBenchmarkPairEvidence,
    _BenchmarkJob,
    _run_benchmark_job,
)
from prpg.model.calibration import (
    DecodedReturnPools,
    decoded_return_pools,
    preliminary_block_calibrations,
    prepare_hmm_feature_matrix,
)
from prpg.model.dependence import fit_dependence_reference
from prpg.model.dependence_execution import DependenceTraceReducer
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_preflight import (
    CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
    CANONICAL_CONFIG_FINGERPRINT,
    CANONICAL_PROCESSED_DATA_FINGERPRINT,
    CANONICAL_RAW_SNAPSHOT_FINGERPRINT,
    _verify_config,
    _verify_environment,
    _verify_fingerprints,
    _verify_geometry_and_seal,
    _verify_module_constants,
    _verify_rng_registry,
    inspect_runtime_environment,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_TASK_REGISTRY,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.g3_resource_projection import (
    G3ResourceProjection,
    G3ResourceProjectionMeasurements,
    project_canonical_g3_resources,
)
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMCandidateInadmissibility,
    HMMFeatureMatrix,
    NoNumericallyValidHMMRestarts,
    StatePathDiagnostic,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_reduced_diagnostic_restarts,
    fit_gaussian_hmm_restarts,
    sequence_lengths_from_continuation,
)
from prpg.model.input import (
    CalibrationInput,
    CalibrationSlice,
    forward_links_from_row_start_mask,
)
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    run_registered_kernel_calibration_cell,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.regime_policy import (
    OWNER_FIXED_K4_PROSPECTIVE_POLICY,
    OwnerFixedK4RecurringHistorySupport,
    owner_fixed_k4_recurring_history_support,
)
from prpg.model.selection import (
    OWNER_FIXED_K4_SELECTION_POLICY,
    OWNER_FIXED_N_STATES,
)
from prpg.model.stability import map_monthly_states_to_daily
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
    inspect_g3_execution_closure,
    verify_g3_execution_closure_identity,
)
from prpg.simulation.rng import ModelFitScope

G3_REHEARSAL_CONTRACT_VERSION: Final = "g3-reduced-rehearsal-v4"
G3_REHEARSAL_REPORT_SCHEMA: Final = "prpg-g3-noncanonical-rehearsal-report-v4"
_CONCRETE_WORKLOAD_CAPABILITY: Final = object()
_MINTED_WORKLOADS: dict[int, _ConcreteReducedG3RehearsalWorkload] = {}
_BOUND_TELEMETRY: dict[
    int, tuple[_ConcreteReducedG3RehearsalWorkload, G3RehearsalTelemetry]
] = {}
_MINTED_REPORTS: dict[int, NoncanonicalG3RehearsalReport] = {}

RehearsalFamily: TypeAlias = Literal[
    "hmm",
    "block_materialization",
    "block_call",
    "kernel",
    "dependence",
    "latent_generation",
    "latent_bootstrap",
    "bridge_benchmark",
    "lightweight",
]


@dataclass(frozen=True, slots=True)
class ReducedG3RehearsalGeometry:
    """The sole reduced geometry accepted by the projection contract."""

    workers: Literal[9] = 9
    hmm_restarts_per_k: Literal[50] = 50
    block_histories: Literal[128] = 128
    kernel_windows_per_cell: Literal[1152] = 1_152
    latent_chains_per_role: Literal[512] = 512
    latent_bootstrap_replicates_per_role: Literal[512] = 512
    latent_bootstrap_resample_size: Literal[512] = 512
    bridge_coordinates: Literal[100] = 100

    def __post_init__(self) -> None:
        expected = (9, 50, 128, 1_152, 512, 512, 512, 100)
        if tuple(asdict(self).values()) != expected:
            raise ModelError("the public rehearsal geometry is code-owned and fixed")


REDUCED_G3_REHEARSAL_GEOMETRY: Final = ReducedG3RehearsalGeometry()


@dataclass(frozen=True, slots=True)
class _ConcreteRehearsalGeometry:
    """Internal execution geometry; only the fixed instance may be projected."""

    workers: int
    hmm_state_counts: tuple[int, ...]
    hmm_restarts_per_k: int
    block_histories: int
    kernel_windows_per_cell: int
    latent_chains_per_role: int
    latent_bootstrap_replicates_per_role: int
    latent_bootstrap_resample_size: int
    bridge_coordinates: int
    latent_measured_months: int
    latent_extension_months: int
    projection_eligible: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.workers, "rehearsal workers"),
            (self.hmm_restarts_per_k, "rehearsal HMM restarts"),
            (self.block_histories, "rehearsal block histories"),
            (self.kernel_windows_per_cell, "rehearsal kernel windows"),
            (self.latent_chains_per_role, "rehearsal latent chains"),
            (
                self.latent_bootstrap_replicates_per_role,
                "rehearsal latent bootstrap replicates",
            ),
            (
                self.latent_bootstrap_resample_size,
                "rehearsal latent resample size",
            ),
            (self.bridge_coordinates, "rehearsal bridge coordinates"),
            (self.latent_measured_months, "rehearsal latent measured months"),
            (self.latent_extension_months, "rehearsal latent extension months"),
        ):
            if type(value) is not int or value <= 0:
                raise ModelError(f"{label} must be a positive integer")
        if (
            not self.hmm_state_counts
            or any(
                type(value) is not int or value < 2 for value in self.hmm_state_counts
            )
            or len(set(self.hmm_state_counts)) != len(self.hmm_state_counts)
        ):
            raise ModelError("rehearsal HMM state counts must be unique integers >= 2")
        if OWNER_FIXED_N_STATES not in self.hmm_state_counts:
            raise ModelError("rehearsal HMM family must include owner-fixed K=4")
        if self.hmm_restarts_per_k > HMM_RESTARTS:
            raise ModelError("rehearsal HMM restarts exceed the registered maximum")
        if self.latent_bootstrap_resample_size != self.latent_chains_per_role:
            raise ModelError("rehearsal latent resample size must equal chain count")
        if type(self.projection_eligible) is not bool:
            raise ModelError("rehearsal projection eligibility must be boolean")
        if self.projection_eligible and (
            self.workers,
            self.hmm_state_counts,
            self.hmm_restarts_per_k,
            self.block_histories,
            self.kernel_windows_per_cell,
            self.latent_chains_per_role,
            self.latent_bootstrap_replicates_per_role,
            self.latent_bootstrap_resample_size,
            self.bridge_coordinates,
            self.latent_measured_months,
            self.latent_extension_months,
        ) != (
            9,
            (2, 3, 4, 5),
            50,
            128,
            1_152,
            512,
            512,
            512,
            100,
            LATENT_MEASURED_MONTHS,
            LATENT_EXTENSION_MONTHS,
        ):
            raise ModelError(
                "projection-eligible rehearsal geometry must equal the fixed "
                "production geometry"
            )


_PRODUCTION_EXECUTION_GEOMETRY: Final = _ConcreteRehearsalGeometry(
    workers=REDUCED_G3_REHEARSAL_GEOMETRY.workers,
    hmm_state_counts=(2, 3, 4, 5),
    hmm_restarts_per_k=REDUCED_G3_REHEARSAL_GEOMETRY.hmm_restarts_per_k,
    block_histories=REDUCED_G3_REHEARSAL_GEOMETRY.block_histories,
    kernel_windows_per_cell=(REDUCED_G3_REHEARSAL_GEOMETRY.kernel_windows_per_cell),
    latent_chains_per_role=(REDUCED_G3_REHEARSAL_GEOMETRY.latent_chains_per_role),
    latent_bootstrap_replicates_per_role=(
        REDUCED_G3_REHEARSAL_GEOMETRY.latent_bootstrap_replicates_per_role
    ),
    latent_bootstrap_resample_size=(
        REDUCED_G3_REHEARSAL_GEOMETRY.latent_bootstrap_resample_size
    ),
    bridge_coordinates=REDUCED_G3_REHEARSAL_GEOMETRY.bridge_coordinates,
    latent_measured_months=LATENT_MEASURED_MONTHS,
    latent_extension_months=LATENT_EXTENSION_MONTHS,
    projection_eligible=True,
)


@dataclass(frozen=True, slots=True)
class RehearsalFamilyTiming:
    """One non-overlapping resource-measurement contribution."""

    family: RehearsalFamily
    cell: str
    wall_seconds: float

    def __post_init__(self) -> None:
        if not self.cell:
            raise ModelError("rehearsal family timing requires a cell label")
        _positive_float(self.wall_seconds, "rehearsal family wall seconds")


@dataclass(frozen=True, slots=True)
class ReducedG3TaskProbe:
    """A non-authoritative observation returned by one reduced task probe."""

    task_id: str
    result_fingerprint: str
    timings: tuple[RehearsalFamilyTiming, ...] = ()
    serialized_bytes: int = 0

    def __post_init__(self) -> None:
        if self.task_id not in G3_GATE_ORDER:
            raise ModelError("rehearsal probe task is outside the closed G3 registry")
        _require_sha256(self.result_fingerprint, "rehearsal probe fingerprint")
        if type(self.timings) is not tuple:
            raise ModelError("rehearsal probe timings must be a tuple")
        if any(type(item) is not RehearsalFamilyTiming for item in self.timings):
            raise ModelError("rehearsal probe timings contain a substituted type")
        if type(self.serialized_bytes) is not int or self.serialized_bytes < 0:
            raise ModelError("rehearsal probe serialized bytes must be non-negative")


class ReducedG3RehearsalWorkload(Protocol):
    """Internal numerical workload used by the closed rehearsal walker.

    This protocol is intentionally private to orchestration.  Public callers
    receive the concrete code-owned workload through
    :func:`run_reduced_g3_rehearsal`; accepting arbitrary probes there would
    allow fabricated resource evidence even though it could not mint G3
    authority.
    """

    calibration_input_fingerprint: str
    config_fingerprint: str
    processed_data_fingerprint: str
    raw_snapshot_fingerprint: str
    source_code_fingerprint: str
    dependency_lock_fingerprint: str
    scientific_version: int
    native_thread_cap_verified: bool
    worker_parity_verified: bool
    worker_parity_fingerprint: str
    worker_parity_families: tuple[str, ...]
    selection_policy: Literal["owner_fixed_k4_v3"]
    selected_n_states: Literal[4]
    bridge_benchmark_n_states: int
    rehearsal_geometry: ReducedG3RehearsalGeometry | None

    def execute_task(
        self,
        task_id: str,
        dependency_fingerprints: tuple[tuple[str, str], ...],
    ) -> ReducedG3TaskProbe: ...


class _BridgeElapsedRecord(Protocol):
    """Minimal measured bridge record consumed by resource reduction."""

    @property
    def dgp(self) -> str: ...

    @property
    def pair(self) -> int: ...

    @property
    def elapsed_seconds(self) -> float: ...


@dataclass(frozen=True, slots=True)
class G3RehearsalTelemetry:
    """Whole-rehearsal host measurements, collected without writing a store."""

    peak_process_tree_rss_bytes: int
    process_tree_cpu_seconds: float
    swap_used_before_bytes: int
    maximum_swap_used_bytes: int
    swap_used_after_bytes: int
    disk_free_bytes: int
    rss_telemetry_complete: bool
    swap_telemetry_complete: bool
    cpu_telemetry_complete: bool
    disk_telemetry_complete: bool

    def __post_init__(self) -> None:
        _positive_integer(
            self.peak_process_tree_rss_bytes,
            "rehearsal peak process-tree RSS",
        )
        _positive_float(self.process_tree_cpu_seconds, "rehearsal CPU seconds")
        for value, label in (
            (self.swap_used_before_bytes, "rehearsal swap before"),
            (self.maximum_swap_used_bytes, "rehearsal maximum swap"),
            (self.swap_used_after_bytes, "rehearsal swap after"),
        ):
            _nonnegative_integer(value, label)
        if self.maximum_swap_used_bytes < max(
            self.swap_used_before_bytes,
            self.swap_used_after_bytes,
        ):
            raise ModelError("rehearsal maximum swap does not cover endpoint readings")
        _positive_integer(self.disk_free_bytes, "rehearsal free disk")
        for name in (
            "rss_telemetry_complete",
            "swap_telemetry_complete",
            "cpu_telemetry_complete",
            "disk_telemetry_complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise ModelError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class _RehearsalCell:
    artifact: Artifact
    frequency: Literal["monthly", "daily"]
    source_slice: CalibrationSlice
    model: GaussianHMMFit
    historical_returns: np.ndarray
    historical_states: np.ndarray
    continuation_links: np.ndarray
    intended_length: int
    monthly_segment_lengths: tuple[int, ...]
    daily_segment_lengths: tuple[int, ...] | None
    materialization: RegisteredBlockUniformMaterialization

    @property
    def cell_id(self) -> str:
        return f"{self.artifact}_{self.frequency}"


class _ConcreteReducedG3RehearsalWorkload:
    """Completed code-owned numerical workload served to the DAG assembler."""

    def __init__(
        self,
        *,
        config_fingerprint: str,
        calibration_input_fingerprint: str,
        processed_data_fingerprint: str,
        raw_snapshot_fingerprint: str,
        source_code_fingerprint: str,
        dependency_lock_fingerprint: str,
        scientific_version: int,
        worker_parity_fingerprint: str,
        worker_parity_families: tuple[str, ...],
        selection_policy: Literal["owner_fixed_k4_v3"],
        selected_n_states: Literal[4],
        bridge_benchmark_n_states: int,
        rehearsal_geometry: ReducedG3RehearsalGeometry | None,
        probes: tuple[ReducedG3TaskProbe, ...],
        expected_dependencies: tuple[tuple[tuple[str, str], ...], ...],
        _seal: object,
    ) -> None:
        if _seal is not _CONCRETE_WORKLOAD_CAPABILITY:
            raise TypeError(
                "concrete G3 rehearsal workloads are minted only by the numerical "
                "executor"
            )
        self.config_fingerprint = config_fingerprint
        self.calibration_input_fingerprint = calibration_input_fingerprint
        self.processed_data_fingerprint = processed_data_fingerprint
        self.raw_snapshot_fingerprint = raw_snapshot_fingerprint
        self.source_code_fingerprint = source_code_fingerprint
        self.dependency_lock_fingerprint = dependency_lock_fingerprint
        self.scientific_version = scientific_version
        self.native_thread_cap_verified = True
        self.worker_parity_verified = True
        self.worker_parity_fingerprint = worker_parity_fingerprint
        self.worker_parity_families = worker_parity_families
        self.selection_policy = selection_policy
        self.selected_n_states = selected_n_states
        self.bridge_benchmark_n_states = bridge_benchmark_n_states
        self.rehearsal_geometry = rehearsal_geometry
        self._probes = probes
        self._expected_dependencies = expected_dependencies
        self._next = 0
        _MINTED_WORKLOADS[id(self)] = self

    def execute_task(
        self,
        task_id: str,
        dependency_fingerprints: tuple[tuple[str, str], ...],
    ) -> ReducedG3TaskProbe:
        if self._next >= len(self._probes):
            raise IntegrityError("rehearsal workload was consumed more than once")
        expected_task = G3_GATE_ORDER[self._next]
        expected_dependencies = self._expected_dependencies[self._next]
        if task_id != expected_task or dependency_fingerprints != expected_dependencies:
            raise IntegrityError("rehearsal workload was consumed out of DAG order")
        result = self._probes[self._next]
        self._next += 1
        return result


class _RehearsalHostSampler:
    """Measure the complete parent/child process tree without writing evidence."""

    def __init__(self, disk_path: Path, *, interval_seconds: float = 0.02) -> None:
        self._disk_path = _existing_ancestor(disk_path)
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="prpg-g3-rehearsal-telemetry",
            daemon=True,
        )
        self._root = psutil.Process(os.getpid())
        self._rss_complete = True
        self._swap_complete = True
        self._disk_complete = True
        self._peak_rss = 0
        self._swap_before = _swap_used_bytes()
        self._swap_after = self._swap_before
        self._maximum_swap = self._swap_before
        self._minimum_disk_free = _disk_free_bytes(self._disk_path)
        self._self_cpu_before = _process_cpu_seconds(self._root)
        self._children_cpu_before = _completed_children_cpu_seconds()
        self._started = False

    def start(self) -> None:
        if self._started:
            raise IntegrityError("rehearsal telemetry sampler started twice")
        self._started = True
        self._sample()
        self._thread.start()

    def stop(self) -> G3RehearsalTelemetry:
        if not self._started:
            raise IntegrityError("rehearsal telemetry sampler was not started")
        self._stop.set()
        self._thread.join()
        self._sample()
        swap_after = _swap_used_bytes()
        if swap_after is None:
            self._swap_complete = False
            swap_after = 0
        self._swap_after = swap_after
        self._maximum_swap = max(self._maximum_swap or 0, swap_after)
        self_cpu_after = _process_cpu_seconds(self._root)
        children_cpu_after = _completed_children_cpu_seconds()
        cpu_seconds = (
            self_cpu_after
            - self._self_cpu_before
            + children_cpu_after
            - self._children_cpu_before
        )
        if not math.isfinite(cpu_seconds) or cpu_seconds <= 0.0:
            raise ModelError("rehearsal process-tree CPU measurement is unavailable")
        if self._minimum_disk_free is None or self._minimum_disk_free <= 0:
            self._disk_complete = False
            disk_free = 1
        else:
            disk_free = self._minimum_disk_free
        before = self._swap_before
        if before is None:
            self._swap_complete = False
            before = 0
        return G3RehearsalTelemetry(
            peak_process_tree_rss_bytes=max(1, self._peak_rss),
            process_tree_cpu_seconds=float(cpu_seconds),
            swap_used_before_bytes=before,
            maximum_swap_used_bytes=max(self._maximum_swap or 0, before, swap_after),
            swap_used_after_bytes=swap_after,
            disk_free_bytes=disk_free,
            rss_telemetry_complete=self._rss_complete,
            swap_telemetry_complete=self._swap_complete,
            cpu_telemetry_complete=True,
            disk_telemetry_complete=self._disk_complete,
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        try:
            processes = (self._root, *self._root.children(recursive=True))
        except (OSError, psutil.Error):
            processes = (self._root,)
            self._rss_complete = False
        rss = 0
        for process in processes:
            try:
                rss += int(process.memory_info().rss)
            except psutil.NoSuchProcess:
                # A short-lived worker may exit after the process tree is
                # enumerated but before its RSS is read.  At read time that
                # process is no longer resident, so it contributes zero; this
                # normal lifecycle race does not make host telemetry
                # unavailable.  Access and operating-system errors still fail
                # the completeness gate below.
                continue
            except (OSError, psutil.AccessDenied):
                self._rss_complete = False
        self._peak_rss = max(self._peak_rss, rss)
        swap = _swap_used_bytes()
        if swap is None:
            self._swap_complete = False
        else:
            self._maximum_swap = max(self._maximum_swap or 0, swap)
        free = _disk_free_bytes(self._disk_path)
        if free is None:
            self._disk_complete = False
        else:
            self._minimum_disk_free = (
                free
                if self._minimum_disk_free is None
                else min(self._minimum_disk_free, free)
            )


@dataclass(frozen=True, slots=True)
class G3RehearsalTaskObservation:
    """One exact-registry task reached under noncanonical reduced geometry."""

    ordinal: int
    task_id: str
    dependency_fingerprints: tuple[tuple[str, str], ...]
    result_fingerprint: str
    outcome: Literal["graph_traversed_noncanonical"] = "graph_traversed_noncanonical"

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 1 <= self.ordinal <= len(G3_GATE_ORDER):
            raise ModelError("rehearsal task ordinal is outside the registry")
        if self.task_id != G3_GATE_ORDER[self.ordinal - 1]:
            raise ModelError("rehearsal task observation is out of registry order")
        _require_sha256(self.result_fingerprint, "rehearsal task fingerprint")
        if type(self.dependency_fingerprints) is not tuple:
            raise ModelError("rehearsal dependencies must be a tuple")
        for task_id, fingerprint in self.dependency_fingerprints:
            if task_id not in G3_GATE_ORDER:
                raise ModelError("rehearsal dependency is outside the registry")
            _require_sha256(fingerprint, "rehearsal dependency fingerprint")


@dataclass(frozen=True, slots=True)
class NoncanonicalG3RehearsalReport:
    """Unreleasable rehearsal and resource-projection report.

    The negative authority fields are literal constants and are reverified by
    :func:`verify_noncanonical_g3_rehearsal_report`.  There is deliberately no
    ``passed`` property: only ``resource_projection.passed`` has gate meaning,
    and even that decision cannot stand in for scientific G3 success.
    """

    schema_id: Literal["prpg-g3-noncanonical-rehearsal-report-v4"] = (
        G3_REHEARSAL_REPORT_SCHEMA
    )
    contract_version: Literal["g3-reduced-rehearsal-v4"] = G3_REHEARSAL_CONTRACT_VERSION
    execution_mode: Literal["rehearsal"] = "rehearsal"
    g3_contract_version: str = G3_CONTRACT_VERSION
    block_policy_version: str = BLOCK_POLICY_VERSION
    config_fingerprint: str = ""
    calibration_input_fingerprint: str = ""
    processed_data_fingerprint: str = ""
    raw_snapshot_fingerprint: str = ""
    source_code_fingerprint: str = ""
    dependency_lock_fingerprint: str = ""
    worker_parity_fingerprint: str = ""
    worker_parity_families: tuple[str, ...] = ()
    selection_policy: Literal["owner_fixed_k4_v3"] = OWNER_FIXED_K4_SELECTION_POLICY
    selected_n_states: Literal[4] = OWNER_FIXED_N_STATES
    scientific_version: int = 0
    task_registry_fingerprint: str = G3_TASK_REGISTRY_FINGERPRINT
    geometry: ReducedG3RehearsalGeometry = REDUCED_G3_REHEARSAL_GEOMETRY
    tasks: tuple[G3RehearsalTaskObservation, ...] = ()
    measurements: G3ResourceProjectionMeasurements | None = None
    resource_projection: G3ResourceProjection | None = None
    report_fingerprint: str = ""
    g3_passed: Literal[False] = False
    canonical_authority: Literal[False] = False
    generation_authorized: Literal[False] = False
    releasable: Literal[False] = False

    def to_json_bytes(self) -> bytes:
        """Return deterministic report bytes without writing any filesystem path."""

        verify_noncanonical_g3_rehearsal_report(self)
        return _canonical_bytes(_report_identity(self, include_fingerprint=True))


def run_reduced_g3_rehearsal(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> NoncanonicalG3RehearsalReport:
    """Run the sole measured, noncanonical target-Mac rehearsal.

    Callers supply only the already resolved configuration and immutable G2
    calibration input.  Geometry, workers, numerical families, telemetry, and
    the projection are code owned.  This path imports no run/artifact/evidence
    store and cannot mint canonical launch or generation authority.
    """

    _verify_rehearsal_inputs(config, calibration_input)
    source = inspect_g3_execution_closure()
    sampler = _RehearsalHostSampler(Path(config.run.output_root))
    sampler.start()
    try:
        workload = _execute_concrete_rehearsal_workload(
            config,
            calibration_input,
            geometry=_PRODUCTION_EXECUTION_GEOMETRY,
            source_code_fingerprint=source.source_code_fingerprint,
            dependency_lock_fingerprint=source.dependency_lock_fingerprint,
        )
    finally:
        telemetry = sampler.stop()
    verify_g3_execution_closure_identity(source)
    _bind_rehearsal_telemetry(workload, telemetry)
    return assemble_noncanonical_g3_rehearsal_report(workload, telemetry)


def _run_test_only_rehearsal_workload(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    geometry: _ConcreteRehearsalGeometry,
) -> tuple[_ConcreteReducedG3RehearsalWorkload, G3RehearsalTelemetry]:
    """Private integration-test seam that can never feed the projection."""

    if geometry.projection_eligible:
        raise ModelError("test-only rehearsal geometry cannot be projection eligible")
    source = inspect_g3_execution_closure()
    sampler = _RehearsalHostSampler(Path(config.run.output_root))
    sampler.start()
    try:
        workload = _execute_concrete_rehearsal_workload(
            config,
            calibration_input,
            geometry=geometry,
            source_code_fingerprint=source.source_code_fingerprint,
            dependency_lock_fingerprint=source.dependency_lock_fingerprint,
        )
    finally:
        telemetry = sampler.stop()
    verify_g3_execution_closure_identity(source)
    if workload.rehearsal_geometry is not None:
        raise AssertionError("test-only workload acquired projection geometry")
    _bind_rehearsal_telemetry(workload, telemetry)
    return workload, telemetry


def _bind_rehearsal_telemetry(
    workload: _ConcreteReducedG3RehearsalWorkload,
    telemetry: G3RehearsalTelemetry,
) -> None:
    if (
        type(workload) is not _ConcreteReducedG3RehearsalWorkload
        or _MINTED_WORKLOADS.get(id(workload)) is not workload
        or type(telemetry) is not G3RehearsalTelemetry
    ):
        raise IntegrityError("rehearsal telemetry requires a minted concrete workload")
    if id(workload) in _BOUND_TELEMETRY:
        raise IntegrityError("rehearsal telemetry was bound more than once")
    _BOUND_TELEMETRY[id(workload)] = (workload, telemetry)


def _execute_concrete_rehearsal_workload(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    *,
    geometry: _ConcreteRehearsalGeometry,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
) -> _ConcreteReducedG3RehearsalWorkload:
    if geometry.projection_eligible and geometry is not _PRODUCTION_EXECUTION_GEOMETRY:
        raise IntegrityError(
            "only the module-owned production rehearsal geometry may be projected"
        )
    started = time.perf_counter()
    timings_by_task: dict[str, list[RehearsalFamilyTiming]] = {
        task_id: [] for task_id in G3_GATE_ORDER
    }
    results_by_task: dict[str, list[str]] = {task_id: [] for task_id in G3_GATE_ORDER}
    parity_results: list[tuple[str, str]] = []

    thread_probes = deterministic_process_map(
        _native_thread_probe,
        tuple(range(geometry.workers)),
        workers=geometry.workers,
    )
    if not thread_probes or not all(thread_probes):
        raise ModelError("rehearsal worker native-thread cap was not observed")
    results_by_task["input_integrity"].append(
        _sha256(_canonical_bytes({"native_thread_probes": list(thread_probes)}))
    )

    design_features = prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="design",
        design_rows=len(calibration_input.design.macro_features),
    )
    fits: dict[int, GaussianHMMFit] = {}
    for n_states in geometry.hmm_state_counts:
        seeds = deterministic_hmm_restart_seeds(
            master_seed=config.run.master_seed,
            scientific_version=config.simulation.scientific_version,
            scope=ModelFitScope.DESIGN_MAIN,
            replicate=0,
            n_states=n_states,
        )[: geometry.hmm_restarts_per_k]
        parallel_started = time.perf_counter()
        parallel_fit, parallel_fp = _fit_rehearsal_hmm_outcome(
            design_features,
            n_states,
            seeds,
            workers=geometry.workers,
        )
        parallel_seconds = _elapsed(parallel_started)
        serial_fit, serial_fp = _fit_rehearsal_hmm_outcome(
            design_features,
            n_states,
            seeds,
            workers=1,
        )
        if parallel_fp != serial_fp or (parallel_fit is None) != (serial_fit is None):
            raise IntegrityError(
                f"rehearsal HMM K={n_states} differs for one versus workers"
            )
        parity_results.append((f"hmm_k{n_states}", parallel_fp))
        results_by_task["design_candidate_viability"].append(parallel_fp)
        timings_by_task["design_candidate_viability"].append(
            RehearsalFamilyTiming("hmm", f"k{n_states}", parallel_seconds)
        )
        if parallel_fit is not None:
            fits[n_states] = parallel_fit

    selected_fit = _select_owner_fixed_k4_rehearsal_fit(fits)
    design_recurring_support = _require_rehearsal_recurring_history_support(
        decoded_return_pools(selected_fit, calibration_input.design),
        role="design",
    )
    results_by_task["design_candidate_viability"].append(
        _scientific_fingerprint(design_recurring_support)
    )
    selected_fp = gaussian_hmm_fit_fingerprint(selected_fit)
    results_by_task["design_predictive_selection"].append(
        _sha256(
            _canonical_bytes(
                {
                    "selection_policy": OWNER_FIXED_K4_SELECTION_POLICY,
                    "selected_n_states": OWNER_FIXED_N_STATES,
                    "selected_fit_fingerprint": selected_fp,
                }
            )
        )
    )
    results_by_task["sealed_holdout_veto"].append(
        _scientific_fingerprint(selected_fit.identification)
    )
    results_by_task["design_macro_stability"].append(
        _scientific_fingerprint(selected_fit.state_path_diagnostics)
    )

    production_features = prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="production",
    )
    production_seeds = deterministic_hmm_restart_seeds(
        master_seed=config.run.master_seed,
        scientific_version=config.simulation.scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=selected_fit.n_states,
    )[: geometry.hmm_restarts_per_k]
    production_fit, production_fp = _fit_rehearsal_hmm_outcome(
        production_features,
        selected_fit.n_states,
        production_seeds,
        workers=geometry.workers,
    )
    serial_production_fit, serial_production_fp = _fit_rehearsal_hmm_outcome(
        production_features,
        selected_fit.n_states,
        production_seeds,
        workers=1,
    )
    if production_fp != serial_production_fp or (production_fit is None) != (
        serial_production_fit is None
    ):
        raise IntegrityError("rehearsal production HMM differs for one versus workers")
    parity_results.append((f"hmm_production_k{selected_fit.n_states}", production_fp))
    if production_fit is None:
        raise ModelError(
            "rehearsal owner-fixed K4 production HMM is inadmissible",
            details={
                "n_states": selected_fit.n_states,
                "rejection_fingerprint": production_fp,
            },
        )
    production_recurring_support = _require_rehearsal_recurring_history_support(
        decoded_return_pools(production_fit, calibration_input.full),
        role="production",
    )
    results_by_task["production_fit_same_k"].append(production_fp)
    results_by_task["production_fit_same_k"].append(
        _scientific_fingerprint(production_recurring_support)
    )
    results_by_task["production_macro_stability"].append(
        _scientific_fingerprint(production_fit.state_path_diagnostics)
    )

    # Only the design selector is scientific input to G3 at this point.  The
    # production resource cells use the approved absolute maxima below so the
    # rehearsal does not inspect a production PPW outcome before design freeze.
    design_preliminary = preliminary_block_calibrations(
        calibration_input.design,
        role="design",
    )
    design_monthly_start = design_preliminary.monthly.combined.starting_length
    design_daily_start = design_preliminary.daily.combined.starting_length
    design_macro_start = design_preliminary.macro.combined.starting_length
    design_starts, production_starts = _rehearsal_block_start_geometry(
        design_monthly_start,
        design_daily_start,
    )
    results_by_task["input_integrity"].append(
        _scientific_fingerprint(design_preliminary)
    )

    materialization_started = time.perf_counter()
    cells = _rehearsal_cells(
        config,
        calibration_input,
        design_fit=selected_fit,
        production_fit=production_fit,
        geometry=geometry,
        design_starts=design_starts,
        production_starts=production_starts,
    )
    materialization_seconds = _elapsed(materialization_started)
    timings_by_task["design_block_calibration"].append(
        RehearsalFamilyTiming(
            "block_materialization", "all_four", materialization_seconds
        )
    )

    for cell in cells:
        block_started = time.perf_counter()
        block_fp, selected_length, dependence_lengths = _block_call_fingerprint(
            cell, chunk_size=min(32, geometry.block_histories)
        )
        block_seconds = _elapsed(block_started)
        block_task = (
            "design_block_calibration"
            if cell.artifact == "design"
            else "production_block_calibration"
        )
        results_by_task[block_task].append(block_fp)
        timings_by_task[block_task].append(
            RehearsalFamilyTiming("block_call", cell.cell_id, block_seconds)
        )

        dependence_started = time.perf_counter()
        dependence_fp = _dependence_fingerprint(
            cell,
            selected_length=selected_length,
            requested_lengths=dependence_lengths,
            history_count=geometry.block_histories,
            chunk_size=min(32, geometry.block_histories),
        )
        dependence_seconds = _elapsed(dependence_started)
        dependence_task = (
            "design_dependence"
            if cell.artifact == "design"
            else "production_dependence"
        )
        results_by_task[dependence_task].append(dependence_fp)
        timings_by_task[dependence_task].append(
            RehearsalFamilyTiming("dependence", cell.cell_id, dependence_seconds)
        )

        kernel_started = time.perf_counter()
        parallel_kernel = _run_rehearsal_kernel(
            config,
            cell,
            selected_length=selected_length,
            windows=geometry.kernel_windows_per_cell,
            workers=geometry.workers,
        )
        kernel_seconds = _elapsed(kernel_started)
        serial_kernel = _run_rehearsal_kernel(
            config,
            cell,
            selected_length=selected_length,
            windows=geometry.kernel_windows_per_cell,
            workers=1,
        )
        if (
            parallel_kernel.execution_fingerprint is None
            or parallel_kernel.execution_fingerprint
            != serial_kernel.execution_fingerprint
        ):
            raise IntegrityError(
                f"rehearsal kernel {cell.cell_id} differs for one versus workers"
            )
        kernel_fp = parallel_kernel.execution_fingerprint
        parity_results.append((f"kernel_{cell.cell_id}", kernel_fp))
        kernel_task = (
            "design_refitted_adequacy"
            if cell.artifact == "design"
            else "production_refitted_adequacy"
        )
        results_by_task[kernel_task].append(kernel_fp)
        timings_by_task[kernel_task].append(
            RehearsalFamilyTiming("kernel", cell.cell_id, kernel_seconds)
        )

    for artifact, fit, task in (
        ("design", selected_fit, "design_latent_correctness"),
        ("production", production_fit, "production_latent_correctness"),
    ):
        generation_started = time.perf_counter()
        latent = generate_latent_chains(
            fit.parameters.transition_matrix,
            chain_count=geometry.latent_chains_per_role,
            measured_months=geometry.latent_measured_months,
            extension_months=geometry.latent_extension_months,
            rng=np.random.default_rng(
                _deterministic_seed(config.run.master_seed, f"latent-{artifact}")
            ),
            chunk_size=min(128, geometry.latent_chains_per_role),
        )
        generation_seconds = _elapsed(generation_started)
        bootstrap_started = time.perf_counter()
        bootstrap = latent_cluster_bootstrap(
            latent,
            expected_chain_count=geometry.latent_chains_per_role,
            bootstrap_replicates=(geometry.latent_bootstrap_replicates_per_role),
            resample_size=geometry.latent_bootstrap_resample_size,
            survival_months=LATENT_SURVIVAL_MONTHS,
            rng=np.random.default_rng(
                _deterministic_seed(config.run.master_seed, f"bootstrap-{artifact}")
            ),
            chunk_size=min(32, geometry.latent_bootstrap_replicates_per_role),
        )
        bootstrap_seconds = _elapsed(bootstrap_started)
        results_by_task[task].extend(
            (_scientific_fingerprint(latent), _scientific_fingerprint(bootstrap))
        )
        timings_by_task[task].extend(
            (
                RehearsalFamilyTiming(
                    "latent_generation", artifact, generation_seconds
                ),
                RehearsalFamilyTiming("latent_bootstrap", artifact, bootstrap_seconds),
            )
        )

    bridge_started = time.perf_counter()
    bridge_jobs = _bridge_benchmark_jobs(
        config,
        selected_fit,
        design_features,
        macro_block_length=design_macro_start,
        coordinates=geometry.bridge_coordinates,
    )
    parallel_bridge = deterministic_process_map(
        _run_benchmark_job,
        bridge_jobs,
        workers=geometry.workers,
    )
    bridge_seconds = _elapsed(bridge_started)
    # Replay the complete two-DGP coordinate family at one worker.  Timing for
    # projection comes only from the nine-worker pass above; this second pass
    # exists solely to prove schedule-invariant numerical content.
    serial_bridge = deterministic_process_map(
        _run_benchmark_job,
        bridge_jobs,
        workers=1,
    )
    parallel_bridge_fp = _bridge_science_fingerprint(parallel_bridge)
    if parallel_bridge_fp != _bridge_science_fingerprint(serial_bridge):
        raise IntegrityError("rehearsal bridge differs for one versus workers")
    parity_results.append(("bridge_all_coordinates", parallel_bridge_fp))
    by_coordinate, qmax = _bridge_elapsed_by_coordinate(
        parallel_bridge,
        coordinates=geometry.bridge_coordinates,
    )
    if geometry.bridge_coordinates == REDUCED_G3_REHEARSAL_GEOMETRY.bridge_coordinates:
        bridge_resource_preflight(
            benchmark_pair_seconds=np.asarray(
                by_coordinate,
                dtype=np.float64,
            ),
            projected_pair_fits=1,
            workers=geometry.workers,
            peak_memory_fraction=0.0,
        )
    results_by_task["bridge_resource_preflight"].append(parallel_bridge_fp)
    for task in (
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
        "actual_artifact_bridge",
    ):
        results_by_task[task].append(
            _sha256(_canonical_bytes({"task": task, "bridge": parallel_bridge_fp}))
        )
    timings_by_task["bridge_resource_preflight"].extend(
        (
            RehearsalFamilyTiming("bridge_benchmark", "total", bridge_seconds),
            RehearsalFamilyTiming("bridge_benchmark", "qmax", qmax),
        )
    )

    for task in (
        "design_fragmentation",
        "production_fragmentation",
        "four_cell_adequacy_holm",
        "four_cell_dependence_holm",
        "artifact_integrity",
    ):
        results_by_task[task].append(
            _sha256(
                _canonical_bytes(
                    {
                        "task": task,
                        "input": calibration_input.fingerprint,
                        "observed_results": sorted(
                            item
                            for values in results_by_task.values()
                            for item in values
                        ),
                    }
                )
            )
        )

    parity_fingerprint = _sha256(_canonical_bytes(parity_results))
    named_seconds = sum(
        timing.wall_seconds
        for timings in timings_by_task.values()
        for timing in timings
        if timing.family != "bridge_benchmark" or timing.cell == "total"
    )
    lightweight_seconds = max(_elapsed(started) - named_seconds, 1e-9)
    timings_by_task["artifact_integrity"].append(
        RehearsalFamilyTiming(
            "lightweight", "remaining_26_task_graph", lightweight_seconds
        )
    )
    probes, dependencies = _build_concrete_task_probes(
        calibration_input_fingerprint=calibration_input.fingerprint,
        scientific_version=config.simulation.scientific_version,
        results_by_task=results_by_task,
        timings_by_task=timings_by_task,
    )
    public_geometry = (
        REDUCED_G3_REHEARSAL_GEOMETRY if geometry.projection_eligible else None
    )
    return _ConcreteReducedG3RehearsalWorkload(
        config_fingerprint=config.fingerprint(),
        calibration_input_fingerprint=calibration_input.fingerprint,
        processed_data_fingerprint=(
            calibration_input.provenance.processed_data_fingerprint
        ),
        raw_snapshot_fingerprint=calibration_input.provenance.raw_snapshot_fingerprint,
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
        scientific_version=config.simulation.scientific_version,
        worker_parity_fingerprint=parity_fingerprint,
        worker_parity_families=("hmm", "kernel", "bridge"),
        selection_policy=OWNER_FIXED_K4_SELECTION_POLICY,
        selected_n_states=OWNER_FIXED_N_STATES,
        bridge_benchmark_n_states=selected_fit.n_states,
        rehearsal_geometry=public_geometry,
        probes=probes,
        expected_dependencies=dependencies,
        _seal=_CONCRETE_WORKLOAD_CAPABILITY,
    )


def _select_owner_fixed_k4_rehearsal_fit(
    fits: dict[int, GaussianHMMFit],
) -> GaussianHMMFit:
    """Return only K4; other fitted families are timing/report evidence."""

    selected = fits.get(OWNER_FIXED_N_STATES)
    if selected is None:
        raise ModelError(
            "rehearsal owner-fixed K4 design HMM is inadmissible",
            details={"report_only_fitted_state_counts": sorted(fits)},
        )
    if not isinstance(selected, GaussianHMMFit) or selected.n_states != 4:
        raise IntegrityError("rehearsal owner-fixed K4 fit mapping is inconsistent")
    return selected


def _require_rehearsal_recurring_history_support(
    pools: DecodedReturnPools,
    *,
    role: Literal["design", "production"],
) -> OwnerFixedK4RecurringHistorySupport:
    """Apply the canonical recurring-state rule before projecting a rehearsal."""

    if not isinstance(pools, DecodedReturnPools) or role not in {
        "design",
        "production",
    }:
        raise ModelError("rehearsal recurring-history source is invalid")
    monthly = pools.monthly_diagnostics
    daily = pools.daily_diagnostics
    if not isinstance(monthly, StatePathDiagnostic) or not isinstance(
        daily,
        StatePathDiagnostic,
    ):
        raise ModelError("rehearsal recurring-history diagnostics are invalid")
    report = owner_fixed_k4_recurring_history_support(
        monthly_episode_counts=tuple(int(value) for value in monthly.episode_counts),
        daily_run_counts=tuple(int(value) for value in daily.episode_counts),
    )
    if not report.passed:
        raise ModelError(
            f"rehearsal {role} K4 recurring-history support failed",
            details={"role": role, **report.as_dict()},
        )
    return report


def _verify_rehearsal_inputs(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> None:
    if not isinstance(config, PRPGConfig):
        raise ModelError("G3 rehearsal requires a resolved PRPGConfig")
    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("G3 rehearsal requires a CalibrationInput")
    _verify_config(config)
    _verify_fingerprints(config, calibration_input)
    _verify_geometry_and_seal(calibration_input)
    _verify_rng_registry()
    _verify_module_constants()
    _verify_environment(inspect_runtime_environment())
    if (
        config.fingerprint() != CANONICAL_CONFIG_FINGERPRINT
        or calibration_input.fingerprint != CANONICAL_CALIBRATION_INPUT_FINGERPRINT
        or calibration_input.provenance.processed_data_fingerprint
        != CANONICAL_PROCESSED_DATA_FINGERPRINT
        or calibration_input.provenance.raw_snapshot_fingerprint
        != CANONICAL_RAW_SNAPSHOT_FINGERPRINT
    ):
        raise IntegrityError("rehearsal canonical input identity changed")


def _fit_rehearsal_hmm(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
    *,
    workers: int,
) -> GaussianHMMFit:
    if len(seeds) == HMM_RESTARTS:
        return fit_gaussian_hmm_restarts(
            features,
            n_states,
            seeds,
            workers=workers,
            owner_fixed_k4_policy=OWNER_FIXED_K4_PROSPECTIVE_POLICY,
        )
    return fit_gaussian_hmm_reduced_diagnostic_restarts(
        features,
        n_states,
        seeds,
        workers=workers,
        owner_fixed_k4_policy=OWNER_FIXED_K4_PROSPECTIVE_POLICY,
    )


def _fit_rehearsal_hmm_outcome(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
    *,
    workers: int,
) -> tuple[GaussianHMMFit | None, str]:
    """Return one fit or a fingerprinted, code-owned scientific rejection.

    Only the two closed HMM rejection types are reduced.  Generic model,
    integrity, filesystem, process, and runtime errors continue to abort the
    rehearsal without a report.
    """

    try:
        fit = _fit_rehearsal_hmm(features, n_states, seeds, workers=workers)
    except (HMMCandidateInadmissibility, NoNumericallyValidHMMRestarts) as error:
        return None, _scientific_fingerprint(
            {
                "schema_id": "prpg-g3-rehearsal-hmm-rejection-v1",
                "n_states": n_states,
                "failure_type": type(error).__name__,
                "message": error.message,
                "details": error.details,
                "authority": False,
            }
        )
    return fit, gaussian_hmm_fit_fingerprint(fit)


def _rehearsal_block_start_geometry(
    design_monthly_start: int,
    design_daily_start: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Bind measured design starts and conservative production maxima."""

    _positive_integer(design_monthly_start, "design monthly rehearsal start")
    _positive_integer(design_daily_start, "design daily rehearsal start")
    if (
        not MONTHLY_BLOCK_LENGTH_BOUNDS[0]
        <= design_monthly_start
        <= (MONTHLY_BLOCK_LENGTH_BOUNDS[1])
    ):
        raise ModelError("design monthly rehearsal start is outside policy bounds")
    if (
        not DAILY_BLOCK_LENGTH_BOUNDS[0]
        <= design_daily_start
        <= (DAILY_BLOCK_LENGTH_BOUNDS[1])
    ):
        raise ModelError("design daily rehearsal start is outside policy bounds")
    return (
        (design_monthly_start, design_daily_start),
        (MONTHLY_BLOCK_LENGTH_BOUNDS[1], DAILY_BLOCK_LENGTH_BOUNDS[1]),
    )


def _rehearsal_cells(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    *,
    design_fit: GaussianHMMFit,
    production_fit: GaussianHMMFit,
    geometry: _ConcreteRehearsalGeometry,
    design_starts: tuple[int, int],
    production_starts: tuple[int, int],
) -> tuple[_RehearsalCell, ...]:
    cells: list[_RehearsalCell] = []
    for artifact, source, fit, starts in (
        ("design", calibration_input.design, design_fit, design_starts),
        ("production", calibration_input.full, production_fit, production_starts),
    ):
        monthly_states = np.asarray(fit.decoded_states, dtype=np.int64)
        if monthly_states.shape != source.monthly_continuation.shape:
            raise ModelError("rehearsal HMM path does not match its source slice")
        daily_states = map_monthly_states_to_daily(
            source.monthly_period_ordinals,
            monthly_states,
            source.daily_session_ns,
        )
        monthly_segments = sequence_lengths_from_continuation(
            source.monthly_continuation
        )
        daily_segments = sequence_lengths_from_continuation(source.daily_continuation)
        monthly_links = forward_links_from_row_start_mask(source.monthly_continuation)
        daily_links = forward_links_from_row_start_mask(source.daily_continuation)
        for frequency, returns, states, links, daily_lengths, intended_length in (
            (
                "monthly",
                source.monthly_returns,
                monthly_states,
                monthly_links,
                None,
                starts[0],
            ),
            (
                "daily",
                source.daily_returns,
                daily_states,
                daily_links,
                daily_segments,
                starts[1],
            ),
        ):
            materialization = materialize_registered_block_uniforms(
                master_seed=config.run.master_seed,
                scientific_version=config.simulation.scientific_version,
                artifact=cast(Artifact, artifact),
                frequency=cast(Literal["monthly", "daily"], frequency),
                history_count=geometry.block_histories,
                chunk_size=min(32, geometry.block_histories),
                strict_canonical=False,
                monthly_segment_lengths=monthly_segments,
                daily_segment_lengths=daily_lengths,
            )
            cells.append(
                _RehearsalCell(
                    artifact=cast(Artifact, artifact),
                    frequency=cast(Literal["monthly", "daily"], frequency),
                    source_slice=source,
                    model=fit,
                    historical_returns=np.asarray(returns, dtype=np.float64),
                    historical_states=np.asarray(states, dtype=np.int64),
                    continuation_links=np.asarray(links, dtype=np.bool_),
                    intended_length=intended_length,
                    monthly_segment_lengths=monthly_segments,
                    daily_segment_lengths=daily_lengths,
                    materialization=materialization,
                )
            )
    return tuple(cells)


def _block_call_fingerprint(
    cell: _RehearsalCell,
    *,
    chunk_size: int,
) -> tuple[str, int, tuple[int, ...]]:
    execution = run_materialized_block_calibration(
        cell.materialization,
        model=cell.model,
        starting_length=cell.intended_length,
        historical_source_states=cell.historical_states,
        source_continuation_links=cell.continuation_links,
        chunk_size=chunk_size,
    )
    return (
        _scientific_fingerprint(execution),
        execution.selected_length,
        execution.dependence_reuse_lengths,
    )


def _dependence_fingerprint(
    cell: _RehearsalCell,
    *,
    selected_length: int,
    requested_lengths: tuple[int, ...],
    history_count: int,
    chunk_size: int,
) -> str:
    row_mask = (
        cell.source_slice.monthly_continuation
        if cell.frequency == "monthly"
        else cell.source_slice.daily_continuation
    )
    reference = fit_dependence_reference(
        cell.historical_returns,
        row_mask,
        frequency=cell.frequency,
    )
    reducer = DependenceTraceReducer(
        artifact=cell.artifact,
        selected_length=selected_length,
        requested_lengths=requested_lengths,
        historical_returns=cell.historical_returns,
        reference=reference,
        history_count=history_count,
        strict_canonical=False,
        monthly_segment_lengths=cell.monthly_segment_lengths,
        daily_segment_lengths=cell.daily_segment_lengths,
    )
    for chunk in iter_materialized_block_trace_chunks(
        cell.materialization,
        model=cell.model,
        historical_source_states=cell.historical_states,
        source_continuation_links=cell.continuation_links,
        intended_lengths=requested_lengths,
        chunk_size=chunk_size,
    ):
        reducer.consume(chunk)
    return _scientific_fingerprint(reducer.finalize())


def _run_rehearsal_kernel(
    config: PRPGConfig,
    cell: _RehearsalCell,
    *,
    selected_length: int,
    windows: int,
    workers: int,
) -> KernelCalibrationExecution:
    return run_registered_kernel_calibration_cell(
        model=cell.model,
        master_seed=config.run.master_seed,
        scientific_version=config.simulation.scientific_version,
        artifact=cell.artifact,
        frequency=cell.frequency,
        selected_length=selected_length,
        historical_returns=cell.historical_returns,
        historical_source_states=cell.historical_states,
        source_continuation_links=cell.continuation_links,
        model_artifact_fingerprint=gaussian_hmm_fit_fingerprint(cell.model),
        target_history_fingerprint=scientific_array_fingerprint(
            cell.historical_returns,
            role=f"rehearsal_{cell.cell_id}_kernel_target",
        ),
        workers=workers,
        prefix_window_counts=(windows,),
        annualized_half_width_max=float(np.finfo(np.float64).max),
        statistics_batch_windows=min(128, windows),
        strict_canonical=False,
        verify=True,
        monthly_segment_lengths=cell.monthly_segment_lengths,
        daily_segment_lengths=cell.daily_segment_lengths,
    )


def _bridge_benchmark_jobs(
    config: PRPGConfig,
    design: GaussianHMMFit,
    design_features: HMMFeatureMatrix,
    *,
    macro_block_length: int,
    coordinates: int,
) -> tuple[_BenchmarkJob, ...]:
    binding_fingerprint = _sha256(
        _canonical_bytes(
            {
                "schema_id": "prpg-g3-rehearsal-bridge-binding-v1",
                "model": gaussian_hmm_fit_fingerprint(design),
                "macro_block_length": macro_block_length,
                "coordinates": coordinates,
                "authority": False,
            }
        )
    )
    history = np.asarray(design_features.values, dtype=np.float64)
    continuation = np.ones(len(history), dtype=np.bool_)
    continuation[0] = False
    return tuple(
        _BenchmarkJob(
            dgp=dgp,
            pair=pair,
            binding_fingerprint=binding_fingerprint,
            selected_n_states=design.n_states,
            master_seed=config.run.master_seed,
            scientific_version=config.simulation.scientific_version,
            design_parameters=design.parameters,
            design_history=history,
            design_continuation=continuation,
            macro_block_length=macro_block_length,
        )
        for dgp in BRIDGE_DGPS
        for pair in range(coordinates)
    )


def _bridge_science_fingerprint(
    values: Sequence[BridgeBenchmarkPairEvidence],
) -> str:
    return _sha256(
        _canonical_bytes(
            [
                {
                    "dgp": item.dgp,
                    "pair": item.pair,
                    "scientific_fingerprint": item.scientific_fingerprint,
                }
                for item in values
            ]
        )
    )


def _bridge_elapsed_by_coordinate(
    values: Sequence[_BridgeElapsedRecord],
    *,
    coordinates: int,
) -> tuple[tuple[float, ...], float]:
    """Reduce the exact two-DGP benchmark family to pair maxima and qmax."""

    _positive_integer(coordinates, "rehearsal bridge coordinates")
    expected = tuple((dgp, pair) for dgp in BRIDGE_DGPS for pair in range(coordinates))
    observed: list[tuple[str, int]] = []
    elapsed_by_pair: list[list[float]] = [[] for _ in range(coordinates)]
    for item in values:
        dgp = item.dgp
        pair = item.pair
        elapsed = item.elapsed_seconds
        if dgp not in BRIDGE_DGPS:
            raise ModelError("rehearsal bridge record has an unregistered DGP")
        if type(pair) is not int or not 0 <= pair < coordinates:
            raise ModelError("rehearsal bridge record has an invalid coordinate")
        _positive_float(elapsed, "rehearsal bridge record elapsed seconds")
        observed.append((dgp, pair))
        elapsed_by_pair[pair].append(float(elapsed))
    if tuple(observed) != expected or any(
        len(items) != len(BRIDGE_DGPS) for items in elapsed_by_pair
    ):
        raise ModelError(
            "rehearsal bridge timing family must contain both DGPs in exact order"
        )
    maxima = tuple(max(items) for items in elapsed_by_pair)
    return maxima, max(maxima)


def _build_concrete_task_probes(
    *,
    calibration_input_fingerprint: str,
    scientific_version: int,
    results_by_task: dict[str, list[str]],
    timings_by_task: dict[str, list[RehearsalFamilyTiming]],
) -> tuple[
    tuple[ReducedG3TaskProbe, ...],
    tuple[tuple[tuple[str, str], ...], ...],
]:
    probes: list[ReducedG3TaskProbe] = []
    dependencies_by_task: list[tuple[tuple[str, str], ...]] = []
    fingerprints: dict[str, str] = {}
    for spec in G3_TASK_REGISTRY:
        dependencies = tuple(
            (task_id, fingerprints[task_id]) for task_id in spec.dependencies
        )
        numerical = tuple(results_by_task[spec.task_id])
        if not numerical:
            numerical = (
                _sha256(
                    _canonical_bytes(
                        {
                            "task": spec.task_id,
                            "dependencies": list(dependencies),
                        }
                    )
                ),
            )
        fingerprint = rehearsal_task_fingerprint(
            calibration_input_fingerprint=calibration_input_fingerprint,
            scientific_version=scientific_version,
            task_id=spec.task_id,
            dependency_fingerprints=dependencies,
            result_fingerprints=numerical,
        )
        serialized = len(
            _canonical_bytes(
                {
                    "task": spec.task_id,
                    "dependencies": list(dependencies),
                    "numerical_results": list(numerical),
                    "fingerprint": fingerprint,
                }
            )
        )
        probes.append(
            ReducedG3TaskProbe(
                spec.task_id,
                fingerprint,
                tuple(timings_by_task[spec.task_id]),
                serialized,
            )
        )
        dependencies_by_task.append(dependencies)
        fingerprints[spec.task_id] = fingerprint
    return tuple(probes), tuple(dependencies_by_task)


def assemble_noncanonical_g3_rehearsal_report(
    workload: _ConcreteReducedG3RehearsalWorkload,
    telemetry: G3RehearsalTelemetry,
) -> NoncanonicalG3RehearsalReport:
    """Walk one executor-minted workload and return an unreleasable report."""

    if (
        type(workload) is not _ConcreteReducedG3RehearsalWorkload
        or _MINTED_WORKLOADS.get(id(workload)) is not workload
    ):
        raise ModelError(
            "G3 rehearsal report requires an executor-minted concrete workload"
        )
    binding = _BOUND_TELEMETRY.get(id(workload))
    if binding is None or binding[0] is not workload or binding[1] is not telemetry:
        raise ModelError(
            "G3 rehearsal report requires telemetry captured with that workload"
        )

    if (
        type(workload.rehearsal_geometry) is not ReducedG3RehearsalGeometry
        or workload.rehearsal_geometry != REDUCED_G3_REHEARSAL_GEOMETRY
    ):
        raise ModelError(
            "only the exact fixed rehearsal geometry may feed the launch projection"
        )
    if (
        workload.selection_policy != OWNER_FIXED_K4_SELECTION_POLICY
        or workload.selected_n_states != OWNER_FIXED_N_STATES
        or workload.bridge_benchmark_n_states != OWNER_FIXED_N_STATES
    ):
        raise IntegrityError("rehearsal workload is not bound to owner-fixed K4")

    for value, label in (
        (workload.config_fingerprint, "rehearsal config fingerprint"),
        (
            workload.calibration_input_fingerprint,
            "rehearsal calibration-input fingerprint",
        ),
        (
            workload.processed_data_fingerprint,
            "rehearsal processed-data fingerprint",
        ),
        (workload.raw_snapshot_fingerprint, "rehearsal raw-snapshot fingerprint"),
        (workload.source_code_fingerprint, "rehearsal source-code fingerprint"),
        (
            workload.dependency_lock_fingerprint,
            "rehearsal dependency-lock fingerprint",
        ),
    ):
        _require_sha256(value, label)
    _positive_integer(workload.scientific_version, "rehearsal scientific version")
    for name in ("native_thread_cap_verified", "worker_parity_verified"):
        if type(getattr(workload, name)) is not bool:
            raise ModelError(f"rehearsal {name} must be boolean")
    _require_sha256(
        workload.worker_parity_fingerprint,
        "rehearsal one-vs-nine parity fingerprint",
    )
    if (
        type(workload.worker_parity_families) is not tuple
        or not workload.worker_parity_families
        or any(
            type(item) is not str or not item
            for item in workload.worker_parity_families
        )
    ):
        raise ModelError("rehearsal parity families must be a non-empty tuple")
    if type(telemetry) is not G3RehearsalTelemetry:
        raise ModelError("rehearsal requires the exact telemetry type")

    del _BOUND_TELEMETRY[id(workload)]
    del _MINTED_WORKLOADS[id(workload)]
    observations, timings, serialized_probe_bytes = _walk_rehearsal_task_graph(workload)

    measurements = _projection_measurements(
        timings,
        telemetry,
        serialized_probe_bytes=max(1, serialized_probe_bytes),
        native_thread_cap_verified=workload.native_thread_cap_verified,
        worker_parity_verified=workload.worker_parity_verified,
        bridge_benchmark_n_states=workload.bridge_benchmark_n_states,
    )
    report = _report_with_stable_serialized_size(
        config_fingerprint=workload.config_fingerprint,
        calibration_input_fingerprint=workload.calibration_input_fingerprint,
        processed_data_fingerprint=workload.processed_data_fingerprint,
        raw_snapshot_fingerprint=workload.raw_snapshot_fingerprint,
        source_code_fingerprint=workload.source_code_fingerprint,
        dependency_lock_fingerprint=workload.dependency_lock_fingerprint,
        worker_parity_fingerprint=workload.worker_parity_fingerprint,
        worker_parity_families=workload.worker_parity_families,
        selection_policy=workload.selection_policy,
        selected_n_states=workload.selected_n_states,
        scientific_version=workload.scientific_version,
        tasks=observations,
        measurements=measurements,
    )
    _MINTED_REPORTS[id(report)] = report
    verify_noncanonical_g3_rehearsal_report(report)
    return report


def _walk_rehearsal_task_graph(
    workload: ReducedG3RehearsalWorkload,
) -> tuple[
    tuple[G3RehearsalTaskObservation, ...],
    tuple[RehearsalFamilyTiming, ...],
    int,
]:
    """Traverse and bind the closed DAG without creating gate evidence.

    This pure helper exists for graph-level tests.  Only the concrete,
    telemetry-bound wrapper above can mint a report or resource projection.
    """

    observations: list[G3RehearsalTaskObservation] = []
    timings: list[RehearsalFamilyTiming] = []
    fingerprints: dict[str, str] = {}
    serialized_probe_bytes = 0
    for ordinal, spec in enumerate(G3_TASK_REGISTRY, start=1):
        dependencies = tuple(
            (task_id, fingerprints[task_id]) for task_id in spec.dependencies
        )
        raw = workload.execute_task(spec.task_id, dependencies)
        if type(raw) is not ReducedG3TaskProbe:
            raise ModelError("rehearsal workload returned a substituted probe type")
        if raw.task_id != spec.task_id:
            raise IntegrityError("rehearsal workload returned the wrong task")
        observation = G3RehearsalTaskObservation(
            ordinal=ordinal,
            task_id=spec.task_id,
            dependency_fingerprints=dependencies,
            result_fingerprint=raw.result_fingerprint,
        )
        observations.append(observation)
        timings.extend(raw.timings)
        serialized_probe_bytes += raw.serialized_bytes
        fingerprints[spec.task_id] = raw.result_fingerprint
    return tuple(observations), tuple(timings), serialized_probe_bytes


def verify_noncanonical_g3_rehearsal_report(
    value: NoncanonicalG3RehearsalReport,
) -> None:
    """Reverify exact task closure, projection algebra, and no-authority flags."""

    if type(value) is not NoncanonicalG3RehearsalReport:
        raise ModelError("G3 rehearsal report has a substituted type")
    if _MINTED_REPORTS.get(id(value)) is not value:
        raise IntegrityError("G3 rehearsal report was not minted by the concrete run")
    if (
        value.schema_id != G3_REHEARSAL_REPORT_SCHEMA
        or value.contract_version != G3_REHEARSAL_CONTRACT_VERSION
        or value.execution_mode != "rehearsal"
        or value.g3_contract_version != G3_CONTRACT_VERSION
        or value.block_policy_version != BLOCK_POLICY_VERSION
        or value.selection_policy != OWNER_FIXED_K4_SELECTION_POLICY
        or value.selected_n_states != OWNER_FIXED_N_STATES
        or value.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT
        or value.geometry != REDUCED_G3_REHEARSAL_GEOMETRY
        or value.g3_passed is not False
        or value.canonical_authority is not False
        or value.generation_authorized is not False
        or value.releasable is not False
    ):
        raise IntegrityError("G3 rehearsal report claims invalid authority or scope")
    _require_sha256(
        value.worker_parity_fingerprint,
        "rehearsal one-vs-nine parity fingerprint",
    )
    if (
        type(value.worker_parity_families) is not tuple
        or not value.worker_parity_families
        or any(
            type(item) is not str or not item for item in value.worker_parity_families
        )
    ):
        raise IntegrityError("G3 rehearsal parity-family evidence is incomplete")
    for identity, label in (
        (value.config_fingerprint, "rehearsal config fingerprint"),
        (
            value.calibration_input_fingerprint,
            "rehearsal calibration-input fingerprint",
        ),
        (
            value.processed_data_fingerprint,
            "rehearsal processed-data fingerprint",
        ),
        (value.raw_snapshot_fingerprint, "rehearsal raw-snapshot fingerprint"),
        (value.source_code_fingerprint, "rehearsal source-code fingerprint"),
        (
            value.dependency_lock_fingerprint,
            "rehearsal dependency-lock fingerprint",
        ),
    ):
        _require_sha256(identity, label)
    _positive_integer(value.scientific_version, "rehearsal scientific version")
    if type(value.tasks) is not tuple or len(value.tasks) != len(G3_GATE_ORDER):
        raise IntegrityError("G3 rehearsal report must contain exactly 26 tasks")
    observed: dict[str, str] = {}
    for ordinal, (spec, task) in enumerate(
        zip(G3_TASK_REGISTRY, value.tasks, strict=True),
        start=1,
    ):
        if type(task) is not G3RehearsalTaskObservation:
            raise IntegrityError("G3 rehearsal task has a substituted type")
        expected_dependencies = tuple(
            (task_id, observed[task_id]) for task_id in spec.dependencies
        )
        if (
            task.ordinal != ordinal
            or task.task_id != spec.task_id
            or task.dependency_fingerprints != expected_dependencies
            or task.outcome != "graph_traversed_noncanonical"
        ):
            raise IntegrityError("G3 rehearsal task graph is inconsistent")
        observed[task.task_id] = task.result_fingerprint
    if type(value.measurements) is not G3ResourceProjectionMeasurements:
        raise IntegrityError("G3 rehearsal report omitted exact measurements")
    if type(value.resource_projection) is not G3ResourceProjection:
        raise IntegrityError("G3 rehearsal report omitted exact resource projection")
    recomputed = project_canonical_g3_resources(value.measurements)
    if recomputed != value.resource_projection:
        raise IntegrityError("G3 rehearsal resource projection changed")
    expected_fingerprint = _sha256(
        _canonical_bytes(_report_identity(value, include_fingerprint=False))
    )
    if value.report_fingerprint != expected_fingerprint:
        raise IntegrityError("G3 rehearsal report fingerprint changed")


def _projection_measurements(
    timings: tuple[RehearsalFamilyTiming, ...],
    telemetry: G3RehearsalTelemetry,
    *,
    serialized_probe_bytes: int,
    native_thread_cap_verified: bool,
    worker_parity_verified: bool,
    bridge_benchmark_n_states: int,
) -> G3ResourceProjectionMeasurements:
    by_family: dict[RehearsalFamily, list[RehearsalFamilyTiming]] = {}
    for item in timings:
        if type(item) is not RehearsalFamilyTiming:
            raise ModelError("rehearsal timings contain a substituted type")
        by_family.setdefault(item.family, []).append(item)

    hmm = cast(
        tuple[float, float, float, float],
        _ordered_cells(by_family, "hmm", ("k2", "k3", "k4", "k5")),
    )
    block_materialization = _single_or_sum(by_family, "block_materialization")
    block_calls = cast(
        tuple[float, float, float, float],
        _ordered_cells(
            by_family,
            "block_call",
            (
                "design_monthly",
                "design_daily",
                "production_monthly",
                "production_daily",
            ),
        ),
    )
    kernels = cast(
        tuple[float, float, float, float],
        _ordered_cells(
            by_family,
            "kernel",
            (
                "design_monthly",
                "design_daily",
                "production_monthly",
                "production_daily",
            ),
        ),
    )
    dependence = _single_or_sum(by_family, "dependence")
    latent_generation = cast(
        tuple[float, float],
        _ordered_cells(
            by_family,
            "latent_generation",
            ("design", "production"),
        ),
    )
    latent_bootstrap = cast(
        tuple[float, float],
        _ordered_cells(
            by_family,
            "latent_bootstrap",
            ("design", "production"),
        ),
    )
    bridge = _ordered_cells(
        by_family,
        "bridge_benchmark",
        ("total", "qmax"),
    )
    if bridge[1] > bridge[0]:
        raise ModelError("rehearsal bridge qmax exceeds total wall time")
    lightweight = _single_or_sum(by_family, "lightweight")

    observed_wall = (
        sum(hmm)
        + block_materialization
        + sum(block_calls)
        + sum(kernels)
        + dependence
        + sum(latent_generation)
        + sum(latent_bootstrap)
        + bridge[0]
        + lightweight
    )
    utilization = telemetry.process_tree_cpu_seconds / (
        REDUCED_G3_REHEARSAL_GEOMETRY.workers * observed_wall
    )
    return G3ResourceProjectionMeasurements(
        workers=REDUCED_G3_REHEARSAL_GEOMETRY.workers,
        hmm_wall_seconds_by_k=hmm,
        block_materialization_wall_seconds=block_materialization,
        block_call_wall_seconds=block_calls,
        kernel_wall_seconds=kernels,
        dependence_wall_seconds=dependence,
        latent_generation_wall_seconds=latent_generation,
        latent_bootstrap_wall_seconds=latent_bootstrap,
        bridge_benchmark_wall_seconds=bridge[0],
        bridge_qmax_seconds=bridge[1],
        bridge_benchmark_n_states=bridge_benchmark_n_states,
        lightweight_wall_seconds=lightweight,
        peak_process_tree_rss_bytes=telemetry.peak_process_tree_rss_bytes,
        process_tree_cpu_seconds=telemetry.process_tree_cpu_seconds,
        worker_capacity_utilization=utilization,
        swap_used_before_bytes=telemetry.swap_used_before_bytes,
        maximum_swap_used_bytes=telemetry.maximum_swap_used_bytes,
        swap_used_after_bytes=telemetry.swap_used_after_bytes,
        disk_free_bytes=telemetry.disk_free_bytes,
        serialized_bytes=serialized_probe_bytes,
        native_thread_cap_verified=native_thread_cap_verified,
        worker_parity_verified=worker_parity_verified,
        rss_telemetry_complete=telemetry.rss_telemetry_complete,
        swap_telemetry_complete=telemetry.swap_telemetry_complete,
        cpu_telemetry_complete=telemetry.cpu_telemetry_complete,
        disk_telemetry_complete=telemetry.disk_telemetry_complete,
    )


def _report_with_stable_serialized_size(
    *,
    config_fingerprint: str,
    calibration_input_fingerprint: str,
    processed_data_fingerprint: str,
    raw_snapshot_fingerprint: str,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
    worker_parity_fingerprint: str,
    worker_parity_families: tuple[str, ...],
    selection_policy: Literal["owner_fixed_k4_v3"],
    selected_n_states: Literal[4],
    scientific_version: int,
    tasks: tuple[G3RehearsalTaskObservation, ...],
    measurements: G3ResourceProjectionMeasurements,
) -> NoncanonicalG3RehearsalReport:
    """Bind the exact emitted byte count without a filesystem side effect."""

    size = measurements.serialized_bytes
    report: NoncanonicalG3RehearsalReport | None = None
    for _ in range(16):
        adjusted = replace(measurements, serialized_bytes=size)
        projection = project_canonical_g3_resources(adjusted)
        provisional = NoncanonicalG3RehearsalReport(
            config_fingerprint=config_fingerprint,
            calibration_input_fingerprint=calibration_input_fingerprint,
            processed_data_fingerprint=processed_data_fingerprint,
            raw_snapshot_fingerprint=raw_snapshot_fingerprint,
            source_code_fingerprint=source_code_fingerprint,
            dependency_lock_fingerprint=dependency_lock_fingerprint,
            worker_parity_fingerprint=worker_parity_fingerprint,
            worker_parity_families=worker_parity_families,
            selection_policy=selection_policy,
            selected_n_states=selected_n_states,
            scientific_version=scientific_version,
            tasks=tasks,
            measurements=adjusted,
            resource_projection=projection,
        )
        fingerprint = _sha256(
            _canonical_bytes(_report_identity(provisional, include_fingerprint=False))
        )
        report = replace(provisional, report_fingerprint=fingerprint)
        encoded_size = len(
            _canonical_bytes(_report_identity(report, include_fingerprint=True))
        )
        if encoded_size == size:
            return report
        size = max(encoded_size, measurements.serialized_bytes)
    raise IntegrityError("G3 rehearsal serialized-size identity did not converge")


def _ordered_cells(
    by_family: dict[RehearsalFamily, list[RehearsalFamilyTiming]],
    family: RehearsalFamily,
    cells: tuple[str, ...],
) -> tuple[float, ...]:
    supplied = by_family.get(family, [])
    if tuple(item.cell for item in supplied) != cells:
        raise ModelError(
            "rehearsal family timing geometry is incomplete",
            details={
                "family": family,
                "actual": [item.cell for item in supplied],
                "expected": list(cells),
            },
        )
    return tuple(item.wall_seconds for item in supplied)


def _single_or_sum(
    by_family: dict[RehearsalFamily, list[RehearsalFamilyTiming]],
    family: RehearsalFamily,
) -> float:
    supplied = by_family.get(family, [])
    if not supplied:
        raise ModelError(
            "rehearsal family timing geometry is incomplete",
            details={"family": family},
        )
    return float(sum(item.wall_seconds for item in supplied))


def _report_identity(
    value: NoncanonicalG3RehearsalReport,
    *,
    include_fingerprint: bool,
) -> dict[str, object]:
    measurements = value.measurements
    projection = value.resource_projection
    result: dict[str, object] = {
        "schema_id": value.schema_id,
        "contract_version": value.contract_version,
        "execution_mode": value.execution_mode,
        "g3_contract_version": value.g3_contract_version,
        "block_policy_version": value.block_policy_version,
        "execution_closure_schema": G3_EXECUTION_CLOSURE_SCHEMA,
        "execution_closure_manifest": G3_EXECUTION_CLOSURE_MANIFEST.as_dict(),
        "execution_closure_manifest_fingerprint": (
            G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
        ),
        "config_fingerprint": value.config_fingerprint,
        "calibration_input_fingerprint": value.calibration_input_fingerprint,
        "processed_data_fingerprint": value.processed_data_fingerprint,
        "raw_snapshot_fingerprint": value.raw_snapshot_fingerprint,
        "source_code_fingerprint": value.source_code_fingerprint,
        "dependency_lock_fingerprint": value.dependency_lock_fingerprint,
        "worker_parity_fingerprint": value.worker_parity_fingerprint,
        "worker_parity_families": list(value.worker_parity_families),
        "selection_policy": value.selection_policy,
        "selected_n_states": value.selected_n_states,
        "scientific_version": value.scientific_version,
        "task_registry_fingerprint": value.task_registry_fingerprint,
        "geometry": asdict(value.geometry),
        "tasks": [asdict(item) for item in value.tasks],
        "measurements": None if measurements is None else asdict(measurements),
        "resource_projection": None if projection is None else asdict(projection),
        "g3_passed": value.g3_passed,
        "canonical_authority": value.canonical_authority,
        "generation_authorized": value.generation_authorized,
        "releasable": value.releasable,
    }
    if include_fingerprint:
        result["report_fingerprint"] = value.report_fingerprint
    return result


def rehearsal_task_fingerprint(
    *,
    calibration_input_fingerprint: str,
    scientific_version: int,
    task_id: str,
    dependency_fingerprints: tuple[tuple[str, str], ...],
    result_fingerprints: tuple[str, ...],
) -> str:
    """Hash a reduced observation without minting a scientific task binding."""

    _require_sha256(calibration_input_fingerprint, "calibration input fingerprint")
    _positive_integer(scientific_version, "scientific version")
    if task_id not in G3_GATE_ORDER:
        raise ModelError("rehearsal fingerprint task is outside the registry")
    for dependency_id, fingerprint in dependency_fingerprints:
        if dependency_id not in G3_GATE_ORDER:
            raise ModelError("rehearsal fingerprint dependency is unregistered")
        _require_sha256(fingerprint, "rehearsal dependency fingerprint")
    for fingerprint in result_fingerprints:
        _require_sha256(fingerprint, "rehearsal numerical result fingerprint")
    return _sha256(
        _canonical_bytes(
            {
                "schema_id": "prpg-g3-reduced-task-observation-v1",
                "rehearsal_contract": G3_REHEARSAL_CONTRACT_VERSION,
                "calibration_input_fingerprint": calibration_input_fingerprint,
                "scientific_version": scientific_version,
                "task_id": task_id,
                "dependency_fingerprints": list(dependency_fingerprints),
                "result_fingerprints": list(result_fingerprints),
                "outcome": "graph_traversed_noncanonical",
                "authority": False,
            }
        )
    )


def _scientific_fingerprint(value: object) -> str:
    return _sha256(_canonical_bytes(_scientific_content(value)))


def _scientific_content(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha256(array.tobytes(order="C")),
        }
    if isinstance(value, np.generic):
        return _scientific_content(value.item())
    if isinstance(value, Enum):
        return _scientific_content(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _scientific_content(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, tuple | list):
        return [_scientific_content(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _scientific_content(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("rehearsal numerical result contains a non-finite scalar")
        return value
    raise ModelError(
        "rehearsal numerical result contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _native_thread_probe(_item: int) -> bool:
    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    environment_capped = all(os.environ.get(name) == "1" for name in variables)
    pools_capped = all(
        int(item.get("num_threads", 0)) <= 1 for item in threadpool_info()
    )
    return environment_capped and pools_capped


def _elapsed(started: float) -> float:
    value = time.perf_counter() - started
    if not math.isfinite(value) or value < 0.0:
        raise ModelError("rehearsal monotonic timing is unavailable")
    return max(float(value), 1e-9)


def _deterministic_seed(master_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():  # pragma: no cover - filesystem root exists.
        raise ModelError("rehearsal disk telemetry path is unavailable")
    return candidate


def _disk_free_bytes(path: Path) -> int | None:
    try:
        values = os.statvfs(path)
    except OSError:
        return None
    return int(values.f_bavail * values.f_frsize)


def _swap_used_bytes() -> int | None:
    try:
        return int(psutil.swap_memory().used)
    except (OSError, NotImplementedError, psutil.Error):
        return None


def _process_cpu_seconds(process: psutil.Process) -> float:
    try:
        values = process.cpu_times()
    except (OSError, psutil.Error) as error:
        raise ModelError("rehearsal parent CPU telemetry is unavailable") from error
    return float(values.user + values.system)


def _completed_children_cpu_seconds() -> float:
    values = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(values.ru_utime + values.ru_stime)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelError(f"{label} must be a lowercase SHA-256")


def _positive_float(value: object, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ModelError(f"{label} must be a positive finite float")


def _positive_integer(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ModelError(f"{label} must be a positive integer")


def _nonnegative_integer(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ModelError(f"{label} must be a non-negative integer")
