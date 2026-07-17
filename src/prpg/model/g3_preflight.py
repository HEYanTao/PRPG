"""Fail-closed canonical preflight for the approved Phase-3 launch.

The function in this module performs no fit, simulation, bootstrap, random
draw, artifact publication, or launch-state mutation.  It verifies that the
resolved contract, sealed calibration input, pinned target-Mac environment,
RNG registry, vendored PPW source, and conservative resource budget are the
exact prerequisites approved in Sections 4.5 and 4.6.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from prpg.config import PRPGConfig
from prpg.errors import IntegrityError, ModelError
from prpg.execution import ResourceSnapshot, memory_preflight
from prpg.model.adequacy import (
    LATENT_BOOTSTRAP_REPLICATES,
    LATENT_BOOTSTRAP_RESAMPLE_SIZE,
    LATENT_CHAIN_COUNT,
    LATENT_EXTENSION_MONTHS,
    LATENT_MEASURED_MONTHS,
)
from prpg.model.alpha import BaseFrequency
from prpg.model.block_execution import (
    CANONICAL_HISTORY_COUNT,
    MINIMUM_COMPLETED_EPISODES,
    MINIMUM_EFFECTIVE_FRACTION,
    Artifact,
)
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.bridge import (
    ACCELERATOR_AUDIT_PAIRS,
    ACCELERATOR_MINIMUM_AGREEMENTS,
    TIER_A_PAIRS,
    TIER_B_BATCH_SIZE,
    TIER_B_INTERVAL_ERROR,
    TIER_B_MAXIMUM_PAIRS,
    TIER_B_TAIL_PROBABILITY,
    BridgeResourcePreflight,
)
from prpg.model.bridge_selection import (
    BRIDGE_FIXED_K4_TIER_A_FITS_PER_MEMBER,
    BRIDGE_FIXED_K4_TIER_A_POLICY,
    BRIDGE_FIXED_K4_TIER_A_RESTARTS_PER_PAIR,
    BRIDGE_FIXED_K4_TIER_A_ROLLING_FOLDS,
    BRIDGE_FIXED_K4_TIER_A_USES_BIC,
    BRIDGE_FIXED_K4_TIER_A_USES_CROSS_K_SELECTION,
)
from prpg.model.candidate_gate import (
    OWNER_FIXED_MINIMUM_OBSERVATIONS,
    OWNER_FIXED_MINIMUM_OBSERVED_EPISODES,
    OWNER_FIXED_MINIMUM_OCCUPANCY,
    OWNER_FIXED_N_STATES,
)
from prpg.model.fixed_point import (
    MAXIMUM_FIXED_POINT_ITERATIONS,
    OWNER_FIXED_K4_FIXED_POINT_STATE_CONTRACT,
)
from prpg.model.g3 import G3_CONTRACT_VERSION, G3_RESOURCE_PROJECTION_VERSION
from prpg.model.g3_registry import (
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.hmm import (
    DEFAULT_EIGENVALUE_FLOOR,
    DEFAULT_MAX_COVARIANCE_CONDITION,
    HMM_RESTARTS,
    LIKELIHOOD_DECREASE_RELATIVE_TOLERANCE,
    MAXIMUM_COVARIANCE_FLOOR_RELATIVE_CHANGE,
    MAXIMUM_MIXING_MONTHS,
    MAXIMUM_NORMALIZED_POSTERIOR_ENTROPY,
    MAXIMUM_SELF_TRANSITION,
    MINIMUM_ASSIGNED_STATE_POSTERIOR,
    MINIMUM_MEAN_MAXIMUM_POSTERIOR,
    MINIMUM_PAIRWISE_MAHALANOBIS_DISTANCE,
    MINIMUM_POSTERIOR_EFFECTIVE_FRACTION,
    MINIMUM_POSTERIOR_EFFECTIVE_MASS,
    MIXING_TOTAL_VARIATION_THRESHOLD,
    NEGATIVE_PROBABILITY_TOLERANCE,
    NUMERICAL_EDGE_THRESHOLD,
    PROBABILITY_ATOL,
    RAW_COVARIANCE_EIGENVALUE_TOLERANCE,
    STATIONARY_RESIDUAL_ATOL,
)
from prpg.model.input import ASSET_COLUMNS, MACRO_COLUMNS, CalibrationInput
from prpg.model.kernel_execution import registered_kernel_memory_estimate
from prpg.model.ppw_reference import (
    REFERENCE_BYTES,
    REFERENCE_SHA256,
    UPSTREAM_BYTES,
    UPSTREAM_SHA256,
    verify_ppw_reference,
)
from prpg.model.refit_adequacy_execution import (
    CANONICAL_DESIGN_LENGTHS,
    CANONICAL_MINIMUM_SUCCESSES,
    CANONICAL_PRODUCTION_LENGTHS,
    CANONICAL_REFIT_ATTEMPTS,
)
from prpg.model.regime_policy import (
    OWNER_FIXED_K4_DISPOSITION_CONTRACT,
    OWNER_FIXED_K4_RECURRING_SUPPORT_CONTRACT,
)
from prpg.model.run_store import CalibrationRunStore, TaskReceipt
from prpg.model.scientific_artifact import CANONICAL_SCIENTIFIC_VERSION
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
    G3ExecutionClosureIdentity,
    inspect_dependency_lock_versions,
    inspect_g3_execution_closure,
)
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    ModelFitScope,
    Stage,
)

CANONICAL_CONFIG_FINGERPRINT: Final = (
    "572e0a843a540b0a6e307cfb72c3c9c43d2fb6cafc8c2c52f6691f408c38ae58"
)
CANONICAL_PROCESSED_DATA_FINGERPRINT: Final = (
    "84a9604691ae43f6e5c9325ac4001e61477931922816b50516f141d91b017fb5"
)
CANONICAL_CALIBRATION_INPUT_FINGERPRINT: Final = (
    "17d61d99a5b2618abeab5e0898ffe34d6bc1ed35a309ea9be3ad7455b45fd38f"
)
CANONICAL_RAW_SNAPSHOT_FINGERPRINT: Final = (
    "7334739d68ddad659df6f7a5acb2c7b96cfe181bf1d839db5b25f49ddac485d5"
)

CANONICAL_PYTHON_VERSION: Final = "3.11.15"
CANONICAL_PLATFORM: Final = "Darwin"
CANONICAL_MACHINE: Final = "arm64"
CANONICAL_LOGICAL_CPUS: Final = 10
CANONICAL_PHYSICAL_MEMORY_BYTES: Final = 17_179_869_184
CANONICAL_WORKERS: Final = 9
CANONICAL_MASTER_SEED: Final = 20_260_714
CANONICAL_DISK_SAFETY_MULTIPLIER: Final = 3

# Resource estimates are deliberately code-owned.  A canonical caller may not
# lower them to make a preflight pass.  The per-process allowance covers the
# imported NumPy/SciPy/sklearn runtime and one active HMM workspace; registered
# NumPy work arrays are then added separately.  These are engineering bounds,
# not scientific parameters, but their exact bytes are identity-bearing in the
# launch preflight.
G3_RESOURCE_ESTIMATE_VERSION: Final = "g3-resource-estimate-v1"
_MIB: Final = 1 << 20
_GIB: Final = 1 << 30
_PYTHON_PROCESS_ALLOWANCE_BYTES: Final = 512 * _MIB
_PARENT_SERIAL_WORKSPACE_BYTES: Final = 1 * _GIB
_NONKERNEL_STAGE_WORKSPACE_BYTES: Final = 1 * _GIB
_RESOURCE_CONTINGENCY_BYTES: Final = 1 * _GIB
_G3_DURABLE_OUTPUT_ESTIMATE_BYTES: Final = 4 * _GIB
G3_PREFLIGHT_RECEIPT_STAGE: Final = "g3_preflight"
G3_PREFLIGHT_RECEIPT_TASK: Final = "canonical"
_G3_PREFLIGHT_CAPABILITY = object()
_CANONICAL_PREFLIGHT_CHECKS: Final = (
    "canonical_config_and_constants",
    "canonical_fingerprints",
    "exact_geometry_columns_masks_and_holdout_seal",
    "closed_rng_registry",
    "section_4_5_4_6_module_constants",
    "vendored_ppw_hash_and_corrected_branches",
    "pinned_target_mac_environment",
    "g3_execution_closure_and_dependency_lock",
    "code_owned_nine_worker_memory_disk_and_descriptor_limits",
)
_G3_PREFLIGHT_SCHEMA_VERSION: Final = 2
_G3_TOOLCHAIN_SCHEMA: Final = "prpg-g3-toolchain-v3"

# Compatibility/readability view of the authoritative complete lock inventory.
# The strict parser rejects malformed, duplicate, and non-exact entries before
# any canonical preflight can be built.
EXPECTED_RUNTIME_VERSIONS: Final = MappingProxyType(
    dict(inspect_dependency_lock_versions())
)

_EXPECTED_GEOMETRY: Final = {
    "holdout_months": 24,
    "full_monthly_rows": 217,
    "full_daily_rows": 4_547,
    "design_monthly_rows": 193,
    "design_daily_rows": 4_049,
    "holdout_monthly_rows": 24,
    "holdout_daily_rows": 498,
    "full_first_month_ordinal": 459,
    "full_last_month_ordinal": 676,
    "design_last_month_ordinal": 651,
    "holdout_first_month_ordinal": 652,
    "full_first_daily_session_ns": 1_207_008_000_000_000_000,
    "full_last_daily_session_ns": 1_780_012_800_000_000_000,
    "design_last_daily_session_ns": 1_714_435_200_000_000_000,
    "holdout_first_daily_session_ns": 1_714_521_600_000_000_000,
}

_EXPECTED_SEGMENTS: Final = {
    "full_monthly": (210, 7),
    "design_monthly": (193,),
    "holdout_monthly": (17, 7),
    "full_daily": (4_404, 143),
    "design_daily": (4_049,),
    "holdout_daily": (355, 143),
}

_EXPECTED_SOURCE_IDENTITIES: Final = (
    ("asset:equity", "yfinance", "ACWI"),
    ("asset:muni_bond", "yfinance", "MUB"),
    ("asset:taxable_bond", "yfinance", "AGG"),
    ("macro:industrial_production", "fred", "INDPRO"),
    ("macro:inflation", "fred", "CPIAUCSL"),
    ("macro:yield_curve_spread", "fred", "T10Y3M"),
    ("macro:credit_spread", "fred", "BAA10Y"),
)


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Injectable exact environment inventory for deterministic testing."""

    python_version: str
    python_implementation: str
    platform_system: str
    machine: str
    dependency_versions: tuple[tuple[str, str | None], ...]
    file_descriptor_soft_limit: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform_system": self.platform_system,
            "machine": self.machine,
            "dependency_versions": dict(self.dependency_versions),
            "file_descriptor_soft_limit": self.file_descriptor_soft_limit,
        }


@dataclass(frozen=True, slots=True)
class CanonicalG3ResourceEstimate:
    """Code-owned conservative RAM and durable-output estimate for G3."""

    estimate_version: str
    workers: int
    process_count: int
    parent_serial_workspace_bytes: int
    python_process_allowance_bytes: int
    largest_registered_kernel_arrays_bytes: int
    nonkernel_stage_workspace_bytes: int
    contingency_bytes: int
    estimated_peak_bytes: int
    estimated_disk_bytes: int
    fingerprint: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "estimate_version": self.estimate_version,
            "workers": self.workers,
            "process_count": self.process_count,
            "components": {
                "parent_serial_workspace_bytes": (self.parent_serial_workspace_bytes),
                "python_process_allowance_bytes": (self.python_process_allowance_bytes),
                "largest_registered_kernel_arrays_bytes": (
                    self.largest_registered_kernel_arrays_bytes
                ),
                "nonkernel_stage_workspace_bytes": (
                    self.nonkernel_stage_workspace_bytes
                ),
                "contingency_bytes": self.contingency_bytes,
            },
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "estimated_disk_bytes": self.estimated_disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class CanonicalG3Preflight:
    """Content-hashed evidence that every launch prerequisite passed."""

    config_fingerprint: str
    processed_data_fingerprint: str
    calibration_input_fingerprint: str
    contract_fingerprint: str
    source_code_fingerprint: str
    dependency_lock_fingerprint: str
    toolchain_fingerprint: str
    ppw_vendored_fingerprint: str
    ppw_upstream_fingerprint: str
    task_registry_fingerprint: str
    planned_look_registry_fingerprint: str
    logical_cpu_count: int
    physical_memory_bytes: int
    disk_free_bytes: int
    workers: int
    estimated_peak_bytes: int
    estimated_disk_bytes: int
    resource_estimate_fingerprint: str
    checks: tuple[str, ...]
    fingerprint: str
    _authority_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _G3_PREFLIGHT_SCHEMA_VERSION,
            "contract_version": G3_CONTRACT_VERSION,
            "execution_closure_schema": G3_EXECUTION_CLOSURE_SCHEMA,
            "execution_closure_manifest": G3_EXECUTION_CLOSURE_MANIFEST.as_dict(),
            "execution_closure_manifest_fingerprint": (
                G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
            ),
            "config_fingerprint": self.config_fingerprint,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "calibration_input_fingerprint": self.calibration_input_fingerprint,
            "contract_fingerprint": self.contract_fingerprint,
            "source_code_fingerprint": self.source_code_fingerprint,
            "dependency_lock_fingerprint": self.dependency_lock_fingerprint,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "ppw_vendored_fingerprint": self.ppw_vendored_fingerprint,
            "ppw_upstream_fingerprint": self.ppw_upstream_fingerprint,
            "task_registry_fingerprint": self.task_registry_fingerprint,
            "planned_look_registry_fingerprint": (
                self.planned_look_registry_fingerprint
            ),
            "resources": {
                "logical_cpu_count": self.logical_cpu_count,
                "physical_memory_bytes": self.physical_memory_bytes,
                "disk_free_bytes": self.disk_free_bytes,
                "workers": self.workers,
                "estimated_peak_bytes": self.estimated_peak_bytes,
                "estimated_disk_bytes": self.estimated_disk_bytes,
                "resource_estimate_fingerprint": (self.resource_estimate_fingerprint),
            },
            "checks": list(self.checks),
        }


@dataclass(frozen=True, slots=True)
class PersistedCanonicalG3Preflight:
    """Verified durable preflight evidence; this is report-only on reload."""

    run_id: str
    preflight: CanonicalG3Preflight
    receipt_fingerprint: str
    path: Path


@dataclass(frozen=True, slots=True)
class CanonicalBridgeResourcePreflight:
    """Hash-bound authorization for the later bridge task, not G3 as a whole."""

    base_preflight_fingerprint: str
    benchmark_pairs: int
    workers: int
    projected_hours: float
    peak_memory_fraction: float
    maximum_hours: float
    maximum_memory_fraction: float
    fingerprint: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "contract_version": G3_CONTRACT_VERSION,
            "base_preflight_fingerprint": self.base_preflight_fingerprint,
            "benchmark_pairs": self.benchmark_pairs,
            "workers": self.workers,
            "projected_hours": self.projected_hours,
            "peak_memory_fraction": self.peak_memory_fraction,
            "maximum_hours": self.maximum_hours,
            "maximum_memory_fraction": self.maximum_memory_fraction,
        }


def inspect_runtime_environment(
    expected_versions: tuple[tuple[str, str], ...] | None = None,
) -> RuntimeEnvironment:
    """Inspect the pinned packages and target platform without writing files."""

    locked = (
        inspect_g3_execution_closure().dependency_versions
        if expected_versions is None
        else expected_versions
    )
    dependencies: list[tuple[str, str | None]] = []
    for distribution, _ in locked:
        try:
            installed: str | None = version(distribution)
        except PackageNotFoundError:
            installed = None
        dependencies.append((distribution, installed))
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = (1 << 63) - 1
    return RuntimeEnvironment(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        machine=platform.machine(),
        dependency_versions=tuple(dependencies),
        file_descriptor_soft_limit=int(soft_limit),
    )


def canonical_g3_resource_estimate(
    *, workers: int = CANONICAL_WORKERS
) -> CanonicalG3ResourceEstimate:
    """Derive the only accepted G3 estimate without materializing RNG streams.

    The registered kernel estimator is evaluated for all four exact
    artifact/frequency cells and its largest all-worker allocation is used.
    The other components are deliberately additive, even though stage working
    sets do not coexist, to preserve a conservative bound on the target Mac.
    """

    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ModelError("G3 resource-estimate workers must be an integer")
    if workers != CANONICAL_WORKERS:
        raise ModelError("canonical G3 resource estimate requires nine workers")
    artifact_roles: tuple[Artifact, ...] = ("design", "production")
    frequencies: tuple[BaseFrequency, ...] = ("monthly", "daily")
    kernel_estimates = tuple(
        registered_kernel_memory_estimate(
            artifact=artifact,
            frequency=frequency,
            workers=workers,
        ).all_workers_bytes
        for artifact in artifact_roles
        for frequency in frequencies
    )
    largest_kernel = max(kernel_estimates)
    process_count = workers + 1
    process_allowance = process_count * _PYTHON_PROCESS_ALLOWANCE_BYTES
    peak = (
        _PARENT_SERIAL_WORKSPACE_BYTES
        + process_allowance
        + largest_kernel
        + _NONKERNEL_STAGE_WORKSPACE_BYTES
        + _RESOURCE_CONTINGENCY_BYTES
    )
    provisional = CanonicalG3ResourceEstimate(
        G3_RESOURCE_ESTIMATE_VERSION,
        workers,
        process_count,
        _PARENT_SERIAL_WORKSPACE_BYTES,
        process_allowance,
        largest_kernel,
        _NONKERNEL_STAGE_WORKSPACE_BYTES,
        _RESOURCE_CONTINGENCY_BYTES,
        peak,
        _G3_DURABLE_OUTPUT_ESTIMATE_BYTES,
        "",
    )
    fingerprint = _sha256(_canonical_bytes(provisional.identity_dict()))
    return CanonicalG3ResourceEstimate(
        provisional.estimate_version,
        provisional.workers,
        provisional.process_count,
        provisional.parent_serial_workspace_bytes,
        provisional.python_process_allowance_bytes,
        provisional.largest_registered_kernel_arrays_bytes,
        provisional.nonkernel_stage_workspace_bytes,
        provisional.contingency_bytes,
        provisional.estimated_peak_bytes,
        provisional.estimated_disk_bytes,
        fingerprint,
    )


def canonical_g3_preflight(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    resources: ResourceSnapshot,
    *,
    runtime: RuntimeEnvironment | None = None,
) -> CanonicalG3Preflight:
    """Verify every immutable prerequisite before a canonical launch is planned."""

    if not isinstance(config, PRPGConfig):
        raise ModelError("canonical G3 preflight requires a resolved PRPGConfig")
    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("canonical G3 preflight requires a CalibrationInput")
    if not isinstance(resources, ResourceSnapshot):
        raise ModelError("canonical G3 preflight requires a resource snapshot")
    checks: list[str] = []
    _verify_config(config)
    checks.append("canonical_config_and_constants")
    _verify_fingerprints(config, calibration_input)
    checks.append("canonical_fingerprints")
    _verify_geometry_and_seal(calibration_input)
    checks.append("exact_geometry_columns_masks_and_holdout_seal")
    _verify_rng_registry()
    checks.append("closed_rng_registry")
    _verify_module_constants()
    checks.append("section_4_5_4_6_module_constants")
    ppw = verify_ppw_reference()
    checks.append("vendored_ppw_hash_and_corrected_branches")
    source = inspect_g3_execution_closure()
    environment = runtime or inspect_runtime_environment(source.dependency_versions)
    runtime_fingerprint = _verify_environment(environment, source.dependency_versions)
    checks.append("pinned_target_mac_environment")
    toolchain_fingerprint = _toolchain_fingerprint(runtime_fingerprint, source)
    checks.append("g3_execution_closure_and_dependency_lock")
    resource_estimate = canonical_g3_resource_estimate(
        workers=resources.resolved_workers
    )
    _verify_resources(
        config,
        resources,
        estimated_peak_bytes=resource_estimate.estimated_peak_bytes,
        estimated_disk_bytes=resource_estimate.estimated_disk_bytes,
        file_descriptor_soft_limit=environment.file_descriptor_soft_limit,
    )
    checks.append("code_owned_nine_worker_memory_disk_and_descriptor_limits")

    contract_fingerprint = _sha256(_canonical_bytes(_contract_identity()))
    provisional = CanonicalG3Preflight(
        config_fingerprint=CANONICAL_CONFIG_FINGERPRINT,
        processed_data_fingerprint=CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_input_fingerprint=CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
        contract_fingerprint=contract_fingerprint,
        source_code_fingerprint=source.source_code_fingerprint,
        dependency_lock_fingerprint=source.dependency_lock_fingerprint,
        toolchain_fingerprint=toolchain_fingerprint,
        ppw_vendored_fingerprint=ppw.vendored_sha256,
        ppw_upstream_fingerprint=ppw.upstream_sha256,
        task_registry_fingerprint=G3_TASK_REGISTRY_FINGERPRINT,
        planned_look_registry_fingerprint=G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
        logical_cpu_count=resources.logical_cpu_count,
        physical_memory_bytes=resources.physical_memory_bytes,
        disk_free_bytes=resources.disk_free_bytes,
        workers=resources.resolved_workers,
        estimated_peak_bytes=resource_estimate.estimated_peak_bytes,
        estimated_disk_bytes=resource_estimate.estimated_disk_bytes,
        resource_estimate_fingerprint=resource_estimate.fingerprint,
        checks=tuple(checks),
        fingerprint="",
    )
    fingerprint = _sha256(_canonical_bytes(provisional.identity_dict()))
    result = CanonicalG3Preflight(
        config_fingerprint=provisional.config_fingerprint,
        processed_data_fingerprint=provisional.processed_data_fingerprint,
        calibration_input_fingerprint=provisional.calibration_input_fingerprint,
        contract_fingerprint=provisional.contract_fingerprint,
        source_code_fingerprint=provisional.source_code_fingerprint,
        dependency_lock_fingerprint=provisional.dependency_lock_fingerprint,
        toolchain_fingerprint=provisional.toolchain_fingerprint,
        ppw_vendored_fingerprint=provisional.ppw_vendored_fingerprint,
        ppw_upstream_fingerprint=provisional.ppw_upstream_fingerprint,
        task_registry_fingerprint=provisional.task_registry_fingerprint,
        planned_look_registry_fingerprint=(
            provisional.planned_look_registry_fingerprint
        ),
        logical_cpu_count=provisional.logical_cpu_count,
        physical_memory_bytes=provisional.physical_memory_bytes,
        disk_free_bytes=provisional.disk_free_bytes,
        workers=provisional.workers,
        estimated_peak_bytes=provisional.estimated_peak_bytes,
        estimated_disk_bytes=provisional.estimated_disk_bytes,
        resource_estimate_fingerprint=provisional.resource_estimate_fingerprint,
        checks=provisional.checks,
        fingerprint=fingerprint,
    )
    object.__setattr__(result, "_authority_seal", _G3_PREFLIGHT_CAPABILITY)
    return result


def verify_live_canonical_g3_preflight(value: CanonicalG3Preflight) -> None:
    """Verify exact evidence and require provenance from the code-owned builder."""

    _verify_preflight_value(value)
    if value._authority_seal is not _G3_PREFLIGHT_CAPABILITY:
        raise ModelError("canonical G3 preflight lacks live builder authority")


def verify_compatible_canonical_g3_preflights(
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
) -> str:
    """Bind durable launch identity to a fresh capacity/source inspection.

    Free disk is an operational observation and may legitimately differ on
    resume.  Every scientific, code, lock, toolchain, and resource-contract
    field must remain identical.  Only the fresh value must retain the
    process-local builder seal; a launch value reloaded from its verified
    durable receipt is intentionally report-only.
    """

    _verify_preflight_value(launch_preflight)
    verify_live_canonical_g3_preflight(live_preflight)
    if _stable_preflight_identity(launch_preflight) != _stable_preflight_identity(
        live_preflight
    ):
        raise IntegrityError("launch and live canonical G3 preflights disagree")
    return launch_preflight.fingerprint


def persist_canonical_g3_preflight(
    store: CalibrationRunStore,
    *,
    run_id: str,
    preflight: CanonicalG3Preflight,
) -> TaskReceipt:
    """Durably bind one live preflight to its exact scientific run identity."""

    if not isinstance(store, CalibrationRunStore):
        raise ModelError("preflight persistence requires a CalibrationRunStore")
    verify_live_canonical_g3_preflight(preflight)
    _verify_preflight_run_binding(store, run_id, preflight)
    return store.write_receipt(
        run_id,
        stage=G3_PREFLIGHT_RECEIPT_STAGE,
        task_id=G3_PREFLIGHT_RECEIPT_TASK,
        payload={
            "preflight_fingerprint": preflight.fingerprint,
            "preflight_identity": preflight.identity_dict(),
        },
    )


def read_persisted_canonical_g3_preflight(
    store: CalibrationRunStore,
    *,
    run_id: str,
) -> PersistedCanonicalG3Preflight:
    """Read and fully verify durable preflight evidence without restoring authority."""

    if not isinstance(store, CalibrationRunStore):
        raise ModelError("preflight lookup requires a CalibrationRunStore")
    receipt = store.read_receipt(
        run_id,
        stage=G3_PREFLIGHT_RECEIPT_STAGE,
        task_id=G3_PREFLIGHT_RECEIPT_TASK,
    )
    if set(receipt.payload) != {"preflight_fingerprint", "preflight_identity"}:
        raise IntegrityError("persisted G3 preflight payload fields are invalid")
    fingerprint = receipt.payload["preflight_fingerprint"]
    identity = receipt.payload["preflight_identity"]
    if not isinstance(fingerprint, str) or not isinstance(identity, Mapping):
        raise IntegrityError("persisted G3 preflight payload is invalid")
    preflight = _preflight_from_identity(identity, fingerprint)
    _verify_preflight_value(preflight, error_type=IntegrityError)
    _verify_preflight_run_binding(
        store,
        run_id,
        preflight,
        error_type=IntegrityError,
    )
    return PersistedCanonicalG3Preflight(
        run_id,
        preflight,
        receipt.fingerprint,
        receipt.path,
    )


def canonical_resume_preflight_fingerprint(
    store: CalibrationRunStore,
    *,
    run_id: str,
    live_preflight: CanonicalG3Preflight,
) -> str:
    """Revalidate current capacity and recover the original launch fingerprint.

    Exact free-disk bytes are an operational observation and can change after
    a crash.  Every other preflight field must match.  The newly built live
    preflight proves current capacity; the persisted fingerprint preserves the
    original launch/checkpoint identity.
    """

    _verify_preflight_run_binding(store, run_id, live_preflight)
    persisted = read_persisted_canonical_g3_preflight(store, run_id=run_id)
    verify_compatible_canonical_g3_preflights(persisted.preflight, live_preflight)
    marker = store.read_launch(run_id)
    if marker.preflight_fingerprint != persisted.preflight.fingerprint:
        raise IntegrityError("launch marker does not match persisted G3 preflight")
    return persisted.preflight.fingerprint


def _verify_preflight_value(
    value: CanonicalG3Preflight,
    *,
    error_type: type[ModelError] | type[IntegrityError] = ModelError,
) -> None:
    if not isinstance(value, CanonicalG3Preflight):
        raise error_type("canonical G3 preflight has the wrong type")
    estimate = canonical_g3_resource_estimate()
    source = inspect_g3_execution_closure()
    expected_runtime = _sha256(
        _canonical_bytes(
            {
                "python_version": CANONICAL_PYTHON_VERSION,
                "python_implementation": "CPython",
                "platform_system": CANONICAL_PLATFORM,
                "machine": CANONICAL_MACHINE,
                "dependency_versions": dict(source.dependency_versions),
            }
        )
    )
    expected_toolchain = _toolchain_fingerprint(expected_runtime, source)
    expected_contract = _sha256(_canonical_bytes(_contract_identity()))
    if (
        value.config_fingerprint != CANONICAL_CONFIG_FINGERPRINT
        or value.processed_data_fingerprint != CANONICAL_PROCESSED_DATA_FINGERPRINT
        or value.calibration_input_fingerprint
        != CANONICAL_CALIBRATION_INPUT_FINGERPRINT
        or value.contract_fingerprint != expected_contract
        or value.source_code_fingerprint != source.source_code_fingerprint
        or value.dependency_lock_fingerprint != source.dependency_lock_fingerprint
        or value.toolchain_fingerprint != expected_toolchain
        or value.ppw_vendored_fingerprint != REFERENCE_SHA256
        or value.ppw_upstream_fingerprint != UPSTREAM_SHA256
        or value.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT
        or value.planned_look_registry_fingerprint
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
        or value.logical_cpu_count != CANONICAL_LOGICAL_CPUS
        or value.physical_memory_bytes != CANONICAL_PHYSICAL_MEMORY_BYTES
        or value.workers != CANONICAL_WORKERS
        or value.estimated_peak_bytes != estimate.estimated_peak_bytes
        or value.estimated_disk_bytes != estimate.estimated_disk_bytes
        or value.resource_estimate_fingerprint != estimate.fingerprint
        or value.checks != _CANONICAL_PREFLIGHT_CHECKS
    ):
        raise error_type("canonical G3 preflight contract fields are inconsistent")
    if (
        isinstance(value.disk_free_bytes, bool)
        or not isinstance(value.disk_free_bytes, int)
        or value.disk_free_bytes
        < CANONICAL_DISK_SAFETY_MULTIPLIER * estimate.estimated_disk_bytes
    ):
        raise error_type("canonical G3 preflight disk evidence is insufficient")
    if (
        memory_preflight(
            value.estimated_peak_bytes,
            value.physical_memory_bytes,
            maximum_fraction=0.70,
        ).passed
        is not True
    ):
        raise error_type("canonical G3 preflight memory evidence is insufficient")
    computed = _sha256(_canonical_bytes(value.identity_dict()))
    if value.fingerprint != computed:
        raise error_type("canonical G3 preflight evidence hash is inconsistent")


def _preflight_from_identity(
    value: Mapping[str, Any], fingerprint: str
) -> CanonicalG3Preflight:
    expected = {
        "schema_version",
        "contract_version",
        "execution_closure_schema",
        "execution_closure_manifest",
        "execution_closure_manifest_fingerprint",
        "config_fingerprint",
        "processed_data_fingerprint",
        "calibration_input_fingerprint",
        "contract_fingerprint",
        "source_code_fingerprint",
        "dependency_lock_fingerprint",
        "toolchain_fingerprint",
        "ppw_vendored_fingerprint",
        "ppw_upstream_fingerprint",
        "task_registry_fingerprint",
        "planned_look_registry_fingerprint",
        "resources",
        "checks",
    }
    if set(value) != expected:
        raise IntegrityError("persisted G3 preflight identity fields are invalid")
    if (
        value["schema_version"] != _G3_PREFLIGHT_SCHEMA_VERSION
        or value["contract_version"] != G3_CONTRACT_VERSION
        or value["execution_closure_schema"] != G3_EXECUTION_CLOSURE_SCHEMA
        or value["execution_closure_manifest"]
        != G3_EXECUTION_CLOSURE_MANIFEST.as_dict()
        or value["execution_closure_manifest_fingerprint"]
        != G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
    ):
        raise IntegrityError("persisted G3 preflight identity version is invalid")
    resources = value["resources"]
    resource_keys = {
        "logical_cpu_count",
        "physical_memory_bytes",
        "disk_free_bytes",
        "workers",
        "estimated_peak_bytes",
        "estimated_disk_bytes",
        "resource_estimate_fingerprint",
    }
    if not isinstance(resources, Mapping) or set(resources) != resource_keys:
        raise IntegrityError("persisted G3 preflight resources are invalid")
    integer_resource_keys = resource_keys - {"resource_estimate_fingerprint"}
    if any(type(resources[key]) is not int for key in integer_resource_keys):
        raise IntegrityError("persisted G3 preflight resource types are invalid")
    checks = value["checks"]
    if not isinstance(checks, list) or any(
        not isinstance(item, str) for item in checks
    ):
        raise IntegrityError("persisted G3 preflight checks are invalid")
    fingerprint_fields = (
        "config_fingerprint",
        "processed_data_fingerprint",
        "calibration_input_fingerprint",
        "contract_fingerprint",
        "source_code_fingerprint",
        "dependency_lock_fingerprint",
        "toolchain_fingerprint",
        "ppw_vendored_fingerprint",
        "ppw_upstream_fingerprint",
        "task_registry_fingerprint",
        "planned_look_registry_fingerprint",
    )
    if any(
        not isinstance(value[key], str) or len(value[key]) != 64
        for key in fingerprint_fields
    ) or not isinstance(resources["resource_estimate_fingerprint"], str):
        raise IntegrityError("persisted G3 preflight fingerprints are invalid")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise IntegrityError("persisted G3 preflight fingerprint is invalid")
    return CanonicalG3Preflight(
        config_fingerprint=value["config_fingerprint"],
        processed_data_fingerprint=value["processed_data_fingerprint"],
        calibration_input_fingerprint=value["calibration_input_fingerprint"],
        contract_fingerprint=value["contract_fingerprint"],
        source_code_fingerprint=value["source_code_fingerprint"],
        dependency_lock_fingerprint=value["dependency_lock_fingerprint"],
        toolchain_fingerprint=value["toolchain_fingerprint"],
        ppw_vendored_fingerprint=value["ppw_vendored_fingerprint"],
        ppw_upstream_fingerprint=value["ppw_upstream_fingerprint"],
        task_registry_fingerprint=value["task_registry_fingerprint"],
        planned_look_registry_fingerprint=(value["planned_look_registry_fingerprint"]),
        logical_cpu_count=resources["logical_cpu_count"],
        physical_memory_bytes=resources["physical_memory_bytes"],
        disk_free_bytes=resources["disk_free_bytes"],
        workers=resources["workers"],
        estimated_peak_bytes=resources["estimated_peak_bytes"],
        estimated_disk_bytes=resources["estimated_disk_bytes"],
        resource_estimate_fingerprint=resources["resource_estimate_fingerprint"],
        checks=tuple(checks),
        fingerprint=fingerprint,
    )


def _verify_preflight_run_binding(
    store: CalibrationRunStore,
    run_id: str,
    preflight: CanonicalG3Preflight,
    *,
    error_type: type[ModelError] | type[IntegrityError] = ModelError,
) -> None:
    reference = store.verify(run_id)
    if (
        preflight.config_fingerprint != reference.identity.config_fingerprint
        or preflight.processed_data_fingerprint
        != reference.identity.processed_data_fingerprint
        or preflight.calibration_input_fingerprint
        != reference.identity.calibration_input_fingerprint
        or preflight.source_code_fingerprint
        != reference.identity.source_code_fingerprint
        or preflight.dependency_lock_fingerprint
        != reference.identity.dependency_lock_fingerprint
        or reference.identity.contract_version != G3_CONTRACT_VERSION
        or reference.identity.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT
        or reference.identity.planned_look_registry_fingerprint
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise error_type("canonical G3 preflight/run identity bindings disagree")


def _stable_preflight_identity(value: CanonicalG3Preflight) -> dict[str, Any]:
    identity = value.identity_dict()
    resources = dict(identity["resources"])
    resources.pop("disk_free_bytes")
    identity["resources"] = resources
    return identity


def canonical_bridge_resource_preflight(
    base: CanonicalG3Preflight,
    bridge: BridgeResourcePreflight,
) -> CanonicalBridgeResourcePreflight:
    """Verify the binding 100-pair, nine-worker, 72-hour bridge projection."""

    if not isinstance(base, CanonicalG3Preflight):
        raise ModelError("bridge preflight requires canonical base-preflight evidence")
    try:
        verify_live_canonical_g3_preflight(base)
    except ModelError as error:
        raise ModelError(
            "canonical base-preflight evidence hash is inconsistent or unauthorized"
        ) from error
    if not isinstance(bridge, BridgeResourcePreflight):
        raise ModelError("bridge preflight requires BridgeResourcePreflight evidence")
    if (
        bridge.benchmark_pairs != 100
        or bridge.workers != CANONICAL_WORKERS
        or bridge.maximum_hours != 72.0
        or bridge.maximum_memory_fraction != 0.70
    ):
        raise ModelError("bridge resource evidence uses an unapproved limit")
    if (
        not bridge.passed
        or bridge.reasons
        or bridge.projected_worst_case_hours > bridge.maximum_hours
        or bridge.peak_memory_fraction > bridge.maximum_memory_fraction
    ):
        raise ModelError("bridge resource projection failed closed")
    provisional = CanonicalBridgeResourcePreflight(
        base.fingerprint,
        bridge.benchmark_pairs,
        bridge.workers,
        bridge.projected_worst_case_hours,
        bridge.peak_memory_fraction,
        bridge.maximum_hours,
        bridge.maximum_memory_fraction,
        "",
    )
    fingerprint = _sha256(_canonical_bytes(provisional.identity_dict()))
    return CanonicalBridgeResourcePreflight(
        provisional.base_preflight_fingerprint,
        provisional.benchmark_pairs,
        provisional.workers,
        provisional.projected_hours,
        provisional.peak_memory_fraction,
        provisional.maximum_hours,
        provisional.maximum_memory_fraction,
        fingerprint,
    )


def _verify_config(config: PRPGConfig) -> None:
    if config.fingerprint() != CANONICAL_CONFIG_FINGERPRINT:
        raise ModelError("resolved config is not the canonical approved config")
    if (
        config.validation.profile != "release"
        or config.run.master_seed != CANONICAL_MASTER_SEED
        or config.simulation.scientific_version != CANONICAL_SCIENTIFIC_VERSION
        or config.model.regime_candidates != (2, 3, 4, 5)
        or config.model.regime_selection_policy != "owner_fixed_k4_v3"
        or config.model.fixed_regime_count != 4
        or config.model.covariance_type != "tied"
        or config.model.deterministic_restarts != 50
        or config.model.design_holdout_months != 24
        or config.model.production_refit != "full_data_mandatory"
        or config.model.alpha_policy_candidates
        != ("historical_vector", "kernel_mean_neutral")
        or config.model.selected_alpha_policy is not None
        or config.model.selected_neutral_lambda is not None
        or config.model.neutral_lambda_grid != (0.25, 0.50, 0.75, 1.00)
        or config.model.initial_state != "stationary"
        or config.execution.workers != "auto"
        or config.execution.reserve_physical_cores != 1
        or config.execution.memory_fraction_limit != 0.70
        or config.validation.familywise_alpha != 0.05
        or config.validation.kernel_mean_calibration.prefix_window_counts
        != (10_000, 20_000, 40_000, 80_000, 160_000, 320_000)
    ):
        raise ModelError("canonical config contains an unapproved G3 constant")


def _verify_fingerprints(config: PRPGConfig, value: CalibrationInput) -> None:
    if (
        value.provenance.processed_data_fingerprint
        != CANONICAL_PROCESSED_DATA_FINGERPRINT
        or value.provenance.raw_snapshot_fingerprint
        != CANONICAL_RAW_SNAPSHOT_FINGERPRINT
        or value.provenance.calibration_config_fingerprint != config.fingerprint()
        or value.fingerprint != CANONICAL_CALIBRATION_INPUT_FINGERPRINT
    ):
        raise ModelError("calibration input fingerprint lineage is not canonical")
    computed = _sha256(_canonical_bytes(value.identity_dict()))
    if computed != value.fingerprint:
        raise ModelError("calibration input identity hash is inconsistent")


def _verify_geometry_and_seal(value: CalibrationInput) -> None:
    if value.geometry.as_dict() != _EXPECTED_GEOMETRY:
        raise ModelError("calibration input geometry is not the exact G2 geometry")
    provenance = value.provenance
    if (
        provenance.asset_columns != ASSET_COLUMNS
        or provenance.macro_columns != MACRO_COLUMNS
    ):
        raise ModelError("calibration semantic columns are not in canonical order")
    identities = tuple(
        (item.role, item.provider, item.identifier)
        for item in provenance.source_identities
    )
    if identities != _EXPECTED_SOURCE_IDENTITIES:
        raise ModelError("calibration source identities are not canonical")

    expected_shapes = {
        "full": ((217, 3), (4_547, 3), (217, 4)),
        "design": ((193, 3), (4_049, 3), (193, 4)),
        "holdout": ((24, 3), (498, 3), (24, 4)),
    }
    for role in ("full", "design", "holdout"):
        split = getattr(value, role)
        monthly_shape, daily_shape, macro_shape = expected_shapes[role]
        if (
            split.role != role
            or split.monthly_returns.shape != monthly_shape
            or split.daily_returns.shape != daily_shape
            or split.macro_features.shape != macro_shape
            or split.monthly_period_ordinals.shape != (monthly_shape[0],)
            or split.daily_session_ns.shape != (daily_shape[0],)
            or split.monthly_continuation.shape != (monthly_shape[0],)
            or split.daily_continuation.shape != (daily_shape[0],)
        ):
            raise ModelError("calibration split shapes are not canonical")
        _verify_split_arrays(split)

    segments = {
        "full_monthly": _segments(value.full.monthly_continuation),
        "design_monthly": _segments(value.design.monthly_continuation),
        "holdout_monthly": _segments(value.holdout.monthly_continuation),
        "full_daily": _segments(value.full.daily_continuation),
        "design_daily": _segments(value.design.daily_continuation),
        "holdout_daily": _segments(value.holdout.daily_continuation),
    }
    if segments != _EXPECTED_SEGMENTS:
        raise ModelError("calibration continuation-mask segments are not canonical")
    _verify_exact_suffixes(value)
    _verify_detached_splits(value)


def _verify_split_arrays(split: Any) -> None:
    float_arrays = (
        split.monthly_returns,
        split.daily_returns,
        split.macro_features,
    )
    int_arrays = (split.monthly_period_ordinals, split.daily_session_ns)
    masks = (split.monthly_continuation, split.daily_continuation)
    if any(array.dtype != np.dtype(np.float64) for array in float_arrays):
        raise ModelError("calibration numeric arrays must be binary64")
    if any(array.dtype != np.dtype(np.int64) for array in int_arrays):
        raise ModelError("calibration index arrays must be int64")
    if any(array.dtype != np.dtype(np.bool_) for array in masks):
        raise ModelError("calibration continuation masks must be bool")
    if not all(np.isfinite(array).all() for array in float_arrays):
        raise ModelError("calibration numeric arrays contain non-finite values")
    for array in (*float_arrays, *int_arrays, *masks):
        if array.flags.writeable:
            raise ModelError("calibration arrays must be read-only")
    if any(bool(mask[0]) for mask in masks):
        raise ModelError("every calibration segment view must reset at row zero")
    if not np.all(np.diff(split.monthly_period_ordinals) > 0) or not np.all(
        np.diff(split.daily_session_ns) > 0
    ):
        raise ModelError("calibration indices must be strictly increasing")


def _verify_exact_suffixes(value: CalibrationInput) -> None:
    monthly_boundary = value.geometry.design_monthly_rows
    daily_boundary = value.geometry.design_daily_rows
    pairs = (
        (value.full.monthly_returns[:monthly_boundary], value.design.monthly_returns),
        (value.full.monthly_returns[monthly_boundary:], value.holdout.monthly_returns),
        (value.full.macro_features[:monthly_boundary], value.design.macro_features),
        (value.full.macro_features[monthly_boundary:], value.holdout.macro_features),
        (
            value.full.monthly_period_ordinals[:monthly_boundary],
            value.design.monthly_period_ordinals,
        ),
        (
            value.full.monthly_period_ordinals[monthly_boundary:],
            value.holdout.monthly_period_ordinals,
        ),
        (value.full.daily_returns[:daily_boundary], value.design.daily_returns),
        (value.full.daily_returns[daily_boundary:], value.holdout.daily_returns),
        (
            value.full.daily_session_ns[:daily_boundary],
            value.design.daily_session_ns,
        ),
        (
            value.full.daily_session_ns[daily_boundary:],
            value.holdout.daily_session_ns,
        ),
    )
    if any(not np.array_equal(left, right) for left, right in pairs):
        raise ModelError("design/holdout arrays are not exact sealed full-data slices")
    if not np.array_equal(
        value.full.monthly_continuation[:monthly_boundary],
        value.design.monthly_continuation,
    ) or not np.array_equal(
        value.full.monthly_continuation[monthly_boundary + 1 :],
        value.holdout.monthly_continuation[1:],
    ):
        raise ModelError("monthly sealed-split continuation masks are inconsistent")
    if not np.array_equal(
        value.full.daily_continuation[:daily_boundary],
        value.design.daily_continuation,
    ) or not np.array_equal(
        value.full.daily_continuation[daily_boundary + 1 :],
        value.holdout.daily_continuation[1:],
    ):
        raise ModelError("daily sealed-split continuation masks are inconsistent")


def _verify_detached_splits(value: CalibrationInput) -> None:
    for name in (
        "monthly_returns",
        "daily_returns",
        "macro_features",
        "monthly_period_ordinals",
        "daily_session_ns",
        "monthly_continuation",
        "daily_continuation",
    ):
        arrays = (
            getattr(value.full, name),
            getattr(value.design, name),
            getattr(value.holdout, name),
        )
        if any(
            np.shares_memory(arrays[left], arrays[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ModelError("sealed calibration splits must be detached in memory")


def _segments(mask: np.ndarray[Any, Any]) -> tuple[int, ...]:
    starts = np.flatnonzero(~mask)
    if starts.size == 0 or int(starts[0]) != 0:
        raise ModelError("continuation mask has no row-zero segment start")
    ends = np.r_[starts[1:], len(mask)]
    return tuple(int(value) for value in ends - starts)


def _verify_rng_registry() -> None:
    actual = {
        "contract": RNG_CONTRACT_VERSION,
        "domains": {item.name: int(item) for item in Domain},
        "families": {item.name: int(item) for item in Family},
        "stages": {item.name: int(item) for item in Stage},
        "model_fit_scopes": {item.name: int(item) for item in ModelFitScope},
        "calibration_purposes": {item.name: int(item) for item in CalibrationPurpose},
        "calibration_artifacts": {item.name: int(item) for item in CalibrationArtifact},
    }
    if actual != _contract_identity()["rng"]:
        raise ModelError("RNG enum allocation differs from the approved registry")


def _verify_module_constants() -> None:
    if _module_constants() != _contract_identity()["scientific_constants"]:
        raise ModelError("implemented scientific constants differ from preregistration")


def _verify_environment(
    value: RuntimeEnvironment,
    expected_dependencies: tuple[tuple[str, str], ...] | None = None,
) -> str:
    expected = (
        inspect_g3_execution_closure().dependency_versions
        if expected_dependencies is None
        else expected_dependencies
    )
    if (
        value.python_version != CANONICAL_PYTHON_VERSION
        or value.python_implementation != "CPython"
        or value.platform_system != CANONICAL_PLATFORM
        or value.machine != CANONICAL_MACHINE
        or value.dependency_versions != expected
    ):
        raise ModelError("runtime is not the pinned target-Mac environment")
    identity = value.as_dict()
    identity.pop("file_descriptor_soft_limit")
    return _sha256(_canonical_bytes(identity))


def _toolchain_fingerprint(
    runtime_fingerprint: str,
    source: G3ExecutionClosureIdentity,
) -> str:
    """Bind the pinned runtime, executable source, and dependency lock."""

    return _sha256(
        _canonical_bytes(
            {
                "schema_id": _G3_TOOLCHAIN_SCHEMA,
                "runtime_fingerprint": runtime_fingerprint,
                "execution_closure_schema": source.schema_id,
                "execution_closure_manifest": source.manifest.as_dict(),
                "execution_closure_manifest_fingerprint": (
                    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
                ),
                "source_code_fingerprint": source.source_code_fingerprint,
                "dependency_lock_fingerprint": source.dependency_lock_fingerprint,
                "locked_dependency_versions": dict(source.dependency_versions),
            }
        )
    )


def _verify_resources(
    config: PRPGConfig,
    value: ResourceSnapshot,
    *,
    estimated_peak_bytes: int,
    estimated_disk_bytes: int,
    file_descriptor_soft_limit: int,
) -> None:
    if (
        value.logical_cpu_count != CANONICAL_LOGICAL_CPUS
        or value.physical_memory_bytes != CANONICAL_PHYSICAL_MEMORY_BYTES
        or value.resolved_workers != CANONICAL_WORKERS
        or value.reserve_cores != 1
    ):
        raise ModelError("resources do not match the approved nine-worker target Mac")
    result = memory_preflight(
        estimated_peak_bytes,
        value.physical_memory_bytes,
        maximum_fraction=config.execution.memory_fraction_limit,
    )
    if not result.passed:
        raise ModelError("estimated G3 peak memory exceeds the 70% limit")
    if (
        isinstance(estimated_disk_bytes, bool)
        or not isinstance(estimated_disk_bytes, int)
        or estimated_disk_bytes <= 0
    ):
        raise ModelError("estimated G3 disk bytes must be a positive integer")
    required_disk = CANONICAL_DISK_SAFETY_MULTIPLIER * estimated_disk_bytes
    if value.disk_free_bytes < required_disk:
        raise ModelError("free disk is below three times estimated G3 output")
    minimum_descriptors = 64 + 4 * value.resolved_workers
    if file_descriptor_soft_limit < minimum_descriptors:
        raise ModelError("file-descriptor limit is insufficient for G3 workers")


def _module_constants() -> dict[str, Any]:
    return {
        "canonical_master_seed": CANONICAL_MASTER_SEED,
        "canonical_scientific_version": CANONICAL_SCIENTIFIC_VERSION,
        "regime_selection_policy": "owner_fixed_k4_v3",
        "owner_fixed_k4_disposition_contract": (OWNER_FIXED_K4_DISPOSITION_CONTRACT),
        "owner_fixed_k4_recurring_support_contract": (
            OWNER_FIXED_K4_RECURRING_SUPPORT_CONTRACT
        ),
        "owner_fixed_n_states": OWNER_FIXED_N_STATES,
        "owner_fixed_minimum_observations": OWNER_FIXED_MINIMUM_OBSERVATIONS,
        "owner_fixed_minimum_occupancy": OWNER_FIXED_MINIMUM_OCCUPANCY,
        "owner_fixed_minimum_observed_episodes": (
            OWNER_FIXED_MINIMUM_OBSERVED_EPISODES
        ),
        "owner_fixed_k4_fixed_point_state_contract": (
            OWNER_FIXED_K4_FIXED_POINT_STATE_CONTRACT
        ),
        "maximum_fixed_point_iterations": MAXIMUM_FIXED_POINT_ITERATIONS,
        "bridge_fixed_k4_tier_a_policy": BRIDGE_FIXED_K4_TIER_A_POLICY,
        "bridge_fixed_k4_tier_a_fits_per_member": (
            BRIDGE_FIXED_K4_TIER_A_FITS_PER_MEMBER
        ),
        "bridge_fixed_k4_tier_a_rolling_folds": (BRIDGE_FIXED_K4_TIER_A_ROLLING_FOLDS),
        "bridge_fixed_k4_tier_a_uses_cross_k_selection": (
            BRIDGE_FIXED_K4_TIER_A_USES_CROSS_K_SELECTION
        ),
        "bridge_fixed_k4_tier_a_uses_bic": BRIDGE_FIXED_K4_TIER_A_USES_BIC,
        "bridge_fixed_k4_tier_a_restarts_per_pair": (
            BRIDGE_FIXED_K4_TIER_A_RESTARTS_PER_PAIR
        ),
        "g3_resource_projection_version": G3_RESOURCE_PROJECTION_VERSION,
        "posterior_effective_mass_binding": False,
        "historical_transition_support_binding": False,
        "block_policy_version": BLOCK_POLICY_VERSION,
        "block_minimum_completed_episodes": MINIMUM_COMPLETED_EPISODES,
        "block_minimum_effective_fraction": MINIMUM_EFFECTIVE_FRACTION,
        "monthly_eligible_start_rule": "max(8,L)",
        "daily_eligible_start_rule": "max(50,2L)",
        "hmm_restarts": HMM_RESTARTS,
        "hmm_eigenvalue_floor": DEFAULT_EIGENVALUE_FLOOR,
        "hmm_condition_limit": DEFAULT_MAX_COVARIANCE_CONDITION,
        "raw_eigenvalue_tolerance": RAW_COVARIANCE_EIGENVALUE_TOLERANCE,
        "floor_relative_change": MAXIMUM_COVARIANCE_FLOOR_RELATIVE_CHANGE,
        "likelihood_decrease_tolerance": LIKELIHOOD_DECREASE_RELATIVE_TOLERANCE,
        "probability_atol": PROBABILITY_ATOL,
        "stationary_residual_atol": STATIONARY_RESIDUAL_ATOL,
        "negative_probability_tolerance": NEGATIVE_PROBABILITY_TOLERANCE,
        "numerical_edge_threshold": NUMERICAL_EDGE_THRESHOLD,
        "maximum_self_transition": MAXIMUM_SELF_TRANSITION,
        "mixing_tv_threshold": MIXING_TOTAL_VARIATION_THRESHOLD,
        "maximum_mixing_months": MAXIMUM_MIXING_MONTHS,
        "maximum_normalized_entropy": MAXIMUM_NORMALIZED_POSTERIOR_ENTROPY,
        "minimum_mean_maximum_posterior": MINIMUM_MEAN_MAXIMUM_POSTERIOR,
        "minimum_posterior_mass": MINIMUM_POSTERIOR_EFFECTIVE_MASS,
        "minimum_posterior_fraction": MINIMUM_POSTERIOR_EFFECTIVE_FRACTION,
        "minimum_assigned_posterior": MINIMUM_ASSIGNED_STATE_POSTERIOR,
        "minimum_mahalanobis": MINIMUM_PAIRWISE_MAHALANOBIS_DISTANCE,
        "latent_chain_count": LATENT_CHAIN_COUNT,
        "latent_measured_months": LATENT_MEASURED_MONTHS,
        "latent_extension_months": LATENT_EXTENSION_MONTHS,
        "latent_bootstrap_replicates": LATENT_BOOTSTRAP_REPLICATES,
        "latent_bootstrap_resample_size": LATENT_BOOTSTRAP_RESAMPLE_SIZE,
        "refit_attempts": CANONICAL_REFIT_ATTEMPTS,
        "refit_minimum_successes": CANONICAL_MINIMUM_SUCCESSES,
        "refit_design_lengths": CANONICAL_DESIGN_LENGTHS,
        "refit_production_lengths": CANONICAL_PRODUCTION_LENGTHS,
        "block_history_count": CANONICAL_HISTORY_COUNT,
        "tier_a_pairs": TIER_A_PAIRS,
        "tier_b_batch_size": TIER_B_BATCH_SIZE,
        "tier_b_maximum_pairs": TIER_B_MAXIMUM_PAIRS,
        "tier_b_tail_probability": TIER_B_TAIL_PROBABILITY,
        "tier_b_interval_error": TIER_B_INTERVAL_ERROR,
        "accelerator_audit_pairs": ACCELERATOR_AUDIT_PAIRS,
        "accelerator_minimum_agreements": ACCELERATOR_MINIMUM_AGREEMENTS,
    }


def _contract_identity() -> dict[str, Any]:
    return {
        "contract_version": G3_CONTRACT_VERSION,
        "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
        "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
        "fingerprints": {
            "g3_execution_closure_manifest": (
                G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
            ),
            "config": CANONICAL_CONFIG_FINGERPRINT,
            "processed_data": CANONICAL_PROCESSED_DATA_FINGERPRINT,
            "calibration_input": CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
            "raw_snapshot": CANONICAL_RAW_SNAPSHOT_FINGERPRINT,
            "ppw_vendored": REFERENCE_SHA256,
            "ppw_vendored_bytes": REFERENCE_BYTES,
            "ppw_upstream": UPSTREAM_SHA256,
            "ppw_upstream_bytes": UPSTREAM_BYTES,
        },
        "scientific_constants": {
            "canonical_master_seed": 20_260_714,
            "canonical_scientific_version": 11,
            "regime_selection_policy": "owner_fixed_k4_v3",
            "owner_fixed_k4_disposition_contract": "owner-fixed-k4-rarity-v1",
            "owner_fixed_k4_recurring_support_contract": (
                "owner-fixed-k4-recurring-history-v1"
            ),
            "owner_fixed_n_states": 4,
            "owner_fixed_minimum_observations": 1,
            "owner_fixed_minimum_occupancy": 0.0,
            "owner_fixed_minimum_observed_episodes": 2,
            "owner_fixed_k4_fixed_point_state_contract": (
                "owner-fixed-k4-lmonthly-ldaily-v1"
            ),
            "maximum_fixed_point_iterations": 5,
            "bridge_fixed_k4_tier_a_policy": "owner_fixed_k4_tier_a_v3",
            "bridge_fixed_k4_tier_a_fits_per_member": 1,
            "bridge_fixed_k4_tier_a_rolling_folds": 0,
            "bridge_fixed_k4_tier_a_uses_cross_k_selection": False,
            "bridge_fixed_k4_tier_a_uses_bic": False,
            "bridge_fixed_k4_tier_a_restarts_per_pair": 100,
            "g3_resource_projection_version": ("g3-rehearsal-resource-projection-v3"),
            "posterior_effective_mass_binding": False,
            "historical_transition_support_binding": False,
            "block_policy_version": "ppw2009-stationary-frequency-wide-v3",
            "block_minimum_completed_episodes": 0,
            "block_minimum_effective_fraction": 0.50,
            "monthly_eligible_start_rule": "max(8,L)",
            "daily_eligible_start_rule": "max(50,2L)",
            "hmm_restarts": 50,
            "hmm_eigenvalue_floor": 1e-6,
            "hmm_condition_limit": 1e6,
            "raw_eigenvalue_tolerance": -1e-10,
            "floor_relative_change": 0.01,
            "likelihood_decrease_tolerance": 1e-8,
            "probability_atol": 1e-12,
            "stationary_residual_atol": 1e-12,
            "negative_probability_tolerance": 1e-15,
            "numerical_edge_threshold": 1e-12,
            "maximum_self_transition": 0.98,
            "mixing_tv_threshold": 0.01,
            "maximum_mixing_months": 120,
            "maximum_normalized_entropy": 0.70,
            "minimum_mean_maximum_posterior": 0.65,
            "minimum_posterior_mass": 24.0,
            "minimum_posterior_fraction": 0.10,
            "minimum_assigned_posterior": 0.60,
            "minimum_mahalanobis": 0.75,
            "latent_chain_count": 10_000,
            "latent_measured_months": 600,
            "latent_extension_months": 2_000,
            "latent_bootstrap_replicates": 10_000,
            "latent_bootstrap_resample_size": 10_000,
            "refit_attempts": 1_000,
            "refit_minimum_successes": 950,
            "refit_design_lengths": (193,),
            "refit_production_lengths": (210, 7),
            "block_history_count": 10_000,
            "tier_a_pairs": 250,
            "tier_b_batch_size": 1_000,
            "tier_b_maximum_pairs": 10_000,
            "tier_b_tail_probability": 0.025,
            "tier_b_interval_error": 0.0025,
            "accelerator_audit_pairs": 100,
            "accelerator_minimum_agreements": 99,
        },
        "rng": {
            "contract": 1,
            "domains": {
                "MODEL_FIT": 1,
                "BLOCK_ALPHA_CALIBRATION": 2,
                "THRESHOLD_CALIBRATION": 3,
                "DEVELOPMENT": 4,
                "QUALIFICATION": 5,
                "PRODUCTION": 6,
                "RELEASE_VALIDATION": 7,
            },
            "families": {"NOT_APPLICABLE": 0, "LF": 1, "HF": 2},
            "stages": {
                "HMM_LIBRARY_SEED": 1,
                "STATE_CHAIN": 2,
                "RESTART_UNIFORMS": 3,
                "SOURCE_RANK_UNIFORMS": 4,
                "OPTIONAL_RESIDUAL_NOISE": 5,
                "REFERENCE_RESAMPLING": 6,
                "VALIDATION_SAMPLING_OR_STRATEGY": 7,
            },
            "model_fit_scopes": {
                "DESIGN_MAIN": 0,
                "DESIGN_MACRO_STABILITY": 1,
                "DESIGN_ROLLING_FOLD": 2,
                "DESIGN_PARAMETRIC_ADEQUACY": 3,
                "PRODUCTION_MAIN_AND_FOLDS": 4,
                "PRODUCTION_MACRO_STABILITY": 5,
                "PRODUCTION_PARAMETRIC_ADEQUACY": 6,
                "BRIDGE_MODEL_SELECTION_PREFIX": 7,
                "BRIDGE_MODEL_SELECTION_FULL": 8,
                "BRIDGE_NONPARAMETRIC_SELECTION_PREFIX": 9,
                "BRIDGE_NONPARAMETRIC_SELECTION_FULL": 10,
                "BRIDGE_MODEL_FIXED_K_PREFIX": 11,
                "BRIDGE_MODEL_FIXED_K_FULL": 12,
                "BRIDGE_NONPARAMETRIC_FIXED_K_PREFIX": 13,
                "BRIDGE_NONPARAMETRIC_FIXED_K_FULL": 14,
                "HMM_IDENTIFICATION_META": 15,
            },
            "calibration_purposes": {
                "MACRO_BOOTSTRAP": 0,
                "ROLLING_SCORE_BOOTSTRAP": 1,
                "LATENT_PPC": 2,
                "PARAMETRIC_ADEQUACY": 3,
                "CONDITIONED_BLOCK": 4,
                "IDEAL_BLOCK": 5,
                "DEPENDENCE_AUDIT": 6,
                "KERNEL_ESTIMATION": 7,
                "KERNEL_VERIFICATION": 8,
                "BRIDGE_HISTORY": 9,
                "G5_OUTER_NULL": 10,
                "G5_QUALIFICATION": 11,
                "META_TYPE_I": 12,
                "META_POWER": 13,
                "SOBOL_OR_STRATEGY": 14,
                "RESERVED": 15,
            },
            "calibration_artifacts": {
                "DESIGN_OR_CONTROL": 0,
                "PRODUCTION_OR_CANDIDATE": 1,
                "BRIDGE_MODEL": 2,
                "BRIDGE_NONPARAMETRIC": 3,
            },
        },
    }


def _canonical_bytes(value: Any) -> bytes:
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
