"""Canonical-ready, non-authorizing orchestration for the dual-DGP bridge.

The lower-level bridge modules deliberately contain no file I/O or process
orchestration.  This module supplies that missing control plane.  It fixes the
registered pair counts and look schedule in code, reduces every worker result
to compact content-addressed evidence before it crosses the process boundary,
and persists an append-only receipt chain that can be resumed after a crash.

Nothing here can authorize generation.  A successful result means only that
the two bridge reference laws passed their registered Tier-A and Tier-B gates;
the enclosing G3 assembler must still verify every other scientific task and
publish the final model artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, TypeAlias, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray

from prpg.errors import IntegrityError, ModelError
from prpg.execution import (
    ExecutionResourceMeasurement,
    MeasuredProcessMapResult,
    MemoryPreflight,
    deterministic_process_map,
    measured_process_map,
)
from prpg.model.alpha import AlphaOffsetGrid
from prpg.model.bridge import (
    ACCELERATOR_AUDIT_PAIRS,
    TIER_A_PAIRS,
    TIER_B_BATCH_SIZE,
    TIER_B_INTERVAL_ERROR,
    TIER_B_MAXIMUM_PAIRS,
    TIER_B_TAIL_PROBABILITY,
    AcceleratorResult,
    BridgeResourcePreflight,
    accelerator_eligibility,
    bridge_resource_preflight,
    tier_a_selection_gate,
    tier_b_tail_gate,
)
from prpg.model.bridge_fixed_k import (
    ActualBridgeExecution,
    FitMode,
    actual_bridge_execution_identity,
    bridge_fixed_k_scope,
    run_bridge_fixed_k_pair,
)
from prpg.model.bridge_history import (
    BRIDGE_FULL_ROWS,
    BRIDGE_PREFIX_ROWS,
    BridgeDGP,
    bridge_parameters_sha256,
    generate_registered_bridge_history,
)
from prpg.model.bridge_selection import (
    BRIDGE_FITS_PER_SELECTION_MEMBER,
    BRIDGE_FIXED_K4_TIER_A_POLICY,
    BRIDGE_FIXED_K4_TIER_A_RESTARTS_PER_PAIR,
    bridge_selection_scope,
    run_bridge_fixed_k4_tier_a_pair,
    run_bridge_tier_a_pair,
)
from prpg.model.calibration import (
    RegisteredRollingBootstrapMaterialization,
    is_registered_rolling_bootstrap_materialization,
    materialize_registered_rolling_bootstrap,
    materialize_rolling_bootstrap_indices,
)
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_preflight import (
    CANONICAL_WORKERS,
    CanonicalBridgeResourcePreflight,
    CanonicalG3Preflight,
    canonical_bridge_resource_preflight,
)
from prpg.model.hmm import HMMParameters
from prpg.model.inference import BinomialInterval, clopper_pearson_interval
from prpg.model.materialization_identity import rng_execution_fingerprint
from prpg.model.scientific_artifact import CANONICAL_SCIENTIFIC_VERSION
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
BridgeDecision: TypeAlias = Literal[
    "continue",
    "pass_above_tail",
    "fail_below_tail",
    "fail_closed_at_maximum",
]
StagedBridgeTaskId: TypeAlias = Literal[
    "bridge_resource_preflight",
    "bridge_tier_a_model_dgp",
    "bridge_tier_a_nonparametric_dgp",
    "bridge_tier_b_model_dgp",
    "bridge_tier_b_nonparametric_dgp",
    "actual_artifact_bridge",
]

BRIDGE_DGPS: Final[tuple[BridgeDGP, BridgeDGP]] = (
    "model",
    "nonparametric",
)
_BRIDGE_MEMBERS: Final[tuple[Literal["prefix", "full"], ...]] = (
    "prefix",
    "full",
)
_BRIDGE_FIT_MODES: Final[tuple[FitMode, ...]] = (
    "accelerated_10",
    "ordinary_50",
)
BRIDGE_ALPHA_GRID_ROLES: Final = (
    "design_monthly",
    "design_daily",
    "production_monthly",
    "production_daily",
)
BRIDGE_ALPHA_LAMBDAS: Final = (0.25, 0.50, 0.75, 1.00)
# A measured benchmark pair runs accelerated and ordinary prefix/full fits:
# two members times (10 + 50) initializations.  Version-3 Tier A runs exactly
# one K=4 fit per member with 50 ordinary starts and no rolling folds.  The
# legacy count remains explicit solely for historical checkpoint verification.
BRIDGE_BENCHMARK_RESTARTS_PER_PAIR: Final = 2 * (10 + 50)
BRIDGE_TIER_A_RESTARTS_PER_PAIR: Final = BRIDGE_FIXED_K4_TIER_A_RESTARTS_PER_PAIR
BRIDGE_LEGACY_TIER_A_RESTARTS_PER_PAIR: Final = 2 * 4 * 7 * 50
BRIDGE_TIER_A_N_STATES: Final = 4
BRIDGE_ACCELERATED_RESTARTS_PER_PAIR: Final = 2 * 10
BRIDGE_ORDINARY_RESTARTS_PER_PAIR: Final = 2 * 50
BRIDGE_TIER_A_STATE_COMPLEXITY_POWER: Final = 3
BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR: Final = 1.25
BRIDGE_TIER_A_BATCH_PAIRS: Final = 5
BRIDGE_CHECKPOINT_SCHEMA_VERSION: Final = 1
BRIDGE_CHECKPOINT_DIRECTORY: Final = "bridge-execution-v1"
STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION: Final = 1
STAGED_BRIDGE_CHECKPOINT_DIRECTORY: Final = "bridge-staged-execution-v1"
_SHA256_CHARS: Final = frozenset("0123456789abcdef")
_BRIDGE_RESULT_CAPABILITY = object()
_STAGED_SESSION_CAPABILITY = object()
_STAGED_TASK_CAPABILITY = object()
_STAGED_LOOK_CAPABILITY = object()
_MINTED_STAGED_SESSIONS: dict[int, CanonicalStagedBridgeSession] = {}
_MINTED_STAGED_TASKS: dict[int, StagedBridgeTaskEvidence] = {}
_MINTED_STAGED_LOOKS: dict[int, StagedBridgePlannedLookEvidence] = {}


@dataclass(frozen=True, slots=True)
class BoundAlphaOffsetGrid:
    """One reverified alpha grid included in the bridge scientific identity."""

    role: str
    frequency: str
    model_artifact_fingerprint: str
    kernel_sample_fingerprint: str
    target_history_fingerprint: str
    grid_fingerprint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "frequency": self.frequency,
            "model_artifact_fingerprint": self.model_artifact_fingerprint,
            "kernel_sample_fingerprint": self.kernel_sample_fingerprint,
            "target_history_fingerprint": self.target_history_fingerprint,
            "grid_fingerprint": self.grid_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class BridgeScientificBinding:
    """Complete worker-count-independent identity of one bridge execution."""

    base_preflight_fingerprint: str
    master_seed: int
    scientific_version: int
    selected_n_states: int
    design_parameters_sha256: str
    design_history_sha256: str
    macro_block_length: int
    actual_bridge_evidence_fingerprint: str
    actual_statistic: float
    alpha_offset_grids: tuple[BoundAlphaOffsetGrid, ...]
    purpose1_trace_fingerprints: tuple[tuple[BridgeDGP, str], ...]
    fingerprint: str

    def identity_dict(self) -> dict[str, Any]:
        fixed_k4_tier_a = self.scientific_version == CANONICAL_SCIENTIFIC_VERSION
        return {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "contract_version": G3_CONTRACT_VERSION,
            "base_preflight_fingerprint": self.base_preflight_fingerprint,
            "master_seed": self.master_seed,
            "scientific_version": self.scientific_version,
            "selected_n_states": self.selected_n_states,
            "design_parameters_sha256": self.design_parameters_sha256,
            "design_history_sha256": self.design_history_sha256,
            "macro_block_length": self.macro_block_length,
            "actual_bridge_evidence_fingerprint": (
                self.actual_bridge_evidence_fingerprint
            ),
            "actual_statistic": self.actual_statistic,
            "alpha_offset_grids": [item.as_dict() for item in self.alpha_offset_grids],
            "purpose1_trace_fingerprints": dict(self.purpose1_trace_fingerprints),
            "tier_a_pairs_per_dgp": TIER_A_PAIRS,
            "accelerator_audit_pairs_per_dgp": ACCELERATOR_AUDIT_PAIRS,
            "tier_b_batch_pairs": TIER_B_BATCH_SIZE,
            "tier_b_maximum_pairs_per_dgp": TIER_B_MAXIMUM_PAIRS,
            "tier_b_interval_error": TIER_B_INTERVAL_ERROR,
            "tier_b_tail_probability": TIER_B_TAIL_PROBABILITY,
            "benchmark_restarts_per_pair": BRIDGE_BENCHMARK_RESTARTS_PER_PAIR,
            "tier_a_policy": (
                BRIDGE_FIXED_K4_TIER_A_POLICY
                if fixed_k4_tier_a
                else "cross_k_rolling_v1"
            ),
            "tier_a_candidate_n_states": (
                [BRIDGE_TIER_A_N_STATES] if fixed_k4_tier_a else [2, 3, 4, 5]
            ),
            "tier_a_fits_per_member": 1 if fixed_k4_tier_a else 28,
            "tier_a_rolling_folds": 0 if fixed_k4_tier_a else 6,
            "tier_a_decision_uses_rolling_bootstrap": not fixed_k4_tier_a,
            "tier_a_restarts_per_pair": (
                BRIDGE_TIER_A_RESTARTS_PER_PAIR
                if fixed_k4_tier_a
                else BRIDGE_LEGACY_TIER_A_RESTARTS_PER_PAIR
            ),
            "tier_a_state_complexity_power": BRIDGE_TIER_A_STATE_COMPLEXITY_POWER,
            "resource_projection_safety_factor": (
                BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR
            ),
            "dgp_order": list(BRIDGE_DGPS),
        }

    @property
    def purpose1_by_dgp(self) -> Mapping[BridgeDGP, str]:
        return MappingProxyType(dict(self.purpose1_trace_fingerprints))


@dataclass(frozen=True, slots=True)
class CanonicalBridgeRequest:
    """All upstream scientific inputs required by the canonical bridge."""

    base_preflight: CanonicalG3Preflight
    master_seed: int
    scientific_version: int
    design_parameters: HMMParameters
    design_standardized_history: FloatArray
    design_continuation: BoolArray
    macro_block_length: int
    actual_bridge: ActualBridgeExecution
    alpha_offset_grids: tuple[
        AlphaOffsetGrid, AlphaOffsetGrid, AlphaOffsetGrid, AlphaOffsetGrid
    ]


@dataclass(frozen=True, slots=True)
class StagedBridgeCommonBinding:
    """Future-outcome-free identity shared by all staged bridge tasks."""

    base_preflight_fingerprint: str
    master_seed: int
    scientific_version: int
    selected_n_states: int
    design_parameters_sha256: str
    design_history_sha256: str
    macro_block_length: int
    fingerprint: str

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "contract_version": G3_CONTRACT_VERSION,
            "base_preflight_fingerprint": self.base_preflight_fingerprint,
            "master_seed": self.master_seed,
            "scientific_version": self.scientific_version,
            "selected_n_states": self.selected_n_states,
            "design_parameters_sha256": self.design_parameters_sha256,
            "design_history_sha256": self.design_history_sha256,
            "macro_block_length": self.macro_block_length,
        }


@dataclass(frozen=True, slots=True, init=False)
class CanonicalStagedBridgeSession:
    """Producer-owned staged bridge session over an exact durable prefix."""

    generation_authorized: ClassVar[Literal[False]] = False

    binding: StagedBridgeCommonBinding
    checkpoint_path: Path
    _request: CanonicalBridgeRequest = field(repr=False, compare=False)
    _legacy_binding: BridgeScientificBinding = field(repr=False, compare=False)
    _rolling_materializations: tuple[
        RegisteredRollingBootstrapMaterialization,
        RegisteredRollingBootstrapMaterialization,
    ] = field(repr=False, compare=False)
    _stage_store: _StagedBridgeCheckpointStore = field(repr=False, compare=False)
    _operational_store: BridgeCheckpointStore = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "CanonicalStagedBridgeSession is created only by "
            "prepare_canonical_staged_bridge"
        )


@dataclass(frozen=True, slots=True)
class StagedBridgeTaskEvidenceIdentity:
    """Reverified task-local scientific identity for one completed stage."""

    task_id: StagedBridgeTaskId
    dgp: BridgeDGP | None
    passed: bool
    base_preflight_fingerprint: str
    selected_n_states: int
    common_binding_fingerprint: str
    parent_task_fingerprints: tuple[tuple[str, str], ...]
    source_materialization_fingerprint: str
    rng_execution_fingerprint: str | None
    result_fingerprint: str
    evidence_fingerprint: str
    stage_receipt_fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class StagedBridgeTaskEvidence:
    """Object-specific task completion evidence minted from one stage receipt."""

    generation_authorized: ClassVar[Literal[False]] = False

    task_id: StagedBridgeTaskId
    dgp: BridgeDGP | None
    passed: bool
    common_binding_fingerprint: str
    parent_task_fingerprints: tuple[tuple[str, str], ...]
    source_materialization_fingerprint: str
    rng_execution_fingerprint: str | None
    result_fingerprint: str
    evidence_fingerprint: str
    stage_receipt_fingerprint: str
    _session: CanonicalStagedBridgeSession = field(repr=False, compare=False)
    _event_index: int = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("StagedBridgeTaskEvidence is minted only by the staged driver")


@dataclass(frozen=True, slots=True)
class StagedBridgePlannedLookIdentity:
    """Reverified cumulative identity for one explicitly emitted Tier-B look."""

    task_id: Literal["bridge_tier_b_model_dgp", "bridge_tier_b_nonparametric_dgp"]
    dgp: BridgeDGP
    look_index: int
    sample_count: int
    exceedances: int
    decision: BridgeDecision
    common_binding_fingerprint: str
    parent_task_fingerprints: tuple[tuple[str, str], ...]
    source_materialization_fingerprint: str
    rng_execution_fingerprint: str
    result_fingerprint: str
    evidence_fingerprint: str
    stage_receipt_fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class StagedBridgePlannedLookEvidence:
    """One durable, registered-order Tier-B planned-look observation."""

    task_id: Literal["bridge_tier_b_model_dgp", "bridge_tier_b_nonparametric_dgp"]
    dgp: BridgeDGP
    look_index: int
    sample_count: int
    exceedances: int
    decision: BridgeDecision
    common_binding_fingerprint: str
    parent_task_fingerprints: tuple[tuple[str, str], ...]
    source_materialization_fingerprint: str
    rng_execution_fingerprint: str
    result_fingerprint: str
    evidence_fingerprint: str
    stage_receipt_fingerprint: str
    _session: CanonicalStagedBridgeSession = field(repr=False, compare=False)
    _event_index: int = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "StagedBridgePlannedLookEvidence is minted only by the staged driver"
        )


@dataclass(frozen=True, slots=True)
class ReverifiedStagedBridgePrefix:
    """Exact valid prefix reconstructed from append-only staged receipts."""

    common_binding_fingerprint: str
    task_evidence: tuple[StagedBridgeTaskEvidence, ...]
    planned_looks: tuple[StagedBridgePlannedLookEvidence, ...]
    receipt_fingerprints: tuple[str, ...]
    next_action: str | None
    terminal: bool


@dataclass(frozen=True, slots=True)
class BridgeTierAPairEvidence:
    """Fit-light Tier-A result retained from one worker."""

    dgp: BridgeDGP
    pair: int
    selected_n_states: int
    history_sha256: str
    purpose1_trace_sha256: str
    prefix_selected_n_states: int | None
    full_selected_n_states: int | None
    prefix_passed: bool
    full_passed: bool
    failure_types: tuple[str, ...]
    success: bool
    binding_fingerprint: str
    evidence_fingerprint: str
    tier_a_policy: str = "cross_k_rolling_v1"
    prefix_observed_episode_counts: tuple[int, ...] = ()
    full_observed_episode_counts: tuple[int, ...] = ()
    prefix_reference_to_candidate: tuple[int, ...] = ()
    full_reference_to_candidate: tuple[int, ...] = ()

    def identity_dict(self) -> dict[str, Any]:
        return {
            "dgp": self.dgp,
            "pair": self.pair,
            "selected_n_states": self.selected_n_states,
            "history_sha256": self.history_sha256,
            "purpose1_trace_sha256": self.purpose1_trace_sha256,
            "prefix_selected_n_states": self.prefix_selected_n_states,
            "full_selected_n_states": self.full_selected_n_states,
            "prefix_passed": self.prefix_passed,
            "full_passed": self.full_passed,
            "failure_types": list(self.failure_types),
            "success": self.success,
            "binding_fingerprint": self.binding_fingerprint,
            "tier_a_policy": self.tier_a_policy,
            "prefix_observed_episode_counts": list(self.prefix_observed_episode_counts),
            "full_observed_episode_counts": list(self.full_observed_episode_counts),
            "prefix_reference_to_candidate": list(self.prefix_reference_to_candidate),
            "full_reference_to_candidate": list(self.full_reference_to_candidate),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class BridgeTierBPairEvidence:
    """One fixed-K pair reduced to only its gate statistic and provenance."""

    dgp: BridgeDGP
    pair: int
    selected_n_states: int
    mode: FitMode
    history_sha256: str
    valid: bool
    statistic: float | None
    failure_type: str | None
    failure_message: str | None
    binding_fingerprint: str
    evidence_fingerprint: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "dgp": self.dgp,
            "pair": self.pair,
            "selected_n_states": self.selected_n_states,
            "mode": self.mode,
            "history_sha256": self.history_sha256,
            "valid": self.valid,
            "statistic": self.statistic,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "binding_fingerprint": self.binding_fingerprint,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class BridgeBenchmarkPairEvidence:
    """Compact first-100 ten-versus-fifty-start comparison for one DGP."""

    dgp: BridgeDGP
    pair: int
    selected_n_states: int
    history_sha256: str
    accelerated: BridgeTierBPairEvidence
    ordinary: BridgeTierBPairEvidence
    accelerated_valid: tuple[bool, bool]
    ordinary_valid: tuple[bool, bool]
    likelihood_difference_per_observation: tuple[float, float]
    maximum_component_difference: tuple[float, float]
    statistic_difference: tuple[float, float]
    component_decisions_identical: tuple[bool, bool]
    elapsed_seconds: float
    binding_fingerprint: str
    scientific_fingerprint: str
    evidence_fingerprint: str

    def scientific_identity_dict(self) -> dict[str, Any]:
        return {
            "dgp": self.dgp,
            "pair": self.pair,
            "selected_n_states": self.selected_n_states,
            "history_sha256": self.history_sha256,
            "accelerated_evidence_fingerprint": self.accelerated.evidence_fingerprint,
            "ordinary_evidence_fingerprint": self.ordinary.evidence_fingerprint,
            "accelerated_valid": list(self.accelerated_valid),
            "ordinary_valid": list(self.ordinary_valid),
            "likelihood_difference_per_observation": list(
                self.likelihood_difference_per_observation
            ),
            "maximum_component_difference": list(self.maximum_component_difference),
            "statistic_difference": list(self.statistic_difference),
            "component_decisions_identical": list(self.component_decisions_identical),
            "binding_fingerprint": self.binding_fingerprint,
        }

    def identity_dict(self) -> dict[str, Any]:
        return {
            **self.scientific_identity_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "scientific_fingerprint": self.scientific_fingerprint,
            "accelerated": self.accelerated.as_dict(),
            "ordinary": self.ordinary.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "evidence_fingerprint": self.evidence_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class BridgeTierAResult:
    """Durable exact-250 Tier-A reduction for one DGP."""

    dgp: BridgeDGP
    pairs: int
    successes: int
    lower_bound: float
    passed: bool
    scientific_fingerprint: str
    receipt_fingerprint: str


@dataclass(frozen=True, slots=True)
class BridgeAcceleratorDecision:
    """Compact accelerator audit plus retained first-100 pair family."""

    dgp: BridgeDGP
    result: AcceleratorResult
    retained_mode: FitMode
    retained_pairs: tuple[BridgeTierBPairEvidence, ...]
    scientific_fingerprint: str


@dataclass(frozen=True, slots=True)
class BridgeBenchmarkResult:
    """Measured first-100 evidence and canonical bridge resource decision."""

    accelerators: tuple[BridgeAcceleratorDecision, ...]
    measurement: ExecutionResourceMeasurement
    projection: BridgeResourcePreflight
    canonical_preflight: CanonicalBridgeResourcePreflight | None
    passed: bool
    reasons: tuple[str, ...]
    receipt_fingerprint: str


@dataclass(frozen=True, slots=True)
class BridgeTierBLookResult:
    """One verified local planned-look receipt."""

    dgp: BridgeDGP
    look_index: int
    sample_count: int
    exceedances: int
    interval: BinomialInterval
    decision: BridgeDecision
    receipt_fingerprint: str


@dataclass(frozen=True, slots=True)
class BridgeDGPResult:
    """The two registered bridge tiers for one reference law."""

    dgp: BridgeDGP
    tier_a: BridgeTierAResult | None
    accelerator_eligible: bool | None
    tier_b_decision: BridgeDecision | None
    tier_b_pairs: int
    tier_b_passed: bool
    scientific_fingerprint: str

    @property
    def passed(self) -> bool:
        return bool(
            self.tier_a is not None and self.tier_a.passed and self.tier_b_passed
        )


@dataclass(frozen=True, slots=True)
class BridgeDriverResult:
    """Non-authorizing dual-reference outcome returned by the driver."""

    generation_authorized: ClassVar[Literal[False]] = False
    artifact_kind: ClassVar[Literal["bridge_execution_evidence"]] = (
        "bridge_execution_evidence"
    )

    binding: BridgeScientificBinding
    checkpoint_path: Path
    dgp_results: tuple[BridgeDGPResult, ...]
    benchmark: BridgeBenchmarkResult | None
    bridge_gate_passed: bool
    reasons: tuple[str, ...]
    scientific_fingerprint: str
    final_receipt_fingerprint: str
    _verification_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class ReverifiedBridgeDriverEvidence:
    """Checkpoint-rederived, non-authorizing evidence for the G3 boundary.

    The in-memory driver seal proves that the object came from this module;
    it does not make an old object authoritative after its checkpoint tree is
    changed.  This value is returned only after the binding, benchmark,
    Tier-A batches/aggregates, Tier-B look chains, and final receipt have all
    been reread and reconciled with the original canonical request.
    """

    generation_authorized: ClassVar[Literal[False]] = False

    driver_result: BridgeDriverResult
    request: CanonicalBridgeRequest
    benchmark: BridgeBenchmarkResult
    dgp_results: tuple[BridgeDGPResult, ...]
    tier_b_looks: tuple[
        tuple[BridgeTierBLookResult, ...], tuple[BridgeTierBLookResult, ...]
    ]
    final_receipt_fingerprint: str

    @property
    def binding(self) -> BridgeScientificBinding:
        return self.driver_result.binding

    @property
    def tier_b_looks_by_dgp(
        self,
    ) -> Mapping[BridgeDGP, tuple[BridgeTierBLookResult, ...]]:
        return MappingProxyType(dict(zip(BRIDGE_DGPS, self.tier_b_looks, strict=True)))


def is_verified_bridge_driver_result(value: object) -> TypeGuard[BridgeDriverResult]:
    """Return whether the result came from verified driver/checkpoint execution."""

    return (
        isinstance(value, BridgeDriverResult)
        and value._verification_seal is _BRIDGE_RESULT_CAPABILITY
    )


def _verified_bridge_result(value: BridgeDriverResult) -> BridgeDriverResult:
    object.__setattr__(value, "_verification_seal", _BRIDGE_RESULT_CAPABILITY)
    return value


@dataclass(frozen=True, slots=True)
class _TierABatchJob:
    dgp: BridgeDGP
    pair_start: int
    pair_stop: int
    binding_fingerprint: str
    selected_n_states: int
    master_seed: int
    scientific_version: int
    design_parameters: HMMParameters
    design_history: FloatArray
    design_continuation: BoolArray
    macro_block_length: int
    purpose1_trace: IntArray
    purpose1_trace_sha256: str


@dataclass(frozen=True, slots=True)
class _FixedKJob:
    dgp: BridgeDGP
    pair: int
    binding_fingerprint: str
    selected_n_states: int
    master_seed: int
    scientific_version: int
    design_parameters: HMMParameters
    design_history: FloatArray
    design_continuation: BoolArray
    macro_block_length: int
    mode: FitMode


@dataclass(frozen=True, slots=True)
class _BenchmarkJob:
    dgp: BridgeDGP
    pair: int
    binding_fingerprint: str
    selected_n_states: int
    master_seed: int
    scientific_version: int
    design_parameters: HMMParameters
    design_history: FloatArray
    design_continuation: BoolArray
    macro_block_length: int


@dataclass(frozen=True, slots=True)
class _Receipt:
    path: Path
    identity: Mapping[str, Any]
    fingerprint: str
    reused: bool


class _StagedBridgeCheckpointStore:
    """Append-only task/look event chain; not a G3 publication authority."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / STAGED_BRIDGE_CHECKPOINT_DIRECTORY

    def initialize(self, binding: StagedBridgeCommonBinding) -> Path:
        _validate_staged_common_binding(binding)
        path = self.root / binding.fingerprint
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink(path, "staged bridge checkpoint directory")
        self._write(
            path / "binding.json",
            {
                "schema_version": STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION,
                "kind": "staged_bridge_binding",
                "binding": binding.identity_dict(),
                "binding_fingerprint": binding.fingerprint,
            },
        )
        return path

    def verify_binding(self, binding: StagedBridgeCommonBinding) -> Path:
        path = self.root / binding.fingerprint
        receipt = self._read(path / "binding.json")
        expected = {
            "schema_version": STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "staged_bridge_binding",
            "binding": binding.identity_dict(),
            "binding_fingerprint": binding.fingerprint,
        }
        if dict(receipt.identity) != expected:
            raise IntegrityError("staged bridge binding differs from the request")
        return path

    def event_receipts(
        self, binding: StagedBridgeCommonBinding
    ) -> tuple[_Receipt, ...]:
        path = self.verify_binding(binding)
        names = sorted(item.name for item in path.iterdir())
        event_names = [name for name in names if name.startswith("event-")]
        allowed_other = {"binding.json"}
        if any(
            name not in allowed_other and not name.startswith("event-")
            for name in names
        ):
            raise IntegrityError("staged bridge directory contains an extra file")
        receipts: list[_Receipt] = []
        for index, name in enumerate(event_names):
            expected = f"event-{index:03d}.json"
            if name != expected:
                raise IntegrityError("staged bridge event chain contains a gap")
            receipts.append(self._read(path / name))
        return tuple(receipts)

    def write_event(
        self,
        binding: StagedBridgeCommonBinding,
        identity: Mapping[str, Any],
    ) -> _Receipt:
        prior = self.event_receipts(binding)
        index = len(prior)
        previous = prior[-1].fingerprint if prior else None
        supplied = _json_object(identity, "staged bridge event")
        if (
            supplied.get("event_index") != index
            or supplied.get("previous_receipt_fingerprint") != previous
            or supplied.get("common_binding_fingerprint") != binding.fingerprint
        ):
            raise ModelError("staged bridge event is not the exact next prefix item")
        return self._write(
            self.verify_binding(binding) / f"event-{index:03d}.json",
            supplied,
        )

    def _write(self, path: Path, identity: Mapping[str, Any]) -> _Receipt:
        normalized = _json_object(identity, "staged bridge checkpoint identity")
        fingerprint = _sha256_bytes(_canonical_bytes(normalized))
        content = _canonical_bytes({**normalized, "receipt_fingerprint": fingerprint})
        reused = _write_or_verify_exact(path, content)
        loaded = self._read(path)
        if loaded.fingerprint != fingerprint:
            raise IntegrityError("staged bridge write changed its receipt hash")
        return _Receipt(path, normalized, fingerprint, reused)

    def _read(self, path: Path) -> _Receipt:
        raw = _read_regular(path)
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "staged bridge checkpoint is not valid JSON"
            ) from error
        value = _json_object(decoded, "staged bridge checkpoint")
        if _canonical_bytes(value) != raw:
            raise IntegrityError("staged bridge checkpoint JSON is not canonical")
        fingerprint = value.pop("receipt_fingerprint", None)
        _require_sha256(
            fingerprint,
            "staged bridge checkpoint receipt fingerprint",
            IntegrityError,
        )
        if fingerprint != _sha256_bytes(_canonical_bytes(value)):
            raise IntegrityError("staged bridge receipt fingerprint is invalid")
        return _Receipt(path, value, fingerprint, True)


class BridgeCheckpointStore:
    """Append-only, content-addressed bridge checkpoint namespace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / BRIDGE_CHECKPOINT_DIRECTORY

    def initialize(self, binding: BridgeScientificBinding) -> Path:
        """Create or exactly verify the bridge binding before any worker starts."""

        _validate_binding(binding)
        path = self.root / binding.fingerprint
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink(path, "bridge checkpoint directory")
        identity = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_binding",
            "binding": binding.identity_dict(),
            "binding_fingerprint": binding.fingerprint,
        }
        self._write(path / "binding.json", identity)
        return path

    def verify_binding(self, binding: BridgeScientificBinding) -> Path:
        """Reverify the exact on-disk binding and reject namespace drift."""

        path = self.root / binding.fingerprint
        receipt = self._read(path / "binding.json")
        expected = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_binding",
            "binding": binding.identity_dict(),
            "binding_fingerprint": binding.fingerprint,
        }
        if dict(receipt.identity) != expected:
            raise IntegrityError("bridge checkpoint binding differs from the request")
        return path

    def tier_a_receipt(
        self, binding: BridgeScientificBinding, dgp: BridgeDGP
    ) -> _Receipt | None:
        return self._optional(self._path(binding, f"tier-a-{dgp}.json"))

    def tier_a_batch_receipts(
        self, binding: BridgeScientificBinding, dgp: BridgeDGP
    ) -> tuple[_Receipt, ...]:
        """Load the contiguous hash chain of fit-light Tier-A batch receipts."""

        directory = self._path(binding, f"tier-a-{dgp}")
        if not directory.exists():
            return ()
        _reject_symlink(directory, "bridge Tier-A batch directory")
        maximum = TIER_A_PAIRS // BRIDGE_TIER_A_BATCH_PAIRS
        allowed = [f"batch-{index:02d}.json" for index in range(maximum)]
        names = sorted(item.name for item in directory.iterdir())
        if any(name not in allowed for name in names):
            raise IntegrityError("bridge Tier-A batch directory contains an extra file")
        receipts: list[_Receipt] = []
        for index in range(maximum):
            path = directory / f"batch-{index:02d}.json"
            if not path.exists():
                break
            receipts.append(self._read(path))
        if names != [item.path.name for item in receipts]:
            raise IntegrityError("bridge Tier-A batch receipts contain a sequence gap")
        return tuple(receipts)

    def write_tier_a_batch(
        self,
        binding: BridgeScientificBinding,
        *,
        dgp: BridgeDGP,
        batch_index: int,
        pairs: Sequence[BridgeTierAPairEvidence],
        prior_receipts: Sequence[_Receipt],
    ) -> _Receipt:
        maximum = TIER_A_PAIRS // BRIDGE_TIER_A_BATCH_PAIRS
        if batch_index != len(prior_receipts) or not 0 <= batch_index < maximum:
            raise ModelError("Tier-A batch is not the next registered batch")
        start = batch_index * BRIDGE_TIER_A_BATCH_PAIRS
        stop = start + BRIDGE_TIER_A_BATCH_PAIRS
        checked = tuple(_validated_tier_a_pair(item, binding, dgp) for item in pairs)
        if len(checked) != BRIDGE_TIER_A_BATCH_PAIRS or tuple(
            item.pair for item in checked
        ) != tuple(range(start, stop)):
            raise ModelError("Tier-A batch has the wrong exact pair interval")
        previous = prior_receipts[-1].fingerprint if prior_receipts else None
        identity = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_tier_a_batch",
            "binding_fingerprint": binding.fingerprint,
            "dgp": dgp,
            "selected_n_states": binding.selected_n_states,
            "purpose1_trace_sha256": binding.purpose1_by_dgp[dgp],
            "batch_index": batch_index,
            "pair_start": start,
            "pair_stop": stop,
            "pairs": [item.as_dict() for item in checked],
            "previous_receipt_fingerprint": previous,
            "scientific_fingerprint": _sha256_json(
                {
                    "binding_fingerprint": binding.fingerprint,
                    "dgp": dgp,
                    "batch_index": batch_index,
                    "pair_evidence": [item.evidence_fingerprint for item in checked],
                    "previous_receipt_fingerprint": previous,
                }
            ),
        }
        directory = self._path(binding, f"tier-a-{dgp}")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return self._write(directory / f"batch-{batch_index:02d}.json", identity)

    def write_tier_a(
        self,
        binding: BridgeScientificBinding,
        dgp: BridgeDGP,
        pairs: Sequence[BridgeTierAPairEvidence],
    ) -> _Receipt:
        records = tuple(_validated_tier_a_pair(item, binding, dgp) for item in pairs)
        if len(records) != TIER_A_PAIRS or tuple(
            item.pair for item in records
        ) != tuple(range(TIER_A_PAIRS)):
            raise ModelError("Tier-A checkpoint requires exact ordered pairs 0..249")
        gate = tier_a_selection_gate(
            np.asarray([item.success for item in records], dtype=np.bool_)
        )
        scientific = _sha256_json(
            {
                "binding_fingerprint": binding.fingerprint,
                "dgp": dgp,
                "pair_evidence": [item.evidence_fingerprint for item in records],
                "successes": gate.successes,
                "lower_bound": gate.one_sided_lower_95,
                "passed": gate.passed,
            }
        )
        identity = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_tier_a",
            "binding_fingerprint": binding.fingerprint,
            "dgp": dgp,
            "selected_n_states": binding.selected_n_states,
            "purpose1_trace_sha256": binding.purpose1_by_dgp[dgp],
            "pairs": [item.as_dict() for item in records],
            "gate": {
                "pairs": gate.pairs,
                "successes": gate.successes,
                "estimate": gate.estimate,
                "one_sided_lower_95": gate.one_sided_lower_95,
                "required_lower_bound": gate.required_lower_bound,
                "passed": gate.passed,
            },
            "scientific_fingerprint": scientific,
        }
        return self._write(self._path(binding, f"tier-a-{dgp}.json"), identity)

    def benchmark_receipt(self, binding: BridgeScientificBinding) -> _Receipt | None:
        return self._optional(self._path(binding, "tier-b-benchmark.json"))

    def write_benchmark(
        self,
        binding: BridgeScientificBinding,
        *,
        records: Sequence[BridgeBenchmarkPairEvidence],
        accelerators: Sequence[BridgeAcceleratorDecision],
        measurement: ExecutionResourceMeasurement,
        projection: BridgeResourcePreflight,
        canonical_preflight: CanonicalBridgeResourcePreflight | None,
        passed: bool,
        reasons: Sequence[str],
    ) -> _Receipt:
        checked_records = tuple(
            _validated_benchmark_pair(item, binding) for item in records
        )
        expected_coordinates = tuple(
            (dgp, pair)
            for dgp in BRIDGE_DGPS
            for pair in range(ACCELERATOR_AUDIT_PAIRS)
        )
        if (
            tuple((item.dgp, item.pair) for item in checked_records)
            != expected_coordinates
        ):
            raise ModelError(
                "bridge benchmark requires exact DGP-major first-100 pairs"
            )
        checked_accelerators = tuple(accelerators)
        if tuple(item.dgp for item in checked_accelerators) != BRIDGE_DGPS:
            raise ModelError("bridge benchmark accelerator order is inconsistent")
        identity = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_tier_b_benchmark",
            "binding_fingerprint": binding.fingerprint,
            "records": [item.as_dict() for item in checked_records],
            "accelerators": [_accelerator_dict(item) for item in checked_accelerators],
            "measurement": _measurement_dict(measurement),
            "projection": _projection_dict(projection),
            "canonical_preflight": (
                None
                if canonical_preflight is None
                else {
                    **canonical_preflight.identity_dict(),
                    "fingerprint": canonical_preflight.fingerprint,
                }
            ),
            "passed": passed,
            "reasons": list(reasons),
        }
        return self._write(self._path(binding, "tier-b-benchmark.json"), identity)

    def tier_b_look_receipts(
        self, binding: BridgeScientificBinding, dgp: BridgeDGP
    ) -> tuple[_Receipt, ...]:
        directory = self._path(binding, f"tier-b-{dgp}")
        if not directory.exists():
            return ()
        _reject_symlink(directory, "bridge Tier-B look directory")
        names = sorted(item.name for item in directory.iterdir())
        allowed = [f"look-{index:02d}.json" for index in range(1, 11)]
        if any(name not in allowed for name in names):
            raise IntegrityError("bridge Tier-B look directory contains an extra file")
        receipts: list[_Receipt] = []
        for index in range(1, 11):
            path = directory / f"look-{index:02d}.json"
            if not path.exists():
                break
            receipts.append(self._read(path))
        if names != [item.path.name for item in receipts]:
            raise IntegrityError("bridge Tier-B look receipts contain a sequence gap")
        return tuple(receipts)

    def verified_tier_b_look_results(
        self, binding: BridgeScientificBinding, dgp: BridgeDGP
    ) -> tuple[BridgeTierBLookResult, ...]:
        """Return typed local looks for the enclosing G3 receipt publisher.

        The bridge layer cannot own a G3 ``LaunchLease`` because Tier-B model
        and nonparametric tasks are separate nodes in that higher-level DAG.
        This adapter supplies fully reverified observations to those code-owned
        task handlers without giving this lower layer authority to mutate the
        run store or mark a G3 task passed.
        """

        receipts = self.tier_b_look_receipts(binding, dgp)
        _statistics_from_look_receipts(receipts, binding, dgp)
        output: list[BridgeTierBLookResult] = []
        for receipt in receipts:
            identity = receipt.identity
            sample_count = _integer(identity["sample_count"], "look sample count")
            exceedances = _integer(identity["exceedances"], "look exceedances")
            interval = clopper_pearson_interval(
                exceedances,
                sample_count,
                confidence=1.0 - TIER_B_INTERVAL_ERROR,
            )
            decision = identity["decision"]
            if decision not in {
                "continue",
                "pass_above_tail",
                "fail_below_tail",
                "fail_closed_at_maximum",
            }:
                raise IntegrityError("bridge planned-look decision is invalid")
            output.append(
                BridgeTierBLookResult(
                    dgp,
                    _integer(identity["look_index"], "look index"),
                    sample_count,
                    exceedances,
                    interval,
                    decision,
                    receipt.fingerprint,
                )
            )
        return tuple(output)

    def write_tier_b_look(
        self,
        binding: BridgeScientificBinding,
        *,
        dgp: BridgeDGP,
        look_index: int,
        pairs: Sequence[BridgeTierBPairEvidence],
        prior_receipts: Sequence[_Receipt],
        cumulative_statistics: Sequence[float],
        decision: BridgeDecision,
        interval: BinomialInterval,
        exceedances: int,
    ) -> _Receipt:
        if look_index != len(prior_receipts) + 1 or not 1 <= look_index <= 10:
            raise ModelError("Tier-B look index is not the next registered look")
        if prior_receipts and prior_receipts[-1].identity["decision"] != "continue":
            raise ModelError("no Tier-B work is allowed after a terminal decision")
        start = (look_index - 1) * TIER_B_BATCH_SIZE
        stop = look_index * TIER_B_BATCH_SIZE
        checked = tuple(_validated_tier_b_pair(item, binding, dgp) for item in pairs)
        if len(checked) != TIER_B_BATCH_SIZE or tuple(
            item.pair for item in checked
        ) != tuple(range(start, stop)):
            raise ModelError("Tier-B look requires one exact ordered 1,000-pair batch")
        values = np.asarray(cumulative_statistics, dtype=np.float64)
        if values.shape != (stop,) or not np.isfinite(values).all():
            raise ModelError(
                "Tier-B cumulative statistics are incomplete or non-finite"
            )
        if any(not item.valid or item.statistic is None for item in checked):
            raise ModelError("Tier-B look cannot retain an invalid fixed-K pair")
        expected_exceedances = int(np.count_nonzero(values >= binding.actual_statistic))
        expected_interval = clopper_pearson_interval(
            expected_exceedances,
            stop,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        expected_decision = _tier_b_decision(expected_interval, stop)
        if (
            exceedances != expected_exceedances
            or interval != expected_interval
            or decision != expected_decision
        ):
            raise ModelError("Tier-B look decision does not match its statistics")
        previous = prior_receipts[-1].fingerprint if prior_receipts else None
        scientific = _sha256_json(
            {
                "binding_fingerprint": binding.fingerprint,
                "dgp": dgp,
                "look_index": look_index,
                "pair_evidence": [item.evidence_fingerprint for item in checked],
                "sample_count": stop,
                "exceedances": exceedances,
                "interval": _interval_dict(interval),
                "decision": decision,
                "previous_receipt_fingerprint": previous,
            }
        )
        identity = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_tier_b_look",
            "binding_fingerprint": binding.fingerprint,
            "dgp": dgp,
            "selected_n_states": binding.selected_n_states,
            "look_index": look_index,
            "pair_start": start,
            "pair_stop": stop,
            "sample_count": stop,
            "pairs": [item.as_dict() for item in checked],
            "exceedances": exceedances,
            "estimate": exceedances / stop,
            "interval": _interval_dict(interval),
            "decision": decision,
            "previous_receipt_fingerprint": previous,
            "scientific_fingerprint": scientific,
        }
        directory = self._path(binding, f"tier-b-{dgp}")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return self._write(directory / f"look-{look_index:02d}.json", identity)

    def final_receipt(self, binding: BridgeScientificBinding) -> _Receipt | None:
        return self._optional(self._path(binding, "FINAL.json"))

    def write_final(
        self,
        binding: BridgeScientificBinding,
        *,
        dgp_results: Sequence[BridgeDGPResult],
        bridge_gate_passed: bool,
        reasons: Sequence[str],
        scientific_fingerprint: str,
        benchmark_receipt_fingerprint: str | None,
    ) -> _Receipt:
        results = tuple(dgp_results)
        if tuple(item.dgp for item in results) != BRIDGE_DGPS:
            raise ModelError("final bridge result requires both DGPs in registry order")
        expected = all(item.passed for item in results)
        if bridge_gate_passed != expected:
            raise ModelError(
                "combined bridge gate must equal the conjunction of both DGPs"
            )
        identity = {
            "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "bridge_final",
            "binding_fingerprint": binding.fingerprint,
            "dgp_results": [_dgp_result_dict(item) for item in results],
            "benchmark_receipt_fingerprint": benchmark_receipt_fingerprint,
            "bridge_gate_passed": bridge_gate_passed,
            "generation_authorized": False,
            "reasons": list(reasons),
            "scientific_fingerprint": scientific_fingerprint,
        }
        return self._write(self._path(binding, "FINAL.json"), identity)

    def _path(self, binding: BridgeScientificBinding, name: str) -> Path:
        self.verify_binding(binding)
        return self.root / binding.fingerprint / name

    def _optional(self, path: Path) -> _Receipt | None:
        return None if not path.exists() else self._read(path)

    def _write(self, path: Path, identity: Mapping[str, Any]) -> _Receipt:
        normalized = _json_object(identity, "bridge checkpoint identity")
        fingerprint = _sha256_bytes(_canonical_bytes(normalized))
        value = {**normalized, "receipt_fingerprint": fingerprint}
        content = _canonical_bytes(value)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        reused = _write_or_verify_exact(path, content)
        loaded = self._read(path)
        if loaded.fingerprint != fingerprint:
            raise IntegrityError("bridge checkpoint write did not preserve its hash")
        return _Receipt(path, normalized, fingerprint, reused)

    def _read(self, path: Path) -> _Receipt:
        raw = _read_regular(path)
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError("bridge checkpoint is not valid JSON") from error
        value = _json_object(decoded, "bridge checkpoint")
        if _canonical_bytes(value) != raw:
            raise IntegrityError("bridge checkpoint JSON is not canonical")
        fingerprint = value.pop("receipt_fingerprint", None)
        _require_sha256(
            fingerprint, "bridge checkpoint receipt fingerprint", IntegrityError
        )
        expected = _sha256_bytes(_canonical_bytes(value))
        if fingerprint != expected:
            raise IntegrityError("bridge checkpoint receipt fingerprint is invalid")
        return _Receipt(path, value, fingerprint, True)


def prepare_canonical_staged_bridge(
    request: CanonicalBridgeRequest,
    checkpoint_root: str | Path,
) -> CanonicalStagedBridgeSession:
    """Prepare or resume the future-outcome-safe canonical bridge state machine."""

    legacy_binding, _ = _prepare_request(request)
    common = _staged_common_binding(request, legacy_binding)
    materializations = tuple(
        materialize_registered_rolling_bootstrap(
            master_seed=request.master_seed,
            scientific_version=request.scientific_version,
            artifact=_bridge_artifact(dgp),
        )
        for dgp in BRIDGE_DGPS
    )
    if len(materializations) != 2 or any(
        not is_registered_rolling_bootstrap_materialization(item)
        for item in materializations
    ):
        raise ModelError("staged bridge rolling materialization is not registered")
    for dgp, materialization in zip(BRIDGE_DGPS, materializations, strict=True):
        if (
            _purpose1_trace_sha256(materialization.indices)
            != legacy_binding.purpose1_by_dgp[dgp]
        ):
            raise ModelError(
                "staged bridge rolling materialization differs from its binding"
            )
    stage_store = _StagedBridgeCheckpointStore(checkpoint_root)
    checkpoint_path = stage_store.initialize(common)
    operational_store = BridgeCheckpointStore(
        Path(checkpoint_root) / ".staged-bridge-operational"
    )
    operational_store.initialize(legacy_binding)
    session = object.__new__(CanonicalStagedBridgeSession)
    for name, value in (
        ("binding", common),
        ("checkpoint_path", checkpoint_path),
        ("_request", request),
        ("_legacy_binding", legacy_binding),
        ("_rolling_materializations", materializations),
        ("_stage_store", stage_store),
        ("_operational_store", operational_store),
        ("_seal", _STAGED_SESSION_CAPABILITY),
    ):
        object.__setattr__(session, name, value)
    _MINTED_STAGED_SESSIONS[id(session)] = session
    reverify_staged_bridge_prefix(session)
    return session


def execute_staged_bridge_resource_benchmark(
    session: CanonicalStagedBridgeSession,
) -> StagedBridgeTaskEvidence:
    """Execute/recover the exact resource task, and no downstream work."""

    prefix = reverify_staged_bridge_prefix(session)
    _require_next_action(prefix, "bridge_resource_preflight")
    binding = session._legacy_binding
    receipt = session._operational_store.benchmark_receipt(binding)
    if receipt is None:
        receipt = _execute_benchmark(
            session._request,
            binding,
            session._operational_store,
        )
    benchmark = _benchmark_from_receipt(receipt, binding)
    return _write_staged_task_event(
        session,
        task_id="bridge_resource_preflight",
        dgp=None,
        passed=benchmark.passed,
        parent_task_fingerprints=(),
        source_materialization_fingerprint=_staged_resource_source_fingerprint(session),
        rng_execution_fingerprint=_staged_resource_rng_fingerprint(session),
        result_fingerprint=_clean_result_fingerprint(
            "bridge_resource_preflight",
            benchmark,
        ),
    )


def execute_staged_bridge_tier_a(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
) -> StagedBridgeTaskEvidence:
    """Execute one Tier-A DGP without binding the other Tier-A outcome."""

    checked_dgp = _dgp(dgp)
    task_id = _tier_a_task_id(checked_dgp)
    prefix = reverify_staged_bridge_prefix(session)
    _require_next_action(prefix, task_id)
    resource = _task_from_prefix(prefix, "bridge_resource_preflight")
    receipt = _execute_or_recover_staged_tier_a(session, checked_dgp)
    result = _tier_a_result_from_receipt(
        receipt,
        session._legacy_binding,
        checked_dgp,
    )
    materialization = _rolling_materialization(session, checked_dgp)
    return _write_staged_task_event(
        session,
        task_id=task_id,
        dgp=checked_dgp,
        passed=result.passed,
        parent_task_fingerprints=((resource.task_id, resource.evidence_fingerprint),),
        source_materialization_fingerprint=_staged_tier_a_source_fingerprint(
            session,
            checked_dgp,
            resource.evidence_fingerprint,
            materialization.materialization_fingerprint,
        ),
        rng_execution_fingerprint=_staged_tier_a_rng_fingerprint(
            session,
            checked_dgp,
            materialization,
        ),
        result_fingerprint=_clean_tier_a_result_fingerprint(receipt),
    )


def execute_staged_bridge_tier_b_next_look(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
) -> StagedBridgePlannedLookEvidence:
    """Execute/recover exactly one next registered Tier-B planned look."""

    checked_dgp = _dgp(dgp)
    prefix = reverify_staged_bridge_prefix(session)
    task_id = _tier_b_task_id(checked_dgp)
    existing = _looks_from_prefix(prefix, checked_dgp)
    expected = f"{task_id}:look:{len(existing) + 1}"
    _require_next_action(prefix, expected)
    resource, tier_a_model, tier_a_nonparametric = _tier_b_parents(prefix)
    _execute_or_recover_staged_tier_b_look(
        session,
        checked_dgp,
        expected_look_index=len(existing) + 1,
    )
    result = session._operational_store.verified_tier_b_look_results(
        session._legacy_binding,
        checked_dgp,
    )[-1]
    parents = (
        (resource.task_id, resource.evidence_fingerprint),
        (tier_a_model.task_id, tier_a_model.evidence_fingerprint),
        (tier_a_nonparametric.task_id, tier_a_nonparametric.evidence_fingerprint),
    )
    actual_statistic_fingerprint = _actual_statistic_input_fingerprint(session._request)
    source = _staged_tier_b_source_fingerprint(
        session,
        checked_dgp,
        parents,
        actual_statistic_fingerprint,
    )
    rng = _staged_tier_b_rng_fingerprint(
        session,
        checked_dgp,
        result.sample_count,
        _accelerator_for_dgp(session, checked_dgp).retained_mode,
        resource.rng_execution_fingerprint,
    )
    return _write_staged_look_event(
        session,
        task_id=task_id,
        dgp=checked_dgp,
        look=result,
        parent_task_fingerprints=parents,
        source_materialization_fingerprint=source,
        rng_execution_fingerprint=rng,
        result_fingerprint=_clean_tier_b_result_fingerprint(
            session,
            checked_dgp,
        ),
    )


def complete_staged_bridge_tier_b_task(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
) -> StagedBridgeTaskEvidence:
    """Mint the Tier-B task result only after its latest look is terminal."""

    checked_dgp = _dgp(dgp)
    task_id = _tier_b_task_id(checked_dgp)
    prefix = reverify_staged_bridge_prefix(session)
    _require_next_action(prefix, f"{task_id}:complete")
    looks = _looks_from_prefix(prefix, checked_dgp)
    if not looks or looks[-1].decision == "continue":
        raise ModelError("Tier-B task cannot complete before a terminal planned look")
    terminal = looks[-1]
    return _write_staged_task_event(
        session,
        task_id=task_id,
        dgp=checked_dgp,
        passed=terminal.decision == "pass_above_tail",
        parent_task_fingerprints=terminal.parent_task_fingerprints,
        source_materialization_fingerprint=(
            terminal.source_materialization_fingerprint
        ),
        rng_execution_fingerprint=terminal.rng_execution_fingerprint,
        result_fingerprint=terminal.result_fingerprint,
    )


def execute_staged_actual_artifact_bridge(
    session: CanonicalStagedBridgeSession,
) -> StagedBridgeTaskEvidence:
    """Mint the RNG-empty actual-artifact task after both Tier-B passes."""

    prefix = reverify_staged_bridge_prefix(session)
    _require_next_action(prefix, "actual_artifact_bridge")
    model = _task_from_prefix(prefix, "bridge_tier_b_model_dgp")
    nonparametric = _task_from_prefix(
        prefix,
        "bridge_tier_b_nonparametric_dgp",
    )
    tier_a_model = _task_from_prefix(prefix, "bridge_tier_a_model_dgp")
    tier_a_nonparametric = _task_from_prefix(
        prefix,
        "bridge_tier_a_nonparametric_dgp",
    )
    if not model.passed or not nonparametric.passed:
        raise ModelError("actual bridge requires both Tier-B tasks to pass")
    _, actual_fingerprint = _actual_bridge_binding(
        session._request.actual_bridge,
        session._request.design_parameters,
        session._request.alpha_offset_grids,
        session.binding.selected_n_states,
    )
    parents = (
        (tier_a_model.task_id, tier_a_model.evidence_fingerprint),
        (
            tier_a_nonparametric.task_id,
            tier_a_nonparametric.evidence_fingerprint,
        ),
        (model.task_id, model.evidence_fingerprint),
        (nonparametric.task_id, nonparametric.evidence_fingerprint),
    )
    source = _sha256_json(
        {
            "schema_id": "prpg-staged-actual-bridge-source-v1",
            "common_binding_fingerprint": session.binding.fingerprint,
            "parent_task_fingerprints": dict(parents),
            "actual_bridge_evidence_fingerprint": actual_fingerprint,
            "alpha_offset_grids": [
                item.as_dict()
                for item in _bound_alpha_grids(
                    session._request.alpha_offset_grids,
                    session.binding.selected_n_states,
                )
            ],
        }
    )
    return _write_staged_task_event(
        session,
        task_id="actual_artifact_bridge",
        dgp=None,
        passed=True,
        parent_task_fingerprints=parents,
        source_materialization_fingerprint=source,
        rng_execution_fingerprint=None,
        result_fingerprint=actual_fingerprint,
    )


def reverify_staged_bridge_prefix(
    session: CanonicalStagedBridgeSession,
) -> ReverifiedStagedBridgePrefix:
    """Reconstruct and revalidate the exact append-only staged prefix."""

    _verify_staged_session(session)
    receipts = session._stage_store.event_receipts(session.binding)
    tasks: list[StagedBridgeTaskEvidence] = []
    looks: list[StagedBridgePlannedLookEvidence] = []
    previous: str | None = None
    for index, receipt in enumerate(receipts):
        identity = receipt.identity
        _exact_keys(
            identity,
            {
                "schema_version",
                "kind",
                "event_index",
                "common_binding_fingerprint",
                "previous_receipt_fingerprint",
                "evidence",
            },
            "staged bridge event",
        )
        if (
            identity.get("schema_version") != STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION
            or identity.get("event_index") != index
            or identity.get("common_binding_fingerprint") != session.binding.fingerprint
            or identity.get("previous_receipt_fingerprint") != previous
        ):
            raise IntegrityError("staged bridge event prefix identity is inconsistent")
        expected = _next_staged_action(tuple(tasks), tuple(looks))
        kind = identity.get("kind")
        if kind == "staged_bridge_task":
            task_evidence = _mint_staged_task_from_receipt(session, receipt, index)
            action: str = task_evidence.task_id
            if task_evidence.task_id in {
                "bridge_tier_b_model_dgp",
                "bridge_tier_b_nonparametric_dgp",
            }:
                action = f"{task_evidence.task_id}:complete"
            if action != expected:
                raise IntegrityError(
                    "staged bridge task is skipped, repeated, or reordered"
                )
            _validate_live_staged_task(
                session,
                task_evidence,
                tuple(tasks),
                tuple(looks),
            )
            tasks.append(task_evidence)
        elif kind == "staged_bridge_planned_look":
            look_evidence = _mint_staged_look_from_receipt(session, receipt, index)
            look_action = f"{look_evidence.task_id}:look:{look_evidence.look_index}"
            if look_action != expected:
                raise IntegrityError(
                    "staged bridge planned look is skipped, repeated, or reordered"
                )
            _validate_live_staged_look(
                session,
                look_evidence,
                tuple(tasks),
                tuple(looks),
            )
            looks.append(look_evidence)
        else:
            raise IntegrityError("staged bridge event kind is not registered")
        previous = receipt.fingerprint
    next_action = _next_staged_action(tuple(tasks), tuple(looks))
    return ReverifiedStagedBridgePrefix(
        session.binding.fingerprint,
        tuple(tasks),
        tuple(looks),
        tuple(item.fingerprint for item in receipts),
        next_action,
        next_action is None,
    )


def staged_bridge_task_evidence_identity(
    value: StagedBridgeTaskEvidence,
) -> StagedBridgeTaskEvidenceIdentity:
    """Reverify one object-specific task result against its exact live prefix."""

    if (
        not isinstance(value, StagedBridgeTaskEvidence)
        or value._seal is not _STAGED_TASK_CAPABILITY
        or _MINTED_STAGED_TASKS.get(id(value)) is not value
    ):
        raise ModelError("staged bridge task evidence lacks producer authority")
    prefix = reverify_staged_bridge_prefix(value._session)
    matching = [
        item for item in prefix.task_evidence if item._event_index == value._event_index
    ]
    if len(matching) != 1 or matching[0].stage_receipt_fingerprint != (
        value.stage_receipt_fingerprint
    ):
        raise IntegrityError("staged bridge task evidence left its exact prefix")
    expected = _task_identity_from_evidence(matching[0])
    actual = _task_identity_from_evidence(value)
    if actual != expected:
        raise IntegrityError("staged bridge task evidence content changed")
    return actual


def is_staged_bridge_task_evidence(
    value: object,
) -> TypeGuard[StagedBridgeTaskEvidence]:
    if not isinstance(value, StagedBridgeTaskEvidence):
        return False
    try:
        staged_bridge_task_evidence_identity(value)
    except (IntegrityError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def staged_bridge_planned_look_identity(
    value: StagedBridgePlannedLookEvidence,
) -> StagedBridgePlannedLookIdentity:
    """Reverify one object-specific planned look against the durable prefix."""

    if (
        not isinstance(value, StagedBridgePlannedLookEvidence)
        or value._seal is not _STAGED_LOOK_CAPABILITY
        or _MINTED_STAGED_LOOKS.get(id(value)) is not value
    ):
        raise ModelError("staged bridge planned look lacks producer authority")
    prefix = reverify_staged_bridge_prefix(value._session)
    matching = [
        item for item in prefix.planned_looks if item._event_index == value._event_index
    ]
    if len(matching) != 1 or matching[0].stage_receipt_fingerprint != (
        value.stage_receipt_fingerprint
    ):
        raise IntegrityError("staged bridge planned look left its exact prefix")
    expected = _look_identity_from_evidence(matching[0])
    actual = _look_identity_from_evidence(value)
    if actual != expected:
        raise IntegrityError("staged bridge planned-look content changed")
    return actual


def is_staged_bridge_planned_look_evidence(
    value: object,
) -> TypeGuard[StagedBridgePlannedLookEvidence]:
    if not isinstance(value, StagedBridgePlannedLookEvidence):
        return False
    try:
        staged_bridge_planned_look_identity(value)
    except (IntegrityError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def staged_bridge_task_planned_looks(
    value: StagedBridgeTaskEvidence,
) -> tuple[StagedBridgePlannedLookEvidence, ...]:
    """Return the exact reverified planned-look prefix owned by one Tier-B task."""

    identity = staged_bridge_task_evidence_identity(value)
    if identity.task_id not in {
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    }:
        return ()
    prefix = reverify_staged_bridge_prefix(value._session)
    looks = tuple(
        item
        for item in prefix.planned_looks
        if item.task_id == identity.task_id and item._event_index < value._event_index
    )
    if (
        not looks
        or looks[-1].decision == "continue"
        or tuple(item.look_index for item in looks) != tuple(range(1, len(looks) + 1))
    ):
        raise IntegrityError("staged Tier-B task lacks its exact terminal look prefix")
    return looks


def _write_staged_task_event(
    session: CanonicalStagedBridgeSession,
    *,
    task_id: StagedBridgeTaskId,
    dgp: BridgeDGP | None,
    passed: bool,
    parent_task_fingerprints: tuple[tuple[str, str], ...],
    source_materialization_fingerprint: str,
    rng_execution_fingerprint: str | None,
    result_fingerprint: str,
) -> StagedBridgeTaskEvidence:
    event_index, previous = _next_stage_coordinates(session)
    payload = _staged_task_payload(
        task_id=task_id,
        dgp=dgp,
        passed=passed,
        common_binding_fingerprint=session.binding.fingerprint,
        parent_task_fingerprints=parent_task_fingerprints,
        source_materialization_fingerprint=source_materialization_fingerprint,
        rng_execution_fingerprint=rng_execution_fingerprint,
        result_fingerprint=result_fingerprint,
    )
    evidence_fingerprint = _sha256_json(payload)
    receipt = session._stage_store.write_event(
        session.binding,
        {
            "schema_version": STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "staged_bridge_task",
            "event_index": event_index,
            "common_binding_fingerprint": session.binding.fingerprint,
            "previous_receipt_fingerprint": previous,
            "evidence": {
                **payload,
                "evidence_fingerprint": evidence_fingerprint,
            },
        },
    )
    return _mint_staged_task_from_receipt(session, receipt, event_index)


def _write_staged_look_event(
    session: CanonicalStagedBridgeSession,
    *,
    task_id: Literal["bridge_tier_b_model_dgp", "bridge_tier_b_nonparametric_dgp"],
    dgp: BridgeDGP,
    look: BridgeTierBLookResult,
    parent_task_fingerprints: tuple[tuple[str, str], ...],
    source_materialization_fingerprint: str,
    rng_execution_fingerprint: str,
    result_fingerprint: str,
) -> StagedBridgePlannedLookEvidence:
    event_index, previous = _next_stage_coordinates(session)
    payload = _staged_look_payload(
        task_id=task_id,
        dgp=dgp,
        look_index=look.look_index,
        sample_count=look.sample_count,
        exceedances=look.exceedances,
        decision=look.decision,
        common_binding_fingerprint=session.binding.fingerprint,
        parent_task_fingerprints=parent_task_fingerprints,
        source_materialization_fingerprint=source_materialization_fingerprint,
        rng_execution_fingerprint=rng_execution_fingerprint,
        result_fingerprint=result_fingerprint,
    )
    evidence_fingerprint = _sha256_json(payload)
    receipt = session._stage_store.write_event(
        session.binding,
        {
            "schema_version": STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "staged_bridge_planned_look",
            "event_index": event_index,
            "common_binding_fingerprint": session.binding.fingerprint,
            "previous_receipt_fingerprint": previous,
            "evidence": {
                **payload,
                "evidence_fingerprint": evidence_fingerprint,
            },
        },
    )
    return _mint_staged_look_from_receipt(session, receipt, event_index)


def _mint_staged_task_from_receipt(
    session: CanonicalStagedBridgeSession,
    receipt: _Receipt,
    event_index: int,
) -> StagedBridgeTaskEvidence:
    evidence = _mapping(receipt.identity.get("evidence"), "staged task evidence")
    _exact_keys(
        evidence,
        {
            "schema_id",
            "task_id",
            "dgp",
            "passed",
            "common_binding_fingerprint",
            "parent_task_fingerprints",
            "source_materialization_fingerprint",
            "rng_execution_fingerprint",
            "result_fingerprint",
            "evidence_fingerprint",
        },
        "staged task evidence",
    )
    if evidence.get("schema_id") != "prpg-staged-bridge-task-evidence-v1":
        raise IntegrityError("staged task evidence schema is not registered")
    task_id = _staged_task_id(evidence.get("task_id"))
    raw_dgp = evidence.get("dgp")
    dgp = None if raw_dgp is None else _dgp(raw_dgp)
    expected_dgp: BridgeDGP | None
    if task_id in {"bridge_resource_preflight", "actual_artifact_bridge"}:
        expected_dgp = None
    elif task_id in {"bridge_tier_a_model_dgp", "bridge_tier_b_model_dgp"}:
        expected_dgp = "model"
    else:
        expected_dgp = "nonparametric"
    if dgp != expected_dgp:
        raise IntegrityError("staged task ID/DGP identity is inconsistent")
    passed = _boolean(evidence.get("passed"), "staged task passed")
    common = _string(
        evidence.get("common_binding_fingerprint"),
        "staged task common binding",
    )
    parents = _parent_fingerprints(evidence.get("parent_task_fingerprints"))
    source = _string(
        evidence.get("source_materialization_fingerprint"),
        "staged task source materialization",
    )
    raw_rng = evidence.get("rng_execution_fingerprint")
    rng = None if raw_rng is None else _string(raw_rng, "staged task RNG")
    result = _string(evidence.get("result_fingerprint"), "staged task result")
    fingerprint = _string(
        evidence.get("evidence_fingerprint"),
        "staged task evidence fingerprint",
    )
    for label, digest in (
        ("common binding", common),
        ("source materialization", source),
        ("result", result),
        ("evidence", fingerprint),
    ):
        _require_sha256(digest, f"staged task {label}", IntegrityError)
    if rng is not None:
        _require_sha256(rng, "staged task RNG", IntegrityError)
    payload = _staged_task_payload(
        task_id=task_id,
        dgp=dgp,
        passed=passed,
        common_binding_fingerprint=common,
        parent_task_fingerprints=parents,
        source_materialization_fingerprint=source,
        rng_execution_fingerprint=rng,
        result_fingerprint=result,
    )
    if _sha256_json(payload) != fingerprint:
        raise IntegrityError("staged task evidence fingerprint is invalid")
    minted = object.__new__(StagedBridgeTaskEvidence)
    for name, attribute in (
        ("task_id", task_id),
        ("dgp", dgp),
        ("passed", passed),
        ("common_binding_fingerprint", common),
        ("parent_task_fingerprints", parents),
        ("source_materialization_fingerprint", source),
        ("rng_execution_fingerprint", rng),
        ("result_fingerprint", result),
        ("evidence_fingerprint", fingerprint),
        ("stage_receipt_fingerprint", receipt.fingerprint),
        ("_session", session),
        ("_event_index", event_index),
        ("_seal", _STAGED_TASK_CAPABILITY),
    ):
        object.__setattr__(minted, name, attribute)
    _MINTED_STAGED_TASKS[id(minted)] = minted
    return minted


def _mint_staged_look_from_receipt(
    session: CanonicalStagedBridgeSession,
    receipt: _Receipt,
    event_index: int,
) -> StagedBridgePlannedLookEvidence:
    evidence = _mapping(receipt.identity.get("evidence"), "staged look evidence")
    _exact_keys(
        evidence,
        {
            "schema_id",
            "task_id",
            "dgp",
            "look_index",
            "sample_count",
            "exceedances",
            "decision",
            "common_binding_fingerprint",
            "parent_task_fingerprints",
            "source_materialization_fingerprint",
            "rng_execution_fingerprint",
            "result_fingerprint",
            "evidence_fingerprint",
        },
        "staged look evidence",
    )
    if evidence.get("schema_id") != "prpg-staged-bridge-planned-look-v1":
        raise IntegrityError("staged look evidence schema is not registered")
    task_id = _tier_b_task_id(_dgp(evidence.get("dgp")))
    if evidence.get("task_id") != task_id:
        raise IntegrityError("staged look task/DGP identity is inconsistent")
    dgp = _dgp(evidence.get("dgp"))
    look_index = _positive_integer(evidence.get("look_index"), "look index")
    sample_count = _positive_integer(evidence.get("sample_count"), "sample count")
    exceedances = _nonnegative_integer(evidence.get("exceedances"), "exceedances")
    decision = _bridge_decision(evidence.get("decision"))
    common = _string(evidence.get("common_binding_fingerprint"), "look binding")
    parents = _parent_fingerprints(evidence.get("parent_task_fingerprints"))
    source = _string(evidence.get("source_materialization_fingerprint"), "look source")
    rng = _string(evidence.get("rng_execution_fingerprint"), "look RNG")
    result = _string(evidence.get("result_fingerprint"), "look result")
    fingerprint = _string(evidence.get("evidence_fingerprint"), "look evidence")
    for label, item in (
        ("common binding", common),
        ("source", source),
        ("RNG", rng),
        ("result", result),
        ("evidence", fingerprint),
    ):
        _require_sha256(item, f"staged look {label}", IntegrityError)
    payload = _staged_look_payload(
        task_id=task_id,
        dgp=dgp,
        look_index=look_index,
        sample_count=sample_count,
        exceedances=exceedances,
        decision=decision,
        common_binding_fingerprint=common,
        parent_task_fingerprints=parents,
        source_materialization_fingerprint=source,
        rng_execution_fingerprint=rng,
        result_fingerprint=result,
    )
    if _sha256_json(payload) != fingerprint:
        raise IntegrityError("staged look evidence fingerprint is invalid")
    minted = object.__new__(StagedBridgePlannedLookEvidence)
    for name, attribute in (
        ("task_id", task_id),
        ("dgp", dgp),
        ("look_index", look_index),
        ("sample_count", sample_count),
        ("exceedances", exceedances),
        ("decision", decision),
        ("common_binding_fingerprint", common),
        ("parent_task_fingerprints", parents),
        ("source_materialization_fingerprint", source),
        ("rng_execution_fingerprint", rng),
        ("result_fingerprint", result),
        ("evidence_fingerprint", fingerprint),
        ("stage_receipt_fingerprint", receipt.fingerprint),
        ("_session", session),
        ("_event_index", event_index),
        ("_seal", _STAGED_LOOK_CAPABILITY),
    ):
        object.__setattr__(minted, name, attribute)
    _MINTED_STAGED_LOOKS[id(minted)] = minted
    return minted


def _staged_task_payload(
    *,
    task_id: StagedBridgeTaskId,
    dgp: BridgeDGP | None,
    passed: bool,
    common_binding_fingerprint: str,
    parent_task_fingerprints: tuple[tuple[str, str], ...],
    source_materialization_fingerprint: str,
    rng_execution_fingerprint: str | None,
    result_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_id": "prpg-staged-bridge-task-evidence-v1",
        "task_id": task_id,
        "dgp": dgp,
        "passed": passed,
        "common_binding_fingerprint": common_binding_fingerprint,
        "parent_task_fingerprints": [
            {"task_id": parent, "evidence_fingerprint": fingerprint}
            for parent, fingerprint in parent_task_fingerprints
        ],
        "source_materialization_fingerprint": source_materialization_fingerprint,
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "result_fingerprint": result_fingerprint,
    }


def _staged_look_payload(
    *,
    task_id: Literal["bridge_tier_b_model_dgp", "bridge_tier_b_nonparametric_dgp"],
    dgp: BridgeDGP,
    look_index: int,
    sample_count: int,
    exceedances: int,
    decision: BridgeDecision,
    common_binding_fingerprint: str,
    parent_task_fingerprints: tuple[tuple[str, str], ...],
    source_materialization_fingerprint: str,
    rng_execution_fingerprint: str,
    result_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_id": "prpg-staged-bridge-planned-look-v1",
        "task_id": task_id,
        "dgp": dgp,
        "look_index": look_index,
        "sample_count": sample_count,
        "exceedances": exceedances,
        "decision": decision,
        "common_binding_fingerprint": common_binding_fingerprint,
        "parent_task_fingerprints": [
            {"task_id": parent, "evidence_fingerprint": fingerprint}
            for parent, fingerprint in parent_task_fingerprints
        ],
        "source_materialization_fingerprint": source_materialization_fingerprint,
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "result_fingerprint": result_fingerprint,
    }


def _verify_staged_session(session: object) -> None:
    if (
        not isinstance(session, CanonicalStagedBridgeSession)
        or session._seal is not _STAGED_SESSION_CAPABILITY
        or _MINTED_STAGED_SESSIONS.get(id(session)) is not session
    ):
        raise ModelError("staged bridge session lacks producer authority")
    expected = _staged_common_binding(session._request, session._legacy_binding)
    if session.binding != expected:
        raise IntegrityError("staged bridge request/common binding changed")
    if Path(session.checkpoint_path) != session._stage_store.verify_binding(
        session.binding
    ):
        raise IntegrityError("staged bridge checkpoint path changed")
    session._operational_store.verify_binding(session._legacy_binding)
    if len(session._rolling_materializations) != 2:
        raise IntegrityError("staged bridge lost a rolling materialization")
    for dgp, materialization in zip(
        BRIDGE_DGPS,
        session._rolling_materializations,
        strict=True,
    ):
        if (
            not is_registered_rolling_bootstrap_materialization(materialization)
            or materialization.master_seed != session.binding.master_seed
            or materialization.scientific_version != session.binding.scientific_version
            or materialization.artifact is not _bridge_artifact(dgp)
        ):
            raise IntegrityError("staged bridge rolling materialization changed")
    actual_statistic, actual_fingerprint = _actual_bridge_binding(
        session._request.actual_bridge,
        session._request.design_parameters,
        session._request.alpha_offset_grids,
        session.binding.selected_n_states,
    )
    provisional_legacy = BridgeScientificBinding(
        session.binding.base_preflight_fingerprint,
        session.binding.master_seed,
        session.binding.scientific_version,
        session.binding.selected_n_states,
        session.binding.design_parameters_sha256,
        session.binding.design_history_sha256,
        session.binding.macro_block_length,
        actual_fingerprint,
        actual_statistic,
        _bound_alpha_grids(
            session._request.alpha_offset_grids,
            session.binding.selected_n_states,
        ),
        tuple(
            (
                dgp,
                _purpose1_trace_sha256(materialization.indices),
            )
            for dgp, materialization in zip(
                BRIDGE_DGPS,
                session._rolling_materializations,
                strict=True,
            )
        ),
        "",
    )
    expected_legacy = replace(
        provisional_legacy,
        fingerprint=_sha256_json(provisional_legacy.identity_dict()),
    )
    if session._legacy_binding != expected_legacy:
        raise IntegrityError("staged bridge full request binding changed")


def _staged_common_binding(
    request: CanonicalBridgeRequest,
    legacy: BridgeScientificBinding,
) -> StagedBridgeCommonBinding:
    if not isinstance(request, CanonicalBridgeRequest):
        raise ModelError("staged bridge request has the wrong type")
    base = request.base_preflight
    if (
        not isinstance(base, CanonicalG3Preflight)
        or _sha256_json(base.identity_dict()) != base.fingerprint
        or base.workers != CANONICAL_WORKERS
    ):
        raise ModelError("staged bridge base preflight is inconsistent")
    selected = int(np.asarray(request.design_parameters.means).shape[0])
    values = np.asarray(request.design_standardized_history, dtype=np.float64)
    continuation = np.asarray(request.design_continuation)
    parameters = bridge_parameters_sha256(request.design_parameters)
    history = _design_history_sha256(values, continuation)
    provisional = StagedBridgeCommonBinding(
        base.fingerprint,
        request.master_seed,
        request.scientific_version,
        selected,
        parameters,
        history,
        request.macro_block_length,
        "",
    )
    fingerprint = _sha256_json(provisional.identity_dict())
    result = StagedBridgeCommonBinding(
        provisional.base_preflight_fingerprint,
        provisional.master_seed,
        provisional.scientific_version,
        provisional.selected_n_states,
        provisional.design_parameters_sha256,
        provisional.design_history_sha256,
        provisional.macro_block_length,
        fingerprint,
    )
    if (
        legacy.base_preflight_fingerprint != result.base_preflight_fingerprint
        or legacy.master_seed != result.master_seed
        or legacy.scientific_version != result.scientific_version
        or legacy.selected_n_states != result.selected_n_states
        or legacy.design_parameters_sha256 != result.design_parameters_sha256
        or legacy.design_history_sha256 != result.design_history_sha256
        or legacy.macro_block_length != result.macro_block_length
    ):
        raise IntegrityError("staged and operational bridge bindings disagree")
    _validate_staged_common_binding(result)
    return result


def _validate_staged_common_binding(binding: object) -> None:
    if not isinstance(binding, StagedBridgeCommonBinding):
        raise ModelError("staged bridge common binding has the wrong type")
    for label, value in (
        ("base preflight", binding.base_preflight_fingerprint),
        ("design parameters", binding.design_parameters_sha256),
        ("design history", binding.design_history_sha256),
        ("binding", binding.fingerprint),
    ):
        _require_sha256(value, f"staged bridge {label}", ModelError)
    if (
        binding.selected_n_states not in {2, 3, 4, 5}
        or _sha256_json(binding.identity_dict()) != binding.fingerprint
    ):
        raise ModelError("staged bridge common binding is inconsistent")


def _next_staged_action(
    tasks: tuple[StagedBridgeTaskEvidence, ...],
    looks: tuple[StagedBridgePlannedLookEvidence, ...],
) -> str | None:
    by_task = {item.task_id: item for item in tasks}
    resource = by_task.get("bridge_resource_preflight")
    if resource is None:
        return "bridge_resource_preflight"
    if not resource.passed:
        return None
    model_a = by_task.get("bridge_tier_a_model_dgp")
    if model_a is None:
        return "bridge_tier_a_model_dgp"
    nonparam_a = by_task.get("bridge_tier_a_nonparametric_dgp")
    if nonparam_a is None:
        return "bridge_tier_a_nonparametric_dgp"
    if not model_a.passed or not nonparam_a.passed:
        return None
    for dgp in BRIDGE_DGPS:
        task_id = _tier_b_task_id(dgp)
        completed = by_task.get(task_id)
        if completed is not None:
            continue
        local = tuple(item for item in looks if item.dgp == dgp)
        if not local or local[-1].decision == "continue":
            return f"{task_id}:look:{len(local) + 1}"
        return f"{task_id}:complete"
    model_b = by_task["bridge_tier_b_model_dgp"]
    nonparam_b = by_task["bridge_tier_b_nonparametric_dgp"]
    if not model_b.passed or not nonparam_b.passed:
        return None
    if "actual_artifact_bridge" not in by_task:
        return "actual_artifact_bridge"
    return None


def _validate_live_staged_task(
    session: CanonicalStagedBridgeSession,
    evidence: StagedBridgeTaskEvidence,
    tasks: tuple[StagedBridgeTaskEvidence, ...],
    looks: tuple[StagedBridgePlannedLookEvidence, ...],
) -> None:
    expected_parents: tuple[tuple[str, str], ...]
    expected_source: str
    expected_rng: str | None
    expected_result: str
    expected_passed: bool
    if evidence.task_id == "bridge_resource_preflight":
        receipt = session._operational_store.benchmark_receipt(session._legacy_binding)
        if receipt is None:
            raise IntegrityError("staged resource task lost its operational receipt")
        benchmark = _benchmark_from_receipt(receipt, session._legacy_binding)
        expected_parents = ()
        expected_source = _staged_resource_source_fingerprint(session)
        expected_rng = _staged_resource_rng_fingerprint(session)
        expected_result = _clean_result_fingerprint(
            "bridge_resource_preflight", benchmark
        )
        expected_passed = benchmark.passed
    elif evidence.task_id in {
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
    }:
        dgp = _task_dgp(evidence)
        resource = _task_by_id(tasks, "bridge_resource_preflight")
        aggregate = session._operational_store.tier_a_receipt(
            session._legacy_binding, dgp
        )
        if aggregate is None:
            raise IntegrityError("staged Tier-A task lost its aggregate receipt")
        result = _tier_a_result_from_receipt(aggregate, session._legacy_binding, dgp)
        materialization = _rolling_materialization(session, dgp)
        expected_parents = ((resource.task_id, resource.evidence_fingerprint),)
        expected_source = _staged_tier_a_source_fingerprint(
            session,
            dgp,
            resource.evidence_fingerprint,
            materialization.materialization_fingerprint,
        )
        expected_rng = _staged_tier_a_rng_fingerprint(session, dgp, materialization)
        expected_result = _clean_tier_a_result_fingerprint(aggregate)
        expected_passed = result.passed
    elif evidence.task_id in {
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    }:
        dgp = _task_dgp(evidence)
        local = tuple(item for item in looks if item.dgp == dgp)
        if not local or local[-1].decision == "continue":
            raise IntegrityError("staged Tier-B completion lacks a terminal look")
        terminal = local[-1]
        expected_parents = terminal.parent_task_fingerprints
        expected_source = terminal.source_materialization_fingerprint
        expected_rng = terminal.rng_execution_fingerprint
        expected_result = terminal.result_fingerprint
        expected_passed = terminal.decision == "pass_above_tail"
    else:
        tier_a_model = _task_by_id(tasks, "bridge_tier_a_model_dgp")
        tier_a_nonparametric = _task_by_id(
            tasks,
            "bridge_tier_a_nonparametric_dgp",
        )
        model = _task_by_id(tasks, "bridge_tier_b_model_dgp")
        nonparametric = _task_by_id(tasks, "bridge_tier_b_nonparametric_dgp")
        _, actual = _actual_bridge_binding(
            session._request.actual_bridge,
            session._request.design_parameters,
            session._request.alpha_offset_grids,
            session.binding.selected_n_states,
        )
        expected_parents = (
            (tier_a_model.task_id, tier_a_model.evidence_fingerprint),
            (
                tier_a_nonparametric.task_id,
                tier_a_nonparametric.evidence_fingerprint,
            ),
            (model.task_id, model.evidence_fingerprint),
            (nonparametric.task_id, nonparametric.evidence_fingerprint),
        )
        expected_source = _sha256_json(
            {
                "schema_id": "prpg-staged-actual-bridge-source-v1",
                "common_binding_fingerprint": session.binding.fingerprint,
                "parent_task_fingerprints": dict(expected_parents),
                "actual_bridge_evidence_fingerprint": actual,
                "alpha_offset_grids": [
                    item.as_dict()
                    for item in _bound_alpha_grids(
                        session._request.alpha_offset_grids,
                        session.binding.selected_n_states,
                    )
                ],
            }
        )
        expected_rng = None
        expected_result = actual
        expected_passed = True
    if (
        evidence.common_binding_fingerprint != session.binding.fingerprint
        or evidence.parent_task_fingerprints != expected_parents
        or evidence.source_materialization_fingerprint != expected_source
        or evidence.rng_execution_fingerprint != expected_rng
        or evidence.result_fingerprint != expected_result
        or evidence.passed != expected_passed
    ):
        raise IntegrityError("staged bridge task differs from its live execution")


def _validate_live_staged_look(
    session: CanonicalStagedBridgeSession,
    evidence: StagedBridgePlannedLookEvidence,
    tasks: tuple[StagedBridgeTaskEvidence, ...],
    prior_looks: tuple[StagedBridgePlannedLookEvidence, ...],
) -> None:
    resource = _task_by_id(tasks, "bridge_resource_preflight")
    model_a = _task_by_id(tasks, "bridge_tier_a_model_dgp")
    nonparam_a = _task_by_id(tasks, "bridge_tier_a_nonparametric_dgp")
    parents = (
        (resource.task_id, resource.evidence_fingerprint),
        (model_a.task_id, model_a.evidence_fingerprint),
        (nonparam_a.task_id, nonparam_a.evidence_fingerprint),
    )
    local_prior = tuple(item for item in prior_looks if item.dgp == evidence.dgp)
    operational_looks = session._operational_store.tier_b_look_receipts(
        session._legacy_binding,
        evidence.dgp,
    )
    if evidence.look_index > len(operational_looks):
        raise IntegrityError("staged planned look lost its operational receipt")
    live = session._operational_store.verified_tier_b_look_results(
        session._legacy_binding,
        evidence.dgp,
    )[evidence.look_index - 1]
    source = _staged_tier_b_source_fingerprint(
        session,
        evidence.dgp,
        parents,
        _actual_statistic_input_fingerprint(session._request),
    )
    rng = _staged_tier_b_rng_fingerprint(
        session,
        evidence.dgp,
        live.sample_count,
        _accelerator_for_dgp(session, evidence.dgp).retained_mode,
        resource.rng_execution_fingerprint,
    )
    if (
        evidence.common_binding_fingerprint != session.binding.fingerprint
        or evidence.look_index != len(local_prior) + 1
        or evidence.sample_count != live.sample_count
        or evidence.exceedances != live.exceedances
        or evidence.decision != live.decision
        or evidence.parent_task_fingerprints != parents
        or evidence.source_materialization_fingerprint != source
        or evidence.rng_execution_fingerprint != rng
        or evidence.result_fingerprint
        != _clean_tier_b_result_fingerprint(
            session, evidence.dgp, stop_look=evidence.look_index
        )
    ):
        raise IntegrityError("staged planned look differs from its live execution")


def _execute_or_recover_staged_tier_a(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
) -> _Receipt:
    binding = session._legacy_binding
    store = session._operational_store
    aggregate = store.tier_a_receipt(binding, dgp)
    batches = list(store.tier_a_batch_receipts(binding, dgp))
    evidence = list(_tier_a_pairs_from_batch_receipts(batches, binding, dgp))
    if aggregate is not None:
        expected_batches = TIER_A_PAIRS // BRIDGE_TIER_A_BATCH_PAIRS
        if len(batches) != expected_batches:
            raise IntegrityError("staged Tier-A aggregate lacks its batch prefix")
        return aggregate
    trace = _rolling_materialization(session, dgp).indices
    jobs = _tier_a_jobs(session._request, binding, dgp, trace)
    while len(batches) < len(jobs):
        start = len(batches)
        wave = jobs[start : start + CANONICAL_WORKERS]
        outputs = deterministic_process_map(
            _run_tier_a_batch,
            wave,
            workers=CANONICAL_WORKERS,
            chunksize=1,
        )
        for batch in outputs:
            index = len(batches)
            receipt = store.write_tier_a_batch(
                binding,
                dgp=dgp,
                batch_index=index,
                pairs=batch,
                prior_receipts=batches,
            )
            batches.append(receipt)
            evidence.extend(batch)
    return store.write_tier_a(binding, dgp, evidence)


def _execute_or_recover_staged_tier_b_look(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
    *,
    expected_look_index: int,
) -> _Receipt:
    binding = session._legacy_binding
    store = session._operational_store
    receipts = list(store.tier_b_look_receipts(binding, dgp))
    accelerator = _accelerator_for_dgp(session, dgp)
    _validate_look_chain(receipts, binding, dgp, accelerator)
    if len(receipts) >= expected_look_index:
        if len(receipts) != expected_look_index:
            raise IntegrityError("operational Tier-B work exceeds the staged prefix")
        return receipts[-1]
    if len(receipts) != expected_look_index - 1:
        raise IntegrityError("operational Tier-B work has a sequence gap")
    if receipts and receipts[-1].identity.get("decision") != "continue":
        raise ModelError("no Tier-B work is allowed after a terminal decision")
    look_index = expected_look_index
    start = (look_index - 1) * TIER_B_BATCH_SIZE
    stop = look_index * TIER_B_BATCH_SIZE
    cumulative = _statistics_from_look_receipts(receipts, binding, dgp)
    pairs: list[BridgeTierBPairEvidence] = []
    if look_index == 1:
        pairs.extend(accelerator.retained_pairs)
        work_start = ACCELERATOR_AUDIT_PAIRS
    else:
        work_start = start
    jobs = tuple(
        _fixed_k_job(
            session._request,
            binding,
            dgp,
            pair,
            accelerator.retained_mode,
        )
        for pair in range(work_start, stop)
    )
    pairs.extend(
        deterministic_process_map(
            _run_fixed_k_job,
            jobs,
            workers=CANONICAL_WORKERS,
            chunksize=1,
        )
    )
    checked = tuple(_validated_tier_b_pair(item, binding, dgp) for item in pairs)
    if tuple(item.pair for item in checked) != tuple(range(start, stop)):
        raise ModelError("staged Tier-B look returned the wrong pair order")
    invalid = [
        item.pair for item in checked if not item.valid or item.statistic is None
    ]
    if invalid:
        raise ModelError(
            "staged Tier-B fixed-K pseudo-fit failed closed",
            details={"dgp": dgp, "pairs": invalid[:25]},
        )
    cumulative.extend(
        float(item.statistic) for item in checked if item.statistic is not None
    )
    values = np.asarray(cumulative, dtype=np.float64)
    exceedances = int(np.count_nonzero(values >= binding.actual_statistic))
    interval = clopper_pearson_interval(
        exceedances,
        stop,
        confidence=1.0 - TIER_B_INTERVAL_ERROR,
    )
    decision = _tier_b_decision(interval, stop)
    return store.write_tier_b_look(
        binding,
        dgp=dgp,
        look_index=look_index,
        pairs=checked,
        prior_receipts=receipts,
        cumulative_statistics=cumulative,
        decision=decision,
        interval=interval,
        exceedances=exceedances,
    )


def _staged_resource_source_fingerprint(
    session: CanonicalStagedBridgeSession,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-staged-bridge-resource-source-v1",
            "common_binding_fingerprint": session.binding.fingerprint,
            "base_preflight_fingerprint": (session.binding.base_preflight_fingerprint),
            "selected_n_states": session.binding.selected_n_states,
        }
    )


def _staged_tier_a_source_fingerprint(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
    resource_evidence_fingerprint: str,
    rolling_materialization_fingerprint: str,
) -> str:
    fixed_k4_tier_a = session.binding.scientific_version == CANONICAL_SCIENTIFIC_VERSION
    return _sha256_json(
        {
            "schema_id": (
                "prpg-staged-bridge-tier-a-source-v2"
                if fixed_k4_tier_a
                else "prpg-staged-bridge-tier-a-source-v1"
            ),
            "common_binding_fingerprint": session.binding.fingerprint,
            "dgp": dgp,
            "resource_evidence_fingerprint": resource_evidence_fingerprint,
            "report_only_rolling_materialization_fingerprint": (
                rolling_materialization_fingerprint
            ),
            "tier_a_policy": (
                BRIDGE_FIXED_K4_TIER_A_POLICY
                if fixed_k4_tier_a
                else "cross_k_rolling_v1"
            ),
            "rolling_selection_used": not fixed_k4_tier_a,
        }
    )


def _staged_tier_b_source_fingerprint(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
    parents: tuple[tuple[str, str], ...],
    actual_statistic_input_fingerprint: str,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-staged-bridge-tier-b-source-v1",
            "common_binding_fingerprint": session.binding.fingerprint,
            "dgp": dgp,
            "parent_task_fingerprints": dict(parents),
            "actual_statistic_input_fingerprint": (actual_statistic_input_fingerprint),
        }
    )


def _staged_resource_rng_fingerprint(
    session: CanonicalStagedBridgeSession,
) -> str:
    return rng_execution_fingerprint(
        master_seed=session.binding.master_seed,
        scientific_version=session.binding.scientific_version,
        namespace="bridge_resource_benchmark_composite",
        contract={
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "dgp_order": list(BRIDGE_DGPS),
            "history_contracts": [
                _bridge_history_rng_contract(
                    dgp,
                    tier="fixed_k",
                    pair_start=0,
                    pair_stop=ACCELERATOR_AUDIT_PAIRS,
                )
                for dgp in BRIDGE_DGPS
            ],
            "fit_contracts": [
                _fixed_k_fit_rng_contract(
                    dgp,
                    member,
                    pair_start=0,
                    pair_stop=ACCELERATOR_AUDIT_PAIRS,
                    selected_n_states=session.binding.selected_n_states,
                    mode=mode,
                )
                for dgp in BRIDGE_DGPS
                for member in _BRIDGE_MEMBERS
                for mode in _BRIDGE_FIT_MODES
            ],
        },
    )


def _staged_tier_a_rng_fingerprint(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
    materialization: RegisteredRollingBootstrapMaterialization,
) -> str:
    if session.binding.scientific_version == CANONICAL_SCIENTIFIC_VERSION:
        return rng_execution_fingerprint(
            master_seed=session.binding.master_seed,
            scientific_version=session.binding.scientific_version,
            namespace=f"bridge_tier_a_{dgp}_fixed_k4_composite",
            contract={
                "rng_contract_version": RNG_CONTRACT_VERSION,
                "tier_a_policy": BRIDGE_FIXED_K4_TIER_A_POLICY,
                "dgp": dgp,
                "history_contract": _bridge_history_rng_contract(
                    dgp,
                    tier="selection",
                    pair_start=0,
                    pair_stop=TIER_A_PAIRS,
                ),
                "fixed_k4_fit_contracts": [
                    {
                        "member": member,
                        "scope": int(bridge_selection_scope(dgp, member)),
                        "n_states": BRIDGE_TIER_A_N_STATES,
                        "pair_start": 0,
                        "pair_stop_exclusive": TIER_A_PAIRS,
                        "replicate_formula": (
                            f"pair*{BRIDGE_FITS_PER_SELECTION_MEMBER}"
                        ),
                        "fit_coordinates_per_member": 1,
                        "rolling_fold_ids": [],
                        "restart_ids": list(range(50)),
                        "warm_start_used": False,
                    }
                    for member in _BRIDGE_MEMBERS
                ],
                "rolling_score_bootstrap_used": False,
                "cross_k_selection_used": False,
                "bic_used": False,
            },
        )
    return rng_execution_fingerprint(
        master_seed=session.binding.master_seed,
        scientific_version=session.binding.scientific_version,
        namespace=f"bridge_tier_a_{dgp}_composite",
        contract={
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "dgp": dgp,
            "history_contract": _bridge_history_rng_contract(
                dgp,
                tier="selection",
                pair_start=0,
                pair_stop=TIER_A_PAIRS,
            ),
            "selection_fit_contracts": [
                {
                    "member": member,
                    "scope": int(bridge_selection_scope(dgp, member)),
                    "candidate_n_states": [2, 3, 4, 5],
                    "pair_start": 0,
                    "pair_stop_exclusive": TIER_A_PAIRS,
                    "replicate_formula": (
                        f"pair*{BRIDGE_FITS_PER_SELECTION_MEMBER}+fold"
                    ),
                    "fold_ids": list(range(BRIDGE_FITS_PER_SELECTION_MEMBER)),
                    "restart_ids": list(range(50)),
                }
                for member in _BRIDGE_MEMBERS
            ],
            "rolling_materialization_fingerprint": (
                materialization.materialization_fingerprint
            ),
            "rolling_rng_execution_fingerprint": (
                materialization.rng_execution_fingerprint
            ),
        },
    )


def _staged_tier_b_rng_fingerprint(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
    sample_count: int,
    mode: FitMode,
    resource_rng_fingerprint: str | None,
) -> str:
    _require_sha256(
        resource_rng_fingerprint,
        "Tier-B inherited resource RNG fingerprint",
        ModelError,
    )
    return rng_execution_fingerprint(
        master_seed=session.binding.master_seed,
        scientific_version=session.binding.scientific_version,
        namespace=f"bridge_tier_b_{dgp}_composite",
        contract={
            "rng_contract_version": RNG_CONTRACT_VERSION,
            "dgp": dgp,
            "sample_count": sample_count,
            "retained_mode": mode,
            "inherited_resource_rng_fingerprint": resource_rng_fingerprint,
            "new_history_contract": _bridge_history_rng_contract(
                dgp,
                tier="fixed_k",
                pair_start=ACCELERATOR_AUDIT_PAIRS,
                pair_stop=sample_count,
            ),
            "new_fit_contracts": [
                _fixed_k_fit_rng_contract(
                    dgp,
                    member,
                    pair_start=ACCELERATOR_AUDIT_PAIRS,
                    pair_stop=sample_count,
                    selected_n_states=session.binding.selected_n_states,
                    mode=mode,
                )
                for member in _BRIDGE_MEMBERS
            ],
        },
    )


def _bridge_history_rng_contract(
    dgp: BridgeDGP,
    *,
    tier: Literal["selection", "fixed_k"],
    pair_start: int,
    pair_stop: int,
) -> dict[str, object]:
    model = dgp == "model"
    return {
        "dgp": dgp,
        "artifact": int(_bridge_artifact(dgp)),
        "purpose": int(CalibrationPurpose.BRIDGE_HISTORY),
        "variant": 0 if tier == "selection" else 1,
        "pair_start": pair_start,
        "pair_stop_exclusive": pair_stop,
        "domain": int(Domain.BLOCK_ALPHA_CALIBRATION),
        "family": int(Family.NOT_APPLICABLE),
        "stages": (
            [int(Stage.STATE_CHAIN), int(Stage.OPTIONAL_RESIDUAL_NOISE)]
            if model
            else [int(Stage.REFERENCE_RESAMPLING)]
        ),
        "draws_per_pair": (
            [BRIDGE_FULL_ROWS, BRIDGE_FULL_ROWS * 4]
            if model
            else [2 * BRIDGE_FULL_ROWS]
        ),
    }


def _fixed_k_fit_rng_contract(
    dgp: BridgeDGP,
    member: Literal["prefix", "full"],
    *,
    pair_start: int,
    pair_stop: int,
    selected_n_states: int,
    mode: FitMode,
) -> dict[str, object]:
    return {
        "dgp": dgp,
        "member": member,
        "scope": int(bridge_fixed_k_scope(dgp, member)),
        "pair_start": pair_start,
        "pair_stop_exclusive": pair_stop,
        "n_states": selected_n_states,
        "mode": mode,
        "restart_ids": list(range(9 if mode == "accelerated_10" else 50)),
        "parent_warm_start": mode == "accelerated_10",
    }


def _actual_statistic_input_fingerprint(request: CanonicalBridgeRequest) -> str:
    actual = request.actual_bridge
    identity = actual_bridge_execution_identity(actual)
    statistic = actual.components.statistic
    if (
        actual.n_states != int(np.asarray(request.design_parameters.means).shape[0])
        or not math.isfinite(statistic.maximum)
        or statistic.maximum < 0.0
    ):
        raise ModelError("Tier-B actual-statistic input is inconsistent")
    return _sha256_json(
        {
            "schema_id": "prpg-bridge-actual-statistic-input-v1",
            "actual_bridge_execution_fingerprint": identity.execution_fingerprint,
            "actual_bridge_source_materialization_fingerprint": (
                identity.source_materialization_fingerprint
            ),
            "selected_n_states": actual.n_states,
            "component_names": list(statistic.component_names),
            "raw_maxima_sha256": _array_sha256(statistic.raw_maxima),
            "hard_caps_sha256": _array_sha256(statistic.hard_caps),
            "normalized_components_sha256": _array_sha256(
                statistic.normalized_components
            ),
            "maximum": statistic.maximum,
        }
    )


def _clean_result_fingerprint(label: str, value: object) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-staged-bridge-clean-result-v1",
            "label": label,
            "value": _clean_scientific_value(value),
        }
    )


def _clean_tier_a_result_fingerprint(receipt: _Receipt) -> str:
    return _clean_result_fingerprint(
        "bridge_tier_a_result",
        _clean_scientific_value(dict(receipt.identity)),
    )


def _clean_tier_b_result_fingerprint(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
    *,
    stop_look: int | None = None,
) -> str:
    receipts = session._operational_store.tier_b_look_receipts(
        session._legacy_binding, dgp
    )
    selected = receipts if stop_look is None else receipts[:stop_look]
    return _clean_result_fingerprint(
        f"bridge_tier_b_{dgp}_cumulative_result",
        tuple(_clean_scientific_value(dict(item.identity)) for item in selected),
    )


_CLEAN_BRIDGE_EXCLUDED_FIELDS: Final = frozenset(
    {
        "binding",
        "binding_fingerprint",
        "receipt_fingerprint",
        "previous_receipt_fingerprint",
        "scientific_fingerprint",
        "evidence_fingerprint",
        "final_receipt_fingerprint",
        "checkpoint_path",
    }
)


def _clean_scientific_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
    if isinstance(value, np.generic):
        return _clean_scientific_value(value.item())
    if isinstance(value, Enum):
        return _clean_scientific_value(value.value)
    if isinstance(value, Path):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _clean_scientific_value(getattr(value, item.name))
                for item in fields(value)
                if not item.name.startswith("_")
                and item.name not in _CLEAN_BRIDGE_EXCLUDED_FIELDS
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ModelError("clean bridge result mappings require string keys")
        return {
            key: _clean_scientific_value(value[key])
            for key in sorted(value)
            if key not in _CLEAN_BRIDGE_EXCLUDED_FIELDS
        }
    if isinstance(value, tuple | list):
        return [_clean_scientific_value(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("clean bridge result contains a non-finite scalar")
        return value
    raise ModelError(
        "clean bridge result contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _accelerator_for_dgp(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
) -> BridgeAcceleratorDecision:
    receipt = session._operational_store.benchmark_receipt(session._legacy_binding)
    if receipt is None:
        raise IntegrityError("staged Tier-B requires its resource benchmark")
    benchmark = _benchmark_from_receipt(receipt, session._legacy_binding)
    selected = [item for item in benchmark.accelerators if item.dgp == dgp]
    if len(selected) != 1:
        raise IntegrityError("staged Tier-B accelerator identity is inconsistent")
    return selected[0]


def _rolling_materialization(
    session: CanonicalStagedBridgeSession,
    dgp: BridgeDGP,
) -> RegisteredRollingBootstrapMaterialization:
    return session._rolling_materializations[BRIDGE_DGPS.index(dgp)]


def _bridge_artifact(dgp: BridgeDGP) -> CalibrationArtifact:
    return (
        CalibrationArtifact.BRIDGE_MODEL
        if dgp == "model"
        else CalibrationArtifact.BRIDGE_NONPARAMETRIC
    )


def _tier_a_task_id(dgp: BridgeDGP) -> StagedBridgeTaskId:
    return (
        "bridge_tier_a_model_dgp"
        if dgp == "model"
        else "bridge_tier_a_nonparametric_dgp"
    )


def _tier_b_task_id(
    dgp: BridgeDGP,
) -> Literal["bridge_tier_b_model_dgp", "bridge_tier_b_nonparametric_dgp"]:
    return (
        "bridge_tier_b_model_dgp"
        if dgp == "model"
        else "bridge_tier_b_nonparametric_dgp"
    )


def _next_stage_coordinates(
    session: CanonicalStagedBridgeSession,
) -> tuple[int, str | None]:
    receipts = session._stage_store.event_receipts(session.binding)
    return len(receipts), receipts[-1].fingerprint if receipts else None


def _require_next_action(prefix: ReverifiedStagedBridgePrefix, expected: str) -> None:
    if prefix.next_action != expected:
        raise ModelError(
            "staged bridge action is skipped, repeated, or out of order",
            details={"expected": prefix.next_action, "attempted": expected},
        )


def _task_from_prefix(
    prefix: ReverifiedStagedBridgePrefix,
    task_id: StagedBridgeTaskId,
) -> StagedBridgeTaskEvidence:
    return _task_by_id(prefix.task_evidence, task_id)


def _task_by_id(
    tasks: Sequence[StagedBridgeTaskEvidence],
    task_id: StagedBridgeTaskId,
) -> StagedBridgeTaskEvidence:
    selected = [item for item in tasks if item.task_id == task_id]
    if len(selected) != 1:
        raise IntegrityError(f"staged bridge prefix lacks exact {task_id} evidence")
    return selected[0]


def _looks_from_prefix(
    prefix: ReverifiedStagedBridgePrefix,
    dgp: BridgeDGP,
) -> tuple[StagedBridgePlannedLookEvidence, ...]:
    return tuple(item for item in prefix.planned_looks if item.dgp == dgp)


def _tier_b_parents(
    prefix: ReverifiedStagedBridgePrefix,
) -> tuple[
    StagedBridgeTaskEvidence,
    StagedBridgeTaskEvidence,
    StagedBridgeTaskEvidence,
]:
    return (
        _task_from_prefix(prefix, "bridge_resource_preflight"),
        _task_from_prefix(prefix, "bridge_tier_a_model_dgp"),
        _task_from_prefix(prefix, "bridge_tier_a_nonparametric_dgp"),
    )


def _task_dgp(evidence: StagedBridgeTaskEvidence) -> BridgeDGP:
    if evidence.dgp is None:
        raise IntegrityError("staged DGP task is missing its DGP")
    return evidence.dgp


def _task_identity_from_evidence(
    value: StagedBridgeTaskEvidence,
) -> StagedBridgeTaskEvidenceIdentity:
    return StagedBridgeTaskEvidenceIdentity(
        value.task_id,
        value.dgp,
        value.passed,
        value._session.binding.base_preflight_fingerprint,
        value._session.binding.selected_n_states,
        value.common_binding_fingerprint,
        value.parent_task_fingerprints,
        value.source_materialization_fingerprint,
        value.rng_execution_fingerprint,
        value.result_fingerprint,
        value.evidence_fingerprint,
        value.stage_receipt_fingerprint,
    )


def _look_identity_from_evidence(
    value: StagedBridgePlannedLookEvidence,
) -> StagedBridgePlannedLookIdentity:
    return StagedBridgePlannedLookIdentity(
        value.task_id,
        value.dgp,
        value.look_index,
        value.sample_count,
        value.exceedances,
        value.decision,
        value.common_binding_fingerprint,
        value.parent_task_fingerprints,
        value.source_materialization_fingerprint,
        value.rng_execution_fingerprint,
        value.result_fingerprint,
        value.evidence_fingerprint,
        value.stage_receipt_fingerprint,
    )


def _parent_fingerprints(value: object) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    for item in _sequence(value, "staged parent fingerprints"):
        mapping = _mapping(item, "staged parent fingerprint")
        _exact_keys(
            mapping,
            {"task_id", "evidence_fingerprint"},
            "staged parent fingerprint",
        )
        task_id = _staged_task_id(mapping.get("task_id"))
        fingerprint = _string(
            mapping.get("evidence_fingerprint"),
            "staged parent evidence fingerprint",
        )
        _require_sha256(
            fingerprint,
            "staged parent evidence fingerprint",
            IntegrityError,
        )
        output.append((task_id, fingerprint))
    if len({item[0] for item in output}) != len(output):
        raise IntegrityError("staged parent task fingerprints contain a duplicate")
    return tuple(output)


def _staged_task_id(value: object) -> StagedBridgeTaskId:
    if value not in {
        "bridge_resource_preflight",
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
        "actual_artifact_bridge",
    }:
        raise IntegrityError("staged bridge task ID is not registered")
    return cast(StagedBridgeTaskId, value)


def _bridge_decision(value: object) -> BridgeDecision:
    if value not in {
        "continue",
        "pass_above_tail",
        "fail_below_tail",
        "fail_closed_at_maximum",
    }:
        raise IntegrityError("staged bridge decision is not registered")
    return cast(BridgeDecision, value)


def run_canonical_bridge_driver(
    request: CanonicalBridgeRequest,
    checkpoint_root: str | Path,
) -> BridgeDriverResult:
    """Execute or resume the exact dual-reference bridge without running G5.

    This is intentionally the only public launch surface.  Pair counts,
    worker count, resource bounds, fit modes, and stopping rules are not
    caller arguments.  The first invocation can be expensive; later calls
    reverify and reuse every completed immutable receipt.
    """

    prepared = _prepare_request(request)
    binding, traces = prepared
    store = BridgeCheckpointStore(checkpoint_root)
    checkpoint_path = store.initialize(binding)

    # The registered resource task precedes both Tier-A tasks.  Its measured
    # first-100 fixed-K work is later reused as the first Tier-B evidence, so
    # this is not a disposable pilot or an unregistered RNG draw.
    benchmark_receipt = store.benchmark_receipt(binding)
    if benchmark_receipt is None:
        benchmark_receipt = _execute_benchmark(request, binding, store)
    benchmark = _benchmark_from_receipt(benchmark_receipt, binding)
    if not benchmark.passed:
        return _finish_preflight_failure(
            binding,
            checkpoint_path,
            store,
            benchmark,
        )

    tier_a_results: list[BridgeTierAResult] = []
    for dgp in BRIDGE_DGPS:
        receipt = store.tier_a_receipt(binding, dgp)
        batch_receipts = list(store.tier_a_batch_receipts(binding, dgp))
        evidence = list(_tier_a_pairs_from_batch_receipts(batch_receipts, binding, dgp))
        if receipt is None:
            jobs = _tier_a_jobs(request, binding, dgp, traces[dgp])
            while len(batch_receipts) < len(jobs):
                start = len(batch_receipts)
                wave = jobs[start : start + CANONICAL_WORKERS]
                batches = deterministic_process_map(
                    _run_tier_a_batch,
                    wave,
                    workers=CANONICAL_WORKERS,
                    chunksize=1,
                )
                for batch in batches:
                    batch_index = len(batch_receipts)
                    batch_receipt = store.write_tier_a_batch(
                        binding,
                        dgp=dgp,
                        batch_index=batch_index,
                        pairs=batch,
                        prior_receipts=batch_receipts,
                    )
                    batch_receipts.append(batch_receipt)
                    evidence.extend(batch)
            receipt = store.write_tier_a(binding, dgp, evidence)
        else:
            expected_batches = TIER_A_PAIRS // BRIDGE_TIER_A_BATCH_PAIRS
            if len(batch_receipts) != expected_batches:
                raise IntegrityError(
                    "completed Tier-A receipt lacks its exact durable batch chain"
                )
            final_records = tuple(
                _tier_a_pair_from_dict(item)
                for item in _sequence(receipt.identity.get("pairs"), "Tier-A pairs")
            )
            if tuple(item.as_dict() for item in evidence) != tuple(
                item.as_dict() for item in final_records
            ):
                raise IntegrityError(
                    "Tier-A aggregate receipt differs from its batch checkpoints"
                )
        tier_a_results.append(_tier_a_result_from_receipt(receipt, binding, dgp))

    if not all(item.passed for item in tier_a_results):
        return _finish_without_tier_b(
            binding,
            checkpoint_path,
            store,
            tuple(tier_a_results),
            benchmark=benchmark,
            reasons=("one_or_both_tier_a_gates_failed",),
        )

    dgp_results: list[BridgeDGPResult] = []
    accelerator_by_dgp = {item.dgp: item for item in benchmark.accelerators}
    for tier_a in tier_a_results:
        accelerator = accelerator_by_dgp[tier_a.dgp]
        look_receipts = store.tier_b_look_receipts(binding, tier_a.dgp)
        _validate_look_chain(look_receipts, binding, tier_a.dgp, accelerator)
        if not look_receipts or look_receipts[-1].identity["decision"] == "continue":
            look_receipts = _run_remaining_tier_b_looks(
                request,
                binding,
                store,
                tier_a.dgp,
                accelerator,
                look_receipts,
            )
        dgp_results.append(
            _dgp_result_from_receipts(tier_a, accelerator, look_receipts, binding)
        )

    combined = all(item.passed for item in dgp_results)
    reasons = () if combined else ("one_or_both_tier_b_gates_failed",)
    scientific = _combined_scientific_fingerprint(binding, dgp_results, combined)
    final = store.write_final(
        binding,
        dgp_results=dgp_results,
        bridge_gate_passed=combined,
        reasons=reasons,
        scientific_fingerprint=scientific,
        benchmark_receipt_fingerprint=benchmark.receipt_fingerprint,
    )
    return _verified_bridge_result(
        BridgeDriverResult(
            binding,
            checkpoint_path,
            tuple(dgp_results),
            benchmark,
            combined,
            reasons,
            scientific,
            final.fingerprint,
        )
    )


def reverify_bridge_driver_result(
    result: BridgeDriverResult,
    request: CanonicalBridgeRequest,
) -> ReverifiedBridgeDriverEvidence:
    """Reread and reconstruct one sealed driver's complete durable evidence.

    This adapter performs no scientific work and writes nothing.  It exists
    so the enclosing G3 layer never has to trust caller-built resource, Tier-A,
    Tier-B, or actual-bridge summaries.  The request is prepared again to bind
    the exact launch preflight, selected K, artifacts, alpha grids, purpose-1
    traces, and typed :class:`ActualBridgeExecution` to the stored execution.
    """

    if not is_verified_bridge_driver_result(result):
        raise ModelError("G3 requires a sealed bridge-driver result")
    binding, _ = _prepare_request(request)
    if result.binding != binding:
        raise IntegrityError("bridge driver result differs from its canonical request")

    checkpoint_path = Path(result.checkpoint_path)
    if checkpoint_path.name != binding.fingerprint:
        raise IntegrityError("bridge driver checkpoint path has the wrong binding")
    store = BridgeCheckpointStore(checkpoint_path.parent.parent)
    verified_path = store.verify_binding(binding)
    try:
        same_path = checkpoint_path.samefile(verified_path)
    except OSError as error:
        raise IntegrityError(
            "bridge driver checkpoint path cannot be verified"
        ) from error
    if not same_path:
        raise IntegrityError("bridge driver checkpoint path differs from its binding")

    benchmark_receipt = store.benchmark_receipt(binding)
    if benchmark_receipt is None:
        raise IntegrityError("bridge driver is missing its benchmark receipt")
    benchmark = _benchmark_from_receipt(benchmark_receipt, binding)
    if (
        result.benchmark is None
        or result.benchmark.receipt_fingerprint != benchmark.receipt_fingerprint
    ):
        raise IntegrityError("bridge driver benchmark result differs from checkpoint")

    tier_a_by_dgp: dict[BridgeDGP, BridgeTierAResult | None] = {}
    if benchmark.passed:
        expected_batches = TIER_A_PAIRS // BRIDGE_TIER_A_BATCH_PAIRS
        for dgp in BRIDGE_DGPS:
            batches = store.tier_a_batch_receipts(binding, dgp)
            if len(batches) != expected_batches:
                raise IntegrityError(
                    "bridge driver lacks its exact Tier-A batch checkpoint chain"
                )
            pairs = _tier_a_pairs_from_batch_receipts(batches, binding, dgp)
            aggregate = store.tier_a_receipt(binding, dgp)
            if aggregate is None:
                raise IntegrityError("bridge driver is missing its Tier-A aggregate")
            aggregate_pairs = tuple(
                _tier_a_pair_from_dict(item)
                for item in _sequence(
                    aggregate.identity.get("pairs"), "Tier-A aggregate pairs"
                )
            )
            if tuple(item.as_dict() for item in pairs) != tuple(
                item.as_dict() for item in aggregate_pairs
            ):
                raise IntegrityError(
                    "bridge Tier-A aggregate differs from its exact batch chain"
                )
            tier_a_by_dgp[dgp] = _tier_a_result_from_receipt(aggregate, binding, dgp)
    else:
        for dgp in BRIDGE_DGPS:
            if (
                store.tier_a_receipt(binding, dgp) is not None
                or store.tier_a_batch_receipts(binding, dgp)
                or store.tier_b_look_receipts(binding, dgp)
            ):
                raise IntegrityError(
                    "failed bridge resource preflight has downstream checkpoints"
                )
            tier_a_by_dgp[dgp] = None

    accelerators = {item.dgp: item for item in benchmark.accelerators}
    all_tier_a_passed = benchmark.passed and all(
        item is not None and item.passed for item in tier_a_by_dgp.values()
    )
    reconstructed: list[BridgeDGPResult] = []
    look_families: list[tuple[BridgeTierBLookResult, ...]] = []
    for dgp in BRIDGE_DGPS:
        tier_a = tier_a_by_dgp[dgp]
        receipts = store.tier_b_look_receipts(binding, dgp)
        if all_tier_a_passed:
            if tier_a is None:  # pragma: no cover - established above
                raise AssertionError("passing Tier-A family lacks a result")
            _validate_look_chain(receipts, binding, dgp, accelerators[dgp])
            if not receipts or receipts[-1].identity.get("decision") == "continue":
                raise IntegrityError(
                    "bridge driver Tier-B checkpoint chain is missing or nonterminal"
                )
            reconstructed.append(
                _dgp_result_from_receipts(tier_a, accelerators[dgp], receipts, binding)
            )
            look_families.append(store.verified_tier_b_look_results(binding, dgp))
        else:
            if receipts:
                raise IntegrityError(
                    "bridge driver contains Tier-B work after an upstream failure"
                )
            reason = (
                "tier_a_failed"
                if benchmark.passed
                else "bridge_resource_preflight_failed"
            )
            reconstructed.append(
                _failed_dgp_result(
                    binding,
                    dgp,
                    tier_a,
                    accelerators[dgp],
                    reason=reason,
                )
            )
            look_families.append(())

    dgp_results = tuple(reconstructed)
    if tuple(result.dgp_results) != dgp_results:
        raise IntegrityError("bridge driver DGP results differ from checkpoints")
    combined = all(item.passed for item in dgp_results)
    if combined:
        reasons: tuple[str, ...] = ()
    elif not benchmark.passed:
        reasons = benchmark.reasons or ("bridge_resource_preflight_failed",)
    elif not all_tier_a_passed:
        reasons = ("one_or_both_tier_a_gates_failed",)
    else:
        reasons = ("one_or_both_tier_b_gates_failed",)
    scientific = _combined_scientific_fingerprint(binding, dgp_results, combined)
    if (
        result.bridge_gate_passed != combined
        or result.reasons != reasons
        or result.scientific_fingerprint != scientific
    ):
        raise IntegrityError("bridge driver terminal reduction is inconsistent")

    final = store.final_receipt(binding)
    if final is None or final.fingerprint != result.final_receipt_fingerprint:
        raise IntegrityError("bridge driver final receipt is missing or mismatched")
    expected_final = {
        "schema_version": BRIDGE_CHECKPOINT_SCHEMA_VERSION,
        "kind": "bridge_final",
        "binding_fingerprint": binding.fingerprint,
        "dgp_results": [_dgp_result_dict(item) for item in dgp_results],
        "benchmark_receipt_fingerprint": benchmark.receipt_fingerprint,
        "bridge_gate_passed": combined,
        "generation_authorized": False,
        "reasons": list(reasons),
        "scientific_fingerprint": scientific,
    }
    if dict(final.identity) != expected_final:
        raise IntegrityError("bridge driver final receipt content is inconsistent")
    return ReverifiedBridgeDriverEvidence(
        result,
        request,
        benchmark,
        dgp_results,
        (look_families[0], look_families[1]),
        final.fingerprint,
    )


def _prepare_request(
    request: CanonicalBridgeRequest,
) -> tuple[BridgeScientificBinding, Mapping[BridgeDGP, IntArray]]:
    if not isinstance(request, CanonicalBridgeRequest):
        raise ModelError("canonical bridge request has the wrong type")
    base = request.base_preflight
    if not isinstance(base, CanonicalG3Preflight):
        raise ModelError("canonical bridge requires canonical G3 preflight evidence")
    _require_sha256(base.fingerprint, "base preflight fingerprint", ModelError)
    if _sha256_json(base.identity_dict()) != base.fingerprint:
        raise ModelError("canonical base-preflight evidence hash is inconsistent")
    if base.workers != CANONICAL_WORKERS:
        raise ModelError("canonical bridge base preflight must bind nine workers")
    _nonnegative_integer(request.master_seed, "master_seed")
    _positive_integer(request.scientific_version, "scientific_version")
    selected = int(np.asarray(request.design_parameters.means).shape[0])
    if selected not in {2, 3, 4, 5}:
        raise ModelError("canonical bridge selected K must be one of 2,3,4,5")
    if (
        request.scientific_version == CANONICAL_SCIENTIFIC_VERSION
        and selected != BRIDGE_TIER_A_N_STATES
    ):
        raise ModelError("version-3 canonical bridge requires fixed K=4")
    parameters_sha256 = bridge_parameters_sha256(request.design_parameters)
    design_values = np.asarray(request.design_standardized_history, dtype=np.float64)
    continuation = np.asarray(request.design_continuation)
    if (
        design_values.shape != (BRIDGE_PREFIX_ROWS, 4)
        or not np.isfinite(design_values).all()
        or continuation.shape != (BRIDGE_PREFIX_ROWS,)
        or continuation.dtype.kind != "b"
        or bool(continuation[0])
        or not bool(continuation[1:].all())
    ):
        raise ModelError(
            "canonical bridge design history must be exact gap-free (193,4)"
        )
    _positive_integer(request.macro_block_length, "macro_block_length")
    if request.macro_block_length > BRIDGE_PREFIX_ROWS:
        raise ModelError("canonical bridge macro block length exceeds design history")
    design_history_sha256 = _design_history_sha256(design_values, continuation)
    alpha = _bound_alpha_grids(request.alpha_offset_grids, selected)
    actual_statistic, actual_fingerprint = _actual_bridge_binding(
        request.actual_bridge,
        request.design_parameters,
        request.alpha_offset_grids,
        selected,
    )

    traces: dict[BridgeDGP, IntArray] = {}
    trace_fingerprints: list[tuple[BridgeDGP, str]] = []
    artifacts = {
        "model": CalibrationArtifact.BRIDGE_MODEL,
        "nonparametric": CalibrationArtifact.BRIDGE_NONPARAMETRIC,
    }
    for dgp in BRIDGE_DGPS:
        trace = materialize_rolling_bootstrap_indices(
            master_seed=request.master_seed,
            scientific_version=request.scientific_version,
            artifact=artifacts[dgp],
        )
        checked = np.asarray(trace)
        if (
            checked.shape != (10_000, 6, 12)
            or checked.dtype != np.dtype(np.int64)
            or checked.flags.writeable
            or bool(((checked < 0) | (checked >= 12)).any())
        ):
            raise ModelError("official purpose-1 bridge trace is malformed")
        fingerprint = _purpose1_trace_sha256(checked)
        traces[dgp] = checked
        trace_fingerprints.append((dgp, fingerprint))

    provisional = BridgeScientificBinding(
        base.fingerprint,
        request.master_seed,
        request.scientific_version,
        selected,
        parameters_sha256,
        design_history_sha256,
        request.macro_block_length,
        actual_fingerprint,
        actual_statistic,
        alpha,
        tuple(trace_fingerprints),
        "",
    )
    fingerprint = _sha256_json(provisional.identity_dict())
    binding = BridgeScientificBinding(
        provisional.base_preflight_fingerprint,
        provisional.master_seed,
        provisional.scientific_version,
        provisional.selected_n_states,
        provisional.design_parameters_sha256,
        provisional.design_history_sha256,
        provisional.macro_block_length,
        provisional.actual_bridge_evidence_fingerprint,
        provisional.actual_statistic,
        provisional.alpha_offset_grids,
        provisional.purpose1_trace_fingerprints,
        fingerprint,
    )
    return binding, MappingProxyType(traces)


def _bound_alpha_grids(
    grids: Sequence[AlphaOffsetGrid], selected_n_states: int
) -> tuple[BoundAlphaOffsetGrid, ...]:
    supplied = tuple(grids)
    if len(supplied) != len(BRIDGE_ALPHA_GRID_ROLES):
        raise ModelError("bridge requires exactly four design/production alpha grids")
    expected_frequencies = ("monthly", "daily", "monthly", "daily")
    bound: list[BoundAlphaOffsetGrid] = []
    for role, expected_frequency, grid in zip(
        BRIDGE_ALPHA_GRID_ROLES,
        expected_frequencies,
        supplied,
        strict=True,
    ):
        if (
            not isinstance(grid, AlphaOffsetGrid)
            or grid.frequency != expected_frequency
        ):
            raise ModelError(f"bridge alpha grid {role} has the wrong role/frequency")
        for label, fingerprint in (
            ("model artifact", grid.model_artifact_fingerprint),
            ("kernel sample", grid.kernel_sample_fingerprint),
            ("target history", grid.target_history_fingerprint),
            ("grid", grid.grid_fingerprint),
        ):
            _require_sha256(fingerprint, f"{role} {label} fingerprint", ModelError)
        lambdas = np.asarray(grid.lambdas, dtype=np.float64)
        offsets = np.asarray(grid.offsets, dtype=np.float64)
        if (
            lambdas.shape != (4,)
            or not np.array_equal(lambdas, np.asarray(BRIDGE_ALPHA_LAMBDAS))
            or offsets.shape != (4, selected_n_states, 3)
            or not np.isfinite(offsets).all()
            or lambdas.flags.writeable
            or offsets.flags.writeable
        ):
            raise ModelError(
                f"bridge alpha grid {role} must be immutable exact (4,K,3) evidence"
            )
        expected = _alpha_grid_fingerprint(grid, lambdas, offsets)
        if expected != grid.grid_fingerprint:
            raise ModelError(f"bridge alpha grid {role} fingerprint is inconsistent")
        bound.append(
            BoundAlphaOffsetGrid(
                role,
                expected_frequency,
                grid.model_artifact_fingerprint,
                grid.kernel_sample_fingerprint,
                grid.target_history_fingerprint,
                grid.grid_fingerprint,
            )
        )
    if bound[0].model_artifact_fingerprint != bound[1].model_artifact_fingerprint:
        raise ModelError(
            "design monthly/daily alpha grids bind different model artifacts"
        )
    if bound[2].model_artifact_fingerprint != bound[3].model_artifact_fingerprint:
        raise ModelError(
            "production monthly/daily alpha grids bind different model artifacts"
        )
    return tuple(bound)


def _alpha_grid_fingerprint(
    grid: AlphaOffsetGrid, lambdas: FloatArray, offsets: FloatArray
) -> str:
    digest = hashlib.sha256()
    tags = (
        "prpg-alpha-offset-grid-v1",
        grid.frequency,
        grid.model_artifact_fingerprint,
        grid.kernel_sample_fingerprint,
        grid.target_history_fingerprint,
    )
    for value in tags:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    for source in (lambdas, offsets):
        canonical = np.ascontiguousarray(source, dtype=np.dtype("<f8"))
        dtype_name = canonical.dtype.str.encode("ascii")
        digest.update(len(dtype_name).to_bytes(1, "big", signed=False))
        digest.update(dtype_name)
        digest.update(canonical.ndim.to_bytes(1, "big", signed=False))
        for dimension in canonical.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _actual_bridge_binding(
    actual: ActualBridgeExecution,
    design_parameters: HMMParameters,
    grids: Sequence[AlphaOffsetGrid],
    selected_n_states: int,
) -> tuple[float, str]:
    """Verify and hash typed actual-artifact evidence used by both DGP tails."""

    identity = actual_bridge_execution_identity(actual)
    if identity.n_states != selected_n_states:
        raise ModelError("actual bridge selected K differs from the design reference")
    statistic = actual.components.statistic
    gate_statistic = actual.gate.fixed_k
    if (
        gate_statistic.component_names != statistic.component_names
        or not np.array_equal(gate_statistic.raw_maxima, statistic.raw_maxima)
        or not np.array_equal(gate_statistic.hard_caps, statistic.hard_caps)
        or not np.array_equal(
            gate_statistic.normalized_components,
            statistic.normalized_components,
        )
        or gate_statistic.maximum != statistic.maximum
        or not actual.gate.passed
        or actual.gate.reasons
        or not math.isfinite(statistic.maximum)
        or statistic.maximum < 0.0
    ):
        raise ModelError("actual bridge hard-cap evidence has not passed")
    prefix_parameters = (
        actual.components.prefix_alignment.parameters_in_original_coordinates
    )
    full_parameters = (
        actual.components.full_alignment.parameters_in_original_coordinates
    )
    if bridge_parameters_sha256(prefix_parameters) != bridge_parameters_sha256(
        design_parameters
    ):
        raise ModelError("actual bridge design parameters differ from the reference")
    lambdas = np.asarray(actual.kernel_lambdas, dtype=np.float64)
    if not np.array_equal(lambdas, np.asarray(BRIDGE_ALPHA_LAMBDAS)):
        raise ModelError("actual bridge kernel lambda order is inconsistent")
    supplied = tuple(grids)
    design_monthly = np.asarray(supplied[0].offsets, dtype=np.float64)
    design_daily = np.asarray(supplied[1].offsets, dtype=np.float64)
    production_monthly = np.asarray(supplied[2].offsets, dtype=np.float64)
    production_daily = np.asarray(supplied[3].offsets, dtype=np.float64)
    if not np.array_equal(
        production_monthly - design_monthly,
        np.asarray(actual.monthly_kernel_offset_differences),
    ) or not np.array_equal(
        production_daily - design_daily,
        np.asarray(actual.daily_kernel_offset_differences),
    ):
        raise ModelError(
            "actual bridge kernel differences do not match the bound alpha grids"
        )
    payload = {
        "schema_version": 1,
        "actual_bridge_contract_version": identity.contract_version,
        "actual_bridge_source_materialization_fingerprint": (
            identity.source_materialization_fingerprint
        ),
        "actual_bridge_execution_fingerprint": identity.execution_fingerprint,
        "actual_bridge_design_fit_fingerprint": identity.design_fit_fingerprint,
        "actual_bridge_production_fit_fingerprint": (
            identity.production_fit_fingerprint
        ),
        "selected_n_states": selected_n_states,
        "design_parameters_sha256": bridge_parameters_sha256(prefix_parameters),
        "production_parameters_sha256": bridge_parameters_sha256(full_parameters),
        "component_names": list(statistic.component_names),
        "raw_maxima_sha256": _array_sha256(statistic.raw_maxima),
        "hard_caps_sha256": _array_sha256(statistic.hard_caps),
        "normalized_components_sha256": _array_sha256(statistic.normalized_components),
        "maximum": statistic.maximum,
        "normalized_coordinates_sha256": _array_sha256(
            actual.components.normalized_coordinates
        ),
        "component_decisions_sha256": _array_sha256(
            actual.components.component_decisions
        ),
        "production_scaler_means_sha256": _array_sha256(
            actual.production_scaler_means_in_design_coordinates
        ),
        "production_scaler_scales_sha256": _array_sha256(
            actual.production_scaler_scales_in_design_coordinates
        ),
        "kernel_lambdas_sha256": _array_sha256(actual.kernel_lambdas),
        "monthly_kernel_difference_sha256": _array_sha256(
            actual.monthly_kernel_offset_differences
        ),
        "daily_kernel_difference_sha256": _array_sha256(
            actual.daily_kernel_offset_differences
        ),
        "monthly_block_change": actual.gate.monthly_block_change,
        "daily_block_change": actual.gate.daily_block_change,
        "monthly_support_ratios_sha256": _array_sha256(
            actual.gate.monthly_support_ratios
        ),
        "daily_support_ratios_sha256": _array_sha256(actual.gate.daily_support_ratios),
        "maximum_annualized_monthly_kernel_change": (
            actual.gate.maximum_annualized_monthly_kernel_change
        ),
        "maximum_annualized_daily_kernel_change": (
            actual.gate.maximum_annualized_daily_kernel_change
        ),
        "same_k": actual.gate.same_k,
        "same_order": actual.gate.same_order,
        "passed": actual.gate.passed,
    }
    return float(statistic.maximum), _sha256_json(payload)


def _tier_a_jobs(
    request: CanonicalBridgeRequest,
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
    trace: IntArray,
) -> tuple[_TierABatchJob, ...]:
    return tuple(
        _TierABatchJob(
            dgp,
            start,
            min(start + BRIDGE_TIER_A_BATCH_PAIRS, TIER_A_PAIRS),
            binding.fingerprint,
            binding.selected_n_states,
            request.master_seed,
            request.scientific_version,
            request.design_parameters,
            request.design_standardized_history,
            request.design_continuation,
            request.macro_block_length,
            trace,
            binding.purpose1_by_dgp[dgp],
        )
        for start in range(0, TIER_A_PAIRS, BRIDGE_TIER_A_BATCH_PAIRS)
    )


def _run_tier_a_batch(job: _TierABatchJob) -> tuple[BridgeTierAPairEvidence, ...]:
    if not isinstance(job, _TierABatchJob):
        raise ModelError("Tier-A worker received the wrong job type")
    evidence: list[BridgeTierAPairEvidence] = []
    for pair in range(job.pair_start, job.pair_stop):
        history = generate_registered_bridge_history(
            dgp=job.dgp,
            tier="selection",
            pair=pair,
            master_seed=job.master_seed,
            scientific_version=job.scientific_version,
            design_parameters=job.design_parameters,
            design_standardized_history=job.design_history,
            design_continuation=job.design_continuation,
            macro_block_length=job.macro_block_length,
        )
        result_selected_k: int
        if job.scientific_version == CANONICAL_SCIENTIFIC_VERSION:
            if job.selected_n_states != BRIDGE_TIER_A_N_STATES:
                raise ModelError("version-3 Tier-A worker is not fixed at K=4")
            fixed_result = run_bridge_fixed_k4_tier_a_pair(
                history,
                parent_parameters=job.design_parameters,
                master_seed=job.master_seed,
                scientific_version=job.scientific_version,
            )
            prefix_selected = (
                BRIDGE_TIER_A_N_STATES if fixed_result.prefix.fit is not None else None
            )
            full_selected = (
                BRIDGE_TIER_A_N_STATES if fixed_result.full.fit is not None else None
            )
            prefix_passed = fixed_result.prefix.passed
            full_passed = fixed_result.full.passed
            failures = tuple(
                f"{member.member}:{reason}"
                for member in (fixed_result.prefix, fixed_result.full)
                for reason in member.rejection_reasons
            )
            result_dgp = fixed_result.dgp
            result_pair = fixed_result.pair
            result_selected_k = fixed_result.actual_selected_k
            result_history_sha256 = fixed_result.history_sha256
            result_success = fixed_result.success
            tier_a_policy = BRIDGE_FIXED_K4_TIER_A_POLICY
            prefix_episode_counts = fixed_result.prefix.observed_episode_counts
            full_episode_counts = fixed_result.full.observed_episode_counts
            prefix_alignment = fixed_result.prefix.reference_to_candidate
            full_alignment = fixed_result.full.reference_to_candidate
        else:
            legacy_result = run_bridge_tier_a_pair(
                history,
                actual_selected_k=job.selected_n_states,
                master_seed=job.master_seed,
                scientific_version=job.scientific_version,
                registered_bootstrap_indices=job.purpose1_trace,
            )
            prefix_selected = (
                None
                if legacy_result.prefix is None
                else legacy_result.prefix.selection.selected_n_states
            )
            full_selected = (
                None
                if legacy_result.full is None
                else legacy_result.full.selection.selected_n_states
            )
            prefix_passed = bool(
                legacy_result.prefix is not None
                and legacy_result.prefix.selection.passed
            )
            full_passed = bool(
                legacy_result.full is not None and legacy_result.full.selection.passed
            )
            failures = tuple(
                f"{item.member}:{item.failure_type}"
                for item in legacy_result.member_failures
            )
            result_dgp = legacy_result.dgp
            result_pair = legacy_result.pair
            result_selected_k = legacy_result.actual_selected_k
            result_history_sha256 = legacy_result.history_sha256
            result_success = legacy_result.success
            tier_a_policy = "cross_k_rolling_v1"
            prefix_episode_counts = ()
            full_episode_counts = ()
            prefix_alignment = ()
            full_alignment = ()
        provisional = BridgeTierAPairEvidence(
            result_dgp,
            result_pair,
            result_selected_k,
            result_history_sha256,
            job.purpose1_trace_sha256,
            prefix_selected,
            full_selected,
            prefix_passed,
            full_passed,
            failures,
            result_success,
            job.binding_fingerprint,
            "",
            tier_a_policy,
            prefix_episode_counts,
            full_episode_counts,
            prefix_alignment,
            full_alignment,
        )
        evidence.append(_with_tier_a_fingerprint(provisional))
    return tuple(evidence)


def _fixed_k_job(
    request: CanonicalBridgeRequest,
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
    pair: int,
    mode: FitMode,
) -> _FixedKJob:
    return _FixedKJob(
        dgp,
        pair,
        binding.fingerprint,
        binding.selected_n_states,
        request.master_seed,
        request.scientific_version,
        request.design_parameters,
        request.design_standardized_history,
        request.design_continuation,
        request.macro_block_length,
        mode,
    )


def _run_fixed_k_job(job: _FixedKJob) -> BridgeTierBPairEvidence:
    if not isinstance(job, _FixedKJob):
        raise ModelError("fixed-K worker received the wrong job type")
    history = generate_registered_bridge_history(
        dgp=job.dgp,
        tier="fixed_k",
        pair=job.pair,
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        design_parameters=job.design_parameters,
        design_standardized_history=job.design_history,
        design_continuation=job.design_continuation,
        macro_block_length=job.macro_block_length,
    )
    result = run_bridge_fixed_k_pair(
        history,
        n_states=job.selected_n_states,
        parent_parameters=job.design_parameters,
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        mode=job.mode,
    )
    statistic = result.statistic
    provisional = BridgeTierBPairEvidence(
        result.dgp,
        result.pair,
        result.n_states,
        result.mode,
        result.history_sha256,
        result.valid,
        None if statistic is None else float(statistic),
        result.failure_type,
        result.failure_message,
        job.binding_fingerprint,
        "",
    )
    return _with_tier_b_fingerprint(provisional)


def _benchmark_job(
    request: CanonicalBridgeRequest,
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
    pair: int,
) -> _BenchmarkJob:
    return _BenchmarkJob(
        dgp,
        pair,
        binding.fingerprint,
        binding.selected_n_states,
        request.master_seed,
        request.scientific_version,
        request.design_parameters,
        request.design_standardized_history,
        request.design_continuation,
        request.macro_block_length,
    )


def _run_benchmark_job(job: _BenchmarkJob) -> BridgeBenchmarkPairEvidence:
    if not isinstance(job, _BenchmarkJob):
        raise ModelError("bridge benchmark worker received the wrong job type")
    started = time.monotonic()
    base = _FixedKJob(
        job.dgp,
        job.pair,
        job.binding_fingerprint,
        job.selected_n_states,
        job.master_seed,
        job.scientific_version,
        job.design_parameters,
        job.design_history,
        job.design_continuation,
        job.macro_block_length,
        "accelerated_10",
    )
    accelerated_result = _run_fixed_k_full(base)
    ordinary_result = _run_fixed_k_full(
        _FixedKJob(
            base.dgp,
            base.pair,
            base.binding_fingerprint,
            base.selected_n_states,
            base.master_seed,
            base.scientific_version,
            base.design_parameters,
            base.design_history,
            base.design_continuation,
            base.macro_block_length,
            "ordinary_50",
        )
    )
    accelerated = _compact_fixed_k_result(accelerated_result, job.binding_fingerprint)
    ordinary = _compact_fixed_k_result(ordinary_result, job.binding_fingerprint)
    if accelerated.history_sha256 != ordinary.history_sha256:
        raise ModelError("benchmark modes did not reuse identical pseudo-history bytes")

    fast_valid = (accelerated_result.prefix.valid, accelerated_result.full.valid)
    full_valid = (ordinary_result.prefix.valid, ordinary_result.full.valid)
    likelihood = [0.0, 0.0]
    for index, member in enumerate(("prefix", "full")):
        fast_member = getattr(accelerated_result, member)
        full_member = getattr(ordinary_result, member)
        if fast_member.valid and full_member.valid:
            if fast_member.fit is None or full_member.fit is None:
                raise AssertionError("valid benchmark member omitted its fit")
            if fast_member.fit.n_observations != full_member.fit.n_observations:
                raise ModelError("benchmark member observation counts differ")
            likelihood[index] = (
                abs(fast_member.fit.log_likelihood - full_member.fit.log_likelihood)
                / full_member.fit.n_observations
            )
    if accelerated_result.valid and ordinary_result.valid:
        fast_components = accelerated_result.components
        full_components = ordinary_result.components
        if fast_components is None or full_components is None:
            raise AssertionError("valid benchmark pair omitted aligned components")
        if (
            fast_components.coordinate_family != full_components.coordinate_family
            or fast_components.coordinate_names != full_components.coordinate_names
            or fast_components.normalized_coordinates.shape
            != full_components.normalized_coordinates.shape
        ):
            raise ModelError("benchmark component coordinates differ")
        component_value = float(
            np.max(
                np.abs(
                    fast_components.normalized_coordinates
                    - full_components.normalized_coordinates
                )
            )
        )
        statistic_value = abs(
            fast_components.statistic.maximum - full_components.statistic.maximum
        )
        decisions_value = bool(
            np.array_equal(
                fast_components.component_decisions,
                full_components.component_decisions,
            )
        )
    else:
        component_value = 0.0
        statistic_value = 0.0
        decisions_value = True
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        elapsed = float(np.nextafter(0.0, 1.0))
    provisional = BridgeBenchmarkPairEvidence(
        job.dgp,
        job.pair,
        job.selected_n_states,
        accelerated.history_sha256,
        accelerated,
        ordinary,
        fast_valid,
        full_valid,
        (float(likelihood[0]), float(likelihood[1])),
        (component_value, component_value),
        (statistic_value, statistic_value),
        (decisions_value, decisions_value),
        elapsed,
        job.binding_fingerprint,
        "",
        "",
    )
    return _with_benchmark_fingerprints(provisional)


def _run_fixed_k_full(job: _FixedKJob) -> Any:
    history = generate_registered_bridge_history(
        dgp=job.dgp,
        tier="fixed_k",
        pair=job.pair,
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        design_parameters=job.design_parameters,
        design_standardized_history=job.design_history,
        design_continuation=job.design_continuation,
        macro_block_length=job.macro_block_length,
    )
    return run_bridge_fixed_k_pair(
        history,
        n_states=job.selected_n_states,
        parent_parameters=job.design_parameters,
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        mode=job.mode,
    )


def _compact_fixed_k_result(
    result: Any, binding_fingerprint: str
) -> BridgeTierBPairEvidence:
    statistic = result.statistic
    provisional = BridgeTierBPairEvidence(
        result.dgp,
        result.pair,
        result.n_states,
        result.mode,
        result.history_sha256,
        result.valid,
        None if statistic is None else float(statistic),
        result.failure_type,
        result.failure_message,
        binding_fingerprint,
        "",
    )
    return _with_tier_b_fingerprint(provisional)


def _execute_benchmark(
    request: CanonicalBridgeRequest,
    binding: BridgeScientificBinding,
    store: BridgeCheckpointStore,
) -> _Receipt:
    jobs = tuple(
        _benchmark_job(request, binding, dgp, pair)
        for dgp in BRIDGE_DGPS
        for pair in range(ACCELERATOR_AUDIT_PAIRS)
    )
    measured: MeasuredProcessMapResult[BridgeBenchmarkPairEvidence] = (
        measured_process_map(
            _run_benchmark_job,
            jobs,
            workers=CANONICAL_WORKERS,
            estimated_peak_bytes=request.base_preflight.estimated_peak_bytes,
            physical_memory_bytes=request.base_preflight.physical_memory_bytes,
            maximum_memory_fraction=0.70,
            chunksize=1,
        )
    )
    records = tuple(measured.outputs)
    accelerators = tuple(
        _reduce_accelerator(records, binding, dgp) for dgp in BRIDGE_DGPS
    )
    by_coordinate = {(item.dgp, item.pair): item for item in records}
    timing = np.asarray(
        [
            max(by_coordinate[(dgp, pair)].elapsed_seconds for dgp in BRIDGE_DGPS)
            for pair in range(ACCELERATOR_AUDIT_PAIRS)
        ],
        dtype=np.float64,
    )
    projection = bridge_resource_preflight(
        benchmark_pair_seconds=timing,
        projected_pair_fits=_projected_benchmark_pair_units(
            accelerators,
            selected_n_states=binding.selected_n_states,
            scientific_version=binding.scientific_version,
        ),
        workers=CANONICAL_WORKERS,
        peak_memory_fraction=measured.resources.peak_memory_fraction,
        maximum_hours=72.0,
        maximum_memory_fraction=0.70,
    )
    reasons: list[str] = []
    if not measured.resources.passed:
        reasons.append("measured_process_resources_failed_closed")
    if not projection.passed:
        reasons.extend(projection.reasons)
    retained_invalid = [
        f"{item.dgp}:{pair.pair}"
        for item in accelerators
        for pair in item.retained_pairs
        if not pair.valid or pair.statistic is None
    ]
    if retained_invalid:
        reasons.append("one_or_more_retained_first_100_pairs_are_invalid")
    canonical: CanonicalBridgeResourcePreflight | None = None
    if not reasons:
        canonical = canonical_bridge_resource_preflight(
            request.base_preflight,
            projection,
        )
    passed = not reasons and canonical is not None
    return store.write_benchmark(
        binding,
        records=records,
        accelerators=accelerators,
        measurement=measured.resources,
        projection=projection,
        canonical_preflight=canonical,
        passed=passed,
        reasons=reasons,
    )


def _reduce_accelerator(
    records: Sequence[BridgeBenchmarkPairEvidence],
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
) -> BridgeAcceleratorDecision:
    selected = tuple(item for item in records if item.dgp == dgp)
    if len(selected) != ACCELERATOR_AUDIT_PAIRS or tuple(
        item.pair for item in selected
    ) != tuple(range(ACCELERATOR_AUDIT_PAIRS)):
        raise ModelError("accelerator reduction requires exact ordered first-100 pairs")
    checked = tuple(_validated_benchmark_pair(item, binding) for item in selected)
    result = accelerator_eligibility(
        accelerated_valid=np.asarray(
            [item.accelerated_valid for item in checked], dtype=np.bool_
        ),
        ordinary_valid=np.asarray(
            [item.ordinary_valid for item in checked], dtype=np.bool_
        ),
        likelihood_difference_per_observation=np.asarray(
            [item.likelihood_difference_per_observation for item in checked],
            dtype=np.float64,
        ),
        maximum_component_difference=np.asarray(
            [item.maximum_component_difference for item in checked], dtype=np.float64
        ),
        statistic_difference=np.asarray(
            [item.statistic_difference for item in checked], dtype=np.float64
        ),
        component_decisions_identical=np.asarray(
            [item.component_decisions_identical for item in checked], dtype=np.bool_
        ),
    )
    mode: FitMode = "accelerated_10" if result.eligible else "ordinary_50"
    retained = tuple(
        item.accelerated if result.eligible else item.ordinary for item in checked
    )
    scientific = _sha256_json(
        {
            "binding_fingerprint": binding.fingerprint,
            "dgp": dgp,
            "benchmark_scientific_fingerprints": [
                item.scientific_fingerprint for item in checked
            ],
            "agreeing_pairs": result.agreeing_pairs,
            "eligible": result.eligible,
            "retained_mode": mode,
            "retained_pair_evidence": [item.evidence_fingerprint for item in retained],
        }
    )
    return BridgeAcceleratorDecision(dgp, result, mode, retained, scientific)


def _projected_benchmark_pair_units(
    accelerators: Sequence[BridgeAcceleratorDecision],
    *,
    selected_n_states: int,
    scientific_version: int = 1,
) -> int:
    """Convert all remaining work to conservative measured-pair units.

    Dense HMM filtering/EM cost is at least cubic in K.  Version-3 Tier A and
    its benchmark both fit K=4, so its exact 100 starts per pair have unit
    state-complexity weight.  Historical projections retain their old
    K=2..5/K=5 conversion for immutable checkpoint verification.
    """

    values = tuple(accelerators)
    if tuple(item.dgp for item in values) != BRIDGE_DGPS:
        raise ModelError("resource projection requires both accelerator decisions")
    if selected_n_states not in {2, 3, 4, 5}:
        raise ModelError("resource projection selected K must be one of 2,3,4,5")
    fixed_k4_tier_a = scientific_version == CANONICAL_SCIENTIFIC_VERSION
    if fixed_k4_tier_a and selected_n_states != BRIDGE_TIER_A_N_STATES:
        raise ModelError("version-3 Tier-A resource projection requires K=4")
    tier_a_reference_k = BRIDGE_TIER_A_N_STATES if fixed_k4_tier_a else 5
    tier_a_restarts_per_pair = (
        BRIDGE_TIER_A_RESTARTS_PER_PAIR
        if fixed_k4_tier_a
        else BRIDGE_LEGACY_TIER_A_RESTARTS_PER_PAIR
    )
    tier_a_complexity_ratio = (tier_a_reference_k / selected_n_states) ** (
        BRIDGE_TIER_A_STATE_COMPLEXITY_POWER
    )
    tier_a_weighted_restarts = (
        len(BRIDGE_DGPS)
        * TIER_A_PAIRS
        * tier_a_restarts_per_pair
        * tier_a_complexity_ratio
    )
    tier_b_restarts = sum(
        (TIER_B_MAXIMUM_PAIRS - ACCELERATOR_AUDIT_PAIRS)
        * (
            BRIDGE_ACCELERATED_RESTARTS_PER_PAIR
            if item.result.eligible
            else BRIDGE_ORDINARY_RESTARTS_PER_PAIR
        )
        for item in values
    )
    total_restarts = tier_a_weighted_restarts + tier_b_restarts
    return math.ceil(
        BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR
        * total_restarts
        / BRIDGE_BENCHMARK_RESTARTS_PER_PAIR
    )


def _run_remaining_tier_b_looks(
    request: CanonicalBridgeRequest,
    binding: BridgeScientificBinding,
    store: BridgeCheckpointStore,
    dgp: BridgeDGP,
    accelerator: BridgeAcceleratorDecision,
    existing: Sequence[_Receipt],
) -> tuple[_Receipt, ...]:
    receipts = list(existing)
    cumulative = _statistics_from_look_receipts(receipts, binding, dgp)
    while len(receipts) < TIER_B_MAXIMUM_PAIRS // TIER_B_BATCH_SIZE:
        if receipts and receipts[-1].identity["decision"] != "continue":
            break
        look_index = len(receipts) + 1
        start = (look_index - 1) * TIER_B_BATCH_SIZE
        stop = look_index * TIER_B_BATCH_SIZE
        pair_evidence: list[BridgeTierBPairEvidence] = []
        if look_index == 1:
            pair_evidence.extend(accelerator.retained_pairs)
            work_start = ACCELERATOR_AUDIT_PAIRS
        else:
            work_start = start
        jobs = tuple(
            _fixed_k_job(
                request,
                binding,
                dgp,
                pair,
                accelerator.retained_mode,
            )
            for pair in range(work_start, stop)
        )
        outputs = deterministic_process_map(
            _run_fixed_k_job,
            jobs,
            workers=CANONICAL_WORKERS,
            chunksize=1,
        )
        pair_evidence.extend(outputs)
        checked = tuple(
            _validated_tier_b_pair(item, binding, dgp) for item in pair_evidence
        )
        if tuple(item.pair for item in checked) != tuple(range(start, stop)):
            raise ModelError("Tier-B worker reduction returned the wrong pair order")
        invalid = [
            item.pair for item in checked if not item.valid or item.statistic is None
        ]
        if invalid:
            raise ModelError(
                "Tier-B fixed-K pseudo-fit failed closed",
                details={"dgp": dgp, "pairs": invalid[:25]},
            )
        batch_statistics = [
            float(item.statistic) for item in checked if item.statistic is not None
        ]
        cumulative.extend(batch_statistics)
        values = np.asarray(cumulative, dtype=np.float64)
        exceedances = int(np.count_nonzero(values >= binding.actual_statistic))
        interval = clopper_pearson_interval(
            exceedances,
            stop,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        decision = _tier_b_decision(interval, stop)
        receipt = store.write_tier_b_look(
            binding,
            dgp=dgp,
            look_index=look_index,
            pairs=checked,
            prior_receipts=receipts,
            cumulative_statistics=cumulative,
            decision=decision,
            interval=interval,
            exceedances=exceedances,
        )
        receipts.append(receipt)
        if decision != "continue":
            gate = tier_b_tail_gate(binding.actual_statistic, values)
            if (
                gate.decision != decision
                or gate.pairs_used != stop
                or gate.exceedances != exceedances
            ):
                raise AssertionError(
                    "incremental Tier-B decision diverged from primitive"
                )
            break
    return tuple(receipts)


def _tier_b_decision(interval: BinomialInterval, sample_count: int) -> BridgeDecision:
    if interval.lower > TIER_B_TAIL_PROBABILITY:
        return "pass_above_tail"
    if interval.upper < TIER_B_TAIL_PROBABILITY:
        return "fail_below_tail"
    if sample_count == TIER_B_MAXIMUM_PAIRS:
        return "fail_closed_at_maximum"
    return "continue"


def _statistics_from_look_receipts(
    receipts: Sequence[_Receipt],
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
) -> list[float]:
    statistics: list[float] = []
    for expected_index, receipt in enumerate(receipts, start=1):
        identity = receipt.identity
        if (
            identity.get("kind") != "bridge_tier_b_look"
            or identity.get("binding_fingerprint") != binding.fingerprint
            or identity.get("dgp") != dgp
            or identity.get("look_index") != expected_index
        ):
            raise IntegrityError("Tier-B look receipt identity is inconsistent")
        records = _sequence(identity.get("pairs"), "Tier-B pair records")
        expected_start = (expected_index - 1) * TIER_B_BATCH_SIZE
        if len(records) != TIER_B_BATCH_SIZE:
            raise IntegrityError("Tier-B look receipt has the wrong pair count")
        for expected_pair, record in enumerate(records, start=expected_start):
            item = _tier_b_pair_from_dict(record)
            _validated_tier_b_pair(item, binding, dgp)
            if item.pair != expected_pair or not item.valid or item.statistic is None:
                raise IntegrityError("Tier-B look pair identity/statistic is invalid")
            statistics.append(item.statistic)
        stop = expected_index * TIER_B_BATCH_SIZE
        values = np.asarray(statistics, dtype=np.float64)
        exceedances = int(np.count_nonzero(values >= binding.actual_statistic))
        interval = clopper_pearson_interval(
            exceedances,
            stop,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        decision = _tier_b_decision(interval, stop)
        if (
            identity.get("sample_count") != stop
            or identity.get("exceedances") != exceedances
            or identity.get("interval") != _interval_dict(interval)
            or identity.get("decision") != decision
        ):
            raise IntegrityError("Tier-B look receipt decision was tampered")
        previous = (
            receipts[expected_index - 2].fingerprint if expected_index > 1 else None
        )
        if identity.get("previous_receipt_fingerprint") != previous:
            raise IntegrityError("Tier-B look receipt hash chain is broken")
        if expected_index < len(receipts) and decision != "continue":
            raise IntegrityError("Tier-B work exists after a terminal decision")
    return statistics


def _validate_look_chain(
    receipts: Sequence[_Receipt],
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
    accelerator: BridgeAcceleratorDecision,
) -> None:
    _statistics_from_look_receipts(receipts, binding, dgp)
    if receipts:
        first_records = _sequence(receipts[0].identity.get("pairs"), "first look pairs")
        retained = tuple(_tier_b_pair_from_dict(item) for item in first_records[:100])
        if tuple(item.as_dict() for item in retained) != tuple(
            item.as_dict() for item in accelerator.retained_pairs
        ):
            raise IntegrityError(
                "first Tier-B look does not reuse benchmark pair evidence"
            )


def _dgp_result_from_receipts(
    tier_a: BridgeTierAResult,
    accelerator: BridgeAcceleratorDecision,
    receipts: Sequence[_Receipt],
    binding: BridgeScientificBinding,
) -> BridgeDGPResult:
    statistics = _statistics_from_look_receipts(receipts, binding, tier_a.dgp)
    if not receipts or receipts[-1].identity["decision"] == "continue":
        raise ModelError("Tier-B execution ended without a terminal decision")
    decision = receipts[-1].identity["decision"]
    if decision not in {
        "pass_above_tail",
        "fail_below_tail",
        "fail_closed_at_maximum",
    }:
        raise IntegrityError("Tier-B terminal decision is invalid")
    passed = decision == "pass_above_tail"
    scientific = _sha256_json(
        {
            "binding_fingerprint": binding.fingerprint,
            "dgp": tier_a.dgp,
            "tier_a_scientific_fingerprint": tier_a.scientific_fingerprint,
            "accelerator_scientific_fingerprint": accelerator.scientific_fingerprint,
            "look_scientific_fingerprints": [
                receipt.identity["scientific_fingerprint"] for receipt in receipts
            ],
            "tier_b_pairs": len(statistics),
            "tier_b_decision": decision,
            "tier_b_passed": passed,
        }
    )
    return BridgeDGPResult(
        tier_a.dgp,
        tier_a,
        accelerator.result.eligible,
        decision,
        len(statistics),
        passed,
        scientific,
    )


def _tier_a_result_from_receipt(
    receipt: _Receipt,
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
) -> BridgeTierAResult:
    identity = receipt.identity
    if (
        identity.get("kind") != "bridge_tier_a"
        or identity.get("binding_fingerprint") != binding.fingerprint
        or identity.get("dgp") != dgp
        or identity.get("selected_n_states") != binding.selected_n_states
        or identity.get("purpose1_trace_sha256") != binding.purpose1_by_dgp[dgp]
    ):
        raise IntegrityError("Tier-A receipt identity is inconsistent")
    records = _sequence(identity.get("pairs"), "Tier-A pair records")
    if len(records) != TIER_A_PAIRS:
        raise IntegrityError("Tier-A receipt does not contain exactly 250 pairs")
    evidence: list[BridgeTierAPairEvidence] = []
    for expected, record in enumerate(records):
        item = _tier_a_pair_from_dict(record)
        _validated_tier_a_pair(item, binding, dgp)
        if item.pair != expected:
            raise IntegrityError("Tier-A receipt pair order is inconsistent")
        evidence.append(item)
    gate = tier_a_selection_gate(
        np.asarray([item.success for item in evidence], dtype=np.bool_)
    )
    expected_gate = {
        "pairs": gate.pairs,
        "successes": gate.successes,
        "estimate": gate.estimate,
        "one_sided_lower_95": gate.one_sided_lower_95,
        "required_lower_bound": gate.required_lower_bound,
        "passed": gate.passed,
    }
    if identity.get("gate") != expected_gate:
        raise IntegrityError("Tier-A receipt gate does not match its pair evidence")
    scientific = _sha256_json(
        {
            "binding_fingerprint": binding.fingerprint,
            "dgp": dgp,
            "pair_evidence": [item.evidence_fingerprint for item in evidence],
            "successes": gate.successes,
            "lower_bound": gate.one_sided_lower_95,
            "passed": gate.passed,
        }
    )
    if identity.get("scientific_fingerprint") != scientific:
        raise IntegrityError("Tier-A scientific fingerprint is inconsistent")
    return BridgeTierAResult(
        dgp,
        gate.pairs,
        gate.successes,
        gate.one_sided_lower_95,
        gate.passed,
        scientific,
        receipt.fingerprint,
    )


def _tier_a_pairs_from_batch_receipts(
    receipts: Sequence[_Receipt],
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
) -> tuple[BridgeTierAPairEvidence, ...]:
    evidence: list[BridgeTierAPairEvidence] = []
    for expected_batch, receipt in enumerate(receipts):
        identity = receipt.identity
        start = expected_batch * BRIDGE_TIER_A_BATCH_PAIRS
        stop = start + BRIDGE_TIER_A_BATCH_PAIRS
        previous = receipts[expected_batch - 1].fingerprint if expected_batch else None
        if (
            identity.get("kind") != "bridge_tier_a_batch"
            or identity.get("binding_fingerprint") != binding.fingerprint
            or identity.get("dgp") != dgp
            or identity.get("selected_n_states") != binding.selected_n_states
            or identity.get("purpose1_trace_sha256") != binding.purpose1_by_dgp[dgp]
            or identity.get("batch_index") != expected_batch
            or identity.get("pair_start") != start
            or identity.get("pair_stop") != stop
            or identity.get("previous_receipt_fingerprint") != previous
        ):
            raise IntegrityError("Tier-A batch receipt identity/hash chain is invalid")
        records = _sequence(identity.get("pairs"), "Tier-A batch pair records")
        if len(records) != BRIDGE_TIER_A_BATCH_PAIRS:
            raise IntegrityError("Tier-A batch receipt has the wrong pair count")
        batch: list[BridgeTierAPairEvidence] = []
        for expected_pair, record in enumerate(records, start=start):
            item = _tier_a_pair_from_dict(record)
            _validated_tier_a_pair(item, binding, dgp)
            if item.pair != expected_pair:
                raise IntegrityError("Tier-A batch pair order is inconsistent")
            batch.append(item)
        expected_scientific = _sha256_json(
            {
                "binding_fingerprint": binding.fingerprint,
                "dgp": dgp,
                "batch_index": expected_batch,
                "pair_evidence": [item.evidence_fingerprint for item in batch],
                "previous_receipt_fingerprint": previous,
            }
        )
        if identity.get("scientific_fingerprint") != expected_scientific:
            raise IntegrityError("Tier-A batch scientific fingerprint is invalid")
        evidence.extend(batch)
    return tuple(evidence)


def _benchmark_from_receipt(
    receipt: _Receipt,
    binding: BridgeScientificBinding,
) -> BridgeBenchmarkResult:
    identity = receipt.identity
    if (
        identity.get("kind") != "bridge_tier_b_benchmark"
        or identity.get("binding_fingerprint") != binding.fingerprint
    ):
        raise IntegrityError("bridge benchmark receipt identity is inconsistent")
    raw_records = _sequence(identity.get("records"), "bridge benchmark records")
    records = tuple(_benchmark_pair_from_dict(item) for item in raw_records)
    expected = tuple(
        (dgp, pair) for dgp in BRIDGE_DGPS for pair in range(ACCELERATOR_AUDIT_PAIRS)
    )
    if tuple((item.dgp, item.pair) for item in records) != expected:
        raise IntegrityError("bridge benchmark pair order is inconsistent")
    for item in records:
        _validated_benchmark_pair(item, binding)
    accelerators = tuple(
        _reduce_accelerator(records, binding, dgp) for dgp in BRIDGE_DGPS
    )
    if identity.get("accelerators") != [
        _accelerator_dict(item) for item in accelerators
    ]:
        raise IntegrityError("bridge benchmark accelerator reduction was tampered")
    measurement = _measurement_from_dict(identity.get("measurement"))
    projection = _projection_from_dict(identity.get("projection"))
    by_coordinate = {(item.dgp, item.pair): item for item in records}
    timing = np.asarray(
        [
            max(by_coordinate[(dgp, pair)].elapsed_seconds for dgp in BRIDGE_DGPS)
            for pair in range(ACCELERATOR_AUDIT_PAIRS)
        ],
        dtype=np.float64,
    )
    expected_projection = bridge_resource_preflight(
        benchmark_pair_seconds=timing,
        projected_pair_fits=_projected_benchmark_pair_units(
            accelerators,
            selected_n_states=binding.selected_n_states,
            scientific_version=binding.scientific_version,
        ),
        workers=CANONICAL_WORKERS,
        peak_memory_fraction=measurement.peak_memory_fraction,
        maximum_hours=72.0,
        maximum_memory_fraction=0.70,
    )
    if projection != expected_projection:
        raise IntegrityError(
            "bridge resource projection does not match benchmark evidence"
        )
    raw_canonical = identity.get("canonical_preflight")
    canonical = (
        None if raw_canonical is None else _canonical_bridge_from_dict(raw_canonical)
    )
    if canonical is not None and (
        canonical.base_preflight_fingerprint != binding.base_preflight_fingerprint
        or canonical.projected_hours != projection.projected_worst_case_hours
        or canonical.peak_memory_fraction != projection.peak_memory_fraction
    ):
        raise IntegrityError(
            "canonical bridge resource preflight binding is inconsistent"
        )
    reasons = tuple(_string_sequence(identity.get("reasons"), "benchmark reasons"))
    retained_invalid = any(
        not pair.valid or pair.statistic is None
        for item in accelerators
        for pair in item.retained_pairs
    )
    expected_pass = bool(
        measurement.passed
        and projection.passed
        and not retained_invalid
        and canonical is not None
        and not reasons
    )
    if identity.get("passed") is not expected_pass:
        raise IntegrityError("bridge benchmark pass flag is inconsistent")
    return BridgeBenchmarkResult(
        accelerators,
        measurement,
        projection,
        canonical,
        expected_pass,
        reasons,
        receipt.fingerprint,
    )


def _finish_without_tier_b(
    binding: BridgeScientificBinding,
    checkpoint_path: Path,
    store: BridgeCheckpointStore,
    tier_a_results: tuple[BridgeTierAResult, ...],
    *,
    benchmark: BridgeBenchmarkResult,
    reasons: tuple[str, ...],
) -> BridgeDriverResult:
    accelerator = {item.dgp: item for item in benchmark.accelerators}
    dgp_results = tuple(
        _failed_dgp_result(
            binding,
            item.dgp,
            item,
            accelerator[item.dgp],
            reason="tier_a_failed",
        )
        for item in tier_a_results
    )
    scientific = _combined_scientific_fingerprint(binding, dgp_results, False)
    final = store.write_final(
        binding,
        dgp_results=dgp_results,
        bridge_gate_passed=False,
        reasons=reasons,
        scientific_fingerprint=scientific,
        benchmark_receipt_fingerprint=benchmark.receipt_fingerprint,
    )
    return _verified_bridge_result(
        BridgeDriverResult(
            binding,
            checkpoint_path,
            dgp_results,
            benchmark,
            False,
            reasons,
            scientific,
            final.fingerprint,
        )
    )


def _finish_preflight_failure(
    binding: BridgeScientificBinding,
    checkpoint_path: Path,
    store: BridgeCheckpointStore,
    benchmark: BridgeBenchmarkResult,
) -> BridgeDriverResult:
    accelerator = {item.dgp: item for item in benchmark.accelerators}
    dgp_results = tuple(
        _failed_dgp_result(
            binding,
            dgp,
            None,
            accelerator[dgp],
            reason="bridge_resource_preflight_failed",
        )
        for dgp in BRIDGE_DGPS
    )
    reasons = benchmark.reasons or ("bridge_resource_preflight_failed",)
    scientific = _combined_scientific_fingerprint(binding, dgp_results, False)
    final = store.write_final(
        binding,
        dgp_results=dgp_results,
        bridge_gate_passed=False,
        reasons=reasons,
        scientific_fingerprint=scientific,
        benchmark_receipt_fingerprint=benchmark.receipt_fingerprint,
    )
    return _verified_bridge_result(
        BridgeDriverResult(
            binding,
            checkpoint_path,
            dgp_results,
            benchmark,
            False,
            reasons,
            scientific,
            final.fingerprint,
        )
    )


def _failed_dgp_result(
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
    tier_a: BridgeTierAResult | None,
    accelerator: BridgeAcceleratorDecision | None,
    *,
    reason: str,
) -> BridgeDGPResult:
    scientific = _sha256_json(
        {
            "binding_fingerprint": binding.fingerprint,
            "dgp": dgp,
            "tier_a_scientific_fingerprint": (
                None if tier_a is None else tier_a.scientific_fingerprint
            ),
            "accelerator_scientific_fingerprint": (
                None if accelerator is None else accelerator.scientific_fingerprint
            ),
            "tier_b_not_run_reason": reason,
        }
    )
    return BridgeDGPResult(
        dgp,
        tier_a,
        None if accelerator is None else accelerator.result.eligible,
        None,
        0,
        False,
        scientific,
    )


def _combined_scientific_fingerprint(
    binding: BridgeScientificBinding,
    dgp_results: Sequence[BridgeDGPResult],
    passed: bool,
) -> str:
    results = tuple(dgp_results)
    if tuple(item.dgp for item in results) != BRIDGE_DGPS:
        raise ModelError(
            "combined bridge reduction requires both DGPs in registry order"
        )
    if passed != all(item.passed for item in results):
        raise ModelError("combined bridge pass flag is not the two-DGP conjunction")
    return _sha256_json(
        {
            "binding_fingerprint": binding.fingerprint,
            "dgp_scientific_fingerprints": [
                item.scientific_fingerprint for item in results
            ],
            "bridge_gate_passed": passed,
            "generation_authorized": False,
        }
    )


def _with_tier_a_fingerprint(
    item: BridgeTierAPairEvidence,
) -> BridgeTierAPairEvidence:
    fingerprint = _sha256_json(item.identity_dict())
    return BridgeTierAPairEvidence(
        item.dgp,
        item.pair,
        item.selected_n_states,
        item.history_sha256,
        item.purpose1_trace_sha256,
        item.prefix_selected_n_states,
        item.full_selected_n_states,
        item.prefix_passed,
        item.full_passed,
        item.failure_types,
        item.success,
        item.binding_fingerprint,
        fingerprint,
        item.tier_a_policy,
        item.prefix_observed_episode_counts,
        item.full_observed_episode_counts,
        item.prefix_reference_to_candidate,
        item.full_reference_to_candidate,
    )


def _with_tier_b_fingerprint(
    item: BridgeTierBPairEvidence,
) -> BridgeTierBPairEvidence:
    fingerprint = _sha256_json(item.identity_dict())
    return BridgeTierBPairEvidence(
        item.dgp,
        item.pair,
        item.selected_n_states,
        item.mode,
        item.history_sha256,
        item.valid,
        item.statistic,
        item.failure_type,
        item.failure_message,
        item.binding_fingerprint,
        fingerprint,
    )


def _with_benchmark_fingerprints(
    item: BridgeBenchmarkPairEvidence,
) -> BridgeBenchmarkPairEvidence:
    scientific = _sha256_json(item.scientific_identity_dict())
    provisional = BridgeBenchmarkPairEvidence(
        item.dgp,
        item.pair,
        item.selected_n_states,
        item.history_sha256,
        item.accelerated,
        item.ordinary,
        item.accelerated_valid,
        item.ordinary_valid,
        item.likelihood_difference_per_observation,
        item.maximum_component_difference,
        item.statistic_difference,
        item.component_decisions_identical,
        item.elapsed_seconds,
        item.binding_fingerprint,
        scientific,
        "",
    )
    evidence = _sha256_json(provisional.identity_dict())
    return BridgeBenchmarkPairEvidence(
        provisional.dgp,
        provisional.pair,
        provisional.selected_n_states,
        provisional.history_sha256,
        provisional.accelerated,
        provisional.ordinary,
        provisional.accelerated_valid,
        provisional.ordinary_valid,
        provisional.likelihood_difference_per_observation,
        provisional.maximum_component_difference,
        provisional.statistic_difference,
        provisional.component_decisions_identical,
        provisional.elapsed_seconds,
        provisional.binding_fingerprint,
        scientific,
        evidence,
    )


def _validated_tier_a_pair(
    item: BridgeTierAPairEvidence,
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
) -> BridgeTierAPairEvidence:
    if not isinstance(item, BridgeTierAPairEvidence):
        raise ModelError("Tier-A compact evidence has the wrong type")
    if (
        item.dgp != dgp
        or not 0 <= item.pair < TIER_A_PAIRS
        or item.selected_n_states != binding.selected_n_states
        or item.binding_fingerprint != binding.fingerprint
        or item.purpose1_trace_sha256 != binding.purpose1_by_dgp[dgp]
    ):
        raise ModelError("Tier-A compact evidence identity is inconsistent")
    _require_sha256(item.history_sha256, "Tier-A history fingerprint", ModelError)
    _require_sha256(
        item.purpose1_trace_sha256,
        "Tier-A purpose-1 trace fingerprint",
        ModelError,
    )
    for value in (item.prefix_selected_n_states, item.full_selected_n_states):
        if value is not None and value not in {2, 3, 4, 5}:
            raise ModelError("Tier-A compact selection is outside K=2..5")
    if any(not isinstance(value, str) or not value for value in item.failure_types):
        raise ModelError("Tier-A compact failure evidence is malformed")
    fixed_k4_tier_a = binding.scientific_version == CANONICAL_SCIENTIFIC_VERSION
    if fixed_k4_tier_a:
        permutation = tuple(range(BRIDGE_TIER_A_N_STATES))
        if (
            item.tier_a_policy != BRIDGE_FIXED_K4_TIER_A_POLICY
            or item.selected_n_states != BRIDGE_TIER_A_N_STATES
        ):
            raise ModelError("version-3 Tier-A compact evidence is not fixed K4")
        for passed, selected, counts, alignment in (
            (
                item.prefix_passed,
                item.prefix_selected_n_states,
                item.prefix_observed_episode_counts,
                item.prefix_reference_to_candidate,
            ),
            (
                item.full_passed,
                item.full_selected_n_states,
                item.full_observed_episode_counts,
                item.full_reference_to_candidate,
            ),
        ):
            if passed and (
                selected != BRIDGE_TIER_A_N_STATES
                or len(counts) != BRIDGE_TIER_A_N_STATES
                or any(count < 2 for count in counts)
                or tuple(sorted(alignment)) != permutation
            ):
                raise ModelError(
                    "passing version-3 Tier-A member evidence is inconsistent"
                )
    elif (
        item.tier_a_policy != "cross_k_rolling_v1"
        or item.prefix_observed_episode_counts
        or item.full_observed_episode_counts
        or item.prefix_reference_to_candidate
        or item.full_reference_to_candidate
    ):
        raise ModelError("legacy Tier-A compact evidence carries version-3 fields")
    expected_success = bool(
        item.prefix_passed
        and item.full_passed
        and item.prefix_selected_n_states == binding.selected_n_states
        and item.full_selected_n_states == binding.selected_n_states
    )
    if item.success != expected_success:
        raise ModelError("Tier-A compact success flag is inconsistent")
    expected_fingerprint = _sha256_json(item.identity_dict())
    if item.evidence_fingerprint != expected_fingerprint:
        raise ModelError("Tier-A compact evidence fingerprint is inconsistent")
    return item


def _validated_tier_b_pair(
    item: BridgeTierBPairEvidence,
    binding: BridgeScientificBinding,
    dgp: BridgeDGP,
) -> BridgeTierBPairEvidence:
    if not isinstance(item, BridgeTierBPairEvidence):
        raise ModelError("Tier-B compact evidence has the wrong type")
    if (
        item.dgp != dgp
        or not 0 <= item.pair < TIER_B_MAXIMUM_PAIRS
        or item.selected_n_states != binding.selected_n_states
        or item.mode not in {"accelerated_10", "ordinary_50"}
        or item.binding_fingerprint != binding.fingerprint
    ):
        raise ModelError("Tier-B compact evidence identity is inconsistent")
    _require_sha256(item.history_sha256, "Tier-B history fingerprint", ModelError)
    if item.valid:
        if (
            item.statistic is None
            or not math.isfinite(item.statistic)
            or item.statistic < 0.0
            or item.failure_type is not None
            or item.failure_message is not None
        ):
            raise ModelError("valid Tier-B compact evidence is inconsistent")
    elif (
        item.statistic is not None
        or not isinstance(item.failure_type, str)
        or not item.failure_type
        or not isinstance(item.failure_message, str)
        or not item.failure_message
    ):
        raise ModelError("invalid Tier-B compact evidence omitted its failure")
    if item.evidence_fingerprint != _sha256_json(item.identity_dict()):
        raise ModelError("Tier-B compact evidence fingerprint is inconsistent")
    return item


def _validated_benchmark_pair(
    item: BridgeBenchmarkPairEvidence,
    binding: BridgeScientificBinding,
) -> BridgeBenchmarkPairEvidence:
    if not isinstance(item, BridgeBenchmarkPairEvidence):
        raise ModelError("bridge benchmark evidence has the wrong type")
    if (
        item.dgp not in BRIDGE_DGPS
        or not 0 <= item.pair < ACCELERATOR_AUDIT_PAIRS
        or item.selected_n_states != binding.selected_n_states
        or item.binding_fingerprint != binding.fingerprint
    ):
        raise ModelError("bridge benchmark evidence identity is inconsistent")
    accelerated = _validated_tier_b_pair(item.accelerated, binding, item.dgp)
    ordinary = _validated_tier_b_pair(item.ordinary, binding, item.dgp)
    if (
        accelerated.pair != item.pair
        or ordinary.pair != item.pair
        or accelerated.mode != "accelerated_10"
        or ordinary.mode != "ordinary_50"
        or accelerated.history_sha256 != ordinary.history_sha256
        or item.history_sha256 != accelerated.history_sha256
    ):
        raise ModelError("bridge benchmark nested pair evidence is inconsistent")
    for bool_values, label in (
        (item.accelerated_valid, "accelerated validity"),
        (item.ordinary_valid, "ordinary validity"),
        (item.component_decisions_identical, "component decisions"),
    ):
        if len(bool_values) != 2 or any(
            not isinstance(entry, bool) for entry in bool_values
        ):
            raise ModelError(f"bridge benchmark {label} must contain two booleans")
    for numeric_values, label in (
        (item.likelihood_difference_per_observation, "likelihood difference"),
        (item.maximum_component_difference, "component difference"),
        (item.statistic_difference, "statistic difference"),
    ):
        if len(numeric_values) != 2 or any(
            not math.isfinite(entry) or entry < 0.0 for entry in numeric_values
        ):
            raise ModelError(f"bridge benchmark {label} must be two nonnegative values")
    if not math.isfinite(item.elapsed_seconds) or item.elapsed_seconds <= 0.0:
        raise ModelError("bridge benchmark elapsed time must be positive and finite")
    if item.scientific_fingerprint != _sha256_json(item.scientific_identity_dict()):
        raise ModelError("bridge benchmark scientific fingerprint is inconsistent")
    if item.evidence_fingerprint != _sha256_json(item.identity_dict()):
        raise ModelError("bridge benchmark evidence fingerprint is inconsistent")
    return item


def _tier_a_pair_from_dict(value: object) -> BridgeTierAPairEvidence:
    item = _mapping(value, "Tier-A pair evidence")
    _exact_keys(
        item,
        {
            "dgp",
            "pair",
            "selected_n_states",
            "history_sha256",
            "purpose1_trace_sha256",
            "prefix_selected_n_states",
            "full_selected_n_states",
            "prefix_passed",
            "full_passed",
            "failure_types",
            "success",
            "binding_fingerprint",
            "evidence_fingerprint",
            "tier_a_policy",
            "prefix_observed_episode_counts",
            "full_observed_episode_counts",
            "prefix_reference_to_candidate",
            "full_reference_to_candidate",
        },
        "Tier-A pair evidence",
    )
    dgp = _dgp(item["dgp"])
    failures = tuple(_string_sequence(item["failure_types"], "Tier-A failure types"))
    return BridgeTierAPairEvidence(
        dgp,
        _integer(item["pair"], "Tier-A pair"),
        _integer(item["selected_n_states"], "Tier-A selected K"),
        _string(item["history_sha256"], "Tier-A history fingerprint"),
        _string(item["purpose1_trace_sha256"], "Tier-A trace fingerprint"),
        _optional_integer(item["prefix_selected_n_states"], "Tier-A prefix K"),
        _optional_integer(item["full_selected_n_states"], "Tier-A full K"),
        _boolean(item["prefix_passed"], "Tier-A prefix pass"),
        _boolean(item["full_passed"], "Tier-A full pass"),
        failures,
        _boolean(item["success"], "Tier-A success"),
        _string(item["binding_fingerprint"], "Tier-A binding fingerprint"),
        _string(item["evidence_fingerprint"], "Tier-A evidence fingerprint"),
        _string(item["tier_a_policy"], "Tier-A policy"),
        tuple(
            _integer(value, "Tier-A prefix observed episode count")
            for value in _sequence(
                item["prefix_observed_episode_counts"],
                "Tier-A prefix observed episode counts",
            )
        ),
        tuple(
            _integer(value, "Tier-A full observed episode count")
            for value in _sequence(
                item["full_observed_episode_counts"],
                "Tier-A full observed episode counts",
            )
        ),
        tuple(
            _integer(value, "Tier-A prefix state alignment")
            for value in _sequence(
                item["prefix_reference_to_candidate"],
                "Tier-A prefix state alignment",
            )
        ),
        tuple(
            _integer(value, "Tier-A full state alignment")
            for value in _sequence(
                item["full_reference_to_candidate"],
                "Tier-A full state alignment",
            )
        ),
    )


def _tier_b_pair_from_dict(value: object) -> BridgeTierBPairEvidence:
    item = _mapping(value, "Tier-B pair evidence")
    _exact_keys(
        item,
        {
            "dgp",
            "pair",
            "selected_n_states",
            "mode",
            "history_sha256",
            "valid",
            "statistic",
            "failure_type",
            "failure_message",
            "binding_fingerprint",
            "evidence_fingerprint",
        },
        "Tier-B pair evidence",
    )
    mode = item["mode"]
    if mode not in {"accelerated_10", "ordinary_50"}:
        raise IntegrityError("Tier-B fit mode is invalid")
    statistic = _optional_float(item["statistic"], "Tier-B statistic")
    return BridgeTierBPairEvidence(
        _dgp(item["dgp"]),
        _integer(item["pair"], "Tier-B pair"),
        _integer(item["selected_n_states"], "Tier-B selected K"),
        mode,
        _string(item["history_sha256"], "Tier-B history fingerprint"),
        _boolean(item["valid"], "Tier-B validity"),
        statistic,
        _optional_string(item["failure_type"], "Tier-B failure type"),
        _optional_string(item["failure_message"], "Tier-B failure message"),
        _string(item["binding_fingerprint"], "Tier-B binding fingerprint"),
        _string(item["evidence_fingerprint"], "Tier-B evidence fingerprint"),
    )


def _benchmark_pair_from_dict(value: object) -> BridgeBenchmarkPairEvidence:
    item = _mapping(value, "bridge benchmark pair evidence")
    _exact_keys(
        item,
        {
            "dgp",
            "pair",
            "selected_n_states",
            "history_sha256",
            "accelerated_evidence_fingerprint",
            "ordinary_evidence_fingerprint",
            "accelerated_valid",
            "ordinary_valid",
            "likelihood_difference_per_observation",
            "maximum_component_difference",
            "statistic_difference",
            "component_decisions_identical",
            "elapsed_seconds",
            "binding_fingerprint",
            "scientific_fingerprint",
            "accelerated",
            "ordinary",
            "evidence_fingerprint",
        },
        "bridge benchmark pair evidence",
    )
    accelerated = _tier_b_pair_from_dict(item["accelerated"])
    ordinary = _tier_b_pair_from_dict(item["ordinary"])
    if (
        item["accelerated_evidence_fingerprint"] != accelerated.evidence_fingerprint
        or item["ordinary_evidence_fingerprint"] != ordinary.evidence_fingerprint
    ):
        raise IntegrityError("benchmark nested evidence fingerprints disagree")
    return BridgeBenchmarkPairEvidence(
        _dgp(item["dgp"]),
        _integer(item["pair"], "benchmark pair"),
        _integer(item["selected_n_states"], "benchmark selected K"),
        _string(item["history_sha256"], "benchmark history fingerprint"),
        accelerated,
        ordinary,
        _bool_pair(item["accelerated_valid"], "accelerated validity"),
        _bool_pair(item["ordinary_valid"], "ordinary validity"),
        _float_pair(item["likelihood_difference_per_observation"], "likelihood"),
        _float_pair(item["maximum_component_difference"], "component"),
        _float_pair(item["statistic_difference"], "statistic"),
        _bool_pair(item["component_decisions_identical"], "component decisions"),
        _float(item["elapsed_seconds"], "benchmark elapsed seconds"),
        _string(item["binding_fingerprint"], "benchmark binding fingerprint"),
        _string(item["scientific_fingerprint"], "benchmark scientific fingerprint"),
        _string(item["evidence_fingerprint"], "benchmark evidence fingerprint"),
    )


def _accelerator_dict(item: BridgeAcceleratorDecision) -> dict[str, Any]:
    return {
        "dgp": item.dgp,
        "agreeing_pairs": item.result.agreeing_pairs,
        "audited_pairs": item.result.audited_pairs,
        "eligible": item.result.eligible,
        "pair_agreement": item.result.pair_agreement.astype(bool).tolist(),
        "retained_mode": item.retained_mode,
        "retained_pair_evidence": [
            pair.evidence_fingerprint for pair in item.retained_pairs
        ],
        "scientific_fingerprint": item.scientific_fingerprint,
    }


def _measurement_dict(item: ExecutionResourceMeasurement) -> dict[str, Any]:
    if not isinstance(item, ExecutionResourceMeasurement):
        raise ModelError("bridge benchmark measurement has the wrong type")
    return {
        "workers": item.workers,
        "items": item.items,
        "wall_seconds": item.wall_seconds,
        "peak_process_tree_rss_bytes": item.peak_process_tree_rss_bytes,
        "peak_memory_fraction": item.peak_memory_fraction,
        "swap_used_before_bytes": item.swap_used_before_bytes,
        "swap_used_after_bytes": item.swap_used_after_bytes,
        "maximum_swap_used_bytes": item.maximum_swap_used_bytes,
        "swap_measurement_available": item.swap_measurement_available,
        "process_tree_measurement_available": item.process_tree_measurement_available,
        "preflight": {
            "estimated_peak_bytes": item.preflight.estimated_peak_bytes,
            "physical_memory_bytes": item.preflight.physical_memory_bytes,
            "maximum_fraction": item.preflight.maximum_fraction,
            "estimated_fraction": item.preflight.estimated_fraction,
            "passed": item.preflight.passed,
        },
        "passed": item.passed,
    }


def _measurement_from_dict(value: object) -> ExecutionResourceMeasurement:
    item = _mapping(value, "bridge resource measurement")
    _exact_keys(
        item,
        {
            "workers",
            "items",
            "wall_seconds",
            "peak_process_tree_rss_bytes",
            "peak_memory_fraction",
            "swap_used_before_bytes",
            "swap_used_after_bytes",
            "maximum_swap_used_bytes",
            "swap_measurement_available",
            "process_tree_measurement_available",
            "preflight",
            "passed",
        },
        "bridge resource measurement",
    )
    raw_preflight = _mapping(item["preflight"], "bridge memory preflight")
    _exact_keys(
        raw_preflight,
        {
            "estimated_peak_bytes",
            "physical_memory_bytes",
            "maximum_fraction",
            "estimated_fraction",
            "passed",
        },
        "bridge memory preflight",
    )
    preflight = MemoryPreflight(
        _integer(raw_preflight["estimated_peak_bytes"], "estimated peak bytes"),
        _integer(raw_preflight["physical_memory_bytes"], "physical memory bytes"),
        _float(raw_preflight["maximum_fraction"], "maximum memory fraction"),
        _float(raw_preflight["estimated_fraction"], "estimated memory fraction"),
        _boolean(raw_preflight["passed"], "memory preflight passed"),
    )
    expected_fraction = preflight.estimated_peak_bytes / preflight.physical_memory_bytes
    if (
        preflight.maximum_fraction != 0.70
        or preflight.estimated_fraction != expected_fraction
        or preflight.passed != (expected_fraction <= preflight.maximum_fraction)
    ):
        raise IntegrityError("bridge memory preflight arithmetic is inconsistent")
    result = ExecutionResourceMeasurement(
        _integer(item["workers"], "measurement workers"),
        _integer(item["items"], "measurement items"),
        _float(item["wall_seconds"], "measurement wall seconds"),
        _integer(item["peak_process_tree_rss_bytes"], "measurement peak RSS"),
        _float(item["peak_memory_fraction"], "measurement peak memory fraction"),
        _optional_integer(item["swap_used_before_bytes"], "swap before"),
        _optional_integer(item["swap_used_after_bytes"], "swap after"),
        _optional_integer(item["maximum_swap_used_bytes"], "maximum swap"),
        _boolean(item["swap_measurement_available"], "swap measurement availability"),
        _boolean(
            item["process_tree_measurement_available"],
            "process-tree measurement availability",
        ),
        preflight,
        _boolean(item["passed"], "measurement passed"),
    )
    expected_peak_fraction = (
        result.peak_process_tree_rss_bytes / preflight.physical_memory_bytes
    )
    expected_pass = bool(
        expected_peak_fraction <= preflight.maximum_fraction
        and result.swap_measurement_available
        and result.process_tree_measurement_available
    )
    if (
        result.workers != CANONICAL_WORKERS
        or result.items != len(BRIDGE_DGPS) * ACCELERATOR_AUDIT_PAIRS
        or result.peak_memory_fraction != expected_peak_fraction
        or result.passed != expected_pass
    ):
        raise IntegrityError("bridge resource measurement arithmetic is inconsistent")
    return result


def _projection_dict(item: BridgeResourcePreflight) -> dict[str, Any]:
    if not isinstance(item, BridgeResourcePreflight):
        raise ModelError("bridge resource projection has the wrong type")
    return {
        "benchmark_pairs": item.benchmark_pairs,
        "workers": item.workers,
        "projected_worst_case_hours": item.projected_worst_case_hours,
        "peak_memory_fraction": item.peak_memory_fraction,
        "maximum_hours": item.maximum_hours,
        "maximum_memory_fraction": item.maximum_memory_fraction,
        "passed": item.passed,
        "reasons": list(item.reasons),
    }


def _projection_from_dict(value: object) -> BridgeResourcePreflight:
    item = _mapping(value, "bridge resource projection")
    _exact_keys(
        item,
        {
            "benchmark_pairs",
            "workers",
            "projected_worst_case_hours",
            "peak_memory_fraction",
            "maximum_hours",
            "maximum_memory_fraction",
            "passed",
            "reasons",
        },
        "bridge resource projection",
    )
    return BridgeResourcePreflight(
        _integer(item["benchmark_pairs"], "projection benchmark pairs"),
        _integer(item["workers"], "projection workers"),
        _float(item["projected_worst_case_hours"], "projected hours"),
        _float(item["peak_memory_fraction"], "projected peak memory"),
        _float(item["maximum_hours"], "maximum projected hours"),
        _float(item["maximum_memory_fraction"], "maximum projected memory"),
        _boolean(item["passed"], "projection passed"),
        tuple(_string_sequence(item["reasons"], "projection reasons")),
    )


def _canonical_bridge_from_dict(value: object) -> CanonicalBridgeResourcePreflight:
    item = _mapping(value, "canonical bridge preflight")
    _exact_keys(
        item,
        {
            "schema_version",
            "contract_version",
            "base_preflight_fingerprint",
            "benchmark_pairs",
            "workers",
            "projected_hours",
            "peak_memory_fraction",
            "maximum_hours",
            "maximum_memory_fraction",
            "fingerprint",
        },
        "canonical bridge preflight",
    )
    if item["schema_version"] != 1 or item["contract_version"] != G3_CONTRACT_VERSION:
        raise IntegrityError("canonical bridge preflight schema is unsupported")
    result = CanonicalBridgeResourcePreflight(
        _string(item["base_preflight_fingerprint"], "base preflight fingerprint"),
        _integer(item["benchmark_pairs"], "canonical benchmark pairs"),
        _integer(item["workers"], "canonical workers"),
        _float(item["projected_hours"], "canonical projected hours"),
        _float(item["peak_memory_fraction"], "canonical peak memory"),
        _float(item["maximum_hours"], "canonical maximum hours"),
        _float(item["maximum_memory_fraction"], "canonical maximum memory"),
        _string(item["fingerprint"], "canonical bridge fingerprint"),
    )
    if _sha256_json(result.identity_dict()) != result.fingerprint:
        raise IntegrityError("canonical bridge preflight fingerprint is inconsistent")
    return result


def _interval_dict(item: BinomialInterval) -> dict[str, Any]:
    return {
        "successes": item.successes,
        "trials": item.trials,
        "confidence": item.confidence,
        "estimate": item.estimate,
        "lower": item.lower,
        "upper": item.upper,
    }


def _dgp_result_dict(item: BridgeDGPResult) -> dict[str, Any]:
    return {
        "dgp": item.dgp,
        "tier_a_receipt_fingerprint": (
            None if item.tier_a is None else item.tier_a.receipt_fingerprint
        ),
        "tier_a_passed": None if item.tier_a is None else item.tier_a.passed,
        "accelerator_eligible": item.accelerator_eligible,
        "tier_b_decision": item.tier_b_decision,
        "tier_b_pairs": item.tier_b_pairs,
        "tier_b_passed": item.tier_b_passed,
        "passed": item.passed,
        "scientific_fingerprint": item.scientific_fingerprint,
    }


def _validate_binding(binding: BridgeScientificBinding) -> None:
    if not isinstance(binding, BridgeScientificBinding):
        raise ModelError("bridge scientific binding has the wrong type")
    for label, value in (
        ("base preflight", binding.base_preflight_fingerprint),
        ("design parameters", binding.design_parameters_sha256),
        ("design history", binding.design_history_sha256),
        ("actual bridge evidence", binding.actual_bridge_evidence_fingerprint),
        ("binding", binding.fingerprint),
    ):
        _require_sha256(value, f"{label} fingerprint", ModelError)
    _nonnegative_integer(binding.master_seed, "binding master seed")
    _positive_integer(binding.scientific_version, "binding scientific version")
    if binding.selected_n_states not in {2, 3, 4, 5}:
        raise ModelError("bridge binding selected K is outside 2..5")
    if (
        binding.scientific_version == CANONICAL_SCIENTIFIC_VERSION
        and binding.selected_n_states != BRIDGE_TIER_A_N_STATES
    ):
        raise ModelError("version-3 bridge binding is not fixed at K=4")
    _positive_integer(binding.macro_block_length, "binding macro block length")
    if not math.isfinite(binding.actual_statistic) or binding.actual_statistic < 0.0:
        raise ModelError("bridge binding actual statistic is invalid")
    if (
        tuple(item.role for item in binding.alpha_offset_grids)
        != BRIDGE_ALPHA_GRID_ROLES
    ):
        raise ModelError("bridge binding alpha-grid order is inconsistent")
    if tuple(dgp for dgp, _ in binding.purpose1_trace_fingerprints) != BRIDGE_DGPS:
        raise ModelError("bridge binding purpose-1 trace order is inconsistent")
    for _, fingerprint in binding.purpose1_trace_fingerprints:
        _require_sha256(fingerprint, "purpose-1 trace fingerprint", ModelError)
    if _sha256_json(binding.identity_dict()) != binding.fingerprint:
        raise ModelError("bridge binding fingerprint is inconsistent")


def _design_history_sha256(values: FloatArray, continuation: BoolArray) -> str:
    canonical = np.asarray(values, dtype=np.float64).copy(order="C")
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256(b"PRPG-BRIDGE-DESIGN-HISTORY-v1\0")
    digest.update(np.ascontiguousarray(canonical, dtype="<f8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(continuation, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _purpose1_trace_sha256(values: IntArray) -> str:
    digest = hashlib.sha256(b"PRPG-BRIDGE-ROLLING-BOOTSTRAP-v1\0")
    digest.update(np.ascontiguousarray(values, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(values: object) -> str:
    source = np.asarray(values)
    if source.dtype.kind == "f":
        canonical = np.asarray(source, dtype=np.float64).copy(order="C")
        canonical[canonical == 0.0] = 0.0
        encoded = np.ascontiguousarray(canonical, dtype="<f8")
    elif source.dtype.kind in "iu":
        encoded = np.ascontiguousarray(source, dtype="<i8")
    elif source.dtype.kind == "b":
        encoded = np.ascontiguousarray(source, dtype=np.bool_)
    else:
        raise ModelError("bridge evidence contains an unsupported array dtype")
    digest = hashlib.sha256(b"PRPG-BRIDGE-EVIDENCE-ARRAY-v1\0")
    digest.update(encoded.dtype.str.encode("ascii"))
    digest.update(encoded.ndim.to_bytes(1, "big", signed=False))
    for dimension in encoded.shape:
        digest.update(int(dimension).to_bytes(8, "big", signed=False))
    digest.update(encoded.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise ModelError("bridge checkpoint content is not canonical JSON") from error


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_or_verify_exact(path: Path, content: bytes) -> bool:
    if path.exists():
        if _read_regular(path) != content:
            raise IntegrityError(
                "existing bridge checkpoint differs from requested bytes"
            )
        return True
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if _read_regular(path) != content:
                raise IntegrityError(
                    "concurrent bridge checkpoint differs from requested bytes"
                ) from error
            return True
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return False
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_regular(path: Path) -> bytes:
    _reject_symlink(path, "bridge checkpoint file")
    try:
        metadata = path.stat()
    except OSError as error:
        raise IntegrityError("bridge checkpoint file cannot be inspected") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise IntegrityError("bridge checkpoint path is not a regular file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise IntegrityError("bridge checkpoint file cannot be read") from error


def _reject_symlink(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise IntegrityError(f"{label} must not be a symbolic link")
    except OSError as error:
        raise IntegrityError(f"{label} cannot be inspected") from error


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelError(f"{label} must be a string-keyed mapping")
    # The canonical round trip both normalizes MappingProxyType and rejects
    # NumPy scalars, tuples with unsupported members, NaN, and infinities.
    try:
        decoded = json.loads(_canonical_bytes(dict(value)))
    except json.JSONDecodeError as error:  # pragma: no cover - generated JSON invariant
        raise AssertionError("canonical bridge JSON failed to decode") from error
    if not isinstance(decoded, dict):  # pragma: no cover - mapping input invariant
        raise AssertionError("canonical bridge mapping decoded as a non-object")
    return decoded


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise IntegrityError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise IntegrityError(f"{label} must be a JSON array")
    return tuple(value)


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    sequence = _sequence(value, label)
    if any(not isinstance(item, str) for item in sequence):
        raise IntegrityError(f"{label} must contain only strings")
    return tuple(item for item in sequence if isinstance(item, str))


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise IntegrityError(f"{label} has unexpected or missing fields")


def _dgp(value: object) -> BridgeDGP:
    if value == "model":
        return "model"
    if value == "nonparametric":
        return "nonparametric"
    raise IntegrityError("bridge DGP is outside the closed registry")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise IntegrityError(f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise IntegrityError(f"{label} must be boolean")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IntegrityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise IntegrityError(f"{label} must be finite")
    return result


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _float(value, label)


def _bool_pair(value: object, label: str) -> tuple[bool, bool]:
    sequence = _sequence(value, label)
    if len(sequence) != 2:
        raise IntegrityError(f"{label} must contain exactly two values")
    return (_boolean(sequence[0], label), _boolean(sequence[1], label))


def _float_pair(value: object, label: str) -> tuple[float, float]:
    sequence = _sequence(value, label)
    if len(sequence) != 2:
        raise IntegrityError(f"{label} must contain exactly two values")
    return (_float(sequence[0], label), _float(sequence[1], label))


def _require_sha256(
    value: object,
    label: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise error_type(f"{label} is invalid")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelError(f"{label} must be a non-negative integer")
    return value
