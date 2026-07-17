"""Closed all-success execution of canonical G3 bridge tasks 20 through 25.

This module is the final scientific stage before artifact integrity.  It
consumes the exact passed design and production stage results, constructs the
actual-artifact bridge only from their retained parents, and executes one
registered staged bridge session in task order.  Every task is bound through
its task-specific prebinding API, converted to a guarded boundary view, and
published against the same active launch.

Only the frozen-scope vertical slice is implemented.  There is no callback
map, caller-authored bridge evidence, reduced geometry, recovery dispatcher,
G4/G5 work, or CSV machinery.  The staged bridge keeps its own intrinsic
append-only receipts; unexpected operational and programming exceptions are
never converted into scientific outcomes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from prpg.errors import ModelError
from prpg.model.alpha import AlphaOffsetGrid
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    RegisteredBlockCoreIdentity,
    registered_block_core_identity,
)
from prpg.model.bridge_driver import (
    BRIDGE_ALPHA_GRID_ROLES,
    CanonicalBridgeRequest,
    CanonicalStagedBridgeSession,
    StagedBridgeTaskEvidence,
    complete_staged_bridge_tier_b_task,
    execute_staged_actual_artifact_bridge,
    execute_staged_bridge_resource_benchmark,
    execute_staged_bridge_tier_a,
    execute_staged_bridge_tier_b_next_look,
    prepare_canonical_staged_bridge,
    reverify_staged_bridge_prefix,
)
from prpg.model.bridge_fixed_k import ActualBridgeExecution, assemble_actual_bridge
from prpg.model.bridge_history import BridgeDGP
from prpg.model.calibration import registered_macro_stability_execution_identity
from prpg.model.g3_boundary_views import (
    BridgeDriverTaskView,
    BridgeDriverTaskViews,
    BridgeTaskId,
    build_bridge_driver_task_view,
)
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_planned_look_execution import (
    PlannedLookParityIdentity,
    RegisteredPlannedLookPrefix,
    publish_raw_planned_look_prefix,
    verify_planned_look_binding_view_parity,
)
from prpg.model.g3_preflight import (
    CANONICAL_WORKERS,
    CanonicalG3Preflight,
    verify_compatible_canonical_g3_preflights,
)
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_success_production_stage import CanonicalG3ProductionStageResult
from prpg.model.g3_success_runner import CanonicalG3DesignStageResult
from prpg.model.g3_task_binding import (
    ScientificTaskBinding,
    build_scientific_task_binding,
)
from prpg.model.g3_task_publication import (
    ScientificTaskPublication,
    ScientificTaskSourceSlots,
    bridge_executor_source_slots,
    publish_exact_next_scientific_task,
)
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    kernel_calibration_execution_identity,
)
from prpg.model.run_store import CalibrationRunStore, LaunchLease

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

_PREFIX_TASKS: Final = G3_GATE_ORDER[:19]
_BRIDGE_TASKS: Final[
    tuple[
        BridgeTaskId,
        BridgeTaskId,
        BridgeTaskId,
        BridgeTaskId,
        BridgeTaskId,
        BridgeTaskId,
    ]
] = (
    "bridge_resource_preflight",
    "bridge_tier_a_model_dgp",
    "bridge_tier_a_nonparametric_dgp",
    "bridge_tier_b_model_dgp",
    "bridge_tier_b_nonparametric_dgp",
    "actual_artifact_bridge",
)
_BRIDGE_CHECKPOINT_DIRECTORY: Final = "canonical-bridge-execution-v1"


@dataclass(frozen=True, slots=True)
class CanonicalG3BridgeStageResult:
    """Exact passed publications and retained bridge parents for task 26."""

    publications: tuple[ScientificTaskPublication, ...]
    actual_bridge: ActualBridgeExecution
    request: CanonicalBridgeRequest
    session: CanonicalStagedBridgeSession
    task_evidence: tuple[StagedBridgeTaskEvidence, ...]
    views: BridgeDriverTaskViews
    model_tier_b_planned_look_prefix: RegisteredPlannedLookPrefix
    model_tier_b_planned_look_parity: PlannedLookParityIdentity
    nonparametric_tier_b_planned_look_prefix: RegisteredPlannedLookPrefix
    nonparametric_tier_b_planned_look_parity: PlannedLookParityIdentity

    @property
    def generation_authorized(self) -> Literal[False]:
        """Tasks 20--25 remain non-authorizing intermediate G3 evidence."""

        return False


@dataclass(frozen=True, slots=True)
class _StageContext:
    run_store: CalibrationRunStore
    checkpoint_store: CalibrationCheckpointStore
    lease: LaunchLease
    live_preflight: CanonicalG3Preflight

    @property
    def run_id(self) -> str:
        return self.lease.run_id


@dataclass(frozen=True, slots=True)
class _PublishedBridgeTask:
    evidence: StagedBridgeTaskEvidence
    view: BridgeDriverTaskView
    publication: ScientificTaskPublication


@dataclass(frozen=True, slots=True)
class _PublishedTierBTask:
    task: _PublishedBridgeTask
    planned_look_prefix: RegisteredPlannedLookPrefix
    planned_look_parity: PlannedLookParityIdentity


def run_canonical_g3_bridge_stage(
    design_stage: CanonicalG3DesignStageResult,
    production_stage: CanonicalG3ProductionStageResult,
    *,
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    lease: LaunchLease,
    master_seed: int,
    scientific_version: int,
) -> CanonicalG3BridgeStageResult:
    """Compute, bind, and publish canonical G3 tasks 20 through 25 once."""

    _verify_stage_inputs(
        design_stage,
        production_stage,
        launch_preflight=launch_preflight,
        live_preflight=live_preflight,
        run_store=run_store,
        lease=lease,
    )
    context = _StageContext(run_store, checkpoint_store, lease, live_preflight)
    actual_bridge = _assemble_actual_bridge_from_stages(
        design_stage,
        production_stage,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    request = _canonical_bridge_request(
        design_stage,
        production_stage,
        actual_bridge=actual_bridge,
        live_preflight=live_preflight,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    checkpoint_root = (
        run_store.verify(context.run_id).path / _BRIDGE_CHECKPOINT_DIRECTORY
    )
    session = prepare_canonical_staged_bridge(request, checkpoint_root)

    # Task 20: measured resource projection under the canonical 72-hour bound.
    resource = _bind_view_publish(
        context,
        execute_staged_bridge_resource_benchmark(session),
        task_id="bridge_resource_preflight",
    )

    # Tasks 21--22: both exact Tier-A reference-law gates, in registry order.
    tier_a_model = _bind_view_publish(
        context,
        execute_staged_bridge_tier_a(session, "model"),
        task_id="bridge_tier_a_model_dgp",
    )
    tier_a_nonparametric = _bind_view_publish(
        context,
        execute_staged_bridge_tier_a(session, "nonparametric"),
        task_id="bridge_tier_a_nonparametric_dgp",
    )

    # Tasks 23--24: run registered looks until the staged state is terminal.
    tier_b_model = _execute_tier_b_success(
        context,
        session,
        dgp="model",
        task_id="bridge_tier_b_model_dgp",
    )
    tier_b_nonparametric = _execute_tier_b_success(
        context,
        session,
        dgp="nonparametric",
        task_id="bridge_tier_b_nonparametric_dgp",
    )

    # Task 25: RNG-empty binding of the actual design/production artifact pair.
    actual = _bind_view_publish(
        context,
        execute_staged_actual_artifact_bridge(session),
        task_id="actual_artifact_bridge",
    )

    tasks = (
        resource,
        tier_a_model,
        tier_a_nonparametric,
        tier_b_model.task,
        tier_b_nonparametric.task,
        actual,
    )
    publications = tuple(item.publication for item in tasks)
    evidence = tuple(item.evidence for item in tasks)
    views = BridgeDriverTaskViews(
        bridge_resource_preflight=resource.view,
        bridge_tier_a_model_dgp=tier_a_model.view,
        bridge_tier_a_nonparametric_dgp=tier_a_nonparametric.view,
        bridge_tier_b_model_dgp=tier_b_model.task.view,
        bridge_tier_b_nonparametric_dgp=tier_b_nonparametric.task.view,
        actual_artifact_bridge=actual.view,
    )
    _verify_staged_completion(session, evidence)
    _verify_returned_publications(context, publications)
    return CanonicalG3BridgeStageResult(
        publications=publications,
        actual_bridge=actual_bridge,
        request=request,
        session=session,
        task_evidence=evidence,
        views=views,
        model_tier_b_planned_look_prefix=(tier_b_model.planned_look_prefix),
        model_tier_b_planned_look_parity=(tier_b_model.planned_look_parity),
        nonparametric_tier_b_planned_look_prefix=(
            tier_b_nonparametric.planned_look_prefix
        ),
        nonparametric_tier_b_planned_look_parity=(
            tier_b_nonparametric.planned_look_parity
        ),
    )


def _verify_stage_inputs(
    design_stage: CanonicalG3DesignStageResult,
    production_stage: CanonicalG3ProductionStageResult,
    *,
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
) -> None:
    if not isinstance(design_stage, CanonicalG3DesignStageResult):
        raise ModelError("bridge stage requires CanonicalG3DesignStageResult")
    if not isinstance(production_stage, CanonicalG3ProductionStageResult):
        raise ModelError("bridge stage requires CanonicalG3ProductionStageResult")
    if production_stage.design_stage is not design_stage:
        raise ModelError(
            "bridge production result belongs to a different exact design stage"
        )
    launch_fingerprint = verify_compatible_canonical_g3_preflights(
        launch_preflight,
        live_preflight,
    )
    # Final G3 assembly cross-binds every bridge view to task 1's launch
    # preflight fingerprint.  This vertical slice is fresh-launch-only, so it
    # requires the original live-authorized launch object instead of allowing
    # a later disk-capacity observation with a compatibility-only fingerprint.
    if live_preflight.fingerprint != launch_fingerprint:
        raise ModelError(
            "fresh bridge stage requires the live-authorized launch preflight"
        )
    if (
        launch_preflight.workers != CANONICAL_WORKERS
        or live_preflight.workers != CANONICAL_WORKERS
    ):
        raise ModelError("canonical bridge stage requires exactly nine workers")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not run_store
        or not lease.active
        or lease.preflight_fingerprint != launch_fingerprint
        or lease.workers != CANONICAL_WORKERS
    ):
        raise ModelError("bridge stage requires the same active canonical launch")
    marker = run_store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != launch_fingerprint
        or marker.workers != CANONICAL_WORKERS
    ):
        raise ModelError("bridge stage launch marker is not the active G3 launch")

    supplied = design_stage.publications + production_stage.publications
    task_ids = tuple(item.binding.task_id for item in supplied)
    if task_ids != _PREFIX_TASKS:
        raise ModelError("bridge stage requires the exact tasks 1--19 prefix")
    if any(
        item.binding.run_id != lease.run_id
        or not item.decision.passed
        or item.checkpoint is None
        or item.task_result.status != "passed"
        for item in supplied
    ):
        raise ModelError("bridge-stage prefix contains a non-pass or another run")
    durable = run_store.verify_task_results(lease.run_id, require_complete=False)
    if tuple(durable) != _PREFIX_TASKS or any(
        durable[item.binding.task_id].fingerprint != item.task_result.fingerprint
        for item in supplied
    ):
        raise ModelError("durable tasks 1--19 differ from the supplied stage results")


def _assemble_actual_bridge_from_stages(
    design_stage: CanonicalG3DesignStageResult,
    production_stage: CanonicalG3ProductionStageResult,
    *,
    master_seed: int,
    scientific_version: int,
) -> ActualBridgeExecution:
    design_blocks = _verified_block_pair(
        design_stage.block_parents,
        role="design",
        selected_k=design_stage.design_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    production_blocks = _verified_block_pair(
        production_stage.block_parents,
        role="production",
        selected_k=production_stage.production_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    design_monthly_grid = _verified_kernel_grid(
        design_stage.monthly_kernel,
        design_stage.monthly_kernel_grid,
        role="design",
        frequency="monthly",
        selected_length=design_blocks[0].selected_length,
        selected_k=design_stage.design_fit.n_states,
        core_model_fingerprint=design_stage.core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    design_daily_grid = _verified_kernel_grid(
        design_stage.daily_kernel,
        design_stage.daily_kernel_grid,
        role="design",
        frequency="daily",
        selected_length=design_blocks[1].selected_length,
        selected_k=design_stage.design_fit.n_states,
        core_model_fingerprint=design_stage.core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    production_monthly_grid = _verified_kernel_grid(
        production_stage.monthly_kernel,
        production_stage.monthly_kernel_grid,
        role="production",
        frequency="monthly",
        selected_length=production_blocks[0].selected_length,
        selected_k=production_stage.production_fit.n_states,
        core_model_fingerprint=production_stage.core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    production_daily_grid = _verified_kernel_grid(
        production_stage.daily_kernel,
        production_stage.daily_kernel_grid,
        role="production",
        frequency="daily",
        selected_length=production_blocks[1].selected_length,
        selected_k=production_stage.production_fit.n_states,
        core_model_fingerprint=production_stage.core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    design_k = design_stage.design_fit.n_states
    production_k = production_stage.production_fit.n_states
    if design_k != production_k:
        raise ModelError("actual bridge requires production to retain design K")
    return assemble_actual_bridge(
        design_fit=design_stage.design_fit,
        production_fit=production_stage.production_fit,
        design_monthly_block_length=design_blocks[0].selected_length,
        production_monthly_block_length=production_blocks[0].selected_length,
        design_daily_block_length=design_blocks[1].selected_length,
        production_daily_block_length=production_blocks[1].selected_length,
        design_monthly_state_counts=_state_counts(
            design_stage.decoded_return_pools.monthly_states,
            n_states=design_k,
            label="design monthly decoded pools",
        ),
        production_monthly_state_counts=_state_counts(
            production_stage.decoded_pools.monthly_states,
            n_states=production_k,
            label="production monthly decoded pools",
        ),
        design_daily_state_counts=_state_counts(
            design_stage.decoded_return_pools.daily_states,
            n_states=design_k,
            label="design daily decoded pools",
        ),
        production_daily_state_counts=_state_counts(
            production_stage.decoded_pools.daily_states,
            n_states=production_k,
            label="production daily decoded pools",
        ),
        design_monthly_kernel_offsets=design_monthly_grid.offsets,
        production_monthly_kernel_offsets=production_monthly_grid.offsets,
        design_daily_kernel_offsets=design_daily_grid.offsets,
        production_daily_kernel_offsets=production_daily_grid.offsets,
    )


def _canonical_bridge_request(
    design_stage: CanonicalG3DesignStageResult,
    production_stage: CanonicalG3ProductionStageResult,
    *,
    actual_bridge: ActualBridgeExecution,
    live_preflight: CanonicalG3Preflight,
    master_seed: int,
    scientific_version: int,
) -> CanonicalBridgeRequest:
    design_macro = registered_macro_stability_execution_identity(
        design_stage.macro_evidence
    )
    if (
        design_macro.role != "design"
        or design_macro.n_states != design_stage.design_fit.n_states
        or design_macro.master_seed != master_seed
        or design_macro.scientific_version != scientific_version
    ):
        raise ModelError("bridge design macro evidence has different exact parents")
    features = design_stage.design_features
    history = np.asarray(features.values)
    if (
        history.shape != (193, 4)
        or history.dtype != np.dtype(np.float64)
        or history.flags.writeable
        or not bool(np.isfinite(history).all())
        or features.lengths != (193,)
        or design_stage.design_fit.lengths != features.lengths
    ):
        raise ModelError("bridge design feature history is not exact canonical input")
    alpha_grids = _ordered_alpha_grids(design_stage, production_stage)
    return CanonicalBridgeRequest(
        base_preflight=live_preflight,
        master_seed=master_seed,
        scientific_version=scientific_version,
        design_parameters=design_stage.design_fit.parameters,
        design_standardized_history=history,
        design_continuation=_gap_free_continuation(len(history)),
        macro_block_length=design_stage.macro_evidence.macro_block_length,
        actual_bridge=actual_bridge,
        alpha_offset_grids=alpha_grids,
    )


def _verified_block_pair(
    blocks: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
    *,
    role: Literal["design", "production"],
    selected_k: int,
    master_seed: int,
    scientific_version: int,
) -> tuple[RegisteredBlockCoreIdentity, RegisteredBlockCoreIdentity]:
    if len(blocks) != 2:
        raise ModelError(f"{role} bridge parents require monthly and daily blocks")
    identities = tuple(registered_block_core_identity(item) for item in blocks)
    expected_frequencies = ("monthly", "daily")
    if any(
        identity.artifact != role
        or identity.frequency != frequency
        or identity.selected_k != selected_k
        or identity.master_seed != master_seed
        or identity.scientific_version != scientific_version
        for identity, frequency in zip(
            identities,
            expected_frequencies,
            strict=True,
        )
    ):
        raise ModelError(f"{role} bridge blocks have different exact parents")
    return identities[0], identities[1]


def _verified_kernel_grid(
    kernel: KernelCalibrationExecution,
    grid: AlphaOffsetGrid,
    *,
    role: Literal["design", "production"],
    frequency: Literal["monthly", "daily"],
    selected_length: int,
    selected_k: int,
    core_model_fingerprint: str,
    master_seed: int,
    scientific_version: int,
) -> AlphaOffsetGrid:
    identity = kernel_calibration_execution_identity(kernel)
    if (
        getattr(kernel, "offset_grid", None) is not grid
        or not getattr(kernel, "passed", False)
        or identity.artifact != role
        or identity.frequency != frequency
        or identity.selected_length != selected_length
        or identity.n_states != selected_k
        or identity.master_seed != master_seed
        or identity.scientific_version != scientific_version
        or grid.frequency != frequency
        or grid.model_artifact_fingerprint != core_model_fingerprint
    ):
        raise ModelError(f"{role} {frequency} kernel grid changed exact ancestry")
    return grid


def _ordered_alpha_grids(
    design_stage: CanonicalG3DesignStageResult,
    production_stage: CanonicalG3ProductionStageResult,
) -> tuple[AlphaOffsetGrid, AlphaOffsetGrid, AlphaOffsetGrid, AlphaOffsetGrid]:
    if BRIDGE_ALPHA_GRID_ROLES != (
        "design_monthly",
        "design_daily",
        "production_monthly",
        "production_daily",
    ):
        raise ModelError("bridge alpha-grid registry order changed")
    grids = (
        design_stage.monthly_kernel_grid,
        design_stage.daily_kernel_grid,
        production_stage.monthly_kernel_grid,
        production_stage.daily_kernel_grid,
    )
    expected_frequencies = ("monthly", "daily", "monthly", "daily")
    if any(
        grid.frequency != frequency
        for grid, frequency in zip(grids, expected_frequencies, strict=True)
    ):
        raise ModelError("bridge alpha grids are not in registered role order")
    return grids


def _state_counts(states: np.ndarray, *, n_states: int, label: str) -> IntArray:
    values = np.asarray(states)
    if (
        values.ndim != 1
        or values.size == 0
        or values.dtype != np.dtype(np.int64)
        or values.flags.writeable
        or bool(((values < 0) | (values >= n_states)).any())
    ):
        raise ModelError(f"{label} states are invalid")
    counts = np.asarray(np.bincount(values, minlength=n_states), dtype=np.int64)
    return np.frombuffer(counts.tobytes(order="C"), dtype=np.dtype("<i8"))


def _gap_free_continuation(rows: int) -> BoolArray:
    if rows != 193:
        raise ModelError("canonical bridge continuation requires 193 design rows")
    continuation = np.ones(rows, dtype=np.bool_)
    continuation[0] = False
    return np.frombuffer(continuation.tobytes(order="C"), dtype=np.dtype(np.bool_))


def _bind_view_publish(
    context: _StageContext,
    evidence: StagedBridgeTaskEvidence,
    *,
    task_id: BridgeTaskId,
) -> _PublishedBridgeTask:
    slots = bridge_executor_source_slots(evidence, task_id=task_id)
    binding = _build_binding(context, slots)
    view = build_bridge_driver_task_view(
        evidence,
        task_id=task_id,
        run_id=context.run_id,
        binding_fingerprint=binding.fingerprint,
    )
    publication = _publish_pass(context, binding, view)
    return _PublishedBridgeTask(evidence, view, publication)


def _execute_tier_b_success(
    context: _StageContext,
    session: CanonicalStagedBridgeSession,
    *,
    dgp: BridgeDGP,
    task_id: Literal[
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    ],
) -> _PublishedTierBTask:
    while True:
        look = execute_staged_bridge_tier_b_next_look(session, dgp)
        if look.decision != "continue":
            break
    evidence = complete_staged_bridge_tier_b_task(session, dgp)
    planned_look_prefix = publish_raw_planned_look_prefix(
        store=context.run_store,
        lease=context.lease,
        task_id=task_id,
        source=evidence,
    )
    slots = bridge_executor_source_slots(evidence, task_id=task_id)
    binding = _build_binding(context, slots)
    view = build_bridge_driver_task_view(
        evidence,
        task_id=task_id,
        run_id=context.run_id,
        binding_fingerprint=binding.fingerprint,
    )
    parity = verify_planned_look_binding_view_parity(
        planned_look_prefix,
        binding=binding,
        final_view=view,
        checkpoint_store=context.checkpoint_store,
        live_preflight=context.live_preflight,
    )
    publication = _publish_pass(context, binding, view)
    return _PublishedTierBTask(
        _PublishedBridgeTask(evidence, view, publication),
        planned_look_prefix,
        parity,
    )


def _build_binding(
    context: _StageContext,
    slots: ScientificTaskSourceSlots,
) -> ScientificTaskBinding:
    return build_scientific_task_binding(
        run_store=context.run_store,
        checkpoint_store=context.checkpoint_store,
        run_id=context.run_id,
        task_id=slots.task_id,
        live_preflight=context.live_preflight,
        scientific_component_fingerprints=dict(slots.scientific_component_fingerprints),
        materialization_fingerprints=dict(slots.materialization_fingerprints),
        rng_fingerprints=dict(slots.rng_fingerprints),
    )


def _publish_pass(
    context: _StageContext,
    binding: ScientificTaskBinding,
    source: BridgeDriverTaskView,
) -> ScientificTaskPublication:
    publication = publish_exact_next_scientific_task(
        run_store=context.run_store,
        lease=context.lease,
        checkpoint_store=context.checkpoint_store,
        live_preflight=context.live_preflight,
        binding=binding,
        source=source,
    )
    if (
        not publication.decision.passed
        or publication.checkpoint is None
        or publication.task_result.status != "passed"
    ):
        raise ModelError(
            "canonical bridge-stage task did not pass and checkpoint",
            details={"task_id": publication.binding.task_id},
        )
    return publication


def _verify_staged_completion(
    session: CanonicalStagedBridgeSession,
    evidence: tuple[StagedBridgeTaskEvidence, ...],
) -> None:
    prefix = reverify_staged_bridge_prefix(session)
    if (
        not prefix.terminal
        or prefix.next_action is not None
        or tuple(item.task_id for item in prefix.task_evidence) != _BRIDGE_TASKS
        or tuple(item.evidence_fingerprint for item in prefix.task_evidence)
        != tuple(item.evidence_fingerprint for item in evidence)
    ):
        raise ModelError("staged bridge did not close its exact six-task prefix")


def _verify_returned_publications(
    context: _StageContext,
    publications: tuple[ScientificTaskPublication, ...],
) -> None:
    if tuple(item.binding.task_id for item in publications) != _BRIDGE_TASKS:
        raise ModelError("canonical bridge-stage publication order is incomplete")
    if any(
        not item.decision.passed
        or item.checkpoint is None
        or item.task_result.status != "passed"
        or item.binding.run_id != context.run_id
        for item in publications
    ):
        raise ModelError("canonical bridge-stage result contains a non-pass")
    durable = context.run_store.verify_task_results(
        context.run_id,
        require_complete=False,
    )
    if tuple(durable) != G3_GATE_ORDER[:25] or any(
        durable[item.binding.task_id].fingerprint != item.task_result.fingerprint
        for item in publications
    ):
        raise ModelError("durable task prefix differs from tasks 20--25 result")
