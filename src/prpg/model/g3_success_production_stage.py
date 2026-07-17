"""Closed all-success execution of canonical G3 tasks 11 through 19.

This stage consumes the exact passed design-stage result and continues the
same initialized launch.  It owns the single production fit, production
calibration families, production block/dependence evidence, and the two
four-cell Holm reductions.  There is no handler map, callback, recovery path,
reduced geometry, or caller-authored scientific decision.

Only the all-success vertical slice is implemented here.  A registered
scientific nonattainment that lacks a launch-critical publication variant
aborts the stage, and every unexpected exception escapes unchanged.  Such an
abort cannot produce a task-19 success result or a later G3 authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from prpg.errors import ModelError
from prpg.model.adequacy_execution import (
    RegisteredLatentCell,
    run_registered_latent_cell,
)
from prpg.model.adequacy_gate import (
    AdequacyCellTaskView,
    FourCellAdequacyHolmView,
    assemble_four_cell_adequacy_holm_view,
    build_latent_adequacy_task_view,
    build_refitted_adequacy_task_view,
)
from prpg.model.alpha import AlphaOffsetGrid, BaseFrequency
from prpg.model.block_execution import CANONICAL_HISTORY_COUNT
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    RegisteredBlockReplaySource,
    RegisteredBlockUniformMaterialization,
    materialize_registered_block_uniforms,
    registered_block_replay_source,
    run_registered_materialized_block_calibration_outcome,
)
from prpg.model.calibration import (
    DecodedReturnPools,
    MacroStabilityEvidence,
    PreliminaryCalibrations,
    RegisteredMacroBootstrapMaterialization,
    RegisteredPreliminaryCalibrationOutcome,
    decoded_return_pools,
    execute_registered_macro_stability,
    execute_registered_preliminary_calibrations,
    materialize_registered_macro_bootstrap,
    prepare_hmm_feature_matrix,
    registered_preliminary_calibrations,
)
from prpg.model.dependence import DependenceReference, fit_dependence_reference
from prpg.model.dependence_gate_execution import (
    DependenceGateCellInput,
    DependenceTaskView,
    FourCellDependenceHolmView,
    RegisteredArtifactDependenceExecution,
    assemble_four_cell_dependence_holm_view,
    build_fragmentation_task_view,
    build_registered_dependence_task_view,
    execute_registered_artifact_dependence_role,
)
from prpg.model.g3_boundary_views import (
    BlockCalibrationGateEvidence,
    BlockCalibrationTaskView,
    build_block_calibration_task_view,
)
from prpg.model.g3_calibration_views import (
    MacroStabilityTaskView,
    ProductionFitSameKView,
    build_macro_stability_task_view,
    build_production_fit_same_k_view,
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
from prpg.model.g3_success_runner import CanonicalG3DesignStageResult
from prpg.model.g3_task_binding import (
    ScientificTaskBinding,
    build_scientific_task_binding,
)
from prpg.model.g3_task_publication import (
    ScientificTaskPublication,
    ScientificTaskSourceSlots,
    adequacy_cell_executor_source_slots,
    adequacy_holm_executor_source_slots,
    block_calibration_executor_source_slots,
    dependence_executor_source_slots,
    dependence_holm_executor_source_slots,
    fragmentation_executor_source_slots,
    macro_stability_executor_source_slots,
    production_fit_executor_source_slots,
    publish_exact_next_scientific_task,
)
from prpg.model.hmm import GaussianHMMFit, HMMFeatureMatrix, StatePathDiagnostic
from prpg.model.input import CalibrationInput, CalibrationSlice
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    run_registered_kernel_calibration_cell,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.production_fit_execution import (
    RegisteredProductionFitSameK,
    execute_registered_production_fit_same_k,
    is_registered_production_fit_same_k,
    registered_production_fit_same_k_sources,
)
from prpg.model.refit_adequacy_execution import (
    RegisteredRefittedAdequacyBatch,
    run_registered_refitted_adequacy_cell,
)
from prpg.model.regime_policy import (
    OwnerFixedK4RecurringHistoryFailure,
    owner_fixed_k4_recurring_history_support,
)
from prpg.model.registered_fixed_point import registered_design_fixed_point_identity
from prpg.model.run_store import CalibrationRunStore, LaunchLease
from prpg.model.scientific_artifact import scientific_core_model_fingerprint
from prpg.model.transition_bootstrap import (
    TransitionBootstrapAggregate,
    aggregate_transition_bootstrap,
)
from prpg.simulation.rng import CalibrationArtifact

_DESIGN_TASKS: Final = G3_GATE_ORDER[:10]
_PRODUCTION_TASKS: Final = G3_GATE_ORDER[10:19]


@dataclass(frozen=True, slots=True)
class CanonicalG3ProductionStageResult:
    """Exact tasks 11--19 publications and retained downstream parents."""

    publications: tuple[ScientificTaskPublication, ...]
    design_stage: CanonicalG3DesignStageResult
    production_fit_execution: RegisteredProductionFitSameK
    production_fit_view: ProductionFitSameKView
    production_fit: GaussianHMMFit
    production_features: HMMFeatureMatrix
    preliminary_outcome: RegisteredPreliminaryCalibrationOutcome
    preliminary: PreliminaryCalibrations
    macro_materialization: RegisteredMacroBootstrapMaterialization
    macro_evidence: MacroStabilityEvidence
    macro_stability: MacroStabilityTaskView
    latent_evidence: RegisteredLatentCell
    latent_correctness: AdequacyCellTaskView
    refitted_evidence: RegisteredRefittedAdequacyBatch
    refitted_adequacy: AdequacyCellTaskView
    decoded_pools: DecodedReturnPools
    monthly_dependence_reference: DependenceReference
    daily_dependence_reference: DependenceReference
    core_model_fingerprint: str
    block_materializations: tuple[
        RegisteredBlockUniformMaterialization,
        RegisteredBlockUniformMaterialization,
    ]
    block_parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration]
    monthly_kernel: KernelCalibrationExecution
    daily_kernel: KernelCalibrationExecution
    monthly_kernel_grid: AlphaOffsetGrid
    daily_kernel_grid: AlphaOffsetGrid
    transition: TransitionBootstrapAggregate
    block_evidence: BlockCalibrationGateEvidence
    block_calibration: BlockCalibrationTaskView
    block_planned_look_prefix: RegisteredPlannedLookPrefix
    block_planned_look_parity: PlannedLookParityIdentity
    fragmentation: DependenceTaskView
    dependence_execution: RegisteredArtifactDependenceExecution
    dependence: DependenceTaskView
    adequacy_holm: FourCellAdequacyHolmView
    dependence_holm: FourCellDependenceHolmView

    @property
    def generation_authorized(self) -> bool:
        """Tasks 11--19 are intermediate G3 evidence only."""

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


def run_canonical_g3_production_stage(
    design_stage: CanonicalG3DesignStageResult,
    calibration_input: CalibrationInput,
    *,
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    lease: LaunchLease,
    master_seed: int,
    scientific_version: int,
) -> CanonicalG3ProductionStageResult:
    """Compute, bind, and publish canonical G3 tasks 11 through 19."""

    _verify_stage_inputs(
        design_stage,
        calibration_input,
        launch_preflight=launch_preflight,
        live_preflight=live_preflight,
        run_store=run_store,
        lease=lease,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    context = _StageContext(run_store, checkpoint_store, lease, live_preflight)
    publications: list[ScientificTaskPublication] = []

    # Task 11: exactly one full-history fixed-K production fit.
    production_features = _prepare_production_features(calibration_input)
    raw_production_fit = execute_registered_production_fit_same_k(
        design_stage.candidate_selection,
        production_features,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
    )
    if not is_registered_production_fit_same_k(raw_production_fit):
        raise ModelError(
            "canonical G3 success path requires a successful production fit"
        )
    production_execution = raw_production_fit
    selection_parent, design_fit, retained_features, production_fit = (
        registered_production_fit_same_k_sources(production_execution)
    )
    if (
        selection_parent is not design_stage.candidate_selection
        or design_fit is not design_stage.design_fit
        or retained_features is not production_features
    ):
        raise ModelError("production fit changed its exact design/full-data parents")
    decoded_pools = decoded_return_pools(production_fit, calibration_input.full)
    _require_production_recurring_history_support(decoded_pools)
    production_slots = production_fit_executor_source_slots(production_execution)
    production_binding = _build_binding(context, production_slots)
    production_view = build_production_fit_same_k_view(
        production_execution,
        run_id=context.run_id,
        binding_fingerprint=production_binding.fingerprint,
    )
    publications.append(_publish_pass(context, production_binding, production_view))

    # Task 12: one production PPW parent and one common macro source/family.
    preliminary_outcome = execute_registered_preliminary_calibrations(
        calibration_input.full,
        role="production",
    )
    preliminary = registered_preliminary_calibrations(preliminary_outcome)
    macro_materialization = materialize_registered_macro_bootstrap(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="production",
        macro_block_length=preliminary.macro.combined.starting_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    macro_evidence = execute_registered_macro_stability(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="production",
        main_fit=production_fit,
        macro_block_length=preliminary.macro.combined.starting_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
        source_materialization=macro_materialization,
        workers=live_preflight.workers,
    )
    macro_slots = macro_stability_executor_source_slots(
        task_id="production_macro_stability",
        evidence=macro_evidence,
        parent=production_view,
        source_slice=calibration_input.full,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    macro_binding = _build_binding(context, macro_slots)
    macro_view = build_macro_stability_task_view(
        task_id="production_macro_stability",
        evidence=macro_evidence,
        parent=production_view,
        source_slice=calibration_input.full,
        master_seed=master_seed,
        scientific_version=scientific_version,
        run_id=context.run_id,
        binding_fingerprint=macro_binding.fingerprint,
    )
    publications.append(_publish_pass(context, macro_binding, macro_view))

    # Task 13: the existing closed vectorized latent producer.
    latent_evidence = run_registered_latent_cell(
        production_fit,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    )
    latent_slots = adequacy_cell_executor_source_slots(
        latent_evidence,
        production_view,
        task_id="production_latent_correctness",
        strict_canonical=True,
    )
    latent_binding = _build_binding(context, latent_slots)
    latent_view = build_latent_adequacy_task_view(
        task_id="production_latent_correctness",
        evidence=latent_evidence,
        parent=production_view,
        run_id=context.run_id,
        binding_fingerprint=latent_binding.fingerprint,
        strict_canonical=True,
    )
    publications.append(_publish_pass(context, latent_binding, latent_view))

    # Task 14: fixed-denominator refitted adequacy on the nine-worker outer map.
    refitted_evidence = run_registered_refitted_adequacy_cell(
        role="production",
        original_fit=production_fit,
        artifact_features=production_features,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
        strict_canonical=True,
    )
    refitted_slots = adequacy_cell_executor_source_slots(
        refitted_evidence,
        production_view,
        task_id="production_refitted_adequacy",
        strict_canonical=True,
    )
    refitted_binding = _build_binding(context, refitted_slots)
    refitted_view = build_refitted_adequacy_task_view(
        task_id="production_refitted_adequacy",
        evidence=refitted_evidence,
        parent=production_view,
        run_id=context.run_id,
        binding_fingerprint=refitted_binding.fingerprint,
        strict_canonical=True,
    )
    publications.append(_publish_pass(context, refitted_binding, refitted_view))

    # Task 15: exact full-history pools, blocks, kernels, and transition gate.
    monthly_materialization = materialize_registered_block_uniforms(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact="production",
        frequency="monthly",
        history_count=CANONICAL_HISTORY_COUNT,
        strict_canonical=True,
    )
    daily_materialization = materialize_registered_block_uniforms(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact="production",
        frequency="daily",
        history_count=CANONICAL_HISTORY_COUNT,
        strict_canonical=True,
    )
    monthly_outcome = run_registered_materialized_block_calibration_outcome(
        monthly_materialization,
        model=production_fit,
        starting_length=preliminary.monthly.combined.starting_length,
        historical_source_states=decoded_pools.monthly_states,
        source_continuation_links=calibration_input.full.monthly_continuation[1:],
    )
    daily_outcome = run_registered_materialized_block_calibration_outcome(
        daily_materialization,
        model=production_fit,
        starting_length=preliminary.daily.combined.starting_length,
        historical_source_states=decoded_pools.daily_states,
        source_continuation_links=calibration_input.full.daily_continuation[1:],
    )
    block_parents = _require_successful_block_pair(monthly_outcome, daily_outcome)
    monthly_replay = registered_block_replay_source(block_parents[0])
    daily_replay = registered_block_replay_source(block_parents[1])
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(production_fit)
    monthly_dependence_reference = fit_dependence_reference(
        calibration_input.full.monthly_returns,
        calibration_input.full.monthly_continuation,
        frequency="monthly",
    )
    daily_dependence_reference = fit_dependence_reference(
        calibration_input.full.daily_returns,
        calibration_input.full.daily_continuation,
        frequency="daily",
    )
    core_model_fingerprint = scientific_core_model_fingerprint(
        role="production",
        fit=production_fit,
        pools=decoded_pools,
        monthly_continuation=calibration_input.full.monthly_continuation,
        daily_continuation=calibration_input.full.daily_continuation,
        monthly_dependence_reference=monthly_dependence_reference,
        daily_dependence_reference=daily_dependence_reference,
    )
    monthly_kernel = _run_production_kernel(
        replay=monthly_replay,
        source_slice=calibration_input.full,
        frequency="monthly",
        model=production_fit,
        parent_model_fingerprint=parent_model_fingerprint,
        core_model_fingerprint=core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
    )
    daily_kernel = _run_production_kernel(
        replay=daily_replay,
        source_slice=calibration_input.full,
        frequency="daily",
        model=production_fit,
        parent_model_fingerprint=parent_model_fingerprint,
        core_model_fingerprint=core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
    )
    transition = aggregate_transition_bootstrap(production_fit, macro_evidence)
    block_evidence = BlockCalibrationGateEvidence(
        role="production",
        selected_k=production_fit.n_states,
        monthly=block_parents[0],
        daily=block_parents[1],
        monthly_kernel=monthly_kernel,
        daily_kernel=daily_kernel,
        transition=transition,
    )
    block_look_prefix = publish_raw_planned_look_prefix(
        store=run_store,
        lease=lease,
        task_id="production_block_calibration",
        source=block_evidence,
    )
    block_slots = block_calibration_executor_source_slots(
        block_evidence,
        task_id="production_block_calibration",
    )
    block_binding = _build_binding(context, block_slots)
    block_view = build_block_calibration_task_view(
        block_evidence,
        task_id="production_block_calibration",
        run_id=context.run_id,
        binding_fingerprint=block_binding.fingerprint,
    )
    block_look_parity = verify_planned_look_binding_view_parity(
        block_look_prefix,
        binding=block_binding,
        final_view=block_view,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
    )
    publications.append(_publish_pass(context, block_binding, block_view))

    # Task 16: fragmentation projects only the exact two production blocks.
    fragmentation_slots = fragmentation_executor_source_slots(
        block_parents,
        task_id="production_fragmentation",
    )
    fragmentation_binding = _build_binding(context, fragmentation_slots)
    fragmentation_view = build_fragmentation_task_view(
        block_parents,
        task_id="production_fragmentation",
        run_id=context.run_id,
        binding_fingerprint=fragmentation_binding.fingerprint,
    )
    publications.append(
        _publish_pass(context, fragmentation_binding, fragmentation_view)
    )

    # Task 17: replay the same purpose-4 materializations for dependence.
    dependence_cells = _production_dependence_cells(
        calibration_input.full,
        monthly_replay=monthly_replay,
        daily_replay=daily_replay,
    )
    dependence_execution = execute_registered_artifact_dependence_role(
        dependence_cells,
        role="production",
        block_parents=block_parents,
        fragmentation_parent=fragmentation_view,
        strict_canonical=True,
    )
    dependence_slots = dependence_executor_source_slots(
        dependence_execution,
        task_id="production_dependence",
    )
    dependence_binding = _build_binding(context, dependence_slots)
    dependence_view = build_registered_dependence_task_view(
        dependence_execution,
        task_id="production_dependence",
        run_id=context.run_id,
        binding_fingerprint=dependence_binding.fingerprint,
    )
    publications.append(_publish_pass(context, dependence_binding, dependence_view))

    # Task 18: the four adequacy cells in the preregistered scientific order.
    adequacy_holm_slots = adequacy_holm_executor_source_slots(
        design_latent=design_stage.latent_correctness,
        production_latent=latent_view,
        design_refitted=design_stage.refitted_adequacy,
        production_refitted=refitted_view,
        strict_canonical=True,
    )
    adequacy_holm_binding = _build_binding(context, adequacy_holm_slots)
    adequacy_holm_view = assemble_four_cell_adequacy_holm_view(
        design_latent=design_stage.latent_correctness,
        production_latent=latent_view,
        design_refitted=design_stage.refitted_adequacy,
        production_refitted=refitted_view,
        binding_fingerprint=adequacy_holm_binding.fingerprint,
        strict_canonical=True,
    )
    publications.append(
        _publish_pass(context, adequacy_holm_binding, adequacy_holm_view)
    )

    # Task 19: design then production dependence, with no caller-supplied cells.
    dependence_holm_slots = dependence_holm_executor_source_slots(
        design_stage.dependence,
        dependence_view,
        run_id=context.run_id,
    )
    dependence_holm_binding = _build_binding(context, dependence_holm_slots)
    dependence_holm_view = assemble_four_cell_dependence_holm_view(
        design_stage.dependence,
        dependence_view,
        run_id=context.run_id,
        binding_fingerprint=dependence_holm_binding.fingerprint,
    )
    publications.append(
        _publish_pass(context, dependence_holm_binding, dependence_holm_view)
    )

    _verify_returned_publications(publications)
    if monthly_kernel.offset_grid is None or daily_kernel.offset_grid is None:
        raise ModelError("passed production block task lacks exact kernel grids")
    return CanonicalG3ProductionStageResult(
        publications=tuple(publications),
        design_stage=design_stage,
        production_fit_execution=production_execution,
        production_fit_view=production_view,
        production_fit=production_fit,
        production_features=production_features,
        preliminary_outcome=preliminary_outcome,
        preliminary=preliminary,
        macro_materialization=macro_materialization,
        macro_evidence=macro_evidence,
        macro_stability=macro_view,
        latent_evidence=latent_evidence,
        latent_correctness=latent_view,
        refitted_evidence=refitted_evidence,
        refitted_adequacy=refitted_view,
        decoded_pools=decoded_pools,
        monthly_dependence_reference=monthly_dependence_reference,
        daily_dependence_reference=daily_dependence_reference,
        core_model_fingerprint=core_model_fingerprint,
        block_materializations=(monthly_materialization, daily_materialization),
        block_parents=block_parents,
        monthly_kernel=monthly_kernel,
        daily_kernel=daily_kernel,
        monthly_kernel_grid=monthly_kernel.offset_grid,
        daily_kernel_grid=daily_kernel.offset_grid,
        transition=transition,
        block_evidence=block_evidence,
        block_calibration=block_view,
        block_planned_look_prefix=block_look_prefix,
        block_planned_look_parity=block_look_parity,
        fragmentation=fragmentation_view,
        dependence_execution=dependence_execution,
        dependence=dependence_view,
        adequacy_holm=adequacy_holm_view,
        dependence_holm=dependence_holm_view,
    )


def _verify_stage_inputs(
    design_stage: CanonicalG3DesignStageResult,
    calibration_input: CalibrationInput,
    *,
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
    master_seed: int,
    scientific_version: int,
) -> None:
    if not isinstance(design_stage, CanonicalG3DesignStageResult):
        raise ModelError("production stage requires CanonicalG3DesignStageResult")
    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("production stage requires CalibrationInput")
    fixed_point_identity = registered_design_fixed_point_identity(
        design_stage.fixed_point
    )
    if (
        fixed_point_identity.master_seed != master_seed
        or fixed_point_identity.scientific_version != scientific_version
        or not fixed_point_identity.passed
    ):
        raise ModelError("production seed/version differs from the design fixed point")
    verify_compatible_canonical_g3_preflights(launch_preflight, live_preflight)
    if (
        calibration_input.fingerprint != launch_preflight.calibration_input_fingerprint
        or calibration_input.fingerprint != live_preflight.calibration_input_fingerprint
    ):
        raise ModelError("production-stage input differs from canonical preflight")
    if launch_preflight.workers != CANONICAL_WORKERS or (
        live_preflight.workers != CANONICAL_WORKERS
    ):
        raise ModelError("canonical production stage requires exactly nine workers")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not run_store
        or not lease.active
        or lease.workers != live_preflight.workers
    ):
        raise ModelError("production stage requires the same active launch lease")
    publication_ids = tuple(item.binding.task_id for item in design_stage.publications)
    if publication_ids != _DESIGN_TASKS:
        raise ModelError("production stage requires the exact tasks 1--10 result")
    if any(
        item.binding.run_id != lease.run_id
        or not item.decision.passed
        or item.checkpoint is None
        or item.task_result.status != "passed"
        for item in design_stage.publications
    ):
        raise ModelError("design-stage result contains a non-pass or another run")
    durable = run_store.verify_task_results(lease.run_id, require_complete=False)
    if tuple(durable) != _DESIGN_TASKS or any(
        durable[item.binding.task_id].fingerprint != item.task_result.fingerprint
        for item in design_stage.publications
    ):
        raise ModelError("durable task prefix differs from the design-stage result")


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
    source: object,
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
            "canonical production-stage task did not pass and checkpoint",
            details={"task_id": publication.binding.task_id},
        )
    return publication


def _prepare_production_features(
    calibration_input: CalibrationInput,
) -> HMMFeatureMatrix:
    return prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="production",
    )


def _require_production_recurring_history_support(
    pools: DecodedReturnPools,
) -> None:
    """Abort the success path unless every production K4 state recurs twice."""

    monthly = pools.monthly_diagnostics
    daily = pools.daily_diagnostics
    if not isinstance(monthly, StatePathDiagnostic) or not isinstance(
        daily, StatePathDiagnostic
    ):
        raise ModelError("production decoded-pool diagnostics have invalid types")
    recurring_support = owner_fixed_k4_recurring_history_support(
        monthly_episode_counts=tuple(int(value) for value in monthly.episode_counts),
        daily_run_counts=tuple(int(value) for value in daily.episode_counts),
    )
    if not recurring_support.passed:
        raise OwnerFixedK4RecurringHistoryFailure(recurring_support)


def _require_successful_block_pair(
    monthly: RegisteredBlockCalibration | object,
    daily: RegisteredBlockCalibration | object,
) -> tuple[RegisteredBlockCalibration, RegisteredBlockCalibration]:
    if not isinstance(monthly, RegisteredBlockCalibration) or not isinstance(
        daily,
        RegisteredBlockCalibration,
    ):
        raise ModelError(
            "canonical G3 success path requires both production block successes",
            details={
                "monthly_outcome": type(monthly).__name__,
                "daily_outcome": type(daily).__name__,
            },
        )
    # Replay-source construction fully revalidates both producer ledgers.
    registered_block_replay_source(monthly)
    registered_block_replay_source(daily)
    return monthly, daily


def _run_production_kernel(
    *,
    replay: RegisteredBlockReplaySource,
    source_slice: CalibrationSlice,
    frequency: BaseFrequency,
    model: GaussianHMMFit,
    parent_model_fingerprint: str,
    core_model_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    workers: int,
) -> KernelCalibrationExecution:
    if frequency == "monthly":
        historical_returns = source_slice.monthly_returns
    elif frequency == "daily":
        historical_returns = source_slice.daily_returns
    else:  # pragma: no cover - closed callers make this impossible.
        raise AssertionError("unsupported production kernel frequency")
    core = replay.core_identity
    if (
        core.frequency != frequency
        or core.artifact != "production"
        or replay.model is not model
        or core.parent_model_fingerprint != parent_model_fingerprint
    ):
        raise ModelError("production kernel replay ancestry is inconsistent")
    target_fingerprint = scientific_array_fingerprint(
        historical_returns,
        role=f"production_{frequency}_kernel_target_history",
    )
    return run_registered_kernel_calibration_cell(
        model=replay.model,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact="production",
        frequency=frequency,
        selected_length=core.selected_length,
        historical_returns=historical_returns,
        historical_source_states=replay.historical_source_states,
        source_continuation_links=replay.source_continuation_links,
        model_artifact_fingerprint=core_model_fingerprint,
        target_history_fingerprint=target_fingerprint,
        workers=workers,
        strict_canonical=True,
    )


def _production_dependence_cells(
    full: CalibrationSlice,
    *,
    monthly_replay: RegisteredBlockReplaySource,
    daily_replay: RegisteredBlockReplaySource,
) -> tuple[DependenceGateCellInput, DependenceGateCellInput]:
    return (
        DependenceGateCellInput(
            cell_id="production_monthly",
            artifact="production",
            frequency="monthly",
            model=monthly_replay.model,
            historical_returns=full.monthly_returns,
            historical_source_states=monthly_replay.historical_source_states,
            source_continuation_links=monthly_replay.source_continuation_links,
            calendar_years=_monthly_calendar_years(full.monthly_period_ordinals),
        ),
        DependenceGateCellInput(
            cell_id="production_daily",
            artifact="production",
            frequency="daily",
            model=daily_replay.model,
            historical_returns=full.daily_returns,
            historical_source_states=daily_replay.historical_source_states,
            source_continuation_links=daily_replay.source_continuation_links,
            calendar_years=_daily_calendar_years(full.daily_session_ns),
        ),
    )


def _monthly_calendar_years(period_ordinals: np.ndarray) -> np.ndarray:
    ordinals = np.asarray(period_ordinals, dtype=np.int64)
    return _immutable_int64(1970 + np.floor_divide(ordinals, 12))


def _daily_calendar_years(session_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(session_ns, dtype=np.int64).astype("datetime64[ns]")
    return _immutable_int64(timestamps.astype("datetime64[Y]").astype(np.int64) + 1970)


def _immutable_int64(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype("<i8"))
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<i8"))


def _verify_returned_publications(
    publications: list[ScientificTaskPublication],
) -> None:
    task_ids = tuple(item.binding.task_id for item in publications)
    if task_ids != _PRODUCTION_TASKS:
        raise ModelError(
            "canonical production-stage publication order is incomplete",
            details={"expected": list(_PRODUCTION_TASKS), "actual": list(task_ids)},
        )
    if any(
        not item.decision.passed
        or item.checkpoint is None
        or item.task_result.status != "passed"
        for item in publications
    ):
        raise ModelError("canonical production-stage result contains a non-pass")
