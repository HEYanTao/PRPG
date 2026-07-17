"""Closed, success-only execution of the canonical G3 design stage.

This module is the first half of the canonical 26-task vertical slice.  It
owns the numerical and durable sequencing for tasks 1 through 10 and exposes
no handler map, callback, caller-provided source mapping, reduced geometry, or
scientific pass flag.  Every task is computed from the exact retained parents,
bound through its task-specific prebinding API, converted to its guarded view,
and published at the next durable registry position.

The entry point intentionally supports only a fresh launch and an all-success
return.  A scientifically terminal non-pass may be durably published by the
normal task publisher, after which this runner raises.  Invalid holdout
benchmarks and other branches for which no launch-critical publisher exists
also raise.  Unexpected operational and programming exceptions are never
translated into scientific evidence here.
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
    build_latent_adequacy_task_view,
    build_refitted_adequacy_task_view,
)
from prpg.model.alpha import AlphaOffsetGrid, BaseFrequency
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    RegisteredBlockReplaySource,
    registered_block_replay_source,
)
from prpg.model.calibration import (
    DecodedReturnPools,
    MacroStabilityEvidence,
    prepare_hmm_feature_matrix,
)
from prpg.model.candidate_fixed_point_execution import (
    execute_registered_design_fixed_point,
)
from prpg.model.candidate_suite import registered_design_candidate_suite_identity
from prpg.model.dependence import DependenceReference, fit_dependence_reference
from prpg.model.dependence_gate_execution import (
    DependenceGateCellInput,
    DependenceTaskView,
    RegisteredArtifactDependenceExecution,
    build_fragmentation_task_view,
    build_registered_dependence_task_view,
    execute_registered_artifact_dependence_role,
)
from prpg.model.g3_boundary_views import (
    BlockCalibrationGateEvidence,
    BlockCalibrationTaskView,
    InputIntegrityDecisionEvidence,
    InputIntegrityTaskView,
    build_block_calibration_task_view,
    build_input_integrity_task_view,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    CandidateViabilityTaskView,
    MacroStabilityTaskView,
    SealedHoldoutVetoView,
    build_candidate_selection_task_view,
    build_candidate_viability_task_view,
    build_macro_stability_task_view,
    build_sealed_holdout_veto_view,
    selected_design_fit_from_view,
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
from prpg.model.g3_task_binding import (
    ScientificTaskBinding,
    build_scientific_task_binding,
)
from prpg.model.g3_task_publication import (
    ScientificTaskPublication,
    ScientificTaskSourceSlots,
    adequacy_cell_executor_source_slots,
    block_calibration_executor_source_slots,
    candidate_selection_executor_source_slots,
    candidate_viability_executor_source_slots,
    dependence_executor_source_slots,
    fragmentation_executor_source_slots,
    input_integrity_executor_source_slots,
    macro_stability_executor_source_slots,
    publish_exact_next_scientific_task,
    sealed_holdout_executor_source_slots,
)
from prpg.model.hmm import GaussianHMMFit, HMMFeatureMatrix
from prpg.model.input import CalibrationInput, CalibrationSlice
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    run_registered_kernel_calibration_cell,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.refit_adequacy_execution import (
    RegisteredRefittedAdequacyBatch,
    run_registered_refitted_adequacy_cell,
)
from prpg.model.registered_fixed_point import (
    RegisteredDesignFixedPoint,
    registered_design_fixed_point_sources,
    registered_final_selected_blocks,
)
from prpg.model.registered_holdout_execution import (
    RegisteredSealedHoldoutEvaluation,
    execute_registered_sealed_holdout,
    is_registered_sealed_holdout_evaluation,
    is_registered_sealed_holdout_invalid_benchmark,
    registered_sealed_holdout_evaluation_sources,
)
from prpg.model.run_store import CalibrationRunStore, LaunchLease
from prpg.model.scientific_artifact import (
    ScientificArtifactBinding,
    scientific_core_model_fingerprint,
)
from prpg.model.transition_bootstrap import (
    TransitionBootstrapAggregate,
    aggregate_transition_bootstrap,
)
from prpg.simulation.rng import CalibrationArtifact

_DESIGN_TASKS: Final = G3_GATE_ORDER[:10]


@dataclass(frozen=True, slots=True)
class CanonicalG3DesignStageResult:
    """Exact passed publications and retained parents needed by tasks 11--26."""

    publications: tuple[ScientificTaskPublication, ...]
    fixed_point: RegisteredDesignFixedPoint
    input_integrity: InputIntegrityTaskView
    candidate_viability: CandidateViabilityTaskView
    candidate_selection: CandidateSelectionTaskView
    sealed_holdout: SealedHoldoutVetoView
    macro_stability: MacroStabilityTaskView
    latent_correctness: AdequacyCellTaskView
    refitted_adequacy: AdequacyCellTaskView
    block_calibration: BlockCalibrationTaskView
    fragmentation: DependenceTaskView
    dependence: DependenceTaskView
    holdout_execution: RegisteredSealedHoldoutEvaluation
    macro_evidence: MacroStabilityEvidence
    decoded_return_pools: DecodedReturnPools
    monthly_dependence_reference: DependenceReference
    daily_dependence_reference: DependenceReference
    core_model_fingerprint: str
    latent_evidence: RegisteredLatentCell
    refitted_evidence: RegisteredRefittedAdequacyBatch
    block_evidence: BlockCalibrationGateEvidence
    dependence_execution: RegisteredArtifactDependenceExecution
    design_fit: GaussianHMMFit
    design_features: HMMFeatureMatrix
    block_parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration]
    monthly_kernel: KernelCalibrationExecution
    daily_kernel: KernelCalibrationExecution
    monthly_kernel_grid: AlphaOffsetGrid
    daily_kernel_grid: AlphaOffsetGrid
    transition: TransitionBootstrapAggregate
    block_planned_look_prefix: RegisteredPlannedLookPrefix
    block_planned_look_parity: PlannedLookParityIdentity

    @property
    def generation_authorized(self) -> bool:
        """The design stage is intermediate G3 evidence, never generation authority."""

        return False


@dataclass(frozen=True, slots=True)
class _RunnerContext:
    run_store: CalibrationRunStore
    checkpoint_store: CalibrationCheckpointStore
    lease: LaunchLease
    live_preflight: CanonicalG3Preflight

    @property
    def run_id(self) -> str:
        return self.lease.run_id


def run_canonical_g3_design_stage(
    calibration_input: CalibrationInput,
    *,
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    lease: LaunchLease,
    master_seed: int,
    scientific_version: int,
) -> CanonicalG3DesignStageResult:
    """Compute, bind, and publish exact canonical G3 tasks 1 through 10.

    The caller must already have initialized the run and checkpoint stores,
    persisted/planned the launch preflight, and acquired ``lease``.  This
    function does not acquire, resume, or recover a launch.
    """

    _verify_fresh_design_launch(
        calibration_input,
        launch_preflight=launch_preflight,
        live_preflight=live_preflight,
        run_store=run_store,
        lease=lease,
    )
    context = _RunnerContext(run_store, checkpoint_store, lease, live_preflight)
    publications: list[ScientificTaskPublication] = []

    # Task 1: exact input/preflight/artifact binding.
    input_evidence = InputIntegrityDecisionEvidence(
        launch_preflight,
        live_preflight,
        ScientificArtifactBinding(
            config_fingerprint=launch_preflight.config_fingerprint,
            processed_data_fingerprint=launch_preflight.processed_data_fingerprint,
            calibration_input_fingerprint=(
                launch_preflight.calibration_input_fingerprint
            ),
            calibration_run_fingerprint=context.run_id,
            preflight_fingerprint=launch_preflight.fingerprint,
            task_registry_fingerprint=launch_preflight.task_registry_fingerprint,
            planned_look_registry_fingerprint=(
                launch_preflight.planned_look_registry_fingerprint
            ),
            scientific_version=scientific_version,
        ),
    )
    input_slots = input_integrity_executor_source_slots(
        input_evidence,
        run_id=context.run_id,
    )
    input_binding = _build_binding(context, input_slots)
    input_view = build_input_integrity_task_view(
        input_evidence,
        run_id=context.run_id,
        binding_fingerprint=input_binding.fingerprint,
    )
    publications.append(_publish_pass(context, input_binding, input_view))

    # Tasks 2--3: one closed numerical fixed point and its two projections.
    fixed_point = execute_registered_design_fixed_point(
        calibration_input,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
    )
    fixed_point_sources, _monthly_parent, _daily_parent = (
        registered_design_fixed_point_sources(fixed_point)
    )
    final_suite, final_viability, final_selection = fixed_point_sources[-1]

    viability_slots = candidate_viability_executor_source_slots(
        final_viability,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    viability_binding = _build_binding(context, viability_slots)
    viability_view = build_candidate_viability_task_view(
        final_viability,
        master_seed=master_seed,
        scientific_version=scientific_version,
        run_id=context.run_id,
        binding_fingerprint=viability_binding.fingerprint,
        require_canonical_suite=True,
    )
    publications.append(_publish_pass(context, viability_binding, viability_view))

    selection_slots = candidate_selection_executor_source_slots(
        viability_view,
        final_selection,
    )
    selection_binding = _build_binding(context, selection_slots)
    selection_view = build_candidate_selection_task_view(
        viability_view,
        final_selection,
        run_id=context.run_id,
        binding_fingerprint=selection_binding.fingerprint,
    )
    publications.append(_publish_pass(context, selection_binding, selection_view))

    # Task 4: execute once and unwrap only the registered valid-benchmark path.
    raw_holdout = execute_registered_sealed_holdout(
        selection_view,
        calibration_input.design,
        calibration_input.holdout,
    )
    if not is_registered_sealed_holdout_evaluation(raw_holdout):
        details: dict[str, object] = {}
        if is_registered_sealed_holdout_invalid_benchmark(raw_holdout):
            details["failure_code"] = raw_holdout.failure_code
        raise ModelError(
            "canonical G3 success path requires a valid holdout benchmark",
            details=details,
        )
    holdout_execution = raw_holdout
    (
        holdout_selection,
        _holdout_fit,
        holdout_design,
        holdout_slice,
        holdout_evidence,
    ) = registered_sealed_holdout_evaluation_sources(holdout_execution)
    if (
        holdout_selection is not selection_view
        or holdout_design is not calibration_input.design
        or holdout_slice is not calibration_input.holdout
    ):
        raise ModelError("registered holdout execution changed its exact parents")
    holdout_slots = sealed_holdout_executor_source_slots(
        selection_view,
        holdout_evidence,
        design=calibration_input.design,
        holdout=calibration_input.holdout,
    )
    holdout_binding = _build_binding(context, holdout_slots)
    holdout_view = build_sealed_holdout_veto_view(
        selection_view,
        holdout_evidence,
        design=calibration_input.design,
        holdout=calibration_input.holdout,
        run_id=context.run_id,
        binding_fingerprint=holdout_binding.fingerprint,
    )
    publications.append(_publish_pass(context, holdout_binding, holdout_view))

    # Task 5: reuse the exact final-iteration macro object; never rerun it.
    registered_design_candidate_suite_identity(final_suite)
    raw_selected_k = selection_view.selected_n_states
    if (
        isinstance(raw_selected_k, bool)
        or not isinstance(raw_selected_k, int)
        or raw_selected_k not in {2, 3, 4, 5}
    ):
        raise ModelError("passed fixed point lacks a registered selected K")
    selected_k = raw_selected_k
    selected_row = next(
        (row for row in final_suite.rows if row.n_states == selected_k),
        None,
    )
    if (
        selected_row is None
        or selected_row.macro_stability is None
        or selected_row.decoded_return_pools is None
    ):
        raise ModelError(
            "final fixed point lacks selected macro or decoded-pool evidence"
        )
    macro_evidence = selected_row.macro_stability
    decoded_pools = selected_row.decoded_return_pools
    macro_slots = macro_stability_executor_source_slots(
        task_id="design_macro_stability",
        evidence=macro_evidence,
        parent=selection_view,
        source_slice=calibration_input.design,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    macro_binding = _build_binding(context, macro_slots)
    macro_view = build_macro_stability_task_view(
        task_id="design_macro_stability",
        evidence=macro_evidence,
        parent=selection_view,
        source_slice=calibration_input.design,
        master_seed=master_seed,
        scientific_version=scientific_version,
        run_id=context.run_id,
        binding_fingerprint=macro_binding.fingerprint,
    )
    publications.append(_publish_pass(context, macro_binding, macro_view))

    selected_fit = selected_design_fit_from_view(selection_view)
    design_features = _prepare_design_features(calibration_input)

    # Task 6: canonical latent correctness.
    latent_evidence = run_registered_latent_cell(
        selected_fit,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )
    latent_slots = adequacy_cell_executor_source_slots(
        latent_evidence,
        selection_view,
        task_id="design_latent_correctness",
        strict_canonical=True,
    )
    latent_binding = _build_binding(context, latent_slots)
    latent_view = build_latent_adequacy_task_view(
        task_id="design_latent_correctness",
        evidence=latent_evidence,
        parent=selection_view,
        run_id=context.run_id,
        binding_fingerprint=latent_binding.fingerprint,
        strict_canonical=True,
    )
    publications.append(_publish_pass(context, latent_binding, latent_view))

    # Task 7: canonical refitted adequacy, deterministically reduced on 9 workers.
    refitted_evidence = run_registered_refitted_adequacy_cell(
        role="design",
        original_fit=selected_fit,
        artifact_features=design_features,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
        strict_canonical=True,
    )
    refitted_slots = adequacy_cell_executor_source_slots(
        refitted_evidence,
        selection_view,
        task_id="design_refitted_adequacy",
        strict_canonical=True,
    )
    refitted_binding = _build_binding(context, refitted_slots)
    refitted_view = build_refitted_adequacy_task_view(
        task_id="design_refitted_adequacy",
        evidence=refitted_evidence,
        parent=selection_view,
        run_id=context.run_id,
        binding_fingerprint=refitted_binding.fingerprint,
        strict_canonical=True,
    )
    publications.append(_publish_pass(context, refitted_binding, refitted_view))

    # Task 8: exact final fixed-point blocks plus kernel/transition evidence.
    block_parents = registered_final_selected_blocks(final_selection)
    monthly_replay = registered_block_replay_source(block_parents[0])
    daily_replay = registered_block_replay_source(block_parents[1])
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(selected_fit)
    monthly_dependence_reference = fit_dependence_reference(
        calibration_input.design.monthly_returns,
        calibration_input.design.monthly_continuation,
        frequency="monthly",
    )
    daily_dependence_reference = fit_dependence_reference(
        calibration_input.design.daily_returns,
        calibration_input.design.daily_continuation,
        frequency="daily",
    )
    core_model_fingerprint = scientific_core_model_fingerprint(
        role="design",
        fit=selected_fit,
        pools=decoded_pools,
        monthly_continuation=calibration_input.design.monthly_continuation,
        daily_continuation=calibration_input.design.daily_continuation,
        monthly_dependence_reference=monthly_dependence_reference,
        daily_dependence_reference=daily_dependence_reference,
    )
    monthly_kernel = _run_design_kernel(
        replay=monthly_replay,
        source_slice=calibration_input.design,
        frequency="monthly",
        parent_model_fingerprint=parent_model_fingerprint,
        core_model_fingerprint=core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
    )
    daily_kernel = _run_design_kernel(
        replay=daily_replay,
        source_slice=calibration_input.design,
        frequency="daily",
        parent_model_fingerprint=parent_model_fingerprint,
        core_model_fingerprint=core_model_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=live_preflight.workers,
    )
    transition = aggregate_transition_bootstrap(selected_fit, macro_evidence)
    block_evidence = BlockCalibrationGateEvidence(
        role="design",
        selected_k=selected_k,
        monthly=block_parents[0],
        daily=block_parents[1],
        monthly_kernel=monthly_kernel,
        daily_kernel=daily_kernel,
        transition=transition,
    )
    block_look_prefix = publish_raw_planned_look_prefix(
        store=run_store,
        lease=lease,
        task_id="design_block_calibration",
        source=block_evidence,
    )
    block_slots = block_calibration_executor_source_slots(
        block_evidence,
        task_id="design_block_calibration",
    )
    block_binding = _build_binding(context, block_slots)
    block_view = build_block_calibration_task_view(
        block_evidence,
        task_id="design_block_calibration",
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

    # Task 9: fragmentation is a projection of the two exact block parents.
    fragmentation_slots = fragmentation_executor_source_slots(
        block_parents,
        task_id="design_fragmentation",
    )
    fragmentation_binding = _build_binding(context, fragmentation_slots)
    fragmentation_view = build_fragmentation_task_view(
        block_parents,
        task_id="design_fragmentation",
        run_id=context.run_id,
        binding_fingerprint=fragmentation_binding.fingerprint,
    )
    publications.append(
        _publish_pass(context, fragmentation_binding, fragmentation_view)
    )

    # Task 10: dependence replays those same purpose-4 block materializations.
    dependence_cells = _design_dependence_cells(
        calibration_input.design,
        monthly_replay=monthly_replay,
        daily_replay=daily_replay,
    )
    dependence_execution = execute_registered_artifact_dependence_role(
        dependence_cells,
        role="design",
        block_parents=block_parents,
        fragmentation_parent=fragmentation_view,
        strict_canonical=True,
    )
    dependence_slots = dependence_executor_source_slots(
        dependence_execution,
        task_id="design_dependence",
    )
    dependence_binding = _build_binding(context, dependence_slots)
    dependence_view = build_registered_dependence_task_view(
        dependence_execution,
        task_id="design_dependence",
        run_id=context.run_id,
        binding_fingerprint=dependence_binding.fingerprint,
    )
    publications.append(_publish_pass(context, dependence_binding, dependence_view))

    _verify_returned_publications(publications)
    if monthly_kernel.offset_grid is None or daily_kernel.offset_grid is None:
        raise ModelError("passed design block task lacks exact kernel offset grids")
    return CanonicalG3DesignStageResult(
        publications=tuple(publications),
        fixed_point=fixed_point,
        input_integrity=input_view,
        candidate_viability=viability_view,
        candidate_selection=selection_view,
        sealed_holdout=holdout_view,
        macro_stability=macro_view,
        latent_correctness=latent_view,
        refitted_adequacy=refitted_view,
        block_calibration=block_view,
        fragmentation=fragmentation_view,
        dependence=dependence_view,
        holdout_execution=holdout_execution,
        macro_evidence=macro_evidence,
        decoded_return_pools=decoded_pools,
        monthly_dependence_reference=monthly_dependence_reference,
        daily_dependence_reference=daily_dependence_reference,
        core_model_fingerprint=core_model_fingerprint,
        latent_evidence=latent_evidence,
        refitted_evidence=refitted_evidence,
        block_evidence=block_evidence,
        dependence_execution=dependence_execution,
        design_fit=selected_fit,
        design_features=design_features,
        block_parents=block_parents,
        monthly_kernel=monthly_kernel,
        daily_kernel=daily_kernel,
        monthly_kernel_grid=monthly_kernel.offset_grid,
        daily_kernel_grid=daily_kernel.offset_grid,
        transition=transition,
        block_planned_look_prefix=block_look_prefix,
        block_planned_look_parity=block_look_parity,
    )


def _verify_fresh_design_launch(
    calibration_input: CalibrationInput,
    *,
    launch_preflight: CanonicalG3Preflight,
    live_preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
) -> None:
    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("canonical G3 design stage requires CalibrationInput")
    verify_compatible_canonical_g3_preflights(launch_preflight, live_preflight)
    if (
        calibration_input.fingerprint != launch_preflight.calibration_input_fingerprint
        or calibration_input.fingerprint != live_preflight.calibration_input_fingerprint
    ):
        raise ModelError("calibration input differs from the canonical preflight")
    if launch_preflight.workers != CANONICAL_WORKERS or (
        live_preflight.workers != CANONICAL_WORKERS
    ):
        raise ModelError("canonical G3 design stage requires exactly nine workers")
    if run_store.verify_task_results(lease.run_id, require_complete=False):
        raise ModelError(
            "canonical G3 design stage requires a fresh task-result prefix"
        )


def _build_binding(
    context: _RunnerContext,
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
    context: _RunnerContext,
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
            "canonical G3 success-path task did not pass and checkpoint",
            details={"task_id": publication.binding.task_id},
        )
    return publication


def _prepare_design_features(calibration_input: CalibrationInput) -> HMMFeatureMatrix:
    return prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="design",
        design_rows=len(calibration_input.design.macro_features),
    )


def _run_design_kernel(
    *,
    replay: RegisteredBlockReplaySource,
    source_slice: CalibrationSlice,
    frequency: BaseFrequency,
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
        raise AssertionError("unsupported design kernel frequency")
    core = replay.core_identity
    if core.frequency != frequency or core.artifact != "design":
        raise ModelError("design kernel replay source has the wrong role or frequency")
    if core.parent_model_fingerprint != parent_model_fingerprint:
        raise ModelError("design kernel replay model differs from selected fit")
    target_fingerprint = scientific_array_fingerprint(
        historical_returns,
        role=f"design_{frequency}_kernel_target_history",
    )
    return run_registered_kernel_calibration_cell(
        model=replay.model,
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact="design",
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


def _design_dependence_cells(
    design: CalibrationSlice,
    *,
    monthly_replay: RegisteredBlockReplaySource,
    daily_replay: RegisteredBlockReplaySource,
) -> tuple[DependenceGateCellInput, DependenceGateCellInput]:
    monthly = DependenceGateCellInput(
        cell_id="design_monthly",
        artifact="design",
        frequency="monthly",
        model=monthly_replay.model,
        historical_returns=design.monthly_returns,
        historical_source_states=monthly_replay.historical_source_states,
        source_continuation_links=monthly_replay.source_continuation_links,
        calendar_years=_monthly_calendar_years(design.monthly_period_ordinals),
    )
    daily = DependenceGateCellInput(
        cell_id="design_daily",
        artifact="design",
        frequency="daily",
        model=daily_replay.model,
        historical_returns=design.daily_returns,
        historical_source_states=daily_replay.historical_source_states,
        source_continuation_links=daily_replay.source_continuation_links,
        calendar_years=_daily_calendar_years(design.daily_session_ns),
    )
    return monthly, daily


def _monthly_calendar_years(period_ordinals: np.ndarray) -> np.ndarray:
    ordinals = np.asarray(period_ordinals, dtype=np.int64)
    years = 1970 + np.floor_divide(ordinals, 12)
    return _immutable_int64(years)


def _daily_calendar_years(session_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(session_ns, dtype=np.int64).astype("datetime64[ns]")
    years = timestamps.astype("datetime64[Y]").astype(np.int64) + 1970
    return _immutable_int64(years)


def _immutable_int64(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype("<i8"))
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<i8"))


def _verify_returned_publications(
    publications: list[ScientificTaskPublication],
) -> None:
    task_ids = tuple(item.binding.task_id for item in publications)
    if task_ids != _DESIGN_TASKS:
        raise ModelError(
            "canonical G3 design-stage publication order is incomplete",
            details={"expected": list(_DESIGN_TASKS), "actual": list(task_ids)},
        )
    if any(
        not item.decision.passed
        or item.checkpoint is None
        or item.task_result.status != "passed"
        for item in publications
    ):
        raise ModelError("canonical G3 design-stage result contains a non-pass")
