"""Closed fresh-launch execution of all 26 canonical G3 tasks.

This module is the deliberately small composition root for the frozen G3
all-success vertical slice.  It accepts one already-built live preflight and
one already-active fresh launch, passes that same preflight object through the
design, production, bridge, and finalization stages.  Each child stage owns
its durable prefix checks; this composition root checks only stage boundaries
and the final exact object lineage.

There is no launch initialization, planning, lease acquisition, recovery,
handler map, callback, caller-authored science, reduced geometry, generation,
or G4/G5 work here.  Every exception from a child stage or a verification
step escapes unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_evidence_store import G3EvidenceStore
from prpg.model.g3_exact_v2_finalization import (
    ExactV2G3FinalizationResult,
    finalize_exact_v2_all_success,
)
from prpg.model.g3_preflight import (
    CANONICAL_MASTER_SEED,
    CANONICAL_WORKERS,
    CanonicalG3Preflight,
    read_persisted_canonical_g3_preflight,
    verify_live_canonical_g3_preflight,
)
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_success_bridge_stage import (
    CanonicalG3BridgeStageResult,
    run_canonical_g3_bridge_stage,
)
from prpg.model.g3_success_production_stage import (
    CanonicalG3ProductionStageResult,
    run_canonical_g3_production_stage,
)
from prpg.model.g3_success_runner import (
    CanonicalG3DesignStageResult,
    run_canonical_g3_design_stage,
)
from prpg.model.g3_task_publication import ScientificTaskPublication
from prpg.model.input import CalibrationInput
from prpg.model.run_store import CalibrationRunStore, LaunchLease
from prpg.model.scientific_artifact import CANONICAL_SCIENTIFIC_VERSION
from prpg.model.scientific_artifact_pair import ScientificArtifactPairStore
from prpg.model.selection import OWNER_FIXED_N_STATES

_DESIGN_TASKS: Final = G3_GATE_ORDER[:10]
_DESIGN_PRODUCTION_TASKS: Final = G3_GATE_ORDER[:19]
_FIRST_25_TASKS: Final = G3_GATE_ORDER[:25]
_ARTIFACT_TASK: Final = G3_GATE_ORDER[25:]


@dataclass(frozen=True, slots=True)
class CanonicalG3SuccessExecutionResult:
    """Exact stage results and durable evidence from one completed G3 run."""

    design_stage: CanonicalG3DesignStageResult
    production_stage: CanonicalG3ProductionStageResult
    bridge_stage: CanonicalG3BridgeStageResult
    first_25_publications: tuple[ScientificTaskPublication, ...]
    finalization: ExactV2G3FinalizationResult
    publications: tuple[ScientificTaskPublication, ...]

    @property
    def generation_authorized(self) -> Literal[False]:
        """A completed G3 run is evidence for later G5, not generation authority."""

        return False


def run_canonical_g3_success_execution(
    calibration_input: CalibrationInput,
    *,
    preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    lease: LaunchLease,
    master_seed: int,
    scientific_version: int,
    model_store: ModelArtifactStore,
    pair_store: ScientificArtifactPairStore,
    evidence_store: G3EvidenceStore,
) -> CanonicalG3SuccessExecutionResult:
    """Execute exact canonical G3 tasks 1--26 on one fresh active launch."""

    _verify_fresh_execution_inputs(
        calibration_input,
        preflight=preflight,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        lease=lease,
        master_seed=master_seed,
        scientific_version=scientific_version,
        model_store=model_store,
        pair_store=pair_store,
        evidence_store=evidence_store,
    )

    design_stage = run_canonical_g3_design_stage(
        calibration_input,
        launch_preflight=preflight,
        live_preflight=preflight,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        lease=lease,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    if type(design_stage) is not CanonicalG3DesignStageResult:
        raise ModelError("design stage returned a substituted result type")
    design_publications = _verified_stage_publications(
        design_stage.publications,
        expected_tasks=_DESIGN_TASKS,
        run_id=lease.run_id,
    )

    production_stage = run_canonical_g3_production_stage(
        design_stage,
        calibration_input,
        launch_preflight=preflight,
        live_preflight=preflight,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        lease=lease,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    if type(production_stage) is not CanonicalG3ProductionStageResult:
        raise ModelError("production stage returned a substituted result type")
    if production_stage.design_stage is not design_stage:
        raise IntegrityError("production stage changed its exact design parent")
    first_19 = (*design_publications, *production_stage.publications)
    first_19 = _verified_stage_publications(
        first_19,
        expected_tasks=_DESIGN_PRODUCTION_TASKS,
        run_id=lease.run_id,
    )

    bridge_stage = run_canonical_g3_bridge_stage(
        design_stage,
        production_stage,
        launch_preflight=preflight,
        live_preflight=preflight,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        lease=lease,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    if type(bridge_stage) is not CanonicalG3BridgeStageResult:
        raise ModelError("bridge stage returned a substituted result type")
    first_25 = (*first_19, *bridge_stage.publications)
    first_25 = _verified_stage_publications(
        first_25,
        expected_tasks=_FIRST_25_TASKS,
        run_id=lease.run_id,
    )
    _require_owner_fixed_k4_before_finalization(design_stage, production_stage)

    finalization = finalize_exact_v2_all_success(
        run_store=run_store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        live_preflight=preflight,
        publications=first_25,
        model_store=model_store,
        pair_store=pair_store,
        evidence_store=evidence_store,
    )
    _require_owner_fixed_k4_after_finalization(finalization)
    publications = _verified_finalization_publications(
        finalization,
        first_25=first_25,
        run_store=run_store,
        lease=lease,
        preflight=preflight,
    )
    return CanonicalG3SuccessExecutionResult(
        design_stage=design_stage,
        production_stage=production_stage,
        bridge_stage=bridge_stage,
        first_25_publications=first_25,
        finalization=finalization,
        publications=publications,
    )


def _verify_fresh_execution_inputs(
    calibration_input: CalibrationInput,
    *,
    preflight: CanonicalG3Preflight,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    lease: LaunchLease,
    master_seed: int,
    scientific_version: int,
    model_store: ModelArtifactStore,
    pair_store: ScientificArtifactPairStore,
    evidence_store: G3EvidenceStore,
) -> None:
    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("canonical G3 execution requires CalibrationInput")
    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("canonical G3 execution requires CalibrationRunStore")
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("canonical G3 execution requires CalibrationCheckpointStore")
    if not isinstance(model_store, ModelArtifactStore):
        raise ModelError("canonical G3 execution requires ModelArtifactStore")
    if not isinstance(pair_store, ScientificArtifactPairStore):
        raise ModelError("canonical G3 execution requires ScientificArtifactPairStore")
    if not isinstance(evidence_store, G3EvidenceStore):
        raise ModelError("canonical G3 execution requires G3EvidenceStore")
    # The authorized preflight is bound to the one approved config fingerprint.
    # Requiring the same code-owned values here prevents callers from attaching
    # a different RNG/science identity to that otherwise-valid preflight.
    if type(master_seed) is not int or master_seed != CANONICAL_MASTER_SEED:
        raise ModelError(
            "canonical G3 execution requires approved master seed 20260714"
        )
    if (
        type(scientific_version) is not int
        or scientific_version != CANONICAL_SCIENTIFIC_VERSION
    ):
        raise ModelError(
            "canonical G3 execution requires canonical scientific version 11"
        )

    verify_live_canonical_g3_preflight(preflight)
    if (
        calibration_input.fingerprint != preflight.calibration_input_fingerprint
        or preflight.workers != CANONICAL_WORKERS
    ):
        raise ModelError("canonical G3 input or worker geometry differs from preflight")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not run_store
        or not lease.active
        or lease.preflight_fingerprint != preflight.fingerprint
        or lease.workers != CANONICAL_WORKERS
    ):
        raise ModelError("canonical G3 execution requires the exact active launch")
    persisted = read_persisted_canonical_g3_preflight(
        run_store,
        run_id=lease.run_id,
    )
    if persisted.preflight.fingerprint != preflight.fingerprint:
        raise IntegrityError("fresh preflight differs from its persisted launch parent")
    marker = run_store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != preflight.fingerprint
        or marker.workers != CANONICAL_WORKERS
    ):
        raise ModelError("canonical G3 launch is not the exact running launch")

    # Fresh-launch-only means task 1 has not been written yet.  Later durable
    # prefix replay belongs to the closed child stages and exact-v2 finalizer.
    if run_store.verify_task_results(lease.run_id, require_complete=False):
        raise ModelError("canonical G3 execution requires a fresh task prefix")


def _require_owner_fixed_k4_before_finalization(
    design_stage: CanonicalG3DesignStageResult,
    production_stage: CanonicalG3ProductionStageResult,
) -> None:
    """Independently fence the v3 composition root before legacy finalization."""

    selected_state_counts = (
        design_stage.candidate_selection.selected_n_states,
        design_stage.design_fit.n_states,
        production_stage.production_fit_view.design_selected_k,
        production_stage.production_fit.n_states,
    )
    if any(
        type(value) is not int or value != OWNER_FIXED_N_STATES
        for value in selected_state_counts
    ):
        raise IntegrityError(
            "canonical v3 finalization boundary requires owner-approved K=4 lineage"
        )


def _require_owner_fixed_k4_after_finalization(
    finalization: ExactV2G3FinalizationResult,
) -> None:
    """Recheck K=4 from independently materialized finalizer outputs."""

    if type(finalization) is not ExactV2G3FinalizationResult:
        raise ModelError("canonical G3 finalizer returned a substituted result type")
    selected_state_counts = (
        finalization.prefix.selected_n_states,
        finalization.artifact_pair.selected_k,
        finalization.artifact_pair.design.selected_k,
        finalization.artifact_pair.production.selected_k,
    )
    if any(
        type(value) is not int or value != OWNER_FIXED_N_STATES
        for value in selected_state_counts
    ):
        raise IntegrityError(
            "canonical v3 finalized artifacts require owner-approved K=4 lineage"
        )


def _verified_stage_publications(
    publications: tuple[ScientificTaskPublication, ...],
    *,
    expected_tasks: tuple[str, ...],
    run_id: str,
) -> tuple[ScientificTaskPublication, ...]:
    if type(publications) is not tuple or len(publications) != len(expected_tasks):
        raise IntegrityError(
            "canonical G3 stage returned a substituted publication set"
        )
    for expected_task, publication in zip(expected_tasks, publications, strict=True):
        if type(publication) is not ScientificTaskPublication:
            raise IntegrityError(
                "canonical G3 stage returned a substituted publication"
            )
        checkpoint = publication.checkpoint
        if (
            checkpoint is None
            or publication.binding.task_id != expected_task
            or publication.binding.run_id != run_id
            or publication.decision.gate_id != expected_task
            or publication.decision.passed is not True
            or checkpoint.task_id != expected_task
            or checkpoint.binding.run_id != run_id
            or publication.task_result.task_id != expected_task
            or publication.task_result.run_id != run_id
            or publication.task_result.status != "passed"
            or publication.task_result.payload.get("checkpoint_fingerprint")
            != checkpoint.fingerprint
        ):
            raise IntegrityError(
                "canonical G3 publications are reordered or substituted"
            )
    return publications


def _verified_finalization_publications(
    finalization: ExactV2G3FinalizationResult,
    *,
    first_25: tuple[ScientificTaskPublication, ...],
    run_store: CalibrationRunStore,
    lease: LaunchLease,
    preflight: CanonicalG3Preflight,
) -> tuple[ScientificTaskPublication, ...]:
    if type(finalization) is not ExactV2G3FinalizationResult:
        raise ModelError("canonical G3 finalizer returned a substituted result type")
    supplied_first_25 = _verified_stage_publications(
        finalization.first_25_publications,
        expected_tasks=_FIRST_25_TASKS,
        run_id=lease.run_id,
    )
    _require_same_publication_objects(
        supplied_first_25,
        first_25,
        label="finalizer first-25 prefix",
    )
    artifact_publication = _verified_stage_publications(
        (finalization.artifact_integrity_publication,),
        expected_tasks=_ARTIFACT_TASK,
        run_id=lease.run_id,
    )[0]
    completion = finalization.completion_marker
    marker = run_store.read_launch(lease.run_id)
    if (
        finalization.generation_authorized is not False
        or not finalization.evidence_bundle.passed
        or finalization.evidence_bundle.generation_authorized is not False
        or finalization.durable_evidence.generation_authorized is not False
        or completion.run_id != lease.run_id
        or completion.state != "completed"
        or completion.preflight_fingerprint != preflight.fingerprint
        or marker.state != "completed"
        or marker.fingerprint != completion.fingerprint
        or finalization.durable_evidence.run_id != lease.run_id
        or finalization.durable_evidence.completion_fingerprint
        != completion.fingerprint
    ):
        raise IntegrityError(
            "canonical G3 finalization is not this launch's "
            "durable evidence-only result"
        )
    return (*first_25, artifact_publication)


def _require_same_publication_objects(
    actual: tuple[ScientificTaskPublication, ...],
    expected: tuple[ScientificTaskPublication, ...],
    *,
    label: str,
) -> None:
    if len(actual) != len(expected) or any(
        left is not right for left, right in zip(actual, expected, strict=True)
    ):
        raise IntegrityError(f"{label} changed exact publication object identity")
