"""Code-owned scientific result publication for canonical Phase-3 execution.

The scientific calculations in Phase 3 intentionally live in small typed
modules.  This module is the closed boundary that turns those typed results
into durable G3 task results and crash-resumable checkpoints.  It accepts no
task-ID-to-callable mapping and no caller supplied pass Boolean.  Every source
is one of the exact result types understood by :mod:`prpg.model.g3_assembly`,
and every decision is re-derived before it can be committed.

Checkpoints are computation state, not final model artifacts.  A passed task
is committed only after its exact registered sequential-look evidence (when
applicable) and a typed checkpoint have both been published.  Resume loads and
recursively verifies every checkpoint, re-derives the supplied typed source,
and compares the resulting scientific fingerprint before doing any new work.

This layer deliberately does not manufacture the expensive scientific
results.  Candidate fitting, adequacy, block/dependence calibration, bridges,
and artifact publication remain in their code-owned execution modules.  The
bundle below closes their orchestration/publication boundary without exposing
an arbitrary handler extension point.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeAlias, cast

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.adequacy_gate import (
    AdequacyCellTaskView,
    FourCellAdequacyHolmView,
    is_adequacy_cell_task_view,
    is_four_cell_adequacy_holm_view,
)
from prpg.model.bridge import TIER_B_INTERVAL_ERROR
from prpg.model.dependence_gate_execution import (
    DependenceTaskView,
    FourCellDependenceHolmView,
    is_registered_dependence_task_view,
    is_registered_four_cell_dependence_holm_view,
)
from prpg.model.g3 import G3EvidenceBundle
from prpg.model.g3_assembly import (
    DerivedG3GateDecision,
    G3TaskFailure,
    assemble_typed_g3_evidence_bundle,
    derive_g3_gate_decision,
    task_view_cross_binding_error,
)
from prpg.model.g3_boundary_views import (
    ArtifactIntegrityDecisionEvidence,
    ArtifactIntegrityTaskView,
    BlockCalibrationTaskView,
    BridgeDriverTaskView,
    InputIntegrityTaskView,
    build_artifact_integrity_task_view,
    is_artifact_integrity_task_view,
    is_block_calibration_task_view,
    is_bridge_driver_task_view,
    is_input_integrity_task_view,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    CandidateViabilityTaskView,
    MacroStabilityTaskView,
    ProductionFitSameKView,
    SealedHoldoutVetoView,
    is_candidate_selection_task_view,
    is_candidate_viability_task_view,
    is_macro_stability_task_view,
    is_production_fit_same_k_view,
    is_sealed_holdout_veto_view,
)
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointBinding,
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_preflight import (
    CanonicalG3Preflight,
    canonical_resume_preflight_fingerprint,
    verify_live_canonical_g3_preflight,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_TASK_REGISTRY,
    G3TaskSpec,
)
from prpg.model.inference import clopper_pearson_interval
from prpg.model.kernel_execution import (
    CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
    KernelCalibrationExecution,
)
from prpg.model.run_store import (
    CalibrationRunReference,
    CalibrationRunStore,
    G3TaskResult,
    LaunchLease,
    LaunchMarker,
    LookDecision,
    PlannedLookReceipt,
)
from prpg.model.scientific_artifact import (
    ArtifactRole,
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    is_verified_scientific_model_artifact,
    required_scientific_documents,
    scientific_report_payload,
    scientific_report_planned_look_families,
    scientific_report_task_ids,
)
from prpg.model.scientific_artifact_pair import (
    LoadedScientificArtifactPair,
    is_verified_scientific_artifact_pair,
    scientific_artifact_pair_identity,
)

if TYPE_CHECKING:
    from prpg.model.artifact import ModelArtifactStore
    from prpg.model.g3_evidence_store import (
        G3EvidenceStore,
        LoadedG3EvidenceAuthority,
    )

CanonicalTaskSource: TypeAlias = (
    InputIntegrityTaskView
    | CandidateViabilityTaskView
    | CandidateSelectionTaskView
    | SealedHoldoutVetoView
    | MacroStabilityTaskView
    | AdequacyCellTaskView
    | FourCellAdequacyHolmView
    | BlockCalibrationTaskView
    | ProductionFitSameKView
    | DependenceTaskView
    | FourCellDependenceHolmView
    | BridgeDriverTaskView
    | ArtifactIntegrityTaskView
    | G3TaskFailure
)

_COORDINATOR_SCHEMA_VERSION: Final = 1
_CHECKPOINT_STATE_SCHEMA_VERSION: Final = 1
_EXECUTION_MODE: Final[Literal["canonical"]] = "canonical"
_SELECTION_TASK: Final = "design_predictive_selection"
_ALLOWED_STATE_COUNTS: Final = frozenset({2, 3, 4, 5})
_ARTIFACT_TASK: Final = "artifact_integrity"
_PRE_ARTIFACT_GATE_ORDER: Final = G3_GATE_ORDER[:-1]
_PREFIX_SEAL = object()
_FINALIZATION_SEAL = object()
_SOURCE_TYPE_BY_TASK: Final[Mapping[str, type[object]]] = MappingProxyType(
    {
        "input_integrity": InputIntegrityTaskView,
        "design_candidate_viability": CandidateViabilityTaskView,
        "design_predictive_selection": CandidateSelectionTaskView,
        "sealed_holdout_veto": SealedHoldoutVetoView,
        "design_macro_stability": MacroStabilityTaskView,
        "design_latent_correctness": AdequacyCellTaskView,
        "design_refitted_adequacy": AdequacyCellTaskView,
        "design_block_calibration": BlockCalibrationTaskView,
        "design_fragmentation": DependenceTaskView,
        "design_dependence": DependenceTaskView,
        "production_fit_same_k": ProductionFitSameKView,
        "production_macro_stability": MacroStabilityTaskView,
        "production_latent_correctness": AdequacyCellTaskView,
        "production_refitted_adequacy": AdequacyCellTaskView,
        "production_block_calibration": BlockCalibrationTaskView,
        "production_fragmentation": DependenceTaskView,
        "production_dependence": DependenceTaskView,
        "four_cell_adequacy_holm": FourCellAdequacyHolmView,
        "four_cell_dependence_holm": FourCellDependenceHolmView,
        "bridge_resource_preflight": BridgeDriverTaskView,
        "bridge_tier_a_model_dgp": BridgeDriverTaskView,
        "bridge_tier_a_nonparametric_dgp": BridgeDriverTaskView,
        "bridge_tier_b_model_dgp": BridgeDriverTaskView,
        "bridge_tier_b_nonparametric_dgp": BridgeDriverTaskView,
        "actual_artifact_bridge": BridgeDriverTaskView,
        "artifact_integrity": ArtifactIntegrityTaskView,
    }
)


@dataclass(frozen=True, slots=True)
class CanonicalG3First25HandlerBundle:
    """Exact typed sources for the pre-artifact scientific task prefix.

    This is the canonical publication input.  It intentionally has no
    ``artifact_integrity`` field: the two report-bound artifacts cannot exist
    until the receipts, checkpoints, and terminal looks for these 25 tasks
    have been durably published.
    """

    input_integrity: InputIntegrityTaskView | G3TaskFailure
    design_candidate_viability: CandidateViabilityTaskView | G3TaskFailure
    design_predictive_selection: CandidateSelectionTaskView | G3TaskFailure
    sealed_holdout_veto: SealedHoldoutVetoView | G3TaskFailure
    design_macro_stability: MacroStabilityTaskView | G3TaskFailure
    design_latent_correctness: AdequacyCellTaskView | G3TaskFailure
    design_refitted_adequacy: AdequacyCellTaskView | G3TaskFailure
    design_block_calibration: BlockCalibrationTaskView | G3TaskFailure
    design_fragmentation: DependenceTaskView | G3TaskFailure
    design_dependence: DependenceTaskView | G3TaskFailure
    production_fit_same_k: ProductionFitSameKView | G3TaskFailure
    production_macro_stability: MacroStabilityTaskView | G3TaskFailure
    production_latent_correctness: AdequacyCellTaskView | G3TaskFailure
    production_refitted_adequacy: AdequacyCellTaskView | G3TaskFailure
    production_block_calibration: BlockCalibrationTaskView | G3TaskFailure
    production_fragmentation: DependenceTaskView | G3TaskFailure
    production_dependence: DependenceTaskView | G3TaskFailure
    four_cell_adequacy_holm: FourCellAdequacyHolmView | G3TaskFailure
    four_cell_dependence_holm: FourCellDependenceHolmView | G3TaskFailure
    bridge_resource_preflight: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_a_model_dgp: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_a_nonparametric_dgp: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_b_model_dgp: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_b_nonparametric_dgp: BridgeDriverTaskView | G3TaskFailure
    actual_artifact_bridge: BridgeDriverTaskView | G3TaskFailure

    def __post_init__(self) -> None:
        values = self.as_mapping()
        _validate_handler_sources(values, expected_order=_PRE_ARTIFACT_GATE_ORDER)
        _validate_shared_executions(values)

    def as_mapping(self) -> Mapping[str, CanonicalTaskSource]:
        """Return the immutable exact first-25 mapping in registry order."""

        return MappingProxyType(
            {task_id: getattr(self, task_id) for task_id in _PRE_ARTIFACT_GATE_ORDER}
        )


@dataclass(frozen=True, slots=True)
class CanonicalG3HandlerBundle:
    """Exact 26-source bundle; none of its fields can contain a callable.

    A :class:`G3TaskFailure` represents a bounded scientific calculation
    failure and can only lower authority.  If that task owns a registered
    sequential-look family, the failure is usable only when the task is
    dependency-blocked: an executable look owner must instead provide typed
    terminal look evidence.  Every other field is checked against the
    task-specific evaluator during construction.  Staged views remain distinct
    task objects while their sealed parent/source execution fingerprints must
    form the exact registered lineage.
    """

    input_integrity: InputIntegrityTaskView | G3TaskFailure
    design_candidate_viability: CandidateViabilityTaskView | G3TaskFailure
    design_predictive_selection: CandidateSelectionTaskView | G3TaskFailure
    sealed_holdout_veto: SealedHoldoutVetoView | G3TaskFailure
    design_macro_stability: MacroStabilityTaskView | G3TaskFailure
    design_latent_correctness: AdequacyCellTaskView | G3TaskFailure
    design_refitted_adequacy: AdequacyCellTaskView | G3TaskFailure
    design_block_calibration: BlockCalibrationTaskView | G3TaskFailure
    design_fragmentation: DependenceTaskView | G3TaskFailure
    design_dependence: DependenceTaskView | G3TaskFailure
    production_fit_same_k: ProductionFitSameKView | G3TaskFailure
    production_macro_stability: MacroStabilityTaskView | G3TaskFailure
    production_latent_correctness: AdequacyCellTaskView | G3TaskFailure
    production_refitted_adequacy: AdequacyCellTaskView | G3TaskFailure
    production_block_calibration: BlockCalibrationTaskView | G3TaskFailure
    production_fragmentation: DependenceTaskView | G3TaskFailure
    production_dependence: DependenceTaskView | G3TaskFailure
    four_cell_adequacy_holm: FourCellAdequacyHolmView | G3TaskFailure
    four_cell_dependence_holm: FourCellDependenceHolmView | G3TaskFailure
    bridge_resource_preflight: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_a_model_dgp: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_a_nonparametric_dgp: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_b_model_dgp: BridgeDriverTaskView | G3TaskFailure
    bridge_tier_b_nonparametric_dgp: BridgeDriverTaskView | G3TaskFailure
    actual_artifact_bridge: BridgeDriverTaskView | G3TaskFailure
    artifact_integrity: ArtifactIntegrityTaskView | G3TaskFailure

    def __post_init__(self) -> None:
        values = self.as_mapping()
        _validate_handler_sources(values, expected_order=G3_GATE_ORDER)
        _validate_shared_executions(values)

    def as_mapping(self) -> Mapping[str, CanonicalTaskSource]:
        """Return a read-only exact-registry mapping in topological order."""

        return MappingProxyType(
            {task_id: getattr(self, task_id) for task_id in G3_GATE_ORDER}
        )


def _validate_handler_sources(
    values: Mapping[str, CanonicalTaskSource],
    *,
    expected_order: tuple[str, ...],
) -> None:
    if tuple(values) != expected_order:
        raise AssertionError("canonical handler fields differ from G3 registry")
    for task_id, source in values.items():
        if callable(source):
            raise ModelError("canonical G3 handlers cannot contain callables")
        if isinstance(source, G3TaskFailure):
            _validated_failure(task_id, source)
            continue
        expected = _SOURCE_TYPE_BY_TASK[task_id]
        if not isinstance(source, expected):
            raise ModelError(
                "canonical G3 task has the wrong typed scientific source",
                details={
                    "task_id": task_id,
                    "expected": expected.__name__,
                    "actual": type(source).__name__,
                },
            )
        if not _is_valid_task_scoped_view(task_id, source):
            raise ModelError(
                "canonical G3 task view lacks sealed exact-task authority",
                details={"task_id": task_id},
            )
        # Structural and decision consistency validation, never a supplied
        # Boolean outcome.
        derive_g3_gate_decision(task_id, source)


def _validate_shared_executions(
    values: Mapping[str, CanonicalTaskSource],
) -> None:
    error = task_view_cross_binding_error(values)
    if error is not None:
        raise ModelError(
            "canonical G3 task-view lineage is invalid",
            details={"reason": error},
        )
    bridge_task_ids = (
        "bridge_resource_preflight",
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
        "actual_artifact_bridge",
    )
    bridge_views = tuple(
        source
        for task_id in bridge_task_ids
        if task_id in values
        and isinstance((source := values[task_id]), BridgeDriverTaskView)
    )
    if len(bridge_views) > 1 and any(
        source.shared_binding_fingerprint != bridge_views[0].shared_binding_fingerprint
        for source in bridge_views[1:]
    ):
        raise ModelError(
            "canonical G3 bridge task views must share one staged execution",
            details={"task_ids": list(bridge_task_ids)},
        )


def _is_valid_task_scoped_view(task_id: str, source: object) -> bool:
    if task_id == "input_integrity":
        return is_input_integrity_task_view(source)
    if task_id == "design_candidate_viability":
        return is_candidate_viability_task_view(source)
    if task_id == "design_predictive_selection":
        return is_candidate_selection_task_view(source)
    if task_id == "sealed_holdout_veto":
        return is_sealed_holdout_veto_view(source)
    if task_id in {"design_macro_stability", "production_macro_stability"}:
        return is_macro_stability_task_view(source)
    if task_id == "production_fit_same_k":
        return is_production_fit_same_k_view(source)
    if task_id in {
        "design_latent_correctness",
        "design_refitted_adequacy",
        "production_latent_correctness",
        "production_refitted_adequacy",
    }:
        return is_adequacy_cell_task_view(source)
    if task_id == "four_cell_adequacy_holm":
        return is_four_cell_adequacy_holm_view(source)
    if task_id in {
        "design_fragmentation",
        "design_dependence",
        "production_fragmentation",
        "production_dependence",
    }:
        return is_registered_dependence_task_view(source)
    if task_id == "four_cell_dependence_holm":
        return is_registered_four_cell_dependence_holm_view(source)
    if task_id in {"design_block_calibration", "production_block_calibration"}:
        return is_block_calibration_task_view(source) and source.task_id == task_id
    if task_id.startswith("bridge_") or task_id == "actual_artifact_bridge":
        return is_bridge_driver_task_view(source) and source.task_id == task_id
    if task_id == "artifact_integrity":
        return is_artifact_integrity_task_view(source)
    return False


@dataclass(frozen=True, slots=True)
class CanonicalG3State:
    """Verified durable prefix restored before canonical execution advances."""

    run_id: str
    task_results: Mapping[str, G3TaskResult]
    checkpoints: Mapping[str, CalibrationCheckpointReference]
    selected_n_states: int | None
    generation_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True, init=False)
class CanonicalG3ArtifactPublicationPrefix:
    """Sealed, reverified tasks-1-through-25 publication evidence.

    The object grants no generation authority.  Its process-local seal means
    report payload construction can distinguish a coordinator-produced prefix
    from a caller-built collection of otherwise plausible hashes.
    """

    generation_authorized: ClassVar[Literal[False]] = False
    verification_scope: ClassVar[Literal["canonical_g3_tasks_1_through_25"]] = (
        "canonical_g3_tasks_1_through_25"
    )

    run_id: str
    task_results: tuple[G3TaskResult, ...]
    checkpoints: tuple[CalibrationCheckpointReference, ...]
    task_result_fingerprints: Mapping[str, str]
    checkpoint_fingerprints: Mapping[str, str]
    planned_look_fingerprints: Mapping[str, tuple[str, ...]]
    selected_n_states: int | None
    binding: ScientificArtifactBinding
    provenance: ScientificArtifactProvenance
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    prefix_fingerprint: str
    _handlers: CanonicalG3First25HandlerBundle = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    @property
    def artifact_publication_ready(self) -> bool:
        """Return a coordinator-derived readiness decision."""

        return (
            self._seal is _PREFIX_SEAL
            and not self.failed_task_ids
            and not self.blocked_task_ids
            and self.selected_n_states in _ALLOWED_STATE_COUNTS
            and len(self.checkpoints) == len(_PRE_ARTIFACT_GATE_ORDER)
        )


@dataclass(frozen=True, slots=True)
class CanonicalG3CoordinationResult:
    """Terminal exact-graph result and its typed G3 evidence bundle."""

    run_id: str
    task_results: tuple[G3TaskResult, ...]
    checkpoints: tuple[CalibrationCheckpointReference, ...]
    evidence_bundle: G3EvidenceBundle
    selected_n_states: int | None
    completion_fingerprint: str
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    execution_mode: Literal["canonical"] = "canonical"

    @property
    def generation_authorized(self) -> Literal[False]:
        """G3 coordination is evidence-only, including when every task passes."""

        return False


@dataclass(frozen=True, slots=True, init=False)
class CanonicalG3FinalizationResult:
    """Completed two-stage G3 result ready for durable evidence publication."""

    coordination: CanonicalG3CoordinationResult
    completion_marker: LaunchMarker
    checkpoint_map: Mapping[str, CalibrationCheckpointReference]
    artifact_pair: LoadedScientificArtifactPair
    design_artifact: LoadedScientificModelArtifact
    production_artifact: LoadedScientificModelArtifact
    _seal: object = field(repr=False, compare=False)

    @property
    def generation_authorized(self) -> Literal[False]:
        """A sealed G3 finalization still grants no generation capability."""

        return False


@dataclass(frozen=True, slots=True)
class CanonicalG3PlannedLook:
    """One decision derived from a typed kernel or Tier-B result."""

    family_id: str
    look_index: int
    sample_count: int
    decision: LookDecision
    observed: Mapping[str, Any]


def coordinate_canonical_g3_first_25(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
    handlers: CanonicalG3First25HandlerBundle,
) -> CanonicalG3ArtifactPublicationPrefix:
    """Publish/resume tasks 1--25 and leave the launch actively running.

    The returned sealed prefix is the only input accepted by the report
    payload builder and finalizer.  In particular this entry point cannot
    accept either artifact or an artifact-integrity decision.
    """

    if not isinstance(handlers, CanonicalG3First25HandlerBundle):
        raise ModelError(
            "canonical G3 first stage requires the exact first-25 handler bundle"
        )
    reference, binding, persisted = _canonical_launch_context(
        store=store,
        lease=lease,
        live_preflight=live_preflight,
        checkpoint_store=checkpoint_store,
    )
    _validate_input_source_binding(
        handlers.input_integrity,
        run_id=lease.run_id,
        persisted_preflight_fingerprint=persisted,
        live_preflight=live_preflight,
    )
    existing = store.verify_task_results(lease.run_id, require_complete=False)
    if tuple(existing) == G3_GATE_ORDER:
        raise IntegrityError("canonical G3 launch is already finalized")
    if tuple(existing) != _PRE_ARTIFACT_GATE_ORDER[: len(existing)]:
        raise IntegrityError("canonical G3 task results are not a first-25 prefix")

    compatibility = CanonicalG3HandlerBundle(
        **cast(dict[str, Any], dict(handlers.as_mapping())),
        artifact_integrity=G3TaskFailure(
            _ARTIFACT_TASK,
            "ArtifactPublicationPending",
            "artifact publication follows the verified first-25 prefix",
        ),
    )
    state = restore_canonical_g3_state(
        store=store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        handlers=compatibility,
    )
    processed, checkpoints, selected_n_states = _advance_canonical_tasks(
        store=store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        binding=binding,
        task_specs=G3_TASK_REGISTRY[:-1],
        sources=handlers.as_mapping(),
        state=state,
    )
    marker = store.read_launch(lease.run_id)
    if marker.state != "running" or not lease.active:
        raise IntegrityError("canonical first stage unexpectedly completed its launch")
    return _mint_artifact_publication_prefix(
        run_id=lease.run_id,
        processed=processed,
        checkpoints=checkpoints,
        selected_n_states=selected_n_states,
        binding=ScientificArtifactBinding(
            config_fingerprint=reference.identity.config_fingerprint,
            processed_data_fingerprint=reference.identity.processed_data_fingerprint,
            calibration_input_fingerprint=(
                reference.identity.calibration_input_fingerprint
            ),
            calibration_run_fingerprint=lease.run_id,
            preflight_fingerprint=lease.preflight_fingerprint,
        ),
        provenance=ScientificArtifactProvenance(
            source_code_fingerprint=reference.identity.source_code_fingerprint,
            dependency_lock_fingerprint=(
                reference.identity.dependency_lock_fingerprint
            ),
        ),
        handlers=handlers,
        store=store,
    )


def canonical_g3_scientific_report_payload(
    prefix: CanonicalG3ArtifactPublicationPrefix,
    *,
    document_type: str,
    role: ArtifactRole,
    metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build a legacy/report-only payload from first-25 receipts.

    Canonical artifacts use the closed metric producers in
    :mod:`prpg.model.scientific_artifact_pair`; caller-supplied metrics from
    this compatibility API can never reach task 26.
    """

    _require_artifact_ready_prefix(prefix)
    task_ids = scientific_report_task_ids(document_type, role)
    if any(task_id not in _PRE_ARTIFACT_GATE_ORDER for task_id in task_ids):
        raise AssertionError("a scientific report unexpectedly depends on task 26")
    families = scientific_report_planned_look_families(document_type, role)
    return scientific_report_payload(
        document_type=document_type,
        role=role,
        task_result_fingerprints={
            task_id: prefix.task_result_fingerprints[task_id] for task_id in task_ids
        },
        checkpoint_fingerprints={
            task_id: prefix.checkpoint_fingerprints[task_id] for task_id in task_ids
        },
        planned_look_fingerprints={
            family_id: prefix.planned_look_fingerprints[family_id]
            for family_id in families
        },
        metrics=metrics,
    )


def finalize_canonical_g3_artifacts(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
    prefix: CanonicalG3ArtifactPublicationPrefix,
    artifact_pair: LoadedScientificArtifactPair,
) -> CanonicalG3FinalizationResult:
    """Validate the exact pair receipt, publish task 26, and complete G3."""

    reference, binding, persisted = _canonical_launch_context(
        store=store,
        lease=lease,
        live_preflight=live_preflight,
        checkpoint_store=checkpoint_store,
    )
    del persisted  # already checked by the launch-context verifier
    _require_artifact_ready_prefix(prefix)
    if prefix.run_id != lease.run_id or prefix.binding != ScientificArtifactBinding(
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=reference.identity.calibration_input_fingerprint,
        calibration_run_fingerprint=lease.run_id,
        preflight_fingerprint=lease.preflight_fingerprint,
    ):
        raise IntegrityError("artifact-publication prefix belongs to another launch")
    expected_provenance = ScientificArtifactProvenance(
        source_code_fingerprint=reference.identity.source_code_fingerprint,
        dependency_lock_fingerprint=reference.identity.dependency_lock_fingerprint,
    )
    if prefix.provenance != expected_provenance:
        raise IntegrityError("artifact-publication prefix provenance is stale")
    if not is_verified_scientific_artifact_pair(artifact_pair):
        raise ModelError("G3 finalization requires a loaded scientific pair receipt")
    pair_identity = scientific_artifact_pair_identity(artifact_pair)
    if (
        artifact_pair._prefix is not prefix
        or artifact_pair.run_id != prefix.run_id
        or artifact_pair.prefix_fingerprint != prefix.prefix_fingerprint
        or artifact_pair.selected_k != prefix.selected_n_states
        or pair_identity["task_result_fingerprints"]
        != dict(prefix.task_result_fingerprints)
        or pair_identity["checkpoint_fingerprints"]
        != dict(prefix.checkpoint_fingerprints)
    ):
        raise IntegrityError("scientific artifact pair differs from first-25 prefix")
    design_artifact = artifact_pair.design
    production_artifact = artifact_pair.production
    _verify_artifact_prefix_reports(
        design_artifact,
        expected_role="design",
        prefix=prefix,
    )
    _verify_artifact_prefix_reports(
        production_artifact,
        expected_role="production",
        prefix=prefix,
    )
    artifact_source = ArtifactIntegrityDecisionEvidence(artifact_pair)
    actual_bridge_parent = prefix._handlers.actual_artifact_bridge
    if not is_bridge_driver_task_view(actual_bridge_parent):
        raise IntegrityError("artifact-ready prefix lacks its sealed bridge parent")
    artifact_view = build_artifact_integrity_task_view(
        artifact_source,
        actual_bridge_parent,
        run_id=lease.run_id,
        binding_fingerprint=_fingerprint(
            {
                "schema_id": "prpg-legacy-handler-artifact-task-binding-v1",
                "run_id": lease.run_id,
                "prefix_fingerprint": prefix.prefix_fingerprint,
                "actual_bridge_parent": actual_bridge_parent.view_fingerprint,
            }
        ),
    )
    artifact_decision = derive_g3_gate_decision(_ARTIFACT_TASK, artifact_view)
    if not artifact_decision.passed:
        raise ModelError("typed design/production artifact integrity did not pass")

    handlers = CanonicalG3HandlerBundle(
        **cast(dict[str, Any], dict(prefix._handlers.as_mapping())),
        artifact_integrity=artifact_view,
    )
    state = restore_canonical_g3_state(
        store=store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        handlers=handlers,
    )
    _verify_restored_prefix(prefix, state, store=store)
    processed = dict(state.task_results)
    checkpoints = dict(state.checkpoints)
    selected_n_states = state.selected_n_states
    if tuple(processed) == _PRE_ARTIFACT_GATE_ORDER:
        document = _decision_document(artifact_decision, artifact_view)
        decision_fingerprint = _fingerprint(document)
        checkpoint = _publish_checkpoint(
            checkpoint_store=checkpoint_store,
            task_id=_ARTIFACT_TASK,
            binding=binding,
            dependencies=(checkpoints["actual_artifact_bridge"],),
            selected_n_states=selected_n_states,
            decision_document=document,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints={},
        )
        result = store.write_task_result(
            lease,
            task_id=_ARTIFACT_TASK,
            status="passed",
            payload=_decision_payload(
                checkpoint_fingerprint=checkpoint.fingerprint,
                decision_fingerprint=decision_fingerprint,
                decision=artifact_decision,
                selected_n_states=selected_n_states,
                planned_look_fingerprints={},
            ),
        )
        processed[_ARTIFACT_TASK] = result
        checkpoints[_ARTIFACT_TASK] = checkpoint
    elif tuple(processed) != G3_GATE_ORDER:
        raise IntegrityError("canonical artifact finalization found an invalid prefix")

    decisions, failures = _derived_sources(handlers)
    bundle = assemble_typed_g3_evidence_bundle(
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=(
            reference.identity.calibration_input_fingerprint
        ),
        decisions=decisions,
        failures=failures,
    )
    _verify_bundle_matches_results(bundle, processed)
    if not bundle.passed or bundle.generation_authorized is not False:
        raise IntegrityError(
            "two-stage G3 finalization did not produce passed evidence"
        )
    completion = store.complete_launch(lease)
    ordered = tuple(processed[task_id] for task_id in G3_GATE_ORDER)
    coordination = CanonicalG3CoordinationResult(
        run_id=lease.run_id,
        task_results=ordered,
        checkpoints=tuple(checkpoints[task_id] for task_id in G3_GATE_ORDER),
        evidence_bundle=bundle,
        selected_n_states=selected_n_states,
        completion_fingerprint=completion.fingerprint,
        failed_task_ids=(),
        blocked_task_ids=(),
    )
    final = object.__new__(CanonicalG3FinalizationResult)
    for name, value in (
        ("coordination", coordination),
        ("completion_marker", completion),
        ("checkpoint_map", MappingProxyType(dict(checkpoints))),
        ("artifact_pair", artifact_pair),
        ("design_artifact", design_artifact),
        ("production_artifact", production_artifact),
        ("_seal", _FINALIZATION_SEAL),
    ):
        object.__setattr__(final, name, value)
    return final


def publish_finalized_canonical_g3_evidence(
    finalization: CanonicalG3FinalizationResult,
    *,
    evidence_store: G3EvidenceStore,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    model_store: ModelArtifactStore,
) -> LoadedG3EvidenceAuthority:
    """Publish durable passed evidence from an exact G3 finalization only."""

    from prpg.model.g3_evidence_store import G3EvidenceStore

    if (
        not isinstance(finalization, CanonicalG3FinalizationResult)
        or finalization._seal is not _FINALIZATION_SEAL
        or not finalization.coordination.evidence_bundle.passed
        or finalization.generation_authorized is not False
    ):
        raise ModelError("durable G3 publication requires sealed finalization")
    if not isinstance(evidence_store, G3EvidenceStore):
        raise ModelError("durable G3 publication requires G3EvidenceStore")
    coordination = finalization.coordination
    marker = finalization.completion_marker
    if (
        marker.state != "completed"
        or marker.run_id != coordination.run_id
        or marker.fingerprint != coordination.completion_fingerprint
        or tuple(finalization.checkpoint_map) != G3_GATE_ORDER
        or not is_verified_scientific_artifact_pair(finalization.artifact_pair)
        or finalization.artifact_pair.design is not finalization.design_artifact
        or finalization.artifact_pair.production is not finalization.production_artifact
    ):
        raise IntegrityError("sealed G3 finalization completion evidence changed")
    return evidence_store.publish(
        bundle=coordination.evidence_bundle,
        run_store=run_store,
        run_id=coordination.run_id,
        completion_marker=marker,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        checkpoints=finalization.checkpoint_map,
        model_store=model_store,
        design_artifact=finalization.design_artifact,
        production_artifact=finalization.production_artifact,
    )


def _canonical_launch_context(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
) -> tuple[
    CalibrationRunReference,
    CalibrationCheckpointBinding,
    str,
]:
    if not isinstance(store, CalibrationRunStore):
        raise ModelError("canonical G3 store has the wrong type")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not store
        or not lease.active
    ):
        raise IntegrityError("an active canonical launch lease is required")
    marker = store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != lease.preflight_fingerprint
        or marker.workers != lease.workers
    ):
        raise IntegrityError("canonical launch lease does not match its marker")
    if not isinstance(live_preflight, CanonicalG3Preflight):
        raise ModelError("canonical G3 handlers require typed live preflight")
    verify_live_canonical_g3_preflight(live_preflight)
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("canonical G3 handlers require a checkpoint store")
    persisted = canonical_resume_preflight_fingerprint(
        store,
        run_id=lease.run_id,
        live_preflight=live_preflight,
    )
    if persisted != lease.preflight_fingerprint:
        raise IntegrityError("canonical persisted preflight differs from launch lease")
    reference = store.verify(lease.run_id)
    if (
        reference.identity.config_fingerprint != live_preflight.config_fingerprint
        or reference.identity.processed_data_fingerprint
        != live_preflight.processed_data_fingerprint
        or reference.identity.calibration_input_fingerprint
        != live_preflight.calibration_input_fingerprint
        or reference.identity.source_code_fingerprint
        != live_preflight.source_code_fingerprint
        or reference.identity.dependency_lock_fingerprint
        != live_preflight.dependency_lock_fingerprint
    ):
        raise IntegrityError("canonical preflight/run/lease identities disagree")
    binding = CalibrationCheckpointBinding(
        run_id=lease.run_id,
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=(
            reference.identity.calibration_input_fingerprint
        ),
        preflight_fingerprint=lease.preflight_fingerprint,
    )
    return reference, binding, persisted


def _advance_canonical_tasks(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    checkpoint_store: CalibrationCheckpointStore,
    binding: CalibrationCheckpointBinding,
    task_specs: Sequence[G3TaskSpec],
    sources: Mapping[str, CanonicalTaskSource],
    state: CanonicalG3State,
) -> tuple[
    dict[str, G3TaskResult],
    dict[str, CalibrationCheckpointReference],
    int | None,
]:
    processed = dict(state.task_results)
    checkpoints = dict(state.checkpoints)
    selected_n_states = state.selected_n_states
    for spec in task_specs[len(processed) :]:
        blocked_by = tuple(
            dependency
            for dependency in spec.dependencies
            if processed[dependency].status != "passed"
        )
        if blocked_by:
            processed[spec.task_id] = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="blocked",
                payload=_blocked_payload(
                    blocked_by=blocked_by,
                    selected_n_states=selected_n_states,
                ),
            )
            continue
        source = sources[spec.task_id]
        if isinstance(source, G3TaskFailure):
            receipts = _commit_canonical_looks(
                store=store,
                lease=lease,
                task_id=spec.task_id,
                proposed=canonical_planned_looks(spec.task_id, source),
            )
            processed[spec.task_id] = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="failed",
                payload=_failure_payload(
                    source,
                    selected_n_states=selected_n_states,
                    planned_look_fingerprints=_look_fingerprint_mapping(receipts),
                ),
            )
            continue
        decision = derive_g3_gate_decision(spec.task_id, source)
        next_k = _next_selected_n_states(
            task_id=spec.task_id,
            decision=decision,
            frozen=selected_n_states,
        )
        receipts = _commit_canonical_looks(
            store=store,
            lease=lease,
            task_id=spec.task_id,
            proposed=canonical_planned_looks(spec.task_id, source),
        )
        look_fingerprints = _look_fingerprint_mapping(receipts)
        decision_document = _decision_document(decision, source)
        decision_fingerprint = _fingerprint(decision_document)
        if not decision.passed:
            processed[spec.task_id] = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="failed",
                payload=_decision_payload(
                    checkpoint_fingerprint=None,
                    decision_fingerprint=decision_fingerprint,
                    decision=decision,
                    selected_n_states=next_k,
                    planned_look_fingerprints=look_fingerprints,
                ),
            )
            if spec.task_id == _SELECTION_TASK:
                selected_n_states = None
            continue
        checkpoint = _publish_checkpoint(
            checkpoint_store=checkpoint_store,
            task_id=spec.task_id,
            binding=binding,
            dependencies=tuple(
                checkpoints[dependency] for dependency in spec.dependencies
            ),
            selected_n_states=next_k,
            decision_document=decision_document,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints=look_fingerprints,
        )
        processed[spec.task_id] = store.write_task_result(
            lease,
            task_id=spec.task_id,
            status="passed",
            payload=_decision_payload(
                checkpoint_fingerprint=checkpoint.fingerprint,
                decision_fingerprint=decision_fingerprint,
                decision=decision,
                selected_n_states=next_k,
                planned_look_fingerprints=look_fingerprints,
            ),
        )
        checkpoints[spec.task_id] = checkpoint
        selected_n_states = next_k
    return processed, checkpoints, selected_n_states


def _mint_artifact_publication_prefix(
    *,
    run_id: str,
    processed: Mapping[str, G3TaskResult],
    checkpoints: Mapping[str, CalibrationCheckpointReference],
    selected_n_states: int | None,
    binding: ScientificArtifactBinding,
    provenance: ScientificArtifactProvenance,
    handlers: CanonicalG3First25HandlerBundle,
    store: CalibrationRunStore,
) -> CanonicalG3ArtifactPublicationPrefix:
    if tuple(processed) != _PRE_ARTIFACT_GATE_ORDER:
        raise IntegrityError("canonical first stage did not publish exactly 25 tasks")
    ordered = tuple(processed[task_id] for task_id in _PRE_ARTIFACT_GATE_ORDER)
    failed = tuple(item.task_id for item in ordered if item.status == "failed")
    blocked = tuple(item.task_id for item in ordered if item.status == "blocked")
    ready = not failed and not blocked
    if ready and tuple(checkpoints) != _PRE_ARTIFACT_GATE_ORDER:
        raise IntegrityError("passing first-25 prefix lacks exact checkpoints")
    all_looks = store.verify_planned_look_receipts(
        run_id,
        require_terminal=ready,
        task_results=dict(processed),
    )
    task_fingerprints = MappingProxyType(
        {
            task_id: processed[task_id].fingerprint
            for task_id in _PRE_ARTIFACT_GATE_ORDER
        }
    )
    checkpoint_fingerprints = MappingProxyType(
        {
            task_id: checkpoints[task_id].fingerprint
            for task_id in _PRE_ARTIFACT_GATE_ORDER
            if task_id in checkpoints
        }
    )
    look_fingerprints = MappingProxyType(
        {
            family_id: tuple(receipt.fingerprint for receipt in receipts)
            for family_id, receipts in all_looks.items()
        }
    )
    identity = {
        "binding": binding.as_dict(),
        "checkpoint_fingerprints": dict(checkpoint_fingerprints),
        "planned_look_fingerprints": {
            key: list(value) for key, value in look_fingerprints.items()
        },
        "provenance": provenance.as_dict(),
        "run_id": run_id,
        "selected_n_states": selected_n_states,
        "task_result_fingerprints": dict(task_fingerprints),
    }
    prefix = object.__new__(CanonicalG3ArtifactPublicationPrefix)
    for name, value in (
        ("run_id", run_id),
        ("task_results", ordered),
        (
            "checkpoints",
            tuple(
                checkpoints[task_id]
                for task_id in _PRE_ARTIFACT_GATE_ORDER
                if task_id in checkpoints
            ),
        ),
        ("task_result_fingerprints", task_fingerprints),
        ("checkpoint_fingerprints", checkpoint_fingerprints),
        ("planned_look_fingerprints", look_fingerprints),
        ("selected_n_states", selected_n_states),
        ("binding", binding),
        ("provenance", provenance),
        ("failed_task_ids", failed),
        ("blocked_task_ids", blocked),
        ("prefix_fingerprint", _fingerprint(identity)),
        ("_handlers", handlers),
        ("_seal", _PREFIX_SEAL),
    ):
        object.__setattr__(prefix, name, value)
    return prefix


def _require_artifact_ready_prefix(
    prefix: CanonicalG3ArtifactPublicationPrefix,
) -> None:
    if (
        not isinstance(prefix, CanonicalG3ArtifactPublicationPrefix)
        or prefix._seal is not _PREFIX_SEAL
    ):
        raise ModelError("artifact publication requires a sealed first-25 prefix")
    if not prefix.artifact_publication_ready:
        raise ModelError("failed or blocked first-25 prefix cannot publish artifacts")
    expected_identity = {
        "binding": prefix.binding.as_dict(),
        "checkpoint_fingerprints": dict(prefix.checkpoint_fingerprints),
        "planned_look_fingerprints": {
            key: list(value) for key, value in prefix.planned_look_fingerprints.items()
        },
        "provenance": prefix.provenance.as_dict(),
        "run_id": prefix.run_id,
        "selected_n_states": prefix.selected_n_states,
        "task_result_fingerprints": dict(prefix.task_result_fingerprints),
    }
    if prefix.prefix_fingerprint != _fingerprint(expected_identity):
        raise IntegrityError("artifact-publication prefix identity changed")


def _verify_artifact_prefix_reports(
    artifact: LoadedScientificModelArtifact,
    *,
    expected_role: ArtifactRole,
    prefix: CanonicalG3ArtifactPublicationPrefix,
) -> None:
    if not is_verified_scientific_model_artifact(artifact):
        raise ModelError("G3 finalization requires typed loaded scientific artifacts")
    if (
        artifact.role != expected_role
        or artifact.selected_k != prefix.selected_n_states
        or artifact.binding != prefix.binding
        or artifact.provenance != prefix.provenance
    ):
        raise IntegrityError("scientific artifact role/K/binding/provenance is stale")
    for path, document_type in required_scientific_documents(expected_role).items():
        task_ids = scientific_report_task_ids(document_type, expected_role)
        families = scientific_report_planned_look_families(
            document_type,
            expected_role,
        )
        payload = artifact.reports[path]["payload"]
        if payload["task_result_fingerprints"] != {
            task_id: prefix.task_result_fingerprints[task_id] for task_id in task_ids
        }:
            raise IntegrityError(
                "scientific report task-result bindings differ from first-25 prefix"
            )
        if payload["checkpoint_fingerprints"] != {
            task_id: prefix.checkpoint_fingerprints[task_id] for task_id in task_ids
        }:
            raise IntegrityError(
                "scientific report checkpoint bindings differ from first-25 prefix"
            )
        actual_looks = {
            family_id: tuple(fingerprints)
            for family_id, fingerprints in cast(
                Mapping[str, Sequence[str]],
                payload["planned_look_fingerprints"],
            ).items()
        }
        if actual_looks != {
            family_id: prefix.planned_look_fingerprints[family_id]
            for family_id in families
        }:
            raise IntegrityError(
                "scientific report planned-look bindings differ from first-25 prefix"
            )


def _verify_restored_prefix(
    prefix: CanonicalG3ArtifactPublicationPrefix,
    state: CanonicalG3State,
    *,
    store: CalibrationRunStore,
) -> None:
    first = {
        task_id: state.task_results[task_id]
        for task_id in _PRE_ARTIFACT_GATE_ORDER
        if task_id in state.task_results
    }
    if tuple(first) != _PRE_ARTIFACT_GATE_ORDER:
        raise IntegrityError("durable first-25 task prefix is incomplete")
    if {task_id: result.fingerprint for task_id, result in first.items()} != dict(
        prefix.task_result_fingerprints
    ):
        raise IntegrityError("durable first-25 task receipts changed")
    first_checkpoints = {
        task_id: state.checkpoints[task_id].fingerprint
        for task_id in _PRE_ARTIFACT_GATE_ORDER
        if task_id in state.checkpoints
    }
    if first_checkpoints != dict(prefix.checkpoint_fingerprints):
        raise IntegrityError("durable first-25 checkpoints changed")
    looks = store.verify_planned_look_receipts(
        prefix.run_id,
        require_terminal=True,
        task_results=dict(state.task_results),
    )
    if {
        family_id: tuple(receipt.fingerprint for receipt in receipts)
        for family_id, receipts in looks.items()
    } != dict(prefix.planned_look_fingerprints):
        raise IntegrityError("durable first-25 planned looks changed")
    if state.selected_n_states != prefix.selected_n_states:
        raise IntegrityError("durable first-25 selected K changed")


def coordinate_canonical_g3_handlers(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
    handlers: CanonicalG3HandlerBundle,
) -> CanonicalG3CoordinationResult:
    """Legacy fail-closed one-shot compatibility publisher.

    This wrapper may retain task-graph failure tests and interrupted historical
    runs, but it cannot accept a prebuilt artifact pair and therefore can
    never complete passing artifact-integrity evidence.  Neither G3 path
    grants generation authority.  New canonical execution must use
    :func:`coordinate_canonical_g3_first_25` followed by artifact publication
    and :func:`finalize_canonical_g3_artifacts`.
    """

    if not isinstance(store, CalibrationRunStore):
        raise ModelError("canonical G3 store has the wrong type")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not store
        or not lease.active
    ):
        raise IntegrityError("an active canonical launch lease is required")
    if not isinstance(live_preflight, CanonicalG3Preflight):
        raise ModelError("canonical G3 handlers require typed live preflight")
    verify_live_canonical_g3_preflight(live_preflight)
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("canonical G3 handlers require a checkpoint store")
    if not isinstance(handlers, CanonicalG3HandlerBundle):
        raise ModelError("canonical G3 execution requires the closed handler bundle")
    if not isinstance(handlers.artifact_integrity, G3TaskFailure):
        raise ModelError(
            "prebuilt artifact-integrity evidence is prohibited; use the "
            "two-stage canonical G3 publication API"
        )
    persisted_preflight_fingerprint = canonical_resume_preflight_fingerprint(
        store,
        run_id=lease.run_id,
        live_preflight=live_preflight,
    )
    if persisted_preflight_fingerprint != lease.preflight_fingerprint:
        raise IntegrityError("canonical persisted preflight differs from launch lease")
    reference = store.verify(lease.run_id)
    if (
        reference.identity.config_fingerprint != live_preflight.config_fingerprint
        or reference.identity.processed_data_fingerprint
        != live_preflight.processed_data_fingerprint
        or reference.identity.calibration_input_fingerprint
        != live_preflight.calibration_input_fingerprint
        or reference.identity.source_code_fingerprint
        != live_preflight.source_code_fingerprint
        or reference.identity.dependency_lock_fingerprint
        != live_preflight.dependency_lock_fingerprint
    ):
        raise IntegrityError("canonical preflight/run/lease identities disagree")
    binding = CalibrationCheckpointBinding(
        run_id=lease.run_id,
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=(
            reference.identity.calibration_input_fingerprint
        ),
        preflight_fingerprint=lease.preflight_fingerprint,
    )
    _validate_input_source_binding(
        handlers.input_integrity,
        run_id=lease.run_id,
        persisted_preflight_fingerprint=persisted_preflight_fingerprint,
        live_preflight=live_preflight,
    )

    state = restore_canonical_g3_state(
        store=store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        handlers=handlers,
    )
    processed = dict(state.task_results)
    checkpoints = dict(state.checkpoints)
    selected_n_states = state.selected_n_states
    sources = handlers.as_mapping()

    for spec in G3_TASK_REGISTRY[len(processed) :]:
        blocked_by = tuple(
            dependency
            for dependency in spec.dependencies
            if processed[dependency].status != "passed"
        )
        if blocked_by:
            result = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="blocked",
                payload=_blocked_payload(
                    blocked_by=blocked_by,
                    selected_n_states=selected_n_states,
                ),
            )
            processed[spec.task_id] = result
            continue

        source = sources[spec.task_id]
        if isinstance(source, G3TaskFailure):
            # This returns an empty tuple for ordinary tasks and fails closed
            # for planned-look owners.  A generic exception record cannot
            # prove that even the first registered sample count was executed.
            looks = canonical_planned_looks(spec.task_id, source)
            receipts = _commit_canonical_looks(
                store=store,
                lease=lease,
                task_id=spec.task_id,
                proposed=looks,
            )
            payload = _failure_payload(
                source,
                selected_n_states=selected_n_states,
                planned_look_fingerprints=_look_fingerprint_mapping(receipts),
            )
            processed[spec.task_id] = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="failed",
                payload=payload,
            )
            continue

        decision = derive_g3_gate_decision(spec.task_id, source)
        next_k = _next_selected_n_states(
            task_id=spec.task_id,
            decision=decision,
            frozen=selected_n_states,
        )
        looks = canonical_planned_looks(spec.task_id, source)
        receipts = _commit_canonical_looks(
            store=store,
            lease=lease,
            task_id=spec.task_id,
            proposed=looks,
        )
        look_fingerprints = _look_fingerprint_mapping(receipts)
        decision_document = _decision_document(decision, source)
        decision_fingerprint = _fingerprint(decision_document)

        if not decision.passed:
            processed[spec.task_id] = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="failed",
                payload=_decision_payload(
                    checkpoint_fingerprint=None,
                    decision_fingerprint=decision_fingerprint,
                    decision=decision,
                    selected_n_states=next_k,
                    planned_look_fingerprints=look_fingerprints,
                ),
            )
            if spec.task_id == _SELECTION_TASK:
                selected_n_states = None
            continue

        dependency_checkpoints = tuple(
            checkpoints[dependency] for dependency in spec.dependencies
        )
        checkpoint = _publish_checkpoint(
            checkpoint_store=checkpoint_store,
            task_id=spec.task_id,
            binding=binding,
            dependencies=dependency_checkpoints,
            selected_n_states=next_k,
            decision_document=decision_document,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints=look_fingerprints,
        )
        result = store.write_task_result(
            lease,
            task_id=spec.task_id,
            status="passed",
            payload=_decision_payload(
                checkpoint_fingerprint=checkpoint.fingerprint,
                decision_fingerprint=decision_fingerprint,
                decision=decision,
                selected_n_states=next_k,
                planned_look_fingerprints=look_fingerprints,
            ),
        )
        processed[spec.task_id] = result
        checkpoints[spec.task_id] = checkpoint
        selected_n_states = next_k

    decisions, failures = _derived_sources(handlers)
    bundle = assemble_typed_g3_evidence_bundle(
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=(
            reference.identity.calibration_input_fingerprint
        ),
        decisions=decisions,
        failures=failures,
    )
    _verify_bundle_matches_results(bundle, processed)
    completion = store.complete_launch(lease)
    ordered = tuple(processed[task_id] for task_id in G3_GATE_ORDER)
    return CanonicalG3CoordinationResult(
        run_id=lease.run_id,
        task_results=ordered,
        checkpoints=tuple(
            checkpoints[task_id] for task_id in G3_GATE_ORDER if task_id in checkpoints
        ),
        evidence_bundle=bundle,
        selected_n_states=selected_n_states,
        completion_fingerprint=completion.fingerprint,
        failed_task_ids=tuple(
            item.task_id for item in ordered if item.status == "failed"
        ),
        blocked_task_ids=tuple(
            item.task_id for item in ordered if item.status == "blocked"
        ),
    )


def restore_canonical_g3_state(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    checkpoint_store: CalibrationCheckpointStore,
    handlers: CanonicalG3HandlerBundle,
) -> CanonicalG3State:
    """Verify and restore the exact durable canonical task prefix.

    The durable result set must be a registry prefix.  This additional rule is
    stricter than the run store's dependency-only validation and prevents a
    resumed coordinator from observing out-of-order receipts written by some
    other component.
    """

    if not isinstance(store, CalibrationRunStore):
        raise ModelError("canonical G3 store has the wrong type")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not store
        or not lease.active
    ):
        raise IntegrityError("an active canonical launch lease is required")
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("canonical G3 state restoration requires a checkpoint store")
    if not isinstance(handlers, CanonicalG3HandlerBundle):
        raise ModelError("canonical G3 state requires the closed handler bundle")
    existing = store.verify_task_results(lease.run_id, require_complete=False)
    existing_ids = tuple(existing)
    if existing_ids != G3_GATE_ORDER[: len(existing_ids)]:
        raise IntegrityError("canonical G3 task results are not a registry prefix")
    all_looks = store.verify_planned_look_receipts(lease.run_id, require_terminal=False)
    checkpoints: dict[str, CalibrationCheckpointReference] = {}
    selected_n_states: int | None = None
    sources = handlers.as_mapping()
    processed: dict[str, G3TaskResult] = {}
    for task_id in existing_ids:
        result = existing[task_id]
        spec = next(item for item in G3_TASK_REGISTRY if item.task_id == task_id)
        blocked_by = tuple(
            dependency
            for dependency in spec.dependencies
            if processed[dependency].status != "passed"
        )
        if blocked_by:
            expected = _blocked_payload(
                blocked_by=blocked_by,
                selected_n_states=selected_n_states,
            )
            if result.status != "blocked" or dict(result.payload) != expected:
                raise IntegrityError(
                    "canonical blocked result is not coordinator-derived"
                )
            processed[task_id] = result
            continue
        if result.status == "blocked":
            raise IntegrityError(
                "canonical task is blocked despite passed dependencies"
            )

        source = sources[task_id]
        if isinstance(source, G3TaskFailure):
            receipts = _receipts_for_task(task_id, all_looks)
            expected_looks = canonical_planned_looks(task_id, source)
            _compare_receipts_to_looks(receipts, expected_looks)
            expected = _failure_payload(
                source,
                selected_n_states=selected_n_states,
                planned_look_fingerprints=_look_fingerprint_mapping(receipts),
            )
            if result.status != "failed" or dict(result.payload) != expected:
                raise IntegrityError("canonical scientific-failure receipt changed")
            processed[task_id] = result
            continue

        decision = derive_g3_gate_decision(task_id, source)
        next_k = _next_selected_n_states(
            task_id=task_id,
            decision=decision,
            frozen=selected_n_states,
        )
        receipts = _receipts_for_task(task_id, all_looks)
        expected_looks = canonical_planned_looks(task_id, source)
        _compare_receipts_to_looks(receipts, expected_looks)
        look_fingerprints = _look_fingerprint_mapping(receipts)
        document = _decision_document(decision, source)
        decision_fingerprint = _fingerprint(document)
        checkpoint_fingerprint = result.payload.get("checkpoint_fingerprint")
        expected = _decision_payload(
            checkpoint_fingerprint=(
                checkpoint_fingerprint
                if isinstance(checkpoint_fingerprint, str)
                else None
            ),
            decision_fingerprint=decision_fingerprint,
            decision=decision,
            selected_n_states=next_k,
            planned_look_fingerprints=look_fingerprints,
        )
        if dict(result.payload) != expected:
            raise IntegrityError("canonical task payload differs from typed source")
        expected_status = "passed" if decision.passed else "failed"
        if result.status != expected_status:
            raise IntegrityError("canonical task status differs from typed decision")
        if decision.passed:
            checkpoint = checkpoint_store.load_from_task_result(
                store,
                run_id=lease.run_id,
                task_id=task_id,
            )
            _verify_checkpoint_state(
                checkpoint,
                decision_document=document,
                decision_fingerprint=decision_fingerprint,
                planned_look_fingerprints=look_fingerprints,
            )
            checkpoints[task_id] = checkpoint
            selected_n_states = next_k
        elif task_id == _SELECTION_TASK:
            selected_n_states = None
        processed[task_id] = result
    return CanonicalG3State(
        run_id=lease.run_id,
        task_results=MappingProxyType(processed),
        checkpoints=MappingProxyType(checkpoints),
        selected_n_states=selected_n_states,
    )


def canonical_planned_looks(
    task_id: str,
    source: CanonicalTaskSource,
) -> tuple[CanonicalG3PlannedLook, ...]:
    """Derive exact registered look receipts from typed scientific evidence."""

    specs = tuple(
        item for item in G3_PLANNED_LOOK_REGISTRY if item.owner_task_id == task_id
    )
    if not specs:
        return ()
    if isinstance(source, G3TaskFailure):
        raise ModelError(
            "registered planned-look owner requires typed terminal look evidence",
            details={"task_id": task_id},
        )
    if task_id in {"design_block_calibration", "production_block_calibration"}:
        if not is_block_calibration_task_view(source) or source.task_id != task_id:
            raise ModelError("kernel look owner lacks block-calibration evidence")
        kernels = (source.source.monthly_kernel, source.source.daily_kernel)
        result: list[CanonicalG3PlannedLook] = []
        for spec, kernel in zip(specs, kernels, strict=True):
            result.extend(_kernel_looks(spec.family_id, spec.sample_counts, kernel))
        return tuple(result)
    if task_id in {
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    }:
        if not isinstance(source, BridgeDriverTaskView) or source.task_id != task_id:
            raise ModelError("Tier-B look owner lacks its sealed driver task view")
        # Re-run the task evaluator before publishing so a hand-built object,
        # wrong task view, or changed local checkpoint-derived sequence cannot
        # reach the durable G3 look store.
        derive_g3_gate_decision(task_id, source)
        spec = specs[0]
        return _tier_b_looks(spec.family_id, spec.sample_counts, source)
    raise AssertionError("registered look owner has no code-owned publisher")


def _kernel_looks(
    family_id: str,
    schedule: tuple[int, ...],
    kernel: KernelCalibrationExecution,
) -> tuple[CanonicalG3PlannedLook, ...]:
    if not isinstance(kernel, KernelCalibrationExecution):
        raise ModelError("kernel planned looks require typed execution evidence")
    completed = kernel.completed_looks
    if not completed or len(completed) > len(schedule):
        raise ModelError("kernel planned-look sequence is empty or too long")
    result: list[CanonicalG3PlannedLook] = []
    for index, look in enumerate(completed, start=1):
        if (
            look.look_index != index - 1
            or look.windows != schedule[index - 1]
            or look.precision.windows != look.windows
            or look.passed
            != (
                look.maximum_annualized_half_width
                <= CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
            )
        ):
            raise ModelError("kernel planned-look evidence is inconsistent")
        is_last = index == len(completed)
        if look.passed and not is_last:
            raise ModelError("kernel execution retained work after a passing look")
        if not look.passed and is_last and index != len(schedule):
            raise ModelError("kernel execution stopped before pass or maximum look")
        decision: LookDecision = (
            "pass" if is_last and look.passed else "fail" if is_last else "continue"
        )
        precision = look.precision
        observed = {
            "calibration_stream_fingerprint": (
                precision.calibration_stream_fingerprint
            ),
            "kernel_sample_fingerprint": precision.kernel_sample_fingerprint,
            "maximum_annualized_half_width": (look.maximum_annualized_half_width),
            "model_artifact_fingerprint": precision.model_artifact_fingerprint,
            "precision_passed": look.passed,
            "source_window_fingerprint": precision.source_window_fingerprint,
            "window_statistics_fingerprint": (precision.window_statistics_fingerprint),
        }
        result.append(
            CanonicalG3PlannedLook(
                family_id,
                index,
                look.windows,
                decision,
                MappingProxyType(observed),
            )
        )
    if kernel.precision_passed != completed[-1].passed:
        raise ModelError("kernel precision result differs from its terminal look")
    return tuple(result)


def _tier_b_looks(
    family_id: str,
    schedule: tuple[int, ...],
    source: BridgeDriverTaskView,
) -> tuple[CanonicalG3PlannedLook, ...]:
    expected_dgp: Literal["model", "nonparametric"] = (
        "nonparametric" if "nonparametric" in source.task_id else "model"
    )
    local_looks = source.planned_looks
    if not local_looks or len(local_looks) > len(schedule):
        raise ModelError("Tier-B planned-look sequence is empty or too long")
    result: list[CanonicalG3PlannedLook] = []
    for index, look in enumerate(local_looks, start=1):
        if (
            look.dgp != expected_dgp
            or look.look_index != index
            or look.sample_count != schedule[index - 1]
        ):
            raise ModelError("Tier-B planned-look schedule is inconsistent")
        is_last = index == len(local_looks)
        if not is_last and look.decision != "continue":
            raise ModelError("Tier-B local receipt is terminal before its last look")
        if is_last and look.decision == "continue":
            raise ModelError("Tier-B local look sequence is nonterminal")
        decision: LookDecision = (
            "continue"
            if look.decision == "continue"
            else "pass"
            if look.decision == "pass_above_tail"
            else "fail"
        )
        interval = clopper_pearson_interval(
            look.exceedances,
            look.sample_count,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        observed = {
            "bridge_binding_fingerprint": source.shared_binding_fingerprint,
            "bridge_task_result_fingerprint": source.task_result_fingerprint,
            "dgp": expected_dgp,
            "estimate": interval.estimate,
            "exceedances": look.exceedances,
            "interval_lower": interval.lower,
            "interval_upper": interval.upper,
            "local_bridge_look_receipt_fingerprint": (look.stage_receipt_fingerprint),
            "look_evidence_fingerprint": look.evidence_fingerprint,
            "look_rng_fingerprint": look.rng_execution_fingerprint,
            "tail_probability": 0.025,
        }
        result.append(
            CanonicalG3PlannedLook(
                family_id,
                index,
                look.sample_count,
                decision,
                MappingProxyType(observed),
            )
        )
    return tuple(result)


def _commit_canonical_looks(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    task_id: str,
    proposed: tuple[CanonicalG3PlannedLook, ...],
) -> tuple[PlannedLookReceipt, ...]:
    current = store.verify_planned_look_receipts(lease.run_id, require_terminal=False)
    existing = _receipts_for_task(task_id, current)
    _compare_receipts_to_looks(existing, proposed, allow_prefix=True)
    grouped_existing = {receipt.family_id: receipt for receipt in existing}
    committed: list[PlannedLookReceipt] = []
    for look in proposed:
        prior = grouped_existing.get(look.family_id)
        # ``grouped_existing`` retains only the last existing receipt per
        # family, while the explicit index distinguishes a durable prefix.
        if prior is not None and look.look_index <= prior.look_index:
            matching = next(
                item
                for item in existing
                if item.family_id == look.family_id
                and item.look_index == look.look_index
            )
            committed.append(matching)
            continue
        receipt = store.write_planned_look_receipt(
            lease,
            family_id=look.family_id,
            look_index=look.look_index,
            sample_count=look.sample_count,
            decision=look.decision,
            observed=look.observed,
        )
        committed.append(receipt)
        grouped_existing[look.family_id] = receipt
    return tuple(committed)


def _compare_receipts_to_looks(
    receipts: Sequence[PlannedLookReceipt],
    proposed: Sequence[CanonicalG3PlannedLook],
    *,
    allow_prefix: bool = False,
) -> None:
    if (allow_prefix and len(receipts) > len(proposed)) or (
        not allow_prefix and len(receipts) != len(proposed)
    ):
        raise IntegrityError("canonical planned-look sequence length changed")
    for receipt, look in zip(receipts, proposed, strict=False):
        if (
            receipt.family_id != look.family_id
            or receipt.look_index != look.look_index
            or receipt.sample_count != look.sample_count
            or receipt.decision != look.decision
            or dict(receipt.observed) != dict(look.observed)
        ):
            raise IntegrityError("canonical planned-look evidence changed on resume")


def _receipts_for_task(
    task_id: str,
    all_receipts: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> tuple[PlannedLookReceipt, ...]:
    result: list[PlannedLookReceipt] = []
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if spec.owner_task_id == task_id:
            result.extend(all_receipts.get(spec.family_id, ()))
    return tuple(result)


def _look_fingerprint_mapping(
    receipts: Sequence[PlannedLookReceipt],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for receipt in receipts:
        result.setdefault(receipt.family_id, []).append(receipt.fingerprint)
    return result


def _publish_checkpoint(
    *,
    checkpoint_store: CalibrationCheckpointStore,
    task_id: str,
    binding: CalibrationCheckpointBinding,
    dependencies: tuple[CalibrationCheckpointReference, ...],
    selected_n_states: int | None,
    decision_document: Mapping[str, Any],
    decision_fingerprint: str,
    planned_look_fingerprints: Mapping[str, Sequence[str]],
) -> CalibrationCheckpointReference:
    look_document = _json_value(dict(planned_look_fingerprints), "planned looks")
    if not isinstance(look_document, dict):  # pragma: no cover - local invariant
        raise AssertionError("planned-look JSON normalization changed type")
    state = {
        "checkpoint_state_schema_version": _CHECKPOINT_STATE_SCHEMA_VERSION,
        "decision": dict(decision_document),
        "decision_fingerprint": decision_fingerprint,
        "planned_look_fingerprints": look_document,
    }
    decision_digest = np.frombuffer(
        bytes.fromhex(decision_fingerprint), dtype=np.uint8
    ).copy()
    looks_digest = np.frombuffer(
        bytes.fromhex(_fingerprint(look_document)), dtype=np.uint8
    ).copy()
    return checkpoint_store.create(
        task_id=task_id,
        binding=binding,
        dependencies=dependencies,
        selected_n_states=selected_n_states,
        arrays={
            "decision_sha256": decision_digest,
            "planned_looks_sha256": looks_digest,
        },
        state=state,
    )


def _verify_checkpoint_state(
    checkpoint: CalibrationCheckpointReference,
    *,
    decision_document: Mapping[str, Any],
    decision_fingerprint: str,
    planned_look_fingerprints: Mapping[str, Sequence[str]],
) -> None:
    expected_state = {
        "checkpoint_state_schema_version": _CHECKPOINT_STATE_SCHEMA_VERSION,
        "decision": dict(decision_document),
        "decision_fingerprint": decision_fingerprint,
        "planned_look_fingerprints": _json_value(
            dict(planned_look_fingerprints), "planned looks"
        ),
    }
    actual_state = _json_value(dict(checkpoint.state), "stored checkpoint state")
    if actual_state != expected_state:
        raise IntegrityError("canonical checkpoint scientific state changed")
    expected_names = {"decision_sha256", "planned_looks_sha256"}
    if set(checkpoint.arrays) != expected_names:
        raise IntegrityError("canonical checkpoint digest arrays are invalid")
    decision_digest = np.frombuffer(bytes.fromhex(decision_fingerprint), dtype=np.uint8)
    look_digest = np.frombuffer(
        bytes.fromhex(_fingerprint(expected_state["planned_look_fingerprints"])),
        dtype=np.uint8,
    )
    if not np.array_equal(
        checkpoint.arrays["decision_sha256"], decision_digest
    ) or not np.array_equal(checkpoint.arrays["planned_looks_sha256"], look_digest):
        raise IntegrityError("canonical checkpoint digest arrays disagree")


def _decision_document(
    decision: DerivedG3GateDecision,
    source: object,
) -> dict[str, Any]:
    document = {
        "evidence": dict(decision.evidence),
        "gate_id": decision.gate_id,
        "passed": decision.passed,
        "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
        "summary": dict(decision.summary),
    }
    normalized = _json_value(document, "typed G3 decision")
    if not isinstance(normalized, dict):  # pragma: no cover - local invariant
        raise AssertionError("decision JSON normalization changed type")
    return normalized


def _decision_payload(
    *,
    checkpoint_fingerprint: str | None,
    decision_fingerprint: str,
    decision: DerivedG3GateDecision,
    selected_n_states: int | None,
    planned_look_fingerprints: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    summary = _json_value(dict(decision.summary), "G3 decision summary")
    looks = _json_value(dict(planned_look_fingerprints), "planned looks")
    return {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "coordinator_schema_version": _COORDINATOR_SCHEMA_VERSION,
        "decision_fingerprint": decision_fingerprint,
        "execution_mode": _EXECUTION_MODE,
        "outcome_kind": "typed_decision",
        "planned_look_fingerprints": looks,
        "selected_n_states": selected_n_states,
        "summary": summary,
    }


def _failure_payload(
    failure: G3TaskFailure,
    *,
    selected_n_states: int | None,
    planned_look_fingerprints: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "checkpoint_fingerprint": None,
        "coordinator_schema_version": _COORDINATOR_SCHEMA_VERSION,
        "decision_fingerprint": _failure_fingerprint(failure),
        "execution_mode": _EXECUTION_MODE,
        "outcome_kind": "scientific_failure",
        "planned_look_fingerprints": _json_value(
            dict(planned_look_fingerprints), "planned looks"
        ),
        "selected_n_states": selected_n_states,
        "summary": {
            "failure_type": failure.failure_type,
            "message": failure.message,
        },
    }


def _blocked_payload(
    *,
    blocked_by: tuple[str, ...],
    selected_n_states: int | None,
) -> dict[str, Any]:
    return {
        "checkpoint_fingerprint": None,
        "coordinator_schema_version": _COORDINATOR_SCHEMA_VERSION,
        "decision_fingerprint": None,
        "execution_mode": _EXECUTION_MODE,
        "outcome_kind": "blocked",
        "planned_look_fingerprints": {},
        "selected_n_states": selected_n_states,
        "summary": {
            "blocked_by": list(blocked_by),
            "reason": "upstream_dependency_not_passed",
        },
    }


def _next_selected_n_states(
    *,
    task_id: str,
    decision: DerivedG3GateDecision,
    frozen: int | None,
) -> int | None:
    reported = decision.summary.get("selected_k")
    if task_id == _SELECTION_TASK:
        if not decision.passed:
            return None
        return _state_count(reported)
    if frozen is None:
        if reported is not None and G3_GATE_ORDER.index(task_id) < G3_GATE_ORDER.index(
            _SELECTION_TASK
        ):
            return None
        if reported is not None:
            raise ModelError("selected K appeared without a passed design selection")
        return None
    if reported is not None and _state_count(reported) != frozen:
        raise ModelError("canonical task attempted to change or promote selected K")
    return frozen


def _state_count(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in _ALLOWED_STATE_COUNTS
    ):
        raise ModelError("canonical selected K must be one of 2,3,4,5")
    return value


def _derived_sources(
    handlers: CanonicalG3HandlerBundle,
) -> tuple[tuple[DerivedG3GateDecision, ...], tuple[G3TaskFailure, ...]]:
    decisions: list[DerivedG3GateDecision] = []
    failures: list[G3TaskFailure] = []
    for task_id, source in handlers.as_mapping().items():
        if isinstance(source, G3TaskFailure):
            failures.append(source)
        else:
            decisions.append(derive_g3_gate_decision(task_id, source))
    return tuple(decisions), tuple(failures)


def _verify_bundle_matches_results(
    bundle: G3EvidenceBundle,
    results: Mapping[str, G3TaskResult],
) -> None:
    if tuple(results) != G3_GATE_ORDER:
        raise IntegrityError("canonical G3 result graph is incomplete")
    if tuple(record.gate_id for record in bundle.records) != G3_GATE_ORDER:
        raise IntegrityError("typed G3 bundle order differs from task registry")
    for record in bundle.records:
        result = results[record.gate_id]
        expected_pass = result.status == "passed"
        if record.passed != expected_pass:
            raise IntegrityError("typed G3 bundle decision differs from task result")
    all_passed = all(result.status == "passed" for result in results.values())
    if bundle.passed != all_passed:
        raise IntegrityError("G3 passed decision differs from exact task graph")
    if bundle.generation_authorized is not False:
        raise IntegrityError("G3 task graph must remain evidence-only")


def _validate_input_source_binding(
    source: InputIntegrityTaskView | G3TaskFailure,
    *,
    run_id: str,
    persisted_preflight_fingerprint: str,
    live_preflight: CanonicalG3Preflight,
) -> None:
    if isinstance(source, G3TaskFailure):
        return
    if not is_input_integrity_task_view(source):
        raise ModelError("canonical input source lacks sealed task authority")
    evidence = source.source
    binding = evidence.binding
    if (
        source.run_id != run_id
        or evidence.launch_preflight.fingerprint != persisted_preflight_fingerprint
        or evidence.live_preflight != live_preflight
        or binding.calibration_run_fingerprint != run_id
        or binding.preflight_fingerprint != persisted_preflight_fingerprint
        or binding.config_fingerprint != live_preflight.config_fingerprint
        or binding.processed_data_fingerprint
        != live_preflight.processed_data_fingerprint
        or binding.calibration_input_fingerprint
        != live_preflight.calibration_input_fingerprint
        or evidence.launch_preflight.config_fingerprint
        != live_preflight.config_fingerprint
        or evidence.launch_preflight.processed_data_fingerprint
        != live_preflight.processed_data_fingerprint
        or evidence.launch_preflight.calibration_input_fingerprint
        != live_preflight.calibration_input_fingerprint
        or evidence.launch_preflight.contract_fingerprint
        != live_preflight.contract_fingerprint
        or evidence.launch_preflight.source_code_fingerprint
        != live_preflight.source_code_fingerprint
        or evidence.launch_preflight.dependency_lock_fingerprint
        != live_preflight.dependency_lock_fingerprint
        or evidence.launch_preflight.toolchain_fingerprint
        != live_preflight.toolchain_fingerprint
        or evidence.launch_preflight.ppw_vendored_fingerprint
        != live_preflight.ppw_vendored_fingerprint
        or evidence.launch_preflight.ppw_upstream_fingerprint
        != live_preflight.ppw_upstream_fingerprint
        or evidence.launch_preflight.task_registry_fingerprint
        != live_preflight.task_registry_fingerprint
        or evidence.launch_preflight.planned_look_registry_fingerprint
        != live_preflight.planned_look_registry_fingerprint
        or evidence.launch_preflight.workers != live_preflight.workers
    ):
        raise IntegrityError("input-integrity source is not bound to this launch")


def _validated_failure(task_id: str, failure: G3TaskFailure) -> None:
    if failure.task_id != task_id:
        raise ModelError("canonical G3 failure is bound to the wrong task")
    if (
        not failure.failure_type
        or len(failure.failure_type) > 128
        or not failure.message
        or len(failure.message) > 512
    ):
        raise ModelError("canonical G3 failure text is empty or unbounded")


def _failure_fingerprint(failure: G3TaskFailure) -> str:
    return _fingerprint(
        {
            "failure_type": failure.failure_type,
            "message": failure.message,
            "task_id": failure.task_id,
        }
    )


def _fingerprint(value: object) -> str:
    normalized = _json_value(value, "fingerprint input")
    content = (
        json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ModelError(f"{path} contains a non-finite float")
        return 0.0 if numeric == 0.0 else numeric
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ModelError(f"{path} contains an invalid mapping key")
        return {key: _json_value(value[key], f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_json_value(item, f"{path}[]") for item in value]
    raise ModelError(f"{path} contains a non-JSON scientific value")
