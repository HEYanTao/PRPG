"""Durable, binding-aware publication for one exact next canonical G3 task.

Scientific computation is deliberately outside this module.  The public
publisher accepts only an already sealed, task-scoped source and an actual
``ScientificTaskBinding``.  It then re-derives the decision, verifies the
current run/checkpoint/look prefix, and durably binds a passing result to the
serialized task binding.  A serialized binding never restores authority by
itself: resume and final-evidence verification must call
``reverify_published_scientific_task`` against a fresh live preflight.

Checkpoints here intentionally contain identity digests, not scientific
arrays.  A topological runner must deterministically reconstruct each source
from immutable inputs, registered RNG streams, and dependency artifacts, then
call ``verify_recomputed_scientific_task_source`` (the publisher does this
unconditionally).  The reconstructed guarded view must reproduce every exact
registered component, materialization, and RNG fingerprint.  Consequently
this module is a durable publication seam, not a complete resume runner or a
license to recompute from mutable ambient state.

Several historical G3 evidence types are not yet task-scoped.  They are
intentionally rejected rather than wrapped here.  A publishable source must
expose exact ``run_id``, ``task_id``, and ``binding_fingerprint`` coordinates,
and its task evaluator must accept its existing code-owned seal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np

import prpg.model.adequacy_gate as adequacy_gate_module
import prpg.model.g3_boundary_views as boundary_view_module
import prpg.model.g3_calibration_views as calibration_view_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.adequacy_execution import (
    RegisteredLatentCell,
    registered_latent_cell_identity,
    registered_latent_cell_source_fit,
)
from prpg.model.adequacy_gate import (
    AdequacyCellTaskId,
    AdequacyCellTaskView,
    is_adequacy_cell_task_view,
    is_four_cell_adequacy_holm_view,
)
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    registered_block_core_identity,
    registered_block_fragmentation_identity,
)
from prpg.model.bridge import TIER_B_INTERVAL_ERROR
from prpg.model.bridge_driver import (
    StagedBridgeTaskEvidence,
    staged_bridge_planned_look_identity,
    staged_bridge_task_evidence_identity,
    staged_bridge_task_planned_looks,
)
from prpg.model.calibration import (
    MacroStabilityEvidence,
    SealedHoldoutEvidence,
    evaluate_sealed_holdout,
    macro_stability_input_fingerprint,
    macro_stability_rng_fingerprint,
    macro_stability_rng_namespace,
    registered_macro_stability_execution_identity,
)
from prpg.model.calibration_identity import (
    calibration_slice_fingerprint,
    macro_stability_evidence_fingerprint,
    sealed_holdout_evidence_fingerprint,
)
from prpg.model.candidate_gate import (
    CandidateGateExecutionIdentity,
    CandidateGateResult,
    CandidateViabilityExecutionIdentity,
    CandidateViabilityResult,
    has_registered_final_fixed_point_candidate_authority,
    verified_registered_candidate_gate_execution_identity,
    verified_registered_candidate_viability_execution_identity,
)
from prpg.model.dependence_gate_execution import (
    DEPENDENCE_CELL_ORDER,
    DependenceTaskView,
    RegisteredArtifactDependenceExecution,
    is_registered_dependence_task_view,
    is_registered_four_cell_dependence_holm_view,
    is_registered_staged_dependence_execution,
)
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_assembly import DerivedG3GateDecision, derive_g3_gate_decision
from prpg.model.g3_boundary_views import (
    ArtifactIntegrityDecisionEvidence,
    BlockCalibrationGateEvidence,
    BridgeDriverTaskView,
    BridgeTaskId,
    InputIntegrityDecisionEvidence,
    is_artifact_integrity_task_view,
    is_block_calibration_task_view,
    is_bridge_driver_task_view,
    is_input_integrity_task_view,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    CandidateViabilityTaskView,
    MacroTaskId,
    ProductionFitSameKView,
    is_candidate_selection_task_view,
    is_macro_stability_task_view,
    is_production_fit_same_k_view,
    is_registered_candidate_selection_task_view,
    is_registered_candidate_viability_task_view,
    is_sealed_holdout_veto_view,
    registered_production_fit_execution_from_view,
)
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointBinding,
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_preflight import (
    CanonicalG3Preflight,
    verify_compatible_canonical_g3_preflights,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    task_spec,
)
from prpg.model.g3_science_handlers import (
    CanonicalG3PlannedLook,
    CanonicalTaskSource,
    canonical_planned_looks,
)
from prpg.model.g3_task_binding import (
    SCIENTIFIC_TASK_SLOT_REGISTRY,
    SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT,
    TASK_SOURCE_MATERIALIZATION_SLOT,
    ScientificTaskBinding,
    load_scientific_task_binding,
    verify_scientific_task_binding,
)
from prpg.model.inference import clopper_pearson_interval, holm_correction
from prpg.model.input import CalibrationSlice
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.production_fit_execution import (
    RegisteredProductionFitSameK,
    registered_production_fit_same_k_identity,
    registered_production_fit_same_k_sources,
)
from prpg.model.refit_adequacy_execution import (
    RegisteredRefittedAdequacyBatch,
    is_registered_refitted_adequacy_batch,
    registered_refitted_adequacy_batch_identity,
    registered_refitted_adequacy_sources,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    LaunchLease,
    PlannedLookReceipt,
)
from prpg.model.scientific_artifact import ScientificArtifactBinding
from prpg.model.selection import OWNER_FIXED_N_STATES
from prpg.simulation.rng import CalibrationArtifact

SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION: Final = 1
SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION: Final = 2
TASK_SOURCE_MATERIALIZATION_SCHEMA_ID: Final = "prpg-g3-task-source-materialization-v2"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SELECTION_TASK: Final = "design_predictive_selection"
_SELECTION_POSITION: Final = G3_GATE_ORDER.index(_SELECTION_TASK)
_ALLOWED_STATE_COUNTS: Final = frozenset({2, 3, 4, 5})
_CHECKPOINT_STATE_KEYS: Final = frozenset(
    {
        "binding_fingerprint",
        "checkpoint_state_schema_version",
        "decision",
        "decision_fingerprint",
        "planned_look_fingerprints",
        "publication_schema_version",
        "scientific_task_binding",
        "serialized_binding_sha256",
    }
)
_TASK_RESULT_PAYLOAD_KEYS: Final = frozenset(
    {
        "binding_fingerprint",
        "checkpoint_fingerprint",
        "coordinator_schema_version",
        "decision",
        "decision_fingerprint",
        "execution_mode",
        "outcome_kind",
        "planned_look_fingerprints",
        "scientific_task_binding",
        "selected_n_states",
        "serialized_binding_sha256",
        "summary",
    }
)
_CHECKPOINT_ARRAY_KEYS: Final = frozenset(
    {
        "decision_sha256",
        "planned_looks_sha256",
        "scientific_task_binding_sha256",
    }
)
_DECISION_KEYS: Final = frozenset(
    {"evidence", "gate_id", "passed", "source_type", "summary"}
)
_COMPONENT_SLOTS: Final = frozenset(
    slot
    for spec in SCIENTIFIC_TASK_SLOT_REGISTRY
    for slot in spec.scientific_component_slots
)

_BRIDGE_SLOT_LAYOUT: Final[Mapping[str, tuple[str, str, str | None]]] = (
    MappingProxyType(
        {
            "bridge_resource_preflight": (
                "bridge_resource_component",
                "bridge_benchmark_materialization",
                "bridge_benchmark_pair_streams",
            ),
            "bridge_tier_a_model_dgp": (
                "bridge_tier_a_component",
                "bridge_tier_a_model_materialization",
                "bridge_tier_a_model_streams",
            ),
            "bridge_tier_a_nonparametric_dgp": (
                "bridge_tier_a_component",
                "bridge_tier_a_nonparametric_materialization",
                "bridge_tier_a_nonparametric_streams",
            ),
            "bridge_tier_b_model_dgp": (
                "bridge_tier_b_component",
                "bridge_tier_b_model_materialization",
                "bridge_tier_b_model_streams",
            ),
            "bridge_tier_b_nonparametric_dgp": (
                "bridge_tier_b_component",
                "bridge_tier_b_nonparametric_materialization",
                "bridge_tier_b_nonparametric_streams",
            ),
            "actual_artifact_bridge": (
                "actual_artifact_bridge_component",
                "actual_artifact_history_materialization",
                None,
            ),
        }
    )
)

# This executable gap register stays present even when no source family is
# report-only.  Any future gap must be explicit here and fail closed below.
SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_REASONS: Final[Mapping[str, str]] = (
    MappingProxyType({})
)
SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_TASKS: Final = tuple(
    SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_REASONS
)


@dataclass(frozen=True, slots=True)
class ScientificTaskSourceSlots:
    """Task slots re-derived only from a verified code-owned source view."""

    task_id: str
    scientific_component_fingerprints: tuple[tuple[str, str], ...]
    materialization_fingerprints: tuple[tuple[str, str], ...]
    rng_fingerprints: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ScientificTaskPublication:
    """One freshly published passing result and its exact scientific identity."""

    binding: ScientificTaskBinding
    decision: DerivedG3GateDecision
    checkpoint: CalibrationCheckpointReference | None
    task_result: G3TaskResult
    serialized_binding_sha256: str

    @property
    def generation_authorized(self) -> bool:
        """A task publication is evidence, never final generation authority."""

        return False


@dataclass(frozen=True, slots=True)
class ReverifiedScientificTaskPublication:
    """A durable publication reloaded against current stores and preflight."""

    binding: ScientificTaskBinding
    checkpoint: CalibrationCheckpointReference | None
    task_result: G3TaskResult
    serialized_binding_sha256: str

    @property
    def generation_authorized(self) -> bool:
        return False


def scientific_component_slot_fingerprint(slot: str) -> str:
    """Return the code-owned semantic identity of one registered component."""

    if slot not in _COMPONENT_SLOTS:
        raise ModelError("scientific component slot is not registered")
    return _fingerprint(
        {
            "schema_id": "prpg-g3-scientific-component-slot-v1",
            "contract_version": G3_CONTRACT_VERSION,
            "slot": slot,
            "slot_registry_fingerprint": (SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT),
        }
    )


def scientific_task_source_slots(source: object) -> ScientificTaskSourceSlots:
    """Derive v2 slots only after the source's object-specific guard passes.

    The universal ``task_source_materialization`` is an explicit projection of
    the task's scientific outcome and executor evidence.  That projection
    never reuses the current view's run, binding, view, or content fingerprint,
    so an executor can reproduce it before the task binding is minted.  The
    remaining slots preserve narrower materialization and RNG audit seams.
    """

    task_id = getattr(source, "task_id", None)
    components: dict[str, str]
    specialized_materializations: dict[str, str]
    rng: dict[str, str]
    source_identity: Mapping[str, object]
    if is_input_integrity_task_view(source):
        components = _component_fingerprints("input_validation_component")
        specialized_materializations = {
            "calibration_input_materialization": source.calibration_input_fingerprint
        }
        rng = {}
        source_identity = _input_task_source_identity(source.source)
    elif is_registered_candidate_viability_task_view(source):
        components = _component_fingerprints("candidate_viability_component")
        specialized_materializations = {
            "design_candidate_fit_set": source.fit_set_fingerprint,
            "design_candidate_suite": _candidate_suite_view_field(
                source,
                "suite_fingerprint",
            ),
            "design_candidate_common_materializations": (
                _candidate_suite_view_field(
                    source,
                    "suite_combined_materialization_fingerprint",
                )
            ),
            "design_candidate_macro_bootstrap_materialization": (
                _candidate_suite_view_field(
                    source,
                    "suite_macro_materialization_fingerprint",
                )
            ),
            "design_candidate_monthly_block_uniform_materialization": (
                _candidate_suite_view_field(
                    source,
                    "suite_monthly_block_materialization_fingerprint",
                )
            ),
            "design_candidate_daily_block_uniform_materialization": (
                _candidate_suite_view_field(
                    source,
                    "suite_daily_block_materialization_fingerprint",
                )
            ),
            "design_fixed_point_viability_projection": (
                _candidate_suite_view_field(
                    source,
                    "fixed_point_viability_projection_fingerprint",
                )
            ),
        }
        rng = {
            "design_candidate_fit_restarts": source.rng_execution_fingerprint,
            "design_candidate_main_and_rolling_fit_restarts": (
                _candidate_suite_view_field(
                    source,
                    "suite_combined_fit_rng_execution_fingerprint",
                )
            ),
            "design_candidate_all_registered_streams": (
                _candidate_suite_view_field(
                    source,
                    "suite_combined_rng_execution_fingerprint",
                )
            ),
        }
        source_identity = _candidate_viability_source_identity(source)
    elif is_registered_candidate_selection_task_view(source):
        components = _component_fingerprints(
            "predictive_selection_component",
            "candidate_viability_parent_component",
        )
        specialized_materializations = {
            "candidate_gate_parent_materialization": _composite_slot_fingerprint(
                "candidate_gate_parent",
                {
                    "fit_set": source.fit_set_fingerprint,
                    "source_content": source.source_content_fingerprint,
                    "viability_result": source.viability_result_fingerprint,
                },
            ),
            "rolling_score_bootstrap_materialization": (
                source.bootstrap_materialization_fingerprint
            ),
            "design_fixed_point_history_materialization": (
                _candidate_suite_view_field(
                    source,
                    "fixed_point_iteration_history_fingerprint",
                )
            ),
            "design_fixed_point_final_selection_materialization": (
                _candidate_suite_view_field(
                    source,
                    "fixed_point_final_selection_fingerprint",
                )
            ),
            "design_fixed_point_final_block_parent_materialization": (
                _composite_slot_fingerprint(
                    "design_fixed_point_final_block_parents",
                    {
                        "monthly": (
                            _optional_slot_fingerprint(
                                source.fixed_point_final_monthly_block_core_fingerprint,
                                "design_fixed_point_final_monthly_block",
                            )
                        ),
                        "daily": (
                            _optional_slot_fingerprint(
                                source.fixed_point_final_daily_block_core_fingerprint,
                                "design_fixed_point_final_daily_block",
                            )
                        ),
                    },
                )
            ),
        }
        rng = {
            "rolling_score_bootstrap_streams": (
                source.bootstrap_rng_execution_fingerprint
            )
        }
        source_identity = _candidate_selection_source_identity(source)
    elif is_sealed_holdout_veto_view(source):
        components = _component_fingerprints(
            "sealed_holdout_veto_component",
            "predictive_selection_parent_component",
        )
        specialized_materializations = {
            "selected_design_model_parent_materialization": (
                source.selected_model_fingerprint
            ),
            "sealed_holdout_score_materialization": source.evidence_fingerprint,
        }
        rng = {}
        source_identity = _holdout_source_identity(source)
    elif is_macro_stability_task_view(source):
        role = source.role
        components = _component_fingerprints("macro_stability_component")
        specialized_materializations = {
            f"{role}_model_parent_materialization": source.parent_model_fingerprint,
            f"{role}_macro_materialization": source.macro_input_fingerprint,
        }
        rng = {f"{role}_macro_bootstrap_streams": source.rng_execution_fingerprint}
        source_identity = _macro_source_identity(source)
    elif is_production_fit_same_k_view(source):
        components = _component_fingerprints("production_fit_same_k_component")
        specialized_materializations = {
            "selected_design_model_parent_materialization": (
                source.design_model_fingerprint
            ),
            "full_calibration_input_materialization": (
                source.production_feature_fingerprint
            ),
        }
        rng = {"production_fit_restarts": source.rng_execution_fingerprint}
        source_identity = _production_fit_source_identity(source)
    elif is_adequacy_cell_task_view(source):
        role = source.role
        specialized_materializations = {
            f"{role}_model_parent_materialization": source.parent_model_fingerprint
        }
        source_identity = _adequacy_cell_source_identity(source)
        if source.task_id in {
            "design_latent_correctness",
            "production_latent_correctness",
        }:
            simulation_rng = source.simulation_rng_execution_fingerprint
            bootstrap_rng = source.bootstrap_rng_execution_fingerprint
            if simulation_rng is None or bootstrap_rng is None:  # pragma: no cover
                raise AssertionError("guarded latent view lacks split RNG identities")
            components = _component_fingerprints("latent_correctness_component")
            rng = {
                f"{role}_latent_simulation_streams": simulation_rng,
                f"{role}_latent_bootstrap_streams": bootstrap_rng,
            }
        elif source.task_id in {
            "design_refitted_adequacy",
            "production_refitted_adequacy",
        }:
            components = _component_fingerprints("refitted_adequacy_component")
            rng = {f"{role}_refit_adequacy_streams": source.rng_execution_fingerprint}
        else:  # pragma: no cover - the guard's task registry is closed.
            raise AssertionError("guarded adequacy view has an unknown task ID")
    elif is_four_cell_adequacy_holm_view(source):
        components = _component_fingerprints(
            "adequacy_holm_component",
            "design_latent_parent_component",
            "design_refitted_parent_component",
            "production_latent_parent_component",
            "production_refitted_parent_component",
        )
        parent_materializations: dict[str, str] = {
            item.task_id: _task_source_materialization_fingerprint(
                item.task_id,
                _adequacy_cell_source_identity(item),
            )
            for item in source.source_views
        }
        specialized_materializations = {
            "four_cell_adequacy_parent_materialization": (
                _composite_slot_fingerprint(
                    "four_cell_adequacy_parent",
                    {
                        task: parent_materializations[task]
                        for task in source.source_task_order
                    },
                )
            )
        }
        rng = {}
        source_identity = _adequacy_holm_source_identity(
            source,
            parent_materializations=parent_materializations,
        )
    elif is_block_calibration_task_view(source):
        role = source.role
        components = _component_fingerprints("block_calibration_component")
        specialized_materializations = {
            f"{role}_model_parent_materialization": source.parent_model_fingerprint,
            f"{role}_block_uniform_materialization": (
                source.block_uniform_materialization_fingerprint
            ),
            f"{role}_kernel_calibration_materialization": (
                _composite_slot_fingerprint(
                    f"{role}_kernel_calibration_materialization",
                    {
                        "monthly": source.monthly_kernel_materialization_fingerprint,
                        "daily": source.daily_kernel_materialization_fingerprint,
                    },
                )
            ),
            f"{role}_transition_bootstrap_materialization": (
                source.transition_materialization_fingerprint
            ),
        }
        if role == "design":
            specialized_materializations[
                "design_fixed_point_final_block_parent_materialization"
            ] = _required_block_identity(
                source.design_fixed_point_final_block_parent_fingerprint,
                "design fixed-point final block parent",
            )
        rng = {
            f"{role}_block_calibration_streams": (
                source.block_calibration_rng_fingerprint
            ),
            f"{role}_kernel_calibration_streams": (
                source.kernel_calibration_rng_fingerprint
            ),
            f"{role}_transition_bootstrap_streams": source.transition_rng_fingerprint,
        }
        source_identity = _block_source_identity(source)
    elif is_registered_dependence_task_view(source):
        role = source.role
        if source.task_kind == "fragmentation":
            components = _component_fingerprints(
                "fragmentation_component",
                f"{role}_block_calibration_parent_component",
            )
            specialized_materializations = {
                f"{role}_block_calibration_parent_materialization": (
                    _dependence_block_parent_materialization(source)
                )
            }
            rng = {}
        else:
            components = _component_fingerprints("dependence_component")
            specialized_materializations = {
                f"{role}_block_calibration_parent_materialization": (
                    _dependence_block_parent_materialization(source)
                ),
                f"{role}_dependence_uniform_materialization": (
                    _dependence_uniform_materialization(source)
                ),
            }
            rng = {f"{role}_dependence_streams": _dependence_rng_execution(source)}
        source_identity = _dependence_task_source_identity(source)
    elif is_registered_four_cell_dependence_holm_view(source):
        components = _component_fingerprints(
            "dependence_holm_component",
            "design_dependence_parent_component",
            "production_dependence_parent_component",
        )
        parent_materializations = _dependence_parent_source_materializations(
            source.source_views
        )
        specialized_materializations = {
            "four_cell_dependence_parent_materialization": (
                _composite_slot_fingerprint(
                    "four_cell_dependence_parent",
                    parent_materializations,
                )
            )
        }
        rng = {}
        source_identity = _dependence_holm_source_identity(
            source,
            parent_materializations=parent_materializations,
        )
    elif is_bridge_driver_task_view(source):
        if not isinstance(source.source, StagedBridgeTaskEvidence):
            raise AssertionError("guarded bridge view lacks staged task evidence")
        return _bridge_source_slots_from_evidence(
            source.source,
            task_id=source.task_id,
        )
    elif is_artifact_integrity_task_view(source):
        components = _component_fingerprints(
            "artifact_integrity_component",
            "actual_artifact_bridge_parent_component",
        )
        specialized_materializations = {
            "final_scientific_artifact_parent_materialization": (
                source.final_artifact_parent_fingerprint
            )
        }
        rng = {}
        source_identity = _artifact_source_identity(source)
    else:
        raise ModelError(
            "canonical task source lacks code-owned binding-slot derivation",
            details={
                "task_id": task_id,
                "reason": SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_REASONS.get(
                    cast(str, task_id),
                    "the source failed every supported object-specific guard",
                ),
                "unsupported_tasks": list(
                    SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_TASKS
                ),
            },
        )
    if not isinstance(task_id, str):  # pragma: no cover - guards imply this.
        raise AssertionError("verified task source lacks a task ID")
    return _source_slots_from_identity(
        task_id,
        components=components,
        specialized_materializations=specialized_materializations,
        rng=rng,
        source_identity=source_identity,
    )


def input_integrity_executor_source_materialization_fingerprint(
    source: InputIntegrityDecisionEvidence,
    *,
    run_id: str,
) -> str:
    """Return the universal slot from the exact prebinding input mapping."""

    return dict(
        input_integrity_executor_source_slots(
            source,
            run_id=run_id,
        ).materialization_fingerprints
    )[TASK_SOURCE_MATERIALIZATION_SLOT]


def input_integrity_executor_source_slots(
    source: InputIntegrityDecisionEvidence,
    *,
    run_id: str,
) -> ScientificTaskSourceSlots:
    """Mint the first task's complete slots before its view/binding exists.

    It accepts typed executor evidence, not an arbitrary digest or identity
    mapping, and repeats the raw checks used by the final view builder.  The
    output deliberately omits the calibration run coordinate even though that
    coordinate is checked here.  Publication independently re-derives every
    value from the final guarded view.
    """

    _verify_input_integrity_executor_source(source, run_id=run_id)
    return _source_slots_from_identity(
        "input_integrity",
        components=_component_fingerprints("input_validation_component"),
        specialized_materializations={
            "calibration_input_materialization": (
                source.binding.calibration_input_fingerprint
            )
        },
        rng={},
        source_identity=_input_task_source_identity(source),
    )


def _verify_input_integrity_executor_source(
    source: InputIntegrityDecisionEvidence,
    *,
    run_id: str,
) -> None:
    if not isinstance(source, InputIntegrityDecisionEvidence):
        raise ModelError("input-integrity source has the wrong type")
    if not isinstance(run_id, str) or _SHA256.fullmatch(run_id) is None:
        raise ModelError("input-integrity run ID is not a SHA-256 fingerprint")
    verify_compatible_canonical_g3_preflights(
        source.launch_preflight,
        source.live_preflight,
    )
    if (
        not isinstance(source.binding, ScientificArtifactBinding)
        or source.binding.calibration_run_fingerprint != run_id
        or source.binding.preflight_fingerprint != source.launch_preflight.fingerprint
    ):
        raise ModelError("input-integrity run/preflight binding is inconsistent")


def candidate_viability_executor_source_slots(
    result: CandidateViabilityResult,
    *,
    master_seed: int,
    scientific_version: int,
) -> ScientificTaskSourceSlots:
    """Derive candidate-viability slots before its task binding/view."""

    if not has_registered_final_fixed_point_candidate_authority(result):
        raise ModelError(
            "canonical candidate viability requires final fixed-point authority"
        )
    identity = verified_registered_candidate_viability_execution_identity(
        result,
        require_strict_canonical_geometry=True,
    )
    if identity.role != "design":
        raise ModelError("G3 candidate viability prebinding requires design evidence")
    if (
        identity.suite_master_seed != master_seed
        or identity.suite_scientific_version != scientific_version
    ):
        raise ModelError("candidate viability prebinding seed/version differ")
    rows = calibration_view_module._viability_rows(result)
    admissible = tuple(row.n_states for row in rows if row.viability.admissible)
    restart_fingerprints, rng_fingerprint = (
        calibration_view_module._candidate_fit_rng_identity(
            result,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )
    )
    suite_fingerprint = _candidate_identity_field(identity, "suite_fingerprint")
    suite_materializations = _candidate_identity_field(
        identity,
        "suite_combined_materialization_fingerprint",
    )
    suite_fit_rng = _candidate_identity_field(
        identity,
        "suite_combined_fit_rng_execution_fingerprint",
    )
    suite_rng = _candidate_identity_field(
        identity,
        "suite_combined_rng_execution_fingerprint",
    )
    suite_macro = _candidate_identity_field(
        identity,
        "suite_macro_materialization_fingerprint",
    )
    suite_monthly = _candidate_identity_field(
        identity,
        "suite_monthly_block_materialization_fingerprint",
    )
    suite_daily = _candidate_identity_field(
        identity,
        "suite_daily_block_materialization_fingerprint",
    )
    fixed_point_viability_projection = _candidate_identity_field(
        identity,
        "fixed_point_viability_projection_fingerprint",
    )
    fixed_point_final_iteration = identity.fixed_point_final_iteration
    if (
        isinstance(fixed_point_final_iteration, bool)
        or not isinstance(fixed_point_final_iteration, int)
        or not 1 <= fixed_point_final_iteration <= 5
    ):
        raise ModelError("registered candidate identity lacks final iteration")
    viability_fingerprint = calibration_view_module._viability_result_fingerprint(rows)
    source_identity = _candidate_viability_identity_document(
        role="design",
        rows=rows,
        admissible_n_states=admissible,
        fit_set_fingerprint=identity.fit_set_fingerprint,
        source_content_fingerprint=identity.source_content_fingerprint,
        viability_result_fingerprint=identity.viability_result_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_execution_fingerprint=rng_fingerprint,
        restart_seed_fingerprints=restart_fingerprints,
        viability_fingerprint=viability_fingerprint,
        suite_fingerprint=suite_fingerprint,
        suite_combined_materialization_fingerprint=suite_materializations,
        suite_combined_fit_rng_execution_fingerprint=suite_fit_rng,
        suite_combined_rng_execution_fingerprint=suite_rng,
        suite_macro_materialization_fingerprint=suite_macro,
        suite_monthly_block_materialization_fingerprint=suite_monthly,
        suite_daily_block_materialization_fingerprint=suite_daily,
        fixed_point_final_iteration=fixed_point_final_iteration,
        fixed_point_viability_projection_fingerprint=(fixed_point_viability_projection),
    )
    return _source_slots_from_identity(
        "design_candidate_viability",
        components=_component_fingerprints("candidate_viability_component"),
        specialized_materializations={
            "design_candidate_fit_set": identity.fit_set_fingerprint,
            "design_candidate_suite": suite_fingerprint,
            "design_candidate_common_materializations": suite_materializations,
            "design_candidate_macro_bootstrap_materialization": suite_macro,
            "design_candidate_monthly_block_uniform_materialization": suite_monthly,
            "design_candidate_daily_block_uniform_materialization": suite_daily,
            "design_fixed_point_viability_projection": (
                fixed_point_viability_projection
            ),
        },
        rng={
            "design_candidate_fit_restarts": rng_fingerprint,
            "design_candidate_main_and_rolling_fit_restarts": suite_fit_rng,
            "design_candidate_all_registered_streams": suite_rng,
        },
        source_identity=source_identity,
    )


def candidate_selection_executor_source_slots(
    viability: CandidateViabilityTaskView,
    result: CandidateGateResult,
) -> ScientificTaskSourceSlots:
    """Derive predictive-selection slots from its parent and raw result."""

    if not is_registered_candidate_viability_task_view(
        viability
    ) or not has_registered_final_fixed_point_candidate_authority(result):
        raise ModelError(
            "candidate selection prebinding requires complete fixed-point authority"
        )
    identity = verified_registered_candidate_gate_execution_identity(
        result,
        require_strict_canonical_geometry=True,
    )
    if (
        identity.role != "design"
        or result.viability_result is not viability._source_result
        or identity.fit_set_fingerprint != viability.fit_set_fingerprint
        or identity.viability_result_fingerprint
        != viability.viability_result_fingerprint
        or identity.bootstrap_master_seed != viability.master_seed
        or identity.bootstrap_scientific_version != viability.scientific_version
    ):
        raise ModelError("candidate selection prebinding ancestry differs")
    selection = result.selection
    selection_fingerprint = calibration_view_module._selection_result_fingerprint(
        selection
    )
    fixed_point_history = _candidate_identity_field(
        identity,
        "fixed_point_iteration_history_fingerprint",
    )
    fixed_point_fingerprint = _candidate_identity_field(
        identity,
        "fixed_point_fingerprint",
    )
    fixed_point_final_selection = _candidate_identity_field(
        identity,
        "fixed_point_final_selection_fingerprint",
    )
    fixed_point_viability_projection = _candidate_identity_field(
        identity,
        "fixed_point_viability_projection_fingerprint",
    )
    if (
        identity.fixed_point_outcome is None
        or identity.fixed_point_converged_at_iteration is None
    ):
        raise ModelError("candidate selection lacks fixed-point outcome authority")
    fixed_point_block_parents = _composite_slot_fingerprint(
        "design_fixed_point_final_block_parents",
        {
            "monthly": _optional_slot_fingerprint(
                identity.fixed_point_final_monthly_block_core_fingerprint,
                "design_fixed_point_final_monthly_block",
            ),
            "daily": _optional_slot_fingerprint(
                identity.fixed_point_final_daily_block_core_fingerprint,
                "design_fixed_point_final_daily_block",
            ),
        },
    )
    source_identity = _candidate_selection_identity_document(
        role="design",
        selection=selection,
        fit_set_fingerprint=identity.fit_set_fingerprint,
        rolling_fingerprints=identity.rolling_fingerprints,
        bootstrap_materialization_fingerprint=(
            identity.bootstrap_materialization_fingerprint
        ),
        bootstrap_rng_execution_fingerprint=(
            identity.bootstrap_rng_execution_fingerprint
        ),
        source_content_fingerprint=identity.source_content_fingerprint,
        viability_view_fingerprint=viability.view_fingerprint,
        viability_result_fingerprint=identity.viability_result_fingerprint,
        selection_fingerprint=selection_fingerprint,
        fixed_point_outcome=identity.fixed_point_outcome,
        fixed_point_failure_reason=identity.fixed_point_failure_reason,
        fixed_point_converged_at_iteration=(
            identity.fixed_point_converged_at_iteration
        ),
        fixed_point_iteration_fingerprints=(
            identity.fixed_point_iteration_fingerprints
        ),
        fixed_point_iteration_history_fingerprint=fixed_point_history,
        fixed_point_fingerprint=fixed_point_fingerprint,
        fixed_point_final_selection_fingerprint=fixed_point_final_selection,
        fixed_point_final_monthly_block_core_fingerprint=(
            identity.fixed_point_final_monthly_block_core_fingerprint
        ),
        fixed_point_final_daily_block_core_fingerprint=(
            identity.fixed_point_final_daily_block_core_fingerprint
        ),
        fixed_point_viability_projection_fingerprint=(fixed_point_viability_projection),
    )
    return _source_slots_from_identity(
        "design_predictive_selection",
        components=_component_fingerprints(
            "predictive_selection_component",
            "candidate_viability_parent_component",
        ),
        specialized_materializations={
            "candidate_gate_parent_materialization": _composite_slot_fingerprint(
                "candidate_gate_parent",
                {
                    "fit_set": identity.fit_set_fingerprint,
                    "source_content": identity.source_content_fingerprint,
                    "viability_result": identity.viability_result_fingerprint,
                },
            ),
            "rolling_score_bootstrap_materialization": (
                identity.bootstrap_materialization_fingerprint
            ),
            "design_fixed_point_history_materialization": fixed_point_history,
            "design_fixed_point_final_selection_materialization": (
                fixed_point_final_selection
            ),
            "design_fixed_point_final_block_parent_materialization": (
                fixed_point_block_parents
            ),
        },
        rng={
            "rolling_score_bootstrap_streams": (
                identity.bootstrap_rng_execution_fingerprint
            )
        },
        source_identity=source_identity,
    )


def sealed_holdout_executor_source_slots(
    selection: CandidateSelectionTaskView,
    evidence: SealedHoldoutEvidence,
    *,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
) -> ScientificTaskSourceSlots:
    """Derive holdout slots from the exact parent, slices, and raw scores."""

    if not is_candidate_selection_task_view(selection):
        raise ModelError("holdout prebinding requires candidate selection")
    if type(evidence) is not SealedHoldoutEvidence:
        raise ModelError("holdout prebinding requires SealedHoldoutEvidence")
    fit = calibration_view_module._selected_design_fit(selection)
    model_fingerprint = gaussian_hmm_fit_fingerprint(fit)
    design_fingerprint = calibration_slice_fingerprint(design)
    holdout_fingerprint = calibration_slice_fingerprint(holdout)
    supplied = sealed_holdout_evidence_fingerprint(evidence)
    recomputed = evaluate_sealed_holdout(
        fit,
        design,
        holdout,
    )
    if (
        design.role != "design"
        or holdout.role != "holdout"
        or sealed_holdout_evidence_fingerprint(recomputed) != supplied
    ):
        raise ModelError("holdout prebinding evidence differs from exact sources")
    source_identity = _holdout_identity_document(
        role="design",
        selected_n_states=fit.n_states,
        selected_model_fingerprint=model_fingerprint,
        design_slice_fingerprint=design_fingerprint,
        holdout_slice_fingerprint=holdout_fingerprint,
        selection_view_fingerprint=selection.view_fingerprint,
        evidence_fingerprint=supplied,
    )
    return _source_slots_from_identity(
        "sealed_holdout_veto",
        components=_component_fingerprints(
            "sealed_holdout_veto_component",
            "predictive_selection_parent_component",
        ),
        specialized_materializations={
            "selected_design_model_parent_materialization": model_fingerprint,
            "sealed_holdout_score_materialization": supplied,
        },
        rng={},
        source_identity=source_identity,
    )


def production_fit_executor_source_slots(
    registered_execution: RegisteredProductionFitSameK,
) -> ScientificTaskSourceSlots:
    """Derive raw slots from the one object-led registered fit execution."""

    _selection, _design_fit, _production_features, _production_fit = (
        registered_production_fit_same_k_sources(registered_execution)
    )
    execution_identity = registered_production_fit_same_k_identity(registered_execution)
    source_identity = _registered_production_fit_identity_document(registered_execution)
    return _source_slots_from_identity(
        "production_fit_same_k",
        components=_component_fingerprints("production_fit_same_k_component"),
        specialized_materializations={
            "selected_design_model_parent_materialization": (
                execution_identity.design_model_fingerprint
            ),
            "full_calibration_input_materialization": (
                execution_identity.production_feature_fingerprint
            ),
        },
        rng={"production_fit_restarts": execution_identity.rng_execution_fingerprint},
        source_identity=source_identity,
    )


def macro_stability_executor_source_slots(
    *,
    task_id: MacroTaskId,
    evidence: MacroStabilityEvidence,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
    source_slice: CalibrationSlice,
    master_seed: int,
    scientific_version: int,
) -> ScientificTaskSourceSlots:
    """Derive one macro-family binding from its registered raw execution."""

    role = calibration_view_module._macro_task_role(task_id)
    parent_fit, parent_view_fingerprint, _ = calibration_view_module._macro_parent(
        role,
        parent,
    )
    expected_slice_role = "design" if role == "design" else "full"
    if source_slice.role != expected_slice_role:
        raise ModelError("macro prebinding source slice has the wrong role")
    execution = registered_macro_stability_execution_identity(evidence)
    parent_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)
    slice_fingerprint = calibration_slice_fingerprint(source_slice)
    input_fingerprint = macro_stability_input_fingerprint(
        source_slice.macro_features,
        source_slice.monthly_period_ordinals,
        source_slice.monthly_continuation,
        role=role,
    )
    rng_fingerprint = macro_stability_rng_fingerprint(
        role=role,
        n_states=parent_fit.n_states,
        rows=len(source_slice.macro_features),
        macro_block_length=evidence.macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    evidence_fingerprint = macro_stability_evidence_fingerprint(evidence)
    if (
        execution.role != role
        or execution.n_states != parent_fit.n_states
        or execution.parent_model_fingerprint != parent_fingerprint
        or execution.macro_input_fingerprint != input_fingerprint
        or execution.master_seed != master_seed
        or execution.scientific_version != scientific_version
        or execution.rng_namespace != macro_stability_rng_namespace(role)
        or execution.rng_execution_fingerprint != rng_fingerprint
    ):
        raise ModelError("macro prebinding execution identity differs")
    source_identity = _macro_identity_document(
        role=role,
        selected_k=parent_fit.n_states,
        parent_model_fingerprint=parent_fingerprint,
        parent_view_fingerprint=parent_view_fingerprint,
        source_slice_fingerprint=slice_fingerprint,
        macro_input_fingerprint=input_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_namespace=execution.rng_namespace,
        rng_execution_fingerprint=rng_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )
    return _source_slots_from_identity(
        task_id,
        components=_component_fingerprints("macro_stability_component"),
        specialized_materializations={
            f"{role}_model_parent_materialization": parent_fingerprint,
            f"{role}_macro_materialization": input_fingerprint,
        },
        rng={f"{role}_macro_bootstrap_streams": rng_fingerprint},
        source_identity=source_identity,
    )


def adequacy_cell_executor_source_slots(
    evidence: RegisteredLatentCell | RegisteredRefittedAdequacyBatch,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
    *,
    task_id: AdequacyCellTaskId,
    strict_canonical: bool = True,
) -> ScientificTaskSourceSlots:
    """Derive one adequacy-cell binding from registered executor evidence."""

    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    task_specification = adequacy_gate_module._TASK_CELL_ROLE.get(task_id)
    if task_specification is None:
        raise ModelError("adequacy cell prebinding task ID is not registered")
    cell_id, role = task_specification
    expected_artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    parent_fit, parent_features, parent_view_fingerprint = (
        adequacy_gate_module._parent_sources(role, parent)
    )
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)

    if task_id in {
        "design_latent_correctness",
        "production_latent_correctness",
    }:
        if not isinstance(evidence, RegisteredLatentCell):
            raise ModelError("latent adequacy prebinding requires latent evidence")
        identity = registered_latent_cell_identity(evidence)
        if registered_latent_cell_source_fit(evidence) is not parent_fit:
            raise ModelError("latent adequacy prebinding has the wrong live parent")
        transition_fingerprint = scientific_array_fingerprint(
            parent_fit.parameters.transition_matrix,
            role="latent_transition_matrix",
        )
        if (
            evidence.artifact is not expected_artifact
            or "latent" not in cell_id
            or identity.parent_model_fingerprint != parent_model_fingerprint
            or identity.transition_matrix_fingerprint != transition_fingerprint
        ):
            raise ModelError("latent adequacy prebinding has cross-role evidence")
        record = adequacy_gate_module._latent_record(
            cast(Literal["design_latent", "production_latent"], cell_id),
            evidence,
            expected_artifact=expected_artifact,
            strict_canonical=strict_canonical,
        )
        canonical_count_ready = adequacy_gate_module._latent_canonical_geometry(
            evidence
        )
        passed = adequacy_gate_module._individual_cell_passed(
            record,
            canonical_count_ready=canonical_count_ready,
            strict_canonical=strict_canonical,
        )
        source_identity = _adequacy_cell_identity_document(
            role=role,
            parent_view_fingerprint=parent_view_fingerprint,
            parent_model_fingerprint=parent_model_fingerprint,
            scientific_input_fingerprint=transition_fingerprint,
            rng_execution_fingerprint=identity.rng_execution_fingerprint,
            simulation_rng_execution_fingerprint=(
                identity.simulation_rng_execution_fingerprint
            ),
            bootstrap_rng_execution_fingerprint=(
                identity.bootstrap_rng_execution_fingerprint
            ),
            source_evidence_fingerprint=identity.source_evidence_fingerprint,
            cell=record,
            canonical_count_ready=canonical_count_ready,
            ready=record.ready,
            passed=passed,
            strict_canonical=strict_canonical,
        )
        return _source_slots_from_identity(
            task_id,
            components=_component_fingerprints("latent_correctness_component"),
            specialized_materializations={
                f"{role}_model_parent_materialization": parent_model_fingerprint
            },
            rng={
                f"{role}_latent_simulation_streams": (
                    identity.simulation_rng_execution_fingerprint
                ),
                f"{role}_latent_bootstrap_streams": (
                    identity.bootstrap_rng_execution_fingerprint
                ),
            },
            source_identity=source_identity,
        )

    if task_id not in {
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    } or not is_registered_refitted_adequacy_batch(evidence):
        raise ModelError("refitted adequacy prebinding requires refit evidence")
    if evidence.role != role or "refitted" not in cell_id:
        raise ModelError("refitted adequacy prebinding has cross-role evidence")
    refit_identity = registered_refitted_adequacy_batch_identity(evidence)
    source_fit, source_features = registered_refitted_adequacy_sources(evidence)
    if source_fit is not parent_fit or (
        role == "production" and source_features is not parent_features
    ):
        raise ModelError("refitted adequacy prebinding has the wrong live parent")
    feature_fingerprint = hmm_feature_matrix_fingerprint(source_features)
    if (
        refit_identity.parent_model_fingerprint != parent_model_fingerprint
        or refit_identity.artifact_feature_fingerprint != feature_fingerprint
    ):
        raise ModelError("refitted adequacy prebinding parent identity differs")
    record = adequacy_gate_module._refit_record(
        cast(
            Literal[
                "design_refitted_adequacy",
                "production_refitted_adequacy",
            ],
            cell_id,
        ),
        evidence,
        expected_role=role,
        strict_canonical=strict_canonical,
    )
    canonical_count_ready = adequacy_gate_module._refit_canonical_count(evidence)
    passed = adequacy_gate_module._individual_cell_passed(
        record,
        canonical_count_ready=canonical_count_ready,
        strict_canonical=strict_canonical,
    )
    source_identity = _adequacy_cell_identity_document(
        role=role,
        parent_view_fingerprint=parent_view_fingerprint,
        parent_model_fingerprint=parent_model_fingerprint,
        scientific_input_fingerprint=feature_fingerprint,
        rng_execution_fingerprint=refit_identity.rng_execution_fingerprint,
        simulation_rng_execution_fingerprint=None,
        bootstrap_rng_execution_fingerprint=None,
        source_evidence_fingerprint=refit_identity.source_evidence_fingerprint,
        cell=record,
        canonical_count_ready=canonical_count_ready,
        ready=record.ready,
        passed=passed,
        strict_canonical=strict_canonical,
    )
    return _source_slots_from_identity(
        task_id,
        components=_component_fingerprints("refitted_adequacy_component"),
        specialized_materializations={
            f"{role}_model_parent_materialization": parent_model_fingerprint
        },
        rng={
            f"{role}_refit_adequacy_streams": (refit_identity.rng_execution_fingerprint)
        },
        source_identity=source_identity,
    )


def adequacy_holm_executor_source_slots(
    *,
    design_latent: AdequacyCellTaskView,
    production_latent: AdequacyCellTaskView,
    design_refitted: AdequacyCellTaskView,
    production_refitted: AdequacyCellTaskView,
    alpha: float = 0.05,
    strict_canonical: bool = True,
) -> ScientificTaskSourceSlots:
    """Derive Holm slots from the four exact upstream adequacy views."""

    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    sources = (
        design_latent,
        production_latent,
        design_refitted,
        production_refitted,
    )
    if any(not is_adequacy_cell_task_view(item) for item in sources):
        raise ModelError("adequacy Holm prebinding requires guarded source views")
    expected_order: tuple[str, str, str, str] = (
        "design_latent_correctness",
        "production_latent_correctness",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )
    if tuple(item.task_id for item in sources) != expected_order:
        raise ModelError("adequacy Holm sources have the wrong task order")
    if len({item.run_id for item in sources}) != 1:
        raise ModelError("adequacy Holm cannot mix source runs")
    if len({item.binding_fingerprint for item in sources}) != 4:
        raise ModelError("adequacy Holm requires four distinct source bindings")
    if any(item.strict_canonical is not strict_canonical for item in sources):
        raise ModelError("adequacy Holm strict mode differs from a source task")
    cells = (
        design_latent.cell,
        production_latent.cell,
        design_refitted.cell,
        production_refitted.cell,
    )
    holm, ready, passed, failure_reasons = adequacy_gate_module._authoritative_holm(
        cells,
        source_passed=(
            design_latent.passed,
            production_latent.passed,
            design_refitted.passed,
            production_refitted.passed,
        ),
        alpha=alpha,
    )
    parent_materializations: dict[str, str] = {
        item.task_id: _task_source_materialization_fingerprint(
            item.task_id,
            _adequacy_cell_source_identity(item),
        )
        for item in sources
    }
    source_identity = _adequacy_holm_identity_document(
        source_task_order=expected_order,
        parent_materializations=parent_materializations,
        cells=cells,
        holm=holm,
        ready=ready,
        passed=passed,
        failure_reasons=failure_reasons,
        alpha=alpha,
        strict_canonical=strict_canonical,
    )
    return _source_slots_from_identity(
        "four_cell_adequacy_holm",
        components=_component_fingerprints(
            "adequacy_holm_component",
            "design_latent_parent_component",
            "design_refitted_parent_component",
            "production_latent_parent_component",
            "production_refitted_parent_component",
        ),
        specialized_materializations={
            "four_cell_adequacy_parent_materialization": (
                _composite_slot_fingerprint(
                    "four_cell_adequacy_parent",
                    {task: parent_materializations[task] for task in expected_order},
                )
            )
        },
        rng={},
        source_identity=source_identity,
    )


def block_calibration_executor_source_slots(
    source: BlockCalibrationGateEvidence,
    *,
    task_id: Literal[
        "design_block_calibration",
        "production_block_calibration",
    ],
) -> ScientificTaskSourceSlots:
    """Derive block/kernel/transition slots from the full raw evidence."""

    identities = boundary_view_module._block_source_identities(
        source,
        task_id=task_id,
    )
    role = source.role
    source_identity = _block_identity_document(
        role=role,
        selected_k=source.selected_k,
        identities=identities,
    )
    specialized_materializations = {
        f"{role}_model_parent_materialization": _required_block_identity(
            identities["parent_model_fingerprint"],
            f"{role} model parent",
        ),
        f"{role}_block_uniform_materialization": _required_block_identity(
            identities["block_uniform_materialization_fingerprint"],
            f"{role} block uniform materialization",
        ),
        f"{role}_kernel_calibration_materialization": (
            _composite_slot_fingerprint(
                f"{role}_kernel_calibration_materialization",
                {
                    "monthly": _required_block_identity(
                        identities["monthly_kernel_materialization_fingerprint"],
                        "monthly kernel materialization",
                    ),
                    "daily": _required_block_identity(
                        identities["daily_kernel_materialization_fingerprint"],
                        "daily kernel materialization",
                    ),
                },
            )
        ),
        f"{role}_transition_bootstrap_materialization": _required_block_identity(
            identities["transition_materialization_fingerprint"],
            f"{role} transition materialization",
        ),
    }
    if role == "design":
        specialized_materializations[
            "design_fixed_point_final_block_parent_materialization"
        ] = _required_block_identity(
            identities["design_fixed_point_final_block_parent_fingerprint"],
            "design fixed-point final block parent",
        )
    return _source_slots_from_identity(
        task_id,
        components=_component_fingerprints("block_calibration_component"),
        specialized_materializations=specialized_materializations,
        rng={
            f"{role}_block_calibration_streams": (
                _required_block_identity(
                    identities["block_calibration_rng_fingerprint"],
                    f"{role} block calibration RNG",
                )
            ),
            f"{role}_kernel_calibration_streams": (
                _required_block_identity(
                    identities["kernel_calibration_rng_fingerprint"],
                    f"{role} kernel calibration RNG",
                )
            ),
            f"{role}_transition_bootstrap_streams": (
                _required_block_identity(
                    identities["transition_rng_fingerprint"],
                    f"{role} transition RNG",
                )
            ),
        },
        source_identity=source_identity,
    )


def _required_block_identity(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ModelError(f"{label} fingerprint is absent")
    return value


def bridge_executor_source_slots(
    source: StagedBridgeTaskEvidence,
    *,
    task_id: BridgeTaskId,
) -> ScientificTaskSourceSlots:
    """Derive one staged bridge task's slots before its own binding/view."""

    return _bridge_source_slots_from_evidence(source, task_id=task_id)


def bridge_executor_planned_looks(
    source: StagedBridgeTaskEvidence,
    *,
    task_id: BridgeTaskId,
) -> tuple[CanonicalG3PlannedLook, ...]:
    """Project a Tier-B executor prefix into canonical G3 planned looks.

    This conversion accepts only producer-authorized staged evidence and is
    therefore available before the task's run/binding view is minted.  It is
    intentionally empty for resource, Tier-A, and actual-artifact tasks.
    """

    identity = staged_bridge_task_evidence_identity(source)
    if identity.task_id != task_id:
        raise ModelError("staged bridge planned-look source has the wrong task")
    local_looks = staged_bridge_task_planned_looks(source)
    tier_b = task_id in {
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    }
    if not tier_b:
        if local_looks:
            raise ModelError("non-Tier-B bridge source acquired planned looks")
        return ()

    specs = tuple(
        item for item in G3_PLANNED_LOOK_REGISTRY if item.owner_task_id == task_id
    )
    if len(specs) != 1:
        raise AssertionError("Tier-B task lacks one planned-look registry entry")
    spec = specs[0]
    schedule = spec.sample_counts
    if not local_looks or len(local_looks) > len(schedule):
        raise ModelError("Tier-B executor planned-look prefix length is invalid")
    expected_dgp: Literal["model", "nonparametric"] = (
        "nonparametric" if "nonparametric" in task_id else "model"
    )
    result: list[CanonicalG3PlannedLook] = []
    for index, look in enumerate(local_looks, start=1):
        look_identity = staged_bridge_planned_look_identity(look)
        if (
            look_identity.task_id != task_id
            or look_identity.dgp != expected_dgp
            or look_identity.look_index != index
            or look_identity.sample_count != schedule[index - 1]
            or look_identity.common_binding_fingerprint
            != identity.common_binding_fingerprint
            or look_identity.parent_task_fingerprints
            != identity.parent_task_fingerprints
            or look_identity.source_materialization_fingerprint
            != identity.source_materialization_fingerprint
        ):
            raise ModelError("Tier-B executor planned-look lineage is inconsistent")
        is_last = index == len(local_looks)
        if (not is_last and look_identity.decision != "continue") or (
            is_last and look_identity.decision == "continue"
        ):
            raise ModelError("Tier-B executor planned-look prefix is nonterminal")
        if is_last and (
            look_identity.rng_execution_fingerprint
            != identity.rng_execution_fingerprint
            or look_identity.result_fingerprint != identity.result_fingerprint
        ):
            raise ModelError("Tier-B terminal look differs from its task result")
        decision: Literal["continue", "pass", "fail"] = (
            "continue"
            if look_identity.decision == "continue"
            else "pass"
            if look_identity.decision == "pass_above_tail"
            else "fail"
        )
        interval = clopper_pearson_interval(
            look_identity.exceedances,
            look_identity.sample_count,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        result.append(
            CanonicalG3PlannedLook(
                spec.family_id,
                index,
                look_identity.sample_count,
                decision,
                MappingProxyType(
                    {
                        "bridge_binding_fingerprint": (
                            identity.common_binding_fingerprint
                        ),
                        "bridge_task_result_fingerprint": (
                            identity.evidence_fingerprint
                        ),
                        "dgp": expected_dgp,
                        "estimate": interval.estimate,
                        "exceedances": look_identity.exceedances,
                        "interval_lower": interval.lower,
                        "interval_upper": interval.upper,
                        "local_bridge_look_receipt_fingerprint": (
                            look_identity.stage_receipt_fingerprint
                        ),
                        "look_evidence_fingerprint": (
                            look_identity.evidence_fingerprint
                        ),
                        "look_rng_fingerprint": (
                            look_identity.rng_execution_fingerprint
                        ),
                        "tail_probability": 0.025,
                    }
                ),
            )
        )
    return tuple(result)


def _bridge_source_slots_from_evidence(
    source: StagedBridgeTaskEvidence,
    *,
    task_id: BridgeTaskId,
) -> ScientificTaskSourceSlots:
    identities = boundary_view_module._staged_bridge_source_identities(
        source,
        task_id=task_id,
    )
    identity = staged_bridge_task_evidence_identity(source)
    canonical_looks = bridge_executor_planned_looks(source, task_id=task_id)
    local_looks = staged_bridge_task_planned_looks(source)
    look_identities = tuple(
        staged_bridge_planned_look_identity(item) for item in local_looks
    )
    source_identity = _bridge_identity_document(
        identity=identity,
        identities=identities,
        look_identities=look_identities,
        canonical_looks=canonical_looks,
    )
    component_slot, materialization_slot, rng_slot = _BRIDGE_SLOT_LAYOUT[task_id]
    materialization = _bridge_fingerprint_field(
        identities,
        "materialization_fingerprint",
    )
    raw_rng = identities["rng_fingerprint"]
    rng: dict[str, str]
    if rng_slot is None:
        if raw_rng is not None:
            raise ModelError("RNG-empty actual-artifact bridge acquired a stream")
        rng = {}
    else:
        if not isinstance(raw_rng, str) or _SHA256.fullmatch(raw_rng) is None:
            raise ModelError("staged bridge RNG identity is invalid")
        rng = {rng_slot: raw_rng}
    return _source_slots_from_identity(
        task_id,
        components=_component_fingerprints(component_slot),
        specialized_materializations={materialization_slot: materialization},
        rng=rng,
        source_identity=source_identity,
    )


def _bridge_identity_document(
    *,
    identity: Any,
    identities: Mapping[str, object],
    look_identities: Sequence[Any],
    canonical_looks: Sequence[CanonicalG3PlannedLook],
) -> Mapping[str, object]:
    parents = tuple(identity.parent_task_fingerprints)
    if identities["parent_result_fingerprints"] != parents:
        raise ModelError("staged bridge direct-parent lineage changed")
    return {
        "dgp": identity.dgp,
        "passed": identity.passed,
        "base_preflight_fingerprint": _bridge_fingerprint_field(
            identities,
            "base_preflight_fingerprint",
        ),
        "selected_k": identities["selected_k"],
        "shared_binding_fingerprint": _bridge_fingerprint_field(
            identities,
            "shared_binding_fingerprint",
        ),
        "parent_result_fingerprints": [
            {"task_id": parent, "evidence_fingerprint": fingerprint}
            for parent, fingerprint in parents
        ],
        "scientific_result_fingerprint": identity.result_fingerprint,
        "task_result_fingerprint": _bridge_fingerprint_field(
            identities,
            "task_result_fingerprint",
        ),
        "source_lineage_fingerprint": _bridge_fingerprint_field(
            identities,
            "source_lineage_fingerprint",
        ),
        "materialization_fingerprint": _bridge_fingerprint_field(
            identities,
            "materialization_fingerprint",
        ),
        "known_trace_fingerprint": identities["known_trace_fingerprint"],
        "rng_fingerprint": identities["rng_fingerprint"],
        "stage_receipt_fingerprint": identity.stage_receipt_fingerprint,
        "planned_look_evidence": [
            {
                "look_index": item.look_index,
                "sample_count": item.sample_count,
                "exceedances": item.exceedances,
                "decision": item.decision,
                "common_binding_fingerprint": item.common_binding_fingerprint,
                "parent_task_fingerprints": [
                    {"task_id": parent, "evidence_fingerprint": fingerprint}
                    for parent, fingerprint in item.parent_task_fingerprints
                ],
                "source_materialization_fingerprint": (
                    item.source_materialization_fingerprint
                ),
                "rng_execution_fingerprint": item.rng_execution_fingerprint,
                "result_fingerprint": item.result_fingerprint,
                "evidence_fingerprint": item.evidence_fingerprint,
                "stage_receipt_fingerprint": item.stage_receipt_fingerprint,
            }
            for item in look_identities
        ],
        "canonical_planned_look_projection": [
            {
                "family_id": item.family_id,
                "look_index": item.look_index,
                "sample_count": item.sample_count,
                "decision": item.decision,
                "observed": dict(item.observed),
            }
            for item in canonical_looks
        ],
    }


def _bridge_fingerprint_field(
    identities: Mapping[str, object],
    field: str,
) -> str:
    value = identities[field]
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"staged bridge {field} is not a SHA-256 fingerprint")
    return value


def artifact_integrity_executor_source_slots(
    source: ArtifactIntegrityDecisionEvidence,
    actual_bridge_parent: BridgeDriverTaskView,
    *,
    run_id: str,
) -> ScientificTaskSourceSlots:
    """Derive final artifact-integrity slots before its own binding exists."""

    if (
        not boundary_view_module.is_bridge_driver_task_view(actual_bridge_parent)
        or actual_bridge_parent.task_id != "actual_artifact_bridge"
        or actual_bridge_parent.run_id != run_id
    ):
        raise ModelError("artifact prebinding requires its exact actual-bridge parent")
    identities = boundary_view_module._artifact_source_identities(
        source,
        run_id=run_id,
        actual_bridge_parent_view_fingerprint=(actual_bridge_parent.view_fingerprint),
    )
    source_identity = _artifact_identity_document(
        source=source,
        identities=identities,
    )
    return _source_slots_from_identity(
        "artifact_integrity",
        components=_component_fingerprints(
            "artifact_integrity_component",
            "actual_artifact_bridge_parent_component",
        ),
        specialized_materializations={
            "final_scientific_artifact_parent_materialization": (
                identities["final_artifact_parent_fingerprint"]
            )
        },
        rng={},
        source_identity=source_identity,
    )


def fragmentation_executor_source_slots(
    block_parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
    *,
    task_id: Literal["design_fragmentation", "production_fragmentation"],
) -> ScientificTaskSourceSlots:
    """Derive exact fragmentation slots before its run-bound view exists."""

    role = _dependence_task_role(task_id, expected_kind="fragmentation")
    source_identity = _fragmentation_executor_source_identity(
        block_parents,
        role=role,
    )
    core_fingerprints = cast(
        list[str],
        source_identity["parent_block_core_fingerprints"],
    )
    return _source_slots_from_identity(
        task_id,
        components=_component_fingerprints(
            "fragmentation_component",
            f"{role}_block_calibration_parent_component",
        ),
        specialized_materializations={
            f"{role}_block_calibration_parent_materialization": (
                _ordered_composite_slot_fingerprint(
                    f"{role}_block_calibration_parent",
                    role=role,
                    fingerprints=core_fingerprints,
                )
            )
        },
        rng={},
        source_identity=source_identity,
    )


def dependence_executor_source_slots(
    source: RegisteredArtifactDependenceExecution,
    *,
    task_id: Literal["design_dependence", "production_dependence"],
) -> ScientificTaskSourceSlots:
    """Derive exact dependence slots from its guarded pre-view execution."""

    if not is_registered_staged_dependence_execution(source):
        raise ModelError("dependence prebinding requires registered staged evidence")
    role = _dependence_task_role(task_id, expected_kind="dependence")
    if source.role != role:
        raise ModelError("dependence prebinding task ID has the wrong role")
    source_identity = _dependence_execution_source_identity(source)
    return _source_slots_from_identity(
        task_id,
        components=_component_fingerprints("dependence_component"),
        specialized_materializations={
            f"{role}_block_calibration_parent_materialization": (
                _dependence_block_parent_materialization(source)
            ),
            f"{role}_dependence_uniform_materialization": (
                _dependence_uniform_materialization(source)
            ),
        },
        rng={f"{role}_dependence_streams": _dependence_rng_execution(source)},
        source_identity=source_identity,
    )


def dependence_holm_executor_source_slots(
    design: DependenceTaskView,
    production: DependenceTaskView,
    *,
    run_id: str,
    alpha: float = 0.05,
) -> ScientificTaskSourceSlots:
    """Derive Holm slots before minting the Holm task's own binding/view."""

    source_identity = _prebinding_dependence_holm_source_identity(
        design,
        production,
        run_id=run_id,
        alpha=alpha,
    )
    parents = _dependence_parent_source_materializations((design, production))
    return _source_slots_from_identity(
        "four_cell_dependence_holm",
        components=_component_fingerprints(
            "dependence_holm_component",
            "design_dependence_parent_component",
            "production_dependence_parent_component",
        ),
        specialized_materializations={
            "four_cell_dependence_parent_materialization": (
                _composite_slot_fingerprint(
                    "four_cell_dependence_parent",
                    parents,
                )
            )
        },
        rng={},
        source_identity=source_identity,
    )


# Audit-facing completeness registry.  Tasks may share a family API, but every
# supported task is named explicitly so a newly enabled task cannot silently
# lack a cycle-free executor-to-binding seam.
SCIENTIFIC_TASK_PREBINDING_API_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "input_integrity": "input_integrity_executor_source_slots",
        "design_candidate_viability": "candidate_viability_executor_source_slots",
        "design_predictive_selection": "candidate_selection_executor_source_slots",
        "sealed_holdout_veto": "sealed_holdout_executor_source_slots",
        "design_macro_stability": "macro_stability_executor_source_slots",
        "design_latent_correctness": "adequacy_cell_executor_source_slots",
        "design_refitted_adequacy": "adequacy_cell_executor_source_slots",
        "design_block_calibration": "block_calibration_executor_source_slots",
        "design_fragmentation": "fragmentation_executor_source_slots",
        "design_dependence": "dependence_executor_source_slots",
        "production_fit_same_k": "production_fit_executor_source_slots",
        "production_macro_stability": "macro_stability_executor_source_slots",
        "production_latent_correctness": "adequacy_cell_executor_source_slots",
        "production_refitted_adequacy": "adequacy_cell_executor_source_slots",
        "production_block_calibration": "block_calibration_executor_source_slots",
        "production_fragmentation": "fragmentation_executor_source_slots",
        "production_dependence": "dependence_executor_source_slots",
        "four_cell_adequacy_holm": "adequacy_holm_executor_source_slots",
        "four_cell_dependence_holm": "dependence_holm_executor_source_slots",
        "bridge_resource_preflight": "bridge_executor_source_slots",
        "bridge_tier_a_model_dgp": "bridge_executor_source_slots",
        "bridge_tier_a_nonparametric_dgp": "bridge_executor_source_slots",
        "bridge_tier_b_model_dgp": "bridge_executor_source_slots",
        "bridge_tier_b_nonparametric_dgp": "bridge_executor_source_slots",
        "actual_artifact_bridge": "bridge_executor_source_slots",
        "artifact_integrity": "artifact_integrity_executor_source_slots",
    }
)


def _component_fingerprints(*slots: str) -> dict[str, str]:
    return {slot: scientific_component_slot_fingerprint(slot) for slot in slots}


def _source_slots_from_identity(
    task_id: str,
    *,
    components: Mapping[str, str],
    specialized_materializations: Mapping[str, str],
    rng: Mapping[str, str],
    source_identity: Mapping[str, object],
) -> ScientificTaskSourceSlots:
    materializations = {
        TASK_SOURCE_MATERIALIZATION_SLOT: (
            _task_source_materialization_fingerprint(task_id, source_identity)
        ),
        **specialized_materializations,
    }
    return ScientificTaskSourceSlots(
        task_id,
        tuple(components.items()),
        tuple(materializations.items()),
        tuple(rng.items()),
    )


def _task_source_materialization_fingerprint(
    task_id: str,
    identity: Mapping[str, object],
) -> str:
    forbidden = {"run_id", "binding_fingerprint", "view_fingerprint"}
    overlap = forbidden.intersection(identity)
    if overlap:
        raise AssertionError(
            "task source materialization contains circular coordinates: "
            + ", ".join(sorted(overlap))
        )
    return _fingerprint(
        {
            "schema_id": TASK_SOURCE_MATERIALIZATION_SCHEMA_ID,
            "contract_version": G3_CONTRACT_VERSION,
            "task_id": task_id,
            "scientific_outcome_and_evidence": dict(identity),
        }
    )


def _input_task_source_identity(
    source: InputIntegrityDecisionEvidence,
) -> Mapping[str, object]:
    artifact_binding = dict(source.binding.as_dict())
    artifact_binding.pop("calibration_run_fingerprint", None)
    return {
        "launch_preflight_fingerprint": source.launch_preflight.fingerprint,
        "live_preflight_fingerprint": source.live_preflight.fingerprint,
        "calibration_input_fingerprint": (source.binding.calibration_input_fingerprint),
        "artifact_binding_without_run": artifact_binding,
    }


def _candidate_identity_field(
    identity: CandidateViabilityExecutionIdentity | CandidateGateExecutionIdentity,
    field: str,
) -> str:
    value = getattr(identity, field, None)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"registered candidate identity lacks {field}")
    return value


def _optional_slot_fingerprint(value: object, label: str) -> str:
    if isinstance(value, str) and _SHA256.fullmatch(value) is not None:
        return value
    if value is None:
        return _fingerprint(
            {
                "schema_id": "prpg-g3-absent-scientific-slot-v1",
                "label": label,
                "present": False,
            }
        )
    raise ModelError(f"registered candidate identity has invalid optional {label}")


def _candidate_suite_view_field(source: Any, field: str) -> str:
    value = getattr(source, field, None)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"registered candidate view lacks {field}")
    return value


def _candidate_viability_source_identity(source: Any) -> Mapping[str, object]:
    return _candidate_viability_identity_document(
        role=source.role,
        rows=source.viability_rows,
        admissible_n_states=source.admissible_n_states,
        fit_set_fingerprint=source.fit_set_fingerprint,
        source_content_fingerprint=source.source_content_fingerprint,
        viability_result_fingerprint=source.viability_result_fingerprint,
        master_seed=source.master_seed,
        scientific_version=source.scientific_version,
        rng_execution_fingerprint=source.rng_execution_fingerprint,
        restart_seed_fingerprints=source.restart_seed_fingerprints,
        viability_fingerprint=source.viability_fingerprint,
        suite_fingerprint=_candidate_suite_view_field(
            source,
            "suite_fingerprint",
        ),
        suite_combined_materialization_fingerprint=_candidate_suite_view_field(
            source,
            "suite_combined_materialization_fingerprint",
        ),
        suite_combined_fit_rng_execution_fingerprint=_candidate_suite_view_field(
            source,
            "suite_combined_fit_rng_execution_fingerprint",
        ),
        suite_combined_rng_execution_fingerprint=_candidate_suite_view_field(
            source,
            "suite_combined_rng_execution_fingerprint",
        ),
        suite_macro_materialization_fingerprint=_candidate_suite_view_field(
            source,
            "suite_macro_materialization_fingerprint",
        ),
        suite_monthly_block_materialization_fingerprint=_candidate_suite_view_field(
            source,
            "suite_monthly_block_materialization_fingerprint",
        ),
        suite_daily_block_materialization_fingerprint=_candidate_suite_view_field(
            source,
            "suite_daily_block_materialization_fingerprint",
        ),
        fixed_point_final_iteration=source.fixed_point_final_iteration,
        fixed_point_viability_projection_fingerprint=(
            _candidate_suite_view_field(
                source,
                "fixed_point_viability_projection_fingerprint",
            )
        ),
    )


def _candidate_viability_identity_document(
    *,
    role: str,
    rows: Sequence[Any],
    admissible_n_states: Sequence[int],
    fit_set_fingerprint: str,
    source_content_fingerprint: str,
    viability_result_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    rng_execution_fingerprint: str,
    restart_seed_fingerprints: Sequence[str],
    viability_fingerprint: str,
    suite_fingerprint: str,
    suite_combined_materialization_fingerprint: str,
    suite_combined_fit_rng_execution_fingerprint: str,
    suite_combined_rng_execution_fingerprint: str,
    suite_macro_materialization_fingerprint: str,
    suite_monthly_block_materialization_fingerprint: str,
    suite_daily_block_materialization_fingerprint: str,
    fixed_point_final_iteration: int,
    fixed_point_viability_projection_fingerprint: str,
) -> Mapping[str, object]:
    admissible = list(admissible_n_states)
    return {
        "role": role,
        "viability_rows": [
            {
                "n_states": row.n_states,
                "fit_succeeded": row.fit_succeeded,
                "viability": row.viability.as_dict(),
            }
            for row in rows
        ],
        "admissible_n_states": admissible,
        "passed": OWNER_FIXED_N_STATES in admissible,
        "fit_set_fingerprint": fit_set_fingerprint,
        "source_content_fingerprint": source_content_fingerprint,
        "viability_result_fingerprint": viability_result_fingerprint,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_namespace": "design_candidate_fit_restarts",
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "restart_seed_fingerprints": list(restart_seed_fingerprints),
        "viability_fingerprint": viability_fingerprint,
        "suite_fingerprint": suite_fingerprint,
        "suite_combined_materialization_fingerprint": (
            suite_combined_materialization_fingerprint
        ),
        "suite_combined_fit_rng_execution_fingerprint": (
            suite_combined_fit_rng_execution_fingerprint
        ),
        "suite_combined_rng_execution_fingerprint": (
            suite_combined_rng_execution_fingerprint
        ),
        "suite_macro_materialization_fingerprint": (
            suite_macro_materialization_fingerprint
        ),
        "suite_monthly_block_materialization_fingerprint": (
            suite_monthly_block_materialization_fingerprint
        ),
        "suite_daily_block_materialization_fingerprint": (
            suite_daily_block_materialization_fingerprint
        ),
        "fixed_point_final_iteration": fixed_point_final_iteration,
        "fixed_point_viability_projection_fingerprint": (
            fixed_point_viability_projection_fingerprint
        ),
    }


def _candidate_selection_source_identity(source: Any) -> Mapping[str, object]:
    return _candidate_selection_identity_document(
        role=source.role,
        selection=source.selection,
        fit_set_fingerprint=source.fit_set_fingerprint,
        rolling_fingerprints=source.rolling_fingerprints,
        bootstrap_materialization_fingerprint=(
            source.bootstrap_materialization_fingerprint
        ),
        bootstrap_rng_execution_fingerprint=(
            source.bootstrap_rng_execution_fingerprint
        ),
        source_content_fingerprint=source.source_content_fingerprint,
        viability_view_fingerprint=source.viability_view_fingerprint,
        viability_result_fingerprint=source.viability_result_fingerprint,
        selection_fingerprint=source.selection_fingerprint,
        fixed_point_outcome=source.fixed_point_outcome,
        fixed_point_failure_reason=source.fixed_point_failure_reason,
        fixed_point_converged_at_iteration=(source.fixed_point_converged_at_iteration),
        fixed_point_iteration_fingerprints=(source.fixed_point_iteration_fingerprints),
        fixed_point_iteration_history_fingerprint=(
            source.fixed_point_iteration_history_fingerprint
        ),
        fixed_point_fingerprint=source.fixed_point_fingerprint,
        fixed_point_final_selection_fingerprint=(
            source.fixed_point_final_selection_fingerprint
        ),
        fixed_point_final_monthly_block_core_fingerprint=(
            source.fixed_point_final_monthly_block_core_fingerprint
        ),
        fixed_point_final_daily_block_core_fingerprint=(
            source.fixed_point_final_daily_block_core_fingerprint
        ),
        fixed_point_viability_projection_fingerprint=(
            source.fixed_point_viability_projection_fingerprint
        ),
    )


def _candidate_selection_identity_document(
    *,
    role: str,
    selection: Any,
    fit_set_fingerprint: str,
    rolling_fingerprints: Sequence[str | None],
    bootstrap_materialization_fingerprint: str,
    bootstrap_rng_execution_fingerprint: str,
    source_content_fingerprint: str,
    viability_view_fingerprint: str,
    viability_result_fingerprint: str,
    selection_fingerprint: str,
    fixed_point_outcome: str,
    fixed_point_failure_reason: str | None,
    fixed_point_converged_at_iteration: int,
    fixed_point_iteration_fingerprints: Sequence[str],
    fixed_point_iteration_history_fingerprint: str,
    fixed_point_fingerprint: str,
    fixed_point_final_selection_fingerprint: str,
    fixed_point_final_monthly_block_core_fingerprint: str | None,
    fixed_point_final_daily_block_core_fingerprint: str | None,
    fixed_point_viability_projection_fingerprint: str,
) -> Mapping[str, object]:
    return {
        "role": role,
        "selection": selection.as_dict(),
        "selected_n_states": selection.selected_n_states,
        "selection_passed": selection.passed,
        "passed": fixed_point_outcome == "converged",
        "fit_set_fingerprint": fit_set_fingerprint,
        "rolling_fingerprints": list(rolling_fingerprints),
        "rolling_bootstrap_materialization_fingerprint": (
            bootstrap_materialization_fingerprint
        ),
        "rolling_bootstrap_rng_execution_fingerprint": (
            bootstrap_rng_execution_fingerprint
        ),
        "source_content_fingerprint": source_content_fingerprint,
        "viability_parent_view_fingerprint": viability_view_fingerprint,
        "viability_result_fingerprint": viability_result_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "fixed_point_outcome": fixed_point_outcome,
        "fixed_point_failure_reason": fixed_point_failure_reason,
        "fixed_point_converged_at_iteration": fixed_point_converged_at_iteration,
        "fixed_point_iteration_fingerprints": list(fixed_point_iteration_fingerprints),
        "fixed_point_iteration_history_fingerprint": (
            fixed_point_iteration_history_fingerprint
        ),
        "fixed_point_fingerprint": fixed_point_fingerprint,
        "fixed_point_final_selection_fingerprint": (
            fixed_point_final_selection_fingerprint
        ),
        "fixed_point_final_monthly_block_core_fingerprint": (
            fixed_point_final_monthly_block_core_fingerprint
        ),
        "fixed_point_final_daily_block_core_fingerprint": (
            fixed_point_final_daily_block_core_fingerprint
        ),
        "fixed_point_viability_projection_fingerprint": (
            fixed_point_viability_projection_fingerprint
        ),
    }


def _holdout_source_identity(source: Any) -> Mapping[str, object]:
    return _holdout_identity_document(
        role=source.role,
        selected_n_states=source.selected_n_states,
        selected_model_fingerprint=source.selected_model_fingerprint,
        design_slice_fingerprint=source.design_slice_fingerprint,
        holdout_slice_fingerprint=source.holdout_slice_fingerprint,
        selection_view_fingerprint=source.selection_view_fingerprint,
        evidence_fingerprint=source.evidence_fingerprint,
    )


def _holdout_identity_document(
    *,
    role: str,
    selected_n_states: int,
    selected_model_fingerprint: str,
    design_slice_fingerprint: str,
    holdout_slice_fingerprint: str,
    selection_view_fingerprint: str,
    evidence_fingerprint: str,
) -> Mapping[str, object]:
    return {
        "role": role,
        "selected_n_states": selected_n_states,
        "selected_model_fingerprint": selected_model_fingerprint,
        "design_slice_fingerprint": design_slice_fingerprint,
        "holdout_slice_fingerprint": holdout_slice_fingerprint,
        "selection_parent_view_fingerprint": selection_view_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
    }


def _macro_source_identity(source: Any) -> Mapping[str, object]:
    return _macro_identity_document(
        role=source.role,
        selected_k=source.selected_k,
        parent_model_fingerprint=source.parent_model_fingerprint,
        parent_view_fingerprint=source.parent_view_fingerprint,
        source_slice_fingerprint=source.source_slice_fingerprint,
        macro_input_fingerprint=source.macro_input_fingerprint,
        master_seed=source.master_seed,
        scientific_version=source.scientific_version,
        rng_namespace=source.rng_namespace,
        rng_execution_fingerprint=source.rng_execution_fingerprint,
        evidence_fingerprint=source.evidence_fingerprint,
    )


def _macro_identity_document(
    *,
    role: str,
    selected_k: int,
    parent_model_fingerprint: str,
    parent_view_fingerprint: str,
    source_slice_fingerprint: str,
    macro_input_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    rng_namespace: str,
    rng_execution_fingerprint: str,
    evidence_fingerprint: str,
) -> Mapping[str, object]:
    return {
        "role": role,
        "selected_k": selected_k,
        "parent_model_fingerprint": parent_model_fingerprint,
        "parent_view_fingerprint": parent_view_fingerprint,
        "source_slice_fingerprint": source_slice_fingerprint,
        "macro_input_fingerprint": macro_input_fingerprint,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_namespace": rng_namespace,
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
    }


def _production_fit_source_identity(source: Any) -> Mapping[str, object]:
    registered_execution = registered_production_fit_execution_from_view(source)
    return _registered_production_fit_identity_document(registered_execution)


def _registered_production_fit_identity_document(
    registered_execution: RegisteredProductionFitSameK,
) -> Mapping[str, object]:
    selection, design_fit, _production_features, _production_fit = (
        registered_production_fit_same_k_sources(registered_execution)
    )
    identity = registered_production_fit_same_k_identity(registered_execution)
    return _production_fit_identity_document(
        role="production",
        design_selected_k=design_fit.n_states,
        design_model_fingerprint=identity.design_model_fingerprint,
        selection_view_fingerprint=selection.view_fingerprint,
        production_feature_fingerprint=identity.production_feature_fingerprint,
        master_seed=registered_execution.master_seed,
        scientific_version=registered_execution.scientific_version,
        rng_execution_fingerprint=identity.rng_execution_fingerprint,
        restart_seed_fingerprint=(identity.restart_seed_materialization_fingerprint),
        production_fit_fingerprint=identity.production_fit_fingerprint,
        registered_source_materialization_fingerprint=(
            identity.source_materialization_fingerprint
        ),
        registered_result_evidence_fingerprint=(identity.result_evidence_fingerprint),
        registered_execution_fingerprint=identity.execution_fingerprint,
    )


def _production_fit_identity_document(
    *,
    role: str,
    design_selected_k: int,
    design_model_fingerprint: str,
    selection_view_fingerprint: str,
    production_feature_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    rng_execution_fingerprint: str,
    restart_seed_fingerprint: str,
    production_fit_fingerprint: str,
    registered_source_materialization_fingerprint: str,
    registered_result_evidence_fingerprint: str,
    registered_execution_fingerprint: str,
) -> Mapping[str, object]:
    return {
        "role": role,
        "design_selected_k": design_selected_k,
        "design_model_fingerprint": design_model_fingerprint,
        "selection_parent_view_fingerprint": selection_view_fingerprint,
        "production_feature_fingerprint": production_feature_fingerprint,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_namespace": "production_fit_restarts",
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "restart_seed_fingerprint": restart_seed_fingerprint,
        "production_fit_fingerprint": production_fit_fingerprint,
        "registered_source_materialization_fingerprint": (
            registered_source_materialization_fingerprint
        ),
        "registered_result_evidence_fingerprint": (
            registered_result_evidence_fingerprint
        ),
        "registered_execution_fingerprint": registered_execution_fingerprint,
    }


def _adequacy_cell_source_identity(source: Any) -> Mapping[str, object]:
    return _adequacy_cell_identity_document(
        role=source.role,
        parent_view_fingerprint=source.parent_view_fingerprint,
        parent_model_fingerprint=source.parent_model_fingerprint,
        scientific_input_fingerprint=source.scientific_input_fingerprint,
        rng_execution_fingerprint=source.rng_execution_fingerprint,
        simulation_rng_execution_fingerprint=(
            source.simulation_rng_execution_fingerprint
        ),
        bootstrap_rng_execution_fingerprint=(
            source.bootstrap_rng_execution_fingerprint
        ),
        source_evidence_fingerprint=source.source_evidence_fingerprint,
        cell=source.cell,
        canonical_count_ready=source.canonical_count_ready,
        ready=source.ready,
        passed=source.passed,
        strict_canonical=source.strict_canonical,
    )


def _adequacy_cell_identity_document(
    *,
    role: str,
    parent_view_fingerprint: str,
    parent_model_fingerprint: str,
    scientific_input_fingerprint: str,
    rng_execution_fingerprint: str,
    simulation_rng_execution_fingerprint: str | None,
    bootstrap_rng_execution_fingerprint: str | None,
    source_evidence_fingerprint: str,
    cell: Any,
    canonical_count_ready: bool,
    ready: bool,
    passed: bool,
    strict_canonical: bool,
) -> Mapping[str, object]:
    return {
        "role": role,
        "parent_view_fingerprint": parent_view_fingerprint,
        "parent_model_fingerprint": parent_model_fingerprint,
        "scientific_input_fingerprint": scientific_input_fingerprint,
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "simulation_rng_execution_fingerprint": (simulation_rng_execution_fingerprint),
        "bootstrap_rng_execution_fingerprint": bootstrap_rng_execution_fingerprint,
        "source_evidence_fingerprint": source_evidence_fingerprint,
        "cell": {
            "cell_id": cell.cell_id,
            "ready": cell.ready,
            "p_value": cell.p_value,
            "repetitions": cell.repetitions,
            "successful_repetitions": cell.successful_repetitions,
            "failure_reason": cell.failure_reason,
        },
        "canonical_count_ready": canonical_count_ready,
        "ready": ready,
        "passed": passed,
        "strict_canonical": strict_canonical,
    }


def _adequacy_holm_source_identity(
    source: Any,
    *,
    parent_materializations: Mapping[str, str],
) -> Mapping[str, object]:
    return _adequacy_holm_identity_document(
        source_task_order=source.source_task_order,
        parent_materializations=parent_materializations,
        cells=source.cells,
        holm=source.holm,
        ready=source.ready,
        passed=source.passed,
        failure_reasons=source.failure_reasons,
        alpha=source.alpha,
        strict_canonical=source.strict_canonical,
    )


def _adequacy_holm_identity_document(
    *,
    source_task_order: Sequence[str],
    parent_materializations: Mapping[str, str],
    cells: Sequence[Any],
    holm: Any,
    ready: bool,
    passed: bool,
    failure_reasons: Sequence[str],
    alpha: float,
    strict_canonical: bool,
) -> Mapping[str, object]:
    return {
        "source_task_order": list(source_task_order),
        "source_materializations": {
            task: parent_materializations[task] for task in source_task_order
        },
        "cells": [_adequacy_cell_record_identity(item) for item in cells],
        "cell_order": [item.cell_id for item in cells],
        "holm": (
            None
            if holm is None
            else {
                "raw_p_values": holm.raw_p_values.tolist(),
                "adjusted_p_values": holm.adjusted_p_values.tolist(),
                "rejected": holm.rejected.tolist(),
                "alpha": holm.alpha,
            }
        ),
        "ready": ready,
        "passed": passed,
        "failure_reasons": list(failure_reasons),
        "alpha": alpha,
        "strict_canonical": strict_canonical,
    }


def _adequacy_cell_record_identity(cell: Any) -> Mapping[str, object]:
    return {
        "cell_id": cell.cell_id,
        "ready": cell.ready,
        "p_value": cell.p_value,
        "repetitions": cell.repetitions,
        "successful_repetitions": cell.successful_repetitions,
        "failure_reason": cell.failure_reason,
    }


def _dependence_task_role(
    task_id: str,
    *,
    expected_kind: Literal["fragmentation", "dependence"],
) -> Literal["design", "production"]:
    expected = {
        "design_fragmentation": ("design", "fragmentation"),
        "design_dependence": ("design", "dependence"),
        "production_fragmentation": ("production", "fragmentation"),
        "production_dependence": ("production", "dependence"),
    }.get(task_id)
    if expected is None or expected[1] != expected_kind:
        raise ModelError(f"{expected_kind} prebinding task ID is not registered")
    return cast(Literal["design", "production"], expected[0])


def _fragmentation_executor_source_identity(
    parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
    *,
    role: Literal["design", "production"],
) -> Mapping[str, object]:
    if (
        not isinstance(parents, tuple)
        or len(parents) != 2
        or any(not isinstance(item, RegisteredBlockCalibration) for item in parents)
    ):
        raise ModelError("fragmentation prebinding requires two block parents")
    cores = tuple(registered_block_core_identity(item) for item in parents)
    fragmentations = tuple(
        registered_block_fragmentation_identity(item) for item in parents
    )
    expected_roles = ((role, "monthly"), (role, "daily"))
    if tuple((item.artifact, item.frequency) for item in cores) != expected_roles:
        raise ModelError("fragmentation prebinding block roles/order differ")
    if (
        cores[0].selected_k != cores[1].selected_k
        or cores[0].parent_model_fingerprint != cores[1].parent_model_fingerprint
        or cores[0].master_seed != cores[1].master_seed
        or cores[0].scientific_version != cores[1].scientific_version
        or cores[0].materialization_fingerprint == cores[1].materialization_fingerprint
    ):
        raise ModelError("fragmentation prebinding block lineage differs")
    strict_modes = tuple(item.execution.geometry.strict_canonical for item in parents)
    if strict_modes[0] != strict_modes[1]:
        raise ModelError("fragmentation prebinding parent modes differ")

    cell_identities: list[Mapping[str, object]] = []
    for parent, core, fragmentation in zip(
        parents,
        cores,
        fragmentations,
        strict=True,
    ):
        if (
            fragmentation.parent_core_fingerprint != core.core_fingerprint
            or fragmentation.selected_k != core.selected_k
            or fragmentation.selected_length != core.selected_length
        ):
            raise ModelError("fragmentation prebinding evidence has the wrong core")
        states = [
            _fragmentation_gate_source_identity(item)
            for item in parent.execution.selected_fragmentation_gates
        ]
        if not states or [item["state"] for item in states] != list(
            range(core.selected_k)
        ):
            raise ModelError("fragmentation prebinding state order differs")
        cell_identities.append(
            {
                "cell_id": f"{role}_{core.frequency}",
                "artifact": core.artifact,
                "frequency": core.frequency,
                "histories": core.histories,
                "selected_length": core.selected_length,
                "common_draw_reused": True,
                "input_fingerprint": core.source_history_fingerprint,
                "model_fingerprint": core.parent_model_fingerprint,
                "materialization_fingerprint": core.materialization_fingerprint,
                "source_fingerprint": core.core_fingerprint,
                "source_evidence_fingerprint": (
                    fragmentation.fragmentation_evidence_fingerprint
                ),
                "starting_length": core.starting_length,
                "parent_block_core_fingerprint": core.core_fingerprint,
                "fragmentation": states,
                "fragmentation_evidence_fingerprint": (
                    fragmentation.fragmentation_evidence_fingerprint
                ),
            }
        )
    return _dependence_task_identity_document(
        role=role,
        task_kind="fragmentation",
        parent_block_core_fingerprints=[item.core_fingerprint for item in cores],
        fragmentation_parent_view_fingerprint=None,
        cells=cell_identities,
        strict_canonical=strict_modes[0],
    )


def _fragmentation_gate_source_identity(value: Any) -> Mapping[str, object]:
    return {
        "state": value.state,
        "intended_length": value.intended_length,
        "conditioned_count": value.conditioned.count,
        "conditioned_mean": value.conditioned.mean,
        "conditioned_median": value.conditioned.median,
        "conditioned_singleton_fraction": value.conditioned.singleton_fraction,
        "ideal_count": value.ideal.count,
        "ideal_median": value.ideal.median,
        "ideal_singleton_fraction": value.ideal.singleton_fraction,
        "reasons": list(value.reasons),
    }


def _fragmentation_state_source_identity(value: Any) -> Mapping[str, object]:
    return {
        "state": value.state,
        "intended_length": value.intended_length,
        "conditioned_count": value.conditioned_count,
        "conditioned_mean": value.conditioned_mean,
        "conditioned_median": value.conditioned_median,
        "conditioned_singleton_fraction": value.conditioned_singleton_fraction,
        "ideal_count": value.ideal_count,
        "ideal_median": value.ideal_median,
        "ideal_singleton_fraction": value.ideal_singleton_fraction,
        "reasons": list(value.reasons),
    }


def _fragmentation_task_cell_source_identity(value: Any) -> Mapping[str, object]:
    return {
        "cell_id": value.cell_id,
        "artifact": value.artifact,
        "frequency": value.frequency,
        "histories": value.histories,
        "selected_length": value.selected_length,
        "common_draw_reused": value.common_draw_reused,
        "input_fingerprint": value.input_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "materialization_fingerprint": value.materialization_fingerprint,
        "source_fingerprint": value.source_fingerprint,
        "source_evidence_fingerprint": value.source_evidence_fingerprint,
        "starting_length": value.starting_length,
        "parent_block_core_fingerprint": value.parent_block_core_fingerprint,
        "fragmentation": [
            _fragmentation_state_source_identity(item) for item in value.fragmentation
        ],
        "fragmentation_evidence_fingerprint": (
            value.fragmentation_evidence_fingerprint
        ),
    }


def _dependence_result_source_identity(value: Any) -> Mapping[str, object]:
    return {
        "intended_length": value.intended_length,
        "decision_role": value.decision_role,
        "observed_statistic": value.observed_statistic,
        "critical_value_95": value.critical_value_95,
        "p_value": value.p_value,
        "active_components": value.active_components,
        "repetitions": value.repetitions,
    }


def _dependence_task_cell_source_identity(value: Any) -> Mapping[str, object]:
    return {
        "cell_id": value.cell_id,
        "artifact": value.artifact,
        "frequency": value.frequency,
        "histories": value.histories,
        "selected_length": value.selected_length,
        "endpoint_count": value.endpoint_count,
        "common_draw_reused": value.common_draw_reused,
        "selected_result": _dependence_result_source_identity(value.selected_result),
        "loyo_fingerprint": value.loyo_fingerprint,
        "input_fingerprint": value.input_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "materialization_fingerprint": value.materialization_fingerprint,
        "source_fingerprint": value.source_fingerprint,
        "source_evidence_fingerprint": value.source_evidence_fingerprint,
        "starting_length": value.starting_length,
        "parent_block_core_fingerprint": value.parent_block_core_fingerprint,
        "purpose4_materialization_fingerprint": (
            value.purpose4_materialization_fingerprint
        ),
        "purpose4_stream_fingerprint": value.purpose4_stream_fingerprint,
        "rng_execution_fingerprint": value.rng_execution_fingerprint,
        "dependence_result_fingerprint": value.dependence_result_fingerprint,
    }


def _registered_dependence_cell_source_identity(value: Any) -> Mapping[str, object]:
    return {
        "cell_id": value.cell_id,
        "artifact": value.artifact,
        "frequency": value.frequency,
        "histories": value.histories,
        "selected_length": value.selected_length,
        "endpoint_count": value.endpoint_count,
        "common_draw_reused": value.common_draw_reused,
        "selected_result": _dependence_result_source_identity(value.selected_result),
        "loyo_fingerprint": value.loyo.fingerprint,
        "input_fingerprint": value.input_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "materialization_fingerprint": value.purpose4_materialization_fingerprint,
        "source_fingerprint": value.source_fingerprint,
        "source_evidence_fingerprint": value.evidence_fingerprint,
        "starting_length": value.starting_length,
        "parent_block_core_fingerprint": value.parent_block_core_fingerprint,
        "purpose4_materialization_fingerprint": (
            value.purpose4_materialization_fingerprint
        ),
        "purpose4_stream_fingerprint": value.purpose4_stream_fingerprint,
        "rng_execution_fingerprint": value.rng_execution_fingerprint,
        "dependence_result_fingerprint": value.dependence_result_fingerprint,
    }


def _dependence_task_identity_document(
    *,
    role: str,
    task_kind: str,
    parent_block_core_fingerprints: list[str],
    fragmentation_parent_view_fingerprint: str | None,
    cells: list[Mapping[str, object]],
    strict_canonical: bool,
) -> Mapping[str, object]:
    return {
        "role": role,
        "task_kind": task_kind,
        "parent_block_core_fingerprints": parent_block_core_fingerprints,
        "fragmentation_parent_view_fingerprint": (
            fragmentation_parent_view_fingerprint
        ),
        "cells": cells,
        "strict_canonical": strict_canonical,
        "canonical_stage": True,
        "report_only": False,
        "neighbors_report_only": True,
        "loyo_report_only": True,
    }


def _dependence_task_source_identity(
    source: DependenceTaskView,
) -> Mapping[str, object]:
    cells = (
        [_fragmentation_task_cell_source_identity(item) for item in source.cells]
        if source.task_kind == "fragmentation"
        else [_dependence_task_cell_source_identity(item) for item in source.cells]
    )
    return _dependence_task_identity_document(
        role=source.role,
        task_kind=source.task_kind,
        parent_block_core_fingerprints=list(source.parent_block_core_fingerprints),
        fragmentation_parent_view_fingerprint=(
            source.fragmentation_parent_view_fingerprint
        ),
        cells=cells,
        strict_canonical=source.strict_canonical,
    )


def _dependence_execution_source_identity(
    source: RegisteredArtifactDependenceExecution,
) -> Mapping[str, object]:
    return _dependence_task_identity_document(
        role=source.role,
        task_kind="dependence",
        parent_block_core_fingerprints=list(source.parent_block_core_fingerprints),
        fragmentation_parent_view_fingerprint=(
            source.fragmentation_parent_view_fingerprint
        ),
        cells=[
            _registered_dependence_cell_source_identity(item) for item in source.cells
        ],
        strict_canonical=source.strict_canonical,
    )


def _ordered_composite_slot_fingerprint(
    label: str,
    *,
    role: str,
    fingerprints: Sequence[str],
) -> str:
    if len(fingerprints) != 2:
        raise ModelError("dependence slot requires monthly and daily fingerprints")
    return _composite_slot_fingerprint(
        label,
        {
            f"{role}_monthly": fingerprints[0],
            f"{role}_daily": fingerprints[1],
        },
    )


def _dependence_block_parent_materialization(source: Any) -> str:
    return _ordered_composite_slot_fingerprint(
        f"{source.role}_block_calibration_parent",
        role=source.role,
        fingerprints=source.parent_block_core_fingerprints,
    )


def _dependence_uniform_materialization(source: Any) -> str:
    return _ordered_composite_slot_fingerprint(
        f"{source.role}_dependence_uniform_materialization",
        role=source.role,
        fingerprints=[
            item.purpose4_materialization_fingerprint for item in source.cells
        ],
    )


def _dependence_rng_execution(source: Any) -> str:
    return _ordered_composite_slot_fingerprint(
        f"{source.role}_dependence_rng_execution",
        role=source.role,
        fingerprints=[item.rng_execution_fingerprint for item in source.cells],
    )


def _dependence_parent_source_materializations(
    sources: tuple[DependenceTaskView, DependenceTaskView],
) -> dict[str, str]:
    design, production = sources
    return {
        "design_dependence": _task_source_materialization_fingerprint(
            design.task_id,
            _dependence_task_source_identity(design),
        ),
        "production_dependence": _task_source_materialization_fingerprint(
            production.task_id,
            _dependence_task_source_identity(production),
        ),
    }


def _dependence_holm_identity_document(
    *,
    parent_materializations: Mapping[str, str],
    cells: Sequence[Mapping[str, object]],
    raw_p_values: Sequence[float],
    adjusted_p_values: Sequence[float],
    rejected: Sequence[bool],
    alpha: float,
    holm_passed: bool,
    passed: bool,
    failure_reasons: Sequence[str],
    strict_canonical: bool,
) -> Mapping[str, object]:
    return {
        "source_materializations": dict(parent_materializations),
        "cell_order": list(DEPENDENCE_CELL_ORDER),
        "cells": list(cells),
        "holm": {
            "raw_p_values": list(raw_p_values),
            "adjusted_p_values": list(adjusted_p_values),
            "rejected": list(rejected),
            "alpha": alpha,
            "passed": holm_passed,
        },
        "passed": passed,
        "failure_reasons": list(failure_reasons),
        "strict_canonical": strict_canonical,
        "neighbors_report_only": True,
        "loyo_report_only": True,
    }


def _dependence_holm_source_identity(
    source: Any,
    *,
    parent_materializations: Mapping[str, str],
) -> Mapping[str, object]:
    holm = source.holm.holm
    return _dependence_holm_identity_document(
        parent_materializations=parent_materializations,
        cells=[_dependence_task_cell_source_identity(item) for item in source.cells],
        raw_p_values=holm.raw_p_values.tolist(),
        adjusted_p_values=holm.adjusted_p_values.tolist(),
        rejected=holm.rejected.tolist(),
        alpha=holm.alpha,
        holm_passed=source.holm.passed,
        passed=source.passed,
        failure_reasons=source.failure_reasons,
        strict_canonical=source.strict_canonical,
    )


def _prebinding_dependence_holm_source_identity(
    design: DependenceTaskView,
    production: DependenceTaskView,
    *,
    run_id: str,
    alpha: float,
) -> Mapping[str, object]:
    if (
        not is_registered_dependence_task_view(design)
        or not is_registered_dependence_task_view(production)
        or design.task_id != "design_dependence"
        or production.task_id != "production_dependence"
    ):
        raise ModelError("dependence Holm prebinding requires exact role views")
    if not isinstance(run_id, str) or _SHA256.fullmatch(run_id) is None:
        raise ModelError("dependence Holm run ID is not a SHA-256 fingerprint")
    if design.run_id != run_id or production.run_id != run_id:
        raise ModelError("dependence Holm prebinding sources use another run")
    if (
        design.binding_fingerprint == production.binding_fingerprint
        or design.strict_canonical != production.strict_canonical
        or design.source_execution_fingerprint
        == production.source_execution_fingerprint
    ):
        raise ModelError("dependence Holm prebinding parent lineage differs")
    cells = (*design.cells, *production.cells)
    if tuple(item.cell_id for item in cells) != DEPENDENCE_CELL_ORDER:
        raise ModelError("dependence Holm prebinding cell order differs")
    if design.strict_canonical and (
        alpha != 0.05 or any(item.histories != 10_000 for item in cells)
    ):
        raise ModelError("canonical dependence Holm prebinding contract differs")
    raw = [float(item.selected_result.p_value) for item in cells]
    holm = holm_correction(raw, alpha=alpha)
    reasons: list[str] = []
    for item, rejected in zip(cells, holm.rejected, strict=True):
        if not item.common_draw_reused:
            reasons.append(f"{item.cell_id}:common_purpose4_draw_not_reused")
        if not item.selected_result.envelope_passed:
            reasons.append(f"{item.cell_id}:selected_L_predictive_envelope_failed")
        if bool(rejected):
            reasons.append(f"{item.cell_id}:holm_rejected")
    parents = _dependence_parent_source_materializations((design, production))
    return _dependence_holm_identity_document(
        parent_materializations=parents,
        cells=[_dependence_task_cell_source_identity(item) for item in cells],
        raw_p_values=holm.raw_p_values.tolist(),
        adjusted_p_values=holm.adjusted_p_values.tolist(),
        rejected=holm.rejected.tolist(),
        alpha=holm.alpha,
        holm_passed=not bool(holm.rejected.any()),
        passed=not reasons,
        failure_reasons=reasons,
        strict_canonical=design.strict_canonical,
    )


def _block_source_identity(source: Any) -> Mapping[str, object]:
    identities = boundary_view_module._block_source_identities(
        source.source,
        task_id=source.task_id,
    )
    return _block_identity_document(
        role=source.role,
        selected_k=source.source.selected_k,
        identities=identities,
    )


def _block_identity_document(
    *,
    role: str,
    selected_k: int,
    identities: Mapping[str, object],
) -> Mapping[str, object]:
    return {"role": role, "selected_k": selected_k, **dict(identities)}


def _artifact_source_identity(source: Any) -> Mapping[str, object]:
    identities = boundary_view_module._artifact_source_identities(
        source.source,
        run_id=source.run_id,
        actual_bridge_parent_view_fingerprint=(
            source.actual_bridge_parent_view_fingerprint
        ),
    )
    return _artifact_identity_document(
        source=source.source,
        identities=identities,
    )


def _artifact_identity_document(
    *,
    source: ArtifactIntegrityDecisionEvidence,
    identities: Mapping[str, str],
) -> Mapping[str, object]:
    return {
        "design_role": source.design.role,
        "production_role": source.production.role,
        "design_selected_k": source.design.selected_k,
        "production_selected_k": source.production.selected_k,
        **dict(identities),
    }


def verify_recomputed_scientific_task_source(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    binding: ScientificTaskBinding,
    source: object,
) -> ScientificTaskSourceSlots:
    """Verify the mandatory source-reconstruction contract for resume.

    This function returns report-only slot identities.  It grants no launch,
    checkpoint, task-result, or generation authority.
    """

    if not isinstance(binding, ScientificTaskBinding):
        raise ModelError("source reconstruction requires a task binding")
    verify_scientific_task_binding(
        binding,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
    )
    _verify_source_coordinates(source, binding)
    return _verify_source_binding_slots(source, binding)


def publish_exact_next_scientific_task(
    *,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    binding: ScientificTaskBinding,
    source: object,
) -> ScientificTaskPublication:
    """Publish one terminal source only at the exact next G3 registry position.

    The caller cannot supply a pass Boolean, task ID, dependencies, selected
    state count, planned-look fingerprints, or checkpoint binding.  Every one
    of those values is derived from the sealed binding, typed source, and live
    durable stores.
    """

    _require_active_lease(run_store, lease)
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("scientific task publication requires a checkpoint store")
    if not isinstance(binding, ScientificTaskBinding):
        raise ModelError("scientific task publication requires a task binding")

    results = _exact_result_prefix(run_store, lease.run_id)
    if len(results) == len(G3_GATE_ORDER):
        raise IntegrityError("all canonical G3 tasks are already published")
    expected_task_id = G3_GATE_ORDER[len(results)]
    if binding.run_id != lease.run_id or binding.task_id != expected_task_id:
        raise IntegrityError(
            "scientific task binding is not for the exact next canonical task"
        )
    verify_recomputed_scientific_task_source(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        binding=binding,
        source=source,
    )

    spec = task_spec(expected_task_id)
    dependencies = tuple(
        checkpoint_store.load_from_task_result(
            run_store,
            run_id=lease.run_id,
            task_id=dependency,
        )
        for dependency in spec.dependencies
    )
    expected_dependencies = tuple(
        (dependency.task_id, dependency.fingerprint) for dependency in dependencies
    )
    if binding.dependency_checkpoint_fingerprints != expected_dependencies:
        raise IntegrityError("scientific task binding dependency lineage changed")

    all_looks = run_store.verify_planned_look_receipts(
        lease.run_id,
        require_terminal=False,
    )
    receipts = _owned_terminal_receipts(
        task_id=expected_task_id,
        all_receipts=all_looks,
    )
    proposed = canonical_planned_looks(
        expected_task_id,
        cast(CanonicalTaskSource, source),
    )
    _verify_receipts_match_source(
        expected_task_id,
        binding.run_id,
        receipts,
        proposed,
    )
    look_fingerprints = _look_fingerprint_document(receipts)
    if dict(binding.owned_planned_look_fingerprints) != {
        family_id: tuple(fingerprints)
        for family_id, fingerprints in look_fingerprints.items()
    }:
        raise IntegrityError("scientific task binding planned-look lineage changed")

    decision = derive_g3_gate_decision(expected_task_id, source)
    terminal_receipts = {receipt.family_id: receipt for receipt in receipts}.values()
    if decision.passed and any(
        receipt.decision != "pass" for receipt in terminal_receipts
    ):
        raise ModelError("a passing task requires passing terminal planned looks")
    selected_n_states = _selected_n_states(binding, decision)
    decision_document = _decision_document(decision, source)
    decision_fingerprint = _fingerprint(decision_document)
    serialized_binding = binding.as_dict()
    serialized_binding_sha256 = _fingerprint(serialized_binding)
    common_payload = {
        "binding_fingerprint": binding.fingerprint,
        "coordinator_schema_version": 1,
        "decision": decision_document,
        "decision_fingerprint": decision_fingerprint,
        "execution_mode": "canonical",
        "outcome_kind": "typed_decision",
        "planned_look_fingerprints": look_fingerprints,
        "scientific_task_binding": serialized_binding,
        "selected_n_states": selected_n_states,
        "serialized_binding_sha256": serialized_binding_sha256,
        "summary": _json_mapping(decision.summary, "G3 decision summary"),
    }
    if not decision.passed:
        result = run_store.write_task_result(
            lease,
            task_id=expected_task_id,
            status="failed",
            payload={**common_payload, "checkpoint_fingerprint": None},
        )
        verified = reverify_published_scientific_task(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=lease.run_id,
            task_id=expected_task_id,
            live_preflight=live_preflight,
        )
        if (
            verified.binding.fingerprint != binding.fingerprint
            or verified.checkpoint is not None
            or verified.task_result.fingerprint != result.fingerprint
        ):
            raise IntegrityError(
                "failed scientific task publication did not replay exactly"
            )
        return ScientificTaskPublication(
            binding,
            decision,
            None,
            result,
            serialized_binding_sha256,
        )

    checkpoint_binding = _checkpoint_binding(run_store, lease)
    state = {
        "binding_fingerprint": binding.fingerprint,
        "checkpoint_state_schema_version": SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION,
        "decision": decision_document,
        "decision_fingerprint": decision_fingerprint,
        "planned_look_fingerprints": look_fingerprints,
        "publication_schema_version": SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION,
        "scientific_task_binding": serialized_binding,
        "serialized_binding_sha256": serialized_binding_sha256,
    }
    checkpoint = checkpoint_store.create(
        task_id=expected_task_id,
        binding=checkpoint_binding,
        dependencies=dependencies,
        selected_n_states=selected_n_states,
        arrays={
            "decision_sha256": _digest_array(decision_fingerprint),
            "planned_looks_sha256": _digest_array(_fingerprint(look_fingerprints)),
            "scientific_task_binding_sha256": _digest_array(serialized_binding_sha256),
        },
        state=state,
    )
    result = run_store.write_task_result(
        lease,
        task_id=expected_task_id,
        status="passed",
        payload={**common_payload, "checkpoint_fingerprint": checkpoint.fingerprint},
    )
    verified = reverify_published_scientific_task(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=lease.run_id,
        task_id=expected_task_id,
        live_preflight=live_preflight,
    )
    if (
        verified.binding.fingerprint != binding.fingerprint
        or verified.checkpoint is None
        or verified.checkpoint.fingerprint != checkpoint.fingerprint
        or verified.task_result.fingerprint != result.fingerprint
    ):
        raise IntegrityError("fresh scientific task publication did not replay exactly")
    return ScientificTaskPublication(
        binding,
        decision,
        checkpoint,
        result,
        serialized_binding_sha256,
    )


def reverify_published_scientific_task(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    run_id: str,
    task_id: str,
    live_preflight: CanonicalG3Preflight,
) -> ReverifiedScientificTaskPublication:
    """Reload one terminal result and rederive its binding against live state."""

    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("task publication replay requires a CalibrationRunStore")
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("task publication replay requires a checkpoint store")
    task_spec(task_id)
    result = run_store.read_task_result(run_id, task_id)
    if (
        result.status not in {"passed", "failed"}
        or set(result.payload) != _TASK_RESULT_PAYLOAD_KEYS
    ):
        raise IntegrityError(
            "terminal G3 task result lacks binding-aware publication fields"
        )
    payload = result.payload
    serialized = payload["scientific_task_binding"]
    if not isinstance(serialized, Mapping):
        raise IntegrityError("scientific task result binding is missing")
    serialized_digest = _fingerprint(serialized)
    binding = load_scientific_task_binding(
        serialized,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
    )
    if (
        binding.run_id != run_id
        or binding.task_id != task_id
        or payload["binding_fingerprint"] != binding.fingerprint
        or payload["serialized_binding_sha256"] != serialized_digest
        or payload["execution_mode"] != "canonical"
        or payload["outcome_kind"] != "typed_decision"
        or payload["coordinator_schema_version"] != 1
    ):
        raise IntegrityError("scientific task binding/result identity differs")

    expected_looks = {
        family_id: list(fingerprints)
        for family_id, fingerprints in binding.owned_planned_look_fingerprints
    }
    if payload["planned_look_fingerprints"] != expected_looks:
        raise IntegrityError("scientific task binding planned-look identity differs")
    decision_fingerprint = payload["decision_fingerprint"]
    result_decision = payload["decision"]
    if (
        not isinstance(decision_fingerprint, str)
        or _SHA256.fullmatch(decision_fingerprint) is None
        or not isinstance(result_decision, Mapping)
        or set(result_decision) != _DECISION_KEYS
        or result_decision["gate_id"] != task_id
        or result_decision["passed"] is not (result.status == "passed")
        or not isinstance(result_decision["source_type"], str)
        or not result_decision["source_type"]
        or not isinstance(result_decision["evidence"], Mapping)
        or not result_decision["evidence"]
        or not isinstance(result_decision["summary"], Mapping)
        or not result_decision["summary"]
        or _fingerprint(result_decision) != decision_fingerprint
        or payload["summary"]
        != _json_mapping(result_decision["summary"], "stored result summary")
    ):
        raise IntegrityError("scientific task result decision identity is invalid")
    if result.status == "failed":
        if payload["checkpoint_fingerprint"] is not None:
            raise IntegrityError("failed scientific task cannot reference a checkpoint")
        return ReverifiedScientificTaskPublication(
            binding,
            None,
            result,
            serialized_digest,
        )

    checkpoint = checkpoint_store.load_from_task_result(
        run_store,
        run_id=run_id,
        task_id=task_id,
    )
    state = _json_mapping(checkpoint.state, "stored task checkpoint")
    if set(state) != _CHECKPOINT_STATE_KEYS:
        raise IntegrityError(
            "passed G3 checkpoint lacks binding-aware publication fields"
        )
    if (
        state["checkpoint_state_schema_version"]
        != SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION
        or state["publication_schema_version"]
        != SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION
    ):
        raise IntegrityError("scientific task publication schema is unsupported")
    stored_decision = state["decision"]
    if (
        not isinstance(stored_decision, Mapping)
        or set(stored_decision) != _DECISION_KEYS
    ):
        raise IntegrityError("scientific task checkpoint decision fields are invalid")
    if (
        stored_decision["gate_id"] != task_id
        or stored_decision["passed"] is not True
        or not isinstance(stored_decision["source_type"], str)
        or not stored_decision["source_type"]
        or not isinstance(stored_decision["evidence"], Mapping)
        or not stored_decision["evidence"]
        or not isinstance(stored_decision["summary"], Mapping)
        or not stored_decision["summary"]
    ):
        raise IntegrityError("scientific task checkpoint decision is invalid")
    if (
        state["scientific_task_binding"] != serialized
        or stored_decision != result_decision
        or state["binding_fingerprint"] != binding.fingerprint
        or state["serialized_binding_sha256"] != serialized_digest
        or state["planned_look_fingerprints"] != expected_looks
        or state["decision_fingerprint"] != decision_fingerprint
        or payload["checkpoint_fingerprint"] != checkpoint.fingerprint
        or checkpoint.dependency_fingerprints
        != binding.dependency_checkpoint_fingerprints
        or _fingerprint(stored_decision) != decision_fingerprint
        or payload["summary"]
        != _json_mapping(stored_decision["summary"], "stored decision summary")
        or payload["selected_n_states"] != checkpoint.selected_n_states
    ):
        raise IntegrityError(
            "scientific task decision/result/checkpoint identity differs"
        )
    if set(checkpoint.arrays) != _CHECKPOINT_ARRAY_KEYS:
        raise IntegrityError("scientific task checkpoint digest arrays are invalid")
    expected_arrays = {
        "decision_sha256": _digest_array(decision_fingerprint),
        "planned_looks_sha256": _digest_array(_fingerprint(expected_looks)),
        "scientific_task_binding_sha256": _digest_array(serialized_digest),
    }
    if any(
        not np.array_equal(checkpoint.arrays[name], expected)
        for name, expected in expected_arrays.items()
    ):
        raise IntegrityError("scientific task checkpoint digest arrays disagree")
    return ReverifiedScientificTaskPublication(
        binding,
        checkpoint,
        result,
        serialized_digest,
    )


def _require_active_lease(run_store: CalibrationRunStore, lease: LaunchLease) -> None:
    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("scientific task publication requires a CalibrationRunStore")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not run_store
        or not lease.active
    ):
        raise IntegrityError("scientific task publication requires its active lease")
    marker = run_store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != lease.preflight_fingerprint
        or marker.workers != lease.workers
    ):
        raise IntegrityError("scientific task publication lease identity changed")


def _exact_result_prefix(
    run_store: CalibrationRunStore,
    run_id: str,
) -> Mapping[str, G3TaskResult]:
    results = run_store.verify_task_results(run_id, require_complete=False)
    if tuple(results) != G3_GATE_ORDER[: len(results)]:
        raise IntegrityError("G3 task results are not the exact registry prefix")
    return MappingProxyType(results)


def _verify_source_coordinates(source: object, binding: ScientificTaskBinding) -> None:
    coordinates = (
        getattr(source, "run_id", None),
        getattr(source, "task_id", None),
        getattr(source, "binding_fingerprint", None),
    )
    expected = (binding.run_id, binding.task_id, binding.fingerprint)
    if coordinates != expected:
        raise ModelError(
            "canonical task source lacks matching sealed task-binding coordinates",
            details={
                "task_id": binding.task_id,
                "required_fields": ["run_id", "task_id", "binding_fingerprint"],
            },
        )


def _verify_source_binding_slots(
    source: object,
    binding: ScientificTaskBinding,
) -> ScientificTaskSourceSlots:
    derived = scientific_task_source_slots(source)
    if derived.task_id != binding.task_id:
        raise IntegrityError("scientific task source slot owner differs")
    if (
        derived.scientific_component_fingerprints
        != binding.scientific_component_fingerprints
        or derived.materialization_fingerprints != binding.materialization_fingerprints
        or derived.rng_fingerprints != binding.rng_fingerprints
    ):
        raise IntegrityError(
            "scientific task binding slots differ from the code-owned source"
        )
    return derived


def _composite_slot_fingerprint(
    label: str,
    values: Mapping[str, str],
) -> str:
    for value in values.values():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ModelError("scientific source exposes an invalid slot fingerprint")
    return _fingerprint(
        {
            "schema_id": "prpg-g3-composite-scientific-slot-v1",
            "label": label,
            "values": dict(values),
        }
    )


def _owned_terminal_receipts(
    *,
    task_id: str,
    all_receipts: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> tuple[PlannedLookReceipt, ...]:
    position = G3_GATE_ORDER.index(task_id)
    owner_by_family = {
        spec.family_id: spec.owner_task_id for spec in G3_PLANNED_LOOK_REGISTRY
    }
    for family_id in all_receipts:
        owner = owner_by_family[family_id]
        if G3_GATE_ORDER.index(owner) > position:
            raise IntegrityError("future-task planned looks precede exact publication")
    result: list[PlannedLookReceipt] = []
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if spec.owner_task_id != task_id:
            continue
        receipts = all_receipts.get(spec.family_id, ())
        if not receipts or receipts[-1].decision == "continue":
            raise IntegrityError("owned planned-look receipts are not terminal")
        result.extend(receipts)
    return tuple(result)


def _verify_receipts_match_source(
    task_id: str,
    run_id: str,
    receipts: Sequence[PlannedLookReceipt],
    proposed: Sequence[CanonicalG3PlannedLook],
) -> None:
    if len(receipts) != len(proposed):
        raise IntegrityError("owned planned-look receipts differ from typed source")
    for receipt, expected in zip(receipts, proposed, strict=True):
        if (
            receipt.run_id != run_id
            or receipt.family_id != expected.family_id
            or receipt.look_index != expected.look_index
            or receipt.sample_count != expected.sample_count
            or receipt.decision != expected.decision
            or dict(receipt.observed)
            != _json_mapping(expected.observed, "typed planned-look observation")
        ):
            raise IntegrityError(
                "owned planned-look receipt differs from typed source",
                details={"task_id": task_id},
            )


def _look_fingerprint_document(
    receipts: Sequence[PlannedLookReceipt],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for receipt in receipts:
        result.setdefault(receipt.family_id, []).append(receipt.fingerprint)
    return result


def _selected_n_states(
    binding: ScientificTaskBinding,
    decision: DerivedG3GateDecision,
) -> int | None:
    position = G3_GATE_ORDER.index(binding.task_id)
    reported = decision.summary.get("selected_k")
    if binding.task_id == _SELECTION_TASK:
        if not decision.passed:
            if binding.selected_n_states is not None:
                raise IntegrityError("failed selection binding prematurely freezes K")
            return None
        if (
            isinstance(reported, bool)
            or not isinstance(reported, int)
            or reported not in _ALLOWED_STATE_COUNTS
        ):
            raise ModelError("passing selection source lacks a valid selected K")
        return reported
    if position < _SELECTION_POSITION:
        if binding.selected_n_states is not None or reported is not None:
            raise ModelError("preselection task cannot bind selected K")
        return None
    selected = binding.selected_n_states
    if selected not in _ALLOWED_STATE_COUNTS:
        raise IntegrityError("post-selection binding lacks frozen selected K")
    if reported is not None and reported != selected:
        raise ModelError("task source differs from the frozen selected K")
    return selected


def _checkpoint_binding(
    run_store: CalibrationRunStore,
    lease: LaunchLease,
) -> CalibrationCheckpointBinding:
    reference = run_store.verify(lease.run_id)
    return CalibrationCheckpointBinding(
        run_id=lease.run_id,
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=(
            reference.identity.calibration_input_fingerprint
        ),
        preflight_fingerprint=lease.preflight_fingerprint,
    )


def _decision_document(
    decision: DerivedG3GateDecision,
    source: object,
) -> dict[str, Any]:
    return _json_mapping(
        {
            "evidence": dict(decision.evidence),
            "gate_id": decision.gate_id,
            "passed": decision.passed,
            "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
            "summary": dict(decision.summary),
        },
        "typed G3 decision",
    )


def _digest_array(fingerprint: str) -> np.ndarray[Any, Any]:
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        raise ModelError("scientific publication digest is not SHA-256")
    return np.frombuffer(bytes.fromhex(fingerprint), dtype=np.uint8).copy()


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


def _json_mapping(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    normalized = _json_value(value, path)
    if not isinstance(normalized, dict):  # pragma: no cover - type narrows locally.
        raise AssertionError("mapping normalization changed type")
    return normalized


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
