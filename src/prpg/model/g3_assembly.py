"""Closed typed derivation and assembly of all 26 Phase-3 gate records.

The legacy :func:`prpg.model.g3.gate_record` API accepts a Boolean and is
therefore deliberately report-only.  This module accepts only registered
result dataclasses, recomputes their decisions from counts/statistics and
invariants, seals the derived decision, and assembles the exact topological
26-record bundle.  Passing and failing bundles are both evidence-only; G5 is
the separate generation-qualification boundary.

Python cannot defend against a caller deliberately reaching into private
module internals.  "Sealed" here means that no public constructor or argument
accepts a pass Boolean for a derived decision.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np

from prpg.errors import ModelError
from prpg.model.adequacy_gate import (
    ADEQUACY_CELL_ORDER,
    AdequacyCellId,
    AdequacyCellTaskView,
    FourCellAdequacyHolmView,
    is_adequacy_cell_task_view,
    is_four_cell_adequacy_holm_view,
)
from prpg.model.block_execution import CANONICAL_HISTORY_COUNT
from prpg.model.block_execution_registered import RegisteredBlockCalibration
from prpg.model.bridge import (
    TIER_B_BATCH_SIZE,
    TIER_B_INTERVAL_ERROR,
    TIER_B_MAXIMUM_PAIRS,
    TIER_B_TAIL_PROBABILITY,
)
from prpg.model.bridge_driver import StagedBridgeTaskEvidence
from prpg.model.candidate_gate import CANONICAL_CANDIDATES
from prpg.model.dependence_gate_execution import (
    DEPENDENCE_CELL_ORDER,
    DependenceTaskCellEvidence,
    DependenceTaskView,
    FourCellDependenceHolmView,
    FragmentationTaskCellEvidence,
    is_registered_dependence_task_view,
    is_registered_four_cell_dependence_holm_view,
)
from prpg.model.g3 import G3EvidenceBundle, build_g3_evidence_bundle, gate_record
from prpg.model.g3_boundary_views import (
    BRIDGE_TASK_IDS,
    ArtifactIntegrityTaskView,
    BlockCalibrationTaskView,
    BridgeDriverTaskView,
    InputIntegrityTaskView,
    is_artifact_integrity_task_view,
    is_block_calibration_task_view,
    is_bridge_driver_task_view,
    is_input_integrity_task_view,
)
from prpg.model.g3_boundary_views import (
    ArtifactIntegrityDecisionEvidence as ArtifactIntegrityDecisionEvidence,
)
from prpg.model.g3_boundary_views import (
    BlockCalibrationGateEvidence as BlockCalibrationGateEvidence,
)
from prpg.model.g3_boundary_views import (
    BridgeDriverTaskViews as BridgeDriverTaskViews,
)
from prpg.model.g3_boundary_views import (
    InputIntegrityDecisionEvidence as InputIntegrityDecisionEvidence,
)
from prpg.model.g3_boundary_views import (
    bridge_driver_task_views as bridge_driver_task_views,
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
from prpg.model.g3_registry import G3_GATE_ORDER, G3_TASK_BY_ID
from prpg.model.inference import HolmResult, clopper_pearson_interval, holm_correction
from prpg.model.kernel_execution import (
    CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
    CANONICAL_PREFIX_WINDOW_COUNTS,
    KernelCalibrationExecution,
)
from prpg.model.refit_adequacy_execution import CANONICAL_MINIMUM_SUCCESSES
from prpg.model.regime_policy import (
    effective_identification_passed,
    effective_transition_report_passed,
    effective_transition_support_passed,
)
from prpg.model.scientific_artifact import CANONICAL_SCIENTIFIC_VERSION
from prpg.model.selection import (
    ORDINARY_SELECTION_POLICY,
    OWNER_FIXED_K4_SELECTION_POLICY,
    OWNER_FIXED_N_STATES,
    ROLLING_BOOTSTRAP_REPLICATES,
    ROLLING_FOLD_COUNT,
    ROLLING_SCORE_COUNT,
)
from prpg.model.transition_bootstrap import (
    MACRO_BOOTSTRAP_ATTEMPTS,
    MINIMUM_SUCCESSFUL_REFITS,
    TransitionBootstrapAggregate,
)

ArtifactRole: TypeAlias = Literal["design", "production"]
BridgeDGP: TypeAlias = Literal["model", "nonparametric"]
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_DECISION_SEAL = object()
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True, init=False)
class DerivedG3GateDecision:
    """Opaque decision emitted only by :func:`derive_g3_gate_decision`."""

    gate_id: str
    passed: bool
    evidence: Mapping[str, Any]
    summary: Mapping[str, Any]
    _source: object = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class G3TaskFailure:
    """Bounded task failure supplied only to lower, never grant, authority."""

    task_id: str
    failure_type: str
    message: str


def derive_g3_gate_decision(gate_id: str, result: object) -> DerivedG3GateDecision:
    """Derive one sealed decision from the registered typed result for a task."""

    if gate_id not in G3_TASK_BY_ID:
        raise ModelError("G3 decision task is not in the closed registry")
    passed, evidence, summary = _evaluate(gate_id, result)
    if not isinstance(passed, bool):  # pragma: no cover - evaluator invariant
        raise AssertionError("typed G3 evaluator did not return a Boolean")
    decision = object.__new__(DerivedG3GateDecision)
    object.__setattr__(decision, "gate_id", gate_id)
    object.__setattr__(decision, "passed", passed)
    object.__setattr__(decision, "evidence", MappingProxyType(dict(evidence)))
    object.__setattr__(decision, "summary", MappingProxyType(dict(summary)))
    object.__setattr__(decision, "_source", result)
    object.__setattr__(decision, "_seal", _DECISION_SEAL)
    return decision


def assemble_typed_g3_evidence_bundle(
    *,
    config_fingerprint: str,
    processed_data_fingerprint: str,
    calibration_input_fingerprint: str,
    decisions: Sequence[DerivedG3GateDecision],
    failures: Sequence[G3TaskFailure] = (),
) -> G3EvidenceBundle:
    """Assemble all 26 records, failing closed for missing/blocked work.

    Decisions may arrive in any order but must be unique and sealed.  Records
    are emitted in the registry's topological order.  A failed dependency
    blocks a supplied downstream decision as well as an absent one.  Artifact
    fingerprints are populated only from a passing typed integrity pair.
    """

    for value, label in (
        (config_fingerprint, "config"),
        (processed_data_fingerprint, "processed data"),
        (calibration_input_fingerprint, "calibration input"),
    ):
        _fingerprint(value, label)
    by_id: dict[str, DerivedG3GateDecision] = {}
    for decision in decisions:
        if (
            not isinstance(decision, DerivedG3GateDecision)
            or decision._seal is not _DECISION_SEAL
        ):
            raise ModelError("G3 assembler accepts only sealed typed decisions")
        if decision.gate_id in by_id:
            raise ModelError("G3 typed decisions contain a duplicate task")
        if decision.gate_id not in G3_TASK_BY_ID:
            raise ModelError("G3 typed decision contains an unregistered task")
        by_id[decision.gate_id] = decision
    failure_by_id: dict[str, G3TaskFailure] = {}
    for failure in failures:
        _validate_failure(failure)
        if failure.task_id in failure_by_id or failure.task_id in by_id:
            raise ModelError("G3 task has duplicate decision/failure evidence")
        failure_by_id[failure.task_id] = failure

    cross_error = _cross_binding_error(
        by_id,
        config_fingerprint=config_fingerprint,
        processed_data_fingerprint=processed_data_fingerprint,
        calibration_input_fingerprint=calibration_input_fingerprint,
    )
    record_passed: dict[str, bool] = {}
    records = []
    for gate_id in G3_GATE_ORDER:
        blocked_by = tuple(
            dependency
            for dependency in G3_TASK_BY_ID[gate_id].dependencies
            if not record_passed.get(dependency, False)
        )
        task_decision = by_id.get(gate_id)
        task_failure = failure_by_id.get(gate_id)
        if blocked_by:
            passed = False
            evidence: Mapping[str, Any] = {
                "availability": "blocked",
                "blocked_by": list(blocked_by),
                "supplied_typed_result": task_decision is not None,
            }
            summary: Mapping[str, Any] = {
                "status": "blocked",
                "blocked_by": list(blocked_by),
            }
        elif task_failure is not None:
            passed = False
            evidence = {
                "availability": "failed",
                "failure_type": task_failure.failure_type,
                "message": task_failure.message,
            }
            summary = {
                "status": "failed",
                "failure_type": task_failure.failure_type,
            }
        elif task_decision is None:
            passed = False
            evidence = {"availability": "missing_typed_result"}
            summary = {"status": "missing"}
        else:
            passed = task_decision.passed
            evidence = task_decision.evidence
            summary = task_decision.summary
        if gate_id == "artifact_integrity" and cross_error is not None:
            passed = False
            evidence = {
                **dict(evidence),
                "cross_binding_failure": cross_error,
            }
            summary = {**dict(summary), "cross_binding_failure": cross_error}
        records.append(
            gate_record(
                gate_id,
                passed=passed,
                evidence=evidence,
                summary=summary,
            )
        )
        record_passed[gate_id] = passed

    artifact_source = by_id.get("artifact_integrity")
    design_fingerprint: str | None = None
    production_fingerprint: str | None = None
    if (
        artifact_source is not None
        and artifact_source.passed
        and cross_error is None
        and isinstance(artifact_source._source, ArtifactIntegrityTaskView)
    ):
        pair = artifact_source._source.source
        design_fingerprint = pair.design.reference.fingerprint
        production_fingerprint = pair.production.reference.fingerprint
    return build_g3_evidence_bundle(
        config_fingerprint=config_fingerprint,
        processed_data_fingerprint=processed_data_fingerprint,
        calibration_input_fingerprint=calibration_input_fingerprint,
        design_artifact_fingerprint=design_fingerprint,
        production_artifact_fingerprint=production_fingerprint,
        records=tuple(records),
    )


def _evaluate(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if gate_id == "input_integrity":
        return _input_integrity(result)
    if gate_id in {"design_candidate_viability", "design_predictive_selection"}:
        return _candidate(gate_id, result)
    if gate_id == "sealed_holdout_veto":
        return _holdout(result)
    if gate_id in {"design_macro_stability", "production_macro_stability"}:
        return _macro_stability(gate_id, result)
    adequacy_ids = {
        "design_latent_correctness",
        "design_refitted_adequacy",
        "production_latent_correctness",
        "production_refitted_adequacy",
        "four_cell_adequacy_holm",
    }
    if gate_id in adequacy_ids:
        return _adequacy(gate_id, result)
    if gate_id in {
        "design_block_calibration",
        "production_block_calibration",
    }:
        return _block(gate_id, result)
    if gate_id == "production_fit_same_k":
        return _production_fit(result)
    if gate_id in {
        "design_fragmentation",
        "design_dependence",
        "production_fragmentation",
        "production_dependence",
        "four_cell_dependence_holm",
    }:
        return _dependence(gate_id, result)
    if gate_id == "bridge_resource_preflight":
        return _bridge_preflight(result)
    if gate_id in {
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
    }:
        return _tier_a(gate_id, result)
    if gate_id in {
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    }:
        return _tier_b(gate_id, result)
    if gate_id == "actual_artifact_bridge":
        return _actual_bridge(result)
    if gate_id == "artifact_integrity":
        return _artifact_integrity(result)
    raise ModelError("G3 task has no registered typed evaluator")  # pragma: no cover


def _input_integrity(result: object) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not is_input_integrity_task_view(result) or result.task_id != "input_integrity":
        raise ModelError("input_integrity requires its sealed task view")
    source = result.source
    preflight = source.launch_preflight
    binding = source.binding
    passed = (
        _FINGERPRINT.fullmatch(preflight.resource_estimate_fingerprint) is not None
        and binding.preflight_fingerprint == preflight.fingerprint
        and binding.config_fingerprint == preflight.config_fingerprint
        and binding.processed_data_fingerprint == preflight.processed_data_fingerprint
        and binding.calibration_input_fingerprint
        == preflight.calibration_input_fingerprint
        and binding.task_registry_fingerprint == preflight.task_registry_fingerprint
        and binding.planned_look_registry_fingerprint
        == preflight.planned_look_registry_fingerprint
        and preflight.workers == 9
        and bool(preflight.checks)
    )
    evidence = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "source_fingerprint": result.source_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "preflight_fingerprint": preflight.fingerprint,
        "live_preflight_fingerprint": result.live_preflight_fingerprint,
        "live_preflight_compatible": True,
        "config_fingerprint": preflight.config_fingerprint,
        "processed_data_fingerprint": preflight.processed_data_fingerprint,
        "calibration_input_fingerprint": preflight.calibration_input_fingerprint,
        "source_code_fingerprint": preflight.source_code_fingerprint,
        "dependency_lock_fingerprint": preflight.dependency_lock_fingerprint,
        "workers": preflight.workers,
        "check_count": len(preflight.checks),
    }
    return passed, evidence, {"status": "passed" if passed else "failed"}


def _candidate(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    evidence: dict[str, Any]
    if gate_id == "design_candidate_viability":
        if (
            not is_candidate_viability_task_view(result)
            or result.task_id != gate_id
            or result.role != "design"
        ):
            raise ModelError(
                "design_candidate_viability requires its sealed pre-BIC task view"
            )
        row_ks = tuple(row.n_states for row in result.viability_rows)
        if row_ks != CANONICAL_CANDIDATES:
            raise ModelError("candidate viability view lacks exact K=2..5 rows")
        admissible = tuple(
            row.n_states for row in result.viability_rows if row.viability.admissible
        )
        passed = OWNER_FIXED_N_STATES in admissible
        if result.passed is not passed or result.admissible_n_states != admissible:
            raise ModelError("candidate viability pass flag is inconsistent")
        evidence = {
            "task_id": gate_id,
            "run_id": result.run_id,
            "binding_fingerprint": result.binding_fingerprint,
            "view_fingerprint": result.view_fingerprint,
            "source_content_fingerprint": result.source_content_fingerprint,
            "viability_result_fingerprint": result.viability_result_fingerprint,
            "fit_set_fingerprint": result.fit_set_fingerprint,
            "rng_execution_fingerprint": result.rng_execution_fingerprint,
            "viability_fingerprint": result.viability_fingerprint,
            "candidate_ks": list(CANONICAL_CANDIDATES),
            "admissible_ks": list(admissible),
        }
        return passed, evidence, {"admissible_count": len(admissible)}
    if (
        not is_candidate_selection_task_view(result)
        or result.task_id != gate_id
        or result.role != "design"
    ):
        raise ModelError(
            "design_predictive_selection requires its sealed staged task view"
        )
    selection = result.selection
    scientific_version = result._viability_parent.scientific_version
    ranking_ks = tuple(row.n_states for row in selection.candidates)
    if ranking_ks != CANONICAL_CANDIDATES:
        raise ModelError("candidate selection rankings are not in exact K order")
    selected_rows = tuple(row for row in selection.candidates if row.selected)
    common_geometry = (
        selection.rolling_fold_count == ROLLING_FOLD_COUNT
        and selection.rolling_score_count == ROLLING_SCORE_COUNT
    )
    if (
        scientific_version == CANONICAL_SCIENTIFIC_VERSION
        and selection.selection_policy == ORDINARY_SELECTION_POLICY
    ):
        raise ModelError(
            "canonical scientific version 11 forbids ordinary cross-K selection"
        )
    if selection.selection_policy == ORDINARY_SELECTION_POLICY:
        # Compatibility for pre-v11 evidence replay only.  This assembler is
        # non-authorizing, and the canonical v11 path is fenced above.
        eligible_count = sum(row.predictively_eligible for row in selection.candidates)
        bootstrap_valid = (
            eligible_count == 1
            and selection.rolling_max_z is None
            and selection.bootstrap_replicates == 0
        ) or (
            eligible_count > 1
            and selection.rolling_max_z is not None
            and selection.bootstrap_replicates == ROLLING_BOOTSTRAP_REPLICATES
        )
        passed = (
            selection.selected_n_states is not None
            and len(selected_rows) == 1
            and selected_rows[0].n_states == selection.selected_n_states
            and selected_rows[0].viability.admissible
            and selected_rows[0].predictively_near_best
            and selected_rows[0].inside_bic_guard
            and common_geometry
            and bootstrap_valid
        )
    elif selection.selection_policy == OWNER_FIXED_K4_SELECTION_POLICY:
        fixed_rows = tuple(
            row for row in selection.candidates if row.n_states == OWNER_FIXED_N_STATES
        )
        if len(fixed_rows) != 1:
            raise ModelError("owner-fixed K4 selection lacks its exact K=4 row")
        fixed_row = fixed_rows[0]
        fixed_report_valid = (
            selection.rolling_max_z is None
            and selection.bootstrap_replicates == 0
            and selection.max_z_critical_value == 0.0
            and all(
                not row.selected
                for row in selection.candidates
                if row.n_states != OWNER_FIXED_N_STATES
            )
            and (
                (
                    fixed_row.viability.admissible
                    and fixed_row.selected
                    and fixed_row.disposition == "selected"
                    and selection.selected_n_states == OWNER_FIXED_N_STATES
                    and selection.failure_reason is None
                )
                or (
                    not fixed_row.viability.admissible
                    and not fixed_row.selected
                    and selection.selected_n_states is None
                    and selection.failure_reason == "owner_fixed_k4_not_admissible"
                )
            )
        )
        passed = (
            fixed_report_valid
            and fixed_row.viability.admissible
            and len(selected_rows) == 1
            and selected_rows[0] is fixed_row
            and common_geometry
        )
    else:
        raise ModelError("candidate selection policy is not recognized")
    if (
        selection.passed is not passed
        or result.passed is not passed
        or result.selected_n_states != selection.selected_n_states
    ):
        raise ModelError("candidate selection pass flag is inconsistent")
    evidence = {
        "task_id": gate_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "source_content_fingerprint": result.source_content_fingerprint,
        "viability_result_fingerprint": result.viability_result_fingerprint,
        "fit_set_fingerprint": result.fit_set_fingerprint,
        "rolling_fingerprints": list(result.rolling_fingerprints),
        "bootstrap_fingerprint": result.bootstrap_fingerprint,
        "viability_view_fingerprint": result.viability_view_fingerprint,
        "selection_fingerprint": result.selection_fingerprint,
        "scientific_version": scientific_version,
        "selection_policy": selection.selection_policy,
        "candidate_ks": list(CANONICAL_CANDIDATES),
        "selected_k": result.selected_n_states,
    }
    return passed, evidence, {"selected_k": result.selected_n_states}


def _holdout(result: object) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if (
        not is_sealed_holdout_veto_view(result)
        or result.task_id != "sealed_holdout_veto"
        or result.role != "design"
    ):
        raise ModelError("sealed_holdout_veto requires its sealed executed task view")
    source = result.evidence
    decision = source.decision
    hmm = np.asarray(source.hmm_log_scores, dtype=np.float64)
    gaussian = np.asarray(source.benchmark_scores.gaussian, dtype=np.float64)
    ridge = np.asarray(source.benchmark_scores.ridge_var1, dtype=np.float64)
    if (
        source.sequence_lengths != (17, 7)
        or hmm.shape != (24,)
        or gaussian.shape != (24,)
        or ridge.shape != (24,)
        or not all(np.isfinite(value).all() for value in (hmm, gaussian, ridge))
    ):
        raise ModelError("sealed holdout evidence geometry is noncanonical")
    hmm_mean = float(np.mean(hmm))
    gaussian_mean = float(np.mean(gaussian))
    ridge_mean = float(np.mean(ridge))
    passed = hmm_mean - gaussian_mean >= 0.0 and hmm_mean - ridge_mean >= 0.0
    observed = (
        decision.hmm_mean_log_score,
        decision.gaussian_mean_log_score,
        decision.ridge_var1_mean_log_score,
        decision.hmm_minus_gaussian,
        decision.hmm_minus_ridge_var1,
    )
    expected = (
        hmm_mean,
        gaussian_mean,
        ridge_mean,
        hmm_mean - gaussian_mean,
        hmm_mean - ridge_mean,
    )
    if (
        not np.allclose(observed, expected, rtol=0.0, atol=1e-12)
        or decision.passed != passed
    ):
        raise ModelError("sealed holdout decision is inconsistent with score arrays")
    evidence = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "selection_view_fingerprint": result.selection_view_fingerprint,
        "selected_model_fingerprint": result.selected_model_fingerprint,
        "design_slice_fingerprint": result.design_slice_fingerprint,
        "holdout_slice_fingerprint": result.holdout_slice_fingerprint,
        "evidence_fingerprint": result.evidence_fingerprint,
        "sequence_lengths": [17, 7],
        "observations": 24,
        "hmm_minus_gaussian": expected[3],
        "hmm_minus_ridge_var1": expected[4],
    }
    return passed, evidence, {"minimum_score_advantage": min(expected[3:])}


def _macro_stability(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not is_macro_stability_task_view(result) or result.task_id != gate_id:
        raise ModelError(f"{gate_id} requires its sealed macro-stability task view")
    expected_role: ArtifactRole = (
        "design" if gate_id.startswith("design_") else "production"
    )
    source = result.result
    if result.role != expected_role or source.role != expected_role:
        raise ModelError("macro-stability role binding is invalid")
    if (
        result.selected_k not in CANONICAL_CANDIDATES
        or source.n_states != result.selected_k
    ):
        raise ModelError("macro-stability selected K binding is invalid")
    if len(source.attempts) != MACRO_BOOTSTRAP_ATTEMPTS or tuple(
        attempt.attempt for attempt in source.attempts
    ) != tuple(range(MACRO_BOOTSTRAP_ATTEMPTS)):
        raise ModelError("macro stability requires exact attempt IDs 0..99")
    summary = source.summary
    median_jaccard = np.asarray(summary.median_jaccard_by_state)
    tenth_jaccard = np.asarray(summary.tenth_percentile_jaccard_by_state)
    passed = (
        summary.successful_attempts >= MINIMUM_SUCCESSFUL_REFITS
        and summary.median_ari >= 0.80
        and summary.tenth_percentile_ari >= 0.50
        and median_jaccard.shape == (result.selected_k,)
        and tenth_jaccard.shape == (result.selected_k,)
        and bool((median_jaccard >= 0.70).all())
        and bool((tenth_jaccard >= 0.50).all())
    )
    if summary.passed != passed:
        raise ModelError("macro-stability pass flag is inconsistent")
    values = {
        "task_id": gate_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "parent_view_fingerprint": result.parent_view_fingerprint,
        "parent_model_fingerprint": result.parent_model_fingerprint,
        "source_slice_fingerprint": result.source_slice_fingerprint,
        "macro_input_fingerprint": result.macro_input_fingerprint,
        "rng_execution_fingerprint": result.rng_execution_fingerprint,
        "evidence_fingerprint": result.evidence_fingerprint,
        "role": expected_role,
        "selected_k": result.selected_k,
        "attempts": len(source.attempts),
        "successful_attempts": summary.successful_attempts,
        "median_ari": summary.median_ari,
        "tenth_percentile_ari": summary.tenth_percentile_ari,
        "median_jaccard": median_jaccard.tolist(),
        "tenth_percentile_jaccard": tenth_jaccard.tolist(),
    }
    return (
        passed,
        values,
        {
            "selected_k": result.selected_k,
            "successful_attempts": summary.successful_attempts,
        },
    )


def _adequacy(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    task_to_cell: dict[str, AdequacyCellId] = {
        "design_latent_correctness": "design_latent",
        "production_latent_correctness": "production_latent",
        "design_refitted_adequacy": "design_refitted_adequacy",
        "production_refitted_adequacy": "production_refitted_adequacy",
    }
    if gate_id != "four_cell_adequacy_holm":
        if not is_adequacy_cell_task_view(result) or result.task_id != gate_id:
            raise ModelError(f"{gate_id} requires its sealed adequacy cell task view")
        cell = result.cell
        if cell.cell_id != task_to_cell[gate_id] or not result.strict_canonical:
            raise ModelError("adequacy task view has the wrong cell or mode")
        latent = "latent" in cell.cell_id and "refitted" not in cell.cell_id
        expected_repetitions = 10_000 if latent else 1_000
        minimum_successes = 10_000 if latent else CANONICAL_MINIMUM_SUCCESSES
        passed = (
            cell.ready
            and result.canonical_count_ready
            and cell.repetitions == expected_repetitions
            and cell.successful_repetitions >= minimum_successes
            and cell.p_value is not None
            and math.isfinite(cell.p_value)
            and 0.0 <= cell.p_value <= 1.0
            and cell.failure_reason is None
        )
        if result.ready is not cell.ready or result.passed is not passed:
            raise ModelError("adequacy task-view decision is inconsistent")
        evidence = {
            "task_id": gate_id,
            "run_id": result.run_id,
            "binding_fingerprint": result.binding_fingerprint,
            "content_fingerprint": result.content_fingerprint,
            "parent_model_fingerprint": result.parent_model_fingerprint,
            "scientific_input_fingerprint": result.scientific_input_fingerprint,
            "rng_execution_fingerprint": result.rng_execution_fingerprint,
            "source_evidence_fingerprint": result.source_evidence_fingerprint,
            "cell_id": cell.cell_id,
            "ready": cell.ready,
            "canonical_count_ready": result.canonical_count_ready,
            "p_value": cell.p_value,
            "repetitions": cell.repetitions,
            "successful_repetitions": cell.successful_repetitions,
            "failure_reason": cell.failure_reason,
        }
        return (
            passed,
            evidence,
            {
                "cell_id": cell.cell_id,
                "p_value": cell.p_value,
            },
        )
    if (
        not is_four_cell_adequacy_holm_view(result)
        or result.task_id != gate_id
        or not result.strict_canonical
        or result.cell_order != ADEQUACY_CELL_ORDER
    ):
        raise ModelError(
            "four_cell_adequacy_holm requires its sealed four-cell Holm view"
        )
    p_values = tuple(cell.p_value for cell in result.cells)
    ready = all(source.passed for source in result.source_views) and all(
        value is not None for value in p_values
    )
    passed = False
    if ready:
        numeric = np.asarray(p_values, dtype=np.float64)
        expected_holm = holm_correction(numeric, alpha=result.alpha)
        if result.holm is None or not _same_holm(result.holm, expected_holm):
            raise ModelError("adequacy Holm result is inconsistent")
        passed = not bool(expected_holm.rejected.any())
    elif result.holm is not None:
        raise ModelError("incomplete adequacy gate must withhold Holm result")
    if result.passed != passed:
        raise ModelError("adequacy aggregate pass flag is inconsistent")
    aggregate_evidence: dict[str, Any] = {
        "task_id": gate_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "content_fingerprint": result.content_fingerprint,
        "source_binding_fingerprints": [
            source.binding_fingerprint for source in result.source_views
        ],
        "source_content_fingerprints": list(result.source_content_fingerprints),
        "cell_order": list(ADEQUACY_CELL_ORDER),
        "ready": ready,
        "p_values": list(p_values),
        "rejected": None if result.holm is None else result.holm.rejected.tolist(),
        "failure_reasons": list(result.failure_reasons),
    }
    return passed, aggregate_evidence, {"ready": ready, "holm_passed": passed}


def _block(gate_id: str, result: object) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not is_block_calibration_task_view(result) or result.task_id != gate_id:
        raise ModelError(f"{gate_id} requires its sealed block-calibration task view")
    view = result
    source = view.source
    expected_role: ArtifactRole = (
        "design" if gate_id.startswith("design_") else "production"
    )
    if source.role != expected_role or source.selected_k != 4:
        raise ModelError("block-calibration role/K binding is invalid")
    cells = (("monthly", source.monthly), ("daily", source.daily))
    expected_monthly_segments = (193,) if expected_role == "design" else (210, 7)
    expected_daily_segments = (4_049,) if expected_role == "design" else (4_404, 143)
    block_ready = True
    fragmentation_passed = True
    block_lengths: dict[str, int] = {}
    for frequency, registered in cells:
        if not isinstance(registered, RegisteredBlockCalibration):
            raise ModelError("block calibration requires registered execution results")
        execution = registered.execution
        if (
            execution.geometry.artifact != expected_role
            or execution.geometry.frequency != frequency
            or not execution.geometry.strict_canonical
            or execution.geometry.monthly_segment_lengths != expected_monthly_segments
            or execution.geometry.target_segment_lengths
            != (
                expected_monthly_segments
                if frequency == "monthly"
                else expected_daily_segments
            )
            or execution.histories != CANONICAL_HISTORY_COUNT
            or registered.history_stream_sets != CANONICAL_HISTORY_COUNT
            or execution.policy_selection.frequency != frequency
            or execution.policy_selection.selected_length <= 0
        ):
            block_ready = False
        block_lengths[frequency] = execution.policy_selection.selected_length
        gates = execution.selected_fragmentation_gates
        if tuple(gate.state for gate in gates) != tuple(
            range(source.selected_k)
        ) or any(
            gate.intended_length != execution.policy_selection.selected_length
            or gate.reasons
            for gate in gates
        ):
            fragmentation_passed = False
    kernels = (("monthly", source.monthly_kernel), ("daily", source.daily_kernel))
    kernel_passed = True
    kernel_windows: dict[str, int] = {}
    model_fingerprints: set[str] = set()
    for frequency, kernel in kernels:
        if not isinstance(kernel, KernelCalibrationExecution):
            raise ModelError("block gate requires registered kernel executions")
        derived_kernel_pass = (
            kernel.artifact == expected_role
            and kernel.frequency == frequency
            and kernel.n_states == source.selected_k
            and kernel.selected_length == block_lengths[frequency]
            and kernel.strict_canonical
            and kernel.planned_looks == CANONICAL_PREFIX_WINDOW_COUNTS
            and bool(kernel.completed_looks)
            and kernel.stopped_windows == kernel.completed_looks[-1].windows
            and kernel.stopped_windows in CANONICAL_PREFIX_WINDOW_COUNTS
            and kernel.precision_passed
            and kernel.verification_passed
            and kernel.offset_grid is not None
            and kernel.verification is not None
            and kernel.verification.passed
            and float(
                np.max(kernel.final_precision.annualized_simultaneous_half_widths)
            )
            <= CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
        )
        if kernel.passed != (kernel.precision_passed and kernel.verification_passed):
            raise ModelError("kernel calibration pass flag is inconsistent")
        kernel_passed &= derived_kernel_pass
        kernel_windows[frequency] = kernel.stopped_windows
        model_fingerprints.add(kernel.model_artifact_fingerprint)
    transition = source.transition
    if not isinstance(transition, TransitionBootstrapAggregate):
        raise ModelError("block gate requires transition-bootstrap aggregate")
    transition_passed = (
        transition.n_states == source.selected_k
        and transition.attempts == MACRO_BOOTSTRAP_ATTEMPTS
        and transition.successful_refits >= MINIMUM_SUCCESSFUL_REFITS
        and transition.failed_refits
        == transition.attempts - transition.successful_refits
        and transition.edge_frequency_denominator == MACRO_BOOTSTRAP_ATTEMPTS
        and len(transition.attempt_audits) == MACRO_BOOTSTRAP_ATTEMPTS
        and effective_transition_report_passed(
            transition.historical_transition_support,
            n_states=transition.n_states,
            disposition=(
                None
                if getattr(transition, "_main_fit", None) is None
                else getattr(
                    transition._main_fit,
                    "owner_fixed_k4_disposition",
                    None,
                )
            ),
        )
    )
    if len(model_fingerprints) != 1 or any(
        _FINGERPRINT.fullmatch(value) is None for value in model_fingerprints
    ):
        kernel_passed = False
    recurring_history_support_passed = bool(
        expected_role == "design"
        or (
            view.recurring_history_support_fingerprint is not None
            and len(view.monthly_source_episode_counts) == 4
            and len(view.daily_source_episode_counts) == 4
            and min(view.monthly_source_episode_counts) >= 2
            and min(view.daily_source_episode_counts) >= 2
        )
    )
    passed = (
        block_ready
        and kernel_passed
        and transition_passed
        and recurring_history_support_passed
    )
    evidence = {
        "task_id": view.task_id,
        "run_id": view.run_id,
        "binding_fingerprint": view.binding_fingerprint,
        "view_fingerprint": view.view_fingerprint,
        "source_fingerprint": view.source_fingerprint,
        "monthly_calibration_fingerprint": view.monthly_calibration_fingerprint,
        "daily_calibration_fingerprint": view.daily_calibration_fingerprint,
        "monthly_kernel_fingerprint": view.monthly_kernel_fingerprint,
        "daily_kernel_fingerprint": view.daily_kernel_fingerprint,
        "transition_fingerprint": view.transition_fingerprint,
        "role": expected_role,
        "selected_k": source.selected_k,
        "histories_per_frequency": CANONICAL_HISTORY_COUNT,
        "block_lengths": block_lengths,
        "kernel_windows": kernel_windows,
        "block_ready": block_ready,
        "kernel_passed": kernel_passed,
        "transition_passed": transition_passed,
        "monthly_source_episode_counts": list(view.monthly_source_episode_counts),
        "daily_source_episode_counts": list(view.daily_source_episode_counts),
        "recurring_history_support_fingerprint": (
            view.recurring_history_support_fingerprint
        ),
        "recurring_history_support_passed": recurring_history_support_passed,
        "fragmentation_passed": fragmentation_passed,
        "core_model_fingerprint": (
            next(iter(model_fingerprints)) if len(model_fingerprints) == 1 else None
        ),
    }
    return (
        passed,
        evidence,
        {
            "selected_k": source.selected_k,
            "monthly_block_length": block_lengths["monthly"],
            "daily_block_length": block_lengths["daily"],
            "core_model_fingerprint": (
                next(iter(model_fingerprints)) if len(model_fingerprints) == 1 else None
            ),
        },
    )


def _production_fit(result: object) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if (
        not is_production_fit_same_k_view(result)
        or result.task_id != "production_fit_same_k"
        or result.role != "production"
    ):
        raise ModelError(
            "production_fit_same_k requires its sealed fixed-K production view"
        )
    fit = result.production_fit
    passed = (
        result.design_selected_k == 4
        and fit.n_states == 4
        and fit.n_observations == 217
        and fit.n_features == 4
        and fit.lengths == (210, 7)
        and effective_identification_passed(fit)
        and effective_transition_support_passed(fit)
        and fit.chain_diagnostics.irreducible
        and fit.chain_diagnostics.aperiodic
        and fit.chain_diagnostics.unique_stationary_distribution
        and not fit.chain_diagnostics.near_absorbing_states
    )
    evidence = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "selection_view_fingerprint": result.selection_view_fingerprint,
        "design_model_fingerprint": result.design_model_fingerprint,
        "production_feature_fingerprint": result.production_feature_fingerprint,
        "rng_execution_fingerprint": result.rng_execution_fingerprint,
        "restart_seed_fingerprint": result.restart_seed_fingerprint,
        "production_fit_fingerprint": result.production_fit_fingerprint,
        "registered_execution_fingerprint": (result.registered_execution_fingerprint),
        "design_selected_k": result.design_selected_k,
        "production_k": fit.n_states,
        "observations": fit.n_observations,
        "lengths": list(fit.lengths),
        "identification_passed": fit.identification.passed,
        "transition_support_passed": fit.historical_transition_support.passed,
        "effective_identification_passed": effective_identification_passed(fit),
        "effective_transition_support_passed": (
            effective_transition_support_passed(fit)
        ),
    }
    return passed, evidence, {"selected_k": fit.n_states}


def _dependence(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if gate_id != "four_cell_dependence_holm":
        if (
            not is_registered_dependence_task_view(result)
            or result.task_id != gate_id
            or not result.strict_canonical
        ):
            raise ModelError(f"{gate_id} requires its sealed dependence task view")
        role = "design" if gate_id.startswith("design_") else "production"
        kind = "fragmentation" if gate_id.endswith("fragmentation") else "dependence"
        cells_evidence: list[dict[str, Any]]
        if result.role != role or result.task_kind != kind:
            raise ModelError("dependence task view has the wrong role or kind")
        if kind == "fragmentation":
            if any(
                not isinstance(cell, FragmentationTaskCellEvidence)
                for cell in result.cells
            ):
                raise ModelError("fragmentation task view exposes dependence evidence")
            fragmentation_cells = tuple(
                cell
                for cell in result.cells
                if isinstance(cell, FragmentationTaskCellEvidence)
            )
            passed = all(
                cell.histories == CANONICAL_HISTORY_COUNT
                and cell.common_draw_reused
                and cell.fragmentation_passed
                for cell in fragmentation_cells
            )
            cells_evidence = [
                {
                    "cell_id": cell.cell_id,
                    "histories": cell.histories,
                    "selected_length": cell.selected_length,
                    "common_draw_reused": cell.common_draw_reused,
                    "fragmentation_passed": cell.fragmentation_passed,
                    "source_evidence_fingerprint": (cell.source_evidence_fingerprint),
                }
                for cell in fragmentation_cells
            ]
        else:
            if any(
                not isinstance(cell, DependenceTaskCellEvidence)
                for cell in result.cells
            ):
                raise ModelError("dependence task view exposes fragmentation evidence")
            dependence_cells = tuple(
                cell
                for cell in result.cells
                if isinstance(cell, DependenceTaskCellEvidence)
            )
            passed = all(
                cell.histories == CANONICAL_HISTORY_COUNT
                and cell.common_draw_reused
                and cell.endpoint_count == (87 if cell.frequency == "monthly" else 31)
                and cell.selected_result.repetitions == CANONICAL_HISTORY_COUNT
                and cell.selected_result.decision_role == "selected"
                and cell.selected_result.observed_statistic
                <= cell.selected_result.critical_value_95
                for cell in dependence_cells
            )
            cells_evidence = [
                {
                    "cell_id": cell.cell_id,
                    "histories": cell.histories,
                    "selected_length": cell.selected_length,
                    "endpoint_count": cell.endpoint_count,
                    "common_draw_reused": cell.common_draw_reused,
                    "p_value": cell.selected_result.p_value,
                    "predictive_envelope_passed": (
                        cell.selected_result.observed_statistic
                        <= cell.selected_result.critical_value_95
                    ),
                    "source_evidence_fingerprint": (cell.source_evidence_fingerprint),
                }
                for cell in dependence_cells
            ]
        if result.passed is not bool(passed):
            raise ModelError("dependence task-view decision is inconsistent")
        evidence: dict[str, Any] = {
            "task_id": gate_id,
            "role": role,
            "task_kind": kind,
            "run_id": result.run_id,
            "binding_fingerprint": result.binding_fingerprint,
            "source_execution_fingerprint": result.source_execution_fingerprint,
            "parent_block_core_fingerprints": list(
                result.parent_block_core_fingerprints
            ),
            "fragmentation_parent_view_fingerprint": (
                result.fragmentation_parent_view_fingerprint
            ),
            "view_fingerprint": result.view_fingerprint,
            "cells": cells_evidence,
        }
        return bool(passed), evidence, {"role": role, "cell_count": len(result.cells)}
    if (
        not is_registered_four_cell_dependence_holm_view(result)
        or result.task_id != gate_id
        or result.cell_order != DEPENDENCE_CELL_ORDER
        or not result.strict_canonical
    ):
        raise ModelError(
            "four_cell_dependence_holm requires its sealed four-cell Holm view"
        )
    p_values = np.asarray(
        [cell.selected_result.p_value for cell in result.cells], dtype=np.float64
    )
    expected_holm = holm_correction(p_values, alpha=result.holm.holm.alpha)
    if not _same_holm(result.holm.holm, expected_holm):
        raise ModelError("dependence Holm result is inconsistent")
    passed = not bool(expected_holm.rejected.any()) and all(
        cell.common_draw_reused
        and cell.selected_result.observed_statistic
        <= cell.selected_result.critical_value_95
        for cell in result.cells
    )
    if result.passed != passed:
        raise ModelError("dependence aggregate pass flag is inconsistent")
    evidence = {
        "task_id": gate_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "source_binding_fingerprints": [
            source.binding_fingerprint for source in result.source_views
        ],
        "source_view_fingerprints": list(result.source_view_fingerprints),
        "cell_order": list(DEPENDENCE_CELL_ORDER),
        "p_values": p_values.tolist(),
        "rejected": expected_holm.rejected.tolist(),
        "failure_reasons": list(result.failure_reasons),
    }
    return passed, evidence, {"holm_passed": passed}


def _bridge_preflight(
    result: object,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    view = _bridge_view("bridge_resource_preflight", result)
    source = _staged_bridge_source(view)
    if source.dgp is not None or view.planned_looks:
        raise ModelError("bridge resource evidence has invalid DGP/look identity")
    passed = source.passed
    evidence = {
        **_bridge_binding_evidence(view),
        "resource_preflight_passed": passed,
    }
    return (
        passed,
        evidence,
        {
            "resource_preflight_passed": passed,
            "selected_k": view.selected_k,
        },
    )


def _tier_a(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    view = _bridge_view(gate_id, result)
    source = _staged_bridge_source(view)
    expected_dgp: BridgeDGP = "nonparametric" if "nonparametric" in gate_id else "model"
    if source.dgp != expected_dgp or view.planned_looks:
        raise ModelError("Tier-A staged evidence has invalid DGP/look identity")
    passed = source.passed
    evidence = {
        **_bridge_binding_evidence(view),
        "dgp": expected_dgp,
        "tier_a_passed": passed,
    }
    return (
        passed,
        evidence,
        {
            "dgp": expected_dgp,
            "selected_k": view.selected_k,
        },
    )


def _tier_b(
    gate_id: str, result: object
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    view = _bridge_view(gate_id, result)
    source = _staged_bridge_source(view)
    expected_dgp: BridgeDGP = "nonparametric" if "nonparametric" in gate_id else "model"
    looks = view.planned_looks
    if source.dgp != expected_dgp or not looks:
        raise ModelError("Tier-B staged evidence lacks its registered DGP/looks")
    if len(looks) > TIER_B_MAXIMUM_PAIRS // TIER_B_BATCH_SIZE:
        raise ModelError("Tier-B planned-look evidence is too long")
    for index, look in enumerate(looks, start=1):
        if (
            look.dgp != expected_dgp
            or look.look_index != index
            or look.sample_count != index * TIER_B_BATCH_SIZE
            or not 0 <= look.exceedances <= look.sample_count
            or _FINGERPRINT.fullmatch(look.stage_receipt_fingerprint) is None
        ):
            raise ModelError("Tier-B look schedule/counts are invalid")
        expected_interval = clopper_pearson_interval(
            look.exceedances,
            look.sample_count,
            confidence=1.0 - TIER_B_INTERVAL_ERROR,
        )
        crosses = (
            expected_interval.lower
            <= TIER_B_TAIL_PROBABILITY
            <= expected_interval.upper
        )
        expected_look_decision = (
            "continue"
            if crosses and look.sample_count < TIER_B_MAXIMUM_PAIRS
            else "pass_above_tail"
            if expected_interval.lower > TIER_B_TAIL_PROBABILITY
            else "fail_below_tail"
            if expected_interval.upper < TIER_B_TAIL_PROBABILITY
            else "fail_closed_at_maximum"
        )
        if look.decision != expected_look_decision:
            raise ModelError("Tier-B local look decision is inconsistent")
        if not crosses and index != len(looks):
            raise ModelError("Tier-B retained looks continue after a decision")
    final = looks[-1]
    expected_decision = final.decision
    if expected_decision == "continue":
        raise ModelError("Tier-B verified local look sequence is nonterminal")
    if (
        expected_decision == "fail_closed_at_maximum"
        and final.sample_count != TIER_B_MAXIMUM_PAIRS
    ):
        raise ModelError("Tier-B crossing evidence ended before 10,000 pairs")
    passed = expected_decision == "pass_above_tail"
    if source.passed != passed:
        raise ModelError("Tier-B result is internally inconsistent")
    final_interval = clopper_pearson_interval(
        final.exceedances,
        final.sample_count,
        confidence=1.0 - TIER_B_INTERVAL_ERROR,
    )
    evidence = {
        **_bridge_binding_evidence(view),
        "dgp": expected_dgp,
        "pairs_used": final.sample_count,
        "exceedances": final.exceedances,
        "decision": expected_decision,
        "look_count": len(looks),
        "look_receipt_fingerprints": [item.stage_receipt_fingerprint for item in looks],
        "tier_b_result_fingerprint": view.task_result_fingerprint,
        "final_interval": {
            "lower": final_interval.lower,
            "upper": final_interval.upper,
        },
    }
    return (
        passed,
        evidence,
        {
            "dgp": expected_dgp,
            "decision": expected_decision,
            "selected_k": view.selected_k,
        },
    )


def _actual_bridge(
    result: object,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    view = _bridge_view("actual_artifact_bridge", result)
    source = _staged_bridge_source(view)
    if source.dgp is not None or view.planned_looks or view.rng_fingerprint is not None:
        raise ModelError("actual bridge staged identity is inconsistent")
    passed = source.passed
    evidence = {
        **_bridge_binding_evidence(view),
        "actual_bridge_passed": passed,
    }
    return (
        bool(passed),
        evidence,
        {
            "actual_bridge_passed": bool(passed),
            "selected_k": view.selected_k,
        },
    )


def _bridge_view(gate_id: str, result: object) -> BridgeDriverTaskView:
    if not is_bridge_driver_task_view(result) or result.task_id != gate_id:
        raise ModelError(f"{gate_id} requires its sealed bridge-driver task view")
    return result


def _staged_bridge_source(view: BridgeDriverTaskView) -> StagedBridgeTaskEvidence:
    source = view.source
    if not isinstance(source, StagedBridgeTaskEvidence):
        raise ModelError("canonical bridge view lacks staged task evidence")
    return source


def _bridge_binding_evidence(
    view: BridgeDriverTaskView,
) -> dict[str, Any]:
    source = _staged_bridge_source(view)
    return {
        "task_id": view.task_id,
        "run_id": view.run_id,
        "task_binding_fingerprint": view.binding_fingerprint,
        "view_fingerprint": view.view_fingerprint,
        "bridge_binding_fingerprint": view.shared_binding_fingerprint,
        "base_preflight_fingerprint": view.base_preflight_fingerprint,
        "selected_k": view.selected_k,
        "bridge_task_result_fingerprint": view.task_result_fingerprint,
        "bridge_parent_result_fingerprints": dict(view.parent_result_fingerprints),
        "bridge_task_source_lineage_fingerprint": (view.source_lineage_fingerprint),
        "bridge_task_materialization_fingerprint": view.materialization_fingerprint,
        "bridge_task_rng_fingerprint": view.rng_fingerprint,
        "bridge_known_trace_fingerprint": view.known_trace_fingerprint,
        "bridge_result_fingerprint": source.result_fingerprint,
        "bridge_evidence_fingerprint": source.evidence_fingerprint,
        "bridge_stage_receipt_fingerprint": source.stage_receipt_fingerprint,
        "bridge_task_passed": source.passed,
        "planned_look_fingerprints": list(view.planned_look_fingerprints),
        "generation_authorized": False,
    }


def _artifact_integrity(
    result: object,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not is_artifact_integrity_task_view(result):
        raise ModelError("artifact_integrity requires its sealed task view")
    source = result.source
    pair = source.pair
    design = source.design
    production = source.production
    passed = (
        design.role == "design"
        and production.role == "production"
        and design.selected_k == production.selected_k
        and design.binding == production.binding
        and design.provenance == production.provenance
        and design.reference.fingerprint != production.reference.fingerprint
        and pair.design is design
        and pair.production is production
        and pair.reference.fingerprint == result.pair_receipt_fingerprint
        and pair.actual_bridge_parent_view_fingerprint
        == result.actual_bridge_parent_view_fingerprint
        and not design.generation_authorized
        and not production.generation_authorized
        and not pair.generation_authorized
    )
    evidence = {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "binding_fingerprint": result.binding_fingerprint,
        "view_fingerprint": result.view_fingerprint,
        "source_fingerprint": result.source_fingerprint,
        "actual_bridge_parent_view_fingerprint": (
            result.actual_bridge_parent_view_fingerprint
        ),
        "final_artifact_parent_fingerprint": (result.final_artifact_parent_fingerprint),
        "report_set_fingerprint": result.report_set_fingerprint,
        "artifact_binding_fingerprint": result.artifact_binding_fingerprint,
        "provenance_fingerprint": result.provenance_fingerprint,
        "design_artifact_fingerprint": design.reference.fingerprint,
        "production_artifact_fingerprint": production.reference.fingerprint,
        "design_parameter_set_fingerprint": design.parameter_set_fingerprint,
        "production_parameter_set_fingerprint": production.parameter_set_fingerprint,
        "pair_receipt_fingerprint": result.pair_receipt_fingerprint,
        "design_role_source_fingerprint": result.design_role_source_fingerprint,
        "production_role_source_fingerprint": (
            result.production_role_source_fingerprint
        ),
        "selected_k": design.selected_k,
        "binding": design.binding.as_dict(),
        "provenance": design.provenance.as_dict(),
        "typed_schema_verified": passed,
    }
    return (
        passed,
        evidence,
        {
            "selected_k": design.selected_k,
            "design_artifact_fingerprint": design.reference.fingerprint,
            "production_artifact_fingerprint": production.reference.fingerprint,
        },
    )


def _cross_binding_error(
    decisions: Mapping[str, DerivedG3GateDecision],
    *,
    config_fingerprint: str,
    processed_data_fingerprint: str,
    calibration_input_fingerprint: str,
) -> str | None:
    view_error = task_view_cross_binding_error(
        {task_id: decision._source for task_id, decision in decisions.items()}
    )
    if view_error is not None:
        return view_error
    selected_ks = {
        int(decision.summary["selected_k"])
        for decision in decisions.values()
        if decision.passed
        and isinstance(decision.summary.get("selected_k"), int)
        and not isinstance(decision.summary.get("selected_k"), bool)
    }
    if len(selected_ks) > 1:
        return "typed decisions disagree on selected K"
    bridge_views = tuple(
        decision._source
        for task_id, decision in decisions.items()
        if task_id in BRIDGE_TASK_IDS
        and isinstance(decision._source, BridgeDriverTaskView)
    )
    if len(bridge_views) > 1 and any(
        view.shared_binding_fingerprint != bridge_views[0].shared_binding_fingerprint
        for view in bridge_views[1:]
    ):
        return "bridge tasks mix distinct staged executions"
    bridge_view = bridge_views[0] if bridge_views else None
    artifact = decisions.get("artifact_integrity")
    if artifact is None or not isinstance(artifact._source, ArtifactIntegrityTaskView):
        return None
    pair = artifact._source.source
    binding = pair.design.binding
    if pair.production.binding != binding:
        return "design and production artifact bindings differ"
    if binding.config_fingerprint != config_fingerprint:
        return "artifact config fingerprint differs from bundle"
    if binding.processed_data_fingerprint != processed_data_fingerprint:
        return "artifact processed-data fingerprint differs from bundle"
    if binding.calibration_input_fingerprint != calibration_input_fingerprint:
        return "artifact calibration-input fingerprint differs from bundle"
    if bridge_view is not None and bridge_view.selected_k != pair.design.selected_k:
        return "bridge selected K differs from typed artifacts"
    for task_id, expected_core in (
        ("design_block_calibration", pair.design.core_model_fingerprint),
        ("production_block_calibration", pair.production.core_model_fingerprint),
    ):
        block = decisions.get(task_id)
        if (
            block is not None
            and block.passed
            and block.summary.get("core_model_fingerprint") != expected_core
        ):
            return f"{task_id} kernel core fingerprint differs from artifact"
    input_decision = decisions.get("input_integrity")
    if input_decision is not None and isinstance(
        input_decision._source, InputIntegrityTaskView
    ):
        input_source = input_decision._source.source
        if input_source.binding != binding:
            return "preflight and artifact bindings differ"
        if (
            pair.design.provenance.source_code_fingerprint
            != input_source.launch_preflight.source_code_fingerprint
            or pair.design.provenance.dependency_lock_fingerprint
            != input_source.launch_preflight.dependency_lock_fingerprint
        ):
            return "artifact provenance differs from launch preflight"
        if (
            bridge_view is not None
            and bridge_view.base_preflight_fingerprint
            != input_source.launch_preflight.fingerprint
        ):
            return "bridge launch preflight differs from input integrity"
    return None


def task_view_cross_binding_error(sources: Mapping[str, object]) -> str | None:
    """Return a structural lineage error across sealed task-scoped views.

    The individual type guards own object authority and content rehashing.
    This cross-task check owns temporal/run ancestry: distinct task bindings,
    a single launch, staged candidate parentage, exact model lineage, and Holm
    aggregation from the already supplied prerequisite views.  Durable
    :class:`ScientificTaskBinding` store verification is intentionally a later
    coordinator integration and is not approximated here.
    """

    view_types = (
        InputIntegrityTaskView,
        CandidateViabilityTaskView,
        CandidateSelectionTaskView,
        SealedHoldoutVetoView,
        MacroStabilityTaskView,
        ProductionFitSameKView,
        AdequacyCellTaskView,
        FourCellAdequacyHolmView,
        DependenceTaskView,
        FourCellDependenceHolmView,
        BlockCalibrationTaskView,
        BridgeDriverTaskView,
        ArtifactIntegrityTaskView,
    )
    scoped = {
        task_id: source
        for task_id, source in sources.items()
        if isinstance(source, view_types)
    }
    if not scoped:
        return None
    run_ids = {source.run_id for source in scoped.values()}
    if len(run_ids) != 1:
        return "task-scoped scientific views mix calibration runs"
    input_view = sources.get("input_integrity")
    if isinstance(input_view, InputIntegrityTaskView) and run_ids != {
        input_view.source.binding.calibration_run_fingerprint
    }:
        return "task-scoped scientific views differ from the input launch"
    bindings = tuple(source.binding_fingerprint for source in scoped.values())
    if len(set(bindings)) != len(bindings):
        return "task-scoped scientific views reuse a task binding"
    if any(source.task_id != task_id for task_id, source in scoped.items()):
        return "task-scoped scientific view is bound to the wrong task"
    bridge_views = tuple(
        source
        for task_id, source in scoped.items()
        if task_id in BRIDGE_TASK_IDS and isinstance(source, BridgeDriverTaskView)
    )
    if len(bridge_views) > 1 and any(
        source.shared_binding_fingerprint != bridge_views[0].shared_binding_fingerprint
        for source in bridge_views[1:]
    ):
        return "bridge tasks mix distinct preregistered bindings"
    bridge_by_task: dict[str, BridgeDriverTaskView] = {
        view.task_id: view for view in bridge_views
    }
    expected_bridge_parents = {
        "bridge_resource_preflight": (),
        "bridge_tier_a_model_dgp": ("bridge_resource_preflight",),
        "bridge_tier_a_nonparametric_dgp": ("bridge_resource_preflight",),
        "bridge_tier_b_model_dgp": (
            "bridge_resource_preflight",
            "bridge_tier_a_model_dgp",
            "bridge_tier_a_nonparametric_dgp",
        ),
        "bridge_tier_b_nonparametric_dgp": (
            "bridge_resource_preflight",
            "bridge_tier_a_model_dgp",
            "bridge_tier_a_nonparametric_dgp",
        ),
        "actual_artifact_bridge": (
            "bridge_tier_a_model_dgp",
            "bridge_tier_a_nonparametric_dgp",
            "bridge_tier_b_model_dgp",
            "bridge_tier_b_nonparametric_dgp",
        ),
    }
    for view in bridge_views:
        if (
            tuple(parent for parent, _ in view.parent_result_fingerprints)
            != (expected_bridge_parents[view.task_id])
        ):
            return f"{view.task_id} has the wrong bridge parent lineage"
        for parent_id, parent_result in view.parent_result_fingerprints:
            bridge_parent_view = bridge_by_task.get(parent_id)
            if (
                bridge_parent_view is not None
                and bridge_parent_view.task_result_fingerprint != parent_result
            ):
                return f"{view.task_id} bridge parent result is inconsistent"
    actual_bridge = _source_as(sources, "actual_artifact_bridge", BridgeDriverTaskView)
    artifact_integrity = _source_as(
        sources, "artifact_integrity", ArtifactIntegrityTaskView
    )
    if (
        actual_bridge is not None
        and artifact_integrity is not None
        and artifact_integrity._actual_bridge_parent is not actual_bridge
    ):
        return "artifact integrity does not descend from the exact bridge view"

    viability = _source_as(
        sources, "design_candidate_viability", CandidateViabilityTaskView
    )
    selection = _source_as(
        sources, "design_predictive_selection", CandidateSelectionTaskView
    )
    if (
        viability is not None
        and selection is not None
        and (
            selection._viability_parent is not viability
            or selection.viability_view_fingerprint != viability.view_fingerprint
            or selection.source_content_fingerprint
            != viability.source_content_fingerprint
            or selection.fit_set_fingerprint != viability.fit_set_fingerprint
            or selection.viability_result_fingerprint
            != viability.viability_result_fingerprint
        )
    ):
        return "candidate selection does not descend from viability"

    holdout = _source_as(sources, "sealed_holdout_veto", SealedHoldoutVetoView)
    production_fit = _source_as(
        sources, "production_fit_same_k", ProductionFitSameKView
    )
    if selection is not None:
        if holdout is not None and (
            holdout._selection_source is not selection
            or holdout.selection_view_fingerprint != selection.view_fingerprint
        ):
            return "sealed holdout does not descend from predictive selection"
        if production_fit is not None and (
            production_fit._selection_source is not selection
            or production_fit.selection_view_fingerprint != selection.view_fingerprint
            or production_fit.design_selected_k != selection.selected_n_states
        ):
            return "production fit does not descend from predictive selection"
    if (
        holdout is not None
        and production_fit is not None
        and (
            holdout.selected_model_fingerprint
            != production_fit.design_model_fingerprint
            or holdout.selected_n_states != production_fit.design_selected_k
        )
    ):
        return "design-model lineage differs between holdout and production fit"

    design_model = (
        holdout.selected_model_fingerprint
        if holdout is not None
        else production_fit.design_model_fingerprint
        if production_fit is not None
        else None
    )
    production_model = (
        None if production_fit is None else production_fit.production_fit_fingerprint
    )
    for task_id, expected_model in (
        ("design_block_calibration", design_model),
        ("production_block_calibration", production_model),
    ):
        block_view = _source_as(sources, task_id, BlockCalibrationTaskView)
        if (
            block_view is not None
            and expected_model is not None
            and block_view.parent_model_fingerprint != expected_model
        ):
            return f"{task_id} parent model fingerprint is inconsistent"
    for task_id, expected_role, parent_view, parent_model in (
        ("design_macro_stability", "design", selection, design_model),
        (
            "production_macro_stability",
            "production",
            production_fit,
            production_model,
        ),
    ):
        macro = _source_as(sources, task_id, MacroStabilityTaskView)
        if macro is None:
            continue
        if macro.role != expected_role:
            return f"{task_id} has cross-role macro evidence"
        if parent_view is not None and (
            macro._parent_view is not parent_view
            or macro.parent_view_fingerprint != parent_view.view_fingerprint
        ):
            return f"{task_id} does not descend from its exact parent view"
        if parent_model is not None and macro.parent_model_fingerprint != parent_model:
            return f"{task_id} parent model fingerprint is inconsistent"

    adequacy_tasks = (
        "design_latent_correctness",
        "production_latent_correctness",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )
    adequacy_views = tuple(
        _source_as(sources, task_id, AdequacyCellTaskView) for task_id in adequacy_tasks
    )
    for task_id, adequacy_view in zip(adequacy_tasks, adequacy_views, strict=True):
        if adequacy_view is None:
            continue
        expected_model = (
            design_model if task_id.startswith("design_") else production_model
        )
        if (
            expected_model is not None
            and adequacy_view.parent_model_fingerprint != expected_model
        ):
            return f"{task_id} parent model fingerprint is inconsistent"
    adequacy_holm = _source_as(
        sources, "four_cell_adequacy_holm", FourCellAdequacyHolmView
    )
    if adequacy_holm is not None:
        for expected, retained in zip(
            adequacy_views, adequacy_holm.source_views, strict=True
        ):
            if expected is not None and retained is not expected:
                return "adequacy Holm does not retain the exact prerequisite views"

    dependence_pairs = (
        ("design_fragmentation", "design_dependence"),
        ("production_fragmentation", "production_dependence"),
    )
    dependence_views: dict[str, DependenceTaskView] = {}
    for fragmentation_id, dependence_id in dependence_pairs:
        fragmentation = _source_as(sources, fragmentation_id, DependenceTaskView)
        dependence = _source_as(sources, dependence_id, DependenceTaskView)
        if fragmentation is not None:
            dependence_views[fragmentation_id] = fragmentation
        if dependence is not None:
            dependence_views[dependence_id] = dependence
        if (
            fragmentation is not None
            and dependence is not None
            and fragmentation is dependence
        ):
            return f"{fragmentation.role} dependence reuses its fragmentation view"
        if fragmentation is not None and (
            fragmentation.fragmentation_parent_view_fingerprint is not None
        ):
            return f"{fragmentation.role} fragmentation contains future lineage"
        if fragmentation is not None and dependence is not None:
            if (
                fragmentation.parent_block_core_fingerprints
                != dependence.parent_block_core_fingerprints
            ):
                return f"{fragmentation.role} dependence block-core lineage differs"
            if dependence.fragmentation_parent_view_fingerprint != (
                fragmentation.view_fingerprint
            ):
                return (
                    f"{fragmentation.role} dependence does not descend from "
                    "fragmentation"
                )
    for task_id, dependence_view in dependence_views.items():
        expected_model = (
            design_model if task_id.startswith("design_") else production_model
        )
        if expected_model is not None and any(
            cell.model_fingerprint != expected_model for cell in dependence_view.cells
        ):
            return f"{task_id} parent model fingerprint is inconsistent"
    dependence_holm = _source_as(
        sources, "four_cell_dependence_holm", FourCellDependenceHolmView
    )
    if dependence_holm is not None:
        expected_design = dependence_views.get("design_dependence")
        expected_production = dependence_views.get("production_dependence")
        if (
            expected_design is not None
            and dependence_holm.source_views[0] is not expected_design
        ) or (
            expected_production is not None
            and dependence_holm.source_views[1] is not expected_production
        ):
            return "dependence Holm does not retain the exact prerequisite views"
    return None


def _source_as(
    sources: Mapping[str, object],
    task_id: str,
    expected: type[_T],
) -> _T | None:
    source = sources.get(task_id)
    return source if isinstance(source, expected) else None


def _same_holm(left: HolmResult, right: HolmResult) -> bool:
    return (
        np.array_equal(left.raw_p_values, right.raw_p_values)
        and np.array_equal(left.adjusted_p_values, right.adjusted_p_values)
        and np.array_equal(left.rejected, right.rejected)
        and left.alpha == right.alpha
    )


def _validate_failure(value: object) -> None:
    if not isinstance(value, G3TaskFailure) or value.task_id not in G3_TASK_BY_ID:
        raise ModelError("G3 task failure is invalid or unregistered")
    if (
        not value.failure_type
        or len(value.failure_type) > 128
        or not value.message
        or len(value.message) > 512
    ):
        raise ModelError("G3 task failure text is empty or unbounded")


def _fingerprint(value: object, label: str) -> None:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ModelError(f"G3 {label} fingerprint is invalid")
