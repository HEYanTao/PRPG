"""Sealed task-scoped views for ordinary Phase-3 calibration families.

The numerical calibration modules intentionally return reusable scientific
records.  A G3 checkpoint needs a narrower object: one exact task, run,
upstream model, input materialization, and registered RNG namespace.  This
module creates those views without importing the durable run/checkpoint
stores.  A verified :class:`~prpg.model.g3_task_binding.ScientificTaskBinding`
supplies ``run_id`` and ``binding_fingerprint`` to the builders.

Every public type guard re-fingerprints the retained live objects.  A frozen
dataclass therefore is not treated as proof by itself: hand construction,
``dataclasses.replace``, task swapping, array mutation, model mutation, and
cross-run/role/input mixing all fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias, TypeGuard, cast

import numpy as np

from prpg.errors import ModelError
from prpg.model.calibration import (
    ArtifactRole,
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
    CANONICAL_CANDIDATES,
    CandidateGateExecutionIdentity,
    CandidateGateResult,
    CandidateViabilityExecutionIdentity,
    CandidateViabilityResult,
    has_registered_final_fixed_point_candidate_authority,
    verified_candidate_gate_execution_identity,
    verified_candidate_viability_execution_identity,
    verified_registered_candidate_gate_execution_identity,
    verified_registered_candidate_viability_execution_identity,
)
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMFeatureMatrix,
    deterministic_hmm_restart_seeds,
)
from prpg.model.input import CalibrationSlice
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    rng_execution_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.production_fit_execution import (
    RegisteredProductionFitSameK,
    registered_production_fit_same_k_identity,
    registered_production_fit_same_k_sources,
)
from prpg.model.selection import (
    OWNER_FIXED_N_STATES,
    CandidateSelectionReport,
    CandidateViabilityRecord,
)
from prpg.simulation.rng import CalibrationArtifact, ModelFitScope, Stage

MacroTaskId: TypeAlias = Literal[
    "design_macro_stability",
    "production_macro_stability",
]

CALIBRATION_TASK_VIEW_CONTRACT: Final = "g3-calibration-task-views-v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_VIEW_CAPABILITY = object()
_HOLDOUT_VIEW_CAPABILITY = object()
_MACRO_VIEW_CAPABILITY = object()
_PRODUCTION_FIT_VIEW_CAPABILITY = object()
_MACRO_TASK_ROLE: Final = {
    "design_macro_stability": "design",
    "production_macro_stability": "production",
}

# Keeping the exact minted object prevents a copied instance from inheriting
# authority even if a caller deliberately reaches for a private seal field.
_MINTED_VIABILITY_VIEWS: dict[int, CandidateViabilityTaskView] = {}
_MINTED_SELECTION_VIEWS: dict[int, CandidateSelectionTaskView] = {}
_MINTED_HOLDOUT_VIEWS: dict[int, SealedHoldoutVetoView] = {}
_MINTED_MACRO_VIEWS: dict[int, MacroStabilityTaskView] = {}
_MINTED_PRODUCTION_FIT_VIEWS: dict[
    int,
    tuple[ProductionFitSameKView, RegisteredProductionFitSameK],
] = {}


@dataclass(frozen=True, slots=True)
class CandidateViabilityTaskRow:
    """Only the fit/pre-BIC evidence needed by the viability decision."""

    n_states: int
    fit_succeeded: bool
    viability: CandidateViabilityRecord


@dataclass(frozen=True, slots=True, init=False)
class CandidateViabilityTaskView:
    """Run-bound pre-BIC view with no predictive-selection outcome."""

    contract_version: str
    task_id: Literal["design_candidate_viability"]
    role: Literal["design"]
    run_id: str
    binding_fingerprint: str
    viability_rows: tuple[
        CandidateViabilityTaskRow,
        CandidateViabilityTaskRow,
        CandidateViabilityTaskRow,
        CandidateViabilityTaskRow,
    ]
    admissible_n_states: tuple[int, ...]
    passed: bool
    fit_set_fingerprint: str
    source_content_fingerprint: str
    viability_result_fingerprint: str
    suite_fingerprint: str | None
    suite_combined_materialization_fingerprint: str | None
    suite_combined_fit_rng_execution_fingerprint: str | None
    suite_combined_rng_execution_fingerprint: str | None
    suite_macro_materialization_fingerprint: str | None
    suite_monthly_block_materialization_fingerprint: str | None
    suite_daily_block_materialization_fingerprint: str | None
    fixed_point_final_iteration: int | None
    fixed_point_viability_projection_fingerprint: str | None
    master_seed: int
    scientific_version: int
    rng_namespace: Literal["design_candidate_fit_restarts"]
    rng_execution_fingerprint: str
    restart_seed_fingerprints: tuple[str, str, str, str]
    viability_fingerprint: str
    view_fingerprint: str
    generation_authorized: Literal[False]
    # ``_canonical_suite_authority`` is retained as a compatibility name for
    # strict *iteration-suite* authority.  Final G3 authority is separate.
    _canonical_suite_authority: bool = field(repr=False, compare=False)
    _final_fixed_point_authority: bool = field(repr=False, compare=False)
    _source_result: CandidateViabilityResult = field(repr=False, compare=False)
    _view_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "CandidateViabilityTaskView is created only by "
            "build_candidate_viability_task_view"
        )


@dataclass(frozen=True, slots=True, init=False)
class CandidateSelectionTaskView:
    """Predictive/BIC selection view descended from the published viability."""

    contract_version: str
    task_id: Literal["design_predictive_selection"]
    role: Literal["design"]
    run_id: str
    binding_fingerprint: str
    selection: CandidateSelectionReport
    selected_n_states: int | None
    passed: bool
    fit_set_fingerprint: str
    rolling_fingerprints: tuple[str | None, ...]
    bootstrap_fingerprint: str
    bootstrap_materialization_fingerprint: str
    bootstrap_rng_execution_fingerprint: str
    source_content_fingerprint: str
    viability_view_fingerprint: str
    viability_result_fingerprint: str
    selection_fingerprint: str
    fixed_point_outcome: str | None
    fixed_point_failure_reason: str | None
    fixed_point_converged_at_iteration: int | None
    fixed_point_iteration_fingerprints: tuple[str, ...]
    fixed_point_iteration_history_fingerprint: str | None
    fixed_point_fingerprint: str | None
    fixed_point_final_selection_fingerprint: str | None
    fixed_point_final_monthly_block_core_fingerprint: str | None
    fixed_point_final_daily_block_core_fingerprint: str | None
    fixed_point_viability_projection_fingerprint: str | None
    view_fingerprint: str
    generation_authorized: Literal[False]
    _canonical_suite_authority: bool = field(repr=False, compare=False)
    _final_fixed_point_authority: bool = field(repr=False, compare=False)
    _source_result: CandidateGateResult = field(repr=False, compare=False)
    _viability_parent: CandidateViabilityTaskView = field(
        repr=False,
        compare=False,
    )
    _view_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "CandidateSelectionTaskView is created only by "
            "build_candidate_selection_task_view"
        )


@dataclass(frozen=True, slots=True)
class CandidateGateTaskViews:
    """Both design candidate tasks minted from one indivisible execution."""

    design_candidate_viability: CandidateViabilityTaskView
    design_predictive_selection: CandidateSelectionTaskView


@dataclass(frozen=True, slots=True, init=False)
class SealedHoldoutVetoView:
    """Selected-design-model holdout execution bound to both exact slices."""

    contract_version: str
    task_id: Literal["sealed_holdout_veto"]
    role: Literal["design"]
    run_id: str
    binding_fingerprint: str
    selected_n_states: int
    selected_model_fingerprint: str
    design_slice_fingerprint: str
    holdout_slice_fingerprint: str
    selection_view_fingerprint: str
    evidence_fingerprint: str
    evidence: SealedHoldoutEvidence
    view_fingerprint: str
    generation_authorized: Literal[False]
    _selection_source: CandidateSelectionTaskView = field(repr=False, compare=False)
    _design_slice: CalibrationSlice = field(repr=False, compare=False)
    _holdout_slice: CalibrationSlice = field(repr=False, compare=False)
    _view_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "SealedHoldoutVetoView is created only by build_sealed_holdout_veto_view"
        )


@dataclass(frozen=True, slots=True, init=False)
class ProductionFitSameKView:
    """Actual fixed-K production fit bound to the frozen design selection."""

    contract_version: str
    task_id: Literal["production_fit_same_k"]
    role: Literal["production"]
    run_id: str
    binding_fingerprint: str
    design_selected_k: int
    design_model_fingerprint: str
    selection_view_fingerprint: str
    production_feature_fingerprint: str
    master_seed: int
    scientific_version: int
    rng_namespace: Literal["production_fit_restarts"]
    rng_execution_fingerprint: str
    restart_seed_fingerprint: str
    production_fit: GaussianHMMFit
    production_fit_fingerprint: str
    registered_execution_fingerprint: str
    view_fingerprint: str
    generation_authorized: Literal[False]
    _selection_source: CandidateSelectionTaskView = field(repr=False, compare=False)
    _production_features: HMMFeatureMatrix = field(repr=False, compare=False)
    _registered_execution: RegisteredProductionFitSameK = field(
        repr=False,
        compare=False,
    )
    _view_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "ProductionFitSameKView is created only by build_production_fit_same_k_view"
        )


@dataclass(frozen=True, slots=True, init=False)
class MacroStabilityTaskView:
    """One exact registered macro family with role-locked parent ancestry."""

    contract_version: str
    task_id: MacroTaskId
    role: ArtifactRole
    run_id: str
    binding_fingerprint: str
    selected_k: int
    parent_model_fingerprint: str
    parent_view_fingerprint: str
    source_slice_fingerprint: str
    macro_input_fingerprint: str
    master_seed: int
    scientific_version: int
    rng_namespace: str
    rng_execution_fingerprint: str
    result: MacroStabilityEvidence
    evidence_fingerprint: str
    view_fingerprint: str
    generation_authorized: Literal[False]
    _parent_view: CandidateSelectionTaskView | ProductionFitSameKView = field(
        repr=False,
        compare=False,
    )
    _source_slice: CalibrationSlice = field(repr=False, compare=False)
    _view_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "MacroStabilityTaskView is created only by build_macro_stability_task_view"
        )


def build_candidate_viability_task_view(
    result: CandidateViabilityResult,
    *,
    master_seed: int,
    scientific_version: int,
    run_id: str,
    binding_fingerprint: str,
    require_canonical_suite: bool = False,
) -> CandidateViabilityTaskView:
    """Mint a pre-BIC view, optionally requiring strict iteration authority.

    ``require_canonical_suite`` is a compatibility spelling for requiring a
    strict registered *iteration* suite.  Even ``True`` does not grant final
    fixed-point authority; canonical publication additionally requires the
    producer-sealed final-iteration wrapper.
    """

    _require_sha256(run_id, "candidate task run ID")
    _require_sha256(binding_fingerprint, "candidate viability binding fingerprint")
    identity = (
        verified_registered_candidate_viability_execution_identity(
            result,
            require_strict_canonical_geometry=True,
        )
        if require_canonical_suite
        else verified_candidate_viability_execution_identity(result)
    )
    if identity.role != "design":
        raise ModelError("G3 candidate task views require design-role execution")
    if require_canonical_suite and (
        identity.suite_master_seed != master_seed
        or identity.suite_scientific_version != scientific_version
    ):
        raise ModelError("candidate task seed/version differ from registered suite")
    restart_fingerprints, rng_fingerprint = _candidate_fit_rng_identity(
        result,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    rows = _viability_rows(result)
    admissible = tuple(row.n_states for row in rows if row.viability.admissible)
    viability_fingerprint = _viability_result_fingerprint(rows)
    payload = _viability_payload(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        identity=identity,
        admissible_n_states=admissible,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_execution_fingerprint=rng_fingerprint,
        restart_seed_fingerprints=restart_fingerprints,
        viability_fingerprint=viability_fingerprint,
    )
    view = object.__new__(CandidateViabilityTaskView)
    for name, value in (
        ("contract_version", CALIBRATION_TASK_VIEW_CONTRACT),
        ("task_id", "design_candidate_viability"),
        ("role", "design"),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("viability_rows", rows),
        ("admissible_n_states", admissible),
        ("passed", OWNER_FIXED_N_STATES in admissible),
        ("fit_set_fingerprint", identity.fit_set_fingerprint),
        ("source_content_fingerprint", identity.source_content_fingerprint),
        (
            "viability_result_fingerprint",
            identity.viability_result_fingerprint,
        ),
        ("suite_fingerprint", identity.suite_fingerprint),
        (
            "suite_combined_materialization_fingerprint",
            identity.suite_combined_materialization_fingerprint,
        ),
        (
            "suite_combined_fit_rng_execution_fingerprint",
            identity.suite_combined_fit_rng_execution_fingerprint,
        ),
        (
            "suite_combined_rng_execution_fingerprint",
            identity.suite_combined_rng_execution_fingerprint,
        ),
        (
            "suite_macro_materialization_fingerprint",
            identity.suite_macro_materialization_fingerprint,
        ),
        (
            "suite_monthly_block_materialization_fingerprint",
            identity.suite_monthly_block_materialization_fingerprint,
        ),
        (
            "suite_daily_block_materialization_fingerprint",
            identity.suite_daily_block_materialization_fingerprint,
        ),
        ("fixed_point_final_iteration", identity.fixed_point_final_iteration),
        (
            "fixed_point_viability_projection_fingerprint",
            identity.fixed_point_viability_projection_fingerprint,
        ),
        ("master_seed", master_seed),
        ("scientific_version", scientific_version),
        ("rng_namespace", "design_candidate_fit_restarts"),
        ("rng_execution_fingerprint", rng_fingerprint),
        ("restart_seed_fingerprints", restart_fingerprints),
        ("viability_fingerprint", viability_fingerprint),
        ("view_fingerprint", _sha256_json(payload)),
        ("generation_authorized", False),
        ("_canonical_suite_authority", require_canonical_suite),
        (
            "_final_fixed_point_authority",
            result._final_fixed_point_authority,
        ),
        ("_source_result", result),
        ("_view_seal", _CANDIDATE_VIEW_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_VIABILITY_VIEWS[id(view)] = view
    return view


def build_candidate_selection_task_view(
    viability: CandidateViabilityTaskView,
    result: CandidateGateResult,
    *,
    run_id: str,
    binding_fingerprint: str,
) -> CandidateSelectionTaskView:
    """Mint predictive selection only after the exact viability parent exists."""

    _verify_viability_view(viability)
    _require_child_binding(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_run_id=viability.run_id,
        parent_binding_fingerprint=viability.binding_fingerprint,
        label="candidate selection",
    )
    identity = (
        verified_registered_candidate_gate_execution_identity(
            result,
            require_strict_canonical_geometry=True,
        )
        if viability._canonical_suite_authority
        else verified_candidate_gate_execution_identity(result)
    )
    if result.viability_result is not viability._source_result:
        raise ModelError(
            "candidate selection result does not descend from viability evidence"
        )
    if (
        identity.bootstrap_artifact is not CalibrationArtifact.DESIGN_OR_CONTROL
        or identity.bootstrap_master_seed != viability.master_seed
        or identity.bootstrap_scientific_version != viability.scientific_version
    ):
        raise ModelError(
            "candidate selection bootstrap identity differs from viability ancestry"
        )
    selection_fingerprint = _selection_result_fingerprint(result.selection)
    payload = _selection_payload(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        identity=identity,
        viability_view_fingerprint=viability.view_fingerprint,
        selection_fingerprint=selection_fingerprint,
    )
    view = object.__new__(CandidateSelectionTaskView)
    for name, value in (
        ("contract_version", CALIBRATION_TASK_VIEW_CONTRACT),
        ("task_id", "design_predictive_selection"),
        ("role", "design"),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("selection", result.selection),
        ("selected_n_states", result.selection.selected_n_states),
        (
            "passed",
            (
                result.selection.passed
                if identity.fixed_point_outcome is None
                else identity.fixed_point_outcome == "converged"
            ),
        ),
        ("fit_set_fingerprint", identity.fit_set_fingerprint),
        ("rolling_fingerprints", identity.rolling_fingerprints),
        ("bootstrap_fingerprint", identity.bootstrap_fingerprint),
        (
            "bootstrap_materialization_fingerprint",
            identity.bootstrap_materialization_fingerprint,
        ),
        (
            "bootstrap_rng_execution_fingerprint",
            identity.bootstrap_rng_execution_fingerprint,
        ),
        ("source_content_fingerprint", identity.source_content_fingerprint),
        ("viability_view_fingerprint", viability.view_fingerprint),
        (
            "viability_result_fingerprint",
            identity.viability_result_fingerprint,
        ),
        ("selection_fingerprint", selection_fingerprint),
        ("fixed_point_outcome", identity.fixed_point_outcome),
        (
            "fixed_point_failure_reason",
            identity.fixed_point_failure_reason,
        ),
        (
            "fixed_point_converged_at_iteration",
            identity.fixed_point_converged_at_iteration,
        ),
        (
            "fixed_point_iteration_fingerprints",
            identity.fixed_point_iteration_fingerprints,
        ),
        (
            "fixed_point_iteration_history_fingerprint",
            identity.fixed_point_iteration_history_fingerprint,
        ),
        ("fixed_point_fingerprint", identity.fixed_point_fingerprint),
        (
            "fixed_point_final_selection_fingerprint",
            identity.fixed_point_final_selection_fingerprint,
        ),
        (
            "fixed_point_final_monthly_block_core_fingerprint",
            identity.fixed_point_final_monthly_block_core_fingerprint,
        ),
        (
            "fixed_point_final_daily_block_core_fingerprint",
            identity.fixed_point_final_daily_block_core_fingerprint,
        ),
        (
            "fixed_point_viability_projection_fingerprint",
            identity.fixed_point_viability_projection_fingerprint,
        ),
        ("view_fingerprint", _sha256_json(payload)),
        ("generation_authorized", False),
        ("_canonical_suite_authority", viability._canonical_suite_authority),
        (
            "_final_fixed_point_authority",
            result._final_fixed_point_authority,
        ),
        ("_source_result", result),
        ("_viability_parent", viability),
        ("_view_seal", _CANDIDATE_VIEW_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_SELECTION_VIEWS[id(view)] = view
    return view


def assemble_candidate_gate_task_views(
    viability: CandidateViabilityTaskView,
    selection: CandidateSelectionTaskView,
) -> CandidateGateTaskViews:
    """Group two already-minted staged views without granting new authority."""

    _verify_viability_view(viability)
    _verify_selection_view(selection)
    if selection._viability_parent is not viability:
        raise ModelError("candidate selection does not descend from viability view")
    return CandidateGateTaskViews(viability, selection)


def is_candidate_viability_task_view(
    value: object,
) -> TypeGuard[CandidateViabilityTaskView]:
    """Return whether ``value`` is an unchanged pre-BIC task view."""

    if not isinstance(value, CandidateViabilityTaskView):
        return False
    try:
        _verify_viability_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_registered_candidate_viability_task_view(
    value: object,
) -> TypeGuard[CandidateViabilityTaskView]:
    """Return whether task 2 has final fixed-point, not just iteration, authority."""

    return bool(
        is_registered_candidate_iteration_task_view(value)
        and has_registered_final_fixed_point_candidate_authority(value._source_result)
    )


def is_registered_candidate_iteration_task_view(
    value: object,
) -> TypeGuard[CandidateViabilityTaskView]:
    """Return whether a report view has strict registered-iteration authority."""

    if not is_candidate_viability_task_view(value):
        return False
    if not value._canonical_suite_authority:
        return False
    try:
        verified_registered_candidate_viability_execution_identity(
            value._source_result,
            require_strict_canonical_geometry=True,
        )
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_candidate_selection_task_view(
    value: object,
) -> TypeGuard[CandidateSelectionTaskView]:
    """Return whether ``value`` is an unchanged staged selection view."""

    if not isinstance(value, CandidateSelectionTaskView):
        return False
    try:
        _verify_selection_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_registered_candidate_selection_task_view(
    value: object,
) -> TypeGuard[CandidateSelectionTaskView]:
    """Return whether task 3 descends from a complete fixed-point authority."""

    if not is_candidate_selection_task_view(value):
        return False
    if (
        not value._canonical_suite_authority
        or not has_registered_final_fixed_point_candidate_authority(
            value._source_result
        )
        or not is_registered_candidate_viability_task_view(value._viability_parent)
    ):
        return False
    try:
        verified_registered_candidate_gate_execution_identity(
            value._source_result,
            require_strict_canonical_geometry=True,
        )
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def selected_design_fit_from_view(
    value: CandidateSelectionTaskView,
) -> GaussianHMMFit:
    """Return the exact retained design fit after full view reverification."""

    if not isinstance(value, CandidateSelectionTaskView):
        raise ModelError("selected design fit requires a candidate-selection view")
    _verify_selection_view(value)
    return _selected_design_fit(value)


def build_sealed_holdout_veto_view(
    selection: CandidateSelectionTaskView,
    evidence: SealedHoldoutEvidence,
    *,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
    run_id: str,
    binding_fingerprint: str,
) -> SealedHoldoutVetoView:
    """Bind precomputed holdout evidence after an independent reevaluation."""

    _verify_selection_view(selection)
    if type(evidence) is not SealedHoldoutEvidence:
        raise ModelError(
            "sealed holdout requires explicit precomputed SealedHoldoutEvidence"
        )
    _require_child_binding(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_run_id=selection.run_id,
        parent_binding_fingerprint=selection.binding_fingerprint,
        label="sealed holdout",
    )
    fit = _selected_design_fit(selection)
    if not isinstance(fit, GaussianHMMFit):
        raise ModelError("sealed holdout requires a successfully selected design fit")
    model_before = gaussian_hmm_fit_fingerprint(fit)
    design_before = calibration_slice_fingerprint(design)
    holdout_before = calibration_slice_fingerprint(holdout)
    supplied_evidence_fingerprint = sealed_holdout_evidence_fingerprint(evidence)
    recomputed = evaluate_sealed_holdout(fit, design, holdout)
    if (
        gaussian_hmm_fit_fingerprint(fit) != model_before
        or calibration_slice_fingerprint(design) != design_before
        or calibration_slice_fingerprint(holdout) != holdout_before
    ):
        raise ModelError("sealed holdout sources changed during evaluation")
    if sealed_holdout_evidence_fingerprint(recomputed) != supplied_evidence_fingerprint:
        raise ModelError(
            "precomputed sealed holdout evidence differs from independent reevaluation"
        )
    evidence_fingerprint = supplied_evidence_fingerprint
    payload = _holdout_payload(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        selected_n_states=fit.n_states,
        selected_model_fingerprint=model_before,
        design_slice_fingerprint=design_before,
        holdout_slice_fingerprint=holdout_before,
        selection_view_fingerprint=selection.view_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )
    view = object.__new__(SealedHoldoutVetoView)
    for name, value in (
        ("contract_version", CALIBRATION_TASK_VIEW_CONTRACT),
        ("task_id", "sealed_holdout_veto"),
        ("role", "design"),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("selected_n_states", fit.n_states),
        ("selected_model_fingerprint", model_before),
        ("design_slice_fingerprint", design_before),
        ("holdout_slice_fingerprint", holdout_before),
        ("selection_view_fingerprint", selection.view_fingerprint),
        ("evidence_fingerprint", evidence_fingerprint),
        ("evidence", evidence),
        ("view_fingerprint", _sha256_json(payload)),
        ("generation_authorized", False),
        ("_selection_source", selection),
        ("_design_slice", design),
        ("_holdout_slice", holdout),
        ("_view_seal", _HOLDOUT_VIEW_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_HOLDOUT_VIEWS[id(view)] = view
    return view


def is_sealed_holdout_veto_view(
    value: object,
) -> TypeGuard[SealedHoldoutVetoView]:
    """Return whether ``value`` is an unchanged executed holdout view."""

    if not isinstance(value, SealedHoldoutVetoView):
        return False
    try:
        _verify_holdout_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def build_production_fit_same_k_view(
    registered_execution: RegisteredProductionFitSameK,
    *,
    run_id: str,
    binding_fingerprint: str,
) -> ProductionFitSameKView:
    """Bind the one original registered production fit without fitting again."""

    selection, design_fit, production_features, production_fit = (
        registered_production_fit_same_k_sources(registered_execution)
    )
    execution_identity = registered_production_fit_same_k_identity(registered_execution)
    _verify_selection_view(selection)
    _require_child_binding(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_run_id=selection.run_id,
        parent_binding_fingerprint=selection.binding_fingerprint,
        label="production fit",
    )
    design_fingerprint = gaussian_hmm_fit_fingerprint(design_fit)
    feature_fingerprint = hmm_feature_matrix_fingerprint(production_features)
    production_fingerprint = gaussian_hmm_fit_fingerprint(production_fit)
    payload = _production_fit_payload(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        design_selected_k=design_fit.n_states,
        design_model_fingerprint=design_fingerprint,
        selection_view_fingerprint=selection.view_fingerprint,
        production_feature_fingerprint=feature_fingerprint,
        master_seed=registered_execution.master_seed,
        scientific_version=registered_execution.scientific_version,
        rng_execution_fingerprint=execution_identity.rng_execution_fingerprint,
        restart_seed_fingerprint=(
            execution_identity.restart_seed_materialization_fingerprint
        ),
        production_fit_fingerprint=production_fingerprint,
        registered_execution_fingerprint=execution_identity.execution_fingerprint,
    )
    view = object.__new__(ProductionFitSameKView)
    for name, value in (
        ("contract_version", CALIBRATION_TASK_VIEW_CONTRACT),
        ("task_id", "production_fit_same_k"),
        ("role", "production"),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("design_selected_k", design_fit.n_states),
        ("design_model_fingerprint", design_fingerprint),
        ("selection_view_fingerprint", selection.view_fingerprint),
        ("production_feature_fingerprint", feature_fingerprint),
        ("master_seed", registered_execution.master_seed),
        ("scientific_version", registered_execution.scientific_version),
        ("rng_namespace", "production_fit_restarts"),
        ("rng_execution_fingerprint", execution_identity.rng_execution_fingerprint),
        (
            "restart_seed_fingerprint",
            execution_identity.restart_seed_materialization_fingerprint,
        ),
        ("production_fit", production_fit),
        ("production_fit_fingerprint", production_fingerprint),
        (
            "registered_execution_fingerprint",
            execution_identity.execution_fingerprint,
        ),
        ("view_fingerprint", _sha256_json(payload)),
        ("generation_authorized", False),
        ("_selection_source", selection),
        ("_production_features", production_features),
        ("_registered_execution", registered_execution),
        ("_view_seal", _PRODUCTION_FIT_VIEW_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_PRODUCTION_FIT_VIEWS[id(view)] = (view, registered_execution)
    _verify_production_fit_view(view)
    return view


def is_production_fit_same_k_view(
    value: object,
) -> TypeGuard[ProductionFitSameKView]:
    """Return whether ``value`` is an unchanged registered fixed-K fit view."""

    if not isinstance(value, ProductionFitSameKView):
        return False
    try:
        _verify_production_fit_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def production_fit_sources_from_view(
    value: ProductionFitSameKView,
) -> tuple[GaussianHMMFit, HMMFeatureMatrix]:
    """Return the exact production fit and features after full reverification."""

    if not isinstance(value, ProductionFitSameKView):
        raise ModelError("production fit sources require a production-fit view")
    _verify_production_fit_view(value)
    _, _, features, fit = registered_production_fit_same_k_sources(
        value._registered_execution
    )
    return fit, features


def registered_production_fit_execution_from_view(
    value: ProductionFitSameKView,
) -> RegisteredProductionFitSameK:
    """Return the exact registered execution retained by one guarded view."""

    if not isinstance(value, ProductionFitSameKView):
        raise ModelError("production fit execution requires a production-fit view")
    _verify_production_fit_view(value)
    return value._registered_execution


def build_macro_stability_task_view(
    *,
    task_id: MacroTaskId,
    evidence: MacroStabilityEvidence,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
    source_slice: CalibrationSlice,
    master_seed: int,
    scientific_version: int,
    run_id: str,
    binding_fingerprint: str,
) -> MacroStabilityTaskView:
    """Bind one registered 100-attempt macro execution to its exact parent."""

    expected_role = _macro_task_role(task_id)
    parent_fit, parent_view_fingerprint, parent_binding = _macro_parent(
        expected_role,
        parent,
    )
    _require_child_binding(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_run_id=parent.run_id,
        parent_binding_fingerprint=parent_binding,
        label=f"{expected_role} macro stability",
    )
    expected_slice_role = "design" if expected_role == "design" else "full"
    if source_slice.role != expected_slice_role:
        raise ModelError("macro stability source slice has the wrong role")
    execution = registered_macro_stability_execution_identity(evidence)
    parent_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)
    slice_fingerprint = calibration_slice_fingerprint(source_slice)
    macro_input_fingerprint = macro_stability_input_fingerprint(
        source_slice.macro_features,
        source_slice.monthly_period_ordinals,
        source_slice.monthly_continuation,
        role=expected_role,
    )
    expected_rng = macro_stability_rng_fingerprint(
        role=expected_role,
        n_states=parent_fit.n_states,
        rows=len(source_slice.macro_features),
        macro_block_length=evidence.macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    if (
        execution.role != expected_role
        or execution.n_states != parent_fit.n_states
        or execution.parent_model_fingerprint != parent_fingerprint
        or execution.macro_input_fingerprint != macro_input_fingerprint
        or execution.master_seed != master_seed
        or execution.scientific_version != scientific_version
        or execution.rng_namespace != macro_stability_rng_namespace(expected_role)
        or execution.rng_execution_fingerprint != expected_rng
    ):
        raise ModelError("macro stability evidence is mixed with another execution")
    evidence_fingerprint = macro_stability_evidence_fingerprint(evidence)
    payload = _macro_payload(
        task_id=task_id,
        role=expected_role,
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        selected_k=parent_fit.n_states,
        parent_model_fingerprint=parent_fingerprint,
        parent_view_fingerprint=parent_view_fingerprint,
        source_slice_fingerprint=slice_fingerprint,
        macro_input_fingerprint=macro_input_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_namespace=execution.rng_namespace,
        rng_execution_fingerprint=expected_rng,
        evidence_fingerprint=evidence_fingerprint,
    )
    view = object.__new__(MacroStabilityTaskView)
    for name, value in (
        ("contract_version", CALIBRATION_TASK_VIEW_CONTRACT),
        ("task_id", task_id),
        ("role", expected_role),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("selected_k", parent_fit.n_states),
        ("parent_model_fingerprint", parent_fingerprint),
        ("parent_view_fingerprint", parent_view_fingerprint),
        ("source_slice_fingerprint", slice_fingerprint),
        ("macro_input_fingerprint", macro_input_fingerprint),
        ("master_seed", master_seed),
        ("scientific_version", scientific_version),
        ("rng_namespace", execution.rng_namespace),
        ("rng_execution_fingerprint", expected_rng),
        ("result", evidence),
        ("evidence_fingerprint", evidence_fingerprint),
        ("view_fingerprint", _sha256_json(payload)),
        ("generation_authorized", False),
        ("_parent_view", parent),
        ("_source_slice", source_slice),
        ("_view_seal", _MACRO_VIEW_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_MACRO_VIEWS[id(view)] = view
    return view


def is_macro_stability_task_view(
    value: object,
) -> TypeGuard[MacroStabilityTaskView]:
    """Return whether ``value`` is an unchanged role-locked macro view."""

    if not isinstance(value, MacroStabilityTaskView):
        return False
    try:
        _verify_macro_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def _verify_viability_view(value: CandidateViabilityTaskView) -> None:
    if (
        value._view_seal is not _CANDIDATE_VIEW_CAPABILITY
        or _MINTED_VIABILITY_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("candidate viability view lacks builder authority")
    if (
        value.contract_version != CALIBRATION_TASK_VIEW_CONTRACT
        or value.task_id != "design_candidate_viability"
        or value.role != "design"
        or value.generation_authorized is not False
    ):
        raise ModelError("candidate viability view contract is invalid")
    _require_sha256(value.run_id, "candidate viability run ID")
    _require_sha256(value.binding_fingerprint, "candidate viability binding")
    identity = (
        verified_registered_candidate_viability_execution_identity(
            value._source_result,
            require_strict_canonical_geometry=True,
        )
        if value._canonical_suite_authority
        else verified_candidate_viability_execution_identity(value._source_result)
    )
    restart_fingerprints, rng_fingerprint = _candidate_fit_rng_identity(
        value._source_result,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
    )
    rows = _viability_rows(value._source_result)
    admissible = tuple(row.n_states for row in rows if row.viability.admissible)
    viability_fingerprint = _viability_result_fingerprint(rows)
    if (
        identity.role != "design"
        or value.fit_set_fingerprint != identity.fit_set_fingerprint
        or value.source_content_fingerprint != identity.source_content_fingerprint
        or value.viability_result_fingerprint != identity.viability_result_fingerprint
        or value.suite_fingerprint != identity.suite_fingerprint
        or value.suite_combined_materialization_fingerprint
        != identity.suite_combined_materialization_fingerprint
        or value.suite_combined_fit_rng_execution_fingerprint
        != identity.suite_combined_fit_rng_execution_fingerprint
        or value.suite_combined_rng_execution_fingerprint
        != identity.suite_combined_rng_execution_fingerprint
        or value.suite_macro_materialization_fingerprint
        != identity.suite_macro_materialization_fingerprint
        or value.suite_monthly_block_materialization_fingerprint
        != identity.suite_monthly_block_materialization_fingerprint
        or value.suite_daily_block_materialization_fingerprint
        != identity.suite_daily_block_materialization_fingerprint
        or value.fixed_point_final_iteration != identity.fixed_point_final_iteration
        or value.fixed_point_viability_projection_fingerprint
        != identity.fixed_point_viability_projection_fingerprint
        or value._final_fixed_point_authority
        is not value._source_result._final_fixed_point_authority
        or (
            value._canonical_suite_authority
            and (
                value.master_seed != identity.suite_master_seed
                or value.scientific_version != identity.suite_scientific_version
            )
        )
        or value.rng_namespace != "design_candidate_fit_restarts"
        or value.rng_execution_fingerprint != rng_fingerprint
        or value.restart_seed_fingerprints != restart_fingerprints
        or value.viability_rows != rows
        or value.admissible_n_states != admissible
        or value.passed is not (OWNER_FIXED_N_STATES in admissible)
        or value.viability_fingerprint != viability_fingerprint
    ):
        raise ModelError("candidate viability source identity changed")
    expected = _sha256_json(
        _viability_payload(
            run_id=value.run_id,
            binding_fingerprint=value.binding_fingerprint,
            identity=identity,
            admissible_n_states=admissible,
            master_seed=value.master_seed,
            scientific_version=value.scientific_version,
            rng_execution_fingerprint=rng_fingerprint,
            restart_seed_fingerprints=restart_fingerprints,
            viability_fingerprint=viability_fingerprint,
        )
    )
    if expected != value.view_fingerprint:
        raise ModelError("candidate viability view fingerprint is inconsistent")


def _verify_selection_view(value: CandidateSelectionTaskView) -> None:
    if (
        value._view_seal is not _CANDIDATE_VIEW_CAPABILITY
        or _MINTED_SELECTION_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("candidate selection view lacks builder authority")
    if (
        value.contract_version != CALIBRATION_TASK_VIEW_CONTRACT
        or value.task_id != "design_predictive_selection"
        or value.role != "design"
        or value.generation_authorized is not False
    ):
        raise ModelError("candidate selection view contract is invalid")
    _verify_viability_view(value._viability_parent)
    if (
        value.run_id != value._viability_parent.run_id
        or value.binding_fingerprint == value._viability_parent.binding_fingerprint
        or value.viability_view_fingerprint != value._viability_parent.view_fingerprint
        or value._source_result.viability_result
        is not value._viability_parent._source_result
    ):
        raise ModelError("candidate selection viability ancestry is invalid")
    identity = (
        verified_registered_candidate_gate_execution_identity(
            value._source_result,
            require_strict_canonical_geometry=True,
        )
        if value._canonical_suite_authority
        else verified_candidate_gate_execution_identity(value._source_result)
    )
    selection = value._source_result.selection
    selection_fingerprint = _selection_result_fingerprint(selection)
    if (
        identity.role != "design"
        or value.selection != selection
        or value.selected_n_states != selection.selected_n_states
        or value.passed
        is not (
            selection.passed
            if identity.fixed_point_outcome is None
            else identity.fixed_point_outcome == "converged"
        )
        or value.fit_set_fingerprint != identity.fit_set_fingerprint
        or value.rolling_fingerprints != identity.rolling_fingerprints
        or value.bootstrap_fingerprint != identity.bootstrap_fingerprint
        or value.bootstrap_materialization_fingerprint
        != identity.bootstrap_materialization_fingerprint
        or value.bootstrap_fingerprint != value.bootstrap_materialization_fingerprint
        or value.bootstrap_rng_execution_fingerprint
        != identity.bootstrap_rng_execution_fingerprint
        or identity.bootstrap_artifact is not CalibrationArtifact.DESIGN_OR_CONTROL
        or identity.bootstrap_master_seed != value._viability_parent.master_seed
        or identity.bootstrap_scientific_version
        != value._viability_parent.scientific_version
        or value.source_content_fingerprint != identity.source_content_fingerprint
        or value.viability_result_fingerprint != identity.viability_result_fingerprint
        or value._final_fixed_point_authority
        is not value._source_result._final_fixed_point_authority
        or value.selection_fingerprint != selection_fingerprint
        or value.fixed_point_outcome != identity.fixed_point_outcome
        or value.fixed_point_failure_reason != identity.fixed_point_failure_reason
        or value.fixed_point_converged_at_iteration
        != identity.fixed_point_converged_at_iteration
        or value.fixed_point_iteration_fingerprints
        != identity.fixed_point_iteration_fingerprints
        or value.fixed_point_iteration_history_fingerprint
        != identity.fixed_point_iteration_history_fingerprint
        or value.fixed_point_fingerprint != identity.fixed_point_fingerprint
        or value.fixed_point_final_selection_fingerprint
        != identity.fixed_point_final_selection_fingerprint
        or value.fixed_point_final_monthly_block_core_fingerprint
        != identity.fixed_point_final_monthly_block_core_fingerprint
        or value.fixed_point_final_daily_block_core_fingerprint
        != identity.fixed_point_final_daily_block_core_fingerprint
        or value.fixed_point_viability_projection_fingerprint
        != identity.fixed_point_viability_projection_fingerprint
    ):
        raise ModelError("candidate selection source identity changed")
    expected = _sha256_json(
        _selection_payload(
            run_id=value.run_id,
            binding_fingerprint=value.binding_fingerprint,
            identity=identity,
            viability_view_fingerprint=value.viability_view_fingerprint,
            selection_fingerprint=selection_fingerprint,
        )
    )
    if expected != value.view_fingerprint:
        raise ModelError("candidate selection view fingerprint is inconsistent")


def _verify_holdout_view(value: SealedHoldoutVetoView) -> None:
    if (
        value._view_seal is not _HOLDOUT_VIEW_CAPABILITY
        or _MINTED_HOLDOUT_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("sealed holdout view lacks builder authority")
    if (
        value.contract_version != CALIBRATION_TASK_VIEW_CONTRACT
        or value.task_id != "sealed_holdout_veto"
        or value.role != "design"
        or value.generation_authorized is not False
    ):
        raise ModelError("sealed holdout view contract is invalid")
    _verify_selection_view(value._selection_source)
    if (
        value._selection_source.task_id != "design_predictive_selection"
        or value._selection_source.run_id != value.run_id
        or value._selection_source.binding_fingerprint == value.binding_fingerprint
        or value._selection_source.view_fingerprint != value.selection_view_fingerprint
    ):
        raise ModelError("sealed holdout selection ancestry is invalid")
    fit = _selected_design_fit(value._selection_source)
    model_fingerprint = gaussian_hmm_fit_fingerprint(fit)
    design_fingerprint = calibration_slice_fingerprint(value._design_slice)
    holdout_fingerprint = calibration_slice_fingerprint(value._holdout_slice)
    recomputed = evaluate_sealed_holdout(
        fit,
        value._design_slice,
        value._holdout_slice,
    )
    evidence_fingerprint = sealed_holdout_evidence_fingerprint(value.evidence)
    if (
        value.selected_n_states != fit.n_states
        or value.selected_model_fingerprint != model_fingerprint
        or value.design_slice_fingerprint != design_fingerprint
        or value.holdout_slice_fingerprint != holdout_fingerprint
        or sealed_holdout_evidence_fingerprint(recomputed) != evidence_fingerprint
        or value.evidence_fingerprint != evidence_fingerprint
    ):
        raise ModelError("sealed holdout live evidence differs from its sources")
    expected = _sha256_json(
        _holdout_payload(
            run_id=value.run_id,
            binding_fingerprint=value.binding_fingerprint,
            selected_n_states=value.selected_n_states,
            selected_model_fingerprint=model_fingerprint,
            design_slice_fingerprint=design_fingerprint,
            holdout_slice_fingerprint=holdout_fingerprint,
            selection_view_fingerprint=value.selection_view_fingerprint,
            evidence_fingerprint=evidence_fingerprint,
        )
    )
    if expected != value.view_fingerprint:
        raise ModelError("sealed holdout view fingerprint is inconsistent")


def _verify_production_fit_view(value: ProductionFitSameKView) -> None:
    ledger = _MINTED_PRODUCTION_FIT_VIEWS.get(id(value))
    if (
        value._view_seal is not _PRODUCTION_FIT_VIEW_CAPABILITY
        or ledger is None
        or ledger[0] is not value
        or ledger[1] is not value._registered_execution
    ):
        raise ModelError("production-fit view lacks builder authority")
    if (
        value.contract_version != CALIBRATION_TASK_VIEW_CONTRACT
        or value.task_id != "production_fit_same_k"
        or value.role != "production"
        or value.rng_namespace != "production_fit_restarts"
        or value.generation_authorized is not False
    ):
        raise ModelError("production-fit view contract is invalid")
    selection, design_fit, features, production_fit = (
        registered_production_fit_same_k_sources(value._registered_execution)
    )
    execution_identity = registered_production_fit_same_k_identity(
        value._registered_execution
    )
    _verify_selection_view(selection)
    if (
        value._selection_source is not selection
        or value._production_features is not features
        or value.production_fit is not production_fit
        or selection.task_id != "design_predictive_selection"
        or selection.run_id != value.run_id
        or selection.binding_fingerprint == value.binding_fingerprint
        or selection.view_fingerprint != value.selection_view_fingerprint
    ):
        raise ModelError("production-fit selection ancestry is invalid")
    design_fingerprint = gaussian_hmm_fit_fingerprint(design_fit)
    feature_fingerprint = hmm_feature_matrix_fingerprint(features)
    production_fingerprint = gaussian_hmm_fit_fingerprint(production_fit)
    if (
        value.design_selected_k != design_fit.n_states
        or value.design_model_fingerprint != design_fingerprint
        or value.production_feature_fingerprint != feature_fingerprint
        or value.master_seed != value._registered_execution.master_seed
        or value.scientific_version != value._registered_execution.scientific_version
        or value.restart_seed_fingerprint
        != execution_identity.restart_seed_materialization_fingerprint
        or value.rng_execution_fingerprint
        != execution_identity.rng_execution_fingerprint
        or value.production_fit_fingerprint != production_fingerprint
        or value.registered_execution_fingerprint
        != execution_identity.execution_fingerprint
    ):
        raise ModelError("production-fit live sources differ from their bindings")
    expected = _sha256_json(
        _production_fit_payload(
            run_id=value.run_id,
            binding_fingerprint=value.binding_fingerprint,
            design_selected_k=value.design_selected_k,
            design_model_fingerprint=design_fingerprint,
            selection_view_fingerprint=value.selection_view_fingerprint,
            production_feature_fingerprint=feature_fingerprint,
            master_seed=value.master_seed,
            scientific_version=value.scientific_version,
            rng_execution_fingerprint=execution_identity.rng_execution_fingerprint,
            restart_seed_fingerprint=(
                execution_identity.restart_seed_materialization_fingerprint
            ),
            production_fit_fingerprint=production_fingerprint,
            registered_execution_fingerprint=execution_identity.execution_fingerprint,
        )
    )
    if expected != value.view_fingerprint:
        raise ModelError("production-fit view fingerprint is inconsistent")


def _verify_macro_view(value: MacroStabilityTaskView) -> None:
    if (
        value._view_seal is not _MACRO_VIEW_CAPABILITY
        or _MINTED_MACRO_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("macro-stability view lacks builder authority")
    expected_role = _macro_task_role(value.task_id)
    if (
        value.contract_version != CALIBRATION_TASK_VIEW_CONTRACT
        or value.role != expected_role
        or value.generation_authorized is not False
    ):
        raise ModelError("macro-stability view contract is invalid")
    parent_fit, parent_view_fingerprint, parent_binding = _macro_parent(
        expected_role,
        value._parent_view,
    )
    if (
        value._parent_view.run_id != value.run_id
        or parent_binding == value.binding_fingerprint
        or parent_view_fingerprint != value.parent_view_fingerprint
    ):
        raise ModelError("macro-stability parent ancestry is invalid")
    execution = registered_macro_stability_execution_identity(value.result)
    parent_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)
    slice_fingerprint = calibration_slice_fingerprint(value._source_slice)
    input_fingerprint = macro_stability_input_fingerprint(
        value._source_slice.macro_features,
        value._source_slice.monthly_period_ordinals,
        value._source_slice.monthly_continuation,
        role=expected_role,
    )
    rng_fingerprint = macro_stability_rng_fingerprint(
        role=expected_role,
        n_states=parent_fit.n_states,
        rows=len(value._source_slice.macro_features),
        macro_block_length=value.result.macro_block_length,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
    )
    evidence_fingerprint = macro_stability_evidence_fingerprint(value.result)
    if (
        value.selected_k != parent_fit.n_states
        or value.parent_model_fingerprint != parent_fingerprint
        or value.source_slice_fingerprint != slice_fingerprint
        or value.macro_input_fingerprint != input_fingerprint
        or value.rng_namespace != macro_stability_rng_namespace(expected_role)
        or value.rng_execution_fingerprint != rng_fingerprint
        or value.evidence_fingerprint != evidence_fingerprint
        or execution.role != expected_role
        or execution.n_states != parent_fit.n_states
        or execution.parent_model_fingerprint != parent_fingerprint
        or execution.macro_input_fingerprint != input_fingerprint
        or execution.master_seed != value.master_seed
        or execution.scientific_version != value.scientific_version
        or execution.rng_execution_fingerprint != rng_fingerprint
    ):
        raise ModelError("macro-stability live sources differ from their bindings")
    expected = _sha256_json(
        _macro_payload(
            task_id=value.task_id,
            role=expected_role,
            run_id=value.run_id,
            binding_fingerprint=value.binding_fingerprint,
            selected_k=value.selected_k,
            parent_model_fingerprint=parent_fingerprint,
            parent_view_fingerprint=parent_view_fingerprint,
            source_slice_fingerprint=slice_fingerprint,
            macro_input_fingerprint=input_fingerprint,
            master_seed=value.master_seed,
            scientific_version=value.scientific_version,
            rng_namespace=value.rng_namespace,
            rng_execution_fingerprint=rng_fingerprint,
            evidence_fingerprint=evidence_fingerprint,
        )
    )
    if expected != value.view_fingerprint:
        raise ModelError("macro-stability view fingerprint is inconsistent")


def _macro_parent(
    role: ArtifactRole,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
) -> tuple[GaussianHMMFit, str, str]:
    if role == "design":
        if not isinstance(parent, CandidateSelectionTaskView):
            raise ModelError("design macro stability requires candidate selection")
        _verify_selection_view(parent)
        fit = _selected_design_fit(parent)
        return fit, parent.view_fingerprint, parent.binding_fingerprint
    if not isinstance(parent, ProductionFitSameKView):
        raise ModelError("production macro stability requires production-fit view")
    _verify_production_fit_view(parent)
    return (
        parent.production_fit,
        parent.view_fingerprint,
        parent.binding_fingerprint,
    )


def _selected_design_fit(selection: CandidateSelectionTaskView) -> GaussianHMMFit:
    _verify_selection_view(selection)
    fit = selection._source_result.selected_fit
    if not isinstance(fit, GaussianHMMFit):
        raise ModelError("candidate selection has no frozen selected design fit")
    if fit.n_states != selection.selected_n_states:
        raise ModelError("candidate selection model and selected K disagree")
    return fit


def _candidate_fit_rng_identity(
    result: CandidateViabilityResult,
    *,
    master_seed: int,
    scientific_version: int,
) -> tuple[tuple[str, str, str, str], str]:
    fit_set = result.fit_set
    if (
        fit_set.scope is not ModelFitScope.DESIGN_MAIN
        or fit_set.replicate != 0
        or tuple(item.n_states for item in fit_set.attempts) != CANONICAL_CANDIDATES
    ):
        raise ModelError("candidate viability fit scope is not design-main replicate 0")
    fingerprints: list[str] = []
    for attempt in fit_set.attempts:
        seeds = deterministic_hmm_restart_seeds(
            master_seed=master_seed,
            scientific_version=scientific_version,
            scope=ModelFitScope.DESIGN_MAIN,
            replicate=0,
            n_states=attempt.n_states,
        )
        fingerprints.append(
            scientific_array_fingerprint(
                np.asarray(seeds, dtype=np.uint32),
                role=f"design_candidate_k{attempt.n_states}_restart_seeds",
            )
        )
        fit = attempt.fit
        if fit is None or not fit.restart_diagnostics:
            continue
        diagnostics = fit.restart_diagnostics
        if (
            len(diagnostics) != HMM_RESTARTS
            or tuple(item.restart_index for item in diagnostics)
            != tuple(range(HMM_RESTARTS))
            or tuple(item.seed for item in diagnostics) != seeds
            or fit.best_restart_index not in range(HMM_RESTARTS)
            or fit.best_seed != seeds[fit.best_restart_index]
        ):
            raise ModelError("candidate fit diagnostics disagree with registered seeds")
    checked = cast(tuple[str, str, str, str], tuple(fingerprints))
    execution = rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="design_candidate_fit_restarts",
        contract={
            "candidate_state_counts": list(CANONICAL_CANDIDATES),
            "model_fit_scope": int(ModelFitScope.DESIGN_MAIN),
            "replicate": 0,
            "restart_seed_fingerprints": list(checked),
            "restarts_per_candidate": HMM_RESTARTS,
            "stage": int(Stage.HMM_LIBRARY_SEED),
        },
    )
    return checked, execution


def _viability_rows(
    result: CandidateViabilityResult,
) -> tuple[
    CandidateViabilityTaskRow,
    CandidateViabilityTaskRow,
    CandidateViabilityTaskRow,
    CandidateViabilityTaskRow,
]:
    if tuple(row.n_states for row in result.audit_rows) != CANONICAL_CANDIDATES:
        raise ModelError("candidate viability source lacks exact K=2,3,4,5 rows")
    return cast(
        tuple[
            CandidateViabilityTaskRow,
            CandidateViabilityTaskRow,
            CandidateViabilityTaskRow,
            CandidateViabilityTaskRow,
        ],
        tuple(
            CandidateViabilityTaskRow(
                row.n_states,
                row.fit_succeeded,
                row.viability,
            )
            for row in result.audit_rows
        ),
    )


def _viability_result_fingerprint(
    rows: tuple[
        CandidateViabilityTaskRow,
        CandidateViabilityTaskRow,
        CandidateViabilityTaskRow,
        CandidateViabilityTaskRow,
    ],
) -> str:
    return _sha256_json(
        {
            "schema_id": "candidate_viability_task_result_v1",
            "rows": [
                {
                    "n_states": row.n_states,
                    "fit_succeeded": row.fit_succeeded,
                    "viability": row.viability.as_dict(),
                }
                for row in rows
            ],
        }
    )


def _selection_result_fingerprint(selection: CandidateSelectionReport) -> str:
    return _sha256_json(
        {
            "schema_id": "candidate_selection_task_result_v1",
            "selection": selection.as_dict(),
        }
    )


def _viability_payload(
    *,
    run_id: str,
    binding_fingerprint: str,
    identity: CandidateViabilityExecutionIdentity,
    admissible_n_states: tuple[int, ...],
    master_seed: int,
    scientific_version: int,
    rng_execution_fingerprint: str,
    restart_seed_fingerprints: tuple[str, str, str, str],
    viability_fingerprint: str,
) -> dict[str, object]:
    return {
        "contract_version": CALIBRATION_TASK_VIEW_CONTRACT,
        "task_id": "design_candidate_viability",
        "role": "design",
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        "fit_set_fingerprint": identity.fit_set_fingerprint,
        "source_content_fingerprint": identity.source_content_fingerprint,
        "viability_result_fingerprint": identity.viability_result_fingerprint,
        "suite_fingerprint": identity.suite_fingerprint,
        "suite_combined_materialization_fingerprint": (
            identity.suite_combined_materialization_fingerprint
        ),
        "suite_combined_fit_rng_execution_fingerprint": (
            identity.suite_combined_fit_rng_execution_fingerprint
        ),
        "suite_combined_rng_execution_fingerprint": (
            identity.suite_combined_rng_execution_fingerprint
        ),
        "suite_macro_materialization_fingerprint": (
            identity.suite_macro_materialization_fingerprint
        ),
        "suite_monthly_block_materialization_fingerprint": (
            identity.suite_monthly_block_materialization_fingerprint
        ),
        "suite_daily_block_materialization_fingerprint": (
            identity.suite_daily_block_materialization_fingerprint
        ),
        "fixed_point_final_iteration": identity.fixed_point_final_iteration,
        "fixed_point_viability_projection_fingerprint": (
            identity.fixed_point_viability_projection_fingerprint
        ),
        "admissible_n_states": list(admissible_n_states),
        "passed": OWNER_FIXED_N_STATES in admissible_n_states,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_namespace": "design_candidate_fit_restarts",
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "restart_seed_fingerprints": list(restart_seed_fingerprints),
        "viability_fingerprint": viability_fingerprint,
        "generation_authorized": False,
    }


def _selection_payload(
    *,
    run_id: str,
    binding_fingerprint: str,
    identity: CandidateGateExecutionIdentity,
    viability_view_fingerprint: str,
    selection_fingerprint: str,
) -> dict[str, object]:
    return {
        "contract_version": CALIBRATION_TASK_VIEW_CONTRACT,
        "task_id": "design_predictive_selection",
        "role": "design",
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        "fit_set_fingerprint": identity.fit_set_fingerprint,
        "rolling_fingerprints": list(identity.rolling_fingerprints),
        "bootstrap_fingerprint": identity.bootstrap_fingerprint,
        "bootstrap_materialization_fingerprint": (
            identity.bootstrap_materialization_fingerprint
        ),
        "bootstrap_rng_execution_fingerprint": (
            identity.bootstrap_rng_execution_fingerprint
        ),
        "source_content_fingerprint": identity.source_content_fingerprint,
        "viability_view_fingerprint": viability_view_fingerprint,
        "viability_result_fingerprint": identity.viability_result_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "fixed_point_outcome": identity.fixed_point_outcome,
        "fixed_point_failure_reason": identity.fixed_point_failure_reason,
        "fixed_point_converged_at_iteration": (
            identity.fixed_point_converged_at_iteration
        ),
        "fixed_point_iteration_fingerprints": list(
            identity.fixed_point_iteration_fingerprints
        ),
        "fixed_point_iteration_history_fingerprint": (
            identity.fixed_point_iteration_history_fingerprint
        ),
        "fixed_point_fingerprint": identity.fixed_point_fingerprint,
        "fixed_point_final_selection_fingerprint": (
            identity.fixed_point_final_selection_fingerprint
        ),
        "fixed_point_final_monthly_block_core_fingerprint": (
            identity.fixed_point_final_monthly_block_core_fingerprint
        ),
        "fixed_point_final_daily_block_core_fingerprint": (
            identity.fixed_point_final_daily_block_core_fingerprint
        ),
        "fixed_point_viability_projection_fingerprint": (
            identity.fixed_point_viability_projection_fingerprint
        ),
        "generation_authorized": False,
    }


def _holdout_payload(
    *,
    run_id: str,
    binding_fingerprint: str,
    selected_n_states: int,
    selected_model_fingerprint: str,
    design_slice_fingerprint: str,
    holdout_slice_fingerprint: str,
    selection_view_fingerprint: str,
    evidence_fingerprint: str,
) -> dict[str, object]:
    return {
        "contract_version": CALIBRATION_TASK_VIEW_CONTRACT,
        "task_id": "sealed_holdout_veto",
        "role": "design",
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        "selected_n_states": selected_n_states,
        "selected_model_fingerprint": selected_model_fingerprint,
        "design_slice_fingerprint": design_slice_fingerprint,
        "holdout_slice_fingerprint": holdout_slice_fingerprint,
        "selection_view_fingerprint": selection_view_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "generation_authorized": False,
    }


def _production_fit_payload(
    *,
    run_id: str,
    binding_fingerprint: str,
    design_selected_k: int,
    design_model_fingerprint: str,
    selection_view_fingerprint: str,
    production_feature_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    rng_execution_fingerprint: str,
    restart_seed_fingerprint: str,
    production_fit_fingerprint: str,
    registered_execution_fingerprint: str,
) -> dict[str, object]:
    return {
        "contract_version": CALIBRATION_TASK_VIEW_CONTRACT,
        "task_id": "production_fit_same_k",
        "role": "production",
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        "design_selected_k": design_selected_k,
        "design_model_fingerprint": design_model_fingerprint,
        "selection_view_fingerprint": selection_view_fingerprint,
        "production_feature_fingerprint": production_feature_fingerprint,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_namespace": "production_fit_restarts",
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "restart_seed_fingerprint": restart_seed_fingerprint,
        "production_fit_fingerprint": production_fit_fingerprint,
        "registered_execution_fingerprint": registered_execution_fingerprint,
        "generation_authorized": False,
    }


def _macro_payload(
    *,
    task_id: MacroTaskId,
    role: ArtifactRole,
    run_id: str,
    binding_fingerprint: str,
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
) -> dict[str, object]:
    return {
        "contract_version": CALIBRATION_TASK_VIEW_CONTRACT,
        "task_id": task_id,
        "role": role,
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
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
        "generation_authorized": False,
    }


def _macro_task_role(task_id: str) -> ArtifactRole:
    role = _MACRO_TASK_ROLE.get(task_id)
    if role is None:
        raise ModelError("macro stability task ID is outside the closed registry")
    return cast(ArtifactRole, role)


def _require_child_binding(
    *,
    run_id: str,
    binding_fingerprint: str,
    parent_run_id: str,
    parent_binding_fingerprint: str,
    label: str,
) -> None:
    _require_sha256(run_id, f"{label} run ID")
    _require_sha256(binding_fingerprint, f"{label} binding fingerprint")
    if run_id != parent_run_id:
        raise ModelError(f"{label} parent belongs to another run")
    if binding_fingerprint == parent_binding_fingerprint:
        raise ModelError(f"{label} binding must differ from its parent task")


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"{label} must be an exact SHA-256 fingerprint")


def _sha256_json(value: object) -> str:
    try:
        content = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "calibration task view content is not canonical JSON"
        ) from error
    return hashlib.sha256(content).hexdigest()
