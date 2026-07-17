"""Authoritative four-cell Holm gate for Phase-3 model adequacy."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeGuard

import numpy as np

from prpg.errors import ModelError
from prpg.model.adequacy import four_cell_holm
from prpg.model.adequacy_execution import (
    RegisteredLatentCell,
    is_registered_latent_cell,
    registered_latent_cell_identity,
    registered_latent_cell_source_fit,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    ProductionFitSameKView,
    production_fit_sources_from_view,
    selected_design_fit_from_view,
)
from prpg.model.inference import HolmResult
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.refit_adequacy_execution import (
    RegisteredRefittedAdequacyBatch,
    is_registered_refitted_adequacy_batch,
    registered_refitted_adequacy_batch_identity,
    registered_refitted_adequacy_sources,
)
from prpg.simulation.rng import CalibrationArtifact

AdequacyCellId: TypeAlias = Literal[
    "design_latent",
    "production_latent",
    "design_refitted_adequacy",
    "production_refitted_adequacy",
]
AdequacyCellTaskId: TypeAlias = Literal[
    "design_latent_correctness",
    "production_latent_correctness",
    "design_refitted_adequacy",
    "production_refitted_adequacy",
]
AdequacyRole: TypeAlias = Literal["design", "production"]

ADEQUACY_TASK_VIEW_CONTRACT = "sections-4.4-4.6-adequacy-task-view-v1"
FOUR_CELL_ADEQUACY_HOLM_TASK_ID = "four_cell_adequacy_holm"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADEQUACY_TASK_VIEW_CAPABILITY = object()
_ADEQUACY_HOLM_VIEW_CAPABILITY = object()
_FOUR_CELL_GATE_CAPABILITY = object()

# A copied Python object otherwise carries the same private capability object.
# These small process-local ledgers make authority object-specific.  Strong
# references are intentional: at most a handful of adequacy views exist in one
# calibration process and an object identifier must not be reused after GC.
_MINTED_CELL_VIEWS: dict[int, object] = {}
_MINTED_HOLM_VIEWS: dict[int, object] = {}
_MINTED_FOUR_CELL_GATES: dict[int, object] = {}

ADEQUACY_CELL_ORDER: tuple[AdequacyCellId, ...] = (
    "design_latent",
    "production_latent",
    "design_refitted_adequacy",
    "production_refitted_adequacy",
)

ADEQUACY_TASK_ORDER: tuple[
    AdequacyCellTaskId,
    AdequacyCellTaskId,
    AdequacyCellTaskId,
    AdequacyCellTaskId,
] = (
    "design_latent_correctness",
    "production_latent_correctness",
    "design_refitted_adequacy",
    "production_refitted_adequacy",
)

_TASK_CELL_ROLE: dict[AdequacyCellTaskId, tuple[AdequacyCellId, AdequacyRole]] = {
    "design_latent_correctness": ("design_latent", "design"),
    "production_latent_correctness": ("production_latent", "production"),
    "design_refitted_adequacy": ("design_refitted_adequacy", "design"),
    "production_refitted_adequacy": (
        "production_refitted_adequacy",
        "production",
    ),
}


@dataclass(frozen=True, slots=True)
class AdequacyCellRecord:
    """One cell's fixed-denominator readiness and empirical p-value."""

    cell_id: AdequacyCellId
    ready: bool
    p_value: float | None
    repetitions: int
    successful_repetitions: int
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class AdequacyCellTaskView:
    """Process-authorized, one-cell evidence for one exact G3 task.

    The registered execution object is deliberately not retained.  This view
    contains only its own compact cell decision and the exact model, input,
    RNG, run, and task bindings needed to prevent evidence substitution.
    """

    contract_version: str
    task_id: AdequacyCellTaskId
    role: AdequacyRole
    run_id: str
    binding_fingerprint: str
    parent_view_fingerprint: str
    parent_model_fingerprint: str
    scientific_input_fingerprint: str
    rng_execution_fingerprint: str
    simulation_rng_execution_fingerprint: str | None
    bootstrap_rng_execution_fingerprint: str | None
    source_evidence_fingerprint: str
    cell: AdequacyCellRecord
    canonical_count_ready: bool
    ready: bool
    passed: bool
    strict_canonical: bool
    content_fingerprint: str
    _source_evidence: RegisteredLatentCell | RegisteredRefittedAdequacyBatch = field(
        repr=False,
        compare=False,
    )
    _parent_view: CandidateSelectionTaskView | ProductionFitSameKView = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class FourCellAdequacyHolmView:
    """Authoritative Holm task assembled from four authorized cell views."""

    contract_version: str
    task_id: Literal["four_cell_adequacy_holm"]
    run_id: str
    binding_fingerprint: str
    source_task_order: tuple[
        AdequacyCellTaskId,
        AdequacyCellTaskId,
        AdequacyCellTaskId,
        AdequacyCellTaskId,
    ]
    source_views: tuple[
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
    ]
    source_content_fingerprints: tuple[str, str, str, str]
    cell_order: tuple[AdequacyCellId, ...]
    holm: HolmResult | None
    ready: bool
    passed: bool
    failure_reasons: tuple[str, ...]
    alpha: float
    strict_canonical: bool
    content_fingerprint: str
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def cells(
        self,
    ) -> tuple[
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
    ]:
        """Return the four compact cells in the preregistered Holm order."""

        return (
            self.source_views[0].cell,
            self.source_views[1].cell,
            self.source_views[2].cell,
            self.source_views[3].cell,
        )


@dataclass(frozen=True, slots=True)
class FourCellAdequacyGate:
    """Only complete four-cell evidence may become a Holm decision."""

    cell_order: tuple[AdequacyCellId, ...]
    cells: tuple[
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
    ]
    holm: HolmResult | None
    ready: bool
    passed: bool
    failure_reasons: tuple[str, ...]
    strict_canonical: bool
    content_fingerprint: str = ""
    alpha: float = 0.05
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


def _parent_sources(
    role: AdequacyRole,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
) -> tuple[Any, Any | None, str]:
    if role == "design":
        if not isinstance(parent, CandidateSelectionTaskView):
            raise ModelError("design adequacy requires its candidate-selection parent")
        fit = selected_design_fit_from_view(parent)
        return fit, None, parent.view_fingerprint
    if role == "production":
        if not isinstance(parent, ProductionFitSameKView):
            raise ModelError("production adequacy requires its production-fit parent")
        fit, features = production_fit_sources_from_view(parent)
        return fit, features, parent.view_fingerprint
    raise ModelError("adequacy parent role is invalid")


def _validate_parent_task_binding(
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
    *,
    run_id: str,
    child_binding_fingerprint: str,
) -> None:
    if parent.run_id != run_id:
        raise ModelError("adequacy task cannot cross parent run boundaries")
    if parent.binding_fingerprint == child_binding_fingerprint:
        raise ModelError("adequacy task binding must differ from its parent binding")


def build_latent_adequacy_task_view(
    *,
    task_id: Literal["design_latent_correctness", "production_latent_correctness"],
    evidence: RegisteredLatentCell,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
    run_id: str,
    binding_fingerprint: str,
    strict_canonical: bool = True,
) -> AdequacyCellTaskView:
    """Build one role-locked latent-correctness task view.

    ``run_id`` and ``binding_fingerprint`` are expected to come from the
    verified :class:`ScientificTaskBinding` owned by ``task_id``.  Keeping
    those values explicit leaves this lower statistical layer independent of
    the durable G3 stores while still making cross-run mixing impossible.
    """

    _validate_view_inputs(task_id, run_id, binding_fingerprint, strict_canonical)
    if task_id not in {
        "design_latent_correctness",
        "production_latent_correctness",
    }:
        raise ModelError("latent adequacy view uses a non-latent task ID")
    if not is_registered_latent_cell(evidence):
        raise ModelError(f"{task_id} is not registered execution evidence")
    identity = registered_latent_cell_identity(evidence)
    source_fit = registered_latent_cell_source_fit(evidence)
    cell_id, role = _TASK_CELL_ROLE[task_id]
    parent_fit, _parent_features, parent_view_fingerprint = _parent_sources(
        role,
        parent,
    )
    _validate_parent_task_binding(
        parent,
        run_id=run_id,
        child_binding_fingerprint=binding_fingerprint,
    )
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)
    transition_fingerprint = scientific_array_fingerprint(
        parent_fit.parameters.transition_matrix,
        role="latent_transition_matrix",
    )
    if (
        source_fit is not parent_fit
        or identity.parent_model_fingerprint != parent_model_fingerprint
        or identity.transition_matrix_fingerprint != transition_fingerprint
    ):
        raise ModelError(f"{task_id} does not descend from its exact parent view")
    expected_artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    if evidence.artifact is not expected_artifact:
        raise ModelError(f"{task_id} uses future or cross-role latent evidence")
    if not isinstance(cell_id, str) or "latent" not in cell_id:
        raise AssertionError("latent task registry is inconsistent")
    record = _latent_record(
        cell_id,  # type: ignore[arg-type]
        evidence,
        expected_artifact=expected_artifact,
        strict_canonical=strict_canonical,
    )
    source_fingerprint = identity.source_evidence_fingerprint
    canonical_count_ready = _latent_canonical_geometry(evidence)
    return _mint_cell_task_view(
        task_id=task_id,
        role=role,
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_view_fingerprint=parent_view_fingerprint,
        parent_model_fingerprint=parent_model_fingerprint,
        scientific_input_fingerprint=identity.transition_matrix_fingerprint,
        rng_execution_fingerprint=identity.rng_execution_fingerprint,
        simulation_rng_execution_fingerprint=(
            identity.simulation_rng_execution_fingerprint
        ),
        bootstrap_rng_execution_fingerprint=(
            identity.bootstrap_rng_execution_fingerprint
        ),
        source_evidence_fingerprint=source_fingerprint,
        cell=record,
        canonical_count_ready=canonical_count_ready,
        strict_canonical=strict_canonical,
        source_evidence=evidence,
        parent_view=parent,
    )


def build_refitted_adequacy_task_view(
    *,
    task_id: Literal["design_refitted_adequacy", "production_refitted_adequacy"],
    evidence: RegisteredRefittedAdequacyBatch,
    parent: CandidateSelectionTaskView | ProductionFitSameKView,
    run_id: str,
    binding_fingerprint: str,
    strict_canonical: bool = True,
) -> AdequacyCellTaskView:
    """Build one role-locked refitted-adequacy task view."""

    _validate_view_inputs(task_id, run_id, binding_fingerprint, strict_canonical)
    if task_id not in {
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    }:
        raise ModelError("refitted adequacy view uses a non-refitted task ID")
    if not is_registered_refitted_adequacy_batch(evidence):
        raise ModelError(f"{task_id} is not registered execution evidence")
    identity = registered_refitted_adequacy_batch_identity(evidence)
    cell_id, role = _TASK_CELL_ROLE[task_id]
    if evidence.role != role:
        raise ModelError(f"{task_id} uses future or cross-role refitted evidence")
    if not isinstance(cell_id, str) or "refitted" not in cell_id:
        raise AssertionError("refitted task registry is inconsistent")
    parent_fit, parent_features, parent_view_fingerprint = _parent_sources(
        role,
        parent,
    )
    _validate_parent_task_binding(
        parent,
        run_id=run_id,
        child_binding_fingerprint=binding_fingerprint,
    )
    source_fit, source_features = registered_refitted_adequacy_sources(evidence)
    if source_fit is not parent_fit:
        raise ModelError(f"{task_id} does not retain its exact parent fit")
    if role == "production" and source_features is not parent_features:
        raise ModelError(f"{task_id} does not retain its exact parent features")
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)
    feature_fingerprint = hmm_feature_matrix_fingerprint(source_features)
    if (
        identity.parent_model_fingerprint != parent_model_fingerprint
        or identity.artifact_feature_fingerprint != feature_fingerprint
    ):
        raise ModelError(f"{task_id} source identities differ from its parent")
    record = _refit_record(
        cell_id,  # type: ignore[arg-type]
        evidence,
        expected_role=role,
        strict_canonical=strict_canonical,
    )
    source_fingerprint = identity.source_evidence_fingerprint
    canonical_count_ready = _refit_canonical_count(evidence)
    return _mint_cell_task_view(
        task_id=task_id,
        role=role,
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_view_fingerprint=parent_view_fingerprint,
        parent_model_fingerprint=parent_model_fingerprint,
        scientific_input_fingerprint=feature_fingerprint,
        rng_execution_fingerprint=identity.rng_execution_fingerprint,
        simulation_rng_execution_fingerprint=None,
        bootstrap_rng_execution_fingerprint=None,
        source_evidence_fingerprint=source_fingerprint,
        cell=record,
        canonical_count_ready=canonical_count_ready,
        strict_canonical=strict_canonical,
        source_evidence=evidence,
        parent_view=parent,
    )


def is_adequacy_cell_task_view(
    value: object,
) -> TypeGuard[AdequacyCellTaskView]:
    """Return whether ``value`` is the unchanged object minted by a builder."""

    if not isinstance(value, AdequacyCellTaskView):
        return False
    try:
        _verify_cell_task_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def assemble_four_cell_adequacy_holm_view(
    *,
    design_latent: AdequacyCellTaskView,
    production_latent: AdequacyCellTaskView,
    design_refitted: AdequacyCellTaskView,
    production_refitted: AdequacyCellTaskView,
    binding_fingerprint: str,
    alpha: float = 0.05,
    strict_canonical: bool = True,
) -> FourCellAdequacyHolmView:
    """Recompute authoritative Holm from four exact sealed task views.

    No source pass flag or source ordering supplied by the caller is trusted.
    Each source view is revalidated, exact task IDs are enforced in the
    preregistered cell order, all four must belong to one run, and Holm is
    recomputed from the four retained p-values.
    """

    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    _require_sha256(binding_fingerprint, "Holm task binding fingerprint")
    sources: tuple[
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
    ] = (
        design_latent,
        production_latent,
        design_refitted,
        production_refitted,
    )
    for source in sources:
        _verify_cell_task_view(source)
    actual_order = tuple(source.task_id for source in sources)
    if actual_order != ADEQUACY_TASK_ORDER:
        raise ModelError("adequacy Holm sources do not have exact task roles/order")
    run_ids = {source.run_id for source in sources}
    if len(run_ids) != 1:
        raise ModelError("adequacy Holm cannot mix evidence from different runs")
    source_bindings = tuple(source.binding_fingerprint for source in sources)
    if len(set(source_bindings)) != 4:
        raise ModelError("adequacy Holm requires four distinct task bindings")
    if binding_fingerprint in source_bindings:
        raise ModelError("adequacy Holm binding must be distinct from source tasks")
    if any(source.strict_canonical is not strict_canonical for source in sources):
        raise ModelError("adequacy Holm strict mode differs from a source task")
    run_id = sources[0].run_id
    cells = (
        sources[0].cell,
        sources[1].cell,
        sources[2].cell,
        sources[3].cell,
    )
    holm, ready, passed, failure_reasons = _authoritative_holm(
        cells,
        source_passed=(
            sources[0].passed,
            sources[1].passed,
            sources[2].passed,
            sources[3].passed,
        ),
        alpha=alpha,
    )
    source_content_fingerprints = (
        sources[0].content_fingerprint,
        sources[1].content_fingerprint,
        sources[2].content_fingerprint,
        sources[3].content_fingerprint,
    )
    content_fingerprint = _holm_view_fingerprint(
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        sources=sources,
        source_content_fingerprints=source_content_fingerprints,
        holm=holm,
        ready=ready,
        passed=passed,
        failure_reasons=failure_reasons,
        alpha=alpha,
        strict_canonical=strict_canonical,
    )
    result = FourCellAdequacyHolmView(
        ADEQUACY_TASK_VIEW_CONTRACT,
        "four_cell_adequacy_holm",
        run_id,
        binding_fingerprint,
        ADEQUACY_TASK_ORDER,
        sources,
        source_content_fingerprints,
        ADEQUACY_CELL_ORDER,
        holm,
        ready,
        passed,
        failure_reasons,
        float(alpha),
        strict_canonical,
        content_fingerprint,
    )
    object.__setattr__(result, "_execution_seal", _ADEQUACY_HOLM_VIEW_CAPABILITY)
    _MINTED_HOLM_VIEWS[id(result)] = result
    return result


def is_four_cell_adequacy_holm_view(
    value: object,
) -> TypeGuard[FourCellAdequacyHolmView]:
    """Return whether a Holm view is authorized and internally unchanged."""

    if not isinstance(value, FourCellAdequacyHolmView):
        return False
    try:
        _verify_holm_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_four_cell_adequacy_gate(
    value: object,
) -> TypeGuard[FourCellAdequacyGate]:
    """Return whether the compatibility aggregate was minted and unchanged."""

    if not isinstance(value, FourCellAdequacyGate):
        return False
    try:
        _verify_compatibility_gate(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def assemble_four_cell_adequacy_gate(
    *,
    design_latent: RegisteredLatentCell,
    production_latent: RegisteredLatentCell,
    design_refitted: RegisteredRefittedAdequacyBatch,
    production_refitted: RegisteredRefittedAdequacyBatch,
    alpha: float = 0.05,
    strict_canonical: bool = True,
) -> FourCellAdequacyGate:
    """Validate exact cell roles/counts, then Holm-correct their p-values."""

    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    design_latent_record = _latent_record(
        "design_latent",
        design_latent,
        expected_artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        strict_canonical=strict_canonical,
    )
    production_latent_record = _latent_record(
        "production_latent",
        production_latent,
        expected_artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        strict_canonical=strict_canonical,
    )
    design_refit_record = _refit_record(
        "design_refitted_adequacy",
        design_refitted,
        expected_role="design",
        strict_canonical=strict_canonical,
    )
    production_refit_record = _refit_record(
        "production_refitted_adequacy",
        production_refitted,
        expected_role="production",
        strict_canonical=strict_canonical,
    )
    cells = (
        design_latent_record,
        production_latent_record,
        design_refit_record,
        production_refit_record,
    )
    holm, ready, passed, failure_reasons = _authoritative_holm(
        cells,
        source_passed=(
            cells[0].ready,
            cells[1].ready,
            cells[2].ready,
            cells[3].ready,
        ),
        alpha=alpha,
    )
    fingerprint = _compatibility_gate_fingerprint(
        cells=cells,
        holm=holm,
        ready=ready,
        passed=passed,
        failure_reasons=failure_reasons,
        alpha=alpha,
        strict_canonical=strict_canonical,
    )
    result = FourCellAdequacyGate(
        ADEQUACY_CELL_ORDER,
        cells,
        holm,
        ready,
        passed,
        failure_reasons,
        strict_canonical,
        fingerprint,
        float(alpha),
    )
    object.__setattr__(result, "_execution_seal", _FOUR_CELL_GATE_CAPABILITY)
    _MINTED_FOUR_CELL_GATES[id(result)] = result
    return result


def _latent_record(
    cell_id: Literal["design_latent", "production_latent"],
    cell: RegisteredLatentCell,
    *,
    expected_artifact: CalibrationArtifact,
    strict_canonical: bool,
) -> AdequacyCellRecord:
    if not is_registered_latent_cell(cell):
        raise ModelError(f"{cell_id} is not registered execution evidence")
    bootstrap = cell.bootstrap
    simulation = cell.simulation
    if cell.artifact is not expected_artifact:
        raise ModelError(f"{cell_id} uses the wrong registered artifact")
    consistent = bool(
        cell.chain_streams == simulation.chain_count == bootstrap.chain_count
        and cell.bootstrap_streams == bootstrap.bootstrap_replicates
        and cell.state_uniforms_per_chain == simulation.uniforms_per_chain
        and cell.resampled_ids_per_bootstrap == bootstrap.resample_size
        and bootstrap.resample_size == bootstrap.chain_count
        and bootstrap.measured_months == simulation.measured_months
        and bootstrap.extension_months == simulation.extension_months
        and math.isfinite(float(bootstrap.p_value))
        and 0.0 <= bootstrap.p_value <= 1.0
    )
    canonical = bool(
        simulation.chain_count == 10_000
        and simulation.measured_months == 600
        and simulation.extension_months == 2_000
        and simulation.uniforms_per_chain == 2_601
        and bootstrap.bootstrap_replicates == 10_000
        and bootstrap.resample_size == 10_000
        and bootstrap.survival_months == 12
    )
    ready = consistent and (canonical or not strict_canonical)
    reason = None
    if not consistent:
        reason = "latent_evidence_inconsistent"
    elif strict_canonical and not canonical:
        reason = "latent_canonical_geometry_or_count_mismatch"
    return AdequacyCellRecord(
        cell_id,
        ready,
        float(bootstrap.p_value) if ready else None,
        bootstrap.bootstrap_replicates,
        bootstrap.bootstrap_replicates if ready else 0,
        reason,
    )


def _refit_record(
    cell_id: Literal["design_refitted_adequacy", "production_refitted_adequacy"],
    cell: RegisteredRefittedAdequacyBatch,
    *,
    expected_role: Literal["design", "production"],
    strict_canonical: bool,
) -> AdequacyCellRecord:
    if not is_registered_refitted_adequacy_batch(cell):
        raise ModelError(f"{cell_id} is not registered execution evidence")
    if cell.role != expected_role:
        raise ModelError(f"{cell_id} uses the wrong artifact role")
    expected_artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if expected_role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    if cell.artifact is not expected_artifact:
        raise ModelError(f"{cell_id} uses the wrong registered artifact")
    max_cell = cell.max_statistic_cell
    consistent = bool(
        cell.successful_attempts + cell.failed_attempts == cell.attempts
        and len(cell.attempt_audits) == cell.attempts
        and len(cell.successful_attempt_indices) == cell.successful_attempts
        and cell.successful_endpoint_matrix.shape[0] == cell.successful_attempts
        and cell.successful_attempts >= cell.minimum_successes
        and cell.success_requirement_passed
        and cell.ready_for_holm
        and max_cell is not None
        and max_cell.successful_replicates == cell.successful_attempts
        and math.isfinite(float(max_cell.p_value))
        and 0.0 <= max_cell.p_value <= 1.0
        and cell.cell_failure is None
    )
    canonical = bool(
        cell.strict_canonical
        and cell.attempts == 1_000
        and cell.minimum_successes == 950
    )
    ready = consistent and (canonical or not strict_canonical)
    reason = None
    if not consistent:
        reason = "refitted_adequacy_not_ready"
    elif strict_canonical and not canonical:
        reason = "refitted_adequacy_canonical_count_mismatch"
    return AdequacyCellRecord(
        cell_id,
        ready,
        float(max_cell.p_value) if ready and max_cell is not None else None,
        cell.attempts,
        cell.successful_attempts,
        reason,
    )


def _validate_view_inputs(
    task_id: str,
    run_id: str,
    binding_fingerprint: str,
    strict_canonical: bool,
) -> None:
    if task_id not in _TASK_CELL_ROLE:
        raise ModelError("adequacy cell task ID is not registered")
    _require_sha256(run_id, "adequacy run ID")
    _require_sha256(binding_fingerprint, "adequacy task binding fingerprint")
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")


def _mint_cell_task_view(
    *,
    task_id: AdequacyCellTaskId,
    role: AdequacyRole,
    run_id: str,
    binding_fingerprint: str,
    parent_view_fingerprint: str,
    parent_model_fingerprint: str,
    scientific_input_fingerprint: str,
    rng_execution_fingerprint: str,
    simulation_rng_execution_fingerprint: str | None,
    bootstrap_rng_execution_fingerprint: str | None,
    source_evidence_fingerprint: str,
    cell: AdequacyCellRecord,
    canonical_count_ready: bool,
    strict_canonical: bool,
    source_evidence: RegisteredLatentCell | RegisteredRefittedAdequacyBatch,
    parent_view: CandidateSelectionTaskView | ProductionFitSameKView,
) -> AdequacyCellTaskView:
    for value, label in (
        (parent_view_fingerprint, "parent view fingerprint"),
        (parent_model_fingerprint, "parent model fingerprint"),
        (scientific_input_fingerprint, "scientific input fingerprint"),
        (rng_execution_fingerprint, "RNG execution fingerprint"),
        (source_evidence_fingerprint, "source evidence fingerprint"),
    ):
        _require_sha256(value, label)
    latent_task = "latent_correctness" in task_id
    split_rng = (
        simulation_rng_execution_fingerprint,
        bootstrap_rng_execution_fingerprint,
    )
    if latent_task:
        for split_value, label in zip(
            split_rng,
            ("latent simulation RNG fingerprint", "latent bootstrap RNG fingerprint"),
            strict=True,
        ):
            if split_value is None:
                raise ModelError(f"{label} is missing")
            _require_sha256(split_value, label)
    elif split_rng != (None, None):
        raise ModelError("refitted adequacy cannot expose latent RNG identities")
    passed = _individual_cell_passed(
        cell,
        canonical_count_ready=canonical_count_ready,
        strict_canonical=strict_canonical,
    )
    payload = _cell_view_payload(
        task_id=task_id,
        role=role,
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        parent_view_fingerprint=parent_view_fingerprint,
        parent_model_fingerprint=parent_model_fingerprint,
        scientific_input_fingerprint=scientific_input_fingerprint,
        rng_execution_fingerprint=rng_execution_fingerprint,
        simulation_rng_execution_fingerprint=(simulation_rng_execution_fingerprint),
        bootstrap_rng_execution_fingerprint=(bootstrap_rng_execution_fingerprint),
        source_evidence_fingerprint=source_evidence_fingerprint,
        cell=cell,
        canonical_count_ready=canonical_count_ready,
        ready=cell.ready,
        passed=passed,
        strict_canonical=strict_canonical,
    )
    result = AdequacyCellTaskView(
        ADEQUACY_TASK_VIEW_CONTRACT,
        task_id,
        role,
        run_id,
        binding_fingerprint,
        parent_view_fingerprint,
        parent_model_fingerprint,
        scientific_input_fingerprint,
        rng_execution_fingerprint,
        simulation_rng_execution_fingerprint,
        bootstrap_rng_execution_fingerprint,
        source_evidence_fingerprint,
        cell,
        canonical_count_ready,
        cell.ready,
        passed,
        strict_canonical,
        _fingerprint(payload),
        source_evidence,
        parent_view,
    )
    object.__setattr__(result, "_execution_seal", _ADEQUACY_TASK_VIEW_CAPABILITY)
    _MINTED_CELL_VIEWS[id(result)] = result
    return result


def _reverify_cell_sources(
    value: AdequacyCellTaskView,
) -> tuple[
    str,
    str,
    str,
    str,
    str | None,
    str | None,
    str,
    AdequacyCellRecord,
    bool,
]:
    parent_fit, parent_features, parent_view_fingerprint = _parent_sources(
        value.role,
        value._parent_view,
    )
    _validate_parent_task_binding(
        value._parent_view,
        run_id=value.run_id,
        child_binding_fingerprint=value.binding_fingerprint,
    )
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(parent_fit)
    expected_artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if value.role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    if "latent_correctness" in value.task_id:
        evidence = value._source_evidence
        if not isinstance(evidence, RegisteredLatentCell):
            raise ModelError("latent adequacy view lost its registered source")
        latent_identity = registered_latent_cell_identity(evidence)
        source_fit = registered_latent_cell_source_fit(evidence)
        transition_fingerprint = scientific_array_fingerprint(
            parent_fit.parameters.transition_matrix,
            role="latent_transition_matrix",
        )
        if (
            source_fit is not parent_fit
            or evidence.artifact is not expected_artifact
            or latent_identity.parent_model_fingerprint != parent_model_fingerprint
            or latent_identity.transition_matrix_fingerprint != transition_fingerprint
        ):
            raise ModelError("latent adequacy source no longer matches its parent")
        latent_cell_id: Literal["design_latent", "production_latent"] = (
            "design_latent" if value.role == "design" else "production_latent"
        )
        record = _latent_record(
            latent_cell_id,
            evidence,
            expected_artifact=expected_artifact,
            strict_canonical=value.strict_canonical,
        )
        return (
            parent_view_fingerprint,
            parent_model_fingerprint,
            transition_fingerprint,
            latent_identity.rng_execution_fingerprint,
            latent_identity.simulation_rng_execution_fingerprint,
            latent_identity.bootstrap_rng_execution_fingerprint,
            latent_identity.source_evidence_fingerprint,
            record,
            _latent_canonical_geometry(evidence),
        )

    evidence = value._source_evidence
    if not isinstance(evidence, RegisteredRefittedAdequacyBatch):
        raise ModelError("refitted adequacy view lost its registered source")
    refit_identity = registered_refitted_adequacy_batch_identity(evidence)
    source_fit, source_features = registered_refitted_adequacy_sources(evidence)
    if source_fit is not parent_fit:
        raise ModelError("refitted adequacy source no longer retains its parent fit")
    if value.role == "production" and source_features is not parent_features:
        raise ModelError(
            "production refitted adequacy no longer retains its parent features"
        )
    feature_fingerprint = hmm_feature_matrix_fingerprint(source_features)
    if (
        evidence.artifact is not expected_artifact
        or refit_identity.parent_model_fingerprint != parent_model_fingerprint
        or refit_identity.artifact_feature_fingerprint != feature_fingerprint
    ):
        raise ModelError("refitted adequacy source no longer matches its parent")
    refit_cell_id: Literal[
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    ] = (
        "design_refitted_adequacy"
        if value.role == "design"
        else "production_refitted_adequacy"
    )
    record = _refit_record(
        refit_cell_id,
        evidence,
        expected_role=value.role,
        strict_canonical=value.strict_canonical,
    )
    return (
        parent_view_fingerprint,
        parent_model_fingerprint,
        feature_fingerprint,
        refit_identity.rng_execution_fingerprint,
        None,
        None,
        refit_identity.source_evidence_fingerprint,
        record,
        _refit_canonical_count(evidence),
    )


def _verify_cell_task_view(value: AdequacyCellTaskView) -> None:
    if (
        value._execution_seal is not _ADEQUACY_TASK_VIEW_CAPABILITY
        or _MINTED_CELL_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("adequacy cell task view lacks builder authority")
    _validate_view_inputs(
        value.task_id,
        value.run_id,
        value.binding_fingerprint,
        value.strict_canonical,
    )
    expected_cell, expected_role = _TASK_CELL_ROLE[value.task_id]
    if value.contract_version != ADEQUACY_TASK_VIEW_CONTRACT:
        raise ModelError("adequacy cell task view contract is invalid")
    if value.role != expected_role or value.cell.cell_id != expected_cell:
        raise ModelError("adequacy cell task view role is inconsistent")
    source_identity = _reverify_cell_sources(value)
    if (
        value.parent_view_fingerprint != source_identity[0]
        or value.parent_model_fingerprint != source_identity[1]
        or value.scientific_input_fingerprint != source_identity[2]
        or value.rng_execution_fingerprint != source_identity[3]
        or value.simulation_rng_execution_fingerprint != source_identity[4]
        or value.bootstrap_rng_execution_fingerprint != source_identity[5]
        or value.source_evidence_fingerprint != source_identity[6]
        or value.cell != source_identity[7]
        or value.canonical_count_ready is not source_identity[8]
    ):
        raise ModelError("adequacy cell task source ancestry changed")
    for fingerprint, label in (
        (value.parent_view_fingerprint, "parent view fingerprint"),
        (value.parent_model_fingerprint, "parent model fingerprint"),
        (value.scientific_input_fingerprint, "scientific input fingerprint"),
        (value.rng_execution_fingerprint, "RNG execution fingerprint"),
        (value.source_evidence_fingerprint, "source evidence fingerprint"),
        (value.content_fingerprint, "cell task content fingerprint"),
    ):
        _require_sha256(fingerprint, label)
    split_rng = (
        value.simulation_rng_execution_fingerprint,
        value.bootstrap_rng_execution_fingerprint,
    )
    if "latent_correctness" in value.task_id:
        simulation_rng, bootstrap_rng = split_rng
        if simulation_rng is None or bootstrap_rng is None:
            raise ModelError("latent adequacy view lacks split RNG identities")
        _require_sha256(simulation_rng, "latent simulation RNG fingerprint")
        _require_sha256(bootstrap_rng, "latent bootstrap RNG fingerprint")
    elif split_rng != (None, None):
        raise ModelError("refitted adequacy view contains latent RNG identities")
    if not isinstance(value.canonical_count_ready, bool):
        raise ModelError("canonical-count readiness must be boolean")
    if value.ready is not value.cell.ready:
        raise ModelError("adequacy cell task readiness is inconsistent")
    expected_passed = _individual_cell_passed(
        value.cell,
        canonical_count_ready=value.canonical_count_ready,
        strict_canonical=value.strict_canonical,
    )
    if value.passed is not expected_passed:
        raise ModelError("adequacy cell task pass flag is inconsistent")
    expected_fingerprint = _fingerprint(
        _cell_view_payload(
            task_id=value.task_id,
            role=value.role,
            run_id=value.run_id,
            binding_fingerprint=value.binding_fingerprint,
            parent_view_fingerprint=value.parent_view_fingerprint,
            parent_model_fingerprint=value.parent_model_fingerprint,
            scientific_input_fingerprint=value.scientific_input_fingerprint,
            rng_execution_fingerprint=value.rng_execution_fingerprint,
            simulation_rng_execution_fingerprint=(
                value.simulation_rng_execution_fingerprint
            ),
            bootstrap_rng_execution_fingerprint=(
                value.bootstrap_rng_execution_fingerprint
            ),
            source_evidence_fingerprint=value.source_evidence_fingerprint,
            cell=value.cell,
            canonical_count_ready=value.canonical_count_ready,
            ready=value.ready,
            passed=value.passed,
            strict_canonical=value.strict_canonical,
        )
    )
    if value.content_fingerprint != expected_fingerprint:
        raise ModelError("adequacy cell task content fingerprint is inconsistent")


def _verify_holm_view(value: FourCellAdequacyHolmView) -> None:
    if (
        value._execution_seal is not _ADEQUACY_HOLM_VIEW_CAPABILITY
        or _MINTED_HOLM_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("adequacy Holm view lacks builder authority")
    if (
        value.contract_version != ADEQUACY_TASK_VIEW_CONTRACT
        or value.task_id != FOUR_CELL_ADEQUACY_HOLM_TASK_ID
        or value.source_task_order != ADEQUACY_TASK_ORDER
        or value.cell_order != ADEQUACY_CELL_ORDER
    ):
        raise ModelError("adequacy Holm view contract/order is invalid")
    _require_sha256(value.run_id, "adequacy Holm run ID")
    _require_sha256(value.binding_fingerprint, "adequacy Holm task binding")
    if not isinstance(value.strict_canonical, bool):
        raise ModelError("adequacy Holm strict mode must be boolean")
    if len(value.source_views) != 4 or len(value.source_content_fingerprints) != 4:
        raise ModelError("adequacy Holm must contain exactly four sources")
    for source in value.source_views:
        _verify_cell_task_view(source)
    if tuple(source.task_id for source in value.source_views) != ADEQUACY_TASK_ORDER:
        raise ModelError("adequacy Holm source task order is invalid")
    if any(source.run_id != value.run_id for source in value.source_views):
        raise ModelError("adequacy Holm sources do not share its run ID")
    if any(
        source.strict_canonical is not value.strict_canonical
        for source in value.source_views
    ):
        raise ModelError("adequacy Holm source strict mode is inconsistent")
    bindings = tuple(source.binding_fingerprint for source in value.source_views)
    if len(set(bindings)) != 4 or value.binding_fingerprint in bindings:
        raise ModelError("adequacy Holm task bindings are not distinct")
    expected_source_fingerprints = tuple(
        source.content_fingerprint for source in value.source_views
    )
    if value.source_content_fingerprints != expected_source_fingerprints:
        raise ModelError("adequacy Holm source fingerprints are inconsistent")
    expected_holm, ready, passed, reasons = _authoritative_holm(
        value.cells,
        source_passed=(
            value.source_views[0].passed,
            value.source_views[1].passed,
            value.source_views[2].passed,
            value.source_views[3].passed,
        ),
        alpha=value.alpha,
    )
    if (
        value.ready is not ready
        or value.passed is not passed
        or value.failure_reasons != reasons
        or not _same_holm(value.holm, expected_holm)
    ):
        raise ModelError("adequacy Holm authoritative result is inconsistent")
    expected_fingerprint = _holm_view_fingerprint(
        run_id=value.run_id,
        binding_fingerprint=value.binding_fingerprint,
        sources=value.source_views,
        source_content_fingerprints=value.source_content_fingerprints,
        holm=value.holm,
        ready=value.ready,
        passed=value.passed,
        failure_reasons=value.failure_reasons,
        alpha=value.alpha,
        strict_canonical=value.strict_canonical,
    )
    if value.content_fingerprint != expected_fingerprint:
        raise ModelError("adequacy Holm content fingerprint is inconsistent")


def _verify_compatibility_gate(value: FourCellAdequacyGate) -> None:
    if (
        value._execution_seal is not _FOUR_CELL_GATE_CAPABILITY
        or _MINTED_FOUR_CELL_GATES.get(id(value)) is not value
    ):
        raise ModelError("four-cell adequacy gate lacks assembler authority")
    if value.cell_order != ADEQUACY_CELL_ORDER or len(value.cells) != 4:
        raise ModelError("four-cell adequacy gate order is invalid")
    if tuple(cell.cell_id for cell in value.cells) != ADEQUACY_CELL_ORDER:
        raise ModelError("four-cell adequacy cell IDs are invalid")
    if not isinstance(value.strict_canonical, bool):
        raise ModelError("four-cell adequacy strict mode must be boolean")
    expected_holm, ready, passed, reasons = _authoritative_holm(
        value.cells,
        source_passed=(
            value.cells[0].ready,
            value.cells[1].ready,
            value.cells[2].ready,
            value.cells[3].ready,
        ),
        alpha=value.alpha,
    )
    if (
        value.ready is not ready
        or value.passed is not passed
        or value.failure_reasons != reasons
        or not _same_holm(value.holm, expected_holm)
    ):
        raise ModelError("four-cell adequacy gate result is inconsistent")
    expected_fingerprint = _compatibility_gate_fingerprint(
        cells=value.cells,
        holm=value.holm,
        ready=value.ready,
        passed=value.passed,
        failure_reasons=value.failure_reasons,
        alpha=value.alpha,
        strict_canonical=value.strict_canonical,
    )
    if value.content_fingerprint != expected_fingerprint:
        raise ModelError("four-cell adequacy gate fingerprint is inconsistent")


def _individual_cell_passed(
    cell: AdequacyCellRecord,
    *,
    canonical_count_ready: bool,
    strict_canonical: bool,
) -> bool:
    p_value = cell.p_value
    return bool(
        cell.ready
        and (canonical_count_ready or not strict_canonical)
        and p_value is not None
        and math.isfinite(p_value)
        and 0.0 <= p_value <= 1.0
        and cell.failure_reason is None
    )


def _authoritative_holm(
    cells: tuple[
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
    ],
    *,
    source_passed: tuple[bool, bool, bool, bool],
    alpha: float,
) -> tuple[HolmResult | None, bool, bool, tuple[str, ...]]:
    if tuple(cell.cell_id for cell in cells) != ADEQUACY_CELL_ORDER:
        raise ModelError("adequacy cells are not in preregistered order")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ModelError("adequacy Holm alpha must be finite and in (0, 1)")
    reasons = tuple(
        f"{cell.cell_id}:{cell.failure_reason or 'cell_not_ready'}"
        for cell, passed in zip(cells, source_passed, strict=True)
        if not passed
    )
    if reasons:
        return None, False, False, reasons
    p_values = tuple(cell.p_value for cell in cells)
    if any(value is None for value in p_values):
        raise ModelError("ready adequacy cell is missing its p-value")
    numeric = tuple(float(value) for value in p_values if value is not None)
    if len(numeric) != 4:  # pragma: no cover - guarded immediately above.
        raise AssertionError("adequacy p-value count is inconsistent")
    holm = four_cell_holm(numeric, alpha=alpha)
    rejected = tuple(bool(value) for value in holm.rejected)
    rejection_reasons = tuple(
        f"{cell.cell_id}:holm_rejected"
        for cell, is_rejected in zip(cells, rejected, strict=True)
        if is_rejected
    )
    return holm, True, not any(rejected), rejection_reasons


def _same_holm(left: HolmResult | None, right: HolmResult | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        left.alpha == right.alpha
        and np.array_equal(left.raw_p_values, right.raw_p_values)
        and np.array_equal(left.adjusted_p_values, right.adjusted_p_values)
        and np.array_equal(left.rejected, right.rejected)
    )


def _latent_canonical_geometry(cell: RegisteredLatentCell) -> bool:
    simulation = cell.simulation
    bootstrap = cell.bootstrap
    return bool(
        simulation.chain_count == 10_000
        and simulation.measured_months == 600
        and simulation.extension_months == 2_000
        and simulation.uniforms_per_chain == 2_601
        and bootstrap.bootstrap_replicates == 10_000
        and bootstrap.resample_size == 10_000
        and bootstrap.survival_months == 12
    )


def _refit_canonical_count(cell: RegisteredRefittedAdequacyBatch) -> bool:
    return bool(
        cell.strict_canonical
        and cell.attempts == 1_000
        and cell.minimum_successes == 950
    )


def _cell_view_payload(
    *,
    task_id: AdequacyCellTaskId,
    role: AdequacyRole,
    run_id: str,
    binding_fingerprint: str,
    parent_view_fingerprint: str,
    parent_model_fingerprint: str,
    scientific_input_fingerprint: str,
    rng_execution_fingerprint: str,
    simulation_rng_execution_fingerprint: str | None,
    bootstrap_rng_execution_fingerprint: str | None,
    source_evidence_fingerprint: str,
    cell: AdequacyCellRecord,
    canonical_count_ready: bool,
    ready: bool,
    passed: bool,
    strict_canonical: bool,
) -> dict[str, object]:
    return {
        "contract_version": ADEQUACY_TASK_VIEW_CONTRACT,
        "task_id": task_id,
        "role": role,
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        "parent_view_fingerprint": parent_view_fingerprint,
        "parent_model_fingerprint": parent_model_fingerprint,
        "scientific_input_fingerprint": scientific_input_fingerprint,
        "rng_execution_fingerprint": rng_execution_fingerprint,
        "simulation_rng_execution_fingerprint": (simulation_rng_execution_fingerprint),
        "bootstrap_rng_execution_fingerprint": bootstrap_rng_execution_fingerprint,
        "source_evidence_fingerprint": source_evidence_fingerprint,
        "cell": _cell_payload(cell),
        "canonical_count_ready": canonical_count_ready,
        "ready": ready,
        "passed": passed,
        "strict_canonical": strict_canonical,
    }


def _holm_view_fingerprint(
    *,
    run_id: str,
    binding_fingerprint: str,
    sources: tuple[
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
    ],
    source_content_fingerprints: tuple[str, str, str, str],
    holm: HolmResult | None,
    ready: bool,
    passed: bool,
    failure_reasons: tuple[str, ...],
    alpha: float,
    strict_canonical: bool,
) -> str:
    return _fingerprint(
        {
            "contract_version": ADEQUACY_TASK_VIEW_CONTRACT,
            "task_id": FOUR_CELL_ADEQUACY_HOLM_TASK_ID,
            "run_id": run_id,
            "binding_fingerprint": binding_fingerprint,
            "source_task_order": list(ADEQUACY_TASK_ORDER),
            "source_binding_fingerprints": [
                source.binding_fingerprint for source in sources
            ],
            "source_content_fingerprints": list(source_content_fingerprints),
            "cell_order": list(ADEQUACY_CELL_ORDER),
            "cells": [_cell_payload(source.cell) for source in sources],
            "holm": _holm_payload(holm),
            "ready": ready,
            "passed": passed,
            "failure_reasons": list(failure_reasons),
            "alpha": alpha,
            "strict_canonical": strict_canonical,
        }
    )


def _compatibility_gate_fingerprint(
    *,
    cells: tuple[
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
        AdequacyCellRecord,
    ],
    holm: HolmResult | None,
    ready: bool,
    passed: bool,
    failure_reasons: tuple[str, ...],
    alpha: float,
    strict_canonical: bool,
) -> str:
    return _fingerprint(
        {
            "contract_version": "four-cell-adequacy-compatibility-gate-v2",
            "cell_order": list(ADEQUACY_CELL_ORDER),
            "cells": [_cell_payload(cell) for cell in cells],
            "holm": _holm_payload(holm),
            "ready": ready,
            "passed": passed,
            "failure_reasons": list(failure_reasons),
            "alpha": alpha,
            "strict_canonical": strict_canonical,
        }
    )


def _cell_payload(cell: AdequacyCellRecord) -> dict[str, object]:
    if not isinstance(cell, AdequacyCellRecord):
        raise ModelError("adequacy cell record has the wrong type")
    return {
        "cell_id": cell.cell_id,
        "ready": cell.ready,
        "p_value": cell.p_value,
        "repetitions": cell.repetitions,
        "successful_repetitions": cell.successful_repetitions,
        "failure_reason": cell.failure_reason,
    }


def _holm_payload(holm: HolmResult | None) -> dict[str, object] | None:
    if holm is None:
        return None
    return {
        "raw_p_values": np.asarray(holm.raw_p_values, dtype=np.float64).tolist(),
        "adjusted_p_values": np.asarray(
            holm.adjusted_p_values, dtype=np.float64
        ).tolist(),
        "rejected": np.asarray(holm.rejected, dtype=np.bool_).tolist(),
        "alpha": float(holm.alpha),
    }


def _latent_source_evidence_fingerprint(cell: RegisteredLatentCell) -> str:
    return registered_latent_cell_identity(cell).source_evidence_fingerprint


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
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
        ).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError("adequacy task evidence is not canonical JSON") from error


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"{label} must be SHA-256")
