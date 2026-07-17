"""Cycle-free registered execution of the sealed design holdout veto.

Task 4 is a producer boundary.  The selected design fit and both immutable
calibration slices are validated before the task binding is minted, the
holdout is evaluated exactly once, and the exact result object is retained in
an object ledger.  A valid benchmark may still lose the veto; that is ordinary
``SealedHoldoutEvidence`` with ``decision.passed=False``.  Only the narrow
``InvalidHoldoutBenchmark`` signal becomes the distinct scientific failure
outcome below.  All generic model, integrity, filesystem, memory, process, and
programming failures propagate so the launch remains resumable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeGuard

import numpy as np

from prpg.errors import ModelError
from prpg.model.benchmarks import (
    HoldoutBenchmarkFailureCode,
    HoldoutBenchmarkScores,
    InvalidHoldoutBenchmark,
    InvalidHoldoutBenchmarkEvidence,
    holdout_veto_decision,
)
from prpg.model.calibration import SealedHoldoutEvidence, evaluate_sealed_holdout
from prpg.model.calibration_identity import (
    calibration_slice_fingerprint,
    sealed_holdout_evidence_fingerprint,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    sequence_lengths_from_continuation,
)
from prpg.model.input import CalibrationSlice
from prpg.model.materialization_identity import gaussian_hmm_fit_fingerprint

if TYPE_CHECKING:
    from prpg.model.g3_calibration_views import CandidateSelectionTaskView


REGISTERED_HOLDOUT_EXECUTION_CONTRACT = "registered-sealed-holdout-execution-v1"
REGISTERED_HOLDOUT_INVALID_BENCHMARK_CONTRACT = (
    "registered-sealed-holdout-invalid-benchmark-v1"
)
REGISTERED_HOLDOUT_SOURCE_CONTRACT = "sealed-holdout-source-materialization-v1"
REGISTERED_HOLDOUT_RESULT_CONTRACT = "sealed-holdout-result-evidence-v1"
REGISTERED_HOLDOUT_FAILURE_CONTRACT = "sealed-holdout-failure-evidence-v1"

_HOLDOUT_EXECUTION_CAPABILITY = object()
_HOLDOUT_FAILURE_CAPABILITY = object()
_DESIGN_ROWS = 193
_HOLDOUT_LENGTHS = (17, 7)
_HOLDOUT_ROWS = sum(_HOLDOUT_LENGTHS)
_FAILURE_CODES = frozenset(
    {
        "gaussian_covariance_invalid",
        "gaussian_score_invalid",
        "var_ridge_solve_failed",
        "var_parameters_nonfinite",
        "var_eigenvalues_nonfinite",
        "var_spectral_radius_invalid",
        "var_residual_covariance_invalid",
        "var_stationary_law_invalid",
        "var_stationary_covariance_invalid",
        "var_score_invalid",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class RegisteredSealedHoldoutEvaluation:
    """One immutable, valid-benchmark holdout evaluation."""

    contract_version: str
    task_id: Literal["sealed_holdout_veto"]
    role: Literal["design"]
    outcome: Literal["evaluated"]
    selected_n_states: int
    selection_view_fingerprint: str
    selected_model_fingerprint: str
    design_slice_fingerprint: str
    holdout_slice_fingerprint: str
    source_materialization_fingerprint: str
    evidence: SealedHoldoutEvidence
    evidence_fingerprint: str
    execution_fingerprint: str
    generation_authorized: Literal[False]
    _selection_source: CandidateSelectionTaskView = field(
        repr=False,
        compare=False,
    )
    _selected_fit_source: GaussianHMMFit = field(repr=False, compare=False)
    _design_source: CalibrationSlice = field(repr=False, compare=False)
    _holdout_source: CalibrationSlice = field(repr=False, compare=False)
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredSealedHoldoutEvaluation is created only by "
            "execute_registered_sealed_holdout"
        )


@dataclass(frozen=True, slots=True, init=False)
class RegisteredSealedHoldoutInvalidBenchmark:
    """One immutable scientific failure caused by an invalid benchmark."""

    contract_version: str
    task_id: Literal["sealed_holdout_veto"]
    role: Literal["design"]
    outcome: Literal["invalid_benchmark"]
    selected_n_states: int
    selection_view_fingerprint: str
    selected_model_fingerprint: str
    design_slice_fingerprint: str
    holdout_slice_fingerprint: str
    source_materialization_fingerprint: str
    failure_code: HoldoutBenchmarkFailureCode
    failure_message: str
    failure_details: tuple[tuple[str, object], ...]
    failure_evidence_fingerprint: str
    execution_fingerprint: str
    generation_authorized: Literal[False]
    _selection_source: CandidateSelectionTaskView = field(
        repr=False,
        compare=False,
    )
    _selected_fit_source: GaussianHMMFit = field(repr=False, compare=False)
    _design_source: CalibrationSlice = field(repr=False, compare=False)
    _holdout_source: CalibrationSlice = field(repr=False, compare=False)
    _failure_source: InvalidHoldoutBenchmarkEvidence = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredSealedHoldoutInvalidBenchmark is created only by "
            "execute_registered_sealed_holdout"
        )


RegisteredSealedHoldoutOutcome: TypeAlias = (
    RegisteredSealedHoldoutEvaluation | RegisteredSealedHoldoutInvalidBenchmark
)


@dataclass(frozen=True, slots=True)
class RegisteredSealedHoldoutEvaluationIdentity:
    """Reverified sources and result identity for a valid benchmark."""

    selected_n_states: int
    selection_view_fingerprint: str
    selected_model_fingerprint: str
    design_slice_fingerprint: str
    holdout_slice_fingerprint: str
    source_materialization_fingerprint: str
    evidence_fingerprint: str
    execution_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredSealedHoldoutInvalidBenchmarkIdentity:
    """Reverified sources and exact invalid-benchmark identity."""

    selected_n_states: int
    selection_view_fingerprint: str
    selected_model_fingerprint: str
    design_slice_fingerprint: str
    holdout_slice_fingerprint: str
    source_materialization_fingerprint: str
    failure_code: HoldoutBenchmarkFailureCode
    failure_evidence_fingerprint: str
    execution_fingerprint: str


_MINTED_HOLDOUT_EVALUATIONS: dict[
    int,
    tuple[
        RegisteredSealedHoldoutEvaluation,
        CandidateSelectionTaskView,
        GaussianHMMFit,
        CalibrationSlice,
        CalibrationSlice,
        SealedHoldoutEvidence,
    ],
] = {}

_MINTED_HOLDOUT_FAILURES: dict[
    int,
    tuple[
        RegisteredSealedHoldoutInvalidBenchmark,
        CandidateSelectionTaskView,
        GaussianHMMFit,
        CalibrationSlice,
        CalibrationSlice,
        InvalidHoldoutBenchmarkEvidence,
    ],
] = {}


def execute_registered_sealed_holdout(
    selection: CandidateSelectionTaskView,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
) -> RegisteredSealedHoldoutOutcome:
    """Evaluate the exact selected design HMM against both frozen benchmarks."""

    fit, selection_fingerprint = _validated_selection(selection)
    _validate_slice_geometry(design, holdout)
    model_fingerprint = gaussian_hmm_fit_fingerprint(fit)
    design_fingerprint = calibration_slice_fingerprint(design)
    holdout_fingerprint = calibration_slice_fingerprint(holdout)
    source_fingerprint = _source_fingerprint(
        selected_n_states=fit.n_states,
        selection_view_fingerprint=selection_fingerprint,
        selected_model_fingerprint=model_fingerprint,
        design_slice_fingerprint=design_fingerprint,
        holdout_slice_fingerprint=holdout_fingerprint,
    )
    try:
        evaluated = evaluate_sealed_holdout(fit, design, holdout)
    except InvalidHoldoutBenchmark as failure:
        _verify_sources_unchanged(
            selection=selection,
            fit=fit,
            design=design,
            holdout=holdout,
            selection_fingerprint=selection_fingerprint,
            model_fingerprint=model_fingerprint,
            design_fingerprint=design_fingerprint,
            holdout_fingerprint=holdout_fingerprint,
        )
        return _mint_failure(
            selection=selection,
            fit=fit,
            design=design,
            holdout=holdout,
            selection_fingerprint=selection_fingerprint,
            model_fingerprint=model_fingerprint,
            design_fingerprint=design_fingerprint,
            holdout_fingerprint=holdout_fingerprint,
            source_fingerprint=source_fingerprint,
            evidence=failure.evidence,
        )

    _verify_sources_unchanged(
        selection=selection,
        fit=fit,
        design=design,
        holdout=holdout,
        selection_fingerprint=selection_fingerprint,
        model_fingerprint=model_fingerprint,
        design_fingerprint=design_fingerprint,
        holdout_fingerprint=holdout_fingerprint,
    )
    evidence = _immutable_evidence(evaluated)
    _validate_evidence(evidence)
    evidence_fingerprint = sealed_holdout_evidence_fingerprint(evidence)
    execution_fingerprint = _success_execution_fingerprint(
        source_fingerprint=source_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )
    value = object.__new__(RegisteredSealedHoldoutEvaluation)
    for name, item in (
        ("contract_version", REGISTERED_HOLDOUT_EXECUTION_CONTRACT),
        ("task_id", "sealed_holdout_veto"),
        ("role", "design"),
        ("outcome", "evaluated"),
        ("selected_n_states", fit.n_states),
        ("selection_view_fingerprint", selection_fingerprint),
        ("selected_model_fingerprint", model_fingerprint),
        ("design_slice_fingerprint", design_fingerprint),
        ("holdout_slice_fingerprint", holdout_fingerprint),
        ("source_materialization_fingerprint", source_fingerprint),
        ("evidence", evidence),
        ("evidence_fingerprint", evidence_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("generation_authorized", False),
        ("_selection_source", selection),
        ("_selected_fit_source", fit),
        ("_design_source", design),
        ("_holdout_source", holdout),
        ("_execution_seal", _HOLDOUT_EXECUTION_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_HOLDOUT_EVALUATIONS[id(value)] = (
        value,
        selection,
        fit,
        design,
        holdout,
        evidence,
    )
    registered_sealed_holdout_evaluation_identity(value)
    return value


def registered_sealed_holdout_evaluation_identity(
    value: RegisteredSealedHoldoutEvaluation,
) -> RegisteredSealedHoldoutEvaluationIdentity:
    """Revalidate one exact object minted by the registered executor."""

    if not isinstance(value, RegisteredSealedHoldoutEvaluation):
        raise ModelError("registered holdout evaluation has the wrong type")
    ledger = _MINTED_HOLDOUT_EVALUATIONS.get(id(value))
    if (
        value._execution_seal is not _HOLDOUT_EXECUTION_CAPABILITY
        or ledger is None
        or ledger[0] is not value
        or ledger[1] is not value._selection_source
        or ledger[2] is not value._selected_fit_source
        or ledger[3] is not value._design_source
        or ledger[4] is not value._holdout_source
        or ledger[5] is not value.evidence
    ):
        raise ModelError("registered holdout evaluation lacks executor authority")
    if (
        value.contract_version != REGISTERED_HOLDOUT_EXECUTION_CONTRACT
        or value.task_id != "sealed_holdout_veto"
        or value.role != "design"
        or value.outcome != "evaluated"
        or value.generation_authorized is not False
    ):
        raise ModelError("registered holdout evaluation contract changed")
    identities = _live_source_identities(
        value._selection_source,
        value._selected_fit_source,
        value._design_source,
        value._holdout_source,
    )
    _validate_evidence(value.evidence)
    evidence_fingerprint = sealed_holdout_evidence_fingerprint(value.evidence)
    source_fingerprint = _source_fingerprint(
        selected_n_states=value._selected_fit_source.n_states,
        selection_view_fingerprint=identities[0],
        selected_model_fingerprint=identities[1],
        design_slice_fingerprint=identities[2],
        holdout_slice_fingerprint=identities[3],
    )
    execution_fingerprint = _success_execution_fingerprint(
        source_fingerprint=source_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )
    if (
        value.selected_n_states != value._selected_fit_source.n_states
        or value.selection_view_fingerprint != identities[0]
        or value.selected_model_fingerprint != identities[1]
        or value.design_slice_fingerprint != identities[2]
        or value.holdout_slice_fingerprint != identities[3]
        or value.source_materialization_fingerprint != source_fingerprint
        or value.evidence_fingerprint != evidence_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("registered holdout sources or result changed")
    return RegisteredSealedHoldoutEvaluationIdentity(
        selected_n_states=value.selected_n_states,
        selection_view_fingerprint=identities[0],
        selected_model_fingerprint=identities[1],
        design_slice_fingerprint=identities[2],
        holdout_slice_fingerprint=identities[3],
        source_materialization_fingerprint=source_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def registered_sealed_holdout_invalid_benchmark_identity(
    value: RegisteredSealedHoldoutInvalidBenchmark,
) -> RegisteredSealedHoldoutInvalidBenchmarkIdentity:
    """Revalidate one exact executor-minted invalid-benchmark outcome."""

    if not isinstance(value, RegisteredSealedHoldoutInvalidBenchmark):
        raise ModelError("registered invalid holdout benchmark has the wrong type")
    ledger = _MINTED_HOLDOUT_FAILURES.get(id(value))
    if (
        value._execution_seal is not _HOLDOUT_FAILURE_CAPABILITY
        or ledger is None
        or ledger[0] is not value
        or ledger[1] is not value._selection_source
        or ledger[2] is not value._selected_fit_source
        or ledger[3] is not value._design_source
        or ledger[4] is not value._holdout_source
        or ledger[5] is not value._failure_source
    ):
        raise ModelError("registered invalid holdout benchmark lacks authority")
    if (
        value.contract_version != REGISTERED_HOLDOUT_INVALID_BENCHMARK_CONTRACT
        or value.task_id != "sealed_holdout_veto"
        or value.role != "design"
        or value.outcome != "invalid_benchmark"
        or value.generation_authorized is not False
    ):
        raise ModelError("registered invalid holdout benchmark contract changed")
    identities = _live_source_identities(
        value._selection_source,
        value._selected_fit_source,
        value._design_source,
        value._holdout_source,
    )
    failure = _validated_failure_evidence(value._failure_source)
    source_fingerprint = _source_fingerprint(
        selected_n_states=value._selected_fit_source.n_states,
        selection_view_fingerprint=identities[0],
        selected_model_fingerprint=identities[1],
        design_slice_fingerprint=identities[2],
        holdout_slice_fingerprint=identities[3],
    )
    failure_fingerprint = _failure_fingerprint(
        source_fingerprint=source_fingerprint,
        evidence=failure,
    )
    execution_fingerprint = _failure_execution_fingerprint(
        source_fingerprint=source_fingerprint,
        failure_evidence_fingerprint=failure_fingerprint,
    )
    if (
        value.selected_n_states != value._selected_fit_source.n_states
        or value.selection_view_fingerprint != identities[0]
        or value.selected_model_fingerprint != identities[1]
        or value.design_slice_fingerprint != identities[2]
        or value.holdout_slice_fingerprint != identities[3]
        or value.source_materialization_fingerprint != source_fingerprint
        or value.failure_code != failure.failure_code
        or value.failure_message != failure.failure_message
        or value.failure_details != failure.failure_details
        or value.failure_evidence_fingerprint != failure_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("registered invalid holdout sources or evidence changed")
    return RegisteredSealedHoldoutInvalidBenchmarkIdentity(
        selected_n_states=value.selected_n_states,
        selection_view_fingerprint=identities[0],
        selected_model_fingerprint=identities[1],
        design_slice_fingerprint=identities[2],
        holdout_slice_fingerprint=identities[3],
        source_materialization_fingerprint=source_fingerprint,
        failure_code=failure.failure_code,
        failure_evidence_fingerprint=failure_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def is_registered_sealed_holdout_evaluation(
    value: object,
) -> TypeGuard[RegisteredSealedHoldoutEvaluation]:
    """Return whether an object is the unchanged valid-benchmark outcome."""

    if not isinstance(value, RegisteredSealedHoldoutEvaluation):
        return False
    try:
        registered_sealed_holdout_evaluation_identity(value)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_registered_sealed_holdout_invalid_benchmark(
    value: object,
) -> TypeGuard[RegisteredSealedHoldoutInvalidBenchmark]:
    """Return whether an object is the unchanged invalid-benchmark outcome."""

    if not isinstance(value, RegisteredSealedHoldoutInvalidBenchmark):
        return False
    try:
        registered_sealed_holdout_invalid_benchmark_identity(value)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_sealed_holdout_evaluation_sources(
    value: RegisteredSealedHoldoutEvaluation,
) -> tuple[
    CandidateSelectionTaskView,
    GaussianHMMFit,
    CalibrationSlice,
    CalibrationSlice,
    SealedHoldoutEvidence,
]:
    """Return exact parents and result after full ledger reverification."""

    registered_sealed_holdout_evaluation_identity(value)
    return (
        value._selection_source,
        value._selected_fit_source,
        value._design_source,
        value._holdout_source,
        value.evidence,
    )


def registered_sealed_holdout_invalid_benchmark_sources(
    value: RegisteredSealedHoldoutInvalidBenchmark,
) -> tuple[
    CandidateSelectionTaskView,
    GaussianHMMFit,
    CalibrationSlice,
    CalibrationSlice,
]:
    """Return exact parents after full invalid-benchmark reverification."""

    registered_sealed_holdout_invalid_benchmark_identity(value)
    return (
        value._selection_source,
        value._selected_fit_source,
        value._design_source,
        value._holdout_source,
    )


def _mint_failure(
    *,
    selection: CandidateSelectionTaskView,
    fit: GaussianHMMFit,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
    selection_fingerprint: str,
    model_fingerprint: str,
    design_fingerprint: str,
    holdout_fingerprint: str,
    source_fingerprint: str,
    evidence: InvalidHoldoutBenchmarkEvidence,
) -> RegisteredSealedHoldoutInvalidBenchmark:
    failure = _validated_failure_evidence(evidence)
    failure_fingerprint = _failure_fingerprint(
        source_fingerprint=source_fingerprint,
        evidence=failure,
    )
    execution_fingerprint = _failure_execution_fingerprint(
        source_fingerprint=source_fingerprint,
        failure_evidence_fingerprint=failure_fingerprint,
    )
    value = object.__new__(RegisteredSealedHoldoutInvalidBenchmark)
    for name, item in (
        ("contract_version", REGISTERED_HOLDOUT_INVALID_BENCHMARK_CONTRACT),
        ("task_id", "sealed_holdout_veto"),
        ("role", "design"),
        ("outcome", "invalid_benchmark"),
        ("selected_n_states", fit.n_states),
        ("selection_view_fingerprint", selection_fingerprint),
        ("selected_model_fingerprint", model_fingerprint),
        ("design_slice_fingerprint", design_fingerprint),
        ("holdout_slice_fingerprint", holdout_fingerprint),
        ("source_materialization_fingerprint", source_fingerprint),
        ("failure_code", failure.failure_code),
        ("failure_message", failure.failure_message),
        ("failure_details", failure.failure_details),
        ("failure_evidence_fingerprint", failure_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("generation_authorized", False),
        ("_selection_source", selection),
        ("_selected_fit_source", fit),
        ("_design_source", design),
        ("_holdout_source", holdout),
        ("_failure_source", failure),
        ("_execution_seal", _HOLDOUT_FAILURE_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_HOLDOUT_FAILURES[id(value)] = (
        value,
        selection,
        fit,
        design,
        holdout,
        failure,
    )
    registered_sealed_holdout_invalid_benchmark_identity(value)
    return value


def _validated_selection(
    selection: CandidateSelectionTaskView,
) -> tuple[GaussianHMMFit, str]:
    from prpg.model.g3_calibration_views import (
        is_registered_candidate_selection_task_view,
        selected_design_fit_from_view,
    )

    if not is_registered_candidate_selection_task_view(selection):
        raise ModelError(
            "registered holdout requires final fixed-point candidate selection"
        )
    if not selection.passed or selection.selected_n_states is None:
        raise ModelError("registered holdout requires a passed design selection")
    fit = selected_design_fit_from_view(selection)
    if (
        fit.n_states != selection.selected_n_states
        or fit.n_observations != _DESIGN_ROWS
        or fit.lengths != (_DESIGN_ROWS,)
        or fit.scaler_scope != "design_training"
    ):
        raise ModelError("registered holdout selected-fit geometry is invalid")
    return fit, selection.view_fingerprint


def _validate_slice_geometry(
    design: CalibrationSlice,
    holdout: CalibrationSlice,
) -> None:
    if not isinstance(design, CalibrationSlice) or design.role != "design":
        raise ModelError("registered holdout requires the exact design slice")
    if not isinstance(holdout, CalibrationSlice) or holdout.role != "holdout":
        raise ModelError("registered holdout requires the exact holdout slice")
    design_macro = np.asarray(design.macro_features)
    holdout_macro = np.asarray(holdout.macro_features)
    if (
        design_macro.shape != (_DESIGN_ROWS, 4)
        or holdout_macro.shape != (_HOLDOUT_ROWS, 4)
        or not np.isfinite(design_macro).all()
        or not np.isfinite(holdout_macro).all()
        or sequence_lengths_from_continuation(design.monthly_continuation)
        != (_DESIGN_ROWS,)
        or sequence_lengths_from_continuation(holdout.monthly_continuation)
        != _HOLDOUT_LENGTHS
    ):
        raise ModelError("registered holdout slice geometry is invalid")
    calibration_slice_fingerprint(design)
    calibration_slice_fingerprint(holdout)


def _verify_sources_unchanged(
    *,
    selection: CandidateSelectionTaskView,
    fit: GaussianHMMFit,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
    selection_fingerprint: str,
    model_fingerprint: str,
    design_fingerprint: str,
    holdout_fingerprint: str,
) -> None:
    identities = _live_source_identities(selection, fit, design, holdout)
    if identities != (
        selection_fingerprint,
        model_fingerprint,
        design_fingerprint,
        holdout_fingerprint,
    ):
        raise ModelError("registered holdout sources changed during execution")


def _live_source_identities(
    selection: CandidateSelectionTaskView,
    fit: GaussianHMMFit,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
) -> tuple[str, str, str, str]:
    current_fit, selection_fingerprint = _validated_selection(selection)
    if current_fit is not fit:
        raise ModelError("registered holdout substituted its selected fit")
    _validate_slice_geometry(design, holdout)
    return (
        selection_fingerprint,
        gaussian_hmm_fit_fingerprint(fit),
        calibration_slice_fingerprint(design),
        calibration_slice_fingerprint(holdout),
    )


def _immutable_evidence(value: SealedHoldoutEvidence) -> SealedHoldoutEvidence:
    if not isinstance(value, SealedHoldoutEvidence):
        raise ModelError("registered holdout executor returned the wrong result type")
    hmm = _read_only_copy(value.hmm_log_scores)
    gaussian = _read_only_copy(value.benchmark_scores.gaussian)
    ridge = _read_only_copy(value.benchmark_scores.ridge_var1)
    decision = holdout_veto_decision(
        hmm,
        HoldoutBenchmarkScores(gaussian=gaussian, ridge_var1=ridge),
    )
    return SealedHoldoutEvidence(
        sequence_lengths=tuple(value.sequence_lengths),
        hmm_log_scores=hmm,
        benchmark_scores=HoldoutBenchmarkScores(
            gaussian=gaussian,
            ridge_var1=ridge,
        ),
        decision=decision,
    )


def _validate_evidence(value: SealedHoldoutEvidence) -> None:
    if not isinstance(value, SealedHoldoutEvidence):
        raise ModelError("registered holdout evidence has the wrong type")
    if value.sequence_lengths != _HOLDOUT_LENGTHS:
        raise ModelError("registered holdout evidence has the wrong geometry")
    hmm = np.asarray(value.hmm_log_scores)
    gaussian = np.asarray(value.benchmark_scores.gaussian)
    ridge = np.asarray(value.benchmark_scores.ridge_var1)
    if (
        hmm.shape != (_HOLDOUT_ROWS,)
        or gaussian.shape != (_HOLDOUT_ROWS,)
        or ridge.shape != (_HOLDOUT_ROWS,)
        or not np.isfinite(hmm).all()
        or not np.isfinite(gaussian).all()
        or not np.isfinite(ridge).all()
        or hmm.flags.writeable
        or gaussian.flags.writeable
        or ridge.flags.writeable
    ):
        raise ModelError("registered holdout evidence vectors are invalid")
    recomputed = holdout_veto_decision(hmm, value.benchmark_scores)
    if recomputed != value.decision:
        raise ModelError("registered holdout decision differs from its score vectors")


def _validated_failure_evidence(
    value: InvalidHoldoutBenchmarkEvidence,
) -> InvalidHoldoutBenchmarkEvidence:
    if (
        not isinstance(value, InvalidHoldoutBenchmarkEvidence)
        or value.failure_code not in _FAILURE_CODES
        or not isinstance(value.failure_message, str)
        or not value.failure_message
        or not isinstance(value.failure_details, tuple)
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            for item in value.failure_details
        )
        or tuple(sorted(value.failure_details, key=lambda item: item[0]))
        != value.failure_details
        or len({item[0] for item in value.failure_details})
        != len(value.failure_details)
    ):
        raise ModelError("invalid holdout benchmark evidence is malformed")
    _canonical_json(value.failure_details)
    return value


def _source_fingerprint(
    *,
    selected_n_states: int,
    selection_view_fingerprint: str,
    selected_model_fingerprint: str,
    design_slice_fingerprint: str,
    holdout_slice_fingerprint: str,
) -> str:
    return _sha256_json(
        {
            "contract_version": REGISTERED_HOLDOUT_SOURCE_CONTRACT,
            "selected_n_states": selected_n_states,
            "selection_view_fingerprint": selection_view_fingerprint,
            "selected_model_fingerprint": selected_model_fingerprint,
            "design_slice_fingerprint": design_slice_fingerprint,
            "holdout_slice_fingerprint": holdout_slice_fingerprint,
        }
    )


def _success_execution_fingerprint(
    *,
    source_fingerprint: str,
    evidence_fingerprint: str,
) -> str:
    return _sha256_json(
        {
            "contract_version": REGISTERED_HOLDOUT_RESULT_CONTRACT,
            "source_materialization_fingerprint": source_fingerprint,
            "evidence_fingerprint": evidence_fingerprint,
        }
    )


def _failure_fingerprint(
    *,
    source_fingerprint: str,
    evidence: InvalidHoldoutBenchmarkEvidence,
) -> str:
    return _sha256_json(
        {
            "contract_version": REGISTERED_HOLDOUT_FAILURE_CONTRACT,
            "source_materialization_fingerprint": source_fingerprint,
            "failure_code": evidence.failure_code,
            "failure_message": evidence.failure_message,
            "failure_details": evidence.failure_details,
        }
    )


def _failure_execution_fingerprint(
    *,
    source_fingerprint: str,
    failure_evidence_fingerprint: str,
) -> str:
    return _sha256_json(
        {
            "contract_version": REGISTERED_HOLDOUT_INVALID_BENCHMARK_CONTRACT,
            "source_materialization_fingerprint": source_fingerprint,
            "failure_evidence_fingerprint": failure_evidence_fingerprint,
        }
    )


def _read_only_copy(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _normalize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ModelError("holdout fingerprint contains a noncanonical value")
