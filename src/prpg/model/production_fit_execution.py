"""Cycle-free registered execution of the sole production fixed-K fit.

The G3 production fit is a producer, not a task view.  It must therefore run
before the task binding is minted.  This module owns that execution boundary:
it derives the exact 50 scope-4/replicate-0 restart seeds, runs the ordinary
production fitter once, freezes the returned scientific tree, and retains the
exact candidate-selection and production-feature parents.

No fitter is accepted as an argument.  Report-only callers that need an
injectable exploratory fit continue to use :func:`fit_production_selected_k`
directly; such results do not acquire the object-ledger authority defined
here.  A fitting error is deliberately allowed to propagate.  Converting an
arbitrary exception into a scientific failure would invent evidence and is a
runner-level concern governed by the task failure protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeGuard, cast

import numpy as np

from prpg.errors import ModelError
from prpg.model.calibration import fit_production_selected_k
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMFeatureMatrix,
    NoNumericallyValidHMMRestarts,
    RestartDiagnostic,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_restarts,
    is_numerical_restart_failure_diagnostic,
)
from prpg.model.input import MACRO_COLUMNS
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    rng_execution_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.regime_policy import owner_fixed_k4_policy_for_scientific_version
from prpg.simulation.rng import ModelFitScope, Stage

if TYPE_CHECKING:
    from prpg.model.g3_calibration_views import CandidateSelectionTaskView


PRODUCTION_FIT_EXECUTION_CONTRACT = "registered-production-fit-same-k-v1"
PRODUCTION_FIT_SOURCE_CONTRACT = "production-fit-source-materialization-v1"
PRODUCTION_FIT_RESULT_CONTRACT = "production-fit-result-evidence-v1"
PRODUCTION_FIT_NO_VALID_RESTARTS_CONTRACT = (
    "registered-production-fit-no-valid-restarts-v1"
)
PRODUCTION_FIT_FAILURE_RESULT_CONTRACT = "production-fit-no-valid-restarts-evidence-v1"
_PRODUCTION_FIT_EXECUTION_CAPABILITY = object()
_PRODUCTION_FIT_FAILURE_CAPABILITY = object()
_CANONICAL_PRODUCTION_LENGTHS = (210, 7)
_CANONICAL_PRODUCTION_ROWS = sum(_CANONICAL_PRODUCTION_LENGTHS)


@dataclass(frozen=True, slots=True, init=False)
class RegisteredProductionFitSameK:
    """One immutable production fit with exact prebinding ancestry."""

    contract_version: str
    task_id: Literal["production_fit_same_k"]
    role: Literal["production"]
    design_selected_k: int
    master_seed: int
    scientific_version: int
    selection_view_fingerprint: str
    design_model_fingerprint: str
    production_feature_fingerprint: str
    restart_seeds: tuple[int, ...]
    restart_seed_materialization_fingerprint: str
    rng_namespace: Literal["production_fit_restarts"]
    rng_execution_fingerprint: str
    source_materialization_fingerprint: str
    production_fit: GaussianHMMFit
    production_fit_fingerprint: str
    result_evidence_fingerprint: str
    execution_fingerprint: str
    generation_authorized: Literal[False]
    _selection_source: CandidateSelectionTaskView = field(
        repr=False,
        compare=False,
    )
    _design_fit_source: GaussianHMMFit = field(repr=False, compare=False)
    _production_features_source: HMMFeatureMatrix = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredProductionFitSameK is created only by "
            "execute_registered_production_fit_same_k"
        )


@dataclass(frozen=True, slots=True)
class RegisteredProductionFitSameKIdentity:
    """Reverified source, RNG, output, and execution identities."""

    design_selected_k: int
    master_seed: int
    scientific_version: int
    selection_view_fingerprint: str
    design_model_fingerprint: str
    production_feature_fingerprint: str
    restart_seed_materialization_fingerprint: str
    rng_execution_fingerprint: str
    source_materialization_fingerprint: str
    production_fit_fingerprint: str
    result_evidence_fingerprint: str
    execution_fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class RegisteredProductionFitNoValidRestarts:
    """Sealed task-11 scientific failure after all fifty invalid restarts."""

    contract_version: str
    task_id: Literal["production_fit_same_k"]
    role: Literal["production"]
    outcome: Literal["no_numerically_valid_restarts"]
    design_selected_k: int
    master_seed: int
    scientific_version: int
    selection_view_fingerprint: str
    design_model_fingerprint: str
    production_feature_fingerprint: str
    restart_seeds: tuple[int, ...]
    restart_seed_materialization_fingerprint: str
    rng_namespace: Literal["production_fit_restarts"]
    rng_execution_fingerprint: str
    source_materialization_fingerprint: str
    restart_diagnostics: tuple[RestartDiagnostic, ...]
    restart_diagnostics_fingerprint: str
    failure_evidence_fingerprint: str
    execution_fingerprint: str
    generation_authorized: Literal[False]
    _selection_source: CandidateSelectionTaskView = field(
        repr=False,
        compare=False,
    )
    _design_fit_source: GaussianHMMFit = field(repr=False, compare=False)
    _production_features_source: HMMFeatureMatrix = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredProductionFitNoValidRestarts is created only by "
            "execute_registered_production_fit_same_k"
        )


@dataclass(frozen=True, slots=True)
class RegisteredProductionFitNoValidRestartsIdentity:
    """Reverified source, RNG, and all-invalid diagnostic identity."""

    design_selected_k: int
    master_seed: int
    scientific_version: int
    selection_view_fingerprint: str
    design_model_fingerprint: str
    production_feature_fingerprint: str
    restart_seed_materialization_fingerprint: str
    rng_execution_fingerprint: str
    source_materialization_fingerprint: str
    restart_diagnostics_fingerprint: str
    failure_evidence_fingerprint: str
    execution_fingerprint: str


RegisteredProductionFitOutcome: TypeAlias = (
    RegisteredProductionFitSameK | RegisteredProductionFitNoValidRestarts
)


_MINTED_PRODUCTION_FITS: dict[
    int,
    tuple[
        RegisteredProductionFitSameK,
        CandidateSelectionTaskView,
        GaussianHMMFit,
        HMMFeatureMatrix,
        GaussianHMMFit,
    ],
] = {}

_MINTED_PRODUCTION_FIT_FAILURES: dict[
    int,
    tuple[
        RegisteredProductionFitNoValidRestarts,
        CandidateSelectionTaskView,
        GaussianHMMFit,
        HMMFeatureMatrix,
        tuple[RestartDiagnostic, ...],
    ],
] = {}


def execute_registered_production_fit_same_k(
    selection: CandidateSelectionTaskView,
    production_features: HMMFeatureMatrix,
    *,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
) -> RegisteredProductionFitOutcome:
    """Execute exactly one noninjectable fixed-K production fit.

    The selected K and design model are read from the exact sealed selection
    object.  The production matrix must have the approved 217-row ``(210, 7)``
    full-history geometry and the four registered macro columns.
    """

    worker_count = _validated_workers(workers)
    design_fit, selection_fingerprint = _validated_selection_sources(selection)
    _validate_production_features(production_features)
    design_model_fingerprint = gaussian_hmm_fit_fingerprint(design_fit)
    production_feature_fingerprint = hmm_feature_matrix_fingerprint(production_features)
    seeds, seed_fingerprint, rng_fingerprint = _restart_identity(
        selected_k=design_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    source_fingerprint = _source_materialization_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
        selection_view_fingerprint=selection_fingerprint,
        design_model_fingerprint=design_model_fingerprint,
        production_feature_fingerprint=production_feature_fingerprint,
        restart_seed_materialization_fingerprint=seed_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
    )

    policy = owner_fixed_k4_policy_for_scientific_version(scientific_version)
    try:
        if worker_count == 1 and policy is None:
            fitted = fit_production_selected_k(
                production_features,
                selected_n_states=design_fit.n_states,
                master_seed=master_seed,
                scientific_version=scientific_version,
            )
        else:
            fitted = fit_production_selected_k(
                production_features,
                selected_n_states=design_fit.n_states,
                master_seed=master_seed,
                scientific_version=scientific_version,
                fit_function=(
                    partial(fit_gaussian_hmm_restarts, workers=worker_count)
                    if policy is None
                    else partial(
                        fit_gaussian_hmm_restarts,
                        workers=worker_count,
                        owner_fixed_k4_policy=policy,
                    )
                ),
            )
    except NoNumericallyValidHMMRestarts as failure:
        _verify_execution_sources_unchanged(
            selection=selection,
            production_features=production_features,
            design_fit=design_fit,
            selection_fingerprint=selection_fingerprint,
            design_model_fingerprint=design_model_fingerprint,
            production_feature_fingerprint=production_feature_fingerprint,
        )
        return _mint_no_valid_restarts(
            selection=selection,
            design_fit=design_fit,
            production_features=production_features,
            master_seed=master_seed,
            scientific_version=scientific_version,
            selection_fingerprint=selection_fingerprint,
            design_model_fingerprint=design_model_fingerprint,
            production_feature_fingerprint=production_feature_fingerprint,
            seeds=seeds,
            seed_fingerprint=seed_fingerprint,
            rng_fingerprint=rng_fingerprint,
            source_fingerprint=source_fingerprint,
            failure=failure,
        )
    if not isinstance(fitted, GaussianHMMFit):
        raise ModelError("registered production executor returned a non-HMM fit")
    _verify_execution_sources_unchanged(
        selection=selection,
        production_features=production_features,
        design_fit=design_fit,
        selection_fingerprint=selection_fingerprint,
        design_model_fingerprint=design_model_fingerprint,
        production_feature_fingerprint=production_feature_fingerprint,
    )

    # The fitter has completed.  Freeze its entire public scientific tree so
    # downstream consumers cannot mutate evidence that has acquired authority.
    production_fit = cast(GaussianHMMFit, _immutable_public_tree(fitted))
    _validate_production_result(
        production_fit,
        production_features,
        selected_k=design_fit.n_states,
        restart_seeds=seeds,
    )
    production_fit_fingerprint = gaussian_hmm_fit_fingerprint(production_fit)
    result_evidence_fingerprint = _fit_result_evidence_fingerprint(production_fit)
    execution_fingerprint = _execution_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
        source_materialization_fingerprint=source_fingerprint,
        production_fit_fingerprint=production_fit_fingerprint,
        result_evidence_fingerprint=result_evidence_fingerprint,
    )

    value = object.__new__(RegisteredProductionFitSameK)
    for name, item in (
        ("contract_version", PRODUCTION_FIT_EXECUTION_CONTRACT),
        ("task_id", "production_fit_same_k"),
        ("role", "production"),
        ("design_selected_k", design_fit.n_states),
        ("master_seed", master_seed),
        ("scientific_version", scientific_version),
        ("selection_view_fingerprint", selection_fingerprint),
        ("design_model_fingerprint", design_model_fingerprint),
        ("production_feature_fingerprint", production_feature_fingerprint),
        ("restart_seeds", seeds),
        ("restart_seed_materialization_fingerprint", seed_fingerprint),
        ("rng_namespace", "production_fit_restarts"),
        ("rng_execution_fingerprint", rng_fingerprint),
        ("source_materialization_fingerprint", source_fingerprint),
        ("production_fit", production_fit),
        ("production_fit_fingerprint", production_fit_fingerprint),
        ("result_evidence_fingerprint", result_evidence_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("generation_authorized", False),
        ("_selection_source", selection),
        ("_design_fit_source", design_fit),
        ("_production_features_source", production_features),
        ("_execution_seal", _PRODUCTION_FIT_EXECUTION_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_PRODUCTION_FITS[id(value)] = (
        value,
        selection,
        design_fit,
        production_features,
        production_fit,
    )
    registered_production_fit_same_k_identity(value)
    return value


def _verify_execution_sources_unchanged(
    *,
    selection: CandidateSelectionTaskView,
    production_features: HMMFeatureMatrix,
    design_fit: GaussianHMMFit,
    selection_fingerprint: str,
    design_model_fingerprint: str,
    production_feature_fingerprint: str,
) -> None:
    current_design_fit, current_selection_fingerprint = _validated_selection_sources(
        selection
    )
    if (
        current_design_fit is not design_fit
        or current_selection_fingerprint != selection_fingerprint
        or gaussian_hmm_fit_fingerprint(design_fit) != design_model_fingerprint
        or hmm_feature_matrix_fingerprint(production_features)
        != production_feature_fingerprint
    ):
        raise ModelError("production-fit sources changed during execution")


def _mint_no_valid_restarts(
    *,
    selection: CandidateSelectionTaskView,
    design_fit: GaussianHMMFit,
    production_features: HMMFeatureMatrix,
    master_seed: int,
    scientific_version: int,
    selection_fingerprint: str,
    design_model_fingerprint: str,
    production_feature_fingerprint: str,
    seeds: tuple[int, ...],
    seed_fingerprint: str,
    rng_fingerprint: str,
    source_fingerprint: str,
    failure: NoNumericallyValidHMMRestarts,
) -> RegisteredProductionFitNoValidRestarts:
    if failure.n_states != design_fit.n_states:
        raise ModelError("production no-valid-restarts signal has the wrong K")
    diagnostics = cast(
        tuple[RestartDiagnostic, ...],
        _immutable_public_tree(failure.restart_diagnostics),
    )
    _validate_no_valid_restart_diagnostics(diagnostics, restart_seeds=seeds)
    diagnostics_fingerprint = _restart_diagnostics_fingerprint(diagnostics)
    failure_evidence_fingerprint = _failure_evidence_fingerprint(
        selected_k=design_fit.n_states,
        restart_diagnostics=diagnostics,
        restart_diagnostics_fingerprint=diagnostics_fingerprint,
    )
    execution_fingerprint = _failure_execution_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
        source_materialization_fingerprint=source_fingerprint,
        restart_diagnostics_fingerprint=diagnostics_fingerprint,
        failure_evidence_fingerprint=failure_evidence_fingerprint,
    )
    value = object.__new__(RegisteredProductionFitNoValidRestarts)
    for name, item in (
        ("contract_version", PRODUCTION_FIT_NO_VALID_RESTARTS_CONTRACT),
        ("task_id", "production_fit_same_k"),
        ("role", "production"),
        ("outcome", "no_numerically_valid_restarts"),
        ("design_selected_k", design_fit.n_states),
        ("master_seed", master_seed),
        ("scientific_version", scientific_version),
        ("selection_view_fingerprint", selection_fingerprint),
        ("design_model_fingerprint", design_model_fingerprint),
        ("production_feature_fingerprint", production_feature_fingerprint),
        ("restart_seeds", seeds),
        ("restart_seed_materialization_fingerprint", seed_fingerprint),
        ("rng_namespace", "production_fit_restarts"),
        ("rng_execution_fingerprint", rng_fingerprint),
        ("source_materialization_fingerprint", source_fingerprint),
        ("restart_diagnostics", diagnostics),
        ("restart_diagnostics_fingerprint", diagnostics_fingerprint),
        ("failure_evidence_fingerprint", failure_evidence_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("generation_authorized", False),
        ("_selection_source", selection),
        ("_design_fit_source", design_fit),
        ("_production_features_source", production_features),
        ("_execution_seal", _PRODUCTION_FIT_FAILURE_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_PRODUCTION_FIT_FAILURES[id(value)] = (
        value,
        selection,
        design_fit,
        production_features,
        diagnostics,
    )
    registered_production_fit_no_valid_restarts_identity(value)
    return value


def registered_production_fit_same_k_identity(
    value: RegisteredProductionFitSameK,
) -> RegisteredProductionFitSameKIdentity:
    """Revalidate one exact object minted by the registered executor."""

    if not isinstance(value, RegisteredProductionFitSameK):
        raise ModelError("registered production fit lacks executor authority")
    ledger = _MINTED_PRODUCTION_FITS.get(id(value))
    if (
        value._execution_seal is not _PRODUCTION_FIT_EXECUTION_CAPABILITY
        or ledger is None
        or ledger[0] is not value
        or ledger[1] is not value._selection_source
        or ledger[2] is not value._design_fit_source
        or ledger[3] is not value._production_features_source
        or ledger[4] is not value.production_fit
    ):
        raise ModelError("registered production fit lacks executor authority")
    if (
        value.contract_version != PRODUCTION_FIT_EXECUTION_CONTRACT
        or value.task_id != "production_fit_same_k"
        or value.role != "production"
        or value.rng_namespace != "production_fit_restarts"
        or value.generation_authorized is not False
    ):
        raise ModelError("registered production-fit contract changed")

    design_fit, selection_fingerprint = _validated_selection_sources(
        value._selection_source
    )
    if design_fit is not value._design_fit_source:
        raise ModelError("registered production fit has substituted design ancestry")
    features = value._production_features_source
    if not isinstance(features, HMMFeatureMatrix):
        raise ModelError("registered production fit lost its feature parent")
    _validate_production_features(features)
    design_model_fingerprint = gaussian_hmm_fit_fingerprint(design_fit)
    feature_fingerprint = hmm_feature_matrix_fingerprint(features)
    seeds, seed_fingerprint, rng_fingerprint = _restart_identity(
        selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
    )
    source_fingerprint = _source_materialization_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        selection_view_fingerprint=selection_fingerprint,
        design_model_fingerprint=design_model_fingerprint,
        production_feature_fingerprint=feature_fingerprint,
        restart_seed_materialization_fingerprint=seed_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
    )
    _validate_production_result(
        value.production_fit,
        features,
        selected_k=design_fit.n_states,
        restart_seeds=seeds,
    )
    production_fit_fingerprint = gaussian_hmm_fit_fingerprint(value.production_fit)
    result_evidence_fingerprint = _fit_result_evidence_fingerprint(value.production_fit)
    execution_fingerprint = _execution_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        source_materialization_fingerprint=source_fingerprint,
        production_fit_fingerprint=production_fit_fingerprint,
        result_evidence_fingerprint=result_evidence_fingerprint,
    )
    if (
        value.design_selected_k != design_fit.n_states
        or value.selection_view_fingerprint != selection_fingerprint
        or value.design_model_fingerprint != design_model_fingerprint
        or value.production_feature_fingerprint != feature_fingerprint
        or value.restart_seeds != seeds
        or value.restart_seed_materialization_fingerprint != seed_fingerprint
        or value.rng_execution_fingerprint != rng_fingerprint
        or value.source_materialization_fingerprint != source_fingerprint
        or value.production_fit_fingerprint != production_fit_fingerprint
        or value.result_evidence_fingerprint != result_evidence_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("registered production-fit sources or result changed")
    return RegisteredProductionFitSameKIdentity(
        design_selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        selection_view_fingerprint=selection_fingerprint,
        design_model_fingerprint=design_model_fingerprint,
        production_feature_fingerprint=feature_fingerprint,
        restart_seed_materialization_fingerprint=seed_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
        source_materialization_fingerprint=source_fingerprint,
        production_fit_fingerprint=production_fit_fingerprint,
        result_evidence_fingerprint=result_evidence_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def is_registered_production_fit_same_k(
    value: object,
) -> TypeGuard[RegisteredProductionFitSameK]:
    """Return whether ``value`` is the unchanged executor-minted instance."""

    if not isinstance(value, RegisteredProductionFitSameK):
        return False
    try:
        registered_production_fit_same_k_identity(value)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_production_fit_same_k_sources(
    value: RegisteredProductionFitSameK,
) -> tuple[
    CandidateSelectionTaskView,
    GaussianHMMFit,
    HMMFeatureMatrix,
    GaussianHMMFit,
]:
    """Return exact parent and result objects after full reverification."""

    registered_production_fit_same_k_identity(value)
    return (
        value._selection_source,
        value._design_fit_source,
        value._production_features_source,
        value.production_fit,
    )


def registered_production_fit_no_valid_restarts_identity(
    value: RegisteredProductionFitNoValidRestarts,
) -> RegisteredProductionFitNoValidRestartsIdentity:
    """Revalidate an executor-minted all-invalid production-fit outcome."""

    if not isinstance(value, RegisteredProductionFitNoValidRestarts):
        raise ModelError("production no-valid-restarts outcome has the wrong type")
    ledger = _MINTED_PRODUCTION_FIT_FAILURES.get(id(value))
    if (
        value._execution_seal is not _PRODUCTION_FIT_FAILURE_CAPABILITY
        or ledger is None
        or ledger[0] is not value
        or ledger[1] is not value._selection_source
        or ledger[2] is not value._design_fit_source
        or ledger[3] is not value._production_features_source
        or ledger[4] is not value.restart_diagnostics
    ):
        raise ModelError(
            "production no-valid-restarts outcome lacks executor authority"
        )
    if (
        value.contract_version != PRODUCTION_FIT_NO_VALID_RESTARTS_CONTRACT
        or value.task_id != "production_fit_same_k"
        or value.role != "production"
        or value.outcome != "no_numerically_valid_restarts"
        or value.rng_namespace != "production_fit_restarts"
        or value.generation_authorized is not False
    ):
        raise ModelError("production no-valid-restarts contract changed")

    design_fit, selection_fingerprint = _validated_selection_sources(
        value._selection_source
    )
    if design_fit is not value._design_fit_source:
        raise ModelError(
            "production no-valid-restarts outcome substituted design ancestry"
        )
    features = value._production_features_source
    if not isinstance(features, HMMFeatureMatrix):
        raise ModelError("production no-valid-restarts outcome lost its feature parent")
    _validate_production_features(features)
    design_model_fingerprint = gaussian_hmm_fit_fingerprint(design_fit)
    feature_fingerprint = hmm_feature_matrix_fingerprint(features)
    seeds, seed_fingerprint, rng_fingerprint = _restart_identity(
        selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
    )
    source_fingerprint = _source_materialization_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        selection_view_fingerprint=selection_fingerprint,
        design_model_fingerprint=design_model_fingerprint,
        production_feature_fingerprint=feature_fingerprint,
        restart_seed_materialization_fingerprint=seed_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
    )
    _validate_no_valid_restart_diagnostics(
        value.restart_diagnostics,
        restart_seeds=seeds,
    )
    diagnostics_fingerprint = _restart_diagnostics_fingerprint(
        value.restart_diagnostics
    )
    failure_evidence_fingerprint = _failure_evidence_fingerprint(
        selected_k=design_fit.n_states,
        restart_diagnostics=value.restart_diagnostics,
        restart_diagnostics_fingerprint=diagnostics_fingerprint,
    )
    execution_fingerprint = _failure_execution_fingerprint(
        design_selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        source_materialization_fingerprint=source_fingerprint,
        restart_diagnostics_fingerprint=diagnostics_fingerprint,
        failure_evidence_fingerprint=failure_evidence_fingerprint,
    )
    if (
        value.design_selected_k != design_fit.n_states
        or value.selection_view_fingerprint != selection_fingerprint
        or value.design_model_fingerprint != design_model_fingerprint
        or value.production_feature_fingerprint != feature_fingerprint
        or value.restart_seeds != seeds
        or value.restart_seed_materialization_fingerprint != seed_fingerprint
        or value.rng_execution_fingerprint != rng_fingerprint
        or value.source_materialization_fingerprint != source_fingerprint
        or value.restart_diagnostics_fingerprint != diagnostics_fingerprint
        or value.failure_evidence_fingerprint != failure_evidence_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("production no-valid-restarts sources or diagnostics changed")
    return RegisteredProductionFitNoValidRestartsIdentity(
        design_selected_k=design_fit.n_states,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        selection_view_fingerprint=selection_fingerprint,
        design_model_fingerprint=design_model_fingerprint,
        production_feature_fingerprint=feature_fingerprint,
        restart_seed_materialization_fingerprint=seed_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
        source_materialization_fingerprint=source_fingerprint,
        restart_diagnostics_fingerprint=diagnostics_fingerprint,
        failure_evidence_fingerprint=failure_evidence_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def is_registered_production_fit_no_valid_restarts(
    value: object,
) -> TypeGuard[RegisteredProductionFitNoValidRestarts]:
    """Return whether ``value`` is the unchanged all-invalid outcome."""

    if not isinstance(value, RegisteredProductionFitNoValidRestarts):
        return False
    try:
        registered_production_fit_no_valid_restarts_identity(value)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_production_fit_no_valid_restarts_sources(
    value: RegisteredProductionFitNoValidRestarts,
) -> tuple[CandidateSelectionTaskView, GaussianHMMFit, HMMFeatureMatrix]:
    """Return exact parents after full all-invalid outcome reverification."""

    registered_production_fit_no_valid_restarts_identity(value)
    return (
        value._selection_source,
        value._design_fit_source,
        value._production_features_source,
    )


def _validated_selection_sources(
    selection: CandidateSelectionTaskView,
) -> tuple[GaussianHMMFit, str]:
    # Lazy import prevents a cycle when the task-view module imports this
    # producer while assembling the pure post-binding production view.
    from prpg.model.g3_calibration_views import (
        is_registered_candidate_selection_task_view,
        selected_design_fit_from_view,
    )

    if not is_registered_candidate_selection_task_view(selection):
        raise ModelError(
            "registered production fit requires final fixed-point candidate selection"
        )
    if not selection.passed or selection.selected_n_states is None:
        raise ModelError("registered production fit requires a passed design selection")
    fit = selected_design_fit_from_view(selection)
    if fit.n_states != selection.selected_n_states:
        raise ModelError("design selection K and retained fit disagree")
    return fit, selection.view_fingerprint


def _validated_workers(value: int) -> int:
    """Validate the local restart-pool width before scientific work begins."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError("production-fit workers must be a positive integer")
    return value


def _validate_production_features(features: HMMFeatureMatrix) -> None:
    if not isinstance(features, HMMFeatureMatrix):
        raise ModelError("registered production fit requires HMMFeatureMatrix input")
    values = np.asarray(features.values)
    means = np.asarray(features.scaler_means)
    scales = np.asarray(features.scaler_standard_deviations)
    if (
        features.scaler_scope != "production_full"
        or features.lengths != _CANONICAL_PRODUCTION_LENGTHS
        or values.shape != (_CANONICAL_PRODUCTION_ROWS, len(MACRO_COLUMNS))
        or features.feature_names != tuple(MACRO_COLUMNS)
        or len(features.row_labels) != _CANONICAL_PRODUCTION_ROWS
        or means.shape != (len(MACRO_COLUMNS),)
        or scales.shape != (len(MACRO_COLUMNS),)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or bool((scales <= 0.0).any())
    ):
        raise ModelError("registered production feature geometry is invalid")
    # This shared identity reader also rejects malformed labels and lengths.
    hmm_feature_matrix_fingerprint(features)


def _validate_production_result(
    fit: GaussianHMMFit,
    features: HMMFeatureMatrix,
    *,
    selected_k: int,
    restart_seeds: tuple[int, ...],
) -> None:
    if not isinstance(fit, GaussianHMMFit):
        raise ModelError("registered production result must be GaussianHMMFit")
    diagnostics = fit.restart_diagnostics
    if (
        fit.n_states != selected_k
        or fit.n_observations != _CANONICAL_PRODUCTION_ROWS
        or fit.n_observations != len(features.values)
        or fit.n_features != len(MACRO_COLUMNS)
        or fit.lengths != features.lengths
        or fit.feature_names != features.feature_names
        or fit.scaler_scope != "production_full"
        or not np.array_equal(fit.scaler_means, features.scaler_means)
        or not np.array_equal(
            fit.scaler_standard_deviations,
            features.scaler_standard_deviations,
        )
        or len(restart_seeds) != HMM_RESTARTS
        or len(diagnostics) != HMM_RESTARTS
        or fit.best_restart_index not in range(HMM_RESTARTS)
        or fit.best_seed != restart_seeds[fit.best_restart_index]
    ):
        raise ModelError("registered production fit is inconsistent with its sources")
    for index, (diagnostic, seed) in enumerate(
        zip(diagnostics, restart_seeds, strict=True)
    ):
        if (
            not isinstance(diagnostic, RestartDiagnostic)
            or diagnostic.restart_index != index
            or diagnostic.seed != seed
        ):
            raise ModelError(
                "production restart diagnostics differ from registered seeds"
            )
    best = diagnostics[fit.best_restart_index]
    if (
        not best.valid
        or not best.converged
        or best.log_likelihood is None
        or best.log_likelihood != fit.log_likelihood
    ):
        raise ModelError("production best-restart evidence is inconsistent")
    gaussian_hmm_fit_fingerprint(fit)
    _require_immutable_public_arrays(fit)


def _validate_no_valid_restart_diagnostics(
    diagnostics: tuple[RestartDiagnostic, ...],
    *,
    restart_seeds: tuple[int, ...],
) -> None:
    if len(restart_seeds) != HMM_RESTARTS or len(diagnostics) != HMM_RESTARTS:
        raise ModelError(
            "production no-valid-restarts outcome requires exactly 50 diagnostics"
        )
    for index, (diagnostic, seed) in enumerate(
        zip(diagnostics, restart_seeds, strict=True)
    ):
        if (
            not isinstance(diagnostic, RestartDiagnostic)
            or diagnostic.restart_index != index
            or diagnostic.seed != seed
            or not is_numerical_restart_failure_diagnostic(diagnostic)
        ):
            raise ModelError(
                "production no-valid-restarts diagnostics differ from the exact "
                "registered seed schedule"
            )


def _restart_diagnostics_fingerprint(
    diagnostics: tuple[RestartDiagnostic, ...],
) -> str:
    return _json_fingerprint(
        {
            "contract_version": PRODUCTION_FIT_FAILURE_RESULT_CONTRACT,
            "restart_diagnostics": _canonical_tree(diagnostics),
        }
    )


def _failure_evidence_fingerprint(
    *,
    selected_k: int,
    restart_diagnostics: tuple[RestartDiagnostic, ...],
    restart_diagnostics_fingerprint: str,
) -> str:
    return _json_fingerprint(
        {
            "contract_version": PRODUCTION_FIT_FAILURE_RESULT_CONTRACT,
            "task_id": "production_fit_same_k",
            "role": "production",
            "outcome": "no_numerically_valid_restarts",
            "selected_k": selected_k,
            "restart_count": HMM_RESTARTS,
            "restart_diagnostics_fingerprint": (restart_diagnostics_fingerprint),
            "restart_diagnostics": _canonical_tree(restart_diagnostics),
        }
    )


def _failure_execution_fingerprint(
    *,
    design_selected_k: int,
    master_seed: int,
    scientific_version: int,
    source_materialization_fingerprint: str,
    restart_diagnostics_fingerprint: str,
    failure_evidence_fingerprint: str,
) -> str:
    return _json_fingerprint(
        {
            "contract_version": PRODUCTION_FIT_NO_VALID_RESTARTS_CONTRACT,
            "task_id": "production_fit_same_k",
            "role": "production",
            "outcome": "no_numerically_valid_restarts",
            "design_selected_k": design_selected_k,
            "master_seed": master_seed,
            "scientific_version": scientific_version,
            "source_materialization_fingerprint": (source_materialization_fingerprint),
            "restart_diagnostics_fingerprint": (restart_diagnostics_fingerprint),
            "failure_evidence_fingerprint": failure_evidence_fingerprint,
            "generation_authorized": False,
        }
    )


def _restart_identity(
    *,
    selected_k: int,
    master_seed: int,
    scientific_version: int,
) -> tuple[tuple[int, ...], str, str]:
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=selected_k,
    )
    if len(seeds) != HMM_RESTARTS or len(set(seeds)) != HMM_RESTARTS:
        raise ModelError("production restart materialization must contain 50 seeds")
    materialization_fingerprint = scientific_array_fingerprint(
        np.asarray(seeds, dtype=np.uint32),
        role="production_fit_restart_seeds",
    )
    execution_fingerprint = rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="production_fit_restarts",
        contract={
            "model_fit_scope": int(ModelFitScope.PRODUCTION_MAIN_AND_FOLDS),
            "n_states": selected_k,
            "replicate": 0,
            "restarts": HMM_RESTARTS,
            "stage": int(Stage.HMM_LIBRARY_SEED),
        },
    )
    return seeds, materialization_fingerprint, execution_fingerprint


def _source_materialization_fingerprint(
    *,
    design_selected_k: int,
    master_seed: int,
    scientific_version: int,
    selection_view_fingerprint: str,
    design_model_fingerprint: str,
    production_feature_fingerprint: str,
    restart_seed_materialization_fingerprint: str,
    rng_execution_fingerprint: str,
) -> str:
    return _json_fingerprint(
        {
            "contract_version": PRODUCTION_FIT_SOURCE_CONTRACT,
            "task_id": "production_fit_same_k",
            "role": "production",
            "design_selected_k": design_selected_k,
            "master_seed": master_seed,
            "scientific_version": scientific_version,
            "selection_view_fingerprint": selection_view_fingerprint,
            "design_model_fingerprint": design_model_fingerprint,
            "production_feature_fingerprint": production_feature_fingerprint,
            "restart_seed_materialization_fingerprint": (
                restart_seed_materialization_fingerprint
            ),
            "rng_execution_fingerprint": rng_execution_fingerprint,
        }
    )


def _execution_fingerprint(
    *,
    design_selected_k: int,
    master_seed: int,
    scientific_version: int,
    source_materialization_fingerprint: str,
    production_fit_fingerprint: str,
    result_evidence_fingerprint: str,
) -> str:
    return _json_fingerprint(
        {
            "contract_version": PRODUCTION_FIT_EXECUTION_CONTRACT,
            "task_id": "production_fit_same_k",
            "role": "production",
            "design_selected_k": design_selected_k,
            "master_seed": master_seed,
            "scientific_version": scientific_version,
            "source_materialization_fingerprint": source_materialization_fingerprint,
            "production_fit_fingerprint": production_fit_fingerprint,
            "result_evidence_fingerprint": result_evidence_fingerprint,
            "generation_authorized": False,
        }
    )


def _fit_result_evidence_fingerprint(fit: GaussianHMMFit) -> str:
    return _json_fingerprint(
        {
            "contract_version": PRODUCTION_FIT_RESULT_CONTRACT,
            "fit": _canonical_tree(fit),
        }
    )


def _canonical_tree(value: object) -> object:
    """Translate a typed scientific tree to deterministic JSON metadata."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        if math.isnan(number):
            return {"__float__": "nan"}
        if math.isinf(number):
            return {"__float__": "inf" if number > 0.0 else "-inf"}
        return number
    if isinstance(value, Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_tree(value.value),
        }
    if isinstance(value, np.ndarray):
        canonical = _canonical_array_bytes(value)
        return {
            "__array__": True,
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
            "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, tuple | list):
        return [_canonical_tree(item) for item in value]
    if isinstance(value, Mapping):
        checked: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelError("scientific result mappings require string keys")
            checked[key] = _canonical_tree(item)
        return {key: checked[key] for key in sorted(checked)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_tree(getattr(value, item.name))
                for item in fields(value)
                if not item.name.startswith("_")
            },
        }
    raise ModelError(
        "production fit contains unsupported result evidence",
        details={"type": type(value).__qualname__},
    )


def _canonical_array_bytes(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ModelError("production fit contains an unsupported array dtype")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ModelError("production fit contains a non-finite result array")
    dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _immutable_public_tree(value: object) -> object:
    if isinstance(value, np.ndarray):
        canonical = np.ascontiguousarray(value)
        return np.frombuffer(
            canonical.tobytes(order="C"),
            dtype=canonical.dtype,
        ).reshape(canonical.shape)
    if isinstance(value, tuple):
        return tuple(_immutable_public_tree(item) for item in value)
    if isinstance(value, list):
        return tuple(_immutable_public_tree(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: _immutable_public_tree(getattr(value, item.name))
            for item in fields(value)
            if item.init and not item.name.startswith("_")
        }
        return replace(value, **updates)
    return value


def _require_immutable_public_arrays(value: object) -> None:
    if isinstance(value, np.ndarray):
        _require_immutable_bytes_array(value)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _require_immutable_public_arrays(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if not item.name.startswith("_"):
                _require_immutable_public_arrays(getattr(value, item.name))


def _require_immutable_bytes_array(value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.flags.writeable or not array.flags.c_contiguous:
        raise ModelError("production fit arrays must be read-only and contiguous")
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        raise ModelError("production fit arrays must be backed by immutable bytes")


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
