"""Deterministic preparation and ordinary-fit core for Phase-3 calibration.

The expensive stability, adequacy, block, and bridge simulations are separate
stages.  This module establishes their common artifact-scoped inputs: exact
feature scaling, preliminary PPW lengths, candidate fit attempts, and the
monthly-to-daily decoded-state mapping.  It performs no file I/O and never
selects a candidate after a downstream veto.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypeGuard, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from prpg.data.transform import StandardizedFeatures, standardize_features
from prpg.errors import ModelError
from prpg.model.benchmarks import (
    HoldoutBenchmarkScores,
    HoldoutVetoDecision,
    holdout_veto_decision,
    score_holdout_benchmarks,
)
from prpg.model.block_length import (
    FrequencyCalibration,
    MacroCalibration,
    PreliminaryBlockLengthCapFailure,
    PreliminaryPPWInsufficientSupport,
    PreliminaryPPWNumericalFailure,
    estimate_frequency_start,
    estimate_macro_start,
)
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMCandidateInadmissibility,
    HMMFeatureMatrix,
    NoNumericallyValidHMMRestarts,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_restarts,
    one_step_predictive_log_scores,
    sequence_lengths_from_continuation,
    summarize_state_path,
)
from prpg.model.input import MACRO_COLUMNS, CalibrationSlice
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    rng_execution_fingerprint,
    scientific_array_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.model.regime_policy import owner_fixed_k4_policy_for_scientific_version
from prpg.model.selection import (
    ROLLING_BOOTSTRAP_REPLICATES,
    ROLLING_FOLD_COUNT,
    ROLLING_FOLD_SIZE,
    rolling_score_bootstrap_trace,
)
from prpg.model.stability import (
    MacroStabilitySummary,
    RollingStabilitySummary,
    align_state_labels,
    aligned_state_jaccards,
    canonical_rolling_folds,
    map_monthly_states_to_daily,
    summarize_macro_bootstrap_stability,
    summarize_rolling_stability,
)
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    ModelFitScope,
    Stage,
    calibration_rng_key,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
ArtifactRole: TypeAlias = Literal["design", "production"]
FitFunction: TypeAlias = Callable[
    [HMMFeatureMatrix, int, Sequence[int]], GaussianHMMFit
]

VERSION_1_CANDIDATES = (2, 3, 4, 5)
MACRO_STABILITY_ATTEMPTS = 100
_MACRO_STABILITY_EXECUTION_CAPABILITY = object()
_MACRO_BOOTSTRAP_MATERIALIZATION_CAPABILITY = object()
_ROLLING_BOOTSTRAP_MATERIALIZATION_CAPABILITY = object()
_PRELIMINARY_CALIBRATION_CAPABILITY = object()
MACRO_BOOTSTRAP_MATERIALIZATION_CONTRACT = (
    "registered-common-macro-bootstrap-materialization-v1"
)
ROLLING_BOOTSTRAP_MATERIALIZATION_CONTRACT = (
    "registered-rolling-score-bootstrap-materialization-v1"
)


@dataclass(frozen=True, slots=True)
class RegisteredRollingBootstrapMaterialization:
    """One producer-sealed common rolling-score bootstrap materialization."""

    contract_version: str
    master_seed: int
    scientific_version: int
    artifact: CalibrationArtifact
    indices: IntArray = field(repr=False, compare=False)
    materialization_fingerprint: str
    rng_execution_fingerprint: str
    _producer_seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


_MINTED_ROLLING_BOOTSTRAP_MATERIALIZATIONS: dict[
    int,
    tuple[str, int, int, CalibrationArtifact, str, str],
] = {}


@dataclass(frozen=True, slots=True)
class PreliminaryCalibrations:
    """Return and macro PPW starts computed before candidate selection."""

    role: ArtifactRole
    monthly: FrequencyCalibration
    daily: FrequencyCalibration
    macro: MacroCalibration


PreliminaryFailureStage: TypeAlias = Literal["monthly", "daily", "macro"]
PreliminaryFailureCode: TypeAlias = Literal[
    "insufficient_support",
    "numerical_failure",
    "hard_cap_exceeded",
]
REGISTERED_PRELIMINARY_CALIBRATION_CONTRACT = (
    "registered-preliminary-calibration-outcome-v1"
)


@dataclass(frozen=True, slots=True, init=False)
class RegisteredPreliminaryCalibrationOutcome:
    """Producer-sealed global preliminary success or scientific failure."""

    contract_version: str
    role: ArtifactRole
    passed: bool
    calibrations: PreliminaryCalibrations | None
    failure_stage: PreliminaryFailureStage | None
    failure_code: PreliminaryFailureCode | None
    failure_type: str | None
    failure_message: str | None
    failure_details: tuple[tuple[str, object], ...]
    source_fingerprint: str
    result_fingerprint: str | None
    outcome_fingerprint: str
    _source_slice: CalibrationSlice = field(repr=False, compare=False)
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredPreliminaryCalibrationOutcome is created only by its executor"
        )


@dataclass(frozen=True, slots=True)
class RegisteredPreliminaryCalibrationIdentity:
    """Reverified source and result identity for a preliminary outcome."""

    role: ArtifactRole
    passed: bool
    failure_stage: PreliminaryFailureStage | None
    failure_code: PreliminaryFailureCode | None
    failure_type: str | None
    failure_message: str | None
    failure_details: tuple[tuple[str, object], ...]
    source_fingerprint: str
    result_fingerprint: str | None
    outcome_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreliminaryMintRecord:
    value: RegisteredPreliminaryCalibrationOutcome
    source_slice: CalibrationSlice
    calibrations: PreliminaryCalibrations | None
    source_fingerprint: str
    result_fingerprint: str | None
    outcome_fingerprint: str


@dataclass(frozen=True, slots=True)
class _DerivedPreliminaryOutcome:
    calibrations: PreliminaryCalibrations | None
    failure_stage: PreliminaryFailureStage | None
    failure_code: PreliminaryFailureCode | None
    failure_type: str | None
    failure_message: str | None
    failure_details: tuple[tuple[str, object], ...]


_MINTED_PRELIMINARY_CALIBRATIONS: dict[int, _PreliminaryMintRecord] = {}


@dataclass(frozen=True, slots=True)
class CandidateFitAttempt:
    """One K attempt, preserving either a fit or a bounded failure record."""

    n_states: int
    fit: GaussianHMMFit | None
    failure_type: str | None
    failure_message: str | None
    failure_details: Mapping[str, object] | None

    @property
    def valid(self) -> bool:
        return self.fit is not None


@dataclass(frozen=True, slots=True)
class CandidateFitSet:
    """All ordinary candidate attempts in exact K order."""

    scope: ModelFitScope
    replicate: int
    attempts: tuple[CandidateFitAttempt, ...]
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _features: HMMFeatureMatrix | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _master_seed: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _scientific_version: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _feature_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _fit_set_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _restart_seed_fingerprints: tuple[str, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _rng_execution_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def valid_fits(self) -> tuple[GaussianHMMFit, ...]:
        return tuple(
            attempt.fit for attempt in self.attempts if attempt.fit is not None
        )

    @property
    def passed(self) -> bool:
        return bool(self.valid_fits)


_REGISTERED_CANDIDATE_FIT_SET_CAPABILITY = object()
_MINTED_CANDIDATE_FIT_SETS: dict[int, CandidateFitSet] = {}


@dataclass(frozen=True, slots=True)
class RegisteredCandidateFitSetIdentity:
    """Reverified design-candidate fit family and its exact restart sources."""

    feature_fingerprint: str
    fit_set_fingerprint: str
    master_seed: int
    scientific_version: int
    scope: ModelFitScope
    replicate: int
    restart_seed_fingerprints: tuple[str, ...]
    rng_namespace: str
    rng_execution_fingerprint: str


@dataclass(frozen=True, slots=True)
class DecodedReturnPools:
    """Monthly and daily observed labels plus gap-aware path diagnostics."""

    monthly_states: IntArray
    daily_states: IntArray
    monthly_diagnostics: object
    daily_diagnostics: object


@dataclass(frozen=True, slots=True)
class RollingFitAttempt:
    """One expanding-fold refit, score vector, and reference alignment."""

    fold: int
    training_rows: int
    validation_rows: int
    validation_lengths: tuple[int, ...]
    fit: GaussianHMMFit | None
    scores: tuple[float | None, ...]
    adjusted_rand_index: float
    jaccards: FloatArray
    failure_type: str | None
    failure_message: str | None

    @property
    def successful(self) -> bool:
        return self.fit is not None


@dataclass(frozen=True, slots=True)
class RollingCandidateEvidence:
    """Six exact folds and their distinct rolling-stability decision."""

    n_states: int
    role: ArtifactRole
    attempts: tuple[RollingFitAttempt, ...]
    stability: RollingStabilitySummary
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _parent_model: GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _macro_features: FloatArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _month_ordinals: IntArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _continuation: BoolArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _master_seed: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _scientific_version: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _parent_model_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _macro_input_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _restart_seed_fingerprints: tuple[str, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _rng_execution_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def rolling_scores(self) -> tuple[float | None, ...]:
        return tuple(score for attempt in self.attempts for score in attempt.scores)


_REGISTERED_ROLLING_CANDIDATE_CAPABILITY = object()
_MINTED_ROLLING_CANDIDATE_EVIDENCE: dict[int, RollingCandidateEvidence] = {}


@dataclass(frozen=True, slots=True)
class RegisteredRollingCandidateIdentity:
    """Reverified six-fold candidate execution and exact model-fit streams."""

    role: ArtifactRole
    n_states: int
    parent_model_fingerprint: str
    macro_input_fingerprint: str
    master_seed: int
    scientific_version: int
    restart_seed_fingerprints: tuple[str, ...]
    rng_namespace: str
    rng_execution_fingerprint: str
    evidence_fingerprint: str


@dataclass(frozen=True, slots=True)
class MacroBootstrapAttempt:
    """One registered macro-history bootstrap/refit/alignment attempt."""

    attempt: int
    source_indices: IntArray
    fit: GaussianHMMFit | None
    adjusted_rand_index: float
    jaccards: FloatArray
    failure_type: str | None
    failure_message: str | None
    candidate_to_reference: IntArray | None = None

    @property
    def successful(self) -> bool:
        return self.fit is not None


@dataclass(frozen=True, slots=True)
class RegisteredMacroBootstrapMaterialization:
    """One common purpose-0 source-index matrix reused across candidate K."""

    contract_version: str
    role: ArtifactRole
    macro_block_length: int
    rows: int
    master_seed: int
    scientific_version: int
    indices: IntArray = field(repr=False, compare=False)
    source_input_fingerprint: str
    rng_execution_fingerprint: str
    materialization_fingerprint: str
    _macro_features: FloatArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _month_ordinals: IntArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _continuation: BoolArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _producer_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


_MINTED_MACRO_BOOTSTRAP_MATERIALIZATIONS: dict[
    int,
    RegisteredMacroBootstrapMaterialization,
] = {}


@dataclass(frozen=True, slots=True)
class RegisteredMacroBootstrapMaterializationIdentity:
    """Reverified source/RNG/content identity of a common macro draw set."""

    role: ArtifactRole
    macro_block_length: int
    rows: int
    master_seed: int
    scientific_version: int
    source_input_fingerprint: str
    rng_namespace: str
    rng_execution_fingerprint: str
    materialization_fingerprint: str


@dataclass(frozen=True, slots=True)
class MacroStabilityEvidence:
    """Exactly 100 macro bootstrap attempts and their binding summary."""

    n_states: int
    role: ArtifactRole
    macro_block_length: int
    attempts: tuple[MacroBootstrapAttempt, ...]
    summary: MacroStabilitySummary
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _parent_model: GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _macro_features: FloatArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _month_ordinals: IntArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _continuation: BoolArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_materialization: RegisteredMacroBootstrapMaterialization | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _parent_model_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _macro_input_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _rng_execution_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _master_seed: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _scientific_version: int | None = field(
        default=None, init=False, repr=False, compare=False
    )


_MINTED_MACRO_STABILITY_EVIDENCE: dict[int, MacroStabilityEvidence] = {}


@dataclass(frozen=True, slots=True)
class MacroStabilityExecutionIdentity:
    """Registered parent/input/RNG identity for one completed macro family."""

    role: ArtifactRole
    n_states: int
    parent_model_fingerprint: str
    macro_input_fingerprint: str
    master_seed: int
    scientific_version: int
    rng_namespace: str
    rng_execution_fingerprint: str
    materialization_fingerprint: str
    evidence_fingerprint: str


def macro_stability_rng_namespace(role: ArtifactRole) -> str:
    """Return the code-owned G3 RNG slot for one artifact role."""

    if role == "design":
        return "design_macro_bootstrap_streams"
    if role == "production":
        return "production_macro_bootstrap_streams"
    raise ModelError("macro stability role must be design or production")


def macro_stability_input_fingerprint(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
) -> str:
    """Fingerprint the exact macro matrix, calendar, and gap mask consumed."""

    values, index, mask = _macro_frame_inputs(
        macro_features,
        month_ordinals,
        continuation,
    )
    if role not in {"design", "production"}:
        raise ModelError("macro stability role must be design or production")
    return scientific_materialization_fingerprint(
        schema_id="macro_stability_input",
        metadata={"role": role},
        arrays={
            "continuation": mask,
            "macro_features": values,
            "month_ordinals": np.asarray(index.asi8, dtype=np.int64),
        },
    )


def macro_stability_rng_fingerprint(
    *,
    role: ArtifactRole,
    n_states: int,
    rows: int,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
) -> str:
    """Fingerprint all 100 registered bootstrap and HMM-restart coordinates."""

    if role not in {"design", "production"}:
        raise ModelError("macro stability role must be design or production")
    if n_states not in VERSION_1_CANDIDATES:
        raise ModelError("macro stability RNG identity has an unregistered K")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ModelError("macro stability RNG row count must be positive")
    artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    scope = (
        ModelFitScope.DESIGN_MACRO_STABILITY
        if role == "design"
        else ModelFitScope.PRODUCTION_MACRO_STABILITY
    )
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace=macro_stability_rng_namespace(role),
        contract={
            "artifact": int(artifact),
            "attempts": MACRO_STABILITY_ATTEMPTS,
            "domain": int(Domain.BLOCK_ALPHA_CALIBRATION),
            "family": int(Family.NOT_APPLICABLE),
            "macro_block_length": macro_block_length,
            "model_fit_scope": int(scope),
            "n_states": n_states,
            "purpose": int(CalibrationPurpose.MACRO_BOOTSTRAP),
            "rows": rows,
            "stage": int(Stage.REFERENCE_RESAMPLING),
        },
    )


def registered_macro_stability_execution_identity(
    value: MacroStabilityEvidence,
) -> MacroStabilityExecutionIdentity:
    """Revalidate and return one closed macro-stability execution identity."""

    if (
        not isinstance(value, MacroStabilityEvidence)
        or value._execution_seal is not _MACRO_STABILITY_EXECUTION_CAPABILITY
        or _MINTED_MACRO_STABILITY_EVIDENCE.get(id(value)) is not value
    ):
        raise ModelError("macro stability evidence lacks registered execution")
    parent = value._parent_model
    features = value._macro_features
    ordinals = value._month_ordinals
    continuation = value._continuation
    materialization = value._source_materialization
    master_seed = value._master_seed
    scientific_version = value._scientific_version
    if (
        parent is None
        or features is None
        or ordinals is None
        or continuation is None
        or materialization is None
        or master_seed is None
        or scientific_version is None
        or value._parent_model_fingerprint is None
        or value._macro_input_fingerprint is None
        or value._rng_execution_fingerprint is None
        or value._evidence_fingerprint is None
    ):
        raise ModelError("macro stability execution identity is incomplete")
    source_identity = registered_macro_bootstrap_materialization_identity(
        materialization
    )
    if (
        value.role not in {"design", "production"}
        or value.n_states not in VERSION_1_CANDIDATES
        or parent.n_states != value.n_states
        or len(value.attempts) != MACRO_STABILITY_ATTEMPTS
        or tuple(item.attempt for item in value.attempts)
        != tuple(range(MACRO_STABILITY_ATTEMPTS))
    ):
        raise ModelError("macro stability registered geometry is invalid")
    successful = np.zeros(MACRO_STABILITY_ATTEMPTS, dtype=np.bool_)
    adjusted_rand_indices = np.zeros(MACRO_STABILITY_ATTEMPTS, dtype=np.float64)
    jaccards = np.zeros(
        (MACRO_STABILITY_ATTEMPTS, value.n_states),
        dtype=np.float64,
    )
    for attempt in value.attempts:
        indices = np.asarray(attempt.source_indices)
        overlap = np.asarray(attempt.jaccards)
        if (
            indices.shape != (len(features),)
            or indices.dtype.kind not in "iu"
            or bool(((indices < 0) | (indices >= len(features))).any())
            or overlap.shape != (value.n_states,)
            or not bool(np.isfinite(overlap).all())
            or not math.isfinite(float(attempt.adjusted_rand_index))
            or (attempt.fit is not None and attempt.fit.n_states != value.n_states)
            or not np.array_equal(
                indices,
                np.asarray(materialization.indices[attempt.attempt]),
            )
        ):
            raise ModelError("macro stability registered attempt is inconsistent")
        if attempt.fit is None:
            if (
                attempt.failure_type is None
                or attempt.failure_message is None
                or attempt.candidate_to_reference is not None
                or attempt.adjusted_rand_index != 0.0
                or bool((overlap != 0.0).any())
            ):
                raise ModelError("failed macro attempt has invalid failure evidence")
        else:
            alignment = np.asarray(attempt.candidate_to_reference)
            if (
                attempt.failure_type is not None
                or attempt.failure_message is not None
                or alignment.shape != (value.n_states,)
                or not np.array_equal(
                    np.sort(alignment.astype(np.int64, copy=False)),
                    np.arange(value.n_states, dtype=np.int64),
                )
            ):
                raise ModelError("successful macro attempt has invalid alignment")
            successful[attempt.attempt] = True
        adjusted_rand_indices[attempt.attempt] = attempt.adjusted_rand_index
        jaccards[attempt.attempt] = overlap
    recomputed_summary = summarize_macro_bootstrap_stability(
        successful,
        adjusted_rand_indices,
        jaccards,
    )
    if not _macro_stability_summaries_equal(value.summary, recomputed_summary):
        raise ModelError("macro stability summary is inconsistent with attempts")
    parent_fingerprint = gaussian_hmm_fit_fingerprint(parent)
    input_fingerprint = macro_stability_input_fingerprint(
        features,
        ordinals,
        continuation,
        role=value.role,
    )
    rng_fingerprint = macro_stability_rng_fingerprint(
        role=value.role,
        n_states=value.n_states,
        rows=len(features),
        macro_block_length=value.macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    if (
        parent_fingerprint != value._parent_model_fingerprint
        or input_fingerprint != value._macro_input_fingerprint
        or rng_fingerprint != value._rng_execution_fingerprint
        or source_identity.role != value.role
        or source_identity.macro_block_length != value.macro_block_length
        or source_identity.rows != len(features)
        or source_identity.master_seed != master_seed
        or source_identity.scientific_version != scientific_version
        or source_identity.source_input_fingerprint != input_fingerprint
    ):
        raise ModelError("macro stability execution sources changed after execution")
    _require_immutable_public_arrays(value)
    # Local import avoids a module cycle: calibration_identity intentionally
    # describes this public result type.
    from prpg.model.calibration_identity import (
        macro_stability_evidence_fingerprint,
    )

    evidence_fingerprint = macro_stability_evidence_fingerprint(value)
    if evidence_fingerprint != value._evidence_fingerprint:
        raise ModelError("macro stability evidence changed after execution")
    return MacroStabilityExecutionIdentity(
        role=value.role,
        n_states=value.n_states,
        parent_model_fingerprint=parent_fingerprint,
        macro_input_fingerprint=input_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_namespace=macro_stability_rng_namespace(value.role),
        rng_execution_fingerprint=rng_fingerprint,
        materialization_fingerprint=(source_identity.materialization_fingerprint),
        evidence_fingerprint=evidence_fingerprint,
    )


def is_registered_macro_stability_evidence(
    value: object,
) -> TypeGuard[MacroStabilityEvidence]:
    """Return whether evidence is an unchanged closed-executor result."""

    if not isinstance(value, MacroStabilityEvidence):
        return False
    try:
        registered_macro_stability_execution_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class SealedHoldoutEvidence:
    """Selected-HMM scores and the two immutable veto benchmarks."""

    sequence_lengths: tuple[int, ...]
    hmm_log_scores: FloatArray
    benchmark_scores: HoldoutBenchmarkScores
    decision: HoldoutVetoDecision


def prepare_hmm_feature_matrix(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    design_rows: int | None = None,
) -> HMMFeatureMatrix:
    """Fit the exact design-only or production-full scaler and retain gaps."""

    values = np.asarray(macro_features, dtype=np.float64)
    ordinals = np.asarray(month_ordinals)
    mask = np.asarray(continuation)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ModelError("macro features must be a non-empty four-column matrix")
    if not np.isfinite(values).all():
        raise ModelError("macro features contain non-finite values")
    if ordinals.shape != (len(values),) or ordinals.dtype.kind not in {"i", "u"}:
        raise ModelError("month ordinals must be an integer vector matching rows")
    if mask.shape != (len(values),) or mask.dtype.kind != "b" or bool(mask[0]):
        raise ModelError("macro continuation must be a false-at-start row mask")
    checked_mask: BoolArray = mask.astype(np.bool_, copy=False)
    if role not in {"design", "production"}:
        raise ModelError("HMM feature role must be design or production")
    index = pd.PeriodIndex.from_ordinals(ordinals.astype(np.int64), freq="M")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ModelError("macro month ordinals must be unique and increasing")
    frame = pd.DataFrame(values, index=index, columns=MACRO_COLUMNS)
    if role == "design":
        if (
            isinstance(design_rows, bool)
            or not isinstance(design_rows, int)
            or not 1 <= design_rows < len(frame)
        ):
            raise ModelError("design_rows must identify a proper positive prefix")
        fit_index = frame.index[:design_rows]
        standardized = standardize_features(frame, training_index=fit_index)
        return _feature_matrix_from_standardized(
            standardized,
            continuation=checked_mask,
            role=role,
            fit_index=fit_index,
        )
    if design_rows is not None:
        raise ModelError("production HMM feature preparation forbids design_rows")
    standardized = standardize_features(frame)
    return _feature_matrix_from_standardized(
        standardized,
        continuation=checked_mask,
        role=role,
        fit_index=None,
    )


def preliminary_block_calibrations(
    data: CalibrationSlice,
    *,
    role: ArtifactRole,
) -> PreliminaryCalibrations:
    """Compute the exact independent monthly, daily, and macro PPW starts."""

    _validate_preliminary_source(data, role)
    monthly = estimate_frequency_start(
        data.monthly_returns,
        frequency="monthly",
        continuation=data.monthly_continuation,
    )
    daily = estimate_frequency_start(
        data.daily_returns,
        frequency="daily",
        continuation=data.daily_continuation,
    )
    macro = estimate_macro_start(
        data.macro_features,
        continuation=data.monthly_continuation,
    )
    return PreliminaryCalibrations(role, monthly, daily, macro)


def execute_registered_preliminary_calibrations(
    data: CalibrationSlice,
    *,
    role: ArtifactRole,
) -> RegisteredPreliminaryCalibrationOutcome:
    """Mint the exact global preliminary success or scientific nonattainment.

    Only the three closed PPW numerical/support exception types are reduced to
    a scientific outcome.  Structural ``ModelError`` values and every other
    exception propagate unchanged.
    """

    _validate_preliminary_source(data, role)
    source_fingerprint = _preliminary_source_fingerprint(data, role)
    derived = _derive_preliminary_outcome(data, role)
    result_fingerprint = (
        None
        if derived.calibrations is None
        else preliminary_calibrations_fingerprint(derived.calibrations)
    )
    outcome_fingerprint = _preliminary_outcome_fingerprint(
        role=role,
        source_fingerprint=source_fingerprint,
        result_fingerprint=result_fingerprint,
        derived=derived,
    )
    value = object.__new__(RegisteredPreliminaryCalibrationOutcome)
    for name, item in (
        ("contract_version", REGISTERED_PRELIMINARY_CALIBRATION_CONTRACT),
        ("role", role),
        ("passed", derived.calibrations is not None),
        ("calibrations", derived.calibrations),
        ("failure_stage", derived.failure_stage),
        ("failure_code", derived.failure_code),
        ("failure_type", derived.failure_type),
        ("failure_message", derived.failure_message),
        ("failure_details", derived.failure_details),
        ("source_fingerprint", source_fingerprint),
        ("result_fingerprint", result_fingerprint),
        ("outcome_fingerprint", outcome_fingerprint),
        ("_source_slice", data),
        ("_execution_seal", _PRELIMINARY_CALIBRATION_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_PRELIMINARY_CALIBRATIONS[id(value)] = _PreliminaryMintRecord(
        value=value,
        source_slice=data,
        calibrations=derived.calibrations,
        source_fingerprint=source_fingerprint,
        result_fingerprint=result_fingerprint,
        outcome_fingerprint=outcome_fingerprint,
    )
    try:
        registered_preliminary_calibration_identity(value)
    except Exception:
        _MINTED_PRELIMINARY_CALIBRATIONS.pop(id(value), None)
        raise
    return value


def registered_preliminary_calibration_identity(
    value: RegisteredPreliminaryCalibrationOutcome,
) -> RegisteredPreliminaryCalibrationIdentity:
    """Recompute the exact source/result graph for one minted outcome."""

    if not isinstance(value, RegisteredPreliminaryCalibrationOutcome):
        raise ModelError("preliminary outcome has the wrong type")
    record = _MINTED_PRELIMINARY_CALIBRATIONS.get(id(value))
    if (
        value._execution_seal is not _PRELIMINARY_CALIBRATION_CAPABILITY
        or record is None
        or record.value is not value
        or value._source_slice is not record.source_slice
        or value.calibrations is not record.calibrations
    ):
        raise ModelError("preliminary outcome lacks exact producer authority")
    _validate_preliminary_source(record.source_slice, value.role)
    source_fingerprint = _preliminary_source_fingerprint(
        record.source_slice,
        value.role,
    )
    derived = _derive_preliminary_outcome(record.source_slice, value.role)
    result_fingerprint = (
        None
        if derived.calibrations is None
        else preliminary_calibrations_fingerprint(derived.calibrations)
    )
    outcome_fingerprint = _preliminary_outcome_fingerprint(
        role=value.role,
        source_fingerprint=source_fingerprint,
        result_fingerprint=result_fingerprint,
        derived=derived,
    )
    if (
        value.contract_version != REGISTERED_PRELIMINARY_CALIBRATION_CONTRACT
        or value.passed is not (derived.calibrations is not None)
        or value.failure_stage != derived.failure_stage
        or value.failure_code != derived.failure_code
        or value.failure_type != derived.failure_type
        or value.failure_message != derived.failure_message
        or value.failure_details != derived.failure_details
        or value.source_fingerprint != source_fingerprint
        or value.result_fingerprint != result_fingerprint
        or value.outcome_fingerprint != outcome_fingerprint
        or record.source_fingerprint != source_fingerprint
        or record.result_fingerprint != result_fingerprint
        or record.outcome_fingerprint != outcome_fingerprint
        or (
            value.calibrations is not None
            and preliminary_calibrations_fingerprint(value.calibrations)
            != result_fingerprint
        )
    ):
        raise ModelError("preliminary outcome source or result changed")
    return RegisteredPreliminaryCalibrationIdentity(
        role=value.role,
        passed=value.passed,
        failure_stage=value.failure_stage,
        failure_code=value.failure_code,
        failure_type=value.failure_type,
        failure_message=value.failure_message,
        failure_details=value.failure_details,
        source_fingerprint=source_fingerprint,
        result_fingerprint=result_fingerprint,
        outcome_fingerprint=outcome_fingerprint,
    )


def is_registered_preliminary_calibration_outcome(
    value: object,
) -> TypeGuard[RegisteredPreliminaryCalibrationOutcome]:
    if not isinstance(value, RegisteredPreliminaryCalibrationOutcome):
        return False
    try:
        registered_preliminary_calibration_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_preliminary_calibrations(
    value: RegisteredPreliminaryCalibrationOutcome,
) -> PreliminaryCalibrations:
    """Return exact successful starts; failed outcomes cannot create a suite."""

    identity = registered_preliminary_calibration_identity(value)
    if not identity.passed or value.calibrations is None:
        raise ModelError("failed preliminary outcome has no calibrations")
    return value.calibrations


def preliminary_calibrations_fingerprint(value: PreliminaryCalibrations) -> str:
    """Fingerprint all three registered preliminary decisions."""

    if not isinstance(value, PreliminaryCalibrations) or value.role not in {
        "design",
        "production",
    }:
        raise ModelError("preliminary calibration is invalid")
    monthly = value.monthly.combined
    daily = value.daily.combined
    macro = value.macro.combined
    return scientific_materialization_fingerprint(
        schema_id="design_candidate_preliminary_calibrations_v1",
        metadata={
            "role": value.role,
            "monthly": {
                "policy_version": value.monthly.policy_version,
                "frequency": monthly.frequency,
                "raw_ceiling_length": monthly.raw_ceiling_length,
                "starting_length": monthly.starting_length,
                "lower_bound": monthly.lower_bound,
                "upper_bound": monthly.upper_bound,
                "reasons": list(monthly.reasons),
            },
            "daily": {
                "policy_version": value.daily.policy_version,
                "frequency": daily.frequency,
                "raw_ceiling_length": daily.raw_ceiling_length,
                "starting_length": daily.starting_length,
                "lower_bound": daily.lower_bound,
                "upper_bound": daily.upper_bound,
                "reasons": list(daily.reasons),
            },
            "macro": {
                "policy_version": value.macro.policy_version,
                "raw_ceiling_length": macro.raw_ceiling_length,
                "starting_length": macro.starting_length,
                "lower_bound": macro.lower_bound,
                "absolute_upper_bound": macro.absolute_upper_bound,
                "sample_support_upper_bound": macro.sample_support_upper_bound,
            },
        },
        arrays={
            "starts": np.asarray(
                [
                    monthly.starting_length,
                    daily.starting_length,
                    macro.starting_length,
                ],
                dtype=np.int64,
            )
        },
    )


def _validate_preliminary_source(
    data: CalibrationSlice,
    role: ArtifactRole,
) -> None:
    if not isinstance(data, CalibrationSlice):
        raise ModelError("preliminary calibration requires CalibrationSlice")
    if role not in {"design", "production"}:
        raise ModelError("preliminary calibration role must be design or production")
    expected_slice = "design" if role == "design" else "full"
    if data.role != expected_slice:
        raise ModelError(
            "calibration slice role does not match requested artifact",
            details={"expected": expected_slice, "actual": data.role},
        )


def _derive_preliminary_outcome(
    data: CalibrationSlice,
    role: ArtifactRole,
) -> _DerivedPreliminaryOutcome:
    _validate_preliminary_source(data, role)
    stages: tuple[
        tuple[
            PreliminaryFailureStage,
            Callable[[], FrequencyCalibration | MacroCalibration],
        ],
        ...,
    ] = (
        (
            "monthly",
            lambda: estimate_frequency_start(
                data.monthly_returns,
                frequency="monthly",
                continuation=data.monthly_continuation,
            ),
        ),
        (
            "daily",
            lambda: estimate_frequency_start(
                data.daily_returns,
                frequency="daily",
                continuation=data.daily_continuation,
            ),
        ),
        (
            "macro",
            lambda: estimate_macro_start(
                data.macro_features,
                continuation=data.monthly_continuation,
            ),
        ),
    )
    results: list[FrequencyCalibration | MacroCalibration] = []
    for stage, execute in stages:
        try:
            results.append(execute())
        except (
            PreliminaryPPWInsufficientSupport,
            PreliminaryPPWNumericalFailure,
            PreliminaryBlockLengthCapFailure,
        ) as error:
            return _DerivedPreliminaryOutcome(
                calibrations=None,
                failure_stage=stage,
                failure_code=_preliminary_failure_code(error),
                failure_type=type(error).__name__,
                failure_message=error.message,
                failure_details=_frozen_scientific_details(error.details),
            )
    monthly, daily, macro = results
    assert isinstance(monthly, FrequencyCalibration)
    assert isinstance(daily, FrequencyCalibration)
    assert isinstance(macro, MacroCalibration)
    return _DerivedPreliminaryOutcome(
        calibrations=PreliminaryCalibrations(role, monthly, daily, macro),
        failure_stage=None,
        failure_code=None,
        failure_type=None,
        failure_message=None,
        failure_details=(),
    )


def _preliminary_failure_code(
    error: ModelError,
) -> PreliminaryFailureCode:
    if isinstance(error, PreliminaryPPWInsufficientSupport):
        return "insufficient_support"
    if isinstance(error, PreliminaryPPWNumericalFailure):
        return "numerical_failure"
    if isinstance(error, PreliminaryBlockLengthCapFailure):
        return "hard_cap_exceeded"
    raise AssertionError("unsupported preliminary scientific exception")


def _preliminary_source_fingerprint(
    data: CalibrationSlice,
    role: ArtifactRole,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_preliminary_calibration_source_v1",
        metadata={"role": role, "slice_role": data.role},
        arrays={
            "daily_continuation": data.daily_continuation,
            "daily_returns": data.daily_returns,
            "daily_session_ns": data.daily_session_ns,
            "macro_features": data.macro_features,
            "monthly_continuation": data.monthly_continuation,
            "monthly_period_ordinals": data.monthly_period_ordinals,
            "monthly_returns": data.monthly_returns,
        },
    )


def _preliminary_outcome_fingerprint(
    *,
    role: ArtifactRole,
    source_fingerprint: str,
    result_fingerprint: str | None,
    derived: _DerivedPreliminaryOutcome,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_preliminary_calibration_outcome_v1",
        metadata={
            "contract_version": REGISTERED_PRELIMINARY_CALIBRATION_CONTRACT,
            "role": role,
            "passed": derived.calibrations is not None,
            "source_fingerprint": source_fingerprint,
            "result_fingerprint": result_fingerprint,
            "failure_stage": derived.failure_stage,
            "failure_code": derived.failure_code,
            "failure_type": derived.failure_type,
            "failure_message": derived.failure_message,
            "failure_details": derived.failure_details,
        },
        arrays={"outcome": np.asarray([derived.calibrations is not None])},
    )


def execute_registered_hmm_candidate_set(
    features: HMMFeatureMatrix,
    *,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
) -> CandidateFitSet:
    """Execute the design K=2..5 family without an injectable fitter."""

    worker_count = _validated_hmm_workers(workers)
    result = fit_hmm_candidate_set(
        features,
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        fit_function=_registered_hmm_fit_function(
            scientific_version=scientific_version,
            workers=worker_count,
        ),
    )
    result = cast(CandidateFitSet, _immutable_public_tree(result))
    feature_fingerprint = hmm_feature_matrix_fingerprint(features)
    from prpg.model.calibration_identity import candidate_fit_set_fingerprint

    fit_set_fingerprint = candidate_fit_set_fingerprint(result)
    restart_fingerprints, rng_fingerprint = _candidate_fit_restart_identity(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=result.scope,
        replicate=result.replicate,
    )
    object.__setattr__(result, "_features", features)
    object.__setattr__(result, "_master_seed", master_seed)
    object.__setattr__(result, "_scientific_version", scientific_version)
    object.__setattr__(result, "_feature_fingerprint", feature_fingerprint)
    object.__setattr__(result, "_fit_set_fingerprint", fit_set_fingerprint)
    object.__setattr__(
        result,
        "_restart_seed_fingerprints",
        restart_fingerprints,
    )
    object.__setattr__(result, "_rng_execution_fingerprint", rng_fingerprint)
    object.__setattr__(
        result,
        "_execution_seal",
        _REGISTERED_CANDIDATE_FIT_SET_CAPABILITY,
    )
    _MINTED_CANDIDATE_FIT_SETS[id(result)] = result
    registered_candidate_fit_set_identity(result)
    return result


def registered_candidate_fit_set_identity(
    value: CandidateFitSet,
) -> RegisteredCandidateFitSetIdentity:
    """Re-hash one executor-minted design-candidate fit family."""

    if (
        not isinstance(value, CandidateFitSet)
        or value._execution_seal is not _REGISTERED_CANDIDATE_FIT_SET_CAPABILITY
        or _MINTED_CANDIDATE_FIT_SETS.get(id(value)) is not value
    ):
        raise ModelError("candidate fit set lacks registered executor authority")
    features = value._features
    master_seed = value._master_seed
    scientific_version = value._scientific_version
    if (
        features is None
        or master_seed is None
        or scientific_version is None
        or value._feature_fingerprint is None
        or value._fit_set_fingerprint is None
        or value._rng_execution_fingerprint is None
        or len(value._restart_seed_fingerprints) != len(VERSION_1_CANDIDATES)
    ):
        raise ModelError("registered candidate fit identity is incomplete")
    if (
        value.scope is not ModelFitScope.DESIGN_MAIN
        or value.replicate != 0
        or features.scaler_scope != "design_training"
        or tuple(attempt.n_states for attempt in value.attempts) != VERSION_1_CANDIDATES
    ):
        raise ModelError("registered candidate fit geometry is invalid")
    for attempt in value.attempts:
        fit = attempt.fit
        if fit is None:
            if attempt.failure_type is None or attempt.failure_message is None:
                raise ModelError("registered failed candidate lacks failure evidence")
            continue
        if (
            any(
                item is not None
                for item in (
                    attempt.failure_type,
                    attempt.failure_message,
                    attempt.failure_details,
                )
            )
            or fit.n_states != attempt.n_states
            or fit.n_observations != len(features.values)
            or fit.n_features != len(features.feature_names)
            or fit.lengths != features.lengths
            or fit.feature_names != features.feature_names
            or fit.scaler_scope != features.scaler_scope
        ):
            raise ModelError("registered candidate fit output is inconsistent")
    feature_fingerprint = hmm_feature_matrix_fingerprint(features)
    from prpg.model.calibration_identity import candidate_fit_set_fingerprint

    fit_set_fingerprint = candidate_fit_set_fingerprint(value)
    restart_fingerprints, rng_fingerprint = _candidate_fit_restart_identity(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=value.scope,
        replicate=value.replicate,
    )
    _require_immutable_public_arrays(value)
    if (
        feature_fingerprint != value._feature_fingerprint
        or fit_set_fingerprint != value._fit_set_fingerprint
        or restart_fingerprints != value._restart_seed_fingerprints
        or rng_fingerprint != value._rng_execution_fingerprint
    ):
        raise ModelError("registered candidate fit sources or outputs changed")
    return RegisteredCandidateFitSetIdentity(
        feature_fingerprint=feature_fingerprint,
        fit_set_fingerprint=fit_set_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=value.scope,
        replicate=value.replicate,
        restart_seed_fingerprints=restart_fingerprints,
        rng_namespace="design_candidate_fit_restarts",
        rng_execution_fingerprint=rng_fingerprint,
    )


def is_registered_candidate_fit_set(
    value: object,
) -> TypeGuard[CandidateFitSet]:
    """Return whether a fit family is the unchanged registered instance."""

    if not isinstance(value, CandidateFitSet):
        return False
    try:
        registered_candidate_fit_set_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def _candidate_fit_restart_identity(
    *,
    master_seed: int,
    scientific_version: int,
    scope: ModelFitScope,
    replicate: int,
) -> tuple[tuple[str, ...], str]:
    if scope is not ModelFitScope.DESIGN_MAIN or replicate != 0:
        raise ModelError("registered candidate fit scope must be design-main/zero")
    fingerprints: list[str] = []
    for n_states in VERSION_1_CANDIDATES:
        seeds = np.asarray(
            deterministic_hmm_restart_seeds(
                master_seed=master_seed,
                scientific_version=scientific_version,
                scope=scope,
                replicate=replicate,
                n_states=n_states,
            ),
            dtype=np.uint32,
        )
        fingerprints.append(
            scientific_array_fingerprint(
                seeds,
                role=f"design_candidate_k{n_states}_restart_seeds",
            )
        )
    result = tuple(fingerprints)
    return result, rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="design_candidate_fit_restarts",
        contract={
            "candidate_order": list(VERSION_1_CANDIDATES),
            "replicate": replicate,
            "restart_seed_fingerprints": list(result),
            "scope": int(scope),
        },
    )


def fit_hmm_candidate_set(
    features: HMMFeatureMatrix,
    *,
    master_seed: int,
    scientific_version: int,
    scope: ModelFitScope,
    replicate: int,
    candidates: tuple[int, ...] = VERSION_1_CANDIDATES,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> CandidateFitSet:
    """Attempt every K independently and retain every scientific failure."""

    if candidates != VERSION_1_CANDIDATES:
        raise ModelError("version 1 candidate order must be exactly K=2,3,4,5")
    if not isinstance(scope, ModelFitScope):
        raise ModelError("candidate fit scope must use the closed ModelFitScope enum")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ModelError("candidate fit replicate must be a non-negative integer")
    attempts: list[CandidateFitAttempt] = []
    for n_states in candidates:
        try:
            seeds = deterministic_hmm_restart_seeds(
                master_seed=master_seed,
                scientific_version=scientific_version,
                scope=scope,
                replicate=replicate,
                n_states=n_states,
            )
            fit = fit_function(features, n_states, seeds)
            if fit.n_states != n_states:
                raise ModelError("HMM fit returned the wrong candidate state count")
            attempts.append(CandidateFitAttempt(n_states, fit, None, None, None))
        except NoNumericallyValidHMMRestarts as error:
            if error.n_states != n_states or tuple(
                item.seed for item in error.restart_diagnostics
            ) != tuple(seeds):
                raise ModelError(
                    "no-valid-restarts evidence does not match the K attempt"
                ) from error
            attempts.append(
                CandidateFitAttempt(
                    n_states,
                    None,
                    type(error).__name__,
                    _bounded(error.message),
                    _immutable_candidate_failure_details(error.details),
                )
            )
        except HMMCandidateInadmissibility as error:
            evidence = error.evidence
            if (
                evidence.n_states != n_states
                or len(evidence.restart_diagnostics) != HMM_RESTARTS
                or tuple(item.seed for item in evidence.restart_diagnostics)
                != tuple(seeds)
            ):
                raise ModelError(
                    "candidate inadmissibility evidence does not match the K attempt"
                ) from error
            attempts.append(
                CandidateFitAttempt(
                    n_states,
                    None,
                    type(error).__name__,
                    _bounded(error.message),
                    _immutable_candidate_failure_details(evidence.as_details()),
                )
            )
    return CandidateFitSet(scope, replicate, tuple(attempts))


def fit_production_selected_k(
    features: HMMFeatureMatrix,
    *,
    selected_n_states: int,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> GaussianHMMFit:
    """Fit production exactly once at the frozen design-selected K.

    This path deliberately has no candidate sequence and therefore cannot
    reselect or promote another state count.  Scope 4 / replicate 0 is the
    only registered production-main coordinate; rolling folds occupy scope 4
    replicates 1..6 separately.
    """

    if (
        isinstance(selected_n_states, bool)
        or not isinstance(selected_n_states, int)
        or selected_n_states not in VERSION_1_CANDIDATES
    ):
        raise ModelError("production selected K must be one of 2,3,4,5")
    if features.scaler_scope != "production_full":
        raise ModelError("production selected-K fit requires a production-full scaler")
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=selected_n_states,
    )
    fitted = fit_function(features, selected_n_states, seeds)
    if (
        fitted.n_states != selected_n_states
        or fitted.n_observations != len(features.values)
        or fitted.lengths != features.lengths
        or fitted.scaler_scope != "production_full"
    ):
        raise ModelError("production selected-K fit returned inconsistent identity")
    return fitted


def decoded_return_pools(
    fit: GaussianHMMFit,
    data: CalibrationSlice,
) -> DecodedReturnPools:
    """Map a fitted monthly Viterbi path onto return-ending daily sessions."""

    monthly = np.asarray(fit.decoded_states, dtype=np.int64)
    if monthly.shape != (len(data.monthly_period_ordinals),):
        raise ModelError("decoded monthly path does not match calibration slice")
    daily = map_monthly_states_to_daily(
        data.monthly_period_ordinals,
        monthly,
        data.daily_session_ns,
    )
    monthly_lengths = _lengths(data.monthly_continuation)
    daily_lengths = _lengths(data.daily_continuation)
    monthly_diagnostics = summarize_state_path(
        monthly,
        fit.n_states,
        lengths=monthly_lengths,
    )
    daily_diagnostics = summarize_state_path(
        daily,
        fit.n_states,
        lengths=daily_lengths,
    )
    monthly = monthly.copy()
    daily = daily.copy()
    monthly.setflags(write=False)
    daily.setflags(write=False)
    return DecodedReturnPools(
        monthly,
        daily,
        monthly_diagnostics,
        daily_diagnostics,
    )


def execute_registered_rolling_candidate(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    main_fit: GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
) -> RollingCandidateEvidence:
    """Execute and seal six rolling fits without a caller-supplied fitter."""

    worker_count = _validated_hmm_workers(workers)
    result = fit_rolling_candidate(
        macro_features,
        month_ordinals,
        continuation,
        role=role,
        main_fit=main_fit,
        master_seed=master_seed,
        scientific_version=scientific_version,
        fit_function=_registered_hmm_fit_function(
            scientific_version=scientific_version,
            workers=worker_count,
        ),
    )
    result = cast(RollingCandidateEvidence, _immutable_public_tree(result))
    values, index, mask = _macro_frame_inputs(
        macro_features,
        month_ordinals,
        continuation,
    )
    retained_features = cast(
        FloatArray,
        _immutable_bytes_array(
            values,
            dtype=np.dtype("<f8"),
            label="rolling candidate source features",
        ),
    )
    retained_ordinals = cast(
        IntArray,
        _immutable_bytes_array(
            index.asi8,
            dtype=np.dtype("<i8"),
            label="rolling candidate source ordinals",
        ),
    )
    retained_continuation = cast(
        BoolArray,
        _immutable_bytes_array(
            mask,
            dtype=np.dtype("?"),
            label="rolling candidate source continuation",
        ),
    )
    parent_fingerprint = gaussian_hmm_fit_fingerprint(main_fit)
    input_fingerprint = rolling_candidate_input_fingerprint(
        retained_features,
        retained_ordinals,
        retained_continuation,
        role=role,
    )
    restart_fingerprints, rng_fingerprint = _rolling_fit_restart_identity(
        role=role,
        n_states=main_fit.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    from prpg.model.calibration_identity import (
        rolling_candidate_evidence_fingerprint,
    )

    evidence_fingerprint = rolling_candidate_evidence_fingerprint(result)
    object.__setattr__(result, "_parent_model", main_fit)
    object.__setattr__(result, "_macro_features", retained_features)
    object.__setattr__(result, "_month_ordinals", retained_ordinals)
    object.__setattr__(result, "_continuation", retained_continuation)
    object.__setattr__(result, "_master_seed", master_seed)
    object.__setattr__(result, "_scientific_version", scientific_version)
    object.__setattr__(result, "_parent_model_fingerprint", parent_fingerprint)
    object.__setattr__(result, "_macro_input_fingerprint", input_fingerprint)
    object.__setattr__(
        result,
        "_restart_seed_fingerprints",
        restart_fingerprints,
    )
    object.__setattr__(result, "_rng_execution_fingerprint", rng_fingerprint)
    object.__setattr__(result, "_evidence_fingerprint", evidence_fingerprint)
    object.__setattr__(
        result,
        "_execution_seal",
        _REGISTERED_ROLLING_CANDIDATE_CAPABILITY,
    )
    _MINTED_ROLLING_CANDIDATE_EVIDENCE[id(result)] = result
    registered_rolling_candidate_identity(result)
    return result


def rolling_candidate_input_fingerprint(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
) -> str:
    """Fingerprint the exact macro history consumed by six rolling fits."""

    values, index, mask = _macro_frame_inputs(
        macro_features,
        month_ordinals,
        continuation,
    )
    if role not in {"design", "production"}:
        raise ModelError("rolling candidate role must be design or production")
    return scientific_materialization_fingerprint(
        schema_id="rolling_candidate_input",
        metadata={"role": role},
        arrays={
            "continuation": mask,
            "macro_features": values,
            "month_ordinals": np.asarray(index.asi8, dtype=np.int64),
        },
    )


def registered_rolling_candidate_identity(
    value: RollingCandidateEvidence,
) -> RegisteredRollingCandidateIdentity:
    """Recompute one producer-sealed six-fold execution and its summaries."""

    if (
        not isinstance(value, RollingCandidateEvidence)
        or value._execution_seal is not _REGISTERED_ROLLING_CANDIDATE_CAPABILITY
        or _MINTED_ROLLING_CANDIDATE_EVIDENCE.get(id(value)) is not value
    ):
        raise ModelError("rolling candidate evidence lacks registered authority")
    parent = value._parent_model
    features = value._macro_features
    ordinals = value._month_ordinals
    continuation = value._continuation
    master_seed = value._master_seed
    scientific_version = value._scientific_version
    if (
        parent is None
        or features is None
        or ordinals is None
        or continuation is None
        or master_seed is None
        or scientific_version is None
        or value._parent_model_fingerprint is None
        or value._macro_input_fingerprint is None
        or value._rng_execution_fingerprint is None
        or value._evidence_fingerprint is None
    ):
        raise ModelError("rolling candidate execution identity is incomplete")
    folds = canonical_rolling_folds(continuation)
    if (
        value.role not in {"design", "production"}
        or value.n_states not in VERSION_1_CANDIDATES
        or parent.n_states != value.n_states
        or len(value.attempts) != ROLLING_FOLD_COUNT
        or len(value._restart_seed_fingerprints) != ROLLING_FOLD_COUNT
    ):
        raise ModelError("rolling candidate registered geometry is invalid")
    successful = np.zeros(ROLLING_FOLD_COUNT, dtype=np.bool_)
    adjusted_rand_indices = np.zeros(ROLLING_FOLD_COUNT, dtype=np.float64)
    jaccards = np.zeros(
        (ROLLING_FOLD_COUNT, value.n_states),
        dtype=np.float64,
    )
    for attempt, fold in zip(value.attempts, folds, strict=True):
        overlap = np.asarray(attempt.jaccards)
        if (
            attempt.fold != fold.fold
            or attempt.training_rows != fold.training_rows
            or attempt.validation_rows != fold.validation_rows
            or attempt.validation_lengths != fold.validation_lengths
            or len(attempt.scores) != ROLLING_FOLD_SIZE
            or overlap.shape != (value.n_states,)
            or not bool(np.isfinite(overlap).all())
            or not math.isfinite(float(attempt.adjusted_rand_index))
        ):
            raise ModelError("rolling candidate attempt geometry is invalid")
        if attempt.fit is None:
            if (
                attempt.failure_type is None
                or attempt.failure_message is None
                or any(score is not None for score in attempt.scores)
                or attempt.adjusted_rand_index != 0.0
                or bool((overlap != 0.0).any())
            ):
                raise ModelError("failed rolling attempt is inconsistent")
        else:
            fit = attempt.fit
            if (
                attempt.failure_type is not None
                or attempt.failure_message is not None
                or fit.n_states != value.n_states
                or fit.n_observations != fold.training_rows
                or fit.scaler_scope != "design_training"
                or any(
                    score is None or not math.isfinite(float(score))
                    for score in attempt.scores
                )
            ):
                raise ModelError("successful rolling attempt is inconsistent")
            successful[attempt.fold] = True
        adjusted_rand_indices[attempt.fold] = attempt.adjusted_rand_index
        jaccards[attempt.fold] = overlap
    recomputed_summary = summarize_rolling_stability(
        successful,
        adjusted_rand_indices,
        jaccards,
    )
    if not _rolling_stability_summaries_equal(value.stability, recomputed_summary):
        raise ModelError("rolling candidate summary differs from its attempts")
    parent_fingerprint = gaussian_hmm_fit_fingerprint(parent)
    input_fingerprint = rolling_candidate_input_fingerprint(
        features,
        ordinals,
        continuation,
        role=value.role,
    )
    restart_fingerprints, rng_fingerprint = _rolling_fit_restart_identity(
        role=value.role,
        n_states=value.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    from prpg.model.calibration_identity import (
        rolling_candidate_evidence_fingerprint,
    )

    evidence_fingerprint = rolling_candidate_evidence_fingerprint(value)
    _require_immutable_public_arrays(value)
    if (
        parent_fingerprint != value._parent_model_fingerprint
        or input_fingerprint != value._macro_input_fingerprint
        or restart_fingerprints != value._restart_seed_fingerprints
        or rng_fingerprint != value._rng_execution_fingerprint
        or evidence_fingerprint != value._evidence_fingerprint
    ):
        raise ModelError("rolling candidate sources or outputs changed")
    return RegisteredRollingCandidateIdentity(
        role=value.role,
        n_states=value.n_states,
        parent_model_fingerprint=parent_fingerprint,
        macro_input_fingerprint=input_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        restart_seed_fingerprints=restart_fingerprints,
        rng_namespace=f"{value.role}_rolling_fit_restarts",
        rng_execution_fingerprint=rng_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
    )


def is_registered_rolling_candidate_evidence(
    value: object,
) -> TypeGuard[RollingCandidateEvidence]:
    """Return whether rolling evidence is unchanged and producer-minted."""

    if not isinstance(value, RollingCandidateEvidence):
        return False
    try:
        registered_rolling_candidate_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def _rolling_fit_restart_identity(
    *,
    role: ArtifactRole,
    n_states: int,
    master_seed: int,
    scientific_version: int,
) -> tuple[tuple[str, ...], str]:
    if role not in {"design", "production"}:
        raise ModelError("rolling candidate role must be design or production")
    if n_states not in VERSION_1_CANDIDATES:
        raise ModelError("rolling candidate state count is unregistered")
    scope = (
        ModelFitScope.DESIGN_ROLLING_FOLD
        if role == "design"
        else ModelFitScope.PRODUCTION_MAIN_AND_FOLDS
    )
    fingerprints: list[str] = []
    for fold in range(ROLLING_FOLD_COUNT):
        replicate = fold if role == "design" else fold + 1
        seeds = np.asarray(
            deterministic_hmm_restart_seeds(
                master_seed=master_seed,
                scientific_version=scientific_version,
                scope=scope,
                replicate=replicate,
                n_states=n_states,
            ),
            dtype=np.uint32,
        )
        fingerprints.append(
            scientific_array_fingerprint(
                seeds,
                role=f"{role}_rolling_k{n_states}_fold{fold}_restart_seeds",
            )
        )
    result = tuple(fingerprints)
    return result, rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace=f"{role}_rolling_fit_restarts",
        contract={
            "folds": ROLLING_FOLD_COUNT,
            "n_states": n_states,
            "restart_seed_fingerprints": list(result),
            "role": role,
            "scope": int(scope),
        },
    )


def fit_rolling_candidate(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    main_fit: GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> RollingCandidateEvidence:
    """Run six expanding fits, 12 scores each, and full-reference alignments."""

    values, index, mask = _macro_frame_inputs(
        macro_features, month_ordinals, continuation
    )
    if role not in {"design", "production"}:
        raise ModelError("rolling artifact role must be design or production")
    if main_fit.n_states not in VERSION_1_CANDIDATES:
        raise ModelError("rolling main fit uses an unregistered state count")
    if np.asarray(main_fit.decoded_states).shape != (len(values),):
        raise ModelError("rolling main fit path does not match the frozen reference")
    frame = pd.DataFrame(values, index=index, columns=MACRO_COLUMNS)
    folds = canonical_rolling_folds(mask)
    attempts: list[RollingFitAttempt] = []
    success = np.zeros(6, dtype=np.bool_)
    ari = np.zeros(6, dtype=np.float64)
    jaccards = np.zeros((6, main_fit.n_states), dtype=np.float64)
    for fold in folds:
        try:
            training_index = frame.index[fold.training]
            standardized = standardize_features(frame, training_index=training_index)
            features = _feature_matrix_from_standardized(
                standardized,
                continuation=mask,
                role="design",
                fit_index=training_index,
            )
            scope = (
                ModelFitScope.DESIGN_ROLLING_FOLD
                if role == "design"
                else ModelFitScope.PRODUCTION_MAIN_AND_FOLDS
            )
            replicate = fold.fold if role == "design" else fold.fold + 1
            seeds = deterministic_hmm_restart_seeds(
                master_seed=master_seed,
                scientific_version=scientific_version,
                scope=scope,
                replicate=replicate,
                n_states=main_fit.n_states,
            )
            fitted = fit_function(features, main_fit.n_states, seeds)
            full_standardized = standardized.values.to_numpy(dtype=np.float64)
            reference_lengths = sequence_lengths_from_continuation(mask)
            candidate_states = decode_viterbi(
                full_standardized,
                fitted.parameters,
                lengths=reference_lengths,
            )
            alignment = align_state_labels(
                np.asarray(main_fit.decoded_states, dtype=np.int64),
                candidate_states,
                main_fit.n_states,
            )
            overlap = aligned_state_jaccards(
                np.asarray(main_fit.decoded_states, dtype=np.int64),
                alignment.aligned_states,
                main_fit.n_states,
            )
            validation_values = full_standardized[fold.validation]
            initial = _rolling_initial_probabilities(
                fitted,
                mask,
                validation_start=int(fold.validation.start or 0),
                validation_lengths=fold.validation_lengths,
            )
            score_result = one_step_predictive_log_scores(
                validation_values,
                fitted.parameters,
                lengths=fold.validation_lengths,
                initial_state_probabilities=initial,
            )
            score_tuple = tuple(float(value) for value in score_result.log_scores)
            if len(score_tuple) != 12:
                raise ModelError("rolling validation did not produce exactly 12 scores")
            success[fold.fold] = True
            ari[fold.fold] = alignment.adjusted_rand_index
            jaccards[fold.fold] = overlap
            attempts.append(
                RollingFitAttempt(
                    fold.fold,
                    fold.training_rows,
                    fold.validation_rows,
                    fold.validation_lengths,
                    fitted,
                    score_tuple,
                    alignment.adjusted_rand_index,
                    overlap,
                    None,
                    None,
                )
            )
        except NoNumericallyValidHMMRestarts as error:
            empty = np.zeros(main_fit.n_states, dtype=np.float64)
            empty.setflags(write=False)
            attempts.append(
                RollingFitAttempt(
                    fold.fold,
                    fold.training_rows,
                    fold.validation_rows,
                    fold.validation_lengths,
                    None,
                    (None,) * 12,
                    0.0,
                    empty,
                    type(error).__name__,
                    _bounded(error.message),
                )
            )
    stability = summarize_rolling_stability(success, ari, jaccards)
    return RollingCandidateEvidence(
        main_fit.n_states,
        role,
        tuple(attempts),
        stability,
    )


def materialize_rolling_bootstrap_indices(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
) -> IntArray:
    """Compatibility raw-array producer; not canonical selection authority."""

    return _materialize_rolling_bootstrap_indices(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
    )


def materialize_registered_rolling_bootstrap(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
) -> RegisteredRollingBootstrapMaterialization:
    """Produce and seal the exact registered 10,000x6x12 common trace."""

    indices = _materialize_rolling_bootstrap_indices(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
    )
    rng_fingerprint = rolling_bootstrap_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
    )
    materialization_fingerprint = _rolling_bootstrap_materialization_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        indices=indices,
        rng_fingerprint=rng_fingerprint,
    )
    result = RegisteredRollingBootstrapMaterialization(
        ROLLING_BOOTSTRAP_MATERIALIZATION_CONTRACT,
        master_seed,
        scientific_version,
        artifact,
        indices,
        materialization_fingerprint,
        rng_fingerprint,
    )
    object.__setattr__(
        result,
        "_producer_seal",
        _ROLLING_BOOTSTRAP_MATERIALIZATION_CAPABILITY,
    )
    _MINTED_ROLLING_BOOTSTRAP_MATERIALIZATIONS[id(result)] = (
        result.contract_version,
        result.master_seed,
        result.scientific_version,
        result.artifact,
        result.materialization_fingerprint,
        result.rng_execution_fingerprint,
    )
    return result


def is_registered_rolling_bootstrap_materialization(
    value: object,
) -> TypeGuard[RegisteredRollingBootstrapMaterialization]:
    """Return whether a producer-minted materialization remains unchanged."""

    try:
        _verify_registered_rolling_bootstrap_materialization(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def rolling_bootstrap_rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
) -> str:
    """Fingerprint the complete registered rolling-score RNG key geometry."""

    _validate_rolling_bootstrap_identity(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
    )
    try:
        for replicate in (0, ROLLING_BOOTSTRAP_REPLICATES - 1):
            calibration_rng_key(
                master_seed=master_seed,
                scientific_version=scientific_version,
                purpose=CalibrationPurpose.ROLLING_SCORE_BOOTSTRAP,
                artifact=artifact,
                variant=0,
                replicate=replicate,
                domain=Domain.BLOCK_ALPHA_CALIBRATION,
                family=Family.NOT_APPLICABLE,
                stage=Stage.REFERENCE_RESAMPLING,
            )
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError("rolling bootstrap RNG identity is invalid") from error
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="rolling_score_bootstrap_materialization",
        contract=_rolling_bootstrap_rng_contract(artifact),
    )


def _materialize_rolling_bootstrap_indices(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
) -> IntArray:
    _validate_rolling_bootstrap_identity(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
    )
    indices = np.empty(
        (ROLLING_BOOTSTRAP_REPLICATES, ROLLING_FOLD_COUNT, ROLLING_FOLD_SIZE),
        dtype=np.int64,
    )
    try:
        for replicate in range(ROLLING_BOOTSTRAP_REPLICATES):
            key = calibration_rng_key(
                master_seed=master_seed,
                scientific_version=scientific_version,
                purpose=CalibrationPurpose.ROLLING_SCORE_BOOTSTRAP,
                artifact=artifact,
                variant=0,
                replicate=replicate,
                domain=Domain.BLOCK_ALPHA_CALIBRATION,
                family=Family.NOT_APPLICABLE,
                stage=Stage.REFERENCE_RESAMPLING,
            )
            indices[replicate] = rolling_score_bootstrap_trace(key.generator())
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError("rolling bootstrap RNG identity is invalid") from error
    immutable = np.frombuffer(
        indices.tobytes(order="C"),
        dtype=np.dtype("<i8"),
    ).reshape(indices.shape)
    _require_immutable_rolling_bootstrap_indices(immutable)
    return immutable


def _validate_rolling_bootstrap_identity(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
) -> None:
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or master_seed < 0
    ):
        raise ModelError("rolling bootstrap master seed is invalid")
    if (
        isinstance(scientific_version, bool)
        or not isinstance(scientific_version, int)
        or scientific_version <= 0
    ):
        raise ModelError("rolling bootstrap scientific version is invalid")
    if not isinstance(artifact, CalibrationArtifact):
        raise ModelError("rolling bootstrap artifact is outside the closed registry")


def _verify_registered_rolling_bootstrap_materialization(value: object) -> None:
    if (
        not isinstance(value, RegisteredRollingBootstrapMaterialization)
        or value._producer_seal is not _ROLLING_BOOTSTRAP_MATERIALIZATION_CAPABILITY
        or _MINTED_ROLLING_BOOTSTRAP_MATERIALIZATIONS.get(id(value))
        != (
            value.contract_version,
            value.master_seed,
            value.scientific_version,
            value.artifact,
            value.materialization_fingerprint,
            value.rng_execution_fingerprint,
        )
    ):
        raise ModelError("rolling bootstrap materialization lacks producer authority")
    if value.contract_version != ROLLING_BOOTSTRAP_MATERIALIZATION_CONTRACT:
        raise ModelError("rolling bootstrap materialization contract is invalid")
    _require_immutable_rolling_bootstrap_indices(value.indices)
    rng_fingerprint = rolling_bootstrap_rng_execution_fingerprint(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        artifact=value.artifact,
    )
    materialization_fingerprint = _rolling_bootstrap_materialization_fingerprint(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        artifact=value.artifact,
        indices=value.indices,
        rng_fingerprint=rng_fingerprint,
    )
    if (
        value.rng_execution_fingerprint != rng_fingerprint
        or value.materialization_fingerprint != materialization_fingerprint
    ):
        raise ModelError("rolling bootstrap materialization identity changed")


def _rolling_bootstrap_rng_contract(
    artifact: CalibrationArtifact,
) -> dict[str, object]:
    return {
        "artifact": int(artifact),
        "bit_generator": "PCG64DXSM",
        "domain": int(Domain.BLOCK_ALPHA_CALIBRATION),
        "entity_layout": "purpose:4|artifact:2|variant:6|replicate:20",
        "family": int(Family.NOT_APPLICABLE),
        "fold_count": ROLLING_FOLD_COUNT,
        "fold_size": ROLLING_FOLD_SIZE,
        "purpose": int(CalibrationPurpose.ROLLING_SCORE_BOOTSTRAP),
        "replicate_count": ROLLING_BOOTSTRAP_REPLICATES,
        "replicate_start": 0,
        "replicate_stop_exclusive": ROLLING_BOOTSTRAP_REPLICATES,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "seed_sequence_spawn_key_fields": [
            "rng_contract_version",
            "scientific_version",
            "domain",
            "family",
            "entity",
            "stage",
        ],
        "stage": int(Stage.REFERENCE_RESAMPLING),
        "trace_shape": [
            ROLLING_BOOTSTRAP_REPLICATES,
            ROLLING_FOLD_COUNT,
            ROLLING_FOLD_SIZE,
        ],
        "uniform_draws_per_replicate": (
            ROLLING_FOLD_COUNT + 2 * ROLLING_FOLD_COUNT * (ROLLING_FOLD_SIZE - 1)
        ),
        "variant": 0,
    }


def _rolling_bootstrap_materialization_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    indices: IntArray,
    rng_fingerprint: str,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_rolling_score_bootstrap_v1",
        metadata={
            "contract_version": ROLLING_BOOTSTRAP_MATERIALIZATION_CONTRACT,
            "master_seed": master_seed,
            "scientific_version": scientific_version,
            "rng_execution_fingerprint": rng_fingerprint,
            **_rolling_bootstrap_rng_contract(artifact),
        },
        arrays={"indices": indices},
    )


def _require_immutable_rolling_bootstrap_indices(value: np.ndarray) -> None:
    indices = np.asarray(value)
    expected = (
        ROLLING_BOOTSTRAP_REPLICATES,
        ROLLING_FOLD_COUNT,
        ROLLING_FOLD_SIZE,
    )
    if (
        indices.shape != expected
        or indices.dtype != np.dtype("<i8")
        or indices.flags.writeable
        or not indices.flags.c_contiguous
        or bool(((indices < 0) | (indices >= ROLLING_FOLD_SIZE)).any())
    ):
        raise ModelError("registered rolling bootstrap indices are invalid")
    base: object = indices
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        raise ModelError("registered rolling bootstrap indices are not immutable")


def macro_bootstrap_rng_namespace(role: ArtifactRole) -> str:
    """Return the purpose-0 namespace for the common source-index matrix."""

    if role == "design":
        return "design_macro_bootstrap_source_indices"
    if role == "production":
        return "production_macro_bootstrap_source_indices"
    raise ModelError("macro bootstrap role must be design or production")


def macro_bootstrap_rng_execution_fingerprint(
    *,
    role: ArtifactRole,
    rows: int,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
) -> str:
    """Fingerprint the K-independent purpose-0 draw geometry."""

    _validate_macro_bootstrap_materialization_identity(
        role=role,
        rows=rows,
        macro_block_length=macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    # Construct the endpoints through the closed RNG registry as an additional
    # fail-closed guard against an enum/coordinate drift hidden by metadata.
    for attempt in (0, MACRO_STABILITY_ATTEMPTS - 1):
        calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.MACRO_BOOTSTRAP,
            artifact=artifact,
            variant=0,
            replicate=attempt,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=Family.NOT_APPLICABLE,
            stage=Stage.REFERENCE_RESAMPLING,
        )
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace=macro_bootstrap_rng_namespace(role),
        contract={
            "artifact": int(artifact),
            "attempt_count": MACRO_STABILITY_ATTEMPTS,
            "domain": int(Domain.BLOCK_ALPHA_CALIBRATION),
            "draws_per_attempt": 2 * rows,
            "family": int(Family.NOT_APPLICABLE),
            "macro_block_length": macro_block_length,
            "purpose": int(CalibrationPurpose.MACRO_BOOTSTRAP),
            "replicate_start": 0,
            "replicate_stop_exclusive": MACRO_STABILITY_ATTEMPTS,
            "role": role,
            "rows": rows,
            "stage": int(Stage.REFERENCE_RESAMPLING),
            "variant": 0,
        },
    )


def materialize_registered_macro_bootstrap(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
) -> RegisteredMacroBootstrapMaterialization:
    """Materialize the exact common purpose-0 source indices once.

    The matrix is deliberately independent of candidate ``K``.  Every
    successful design candidate consumes rows from this same immutable object;
    per-K HMM restart streams remain distinct and are bound by each macro
    evidence identity.
    """

    values, index, mask = _macro_frame_inputs(
        macro_features,
        month_ordinals,
        continuation,
    )
    _validate_macro_bootstrap_materialization_identity(
        role=role,
        rows=len(values),
        macro_block_length=macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    indices = np.empty(
        (MACRO_STABILITY_ATTEMPTS, len(values)),
        dtype=np.int64,
    )
    for attempt in range(MACRO_STABILITY_ATTEMPTS):
        key = calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.MACRO_BOOTSTRAP,
            artifact=artifact,
            variant=0,
            replicate=attempt,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=Family.NOT_APPLICABLE,
            stage=Stage.REFERENCE_RESAMPLING,
        )
        draws = key.generator().random(2 * len(values))
        indices[attempt] = macro_bootstrap_indices_from_uniforms(
            mask,
            draws[: len(values)],
            draws[len(values) :],
            block_length=macro_block_length,
        )
    immutable_indices = cast(
        IntArray,
        _immutable_bytes_array(
            indices,
            dtype=np.dtype("<i8"),
            label="macro bootstrap source indices",
        ),
    )
    retained_features = cast(
        FloatArray,
        _immutable_bytes_array(
            values,
            dtype=np.dtype("<f8"),
            label="macro bootstrap source features",
        ),
    )
    retained_ordinals = cast(
        IntArray,
        _immutable_bytes_array(
            index.asi8,
            dtype=np.dtype("<i8"),
            label="macro bootstrap source ordinals",
        ),
    )
    retained_continuation = cast(
        BoolArray,
        _immutable_bytes_array(
            mask,
            dtype=np.dtype("?"),
            label="macro bootstrap source continuation",
        ),
    )
    source_fingerprint = macro_stability_input_fingerprint(
        retained_features,
        retained_ordinals,
        retained_continuation,
        role=role,
    )
    rng_fingerprint = macro_bootstrap_rng_execution_fingerprint(
        role=role,
        rows=len(values),
        macro_block_length=macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    materialization_fingerprint = _macro_bootstrap_materialization_fingerprint(
        role=role,
        macro_block_length=macro_block_length,
        rows=len(values),
        master_seed=master_seed,
        scientific_version=scientific_version,
        source_input_fingerprint=source_fingerprint,
        rng_fingerprint=rng_fingerprint,
        indices=immutable_indices,
    )
    result = RegisteredMacroBootstrapMaterialization(
        contract_version=MACRO_BOOTSTRAP_MATERIALIZATION_CONTRACT,
        role=role,
        macro_block_length=macro_block_length,
        rows=len(values),
        master_seed=master_seed,
        scientific_version=scientific_version,
        indices=immutable_indices,
        source_input_fingerprint=source_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
    )
    object.__setattr__(result, "_macro_features", retained_features)
    object.__setattr__(result, "_month_ordinals", retained_ordinals)
    object.__setattr__(result, "_continuation", retained_continuation)
    object.__setattr__(
        result,
        "_producer_seal",
        _MACRO_BOOTSTRAP_MATERIALIZATION_CAPABILITY,
    )
    _MINTED_MACRO_BOOTSTRAP_MATERIALIZATIONS[id(result)] = result
    registered_macro_bootstrap_materialization_identity(result)
    return result


def registered_macro_bootstrap_materialization_identity(
    value: RegisteredMacroBootstrapMaterialization,
) -> RegisteredMacroBootstrapMaterializationIdentity:
    """Reverify one producer-minted common macro source matrix."""

    if (
        not isinstance(value, RegisteredMacroBootstrapMaterialization)
        or value._producer_seal is not _MACRO_BOOTSTRAP_MATERIALIZATION_CAPABILITY
        or _MINTED_MACRO_BOOTSTRAP_MATERIALIZATIONS.get(id(value)) is not value
    ):
        raise ModelError("macro bootstrap materialization lacks producer authority")
    if value.contract_version != MACRO_BOOTSTRAP_MATERIALIZATION_CONTRACT:
        raise ModelError("macro bootstrap materialization contract is invalid")
    features = value._macro_features
    ordinals = value._month_ordinals
    continuation = value._continuation
    if features is None or ordinals is None or continuation is None:
        raise ModelError("macro bootstrap materialization source is incomplete")
    _validate_macro_bootstrap_materialization_identity(
        role=value.role,
        rows=value.rows,
        macro_block_length=value.macro_block_length,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
    )
    indices = np.asarray(value.indices)
    if (
        indices.shape != (MACRO_STABILITY_ATTEMPTS, value.rows)
        or indices.dtype != np.dtype("<i8")
        or bool(((indices < 0) | (indices >= value.rows)).any())
    ):
        raise ModelError("macro bootstrap source-index geometry is invalid")
    _require_immutable_bytes_array(indices, "macro bootstrap source indices")
    _require_immutable_bytes_array(features, "macro bootstrap source features")
    _require_immutable_bytes_array(ordinals, "macro bootstrap source ordinals")
    _require_immutable_bytes_array(
        continuation,
        "macro bootstrap source continuation",
    )
    source_fingerprint = macro_stability_input_fingerprint(
        features,
        ordinals,
        continuation,
        role=value.role,
    )
    rng_fingerprint = macro_bootstrap_rng_execution_fingerprint(
        role=value.role,
        rows=value.rows,
        macro_block_length=value.macro_block_length,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
    )
    materialization_fingerprint = _macro_bootstrap_materialization_fingerprint(
        role=value.role,
        macro_block_length=value.macro_block_length,
        rows=value.rows,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        source_input_fingerprint=source_fingerprint,
        rng_fingerprint=rng_fingerprint,
        indices=indices,
    )
    if (
        source_fingerprint != value.source_input_fingerprint
        or rng_fingerprint != value.rng_execution_fingerprint
        or materialization_fingerprint != value.materialization_fingerprint
    ):
        raise ModelError("macro bootstrap materialization identity changed")
    return RegisteredMacroBootstrapMaterializationIdentity(
        role=value.role,
        macro_block_length=value.macro_block_length,
        rows=value.rows,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        source_input_fingerprint=source_fingerprint,
        rng_namespace=macro_bootstrap_rng_namespace(value.role),
        rng_execution_fingerprint=rng_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
    )


def is_registered_macro_bootstrap_materialization(
    value: object,
) -> TypeGuard[RegisteredMacroBootstrapMaterialization]:
    """Return whether a macro source matrix is live and producer-minted."""

    if not isinstance(value, RegisteredMacroBootstrapMaterialization):
        return False
    try:
        registered_macro_bootstrap_materialization_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def _validate_macro_bootstrap_materialization_identity(
    *,
    role: ArtifactRole,
    rows: int,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
) -> None:
    if role not in {"design", "production"}:
        raise ModelError("macro bootstrap role must be design or production")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 72:
        raise ModelError("macro bootstrap row count must be at least 72")
    if (
        isinstance(macro_block_length, bool)
        or not isinstance(macro_block_length, int)
        or not 2 <= macro_block_length <= 24
        or macro_block_length > rows // 8
    ):
        raise ModelError("macro bootstrap block length violates a binding hard cap")
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or master_seed < 0
    ):
        raise ModelError("macro bootstrap master seed is invalid")
    if (
        isinstance(scientific_version, bool)
        or not isinstance(scientific_version, int)
        or scientific_version <= 0
    ):
        raise ModelError("macro bootstrap scientific version is invalid")


def _macro_bootstrap_materialization_fingerprint(
    *,
    role: ArtifactRole,
    macro_block_length: int,
    rows: int,
    master_seed: int,
    scientific_version: int,
    source_input_fingerprint: str,
    rng_fingerprint: str,
    indices: IntArray,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_common_macro_bootstrap_materialization_v1",
        metadata={
            "attempt_count": MACRO_STABILITY_ATTEMPTS,
            "contract_version": MACRO_BOOTSTRAP_MATERIALIZATION_CONTRACT,
            "macro_block_length": macro_block_length,
            "master_seed": master_seed,
            "rng_execution_fingerprint": rng_fingerprint,
            "role": role,
            "rows": rows,
            "scientific_version": scientific_version,
            "source_input_fingerprint": source_input_fingerprint,
        },
        arrays={"source_indices": indices},
    )


def macro_bootstrap_indices_from_uniforms(
    continuation: ArrayLike,
    restart_uniforms: ArrayLike,
    source_rank_uniforms: ArrayLike,
    *,
    block_length: int,
) -> IntArray:
    """Build one synchronized non-circular, gap-aware macro source trace."""

    raw_mask = np.asarray(continuation)
    if raw_mask.ndim != 1 or raw_mask.size == 0 or raw_mask.dtype.kind != "b":
        raise ModelError("macro continuation must be a non-empty boolean row mask")
    if bool(raw_mask[0]):
        raise ModelError("macro continuation must be false at the first row")
    if isinstance(block_length, bool) or not isinstance(block_length, int):
        raise ModelError("macro block length must be an integer")
    if block_length < 2:
        raise ModelError("macro block length must be at least two")
    mask = raw_mask.astype(np.bool_, copy=False)
    restarts = _uniforms(restart_uniforms, len(mask), "macro restart uniforms")
    ranks = _uniforms(source_rank_uniforms, len(mask), "macro source-rank uniforms")
    probability = 1.0 / block_length
    indices = np.empty(len(mask), dtype=np.int64)
    for row in range(len(mask)):
        forced = (
            row == 0
            or not bool(mask[row])
            or indices[row - 1] + 1 >= len(mask)
            or not bool(mask[indices[row - 1] + 1])
        )
        if forced or restarts[row] < probability:
            indices[row] = min(int(ranks[row] * len(mask)), len(mask) - 1)
        else:
            indices[row] = indices[row - 1] + 1
    indices.setflags(write=False)
    return indices


def execute_registered_macro_stability(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    main_fit: GaussianHMMFit,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
    source_materialization: RegisteredMacroBootstrapMaterialization | None = None,
    workers: int = 1,
) -> MacroStabilityEvidence:
    """Run and producer-seal the canonical 100-attempt macro family.

    The canonical entry point deliberately exposes no injectable fitter.  Test
    doubles and exploratory algorithms belong to :func:`run_macro_stability`,
    whose result is report-only and cannot cross a registered task boundary.
    """

    worker_count = _validated_hmm_workers(workers)
    return _execute_macro_stability(
        macro_features,
        month_ordinals,
        continuation,
        role=role,
        main_fit=main_fit,
        macro_block_length=macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
        fit_function=_registered_hmm_fit_function(
            scientific_version=scientific_version,
            workers=worker_count,
        ),
        registered=True,
        source_materialization=source_materialization,
    )


def run_macro_stability(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    main_fit: GaussianHMMFit,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> MacroStabilityEvidence:
    """Run a report-only macro family with an optionally injected fitter.

    This compatibility API is intentionally incapable of minting canonical
    evidence.  Production orchestration must call
    :func:`execute_registered_macro_stability`.
    """

    return _execute_macro_stability(
        macro_features,
        month_ordinals,
        continuation,
        role=role,
        main_fit=main_fit,
        macro_block_length=macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
        fit_function=fit_function,
        registered=False,
        source_materialization=None,
    )


def _execute_macro_stability(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
    *,
    role: ArtifactRole,
    main_fit: GaussianHMMFit,
    macro_block_length: int,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction,
    registered: bool,
    source_materialization: RegisteredMacroBootstrapMaterialization | None,
) -> MacroStabilityEvidence:
    """Shared numerical implementation for registered and report-only runs."""

    values, index, mask = _macro_frame_inputs(
        macro_features, month_ordinals, continuation
    )
    if role not in {"design", "production"}:
        raise ModelError("macro stability role must be design or production")
    if main_fit.n_states not in VERSION_1_CANDIDATES:
        raise ModelError("macro stability main fit has an unregistered K")
    if np.asarray(main_fit.decoded_states).shape != (len(values),):
        raise ModelError("macro stability reference path has the wrong length")
    if (
        isinstance(macro_block_length, bool)
        or not isinstance(macro_block_length, int)
        or not 2 <= macro_block_length <= 24
        or macro_block_length > len(values) // 8
    ):
        raise ModelError("macro stability block length violates a binding hard cap")
    artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    scope = (
        ModelFitScope.DESIGN_MACRO_STABILITY
        if role == "design"
        else ModelFitScope.PRODUCTION_MACRO_STABILITY
    )
    if registered:
        if source_materialization is None:
            source_materialization = materialize_registered_macro_bootstrap(
                values,
                index.asi8,
                mask,
                role=role,
                macro_block_length=macro_block_length,
                master_seed=master_seed,
                scientific_version=scientific_version,
            )
        source_identity = registered_macro_bootstrap_materialization_identity(
            source_materialization
        )
        expected_input = macro_stability_input_fingerprint(
            values,
            index.asi8,
            mask,
            role=role,
        )
        if (
            source_identity.role != role
            or source_identity.rows != len(values)
            or source_identity.macro_block_length != macro_block_length
            or source_identity.master_seed != master_seed
            or source_identity.scientific_version != scientific_version
            or source_identity.source_input_fingerprint != expected_input
        ):
            raise ModelError(
                "macro stability source materialization does not match execution"
            )
    elif source_materialization is not None:
        raise ModelError("report-only macro stability forbids registered sources")
    attempts: list[MacroBootstrapAttempt] = []
    success = np.zeros(100, dtype=np.bool_)
    ari = np.zeros(100, dtype=np.float64)
    jaccards = np.zeros((100, main_fit.n_states), dtype=np.float64)
    reference_states = np.asarray(main_fit.decoded_states, dtype=np.int64)
    reference_lengths = sequence_lengths_from_continuation(mask)
    for attempt in range(100):
        if source_materialization is None:
            key = calibration_rng_key(
                master_seed=master_seed,
                scientific_version=scientific_version,
                purpose=CalibrationPurpose.MACRO_BOOTSTRAP,
                artifact=artifact,
                variant=0,
                replicate=attempt,
                domain=Domain.BLOCK_ALPHA_CALIBRATION,
                family=Family.NOT_APPLICABLE,
                stage=Stage.REFERENCE_RESAMPLING,
            )
            draws = key.generator().random(2 * len(values))
            source_indices = macro_bootstrap_indices_from_uniforms(
                mask,
                draws[: len(values)],
                draws[len(values) :],
                block_length=macro_block_length,
            )
        else:
            source_indices = cast(
                IntArray,
                np.asarray(source_materialization.indices[attempt]),
            )
        try:
            resampled = values[source_indices]
            fitted_features = _all_row_feature_matrix(
                resampled,
                index=index,
                continuation=mask,
                scaler_scope=main_fit.scaler_scope,
            )
            seeds = deterministic_hmm_restart_seeds(
                master_seed=master_seed,
                scientific_version=scientific_version,
                scope=scope,
                replicate=attempt,
                n_states=main_fit.n_states,
            )
            fitted = fit_function(fitted_features, main_fit.n_states, seeds)
            reference_standardized = (
                values - fitted.scaler_means
            ) / fitted.scaler_standard_deviations
            candidate_states = decode_viterbi(
                reference_standardized,
                fitted.parameters,
                lengths=reference_lengths,
            )
            alignment = align_state_labels(
                reference_states, candidate_states, main_fit.n_states
            )
            overlap = aligned_state_jaccards(
                reference_states, alignment.aligned_states, main_fit.n_states
            )
            success[attempt] = True
            ari[attempt] = alignment.adjusted_rand_index
            jaccards[attempt] = overlap
            candidate_to_reference = alignment.candidate_to_reference.astype(
                np.int64, copy=True
            )
            candidate_to_reference.setflags(write=False)
            attempts.append(
                MacroBootstrapAttempt(
                    attempt,
                    source_indices,
                    fitted,
                    alignment.adjusted_rand_index,
                    overlap,
                    None,
                    None,
                    candidate_to_reference,
                )
            )
        except NoNumericallyValidHMMRestarts as error:
            empty = np.zeros(main_fit.n_states, dtype=np.float64)
            empty.setflags(write=False)
            attempts.append(
                MacroBootstrapAttempt(
                    attempt,
                    source_indices,
                    None,
                    0.0,
                    empty,
                    type(error).__name__,
                    _bounded(error.message),
                )
            )
    summary = summarize_macro_bootstrap_stability(success, ari, jaccards)
    result = MacroStabilityEvidence(
        main_fit.n_states,
        role,
        macro_block_length,
        tuple(attempts),
        summary,
    )
    if not registered:
        return result
    result = cast(MacroStabilityEvidence, _immutable_public_tree(result))
    retained_features = cast(
        FloatArray,
        _immutable_bytes_array(
            values,
            dtype=np.dtype("<f8"),
            label="macro stability source features",
        ),
    )
    retained_ordinals = cast(
        IntArray,
        _immutable_bytes_array(
            index.asi8,
            dtype=np.dtype("<i8"),
            label="macro stability source ordinals",
        ),
    )
    retained_continuation = cast(
        BoolArray,
        _immutable_bytes_array(
            mask,
            dtype=np.dtype("?"),
            label="macro stability source continuation",
        ),
    )
    object.__setattr__(result, "_parent_model", main_fit)
    object.__setattr__(result, "_macro_features", retained_features)
    object.__setattr__(result, "_month_ordinals", retained_ordinals)
    object.__setattr__(result, "_continuation", retained_continuation)
    object.__setattr__(result, "_source_materialization", source_materialization)
    object.__setattr__(result, "_master_seed", master_seed)
    object.__setattr__(result, "_scientific_version", scientific_version)
    parent_fingerprint = gaussian_hmm_fit_fingerprint(main_fit)
    input_fingerprint = macro_stability_input_fingerprint(
        retained_features,
        retained_ordinals,
        retained_continuation,
        role=role,
    )
    rng_fingerprint = macro_stability_rng_fingerprint(
        role=role,
        n_states=main_fit.n_states,
        rows=len(retained_features),
        macro_block_length=macro_block_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    from prpg.model.calibration_identity import (
        macro_stability_evidence_fingerprint,
    )

    evidence_fingerprint = macro_stability_evidence_fingerprint(result)
    object.__setattr__(result, "_parent_model_fingerprint", parent_fingerprint)
    object.__setattr__(result, "_macro_input_fingerprint", input_fingerprint)
    object.__setattr__(result, "_rng_execution_fingerprint", rng_fingerprint)
    object.__setattr__(result, "_evidence_fingerprint", evidence_fingerprint)
    object.__setattr__(
        result,
        "_execution_seal",
        _MACRO_STABILITY_EXECUTION_CAPABILITY,
    )
    _MINTED_MACRO_STABILITY_EVIDENCE[id(result)] = result
    registered_macro_stability_execution_identity(result)
    return result


def evaluate_sealed_holdout(
    selected_fit: GaussianHMMFit,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
) -> SealedHoldoutEvidence:
    """Apply the selected-design-model veto without any runner-up promotion."""

    if design.role != "design" or holdout.role != "holdout":
        raise ModelError("sealed holdout requires design and holdout slices")
    means = np.asarray(selected_fit.scaler_means, dtype=np.float64)
    scales = np.asarray(selected_fit.scaler_standard_deviations, dtype=np.float64)
    if means.shape != (4,) or scales.shape != (4,) or bool((scales <= 0.0).any()):
        raise ModelError("selected design scaler is invalid for holdout evaluation")
    design_values = (np.asarray(design.macro_features) - means) / scales
    holdout_values = (np.asarray(holdout.macro_features) - means) / scales
    if not np.isfinite(design_values).all() or not np.isfinite(holdout_values).all():
        raise ModelError("sealed holdout standardization produced non-finite values")
    lengths = sequence_lengths_from_continuation(holdout.monthly_continuation)
    stationary = selected_fit.chain_diagnostics.stationary_distribution
    initial = np.tile(stationary, (len(lengths), 1)).astype(np.float64)
    terminal = selected_fit.forward_backward.filtered_probabilities[-1]
    initial[0] = terminal @ selected_fit.parameters.transition_matrix
    hmm_scores = one_step_predictive_log_scores(
        holdout_values,
        selected_fit.parameters,
        lengths=lengths,
        initial_state_probabilities=initial,
    ).log_scores
    benchmarks = score_holdout_benchmarks(
        design_values,
        holdout_values,
        holdout.monthly_continuation,
    )
    decision = holdout_veto_decision(hmm_scores, benchmarks)
    return SealedHoldoutEvidence(lengths, hmm_scores, benchmarks, decision)


def _feature_matrix_from_standardized(
    standardized: StandardizedFeatures,
    *,
    continuation: BoolArray,
    role: ArtifactRole,
    fit_index: pd.Index | None,
) -> HMMFeatureMatrix:
    # Imported here to keep the public preparation contract in one function.
    from prpg.model.hmm import consume_standardized_features

    return consume_standardized_features(
        standardized,
        continuation=continuation,
        scaler_scope=("design_training" if role == "design" else "production_full"),
        fit_index=fit_index,
    )


def _macro_frame_inputs(
    macro_features: ArrayLike,
    month_ordinals: ArrayLike,
    continuation: ArrayLike,
) -> tuple[FloatArray, pd.PeriodIndex, BoolArray]:
    values = np.asarray(macro_features, dtype=np.float64)
    ordinals = np.asarray(month_ordinals)
    raw_mask = np.asarray(continuation)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 72:
        raise ModelError("rolling macro features require at least 72 four-column rows")
    if not np.isfinite(values).all():
        raise ModelError("rolling macro features contain non-finite values")
    if ordinals.shape != (len(values),) or ordinals.dtype.kind not in {"i", "u"}:
        raise ModelError("rolling month ordinals do not match feature rows")
    if (
        raw_mask.shape != (len(values),)
        or raw_mask.dtype.kind != "b"
        or bool(raw_mask[0])
    ):
        raise ModelError("rolling continuation must be a false-at-start row mask")
    index = pd.PeriodIndex.from_ordinals(ordinals.astype(np.int64), freq="M")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ModelError("rolling month ordinals must be unique and increasing")
    return values, index, raw_mask.astype(np.bool_, copy=False)


def _all_row_feature_matrix(
    values: FloatArray,
    *,
    index: pd.PeriodIndex,
    continuation: BoolArray,
    scaler_scope: Literal["design_training", "production_full"],
) -> HMMFeatureMatrix:
    frame = pd.DataFrame(values, index=index, columns=MACRO_COLUMNS)
    standardized = standardize_features(frame)
    matrix = standardized.values.to_numpy(dtype=np.float64)
    means = standardized.means.to_numpy(dtype=np.float64)
    scales = standardized.standard_deviations.to_numpy(dtype=np.float64)
    return HMMFeatureMatrix(
        values=matrix,
        lengths=sequence_lengths_from_continuation(continuation),
        feature_names=tuple(MACRO_COLUMNS),
        row_labels=tuple(str(value) for value in index),
        scaler_means=means,
        scaler_standard_deviations=scales,
        scaler_scope=scaler_scope,
    )


def _rolling_initial_probabilities(
    fitted: GaussianHMMFit,
    continuation: BoolArray,
    *,
    validation_start: int,
    validation_lengths: tuple[int, ...],
) -> FloatArray:
    transition = fitted.parameters.transition_matrix
    stationary = fitted.chain_diagnostics.stationary_distribution
    initial = np.tile(stationary, (len(validation_lengths), 1)).astype(np.float64)
    if validation_start > 0 and bool(continuation[validation_start]):
        terminal = fitted.forward_backward.filtered_probabilities[-1]
        initial[0] = terminal @ transition
    return initial


def _uniforms(values: ArrayLike, expected: int, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ModelError(f"{label} must contain exactly {expected} finite values")
    if bool(((result < 0.0) | (result >= 1.0)).any()):
        raise ModelError(f"{label} must lie in [0, 1)")
    return result


def _lengths(mask: BoolArray) -> tuple[int, ...]:
    starts = np.flatnonzero(~mask)
    boundaries = np.r_[starts, len(mask)]
    return tuple(int(value) for value in np.diff(boundaries))


def _macro_stability_summaries_equal(
    actual: MacroStabilitySummary,
    expected: MacroStabilitySummary,
) -> bool:
    """Compare a stored macro decision with its exact attempt reduction."""

    return bool(
        isinstance(actual, MacroStabilitySummary)
        and actual.successful_attempts == expected.successful_attempts
        and actual.median_ari == expected.median_ari
        and actual.tenth_percentile_ari == expected.tenth_percentile_ari
        and actual.passed is expected.passed
        and np.array_equal(
            np.asarray(actual.median_jaccard_by_state),
            np.asarray(expected.median_jaccard_by_state),
        )
        and np.array_equal(
            np.asarray(actual.tenth_percentile_jaccard_by_state),
            np.asarray(expected.tenth_percentile_jaccard_by_state),
        )
    )


def _rolling_stability_summaries_equal(
    actual: RollingStabilitySummary,
    expected: RollingStabilitySummary,
) -> bool:
    """Compare a stored rolling decision with its exact six-fold reduction."""

    return bool(
        isinstance(actual, RollingStabilitySummary)
        and actual.successful_fits == expected.successful_fits
        and actual.median_ari == expected.median_ari
        and actual.minimum_successful_ari == expected.minimum_successful_ari
        and actual.passed is expected.passed
        and np.array_equal(
            np.asarray(actual.median_jaccard_by_state),
            np.asarray(expected.median_jaccard_by_state),
        )
    )


def _immutable_public_tree(value: object) -> object:
    """Copy every public ndarray into an immutable bytes-backed tree."""

    if isinstance(value, np.ndarray):
        return _immutable_bytes_array(
            value,
            dtype=value.dtype,
            label="macro stability output",
        )
    if isinstance(value, tuple):
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
    """Fail closed unless every public output ndarray is bytes-backed."""

    if isinstance(value, np.ndarray):
        _require_immutable_bytes_array(value, "macro stability output")
        return
    if isinstance(value, tuple | list):
        for item in value:
            _require_immutable_public_arrays(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if not item.name.startswith("_"):
                _require_immutable_public_arrays(getattr(value, item.name))


def _immutable_bytes_array(
    value: ArrayLike,
    *,
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    """Return a C-contiguous array whose ultimate base is immutable bytes."""

    source = np.asarray(value)
    canonical = np.ascontiguousarray(source.astype(dtype, copy=False))
    immutable = np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=dtype,
    ).reshape(canonical.shape)
    _require_immutable_bytes_array(immutable, label)
    return immutable


def _require_immutable_bytes_array(value: np.ndarray, label: str) -> None:
    array = np.asarray(value)
    if array.flags.writeable or not array.flags.c_contiguous:
        raise ModelError(f"{label} must be a read-only contiguous array")
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        raise ModelError(f"{label} must be backed by immutable bytes")


def _bounded(value: str, limit: int = 500) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _registered_hmm_fit_function(
    *,
    scientific_version: int,
    workers: int,
) -> FitFunction:
    """Bind only the prospective version to the owner-fixed K=4 policy."""

    policy = owner_fixed_k4_policy_for_scientific_version(scientific_version)
    if workers == 1 and policy is None:
        return fit_gaussian_hmm_restarts
    if policy is None:
        return partial(fit_gaussian_hmm_restarts, workers=workers)
    return partial(
        fit_gaussian_hmm_restarts,
        workers=workers,
        owner_fixed_k4_policy=policy,
    )


def _validated_hmm_workers(value: int) -> int:
    """Validate the local restart-pool width for a registered HMM producer."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError("HMM workers must be a positive integer")
    return value


def _frozen_scientific_details(
    values: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, _frozen_metadata(value)) for key, value in sorted(values.items())
    )


def _immutable_candidate_failure_details(
    values: Mapping[str, object],
) -> Mapping[str, object]:
    """Own and deeply freeze one candidate-family rejection document."""

    return MappingProxyType(
        {
            str(key): _frozen_metadata(value)
            for key, value in sorted(values.items(), key=lambda pair: str(pair[0]))
        }
    )


def _frozen_metadata(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _frozen_metadata(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list | tuple):
        return tuple(_frozen_metadata(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ModelError("scientific failure details contain an unsupported value")
