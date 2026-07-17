"""Tier-A bridge execution, including the prospective fixed-K4 path.

Tier A intentionally repeats only the macro/HMM order-selection core.  It
does not recursively invoke stability bootstraps, adequacy, return-pool,
block, dependence, kernel, or G5 work.  Each nested prefix/full member is
freshly population-standardized, fits K=2..5 with the registered ordinary
50-start namespaces, runs the six exact expanding predictive folds, and then
delegates the all-pairs max-z/BIC decision to :mod:`prpg.model.selection`.

The historical order-selection implementation remains available for audit.
The canonical version-3 path fits only K=4 at the main-fit coordinate
``pair*7`` with 50 ordinary registered starts.  It does not run rolling folds,
cross-K selection, or BIC.  Scientific fit failures remain explicit evidence;
malformed input, unexpected model faults, and registry drift raise immediately.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.bridge import TierAResult, tier_a_selection_gate
from prpg.model.bridge_history import (
    BRIDGE_FULL_ROWS,
    BRIDGE_GAP_ROW,
    BRIDGE_PREFIX_ROWS,
    BridgeDGP,
    BridgePseudoHistory,
    bridge_fit_input_sha256,
    bridge_parameters_sha256,
    validate_bridge_pseudo_history,
)
from prpg.model.calibration import (
    VERSION_1_CANDIDATES,
    CandidateFitAttempt,
    CandidateFitSet,
    FitFunction,
    fit_hmm_candidate_set,
)
from prpg.model.candidate_gate import (
    MAXIMUM_TIED_COVARIANCE_CONDITION,
    MINIMUM_COMPLETED_MONTHLY_EPISODES,
    MINIMUM_MONTHLY_OBSERVATIONS,
    MINIMUM_STATE_OCCUPANCY,
    OWNER_FIXED_MINIMUM_OBSERVED_EPISODES,
    OWNER_FIXED_N_STATES,
)
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMCandidateInadmissibility,
    HMMFeatureMatrix,
    HMMParameters,
    NoNumericallyValidHMMRestarts,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_restarts,
    one_step_predictive_log_scores,
    sequence_lengths_from_continuation,
)
from prpg.model.input import MACRO_COLUMNS
from prpg.model.refit_adequacy import align_refit_to_artifact
from prpg.model.regime_policy import (
    effective_identification_passed,
    effective_transition_support_passed,
    owner_fixed_k4_policy_for_scientific_version,
)
from prpg.model.selection import (
    CandidateSelectionInput,
    CandidateSelectionReport,
    CandidateViabilityEvidence,
    CandidateViabilityRecord,
    PreselectionCheck,
    StateViabilityEvidence,
    ViabilityThresholds,
    evaluate_candidate_viability,
    select_candidate,
)
from prpg.model.stability import RollingFold, canonical_rolling_folds
from prpg.simulation.rng import ModelFitScope

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
BridgeMember: TypeAlias = Literal["prefix", "full"]
MemberRunner: TypeAlias = Callable[..., "BridgeSelectionMemberResult"]
FixedK4MemberRunner: TypeAlias = Callable[..., "BridgeFixedK4TierAMemberResult"]

BRIDGE_FITS_PER_SELECTION_MEMBER = 7
BRIDGE_FIXED_K4_TIER_A_POLICY = "owner_fixed_k4_tier_a_v3"
BRIDGE_FIXED_K4_TIER_A_FITS_PER_MEMBER = 1
BRIDGE_FIXED_K4_TIER_A_ROLLING_FOLDS = 0
BRIDGE_FIXED_K4_TIER_A_USES_CROSS_K_SELECTION = False
BRIDGE_FIXED_K4_TIER_A_USES_BIC = False
BRIDGE_FIXED_K4_TIER_A_RESTARTS_PER_PAIR = 2 * HMM_RESTARTS


@dataclass(frozen=True, slots=True)
class BridgeRollingAttempt:
    """One exact bridge fold, including a retained scientific failure."""

    fold: int
    scope: ModelFitScope
    replicate: int
    training_rows: int
    validation_rows: int
    validation_lengths: tuple[int, ...]
    fit: GaussianHMMFit | None
    scores: tuple[float | None, ...]
    failure_type: str | None
    failure_message: str | None

    @property
    def successful(self) -> bool:
        return self.fit is not None


@dataclass(frozen=True, slots=True)
class BridgeCandidateSelectionEvidence:
    """One K's exact Tier-A main/rolling evidence and viability record."""

    n_states: int
    main_attempt: CandidateFitAttempt
    rolling_attempts: tuple[BridgeRollingAttempt, ...]
    viability: CandidateViabilityRecord

    @property
    def rolling_scores(self) -> tuple[float | None, ...]:
        return tuple(
            score for attempt in self.rolling_attempts for score in attempt.scores
        )


@dataclass(frozen=True, slots=True)
class BridgeSelectionMemberResult:
    """Complete prefix or full ordinary selection result for one pair."""

    dgp: BridgeDGP
    member: BridgeMember
    pair: int
    scope: ModelFitScope
    main_replicate: int
    lengths: tuple[int, ...]
    rolling_bootstrap_sha256: str
    fit_set: CandidateFitSet
    candidates: tuple[BridgeCandidateSelectionEvidence, ...]
    selection: CandidateSelectionReport


@dataclass(frozen=True, slots=True)
class BridgeTierAPairResult:
    """One nested pseudo-pair's two selections and agreement outcome."""

    dgp: BridgeDGP
    pair: int
    actual_selected_k: int
    history_sha256: str
    rolling_bootstrap_sha256: str
    prefix: BridgeSelectionMemberResult | None
    full: BridgeSelectionMemberResult | None
    member_failures: tuple[BridgeTierAMemberFailure, ...]
    success: bool


@dataclass(frozen=True, slots=True)
class BridgeTierAMemberFailure:
    """A scientific member failure retained as a Tier-A disagreement."""

    member: BridgeMember
    failure_type: str
    failure_message: str


@dataclass(frozen=True, slots=True)
class BridgeTierAExecution:
    """Exactly 250 ordered pseudo-pairs and the one-sided CP gate."""

    dgp: BridgeDGP
    actual_selected_k: int
    pairs: tuple[BridgeTierAPairResult, ...]
    gate: TierAResult


@dataclass(frozen=True, slots=True)
class BridgeFixedK4TierAMemberResult:
    """One version-3 Tier-A member fitted exactly once at K=4."""

    dgp: BridgeDGP
    member: BridgeMember
    pair: int
    scope: ModelFitScope
    replicate: int
    lengths: tuple[int, ...]
    random_restart_ids: tuple[int, ...]
    fit: GaussianHMMFit | None
    observed_episode_counts: tuple[int, ...]
    reference_to_candidate: tuple[int, ...]
    numerical_chain_passed: bool
    effective_identification_passed: bool
    observed_episode_support_passed: bool
    alignment_succeeded: bool
    rejection_reasons: tuple[str, ...]
    failure_type: str | None
    failure_message: str | None

    @property
    def passed(self) -> bool:
        return bool(
            self.fit is not None
            and self.numerical_chain_passed
            and self.effective_identification_passed
            and self.observed_episode_support_passed
            and self.alignment_succeeded
            and not self.rejection_reasons
            and self.failure_type is None
            and self.failure_message is None
        )


@dataclass(frozen=True, slots=True)
class BridgeFixedK4TierAPairResult:
    """One version-3 fixed-K4 nested pseudo-pair decision."""

    dgp: BridgeDGP
    pair: int
    actual_selected_k: Literal[4]
    history_sha256: str
    prefix: BridgeFixedK4TierAMemberResult
    full: BridgeFixedK4TierAMemberResult
    success: bool


@dataclass(frozen=True, slots=True)
class BridgeFixedK4TierAExecution:
    """Exactly 250 ordered fixed-K4 pseudo-pairs and the retained CP gate."""

    dgp: BridgeDGP
    actual_selected_k: Literal[4]
    pairs: tuple[BridgeFixedK4TierAPairResult, ...]
    gate: TierAResult


def bridge_selection_scope(
    dgp: BridgeDGP,
    member: BridgeMember,
) -> ModelFitScope:
    """Return the closed ordinary bridge-selection fit scope."""

    mapping = {
        ("model", "prefix"): ModelFitScope.BRIDGE_MODEL_SELECTION_PREFIX,
        ("model", "full"): ModelFitScope.BRIDGE_MODEL_SELECTION_FULL,
        (
            "nonparametric",
            "prefix",
        ): ModelFitScope.BRIDGE_NONPARAMETRIC_SELECTION_PREFIX,
        (
            "nonparametric",
            "full",
        ): ModelFitScope.BRIDGE_NONPARAMETRIC_SELECTION_FULL,
    }
    try:
        return mapping[(dgp, member)]
    except KeyError as error:
        raise ModelError(
            "bridge selection DGP/member is outside the registry"
        ) from error


def run_bridge_selection_member(
    values: ArrayLike,
    continuation: ArrayLike,
    *,
    dgp: BridgeDGP,
    member: BridgeMember,
    pair: int,
    master_seed: int,
    scientific_version: int,
    registered_bootstrap_indices: ArrayLike,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> BridgeSelectionMemberResult:
    """Run the exact K=2..5 order-selection core for one nested member."""

    matrix, mask, lengths = _member_inputs(values, continuation, member=member)
    checked_pair = _pair(pair)
    scope = bridge_selection_scope(dgp, member)
    main_replicate = checked_pair * BRIDGE_FITS_PER_SELECTION_MEMBER
    features = bridge_feature_matrix(matrix, mask)
    bootstrap = _registered_bootstrap_indices(registered_bootstrap_indices)
    bootstrap_sha256 = _rolling_bootstrap_sha256(bootstrap)
    fit_set = fit_hmm_candidate_set(
        features,
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=scope,
        replicate=main_replicate,
        fit_function=fit_function,
    )
    evidence: list[BridgeCandidateSelectionEvidence] = []
    inputs: list[CandidateSelectionInput] = []
    for attempt in fit_set.attempts:
        rolling = _rolling_attempts(
            matrix,
            mask,
            n_states=attempt.n_states,
            scope=scope,
            pair=checked_pair,
            master_seed=master_seed,
            scientific_version=scientific_version,
            fit_function=fit_function,
            main_fit=attempt.fit,
        )
        viability = _bridge_viability(attempt)
        candidate = BridgeCandidateSelectionEvidence(
            attempt.n_states,
            attempt,
            rolling,
            viability,
        )
        evidence.append(candidate)
        inputs.append(CandidateSelectionInput(viability, candidate.rolling_scores))
    selection = select_candidate(
        tuple(inputs),
        bootstrap_indices=bootstrap,
    )
    return BridgeSelectionMemberResult(
        dgp,
        member,
        checked_pair,
        scope,
        main_replicate,
        lengths,
        bootstrap_sha256,
        fit_set,
        tuple(evidence),
        selection,
    )


def run_bridge_fixed_k4_tier_a_member(
    values: ArrayLike,
    continuation: ArrayLike,
    *,
    parent_parameters: HMMParameters,
    dgp: BridgeDGP,
    member: BridgeMember,
    pair: int,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> BridgeFixedK4TierAMemberResult:
    """Fit one Tier-A member at exactly K=4 with 50 ordinary starts.

    Only the two closed scientific fit exceptions become a member non-pass.
    Generic ``ModelError`` and operational/programming exceptions propagate.
    Historical transition support, BIC, rolling scores, occupancy, raw state
    counts, and completed-episode counts are deliberately absent from the
    decision.
    """

    matrix, mask, lengths = _member_inputs(values, continuation, member=member)
    checked_pair = _pair(pair)
    scope = bridge_selection_scope(dgp, member)
    replicate = checked_pair * BRIDGE_FITS_PER_SELECTION_MEMBER
    features = bridge_feature_matrix(matrix, mask)
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=scope,
        replicate=replicate,
        n_states=OWNER_FIXED_N_STATES,
    )
    if len(seeds) != HMM_RESTARTS:
        raise AssertionError("fixed-K4 Tier-A seed registry did not return 50 starts")
    try:
        policy = owner_fixed_k4_policy_for_scientific_version(scientific_version)
        if fit_function is fit_gaussian_hmm_restarts and policy is not None:
            fit = fit_gaussian_hmm_restarts(
                features,
                OWNER_FIXED_N_STATES,
                seeds,
                owner_fixed_k4_policy=policy,
            )
        else:
            fit = fit_function(features, OWNER_FIXED_N_STATES, seeds)
    except NoNumericallyValidHMMRestarts as error:
        _validate_fixed_k4_scientific_failure(error, seeds)
        return _failed_fixed_k4_member(
            dgp=dgp,
            member=member,
            pair=checked_pair,
            scope=scope,
            replicate=replicate,
            lengths=lengths,
            failure_type=type(error).__name__,
            failure_message=_bounded(error.message),
            rejection_reason="no_numerically_valid_restart",
        )
    except HMMCandidateInadmissibility as error:
        _validate_fixed_k4_scientific_failure(error, seeds)
        if error.evidence.failure_code == "historical_transition_support":
            raise ModelError(
                "fixed-K4 Tier-A fitter elevated report-only historical "
                "transition support"
            ) from error
        return _failed_fixed_k4_member(
            dgp=dgp,
            member=member,
            pair=checked_pair,
            scope=scope,
            replicate=replicate,
            lengths=lengths,
            failure_type=type(error).__name__,
            failure_message=_bounded(error.message),
            rejection_reason="posterior_identification",
        )

    if (
        fit.n_states != OWNER_FIXED_N_STATES
        or fit.n_observations != len(features.values)
        or fit.n_features != len(MACRO_COLUMNS)
        or fit.lengths != features.lengths
        or fit.scaler_scope != features.scaler_scope
    ):
        raise ModelError("fixed-K4 Tier-A fit returned inconsistent geometry")
    numerical_chain_passed = _numerically_valid(fit)
    identification_passed = effective_identification_passed(fit)
    episode_counts = _fixed_k4_observed_episode_counts(fit)
    episode_support_passed = all(
        count >= OWNER_FIXED_MINIMUM_OBSERVED_EPISODES for count in episode_counts
    )
    reasons: list[str] = []
    if not numerical_chain_passed:
        reasons.append("numerical_or_chain_gate_failed")
    if not identification_passed:
        reasons.append("effective_identification_failed")
    if not episode_support_passed:
        reasons.append("observed_monthly_episode_support_failed")

    reference_to_candidate: tuple[int, ...] = ()
    alignment_succeeded = False
    if not reasons:
        alignment = align_refit_to_artifact(
            parent_parameters,
            fit.parameters,
            fit.scaler_means,
            fit.scaler_standard_deviations,
            decoded_states=fit.decoded_states,
        )
        raw_order = np.asarray(alignment.reference_to_candidate)
        if (
            raw_order.shape != (OWNER_FIXED_N_STATES,)
            or raw_order.dtype.kind not in {"i", "u"}
            or tuple(sorted(int(value) for value in raw_order))
            != tuple(range(OWNER_FIXED_N_STATES))
            or alignment.aligned_decoded_states is None
        ):
            raise ModelError("fixed-K4 Tier-A state alignment is inconsistent")
        reference_to_candidate = tuple(int(value) for value in raw_order)
        alignment_succeeded = True

    return BridgeFixedK4TierAMemberResult(
        dgp=dgp,
        member=member,
        pair=checked_pair,
        scope=scope,
        replicate=replicate,
        lengths=lengths,
        random_restart_ids=tuple(range(HMM_RESTARTS)),
        fit=fit,
        observed_episode_counts=episode_counts,
        reference_to_candidate=reference_to_candidate,
        numerical_chain_passed=numerical_chain_passed,
        effective_identification_passed=identification_passed,
        observed_episode_support_passed=episode_support_passed,
        alignment_succeeded=alignment_succeeded,
        rejection_reasons=tuple(reasons),
        failure_type=None,
        failure_message=None,
    )


def run_bridge_fixed_k4_tier_a_pair(
    history: BridgePseudoHistory,
    *,
    parent_parameters: HMMParameters,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
    member_runner: FixedK4MemberRunner = run_bridge_fixed_k4_tier_a_member,
) -> BridgeFixedK4TierAPairResult:
    """Run the two fixed-K4 members without cross-K or rolling selection."""

    validate_bridge_pseudo_history(
        history,
        expected_tier="selection",
        expected_n_states=OWNER_FIXED_N_STATES,
        expected_master_seed=master_seed,
        expected_scientific_version=scientific_version,
    )
    if bridge_parameters_sha256(parent_parameters) != (
        history.design_parameters_sha256
    ):
        raise ModelError("fixed-K4 Tier-A parent parameters do not match history")

    def run(member: BridgeMember) -> BridgeFixedK4TierAMemberResult:
        result = member_runner(
            history.prefix_values if member == "prefix" else history.values,
            (
                history.prefix_continuation
                if member == "prefix"
                else history.continuation
            ),
            parent_parameters=parent_parameters,
            dgp=history.dgp,
            member=member,
            pair=history.pair,
            master_seed=master_seed,
            scientific_version=scientific_version,
            fit_function=fit_function,
        )
        _fixed_k4_member_result_identity(result, history, member)
        return result

    prefix = run("prefix")
    full = run("full")
    success = prefix.passed and full.passed
    return BridgeFixedK4TierAPairResult(
        dgp=history.dgp,
        pair=history.pair,
        actual_selected_k=4,
        history_sha256=bridge_fit_input_sha256(history),
        prefix=prefix,
        full=full,
        success=success,
    )


def run_bridge_tier_a_pair(
    history: BridgePseudoHistory,
    *,
    actual_selected_k: int,
    master_seed: int,
    scientific_version: int,
    registered_bootstrap_indices: ArrayLike,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
    member_runner: MemberRunner = run_bridge_selection_member,
) -> BridgeTierAPairResult:
    """Run both ordinary selections; scientific failures are disagreements."""

    selected = _selected_k(actual_selected_k)
    validate_bridge_pseudo_history(
        history,
        expected_tier="selection",
        expected_n_states=selected,
        expected_master_seed=master_seed,
        expected_scientific_version=scientific_version,
    )
    bootstrap = _registered_bootstrap_indices(registered_bootstrap_indices)
    bootstrap_sha256 = _rolling_bootstrap_sha256(bootstrap)
    history_sha256 = bridge_fit_input_sha256(history)
    failures: list[BridgeTierAMemberFailure] = []

    def run(member: BridgeMember) -> BridgeSelectionMemberResult | None:
        values = history.prefix_values if member == "prefix" else history.values
        continuation = (
            history.prefix_continuation if member == "prefix" else history.continuation
        )
        try:
            result = member_runner(
                values,
                continuation,
                dgp=history.dgp,
                member=member,
                pair=history.pair,
                master_seed=master_seed,
                scientific_version=scientific_version,
                registered_bootstrap_indices=bootstrap,
                fit_function=fit_function,
            )
        except ModelError as error:
            failures.append(
                BridgeTierAMemberFailure(
                    member,
                    type(error).__name__,
                    _bounded(error.message),
                )
            )
            return None
        _member_result_identity(result, history, member)
        if result.rolling_bootstrap_sha256 != bootstrap_sha256:
            raise ModelError("Tier-A member used a different rolling-bootstrap trace")
        return result

    prefix = run("prefix")
    full = run("full")
    success = bool(
        prefix is not None
        and full is not None
        and prefix.selection.passed
        and full.selection.passed
        and prefix.selection.selected_n_states == selected
        and full.selection.selected_n_states == selected
    )
    return BridgeTierAPairResult(
        history.dgp,
        history.pair,
        selected,
        history_sha256,
        bootstrap_sha256,
        prefix,
        full,
        tuple(failures),
        success,
    )


def assemble_bridge_tier_a(
    results: Sequence[BridgeTierAPairResult],
    *,
    dgp: BridgeDGP,
    actual_selected_k: int,
) -> BridgeTierAExecution:
    """Reduce exact ordered pair IDs 0..249; failures stay disagreements."""

    selected = _selected_k(actual_selected_k)
    supplied = tuple(results)
    if len(supplied) != 250:
        raise ModelError("Tier A requires exactly 250 pseudo-pair results")
    bootstrap_sha256 = supplied[0].rolling_bootstrap_sha256
    if not _is_sha256(bootstrap_sha256):
        raise ModelError("Tier A rolling-bootstrap fingerprint is invalid")
    for expected, result in enumerate(supplied):
        if not isinstance(result, BridgeTierAPairResult):
            raise ModelError("Tier A pair result has the wrong record type")
        if (
            result.pair != expected
            or result.dgp != dgp
            or result.actual_selected_k != selected
            or not _is_sha256(result.history_sha256)
            or result.rolling_bootstrap_sha256 != bootstrap_sha256
        ):
            raise ModelError("Tier A pair identity/order is inconsistent")
        failed_members = tuple(failure.member for failure in result.member_failures)
        if len(set(failed_members)) != len(failed_members) or any(
            member not in {"prefix", "full"} for member in failed_members
        ):
            raise ModelError("Tier A member-failure evidence is inconsistent")
        if (result.prefix is None) != ("prefix" in failed_members) or (
            result.full is None
        ) != ("full" in failed_members):
            raise ModelError("Tier A member result/failure evidence is inconsistent")
        if (
            result.prefix is not None
            and result.prefix.rolling_bootstrap_sha256 != bootstrap_sha256
        ) or (
            result.full is not None
            and result.full.rolling_bootstrap_sha256 != bootstrap_sha256
        ):
            raise ModelError("Tier A member rolling-bootstrap trace is inconsistent")
        expected_success = bool(
            result.prefix is not None
            and result.full is not None
            and result.prefix.selection.passed
            and result.full.selection.passed
            and result.prefix.selection.selected_n_states == selected
            and result.full.selection.selected_n_states == selected
        )
        if result.success != expected_success:
            raise ModelError(
                "Tier A success flag is inconsistent with member selection"
            )
    outcomes = np.asarray([result.success for result in supplied], dtype=np.bool_)
    gate = tier_a_selection_gate(outcomes)
    return BridgeTierAExecution(dgp, selected, supplied, gate)


def assemble_bridge_fixed_k4_tier_a(
    results: Sequence[BridgeFixedK4TierAPairResult],
    *,
    dgp: BridgeDGP,
) -> BridgeFixedK4TierAExecution:
    """Reduce exact ordered fixed-K4 pair IDs 0..249."""

    supplied = tuple(results)
    if len(supplied) != 250:
        raise ModelError("fixed-K4 Tier A requires exactly 250 pseudo-pair results")
    for expected, result in enumerate(supplied):
        if (
            not isinstance(result, BridgeFixedK4TierAPairResult)
            or result.pair != expected
            or result.dgp != dgp
            or result.actual_selected_k != OWNER_FIXED_N_STATES
            or not _is_sha256(result.history_sha256)
        ):
            raise ModelError("fixed-K4 Tier-A pair identity/order is inconsistent")
        if result.prefix.member != "prefix" or result.full.member != "full":
            raise ModelError("fixed-K4 Tier-A member order is inconsistent")
        if result.success != (result.prefix.passed and result.full.passed):
            raise ModelError("fixed-K4 Tier-A success flag is inconsistent")
    outcomes = np.asarray([result.success for result in supplied], dtype=np.bool_)
    return BridgeFixedK4TierAExecution(
        dgp=dgp,
        actual_selected_k=4,
        pairs=supplied,
        gate=tier_a_selection_gate(outcomes),
    )


def bridge_feature_matrix(
    values: ArrayLike,
    continuation: ArrayLike,
    *,
    training_rows: int | None = None,
) -> HMMFeatureMatrix:
    """Population-standardize all rows or an exact leading training subset."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = _mask(continuation, len(matrix))
    if matrix.ndim != 2 or matrix.shape[1] != 4 or not np.isfinite(matrix).all():
        raise ModelError("bridge feature values must be a finite four-column matrix")
    rows = len(matrix)
    if training_rows is None:
        selected_rows = rows
        scaler_scope: Literal["design_training", "production_full"] = "production_full"
    else:
        if (
            isinstance(training_rows, bool)
            or not isinstance(training_rows, int)
            or not 1 <= training_rows < rows
        ):
            raise ModelError("bridge training rows must identify a proper prefix")
        selected_rows = training_rows
        scaler_scope = "design_training"
    scaler_source = matrix[:selected_rows]
    means = scaler_source.mean(axis=0)
    scales = scaler_source.std(axis=0, ddof=0)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or bool((scales <= 0.0).any())
    ):
        raise ModelError("bridge feature scaler is non-finite or degenerate")
    standardized = (matrix[:selected_rows] - means) / scales
    selected_mask = mask[:selected_rows].astype(np.bool_, copy=True)
    selected_mask[0] = False
    result_values = standardized.astype(np.float64, copy=True)
    result_means = means.astype(np.float64, copy=True)
    result_scales = scales.astype(np.float64, copy=True)
    for array in (result_values, result_means, result_scales):
        array.setflags(write=False)
    return HMMFeatureMatrix(
        values=result_values,
        lengths=sequence_lengths_from_continuation(selected_mask),
        feature_names=tuple(MACRO_COLUMNS),
        row_labels=tuple(f"bridge:{row}" for row in range(selected_rows)),
        scaler_means=result_means,
        scaler_standard_deviations=result_scales,
        scaler_scope=scaler_scope,
    )


def _rolling_attempts(
    values: FloatArray,
    continuation: BoolArray,
    *,
    n_states: int,
    scope: ModelFitScope,
    pair: int,
    master_seed: int,
    scientific_version: int,
    fit_function: FitFunction,
    main_fit: GaussianHMMFit | None,
) -> tuple[BridgeRollingAttempt, ...]:
    folds = canonical_rolling_folds(continuation)
    attempts: list[BridgeRollingAttempt] = []
    for fold in folds:
        replicate = pair * BRIDGE_FITS_PER_SELECTION_MEMBER + fold.fold + 1
        if main_fit is None:
            attempts.append(
                _failed_fold(
                    fold,
                    scope=scope,
                    replicate=replicate,
                    failure_type="MainFitFailure",
                    failure_message="candidate main fit failed",
                )
            )
            continue
        try:
            features = bridge_feature_matrix(
                values,
                continuation,
                training_rows=fold.training_rows,
            )
            seeds = deterministic_hmm_restart_seeds(
                master_seed=master_seed,
                scientific_version=scientific_version,
                scope=scope,
                replicate=replicate,
                n_states=n_states,
            )
            fitted = fit_function(features, n_states, seeds)
            validation = values[fold.validation]
            transformed = (validation - np.asarray(features.scaler_means)) / np.asarray(
                features.scaler_standard_deviations
            )
            initial = np.tile(
                np.asarray(fitted.chain_diagnostics.stationary_distribution),
                (len(fold.validation_lengths), 1),
            ).astype(np.float64)
            start = int(fold.validation.start or 0)
            if start > 0 and bool(continuation[start]):
                terminal = np.asarray(
                    fitted.forward_backward.filtered_probabilities[-1],
                    dtype=np.float64,
                )
                initial[0] = terminal @ fitted.parameters.transition_matrix
            score = one_step_predictive_log_scores(
                transformed,
                fitted.parameters,
                lengths=fold.validation_lengths,
                initial_state_probabilities=initial,
            )
            scores = tuple(float(value) for value in score.log_scores)
            if len(scores) != 12 or not all(math.isfinite(value) for value in scores):
                raise ModelError("bridge rolling fold did not produce 12 finite scores")
            attempts.append(
                BridgeRollingAttempt(
                    fold.fold,
                    scope,
                    replicate,
                    fold.training_rows,
                    fold.validation_rows,
                    fold.validation_lengths,
                    fitted,
                    scores,
                    None,
                    None,
                )
            )
        except ModelError as error:
            attempts.append(
                _failed_fold(
                    fold,
                    scope=scope,
                    replicate=replicate,
                    failure_type=type(error).__name__,
                    failure_message=_bounded(error.message),
                )
            )
    return tuple(attempts)


def _bridge_viability(attempt: CandidateFitAttempt) -> CandidateViabilityRecord:
    fit = attempt.fit
    checks = (
        PreselectionCheck("fit_succeeded", fit is not None),
        PreselectionCheck("numerical_chain", _numerically_valid(fit)),
        PreselectionCheck(
            "posterior_identification",
            bool(fit is not None and effective_identification_passed(fit)),
        ),
        PreselectionCheck(
            "historical_transition_support",
            bool(fit is not None and effective_transition_support_passed(fit)),
        ),
    )
    states: tuple[StateViabilityEvidence, ...] = ()
    if fit is not None:
        path = fit.state_path_diagnostics
        condition = float(fit.covariance_diagnostic.condition_number)
        states = tuple(
            StateViabilityEvidence(
                state=state,
                monthly_observations=int(path.state_counts[state]),
                occupancy=float(path.occupancy[state]),
                completed_monthly_episodes=int(path.completed_episode_counts[state]),
                daily_observations=1,
                daily_runs=0,
                covariance_condition_number=condition,
            )
            for state in range(attempt.n_states)
        )
    thresholds = ViabilityThresholds(
        minimum_monthly_observations=MINIMUM_MONTHLY_OBSERVATIONS,
        minimum_occupancy=MINIMUM_STATE_OCCUPANCY,
        minimum_completed_monthly_episodes=MINIMUM_COMPLETED_MONTHLY_EPISODES,
        minimum_daily_observations=1,
        minimum_daily_runs=0,
        maximum_covariance_condition_number=MAXIMUM_TIED_COVARIANCE_CONDITION,
        minimum_bootstrap_median_ari=-1.0,
        minimum_rolling_median_ari=-1.0,
    )
    evidence = CandidateViabilityEvidence(
        attempt.n_states,
        None if fit is None else float(fit.bic),
        states,
        0.0 if fit is not None else None,
        0.0 if fit is not None else None,
        checks,
    )
    return evaluate_candidate_viability(evidence, thresholds=thresholds)


def _numerically_valid(fit: GaussianHMMFit | None) -> bool:
    if fit is None:
        return False
    covariance = fit.covariance_diagnostic
    chain = fit.chain_diagnostics
    return bool(
        covariance.symmetric
        and covariance.positive_definite
        and covariance.eigenvalue_floor_passed
        and covariance.condition_number_passed
        and math.isfinite(float(covariance.condition_number))
        and float(covariance.condition_number) <= MAXIMUM_TIED_COVARIANCE_CONDITION
        and chain.irreducible
        and chain.aperiodic
        and chain.unique_stationary_distribution
        and chain.period == 1
        and chain.stationary_residual <= 1e-12
        and not chain.near_absorbing_states
        and chain.mixing_time_to_one_percent <= 120
    )


def _failed_fold(
    fold: RollingFold,
    *,
    scope: ModelFitScope,
    replicate: int,
    failure_type: str,
    failure_message: str,
) -> BridgeRollingAttempt:
    return BridgeRollingAttempt(
        fold.fold,
        scope,
        replicate,
        fold.training_rows,
        fold.validation_rows,
        fold.validation_lengths,
        None,
        (None,) * 12,
        failure_type,
        failure_message,
    )


def _member_inputs(
    values: ArrayLike,
    continuation: ArrayLike,
    *,
    member: BridgeMember,
) -> tuple[FloatArray, BoolArray, tuple[int, ...]]:
    matrix = np.asarray(values, dtype=np.float64)
    expected_lengths: tuple[int, ...]
    if member == "prefix":
        expected_rows = BRIDGE_PREFIX_ROWS
        expected_lengths = (BRIDGE_PREFIX_ROWS,)
    elif member == "full":
        expected_rows = BRIDGE_FULL_ROWS
        expected_lengths = (BRIDGE_GAP_ROW, BRIDGE_FULL_ROWS - BRIDGE_GAP_ROW)
    else:
        raise ModelError("bridge selection member must be prefix or full")
    if matrix.shape != (expected_rows, 4) or not np.isfinite(matrix).all():
        raise ModelError("bridge selection member has the wrong finite geometry")
    mask = _mask(continuation, expected_rows)
    lengths = sequence_lengths_from_continuation(mask)
    if lengths != expected_lengths:
        raise ModelError(
            "bridge selection continuation has the wrong segment geometry",
            details={"actual": list(lengths), "expected": list(expected_lengths)},
        )
    return matrix, mask, lengths


def _mask(values: ArrayLike, rows: int) -> BoolArray:
    result = np.asarray(values)
    if result.shape != (rows,) or result.dtype.kind != "b" or bool(result[0]):
        raise ModelError("bridge continuation must be a false-at-start Boolean mask")
    return result.astype(np.bool_, copy=False)


def _pair(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 250:
        raise ModelError("Tier-A pair must be an integer in [0, 249]")
    return value


def _selected_k(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (VERSION_1_CANDIDATES)
    ):
        raise ModelError("actual bridge selected K must be one of 2,3,4,5")
    return value


def _registered_bootstrap_indices(values: ArrayLike) -> NDArray[np.int64]:
    result = np.asarray(values)
    if (
        result.shape != (10_000, 6, 12)
        or result.dtype.kind not in {"i", "u"}
        or bool(((result < 0) | (result >= 12)).any())
    ):
        raise ModelError(
            "Tier-A registered bootstrap indices must have exact shape "
            "(10000, 6, 12) with entries in [0, 11]"
        )
    return result.astype(np.int64, copy=False)


def _member_result_identity(
    result: BridgeSelectionMemberResult,
    history: BridgePseudoHistory,
    member: BridgeMember,
) -> None:
    expected_scope = bridge_selection_scope(history.dgp, member)
    expected_replicate = history.pair * BRIDGE_FITS_PER_SELECTION_MEMBER
    expected_continuation = (
        history.prefix_continuation if member == "prefix" else history.continuation
    )
    expected_folds = canonical_rolling_folds(expected_continuation)
    expected_lengths = (
        (BRIDGE_PREFIX_ROWS,)
        if member == "prefix"
        else (BRIDGE_GAP_ROW, BRIDGE_FULL_ROWS - BRIDGE_GAP_ROW)
    )
    if (
        not isinstance(result, BridgeSelectionMemberResult)
        or result.dgp != history.dgp
        or result.member != member
        or result.pair != history.pair
        or result.scope is not expected_scope
        or result.main_replicate != expected_replicate
        or result.lengths != expected_lengths
        or not _is_sha256(result.rolling_bootstrap_sha256)
        or result.fit_set.scope is not expected_scope
        or result.fit_set.replicate != expected_replicate
        or tuple(candidate.n_states for candidate in result.candidates)
        != VERSION_1_CANDIDATES
    ):
        raise ModelError("Tier-A member runner returned inconsistent identity")
    for expected_k, candidate in zip(
        VERSION_1_CANDIDATES,
        result.candidates,
        strict=True,
    ):
        if (
            candidate.main_attempt.n_states != expected_k
            or candidate.viability.evidence.n_states != expected_k
            or len(candidate.rolling_attempts) != len(expected_folds)
        ):
            raise ModelError("Tier-A candidate evidence identity is inconsistent")
        for fold, (expected_fold, attempt) in enumerate(
            zip(expected_folds, candidate.rolling_attempts, strict=True)
        ):
            if (
                attempt.fold != fold
                or attempt.scope is not expected_scope
                or attempt.replicate != expected_replicate + fold + 1
                or attempt.training_rows != expected_fold.training_rows
                or attempt.validation_rows != expected_fold.validation_rows
                or attempt.validation_lengths != expected_fold.validation_lengths
            ):
                raise ModelError("Tier-A rolling-fold identity is inconsistent")


def _fixed_k4_member_result_identity(
    result: BridgeFixedK4TierAMemberResult,
    history: BridgePseudoHistory,
    member: BridgeMember,
) -> None:
    expected_scope = bridge_selection_scope(history.dgp, member)
    expected_lengths = (
        (BRIDGE_PREFIX_ROWS,)
        if member == "prefix"
        else (BRIDGE_GAP_ROW, BRIDGE_FULL_ROWS - BRIDGE_GAP_ROW)
    )
    if (
        not isinstance(result, BridgeFixedK4TierAMemberResult)
        or result.dgp != history.dgp
        or result.member != member
        or result.pair != history.pair
        or result.scope is not expected_scope
        or result.replicate != history.pair * BRIDGE_FITS_PER_SELECTION_MEMBER
        or result.lengths != expected_lengths
        or result.random_restart_ids != tuple(range(HMM_RESTARTS))
    ):
        raise ModelError("fixed-K4 Tier-A member returned inconsistent identity")
    if result.fit is not None and result.fit.n_states != OWNER_FIXED_N_STATES:
        raise ModelError("fixed-K4 Tier-A member returned a non-K4 fit")
    if result.passed and (
        len(result.observed_episode_counts) != OWNER_FIXED_N_STATES
        or any(
            count < OWNER_FIXED_MINIMUM_OBSERVED_EPISODES
            for count in result.observed_episode_counts
        )
        or tuple(sorted(result.reference_to_candidate))
        != tuple(range(OWNER_FIXED_N_STATES))
    ):
        raise ModelError("fixed-K4 Tier-A passing evidence is inconsistent")


def _validate_fixed_k4_scientific_failure(
    error: NoNumericallyValidHMMRestarts | HMMCandidateInadmissibility,
    seeds: Sequence[int],
) -> None:
    diagnostics = (
        error.restart_diagnostics
        if isinstance(error, NoNumericallyValidHMMRestarts)
        else error.evidence.restart_diagnostics
    )
    n_states = (
        error.n_states
        if isinstance(error, NoNumericallyValidHMMRestarts)
        else error.evidence.n_states
    )
    if (
        n_states != OWNER_FIXED_N_STATES
        or len(diagnostics) != HMM_RESTARTS
        or tuple(item.seed for item in diagnostics) != tuple(seeds)
    ):
        raise ModelError(
            "fixed-K4 Tier-A scientific-failure evidence is inconsistent"
        ) from error


def _failed_fixed_k4_member(
    *,
    dgp: BridgeDGP,
    member: BridgeMember,
    pair: int,
    scope: ModelFitScope,
    replicate: int,
    lengths: tuple[int, ...],
    failure_type: str,
    failure_message: str,
    rejection_reason: str,
) -> BridgeFixedK4TierAMemberResult:
    return BridgeFixedK4TierAMemberResult(
        dgp=dgp,
        member=member,
        pair=pair,
        scope=scope,
        replicate=replicate,
        lengths=lengths,
        random_restart_ids=tuple(range(HMM_RESTARTS)),
        fit=None,
        observed_episode_counts=(),
        reference_to_candidate=(),
        numerical_chain_passed=False,
        effective_identification_passed=False,
        observed_episode_support_passed=False,
        alignment_succeeded=False,
        rejection_reasons=(rejection_reason,),
        failure_type=failure_type,
        failure_message=failure_message,
    )


def _fixed_k4_observed_episode_counts(
    fit: GaussianHMMFit,
) -> tuple[int, ...]:
    raw = np.asarray(fit.state_path_diagnostics.episode_counts)
    if (
        raw.shape != (OWNER_FIXED_N_STATES,)
        or raw.dtype.kind not in {"i", "u"}
        or bool((raw < 0).any())
    ):
        raise ModelError("fixed-K4 Tier-A observed episode evidence is malformed")
    return tuple(int(value) for value in raw)


def _bounded(value: str, limit: int = 500) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _rolling_bootstrap_sha256(values: NDArray[np.int64]) -> str:
    canonical = np.ascontiguousarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"PRPG-BRIDGE-ROLLING-BOOTSTRAP-v1\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
