"""Pure, auditable HMM-candidate selection and holdout-veto records.

This module deliberately contains no fitted model objects, implicit/global
random state, or file I/O.  The frozen rolling constants are explicit below;
trace generation accepts only a caller-owned registered ``Generator`` and the
selector accepts materialized common indices.  The three ordered decisions in
design Section 8.2 remain separate:

1. evaluate pre-BIC candidate viability;
2. apply the common-trace all-pairs rolling max-z selector and BIC guard; and
3. apply the sealed holdout as a veto to the already selected candidate.

That separation makes it impossible for a failed holdout to promote a
runner-up.  Frozen dataclasses retain all evidence and deterministic rejection
codes for later JSON report serialization.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError

Number: TypeAlias = int | float
RequiredRelation: TypeAlias = Literal["passed", "finite", ">=", "<="]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

ROLLING_FOLD_COUNT = 6
ROLLING_FOLD_SIZE = 12
ROLLING_SCORE_COUNT = ROLLING_FOLD_COUNT * ROLLING_FOLD_SIZE
ROLLING_BOOTSTRAP_REPLICATES = 10_000
ROLLING_ADVANCE_PROBABILITY = 0.75
ROLLING_RESTART_PROBABILITIES = (4.0 / 15.0,) + (1.0 / 15.0,) * 11
ROLLING_MAX_Z_CONFIDENCE = 0.95
ROLLING_NUMERICAL_TIE_TOLERANCE = 1e-12
ROLLING_ZERO_SCALE_TOLERANCE = 1e-12
BIC_DELTA_MAXIMUM = 10.0
_ALLOWED_STATE_COUNTS = frozenset({2, 3, 4, 5})
ORDINARY_SELECTION_POLICY: Literal["predictive_bic_v1"] = "predictive_bic_v1"
OWNER_FIXED_K4_SELECTION_POLICY: Literal["owner_fixed_k4_v3"] = "owner_fixed_k4_v3"
OWNER_FIXED_N_STATES: Literal[4] = 4

_CHECK_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class PreselectionCheck:
    """One caller-owned hard check that must pass before BIC comparison."""

    check_id: str
    passed: bool


@dataclass(frozen=True, slots=True)
class StateViabilityEvidence:
    """Support and covariance evidence for one canonical candidate state."""

    state: int
    monthly_observations: int
    occupancy: float
    completed_monthly_episodes: int
    daily_observations: int
    daily_runs: int
    covariance_condition_number: float


@dataclass(frozen=True, slots=True)
class CandidateViabilityEvidence:
    """All candidate-level evidence consumed by the pre-BIC filter.

    ``bic`` and stability summaries may be absent when an upstream fit/check
    failed.  Their absence becomes a retained rejection reason rather than
    erasing the failed candidate from the report.
    """

    n_states: int
    bic: float | None
    states: tuple[StateViabilityEvidence, ...]
    bootstrap_median_ari: float | None
    rolling_median_ari: float | None
    preselection_checks: tuple[PreselectionCheck, ...]


@dataclass(frozen=True, slots=True)
class ViabilityThresholds:
    """Explicit caller-supplied pre-BIC thresholds; there are no defaults."""

    minimum_monthly_observations: int
    minimum_occupancy: float
    minimum_completed_monthly_episodes: int
    minimum_daily_observations: int
    minimum_daily_runs: int
    maximum_covariance_condition_number: float
    minimum_bootstrap_median_ari: float
    minimum_rolling_median_ari: float


@dataclass(frozen=True, slots=True)
class RejectionReason:
    """Stable machine-readable explanation for one failed viability check."""

    code: str
    required_relation: RequiredRelation
    state: int | None = None
    check_id: str | None = None
    observed: Number | None = None
    threshold: Number | None = None

    def as_dict(self) -> dict[str, str | int | float | None]:
        """Return a primitive, insertion-order-stable audit representation."""

        return {
            "code": self.code,
            "required_relation": self.required_relation,
            "state": self.state,
            "check_id": self.check_id,
            "observed": _json_number(self.observed),
            "threshold": _json_number(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class CandidateViabilityRecord:
    """Normalized evidence and every deterministic pre-BIC rejection."""

    evidence: CandidateViabilityEvidence
    thresholds: ViabilityThresholds
    rejection_reasons: tuple[RejectionReason, ...]

    @property
    def admissible(self) -> bool:
        """Whether the candidate is eligible for BIC comparison."""

        return not self.rejection_reasons

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready candidate report without model objects."""

        evidence = self.evidence
        thresholds = self.thresholds
        return {
            "n_states": evidence.n_states,
            "bic": _json_number(evidence.bic),
            "bootstrap_median_ari": _json_number(evidence.bootstrap_median_ari),
            "rolling_median_ari": _json_number(evidence.rolling_median_ari),
            "states": [
                {
                    "state": state.state,
                    "monthly_observations": state.monthly_observations,
                    "occupancy": state.occupancy,
                    "completed_monthly_episodes": (state.completed_monthly_episodes),
                    "daily_observations": state.daily_observations,
                    "daily_runs": state.daily_runs,
                    "covariance_condition_number": _json_number(
                        state.covariance_condition_number
                    ),
                }
                for state in evidence.states
            ],
            "preselection_checks": [
                {"check_id": check.check_id, "passed": check.passed}
                for check in evidence.preselection_checks
            ],
            "thresholds": {
                "minimum_monthly_observations": (
                    thresholds.minimum_monthly_observations
                ),
                "minimum_occupancy": thresholds.minimum_occupancy,
                "minimum_completed_monthly_episodes": (
                    thresholds.minimum_completed_monthly_episodes
                ),
                "minimum_daily_observations": thresholds.minimum_daily_observations,
                "minimum_daily_runs": thresholds.minimum_daily_runs,
                "maximum_covariance_condition_number": (
                    thresholds.maximum_covariance_condition_number
                ),
                "minimum_bootstrap_median_ari": (
                    thresholds.minimum_bootstrap_median_ari
                ),
                "minimum_rolling_median_ari": thresholds.minimum_rolling_median_ari,
            },
            "admissible": self.admissible,
            "rejection_reasons": [
                reason.as_dict() for reason in self.rejection_reasons
            ],
        }


@dataclass(frozen=True, slots=True)
class RollingPairwiseInterval:
    """One canonical smaller-K-minus-larger-K simultaneous interval."""

    first_n_states: int
    second_n_states: int
    observed_mean_difference: float
    bootstrap_standard_error: float
    interval_lower: float
    interval_upper: float
    zero_scale: bool
    exact_tie: bool
    contains_zero: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready pairwise interval."""

        return {
            "first_n_states": self.first_n_states,
            "second_n_states": self.second_n_states,
            "observed_mean_difference": self.observed_mean_difference,
            "bootstrap_standard_error": self.bootstrap_standard_error,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
            "zero_scale": self.zero_scale,
            "exact_tie": self.exact_tie,
            "contains_zero": self.contains_zero,
        }


@dataclass(frozen=True, slots=True)
class RollingMaxZSummary:
    """The complete fixed-size all-pairs common-trace max-z family."""

    candidate_state_counts: tuple[int, ...]
    candidate_means: tuple[float, ...]
    pairwise_intervals: tuple[RollingPairwiseInterval, ...]
    bootstrap_replicates: int
    confidence: float
    critical_value: float

    def as_dict(self) -> dict[str, object]:
        """Return the max-z family as JSON-ready primitives."""

        return {
            "candidate_state_counts": list(self.candidate_state_counts),
            "candidate_means": list(self.candidate_means),
            "pairwise_intervals": [
                interval.as_dict() for interval in self.pairwise_intervals
            ],
            "bootstrap_replicates": self.bootstrap_replicates,
            "confidence": self.confidence,
            "critical_value": self.critical_value,
        }


@dataclass(frozen=True, slots=True)
class CandidateSelectionInput:
    """One viability record and 72 fold-major rolling monthly scores.

    A predictively invalid fold is represented by twelve ``None`` values.  A
    partially missing fold is invalid input rather than an invalid fitted fold.
    """

    viability: CandidateViabilityRecord
    rolling_scores: tuple[float | None, ...]


SelectionDisposition: TypeAlias = Literal[
    "selected",
    "inadmissible",
    "report_only_nonfixed_k",
    "predictively_ineligible",
    "outside_simultaneous_near_best",
    "outside_bic_guard",
    "larger_state_count_in_intersection",
]
SelectionPolicy: TypeAlias = Literal["predictive_bic_v1", "owner_fixed_k4_v3"]


@dataclass(frozen=True, slots=True)
class CandidateRankingRecord:
    """One normalized candidate row in the deterministic selection report."""

    n_states: int
    admissible: bool
    bic: float | None
    bic_rank: int | None
    delta_bic: float | None
    predictively_eligible: bool
    invalid_rolling_folds: tuple[int, ...]
    rolling_mean: float | None
    candidate_minus_best_mean: float | None
    candidate_minus_best_lower: float | None
    candidate_minus_best_upper: float | None
    predictively_near_best: bool
    inside_bic_guard: bool
    selected: bool
    disposition: SelectionDisposition
    rejection_reasons: tuple[RejectionReason, ...]
    viability: CandidateViabilityRecord

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready ranking row."""

        return {
            "n_states": self.n_states,
            "admissible": self.admissible,
            "bic": _json_number(self.bic),
            "bic_rank": self.bic_rank,
            "delta_bic": self.delta_bic,
            "predictively_eligible": self.predictively_eligible,
            "invalid_rolling_folds": list(self.invalid_rolling_folds),
            "rolling_mean": self.rolling_mean,
            "candidate_minus_best_mean": self.candidate_minus_best_mean,
            "candidate_minus_best_lower": self.candidate_minus_best_lower,
            "candidate_minus_best_upper": self.candidate_minus_best_upper,
            "predictively_near_best": self.predictively_near_best,
            "inside_bic_guard": self.inside_bic_guard,
            "selected": self.selected,
            "disposition": self.disposition,
            "rejection_reasons": [
                reason.as_dict() for reason in self.rejection_reasons
            ],
            "viability": self.viability.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class CandidateSelectionReport:
    """Complete all-pairs predictive/BIC result, including failed candidates."""

    selection_policy: SelectionPolicy
    candidates: tuple[CandidateRankingRecord, ...]
    rolling_max_z: RollingMaxZSummary | None
    minimum_bic_n_states: int | None
    predictive_best_n_states: int | None
    selected_n_states: int | None
    bic_delta_maximum: float
    rolling_confidence: float
    rolling_fold_count: int
    rolling_score_count: int
    bootstrap_replicates: int
    max_z_critical_value: float
    passed: bool
    failure_reason: str | None

    def as_dict(self) -> dict[str, object]:
        """Return the complete selection report as JSON-ready primitives."""

        return {
            "selection_policy": self.selection_policy,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "rolling_max_z": (
                None if self.rolling_max_z is None else self.rolling_max_z.as_dict()
            ),
            "minimum_bic_n_states": self.minimum_bic_n_states,
            "predictive_best_n_states": self.predictive_best_n_states,
            "selected_n_states": self.selected_n_states,
            "bic_delta_maximum": self.bic_delta_maximum,
            "rolling_confidence": self.rolling_confidence,
            "rolling_fold_count": self.rolling_fold_count,
            "rolling_score_count": self.rolling_score_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "max_z_critical_value": self.max_z_critical_value,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class HoldoutVetoSummary:
    """Selection-independent predictive-score veto for a sealed holdout."""

    hmm_log_score: float
    baseline_log_score: float
    sequence_lengths: tuple[int, ...]
    observations: int
    score_difference: float
    difference_per_observation: float
    minimum_difference_per_observation: float
    passed: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready holdout evidence."""

        return {
            "hmm_log_score": self.hmm_log_score,
            "baseline_log_score": self.baseline_log_score,
            "sequence_lengths": list(self.sequence_lengths),
            "observations": self.observations,
            "score_difference": self.score_difference,
            "difference_per_observation": self.difference_per_observation,
            "minimum_difference_per_observation": (
                self.minimum_difference_per_observation
            ),
            "passed": self.passed,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class HoldoutQualification:
    """Holdout result that retains selection but withholds authorization."""

    selected_n_states: int
    authorized_n_states: int | None
    passed: bool
    holdout: HoldoutVetoSummary

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready qualification evidence."""

        return {
            "selected_n_states": self.selected_n_states,
            "authorized_n_states": self.authorized_n_states,
            "passed": self.passed,
            "holdout": self.holdout.as_dict(),
        }


def evaluate_candidate_viability(
    evidence: CandidateViabilityEvidence,
    *,
    thresholds: ViabilityThresholds,
) -> CandidateViabilityRecord:
    """Apply caller-supplied pre-BIC thresholds in a deterministic order."""

    _positive_integer(evidence.n_states, "n_states")
    _validate_thresholds(thresholds)
    checks = _normalize_checks(evidence.preselection_checks)
    states = _normalize_states(evidence.states, evidence.n_states)
    normalized = CandidateViabilityEvidence(
        n_states=evidence.n_states,
        bic=evidence.bic,
        states=states,
        bootstrap_median_ari=evidence.bootstrap_median_ari,
        rolling_median_ari=evidence.rolling_median_ari,
        preselection_checks=checks,
    )
    reasons: list[RejectionReason] = []
    for check in checks:
        if not check.passed:
            reasons.append(
                RejectionReason(
                    code="preselection_check_failed",
                    required_relation="passed",
                    check_id=check.check_id,
                )
            )

    _append_optional_finite_reason(reasons, "bic", evidence.bic)
    by_state = {state.state: state for state in states}
    for state_id in range(evidence.n_states):
        state = by_state.get(state_id)
        if state is None:
            reasons.append(
                RejectionReason(
                    code="state_evidence_missing",
                    required_relation="passed",
                    state=state_id,
                )
            )
            continue
        _append_minimum_reason(
            reasons,
            code="monthly_observations_below_minimum",
            state=state_id,
            observed=state.monthly_observations,
            threshold=thresholds.minimum_monthly_observations,
        )
        _append_minimum_reason(
            reasons,
            code="occupancy_below_minimum",
            state=state_id,
            observed=state.occupancy,
            threshold=thresholds.minimum_occupancy,
        )
        _append_minimum_reason(
            reasons,
            code="completed_monthly_episodes_below_minimum",
            state=state_id,
            observed=state.completed_monthly_episodes,
            threshold=thresholds.minimum_completed_monthly_episodes,
        )
        _append_minimum_reason(
            reasons,
            code="daily_observations_below_minimum",
            state=state_id,
            observed=state.daily_observations,
            threshold=thresholds.minimum_daily_observations,
        )
        _append_minimum_reason(
            reasons,
            code="daily_runs_below_minimum",
            state=state_id,
            observed=state.daily_runs,
            threshold=thresholds.minimum_daily_runs,
        )
        if not math.isfinite(state.covariance_condition_number):
            reasons.append(
                RejectionReason(
                    code="covariance_condition_number_nonfinite",
                    required_relation="finite",
                    state=state_id,
                    observed=state.covariance_condition_number,
                )
            )
        elif (
            state.covariance_condition_number
            > thresholds.maximum_covariance_condition_number
        ):
            reasons.append(
                RejectionReason(
                    code="covariance_condition_number_above_maximum",
                    required_relation="<=",
                    state=state_id,
                    observed=state.covariance_condition_number,
                    threshold=thresholds.maximum_covariance_condition_number,
                )
            )

    _append_ari_reason(
        reasons,
        name="bootstrap_median_ari",
        observed=evidence.bootstrap_median_ari,
        threshold=thresholds.minimum_bootstrap_median_ari,
    )
    _append_ari_reason(
        reasons,
        name="rolling_median_ari",
        observed=evidence.rolling_median_ari,
        threshold=thresholds.minimum_rolling_median_ari,
    )
    return CandidateViabilityRecord(normalized, thresholds, tuple(reasons))


def rolling_score_fold_trace_from_uniforms(
    first_uniform: float,
    advance_uniforms: ArrayLike,
    restart_uniforms: ArrayLike,
) -> IntArray:
    """Build one frozen 12-position boundary-corrected bootstrap trace.

    Both eleven-element transition vectors are required even when a forced
    restart makes an advance uniform irrelevant.  This fixed-consumption
    interface keeps registered RNG coordinates independent of branch outcomes.
    """

    first = _unit_uniform_scalar(first_uniform, "first_uniform")
    advances = _unit_uniform_vector(
        advance_uniforms,
        expected_length=ROLLING_FOLD_SIZE - 1,
        label="advance_uniforms",
    )
    restarts = _unit_uniform_vector(
        restart_uniforms,
        expected_length=ROLLING_FOLD_SIZE - 1,
        label="restart_uniforms",
    )
    trace = np.empty(ROLLING_FOLD_SIZE, dtype=np.int64)
    trace[0] = math.floor(ROLLING_FOLD_SIZE * first)
    for position in range(1, ROLLING_FOLD_SIZE):
        current = int(trace[position - 1])
        if current < ROLLING_FOLD_SIZE - 1 and (
            advances[position - 1] < ROLLING_ADVANCE_PROBABILITY
        ):
            trace[position] = current + 1
        else:
            restart_bin = math.floor(15.0 * restarts[position - 1])
            trace[position] = 0 if restart_bin <= 3 else restart_bin - 3
    trace.setflags(write=False)
    return trace


def rolling_score_bootstrap_trace(rng: np.random.Generator) -> IntArray:
    """Generate the six common fold-local traces for one registered replicate."""

    if not isinstance(rng, np.random.Generator):
        raise ModelError("rolling bootstrap RNG must be a numpy Generator")
    first_uniforms = rng.random(ROLLING_FOLD_COUNT)
    advance_uniforms = rng.random((ROLLING_FOLD_COUNT, ROLLING_FOLD_SIZE - 1))
    restart_uniforms = rng.random((ROLLING_FOLD_COUNT, ROLLING_FOLD_SIZE - 1))
    traces = np.empty((ROLLING_FOLD_COUNT, ROLLING_FOLD_SIZE), dtype=np.int64)
    for fold in range(ROLLING_FOLD_COUNT):
        traces[fold] = rolling_score_fold_trace_from_uniforms(
            float(first_uniforms[fold]),
            advance_uniforms[fold],
            restart_uniforms[fold],
        )
    traces.setflags(write=False)
    return traces


def simultaneous_rolling_score_intervals(
    candidate_state_counts: Sequence[int],
    rolling_scores: ArrayLike,
    bootstrap_indices: ArrayLike,
) -> RollingMaxZSummary:
    """Form the binding all-pairs common-trace simultaneous max-z family.

    ``rolling_scores`` has shape ``(candidate, 6, 12)`` and
    ``bootstrap_indices`` has the exact shape ``(10_000, 6, 12)``.  Pair
    orientation in the returned family is always smaller K minus larger K.
    """

    state_counts = _selection_state_counts(candidate_state_counts, minimum=2)
    scores = _finite_rolling_score_cube(
        rolling_scores,
        expected_candidates=len(state_counts),
    )
    indices = _rolling_bootstrap_indices(bootstrap_indices)
    order = np.argsort(np.asarray(state_counts, dtype=np.int64))
    ordered_counts = tuple(state_counts[int(index)] for index in order)
    scores = scores[order]
    with np.errstate(over="ignore", invalid="ignore"):
        means_array = scores.mean(axis=(1, 2))
    if not np.isfinite(means_array).all():
        raise ModelError("rolling-score candidate means are non-finite")

    # The complete unordered family is fixed here, before any caller can name
    # an observed predictive best.
    pair_positions = tuple(combinations(range(len(ordered_counts)), 2))
    fold_positions = np.arange(ROLLING_FOLD_COUNT, dtype=np.int64)[None, :, None]
    intermediate: list[tuple[int, int, float, float, bool]] = []
    max_z = np.zeros(ROLLING_BOOTSTRAP_REPLICATES, dtype=np.float64)
    any_variable_pair = False
    for first_position, second_position in pair_positions:
        with np.errstate(over="ignore", invalid="ignore"):
            differences = scores[first_position] - scores[second_position]
            observed_mean = float(differences.mean())
            bootstrap_means = differences[fold_positions, indices].mean(axis=(1, 2))
            standard_error = float(bootstrap_means.std(ddof=1))
        if (
            not np.isfinite(differences).all()
            or not math.isfinite(observed_mean)
            or not np.isfinite(bootstrap_means).all()
            or not math.isfinite(standard_error)
        ):
            raise ModelError("rolling-score bootstrap arithmetic is non-finite")
        zero_scale = standard_error <= ROLLING_ZERO_SCALE_TOLERANCE
        if not zero_scale:
            any_variable_pair = True
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                standardized = np.abs(bootstrap_means - observed_mean) / standard_error
            if not np.isfinite(standardized).all():
                raise ModelError("rolling-score bootstrap z statistic is non-finite")
            np.maximum(max_z, standardized, out=max_z)
        intermediate.append(
            (
                first_position,
                second_position,
                observed_mean,
                standard_error,
                zero_scale,
            )
        )

    critical_value = (
        float(
            np.quantile(
                max_z,
                ROLLING_MAX_Z_CONFIDENCE,
                method="higher",
            )
        )
        if any_variable_pair
        else 0.0
    )
    if not math.isfinite(critical_value):  # pragma: no cover - prior finite checks
        raise ModelError("rolling-score max-z critical value is non-finite")
    intervals: list[RollingPairwiseInterval] = []
    for (
        first_position,
        second_position,
        mean,
        standard_error,
        zero_scale,
    ) in intermediate:
        if zero_scale:
            lower = mean
            upper = mean
            exact_tie = abs(mean) <= ROLLING_NUMERICAL_TIE_TOLERANCE
            contains_zero = exact_tie
        else:
            margin = critical_value * standard_error
            lower = mean - margin
            upper = mean + margin
            if not all(math.isfinite(value) for value in (margin, lower, upper)):
                raise ModelError("rolling-score simultaneous interval is non-finite")
            exact_tie = False
            contains_zero = lower <= 0.0 <= upper
        intervals.append(
            RollingPairwiseInterval(
                first_n_states=ordered_counts[first_position],
                second_n_states=ordered_counts[second_position],
                observed_mean_difference=mean,
                bootstrap_standard_error=standard_error,
                interval_lower=lower,
                interval_upper=upper,
                zero_scale=zero_scale,
                exact_tie=exact_tie,
                contains_zero=contains_zero,
            )
        )
    return RollingMaxZSummary(
        candidate_state_counts=ordered_counts,
        candidate_means=tuple(float(value) for value in means_array),
        pairwise_intervals=tuple(intervals),
        bootstrap_replicates=ROLLING_BOOTSTRAP_REPLICATES,
        confidence=ROLLING_MAX_Z_CONFIDENCE,
        critical_value=critical_value,
    )


def conditional_rolling_log_score(
    *,
    training_and_validation_log_likelihood: float,
    training_log_likelihood: float,
    validation_observations: int,
) -> float:
    """Return the exact per-observation conditional rolling-fold score.

    The score is ``(log L(train + validation) - log L(train)) / n_validation``.
    Both likelihoods must have been evaluated under the same fold-fit model;
    this pure function records only the arithmetic contract.
    """

    if not math.isfinite(training_and_validation_log_likelihood) or not math.isfinite(
        training_log_likelihood
    ):
        raise ModelError("rolling-fold log likelihoods must be finite")
    _positive_integer(validation_observations, "validation_observations")
    difference = training_and_validation_log_likelihood - training_log_likelihood
    result = difference / validation_observations
    if not math.isfinite(difference) or not math.isfinite(result):
        raise ModelError("rolling-fold conditional score is non-finite")
    return float(result)


def select_candidate(
    candidates: Sequence[CandidateSelectionInput],
    *,
    bootstrap_indices: ArrayLike | None = None,
) -> CandidateSelectionReport:
    """Apply the binding all-pairs predictive selector and inclusive BIC guard."""

    normalized = _normalize_selection_inputs(candidates)
    pre_bic_viable = tuple(
        candidate for candidate in normalized if candidate.viability.admissible
    )
    if not pre_bic_viable:
        failure_rows = tuple(
            _inadmissible_ranking(candidate.viability) for candidate in normalized
        )
        return CandidateSelectionReport(
            selection_policy=ORDINARY_SELECTION_POLICY,
            candidates=failure_rows,
            rolling_max_z=None,
            minimum_bic_n_states=None,
            predictive_best_n_states=None,
            selected_n_states=None,
            bic_delta_maximum=BIC_DELTA_MAXIMUM,
            rolling_confidence=ROLLING_MAX_Z_CONFIDENCE,
            rolling_fold_count=ROLLING_FOLD_COUNT,
            rolling_score_count=ROLLING_SCORE_COUNT,
            bootstrap_replicates=0,
            max_z_critical_value=0.0,
            passed=False,
            failure_reason="no_pre_bic_viable_candidates",
        )

    score_matrices: dict[int, FloatArray] = {}
    invalid_folds: dict[int, tuple[int, ...]] = {}
    for candidate in pre_bic_viable:
        state_count = candidate.viability.evidence.n_states
        matrix, invalid = _selection_score_matrix(
            candidate.rolling_scores,
            label=f"K={state_count}",
        )
        score_matrices[state_count] = matrix
        invalid_folds[state_count] = invalid
    bic_by_state = {
        candidate.viability.evidence.n_states: _required_finite_bic(candidate.viability)
        for candidate in pre_bic_viable
    }

    minimum_bic_candidate = min(
        pre_bic_viable,
        key=lambda candidate: (
            _required_finite_bic(candidate.viability),
            candidate.viability.evidence.n_states,
        ),
    )
    minimum_bic = _required_finite_bic(minimum_bic_candidate.viability)
    eligible = tuple(
        candidate
        for candidate in pre_bic_viable
        if not invalid_folds[candidate.viability.evidence.n_states]
    )
    if not eligible:
        rows = _selection_rows(
            normalized,
            invalid_folds=invalid_folds,
            score_matrices=score_matrices,
            minimum_bic=minimum_bic,
            predictive_best=None,
            candidate_best_intervals={},
            near_best=frozenset(),
            selected=None,
        )
        return CandidateSelectionReport(
            selection_policy=ORDINARY_SELECTION_POLICY,
            candidates=rows,
            rolling_max_z=None,
            minimum_bic_n_states=(minimum_bic_candidate.viability.evidence.n_states),
            predictive_best_n_states=None,
            selected_n_states=None,
            bic_delta_maximum=BIC_DELTA_MAXIMUM,
            rolling_confidence=ROLLING_MAX_Z_CONFIDENCE,
            rolling_fold_count=ROLLING_FOLD_COUNT,
            rolling_score_count=ROLLING_SCORE_COUNT,
            bootstrap_replicates=0,
            max_z_critical_value=0.0,
            passed=False,
            failure_reason="no_predictively_eligible_candidates",
        )

    eligible_state_counts = tuple(
        candidate.viability.evidence.n_states for candidate in eligible
    )
    rolling_max_z: RollingMaxZSummary | None
    candidate_means: dict[int, float]
    interval_by_pair: dict[tuple[int, int], RollingPairwiseInterval]
    if len(eligible) == 1:
        sole_state_count = eligible_state_counts[0]
        sole_mean = _finite_matrix_mean(score_matrices[sole_state_count])
        rolling_max_z = None
        candidate_means = {sole_state_count: sole_mean}
        interval_by_pair = {}
        bootstrap_replicates = 0
        critical_value = 0.0
    else:
        if bootstrap_indices is None:
            raise ModelError(
                "two or more eligible candidates require precomputed bootstrap indices"
            )
        rolling_max_z = simultaneous_rolling_score_intervals(
            eligible_state_counts,
            np.stack(
                [score_matrices[state_count] for state_count in eligible_state_counts]
            ),
            bootstrap_indices,
        )
        candidate_means = dict(
            zip(
                rolling_max_z.candidate_state_counts,
                rolling_max_z.candidate_means,
                strict=True,
            )
        )
        interval_by_pair = {
            (interval.first_n_states, interval.second_n_states): interval
            for interval in rolling_max_z.pairwise_intervals
        }
        bootstrap_replicates = rolling_max_z.bootstrap_replicates
        critical_value = rolling_max_z.critical_value

    greatest_mean = max(candidate_means.values())
    predictive_best = min(
        state_count
        for state_count, mean in candidate_means.items()
        if greatest_mean - mean <= ROLLING_NUMERICAL_TIE_TOLERANCE
    )
    candidate_best_intervals: dict[int, tuple[float, float, float]] = {
        predictive_best: (0.0, 0.0, 0.0)
    }
    near_best = {predictive_best}
    for state_count in eligible_state_counts:
        if state_count == predictive_best:
            continue
        first, second = sorted((state_count, predictive_best))
        pair = interval_by_pair[(first, second)]
        if state_count == first:
            mean = pair.observed_mean_difference
            lower = pair.interval_lower
            upper = pair.interval_upper
        else:
            mean = -pair.observed_mean_difference
            lower = -pair.interval_upper
            upper = -pair.interval_lower
        candidate_best_intervals[state_count] = (mean, lower, upper)
        if pair.contains_zero:
            near_best.add(state_count)

    intersection = {
        state_count
        for state_count in near_best
        if _bic_delta(bic_by_state[state_count], minimum_bic) <= BIC_DELTA_MAXIMUM
    }
    selected_n_states = min(intersection) if intersection else None
    rows = _selection_rows(
        normalized,
        invalid_folds=invalid_folds,
        score_matrices=score_matrices,
        minimum_bic=minimum_bic,
        predictive_best=predictive_best,
        candidate_best_intervals=candidate_best_intervals,
        near_best=frozenset(near_best),
        selected=selected_n_states,
    )
    return CandidateSelectionReport(
        selection_policy=ORDINARY_SELECTION_POLICY,
        candidates=rows,
        rolling_max_z=rolling_max_z,
        minimum_bic_n_states=minimum_bic_candidate.viability.evidence.n_states,
        predictive_best_n_states=predictive_best,
        selected_n_states=selected_n_states,
        bic_delta_maximum=BIC_DELTA_MAXIMUM,
        rolling_confidence=ROLLING_MAX_Z_CONFIDENCE,
        rolling_fold_count=ROLLING_FOLD_COUNT,
        rolling_score_count=ROLLING_SCORE_COUNT,
        bootstrap_replicates=bootstrap_replicates,
        max_z_critical_value=critical_value,
        passed=selected_n_states is not None,
        failure_reason=(
            None
            if selected_n_states is not None
            else "no_candidate_in_near_best_bic_intersection"
        ),
    )


def select_owner_fixed_k4(
    candidates: Sequence[CandidateSelectionInput],
) -> CandidateSelectionReport:
    """Select only owner-fixed K=4; retain cross-K BIC/rolling as report-only.

    The complete K=2..5 family remains visible for audit, but neither a BIC
    rank nor a rolling-score comparison can select or promote a non-K4 model.
    K=4 is selected exactly when its upstream viability record is admissible.
    """

    normalized = _normalize_selection_inputs(candidates)
    state_counts = tuple(
        candidate.viability.evidence.n_states for candidate in normalized
    )
    if state_counts != tuple(sorted(_ALLOWED_STATE_COUNTS)):
        raise ModelError("owner-fixed K4 selection requires exact K=2,3,4,5 inputs")

    rows, minimum_bic_n_states, predictive_best_n_states = _fixed_k4_rows(normalized)
    fixed = next(
        candidate
        for candidate in normalized
        if candidate.viability.evidence.n_states == OWNER_FIXED_N_STATES
    )
    passed = fixed.viability.admissible
    return CandidateSelectionReport(
        selection_policy=OWNER_FIXED_K4_SELECTION_POLICY,
        candidates=rows,
        rolling_max_z=None,
        minimum_bic_n_states=minimum_bic_n_states,
        predictive_best_n_states=predictive_best_n_states,
        selected_n_states=OWNER_FIXED_N_STATES if passed else None,
        bic_delta_maximum=BIC_DELTA_MAXIMUM,
        rolling_confidence=ROLLING_MAX_Z_CONFIDENCE,
        rolling_fold_count=ROLLING_FOLD_COUNT,
        rolling_score_count=ROLLING_SCORE_COUNT,
        bootstrap_replicates=0,
        max_z_critical_value=0.0,
        passed=passed,
        failure_reason=None if passed else "owner_fixed_k4_not_admissible",
    )


def summarize_holdout_veto(
    *,
    hmm_log_score: float,
    baseline_log_score: float,
    sequence_lengths: Sequence[int],
    minimum_difference_per_observation: float,
) -> HoldoutVetoSummary:
    """Compute the sealed-holdout HMM-minus-baseline predictive-score veto."""

    if not math.isfinite(hmm_log_score) or not math.isfinite(baseline_log_score):
        raise ModelError("holdout log scores must be finite")
    if not math.isfinite(minimum_difference_per_observation):
        raise ModelError("holdout score threshold must be finite")
    lengths = _positive_lengths(sequence_lengths)
    observations = sum(lengths)
    difference = hmm_log_score - baseline_log_score
    per_observation = difference / observations
    if not math.isfinite(difference) or not math.isfinite(per_observation):
        raise ModelError("holdout score difference is non-finite")
    passed = per_observation >= minimum_difference_per_observation
    reasons = () if passed else ("holdout_score_below_minimum",)
    return HoldoutVetoSummary(
        hmm_log_score=hmm_log_score,
        baseline_log_score=baseline_log_score,
        sequence_lengths=lengths,
        observations=observations,
        score_difference=difference,
        difference_per_observation=per_observation,
        minimum_difference_per_observation=minimum_difference_per_observation,
        passed=passed,
        rejection_reasons=reasons,
    )


def apply_holdout_veto(
    selection: CandidateSelectionReport,
    holdout: HoldoutVetoSummary,
) -> HoldoutQualification:
    """Veto authorization without changing the design-selected candidate."""

    if not selection.passed or selection.selected_n_states is None:
        raise ModelError("holdout veto requires a successful candidate selection")
    selected = selection.selected_n_states
    return HoldoutQualification(
        selected_n_states=selected,
        authorized_n_states=selected if holdout.passed else None,
        passed=holdout.passed,
        holdout=holdout,
    )


def _validate_thresholds(thresholds: ViabilityThresholds) -> None:
    _positive_integer(
        thresholds.minimum_monthly_observations,
        "minimum_monthly_observations",
    )
    _nonnegative_integer(
        thresholds.minimum_completed_monthly_episodes,
        "minimum_completed_monthly_episodes",
    )
    _positive_integer(
        thresholds.minimum_daily_observations,
        "minimum_daily_observations",
    )
    _nonnegative_integer(thresholds.minimum_daily_runs, "minimum_daily_runs")
    if not math.isfinite(thresholds.minimum_occupancy) or not (
        0.0 <= thresholds.minimum_occupancy <= 1.0
    ):
        raise ModelError("minimum occupancy must be finite and in [0, 1]")
    if not math.isfinite(thresholds.maximum_covariance_condition_number) or (
        thresholds.maximum_covariance_condition_number <= 0.0
    ):
        raise ModelError(
            "maximum covariance condition number must be finite and positive"
        )
    for name, value in (
        ("minimum_bootstrap_median_ari", thresholds.minimum_bootstrap_median_ari),
        ("minimum_rolling_median_ari", thresholds.minimum_rolling_median_ari),
    ):
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ModelError(f"{name} must be finite and in [-1, 1]")


def _normalize_checks(
    checks: Sequence[PreselectionCheck],
) -> tuple[PreselectionCheck, ...]:
    normalized: list[PreselectionCheck] = []
    identifiers: set[str] = set()
    for check in checks:
        if not isinstance(check, PreselectionCheck):
            raise ModelError("preselection checks have an invalid record type")
        if not _CHECK_ID.fullmatch(check.check_id):
            raise ModelError(
                "preselection check ID must use lower snake_case",
                details={"check_id": check.check_id},
            )
        if check.check_id in identifiers:
            raise ModelError(
                "preselection check IDs must be unique",
                details={"check_id": check.check_id},
            )
        if not isinstance(check.passed, bool):
            raise ModelError("preselection check outcome must be boolean")
        identifiers.add(check.check_id)
        normalized.append(check)
    return tuple(sorted(normalized, key=lambda check: check.check_id))


def _normalize_states(
    states: Sequence[StateViabilityEvidence], n_states: int
) -> tuple[StateViabilityEvidence, ...]:
    normalized: list[StateViabilityEvidence] = []
    identifiers: set[int] = set()
    for state in states:
        if not isinstance(state, StateViabilityEvidence):
            raise ModelError("state viability evidence has an invalid record type")
        _nonnegative_integer(state.state, "state")
        if state.state >= n_states:
            raise ModelError("state evidence falls outside the candidate state range")
        if state.state in identifiers:
            raise ModelError("state viability evidence IDs must be unique")
        _nonnegative_integer(state.monthly_observations, "monthly_observations")
        _nonnegative_integer(
            state.completed_monthly_episodes,
            "completed_monthly_episodes",
        )
        _nonnegative_integer(state.daily_observations, "daily_observations")
        _nonnegative_integer(state.daily_runs, "daily_runs")
        if not math.isfinite(state.occupancy) or not 0.0 <= state.occupancy <= 1.0:
            raise ModelError("state occupancy must be finite and in [0, 1]")
        if state.covariance_condition_number < 0.0:
            raise ModelError("covariance condition number cannot be negative")
        identifiers.add(state.state)
        normalized.append(state)
    return tuple(sorted(normalized, key=lambda state: state.state))


def _append_optional_finite_reason(
    reasons: list[RejectionReason], name: str, observed: float | None
) -> None:
    if observed is None:
        reasons.append(
            RejectionReason(code=f"{name}_missing", required_relation="finite")
        )
    elif not isinstance(observed, int | float) or isinstance(observed, bool):
        raise ModelError(f"{name} must be numeric when present")
    elif not math.isfinite(observed):
        reasons.append(
            RejectionReason(
                code=f"{name}_nonfinite",
                required_relation="finite",
                observed=observed,
            )
        )


def _append_minimum_reason(
    reasons: list[RejectionReason],
    *,
    code: str,
    state: int,
    observed: Number,
    threshold: Number,
) -> None:
    if observed < threshold:
        reasons.append(
            RejectionReason(
                code=code,
                required_relation=">=",
                state=state,
                observed=observed,
                threshold=threshold,
            )
        )


def _append_ari_reason(
    reasons: list[RejectionReason],
    *,
    name: str,
    observed: float | None,
    threshold: float,
) -> None:
    if observed is None:
        reasons.append(
            RejectionReason(code=f"{name}_missing", required_relation="finite")
        )
        return
    if not isinstance(observed, int | float) or isinstance(observed, bool):
        raise ModelError(f"{name} must be numeric when present")
    if not math.isfinite(observed):
        reasons.append(
            RejectionReason(
                code=f"{name}_nonfinite",
                required_relation="finite",
                observed=observed,
            )
        )
    elif not -1.0 <= observed <= 1.0:
        raise ModelError(f"{name} must be in [-1, 1] when finite")
    elif observed < threshold:
        reasons.append(
            RejectionReason(
                code=f"{name}_below_minimum",
                required_relation=">=",
                observed=observed,
                threshold=threshold,
            )
        )


def _normalize_selection_inputs(
    candidates: Sequence[CandidateSelectionInput],
) -> tuple[CandidateSelectionInput, ...]:
    if not candidates:
        raise ModelError("candidate selection requires at least one candidate")
    normalized: list[CandidateSelectionInput] = []
    state_counts: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, CandidateSelectionInput):
            raise ModelError("candidate selection input has an invalid record type")
        viability = candidate.viability
        if not isinstance(viability, CandidateViabilityRecord):
            raise ModelError("candidate viability input has an invalid record type")
        state_count = viability.evidence.n_states
        if (
            isinstance(state_count, bool)
            or not isinstance(state_count, int)
            or state_count not in _ALLOWED_STATE_COUNTS
        ):
            raise ModelError("candidate state count must be one of 2, 3, 4, or 5")
        if state_count in state_counts:
            raise ModelError("candidate state counts must be unique")
        if viability.admissible:
            _required_finite_bic(viability)
            _selection_score_matrix(candidate.rolling_scores, label=f"K={state_count}")
        else:
            try:
                has_scores = len(candidate.rolling_scores) > 0
            except TypeError as error:
                raise ModelError(
                    "candidate rolling scores must be a sequence"
                ) from error
            if has_scores:
                _selection_score_matrix(
                    candidate.rolling_scores,
                    label=f"K={state_count}",
                )
        state_counts.add(state_count)
        normalized.append(candidate)
    return tuple(
        sorted(normalized, key=lambda candidate: candidate.viability.evidence.n_states)
    )


def _selection_state_counts(values: Sequence[int], *, minimum: int) -> tuple[int, ...]:
    if len(values) < minimum:
        raise ModelError(
            f"rolling max-z requires at least {minimum} candidate state counts"
        )
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in _ALLOWED_STATE_COUNTS
        ):
            raise ModelError("candidate state count must be one of 2, 3, 4, or 5")
        if value in seen:
            raise ModelError("candidate state counts must be unique")
        seen.add(value)
        result.append(value)
    return tuple(result)


def _finite_rolling_score_cube(
    values: ArrayLike, *, expected_candidates: int
) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError("rolling-score cube must be numeric") from error
    if raw.dtype.kind not in "fiu":
        raise ModelError("rolling-score cube must be numeric and cannot be boolean")
    expected_shape = (
        expected_candidates,
        ROLLING_FOLD_COUNT,
        ROLLING_FOLD_SIZE,
    )
    if raw.shape != expected_shape:
        raise ModelError(
            "rolling-score cube must have shape (candidate, 6, 12)",
            details={"expected": expected_shape, "observed": raw.shape},
        )
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "rolling-score cube cannot be represented as float64"
        ) from error
    if not np.isfinite(result).all():
        raise ModelError("rolling-score cube must be finite")
    return result


def _rolling_bootstrap_indices(values: ArrayLike) -> IntArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "rolling bootstrap indices must be an integer array"
        ) from error
    expected_shape = (
        ROLLING_BOOTSTRAP_REPLICATES,
        ROLLING_FOLD_COUNT,
        ROLLING_FOLD_SIZE,
    )
    if raw.shape != expected_shape:
        raise ModelError(
            "rolling bootstrap indices must have shape (10000, 6, 12)",
            details={"expected": expected_shape, "observed": raw.shape},
        )
    if raw.dtype.kind not in "iu":
        raise ModelError("rolling bootstrap indices must have integer dtype")
    if np.any(raw < 0) or np.any(raw >= ROLLING_FOLD_SIZE):
        raise ModelError("rolling bootstrap indices must be in [0, 11]")
    return np.asarray(raw, dtype=np.int64)


def _selection_score_matrix(
    values: Sequence[float | None], *, label: str
) -> tuple[FloatArray, tuple[int, ...]]:
    try:
        observed_length = len(values)
    except TypeError as error:
        raise ModelError(f"{label} rolling scores must be a sequence") from error
    if observed_length != ROLLING_SCORE_COUNT:
        raise ModelError(
            f"{label} rolling scores must contain exactly 72 fold-major values"
        )
    matrix = np.empty((ROLLING_FOLD_COUNT, ROLLING_FOLD_SIZE), dtype=np.float64)
    invalid_folds: list[int] = []
    for fold in range(ROLLING_FOLD_COUNT):
        start = fold * ROLLING_FOLD_SIZE
        fold_values = values[start : start + ROLLING_FOLD_SIZE]
        missing = tuple(value is None for value in fold_values)
        if all(missing):
            matrix[fold] = np.nan
            invalid_folds.append(fold)
            continue
        if any(missing):
            raise ModelError(
                f"{label} invalid rolling fold must contain exactly twelve None values"
            )
        for position, value in enumerate(fold_values):
            if isinstance(value, bool) or not isinstance(
                value,
                int | float | np.integer | np.floating,
            ):
                raise ModelError(f"{label} rolling scores must be numeric or None")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ModelError(f"{label} rolling scores must be finite when present")
            matrix[fold, position] = numeric
    matrix.setflags(write=False)
    return matrix, tuple(invalid_folds)


def _finite_matrix_mean(values: FloatArray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(values.mean())
    if not math.isfinite(result):
        raise ModelError("rolling-score candidate mean is non-finite")
    return result


def _fixed_k4_rows(
    candidates: Sequence[CandidateSelectionInput],
) -> tuple[tuple[CandidateRankingRecord, ...], int | None, int | None]:
    """Build descriptive cross-K rows without granting selection authority."""

    viable = tuple(
        candidate for candidate in candidates if candidate.viability.admissible
    )
    viable_bics = tuple(
        _required_finite_bic(candidate.viability) for candidate in viable
    )
    if viable:
        minimum_bic_candidate = min(
            viable,
            key=lambda candidate: (
                _required_finite_bic(candidate.viability),
                candidate.viability.evidence.n_states,
            ),
        )
        minimum_bic = _required_finite_bic(minimum_bic_candidate.viability)
        minimum_bic_n_states = minimum_bic_candidate.viability.evidence.n_states
    else:
        minimum_bic = None
        minimum_bic_n_states = None

    score_matrices: dict[int, FloatArray] = {}
    invalid_folds: dict[int, tuple[int, ...]] = {}
    rolling_means: dict[int, float] = {}
    for candidate in viable:
        state_count = candidate.viability.evidence.n_states
        matrix, invalid = _selection_score_matrix(
            candidate.rolling_scores,
            label=f"K={state_count}",
        )
        score_matrices[state_count] = matrix
        invalid_folds[state_count] = invalid
        if not invalid:
            rolling_means[state_count] = _finite_matrix_mean(matrix)
    predictive_best = None
    if rolling_means:
        greatest_mean = max(rolling_means.values())
        predictive_best = min(
            state_count
            for state_count, mean in rolling_means.items()
            if greatest_mean - mean <= ROLLING_NUMERICAL_TIE_TOLERANCE
        )
    best_mean = None if predictive_best is None else rolling_means[predictive_best]

    rows: list[CandidateRankingRecord] = []
    for candidate in candidates:
        viability = candidate.viability
        state_count = viability.evidence.n_states
        if not viability.admissible:
            rows.append(_inadmissible_ranking(viability))
            continue
        if minimum_bic is None:
            raise AssertionError("viable fixed-K4 row lacks a minimum BIC")
        bic = _required_finite_bic(viability)
        missing_folds = invalid_folds[state_count]
        eligible = not missing_folds
        rolling_mean = rolling_means.get(state_count)
        selected = state_count == OWNER_FIXED_N_STATES
        rows.append(
            CandidateRankingRecord(
                n_states=state_count,
                admissible=True,
                bic=bic,
                bic_rank=1 + sum(value < bic for value in viable_bics),
                delta_bic=_bic_delta(bic, minimum_bic),
                predictively_eligible=eligible,
                invalid_rolling_folds=missing_folds,
                rolling_mean=rolling_mean,
                candidate_minus_best_mean=(
                    None
                    if rolling_mean is None or best_mean is None
                    else rolling_mean - best_mean
                ),
                candidate_minus_best_lower=None,
                candidate_minus_best_upper=None,
                predictively_near_best=state_count == predictive_best,
                inside_bic_guard=(_bic_delta(bic, minimum_bic) <= BIC_DELTA_MAXIMUM),
                selected=selected,
                disposition="selected" if selected else "report_only_nonfixed_k",
                rejection_reasons=(),
                viability=viability,
            )
        )
    return tuple(rows), minimum_bic_n_states, predictive_best


def _selection_rows(
    candidates: Sequence[CandidateSelectionInput],
    *,
    invalid_folds: dict[int, tuple[int, ...]],
    score_matrices: dict[int, FloatArray],
    minimum_bic: float,
    predictive_best: int | None,
    candidate_best_intervals: dict[int, tuple[float, float, float]],
    near_best: frozenset[int],
    selected: int | None,
) -> tuple[CandidateRankingRecord, ...]:
    viable_bics = tuple(
        _required_finite_bic(candidate.viability)
        for candidate in candidates
        if candidate.viability.admissible
    )
    rows: list[CandidateRankingRecord] = []
    for candidate in candidates:
        viability = candidate.viability
        state_count = viability.evidence.n_states
        if not viability.admissible:
            rows.append(_inadmissible_ranking(viability))
            continue
        bic = _required_finite_bic(viability)
        delta = _bic_delta(bic, minimum_bic)
        missing_folds = invalid_folds[state_count]
        eligible = not missing_folds
        rolling_mean = (
            _finite_matrix_mean(score_matrices[state_count]) if eligible else None
        )
        difference, lower, upper = candidate_best_intervals.get(
            state_count,
            (None, None, None),
        )
        is_near_best = state_count in near_best
        inside_bic_guard = delta <= BIC_DELTA_MAXIMUM
        is_selected = state_count == selected
        if not eligible:
            disposition: SelectionDisposition = "predictively_ineligible"
        elif not is_near_best:
            disposition = "outside_simultaneous_near_best"
        elif not inside_bic_guard:
            disposition = "outside_bic_guard"
        elif is_selected:
            disposition = "selected"
        else:
            disposition = "larger_state_count_in_intersection"
        rows.append(
            CandidateRankingRecord(
                n_states=state_count,
                admissible=True,
                bic=bic,
                bic_rank=1 + sum(value < bic for value in viable_bics),
                delta_bic=delta,
                predictively_eligible=eligible,
                invalid_rolling_folds=missing_folds,
                rolling_mean=rolling_mean,
                candidate_minus_best_mean=difference,
                candidate_minus_best_lower=lower,
                candidate_minus_best_upper=upper,
                predictively_near_best=is_near_best,
                inside_bic_guard=inside_bic_guard,
                selected=is_selected,
                disposition=disposition,
                rejection_reasons=(),
                viability=viability,
            )
        )
    if predictive_best is None and any(row.predictively_eligible for row in rows):
        raise ModelError("predictive-best report invariant failed")
    return tuple(rows)


def _unit_uniform_scalar(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        raise ModelError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ModelError(f"{label} must be finite and in [0, 1)")
    return result


def _unit_uniform_vector(
    values: ArrayLike, *, expected_length: int, label: str
) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(f"{label} must be a numeric vector") from error
    if raw.shape != (expected_length,):
        raise ModelError(f"{label} must contain exactly {expected_length} values")
    if raw.dtype.kind not in "fiu":
        raise ModelError(f"{label} must be numeric and cannot be boolean")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result >= 1.0):
        raise ModelError(f"{label} must contain finite values in [0, 1)")
    return result


def _required_finite_bic(viability: CandidateViabilityRecord) -> float:
    bic = viability.evidence.bic
    if bic is None or not isinstance(bic, int | float) or isinstance(bic, bool):
        raise ModelError("admissible candidate must have a numeric BIC")
    if not math.isfinite(bic):
        raise ModelError("admissible candidate must have a finite BIC")
    return float(bic)


def _inadmissible_ranking(
    viability: CandidateViabilityRecord,
) -> CandidateRankingRecord:
    return CandidateRankingRecord(
        n_states=viability.evidence.n_states,
        admissible=False,
        bic=viability.evidence.bic,
        bic_rank=None,
        delta_bic=None,
        predictively_eligible=False,
        invalid_rolling_folds=(),
        rolling_mean=None,
        candidate_minus_best_mean=None,
        candidate_minus_best_lower=None,
        candidate_minus_best_upper=None,
        predictively_near_best=False,
        inside_bic_guard=False,
        selected=False,
        disposition="inadmissible",
        rejection_reasons=viability.rejection_reasons,
        viability=viability,
    )


def _bic_delta(bic: float, minimum_bic: float) -> float:
    difference = bic - minimum_bic
    if not math.isfinite(difference):
        raise ModelError("BIC difference is non-finite")
    return max(0.0, difference)


def _positive_lengths(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ModelError("holdout sequence lengths must be non-empty")
    result: list[int] = []
    for value in values:
        _positive_integer(value, "holdout sequence length")
        result.append(value)
    return tuple(result)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{name} must be a positive integer")


def _nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelError(f"{name} must be a non-negative integer")


def _json_number(value: Number | None) -> Number | str | None:
    """Encode non-finite diagnostic evidence without invalid JSON numbers."""

    if value is None or isinstance(value, int):
        return value
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "positive_infinity" if value > 0.0 else "negative_infinity"
    return value
