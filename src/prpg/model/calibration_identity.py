"""Exact content identities for Phase-3 calibration inputs and results.

These helpers make task bindings explicit without turning ordinary dataclasses
into authorization capabilities.  Every digest includes the role/geometry and
all numeric content consumed by the next task.  Fitted models are represented
by their platform-normalized HMM materialization fingerprint rather than by a
pickle or Python object representation.
"""

from __future__ import annotations

import math

import numpy as np

from prpg.errors import ModelError
from prpg.model.calibration import (
    CandidateFitSet,
    DecodedReturnPools,
    MacroStabilityEvidence,
    RollingCandidateEvidence,
    SealedHoldoutEvidence,
)
from prpg.model.input import CalibrationSlice
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_materialization_fingerprint,
)


def calibration_slice_fingerprint(value: CalibrationSlice) -> str:
    """Fingerprint one detached full/design/holdout slice and all masks."""

    if not isinstance(value, CalibrationSlice):
        raise ModelError("calibration slice fingerprint requires CalibrationSlice")
    if value.role not in {"full", "design", "holdout"}:
        raise ModelError("calibration slice role is invalid")
    return scientific_materialization_fingerprint(
        schema_id="calibration_slice",
        metadata={"role": value.role},
        arrays={
            "daily_continuation": value.daily_continuation,
            "daily_returns": value.daily_returns,
            "daily_session_ns": value.daily_session_ns,
            "macro_features": value.macro_features,
            "monthly_continuation": value.monthly_continuation,
            "monthly_period_ordinals": value.monthly_period_ordinals,
            "monthly_returns": value.monthly_returns,
        },
    )


def candidate_fit_set_fingerprint(value: CandidateFitSet) -> str:
    """Fingerprint exact ordered ordinary-fit successes and failures."""

    if not isinstance(value, CandidateFitSet):
        raise ModelError("candidate fit identity requires CandidateFitSet")
    attempts: list[dict[str, object]] = []
    for attempt in value.attempts:
        if attempt.fit is None:
            if attempt.failure_type is None or attempt.failure_message is None:
                raise ModelError("failed candidate fit lacks bounded failure evidence")
            fit_fingerprint = None
        else:
            if any(
                item is not None
                for item in (
                    attempt.failure_type,
                    attempt.failure_message,
                    attempt.failure_details,
                )
            ):
                raise ModelError("successful candidate fit retains failure evidence")
            fit_fingerprint = gaussian_hmm_fit_fingerprint(attempt.fit)
        attempts.append(
            {
                "n_states": attempt.n_states,
                "fit_fingerprint": fit_fingerprint,
                "failure_type": attempt.failure_type,
                "failure_message": attempt.failure_message,
                "failure_details": (
                    None
                    if attempt.failure_details is None
                    else dict(attempt.failure_details)
                ),
            }
        )
    return scientific_materialization_fingerprint(
        schema_id="candidate_fit_set",
        metadata={
            "scope": int(value.scope),
            "replicate": value.replicate,
            "attempts": attempts,
        },
        arrays={
            "attempt_order": np.asarray([item.n_states for item in value.attempts])
        },
    )


def decoded_return_pools_fingerprint(value: DecodedReturnPools) -> str:
    """Fingerprint the exact monthly/daily Viterbi labels used as source pools."""

    if not isinstance(value, DecodedReturnPools):
        raise ModelError("decoded-pool identity requires DecodedReturnPools")
    monthly = np.asarray(value.monthly_states)
    daily = np.asarray(value.daily_states)
    if monthly.ndim != 1 or daily.ndim != 1 or not monthly.size or not daily.size:
        raise ModelError("decoded return pools must be nonempty vectors")
    return scientific_materialization_fingerprint(
        schema_id="decoded_return_pools",
        metadata={
            "monthly_observations": int(monthly.size),
            "daily_observations": int(daily.size),
        },
        arrays={"daily_states": daily, "monthly_states": monthly},
    )


def rolling_candidate_evidence_fingerprint(
    value: RollingCandidateEvidence,
) -> str:
    """Fingerprint six folds, all scores/fits, alignments, and the summary."""

    if not isinstance(value, RollingCandidateEvidence):
        raise ModelError("rolling identity requires RollingCandidateEvidence")
    arrays: dict[str, np.ndarray] = {
        "summary_median_jaccard": np.asarray(value.stability.median_jaccard_by_state)
    }
    attempts: list[dict[str, object]] = []
    for attempt in value.attempts:
        scores = tuple(attempt.scores)
        if any(
            score is not None and not math.isfinite(float(score)) for score in scores
        ):
            raise ModelError("rolling evidence contains a non-finite score")
        arrays[f"fold_{attempt.fold}_jaccards"] = np.asarray(attempt.jaccards)
        attempts.append(
            {
                "fold": attempt.fold,
                "training_rows": attempt.training_rows,
                "validation_rows": attempt.validation_rows,
                "validation_lengths": list(attempt.validation_lengths),
                "fit_fingerprint": (
                    None
                    if attempt.fit is None
                    else gaussian_hmm_fit_fingerprint(attempt.fit)
                ),
                "scores": list(scores),
                "adjusted_rand_index": attempt.adjusted_rand_index,
                "failure_type": attempt.failure_type,
                "failure_message": attempt.failure_message,
            }
        )
    return scientific_materialization_fingerprint(
        schema_id="rolling_candidate_evidence",
        metadata={
            "n_states": value.n_states,
            "role": value.role,
            "attempts": attempts,
            "summary": {
                "successful_fits": value.stability.successful_fits,
                "median_ari": value.stability.median_ari,
                "minimum_successful_ari": value.stability.minimum_successful_ari,
                "passed": value.stability.passed,
            },
        },
        arrays=arrays,
    )


def macro_stability_evidence_fingerprint(value: MacroStabilityEvidence) -> str:
    """Fingerprint all 100 registered macro attempts and binding summary."""

    if not isinstance(value, MacroStabilityEvidence):
        raise ModelError("macro identity requires MacroStabilityEvidence")
    arrays: dict[str, np.ndarray] = {
        "summary_median_jaccard": np.asarray(value.summary.median_jaccard_by_state),
        "summary_tenth_jaccard": np.asarray(
            value.summary.tenth_percentile_jaccard_by_state
        ),
    }
    attempts: list[dict[str, object]] = []
    for attempt in value.attempts:
        arrays[f"attempt_{attempt.attempt}_source_indices"] = np.asarray(
            attempt.source_indices
        )
        arrays[f"attempt_{attempt.attempt}_jaccards"] = np.asarray(attempt.jaccards)
        if attempt.candidate_to_reference is not None:
            arrays[f"attempt_{attempt.attempt}_alignment"] = np.asarray(
                attempt.candidate_to_reference
            )
        attempts.append(
            {
                "attempt": attempt.attempt,
                "fit_fingerprint": (
                    None
                    if attempt.fit is None
                    else gaussian_hmm_fit_fingerprint(attempt.fit)
                ),
                "adjusted_rand_index": attempt.adjusted_rand_index,
                "has_alignment": attempt.candidate_to_reference is not None,
                "failure_type": attempt.failure_type,
                "failure_message": attempt.failure_message,
            }
        )
    return scientific_materialization_fingerprint(
        schema_id="macro_stability_evidence",
        metadata={
            "n_states": value.n_states,
            "role": value.role,
            "macro_block_length": value.macro_block_length,
            "attempts": attempts,
            "summary": {
                "successful_attempts": value.summary.successful_attempts,
                "median_ari": value.summary.median_ari,
                "tenth_percentile_ari": value.summary.tenth_percentile_ari,
                "passed": value.summary.passed,
            },
        },
        arrays=arrays,
    )


def sealed_holdout_evidence_fingerprint(value: SealedHoldoutEvidence) -> str:
    """Fingerprint the exact sealed score vectors and their recomputed decision."""

    if not isinstance(value, SealedHoldoutEvidence):
        raise ModelError("holdout identity requires SealedHoldoutEvidence")
    decision = value.decision
    return scientific_materialization_fingerprint(
        schema_id="sealed_holdout_evidence",
        metadata={
            "sequence_lengths": list(value.sequence_lengths),
            "decision": {
                "hmm_mean_log_score": decision.hmm_mean_log_score,
                "gaussian_mean_log_score": decision.gaussian_mean_log_score,
                "ridge_var1_mean_log_score": decision.ridge_var1_mean_log_score,
                "hmm_minus_gaussian": decision.hmm_minus_gaussian,
                "hmm_minus_ridge_var1": decision.hmm_minus_ridge_var1,
                "passed": decision.passed,
            },
        },
        arrays={
            "gaussian_scores": value.benchmark_scores.gaussian,
            "hmm_scores": value.hmm_log_scores,
            "ridge_var1_scores": value.benchmark_scores.ridge_var1,
        },
    )
