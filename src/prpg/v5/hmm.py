"""Fixed return-fitted K4 calibration and owner-review publication for v5."""

from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactReference, ModelArtifactStore
from prpg.model.hmm import (
    deterministic_hmm_restart_seeds,
    sequence_lengths_from_continuation,
    summarize_state_path,
)
from prpg.simulation.rng import ModelFitScope
from prpg.v5.config import ResolvedV5Inputs
from prpg.v5.data import V5MonthLibrary, load_v5_month_library

K4_REVIEW_SCHEMA = "prpg-v5-return-k4-owner-review-v1"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class RestartOutcome:
    """One deterministic restart outcome, with arrays only for valid fits."""

    restart_index: int
    seed: int
    valid: bool
    iterations: int
    log_likelihood: float | None
    final_likelihood_increment: float | None
    reason: str | None
    start_probabilities: np.ndarray[Any, Any] | None = None
    transition_matrix: np.ndarray[Any, Any] | None = None
    means_standardized: np.ndarray[Any, Any] | None = None
    variances_standardized: np.ndarray[Any, Any] | None = None
    decoded_states: np.ndarray[Any, Any] | None = None

    def diagnostic(self) -> dict[str, object]:
        return {
            "restart_index": self.restart_index,
            "seed": self.seed,
            "valid": self.valid,
            "iterations": self.iterations,
            "log_likelihood": self.log_likelihood,
            "final_likelihood_increment": self.final_likelihood_increment,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class V5K4Publication:
    """Verified model artifact and the two exact owner-facing files."""

    reference: ModelArtifactReference
    review_markdown_path: Path
    month_assignments_path: Path
    selected_restart_index: int
    selected_seed: int
    state_counts: tuple[int, int, int, int]
    pool_floor_passed: bool


class _FlooredDiagonalGaussianHMM(GaussianHMM):  # type: ignore[misc]
    """Pinned hmmlearn diagonal HMM with the approved binding variance floor."""

    _covars_: np.ndarray[Any, Any]

    def _do_mstep(self, stats: dict[str, Any]) -> None:
        super()._do_mstep(stats)
        covariances = np.asarray(self._covars_, dtype=np.float64)
        if covariances.ndim != 2 or not np.isfinite(covariances).all():
            raise FloatingPointError("diagonal covariance M-step was non-finite")
        self._covars_ = np.maximum(covariances, float(self.min_covar))


def fit_and_publish_v5_k4(
    inputs: ResolvedV5Inputs,
    processed_data_fingerprint: str,
    *,
    source_commit: str,
    review_directory: str | Path,
) -> V5K4Publication:
    """Fit K4 once, persist the artifact, and write the owner-review packet."""

    if not _COMMIT_PATTERN.fullmatch(source_commit):
        raise ModelError("version-5 K4 source commit must be a full SHA-1")
    library = load_v5_month_library(inputs, processed_data_fingerprint)
    values = np.asarray(library.monthly_returns, dtype=np.float64)
    scaler_means = values.mean(axis=0)
    scaler_ddof = inputs.config.model.standard_deviation_ddof
    scaler_standard_deviations = values.std(axis=0, ddof=scaler_ddof)
    if (
        not np.isfinite(scaler_means).all()
        or not np.isfinite(scaler_standard_deviations).all()
        or bool((scaler_standard_deviations <= 0.0).any())
    ):
        raise ModelError("version-5 K4 scaler is numerically invalid")
    standardized = (values - scaler_means) / scaler_standard_deviations
    lengths = sequence_lengths_from_continuation(library.monthly_continuation)
    seeds = deterministic_hmm_restart_seeds(
        master_seed=inputs.config.run.master_seed,
        scientific_version=inputs.config.model.scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=4,
        restart_count=inputs.config.model.deterministic_restarts,
    )
    outcomes = tuple(
        _fit_one_restart(
            standardized,
            lengths,
            restart_index=index,
            seed=seed,
            inputs=inputs,
        )
        for index, seed in enumerate(seeds)
    )
    best = _select_best_outcome(
        outcomes, expected_restarts=inputs.config.model.deterministic_restarts
    )
    canonical = _canonicalize_selected(best)
    start = canonical["start_probabilities"]
    transition = canonical["transition_matrix"]
    means = canonical["means_standardized"]
    variances = canonical["variances_standardized"]
    states = canonical["decoded_states"]
    stationary = _stationary_distribution(
        transition,
        probability_tolerance=inputs.config.model.probability_tolerance,
        residual_tolerance=inputs.config.model.stationary_residual_tolerance,
    )
    state_counts_array = np.bincount(states, minlength=4).astype(np.int64)
    if int(state_counts_array.sum()) != library.month_count:
        raise ModelError("version-5 K4 did not assign every eligible month")
    state_counts = tuple(int(value) for value in state_counts_array)
    if len(state_counts) != 4:
        raise AssertionError("K4 state count tuple has the wrong length")
    pool_floor_passed = bool(
        (state_counts_array >= inputs.config.model.minimum_state_months).all()
    )

    expected_counts, forward_log_likelihood = _expected_transition_counts(
        standardized,
        lengths,
        start,
        transition,
        means,
        variances,
    )
    path = summarize_state_path(states, 4, lengths=lengths)
    decoded_counts = np.asarray(path.transition_counts, dtype=np.int64)
    count_only_transition = _row_normalize(expected_counts)
    smoothed_transition = _row_normalize(expected_counts + 0.5)
    count_only_stationary, count_only_stationary_warning = (
        _report_only_stationary_distribution(
            count_only_transition,
            probability_tolerance=inputs.config.model.probability_tolerance,
            residual_tolerance=inputs.config.model.stationary_residual_tolerance,
        )
    )
    smoothed_stationary, smoothed_stationary_warning = (
        _report_only_stationary_distribution(
            smoothed_transition,
            probability_tolerance=inputs.config.model.probability_tolerance,
            residual_tolerance=inputs.config.model.stationary_residual_tolerance,
        )
    )
    review = _build_owner_review(
        inputs=inputs,
        library=library,
        values=values,
        states=states,
        lengths=lengths,
        scaler_means=scaler_means,
        scaler_standard_deviations=scaler_standard_deviations,
        outcomes=outcomes,
        best=best,
        canonical=canonical,
        stationary=stationary,
        expected_counts=expected_counts,
        decoded_counts=decoded_counts,
        count_only_transition=count_only_transition,
        smoothed_transition=smoothed_transition,
        count_only_stationary=count_only_stationary,
        smoothed_stationary=smoothed_stationary,
        count_only_stationary_warning=count_only_stationary_warning,
        smoothed_stationary_warning=smoothed_stationary_warning,
        forward_log_likelihood=forward_log_likelihood,
        state_counts=state_counts,
        pool_floor_passed=pool_floor_passed,
    )
    reference = ModelArtifactStore(inputs.artifact_root).create(
        artifact_role="production",
        data_fingerprint=processed_data_fingerprint,
        model_metadata={
            "schema": K4_REVIEW_SCHEMA,
            "model": "return-fitted-state-specific-diagonal-gaussian-k4",
            "feature_names": list(library.feature_names),
            "selected_restart_index": best.restart_index,
            "selected_seed": best.seed,
            "state_counts": list(state_counts),
            "minimum_state_months": inputs.config.model.minimum_state_months,
            "pool_floor_passed": pool_floor_passed,
            "owner_review_required": True,
            "generation_authorized": False,
        },
        arrays={
            "start_probabilities": start,
            "transition_matrix": transition,
            "means_standardized": means,
            "variances_standardized": variances,
            "scaler_means": scaler_means,
            "scaler_standard_deviations": scaler_standard_deviations,
            "stationary_distribution": stationary,
            "decoded_states": states,
            "monthly_period_ordinals": library.monthly_period_ordinals,
            "canonical_to_original": canonical["canonical_to_original"],
            "original_to_canonical": canonical["original_to_canonical"],
            "expected_transition_counts": expected_counts,
            "decoded_transition_counts": decoded_counts,
            "count_only_transition_matrix": count_only_transition,
            "smoothed_transition_matrix": smoothed_transition,
        },
        calibration_documents={"k4-calibration-review.json": review},
        provenance={
            "source_commit": source_commit,
            "config_fingerprint": inputs.config_fingerprint,
            "seed_registry_fingerprint": inputs.seed_registry_fingerprint,
            "processed_data_fingerprint": processed_data_fingerprint,
            "hmmlearn_version": version("hmmlearn"),
            "numpy_version": version("numpy"),
            "scipy_version": version("scipy"),
        },
    )
    review_root = Path(review_directory).expanduser().resolve(strict=False)
    prefix = reference.fingerprint[:12]
    assignment_path = review_root / f"v5-k4-month-assignments-{prefix}.csv"
    markdown_path = review_root / f"v5-k4-owner-review-{prefix}.md"
    assignment_content = _month_assignment_csv(library, values, states, review)
    markdown_content = _review_markdown(
        reference.fingerprint,
        review,
        assignment_path.name,
    ).encode("utf-8")
    _write_or_verify(assignment_path, assignment_content)
    _write_or_verify(markdown_path, markdown_content)
    # Verify the durable artifact and review document after publication.
    store = ModelArtifactStore(inputs.artifact_root)
    store.verify(reference.fingerprint)
    store.read_document(reference.fingerprint, "k4-calibration-review.json")
    return V5K4Publication(
        reference=reference,
        review_markdown_path=markdown_path,
        month_assignments_path=assignment_path,
        selected_restart_index=best.restart_index,
        selected_seed=best.seed,
        state_counts=(
            state_counts[0],
            state_counts[1],
            state_counts[2],
            state_counts[3],
        ),
        pool_floor_passed=pool_floor_passed,
    )


def _fit_one_restart(
    values: np.ndarray[Any, Any],
    lengths: Sequence[int],
    *,
    restart_index: int,
    seed: int,
    inputs: ResolvedV5Inputs,
) -> RestartOutcome:
    model_config = inputs.config.model
    model = _FlooredDiagonalGaussianHMM(
        n_components=4,
        covariance_type="diag",
        min_covar=model_config.variance_floor,
        startprob_prior=model_config.start_probability_prior,
        transmat_prior=model_config.transition_probability_prior,
        means_prior=model_config.means_prior,
        means_weight=model_config.means_weight,
        covars_prior=model_config.covariance_prior,
        covars_weight=model_config.covariance_weight,
        n_iter=model_config.maximum_iterations,
        tol=model_config.tolerance,
        random_state=seed,
        algorithm="viterbi",
        implementation="log",
        params="stmc",
        init_params="stmc",
    )
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            model.fit(values, lengths=lengths)
            history = np.asarray(tuple(model.monitor_.history), dtype=np.float64)
            converged, increment, reason = _convergence_result(
                history, tolerance=model_config.tolerance
            )
            if not converged:
                return RestartOutcome(
                    restart_index,
                    seed,
                    False,
                    len(history),
                    None,
                    increment,
                    reason,
                )
            likelihood = float(model.score(values, lengths=lengths))
            start = np.asarray(model.startprob_, dtype=np.float64).copy()
            transition = np.asarray(model.transmat_, dtype=np.float64).copy()
            means = np.asarray(model.means_, dtype=np.float64).copy()
            variances = np.asarray(model._covars_, dtype=np.float64).copy()
            states = np.asarray(model.predict(values, lengths=lengths), dtype=np.int64)
            _validate_restart_arrays(
                start,
                transition,
                means,
                variances,
                states,
                rows=len(values),
                variance_floor=model_config.variance_floor,
                probability_tolerance=model_config.probability_tolerance,
            )
            _stationary_distribution(
                transition,
                probability_tolerance=model_config.probability_tolerance,
                residual_tolerance=model_config.stationary_residual_tolerance,
            )
            if not math.isfinite(likelihood):
                raise FloatingPointError("training likelihood is non-finite")
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        return RestartOutcome(
            restart_index,
            seed,
            False,
            len(tuple(model.monitor_.history)),
            None,
            None,
            f"numerical:{type(error).__name__}",
        )
    return RestartOutcome(
        restart_index,
        seed,
        True,
        len(history),
        likelihood,
        increment,
        None,
        start,
        transition,
        means,
        variances,
        states,
    )


def _convergence_result(
    history: np.ndarray[Any, Any], *, tolerance: float
) -> tuple[bool, float | None, str | None]:
    if history.ndim != 1 or len(history) < 2:
        return False, None, "fewer_than_two_likelihood_iterations"
    if not np.isfinite(history).all():
        return False, None, "non_finite_likelihood_history"
    increment = float(history[-1] - history[-2])
    if abs(increment) >= tolerance:
        return False, increment, "final_absolute_likelihood_increment_not_below_tol"
    return True, increment, None


def _select_best_outcome(
    outcomes: Sequence[RestartOutcome], *, expected_restarts: int = 50
) -> RestartOutcome:
    if (
        expected_restarts <= 0
        or len(outcomes) != expected_restarts
        or tuple(item.restart_index for item in outcomes)
        != tuple(range(expected_restarts))
    ):
        raise ModelError(
            f"version-5 K4 requires exactly {expected_restarts} ordered restarts"
        )
    valid = [
        item for item in outcomes if item.valid and item.log_likelihood is not None
    ]
    if not valid:
        raise ModelError(
            "version-5 K4 has no numerically valid converged restart",
            details={"restarts": [item.diagnostic() for item in outcomes]},
        )

    def ranking_key(item: RestartOutcome) -> tuple[float, int]:
        assert item.log_likelihood is not None
        return item.log_likelihood, -item.restart_index

    return max(valid, key=ranking_key)


def _validate_restart_arrays(
    start: np.ndarray[Any, Any],
    transition: np.ndarray[Any, Any],
    means: np.ndarray[Any, Any],
    variances: np.ndarray[Any, Any],
    states: np.ndarray[Any, Any],
    *,
    rows: int,
    variance_floor: float,
    probability_tolerance: float,
) -> None:
    arrays = (start, transition, means, variances)
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("HMM parameters are non-finite")
    if (
        start.shape != (4,)
        or transition.shape != (4, 4)
        or means.shape != (4, 3)
        or variances.shape != (4, 3)
        or states.shape != (rows,)
    ):
        raise FloatingPointError("HMM parameter or state shape is invalid")
    if (
        float(start.min()) < -probability_tolerance
        or abs(float(start.sum()) - 1.0) > probability_tolerance
        or float(transition.min()) < -probability_tolerance
        or float(np.max(np.abs(transition.sum(axis=1) - 1.0))) > probability_tolerance
    ):
        raise FloatingPointError("HMM probability arrays are invalid")
    if bool((variances < variance_floor).any()):
        raise FloatingPointError("HMM diagonal variance floor was not enforced")
    if bool(((states < 0) | (states >= 4)).any()):
        raise FloatingPointError("HMM decoded states are invalid")


def _canonicalize_selected(best: RestartOutcome) -> dict[str, np.ndarray[Any, Any]]:
    if any(
        value is None
        for value in (
            best.start_probabilities,
            best.transition_matrix,
            best.means_standardized,
            best.variances_standardized,
            best.decoded_states,
        )
    ):
        raise ModelError("selected version-5 K4 restart omitted parameters")
    start = np.asarray(best.start_probabilities, dtype=np.float64)
    transition = np.asarray(best.transition_matrix, dtype=np.float64)
    means = np.asarray(best.means_standardized, dtype=np.float64)
    variances = np.asarray(best.variances_standardized, dtype=np.float64)
    decoded = np.asarray(best.decoded_states, dtype=np.int64)
    order = np.asarray(
        sorted(range(4), key=lambda state: (*means[state].tolist(), state)),
        dtype=np.int64,
    )
    inverse = np.empty(4, dtype=np.int64)
    inverse[order] = np.arange(4, dtype=np.int64)
    return {
        "start_probabilities": start[order],
        "transition_matrix": transition[np.ix_(order, order)],
        "means_standardized": means[order],
        "variances_standardized": variances[order],
        "decoded_states": inverse[decoded],
        "canonical_to_original": order,
        "original_to_canonical": inverse,
    }


def _stationary_distribution(
    transition: np.ndarray[Any, Any],
    *,
    probability_tolerance: float,
    residual_tolerance: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise FloatingPointError("K4 transition matrix is invalid")
    if (
        float(matrix.min()) < -probability_tolerance
        or float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) > probability_tolerance
    ):
        raise FloatingPointError("K4 transition matrix is not row-stochastic")
    system = np.vstack((matrix.T - np.eye(4), np.ones(4)))
    target = np.concatenate((np.zeros(4), np.ones(1)))
    if int(np.linalg.matrix_rank(system)) != 4:
        raise FloatingPointError("K4 stationary distribution is not unique")
    solved, *_ = np.linalg.lstsq(system, target, rcond=None)
    stationary: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(
        solved, dtype=np.float64
    )
    if (
        not np.isfinite(stationary).all()
        or float(stationary.min()) < -probability_tolerance
    ):
        raise FloatingPointError("K4 stationary distribution is invalid")
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    residual = float(np.max(np.abs(stationary @ matrix - stationary)))
    if residual > residual_tolerance:
        raise FloatingPointError("K4 stationary residual exceeds tolerance")
    stationary.setflags(write=False)
    return stationary


def _report_only_stationary_distribution(
    transition: np.ndarray[Any, Any],
    *,
    probability_tolerance: float,
    residual_tolerance: float,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]] | None, str | None]:
    """Return a diagnostic stationary law without turning it into a fit veto."""

    try:
        stationary = _stationary_distribution(
            transition,
            probability_tolerance=probability_tolerance,
            residual_tolerance=residual_tolerance,
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        return None, str(error)
    return stationary, None


def _expected_transition_counts(
    values: np.ndarray[Any, Any],
    lengths: Sequence[int],
    start: np.ndarray[Any, Any],
    transition: np.ndarray[Any, Any],
    means: np.ndarray[Any, Any],
    variances: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], float]:
    emissions = _diagonal_log_emissions(values, means, variances)
    log_start = np.where(start > 0.0, np.log(start), -np.inf)
    log_transition = np.where(transition > 0.0, np.log(transition), -np.inf)
    counts = np.zeros((4, 4), dtype=np.float64)
    total_likelihood = 0.0
    offset = 0
    for length in lengths:
        segment = emissions[offset : offset + length]
        alpha = np.empty((length, 4), dtype=np.float64)
        beta = np.empty((length, 4), dtype=np.float64)
        alpha[0] = log_start + segment[0]
        for index in range(1, length):
            alpha[index] = segment[index] + logsumexp(
                alpha[index - 1][:, None] + log_transition,
                axis=0,
            )
        segment_likelihood = float(logsumexp(alpha[-1]))
        beta[-1] = 0.0
        for index in range(length - 2, -1, -1):
            beta[index] = logsumexp(
                log_transition + segment[index + 1][None, :] + beta[index + 1][None, :],
                axis=1,
            )
        for index in range(length - 1):
            log_xi = (
                alpha[index][:, None]
                + log_transition
                + segment[index + 1][None, :]
                + beta[index + 1][None, :]
                - segment_likelihood
            )
            counts += np.exp(log_xi)
        total_likelihood += segment_likelihood
        offset += length
    if offset != len(values) or not np.isfinite(counts).all():
        raise ModelError("K4 forward-backward transition counts are invalid")
    return counts, total_likelihood


def _diagonal_log_emissions(
    values: np.ndarray[Any, Any],
    means: np.ndarray[Any, Any],
    variances: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    difference = values[:, None, :] - means[None, :, :]
    result: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(
        -0.5
        * (
            values.shape[1] * math.log(2.0 * math.pi)
            + np.log(variances).sum(axis=1)[None, :]
            + ((difference**2) / variances[None, :, :]).sum(axis=2)
        ),
        dtype=np.float64,
    )
    return result


def _row_normalize(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    if matrix.shape != (4, 4) or bool((totals <= 0.0).any()):
        raise ModelError("K4 transition count rows must be positive")
    result: np.ndarray[Any, np.dtype[np.float64]] = matrix / totals
    return result


def _build_owner_review(
    *,
    inputs: ResolvedV5Inputs,
    library: V5MonthLibrary,
    values: np.ndarray[Any, Any],
    states: np.ndarray[Any, Any],
    lengths: Sequence[int],
    scaler_means: np.ndarray[Any, Any],
    scaler_standard_deviations: np.ndarray[Any, Any],
    outcomes: Sequence[RestartOutcome],
    best: RestartOutcome,
    canonical: Mapping[str, np.ndarray[Any, Any]],
    stationary: np.ndarray[Any, Any],
    expected_counts: np.ndarray[Any, Any],
    decoded_counts: np.ndarray[Any, Any],
    count_only_transition: np.ndarray[Any, Any],
    smoothed_transition: np.ndarray[Any, Any],
    count_only_stationary: np.ndarray[Any, Any] | None,
    smoothed_stationary: np.ndarray[Any, Any] | None,
    count_only_stationary_warning: str | None,
    smoothed_stationary_warning: str | None,
    forward_log_likelihood: float,
    state_counts: tuple[int, ...],
    pool_floor_passed: bool,
) -> dict[str, object]:
    feature_names = library.feature_names
    months = pd.PeriodIndex.from_ordinals(library.monthly_period_ordinals, freq="M")
    path = summarize_state_path(states, 4, lengths=lengths)
    simple = np.expm1(values)
    fitted_monthly_log_means = (
        scaler_means[None, :]
        + np.asarray(canonical["means_standardized"], dtype=np.float64)
        * scaler_standard_deviations[None, :]
    )
    proposed = _proposed_state_labels(
        np.expm1(fitted_monthly_log_means), feature_names=feature_names
    )
    episodes = _decoded_episodes(months, states, lengths)
    transition = np.asarray(canonical["transition_matrix"], dtype=np.float64)
    state_rows: list[dict[str, object]] = []
    for state in range(4):
        selected = states == state
        state_values = values[selected]
        state_simple = simple[selected]
        state_months = months[selected]
        correlations = _correlation_matrix_or_none(state_values)
        worst: dict[str, list[dict[str, object]]] = {}
        for column, feature in enumerate(feature_names):
            order = np.argsort(state_simple[:, column], kind="stable")[:3]
            worst[feature] = [
                {
                    "month": str(state_months[index]),
                    "simple_return": float(state_simple[index, column]),
                }
                for index in order
            ]
        dwell = np.asarray(path.all_dwell_lengths_by_state[state], dtype=np.int64)
        entries = int(decoded_counts[:, state].sum() - decoded_counts[state, state])
        exits = int(decoded_counts[state].sum() - decoded_counts[state, state])
        warnings: list[str] = []
        if int(state_counts[state]) < inputs.config.model.minimum_state_months:
            warnings.append(
                f"Decoded pool has {state_counts[state]} months, below the binding "
                f"minimum of {inputs.config.model.minimum_state_months}; generation "
                "remains unauthorized."
            )
        if len(state_values) == 0:
            warnings.append(
                "No decoded observations; empirical return and duration statistics "
                "are not estimable."
            )
        elif len(state_values) == 1:
            warnings.append(
                "Only one decoded observation; sample volatility and correlation "
                "are not estimable."
            )
        if len(episodes[state]) < 2:
            warnings.append(
                "Only one decoded historical episode; recurrence and transitions are "
                "estimated from limited history."
            )
        state_rows.append(
            {
                "state_id": state,
                "proposed_name": proposed[state][0],
                "investment_interpretation": proposed[state][1],
                "months": [str(month) for month in state_months],
                "count": int(state_counts[state]),
                "share": float(state_counts[state] / len(states)),
                "episodes": episodes[state],
                "distinct_episode_count": len(episodes[state]),
                "episode_length_mean_months": (
                    None if len(dwell) == 0 else float(dwell.mean())
                ),
                "episode_length_median_months": (
                    None if len(dwell) == 0 else float(np.median(dwell))
                ),
                "decoded_entries": entries,
                "decoded_exits": exits,
                "mean_monthly_simple_return": {
                    feature: (
                        None
                        if len(state_simple) == 0
                        else float(state_simple[:, column].mean())
                    )
                    for column, feature in enumerate(feature_names)
                },
                "monthly_simple_return_volatility_ddof1": {
                    feature: (
                        None
                        if len(state_simple) < 2
                        else float(state_simple[:, column].std(ddof=1))
                    )
                    for column, feature in enumerate(feature_names)
                },
                "negative_month_fraction": {
                    feature: (
                        None
                        if len(state_simple) == 0
                        else float((state_simple[:, column] < 0.0).mean())
                    )
                    for column, feature in enumerate(feature_names)
                },
                "three_worst_months": worst,
                "monthly_log_return_correlation": (
                    None if correlations is None else correlations.tolist()
                ),
                "fitted_transition_row": transition[state].tolist(),
                "stationary_weight": float(stationary[state]),
                "decoded_transition_counts": decoded_counts[state].tolist(),
                "posterior_expected_transition_counts": expected_counts[state].tolist(),
                "count_only_transition_row": count_only_transition[state].tolist(),
                "half_pseudocount_transition_row": smoothed_transition[state].tolist(),
                "report_only_warnings": warnings,
            }
        )
    state_mean_logs = (
        None
        if any(count == 0 for count in state_counts)
        else np.vstack([values[states == state].mean(axis=0) for state in range(4)])
    )
    implied_monthly_log = (
        None if state_mean_logs is None else stationary @ state_mean_logs
    )
    historical_monthly_log = values.mean(axis=0)
    raw_identity = library.reference.manifest["identity"]
    scaler_deviation_key = (
        "scaler_standard_deviations_monthly_log_ddof"
        f"{inputs.config.model.standard_deviation_ddof}"
    )
    return {
        "schema": K4_REVIEW_SCHEMA,
        "status": "owner_review_required",
        "generation_authorized": False,
        "no_k3_or_cross_k_result_produced": True,
        "config_fingerprint": inputs.config_fingerprint,
        "seed_registry_fingerprint": inputs.seed_registry_fingerprint,
        "raw_snapshot_fingerprint": raw_identity["raw_snapshot_fingerprint"],
        "processed_data_fingerprint": library.reference.fingerprint,
        "eligible_first_month": str(months[0]),
        "eligible_last_month": str(months[-1]),
        "eligible_month_count": len(months),
        "contiguous_sequence_lengths": list(lengths),
        "feature_names": list(feature_names),
        "scaler_means_monthly_log": scaler_means.tolist(),
        scaler_deviation_key: scaler_standard_deviations.tolist(),
        "estimator_settings": inputs.config.model.model_dump(mode="json"),
        "selected_restart": {
            "restart_index": best.restart_index,
            "seed": best.seed,
            "iterations": best.iterations,
            "training_log_likelihood": best.log_likelihood,
            "forward_recomputed_log_likelihood": forward_log_likelihood,
            "final_likelihood_increment": best.final_likelihood_increment,
        },
        "restart_diagnostics": [item.diagnostic() for item in outcomes],
        "canonical_to_original": canonical["canonical_to_original"].tolist(),
        "original_to_canonical": canonical["original_to_canonical"].tolist(),
        "transition_matrix": transition.tolist(),
        "stationary_distribution": stationary.tolist(),
        "state_counts": list(state_counts),
        "minimum_state_months": inputs.config.model.minimum_state_months,
        "pool_floor_passed": pool_floor_passed,
        "transition_prior_sensitivity_report_only": {
            "posterior_expected_counts": expected_counts.tolist(),
            "count_only_transition_matrix": count_only_transition.tolist(),
            "half_pseudocount_transition_matrix": smoothed_transition.tolist(),
            "fitted_transition_matrix": transition.tolist(),
            "maximum_absolute_count_only_to_half_pseudocount_change": float(
                np.max(np.abs(count_only_transition - smoothed_transition))
            ),
            "maximum_absolute_fitted_to_recomputed_smoothed_change": float(
                np.max(np.abs(transition - smoothed_transition))
            ),
            "count_only_stationary_distribution": (
                None
                if count_only_stationary is None
                else count_only_stationary.tolist()
            ),
            "half_pseudocount_stationary_distribution": (
                None if smoothed_stationary is None else smoothed_stationary.tolist()
            ),
            "count_only_stationary_warning": count_only_stationary_warning,
            "half_pseudocount_stationary_warning": smoothed_stationary_warning,
        },
        "annualized_return_comparison": {
            feature: {
                "stationary_pool_implied_simple_return": (
                    None
                    if implied_monthly_log is None
                    else float(np.expm1(12.0 * implied_monthly_log[column]))
                ),
                "historical_simple_return": float(
                    np.expm1(12.0 * historical_monthly_log[column])
                ),
            }
            for column, feature in enumerate(feature_names)
        },
        "states": state_rows,
    }


def _proposed_state_labels(
    fitted_state_simple_return_means: np.ndarray[Any, Any],
    *,
    feature_names: Sequence[str] = ("ACWI", "LQD", "MUB"),
) -> dict[int, tuple[str, str]]:
    state_means = np.asarray(fitted_state_simple_return_means, dtype=np.float64)
    if (
        state_means.shape != (4, 3)
        or len(feature_names) != 3
        or not np.isfinite(state_means).all()
    ):
        raise ModelError("K4 fitted state means are invalid for owner-review labels")
    equity, taxable_bond, muni_bond = feature_names
    bond_stress = min((1, 2), key=lambda state: float(state_means[state, 1:].mean()))
    moderate = 1 if bond_stress == 2 else 2
    labels = {
        0: (
            "Lower-equity return environment",
            f"The lowest canonical {equity} centroid; inspect assigned months for "
            "broad risk-off or contraction intuition.",
        ),
        3: (
            "Higher-equity return environment",
            f"The highest canonical {equity} centroid; inspect assigned months for "
            "risk-on or recovery intuition.",
        ),
    }
    labels[bond_stress] = (
        "Bond-stress / mixed-risk environment",
        "Among the two middle-equity states, this state has the weaker average "
        f"{taxable_bond}/{muni_bond} return and may represent rate or credit pressure.",
    )
    labels[moderate] = (
        "Moderate expansion / defensive environment",
        "The remaining middle-equity state has comparatively stronger bond behavior "
        "and may represent an ordinary expansion or defensive market.",
    )
    return labels


def _correlation_matrix_or_none(
    values: np.ndarray[Any, Any],
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """Return a finite three-asset correlation matrix when it is estimable."""

    if len(values) < 2:
        return None
    correlation = np.asarray(np.corrcoef(values, rowvar=False), dtype=np.float64)
    if correlation.shape != (3, 3) or not np.isfinite(correlation).all():
        return None
    return correlation


def _decoded_episodes(
    months: pd.PeriodIndex,
    states: np.ndarray[Any, Any],
    lengths: Sequence[int],
) -> dict[int, list[dict[str, object]]]:
    result: dict[int, list[dict[str, object]]] = {state: [] for state in range(4)}
    offset = 0
    for segment, length in enumerate(lengths):
        stop = offset + length
        index = offset
        while index < stop:
            state = int(states[index])
            end = index + 1
            while end < stop and int(states[end]) == state:
                end += 1
            result[state].append(
                {
                    "start_month": str(months[index]),
                    "end_month": str(months[end - 1]),
                    "length_months": end - index,
                    "source_segment": segment,
                }
            )
            index = end
        offset = stop
    return result


def _month_assignment_csv(
    library: V5MonthLibrary,
    values: np.ndarray[Any, Any],
    states: np.ndarray[Any, Any],
    review: Mapping[str, object],
) -> bytes:
    state_rows = review["states"]
    if not isinstance(state_rows, list):
        raise ModelError("K4 review states are invalid")
    names = {
        int(item["state_id"]): str(item["proposed_name"])
        for item in state_rows
        if isinstance(item, Mapping)
    }
    months = pd.PeriodIndex.from_ordinals(
        library.monthly_period_ordinals, freq="M"
    ).astype(str)
    equity, taxable_bond, muni_bond = library.feature_names
    frame = pd.DataFrame(
        {
            "month": months,
            "state_id": states,
            "proposed_name": [names[int(state)] for state in states],
            f"{equity}_log_return": values[:, 0],
            f"{taxable_bond}_log_return": values[:, 1],
            f"{muni_bond}_log_return": values[:, 2],
            f"{equity}_simple_return": np.expm1(values[:, 0]),
            f"{taxable_bond}_simple_return": np.expm1(values[:, 1]),
            f"{muni_bond}_simple_return": np.expm1(values[:, 2]),
        }
    )
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def _review_markdown(
    model_fingerprint: str,
    review: Mapping[str, object],
    assignment_filename: str,
) -> str:
    if (
        not isinstance(review["selected_restart"], Mapping)
        or not isinstance(review["states"], list)
        or not isinstance(review["transition_matrix"], list)
        or not isinstance(review["stationary_distribution"], list)
        or not isinstance(review["annualized_return_comparison"], Mapping)
        or not isinstance(review["restart_diagnostics"], list)
        or not isinstance(review["feature_names"], list)
    ):
        raise ModelError("K4 owner review structure is invalid")
    selected = cast(Mapping[str, Any], review["selected_restart"])
    states = cast(list[Mapping[str, Any]], review["states"])
    transition = cast(list[list[float]], review["transition_matrix"])
    stationary = cast(list[float], review["stationary_distribution"])
    annualized = cast(
        Mapping[str, Mapping[str, float | None]],
        review["annualized_return_comparison"],
    )
    diagnostics = cast(list[Mapping[str, Any]], review["restart_diagnostics"])
    feature_names = cast(list[str], review["feature_names"])
    pool_status = "PASS" if review["pool_floor_passed"] else "PAUSE/NON-PASS"
    lines = [
        "# PRPG version-5 return-fitted K4 owner review",
        "",
        (
            "**Status:** Owner review required. This artifact does not authorize "
            "generation."
        ),
        "",
        f"- Model artifact: `{model_fingerprint}`",
        f"- Processed data: `{review['processed_data_fingerprint']}`",
        f"- Eligible months: {review['eligible_first_month']} through "
        f"{review['eligible_last_month']} ({review['eligible_month_count']} months)",
        f"- Feature order: {', '.join(feature_names)}",
        f"- Selected restart/seed: {selected['restart_index']} / {selected['seed']}",
        f"- Training log likelihood: {float(selected['training_log_likelihood']):.8f}",
        f"- State counts: {review['state_counts']}",
        f"- Minimum 10-month pool check: {pool_status}",
        f"- Exact assignment CSV: `{assignment_filename}`",
        "- No K3 or cross-K result was produced.",
        "",
        "## State summary",
        "",
        (
            "| State | Proposed descriptive name | Months | Share | Episodes | "
            "Mean episode | Median episode | Stationary weight |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in states:
        lines.append(
            f"| {item['state_id']} | {item['proposed_name']} | {item['count']} | "
            f"{item['share']:.3f} | {item['distinct_episode_count']} | "
            f"{_format_optional(item['episode_length_mean_months'], '.2f')} | "
            f"{_format_optional(item['episode_length_median_months'], '.2f')} | "
            f"{item['stationary_weight']:.3f} |"
        )
    lines.extend(["", "## Fitted transition matrix", ""])
    lines.extend(_matrix_markdown(transition, row_prefix="State"))
    lines.extend(
        [
            "",
            "Stationary distribution: "
            + ", ".join(
                f"S{i}={float(value):.4f}" for i, value in enumerate(stationary)
            ),
            "",
            "## Annualized return comparison",
            "",
            "| Asset | Stationary-pool implied | Historical |",
            "|---|---:|---:|",
        ]
    )
    for feature, values in annualized.items():
        implied = _format_optional(
            values["stationary_pool_implied_simple_return"], ".2%"
        )
        lines.append(
            f"| {feature} | {implied} | {values['historical_simple_return']:.2%} |"
        )
    for item in states:
        lines.extend(
            [
                "",
                f"## State {item['state_id']}: {item['proposed_name']}",
                "",
                str(item["investment_interpretation"]),
                "",
                "Assigned months:",
                "",
                ", ".join(item["months"]),
                "",
                "Decoded episodes:",
                "",
                "; ".join(
                    f"{episode['start_month']}-{episode['end_month']} "
                    f"({episode['length_months']}m)"
                    for episode in item["episodes"]
                )
                or "Not estimable (no decoded episode).",
                "",
                f"Decoded entries/exits: {item['decoded_entries']} / "
                f"{item['decoded_exits']}",
                "",
                (
                    "| Asset | Mean monthly simple return | Monthly volatility | "
                    "Negative-month fraction |"
                ),
                "|---|---:|---:|---:|",
            ]
        )
        for feature in feature_names:
            mean = _format_optional(item["mean_monthly_simple_return"][feature], ".3%")
            volatility = _format_optional(
                item["monthly_simple_return_volatility_ddof1"][feature], ".3%"
            )
            negative = _format_optional(item["negative_month_fraction"][feature], ".3f")
            lines.append(f"| {feature} | {mean} | {volatility} | {negative} |")
        lines.extend(["", "Three worst decoded months:", ""])
        for feature in feature_names:
            worst = item["three_worst_months"][feature]
            rendered = ", ".join(
                f"{entry['month']} ({entry['simple_return']:.2%})" for entry in worst
            )
            lines.append(f"- {feature}: {rendered or 'not estimable'}")
        correlation = item["monthly_log_return_correlation"]
        lines.extend(["", "Within-state monthly log-return correlation:", ""])
        if correlation is None:
            lines.append("Not estimable from this decoded pool.")
        else:
            lines.extend(
                _labeled_matrix_markdown(
                    cast(list[list[float]], correlation),
                    labels=feature_names,
                )
            )
        lines.extend(
            [
                "",
                "Fitted transition row: "
                + ", ".join(
                    f"S{index}={float(value):.4f}"
                    for index, value in enumerate(item["fitted_transition_row"])
                ),
                "Decoded transition counts: "
                + ", ".join(
                    f"S{index}={int(value)}"
                    for index, value in enumerate(item["decoded_transition_counts"])
                ),
                "Posterior-expected transition counts: "
                + ", ".join(
                    f"S{index}={float(value):.3f}"
                    for index, value in enumerate(
                        item["posterior_expected_transition_counts"]
                    )
                ),
            ]
        )
        if item["report_only_warnings"]:
            lines.extend(
                [
                    "",
                    "Report-only caution: " + " ".join(item["report_only_warnings"]),
                ]
            )
    lines.extend(
        [
            "",
            "## Restart outcomes",
            "",
            (
                "| Restart | Seed | Valid | Iterations | Log likelihood | "
                "Final increment | Reason |"
            ),
            "|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in diagnostics:
        likelihood = (
            "" if item["log_likelihood"] is None else f"{item['log_likelihood']:.8f}"
        )
        increment = (
            ""
            if item["final_likelihood_increment"] is None
            else f"{item['final_likelihood_increment']:.3e}"
        )
        lines.append(
            f"| {item['restart_index']} | {item['seed']} | {item['valid']} | "
            f"{item['iterations']} | {likelihood} | {increment} | "
            f"{item['reason'] or ''} |"
        )
    lines.extend(
        [
            "",
            "Transition-prior sensitivity, state semantics, and sparse-episode "
            "cautions "
            "are review information only. The exact numerical values are retained in "
            "`k4-calibration-review.json` inside the model artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _matrix_markdown(
    matrix: Sequence[Sequence[float]], *, row_prefix: str
) -> list[str]:
    lines = ["| From / To | S0 | S1 | S2 | S3 |", "|---|---:|---:|---:|---:|"]
    for index, row in enumerate(matrix):
        lines.append(
            f"| {row_prefix} {index} | "
            + " | ".join(f"{float(value):.4f}" for value in row)
            + " |"
        )
    return lines


def _labeled_matrix_markdown(
    matrix: Sequence[Sequence[float]], *, labels: Sequence[str]
) -> list[str]:
    header = "| Asset | " + " | ".join(labels) + " |"
    separator = "|---|" + "---:|" * len(labels)
    lines = [header, separator]
    for label, row in zip(labels, matrix, strict=True):
        lines.append(
            f"| {label} | " + " | ".join(f"{float(value):.3f}" for value in row) + " |"
        )
    return lines


def _format_optional(value: object, format_specification: str) -> str:
    if value is None:
        return "not estimable"
    return format(float(cast(float, value)), format_specification)


def _write_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise IntegrityError(
                "version-5 K4 review path already contains different content",
                details={"path": str(path)},
            )
        return
    path.write_bytes(content)
