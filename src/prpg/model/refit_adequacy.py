"""Pure execution primitives for the binding refitted-adequacy cells.

This module owns one parametric-bootstrap attempt, not the 1,000-attempt
orchestrator.  Randomness is always supplied by the caller: one state uniform
and exactly four standard normals are consumed for every macro row.  The
resulting history is refitted after a fresh population z-score transform,
aligned to the source artifact in that artifact's standardized coordinates,
and reduced to the frozen Section 4.5/4.6 endpoint vector.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.adequacy import (
    EndpointDirection,
    EndpointVector,
    marginal_moments,
    predictive_mixture_diagnostics,
    residual_acf_diagnostics,
)
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    StatePathDiagnostic,
    decode_viterbi,
    fit_gaussian_hmm_restarts,
    forward_backward,
    one_step_predictive_log_scores,
    posterior_identification_report,
    summarize_state_path,
    validate_numerical_chain,
)
from prpg.model.regime_policy import (
    OwnerFixedK4Disposition,
    effective_identification_passed,
    effective_identification_report_passed,
    effective_transition_support_passed,
)
from prpg.simulation.rng import ModelFitScope

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
FitFunction: TypeAlias = Callable[
    [HMMFeatureMatrix, int, Sequence[int]], GaussianHMMFit
]

MACRO_FEATURE_NAMES = (
    "industrial_production_growth",
    "inflation_yoy",
    "yield_curve_spread",
    "credit_spread",
)
PREDICTIVE_ACF_LAGS = (1, 3, 6, 12)
WITHIN_STATE_ACF_LAGS = (1, 2, 3)
DWELL_SURVIVAL_MONTHS = 12
MINIMUM_ACF_PAIRS = 10
MAXIMUM_STATE_COUNT = 5


class RefittedAdequacyEndpointSupportFailure(ModelError):
    """A valid refit path lacks realized support for a frozen endpoint.

    Only data-dependent scarcity in a generated replicate belongs to this
    type.  Malformed paths, invalid values, numerical faults, and endpoint
    contract drift remain ordinary ``ModelError`` failures and abort the
    registered batch.
    """


@dataclass(frozen=True, slots=True)
class MacroHistorySimulation:
    """One exact artifact-geometry history in source-artifact coordinates."""

    values: FloatArray
    latent_states: IntArray
    lengths: tuple[int, ...]
    state_uniforms_consumed: int
    standard_normals_per_row: int


@dataclass(frozen=True, slots=True)
class AlignedRefit:
    """A refit label assignment and its coordinate-audit evidence."""

    parameters: HMMParameters
    parameters_in_original_coordinates: HMMParameters
    candidate_to_reference: IntArray
    reference_to_candidate: IntArray
    aligned_decoded_states: IntArray | None
    squared_centroid_costs: FloatArray
    selected_total_cost: float
    refit_scaler_means_in_original_coordinates: FloatArray
    refit_scaler_standard_deviations_in_original_coordinates: FloatArray


@dataclass(frozen=True, slots=True)
class RefittedAdequacyEndpoint:
    """Complete named endpoint vector, directions, and support evidence."""

    vector: EndpointVector
    directions: tuple[EndpointDirection, ...]
    decoded_states: IntArray
    within_state_pair_counts: IntArray


@dataclass(frozen=True, slots=True)
class RefitAdequacyAttempt:
    """One successful registered refit attempt; failures propagate closed."""

    scope: ModelFitScope
    seeds: tuple[int, ...]
    simulation: MacroHistorySimulation
    recomputed_scaler_means: FloatArray
    recomputed_scaler_standard_deviations: FloatArray
    refit_standardized_values: FloatArray
    refit: GaussianHMMFit
    alignment: AlignedRefit
    endpoint: RefittedAdequacyEndpoint


def simulate_artifact_macro_history(
    parameters: HMMParameters,
    lengths: Sequence[int],
    state_uniforms: ArrayLike,
    emission_standard_normals: ArrayLike,
) -> MacroHistorySimulation:
    """Generate one tied-HMM history with a stationary reset per segment.

    There is no hidden RNG.  ``state_uniforms`` must contain exactly one value
    per output row and ``emission_standard_normals`` must have exact shape
    ``(rows, 4)``.  A segment's first uniform selects from the fitted stationary
    law; every later uniform selects from the preceding state's transition row.
    """

    start, transition, means, covariance = _validated_parameters(parameters)
    del start  # Segment starts deliberately use the stationary law.
    checked_lengths = _validated_lengths(lengths)
    row_count = sum(checked_lengths)
    if means.shape[1] != len(MACRO_FEATURE_NAMES):
        raise ModelError("refitted adequacy requires exactly four macro features")
    uniforms = _uniform_vector(state_uniforms, row_count, "state uniforms")
    normals = _normal_matrix(
        emission_standard_normals,
        (row_count, len(MACRO_FEATURE_NAMES)),
        "emission standard normals",
    )
    chain = validate_numerical_chain(transition)
    stationary = np.asarray(chain.stationary_distribution, dtype=np.float64)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ModelError("tied covariance Cholesky failed during simulation") from error
    if bool((np.diag(cholesky) <= 0.0).any()):
        raise ModelError("tied covariance Cholesky is not positive diagonal")

    states = np.empty(row_count, dtype=np.int64)
    values = np.empty((row_count, means.shape[1]), dtype=np.float64)
    offset = 0
    for length in checked_lengths:
        for within_segment in range(length):
            row = offset + within_segment
            probabilities = (
                stationary if within_segment == 0 else transition[states[row - 1]]
            )
            states[row] = _categorical_from_uniform(probabilities, uniforms[row])
            values[row] = means[states[row]] + cholesky @ normals[row]
        offset += length
    if not np.isfinite(values).all():
        raise ModelError("simulated macro history contains non-finite values")
    values = values.astype(np.float64, copy=True)
    states = states.astype(np.int64, copy=True)
    _readonly(values, states)
    return MacroHistorySimulation(
        values=values,
        latent_states=states,
        lengths=checked_lengths,
        state_uniforms_consumed=row_count,
        standard_normals_per_row=len(MACRO_FEATURE_NAMES),
    )


def align_refit_to_artifact(
    original_parameters: HMMParameters,
    refit_parameters: HMMParameters,
    refit_scaler_means_in_original_coordinates: ArrayLike,
    refit_scaler_standard_deviations_in_original_coordinates: ArrayLike,
    *,
    decoded_states: ArrayLike | None = None,
) -> AlignedRefit:
    """Align refit centroids in the original artifact's standardized space.

    The refit emissions live in its freshly standardized coordinates.  They
    are first mapped back with ``y = scaler_mean + scaler_sd * x``.  For K at
    most five, all assignments are enumerated and the one with minimum total
    squared Euclidean centroid cost is selected.  Exact assignment ties choose
    the lexicographically smallest reference-to-candidate permutation.
    """

    original = _validated_parameters(original_parameters)
    refit = _validated_parameters(refit_parameters)
    original_start, _original_transition, original_means, _original_covariance = (
        original
    )
    refit_start, refit_transition, refit_means, refit_covariance = refit
    n_states, n_features = original_means.shape
    if not 1 < n_states <= MAXIMUM_STATE_COUNT:
        raise ModelError("refit alignment requires K in the registered range 2..5")
    if refit_means.shape != original_means.shape:
        raise ModelError("original and refit HMM dimensions do not match")
    scaler_means = _finite_vector(
        refit_scaler_means_in_original_coordinates,
        n_features,
        "refit scaler means",
    )
    scaler_scales = _finite_vector(
        refit_scaler_standard_deviations_in_original_coordinates,
        n_features,
        "refit scaler standard deviations",
    )
    if bool((scaler_scales <= 0.0).any()):
        raise ModelError("refit scaler standard deviations must be positive")

    converted_means = scaler_means + refit_means * scaler_scales
    converted_covariance = (
        scaler_scales[:, None] * refit_covariance * scaler_scales[None, :]
    )
    differences = original_means[:, None, :] - converted_means[None, :, :]
    costs = np.einsum("rcf,rcf->rc", differences, differences)
    assignments = itertools.permutations(range(n_states))
    reference_to_candidate = min(
        assignments,
        key=lambda assignment: (
            math.fsum(
                float(costs[state, assignment[state]]) for state in range(n_states)
            ),
            assignment,
        ),
    )
    order = np.asarray(reference_to_candidate, dtype=np.int64)
    candidate_to_reference = np.empty(n_states, dtype=np.int64)
    candidate_to_reference[order] = np.arange(n_states, dtype=np.int64)
    selected_cost = math.fsum(
        float(costs[state, order[state]]) for state in range(n_states)
    )

    aligned_parameters = HMMParameters(
        start_probabilities=refit_start[order].copy(),
        transition_matrix=refit_transition[np.ix_(order, order)].copy(),
        means=refit_means[order].copy(),
        tied_covariance=refit_covariance.copy(),
    )
    original_coordinate_parameters = HMMParameters(
        start_probabilities=refit_start[order].copy(),
        transition_matrix=refit_transition[np.ix_(order, order)].copy(),
        means=converted_means[order].copy(),
        tied_covariance=converted_covariance.copy(),
    )
    aligned_states: IntArray | None = None
    if decoded_states is not None:
        raw_states = _state_vector(decoded_states, n_states)
        aligned_states = candidate_to_reference[raw_states].astype(np.int64, copy=True)
        _readonly(aligned_states)
    costs = costs.astype(np.float64, copy=True)
    scaler_means = scaler_means.astype(np.float64, copy=True)
    scaler_scales = scaler_scales.astype(np.float64, copy=True)
    _readonly(costs, order, candidate_to_reference, scaler_means, scaler_scales)
    # A validated original start vector is evidence that K was checked even
    # though it is otherwise irrelevant to centroid matching.
    if len(original_start) != len(order):  # pragma: no cover - defensive.
        raise AssertionError("validated original HMM state count changed")
    return AlignedRefit(
        parameters=aligned_parameters,
        parameters_in_original_coordinates=original_coordinate_parameters,
        candidate_to_reference=candidate_to_reference,
        reference_to_candidate=order,
        aligned_decoded_states=aligned_states,
        squared_centroid_costs=costs,
        selected_total_cost=float(selected_cost),
        refit_scaler_means_in_original_coordinates=scaler_means,
        refit_scaler_standard_deviations_in_original_coordinates=scaler_scales,
    )


def build_refitted_adequacy_endpoint(
    values: ArrayLike,
    parameters: HMMParameters,
    lengths: Sequence[int],
    *,
    feature_names: Sequence[str] = MACRO_FEATURE_NAMES,
    owner_fixed_k4_disposition: OwnerFixedK4Disposition | None = None,
) -> RefittedAdequacyEndpoint:
    """Construct the complete frozen refitted-adequacy endpoint vector."""

    matrix = _finite_matrix(values, "refitted adequacy values")
    checked_lengths = _validated_lengths(lengths, expected_rows=len(matrix))
    _start, _transition, means, covariance = _validated_parameters(parameters)
    if matrix.shape[1] != len(MACRO_FEATURE_NAMES) or means.shape[1] != len(
        MACRO_FEATURE_NAMES
    ):
        raise ModelError("refitted adequacy requires exactly four macro features")
    checked_names = tuple(feature_names)
    if checked_names != MACRO_FEATURE_NAMES:
        raise ModelError("refitted adequacy macro feature order is not canonical")
    n_states = len(means)
    if not 1 < n_states <= MAXIMUM_STATE_COUNT:
        raise ModelError("refitted adequacy requires K in the range 2..5")

    continuation = _continuation_from_lengths(checked_lengths)
    decoded = decode_viterbi(matrix, parameters, lengths=checked_lengths)
    path = summarize_state_path(decoded, n_states, lengths=checked_lengths)
    decoded_labels, decoded_values = _decoded_state_duration_values(path)

    posterior = forward_backward(matrix, parameters, lengths=checked_lengths)
    identification = posterior_identification_report(
        posterior.posterior_probabilities,
        decoded,
        parameters,
    )
    if not effective_identification_report_passed(
        identification,
        n_states=n_states,
        disposition=owner_fixed_k4_disposition,
    ):
        raise ModelError(
            "refitted adequacy posterior-identification gates failed",
            details={"rejection_reasons": list(identification.rejection_reasons)},
        )
    assigned = np.asarray(
        identification.assigned_state_mean_posteriors, dtype=np.float64
    )
    if not np.isfinite(assigned).all():
        raise ModelError("posterior assigned-state summaries are unavailable")

    predictive_scores = one_step_predictive_log_scores(
        matrix,
        parameters,
        lengths=checked_lengths,
    )
    predictive = predictive_mixture_diagnostics(
        matrix,
        predictive_scores.predictive_state_probabilities,
        means,
        covariance,
    )
    predictive_acf = residual_acf_diagnostics(
        predictive.predictive_residuals,
        continuation,
        PREDICTIVE_ACF_LAGS,
        minimum_pairs=MINIMUM_ACF_PAIRS,
    )
    predictive_raw_max = np.max(np.abs(predictive_acf.raw), axis=1)
    predictive_squared_max = np.max(np.abs(predictive_acf.squared), axis=1)
    moments = marginal_moments(matrix)

    inverse_root = _symmetric_inverse_square_root(covariance)
    centered_by_state = matrix - means[decoded]
    state_residuals = centered_by_state @ inverse_root
    within_raw, within_squared, within_counts = _within_state_acfs(
        state_residuals,
        decoded,
        continuation,
        n_states=n_states,
    )

    labels: list[str] = list(decoded_labels)
    components: list[float] = list(decoded_values)
    directions: list[EndpointDirection] = [EndpointDirection.TWO_SIDED] * len(
        decoded_values
    )
    labels.extend(
        (
            "posterior/normalized_entropy",
            "posterior/mean_maximum",
        )
    )
    components.extend(
        (
            identification.normalized_posterior_entropy,
            identification.mean_maximum_posterior,
        )
    )
    directions.extend((EndpointDirection.TWO_SIDED,) * 2)
    for state in range(n_states):
        labels.append(f"posterior/assigned/{state}")
        components.append(float(assigned[state]))
        directions.append(EndpointDirection.TWO_SIDED)

    for feature, name in enumerate(checked_names):
        labels.append(f"predictive_residual/mean/{name}")
        components.append(float(predictive.predictive_residuals[:, feature].mean()))
        directions.append(EndpointDirection.TWO_SIDED)
    upper_index = 0
    for row, row_name in enumerate(checked_names):
        for column in range(row, len(checked_names)):
            labels.append(
                f"predictive_residual/covariance_error/{row_name}/{checked_names[column]}"
            )
            components.append(
                float(predictive.predictive_covariance_error_upper[upper_index])
            )
            directions.append(EndpointDirection.TWO_SIDED)
            upper_index += 1
    for feature, name in enumerate(checked_names):
        labels.append(f"predictive_residual/raw_max_abs_acf/{name}")
        components.append(float(predictive_raw_max[feature]))
        directions.append(EndpointDirection.UPPER_TAIL)
    for feature, name in enumerate(checked_names):
        labels.append(f"predictive_residual/squared_max_abs_acf/{name}")
        components.append(float(predictive_squared_max[feature]))
        directions.append(EndpointDirection.UPPER_TAIL)
    for feature, name in enumerate(checked_names):
        labels.append(f"pit/cvm/{name}")
        components.append(float(predictive.pit_cramer_von_mises[feature]))
        directions.append(EndpointDirection.UPPER_TAIL)
    moment_names = ("mean", "variance", "skewness", "kurtosis")
    for feature, name in enumerate(checked_names):
        for moment_index, moment_name in enumerate(moment_names):
            labels.append(f"marginal/{name}/{moment_name}")
            components.append(float(moments[feature, moment_index]))
            directions.append(EndpointDirection.TWO_SIDED)

    for state in range(n_states):
        for feature, name in enumerate(checked_names):
            for lag_index, lag in enumerate(WITHIN_STATE_ACF_LAGS):
                labels.append(f"within_state/{state}/{name}/lag_{lag}/raw")
                components.append(float(within_raw[state, feature, lag_index]))
                directions.append(EndpointDirection.TWO_SIDED)
                labels.append(f"within_state/{state}/{name}/lag_{lag}/squared")
                components.append(float(within_squared[state, feature, lag_index]))
                directions.append(EndpointDirection.TWO_SIDED)

    result = np.asarray(components, dtype=np.float64)
    expected = n_states * (n_states - 1) + 41 * n_states + 44
    if len(labels) != expected or result.shape != (expected,):
        raise AssertionError("refitted adequacy endpoint order is inconsistent")
    if len(directions) != expected or not np.isfinite(result).all():
        raise ModelError("refitted adequacy endpoint vector is invalid")
    result = result.astype(np.float64, copy=True)
    decoded = decoded.astype(np.int64, copy=True)
    within_counts = within_counts.astype(np.int64, copy=True)
    _readonly(result, decoded, within_counts)
    return RefittedAdequacyEndpoint(
        vector=EndpointVector(tuple(labels), result),
        directions=tuple(directions),
        decoded_states=decoded,
        within_state_pair_counts=within_counts,
    )


def run_refitted_adequacy_attempt(
    original_fit: GaussianHMMFit,
    artifact_features: HMMFeatureMatrix,
    state_uniforms: ArrayLike,
    emission_standard_normals: ArrayLike,
    *,
    scope: ModelFitScope,
    seeds: Sequence[int],
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> RefitAdequacyAttempt:
    """Execute one attempt using an injected, explicitly seeded 50-start fit.

    Narrow HMM scientific non-passes and endpoint-support failures propagate as
    their explicit subtypes so the outer fixed 1,000-attempt orchestrator can
    retain them without replacement.  Every generic numerical, fit, alignment,
    decode, or contract ``ModelError`` propagates and aborts the batch.
    """

    _validate_attempt_inputs(original_fit, artifact_features, scope)
    checked_seeds = _validated_seeds(seeds)
    simulation = simulate_artifact_macro_history(
        original_fit.parameters,
        original_fit.lengths,
        state_uniforms,
        emission_standard_normals,
    )
    scaler_means = simulation.values.mean(axis=0)
    scaler_scales = simulation.values.std(axis=0, ddof=0)
    if not np.isfinite(scaler_means).all() or not np.isfinite(scaler_scales).all():
        raise ModelError("recomputed parametric-adequacy scaler is non-finite")
    if bool((scaler_scales <= 0.0).any()):
        raise ModelError("recomputed parametric-adequacy scaler is non-positive")
    standardized = (simulation.values - scaler_means) / scaler_scales
    if not np.isfinite(standardized).all():
        raise ModelError("refit-standardized macro history is non-finite")
    refit_features = HMMFeatureMatrix(
        values=standardized.astype(np.float64, copy=True),
        lengths=original_fit.lengths,
        feature_names=artifact_features.feature_names,
        row_labels=artifact_features.row_labels,
        scaler_means=scaler_means.astype(np.float64, copy=True),
        scaler_standard_deviations=scaler_scales.astype(np.float64, copy=True),
        scaler_scope=artifact_features.scaler_scope,
    )
    refit = fit_function(refit_features, original_fit.n_states, checked_seeds)
    _validate_refit_output(refit, original_fit, refit_features, checked_seeds)
    alignment = align_refit_to_artifact(
        original_fit.parameters,
        refit.parameters,
        scaler_means,
        scaler_scales,
        decoded_states=refit.decoded_states,
    )
    endpoint = build_refitted_adequacy_endpoint(
        standardized,
        alignment.parameters,
        original_fit.lengths,
        feature_names=artifact_features.feature_names,
        owner_fixed_k4_disposition=refit.owner_fixed_k4_disposition,
    )
    if alignment.aligned_decoded_states is None or not np.array_equal(
        endpoint.decoded_states, alignment.aligned_decoded_states
    ):
        raise ModelError("aligned refit decoder is inconsistent with fitted path")
    scaler_means = scaler_means.astype(np.float64, copy=True)
    scaler_scales = scaler_scales.astype(np.float64, copy=True)
    standardized = standardized.astype(np.float64, copy=True)
    _readonly(scaler_means, scaler_scales, standardized)
    return RefitAdequacyAttempt(
        scope=scope,
        seeds=checked_seeds,
        simulation=simulation,
        recomputed_scaler_means=scaler_means,
        recomputed_scaler_standard_deviations=scaler_scales,
        refit_standardized_values=standardized,
        refit=refit,
        alignment=alignment,
        endpoint=endpoint,
    )


def _decoded_state_duration_values(
    path: StatePathDiagnostic,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    occupancy = np.asarray(path.occupancy, dtype=np.float64)
    transitions = np.asarray(path.transition_counts, dtype=np.int64)
    dwell_by_state = path.dwell_lengths_by_state
    n_states = len(occupancy)
    if transitions.shape != (n_states, n_states):
        raise ModelError("decoded transition-count shape is inconsistent")
    departures = transitions.sum(axis=1)
    if bool((departures == 0).any()):
        raise RefittedAdequacyEndpointSupportFailure(
            "decoded transition denominator is zero"
        )
    labels: list[str] = [f"decoded/occupancy/{state}" for state in range(n_states)]
    values: list[float] = [float(value) for value in occupancy]
    for source in range(n_states):
        for target in range(n_states):
            if source == target:
                continue
            labels.append(f"decoded/transition/{source}/{target}")
            values.append(float(transitions[source, target] / departures[source]))
    row_count = int(np.asarray(path.state_counts, dtype=np.int64).sum())
    for state in range(n_states):
        labels.append(f"decoded/episode_rate/{state}")
        entries = int(transitions[:, state].sum() - transitions[state, state])
        values.append(float(entries / row_count))
    checked_dwell: list[IntArray] = []
    for state, dwell in enumerate(dwell_by_state):
        array = np.asarray(dwell)
        if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
            raise ModelError("decoded completed dwell values must be integer vectors")
        if len(array) == 0:
            raise RefittedAdequacyEndpointSupportFailure(
                "decoded completed dwell support is unavailable",
                details={"state": state},
            )
        if bool((array <= 0).any()):
            raise ModelError(
                "decoded completed dwell values must be positive",
                details={"state": state},
            )
        checked_dwell.append(array.astype(np.int64, copy=False))
    for state, dwell in enumerate(checked_dwell):
        labels.append(f"decoded/one_period/{state}")
        values.append(float(np.count_nonzero(dwell == 1) / len(dwell)))
    for state, dwell in enumerate(checked_dwell):
        labels.append(f"decoded/median_dwell/{state}")
        values.append(float(np.quantile(dwell, 0.5, method="higher")))
    for state, dwell in enumerate(checked_dwell):
        for month in range(1, DWELL_SURVIVAL_MONTHS + 1):
            labels.append(f"decoded/dwell_survival/{state}/{month}")
            values.append(float(np.count_nonzero(dwell >= month) / len(dwell)))
    return tuple(labels), tuple(values)


def _within_state_acfs(
    residuals: FloatArray,
    states: IntArray,
    continuation: BoolArray,
    *,
    n_states: int,
) -> tuple[FloatArray, FloatArray, IntArray]:
    n_features = residuals.shape[1]
    raw = np.empty((n_states, n_features, len(WITHIN_STATE_ACF_LAGS)), dtype=np.float64)
    squared = np.empty_like(raw)
    counts = np.empty(raw.shape, dtype=np.int64)
    run_ids = np.zeros(len(states), dtype=np.int64)
    for row in range(1, len(states)):
        run_ids[row] = run_ids[row - 1]
        if not bool(continuation[row]) or states[row] != states[row - 1]:
            run_ids[row] += 1
    for state in range(n_states):
        state_rows = states == state
        if not bool(state_rows.any()):
            raise RefittedAdequacyEndpointSupportFailure(
                "within-state residual support is unavailable",
                details={"state": state},
            )
        centered = residuals - residuals[state_rows].mean(axis=0)
        centered_squared = residuals * residuals
        centered_squared -= centered_squared[state_rows].mean(axis=0)
        for lag_index, lag in enumerate(WITHIN_STATE_ACF_LAGS):
            eligible = (
                (states[lag:] == state)
                & (states[:-lag] == state)
                & (run_ids[lag:] == run_ids[:-lag])
            )
            count = int(np.count_nonzero(eligible))
            if count < MINIMUM_ACF_PAIRS:
                raise RefittedAdequacyEndpointSupportFailure(
                    "within-state residual ACF has insufficient valid pairs",
                    details={
                        "state": state,
                        "lag": lag,
                        "actual": count,
                        "required": MINIMUM_ACF_PAIRS,
                    },
                )
            for feature in range(n_features):
                raw[state, feature, lag_index] = _cosine_acf(
                    centered[lag:, feature][eligible],
                    centered[:-lag, feature][eligible],
                )
                squared[state, feature, lag_index] = _cosine_acf(
                    centered_squared[lag:, feature][eligible],
                    centered_squared[:-lag, feature][eligible],
                )
                counts[state, feature, lag_index] = count
    _readonly(raw, squared, counts)
    return raw, squared, counts


def _cosine_acf(left: FloatArray, right: FloatArray) -> float:
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if not math.isfinite(denominator):
        raise ModelError("within-state residual ACF denominator is non-finite")
    if denominator <= 1e-12:
        raise RefittedAdequacyEndpointSupportFailure(
            "within-state residual ACF denominator is unavailable"
        )
    result = float(left @ right) / denominator
    if not math.isfinite(result):
        raise ModelError("within-state residual ACF is non-finite")
    return result


def _symmetric_inverse_square_root(covariance: FloatArray) -> FloatArray:
    symmetric = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as error:
        raise ModelError(
            "state-residual covariance eigendecomposition failed"
        ) from error
    if not np.isfinite(eigenvalues).all() or bool((eigenvalues <= 0.0).any()):
        raise ModelError("state-residual covariance is not positive definite")
    result = np.asarray(
        (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T,
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ModelError("state-residual inverse square root is non-finite")
    return result


def _validate_attempt_inputs(
    original_fit: GaussianHMMFit,
    artifact_features: HMMFeatureMatrix,
    scope: ModelFitScope,
) -> None:
    if not isinstance(original_fit, GaussianHMMFit):
        raise ModelError("original adequacy artifact must be a GaussianHMMFit")
    if not isinstance(artifact_features, HMMFeatureMatrix):
        raise ModelError("artifact geometry must be an HMMFeatureMatrix")
    if not isinstance(scope, ModelFitScope):
        raise ModelError("parametric-adequacy scope must be a ModelFitScope")
    expected_scope = (
        ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY
        if original_fit.scaler_scope == "design_training"
        else ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY
    )
    if original_fit.scaler_scope not in {"design_training", "production_full"}:
        raise ModelError("original adequacy artifact has an invalid scaler scope")
    if scope is not expected_scope:
        raise ModelError("parametric-adequacy scope does not match artifact role")
    if original_fit.n_states not in range(2, MAXIMUM_STATE_COUNT + 1):
        raise ModelError("original adequacy artifact has an invalid state count")
    if original_fit.n_features != len(MACRO_FEATURE_NAMES):
        raise ModelError("original adequacy artifact must have four features")
    if original_fit.feature_names != MACRO_FEATURE_NAMES or (
        artifact_features.feature_names != MACRO_FEATURE_NAMES
    ):
        raise ModelError("artifact macro feature order is not canonical")
    if artifact_features.values.shape != (
        original_fit.n_observations,
        original_fit.n_features,
    ):
        raise ModelError("artifact feature geometry does not match original fit")
    if artifact_features.lengths != original_fit.lengths:
        raise ModelError("artifact segment geometry does not match original fit")
    if len(artifact_features.row_labels) != original_fit.n_observations:
        raise ModelError("artifact row-label geometry does not match original fit")


def _validate_refit_output(
    refit: GaussianHMMFit,
    original: GaussianHMMFit,
    features: HMMFeatureMatrix,
    expected_seeds: tuple[int, ...],
) -> None:
    if not isinstance(refit, GaussianHMMFit):
        raise ModelError("injected refit function returned an invalid fit record")
    if refit.n_states != original.n_states or refit.n_features != original.n_features:
        raise ModelError("parametric-adequacy refit changed HMM dimensions")
    if refit.n_observations != original.n_observations:
        raise ModelError("parametric-adequacy refit changed artifact row count")
    if refit.lengths != original.lengths or refit.lengths != features.lengths:
        raise ModelError("parametric-adequacy refit changed segment geometry")
    if refit.feature_names != original.feature_names:
        raise ModelError("parametric-adequacy refit changed macro feature order")
    if np.asarray(refit.decoded_states).shape != (refit.n_observations,):
        raise ModelError("parametric-adequacy refit returned an invalid decoded path")
    if len(refit.restart_diagnostics) != HMM_RESTARTS:
        raise ModelError("parametric-adequacy refit did not audit exactly 50 starts")
    restart_ids = tuple(item.restart_index for item in refit.restart_diagnostics)
    restart_seeds = tuple(item.seed for item in refit.restart_diagnostics)
    if restart_ids != tuple(range(HMM_RESTARTS)) or restart_seeds != expected_seeds:
        raise ModelError("parametric-adequacy restart audit does not match seeds")
    if not 0 <= refit.best_restart_index < HMM_RESTARTS:
        raise ModelError("parametric-adequacy best restart ID is out of range")
    best = refit.restart_diagnostics[refit.best_restart_index]
    if (
        not best.valid
        or best.restart_index != refit.best_restart_index
        or best.seed != refit.best_seed
    ):
        raise ModelError("parametric-adequacy best restart audit is inconsistent")
    if not effective_identification_passed(refit):
        raise ModelError("parametric-adequacy refit failed identification gates")
    if not (
        refit.covariance_diagnostic.positive_definite
        and refit.covariance_diagnostic.eigenvalue_floor_passed
        and refit.covariance_diagnostic.condition_number_passed
    ):
        raise ModelError("parametric-adequacy refit failed covariance gates")
    validate_numerical_chain(refit.parameters.transition_matrix)
    if not effective_transition_support_passed(refit):
        raise ModelError("parametric-adequacy refit failed historical support gates")
    if not np.array_equal(refit.scaler_means, features.scaler_means) or not (
        np.array_equal(
            refit.scaler_standard_deviations,
            features.scaler_standard_deviations,
        )
    ):
        raise ModelError("parametric-adequacy refit scaler evidence is inconsistent")


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if len(seeds) != HMM_RESTARTS:
        raise ModelError("parametric adequacy requires exactly 50 HMM seeds")
    checked: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise ModelError("parametric-adequacy seeds must be uint32 integers")
        value = int(seed)
        if not 0 <= value <= np.iinfo(np.uint32).max:
            raise ModelError("parametric-adequacy seeds must be uint32 integers")
        checked.append(value)
    if len(set(checked)) != HMM_RESTARTS:
        raise ModelError("parametric-adequacy HMM seeds must be distinct")
    return tuple(checked)


def _validated_parameters(
    parameters: HMMParameters,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if not isinstance(parameters, HMMParameters):
        raise ModelError("tied-HMM parameters must be HMMParameters")
    start = np.asarray(parameters.start_probabilities, dtype=np.float64)
    transition = np.asarray(parameters.transition_matrix, dtype=np.float64)
    means = np.asarray(parameters.means, dtype=np.float64)
    covariance = np.asarray(parameters.tied_covariance, dtype=np.float64)
    if start.ndim != 1 or len(start) < 2:
        raise ModelError("HMM start probabilities must contain at least two states")
    n_states = len(start)
    if transition.shape != (n_states, n_states):
        raise ModelError("HMM transition matrix shape is inconsistent")
    if means.ndim != 2 or means.shape[0] != n_states or means.shape[1] == 0:
        raise ModelError("HMM mean matrix shape is inconsistent")
    if covariance.shape != (means.shape[1], means.shape[1]):
        raise ModelError("HMM tied covariance shape is inconsistent")
    if not all(
        np.isfinite(array).all() for array in (start, transition, means, covariance)
    ):
        raise ModelError("HMM parameters contain non-finite values")
    if bool((start < 0.0).any()) or abs(float(start.sum()) - 1.0) > 1e-12:
        raise ModelError("HMM start probabilities are invalid")
    if (
        bool((transition < 0.0).any())
        or float(np.max(np.abs(transition.sum(axis=1) - 1.0))) > 1e-12
    ):
        raise ModelError("HMM transition probabilities are invalid")
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    if symmetry_error > 1e-12:
        raise ModelError("HMM tied covariance is not symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as error:
        raise ModelError("HMM tied covariance eigendecomposition failed") from error
    if not np.isfinite(eigenvalues).all() or bool((eigenvalues <= 0.0).any()):
        raise ModelError("HMM tied covariance is not positive definite")
    return start, transition, means, covariance


def _validated_lengths(
    lengths: Sequence[int], *, expected_rows: int | None = None
) -> tuple[int, ...]:
    if not lengths:
        raise ModelError("artifact segment lengths must be non-empty")
    checked: list[int] = []
    for length in lengths:
        if isinstance(length, bool) or not isinstance(length, int | np.integer):
            raise ModelError("artifact segment lengths must be positive integers")
        value = int(length)
        if value <= 0:
            raise ModelError("artifact segment lengths must be positive integers")
        checked.append(value)
    if expected_rows is not None and sum(checked) != expected_rows:
        raise ModelError("artifact segment lengths do not sum to the row count")
    return tuple(checked)


def _categorical_from_uniform(probabilities: FloatArray, uniform: float) -> int:
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    state = int(np.searchsorted(cumulative, uniform, side="right"))
    if not 0 <= state < len(probabilities):  # pragma: no cover - guarded above.
        raise ModelError("state uniform produced an out-of-range state")
    return state


def _uniform_vector(values: ArrayLike, length: int, label: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.shape != (length,):
        raise ModelError(
            f"{label} has the wrong exact shape",
            details={"actual": raw.shape, "expected": (length,)},
        )
    if raw.dtype.kind == "b":
        raise ModelError(f"{label} must be numeric non-boolean values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if (
        not np.isfinite(result).all()
        or bool((result < 0.0).any())
        or bool((result >= 1.0).any())
    ):
        raise ModelError(f"{label} must be finite and lie in [0, 1)")
    return result


def _normal_matrix(
    values: ArrayLike, expected_shape: tuple[int, int], label: str
) -> FloatArray:
    raw = np.asarray(values)
    if raw.shape != expected_shape:
        raise ModelError(
            f"{label} has the wrong exact shape",
            details={"actual": raw.shape, "expected": expected_shape},
        )
    if raw.dtype.kind == "b":
        raise ModelError(f"{label} must be numeric non-boolean values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if not np.isfinite(result).all():
        raise ModelError(f"{label} contains non-finite values")
    return result


def _finite_matrix(values: ArrayLike, label: str) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ModelError(f"{label} must be a non-empty matrix")
    if not np.isfinite(result).all():
        raise ModelError(f"{label} contains non-finite values")
    return result


def _finite_vector(values: ArrayLike, length: int, label: str) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ModelError(f"{label} must be a finite vector of length {length}")
    return result


def _state_vector(values: ArrayLike, n_states: int) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ModelError("decoded states must be a one-dimensional integer vector")
    if bool((raw < 0).any()) or bool((raw >= n_states).any()):
        raise ModelError("decoded states contain an out-of-range state")
    return raw.astype(np.int64, copy=False)


def _continuation_from_lengths(lengths: tuple[int, ...]) -> BoolArray:
    mask = np.ones(sum(lengths), dtype=np.bool_)
    offset = 0
    for length in lengths:
        mask[offset] = False
        offset += length
    _readonly(mask)
    return mask


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
