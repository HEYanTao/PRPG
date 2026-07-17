from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

import prpg.model.refit_adequacy as refit_adequacy
from prpg.errors import ModelError
from prpg.model.adequacy import (
    EndpointDirection,
    marginal_moments,
    predictive_mixture_diagnostics,
    residual_acf_diagnostics,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    RestartDiagnostic,
    decode_viterbi,
    forward_backward,
    gaussian_hmm_bic,
    gaussian_hmm_parameter_count,
    historical_transition_support_report,
    posterior_identification_report,
    summarize_state_path,
    tied_covariance_diagnostic,
    validate_numerical_chain,
)
from prpg.model.refit_adequacy import (
    MACRO_FEATURE_NAMES,
    align_refit_to_artifact,
    build_refitted_adequacy_endpoint,
    run_refitted_adequacy_attempt,
    simulate_artifact_macro_history,
)
from prpg.simulation.rng import ModelFitScope


def _parameters(
    *,
    transition: np.ndarray | None = None,
    means: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> HMMParameters:
    return HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5], dtype=np.float64),
        transition_matrix=(
            np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
            if transition is None
            else np.asarray(transition, dtype=np.float64)
        ),
        means=(
            np.asarray(
                [[-4.0, -0.2, 0.1, -0.1], [4.0, 0.2, -0.1, 0.1]],
                dtype=np.float64,
            )
            if means is None
            else np.asarray(means, dtype=np.float64)
        ),
        tied_covariance=(
            np.eye(4, dtype=np.float64) * 0.25
            if covariance is None
            else np.asarray(covariance, dtype=np.float64)
        ),
    )


def _long_history() -> tuple[HMMParameters, np.ndarray, tuple[int, ...], np.ndarray]:
    parameters = _parameters()
    states = np.tile(np.repeat(np.asarray([0, 1], dtype=np.int64), 10), 6)
    rng = np.random.default_rng(907)
    noise = rng.normal(0.0, 0.12, size=(len(states), 4))
    values = parameters.means[states] + noise
    return parameters, values, (60, 60), states


def _fit_record(
    parameters: HMMParameters,
    values: np.ndarray,
    lengths: tuple[int, ...],
    *,
    scaler_scope: str = "design_training",
    scaler_means: np.ndarray | None = None,
    scaler_scales: np.ndarray | None = None,
    restart_seeds: Sequence[int] = tuple(range(50)),
) -> GaussianHMMFit:
    decoded = decode_viterbi(values, parameters, lengths=lengths)
    path = summarize_state_path(decoded, 2, lengths=lengths)
    posterior = forward_backward(values, parameters, lengths=lengths)
    identification = posterior_identification_report(
        posterior.posterior_probabilities,
        decoded,
        parameters,
    )
    support = historical_transition_support_report(
        parameters.transition_matrix,
        path.transition_counts,
        posterior.expected_transition_counts,
    )
    diagnostics = tuple(
        RestartDiagnostic(
            restart_index=restart,
            seed=int(seed),
            converged=True,
            iterations=2,
            log_likelihood=posterior.total_log_likelihood,
            valid=True,
            raw_minimum_eigenvalue=float(
                np.linalg.eigvalsh(parameters.tied_covariance).min()
            ),
            maximum_floor_relative_change=0.0,
            covariance_condition_number=float(
                np.linalg.cond(parameters.tied_covariance)
            ),
            maximum_likelihood_decrease=0.0,
            preliminary_chain_passed=True,
            failure_type=None,
            failure_message=None,
        )
        for restart, seed in enumerate(restart_seeds)
    )
    return GaussianHMMFit(
        parameters=parameters,
        decoded_states=decoded,
        canonical_to_original=np.asarray([0, 1], dtype=np.int64),
        original_to_canonical=np.asarray([0, 1], dtype=np.int64),
        log_likelihood=posterior.total_log_likelihood,
        bic=gaussian_hmm_bic(posterior.total_log_likelihood, len(values), 2, 4),
        parameter_count=gaussian_hmm_parameter_count(2, 4),
        n_observations=len(values),
        n_features=4,
        n_states=2,
        lengths=lengths,
        best_restart_index=0,
        best_seed=0,
        restart_diagnostics=diagnostics,
        covariance_diagnostic=tied_covariance_diagnostic(parameters.tied_covariance),
        state_path_diagnostics=path,
        chain_diagnostics=validate_numerical_chain(parameters.transition_matrix),
        forward_backward=posterior,
        identification=identification,
        historical_transition_support=support,
        scaler_scope=scaler_scope,  # type: ignore[arg-type]
        scaler_means=(
            np.zeros(4, dtype=np.float64) if scaler_means is None else scaler_means
        ),
        scaler_standard_deviations=(
            np.ones(4, dtype=np.float64) if scaler_scales is None else scaler_scales
        ),
        feature_names=MACRO_FEATURE_NAMES,
    )


def _artifact_case(
    *, scaler_scope: str = "design_training"
) -> tuple[
    HMMParameters,
    GaussianHMMFit,
    HMMFeatureMatrix,
    np.ndarray,
    np.ndarray,
]:
    parameters, values, lengths, states = _long_history()
    original_fit = _fit_record(parameters, values, lengths, scaler_scope=scaler_scope)
    features = HMMFeatureMatrix(
        values=values,
        lengths=lengths,
        feature_names=MACRO_FEATURE_NAMES,
        row_labels=tuple(f"row-{row}" for row in range(len(values))),
        scaler_means=np.zeros(4, dtype=np.float64),
        scaler_standard_deviations=np.ones(4, dtype=np.float64),
        scaler_scope=scaler_scope,  # type: ignore[arg-type]
    )
    uniforms = _state_uniforms_for_path(states, lengths)
    normals = np.random.default_rng(445).normal(0.0, 0.12, size=values.shape)
    return parameters, original_fit, features, uniforms, normals


def _refit_for_features(
    source_parameters: HMMParameters,
    features: HMMFeatureMatrix,
    seeds: Sequence[int],
) -> GaussianHMMFit:
    means = (
        source_parameters.means - features.scaler_means
    ) / features.scaler_standard_deviations
    inverse_scales = 1.0 / features.scaler_standard_deviations
    covariance = (
        inverse_scales[:, None]
        * source_parameters.tied_covariance
        * inverse_scales[None, :]
    )
    parameters = HMMParameters(
        source_parameters.start_probabilities.copy(),
        source_parameters.transition_matrix.copy(),
        means,
        covariance,
    )
    return _fit_record(
        parameters,
        features.values,
        features.lengths,
        scaler_scope=features.scaler_scope,
        scaler_means=features.scaler_means,
        scaler_scales=features.scaler_standard_deviations,
        restart_seeds=seeds,
    )


def _parameters_for_state_count(n_states: int) -> HMMParameters:
    start = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    transition = np.full(
        (n_states, n_states),
        0.1 / (n_states - 1),
        dtype=np.float64,
    )
    np.fill_diagonal(transition, 0.9)
    means = np.zeros((n_states, 4), dtype=np.float64)
    means[:, 0] = np.linspace(-8.0, 8.0, n_states)
    return HMMParameters(start, transition, means, np.eye(4, dtype=np.float64) * 0.25)


def test_macro_history_uses_exact_draw_geometry_and_segment_resets() -> None:
    parameters = _parameters(
        transition=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        means=np.asarray([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]),
        covariance=np.eye(4),
    )
    uniforms = np.asarray([0.99, 0.1, 0.9, 0.5, 0.1])
    normals = np.zeros((5, 4), dtype=np.float64)

    result = simulate_artifact_macro_history(
        parameters,
        (3, 2),
        uniforms,
        normals,
    )

    # Stationary law is (0.6, 0.4).  Row 3 therefore resets to state 0;
    # continuing from row-2 state 1 with u=.5 would instead remain state 1.
    assert result.latent_states.tolist() == [1, 0, 1, 0, 0]
    np.testing.assert_array_equal(result.values, parameters.means[result.latent_states])
    assert result.lengths == (3, 2)
    assert result.state_uniforms_consumed == 5
    assert result.standard_normals_per_row == 4
    assert not result.values.flags.writeable
    assert not result.latent_states.flags.writeable


def test_macro_history_is_uniform_deterministic_and_rejects_wrong_draw_shape() -> None:
    parameters = _parameters()
    uniforms = np.linspace(0.01, 0.99, 20)
    normals = np.arange(80, dtype=np.float64).reshape(20, 4) / 100.0
    first = simulate_artifact_macro_history(parameters, (20,), uniforms, normals)
    second = simulate_artifact_macro_history(parameters, (20,), uniforms, normals)
    np.testing.assert_array_equal(first.latent_states, second.latent_states)
    np.testing.assert_array_equal(first.values, second.values)
    with pytest.raises(ModelError, match="wrong exact shape"):
        simulate_artifact_macro_history(
            parameters,
            (20,),
            uniforms,
            np.zeros((20, 3)),
        )
    with pytest.raises(ModelError, match=r"\[0, 1\)"):
        simulate_artifact_macro_history(
            parameters,
            (20,),
            np.ones(20),
            normals,
        )


def test_refit_alignment_uses_original_coordinates_and_reorders_every_axis() -> None:
    original = _parameters(
        means=np.asarray([[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
        covariance=np.eye(4),
    )
    refit = HMMParameters(
        start_probabilities=np.asarray([0.4, 0.6]),
        transition_matrix=np.asarray([[0.7, 0.3], [0.2, 0.8]]),
        means=np.asarray([[0.5, 0.0, 0.0, 0.0], [-1.5, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )
    result = align_refit_to_artifact(
        original,
        refit,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([2.0, 1.0, 1.0, 1.0]),
        decoded_states=np.asarray([0, 1, 0, 1]),
    )

    assert result.reference_to_candidate.tolist() == [1, 0]
    assert result.candidate_to_reference.tolist() == [1, 0]
    assert result.aligned_decoded_states is not None
    assert result.aligned_decoded_states.tolist() == [1, 0, 1, 0]
    np.testing.assert_array_equal(
        result.parameters.start_probabilities,
        np.asarray([0.6, 0.4]),
    )
    np.testing.assert_array_equal(
        result.parameters.transition_matrix,
        np.asarray([[0.8, 0.2], [0.3, 0.7]]),
    )
    np.testing.assert_array_equal(
        result.parameters_in_original_coordinates.means,
        original.means,
    )
    assert result.selected_total_cost == pytest.approx(0.0)
    assert result.squared_centroid_costs.shape == (2, 2)


def test_refit_alignment_exact_tie_uses_canonical_permutation_order() -> None:
    tied_means = np.zeros((2, 4), dtype=np.float64)
    result = align_refit_to_artifact(
        _parameters(means=tied_means),
        _parameters(means=tied_means),
        np.zeros(4),
        np.ones(4),
    )
    assert result.reference_to_candidate.tolist() == [0, 1]
    assert result.candidate_to_reference.tolist() == [0, 1]


def test_endpoint_has_exact_order_dimensions_and_directions() -> None:
    parameters, values, lengths, _states = _long_history()
    result = build_refitted_adequacy_endpoint(values, parameters, lengths)
    labels = result.vector.labels

    assert len(labels) == 128
    assert result.vector.values.shape == (128,)
    assert len(result.directions) == 128
    assert labels[:4] == (
        "decoded/occupancy/0",
        "decoded/occupancy/1",
        "decoded/transition/0/1",
        "decoded/transition/1/0",
    )
    assert labels[34:38] == (
        "posterior/normalized_entropy",
        "posterior/mean_maximum",
        "posterior/assigned/0",
        "posterior/assigned/1",
    )
    assert labels[38] == ("predictive_residual/mean/industrial_production_growth")
    assert labels[42] == (
        "predictive_residual/covariance_error/industrial_production_growth/"
        "industrial_production_growth"
    )
    assert (
        labels[51] == "predictive_residual/covariance_error/credit_spread/credit_spread"
    )
    assert labels[52] == (
        "predictive_residual/raw_max_abs_acf/industrial_production_growth"
    )
    assert labels[56] == (
        "predictive_residual/squared_max_abs_acf/industrial_production_growth"
    )
    assert labels[60] == "pit/cvm/industrial_production_growth"
    assert labels[64:68] == (
        "marginal/industrial_production_growth/mean",
        "marginal/industrial_production_growth/variance",
        "marginal/industrial_production_growth/skewness",
        "marginal/industrial_production_growth/kurtosis",
    )
    assert labels[80:84] == (
        "within_state/0/industrial_production_growth/lag_1/raw",
        "within_state/0/industrial_production_growth/lag_1/squared",
        "within_state/0/industrial_production_growth/lag_2/raw",
        "within_state/0/industrial_production_growth/lag_2/squared",
    )
    assert all(
        direction is EndpointDirection.UPPER_TAIL
        for direction in result.directions[52:64]
    )
    assert all(
        direction is EndpointDirection.TWO_SIDED
        for index, direction in enumerate(result.directions)
        if not 52 <= index < 64
    )
    assert result.within_state_pair_counts.shape == (2, 4, 3)
    assert int(result.within_state_pair_counts.min()) >= 10
    assert not result.vector.values.flags.writeable


def test_endpoint_fails_closed_when_within_state_pairs_are_insufficient() -> None:
    parameters = _parameters(
        transition=np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    )
    states = np.tile(np.asarray([0, 1], dtype=np.int64), 30)
    noise = np.random.default_rng(8).normal(0.0, 0.1, size=(len(states), 4))
    values = parameters.means[states] + noise
    with pytest.raises(ModelError, match="insufficient valid pairs"):
        build_refitted_adequacy_endpoint(values, parameters, (60,))


def _state_uniforms_for_path(
    states: np.ndarray, lengths: tuple[int, ...]
) -> np.ndarray:
    result = np.empty(len(states), dtype=np.float64)
    offset = 0
    for length in lengths:
        for within in range(length):
            row = offset + within
            target = int(states[row])
            if within == 0:
                result[row] = 0.25 if target == 0 else 0.75
            else:
                previous = int(states[row - 1])
                if previous == 0:
                    result[row] = 0.10 if target == 0 else 0.99
                else:
                    result[row] = 0.01 if target == 0 else 0.90
        offset += length
    return result


def test_single_attempt_recomputes_scaler_injects_50_seeds_and_builds_endpoint() -> (
    None
):
    parameters, original_values, lengths, target_states = _long_history()
    original_fit = _fit_record(parameters, original_values, lengths)
    artifact_features = HMMFeatureMatrix(
        values=original_values,
        lengths=lengths,
        feature_names=MACRO_FEATURE_NAMES,
        row_labels=tuple(f"row-{row}" for row in range(len(original_values))),
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        scaler_scope="design_training",
    )
    normals = np.random.default_rng(445).normal(
        0.0, 0.12, size=(len(original_values), 4)
    )
    state_uniforms = _state_uniforms_for_path(target_states, lengths)
    called: list[tuple[int, tuple[int, ...]]] = []

    def fit_function(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: Sequence[int],
    ) -> GaussianHMMFit:
        called.append((n_states, tuple(seeds)))
        means = (
            parameters.means - features.scaler_means
        ) / features.scaler_standard_deviations
        inverse_scales = 1.0 / features.scaler_standard_deviations
        covariance = (
            inverse_scales[:, None]
            * parameters.tied_covariance
            * inverse_scales[None, :]
        )
        refit_parameters = HMMParameters(
            parameters.start_probabilities.copy(),
            parameters.transition_matrix.copy(),
            means,
            covariance,
        )
        return _fit_record(
            refit_parameters,
            features.values,
            features.lengths,
            scaler_means=features.scaler_means,
            scaler_scales=features.scaler_standard_deviations,
            restart_seeds=seeds,
        )

    seeds = tuple(range(50))
    result = run_refitted_adequacy_attempt(
        original_fit,
        artifact_features,
        state_uniforms,
        normals,
        scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
        seeds=seeds,
        fit_function=fit_function,
    )

    assert called == [(2, seeds)]
    assert result.scope is ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY
    assert result.seeds == seeds
    assert result.endpoint.vector.values.shape == (128,)
    np.testing.assert_allclose(
        result.refit_standardized_values.mean(axis=0), 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.refit_standardized_values.std(axis=0, ddof=0), 1.0
    )
    np.testing.assert_allclose(
        result.alignment.parameters_in_original_coordinates.means,
        parameters.means,
    )
    assert result.simulation.state_uniforms_consumed == len(original_values)
    assert result.simulation.standard_normals_per_row == 4


def test_single_attempt_rejects_wrong_scope_and_seed_contract() -> None:
    parameters, values, lengths, _states = _long_history()
    original_fit = _fit_record(parameters, values, lengths)
    artifact_features = HMMFeatureMatrix(
        values=values,
        lengths=lengths,
        feature_names=MACRO_FEATURE_NAMES,
        row_labels=tuple(str(row) for row in range(len(values))),
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        scaler_scope="design_training",
    )
    uniforms = np.full(len(values), 0.1)
    normals = np.zeros_like(values)
    with pytest.raises(ModelError, match="scope does not match"):
        run_refitted_adequacy_attempt(
            original_fit,
            artifact_features,
            uniforms,
            normals,
            scope=ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY,
            seeds=tuple(range(50)),
        )
    with pytest.raises(ModelError, match="exactly 50"):
        run_refitted_adequacy_attempt(
            original_fit,
            artifact_features,
            uniforms,
            normals,
            scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            seeds=tuple(range(49)),
        )


def test_endpoint_values_match_registered_predictive_pit_acf_and_moments() -> None:
    parameters, values, lengths, _states = _long_history()
    endpoint = build_refitted_adequacy_endpoint(values, parameters, lengths)
    continuation = np.ones(len(values), dtype=np.bool_)
    continuation[0] = False
    continuation[lengths[0]] = False
    scores = refit_adequacy.one_step_predictive_log_scores(
        values,
        parameters,
        lengths=lengths,
    )
    predictive = predictive_mixture_diagnostics(
        values,
        scores.predictive_state_probabilities,
        parameters.means,
        parameters.tied_covariance,
    )
    predictive_acf = residual_acf_diagnostics(
        predictive.predictive_residuals,
        continuation,
        refit_adequacy.PREDICTIVE_ACF_LAGS,
        minimum_pairs=refit_adequacy.MINIMUM_ACF_PAIRS,
    )
    moments = marginal_moments(values)

    np.testing.assert_allclose(
        endpoint.vector.values[38:42],
        predictive.predictive_residuals.mean(axis=0),
    )
    np.testing.assert_allclose(
        endpoint.vector.values[42:52],
        predictive.predictive_covariance_error_upper,
    )
    np.testing.assert_allclose(
        endpoint.vector.values[52:56],
        np.max(np.abs(predictive_acf.raw), axis=1),
    )
    np.testing.assert_allclose(
        endpoint.vector.values[56:60],
        np.max(np.abs(predictive_acf.squared), axis=1),
    )
    np.testing.assert_allclose(
        endpoint.vector.values[60:64],
        predictive.pit_cramer_von_mises,
    )
    np.testing.assert_allclose(endpoint.vector.values[64:80], moments.ravel())


def test_endpoint_dimension_tracks_the_fitted_state_count() -> None:
    parameters = _parameters_for_state_count(3)
    states = np.tile(np.repeat(np.arange(3, dtype=np.int64), 15), 4)
    values = parameters.means[states] + np.random.default_rng(81).normal(
        0.0,
        0.08,
        size=(len(states), 4),
    )

    endpoint = build_refitted_adequacy_endpoint(values, parameters, (len(values),))

    expected = 3 * (3 - 1) + 41 * 3 + 44
    assert len(endpoint.vector.labels) == expected == 173
    assert endpoint.vector.values.shape == (expected,)
    assert endpoint.within_state_pair_counts.shape == (3, 4, 3)
    assert endpoint.vector.labels[0:3] == (
        "decoded/occupancy/0",
        "decoded/occupancy/1",
        "decoded/occupancy/2",
    )
    assert endpoint.vector.labels[-1] == "within_state/2/credit_spread/lag_3/squared"


def test_simulation_alignment_and_endpoint_reject_contract_drift() -> None:
    three_features = HMMParameters(
        np.asarray([0.5, 0.5]),
        np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        np.zeros((2, 3), dtype=np.float64),
        np.eye(3, dtype=np.float64),
    )
    with pytest.raises(ModelError, match="exactly four macro features"):
        simulate_artifact_macro_history(
            three_features,
            (2,),
            np.asarray([0.1, 0.2]),
            np.zeros((2, 4)),
        )

    six_states = _parameters_for_state_count(6)
    with pytest.raises(ModelError, match=r"registered range 2\.\.5"):
        align_refit_to_artifact(six_states, six_states, np.zeros(4), np.ones(4))
    with pytest.raises(ModelError, match="dimensions do not match"):
        align_refit_to_artifact(
            _parameters(),
            _parameters_for_state_count(3),
            np.zeros(4),
            np.ones(4),
        )
    with pytest.raises(ModelError, match="standard deviations must be positive"):
        align_refit_to_artifact(
            _parameters(),
            _parameters(),
            np.zeros(4),
            np.asarray([1.0, 1.0, 0.0, 1.0]),
        )

    parameters, values, lengths, _states = _long_history()
    with pytest.raises(ModelError, match="exactly four macro features"):
        build_refitted_adequacy_endpoint(values[:, :3], parameters, lengths)
    with pytest.raises(ModelError, match="feature order is not canonical"):
        build_refitted_adequacy_endpoint(
            values,
            parameters,
            lengths,
            feature_names=tuple(reversed(MACRO_FEATURE_NAMES)),
        )
    with pytest.raises(ModelError, match=r"range 2\.\.5"):
        build_refitted_adequacy_endpoint(values, six_states, lengths)


def test_endpoint_fails_closed_on_posterior_and_component_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters, values, lengths, _states = _long_history()
    fit = _fit_record(parameters, values, lengths)
    failed_identification = replace(
        fit.identification,
        passed=False,
        rejection_reasons=("tampered posterior",),
    )
    monkeypatch.setattr(
        refit_adequacy,
        "posterior_identification_report",
        lambda *_args, **_kwargs: failed_identification,
    )
    with pytest.raises(ModelError, match="posterior-identification gates failed"):
        build_refitted_adequacy_endpoint(values, parameters, lengths)

    unavailable_assignment = replace(
        fit.identification,
        assigned_state_mean_posteriors=np.asarray([np.nan, 0.9]),
    )
    monkeypatch.setattr(
        refit_adequacy,
        "posterior_identification_report",
        lambda *_args, **_kwargs: unavailable_assignment,
    )
    with pytest.raises(ModelError, match="assigned-state summaries are unavailable"):
        build_refitted_adequacy_endpoint(values, parameters, lengths)

    monkeypatch.undo()
    original_moments = marginal_moments(values).copy()
    original_moments[0, 0] = np.nan
    monkeypatch.setattr(
        refit_adequacy,
        "marginal_moments",
        lambda _values: original_moments,
    )
    with pytest.raises(ModelError, match="endpoint vector is invalid"):
        build_refitted_adequacy_endpoint(values, parameters, lengths)

    monkeypatch.undo()
    original_duration = refit_adequacy._decoded_state_duration_values

    def extra_duration_component(
        path: refit_adequacy.StatePathDiagnostic,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        labels, components = original_duration(path)
        return (*labels, "unexpected/component"), (*components, 0.0)

    monkeypatch.setattr(
        refit_adequacy,
        "_decoded_state_duration_values",
        extra_duration_component,
    )
    with pytest.raises(AssertionError, match="endpoint order is inconsistent"):
        build_refitted_adequacy_endpoint(values, parameters, lengths)


def test_decoded_duration_and_within_state_support_fail_closed() -> None:
    parameters, values, lengths, _states = _long_history()
    path = _fit_record(parameters, values, lengths).state_path_diagnostics
    malformed_paths = (
        (
            replace(path, transition_counts=np.zeros((2, 3), dtype=np.int64)),
            "transition-count shape",
        ),
        (
            replace(path, transition_counts=np.zeros((2, 2), dtype=np.int64)),
            "transition denominator is zero",
        ),
        (
            replace(
                path,
                dwell_lengths_by_state=(
                    np.asarray([2.0, 3.0]),
                    path.dwell_lengths_by_state[1],
                ),
            ),
            "must be integer vectors",
        ),
        (
            replace(
                path,
                dwell_lengths_by_state=(
                    np.asarray([], dtype=np.int64),
                    path.dwell_lengths_by_state[1],
                ),
            ),
            "dwell support is unavailable",
        ),
        (
            replace(
                path,
                dwell_lengths_by_state=(
                    np.asarray([0], dtype=np.int64),
                    path.dwell_lengths_by_state[1],
                ),
            ),
            "dwell values must be positive",
        ),
    )
    for malformed, message in malformed_paths:
        with pytest.raises(ModelError, match=message):
            refit_adequacy._decoded_state_duration_values(malformed)

    residuals = np.random.default_rng(31).normal(size=(30, 4))
    with pytest.raises(ModelError, match="support is unavailable"):
        refit_adequacy._within_state_acfs(
            residuals,
            np.zeros(30, dtype=np.int64),
            np.ones(30, dtype=np.bool_),
            n_states=2,
        )
    with pytest.raises(ModelError, match="denominator is unavailable"):
        refit_adequacy._cosine_acf(np.zeros(12), np.zeros(12))


def test_covariance_and_simulation_numerical_backstops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = _parameters()
    uniforms = np.linspace(0.01, 0.99, 20)
    normals = np.zeros((20, 4), dtype=np.float64)

    def cholesky_failure(_covariance: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(refit_adequacy.np.linalg, "cholesky", cholesky_failure)
    with pytest.raises(ModelError, match="Cholesky failed during simulation"):
        simulate_artifact_macro_history(parameters, (20,), uniforms, normals)

    monkeypatch.setattr(
        refit_adequacy.np.linalg,
        "cholesky",
        lambda covariance: np.zeros_like(covariance),
    )
    with pytest.raises(ModelError, match="not positive diagonal"):
        simulate_artifact_macro_history(parameters, (20,), uniforms, normals)

    monkeypatch.undo()
    huge_parameters = _parameters(covariance=np.eye(4, dtype=np.float64) * 1e300)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ModelError, match="non-finite values"),
    ):
        simulate_artifact_macro_history(
            huge_parameters,
            (20,),
            uniforms,
            np.full((20, 4), 1e308),
        )

    monkeypatch.setattr(
        refit_adequacy.np.linalg,
        "eigh",
        lambda _covariance: (_ for _ in ()).throw(np.linalg.LinAlgError("injected")),
    )
    with pytest.raises(ModelError, match="eigendecomposition failed"):
        refit_adequacy._symmetric_inverse_square_root(np.eye(4))

    monkeypatch.undo()
    with pytest.raises(ModelError, match="not positive definite"):
        refit_adequacy._symmetric_inverse_square_root(
            np.diag(np.asarray([1.0, 1.0, 1.0, -1.0]))
        )

    def nonfinite_inverse_root(
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(len(covariance)), np.full_like(covariance, 1e308)

    monkeypatch.setattr(refit_adequacy.np.linalg, "eigh", nonfinite_inverse_root)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ModelError, match="inverse square root is non-finite"),
    ):
        refit_adequacy._symmetric_inverse_square_root(np.eye(4))

    monkeypatch.undo()
    monkeypatch.setattr(refit_adequacy.math, "sqrt", lambda _value: 1.0)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ModelError, match="ACF is non-finite"),
    ):
        refit_adequacy._cosine_acf(np.full(2, 1e308), np.full(2, 1e308))


def test_hmm_parameter_validation_rejects_tampered_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _parameters()
    invalid_parameters = (
        (object(), "must be HMMParameters"),
        (
            replace(valid, start_probabilities=np.asarray([1.0])),
            "at least two states",
        ),
        (
            replace(valid, transition_matrix=np.eye(3)),
            "transition matrix shape",
        ),
        (replace(valid, means=np.zeros(2)), "mean matrix shape"),
        (replace(valid, tied_covariance=np.eye(3)), "covariance shape"),
        (
            replace(
                valid,
                means=np.asarray([[-4.0, np.nan, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]),
            ),
            "contain non-finite",
        ),
        (
            replace(valid, start_probabilities=np.asarray([1.1, -0.1])),
            "start probabilities are invalid",
        ),
        (
            replace(valid, start_probabilities=np.asarray([0.4, 0.4])),
            "start probabilities are invalid",
        ),
        (
            replace(
                valid,
                transition_matrix=np.asarray([[-0.1, 1.1], [0.1, 0.9]]),
            ),
            "transition probabilities are invalid",
        ),
        (
            replace(
                valid,
                transition_matrix=np.asarray([[0.8, 0.1], [0.1, 0.9]]),
            ),
            "transition probabilities are invalid",
        ),
        (
            replace(
                valid,
                tied_covariance=np.asarray(
                    [
                        [1.0, 0.1, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ),
            "covariance is not symmetric",
        ),
        (
            replace(valid, tied_covariance=np.diag([1.0, 1.0, 1.0, -1.0])),
            "covariance is not positive definite",
        ),
    )
    for malformed, message in invalid_parameters:
        with pytest.raises(ModelError, match=message):
            refit_adequacy._validated_parameters(malformed)  # type: ignore[arg-type]

    monkeypatch.setattr(
        refit_adequacy.np.linalg,
        "eigvalsh",
        lambda _covariance: (_ for _ in ()).throw(np.linalg.LinAlgError("injected")),
    )
    with pytest.raises(ModelError, match="eigendecomposition failed"):
        refit_adequacy._validated_parameters(valid)


def test_array_and_seed_boundary_validation_is_fail_closed() -> None:
    seeds = tuple(range(50))
    assert refit_adequacy._validated_seeds(seeds) == seeds
    invalid_seed_sets = (
        ((True, *range(1, 50)), "uint32 integers"),
        ((1.5, *range(1, 50)), "uint32 integers"),
        ((-1, *range(1, 50)), "uint32 integers"),
        ((2**32, *range(1, 50)), "uint32 integers"),
        ((0, 0, *range(2, 50)), "must be distinct"),
    )
    for malformed, message in invalid_seed_sets:
        with pytest.raises(ModelError, match=message):
            refit_adequacy._validated_seeds(malformed)  # type: ignore[arg-type]

    for lengths in ((), (True,), (1.5,), (0,), (-1,)):
        with pytest.raises(ModelError, match="segment lengths"):
            refit_adequacy._validated_lengths(lengths)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="do not sum"):
        refit_adequacy._validated_lengths((2, 3), expected_rows=6)

    with pytest.raises(ModelError, match="non-boolean"):
        refit_adequacy._uniform_vector(np.ones(3, dtype=np.bool_), 3, "uniforms")
    with pytest.raises(ModelError, match="must be numeric"):
        refit_adequacy._uniform_vector(["x"], 1, "uniforms")
    for values in ([np.nan], [-0.1], [1.0]):
        with pytest.raises(ModelError, match=r"\[0, 1\)"):
            refit_adequacy._uniform_vector(values, 1, "uniforms")
    with pytest.raises(ModelError, match="wrong exact shape"):
        refit_adequacy._uniform_vector([0.1, 0.2], 1, "uniforms")

    with pytest.raises(ModelError, match="non-boolean"):
        refit_adequacy._normal_matrix(
            np.ones((2, 4), dtype=np.bool_),
            (2, 4),
            "normals",
        )
    with pytest.raises(ModelError, match="must be numeric"):
        refit_adequacy._normal_matrix([["x"]], (1, 1), "normals")
    with pytest.raises(ModelError, match="non-finite"):
        refit_adequacy._normal_matrix([[np.nan]], (1, 1), "normals")

    with pytest.raises(ModelError, match="must be numeric"):
        refit_adequacy._finite_matrix([["x"]], "matrix")
    with pytest.raises(ModelError, match="non-empty matrix"):
        refit_adequacy._finite_matrix([], "matrix")
    with pytest.raises(ModelError, match="non-finite"):
        refit_adequacy._finite_matrix([[np.nan]], "matrix")
    with pytest.raises(ModelError, match="must be numeric"):
        refit_adequacy._finite_vector(["x"], 1, "vector")
    with pytest.raises(ModelError, match="finite vector"):
        refit_adequacy._finite_vector([np.nan], 1, "vector")

    with pytest.raises(ModelError, match="integer vector"):
        refit_adequacy._state_vector(np.asarray([0.0, 1.0]), 2)
    for values in (np.asarray([-1, 0]), np.asarray([0, 2])):
        with pytest.raises(ModelError, match="out-of-range state"):
            refit_adequacy._state_vector(values, 2)


def test_attempt_input_envelope_rejects_geometry_and_role_tampering() -> None:
    _parameters_value, original, features, _uniforms, _normals = _artifact_case()
    refit_adequacy._validate_attempt_inputs(
        original,
        features,
        ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
    )
    production_original = replace(original, scaler_scope="production_full")
    production_features = replace(features, scaler_scope="production_full")
    refit_adequacy._validate_attempt_inputs(
        production_original,
        production_features,
        ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY,
    )

    malformed_cases = (
        (
            object(),
            features,
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "must be a GaussianHMMFit",
        ),
        (
            original,
            object(),
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "must be an HMMFeatureMatrix",
        ),
        (original, features, "design", "must be a ModelFitScope"),
        (
            replace(original, scaler_scope="invalid"),
            features,
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "invalid scaler scope",
        ),
        (
            replace(original, n_states=6),
            features,
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "invalid state count",
        ),
        (
            replace(original, n_features=3),
            features,
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "must have four features",
        ),
        (
            replace(original, feature_names=tuple(reversed(MACRO_FEATURE_NAMES))),
            features,
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "order is not canonical",
        ),
        (
            original,
            replace(features, feature_names=tuple(reversed(MACRO_FEATURE_NAMES))),
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "order is not canonical",
        ),
        (
            original,
            replace(features, values=features.values[:-1]),
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "feature geometry",
        ),
        (
            original,
            replace(features, lengths=(len(features.values),)),
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "segment geometry",
        ),
        (
            original,
            replace(features, row_labels=features.row_labels[:-1]),
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            "row-label geometry",
        ),
    )
    for malformed_original, malformed_features, scope, message in malformed_cases:
        with pytest.raises(ModelError, match=message):
            refit_adequacy._validate_attempt_inputs(  # type: ignore[arg-type]
                malformed_original,
                malformed_features,
                scope,
            )


def test_refit_output_envelope_rejects_every_audited_tamper() -> None:
    parameters, original, features, _uniforms, _normals = _artifact_case()
    seeds = tuple(range(50))
    valid = _refit_for_features(parameters, features, seeds)
    refit_adequacy._validate_refit_output(valid, original, features, seeds)

    bad_restart_index = replace(valid.restart_diagnostics[0], restart_index=9)
    bad_restart_seed = replace(valid.restart_diagnostics[0], seed=999)
    invalid_best = replace(valid.restart_diagnostics[0], valid=False)
    failed_identification = replace(
        valid.identification,
        passed=False,
        rejection_reasons=("injected",),
    )
    failed_support = replace(
        valid.historical_transition_support,
        passed=False,
        rejection_reasons=("injected",),
    )
    bad_chain_parameters = replace(
        valid.parameters,
        transition_matrix=np.eye(2, dtype=np.float64),
    )
    malformed_cases = (
        (object(), "invalid fit record"),
        (replace(valid, n_states=3), "changed HMM dimensions"),
        (replace(valid, n_features=3), "changed HMM dimensions"),
        (replace(valid, n_observations=119), "changed artifact row count"),
        (replace(valid, lengths=(120,)), "changed segment geometry"),
        (
            replace(valid, feature_names=tuple(reversed(MACRO_FEATURE_NAMES))),
            "changed macro feature order",
        ),
        (
            replace(valid, decoded_states=valid.decoded_states[:-1]),
            "invalid decoded path",
        ),
        (
            replace(valid, restart_diagnostics=valid.restart_diagnostics[:-1]),
            "exactly 50 starts",
        ),
        (
            replace(
                valid,
                restart_diagnostics=(
                    bad_restart_index,
                    *valid.restart_diagnostics[1:],
                ),
            ),
            "restart audit does not match seeds",
        ),
        (
            replace(
                valid,
                restart_diagnostics=(bad_restart_seed, *valid.restart_diagnostics[1:]),
            ),
            "restart audit does not match seeds",
        ),
        (replace(valid, best_restart_index=50), "best restart ID is out of range"),
        (
            replace(
                valid,
                restart_diagnostics=(invalid_best, *valid.restart_diagnostics[1:]),
            ),
            "best restart audit is inconsistent",
        ),
        (replace(valid, best_seed=999), "best restart audit is inconsistent"),
        (
            replace(valid, identification=failed_identification),
            "failed identification gates",
        ),
        (
            replace(
                valid,
                covariance_diagnostic=replace(
                    valid.covariance_diagnostic,
                    positive_definite=False,
                ),
            ),
            "failed covariance gates",
        ),
        (
            replace(
                valid,
                covariance_diagnostic=replace(
                    valid.covariance_diagnostic,
                    eigenvalue_floor_passed=False,
                ),
            ),
            "failed covariance gates",
        ),
        (
            replace(
                valid,
                covariance_diagnostic=replace(
                    valid.covariance_diagnostic,
                    condition_number_passed=False,
                ),
            ),
            "failed covariance gates",
        ),
        (replace(valid, parameters=bad_chain_parameters), "reducible"),
        (
            replace(valid, historical_transition_support=failed_support),
            "failed historical support gates",
        ),
        (
            replace(valid, scaler_means=np.ones(4)),
            "scaler evidence is inconsistent",
        ),
        (
            replace(valid, scaler_standard_deviations=np.full(4, 2.0)),
            "scaler evidence is inconsistent",
        ),
    )
    for malformed, message in malformed_cases:
        with pytest.raises(ModelError, match=message):
            refit_adequacy._validate_refit_output(  # type: ignore[arg-type]
                malformed,
                original,
                features,
                seeds,
            )

    mismatched_features = replace(features, lengths=(120,))
    with pytest.raises(ModelError, match="changed segment geometry"):
        refit_adequacy._validate_refit_output(
            valid,
            original,
            mismatched_features,
            seeds,
        )


def test_attempt_scaler_and_decoder_backstops_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters, original, features, uniforms, normals = _artifact_case()
    seeds = tuple(range(50))

    with pytest.raises(ModelError, match="scaler is non-positive"):
        run_refitted_adequacy_attempt(
            original,
            features,
            np.full(len(features.values), 0.01),
            np.zeros_like(features.values),
            scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            seeds=seeds,
        )

    simulation = simulate_artifact_macro_history(
        parameters,
        original.lengths,
        uniforms,
        normals,
    )
    nonfinite_values = simulation.values.copy()
    nonfinite_values[0, 0] = np.nan
    monkeypatch.setattr(
        refit_adequacy,
        "simulate_artifact_macro_history",
        lambda *_args, **_kwargs: replace(simulation, values=nonfinite_values),
    )
    with pytest.raises(ModelError, match="scaler is non-finite"):
        run_refitted_adequacy_attempt(
            original,
            features,
            uniforms,
            normals,
            scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            seeds=seeds,
        )

    monkeypatch.undo()

    class NonfiniteStandardization(np.ndarray):
        def mean(self, *_args: object, **_kwargs: object) -> np.ndarray:
            return np.zeros(self.shape[1], dtype=np.float64)

        def std(self, *_args: object, **_kwargs: object) -> np.ndarray:
            return np.ones(self.shape[1], dtype=np.float64)

        def __sub__(self, _other: object) -> np.ndarray:
            return np.full(self.shape, np.nan, dtype=np.float64)

    adversarial_values = simulation.values.copy().view(NonfiniteStandardization)
    monkeypatch.setattr(
        refit_adequacy,
        "simulate_artifact_macro_history",
        lambda *_args, **_kwargs: replace(simulation, values=adversarial_values),
    )
    with pytest.raises(ModelError, match="standardized macro history is non-finite"):
        run_refitted_adequacy_attempt(
            original,
            features,
            uniforms,
            normals,
            scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            seeds=seeds,
        )

    monkeypatch.undo()

    def inconsistent_decoder(
        refit_features: HMMFeatureMatrix,
        _n_states: int,
        restart_seeds: Sequence[int],
    ) -> GaussianHMMFit:
        valid = _refit_for_features(parameters, refit_features, restart_seeds)
        return replace(valid, decoded_states=1 - valid.decoded_states)

    with pytest.raises(ModelError, match="decoder is inconsistent"):
        run_refitted_adequacy_attempt(
            original,
            features,
            uniforms,
            normals,
            scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            seeds=seeds,
            fit_function=inconsistent_decoder,
        )
