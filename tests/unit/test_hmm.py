from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from prpg.data.transform import StandardizedFeatures, standardize_features
from prpg.errors import IntegrityError, ModelError
from prpg.model.hmm import (
    HMM_RESTARTS,
    HMMCandidateInadmissibility,
    HMMFeatureMatrix,
    HMMParameters,
    HMMRestartNumericalFailure,
    NoNumericallyValidHMMRestarts,
    canonicalize_hmm,
    consume_standardized_features,
    covariance_diagnostics,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_restarts,
    forward_backward,
    gaussian_hmm_bic,
    gaussian_hmm_parameter_count,
    historical_transition_support_report,
    one_step_predictive_log_scores,
    posterior_identification_report,
    sequence_lengths_from_continuation,
    summarize_state_path,
    tied_covariance_diagnostic,
    validate_historical_transition_support,
    validate_markov_chain,
    validate_numerical_chain,
)
from prpg.model.materialization_identity import gaussian_hmm_fit_fingerprint
from prpg.model.regime_policy import (
    OWNER_FIXED_K4_PROSPECTIVE_POLICY,
    OwnerFixedK4RecurringHistoryFailure,
    effective_identification_passed,
    effective_transition_support_passed,
    owner_fixed_k4_policy_for_scientific_version,
    owner_fixed_k4_recurring_history_support,
)
from prpg.simulation.rng import ModelFitScope, hmm_library_seed


def _raw_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "growth": [-2.0, -1.0, 1.0, 2.0, 6.0, 8.0],
            "inflation": [3.0, -1.0, -2.0, 0.0, 5.0, -7.0],
        },
        index=pd.period_range("2024-01", periods=6, freq="M"),
    )


def _production_features() -> HMMFeatureMatrix:
    standardized = standardize_features(_raw_features())
    return consume_standardized_features(
        standardized,
        continuation=np.asarray([False, True, True, False, True, True]),
        scaler_scope="production_full",
    )


def _parameters() -> HMMParameters:
    return HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        means=np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
        tied_covariance=np.eye(2),
    )


def _fit_features(*, single_switch: bool = False) -> HMMFeatureMatrix:
    if single_switch:
        latent = np.repeat(np.asarray([0, 1], dtype=np.int64), 50)
        lengths = (100,)
    else:
        latent = np.tile(np.repeat(np.asarray([0, 1]), 5), 10)
        lengths = (50, 50)
    values = np.column_stack(
        (
            np.where(latent == 0, -2.0, 2.0),
            np.where(latent == 0, -2.0, 2.0),
        )
    )
    return HMMFeatureMatrix(
        values=values,
        lengths=lengths,
        feature_names=("growth", "inflation"),
        row_labels=tuple(str(value) for value in range(len(values))),
        scaler_means=np.zeros(2),
        scaler_standard_deviations=np.ones(2),
        scaler_scope="production_full",
    )


def _fit_features_k4() -> HMMFeatureMatrix:
    latent = np.tile(np.repeat(np.arange(4, dtype=np.int64), 4), 8)
    means = np.asarray([[-3.0, -3.0], [-1.0, -1.0], [1.0, 1.0], [3.0, 3.0]])
    values = means[latent]
    return HMMFeatureMatrix(
        values=values,
        lengths=(64, 64),
        feature_names=("growth", "inflation"),
        row_labels=tuple(str(value) for value in range(len(values))),
        scaler_means=np.zeros(2),
        scaler_standard_deviations=np.ones(2),
        scaler_scope="production_full",
    )


def _k4_factory() -> _FakeFactory:
    return _FakeFactory(
        default_outcome={
            "start": [0.25, 0.25, 0.25, 0.25],
            "transition": [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ],
            "means": [
                [-3.0, -3.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [3.0, 3.0],
            ],
        }
    )


def test_continuation_mask_maps_exactly_to_sequence_lengths() -> None:
    assert sequence_lengths_from_continuation(
        np.asarray([False, True, False, True, True], dtype=np.bool_)
    ) == (2, 3)
    assert sequence_lengths_from_continuation(
        np.asarray([False, False, False], dtype=np.bool_)
    ) == (1, 1, 1)


@pytest.mark.parametrize(
    "mask",
    [
        np.asarray([], dtype=np.bool_),
        np.asarray([[False, True]], dtype=np.bool_),
        np.asarray([True, False], dtype=np.bool_),
        np.asarray([0, 1], dtype=np.int64),
    ],
)
def test_continuation_mask_validation_fails_closed(mask: NDArray[Any]) -> None:
    with pytest.raises(ModelError, match="continuation"):
        sequence_lengths_from_continuation(mask)


def test_scaler_consumption_honors_scope_subset_and_source_gaps() -> None:
    raw = _raw_features()
    training = raw.index[:4]
    standardized = standardize_features(raw, training_index=training)

    result = consume_standardized_features(
        standardized,
        continuation=np.asarray([False, True, True, False, True, True]),
        scaler_scope="design_training",
        fit_index=training,
    )

    assert result.values.shape == (4, 2)
    assert result.lengths == (3, 1)
    assert result.row_labels == tuple(str(value) for value in training)
    np.testing.assert_allclose(result.values.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(result.values.std(axis=0), 1.0, atol=1e-15)


def test_scaler_subset_adds_break_for_nonadjacent_rows() -> None:
    raw = _raw_features()
    selected = raw.index[[0, 1, 4, 5]]
    standardized = standardize_features(raw, training_index=selected)
    result = consume_standardized_features(
        standardized,
        continuation=np.asarray([False, True, True, True, True, True]),
        scaler_scope="design_training",
        fit_index=selected,
    )
    assert result.lengths == (2, 2)


def test_production_scaler_consumes_full_matrix() -> None:
    result = _production_features()
    assert result.values.shape == (6, 2)
    assert result.lengths == (3, 3)
    assert result.scaler_scope == "production_full"


@pytest.mark.parametrize(
    ("scope", "fit_index", "message"),
    [
        ("production_full", pd.Index([0]), "forbids a row subset"),
        ("design_training", None, "requires fit_index"),
        ("design_training", pd.Index([]), "non-empty and unique"),
        ("design_training", pd.Index(["missing"]), "unknown rows"),
    ],
)
def test_scaler_scope_and_index_contracts(
    scope: str, fit_index: pd.Index | None, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        consume_standardized_features(
            standardize_features(_raw_features()),
            continuation=np.asarray([False, True, True, False, True, True]),
            scaler_scope=scope,  # type: ignore[arg-type]
            fit_index=fit_index,
        )


def test_scaler_consumption_detects_leakage_alignment_and_nonfinite_values() -> None:
    raw = _raw_features()
    standardized = standardize_features(raw)
    with pytest.raises(ModelError, match="inconsistent with the declared scaler"):
        consume_standardized_features(
            standardized,
            continuation=np.asarray([False, True, True, False, True, True]),
            scaler_scope="design_training",
            fit_index=raw.index[:4],
        )
    misaligned = StandardizedFeatures(
        values=standardized.values,
        means=standardized.means.rename(index={"growth": "wrong"}),
        standard_deviations=standardized.standard_deviations,
    )
    with pytest.raises(ModelError, match="do not align"):
        consume_standardized_features(
            misaligned,
            continuation=np.asarray([False, True, True, False, True, True]),
            scaler_scope="production_full",
        )
    standardized.values.iloc[0, 0] = np.nan
    with pytest.raises(ModelError, match="non-finite"):
        consume_standardized_features(
            standardized,
            continuation=np.asarray([False, True, True, False, True, True]),
            scaler_scope="production_full",
        )


def test_deterministic_restart_seeds_use_scoped_rng_registry() -> None:
    seeds = deterministic_hmm_restart_seeds(
        master_seed=123456,
        scientific_version=7,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        n_states=3,
    )
    assert len(seeds) == HMM_RESTARTS == 50
    assert len(set(seeds)) == 50
    assert seeds[17] == hmm_library_seed(
        master_seed=123456,
        scientific_version=7,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        k=3,
        restart_index=17,
    )


def test_tied_parameter_count_and_bic_are_exact() -> None:
    assert [gaussian_hmm_parameter_count(k, 4) for k in range(2, 6)] == [
        21,
        30,
        41,
        54,
    ]
    likelihood = -123.5
    expected = -2.0 * likelihood + 21 * math.log(200)
    assert gaussian_hmm_bic(likelihood, 200, 2, 4) == pytest.approx(expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda: gaussian_hmm_parameter_count(0, 4),
        lambda: gaussian_hmm_parameter_count(2, 0),
        lambda: gaussian_hmm_bic(np.inf, 10, 2, 4),
        lambda: gaussian_hmm_bic(-1.0, 0, 2, 4),
    ],
)
def test_parameter_count_and_bic_validate_inputs(call: Any) -> None:
    with pytest.raises(ModelError):
        call()


def test_canonicalization_permutes_labels_but_not_tied_covariance() -> None:
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.2, 0.3, 0.5]),
        transition_matrix=np.asarray(
            [[0.70, 0.20, 0.10], [0.10, 0.80, 0.10], [0.20, 0.30, 0.50]]
        ),
        means=np.asarray([[1.0, 0.0], [0.0, 5.0], [1.0, 0.0]]),
        tied_covariance=np.asarray([[2.0, 0.2], [0.2, 1.0]]),
    )
    result = canonicalize_hmm(parameters, states=np.asarray([0, 1, 2, 0]))
    np.testing.assert_array_equal(result.canonical_to_original, [1, 0, 2])
    np.testing.assert_array_equal(result.original_to_canonical, [1, 0, 2])
    np.testing.assert_allclose(
        result.parameters.transition_matrix,
        parameters.transition_matrix[np.ix_([1, 0, 2], [1, 0, 2])],
    )
    np.testing.assert_allclose(
        result.parameters.tied_covariance, parameters.tied_covariance
    )
    np.testing.assert_array_equal(result.states, [1, 0, 2, 1])


def test_canonicalization_ties_use_original_state_id() -> None:
    parameters = HMMParameters(
        np.asarray([0.4, 0.3, 0.3]),
        np.full((3, 3), 1.0 / 3.0),
        np.zeros((3, 2)),
        np.eye(2),
    )
    result = canonicalize_hmm(parameters)
    np.testing.assert_array_equal(result.canonical_to_original, [0, 1, 2])
    assert result.states is None


@pytest.mark.parametrize(
    "parameters",
    [
        HMMParameters(np.asarray(1.0), np.eye(2), np.zeros((2, 1)), np.eye(1)),
        HMMParameters(
            np.asarray([0.5, 0.5]), np.ones((2, 3)), np.zeros((2, 1)), np.eye(1)
        ),
        HMMParameters(np.asarray([0.5, 0.5]), np.eye(2), np.zeros((3, 1)), np.eye(1)),
        HMMParameters(np.asarray([0.5, 0.5]), np.eye(2), np.zeros((2, 1)), np.eye(2)),
        HMMParameters(
            np.asarray([0.5, 0.5]), np.eye(2), np.asarray([[np.nan], [0.0]]), np.eye(1)
        ),
        HMMParameters(np.asarray([1.1, -0.1]), np.eye(2), np.zeros((2, 1)), np.eye(1)),
        HMMParameters(np.asarray([0.4, 0.4]), np.eye(2), np.zeros((2, 1)), np.eye(1)),
    ],
)
def test_parameter_shape_probability_and_finite_validation(
    parameters: HMMParameters,
) -> None:
    with pytest.raises(ModelError):
        canonicalize_hmm(parameters)


def test_viterbi_resets_at_segments_and_ties_choose_smallest_state() -> None:
    parameters = HMMParameters(
        np.asarray([0.5, 0.5]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[-5.0], [5.0]]),
        np.asarray([[1.0]]),
    )
    values = np.asarray([[-5.0], [-5.0], [5.0], [5.0]])
    bridged = decode_viterbi(values, parameters)
    segmented = decode_viterbi(values, parameters, lengths=(2, 2))
    assert np.unique(bridged).size == 1
    np.testing.assert_array_equal(segmented, [0, 0, 1, 1])

    tied = replace(
        parameters,
        transition_matrix=np.full((2, 2), 0.5),
        means=np.zeros((2, 1)),
    )
    np.testing.assert_array_equal(
        decode_viterbi(np.zeros((4, 1)), tied, lengths=(2, 2)), [0, 0, 0, 0]
    )


def test_viterbi_rejects_dimension_singular_covariance_and_overflow() -> None:
    with pytest.raises(ModelError, match="dimension"):
        decode_viterbi(np.zeros((2, 3)), _parameters())
    with pytest.raises(ModelError, match="not positive definite"):
        decode_viterbi(
            np.zeros((2, 2)), replace(_parameters(), tied_covariance=np.zeros((2, 2)))
        )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ModelError, match="emission likelihoods are non-finite"),
    ):
        decode_viterbi(np.full((2, 2), 1e308), _parameters())


def test_tied_covariance_diagnostics_cover_floor_condition_and_symmetry() -> None:
    healthy = tied_covariance_diagnostic(np.diag([2.0, 1.0]))
    assert healthy.positive_definite
    assert healthy.eigenvalue_floor_passed
    assert healthy.condition_number == pytest.approx(2.0)
    below = tied_covariance_diagnostic(np.diag([1.0, 1e-8]))
    assert not below.eigenvalue_floor_passed
    assert not below.condition_number_passed
    asymmetric = tied_covariance_diagnostic(np.asarray([[1.0, 0.2], [0.0, 1.0]]))
    assert not asymmetric.symmetric
    assert not asymmetric.positive_definite
    assert covariance_diagnostics(np.eye(2))[0].positive_definite


@pytest.mark.parametrize(
    ("covariance", "kwargs"),
    [
        (np.asarray([1.0]), {}),
        (np.asarray([[np.nan]]), {}),
        (np.asarray([[1.0]]), {"eigenvalue_floor": -1.0}),
        (np.asarray([[1.0]]), {"maximum_condition_number": 0.0}),
        (np.asarray([[1.0]]), {"symmetry_tolerance": -1.0}),
    ],
)
def test_tied_covariance_diagnostic_contract(
    covariance: NDArray[np.float64], kwargs: dict[str, float]
) -> None:
    with pytest.raises(ModelError):
        tied_covariance_diagnostic(covariance, **kwargs)


def test_state_path_exposes_completed_only_dwell_evidence() -> None:
    states = np.asarray([0, 0, 1, 1, 0, 2, 1, 2], dtype=np.int64)
    report = summarize_state_path(states, 3, lengths=(5, 3))
    np.testing.assert_array_equal(report.state_counts, [3, 3, 2])
    assert report.transition_counts[0, 2] == 0
    np.testing.assert_array_equal(report.episode_counts, [2, 2, 2])
    np.testing.assert_array_equal(report.completed_episode_counts, [0, 2, 0])
    np.testing.assert_array_equal(report.censored_episode_counts, [2, 0, 2])
    np.testing.assert_array_equal(report.all_one_period_episode_counts, [1, 1, 2])
    np.testing.assert_array_equal(report.one_period_episode_counts, [0, 1, 0])
    assert [value.tolist() for value in report.dwell_lengths_by_state] == [
        [],
        [2, 1],
        [],
    ]
    assert [value.tolist() for value in report.all_dwell_lengths_by_state] == [
        [2, 1],
        [2, 1],
        [1, 1],
    ]


@pytest.mark.parametrize(
    ("states", "n_states", "lengths"),
    [([], 2, None), ([0.0, 1.0], 2, None), ([0, 2], 2, None), ([0, 1], 2, (1,))],
)
def test_state_path_and_length_validation(
    states: Sequence[int] | Sequence[float],
    n_states: int,
    lengths: tuple[int, ...] | None,
) -> None:
    with pytest.raises(ModelError):
        summarize_state_path(states, n_states, lengths=lengths)  # type: ignore[arg-type]


def _normal_density(value: float, mean: float, variance: float) -> float:
    exponent = math.log(2.0 * math.pi * variance) + (value - mean) ** 2 / variance
    return math.exp(-0.5 * exponent)


def _brute_force_segment(
    values: NDArray[np.float64], parameters: HMMParameters
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    states = range(parameters.n_states)
    weights: list[tuple[tuple[int, ...], float]] = []
    for path in product(states, repeat=len(values)):
        weight = float(parameters.start_probabilities[path[0]])
        for row, state in enumerate(path):
            if row:
                weight *= float(parameters.transition_matrix[path[row - 1], state])
            weight *= _normal_density(
                float(values[row, 0]),
                float(parameters.means[state, 0]),
                float(parameters.tied_covariance[0, 0]),
            )
        weights.append((path, weight))
    likelihood = math.fsum(weight for _path, weight in weights)
    gamma = np.zeros((len(values), parameters.n_states))
    xi = np.zeros((parameters.n_states, parameters.n_states))
    for path, weight in weights:
        probability = weight / likelihood
        for row, state in enumerate(path):
            gamma[row, state] += probability
            if row:
                xi[path[row - 1], state] += probability
    return math.log(likelihood), gamma, xi


def test_forward_backward_matches_brute_force_and_resets_segments() -> None:
    parameters = HMMParameters(
        np.asarray([0.6, 0.4]),
        np.asarray([[0.75, 0.25], [0.2, 0.8]]),
        np.asarray([[-1.0], [1.0]]),
        np.asarray([[0.7]]),
    )
    values = np.asarray([[-0.3], [0.4], [1.2]])
    result = forward_backward(values, parameters, lengths=(2, 1))
    first_ll, first_gamma, first_xi = _brute_force_segment(values[:2], parameters)
    second_ll, second_gamma, second_xi = _brute_force_segment(values[2:], parameters)
    np.testing.assert_allclose(
        result.posterior_probabilities,
        np.vstack((first_gamma, second_gamma)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.expected_transition_counts, first_xi + second_xi, atol=1e-12
    )
    assert result.sequence_log_likelihoods == pytest.approx((first_ll, second_ll))
    assert result.total_log_likelihood == pytest.approx(first_ll + second_ll)
    assert not result.posterior_probabilities.flags.writeable


def test_forward_backward_validates_dimensions_lengths_and_reachability() -> None:
    with pytest.raises(ModelError, match="dimension"):
        forward_backward(np.zeros((2, 3)), _parameters())
    with pytest.raises(ModelError, match="lengths"):
        forward_backward(np.zeros((2, 2)), _parameters(), lengths=(1,))
    unreachable = HMMParameters(
        np.asarray([1.0, 0.0]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.0], [1.0]]),
        np.asarray([[1.0]]),
    )
    # A zero-probability state remains legal because the reachable state has a
    # positive Gaussian density everywhere.
    result = forward_backward(np.asarray([[100.0]]), unreachable)
    assert np.isfinite(result.total_log_likelihood)


def test_identification_report_passes_exact_mass_and_distance_boundaries() -> None:
    gamma = np.zeros((48, 2))
    gamma[:24, 0] = 1.0
    gamma[24:, 1] = 1.0
    states = np.repeat(np.asarray([0, 1]), 24)
    parameters = HMMParameters(
        np.asarray([0.5, 0.5]),
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        np.asarray([[0.0], [0.75]]),
        np.asarray([[1.0]]),
    )
    report = posterior_identification_report(gamma, states, parameters)
    assert report.passed
    assert report.normalized_posterior_entropy == 0.0
    np.testing.assert_array_equal(report.posterior_effective_masses, [24.0, 24.0])
    assert report.minimum_pairwise_mahalanobis_distance == pytest.approx(0.75)


def test_identification_report_records_every_failure_family() -> None:
    parameters = HMMParameters(
        np.asarray([0.5, 0.5]),
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        np.asarray([[0.0], [0.1]]),
        np.asarray([[1.0]]),
    )
    uniform = np.full((48, 2), 0.5)
    states = np.repeat(np.asarray([0, 1]), 24)
    report = posterior_identification_report(uniform, states, parameters)
    assert "normalized_posterior_entropy_above_maximum" in report.rejection_reasons
    assert "mean_maximum_posterior_below_minimum" in report.rejection_reasons
    assert "state_0_assigned_posterior_below_minimum" in report.rejection_reasons
    assert "pairwise_mahalanobis_distance_below_minimum" in report.rejection_reasons

    imbalanced = np.zeros((300, 2))
    imbalanced[:271, 0] = 1.0
    imbalanced[271:, 1] = 1.0
    imbalanced_states = np.argmax(imbalanced, axis=1)
    mass = posterior_identification_report(imbalanced, imbalanced_states, parameters)
    assert "state_1_posterior_effective_mass_below_minimum" in mass.rejection_reasons

    no_assigned = posterior_identification_report(
        np.tile(np.asarray([0.5, 0.5]), (48, 1)), np.zeros(48, dtype=int), parameters
    )
    assert "state_1_has_no_viterbi_assigned_row" in no_assigned.rejection_reasons
    assert no_assigned.as_dict()["assigned_state_mean_posteriors"] == [0.5, None]


@pytest.mark.parametrize(
    "gamma",
    [np.asarray([]), np.asarray([[0.5, 0.4]]), np.asarray([[np.nan, 0.0]])],
)
def test_identification_report_rejects_invalid_posterior(gamma: NDArray[Any]) -> None:
    with pytest.raises(ModelError, match="posterior"):
        posterior_identification_report(gamma, np.asarray([0]), _parameters())


def test_one_step_predictive_scores_condition_sequentially_and_reset() -> None:
    parameters = HMMParameters(
        np.asarray([0.5, 0.5]),
        np.asarray([[0.8, 0.2], [0.25, 0.75]]),
        np.asarray([[-1.0], [1.0]]),
        np.asarray([[1.0]]),
    )
    values = np.asarray([[-0.5], [0.4], [1.2]])
    initial = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    result = one_step_predictive_log_scores(
        values, parameters, lengths=(2, 1), initial_state_probabilities=initial
    )
    densities0 = np.asarray(
        [_normal_density(-0.5, -1.0, 1.0), _normal_density(-0.5, 1.0, 1.0)]
    )
    expected0 = math.log(float(initial[0] @ densities0))
    posterior0 = initial[0] * densities0 / float(initial[0] @ densities0)
    prior1 = posterior0 @ parameters.transition_matrix
    densities1 = np.asarray(
        [_normal_density(0.4, -1.0, 1.0), _normal_density(0.4, 1.0, 1.0)]
    )
    expected1 = math.log(float(prior1 @ densities1))
    densities2 = np.asarray(
        [_normal_density(1.2, -1.0, 1.0), _normal_density(1.2, 1.0, 1.0)]
    )
    expected2 = math.log(float(initial[1] @ densities2))
    np.testing.assert_allclose(result.log_scores, [expected0, expected1, expected2])
    np.testing.assert_allclose(result.predictive_state_probabilities[2], initial[1])
    assert result.total_log_score == pytest.approx(expected0 + expected1 + expected2)
    assert not result.filtered_probabilities.flags.writeable


@pytest.mark.parametrize(
    "initial",
    [np.asarray([0.5, 0.5]), np.ones((2, 3)), np.asarray([[0.4, 0.4], [0.5, 0.5]])],
)
def test_predictive_score_initial_state_contract(initial: NDArray[np.float64]) -> None:
    with pytest.raises(ModelError, match="initial state"):
        one_step_predictive_log_scores(
            np.zeros((3, 2)),
            _parameters(),
            lengths=(2, 1),
            initial_state_probabilities=initial,
        )


def test_numerical_chain_uses_exact_edges_absorption_and_tv_mixing() -> None:
    matrix = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    report = validate_numerical_chain(matrix)
    np.testing.assert_allclose(report.stationary_distribution, [2 / 3, 1 / 3])
    assert report.period == 1
    assert report.second_largest_eigenvalue_modulus == pytest.approx(0.7)
    assert report.mixing_time_to_one_percent == 12
    assert report.maximum_total_variation_at_mixing <= 0.01
    assert validate_markov_chain(matrix).mixing_time_to_one_percent == 12

    accepted_noise = np.asarray([[-5e-16, 1.0 + 5e-16], [0.5, 0.5]], dtype=np.float64)
    assert validate_numerical_chain(accepted_noise).irreducible


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.asarray([[np.nan]]), "non-finite"),
        (np.asarray([[-2e-15, 1.0 + 2e-15], [0.5, 0.5]]), "bounds"),
        (np.asarray([[0.4, 0.4], [0.2, 0.8]]), "row-stochastic"),
        (np.asarray([[0.9, 0.1], [1e-13, 1.0 - 1e-13]]), "reducible"),
        (np.asarray([[0.0, 1.0], [1.0, 0.0]]), "periodic"),
        (np.asarray([[0.98, 0.02], [0.2, 0.8]]), "at or above 0.98"),
        (
            np.asarray(
                [
                    [0.97, 0.02999, 0.00001, 0.0],
                    [0.02999, 0.97, 0.0, 0.00001],
                    [0.00001, 0.0, 0.97, 0.02999],
                    [0.0, 0.00001, 0.02999, 0.97],
                ]
            ),
            "does not mix within 120",
        ),
    ],
)
def test_numerical_chain_rejects_every_invalidity(
    matrix: NDArray[np.float64], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        validate_numerical_chain(matrix)


def test_historical_transition_support_stores_cells_and_bootstrap_evidence() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    viterbi = np.asarray([[8, 3], [3, 8]], dtype=np.int64)
    expected = np.asarray([[8.5, 2.5], [2.2, 8.8]])
    frequencies = np.asarray([[1.0, 0.95], [0.9, 1.0]])
    quantiles = np.stack(
        (probabilities - 0.01, probabilities, probabilities + 0.01), axis=2
    )
    report = validate_historical_transition_support(
        probabilities,
        viterbi,
        expected,
        bootstrap_edge_frequencies=frequencies,
        bootstrap_probability_quantiles=quantiles,
    )
    assert report.passed
    assert report.supported_self_loop_states == (0, 1)
    np.testing.assert_array_equal(report.decoded_entries, [3, 3])
    np.testing.assert_array_equal(report.decoded_exits, [3, 3])
    np.testing.assert_allclose(report.bootstrap_edge_frequencies, frequencies)
    assert not report.supported_cells.flags.writeable


def test_historical_support_thresholds_are_inclusive_and_fail_closed() -> None:
    probabilities = np.asarray([[0.99, 0.01], [0.01, 0.99]])
    viterbi = np.asarray([[1, 3], [3, 1]], dtype=np.int64)
    expected = np.full((2, 2), 2.0)
    assert validate_historical_transition_support(
        probabilities, viterbi, expected
    ).passed

    failed = historical_transition_support_report(
        np.asarray([[0.995, 0.005], [0.3, 0.7]]), viterbi, expected
    )
    assert not failed.passed
    assert "supported_off_diagonal_graph_not_strongly_connected" in (
        failed.rejection_reasons
    )
    with pytest.raises(ModelError, match="historically supported"):
        validate_historical_transition_support(
            np.asarray([[0.995, 0.005], [0.3, 0.7]]), viterbi, expected
        )


@pytest.mark.parametrize(
    ("viterbi", "expected"),
    [
        (np.ones((2, 3), dtype=int), np.ones((2, 2))),
        (np.asarray([[1.0, 2.0], [2.0, 1.0]]), np.ones((2, 2))),
        (np.ones((2, 2), dtype=int), np.asarray([[1.0, -1.0], [1.0, 1.0]])),
    ],
)
def test_historical_transition_support_input_contract(
    viterbi: NDArray[Any], expected: NDArray[np.float64]
) -> None:
    with pytest.raises(ModelError):
        historical_transition_support_report(
            np.asarray([[0.8, 0.2], [0.2, 0.8]]), viterbi, expected
        )


@dataclass
class _FakeMonitor:
    converged: bool
    iter: int
    history: tuple[float, ...]


class _FakeModel:
    def __init__(
        self, owner: _FakeFactory, kwargs: dict[str, Any], outcome: dict[str, Any]
    ) -> None:
        self.owner = owner
        self.seed = int(kwargs["random_state"])
        self.outcome = outcome
        likelihood = float(outcome.get("likelihood", self.seed))
        iterations = int(outcome.get("iterations", 3))
        history = outcome.get(
            "history", tuple(float(value) for value in range(iterations))
        )
        self.monitor_ = _FakeMonitor(
            converged=bool(outcome.get("converged", True)),
            iter=iterations,
            history=tuple(history),
        )
        self.startprob_ = np.asarray(outcome.get("start", [0.5, 0.5]), dtype=float)
        self.transmat_ = np.asarray(
            outcome.get("transition", [[0.8, 0.2], [0.3, 0.7]]), dtype=float
        )
        self.means_ = np.asarray(
            outcome.get("means", [[2.0, 2.0], [-2.0, -2.0]]), dtype=float
        )
        self._covars_ = np.asarray(outcome.get("covariance", 0.2 * np.eye(2)))
        self.covars_ = np.repeat(self._covars_[None, :, :], 2, axis=0)
        minimum = float(
            outcome.get("raw_minimum", np.linalg.eigvalsh(self._covars_).min())
        )
        floor_change = float(outcome.get("floor_change", 0.0))
        self.raw_covariance_min_eigenvalues_ = [minimum] * iterations
        self.covariance_floor_relative_changes_ = [floor_change] * iterations
        if outcome.get("audit_missing"):
            del self.raw_covariance_min_eigenvalues_
            del self.covariance_floor_relative_changes_
        self.likelihood = likelihood

    def fit(
        self,
        values: NDArray[np.float64],
        lengths: Sequence[int] | None = None,
    ) -> _FakeModel:
        self.owner.fit_lengths.append(tuple(lengths or ()))
        if self.outcome.get("fit_error"):
            raise HMMRestartNumericalFailure("synthetic restart failure")
        return self

    def score(
        self,
        values: NDArray[np.float64],
        lengths: Sequence[int] | None = None,
    ) -> float:
        self.owner.score_lengths.append(tuple(lengths or ()))
        if self.outcome.get("score_error"):
            raise HMMRestartNumericalFailure("synthetic score failure")
        return self.likelihood


class _FakeFactory:
    def __init__(
        self,
        *,
        outcomes: dict[int, dict[str, Any]] | None = None,
        default_outcome: dict[str, Any] | None = None,
        always_fail: bool = False,
    ) -> None:
        self.outcomes = outcomes or {}
        self.default_outcome = default_outcome or {}
        self.always_fail = always_fail
        self.kwargs: list[dict[str, Any]] = []
        self.fit_lengths: list[tuple[int, ...]] = []
        self.score_lengths: list[tuple[int, ...]] = []

    def __call__(self, **kwargs: Any) -> _FakeModel:
        self.kwargs.append(kwargs)
        if self.always_fail:
            raise RuntimeError("all starts fail")
        seed = int(kwargs["random_state"])
        outcome = {**self.default_outcome, **self.outcomes.get(seed, {})}
        return _FakeModel(self, kwargs, outcome)


class _ParallelRuntimeFaultFactory:
    """Pickleable operational-fault source for spawned-worker coverage."""

    def __call__(self, **_kwargs: object) -> _FakeModel:
        raise RuntimeError("parallel synthetic operational fault")


class _ParallelModelFaultFactory:
    """Pickleable generic model-fault source for spawned-worker coverage."""

    def __call__(self, **_kwargs: object) -> _FakeModel:
        raise ModelError(
            "parallel synthetic generic model fault",
            details={"source": "restart worker"},
        )


def test_restart_runner_uses_exact_factory_and_audits_every_restart() -> None:
    outcomes: dict[int, dict[str, Any]] = {
        0: {"fit_error": True},
        1: {"converged": False},
        2: {"likelihood": math.nan},
        3: {"iterations": 1},
        4: {"history": (0.0, -1.0, -2.0)},
        5: {"raw_minimum": -1e-9},
        6: {"floor_change": 0.02},
        7: {"covariance": np.diag([1.0, 1e-7])},
        8: {"transition": [[0.99, 0.01], [0.01, 0.99]]},
        9: {"audit_missing": True},
        10: {"score_error": True},
    }
    factory = _FakeFactory(outcomes=outcomes)
    seeds = tuple(range(HMM_RESTARTS))
    result = fit_gaussian_hmm_restarts(_fit_features(), 2, seeds, model_factory=factory)
    assert len(factory.kwargs) == HMM_RESTARTS
    expected_kwargs = {
        "n_components": 2,
        "covariance_type": "tied",
        "min_covar": 1e-6,
        "startprob_prior": 1.0,
        "transmat_prior": 1.0,
        "means_weight": 0.0,
        "covars_prior": 0.0,
        "covars_weight": 0.0,
        "n_iter": 1_000,
        "tol": 1e-6,
        "algorithm": "viterbi",
        "implementation": "log",
        "params": "stmc",
        "init_params": "stmc",
    }
    for seed, kwargs in enumerate(factory.kwargs):
        assert kwargs == {**expected_kwargs, "random_state": seed}
    assert factory.fit_lengths == [(50, 50)] * 50
    assert factory.score_lengths == [(50, 50)] * 49
    failure_types = [record.failure_type for record in result.restart_diagnostics]
    assert failure_types[:11] == [
        "NumericalRestartFailure",
        "NonConvergence",
        "NonFiniteLikelihood",
        "InsufficientIterations",
        "LikelihoodDecrease",
        "RawCovarianceEigenvalue",
        "CovarianceFloorPerturbation",
        "ParameterAuditFailure",
        "PreliminaryMarkovFailure",
        "CovarianceAuditMissing",
        "LikelihoodEvaluationFailure",
    ]
    assert result.restart_diagnostics[7].covariance_condition_number == pytest.approx(
        1e7
    )
    assert result.best_restart_index == 49
    assert result.best_seed == 49
    assert result.identification.passed
    assert result.historical_transition_support.passed
    assert result.parameter_count == gaussian_hmm_parameter_count(2, 2)
    assert result.covariance_diagnostic.condition_number == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("lower_difference", "expected_restart"),
    [(5e-9, 2), (2e-8, 5)],
)
def test_restart_likelihood_tie_uses_scaled_tolerance_and_smallest_id(
    lower_difference: float, expected_restart: int
) -> None:
    factory = _FakeFactory(
        outcomes={
            2: {"likelihood": 100.0 - lower_difference},
            5: {"likelihood": 100.0},
        }
    )
    result = fit_gaussian_hmm_restarts(
        _fit_features(), 2, tuple(range(50)), model_factory=factory
    )
    assert result.best_restart_index == expected_restart


def test_nine_worker_restart_fit_is_bit_identical_ordered_and_parent_assembled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _fit_features()
    seeds = tuple(range(HMM_RESTARTS))
    outcomes: dict[int, dict[str, Any]] = {
        0: {"fit_error": True},
        7: {"converged": False},
        11: {"likelihood": 1_000.0},
    }
    serial = fit_gaussian_hmm_restarts(
        features,
        2,
        seeds,
        model_factory=_FakeFactory(outcomes=outcomes),
        workers=1,
    )

    parent_pid = os.getpid()
    assembly_pids: list[int] = []
    mapped_workers: list[tuple[int, int]] = []
    inside_map = False
    original = canonicalize_hmm

    def observed_parent_assembly(parameters: HMMParameters) -> object:
        assert not inside_map
        assembly_pids.append(os.getpid())
        return original(parameters)

    def observed_ordered_map(
        function: Any,
        items: Sequence[Any],
        *,
        workers: int,
        chunksize: int,
    ) -> tuple[Any, ...]:
        nonlocal inside_map
        mapped_workers.append((workers, chunksize))
        inside_map = True
        try:
            return tuple(function(item) for item in items)
        finally:
            inside_map = False

    monkeypatch.setattr(
        "prpg.model.hmm.canonicalize_hmm",
        observed_parent_assembly,
    )
    monkeypatch.setattr(
        "prpg.model.hmm.deterministic_process_map",
        observed_ordered_map,
    )
    parallel = fit_gaussian_hmm_restarts(
        features,
        2,
        seeds,
        model_factory=_FakeFactory(outcomes=outcomes),
        workers=9,
    )

    assert assembly_pids == [parent_pid]
    assert mapped_workers == [(9, 1)]
    assert gaussian_hmm_fit_fingerprint(parallel) == gaussian_hmm_fit_fingerprint(
        serial
    )
    with pytest.raises(
        ModelError,
        match="parallel synthetic generic model fault",
    ) as caught:
        fit_gaussian_hmm_restarts(
            features,
            2,
            seeds,
            model_factory=_ParallelModelFaultFactory(),
            workers=9,
        )
    assert caught.value.details == {"source": "restart worker"}
    assert tuple(item.restart_index for item in parallel.restart_diagnostics) == tuple(
        range(HMM_RESTARTS)
    )
    assert tuple(item.seed for item in parallel.restart_diagnostics) == seeds
    assert parallel.restart_diagnostics == serial.restart_diagnostics


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_restart_runner_rejects_invalid_worker_counts(workers: object) -> None:
    with pytest.raises(ModelError, match="workers must be a positive integer"):
        fit_gaussian_hmm_restarts(
            _fit_features(),
            2,
            tuple(range(HMM_RESTARTS)),
            model_factory=_FakeFactory(),
            workers=workers,  # type: ignore[arg-type]
        )


def test_parallel_restart_runner_propagates_generic_operational_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def observed_ordered_map(
        function: Any,
        items: Sequence[Any],
        *,
        workers: int,
        chunksize: int,
    ) -> tuple[Any, ...]:
        assert (workers, chunksize) == (9, 1)
        return tuple(function(item) for item in items)

    monkeypatch.setattr(
        "prpg.model.hmm.deterministic_process_map",
        observed_ordered_map,
    )
    with pytest.raises(RuntimeError, match="parallel synthetic operational fault"):
        fit_gaussian_hmm_restarts(
            _fit_features(),
            2,
            tuple(range(HMM_RESTARTS)),
            model_factory=_ParallelRuntimeFaultFactory(),
            workers=9,
        )


def test_no_valid_restart_retains_all_fifty_diagnostics() -> None:
    with pytest.raises(
        NoNumericallyValidHMMRestarts,
        match="no numerically valid",
    ) as caught:
        fit_gaussian_hmm_restarts(
            _fit_features(),
            2,
            tuple(range(50)),
            model_factory=_FakeFactory(default_outcome={"converged": False}),
        )
    records = caught.value.details["restart_diagnostics"]
    assert len(records) == 50
    assert records[0]["restart_index"] == 0
    assert records[-1]["restart_index"] == 49
    assert caught.value.restart_diagnostics[0].seed == 0
    assert caught.value.restart_diagnostics[-1].seed == 49


@pytest.mark.parametrize(
    "failure",
    [
        ModelError("generic model defect"),
        IntegrityError("integrity defect"),
        RuntimeError("runtime defect"),
        OSError("filesystem defect"),
        MemoryError("memory defect"),
    ],
)
def test_restart_runner_does_not_convert_generic_faults(
    failure: BaseException,
) -> None:
    class _FaultFactory:
        def __call__(self, **_kwargs: object) -> _FakeModel:
            raise failure

    with pytest.raises(type(failure)) as caught:
        fit_gaussian_hmm_restarts(
            _fit_features(),
            2,
            tuple(range(HMM_RESTARTS)),
            model_factory=_FaultFactory(),
        )
    assert caught.value is failure


def test_selected_restart_must_pass_identification_and_historical_support() -> None:
    overlapping = _FakeFactory(default_outcome={"means": [[-0.1, -0.1], [0.1, 0.1]]})
    with pytest.raises(
        HMMCandidateInadmissibility, match="posterior-identification"
    ) as identified:
        fit_gaussian_hmm_restarts(
            _fit_features(), 2, tuple(range(50)), model_factory=overlapping
        )
    identification_evidence = identified.value.evidence
    assert identification_evidence.failure_code == "posterior_identification"
    assert len(identification_evidence.restart_diagnostics) == 50
    assert identification_evidence.rejection_reasons
    assert dict(identification_evidence.gate_details)["passed"] is False

    with pytest.raises(
        HMMCandidateInadmissibility, match="historically supported"
    ) as supported:
        fit_gaussian_hmm_restarts(
            _fit_features(single_switch=True),
            2,
            tuple(range(50)),
            model_factory=_FakeFactory(),
        )
    support_evidence = supported.value.evidence
    assert support_evidence.failure_code == "historical_transition_support"
    assert len(support_evidence.restart_diagnostics) == 50
    assert support_evidence.rejection_reasons
    assert dict(support_evidence.gate_details)["passed"] is False


def test_k4_fit_retains_raw_rarity_failures_with_effective_owner_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_identification = posterior_identification_report
    raw_transition = historical_transition_support_report

    def rarity_identification(*args: Any, **kwargs: Any) -> object:
        report = raw_identification(*args, **kwargs)
        masses = np.asarray(report.posterior_effective_masses).copy()
        masses[-1] = 12.0
        return replace(
            report,
            posterior_effective_masses=masses,
            passed=False,
            rejection_reasons=("state_3_posterior_effective_mass_below_minimum",),
        )

    def scarcity_transition(*args: Any, **kwargs: Any) -> object:
        report = raw_transition(*args, **kwargs)
        entries = np.asarray(report.decoded_entries).copy()
        entries[-1] = 2
        return replace(
            report,
            decoded_entries=entries,
            passed=False,
            rejection_reasons=("state_3_decoded_entries_below_three",),
        )

    monkeypatch.setattr(
        "prpg.model.hmm.posterior_identification_report",
        rarity_identification,
    )
    monkeypatch.setattr(
        "prpg.model.hmm.historical_transition_support_report",
        scarcity_transition,
    )

    result = fit_gaussian_hmm_restarts(
        _fit_features_k4(),
        4,
        tuple(range(HMM_RESTARTS)),
        model_factory=_k4_factory(),
        owner_fixed_k4_policy=OWNER_FIXED_K4_PROSPECTIVE_POLICY,
    )

    assert result.identification.passed is False
    assert result.identification.rejection_reasons == (
        "state_3_posterior_effective_mass_below_minimum",
    )
    assert result.historical_transition_support.passed is False
    assert result.historical_transition_support.rejection_reasons == (
        "state_3_decoded_entries_below_three",
    )
    assert effective_identification_passed(result)
    assert effective_transition_support_passed(result)
    assert result.owner_fixed_k4_disposition is not None
    assert gaussian_hmm_fit_fingerprint(result) != gaussian_hmm_fit_fingerprint(
        replace(result, owner_fixed_k4_disposition=None)
    )


def test_legacy_k4_fit_does_not_apply_prospective_owner_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_identification = posterior_identification_report

    def rarity_identification(*args: Any, **kwargs: Any) -> object:
        report = raw_identification(*args, **kwargs)
        return replace(
            report,
            passed=False,
            rejection_reasons=("state_3_posterior_effective_mass_below_minimum",),
        )

    monkeypatch.setattr(
        "prpg.model.hmm.posterior_identification_report",
        rarity_identification,
    )
    with pytest.raises(HMMCandidateInadmissibility) as caught:
        fit_gaussian_hmm_restarts(
            _fit_features_k4(),
            4,
            tuple(range(HMM_RESTARTS)),
            model_factory=_k4_factory(),
        )

    assert caught.value.evidence.failure_code == "posterior_identification"
    assert caught.value.evidence.rejection_reasons == (
        "state_3_posterior_effective_mass_below_minimum",
    )


def test_owner_fixed_k4_policy_is_exactly_scoped_to_scientific_version_11() -> None:
    assert owner_fixed_k4_policy_for_scientific_version(1) is None
    assert owner_fixed_k4_policy_for_scientific_version(2) is None
    assert (
        owner_fixed_k4_policy_for_scientific_version(11)
        == OWNER_FIXED_K4_PROSPECTIVE_POLICY
    )
    assert owner_fixed_k4_policy_for_scientific_version(12) is None
    with pytest.raises(ModelError, match="scientific version"):
        owner_fixed_k4_policy_for_scientific_version(True)


def test_passing_legacy_k4_fit_has_no_disposition_but_explicit_v11_fit_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_identification = posterior_identification_report
    raw_transition = historical_transition_support_report

    def passing_identification(*args: Any, **kwargs: Any) -> object:
        return replace(
            raw_identification(*args, **kwargs),
            passed=True,
            rejection_reasons=(),
        )

    def passing_transition(*args: Any, **kwargs: Any) -> object:
        return replace(
            raw_transition(*args, **kwargs),
            passed=True,
            rejection_reasons=(),
        )

    monkeypatch.setattr(
        "prpg.model.hmm.posterior_identification_report",
        passing_identification,
    )
    monkeypatch.setattr(
        "prpg.model.hmm.historical_transition_support_report",
        passing_transition,
    )
    features = _fit_features_k4()
    seeds = tuple(range(HMM_RESTARTS))
    legacy = fit_gaussian_hmm_restarts(
        features,
        4,
        seeds,
        model_factory=_k4_factory(),
    )
    prospective = fit_gaussian_hmm_restarts(
        features,
        4,
        seeds,
        model_factory=_k4_factory(),
        owner_fixed_k4_policy=OWNER_FIXED_K4_PROSPECTIVE_POLICY,
    )

    assert legacy.owner_fixed_k4_disposition is None
    assert prospective.owner_fixed_k4_disposition is not None


def test_k4_non_rarity_identification_failure_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_identification = posterior_identification_report

    def binding_identification(*args: Any, **kwargs: Any) -> object:
        report = raw_identification(*args, **kwargs)
        return replace(
            report,
            passed=False,
            rejection_reasons=("normalized_posterior_entropy_above_maximum",),
        )

    monkeypatch.setattr(
        "prpg.model.hmm.posterior_identification_report",
        binding_identification,
    )
    with pytest.raises(
        HMMCandidateInadmissibility,
        match="posterior-identification",
    ) as caught:
        fit_gaussian_hmm_restarts(
            _fit_features_k4(),
            4,
            tuple(range(HMM_RESTARTS)),
            model_factory=_k4_factory(),
        )
    assert caught.value.evidence.rejection_reasons == (
        "normalized_posterior_entropy_above_maximum",
    )


def test_k4_unknown_transition_failure_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_transition = historical_transition_support_report

    def binding_transition(*args: Any, **kwargs: Any) -> object:
        report = raw_transition(*args, **kwargs)
        return replace(
            report,
            passed=False,
            rejection_reasons=("transition_probability_calibration_failed",),
        )

    monkeypatch.setattr(
        "prpg.model.hmm.historical_transition_support_report",
        binding_transition,
    )
    with pytest.raises(
        HMMCandidateInadmissibility,
        match="historically supported",
    ) as caught:
        fit_gaussian_hmm_restarts(
            _fit_features_k4(),
            4,
            tuple(range(HMM_RESTARTS)),
            model_factory=_k4_factory(),
        )
    assert caught.value.evidence.rejection_reasons == (
        "transition_probability_calibration_failed",
    )


def test_owner_fixed_k4_recurring_history_support_counts_censored_runs() -> None:
    passed = owner_fixed_k4_recurring_history_support(
        monthly_episode_counts=(2, 3, 2, 4),
        daily_run_counts=(5, 2, 8, 2),
    )
    failed = owner_fixed_k4_recurring_history_support(
        monthly_episode_counts=(2, 1, 2, 4),
        daily_run_counts=(5, 2, 1, 2),
    )

    assert passed.passed
    assert passed.rejection_reasons == ()
    assert not failed.passed
    assert failed.rejection_reasons == (
        "state_1_monthly_observed_episodes_below_two",
        "state_2_daily_runs_below_two",
    )
    with pytest.raises(OwnerFixedK4RecurringHistoryFailure) as caught:
        raise OwnerFixedK4RecurringHistoryFailure(failed)
    assert caught.value.report is failed


@pytest.mark.parametrize(
    ("monthly", "daily"),
    [
        ((2, 2, 2), (2, 2, 2, 2)),
        ((2, 2, 2, -1), (2, 2, 2, 2)),
        ((2, 2, 2, 2), (2, True, 2, 2)),
    ],
)
def test_owner_fixed_k4_recurring_history_support_rejects_malformed_counts(
    monthly: tuple[int, ...], daily: tuple[int, ...]
) -> None:
    with pytest.raises(ModelError, match="must contain four counts"):
        owner_fixed_k4_recurring_history_support(
            monthly_episode_counts=monthly,
            daily_run_counts=daily,
        )


@pytest.mark.parametrize(
    "seeds",
    [tuple(range(49)), tuple([0] * 50), (*range(49), -1), (*range(49), 2**32)],
)
def test_restart_seed_contract_is_strict(seeds: tuple[int, ...]) -> None:
    with pytest.raises(ModelError):
        fit_gaussian_hmm_restarts(_fit_features(), 2, seeds)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"feature_names": ("duplicate", "duplicate")}, "feature names"),
        ({"row_labels": ("too-short",)}, "row labels"),
        ({"scaler_scope": "invalid"}, "scaler scope"),
        ({"scaler_means": np.asarray([0.0])}, "does not match"),
        ({"scaler_means": np.asarray([0.0, np.inf])}, "non-finite"),
        ({"scaler_standard_deviations": np.asarray([1.0, 0.0])}, "positive"),
    ],
)
def test_restart_runner_validates_feature_metadata_before_fitting(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        fit_gaussian_hmm_restarts(
            replace(_fit_features(), **change),
            2,
            tuple(range(50)),
            model_factory=_FakeFactory(),
        )


def test_restart_runner_rejects_too_few_observations() -> None:
    features = _fit_features()
    one_row = replace(
        features,
        values=features.values[:1],
        lengths=(1,),
        row_labels=features.row_labels[:1],
    )
    with pytest.raises(ModelError, match="at least n_states"):
        fit_gaussian_hmm_restarts(one_row, 2, tuple(range(50)))


def test_real_hmmlearn_fifty_restart_tied_smoke_with_two_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    rng = np.random.default_rng(20260714)
    latent = np.tile(np.repeat(np.asarray([0, 1]), 5), 10)
    raw_values = np.column_stack(
        (
            np.where(latent == 0, -2.0, 2.0) + rng.normal(0.0, 0.2, len(latent)),
            np.where(latent == 0, -2.0, 2.0) + rng.normal(0.0, 0.2, len(latent)),
        )
    )
    raw = pd.DataFrame(raw_values, columns=["growth", "inflation"])
    standardized = standardize_features(raw)
    continuation = np.ones(len(raw), dtype=np.bool_)
    continuation[[0, 50]] = False
    features = consume_standardized_features(
        standardized,
        continuation=continuation,
        scaler_scope="production_full",
    )
    result = fit_gaussian_hmm_restarts(
        features, 2, tuple(range(1_000, 1_000 + HMM_RESTARTS))
    )
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except PermissionError:
        parallel = None
    else:
        parallel = fit_gaussian_hmm_restarts(
            features,
            2,
            tuple(range(1_000, 1_000 + HMM_RESTARTS)),
            workers=9,
        )
    assert result.lengths == (50, 50)
    assert len(result.restart_diagnostics) == HMM_RESTARTS
    assert any(record.valid for record in result.restart_diagnostics)
    assert all(
        record.raw_minimum_eigenvalue is not None
        for record in result.restart_diagnostics
        if record.valid
    )
    assert result.parameters.tied_covariance.shape == (2, 2)
    assert result.identification.passed
    assert result.historical_transition_support.passed
    if parallel is not None:
        assert gaussian_hmm_fit_fingerprint(parallel) == gaussian_hmm_fit_fingerprint(
            result
        )
