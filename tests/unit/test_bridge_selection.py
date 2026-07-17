"""Tests for exact ordinary Tier-A bridge selection execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.bridge_history import (
    BridgePseudoHistory,
    generate_registered_bridge_history,
)
from prpg.model.bridge_selection import (
    BridgeTierAPairResult,
    assemble_bridge_tier_a,
    bridge_feature_matrix,
    bridge_selection_scope,
    run_bridge_fixed_k4_tier_a_pair,
    run_bridge_selection_member,
    run_bridge_tier_a_pair,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    NoNumericallyValidHMMRestarts,
    RestartDiagnostic,
    summarize_state_path,
)
from prpg.model.regime_policy import OwnerFixedK4Disposition
from prpg.simulation.rng import ModelFitScope


@pytest.fixture(scope="module")
def bootstrap_indices() -> np.ndarray:
    result = np.broadcast_to(
        np.arange(12, dtype=np.int64),
        (10_000, 6, 12),
    ).copy()
    result.setflags(write=False)
    return result


def _values(rows: int) -> np.ndarray:
    return np.random.default_rng(711 + rows).normal(size=(rows, 4))


def _mask(rows: int, *, gap: int | None = None) -> np.ndarray:
    result = np.ones(rows, dtype=np.bool_)
    result[0] = False
    if gap is not None:
        result[gap] = False
    return result


def _parameters() -> HMMParameters:
    return HMMParameters(
        np.array([0.5, 0.5]),
        np.array([[0.85, 0.15], [0.20, 0.80]]),
        np.array([[-0.4, -0.2, 0.2, 0.4], [0.5, 0.3, -0.2, -0.5]]),
        np.eye(4),
    )


def _k4_parameters() -> HMMParameters:
    transition = np.full((4, 4), 0.05)
    np.fill_diagonal(transition, 0.85)
    return HMMParameters(
        np.full(4, 0.25),
        transition,
        np.array(
            [
                [-1.0, -0.6, 0.4, 0.8],
                [-0.3, 0.5, -0.8, 0.2],
                [0.4, -0.2, 0.7, -0.5],
                [1.0, 0.7, -0.3, -0.9],
            ]
        ),
        np.eye(4),
    )


def _fake_fixed_k4_fit(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
    *,
    episode_counts: tuple[int, int, int, int] = (2, 2, 2, 2),
) -> GaussianHMMFit:
    assert n_states == 4
    assert len(seeds) == 50
    rows = len(features.values)
    parameters = _k4_parameters()
    decoded = np.resize(np.repeat(np.arange(4), 5), rows).astype(np.int64)
    identification_reasons = ("state_3_posterior_effective_mass_below_minimum",)
    transition_reasons = ("state_3_decoded_entries_below_three",)
    disposition = OwnerFixedK4Disposition(
        "owner-fixed-k4-rarity-v1",
        4,
        identification_reasons,
        transition_reasons,
    )
    fit = SimpleNamespace(
        parameters=parameters,
        decoded_states=decoded,
        n_states=4,
        n_features=4,
        n_observations=rows,
        lengths=features.lengths,
        scaler_scope=features.scaler_scope,
        scaler_means=np.asarray(features.scaler_means).copy(),
        scaler_standard_deviations=np.asarray(
            features.scaler_standard_deviations
        ).copy(),
        covariance_diagnostic=SimpleNamespace(
            symmetric=True,
            positive_definite=True,
            eigenvalue_floor_passed=True,
            condition_number_passed=True,
            condition_number=1.0,
        ),
        chain_diagnostics=SimpleNamespace(
            irreducible=True,
            aperiodic=True,
            unique_stationary_distribution=True,
            period=1,
            stationary_residual=0.0,
            near_absorbing_states=(),
            mixing_time_to_one_percent=1,
        ),
        state_path_diagnostics=SimpleNamespace(
            episode_counts=np.asarray(episode_counts, dtype=np.int64),
            state_counts=np.zeros(4, dtype=np.int64),
            occupancy=np.zeros(4, dtype=np.float64),
            completed_episode_counts=np.zeros(4, dtype=np.int64),
        ),
        identification=SimpleNamespace(
            passed=False,
            rejection_reasons=identification_reasons,
        ),
        historical_transition_support=SimpleNamespace(
            passed=False,
            rejection_reasons=transition_reasons,
        ),
        owner_fixed_k4_disposition=disposition,
    )
    return cast(GaussianHMMFit, fit)


def _k4_history() -> BridgePseudoHistory:
    return generate_registered_bridge_history(
        dgp="model",
        tier="selection",
        pair=0,
        master_seed=1234,
        scientific_version=11,
        design_parameters=_k4_parameters(),
        design_standardized_history=_values(193),
        design_continuation=_mask(193),
        macro_block_length=4,
    )


def _fake_fit(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
) -> GaussianHMMFit:
    assert len(seeds) == 50
    rows = len(features.values)
    stationary = np.full(n_states, 1.0 / n_states)
    transition = np.full((n_states, n_states), 1.0 / n_states)
    parameters = HMMParameters(
        stationary.copy(),
        transition,
        np.zeros((n_states, 4)),
        np.eye(4),
    )
    decoded = np.arange(rows, dtype=np.int64) % n_states
    path = summarize_state_path(decoded, n_states, lengths=features.lengths)
    fit = SimpleNamespace(
        parameters=parameters,
        decoded_states=decoded,
        n_states=n_states,
        n_features=4,
        n_observations=rows,
        bic=100.0 + n_states,
        log_likelihood=-100.0,
        covariance_diagnostic=SimpleNamespace(
            symmetric=True,
            positive_definite=True,
            eigenvalue_floor_passed=True,
            condition_number_passed=True,
            condition_number=1.0,
        ),
        state_path_diagnostics=path,
        chain_diagnostics=SimpleNamespace(
            stationary_distribution=stationary,
            irreducible=True,
            aperiodic=True,
            unique_stationary_distribution=True,
            period=1,
            stationary_residual=0.0,
            near_absorbing_states=(),
            mixing_time_to_one_percent=1,
        ),
        forward_backward=SimpleNamespace(
            filtered_probabilities=np.tile(stationary, (rows, 1))
        ),
        identification=SimpleNamespace(passed=True),
        historical_transition_support=SimpleNamespace(passed=True),
    )
    return cast(GaussianHMMFit, fit)


def test_bridge_feature_matrix_recomputes_exact_population_scaler() -> None:
    values = _values(193)
    feature = bridge_feature_matrix(values, _mask(193))
    np.testing.assert_allclose(feature.values.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(feature.values.std(axis=0), 1.0, atol=1e-14)
    assert feature.lengths == (193,)
    assert feature.scaler_scope == "production_full"
    assert not feature.values.flags.writeable

    rolling = bridge_feature_matrix(values, _mask(193), training_rows=121)
    assert rolling.values.shape == (121, 4)
    assert rolling.scaler_scope == "design_training"


def test_bridge_scope_registry_is_exact() -> None:
    assert bridge_selection_scope("model", "prefix") is (
        ModelFitScope.BRIDGE_MODEL_SELECTION_PREFIX
    )
    assert bridge_selection_scope("model", "full") is (
        ModelFitScope.BRIDGE_MODEL_SELECTION_FULL
    )
    assert bridge_selection_scope("nonparametric", "prefix") is (
        ModelFitScope.BRIDGE_NONPARAMETRIC_SELECTION_PREFIX
    )
    assert bridge_selection_scope("nonparametric", "full") is (
        ModelFitScope.BRIDGE_NONPARAMETRIC_SELECTION_FULL
    )
    with pytest.raises(ModelError, match="outside the registry"):
        bridge_selection_scope("bad", "prefix")  # type: ignore[arg-type]


def test_fixed_k4_tier_a_uses_two_main_fits_and_only_retained_rules() -> None:
    calls: list[tuple[int, int, int]] = []

    def fit(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: Sequence[int],
    ) -> GaussianHMMFit:
        calls.append((len(features.values), n_states, len(seeds)))
        return _fake_fixed_k4_fit(features, n_states, seeds)

    result = run_bridge_fixed_k4_tier_a_pair(
        _k4_history(),
        parent_parameters=_k4_parameters(),
        master_seed=1234,
        scientific_version=11,
        fit_function=fit,
    )

    assert calls == [(193, 4, 50), (217, 4, 50)]
    assert result.success
    for member in (result.prefix, result.full):
        assert member.passed
        assert member.random_restart_ids == tuple(range(50))
        assert member.observed_episode_counts == (2, 2, 2, 2)
        assert sorted(member.reference_to_candidate) == [0, 1, 2, 3]
        assert member.fit is not None
        assert member.fit.identification.passed is False
        assert member.fit.historical_transition_support.passed is False
        # Deliberately malformed old count/occupancy/completed-episode fields
        # are not part of the version-3 Tier-A decision.
        assert not np.asarray(
            member.fit.state_path_diagnostics.completed_episode_counts
        ).any()


def test_fixed_k4_tier_a_rejects_one_observed_episode_without_cross_k_rescue() -> None:
    calls = 0

    def fit(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: Sequence[int],
    ) -> GaussianHMMFit:
        nonlocal calls
        calls += 1
        counts = (2, 2, 2, 1) if calls == 1 else (2, 2, 2, 2)
        return _fake_fixed_k4_fit(
            features,
            n_states,
            seeds,
            episode_counts=counts,
        )

    result = run_bridge_fixed_k4_tier_a_pair(
        _k4_history(),
        parent_parameters=_k4_parameters(),
        master_seed=1234,
        scientific_version=11,
        fit_function=fit,
    )

    assert calls == 2
    assert not result.prefix.passed
    assert result.prefix.rejection_reasons == (
        "observed_monthly_episode_support_failed",
    )
    assert result.full.passed
    assert not result.success


def test_fixed_k4_tier_a_propagates_unexpected_model_fault() -> None:
    failure = ModelError("synthetic fixed-K4 programming defect")

    def fails(
        _features: HMMFeatureMatrix,
        _n_states: int,
        _seeds: Sequence[int],
    ) -> GaussianHMMFit:
        raise failure

    with pytest.raises(
        ModelError, match="synthetic fixed-K4 programming defect"
    ) as caught:
        run_bridge_fixed_k4_tier_a_pair(
            _k4_history(),
            parent_parameters=_k4_parameters(),
            master_seed=1234,
            scientific_version=11,
            fit_function=fails,
        )
    assert caught.value is failure


def test_prefix_member_runs_main_plus_six_registered_folds(
    bootstrap_indices: np.ndarray,
) -> None:
    result = run_bridge_selection_member(
        _values(193),
        _mask(193),
        dgp="model",
        member="prefix",
        pair=3,
        master_seed=100,
        scientific_version=1,
        registered_bootstrap_indices=bootstrap_indices,
        fit_function=_fake_fit,
    )

    assert result.main_replicate == 21
    assert result.lengths == (193,)
    assert result.selection.passed
    assert result.selection.selected_n_states == 2
    assert tuple(candidate.n_states for candidate in result.candidates) == (2, 3, 4, 5)
    for candidate in result.candidates:
        assert len(candidate.rolling_attempts) == 6
        assert tuple(attempt.replicate for attempt in candidate.rolling_attempts) == (
            22,
            23,
            24,
            25,
            26,
            27,
        )
        assert all(attempt.successful for attempt in candidate.rolling_attempts)
        assert len(candidate.rolling_scores) == 72


def test_full_member_uses_exact_gap_aware_six_folds(
    bootstrap_indices: np.ndarray,
) -> None:
    result = run_bridge_selection_member(
        _values(217),
        _mask(217, gap=210),
        dgp="nonparametric",
        member="full",
        pair=0,
        master_seed=101,
        scientific_version=1,
        registered_bootstrap_indices=bootstrap_indices,
        fit_function=_fake_fit,
    )

    assert result.lengths == (210, 7)
    folds = result.candidates[0].rolling_attempts
    assert tuple(attempt.training_rows for attempt in folds) == (
        145,
        157,
        169,
        181,
        193,
        205,
    )
    assert folds[-1].validation_lengths == (5, 7)


def test_failed_candidate_and_all_its_folds_are_retained(
    bootstrap_indices: np.ndarray,
) -> None:
    def sometimes_fails(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: Sequence[int],
    ) -> GaussianHMMFit:
        if n_states == 3:
            raise NoNumericallyValidHMMRestarts(
                n_states=n_states,
                restart_diagnostics=tuple(
                    RestartDiagnostic(
                        restart_index=index,
                        seed=seed,
                        converged=False,
                        iterations=1,
                        log_likelihood=None,
                        valid=False,
                        raw_minimum_eigenvalue=None,
                        maximum_floor_relative_change=None,
                        covariance_condition_number=None,
                        maximum_likelihood_decrease=None,
                        preliminary_chain_passed=False,
                        failure_type="NonConvergence",
                        failure_message="synthetic numerical nonconvergence",
                    )
                    for index, seed in enumerate(seeds)
                ),
            )
        return _fake_fit(features, n_states, seeds)

    result = run_bridge_selection_member(
        _values(193),
        _mask(193),
        dgp="model",
        member="prefix",
        pair=0,
        master_seed=100,
        scientific_version=1,
        registered_bootstrap_indices=bootstrap_indices,
        fit_function=sometimes_fails,
    )
    failed = result.candidates[1]
    assert failed.main_attempt.fit is None
    assert not failed.viability.admissible
    assert all(not attempt.successful for attempt in failed.rolling_attempts)
    assert all(
        attempt.failure_type == "MainFitFailure" for attempt in failed.rolling_attempts
    )
    assert result.selection.selected_n_states == 2


def test_member_propagates_generic_model_fault(
    bootstrap_indices: np.ndarray,
) -> None:
    failure = ModelError("synthetic generic bridge fit defect")

    def fails(
        _features: HMMFeatureMatrix,
        _n_states: int,
        _seeds: Sequence[int],
    ) -> GaussianHMMFit:
        raise failure

    with pytest.raises(
        ModelError,
        match="synthetic generic bridge fit defect",
    ) as caught:
        run_bridge_selection_member(
            _values(193),
            _mask(193),
            dgp="model",
            member="prefix",
            pair=0,
            master_seed=100,
            scientific_version=1,
            registered_bootstrap_indices=bootstrap_indices,
            fit_function=fails,
        )
    assert caught.value is failure


def test_pair_and_exact_250_result_aggregate(
    bootstrap_indices: np.ndarray,
) -> None:
    design = _values(193)
    history = generate_registered_bridge_history(
        dgp="nonparametric",
        tier="selection",
        pair=0,
        master_seed=1234,
        scientific_version=1,
        design_parameters=_parameters(),
        design_standardized_history=design,
        design_continuation=_mask(193),
        macro_block_length=4,
    )
    pair = run_bridge_tier_a_pair(
        history,
        actual_selected_k=2,
        master_seed=1234,
        scientific_version=1,
        registered_bootstrap_indices=bootstrap_indices,
        fit_function=_fake_fit,
    )
    assert pair.success
    assert len(pair.history_sha256) == 64
    assert len(pair.rolling_bootstrap_sha256) == 64
    results: list[BridgeTierAPairResult] = []
    for pair_id in range(250):
        results.append(replace(pair, pair=pair_id))
    execution = assemble_bridge_tier_a(
        results,
        dgp="nonparametric",
        actual_selected_k=2,
    )
    assert execution.gate.successes == 250
    assert execution.gate.passed
    with pytest.raises(ModelError, match="exactly 250"):
        assemble_bridge_tier_a(
            results[:-1],
            dgp="nonparametric",
            actual_selected_k=2,
        )
    inconsistent_success = list(results)
    inconsistent_success[0] = replace(inconsistent_success[0], success=False)
    with pytest.raises(ModelError, match="success flag"):
        assemble_bridge_tier_a(
            inconsistent_success,
            dgp="nonparametric",
            actual_selected_k=2,
        )
    inconsistent_member = list(results)
    inconsistent_member[0] = replace(inconsistent_member[0], prefix=None)
    with pytest.raises(ModelError, match="result/failure evidence"):
        assemble_bridge_tier_a(
            inconsistent_member,
            dgp="nonparametric",
            actual_selected_k=2,
        )
    wrong_identity = list(results)
    wrong_identity[0] = replace(wrong_identity[0], dgp="model")
    with pytest.raises(ModelError, match="identity/order"):
        assemble_bridge_tier_a(
            wrong_identity,
            dgp="nonparametric",
            actual_selected_k=2,
        )


def test_tier_a_member_scientific_failures_are_retained_as_disagreement(
    bootstrap_indices: np.ndarray,
) -> None:
    history = generate_registered_bridge_history(
        dgp="model",
        tier="selection",
        pair=0,
        master_seed=1234,
        scientific_version=1,
        design_parameters=_parameters(),
        design_standardized_history=_values(193),
        design_continuation=_mask(193),
        macro_block_length=4,
    )

    def fails(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ModelError("synthetic scientific selection failure")

    result = run_bridge_tier_a_pair(
        history,
        actual_selected_k=2,
        master_seed=1234,
        scientific_version=1,
        registered_bootstrap_indices=bootstrap_indices,
        fit_function=_fake_fit,
        member_runner=fails,  # type: ignore[arg-type]
    )
    assert not result.success
    assert result.prefix is None
    assert result.full is None
    assert tuple(failure.member for failure in result.member_failures) == (
        "prefix",
        "full",
    )


def test_tier_a_rejects_malformed_bootstrap_evidence_before_execution() -> None:
    history = generate_registered_bridge_history(
        dgp="model",
        tier="selection",
        pair=0,
        master_seed=1234,
        scientific_version=1,
        design_parameters=_parameters(),
        design_standardized_history=_values(193),
        design_continuation=_mask(193),
        macro_block_length=4,
    )
    with pytest.raises(ModelError, match="registered bootstrap indices"):
        run_bridge_tier_a_pair(
            history,
            actual_selected_k=2,
            master_seed=1234,
            scientific_version=1,
            registered_bootstrap_indices=np.zeros((1, 6, 12), dtype=np.int64),
            fit_function=_fake_fit,
        )
    valid_bootstrap = np.broadcast_to(
        np.arange(12, dtype=np.int64),
        (10_000, 6, 12),
    )
    with pytest.raises(ModelError, match="selected K.*design reference"):
        run_bridge_tier_a_pair(
            history,
            actual_selected_k=3,
            master_seed=1234,
            scientific_version=1,
            registered_bootstrap_indices=valid_bootstrap,
            fit_function=_fake_fit,
        )


def test_member_rejects_wrong_full_gap_geometry(
    bootstrap_indices: np.ndarray,
) -> None:
    with pytest.raises(ModelError, match="segment geometry"):
        run_bridge_selection_member(
            _values(217),
            _mask(217),
            dgp="model",
            member="full",
            pair=0,
            master_seed=1,
            scientific_version=1,
            registered_bootstrap_indices=bootstrap_indices,
            fit_function=_fake_fit,
        )


def test_bridge_feature_matrix_rejects_degenerate_or_malformed_inputs() -> None:
    with pytest.raises(ModelError, match="four-column"):
        bridge_feature_matrix(np.zeros((193, 3)), _mask(193))
    with pytest.raises(ModelError, match="proper prefix"):
        bridge_feature_matrix(_values(193), _mask(193), training_rows=193)
    with pytest.raises(ModelError, match="degenerate"):
        bridge_feature_matrix(np.ones((193, 4)), _mask(193))
