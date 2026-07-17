from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import prpg.model.calibration as calibration_module
from prpg.errors import ModelError
from prpg.model.calibration import (
    MacroBootstrapAttempt,
    MacroStabilityEvidence,
    execute_registered_macro_stability,
    is_registered_macro_stability_evidence,
    registered_macro_stability_execution_identity,
    run_macro_stability,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMParameters,
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
from prpg.model.stability import MacroStabilitySummary
from prpg.model.transition_bootstrap import (
    MINIMUM_SUCCESSFUL_REFITS,
    TRANSITION_PROBABILITY_QUANTILES,
    aggregate_transition_bootstrap,
    is_registered_transition_bootstrap_aggregate,
    transition_bootstrap_aggregate_identity,
)

FEATURE_NAMES = (
    "industrial_production_growth",
    "inflation_yoy",
    "yield_curve_spread",
    "credit_spread",
)


def _base_fit() -> GaussianHMMFit:
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5], dtype=np.float64),
        transition_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64),
        means=np.asarray(
            [[-3.0, -0.2, 0.1, -0.1], [3.0, 0.2, -0.1, 0.1]],
            dtype=np.float64,
        ),
        tied_covariance=np.eye(4, dtype=np.float64) * 0.25,
    )
    states = np.tile(np.repeat(np.asarray([0, 1], dtype=np.int64), 5), 20)
    values = parameters.means[states] + np.random.default_rng(55).normal(
        0.0, 0.1, size=(len(states), 4)
    )
    decoded = decode_viterbi(values, parameters, lengths=(len(values),))
    path = summarize_state_path(decoded, 2, lengths=(len(values),))
    posterior = forward_backward(values, parameters, lengths=(len(values),))
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
    assert identification.passed
    assert support.passed
    return GaussianHMMFit(
        parameters=parameters,
        decoded_states=decoded,
        canonical_to_original=np.asarray([0, 1], dtype=np.int64),
        original_to_canonical=np.asarray([0, 1], dtype=np.int64),
        log_likelihood=posterior.total_log_likelihood,
        bic=gaussian_hmm_bic(
            posterior.total_log_likelihood,
            len(values),
            2,
            4,
        ),
        parameter_count=gaussian_hmm_parameter_count(2, 4),
        n_observations=len(values),
        n_features=4,
        n_states=2,
        lengths=(len(values),),
        best_restart_index=0,
        best_seed=1,
        restart_diagnostics=(),
        covariance_diagnostic=tied_covariance_diagnostic(parameters.tied_covariance),
        state_path_diagnostics=path,
        chain_diagnostics=validate_numerical_chain(parameters.transition_matrix),
        forward_backward=posterior,
        identification=identification,
        historical_transition_support=support,
        scaler_scope="design_training",
        scaler_means=np.zeros(4, dtype=np.float64),
        scaler_standard_deviations=np.ones(4, dtype=np.float64),
        feature_names=FEATURE_NAMES,
    )


def _aligned_fit(
    base: GaussianHMMFit,
    aligned_probability: np.ndarray,
    aligned_viterbi: np.ndarray,
    aligned_expected: np.ndarray,
    mapping: np.ndarray,
) -> GaussianHMMFit:
    raw_probability = aligned_probability[np.ix_(mapping, mapping)]
    raw_viterbi = aligned_viterbi[np.ix_(mapping, mapping)]
    raw_expected = aligned_expected[np.ix_(mapping, mapping)]
    return replace(
        base,
        parameters=replace(
            base.parameters,
            transition_matrix=raw_probability.astype(np.float64),
        ),
        state_path_diagnostics=replace(
            base.state_path_diagnostics,
            transition_counts=raw_viterbi.astype(np.int64),
        ),
        forward_backward=replace(
            base.forward_backward,
            expected_transition_counts=raw_expected.astype(np.float64),
        ),
    )


def _evidence(
    base: GaussianHMMFit,
    *,
    successes: int = 90,
) -> MacroStabilityEvidence:
    attempts: list[MacroBootstrapAttempt] = []
    for attempt in range(100):
        if attempt >= successes:
            attempts.append(
                MacroBootstrapAttempt(
                    attempt=attempt,
                    source_indices=np.arange(20, dtype=np.int64),
                    fit=None,
                    adjusted_rand_index=0.0,
                    jaccards=np.zeros(2, dtype=np.float64),
                    failure_type="ModelError",
                    failure_message="synthetic failure",
                    candidate_to_reference=None,
                )
            )
            continue
        mapping = np.asarray([0, 1] if attempt % 2 == 0 else [1, 0], dtype=np.int64)
        aligned_probability = np.asarray(
            [
                [0.600 + 0.001 * attempt, 0.400 - 0.001 * attempt],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )
        aligned_viterbi = np.asarray([[10, 5], [4, 11]], dtype=np.int64)
        if attempt < 10:
            aligned_viterbi[0, 1] = 0
        aligned_expected = np.asarray([[10.0, 4.0], [3.0, 12.0]])
        fitted = _aligned_fit(
            base,
            aligned_probability,
            aligned_viterbi,
            aligned_expected,
            mapping,
        )
        attempts.append(
            MacroBootstrapAttempt(
                attempt=attempt,
                source_indices=np.arange(20, dtype=np.int64),
                fit=fitted,
                adjusted_rand_index=0.9,
                jaccards=np.full(2, 0.8, dtype=np.float64),
                failure_type=None,
                failure_message=None,
                candidate_to_reference=mapping,
            )
        )
    return MacroStabilityEvidence(
        n_states=2,
        role="design",
        macro_block_length=4,
        attempts=tuple(attempts),
        summary=MacroStabilitySummary(
            successful_attempts=successes,
            median_ari=0.9,
            tenth_percentile_ari=0.8,
            median_jaccard_by_state=np.full(2, 0.8),
            tenth_percentile_jaccard_by_state=np.full(2, 0.7),
            passed=successes >= 90,
        ),
    )


def _registered_evidence(main: GaussianHMMFit) -> MacroStabilityEvidence:
    rng = np.random.default_rng(20260714)
    values = rng.normal(size=(main.n_observations, 4))
    month_ordinals = np.arange(600, 600 + main.n_observations, dtype=np.int64)
    continuation = np.ones(main.n_observations, dtype=np.bool_)
    continuation[0] = False

    def fitted(_: Any, n_states: int, __: Any) -> GaussianHMMFit:
        assert n_states == main.n_states
        return main

    original = calibration_module.fit_gaussian_hmm_restarts
    calibration_module.fit_gaussian_hmm_restarts = fitted
    try:
        return execute_registered_macro_stability(
            values,
            month_ordinals,
            continuation,
            role="design",
            main_fit=main,
            macro_block_length=4,
            master_seed=8128,
            scientific_version=1,
        )
    finally:
        calibration_module.fit_gaussian_hmm_restarts = original


def test_registered_macro_evidence_has_object_specific_deep_authority() -> None:
    evidence = _registered_evidence(_base_fit())
    identity = registered_macro_stability_execution_identity(evidence)

    assert is_registered_macro_stability_evidence(evidence)
    assert identity.evidence_fingerprint == evidence._evidence_fingerprint
    assert not is_registered_macro_stability_evidence(copy.copy(evidence))

    changed_summary = replace(evidence.summary, passed=not evidence.summary.passed)
    object.__setattr__(evidence, "summary", changed_summary)
    assert not is_registered_macro_stability_evidence(evidence)


def test_injectable_macro_executor_is_report_only() -> None:
    main = _base_fit()
    rng = np.random.default_rng(88)
    values = rng.normal(size=(main.n_observations, 4))
    ordinals = np.arange(900, 900 + main.n_observations, dtype=np.int64)
    continuation = np.ones(main.n_observations, dtype=np.bool_)
    continuation[0] = False

    def injected(_: Any, n_states: int, __: Any) -> GaussianHMMFit:
        assert n_states == main.n_states
        return main

    evidence = run_macro_stability(
        values,
        ordinals,
        continuation,
        role="design",
        main_fit=main,
        macro_block_length=4,
        master_seed=99,
        scientific_version=1,
        fit_function=injected,
    )

    assert not is_registered_macro_stability_evidence(evidence)


def test_aggregate_aligns_all_matrices_and_keeps_failures_in_denominator() -> None:
    main = _base_fit()
    evidence = _evidence(main, successes=90)
    original_report = main.historical_transition_support
    assert original_report.bootstrap_edge_frequencies is None
    assert original_report.bootstrap_probability_quantiles is None

    result = aggregate_transition_bootstrap(main, evidence)

    assert result.attempts == 100
    assert result.successful_refits == 90
    assert result.failed_refits == 10
    assert result.edge_frequency_denominator == 100
    assert result.quantile_method == "higher"
    assert result.quantile_probabilities == TRANSITION_PROBABILITY_QUANTILES
    # Ten successful aligned refits have unsupported reference edge 0->1;
    # all failures are unsupported everywhere in the fixed denominator.
    assert result.bootstrap_edge_frequencies[0, 1] == pytest.approx(0.80)
    assert result.bootstrap_edge_frequencies[1, 0] == pytest.approx(0.90)
    assert result.bootstrap_edge_frequencies[0, 0] == pytest.approx(0.90)
    expected_p00 = np.quantile(
        np.asarray([0.600 + 0.001 * value for value in range(90)]),
        TRANSITION_PROBABILITY_QUANTILES,
        method="higher",
    )
    np.testing.assert_array_equal(
        result.bootstrap_probability_quantiles[0, 0],
        expected_p00,
    )
    np.testing.assert_array_equal(
        result.historical_transition_support.bootstrap_edge_frequencies,
        result.bootstrap_edge_frequencies,
    )
    np.testing.assert_array_equal(
        result.historical_transition_support.bootstrap_probability_quantiles,
        result.bootstrap_probability_quantiles,
    )
    assert main.historical_transition_support is original_report
    assert original_report.bootstrap_edge_frequencies is None
    assert original_report.bootstrap_probability_quantiles is None
    assert not result.bootstrap_edge_frequencies.flags.writeable
    assert not result.bootstrap_probability_quantiles.flags.writeable
    assert not is_registered_transition_bootstrap_aggregate(result)


def test_registered_aggregate_revalidates_both_parents_and_rejects_copies() -> None:
    main = _base_fit()
    evidence = _registered_evidence(main)
    result = aggregate_transition_bootstrap(main, evidence)
    identity = transition_bootstrap_aggregate_identity(result)

    assert is_registered_transition_bootstrap_aggregate(result)
    assert identity.role == "design"
    assert identity.n_states == 2
    assert identity.master_seed == 8128
    assert identity.scientific_version == 1
    assert identity.parent_model_fingerprint == result.parent_model_fingerprint
    assert identity.macro_input_fingerprint == result.macro_input_fingerprint
    assert identity.macro_evidence_fingerprint == result.macro_evidence_fingerprint
    assert identity.source_materialization_fingerprint == (
        result.source_materialization_fingerprint
    )
    assert identity.rng_execution_fingerprint == result.rng_execution_fingerprint
    assert identity.execution_fingerprint == result.execution_fingerprint
    assert not is_registered_transition_bootstrap_aggregate(copy.copy(result))
    assert not is_registered_transition_bootstrap_aggregate(replace(result))

    with pytest.raises(ValueError):
        result.bootstrap_edge_frequencies.setflags(write=True)
    with pytest.raises(ValueError):
        result.historical_transition_support.probabilities.setflags(write=True)


@pytest.mark.parametrize(
    ("field", "value"),
    (("_master_seed", 8129), ("_scientific_version", 2), ("n_states", 3)),
)
def test_registered_aggregate_rejects_tampered_contract_fields(
    field: str,
    value: object,
) -> None:
    main = _base_fit()
    result = aggregate_transition_bootstrap(main, _registered_evidence(main))
    object.__setattr__(result, field, value)
    assert not is_registered_transition_bootstrap_aggregate(result)


def test_registered_aggregate_rejects_changed_live_parent_content() -> None:
    main = _base_fit()
    evidence = _registered_evidence(main)
    result = aggregate_transition_bootstrap(main, evidence)
    main.parameters.transition_matrix.setflags(write=True)
    main.parameters.transition_matrix[0, 0] = 0.79
    assert not is_registered_transition_bootstrap_aggregate(result)


def test_registered_aggregate_rejects_changed_macro_attempt_content() -> None:
    main = _base_fit()
    evidence = _registered_evidence(main)
    result = aggregate_transition_bootstrap(main, evidence)
    indices = evidence.attempts[0].source_indices
    with pytest.raises(ValueError):
        indices.setflags(write=True)
    changed = indices.copy()
    changed[0] = (int(changed[0]) + 1) % len(changed)
    object.__setattr__(
        evidence,
        "attempts",
        (
            replace(evidence.attempts[0], source_indices=changed),
            *evidence.attempts[1:],
        ),
    )
    assert not is_registered_transition_bootstrap_aggregate(result)


def test_registered_aggregate_rejects_macro_evidence_from_another_main_fit() -> None:
    main = _base_fit()
    evidence = _registered_evidence(main)
    changed = replace(
        main,
        parameters=replace(
            main.parameters,
            means=main.parameters.means + 0.01,
        ),
    )
    with pytest.raises(ModelError, match="belongs to another main fit"):
        aggregate_transition_bootstrap(changed, evidence)


def test_compact_attempt_audit_preserves_permutations_and_failure_slots() -> None:
    result = aggregate_transition_bootstrap(_base_fit(), _evidence(_base_fit()))
    assert len(result.attempt_audits) == 100
    assert [audit.attempt for audit in result.attempt_audits] == list(range(100))
    assert result.attempt_audits[0].candidate_to_reference == (0, 1)
    assert result.attempt_audits[1].candidate_to_reference == (1, 0)
    assert result.attempt_audits[0].supported_cell_count == 3
    assert result.attempt_audits[10].supported_cell_count == 4
    assert len(result.attempt_audits[0].aligned_input_sha256 or "") == 64
    for audit in result.attempt_audits[90:]:
        assert not audit.successful
        assert audit.candidate_to_reference is None
        assert audit.supported_cell_count == 0
        assert audit.aligned_input_sha256 is None


def test_fewer_than_90_successes_fails_closed() -> None:
    main = _base_fit()
    with pytest.raises(ModelError, match="fewer than 90") as captured:
        aggregate_transition_bootstrap(main, _evidence(main, successes=89))
    assert captured.value.details == {
        "actual": 89,
        "required": MINIMUM_SUCCESSFUL_REFITS,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda evidence: replace(evidence, attempts=evidence.attempts[:-1]),
            "exactly 100",
        ),
        (
            lambda evidence: replace(
                evidence,
                attempts=(
                    replace(evidence.attempts[0], attempt=1),
                    *evidence.attempts[1:],
                ),
            ),
            "exactly 0..99",
        ),
        (
            lambda evidence: replace(
                evidence,
                attempts=(
                    replace(
                        evidence.attempts[0],
                        candidate_to_reference=np.asarray([0, 0]),
                    ),
                    *evidence.attempts[1:],
                ),
            ),
            "exact permutation",
        ),
        (
            lambda evidence: replace(
                evidence,
                attempts=(
                    replace(evidence.attempts[0], candidate_to_reference=None),
                    *evidence.attempts[1:],
                ),
            ),
            "missing its label permutation",
        ),
        (
            lambda evidence: replace(
                evidence,
                summary=replace(evidence.summary, successful_attempts=99),
            ),
            "success metadata",
        ),
    ],
)
def test_attempt_ids_permutations_and_metadata_fail_closed(
    mutate: object, message: str
) -> None:
    main = _base_fit()
    evidence = _evidence(main, successes=90)
    with pytest.raises(ModelError, match=message):
        aggregate_transition_bootstrap(main, mutate(evidence))  # type: ignore[operator]


def test_failed_attempt_cannot_retain_an_alignment() -> None:
    main = _base_fit()
    evidence = _evidence(main, successes=90)
    changed = replace(
        evidence,
        attempts=(
            *evidence.attempts[:90],
            replace(
                evidence.attempts[90],
                candidate_to_reference=np.asarray([0, 1], dtype=np.int64),
            ),
            *evidence.attempts[91:],
        ),
    )
    with pytest.raises(ModelError, match="retains fit or alignment"):
        aggregate_transition_bootstrap(main, changed)


def test_successful_refit_k_must_match_main_fit() -> None:
    main = _base_fit()
    evidence = _evidence(main, successes=90)
    assert evidence.attempts[0].fit is not None
    wrong_k = replace(evidence.attempts[0].fit, n_states=3)
    changed = replace(
        evidence,
        attempts=(replace(evidence.attempts[0], fit=wrong_k), *evidence.attempts[1:]),
    )
    with pytest.raises(ModelError, match="K does not match"):
        aggregate_transition_bootstrap(main, changed)


def test_top_level_types_and_state_counts_fail_closed() -> None:
    main = _base_fit()
    evidence = _evidence(main)
    with pytest.raises(ModelError, match="main fit must be"):
        aggregate_transition_bootstrap(object(), evidence)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="must be MacroStabilityEvidence"):
        aggregate_transition_bootstrap(main, object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="unregistered K"):
        aggregate_transition_bootstrap(replace(main, n_states=1), evidence)
    with pytest.raises(ModelError, match="evidence K"):
        aggregate_transition_bootstrap(main, replace(evidence, n_states=3))

    bad_attempt = replace(evidence.attempts[0], fit=object())  # type: ignore[arg-type]
    bad_evidence = replace(
        evidence,
        attempts=(bad_attempt, *evidence.attempts[1:]),
    )
    with pytest.raises(ModelError, match="invalid fit record"):
        aggregate_transition_bootstrap(main, bad_evidence)
