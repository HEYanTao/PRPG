from __future__ import annotations

import copy
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

import prpg.model.calibration as calibration_module
import prpg.model.calibration_identity as calibration_identity_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.block_length import (
    DiagnosticSeriesSet,
    FrequencyCalibration,
    FrequencyWideLength,
    MacroCalibration,
    MacroWideLength,
    PreliminaryBlockLengthCapFailure,
    PreliminaryPPWInsufficientSupport,
    PreliminaryPPWNumericalFailure,
)
from prpg.model.calibration import (
    CandidateFitAttempt,
    CandidateFitSet,
    PreliminaryCalibrations,
    RegisteredPreliminaryCalibrationOutcome,
    RegisteredRollingBootstrapMaterialization,
    decoded_return_pools,
    evaluate_sealed_holdout,
    execute_registered_hmm_candidate_set,
    execute_registered_preliminary_calibrations,
    fit_hmm_candidate_set,
    fit_production_selected_k,
    fit_rolling_candidate,
    is_registered_candidate_fit_set,
    is_registered_preliminary_calibration_outcome,
    is_registered_rolling_bootstrap_materialization,
    macro_bootstrap_indices_from_uniforms,
    materialize_registered_rolling_bootstrap,
    materialize_rolling_bootstrap_indices,
    preliminary_block_calibrations,
    preliminary_calibrations_fingerprint,
    prepare_hmm_feature_matrix,
    registered_candidate_fit_set_identity,
    registered_preliminary_calibration_identity,
    registered_preliminary_calibrations,
    rolling_candidate_input_fingerprint,
    run_macro_stability,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMCandidateInadmissibility,
    HMMCandidateInadmissibilityEvidence,
    HMMFeatureMatrix,
    HMMParameters,
    NoNumericallyValidHMMRestarts,
    RestartDiagnostic,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
)
from prpg.model.input import CalibrationSlice
from prpg.model.regime_policy import OWNER_FIXED_K4_PROSPECTIVE_POLICY
from prpg.simulation.rng import CalibrationArtifact, ModelFitScope


def _macro(rows: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(rows, 4))
    months = pd.period_range("2020-01", periods=rows, freq="M").asi8
    mask = np.ones(rows, dtype=np.bool_)
    mask[0] = False
    return values, months, mask


def _dummy_diagnostics(columns: int) -> DiagnosticSeriesSet:
    raw = np.arange(4 * columns, dtype=np.float64).reshape(4, columns)
    return DiagnosticSeriesSet(
        names=tuple(f"series_{index}" for index in range(columns)),
        raw=raw,
        standardized=raw,
        pc1_loadings=np.ones(3, dtype=np.float64),
    )


def _dummy_frequency_calibration(
    frequency: Literal["monthly", "daily"],
) -> FrequencyCalibration:
    upper = 12 if frequency == "monthly" else 60
    start = 4 if frequency == "monthly" else 20
    return FrequencyCalibration(
        policy_version="test-policy",
        diagnostics=_dummy_diagnostics(7),
        estimates=(),
        combined=FrequencyWideLength(
            frequency=frequency,
            diagnostic_lengths=(("series_0", float(start)),),
            controlling_series=("series_0",),
            raw_ceiling_length=start,
            starting_length=start,
            lower_bound=2,
            upper_bound=upper,
            reasons=(),
        ),
    )


def _dummy_macro_calibration() -> MacroCalibration:
    return MacroCalibration(
        policy_version="test-policy",
        diagnostics=_dummy_diagnostics(7),
        estimates=(),
        combined=MacroWideLength(
            diagnostic_lengths=(("series_0", 3.0),),
            controlling_series=("series_0",),
            raw_ceiling_length=3,
            starting_length=3,
            lower_bound=2,
            absolute_upper_bound=24,
            sample_support_upper_bound=8,
        ),
    )


def _ppw_slice(role: Literal["design", "full"] = "design") -> CalibrationSlice:
    monthly_rows = 8
    daily_rows = 16
    monthly_continuation = np.ones(monthly_rows, dtype=np.bool_)
    daily_continuation = np.ones(daily_rows, dtype=np.bool_)
    monthly_continuation[0] = False
    daily_continuation[0] = False
    return CalibrationSlice(
        role=role,
        monthly_returns=np.arange(monthly_rows * 3, dtype=np.float64).reshape(
            monthly_rows, 3
        ),
        daily_returns=np.arange(daily_rows * 3, dtype=np.float64).reshape(
            daily_rows, 3
        ),
        macro_features=np.arange(monthly_rows * 4, dtype=np.float64).reshape(
            monthly_rows, 4
        ),
        monthly_period_ordinals=np.arange(monthly_rows, dtype=np.int64),
        daily_session_ns=np.arange(daily_rows, dtype=np.int64),
        monthly_continuation=monthly_continuation,
        daily_continuation=daily_continuation,
    )


def _install_successful_ppw_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        calibration_module,
        "estimate_frequency_start",
        lambda _returns, *, frequency, continuation: _dummy_frequency_calibration(
            frequency
        ),
    )
    monkeypatch.setattr(
        calibration_module,
        "estimate_macro_start",
        lambda _features, *, continuation: _dummy_macro_calibration(),
    )


def test_preliminary_block_calibrations_execute_all_three_independent_ppw_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def frequency_start(
        returns: np.ndarray,
        *,
        frequency: Literal["monthly", "daily"],
        continuation: np.ndarray,
    ) -> FrequencyCalibration:
        assert len(returns) == len(continuation)
        calls.append((frequency, len(returns)))
        return _dummy_frequency_calibration(frequency)

    def macro_start(
        features: np.ndarray,
        *,
        continuation: np.ndarray,
    ) -> MacroCalibration:
        assert len(features) == len(continuation)
        calls.append(("macro", len(features)))
        return _dummy_macro_calibration()

    monkeypatch.setattr(calibration_module, "estimate_frequency_start", frequency_start)
    monkeypatch.setattr(calibration_module, "estimate_macro_start", macro_start)

    result = preliminary_block_calibrations(_ppw_slice(), role="design")

    assert result.role == "design"
    assert result.monthly.combined.starting_length == 4
    assert result.daily.combined.starting_length == 20
    assert result.macro.combined.starting_length == 3
    assert calls == [("monthly", 8), ("daily", 16), ("macro", 8)]


def test_registered_preliminary_success_reverifies_exact_source_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_ppw_stubs(monkeypatch)
    source = _ppw_slice()

    outcome = execute_registered_preliminary_calibrations(source, role="design")
    identity = registered_preliminary_calibration_identity(outcome)

    assert identity.passed
    assert identity.failure_stage is None
    assert identity.result_fingerprint is not None
    assert outcome.calibrations is registered_preliminary_calibrations(outcome)
    assert is_registered_preliminary_calibration_outcome(outcome)
    assert not is_registered_preliminary_calibration_outcome(object())
    assert len(preliminary_calibrations_fingerprint(outcome.calibrations)) == 64


def test_registered_preliminary_mint_is_atomic_when_self_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_ppw_stubs(monkeypatch)
    retained_ids = set(calibration_module._MINTED_PRELIMINARY_CALIBRATIONS)

    def reject_identity(_value: object) -> object:
        raise ModelError("synthetic preliminary self-verification failure")

    monkeypatch.setattr(
        calibration_module,
        "registered_preliminary_calibration_identity",
        reject_identity,
    )
    with pytest.raises(ModelError, match="self-verification failure"):
        execute_registered_preliminary_calibrations(_ppw_slice(), role="design")
    assert set(calibration_module._MINTED_PRELIMINARY_CALIBRATIONS) == retained_ids


def test_registered_preliminary_identity_rejects_wrong_type_copy_and_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_ppw_stubs(monkeypatch)
    outcome = execute_registered_preliminary_calibrations(
        _ppw_slice(),
        role="design",
    )

    with pytest.raises(ModelError, match="wrong type"):
        registered_preliminary_calibration_identity(object())  # type: ignore[arg-type]
    copied = copy.copy(outcome)
    with pytest.raises(ModelError, match="producer authority"):
        registered_preliminary_calibration_identity(copied)
    original = outcome.outcome_fingerprint
    object.__setattr__(outcome, "outcome_fingerprint", "0" * 64)
    try:
        with pytest.raises(ModelError, match="source or result changed"):
            registered_preliminary_calibration_identity(outcome)
        assert not is_registered_preliminary_calibration_outcome(outcome)
    finally:
        object.__setattr__(outcome, "outcome_fingerprint", original)
    assert is_registered_preliminary_calibration_outcome(outcome)


@pytest.mark.parametrize(
    ("failure_stage", "error_type", "failure_code"),
    (
        ("monthly", PreliminaryPPWInsufficientSupport, "insufficient_support"),
        ("daily", PreliminaryPPWNumericalFailure, "numerical_failure"),
        ("macro", PreliminaryBlockLengthCapFailure, "hard_cap_exceeded"),
    ),
)
def test_registered_preliminary_scientific_failures_preserve_stage_and_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    error_type: type[ModelError],
    failure_code: str,
) -> None:
    def scientific_failure() -> None:
        raise error_type(
            f"synthetic {failure_stage} scientific nonattainment",
            details={"observed": 108, "context": {"limits": [2, 60]}},
        )

    def frequency_start(
        _returns: np.ndarray,
        *,
        frequency: Literal["monthly", "daily"],
        continuation: np.ndarray,
    ) -> FrequencyCalibration:
        del continuation
        if frequency == failure_stage:
            scientific_failure()
        return _dummy_frequency_calibration(frequency)

    def macro_start(
        _features: np.ndarray,
        *,
        continuation: np.ndarray,
    ) -> MacroCalibration:
        del continuation
        if failure_stage == "macro":
            scientific_failure()
        return _dummy_macro_calibration()

    monkeypatch.setattr(calibration_module, "estimate_frequency_start", frequency_start)
    monkeypatch.setattr(calibration_module, "estimate_macro_start", macro_start)

    outcome = execute_registered_preliminary_calibrations(
        _ppw_slice(),
        role="design",
    )
    identity = registered_preliminary_calibration_identity(outcome)

    assert not identity.passed
    assert identity.failure_stage == failure_stage
    assert identity.failure_code == failure_code
    assert identity.failure_type == error_type.__name__
    assert identity.failure_details == (
        ("context", (("limits", (2, 60)),)),
        ("observed", 108),
    )
    with pytest.raises(ModelError, match="failed preliminary outcome"):
        registered_preliminary_calibrations(outcome)


def test_preliminary_executor_rejects_unsupported_failure_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise PreliminaryPPWNumericalFailure(
            "invalid scientific details",
            details={"unsupported": {1, 2}},
        )

    monkeypatch.setattr(calibration_module, "estimate_frequency_start", fail)
    with pytest.raises(ModelError, match="unsupported value"):
        execute_registered_preliminary_calibrations(_ppw_slice(), role="design")


def test_preliminary_public_contracts_fail_closed_on_forgery_or_wrong_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_ppw_stubs(monkeypatch)
    with pytest.raises(TypeError, match="created only by its executor"):
        RegisteredPreliminaryCalibrationOutcome()
    with pytest.raises(ModelError, match="requires CalibrationSlice"):
        preliminary_block_calibrations(object(), role="design")  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="role must be design or production"):
        preliminary_block_calibrations(_ppw_slice(), role="wrong")  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="does not match"):
        preliminary_block_calibrations(_ppw_slice("full"), role="design")
    with pytest.raises(ModelError, match="preliminary calibration is invalid"):
        preliminary_calibrations_fingerprint(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="preliminary calibration is invalid"):
        preliminary_calibrations_fingerprint(
            PreliminaryCalibrations(
                "wrong",  # type: ignore[arg-type]
                _dummy_frequency_calibration("monthly"),
                _dummy_frequency_calibration("daily"),
                _dummy_macro_calibration(),
            )
        )


def test_registered_preliminary_production_requires_the_full_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_ppw_stubs(monkeypatch)
    outcome = execute_registered_preliminary_calibrations(
        _ppw_slice("full"),
        role="production",
    )
    assert registered_preliminary_calibrations(outcome).role == "production"


def test_prepare_design_and_production_scalers_have_exact_scope() -> None:
    values, months, mask = _macro()
    design = prepare_hmm_feature_matrix(
        values,
        months,
        mask,
        role="design",
        design_rows=30,
    )
    production = prepare_hmm_feature_matrix(
        values,
        months,
        mask,
        role="production",
    )

    assert design.values.shape == (30, 4)
    np.testing.assert_allclose(design.values.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(design.values.std(axis=0), 1.0, atol=1e-14)
    assert design.scaler_scope == "design_training"
    assert production.values.shape == (40, 4)
    assert production.scaler_scope == "production_full"


def test_prepare_features_retains_gap_geometry() -> None:
    values, months, mask = _macro()
    mask[20] = False
    result = prepare_hmm_feature_matrix(values, months, mask, role="production")
    assert result.lengths == (20, 20)


def test_candidate_set_retains_successes_and_scientific_failures() -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(values, months, mask, role="production")

    def fake_fit(
        supplied: HMMFeatureMatrix, n_states: int, seeds: object
    ) -> GaussianHMMFit:
        assert supplied is features
        checked_seeds = cast(tuple[int, ...], seeds)
        assert len(checked_seeds) == 50
        if n_states == 3:
            raise NoNumericallyValidHMMRestarts(
                n_states=n_states,
                restart_diagnostics=tuple(
                    RestartDiagnostic(
                        restart_index=index,
                        seed=seed,
                        converged=False,
                        iterations=0,
                        log_likelihood=None,
                        valid=False,
                        raw_minimum_eigenvalue=None,
                        maximum_floor_relative_change=None,
                        covariance_condition_number=None,
                        maximum_likelihood_decrease=None,
                        preliminary_chain_passed=False,
                        failure_type="NonConvergence",
                        failure_message="synthetic numerical failure",
                    )
                    for index, seed in enumerate(checked_seeds)
                ),
            )
        return cast(GaussianHMMFit, _FitStub(n_states))

    result = fit_hmm_candidate_set(
        features,
        master_seed=5,
        scientific_version=1,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        fit_function=fake_fit,
    )

    assert [attempt.n_states for attempt in result.attempts] == [2, 3, 4, 5]
    assert [attempt.valid for attempt in result.attempts] == [True, False, True, True]
    assert result.attempts[1].failure_type == "NoNumericallyValidHMMRestarts"
    assert result.attempts[1].failure_details is not None
    assert result.attempts[1].failure_details["n_states"] == 3
    assert result.passed


def test_candidate_set_retains_k5_posterior_mass_rejection_evidence() -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(values, months, mask, role="production")
    k5_seeds = deterministic_hmm_restart_seeds(
        master_seed=5,
        scientific_version=1,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=5,
    )
    diagnostics = tuple(
        RestartDiagnostic(
            restart_index=index,
            seed=seed,
            converged=True,
            iterations=3,
            log_likelihood=float(index),
            valid=True,
            raw_minimum_eigenvalue=1.0,
            maximum_floor_relative_change=0.0,
            covariance_condition_number=1.0,
            maximum_likelihood_decrease=0.0,
            preliminary_chain_passed=True,
            failure_type=None,
            failure_message=None,
        )
        for index, seed in enumerate(k5_seeds)
    )
    rejection = HMMCandidateInadmissibility(
        HMMCandidateInadmissibilityEvidence(
            n_states=5,
            failure_code="posterior_identification",
            failure_message="HMM posterior-identification gates failed",
            rejection_reasons=("state_4_posterior_effective_mass_below_minimum",),
            gate_details=(
                (
                    "assigned_state_mean_posteriors",
                    (0.9, 0.9, 0.9, 0.9, 0.9),
                ),
                ("mean_maximum_posterior", 0.9),
                ("minimum_pairwise_mahalanobis_distance", 1.0),
                ("minimum_required_effective_mass", 24.0),
                ("normalized_posterior_entropy", 0.1),
                ("passed", False),
                ("posterior_effective_masses", (40.0, 40.0, 40.0, 40.0, 1.0)),
                (
                    "rejection_reasons",
                    ("state_4_posterior_effective_mass_below_minimum",),
                ),
            ),
            best_restart_index=49,
            best_seed=k5_seeds[49],
            restart_diagnostics=diagnostics,
        )
    )
    calls: list[int] = []

    def fake_fit(
        supplied: HMMFeatureMatrix, n_states: int, seeds: object
    ) -> GaussianHMMFit:
        assert supplied is features
        assert len(cast(tuple[int, ...], seeds)) == 50
        calls.append(n_states)
        if n_states == 5:
            raise rejection
        return cast(GaussianHMMFit, _FitStub(n_states))

    result = fit_hmm_candidate_set(
        features,
        master_seed=5,
        scientific_version=1,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        fit_function=fake_fit,
    )

    assert calls == [2, 3, 4, 5]
    assert [attempt.valid for attempt in result.attempts] == [True, True, True, False]
    failed = result.attempts[3]
    assert failed.failure_type == "HMMCandidateInadmissibility"
    assert failed.failure_details is not None
    assert failed.failure_details["failure_code"] == "posterior_identification"
    gate_details = dict(
        cast(tuple[tuple[str, object], ...], failed.failure_details["gate_details"])
    )
    assert gate_details["posterior_effective_masses"] == (
        40.0,
        40.0,
        40.0,
        40.0,
        1.0,
    )
    with pytest.raises(TypeError):
        failed.failure_details["failure_code"] = "tampered"  # type: ignore[index]


@pytest.mark.parametrize(
    "failure",
    [
        ModelError("unexpected generic fault"),
        IntegrityError("unexpected integrity fault"),
        RuntimeError("unexpected runtime fault"),
        OSError("unexpected filesystem fault"),
    ],
)
def test_candidate_set_propagates_generic_and_operational_faults(
    failure: BaseException,
) -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(values, months, mask, role="production")

    def fail(*_args: object) -> GaussianHMMFit:
        raise failure

    with pytest.raises(type(failure)) as caught:
        fit_hmm_candidate_set(
            features,
            master_seed=5,
            scientific_version=1,
            scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
            replicate=0,
            fit_function=fail,
        )
    assert caught.value is failure


def test_production_selected_k_has_no_reselection_path() -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(values, months, mask, role="production")
    calls: list[int] = []

    def fake_fit(
        supplied: HMMFeatureMatrix, n_states: int, seeds: object
    ) -> GaussianHMMFit:
        calls.append(n_states)
        assert supplied is features
        assert len(cast(tuple[int, ...], seeds)) == 50
        result = _FitStub(n_states)
        result.n_observations = len(features.values)
        result.lengths = features.lengths
        result.scaler_scope = "production_full"
        return cast(GaussianHMMFit, result)

    fitted = fit_production_selected_k(
        features,
        selected_n_states=4,
        master_seed=5,
        scientific_version=1,
        fit_function=fake_fit,
    )
    assert fitted.n_states == 4
    assert calls == [4]

    with pytest.raises(ModelError, match="inconsistent identity"):
        fit_production_selected_k(
            features,
            selected_n_states=4,
            master_seed=5,
            scientific_version=1,
            fit_function=lambda *_args: cast(GaussianHMMFit, _FitStub(3)),
        )


class _FitStub:
    def __init__(self, n_states: int) -> None:
        self.n_states = n_states


def test_decoded_return_pools_maps_by_return_ending_month_and_gaps() -> None:
    months = pd.period_range("2024-01", periods=3, freq="M")
    sessions = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-03-28"])
    data = CalibrationSlice(
        role="full",
        monthly_returns=np.ones((3, 3)),
        daily_returns=np.ones((3, 3)),
        macro_features=np.ones((3, 4)),
        monthly_period_ordinals=months.asi8,
        daily_session_ns=sessions.asi8,
        monthly_continuation=np.asarray([False, True, True]),
        daily_continuation=np.asarray([False, False, False]),
    )
    fit = cast(GaussianHMMFit, _DecodedFitStub())

    result = decoded_return_pools(fit, data)

    assert result.monthly_states.tolist() == [0, 1, 0]
    assert result.daily_states.tolist() == [0, 1, 0]
    assert result.daily_diagnostics.lengths == (1, 1, 1)  # type: ignore[attr-defined]


class _DecodedFitStub:
    decoded_states = np.asarray([0, 1, 0], dtype=np.int64)
    n_states = 2


def test_rolling_candidate_produces_exact_72_scores_and_six_alignments() -> None:
    values, months, mask = _macro(193)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        means=np.asarray([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )
    # Main labels need only be a fixed canonical reference for this orchestration test.
    main_values = (values - values.mean(axis=0)) / values.std(axis=0)
    main_states = decode_viterbi(main_values, parameters, lengths=(193,))
    main = cast(GaussianHMMFit, _MainRollingFit(parameters, main_states))

    def fake_fit(
        features: HMMFeatureMatrix, n_states: int, seeds: object
    ) -> GaussianHMMFit:
        assert n_states == 2
        assert len(cast(tuple[int, ...], seeds)) == 50
        return cast(GaussianHMMFit, _FoldRollingFit(parameters, features))

    result = fit_rolling_candidate(
        values,
        months,
        mask,
        role="design",
        main_fit=main,
        master_seed=7,
        scientific_version=1,
        fit_function=fake_fit,
    )

    assert len(result.attempts) == 6
    assert len(result.rolling_scores) == 72
    assert all(score is not None for score in result.rolling_scores)
    assert result.stability.successful_fits == 6


class _ChainStub:
    stationary_distribution = np.asarray([0.5, 0.5])


class _ForwardStub:
    def __init__(self, rows: int) -> None:
        self.filtered_probabilities = np.full((rows, 2), 0.5)


class _MainRollingFit:
    def __init__(self, parameters: HMMParameters, states: np.ndarray) -> None:
        self.parameters = parameters
        self.decoded_states = states
        self.n_states = 2


class _FoldRollingFit:
    def __init__(self, parameters: HMMParameters, features: HMMFeatureMatrix) -> None:
        self.parameters = parameters
        self.n_states = 2
        self.forward_backward = _ForwardStub(len(features.values))
        self.chain_diagnostics = _ChainStub()


def _all_restart_failure(
    n_states: int,
    seeds: object,
) -> NoNumericallyValidHMMRestarts:
    checked = cast(tuple[int, ...], seeds)
    return NoNumericallyValidHMMRestarts(
        n_states=n_states,
        restart_diagnostics=tuple(
            RestartDiagnostic(
                restart_index=index,
                seed=seed,
                converged=False,
                iterations=0,
                log_likelihood=None,
                valid=False,
                raw_minimum_eigenvalue=None,
                maximum_floor_relative_change=None,
                covariance_condition_number=None,
                maximum_likelihood_decrease=None,
                preliminary_chain_passed=False,
                failure_type="NonConvergence",
                failure_message="synthetic numerical failure",
            )
            for index, seed in enumerate(checked)
        ),
    )


def test_rolling_candidate_reduces_only_all_restart_numerical_failure() -> None:
    values, months, mask = _macro(193)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        means=np.asarray([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )
    standardized = (values - values.mean(axis=0)) / values.std(axis=0)
    states = decode_viterbi(standardized, parameters, lengths=(193,))
    main = cast(GaussianHMMFit, _MainRollingFit(parameters, states))

    with pytest.raises(ModelError, match="structural rolling fault"):
        fit_rolling_candidate(
            values,
            months,
            mask,
            role="design",
            main_fit=main,
            master_seed=7,
            scientific_version=1,
            fit_function=lambda *_args: (_ for _ in ()).throw(
                ModelError("structural rolling fault")
            ),
        )

    result = fit_rolling_candidate(
        values,
        months,
        mask,
        role="design",
        main_fit=main,
        master_seed=7,
        scientific_version=1,
        fit_function=lambda _features, n_states, seeds: (_ for _ in ()).throw(
            _all_restart_failure(n_states, seeds)
        ),
    )
    assert len(result.attempts) == 6
    assert result.stability.successful_fits == 0
    assert all(
        item.failure_type == "NoNumericallyValidHMMRestarts" for item in result.attempts
    )


def test_materialized_rolling_indices_use_all_registered_replicates() -> None:
    first = materialize_rolling_bootstrap_indices(
        master_seed=99,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )
    second = materialize_rolling_bootstrap_indices(
        master_seed=99,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )

    assert first.shape == (10_000, 6, 12)
    assert int(first.min()) >= 0 and int(first.max()) <= 11
    np.testing.assert_array_equal(first, second)
    assert not first.flags.writeable


def test_registered_rolling_bootstrap_is_deterministic_sealed_and_immutable() -> None:
    first = materialize_registered_rolling_bootstrap(
        master_seed=99,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )
    second = materialize_registered_rolling_bootstrap(
        master_seed=99,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )
    changed = materialize_registered_rolling_bootstrap(
        master_seed=100,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )

    assert is_registered_rolling_bootstrap_materialization(first)
    assert first.indices.shape == (10_000, 6, 12)
    assert first.indices.dtype == np.dtype("<i8")
    assert first.materialization_fingerprint == second.materialization_fingerprint
    assert first.rng_execution_fingerprint == second.rng_execution_fingerprint
    assert first.materialization_fingerprint != changed.materialization_fingerprint
    assert first.rng_execution_fingerprint != changed.rng_execution_fingerprint
    np.testing.assert_array_equal(first.indices, second.indices)
    with pytest.raises(ValueError, match="WRITEABLE"):
        first.indices.setflags(write=True)

    assert not is_registered_rolling_bootstrap_materialization(copy.copy(first))
    assert not is_registered_rolling_bootstrap_materialization(replace(first))
    forged = RegisteredRollingBootstrapMaterialization(
        first.contract_version,
        first.master_seed,
        first.scientific_version,
        first.artifact,
        first.indices,
        first.materialization_fingerprint,
        first.rng_execution_fingerprint,
    )
    assert not is_registered_rolling_bootstrap_materialization(forged)

    original = first.rng_execution_fingerprint
    object.__setattr__(first, "rng_execution_fingerprint", "0" * 64)
    assert not is_registered_rolling_bootstrap_materialization(first)
    object.__setattr__(first, "rng_execution_fingerprint", original)
    assert is_registered_rolling_bootstrap_materialization(first)


@pytest.mark.parametrize(
    "overrides",
    (
        {"master_seed": -1},
        {"master_seed": True},
        {"scientific_version": 0},
        {"scientific_version": True},
        {"artifact": 0},
    ),
)
def test_registered_rolling_bootstrap_rejects_wrong_seed_version_or_artifact(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "master_seed": 99,
        "scientific_version": 1,
        "artifact": CalibrationArtifact.DESIGN_OR_CONTROL,
    }
    values.update(overrides)
    with pytest.raises(ModelError, match="rolling bootstrap"):
        materialize_registered_rolling_bootstrap(**values)  # type: ignore[arg-type]


def test_macro_bootstrap_trace_forces_target_and_source_boundaries() -> None:
    mask = np.asarray([False, True, True, False, True], dtype=np.bool_)
    restart = np.full(5, 0.99)
    ranks = np.asarray([0.0, 0.0, 0.0, 0.8, 0.0])
    trace = macro_bootstrap_indices_from_uniforms(mask, restart, ranks, block_length=2)

    assert trace.tolist() == [0, 1, 2, 4, 0]
    assert not trace.flags.writeable


def test_macro_bootstrap_trace_rejects_invalid_geometry_and_uniforms() -> None:
    valid_mask = np.asarray([False, True, True], dtype=np.bool_)
    valid_uniforms = np.asarray([0.1, 0.2, 0.3])
    invalid_masks = (
        np.asarray([], dtype=np.bool_),
        np.asarray([[False, True]], dtype=np.bool_),
        np.asarray([0, 1, 1], dtype=np.int64),
    )
    for mask in invalid_masks:
        with pytest.raises(ModelError, match="non-empty boolean row mask"):
            macro_bootstrap_indices_from_uniforms(
                mask,
                valid_uniforms,
                valid_uniforms,
                block_length=2,
            )
    with pytest.raises(ModelError, match="false at the first row"):
        macro_bootstrap_indices_from_uniforms(
            np.asarray([True, True, True]),
            valid_uniforms,
            valid_uniforms,
            block_length=2,
        )
    for block_length in (True, 2.5):
        with pytest.raises(ModelError, match="must be an integer"):
            macro_bootstrap_indices_from_uniforms(
                valid_mask,
                valid_uniforms,
                valid_uniforms,
                block_length=block_length,  # type: ignore[arg-type]
            )
    with pytest.raises(ModelError, match="at least two"):
        macro_bootstrap_indices_from_uniforms(
            valid_mask,
            valid_uniforms,
            valid_uniforms,
            block_length=1,
        )

    for invalid in (
        np.asarray([0.1, 0.2]),
        np.asarray([0.1, np.nan, 0.3]),
    ):
        with pytest.raises(ModelError, match="exactly 3 finite values"):
            macro_bootstrap_indices_from_uniforms(
                valid_mask,
                invalid,
                valid_uniforms,
                block_length=2,
            )
    for invalid in (
        np.asarray([-0.1, 0.2, 0.3]),
        np.asarray([0.1, 0.2, 1.0]),
    ):
        with pytest.raises(ModelError, match=r"must lie in \[0, 1\)"):
            macro_bootstrap_indices_from_uniforms(
                valid_mask,
                valid_uniforms,
                invalid,
                block_length=2,
            )


def test_rolling_candidate_input_fingerprint_rejects_invalid_macro_geometry() -> None:
    values, months, mask = _macro(72)
    nonfinite = values.copy()
    nonfinite[1, 1] = np.inf
    first_true = mask.copy()
    first_true[0] = True
    unordered = months.copy()
    unordered[[20, 21]] = unordered[[21, 20]]
    duplicated = months.copy()
    duplicated[21] = duplicated[20]

    invalid_cases = (
        (values[:-1], months[:-1], mask[:-1], "at least 72 four-column rows"),
        (values[:, :3], months, mask, "at least 72 four-column rows"),
        (nonfinite, months, mask, "non-finite values"),
        (values, months[:-1], mask, "do not match feature rows"),
        (values, months.astype(np.float64), mask, "do not match feature rows"),
        (values, months, mask[:-1], "false-at-start row mask"),
        (values, months, mask.astype(np.int64), "false-at-start row mask"),
        (values, months, first_true, "false-at-start row mask"),
        (values, unordered, mask, "unique and increasing"),
        (values, duplicated, mask, "unique and increasing"),
    )
    for supplied_values, supplied_months, supplied_mask, message in invalid_cases:
        with pytest.raises(ModelError, match=message):
            rolling_candidate_input_fingerprint(
                supplied_values,
                supplied_months,
                supplied_mask,
                role="design",
            )
    with pytest.raises(ModelError, match="role must be design or production"):
        rolling_candidate_input_fingerprint(
            values,
            months,
            mask,
            role="wrong",  # type: ignore[arg-type]
        )


def test_macro_stability_runs_exactly_100_registered_attempts() -> None:
    values, months, mask = _macro(193)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        means=np.asarray([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )
    main_values = (values - values.mean(axis=0)) / values.std(axis=0)
    main_states = decode_viterbi(main_values, parameters, lengths=(193,))
    main = cast(GaussianHMMFit, _MacroMainFit(parameters, main_states))

    def fake_fit(
        features: HMMFeatureMatrix, n_states: int, seeds: object
    ) -> GaussianHMMFit:
        assert n_states == 2
        return cast(GaussianHMMFit, _MacroFoldFit(parameters, features))

    result = run_macro_stability(
        values,
        months,
        mask,
        role="design",
        main_fit=main,
        macro_block_length=4,
        master_seed=101,
        scientific_version=1,
        fit_function=fake_fit,
    )

    assert len(result.attempts) == 100
    assert result.summary.successful_attempts == 100
    assert all(attempt.successful for attempt in result.attempts)
    assert all(
        attempt.candidate_to_reference is not None
        and sorted(attempt.candidate_to_reference.tolist()) == [0, 1]
        and not attempt.candidate_to_reference.flags.writeable
        for attempt in result.attempts
    )


class _MacroMainFit(_MainRollingFit):
    scaler_scope = "design_training"


class _MacroFoldFit:
    def __init__(self, parameters: HMMParameters, features: HMMFeatureMatrix) -> None:
        self.parameters = parameters
        self.n_states = 2
        self.scaler_means = features.scaler_means
        self.scaler_standard_deviations = features.scaler_standard_deviations


def test_macro_stability_reduces_only_all_restart_numerical_failure() -> None:
    values, months, mask = _macro(193)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        means=np.asarray([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )
    standardized = (values - values.mean(axis=0)) / values.std(axis=0)
    states = decode_viterbi(standardized, parameters, lengths=(193,))
    main = cast(GaussianHMMFit, _MacroMainFit(parameters, states))

    with pytest.raises(ModelError, match="structural macro fault"):
        run_macro_stability(
            values,
            months,
            mask,
            role="design",
            main_fit=main,
            macro_block_length=4,
            master_seed=101,
            scientific_version=1,
            fit_function=lambda *_args: (_ for _ in ()).throw(
                ModelError("structural macro fault")
            ),
        )

    result = run_macro_stability(
        values,
        months,
        mask,
        role="design",
        main_fit=main,
        macro_block_length=4,
        master_seed=101,
        scientific_version=1,
        fit_function=lambda _features, n_states, seeds: (_ for _ in ()).throw(
            _all_restart_failure(n_states, seeds)
        ),
    )
    assert len(result.attempts) == 100
    assert result.summary.successful_attempts == 0
    assert all(
        item.failure_type == "NoNumericallyValidHMMRestarts" for item in result.attempts
    )


def test_sealed_holdout_conditions_first_row_and_resets_at_gap() -> None:
    rng = np.random.default_rng(55)
    design_rows = 80
    holdout_rows = 24
    design_values = rng.normal(size=(design_rows, 4))
    holdout_values = rng.normal(size=(holdout_rows, 4))
    design = _calibration_slice("design", design_values, gap=None)
    holdout = _calibration_slice("holdout", holdout_values, gap=17)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        means=np.asarray([[-0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )
    selected = cast(GaussianHMMFit, _SelectedHoldoutFit(parameters, design_rows))

    result = evaluate_sealed_holdout(selected, design, holdout)

    assert result.sequence_lengths == (17, 7)
    assert result.hmm_log_scores.shape == (24,)
    assert result.benchmark_scores.n_observations == 24
    assert isinstance(result.decision.passed, bool)


def _calibration_slice(
    role: Literal["design", "holdout"], macro: np.ndarray, *, gap: int | None
) -> CalibrationSlice:
    rows = len(macro)
    months = pd.period_range("2010-01", periods=rows, freq="M")
    sessions = pd.date_range("2010-01-04", periods=rows, freq="B")
    mask = np.ones(rows, dtype=np.bool_)
    mask[0] = False
    if gap is not None:
        mask[gap] = False
    return CalibrationSlice(
        role=role,
        monthly_returns=np.zeros((rows, 3)),
        daily_returns=np.zeros((rows, 3)),
        macro_features=macro,
        monthly_period_ordinals=months.asi8,
        daily_session_ns=sessions.asi8,
        monthly_continuation=mask,
        daily_continuation=mask.copy(),
    )


class _SelectedHoldoutFit:
    def __init__(self, parameters: HMMParameters, rows: int) -> None:
        self.parameters = parameters
        self.scaler_means = np.zeros(4)
        self.scaler_standard_deviations = np.ones(4)
        self.chain_diagnostics = _ChainStub()
        self.forward_backward = _ForwardStub(rows)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: prepare_hmm_feature_matrix(*_macro(), role="design", design_rows=40),
        lambda: prepare_hmm_feature_matrix(
            *_macro(), role="production", design_rows=30
        ),
        lambda: prepare_hmm_feature_matrix(*_macro(), role="unknown"),
    ],
)
def test_calibration_preparation_fails_closed(operation: object) -> None:
    with pytest.raises(ModelError):
        operation()  # type: ignore[operator]


def test_feature_preparation_rejects_every_invalid_input_geometry() -> None:
    values, months, mask = _macro()
    nonfinite = values.copy()
    nonfinite[2, 1] = np.nan
    unordered = months.copy()
    unordered[[10, 11]] = unordered[[11, 10]]
    duplicated = months.copy()
    duplicated[11] = duplicated[10]
    starts_true = mask.copy()
    starts_true[0] = True

    cases = (
        (np.ones(40), months, mask, "four-column"),
        (np.ones((40, 3)), months, mask, "four-column"),
        (
            np.empty((0, 4)),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=bool),
            "four-column",
        ),
        (nonfinite, months, mask, "non-finite"),
        (values, months[:-1], mask, "integer vector"),
        (values, months.astype(np.float64), mask, "integer vector"),
        (values, months, mask[:-1], "false-at-start"),
        (values, months, mask.astype(np.int64), "false-at-start"),
        (values, months, starts_true, "false-at-start"),
        (values, unordered, mask, "unique and increasing"),
        (values, duplicated, mask, "unique and increasing"),
    )
    for supplied_values, supplied_months, supplied_mask, message in cases:
        with pytest.raises(ModelError, match=message):
            prepare_hmm_feature_matrix(
                supplied_values,
                supplied_months,
                supplied_mask,
                role="production",
            )

    for design_rows in (True, 0, -1, 40, 41, 2.5):
        with pytest.raises(ModelError, match="proper positive prefix"):
            prepare_hmm_feature_matrix(
                values,
                months,
                mask,
                role="design",
                design_rows=design_rows,  # type: ignore[arg-type]
            )


def test_candidate_fit_contract_rejects_unregistered_coordinates() -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(values, months, mask, role="production")

    with pytest.raises(ModelError, match="candidate order"):
        fit_hmm_candidate_set(
            features,
            master_seed=5,
            scientific_version=1,
            scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
            replicate=0,
            candidates=(2, 3),
        )
    with pytest.raises(ModelError, match="closed ModelFitScope"):
        fit_hmm_candidate_set(
            features,
            master_seed=5,
            scientific_version=1,
            scope=1,  # type: ignore[arg-type]
            replicate=0,
        )
    for replicate in (True, -1, 1.5):
        with pytest.raises(ModelError, match="non-negative integer"):
            fit_hmm_candidate_set(
                features,
                master_seed=5,
                scientific_version=1,
                scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
                replicate=replicate,  # type: ignore[arg-type]
            )

    with pytest.raises(ModelError, match="wrong candidate state count"):
        fit_hmm_candidate_set(
            features,
            master_seed=5,
            scientific_version=1,
            scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
            replicate=0,
            fit_function=lambda _features, n_states, _seeds: cast(
                GaussianHMMFit, _FitStub(n_states + 1)
            ),
        )


def test_registered_candidate_executor_binds_workers_sources_and_fit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(
        values,
        months,
        mask,
        role="design",
        design_rows=30,
    )
    worker_calls: list[int] = []

    def fake_official_fit(
        supplied: HMMFeatureMatrix,
        n_states: int,
        _seeds: object,
        *,
        workers: int = 1,
    ) -> GaussianHMMFit:
        worker_calls.append(workers)
        result = _FitStub(n_states)
        result.n_observations = len(supplied.values)
        result.n_features = len(supplied.feature_names)
        result.lengths = supplied.lengths
        result.feature_names = supplied.feature_names
        result.scaler_scope = supplied.scaler_scope
        return cast(GaussianHMMFit, result)

    monkeypatch.setattr(
        calibration_module,
        "fit_gaussian_hmm_restarts",
        fake_official_fit,
    )
    monkeypatch.setattr(
        calibration_identity_module,
        "gaussian_hmm_fit_fingerprint",
        lambda fit: f"{fit.n_states:064x}",
    )

    result = execute_registered_hmm_candidate_set(
        features,
        master_seed=11,
        scientific_version=1,
        workers=2,
    )
    identity = registered_candidate_fit_set_identity(result)

    assert worker_calls == [2, 2, 2, 2]
    assert identity.scope is ModelFitScope.DESIGN_MAIN
    assert identity.replicate == 0
    assert len(identity.restart_seed_fingerprints) == 4
    assert is_registered_candidate_fit_set(result)
    assert not is_registered_candidate_fit_set(copy.copy(result))
    assert not is_registered_candidate_fit_set(object())

    fit = result.attempts[0].fit
    assert fit is not None
    original_features = fit.n_features
    fit.n_features = original_features + 1
    try:
        with pytest.raises(ModelError, match="fit output is inconsistent"):
            registered_candidate_fit_set_identity(result)
        assert not is_registered_candidate_fit_set(result)
    finally:
        fit.n_features = original_features
    assert is_registered_candidate_fit_set(result)
    original_master_seed = result._master_seed
    object.__setattr__(result, "_master_seed", None)
    try:
        with pytest.raises(ModelError, match="identity is incomplete"):
            registered_candidate_fit_set_identity(result)
    finally:
        object.__setattr__(result, "_master_seed", original_master_seed)

    original_scope = result.scope
    object.__setattr__(result, "scope", ModelFitScope.PRODUCTION_MAIN_AND_FOLDS)
    try:
        with pytest.raises(ModelError, match="geometry is invalid"):
            registered_candidate_fit_set_identity(result)
    finally:
        object.__setattr__(result, "scope", original_scope)

    original_attempts = result.attempts
    object.__setattr__(
        result,
        "attempts",
        (
            CandidateFitAttempt(2, None, None, None, None),
            *original_attempts[1:],
        ),
    )
    try:
        with pytest.raises(ModelError, match="lacks failure evidence"):
            registered_candidate_fit_set_identity(result)
    finally:
        object.__setattr__(result, "attempts", original_attempts)

    object.__setattr__(
        result,
        "attempts",
        (
            replace(original_attempts[0], failure_type="stale-success-metadata"),
            *original_attempts[1:],
        ),
    )
    try:
        with pytest.raises(ModelError, match="fit output is inconsistent"):
            registered_candidate_fit_set_identity(result)
    finally:
        object.__setattr__(result, "attempts", original_attempts)

    original_feature_fingerprint = result._feature_fingerprint
    object.__setattr__(result, "_feature_fingerprint", "0" * 64)
    try:
        with pytest.raises(ModelError, match="sources or outputs changed"):
            registered_candidate_fit_set_identity(result)
    finally:
        object.__setattr__(
            result,
            "_feature_fingerprint",
            original_feature_fingerprint,
        )
    assert is_registered_candidate_fit_set(result)


def test_registered_v11_candidate_executor_binds_prospective_k4_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(
        values,
        months,
        mask,
        role="design",
        design_rows=30,
    )
    policy_calls: list[object | None] = []

    def fake_official_fit(
        supplied: HMMFeatureMatrix,
        n_states: int,
        _seeds: object,
        *,
        workers: int = 1,
        owner_fixed_k4_policy: object | None = None,
    ) -> GaussianHMMFit:
        assert workers == 2
        policy_calls.append(owner_fixed_k4_policy)
        result = _FitStub(n_states)
        result.n_observations = len(supplied.values)
        result.n_features = len(supplied.feature_names)
        result.lengths = supplied.lengths
        result.feature_names = supplied.feature_names
        result.scaler_scope = supplied.scaler_scope
        return cast(GaussianHMMFit, result)

    monkeypatch.setattr(
        calibration_module,
        "fit_gaussian_hmm_restarts",
        fake_official_fit,
    )
    monkeypatch.setattr(
        calibration_identity_module,
        "gaussian_hmm_fit_fingerprint",
        lambda fit: f"{fit.n_states:064x}",
    )

    execute_registered_hmm_candidate_set(
        features,
        master_seed=11,
        scientific_version=11,
        workers=2,
    )

    assert policy_calls == [OWNER_FIXED_K4_PROSPECTIVE_POLICY] * 4


def test_registered_candidate_executor_rejects_invalid_worker_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(
        values,
        months,
        mask,
        role="design",
        design_rows=30,
    )
    monkeypatch.setattr(
        calibration_module,
        "fit_gaussian_hmm_restarts",
        lambda *_args, **_kwargs: pytest.fail("invalid workers reached the fitter"),
    )
    for workers in (True, 0, -1, 1.5):
        with pytest.raises(ModelError, match="workers must be a positive integer"):
            execute_registered_hmm_candidate_set(
                features,
                master_seed=11,
                scientific_version=1,
                workers=workers,  # type: ignore[arg-type]
            )


def test_candidate_fit_set_can_retain_an_all_candidate_scientific_nonpass() -> None:
    values, months, mask = _macro()
    features = prepare_hmm_feature_matrix(values, months, mask, role="production")
    result = fit_hmm_candidate_set(
        features,
        master_seed=5,
        scientific_version=1,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        fit_function=lambda _features, n_states, seeds: (_ for _ in ()).throw(
            _all_restart_failure(n_states, seeds)
        ),
    )

    assert isinstance(result, CandidateFitSet)
    assert result.valid_fits == ()
    assert not result.passed
    assert all(not attempt.valid for attempt in result.attempts)


def test_production_selected_k_rejects_wrong_k_scaler_and_fit_identity() -> None:
    values, months, mask = _macro()
    production = prepare_hmm_feature_matrix(values, months, mask, role="production")
    design = prepare_hmm_feature_matrix(
        values,
        months,
        mask,
        role="design",
        design_rows=30,
    )
    for selected in (True, 1, 6, 2.5):
        with pytest.raises(ModelError, match="must be one of"):
            fit_production_selected_k(
                production,
                selected_n_states=selected,  # type: ignore[arg-type]
                master_seed=5,
                scientific_version=1,
            )
    with pytest.raises(ModelError, match="production-full scaler"):
        fit_production_selected_k(
            design,
            selected_n_states=2,
            master_seed=5,
            scientific_version=1,
        )

    for changed_field, changed_value in (
        ("n_states", 3),
        ("n_observations", len(production.values) - 1),
        ("lengths", (len(production.values) - 1,)),
        ("scaler_scope", "design_training"),
    ):

        def inconsistent_fit(
            _features: HMMFeatureMatrix,
            n_states: int,
            _seeds: object,
            *,
            field: str = changed_field,
            value: object = changed_value,
        ) -> GaussianHMMFit:
            result = _FitStub(n_states)
            result.n_observations = len(production.values)
            result.lengths = production.lengths
            result.scaler_scope = "production_full"
            setattr(result, field, value)
            return cast(GaussianHMMFit, result)

        with pytest.raises(ModelError, match="inconsistent identity"):
            fit_production_selected_k(
                production,
                selected_n_states=2,
                master_seed=5,
                scientific_version=1,
                fit_function=inconsistent_fit,
            )


def test_decoded_return_pools_rejects_wrong_monthly_path_geometry() -> None:
    data = _ppw_slice()
    fit = _DecodedFitStub()
    fit.decoded_states = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(ModelError, match="does not match calibration slice"):
        decoded_return_pools(cast(GaussianHMMFit, fit), data)


def test_sealed_holdout_rejects_wrong_roles_scalers_and_nonfinite_values() -> None:
    rng = np.random.default_rng(551)
    design = _calibration_slice("design", rng.normal(size=(80, 4)), gap=None)
    holdout = _calibration_slice("holdout", rng.normal(size=(24, 4)), gap=None)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        means=np.asarray([[-0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]),
        tied_covariance=np.eye(4),
    )

    selected = _SelectedHoldoutFit(parameters, 80)
    with pytest.raises(ModelError, match="design and holdout slices"):
        evaluate_sealed_holdout(
            cast(GaussianHMMFit, selected),
            replace(design, role="full"),
            holdout,
        )
    with pytest.raises(ModelError, match="design and holdout slices"):
        evaluate_sealed_holdout(
            cast(GaussianHMMFit, selected),
            design,
            replace(holdout, role="design"),
        )

    for means, scales in (
        (np.zeros(3), np.ones(4)),
        (np.zeros(4), np.ones(3)),
        (np.zeros(4), np.asarray([1.0, 1.0, 0.0, 1.0])),
        (np.zeros(4), np.asarray([1.0, 1.0, -1.0, 1.0])),
    ):
        selected.scaler_means = means
        selected.scaler_standard_deviations = scales
        with pytest.raises(ModelError, match="design scaler is invalid"):
            evaluate_sealed_holdout(
                cast(GaussianHMMFit, selected),
                design,
                holdout,
            )

    selected.scaler_means = np.zeros(4)
    selected.scaler_standard_deviations = np.ones(4)
    nonfinite_macro = holdout.macro_features.copy()
    nonfinite_macro[0, 0] = np.inf
    with pytest.raises(ModelError, match="produced non-finite values"):
        evaluate_sealed_holdout(
            cast(GaussianHMMFit, selected),
            design,
            replace(holdout, macro_features=nonfinite_macro),
        )
