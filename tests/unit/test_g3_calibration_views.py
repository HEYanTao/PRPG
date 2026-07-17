from __future__ import annotations

import copy
from dataclasses import fields, replace
from functools import lru_cache
from typing import cast

import numpy as np
import pandas as pd
import pytest

import prpg.model.calibration as calibration_module
import prpg.model.g3_calibration_views as views_module
import prpg.model.production_fit_execution as production_execution_module
from prpg.errors import ModelError
from prpg.model.adequacy_execution import run_registered_latent_cell
from prpg.model.adequacy_gate import (
    assemble_four_cell_adequacy_holm_view,
    build_latent_adequacy_task_view,
    build_refitted_adequacy_task_view,
)
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    materialize_registered_block_uniforms,
    run_registered_materialized_block_calibration,
)
from prpg.model.block_length import BlockPolicySelection, select_supported_length
from prpg.model.calibration import (
    CandidateFitAttempt,
    CandidateFitSet,
    DecodedReturnPools,
    MacroBootstrapAttempt,
    MacroStabilityEvidence,
    RegisteredRollingBootstrapMaterialization,
    RollingCandidateEvidence,
    RollingFitAttempt,
    SealedHoldoutEvidence,
    evaluate_sealed_holdout,
    execute_registered_hmm_candidate_set,
    execute_registered_macro_stability,
    execute_registered_preliminary_calibrations,
    execute_registered_rolling_candidate,
    is_registered_macro_stability_evidence,
    materialize_registered_macro_bootstrap,
    materialize_registered_rolling_bootstrap,
    registered_preliminary_calibrations,
)
from prpg.model.candidate_gate import (
    CANONICAL_CANDIDATES,
    CandidateGateEvidence,
    CandidateGateResult,
    CandidateViabilityResult,
    execute_registered_candidate_viability,
    run_candidate_gate,
    run_candidate_selection,
    run_candidate_viability,
    verified_registered_candidate_viability_execution_identity,
)
from prpg.model.candidate_suite import (
    RegisteredDesignCandidateSuite,
    assemble_registered_design_candidate_suite,
    is_registered_design_candidate_suite,
    registered_design_candidate_gate_evidence,
    registered_design_candidate_suite_identity,
)
from prpg.model.g3_assembly import (
    G3TaskFailure,
    derive_g3_gate_decision,
    task_view_cross_binding_error,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    CandidateViabilityTaskView,
    MacroStabilityTaskView,
    ProductionFitSameKView,
    SealedHoldoutVetoView,
    assemble_candidate_gate_task_views,
    build_candidate_selection_task_view,
    build_candidate_viability_task_view,
    build_macro_stability_task_view,
    build_production_fit_same_k_view,
    build_sealed_holdout_veto_view,
    is_candidate_selection_task_view,
    is_candidate_viability_task_view,
    is_macro_stability_task_view,
    is_production_fit_same_k_view,
    is_registered_candidate_iteration_task_view,
    is_registered_candidate_viability_task_view,
    is_sealed_holdout_veto_view,
    production_fit_sources_from_view,
    registered_production_fit_execution_from_view,
    selected_design_fit_from_view,
)
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_science_handlers import CanonicalG3HandlerBundle
from prpg.model.g3_task_publication import (
    adequacy_cell_executor_source_slots,
    adequacy_holm_executor_source_slots,
    candidate_selection_executor_source_slots,
    candidate_viability_executor_source_slots,
    macro_stability_executor_source_slots,
    production_fit_executor_source_slots,
    scientific_component_slot_fingerprint,
    scientific_task_source_slots,
    sealed_holdout_executor_source_slots,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    NoNumericallyValidHMMRestarts,
    RestartDiagnostic,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
    forward_backward,
    gaussian_hmm_bic,
    gaussian_hmm_parameter_count,
    historical_transition_support_report,
    posterior_identification_report,
    summarize_state_path,
    tied_covariance_diagnostic,
    validate_numerical_chain,
)
from prpg.model.input import CalibrationSlice
from prpg.model.production_fit_execution import (
    RegisteredProductionFitSameK,
    execute_registered_production_fit_same_k,
)
from prpg.model.refit_adequacy_execution import (
    run_registered_refitted_adequacy_cell,
)
from prpg.model.regime_policy import owner_fixed_k4_disposition
from prpg.model.stability import MacroStabilitySummary, RollingStabilitySummary
from prpg.simulation.rng import CalibrationArtifact, ModelFitScope

_RUN = "a" * 64
_VIABILITY_BINDING = "b" * 64
_SELECTION_BINDING = "c" * 64
_HOLDOUT_BINDING = "d" * 64
_PRODUCTION_BINDING = "e" * 64
_DESIGN_MACRO_BINDING = "f" * 64
_PRODUCTION_MACRO_BINDING = "1" * 64
_MASTER_SEED = 20250308
_SCIENTIFIC_VERSION = 1


@lru_cache(maxsize=4)
def _registered_selection_bootstrap(
    master_seed: int = _MASTER_SEED,
    scientific_version: int = _SCIENTIFIC_VERSION,
    artifact: CalibrationArtifact = CalibrationArtifact.DESIGN_OR_CONTROL,
) -> RegisteredRollingBootstrapMaterialization:
    return materialize_registered_rolling_bootstrap(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
    )


_FEATURE_NAMES = (
    "industrial_production_growth",
    "inflation_yoy",
    "yield_curve_spread",
    "credit_spread",
)


def _fit(
    rows: int,
    *,
    scaler_scope: str = "design_training",
    lengths: tuple[int, ...] | None = None,
    scaler_means: np.ndarray | None = None,
    scaler_scales: np.ndarray | None = None,
    values: np.ndarray | None = None,
    parameters: HMMParameters | None = None,
) -> GaussianHMMFit:
    geometry = lengths or (rows,)
    parameters = parameters or HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.90, 0.10], [0.10, 0.90]]),
        means=np.asarray([[-3.0, -0.2, 0.1, -0.1], [3.0, 0.2, -0.1, 0.1]]),
        tied_covariance=np.eye(4) * 0.25,
    )
    n_states = len(parameters.start_probabilities)
    source_states = np.resize(np.repeat(np.arange(n_states), 10), rows)
    model_values = (
        parameters.means[source_states]
        + np.random.default_rng(rows).normal(0.0, 0.05, size=(rows, 4))
        if values is None
        else np.asarray(values, dtype=np.float64)
    )
    decoded = decode_viterbi(model_values, parameters, lengths=geometry)
    path = summarize_state_path(decoded, n_states, lengths=geometry)
    posterior = forward_backward(model_values, parameters, lengths=geometry)
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
    disposition = owner_fixed_k4_disposition(
        n_states=n_states,
        identification=identification,
        transition=support,
    )
    assert identification.passed or disposition is not None
    assert support.passed or disposition is not None
    means = np.zeros(4) if scaler_means is None else scaler_means
    scales = np.ones(4) if scaler_scales is None else scaler_scales
    return GaussianHMMFit(
        parameters=parameters,
        decoded_states=decoded,
        canonical_to_original=np.arange(n_states, dtype=np.int64),
        original_to_canonical=np.arange(n_states, dtype=np.int64),
        log_likelihood=posterior.total_log_likelihood,
        bic=gaussian_hmm_bic(posterior.total_log_likelihood, rows, n_states, 4),
        parameter_count=gaussian_hmm_parameter_count(n_states, 4),
        n_observations=rows,
        n_features=4,
        n_states=n_states,
        lengths=geometry,
        best_restart_index=0,
        best_seed=1,
        restart_diagnostics=(),
        covariance_diagnostic=tied_covariance_diagnostic(parameters.tied_covariance),
        state_path_diagnostics=path,
        chain_diagnostics=validate_numerical_chain(parameters.transition_matrix),
        forward_backward=posterior,
        identification=identification,
        historical_transition_support=support,
        scaler_scope=scaler_scope,  # type: ignore[arg-type]
        scaler_means=means,
        scaler_standard_deviations=scales,
        feature_names=_FEATURE_NAMES,
        owner_fixed_k4_disposition=disposition,
    )


def _candidate_stages() -> tuple[CandidateViabilityResult, CandidateGateResult]:
    fit = _fit(193)
    monthly = np.asarray(fit.decoded_states, dtype=np.int64)
    daily = np.resize(np.repeat(np.asarray([0, 1]), 100), 4_049).astype(np.int64)
    pools = DecodedReturnPools(
        monthly,
        daily,
        summarize_state_path(monthly, 2, lengths=(193,)),
        summarize_state_path(daily, 2, lengths=(4_049,)),
    )
    rolling = RollingCandidateEvidence(
        2,
        "design",
        tuple(
            RollingFitAttempt(
                fold,
                121 + 12 * fold,
                12,
                (12,),
                fit,
                (1.0,) * 12,
                0.80,
                np.full(2, 0.80),
                None,
                None,
            )
            for fold in range(6)
        ),
        RollingStabilitySummary(6, 0.80, 0.80, np.full(2, 0.80), True),
    )
    macro = MacroStabilityEvidence(
        2,
        "design",
        4,
        tuple(
            MacroBootstrapAttempt(
                attempt,
                np.arange(193, dtype=np.int64),
                fit,
                0.90,
                np.full(2, 0.80),
                None,
                None,
                np.asarray([0, 1]),
            )
            for attempt in range(100)
        ),
        MacroStabilitySummary(
            100,
            0.90,
            0.90,
            np.full(2, 0.80),
            np.full(2, 0.80),
            True,
        ),
    )
    monthly_policy = _policy(monthly, "monthly", 6)
    daily_policy = _policy(daily, "daily", 60)
    attempts = [CandidateFitAttempt(2, fit, None, None, None)]
    evidence = [
        CandidateGateEvidence(
            2,
            pools,
            rolling,
            macro,
            monthly_policy,
            daily_policy,
        )
    ]
    for n_states in (3, 4, 5):
        attempts.append(
            CandidateFitAttempt(
                n_states,
                None,
                "ModelError",
                f"no valid K={n_states} restart",
                {"valid_restarts": 0},
            )
        )
        evidence.append(CandidateGateEvidence(n_states, None, None, None, None, None))
    viability = run_candidate_viability(
        CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, tuple(attempts)),
        tuple(evidence),
        role="design",
    )
    return viability, run_candidate_selection(
        viability,
        _registered_selection_bootstrap(),
    )


def _candidate_result() -> CandidateGateResult:
    return _candidate_stages()[1]


def _policy(
    states: np.ndarray,
    frequency: str,
    length: int,
) -> BlockPolicySelection:
    return select_supported_length(
        starting_length=length,
        frequency=frequency,  # type: ignore[arg-type]
        states=states,
        continuation=np.ones(len(states) - 1, dtype=np.bool_),
        realized_means_by_length={length: {0: length / 2, 1: length / 2}},
        selected_states=(0, 1),
    )


def _candidate_views() -> tuple[
    CandidateGateResult,
    CandidateViabilityTaskView,
    CandidateSelectionTaskView,
]:
    viability_result, result = _candidate_stages()
    viability = build_candidate_viability_task_view(
        viability_result,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
    )
    selection = build_candidate_selection_task_view(
        viability,
        result,
        run_id=_RUN,
        binding_fingerprint=_SELECTION_BINDING,
    )
    return result, viability, selection


def _slices() -> tuple[CalibrationSlice, CalibrationSlice, CalibrationSlice]:
    rng = np.random.default_rng(55)
    design_macro = rng.normal(size=(193, 4))
    holdout_macro = rng.normal(size=(24, 4))

    def make(role: str, macro: np.ndarray, *, gap: int | None) -> CalibrationSlice:
        rows = len(macro)
        continuation = np.ones(rows, dtype=np.bool_)
        continuation[0] = False
        if gap is not None:
            continuation[gap] = False
        return CalibrationSlice(
            role=role,  # type: ignore[arg-type]
            monthly_returns=rng.normal(0.0, 0.01, size=(rows, 3)),
            daily_returns=rng.normal(0.0, 0.001, size=(rows * 2, 3)),
            macro_features=macro,
            monthly_period_ordinals=np.arange(600, 600 + rows, dtype=np.int64),
            daily_session_ns=np.arange(rows * 2, dtype=np.int64),
            monthly_continuation=continuation,
            daily_continuation=np.r_[
                False,
                np.ones(rows * 2 - 1, dtype=np.bool_),
            ],
        )

    design = make("design", design_macro, gap=None)
    holdout = make("holdout", holdout_macro, gap=17)
    full_macro = np.vstack((design_macro, holdout_macro))
    full = make("full", full_macro, gap=210)
    return design, holdout, full


def _registered_iteration_suite(
    monkeypatch: pytest.MonkeyPatch,
    *,
    viable: bool = True,
    fixed_k4: bool = False,
    hmm_workers: int = 1,
    worker_calls: list[int] | None = None,
) -> RegisteredDesignCandidateSuite:
    """Build a reduced-geometry registered iteration without expensive fits."""

    rng = np.random.default_rng(919)
    periods = pd.period_range("2000-01", periods=193, freq="M")
    fixed_states = np.resize(np.repeat(np.arange(4), 24), len(periods))
    fixed_centers = np.asarray(
        [
            [-3.0, -0.6, 0.3, -0.2],
            [-1.0, 0.6, -0.3, 0.2],
            [1.0, -0.6, -0.3, 0.2],
            [3.0, 0.6, 0.3, -0.2],
        ]
    )
    macro = (
        fixed_centers[fixed_states] + rng.normal(0.0, 0.03, size=(len(periods), 4))
        if fixed_k4
        else rng.normal(size=(len(periods), 4))
    )
    daily_dates = pd.DatetimeIndex(
        [
            period.start_time + pd.Timedelta(days=day)
            for period in periods
            for day in range(21)
        ]
    )
    monthly_continuation = np.ones(len(periods), dtype=np.bool_)
    monthly_continuation[0] = False
    daily_continuation = np.ones(len(daily_dates), dtype=np.bool_)
    daily_continuation[0] = False
    design = CalibrationSlice(
        role="design",
        monthly_returns=rng.normal(size=(len(periods), 3)),
        daily_returns=rng.normal(size=(len(daily_dates), 3)),
        macro_features=macro,
        monthly_period_ordinals=periods.asi8.astype(np.int64),
        daily_session_ns=daily_dates.asi8.astype(np.int64),
        monthly_continuation=monthly_continuation,
        daily_continuation=daily_continuation,
    )
    means = macro.mean(axis=0)
    scales = macro.std(axis=0, ddof=0)
    feature_values = (macro - means) / scales
    features = HMMFeatureMatrix(
        feature_values,
        (len(periods),),
        _FEATURE_NAMES,
        tuple(str(value) for value in periods),
        means,
        scales,
        "design_training",
    )

    fixed_parameters: HMMParameters | None = None

    def fast_official_fit(
        supplied: HMMFeatureMatrix,
        n_states: int,
        seeds: object,
        *,
        workers: int = 1,
    ) -> GaussianHMMFit:
        nonlocal fixed_parameters
        if worker_calls is not None:
            worker_calls.append(workers)
        checked_seeds = tuple(cast(tuple[int, ...], seeds))
        successful_n_states = 4 if fixed_k4 else 2
        if n_states != successful_n_states:
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
        if (
            fixed_k4
            and not viable
            and len(supplied.values) < len(fixed_states)
            and len(supplied.values) != 121
        ):
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
                        failure_message="synthetic rolling failure",
                    )
                    for index, seed in enumerate(checked_seeds)
                ),
            )
        if fixed_k4 and fixed_parameters is None:
            if len(supplied.values) != len(fixed_states):
                raise AssertionError(
                    "K4 fixture parameters require the full design fit"
                )
            transition = np.full((4, 4), 0.02)
            np.fill_diagonal(transition, 0.94)
            fixed_parameters = HMMParameters(
                start_probabilities=np.full(4, 0.25),
                transition_matrix=transition,
                means=np.vstack(
                    [
                        supplied.values[fixed_states == state].mean(axis=0)
                        for state in range(4)
                    ]
                ),
                tied_covariance=np.eye(4) * 0.05,
            )
        parameters = (
            fixed_parameters
            if fixed_k4
            else (
                HMMParameters(
                    start_probabilities=np.asarray([0.5, 0.5]),
                    transition_matrix=np.asarray([[0.95, 0.05], [0.05, 0.95]]),
                    means=np.asarray([[-1.6, 0.0, 0.0, 0.0], [1.6, 0.0, 0.0, 0.0]]),
                    tied_covariance=np.eye(4) * 0.5,
                )
                if viable
                else None
            )
        )
        return _fit(
            len(supplied.values),
            lengths=supplied.lengths,
            scaler_means=supplied.scaler_means,
            scaler_scales=supplied.scaler_standard_deviations,
            values=supplied.values,
            parameters=parameters,
        )

    monkeypatch.setattr(
        calibration_module,
        "fit_gaussian_hmm_restarts",
        fast_official_fit,
    )
    fit_set = execute_registered_hmm_candidate_set(
        features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        workers=hmm_workers,
    )
    fitted = fit_set.attempts[2 if fixed_k4 else 0].fit
    assert isinstance(fitted, GaussianHMMFit)
    preliminary_outcome = execute_registered_preliminary_calibrations(
        design,
        role="design",
    )
    preliminary = registered_preliminary_calibrations(preliminary_outcome)
    macro_materialization = materialize_registered_macro_bootstrap(
        design.macro_features,
        design.monthly_period_ordinals,
        design.monthly_continuation,
        role="design",
        macro_block_length=preliminary.macro.combined.starting_length,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    rolling = execute_registered_rolling_candidate(
        design.macro_features,
        design.monthly_period_ordinals,
        design.monthly_continuation,
        role="design",
        main_fit=fitted,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        workers=hmm_workers,
    )
    macro_evidence = execute_registered_macro_stability(
        design.macro_features,
        design.monthly_period_ordinals,
        design.monthly_continuation,
        role="design",
        main_fit=fitted,
        macro_block_length=preliminary.macro.combined.starting_length,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        source_materialization=macro_materialization,
        workers=hmm_workers,
    )
    pools = calibration_module.decoded_return_pools(fitted, design)
    monthly_materialization = materialize_registered_block_uniforms(
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        artifact="design",
        frequency="monthly",
        history_count=8,
        strict_canonical=False,
        monthly_segment_lengths=(len(design.monthly_returns),),
    )
    daily_materialization = materialize_registered_block_uniforms(
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        artifact="design",
        frequency="daily",
        history_count=8,
        strict_canonical=False,
        monthly_segment_lengths=(len(design.monthly_returns),),
        daily_segment_lengths=(len(design.daily_returns),),
    )
    monthly_block = run_registered_materialized_block_calibration(
        monthly_materialization,
        model=fitted,
        starting_length=preliminary.monthly.combined.starting_length,
        historical_source_states=pools.monthly_states,
        source_continuation_links=design.monthly_continuation[1:],
    )
    daily_block = run_registered_materialized_block_calibration(
        daily_materialization,
        model=fitted,
        starting_length=preliminary.daily.combined.starting_length,
        historical_source_states=pools.daily_states,
        source_continuation_links=design.daily_continuation[1:],
    )
    return assemble_registered_design_candidate_suite(
        design,
        features,
        fit_set,
        preliminary_outcome=preliminary_outcome,
        macro_materialization=macro_materialization,
        monthly_materialization=monthly_materialization,
        daily_materialization=daily_materialization,
        rolling_by_k=(None, None, rolling, None)
        if fixed_k4
        else (rolling, None, None, None),
        macro_by_k=(None, None, macro_evidence, None)
        if fixed_k4
        else (macro_evidence, None, None, None),
        monthly_blocks_by_k=(None, None, monthly_block, None)
        if fixed_k4
        else (monthly_block, None, None, None),
        daily_blocks_by_k=(None, None, daily_block, None)
        if fixed_k4
        else (daily_block, None, None, None),
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )


def _next_registered_iteration_suite(
    previous: RegisteredDesignCandidateSuite,
) -> RegisteredDesignCandidateSuite:
    """Re-run every successful K at the prior finalized selected-K lengths."""

    selected_row = next(row for row in previous.rows if row.monthly_block is not None)
    assert selected_row.monthly_block is not None
    assert selected_row.daily_block is not None
    monthly_start = selected_row.monthly_block.execution.selected_length
    daily_start = selected_row.daily_block.execution.selected_length
    monthly_blocks: list[RegisteredBlockCalibration | None] = []
    daily_blocks: list[RegisteredBlockCalibration | None] = []
    for attempt, row in zip(previous._fit_set.attempts, previous.rows, strict=True):
        if attempt.fit is None:
            monthly_blocks.append(None)
            daily_blocks.append(None)
            continue
        assert row.decoded_return_pools is not None
        monthly_blocks.append(
            run_registered_materialized_block_calibration(
                previous._monthly_materialization,
                model=attempt.fit,
                starting_length=monthly_start,
                historical_source_states=row.decoded_return_pools.monthly_states,
                source_continuation_links=(
                    previous._design_slice.monthly_continuation[1:]
                ),
            )
        )
        daily_blocks.append(
            run_registered_materialized_block_calibration(
                previous._daily_materialization,
                model=attempt.fit,
                starting_length=daily_start,
                historical_source_states=row.decoded_return_pools.daily_states,
                source_continuation_links=(
                    previous._design_slice.daily_continuation[1:]
                ),
            )
        )
    return assemble_registered_design_candidate_suite(
        previous._design_slice,
        previous._design_features,
        previous._fit_set,
        preliminary_outcome=previous._preliminary_outcome,
        macro_materialization=previous._macro_materialization,
        monthly_materialization=previous._monthly_materialization,
        daily_materialization=previous._daily_materialization,
        rolling_by_k=tuple(row.rolling for row in previous.rows),
        macro_by_k=tuple(row.macro_stability for row in previous.rows),
        monthly_blocks_by_k=tuple(monthly_blocks),
        daily_blocks_by_k=tuple(daily_blocks),
        master_seed=previous.master_seed,
        scientific_version=previous.scientific_version,
        iteration=previous.iteration + 1,
        monthly_input_start=monthly_start,
        daily_input_start=daily_start,
    )


def test_registered_candidate_iteration_suite_is_exact_and_report_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _registered_iteration_suite(monkeypatch)
    identity = registered_design_candidate_suite_identity(suite)

    assert is_registered_design_candidate_suite(suite)
    assert identity.suite_fingerprint == suite.suite_fingerprint
    assert not identity.strict_canonical_geometry
    assert identity.rolling_fit_rng_execution_fingerprints[0] is not None
    assert identity.rolling_fit_rng_execution_fingerprints[1:] == (None, None, None)
    assert identity.macro_fit_rng_execution_fingerprints[0] is not None
    assert identity.block_core_fingerprints[0][0] is not None
    assert identity.block_core_fingerprints[0][1] is not None
    assert all(
        value is None
        for row in suite.rows[1:]
        for value in (
            row.decoded_return_pools,
            row.rolling,
            row.macro_stability,
            row.monthly_block,
            row.daily_block,
        )
    )
    assert suite.rows[0].macro_stability is not None
    assert (
        suite.rows[0].macro_stability._source_materialization
        is suite._macro_materialization
    )
    assert suite.rows[0].monthly_block is not None
    assert suite.rows[0].daily_block is not None
    assert (
        suite.rows[0].monthly_block._source_materialization
        is suite._monthly_materialization
    )
    assert (
        suite.rows[0].daily_block._source_materialization
        is suite._daily_materialization
    )

    viability = execute_registered_candidate_viability(suite)
    viability_identity = verified_registered_candidate_viability_execution_identity(
        viability
    )
    assert viability_identity.suite_fingerprint == suite.suite_fingerprint
    assert viability_identity.suite_combined_fit_rng_execution_fingerprint == (
        identity.combined_fit_rng_execution_fingerprint
    )
    report_view = build_candidate_viability_task_view(
        viability,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
    )
    assert is_candidate_viability_task_view(report_view)
    assert not is_registered_candidate_iteration_task_view(report_view)
    assert not is_registered_candidate_viability_task_view(report_view)
    with pytest.raises(ModelError, match="strict block geometry"):
        build_candidate_viability_task_view(
            viability,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
            run_id=_RUN,
            binding_fingerprint=_VIABILITY_BINDING,
            require_canonical_suite=True,
        )

    raw = run_candidate_viability(
        viability.fit_set,
        registered_design_candidate_gate_evidence(suite),
        role="design",
    )
    with pytest.raises(ModelError, match="registered design suite"):
        verified_registered_candidate_viability_execution_identity(raw)

    assert not is_registered_design_candidate_suite(copy.copy(suite))
    original_fingerprint = suite.suite_fingerprint
    object.__setattr__(suite, "suite_fingerprint", "0" * 64)
    try:
        assert not is_registered_design_candidate_suite(suite)
    finally:
        object.__setattr__(suite, "suite_fingerprint", original_fingerprint)
    assert is_registered_design_candidate_suite(suite)

    original_fit_set = suite._fit_set
    object.__setattr__(suite, "_fit_set", copy.copy(original_fit_set))
    try:
        assert not is_registered_design_candidate_suite(suite)
    finally:
        object.__setattr__(suite, "_fit_set", original_fit_set)
    assert is_registered_design_candidate_suite(suite)


def test_registered_design_hmm_families_propagate_nine_restart_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_calls: list[int] = []
    _registered_iteration_suite(
        monkeypatch,
        hmm_workers=9,
        worker_calls=worker_calls,
    )

    # Four main candidates, six rolling folds for the sole successful K, and
    # one hundred macro-stability attempts share the same local pool width.
    assert worker_calls == [9] * 110


def _holdout_evidence(
    selection: CandidateSelectionTaskView,
    design: CalibrationSlice,
    holdout: CalibrationSlice,
) -> SealedHoldoutEvidence:
    fit = selection._source_result.selected_fit
    assert isinstance(fit, GaussianHMMFit)
    return evaluate_sealed_holdout(fit, design, holdout)


def _production_features() -> HMMFeatureMatrix:
    values = np.random.default_rng(77).normal(size=(217, 4))
    return HMMFeatureMatrix(
        values,
        (210, 7),
        _FEATURE_NAMES,
        tuple(f"2000-{row:03d}" for row in range(217)),
        np.zeros(4),
        np.ones(4),
        "production_full",
    )


def _fake_production_fit(
    features: HMMFeatureMatrix,
    *,
    selected_n_states: int,
    master_seed: int,
    scientific_version: int,
) -> GaussianHMMFit:
    assert selected_n_states == 2
    assert master_seed == _MASTER_SEED
    assert scientific_version == _SCIENTIFIC_VERSION
    return _fit(
        len(features.values),
        scaler_scope="production_full",
        lengths=features.lengths,
        scaler_means=features.scaler_means,
        scaler_scales=features.scaler_standard_deviations,
        values=features.values,
    )


def _registered_fake_production_fit(
    features: HMMFeatureMatrix,
    *,
    selected_n_states: int,
    master_seed: int,
    scientific_version: int,
) -> GaussianHMMFit:
    base = _fake_production_fit(
        features,
        selected_n_states=selected_n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=selected_n_states,
    )
    diagnostics = tuple(
        RestartDiagnostic(
            restart_index=index,
            seed=seed,
            converged=index == 0,
            iterations=12 if index == 0 else 1,
            log_likelihood=base.log_likelihood if index == 0 else None,
            valid=index == 0,
            raw_minimum_eigenvalue=0.25 if index == 0 else None,
            maximum_floor_relative_change=0.0 if index == 0 else None,
            covariance_condition_number=1.0 if index == 0 else None,
            maximum_likelihood_decrease=0.0 if index == 0 else None,
            preliminary_chain_passed=index == 0,
            failure_type=None if index == 0 else "NonConvergence",
            failure_message=None if index == 0 else "synthetic non-convergence",
        )
        for index, seed in enumerate(seeds)
    )
    return replace(
        base,
        best_restart_index=0,
        best_seed=seeds[0],
        restart_diagnostics=diagnostics,
    )


def _registered_production_execution(
    monkeypatch: pytest.MonkeyPatch,
    selection: CandidateSelectionTaskView,
    features: HMMFeatureMatrix,
    *,
    fitter: object = _registered_fake_production_fit,
) -> RegisteredProductionFitSameK:
    # These focused fixtures predate the final fixed-point wrapper.  Preserve
    # the production guard while standing in only for that wrapper authority.
    monkeypatch.setattr(
        views_module,
        "is_registered_candidate_selection_task_view",
        views_module.is_candidate_selection_task_view,
    )
    monkeypatch.setattr(
        production_execution_module,
        "fit_production_selected_k",
        fitter,
    )
    result = execute_registered_production_fit_same_k(
        selection,
        features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    assert isinstance(result, RegisteredProductionFitSameK)
    return result


def _production_view(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CandidateSelectionTaskView, HMMFeatureMatrix, ProductionFitSameKView]:
    _, _, selection = _candidate_views()
    features = _production_features()
    execution = _registered_production_execution(monkeypatch, selection, features)
    view = build_production_fit_same_k_view(
        execution,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_BINDING,
    )
    return selection, features, view


def _registered_macro(
    fit: GaussianHMMFit,
    source: CalibrationSlice,
    *,
    role: str,
) -> MacroStabilityEvidence:
    def fast_fit(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: object,
    ) -> GaussianHMMFit:
        del seeds
        assert n_states == 2
        return _fit(
            len(features.values),
            scaler_scope=features.scaler_scope,
            lengths=features.lengths,
            scaler_means=features.scaler_means,
            scaler_scales=features.scaler_standard_deviations,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            calibration_module,
            "fit_gaussian_hmm_restarts",
            fast_fit,
        )
        return execute_registered_macro_stability(
            source.macro_features,
            source.monthly_period_ordinals,
            source.monthly_continuation,
            role=role,  # type: ignore[arg-type]
            main_fit=fit,
            macro_block_length=4,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )


def test_candidate_views_are_staged_task_scoped_and_private() -> None:
    viability_result, result = _candidate_stages()
    assert not hasattr(viability_result, "selection")
    assert not hasattr(viability_result, "_registered_bootstrap_indices")
    assert result.viability_result is viability_result
    viability = build_candidate_viability_task_view(
        viability_result,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
    )
    assert is_candidate_viability_task_view(viability)
    assert viability.admissible_n_states == (2,)
    # Cross-K admissibility remains visible as report evidence, but only K=4
    # can pass the prospective owner-fixed design-viability task.
    assert not viability.passed
    assert viability.rng_namespace == "design_candidate_fit_restarts"
    assert len(viability.restart_seed_fingerprints) == len(CANONICAL_CANDIDATES)

    public_fields = {
        item.name for item in fields(viability) if not item.name.startswith("_")
    }
    assert "result" not in public_fields
    assert "selection" not in public_fields
    assert "selected_n_states" not in public_fields
    assert all("selected" not in name for name in public_fields)

    selection = build_candidate_selection_task_view(
        viability,
        result,
        run_id=_RUN,
        binding_fingerprint=_SELECTION_BINDING,
    )
    assert is_candidate_selection_task_view(selection)
    assert selection.selected_n_states == 2
    assert selection.passed
    materialization = _registered_selection_bootstrap()
    assert selection.bootstrap_fingerprint == (
        materialization.materialization_fingerprint
    )
    assert selection.bootstrap_materialization_fingerprint == (
        materialization.materialization_fingerprint
    )
    assert selection.bootstrap_rng_execution_fingerprint == (
        materialization.rng_execution_fingerprint
    )
    grouped = assemble_candidate_gate_task_views(viability, selection)
    assert grouped.design_candidate_viability is viability
    assert grouped.design_predictive_selection is selection

    viability_fingerprint = viability.view_fingerprint
    object.__setattr__(viability, "view_fingerprint", "0" * 64)
    assert not is_candidate_viability_task_view(viability)
    object.__setattr__(viability, "view_fingerprint", viability_fingerprint)
    selection_fingerprint = selection.view_fingerprint
    object.__setattr__(selection, "view_fingerprint", "0" * 64)
    assert not is_candidate_selection_task_view(selection)
    object.__setattr__(selection, "view_fingerprint", selection_fingerprint)


def test_one_shot_raw_candidate_gate_remains_report_only() -> None:
    staged_viability, _ = _candidate_stages()
    compatibility = run_candidate_gate(
        staged_viability.fit_set,
        staged_viability._source_evidence,
        role="design",
        registered_bootstrap_indices=_registered_selection_bootstrap().indices,
    )
    viability = build_candidate_viability_task_view(
        compatibility.viability_result,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
    )
    with pytest.raises(ModelError, match="not executed as a staged task"):
        build_candidate_selection_task_view(
            viability,
            compatibility,
            run_id=_RUN,
            binding_fingerprint=_SELECTION_BINDING,
        )


def test_selection_requires_actual_viability_parent_and_distinct_binding() -> None:
    result, viability, _ = _candidate_views()
    copied = copy.copy(viability)
    assert not is_candidate_viability_task_view(copied)
    with pytest.raises(ModelError, match="builder authority"):
        build_candidate_selection_task_view(
            copied,
            result,
            run_id=_RUN,
            binding_fingerprint=_SELECTION_BINDING,
        )
    with pytest.raises(ModelError, match="differ"):
        build_candidate_selection_task_view(
            viability,
            result,
            run_id=_RUN,
            binding_fingerprint=_VIABILITY_BINDING,
        )
    with pytest.raises(ModelError, match="another run"):
        build_candidate_selection_task_view(
            viability,
            result,
            run_id="9" * 64,
            binding_fingerprint=_SELECTION_BINDING,
        )
    _, unrelated_result = _candidate_stages()
    with pytest.raises(ModelError, match="does not descend"):
        build_candidate_selection_task_view(
            viability,
            unrelated_result,
            run_id=_RUN,
            binding_fingerprint=_SELECTION_BINDING,
        )


@pytest.mark.parametrize(
    ("master_seed", "scientific_version"),
    ((_MASTER_SEED + 1, _SCIENTIFIC_VERSION), (_MASTER_SEED, 2)),
)
def test_selection_rejects_bootstrap_seed_or_version_outside_parent_identity(
    master_seed: int,
    scientific_version: int,
) -> None:
    viability_result, _ = _candidate_stages()
    viability = build_candidate_viability_task_view(
        viability_result,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
    )
    result = run_candidate_selection(
        viability_result,
        _registered_selection_bootstrap(master_seed, scientific_version),
    )
    with pytest.raises(ModelError, match="bootstrap identity differs"):
        build_candidate_selection_task_view(
            viability,
            result,
            run_id=_RUN,
            binding_fingerprint=_SELECTION_BINDING,
        )


def test_candidate_views_reject_copy_hand_construction_and_source_mutation() -> None:
    result, viability, selection = _candidate_views()
    assert not is_candidate_viability_task_view(copy.copy(viability))
    assert not is_candidate_selection_task_view(copy.copy(selection))
    assert selected_design_fit_from_view(selection) is result.selected_fit
    with pytest.raises(ModelError, match="lacks builder authority"):
        selected_design_fit_from_view(copy.copy(selection))
    with pytest.raises(TypeError, match="created only"):
        CandidateViabilityTaskView()
    with pytest.raises(TypeError, match="created only"):
        CandidateSelectionTaskView()

    materialization = result._registered_bootstrap_materialization
    assert materialization is not None
    original = materialization.materialization_fingerprint
    object.__setattr__(materialization, "materialization_fingerprint", "1" * 64)
    # The viability task is deliberately independent of the later common
    # rolling-bootstrap materialization; only selection is invalidated.
    assert is_candidate_viability_task_view(viability)
    assert not is_candidate_selection_task_view(selection)
    object.__setattr__(materialization, "materialization_fingerprint", original)
    assert is_candidate_selection_task_view(selection)


def test_viability_rejects_cross_role_and_wrong_rng_identity() -> None:
    result, _ = _candidate_stages()
    object.__setattr__(result, "_source_role", "production")
    with pytest.raises(ModelError, match="changed|design-role"):
        build_candidate_viability_task_view(
            result,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
            run_id=_RUN,
            binding_fingerprint=_VIABILITY_BINDING,
        )

    valid, _ = _candidate_stages()
    view = build_candidate_viability_task_view(
        valid,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
    )
    object.__setattr__(view, "master_seed", _MASTER_SEED + 1)
    assert not is_candidate_viability_task_view(view)


def test_holdout_view_executes_and_binds_model_both_slices_and_parent() -> None:
    _, _, selection = _candidate_views()
    design, holdout, _ = _slices()
    view = build_sealed_holdout_veto_view(
        selection,
        _holdout_evidence(selection, design, holdout),
        design=design,
        holdout=holdout,
        run_id=_RUN,
        binding_fingerprint=_HOLDOUT_BINDING,
    )
    assert is_sealed_holdout_veto_view(view)
    assert view.selected_n_states == 2
    assert view.evidence.sequence_lengths == (17, 7)
    assert not is_sealed_holdout_veto_view(copy.copy(view))
    with pytest.raises(TypeError, match="created only"):
        SealedHoldoutVetoView()

    fingerprint = view.view_fingerprint
    object.__setattr__(view, "view_fingerprint", "0" * 64)
    assert not is_sealed_holdout_veto_view(view)
    object.__setattr__(view, "view_fingerprint", fingerprint)
    design.macro_features.setflags(write=True)
    design.macro_features[0, 0] += 1.0
    assert not is_sealed_holdout_veto_view(view)


def test_holdout_rejects_cross_run_parent_task_and_slice_roles() -> None:
    _, viability, selection = _candidate_views()
    design, holdout, _ = _slices()
    evidence = _holdout_evidence(selection, design, holdout)
    with pytest.raises(ModelError, match="another run"):
        build_sealed_holdout_veto_view(
            selection,
            evidence,
            design=design,
            holdout=holdout,
            run_id="9" * 64,
            binding_fingerprint=_HOLDOUT_BINDING,
        )
    with pytest.raises((ModelError, TypeError), match="selection|wrong type"):
        build_sealed_holdout_veto_view(
            viability,  # type: ignore[arg-type]
            evidence,
            design=design,
            holdout=holdout,
            run_id=_RUN,
            binding_fingerprint=_HOLDOUT_BINDING,
        )
    with pytest.raises(ModelError, match="design and holdout"):
        build_sealed_holdout_veto_view(
            selection,
            evidence,
            design=replace(design, role="full"),
            holdout=holdout,
            run_id=_RUN,
            binding_fingerprint=_HOLDOUT_BINDING,
        )

    with pytest.raises(ModelError, match="differs from independent reevaluation"):
        build_sealed_holdout_veto_view(
            selection,
            replace(evidence, sequence_lengths=(24,)),
            design=design,
            holdout=holdout,
            run_id=_RUN,
            binding_fingerprint=_HOLDOUT_BINDING,
        )


def test_production_fit_view_calls_fixed_k_executor_and_revalidates_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, features, view = _production_view(monkeypatch)
    assert is_production_fit_same_k_view(view)
    assert view.design_selected_k == view.production_fit.n_states == 2
    assert view.rng_namespace == "production_fit_restarts"
    execution = registered_production_fit_execution_from_view(view)
    assert execution.production_fit is view.production_fit
    assert view.registered_execution_fingerprint == execution.execution_fingerprint
    retained_fit, retained_features = production_fit_sources_from_view(view)
    assert retained_fit is view.production_fit
    assert retained_features is features
    assert not is_production_fit_same_k_view(copy.copy(view))
    with pytest.raises(ModelError, match="lacks builder authority"):
        production_fit_sources_from_view(copy.copy(view))
    with pytest.raises(TypeError, match="created only"):
        ProductionFitSameKView()

    fingerprint = view.view_fingerprint
    object.__setattr__(view, "view_fingerprint", "0" * 64)
    assert not is_production_fit_same_k_view(view)
    object.__setattr__(view, "view_fingerprint", fingerprint)
    features.values.setflags(write=True)
    features.values[0, 0] += 1.0
    assert not is_production_fit_same_k_view(view)


def test_production_fit_rejects_cross_run_reused_binding_and_raw_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    features = _production_features()
    execution = _registered_production_execution(monkeypatch, selection, features)
    with pytest.raises(ModelError, match="another run"):
        build_production_fit_same_k_view(
            execution,
            run_id="9" * 64,
            binding_fingerprint=_PRODUCTION_BINDING,
        )
    with pytest.raises(ModelError, match="differ"):
        build_production_fit_same_k_view(
            execution,
            run_id=_RUN,
            binding_fingerprint=_SELECTION_BINDING,
        )
    with pytest.raises((ModelError, TypeError), match="executor authority|wrong type"):
        build_production_fit_same_k_view(
            execution.production_fit,  # type: ignore[arg-type]
            run_id=_RUN,
            binding_fingerprint=_PRODUCTION_BINDING,
        )


def test_production_fit_registered_execution_runs_once_and_has_raw_final_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    features = _production_features()
    calls = 0

    def spy(
        supplied: HMMFeatureMatrix,
        *,
        selected_n_states: int,
        master_seed: int,
        scientific_version: int,
    ) -> GaussianHMMFit:
        nonlocal calls
        calls += 1
        return _registered_fake_production_fit(
            supplied,
            selected_n_states=selected_n_states,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )

    execution = _registered_production_execution(
        monkeypatch,
        selection,
        features,
        fitter=spy,
    )
    raw_slots = production_fit_executor_source_slots(execution)
    view = build_production_fit_same_k_view(
        execution,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_BINDING,
    )
    assert calls == 1
    assert registered_production_fit_execution_from_view(view) is execution
    assert view.production_fit is execution.production_fit
    assert raw_slots == scientific_task_source_slots(view)

    copied_execution = copy.copy(execution)
    with pytest.raises(ModelError, match="executor authority"):
        production_fit_executor_source_slots(copied_execution)
    with pytest.raises(ModelError, match="executor authority"):
        build_production_fit_same_k_view(
            copied_execution,
            run_id=_RUN,
            binding_fingerprint="7" * 64,
        )

    equivalent_execution = _registered_production_execution(
        monkeypatch,
        selection,
        features,
        fitter=spy,
    )
    assert equivalent_execution is not execution
    assert equivalent_execution.execution_fingerprint == execution.execution_fingerprint
    object.__setattr__(view, "_registered_execution", equivalent_execution)
    assert not is_production_fit_same_k_view(view)
    object.__setattr__(view, "_registered_execution", execution)
    assert is_production_fit_same_k_view(view)


def test_candidate_holdout_and_production_prebinding_equal_final_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_result, viability, selection = _candidate_views()
    design, holdout, _ = _slices()
    holdout_evidence = _holdout_evidence(selection, design, holdout)
    holdout_view = build_sealed_holdout_veto_view(
        selection,
        holdout_evidence,
        design=design,
        holdout=holdout,
        run_id=_RUN,
        binding_fingerprint=_HOLDOUT_BINDING,
    )
    features = _production_features()
    execution = _registered_production_execution(monkeypatch, selection, features)
    production = build_production_fit_same_k_view(
        execution,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_BINDING,
    )

    with pytest.raises(ModelError, match="final fixed-point authority"):
        candidate_viability_executor_source_slots(
            candidate_result.viability_result,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )
    with pytest.raises(ModelError, match="complete fixed-point authority"):
        candidate_selection_executor_source_slots(
            viability,
            candidate_result,
        )
    with pytest.raises(ModelError, match="canonical task source"):
        scientific_task_source_slots(viability)
    with pytest.raises(ModelError, match="canonical task source"):
        scientific_task_source_slots(selection)
    assert sealed_holdout_executor_source_slots(
        selection,
        holdout_evidence,
        design=design,
        holdout=holdout,
    ) == scientific_task_source_slots(holdout_view)
    assert production_fit_executor_source_slots(execution) == (
        scientific_task_source_slots(production)
    )

    alternate_viability = build_candidate_viability_task_view(
        candidate_result.viability_result,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id="9" * 64,
        binding_fingerprint="8" * 64,
    )
    with pytest.raises(ModelError, match="canonical task source"):
        scientific_task_source_slots(alternate_viability)
    with pytest.raises(ModelError, match="complete fixed-point authority"):
        candidate_selection_executor_source_slots(
            viability,
            _candidate_stages()[1],
        )


def test_adequacy_prebinding_binds_exact_calibration_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    features = _production_features()
    execution = _registered_production_execution(monkeypatch, selection, features)
    production = build_production_fit_same_k_view(
        execution,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_BINDING,
    )
    design_fit = selected_design_fit_from_view(selection)
    production_fit, production_features = production_fit_sources_from_view(production)
    design_states = np.resize(
        np.repeat(np.asarray([0, 1], dtype=np.int64), 10),
        design_fit.n_observations,
    )
    design_values = design_fit.parameters.means[design_states] + np.random.default_rng(
        design_fit.n_observations
    ).normal(0.0, 0.05, size=(design_fit.n_observations, design_fit.n_features))
    design_features = HMMFeatureMatrix(
        design_values,
        design_fit.lengths,
        design_fit.feature_names,
        tuple(f"design-{row}" for row in range(design_fit.n_observations)),
        np.asarray(design_fit.scaler_means),
        np.asarray(design_fit.scaler_standard_deviations),
        "design_training",
    )

    def fast_refit(
        features: HMMFeatureMatrix,
        n_states: int,
        supplied_seeds: object,
    ) -> GaussianHMMFit:
        assert n_states == 2
        seeds = tuple(cast(tuple[int, ...], supplied_seeds))
        base = _fit(
            len(features.values),
            scaler_scope=features.scaler_scope,
            lengths=features.lengths,
            scaler_means=features.scaler_means,
            scaler_scales=features.scaler_standard_deviations,
            values=features.values,
        )
        diagnostics = tuple(
            RestartDiagnostic(
                restart_index=index,
                seed=seed,
                converged=index == 0,
                iterations=12 if index == 0 else 1,
                log_likelihood=base.log_likelihood if index == 0 else None,
                valid=index == 0,
                raw_minimum_eigenvalue=0.25 if index == 0 else None,
                maximum_floor_relative_change=0.0 if index == 0 else None,
                covariance_condition_number=1.0 if index == 0 else None,
                maximum_likelihood_decrease=0.0 if index == 0 else None,
                preliminary_chain_passed=index == 0,
                failure_type=None if index == 0 else "NonConvergence",
                failure_message=(None if index == 0 else "synthetic non-convergence"),
            )
            for index, seed in enumerate(seeds)
        )
        return replace(
            base,
            best_seed=seeds[0],
            restart_diagnostics=diagnostics,
        )

    design_latent_evidence = run_registered_latent_cell(
        design_fit,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        chain_count=20,
        measured_months=25,
        extension_months=100,
        bootstrap_replicates=30,
        resample_size=20,
        survival_months=5,
    )
    production_latent_evidence = run_registered_latent_cell(
        production_fit,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        chain_count=20,
        measured_months=25,
        extension_months=100,
        bootstrap_replicates=30,
        resample_size=20,
        survival_months=5,
    )
    design_refit_evidence = run_registered_refitted_adequacy_cell(
        role="design",
        original_fit=design_fit,
        artifact_features=design_features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        attempt_count=3,
        minimum_successes=2,
        strict_canonical=False,
        fit_function=fast_refit,
    )
    production_refit_evidence = run_registered_refitted_adequacy_cell(
        role="production",
        original_fit=production_fit,
        artifact_features=production_features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        attempt_count=3,
        minimum_successes=2,
        strict_canonical=False,
        fit_function=fast_refit,
    )
    evidence = (
        design_latent_evidence,
        production_latent_evidence,
        design_refit_evidence,
        production_refit_evidence,
    )
    parents = (selection, production, selection, production)
    task_ids = (
        "design_latent_correctness",
        "production_latent_correctness",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )
    bindings = ("2" * 64, "3" * 64, "4" * 64, "5" * 64)
    views = (
        build_latent_adequacy_task_view(
            task_id=task_ids[0],
            evidence=evidence[0],
            parent=parents[0],
            run_id=_RUN,
            binding_fingerprint=bindings[0],
            strict_canonical=False,
        ),
        build_latent_adequacy_task_view(
            task_id=task_ids[1],
            evidence=evidence[1],
            parent=parents[1],
            run_id=_RUN,
            binding_fingerprint=bindings[1],
            strict_canonical=False,
        ),
        build_refitted_adequacy_task_view(
            task_id=task_ids[2],
            evidence=evidence[2],
            parent=parents[2],
            run_id=_RUN,
            binding_fingerprint=bindings[2],
            strict_canonical=False,
        ),
        build_refitted_adequacy_task_view(
            task_id=task_ids[3],
            evidence=evidence[3],
            parent=parents[3],
            run_id=_RUN,
            binding_fingerprint=bindings[3],
            strict_canonical=False,
        ),
    )
    for raw, parent, view in zip(evidence, parents, views, strict=True):
        assert adequacy_cell_executor_source_slots(
            raw,
            parent,
            task_id=view.task_id,
            strict_canonical=False,
        ) == scientific_task_source_slots(view)

    holm = assemble_four_cell_adequacy_holm_view(
        design_latent=views[0],
        production_latent=views[1],
        design_refitted=views[2],
        production_refitted=views[3],
        binding_fingerprint="6" * 64,
        strict_canonical=False,
    )
    assert adequacy_holm_executor_source_slots(
        design_latent=views[0],
        production_latent=views[1],
        design_refitted=views[2],
        production_refitted=views[3],
        strict_canonical=False,
    ) == scientific_task_source_slots(holm)
    with pytest.raises(ModelError, match="parent|role"):
        adequacy_cell_executor_source_slots(
            design_latent_evidence,
            production,
            task_id="design_latent_correctness",
            strict_canonical=False,
        )


def test_design_and_production_macro_views_bind_registered_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    design, _, full = _slices()
    design_fit = selection._source_result.selected_fit
    assert isinstance(design_fit, GaussianHMMFit)
    design_evidence = _registered_macro(design_fit, design, role="design")
    assert is_registered_macro_stability_evidence(design_evidence)
    design_view = build_macro_stability_task_view(
        task_id="design_macro_stability",
        evidence=design_evidence,
        parent=selection,
        source_slice=design,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_DESIGN_MACRO_BINDING,
    )
    assert is_macro_stability_task_view(design_view)
    assert design_view.role == "design"
    assert design_view.rng_namespace == "design_macro_bootstrap_streams"

    _, _, production = _production_view(monkeypatch)
    production_evidence = _registered_macro(
        production.production_fit,
        full,
        role="production",
    )
    production_view = build_macro_stability_task_view(
        task_id="production_macro_stability",
        evidence=production_evidence,
        parent=production,
        source_slice=full,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_MACRO_BINDING,
    )
    assert is_macro_stability_task_view(production_view)
    assert production_view.role == "production"
    assert production_view.rng_namespace == "production_macro_bootstrap_streams"
    assert not is_macro_stability_task_view(copy.copy(production_view))
    with pytest.raises(TypeError, match="created only"):
        MacroStabilityTaskView()
    fingerprint = production_view.view_fingerprint
    object.__setattr__(production_view, "view_fingerprint", "0" * 64)
    assert not is_macro_stability_task_view(production_view)
    object.__setattr__(production_view, "view_fingerprint", fingerprint)


def test_macro_views_reject_unregistered_cross_role_seed_and_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    design, _, full = _slices()
    fit = selection._source_result.selected_fit
    assert isinstance(fit, GaussianHMMFit)
    registered = _registered_macro(fit, design, role="design")
    unregistered = replace(registered)
    with pytest.raises(ModelError, match="registered execution"):
        build_macro_stability_task_view(
            task_id="design_macro_stability",
            evidence=unregistered,
            parent=selection,
            source_slice=design,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
            run_id=_RUN,
            binding_fingerprint=_DESIGN_MACRO_BINDING,
        )
    with pytest.raises(ModelError, match="mixed"):
        build_macro_stability_task_view(
            task_id="design_macro_stability",
            evidence=registered,
            parent=selection,
            source_slice=design,
            master_seed=_MASTER_SEED + 1,
            scientific_version=_SCIENTIFIC_VERSION,
            run_id=_RUN,
            binding_fingerprint=_DESIGN_MACRO_BINDING,
        )

    _, _, production = _production_view(monkeypatch)
    production_evidence = _registered_macro(
        production.production_fit,
        full,
        role="production",
    )
    with pytest.raises(ModelError, match="design macro"):
        build_macro_stability_task_view(
            task_id="design_macro_stability",
            evidence=production_evidence,
            parent=production,
            source_slice=full,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
            run_id=_RUN,
            binding_fingerprint=_DESIGN_MACRO_BINDING,
        )

    view = build_macro_stability_task_view(
        task_id="production_macro_stability",
        evidence=production_evidence,
        parent=production,
        source_slice=full,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_MACRO_BINDING,
    )
    full.macro_features.setflags(write=True)
    full.macro_features[0, 0] += 1.0
    assert not is_macro_stability_task_view(view)


def test_all_type_guards_and_registry_boundaries_reject_unrelated_values() -> None:
    unrelated = object()
    assert not is_candidate_viability_task_view(unrelated)
    assert not is_candidate_selection_task_view(unrelated)
    assert not is_sealed_holdout_veto_view(unrelated)
    assert not is_production_fit_same_k_view(unrelated)
    assert not is_macro_stability_task_view(unrelated)
    with pytest.raises(ModelError, match="SHA-256"):
        build_candidate_viability_task_view(
            _candidate_stages()[0],
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
            run_id="not-a-run-id",
            binding_fingerprint=_VIABILITY_BINDING,
        )


def test_g3_assembly_derives_only_exact_sealed_calibration_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_result, viability, selection = _candidate_views()
    design, holdout, full = _slices()
    holdout_evidence = _holdout_evidence(selection, design, holdout)
    holdout_view = build_sealed_holdout_veto_view(
        selection,
        holdout_evidence,
        design=design,
        holdout=holdout,
        run_id=_RUN,
        binding_fingerprint=_HOLDOUT_BINDING,
    )
    design_fit = selected_design_fit_from_view(selection)
    design_macro_evidence = _registered_macro(design_fit, design, role="design")
    design_macro = build_macro_stability_task_view(
        task_id="design_macro_stability",
        evidence=design_macro_evidence,
        parent=selection,
        source_slice=design,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_DESIGN_MACRO_BINDING,
    )
    production_features = _production_features()
    production_execution = _registered_production_execution(
        monkeypatch,
        selection,
        production_features,
    )
    production = build_production_fit_same_k_view(
        production_execution,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_BINDING,
    )
    production_macro_evidence = _registered_macro(
        production.production_fit,
        full,
        role="production",
    )
    production_macro = build_macro_stability_task_view(
        task_id="production_macro_stability",
        evidence=production_macro_evidence,
        parent=production,
        source_slice=full,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_PRODUCTION_MACRO_BINDING,
    )

    sources = {
        "design_candidate_viability": viability,
        "design_predictive_selection": selection,
        "sealed_holdout_veto": holdout_view,
        "design_macro_stability": design_macro,
        "production_fit_same_k": production,
        "production_macro_stability": production_macro,
    }
    for task_id, source in sources.items():
        decision = derive_g3_gate_decision(task_id, source)
        assert decision.evidence["run_id"] == _RUN
        assert decision.evidence["binding_fingerprint"] == source.binding_fingerprint
        if task_id in {
            "design_candidate_viability",
            "design_predictive_selection",
        }:
            with pytest.raises(ModelError, match="canonical task source"):
                scientific_task_source_slots(source)
        else:
            slots = scientific_task_source_slots(source)
            assert slots.task_id == task_id
            assert "task_source_materialization" in dict(
                slots.materialization_fingerprints
            )
            assert all(
                fingerprint == scientific_component_slot_fingerprint(slot)
                for slot, fingerprint in slots.scientific_component_fingerprints
            )

    with pytest.raises(ModelError, match="final fixed-point authority"):
        candidate_viability_executor_source_slots(
            candidate_result.viability_result,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )
    with pytest.raises(ModelError, match="complete fixed-point authority"):
        candidate_selection_executor_source_slots(viability, candidate_result)

    prebinding = {
        "sealed_holdout_veto": sealed_holdout_executor_source_slots(
            selection,
            holdout_evidence,
            design=design,
            holdout=holdout,
        ),
        "design_macro_stability": macro_stability_executor_source_slots(
            task_id="design_macro_stability",
            evidence=design_macro_evidence,
            parent=selection,
            source_slice=design,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        ),
        "production_fit_same_k": production_fit_executor_source_slots(
            production_execution,
        ),
        "production_macro_stability": macro_stability_executor_source_slots(
            task_id="production_macro_stability",
            evidence=production_macro_evidence,
            parent=production,
            source_slice=full,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        ),
    }
    assert prebinding == {
        task_id: scientific_task_source_slots(source)
        for task_id, source in sources.items()
        if task_id not in {"design_candidate_viability", "design_predictive_selection"}
    }
    assert task_view_cross_binding_error(sources) is None

    with pytest.raises(ModelError, match="sealed staged task view"):
        derive_g3_gate_decision(
            "design_predictive_selection",
            copy.copy(selection),
        )
    with pytest.raises(ModelError, match="design_macro_stability"):
        derive_g3_gate_decision("design_macro_stability", production_macro)


def test_handler_enforces_staged_candidate_run_binding_and_parent_lineage() -> None:
    _, viability, selection = _candidate_views()

    def bundle(
        viability_source: object,
        selection_source: object,
    ) -> CanonicalG3HandlerBundle:
        values: dict[str, object] = {
            task_id: G3TaskFailure(task_id, "SyntheticFailure", "not executed")
            for task_id in G3_GATE_ORDER
        }
        values["design_candidate_viability"] = viability_source
        values["design_predictive_selection"] = selection_source
        return CanonicalG3HandlerBundle(**values)  # type: ignore[arg-type]

    assert bundle(viability, selection).design_predictive_selection is selection

    other_viability_result, other_result = _candidate_stages()
    other_viability = build_candidate_viability_task_view(
        other_viability_result,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id="9" * 64,
        binding_fingerprint="8" * 64,
    )
    other_selection = build_candidate_selection_task_view(
        other_viability,
        other_result,
        run_id="9" * 64,
        binding_fingerprint="7" * 64,
    )
    with pytest.raises(ModelError, match="lineage is invalid"):
        bundle(viability, other_selection)
    with pytest.raises(ModelError, match="sealed exact-task authority"):
        bundle(viability, copy.copy(selection))
