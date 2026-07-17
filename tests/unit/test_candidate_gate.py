from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import prpg.model.candidate_gate as gate_module
from prpg.errors import ModelError
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
    materialize_registered_rolling_bootstrap,
)
from prpg.model.candidate_gate import (
    CANONICAL_CANDIDATES,
    CandidateGateEvidence,
    SelectedOnlyVeto,
    apply_selected_only_vetoes,
    candidate_viability_thresholds,
    is_candidate_gate_result,
    run_candidate_gate,
    run_candidate_selection,
    run_candidate_viability,
    run_owner_fixed_k4_selection,
)
from prpg.model.hmm import GaussianHMMFit, summarize_state_path
from prpg.model.selection import CandidateSelectionInput, CandidateSelectionReport
from prpg.model.stability import MacroStabilitySummary, RollingStabilitySummary
from prpg.simulation.rng import CalibrationArtifact, ModelFitScope


class _Covariance:
    symmetric = True
    positive_definite = True
    eigenvalue_floor_passed = True
    condition_number_passed = True
    condition_number = 1_000_000.0


class _Chain:
    irreducible = True
    aperiodic = True
    unique_stationary_distribution = True
    period = 1
    stationary_residual = 1e-12
    near_absorbing_states: tuple[int, ...] = ()
    mixing_time_to_one_percent = 120


class _Passed:
    passed = True


class _Fit:
    def __init__(self, n_states: int, decoded: NDArray[np.int64], bic: float) -> None:
        self.n_states = n_states
        self.n_features = 4
        self.n_observations = len(decoded)
        self.bic = bic
        self.log_likelihood = -100.0
        self.decoded_states = decoded
        self.covariance_diagnostic = _Covariance()
        self.chain_diagnostics = _Chain()
        self.identification = _Passed()
        self.historical_transition_support = _Passed()


def _states(n_states: int, rows: int, run_length: int) -> NDArray[np.int64]:
    values = np.resize(np.repeat(np.arange(n_states), run_length), rows)
    return values.astype(np.int64)


def _candidate_components(
    n_states: int,
    *,
    bic: float,
    score: float,
    invalid_fold: int | None = None,
) -> tuple[
    GaussianHMMFit,
    DecodedReturnPools,
    RollingCandidateEvidence,
    MacroStabilityEvidence,
    BlockPolicySelection,
    BlockPolicySelection,
]:
    monthly = _states(n_states, 193, 10)
    daily = _states(n_states, 4_049, 100)
    monthly_diagnostic = summarize_state_path(monthly, n_states, lengths=(193,))
    daily_diagnostic = summarize_state_path(daily, n_states, lengths=(4_049,))
    fit = cast(GaussianHMMFit, _Fit(n_states, monthly, bic))
    pools = DecodedReturnPools(
        monthly,
        daily,
        monthly_diagnostic,
        daily_diagnostic,
    )

    rolling_attempts: list[RollingFitAttempt] = []
    for fold in range(6):
        failed = fold == invalid_fold
        jaccards = np.zeros(n_states) if failed else np.full(n_states, 0.60)
        rolling_attempts.append(
            RollingFitAttempt(
                fold=fold,
                training_rows=121 + 12 * fold,
                validation_rows=12,
                validation_lengths=(12,),
                fit=None if failed else fit,
                scores=(None,) * 12 if failed else (score,) * 12,
                adjusted_rand_index=0.0 if failed else 0.70,
                jaccards=jaccards,
                failure_type="ModelError" if failed else None,
                failure_message="synthetic failed fold" if failed else None,
            )
        )
    rolling_successes = 5 if invalid_fold is not None else 6
    rolling = RollingCandidateEvidence(
        n_states=n_states,
        role="design",
        attempts=tuple(rolling_attempts),
        stability=RollingStabilitySummary(
            successful_fits=rolling_successes,
            median_ari=0.70,
            minimum_successful_ari=0.40,
            median_jaccard_by_state=np.full(n_states, 0.60),
            passed=True,
        ),
    )

    macro_attempts = tuple(
        MacroBootstrapAttempt(
            attempt=attempt,
            source_indices=np.arange(193, dtype=np.int64),
            fit=fit,
            adjusted_rand_index=0.80,
            jaccards=np.full(n_states, 0.70),
            failure_type=None,
            failure_message=None,
        )
        for attempt in range(100)
    )
    macro = MacroStabilityEvidence(
        n_states=n_states,
        role="design",
        macro_block_length=2,
        attempts=macro_attempts,
        summary=MacroStabilitySummary(
            successful_attempts=100,
            median_ari=0.80,
            tenth_percentile_ari=0.50,
            median_jaccard_by_state=np.full(n_states, 0.70),
            tenth_percentile_jaccard_by_state=np.full(n_states, 0.50),
            passed=True,
        ),
    )

    monthly_policy = select_supported_length(
        starting_length=6,
        frequency="monthly",
        states=monthly,
        continuation=np.ones(len(monthly) - 1, dtype=np.bool_),
        realized_means_by_length={6: dict.fromkeys(range(n_states), 3.0)},
        selected_states=tuple(range(n_states)),
    )
    daily_policy = select_supported_length(
        starting_length=60,
        frequency="daily",
        states=daily,
        continuation=np.ones(len(daily) - 1, dtype=np.bool_),
        realized_means_by_length={60: dict.fromkeys(range(n_states), 30.0)},
        selected_states=tuple(range(n_states)),
    )
    return fit, pools, rolling, macro, monthly_policy, daily_policy


@pytest.fixture(scope="module")
def registered_indices() -> NDArray[np.int64]:
    values = np.broadcast_to(
        np.arange(12, dtype=np.int64),
        (10_000, 6, 12),
    ).copy()
    values.setflags(write=False)
    return values


@pytest.fixture(scope="module")
def registered_materialization() -> RegisteredRollingBootstrapMaterialization:
    return materialize_registered_rolling_bootstrap(
        master_seed=20_250_308,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )


@pytest.fixture()
def complete_inputs() -> tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]]:
    components = {
        2: _candidate_components(2, bic=100.0, score=1.0),
        4: _candidate_components(4, bic=90.0, score=2.0, invalid_fold=2),
        5: _candidate_components(5, bic=108.0, score=0.0),
    }
    attempts: list[CandidateFitAttempt] = []
    evidence: list[CandidateGateEvidence] = []
    for n_states in CANONICAL_CANDIDATES:
        if n_states == 3:
            attempts.append(
                CandidateFitAttempt(
                    n_states=3,
                    fit=None,
                    failure_type="ModelError",
                    failure_message="no valid restart",
                    failure_details={"valid_restarts": 0},
                )
            )
            evidence.append(CandidateGateEvidence(3, None, None, None, None, None))
            continue
        fit, pools, rolling, macro, monthly_policy, daily_policy = components[n_states]
        attempts.append(CandidateFitAttempt(n_states, fit, None, None, None))
        evidence.append(
            CandidateGateEvidence(
                n_states=n_states,
                decoded_return_pools=pools,
                rolling=rolling,
                macro_stability=macro,
                monthly_block_policy=monthly_policy,
                daily_block_policy=daily_policy,
            )
        )
    return (
        CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, tuple(attempts)),
        tuple(evidence),
    )


def test_canonical_selection_rejects_raw_array_and_wrong_artifact(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    viability = run_candidate_viability(fit_set, evidence, role="design")
    with pytest.raises(ModelError, match="registered rolling bootstrap"):
        run_candidate_selection(viability, registered_indices)  # type: ignore[arg-type]

    production = materialize_registered_rolling_bootstrap(
        master_seed=20_250_308,
        scientific_version=1,
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    )
    with pytest.raises(ModelError, match="wrong artifact"):
        run_candidate_selection(viability, production)


def test_canonical_selection_reuses_and_reverifies_exact_registered_trace(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_materialization: RegisteredRollingBootstrapMaterialization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_set, evidence = complete_inputs
    viability = run_candidate_viability(fit_set, evidence, role="design")
    viability_identity = gate_module.CandidateViabilityExecutionIdentity(
        role="design",
        fit_set_fingerprint="1" * 64,
        rolling_fingerprints=(None, None, None, None),
        source_content_fingerprint="2" * 64,
        viability_result_fingerprint="3" * 64,
    )
    monkeypatch.setattr(
        gate_module,
        "verified_candidate_viability_execution_identity",
        lambda _value: viability_identity,
    )
    original = gate_module.select_candidate
    observed: list[object] = []

    def observe(
        candidates: Sequence[CandidateSelectionInput],
        *,
        bootstrap_indices: ArrayLike | None = None,
    ) -> CandidateSelectionReport:
        observed.append(bootstrap_indices)
        return original(candidates, bootstrap_indices=bootstrap_indices)

    monkeypatch.setattr(gate_module, "select_candidate", observe)
    result = run_candidate_selection(viability, registered_materialization)
    identity = gate_module.verified_candidate_gate_execution_identity(result)

    assert len(observed) == 1
    assert observed[0] is registered_materialization.indices
    assert result._registered_bootstrap_indices is registered_materialization.indices
    assert result._registered_bootstrap_materialization is registered_materialization
    assert identity.bootstrap_fingerprint == (
        registered_materialization.materialization_fingerprint
    )
    assert identity.bootstrap_materialization_fingerprint == (
        registered_materialization.materialization_fingerprint
    )
    assert identity.bootstrap_rng_execution_fingerprint == (
        registered_materialization.rng_execution_fingerprint
    )
    assert identity.bootstrap_master_seed == registered_materialization.master_seed
    assert identity.bootstrap_scientific_version == (
        registered_materialization.scientific_version
    )

    original_fingerprint = registered_materialization.materialization_fingerprint
    object.__setattr__(
        registered_materialization,
        "materialization_fingerprint",
        "f" * 64,
    )
    with pytest.raises(ModelError, match="execution-source fingerprints"):
        gate_module.verified_candidate_gate_execution_identity(result)
    object.__setattr__(
        registered_materialization,
        "materialization_fingerprint",
        original_fingerprint,
    )


def test_canonical_selection_reverifies_materialization_after_selector(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_set, evidence = complete_inputs
    viability = run_candidate_viability(fit_set, evidence, role="design")
    materialization = materialize_registered_rolling_bootstrap(
        master_seed=20_250_309,
        scientific_version=1,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )
    identity = gate_module.CandidateViabilityExecutionIdentity(
        role="design",
        fit_set_fingerprint="1" * 64,
        rolling_fingerprints=(None, None, None, None),
        source_content_fingerprint="2" * 64,
        viability_result_fingerprint="3" * 64,
    )
    monkeypatch.setattr(
        gate_module,
        "verified_candidate_viability_execution_identity",
        lambda _value: identity,
    )
    original_select = gate_module.select_candidate
    original_fingerprint = materialization.materialization_fingerprint

    def mutate_after_use(
        candidates: Sequence[CandidateSelectionInput],
        *,
        bootstrap_indices: ArrayLike | None = None,
    ) -> CandidateSelectionReport:
        result = original_select(candidates, bootstrap_indices=bootstrap_indices)
        object.__setattr__(
            materialization,
            "materialization_fingerprint",
            "e" * 64,
        )
        return result

    monkeypatch.setattr(gate_module, "select_candidate", mutate_after_use)
    try:
        with pytest.raises(ModelError, match="changed during selection"):
            run_candidate_selection(viability, materialization)
    finally:
        object.__setattr__(
            materialization,
            "materialization_fingerprint",
            original_fingerprint,
        )


def test_threshold_factory_uses_exact_block_multiples_and_absolute_floors() -> None:
    binding = candidate_viability_thresholds(6, 60)
    assert binding.minimum_monthly_observations == 30
    assert binding.minimum_daily_observations == 300
    assert binding.minimum_occupancy == 0.10
    assert binding.minimum_completed_monthly_episodes == 3
    assert binding.minimum_daily_runs == 3
    assert binding.maximum_covariance_condition_number == 1_000_000.0
    assert binding.minimum_bootstrap_median_ari == 0.80
    assert binding.minimum_rolling_median_ari == 0.70

    floors = candidate_viability_thresholds(None, 2)
    assert floors.minimum_monthly_observations == 24
    assert floors.minimum_daily_observations == 252

    fixed_k4 = candidate_viability_thresholds(6, 60, n_states=4)
    assert fixed_k4.minimum_monthly_observations == 1
    assert fixed_k4.minimum_daily_observations == 1
    assert fixed_k4.minimum_occupancy == 0.0
    assert fixed_k4.minimum_completed_monthly_episodes == 0
    assert fixed_k4.minimum_daily_runs == 0
    with pytest.raises(ModelError, match="monthly block length"):
        candidate_viability_thresholds(13, 2)
    with pytest.raises(ModelError, match="state count"):
        candidate_viability_thresholds(6, 60, n_states=6)


def test_owner_fixed_k4_selector_cannot_promote_cross_k_report_winner(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_materialization: RegisteredRollingBootstrapMaterialization,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_set, evidence = complete_inputs
    viability = run_candidate_viability(fit_set, evidence, role="design")
    identity = gate_module.CandidateViabilityExecutionIdentity(
        role="design",
        fit_set_fingerprint="1" * 64,
        rolling_fingerprints=(None, None, None, None),
        source_content_fingerprint="2" * 64,
        viability_result_fingerprint="3" * 64,
    )
    monkeypatch.setattr(
        gate_module,
        "verified_candidate_viability_execution_identity",
        lambda _value: identity,
    )

    result = run_owner_fixed_k4_selection(viability, registered_materialization)

    assert result.selection.selection_policy == "owner_fixed_k4_v3"
    assert result.selection.selected_n_states == 4
    assert result.selection.minimum_bic_n_states == 4
    assert result.selection.predictive_best_n_states == 2
    assert result.selection.bootstrap_replicates == 0
    rows = {row.n_states: row for row in result.selection.candidates}
    assert rows[4].selected
    assert rows[4].invalid_rolling_folds == (2,)
    assert rows[2].disposition == "report_only_nonfixed_k"


def test_owner_fixed_k4_requires_two_observed_monthly_episodes_per_state(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
) -> None:
    fit_set, evidence = complete_inputs
    changed = list(evidence)
    pools = changed[2].decoded_return_pools
    assert pools is not None
    monthly = replace(
        pools.monthly_diagnostics,
        episode_counts=np.asarray([1, 2, 2, 2], dtype=np.int64),
    )
    changed[2] = replace(
        changed[2],
        decoded_return_pools=replace(pools, monthly_diagnostics=monthly),
    )

    viability = run_candidate_viability(fit_set, tuple(changed), role="design")
    fixed = viability.audit_rows[2].viability

    assert not fixed.admissible
    assert any(
        reason.check_id == "owner_fixed_k4_observed_episode_support"
        for reason in fixed.rejection_reasons
    )


def test_owner_fixed_k4_completed_episode_counts_are_nonbinding(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
) -> None:
    fit_set, evidence = complete_inputs
    changed = list(evidence)
    pools = changed[2].decoded_return_pools
    assert pools is not None
    monthly = replace(
        pools.monthly_diagnostics,
        completed_episode_counts=np.zeros(4, dtype=np.int64),
    )
    changed[2] = replace(
        changed[2],
        decoded_return_pools=replace(pools, monthly_diagnostics=monthly),
    )

    viability = run_candidate_viability(fit_set, tuple(changed), role="design")
    fixed = viability.audit_rows[2].viability

    assert fixed.admissible
    assert fixed.thresholds.minimum_completed_monthly_episodes == 0


def test_gate_retains_failed_k_invalid_fold_and_selects_from_complete_family(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    result = run_candidate_gate(
        fit_set,
        evidence,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )

    assert is_candidate_gate_result(result)
    assert not is_candidate_gate_result(replace(result))
    assert tuple(row.n_states for row in result.audit_rows) == (2, 3, 4, 5)
    failed = result.audit_rows[1]
    assert not failed.fit_succeeded
    assert failed.fit_failure_message == "no valid restart"
    assert failed.fit_failure_details == {"valid_restarts": 0}
    assert not failed.viability.admissible
    assert "fit_succeeded" in {
        reason.check_id for reason in failed.viability.rejection_reasons
    }

    invalid_fold = result.selection.candidates[2]
    assert invalid_fold.n_states == 4
    assert invalid_fold.admissible
    assert invalid_fold.invalid_rolling_folds == (2,)
    assert not invalid_fold.predictively_eligible
    assert invalid_fold.disposition == "predictively_ineligible"

    assert result.selection.minimum_bic_n_states == 4
    assert result.selection.predictive_best_n_states == 2
    assert result.selection.selected_n_states == 2
    assert result.selection.passed
    assert result.selected_fit is fit_set.attempts[0].fit
    assert (
        json.loads(json.dumps(result.as_dict()))["selection"]["selected_n_states"] == 2
    )


def test_exact_threshold_equalities_are_admissible_before_predictive_selection(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    result = run_candidate_gate(
        fit_set,
        evidence,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    k5 = result.audit_rows[3].viability

    assert k5.admissible
    assert k5.thresholds.minimum_monthly_observations == 30
    assert k5.thresholds.minimum_daily_observations == 300
    assert all(
        state.covariance_condition_number == 1_000_000.0 for state in k5.evidence.states
    )
    assert k5.evidence.bootstrap_median_ari == 0.80
    assert k5.evidence.rolling_median_ari == 0.70


def test_applicable_fragmentation_failure_is_named_and_pre_bic(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    changed = list(evidence)
    changed[3] = replace(changed[3], daily_fragmentation_passed=False)
    result = run_candidate_gate(
        fit_set,
        changed,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    k5 = result.audit_rows[3].viability

    assert not k5.admissible
    assert any(
        reason.check_id == "daily_fragmentation" for reason in k5.rejection_reasons
    )


def test_canonical_boundary_rejects_a_missing_candidate(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    with pytest.raises(ModelError, match="exact K=2,3,4,5"):
        run_candidate_gate(
            fit_set,
            evidence[:-1],
            role="design",
            registered_bootstrap_indices=registered_indices,
        )


def test_partial_invalid_fold_is_structural_error(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    changed = list(evidence)
    rolling = cast(RollingCandidateEvidence, changed[0].rolling)
    attempts = list(rolling.attempts)
    attempts[0] = replace(attempts[0], scores=(None,) + (1.0,) * 11)
    changed[0] = replace(changed[0], rolling=replace(rolling, attempts=tuple(attempts)))

    with pytest.raises(ModelError, match="successful rolling fold"):
        run_candidate_gate(
            fit_set,
            changed,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )


def test_registered_index_contract_is_checked_even_before_pairwise_need(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
) -> None:
    fit_set, evidence = complete_inputs
    with pytest.raises(ModelError, match="registered rolling bootstrap"):
        run_candidate_gate(
            fit_set,
            evidence,
            role="design",
            registered_bootstrap_indices=np.zeros((1, 6, 12), dtype=np.int64),
        )


def test_selected_only_veto_fails_without_runner_up_promotion(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    gate = run_candidate_gate(
        fit_set,
        evidence,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    authorization = apply_selected_only_vetoes(
        gate,
        (
            SelectedOnlyVeto("refitted_adequacy", True),
            SelectedOnlyVeto("sealed_holdout", False),
        ),
    )

    assert authorization.selected_n_states == 2
    assert authorization.authorized_n_states is None
    assert not authorization.passed
    assert authorization.failure_reason == "selected_candidate_veto_failed"
    assert gate.selection.candidates[3].n_states == 5
    assert not gate.selection.candidates[3].selected


def test_veto_ids_are_unique_and_do_not_authorize_failed_selection(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    gate = run_candidate_gate(
        fit_set,
        evidence,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    with pytest.raises(ModelError, match="unique lower snake_case"):
        apply_selected_only_vetoes(
            gate,
            (SelectedOnlyVeto("holdout", True), SelectedOnlyVeto("holdout", True)),
        )


def test_successful_and_failed_selected_only_authorization_are_explicit(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    gate = run_candidate_gate(
        fit_set,
        evidence,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    authorized = apply_selected_only_vetoes(
        gate,
        (SelectedOnlyVeto("sealed_holdout", True),),
    )
    assert authorized.authorized_n_states == 2
    assert authorized.passed
    assert authorized.as_dict()["failure_reason"] is None

    failed_rows = tuple(
        replace(row, monthly_fragmentation_passed=False) if row.n_states != 3 else row
        for row in evidence
    )
    failed_gate = run_candidate_gate(
        fit_set,
        failed_rows,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    assert not failed_gate.selection.passed
    assert failed_gate.selected_fit is None
    failed_authorization = apply_selected_only_vetoes(
        failed_gate,
        (SelectedOnlyVeto("sealed_holdout", True),),
    )
    assert failed_authorization.selected_n_states is None
    assert failed_authorization.failure_reason == "candidate_selection_failed"


def test_structural_boundary_rejects_bad_role_fit_set_and_evidence_rows(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    with pytest.raises(ModelError, match="role"):
        run_candidate_gate(
            fit_set,
            evidence,
            role="other",  # type: ignore[arg-type]
            registered_bootstrap_indices=registered_indices,
        )

    reversed_fit_set = replace(fit_set, attempts=fit_set.attempts[::-1])
    with pytest.raises(ModelError, match="exact ordered"):
        run_candidate_gate(
            reversed_fit_set,
            evidence,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )

    duplicate = list(evidence)
    duplicate[-1] = replace(duplicate[-1], n_states=2)
    with pytest.raises(ModelError, match="unique"):
        run_candidate_gate(
            fit_set,
            duplicate,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )

    outside = list(evidence)
    outside[-1] = replace(outside[-1], n_states=6)
    with pytest.raises(ModelError, match="must be 2, 3, 4, or 5"):
        run_candidate_gate(
            fit_set,
            outside,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )


def test_inconsistent_decoded_pool_fails_named_check_without_dropping_k(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    changed = list(evidence)
    pools = cast(DecodedReturnPools, changed[0].decoded_return_pools)
    mismatched = pools.monthly_states.copy()
    mismatched[0] = 1
    changed[0] = replace(
        changed[0],
        decoded_return_pools=replace(pools, monthly_states=mismatched),
    )
    result = run_candidate_gate(
        fit_set,
        changed,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    k2 = result.audit_rows[0]
    assert not k2.viability.admissible
    assert any(
        reason.check_id == "decoded_return_pools"
        for reason in k2.viability.rejection_reasons
    )
    assert tuple(row.n_states for row in result.audit_rows) == (2, 3, 4, 5)


def test_malformed_rolling_records_fail_closed(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    rolling = cast(RollingCandidateEvidence, evidence[0].rolling)

    short_attempts = list(rolling.attempts)
    short_attempts[0] = replace(short_attempts[0], scores=(1.0,) * 11)
    short_rows = list(evidence)
    short_rows[0] = replace(
        short_rows[0], rolling=replace(rolling, attempts=tuple(short_attempts))
    )
    with pytest.raises(ModelError, match="exactly 12"):
        run_candidate_gate(
            fit_set,
            short_rows,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )

    nonfinite_attempts = list(rolling.attempts)
    nonfinite_attempts[0] = replace(
        nonfinite_attempts[0], scores=(float("inf"),) + (1.0,) * 11
    )
    nonfinite_rows = list(evidence)
    nonfinite_rows[0] = replace(
        nonfinite_rows[0],
        rolling=replace(rolling, attempts=tuple(nonfinite_attempts)),
    )
    with pytest.raises(ModelError, match="finite numeric"):
        run_candidate_gate(
            fit_set,
            nonfinite_rows,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )

    failed_rolling = cast(RollingCandidateEvidence, evidence[2].rolling)
    failed_attempts = list(failed_rolling.attempts)
    failed_attempts[2] = replace(failed_attempts[2], scores=(2.0,) * 12)
    failed_rows = list(evidence)
    failed_rows[2] = replace(
        failed_rows[2],
        rolling=replace(failed_rolling, attempts=tuple(failed_attempts)),
    )
    with pytest.raises(ModelError, match="failed rolling fold"):
        run_candidate_gate(
            fit_set,
            failed_rows,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )


def test_malformed_block_policy_and_fragmentation_fail_named_checks(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    changed = list(evidence)
    monthly = cast(BlockPolicySelection, changed[0].monthly_block_policy)
    changed[0] = replace(
        changed[0], monthly_block_policy=replace(monthly, restart_probability=0.5)
    )
    result = run_candidate_gate(
        fit_set,
        changed,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    assert any(
        reason.check_id == "monthly_block_support"
        for reason in result.audit_rows[0].viability.rejection_reasons
    )

    bad_fragmentation = list(evidence)
    bad_fragmentation[0] = replace(
        bad_fragmentation[0],
        monthly_fragmentation_passed=cast(bool, 1),
    )
    with pytest.raises(ModelError, match="outcome must be Boolean"):
        run_candidate_gate(
            fit_set,
            bad_fragmentation,
            role="design",
            registered_bootstrap_indices=registered_indices,
        )


def test_index_values_and_veto_record_types_fail_closed(
    complete_inputs: tuple[CandidateFitSet, tuple[CandidateGateEvidence, ...]],
    registered_indices: NDArray[np.int64],
) -> None:
    fit_set, evidence = complete_inputs
    invalid_indices = registered_indices.copy()
    invalid_indices[0, 0, 0] = 12
    with pytest.raises(ModelError, match=r"in \[0, 11\]"):
        run_candidate_gate(
            fit_set,
            evidence,
            role="design",
            registered_bootstrap_indices=invalid_indices,
        )

    gate = run_candidate_gate(
        fit_set,
        evidence,
        role="design",
        registered_bootstrap_indices=registered_indices,
    )
    with pytest.raises(ModelError, match="at least one veto"):
        apply_selected_only_vetoes(gate, ())
    with pytest.raises(ModelError, match="invalid record type"):
        apply_selected_only_vetoes(gate, (cast(SelectedOnlyVeto, object()),))
    with pytest.raises(ModelError, match="outcome must be Boolean"):
        apply_selected_only_vetoes(
            gate,
            (SelectedOnlyVeto("holdout", cast(bool, 1)),),
        )
