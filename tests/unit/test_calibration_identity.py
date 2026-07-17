from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import prpg.model.calibration_identity as identity
from prpg.errors import ModelError
from prpg.model.benchmarks import HoldoutBenchmarkScores, HoldoutVetoDecision
from prpg.model.calibration import (
    CandidateFitAttempt,
    CandidateFitSet,
    DecodedReturnPools,
    MacroBootstrapAttempt,
    MacroStabilityEvidence,
    RollingCandidateEvidence,
    RollingFitAttempt,
    SealedHoldoutEvidence,
)
from prpg.model.calibration_identity import (
    calibration_slice_fingerprint,
    candidate_fit_set_fingerprint,
    decoded_return_pools_fingerprint,
    macro_stability_evidence_fingerprint,
    rolling_candidate_evidence_fingerprint,
    sealed_holdout_evidence_fingerprint,
)
from prpg.model.hmm import GaussianHMMFit
from prpg.model.input import CalibrationSlice
from prpg.model.stability import MacroStabilitySummary, RollingStabilitySummary
from prpg.simulation.rng import ModelFitScope


def _slice() -> CalibrationSlice:
    return CalibrationSlice(
        role="design",
        monthly_returns=np.asarray([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
        daily_returns=np.asarray([[0.001, 0.002, 0.003]]),
        macro_features=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]),
        monthly_period_ordinals=np.asarray([600, 601], dtype=np.int64),
        daily_session_ns=np.asarray([1], dtype=np.int64),
        monthly_continuation=np.asarray([False, True]),
        daily_continuation=np.asarray([False]),
    )


def test_slice_and_decoded_pool_identities_bind_content_and_geometry() -> None:
    source = _slice()
    changed = replace(
        source,
        monthly_returns=np.asarray([[0.01, 0.02, 0.03], [0.04, 0.05, 0.07]]),
    )
    assert calibration_slice_fingerprint(source) != calibration_slice_fingerprint(
        changed
    )

    pools = DecodedReturnPools(
        monthly_states=np.asarray([0, 1]),
        daily_states=np.asarray([1]),
        monthly_diagnostics=object(),
        daily_diagnostics=object(),
    )
    changed_pools = replace(pools, monthly_states=np.asarray([1, 0]))
    assert decoded_return_pools_fingerprint(pools) != decoded_return_pools_fingerprint(
        changed_pools
    )


def test_candidate_fit_set_identity_retains_every_failed_candidate() -> None:
    attempts = tuple(
        CandidateFitAttempt(
            n_states=n_states,
            fit=None,
            failure_type="ModelError",
            failure_message=f"failed-{n_states}",
            failure_details={"k": n_states},
        )
        for n_states in (2, 3, 4, 5)
    )
    fit_set = CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, attempts)
    changed = replace(
        fit_set,
        attempts=(replace(attempts[0], failure_message="different"), *attempts[1:]),
    )
    assert candidate_fit_set_fingerprint(fit_set) != candidate_fit_set_fingerprint(
        changed
    )


def test_rolling_and_macro_identities_bind_registered_attempt_materializations() -> (
    None
):
    rolling_attempts = tuple(
        RollingFitAttempt(
            fold=fold,
            training_rows=100 + fold,
            validation_rows=12,
            validation_lengths=(12,),
            fit=None,
            scores=(None,) * 12,
            adjusted_rand_index=0.0,
            jaccards=np.zeros(2),
            failure_type="ModelError",
            failure_message="failed",
        )
        for fold in range(6)
    )
    rolling = RollingCandidateEvidence(
        n_states=2,
        role="design",
        attempts=rolling_attempts,
        stability=RollingStabilitySummary(
            successful_fits=0,
            median_ari=0.0,
            minimum_successful_ari=0.0,
            median_jaccard_by_state=np.zeros(2),
            passed=False,
        ),
    )
    changed_rolling = replace(
        rolling,
        attempts=(
            replace(rolling_attempts[0], failure_message="different"),
            *rolling_attempts[1:],
        ),
    )
    assert rolling_candidate_evidence_fingerprint(
        rolling
    ) != rolling_candidate_evidence_fingerprint(changed_rolling)

    macro_attempts = tuple(
        MacroBootstrapAttempt(
            attempt=attempt,
            source_indices=np.asarray([0, 1], dtype=np.int64),
            fit=None,
            adjusted_rand_index=0.0,
            jaccards=np.zeros(2),
            failure_type="ModelError",
            failure_message="failed",
        )
        for attempt in range(100)
    )
    macro = MacroStabilityEvidence(
        n_states=2,
        role="design",
        macro_block_length=2,
        attempts=macro_attempts,
        summary=MacroStabilitySummary(
            successful_attempts=0,
            median_ari=0.0,
            tenth_percentile_ari=0.0,
            median_jaccard_by_state=np.zeros(2),
            tenth_percentile_jaccard_by_state=np.zeros(2),
            passed=False,
        ),
    )
    changed_macro = replace(
        macro,
        attempts=(
            replace(macro_attempts[0], source_indices=np.asarray([1, 0])),
            *macro_attempts[1:],
        ),
    )
    assert macro_stability_evidence_fingerprint(
        macro
    ) != macro_stability_evidence_fingerprint(changed_macro)


def test_holdout_identity_binds_all_three_score_vectors_and_decision() -> None:
    evidence = SealedHoldoutEvidence(
        sequence_lengths=(2,),
        hmm_log_scores=np.asarray([-1.0, -2.0]),
        benchmark_scores=HoldoutBenchmarkScores(
            gaussian=np.asarray([-2.0, -3.0]),
            ridge_var1=np.asarray([-3.0, -4.0]),
        ),
        decision=HoldoutVetoDecision(
            hmm_mean_log_score=-1.5,
            gaussian_mean_log_score=-2.5,
            ridge_var1_mean_log_score=-3.5,
            hmm_minus_gaussian=1.0,
            hmm_minus_ridge_var1=2.0,
            passed=True,
        ),
    )
    changed = replace(evidence, hmm_log_scores=np.asarray([-1.0, -2.1]))
    assert sealed_holdout_evidence_fingerprint(
        evidence
    ) != sealed_holdout_evidence_fingerprint(changed)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (calibration_slice_fingerprint, "CalibrationSlice"),
        (candidate_fit_set_fingerprint, "CandidateFitSet"),
        (decoded_return_pools_fingerprint, "DecodedReturnPools"),
        (rolling_candidate_evidence_fingerprint, "RollingCandidateEvidence"),
        (macro_stability_evidence_fingerprint, "MacroStabilityEvidence"),
        (sealed_holdout_evidence_fingerprint, "SealedHoldoutEvidence"),
    ],
)
def test_identity_functions_reject_wrong_record_types(
    function: object, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        function(object())  # type: ignore[operator]


def test_invalid_roles_candidates_and_pool_geometry_fail_closed() -> None:
    with pytest.raises(ModelError, match="role"):
        calibration_slice_fingerprint(
            replace(_slice(), role="wrong")  # type: ignore[arg-type]
        )

    failed = CandidateFitAttempt(2, None, None, None, None)
    with pytest.raises(ModelError, match="failure evidence"):
        candidate_fit_set_fingerprint(
            CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, (failed,))
        )

    fake_fit = cast(GaussianHMMFit, object())
    inconsistent = CandidateFitAttempt(
        2,
        fake_fit,
        "ModelError",
        "stale",
        None,
    )
    with pytest.raises(ModelError, match="retains failure"):
        candidate_fit_set_fingerprint(
            CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, (inconsistent,))
        )

    for pools in (
        DecodedReturnPools(np.asarray([]), np.asarray([0]), object(), object()),
        DecodedReturnPools(np.asarray([[0]]), np.asarray([0]), object(), object()),
    ):
        with pytest.raises(ModelError, match="nonempty vectors"):
            decoded_return_pools_fingerprint(pools)


def test_successful_fit_and_alignment_branches_are_identity_bearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_fit = cast(GaussianHMMFit, object())
    monkeypatch.setattr(
        identity,
        "gaussian_hmm_fit_fingerprint",
        lambda _fit: "a" * 64,
    )
    successful = CandidateFitAttempt(2, fake_fit, None, None, None)
    assert (
        len(
            candidate_fit_set_fingerprint(
                CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, (successful,))
            )
        )
        == 64
    )

    rolling_attempt = RollingFitAttempt(
        fold=0,
        training_rows=100,
        validation_rows=12,
        validation_lengths=(12,),
        fit=fake_fit,
        scores=tuple(float(value) for value in range(12)),
        adjusted_rand_index=1.0,
        jaccards=np.ones(2),
        failure_type=None,
        failure_message=None,
    )
    rolling = RollingCandidateEvidence(
        2,
        "design",
        (rolling_attempt,),
        RollingStabilitySummary(1, 1.0, 1.0, np.ones(2), True),
    )
    assert len(rolling_candidate_evidence_fingerprint(rolling)) == 64

    macro_attempt = MacroBootstrapAttempt(
        attempt=0,
        source_indices=np.asarray([0, 1]),
        fit=fake_fit,
        adjusted_rand_index=1.0,
        jaccards=np.ones(2),
        failure_type=None,
        failure_message=None,
        candidate_to_reference=np.asarray([0, 1]),
    )
    macro = MacroStabilityEvidence(
        2,
        "design",
        2,
        (macro_attempt,),
        MacroStabilitySummary(1, 1.0, 1.0, np.ones(2), np.ones(2), True),
    )
    assert len(macro_stability_evidence_fingerprint(macro)) == 64


def test_nonfinite_rolling_score_fails_closed() -> None:
    attempt = RollingFitAttempt(
        fold=0,
        training_rows=100,
        validation_rows=12,
        validation_lengths=(12,),
        fit=None,
        scores=(float("nan"),),
        adjusted_rand_index=0.0,
        jaccards=np.zeros(2),
        failure_type="ModelError",
        failure_message="failed",
    )
    evidence = RollingCandidateEvidence(
        2,
        "design",
        (attempt,),
        RollingStabilitySummary(0, 0.0, 0.0, np.zeros(2), False),
    )
    with pytest.raises(ModelError, match="non-finite"):
        rolling_candidate_evidence_fingerprint(evidence)
