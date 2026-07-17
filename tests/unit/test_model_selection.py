from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from prpg.errors import ModelError
from prpg.model.selection import (
    ROLLING_ADVANCE_PROBABILITY,
    ROLLING_BOOTSTRAP_REPLICATES,
    ROLLING_FOLD_COUNT,
    ROLLING_FOLD_SIZE,
    ROLLING_RESTART_PROBABILITIES,
    ROLLING_SCORE_COUNT,
    CandidateSelectionInput,
    CandidateViabilityEvidence,
    CandidateViabilityRecord,
    PreselectionCheck,
    StateViabilityEvidence,
    ViabilityThresholds,
    apply_holdout_veto,
    conditional_rolling_log_score,
    evaluate_candidate_viability,
    rolling_score_bootstrap_trace,
    rolling_score_fold_trace_from_uniforms,
    select_candidate,
    select_owner_fixed_k4,
    simultaneous_rolling_score_intervals,
    summarize_holdout_veto,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)


def _thresholds() -> ViabilityThresholds:
    return ViabilityThresholds(
        minimum_monthly_observations=24,
        minimum_occupancy=0.10,
        minimum_completed_monthly_episodes=3,
        minimum_daily_observations=252,
        minimum_daily_runs=3,
        maximum_covariance_condition_number=1_000_000.0,
        minimum_bootstrap_median_ari=0.70,
        minimum_rolling_median_ari=0.70,
    )


def _state(state: int, **changes: object) -> StateViabilityEvidence:
    result = StateViabilityEvidence(
        state=state,
        monthly_observations=50,
        occupancy=0.25,
        completed_monthly_episodes=5,
        daily_observations=900,
        daily_runs=8,
        covariance_condition_number=25.0,
    )
    return replace(result, **changes)


def _evidence(
    n_states: int,
    *,
    bic: float | None = 100.0,
    states: tuple[StateViabilityEvidence, ...] | None = None,
    bootstrap_ari: float | None = 0.8,
    rolling_ari: float | None = 0.8,
    checks: tuple[PreselectionCheck, ...] = (
        PreselectionCheck("duration_gate", True),
        PreselectionCheck("markov_gate", True),
    ),
) -> CandidateViabilityEvidence:
    return CandidateViabilityEvidence(
        n_states=n_states,
        bic=bic,
        states=tuple(_state(state) for state in range(n_states))
        if states is None
        else states,
        bootstrap_median_ari=bootstrap_ari,
        rolling_median_ari=rolling_ari,
        preselection_checks=checks,
    )


def _viable(n_states: int, bic: float) -> CandidateViabilityRecord:
    return evaluate_candidate_viability(
        _evidence(n_states, bic=bic), thresholds=_thresholds()
    )


def _candidate(
    n_states: int,
    bic: float,
    scores: float | tuple[float | None, ...],
) -> CandidateSelectionInput:
    expanded = (scores,) * ROLLING_SCORE_COUNT if isinstance(scores, float) else scores
    return CandidateSelectionInput(_viable(n_states, bic), expanded)


def _scores_with_invalid_fold(fold: int = 0) -> tuple[float | None, ...]:
    scores: list[float | None] = [0.0] * ROLLING_SCORE_COUNT
    start = fold * ROLLING_FOLD_SIZE
    scores[start : start + ROLLING_FOLD_SIZE] = [None] * ROLLING_FOLD_SIZE
    return tuple(scores)


@pytest.fixture(scope="module")
def bootstrap_indices() -> NDArray[np.int64]:
    indices = np.empty(
        (
            ROLLING_BOOTSTRAP_REPLICATES,
            ROLLING_FOLD_COUNT,
            ROLLING_FOLD_SIZE,
        ),
        dtype=np.int64,
    )
    for replicate in range(ROLLING_BOOTSTRAP_REPLICATES):
        rng = calibration_rng_key(
            master_seed=20_260_714,
            scientific_version=1,
            purpose=CalibrationPurpose.ROLLING_SCORE_BOOTSTRAP,
            artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
            variant=0,
            replicate=replicate,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=Family.NOT_APPLICABLE,
            stage=Stage.REFERENCE_RESAMPLING,
        ).generator()
        indices[replicate] = rolling_score_bootstrap_trace(rng)
    indices.setflags(write=False)
    return indices


def test_viability_accepts_threshold_equality_and_normalizes_order() -> None:
    thresholds = _thresholds()
    state0 = _state(
        0,
        monthly_observations=24,
        occupancy=0.10,
        completed_monthly_episodes=3,
        daily_observations=252,
        daily_runs=3,
        covariance_condition_number=1_000_000.0,
    )
    state1 = replace(state0, state=1)
    evidence = _evidence(
        2,
        states=(state1, state0),
        bootstrap_ari=0.70,
        rolling_ari=0.70,
        checks=(
            PreselectionCheck("z_check", True),
            PreselectionCheck("a_check", True),
        ),
    )

    record = evaluate_candidate_viability(evidence, thresholds=thresholds)

    assert record.admissible
    assert [state.state for state in record.evidence.states] == [0, 1]
    assert [check.check_id for check in record.evidence.preselection_checks] == [
        "a_check",
        "z_check",
    ]
    report = record.as_dict()
    assert report["admissible"] is True
    assert report["thresholds"] == {
        "minimum_monthly_observations": 24,
        "minimum_occupancy": 0.10,
        "minimum_completed_monthly_episodes": 3,
        "minimum_daily_observations": 252,
        "minimum_daily_runs": 3,
        "maximum_covariance_condition_number": 1_000_000.0,
        "minimum_bootstrap_median_ari": 0.70,
        "minimum_rolling_median_ari": 0.70,
    }


def test_viability_retains_every_rejection_in_fixed_order() -> None:
    evidence = _evidence(
        2,
        bic=float("nan"),
        states=(
            _state(
                1,
                monthly_observations=23,
                occupancy=0.09,
                completed_monthly_episodes=2,
                daily_observations=251,
                daily_runs=2,
                covariance_condition_number=float("inf"),
            ),
        ),
        bootstrap_ari=None,
        rolling_ari=0.69,
        checks=(
            PreselectionCheck("z_gate", False),
            PreselectionCheck("a_gate", False),
        ),
    )

    record = evaluate_candidate_viability(evidence, thresholds=_thresholds())

    assert not record.admissible
    assert [reason.code for reason in record.rejection_reasons] == [
        "preselection_check_failed",
        "preselection_check_failed",
        "bic_nonfinite",
        "state_evidence_missing",
        "monthly_observations_below_minimum",
        "occupancy_below_minimum",
        "completed_monthly_episodes_below_minimum",
        "daily_observations_below_minimum",
        "daily_runs_below_minimum",
        "covariance_condition_number_nonfinite",
        "bootstrap_median_ari_missing",
        "rolling_median_ari_below_minimum",
    ]
    assert [reason.check_id for reason in record.rejection_reasons[:2]] == [
        "a_gate",
        "z_gate",
    ]
    reason = record.rejection_reasons[4]
    assert reason.as_dict() == {
        "code": "monthly_observations_below_minimum",
        "required_relation": ">=",
        "state": 1,
        "check_id": None,
        "observed": 23,
        "threshold": 24,
    }
    serialized = record.as_dict()
    assert serialized["bic"] == "nan"
    states = serialized["states"]
    assert isinstance(states, list)
    assert states[0]["covariance_condition_number"] == "positive_infinity"
    json.dumps(serialized, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"minimum_monthly_observations": 0}, "positive integer"),
        ({"minimum_completed_monthly_episodes": -1}, "non-negative integer"),
        ({"minimum_daily_observations": True}, "positive integer"),
        ({"minimum_daily_runs": -1}, "non-negative integer"),
        ({"minimum_occupancy": 1.1}, "minimum occupancy"),
        ({"maximum_covariance_condition_number": 0.0}, "condition number"),
        ({"minimum_bootstrap_median_ari": -1.1}, "bootstrap"),
        ({"minimum_rolling_median_ari": float("nan")}, "rolling"),
    ],
)
def test_viability_threshold_contract_is_strict(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        evaluate_candidate_viability(
            _evidence(2), thresholds=replace(_thresholds(), **change)
        )


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (_evidence(0), "n_states"),
        (
            _evidence(2, states=(_state(0), _state(0))),
            "IDs must be unique",
        ),
        (_evidence(2, states=(_state(2),)), "outside"),
        (_evidence(2, states=(_state(-1),)), "non-negative"),
        (_evidence(2, states=(_state(0, occupancy=1.1),)), "occupancy"),
        (
            _evidence(2, states=(_state(0, covariance_condition_number=-1.0),)),
            "cannot be negative",
        ),
        (
            _evidence(
                2,
                checks=(
                    PreselectionCheck("same", True),
                    PreselectionCheck("same", False),
                ),
            ),
            "unique",
        ),
        (
            _evidence(2, checks=(PreselectionCheck("Not-Snake", True),)),
            "lower snake_case",
        ),
        (_evidence(2, bootstrap_ari=1.1), "bootstrap_median_ari"),
        (_evidence(2, rolling_ari="bad"), "rolling_median_ari"),  # type: ignore[arg-type]
        (_evidence(2, bic="bad"), "bic must be numeric"),  # type: ignore[arg-type]
    ],
)
def test_viability_evidence_contract_is_strict(
    evidence: CandidateViabilityEvidence, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        evaluate_candidate_viability(evidence, thresholds=_thresholds())


def test_missing_bic_and_nonfinite_stability_are_reported() -> None:
    record = evaluate_candidate_viability(
        _evidence(2, bic=None, bootstrap_ari=float("inf"), rolling_ari=float("nan")),
        thresholds=_thresholds(),
    )
    assert [reason.code for reason in record.rejection_reasons] == [
        "bic_missing",
        "bootstrap_median_ari_nonfinite",
        "rolling_median_ari_nonfinite",
    ]


def test_fold_trace_advances_without_circular_wrapping() -> None:
    trace = rolling_score_fold_trace_from_uniforms(
        0.0,
        np.zeros(11),
        np.zeros(11),
    )

    assert tuple(trace) == tuple(range(12))
    assert not trace.flags.writeable


def test_fold_trace_advance_boundary_is_exactly_three_quarters() -> None:
    below = np.full(11, np.nextafter(0.75, 0.0))
    at_boundary = below.copy()
    at_boundary[0] = 0.75

    advancing = rolling_score_fold_trace_from_uniforms(0.0, below, np.zeros(11))
    restarting = rolling_score_fold_trace_from_uniforms(
        0.0,
        at_boundary,
        np.zeros(11),
    )

    assert advancing[1] == 1
    assert restarting[1] == 0


def test_fold_trace_uses_boundary_corrected_restart_distribution() -> None:
    just_below_four_fifteenths = np.nextafter(4.0 / 15.0, 0.0)
    just_above_four_fifteenths = np.nextafter(4.0 / 15.0, 1.0)
    restart_uniforms = (
        0.0,
        just_below_four_fifteenths,
        just_above_four_fifteenths,
        np.nextafter(5.0 / 15.0, 1.0),
        np.nextafter(6.0 / 15.0, 1.0),
        np.nextafter(7.0 / 15.0, 1.0),
        np.nextafter(8.0 / 15.0, 1.0),
        np.nextafter(9.0 / 15.0, 1.0),
        np.nextafter(10.0 / 15.0, 1.0),
        np.nextafter(14.0 / 15.0, 1.0),
        np.nextafter(1.0, 0.0),
    )

    trace = rolling_score_fold_trace_from_uniforms(
        np.nextafter(1.0, 0.0),
        np.full(11, np.nextafter(1.0, 0.0)),
        restart_uniforms,
    )

    assert tuple(trace) == (11, 0, 0, 1, 2, 3, 4, 5, 6, 7, 11, 11)


@pytest.mark.parametrize("target", range(12))
def test_fold_trace_restart_distribution_reaches_every_index(target: int) -> None:
    restart_uniform = 2.0 / 15.0 if target == 0 else (target + 3.5) / 15.0
    trace = rolling_score_fold_trace_from_uniforms(
        np.nextafter(1.0, 0.0),
        np.zeros(11),
        np.full(11, restart_uniform),
    )
    assert trace[1] == target


def test_boundary_corrected_transition_has_uniform_invariant_distribution() -> None:
    restart = np.asarray(ROLLING_RESTART_PROBABILITIES)
    transition = np.tile(
        (1.0 - ROLLING_ADVANCE_PROBABILITY) * restart,
        (ROLLING_FOLD_SIZE, 1),
    )
    for current in range(ROLLING_FOLD_SIZE - 1):
        transition[current, current + 1] += ROLLING_ADVANCE_PROBABILITY
    transition[-1] = restart
    uniform = np.full(ROLLING_FOLD_SIZE, 1.0 / ROLLING_FOLD_SIZE)

    np.testing.assert_allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(uniform @ transition, uniform, rtol=0.0, atol=1e-15)


def test_six_fold_trace_is_deterministic_and_read_only() -> None:
    first = rolling_score_bootstrap_trace(np.random.default_rng(12345))
    second = rolling_score_bootstrap_trace(np.random.default_rng(12345))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (6, 12)
    assert first.min() >= 0
    assert first.max() <= 11
    assert not first.flags.writeable


def test_six_fold_trace_has_fixed_branch_independent_rng_consumption() -> None:
    observed_rng = np.random.default_rng(54321)
    rolling_score_bootstrap_trace(observed_rng)
    observed_next = observed_rng.random()

    reference_rng = np.random.default_rng(54321)
    reference_rng.random(6 + 6 * 11 + 6 * 11)
    expected_next = reference_rng.random()

    assert observed_next == expected_next


@pytest.mark.parametrize(
    ("first", "advances", "restarts", "message"),
    [
        (1.0, np.zeros(11), np.zeros(11), r"\[0, 1\)"),
        (0.0, np.zeros(10), np.zeros(11), "exactly 11"),
        (0.0, np.zeros(11), np.full(11, np.nan), "finite values"),
        (0.0, np.zeros((1, 11)), np.zeros(11), "exactly 11"),
        (0.0, np.zeros(11, dtype=bool), np.zeros(11), "cannot be boolean"),
    ],
)
def test_fold_trace_uniform_contract_fails_closed(
    first: float,
    advances: object,
    restarts: object,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        rolling_score_fold_trace_from_uniforms(
            first,
            advances,  # type: ignore[arg-type]
            restarts,  # type: ignore[arg-type]
        )


def test_six_fold_trace_requires_numpy_generator() -> None:
    with pytest.raises(ModelError, match="numpy Generator"):
        rolling_score_bootstrap_trace(object())  # type: ignore[arg-type]


def test_conditional_rolling_log_score_implements_registered_formula() -> None:
    assert (
        conditional_rolling_log_score(
            training_and_validation_log_likelihood=-130.0,
            training_log_likelihood=-100.0,
            validation_observations=12,
        )
        == -2.5
    )


@pytest.mark.parametrize(
    ("joint", "training", "observations", "message"),
    [
        (float("nan"), 0.0, 1, "likelihoods"),
        (0.0, float("inf"), 1, "likelihoods"),
        (0.0, 0.0, 0, "positive integer"),
        (1e308, -1e308, 1, "non-finite"),
    ],
)
def test_conditional_rolling_log_score_fails_closed(
    joint: float, training: float, observations: int, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        conditional_rolling_log_score(
            training_and_validation_log_likelihood=joint,
            training_log_likelihood=training,
            validation_observations=observations,
        )


def test_max_z_matches_direct_all_pairs_calculation(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    position = np.arange(ROLLING_SCORE_COUNT, dtype=np.float64)
    base = np.linspace(-2.0, 2.0, ROLLING_SCORE_COUNT)
    scores = np.stack(
        (
            base,
            base + 0.10 * np.sin(position),
            base - 0.20 * np.cos(position / 2.0),
        )
    ).reshape(3, ROLLING_FOLD_COUNT, ROLLING_FOLD_SIZE)

    result = simultaneous_rolling_score_intervals(
        (4, 2, 3),
        scores,
        bootstrap_indices,
    )

    assert result.candidate_state_counts == (2, 3, 4)
    assert len(result.pairwise_intervals) == 3
    assert result.bootstrap_replicates == 10_000
    ordered_scores = scores[[1, 2, 0]]
    fold_positions = np.arange(6)[None, :, None]
    direct_max_z = np.zeros(10_000)
    direct_standard_errors: list[float] = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        differences = ordered_scores[first] - ordered_scores[second]
        observed = float(differences.mean())
        bootstrap_means = differences[fold_positions, bootstrap_indices].mean(
            axis=(1, 2)
        )
        standard_error = float(bootstrap_means.std(ddof=1))
        direct_standard_errors.append(standard_error)
        np.maximum(
            direct_max_z,
            np.abs(bootstrap_means - observed) / standard_error,
            out=direct_max_z,
        )
    direct_q = float(np.quantile(direct_max_z, 0.95, method="higher"))
    assert result.critical_value == direct_q
    assert [pair.bootstrap_standard_error for pair in result.pairwise_intervals] == (
        pytest.approx(direct_standard_errors)
    )
    first_pair = result.pairwise_intervals[0]
    expected_difference = float((ordered_scores[0] - ordered_scores[1]).mean())
    assert first_pair.observed_mean_difference == expected_difference
    assert result.as_dict()["candidate_state_counts"] == [2, 3, 4]


def test_all_constant_pairs_set_q_zero_and_apply_exact_tie_tolerance(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    scores = np.stack(
        (
            np.full((6, 12), 0.0),
            np.full((6, 12), -5e-13),
            np.full((6, 12), 1.0),
        )
    )

    result = simultaneous_rolling_score_intervals((2, 3, 4), scores, bootstrap_indices)

    assert result.critical_value == 0.0
    assert all(pair.zero_scale for pair in result.pairwise_intervals)
    within_tolerance = result.pairwise_intervals[0]
    assert within_tolerance.interval_lower == within_tolerance.interval_upper
    assert within_tolerance.exact_tie
    assert within_tolerance.contains_zero
    assert not result.pairwise_intervals[1].contains_zero


def test_zero_scale_threshold_includes_exactly_one_e_minus_twelve() -> None:
    bootstrap_indices = np.ones((10_000, 6, 12), dtype=np.int64)
    bootstrap_indices[0] = 0
    scores = np.zeros((2, 6, 12))
    scores[0, :, 0] = 1e-10

    at_boundary = simultaneous_rolling_score_intervals(
        (2, 3), scores, bootstrap_indices
    )
    pair = at_boundary.pairwise_intervals[0]
    assert pair.bootstrap_standard_error == 1e-12
    assert pair.zero_scale
    assert at_boundary.critical_value == 0.0
    assert not pair.exact_tie

    scores[0, :, 0] = np.nextafter(1e-10, np.inf)
    above_boundary = simultaneous_rolling_score_intervals(
        (2, 3), scores, bootstrap_indices
    )
    assert above_boundary.pairwise_intervals[0].bootstrap_standard_error > 1e-12
    assert not above_boundary.pairwise_intervals[0].zero_scale


def test_max_z_input_contract_fails_closed(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    valid_scores = np.zeros((2, 6, 12))
    with pytest.raises(ModelError, match="at least 2"):
        simultaneous_rolling_score_intervals((2,), valid_scores[:1], bootstrap_indices)
    with pytest.raises(ModelError, match="unique"):
        simultaneous_rolling_score_intervals((2, 2), valid_scores, bootstrap_indices)
    with pytest.raises(ModelError, match="one of 2, 3, 4, or 5"):
        simultaneous_rolling_score_intervals(
            (2.0, 3),  # type: ignore[arg-type]
            valid_scores,
            bootstrap_indices,
        )
    with pytest.raises(ModelError, match=r"shape \(candidate, 6, 12\)"):
        simultaneous_rolling_score_intervals(
            (2, 3), np.zeros((2, 72)), bootstrap_indices
        )
    with pytest.raises(ModelError, match=r"shape \(10000, 6, 12\)"):
        simultaneous_rolling_score_intervals(
            (2, 3), valid_scores, bootstrap_indices[:-1]
        )
    with pytest.raises(ModelError, match="integer dtype"):
        simultaneous_rolling_score_intervals(
            (2, 3), valid_scores, bootstrap_indices.astype(np.float64)
        )
    invalid_indices = bootstrap_indices.copy()
    invalid_indices[0, 0, 0] = 12
    with pytest.raises(ModelError, match=r"\[0, 11\]"):
        simultaneous_rolling_score_intervals((2, 3), valid_scores, invalid_indices)
    invalid_scores = valid_scores.copy()
    invalid_scores[0, 0, 0] = np.nan
    with pytest.raises(ModelError, match="must be finite"):
        simultaneous_rolling_score_intervals((2, 3), invalid_scores, bootstrap_indices)


def test_selection_builds_all_pairs_and_chooses_smallest_near_best_k(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    alternating = np.where(np.arange(72) % 2 == 0, 0.80, 1.18)
    result = select_candidate(
        (
            _candidate(4, 102.0, 1.0),
            _candidate(2, 105.0, tuple(float(value) for value in alternating)),
            _candidate(3, 100.0, 0.0),
        ),
        bootstrap_indices=bootstrap_indices,
    )

    assert result.passed
    assert result.minimum_bic_n_states == 3
    assert result.predictive_best_n_states == 4
    assert result.selected_n_states == 2
    assert result.rolling_max_z is not None
    assert len(result.rolling_max_z.pairwise_intervals) == 3
    rows = {row.n_states: row for row in result.candidates}
    assert rows[2].predictively_near_best
    assert rows[2].candidate_minus_best_lower < 0.0  # type: ignore[operator]
    assert rows[2].candidate_minus_best_upper > 0.0  # type: ignore[operator]
    assert rows[3].disposition == "outside_simultaneous_near_best"
    assert result.as_dict()["selected_n_states"] == 2
    assert result.selection_policy == "predictive_bic_v1"


def test_owner_fixed_k4_selection_keeps_cross_k_comparisons_report_only() -> None:
    result = select_owner_fixed_k4(
        (
            _candidate(2, 90.0, 5.0),
            _candidate(3, 100.0, 3.0),
            _candidate(4, 120.0, _scores_with_invalid_fold(2)),
            _candidate(5, 110.0, 4.0),
        )
    )

    assert result.passed
    assert result.selection_policy == "owner_fixed_k4_v3"
    assert result.selected_n_states == 4
    assert result.minimum_bic_n_states == 2
    assert result.predictive_best_n_states == 2
    assert result.rolling_max_z is None
    assert result.bootstrap_replicates == 0
    rows = {row.n_states: row for row in result.candidates}
    assert rows[4].selected
    assert not rows[4].predictively_eligible
    assert rows[2].disposition == "report_only_nonfixed_k"
    assert not rows[2].selected


def test_owner_fixed_k4_selection_fails_without_promoting_other_k() -> None:
    failed_k4 = evaluate_candidate_viability(
        _evidence(4, bic=80.0, checks=(PreselectionCheck("fixed_gate", False),)),
        thresholds=_thresholds(),
    )
    result = select_owner_fixed_k4(
        (
            _candidate(2, 90.0, 5.0),
            _candidate(3, 100.0, 3.0),
            CandidateSelectionInput(failed_k4, ()),
            _candidate(5, 110.0, 4.0),
        )
    )

    assert not result.passed
    assert result.selected_n_states is None
    assert result.failure_reason == "owner_fixed_k4_not_admissible"
    assert all(not row.selected for row in result.candidates)
    assert {row.n_states for row in result.candidates} == {2, 3, 4, 5}


def test_owner_fixed_k4_selection_requires_complete_report_family() -> None:
    with pytest.raises(ModelError, match="exact K=2,3,4,5"):
        select_owner_fixed_k4(
            (
                _candidate(2, 90.0, 5.0),
                _candidate(4, 120.0, 1.0),
            )
        )


def test_predictive_tolerance_and_inclusive_bic_guard_choose_smaller_k(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    result = select_candidate(
        (
            _candidate(4, 100.0, 0.0),
            _candidate(3, 111.0, 5e-13),
            _candidate(2, 110.0, 0.0),
        ),
        bootstrap_indices=bootstrap_indices,
    )

    assert result.predictive_best_n_states == 2
    assert result.selected_n_states == 2
    rows = {row.n_states: row for row in result.candidates}
    assert rows[2].delta_bic == 10.0
    assert rows[2].inside_bic_guard
    assert not rows[3].inside_bic_guard
    assert rows[3].disposition == "outside_bic_guard"
    assert [row.bic_rank for row in result.candidates] == [2, 3, 1]


def test_higher_score_is_predictive_best(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    result = select_candidate(
        (
            _candidate(2, 100.0, 0.0),
            _candidate(3, 101.0, 1.0),
        ),
        bootstrap_indices=bootstrap_indices,
    )
    assert result.predictive_best_n_states == 3
    assert result.selected_n_states == 3


def test_one_eligible_candidate_skips_bootstrap_but_still_applies_bic_guard() -> None:
    result = select_candidate(
        (
            _candidate(2, 100.0, _scores_with_invalid_fold()),
            _candidate(3, 111.0, 1.0),
        )
    )

    assert not result.passed
    assert result.predictive_best_n_states == 3
    assert result.bootstrap_replicates == 0
    assert result.max_z_critical_value == 0.0
    assert result.failure_reason == "no_candidate_in_near_best_bic_intersection"
    rows = {row.n_states: row for row in result.candidates}
    assert rows[2].disposition == "predictively_ineligible"
    assert rows[2].invalid_rolling_folds == (0,)
    assert rows[3].disposition == "outside_bic_guard"


def test_zero_predictively_eligible_candidates_returns_failure() -> None:
    result = select_candidate(
        (
            _candidate(2, 100.0, _scores_with_invalid_fold(0)),
            _candidate(3, 101.0, _scores_with_invalid_fold(5)),
        )
    )
    assert not result.passed
    assert result.predictive_best_n_states is None
    assert result.failure_reason == "no_predictively_eligible_candidates"


def test_inadmissible_candidate_is_retained_but_excluded_from_bic() -> None:
    failed = evaluate_candidate_viability(
        _evidence(2, bic=1.0, checks=(PreselectionCheck("fit_gate", False),)),
        thresholds=_thresholds(),
    )
    result = select_candidate(
        (
            CandidateSelectionInput(failed, ()),
            _candidate(3, 100.0, 0.0),
        )
    )
    assert result.selected_n_states == 3
    rejected = result.candidates[0]
    assert rejected.disposition == "inadmissible"
    assert rejected.bic_rank is None
    assert rejected.candidate_minus_best_mean is None
    assert rejected.rejection_reasons[0].check_id == "fit_gate"
    serialized_viability = rejected.as_dict()["viability"]
    assert isinstance(serialized_viability, dict)
    assert serialized_viability["admissible"] is False


def test_no_pre_bic_viable_candidates_returns_fail_closed_report() -> None:
    failed = evaluate_candidate_viability(
        _evidence(2, bic=None), thresholds=_thresholds()
    )
    result = select_candidate((CandidateSelectionInput(failed, ()),))
    assert not result.passed
    assert result.selected_n_states is None
    assert result.minimum_bic_n_states is None
    assert result.rolling_fold_count == 6
    assert result.failure_reason == "no_pre_bic_viable_candidates"


def test_selection_contract_fails_closed(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    with pytest.raises(ModelError, match="at least one"):
        select_candidate(())
    with pytest.raises(ModelError, match="unique"):
        select_candidate((_candidate(2, 1.0, 0.0), _candidate(2, 2.0, 0.0)))
    with pytest.raises(ModelError, match="one of 2, 3, 4, or 5"):
        select_candidate((_candidate(6, 1.0, 0.0),))
    with pytest.raises(ModelError, match="exactly 72"):
        select_candidate((_candidate(2, 1.0, (0.0,) * 71),))
    partial_fold: list[float | None] = list((0.0,) * 72)
    partial_fold[0] = None
    with pytest.raises(ModelError, match="exactly twelve None"):
        select_candidate((_candidate(2, 1.0, tuple(partial_fold)),))
    nonfinite = list((0.0,) * 72)
    nonfinite[0] = float("inf")
    with pytest.raises(ModelError, match="finite when present"):
        select_candidate((_candidate(2, 1.0, tuple(nonfinite)),))
    with pytest.raises(ModelError, match="precomputed bootstrap indices"):
        select_candidate((_candidate(2, 1.0, 0.0), _candidate(3, 2.0, 0.0)))
    with pytest.raises(ModelError, match=r"shape \(10000, 6, 12\)"):
        select_candidate(
            (_candidate(2, 1.0, 0.0), _candidate(3, 2.0, 0.0)),
            bootstrap_indices=bootstrap_indices[:-1],
        )


def test_holdout_veto_equality_passes_and_serializes() -> None:
    summary = summarize_holdout_veto(
        hmm_log_score=-10.0,
        baseline_log_score=-10.0,
        sequence_lengths=(17, 7),
        minimum_difference_per_observation=0.0,
    )
    assert summary.passed
    assert summary.observations == 24
    assert summary.difference_per_observation == 0.0
    assert summary.rejection_reasons == ()
    assert summary.as_dict()["sequence_lengths"] == [17, 7]


def test_failed_holdout_withholds_authorization_without_promoting_runner_up(
    bootstrap_indices: NDArray[np.int64],
) -> None:
    selection = select_candidate(
        (
            _candidate(2, 101.0, 0.0),
            _candidate(3, 100.0, 1.0),
        ),
        bootstrap_indices=bootstrap_indices,
    )
    assert selection.selected_n_states == 3
    holdout = summarize_holdout_veto(
        hmm_log_score=-25.0,
        baseline_log_score=-24.0,
        sequence_lengths=(17, 7),
        minimum_difference_per_observation=0.0,
    )

    result = apply_holdout_veto(selection, holdout)

    assert not result.passed
    assert result.selected_n_states == 3
    assert result.authorized_n_states is None
    assert result.holdout.rejection_reasons == ("holdout_score_below_minimum",)
    assert result.as_dict()["authorized_n_states"] is None


def test_passed_holdout_authorizes_the_preselected_candidate() -> None:
    selection = select_candidate((_candidate(2, 100.0, 0.0),))
    holdout = summarize_holdout_veto(
        hmm_log_score=-9.0,
        baseline_log_score=-10.0,
        sequence_lengths=(2,),
        minimum_difference_per_observation=0.5,
    )
    result = apply_holdout_veto(selection, holdout)
    assert result.passed
    assert result.authorized_n_states == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "hmm_log_score": float("nan"),
                "baseline_log_score": 0.0,
                "sequence_lengths": (1,),
                "minimum_difference_per_observation": 0.0,
            },
            "log scores",
        ),
        (
            {
                "hmm_log_score": 0.0,
                "baseline_log_score": 0.0,
                "sequence_lengths": (),
                "minimum_difference_per_observation": 0.0,
            },
            "non-empty",
        ),
        (
            {
                "hmm_log_score": 0.0,
                "baseline_log_score": 0.0,
                "sequence_lengths": (0,),
                "minimum_difference_per_observation": 0.0,
            },
            "positive integer",
        ),
        (
            {
                "hmm_log_score": 0.0,
                "baseline_log_score": 0.0,
                "sequence_lengths": (1,),
                "minimum_difference_per_observation": float("inf"),
            },
            "threshold",
        ),
        (
            {
                "hmm_log_score": 1e308,
                "baseline_log_score": -1e308,
                "sequence_lengths": (1,),
                "minimum_difference_per_observation": 0.0,
            },
            "non-finite",
        ),
    ],
)
def test_holdout_contract_fails_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        summarize_holdout_veto(**kwargs)  # type: ignore[arg-type]


def test_holdout_cannot_be_applied_after_failed_selection() -> None:
    failed = evaluate_candidate_viability(
        _evidence(2, bic=None), thresholds=_thresholds()
    )
    selection = select_candidate(
        (CandidateSelectionInput(failed, ()),),
    )
    holdout = summarize_holdout_veto(
        hmm_log_score=0.0,
        baseline_log_score=0.0,
        sequence_lengths=(1,),
        minimum_difference_per_observation=0.0,
    )
    with pytest.raises(ModelError, match="successful candidate selection"):
        apply_holdout_veto(selection, holdout)
