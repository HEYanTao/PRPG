from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prpg.errors import ModelError
from prpg.model.stability import (
    align_state_labels,
    aligned_state_jaccards,
    canonical_rolling_folds,
    map_monthly_states_to_daily,
    rolling_origin_splits,
    sequence_lengths,
    simultaneous_predictive_envelope,
    summarize_macro_bootstrap_stability,
    summarize_rolling_stability,
)


def test_sequence_lengths_preserves_every_observation() -> None:
    mask = np.array([False, True, True, False, True, False])

    assert sequence_lengths(mask) == (3, 2, 1)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([], dtype=bool),
        np.array([True, False]),
        np.array([0, 1]),
        np.array([[False]]),
    ],
)
def test_sequence_lengths_rejects_invalid_contract(mask: np.ndarray) -> None:
    with pytest.raises(ModelError):
        sequence_lengths(mask)


def test_hungarian_alignment_recovers_permutation_and_ari() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2, 2])
    candidate = np.array([2, 2, 0, 0, 1, 1, 1])

    result = align_state_labels(reference, candidate, 3)

    assert result.candidate_to_reference.tolist() == [1, 2, 0]
    assert np.array_equal(result.aligned_states, reference)
    assert result.adjusted_rand_index == 1.0
    assert result.contingency.sum() == len(reference)


def test_aligned_jaccards_report_each_state_and_missing_state_as_zero() -> None:
    reference = np.asarray([0, 0, 1, 1, 2, 2])
    candidate = np.asarray([0, 1, 1, 1, 0, 0])

    result = aligned_state_jaccards(reference, candidate, 3)

    np.testing.assert_allclose(result, [0.25, 2 / 3, 0.0])
    assert not result.flags.writeable


@pytest.mark.parametrize(
    ("reference", "candidate", "states"),
    [
        (np.array([0]), np.array([0, 0]), 1),
        (np.array([], dtype=int), np.array([], dtype=int), 1),
        (np.array([0.0]), np.array([0]), 1),
        (np.array([2]), np.array([0]), 2),
        (np.array([0]), np.array([0]), 0),
    ],
)
def test_hungarian_alignment_rejects_invalid_inputs(
    reference: np.ndarray, candidate: np.ndarray, states: int
) -> None:
    with pytest.raises(ModelError):
        align_state_labels(reference, candidate, states)


def test_rolling_splits_are_expanding_and_final() -> None:
    splits = rolling_origin_splits(
        100, folds=3, validation_size=10, minimum_training_size=50
    )

    assert splits == (
        (slice(0, 70), slice(70, 80)),
        (slice(0, 80), slice(80, 90)),
        (slice(0, 90), slice(90, 100)),
    )


def test_canonical_rolling_folds_freeze_design_and_production_geometry() -> None:
    design = np.ones(193, dtype=np.bool_)
    design[0] = False
    design_folds = canonical_rolling_folds(design)
    assert [fold.training_rows for fold in design_folds] == [
        121,
        133,
        145,
        157,
        169,
        181,
    ]
    assert all(fold.validation_lengths == (12,) for fold in design_folds)

    production = np.ones(217, dtype=np.bool_)
    production[[0, 210]] = False
    production_folds = canonical_rolling_folds(production)
    assert [fold.training_rows for fold in production_folds] == [
        145,
        157,
        169,
        181,
        193,
        205,
    ]
    assert production_folds[-1].validation == slice(205, 217)
    assert production_folds[-1].validation_lengths == (5, 7)


def test_macro_stability_failure_accounting_and_boundaries() -> None:
    success = np.ones(100, dtype=np.bool_)
    success[:10] = False
    ari = np.full(100, 0.80)
    ari[10] = 0.50
    jaccards = np.full((100, 2), 0.70)
    jaccards[10] = 0.50

    result = summarize_macro_bootstrap_stability(success, ari, jaccards)

    assert result.successful_attempts == 90
    assert result.median_ari == pytest.approx(0.80)
    assert result.tenth_percentile_ari == pytest.approx(0.50)
    np.testing.assert_allclose(result.tenth_percentile_jaccard_by_state, 0.50)
    assert result.passed
    failed = summarize_macro_bootstrap_stability(
        np.r_[np.zeros(11, dtype=np.bool_), np.ones(89, dtype=np.bool_)],
        ari,
        jaccards,
    )
    assert not failed.passed


def test_rolling_stability_uses_distinct_success_and_jaccard_gates() -> None:
    success = np.asarray([False, True, True, True, True, True])
    ari = np.asarray([1.0, 0.40, 0.70, 0.70, 0.80, 0.90])
    jaccards = np.full((6, 2), 0.60)

    result = summarize_rolling_stability(success, ari, jaccards)

    assert result.successful_fits == 5
    assert result.median_ari == pytest.approx(0.70)
    assert result.minimum_successful_ari == pytest.approx(0.40)
    np.testing.assert_allclose(result.median_jaccard_by_state, 0.60)
    assert result.passed
    too_few = summarize_rolling_stability(
        np.asarray([False, False, True, True, True, True]), ari, jaccards
    )
    assert not too_few.passed


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "observations": 10,
            "folds": 2,
            "validation_size": 3,
            "minimum_training_size": 5,
        },
        {
            "observations": 10,
            "folds": 0,
            "validation_size": 1,
            "minimum_training_size": 1,
        },
        {
            "observations": True,
            "folds": 1,
            "validation_size": 1,
            "minimum_training_size": 1,
        },
    ],
)
def test_rolling_splits_fail_closed(kwargs: dict[str, int]) -> None:
    with pytest.raises(ModelError):
        rolling_origin_splits(**kwargs)


def test_daily_state_mapping_uses_return_ending_month() -> None:
    months = pd.period_range("2024-01", periods=3, freq="M")
    sessions = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-03-28"])

    result = map_monthly_states_to_daily(
        months.asi8, np.array([2, 0, 1]), sessions.asi8
    )

    assert result.tolist() == [2, 0, 1]


def test_daily_state_mapping_rejects_gap_and_bad_shapes() -> None:
    months = pd.period_range("2024-01", periods=2, freq="M")
    with pytest.raises(ModelError, match="without decoded states"):
        map_monthly_states_to_daily(
            months.asi8,
            np.array([0, 1]),
            pd.DatetimeIndex(["2024-03-01"]).asi8,
        )
    with pytest.raises(ModelError):
        map_monthly_states_to_daily(
            months.asi8[::-1], np.array([0, 1]), pd.DatetimeIndex(["2024-01-02"]).asi8
        )
    with pytest.raises(ModelError):
        map_monthly_states_to_daily(
            months.asi8, np.array([0]), pd.DatetimeIndex(["2024-01-02"]).asi8
        )


def test_simultaneous_envelope_uses_max_studentized_family() -> None:
    replicates = np.array(
        [
            [-1.0, 5.0, 2.0],
            [0.0, 5.0, 3.0],
            [1.0, 5.0, 4.0],
            [0.5, 5.0, 3.5],
        ]
    )

    passed = simultaneous_predictive_envelope(
        replicates, np.array([0.0, 5.0, 3.0]), confidence=0.75
    )
    failed = simultaneous_predictive_envelope(
        replicates, np.array([100.0, 5.1, 3.0]), confidence=0.75
    )

    assert passed.passed
    assert passed.component_passed.tolist() == [True, True, True]
    assert not failed.passed
    assert failed.component_passed.tolist() == [False, False, True]
    assert passed.repetitions == 4


@pytest.mark.parametrize(
    ("replicates", "observed", "confidence", "tolerance"),
    [
        (np.ones((1, 2)), np.ones(2), 0.95, 1e-12),
        (np.ones((2, 2)), np.ones(3), 0.95, 1e-12),
        (np.array([[1.0], [np.nan]]), np.ones(1), 0.95, 1e-12),
        (np.ones((2, 1)), np.ones(1), 1.0, 1e-12),
        (np.ones((2, 1)), np.ones(1), 0.95, -1.0),
    ],
)
def test_simultaneous_envelope_rejects_invalid_inputs(
    replicates: np.ndarray,
    observed: np.ndarray,
    confidence: float,
    tolerance: float,
) -> None:
    with pytest.raises(ModelError):
        simultaneous_predictive_envelope(
            replicates,
            observed,
            confidence=confidence,
            constant_tolerance=tolerance,
        )
