"""Tests for version-1 alpha-policy arithmetic and streaming statistics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.stats import t as student_t

from prpg.errors import ModelError
from prpg.model.alpha import (
    AlphaPolicy,
    KernelMeanStatistics,
    KernelWindowStatistics,
    alpha_offset_grid,
    apply_alpha_policy,
    finalize_kernel_mean_precision,
    finalize_kernel_means,
    historical_target_mean,
    kernel_mean_statistics,
    make_return_vector_batch,
    merge_kernel_mean_statistics,
    merge_kernel_window_statistics,
    select_smallest_qualified_lambda,
)
from prpg.model.alpha import (
    kernel_window_statistics as _kernel_window_statistics_impl,
)

_MODEL_FP = "a" * 64
_STREAM_FP = "b" * 64
_TARGET_FP = "c" * 64


def _returns() -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(
        [
            [0.01, 0.001, 0.002],
            [0.03, 0.003, 0.006],
            [-0.02, 0.002, 0.004],
            [0.00, 0.004, 0.008],
        ],
        dtype=np.float64,
    )


def kernel_window_statistics(
    sampled_return_windows: Any,
    target_state_windows: Any,
    *,
    n_states: int,
    first_window_index: int,
    frequency: str = "monthly",
    model_artifact_fingerprint: str = _MODEL_FP,
    calibration_stream_fingerprint: str = _STREAM_FP,
) -> KernelWindowStatistics:
    """Test helper supplying explicit synthetic provenance."""

    return _kernel_window_statistics_impl(
        sampled_return_windows,
        target_state_windows,
        n_states=n_states,
        first_window_index=first_window_index,
        frequency=frequency,  # type: ignore[arg-type]
        model_artifact_fingerprint=model_artifact_fingerprint,
        calibration_stream_fingerprint=calibration_stream_fingerprint,
    )


def _kernel_precision(
    means: Any = ((0.01, 0.001, 0.002), (-0.01, 0.002, 0.003)),
    *,
    frequency: str = "monthly",
    model_fingerprint: str = _MODEL_FP,
) -> Any:
    values = np.asarray(means, dtype=np.float64)
    returns = np.stack((values, values), axis=0)
    states = np.tile(np.arange(values.shape[0], dtype=np.int64), (2, 1))
    statistics = kernel_window_statistics(
        returns,
        states,
        n_states=values.shape[0],
        first_window_index=0,
        frequency=frequency,
        model_artifact_fingerprint=model_fingerprint,
    )
    return finalize_kernel_mean_precision(
        statistics,
        periods_per_year=12 if frequency == "monthly" else 252,
        confidence=0.95,
    )


def test_statistics_and_final_estimates_match_direct_grouped_math() -> None:
    values = _returns()
    result = finalize_kernel_means(
        kernel_mean_statistics(values, [0, 0, 1, 1], n_states=2)
    )

    np.testing.assert_array_equal(result.counts, [2, 2])
    np.testing.assert_allclose(
        result.means, [[0.02, 0.002, 0.004], [-0.01, 0.003, 0.006]]
    )
    np.testing.assert_allclose(
        result.sample_variances,
        np.asarray([values[:2].var(axis=0, ddof=1), values[2:].var(axis=0, ddof=1)]),
    )
    np.testing.assert_allclose(
        result.independent_vector_standard_errors,
        np.sqrt(result.sample_variances / 2.0),
    )


def test_batch_statistics_can_omit_a_state_and_merge_exactly() -> None:
    values = _returns()
    first = kernel_mean_statistics(values[:2], [0, 0], n_states=2)
    second = kernel_mean_statistics(values[2:], [1, 1], n_states=2)

    merged = finalize_kernel_means(merge_kernel_mean_statistics([first, second]))
    direct = finalize_kernel_means(
        kernel_mean_statistics(values, [0, 0, 1, 1], n_states=2)
    )

    np.testing.assert_array_equal(merged.counts, direct.counts)
    np.testing.assert_allclose(merged.means, direct.means)
    np.testing.assert_allclose(merged.sample_variances, direct.sample_variances)


def test_chan_merge_handles_split_observations_within_each_state() -> None:
    values = np.vstack([_returns(), _returns() * 2.0])
    states = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    batches = [
        kernel_mean_statistics(values[:4], states[:4], n_states=2),
        kernel_mean_statistics(values[4:], states[4:], n_states=2),
    ]

    merged = finalize_kernel_means(merge_kernel_mean_statistics(batches))
    direct = finalize_kernel_means(kernel_mean_statistics(values, states, n_states=2))

    np.testing.assert_allclose(merged.means, direct.means, rtol=0.0, atol=1e-17)
    np.testing.assert_allclose(
        merged.sample_variances, direct.sample_variances, rtol=0.0, atol=1e-18
    )


def test_statistics_and_estimates_are_read_only_and_do_not_mutate_inputs() -> None:
    values = _returns()
    before = values.copy()
    stats = kernel_mean_statistics(values, [0, 0, 1, 1], n_states=2)
    estimate = finalize_kernel_means(stats)

    np.testing.assert_array_equal(values, before)
    for array in (
        stats.counts,
        stats.means,
        stats.centered_sum_squares,
        estimate.counts,
        estimate.means,
        estimate.sample_variances,
        estimate.independent_vector_standard_errors,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


def test_finalize_rejects_states_with_fewer_than_two_observations() -> None:
    stats = kernel_mean_statistics(_returns()[:2], [0, 1], n_states=3)

    with pytest.raises(ModelError, match="at least two") as captured:
        finalize_kernel_means(stats)

    assert captured.value.details == {"states": [0, 1, 2], "counts": [1, 1, 0]}


def _window_fixture() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    returns = np.asarray(
        [
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
            [[2.0, 1.0, 4.0], [4.0, 2.0, 8.0], [6.0, 3.0, 12.0]],
            [[3.0, 3.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[0, 0, 1], [0, 1, 1], [0, 1, 1]], dtype=np.int64)
    return returns, states


def test_window_cluster_precision_matches_ratio_score_formula() -> None:
    returns, states = _window_fixture()
    statistics = kernel_window_statistics(
        returns, states, n_states=2, first_window_index=0
    )
    result = finalize_kernel_mean_precision(
        statistics,
        periods_per_year=12,
        confidence=0.95,
    )

    expected_counts = np.asarray([4, 5])
    expected_means = np.vstack(
        [returns[states == state].mean(axis=0) for state in range(2)]
    )
    window_sums = np.zeros((3, 2, 3))
    window_counts = np.zeros((3, 2))
    for window in range(3):
        for state in range(2):
            selected = returns[window][states[window] == state]
            window_counts[window, state] = len(selected)
            window_sums[window, state] = selected.sum(axis=0)
    scores = window_sums - window_counts[:, :, None] * expected_means
    expected_se = (
        np.sqrt(3.0 / 2.0 * np.sum(scores * scores, axis=0)) / expected_counts[:, None]
    )

    assert result.windows == 3
    assert result.frequency == "monthly"
    assert result.model_artifact_fingerprint == _MODEL_FP
    assert result.calibration_stream_fingerprint == _STREAM_FP
    assert result.sample_stage == "unadjusted_kernel"
    assert result.periods_per_year == 12
    assert result.confidence == pytest.approx(0.95)
    assert len(result.kernel_sample_fingerprint) == 64
    assert result.degrees_freedom == 2
    assert result.simultaneous_comparisons == 72
    assert result.critical_value == pytest.approx(
        student_t.ppf(1.0 - 0.05 / (2.0 * 72.0), df=2)
    )
    np.testing.assert_array_equal(result.state_window_counts, [3, 3])
    np.testing.assert_array_equal(result.state_observation_counts, expected_counts)
    np.testing.assert_allclose(result.means, expected_means)
    np.testing.assert_allclose(result.cluster_standard_errors, expected_se)
    np.testing.assert_allclose(result.annualized_means, expected_means * 12.0)
    np.testing.assert_allclose(
        result.annualized_simultaneous_half_widths,
        result.critical_value * expected_se * 12.0,
    )
    for array in (
        result.state_window_counts,
        result.state_observation_counts,
        result.means,
        result.cluster_standard_errors,
        result.annualized_means,
        result.annualized_simultaneous_half_widths,
    ):
        assert not array.flags.writeable


def test_window_statistics_merge_matches_direct_summary() -> None:
    returns, states = _window_fixture()
    first = kernel_window_statistics(
        returns[:1], states[:1], n_states=2, first_window_index=0
    )
    second = kernel_window_statistics(
        returns[1:], states[1:], n_states=2, first_window_index=1
    )

    merged = merge_kernel_window_statistics([first, second])
    direct = kernel_window_statistics(returns, states, n_states=2, first_window_index=0)

    assert merged.windows == direct.windows
    for name in (
        "state_window_counts",
        "state_observation_counts",
        "state_return_sums",
        "cluster_return_sum_squares",
        "cluster_count_return_products",
        "cluster_count_squares",
    ):
        np.testing.assert_allclose(getattr(merged, name), getattr(direct, name))
        assert not getattr(merged, name).flags.writeable


def test_window_merge_rejects_duplicate_gap_and_out_of_order_ranges() -> None:
    returns, states = _window_fixture()
    first = kernel_window_statistics(
        returns[:1], states[:1], n_states=2, first_window_index=0
    )
    duplicate = kernel_window_statistics(
        returns[:1], states[:1], n_states=2, first_window_index=0
    )
    gap = kernel_window_statistics(
        returns[1:], states[1:], n_states=2, first_window_index=2
    )
    nonzero_start = kernel_window_statistics(
        returns[1:], states[1:], n_states=2, first_window_index=1
    )

    with pytest.raises(ModelError, match="contiguous, ordered, and disjoint"):
        merge_kernel_window_statistics([first, duplicate])
    with pytest.raises(ModelError, match="contiguous, ordered, and disjoint"):
        merge_kernel_window_statistics([first, gap])
    with pytest.raises(ModelError, match="begin with window index zero"):
        merge_kernel_window_statistics([nonzero_start])


@pytest.mark.parametrize(
    ("identity_change", "value"),
    [
        ("frequency", "daily"),
        ("model_artifact_fingerprint", "d" * 64),
        ("calibration_stream_fingerprint", "e" * 64),
    ],
)
def test_window_merge_rejects_cross_identity_batches(
    identity_change: str, value: str
) -> None:
    returns, states = _window_fixture()
    first = kernel_window_statistics(
        returns[:1], states[:1], n_states=2, first_window_index=0
    )
    kwargs = {
        "frequency": "monthly",
        "model_artifact_fingerprint": _MODEL_FP,
        "calibration_stream_fingerprint": _STREAM_FP,
    }
    kwargs[identity_change] = value
    second = kernel_window_statistics(
        returns[1:],
        states[1:],
        n_states=2,
        first_window_index=1,
        **kwargs,  # type: ignore[arg-type]
    )

    with pytest.raises(ModelError, match="different frequency, artifact, or stream"):
        merge_kernel_window_statistics([first, second])


def test_window_batch_fingerprint_binds_range_and_values() -> None:
    returns, states = _window_fixture()
    original = kernel_window_statistics(
        returns, states, n_states=2, first_window_index=0
    )
    changed = returns.copy()
    changed[0, 0, 0] += 1e-12
    changed_batch = kernel_window_statistics(
        changed, states, n_states=2, first_window_index=0
    )
    shifted = kernel_window_statistics(
        returns, states, n_states=2, first_window_index=1
    )

    assert len(original.window_fingerprint) == 64
    assert len(original.source_window_fingerprint) == 64
    assert original.window_fingerprint != changed_batch.window_fingerprint
    assert original.window_fingerprint != shifted.window_fingerprint


@pytest.mark.parametrize(
    ("returns", "states", "n_states", "message"),
    [
        ([], [], 2, "shape"),
        (np.zeros((1, 2, 2)), np.zeros((1, 2), dtype=int), 2, "shape"),
        (np.zeros((1, 2, 3)), np.zeros((2, 1), dtype=int), 2, "geometry"),
        (np.zeros((1, 2, 3)), np.zeros((1, 2)), 2, "integer dtype"),
        (np.zeros((1, 2, 3)), np.asarray([[0, -1]]), 2, "negative"),
        (np.zeros((1, 2, 3)), np.asarray([[0, 2]]), 2, "outside"),
        (np.full((1, 2, 3), np.nan), np.zeros((1, 2), dtype=int), 2, "finite"),
    ],
)
def test_window_statistics_input_contract(
    returns: Any, states: Any, n_states: Any, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        kernel_window_statistics(
            returns, states, n_states=n_states, first_window_index=0
        )


def test_window_merge_rejects_absent_wrong_and_state_mismatched_batches() -> None:
    returns, states = _window_fixture()
    two = kernel_window_statistics(returns, states, n_states=2, first_window_index=0)
    three = kernel_window_statistics(returns, states, n_states=3, first_window_index=0)

    with pytest.raises(ModelError, match="at least one"):
        merge_kernel_window_statistics([])
    with pytest.raises(ModelError, match="wrong type"):
        merge_kernel_window_statistics([object()])  # type: ignore[list-item]
    with pytest.raises(ModelError, match="different state counts"):
        merge_kernel_window_statistics([two, three])


def test_window_precision_requires_two_windows_per_state() -> None:
    returns, states = _window_fixture()
    one_window = kernel_window_statistics(
        returns[:1], states[:1], n_states=2, first_window_index=0
    )

    with pytest.raises(ModelError, match="at least two windows"):
        finalize_kernel_mean_precision(
            one_window,
            periods_per_year=12,
            confidence=0.95,
        )

    sparse = kernel_window_statistics(
        returns[:2],
        np.asarray([[0, 0, 1], [0, 0, 0]]),
        n_states=2,
        first_window_index=0,
    )
    with pytest.raises(ModelError, match="two independent") as captured:
        finalize_kernel_mean_precision(
            sparse,
            periods_per_year=12,
            confidence=0.95,
        )
    assert captured.value.details == {"states": [1], "window_counts": [1]}


@pytest.mark.parametrize(
    ("periods", "confidence", "message"),
    [
        (0, 0.95, "periods_per_year"),
        (252, 0.95, "does not match"),
        (12, 1, "confidence"),
        (12, True, "confidence"),
        (12, np.nan, "confidence"),
    ],
)
def test_window_precision_parameter_contract(
    periods: Any, confidence: Any, message: str
) -> None:
    returns, states = _window_fixture()
    statistics = kernel_window_statistics(
        returns, states, n_states=2, first_window_index=0
    )
    with pytest.raises(ModelError, match=message):
        finalize_kernel_mean_precision(
            statistics,
            periods_per_year=periods,
            confidence=confidence,
        )


def _window_record(**changes: Any) -> KernelWindowStatistics:
    result = kernel_window_statistics(
        np.zeros((1, 1, 3)),
        np.zeros((1, 1), dtype=np.int64),
        n_states=1,
        first_window_index=0,
    )
    return replace(result, **changes)


@pytest.mark.parametrize(
    "invalid",
    [
        _window_record(windows=0, stop_window_index=0),
        _window_record(windows=True),
        _window_record(stop_window_index=2),
        _window_record(window_fingerprint="INVALID"),
        _window_record(state_window_counts=np.asarray([2])),
        _window_record(state_window_counts=np.asarray([1.0])),
        _window_record(state_return_sums=np.zeros((1, 2))),
        _window_record(cluster_return_sum_squares=np.full((1, 3), np.nan)),
        _window_record(
            state_window_counts=np.asarray([0]),
            state_observation_counts=np.asarray([1]),
        ),
        _window_record(
            state_window_counts=np.asarray([0]),
            state_observation_counts=np.asarray([0]),
            state_return_sums=np.ones((1, 3)),
            cluster_count_squares=np.zeros(1),
        ),
        KernelWindowStatistics(
            frequency="monthly",
            model_artifact_fingerprint=_MODEL_FP,
            calibration_stream_fingerprint=_STREAM_FP,
            windows=2,
            first_window_index=0,
            stop_window_index=2,
            source_window_fingerprint="d" * 64,
            window_fingerprint="b" * 64,
            state_window_counts=np.asarray([2]),
            state_observation_counts=np.asarray([2]),
            state_return_sums=np.full((1, 3), 100.0),
            cluster_return_sum_squares=np.ones((1, 3)),
            cluster_count_return_products=np.full((1, 3), 50.01),
            cluster_count_squares=np.asarray([2.0]),
        ),
    ],
)
def test_window_statistics_object_validation_fails_closed(
    invalid: KernelWindowStatistics,
) -> None:
    with pytest.raises(ModelError):
        merge_kernel_window_statistics([invalid])


def test_window_validator_normalizes_list_backed_public_records() -> None:
    list_backed = _window_record(
        state_window_counts=[1],  # type: ignore[arg-type]
        state_observation_counts=[1],  # type: ignore[arg-type]
        state_return_sums=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        cluster_return_sum_squares=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        cluster_count_return_products=[[0.0, 0.0, 0.0]],  # type: ignore[arg-type]
        cluster_count_squares=[1.0],  # type: ignore[arg-type]
    )

    normalized = merge_kernel_window_statistics([list_backed])

    assert isinstance(normalized.state_window_counts, np.ndarray)
    assert not normalized.state_window_counts.flags.writeable


@pytest.mark.parametrize(
    ("returns", "states", "n_states", "message"),
    [
        ([], [], 2, "shape"),
        ([[1.0, 2.0]], [0], 2, "shape"),
        ([[1.0, 2.0, np.nan]], [0], 2, "finite"),
        (_returns(), [0], 2, "match"),
        (_returns(), [0.0] * 4, 2, "integer dtype"),
        (_returns(), [0, 0, 1, -1], 2, "negative"),
        (_returns(), [0, 0, 1, 2], 2, "outside"),
        (_returns(), [0, 0, 1, 1], 0, "positive integer"),
        (_returns(), [0, 0, 1, 1], True, "positive integer"),
    ],
)
def test_statistics_input_contract(
    returns: Any, states: Any, n_states: Any, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        kernel_mean_statistics(returns, states, n_states=n_states)


def test_merge_rejects_absent_wrong_or_inconsistent_batches() -> None:
    valid = kernel_mean_statistics(_returns(), [0, 0, 1, 1], n_states=2)
    three_states = kernel_mean_statistics(_returns(), [0, 0, 1, 1], n_states=3)

    with pytest.raises(ModelError, match="at least one"):
        merge_kernel_mean_statistics([])
    with pytest.raises(ModelError, match="wrong type"):
        merge_kernel_mean_statistics([object()])  # type: ignore[list-item]
    with pytest.raises(ModelError, match="different state counts"):
        merge_kernel_mean_statistics([valid, three_states])


@pytest.mark.parametrize(
    "invalid",
    [
        KernelMeanStatistics(np.asarray([1.0]), np.zeros((1, 3)), np.zeros((1, 3)), 1),
        KernelMeanStatistics(
            np.asarray([], dtype=int), np.zeros((0, 3)), np.zeros((0, 3)), 0
        ),
        KernelMeanStatistics(np.asarray([-1]), np.zeros((1, 3)), np.zeros((1, 3)), -1),
        KernelMeanStatistics(np.asarray([1]), np.zeros((1, 2)), np.zeros((1, 3)), 1),
        KernelMeanStatistics(
            np.asarray([1]), np.full((1, 3), np.nan), np.zeros((1, 3)), 1
        ),
        KernelMeanStatistics(
            np.asarray([1]), np.zeros((1, 3)), np.full((1, 3), -1.0), 1
        ),
        KernelMeanStatistics(np.asarray([0]), np.ones((1, 3)), np.zeros((1, 3)), 0),
        KernelMeanStatistics(np.asarray([1]), np.zeros((1, 3)), np.zeros((1, 3)), 2),
        KernelMeanStatistics(np.asarray([1]), np.zeros((1, 3)), np.zeros((1, 3)), True),
    ],
)
def test_statistics_object_validation_fails_closed(
    invalid: KernelMeanStatistics,
) -> None:
    with pytest.raises(ModelError):
        merge_kernel_mean_statistics([invalid])


def test_historical_target_mean_uses_all_synchronized_vectors() -> None:
    mean = historical_target_mean(_returns())

    np.testing.assert_allclose(mean, np.mean(_returns(), axis=0))
    assert not mean.flags.writeable


@pytest.mark.parametrize("invalid", [[], [[1.0, 2.0]], [[1.0, 2.0, np.inf]]])
def test_historical_target_mean_validates_shape_and_finiteness(invalid: Any) -> None:
    with pytest.raises(ModelError):
        historical_target_mean(invalid)


def test_offset_grid_uses_exact_documented_formula() -> None:
    kernel = np.asarray([[0.04, 0.01, 0.02], [0.00, 0.03, 0.04]])
    target = np.asarray([0.02, 0.02, 0.03])
    precision = _kernel_precision(kernel)
    grid = alpha_offset_grid(
        precision,
        target,
        [0.25, 0.5, 1.0],
        target_history_fingerprint=_TARGET_FP,
    )

    expected = np.asarray([0.25, 0.5, 1.0])[:, None, None] * (kernel - target)[None]
    np.testing.assert_allclose(grid.offsets, expected)
    assert grid.frequency == "monthly"
    assert grid.model_artifact_fingerprint == _MODEL_FP
    assert grid.kernel_sample_fingerprint == precision.kernel_sample_fingerprint
    assert len(grid.grid_fingerprint) == 64
    assert not grid.lambdas.flags.writeable
    assert not grid.offsets.flags.writeable


@pytest.mark.parametrize(
    ("target", "lambdas", "message"),
    [
        ([0.0, 0.0], [0.5], "exactly"),
        ([0.0] * 3, [], "non-empty"),
        ([0.0] * 3, [0.0], "lie in"),
        ([0.0] * 3, [1.1], "lie in"),
        ([0.0] * 3, [0.5, 0.5], "strictly increasing"),
        ([0.0] * 3, [np.nan], "finite"),
    ],
)
def test_offset_grid_input_contract(target: Any, lambdas: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        alpha_offset_grid(
            _kernel_precision(),
            target,
            lambdas,
            target_history_fingerprint=_TARGET_FP,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frequency", "weekly", "base frequency"),
        ("model_artifact_fingerprint", "bad", "model_artifact"),
        ("kernel_sample_fingerprint", "bad", "kernel_sample"),
        ("sample_stage", "adjusted", "unadjusted"),
    ],
)
def test_offset_grid_rejects_tampered_precision_provenance(
    field: str, value: str, message: str
) -> None:
    precision = replace(_kernel_precision(), **{field: value})
    with pytest.raises(ModelError, match=message):
        alpha_offset_grid(
            precision,
            np.zeros(3),
            [0.5],
            target_history_fingerprint=_TARGET_FP,
        )


def test_offset_grid_rejects_wrong_precision_type_and_history_fingerprint() -> None:
    with pytest.raises(ModelError, match="wrong type"):
        alpha_offset_grid(  # type: ignore[arg-type]
            object(), np.zeros(3), [0.5], target_history_fingerprint=_TARGET_FP
        )
    with pytest.raises(ModelError, match="target_history"):
        alpha_offset_grid(
            _kernel_precision(), np.zeros(3), [0.5], target_history_fingerprint="bad"
        )


def _offset_grid(
    *, frequency: str = "monthly", model_fingerprint: str = _MODEL_FP
) -> Any:
    return alpha_offset_grid(
        _kernel_precision(frequency=frequency, model_fingerprint=model_fingerprint),
        [0.0, 0.0, 0.0],
        [0.25, 0.5, 1.0],
        target_history_fingerprint=_TARGET_FP,
    )


def test_return_batch_binds_frequency_artifact_values_and_states() -> None:
    batch = make_return_vector_batch(
        _returns(),
        [0, 0, 1, 1],
        frequency="daily",
        model_artifact_fingerprint=_MODEL_FP,
    )

    assert batch.frequency == "daily"
    assert batch.model_artifact_fingerprint == _MODEL_FP
    assert batch.source_batch_fingerprint == batch.batch_fingerprint
    assert batch.applied_policy is None
    assert not batch.values.flags.writeable
    assert not batch.target_states.flags.writeable


@pytest.mark.parametrize(
    ("returns", "states", "frequency", "fingerprint", "message"),
    [
        ([], [], "monthly", _MODEL_FP, "shape"),
        (_returns(), [0], "monthly", _MODEL_FP, "match"),
        (_returns(), [0, 0, 1, 1], "weekly", _MODEL_FP, "base frequency"),
        (_returns(), [0, 0, 1, 1], "monthly", "bad", "model_artifact"),
    ],
)
def test_return_batch_input_contract(
    returns: Any,
    states: Any,
    frequency: Any,
    fingerprint: str,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        make_return_vector_batch(
            returns,
            states,
            frequency=frequency,
            model_artifact_fingerprint=fingerprint,
        )


def test_grid_and_raw_batch_content_tampering_is_detected() -> None:
    grid = _offset_grid()
    changed_offsets = grid.offsets.copy()
    changed_offsets[0, 0, 0] += 1e-6
    with pytest.raises(ModelError, match="grid fingerprint"):
        select_smallest_qualified_lambda(
            replace(grid, offsets=changed_offsets), [True, False, False]
        )

    batch = make_return_vector_batch(
        _returns(),
        [0, 0, 1, 1],
        frequency="monthly",
        model_artifact_fingerprint=_MODEL_FP,
    )
    changed_values = batch.values.copy()
    changed_values[0, 0] += 1e-6
    with pytest.raises(ModelError, match="batch fingerprint"):
        apply_alpha_policy(
            replace(batch, values=changed_values),
            policy=AlphaPolicy.HISTORICAL_VECTOR,
        )


def test_historical_policy_is_exact_copy_and_rejects_offsets() -> None:
    values = _returns()
    batch = make_return_vector_batch(
        values,
        [0, 0, 1, 1],
        frequency="monthly",
        model_artifact_fingerprint=_MODEL_FP,
    )
    selected = select_smallest_qualified_lambda(_offset_grid(), [False, True, True])
    result = apply_alpha_policy(batch, policy="historical_vector")

    np.testing.assert_array_equal(result.values, values)
    assert result.values is not values
    assert result.applied_policy is AlphaPolicy.HISTORICAL_VECTOR
    assert result.source_batch_fingerprint != result.batch_fingerprint
    assert not result.values.flags.writeable
    with pytest.raises(ModelError, match="does not accept"):
        apply_alpha_policy(
            batch,
            policy=AlphaPolicy.HISTORICAL_VECTOR,
            selected_offsets=selected,
        )
    with pytest.raises(ModelError, match="already been applied"):
        apply_alpha_policy(result, policy=AlphaPolicy.HISTORICAL_VECTOR)


def test_neutral_policy_subtracts_one_state_vector_once() -> None:
    values = _returns()
    states = np.asarray([0, 0, 1, 1])
    batch = make_return_vector_batch(
        values,
        states,
        frequency="monthly",
        model_artifact_fingerprint=_MODEL_FP,
    )
    selected = select_smallest_qualified_lambda(_offset_grid(), [False, False, True])

    result = apply_alpha_policy(
        batch,
        policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
        selected_offsets=selected,
    )

    np.testing.assert_allclose(result.values, values - selected.state_offsets[states])
    np.testing.assert_array_equal(values, _returns())
    assert result.applied_policy is AlphaPolicy.KERNEL_MEAN_NEUTRAL
    assert result.policy_parameter_fingerprint == selected.selection_fingerprint
    assert not result.values.flags.writeable


@pytest.mark.parametrize(
    ("policy", "use_offsets", "states", "message"),
    [
        ("unknown", False, [0, 0, 1, 1], "unknown"),
        ("kernel_mean_neutral", False, [0, 0, 1, 1], "requires"),
        ("kernel_mean_neutral", True, [0, 0, 2, 2], "no corresponding"),
    ],
)
def test_apply_policy_fails_closed(
    policy: str, use_offsets: bool, states: Any, message: str
) -> None:
    batch = make_return_vector_batch(
        _returns(),
        states,
        frequency="monthly",
        model_artifact_fingerprint=_MODEL_FP,
    )
    selected = (
        select_smallest_qualified_lambda(_offset_grid(), [True, False, False])
        if use_offsets
        else None
    )
    with pytest.raises(ModelError, match=message):
        apply_alpha_policy(batch, policy=policy, selected_offsets=selected)


def test_apply_policy_rejects_frequency_artifact_and_offset_tampering() -> None:
    batch = make_return_vector_batch(
        _returns(),
        [0, 0, 1, 1],
        frequency="monthly",
        model_artifact_fingerprint=_MODEL_FP,
    )
    selected = select_smallest_qualified_lambda(_offset_grid(), [True, False, False])

    with pytest.raises(ModelError, match="fingerprint"):
        apply_alpha_policy(
            batch,
            policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
            selected_offsets=replace(selected, frequency="daily"),
        )
    with pytest.raises(ModelError, match="fingerprint"):
        apply_alpha_policy(
            batch,
            policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
            selected_offsets=replace(selected, model_artifact_fingerprint="d" * 64),
        )
    tampered = selected.state_offsets.copy()
    tampered[0, 0] += 1e-6
    with pytest.raises(ModelError, match="fingerprint"):
        apply_alpha_policy(
            batch,
            policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
            selected_offsets=replace(selected, state_offsets=tampered),
        )


def test_policy_application_rejects_wrong_record_types_and_invalid_stage_metadata() -> (
    None
):
    batch = make_return_vector_batch(
        _returns(),
        [0, 0, 1, 1],
        frequency="monthly",
        model_artifact_fingerprint=_MODEL_FP,
    )
    selected = select_smallest_qualified_lambda(_offset_grid(), [True, False, False])

    with pytest.raises(ModelError, match="wrong type"):
        apply_alpha_policy(  # type: ignore[arg-type]
            object(), policy=AlphaPolicy.HISTORICAL_VECTOR
        )
    with pytest.raises(ModelError, match="wrong type"):
        apply_alpha_policy(
            batch,
            policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
            selected_offsets=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="lambda"):
        apply_alpha_policy(
            batch,
            policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
            selected_offsets=replace(selected, lambda_value=0.0),
        )

    applied = apply_alpha_policy(batch, policy=AlphaPolicy.HISTORICAL_VECTOR)
    with pytest.raises(ModelError, match="identity policy"):
        apply_alpha_policy(
            replace(
                applied,
                applied_policy=AlphaPolicy.KERNEL_MEAN_NEUTRAL,
            ),
            policy=AlphaPolicy.HISTORICAL_VECTOR,
        )


def test_selects_first_qualified_lambda_in_strictly_increasing_grid() -> None:
    selected = select_smallest_qualified_lambda(_offset_grid(), [False, True, True])

    assert selected.lambda_value == pytest.approx(0.5)
    np.testing.assert_allclose(selected.state_offsets, _offset_grid().offsets[1])
    assert not selected.state_offsets.flags.writeable


@pytest.mark.parametrize(
    ("qualified", "message"),
    [
        ([False, False, False], "no alpha"),
        ([True], "must be Boolean"),
        ([1, 0, 0], "must be Boolean"),
    ],
)
def test_lambda_selection_contract(qualified: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        select_smallest_qualified_lambda(_offset_grid(), qualified)


def test_lambda_selection_rejects_wrong_or_forged_grid_records() -> None:
    grid = _offset_grid()
    with pytest.raises(ModelError, match="wrong type"):
        select_smallest_qualified_lambda(  # type: ignore[arg-type]
            object(), [True, False, False]
        )
    with pytest.raises(ModelError, match="grid_fingerprint"):
        select_smallest_qualified_lambda(
            replace(grid, grid_fingerprint="bad"), [True, False, False]
        )


def test_replaced_statistics_cannot_smuggle_noncanonical_empty_values() -> None:
    stats = kernel_mean_statistics(_returns()[:2], [0, 0], n_states=2)
    invalid = replace(stats, means=np.asarray([[0.02, 0.002, 0.004], [1.0, 0.0, 0.0]]))

    with pytest.raises(ModelError, match="canonical zero"):
        merge_kernel_mean_statistics([invalid])
