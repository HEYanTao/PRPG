from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.block_length import (
    BLOCK_POLICY_VERSION,
    DIAGNOSTIC_NAMES,
    MACRO_DIAGNOSTIC_NAMES,
    BlockLengthEstimate,
    RealizedBlockStats,
    build_diagnostic_series,
    build_macro_diagnostic_series,
    combine_diagnostic_lengths,
    combine_macro_diagnostic_lengths,
    estimate_frequency_start,
    estimate_macro_start,
    paired_fragmentation_gates,
    realized_block_statistics,
    realized_mean_mapping,
    select_supported_length,
    state_support,
    stationary_bootstrap_length,
)


def _dependent_series(nobs: int = 400) -> np.ndarray:
    time = np.arange(nobs, dtype=np.float64)
    innovations = np.sin(time * 0.37) + 0.3 * np.cos(time * 0.11)
    values = np.empty(nobs, dtype=np.float64)
    values[0] = innovations[0]
    for index in range(1, nobs):
        values[index] = 0.75 * values[index - 1] + innovations[index]
    return values


def _return_matrix(nobs: int = 400) -> np.ndarray:
    factor = _dependent_series(nobs)
    time = np.arange(nobs, dtype=np.float64)
    return np.column_stack(
        (
            0.010 * factor + 0.0020 * np.sin(time),
            0.004 * factor + 0.0010 * np.cos(time * 0.19),
            0.005 * factor + 0.0015 * np.sin(time * 0.13),
        )
    )


def _tagged_runs(run_length: int = 4) -> tuple[np.ndarray, np.ndarray]:
    # Sentinel state 9 makes all selected-state episodes fully observed.
    labels = np.repeat([9, 0, 1, 0, 1, 0, 1, 0, 1, 9], run_length)
    continuation = np.ones(labels.size - 1, dtype=np.bool_)
    return labels, continuation


def _lengths(value: float) -> dict[str, float]:
    return dict.fromkeys(DIAGNOSTIC_NAMES, value)


def _stats_from_lengths(state: int, lengths: tuple[int, ...]) -> RealizedBlockStats:
    values = np.asarray(lengths, dtype=np.int64)
    return RealizedBlockStats(
        state=state,
        lengths=lengths,
        mean=float(np.mean(values)),
        minimum=int(np.min(values)),
        maximum=int(np.max(values)),
        median=float(np.quantile(values, 0.50, method="higher")),
        singleton_fraction=float(np.mean(values == 1)),
        quantile_05=float(np.quantile(values, 0.05, method="higher")),
        quantile_95=float(np.quantile(values, 0.95, method="higher")),
    )


def test_builds_exact_seven_series_and_fixes_pc1_sign() -> None:
    returns = _return_matrix(120)
    diagnostics = build_diagnostic_series(returns)

    assert diagnostics.names == DIAGNOSTIC_NAMES
    assert diagnostics.raw.shape == (120, 7)
    assert diagnostics.standardized.shape == (120, 7)
    np.testing.assert_array_equal(diagnostics.raw[:, :3], returns)
    np.testing.assert_allclose(diagnostics.raw[:, 4], returns[:, 2] - returns[:, 1])
    np.testing.assert_allclose(diagnostics.raw[:, 5], np.abs(diagnostics.raw[:, 3]))
    np.testing.assert_allclose(diagnostics.raw[:, 6], np.square(diagnostics.raw[:, 3]))
    assert diagnostics.pc1_loadings[0] > 0.0
    np.testing.assert_allclose(diagnostics.standardized.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(diagnostics.standardized.std(axis=0), 1.0)
    assert not diagnostics.raw.flags.writeable
    assert not diagnostics.pc1_loadings.flags.writeable


def test_return_correlation_pc1_is_invariant_to_positive_column_scaling() -> None:
    returns = _return_matrix(120)
    baseline = build_diagnostic_series(returns)
    rescaled = build_diagnostic_series(returns * np.array([100.0, 0.1, 17.0]))

    np.testing.assert_allclose(
        rescaled.raw[:, 3], baseline.raw[:, 3], rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(rescaled.pc1_loadings, baseline.pc1_loadings)


def test_return_correlation_pc1_fails_on_eigengap_or_equity_anchor() -> None:
    # Orthogonal balanced columns give the identity correlation matrix.
    orthogonal = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ModelError, match="top eigenvalue is not identified"):
        build_diagnostic_series(orthogonal)

    equity = orthogonal[:, 0]
    shared = orthogonal[:, 1]
    with pytest.raises(ModelError, match="sign anchor loading"):
        build_diagnostic_series(np.column_stack((equity, shared, shared)))


def test_macro_diagnostics_use_separate_registered_schema() -> None:
    factor = _dependent_series(120)
    time = np.arange(120, dtype=np.float64)
    features = np.column_stack(
        (
            factor,
            0.7 * factor + np.sin(time * 0.2),
            -0.4 * factor + np.cos(time * 0.3),
            0.2 * factor + np.sin(time * 0.07),
        )
    )
    diagnostics = build_macro_diagnostic_series(features)

    assert diagnostics.names == MACRO_DIAGNOSTIC_NAMES
    assert diagnostics.raw.shape == (120, 7)
    np.testing.assert_allclose(diagnostics.raw[:, :4].mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(diagnostics.raw[:, :4].std(axis=0), 1.0)


def test_macro_estimation_uses_separate_selector_and_support_bound() -> None:
    rng = np.random.default_rng(71)
    common = rng.normal(size=193)
    features = np.column_stack(
        tuple(common + rng.normal(scale=0.8, size=193) for _ in range(4))
    )

    calibration = estimate_macro_start(
        features,
        continuation=np.r_[False, np.ones(192, dtype=np.bool_)],
    )

    assert tuple(item.series_name for item in calibration.estimates) == (
        MACRO_DIAGNOSTIC_NAMES
    )
    assert calibration.combined.starting_length <= 24
    assert calibration.combined.sample_support_upper_bound == 24


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones((8, 2)), "shape"),
        (np.ones((2, 3)), "three synchronized"),
        (np.full((8, 3), np.nan), "NaN or infinity"),
        (np.ones((8, 3)), "constant or invalid column"),
        ([["not-a-number", 1.0, 2.0]], "numeric array"),
    ],
)
def test_diagnostic_input_validation(values: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        build_diagnostic_series(values)


def test_corrected_ppw_selector_matches_frozen_numeric_fixture() -> None:
    estimate = stationary_bootstrap_length(_dependent_series(), series_name="fixture")

    assert estimate.series_name == "fixture"
    assert estimate.observations == 400
    assert estimate.consecutive_insignificant_lags == 5
    assert estimate.maximum_lag == 25
    assert estimate.m_hat == 25
    assert estimate.selected_bandwidth == 25
    assert estimate.g_hat == pytest.approx(19.305119226352453)
    assert estimate.long_run_variance == pytest.approx(3.413237772928984)
    assert estimate.variance_constant == pytest.approx(
        2.0 * estimate.long_run_variance**2
    )
    assert estimate.stationary_length == pytest.approx(23.389662780210266)
    assert not estimate.finite_sample_truncated


def test_selector_returns_exact_one_when_no_lag_is_significant() -> None:
    series = np.random.default_rng(123).normal(size=500)
    estimate = stationary_bootstrap_length(series)

    assert estimate.m_hat == 0
    assert estimate.selected_bandwidth == 0
    assert estimate.stationary_length == estimate.unconstrained_length == 1.0
    assert estimate.g_hat is None
    assert estimate.long_run_variance is None
    assert estimate.variance_constant is None
    assert estimate.correlation_band == pytest.approx(
        2.0 * math.sqrt(math.log10(500) / 500)
    )


def test_selector_excludes_cross_gap_pairs_and_uses_longest_segment() -> None:
    first = _dependent_series(120)
    second = -_dependent_series(80) + 50.0
    series = np.concatenate((first, second))
    row_continuation = np.ones(series.size, dtype=np.bool_)
    row_continuation[[0, first.size]] = False

    estimate = stationary_bootstrap_length(
        series,
        series_name="segmented",
        continuation=row_continuation,
    )

    assert estimate.maximum_lag == min(119, math.ceil(math.sqrt(200)) + 5)


def test_selector_rejects_insufficient_segment_support() -> None:
    series = np.arange(40, dtype=np.float64)
    links = np.ones(39, dtype=np.bool_)
    links[3::4] = False

    with pytest.raises(ModelError, match="insufficient within-segment lag support"):
        stationary_bootstrap_length(series, continuation=links)


def test_selector_rejects_unavailable_available_pair_denominator() -> None:
    series = np.array(
        [0.0, 1.0, -1.0, 1.0, -1.0, 0.0] + [1.0, -1.0] * 7,
        dtype=np.float64,
    )
    links = np.zeros(series.size - 1, dtype=np.bool_)
    links[:5] = True

    with pytest.raises(ModelError, match="denominator is unavailable") as captured:
        stationary_bootstrap_length(series, continuation=links)
    assert captured.value.details["lag"] == 5


@pytest.mark.parametrize(
    "continuation",
    [
        np.ones(18, dtype=np.bool_),
        np.ones(20, dtype=np.bool_),
        np.ones(19, dtype=np.int64),
    ],
)
def test_selector_rejects_invalid_continuation_contract(
    continuation: np.ndarray,
) -> None:
    with pytest.raises(ModelError, match="continuation"):
        stationary_bootstrap_length(
            np.arange(20, dtype=np.float64), continuation=continuation
        )


def test_selector_variance_floor_is_inclusive() -> None:
    alternating = np.tile([-1.0, 1.0], 20)

    with pytest.raises(ModelError, match="variance is at or below"):
        stationary_bootstrap_length(alternating * 1e-6)


def test_selector_applies_patton_finite_sample_bound() -> None:
    estimate = stationary_bootstrap_length(np.abs(_dependent_series()))

    assert estimate.unconstrained_length > estimate.finite_sample_upper_bound
    assert estimate.stationary_length == estimate.finite_sample_upper_bound == 60.0
    assert estimate.finite_sample_truncated


@pytest.mark.parametrize(
    ("series", "message"),
    [
        (np.arange(19.0), "at least 20"),
        (np.ones(40), "variance is at or below"),
        (np.full(40, np.nan), "NaN or infinity"),
        (np.ones((40, 1)), "one-dimensional"),
        (["bad"] * 40, "numeric vector"),
        ([], "non-empty"),
    ],
)
def test_selector_rejects_invalid_series(series: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        stationary_bootstrap_length(series)


def test_frequency_estimation_runs_all_seven_and_hard_fails_monthly_cap() -> None:
    calibration = estimate_frequency_start(_return_matrix(), frequency="daily")

    assert calibration.policy_version == BLOCK_POLICY_VERSION
    assert tuple(item.series_name for item in calibration.estimates) == DIAGNOSTIC_NAMES
    assert calibration.combined.raw_ceiling_length == 60
    assert calibration.combined.starting_length == 60
    assert calibration.combined.controlling_series == ("abs_pc1", "squared_pc1")
    assert not calibration.combined.reasons

    with pytest.raises(ModelError, match="absolute hard cap") as captured:
        estimate_frequency_start(_return_matrix(), frequency="monthly")
    assert captured.value.details["raw_ceiling_length"] == 60
    assert captured.value.details["upper_bound"] == 12


def test_frequency_wide_ceil_max_and_bounds() -> None:
    assert BLOCK_POLICY_VERSION == "ppw2009-stationary-frequency-wide-v3"

    estimates = _lengths(3.01)
    estimates["pc1"] = 5.2
    combined = combine_diagnostic_lengths(estimates, frequency="monthly")
    assert combined.raw_ceiling_length == 6
    assert combined.starting_length == 6
    assert combined.controlling_series == ("pc1",)
    assert not combined.reasons

    lower = combine_diagnostic_lengths(_lengths(0.5), frequency="daily")
    assert lower.raw_ceiling_length == 2
    assert lower.starting_length == 2
    assert lower.reasons == ("raised_to_absolute_lower_bound",)

    known_v2_result = combine_diagnostic_lengths(_lengths(108.0), frequency="daily")
    assert known_v2_result.raw_ceiling_length == 108
    assert known_v2_result.starting_length == 108
    assert known_v2_result.upper_bound == 126

    exact_upper = combine_diagnostic_lengths(_lengths(126.0), frequency="daily")
    assert exact_upper.raw_ceiling_length == 126
    assert exact_upper.starting_length == 126
    assert exact_upper.upper_bound == 126

    with pytest.raises(ModelError, match="absolute hard cap"):
        combine_diagnostic_lengths(_lengths(127.0), frequency="daily")


def test_frequency_combiner_accepts_estimate_objects() -> None:
    fixture = stationary_bootstrap_length(_dependent_series(), series_name="source")
    estimates: dict[str, BlockLengthEstimate | float] = dict(_lengths(2.1))
    estimates["equity_return"] = fixture

    combined = combine_diagnostic_lengths(estimates, frequency="daily")

    assert combined.raw_ceiling_length == 24
    assert combined.starting_length == 24
    assert combined.controlling_series == ("equity_return",)


def test_macro_combiner_enforces_absolute_and_sample_support_caps() -> None:
    estimates = dict.fromkeys(MACRO_DIAGNOSTIC_NAMES, 3.2)
    combined = combine_macro_diagnostic_lengths(estimates, observations=193)

    assert combined.raw_ceiling_length == combined.starting_length == 4
    assert combined.absolute_upper_bound == 24
    assert combined.sample_support_upper_bound == 24

    exact_sample = dict.fromkeys(MACRO_DIAGNOSTIC_NAMES, 4.0)
    assert (
        combine_macro_diagnostic_lengths(exact_sample, observations=32).starting_length
        == 4
    )
    above_sample = dict.fromkeys(MACRO_DIAGNOSTIC_NAMES, 4.01)
    with pytest.raises(ModelError, match="binding hard cap") as captured:
        combine_macro_diagnostic_lengths(above_sample, observations=32)
    assert captured.value.details["sample_support_upper_bound"] == 4

    above_absolute = dict.fromkeys(MACRO_DIAGNOSTIC_NAMES, 24.01)
    with pytest.raises(ModelError, match="binding hard cap"):
        combine_macro_diagnostic_lengths(above_absolute, observations=400)


def test_frequency_combiner_fails_closed_on_schema_or_value_error() -> None:
    missing = _lengths(3.0)
    del missing["pc1"]
    with pytest.raises(ModelError, match="exactly seven") as captured:
        combine_diagnostic_lengths(missing, frequency="monthly")
    assert captured.value.details["missing"] == ["pc1"]

    invalid = _lengths(3.0)
    invalid["pc1"] = math.inf
    with pytest.raises(ModelError, match="non-finite or non-positive"):
        combine_diagnostic_lengths(invalid, frequency="monthly")

    with pytest.raises(ModelError, match="monthly or daily"):
        combine_diagnostic_lengths(_lengths(3.0), frequency="weekly")  # type: ignore[arg-type]


def test_state_support_counts_only_completed_gap_free_episodes() -> None:
    states, continuation = _tagged_runs()
    baseline = state_support(
        states,
        continuation,
        intended_length=3,
        selected_states=(0, 1),
    )
    assert [(item.state, item.completed_episodes) for item in baseline] == [
        (0, 4),
        (1, 4),
    ]
    assert [item.eligible_start_count for item in baseline] == [8, 8]

    # Split the first selected state-0 episode.  Both fragments are censored by
    # the source gap and no length-3 start may cross it.
    gap_after = 4 + 1
    continuation[gap_after] = False
    with_gap = state_support(
        states,
        continuation,
        intended_length=3,
        selected_states=(0, 1),
    )
    assert with_gap[0].completed_episodes == 3
    assert with_gap[0].eligible_start_count == 6
    assert with_gap[1].completed_episodes == 4
    assert with_gap[1].eligible_start_count == 8


def test_support_handles_length_longer_than_history() -> None:
    support = state_support(
        np.array([0, 0, 1, 1]),
        np.ones(3, dtype=np.bool_),
        intended_length=10,
        selected_states=(0, 1),
    )
    assert [item.eligible_start_count for item in support] == [0, 0]


@pytest.mark.parametrize(
    ("states", "continuation", "selected", "length", "message"),
    [
        ([0, 1], [True], (0, 1), 1, "at least two"),
        ([0, 1], [], (0, 1), 2, "length n-1"),
        ([0, 1], [1], (0, 1), 2, "Boolean vector"),
        ([0.0, 1.0], [True], (0, 1), 2, "integer vector"),
        ([-1, 0], [True], (0,), 2, "non-negative"),
        ([0, 1], [True], (), 2, "non-empty and unique"),
        ([0, 1], [True], (0, 0), 2, "non-empty and unique"),
        ([0, 1], [True], (2,), 2, "absent"),
    ],
)
def test_support_input_validation(
    states: Any,
    continuation: Any,
    selected: tuple[int, ...],
    length: int,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        state_support(
            states,
            continuation,
            intended_length=length,
            selected_states=selected,
        )


def test_downward_rule_uses_effective_mean_after_support() -> None:
    states, continuation = _tagged_runs(run_length=10)
    realized = {
        6: {0: 2.9, 1: 4.0},
        5: {0: 2.5, 1: 3.0},
    }

    selected = select_supported_length(
        starting_length=6,
        frequency="monthly",
        states=states,
        continuation=continuation,
        realized_means_by_length=realized,
        selected_states=(0, 1),
    )

    assert selected.selected_length == 5
    assert selected.restart_probability == 0.2
    assert len(selected.trials) == 2
    assert selected.trials[0].reasons == ("state_0:effective_mean_below_fraction",)
    assert selected.trials[1].feasible
    assert selected.trials[1].required_eligible_starts == 8


def test_downward_rule_allows_zero_completed_episodes_when_starts_are_sufficient() -> (
    None
):
    states = np.concatenate((np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)))
    selected = select_supported_length(
        starting_length=3,
        frequency="monthly",
        states=states,
        continuation=np.ones(states.size - 1, dtype=np.bool_),
        realized_means_by_length={3: {0: 3.0, 1: 3.0}},
        selected_states=(0, 1),
        minimum_completed_episodes=0,
    )

    assert selected.selected_length == 3
    assert [item.completed_episodes for item in selected.trials[-1].support] == [0, 0]
    assert [item.eligible_start_count for item in selected.trials[-1].support] == [8, 8]


def test_daily_rule_uses_fifty_start_floor() -> None:
    states, continuation = _tagged_runs(run_length=20)
    means = {length: {0: float(length), 1: float(length)} for length in range(2, 11)}

    selected = select_supported_length(
        starting_length=10,
        frequency="daily",
        states=states,
        continuation=continuation,
        realized_means_by_length=means,
        selected_states=(0, 1),
    )

    assert selected.selected_length == 8
    assert [trial.intended_length for trial in selected.trials] == [10, 9, 8]
    assert selected.trials[-1].required_eligible_starts == 50


def test_downward_rule_records_missing_means_and_fails_at_two() -> None:
    states, continuation = _tagged_runs(run_length=2)
    with pytest.raises(ModelError, match="no block length") as captured:
        select_supported_length(
            starting_length=4,
            frequency="monthly",
            states=states,
            continuation=continuation,
            realized_means_by_length={},
            selected_states=(0, 1),
        )
    assert [trial["length"] for trial in captured.value.details["trials"]] == [4, 3, 2]
    assert (
        "state_0:missing_realized_mean"
        in captured.value.details["trials"][-1]["reasons"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"starting_length": 1}, "outside"),
        ({"starting_length": 13}, "outside"),
        ({"minimum_completed_episodes": -1}, "thresholds"),
        ({"minimum_effective_fraction": 1.1}, "thresholds"),
    ],
)
def test_downward_rule_validates_policy_inputs(
    kwargs: dict[str, float | int], message: str
) -> None:
    states, continuation = _tagged_runs(run_length=10)
    arguments: dict[str, object] = {
        "starting_length": 4,
        "frequency": "monthly",
        "states": states,
        "continuation": continuation,
        "realized_means_by_length": {4: {0: 4.0, 1: 4.0}},
        "selected_states": (0, 1),
    }
    arguments.update(kwargs)
    with pytest.raises(ModelError, match=message):
        select_supported_length(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, math.nan, math.inf])
def test_downward_rule_rejects_invalid_realized_mean(value: float) -> None:
    states, continuation = _tagged_runs(run_length=10)
    with pytest.raises(ModelError, match="effective block length is invalid"):
        select_supported_length(
            starting_length=4,
            frequency="monthly",
            states=states,
            continuation=continuation,
            realized_means_by_length={4: {0: value, 1: 4.0}},
            selected_states=(0, 1),
        )


def test_realized_effective_distribution_and_mapping() -> None:
    stats = realized_block_statistics(
        np.array([0, 0, 0, 0, 1, 1, 1]),
        np.array([True, False, True, False, True, False, False]),
        source_indices=np.array([0, 1, 2, 3, 6, 7, 8]),
        source_continuation_links=np.ones(8, dtype=np.bool_),
    )

    assert stats == (
        _stats_from_lengths(0, (4,)),
        _stats_from_lengths(1, (3,)),
    )
    assert stats[0].count == 1
    assert stats[0].median == stats[0].quantile_05 == stats[0].quantile_95 == 4.0
    assert stats[0].singleton_fraction == 0.0
    assert realized_mean_mapping(stats) == {0: 4.0, 1: 3.0}


def test_conditioned_blocks_split_nonadjacency_gap_and_target_segment() -> None:
    states = np.zeros(7, dtype=np.int64)
    restarts = np.array([True, False, True, True, False, True, False])
    sources = np.array([0, 1, 2, 4, 5, 6, 7])
    source_links = np.ones(7, dtype=np.bool_)
    source_links[5] = False
    target_links = np.ones(6, dtype=np.bool_)
    target_links[4] = False

    stats = realized_block_statistics(
        states,
        restarts,
        source_indices=sources,
        source_continuation_links=source_links,
        target_continuation_links=target_links,
    )

    # The random restart at row 2 selected source row 2 and therefore merges.
    # Nonadjacency at row 3, target segment start at row 5, and the coincident
    # source gap split the trace; the terminal block is retained.
    assert stats == (_stats_from_lengths(0, (3, 2, 2)),)


def test_ideal_blocks_split_every_registered_restart_and_use_higher_quantiles() -> None:
    stats = realized_block_statistics(
        np.zeros(7, dtype=np.int64),
        np.array([True, False, True, False, True, False, False]),
        ideal=True,
    )

    assert stats == (_stats_from_lengths(0, (2, 2, 3)),)
    assert stats[0].median == 2.0
    assert stats[0].quantile_95 == 3.0


@pytest.mark.parametrize(
    ("states", "restarts", "message"),
    [
        ([], [], "integer vector"),
        ([0, 0], [True], "matching"),
        ([0, 0], [1, 0], "Boolean"),
        ([0, 0], [False, False], "first"),
        ([0, 1], [True, False], "transition"),
    ],
)
def test_realized_trace_validation(states: Any, restarts: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        realized_block_statistics(states, restarts, ideal=True)


def test_conditioned_statistics_fail_closed_without_source_provenance() -> None:
    with pytest.raises(ModelError, match="require source indices"):
        realized_block_statistics([0, 0], [True, False])


def test_paired_fragmentation_gate_arithmetic_and_boundaries() -> None:
    passing = paired_fragmentation_gates(
        [_stats_from_lengths(0, (2, 2))],
        [_stats_from_lengths(0, (4, 4))],
        intended_length=4,
    )[0]
    assert passing.passed
    assert passing.conditioned_mean_passed
    assert passing.conditioned_median_passed
    assert passing.relative_singleton_passed
    assert passing.absolute_singleton_passed
    assert passing.conditioned_mean_minimum == 2.0
    assert passing.conditioned_median_minimum == 2.0

    mean_failure = paired_fragmentation_gates(
        [_stats_from_lengths(0, (1, 2))],
        [_stats_from_lengths(0, (1, 4))],
        intended_length=4,
    )[0]
    assert mean_failure.reasons == ("conditioned_mean_below_half_intended_length",)
    assert not mean_failure.conditioned_mean_passed

    median_failure = paired_fragmentation_gates(
        [_stats_from_lengths(0, (2, 2, 2, 10))],
        [_stats_from_lengths(0, (6, 6))],
        intended_length=8,
    )[0]
    assert median_failure.reasons == ("conditioned_median_below_half_ideal_median",)
    assert not median_failure.conditioned_median_passed

    relative_failure = paired_fragmentation_gates(
        [_stats_from_lengths(0, (1, 1, 2))],
        [_stats_from_lengths(0, (2, 2, 2))],
        intended_length=2,
    )[0]
    assert relative_failure.reasons == (
        "conditioned_singleton_exceeds_ideal_plus_0.10",
    )
    assert not relative_failure.relative_singleton_passed

    absolute_failure = paired_fragmentation_gates(
        [_stats_from_lengths(0, (1, 1, 1, 1, 2))],
        [_stats_from_lengths(0, (1, 1, 1, 1, 2))],
        intended_length=2,
    )[0]
    assert absolute_failure.reasons == ("conditioned_singleton_exceeds_0.75",)
    assert not absolute_failure.absolute_singleton_passed


def test_paired_fragmentation_requires_matching_nonempty_state_sets() -> None:
    with pytest.raises(ModelError, match="same states"):
        paired_fragmentation_gates(
            [_stats_from_lengths(0, (2,))],
            [_stats_from_lengths(1, (2,))],
            intended_length=2,
        )
    with pytest.raises(ModelError, match="empty"):
        paired_fragmentation_gates([], [], intended_length=2)


def test_realized_mean_mapping_rejects_empty_duplicate_or_invalid() -> None:
    with pytest.raises(ModelError, match="empty"):
        realized_mean_mapping([])

    valid = RealizedBlockStats(0, (2,), 2.0, 2, 2)
    with pytest.raises(ModelError, match="duplicate"):
        realized_mean_mapping([valid, valid])

    invalid = RealizedBlockStats(0, (), math.nan, 0, 0)
    with pytest.raises(ModelError, match="invalid"):
        realized_mean_mapping([invalid])
