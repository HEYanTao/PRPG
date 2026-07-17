"""Tests for the exact conditioned stationary-bootstrap trace contract."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import ArrayLike

from prpg.errors import ModelError
from prpg.model.bootstrap import (
    BootstrapTraceReason,
    SourceIndexTrace,
    conditioned_source_index_trace,
    expand_monthly_states_to_daily,
    ideal_restart_trace,
    simulate_segmented_target_states_from_uniforms,
    simulate_target_states_from_uniforms,
)


def _trace(
    target: ArrayLike,
    source: ArrayLike,
    links: ArrayLike,
    *,
    restart: Sequence[float] | None = None,
    rank: Sequence[float] | None = None,
    probability: float = 0.2,
    target_links: ArrayLike | None = None,
) -> SourceIndexTrace:
    size = len(np.asarray(target))
    return conditioned_source_index_trace(
        target,
        source,
        links,
        np.full(size, 0.9) if restart is None else restart,
        np.zeros(size) if rank is None else rank,
        restart_probability=probability,
        target_continuation_links=target_links,
    )


def test_first_observation_forces_restart_and_ignores_restart_uniform() -> None:
    trace = _trace([0], [0], np.array([], dtype=bool), restart=[0.0], rank=[0.0])

    assert trace.source_indices.tolist() == [0]
    assert trace.restart_flags.tolist() == [True]
    assert trace.reasons == (BootstrapTraceReason.FIRST_OBSERVATION,)
    assert trace.restart_uniform_used.tolist() == [False]
    assert trace.source_rank_uniform_used.tolist() == [True]


def test_continuation_advances_exactly_one_source_row() -> None:
    trace = _trace([0, 0], [0, 0], [True])

    assert trace.source_indices.tolist() == [0, 1]
    assert trace.restart_flags.tolist() == [True, False]
    assert trace.reasons[-1] is BootstrapTraceReason.CONTINUATION
    assert trace.restart_uniform_used.tolist() == [False, True]
    assert trace.source_rank_uniform_used.tolist() == [True, False]


def test_random_restart_uses_strict_uniform_below_probability_rule() -> None:
    trace = _trace(
        [0, 0, 0],
        [0, 0, 0],
        [True, True],
        restart=[0.0, 0.199, 0.2],
        rank=[0.0, 0.999, 0.0],
        probability=0.2,
    )

    assert trace.source_indices.tolist() == [0, 2, 0]
    assert trace.reasons == (
        BootstrapTraceReason.FIRST_OBSERVATION,
        BootstrapTraceReason.RANDOM_RESTART,
        BootstrapTraceReason.SOURCE_END,
    )
    # Position two is forced by source end, so equality to p is not consulted.
    assert trace.restart_uniform_used.tolist() == [False, True, False]


def test_restart_uniform_equal_to_probability_continues_when_eligible() -> None:
    trace = _trace(
        [0, 0],
        [0, 0, 0],
        [True, True],
        restart=[0.0, 0.2],
        rank=[0.0, 0.9],
        probability=0.2,
    )

    assert trace.source_indices.tolist() == [0, 1]
    assert trace.reasons[-1] is BootstrapTraceReason.CONTINUATION


def test_target_state_transition_forces_restart_before_other_conditions() -> None:
    trace = _trace(
        [0, 1],
        [0, 1],
        [True],
        restart=[0.9, 0.0],
        rank=[0.0, 0.0],
    )

    assert trace.source_indices.tolist() == [0, 1]
    assert trace.reasons[-1] is BootstrapTraceReason.TARGET_STATE_TRANSITION
    assert not trace.restart_uniform_used[1]
    assert trace.source_rank_uniform_used[1]


def test_target_segment_start_forces_restart_before_state_transition() -> None:
    trace = _trace(
        [0, 1],
        [0, 1],
        [True],
        restart=[0.9, 0.0],
        rank=[0.0, 0.0],
        target_links=[False],
    )

    assert trace.reasons[-1] is BootstrapTraceReason.TARGET_SEGMENT_START
    assert trace.restart_flags.tolist() == [True, True]
    assert not trace.restart_uniform_used[1]


@pytest.mark.parametrize("target_links", [[], [1]])
def test_source_trace_rejects_invalid_target_segment_links(
    target_links: ArrayLike,
) -> None:
    with pytest.raises(ModelError, match="target continuation links"):
        _trace([0, 0], [0, 0], [True], target_links=target_links)


def test_source_end_forces_restart_without_circular_wrap() -> None:
    trace = _trace([0, 0], [0], np.array([], dtype=bool))

    assert trace.source_indices.tolist() == [0, 0]
    assert trace.reasons[-1] is BootstrapTraceReason.SOURCE_END
    assert trace.restart_flags.tolist() == [True, True]


def test_source_gap_forces_restart_before_state_mismatch() -> None:
    trace = _trace(
        [0, 0],
        [0, 1, 0],
        [False, True],
        rank=[0.0, 0.999],
    )

    assert trace.source_indices.tolist() == [0, 2]
    assert trace.reasons[-1] is BootstrapTraceReason.SOURCE_GAP


def test_source_state_mismatch_forces_restart() -> None:
    trace = _trace(
        [0, 0],
        [0, 1, 0],
        [True, True],
        rank=[0.0, 0.999],
    )

    assert trace.source_indices.tolist() == [0, 2]
    assert trace.reasons[-1] is BootstrapTraceReason.SOURCE_STATE_MISMATCH


def test_target_transition_precedes_source_end() -> None:
    trace = _trace([0, 1], [0, 1], [True], rank=[0.999, 0.0])

    assert trace.source_indices.tolist() == [0, 1]
    assert trace.reasons[-1] is BootstrapTraceReason.TARGET_STATE_TRANSITION


def test_source_rank_is_floor_mapping_over_chronological_eligible_indices() -> None:
    low = _trace([1], [1, 0, 1, 1, 0], [True] * 4, rank=[0.0])
    high = _trace(
        [1],
        [1, 0, 1, 1, 0],
        [True] * 4,
        rank=[np.nextafter(1.0, 0.0)],
    )

    assert low.source_indices.tolist() == [0]
    assert high.source_indices.tolist() == [3]


def test_fixed_draw_consumption_is_independent_of_branching() -> None:
    trace = _trace(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [True, True, True],
        restart=[0.123] * 4,
        rank=[0.0] * 4,
        probability=0.5,
    )

    assert trace.restart_uniforms_consumed == 4
    assert trace.source_rank_uniforms_consumed == 4
    assert len(trace.reasons) == 4


def test_trace_outputs_are_read_only_and_inputs_are_not_mutated() -> None:
    target = np.array([0, 0], dtype=np.int64)
    source = np.array([0, 0], dtype=np.int64)
    links = np.array([True])
    restart = np.array([0.1, 0.9])
    rank = np.array([0.0, 0.0])
    snapshots = tuple(item.copy() for item in (target, source, links, restart, rank))

    trace = conditioned_source_index_trace(
        target,
        source,
        links,
        restart,
        rank,
        restart_probability=0.2,
    )

    for actual, expected in zip(
        (target, source, links, restart, rank), snapshots, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for output in (
        trace.source_indices,
        trace.restart_flags,
        trace.restart_uniform_used,
        trace.source_rank_uniform_used,
    ):
        assert not output.flags.writeable
        with pytest.raises(ValueError):
            output[0] = output[0]


@pytest.mark.parametrize(
    ("target", "source", "links", "match"),
    [
        ([], [0], np.array([], dtype=bool), "target states must be a non-empty"),
        ([[0]], [0], np.array([], dtype=bool), "target states must be a non-empty"),
        ([0.0], [0], np.array([], dtype=bool), "target states must have integer"),
        ([-1], [0], np.array([], dtype=bool), "target states must contain"),
        ([0], [], np.array([], dtype=bool), "historical source states must be"),
        ([0], [0.0], np.array([], dtype=bool), "historical source states must have"),
        ([0], [-1], np.array([], dtype=bool), "historical source states must contain"),
        ([0], [0, 0], [], "historical length minus one"),
        ([0], [0, 0], [1], "boolean dtype"),
    ],
)
def test_source_trace_rejects_invalid_state_and_link_contracts(
    target: ArrayLike, source: ArrayLike, links: ArrayLike, match: str
) -> None:
    with pytest.raises(ModelError, match=match):
        _trace(target, source, links)


@pytest.mark.parametrize(
    ("restart", "rank", "match"),
    [
        ([0.0, 0.1], [0.0], "one draw per target"),
        ([0.0], [0.0, 0.1], "one draw per target"),
        ([[0.0]], [0.0], "one-dimensional"),
        ([False], [0.0], "numeric, not boolean"),
        ([np.nan], [0.0], "finite and in"),
        ([1.0], [0.0], "finite and in"),
        ([0.0], [-0.1], "finite and in"),
        (["bad"], [0.0], "must be numeric"),
    ],
)
def test_source_trace_rejects_invalid_uniform_contracts(
    restart: ArrayLike, rank: ArrayLike, match: str
) -> None:
    with pytest.raises(ModelError, match=match):
        conditioned_source_index_trace(
            [0],
            [0],
            np.array([], dtype=bool),
            restart,
            rank,
            restart_probability=0.2,
        )


@pytest.mark.parametrize("value", [True, "0.2", 0.0, -0.1, 1.1, np.inf, np.nan])
def test_source_trace_rejects_invalid_restart_probability(value: object) -> None:
    with pytest.raises(ModelError, match="restart probability"):
        conditioned_source_index_trace(
            [0],
            [0],
            np.array([], dtype=bool),
            [0.0],
            [0.0],
            restart_probability=value,  # type: ignore[arg-type]
        )


def test_source_trace_rejects_target_state_without_historical_support() -> None:
    with pytest.raises(ModelError, match="no eligible historical") as captured:
        _trace([2], [0, 1], [True])

    assert captured.value.details == {"state": 2}


def test_target_state_simulation_uses_initial_then_transition_uniforms() -> None:
    stationary = np.array([8.0 / 13.0, 5.0 / 13.0])
    transition = np.array([[0.75, 0.25], [0.4, 0.6]])

    trace = simulate_target_states_from_uniforms(
        stationary,
        transition,
        [0.61, 0.75, 0.399, 0.999],
    )

    assert trace.states.tolist() == [0, 1, 0, 1]
    assert trace.uniforms_consumed == 4
    assert not trace.states.flags.writeable


def test_segmented_target_simulation_resets_to_stationary_at_each_segment() -> None:
    stationary = np.array([0.8, 0.2])
    transition = np.array([[0.9, 0.1], [0.4, 0.6]])

    trace = simulate_segmented_target_states_from_uniforms(
        stationary,
        transition,
        [0.1, 0.95, 0.7, 0.95],
        segment_lengths=(2, 2),
    )

    # Row two uses pi and is state 0; treating it as a continuation from state
    # 1 would instead select state 1 at u=.7.
    assert trace.states.tolist() == [0, 1, 0, 1]
    assert trace.segment_lengths == (2, 2)
    assert trace.continuation_links.tolist() == [True, False, True]
    assert trace.uniforms_consumed == 4
    assert not trace.states.flags.writeable
    assert not trace.continuation_links.flags.writeable


@pytest.mark.parametrize(
    ("lengths", "match"),
    [
        ((), "non-empty"),
        ((1, 2), "sum to"),
        ((2, 0), "positive"),
        ((True, 3), "integers"),
    ],
)
def test_segmented_target_simulation_rejects_invalid_geometry(
    lengths: tuple[object, ...], match: str
) -> None:
    with pytest.raises(ModelError, match=match):
        simulate_segmented_target_states_from_uniforms(
            [8.0 / 13.0, 5.0 / 13.0],
            [[0.75, 0.25], [0.4, 0.6]],
            [0.1, 0.2, 0.3, 0.4],
            segment_lengths=lengths,  # type: ignore[arg-type]
        )


def test_daily_state_expansion_preserves_design_and_production_segments() -> None:
    design_monthly = np.arange(193, dtype=np.int64) % 3
    design = expand_monthly_states_to_daily(
        design_monthly,
        monthly_segment_lengths=(193,),
        daily_segment_lengths=(4049,),
    )
    assert design.states.size == 4049
    assert design.segment_lengths == (4049,)
    assert np.all(design.states[:21] == design_monthly[0])
    assert np.all(design.states[-17:] == design_monthly[-1])

    production_monthly = np.arange(217, dtype=np.int64) % 4
    production = expand_monthly_states_to_daily(
        production_monthly,
        monthly_segment_lengths=(210, 7),
        daily_segment_lengths=(4404, 143),
    )
    assert production.states.size == 4547
    assert production.segment_lengths == (4404, 143)
    assert not production.continuation_links[4403]
    assert production.states[4404] == production_monthly[210]
    assert production.states[-1] == production_monthly[-1]


def test_daily_state_expansion_rejects_global_or_over_truncation_geometry() -> None:
    with pytest.raises(ModelError, match="same segment count"):
        expand_monthly_states_to_daily(
            [0, 1],
            monthly_segment_lengths=(1, 1),
            daily_segment_lengths=(42,),
        )
    with pytest.raises(ModelError, match="terminal truncation"):
        expand_monthly_states_to_daily(
            [0, 1],
            monthly_segment_lengths=(2,),
            daily_segment_lengths=(21,),
        )


def test_ideal_restart_trace_uses_only_target_and_shared_bernoulli_rules() -> None:
    trace = ideal_restart_trace(
        [0, 0, 1, 1, 1],
        [0.0, 0.199, 0.0, 0.0, 0.2],
        restart_probability=0.2,
        target_continuation_links=[True, True, False, True],
    )

    assert trace.restart_flags.tolist() == [True, True, True, True, False]
    assert trace.reasons == (
        BootstrapTraceReason.FIRST_OBSERVATION,
        BootstrapTraceReason.RANDOM_RESTART,
        BootstrapTraceReason.TARGET_STATE_TRANSITION,
        BootstrapTraceReason.TARGET_SEGMENT_START,
        BootstrapTraceReason.CONTINUATION,
    )
    # Equality u==p continues; forced rows deliberately ignore their values.
    assert trace.restart_uniform_used.tolist() == [False, True, False, False, True]
    assert trace.restart_uniforms_consumed == 5
    assert not trace.restart_flags.flags.writeable


def test_target_state_categorical_rule_has_last_bin_fallback() -> None:
    almost_one = 1.0 - 5e-13

    trace = simulate_target_states_from_uniforms(
        [0.5, 0.5],
        np.array([[0.6, 0.4], [0.4, 0.6]]) * almost_one,
        [0.1, np.nextafter(1.0, 0.0)],
    )

    assert trace.states.tolist() == [0, 1]


def test_target_state_simulation_reuses_markov_chain_validation() -> None:
    with pytest.raises(ModelError, match="reducible"):
        simulate_target_states_from_uniforms(
            [0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], [0.1]
        )


def test_target_state_simulation_rejects_non_numeric_transition_matrix() -> None:
    with pytest.raises(ModelError, match="transition matrix must be numeric"):
        simulate_target_states_from_uniforms([1.0], [["bad"]], [0.1])


def test_target_state_simulation_rejects_a_distinct_near_stationary_vector() -> None:
    transition = [[0.979, 0.021], [0.021, 0.979]]
    almost_stationary = [0.50000000002, 0.49999999998]

    with pytest.raises(ModelError, match="differs from the validated chain"):
        simulate_target_states_from_uniforms(almost_stationary, transition, [0.1])


@pytest.mark.parametrize(
    ("stationary", "match"),
    [
        ([1.0], "incompatible shape"),
        ([0.5, np.nan], "must be finite"),
        ([-0.1, 1.1], "must be in"),
        ([0.4, 0.4], "sum to one"),
        ([0.5, "bad"], "must be numeric"),
        ([0.5, 0.5], "inconsistent with the chain"),
    ],
)
def test_target_state_simulation_rejects_invalid_stationary_distribution(
    stationary: ArrayLike, match: str
) -> None:
    with pytest.raises(ModelError, match=match):
        simulate_target_states_from_uniforms(
            stationary, [[0.75, 0.25], [0.4, 0.6]], [0.1]
        )


@pytest.mark.parametrize(
    ("uniforms", "match"),
    [
        ([], "must be non-empty"),
        ([[0.1]], "one-dimensional"),
        ([True], "numeric, not boolean"),
        ([np.inf], "finite and in"),
        ([-0.1], "finite and in"),
        ([1.0], "finite and in"),
    ],
)
def test_target_state_simulation_rejects_invalid_uniforms(
    uniforms: ArrayLike, match: str
) -> None:
    with pytest.raises(ModelError, match=match):
        simulate_target_states_from_uniforms(
            [8.0 / 13.0, 5.0 / 13.0],
            [[0.75, 0.25], [0.4, 0.6]],
            uniforms,
        )
