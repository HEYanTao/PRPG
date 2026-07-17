"""Tests for streaming approved return-block calibration execution."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.block_execution import (
    BlockStateModel,
    PairedTraceChunk,
    PooledBlockStatistics,
    RegisteredUniformChunk,
    RegisteredUniformMatrices,
    execute_block_calibration,
    iter_paired_trace_chunks,
    pooled_fragmentation_gates,
    resolve_calibration_geometry,
    uniform_chunks_from_matrices,
)
from prpg.model.bootstrap import (
    conditioned_source_index_trace,
    ideal_restart_trace,
    simulate_segmented_target_states_from_uniforms,
)


def _model() -> BlockStateModel:
    return BlockStateModel(
        stationary_distribution=np.array([0.5, 0.5]),
        transition_matrix=np.array([[0.8, 0.2], [0.2, 0.8]]),
    )


def _supported_source(*, run_length: int = 10, repetitions: int = 10) -> np.ndarray:
    one_pair = np.concatenate(
        [
            np.zeros(run_length, dtype=np.int64),
            np.ones(run_length, dtype=np.int64),
        ]
    )
    return np.tile(one_pair, repetitions)


def _constant_state_uniforms(observations: int) -> np.ndarray:
    return np.vstack(
        [
            np.full(observations, 0.1),
            np.full(observations, 0.1),
            np.full(observations, 0.9),
            np.full(observations, 0.9),
        ]
    )


def _matrices(
    *, observations: int = 8, restart_value: float = 0.99
) -> RegisteredUniformMatrices:
    return RegisteredUniformMatrices(
        monthly_state_uniforms=_constant_state_uniforms(observations),
        restart_uniforms=np.full((4, observations), restart_value),
        source_rank_uniforms=np.zeros((4, observations)),
    )


def _pooled(state: int, lengths: tuple[int, ...]) -> PooledBlockStatistics:
    values = np.asarray(lengths, dtype=np.int64)
    histogram = np.bincount(values, minlength=int(np.max(values)) + 1)
    return PooledBlockStatistics(
        state=state,
        count=len(lengths),
        mean=float(np.mean(values)),
        minimum=int(np.min(values)),
        maximum=int(np.max(values)),
        median=float(np.quantile(values, 0.50, method="higher")),
        singleton_fraction=float(np.mean(values == 1)),
        quantile_05=float(np.quantile(values, 0.05, method="higher")),
        quantile_95=float(np.quantile(values, 0.95, method="higher")),
        length_histogram=tuple(int(item) for item in histogram),
    )


def _execute(
    matrices: RegisteredUniformMatrices,
    *,
    chunk_size: int = 2,
    starting_length: int = 3,
):
    source = _supported_source()
    return execute_block_calibration(
        model=_model(),
        artifact="design",
        frequency="monthly",
        starting_length=starting_length,
        historical_source_states=source,
        source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
        uniform_chunks=uniform_chunks_from_matrices(matrices, chunk_size=chunk_size),
        history_count=4,
        strict_canonical=False,
        monthly_segment_lengths=(8,),
    )


def test_exact_canonical_geometries() -> None:
    design_monthly = resolve_calibration_geometry(
        artifact="design", frequency="monthly"
    )
    design_daily = resolve_calibration_geometry(artifact="design", frequency="daily")
    production_monthly = resolve_calibration_geometry(
        artifact="production", frequency="monthly"
    )
    production_daily = resolve_calibration_geometry(
        artifact="production", frequency="daily"
    )

    assert design_monthly.monthly_segment_lengths == (193,)
    assert design_monthly.target_segment_lengths == (193,)
    assert design_daily.monthly_segment_lengths == (193,)
    assert design_daily.target_segment_lengths == (4_049,)
    assert production_monthly.monthly_segment_lengths == (210, 7)
    assert production_monthly.target_observations == 217
    assert production_daily.monthly_segment_lengths == (210, 7)
    assert production_daily.target_segment_lengths == (4_404, 143)
    assert production_daily.target_observations == 4_547
    assert not production_daily.target_continuation_links[4_403]
    assert production_daily.target_continuation_links.sum() == 4_545


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "artifact": "design",
                "frequency": "monthly",
                "monthly_segment_lengths": (8,),
            },
            "strict canonical monthly",
        ),
        (
            {
                "artifact": "production",
                "frequency": "daily",
                "daily_segment_lengths": (4_389, 143),
            },
            "terminal 21-session",
        ),
        (
            {
                "artifact": "design",
                "frequency": "monthly",
                "daily_segment_lengths": (4_049,),
            },
            "cannot accept daily",
        ),
    ],
)
def test_geometry_fails_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        resolve_calibration_geometry(**kwargs)  # type: ignore[arg-type]


def test_explicit_small_daily_geometry_repeats_states_and_resets_segments() -> None:
    geometry = resolve_calibration_geometry(
        artifact="production",
        frequency="daily",
        strict_canonical=False,
        monthly_segment_lengths=(2, 2),
        daily_segment_lengths=(42, 40),
    )
    source = _supported_source(run_length=50)
    chunks = [
        RegisteredUniformChunk(
            history_start=0,
            monthly_state_uniforms=np.array([[0.1, 0.1, 0.9, 0.9]]),
            restart_uniforms=np.full((1, 82), 0.99),
            source_rank_uniforms=np.zeros((1, 82)),
        )
    ]

    trace = next(
        iter_paired_trace_chunks(
            model=_model(),
            geometry=geometry,
            historical_source_states=source,
            source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
            intended_lengths=(2,),
            uniform_chunks=chunks,
            expected_history_count=1,
        )
    )

    assert trace.target_states.shape == (1, 82)
    assert trace.target_states[0, :42].tolist() == [0] * 42
    assert trace.target_states[0, 42:].tolist() == [1] * 40
    assert not trace.target_continuation_links[41]
    assert trace.conditioned_restart_flags[0, 42]
    assert trace.ideal_restart_flags[0, 42]


def test_vectorized_trace_matches_frozen_scalar_primitives() -> None:
    geometry = resolve_calibration_geometry(
        artifact="design",
        frequency="monthly",
        strict_canonical=False,
        monthly_segment_lengths=(3, 3),
    )
    state_uniforms = np.array([[0.1, 0.9, 0.9, 0.9, 0.1, 0.1]])
    restart = np.array([[0.0, 0.8, 0.2, 0.9, 0.1, 0.9]])
    rank = np.array([[0.0, 0.6, 0.1, 0.8, 0.4, 0.9]])
    source = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int64)
    source_links = np.array([True, True, False, True, True, True, True])
    chunk = RegisteredUniformChunk(0, state_uniforms, restart, rank)

    observed = next(
        iter_paired_trace_chunks(
            model=_model(),
            geometry=geometry,
            historical_source_states=source,
            source_continuation_links=source_links,
            intended_lengths=(3,),
            uniform_chunks=[chunk],
            expected_history_count=1,
        )
    )
    expected_target = simulate_segmented_target_states_from_uniforms(
        _model().stationary_distribution,
        _model().transition_matrix,
        state_uniforms[0],
        segment_lengths=(3, 3),
    )
    expected_conditioned = conditioned_source_index_trace(
        expected_target.states,
        source,
        source_links,
        restart[0],
        rank[0],
        restart_probability=1 / 3,
        target_continuation_links=expected_target.continuation_links,
    )
    expected_ideal = ideal_restart_trace(
        expected_target.states,
        restart[0],
        restart_probability=1 / 3,
        target_continuation_links=expected_target.continuation_links,
    )

    assert observed.target_states[0].tolist() == expected_target.states.tolist()
    assert (
        observed.source_indices[0].tolist()
        == expected_conditioned.source_indices.tolist()
    )
    assert (
        observed.conditioned_restart_flags[0].tolist()
        == expected_conditioned.restart_flags.tolist()
    )
    assert (
        observed.ideal_restart_flags[0].tolist()
        == expected_ideal.restart_flags.tolist()
    )
    assert not observed.target_states.flags.writeable
    assert not observed.source_indices.flags.writeable


def test_uniform_matrix_chunking_is_contiguous_and_noncopying() -> None:
    matrices = _matrices()
    chunks = list(uniform_chunks_from_matrices(matrices, chunk_size=3))

    assert [chunk.history_start for chunk in chunks] == [0, 3]
    assert [np.asarray(chunk.monthly_state_uniforms).shape[0] for chunk in chunks] == [
        3,
        1,
    ]
    assert np.shares_memory(
        np.asarray(chunks[0].monthly_state_uniforms),
        np.asarray(matrices.monthly_state_uniforms),
    )


def test_execution_selects_largest_supported_effective_length() -> None:
    result = _execute(_matrices())

    assert result.selected_length == 3
    assert result.structurally_feasible_lengths == (4, 3, 2)
    assert result.dependence_reuse_lengths == (2, 3, 4)
    assert result.fragmentation_passed
    assert result.selected_simulation.histories == 4
    assert [item.state for item in result.selected_simulation.conditioned] == [0, 1]
    for item in result.selected_simulation.conditioned:
        assert item.count == 2
        assert item.mean == item.median == 8.0
        assert item.minimum == item.maximum == 8
        assert item.singleton_fraction == 0.0
        assert item.quantile_05 == item.quantile_95 == 8.0
        assert item.length_histogram[8] == 2
    assert all(gate.passed for gate in result.selected_fragmentation_gates)


def test_execution_is_invariant_to_uniform_chunk_size() -> None:
    matrices = _matrices()
    one = _execute(matrices, chunk_size=1)
    four = _execute(matrices, chunk_size=4)

    assert one.simulations == four.simulations
    assert one.policy_selection == four.policy_selection
    assert one.selected_fragmentation_gates == four.selected_fragmentation_gates


def test_execution_streams_immutable_common_source_histories_to_consumer() -> None:
    observed: list[tuple[int, int, tuple[int, ...], bool]] = []

    def consume(chunk: PairedTraceChunk) -> None:
        observed.append(
            (
                chunk.history_start,
                chunk.intended_length,
                tuple(chunk.source_indices.shape),
                chunk.source_indices.flags.writeable,
            )
        )

    source = _supported_source()
    result = execute_block_calibration(
        model=_model(),
        artifact="design",
        frequency="monthly",
        starting_length=3,
        historical_source_states=source,
        source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
        uniform_chunks=uniform_chunks_from_matrices(_matrices(), chunk_size=2),
        history_count=4,
        strict_canonical=False,
        monthly_segment_lengths=(8,),
        trace_consumer=consume,
    )

    assert [item[:2] for item in observed] == [
        (0, 4),
        (0, 3),
        (0, 2),
        (2, 4),
        (2, 3),
        (2, 2),
    ]
    assert all(shape == (2, 8) and not writeable for _, _, shape, writeable in observed)
    assert result.dependence_reuse_lengths == (2, 3, 4)


def test_downward_search_uses_pooled_effective_mean_then_applies_fragmentation() -> (
    None
):
    matrices = _matrices(restart_value=0.0)
    result = _execute(matrices, starting_length=3)

    assert result.selected_length == 2
    assert result.policy_selection.trials[0].intended_length == 3
    assert set(result.policy_selection.trials[0].reasons) == {
        "state_0:effective_mean_below_fraction",
        "state_1:effective_mean_below_fraction",
    }
    selected = result.selected_simulation
    assert all(item.mean == 1.0 for item in selected.conditioned)
    assert all(item.singleton_fraction == 1.0 for item in selected.conditioned)
    assert not result.fragmentation_passed
    assert all(
        gate.reasons == ("conditioned_singleton_exceeds_0.75",)
        for gate in result.selected_fragmentation_gates
    )


def test_pooled_higher_quantiles_are_exact() -> None:
    restart = np.full((4, 8), 0.99)
    restart[:, [1, 3]] = 0.0
    matrices = RegisteredUniformMatrices(
        monthly_state_uniforms=_constant_state_uniforms(8),
        restart_uniforms=restart,
        source_rank_uniforms=np.zeros((4, 8)),
    )
    result = _execute(matrices, starting_length=2)

    for item in result.selected_simulation.ideal:
        assert item.count == 6
        assert item.mean == pytest.approx(8 / 3)
        assert item.median == 2.0
        assert item.quantile_05 == 1.0
        assert item.quantile_95 == 5.0
        assert item.singleton_fraction == pytest.approx(1 / 3)


def test_compact_pooled_fragmentation_gate_arithmetic() -> None:
    passing = pooled_fragmentation_gates(
        [_pooled(0, (2, 2))], [_pooled(0, (4, 4))], intended_length=4
    )[0]
    assert passing.passed
    assert passing.conditioned_mean_passed
    assert passing.conditioned_median_passed
    assert passing.relative_singleton_passed
    assert passing.absolute_singleton_passed

    mean_failure = pooled_fragmentation_gates(
        [_pooled(0, (1, 2))], [_pooled(0, (1, 4))], intended_length=4
    )[0]
    assert mean_failure.reasons == ("conditioned_mean_below_half_intended_length",)
    assert not mean_failure.conditioned_mean_passed

    median_failure = pooled_fragmentation_gates(
        [_pooled(0, (2, 2, 2, 10))],
        [_pooled(0, (6, 6))],
        intended_length=8,
    )[0]
    assert median_failure.reasons == ("conditioned_median_below_half_ideal_median",)
    assert not median_failure.conditioned_median_passed

    relative_failure = pooled_fragmentation_gates(
        [_pooled(0, (1, 1, 2))],
        [_pooled(0, (2, 2, 2))],
        intended_length=2,
    )[0]
    assert relative_failure.reasons == (
        "conditioned_singleton_exceeds_ideal_plus_0.10",
    )
    assert not relative_failure.relative_singleton_passed

    absolute_failure = pooled_fragmentation_gates(
        [_pooled(0, (1, 1, 1, 1, 2))],
        [_pooled(0, (1, 1, 1, 1, 2))],
        intended_length=2,
    )[0]
    assert absolute_failure.reasons == ("conditioned_singleton_exceeds_0.75",)
    assert not absolute_failure.absolute_singleton_passed


def test_compact_pooled_fragmentation_gate_validates_inputs() -> None:
    valid = _pooled(0, (2, 3))
    with pytest.raises(ModelError, match="at least two"):
        pooled_fragmentation_gates([valid], [valid], intended_length=1)
    with pytest.raises(ModelError, match="matching states"):
        pooled_fragmentation_gates([valid], [_pooled(1, (2, 3))], intended_length=2)
    with pytest.raises(ModelError, match="must be non-empty"):
        pooled_fragmentation_gates([], [], intended_length=2)
    with pytest.raises(ModelError, match="inconsistent"):
        pooled_fragmentation_gates(
            [replace(valid, mean=99.0)], [valid], intended_length=2
        )


def test_strict_execution_requires_exact_history_count_before_consuming_stream() -> (
    None
):
    consumed = False

    def chunks() -> Iterator[RegisteredUniformChunk]:
        nonlocal consumed
        consumed = True
        yield RegisteredUniformChunk(
            0, np.zeros((1, 193)), np.zeros((1, 193)), np.zeros((1, 193))
        )

    source = _supported_source()
    with pytest.raises(ModelError, match="exactly 10,000"):
        execute_block_calibration(
            model=_model(),
            artifact="design",
            frequency="monthly",
            starting_length=3,
            historical_source_states=source,
            source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
            uniform_chunks=chunks(),
            history_count=4,
        )
    assert not consumed


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        (
            [
                RegisteredUniformChunk(
                    1, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))
                )
            ],
            "contiguous and zero based",
        ),
        (
            [
                RegisteredUniformChunk(
                    0, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))
                )
            ],
            "expected number",
        ),
        (
            [
                RegisteredUniformChunk(
                    0, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
                )
            ],
            "exceed",
        ),
    ],
)
def test_trace_stream_validates_history_accounting(
    chunks: list[RegisteredUniformChunk], message: str
) -> None:
    geometry = resolve_calibration_geometry(
        artifact="design",
        frequency="monthly",
        strict_canonical=False,
        monthly_segment_lengths=(2,),
    )
    source = _supported_source()
    with pytest.raises(ModelError, match=message):
        list(
            iter_paired_trace_chunks(
                model=_model(),
                geometry=geometry,
                historical_source_states=source,
                source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
                intended_lengths=(2,),
                uniform_chunks=chunks,
                expected_history_count=2 if "exceed" not in message else 1,
            )
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            {"monthly_state_uniforms": np.zeros((4, 7))},
            "wrong observation count",
        ),
        (
            {"restart_uniforms": np.full((4, 8), 1.0)},
            r"in \[0, 1\)",
        ),
        (
            {"source_rank_uniforms": np.zeros((3, 8))},
            "same history count",
        ),
    ],
)
def test_uniform_stream_rejects_malformed_registered_matrices(
    replacement: dict[str, np.ndarray], message: str
) -> None:
    base = _matrices()
    matrices = replace(base, **replacement)
    source = _supported_source()
    geometry = resolve_calibration_geometry(
        artifact="design",
        frequency="monthly",
        strict_canonical=False,
        monthly_segment_lengths=(8,),
    )
    with pytest.raises(ModelError, match=message):
        list(
            iter_paired_trace_chunks(
                model=_model(),
                geometry=geometry,
                historical_source_states=source,
                source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
                intended_lengths=(2,),
                uniform_chunks=uniform_chunks_from_matrices(matrices),
                expected_history_count=4,
            )
        )


@pytest.mark.parametrize(
    ("lengths", "message"),
    [((2, 2), "unique"), ((1,), "outside"), ((), "non-empty")],
)
def test_trace_stream_rejects_invalid_intended_lengths(
    lengths: tuple[int, ...], message: str
) -> None:
    geometry = resolve_calibration_geometry(
        artifact="design",
        frequency="monthly",
        strict_canonical=False,
        monthly_segment_lengths=(8,),
    )
    source = _supported_source()
    with pytest.raises(ModelError, match=message):
        list(
            iter_paired_trace_chunks(
                model=_model(),
                geometry=geometry,
                historical_source_states=source,
                source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
                intended_lengths=lengths,
                uniform_chunks=uniform_chunks_from_matrices(_matrices()),
                expected_history_count=4,
            )
        )


def test_state_model_and_source_history_fail_closed() -> None:
    with pytest.raises(ModelError, match="inconsistent"):
        BlockStateModel(
            stationary_distribution=np.array([0.8, 0.2]),
            transition_matrix=np.array([[0.8, 0.2], [0.2, 0.8]]),
        )

    geometry = resolve_calibration_geometry(
        artifact="design",
        frequency="monthly",
        strict_canonical=False,
        monthly_segment_lengths=(8,),
    )
    source = np.zeros(20, dtype=np.int64)
    with pytest.raises(ModelError, match="omit fitted model states"):
        list(
            iter_paired_trace_chunks(
                model=_model(),
                geometry=geometry,
                historical_source_states=source,
                source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
                intended_lengths=(2,),
                uniform_chunks=uniform_chunks_from_matrices(_matrices()),
                expected_history_count=4,
            )
        )


def test_structural_support_failure_does_not_consume_uniforms() -> None:
    consumed = False

    def chunks() -> Iterator[RegisteredUniformChunk]:
        nonlocal consumed
        consumed = True
        yield RegisteredUniformChunk(
            0, np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((4, 8))
        )

    source = np.concatenate([np.zeros(10, dtype=np.int64), np.ones(8, dtype=np.int64)])
    with pytest.raises(ModelError, match="episode and source-start support"):
        execute_block_calibration(
            model=_model(),
            artifact="design",
            frequency="monthly",
            starting_length=3,
            historical_source_states=source,
            source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
            uniform_chunks=chunks(),
            history_count=4,
            strict_canonical=False,
            monthly_segment_lengths=(8,),
        )
    assert not consumed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"artifact": "bad", "frequency": "monthly"}, "design or production"),
        ({"artifact": "design", "frequency": "bad"}, "monthly or daily"),
        (
            {"artifact": "design", "frequency": "monthly", "strict_canonical": 1},
            "must be Boolean",
        ),
        (
            {
                "artifact": "design",
                "frequency": "daily",
                "daily_segment_lengths": (4_050,),
            },
            "strict canonical daily",
        ),
        (
            {
                "artifact": "design",
                "frequency": "daily",
                "strict_canonical": False,
                "monthly_segment_lengths": (2, 1),
                "daily_segment_lengths": (42,),
            },
            "same segments",
        ),
        (
            {
                "artifact": "design",
                "frequency": "monthly",
                "strict_canonical": False,
                "monthly_segment_lengths": (),
            },
            "positive integers",
        ),
    ],
)
def test_geometry_validates_all_policy_axes(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        resolve_calibration_geometry(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("stationary", "transition", "message"),
    [
        (["x", "y"], [[0.8, 0.2], [0.2, 0.8]], "must be numeric"),
        ([1.0], [[0.8, 0.2], [0.2, 0.8]], "incompatible shape"),
        ([1.1, -0.1], [[0.8, 0.2], [0.2, 0.8]], r"in \[0, 1\]"),
        ([0.4, 0.4], [[0.8, 0.2], [0.2, 0.8]], "sum to one"),
    ],
)
def test_state_model_validates_probability_inputs(
    stationary: object, transition: object, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        BlockStateModel(
            stationary_distribution=stationary,  # type: ignore[arg-type]
            transition_matrix=transition,  # type: ignore[arg-type]
        )


def test_trace_stream_rejects_empty_and_row_mismatched_chunks() -> None:
    geometry = resolve_calibration_geometry(
        artifact="design",
        frequency="monthly",
        strict_canonical=False,
        monthly_segment_lengths=(2,),
    )
    source = _supported_source()
    common = {
        "model": _model(),
        "geometry": geometry,
        "historical_source_states": source,
        "source_continuation_links": np.ones(source.size - 1, dtype=np.bool_),
        "intended_lengths": (2,),
        "expected_history_count": 1,
    }
    empty = RegisteredUniformChunk(
        0, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    )
    with pytest.raises(ModelError, match="at least one history"):
        list(iter_paired_trace_chunks(**common, uniform_chunks=[empty]))  # type: ignore[arg-type]

    mismatched = RegisteredUniformChunk(
        0, np.zeros((1, 2)), np.zeros((2, 2)), np.zeros((1, 2))
    )
    with pytest.raises(ModelError, match="same row count"):
        list(iter_paired_trace_chunks(**common, uniform_chunks=[mismatched]))  # type: ignore[arg-type]


def test_public_entry_points_reject_invalid_scalar_controls() -> None:
    matrices = _matrices()
    with pytest.raises(ModelError, match="positive integer"):
        list(uniform_chunks_from_matrices(matrices, chunk_size=True))

    source = _supported_source()
    common = {
        "model": _model(),
        "artifact": "design",
        "frequency": "monthly",
        "historical_source_states": source,
        "source_continuation_links": np.ones(source.size - 1, dtype=np.bool_),
        "uniform_chunks": uniform_chunks_from_matrices(matrices),
        "history_count": 4,
        "strict_canonical": False,
        "monthly_segment_lengths": (8,),
    }
    with pytest.raises(ModelError, match="outside"):
        execute_block_calibration(**common, starting_length=13)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="must be callable"):
        execute_block_calibration(
            **common,
            starting_length=3,
            trace_consumer=1,  # type: ignore[arg-type]
        )


def test_daily_execution_uses_daily_start_support_rule() -> None:
    source = _supported_source(run_length=60, repetitions=4)
    matrices = RegisteredUniformMatrices(
        monthly_state_uniforms=np.array([[0.1], [0.1], [0.9], [0.9]]),
        restart_uniforms=np.full((4, 21), 0.99),
        source_rank_uniforms=np.zeros((4, 21)),
    )
    result = execute_block_calibration(
        model=_model(),
        artifact="design",
        frequency="daily",
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=np.ones(source.size - 1, dtype=np.bool_),
        uniform_chunks=uniform_chunks_from_matrices(matrices),
        history_count=4,
        strict_canonical=False,
        monthly_segment_lengths=(1,),
        daily_segment_lengths=(21,),
    )

    assert result.selected_length == 2
    assert result.policy_selection.trials[0].required_eligible_starts == 50


def test_strict_trace_stream_enforces_source_length_and_segment_mask() -> None:
    design = resolve_calibration_geometry(artifact="design", frequency="monthly")
    wrong_length = _supported_source()
    with pytest.raises(ModelError, match="wrong exact geometry"):
        list(
            iter_paired_trace_chunks(
                model=_model(),
                geometry=design,
                historical_source_states=wrong_length,
                source_continuation_links=np.ones(
                    wrong_length.size - 1, dtype=np.bool_
                ),
                intended_lengths=(2,),
                uniform_chunks=[],
            )
        )

    production = resolve_calibration_geometry(
        artifact="production", frequency="monthly"
    )
    source = np.resize(_supported_source(), 217)
    with pytest.raises(ModelError, match="wrong segments"):
        list(
            iter_paired_trace_chunks(
                model=_model(),
                geometry=production,
                historical_source_states=source,
                source_continuation_links=np.ones(216, dtype=np.bool_),
                intended_lengths=(2,),
                uniform_chunks=[],
            )
        )
