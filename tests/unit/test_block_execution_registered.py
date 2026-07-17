from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.block_execution import BlockStateModel, RegisteredUniformMatrices
from prpg.model.block_execution_registered import (
    RegisteredBlockUniformMaterialization,
    block_calibration_core_execution_fingerprint,
    block_calibration_execution_fingerprint,
    is_registered_block_calibration,
    is_registered_block_uniform_materialization,
    iter_materialized_block_trace_chunks,
    iter_registered_block_uniform_chunks,
    materialize_registered_block_uniforms,
    registered_block_calibration_identity,
    registered_block_core_identity,
    registered_block_fragmentation_identity,
    registered_block_replay_source,
    registered_block_source_episode_counts,
    registered_block_uniform_bytes,
    registered_block_uniform_identity,
    run_materialized_block_calibration,
    run_registered_block_calibration,
    run_registered_materialized_block_calibration,
)


def _state_model() -> BlockStateModel:
    return BlockStateModel(
        stationary_distribution=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )


def _source_history() -> tuple[np.ndarray, np.ndarray]:
    states = np.tile(np.asarray([0, 0, 0, 1, 1, 1]), 30)
    return states, np.ones(len(states) - 1, dtype=np.bool_)


def test_registered_chunks_are_schedule_independent_and_read_only() -> None:
    kwargs = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": "design",
        "frequency": "monthly",
        "history_count": 7,
        "strict_canonical": False,
        "monthly_segment_lengths": (12,),
    }
    first = tuple(iter_registered_block_uniform_chunks(chunk_size=3, **kwargs))  # type: ignore[arg-type]
    second = tuple(iter_registered_block_uniform_chunks(chunk_size=5, **kwargs))  # type: ignore[arg-type]

    first_state = np.concatenate([item.monthly_state_uniforms for item in first])
    second_state = np.concatenate([item.monthly_state_uniforms for item in second])
    first_restart = np.concatenate([item.restart_uniforms for item in first])
    second_restart = np.concatenate([item.restart_uniforms for item in second])
    first_rank = np.concatenate([item.source_rank_uniforms for item in first])
    second_rank = np.concatenate([item.source_rank_uniforms for item in second])
    np.testing.assert_array_equal(first_state, second_state)
    np.testing.assert_array_equal(first_restart, second_restart)
    np.testing.assert_array_equal(first_rank, second_rank)
    assert not np.asarray(first[0].monthly_state_uniforms).flags.writeable
    assert not np.array_equal(first_state, first_restart)
    assert not np.array_equal(first_restart, first_rank)


def test_registered_wrapper_delegates_and_accounts_for_every_stream() -> None:
    source, links = _source_history()
    result = run_registered_block_calibration(
        model=_state_model(),
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        history_count=9,
        chunk_size=4,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )

    assert result.execution.histories == 9
    assert result.history_stream_sets == 9
    assert result.state_uniforms_per_history == 12
    assert result.restart_uniforms_per_history == 12
    assert result.source_rank_uniforms_per_history == 12
    assert result.uniform_chunk_size == 4
    assert is_registered_block_calibration(result)
    identity = registered_block_calibration_identity(result)
    assert identity.master_seed == 123
    assert identity.scientific_version == 1
    assert identity.artifact == "design"
    assert identity.frequency == "monthly"
    assert identity.materialization_fingerprint == result.materialization_fingerprint
    assert identity.execution_fingerprint == result.execution_fingerprint
    assert identity.calibration_fingerprint == result.calibration_fingerprint
    assert registered_block_source_episode_counts(result) == (30, 30)
    core = registered_block_core_identity(result)
    fragmentation = registered_block_fragmentation_identity(result)
    replay = registered_block_replay_source(result)
    assert core.selected_k == 2
    assert core.selected_length == result.execution.selected_length
    assert core.materialization_fingerprint == result.materialization_fingerprint
    assert fragmentation.parent_core_fingerprint == core.core_fingerprint
    assert fragmentation.selected_length == core.selected_length
    assert replay.core_identity == core
    assert replay.uniform_identity.materialization_fingerprint == (
        core.materialization_fingerprint
    )
    assert replay.materialization is result._source_materialization


def test_block_core_identity_excludes_fragmentation_outcome_exactly() -> None:
    source, links = _source_history()
    result = run_registered_block_calibration(
        model=_state_model(),
        master_seed=321,
        scientific_version=4,
        artifact="design",
        frequency="monthly",
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        history_count=9,
        chunk_size=4,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    execution = result.execution
    changed_gate = replace(
        execution.selected_fragmentation_gates[0], reasons=("future-result",)
    )
    changed = replace(
        execution,
        selected_fragmentation_gates=(
            changed_gate,
            *execution.selected_fragmentation_gates[1:],
        ),
    )

    assert block_calibration_core_execution_fingerprint(execution) == (
        block_calibration_core_execution_fingerprint(changed)
    )
    assert block_calibration_execution_fingerprint(execution) != (
        block_calibration_execution_fingerprint(changed)
    )


def test_common_materialization_is_reusable_across_chunk_schedules() -> None:
    source, links = _source_history()
    materialized = materialize_registered_block_uniforms(
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        history_count=7,
        chunk_size=3,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    assert materialized.allocated_bytes == 7 * 12 * 3 * 8
    assert materialized.allocated_bytes == registered_block_uniform_bytes(
        artifact="design",
        frequency="monthly",
        history_count=7,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    assert is_registered_block_uniform_materialization(materialized)
    identity = registered_block_uniform_identity(materialized)
    assert identity.master_seed == 123
    assert identity.scientific_version == 1
    assert identity.rng_namespace == ("purpose4_conditioned_block_design_monthly")
    assert identity.materialization_fingerprint == (
        materialized.materialization_fingerprint
    )
    for matrix in (
        materialized.matrices.monthly_state_uniforms,
        materialized.matrices.restart_uniforms,
        materialized.matrices.source_rank_uniforms,
    ):
        assert not np.asarray(matrix).flags.writeable

    first = run_materialized_block_calibration(
        materialized,
        model=_state_model(),
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        chunk_size=2,
    )
    second = run_materialized_block_calibration(
        materialized,
        model=_state_model(),
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        chunk_size=5,
    )
    assert first.selected_length == second.selected_length
    assert first.selected_simulation == second.selected_simulation

    traces = tuple(
        iter_materialized_block_trace_chunks(
            materialized,
            model=_state_model(),
            historical_source_states=source,
            source_continuation_links=links,
            intended_lengths=(2, 3),
            chunk_size=4,
        )
    )
    assert [(trace.history_start, trace.intended_length) for trace in traces] == [
        (0, 2),
        (0, 3),
        (4, 2),
        (4, 3),
    ]
    selected_histories = sum(
        trace.history_count for trace in traces if trace.intended_length == 2
    )
    assert selected_histories == 7


@pytest.mark.parametrize(
    "updates",
    [
        {"history_count": 9_999},
        {"chunk_size": 0},
        {"artifact": "other"},
        {"frequency": "weekly"},
    ],
)
def test_registered_chunk_inputs_fail_closed(updates: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": "design",
        "frequency": "monthly",
        "history_count": 10_000,
        "chunk_size": 4,
        "strict_canonical": True,
    }
    kwargs.update(updates)
    with pytest.raises(ModelError):
        tuple(iter_registered_block_uniform_chunks(**kwargs))  # type: ignore[arg-type]


def test_materialization_identity_is_schedule_independent_and_seed_bound() -> None:
    common = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": "design",
        "frequency": "monthly",
        "history_count": 7,
        "strict_canonical": False,
        "monthly_segment_lengths": (12,),
    }
    first = materialize_registered_block_uniforms(chunk_size=2, **common)  # type: ignore[arg-type]
    second = materialize_registered_block_uniforms(chunk_size=5, **common)  # type: ignore[arg-type]
    changed_seed = materialize_registered_block_uniforms(
        chunk_size=2,
        **{**common, "master_seed": 124},  # type: ignore[arg-type]
    )

    assert first.stream_fingerprint == second.stream_fingerprint
    assert first.rng_execution_fingerprint == second.rng_execution_fingerprint
    assert first.materialization_fingerprint == second.materialization_fingerprint
    assert first.stream_fingerprint != changed_seed.stream_fingerprint
    assert first.rng_execution_fingerprint != (changed_seed.rng_execution_fingerprint)
    assert first.materialization_fingerprint != (
        changed_seed.materialization_fingerprint
    )


def test_materialization_is_bytes_immutable_and_copy_or_forgery_loses_authority() -> (
    None
):
    materialized = materialize_registered_block_uniforms(
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        history_count=7,
        chunk_size=3,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    for matrix in (
        materialized.matrices.monthly_state_uniforms,
        materialized.matrices.restart_uniforms,
        materialized.matrices.source_rank_uniforms,
    ):
        array = np.asarray(matrix)
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)

    assert not is_registered_block_uniform_materialization(copy.copy(materialized))
    assert not is_registered_block_uniform_materialization(replace(materialized))
    forged = RegisteredBlockUniformMaterialization(
        materialized.geometry,
        RegisteredUniformMatrices(
            materialized.matrices.monthly_state_uniforms,
            materialized.matrices.restart_uniforms,
            materialized.matrices.source_rank_uniforms,
        ),
        materialized.history_count,
        materialized.allocated_bytes,
        materialized.master_seed,
        materialized.scientific_version,
        materialized.rng_namespace,
        materialized.rng_execution_fingerprint,
        materialized.stream_fingerprint,
        materialized.materialization_fingerprint,
    )
    assert not is_registered_block_uniform_materialization(forged)

    original = materialized.stream_fingerprint
    object.__setattr__(materialized, "stream_fingerprint", "0" * 64)
    assert not is_registered_block_uniform_materialization(materialized)
    object.__setattr__(materialized, "stream_fingerprint", original)
    assert is_registered_block_uniform_materialization(materialized)


def test_registered_calibration_reuses_one_materialization_across_l_and_schedule() -> (
    None
):
    source, links = _source_history()
    model = _state_model()
    materialized = materialize_registered_block_uniforms(
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        history_count=9,
        chunk_size=3,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    first = run_registered_materialized_block_calibration(
        materialized,
        model=model,
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        chunk_size=2,
    )
    same_science_other_schedule = run_registered_materialized_block_calibration(
        materialized,
        model=model,
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        chunk_size=5,
    )
    neighboring_l = run_registered_materialized_block_calibration(
        materialized,
        model=model,
        starting_length=3,
        historical_source_states=source,
        source_continuation_links=links,
        chunk_size=4,
    )

    assert first.materialization_fingerprint == materialized.materialization_fingerprint
    assert same_science_other_schedule.materialization_fingerprint == (
        materialized.materialization_fingerprint
    )
    assert neighboring_l.materialization_fingerprint == (
        materialized.materialization_fingerprint
    )
    assert first.execution_fingerprint == (
        same_science_other_schedule.execution_fingerprint
    )
    assert first.calibration_fingerprint == (
        same_science_other_schedule.calibration_fingerprint
    )
    assert first.uniform_chunk_size != same_science_other_schedule.uniform_chunk_size
    assert first.calibration_fingerprint != neighboring_l.calibration_fingerprint


def test_registered_calibration_rejects_copy_public_tamper_and_parent_mutation() -> (
    None
):
    source, links = _source_history()
    model = _state_model()
    result = run_registered_block_calibration(
        model=model,
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        history_count=9,
        chunk_size=4,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    assert is_registered_block_calibration(result)
    assert not is_registered_block_calibration(copy.copy(result))
    assert not is_registered_block_calibration(replace(result))

    original = result.execution_fingerprint
    object.__setattr__(result, "execution_fingerprint", "0" * 64)
    assert not is_registered_block_calibration(result)
    object.__setattr__(result, "execution_fingerprint", original)
    assert is_registered_block_calibration(result)

    transition = np.asarray(model.transition_matrix)
    transition.setflags(write=True)
    transition[0, 0] -= 0.01
    transition[0, 1] += 0.01
    assert not is_registered_block_calibration(result)


def test_identity_guards_fail_closed_for_malformed_live_state() -> None:
    assert not is_registered_block_uniform_materialization(object())
    assert not is_registered_block_calibration(object())
    with pytest.raises(ModelError, match="exactly 10,000"):
        registered_block_uniform_bytes(
            artifact="design",
            frequency="monthly",
            history_count=1,
        )
    with pytest.raises(ModelError, match="block execution"):
        block_calibration_execution_fingerprint(object())  # type: ignore[arg-type]

    materialized = materialize_registered_block_uniforms(
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        history_count=7,
        chunk_size=3,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    original_seed = materialized.master_seed
    object.__setattr__(materialized, "master_seed", None)
    assert not is_registered_block_uniform_materialization(materialized)
    object.__setattr__(materialized, "master_seed", original_seed)

    original_bytes = materialized.allocated_bytes
    object.__setattr__(materialized, "allocated_bytes", original_bytes + 8)
    assert not is_registered_block_uniform_materialization(materialized)
    object.__setattr__(materialized, "allocated_bytes", original_bytes)

    original_matrices = materialized.matrices
    invalid_values = np.full(
        np.asarray(original_matrices.monthly_state_uniforms).shape,
        1.5,
        dtype=np.float64,
    )
    immutable_invalid = np.frombuffer(
        invalid_values.tobytes(), dtype=np.float64
    ).reshape(invalid_values.shape)
    object.__setattr__(
        materialized,
        "matrices",
        RegisteredUniformMatrices(
            immutable_invalid,
            original_matrices.restart_uniforms,
            original_matrices.source_rank_uniforms,
        ),
    )
    assert not is_registered_block_uniform_materialization(materialized)
    object.__setattr__(materialized, "matrices", original_matrices)
    assert is_registered_block_uniform_materialization(materialized)

    source, links = _source_history()
    with pytest.raises(ModelError, match="non-empty and unique"):
        tuple(
            iter_materialized_block_trace_chunks(
                materialized,
                model=_state_model(),
                historical_source_states=source,
                source_continuation_links=links,
                intended_lengths=(2, 2),
            )
        )


def test_calibration_private_source_and_accounting_tamper_fail_closed() -> None:
    source, links = _source_history()
    result = run_registered_block_calibration(
        model=_state_model(),
        master_seed=123,
        scientific_version=1,
        artifact="design",
        frequency="monthly",
        starting_length=2,
        historical_source_states=source,
        source_continuation_links=links,
        history_count=9,
        chunk_size=4,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    original_states = result._source_states
    object.__setattr__(result, "_source_states", None)
    assert not is_registered_block_calibration(result)
    object.__setattr__(result, "_source_states", original_states)

    original_count = result.history_stream_sets
    object.__setattr__(result, "history_stream_sets", original_count + 1)
    assert not is_registered_block_calibration(result)
    object.__setattr__(result, "history_stream_sets", original_count)
    assert is_registered_block_calibration(result)


def test_production_daily_rng_namespace_is_distinct_and_registered() -> None:
    result = materialize_registered_block_uniforms(
        master_seed=123,
        scientific_version=1,
        artifact="production",
        frequency="daily",
        history_count=2,
        chunk_size=1,
        strict_canonical=False,
        monthly_segment_lengths=(2,),
        daily_segment_lengths=(42,),
    )
    identity = registered_block_uniform_identity(result)
    assert identity.artifact == "production"
    assert identity.frequency == "daily"
    assert identity.rng_namespace == ("purpose4_conditioned_block_production_daily")
