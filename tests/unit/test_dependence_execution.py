"""Tests for bounded-memory block-to-dependence execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.block_execution import (
    BlockStateModel,
    CalibrationGeometry,
    PairedTraceChunk,
    RegisteredUniformChunk,
    iter_paired_trace_chunks,
    resolve_calibration_geometry,
)
from prpg.model.dependence import (
    DependenceReference,
    dependence_max_statistic_cell,
    evaluate_dependence_vector,
    fit_dependence_reference,
)
from prpg.model.dependence_execution import DependenceTraceReducer


def _returns(rows: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(size=rows)
    return np.column_stack(
        (
            common + 0.35 * rng.normal(size=rows),
            0.45 * common + rng.normal(size=rows),
            -0.25 * common + rng.normal(size=rows),
        )
    )


def _fixture(
    frequency: Literal["monthly", "daily"],
    *,
    histories: int = 8,
) -> tuple[
    CalibrationGeometry,
    np.ndarray,
    DependenceReference,
    dict[int, np.ndarray],
]:
    if frequency == "monthly":
        geometry = resolve_calibration_geometry(
            artifact="production",
            frequency="monthly",
            strict_canonical=False,
            monthly_segment_lengths=(50, 40),
        )
    else:
        geometry = resolve_calibration_geometry(
            artifact="production",
            frequency="daily",
            strict_canonical=False,
            monthly_segment_lengths=(4, 4),
            daily_segment_lengths=(80, 70),
        )
    historical = _returns(geometry.target_observations, seed=701)
    row_continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), geometry.target_continuation_links)
    )
    reference = fit_dependence_reference(
        historical, row_continuation, frequency=frequency
    )
    rng = np.random.default_rng(19_003)
    source_indices = {
        intended: rng.integers(
            0,
            geometry.target_observations,
            size=(histories, geometry.target_observations),
            dtype=np.int64,
        )
        for intended in (2, 3, 4)
    }
    return geometry, historical, reference, source_indices


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _chunk(
    *,
    geometry: CalibrationGeometry,
    intended_length: int,
    history_start: int,
    source_indices: np.ndarray,
) -> PairedTraceChunk:
    rows, observations = source_indices.shape
    target = np.tile(
        np.arange(observations, dtype=np.int64) % 3,
        (rows, 1),
    )
    conditioned = np.zeros((rows, observations), dtype=np.bool_)
    ideal = np.zeros((rows, observations), dtype=np.bool_)
    conditioned[:, 0] = True
    ideal[:, 0] = True
    offset = 0
    for length in geometry.target_segment_lengths[:-1]:
        offset += length
        conditioned[:, offset] = True
        ideal[:, offset] = True
    return PairedTraceChunk(
        history_start=history_start,
        intended_length=intended_length,
        target_states=_readonly(target),
        source_indices=_readonly(source_indices.astype(np.int64, copy=True)),
        conditioned_restart_flags=_readonly(conditioned),
        ideal_restart_flags=_readonly(ideal),
        target_continuation_links=_readonly(geometry.target_continuation_links.copy()),
    )


def _reducer(
    *,
    frequency: Literal["monthly", "daily"] = "monthly",
    histories: int = 8,
    requested_lengths: tuple[int, ...] = (2, 3, 4),
) -> tuple[
    DependenceTraceReducer,
    CalibrationGeometry,
    np.ndarray,
    DependenceReference,
    dict[int, np.ndarray],
]:
    geometry, historical, reference, indices = _fixture(frequency, histories=histories)
    kwargs: dict[str, object] = {
        "monthly_segment_lengths": geometry.monthly_segment_lengths,
    }
    if frequency == "daily":
        kwargs["daily_segment_lengths"] = geometry.target_segment_lengths
    reducer = DependenceTraceReducer(
        artifact="production",
        selected_length=3,
        requested_lengths=requested_lengths,
        historical_returns=historical,
        reference=reference,
        history_count=histories,
        strict_canonical=False,
        **kwargs,  # type: ignore[arg-type]
    )
    return reducer, geometry, historical, reference, indices


@pytest.mark.parametrize(("frequency", "components"), [("monthly", 87), ("daily", 31)])
def test_vectorized_reducer_matches_scalar_gap_aware_evaluation(
    frequency: Literal["monthly", "daily"], components: int
) -> None:
    histories = 8
    reducer, geometry, historical, reference, indices = _reducer(
        frequency=frequency, histories=histories
    )
    assert reducer.geometry == geometry
    assert reducer.requested_lengths == (2, 3, 4)

    # Deliberately fill out of order and interleave L values.  The final matrix
    # must still use absolute history indices.
    for start, stop in ((4, 8), (0, 4)):
        for intended in (4, 2, 3):
            reducer.consume(
                _chunk(
                    geometry=geometry,
                    intended_length=intended,
                    history_start=start,
                    source_indices=indices[intended][start:stop],
                )
            )
    execution = reducer.finalize()
    continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), geometry.target_continuation_links)
    )

    assert execution.requested_lengths == (2, 3, 4)
    assert len(execution.labels) == components
    assert execution.labels == reference.labels
    assert execution.labels[:3] == (
        "corr_equity_muni_bond",
        "corr_equity_taxable_bond",
        "corr_muni_bond_taxable_bond",
    )
    assert execution.labels[-1].endswith(
        "lag_12" if frequency == "monthly" else "lag_63"
    )
    for intended in (2, 3, 4):
        result = execution.result_for_length(intended)
        scalar = np.vstack(
            [
                evaluate_dependence_vector(
                    historical[source_history], continuation, reference
                ).transformed_values
                for source_history in indices[intended]
            ]
        )
        np.testing.assert_allclose(
            result.transformed_replicates,
            scalar,
            rtol=0.0,
            atol=1e-12,
        )
        assert result.transformed_replicates.shape == (histories, components)
        assert not result.transformed_replicates.flags.writeable
        expected_cell = dependence_max_statistic_cell(
            reference.transformed_values,
            scalar,
            expected_repetitions=histories,
        )
        assert result.cell.observed_statistic == pytest.approx(
            expected_cell.observed_statistic
        )
        assert result.cell.p_value == expected_cell.p_value
        np.testing.assert_allclose(
            result.cell.replicate_statistics,
            expected_cell.replicate_statistics,
            rtol=0.0,
            atol=1e-12,
        )


def test_fill_counts_finalize_idempotence_and_post_finalize_rejection() -> None:
    reducer, geometry, _, _, indices = _reducer(histories=6, requested_lengths=(3,))
    first = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=0,
        source_indices=indices[3][:3],
    )
    second = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=3,
        source_indices=indices[3][3:],
    )
    reducer(first)
    assert reducer.histories_filled == {3: 3}
    reducer(second)
    result = reducer.finalize()

    assert reducer.finalize() is result
    assert result.result_for_length(3).intended_length == 3
    with pytest.raises(ModelError, match="does not contain"):
        result.result_for_length(2)
    with pytest.raises(ModelError, match="cannot consume"):
        reducer.consume(second)


def test_reducer_consumes_actual_block_executor_chunks_without_trace_replay() -> None:
    histories = 6
    geometry, historical, reference, _ = _fixture("monthly", histories=histories)
    rng = np.random.default_rng(91)
    source_states = rng.integers(
        0, 2, size=geometry.target_observations, dtype=np.int64
    )
    source_states[:2] = (0, 1)
    source_links = geometry.target_continuation_links.copy()
    uniform_chunk = RegisteredUniformChunk(
        history_start=0,
        monthly_state_uniforms=rng.random((histories, geometry.monthly_observations)),
        restart_uniforms=rng.random((histories, geometry.target_observations)),
        source_rank_uniforms=rng.random((histories, geometry.target_observations)),
    )
    reducer = DependenceTraceReducer(
        artifact="production",
        selected_length=3,
        requested_lengths=(3,),
        historical_returns=historical,
        reference=reference,
        history_count=histories,
        strict_canonical=False,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
    )
    traces = iter_paired_trace_chunks(
        model=BlockStateModel(
            stationary_distribution=np.array([0.5, 0.5]),
            transition_matrix=np.array([[0.8, 0.2], [0.2, 0.8]]),
        ),
        geometry=geometry,
        historical_source_states=source_states,
        source_continuation_links=source_links,
        intended_lengths=(3,),
        uniform_chunks=[uniform_chunk],
        expected_history_count=histories,
    )
    for trace in traces:
        reducer(trace)

    result = reducer.finalize().result_for_length(3)
    assert result.transformed_replicates.shape == (histories, 87)
    assert result.cell.repetitions == histories


def test_overlap_is_rejected_without_changing_fill_count() -> None:
    reducer, geometry, _, _, indices = _reducer(histories=6, requested_lengths=(3,))
    chunk = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=1,
        source_indices=indices[3][1:4],
    )
    reducer.consume(chunk)
    with pytest.raises(ModelError, match="overlap") as captured:
        reducer.consume(chunk)
    assert captured.value.details["first_overlapping_history"] == 1
    assert reducer.histories_filled == {3: 3}


def test_finalize_reports_exact_gap_and_can_continue_after_failure() -> None:
    reducer, geometry, _, _, indices = _reducer(histories=6, requested_lengths=(3,))
    reducer.consume(
        _chunk(
            geometry=geometry,
            intended_length=3,
            history_start=2,
            source_indices=indices[3][2:],
        )
    )
    with pytest.raises(ModelError, match="missing history") as captured:
        reducer.finalize()
    assert captured.value.details == {
        "intended_length": 3,
        "first_missing_history": 0,
        "missing_count": 2,
    }

    reducer.consume(
        _chunk(
            geometry=geometry,
            intended_length=3,
            history_start=0,
            source_indices=indices[3][:2],
        )
    )
    assert reducer.finalize().histories == 6


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ((), "non-empty"),
        ((3, 3), "unique"),
        ((True, 3), "integers"),
        ((2, 4), "selected L"),
        ((3, 5), "feasible neighbors"),
    ],
)
def test_explicit_requested_length_set_fails_closed(
    requested: tuple[Any, ...], message: str
) -> None:
    geometry, historical, reference, _ = _fixture("monthly")
    with pytest.raises(ModelError, match=message):
        DependenceTraceReducer(
            artifact="production",
            selected_length=3,
            requested_lengths=requested,  # type: ignore[arg-type]
            historical_returns=historical,
            reference=reference,
            history_count=8,
            strict_canonical=False,
            monthly_segment_lengths=geometry.monthly_segment_lengths,
        )


def test_strict_mode_requires_exactly_ten_thousand_before_allocation() -> None:
    geometry, historical, reference, _ = _fixture("monthly")
    with pytest.raises(ModelError, match="exactly 10,000"):
        DependenceTraceReducer(
            artifact="production",
            selected_length=3,
            requested_lengths=(3,),
            historical_returns=historical,
            reference=reference,
            history_count=8,
            monthly_segment_lengths=geometry.monthly_segment_lengths,
        )


def test_strict_canonical_reducer_requires_every_feasible_neighbor() -> None:
    geometry = resolve_calibration_geometry(
        artifact="production",
        frequency="monthly",
    )
    historical = _returns(geometry.target_observations, seed=91)
    row_continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), geometry.target_continuation_links)
    )
    reference = fit_dependence_reference(
        historical,
        row_continuation,
        frequency="monthly",
    )
    with pytest.raises(ModelError, match="every feasible neighbor"):
        DependenceTraceReducer(
            artifact="production",
            selected_length=3,
            requested_lengths=(3,),
            historical_returns=historical,
            reference=reference,
        )


def test_unrequested_length_and_invalid_history_start_are_rejected() -> None:
    reducer, geometry, _, _, indices = _reducer(histories=6, requested_lengths=(3,))
    with pytest.raises(ModelError, match="unrequested"):
        reducer.consume(
            _chunk(
                geometry=geometry,
                intended_length=2,
                history_start=0,
                source_indices=indices[2][:2],
            )
        )
    with pytest.raises(ModelError, match="non-negative"):
        reducer.consume(
            _chunk(
                geometry=geometry,
                intended_length=3,
                history_start=-1,
                source_indices=indices[3][:2],
            )
        )


def test_wrong_continuation_geometry_and_writable_chunks_are_rejected() -> None:
    reducer, geometry, _, _, indices = _reducer(histories=6, requested_lengths=(3,))
    valid = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=0,
        source_indices=indices[3][:2],
    )
    wrong_links = valid.target_continuation_links.copy()
    wrong_links[49] = True
    wrong_links.setflags(write=False)
    with pytest.raises(ModelError, match="continuation mask"):
        reducer.consume(replace(valid, target_continuation_links=wrong_links))

    writable_source = valid.source_indices.copy()
    with pytest.raises(ModelError, match="source indices must be immutable"):
        reducer.consume(replace(valid, source_indices=writable_source))


@pytest.mark.parametrize(
    ("field", "replacement_factory", "message"),
    [
        (
            "target_states",
            lambda chunk: _readonly(chunk.target_states[:, :-1].copy()),
            "target states.*wrong exact geometry",
        ),
        (
            "source_indices",
            lambda chunk: _readonly(chunk.source_indices[:, :-1].copy()),
            "source indices.*wrong exact geometry",
        ),
        (
            "conditioned_restart_flags",
            lambda chunk: _readonly(chunk.conditioned_restart_flags[:, :-1].copy()),
            "restart flags.*wrong exact geometry",
        ),
    ],
)
def test_trace_tensor_shapes_fail_closed(
    field: str, replacement_factory: Any, message: str
) -> None:
    reducer, geometry, _, _, indices = _reducer(histories=6, requested_lengths=(3,))
    valid = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=0,
        source_indices=indices[3][:2],
    )
    malformed = replace(valid, **{field: replacement_factory(valid)})
    with pytest.raises(ModelError, match=message):
        reducer.consume(malformed)


def test_trace_values_bounds_and_first_restart_fail_closed() -> None:
    reducer, geometry, historical, _, indices = _reducer(
        histories=6, requested_lengths=(3,)
    )
    valid = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=0,
        source_indices=indices[3][:2],
    )
    negative_target = valid.target_states.copy()
    negative_target[0, 0] = -1
    negative_target.setflags(write=False)
    with pytest.raises(ModelError, match="target states must be non-negative"):
        reducer.consume(replace(valid, target_states=negative_target))

    outside_source = valid.source_indices.copy()
    outside_source[0, 0] = historical.shape[0]
    outside_source.setflags(write=False)
    with pytest.raises(ModelError, match="outside return history"):
        reducer.consume(replace(valid, source_indices=outside_source))

    missing_restart = valid.ideal_restart_flags.copy()
    missing_restart[0, 0] = False
    missing_restart.setflags(write=False)
    with pytest.raises(ModelError, match="first observation"):
        reducer.consume(replace(valid, ideal_restart_flags=missing_restart))

    beyond = replace(valid, history_start=5)
    with pytest.raises(ModelError, match="exceed"):
        reducer.consume(beyond)


def test_failed_vectorized_endpoint_does_not_fill_rows() -> None:
    reducer, geometry, _, _, _ = _reducer(histories=6, requested_lengths=(3,))
    constant_sources = np.zeros((2, geometry.target_observations), dtype=np.int64)
    chunk = _chunk(
        geometry=geometry,
        intended_length=3,
        history_start=0,
        source_indices=constant_sources,
    )
    with pytest.raises(ModelError, match="correlation denominator"):
        reducer.consume(chunk)
    assert reducer.histories_filled == {3: 0}


def test_reference_must_be_frozen_and_match_history_geometry() -> None:
    geometry, historical, reference, _ = _fixture("monthly")
    mutable_reference = replace(reference, asset_center=reference.asset_center.copy())
    with pytest.raises(ModelError, match="frozen and immutable"):
        DependenceTraceReducer(
            artifact="production",
            selected_length=3,
            requested_lengths=(3,),
            historical_returns=historical,
            reference=mutable_reference,
            history_count=8,
            strict_canonical=False,
            monthly_segment_lengths=geometry.monthly_segment_lengths,
        )

    changed = historical.copy()
    changed[:, 0] += np.linspace(0.0, 1.0, changed.shape[0])
    with pytest.raises(ModelError, match="inconsistent with historical"):
        DependenceTraceReducer(
            artifact="production",
            selected_length=3,
            requested_lengths=(3,),
            historical_returns=changed,
            reference=reference,
            history_count=8,
            strict_canonical=False,
            monthly_segment_lengths=geometry.monthly_segment_lengths,
        )


def test_historical_return_geometry_and_finiteness_fail_closed() -> None:
    geometry, historical, reference, _ = _fixture("monthly")
    common = {
        "artifact": "production",
        "selected_length": 3,
        "requested_lengths": (3,),
        "reference": reference,
        "history_count": 8,
        "strict_canonical": False,
        "monthly_segment_lengths": geometry.monthly_segment_lengths,
    }
    with pytest.raises(ModelError, match="wrong exact geometry"):
        DependenceTraceReducer(
            **common,
            historical_returns=historical[:-1],  # type: ignore[arg-type]
        )
    nonfinite = historical.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ModelError, match="non-finite"):
        DependenceTraceReducer(
            **common,
            historical_returns=nonfinite,  # type: ignore[arg-type]
        )


def test_reference_type_frequency_shape_and_finiteness_fail_closed() -> None:
    geometry, historical, reference, _ = _fixture("monthly")
    common = {
        "artifact": "production",
        "selected_length": 3,
        "requested_lengths": (3,),
        "historical_returns": historical,
        "history_count": 8,
        "strict_canonical": False,
        "monthly_segment_lengths": geometry.monthly_segment_lengths,
    }
    with pytest.raises(ModelError, match="requires a DependenceReference"):
        DependenceTraceReducer(
            **common,
            reference=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="frequency"):
        DependenceTraceReducer(
            **common,
            reference=replace(reference, frequency="weekly"),  # type: ignore[arg-type]
        )

    short_raw = reference.raw_values[:-1].copy()
    short_raw.setflags(write=False)
    with pytest.raises(ModelError, match="invalid exact shape"):
        DependenceTraceReducer(
            **common,
            reference=replace(reference, raw_values=short_raw),  # type: ignore[arg-type]
        )

    nonfinite_center = reference.asset_center.copy()
    nonfinite_center[0] = np.nan
    nonfinite_center.setflags(write=False)
    with pytest.raises(ModelError, match="must be finite"):
        DependenceTraceReducer(
            **common,
            reference=replace(reference, asset_center=nonfinite_center),  # type: ignore[arg-type]
        )


def test_nonsequence_lengths_nonnumeric_returns_and_scalar_controls_fail_closed() -> (
    None
):
    geometry, historical, reference, _ = _fixture("monthly")
    common = {
        "artifact": "production",
        "historical_returns": historical,
        "reference": reference,
        "history_count": 8,
        "strict_canonical": False,
        "monthly_segment_lengths": geometry.monthly_segment_lengths,
    }
    with pytest.raises(ModelError, match="explicit sequence"):
        DependenceTraceReducer(
            **common,
            selected_length=3,
            requested_lengths=3,  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="must be numeric"):
        DependenceTraceReducer(
            **{
                **common,
                "historical_returns": [["x", "y", "z"]] * 90,
            },
            selected_length=3,
            requested_lengths=(3,),
        )
    with pytest.raises(ModelError, match="selected block length"):
        DependenceTraceReducer(
            **common,
            selected_length=True,
            requested_lengths=(3,),
        )
    with pytest.raises(ModelError, match="history count"):
        DependenceTraceReducer(
            **{**common, "history_count": 0},
            selected_length=3,
            requested_lengths=(3,),
        )
    with pytest.raises(ModelError, match="strict canonical mode"):
        DependenceTraceReducer(
            **{**common, "strict_canonical": 1},
            selected_length=3,
            requested_lengths=(3,),
        )


def test_consumer_rejects_non_trace_object() -> None:
    reducer, _, _, _, _ = _reducer(histories=6, requested_lengths=(3,))
    with pytest.raises(ModelError, match="requires a PairedTraceChunk"):
        reducer.consume(object())  # type: ignore[arg-type]
