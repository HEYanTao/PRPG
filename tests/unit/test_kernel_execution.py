"""Tests for registered kernel-calibration and verification execution."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import prpg.model.kernel_execution as kernel_execution
from prpg.errors import ModelError
from prpg.model.block_execution import BlockStateModel
from prpg.model.kernel_execution import (
    CANONICAL_PREFIX_WINDOW_COUNTS,
    PURPOSE_7_DRAW_ORDER,
    KernelCalibrationCellSpec,
    RegisteredKernelWindowBatch,
    is_registered_kernel_calibration_execution,
    kernel_calibration_execution_identity,
    merge_registered_kernel_batches,
    registered_kernel_memory_estimate,
    registered_kernel_window_batch,
    replay_registered_kernel_verification,
    run_registered_kernel_calibration_cell,
    run_registered_kernel_calibration_suite,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    calibration_entity,
)

MODEL_FP = "a" * 64
TARGET_FP = "b" * 64


def _model() -> BlockStateModel:
    return BlockStateModel(
        stationary_distribution=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.82, 0.18], [0.18, 0.82]]),
    )


def _monthly_source() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    links = np.ones(states.size - 1, dtype=np.bool_)
    time = np.arange(states.size, dtype=np.float64)
    returns = np.column_stack(
        (
            0.001 * time + 0.004 * states,
            -0.0004 * time + 0.002 * states,
            0.0007 * time - 0.001 * states,
        )
    )
    return returns, states, links


def _daily_source() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = 240
    states = np.repeat(np.asarray([0, 1, 0, 1]), 60).astype(np.int64)
    links = np.ones(observations - 1, dtype=np.bool_)
    time = np.arange(observations, dtype=np.float64)
    returns = np.column_stack(
        (
            np.sin(time / 11.0) * 0.003 + states * 0.001,
            np.cos(time / 17.0) * 0.002 - states * 0.0005,
            np.sin(time / 23.0) * 0.0015 + states * 0.0007,
        )
    )
    return returns, states, links


def _batch(
    first: int,
    windows: int,
    *,
    frequency: str = "monthly",
) -> RegisteredKernelWindowBatch:
    if frequency == "monthly":
        returns, states, links = _monthly_source()
        kwargs: dict[str, object] = {"monthly_segment_lengths": (12,)}
    else:
        returns, states, links = _daily_source()
        kwargs = {
            "monthly_segment_lengths": (12,),
            "daily_segment_lengths": (240,),
        }
    return registered_kernel_window_batch(
        model=_model(),
        master_seed=987,
        scientific_version=1,
        artifact="design",
        frequency=frequency,  # type: ignore[arg-type]
        selected_length=2,
        historical_returns=returns,
        historical_source_states=states,
        source_continuation_links=links,
        model_artifact_fingerprint=MODEL_FP,
        first_window_index=first,
        window_count=windows,
        strict_canonical=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _run_cell(
    **updates: object,
) -> Any:
    returns, states, links = _monthly_source()
    kwargs: dict[str, object] = {
        "model": _model(),
        "master_seed": 987,
        "scientific_version": 1,
        "artifact": "design",
        "frequency": "monthly",
        "selected_length": 2,
        "historical_returns": returns,
        "historical_source_states": states,
        "source_continuation_links": links,
        "model_artifact_fingerprint": MODEL_FP,
        "target_history_fingerprint": TARGET_FP,
        "workers": 1,
        "prefix_window_counts": (4, 8),
        "annualized_half_width_max": 100.0,
        "confidence": 0.95,
        "lambdas": (0.25, 0.5, 0.75, 1.0),
        "statistics_batch_windows": 2,
        "strict_canonical": False,
        "monthly_segment_lengths": (12,),
    }
    kwargs.update(updates)
    return run_registered_kernel_calibration_cell(**kwargs)  # type: ignore[arg-type]


def test_registered_batch_is_prefix_stable_compact_and_auditable() -> None:
    first = _batch(0, 3)
    replay = _batch(0, 3)
    later = _batch(3, 2)

    assert (
        first.statistics.source_window_fingerprint
        == replay.statistics.source_window_fingerprint
    )
    assert first.statistics.window_fingerprint == replay.statistics.window_fingerprint
    assert (
        first.statistics.source_window_fingerprint
        != later.statistics.source_window_fingerprint
    )
    assert first.draw_order == PURPOSE_7_DRAW_ORDER
    assert first.windows == 3
    assert first.draws_per_window == 36
    assert first.first_stream_entity == calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION,
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0,
        0,
    )
    assert first.last_stream_entity == calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION,
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0,
        2,
    )
    for array in (
        first.statistics.state_window_counts,
        first.statistics.state_observation_counts,
        first.statistics.state_return_sums,
    ):
        assert not array.flags.writeable


def test_daily_batch_uses_monthly_states_then_target_frequency_draws() -> None:
    batch = _batch(0, 2, frequency="daily")

    assert batch.frequency == "daily"
    assert batch.draws_per_window == 12 + 2 * 240
    assert batch.statistics.windows == 2
    assert int(batch.statistics.state_observation_counts.sum()) == 2 * 240


def test_merge_sorts_worker_completion_order_and_is_deterministic() -> None:
    zero = _batch(0, 2)
    two = _batch(2, 2)
    ordered = merge_registered_kernel_batches((zero, two))
    reversed_result = merge_registered_kernel_batches((two, zero))

    assert ordered.window_fingerprint == reversed_result.window_fingerprint
    assert (
        ordered.source_window_fingerprint == reversed_result.source_window_fingerprint
    )
    np.testing.assert_array_equal(
        ordered.state_observation_counts,
        reversed_result.state_observation_counts,
    )


@pytest.mark.parametrize(
    "batches, message",
    [
        ((), "at least one"),
        ((_batch(2, 2),), "begin at window zero"),
        ((_batch(0, 2), _batch(3, 1)), "contiguous"),
        ((_batch(0, 2), _batch(1, 2)), "contiguous"),
    ],
)
def test_merge_rejects_missing_and_overlapping_ranges(
    batches: tuple[RegisteredKernelWindowBatch, ...], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        merge_registered_kernel_batches(batches)


def test_merge_rejects_tampered_entities_and_identity() -> None:
    first = _batch(0, 2)
    second = _batch(2, 2)
    with pytest.raises(ModelError, match="entities"):
        merge_registered_kernel_batches((replace(first, first_stream_entity=0), second))
    with pytest.raises(ModelError, match="draw contract"):
        merge_registered_kernel_batches(
            (first, replace(second, draw_order=("a", "b", "c")))
        )
    changed_statistics = replace(second.statistics, model_artifact_fingerprint="c" * 64)
    with pytest.raises(ModelError, match="identity"):
        merge_registered_kernel_batches(
            (first, replace(second, statistics=changed_statistics))
        )


def test_calibration_stops_at_first_passing_look_and_verifies_all_offsets() -> None:
    result = _run_cell()

    assert result.passed
    assert result.precision_passed
    assert result.verification_passed
    assert result.stopped_windows == 4
    assert len(result.completed_looks) == 1
    assert result.completed_looks[0].passed
    assert result.final_precision.simultaneous_comparisons == 72
    assert result.offset_grid is not None
    assert result.offset_grid.offsets.shape == (4, 2, 3)
    assert result.verification is not None
    assert result.verification.purpose is CalibrationPurpose.KERNEL_VERIFICATION
    assert result.verification.stream_count == 0
    assert result.verification.directly_adjusted_means.shape == (4, 2, 3)
    assert result.verification.maximum_absolute_mean_error <= 1e-12
    assert result.purpose_7_streams == 4
    assert result.purpose_8_streams == 0
    assert result.first_purpose_7_entity == calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION,
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0,
        0,
    )
    assert result.last_purpose_7_entity == calibration_entity(
        CalibrationPurpose.KERNEL_ESTIMATION,
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0,
        3,
    )
    assert result.failure_reason is None
    for array in (
        result.verification.observation_counts,
        result.verification.directly_adjusted_means,
        result.verification.algebraic_adjusted_means,
        result.verification.algebraic_residual_means,
    ):
        assert not array.flags.writeable


def test_kernel_execution_identity_revalidates_sources_and_rejects_copies() -> None:
    model = _model()
    result = _run_cell(model=model)
    identity = kernel_calibration_execution_identity(result)

    assert is_registered_kernel_calibration_execution(result)
    assert identity.master_seed == 987
    assert identity.scientific_version == 1
    assert identity.artifact == "design"
    assert identity.frequency == "monthly"
    assert identity.parent_model_fingerprint == result.parent_model_fingerprint
    assert identity.source_materialization_fingerprint == (
        result.source_materialization_fingerprint
    )
    assert identity.rng_execution_fingerprint == result.calibration_stream_fingerprint
    assert identity.execution_fingerprint == result.execution_fingerprint
    assert not is_registered_kernel_calibration_execution(copy.copy(result))
    assert not is_registered_kernel_calibration_execution(replace(result))

    assert result.verification is not None
    immutable = result.verification.directly_adjusted_means
    with pytest.raises(ValueError):
        immutable.setflags(write=True)

    model.transition_matrix.setflags(write=True)
    model.transition_matrix[0, 0] = 0.81
    assert not is_registered_kernel_calibration_execution(result)


@pytest.mark.parametrize(
    ("field", "value"),
    (("_master_seed", 988), ("_scientific_version", 2), ("n_states", 3)),
)
def test_kernel_execution_identity_rejects_tampered_contract_fields(
    field: str,
    value: object,
) -> None:
    result = _run_cell()
    object.__setattr__(result, field, value)
    assert not is_registered_kernel_calibration_execution(result)


def test_lambda_one_verification_targets_the_historical_mean() -> None:
    result = _run_cell()
    assert result.verification is not None
    returns, _, _ = _monthly_source()

    np.testing.assert_allclose(
        result.verification.algebraic_adjusted_means[-1],
        np.broadcast_to(np.mean(returns, axis=0), (2, 3)),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.verification.algebraic_residual_means[-1],
        0.0,
        rtol=0.0,
        atol=1e-15,
    )


def test_public_verification_replay_reproduces_embedded_evidence() -> None:
    result = _run_cell()
    returns, states, links = _monthly_source()

    replay = replay_registered_kernel_verification(
        result,
        model=_model(),
        master_seed=987,
        scientific_version=1,
        historical_returns=returns,
        historical_source_states=states,
        source_continuation_links=links,
    )

    assert result.verification is not None
    assert replay.replay_kernel_sample_fingerprint == (
        result.verification.replay_kernel_sample_fingerprint
    )
    np.testing.assert_array_equal(
        replay.directly_adjusted_means,
        result.verification.directly_adjusted_means,
    )


def test_public_verification_replay_rejects_failed_or_changed_execution() -> None:
    failed = _run_cell(annualized_half_width_max=1e-20)
    returns, states, links = _monthly_source()
    kwargs = {
        "model": _model(),
        "master_seed": 987,
        "scientific_version": 1,
        "historical_returns": returns,
        "historical_source_states": states,
        "source_continuation_links": links,
    }
    with pytest.raises(ModelError, match="precision-passed offset grid"):
        replay_registered_kernel_verification(failed, **kwargs)

    passed = _run_cell()
    with pytest.raises(ModelError, match="stream identity changed"):
        replay_registered_kernel_verification(
            passed,
            **{**kwargs, "master_seed": 988},
        )
    with pytest.raises(ModelError, match="calibration execution"):
        replay_registered_kernel_verification(  # type: ignore[arg-type]
            object(),
            **kwargs,
        )


def test_precision_failure_at_last_look_creates_no_offset_or_verification() -> None:
    result = _run_cell(annualized_half_width_max=1e-20)

    assert not result.passed
    assert not result.precision_passed
    assert not result.verification_passed
    assert result.stopped_windows == 8
    assert len(result.completed_looks) == 2
    assert result.offset_grid is None
    assert result.verification is None
    assert (
        result.failure_reason == "precision_target_not_met_at_maximum_registered_look"
    )


def test_precision_without_required_verification_fails_closed() -> None:
    result = _run_cell(verify=False)

    assert result.precision_passed
    assert not result.passed
    assert result.offset_grid is not None
    assert result.verification is None
    assert result.failure_reason == "purpose_8_offset_verification_not_run"


def test_process_map_completion_order_does_not_change_result() -> None:
    def ordered_map(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        return tuple(function(item) for item in items)

    def reversed_map(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        return tuple(reversed(tuple(function(item) for item in items)))

    baseline = _run_cell(
        workers=2,
        statistics_batch_windows=1,
        process_map=ordered_map,
    )
    reordered = _run_cell(
        workers=2,
        statistics_batch_windows=1,
        process_map=reversed_map,
    )

    assert (
        baseline.final_precision.kernel_sample_fingerprint
        == reordered.final_precision.kernel_sample_fingerprint
    )
    assert baseline.offset_grid is not None and reordered.offset_grid is not None
    assert (
        baseline.offset_grid.grid_fingerprint == reordered.offset_grid.grid_fingerprint
    )
    baseline_identity = kernel_calibration_execution_identity(baseline)
    reordered_identity = kernel_calibration_execution_identity(reordered)
    assert baseline_identity.parent_model_fingerprint == (
        reordered_identity.parent_model_fingerprint
    )
    assert baseline_identity.source_materialization_fingerprint == (
        reordered_identity.source_materialization_fingerprint
    )
    assert baseline_identity.rng_execution_fingerprint == (
        reordered_identity.rng_execution_fingerprint
    )
    assert baseline_identity.execution_fingerprint == (
        reordered_identity.execution_fingerprint
    )


def test_process_map_missing_or_malformed_results_fail_closed() -> None:
    def missing_map(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        results = tuple(function(item) for item in items)
        return results[:-1]

    with pytest.raises(ModelError, match="missing, duplicated, or misranged"):
        _run_cell(workers=2, statistics_batch_windows=1, process_map=missing_map)

    def malformed_map(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        del function, items
        return (object(),)

    with pytest.raises(ModelError, match="malformed worker group"):
        _run_cell(process_map=malformed_map)

    def malformed_batch_map(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        del function, items
        return ((object(),),)

    with pytest.raises(ModelError, match="malformed batch"):
        _run_cell(process_map=malformed_batch_map)

    def noniterable_map(function: Any, items: Any, **_: Any) -> Any:
        del function, items
        return 4

    with pytest.raises(ModelError, match="malformed results"):
        _run_cell(process_map=noniterable_map)


def test_strict_canonical_kernel_refuses_an_injected_process_map() -> None:
    def injected_map(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        return tuple(function(item) for item in items)

    returns, states, links = _monthly_source()
    with pytest.raises(ModelError, match="does not accept an injected process map"):
        run_registered_kernel_calibration_cell(
            model=_model(),
            master_seed=987,
            scientific_version=1,
            artifact="design",
            frequency="monthly",
            selected_length=2,
            historical_returns=returns,
            historical_source_states=states,
            source_continuation_links=links,
            model_artifact_fingerprint=MODEL_FP,
            target_history_fingerprint=TARGET_FP,
            process_map=injected_map,
        )


def test_verification_disagreement_is_retained_as_a_hard_failure() -> None:
    def corrupt_direct_sums(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        groups = tuple(function(item) for item in items)
        if items and items[0].verification_offsets is not None:
            changed: list[object] = []
            for group in groups:
                outputs = []
                for output in group:
                    assert output.directly_adjusted_sums is not None
                    values = output.directly_adjusted_sums.copy()
                    values += 1.0
                    values.setflags(write=False)
                    outputs.append(replace(output, directly_adjusted_sums=values))
                changed.append(tuple(outputs))
            return tuple(changed)
        return groups

    result = _run_cell(process_map=corrupt_direct_sums)

    assert result.precision_passed
    assert not result.verification_passed
    assert not result.passed
    assert result.failure_reason == "purpose_8_offset_verification_failed"


def test_verification_requires_direct_replay_sums() -> None:
    def omit_direct_sums(function: Any, items: Any, **_: Any) -> tuple[object, ...]:
        groups = tuple(function(item) for item in items)
        if items and items[0].verification_offsets is not None:
            return tuple(
                tuple(replace(output, directly_adjusted_sums=None) for output in group)
                for group in groups
            )
        return groups

    with pytest.raises(ModelError, match="omitted direct adjusted sums"):
        _run_cell(process_map=omit_direct_sums)


def test_calibration_and_streamless_verification_use_only_purpose_7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[CalibrationPurpose] = []
    original = kernel_execution.calibration_rng_key

    def recording_key(**kwargs: Any) -> Any:
        observed.append(CalibrationPurpose(kwargs["purpose"]))
        return original(**kwargs)

    monkeypatch.setattr(kernel_execution, "calibration_rng_key", recording_key)
    result = _run_cell()

    assert result.passed
    assert observed
    assert set(observed) == {CalibrationPurpose.KERNEL_ESTIMATION}


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"prefix_window_counts": (4, 4)}, "increasing"),
        ({"prefix_window_counts": ()}, "increasing"),
        ({"prefix_window_counts": (320_001,)}, "registered"),
        ({"annualized_half_width_max": 0.0}, "positive float"),
        ({"annualized_half_width_max": 1}, "positive float"),
        ({"confidence": 1.0}, "float in"),
        ({"confidence": 1}, "float in"),
        ({"lambdas": (0.5, 0.25)}, "increasing"),
        ({"lambdas": ("bad",)}, "numeric"),
        ({"lambdas": (0.25, 0.5, 1.0)}, "exactly lambdas"),
        ({"workers": 0}, "positive integer"),
        ({"workers": True}, "positive integer"),
        ({"statistics_batch_windows": 0}, "positive integer"),
        ({"process_chunksize": 0}, "positive integer"),
        ({"verify": 1}, "boolean"),
    ],
)
def test_execution_control_validation(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        _run_cell(**updates)


def test_strict_canonical_rejects_any_reduced_control_before_execution() -> None:
    with pytest.raises(ModelError, match="strict canonical kernel controls"):
        _run_cell(strict_canonical=True)
    with pytest.raises(ModelError, match="strict canonical kernel controls"):
        _run_cell(
            strict_canonical=True,
            prefix_window_counts=CANONICAL_PREFIX_WINDOW_COUNTS,
            statistics_batch_windows=64,
        )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"first_window_index": -1}, "non-negative"),
        ({"window_count": 0}, "positive"),
        ({"first_window_index": 319_999, "window_count": 2}, "registered"),
        ({"selected_length": 1}, "hard bounds"),
        ({"model_artifact_fingerprint": "bad"}, "SHA-256"),
        ({"frequency": "weekly"}, "frequency"),
        ({"artifact": "other"}, "artifact"),
    ],
)
def test_registered_batch_input_validation(
    updates: dict[str, object], message: str
) -> None:
    returns, states, links = _monthly_source()
    kwargs: dict[str, object] = {
        "model": _model(),
        "master_seed": 987,
        "scientific_version": 1,
        "artifact": "design",
        "frequency": "monthly",
        "selected_length": 2,
        "historical_returns": returns,
        "historical_source_states": states,
        "source_continuation_links": links,
        "model_artifact_fingerprint": MODEL_FP,
        "first_window_index": 0,
        "window_count": 2,
        "strict_canonical": False,
        "monthly_segment_lengths": (12,),
    }
    kwargs.update(updates)
    with pytest.raises(ModelError, match=message):
        registered_kernel_window_batch(**kwargs)  # type: ignore[arg-type]


def test_registered_batch_rejects_wrong_model_and_non_numeric_returns() -> None:
    with pytest.raises(ModelError, match="fitted HMM or block state model"):
        _run_cell(model=object())
    with pytest.raises(ModelError, match="must be numeric"):
        _run_cell(historical_returns=[[object(), 0.0, 0.0]])


def test_registered_batch_rejects_invalid_states_and_strict_geometry() -> None:
    returns, states, links = _monthly_source()
    bad_states = states.copy()
    bad_states[0] = 2
    with pytest.raises(ModelError, match="invalid state"):
        _run_cell(historical_source_states=bad_states)
    with pytest.raises(ModelError, match="wrong row count"):
        registered_kernel_window_batch(
            model=_model(),
            master_seed=987,
            scientific_version=1,
            artifact="design",
            frequency="monthly",
            selected_length=2,
            historical_returns=returns,
            historical_source_states=states,
            source_continuation_links=links,
            model_artifact_fingerprint=MODEL_FP,
            first_window_index=0,
            window_count=2,
            strict_canonical=True,
        )


def test_source_contract_validation() -> None:
    returns, states, links = _monthly_source()
    with pytest.raises(ModelError, match="match returns"):
        _run_cell(historical_source_states=states[:-1])
    with pytest.raises(ModelError, match="omit fitted states"):
        _run_cell(historical_source_states=np.zeros_like(states))
    with pytest.raises(ModelError, match="Boolean"):
        _run_cell(source_continuation_links=links.astype(np.int64))
    with pytest.raises(ModelError, match="shape"):
        _run_cell(historical_returns=returns[:, :2])
    changed = returns.copy()
    changed[0, 0] = np.nan
    with pytest.raises(ModelError, match="finite"):
        _run_cell(historical_returns=changed)


def _spec(
    artifact: str,
    frequency: str,
) -> KernelCalibrationCellSpec:
    if frequency == "monthly":
        returns, states, links = _monthly_source()
        daily = None
    else:
        returns, states, links = _daily_source()
        daily = (240,)
    return KernelCalibrationCellSpec(
        artifact=artifact,  # type: ignore[arg-type]
        frequency=frequency,  # type: ignore[arg-type]
        model=_model(),
        selected_length=2,
        historical_returns=returns,
        historical_source_states=states,
        source_continuation_links=links,
        model_artifact_fingerprint=("a" if artifact == "design" else "c") * 64,
        target_history_fingerprint=("b" if artifact == "design" else "d") * 64,
        monthly_segment_lengths=(12,),
        daily_segment_lengths=daily,
    )


def test_four_cell_suite_has_exact_order_and_all_gate_semantics() -> None:
    specs = (
        _spec("design", "monthly"),
        _spec("design", "daily"),
        _spec("production", "monthly"),
        _spec("production", "daily"),
    )
    result = run_registered_kernel_calibration_suite(
        specs,
        master_seed=987,
        scientific_version=1,
        workers=1,
        prefix_window_counts=(4,),
        annualized_half_width_max=100.0,
        confidence=0.95,
        lambdas=(0.25, 0.5, 0.75, 1.0),
        statistics_batch_windows=2,
        strict_canonical=False,
    )

    assert result.passed
    assert result.all_precision_passed
    assert result.all_verification_passed
    assert tuple((cell.artifact, cell.frequency) for cell in result.cells) == (
        ("design", "monthly"),
        ("design", "daily"),
        ("production", "monthly"),
        ("production", "daily"),
    )


def test_four_cell_suite_rejects_missing_or_reordered_cells() -> None:
    specs = (
        _spec("design", "daily"),
        _spec("design", "monthly"),
        _spec("production", "monthly"),
        _spec("production", "daily"),
    )
    with pytest.raises(ModelError, match="exact design-monthly"):
        run_registered_kernel_calibration_suite(
            specs,
            master_seed=987,
            scientific_version=1,
            strict_canonical=False,
        )


def test_strict_four_cell_suite_rejects_different_state_counts() -> None:
    three_state = BlockStateModel(
        stationary_distribution=np.asarray([1.0 / 3.0] * 3),
        transition_matrix=np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        ),
    )
    specs = [
        _spec("design", "monthly"),
        _spec("design", "daily"),
        _spec("production", "monthly"),
        _spec("production", "daily"),
    ]
    specs[3] = replace(specs[3], model=three_state)
    with pytest.raises(ModelError, match="same K"):
        run_registered_kernel_calibration_suite(
            specs,
            master_seed=987,
            scientific_version=1,
            strict_canonical=True,
        )


def test_strict_four_cell_suite_requires_one_model_per_artifact() -> None:
    specs = [
        _spec("design", "monthly"),
        _spec("design", "daily"),
        _spec("production", "monthly"),
        _spec("production", "daily"),
    ]
    specs[1] = replace(specs[1], model_artifact_fingerprint="e" * 64)
    with pytest.raises(ModelError, match="share one artifact model"):
        run_registered_kernel_calibration_suite(
            specs,
            master_seed=987,
            scientific_version=1,
            strict_canonical=True,
        )


def _canonical_spec(artifact: str, frequency: str) -> KernelCalibrationCellSpec:
    monthly_lengths = (193,) if artifact == "design" else (210, 7)
    daily_lengths = (4_049,) if artifact == "design" else (4_404, 143)
    monthly_states = np.arange(sum(monthly_lengths), dtype=np.int64) % 2
    if frequency == "monthly":
        states = monthly_states
        segment_lengths = monthly_lengths
        daily: tuple[int, ...] | None = None
    else:
        expanded: list[np.ndarray] = []
        offset = 0
        for monthly_length, daily_length in zip(
            monthly_lengths, daily_lengths, strict=True
        ):
            expanded.append(
                np.repeat(monthly_states[offset : offset + monthly_length], 21)[
                    :daily_length
                ]
            )
            offset += monthly_length
        states = np.concatenate(expanded)
        segment_lengths = daily_lengths
        daily = daily_lengths
    links = np.ones(states.size - 1, dtype=np.bool_)
    offset = 0
    for length in segment_lengths[:-1]:
        offset += length
        links[offset - 1] = False
    returns = np.column_stack(
        (
            np.arange(states.size, dtype=np.float64) * 1e-8,
            states.astype(np.float64) * 1e-5,
            -np.arange(states.size, dtype=np.float64) * 1e-8,
        )
    )
    return KernelCalibrationCellSpec(
        artifact=artifact,  # type: ignore[arg-type]
        frequency=frequency,  # type: ignore[arg-type]
        model=_model(),
        selected_length=2,
        historical_returns=returns,
        historical_source_states=states,
        source_continuation_links=links,
        model_artifact_fingerprint=("a" if artifact == "design" else "c") * 64,
        target_history_fingerprint=("b" if artifact == "design" else "d") * 64,
        monthly_segment_lengths=monthly_lengths,
        daily_segment_lengths=daily,
    )


def test_strict_four_cell_suite_rejects_nonexpanded_daily_source_states() -> None:
    specs = [
        _canonical_spec("design", "monthly"),
        _canonical_spec("design", "daily"),
        _canonical_spec("production", "monthly"),
        _canonical_spec("production", "daily"),
    ]
    changed = np.asarray(specs[1].historical_source_states).copy()
    changed[10] = 1 - changed[10]
    specs[1] = replace(specs[1], historical_source_states=changed)
    with pytest.raises(ModelError, match="exact 21-session expansion"):
        run_registered_kernel_calibration_suite(
            specs,
            master_seed=987,
            scientific_version=1,
            strict_canonical=True,
        )


def test_batch_validator_rejects_type_range_statistics_and_frequency() -> None:
    first = _batch(0, 2)
    with pytest.raises(ModelError, match="wrong type"):
        merge_registered_kernel_batches((object(),))  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="invalid window range"):
        merge_registered_kernel_batches(
            (replace(first, first_window_index=1, stop_window_index=1),)
        )
    inconsistent = replace(first.statistics, windows=1)
    with pytest.raises(ModelError, match="statistics range"):
        merge_registered_kernel_batches((replace(first, statistics=inconsistent),))
    changed_frequency = replace(first.statistics, frequency="daily")
    with pytest.raises(ModelError, match="frequency"):
        merge_registered_kernel_batches((replace(first, statistics=changed_frequency),))


def test_target_fingerprint_is_validated() -> None:
    with pytest.raises(ModelError, match="target_history_fingerprint"):
        _run_cell(target_history_fingerprint="bad")


def test_kernel_memory_estimate_is_exact_and_side_effect_free() -> None:
    estimate = registered_kernel_memory_estimate(
        artifact="design",
        frequency="monthly",
        workers=3,
        batch_windows=128,
        verification=True,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    expected = (
        128 * 36 + 128 * 36 + 128 * 12 * 3 + 2 * 128 * 12 * 3
    ) * 8 + 128 * 2 * 12
    assert estimate.per_worker_bytes == expected
    assert estimate.all_workers_bytes == 3 * expected
    assert estimate.verification

    without_verification = registered_kernel_memory_estimate(
        artifact="design",
        frequency="monthly",
        workers=1,
        batch_windows=128,
        verification=False,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    assert without_verification.per_worker_bytes < estimate.per_worker_bytes
    with pytest.raises(ModelError, match="verification must be boolean"):
        registered_kernel_memory_estimate(
            artifact="design",
            frequency="monthly",
            workers=1,
            verification=1,  # type: ignore[arg-type]
        )
