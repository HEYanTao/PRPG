from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from prpg.errors import ValidationError
from prpg.simulation.rng import Family
from prpg.validation.outer_null import (
    OUTER_NULL_STATISTIC_NAMES,
    OuterBootstrapReason,
    calibrate_outer_null,
    compute_outer_null_base_max_statistics,
    estimate_outer_null_peak_memory_bytes,
    project_outer_null_resources,
    registered_outer_null_pair,
    summarize_outer_null_statistics,
    unconditioned_stationary_bootstrap_trace,
)
from prpg.validation.statistics import OUTER_NULL_REPLICATES


def _monthly_source() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(314159)
    source = generator.normal(
        loc=(0.006, 0.002, 0.003),
        scale=(0.04, 0.012, 0.018),
        size=(30, 3),
    )
    links = np.ones(29, dtype=np.bool_)
    links[13] = False
    return source, links


def _daily_source() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(271828)
    source = generator.normal(
        loc=(0.0003, 0.00008, 0.00012),
        scale=(0.012, 0.003, 0.005),
        size=(80, 3),
    )
    links = np.ones(79, dtype=np.bool_)
    links[39] = False
    return source, links


def test_unconditioned_trace_obeys_restart_precedence_and_never_wraps() -> None:
    trace = unconditioned_stationary_bootstrap_trace(
        5,
        np.asarray((True, False, True, True), dtype=np.bool_),
        np.asarray((0.9, 0.9, 0.9, 0.9, 0.1)),
        np.asarray((0.21, 0.65, 0.20, 0.00, 0.41)),
        mean_block_length=5.0,
    )

    assert trace.source_indices.tolist() == [1, 3, 4, 0, 2]
    assert trace.restart_flags.tolist() == [True, True, False, True, True]
    assert trace.reasons == (
        OuterBootstrapReason.FIRST_OBSERVATION,
        OuterBootstrapReason.SOURCE_GAP,
        OuterBootstrapReason.CONTINUATION,
        OuterBootstrapReason.SOURCE_END,
        OuterBootstrapReason.RANDOM_RESTART,
    )
    assert trace.restart_uniforms_consumed == 5
    assert trace.source_uniforms_consumed == 5
    assert not trace.source_indices.flags.writeable
    assert not trace.restart_flags.flags.writeable


def test_registered_pair_is_reproducible_independent_and_gap_safe() -> None:
    links = np.asarray((True, True, False, True, True), dtype=np.bool_)
    first = registered_outer_null_pair(
        6,
        links,
        output_observations=200,
        mean_block_length=20.0,
        master_seed=20260714,
        scientific_version=11,
        family=Family.LF,
        replicate=7,
    )
    repeated = registered_outer_null_pair(
        6,
        links,
        output_observations=200,
        mean_block_length=20.0,
        master_seed=20260714,
        scientific_version=11,
        family=Family.LF,
        replicate=7,
    )

    assert np.array_equal(
        first.first_trace.source_indices,
        repeated.first_trace.source_indices,
    )
    assert np.array_equal(
        first.second_trace.source_indices,
        repeated.second_trace.source_indices,
    )
    assert not np.array_equal(
        first.first_trace.source_indices,
        first.second_trace.source_indices,
    )
    for trace in (first.first_trace, first.second_trace):
        apparent_crossings = np.flatnonzero(
            (trace.source_indices[:-1] == 2) & (trace.source_indices[1:] == 3)
        )
        for previous_position in apparent_crossings:
            position = int(previous_position) + 1
            assert trace.restart_flags[position]
            assert trace.reasons[position] is OuterBootstrapReason.SOURCE_GAP


def test_reduced_calibration_is_reproducible_and_non_authorizing() -> None:
    source, links = _monthly_source()
    first = calibrate_outer_null(
        source,
        links,
        frequency="monthly",
        mean_block_length=4.0,
        master_seed=20260714,
        scientific_version=11,
        replicate_count=6,
        workers=1,
        batch_size=2,
    )
    second = calibrate_outer_null(
        source,
        links,
        frequency="monthly",
        mean_block_length=4.0,
        master_seed=20260714,
        scientific_version=11,
        replicate_count=6,
        workers=1,
        batch_size=3,
    )

    assert first.replicate_statistics.shape == (6, len(OUTER_NULL_STATISTIC_NAMES))
    assert np.array_equal(first.replicate_statistics, second.replicate_statistics)
    assert first.statistics_sha256 == second.statistics_sha256
    assert first.fingerprint == second.fingerprint
    assert not first.fixed_sample_canonical
    assert first.canonical_thresholds is None
    assert not first.sealed_g5_publication
    assert not first.generation_authorized
    assert not first.replicate_statistics.flags.writeable
    assert all(item.sample_count == 6 for item in first.critical_values)

    direct = compute_outer_null_base_max_statistics(
        source,
        source,
        frequency="monthly",
    )
    assert direct.statistic_names == OUTER_NULL_STATISTIC_NAMES
    assert direct.as_mapping() == dict.fromkeys(OUTER_NULL_STATISTIC_NAMES, 0.0)


def test_outer_null_statistics_are_identical_across_worker_counts() -> None:
    source, links = _monthly_source()
    serial = calibrate_outer_null(
        source,
        links,
        frequency="monthly",
        mean_block_length=3.0,
        master_seed=20260714,
        scientific_version=11,
        replicate_count=4,
        workers=1,
        batch_size=2,
    )
    # macOS sandbox compatibility: -1 is the documented indeterminate
    # semaphore-capacity response and leaves actual spawn behavior intact.
    with patch("concurrent.futures.process.os.sysconf", return_value=-1):
        parallel = calibrate_outer_null(
            source,
            links,
            frequency="monthly",
            mean_block_length=3.0,
            master_seed=20260714,
            scientific_version=11,
            replicate_count=4,
            workers=2,
            batch_size=2,
        )

    assert np.array_equal(serial.replicate_statistics, parallel.replicate_statistics)
    assert serial.statistics_sha256 == parallel.statistics_sha256
    assert serial.fingerprint == parallel.fingerprint


def test_exact_50000_summary_uses_strict_records_and_remains_unsealed() -> None:
    base = np.arange(OUTER_NULL_REPLICATES, dtype=np.float64)[:, None]
    statistics = np.concatenate(
        tuple(base / (index + 1.0) for index in range(len(OUTER_NULL_STATISTIC_NAMES))),
        axis=1,
    )
    result = summarize_outer_null_statistics(
        statistics,
        frequency="daily",
        observations=80,
        mean_block_length=20.0,
        workers=9,
        batch_size=512,
        source_sha256="a" * 64,
        wall_seconds=12.5,
        master_seed=20260714,
        scientific_version=11,
        threshold_set_id="raw-daily-v3",
        binding_fingerprints={"historical_source": "b" * 64},
        sobol_direction_fingerprint="c" * 64,
    )

    assert result.replicate_count == OUTER_NULL_REPLICATES
    assert result.fixed_sample_canonical
    assert result.canonical_thresholds is not None
    assert len(result.canonical_thresholds.thresholds) == len(
        OUTER_NULL_STATISTIC_NAMES
    )
    assert len(result.canonical_thresholds.audit_intervals) == 4
    assert result.canonical_thresholds.precision_passed
    assert all(
        item.sample_count == OUTER_NULL_REPLICATES and item.one_based_rank == 47_501
        for item in result.critical_values
    )
    assert not result.sealed_g5_publication
    assert not result.generation_authorized


def test_canonical_metadata_is_required_before_expensive_execution() -> None:
    source, links = _monthly_source()
    with (
        patch("prpg.validation.outer_null.deterministic_process_map") as process_map,
        pytest.raises(ValidationError, match="canonical outer-null"),
    ):
        calibrate_outer_null(
            source,
            links,
            frequency="monthly",
            mean_block_length=4.0,
            master_seed=20260714,
            scientific_version=11,
            replicate_count=OUTER_NULL_REPLICATES,
            workers=9,
        )
    process_map.assert_not_called()


def test_unexpected_worker_failure_aborts_without_a_partial_result() -> None:
    source, links = _monthly_source()
    with (
        patch(
            "prpg.validation.outer_null._run_outer_null_chunk",
            side_effect=RuntimeError("unexpected failure"),
        ),
        pytest.raises(RuntimeError, match="unexpected failure"),
    ):
        calibrate_outer_null(
            source,
            links,
            frequency="monthly",
            mean_block_length=4.0,
            master_seed=20260714,
            scientific_version=11,
            replicate_count=2,
            workers=1,
        )


def test_resource_projection_is_bounded_and_scales_from_measured_pilot() -> None:
    values = np.ones((10, len(OUTER_NULL_STATISTIC_NAMES)), dtype=np.float64)
    pilot = summarize_outer_null_statistics(
        values,
        frequency="monthly",
        observations=30,
        mean_block_length=4.0,
        workers=9,
        batch_size=10,
        source_sha256="a" * 64,
        wall_seconds=2.0,
        master_seed=20260714,
        scientific_version=11,
    )
    projection = project_outer_null_resources(
        pilot,
        target_replicates=OUTER_NULL_REPLICATES,
        physical_memory_bytes=16 * 1024**3,
    )

    assert projection.target_is_canonical_count
    assert projection.projected_wall_seconds == pytest.approx(10_000.0)
    assert projection.projected_wall_hours == pytest.approx(10_000.0 / 3600.0)
    assert projection.memory_passed
    assert projection.estimated_memory_fraction < 0.70
    assert not projection.generation_authorized
    assert (
        projection.estimated_peak_memory_bytes
        == estimate_outer_null_peak_memory_bytes(
            observations=30,
            replicate_count=OUTER_NULL_REPLICATES,
            workers=9,
        )
    )


def test_reduced_daily_performance_and_finite_output() -> None:
    source, links = _daily_source()
    started = time.monotonic()
    result = calibrate_outer_null(
        source,
        links,
        frequency="daily",
        mean_block_length=20.0,
        master_seed=20260714,
        scientific_version=11,
        replicate_count=8,
        workers=1,
        batch_size=4,
    )
    elapsed = time.monotonic() - started

    assert np.isfinite(result.replicate_statistics).all()
    assert elapsed < 5.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nan", "finite"),
        ("constant", "constant asset"),
        ("links", "continuation links"),
        ("short_daily", "at least 61"),
        ("workers", "cannot exceed"),
    ],
)
def test_outer_null_inputs_fail_closed(mutation: str, message: str) -> None:
    source, links = _daily_source()
    replicate_count = 2
    workers = 1
    if mutation == "nan":
        source[0, 0] = np.nan
    elif mutation == "constant":
        source[:, 0] = 0.0
    elif mutation == "links":
        links = links[:-1]
    elif mutation == "short_daily":
        source = source[:60]
        links = links[:59]
    else:
        workers = 3

    with pytest.raises(ValidationError, match=message):
        calibrate_outer_null(
            source,
            links,
            frequency="daily",
            mean_block_length=20.0,
            master_seed=20260714,
            scientific_version=11,
            replicate_count=replicate_count,
            workers=workers,
        )
