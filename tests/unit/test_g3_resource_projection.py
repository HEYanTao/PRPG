from __future__ import annotations

import dataclasses
import math
from typing import Any, cast

import pytest

from prpg.errors import ModelError
from prpg.model.g3_preflight import (
    CANONICAL_PHYSICAL_MEMORY_BYTES,
    canonical_g3_resource_estimate,
)
from prpg.model.g3_resource_projection import (
    G3ResourceProjectionMeasurements,
    project_canonical_g3_resources,
)

_GIB = 1 << 30


def _measurements(
    *,
    hmm_wall_seconds_by_k: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
    peak_process_tree_rss_bytes: int = 1 << 20,
    disk_free_bytes: int = 64 * _GIB,
    serialized_bytes: int = 1 << 20,
    bridge_benchmark_n_states: int = 4,
    utilization: float = 0.5,
    native_thread_cap_verified: bool = True,
    worker_parity_verified: bool = True,
    rss_telemetry_complete: bool = True,
    swap_telemetry_complete: bool = True,
    cpu_telemetry_complete: bool = True,
    disk_telemetry_complete: bool = True,
) -> G3ResourceProjectionMeasurements:
    block_materialization = 5.0
    block_calls = (6.0, 7.0, 8.0, 9.0)
    kernel = (10.0, 11.0, 12.0, 13.0)
    dependence = 14.0
    latent_generation = (15.0, 16.0)
    latent_bootstrap = (17.0, 18.0)
    bridge_benchmark = 20.0
    bridge_qmax = 19.0
    lightweight = 21.0
    observed_wall = (
        sum(hmm_wall_seconds_by_k)
        + block_materialization
        + sum(block_calls)
        + sum(kernel)
        + dependence
        + sum(latent_generation)
        + sum(latent_bootstrap)
        + bridge_benchmark
        + lightweight
    )
    return G3ResourceProjectionMeasurements(
        workers=9,
        hmm_wall_seconds_by_k=hmm_wall_seconds_by_k,
        block_materialization_wall_seconds=block_materialization,
        block_call_wall_seconds=block_calls,
        kernel_wall_seconds=kernel,
        dependence_wall_seconds=dependence,
        latent_generation_wall_seconds=latent_generation,
        latent_bootstrap_wall_seconds=latent_bootstrap,
        bridge_benchmark_wall_seconds=bridge_benchmark,
        bridge_qmax_seconds=bridge_qmax,
        bridge_benchmark_n_states=bridge_benchmark_n_states,
        lightweight_wall_seconds=lightweight,
        peak_process_tree_rss_bytes=peak_process_tree_rss_bytes,
        process_tree_cpu_seconds=utilization * 9 * observed_wall,
        worker_capacity_utilization=utilization,
        swap_used_before_bytes=0,
        maximum_swap_used_bytes=128,
        swap_used_after_bytes=64,
        disk_free_bytes=disk_free_bytes,
        serialized_bytes=serialized_bytes,
        native_thread_cap_verified=native_thread_cap_verified,
        worker_parity_verified=worker_parity_verified,
        rss_telemetry_complete=rss_telemetry_complete,
        swap_telemetry_complete=swap_telemetry_complete,
        cpu_telemetry_complete=cpu_telemetry_complete,
        disk_telemetry_complete=disk_telemetry_complete,
    )


def test_exact_code_owned_projection_algebra_and_safe_result() -> None:
    measurements = _measurements()

    result = project_canonical_g3_resources(measurements)
    families = {item.family: item for item in result.family_projections}

    assert families["nonbridge_hmm"].projected_seconds == pytest.approx(
        1.50 * (sum((1.0, 2.0, 3.0, 4.0)) + 3.0 * ((5_300 + 105_050) / 50))
    )
    assert families["nonbridge_hmm"].formula_id == (
        "1.50*(sum(K2..K5_main_50)+K4*(5300+105050)/50)"
    )
    assert families["block_materialization"].projected_seconds == pytest.approx(
        1.35 * (10_000 / 128) * 5.0
    )
    assert families["block_calls"].projected_seconds == pytest.approx(
        1.35 * (10_000 / 128) * (5 * 6.0 + 5 * 7.0 + 8.0 + 9.0)
    )
    assert families["block_calls"].formula_id == (
        "1.35*(10000/128)*(5*design_monthly+5*design_daily+production)"
    )
    assert families["kernel"].projected_seconds == pytest.approx(
        1.35 * ((280 + 278) / 2) * sum((10.0, 11.0, 12.0, 13.0))
    )
    assert families["dependence"].projected_seconds == pytest.approx(
        1.35 * (10_000 / 128) * 14.0
    )
    assert families["latent_generation"].projected_seconds == pytest.approx(
        1.35 * 2 * 16.0 * (10_000 / 512)
    )
    assert families["latent_bootstrap"].projected_seconds == pytest.approx(
        1.50 * 2 * 18.0 * ((10_000 * 10_000) / (512 * 512))
    )
    assert families["bridge_benchmark"].projected_seconds == 1.25 * 20.0
    assert families["bridge_benchmark"].formula_id == (
        "1.25*measured_K4_100_coordinate_total"
    )
    assert families["bridge_remaining"].projected_seconds == pytest.approx(
        19.0 * 21_146 / 9
    )
    assert families["lightweight"].projected_seconds == 900.0
    assert result.bridge_pair_units == 21_146
    assert result.bridge_benchmark_n_states == 4
    assert result.observed_total_wall_seconds == 212.0
    assert result.projected_total_seconds == pytest.approx(
        1.10 * sum(item.projected_seconds for item in result.family_projections)
    )
    assert result.whole_graph_contingency_seconds == pytest.approx(
        result.projected_total_seconds - result.projected_subtotal_seconds
    )
    code_estimate = canonical_g3_resource_estimate()
    assert result.projected_peak_rss_bytes == code_estimate.estimated_peak_bytes
    assert result.projected_disk_bytes == code_estimate.estimated_disk_bytes
    assert result.serialized_extrapolation_factor == 348
    assert result.required_free_disk_bytes == 3 * result.projected_disk_bytes
    assert result.passed
    assert result.reasons == ()
    assert result.generation_authorized is False


def test_bridge_resource_measurement_rejects_non_k4_benchmark() -> None:
    with pytest.raises(ModelError, match="benchmark K must be exactly 4"):
        _measurements(bridge_benchmark_n_states=5)


def test_measurements_and_projection_are_frozen() -> None:
    measurements = _measurements()
    projection = project_canonical_g3_resources(measurements)

    with pytest.raises(dataclasses.FrozenInstanceError):
        measurements.workers = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        projection.passed = False  # type: ignore[misc]


def test_observed_rss_projection_dominates_and_fails_above_seventy_percent() -> None:
    rss = math.ceil(0.71 * CANONICAL_PHYSICAL_MEMORY_BYTES / 1.25)

    result = project_canonical_g3_resources(
        _measurements(peak_process_tree_rss_bytes=rss)
    )

    assert result.projected_peak_rss_bytes == math.ceil(1.25 * rss)
    assert result.projected_memory_fraction > 0.70
    assert result.reasons == ("projected_peak_rss_exceeds_70_percent",)
    assert not result.passed


def test_serialized_extrapolation_can_dominate_code_owned_disk_floor() -> None:
    result = project_canonical_g3_resources(
        _measurements(serialized_bytes=20_000_000, disk_free_bytes=64 * _GIB)
    )

    assert result.extrapolated_serialized_bytes == 20_000_000 * 348
    assert result.projected_disk_bytes == result.extrapolated_serialized_bytes
    assert result.passed


def test_disk_requires_three_times_the_larger_projection() -> None:
    code_disk = canonical_g3_resource_estimate().estimated_disk_bytes

    result = project_canonical_g3_resources(
        _measurements(disk_free_bytes=3 * code_disk - 1)
    )

    assert result.reasons == ("free_disk_below_three_times_projection",)
    assert not result.passed


def test_projection_fails_the_fixed_72_hour_wall_limit() -> None:
    result = project_canonical_g3_resources(
        _measurements(hmm_wall_seconds_by_k=(100.0, 100.0, 100.0, 100.0))
    )

    assert result.projected_total_seconds > 72 * 60 * 60
    assert "projected_wall_clock_exceeds_72_hours" in result.reasons
    assert not result.passed


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("native_thread_cap_verified", "native_thread_cap_not_verified"),
        ("worker_parity_verified", "worker_parity_not_verified"),
        ("rss_telemetry_complete", "rss_telemetry_incomplete"),
        ("swap_telemetry_complete", "swap_telemetry_incomplete"),
        ("cpu_telemetry_complete", "cpu_telemetry_incomplete"),
        ("disk_telemetry_complete", "disk_telemetry_incomplete"),
    ],
)
def test_each_required_verification_flag_fails_closed(field: str, reason: str) -> None:
    measurements = dataclasses.replace(_measurements(), **{field: False})

    result = project_canonical_g3_resources(measurements)

    assert result.reasons == (reason,)
    assert not result.passed


def test_utilization_has_no_acceptance_threshold() -> None:
    result = project_canonical_g3_resources(_measurements(utilization=1e-12))

    assert result.worker_capacity_utilization == 1e-12
    assert result.passed


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"workers": 8}, "nine workers"),
        ({"hmm_wall_seconds_by_k": (1.0, 2.0, 3.0)}, "fixed geometry"),
        ({"hmm_wall_seconds_by_k": (1.0, 2.0, 3.0, math.inf)}, "positive finite"),
        ({"dependence_wall_seconds": 0.0}, "positive finite"),
        ({"bridge_qmax_seconds": 21.0}, "cannot exceed"),
        ({"bridge_benchmark_n_states": 6}, "benchmark K"),
        ({"serialized_bytes": 0}, "positive integer"),
        ({"swap_used_before_bytes": -1}, "non-negative integer"),
        ({"maximum_swap_used_bytes": 32}, "maximum swap"),
        ({"native_thread_cap_verified": 1}, "must be boolean"),
    ],
)
def test_malformed_or_substituted_measurements_are_rejected(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        dataclasses.replace(_measurements(), **changes)


def test_list_cannot_substitute_for_fixed_timing_tuple() -> None:
    substituted = cast(Any, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ModelError, match="fixed geometry"):
        dataclasses.replace(_measurements(), hmm_wall_seconds_by_k=substituted)


def test_inconsistent_cpu_utilization_is_rejected() -> None:
    with pytest.raises(ModelError, match="inconsistent with CPU"):
        dataclasses.replace(_measurements(), worker_capacity_utilization=0.4)


def test_projection_revalidates_a_tampered_frozen_measurement() -> None:
    measurements = _measurements()
    object.__setattr__(measurements, "serialized_bytes", 0)

    with pytest.raises(ModelError, match="serialized bytes"):
        project_canonical_g3_resources(measurements)


def test_projection_rejects_a_substituted_measurement_type() -> None:
    substituted = cast(Any, object())

    with pytest.raises(ModelError, match="exact measurement type"):
        project_canonical_g3_resources(substituted)
