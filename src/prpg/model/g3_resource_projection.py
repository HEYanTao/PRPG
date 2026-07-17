"""Pure full-graph resource projection from the reduced G3 rehearsal.

This module deliberately owns no rehearsal orchestration, filesystem access,
canonical scientific execution, or generation authority.  It accepts one
frozen bundle of measurements at the only approved reduced geometry and
projects the complete canonical G3 graph with code-owned counts, safety
factors, and limits.  A projection is evidence for the launch gate only; even
a passing result can never authorize Phase-4 or Phase-5 generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

from prpg.errors import ModelError
from prpg.model.adequacy import (
    LATENT_BOOTSTRAP_REPLICATES,
    LATENT_BOOTSTRAP_RESAMPLE_SIZE,
    LATENT_CHAIN_COUNT,
)
from prpg.model.block_execution import CANONICAL_HISTORY_COUNT
from prpg.model.bridge import (
    ACCELERATOR_AUDIT_PAIRS,
    TIER_A_PAIRS,
    TIER_B_MAXIMUM_PAIRS,
)
from prpg.model.bridge_driver import (
    BRIDGE_BENCHMARK_RESTARTS_PER_PAIR,
    BRIDGE_DGPS,
    BRIDGE_ORDINARY_RESTARTS_PER_PAIR,
    BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR,
    BRIDGE_TIER_A_RESTARTS_PER_PAIR,
)
from prpg.model.calibration import MACRO_STABILITY_ATTEMPTS, VERSION_1_CANDIDATES
from prpg.model.fixed_point import MAXIMUM_FIXED_POINT_ITERATIONS
from prpg.model.g3 import G3_RESOURCE_PROJECTION_VERSION
from prpg.model.g3_preflight import (
    CANONICAL_DISK_SAFETY_MULTIPLIER,
    CANONICAL_PHYSICAL_MEMORY_BYTES,
    CANONICAL_WORKERS,
    canonical_g3_resource_estimate,
)
from prpg.model.hmm import HMM_RESTARTS
from prpg.model.kernel_execution import (
    CANONICAL_PREFIX_WINDOW_COUNTS,
    CANONICAL_STATISTICS_BATCH_WINDOWS,
)
from prpg.model.refit_adequacy_execution import CANONICAL_REFIT_ATTEMPTS
from prpg.model.selection import ROLLING_FOLD_COUNT

_HMM_SAFETY_FACTOR: Final = 1.50
_BLOCK_SAFETY_FACTOR: Final = 1.35
_KERNEL_SAFETY_FACTOR: Final = 1.35
_DEPENDENCE_SAFETY_FACTOR: Final = 1.35
_LATENT_GENERATION_SAFETY_FACTOR: Final = 1.35
_LATENT_BOOTSTRAP_SAFETY_FACTOR: Final = 1.50
_WHOLE_GRAPH_SAFETY_FACTOR: Final = 1.10
_OBSERVED_RSS_SAFETY_FACTOR: Final = 1.25
_SERIALIZED_BYTES_SAFETY_FACTOR: Final = 1.25

_MAXIMUM_WALL_SECONDS: Final = 72.0 * 60.0 * 60.0
_MAXIMUM_MEMORY_FRACTION: Final = 0.70
_LIGHTWEIGHT_MINIMUM_SECONDS: Final = 15.0 * 60.0
_LIGHTWEIGHT_SAFETY_FACTOR: Final = 2.0

# The geometry below is part of this module's contract, not caller input.
_BLOCK_SAMPLE_HISTORIES: Final = 128
_KERNEL_SAMPLE_WINDOWS: Final = 1_152
_LATENT_SAMPLE_CHAINS: Final = 512
_LATENT_SAMPLE_BOOTSTRAP_REPLICATES: Final = 512
_LATENT_SAMPLE_BOOTSTRAP_RESAMPLE_SIZE: Final = 512
_BRIDGE_SAMPLE_COORDINATES: Final = 100

_OWNER_FIXED_K4_INDEX: Final = VERSION_1_CANDIDATES.index(4)
_HMM_MAIN_DIAGNOSTIC_RESTARTS: Final = len(VERSION_1_CANDIDATES) * HMM_RESTARTS
_HMM_FIXED_K4_DESIGN_STABILITY_RESTARTS: Final = HMM_RESTARTS * (
    ROLLING_FOLD_COUNT + MACRO_STABILITY_ATTEMPTS
)
_HMM_RESTARTS_SELECTED_CANDIDATE: Final = HMM_RESTARTS * (
    1 + MACRO_STABILITY_ATTEMPTS + 2 * CANONICAL_REFIT_ATTEMPTS
)
_NONBRIDGE_HMM_RESTARTS: Final = (
    _HMM_MAIN_DIAGNOSTIC_RESTARTS
    + _HMM_FIXED_K4_DESIGN_STABILITY_RESTARTS
    + _HMM_RESTARTS_SELECTED_CANDIDATE
)
assert _HMM_MAIN_DIAGNOSTIC_RESTARTS == 200
assert _HMM_FIXED_K4_DESIGN_STABILITY_RESTARTS == 5_300
assert _HMM_RESTARTS_SELECTED_CANDIDATE == 105_050
assert _NONBRIDGE_HMM_RESTARTS == 110_550

_DESIGN_BLOCK_CALLS_PER_CELL: Final = MAXIMUM_FIXED_POINT_ITERATIONS
assert _DESIGN_BLOCK_CALLS_PER_CELL == 5

_PREFIX_DELTAS: Final = tuple(
    count - (CANONICAL_PREFIX_WINDOW_COUNTS[index - 1] if index else 0)
    for index, count in enumerate(CANONICAL_PREFIX_WINDOW_COUNTS)
)


def _kernel_waves(windows: int) -> int:
    batches = math.ceil(windows / CANONICAL_STATISTICS_BATCH_WINDOWS)
    return math.ceil(batches / CANONICAL_WORKERS)


_KERNEL_PRIMARY_WAVES: Final = sum(_kernel_waves(item) for item in _PREFIX_DELTAS)
_KERNEL_VERIFICATION_WAVES: Final = _kernel_waves(CANONICAL_PREFIX_WINDOW_COUNTS[-1])
_KERNEL_SAMPLE_WAVES: Final = 2
_KERNEL_WAVE_FACTOR: Final = (
    _KERNEL_PRIMARY_WAVES + _KERNEL_VERIFICATION_WAVES
) / _KERNEL_SAMPLE_WAVES
assert _KERNEL_PRIMARY_WAVES == 280
assert _KERNEL_VERIFICATION_WAVES == 278
assert _KERNEL_WAVE_FACTOR == 279.0

_BRIDGE_TIER_A_RESTARTS: Final = (
    len(BRIDGE_DGPS) * TIER_A_PAIRS * BRIDGE_TIER_A_RESTARTS_PER_PAIR
)
_BRIDGE_TIER_B_WORST_RESTARTS: Final = (
    len(BRIDGE_DGPS)
    * (TIER_B_MAXIMUM_PAIRS - ACCELERATOR_AUDIT_PAIRS)
    * BRIDGE_ORDINARY_RESTARTS_PER_PAIR
)
assert BRIDGE_TIER_A_RESTARTS_PER_PAIR == 100
assert _BRIDGE_TIER_A_RESTARTS == 50_000
assert _BRIDGE_TIER_B_WORST_RESTARTS == 1_980_000


def _bridge_pair_units(benchmark_n_states: int) -> int:
    """Convert exact fixed-K4 bridge work to measured-pair units."""

    if type(benchmark_n_states) is not int or benchmark_n_states != 4:
        raise ModelError("version-3 bridge benchmark K must be exactly 4")
    return math.ceil(
        BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR
        * (_BRIDGE_TIER_A_RESTARTS + _BRIDGE_TIER_B_WORST_RESTARTS)
        / BRIDGE_BENCHMARK_RESTARTS_PER_PAIR
    )


assert _bridge_pair_units(4) == 21_146

# Serialized size is projected by the largest canonical-to-rehearsal geometry
# ratio, then given its own 25% margin.  The maximum is the kernel's 320,000
# windows relative to the fixed 1,152-window rehearsal; the explicit formula
# keeps later geometry changes from silently weakening the disk bound.
_SERIALIZED_EXTRAPOLATION_FACTOR: Final = math.ceil(
    _SERIALIZED_BYTES_SAFETY_FACTOR
    * max(
        CANONICAL_HISTORY_COUNT / _BLOCK_SAMPLE_HISTORIES,
        CANONICAL_PREFIX_WINDOW_COUNTS[-1] / _KERNEL_SAMPLE_WINDOWS,
        LATENT_CHAIN_COUNT / _LATENT_SAMPLE_CHAINS,
        LATENT_BOOTSTRAP_REPLICATES / _LATENT_SAMPLE_BOOTSTRAP_REPLICATES,
        TIER_A_PAIRS / _BRIDGE_SAMPLE_COORDINATES,
        TIER_B_MAXIMUM_PAIRS / _BRIDGE_SAMPLE_COORDINATES,
    )
)
assert _SERIALIZED_EXTRAPOLATION_FACTOR == 348

HMMWallTimes: TypeAlias = tuple[float, float, float, float]
FourCellWallTimes: TypeAlias = tuple[float, float, float, float]
RoleWallTimes: TypeAlias = tuple[float, float]
G3ResourceFamily: TypeAlias = Literal[
    "nonbridge_hmm",
    "block_materialization",
    "block_calls",
    "kernel",
    "dependence",
    "latent_generation",
    "latent_bootstrap",
    "bridge_benchmark",
    "bridge_remaining",
    "lightweight",
]


@dataclass(frozen=True, slots=True)
class G3ResourceProjectionMeasurements:
    """Complete measurement bundle at the fixed reduced-rehearsal geometry.

    Tuple orders are binding:

    * ``hmm_wall_seconds_by_k`` is K=2, 3, 4, 5, each with exactly 50 starts;
    * block-call and kernel tuples are design-monthly, design-daily,
      production-monthly, production-daily;
    * latent tuples are design, production.

    ``worker_capacity_utilization`` is the process-tree CPU seconds divided by
    nine times the unique measured wall seconds.  It is recorded and checked
    for arithmetic consistency, but deliberately has no pass threshold.
    """

    workers: int
    hmm_wall_seconds_by_k: HMMWallTimes
    block_materialization_wall_seconds: float
    block_call_wall_seconds: FourCellWallTimes
    kernel_wall_seconds: FourCellWallTimes
    dependence_wall_seconds: float
    latent_generation_wall_seconds: RoleWallTimes
    latent_bootstrap_wall_seconds: RoleWallTimes
    bridge_benchmark_wall_seconds: float
    bridge_qmax_seconds: float
    bridge_benchmark_n_states: int
    lightweight_wall_seconds: float
    peak_process_tree_rss_bytes: int
    process_tree_cpu_seconds: float
    worker_capacity_utilization: float
    swap_used_before_bytes: int
    maximum_swap_used_bytes: int
    swap_used_after_bytes: int
    disk_free_bytes: int
    serialized_bytes: int
    native_thread_cap_verified: bool
    worker_parity_verified: bool
    rss_telemetry_complete: bool
    swap_telemetry_complete: bool
    cpu_telemetry_complete: bool
    disk_telemetry_complete: bool

    def __post_init__(self) -> None:
        _validate_measurements(self)


@dataclass(frozen=True, slots=True)
class G3FamilyResourceProjection:
    """One auditable family contribution to projected wall time."""

    family: G3ResourceFamily
    measured_seconds: float
    projected_seconds: float
    formula_id: str


@dataclass(frozen=True, slots=True)
class G3ResourceProjection:
    """Fail-closed full-graph projection and resource-gate decision."""

    projection_version: str
    workers: int
    family_projections: tuple[G3FamilyResourceProjection, ...]
    observed_total_wall_seconds: float
    projected_subtotal_seconds: float
    whole_graph_contingency_seconds: float
    projected_total_seconds: float
    maximum_wall_seconds: float
    process_tree_cpu_seconds: float
    worker_capacity_utilization: float
    measured_peak_process_tree_rss_bytes: int
    observed_rss_with_safety_bytes: int
    code_owned_peak_estimate_bytes: int
    projected_peak_rss_bytes: int
    physical_memory_bytes: int
    projected_memory_fraction: float
    maximum_memory_fraction: float
    swap_used_before_bytes: int
    maximum_swap_used_bytes: int
    swap_used_after_bytes: int
    serialized_bytes: int
    serialized_extrapolation_factor: int
    extrapolated_serialized_bytes: int
    code_owned_disk_estimate_bytes: int
    projected_disk_bytes: int
    disk_free_bytes: int
    disk_safety_multiplier: int
    required_free_disk_bytes: int
    bridge_benchmark_n_states: int
    bridge_pair_units: int
    native_thread_cap_verified: bool
    worker_parity_verified: bool
    rss_telemetry_complete: bool
    swap_telemetry_complete: bool
    cpu_telemetry_complete: bool
    disk_telemetry_complete: bool
    reasons: tuple[str, ...]
    passed: bool
    generation_authorized: Literal[False] = field(default=False, init=False)


def project_canonical_g3_resources(
    measurements: G3ResourceProjectionMeasurements,
) -> G3ResourceProjection:
    """Project the canonical G3 graph without executing or publishing it."""

    if type(measurements) is not G3ResourceProjectionMeasurements:
        raise ModelError("G3 resource projection requires the exact measurement type")
    # Revalidate so mutation through ``object.__setattr__`` cannot bypass the
    # frozen bundle's construction-time checks.
    _validate_measurements(measurements)

    hmm_measured = sum(measurements.hmm_wall_seconds_by_k)
    fixed_k4_measured = measurements.hmm_wall_seconds_by_k[_OWNER_FIXED_K4_INDEX]
    hmm_projected = _HMM_SAFETY_FACTOR * (
        hmm_measured
        + fixed_k4_measured
        * (
            (_HMM_FIXED_K4_DESIGN_STABILITY_RESTARTS + _HMM_RESTARTS_SELECTED_CANDIDATE)
            / HMM_RESTARTS
        )
    )
    block_scale = CANONICAL_HISTORY_COUNT / _BLOCK_SAMPLE_HISTORIES
    block_materialization_projected = (
        _BLOCK_SAFETY_FACTOR
        * block_scale
        * measurements.block_materialization_wall_seconds
    )
    design_monthly, design_daily, production_monthly, production_daily = (
        measurements.block_call_wall_seconds
    )
    block_calls_projected = (
        _BLOCK_SAFETY_FACTOR
        * block_scale
        * (
            _DESIGN_BLOCK_CALLS_PER_CELL * design_monthly
            + _DESIGN_BLOCK_CALLS_PER_CELL * design_daily
            + production_monthly
            + production_daily
        )
    )
    kernel_projected = (
        _KERNEL_SAFETY_FACTOR
        * _KERNEL_WAVE_FACTOR
        * sum(measurements.kernel_wall_seconds)
    )
    dependence_projected = (
        _DEPENDENCE_SAFETY_FACTOR * block_scale * measurements.dependence_wall_seconds
    )
    latent_generation_projected = (
        _LATENT_GENERATION_SAFETY_FACTOR
        * len(measurements.latent_generation_wall_seconds)
        * max(measurements.latent_generation_wall_seconds)
        * (LATENT_CHAIN_COUNT / _LATENT_SAMPLE_CHAINS)
    )
    latent_bootstrap_projected = (
        _LATENT_BOOTSTRAP_SAFETY_FACTOR
        * len(measurements.latent_bootstrap_wall_seconds)
        * max(measurements.latent_bootstrap_wall_seconds)
        * (
            LATENT_BOOTSTRAP_REPLICATES
            * LATENT_BOOTSTRAP_RESAMPLE_SIZE
            / (
                _LATENT_SAMPLE_BOOTSTRAP_REPLICATES
                * _LATENT_SAMPLE_BOOTSTRAP_RESAMPLE_SIZE
            )
        )
    )
    bridge_pair_units = _bridge_pair_units(measurements.bridge_benchmark_n_states)
    bridge_remaining_projected = (
        measurements.bridge_qmax_seconds * bridge_pair_units / CANONICAL_WORKERS
    )
    lightweight_projected = max(
        _LIGHTWEIGHT_MINIMUM_SECONDS,
        _LIGHTWEIGHT_SAFETY_FACTOR * measurements.lightweight_wall_seconds,
    )

    families = (
        G3FamilyResourceProjection(
            "nonbridge_hmm",
            hmm_measured,
            hmm_projected,
            "1.50*(sum(K2..K5_main_50)+K4*(5300+105050)/50)",
        ),
        G3FamilyResourceProjection(
            "block_materialization",
            measurements.block_materialization_wall_seconds,
            block_materialization_projected,
            "1.35*(10000/128)*aggregate_materialization",
        ),
        G3FamilyResourceProjection(
            "block_calls",
            sum(measurements.block_call_wall_seconds),
            block_calls_projected,
            "1.35*(10000/128)*(5*design_monthly+5*design_daily+production)",
        ),
        G3FamilyResourceProjection(
            "kernel",
            sum(measurements.kernel_wall_seconds),
            kernel_projected,
            "1.35*((280+278)/2)*sum(four_fixed_1152_cells)",
        ),
        G3FamilyResourceProjection(
            "dependence",
            measurements.dependence_wall_seconds,
            dependence_projected,
            "1.35*(10000/128)*four_cell_dependence",
        ),
        G3FamilyResourceProjection(
            "latent_generation",
            sum(measurements.latent_generation_wall_seconds),
            latent_generation_projected,
            "1.35*2*max(two_roles)*(10000/512)",
        ),
        G3FamilyResourceProjection(
            "latent_bootstrap",
            sum(measurements.latent_bootstrap_wall_seconds),
            latent_bootstrap_projected,
            "1.50*2*max(two_roles)*(10000^2/512^2)",
        ),
        G3FamilyResourceProjection(
            "bridge_benchmark",
            measurements.bridge_benchmark_wall_seconds,
            BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR
            * measurements.bridge_benchmark_wall_seconds,
            (
                "1.25*measured_K"
                f"{measurements.bridge_benchmark_n_states}_100_coordinate_total"
            ),
        ),
        G3FamilyResourceProjection(
            "bridge_remaining",
            measurements.bridge_qmax_seconds,
            bridge_remaining_projected,
            ("qmax*bridge_pair_units/9_with_fixed_K4_tierA_100_restarts_per_pair"),
        ),
        G3FamilyResourceProjection(
            "lightweight",
            measurements.lightweight_wall_seconds,
            lightweight_projected,
            "max(900,2*measured_lightweight)",
        ),
    )
    projected_subtotal = sum(item.projected_seconds for item in families)
    projected_total = _WHOLE_GRAPH_SAFETY_FACTOR * projected_subtotal
    whole_graph_contingency = projected_total - projected_subtotal

    resource_estimate = canonical_g3_resource_estimate()
    observed_rss_with_safety = math.ceil(
        _OBSERVED_RSS_SAFETY_FACTOR * measurements.peak_process_tree_rss_bytes
    )
    projected_peak_rss = max(
        resource_estimate.estimated_peak_bytes, observed_rss_with_safety
    )
    projected_memory_fraction = projected_peak_rss / CANONICAL_PHYSICAL_MEMORY_BYTES

    extrapolated_serialized = (
        measurements.serialized_bytes * _SERIALIZED_EXTRAPOLATION_FACTOR
    )
    projected_disk = max(
        resource_estimate.estimated_disk_bytes, extrapolated_serialized
    )
    required_free_disk = CANONICAL_DISK_SAFETY_MULTIPLIER * projected_disk

    reasons: list[str] = []
    flag_reasons = (
        (
            measurements.native_thread_cap_verified,
            "native_thread_cap_not_verified",
        ),
        (measurements.worker_parity_verified, "worker_parity_not_verified"),
        (measurements.rss_telemetry_complete, "rss_telemetry_incomplete"),
        (measurements.swap_telemetry_complete, "swap_telemetry_incomplete"),
        (measurements.cpu_telemetry_complete, "cpu_telemetry_incomplete"),
        (measurements.disk_telemetry_complete, "disk_telemetry_incomplete"),
    )
    reasons.extend(reason for verified, reason in flag_reasons if not verified)
    if projected_total > _MAXIMUM_WALL_SECONDS:
        reasons.append("projected_wall_clock_exceeds_72_hours")
    if projected_memory_fraction > _MAXIMUM_MEMORY_FRACTION:
        reasons.append("projected_peak_rss_exceeds_70_percent")
    if measurements.disk_free_bytes < required_free_disk:
        reasons.append("free_disk_below_three_times_projection")

    return G3ResourceProjection(
        projection_version=G3_RESOURCE_PROJECTION_VERSION,
        workers=measurements.workers,
        family_projections=families,
        observed_total_wall_seconds=_observed_total_wall_seconds(measurements),
        projected_subtotal_seconds=projected_subtotal,
        whole_graph_contingency_seconds=whole_graph_contingency,
        projected_total_seconds=projected_total,
        maximum_wall_seconds=_MAXIMUM_WALL_SECONDS,
        process_tree_cpu_seconds=measurements.process_tree_cpu_seconds,
        worker_capacity_utilization=measurements.worker_capacity_utilization,
        measured_peak_process_tree_rss_bytes=(measurements.peak_process_tree_rss_bytes),
        observed_rss_with_safety_bytes=observed_rss_with_safety,
        code_owned_peak_estimate_bytes=resource_estimate.estimated_peak_bytes,
        projected_peak_rss_bytes=projected_peak_rss,
        physical_memory_bytes=CANONICAL_PHYSICAL_MEMORY_BYTES,
        projected_memory_fraction=projected_memory_fraction,
        maximum_memory_fraction=_MAXIMUM_MEMORY_FRACTION,
        swap_used_before_bytes=measurements.swap_used_before_bytes,
        maximum_swap_used_bytes=measurements.maximum_swap_used_bytes,
        swap_used_after_bytes=measurements.swap_used_after_bytes,
        serialized_bytes=measurements.serialized_bytes,
        serialized_extrapolation_factor=_SERIALIZED_EXTRAPOLATION_FACTOR,
        extrapolated_serialized_bytes=extrapolated_serialized,
        code_owned_disk_estimate_bytes=resource_estimate.estimated_disk_bytes,
        projected_disk_bytes=projected_disk,
        disk_free_bytes=measurements.disk_free_bytes,
        disk_safety_multiplier=CANONICAL_DISK_SAFETY_MULTIPLIER,
        required_free_disk_bytes=required_free_disk,
        bridge_benchmark_n_states=measurements.bridge_benchmark_n_states,
        bridge_pair_units=bridge_pair_units,
        native_thread_cap_verified=measurements.native_thread_cap_verified,
        worker_parity_verified=measurements.worker_parity_verified,
        rss_telemetry_complete=measurements.rss_telemetry_complete,
        swap_telemetry_complete=measurements.swap_telemetry_complete,
        cpu_telemetry_complete=measurements.cpu_telemetry_complete,
        disk_telemetry_complete=measurements.disk_telemetry_complete,
        reasons=tuple(reasons),
        passed=not reasons,
    )


def _observed_total_wall_seconds(
    measurements: G3ResourceProjectionMeasurements,
) -> float:
    """Sum each unique measured stage once; qmax is inside bridge total."""

    return (
        sum(measurements.hmm_wall_seconds_by_k)
        + measurements.block_materialization_wall_seconds
        + sum(measurements.block_call_wall_seconds)
        + sum(measurements.kernel_wall_seconds)
        + measurements.dependence_wall_seconds
        + sum(measurements.latent_generation_wall_seconds)
        + sum(measurements.latent_bootstrap_wall_seconds)
        + measurements.bridge_benchmark_wall_seconds
        + measurements.lightweight_wall_seconds
    )


def _validate_measurements(value: G3ResourceProjectionMeasurements) -> None:
    if type(value.workers) is not int or value.workers != CANONICAL_WORKERS:
        raise ModelError("G3 resource projection requires exactly nine workers")
    _positive_float_tuple(value.hmm_wall_seconds_by_k, 4, "HMM K2..K5 walls")
    _positive_float(
        value.block_materialization_wall_seconds, "block materialization wall"
    )
    _positive_float_tuple(value.block_call_wall_seconds, 4, "block-call walls")
    _positive_float_tuple(value.kernel_wall_seconds, 4, "kernel walls")
    _positive_float(value.dependence_wall_seconds, "dependence wall")
    _positive_float_tuple(
        value.latent_generation_wall_seconds, 2, "latent-generation role walls"
    )
    _positive_float_tuple(
        value.latent_bootstrap_wall_seconds, 2, "latent-bootstrap role walls"
    )
    _positive_float(value.bridge_benchmark_wall_seconds, "bridge benchmark wall")
    _positive_float(value.bridge_qmax_seconds, "bridge qmax")
    if (
        type(value.bridge_benchmark_n_states) is not int
        or value.bridge_benchmark_n_states != 4
    ):
        raise ModelError("version-3 bridge benchmark K must be exactly 4")
    if value.bridge_qmax_seconds > value.bridge_benchmark_wall_seconds:
        raise ModelError("bridge qmax cannot exceed total bridge benchmark wall")
    _positive_float(value.lightweight_wall_seconds, "lightweight wall")
    _positive_integer(value.peak_process_tree_rss_bytes, "peak process-tree RSS")
    _positive_float(value.process_tree_cpu_seconds, "process-tree CPU seconds")
    _positive_float(value.worker_capacity_utilization, "worker utilization")
    _nonnegative_integer(value.swap_used_before_bytes, "swap before")
    _nonnegative_integer(value.maximum_swap_used_bytes, "maximum swap")
    _nonnegative_integer(value.swap_used_after_bytes, "swap after")
    if value.maximum_swap_used_bytes < max(
        value.swap_used_before_bytes, value.swap_used_after_bytes
    ):
        raise ModelError("maximum swap must cover the before and after readings")
    _positive_integer(value.disk_free_bytes, "free disk bytes")
    _positive_integer(value.serialized_bytes, "serialized bytes")
    for name in (
        "native_thread_cap_verified",
        "worker_parity_verified",
        "rss_telemetry_complete",
        "swap_telemetry_complete",
        "cpu_telemetry_complete",
        "disk_telemetry_complete",
    ):
        if type(getattr(value, name)) is not bool:
            raise ModelError(f"{name} must be boolean")

    observed_wall = _observed_total_wall_seconds(value)
    expected_utilization = value.process_tree_cpu_seconds / (
        CANONICAL_WORKERS * observed_wall
    )
    if not math.isclose(
        value.worker_capacity_utilization,
        expected_utilization,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ModelError(
            "worker utilization is inconsistent with CPU and measured wall time"
        )


def _positive_float_tuple(value: object, length: int, label: str) -> None:
    if type(value) is not tuple or len(value) != length:
        raise ModelError(f"{label} require the exact fixed geometry")
    for index, item in enumerate(value):
        _positive_float(item, f"{label}[{index}]")


def _positive_float(value: object, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ModelError(f"{label} must be a positive finite float")


def _positive_integer(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ModelError(f"{label} must be a positive integer")


def _nonnegative_integer(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ModelError(f"{label} must be a non-negative integer")
