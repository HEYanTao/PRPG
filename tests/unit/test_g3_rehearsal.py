from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import psutil
import pytest

import prpg.model.g3_rehearsal as subject
from prpg.errors import IntegrityError, ModelError
from prpg.model.calibration import DecodedReturnPools
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_rehearsal import (
    REDUCED_G3_REHEARSAL_GEOMETRY,
    G3RehearsalTelemetry,
    NoncanonicalG3RehearsalReport,
    ReducedG3RehearsalWorkload,
    ReducedG3TaskProbe,
    RehearsalFamilyTiming,
    assemble_noncanonical_g3_rehearsal_report,
    rehearsal_task_fingerprint,
    verify_noncanonical_g3_rehearsal_report,
)
from prpg.model.hmm import summarize_state_path


def _fp(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _all_timings(*, hmm_seconds: float = 1.0) -> tuple[RehearsalFamilyTiming, ...]:
    return (
        RehearsalFamilyTiming("hmm", "k2", hmm_seconds),
        RehearsalFamilyTiming("hmm", "k3", hmm_seconds),
        RehearsalFamilyTiming("hmm", "k4", hmm_seconds),
        RehearsalFamilyTiming("hmm", "k5", hmm_seconds),
        RehearsalFamilyTiming("block_materialization", "all_four", 1.0),
        RehearsalFamilyTiming("block_call", "design_monthly", 1.0),
        RehearsalFamilyTiming("block_call", "design_daily", 1.0),
        RehearsalFamilyTiming("block_call", "production_monthly", 1.0),
        RehearsalFamilyTiming("block_call", "production_daily", 1.0),
        RehearsalFamilyTiming("kernel", "design_monthly", 1.0),
        RehearsalFamilyTiming("kernel", "design_daily", 1.0),
        RehearsalFamilyTiming("kernel", "production_monthly", 1.0),
        RehearsalFamilyTiming("kernel", "production_daily", 1.0),
        RehearsalFamilyTiming("dependence", "all_four", 1.0),
        RehearsalFamilyTiming("latent_generation", "design", 1.0),
        RehearsalFamilyTiming("latent_generation", "production", 1.0),
        RehearsalFamilyTiming("latent_bootstrap", "design", 1.0),
        RehearsalFamilyTiming("latent_bootstrap", "production", 1.0),
        RehearsalFamilyTiming("bridge_benchmark", "total", 1.0),
        RehearsalFamilyTiming("bridge_benchmark", "qmax", 0.5),
        RehearsalFamilyTiming("lightweight", "remaining_26_task_graph", 1.0),
    )


def _decoded_pools(
    monthly_counts: tuple[int, int, int, int],
    daily_counts: tuple[int, int, int, int],
) -> DecodedReturnPools:
    monthly_states = np.asarray(
        [state for state, count in enumerate(monthly_counts) for _ in range(count)],
        dtype=np.int64,
    )
    daily_states = np.asarray(
        [state for state, count in enumerate(daily_counts) for _ in range(count)],
        dtype=np.int64,
    )
    return DecodedReturnPools(
        monthly_states=monthly_states,
        daily_states=daily_states,
        monthly_diagnostics=summarize_state_path(
            monthly_states,
            4,
            lengths=(1,) * len(monthly_states),
        ),
        daily_diagnostics=summarize_state_path(
            daily_states,
            4,
            lengths=(1,) * len(daily_states),
        ),
    )


class _Workload:
    config_fingerprint = _fp("config")
    calibration_input_fingerprint = _fp("input")
    processed_data_fingerprint = _fp("processed")
    raw_snapshot_fingerprint = _fp("raw")
    source_code_fingerprint = _fp("source")
    dependency_lock_fingerprint = _fp("lock")
    scientific_version = 6
    native_thread_cap_verified = True
    worker_parity_verified = True
    worker_parity_fingerprint = _fp("one-vs-nine")
    worker_parity_families: tuple[str, ...] = ("hmm", "kernel", "bridge")
    bridge_benchmark_n_states = 5
    rehearsal_geometry = REDUCED_G3_REHEARSAL_GEOMETRY

    def __init__(
        self,
        *,
        timings: tuple[RehearsalFamilyTiming, ...] | None = None,
        wrong_task_at: int | None = None,
    ) -> None:
        self._timings = _all_timings() if timings is None else timings
        self._wrong_task_at = wrong_task_at
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def execute_task(
        self,
        task_id: str,
        dependency_fingerprints: tuple[tuple[str, str], ...],
    ) -> ReducedG3TaskProbe:
        self.calls.append((task_id, dependency_fingerprints))
        ordinal = len(self.calls)
        returned_task = (
            G3_GATE_ORDER[(ordinal + 1) % len(G3_GATE_ORDER)]
            if self._wrong_task_at == ordinal
            else task_id
        )
        fingerprint = rehearsal_task_fingerprint(
            calibration_input_fingerprint=self.calibration_input_fingerprint,
            scientific_version=self.scientific_version,
            task_id=returned_task,
            dependency_fingerprints=dependency_fingerprints,
            result_fingerprints=(_fp(f"numerical-{ordinal}"),),
        )
        return ReducedG3TaskProbe(
            returned_task,
            fingerprint,
            self._timings if ordinal == 1 else (),
            serialized_bytes=64,
        )


def _telemetry(*, peak_rss: int = 1 << 20) -> G3RehearsalTelemetry:
    return G3RehearsalTelemetry(
        peak_process_tree_rss_bytes=peak_rss,
        process_tree_cpu_seconds=90.0,
        swap_used_before_bytes=0,
        maximum_swap_used_bytes=0,
        swap_used_after_bytes=0,
        disk_free_bytes=64 << 30,
        rss_telemetry_complete=True,
        swap_telemetry_complete=True,
        cpu_telemetry_complete=True,
        disk_telemetry_complete=True,
    )


def test_rehearsal_rss_sampler_tolerates_normal_worker_exit(
    tmp_path: Path,
) -> None:
    class _LiveChild:
        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=512)

    class _ExitedChild:
        def memory_info(self) -> object:
            raise psutil.NoSuchProcess(pid=123)

    class _Root:
        def children(self, *, recursive: bool) -> list[object]:
            assert recursive
            return [_LiveChild(), _ExitedChild()]

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=4096)

    sampler = subject._RehearsalHostSampler(tmp_path)
    sampler._root = cast(psutil.Process, _Root())
    sampler._peak_rss = 0

    sampler._sample()

    assert sampler._rss_complete
    assert sampler._peak_rss == 4608


def test_rehearsal_rss_sampler_fails_closed_on_access_error(
    tmp_path: Path,
) -> None:
    class _UnreadableChild:
        def memory_info(self) -> object:
            raise psutil.AccessDenied(pid=123)

    class _Root:
        def children(self, *, recursive: bool) -> list[object]:
            assert recursive
            return [_UnreadableChild()]

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=4096)

    sampler = subject._RehearsalHostSampler(tmp_path)
    sampler._root = cast(psutil.Process, _Root())

    sampler._sample()

    assert not sampler._rss_complete


def test_rehearsal_rss_sampler_fails_closed_on_enumeration_error(
    tmp_path: Path,
) -> None:
    class _Root:
        def children(self, *, recursive: bool) -> list[object]:
            assert recursive
            raise psutil.AccessDenied(pid=123)

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=4096)

    sampler = subject._RehearsalHostSampler(tmp_path)
    sampler._root = cast(psutil.Process, _Root())
    sampler._peak_rss = 0

    sampler._sample()

    assert not sampler._rss_complete
    assert sampler._peak_rss == 4096


def test_closed_26_task_rehearsal_is_in_memory_unreleasable_and_projected() -> None:
    workload = _Workload()

    observations, timings, serialized_bytes = subject._walk_rehearsal_task_graph(
        workload
    )

    assert tuple(item.task_id for item in observations) == G3_GATE_ORDER
    assert len(workload.calls) == 26
    assert all(item.outcome == "graph_traversed_noncanonical" for item in observations)
    assert timings == _all_timings()
    assert serialized_bytes == 26 * 64


def test_task_observations_bind_exact_direct_dependencies() -> None:
    observations, _, _ = subject._walk_rehearsal_task_graph(_Workload())
    by_id = {item.task_id: item.result_fingerprint for item in observations}

    artifact = observations[-1]
    assert artifact.dependency_fingerprints == (
        ("actual_artifact_bridge", by_id["actual_artifact_bridge"]),
    )
    resource = observations[19]
    assert tuple(item[0] for item in resource.dependency_fingerprints) == (
        "design_macro_stability",
        "production_macro_stability",
        "design_fragmentation",
        "production_fragmentation",
        "four_cell_adequacy_holm",
        "four_cell_dependence_holm",
    )


def test_rehearsal_has_no_canonical_store_or_authority_input_surface() -> None:
    parameters = inspect.signature(assemble_noncanonical_g3_rehearsal_report).parameters
    forbidden = {
        "run_store",
        "checkpoint_store",
        "model_store",
        "pair_store",
        "evidence_store",
        "lease",
        "preflight",
        "canonical_launch",
    }
    assert forbidden.isdisjoint(parameters)
    assert forbidden.isdisjoint(NoncanonicalG3RehearsalReport.__dataclass_fields__)

    with pytest.raises(TypeError):
        assemble_noncanonical_g3_rehearsal_report(  # type: ignore[call-arg]
            _Workload(),
            _telemetry(),
            run_store=object(),
        )


def test_wrong_task_or_substituted_probe_aborts_before_a_report_exists() -> None:
    with pytest.raises(IntegrityError, match="wrong task"):
        subject._walk_rehearsal_task_graph(_Workload(wrong_task_at=3))

    class _Substituted(_Workload):
        def execute_task(  # type: ignore[override]
            self,
            task_id: str,
            dependency_fingerprints: tuple[tuple[str, str], ...],
        ) -> object:
            del task_id, dependency_fingerprints
            return object()

    with pytest.raises(ModelError, match="substituted probe"):
        subject._walk_rehearsal_task_graph(
            cast(ReducedG3RehearsalWorkload, _Substituted())
        )


def test_unminted_report_and_caller_authored_workload_are_rejected() -> None:
    with pytest.raises(IntegrityError, match="not minted"):
        verify_noncanonical_g3_rehearsal_report(NoncanonicalG3RehearsalReport())

    with pytest.raises(ModelError, match="executor-minted"):
        assemble_noncanonical_g3_rehearsal_report(  # type: ignore[arg-type]
            _Workload(),
            _telemetry(),
        )


def test_projection_fails_closed_above_72_hours_without_becoming_g3_failure() -> None:
    timings = _all_timings(hmm_seconds=200.0)
    measurements = subject._projection_measurements(
        timings,
        _telemetry(),
        serialized_probe_bytes=1,
        native_thread_cap_verified=True,
        worker_parity_verified=True,
        bridge_benchmark_n_states=4,
    )
    projection = subject.project_canonical_g3_resources(measurements)

    assert projection.projected_total_seconds > 72 * 60 * 60
    assert not projection.passed
    assert "projected_wall_clock_exceeds_72_hours" in projection.reasons


def test_missing_family_or_unverified_one_vs_nine_parity_fails_closed() -> None:
    missing = tuple(item for item in _all_timings() if item.family != "dependence")
    with pytest.raises(ModelError, match="timing geometry is incomplete"):
        subject._projection_measurements(
            missing,
            _telemetry(),
            serialized_probe_bytes=1,
            native_thread_cap_verified=True,
            worker_parity_verified=True,
            bridge_benchmark_n_states=4,
        )

    measurements = subject._projection_measurements(
        _all_timings(),
        _telemetry(),
        serialized_probe_bytes=1,
        native_thread_cap_verified=True,
        worker_parity_verified=False,
        bridge_benchmark_n_states=4,
    )
    projection = subject.project_canonical_g3_resources(measurements)
    assert not projection.passed
    assert projection.reasons == ("worker_parity_not_verified",)


def test_concrete_resource_seams_bind_block_starts_and_bridge_qmax() -> None:
    from prpg.model import g3_rehearsal as subject

    assert subject._rehearsal_block_start_geometry(10, 108) == (
        (10, 108),
        (12, 126),
    )
    records = tuple(
        SimpleNamespace(dgp=dgp, pair=pair, elapsed_seconds=seconds)
        for dgp, elapsed in (
            ("model", (1.0, 4.0, 2.0)),
            ("nonparametric", (3.0, 2.0, 5.0)),
        )
        for pair, seconds in enumerate(elapsed)
    )
    maxima, qmax = subject._bridge_elapsed_by_coordinate(records, coordinates=3)
    assert maxima == (3.0, 4.0, 5.0)
    assert qmax == 5.0

    with pytest.raises(ModelError, match="both DGPs in exact order"):
        subject._bridge_elapsed_by_coordinate(records[::-1], coordinates=3)


def test_private_geometry_cannot_feed_launch_projection() -> None:
    workload = _Workload()
    workload.rehearsal_geometry = None

    with pytest.raises(ModelError, match="executor-minted"):
        assemble_noncanonical_g3_rehearsal_report(  # type: ignore[arg-type]
            workload, _telemetry()
        )


def test_projection_eligible_geometry_cannot_change_any_fixed_dimension() -> None:
    from prpg.model import g3_rehearsal as subject

    with pytest.raises(ModelError, match="fixed production geometry"):
        subject._ConcreteRehearsalGeometry(
            workers=9,
            hmm_state_counts=(2, 3, 4, 5),
            hmm_restarts_per_k=49,
            block_histories=128,
            kernel_windows_per_cell=1_152,
            latent_chains_per_role=512,
            latent_bootstrap_replicates_per_role=512,
            latent_bootstrap_resample_size=512,
            bridge_coordinates=100,
            latent_measured_months=subject.LATENT_MEASURED_MONTHS,
            latent_extension_months=subject.LATENT_EXTENSION_MONTHS,
            projection_eligible=True,
        )


def test_rehearsal_geometry_and_selector_require_owner_fixed_k4() -> None:
    from prpg.model import g3_rehearsal as subject

    with pytest.raises(ModelError, match="include owner-fixed K=4"):
        subject._ConcreteRehearsalGeometry(
            workers=1,
            hmm_state_counts=(2, 3, 5),
            hmm_restarts_per_k=1,
            block_histories=1,
            kernel_windows_per_cell=1,
            latent_chains_per_role=1,
            latent_bootstrap_replicates_per_role=1,
            latent_bootstrap_resample_size=1,
            bridge_coordinates=1,
            latent_measured_months=1,
            latent_extension_months=1,
            projection_eligible=False,
        )
    with pytest.raises(ModelError, match="owner-fixed K4 design HMM"):
        subject._select_owner_fixed_k4_rehearsal_fit({})


def test_rehearsal_requires_recurring_monthly_and_daily_k4_history() -> None:
    passed = subject._require_rehearsal_recurring_history_support(
        _decoded_pools((2, 2, 2, 3), (2, 2, 2, 3)),
        role="design",
    )
    assert passed.passed
    assert passed.monthly_episode_counts == (2, 2, 2, 3)

    with pytest.raises(ModelError, match="production K4 recurring-history") as caught:
        subject._require_rehearsal_recurring_history_support(
            _decoded_pools((2, 1, 2, 3), (2, 2, 1, 3)),
            role="production",
        )
    assert caught.value.details["rejection_reasons"] == [
        "state_1_monthly_observed_episodes_below_two",
        "state_2_daily_runs_below_two",
    ]
