from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil
import pytest

import prpg.execution as subject
from prpg.errors import ModelError
from prpg.execution import (
    deterministic_process_map,
    measured_process_map,
    memory_preflight,
    resolve_worker_count,
    resource_snapshot,
)


def test_auto_workers_reserve_one_visible_core() -> None:
    assert resolve_worker_count("auto", reserve_cores=1, cpu_count=10) == 9
    assert resolve_worker_count(4, reserve_cores=1, cpu_count=10) == 4
    assert resolve_worker_count("auto", reserve_cores=20, cpu_count=10) == 1


def test_resource_snapshot_and_memory_fraction(tmp_path: Path) -> None:
    snapshot = resource_snapshot(
        tmp_path / "future" / "runs",
        requested_workers="auto",
        reserve_cores=1,
    )
    assert snapshot.logical_cpu_count >= 1
    assert snapshot.physical_memory_bytes > 0
    assert snapshot.disk_free_bytes > 0
    passed = memory_preflight(10, 100, maximum_fraction=0.70)
    failed = memory_preflight(71, 100, maximum_fraction=0.70)
    assert passed.passed
    assert not failed.passed


def test_ordered_map_has_identical_serial_and_spawn_results() -> None:
    values = (-3, -1, 2, 5)
    serial = deterministic_process_map(abs, values, workers=1)
    # The managed macOS test sandbox denies only the stdlib semaphore-capacity
    # query.  Returning ``-1`` reproduces the platform's documented
    # "indeterminate limit" value while leaving real process spawning intact.
    with patch("concurrent.futures.process.os.sysconf", return_value=-1):
        parallel = deterministic_process_map(abs, values, workers=2)
    assert serial == parallel == (3, 1, 2, 5)


def test_measured_map_preflights_before_work_and_retains_order() -> None:
    with (
        patch(
            "prpg.execution.psutil.swap_memory",
            return_value=SimpleNamespace(used=0),
        ),
        patch("prpg.execution.psutil.Process.children", return_value=[]),
    ):
        result = measured_process_map(
            abs,
            (-3, 2, -1),
            workers=1,
            estimated_peak_bytes=1,
            physical_memory_bytes=10**15,
            sample_interval_seconds=0.001,
        )
    assert result.outputs == (3, 2, 1)
    assert result.resources.items == 3
    assert result.resources.workers == 1
    assert result.resources.wall_seconds >= 0.0
    assert result.resources.peak_process_tree_rss_bytes > 0
    assert result.resources.passed

    consumed = False

    def values():
        nonlocal consumed
        consumed = True
        yield -1

    with pytest.raises(ModelError, match="preflight"):
        measured_process_map(
            abs,
            values(),
            workers=1,
            estimated_peak_bytes=71,
            physical_memory_bytes=100,
        )
    assert not consumed


def test_process_tree_sampler_tolerates_normal_worker_exit() -> None:
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

    sampler = subject._ProcessTreeSampler(1.0)
    with patch("prpg.execution.psutil.Process", return_value=_Root()):
        sampler._sample()

    assert sampler.process_tree_available
    assert sampler.peak_rss_bytes == 4608


def test_process_tree_sampler_fails_closed_on_access_error() -> None:
    class _UnreadableChild:
        def memory_info(self) -> object:
            raise psutil.AccessDenied(pid=123)

    class _Root:
        def children(self, *, recursive: bool) -> list[object]:
            assert recursive
            return [_UnreadableChild()]

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=4096)

    sampler = subject._ProcessTreeSampler(1.0)
    with patch("prpg.execution.psutil.Process", return_value=_Root()):
        sampler._sample()

    assert not sampler.process_tree_available


def test_process_tree_sampler_fails_closed_on_enumeration_error() -> None:
    class _Root:
        def children(self, *, recursive: bool) -> list[object]:
            assert recursive
            raise psutil.AccessDenied(pid=123)

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=4096)

    sampler = subject._ProcessTreeSampler(1.0)
    with patch("prpg.execution.psutil.Process", return_value=_Root()):
        sampler._sample()

    assert not sampler.process_tree_available
    assert sampler.peak_rss_bytes == 4096


@pytest.mark.parametrize(
    "operation",
    [
        lambda: resolve_worker_count(11, reserve_cores=0, cpu_count=10),
        lambda: resolve_worker_count("bad", reserve_cores=0, cpu_count=10),
        lambda: memory_preflight(0, 100, maximum_fraction=0.70),
        lambda: memory_preflight(1, 100, maximum_fraction=0.71),
        lambda: deterministic_process_map(abs, [1], workers=0),
    ],
)
def test_execution_inputs_fail_closed(operation: object) -> None:
    with pytest.raises(ModelError):
        operation()  # type: ignore[operator]
