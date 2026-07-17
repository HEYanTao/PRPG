"""Deterministic multicore execution and local resource preflight utilities."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import psutil
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from prpg.errors import ModelError

Input = TypeVar("Input")
Output = TypeVar("Output")

_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Read-only host capacity used by deterministic preflight decisions."""

    logical_cpu_count: int
    physical_memory_bytes: int
    disk_free_bytes: int
    resolved_workers: int
    reserve_cores: int


@dataclass(frozen=True, slots=True)
class MemoryPreflight:
    """One conservative working-set check against a fixed RAM fraction."""

    estimated_peak_bytes: int
    physical_memory_bytes: int
    maximum_fraction: float
    estimated_fraction: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ExecutionResourceMeasurement:
    """Observed process-tree and system-swap evidence for one ordered map."""

    workers: int
    items: int
    wall_seconds: float
    peak_process_tree_rss_bytes: int
    peak_memory_fraction: float
    swap_used_before_bytes: int | None
    swap_used_after_bytes: int | None
    maximum_swap_used_bytes: int | None
    swap_measurement_available: bool
    process_tree_measurement_available: bool
    preflight: MemoryPreflight
    passed: bool


@dataclass(frozen=True, slots=True)
class MeasuredProcessMapResult(Generic[Output]):
    """Deterministically ordered outputs plus resource evidence."""

    outputs: tuple[Output, ...]
    resources: ExecutionResourceMeasurement


def resolve_worker_count(
    requested: str | int,
    *,
    reserve_cores: int,
    cpu_count: int | None = None,
) -> int:
    """Resolve ``auto`` without exceeding visible cores or consuming reserve."""

    if isinstance(reserve_cores, bool) or not isinstance(reserve_cores, int):
        raise ModelError("reserve_cores must be a non-negative integer")
    if reserve_cores < 0:
        raise ModelError("reserve_cores must be a non-negative integer")
    visible = os.cpu_count() if cpu_count is None else cpu_count
    if isinstance(visible, bool) or not isinstance(visible, int) or visible <= 0:
        raise ModelError("visible CPU count is unavailable")
    available = max(1, visible - reserve_cores)
    if requested == "auto":
        return available
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ModelError("requested workers must be 'auto' or a positive integer")
    if requested > available:
        raise ModelError(
            "requested workers exceed visible capacity after reserve",
            details={"requested": requested, "available": available},
        )
    return requested


def resource_snapshot(
    output_root: str | Path,
    *,
    requested_workers: str | int,
    reserve_cores: int,
) -> ResourceSnapshot:
    """Capture CPU, physical RAM, and free disk without mutating the workspace."""

    logical = os.cpu_count()
    if logical is None or logical <= 0:
        raise ModelError("logical CPU count is unavailable")
    memory = _physical_memory_bytes()
    if memory is None or memory <= 0:
        raise ModelError("physical memory is unavailable")
    root = Path(output_root)
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        disk_free = shutil.disk_usage(probe).free
    except OSError as error:
        raise ModelError("free disk capacity is unavailable") from error
    workers = resolve_worker_count(
        requested_workers,
        reserve_cores=reserve_cores,
        cpu_count=logical,
    )
    return ResourceSnapshot(logical, memory, disk_free, workers, reserve_cores)


def memory_preflight(
    estimated_peak_bytes: int,
    physical_memory_bytes: int,
    *,
    maximum_fraction: float,
) -> MemoryPreflight:
    """Apply a conservative fixed-fraction memory gate."""

    for value, label in (
        (estimated_peak_bytes, "estimated_peak_bytes"),
        (physical_memory_bytes, "physical_memory_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelError(f"{label} must be a positive integer")
    if not isinstance(maximum_fraction, float) or not 0.0 < maximum_fraction <= 0.70:
        raise ModelError("maximum memory fraction must be a float in (0, 0.70]")
    fraction = estimated_peak_bytes / physical_memory_bytes
    return MemoryPreflight(
        estimated_peak_bytes,
        physical_memory_bytes,
        maximum_fraction,
        fraction,
        fraction <= maximum_fraction,
    )


def deterministic_process_map(
    function: Callable[[Input], Output],
    items: Sequence[Input] | Iterable[Input],
    *,
    workers: int,
    chunksize: int = 1,
) -> tuple[Output, ...]:
    """Map in input order under spawn with one native math thread per worker."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ModelError("process-map workers must be a positive integer")
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
        raise ModelError("process-map chunksize must be a positive integer")
    materialized = tuple(items)
    if not materialized:
        return ()
    if workers == 1:
        with threadpool_limits(limits=1):
            return tuple(function(item) for item in materialized)
    context = mp.get_context("spawn")
    with (
        _single_thread_environment(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_limit_worker_native_threads,
        ) as executor,
    ):
        return tuple(executor.map(function, materialized, chunksize=chunksize))


def measured_process_map(
    function: Callable[[Input], Output],
    items: Sequence[Input] | Iterable[Input],
    *,
    workers: int,
    estimated_peak_bytes: int,
    physical_memory_bytes: int,
    maximum_memory_fraction: float = 0.70,
    chunksize: int = 1,
    sample_interval_seconds: float = 0.02,
) -> MeasuredProcessMapResult[Output]:
    """Preflight, execute in order, and sample parent-plus-child RSS/swap.

    The estimate gate runs before the input iterable is materialized or any
    worker starts.  The observed result is fail-closed when process-tree RSS
    exceeds the same limit; callers must not launch a downstream large stage
    from a failed measurement.  This is the bridge 100-pair benchmark surface
    and the general audit wrapper for other expensive process maps.
    """

    preflight = memory_preflight(
        estimated_peak_bytes,
        physical_memory_bytes,
        maximum_fraction=maximum_memory_fraction,
    )
    if not preflight.passed:
        raise ModelError(
            "process-map memory estimate exceeds the preflight limit",
            details={
                "estimated_peak_bytes": estimated_peak_bytes,
                "physical_memory_bytes": physical_memory_bytes,
                "maximum_fraction": maximum_memory_fraction,
            },
        )
    if not isinstance(sample_interval_seconds, float) or not (
        0.001 <= sample_interval_seconds <= 1.0
    ):
        raise ModelError("resource sample interval must be a float in [0.001, 1]")
    materialized = tuple(items)
    sampler = _ProcessTreeSampler(sample_interval_seconds)
    started = time.monotonic()
    sampler.start()
    try:
        outputs = deterministic_process_map(
            function,
            materialized,
            workers=workers,
            chunksize=chunksize,
        )
    finally:
        sampler.stop()
    elapsed = time.monotonic() - started
    if not elapsed >= 0.0:  # pragma: no cover - monotonic clock invariant
        raise AssertionError("monotonic execution clock moved backwards")
    peak_fraction = sampler.peak_rss_bytes / physical_memory_bytes
    resources = ExecutionResourceMeasurement(
        workers,
        len(materialized),
        elapsed,
        sampler.peak_rss_bytes,
        peak_fraction,
        sampler.swap_before_bytes,
        sampler.swap_after_bytes,
        sampler.maximum_swap_bytes,
        sampler.swap_available,
        sampler.process_tree_available,
        preflight,
        peak_fraction <= maximum_memory_fraction
        and sampler.swap_available
        and sampler.process_tree_available,
    )
    return MeasuredProcessMapResult(outputs, resources)


class _ProcessTreeSampler:
    """Small background sampler scoped to one parent-owned process map."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="prpg-resource-sampler",
            daemon=True,
        )
        self.peak_rss_bytes = 0
        initial_swap = _swap_used_bytes()
        self.swap_before_bytes = initial_swap
        self.swap_after_bytes = initial_swap
        self.maximum_swap_bytes = initial_swap
        self.swap_available = initial_swap is not None
        self.process_tree_available = True

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
        self.swap_after_bytes = _swap_used_bytes()
        self.swap_available = self.swap_available and self.swap_after_bytes is not None
        if self.swap_after_bytes is not None:
            self.maximum_swap_bytes = (
                self.swap_after_bytes
                if self.maximum_swap_bytes is None
                else max(self.maximum_swap_bytes, self.swap_after_bytes)
            )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        root = psutil.Process(os.getpid())
        try:
            processes = (root, *root.children(recursive=True))
        except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
            processes = (root,)
            self.process_tree_available = False
        rss = 0
        for process in processes:
            try:
                rss += int(process.memory_info().rss)
            except psutil.NoSuchProcess:
                # Workers can exit between enumeration and measurement.  A
                # process that no longer exists has no resident memory at read
                # time, so this expected race contributes zero without making
                # the process-tree API unavailable.
                continue
            except (OSError, psutil.AccessDenied):
                self.process_tree_available = False
                continue
        swap = _swap_used_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.swap_available = self.swap_available and swap is not None
        if swap is not None:
            self.maximum_swap_bytes = (
                swap
                if self.maximum_swap_bytes is None
                else max(self.maximum_swap_bytes, swap)
            )


def _swap_used_bytes() -> int | None:
    try:
        return int(psutil.swap_memory().used)
    except (OSError, NotImplementedError, psutil.Error):
        return None


def _limit_worker_native_threads() -> None:
    for name in _NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"
    threadpool_limits(limits=1)


@contextmanager
def _single_thread_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _NATIVE_THREAD_VARIABLES}
    for name in _NATIVE_THREAD_VARIABLES:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * page_count)
