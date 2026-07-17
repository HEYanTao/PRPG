from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prpg.config import load_config
from prpg.simulation import production as subject
from prpg.storage.group_commit import StorageCommitBinding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "canonical.yaml"


def _loader() -> subject.ProductionLoaderSpec:
    return subject.ProductionLoaderSpec(
        project_root=str(PROJECT_ROOT),
        config_path=str(CONFIG_PATH),
        processed_data_fingerprint="1" * 64,
        g3_evidence_fingerprint="2" * 64,
        g5_authority_fingerprint="3" * 64,
        config_fingerprint=load_config(CONFIG_PATH).fingerprint(),
    )


def _binding(marker: str = "4") -> subject.ProductionBinding:
    return subject.ProductionBinding(
        data_fingerprint="1" * 64,
        simulation_fingerprint=marker * 64,
        qualification_fingerprint="5" * 64,
        materialization_fingerprint="6" * 64,
        toolchain_fingerprint="7" * 64,
        run_fingerprint="8" * 64,
    )


def _small_plan() -> subject.ProductionPlan:
    return subject.ProductionPlan(
        profile_id="test-50y-v1",
        years=50,
        lf_paths=2,
        hf_paths=1,
        lf_paths_per_shard=1,
        hf_paths_per_shard=1,
    )


def _fake_receipt(scope: subject.ProductionWorkUnit) -> SimpleNamespace:
    count = 3 if scope.family == "LF" else 2
    siblings = [{"sha256": f"{index + 1:x}" * 64} for index in range(count)]
    return SimpleNamespace(
        work_unit_id=scope.work_unit_id,
        family=scope.family,
        first_path_entity=scope.first_path_entity,
        last_path_entity=scope.last_path_entity,
        fingerprint="9" * 64,
        duration_monotonic_ns=1_000_000,
        worker_identity="test-worker",
        identity={"siblings": siblings},
    )


def _install_fake_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    scopes: dict[str, subject.ProductionWorkUnit] = {}

    def run_tasks(
        tasks: tuple[Any, ...], *, workers: int
    ) -> tuple[subject.ProductionWorkResult, ...]:
        assert workers > 0
        results = []
        for task in tasks:
            scopes[task.scope.work_unit_id] = task.scope
            results.append(
                subject._work_result_from_receipt(_fake_receipt(task.scope), task.scope)
            )
        return tuple(results)

    def verify(
        output_root: Path,
        *,
        binding: StorageCommitBinding,
        work_unit_id: str,
    ) -> SimpleNamespace:
        _ = output_root, binding
        scope = scopes[work_unit_id]
        return _fake_receipt(scope)

    monkeypatch.setattr(subject, "_run_worker_tasks", run_tasks)
    monkeypatch.setattr(subject, "verify_serial_work_unit", verify)
    monkeypatch.setattr(
        subject, "_verify_production_binding", lambda *args, **kwargs: None
    )

    class _Sampler:
        def __init__(self, interval_seconds: float) -> None:
            assert interval_seconds > 0.0
            self.peak_rss_bytes = 100_000_000
            self.swap_before_bytes = 0
            self.swap_after_bytes = 0

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(subject, "_ProductionResourceSampler", _Sampler)


def test_canonical_plan_has_exact_contract_arithmetic() -> None:
    plan = subject.CANONICAL_PRODUCTION_PLAN
    scopes = subject.production_work_units(plan)

    assert plan.canonical
    assert plan.total_rows == 19_450_000
    assert plan.total_values == 58_350_000
    assert plan.work_unit_count == 70
    assert plan.csv_shard_count == 190
    assert len(scopes) == 70
    assert scopes[0].work_unit_id == "lf-00000-lf000001-lf000100"
    assert scopes[49].work_unit_id == "lf-00049-lf004901-lf005000"
    assert scopes[50].work_unit_id == "hf-00000-hf000001-hf000050"
    assert scopes[-1].work_unit_id == "hf-00019-hf000951-hf001000"


def test_plan_rejects_invalid_sizes_and_namespace() -> None:
    with pytest.raises(subject.GenerationError, match="LF shard size"):
        subject.ProductionPlan("test", 50, 1, 1, 2, 1)
    with pytest.raises(subject.GenerationError, match="CSV namespace"):
        subject.ProductionPlan("test", 50, 1_000_000, 1, 1, 1)
    with pytest.raises(subject.GenerationError, match="must be positive"):
        subject.ProductionPlan("test", 0, 1, 1, 1, 1)


def test_loader_fingerprint_is_stable_and_paths_are_canonical() -> None:
    first = _loader()
    second = _loader()
    assert first == second
    assert first.fingerprint == second.fingerprint
    assert Path(first.project_root).is_absolute()
    assert Path(first.config_path).is_absolute()


def test_resource_preflight_uses_nine_workers_and_three_times_disk(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    estimate = subject.production_resource_estimate(
        plan=subject.CANONICAL_PRODUCTION_PLAN,
        config=config,
        output_root=tmp_path / "new-run",
        requested_workers=9,
        measured_rows_per_second=100_000.0,
    )

    assert estimate.workers == 9
    assert estimate.final_output_bytes_estimate == (
        19_450_000 * subject.CSV_ROW_ESTIMATE_BYTES
        + 70 * subject.RECEIPT_OVERHEAD_BYTES_PER_WORK_UNIT
    )
    assert estimate.required_free_disk_bytes == (
        subject.DISK_SAFETY_MULTIPLIER * estimate.final_output_bytes_estimate
    )
    assert estimate.projected_wall_seconds == pytest.approx(194.5)
    assert estimate.memory_passed
    assert estimate.disk_passed
    assert estimate.passed


def test_resource_preflight_rejects_invalid_throughput(tmp_path: Path) -> None:
    with pytest.raises(subject.GenerationError, match="rows/second"):
        subject.production_resource_estimate(
            plan=_small_plan(),
            config=load_config(CONFIG_PATH),
            output_root=tmp_path / "new-run",
            measured_rows_per_second=0.0,
            requested_workers=1,
        )


def test_run_rejects_worker_count_above_resource_safe_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    estimate = subject.ProductionResourceEstimate(
        workers=1,
        logical_cpu_count=2,
        physical_memory_bytes=16 * 1024**3,
        estimated_peak_memory_bytes=2 * 1024**3,
        estimated_memory_fraction=0.125,
        final_output_bytes_estimate=1_000_000,
        required_free_disk_bytes=3_000_000,
        disk_free_bytes=10_000_000,
        projected_wall_seconds=None,
        memory_passed=True,
        disk_passed=True,
        time_projection_available=False,
    )
    monkeypatch.setattr(
        subject, "production_resource_estimate", lambda **kwargs: estimate
    )

    with pytest.raises(subject.GenerationError, match="resource-safe resolved"):
        subject.run_production_generation(
            loader=_loader(),
            plan=_small_plan(),
            binding=_binding(),
            output_root=tmp_path / "production",
            run_name="test-production",
            workers=2,
        )
    assert not (tmp_path / "production").exists()


def test_production_binding_is_rederived_from_loader_plan_and_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "build_production_binding",
        lambda *args, **kwargs: _binding("a"),
    )

    with pytest.raises(subject.IntegrityError, match="frozen loader"):
        subject._verify_production_binding(
            _loader(),
            plan=_small_plan(),
            binding=_binding("4"),
            run_name="test-production",
        )


def test_run_coordinator_writes_generated_state_without_release_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_execution(monkeypatch)
    root = tmp_path / "production"

    result = subject.run_production_generation(
        loader=_loader(),
        plan=_small_plan(),
        binding=_binding(),
        output_root=root,
        run_name="test-production",
        workers=1,
    )

    assert result.generated_work_units == 3
    assert result.reused_work_units == 0
    assert len(result.work_unit_results) == 3
    assert len(result.fingerprint) == 64
    state = json.loads((root / "run-state.json").read_bytes())
    assert state["disposition"] == "generated"
    assert state["result_fingerprint"] == result.fingerprint
    assert not (root / "RUNNING").exists()
    assert not (root / "INTERRUPTED").exists()
    assert not (root / "FAILED").exists()
    assert not (root / "DATA_VALIDATED").exists()
    assert not (root / "COMPLETE").exists()
    assert not (root / "RELEASED").exists()
    assert (root / "logs" / "generation-result.json").is_file()
    assert (root / "config" / "resolved-config.json").is_file()
    assert (root / "data" / "data-manifest.json").is_file()
    assert (root / "model" / "model-manifest.json").is_file()


def test_measured_memory_over_limit_fails_without_generated_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_execution(monkeypatch)
    root = tmp_path / "production"
    monkeypatch.setattr(
        subject,
        "_mint_execution_result",
        lambda **kwargs: SimpleNamespace(
            peak_process_tree_rss_bytes=14_000_000_000,
            peak_memory_fraction=0.80,
        ),
    )

    with pytest.raises(subject.GenerationError, match="measured-memory gate"):
        subject.run_production_generation(
            loader=_loader(),
            plan=_small_plan(),
            binding=_binding(),
            output_root=root,
            run_name="test-production",
            workers=1,
        )

    assert (root / "FAILED").is_file()
    assert not (root / "logs" / "generation-result.json").exists()
    assert json.loads((root / "run-state.json").read_bytes())["disposition"] == (
        "failed"
    )


def test_interrupted_run_can_resume_with_same_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "production"
    monkeypatch.setattr(
        subject, "_verify_production_binding", lambda *args, **kwargs: None
    )

    def interrupt(tasks: tuple[Any, ...], *, workers: int) -> tuple[Any, ...]:
        _ = tasks, workers
        raise KeyboardInterrupt

    monkeypatch.setattr(subject, "_run_worker_tasks", interrupt)
    with pytest.raises(KeyboardInterrupt):
        subject.run_production_generation(
            loader=_loader(),
            plan=_small_plan(),
            binding=_binding(),
            output_root=root,
            run_name="test-production",
            workers=1,
        )
    assert (root / "INTERRUPTED").is_file()
    assert json.loads((root / "run-state.json").read_bytes())["disposition"] == (
        "interrupted"
    )

    _install_fake_execution(monkeypatch)
    result = subject.run_production_generation(
        loader=_loader(),
        plan=_small_plan(),
        binding=_binding(),
        output_root=root,
        run_name="test-production",
        workers=1,
        resume=True,
    )
    assert result.generated_work_units == 3
    assert not (root / "INTERRUPTED").exists()
    state = json.loads((root / "run-state.json").read_bytes())
    assert state["disposition"] == "generated"
    assert len(state["attempts"]) == 2


def test_unexpected_error_is_terminal_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "production"
    monkeypatch.setattr(
        subject, "_verify_production_binding", lambda *args, **kwargs: None
    )

    def fail(tasks: tuple[Any, ...], *, workers: int) -> tuple[Any, ...]:
        _ = tasks, workers
        raise RuntimeError("worker failed")

    monkeypatch.setattr(subject, "_run_worker_tasks", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        subject.run_production_generation(
            loader=_loader(),
            plan=_small_plan(),
            binding=_binding(),
            output_root=root,
            run_name="test-production",
            workers=1,
        )
    assert (root / "FAILED").is_file()
    with pytest.raises(subject.GenerationError, match="interrupted or stale-running"):
        subject.run_production_generation(
            loader=_loader(),
            plan=_small_plan(),
            binding=_binding(),
            output_root=root,
            run_name="test-production",
            workers=1,
            resume=True,
        )


def test_generated_run_reloads_and_reverifies_every_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_execution(monkeypatch)
    root = tmp_path / "production"
    produced = subject.run_production_generation(
        loader=_loader(),
        plan=_small_plan(),
        binding=_binding(),
        output_root=root,
        run_name="test-production",
        workers=2,
    )

    loaded = subject.load_generated_production_run(root)

    assert loaded.run_root == root
    assert loaded.loader == _loader()
    assert loaded.plan == _small_plan()
    assert loaded.binding == _binding()
    assert loaded.workers == 2
    assert loaded.result_fingerprint == produced.fingerprint
    assert loaded.source_code_fingerprint == produced.source_code_fingerprint
    assert loaded.dependency_lock_fingerprint == produced.dependency_lock_fingerprint


def test_generated_run_loader_rejects_nonterminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_execution(monkeypatch)
    root = tmp_path / "production"
    subject.run_production_generation(
        loader=_loader(),
        plan=_small_plan(),
        binding=_binding(),
        output_root=root,
        run_name="test-production",
        workers=1,
    )
    state = json.loads((root / "run-state.json").read_bytes())
    state["disposition"] = "running"
    (root / "run-state.json").write_bytes(subject._canonical_bytes(state))

    with pytest.raises(subject.IntegrityError, match="generated disposition"):
        subject.load_generated_production_run(root)


def test_clean_replay_regenerates_one_lf_and_hf_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    plan = _small_plan()
    binding = _binding()
    loaded = SimpleNamespace(
        run_root=source_root,
        loader=_loader(),
        plan=plan,
        binding=binding,
        result_fingerprint="a" * 64,
        source_code_fingerprint="b" * 64,
        dependency_lock_fingerprint="c" * 64,
    )
    monkeypatch.setattr(subject, "load_generated_production_run", lambda root: loaded)
    monkeypatch.setattr(
        subject,
        "inspect_executable_source",
        lambda root: SimpleNamespace(
            source_code_fingerprint="b" * 64,
            dependency_lock_fingerprint="c" * 64,
        ),
    )
    monkeypatch.setattr(
        subject, "load_production_generation_input", lambda spec: object()
    )
    executed: list[str] = []
    monkeypatch.setattr(
        subject,
        "execute_serial_work_unit",
        lambda *args, **kwargs: executed.append(kwargs["scope"].work_unit_id),
    )
    scopes = {item.work_unit_id: item for item in subject.production_work_units(plan)}
    monkeypatch.setattr(
        subject,
        "verify_serial_work_unit",
        lambda *args, **kwargs: _fake_receipt(scopes[kwargs["work_unit_id"]]),
    )
    output = tmp_path / "replay"

    report = subject.replay_representative_work_units(source_root, output_root=output)

    assert executed == [
        "lf-00000-lf000001-lf000001",
        "hf-00000-hf000001-hf000001",
    ]
    assert report.identical
    assert report.csv_shards_compared == 5
    assert (output / "clean-replay-report.json").is_file()


def test_execute_serial_work_unit_dispatches_lf_and_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = subject.ProductionPlan("test-50y", 50, 1, 1, 1, 1)
    lf, hf = subject.production_work_units(plan)
    generated: list[tuple[str, int]] = []
    receipts = {lf.work_unit_id: _fake_receipt(lf), hf.work_unit_id: _fake_receipt(hf)}

    monkeypatch.setattr(
        subject, "is_sealed_serial_generation_input", lambda value: True
    )
    monkeypatch.setattr(subject, "_validate_scope", lambda scope, value: None)

    def generate(value: object, *, family: str, path_entity: int) -> object:
        _ = value
        generated.append((family, path_entity))
        return SimpleNamespace(family=family)

    monkeypatch.setattr(subject, "generate_configured_serial_base_path", generate)
    monkeypatch.setattr(subject, "aggregate_lf_log_returns", lambda path: path)
    monkeypatch.setattr(subject, "aggregate_hf_weekly_log_returns", lambda path: path)
    monkeypatch.setattr(
        subject,
        "commit_lf_serial_work_unit",
        lambda *args, **kwargs: receipts[kwargs["work_unit_id"]],
    )
    monkeypatch.setattr(
        subject,
        "commit_hf_serial_work_unit",
        lambda *args, **kwargs: receipts[kwargs["work_unit_id"]],
    )
    monkeypatch.setattr(
        subject,
        "verify_serial_work_unit",
        lambda *args, **kwargs: receipts[kwargs["work_unit_id"]],
    )
    binding = StorageCommitBinding("1" * 64, "2" * 64, "3" * 64, "4" * 64)

    lf_result = subject.execute_serial_work_unit(
        object(),  # type: ignore[arg-type]
        output_root=tmp_path,
        binding=binding,
        scope=lf,
        worker_identity="worker",
    )
    hf_result = subject.execute_serial_work_unit(
        object(),  # type: ignore[arg-type]
        output_root=tmp_path,
        binding=binding,
        scope=hf,
        worker_identity="worker",
    )

    assert generated == [("LF", 1), ("HF", 1)]
    assert len(lf_result.sibling_sha256) == 3
    assert len(hf_result.sibling_sha256) == 2


def test_execute_serial_work_unit_real_engine_and_group_commit(tmp_path: Path) -> None:
    from test_serial_generator import _fixture

    generation_input, _, config, _ = _fixture()
    plan = subject.ProductionPlan.from_config(config)
    binding = StorageCommitBinding("1" * 64, "2" * 64, "3" * 64, "4" * 64)

    results = tuple(
        subject.execute_serial_work_unit(
            generation_input,
            output_root=tmp_path,
            binding=binding,
            scope=scope,
            worker_identity="real-test-worker",
        )
        for scope in subject.production_work_units(plan)
    )

    assert len(results) == 4
    assert sum(len(item.sibling_sha256) for item in results) == 10
    assert len(tuple((tmp_path / "consumer" / "returns").rglob("*.csv"))) == 10
    assert len(tuple((tmp_path / "receipts" / "storage").glob("*.json"))) == 4


def test_comparison_reports_exact_csv_hash_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    scopes = {
        scope.work_unit_id: scope
        for scope in subject.production_work_units(_small_plan())
    }
    monkeypatch.setattr(
        subject,
        "verify_serial_work_unit",
        lambda *args, **kwargs: _fake_receipt(scopes[kwargs["work_unit_id"]]),
    )

    report = subject.compare_production_outputs(
        left_root=left,
        right_root=right,
        plan=_small_plan(),
        left_binding=_binding(),
        right_binding=_binding(),
    )

    assert report.identical
    assert report.work_units_compared == 3
    assert report.csv_shards_compared == 8
    assert report.mismatches == ()


def test_comparison_rejects_different_simulation_identity(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    with pytest.raises(subject.GenerationError, match="equal simulation"):
        subject.compare_production_outputs(
            left_root=left,
            right_root=right,
            plan=_small_plan(),
            left_binding=_binding("4"),
            right_binding=_binding("a"),
        )


def test_rebind_changes_only_run_identity() -> None:
    original = _binding()
    rebound = subject.rebind_production_run(
        original, plan=_small_plan(), run_name="benchmark-w1"
    )

    assert rebound.run_fingerprint != original.run_fingerprint
    assert rebound.data_fingerprint == original.data_fingerprint
    assert rebound.simulation_fingerprint == original.simulation_fingerprint
    assert rebound.qualification_fingerprint == original.qualification_fingerprint
    assert rebound.materialization_fingerprint == original.materialization_fingerprint
    assert rebound.toolchain_fingerprint == original.toolchain_fingerprint


def test_benchmark_selects_fastest_worker_and_projects_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    throughputs = {1: 100.0, 2: 180.0, 4: 300.0, 9: 250.0}

    def run(**kwargs: object) -> object:
        workers = kwargs["workers"]
        assert isinstance(workers, int)
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        return SimpleNamespace(
            run_root=root,
            binding=kwargs["binding"],
            fingerprint=f"{workers:x}" * 64,
            workers=workers,
            wall_seconds=_small_plan().total_rows / throughputs[workers],
            rows_per_second=throughputs[workers],
            peak_process_tree_rss_bytes=workers * 1000,
        )

    monkeypatch.setattr(subject, "run_production_generation", run)
    monkeypatch.setattr(
        subject,
        "compare_production_outputs",
        lambda **kwargs: SimpleNamespace(identical=True),
    )

    report = subject.benchmark_production_worker_counts(
        loader=_loader(),
        plan=_small_plan(),
        base_binding=_binding(),
        output_parent=tmp_path / "benchmark",
    )

    assert report.worker_counts == (1, 2, 4, 9)
    assert report.selected_workers == 4
    assert report.selected_rows_per_second == 300.0
    assert report.canonical_projected_wall_seconds == pytest.approx(
        subject.CANONICAL_PRODUCTION_PLAN.total_rows / 300.0
    )
    assert report.all_csv_parity_passed
    assert report.minimum_parallel_efficiency == pytest.approx(250.0 / 900.0)
    stored = json.loads((tmp_path / "benchmark" / "benchmark-report.json").read_bytes())
    assert stored["fingerprint"] == report.fingerprint
    assert stored["selected_workers"] == 4


def test_benchmark_withholds_selection_when_csv_parity_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(**kwargs: object) -> object:
        workers = kwargs["workers"]
        assert isinstance(workers, int)
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        return SimpleNamespace(
            run_root=root,
            binding=kwargs["binding"],
            fingerprint=f"{workers:x}" * 64,
            workers=workers,
            wall_seconds=10.0,
            rows_per_second=float(workers * 100),
            peak_process_tree_rss_bytes=workers * 1000,
        )

    monkeypatch.setattr(subject, "run_production_generation", run)
    monkeypatch.setattr(
        subject,
        "compare_production_outputs",
        lambda **kwargs: SimpleNamespace(identical=False),
    )

    with pytest.raises(subject.IntegrityError, match="selection withheld"):
        subject.benchmark_production_worker_counts(
            loader=_loader(),
            plan=_small_plan(),
            base_binding=_binding(),
            output_parent=tmp_path / "benchmark",
            worker_counts=(1, 2),
        )
    assert not (tmp_path / "benchmark" / "benchmark-report.json").exists()


@pytest.mark.parametrize("counts", [(), (2, 4), (1, 1), (1, 0), (1, 4, 2)])
def test_benchmark_worker_registry_rejects_invalid_counts(
    counts: tuple[int, ...], tmp_path: Path
) -> None:
    with pytest.raises(subject.GenerationError, match="worker counts"):
        subject.benchmark_production_worker_counts(
            loader=_loader(),
            plan=_small_plan(),
            base_binding=_binding(),
            output_parent=tmp_path / "benchmark",
            worker_counts=counts,
        )
