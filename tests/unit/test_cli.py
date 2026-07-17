from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import prpg.cli as cli
import prpg.g3_launch as g3_launch
from prpg.cli import app
from prpg.errors import DataAcquisitionError, DataValidationError, ExitCode, ModelError

ROOT = Path(__file__).parents[2]
RUNNER = CliRunner()


class _FakeLaunchLease:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.closed = False

    def __enter__(self) -> _FakeLaunchLease:
        self.calls.append("lease_enter")
        return self

    def __exit__(self, *_: object) -> None:
        self.calls.append("lease_exit")
        self.closed = True


def _install_canonical_launch_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runner_error: ModelError | None = None,
) -> tuple[list[str], _FakeLaunchLease, dict[str, object]]:
    calls: list[str] = []
    state: dict[str, object] = {}
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "runs"
    resolved = SimpleNamespace(
        data=SimpleNamespace(artifact_root=artifact_root),
        model=SimpleNamespace(design_holdout_months=24),
        run=SimpleNamespace(output_root=output_root, master_seed=20_260_714),
        simulation=SimpleNamespace(scientific_version=6),
        execution=SimpleNamespace(workers="auto", reserve_physical_cores=1),
        fingerprint=lambda: "a" * 64,
    )
    calibration_input = object()
    resources = object()
    preflight = SimpleNamespace(
        fingerprint="f" * 64,
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        workers=9,
    )
    reference = SimpleNamespace(
        run_id="2" * 64,
        path=output_root / "calibration" / ("2" * 64),
    )
    receipt = SimpleNamespace(fingerprint="3" * 64)
    lease = _FakeLaunchLease(calls)
    checkpoint_store = object()
    model_store = object()
    pair_store = object()
    evidence_store = object()
    artifact_pair = SimpleNamespace(
        selected_k=4,
        reference=SimpleNamespace(
            fingerprint="5" * 64,
            path=artifact_root / "scientific-artifact-pairs" / "sha256" / ("5" * 64),
        ),
        design=SimpleNamespace(
            reference=SimpleNamespace(
                fingerprint="6" * 64,
                path=artifact_root / "model" / "sha256" / ("6" * 64),
            )
        ),
        production=SimpleNamespace(
            reference=SimpleNamespace(
                fingerprint="7" * 64,
                path=artifact_root / "model" / "sha256" / ("7" * 64),
            )
        ),
    )
    finalization = SimpleNamespace(
        completion_marker=SimpleNamespace(fingerprint="4" * 64),
        artifact_pair=artifact_pair,
        durable_evidence=SimpleNamespace(
            reference=SimpleNamespace(
                fingerprint="8" * 64,
                path=artifact_root / "g3-evidence" / "sha256" / ("8" * 64),
            )
        ),
    )
    execution_result = SimpleNamespace(
        finalization=finalization,
        publications=tuple(object() for _ in range(26)),
        generation_authorized=False,
    )

    def fake_load_config(path: Path) -> object:
        calls.append("load_config")
        state["config_path"] = path
        return resolved

    def fake_processed_store(root: Path) -> object:
        calls.append("processed_store")
        state["processed_store_root"] = root
        return object()

    def fake_load_input(
        store: object,
        fingerprint: str,
        *,
        calibration_config_fingerprint: str,
        holdout_months: int,
    ) -> object:
        calls.append("load_input")
        state["input_arguments"] = (
            store,
            fingerprint,
            calibration_config_fingerprint,
            holdout_months,
        )
        return calibration_input

    def fake_resources(
        root: Path,
        *,
        requested_workers: str | int,
        reserve_cores: int,
    ) -> object:
        calls.append("resources")
        state["resource_arguments"] = (
            root,
            requested_workers,
            reserve_cores,
        )
        return resources

    def fake_preflight(config: object, value: object, snapshot: object) -> object:
        calls.append("preflight")
        state["preflight_arguments"] = (config, value, snapshot)
        return preflight

    class FakeRunStore:
        def initialize(self, identity: object) -> object:
            calls.append("initialize")
            state["initialized_identity"] = identity
            return reference

        def plan_launch(self, run_id: str, **kwargs: object) -> object:
            calls.append("plan")
            state["plan_arguments"] = (run_id, kwargs)
            return object()

        def start_launch(self, run_id: str, **kwargs: object) -> _FakeLaunchLease:
            calls.append("start")
            state["start_arguments"] = (run_id, kwargs)
            return lease

    run_store = FakeRunStore()

    def fake_run_store(root: Path) -> object:
        calls.append("run_store")
        state["run_store_root"] = root
        return run_store

    def fake_store(name: str, expected: object):
        def constructor(root: Path) -> object:
            calls.append(name)
            state[f"{name}_root"] = root
            return expected

        return constructor

    def fake_identity(**kwargs: object) -> object:
        calls.append("identity")
        state["identity_arguments"] = kwargs
        return "identity"

    def fake_persist(
        store: object,
        *,
        run_id: str,
        preflight: object,
    ) -> object:
        calls.append("persist")
        state["persist_arguments"] = (store, run_id, preflight)
        return receipt

    def fake_runner(value: object, **kwargs: object) -> object:
        calls.append("runner")
        state["runner_arguments"] = (value, kwargs)
        if runner_error is not None:
            raise runner_error
        return execution_result

    monkeypatch.setattr(g3_launch, "load_config", fake_load_config)
    monkeypatch.setattr(g3_launch, "ProcessedSnapshotStore", fake_processed_store)
    monkeypatch.setattr(g3_launch, "load_calibration_input", fake_load_input)
    monkeypatch.setattr(g3_launch, "resource_snapshot", fake_resources)
    monkeypatch.setattr(g3_launch, "canonical_g3_preflight", fake_preflight)
    monkeypatch.setattr(g3_launch, "CalibrationRunStore", fake_run_store)
    monkeypatch.setattr(
        g3_launch,
        "CalibrationCheckpointStore",
        fake_store("checkpoint_store", checkpoint_store),
    )
    monkeypatch.setattr(
        g3_launch,
        "ModelArtifactStore",
        fake_store("model_store", model_store),
    )
    monkeypatch.setattr(
        g3_launch,
        "ScientificArtifactPairStore",
        fake_store("pair_store", pair_store),
    )
    monkeypatch.setattr(
        g3_launch,
        "G3EvidenceStore",
        fake_store("evidence_store", evidence_store),
    )
    monkeypatch.setattr(g3_launch, "default_run_identity", fake_identity)
    monkeypatch.setattr(g3_launch, "persist_canonical_g3_preflight", fake_persist)
    monkeypatch.setattr(g3_launch, "run_canonical_g3_success_execution", fake_runner)
    state.update(
        {
            "resolved": resolved,
            "calibration_input": calibration_input,
            "preflight": preflight,
            "run_store": run_store,
            "checkpoint_store": checkpoint_store,
            "model_store": model_store,
            "pair_store": pair_store,
            "evidence_store": evidence_store,
        }
    )
    return calls, lease, state


def test_root_help_exposes_full_planned_surface() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "config",
        "doctor",
        "data",
        "calibrate",
        "g4-reference",
        "validation",
        "benchmark",
        "plan",
        "generate",
        "validate",
        "qualify",
        "release",
        "inspect",
        "reproduce",
        "run",
    ):
        assert command in result.output


def test_version() -> None:
    result = RUNNER.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0"


def test_config_validate_json_is_side_effect_free() -> None:
    config_path = ROOT / "configs" / "canonical.yaml"
    before = config_path.stat()

    result = RUNNER.invoke(app, ["config", "validate", str(config_path), "--json"])

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event["event"] == "config_validated"
    assert len(event["config_fingerprint"]) == 64
    after = config_path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)


def test_config_validate_missing_file_uses_stable_exit_code(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app, ["config", "validate", str(tmp_path / "missing.yaml"), "--json"]
    )

    assert result.exit_code == ExitCode.CONFIGURATION
    event = json.loads(result.output)
    assert event["error_type"] == "ConfigurationError"
    assert event["exit_code"] == ExitCode.CONFIGURATION


def test_canonical_plan_has_exact_rows_values_and_shards() -> None:
    result = RUNNER.invoke(
        app, ["plan", str(ROOT / "configs" / "canonical.yaml"), "--json"]
    )

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event["rows"] == {
        "low_monthly": 3_000_000,
        "low_quarterly": 1_000_000,
        "low_annual": 250_000,
        "high_daily": 12_600_000,
        "high_weekly": 2_600_000,
    }
    assert event["total_rows"] == 19_450_000
    assert event["return_values"] == 58_350_000
    assert event["csv_shards"]["total"] == 190


@pytest.mark.parametrize(
    "arguments",
    [
        ["calibrate"],
        ["validation", "calibrate"],
        ["validate"],
        ["qualify"],
        ["release"],
        ["run"],
    ],
)
def test_later_phase_commands_fail_explicitly(arguments: list[str]) -> None:
    result = RUNNER.invoke(app, arguments)

    assert result.exit_code == ExitCode.NOT_IMPLEMENTED
    assert "not implemented" in result.output
    assert "Traceback" not in result.output


def test_release_attest_requires_explicit_completed_run_arguments() -> None:
    result = RUNNER.invoke(app, ["release", "attest"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output
    assert "Traceback" not in result.output


def test_release_finalize_uses_explicit_plan_and_emits_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = object()
    monkeypatch.setattr(cli, "_release_plan_from_json", lambda path: plan)
    monkeypatch.setattr(
        cli,
        "finalize_complete",
        lambda run_root, *, plan: SimpleNamespace(
            run_id="canonical-prpg-v1",
            root_sha256sums_sha256="1" * 64,
            complete_sha256="2" * 64,
            run_manifest_sha256="3" * 64,
            covered_file_count=205,
        ),
    )

    result = RUNNER.invoke(
        app,
        [
            "release",
            "finalize",
            str(tmp_path / "run"),
            "--plan",
            str(tmp_path / "plan.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event["event"] == "release_complete_finalized"
    assert event["covered_file_count"] == 205
    assert event["released"] is False


def test_release_attest_writes_external_bytes_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b'{"decision":"release"}\n'
    monkeypatch.setattr(
        cli, "build_external_release_attestation", lambda *args, **kwargs: content
    )
    output = tmp_path / "attestations" / "release.json"

    result = RUNNER.invoke(
        app,
        [
            "release",
            "attest",
            str(tmp_path / "run"),
            "--output",
            str(output),
            "--verifier",
            "independent-reviewer",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert output.read_bytes() == content
    event = json.loads(result.output)
    assert event["event"] == "external_release_attestation_created"
    assert len(event["sha256"]) == 64


def test_release_publish_reverifies_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    released = SimpleNamespace(
        complete=SimpleNamespace(
            run_id="canonical-prpg-v1", root_sha256sums_sha256="1" * 64
        ),
        attestation_sha256="2" * 64,
        verifier_identity="independent-reviewer",
        released_sha256="3" * 64,
    )
    monkeypatch.setattr(cli, "publish_released", lambda *args, **kwargs: released)
    monkeypatch.setattr(cli, "verify_released", lambda *args, **kwargs: released)

    result = RUNNER.invoke(
        app,
        [
            "release",
            "publish",
            str(tmp_path / "run"),
            "--attestation",
            str(tmp_path / "release.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event["event"] == "release_published"
    assert event["released"] is True


def test_inspect_reloads_generated_run_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    loaded = SimpleNamespace(
        run_root=root,
        result_fingerprint="1" * 64,
        binding=SimpleNamespace(
            run_fingerprint="2" * 64,
            simulation_fingerprint="3" * 64,
            qualification_fingerprint="4" * 64,
            materialization_fingerprint="5" * 64,
        ),
        loader=SimpleNamespace(
            g3_evidence_fingerprint="6" * 64,
            g5_authority_fingerprint="7" * 64,
        ),
        workers=9,
        plan=SimpleNamespace(
            lf_paths=5_000,
            hf_paths=1_000,
            years=50,
            total_rows=19_450_000,
            csv_shard_count=190,
        ),
    )
    monkeypatch.setattr(cli, "load_generated_production_run", lambda path: loaded)

    result = RUNNER.invoke(app, ["inspect", str(root), "--json"])

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event["event"] == "production_run_inspected"
    assert event["paths"] == {"low_frequency": 5_000, "high_frequency": 1_000}
    assert event["total_rows"] == 19_450_000
    assert event["released"] is False


def test_reproduce_runs_representative_clean_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = SimpleNamespace(
        source_run_fingerprint="1" * 64,
        replay_run_fingerprint="2" * 64,
        source_result_fingerprint="3" * 64,
        work_unit_ids=("lf-unit", "hf-unit"),
        csv_shards_compared=5,
        elapsed_seconds=1.25,
        identical=True,
        fingerprint="4" * 64,
    )
    monkeypatch.setattr(
        cli, "replay_representative_work_units", lambda *args, **kwargs: report
    )
    output = tmp_path / "replay"

    result = RUNNER.invoke(
        app,
        [
            "reproduce",
            str(tmp_path / "run"),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event["event"] == "production_clean_replay_completed"
    assert event["identical"] is True
    assert event["csv_shards_compared"] == 5


def test_generate_runs_frozen_production_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from prpg.simulation.production import ProductionBinding

    loader = object()
    binding = ProductionBinding(
        data_fingerprint="1" * 64,
        simulation_fingerprint="2" * 64,
        qualification_fingerprint="3" * 64,
        materialization_fingerprint="4" * 64,
        toolchain_fingerprint="5" * 64,
        run_fingerprint="6" * 64,
    )
    calls: dict[str, object] = {}

    def build_loader(**kwargs: object) -> object:
        calls["loader"] = kwargs
        return loader

    def build_binding(
        observed_loader: object, *, plan: object, run_name: str
    ) -> ProductionBinding:
        calls["binding"] = (observed_loader, plan, run_name)
        return binding

    def resources(*args: object, **kwargs: object) -> object:
        calls["resources"] = (args, kwargs)
        return SimpleNamespace(resolved_workers=9)

    estimate = SimpleNamespace(
        estimated_peak_memory_bytes=5_905_580_032,
        required_free_disk_bytes=7_482_562_560,
    )

    def resource_estimate(**kwargs: object) -> object:
        calls["estimate"] = kwargs
        return estimate

    output = tmp_path / "production"

    def run_generation(**kwargs: object) -> object:
        calls["run"] = kwargs
        plan = kwargs["plan"]
        return SimpleNamespace(
            run_root=output.resolve(),
            binding=binding,
            fingerprint="7" * 64,
            workers=9,
            generated_work_units=70,
            reused_work_units=0,
            wall_seconds=120.0,
            rows_per_second=162_083.33333333334,
            peak_process_tree_rss_bytes=1_000_000_000,
            plan=plan,
        )

    monkeypatch.setattr(cli, "build_production_loader_spec", build_loader)
    monkeypatch.setattr(cli, "build_production_binding", build_binding)
    monkeypatch.setattr(cli, "resource_snapshot", resources)
    monkeypatch.setattr(cli, "production_resource_estimate", resource_estimate)
    monkeypatch.setattr(cli, "run_production_generation", run_generation)

    invoked = RUNNER.invoke(
        app,
        [
            "generate",
            "a" * 64,
            "b" * 64,
            "--output",
            str(output),
            "--workers",
            "9",
            "--config",
            str(ROOT / "configs" / "canonical.yaml"),
            "--json",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    event = json.loads(invoked.output)
    assert event["event"] == "production_generation_completed"
    assert event["workers"] == 9
    assert event["csv_shards"] == 190
    assert event["total_rows"] == 19_450_000
    assert event["total_values"] == 58_350_000
    assert event["generation_complete"] is True
    assert event["data_validated"] is False
    assert event["released"] is False
    run_kwargs = calls["run"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["loader"] is loader
    assert run_kwargs["binding"] == binding
    assert run_kwargs["run_name"] == "canonical-prpg-v1"
    assert run_kwargs["workers"] == 9
    assert run_kwargs["resume"] is False


def test_benchmark_runs_fixed_worker_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from prpg.simulation.production import ProductionBinding

    loader = object()
    binding = ProductionBinding(
        data_fingerprint="1" * 64,
        simulation_fingerprint="2" * 64,
        qualification_fingerprint="3" * 64,
        materialization_fingerprint="4" * 64,
        toolchain_fingerprint="5" * 64,
        run_fingerprint="6" * 64,
    )
    observations = tuple(
        SimpleNamespace(
            workers=workers,
            wall_seconds=100.0 / workers,
            rows_per_second=workers * 1000.0,
            parallel_efficiency=1.0,
            peak_process_tree_rss_bytes=workers * 100_000_000,
            csv_parity_with_one_worker=True,
            run_root=tmp_path / f"workers-{workers}",
        )
        for workers in (1, 2, 4, 9)
    )
    report = SimpleNamespace(
        fingerprint="7" * 64,
        benchmark_plan_fingerprint="8" * 64,
        worker_counts=(1, 2, 4, 9),
        observations=observations,
        selected_workers=9,
        selected_rows_per_second=9000.0,
        canonical_projected_wall_seconds=2161.1111111111113,
        canonical_projected_wall_hours=0.6003086419753087,
        all_csv_parity_passed=True,
        minimum_parallel_efficiency=1.0,
    )
    monkeypatch.setattr(cli, "build_production_loader_spec", lambda **kwargs: loader)
    monkeypatch.setattr(
        cli,
        "build_production_binding",
        lambda *args, **kwargs: binding,
    )
    captured: dict[str, object] = {}

    def run_benchmark(**kwargs: object) -> object:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(cli, "benchmark_production_worker_counts", run_benchmark)

    invoked = RUNNER.invoke(
        app,
        [
            "benchmark",
            "a" * 64,
            "b" * 64,
            "--output",
            str(tmp_path / "benchmark"),
            "--config",
            str(ROOT / "configs" / "canonical.yaml"),
            "--json",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    event = json.loads(invoked.output)
    assert event["event"] == "g6_production_benchmark_completed"
    assert event["worker_counts"] == [1, 2, 4, 9]
    assert event["benchmark_rows"] == 2_875_000
    assert event["selected_workers"] == 9
    assert event["all_csv_parity_passed"] is True
    assert event["benchmark_report_path"].endswith("benchmark-report.json")
    assert captured["worker_counts"] == (1, 2, 4, 9)


def test_calibrate_preflight_only_is_read_only_and_emits_bound_estimate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved = SimpleNamespace(
        data=SimpleNamespace(artifact_root=tmp_path / "artifacts"),
        model=SimpleNamespace(design_holdout_months=24),
        run=SimpleNamespace(output_root=tmp_path / "runs"),
        execution=SimpleNamespace(workers="auto", reserve_physical_cores=1),
        fingerprint=lambda: "a" * 64,
    )
    calibration_input = object()
    resources = object()
    preflight = SimpleNamespace(
        fingerprint="f" * 64,
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        toolchain_fingerprint="1" * 64,
        workers=9,
        estimated_peak_bytes=9_000_000_000,
        physical_memory_bytes=18_000_000_000,
        estimated_disk_bytes=4_000_000_000,
        disk_free_bytes=80_000_000_000,
        checks=("one", "two"),
    )
    calls: list[str] = []
    monkeypatch.setattr(g3_launch, "load_config", lambda path: resolved)
    monkeypatch.setattr(
        g3_launch,
        "ProcessedSnapshotStore",
        lambda root: calls.append(f"store:{root}") or object(),
    )
    monkeypatch.setattr(
        g3_launch,
        "load_calibration_input",
        lambda *args, **kwargs: calls.append("input") or calibration_input,
    )
    monkeypatch.setattr(
        g3_launch,
        "resource_snapshot",
        lambda *args, **kwargs: calls.append("resources") or resources,
    )
    monkeypatch.setattr(
        g3_launch,
        "canonical_g3_preflight",
        lambda config, value, snapshot: calls.append("preflight") or preflight,
    )

    event = g3_launch.execute_g3_launch(
        config_path=tmp_path / "canonical.yaml",
        canonical_launch=False,
    )

    assert event["event"] == "g3_preflight_passed"
    assert event["preflight_fingerprint"] == "f" * 64
    assert event["source_code_fingerprint"] == "d" * 64
    assert event["dependency_lock_fingerprint"] == "e" * 64
    assert event["toolchain_fingerprint"] == "1" * 64
    assert event["estimated_memory_fraction"] == 0.5
    assert event["launch_created"] is False
    assert event["scientific_result_observed"] is False
    assert calls == [
        f"store:{tmp_path / 'artifacts'}",
        "input",
        "resources",
        "preflight",
    ]
    assert not (tmp_path / "runs").exists()


def test_calibrate_rejects_combined_modes_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: calls.append(str(path)),
    )

    result = RUNNER.invoke(
        app,
        [
            "calibrate",
            "--preflight-only",
            "--canonical-launch",
            "--json",
        ],
    )

    assert result.exit_code == ExitCode.CONFIGURATION
    assert calls == []
    event = json.loads(result.output)
    assert event["event"] == "error"
    assert event["error_type"] == "ConfigurationError"
    assert event["details"] == {"modes": ["--preflight-only", "--canonical-launch"]}


def test_g4_reference_cli_loads_exact_evidence_and_emits_nonauthorizing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"
    output_root = tmp_path / "g4-output"
    evidence_fingerprint = "a" * 64
    resolved = SimpleNamespace(
        data=SimpleNamespace(artifact_root=artifact_root),
        model=SimpleNamespace(design_holdout_months=24),
        run=SimpleNamespace(output_root=run_root),
        fingerprint=lambda: "b" * 64,
    )
    calibration_input = object()
    evidence = object()
    stores = {
        "run": object(),
        "checkpoint": object(),
        "model": object(),
    }

    class FakeEvidenceStore:
        def load(self, fingerprint: str, **kwargs: object) -> object:
            calls.append(("evidence_load", fingerprint, kwargs))
            return evidence

    report = SimpleNamespace(
        fingerprint="c" * 64,
        sha256="d" * 64,
        path=output_root / "g4-reference-report.json",
        identity={"fixture": {"base_paths": 30, "rows": 6_760, "files": 25}},
        structural_validation=SimpleNamespace(passed=True),
        g4_passed=True,
        canonical_authority=False,
        generation_authorized=False,
        releasable=False,
    )

    monkeypatch.setattr(cli, "load_config", lambda path: resolved)
    monkeypatch.setattr(
        cli,
        "ProcessedSnapshotStore",
        lambda root: calls.append(("processed_store", root)) or object(),
    )
    monkeypatch.setattr(
        cli,
        "load_calibration_input",
        lambda *args, **kwargs: calls.append(("calibration_input", args, kwargs))
        or calibration_input,
    )
    monkeypatch.setattr(
        cli,
        "CalibrationRunStore",
        lambda root: calls.append(("run_store", root)) or stores["run"],
    )
    monkeypatch.setattr(
        cli,
        "CalibrationCheckpointStore",
        lambda root: calls.append(("checkpoint_store", root)) or stores["checkpoint"],
    )
    monkeypatch.setattr(
        cli,
        "ModelArtifactStore",
        lambda root: calls.append(("model_store", root)) or stores["model"],
    )
    monkeypatch.setattr(
        cli,
        "G3EvidenceStore",
        lambda root: calls.append(("evidence_store", root)) or FakeEvidenceStore(),
    )

    def fake_reference_runner(**kwargs: object) -> object:
        calls.append(("reference_runner", kwargs))
        return report

    monkeypatch.setattr(cli, "run_g4_reference_fixture", fake_reference_runner)

    result = RUNNER.invoke(
        app,
        [
            "g4-reference",
            evidence_fingerprint,
            "--output",
            str(output_root),
            "--config",
            str(tmp_path / "canonical.yaml"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event == {
        "canonical_authority": False,
        "event": "g4_reference_completed_noncanonical",
        "files": 25,
        "g3_evidence_fingerprint": evidence_fingerprint,
        "g4_passed": True,
        "generation_authorized": False,
        "paths": 30,
        "releasable": False,
        "report_fingerprint": "c" * 64,
        "report_path": str(output_root / "g4-reference-report.json"),
        "report_sha256": "d" * 64,
        "rows": 6_760,
        "structural_validation_passed": True,
    }
    evidence_call = next(item for item in calls if item[0] == "evidence_load")
    assert evidence_call[1] == evidence_fingerprint
    assert evidence_call[2] == {
        "run_store": stores["run"],
        "checkpoint_store": stores["checkpoint"],
        "model_store": stores["model"],
    }
    runner_call = next(item for item in calls if item[0] == "reference_runner")
    assert runner_call[1] == {
        "g3_evidence": evidence,
        "config": resolved,
        "calibration_input": calibration_input,
        "output_root": output_root.resolve(),
    }


def test_canonical_launch_does_not_create_stores_before_preflight_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved = SimpleNamespace(
        data=SimpleNamespace(artifact_root=tmp_path / "artifacts"),
        model=SimpleNamespace(design_holdout_months=24),
        run=SimpleNamespace(
            output_root=tmp_path / "runs",
            master_seed=20_260_714,
        ),
        simulation=SimpleNamespace(scientific_version=6),
        execution=SimpleNamespace(workers="auto", reserve_physical_cores=1),
        fingerprint=lambda: "a" * 64,
    )
    store_calls: list[str] = []
    monkeypatch.setattr(g3_launch, "load_config", lambda path: resolved)
    monkeypatch.setattr(g3_launch, "ProcessedSnapshotStore", lambda root: object())
    monkeypatch.setattr(
        g3_launch, "load_calibration_input", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        g3_launch, "resource_snapshot", lambda *args, **kwargs: object()
    )

    def reject_preflight(*_: object) -> object:
        raise ModelError("synthetic preflight rejection")

    monkeypatch.setattr(g3_launch, "canonical_g3_preflight", reject_preflight)
    for name in (
        "CalibrationRunStore",
        "CalibrationCheckpointStore",
        "ModelArtifactStore",
        "ScientificArtifactPairStore",
        "G3EvidenceStore",
    ):
        monkeypatch.setattr(
            g3_launch,
            name,
            lambda root, store=name: store_calls.append(store) or object(),
        )

    with pytest.raises(ModelError, match="synthetic preflight rejection"):
        g3_launch.execute_g3_launch(
            config_path=tmp_path / "canonical.yaml",
            canonical_launch=True,
        )
    assert store_calls == []
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "artifacts").exists()


def test_canonical_launch_composes_exact_fresh_run_and_emits_one_json_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, lease, state = _install_canonical_launch_fakes(monkeypatch, tmp_path)

    event = g3_launch.execute_g3_launch(
        config_path=tmp_path / "canonical.yaml",
        canonical_launch=True,
    )

    assert event == {
        "artifact_pair_fingerprint": "5" * 64,
        "artifact_pair_path": str(
            tmp_path / "artifacts" / "scientific-artifact-pairs" / "sha256" / ("5" * 64)
        ),
        "completion_fingerprint": "4" * 64,
        "design_artifact_fingerprint": "6" * 64,
        "design_artifact_path": str(
            tmp_path / "artifacts" / "model" / "sha256" / ("6" * 64)
        ),
        "event": "g3_canonical_launch_completed",
        "g3_evidence_fingerprint": "8" * 64,
        "g3_evidence_path": str(
            tmp_path / "artifacts" / "g3-evidence" / "sha256" / ("8" * 64)
        ),
        "generation_authorized": False,
        "preflight_fingerprint": "f" * 64,
        "preflight_receipt_fingerprint": "3" * 64,
        "production_artifact_fingerprint": "7" * 64,
        "production_artifact_path": str(
            tmp_path / "artifacts" / "model" / "sha256" / ("7" * 64)
        ),
        "run_id": "2" * 64,
        "run_path": str(tmp_path / "runs" / "calibration" / ("2" * 64)),
        "selected_n_states": 4,
        "task_count": 26,
        "workers": 9,
    }
    assert calls == [
        "load_config",
        "processed_store",
        "load_input",
        "resources",
        "preflight",
        "run_store",
        "checkpoint_store",
        "model_store",
        "pair_store",
        "evidence_store",
        "identity",
        "initialize",
        "persist",
        "plan",
        "start",
        "lease_enter",
        "runner",
        "lease_exit",
    ]
    assert lease.closed
    preflight = state["preflight"]
    assert state["identity_arguments"] == {
        "config_fingerprint": "a" * 64,
        "processed_data_fingerprint": "b" * 64,
        "calibration_input_fingerprint": "c" * 64,
        "source_code_fingerprint": "d" * 64,
        "dependency_lock_fingerprint": "e" * 64,
        "workers": 9,
    }
    persist_store, persist_run_id, persist_preflight = state["persist_arguments"]
    assert persist_store is state["run_store"]
    assert persist_run_id == "2" * 64
    assert persist_preflight is preflight
    plan_run_id, plan_kwargs = state["plan_arguments"]
    assert plan_run_id == "2" * 64
    assert plan_kwargs == {
        "preflight_fingerprint": "f" * 64,
        "workers": 9,
    }
    start_run_id, start_kwargs = state["start_arguments"]
    assert start_run_id == "2" * 64
    assert start_kwargs == {
        "preflight_fingerprint": "f" * 64,
        "workers": 9,
        "resume": False,
    }
    runner_input, runner_kwargs = state["runner_arguments"]
    assert runner_input is state["calibration_input"]
    assert runner_kwargs == {
        "preflight": preflight,
        "run_store": state["run_store"],
        "checkpoint_store": state["checkpoint_store"],
        "lease": lease,
        "master_seed": 20_260_714,
        "scientific_version": 6,
        "model_store": state["model_store"],
        "pair_store": state["pair_store"],
        "evidence_store": state["evidence_store"],
    }
    assert state["run_store_root"] == tmp_path / "runs"
    for name in (
        "checkpoint_store",
        "model_store",
        "pair_store",
        "evidence_store",
    ):
        assert state[f"{name}_root"] == tmp_path / "artifacts"


def test_canonical_launch_runner_error_closes_lease_without_completion_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls, lease, _state = _install_canonical_launch_fakes(
        monkeypatch,
        tmp_path,
        runner_error=ModelError("synthetic runner failure"),
    )

    exit_code = g3_launch.main(
        ["--canonical-launch", "--json"],
    )

    output = capsys.readouterr().out
    assert exit_code == ExitCode.MODEL
    assert len(output.splitlines()) == 1
    event = json.loads(output)
    assert event["event"] == "error"
    assert event["error_type"] == "ModelError"
    assert event["message"] == "synthetic runner failure"
    assert "g3_canonical_launch_completed" not in output
    assert calls[-3:] == ["lease_enter", "runner", "lease_exit"]
    assert lease.closed


def test_canonical_launch_runner_error_has_clean_human_output_without_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _calls, lease, _state = _install_canonical_launch_fakes(
        monkeypatch,
        tmp_path,
        runner_error=ModelError("synthetic runner failure"),
    )

    exit_code = g3_launch.main(["--canonical-launch"])
    output = capsys.readouterr().out

    assert exit_code == ExitCode.MODEL
    assert output.strip() == "ERROR [20]: synthetic runner failure"
    assert "{}" not in output
    assert lease.closed


def test_doctor_reports_credential_presence_not_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "NEVER-PRINT-THIS")

    result = RUNNER.invoke(app, ["doctor", "--json"])

    assert "NEVER-PRINT-THIS" not in result.output
    event = json.loads(result.output)
    expected_exit = (
        0
        if sys.version_info[:2] == (3, 11) and event["dependency_versions_match"]
        else ExitCode.CONFIGURATION
    )
    assert result.exit_code == expected_exit
    assert event["fred_api_key_present"] is True
    assert "physical_memory_bytes" in event


def test_data_fetch_emits_snapshot_and_journal_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = SimpleNamespace(
        snapshot=SimpleNamespace(
            fingerprint="a" * 64,
            path=tmp_path / "artifacts" / "data" / "sha256" / ("a" * 64),
            reused=False,
        ),
        acquisition_group_id="acq-20260714T120000Z-123456789abc",
        journal_path=tmp_path / "journal",
        source_count=7,
    )
    monkeypatch.setattr("prpg.cli.acquire_raw_snapshot", lambda config: result)

    invoked = RUNNER.invoke(
        app,
        ["data", "fetch", str(ROOT / "configs" / "canonical.yaml"), "--json"],
    )

    assert invoked.exit_code == 0
    event = json.loads(invoked.output)
    assert event["event"] == "raw_snapshot_created"
    assert event["snapshot_fingerprint"] == "a" * 64
    assert event["source_count"] == 7
    assert event["acquisition_group_id"] == result.acquisition_group_id


def test_data_validate_emits_processed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = {
        "status": "passed",
        "aligned": {
            "monthly_vectors": 217,
            "daily_vectors": 4547,
            "first_month": "2008-04",
            "last_month": "2026-05",
        },
    }
    reference = SimpleNamespace(
        fingerprint="b" * 64,
        path=tmp_path / "processed" / ("b" * 64),
        reused=False,
        manifest={"qc_report": qc},
    )
    monkeypatch.setattr(
        "prpg.cli.build_processed_snapshot", lambda config, fingerprint: reference
    )

    invoked = RUNNER.invoke(
        app,
        [
            "data",
            "validate",
            "a" * 64,
            "--config",
            str(ROOT / "configs" / "canonical.yaml"),
            "--json",
        ],
    )

    assert invoked.exit_code == 0
    event = json.loads(invoked.output)
    assert event["event"] == "processed_snapshot_validated"
    assert event["data_fingerprint"] == "b" * 64
    assert event["monthly_vectors"] == 217
    assert event["daily_vectors"] == 4547


def test_human_output_covers_success_and_structured_error(tmp_path: Path) -> None:
    success = RUNNER.invoke(
        app, ["config", "validate", str(ROOT / "configs" / "canonical.yaml")]
    )
    failure = RUNNER.invoke(app, ["config", "validate", str(tmp_path / "missing.yaml")])

    assert success.exit_code == 0
    assert "event: config_validated" in success.output
    assert "config_fingerprint:" in success.output
    assert failure.exit_code == ExitCode.CONFIGURATION
    assert "ERROR [2]: configuration file could not be read" in failure.output
    assert '"reason": "FileNotFoundError"' in failure.output


def test_plan_missing_config_uses_configuration_exit(tmp_path: Path) -> None:
    result = RUNNER.invoke(app, ["plan", str(tmp_path / "missing.yaml"), "--json"])

    assert result.exit_code == ExitCode.CONFIGURATION
    assert json.loads(result.output)["error_type"] == "ConfigurationError"


def test_doctor_fails_closed_when_a_pinned_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "numpy":
            raise cli.PackageNotFoundError(distribution)
        return cli._EXPECTED_RUNTIME_VERSIONS[distribution]

    monkeypatch.setattr(cli, "version", fake_version)

    result = RUNNER.invoke(app, ["doctor", "--json"])

    event = json.loads(result.output)
    assert result.exit_code == ExitCode.CONFIGURATION
    assert event["status"] == "unsupported_environment"
    assert event["dependency_versions"]["numpy"] is None
    assert event["dependency_versions_match"] is False


def test_physical_memory_probe_handles_os_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sysconf(name: str) -> int:
        raise OSError(name)

    monkeypatch.setattr(cli.os, "sysconf", fail_sysconf)

    assert cli._physical_memory_bytes() is None


def test_data_fetch_reports_acquisition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fetch(config: object) -> None:
        raise DataAcquisitionError("bounded test failure", details={"source": "ACWI"})

    monkeypatch.setattr(cli, "acquire_raw_snapshot", fail_fetch)

    result = RUNNER.invoke(
        app,
        ["data", "fetch", str(ROOT / "configs" / "canonical.yaml"), "--json"],
    )

    event = json.loads(result.output)
    assert result.exit_code == ExitCode.DATA_ACQUISITION
    assert event["error_type"] == "DataAcquisitionError"
    assert event["details"] == {"source": "ACWI"}


def test_data_validate_reports_processed_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_validation(config: object, fingerprint: str) -> None:
        raise DataValidationError(
            "processed snapshot rejected", details={"fingerprint": fingerprint}
        )

    monkeypatch.setattr(cli, "build_processed_snapshot", fail_validation)

    result = RUNNER.invoke(
        app,
        [
            "data",
            "validate",
            "a" * 64,
            "--config",
            str(ROOT / "configs" / "canonical.yaml"),
            "--json",
        ],
    )

    event = json.loads(result.output)
    assert result.exit_code == ExitCode.DATA_VALIDATION
    assert event["error_type"] == "DataValidationError"
    assert event["details"]["fingerprint"] == "a" * 64


def test_v5_help_exposes_lean_generation_and_validation_surface() -> None:
    v5_help = RUNNER.invoke(app, ["v5", "--help"])
    generate_help = RUNNER.invoke(app, ["v5", "generate", "--help"])

    assert v5_help.exit_code == 0
    assert "generate" in v5_help.output
    assert "validate" in v5_help.output
    assert generate_help.exit_code == 0
    for option in (
        "--model",
        "--processed",
        "--output",
        "--run-name",
        "--workers",
        "--path-seed",
        "--profile",
        "--config",
        "--json",
    ):
        assert option in generate_help.output


@pytest.mark.parametrize(
    ("profile", "path_count"),
    [
        ("smoke", 1),
        ("parity", 90),
        ("pilot", 500),
        ("configured", 73),
        ("canonical", 5_000),
    ],
)
def test_v5_generate_delegates_exact_profile_and_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    path_count: int,
) -> None:
    calls: dict[str, object] = {}
    inputs = object()

    def fake_load(path: Path) -> object:
        calls["config"] = path
        return inputs

    def fake_generate(value: object, **kwargs: object) -> object:
        calls["inputs"] = value
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            as_dict=lambda: {
                "profile": profile,
                "path_count": path_count,
                "total_rows": 123,
            }
        )

    monkeypatch.setattr(cli, "load_v5_inputs", fake_load)
    monkeypatch.setattr(cli, "run_v5_generation", fake_generate)
    output = tmp_path / f"v5-{profile}"

    result = RUNNER.invoke(
        app,
        [
            "v5",
            "generate",
            "--model",
            "a" * 64,
            "--processed",
            "b" * 64,
            "--output",
            str(output),
            "--run-name",
            f"v5-{profile}",
            "--workers",
            "9",
            "--profile",
            profile,
            "--json",
        ],
    )

    assert result.exit_code == 0
    event = json.loads(result.output)
    assert event == {
        "event": "v5_generation_completed",
        "path_count": path_count,
        "profile": profile,
        "total_rows": 123,
    }
    assert calls["inputs"] is inputs
    assert calls["config"] == Path("configs/v5-canonical.yaml")
    assert calls["kwargs"] == {
        "model_fingerprint": "a" * 64,
        "processed_data_fingerprint": "b" * 64,
        "output_root": output,
        "run_name": f"v5-{profile}",
        "profile": profile,
        "workers": 9,
        "path_generation_seed": None,
    }


def test_v5_generate_defaults_workers_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    inputs = SimpleNamespace(
        config=SimpleNamespace(execution=SimpleNamespace(workers=7))
    )
    monkeypatch.setattr(cli, "load_v5_inputs", lambda path: inputs)

    def fake_generate(value: object, **kwargs: object) -> object:
        calls["arguments"] = (value, kwargs)
        return SimpleNamespace(as_dict=lambda: {"profile": "configured"})

    monkeypatch.setattr(cli, "run_v5_generation", fake_generate)
    result = RUNNER.invoke(
        app,
        [
            "v5",
            "generate",
            "--model",
            "a" * 64,
            "--processed",
            "b" * 64,
            "--output",
            str(tmp_path / "configured"),
            "--run-name",
            "configured-run",
            "--profile",
            "configured",
            "--json",
        ],
    )

    assert result.exit_code == 0
    arguments = calls["arguments"]
    assert isinstance(arguments, tuple)
    kwargs = arguments[1]
    assert isinstance(kwargs, dict)
    assert kwargs["workers"] == 7
    assert kwargs["path_generation_seed"] is None


def test_v5_generate_delegates_path_seed_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = SimpleNamespace(
        config=SimpleNamespace(execution=SimpleNamespace(workers=3))
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_v5_inputs", lambda path: inputs)

    def fake_generate(value: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return SimpleNamespace(as_dict=lambda: {"profile": "configured"})

    monkeypatch.setattr(cli, "run_v5_generation", fake_generate)

    result = RUNNER.invoke(
        app,
        [
            "v5",
            "generate",
            "--model",
            "a" * 64,
            "--processed",
            "b" * 64,
            "--output",
            str(tmp_path / "configured"),
            "--run-name",
            "configured-seed-run",
            "--profile",
            "configured",
            "--path-seed",
            "424242",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert observed["path_generation_seed"] == 424_242


@pytest.mark.parametrize(
    ("profile", "expected_paths"),
    [
        ("smoke", 1),
        ("parity", 90),
        ("pilot", 500),
        ("configured", 73),
        ("canonical", 5_000),
    ],
)
def test_v5_validate_emits_json_and_enforces_profile_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
    expected_paths: int,
) -> None:
    calls: dict[str, object] = {}
    inputs = SimpleNamespace(
        config=SimpleNamespace(simulation=SimpleNamespace(paths=73))
    )

    monkeypatch.setattr(cli, "load_v5_inputs", lambda path: inputs)

    def fake_validate(
        value: object,
        run_root: Path,
        *,
        expected_paths: int | None = None,
    ) -> object:
        calls["arguments"] = (value, run_root, expected_paths)
        return SimpleNamespace(
            as_dict=lambda: {
                "status": "passed",
                "path_count": expected_paths,
                "total_rows": 456,
            }
        )

    monkeypatch.setattr(cli, "validate_v5_run", fake_validate)
    run_root = tmp_path / f"v5-{profile}"
    result = RUNNER.invoke(
        app,
        ["v5", "validate", str(run_root), "--profile", profile, "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "event": "v5_run_validated",
        "path_count": expected_paths,
        "status": "passed",
        "total_rows": 456,
    }
    assert calls["arguments"] == (inputs, run_root, expected_paths)


def test_console_main_invokes_typer_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(cli, "app", lambda: calls.append(True))

    cli.main()

    assert calls == [True]
