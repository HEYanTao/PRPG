from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import prpg.cli as cli

ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def test_generate_k4_supports_smoke_config_and_path_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    loader = object()
    monkeypatch.setattr(
        cli,
        "build_practical_production_loader",
        lambda **kwargs: captured.setdefault("loader", kwargs) and loader,
    )
    monkeypatch.setattr(
        cli,
        "resolve_practical_workers",
        lambda config, *, requested_workers: 1,
    )

    def run(**kwargs: object) -> SimpleNamespace:
        captured["run"] = kwargs
        plan = kwargs["plan"]
        output = Path(kwargs["output_root"])
        return SimpleNamespace(
            run_root=output,
            manifest_path=output / "manifest.json",
            checksum_path=output / "checksums.sha256",
            manifest_fingerprint="a" * 64,
            workers=1,
            work_units=3,
            csv_shards=8,
            total_rows=plan.total_rows,
            wall_seconds=1.25,
        )

    monkeypatch.setattr(cli, "run_practical_production", run)
    output = tmp_path / "smoke"
    result = RUNNER.invoke(
        cli.app,
        [
            "generate-k4",
            "1" * 64,
            "--output",
            str(output),
            "--config",
            str(ROOT / "configs" / "smoke.yaml"),
            "--lf-paths",
            "3",
            "--hf-paths",
            "1",
            "--lf-shard-size",
            "2",
            "--workers",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["event"] == "practical_k4_generation_completed"
    assert payload["paths"] == 4
    run_call = captured["run"]
    assert isinstance(run_call, dict)
    plan = run_call["plan"]
    assert plan.years == 2
    assert plan.lf_paths == 3
    assert plan.hf_paths == 1
    assert plan.lf_paths_per_shard == 2
    assert plan.hf_paths_per_shard == 1


def test_root_help_includes_practical_k4_generation() -> None:
    result = RUNNER.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "prepare-k4" in result.stdout
    assert "generate-k4" in result.stdout
    assert "validate-k4" in result.stdout


def test_prepare_k4_persists_reproduced_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = SimpleNamespace(
        fingerprint="1" * 64,
        model_fingerprint="2" * 64,
        monthly_source_states=np.asarray(
            [0] * 11 + [1] * 26 + [2] * 95 + [3] * 61,
            dtype=np.int64,
        ),
        iterations=37,
        log_likelihood=-752.7283424245143,
    )
    monkeypatch.setattr(cli, "load_calibration_input", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "reproduce_owner_approved_k4", lambda value: artifact)
    monkeypatch.setattr(
        cli,
        "PracticalK4ArtifactStore",
        lambda root: SimpleNamespace(
            persist=lambda value: SimpleNamespace(directory=tmp_path / "artifact")
        ),
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "prepare-k4",
            "--config",
            str(ROOT / "configs" / "canonical.yaml"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["event"] == "practical_k4_artifact_prepared"
    assert payload["artifact_fingerprint"] == "1" * 64
    assert payload["state_counts"] == [11, 26, 95, 61]


def test_validate_k4_emits_compact_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = SimpleNamespace(
        family="LF",
        frequency="monthly",
        paths=2,
        periods_per_path=24,
        shards=1,
        rows=48,
    )
    monkeypatch.setattr(
        cli,
        "validate_practical_delivery",
        lambda root: SimpleNamespace(
            status="passed",
            profile_id="test",
            years=2,
            lf_paths=2,
            hf_paths=1,
            work_units=2,
            csv_shards=5,
            total_rows=690,
            total_bytes=1234,
            checksums_verified=5,
            aggregation_paths_checked=3,
            aggregation_values_checked=378,
            duplicate_base_paths=0,
            manifest_sha256="3" * 64,
            elapsed_seconds=0.1,
            rows_per_second=6900.0,
            datasets=(dataset,),
        ),
    )

    result = RUNNER.invoke(cli.app, ["validate-k4", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["event"] == "practical_k4_delivery_validated"
    assert payload["status"] == "passed"
    assert payload["datasets"] == [
        {
            "family": "LF",
            "frequency": "monthly",
            "paths": 2,
            "periods_per_path": 24,
            "shards": 1,
            "rows": 48,
        }
    ]


def test_practical_commands_report_missing_config_cleanly(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        cli.app,
        ["prepare-k4", "--config", str(tmp_path / "missing.yaml"), "--json"],
    )

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["event"] == "error"
    assert payload["message"] == "practical configuration path is invalid"
