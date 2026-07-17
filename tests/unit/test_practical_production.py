from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prpg.config import load_config
from prpg.errors import GenerationError
from prpg.simulation import practical_production as subject
from prpg.simulation.production import ProductionPlan, production_work_units

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "canonical.yaml"


def _loader() -> subject.PracticalProductionLoader:
    return subject.build_practical_production_loader(
        project_root=ROOT,
        config_path=CONFIG,
        artifact_root=ROOT / "artifacts",
        artifact_fingerprint="1" * 64,
        processed_data_fingerprint="2" * 64,
    )


def _plan() -> ProductionPlan:
    return ProductionPlan(
        profile_id="practical-test-v1",
        years=50,
        lf_paths=2,
        hf_paths=1,
        lf_paths_per_shard=1,
        hf_paths_per_shard=1,
    )


def _receipt(scope: Any) -> SimpleNamespace:
    frequencies: tuple[tuple[str, int], ...]
    if scope.family == "LF":
        frequencies = (("monthly", 600), ("quarterly", 200), ("annual", 50))
    else:
        frequencies = (("daily", 12_600), ("weekly", 2_600))
    siblings = []
    for index, (frequency, rows) in enumerate(frequencies):
        siblings.append(
            {
                "frequency": frequency,
                "relative_path": (
                    f"consumer/returns/{scope.work_unit_id}-{frequency}.csv"
                ),
                "rows": rows,
                "bytes": rows * 100,
                "sha256": f"{index + 3:x}" * 64,
            }
        )
    return SimpleNamespace(
        work_unit_id=scope.work_unit_id,
        family=scope.family,
        first_path_entity=scope.first_path_entity,
        last_path_entity=scope.last_path_entity,
        fingerprint="9" * 64,
        identity={"siblings": siblings},
    )


def test_loader_and_binding_are_stable() -> None:
    loader = _loader()
    first = subject.build_practical_production_binding(
        loader, plan=_plan(), run_name="practical-test"
    )
    second = subject.build_practical_production_binding(
        loader, plan=_plan(), run_name="practical-test"
    )

    assert first == second
    assert len(loader.fingerprint) == 64
    assert first.storage.run_fingerprint == first.run_fingerprint


def test_worker_resolution_caps_auto_at_nine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject,
        "resource_snapshot",
        lambda *args, **kwargs: SimpleNamespace(
            resolved_workers=15,
            disk_free_bytes=10**12,
        ),
    )

    assert subject.resolve_practical_workers(load_config(CONFIG)) == 9


def test_run_verifies_receipts_and_writes_one_manifest_and_checksum_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scopes = production_work_units(_plan())
    by_id = {scope.work_unit_id: _receipt(scope) for scope in scopes}
    monkeypatch.setattr(subject, "_run_worker_tasks", lambda tasks, workers: ())
    monkeypatch.setattr(
        subject,
        "verify_serial_work_unit",
        lambda root, *, binding, work_unit_id: by_id[work_unit_id],
    )
    monkeypatch.setattr(
        subject,
        "resource_snapshot",
        lambda *args, **kwargs: SimpleNamespace(
            resolved_workers=1,
            disk_free_bytes=10**12,
        ),
    )
    output = tmp_path / "delivery"

    result = subject.run_practical_production(
        loader=_loader(),
        plan=_plan(),
        output_root=output,
        run_name="practical-test",
        workers=1,
    )

    assert result.work_units == 3
    assert result.csv_shards == 8
    assert result.total_rows == 16_900
    manifest = json.loads(result.manifest_path.read_bytes())
    assert manifest["status"] == "complete"
    assert manifest["run_name"] == "practical-test"
    assert manifest["git_commit"] == subject._git_output(
        ROOT, "rev-parse", "--verify", "HEAD"
    )
    assert isinstance(manifest["dirty_worktree"], bool)
    assert (
        manifest["requirements_lock_sha256"]
        == hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()
    )
    assert manifest["totals"] == {
        "paths": 3,
        "work_units": 3,
        "csv_shards": 8,
        "rows": 16_900,
        "bytes": 1_690_000,
    }
    assert manifest["manifest_fingerprint"] == result.manifest_fingerprint
    lines = result.checksum_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    assert all("  consumer/returns/" in line for line in lines)


def test_run_rejects_existing_output_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "delivery"
    output.mkdir()
    monkeypatch.setattr(
        subject,
        "resource_snapshot",
        lambda *args, **kwargs: SimpleNamespace(
            resolved_workers=1,
            disk_free_bytes=10**12,
        ),
    )

    with pytest.raises(GenerationError, match="already exists"):
        subject.run_practical_production(
            loader=_loader(),
            plan=_plan(),
            output_root=output,
            run_name="practical-test",
            workers=1,
        )


def test_run_rejects_more_than_nine_workers(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="at most nine"):
        subject.run_practical_production(
            loader=_loader(),
            plan=_plan(),
            output_root=tmp_path / "delivery",
            run_name="practical-test",
            workers=10,
        )


def test_source_provenance_records_commit_dirty_flag_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_content = b"numpy==2.3.5\n"
    (tmp_path / "requirements.lock").write_bytes(lock_content)
    responses = iter(("a" * 40, " M src/prpg/example.py"))
    monkeypatch.setattr(subject, "_git_output", lambda *args, **kwargs: next(responses))

    observed = subject._practical_source_provenance(tmp_path)

    assert observed == {
        "git_commit": "a" * 40,
        "dirty_worktree": True,
        "requirements_lock_sha256": hashlib.sha256(lock_content).hexdigest(),
    }


def test_only_full_canonical_plan_requires_clean_worktree() -> None:
    config = load_config(CONFIG)
    full = ProductionPlan.from_config(config)

    assert subject._requires_clean_worktree(
        plan=full, config=config, run_name="practical-k4-v1"
    )
    assert not subject._requires_clean_worktree(
        plan=full, config=config, run_name="practical-k4-smoke"
    )
    assert not subject._requires_clean_worktree(
        plan=_plan(), config=config, run_name="practical-test"
    )


def test_full_canonical_run_rejects_dirty_worktree_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subject,
        "_practical_source_provenance",
        lambda root: {
            "git_commit": "a" * 40,
            "dirty_worktree": True,
            "requirements_lock_sha256": "b" * 64,
        },
    )

    with pytest.raises(GenerationError, match="clean committed worktree"):
        subject.run_practical_production(
            loader=_loader(),
            plan=ProductionPlan.from_config(load_config(CONFIG)),
            output_root=tmp_path / "delivery",
            run_name="practical-k4-v1",
            workers=1,
        )
    assert not (tmp_path / "delivery").exists()


def test_invalid_config_path_is_a_clean_application_error(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="configuration path is invalid"):
        subject.build_practical_production_loader(
            project_root=ROOT,
            config_path=tmp_path / "missing.yaml",
            artifact_root=ROOT / "artifacts",
            artifact_fingerprint="1" * 64,
            processed_data_fingerprint="2" * 64,
        )
