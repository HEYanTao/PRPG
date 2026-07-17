from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import BaseModel

import prpg.v5.generation as generation_module
from prpg.errors import GenerationError, IntegrityError
from prpg.v5.config import ResolvedV5Inputs, load_v5_inputs
from prpg.v5.generation import (
    COARSE_HEADER,
    DAILY_HEADER,
    V5GenerationInput,
    V5GenerationPlan,
    V5PathRange,
    V5PathSelection,
    V5WorkUnit,
    build_v5_generation_plan,
    iter_v5_month_blocks,
    load_v5_generation_input,
    materialize_v5_path,
    run_v5_generation,
    select_v5_path,
    write_v5_work_unit,
)
from prpg.v5.validation import validate_v5_run

PROCESSED = "e2caa7756ec5f6b37806e6ef58e44fe827f741ed544e82b132a0cd2f1bbe1760"
MODEL = "be5df93d382a499dbf21c00b1e0be9d47bfda08bd387d0ac70b36d47ba88d6c7"


def _model_fingerprint(value: BaseModel) -> str:
    payload = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _configured_inputs(tmp_path: Path) -> ResolvedV5Inputs:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs" / "v5-canonical.yaml").read_text(encoding="utf-8")
    )
    seeds = yaml.safe_load(
        (root / "configs" / "v5-seed-registry.yaml").read_text(encoding="utf-8")
    )
    config["run"]["seed_registry"] = "seeds.yaml"
    config["run"]["artifact_root"] = str(root / "artifacts" / "v5")
    config["simulation"].update(
        {
            "paths": 11,
            "horizon_months": 24,
            "split": {"training": 5, "validation": 3, "test": 3},
            "paths_per_shard": 4,
        }
    )
    seeds["path_generation"].update(
        {
            "training_path_ids": [1, 5],
            "validation_path_ids": [6, 8],
            "test_path_ids": [9, 11],
        }
    )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "seeds.yaml").write_text(yaml.safe_dump(seeds), encoding="utf-8")
    return load_v5_inputs(tmp_path / "config.yaml")


def _synthetic_input(
    feature_names: tuple[str, str, str] = ("ACWI", "LQD", "MUB"),
) -> V5GenerationInput:
    daily = np.asarray(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
            [-0.01, -0.02, -0.03],
            [-0.04, -0.05, -0.06],
            [0.02, 0.01, 0.03],
            [0.01, 0.02, 0.04],
            [-0.02, 0.01, -0.01],
            [0.03, -0.01, 0.02],
        ],
        dtype=np.float64,
    )
    return V5GenerationInput(
        master_seed=20260716,
        scientific_version=16,
        maximum_path_entity=5_000,
        config_fingerprint="a" * 64,
        seed_registry_fingerprint="b" * 64,
        data_fingerprint="c" * 64,
        model_fingerprint="d" * 64,
        model_config_fingerprint="e" * 64,
        model_fit_seed_fingerprint="f" * 64,
        feature_names=feature_names,
        daily_returns=daily,
        month_start_offsets=np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        monthly_period_ordinals=np.arange(4, dtype=np.int64),
        decoded_states=np.arange(4, dtype=np.int64),
        transition_matrix=np.eye(4, dtype=np.float64),
        stationary_distribution=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )


def _file_hashes(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        file["relative_path"]: file["sha256"]
        for work_unit in manifest["work_units"]
        for file in work_unit["files"]
    }


def test_exact_two_stream_selection_is_repeatable_and_pool_conditioned() -> None:
    generation_input = _synthetic_input()

    first = select_v5_path(generation_input, 17, 12)
    second = select_v5_path(generation_input, 17, 12)

    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(
        first.source_month_indices, second.source_month_indices
    )
    np.testing.assert_array_equal(first.states, np.zeros(12, dtype=np.int64))
    np.testing.assert_array_equal(
        first.source_month_indices, np.zeros(12, dtype=np.int64)
    )


def test_whole_block_iteration_reorders_only_consumer_asset_columns() -> None:
    generation_input = _synthetic_input()
    selection = V5PathSelection(
        path_entity=1,
        states=np.asarray([0], dtype=np.int64),
        source_month_indices=np.asarray([0], dtype=np.int64),
    )

    block = next(iter_v5_month_blocks(generation_input, selection))

    np.testing.assert_array_equal(
        block.daily_log_returns,
        np.asarray([[0.01, 0.03, 0.02], [0.04, 0.06, 0.05]]),
    )


def test_generation_input_accepts_configured_proxy_tickers() -> None:
    generation_input = _synthetic_input(("SPY", "VCIT", "VTEB"))

    assert generation_input.feature_names == ("SPY", "VCIT", "VTEB")


def test_manifest_asset_mapping_uses_configured_proxy_tickers() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    assets = inputs.config.data.assets.model_copy(
        update={"equity": "SPY", "taxable_bond": "VCIT", "muni_bond": "VTEB"}
    )
    data = inputs.config.data.model_copy(update={"assets": assets})
    changed = replace(inputs, config=inputs.config.model_copy(update={"data": data}))

    assert generation_module._consumer_assets(changed) == {
        "equity": "SPY",
        "muni_bond": "VTEB",
        "taxable_bond": "VCIT",
    }


def test_daily_authoritative_path_uses_stable_log_sums() -> None:
    path = materialize_v5_path(_synthetic_input(), 1, 12)

    assert path.daily_log_returns.shape == (24, 3)
    assert path.monthly_log_returns.shape == (12, 3)
    assert path.quarterly_log_returns.shape == (4, 3)
    assert path.annual_log_returns.shape == (1, 3)
    np.testing.assert_array_equal(path.month_lengths, np.full(12, 2))
    np.testing.assert_allclose(path.monthly_log_returns[0], [0.05, 0.09, 0.07])
    np.testing.assert_allclose(
        path.quarterly_log_returns[0], path.monthly_log_returns[:3].sum(axis=0)
    )
    np.testing.assert_allclose(
        path.annual_log_returns[0], path.monthly_log_returns.sum(axis=0)
    )


def test_registered_and_explicit_plans_are_role_aligned() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    canonical = build_v5_generation_plan(inputs, profile="canonical")
    explicit = build_v5_generation_plan(
        inputs,
        profile="parity",
        path_ranges=(
            V5PathRange("training", 1, 7),
            V5PathRange("validation", 4001, 4001),
            V5PathRange("test", 4501, 4501),
        ),
        horizon_months=12,
        paths_per_shard=1,
    )

    assert canonical.path_count == 5_000
    assert len(canonical.work_units) == 100
    assert canonical.csv_file_count == 400
    assert len(explicit.work_units) == 9
    assert [unit.role for unit in explicit.work_units][-2:] == ["validation", "test"]


def test_configured_plan_uses_full_geometry_and_role_pure_partial_shards(
    tmp_path: Path,
) -> None:
    inputs = _configured_inputs(tmp_path)

    plan = build_v5_generation_plan(inputs, profile="configured")

    assert plan.path_count == 11
    assert plan.horizon_months == 24
    observed = [
        (unit.role, unit.first_path, unit.last_path) for unit in plan.work_units
    ]
    assert observed == [
        ("training", 1, 4),
        ("training", 5, 5),
        ("validation", 6, 8),
        ("test", 9, 11),
    ]
    assert plan.csv_file_count == 16


def test_configured_full_geometry_generates_with_reviewed_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _configured_inputs(tmp_path)
    monkeypatch.setattr(
        generation_module,
        "_source_provenance",
        lambda _path: {
            "git_commit": "1" * 40,
            "dirty_worktree": True,
            "requirements_lock_sha256": "2" * 64,
        },
    )

    result = run_v5_generation(
        inputs,
        model_fingerprint=MODEL,
        processed_data_fingerprint=PROCESSED,
        output_root=tmp_path / "run",
        run_name="unit-configured",
        profile="configured",
        workers=1,
        path_generation_seed=20_260_717,
    )

    assert result.path_count == 11
    assert result.work_units == 4
    assert result.csv_files == 16
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["path_generation_seed"] == 20_260_717
    report = validate_v5_run(inputs, result.run_root, expected_paths=11)
    assert report.paths == 11


def test_shard_writer_publishes_exact_four_schemas_and_rejects_overwrite(
    tmp_path: Path,
) -> None:
    work_unit = V5WorkUnit("training-00000-p000001-p000001", "training", 0, 1, 1)

    result = write_v5_work_unit(
        _synthetic_input(),
        output_root=tmp_path,
        work_unit=work_unit,
        horizon_months=12,
    )

    assert [item.frequency for item in result.files] == [
        "daily",
        "monthly",
        "quarterly",
        "annual",
    ]
    for record in result.files:
        lines = (
            (tmp_path / record.relative_path).read_text(encoding="utf-8").splitlines()
        )
        assert lines[0] == (
            DAILY_HEADER if record.frequency == "daily" else COARSE_HEADER
        )
        assert lines[1].startswith("P000001,")
        assert len(lines) - 1 == record.rows
    with pytest.raises(IntegrityError, match="not fresh"):
        write_v5_work_unit(
            _synthetic_input(),
            output_root=tmp_path,
            work_unit=work_unit,
            horizon_months=12,
        )


def test_canonical_loader_binds_exact_reviewed_model_and_month_library() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")

    generation_input = load_v5_generation_input(inputs, PROCESSED, MODEL)

    assert generation_input.feature_names == ("ACWI", "LQD", "MUB")
    assert [len(pool) for pool in generation_input.state_pools] == [30, 41, 103, 45]
    assert len(generation_input.monthly_period_ordinals) == 219


def test_path_seed_override_reuses_model_but_changes_path_rng() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    default = load_v5_generation_input(inputs, PROCESSED, MODEL)
    loader = generation_module.build_v5_generation_loader(
        inputs,
        processed_data_fingerprint=PROCESSED,
        model_fingerprint=MODEL,
        path_generation_seed=20_260_717,
    )
    overridden = generation_module._load_from_loader(loader)

    assert (
        "path_generation_seed"
        not in generation_module.build_v5_generation_loader(
            inputs,
            processed_data_fingerprint=PROCESSED,
            model_fingerprint=MODEL,
        ).as_dict()
    )
    assert loader.as_dict()["path_generation_seed"] == 20_260_717
    assert default.model_fingerprint == overridden.model_fingerprint
    assert default.model_config_fingerprint == overridden.model_config_fingerprint
    assert default.model_fit_seed_fingerprint == overridden.model_fit_seed_fingerprint
    assert default.master_seed == inputs.config.run.master_seed
    assert overridden.master_seed == 20_260_717
    first = select_v5_path(default, 1, 12)
    second = select_v5_path(overridden, 1, 12)
    assert not np.array_equal(first.source_month_indices, second.source_month_indices)
    identity = generation_module._simulation_identity(
        overridden,
        build_v5_generation_plan(inputs, profile="configured"),
        path_generation_seed_overridden=True,
    )
    assert identity["path_generation_seed"] == 20_260_717


def test_scoped_model_provenance_uses_configured_restart_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    model = inputs.config.model.model_copy(update={"deterministic_restarts": 7})
    changed = replace(inputs, config=inputs.config.model_copy(update={"model": model}))
    observed: dict[str, object] = {}

    class _ReviewStore:
        @staticmethod
        def read_document(_fingerprint: str, _name: str) -> dict[str, object]:
            return {
                "estimator_settings": model.model_dump(mode="python"),
                "restart_diagnostics": [
                    {"restart_index": index, "seed": index} for index in range(7)
                ],
            }

    def fake_restart_seeds(**kwargs: object) -> tuple[int, ...]:
        observed.update(kwargs)
        return tuple(range(7))

    monkeypatch.setattr(
        generation_module,
        "deterministic_hmm_restart_seeds",
        fake_restart_seeds,
    )

    generation_module._verify_scoped_model_fit_provenance(
        changed,
        _ReviewStore(),  # type: ignore[arg-type]
        MODEL,
    )

    assert observed["restart_count"] == 7


def test_reviewed_model_is_reusable_after_simulation_only_changes() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    split = inputs.config.simulation.split.model_copy(
        update={"training": 6, "validation": 3, "test": 3}
    )
    simulation = inputs.config.simulation.model_copy(
        update={
            "paths": 12,
            "horizon_months": 120,
            "split": split,
            "paths_per_shard": 5,
        }
    )
    execution = inputs.config.execution.model_copy(update={"workers": 3})
    run = inputs.config.run.model_copy(update={"output_root": Path("../runs/custom")})
    config = inputs.config.model_copy(
        update={"run": run, "simulation": simulation, "execution": execution}
    )
    path_seeds = inputs.seed_registry.path_generation.model_copy(
        update={
            "training_path_ids": (1, 6),
            "validation_path_ids": (7, 9),
            "test_path_ids": (10, 12),
        }
    )
    registry = inputs.seed_registry.model_copy(update={"path_generation": path_seeds})
    changed = replace(
        inputs,
        config=config,
        config_fingerprint=_model_fingerprint(config),
        seed_registry=registry,
        seed_registry_fingerprint=_model_fingerprint(registry),
    )

    generation_input = load_v5_generation_input(changed, PROCESSED, MODEL)

    assert generation_input.maximum_path_entity == 12
    assert generation_input.feature_names == ("ACWI", "LQD", "MUB")


def test_reviewed_model_rejects_model_or_model_fit_seed_changes() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    changed_model = inputs.config.model.model_copy(update={"minimum_state_months": 11})
    model_config = inputs.config.model_copy(update={"model": changed_model})
    with pytest.raises(IntegrityError, match="estimator settings"):
        load_v5_generation_input(
            replace(
                inputs,
                config=model_config,
                config_fingerprint=_model_fingerprint(model_config),
            ),
            PROCESSED,
            MODEL,
        )

    changed_run = inputs.config.run.model_copy(update={"master_seed": 20260717})
    seed_config = inputs.config.model_copy(update={"run": changed_run})
    changed_registry = inputs.seed_registry.model_copy(update={"master_seed": 20260717})
    with pytest.raises(IntegrityError, match="model-fit seed scope"):
        load_v5_generation_input(
            replace(
                inputs,
                config=seed_config,
                config_fingerprint=_model_fingerprint(seed_config),
                seed_registry=changed_registry,
                seed_registry_fingerprint=_model_fingerprint(changed_registry),
            ),
            PROCESSED,
            MODEL,
        )


def test_reviewed_model_rejects_changed_data_contract() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    flags = inputs.config.data.quality_flags.model_copy(
        update={"equity_abs_simple_return": 0.21}
    )
    data = inputs.config.data.model_copy(update={"quality_flags": flags})
    config = inputs.config.model_copy(update={"data": data})

    with pytest.raises(IntegrityError, match="transform contract"):
        load_v5_generation_input(
            replace(
                inputs,
                config=config,
                config_fingerprint=_model_fingerprint(config),
            ),
            PROCESSED,
            MODEL,
        )


def test_canonical_run_rejects_a_dirty_source_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    monkeypatch.setattr(
        generation_module,
        "_source_provenance",
        lambda _path: {
            "git_commit": "1" * 40,
            "dirty_worktree": True,
            "requirements_lock_sha256": "2" * 64,
        },
    )
    output = tmp_path / "canonical"

    with pytest.raises(GenerationError, match="clean worktree"):
        run_v5_generation(
            inputs,
            model_fingerprint=MODEL,
            processed_data_fingerprint=PROCESSED,
            output_root=output,
            run_name="unit-canonical",
            profile="canonical",
            workers=9,
        )

    assert not output.exists()


def test_canonical_run_rejects_path_seed_override_before_creating_output(
    tmp_path: Path,
) -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    output = tmp_path / "canonical-seed-override"

    with pytest.raises(GenerationError, match="does not permit a path-seed override"):
        run_v5_generation(
            inputs,
            model_fingerprint=MODEL,
            processed_data_fingerprint=PROCESSED,
            output_root=output,
            run_name="unit-canonical-seed-override",
            profile="canonical",
            workers=9,
            path_generation_seed=20_260_717,
        )

    assert not output.exists()


def test_same_explicit_plan_is_byte_identical_with_one_and_nine_workers(
    tmp_path: Path,
) -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    plan = V5GenerationPlan(
        profile="parity",
        horizon_months=12,
        paths_per_shard=1,
        path_ranges=(
            V5PathRange("training", 1, 7),
            V5PathRange("validation", 4001, 4001),
            V5PathRange("test", 4501, 4501),
        ),
    )

    serial = run_v5_generation(
        inputs,
        model_fingerprint=MODEL,
        processed_data_fingerprint=PROCESSED,
        output_root=tmp_path / "one",
        run_name="unit-parity-one",
        profile="parity",
        workers=1,
        plan=plan,
    )
    parallel = run_v5_generation(
        inputs,
        model_fingerprint=MODEL,
        processed_data_fingerprint=PROCESSED,
        output_root=tmp_path / "nine",
        run_name="unit-parity-nine",
        profile="parity",
        workers=9,
        plan=plan,
    )

    assert serial.simulation_fingerprint == parallel.simulation_fingerprint
    assert serial.materialization_fingerprint == parallel.materialization_fingerprint
    assert _file_hashes(serial.manifest_path) == _file_hashes(parallel.manifest_path)
    assert len(_file_hashes(serial.manifest_path)) == 36
    assert serial.total_rows == parallel.total_rows
