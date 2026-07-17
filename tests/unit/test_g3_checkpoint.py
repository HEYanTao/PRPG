from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_checkpoint import (
    CHECKPOINT_ARTIFACT_KIND,
    CHECKPOINT_STORAGE_DIRECTORY,
    CalibrationCheckpointBinding,
    CalibrationCheckpointStore,
    checkpoint_dependency_map,
)
from prpg.model.run_store import CalibrationRunStore, default_run_identity


def _binding(**changes: str) -> CalibrationCheckpointBinding:
    values = {
        "run_id": "a" * 64,
        "config_fingerprint": "b" * 64,
        "processed_data_fingerprint": "c" * 64,
        "calibration_input_fingerprint": "d" * 64,
        "preflight_fingerprint": "e" * 64,
    }
    values.update(changes)
    return CalibrationCheckpointBinding(**values)


def _prefix(store: CalibrationCheckpointStore, binding: CalibrationCheckpointBinding):
    input_checkpoint = store.create(
        task_id="input_integrity",
        binding=binding,
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"verified": True},
    )
    viability = store.create(
        task_id="design_candidate_viability",
        binding=binding,
        dependencies=(input_checkpoint,),
        selected_n_states=None,
        arrays={"eligible_states": np.asarray([2, 3], dtype=np.int64)},
        state={"eligible": [2, 3]},
    )
    selected = store.create(
        task_id="design_predictive_selection",
        binding=binding,
        dependencies=(viability,),
        selected_n_states=3,
        arrays={"state_means": np.zeros((3, 4), dtype=np.float64)},
        state={"selected_n_states": 3},
    )
    return input_checkpoint, viability, selected


def test_checkpoint_is_separate_typed_and_never_authorizes_generation(
    tmp_path: Path,
) -> None:
    store = CalibrationCheckpointStore(tmp_path / "artifacts")
    checkpoint = store.create(
        task_id="input_integrity",
        binding=_binding(),
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"verified": True},
    )

    assert not checkpoint.reused
    assert checkpoint.artifact_kind == CHECKPOINT_ARTIFACT_KIND
    assert not checkpoint.generation_authorized
    assert checkpoint.verification_scope == "checkpoint_integrity_and_schema"
    assert CHECKPOINT_STORAGE_DIRECTORY in checkpoint.path.parts
    assert checkpoint.path.is_relative_to(store.checkpoint_root)
    assert tuple(checkpoint.arrays) == ("checkpoint_marker",)
    assert int(checkpoint.arrays["checkpoint_marker"][0]) == 1
    assert not checkpoint.arrays["checkpoint_marker"].flags.writeable
    loaded = store.load(checkpoint.fingerprint)
    assert loaded.reused
    assert loaded.state["verified"] is True


def test_final_model_namespace_categorically_rejects_checkpoint_fingerprint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    checkpoint = CalibrationCheckpointStore(root).create(
        task_id="input_integrity",
        binding=_binding(),
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"verified": True},
    )

    with pytest.raises(IntegrityError, match="store path|missing"):
        ModelArtifactStore(root).verify(checkpoint.fingerprint)


def test_checkpoint_enforces_exact_dependency_graph_and_frozen_selected_k(
    tmp_path: Path,
) -> None:
    store = CalibrationCheckpointStore(tmp_path / "artifacts")
    input_checkpoint, viability, selected = _prefix(store, _binding())

    with pytest.raises(ModelError, match="exact task graph"):
        store.create(
            task_id="design_predictive_selection",
            binding=_binding(),
            dependencies=(input_checkpoint,),
            selected_n_states=3,
            arrays={},
            state={"selected": 3},
        )
    with pytest.raises(ModelError, match="preselection"):
        store.create(
            task_id="design_candidate_viability",
            binding=_binding(),
            dependencies=(input_checkpoint,),
            selected_n_states=3,
            arrays={},
            state={"eligible": True},
        )
    with pytest.raises(ModelError, match="change or promote"):
        store.create(
            task_id="sealed_holdout_veto",
            binding=_binding(),
            dependencies=(selected,),
            selected_n_states=4,
            arrays={},
            state={"passed": True},
        )
    holdout = store.create(
        task_id="sealed_holdout_veto",
        binding=_binding(),
        dependencies=(selected,),
        selected_n_states=3,
        arrays={},
        state={"passed": True},
    )
    assert holdout.selected_n_states == 3
    assert checkpoint_dependency_map(holdout) == {
        "design_predictive_selection": selected.fingerprint
    }
    assert viability.selected_n_states is None


def test_checkpoint_rejects_cross_run_dependency_and_bad_inputs(tmp_path: Path) -> None:
    store = CalibrationCheckpointStore(tmp_path / "artifacts")
    input_checkpoint = store.create(
        task_id="input_integrity",
        binding=_binding(),
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"verified": True},
    )
    with pytest.raises(ModelError, match="binding"):
        store.create(
            task_id="design_candidate_viability",
            binding=_binding(run_id="f" * 64),
            dependencies=(input_checkpoint,),
            selected_n_states=None,
            arrays={},
            state={"eligible": True},
        )
    with pytest.raises(ModelError, match="non-empty"):
        store.create(
            task_id="design_candidate_viability",
            binding=_binding(),
            dependencies=(input_checkpoint,),
            selected_n_states=None,
            arrays={},
            state={},
        )
    with pytest.raises(ModelError, match="run id"):
        store.create(
            task_id="input_integrity",
            binding=_binding(run_id="bad"),
            dependencies=(),
            selected_n_states=None,
            arrays={},
            state={"verified": True},
        )
    with pytest.raises(ModelError, match="closed registry"):
        store.create(
            task_id="invented_task",
            binding=_binding(),
            dependencies=(),
            selected_n_states=None,
            arrays={},
            state={"verified": True},
        )


def test_checkpoint_rejects_secrets_nonfinite_arrays_and_unsafe_names(
    tmp_path: Path,
) -> None:
    store = CalibrationCheckpointStore(tmp_path / "artifacts")
    kwargs = {
        "task_id": "input_integrity",
        "binding": _binding(),
        "dependencies": (),
        "selected_n_states": None,
    }
    with pytest.raises(ModelError, match="secret-like"):
        store.create(
            **kwargs,
            arrays={},
            state={"api_token": "not-allowed"},
        )
    with pytest.raises(ModelError, match="finite"):
        store.create(
            **kwargs,
            arrays={"bad": np.asarray([np.nan])},
            state={"verified": True},
        )
    with pytest.raises(ModelError, match="name"):
        store.create(
            **kwargs,
            arrays={"../bad": np.asarray([1])},
            state={"verified": True},
        )


def test_recursive_load_detects_tamper_in_dependency_prefix(tmp_path: Path) -> None:
    store = CalibrationCheckpointStore(tmp_path / "artifacts")
    input_checkpoint, _, selected = _prefix(store, _binding())
    manifest = input_checkpoint.path / "model-manifest.json"
    value = json.loads(manifest.read_text())
    value["identity"]["model_metadata"]["task_id"] = "artifact_integrity"
    manifest.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(IntegrityError, match="hash"):
        store.load(selected.fingerprint)


def test_checkpoint_lookup_uses_verified_run_store_result(tmp_path: Path) -> None:
    run_store = CalibrationRunStore(tmp_path / "runs")
    identity = default_run_identity(
        config_fingerprint="b" * 64,
        processed_data_fingerprint="c" * 64,
        calibration_input_fingerprint="d" * 64,
        source_code_fingerprint="f" * 64,
        dependency_lock_fingerprint="1" * 64,
        workers=9,
    )
    run = run_store.initialize(identity)
    preflight = "e" * 64
    binding = _binding(run_id=run.run_id)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    checkpoint = checkpoint_store.create(
        task_id="input_integrity",
        binding=binding,
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"verified": True},
    )
    run_store.plan_launch(run.run_id, preflight_fingerprint=preflight, workers=9)
    with run_store.start_launch(
        run.run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        run_store.write_task_result(
            lease,
            task_id="input_integrity",
            status="passed",
            payload={"checkpoint_fingerprint": checkpoint.fingerprint},
        )
        loaded = checkpoint_store.load_from_task_result(
            run_store,
            run_id=run.run_id,
            task_id="input_integrity",
        )
        assert loaded.fingerprint == checkpoint.fingerprint

    wrong = replace(binding, run_id="f" * 64)
    wrong_checkpoint = checkpoint_store.create(
        task_id="input_integrity",
        binding=wrong,
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"verified": True},
    )
    assert wrong_checkpoint.binding != loaded.binding


def test_checkpoint_store_rejects_symlink_namespace(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / CHECKPOINT_STORAGE_DIRECTORY).symlink_to(external, target_is_directory=True)
    store = CalibrationCheckpointStore(root)

    with pytest.raises(IntegrityError, match="symbolic-link"):
        store.create(
            task_id="input_integrity",
            binding=_binding(),
            dependencies=(),
            selected_n_states=None,
            arrays={},
            state={"verified": True},
        )


def test_checkpoint_dependency_map_rejects_wrong_type() -> None:
    with pytest.raises(ModelError, match="wrong type"):
        checkpoint_dependency_map(object())  # type: ignore[arg-type]
