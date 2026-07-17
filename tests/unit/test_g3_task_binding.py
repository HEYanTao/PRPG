from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prpg.errors import IntegrityError, ModelError
from prpg.model import g3_preflight as preflight_module
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointBinding,
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_preflight import (
    CanonicalG3Preflight,
    canonical_g3_resource_estimate,
    persist_canonical_g3_preflight,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.g3_task_binding import (
    SCIENTIFIC_TASK_BINDING_SCHEMA_ID,
    SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION,
    SCIENTIFIC_TASK_SLOT_BY_ID,
    SCIENTIFIC_TASK_SLOT_REGISTRY,
    SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT,
    SCIENTIFIC_TASK_SLOT_REGISTRY_SCHEMA_ID,
    TASK_SOURCE_MATERIALIZATION_SLOT,
    ScientificTaskBinding,
    build_scientific_task_binding,
    load_scientific_task_binding,
    scientific_task_slot_spec,
    verify_scientific_task_binding,
)
from prpg.model.ppw_reference import REFERENCE_SHA256, UPSTREAM_SHA256
from prpg.model.run_store import (
    CalibrationRunStore,
    LaunchLease,
    default_run_identity,
)
from prpg.provenance import inspect_g3_execution_closure


def _sha(value: object) -> str:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _exact_preflight(disk_free_bytes: int) -> CanonicalG3Preflight:
    """Build exact evidence without reproducing the large CalibrationInput fixture."""

    source = inspect_g3_execution_closure()
    runtime_fingerprint = _sha(
        {
            "python_version": preflight_module.CANONICAL_PYTHON_VERSION,
            "python_implementation": "CPython",
            "platform_system": preflight_module.CANONICAL_PLATFORM,
            "machine": preflight_module.CANONICAL_MACHINE,
            "dependency_versions": dict(preflight_module.EXPECTED_RUNTIME_VERSIONS),
        }
    )
    estimate = canonical_g3_resource_estimate()
    provisional = CanonicalG3Preflight(
        config_fingerprint=preflight_module.CANONICAL_CONFIG_FINGERPRINT,
        processed_data_fingerprint=(
            preflight_module.CANONICAL_PROCESSED_DATA_FINGERPRINT
        ),
        calibration_input_fingerprint=(
            preflight_module.CANONICAL_CALIBRATION_INPUT_FINGERPRINT
        ),
        contract_fingerprint=_sha(preflight_module._contract_identity()),
        source_code_fingerprint=source.source_code_fingerprint,
        dependency_lock_fingerprint=source.dependency_lock_fingerprint,
        toolchain_fingerprint=preflight_module._toolchain_fingerprint(
            runtime_fingerprint,
            source,
        ),
        ppw_vendored_fingerprint=REFERENCE_SHA256,
        ppw_upstream_fingerprint=UPSTREAM_SHA256,
        task_registry_fingerprint=G3_TASK_REGISTRY_FINGERPRINT,
        planned_look_registry_fingerprint=G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
        logical_cpu_count=preflight_module.CANONICAL_LOGICAL_CPUS,
        physical_memory_bytes=preflight_module.CANONICAL_PHYSICAL_MEMORY_BYTES,
        disk_free_bytes=disk_free_bytes,
        workers=preflight_module.CANONICAL_WORKERS,
        estimated_peak_bytes=estimate.estimated_peak_bytes,
        estimated_disk_bytes=estimate.estimated_disk_bytes,
        resource_estimate_fingerprint=estimate.fingerprint,
        checks=preflight_module._CANONICAL_PREFLIGHT_CHECKS,
        fingerprint="",
    )
    value = replace(provisional, fingerprint=_sha(provisional.identity_dict()))
    object.__setattr__(
        value,
        "_authority_seal",
        preflight_module._G3_PREFLIGHT_CAPABILITY,
    )
    return value


def _stores(
    tmp_path: Path,
) -> tuple[
    CalibrationRunStore,
    CalibrationCheckpointStore,
    str,
    CanonicalG3Preflight,
    CanonicalG3Preflight,
]:
    launch = _exact_preflight(20_000_000_000)
    live = _exact_preflight(21_000_000_000)
    run_store = CalibrationRunStore(tmp_path / "runs")
    run = run_store.initialize(
        default_run_identity(
            config_fingerprint=launch.config_fingerprint,
            processed_data_fingerprint=launch.processed_data_fingerprint,
            calibration_input_fingerprint=launch.calibration_input_fingerprint,
            source_code_fingerprint=launch.source_code_fingerprint,
            dependency_lock_fingerprint=launch.dependency_lock_fingerprint,
            workers=launch.workers,
        )
    )
    persist_canonical_g3_preflight(
        run_store,
        run_id=run.run_id,
        preflight=launch,
    )
    run_store.plan_launch(
        run.run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    )
    return (
        run_store,
        CalibrationCheckpointStore(tmp_path / "artifacts"),
        run.run_id,
        launch,
        live,
    )


def _fingerprint(task_id: str, namespace: str, slot: str) -> str:
    return hashlib.sha256(f"{task_id}:{namespace}:{slot}".encode()).hexdigest()


def _slots(task_id: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    spec = scientific_task_slot_spec(task_id)
    return (
        {
            slot: _fingerprint(task_id, "component", slot)
            for slot in spec.scientific_component_slots
        },
        {
            slot: _fingerprint(task_id, "materialization", slot)
            for slot in spec.materialization_slots
        },
        {slot: _fingerprint(task_id, "rng", slot) for slot in spec.rng_slots},
    )


def _build(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    run_id: str,
    task_id: str,
    live_preflight: CanonicalG3Preflight,
) -> ScientificTaskBinding:
    components, materializations, rng = _slots(task_id)
    return build_scientific_task_binding(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id=task_id,
        live_preflight=live_preflight,
        scientific_component_fingerprints=components,
        materialization_fingerprints=materializations,
        rng_fingerprints=rng,
    )


def _checkpoint_binding(
    run_store: CalibrationRunStore,
    run_id: str,
) -> CalibrationCheckpointBinding:
    reference = run_store.verify(run_id)
    launch = run_store.read_launch(run_id)
    return CalibrationCheckpointBinding(
        run_id=run_id,
        config_fingerprint=reference.identity.config_fingerprint,
        processed_data_fingerprint=reference.identity.processed_data_fingerprint,
        calibration_input_fingerprint=(
            reference.identity.calibration_input_fingerprint
        ),
        preflight_fingerprint=launch.preflight_fingerprint,
    )


def _publish_prefix_through_selection(
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    lease: LaunchLease,
) -> CalibrationCheckpointReference:
    binding = _checkpoint_binding(run_store, lease.run_id)
    input_checkpoint = checkpoint_store.create(
        task_id="input_integrity",
        binding=binding,
        dependencies=(),
        selected_n_states=None,
        arrays={},
        state={"passed": True},
    )
    run_store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload={"checkpoint_fingerprint": input_checkpoint.fingerprint},
    )
    viability = checkpoint_store.create(
        task_id="design_candidate_viability",
        binding=binding,
        dependencies=(input_checkpoint,),
        selected_n_states=None,
        arrays={"candidate_states": np.asarray([2, 3], dtype=np.int64)},
        state={"viable": [2, 3]},
    )
    run_store.write_task_result(
        lease,
        task_id="design_candidate_viability",
        status="passed",
        payload={"checkpoint_fingerprint": viability.fingerprint},
    )
    selection = checkpoint_store.create(
        task_id="design_predictive_selection",
        binding=binding,
        dependencies=(viability,),
        selected_n_states=3,
        arrays={"selected": np.asarray([3], dtype=np.int64)},
        state={"selected_n_states": 3},
    )
    run_store.write_task_result(
        lease,
        task_id="design_predictive_selection",
        status="passed",
        payload={"checkpoint_fingerprint": selection.fingerprint},
    )
    return selection


def _rehash_serialized(value: dict[str, object]) -> dict[str, object]:
    updated = dict(value)
    identity = dict(updated)
    identity.pop("binding_fingerprint")
    updated["binding_fingerprint"] = _sha(identity)
    return updated


def test_v2_slot_registry_is_exact_and_streamless_consumers_bind_parents() -> None:
    assert tuple(item.task_id for item in SCIENTIFIC_TASK_SLOT_REGISTRY) == (
        G3_GATE_ORDER
    )
    assert tuple(SCIENTIFIC_TASK_SLOT_BY_ID) == G3_GATE_ORDER
    assert len(SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT) == 64
    assert SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION == 2
    assert SCIENTIFIC_TASK_BINDING_SCHEMA_ID.endswith("-v2")
    assert SCIENTIFIC_TASK_SLOT_REGISTRY_SCHEMA_ID.endswith("-v2")
    assert all(
        spec.materialization_slots.count(TASK_SOURCE_MATERIALIZATION_SLOT) == 1
        and spec.materialization_slots[0] == TASK_SOURCE_MATERIALIZATION_SLOT
        for spec in SCIENTIFIC_TASK_SLOT_REGISTRY
    )

    streamless_parents = {
        "design_fragmentation": ("design_block_calibration_parent_materialization"),
        "production_fragmentation": (
            "production_block_calibration_parent_materialization"
        ),
        "four_cell_adequacy_holm": "four_cell_adequacy_parent_materialization",
        "four_cell_dependence_holm": ("four_cell_dependence_parent_materialization"),
        "actual_artifact_bridge": "actual_artifact_history_materialization",
        "artifact_integrity": "final_scientific_artifact_parent_materialization",
    }
    for task_id, parent_slot in streamless_parents.items():
        spec = scientific_task_slot_spec(task_id)
        assert spec.rng_slots == ()
        assert parent_slot in spec.materialization_slots
    assert all(
        "official_purpose_one_trace_streams" not in spec.rng_slots
        for spec in SCIENTIFIC_TASK_SLOT_REGISTRY
    )
    selection = scientific_task_slot_spec("design_predictive_selection")
    assert "candidate_gate_parent_materialization" in selection.materialization_slots
    assert "rolling_score_bootstrap_materialization" in (
        selection.materialization_slots
    )
    assert selection.rng_slots == ("rolling_score_bootstrap_streams",)
    for role in ("design", "production"):
        block = scientific_task_slot_spec(f"{role}_block_calibration")
        assert f"{role}_kernel_calibration_materialization" in (
            block.materialization_slots
        )
        assert f"{role}_transition_bootstrap_materialization" in (
            block.materialization_slots
        )
        assert block.rng_slots == (
            f"{role}_block_calibration_streams",
            f"{role}_kernel_calibration_streams",
            f"{role}_transition_bootstrap_streams",
        )
    assert (
        "design_block_calibration_parent_component"
        in scientific_task_slot_spec("design_fragmentation").scientific_component_slots
    )
    assert (
        "production_refitted_parent_component"
        in scientific_task_slot_spec(
            "four_cell_adequacy_holm"
        ).scientific_component_slots
    )


def test_binding_round_trip_is_sealed_json_safe_and_disk_change_stable(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    first = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
    )
    later = _exact_preflight(22_000_000_000)
    second = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=later,
    )

    assert first == second
    assert first.schema_id == SCIENTIFIC_TASK_BINDING_SCHEMA_ID
    assert first.launch_preflight_fingerprint == launch.fingerprint
    assert first.run_identity.source_code_fingerprint == (live.source_code_fingerprint)
    assert first.run_identity.dependency_lock_fingerprint == (
        live.dependency_lock_fingerprint
    )
    assert first.dependency_checkpoint_fingerprints == ()
    assert first.owned_planned_look_fingerprints == ()
    assert first.generation_authorized is False
    json.loads(json.dumps(first.as_dict()))

    verify_scientific_task_binding(
        first,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=later,
    )
    loaded = load_scientific_task_binding(
        first.as_dict(),
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=later,
    )
    assert loaded == first
    verify_scientific_task_binding(
        loaded,
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=later,
    )


def test_builder_derives_dependency_checkpoint_and_frozen_selected_k(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=9,
    ) as lease:
        selection = _publish_prefix_through_selection(
            run_store,
            checkpoint_store,
            lease,
        )
        binding = _build(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="design_macro_stability",
            live_preflight=live,
        )
        selection_binding = _build(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="design_predictive_selection",
            live_preflight=live,
        )

    assert binding.selected_n_states == 3
    assert selection_binding.selected_n_states is None
    assert binding.dependency_checkpoint_fingerprints == (
        ("design_predictive_selection", selection.fingerprint),
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("selected_n_states", 4),
        (
            "dependency_checkpoint_fingerprints",
            {"design_predictive_selection": "a" * 64},
        ),
    ],
)
def test_loader_rejects_wrong_k_promotion_and_dependency_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=9,
    ) as lease:
        _publish_prefix_through_selection(run_store, checkpoint_store, lease)
        binding = _build(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="design_macro_stability",
            live_preflight=live,
        )
        serialized = binding.as_dict()
        serialized[field] = replacement
        forged = _rehash_serialized(serialized)
        with pytest.raises(IntegrityError, match="stale or forged"):
            load_scientific_task_binding(
                forged,
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                live_preflight=live,
            )


def test_owned_look_chain_is_exact_and_old_binding_becomes_stale(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=9,
    ) as lease:
        _publish_prefix_through_selection(run_store, checkpoint_store, lease)
        before = _build(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="design_block_calibration",
            live_preflight=live,
        )
        assert before.owned_planned_look_fingerprints == (
            ("kernel_design_monthly", ()),
            ("kernel_design_daily", ()),
        )
        receipt = run_store.write_planned_look_receipt(
            lease,
            family_id="kernel_design_monthly",
            look_index=1,
            sample_count=10_000,
            decision="continue",
            observed={"maximum_half_width": 0.01},
        )
        after = _build(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="design_block_calibration",
            live_preflight=live,
        )
        assert after.owned_planned_look_fingerprints == (
            ("kernel_design_monthly", (receipt.fingerprint,)),
            ("kernel_design_daily", ()),
        )
        assert after.fingerprint != before.fingerprint
        with pytest.raises(IntegrityError, match="durable state"):
            verify_scientific_task_binding(
                before,
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                live_preflight=live,
            )

        forged = after.as_dict()
        forged_looks = dict(forged["owned_planned_look_fingerprints"])
        forged_looks["kernel_design_monthly"] = ["f" * 64]
        forged["owned_planned_look_fingerprints"] = forged_looks
        with pytest.raises(IntegrityError, match="stale or forged"):
            load_scientific_task_binding(
                _rehash_serialized(forged),
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                live_preflight=live,
            )


@pytest.mark.parametrize(
    "identity_field",
    ["source_code_fingerprint", "dependency_lock_fingerprint"],
)
def test_loader_rejects_stale_source_or_lock_identity(
    tmp_path: Path,
    identity_field: str,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    binding = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
    )
    serialized = binding.as_dict()
    run_identity = dict(serialized["run_identity"])
    run_identity[identity_field] = "f" * 64
    serialized["run_identity"] = run_identity

    with pytest.raises(IntegrityError, match="stale or forged"):
        load_scientific_task_binding(
            _rehash_serialized(serialized),
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )


def test_task_slot_omissions_extras_and_invalid_fingerprints_fail_closed(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    components, materializations, _ = _slots("input_integrity")
    with pytest.raises(ModelError, match="component slots"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints={},
            materialization_fingerprints=materializations,
            rng_fingerprints={},
        )
    with pytest.raises(ModelError, match="materialization slots"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints=components,
            materialization_fingerprints={**materializations, "invented": "a" * 64},
            rng_fingerprints={},
        )
    missing_source = dict(materializations)
    del missing_source[TASK_SOURCE_MATERIALIZATION_SLOT]
    with pytest.raises(ModelError, match="materialization slots"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints=components,
            materialization_fingerprints=missing_source,
            rng_fingerprints={},
        )
    with pytest.raises(ModelError, match="RNG slots"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints=components,
            materialization_fingerprints=materializations,
            rng_fingerprints={"invented": "a" * 64},
        )
    bad_components = dict(components)
    bad_components["input_validation_component"] = "not-a-fingerprint"
    with pytest.raises(ModelError, match="component fingerprint"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints=bad_components,
            materialization_fingerprints=materializations,
            rng_fingerprints={},
        )


def test_unsealed_direct_construction_and_serialized_field_changes_are_rejected(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    binding = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
    )
    unsealed = replace(binding)
    with pytest.raises(ModelError, match="builder authority"):
        verify_scientific_task_binding(
            unsealed,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )

    missing = binding.as_dict()
    missing.pop("rng_fingerprints")
    with pytest.raises(IntegrityError, match="fields are invalid"):
        load_scientific_task_binding(
            missing,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )
    extra = binding.as_dict()
    extra["authority_seal"] = "forged"
    with pytest.raises(IntegrityError, match="fields are invalid"):
        load_scientific_task_binding(
            extra,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )


def test_public_builder_verifier_and_loader_reject_wrong_object_types(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    components, materializations, rng = _slots("input_integrity")
    with pytest.raises(ModelError, match="wrong type"):
        verify_scientific_task_binding(
            object(),  # type: ignore[arg-type]
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )
    with pytest.raises(ModelError, match="CalibrationRunStore"):
        build_scientific_task_binding(
            run_store=object(),  # type: ignore[arg-type]
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints=components,
            materialization_fingerprints=materializations,
            rng_fingerprints=rng,
        )
    with pytest.raises(ModelError, match="checkpoint store"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=object(),  # type: ignore[arg-type]
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
            scientific_component_fingerprints=components,
            materialization_fingerprints=materializations,
            rng_fingerprints=rng,
        )
    with pytest.raises(IntegrityError, match="must be a mapping"):
        load_scientific_task_binding(
            object(),  # type: ignore[arg-type]
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "contract is unsupported"),
        ("registry", "registry identities disagree"),
        ("task_type", "task ID is invalid"),
        ("selected", "selected K is invalid"),
        ("fingerprint", "fingerprint field is invalid"),
        ("run_identity_type", "run identity is missing"),
        ("run_identity_fingerprint", "run identity fields are invalid"),
        ("binding_hash", "binding hash is inconsistent"),
    ],
)
def test_serialized_binding_parser_rejects_malformed_contract_fields(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    binding = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
    )
    serialized = binding.as_dict()
    if mutation == "schema":
        serialized["schema_id"] = "unsupported"
    elif mutation == "registry":
        identity = dict(serialized["run_identity"])
        identity["task_registry_fingerprint"] = "a" * 64
        serialized["run_identity"] = identity
    elif mutation == "task_type":
        serialized["task_id"] = 1
    elif mutation == "selected":
        serialized["selected_n_states"] = True
    elif mutation == "fingerprint":
        serialized["run_id"] = "bad"
    elif mutation == "run_identity_type":
        serialized["run_identity"] = 1
    elif mutation == "run_identity_fingerprint":
        identity = dict(serialized["run_identity"])
        identity["source_code_fingerprint"] = "bad"
        serialized["run_identity"] = identity
    elif mutation == "binding_hash":
        serialized["binding_fingerprint"] = "a" * 64
    else:  # pragma: no cover - closed parametrization.
        raise AssertionError("unknown parser mutation")

    with pytest.raises(IntegrityError, match=message):
        load_scientific_task_binding(
            serialized,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )


def test_v1_binding_has_no_migration_path_before_any_canonical_g3_run(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    serialized = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
    ).as_dict()
    serialized["schema_id"] = "prpg-g3-scientific-task-binding-v1"
    serialized["schema_version"] = 1
    with pytest.raises(IntegrityError, match="contract is unsupported"):
        load_scientific_task_binding(
            _rehash_serialized(serialized),
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("dependency_checkpoint_fingerprints", {}, "checkpoint mapping"),
        (
            "dependency_checkpoint_fingerprints",
            {"design_predictive_selection": "bad"},
            "checkpoint fingerprint",
        ),
        ("owned_planned_look_fingerprints", {}, "planned-look mapping"),
        (
            "owned_planned_look_fingerprints",
            {
                "kernel_design_monthly": "not-a-sequence",
                "kernel_design_daily": [],
            },
            "planned-look chain",
        ),
    ],
)
def test_serialized_dependency_and_look_mapping_shapes_fail_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=9,
    ) as lease:
        _publish_prefix_through_selection(run_store, checkpoint_store, lease)
        task_id = (
            "design_block_calibration"
            if field == "owned_planned_look_fingerprints"
            else "design_macro_stability"
        )
        binding = _build(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id=task_id,
            live_preflight=live,
        )
        serialized = binding.as_dict()
        serialized[field] = replacement
        with pytest.raises(IntegrityError, match=message):
            load_scientific_task_binding(
                serialized,
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                live_preflight=live,
            )


def test_missing_dependency_and_fresh_source_change_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store, checkpoint_store, run_id, _, live = _stores(tmp_path)
    components, materializations, rng = _slots("design_macro_stability")
    with pytest.raises(IntegrityError, match="result is missing"):
        build_scientific_task_binding(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="design_macro_stability",
            live_preflight=live,
            scientific_component_fingerprints=components,
            materialization_fingerprints=materializations,
            rng_fingerprints=rng,
        )

    binding = _build(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
    )
    observed = inspect_g3_execution_closure()
    changed = replace(observed, source_code_fingerprint="f" * 64)
    monkeypatch.setattr(
        preflight_module,
        "inspect_g3_execution_closure",
        lambda: changed,
    )
    with pytest.raises(IntegrityError, match="contract fields"):
        verify_scientific_task_binding(
            binding,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
        )
