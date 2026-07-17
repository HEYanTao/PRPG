from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import prpg.model.g3_task_publication as publication_module
from prpg.errors import IntegrityError, ModelError
from prpg.model import g3_preflight as preflight_module
from prpg.model.g3_boundary_views import (
    InputIntegrityDecisionEvidence,
    build_input_integrity_task_view,
)
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
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
    ScientificTaskBinding,
    build_scientific_task_binding,
)
from prpg.model.g3_task_publication import (
    SCIENTIFIC_TASK_PREBINDING_API_NAMES,
    SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_REASONS,
    input_integrity_executor_source_materialization_fingerprint,
    input_integrity_executor_source_slots,
    publish_exact_next_scientific_task,
    reverify_published_scientific_task,
    scientific_task_source_slots,
)
from prpg.model.ppw_reference import REFERENCE_SHA256, UPSTREAM_SHA256
from prpg.model.run_store import CalibrationRunStore, default_run_identity
from prpg.model.scientific_artifact import ScientificArtifactBinding
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


def _input_binding(
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    run_id: str,
    live: CanonicalG3Preflight,
    source: InputIntegrityDecisionEvidence,
    *,
    materialization_fingerprint: str | None = None,
    source_materialization_fingerprint: str | None = None,
) -> ScientificTaskBinding:
    slots = input_integrity_executor_source_slots(source, run_id=run_id)
    materializations = dict(slots.materialization_fingerprints)
    if source_materialization_fingerprint is not None:
        materializations["task_source_materialization"] = (
            source_materialization_fingerprint
        )
    if materialization_fingerprint is not None:
        materializations["calibration_input_materialization"] = (
            materialization_fingerprint
        )
    return build_scientific_task_binding(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id="input_integrity",
        live_preflight=live,
        scientific_component_fingerprints=dict(slots.scientific_component_fingerprints),
        materialization_fingerprints=materializations,
        rng_fingerprints=dict(slots.rng_fingerprints),
    )


def _input_evidence(
    launch: CanonicalG3Preflight,
    live: CanonicalG3Preflight,
    run_id: str,
    *,
    passing: bool,
) -> InputIntegrityDecisionEvidence:
    artifact_binding = ScientificArtifactBinding(
        config_fingerprint=launch.config_fingerprint,
        processed_data_fingerprint=launch.processed_data_fingerprint,
        calibration_input_fingerprint=launch.calibration_input_fingerprint,
        calibration_run_fingerprint=run_id,
        preflight_fingerprint=launch.fingerprint,
        task_registry_fingerprint=(
            launch.task_registry_fingerprint if passing else "0" * 64
        ),
    )
    return InputIntegrityDecisionEvidence(launch, live, artifact_binding)


def _input_source(
    evidence: InputIntegrityDecisionEvidence,
    run_id: str,
    binding: ScientificTaskBinding,
):
    return build_input_integrity_task_view(
        evidence,
        run_id=run_id,
        binding_fingerprint=binding.fingerprint,
    )


def test_prebinding_input_source_materialization_excludes_run_coordinate() -> None:
    launch = _exact_preflight(20_000_000_000)
    live = _exact_preflight(21_000_000_000)
    first_run = "a" * 64
    second_run = "b" * 64

    def evidence(run_id: str) -> InputIntegrityDecisionEvidence:
        return InputIntegrityDecisionEvidence(
            launch,
            live,
            ScientificArtifactBinding(
                config_fingerprint=launch.config_fingerprint,
                processed_data_fingerprint=launch.processed_data_fingerprint,
                calibration_input_fingerprint=launch.calibration_input_fingerprint,
                calibration_run_fingerprint=run_id,
                preflight_fingerprint=launch.fingerprint,
                task_registry_fingerprint=launch.task_registry_fingerprint,
            ),
        )

    assert input_integrity_executor_source_materialization_fingerprint(
        evidence(first_run),
        run_id=first_run,
    ) == input_integrity_executor_source_materialization_fingerprint(
        evidence(second_run),
        run_id=second_run,
    )


def test_unsupported_register_is_empty_after_all_view_families_integrate() -> None:
    assert tuple(SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_REASONS) == ()

    class UnsupportedSource:
        task_id = "not_registered"

    with pytest.raises(ModelError, match="code-owned binding-slot derivation") as error:
        scientific_task_source_slots(UnsupportedSource())
    assert error.value.details["reason"] == (
        "the source failed every supported object-specific guard"
    )


def test_every_supported_task_has_an_explicit_cycle_free_prebinding_api() -> None:
    supported = tuple(
        task
        for task in G3_GATE_ORDER
        if task not in SCIENTIFIC_TASK_PUBLICATION_UNSUPPORTED_REASONS
    )
    assert tuple(SCIENTIFIC_TASK_PREBINDING_API_NAMES) == supported
    assert all(
        callable(getattr(publication_module, api_name, None))
        for api_name in SCIENTIFIC_TASK_PREBINDING_API_NAMES.values()
    )


def test_first_task_pass_publishes_binding_and_replays_against_live_preflight(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)

    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        published = publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
        replayed = reverify_published_scientific_task(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=_exact_preflight(22_000_000_000),
        )

    assert published.decision.passed
    assert not published.generation_authorized
    assert published.checkpoint is not None
    assert published.task_result.status == "passed"
    assert replayed.binding == binding
    assert not replayed.generation_authorized
    assert replayed.checkpoint is not None
    assert replayed.checkpoint.fingerprint == published.checkpoint.fingerprint
    assert replayed.serialized_binding_sha256 == published.serialized_binding_sha256
    assert published.task_result.payload["scientific_task_binding"] == binding.as_dict()
    derived = scientific_task_source_slots(source)
    assert dict(derived.materialization_fingerprints)[
        "task_source_materialization"
    ] == input_integrity_executor_source_materialization_fingerprint(
        evidence,
        run_id=run_id,
    )


def test_first_task_failure_is_binding_aware_and_replays_without_checkpoint(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=False)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)

    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        published = publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
        replayed = reverify_published_scientific_task(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
        )

    assert not published.decision.passed
    assert published.checkpoint is None
    assert published.task_result.status == "failed"
    assert replayed.checkpoint is None
    assert replayed.binding == binding


def test_first_task_rejects_binding_slots_not_derived_from_sealed_source(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(
        run_store,
        checkpoint_store,
        run_id,
        live,
        evidence,
        materialization_fingerprint="f" * 64,
    )
    source = _input_source(evidence, run_id, binding)

    with (
        run_store.start_launch(
            run_id,
            preflight_fingerprint=launch.fingerprint,
            workers=launch.workers,
        ) as lease,
        pytest.raises(IntegrityError, match="slots differ"),
    ):
        publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
    assert not run_store.verify_task_results(run_id, require_complete=False)


def test_first_task_rejects_forged_universal_source_materialization(
    tmp_path: Path,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(
        run_store,
        checkpoint_store,
        run_id,
        live,
        evidence,
        source_materialization_fingerprint="e" * 64,
    )
    source = _input_source(evidence, run_id, binding)

    with (
        run_store.start_launch(
            run_id,
            preflight_fingerprint=launch.fingerprint,
            workers=launch.workers,
        ) as lease,
        pytest.raises(IntegrityError, match="slots differ"),
    ):
        publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )


def test_replay_rejects_missing_binding_fields_and_malformed_checkpoint_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        published = publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )

        original_result = run_store.read_task_result

        def missing_binding(read_run_id: str, task_id: str):
            result = original_result(read_run_id, task_id)
            payload = dict(result.payload)
            del payload["scientific_task_binding"]
            return replace(result, payload=payload)

        monkeypatch.setattr(run_store, "read_task_result", missing_binding)
        with pytest.raises(IntegrityError, match="binding-aware publication fields"):
            reverify_published_scientific_task(
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
                task_id="input_integrity",
                live_preflight=live,
            )
        monkeypatch.undo()

        assert published.checkpoint is not None
        original_checkpoint = checkpoint_store.load_from_task_result

        def malformed_decision(*args: object, **kwargs: object):
            checkpoint = original_checkpoint(*args, **kwargs)
            state = dict(checkpoint.state)
            state["decision"] = {"gate_id": "input_integrity"}
            return replace(checkpoint, state=state)

        monkeypatch.setattr(
            checkpoint_store,
            "load_from_task_result",
            malformed_decision,
        )
        with pytest.raises(IntegrityError, match="decision fields are invalid"):
            reverify_published_scientific_task(
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
                task_id="input_integrity",
                live_preflight=live,
            )


def test_retry_reuses_exact_orphan_checkpoint_after_result_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)
    original_write = run_store.write_task_result

    def crash_before_result(*args: object, **kwargs: object):
        raise RuntimeError("synthetic crash after checkpoint create")

    monkeypatch.setattr(run_store, "write_task_result", crash_before_result)
    with (
        run_store.start_launch(
            run_id,
            preflight_fingerprint=launch.fingerprint,
            workers=launch.workers,
        ) as lease,
        pytest.raises(RuntimeError, match="synthetic crash"),
    ):
        publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
    assert not run_store.verify_task_results(run_id, require_complete=False)

    monkeypatch.setattr(run_store, "write_task_result", original_write)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
        resume=True,
    ) as lease:
        recovered = publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )

    assert recovered.checkpoint is not None
    assert recovered.checkpoint.reused
    assert recovered.task_result.status == "passed"


@pytest.mark.parametrize(
    ("task_id", "binding_k", "passed", "reported", "expected"),
    (
        ("design_predictive_selection", None, True, 3, 3),
        ("design_predictive_selection", None, False, None, None),
        ("input_integrity", None, True, None, None),
        ("sealed_holdout_veto", 2, True, None, 2),
        ("sealed_holdout_veto", 4, True, 4, 4),
    ),
)
def test_selected_state_count_follows_the_exact_selection_boundary(
    task_id: str,
    binding_k: int | None,
    passed: bool,
    reported: int | None,
    expected: int | None,
) -> None:
    binding = SimpleNamespace(task_id=task_id, selected_n_states=binding_k)
    decision = SimpleNamespace(
        passed=passed,
        summary={} if reported is None else {"selected_k": reported},
    )

    assert publication_module._selected_n_states(binding, decision) == expected


@pytest.mark.parametrize(
    ("task_id", "binding_k", "passed", "reported", "message"),
    (
        (
            "design_predictive_selection",
            2,
            False,
            None,
            "prematurely freezes K",
        ),
        (
            "design_predictive_selection",
            None,
            True,
            True,
            "lacks a valid selected K",
        ),
        (
            "design_predictive_selection",
            None,
            True,
            6,
            "lacks a valid selected K",
        ),
        ("input_integrity", 2, True, None, "preselection task"),
        ("input_integrity", None, True, 2, "preselection task"),
        ("sealed_holdout_veto", None, True, None, "lacks frozen selected K"),
        ("sealed_holdout_veto", 2, True, 3, "differs from the frozen selected K"),
    ),
)
def test_selected_state_count_rejects_early_late_and_inconsistent_k(
    task_id: str,
    binding_k: int | None,
    passed: bool,
    reported: object,
    message: str,
) -> None:
    binding = SimpleNamespace(task_id=task_id, selected_n_states=binding_k)
    decision = SimpleNamespace(passed=passed, summary={"selected_k": reported})

    with pytest.raises((ModelError, IntegrityError), match=message):
        publication_module._selected_n_states(binding, decision)


def test_planned_look_receipts_are_terminal_owned_and_source_exact() -> None:
    design_monthly = SimpleNamespace(
        run_id="a" * 64,
        family_id="kernel_design_monthly",
        look_index=0,
        sample_count=100,
        decision="pass",
        observed={"half_width": 0.001},
        fingerprint="b" * 64,
    )
    design_daily = SimpleNamespace(
        run_id="a" * 64,
        family_id="kernel_design_daily",
        look_index=0,
        sample_count=100,
        decision="fail",
        observed={"half_width": 0.01},
        fingerprint="c" * 64,
    )
    all_receipts = {
        design_monthly.family_id: (design_monthly,),
        design_daily.family_id: (design_daily,),
    }

    owned = publication_module._owned_terminal_receipts(
        task_id="design_block_calibration",
        all_receipts=all_receipts,
    )
    proposed = tuple(
        SimpleNamespace(
            family_id=item.family_id,
            look_index=item.look_index,
            sample_count=item.sample_count,
            decision=item.decision,
            observed=item.observed,
        )
        for item in owned
    )
    publication_module._verify_receipts_match_source(
        "design_block_calibration",
        "a" * 64,
        owned,
        proposed,
    )

    assert owned == (design_monthly, design_daily)
    assert publication_module._look_fingerprint_document(owned) == {
        "kernel_design_monthly": ["b" * 64],
        "kernel_design_daily": ["c" * 64],
    }


def test_planned_look_receipts_reject_future_missing_continuing_and_changed() -> None:
    terminal = SimpleNamespace(
        run_id="a" * 64,
        family_id="kernel_design_monthly",
        look_index=0,
        sample_count=100,
        decision="pass",
        observed={"half_width": 0.001},
        fingerprint="b" * 64,
    )
    daily = SimpleNamespace(
        **{
            **vars(terminal),
            "family_id": "kernel_design_daily",
            "fingerprint": "c" * 64,
        }
    )
    with pytest.raises(IntegrityError, match="future-task planned looks"):
        publication_module._owned_terminal_receipts(
            task_id="input_integrity",
            all_receipts={terminal.family_id: (terminal,)},
        )
    with pytest.raises(IntegrityError, match="not terminal"):
        publication_module._owned_terminal_receipts(
            task_id="design_block_calibration",
            all_receipts={terminal.family_id: (terminal,)},
        )
    continuing = SimpleNamespace(**{**vars(terminal), "decision": "continue"})
    with pytest.raises(IntegrityError, match="not terminal"):
        publication_module._owned_terminal_receipts(
            task_id="design_block_calibration",
            all_receipts={
                continuing.family_id: (continuing,),
                daily.family_id: (daily,),
            },
        )
    with pytest.raises(IntegrityError, match="differ from typed source"):
        publication_module._verify_receipts_match_source(
            "design_block_calibration",
            "a" * 64,
            (terminal,),
            (),
        )
    changed = SimpleNamespace(
        family_id=terminal.family_id,
        look_index=terminal.look_index,
        sample_count=terminal.sample_count + 1,
        decision=terminal.decision,
        observed=terminal.observed,
    )
    with pytest.raises(IntegrityError, match="differs from typed source"):
        publication_module._verify_receipts_match_source(
            "design_block_calibration",
            "a" * 64,
            (terminal,),
            (changed,),
        )


def test_publication_helper_coordinates_slots_and_prefixes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = SimpleNamespace(
        run_id="a" * 64,
        task_id="input_integrity",
        fingerprint="b" * 64,
        scientific_component_fingerprints=(("component", "c" * 64),),
        materialization_fingerprints=(("materialization", "d" * 64),),
        rng_fingerprints=(),
    )
    source = SimpleNamespace(
        run_id=binding.run_id,
        task_id=binding.task_id,
        binding_fingerprint=binding.fingerprint,
    )
    slots = publication_module.ScientificTaskSourceSlots(
        binding.task_id,
        binding.scientific_component_fingerprints,
        binding.materialization_fingerprints,
        binding.rng_fingerprints,
    )
    publication_module._verify_source_coordinates(source, binding)
    monkeypatch.setattr(
        publication_module,
        "scientific_task_source_slots",
        lambda _source: slots,
    )
    assert publication_module._verify_source_binding_slots(source, binding) is slots

    changed_source = SimpleNamespace(
        run_id=binding.run_id,
        task_id=binding.task_id,
        binding_fingerprint="e" * 64,
    )
    with pytest.raises(ModelError, match="matching sealed task-binding coordinates"):
        publication_module._verify_source_coordinates(changed_source, binding)

    wrong_owner = replace(slots, task_id="design_candidate_viability")
    monkeypatch.setattr(
        publication_module,
        "scientific_task_source_slots",
        lambda _source: wrong_owner,
    )
    with pytest.raises(IntegrityError, match="slot owner differs"):
        publication_module._verify_source_binding_slots(source, binding)

    wrong_slots = replace(
        slots,
        materialization_fingerprints=(("materialization", "f" * 64),),
    )
    monkeypatch.setattr(
        publication_module,
        "scientific_task_source_slots",
        lambda _source: wrong_slots,
    )
    with pytest.raises(IntegrityError, match="slots differ"):
        publication_module._verify_source_binding_slots(source, binding)

    exact_store = SimpleNamespace(
        verify_task_results=lambda *_args, **_kwargs: {
            G3_GATE_ORDER[0]: object(),
        }
    )
    assert tuple(
        publication_module._exact_result_prefix(exact_store, binding.run_id)
    ) == (G3_GATE_ORDER[0],)
    reordered_store = SimpleNamespace(
        verify_task_results=lambda *_args, **_kwargs: {
            G3_GATE_ORDER[1]: object(),
        }
    )
    with pytest.raises(IntegrityError, match="exact registry prefix"):
        publication_module._exact_result_prefix(reordered_store, binding.run_id)


def test_publication_json_and_fingerprint_helpers_cover_numpy_and_reject_tamper() -> (
    None
):
    normalized = publication_module._json_value(
        {
            "bool": np.bool_(True),
            "int": np.int64(7),
            "float": np.float64(1.5),
            "zero": -0.0,
            "array": np.asarray([1, 2]),
            "sequence": ("x", None),
        },
        "value",
    )
    assert normalized == {
        "array": [1, 2],
        "bool": True,
        "float": 1.5,
        "int": 7,
        "sequence": ["x", None],
        "zero": 0.0,
    }
    assert publication_module._json_mapping({"value": np.int64(7)}, "mapping") == {
        "value": 7
    }
    assert publication_module._fingerprint({"value": np.int64(7)}) == _sha({"value": 7})
    digest = publication_module._digest_array("ab" * 32)
    assert digest.dtype == np.uint8
    assert digest.shape == (32,)
    assert publication_module._composite_slot_fingerprint(
        "test",
        {"source": "a" * 64},
    ) == _sha(
        {
            "schema_id": "prpg-g3-composite-scientific-slot-v1",
            "label": "test",
            "values": {"source": "a" * 64},
        }
    )

    with pytest.raises(ModelError, match="non-finite float"):
        publication_module._json_value(float("nan"), "value")
    with pytest.raises(ModelError, match="invalid mapping key"):
        publication_module._json_value({"": 1}, "value")
    with pytest.raises(ModelError, match="non-JSON scientific value"):
        publication_module._json_value(object(), "value")
    with pytest.raises(ModelError, match="not SHA-256"):
        publication_module._digest_array("bad")
    with pytest.raises(ModelError, match="invalid slot fingerprint"):
        publication_module._composite_slot_fingerprint(
            "test",
            {"source": "bad"},
        )


def test_prebinding_entrypoints_reject_unsealed_or_cross_role_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _exact_preflight(20_000_000_000)
    live = _exact_preflight(21_000_000_000)
    run_id = "a" * 64
    evidence = _input_evidence(launch, live, run_id, passing=True)

    with pytest.raises(ModelError, match="wrong type"):
        publication_module.input_integrity_executor_source_slots(
            object(),  # type: ignore[arg-type]
            run_id=run_id,
        )
    with pytest.raises(ModelError, match="run ID is not"):
        publication_module.input_integrity_executor_source_slots(
            evidence,
            run_id="bad",
        )
    with pytest.raises(ModelError, match="run/preflight binding"):
        publication_module.input_integrity_executor_source_slots(
            evidence,
            run_id="b" * 64,
        )
    with pytest.raises(ModelError, match="fixed-point authority"):
        publication_module.candidate_viability_executor_source_slots(
            object(),  # type: ignore[arg-type]
            master_seed=1,
            scientific_version=1,
        )
    with pytest.raises(ModelError, match="complete fixed-point authority"):
        publication_module.candidate_selection_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="candidate selection"):
        publication_module.sealed_holdout_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            design=object(),  # type: ignore[arg-type]
            holdout=object(),  # type: ignore[arg-type]
        )
    monkeypatch.setattr(
        publication_module,
        "is_candidate_selection_task_view",
        lambda _value: True,
    )
    with pytest.raises(ModelError, match="SealedHoldoutEvidence"):
        publication_module.sealed_holdout_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            design=object(),  # type: ignore[arg-type]
            holdout=object(),  # type: ignore[arg-type]
        )
    monkeypatch.undo()

    with pytest.raises(ModelError, match="strict_canonical must be boolean"):
        publication_module.adequacy_cell_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            task_id="design_latent_correctness",
            strict_canonical=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="task ID is not registered"):
        publication_module.adequacy_cell_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            task_id="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="strict_canonical must be boolean"):
        publication_module.adequacy_holm_executor_source_slots(
            design_latent=object(),  # type: ignore[arg-type]
            production_latent=object(),  # type: ignore[arg-type]
            design_refitted=object(),  # type: ignore[arg-type]
            production_refitted=object(),  # type: ignore[arg-type]
            strict_canonical=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="guarded source views"):
        publication_module.adequacy_holm_executor_source_slots(
            design_latent=object(),  # type: ignore[arg-type]
            production_latent=object(),  # type: ignore[arg-type]
            design_refitted=object(),  # type: ignore[arg-type]
            production_refitted=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="task ID is not registered"):
        publication_module.fragmentation_executor_source_slots(
            (),
            task_id="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="requires two block parents"):
        publication_module.fragmentation_executor_source_slots(
            (),
            task_id="design_fragmentation",
        )
    with pytest.raises(ModelError, match="registered staged evidence"):
        publication_module.dependence_executor_source_slots(
            object(),  # type: ignore[arg-type]
            task_id="design_dependence",
        )
    with pytest.raises(ModelError, match="exact role views"):
        publication_module.dependence_holm_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            run_id=run_id,
        )
    with pytest.raises(ModelError, match="exact actual-bridge parent"):
        publication_module.artifact_integrity_executor_source_slots(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            run_id=run_id,
        )

    with monkeypatch.context() as context:
        context.setattr(
            publication_module,
            "has_registered_final_fixed_point_candidate_authority",
            lambda _value: False,
        )
        with pytest.raises(ModelError, match="fixed-point authority"):
            publication_module.candidate_viability_executor_source_slots(
                object(),  # type: ignore[arg-type]
                master_seed=1,
                scientific_version=1,
            )
    with monkeypatch.context() as context:
        context.setattr(
            publication_module,
            "has_registered_final_fixed_point_candidate_authority",
            lambda _value: True,
        )
        context.setattr(
            publication_module,
            "verified_registered_candidate_viability_execution_identity",
            lambda *_args, **_kwargs: SimpleNamespace(role="production"),
        )
        with pytest.raises(ModelError, match="requires design evidence"):
            publication_module.candidate_viability_executor_source_slots(
                object(),  # type: ignore[arg-type]
                master_seed=1,
                scientific_version=1,
            )
    with monkeypatch.context() as context:
        context.setattr(
            publication_module,
            "has_registered_final_fixed_point_candidate_authority",
            lambda _value: True,
        )
        context.setattr(
            publication_module,
            "verified_registered_candidate_viability_execution_identity",
            lambda *_args, **_kwargs: SimpleNamespace(
                role="design",
                suite_master_seed=2,
                suite_scientific_version=1,
            ),
        )
        with pytest.raises(ModelError, match="seed/version differ"):
            publication_module.candidate_viability_executor_source_slots(
                object(),  # type: ignore[arg-type]
                master_seed=1,
                scientific_version=1,
            )
    with monkeypatch.context() as context:
        context.setattr(
            publication_module,
            "is_registered_candidate_viability_task_view",
            lambda _value: False,
        )
        context.setattr(
            publication_module,
            "has_registered_final_fixed_point_candidate_authority",
            lambda _value: False,
        )
        with pytest.raises(ModelError, match="complete fixed-point authority"):
            publication_module.candidate_selection_executor_source_slots(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )
    with monkeypatch.context() as context:
        context.setattr(
            publication_module,
            "is_registered_candidate_viability_task_view",
            lambda _value: True,
        )
        context.setattr(
            publication_module,
            "has_registered_final_fixed_point_candidate_authority",
            lambda _value: True,
        )
        context.setattr(
            publication_module,
            "verified_registered_candidate_gate_execution_identity",
            lambda *_args, **_kwargs: SimpleNamespace(role="production"),
        )
        with pytest.raises(ModelError, match="ancestry differs"):
            publication_module.candidate_selection_executor_source_slots(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )
    with monkeypatch.context() as context:
        context.setattr(
            publication_module.calibration_view_module,
            "_macro_task_role",
            lambda _task_id: "design",
        )
        context.setattr(
            publication_module.calibration_view_module,
            "_macro_parent",
            lambda _role, _parent: (object(), "a" * 64, None),
        )
        with pytest.raises(ModelError, match="source slice has the wrong role"):
            publication_module.macro_stability_executor_source_slots(
                task_id="design_macro_stability",
                evidence=object(),  # type: ignore[arg-type]
                parent=object(),  # type: ignore[arg-type]
                source_slice=SimpleNamespace(role="full"),  # type: ignore[arg-type]
                master_seed=1,
                scientific_version=1,
            )
    with monkeypatch.context() as context:
        context.setattr(
            publication_module.adequacy_gate_module,
            "_parent_sources",
            lambda _role, _parent: (object(), object(), "a" * 64),
        )
        context.setattr(
            publication_module,
            "gaussian_hmm_fit_fingerprint",
            lambda _fit: "a" * 64,
        )
        with pytest.raises(ModelError, match="requires latent evidence"):
            publication_module.adequacy_cell_executor_source_slots(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                task_id="design_latent_correctness",
            )
        context.setattr(
            publication_module,
            "is_registered_refitted_adequacy_batch",
            lambda _value: False,
        )
        with pytest.raises(ModelError, match="requires refit evidence"):
            publication_module.adequacy_cell_executor_source_slots(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                task_id="design_refitted_adequacy",
            )

    expected_tasks = (
        "design_latent_correctness",
        "production_latent_correctness",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )

    def adequacy_source(index: int) -> SimpleNamespace:
        return SimpleNamespace(
            task_id=expected_tasks[index],
            run_id="a" * 64,
            binding_fingerprint=f"{index + 1:064x}",
            strict_canonical=True,
        )

    exact_sources = tuple(adequacy_source(index) for index in range(4))
    with monkeypatch.context() as context:
        context.setattr(
            publication_module,
            "is_adequacy_cell_task_view",
            lambda _value: True,
        )
        with pytest.raises(ModelError, match="wrong task order"):
            publication_module.adequacy_holm_executor_source_slots(
                design_latent=exact_sources[1],  # type: ignore[arg-type]
                production_latent=exact_sources[0],  # type: ignore[arg-type]
                design_refitted=exact_sources[2],  # type: ignore[arg-type]
                production_refitted=exact_sources[3],  # type: ignore[arg-type]
            )
        mixed_run = SimpleNamespace(**vars(exact_sources[3]))
        mixed_run.run_id = "b" * 64
        with pytest.raises(ModelError, match="cannot mix source runs"):
            publication_module.adequacy_holm_executor_source_slots(
                design_latent=exact_sources[0],  # type: ignore[arg-type]
                production_latent=exact_sources[1],  # type: ignore[arg-type]
                design_refitted=exact_sources[2],  # type: ignore[arg-type]
                production_refitted=mixed_run,  # type: ignore[arg-type]
            )
        duplicate_binding = SimpleNamespace(**vars(exact_sources[3]))
        duplicate_binding.binding_fingerprint = exact_sources[0].binding_fingerprint
        with pytest.raises(ModelError, match="four distinct source bindings"):
            publication_module.adequacy_holm_executor_source_slots(
                design_latent=exact_sources[0],  # type: ignore[arg-type]
                production_latent=exact_sources[1],  # type: ignore[arg-type]
                design_refitted=exact_sources[2],  # type: ignore[arg-type]
                production_refitted=duplicate_binding,  # type: ignore[arg-type]
            )
        changed_mode = SimpleNamespace(**vars(exact_sources[3]))
        changed_mode.strict_canonical = False
        with pytest.raises(ModelError, match="strict mode differs"):
            publication_module.adequacy_holm_executor_source_slots(
                design_latent=exact_sources[0],  # type: ignore[arg-type]
                production_latent=exact_sources[1],  # type: ignore[arg-type]
                design_refitted=exact_sources[2],  # type: ignore[arg-type]
                production_refitted=changed_mode,  # type: ignore[arg-type]
            )


def test_slot_identity_helpers_reject_unknown_circular_and_malformed_values() -> None:
    valid = "a" * 64
    assert publication_module.scientific_component_slot_fingerprint(
        "input_validation_component"
    )
    with pytest.raises(ModelError, match="slot is not registered"):
        publication_module.scientific_component_slot_fingerprint("unknown")
    with pytest.raises(AssertionError, match="circular coordinates"):
        publication_module._task_source_materialization_fingerprint(
            "input_integrity",
            {"run_id": valid},
        )
    with pytest.raises(ModelError, match="registered candidate identity lacks"):
        publication_module._candidate_identity_field(
            SimpleNamespace(),
            "suite_fingerprint",
        )
    assert publication_module._optional_slot_fingerprint(valid, "slot") == valid
    assert len(publication_module._optional_slot_fingerprint(None, "slot")) == 64
    with pytest.raises(ModelError, match="invalid optional"):
        publication_module._optional_slot_fingerprint("bad", "slot")
    with pytest.raises(ModelError, match="registered candidate view lacks"):
        publication_module._candidate_suite_view_field(
            SimpleNamespace(),
            "suite_fingerprint",
        )
    with pytest.raises(ModelError, match="fingerprint is absent"):
        publication_module._required_block_identity(None, "block")
    with pytest.raises(ModelError, match="monthly and daily"):
        publication_module._ordered_composite_slot_fingerprint(
            "dependence",
            role="design",
            fingerprints=(valid,),
        )
    with pytest.raises(ModelError, match="not a SHA-256 fingerprint"):
        publication_module._bridge_fingerprint_field(
            {"materialization": "bad"},
            "materialization",
        )


def test_bridge_planned_look_projection_rejects_task_prefix_and_terminal_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = object()
    common = "a" * 64
    parents = (("bridge_resource_preflight", "b" * 64),)
    task_identity = SimpleNamespace(
        task_id="bridge_tier_b_model_dgp",
        common_binding_fingerprint=common,
        parent_task_fingerprints=parents,
        source_materialization_fingerprint="c" * 64,
        rng_execution_fingerprint="d" * 64,
        result_fingerprint="e" * 64,
        evidence_fingerprint="f" * 64,
    )
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_task_evidence_identity",
        lambda _source: task_identity,
    )
    with pytest.raises(ModelError, match="wrong task"):
        publication_module.bridge_executor_planned_looks(
            source,  # type: ignore[arg-type]
            task_id="bridge_tier_b_nonparametric_dgp",
        )

    non_tier_identity = SimpleNamespace(**vars(task_identity))
    non_tier_identity.task_id = "bridge_resource_preflight"
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_task_evidence_identity",
        lambda _source: non_tier_identity,
    )
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_task_planned_looks",
        lambda _source: (object(),),
    )
    with pytest.raises(ModelError, match="non-Tier-B"):
        publication_module.bridge_executor_planned_looks(
            source,  # type: ignore[arg-type]
            task_id="bridge_resource_preflight",
        )

    monkeypatch.setattr(
        publication_module,
        "staged_bridge_task_evidence_identity",
        lambda _source: task_identity,
    )
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_task_planned_looks",
        lambda _source: (),
    )
    with pytest.raises(ModelError, match="prefix length is invalid"):
        publication_module.bridge_executor_planned_looks(
            source,  # type: ignore[arg-type]
            task_id="bridge_tier_b_model_dgp",
        )

    local_look = object()
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_task_planned_looks",
        lambda _source: (local_look,),
    )
    valid_look = SimpleNamespace(
        task_id="bridge_tier_b_model_dgp",
        dgp="model",
        look_index=1,
        sample_count=1_000,
        exceedances=50,
        decision="pass_above_tail",
        common_binding_fingerprint=common,
        parent_task_fingerprints=parents,
        source_materialization_fingerprint="c" * 64,
        rng_execution_fingerprint="d" * 64,
        result_fingerprint="e" * 64,
        stage_receipt_fingerprint="1" * 64,
        evidence_fingerprint="2" * 64,
    )
    wrong_lineage = SimpleNamespace(**vars(valid_look))
    wrong_lineage.dgp = "nonparametric"
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_planned_look_identity",
        lambda _look: wrong_lineage,
    )
    with pytest.raises(ModelError, match="lineage is inconsistent"):
        publication_module.bridge_executor_planned_looks(
            source,  # type: ignore[arg-type]
            task_id="bridge_tier_b_model_dgp",
        )

    continuing = SimpleNamespace(**vars(valid_look))
    continuing.decision = "continue"
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_planned_look_identity",
        lambda _look: continuing,
    )
    with pytest.raises(ModelError, match="prefix is nonterminal"):
        publication_module.bridge_executor_planned_looks(
            source,  # type: ignore[arg-type]
            task_id="bridge_tier_b_model_dgp",
        )

    wrong_terminal = SimpleNamespace(**vars(valid_look))
    wrong_terminal.result_fingerprint = "3" * 64
    monkeypatch.setattr(
        publication_module,
        "staged_bridge_planned_look_identity",
        lambda _look: wrong_terminal,
    )
    with pytest.raises(ModelError, match="differs from its task result"):
        publication_module.bridge_executor_planned_looks(
            source,  # type: ignore[arg-type]
            task_id="bridge_tier_b_model_dgp",
        )


def test_publication_entry_and_lineage_guards_fail_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)
    slots = scientific_task_source_slots(source)

    with pytest.raises(ModelError, match="requires a task binding"):
        publication_module.verify_recomputed_scientific_task_source(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=object(),  # type: ignore[arg-type]
            source=source,
        )
    with pytest.raises(ModelError, match="requires a CalibrationRunStore"):
        publication_module._require_active_lease(object(), object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="requires a CalibrationRunStore"):
        reverify_published_scientific_task(
            run_store=object(),  # type: ignore[arg-type]
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
        )
    with pytest.raises(ModelError, match="requires a checkpoint store"):
        reverify_published_scientific_task(
            run_store=run_store,
            checkpoint_store=object(),  # type: ignore[arg-type]
            run_id=run_id,
            task_id="input_integrity",
            live_preflight=live,
        )

    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        with pytest.raises(IntegrityError, match="active lease"):
            publication_module._require_active_lease(run_store, object())  # type: ignore[arg-type]

        marker = run_store.read_launch(run_id)
        with monkeypatch.context() as context:
            context.setattr(
                run_store,
                "read_launch",
                lambda _run_id: replace(marker, workers=marker.workers + 1),
            )
            with pytest.raises(IntegrityError, match="lease identity changed"):
                publication_module._require_active_lease(run_store, lease)

        with pytest.raises(ModelError, match="requires a checkpoint store"):
            publish_exact_next_scientific_task(
                run_store=run_store,
                lease=lease,
                checkpoint_store=object(),  # type: ignore[arg-type]
                live_preflight=live,
                binding=binding,
                source=source,
            )
        with pytest.raises(ModelError, match="requires a task binding"):
            publish_exact_next_scientific_task(
                run_store=run_store,
                lease=lease,
                checkpoint_store=checkpoint_store,
                live_preflight=live,
                binding=object(),  # type: ignore[arg-type]
                source=source,
            )

        with monkeypatch.context() as context:
            context.setattr(
                publication_module,
                "_exact_result_prefix",
                lambda *_args, **_kwargs: {
                    task_id: object() for task_id in G3_GATE_ORDER
                },
            )
            with pytest.raises(IntegrityError, match="already published"):
                publish_exact_next_scientific_task(
                    run_store=run_store,
                    lease=lease,
                    checkpoint_store=checkpoint_store,
                    live_preflight=live,
                    binding=binding,
                    source=source,
                )

        with pytest.raises(IntegrityError, match="exact next canonical task"):
            publish_exact_next_scientific_task(
                run_store=run_store,
                lease=lease,
                checkpoint_store=checkpoint_store,
                live_preflight=live,
                binding=replace(binding, task_id=G3_GATE_ORDER[1]),
                source=source,
            )

        with monkeypatch.context() as context:
            context.setattr(
                publication_module,
                "verify_recomputed_scientific_task_source",
                lambda **_kwargs: slots,
            )
            with pytest.raises(IntegrityError, match="dependency lineage changed"):
                publish_exact_next_scientific_task(
                    run_store=run_store,
                    lease=lease,
                    checkpoint_store=checkpoint_store,
                    live_preflight=live,
                    binding=replace(
                        binding,
                        dependency_checkpoint_fingerprints=(("ghost", "a" * 64),),
                    ),
                    source=source,
                )
            with pytest.raises(IntegrityError, match="planned-look lineage changed"):
                publish_exact_next_scientific_task(
                    run_store=run_store,
                    lease=lease,
                    checkpoint_store=checkpoint_store,
                    live_preflight=live,
                    binding=replace(
                        binding,
                        owned_planned_look_fingerprints=(
                            ("ghost_family", ("a" * 64,)),
                        ),
                    ),
                    source=source,
                )

        failed_look = SimpleNamespace(
            family_id="synthetic",
            decision="fail",
            fingerprint="a" * 64,
        )
        with monkeypatch.context() as context:
            context.setattr(
                publication_module,
                "_owned_terminal_receipts",
                lambda **_kwargs: (failed_look,),
            )
            context.setattr(
                publication_module,
                "_verify_receipts_match_source",
                lambda *_args, **_kwargs: None,
            )
            context.setattr(
                publication_module,
                "_look_fingerprint_document",
                lambda _receipts: {},
            )
            with pytest.raises(ModelError, match="requires passing terminal"):
                publish_exact_next_scientific_task(
                    run_store=run_store,
                    lease=lease,
                    checkpoint_store=checkpoint_store,
                    live_preflight=live,
                    binding=binding,
                    source=source,
                )

    assert not run_store.verify_task_results(run_id, require_complete=False)


def test_replay_rejects_result_and_checkpoint_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=True)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)

    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        published = publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
        result = published.task_result
        checkpoint = published.checkpoint
        assert checkpoint is not None

        def replay() -> None:
            reverify_published_scientific_task(
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
                task_id="input_integrity",
                live_preflight=live,
            )

        def reject_payload(
            payload: dict[str, object],
            message: str,
            *,
            status: str | None = None,
        ) -> None:
            forged = replace(
                result,
                payload=payload,
                status=result.status if status is None else status,
            )
            with monkeypatch.context() as context:
                context.setattr(
                    run_store,
                    "read_task_result",
                    lambda *_args, **_kwargs: forged,
                )
                with pytest.raises(IntegrityError, match=message):
                    replay()

        reject_payload(dict(result.payload), "binding-aware", status="running")
        missing_binding = dict(result.payload)
        missing_binding["scientific_task_binding"] = []
        reject_payload(missing_binding, "binding is missing")
        wrong_mode = dict(result.payload)
        wrong_mode["execution_mode"] = "rehearsal"
        reject_payload(wrong_mode, "binding/result identity differs")
        wrong_looks = dict(result.payload)
        wrong_looks["planned_look_fingerprints"] = {"unexpected": []}
        reject_payload(wrong_looks, "planned-look identity differs")
        wrong_decision = dict(result.payload)
        wrong_decision["decision_fingerprint"] = "bad"
        reject_payload(wrong_decision, "decision identity is invalid")

        def reject_checkpoint(
            forged_checkpoint: object,
            message: str,
        ) -> None:
            with monkeypatch.context() as context:
                context.setattr(
                    checkpoint_store,
                    "load_from_task_result",
                    lambda *_args, **_kwargs: forged_checkpoint,
                )
                with pytest.raises(IntegrityError, match=message):
                    replay()

        missing_state = dict(checkpoint.state)
        del missing_state["publication_schema_version"]
        reject_checkpoint(
            replace(checkpoint, state=missing_state),
            "lacks binding-aware publication fields",
        )
        wrong_schema = dict(checkpoint.state)
        wrong_schema["publication_schema_version"] = 99
        reject_checkpoint(
            replace(checkpoint, state=wrong_schema),
            "schema is unsupported",
        )
        invalid_decision = dict(checkpoint.state)
        stored_decision = dict(invalid_decision["decision"])
        stored_decision["passed"] = False
        invalid_decision["decision"] = stored_decision
        reject_checkpoint(
            replace(checkpoint, state=invalid_decision),
            "checkpoint decision is invalid",
        )
        changed_state = dict(checkpoint.state)
        changed_state["binding_fingerprint"] = "e" * 64
        reject_checkpoint(
            replace(checkpoint, state=changed_state),
            "result/checkpoint identity differs",
        )
        missing_array = dict(checkpoint.arrays)
        del missing_array["decision_sha256"]
        reject_checkpoint(
            replace(checkpoint, arrays=missing_array),
            "digest arrays are invalid",
        )
        changed_arrays = dict(checkpoint.arrays)
        changed_arrays["decision_sha256"] = np.zeros(32, dtype=np.uint8)
        reject_checkpoint(
            replace(checkpoint, arrays=changed_arrays),
            "digest arrays disagree",
        )


def test_failed_replay_rejects_checkpoint_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=False)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)
    with run_store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        published = publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
        payload = dict(published.task_result.payload)
        payload["checkpoint_fingerprint"] = "a" * 64
        forged = replace(published.task_result, payload=payload)
        monkeypatch.setattr(
            run_store,
            "read_task_result",
            lambda *_args, **_kwargs: forged,
        )
        with pytest.raises(IntegrityError, match="cannot reference a checkpoint"):
            reverify_published_scientific_task(
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
                task_id="input_integrity",
                live_preflight=live,
            )


@pytest.mark.parametrize("passing", (False, True))
def test_publish_rejects_nonexact_internal_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing: bool,
) -> None:
    run_store, checkpoint_store, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=passing)
    binding = _input_binding(run_store, checkpoint_store, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)
    monkeypatch.setattr(
        publication_module,
        "reverify_published_scientific_task",
        lambda **_kwargs: SimpleNamespace(
            binding=SimpleNamespace(fingerprint="e" * 64),
            checkpoint=None,
            task_result=SimpleNamespace(fingerprint="e" * 64),
        ),
    )
    message = "fresh scientific" if passing else "failed scientific"
    with (
        run_store.start_launch(
            run_id,
            preflight_fingerprint=launch.fingerprint,
            workers=launch.workers,
        ) as lease,
        pytest.raises(IntegrityError, match=message),
    ):
        publish_exact_next_scientific_task(
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            live_preflight=live,
            binding=binding,
            source=source,
        )
