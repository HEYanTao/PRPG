from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import prpg.model.g3_evidence_store as evidence_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.g3 import (
    build_g3_evidence_bundle,
    gate_record,
)
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointBinding,
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_evidence_store import (
    G3_EVIDENCE_DIRECTORY,
    G3_EVIDENCE_MANIFEST_NAME,
    G3_EVIDENCE_SCHEMA_VERSION,
    G3EvidenceStore,
    verify_loaded_g3_evidence,
)
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_TASK_BY_ID,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    LaunchMarker,
    default_run_identity,
)
from prpg.model.scientific_artifact import (
    CANONICAL_KERNEL_LOOKS,
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    ScientificModelArtifactDraft,
    ScientificReportPayload,
    publish_scientific_model_artifact,
    required_scientific_documents,
    scientific_report_payload,
    scientific_report_planned_look_families,
    scientific_report_task_ids,
)
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificGenerationPolicy,
)
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3ExecutionClosureIdentity,
)

_CONFIG = "a" * 64
_PROCESSED = "b" * 64
_INPUT = "c" * 64
_SOURCE = "d" * 64
_LOCK = "e" * 64
_PREFLIGHT = "f" * 64
_K = 2


def _generation_policy() -> ScientificGenerationPolicy:
    return ScientificGenerationPolicy(
        schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=4,
        daily_block_length=20,
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
        neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


@dataclass(frozen=True, slots=True)
class _Fixture:
    evidence_store: G3EvidenceStore
    run_store: CalibrationRunStore
    checkpoint_store: CalibrationCheckpointStore
    model_store: ModelArtifactStore
    run_id: str
    completion: LaunchMarker
    checkpoints: Mapping[str, CalibrationCheckpointReference]
    design: LoadedScientificModelArtifact
    production: LoadedScientificModelArtifact
    bundle: Any
    evidence_fingerprint: str
    live_preflight: CanonicalG3Preflight


@dataclass(frozen=True, slots=True)
class _LoadFixture:
    evidence_store: G3EvidenceStore
    run_store: CalibrationRunStore
    checkpoint_store: CalibrationCheckpointStore
    model_store: ModelArtifactStore
    run_id: str
    evidence_fingerprint: str
    live_preflight: CanonicalG3Preflight


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _source_identity(source: str = _SOURCE) -> G3ExecutionClosureIdentity:
    return G3ExecutionClosureIdentity(
        schema_id="prpg-g3-execution-closure-v2",
        manifest=G3_EXECUTION_CLOSURE_MANIFEST,
        manifest_fingerprint=G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
        source_files=(),
        source_code_fingerprint=source,
        dependency_lock_relative_path="requirements.lock",
        dependency_lock_bytes=1,
        dependency_lock_fingerprint=_LOCK,
        dependency_versions=(("numpy", "2.3.5"),),
    )


def _continuation(segments: tuple[int, ...]) -> np.ndarray[Any, Any]:
    result = np.ones(sum(segments), dtype=np.bool_)
    start = 0
    for length in segments:
        result[start] = False
        start += length
    return result


def _arrays(role: str) -> dict[str, np.ndarray[Any, Any]]:
    monthly_segments = (193,) if role == "design" else (210, 7)
    daily_segments = (4_049,) if role == "design" else (4_404, 143)
    monthly_states = np.arange(sum(monthly_segments), dtype=np.int64) % _K
    daily_states = np.arange(sum(daily_segments), dtype=np.int64) % _K
    transition = np.full((_K, _K), 0.1 / (_K - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.9)
    lambdas = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    return {
        "hmm_start_probabilities": np.full(_K, 1.0 / _K),
        "hmm_transition_matrix": transition,
        "hmm_state_means": np.arange(_K * 4, dtype=np.float64).reshape(_K, 4) / 10.0,
        "hmm_tied_covariance": np.eye(4, dtype=np.float64),
        "hmm_scaler_means": np.arange(4, dtype=np.float64),
        "hmm_scaler_standard_deviations": np.ones(4, dtype=np.float64),
        "canonical_to_original": np.arange(_K, dtype=np.int64),
        "original_to_canonical": np.arange(_K, dtype=np.int64),
        "monthly_source_states": monthly_states,
        "daily_source_states": daily_states,
        "monthly_continuation": _continuation(monthly_segments),
        "daily_continuation": _continuation(daily_segments),
        "monthly_state_counts": np.bincount(monthly_states, minlength=_K),
        "daily_state_counts": np.bincount(daily_states, minlength=_K),
        "monthly_pc1_loadings": np.asarray([1.0, 0.0, 0.0]),
        "daily_pc1_loadings": np.asarray([1.0, 0.0, 0.0]),
        "monthly_kernel_lambdas": lambdas,
        "daily_kernel_lambdas": lambdas,
        "monthly_kernel_means": np.zeros((_K, 3), dtype=np.float64),
        "daily_kernel_means": np.zeros((_K, 3), dtype=np.float64),
        "monthly_kernel_half_widths": np.full((_K, 3), 0.0009),
        "daily_kernel_half_widths": np.full((_K, 3), 0.0009),
        "monthly_kernel_state_counts": np.full(_K, 10_000, dtype=np.int64),
        "daily_kernel_state_counts": np.full(_K, 10_000, dtype=np.int64),
        "monthly_kernel_offsets": np.zeros((4, _K, 3), dtype=np.float64),
        "daily_kernel_offsets": np.zeros((4, _K, 3), dtype=np.float64),
    }


def _counts(document_type: str, role: str) -> dict[str, int]:
    rows = 193 if role == "design" else 217
    values = {
        "model_fit": {
            "observations": rows,
            "features": 4,
            "ordinary_restarts": 50,
        },
        "macro_stability": {"attempts": 100, "minimum_successes": 90},
        "latent_correctness": {
            "chains": 10_000,
            "measured_months": 600,
            "extension_months": 2_000,
            "cluster_bootstrap_replicates": 10_000,
            "cluster_resample_size": 10_000,
        },
        "refitted_adequacy": {"attempts": 1_000, "minimum_successes": 950},
        "transition_support": {"attempts": 100},
        "block_calibration": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "fragmentation": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "dependence": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "kernel_calibration": {
            "monthly_windows": 10_000,
            "daily_windows": 20_000,
            "monthly_verification_windows": 10_000,
            "daily_verification_windows": 20_000,
            "planned_looks": len(CANONICAL_KERNEL_LOOKS),
        },
        "adequacy_holm": {"cells": 4},
        "dependence_holm": {"cells": 4},
        "design_production_bridge": {
            "resource_benchmark_pairs": 100,
            "tier_a_pairs_per_dgp": 250,
            "tier_b_maximum_pairs_per_dgp": 10_000,
            "accelerator_audit_pairs_per_dgp": 100,
            "dgp_families": 2,
        },
        "candidate_selection": {
            "candidates": 4,
            "rolling_folds": 6,
            "rolling_scores_per_candidate": 72,
            "rolling_bootstrap_replicates": 10_000,
            "ordinary_restarts_per_fit": 50,
        },
        "sealed_holdout_veto": {"observations": 24, "segments": 2},
        "production_fit_same_k": {
            "observations": 217,
            "ordinary_restarts": 50,
        },
    }
    return values[document_type]


def _reports(
    role: str,
    *,
    task_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    look_fingerprints: Mapping[str, tuple[str, ...]],
) -> dict[str, ScientificReportPayload]:
    result: dict[str, ScientificReportPayload] = {}
    for path, document_type in required_scientific_documents(role).items():
        tasks = scientific_report_task_ids(document_type, role)  # type: ignore[arg-type]
        families = scientific_report_planned_look_families(
            document_type,
            role,  # type: ignore[arg-type]
        )
        result[path] = ScientificReportPayload(
            document_type=document_type,
            counts=_counts(document_type, role),
            payload=scientific_report_payload(
                document_type=document_type,
                role=role,  # type: ignore[arg-type]
                task_result_fingerprints={
                    task_id: task_fingerprints[task_id] for task_id in tasks
                },
                checkpoint_fingerprints={
                    task_id: checkpoint_fingerprints[task_id] for task_id in tasks
                },
                planned_look_fingerprints={
                    family_id: look_fingerprints[family_id] for family_id in families
                },
                metrics={"recorded": True},
            ),
        )
    return result


def _decision(
    task_id: str,
    *,
    binding: CalibrationCheckpointBinding,
    workers: int = 9,
    artifact_evidence: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if task_id == "input_integrity":
        evidence = {
            "preflight_fingerprint": binding.preflight_fingerprint,
            "live_preflight_compatible": True,
            "config_fingerprint": binding.config_fingerprint,
            "processed_data_fingerprint": binding.processed_data_fingerprint,
            "calibration_input_fingerprint": (binding.calibration_input_fingerprint),
            "source_code_fingerprint": _SOURCE,
            "dependency_lock_fingerprint": _LOCK,
            "workers": workers,
            "check_count": 9,
        }
    elif artifact_evidence is not None:
        evidence = dict(artifact_evidence)
    else:
        evidence = {"scientific_passed": True, "task_id": task_id}
    summary = (
        {"selected_k": _K}
        if task_id == "design_predictive_selection"
        else {"status": "passed"}
    )
    document = {
        "evidence": evidence,
        "gate_id": task_id,
        "passed": True,
        "source_type": "tests.CanonicalTypedSource",
        "summary": summary,
    }
    return document, summary


def _write_task(
    *,
    task_id: str,
    run_store: CalibrationRunStore,
    lease: Any,
    checkpoint_store: CalibrationCheckpointStore,
    binding: CalibrationCheckpointBinding,
    checkpoints: dict[str, CalibrationCheckpointReference],
    artifact_evidence: Mapping[str, Any] | None = None,
) -> None:
    task_looks: dict[str, list[str]] = {}
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if spec.owner_task_id != task_id:
            continue
        receipt = run_store.write_planned_look_receipt(
            lease,
            family_id=spec.family_id,
            look_index=1,
            sample_count=spec.sample_counts[0],
            decision="pass",
            observed={"passed": True},
        )
        task_looks[spec.family_id] = [receipt.fingerprint]
    document, summary = _decision(
        task_id,
        binding=binding,
        artifact_evidence=artifact_evidence,
    )
    decision_fingerprint = _fingerprint(document)
    binding_fingerprint = _fingerprint({"task_id": task_id, "kind": "binding"})
    serialized_binding = {
        "binding_fingerprint": binding_fingerprint,
        "legacy_fixture_task_id": task_id,
    }
    serialized_binding_sha256 = _fingerprint(serialized_binding)
    selected_k = (
        None
        if G3_GATE_ORDER.index(task_id)
        < G3_GATE_ORDER.index("design_predictive_selection")
        else _K
    )
    checkpoint = checkpoint_store.create(
        task_id=task_id,
        binding=binding,
        dependencies=tuple(
            checkpoints[dependency]
            for dependency in G3_TASK_BY_ID[task_id].dependencies
        ),
        selected_n_states=selected_k,
        arrays={
            "decision_sha256": np.frombuffer(
                bytes.fromhex(decision_fingerprint), dtype=np.uint8
            ).copy(),
            "planned_looks_sha256": np.frombuffer(
                bytes.fromhex(_fingerprint(task_looks)), dtype=np.uint8
            ).copy(),
            "scientific_task_binding_sha256": np.frombuffer(
                bytes.fromhex(serialized_binding_sha256), dtype=np.uint8
            ).copy(),
        },
        state={
            "binding_fingerprint": binding_fingerprint,
            "checkpoint_state_schema_version": 2,
            "decision": document,
            "decision_fingerprint": decision_fingerprint,
            "planned_look_fingerprints": task_looks,
            "publication_schema_version": 1,
            "scientific_task_binding": serialized_binding,
            "serialized_binding_sha256": serialized_binding_sha256,
        },
    )
    checkpoints[task_id] = checkpoint
    run_store.write_task_result(
        lease,
        task_id=task_id,
        status="passed",
        payload={
            "binding_fingerprint": binding_fingerprint,
            "checkpoint_fingerprint": checkpoint.fingerprint,
            "coordinator_schema_version": 1,
            "decision": document,
            "decision_fingerprint": decision_fingerprint,
            "execution_mode": "canonical",
            "outcome_kind": "typed_decision",
            "planned_look_fingerprints": task_looks,
            "scientific_task_binding": serialized_binding,
            "selected_n_states": selected_k,
            "serialized_binding_sha256": serialized_binding_sha256,
            "summary": summary,
        },
    )


def _build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    import prpg.model.g3_evidence_store as module

    monkeypatch.setattr(module, "inspect_g3_execution_closure", _source_identity)
    live_preflight = object.__new__(CanonicalG3Preflight)

    def legacy_reverify(**kwargs: Any) -> SimpleNamespace:
        result = kwargs["run_store"].read_task_result(
            kwargs["run_id"], kwargs["task_id"]
        )
        checkpoint = kwargs["checkpoint_store"].load_from_task_result(
            kwargs["run_store"],
            run_id=kwargs["run_id"],
            task_id=kwargs["task_id"],
        )
        return SimpleNamespace(
            binding=None,
            checkpoint=checkpoint,
            task_result=result,
        )

    # This legacy fixture exercises the evidence store's other external-state
    # checks. Dedicated task-publication and artifact-pair tests exercise the
    # real live-binding/receipt verifiers; production code never installs
    # these test-only replacements.
    monkeypatch.setattr(module, "reverify_published_scientific_task", legacy_reverify)
    monkeypatch.setattr(
        module,
        "verify_durable_scientific_artifact_pair_receipt",
        lambda **_kwargs: SimpleNamespace(),
    )
    run_store = CalibrationRunStore(tmp_path / "runs")
    run_reference = run_store.initialize(
        default_run_identity(
            config_fingerprint=_CONFIG,
            processed_data_fingerprint=_PROCESSED,
            calibration_input_fingerprint=_INPUT,
            source_code_fingerprint=_SOURCE,
            dependency_lock_fingerprint=_LOCK,
            workers=9,
        )
    )
    run_store.plan_launch(
        run_reference.run_id,
        preflight_fingerprint=_PREFLIGHT,
        workers=9,
    )
    lease = run_store.start_launch(
        run_reference.run_id,
        preflight_fingerprint=_PREFLIGHT,
        workers=9,
    )
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    model_store = ModelArtifactStore(tmp_path / "artifacts")
    binding = CalibrationCheckpointBinding(
        run_id=run_reference.run_id,
        config_fingerprint=_CONFIG,
        processed_data_fingerprint=_PROCESSED,
        calibration_input_fingerprint=_INPUT,
        preflight_fingerprint=_PREFLIGHT,
    )
    checkpoints: dict[str, CalibrationCheckpointReference] = {}
    try:
        for task_id in G3_GATE_ORDER[:-1]:
            _write_task(
                task_id=task_id,
                run_store=run_store,
                lease=lease,
                checkpoint_store=checkpoint_store,
                binding=binding,
                checkpoints=checkpoints,
            )
        partial_results = run_store.verify_task_results(
            run_reference.run_id, require_complete=False
        )
        look_receipts = run_store.verify_planned_look_receipts(
            run_reference.run_id, require_terminal=True, task_results=partial_results
        )
        task_fingerprints = {
            task_id: partial_results[task_id].fingerprint
            for task_id in G3_GATE_ORDER[:-1]
        }
        task_fingerprints["artifact_integrity"] = "1" * 64
        checkpoint_fingerprints = {
            task_id: checkpoints[task_id].fingerprint for task_id in G3_GATE_ORDER[:-1]
        }
        checkpoint_fingerprints["artifact_integrity"] = "2" * 64
        look_fingerprints = {
            family_id: tuple(receipt.fingerprint for receipt in receipts)
            for family_id, receipts in look_receipts.items()
        }
        scientific_binding = ScientificArtifactBinding(
            config_fingerprint=_CONFIG,
            processed_data_fingerprint=_PROCESSED,
            calibration_input_fingerprint=_INPUT,
            calibration_run_fingerprint=run_reference.run_id,
            preflight_fingerprint=_PREFLIGHT,
        )
        provenance = ScientificArtifactProvenance(
            source_code_fingerprint=_SOURCE,
            dependency_lock_fingerprint=_LOCK,
        )
        design = publish_scientific_model_artifact(
            model_store,
            ScientificModelArtifactDraft(
                role="design",
                selected_k=_K,
                binding=scientific_binding,
                provenance=provenance,
                generation_policy=_generation_policy(),
                arrays=_arrays("design"),
                reports=_reports(
                    "design",
                    task_fingerprints=task_fingerprints,
                    checkpoint_fingerprints=checkpoint_fingerprints,
                    look_fingerprints=look_fingerprints,
                ),
            ),
        )
        production = publish_scientific_model_artifact(
            model_store,
            ScientificModelArtifactDraft(
                role="production",
                selected_k=_K,
                binding=scientific_binding,
                provenance=provenance,
                generation_policy=_generation_policy(),
                arrays=_arrays("production"),
                reports=_reports(
                    "production",
                    task_fingerprints=task_fingerprints,
                    checkpoint_fingerprints=checkpoint_fingerprints,
                    look_fingerprints=look_fingerprints,
                ),
            ),
        )
        artifact_evidence = {
            "design_artifact_fingerprint": design.reference.fingerprint,
            "production_artifact_fingerprint": production.reference.fingerprint,
            "design_parameter_set_fingerprint": design.parameter_set_fingerprint,
            "production_parameter_set_fingerprint": (
                production.parameter_set_fingerprint
            ),
            "selected_k": _K,
            "binding": scientific_binding.as_dict(),
            "provenance": provenance.as_dict(),
            "typed_schema_verified": True,
            "pair_receipt_fingerprint": "3" * 64,
            "design_role_source_fingerprint": "4" * 64,
            "production_role_source_fingerprint": "5" * 64,
            "actual_bridge_parent_view_fingerprint": "6" * 64,
        }
        _write_task(
            task_id="artifact_integrity",
            run_store=run_store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            binding=binding,
            checkpoints=checkpoints,
            artifact_evidence=artifact_evidence,
        )
        completion = run_store.complete_launch(lease)
    finally:
        lease.close()

    records = []
    for task_id in G3_GATE_ORDER:
        decision = checkpoints[task_id].state["decision"]
        records.append(
            gate_record(
                task_id,
                passed=True,
                evidence=decision["evidence"],
                summary=decision["summary"],
            )
        )
    bundle = build_g3_evidence_bundle(
        config_fingerprint=_CONFIG,
        processed_data_fingerprint=_PROCESSED,
        calibration_input_fingerprint=_INPUT,
        design_artifact_fingerprint=design.reference.fingerprint,
        production_artifact_fingerprint=production.reference.fingerprint,
        records=records,
    )
    evidence_store = G3EvidenceStore(tmp_path / "artifacts")
    loaded = evidence_store.publish(
        bundle=bundle,
        run_store=run_store,
        run_id=run_reference.run_id,
        completion_marker=completion,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        checkpoints=checkpoints,
        model_store=model_store,
        design_artifact=design,
        production_artifact=production,
    )
    return _Fixture(
        evidence_store,
        run_store,
        checkpoint_store,
        model_store,
        run_reference.run_id,
        completion,
        checkpoints,
        design,
        production,
        bundle,
        loaded.reference.fingerprint,
        live_preflight,
    )


@pytest.fixture(scope="module")
def canonical_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_Fixture]:
    monkeypatch = pytest.MonkeyPatch()
    try:
        yield _build_fixture(
            tmp_path_factory.mktemp("g3-evidence-canonical"), monkeypatch
        )
    finally:
        monkeypatch.undo()


def _copy_fixture(fixture: _Fixture, destination: Path) -> _LoadFixture:
    source = fixture.evidence_store.artifact_root.parent
    shutil.copytree(source, destination)
    return _LoadFixture(
        G3EvidenceStore(destination / "artifacts"),
        CalibrationRunStore(destination / "runs"),
        CalibrationCheckpointStore(destination / "artifacts"),
        ModelArtifactStore(destination / "artifacts"),
        fixture.run_id,
        fixture.evidence_fingerprint,
        fixture.live_preflight,
    )


def _counterfeit_manifest(
    fixture: _Fixture,
    destination: Path,
    mutate: Any,
) -> tuple[G3EvidenceStore, str]:
    source = (
        fixture.evidence_store.store_root
        / fixture.evidence_fingerprint
        / G3_EVIDENCE_MANIFEST_NAME
    )
    value = json.loads(source.read_text())
    mutate(value)
    fingerprint = _fingerprint(value["identity"])
    value["evidence_fingerprint"] = fingerprint
    store = G3EvidenceStore(destination / "artifacts")
    root = store.store_root / fingerprint
    root.mkdir(parents=True)
    (root / G3_EVIDENCE_MANIFEST_NAME).write_bytes(_canonical_bytes(value))
    return store, fingerprint


def _readdress_fixture(fixture: _LoadFixture, mutate: Any) -> str:
    old_root = fixture.evidence_store.store_root / fixture.evidence_fingerprint
    value = json.loads((old_root / G3_EVIDENCE_MANIFEST_NAME).read_text())
    mutate(value)
    fingerprint = _fingerprint(value["identity"])
    value["evidence_fingerprint"] = fingerprint
    root = fixture.evidence_store.store_root / fingerprint
    root.mkdir()
    (root / G3_EVIDENCE_MANIFEST_NAME).write_bytes(_canonical_bytes(value))
    return fingerprint


def test_round_trip_reloads_evidence_only_and_reuses_exact_content(
    canonical_fixture: _Fixture,
) -> None:
    fixture = canonical_fixture
    loaded = fixture.evidence_store.load(
        fixture.evidence_fingerprint,
        run_store=fixture.run_store,
        checkpoint_store=fixture.checkpoint_store,
        live_preflight=fixture.live_preflight,
        model_store=fixture.model_store,
    )
    replay = fixture.evidence_store.publish(
        bundle=fixture.bundle,
        run_store=fixture.run_store,
        run_id=fixture.run_id,
        completion_marker=fixture.completion,
        checkpoint_store=fixture.checkpoint_store,
        live_preflight=fixture.live_preflight,
        checkpoints=fixture.checkpoints,
        model_store=fixture.model_store,
        design_artifact=fixture.design,
        production_artifact=fixture.production,
    )

    assert loaded.generation_authorized is False
    assert loaded.evidence_bundle.generation_authorized is False
    assert loaded.evidence_bundle.passed
    assert loaded.evidence_bundle.fingerprint == fixture.bundle.fingerprint
    assert loaded.run_id == fixture.run_id
    assert loaded.completion_fingerprint == fixture.completion.fingerprint
    assert tuple(loaded.task_result_fingerprints) == G3_GATE_ORDER
    assert tuple(loaded.checkpoint_fingerprints) == G3_GATE_ORDER
    assert replay.reference.reused
    assert replay.reference.fingerprint == fixture.evidence_fingerprint
    assert G3_EVIDENCE_DIRECTORY in loaded.reference.path.parts
    persisted = loaded.reference.manifest["identity"]["evidence_bundle"]["identity"]
    assert persisted["generation_authorized"] is False
    assert loaded.reference.manifest["identity"]["passed"] is True
    assert loaded.reference.manifest["identity"]["generation_authorized"] is False
    assert verify_loaded_g3_evidence(loaded) is loaded
    with pytest.raises(ModelError, match="store-loader provenance"):
        verify_loaded_g3_evidence(copy.copy(loaded))


def test_passed_evidence_can_publish_but_inexact_checkpoint_map_cannot(
    canonical_fixture: _Fixture,
) -> None:
    fixture = canonical_fixture
    report_only = build_g3_evidence_bundle(
        config_fingerprint=_CONFIG,
        processed_data_fingerprint=_PROCESSED,
        calibration_input_fingerprint=_INPUT,
        design_artifact_fingerprint=fixture.design.reference.fingerprint,
        production_artifact_fingerprint=fixture.production.reference.fingerprint,
        records=fixture.bundle.records,
    )
    replay = fixture.evidence_store.publish(
        bundle=report_only,
        run_store=fixture.run_store,
        run_id=fixture.run_id,
        completion_marker=fixture.completion,
        checkpoint_store=fixture.checkpoint_store,
        live_preflight=fixture.live_preflight,
        checkpoints=fixture.checkpoints,
        model_store=fixture.model_store,
        design_artifact=fixture.design,
        production_artifact=fixture.production,
    )
    assert replay.evidence_bundle.passed
    assert replay.generation_authorized is False
    incomplete = dict(fixture.checkpoints)
    del incomplete["artifact_integrity"]
    with pytest.raises(ModelError, match="exact ordered checkpoint"):
        fixture.evidence_store.publish(
            bundle=fixture.bundle,
            run_store=fixture.run_store,
            run_id=fixture.run_id,
            completion_marker=fixture.completion,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            checkpoints=incomplete,
            model_store=fixture.model_store,
            design_artifact=fixture.design,
            production_artifact=fixture.production,
        )


def test_loader_preserves_historical_provenance_after_downstream_source_change(
    canonical_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = canonical_fixture
    import prpg.model.g3_evidence_store as module

    monkeypatch.setattr(
        module,
        "inspect_g3_execution_closure",
        lambda: _source_identity("9" * 64),
    )
    loaded = fixture.evidence_store.load(
        fixture.evidence_fingerprint,
        run_store=fixture.run_store,
        checkpoint_store=fixture.checkpoint_store,
        model_store=fixture.model_store,
    )

    assert loaded.g3_source_code_fingerprint == _SOURCE
    assert loaded.g3_dependency_lock_fingerprint == _LOCK
    assert (
        loaded.reference.manifest["identity"]["provenance"]["source_code_fingerprint"]
        == _SOURCE
    )


def test_publication_still_requires_the_recorded_g3_executable_identity(
    canonical_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = canonical_fixture
    monkeypatch.setattr(
        evidence_module,
        "inspect_g3_execution_closure",
        lambda: _source_identity("9" * 64),
    )
    with pytest.raises(IntegrityError, match="source or dependency lock is stale"):
        fixture.evidence_store.publish(
            bundle=fixture.bundle,
            run_store=fixture.run_store,
            run_id=fixture.run_id,
            completion_marker=fixture.completion,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            checkpoints=fixture.checkpoints,
            model_store=fixture.model_store,
            design_artifact=fixture.design,
            production_artifact=fixture.production,
        )


def test_loader_rejects_readdressed_historical_provenance_tampering(
    tmp_path: Path, canonical_fixture: _Fixture
) -> None:
    fixture = _copy_fixture(canonical_fixture, tmp_path / "provenance-tamper")
    fingerprint = _readdress_fixture(
        fixture,
        lambda value: value["identity"]["provenance"].__setitem__(
            "source_code_fingerprint", "9" * 64
        ),
    )

    with pytest.raises(IntegrityError, match="immutable G3 run"):
        fixture.evidence_store.load(
            fingerprint,
            run_store=fixture.run_store,
            checkpoint_store=fixture.checkpoint_store,
            model_store=fixture.model_store,
        )


def test_loader_rejects_failed_and_missing_task_results(
    tmp_path: Path,
    canonical_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = canonical_fixture
    original = fixture.run_store.verify_task_results

    def failed_results(run_id: str, *, require_complete: bool):
        results = original(run_id, require_complete=require_complete)
        results["design_macro_stability"] = replace(
            results["design_macro_stability"], status="failed"
        )
        return results

    monkeypatch.setattr(fixture.run_store, "verify_task_results", failed_results)
    with pytest.raises(IntegrityError, match="all exact 26 tasks to pass"):
        fixture.evidence_store.load(
            fixture.evidence_fingerprint,
            run_store=fixture.run_store,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            model_store=fixture.model_store,
        )

    missing = _copy_fixture(canonical_fixture, tmp_path / "missing-task")
    task_path = (
        missing.run_store.verify(missing.run_id).path
        / "task-results"
        / "design_macro_stability.json"
    )
    task_path.unlink()
    with pytest.raises(
        IntegrityError,
        match="exact 26 G3 task results are incomplete|precedes a required dependency",
    ):
        missing.evidence_store.load(
            missing.evidence_fingerprint,
            run_store=missing.run_store,
            checkpoint_store=missing.checkpoint_store,
            live_preflight=missing.live_preflight,
            model_store=missing.model_store,
        )


def test_loader_rejects_missing_terminal_look_and_tampered_checkpoint(
    tmp_path: Path, canonical_fixture: _Fixture
) -> None:
    fixture = _copy_fixture(canonical_fixture, tmp_path / "look")
    look_path = (
        fixture.run_store.verify(fixture.run_id).path
        / "planned-looks"
        / "kernel_design_monthly"
        / "01.json"
    )
    look_path.unlink()
    with pytest.raises(
        IntegrityError,
        match="required planned-look receipts are missing|lacks a durable terminal",
    ):
        fixture.evidence_store.load(
            fixture.evidence_fingerprint,
            run_store=fixture.run_store,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            model_store=fixture.model_store,
        )

    other = _copy_fixture(canonical_fixture, tmp_path / "checkpoint")
    checkpoint_manifest = (
        other.checkpoint_store.load(
            canonical_fixture.checkpoints["design_predictive_selection"].fingerprint
        ).path
        / "model-manifest.json"
    )
    value = json.loads(checkpoint_manifest.read_text())
    value["identity"]["model_metadata"]["selected_n_states"] = 3
    checkpoint_manifest.write_bytes(_canonical_bytes(value))
    with pytest.raises(IntegrityError, match="hash"):
        other.evidence_store.load(
            other.evidence_fingerprint,
            run_store=other.run_store,
            checkpoint_store=other.checkpoint_store,
            live_preflight=other.live_preflight,
            model_store=other.model_store,
        )


def test_loader_rejects_noncompleted_run_and_missing_artifact(
    tmp_path: Path, canonical_fixture: _Fixture
) -> None:
    fixture = _copy_fixture(canonical_fixture, tmp_path / "launch")
    launch_path = fixture.run_store.verify(fixture.run_id).path / "launch-state.json"
    launch = json.loads(launch_path.read_text())
    launch["state"] = "running"
    identity = {
        key: value for key, value in launch.items() if key != "launch_fingerprint"
    }
    launch["launch_fingerprint"] = _fingerprint(identity)
    launch_path.write_bytes(_canonical_bytes(launch))
    with pytest.raises(IntegrityError, match="completion marker changed|not completed"):
        fixture.evidence_store.load(
            fixture.evidence_fingerprint,
            run_store=fixture.run_store,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            model_store=fixture.model_store,
        )

    other = _copy_fixture(canonical_fixture, tmp_path / "artifact")
    design_path = other.model_store.verify(
        canonical_fixture.design.reference.fingerprint
    ).path
    (design_path / "reports" / "model-fit.json").unlink()
    with pytest.raises(IntegrityError, match="allowlist|missing|opened"):
        other.evidence_store.load(
            other.evidence_fingerprint,
            run_store=other.run_store,
            checkpoint_store=other.checkpoint_store,
            live_preflight=other.live_preflight,
            model_store=other.model_store,
        )


def test_loader_rejects_manifest_tamper_extra_files_and_serialized_true_flag(
    tmp_path: Path, canonical_fixture: _Fixture
) -> None:
    fixture = _copy_fixture(canonical_fixture, tmp_path / "tamper")
    manifest_path = (
        fixture.evidence_store.store_root
        / fixture.evidence_fingerprint
        / G3_EVIDENCE_MANIFEST_NAME
    )
    value = json.loads(manifest_path.read_text())
    value["identity"]["run"]["workers"] = 8
    manifest_path.write_bytes(_canonical_bytes(value))
    with pytest.raises(IntegrityError, match="hash"):
        fixture.evidence_store.load(
            fixture.evidence_fingerprint,
            run_store=fixture.run_store,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            model_store=fixture.model_store,
        )

    extra = _copy_fixture(canonical_fixture, tmp_path / "extra")
    extra_path = (
        extra.evidence_store.store_root / extra.evidence_fingerprint / "extra.json"
    )
    extra_path.touch()
    with pytest.raises(IntegrityError, match="allowlist"):
        extra.evidence_store.load(
            extra.evidence_fingerprint,
            run_store=extra.run_store,
            checkpoint_store=extra.checkpoint_store,
            live_preflight=extra.live_preflight,
            model_store=extra.model_store,
        )

    true_flag = _copy_fixture(canonical_fixture, tmp_path / "true")
    old_root = true_flag.evidence_store.store_root / true_flag.evidence_fingerprint
    true_value = json.loads((old_root / G3_EVIDENCE_MANIFEST_NAME).read_text())
    true_value["identity"]["evidence_bundle"]["identity"]["generation_authorized"] = (
        True
    )
    new_fingerprint = _fingerprint(true_value["identity"])
    true_value["evidence_fingerprint"] = new_fingerprint
    new_root = true_flag.evidence_store.store_root / new_fingerprint
    new_root.mkdir()
    (new_root / G3_EVIDENCE_MANIFEST_NAME).write_bytes(_canonical_bytes(true_value))
    with pytest.raises(
        IntegrityError,
        match="complete passing decision|decision fields are invalid",
    ):
        true_flag.evidence_store.load(
            new_fingerprint,
            run_store=true_flag.run_store,
            checkpoint_store=true_flag.checkpoint_store,
            live_preflight=true_flag.live_preflight,
            model_store=true_flag.model_store,
        )


def _invalid_manifest_case(value: dict[str, Any], case: str) -> None:
    identity = value["identity"]
    if case == "identity_type":
        value["identity"] = []
    elif case == "contract":
        identity["schema_id"] = "unsupported"
    elif case == "top_passed":
        identity["passed"] = False
    elif case == "top_authorized":
        identity["generation_authorized"] = True
    elif case == "bundle_identity_type":
        identity["evidence_bundle"]["identity"] = []
    elif case == "record_count":
        identity["evidence_bundle"]["identity"]["records"] = []
    elif case == "record_type":
        identity["evidence_bundle"]["identity"]["records"][0] = []
    elif case == "record_status":
        identity["evidence_bundle"]["identity"]["records"][0]["status"] = "failed"
    elif case == "record_summary":
        identity["evidence_bundle"]["identity"]["records"][0]["summary"] = {}
    elif case == "completion":
        identity["run"]["completion_sequence"] = 1
    elif case == "task_mapping":
        del identity["run"]["task_result_fingerprints"]["input_integrity"]
    elif case == "look_sequence":
        identity["run"]["planned_look_fingerprints"]["kernel_design_monthly"] = []
    elif case == "artifact_fingerprint":
        identity["artifacts"]["design_artifact_fingerprint"] = "bad"
    elif case == "provenance_fingerprint":
        identity["provenance"]["dependency_lock_fingerprint"] = "bad"
    else:  # pragma: no cover - closed test parameter registry
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case,match",
    [
        ("identity_type", "identity is invalid"),
        ("contract", "contract identity is unsupported"),
        ("top_passed", "contract identity is unsupported"),
        ("top_authorized", "contract identity is unsupported"),
        ("bundle_identity_type", "bundle identity is invalid"),
        ("record_count", "records are invalid"),
        ("record_type", "record is invalid"),
        ("record_status", "record summary is invalid"),
        ("record_summary", "record summary is invalid"),
        ("completion", "completion fields are invalid"),
        ("task_mapping", "mapping is not exact"),
        ("look_sequence", "sequence is invalid"),
        ("artifact_fingerprint", "SHA-256 fingerprint"),
        ("provenance_fingerprint", "SHA-256 fingerprint"),
    ],
)
def test_manifest_schema_rejects_readdressed_counterfeits(
    tmp_path: Path,
    canonical_fixture: _Fixture,
    case: str,
    match: str,
) -> None:
    store, fingerprint = _counterfeit_manifest(
        canonical_fixture,
        tmp_path / case,
        lambda value: _invalid_manifest_case(value, case),
    )
    with pytest.raises(IntegrityError, match=match):
        store.verify(fingerprint)


def test_manifest_reader_rejects_missing_invalid_noncanonical_and_wrong_headers(
    tmp_path: Path,
    canonical_fixture: _Fixture,
) -> None:
    with pytest.raises(IntegrityError, match="lowercase SHA-256"):
        G3EvidenceStore(tmp_path / "bad-fingerprint").verify("bad")
    with pytest.raises(IntegrityError, match="missing or unsafe"):
        G3EvidenceStore(tmp_path / "missing").verify("1" * 64)

    source = (
        canonical_fixture.evidence_store.store_root
        / canonical_fixture.evidence_fingerprint
        / G3_EVIDENCE_MANIFEST_NAME
    )
    canonical = json.loads(source.read_text())
    for name, content, match in (
        ("invalid-json", b"{", "invalid JSON"),
        (
            "noncanonical",
            json.dumps(canonical, indent=2).encode(),
            "not canonical JSON",
        ),
    ):
        store = G3EvidenceStore(tmp_path / name / "artifacts")
        root = store.store_root / canonical_fixture.evidence_fingerprint
        root.mkdir(parents=True)
        (root / G3_EVIDENCE_MANIFEST_NAME).write_bytes(content)
        with pytest.raises(IntegrityError, match=match):
            store.verify(canonical_fixture.evidence_fingerprint)

    for name, key, replacement, match in (
        (
            "schema",
            "schema_version",
            G3_EVIDENCE_SCHEMA_VERSION - 1,
            "schema is unsupported",
        ),
        ("address", "evidence_fingerprint", "0" * 64, "address is inconsistent"),
    ):
        value = json.loads(source.read_text())
        value[key] = replacement
        store = G3EvidenceStore(tmp_path / name / "artifacts")
        root = store.store_root / canonical_fixture.evidence_fingerprint
        root.mkdir(parents=True)
        (root / G3_EVIDENCE_MANIFEST_NAME).write_bytes(_canonical_bytes(value))
        with pytest.raises(IntegrityError, match=match):
            store.verify(canonical_fixture.evidence_fingerprint)


def test_loader_rejects_readdressed_external_fingerprint_and_bundle_mismatches(
    tmp_path: Path,
    canonical_fixture: _Fixture,
) -> None:
    task = _copy_fixture(canonical_fixture, tmp_path / "task-fingerprint")
    task_fingerprint = _readdress_fixture(
        task,
        lambda value: value["identity"]["run"]["task_result_fingerprints"].__setitem__(
            "input_integrity", "0" * 64
        ),
    )
    with pytest.raises(IntegrityError, match="task-result fingerprints changed"):
        task.evidence_store.load(
            task_fingerprint,
            run_store=task.run_store,
            checkpoint_store=task.checkpoint_store,
            live_preflight=task.live_preflight,
            model_store=task.model_store,
        )

    look = _copy_fixture(canonical_fixture, tmp_path / "look-fingerprint")
    look_fingerprint = _readdress_fixture(
        look,
        lambda value: value["identity"]["run"]["planned_look_fingerprints"]
        .__getitem__("kernel_design_monthly")
        .__setitem__(0, "0" * 64),
    )
    with pytest.raises(IntegrityError, match="planned-look fingerprints changed"):
        look.evidence_store.load(
            look_fingerprint,
            run_store=look.run_store,
            checkpoint_store=look.checkpoint_store,
            live_preflight=look.live_preflight,
            model_store=look.model_store,
        )

    bundle = _copy_fixture(canonical_fixture, tmp_path / "bundle-fingerprint")
    bundle_fingerprint = _readdress_fixture(
        bundle,
        lambda value: value["identity"]["evidence_bundle"].__setitem__(
            "fingerprint", "0" * 64
        ),
    )
    with pytest.raises(IntegrityError, match="bundle identity changed"):
        bundle.evidence_store.load(
            bundle_fingerprint,
            run_store=bundle.run_store,
            checkpoint_store=bundle.checkpoint_store,
            live_preflight=bundle.live_preflight,
            model_store=bundle.model_store,
        )


def test_evidence_directory_symlink_and_manifest_hardlink_fail_closed(
    tmp_path: Path, canonical_fixture: _Fixture
) -> None:
    fixture = _copy_fixture(canonical_fixture, tmp_path / "hardlink")
    manifest = (
        fixture.evidence_store.store_root
        / fixture.evidence_fingerprint
        / G3_EVIDENCE_MANIFEST_NAME
    )
    hardlink = tmp_path / "manifest-hardlink.json"
    hardlink.hardlink_to(manifest)
    with pytest.raises(IntegrityError, match="private regular"):
        fixture.evidence_store.verify(fixture.evidence_fingerprint)

    real = tmp_path / "real"
    real.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symbolic link"):
        G3EvidenceStore(linked_root).verify("1" * 64)


def test_publish_rejects_swapped_artifacts_and_changed_completion_marker(
    canonical_fixture: _Fixture,
) -> None:
    fixture = canonical_fixture
    with pytest.raises(IntegrityError, match="artifact pair identity"):
        fixture.evidence_store.publish(
            bundle=fixture.bundle,
            run_store=fixture.run_store,
            run_id=fixture.run_id,
            completion_marker=fixture.completion,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            checkpoints=fixture.checkpoints,
            model_store=fixture.model_store,
            design_artifact=fixture.production,
            production_artifact=fixture.design,
        )
    changed = replace(fixture.completion, sequence=fixture.completion.sequence + 1)
    with pytest.raises(IntegrityError, match="completion marker changed"):
        fixture.evidence_store.publish(
            bundle=fixture.bundle,
            run_store=fixture.run_store,
            run_id=fixture.run_id,
            completion_marker=changed,
            checkpoint_store=fixture.checkpoint_store,
            live_preflight=fixture.live_preflight,
            checkpoints=fixture.checkpoints,
            model_store=fixture.model_store,
            design_artifact=fixture.design,
            production_artifact=fixture.production,
        )


def _failed_bundle() -> Any:
    records = tuple(
        gate_record(
            task_id,
            passed=task_id != "input_integrity",
            evidence={"task_id": task_id},
            summary={"status": "failed" if task_id == "input_integrity" else "passed"},
        )
        for task_id in G3_GATE_ORDER
    )
    return build_g3_evidence_bundle(
        config_fingerprint=_CONFIG,
        processed_data_fingerprint=_PROCESSED,
        calibration_input_fingerprint=_INPUT,
        design_artifact_fingerprint=None,
        production_artifact_fingerprint=None,
        records=records,
    )


def test_loaded_evidence_constructor_and_publish_type_decision_guards(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="created only by G3EvidenceStore.load"):
        evidence_module.LoadedG3EvidenceAuthority()

    store = G3EvidenceStore(tmp_path / "artifacts")
    common = {
        "run_store": object(),
        "run_id": "1" * 64,
        "completion_marker": object(),
        "checkpoint_store": object(),
        "live_preflight": object(),
        "checkpoints": {},
        "model_store": object(),
        "design_artifact": object(),
        "production_artifact": object(),
    }
    with pytest.raises(ModelError, match="requires a typed bundle"):
        store.publish(bundle=object(), **common)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="requires passed evidence only"):
        store.publish(bundle=_failed_bundle(), **common)  # type: ignore[arg-type]


def test_loaded_evidence_in_memory_reverification_rejects_manifest_and_bindings(
    canonical_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = canonical_fixture.evidence_store.load(
        canonical_fixture.evidence_fingerprint,
        run_store=canonical_fixture.run_store,
        checkpoint_store=canonical_fixture.checkpoint_store,
        live_preflight=canonical_fixture.live_preflight,
        model_store=canonical_fixture.model_store,
    )

    reference = loaded.reference
    manifest = reference.manifest
    object.__setattr__(reference, "manifest", object())
    try:
        with pytest.raises(IntegrityError, match="manifest is unavailable"):
            verify_loaded_g3_evidence(loaded)
    finally:
        object.__setattr__(reference, "manifest", manifest)

    original_reference_fingerprint = reference.fingerprint
    object.__setattr__(reference, "fingerprint", "0" * 64)
    try:
        with pytest.raises(IntegrityError, match="reference identity changed"):
            verify_loaded_g3_evidence(loaded)
    finally:
        object.__setattr__(reference, "fingerprint", original_reference_fingerprint)

    original_run_id = loaded.run_id
    object.__setattr__(loaded, "run_id", "0" * 64)
    try:
        with pytest.raises(IntegrityError, match="bindings changed"):
            verify_loaded_g3_evidence(loaded)
    finally:
        object.__setattr__(loaded, "run_id", original_run_id)

    original_source_fingerprint = loaded.g3_source_code_fingerprint
    object.__setattr__(loaded, "g3_source_code_fingerprint", "0" * 64)
    try:
        with pytest.raises(IntegrityError, match="bindings changed"):
            verify_loaded_g3_evidence(loaded)
    finally:
        object.__setattr__(
            loaded,
            "g3_source_code_fingerprint",
            original_source_fingerprint,
        )

    original_bundle = loaded.evidence_bundle
    object.__setattr__(
        loaded, "evidence_bundle", replace(original_bundle, passed=False)
    )
    with monkeypatch.context() as context:
        context.setattr(
            evidence_module, "verify_g3_evidence_bundle", lambda _value: None
        )
        with pytest.raises(IntegrityError, match="not a passed evidence-only run"):
            verify_loaded_g3_evidence(loaded)
    object.__setattr__(loaded, "evidence_bundle", original_bundle)
    assert verify_loaded_g3_evidence(loaded) is loaded


def test_canonical_json_and_schema_helpers_reject_noncanonical_shapes() -> None:
    for value in ({"value": np.nan}, {"value": np.inf}, {"value": {1, 2}}):
        with pytest.raises(IntegrityError, match="noncanonical JSON values"):
            evidence_module._canonical_bytes(value)

    with pytest.raises(IntegrityError, match="persisted G3 bundle is invalid"):
        evidence_module._persisted_bundle(object())
    with pytest.raises(IntegrityError, match="mapping is not exact"):
        evidence_module._fingerprint_map(
            object(),
            expected=("a",),
            label="test fingerprints",
        )
    with pytest.raises(IntegrityError, match="mapping is not exact"):
        evidence_module._look_fingerprint_map(
            object(),
            label="test planned looks",
        )
    with pytest.raises(IntegrityError, match="keys are invalid"):
        evidence_module._exact_keys(object(), {"a"}, "test object")


def test_manifest_directory_changed_between_allowlist_checks(
    canonical_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (
        canonical_fixture.evidence_store.store_root
        / canonical_fixture.evidence_fingerprint
    )
    original_iterdir = Path.iterdir
    calls = 0

    def changing_iterdir(path: Path):
        nonlocal calls
        entries = tuple(original_iterdir(path))
        if path == root:
            calls += 1
            if calls == 2:
                return iter((*entries, root / "late-file"))
        return iter(entries)

    with monkeypatch.context() as context:
        context.setattr(Path, "iterdir", changing_iterdir)
        with pytest.raises(IntegrityError, match="changed during verification"):
            canonical_fixture.evidence_store.verify(
                canonical_fixture.evidence_fingerprint
            )


def test_write_or_verify_rejects_existing_or_concurrent_byte_disagreement(
    tmp_path: Path,
    canonical_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = canonical_fixture.evidence_fingerprint
    manifest = (
        canonical_fixture.evidence_store.store_root
        / fingerprint
        / G3_EVIDENCE_MANIFEST_NAME
    ).read_bytes()
    with pytest.raises(IntegrityError, match="existing G3 evidence bytes differ"):
        canonical_fixture.evidence_store._write_or_verify(
            fingerprint=fingerprint,
            content=b"different",
        )

    for name, published_content, requested_content, expected in (
        ("matching", manifest, manifest, None),
        (
            "different",
            manifest,
            b"different",
            "concurrent G3 evidence publication differs",
        ),
    ):
        store = G3EvidenceStore(tmp_path / name / "artifacts")
        destination = store.store_root / fingerprint

        def concurrent_publish(
            _source: object,
            _destination: object,
            *,
            target: Path = destination,
            content: bytes = published_content,
        ) -> None:
            target.mkdir()
            (target / G3_EVIDENCE_MANIFEST_NAME).write_bytes(content)
            raise OSError("synthetic publication race")

        with monkeypatch.context() as context:
            context.setattr(evidence_module.os, "rename", concurrent_publish)
            if expected is None:
                assert store._write_or_verify(
                    fingerprint=fingerprint,
                    content=requested_content,
                )
            else:
                with pytest.raises(IntegrityError, match=expected):
                    store._write_or_verify(
                        fingerprint=fingerprint,
                        content=requested_content,
                    )

    store = G3EvidenceStore(tmp_path / "rename-failure" / "artifacts")
    with monkeypatch.context() as context:
        context.setattr(
            evidence_module.os,
            "rename",
            lambda *_args: (_ for _ in ()).throw(OSError("synthetic rename failure")),
        )
        with pytest.raises(OSError, match="synthetic rename failure"):
            store._write_or_verify(fingerprint=fingerprint, content=manifest)


def test_path_and_stable_regular_file_operating_system_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(IntegrityError, match="manifest is missing"):
        evidence_module._read_stable_regular(missing)

    regular = tmp_path / "manifest.json"
    regular.write_bytes(b"manifest")
    with monkeypatch.context() as context:
        context.setattr(
            evidence_module.os,
            "open",
            lambda *_args: (_ for _ in ()).throw(OSError("synthetic open failure")),
        )
        with pytest.raises(IntegrityError, match="cannot be opened safely"):
            evidence_module._read_stable_regular(regular)

    original_lstat = Path.lstat
    calls = 0

    def disappearing_lstat(path: Path):
        nonlocal calls
        if path == regular:
            calls += 1
            if calls == 2:
                raise OSError("synthetic disappearance")
        return original_lstat(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "lstat", disappearing_lstat)
        with pytest.raises(IntegrityError, match="disappeared while reading"):
            evidence_module._read_stable_regular(regular)

    with monkeypatch.context() as context:
        context.setattr(evidence_module.os, "read", lambda *_args: b"")
        with pytest.raises(IntegrityError, match="byte count changed"):
            evidence_module._read_stable_regular(regular)


def test_path_inspection_and_store_root_type_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    original_lstat = Path.lstat

    def failed_lstat(path: Path):
        if path == target:
            raise OSError("synthetic lstat failure")
        return original_lstat(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "lstat", failed_lstat)
        with pytest.raises(IntegrityError, match="path cannot be inspected"):
            evidence_module._reject_symlink_components(target)

    nonexistent = tmp_path / "not-a-directory"
    with monkeypatch.context() as context:
        context.setattr(evidence_module, "_reject_symlink_components", lambda _p: None)
        context.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)
        context.setattr(Path, "is_dir", lambda _path: False)
        with pytest.raises(IntegrityError, match="root is not a directory"):
            evidence_module._prepare_directory(nonexistent)


def _external_state_kwargs(fixture: _Fixture) -> dict[str, Any]:
    return {
        "run_store": fixture.run_store,
        "run_id": fixture.run_id,
        "checkpoint_store": fixture.checkpoint_store,
        "live_preflight": fixture.live_preflight,
        "model_store": fixture.model_store,
        "expected_bundle": fixture.bundle,
        "expected_completion": fixture.completion,
        "expected_task_fingerprints": None,
        "expected_checkpoint_fingerprints": None,
        "expected_look_fingerprints": None,
        "expected_artifact_fingerprints": None,
        "expected_pair_receipt_fingerprint": None,
        "supplied_checkpoints": None,
        "supplied_artifacts": (fixture.design, fixture.production),
        "require_persisted_bundle": False,
    }


def test_external_state_rejects_untyped_stores_markers_bundles_and_failed_runs(
    canonical_fixture: _Fixture,
) -> None:
    cases = (
        ("run_store", object(), "requires a CalibrationRunStore"),
        ("checkpoint_store", object(), "requires a checkpoint store"),
        ("model_store", object(), "requires a model artifact store"),
        ("expected_completion", object(), "requires a typed completion marker"),
        ("expected_bundle", object(), "bundle has the wrong type"),
    )
    for key, invalid, message in cases:
        arguments = _external_state_kwargs(canonical_fixture)
        arguments[key] = invalid
        with pytest.raises(ModelError, match=message):
            evidence_module._verify_external_state(**arguments)

    for persisted, error_type, message in (
        (True, IntegrityError, "persisted G3 evidence is not passed"),
        (False, ModelError, "bundle is not passed"),
    ):
        arguments = _external_state_kwargs(canonical_fixture)
        arguments["expected_bundle"] = _failed_bundle()
        arguments["require_persisted_bundle"] = persisted
        with pytest.raises(error_type, match=message):
            evidence_module._verify_external_state(**arguments)

    arguments = _external_state_kwargs(canonical_fixture)
    arguments["expected_bundle"] = replace(
        canonical_fixture.bundle,
        config_fingerprint="9" * 64,
    )
    arguments["require_persisted_bundle"] = True
    with pytest.raises(IntegrityError, match="completed-run identities disagree"):
        evidence_module._verify_external_state(**arguments)


def _verified_run_evidence(
    fixture: _Fixture,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    results = fixture.run_store.verify_task_results(
        fixture.run_id,
        require_complete=True,
    )
    receipts = fixture.run_store.verify_planned_look_receipts(
        fixture.run_id,
        require_terminal=True,
        task_results=results,
    )
    return results, evidence_module._verified_look_fingerprints(receipts)


def test_completion_look_input_and_checkpoint_digest_guards(
    canonical_fixture: _Fixture,
) -> None:
    with pytest.raises(IntegrityError, match="exact planned-look families"):
        evidence_module._verified_look_fingerprints({})

    results, _looks = _verified_run_evidence(canonical_fixture)
    checkpoint = canonical_fixture.checkpoints["input_integrity"]
    result = results["input_integrity"]
    payload = result.payload
    evidence = dict(checkpoint.state["decision"]["evidence"])

    missing_field = dict(evidence)
    missing_field.pop("check_count")
    with pytest.raises(IntegrityError, match="evidence fields are invalid"):
        evidence_module._verify_input_integrity_evidence(
            missing_field,
            checkpoint,
            result,
            source_code_fingerprint=_SOURCE,
            dependency_lock_fingerprint=_LOCK,
            workers=canonical_fixture.completion.workers,
        )

    invalid_binding = dict(evidence)
    invalid_binding["check_count"] = False
    with pytest.raises(IntegrityError, match="evidence binding is invalid"):
        evidence_module._verify_input_integrity_evidence(
            invalid_binding,
            checkpoint,
            result,
            source_code_fingerprint=_SOURCE,
            dependency_lock_fingerprint=_LOCK,
            workers=canonical_fixture.completion.workers,
        )

    digest_arguments = {
        "decision_fingerprint": payload["decision_fingerprint"],
        "planned_look_fingerprints": payload["planned_look_fingerprints"],
        "serialized_binding_sha256": payload["serialized_binding_sha256"],
    }
    missing_digest = dict(checkpoint.arrays)
    missing_digest.pop("decision_sha256")
    with pytest.raises(IntegrityError, match="digest arrays are invalid"):
        evidence_module._verify_checkpoint_digest_arrays(
            replace(checkpoint, arrays=missing_digest),
            **digest_arguments,
        )

    changed_digest = dict(checkpoint.arrays)
    changed_decision = np.array(changed_digest["decision_sha256"], copy=True)
    changed_decision[0] ^= 1
    changed_digest["decision_sha256"] = changed_decision
    with pytest.raises(IntegrityError, match="digest arrays disagree"):
        evidence_module._verify_checkpoint_digest_arrays(
            replace(checkpoint, arrays=changed_digest),
            **digest_arguments,
        )


def test_task_checkpoint_bundle_binding_guards(
    canonical_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, looks = _verified_run_evidence(canonical_fixture)

    def verify(
        *,
        bundle: Any = canonical_fixture.bundle,
        changed_results: Mapping[str, Any] = results,
        checkpoints: Mapping[str, CalibrationCheckpointReference] = (
            canonical_fixture.checkpoints
        ),
    ) -> None:
        evidence_module._verify_task_checkpoint_bundle_bindings(
            bundle,
            results=changed_results,
            checkpoints=checkpoints,
            planned_looks=looks,
            source_code_fingerprint=_SOURCE,
            dependency_lock_fingerprint=_LOCK,
            workers=canonical_fixture.completion.workers,
        )

    def with_payload(task_id: str, **updates: Any) -> dict[str, Any]:
        changed = dict(results)
        payload = dict(results[task_id].payload)
        payload.update(updates)
        changed[task_id] = replace(results[task_id], payload=payload)
        return changed

    with pytest.raises(IntegrityError, match="record order differs"):
        verify(bundle=replace(canonical_fixture.bundle, records=()))

    first = G3_GATE_ORDER[0]
    missing_payload = with_payload(first)
    first_payload = dict(missing_payload[first].payload)
    first_payload.pop("summary")
    missing_payload[first] = replace(missing_payload[first], payload=first_payload)
    with pytest.raises(IntegrityError, match="payload fields are invalid"):
        verify(changed_results=missing_payload)

    with pytest.raises(IntegrityError, match="payload identity is invalid"):
        verify(changed_results=with_payload(first, coordinator_schema_version="wrong"))
    with pytest.raises(IntegrityError, match="payload identity is invalid"):
        verify(
            changed_results=with_payload(
                first,
                scientific_task_binding={"binding_fingerprint": "9" * 64},
            )
        )
    with pytest.raises(IntegrityError, match="planned-look binding is invalid"):
        verify(
            changed_results=with_payload(
                first,
                planned_look_fingerprints={"unexpected": []},
            )
        )

    selection = "design_predictive_selection"
    with pytest.raises(IntegrityError, match="no valid frozen K"):
        verify(changed_results=with_payload(selection, selected_n_states=None))
    with pytest.raises(IntegrityError, match="prematurely binds K"):
        verify(changed_results=with_payload(first, selected_n_states=_K))

    selection_index = G3_GATE_ORDER.index(selection)
    post_selection = G3_GATE_ORDER[selection_index + 1]
    with pytest.raises(IntegrityError, match="changed the frozen selected K"):
        verify(changed_results=with_payload(post_selection, selected_n_states=5))

    changed_checkpoints = dict(canonical_fixture.checkpoints)
    changed_checkpoints[first] = replace(
        changed_checkpoints[first],
        selected_n_states=_K,
    )
    with pytest.raises(IntegrityError, match="selected-K binding differs"):
        verify(checkpoints=changed_checkpoints)

    changed_checkpoints = dict(canonical_fixture.checkpoints)
    state = dict(changed_checkpoints[first].state)
    state.pop("decision")
    changed_checkpoints[first] = replace(changed_checkpoints[first], state=state)
    with pytest.raises(IntegrityError, match="checkpoint state fields are invalid"):
        verify(checkpoints=changed_checkpoints)

    changed_checkpoints = dict(canonical_fixture.checkpoints)
    state = dict(changed_checkpoints[first].state)
    decision = dict(state["decision"])
    decision.pop("summary")
    state["decision"] = decision
    changed_checkpoints[first] = replace(changed_checkpoints[first], state=state)
    with pytest.raises(IntegrityError, match="checkpoint decision is invalid"):
        verify(checkpoints=changed_checkpoints)

    changed_checkpoints = dict(canonical_fixture.checkpoints)
    state = dict(changed_checkpoints[first].state)
    state["publication_schema_version"] = "wrong"
    changed_checkpoints[first] = replace(changed_checkpoints[first], state=state)
    with pytest.raises(IntegrityError, match="decision binding is invalid"):
        verify(checkpoints=changed_checkpoints)

    with monkeypatch.context() as context:
        context.setattr(
            evidence_module,
            "gate_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ModelError("synthetic invalid decision")
            ),
        )
        with pytest.raises(IntegrityError, match="cannot form a gate record"):
            verify()

    changed_record = replace(
        canonical_fixture.bundle.records[0],
        summary={"status": "changed"},
    )
    changed_bundle = replace(
        canonical_fixture.bundle,
        records=(changed_record, *canonical_fixture.bundle.records[1:]),
    )
    with pytest.raises(IntegrityError, match="record differs from checkpoint"):
        verify(bundle=changed_bundle)


def test_loaded_artifact_and_artifact_binding_identity_guards(
    canonical_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelError, match="two typed scientific artifacts"):
        evidence_module._verified_artifacts(
            model_store=canonical_fixture.model_store,
            expected_fingerprints=None,
            supplied=(object(), canonical_fixture.production),
        )

    with monkeypatch.context() as context:
        context.setattr(
            evidence_module,
            "load_scientific_model_artifact",
            lambda *_args: canonical_fixture.production,
        )
        with pytest.raises(ModelError, match="differ from their store"):
            evidence_module._verified_artifacts(
                model_store=canonical_fixture.model_store,
                expected_fingerprints=None,
                supplied=(canonical_fixture.design, canonical_fixture.production),
            )

    results, looks = _verified_run_evidence(canonical_fixture)
    task_fingerprints = {
        task_id: results[task_id].fingerprint for task_id in G3_GATE_ORDER
    }
    checkpoint_fingerprints = {
        task_id: canonical_fixture.checkpoints[task_id].fingerprint
        for task_id in G3_GATE_ORDER
    }
    artifact_evidence = dict(
        canonical_fixture.checkpoints["artifact_integrity"].state["decision"][
            "evidence"
        ]
    )

    def verify_artifacts(
        evidence: Mapping[str, Any],
        checkpoints: Mapping[str, CalibrationCheckpointReference],
    ) -> None:
        evidence_module._verify_artifact_bindings(
            bundle=canonical_fixture.bundle,
            model_store=canonical_fixture.model_store,
            design=canonical_fixture.design,
            production=canonical_fixture.production,
            run_id=canonical_fixture.run_id,
            completion=canonical_fixture.completion,
            source_code_fingerprint=_SOURCE,
            dependency_lock_fingerprint=_LOCK,
            task_result_fingerprints=task_fingerprints,
            checkpoint_fingerprints=checkpoint_fingerprints,
            planned_look_fingerprints=looks,
            artifact_integrity_evidence=evidence,
            checkpoints=checkpoints,
        )

    changed_evidence = dict(artifact_evidence)
    changed_evidence["design_artifact_fingerprint"] = "9" * 64
    with pytest.raises(IntegrityError, match="differs from artifacts"):
        verify_artifacts(changed_evidence, canonical_fixture.checkpoints)

    changed_checkpoints = dict(canonical_fixture.checkpoints)
    first = G3_GATE_ORDER[0]
    changed_checkpoints[first] = replace(
        changed_checkpoints[first],
        state={"decision": None},
    )
    with pytest.raises(IntegrityError, match="lacks its durable decision"):
        verify_artifacts(artifact_evidence, changed_checkpoints)


def test_stable_regular_file_rejects_metadata_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"manifest")
    original_fstat = evidence_module.os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        info = original_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_mode=info.st_mode,
                st_nlink=info.st_nlink,
                st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns + 1,
                st_ctime_ns=info.st_ctime_ns,
            )
        return info

    with monkeypatch.context() as context:
        context.setattr(evidence_module.os, "fstat", changing_fstat)
        with pytest.raises(IntegrityError, match="changed while reading"):
            evidence_module._read_stable_regular(manifest)
