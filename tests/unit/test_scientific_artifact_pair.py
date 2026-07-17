from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import tests.unit.test_adequacy_gate as adequacy_fixtures

from prpg.errors import IntegrityError, ModelError
from prpg.model import g3_assembly as assembly_module
from prpg.model import g3_calibration_views as calibration_views_module
from prpg.model import scientific_artifact_pair as pair_module
from prpg.model.artifact import ModelArtifactStore
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.scientific_artifact import (
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    ScientificReportPayload,
)
from prpg.model.scientific_artifact_pair import (
    SCIENTIFIC_ARTIFACT_PAIR_RECEIPT,
    SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
    SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
    CanonicalScientificModelArtifactDraft,
    LoadedScientificArtifactPair,
    ScientificArtifactPairStore,
)
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID,
    SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificCandidatePolicySet,
)


def _canonical(value: object) -> bytes:
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


def _stored_receipt(tmp_path: Path) -> tuple[ScientificArtifactPairStore, str, Path]:
    store = ScientificArtifactPairStore(tmp_path / "artifacts")
    identity = {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "test_identity": "receipt-storage-fixture",
    }
    fingerprint = pair_module._sha256_bytes(_canonical(identity))
    manifest = {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "pair_receipt_fingerprint": fingerprint,
        "identity": identity,
    }
    assert not store._write_or_reuse(fingerprint, _canonical(manifest))
    reference = store.verify(fingerprint, reused=False)
    return store, fingerprint, reference.path / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT


def _fingerprint(label: str) -> str:
    return pair_module._sha256_bytes(label.encode())


def _real_adequacy_views(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[Any, Any, Any, Any], Any]:
    """Return reduced but producer-sealed adequacy views for report tests."""

    adequacy_fixtures._parents.cache_clear()
    monkeypatch.setattr(
        calibration_views_module,
        "is_registered_candidate_selection_task_view",
        calibration_views_module.is_candidate_selection_task_view,
    )
    views = adequacy_fixtures._cell_views(strict_canonical=False)
    return views, adequacy_fixtures._holm_view(views)


def _candidate_policy(
    *, monthly_block_length: int, daily_block_length: int
) -> ScientificCandidatePolicySet:
    return ScientificCandidatePolicySet(
        schema_id=SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID,
        schema_version=SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=monthly_block_length,
        daily_block_length=daily_block_length,
        candidate_policies=("historical_vector", "kernel_mean_neutral"),
        neutral_lambda_candidates=CANONICAL_NEUTRAL_LAMBDA_GRID,
        selected_policy=None,
        selected_neutral_lambda=None,
        qualification_gate="G5",
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


def _write_pair_identity(
    store: ScientificArtifactPairStore,
    identity: dict[str, Any],
) -> str:
    fingerprint = pair_module._sha256_bytes(_canonical(identity))
    manifest = {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "pair_receipt_fingerprint": fingerprint,
        "identity": identity,
    }
    store._write_or_reuse(fingerprint, _canonical(manifest))
    return fingerprint


def _durable_receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ScientificArtifactPairStore,
    dict[str, Any],
    dict[str, Any],
]:
    artifact_root = tmp_path / "artifacts"
    pair_store = ScientificArtifactPairStore(artifact_root)
    model_store = ModelArtifactStore(artifact_root)
    run_id = _fingerprint("run")
    selected_k = 3
    binding = ScientificArtifactBinding(
        config_fingerprint=_fingerprint("config"),
        processed_data_fingerprint=_fingerprint("processed"),
        calibration_input_fingerprint=_fingerprint("calibration-input"),
        calibration_run_fingerprint=run_id,
        preflight_fingerprint=_fingerprint("preflight"),
    )
    provenance = ScientificArtifactProvenance(
        source_code_fingerprint=_fingerprint("source"),
        dependency_lock_fingerprint=_fingerprint("lock"),
    )
    first_25 = G3_GATE_ORDER[:-1]
    task_fingerprints = {
        task_id: _fingerprint(f"task:{task_id}") for task_id in first_25
    }
    checkpoint_fingerprints = {
        task_id: _fingerprint(f"checkpoint:{task_id}") for task_id in first_25
    }
    planned_look_fingerprints = {
        "durable-test-family": (
            _fingerprint("look:0"),
            _fingerprint("look:1"),
        )
    }
    bridge_parent = _fingerprint("bridge-parent")
    role_sources = {
        "design": _fingerprint("design-role-source"),
        "production": _fingerprint("production-role-source"),
    }
    report_fingerprints = {
        "design": _fingerprint("design-reports"),
        "production": _fingerprint("production-reports"),
    }
    policies = {
        "design": _candidate_policy(monthly_block_length=4, daily_block_length=20),
        "production": _candidate_policy(monthly_block_length=7, daily_block_length=35),
    }

    def artifact(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            role=role,
            selected_k=selected_k,
            binding=binding,
            provenance=provenance,
            generation_policy=policies[role],
            reference=SimpleNamespace(fingerprint=_fingerprint(f"{role}-artifact")),
            core_model_fingerprint=_fingerprint(f"{role}-core"),
            parameter_set_fingerprint=_fingerprint(f"{role}-parameters"),
            reports={},
        )

    design = artifact("design")
    production = artifact("production")
    monkeypatch.setattr(
        pair_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: True,
    )
    monkeypatch.setattr(
        pair_module,
        "_verify_closed_reports_from_durable_decisions",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pair_module,
        "_loaded_report_set_fingerprint",
        lambda supplied: report_fingerprints[cast(Any, supplied).role],
    )
    policy_mappings = {role: policy.as_dict() for role, policy in policies.items()}
    artifact_identities = {
        role: {
            "role": role,
            "artifact_fingerprint": supplied.reference.fingerprint,
            "core_model_fingerprint": supplied.core_model_fingerprint,
            "parameter_set_fingerprint": supplied.parameter_set_fingerprint,
            "role_source_fingerprint": role_sources[role],
            "report_set_fingerprint": report_fingerprints[role],
            "candidate_policy_set_fingerprint": _fingerprint_from_value(
                policy_mappings[role]
            ),
        }
        for role, supplied in {"design": design, "production": production}.items()
    }
    prefix_identity = {
        "binding": binding.as_dict(),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "planned_look_fingerprints": {
            key: list(values) for key, values in planned_look_fingerprints.items()
        },
        "provenance": provenance.as_dict(),
        "run_id": run_id,
        "selected_n_states": selected_k,
        "task_result_fingerprints": task_fingerprints,
    }
    identity = {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "run_id": run_id,
        "selected_k": selected_k,
        "prefix_fingerprint": _fingerprint_from_value(prefix_identity),
        "binding": binding.as_dict(),
        "provenance": provenance.as_dict(),
        "task_result_fingerprints": task_fingerprints,
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "planned_look_fingerprints": {
            key: list(values) for key, values in planned_look_fingerprints.items()
        },
        "actual_bridge_parent_view_fingerprint": bridge_parent,
        "candidate_policy_sets": policy_mappings,
        "artifacts": artifact_identities,
        "report_pair_fingerprint": _fingerprint_from_value(
            {
                "schema_id": "prpg-g3-scientific-report-pair-v1",
                "design": report_fingerprints["design"],
                "production": report_fingerprints["production"],
            }
        ),
        "generation_authorized": False,
    }
    kwargs = {
        "pair_store": pair_store,
        "model_store": model_store,
        "design": design,
        "production": production,
        "run_id": run_id,
        "selected_k": selected_k,
        "binding": binding,
        "provenance": provenance,
        "task_result_fingerprints": task_fingerprints,
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "planned_look_fingerprints": planned_look_fingerprints,
        "decision_documents": {
            "actual_artifact_bridge": {"evidence": {"view_fingerprint": bridge_parent}}
        },
        "actual_bridge_parent_view_fingerprint": bridge_parent,
        "design_role_source_fingerprint": role_sources["design"],
        "production_role_source_fingerprint": role_sources["production"],
    }
    return pair_store, identity, kwargs


def _fingerprint_from_value(value: object) -> str:
    return pair_module._sha256_bytes(_canonical(value))


def _closed_report_verification_fixture(
    *,
    role: str = "design",
) -> tuple[
    SimpleNamespace,
    dict[str, str],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, Any]],
]:
    """Build durable report envelopes independently of the producer helper."""

    documents = pair_module.required_scientific_documents(cast(Any, role))
    task_ids = tuple(
        dict.fromkeys(
            task_id
            for document_type in documents.values()
            for task_id in pair_module.scientific_report_task_ids(
                document_type,
                cast(Any, role),
            )
        )
    )
    family_ids = tuple(
        dict.fromkeys(
            family_id
            for document_type in documents.values()
            for family_id in pair_module.scientific_report_planned_look_families(
                document_type,
                cast(Any, role),
            )
        )
    )
    task_result_fingerprints = {
        task_id: _fingerprint(f"closed-report-task:{task_id}") for task_id in task_ids
    }
    checkpoint_fingerprints = {
        task_id: _fingerprint(f"closed-report-checkpoint:{task_id}")
        for task_id in task_ids
    }
    planned_look_fingerprints = {
        family_id: (_fingerprint(f"closed-report-look:{family_id}:0"),)
        for family_id in family_ids
    }
    decision_documents: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        decision_documents[task_id] = {
            "gate_id": task_id,
            "passed": True,
            "source_type": "tests.ClosedReportSource",
            "evidence": {"task_id": task_id, "verified": True},
            "summary": {"status": "passed"},
        }

    reports: dict[str, dict[str, Any]] = {}
    for path, document_type in documents.items():
        report_tasks = pair_module.scientific_report_task_ids(
            document_type,
            cast(Any, role),
        )
        report_families = pair_module.scientific_report_planned_look_families(
            document_type,
            cast(Any, role),
        )
        entries: list[dict[str, Any]] = []
        for task_id in report_tasks:
            decision = decision_documents[task_id]
            identity = {
                "gate_id": task_id,
                "passed": True,
                "evidence": decision["evidence"],
                "summary": decision["summary"],
            }
            entries.append(
                {
                    **identity,
                    "source_type": decision["source_type"],
                    "source_view_fingerprint": _fingerprint(
                        f"closed-report-view:{task_id}"
                    ),
                    "decision_fingerprint": _fingerprint_from_value(identity),
                }
            )
        metrics = {
            "schema_id": pair_module.CLOSED_REPORT_METRICS_SCHEMA_ID,
            "schema_version": pair_module.CLOSED_REPORT_METRICS_SCHEMA_VERSION,
            "document_type": document_type,
            "artifact_role": role,
            "selected_k": 3,
            "source_decisions": entries,
            "source_decision_set_fingerprint": _fingerprint_from_value(
                {
                    "schema_id": "prpg-g3-report-source-decision-set-v1",
                    "source_decisions": entries,
                }
            ),
        }
        reports[path] = {
            "payload": pair_module.scientific_report_payload(
                document_type=document_type,
                role=cast(Any, role),
                task_result_fingerprints={
                    task_id: task_result_fingerprints[task_id]
                    for task_id in report_tasks
                },
                checkpoint_fingerprints={
                    task_id: checkpoint_fingerprints[task_id]
                    for task_id in report_tasks
                },
                planned_look_fingerprints={
                    family_id: planned_look_fingerprints[family_id]
                    for family_id in report_families
                },
                metrics=metrics,
            )
        }
    return (
        SimpleNamespace(role=role, selected_k=3, reports=reports),
        task_result_fingerprints,
        checkpoint_fingerprints,
        planned_look_fingerprints,
        decision_documents,
    )


def _verify_closed_report_fixture(
    fixture: tuple[
        SimpleNamespace,
        dict[str, str],
        dict[str, str],
        dict[str, tuple[str, ...]],
        dict[str, dict[str, Any]],
    ],
) -> None:
    artifact, tasks, checkpoints, looks, decisions = fixture
    pair_module._verify_closed_reports_from_durable_decisions(
        cast(Any, artifact),
        task_result_fingerprints=tasks,
        checkpoint_fingerprints=checkpoints,
        planned_look_fingerprints=looks,
        decision_documents=decisions,
    )


@pytest.mark.parametrize("role", ("design", "production"))
def test_durable_closed_reports_reverify_every_registered_decision(role: str) -> None:
    fixture = _closed_report_verification_fixture(role=role)

    _verify_closed_report_fixture(fixture)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("payload", "payload is invalid"),
        ("task-receipt", "receipt bindings changed"),
        ("checkpoint-receipt", "receipt bindings changed"),
        ("look-receipt", "receipt bindings changed"),
        ("metrics", "metric schema is invalid"),
        ("metric-fields", "metric schema is invalid"),
        ("metric-identity", "metric identity is invalid"),
        ("entry-count", "source decisions are invalid"),
        ("entry-shape", "decision is missing"),
        ("decision-missing", "decision is missing"),
        ("entry-lineage", "decision lineage changed"),
        ("entry-view-fingerprint", "decision lineage changed"),
        ("decision-set", "decision set changed"),
    ),
)
def test_durable_closed_reports_fail_closed_on_lineage_tamper(
    tamper: str,
    message: str,
) -> None:
    artifact, tasks, checkpoints, looks, decisions = (
        _closed_report_verification_fixture()
    )
    artifact.reports = pair_module._json_value(artifact.reports, "reports")
    if tamper == "look-receipt":
        first_report = next(
            report
            for report in artifact.reports.values()
            if report["payload"]["planned_look_fingerprints"]
        )
    else:
        first_report = next(iter(artifact.reports.values()))
    payload = first_report["payload"]
    report_tasks = tuple(payload["task_result_fingerprints"])
    report_looks = tuple(payload["planned_look_fingerprints"])
    metrics = payload["metrics"]
    entries = metrics["source_decisions"]
    first_task = report_tasks[0]

    if tamper == "payload":
        first_report["payload"] = []
    elif tamper == "task-receipt":
        payload["task_result_fingerprints"][first_task] = "0" * 64
    elif tamper == "checkpoint-receipt":
        payload["checkpoint_fingerprints"][first_task] = "0" * 64
    elif tamper == "look-receipt":
        assert report_looks
        payload["planned_look_fingerprints"][report_looks[0]] = ["0" * 64]
    elif tamper == "metrics":
        payload["metrics"] = []
    elif tamper == "metric-fields":
        metrics.pop("document_type")
    elif tamper == "metric-identity":
        metrics["selected_k"] = 4
    elif tamper == "entry-count":
        entries.pop()
    elif tamper == "entry-shape":
        entries[0] = None
    elif tamper == "decision-missing":
        decisions.pop(first_task)
    elif tamper == "entry-lineage":
        entries[0]["passed"] = False
    elif tamper == "entry-view-fingerprint":
        entries[0]["source_view_fingerprint"] = "not-a-fingerprint"
    elif tamper == "decision-set":
        metrics["source_decision_set_fingerprint"] = "0" * 64
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(f"unsupported tamper case: {tamper}")

    with pytest.raises(IntegrityError, match=message):
        _verify_closed_report_fixture((artifact, tasks, checkpoints, looks, decisions))


def test_pair_authority_types_are_not_publicly_constructible() -> None:
    with pytest.raises(TypeError, match="sealed first-25 prefix"):
        CanonicalScientificModelArtifactDraft()
    with pytest.raises(TypeError, match="created only by its store"):
        LoadedScientificArtifactPair()
    forged = object.__new__(CanonicalScientificModelArtifactDraft)
    with pytest.raises(ModelError, match="producer authority"):
        pair_module._verify_canonical_draft(
            forged,
            expected_role="design",
        )


def test_durable_pair_receipt_reconstructs_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, identity, kwargs = _durable_receipt_fixture(tmp_path, monkeypatch)
    fingerprint = _write_pair_identity(store, identity)

    reference = pair_module.verify_durable_scientific_artifact_pair_receipt(
        fingerprint=fingerprint,
        **kwargs,
    )

    assert reference.fingerprint == fingerprint


def test_g3_evidence_artifact_boundary_slices_only_pair_maps_to_first_25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real durable-pair verifier through the G3 boundary."""

    import prpg.model.g3_evidence_store as evidence_module

    store, identity, kwargs = _durable_receipt_fixture(tmp_path, monkeypatch)
    fingerprint = _write_pair_identity(store, identity)
    first_25 = G3_GATE_ORDER[:-1]
    task_fingerprints = dict(kwargs["task_result_fingerprints"])
    checkpoint_fingerprints = dict(kwargs["checkpoint_fingerprints"])
    task_fingerprints["artifact_integrity"] = _fingerprint("task:artifact_integrity")
    checkpoint_fingerprints["artifact_integrity"] = _fingerprint(
        "checkpoint:artifact_integrity"
    )
    decision_documents = cast(
        dict[str, dict[str, Any]],
        kwargs["decision_documents"],
    )
    checkpoints = {
        task_id: SimpleNamespace(
            state={
                "decision": decision_documents.get(
                    task_id,
                    {"evidence": {"task_id": task_id}},
                )
            }
        )
        for task_id in G3_GATE_ORDER
    }
    design = cast(Any, kwargs["design"])
    production = cast(Any, kwargs["production"])
    binding = cast(ScientificArtifactBinding, kwargs["binding"])
    provenance = cast(ScientificArtifactProvenance, kwargs["provenance"])
    bridge_parent = cast(str, kwargs["actual_bridge_parent_view_fingerprint"])
    monkeypatch.setattr(
        evidence_module,
        "verify_scientific_artifact_run_evidence",
        lambda *_args, **_kwargs: None,
    )

    observed = evidence_module._verify_artifact_bindings(
        bundle=SimpleNamespace(
            config_fingerprint=binding.config_fingerprint,
            processed_data_fingerprint=binding.processed_data_fingerprint,
            calibration_input_fingerprint=binding.calibration_input_fingerprint,
            design_artifact_fingerprint=design.reference.fingerprint,
            production_artifact_fingerprint=production.reference.fingerprint,
        ),
        model_store=cast(ModelArtifactStore, kwargs["model_store"]),
        design=design,
        production=production,
        run_id=cast(str, kwargs["run_id"]),
        completion=SimpleNamespace(preflight_fingerprint=binding.preflight_fingerprint),
        source_code_fingerprint=provenance.source_code_fingerprint,
        dependency_lock_fingerprint=provenance.dependency_lock_fingerprint,
        task_result_fingerprints=task_fingerprints,
        checkpoint_fingerprints=checkpoint_fingerprints,
        planned_look_fingerprints=cast(
            dict[str, tuple[str, ...]],
            kwargs["planned_look_fingerprints"],
        ),
        artifact_integrity_evidence={
            "design_artifact_fingerprint": design.reference.fingerprint,
            "production_artifact_fingerprint": production.reference.fingerprint,
            "design_parameter_set_fingerprint": design.parameter_set_fingerprint,
            "production_parameter_set_fingerprint": (
                production.parameter_set_fingerprint
            ),
            "selected_k": design.selected_k,
            "binding": binding.as_dict(),
            "provenance": provenance.as_dict(),
            "typed_schema_verified": True,
            "pair_receipt_fingerprint": fingerprint,
            "design_role_source_fingerprint": kwargs["design_role_source_fingerprint"],
            "production_role_source_fingerprint": kwargs[
                "production_role_source_fingerprint"
            ],
            "actual_bridge_parent_view_fingerprint": bridge_parent,
        },
        checkpoints=checkpoints,
    )

    assert observed == fingerprint
    assert set(task_fingerprints) == set(G3_GATE_ORDER)
    assert set(checkpoint_fingerprints) == set(G3_GATE_ORDER)
    assert set(identity["task_result_fingerprints"]) == set(first_25)


@pytest.mark.parametrize(
    "tamper",
    (
        "candidate-policy-fingerprint",
        "report-pair-fingerprint",
        "extra-top-level-field",
        "extra-artifact-field",
        "swapped-role-policy-set",
    ),
)
def test_durable_pair_receipt_rejects_readdressed_identity_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    store, identity, kwargs = _durable_receipt_fixture(tmp_path, monkeypatch)
    changed = cast(dict[str, Any], json.loads(json.dumps(identity)))
    artifacts = cast(dict[str, dict[str, Any]], changed["artifacts"])
    policy_sets = cast(dict[str, dict[str, Any]], changed["candidate_policy_sets"])
    if tamper == "candidate-policy-fingerprint":
        artifacts["design"]["candidate_policy_set_fingerprint"] = "0" * 64
    elif tamper == "report-pair-fingerprint":
        changed["report_pair_fingerprint"] = "0" * 64
    elif tamper == "extra-top-level-field":
        changed["unregistered_identity_field"] = False
    elif tamper == "extra-artifact-field":
        artifacts["production"]["unregistered_artifact_field"] = False
    elif tamper == "swapped-role-policy-set":
        design_policy = policy_sets["design"]
        production_policy = policy_sets["production"]
        policy_sets["design"] = production_policy
        policy_sets["production"] = design_policy
        design_policy_fingerprint = artifacts["design"][
            "candidate_policy_set_fingerprint"
        ]
        artifacts["design"]["candidate_policy_set_fingerprint"] = artifacts[
            "production"
        ]["candidate_policy_set_fingerprint"]
        artifacts["production"]["candidate_policy_set_fingerprint"] = (
            design_policy_fingerprint
        )
    else:  # pragma: no cover - the parametrization is closed above
        raise AssertionError(f"unsupported tamper case: {tamper}")
    fingerprint = _write_pair_identity(store, changed)

    with pytest.raises(IntegrityError, match="reconstructed lineage"):
        pair_module.verify_durable_scientific_artifact_pair_receipt(
            fingerprint=fingerprint,
            **kwargs,
        )


def test_pair_receipt_rejects_extra_file_symlink_hardlink_and_tamper(
    tmp_path: Path,
) -> None:
    extra_store, extra_fingerprint, extra_receipt = _stored_receipt(tmp_path / "extra")
    (extra_receipt.parent / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IntegrityError, match="allowlist"):
        extra_store.verify(extra_fingerprint)

    link_store, link_fingerprint, link_receipt = _stored_receipt(tmp_path / "link")
    original = link_receipt.with_name("original.json")
    link_receipt.rename(original)
    link_receipt.symlink_to(original.name)
    with pytest.raises(IntegrityError, match="allowlist|unsafe|opened"):
        link_store.verify(link_fingerprint)

    hard_store, hard_fingerprint, hard_receipt = _stored_receipt(tmp_path / "hard")
    os.link(hard_receipt, tmp_path / "hard-link-copy.json")
    with pytest.raises(IntegrityError, match="single-link"):
        hard_store.verify(hard_fingerprint)

    json_store, json_fingerprint, json_receipt = _stored_receipt(tmp_path / "json")
    parsed = json.loads(json_receipt.read_bytes())
    json_receipt.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    with pytest.raises(IntegrityError, match="canonical JSON"):
        json_store.verify(json_fingerprint)

    hash_store, hash_fingerprint, hash_receipt = _stored_receipt(tmp_path / "hash")
    parsed = json.loads(hash_receipt.read_bytes())
    parsed["identity"]["test_identity"] = "tampered"
    hash_receipt.write_bytes(_canonical(parsed))
    with pytest.raises(IntegrityError, match="identity"):
        hash_store.verify(hash_fingerprint)


def test_pair_receipt_rejects_symlink_root_and_descriptor_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_root = tmp_path / "linked-artifacts"
    linked_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symlink"):
        ScientificArtifactPairStore(linked_root)._prepare_root()

    store, fingerprint, receipt = _stored_receipt(tmp_path / "race")
    original_read = os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if not swapped:
            swapped = True
            content = receipt.read_bytes()
            receipt.rename(receipt.with_name("displaced-receipt.json"))
            receipt.write_bytes(content)
        return chunk

    monkeypatch.setattr(os, "read", swapping_read)
    with pytest.raises(IntegrityError, match="path changed"):
        store.verify(fingerprint)


def test_logical_pair_receipt_is_last_written_and_retry_reuses_exact_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    pair_store = ScientificArtifactPairStore(artifact_root)
    model_store = ModelArtifactStore(artifact_root)
    prefix = object()
    binding = SimpleNamespace(identity="binding")
    provenance = SimpleNamespace(identity="provenance")
    policy = SimpleNamespace(identity="policy")

    def draft(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            role=role,
            _prefix=prefix,
            run_id="a" * 64,
            selected_k=2,
            prefix_fingerprint="b" * 64,
            actual_bridge_parent_view_fingerprint="c" * 64,
            role_source_fingerprint=("d" if role == "design" else "e") * 64,
            report_set_fingerprint=("f" if role == "design" else "1") * 64,
            candidate_policy_set=policy,
            draft=SimpleNamespace(
                role=role,
                binding=binding,
                provenance=provenance,
            ),
        )

    design = draft("design")
    production = draft("production")
    loaded = {
        "design": SimpleNamespace(reference=SimpleNamespace(fingerprint="2" * 64)),
        "production": SimpleNamespace(reference=SimpleNamespace(fingerprint="3" * 64)),
    }
    calls: list[str] = []
    fail_second = True

    def publish(_store: object, supplied: object) -> object:
        nonlocal fail_second
        role = cast(Any, supplied).role
        calls.append(role)
        if role == "production" and fail_second:
            fail_second = False
            raise RuntimeError("simulated crash before production publication")
        return loaded[role]

    identity = {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "logical_pair": "exact-retry-fixture",
    }
    loaded_calls: list[bool] = []

    monkeypatch.setattr(pair_module, "_verify_canonical_draft", lambda *a, **k: None)
    monkeypatch.setattr(pair_module, "_verify_loaded_against_draft", lambda *a: None)
    monkeypatch.setattr(pair_module, "publish_scientific_model_artifact", publish)
    monkeypatch.setattr(pair_module, "_pair_identity", lambda **_kwargs: identity)

    def fake_load(*_args: object, reused: bool = True, **_kwargs: object) -> object:
        loaded_calls.append(reused)
        return object()

    monkeypatch.setattr(pair_store, "load", fake_load)

    with pytest.raises(RuntimeError, match="simulated crash"):
        pair_store.publish(
            model_store=model_store,
            design=cast(Any, design),
            production=cast(Any, production),
        )
    assert not pair_store.store_root.exists()

    pair_store.publish(
        model_store=model_store,
        design=cast(Any, design),
        production=cast(Any, production),
    )
    receipt_fingerprint = pair_module._sha256_bytes(_canonical(identity))
    first = pair_store.verify(receipt_fingerprint, reused=False)
    first_bytes = (first.path / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT).read_bytes()

    pair_store.publish(
        model_store=model_store,
        design=cast(Any, design),
        production=cast(Any, production),
    )
    second = pair_store.verify(receipt_fingerprint)
    assert (second.path / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT).read_bytes() == first_bytes
    assert calls == [
        "design",
        "production",
        "design",
        "production",
        "design",
        "production",
    ]
    assert loaded_calls == [False, True]


def test_pair_publish_rejects_cross_prefix_and_cross_run_before_artifact_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_store = ScientificArtifactPairStore(tmp_path / "artifacts")
    model_store = ModelArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(pair_module, "_verify_canonical_draft", lambda *a, **k: None)
    published = False

    def should_not_publish(*_args: object, **_kwargs: object) -> object:
        nonlocal published
        published = True
        return object()

    monkeypatch.setattr(
        pair_module,
        "publish_scientific_model_artifact",
        should_not_publish,
    )
    common = {
        "selected_k": 2,
        "prefix_fingerprint": "a" * 64,
        "actual_bridge_parent_view_fingerprint": "b" * 64,
        "candidate_policy_set": object(),
        "draft": SimpleNamespace(binding=object(), provenance=object()),
    }
    design = SimpleNamespace(role="design", _prefix=object(), run_id="c" * 64, **common)
    production = SimpleNamespace(
        role="production", _prefix=object(), run_id="d" * 64, **common
    )
    with pytest.raises(ModelError, match="different first-25 prefixes"):
        pair_store.publish(
            model_store=model_store,
            design=cast(Any, design),
            production=cast(Any, production),
        )
    assert not published


def test_closed_report_drafts_bind_real_adequacy_content_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views, holm = _real_adequacy_views(monkeypatch)
    sources = {view.task_id: view for view in views} | {holm.task_id: holm}
    task_ids = tuple(sources)
    prefix = SimpleNamespace(
        selected_n_states=2,
        _handlers=SimpleNamespace(as_mapping=lambda: sources),
        task_result_fingerprints={
            task_id: _fingerprint(f"task:{task_id}") for task_id in task_ids
        },
        checkpoint_fingerprints={
            task_id: _fingerprint(f"checkpoint:{task_id}") for task_id in task_ids
        },
        planned_look_fingerprints={},
    )

    monkeypatch.setattr(
        pair_module,
        "required_scientific_documents",
        lambda role: {
            "reports/latent-correctness.json": "latent_correctness",
            "reports/adequacy-holm.json": "adequacy_holm",
        }
        if role == "design"
        else {},
    )

    def passed_decision(task_id: str, source: object) -> SimpleNamespace:
        assert sources[task_id] is source
        return SimpleNamespace(
            passed=True,
            evidence={"task_id": task_id},
            summary={"status": "passed"},
        )

    monkeypatch.setattr(assembly_module, "derive_g3_gate_decision", passed_decision)
    reports = pair_module._closed_reports(
        prefix,
        cast(Any, SimpleNamespace(role="design")),
    )

    latent_entries = reports["reports/latent-correctness.json"].payload["metrics"][
        "source_decisions"
    ]
    assert [entry["source_view_fingerprint"] for entry in latent_entries] == [
        sources["design_latent_correctness"].content_fingerprint
    ]
    holm_entries = reports["reports/adequacy-holm.json"].payload["metrics"][
        "source_decisions"
    ]
    expected_holm_tasks = pair_module.scientific_report_task_ids(
        "adequacy_holm",
        "design",
    )
    assert [entry["source_view_fingerprint"] for entry in holm_entries] == [
        sources[task_id].content_fingerprint for task_id in expected_holm_tasks
    ]


def test_closed_report_source_fingerprint_rejects_missing_and_substituted_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views, _holm = _real_adequacy_views(monkeypatch)
    source = views[0]
    task_id = source.task_id
    assert (
        pair_module._closed_report_source_fingerprint(task_id, source)
        == source.content_fingerprint
    )

    field_only_substitute = SimpleNamespace(
        task_id=task_id,
        content_fingerprint=source.content_fingerprint,
    )
    with pytest.raises(ModelError, match="sealed task-source authority"):
        pair_module._closed_report_source_fingerprint(task_id, field_only_substitute)

    changed_fingerprint = replace(
        source,
        content_fingerprint=_fingerprint("substituted-adequacy-content"),
    )
    with pytest.raises(ModelError, match="sealed task-source authority"):
        pair_module._closed_report_source_fingerprint(task_id, changed_fingerprint)

    with pytest.raises(ModelError, match="different task"):
        pair_module._closed_report_source_fingerprint(
            "production_latent_correctness",
            source,
        )


def test_pair_store_load_reconstructs_typed_authority_from_live_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    store = ScientificArtifactPairStore(artifact_root)
    model_store = ModelArtifactStore(artifact_root)
    prefix = object()
    binding = object()
    provenance = object()
    report_pair_fingerprint = _fingerprint("loaded-report-pair")
    identity = {
        "artifacts": {
            "design": {"artifact_fingerprint": _fingerprint("loaded-design")},
            "production": {"artifact_fingerprint": _fingerprint("loaded-production")},
        },
        "report_pair_fingerprint": report_pair_fingerprint,
    }
    reference = pair_module.ScientificArtifactPairReference(
        _fingerprint_from_value(identity),
        artifact_root,
        {"identity": identity},
        False,
    )
    loaded = {
        "design": SimpleNamespace(role="design"),
        "production": SimpleNamespace(role="production"),
    }

    def draft(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            role=role,
            selected_k=3,
            run_id=_fingerprint("loaded-run"),
            prefix_fingerprint=_fingerprint("loaded-prefix"),
            actual_bridge_parent_view_fingerprint=_fingerprint("loaded-bridge"),
            role_source_fingerprint=_fingerprint(f"loaded-source:{role}"),
            draft=SimpleNamespace(binding=binding, provenance=provenance),
        )

    drafts = {role: draft(role) for role in ("design", "production")}
    fingerprints_to_roles = {
        identity["artifacts"][role]["artifact_fingerprint"]: role
        for role in ("design", "production")
    }
    monkeypatch.setattr(pair_module, "_checked_prefix", lambda supplied: supplied)
    monkeypatch.setattr(store, "verify", lambda *_args, **_kwargs: reference)
    monkeypatch.setattr(
        pair_module,
        "load_scientific_model_artifact",
        lambda _store, fingerprint: loaded[fingerprints_to_roles[fingerprint]],
    )
    monkeypatch.setattr(
        pair_module,
        "build_canonical_scientific_model_artifact_draft",
        lambda _prefix, *, role: drafts[role],
    )
    monkeypatch.setattr(
        pair_module,
        "_verify_loaded_against_draft",
        lambda *_args: None,
    )
    monkeypatch.setattr(pair_module, "_pair_identity", lambda **_kwargs: identity)

    result = store.load(
        reference.fingerprint,
        model_store=model_store,
        prefix=prefix,
        reused=False,
    )

    assert result.reference is reference
    assert result.design is loaded["design"]
    assert result.production is loaded["production"]
    assert result.selected_k == 3
    assert result.run_id == drafts["design"].run_id
    assert result.binding is binding
    assert result.provenance is provenance
    assert result.report_pair_fingerprint == report_pair_fingerprint
    assert result._prefix is prefix
    assert result._seal is pair_module._PAIR_CAPABILITY
    assert pair_module._MINTED_PAIRS[id(result)] is result
    assert not result.generation_authorized


@pytest.mark.parametrize(
    ("artifacts", "message"),
    (
        (None, "artifact identities are missing"),
        ({}, "role identities are invalid"),
        ({"design": {}, "production": None}, "role identities are invalid"),
    ),
)
def test_pair_store_load_rejects_incomplete_role_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts: object,
    message: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    store = ScientificArtifactPairStore(artifact_root)
    model_store = ModelArtifactStore(artifact_root)
    identity = {"artifacts": artifacts}
    monkeypatch.setattr(pair_module, "_checked_prefix", lambda supplied: supplied)
    monkeypatch.setattr(
        store,
        "verify",
        lambda *_args, **_kwargs: pair_module.ScientificArtifactPairReference(
            _fingerprint("incomplete-pair"),
            artifact_root,
            {"identity": identity},
            True,
        ),
    )

    with pytest.raises(IntegrityError, match=message):
        store.load(
            _fingerprint("incomplete-pair"),
            model_store=model_store,
            prefix=object(),
        )


def test_canonical_draft_builder_mints_only_from_closed_prefix_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = object()
    provenance = object()
    prefix = SimpleNamespace(
        binding=binding,
        provenance=provenance,
        selected_n_states=3,
        run_id=_fingerprint("draft-run"),
        prefix_fingerprint=_fingerprint("draft-prefix"),
        _handlers=SimpleNamespace(
            actual_artifact_bridge=SimpleNamespace(
                view_fingerprint=_fingerprint("draft-bridge")
            )
        ),
    )
    inputs = SimpleNamespace(
        fit=object(),
        pools=object(),
        monthly_continuation=np.asarray([False, True]),
        daily_continuation=np.asarray([False, True, True]),
        monthly_reference=object(),
        daily_reference=object(),
        monthly_kernel=object(),
        daily_kernel=object(),
        role_source_fingerprint=_fingerprint("draft-role-source"),
    )
    policy = SimpleNamespace(name="policy")
    reports = {
        "reports/model.json": ScientificReportPayload(
            "model_fit",
            {"observations": 193},
            {"status": "passed"},
        )
    }
    scientific_draft = SimpleNamespace(binding=binding, provenance=provenance)
    observed: dict[str, object] = {}
    monkeypatch.setattr(pair_module, "_checked_prefix", lambda value: value)
    monkeypatch.setattr(pair_module, "_role_inputs", lambda *_args: inputs)

    def candidate_policy(**kwargs: object) -> object:
        observed["policy_kwargs"] = kwargs
        return policy

    monkeypatch.setattr(
        pair_module,
        "candidate_policy_set_from_kernel_results",
        candidate_policy,
    )
    monkeypatch.setattr(pair_module, "_closed_reports", lambda *_args: reports)

    def draft(**kwargs: object) -> object:
        observed["draft_kwargs"] = kwargs
        return scientific_draft

    monkeypatch.setattr(
        pair_module,
        "draft_scientific_model_artifact_from_phase3_results",
        draft,
    )

    result = pair_module.build_canonical_scientific_model_artifact_draft(
        prefix,
        role="design",
    )

    assert result.role == "design"
    assert result.selected_k == 3
    assert result.run_id == prefix.run_id
    assert result.prefix_fingerprint == prefix.prefix_fingerprint
    assert result.actual_bridge_parent_view_fingerprint == (
        prefix._handlers.actual_artifact_bridge.view_fingerprint
    )
    assert result.role_source_fingerprint == inputs.role_source_fingerprint
    assert result.candidate_policy_set is policy
    assert result.draft is scientific_draft
    assert result._prefix is prefix
    assert result._seal is pair_module._DRAFT_CAPABILITY
    assert pair_module._MINTED_DRAFTS[id(result)] is result
    assert observed["policy_kwargs"] == {
        "monthly": inputs.monthly_kernel,
        "daily": inputs.daily_kernel,
    }
    assert cast(dict[str, object], observed["draft_kwargs"])["reports"] is reports


def test_pair_identity_recomputes_live_drafts_and_detects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = object()
    binding = object()
    provenance = object()

    def draft(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            role=role,
            selected_k=3,
            run_id=_fingerprint("identity-run"),
            prefix_fingerprint=_fingerprint("identity-prefix"),
            actual_bridge_parent_view_fingerprint=_fingerprint("identity-bridge"),
            role_source_fingerprint=_fingerprint(f"identity-source:{role}"),
            draft=SimpleNamespace(binding=binding, provenance=provenance),
        )

    drafts = {role: draft(role) for role in ("design", "production")}
    expected = {
        "report_pair_fingerprint": _fingerprint("identity-report-pair"),
        "identity": "pair",
    }
    reference = pair_module.ScientificArtifactPairReference(
        _fingerprint_from_value(expected),
        Path("/pair"),
        {"identity": expected},
        False,
    )
    pair = object.__new__(LoadedScientificArtifactPair)
    for name, value in (
        ("reference", reference),
        ("design", SimpleNamespace(role="design")),
        ("production", SimpleNamespace(role="production")),
        ("selected_k", 3),
        ("run_id", drafts["design"].run_id),
        ("binding", binding),
        ("provenance", provenance),
        ("prefix_fingerprint", drafts["design"].prefix_fingerprint),
        (
            "actual_bridge_parent_view_fingerprint",
            drafts["design"].actual_bridge_parent_view_fingerprint,
        ),
        ("design_role_source_fingerprint", drafts["design"].role_source_fingerprint),
        (
            "production_role_source_fingerprint",
            drafts["production"].role_source_fingerprint,
        ),
        ("report_pair_fingerprint", expected["report_pair_fingerprint"]),
        ("_prefix", prefix),
        ("_seal", pair_module._PAIR_CAPABILITY),
    ):
        object.__setattr__(pair, name, value)
    pair_module._MINTED_PAIRS[id(pair)] = pair
    monkeypatch.setattr(pair_module, "_checked_prefix", lambda value: value)
    monkeypatch.setattr(
        pair_module,
        "build_canonical_scientific_model_artifact_draft",
        lambda _prefix, *, role: drafts[role],
    )
    monkeypatch.setattr(
        pair_module,
        "_verify_loaded_against_draft",
        lambda *_args: None,
    )
    monkeypatch.setattr(pair_module, "_pair_identity", lambda **_kwargs: expected)

    assert pair_module.scientific_artifact_pair_identity(pair) == expected
    assert pair_module.is_verified_scientific_artifact_pair(pair)
    assert not pair_module.is_verified_scientific_artifact_pair(object())

    object.__setattr__(pair, "selected_k", 4)
    with pytest.raises(ModelError, match="in-memory identity changed"):
        pair_module.scientific_artifact_pair_identity(pair)
    assert not pair_module.is_verified_scientific_artifact_pair(pair)


def test_loaded_artifact_verifier_checks_arrays_reports_and_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = SimpleNamespace(as_dict=lambda: {"policy": "candidate"})
    report = ScientificReportPayload(
        "model_fit",
        {"observations": 2},
        {"status": "passed"},
    )
    draft = SimpleNamespace(
        role="design",
        selected_k=2,
        binding=object(),
        provenance=object(),
        generation_policy=policy,
        arrays={"values": np.asarray([1.0, 2.0])},
        reports={"reports/model.json": report},
    )
    expected = SimpleNamespace(role="design", selected_k=2, draft=draft)
    loaded = SimpleNamespace(
        role="design",
        selected_k=2,
        binding=draft.binding,
        provenance=draft.provenance,
        generation_policy=policy,
        arrays={"values": np.asarray([1.0, 2.0])},
        reports={
            "reports/model.json": {
                "document_type": report.document_type,
                "counts": dict(report.counts),
                "payload": report.payload,
                "generation_policy": policy.as_dict(),
            }
        },
    )
    monkeypatch.setattr(
        pair_module,
        "is_verified_scientific_model_artifact",
        lambda _value: True,
    )

    pair_module._verify_loaded_against_draft(loaded, expected)
    assert pair_module._loaded_report_set_fingerprint(loaded) == (
        pair_module._report_set_fingerprint(draft.reports)
    )
    assert pair_module._drafts_equal(draft, draft)

    changed_array = SimpleNamespace(**vars(loaded))
    changed_array.arrays = {"values": np.asarray([1.0, 3.0])}
    with pytest.raises(IntegrityError, match="differs from canonical draft"):
        pair_module._verify_loaded_against_draft(changed_array, expected)
    changed_report = SimpleNamespace(**vars(loaded))
    changed_report.reports = pair_module._json_value(loaded.reports, "reports")
    changed_report.reports["reports/model.json"]["payload"] = {"status": "failed"}
    with pytest.raises(IntegrityError, match="report differs"):
        pair_module._verify_loaded_against_draft(changed_report, expected)


def test_pair_scalar_helpers_normalize_supported_values_and_reject_invalid() -> None:
    continuation = pair_module._continuation_mask(np.asarray([True, False]))
    assert continuation.tolist() == [False, True, False]
    assert not continuation.flags.writeable
    with pytest.raises(ModelError, match="continuation links are invalid"):
        pair_module._continuation_mask(np.asarray([1, 0]))
    assert pair_module._role("design") == "design"
    with pytest.raises(ModelError, match="role is invalid"):
        pair_module._role("other")

    assert pair_module._json_value(np.int64(2), "value") == 2
    assert pair_module._json_value(np.asarray([1, 2]), "value") == [1, 2]
    with pytest.raises(ModelError, match="non-finite"):
        pair_module._json_value(float("inf"), "value")
    with pytest.raises(ModelError, match="invalid mapping key"):
        pair_module._json_value({"": 1}, "value")
    with pytest.raises(ModelError, match="unsupported value"):
        pair_module._json_value(object(), "value")
    with pytest.raises(IntegrityError, match="fingerprint is invalid"):
        pair_module._required_sha("bad", "fingerprint", IntegrityError)


class _RoleFitStub:
    def __init__(self, decoded_states: np.ndarray[Any, Any], fingerprint: str) -> None:
        self.decoded_states = decoded_states
        self.n_states = 2
        self.fingerprint = fingerprint


def _role_input_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, _RoleFitStub]:
    from prpg.model import g3_boundary_views as boundary_module

    fit_fingerprint = _fingerprint("role-input-fit")
    monthly_states = np.asarray([0, 1, 0], dtype=np.int64)
    daily_states = np.asarray([0, 1, 1, 0], dtype=np.int64)
    fit = _RoleFitStub(monthly_states, fit_fingerprint)

    def block(states: np.ndarray[Any, Any], frequency: str) -> SimpleNamespace:
        links = np.ones(len(states) - 1, dtype=np.bool_)
        return SimpleNamespace(
            model=fit,
            historical_source_states=states,
            source_continuation_links=links,
            core_identity=SimpleNamespace(
                parent_model_fingerprint=fit_fingerprint,
                core_fingerprint=_fingerprint(f"block-core:{frequency}"),
            ),
            execution=SimpleNamespace(
                geometry=SimpleNamespace(target_segment_lengths=(len(states),))
            ),
        )

    def kernel(states: np.ndarray[Any, Any], frequency: str) -> SimpleNamespace:
        links = np.ones(len(states) - 1, dtype=np.bool_)
        return SimpleNamespace(
            model=fit,
            historical_states=states,
            continuation_links=links,
            historical_returns=np.arange(len(states) * 3, dtype=np.float64).reshape(
                len(states), 3
            ),
            identity=SimpleNamespace(
                artifact="design",
                frequency=frequency,
                parent_model_fingerprint=fit_fingerprint,
                execution_fingerprint=_fingerprint(f"kernel:{frequency}"),
            ),
        )

    monthly = block(monthly_states, "monthly")
    daily = block(daily_states, "daily")
    monthly_kernel = kernel(monthly_states, "monthly")
    daily_kernel = kernel(daily_states, "daily")
    source = SimpleNamespace(
        selected_k=2,
        monthly=monthly,
        daily=daily,
        monthly_kernel=monthly_kernel,
        daily_kernel=daily_kernel,
    )
    view = SimpleNamespace(
        role="design",
        run_id=_fingerprint("role-input-run"),
        view_fingerprint=_fingerprint("role-input-view"),
        source=source,
    )
    prefix = SimpleNamespace(
        run_id=view.run_id,
        selected_n_states=2,
        _handlers=SimpleNamespace(
            as_mapping=lambda: {"design_block_calibration": view}
        ),
    )
    monkeypatch.setattr(
        boundary_module,
        "is_block_calibration_task_view",
        lambda value: value is view,
    )
    monkeypatch.setattr(pair_module, "GaussianHMMFit", _RoleFitStub)
    monkeypatch.setattr(
        pair_module,
        "registered_block_replay_source",
        lambda value: value,
    )
    monkeypatch.setattr(
        pair_module,
        "kernel_calibration_replay_source",
        lambda value: value,
    )
    monkeypatch.setattr(
        pair_module,
        "gaussian_hmm_fit_fingerprint",
        lambda value: value.fingerprint,
    )
    monkeypatch.setattr(
        pair_module,
        "summarize_state_path",
        lambda *_args, **_kwargs: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        pair_module,
        "fit_dependence_reference",
        lambda *_args, **_kwargs: SimpleNamespace(
            pc1_loadings=np.asarray([1.0, 0.0, 0.0])
        ),
    )
    monkeypatch.setattr(
        pair_module,
        "decoded_return_pools_fingerprint",
        lambda _value: _fingerprint("role-input-pools"),
    )
    monkeypatch.setattr(
        pair_module,
        "scientific_materialization_fingerprint",
        lambda **_kwargs: _fingerprint("role-input-materialization"),
    )
    monkeypatch.setattr(
        pair_module,
        "kernel_calibration_execution_identity",
        lambda _value: SimpleNamespace(verified=True),
    )
    return prefix, view, monthly_kernel, fit


def test_role_inputs_reconstruct_exact_block_kernel_fit_and_dependence_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, view, monthly_kernel, fit = _role_input_fixture(monkeypatch)

    result = pair_module._role_inputs(prefix, "design")

    assert result.role == "design"
    assert result.block_view is view
    assert result.fit is fit
    assert np.array_equal(result.pools.monthly_states, fit.decoded_states)
    assert result.monthly_kernel is monthly_kernel
    assert result.monthly_continuation.tolist() == [False, True, True]
    assert result.daily_continuation.tolist() == [False, True, True, True]
    assert result.role_source_fingerprint == _fingerprint("role-input-materialization")


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("view-role", "lacks exact block parent"),
        ("kernel-model", "kernel parents must retain fitted HMMs"),
        ("block-model", "block parents must retain fitted HMMs"),
        ("fit-lineage", "use different fitted models"),
        ("states", "block/kernel live sources disagree"),
        ("decoded", "monthly states differ from fitted path"),
    ),
)
def test_role_inputs_reject_cross_role_model_and_live_source_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    prefix, view, monthly_kernel, fit = _role_input_fixture(monkeypatch)
    if tamper == "view-role":
        view.role = "production"
    elif tamper == "kernel-model":
        monthly_kernel.model = object()
    elif tamper == "block-model":
        view.source.monthly.model = object()
    elif tamper == "fit-lineage":
        view.source.daily_kernel.identity.parent_model_fingerprint = "0" * 64
    elif tamper == "states":
        view.source.daily_kernel.historical_states = np.asarray([1, 1, 1, 1])
    elif tamper == "decoded":
        fit.decoded_states = np.asarray([1, 1, 1])
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(f"unsupported tamper: {tamper}")

    with pytest.raises(ModelError, match=message):
        pair_module._role_inputs(prefix, "design")
