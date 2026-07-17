from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import prpg.model.g3_exact_v2_finalization as module
from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactReference, ModelArtifactStore
from prpg.model.g3_assembly import DerivedG3GateDecision
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointBinding,
    CalibrationCheckpointReference,
)
from prpg.model.g3_evidence_store import G3EvidenceStore
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_science_handlers import CanonicalG3ArtifactPublicationPrefix
from prpg.model.g3_task_binding import (
    SCIENTIFIC_TASK_BINDING_SCHEMA_ID,
    SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION,
    SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT,
    ScientificTaskBinding,
)
from prpg.model.g3_task_publication import (
    SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION,
    SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION,
    ReverifiedScientificTaskPublication,
    ScientificTaskPublication,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    default_run_identity,
)
from prpg.model.scientific_artifact_pair import (
    LoadedScientificArtifactPair,
    ScientificArtifactPairStore,
)

_RUN_ID = "a" * 64
_K = 3
_FIRST_25 = G3_GATE_ORDER[:-1]
_SELECTION_POSITION = _FIRST_25.index("design_predictive_selection")


def test_exact_finalization_result_cannot_be_caller_constructed() -> None:
    with pytest.raises(TypeError, match="created only by its closed finalizer"):
        module.ExactV2G3FinalizationResult()


def _binding(
    task_id: str,
    position: int,
    *,
    schema_version: int = SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION,
) -> ScientificTaskBinding:
    return ScientificTaskBinding(
        schema_id=SCIENTIFIC_TASK_BINDING_SCHEMA_ID,
        schema_version=schema_version,
        run_id=_RUN_ID,
        run_identity=default_run_identity(
            config_fingerprint="b" * 64,
            processed_data_fingerprint="c" * 64,
            calibration_input_fingerprint="d" * 64,
            source_code_fingerprint="e" * 64,
            dependency_lock_fingerprint="f" * 64,
            workers=4,
        ),
        launch_preflight_fingerprint="1" * 64,
        persisted_preflight_receipt_fingerprint="2" * 64,
        compatible_preflight_fingerprint="3" * 64,
        task_id=task_id,
        selected_n_states=_K if position > _SELECTION_POSITION else None,
        dependency_checkpoint_fingerprints=(),
        owned_planned_look_fingerprints=(),
        scientific_component_fingerprints=(),
        materialization_fingerprints=(),
        rng_fingerprints=(),
        slot_registry_fingerprint=SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT,
        fingerprint=f"{position + 10:064x}",
    )


def _decision(task_id: str, *, passed: bool = True) -> DerivedG3GateDecision:
    decision = object.__new__(DerivedG3GateDecision)
    for name, value in (
        ("gate_id", task_id),
        ("passed", passed),
        ("evidence", MappingProxyType({"task_id": task_id})),
        ("summary", MappingProxyType({"status": "passed"})),
        ("_source", SimpleNamespace(task_id=task_id)),
        ("_seal", object()),
    ):
        object.__setattr__(decision, name, value)
    return decision


def _result(task_id: str, position: int, *, status: str = "passed") -> G3TaskResult:
    selected = _K if position >= _SELECTION_POSITION else None
    return G3TaskResult(
        run_id=_RUN_ID,
        task_id=task_id,
        status=cast(Any, status),
        payload=MappingProxyType({"selected_n_states": selected}),
        dependency_fingerprints=MappingProxyType({}),
        blocked_by=(),
        fingerprint=f"{position + 100:064x}",
        path=Path(f"/{task_id}.json"),
        reused=False,
    )


def _publication(
    task_id: str,
    position: int,
    *,
    schema_version: int = SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION,
    status: str = "passed",
    checkpoint: object | None = None,
) -> ScientificTaskPublication:
    selected = _K if position >= _SELECTION_POSITION else None
    supplied_checkpoint = checkpoint or SimpleNamespace(
        task_id=task_id,
        selected_n_states=selected,
        fingerprint=f"{position + 200:064x}",
    )
    return ScientificTaskPublication(
        binding=_binding(task_id, position, schema_version=schema_version),
        decision=_decision(task_id),
        checkpoint=cast(Any, supplied_checkpoint),
        task_result=_result(task_id, position, status=status),
        serialized_binding_sha256=f"{position + 300:064x}",
    )


def _publications() -> tuple[ScientificTaskPublication, ...]:
    return tuple(
        _publication(task_id, position) for position, task_id in enumerate(_FIRST_25)
    )


def _exact_publication_fixture(tmp_path: Path) -> SimpleNamespace:
    task_id = _FIRST_25[0]
    binding = _binding(task_id, 0)
    decision = _decision(task_id)
    source = decision._source
    serialized_binding_sha256 = "7" * 64
    checkpoint_fingerprint = "8" * 64
    expected_decision = {
        "evidence": dict(decision.evidence),
        "gate_id": decision.gate_id,
        "passed": decision.passed,
        "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
        "summary": dict(decision.summary),
    }
    result = G3TaskResult(
        run_id=_RUN_ID,
        task_id=task_id,
        status="passed",
        payload=MappingProxyType(
            {
                "binding_fingerprint": binding.fingerprint,
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "decision": expected_decision,
                "scientific_task_binding": binding.as_dict(),
                "serialized_binding_sha256": serialized_binding_sha256,
            }
        ),
        dependency_fingerprints=MappingProxyType({}),
        blocked_by=(),
        fingerprint="9" * 64,
        path=tmp_path / "task-result.json",
        reused=False,
    )
    checkpoint_state = MappingProxyType(
        {
            "binding_fingerprint": binding.fingerprint,
            "checkpoint_state_schema_version": (
                SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION
            ),
            "decision": expected_decision,
            "publication_schema_version": SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION,
            "scientific_task_binding": binding.as_dict(),
            "serialized_binding_sha256": serialized_binding_sha256,
        }
    )
    storage_reference = ModelArtifactReference(
        fingerprint="6" * 64,
        path=tmp_path / "checkpoint-artifact",
        manifest=MappingProxyType({"identity": "checkpoint"}),
        reused=False,
    )
    checkpoint = CalibrationCheckpointReference(
        fingerprint=checkpoint_fingerprint,
        path=tmp_path / "checkpoint",
        task_id=task_id,
        state_schema_id="prpg-test-checkpoint-state-v1",
        binding=CalibrationCheckpointBinding(
            run_id=_RUN_ID,
            config_fingerprint="b" * 64,
            processed_data_fingerprint="c" * 64,
            calibration_input_fingerprint="d" * 64,
            preflight_fingerprint="1" * 64,
        ),
        selected_n_states=None,
        dependency_fingerprints=(),
        arrays=MappingProxyType({"state": np.array([1.0, 2.0])}),
        state=checkpoint_state,
        storage_reference=storage_reference,
        reused=False,
    )
    publication = ScientificTaskPublication(
        binding=binding,
        decision=decision,
        checkpoint=checkpoint,
        task_result=result,
        serialized_binding_sha256=serialized_binding_sha256,
    )
    replay = ReverifiedScientificTaskPublication(
        binding=binding,
        checkpoint=checkpoint,
        task_result=result,
        serialized_binding_sha256=serialized_binding_sha256,
    )
    return SimpleNamespace(
        task_id=task_id,
        binding=binding,
        decision=decision,
        source=source,
        result=result,
        checkpoint=checkpoint,
        publication=publication,
        replay=replay,
    )


@pytest.mark.parametrize("kind", ("incomplete", "extra", "reordered"))
def test_exact_publication_sequence_rejects_nonexact_inputs(kind: str) -> None:
    supplied = list(_publications())
    if kind == "incomplete":
        supplied.pop()
    elif kind == "extra":
        supplied.append(_publication("artifact_integrity", len(_FIRST_25)))
    elif kind == "reordered":
        supplied[3], supplied[4] = supplied[4], supplied[3]
    else:  # pragma: no cover - closed parametrization.
        raise AssertionError(kind)

    error = ModelError if kind != "reordered" else IntegrityError
    with pytest.raises(error, match="exactly 25|reordered or substituted"):
        module._exact_publication_sequence(supplied)


def test_exact_publication_sequence_accepts_only_exact_wrapper_type() -> None:
    supplied = list(_publications())
    supplied[0] = cast(Any, SimpleNamespace(binding=supplied[0].binding))

    with pytest.raises(ModelError, match="only scientific task publications"):
        module._exact_publication_sequence(supplied)


def test_exact_publication_sequence_requires_a_sequence_and_sealed_bindings() -> None:
    with pytest.raises(ModelError, match="requires a publication sequence"):
        module._exact_publication_sequence(cast(Any, object()))

    supplied = list(_publications())
    supplied[0] = replace(
        supplied[0],
        binding=cast(Any, SimpleNamespace(task_id=_FIRST_25[0])),
    )
    with pytest.raises(ModelError, match="lacks its sealed task binding"):
        module._exact_publication_sequence(supplied)


def test_publication_verifier_rejects_legacy_failed_and_substituted_inputs() -> None:
    expected = _result(_FIRST_25[0], 0)
    common = {
        "expected_task_id": _FIRST_25[0],
        "run_store": cast(Any, object()),
        "checkpoint_store": cast(Any, object()),
        "live_preflight": cast(Any, object()),
        "expected_result": expected,
    }
    legacy = _publication(_FIRST_25[0], 0, schema_version=1)
    with pytest.raises(ModelError, match="rejects legacy"):
        module._verify_exact_v2_publication(legacy, **common)

    failed = _publication(_FIRST_25[0], 0, status="failed")
    with pytest.raises(IntegrityError, match="exact passing publication"):
        module._verify_exact_v2_publication(failed, **common)

    substituted = _publication(_FIRST_25[0], 0)
    different = replace(
        expected,
        payload=MappingProxyType(
            {"selected_n_states": None, "substituted_content": True}
        ),
    )
    with pytest.raises(IntegrityError, match="exact passing publication"):
        module._verify_exact_v2_publication(
            substituted,
            **{**common, "expected_result": different},
        )


def test_publication_verifier_recomputes_source_and_returns_exact_durable_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _exact_publication_fixture(tmp_path)
    calls: list[str] = []

    def verify_source(**kwargs: object) -> None:
        calls.append("source")
        assert kwargs["binding"] is fixture.binding
        assert kwargs["source"] is fixture.source

    monkeypatch.setattr(
        module,
        "verify_recomputed_scientific_task_source",
        verify_source,
    )
    monkeypatch.setattr(
        module,
        "derive_g3_gate_decision",
        lambda task_id, source: calls.append("decision") or fixture.decision,
    )
    monkeypatch.setattr(
        module,
        "reverify_published_scientific_task",
        lambda **kwargs: calls.append("replay") or fixture.replay,
    )

    observed = module._verify_exact_v2_publication(
        fixture.publication,
        expected_task_id=fixture.task_id,
        run_store=cast(Any, object()),
        checkpoint_store=cast(Any, object()),
        live_preflight=cast(Any, object()),
        expected_result=fixture.result,
    )

    assert observed is fixture.replay
    assert calls == ["source", "decision", "replay"]


def test_publication_verifier_rejects_internal_checkpoint_binding_tamper(
    tmp_path: Path,
) -> None:
    fixture = _exact_publication_fixture(tmp_path)
    tampered_state = {
        **dict(fixture.checkpoint.state),
        "binding_fingerprint": "0" * 64,
    }
    tampered_checkpoint = replace(
        fixture.checkpoint,
        state=MappingProxyType(tampered_state),
    )
    publication = replace(fixture.publication, checkpoint=tampered_checkpoint)

    with pytest.raises(IntegrityError, match="internally inconsistent"):
        module._verify_exact_v2_publication(
            publication,
            expected_task_id=fixture.task_id,
            run_store=cast(Any, object()),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            expected_result=fixture.result,
        )


def test_publication_verifier_rejects_live_decision_recomputation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _exact_publication_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_recomputed_scientific_task_source",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "derive_g3_gate_decision",
        lambda task_id, source: _decision(task_id, passed=False),
    )

    with pytest.raises(IntegrityError, match="differs from its live source"):
        module._verify_exact_v2_publication(
            fixture.publication,
            expected_task_id=fixture.task_id,
            run_store=cast(Any, object()),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            expected_result=fixture.result,
        )


def test_publication_verifier_rejects_changed_durable_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _exact_publication_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_recomputed_scientific_task_source",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "derive_g3_gate_decision",
        lambda task_id, source: fixture.decision,
    )
    monkeypatch.setattr(
        module,
        "reverify_published_scientific_task",
        lambda **kwargs: replace(
            fixture.replay,
            serialized_binding_sha256="0" * 64,
        ),
    )

    with pytest.raises(IntegrityError, match="differs from durable replay"):
        module._verify_exact_v2_publication(
            fixture.publication,
            expected_task_id=fixture.task_id,
            run_store=cast(Any, object()),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            expected_result=fixture.result,
        )


def test_selected_state_lineage_rejects_changed_postselection_value() -> None:
    supplied = list(_publications())
    position = _SELECTION_POSITION + 1
    publication = supplied[position]
    supplied[position] = ScientificTaskPublication(
        publication.binding,
        publication.decision,
        cast(
            Any,
            SimpleNamespace(
                task_id=publication.binding.task_id,
                selected_n_states=4,
                fingerprint=publication.checkpoint.fingerprint,
            ),
        ),
        publication.task_result,
        publication.serialized_binding_sha256,
    )

    with pytest.raises(IntegrityError, match="selected-state lineage"):
        module._verify_selected_state_lineage(tuple(supplied))


def test_selected_state_lineage_accepts_exact_frozen_k_and_rejects_invalid_k() -> None:
    supplied = _publications()

    assert module._verify_selected_state_lineage(supplied) == _K

    changed = list(supplied)
    selection = changed[_SELECTION_POSITION]
    changed[_SELECTION_POSITION] = replace(
        selection,
        checkpoint=cast(
            Any,
            SimpleNamespace(
                task_id=selection.binding.task_id,
                selected_n_states=6,
                fingerprint=selection.checkpoint.fingerprint,
            ),
        ),
    )
    with pytest.raises(IntegrityError, match="allowed state count"):
        module._verify_selected_state_lineage(tuple(changed))


def test_checkpoint_content_comparison_ignores_only_loader_reuse_metadata(
    tmp_path: Path,
) -> None:
    fixture = _exact_publication_fixture(tmp_path)
    checkpoint = fixture.checkpoint
    reloaded = replace(
        checkpoint,
        arrays=MappingProxyType(
            {
                name: np.array(values, copy=True)
                for name, values in checkpoint.arrays.items()
            }
        ),
        storage_reference=replace(checkpoint.storage_reference, reused=True),
        reused=True,
    )

    assert module._same_checkpoint_content(reloaded, checkpoint)

    changed_manifest = replace(
        reloaded,
        storage_reference=replace(
            reloaded.storage_reference,
            manifest=MappingProxyType({"identity": "changed"}),
        ),
    )
    assert not module._same_checkpoint_content(changed_manifest, checkpoint)

    changed_array = replace(
        reloaded,
        arrays=MappingProxyType({"state": np.array([1.0, 9.0])}),
    )
    assert not module._same_checkpoint_content(changed_array, checkpoint)


def test_reloaded_pair_requires_authority_and_exact_prefix_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = object.__new__(CanonicalG3ArtifactPublicationPrefix)
    for name, value in (
        ("run_id", _RUN_ID),
        ("prefix_fingerprint", "1" * 64),
        ("selected_n_states", _K),
        ("task_result_fingerprints", MappingProxyType({"task": "2" * 64})),
        ("checkpoint_fingerprints", MappingProxyType({"task": "3" * 64})),
        (
            "planned_look_fingerprints",
            MappingProxyType({"family": ("4" * 64,)}),
        ),
    ):
        object.__setattr__(prefix, name, value)
    artifact_pair = object.__new__(LoadedScientificArtifactPair)
    for name, value in (
        ("_prefix", prefix),
        ("run_id", _RUN_ID),
        ("prefix_fingerprint", "1" * 64),
        ("selected_k", _K),
    ):
        object.__setattr__(artifact_pair, name, value)
    identity = {
        "task_result_fingerprints": {"task": "2" * 64},
        "checkpoint_fingerprints": {"task": "3" * 64},
        "planned_look_fingerprints": {"family": ["4" * 64]},
    }
    authority = {"valid": True}
    monkeypatch.setattr(
        module,
        "is_verified_scientific_artifact_pair",
        lambda value: authority["valid"],
    )
    monkeypatch.setattr(
        module,
        "scientific_artifact_pair_identity",
        lambda value: identity,
    )

    module._verify_reloaded_pair(artifact_pair, prefix)

    identity["checkpoint_fingerprints"] = {"task": "0" * 64}
    with pytest.raises(IntegrityError, match="differs from exact-v2 prefix"):
        module._verify_reloaded_pair(artifact_pair, prefix)

    authority["valid"] = False
    with pytest.raises(IntegrityError, match="did not reload with authority"):
        module._verify_reloaded_pair(artifact_pair, prefix)


def test_first_25_assembler_rechecks_every_publication_before_minting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = _publications()
    run_store = CalibrationRunStore(tmp_path / "runs")
    durable_results = {
        task_id: publication.task_result
        for task_id, publication in zip(_FIRST_25, supplied, strict=True)
    }
    identity = SimpleNamespace(
        config_fingerprint="b" * 64,
        processed_data_fingerprint="c" * 64,
        calibration_input_fingerprint="d" * 64,
        source_code_fingerprint="e" * 64,
        dependency_lock_fingerprint="f" * 64,
    )
    monkeypatch.setattr(
        module,
        "_canonical_launch_context",
        lambda **_kwargs: (SimpleNamespace(identity=identity), object(), "1" * 64),
    )
    monkeypatch.setattr(
        run_store,
        "verify_task_results",
        lambda *_args, **_kwargs: durable_results,
    )
    checked: list[str] = []

    def verify_publication(
        publication: ScientificTaskPublication,
        *,
        expected_task_id: str,
        **_kwargs: object,
    ) -> object:
        checked.append(expected_task_id)
        return SimpleNamespace(checkpoint=publication.checkpoint)

    monkeypatch.setattr(module, "_verify_exact_v2_publication", verify_publication)
    monkeypatch.setattr(module, "_verify_selected_state_lineage", lambda _items: _K)
    handler_sources: dict[str, object] = {}

    def handlers(**kwargs: object) -> object:
        handler_sources.update(kwargs)
        return SimpleNamespace(as_mapping=lambda: MappingProxyType(kwargs))

    monkeypatch.setattr(module, "CanonicalG3First25HandlerBundle", handlers)
    prefix = SimpleNamespace(artifact_publication_ready=True)
    mint_kwargs: dict[str, object] = {}

    def mint(**kwargs: object) -> object:
        mint_kwargs.update(kwargs)
        return prefix

    monkeypatch.setattr(module, "_mint_artifact_publication_prefix", mint)
    monkeypatch.setattr(module, "_require_artifact_ready_prefix", lambda value: None)

    observed = module.assemble_exact_v2_first_25_prefix(
        run_store=run_store,
        lease=cast(
            Any,
            SimpleNamespace(run_id=_RUN_ID, preflight_fingerprint="1" * 64),
        ),
        checkpoint_store=cast(Any, object()),
        live_preflight=cast(Any, object()),
        publications=supplied,
    )

    assert observed is prefix
    assert tuple(checked) == _FIRST_25
    assert tuple(handler_sources) == _FIRST_25
    assert tuple(cast(Mapping[str, object], mint_kwargs["processed"])) == _FIRST_25
    assert cast(Any, mint_kwargs["binding"]).calibration_run_fingerprint == _RUN_ID


def test_first_25_assembler_rejects_incomplete_durable_task_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_store = CalibrationRunStore(tmp_path / "runs")
    durable_results = {
        task_id: publication.task_result
        for task_id, publication in zip(_FIRST_25[:-1], _publications(), strict=False)
    }
    monkeypatch.setattr(
        module,
        "_canonical_launch_context",
        lambda **kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(
        run_store,
        "verify_task_results",
        lambda *args, **kwargs: durable_results,
    )

    with pytest.raises(IntegrityError, match="exactly the first 25 durable tasks"):
        module.assemble_exact_v2_first_25_prefix(
            run_store=run_store,
            lease=cast(Any, SimpleNamespace(run_id=_RUN_ID)),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            publications=_publications(),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("model_store", "requires a model artifact store"),
        ("pair_store", "requires an artifact pair store"),
        ("evidence_store", "requires a G3 evidence store"),
    ),
)
def test_all_success_finalizer_rejects_substituted_store_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        module,
        "assemble_exact_v2_first_25_prefix",
        lambda **kwargs: object(),
    )
    stores: dict[str, object] = {
        "model_store": ModelArtifactStore(tmp_path / "artifacts"),
        "pair_store": ScientificArtifactPairStore(tmp_path / "artifacts"),
        "evidence_store": G3EvidenceStore(tmp_path / "artifacts"),
    }
    stores[field] = object()

    with pytest.raises(ModelError, match=message):
        module.finalize_exact_v2_all_success(
            run_store=CalibrationRunStore(tmp_path / "runs"),
            lease=cast(Any, object()),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            publications=_publications(),
            model_store=cast(Any, stores["model_store"]),
            pair_store=cast(Any, stores["pair_store"]),
            evidence_store=cast(Any, stores["evidence_store"]),
        )


def test_all_success_finalizer_requires_sealed_actual_bridge_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = SimpleNamespace(
        _handlers=SimpleNamespace(actual_artifact_bridge=object()),
    )
    pair = SimpleNamespace(reference=SimpleNamespace(fingerprint="4" * 64))
    pair_store = ScientificArtifactPairStore(tmp_path / "artifacts")
    monkeypatch.setattr(
        module,
        "assemble_exact_v2_first_25_prefix",
        lambda **kwargs: prefix,
    )
    monkeypatch.setattr(
        module,
        "build_canonical_scientific_model_artifact_draft",
        lambda prefix, *, role: SimpleNamespace(role=role),
    )
    monkeypatch.setattr(pair_store, "publish", lambda **kwargs: pair)
    monkeypatch.setattr(pair_store, "load", lambda *args, **kwargs: pair)
    monkeypatch.setattr(module, "_verify_reloaded_pair", lambda *args: None)
    monkeypatch.setattr(module, "is_bridge_driver_task_view", lambda value: False)

    with pytest.raises(IntegrityError, match="sealed actual-bridge parent"):
        module.finalize_exact_v2_all_success(
            run_store=CalibrationRunStore(tmp_path / "runs"),
            lease=cast(Any, SimpleNamespace(run_id=_RUN_ID)),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            publications=_publications(),
            model_store=ModelArtifactStore(tmp_path / "artifacts"),
            pair_store=pair_store,
            evidence_store=G3EvidenceStore(tmp_path / "artifacts"),
        )


@pytest.mark.parametrize(
    ("failure_at", "message"),
    (
        (None, None),
        ("bundle", "did not produce passing G3 evidence"),
        ("checkpoint_graph", "lacks the exact checkpoint graph"),
        ("durable_authority", "unexpectedly grants generation"),
    ),
)
def test_all_success_finalizer_publishes_reloads_task26_and_durable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str | None,
    message: str | None,
) -> None:
    supplied = _publications()
    run_store = CalibrationRunStore(tmp_path / "runs")
    model_store = ModelArtifactStore(tmp_path / "artifacts")
    pair_store = ScientificArtifactPairStore(tmp_path / "artifacts")
    evidence_store = G3EvidenceStore(tmp_path / "artifacts")
    bridge_parent = SimpleNamespace(task_id="actual_artifact_bridge")
    prefix = SimpleNamespace(
        binding=SimpleNamespace(
            config_fingerprint="b" * 64,
            processed_data_fingerprint="c" * 64,
            calibration_input_fingerprint="d" * 64,
        ),
        _handlers=SimpleNamespace(actual_artifact_bridge=bridge_parent),
        checkpoints=tuple(item.checkpoint for item in supplied),
    )
    pair = SimpleNamespace(
        reference=SimpleNamespace(fingerprint="4" * 64),
        design=SimpleNamespace(role="design"),
        production=SimpleNamespace(role="production"),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "assemble_exact_v2_first_25_prefix",
        lambda **_kwargs: prefix,
    )
    monkeypatch.setattr(
        module,
        "build_canonical_scientific_model_artifact_draft",
        lambda _prefix, *, role: SimpleNamespace(role=role),
    )

    def publish_pair(**_kwargs: object) -> object:
        calls.append("pair-publish")
        return pair

    def load_pair(*_args: object, **_kwargs: object) -> object:
        calls.append("pair-load")
        return pair

    monkeypatch.setattr(pair_store, "publish", publish_pair)
    monkeypatch.setattr(pair_store, "load", load_pair)
    monkeypatch.setattr(module, "_verify_reloaded_pair", lambda *_args: None)
    monkeypatch.setattr(module, "is_bridge_driver_task_view", lambda _value: True)
    monkeypatch.setattr(
        module,
        "artifact_integrity_executor_source_slots",
        lambda *_args, **_kwargs: SimpleNamespace(
            scientific_component_fingerprints=(),
            materialization_fingerprints=(),
            rng_fingerprints=(),
        ),
    )
    artifact_binding = SimpleNamespace(fingerprint="5" * 64)
    monkeypatch.setattr(
        module,
        "build_scientific_task_binding",
        lambda **_kwargs: artifact_binding,
    )
    artifact_view = SimpleNamespace(task_id="artifact_integrity")
    monkeypatch.setattr(
        module,
        "build_artifact_integrity_task_view",
        lambda *_args, **_kwargs: artifact_view,
    )
    artifact_publication = SimpleNamespace(
        decision=SimpleNamespace(gate_id="artifact_integrity"),
        checkpoint=SimpleNamespace(fingerprint="6" * 64),
    )

    def publish_task(**_kwargs: object) -> object:
        calls.append("task-26")
        return artifact_publication

    monkeypatch.setattr(module, "publish_exact_next_scientific_task", publish_task)
    monkeypatch.setattr(
        module,
        "_verify_exact_v2_publication",
        lambda *_a, **_k: SimpleNamespace(checkpoint=artifact_publication.checkpoint),
    )
    bundle = SimpleNamespace(
        passed=failure_at != "bundle",
        generation_authorized=False,
    )
    monkeypatch.setattr(
        module,
        "assemble_typed_g3_evidence_bundle",
        lambda **_kwargs: bundle,
    )
    results = {task_id: object() for task_id in G3_GATE_ORDER}
    monkeypatch.setattr(
        run_store,
        "read_task_result",
        lambda *_args: SimpleNamespace(run_id=_RUN_ID),
    )
    monkeypatch.setattr(
        run_store,
        "verify_task_results",
        lambda *_args, **_kwargs: results,
    )
    monkeypatch.setattr(module, "_verify_bundle_matches_results", lambda *_args: None)
    if failure_at == "checkpoint_graph":
        monkeypatch.setattr(
            module,
            "G3_GATE_ORDER",
            (*G3_GATE_ORDER[:-1], "substituted_artifact_integrity"),
        )
    completion = SimpleNamespace(state="completed", fingerprint="7" * 64)

    def complete(_lease: object) -> object:
        calls.append("complete")
        return completion

    monkeypatch.setattr(run_store, "complete_launch", complete)
    durable = SimpleNamespace(generation_authorized=failure_at == "durable_authority")

    def publish_evidence(**_kwargs: object) -> object:
        calls.append("evidence")
        return durable

    monkeypatch.setattr(evidence_store, "publish", publish_evidence)

    def invoke() -> Any:
        return module.finalize_exact_v2_all_success(
            run_store=run_store,
            lease=cast(Any, SimpleNamespace(run_id=_RUN_ID)),
            checkpoint_store=cast(Any, object()),
            live_preflight=cast(Any, object()),
            publications=supplied,
            model_store=model_store,
            pair_store=pair_store,
            evidence_store=evidence_store,
        )

    if message is not None:
        with pytest.raises(IntegrityError, match=message):
            invoke()
        if failure_at == "durable_authority":
            assert calls == [
                "pair-publish",
                "pair-load",
                "task-26",
                "complete",
                "evidence",
            ]
        else:
            assert calls == ["pair-publish", "pair-load", "task-26"]
        return

    observed = invoke()

    assert calls == ["pair-publish", "pair-load", "task-26", "complete", "evidence"]
    assert observed.first_25_publications == supplied
    assert observed.artifact_pair is pair
    assert observed.artifact_integrity_publication is artifact_publication
    assert tuple(observed.checkpoints) == G3_GATE_ORDER
    assert observed.durable_evidence is durable
    assert observed.generation_authorized is False
