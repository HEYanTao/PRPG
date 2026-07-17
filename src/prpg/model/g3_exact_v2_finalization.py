"""Closed all-success finalization for exact registry-v2 G3 publications.

This module is deliberately narrower than a general coordinator.  It accepts
only the already-published, passing task publications for registry positions
1 through 25, replays each one from the durable run/checkpoint stores, derives
the canonical scientific artifact pair, publishes task 26 from that exact
pair, completes the launch, and publishes durable evidence-only G3 evidence.

There is intentionally no recovery, failure continuation, legacy-publication,
G5, or generation path here.  Any input other than the exact live all-success
prefix fails closed before artifact publication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, cast

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3 import G3EvidenceBundle
from prpg.model.g3_assembly import (
    DerivedG3GateDecision,
    assemble_typed_g3_evidence_bundle,
    derive_g3_gate_decision,
)
from prpg.model.g3_boundary_views import (
    ArtifactIntegrityDecisionEvidence,
    build_artifact_integrity_task_view,
    is_bridge_driver_task_view,
)
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_evidence_store import (
    G3EvidenceStore,
    LoadedG3EvidenceAuthority,
)
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_science_handlers import (
    CanonicalG3ArtifactPublicationPrefix,
    CanonicalG3First25HandlerBundle,
    _canonical_launch_context,
    _mint_artifact_publication_prefix,
    _require_artifact_ready_prefix,
    _verify_bundle_matches_results,
)
from prpg.model.g3_task_binding import (
    SCIENTIFIC_TASK_BINDING_SCHEMA_ID,
    SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION,
    SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT,
    ScientificTaskBinding,
    build_scientific_task_binding,
)
from prpg.model.g3_task_publication import (
    SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION,
    SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION,
    ReverifiedScientificTaskPublication,
    ScientificTaskPublication,
    artifact_integrity_executor_source_slots,
    publish_exact_next_scientific_task,
    reverify_published_scientific_task,
    verify_recomputed_scientific_task_source,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    LaunchLease,
    LaunchMarker,
)
from prpg.model.scientific_artifact import (
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
)
from prpg.model.scientific_artifact_pair import (
    LoadedScientificArtifactPair,
    ScientificArtifactPairStore,
    build_canonical_scientific_model_artifact_draft,
    is_verified_scientific_artifact_pair,
    scientific_artifact_pair_identity,
)

_FIRST_25: Final = G3_GATE_ORDER[:-1]
_SELECTION_TASK: Final = "design_predictive_selection"
_ALLOWED_STATE_COUNTS: Final = frozenset({2, 3, 4, 5})
_RESULT_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class ExactV2G3FinalizationResult:
    """One completed, durable, evidence-only exact-v2 G3 finalization."""

    generation_authorized: ClassVar[Literal[False]] = False
    verification_scope: ClassVar[
        Literal["exact_v2_g3_all_success_durable_evidence"]
    ] = "exact_v2_g3_all_success_durable_evidence"

    prefix: CanonicalG3ArtifactPublicationPrefix
    first_25_publications: tuple[ScientificTaskPublication, ...]
    artifact_pair: LoadedScientificArtifactPair
    artifact_integrity_publication: ScientificTaskPublication
    evidence_bundle: G3EvidenceBundle
    completion_marker: LaunchMarker
    checkpoints: Mapping[str, CalibrationCheckpointReference]
    durable_evidence: LoadedG3EvidenceAuthority
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "ExactV2G3FinalizationResult is created only by its closed finalizer"
        )


def assemble_exact_v2_first_25_prefix(
    *,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    publications: Sequence[ScientificTaskPublication],
) -> CanonicalG3ArtifactPublicationPrefix:
    """Reverify exactly the first 25 passing registry-v2 publications.

    The returned prefix is the existing canonical sealed artifact-publication
    capability.  No artifact or task-26 write occurs in this stage.
    """

    reference, _checkpoint_binding, _persisted = _canonical_launch_context(
        store=run_store,
        lease=lease,
        live_preflight=live_preflight,
        checkpoint_store=checkpoint_store,
    )
    supplied = _exact_publication_sequence(publications)
    durable_results = run_store.verify_task_results(
        lease.run_id,
        require_complete=False,
    )
    if tuple(durable_results) != _FIRST_25:
        raise IntegrityError(
            "exact-v2 finalization requires exactly the first 25 durable tasks"
        )

    decisions: dict[str, DerivedG3GateDecision] = {}
    checkpoints: dict[str, CalibrationCheckpointReference] = {}
    for task_id, publication in zip(_FIRST_25, supplied, strict=True):
        replay = _verify_exact_v2_publication(
            publication,
            expected_task_id=task_id,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live_preflight,
            expected_result=durable_results[task_id],
        )
        checkpoint = replay.checkpoint
        if checkpoint is None:  # pragma: no cover - checked above for typing.
            raise AssertionError("passing exact-v2 publication lacks a checkpoint")
        decisions[task_id] = publication.decision
        checkpoints[task_id] = checkpoint

    selected_n_states = _verify_selected_state_lineage(supplied)
    handlers = CanonicalG3First25HandlerBundle(
        **cast(
            dict[str, Any],
            {task_id: decisions[task_id]._source for task_id in _FIRST_25},
        )
    )
    prefix = _mint_artifact_publication_prefix(
        run_id=lease.run_id,
        processed=durable_results,
        checkpoints=checkpoints,
        selected_n_states=selected_n_states,
        binding=ScientificArtifactBinding(
            config_fingerprint=reference.identity.config_fingerprint,
            processed_data_fingerprint=reference.identity.processed_data_fingerprint,
            calibration_input_fingerprint=(
                reference.identity.calibration_input_fingerprint
            ),
            calibration_run_fingerprint=lease.run_id,
            preflight_fingerprint=lease.preflight_fingerprint,
        ),
        provenance=ScientificArtifactProvenance(
            source_code_fingerprint=reference.identity.source_code_fingerprint,
            dependency_lock_fingerprint=(
                reference.identity.dependency_lock_fingerprint
            ),
        ),
        handlers=handlers,
        store=run_store,
    )
    _require_artifact_ready_prefix(prefix)
    return prefix


def finalize_exact_v2_all_success(
    *,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    publications: Sequence[ScientificTaskPublication],
    model_store: ModelArtifactStore,
    pair_store: ScientificArtifactPairStore,
    evidence_store: G3EvidenceStore,
) -> ExactV2G3FinalizationResult:
    """Publish/reload the exact pair, task 26, and durable G3 evidence.

    This is an intentionally one-way all-success path.  It assumes an active
    launch with exactly 25 already-published registry-v2 tasks and implements
    no completed-run or partial-finalization recovery behavior.
    """

    supplied = _exact_publication_sequence(publications)
    prefix = assemble_exact_v2_first_25_prefix(
        run_store=run_store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        publications=supplied,
    )
    if not isinstance(model_store, ModelArtifactStore):
        raise ModelError("exact-v2 finalization requires a model artifact store")
    if not isinstance(pair_store, ScientificArtifactPairStore):
        raise ModelError("exact-v2 finalization requires an artifact pair store")
    if not isinstance(evidence_store, G3EvidenceStore):
        raise ModelError("exact-v2 finalization requires a G3 evidence store")

    design = build_canonical_scientific_model_artifact_draft(
        prefix,
        role="design",
    )
    production = build_canonical_scientific_model_artifact_draft(
        prefix,
        role="production",
    )
    published_pair = pair_store.publish(
        model_store=model_store,
        design=design,
        production=production,
    )
    artifact_pair = pair_store.load(
        published_pair.reference.fingerprint,
        model_store=model_store,
        prefix=prefix,
    )
    _verify_reloaded_pair(artifact_pair, prefix)

    actual_bridge_parent = prefix._handlers.actual_artifact_bridge
    if not is_bridge_driver_task_view(actual_bridge_parent):
        raise IntegrityError("exact-v2 prefix lacks its sealed actual-bridge parent")
    checked_bridge_parent = actual_bridge_parent
    artifact_source = ArtifactIntegrityDecisionEvidence(artifact_pair)
    source_slots = artifact_integrity_executor_source_slots(
        artifact_source,
        checked_bridge_parent,
        run_id=lease.run_id,
    )
    artifact_binding = build_scientific_task_binding(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=lease.run_id,
        task_id="artifact_integrity",
        live_preflight=live_preflight,
        scientific_component_fingerprints=dict(
            source_slots.scientific_component_fingerprints
        ),
        materialization_fingerprints=dict(source_slots.materialization_fingerprints),
        rng_fingerprints=dict(source_slots.rng_fingerprints),
    )
    artifact_view = build_artifact_integrity_task_view(
        artifact_source,
        checked_bridge_parent,
        run_id=lease.run_id,
        binding_fingerprint=artifact_binding.fingerprint,
    )
    artifact_publication = publish_exact_next_scientific_task(
        run_store=run_store,
        lease=lease,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        binding=artifact_binding,
        source=artifact_view,
    )
    artifact_replay = _verify_exact_v2_publication(
        artifact_publication,
        expected_task_id="artifact_integrity",
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        expected_result=run_store.read_task_result(
            lease.run_id,
            "artifact_integrity",
        ),
    )

    all_publications = (*supplied, artifact_publication)
    evidence_bundle = assemble_typed_g3_evidence_bundle(
        config_fingerprint=prefix.binding.config_fingerprint,
        processed_data_fingerprint=prefix.binding.processed_data_fingerprint,
        calibration_input_fingerprint=prefix.binding.calibration_input_fingerprint,
        decisions=tuple(item.decision for item in all_publications),
    )
    results = run_store.verify_task_results(lease.run_id, require_complete=True)
    _verify_bundle_matches_results(evidence_bundle, results)
    if not evidence_bundle.passed or evidence_bundle.generation_authorized is not False:
        raise IntegrityError(
            "exact-v2 finalization did not produce passing G3 evidence"
        )

    artifact_checkpoint = artifact_replay.checkpoint
    if artifact_checkpoint is None:  # pragma: no cover - verifier requires passing.
        raise AssertionError("passing artifact publication lacks a checkpoint")
    checkpoint_values = dict(
        zip(
            _FIRST_25,
            prefix.checkpoints,
            strict=True,
        )
    )
    checkpoint_values["artifact_integrity"] = artifact_checkpoint
    checkpoints = MappingProxyType(checkpoint_values)
    if tuple(checkpoints) != G3_GATE_ORDER:
        raise IntegrityError("exact-v2 finalization lacks the exact checkpoint graph")
    completion = run_store.complete_launch(lease)
    durable_evidence = evidence_store.publish(
        bundle=evidence_bundle,
        run_store=run_store,
        run_id=lease.run_id,
        completion_marker=completion,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        checkpoints=checkpoints,
        model_store=model_store,
        design_artifact=artifact_pair.design,
        production_artifact=artifact_pair.production,
    )
    if durable_evidence.generation_authorized is not False:
        raise IntegrityError(
            "durable exact-v2 G3 evidence unexpectedly grants generation"
        )

    result = object.__new__(ExactV2G3FinalizationResult)
    for name, value in (
        ("prefix", prefix),
        ("first_25_publications", supplied),
        ("artifact_pair", artifact_pair),
        ("artifact_integrity_publication", artifact_publication),
        ("evidence_bundle", evidence_bundle),
        ("completion_marker", completion),
        ("checkpoints", checkpoints),
        ("durable_evidence", durable_evidence),
        ("_seal", _RESULT_SEAL),
    ):
        object.__setattr__(result, name, value)
    return result


def _exact_publication_sequence(
    publications: Sequence[ScientificTaskPublication],
) -> tuple[ScientificTaskPublication, ...]:
    if not isinstance(publications, Sequence):
        raise ModelError("exact-v2 finalization requires a publication sequence")
    supplied = tuple(publications)
    if len(supplied) != len(_FIRST_25):
        raise ModelError("exact-v2 finalization requires exactly 25 publications")
    for expected_task_id, publication in zip(_FIRST_25, supplied, strict=True):
        if type(publication) is not ScientificTaskPublication:
            raise ModelError(
                "exact-v2 finalization accepts only scientific task publications"
            )
        if type(publication.binding) is not ScientificTaskBinding:
            raise ModelError("exact-v2 publication lacks its sealed task binding")
        if publication.binding.task_id != expected_task_id:
            raise IntegrityError(
                "exact-v2 publications are reordered or substituted",
                details={"expected_task_id": expected_task_id},
            )
    return supplied


def _verify_exact_v2_publication(
    publication: ScientificTaskPublication,
    *,
    expected_task_id: str,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    expected_result: G3TaskResult,
) -> ReverifiedScientificTaskPublication:
    binding = publication.binding
    decision = publication.decision
    checkpoint = publication.checkpoint
    result = publication.task_result
    if (
        type(binding) is not ScientificTaskBinding
        or binding.schema_id != SCIENTIFIC_TASK_BINDING_SCHEMA_ID
        or binding.schema_version != SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION
        or binding.slot_registry_fingerprint
        != SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT
    ):
        raise ModelError("exact-v2 finalization rejects legacy task bindings")
    if (
        binding.task_id != expected_task_id
        or binding.run_id != expected_result.run_id
        or type(decision) is not DerivedG3GateDecision
        or decision.gate_id != expected_task_id
        or decision.passed is not True
        or type(checkpoint) is not CalibrationCheckpointReference
        or type(result) is not G3TaskResult
        or result.run_id != binding.run_id
        or result.task_id != expected_task_id
        or result.status != "passed"
        or not _same_task_result_content(result, expected_result)
    ):
        raise IntegrityError(
            "exact-v2 finalization requires one exact passing publication per task"
        )
    if (
        checkpoint.task_id != expected_task_id
        or checkpoint.binding.run_id != binding.run_id
        or result.payload.get("checkpoint_fingerprint") != checkpoint.fingerprint
        or result.payload.get("binding_fingerprint") != binding.fingerprint
        or result.payload.get("scientific_task_binding") != binding.as_dict()
        or result.payload.get("serialized_binding_sha256")
        != publication.serialized_binding_sha256
        or checkpoint.state.get("checkpoint_state_schema_version")
        != SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION
        or checkpoint.state.get("publication_schema_version")
        != SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION
        or checkpoint.state.get("binding_fingerprint") != binding.fingerprint
        or checkpoint.state.get("scientific_task_binding") != binding.as_dict()
        or checkpoint.state.get("serialized_binding_sha256")
        != publication.serialized_binding_sha256
    ):
        raise IntegrityError("exact-v2 publication content is internally inconsistent")

    source = decision._source
    verify_recomputed_scientific_task_source(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        binding=binding,
        source=source,
    )
    fresh_decision = derive_g3_gate_decision(expected_task_id, source)
    stored_decision = result.payload.get("decision")
    expected_decision = {
        "evidence": dict(fresh_decision.evidence),
        "gate_id": fresh_decision.gate_id,
        "passed": fresh_decision.passed,
        "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
        "summary": dict(fresh_decision.summary),
    }
    if (
        decision.passed != fresh_decision.passed
        or dict(decision.evidence) != dict(fresh_decision.evidence)
        or dict(decision.summary) != dict(fresh_decision.summary)
        or stored_decision != expected_decision
        or checkpoint.state.get("decision") != expected_decision
    ):
        raise IntegrityError(
            "exact-v2 scientific decision differs from its live source"
        )

    replay = reverify_published_scientific_task(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=binding.run_id,
        task_id=expected_task_id,
        live_preflight=live_preflight,
    )
    if (
        replay.binding.as_dict() != binding.as_dict()
        or replay.serialized_binding_sha256 != publication.serialized_binding_sha256
        or not _same_task_result_content(replay.task_result, result)
        or replay.checkpoint is None
        or not _same_checkpoint_content(replay.checkpoint, checkpoint)
    ):
        raise IntegrityError("exact-v2 publication differs from durable replay")
    return replay


def _same_task_result_content(left: G3TaskResult, right: G3TaskResult) -> bool:
    """Compare immutable result content while ignoring loader reuse metadata."""

    return (
        type(left) is G3TaskResult
        and type(right) is G3TaskResult
        and left.run_id == right.run_id
        and left.task_id == right.task_id
        and left.status == right.status
        and dict(left.payload) == dict(right.payload)
        and dict(left.dependency_fingerprints) == dict(right.dependency_fingerprints)
        and left.blocked_by == right.blocked_by
        and left.fingerprint == right.fingerprint
        and left.path == right.path
    )


def _same_checkpoint_content(
    left: CalibrationCheckpointReference,
    right: CalibrationCheckpointReference,
) -> bool:
    """Compare loaded checkpoint content while ignoring loader reuse metadata."""

    if (
        type(left) is not CalibrationCheckpointReference
        or type(right) is not CalibrationCheckpointReference
        or left.fingerprint != right.fingerprint
        or left.path != right.path
        or left.task_id != right.task_id
        or left.state_schema_id != right.state_schema_id
        or left.binding != right.binding
        or left.selected_n_states != right.selected_n_states
        or left.dependency_fingerprints != right.dependency_fingerprints
        or dict(left.state) != dict(right.state)
        or left.storage_reference.fingerprint != right.storage_reference.fingerprint
        or left.storage_reference.path != right.storage_reference.path
        or dict(left.storage_reference.manifest)
        != dict(right.storage_reference.manifest)
        or set(left.arrays) != set(right.arrays)
    ):
        return False
    return all(
        np.array_equal(left.arrays[name], right.arrays[name]) for name in left.arrays
    )


def _verify_selected_state_lineage(
    publications: tuple[ScientificTaskPublication, ...],
) -> int:
    selection_position = _FIRST_25.index(_SELECTION_TASK)
    selection_checkpoint = publications[selection_position].checkpoint
    if selection_checkpoint is None:  # pragma: no cover - caller already checked.
        raise AssertionError("selection publication lacks a checkpoint")
    selected = selection_checkpoint.selected_n_states
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in _ALLOWED_STATE_COUNTS
    ):
        raise IntegrityError("exact-v2 selection did not freeze an allowed state count")
    for position, publication in enumerate(publications):
        checkpoint = publication.checkpoint
        if checkpoint is None:  # pragma: no cover - caller already checked.
            raise AssertionError("passing publication lacks a checkpoint")
        expected = selected if position >= selection_position else None
        expected_binding = selected if position > selection_position else None
        if (
            checkpoint.selected_n_states != expected
            or publication.task_result.payload.get("selected_n_states") != expected
            or publication.binding.selected_n_states != expected_binding
        ):
            raise IntegrityError("exact-v2 selected-state lineage is inconsistent")
    return selected


def _verify_reloaded_pair(
    artifact_pair: LoadedScientificArtifactPair,
    prefix: CanonicalG3ArtifactPublicationPrefix,
) -> None:
    if not is_verified_scientific_artifact_pair(artifact_pair):
        raise IntegrityError("exact-v2 artifact pair did not reload with authority")
    identity = scientific_artifact_pair_identity(artifact_pair)
    if (
        artifact_pair._prefix is not prefix
        or artifact_pair.run_id != prefix.run_id
        or artifact_pair.prefix_fingerprint != prefix.prefix_fingerprint
        or artifact_pair.selected_k != prefix.selected_n_states
        or identity.get("task_result_fingerprints")
        != dict(prefix.task_result_fingerprints)
        or identity.get("checkpoint_fingerprints")
        != dict(prefix.checkpoint_fingerprints)
        or identity.get("planned_look_fingerprints")
        != {key: list(value) for key, value in prefix.planned_look_fingerprints.items()}
    ):
        raise IntegrityError("reloaded scientific pair differs from exact-v2 prefix")
