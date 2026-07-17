"""Durable, content-addressed publication of completed G3 evidence.

G3 never authorizes generation.  Publication requires a complete passing
evidence bundle plus the entire completed run, checkpoint DAG, planned-look
chains, typed scientific artifacts, pair receipt, and executable provenance at
the time of G3.  Loading repeats every durable-state check against the
external stores and returns an evidence-only, process-ledger-backed object for
future G5 binding.  It does not require later-phase source bytes to equal the
historical G3 executable identity.

The evidence object itself is immutable canonical JSON below
``<artifact-root>/g3-evidence/sha256/<fingerprint>``.  It contains no model
arrays and grants nothing when copied or parsed in isolation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, cast

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3 import (
    G3_CONTRACT_VERSION,
    G3_SCHEMA_VERSION,
    G3EvidenceBundle,
    G3GateRecord,
    gate_record,
    verify_g3_evidence_bundle,
)
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.g3_task_publication import (
    SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION,
    SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION,
    reverify_published_scientific_task,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    LaunchMarker,
    PlannedLookReceipt,
)
from prpg.model.scientific_artifact import (
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    load_scientific_model_artifact,
    verify_scientific_artifact_run_evidence,
)
from prpg.model.scientific_artifact_pair import (
    ScientificArtifactPairStore,
    verify_durable_scientific_artifact_pair_receipt,
)
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
    inspect_g3_execution_closure,
)

G3_EVIDENCE_SCHEMA_VERSION: Final = 4
G3_EVIDENCE_SCHEMA_ID: Final = "prpg-g3-durable-evidence-v4"
G3_EVIDENCE_DIRECTORY: Final = "g3-evidence"
G3_EVIDENCE_MANIFEST_NAME: Final = "evidence-manifest.json"

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_LOAD_SEAL = object()
_COORDINATOR_SCHEMA_VERSION: Final = 1
_TASK_PAYLOAD_KEYS: Final = frozenset(
    {
        "binding_fingerprint",
        "checkpoint_fingerprint",
        "coordinator_schema_version",
        "decision",
        "decision_fingerprint",
        "execution_mode",
        "outcome_kind",
        "planned_look_fingerprints",
        "scientific_task_binding",
        "selected_n_states",
        "serialized_binding_sha256",
        "summary",
    }
)
_CHECKPOINT_STATE_KEYS: Final = frozenset(
    {
        "binding_fingerprint",
        "checkpoint_state_schema_version",
        "decision",
        "decision_fingerprint",
        "planned_look_fingerprints",
        "publication_schema_version",
        "scientific_task_binding",
        "serialized_binding_sha256",
    }
)
_DECISION_KEYS: Final = frozenset(
    {"evidence", "gate_id", "passed", "source_type", "summary"}
)


@dataclass(frozen=True, slots=True)
class G3EvidenceReference:
    """Integrity-verified reference to one immutable evidence manifest."""

    fingerprint: str
    path: Path
    manifest: Mapping[str, Any]
    reused: bool


@dataclass(frozen=True, slots=True, init=False)
class LoadedG3EvidenceAuthority:
    """A completed G3 run whose durable evidence was freshly reverified.

    The historical class name is retained for API migration, but this object
    is explicitly not an authority.  It can be bound into a future G5
    qualification; it cannot be consumed directly by a generator.
    """

    generation_authorized: ClassVar[Literal[False]] = False

    verification_scope: ClassVar[
        Literal["durable_g3_run_checkpoints_looks_artifacts_and_g3_provenance"]
    ] = "durable_g3_run_checkpoints_looks_artifacts_and_g3_provenance"

    reference: G3EvidenceReference
    evidence_bundle: G3EvidenceBundle
    run_id: str
    completion_fingerprint: str
    task_result_fingerprints: Mapping[str, str]
    checkpoint_fingerprints: Mapping[str, str]
    planned_look_fingerprints: Mapping[str, tuple[str, ...]]
    design_artifact: LoadedScientificModelArtifact
    production_artifact: LoadedScientificModelArtifact
    pair_receipt_fingerprint: str
    g3_source_code_fingerprint: str
    g3_dependency_lock_fingerprint: str
    _load_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "LoadedG3EvidenceAuthority is created only by G3EvidenceStore.load"
        )


_LOADED_G3_EVIDENCE: dict[int, LoadedG3EvidenceAuthority] = {}


def verify_loaded_g3_evidence(value: object) -> LoadedG3EvidenceAuthority:
    """Recheck an original store-loaded G3 evidence object in memory."""

    if (
        type(value) is not LoadedG3EvidenceAuthority
        or value._load_seal is not _LOAD_SEAL
        or _LOADED_G3_EVIDENCE.get(id(value)) is not value
    ):
        raise ModelError("G3 evidence lacks store-loader provenance")
    try:
        verify_g3_evidence_bundle(value.evidence_bundle)
    except ModelError as error:
        raise IntegrityError("loaded G3 evidence bundle identity changed") from error
    if (
        not value.evidence_bundle.passed
        or value.evidence_bundle.generation_authorized is not False
    ):
        raise IntegrityError("loaded G3 evidence is not a passed evidence-only run")
    reference = value.reference
    manifest = reference.manifest
    if not isinstance(manifest, Mapping):
        raise IntegrityError("loaded G3 evidence manifest is unavailable")
    _exact_keys(
        manifest,
        {"schema_version", "evidence_fingerprint", "identity"},
        "loaded G3 evidence manifest",
    )
    identity = manifest["identity"]
    if (
        manifest["schema_version"] != G3_EVIDENCE_SCHEMA_VERSION
        or manifest["evidence_fingerprint"] != reference.fingerprint
        or _sha256(_canonical_bytes(identity)) != reference.fingerprint
    ):
        raise IntegrityError("loaded G3 evidence reference identity changed")
    _validate_evidence_identity(identity)
    persisted = _persisted_bundle(identity["evidence_bundle"])
    artifacts = identity["artifacts"]
    run = identity["run"]
    if (
        persisted.identity_dict() != value.evidence_bundle.identity_dict()
        or persisted.fingerprint != value.evidence_bundle.fingerprint
        or run["run_id"] != value.run_id
        or run["completion_fingerprint"] != value.completion_fingerprint
        or artifacts["design_artifact_fingerprint"]
        != value.design_artifact.reference.fingerprint
        or artifacts["production_artifact_fingerprint"]
        != value.production_artifact.reference.fingerprint
        or artifacts["pair_receipt_fingerprint"] != value.pair_receipt_fingerprint
        or identity["provenance"]["source_code_fingerprint"]
        != value.g3_source_code_fingerprint
        or identity["provenance"]["dependency_lock_fingerprint"]
        != value.g3_dependency_lock_fingerprint
        or value.evidence_bundle.design_artifact_fingerprint
        != value.design_artifact.reference.fingerprint
        or value.evidence_bundle.production_artifact_fingerprint
        != value.production_artifact.reference.fingerprint
    ):
        raise IntegrityError("loaded G3 evidence bindings changed")
    return value


class G3EvidenceStore:
    """Publish and reload exact completed-G3 evidence by content hash."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store_root = self.artifact_root / G3_EVIDENCE_DIRECTORY / "sha256"

    def publish(
        self,
        *,
        bundle: G3EvidenceBundle,
        run_store: CalibrationRunStore,
        run_id: str,
        completion_marker: LaunchMarker,
        checkpoint_store: CalibrationCheckpointStore,
        live_preflight: CanonicalG3Preflight,
        checkpoints: Mapping[str, CalibrationCheckpointReference],
        model_store: ModelArtifactStore,
        design_artifact: LoadedScientificModelArtifact,
        production_artifact: LoadedScientificModelArtifact,
    ) -> LoadedG3EvidenceAuthority:
        """Publish only a fully passed, externally reverified G3 run."""

        if not isinstance(bundle, G3EvidenceBundle):
            raise ModelError("G3 evidence publication requires a typed bundle")
        verify_g3_evidence_bundle(bundle)
        if not bundle.passed or bundle.generation_authorized is not False:
            raise ModelError("G3 evidence publication requires passed evidence only")
        if not isinstance(checkpoints, Mapping) or tuple(checkpoints) != G3_GATE_ORDER:
            raise ModelError(
                "G3 evidence publication requires the exact ordered checkpoint map"
            )

        verified = _verify_external_state(
            run_store=run_store,
            run_id=run_id,
            checkpoint_store=checkpoint_store,
            live_preflight=live_preflight,
            model_store=model_store,
            expected_bundle=bundle,
            expected_completion=completion_marker,
            expected_task_fingerprints=None,
            expected_checkpoint_fingerprints=None,
            expected_look_fingerprints=None,
            expected_artifact_fingerprints=None,
            expected_pair_receipt_fingerprint=None,
            supplied_checkpoints=checkpoints,
            supplied_artifacts=(design_artifact, production_artifact),
            require_persisted_bundle=False,
        )
        identity = _evidence_identity(
            bundle=bundle,
            run_id=run_id,
            completion=verified.completion,
            task_result_fingerprints=verified.task_result_fingerprints,
            checkpoint_fingerprints=verified.checkpoint_fingerprints,
            planned_look_fingerprints=verified.planned_look_fingerprints,
            design_artifact_fingerprint=verified.design.reference.fingerprint,
            production_artifact_fingerprint=(verified.production.reference.fingerprint),
            pair_receipt_fingerprint=verified.pair_receipt_fingerprint,
            source_code_fingerprint=verified.source_code_fingerprint,
            dependency_lock_fingerprint=verified.dependency_lock_fingerprint,
        )
        fingerprint = _sha256(_canonical_bytes(identity))
        manifest = {
            "schema_version": G3_EVIDENCE_SCHEMA_VERSION,
            "evidence_fingerprint": fingerprint,
            "identity": identity,
        }
        reused = self._write_or_verify(
            fingerprint=fingerprint,
            content=_canonical_bytes(manifest),
        )
        return self.load(
            fingerprint,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live_preflight,
            model_store=model_store,
            reused=reused,
        )

    def load(
        self,
        fingerprint: str,
        *,
        run_store: CalibrationRunStore,
        checkpoint_store: CalibrationCheckpointStore,
        model_store: ModelArtifactStore,
        live_preflight: CanonicalG3Preflight | None = None,
        reused: bool = True,
    ) -> LoadedG3EvidenceAuthority:
        """Reverify immutable G3 state without granting generation authority.

        ``live_preflight`` is retained as an optional compatibility argument.
        It is deliberately not used: a later phase has its own executable
        identity, while this loader verifies the source and lock recorded by
        G3 against the immutable G3 run, artifacts, and evidence.
        """

        reference = self.verify(fingerprint, reused=reused)
        identity = reference.manifest["identity"]
        persisted_bundle = _persisted_bundle(identity["evidence_bundle"])
        run = identity["run"]
        artifacts = identity["artifacts"]
        provenance = identity["provenance"]
        verified = _verify_external_state(
            run_store=run_store,
            run_id=_required_fingerprint(run["run_id"], "evidence run ID"),
            checkpoint_store=checkpoint_store,
            live_preflight=live_preflight,
            model_store=model_store,
            expected_bundle=persisted_bundle,
            expected_completion=_completion_from_identity(run),
            expected_task_fingerprints=_fingerprint_map(
                run["task_result_fingerprints"],
                expected=G3_GATE_ORDER,
                label="evidence task-result",
            ),
            expected_checkpoint_fingerprints=_fingerprint_map(
                run["checkpoint_fingerprints"],
                expected=G3_GATE_ORDER,
                label="evidence checkpoint",
            ),
            expected_look_fingerprints=_look_fingerprint_map(
                run["planned_look_fingerprints"],
                label="evidence planned-look",
            ),
            expected_artifact_fingerprints=(
                _required_fingerprint(
                    artifacts["design_artifact_fingerprint"],
                    "evidence design artifact",
                ),
                _required_fingerprint(
                    artifacts["production_artifact_fingerprint"],
                    "evidence production artifact",
                ),
            ),
            expected_pair_receipt_fingerprint=_required_fingerprint(
                artifacts["pair_receipt_fingerprint"],
                "evidence scientific artifact pair receipt",
            ),
            supplied_checkpoints=None,
            supplied_artifacts=None,
            require_persisted_bundle=True,
        )
        if (
            provenance["source_code_fingerprint"] != verified.source_code_fingerprint
            or provenance["dependency_lock_fingerprint"]
            != verified.dependency_lock_fingerprint
        ):
            raise IntegrityError(
                "durable G3 evidence provenance disagrees with the immutable G3 run"
            )

        loaded = object.__new__(LoadedG3EvidenceAuthority)
        for name, value in (
            ("reference", reference),
            ("evidence_bundle", persisted_bundle),
            ("run_id", verified.completion.run_id),
            ("completion_fingerprint", verified.completion.fingerprint),
            (
                "task_result_fingerprints",
                MappingProxyType(dict(verified.task_result_fingerprints)),
            ),
            (
                "checkpoint_fingerprints",
                MappingProxyType(dict(verified.checkpoint_fingerprints)),
            ),
            (
                "planned_look_fingerprints",
                MappingProxyType(dict(verified.planned_look_fingerprints)),
            ),
            ("design_artifact", verified.design),
            ("production_artifact", verified.production),
            ("pair_receipt_fingerprint", verified.pair_receipt_fingerprint),
            ("g3_source_code_fingerprint", verified.source_code_fingerprint),
            (
                "g3_dependency_lock_fingerprint",
                verified.dependency_lock_fingerprint,
            ),
            ("_load_seal", _LOAD_SEAL),
        ):
            object.__setattr__(loaded, name, value)
        _LOADED_G3_EVIDENCE[id(loaded)] = loaded
        verify_loaded_g3_evidence(loaded)
        return loaded

    def verify(self, fingerprint: str, *, reused: bool = True) -> G3EvidenceReference:
        """Verify manifest bytes, exact schema, tree allowlist, and address."""

        checked = _required_fingerprint(fingerprint, "G3 evidence fingerprint")
        _reject_symlink_components(self.store_root)
        root = self.store_root / checked
        _reject_symlink_components(root)
        if not root.is_dir() or root.is_symlink():
            raise IntegrityError("G3 evidence directory is missing or unsafe")
        entries = tuple(root.iterdir())
        if {item.name for item in entries} != {G3_EVIDENCE_MANIFEST_NAME}:
            raise IntegrityError("G3 evidence directory allowlist is invalid")
        content = _read_stable_regular(root / G3_EVIDENCE_MANIFEST_NAME)
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError("G3 evidence manifest is invalid JSON") from error
        if not isinstance(value, dict) or content != _canonical_bytes(value):
            raise IntegrityError("G3 evidence manifest is not canonical JSON")
        _exact_keys(
            value,
            {"schema_version", "evidence_fingerprint", "identity"},
            "G3 evidence manifest",
        )
        if value["schema_version"] != G3_EVIDENCE_SCHEMA_VERSION:
            raise IntegrityError("G3 evidence manifest schema is unsupported")
        if value["evidence_fingerprint"] != checked:
            raise IntegrityError("G3 evidence manifest address is inconsistent")
        identity = value["identity"]
        _validate_evidence_identity(identity)
        if _sha256(_canonical_bytes(identity)) != checked:
            raise IntegrityError("G3 evidence content hash is inconsistent")
        if {item.name for item in root.iterdir()} != {G3_EVIDENCE_MANIFEST_NAME}:
            raise IntegrityError("G3 evidence directory changed during verification")
        return G3EvidenceReference(
            checked,
            root,
            _deep_freeze(value),
            reused,
        )

    def _write_or_verify(self, *, fingerprint: str, content: bytes) -> bool:
        _prepare_directory(self.store_root)
        destination = self.store_root / fingerprint
        if destination.exists() or destination.is_symlink():
            existing = self.verify(fingerprint)
            if (
                _read_stable_regular(existing.path / G3_EVIDENCE_MANIFEST_NAME)
                != content
            ):
                raise IntegrityError("existing G3 evidence bytes differ")
            return True
        stage = Path(
            tempfile.mkdtemp(prefix=f".stage-{fingerprint[:12]}-", dir=self.store_root)
        )
        try:
            _write_new_file(stage / G3_EVIDENCE_MANIFEST_NAME, content)
            _fsync_directory(stage)
            try:
                os.rename(stage, destination)
            except OSError as error:
                if destination.exists() or destination.is_symlink():
                    existing = self.verify(fingerprint)
                    if (
                        _read_stable_regular(existing.path / G3_EVIDENCE_MANIFEST_NAME)
                        != content
                    ):
                        raise IntegrityError(
                            "concurrent G3 evidence publication differs"
                        ) from error
                    return True
                raise
            _fsync_directory(self.store_root)
            self.verify(fingerprint, reused=False)
            return False
        finally:
            if stage.exists():
                shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class _VerifiedExternalState:
    completion: LaunchMarker
    task_result_fingerprints: Mapping[str, str]
    checkpoint_fingerprints: Mapping[str, str]
    planned_look_fingerprints: Mapping[str, tuple[str, ...]]
    design: LoadedScientificModelArtifact
    production: LoadedScientificModelArtifact
    pair_receipt_fingerprint: str
    source_code_fingerprint: str
    dependency_lock_fingerprint: str


def _verify_external_state(
    *,
    run_store: CalibrationRunStore,
    run_id: str,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight | None,
    model_store: ModelArtifactStore,
    expected_bundle: G3EvidenceBundle,
    expected_completion: LaunchMarker,
    expected_task_fingerprints: Mapping[str, str] | None,
    expected_checkpoint_fingerprints: Mapping[str, str] | None,
    expected_look_fingerprints: Mapping[str, tuple[str, ...]] | None,
    expected_artifact_fingerprints: tuple[str, str] | None,
    expected_pair_receipt_fingerprint: str | None,
    supplied_checkpoints: Mapping[str, CalibrationCheckpointReference] | None,
    supplied_artifacts: tuple[
        LoadedScientificModelArtifact, LoadedScientificModelArtifact
    ]
    | None,
    require_persisted_bundle: bool,
) -> _VerifiedExternalState:
    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("G3 durable evidence requires a CalibrationRunStore")
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("G3 durable evidence requires a checkpoint store")
    if not isinstance(model_store, ModelArtifactStore):
        raise ModelError("G3 durable evidence requires a model artifact store")
    _required_fingerprint(run_id, "G3 evidence run ID")
    if not isinstance(expected_completion, LaunchMarker):
        raise ModelError("G3 evidence requires a typed completion marker")
    if not isinstance(expected_bundle, G3EvidenceBundle):
        raise ModelError("G3 durable evidence bundle has the wrong type")
    if require_persisted_bundle:
        if (
            not expected_bundle.passed
            or expected_bundle.generation_authorized is not False
        ):
            raise IntegrityError("persisted G3 evidence is not passed evidence-only")
    else:
        verify_g3_evidence_bundle(expected_bundle)
        if (
            not expected_bundle.passed
            or expected_bundle.generation_authorized is not False
        ):
            raise ModelError("G3 evidence bundle is not passed evidence-only")

    reference = run_store.verify(run_id)
    identity = reference.identity
    source_before = None
    if not require_persisted_bundle:
        if live_preflight is None:
            raise ModelError("G3 evidence publication requires a live preflight")
        source_before = inspect_g3_execution_closure()
        if (
            identity.source_code_fingerprint != source_before.source_code_fingerprint
            or identity.dependency_lock_fingerprint
            != source_before.dependency_lock_fingerprint
        ):
            raise IntegrityError("G3 run source or dependency lock is stale")
    source_code_fingerprint = identity.source_code_fingerprint
    dependency_lock_fingerprint = identity.dependency_lock_fingerprint
    if (
        identity.config_fingerprint != expected_bundle.config_fingerprint
        or identity.processed_data_fingerprint
        != expected_bundle.processed_data_fingerprint
        or identity.calibration_input_fingerprint
        != expected_bundle.calibration_input_fingerprint
        or identity.contract_version != G3_CONTRACT_VERSION
        or identity.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT
        or identity.planned_look_registry_fingerprint
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise IntegrityError("G3 bundle and completed-run identities disagree")

    completion = run_store.read_launch(run_id)
    _verify_completion(expected_completion, completion)
    if completion.state != "completed":
        raise IntegrityError("G3 evidence run is not completed")

    results = run_store.verify_task_results(run_id, require_complete=True)
    if tuple(results) != G3_GATE_ORDER or any(
        result.status != "passed" for result in results.values()
    ):
        raise IntegrityError("G3 evidence requires all exact 26 tasks to pass")
    task_fingerprints = {
        task_id: results[task_id].fingerprint for task_id in G3_GATE_ORDER
    }
    if (
        expected_task_fingerprints is not None
        and dict(expected_task_fingerprints) != task_fingerprints
    ):
        raise IntegrityError("G3 task-result fingerprints changed after publication")

    looks = run_store.verify_planned_look_receipts(
        run_id,
        require_terminal=True,
        task_results=results,
    )
    look_fingerprints = _verified_look_fingerprints(looks)
    if any(receipts[-1].decision != "pass" for receipts in looks.values()):
        raise IntegrityError("a passed G3 task has a non-passing terminal look")
    if (
        expected_look_fingerprints is not None
        and dict(expected_look_fingerprints) != look_fingerprints
    ):
        raise IntegrityError("G3 planned-look fingerprints changed after publication")

    checkpoints: dict[str, CalibrationCheckpointReference] = {}
    for task_id in G3_GATE_ORDER:
        if require_persisted_bundle:
            loaded = checkpoint_store.load_from_task_result(
                run_store,
                run_id=run_id,
                task_id=task_id,
            )
        else:
            if live_preflight is None:  # pragma: no cover - guarded above.
                raise AssertionError("live G3 preflight was not supplied")
            publication = reverify_published_scientific_task(
                run_store=run_store,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
                task_id=task_id,
                live_preflight=live_preflight,
            )
            published_checkpoint = publication.checkpoint
            if published_checkpoint is None:  # pragma: no cover - passed results.
                raise IntegrityError("passed G3 task has no binding-aware checkpoint")
            loaded = published_checkpoint
        if supplied_checkpoints is not None:
            supplied = supplied_checkpoints.get(task_id)
            if (
                not isinstance(supplied, CalibrationCheckpointReference)
                or supplied.task_id != task_id
                or supplied.fingerprint != loaded.fingerprint
            ):
                raise ModelError(
                    "supplied G3 checkpoint map differs from the completed run"
                )
        checkpoints[task_id] = loaded
    checkpoint_fingerprints = {
        task_id: checkpoints[task_id].fingerprint for task_id in G3_GATE_ORDER
    }
    if (
        expected_checkpoint_fingerprints is not None
        and dict(expected_checkpoint_fingerprints) != checkpoint_fingerprints
    ):
        raise IntegrityError("G3 checkpoint fingerprints changed after publication")

    decision_evidence = _verify_task_checkpoint_bundle_bindings(
        expected_bundle,
        results=results,
        checkpoints=checkpoints,
        planned_looks=look_fingerprints,
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
        workers=completion.workers,
    )
    design, production = _verified_artifacts(
        model_store=model_store,
        expected_fingerprints=expected_artifact_fingerprints,
        supplied=supplied_artifacts,
    )
    pair_receipt_fingerprint = _verify_artifact_bindings(
        bundle=expected_bundle,
        model_store=model_store,
        design=design,
        production=production,
        run_id=run_id,
        completion=completion,
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
        task_result_fingerprints=task_fingerprints,
        checkpoint_fingerprints=checkpoint_fingerprints,
        planned_look_fingerprints=look_fingerprints,
        artifact_integrity_evidence=decision_evidence["artifact_integrity"],
        checkpoints=checkpoints,
    )
    if (
        expected_pair_receipt_fingerprint is not None
        and pair_receipt_fingerprint != expected_pair_receipt_fingerprint
    ):
        raise IntegrityError("G3 scientific artifact pair receipt changed")
    if source_before is not None:
        source_after = inspect_g3_execution_closure()
        if source_after != source_before:
            raise IntegrityError(
                "executable source changed during G3 evidence verification"
            )
    return _VerifiedExternalState(
        completion,
        MappingProxyType(task_fingerprints),
        MappingProxyType(checkpoint_fingerprints),
        MappingProxyType(look_fingerprints),
        design,
        production,
        pair_receipt_fingerprint,
        source_code_fingerprint,
        dependency_lock_fingerprint,
    )


def _verify_completion(expected: LaunchMarker, observed: LaunchMarker) -> None:
    expected_identity = (
        expected.run_id,
        expected.state,
        expected.sequence,
        expected.preflight_fingerprint,
        expected.workers,
        expected.fingerprint,
    )
    observed_identity = (
        observed.run_id,
        observed.state,
        observed.sequence,
        observed.preflight_fingerprint,
        observed.workers,
        observed.fingerprint,
    )
    if expected_identity != observed_identity or observed.state != "completed":
        raise IntegrityError("G3 completion marker changed or is nonterminal")


def _verified_look_fingerprints(
    looks: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> dict[str, tuple[str, ...]]:
    expected = tuple(item.family_id for item in G3_PLANNED_LOOK_REGISTRY)
    if tuple(looks) != expected:
        raise IntegrityError("G3 evidence requires all exact planned-look families")
    return {
        family_id: tuple(item.fingerprint for item in looks[family_id])
        for family_id in expected
    }


def _verify_task_checkpoint_bundle_bindings(
    bundle: G3EvidenceBundle,
    *,
    results: Mapping[str, G3TaskResult],
    checkpoints: Mapping[str, CalibrationCheckpointReference],
    planned_looks: Mapping[str, tuple[str, ...]],
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
    workers: int,
) -> Mapping[str, Mapping[str, Any]]:
    if tuple(record.gate_id for record in bundle.records) != G3_GATE_ORDER:
        raise IntegrityError("G3 bundle record order differs from the task registry")
    frozen_k: int | None = None
    look_owner = {
        item.family_id: item.owner_task_id for item in G3_PLANNED_LOOK_REGISTRY
    }
    decision_evidence: dict[str, Mapping[str, Any]] = {}
    for record in bundle.records:
        task_id = record.gate_id
        result = results[task_id]
        checkpoint = checkpoints[task_id]
        payload = result.payload
        if set(payload) != _TASK_PAYLOAD_KEYS:
            raise IntegrityError("canonical G3 task payload fields are invalid")
        decision_fingerprint = payload["decision_fingerprint"]
        binding_fingerprint = payload["binding_fingerprint"]
        serialized_binding_sha256 = payload["serialized_binding_sha256"]
        serialized_binding = payload["scientific_task_binding"]
        if (
            payload["coordinator_schema_version"] != _COORDINATOR_SCHEMA_VERSION
            or payload["execution_mode"] != "canonical"
            or payload["outcome_kind"] != "typed_decision"
            or not isinstance(decision_fingerprint, str)
            or _FINGERPRINT.fullmatch(decision_fingerprint) is None
            or not isinstance(binding_fingerprint, str)
            or _FINGERPRINT.fullmatch(binding_fingerprint) is None
            or not isinstance(serialized_binding_sha256, str)
            or _FINGERPRINT.fullmatch(serialized_binding_sha256) is None
            or not isinstance(serialized_binding, Mapping)
            or _scientific_fingerprint(serialized_binding) != serialized_binding_sha256
            or payload["checkpoint_fingerprint"] != checkpoint.fingerprint
        ):
            raise IntegrityError("canonical G3 task payload identity is invalid")
        expected_task_looks = {
            family_id: list(planned_looks[family_id])
            for family_id, owner in look_owner.items()
            if owner == task_id
        }
        if payload["planned_look_fingerprints"] != expected_task_looks:
            raise IntegrityError("G3 task payload planned-look binding is invalid")

        selected_k = payload["selected_n_states"]
        position = G3_GATE_ORDER.index(task_id)
        selection_position = G3_GATE_ORDER.index("design_predictive_selection")
        if task_id == "design_predictive_selection":
            if (
                isinstance(selected_k, bool)
                or not isinstance(selected_k, int)
                or selected_k not in {2, 3, 4, 5}
            ):
                raise IntegrityError("G3 selection task has no valid frozen K")
            frozen_k = selected_k
        elif position < selection_position:
            if selected_k is not None:
                raise IntegrityError("G3 preselection task prematurely binds K")
        elif selected_k != frozen_k:
            raise IntegrityError("G3 task changed the frozen selected K")
        if checkpoint.selected_n_states != selected_k:
            raise IntegrityError("G3 result/checkpoint selected-K binding differs")

        state = _json_thaw(checkpoint.state)
        if not isinstance(state, Mapping) or set(state) != _CHECKPOINT_STATE_KEYS:
            raise IntegrityError("canonical G3 checkpoint state fields are invalid")
        decision = state["decision"]
        if not isinstance(decision, Mapping) or set(decision) != _DECISION_KEYS:
            raise IntegrityError("canonical G3 checkpoint decision is invalid")
        if (
            state["checkpoint_state_schema_version"]
            != SCIENTIFIC_TASK_CHECKPOINT_STATE_VERSION
            or state["publication_schema_version"]
            != SCIENTIFIC_TASK_PUBLICATION_SCHEMA_VERSION
            or state["decision_fingerprint"] != decision_fingerprint
            or state["planned_look_fingerprints"] != expected_task_looks
            or state["binding_fingerprint"] != binding_fingerprint
            or state["serialized_binding_sha256"] != serialized_binding_sha256
            or state["scientific_task_binding"] != payload["scientific_task_binding"]
            or payload["decision"] != decision
            or decision["gate_id"] != task_id
            or decision["passed"] is not True
            or not isinstance(decision["source_type"], str)
            or not decision["source_type"]
            or not isinstance(decision["evidence"], Mapping)
            or not decision["evidence"]
            or not isinstance(decision["summary"], Mapping)
            or not decision["summary"]
            or payload["summary"] != decision["summary"]
            or _scientific_fingerprint(decision) != decision_fingerprint
        ):
            raise IntegrityError("canonical G3 checkpoint decision binding is invalid")
        try:
            expected_record = gate_record(
                task_id,
                passed=True,
                evidence=decision["evidence"],
                summary=decision["summary"],
            )
        except ModelError as error:
            raise IntegrityError(
                "canonical G3 checkpoint decision cannot form a gate record"
            ) from error
        if record.as_dict() != expected_record.as_dict():
            raise IntegrityError("G3 bundle record differs from checkpoint decision")
        decision_evidence[task_id] = decision["evidence"]
        _verify_checkpoint_digest_arrays(
            checkpoint,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints=expected_task_looks,
            serialized_binding_sha256=serialized_binding_sha256,
        )
    _verify_input_integrity_evidence(
        decision_evidence["input_integrity"],
        checkpoints["input_integrity"],
        results["input_integrity"],
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
        workers=workers,
    )
    return MappingProxyType(decision_evidence)


def _verify_input_integrity_evidence(
    evidence: Mapping[str, Any],
    checkpoint: CalibrationCheckpointReference,
    result: G3TaskResult,
    *,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
    workers: int,
) -> None:
    binding = checkpoint.binding
    expected_keys = {
        "preflight_fingerprint",
        "live_preflight_compatible",
        "config_fingerprint",
        "processed_data_fingerprint",
        "calibration_input_fingerprint",
        "source_code_fingerprint",
        "dependency_lock_fingerprint",
        "workers",
        "check_count",
    }
    if set(evidence) != expected_keys:
        raise IntegrityError("input-integrity G3 evidence fields are invalid")
    if (
        evidence["preflight_fingerprint"] != binding.preflight_fingerprint
        or evidence["live_preflight_compatible"] is not True
        or evidence["config_fingerprint"] != binding.config_fingerprint
        or evidence["processed_data_fingerprint"] != binding.processed_data_fingerprint
        or evidence["calibration_input_fingerprint"]
        != binding.calibration_input_fingerprint
        or evidence["source_code_fingerprint"] != source_code_fingerprint
        or evidence["dependency_lock_fingerprint"] != dependency_lock_fingerprint
        or evidence["workers"] != workers
        or isinstance(evidence["check_count"], bool)
        or not isinstance(evidence["check_count"], int)
        or evidence["check_count"] <= 0
        or result.run_id != binding.run_id
    ):
        raise IntegrityError("input-integrity G3 evidence binding is invalid")


def _verify_checkpoint_digest_arrays(
    checkpoint: CalibrationCheckpointReference,
    *,
    decision_fingerprint: str,
    planned_look_fingerprints: Mapping[str, list[str]],
    serialized_binding_sha256: str,
) -> None:
    if set(checkpoint.arrays) != {
        "decision_sha256",
        "planned_looks_sha256",
        "scientific_task_binding_sha256",
    }:
        raise IntegrityError("canonical G3 checkpoint digest arrays are invalid")
    expected_decision = np.frombuffer(
        bytes.fromhex(decision_fingerprint), dtype=np.uint8
    )
    expected_looks = np.frombuffer(
        bytes.fromhex(_scientific_fingerprint(planned_look_fingerprints)),
        dtype=np.uint8,
    )
    expected_binding = np.frombuffer(
        bytes.fromhex(serialized_binding_sha256),
        dtype=np.uint8,
    )
    if (
        not np.array_equal(checkpoint.arrays["decision_sha256"], expected_decision)
        or not np.array_equal(checkpoint.arrays["planned_looks_sha256"], expected_looks)
        or not np.array_equal(
            checkpoint.arrays["scientific_task_binding_sha256"], expected_binding
        )
    ):
        raise IntegrityError("canonical G3 checkpoint digest arrays disagree")


def _verified_artifacts(
    *,
    model_store: ModelArtifactStore,
    expected_fingerprints: tuple[str, str] | None,
    supplied: tuple[LoadedScientificModelArtifact, LoadedScientificModelArtifact]
    | None,
) -> tuple[LoadedScientificModelArtifact, LoadedScientificModelArtifact]:
    if supplied is not None:
        supplied_design, supplied_production = supplied
        if not isinstance(
            supplied_design, LoadedScientificModelArtifact
        ) or not isinstance(supplied_production, LoadedScientificModelArtifact):
            raise ModelError("G3 publication requires two typed scientific artifacts")
        fingerprints = (
            supplied_design.reference.fingerprint,
            supplied_production.reference.fingerprint,
        )
    elif expected_fingerprints is not None:
        fingerprints = expected_fingerprints
    else:  # pragma: no cover - internal caller invariant
        raise AssertionError("artifact fingerprints were not supplied")
    design = load_scientific_model_artifact(model_store, fingerprints[0])
    production = load_scientific_model_artifact(model_store, fingerprints[1])
    if supplied is not None and (
        supplied[0].reference.fingerprint != design.reference.fingerprint
        or supplied[1].reference.fingerprint != production.reference.fingerprint
    ):
        raise ModelError("supplied scientific artifacts differ from their store")
    return design, production


def _verify_artifact_bindings(
    *,
    bundle: G3EvidenceBundle,
    model_store: ModelArtifactStore,
    design: LoadedScientificModelArtifact,
    production: LoadedScientificModelArtifact,
    run_id: str,
    completion: LaunchMarker,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
    task_result_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    planned_look_fingerprints: Mapping[str, tuple[str, ...]],
    artifact_integrity_evidence: Mapping[str, Any],
    checkpoints: Mapping[str, CalibrationCheckpointReference],
) -> str:
    binding = ScientificArtifactBinding(
        config_fingerprint=bundle.config_fingerprint,
        processed_data_fingerprint=bundle.processed_data_fingerprint,
        calibration_input_fingerprint=bundle.calibration_input_fingerprint,
        calibration_run_fingerprint=run_id,
        preflight_fingerprint=completion.preflight_fingerprint,
    )
    provenance = ScientificArtifactProvenance(
        source_code_fingerprint=source_code_fingerprint,
        dependency_lock_fingerprint=dependency_lock_fingerprint,
    )
    if (
        design.role != "design"
        or production.role != "production"
        or design.selected_k != production.selected_k
        or design.reference.fingerprint == production.reference.fingerprint
        or bundle.design_artifact_fingerprint != design.reference.fingerprint
        or bundle.production_artifact_fingerprint != production.reference.fingerprint
    ):
        raise IntegrityError("G3 scientific artifact pair identity is invalid")
    for artifact in (design, production):
        verify_scientific_artifact_run_evidence(
            artifact,
            binding=binding,
            provenance=provenance,
            task_result_fingerprints=task_result_fingerprints,
            checkpoint_fingerprints=checkpoint_fingerprints,
            planned_look_fingerprints=planned_look_fingerprints,
        )
    artifact_decision = dict(artifact_integrity_evidence)
    pair_receipt_fingerprint = artifact_decision.get("pair_receipt_fingerprint")
    design_role_source_fingerprint = artifact_decision.get(
        "design_role_source_fingerprint"
    )
    production_role_source_fingerprint = artifact_decision.get(
        "production_role_source_fingerprint"
    )
    actual_bridge_parent_view_fingerprint = artifact_decision.get(
        "actual_bridge_parent_view_fingerprint"
    )
    if (
        artifact_decision.get("design_artifact_fingerprint")
        != design.reference.fingerprint
        or artifact_decision.get("production_artifact_fingerprint")
        != production.reference.fingerprint
        or artifact_decision.get("design_parameter_set_fingerprint")
        != design.parameter_set_fingerprint
        or artifact_decision.get("production_parameter_set_fingerprint")
        != production.parameter_set_fingerprint
        or artifact_decision.get("selected_k") != design.selected_k
        or artifact_decision.get("binding") != binding.as_dict()
        or artifact_decision.get("provenance") != provenance.as_dict()
        or artifact_decision.get("typed_schema_verified") is not True
        or not isinstance(pair_receipt_fingerprint, str)
        or _FINGERPRINT.fullmatch(pair_receipt_fingerprint) is None
        or not isinstance(design_role_source_fingerprint, str)
        or _FINGERPRINT.fullmatch(design_role_source_fingerprint) is None
        or not isinstance(production_role_source_fingerprint, str)
        or _FINGERPRINT.fullmatch(production_role_source_fingerprint) is None
        or not isinstance(actual_bridge_parent_view_fingerprint, str)
        or _FINGERPRINT.fullmatch(actual_bridge_parent_view_fingerprint) is None
    ):
        raise IntegrityError("artifact-integrity G3 evidence differs from artifacts")
    decision_documents: dict[str, Mapping[str, Any]] = {}
    for task_id in G3_GATE_ORDER[:-1]:
        state = _json_thaw(checkpoints[task_id].state)
        if not isinstance(state, Mapping) or not isinstance(
            state.get("decision"), Mapping
        ):
            raise IntegrityError("G3 checkpoint lacks its durable decision document")
        decision_documents[task_id] = cast(Mapping[str, Any], state["decision"])
    first_25_task_result_fingerprints = {
        task_id: task_result_fingerprints[task_id] for task_id in G3_GATE_ORDER[:-1]
    }
    first_25_checkpoint_fingerprints = {
        task_id: checkpoint_fingerprints[task_id] for task_id in G3_GATE_ORDER[:-1]
    }
    verify_durable_scientific_artifact_pair_receipt(
        pair_store=ScientificArtifactPairStore(model_store.artifact_root),
        model_store=model_store,
        fingerprint=pair_receipt_fingerprint,
        design=design,
        production=production,
        run_id=run_id,
        selected_k=design.selected_k,
        binding=binding,
        provenance=provenance,
        task_result_fingerprints=first_25_task_result_fingerprints,
        checkpoint_fingerprints=first_25_checkpoint_fingerprints,
        planned_look_fingerprints=planned_look_fingerprints,
        decision_documents=decision_documents,
        actual_bridge_parent_view_fingerprint=(actual_bridge_parent_view_fingerprint),
        design_role_source_fingerprint=design_role_source_fingerprint,
        production_role_source_fingerprint=production_role_source_fingerprint,
    )
    return pair_receipt_fingerprint


def _evidence_identity(
    *,
    bundle: G3EvidenceBundle,
    run_id: str,
    completion: LaunchMarker,
    task_result_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    planned_look_fingerprints: Mapping[str, tuple[str, ...]],
    design_artifact_fingerprint: str,
    production_artifact_fingerprint: str,
    pair_receipt_fingerprint: str,
    source_code_fingerprint: str,
    dependency_lock_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_id": G3_EVIDENCE_SCHEMA_ID,
        "schema_version": G3_EVIDENCE_SCHEMA_VERSION,
        "contract_version": G3_CONTRACT_VERSION,
        "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
        "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
        "passed": bundle.passed,
        "generation_authorized": bundle.generation_authorized,
        "evidence_bundle": {
            "identity": bundle.identity_dict(),
            "fingerprint": bundle.fingerprint,
        },
        "run": {
            "run_id": run_id,
            "completion_fingerprint": completion.fingerprint,
            "completion_sequence": completion.sequence,
            "preflight_fingerprint": completion.preflight_fingerprint,
            "workers": completion.workers,
            "task_result_fingerprints": dict(task_result_fingerprints),
            "checkpoint_fingerprints": dict(checkpoint_fingerprints),
            "planned_look_fingerprints": {
                key: list(value) for key, value in planned_look_fingerprints.items()
            },
        },
        "artifacts": {
            "design_artifact_fingerprint": design_artifact_fingerprint,
            "production_artifact_fingerprint": production_artifact_fingerprint,
            "pair_receipt_fingerprint": pair_receipt_fingerprint,
        },
        "provenance": {
            "execution_closure_schema": G3_EXECUTION_CLOSURE_SCHEMA,
            "execution_closure_manifest": G3_EXECUTION_CLOSURE_MANIFEST.as_dict(),
            "execution_closure_manifest_fingerprint": (
                G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
            ),
            "source_code_fingerprint": source_code_fingerprint,
            "dependency_lock_fingerprint": dependency_lock_fingerprint,
        },
    }


def _validate_evidence_identity(value: object) -> None:
    if not isinstance(value, Mapping):
        raise IntegrityError("G3 evidence identity is invalid")
    _exact_keys(
        value,
        {
            "schema_id",
            "schema_version",
            "contract_version",
            "task_registry_fingerprint",
            "planned_look_registry_fingerprint",
            "passed",
            "generation_authorized",
            "evidence_bundle",
            "run",
            "artifacts",
            "provenance",
        },
        "G3 evidence identity",
    )
    if (
        value["schema_id"] != G3_EVIDENCE_SCHEMA_ID
        or value["schema_version"] != G3_EVIDENCE_SCHEMA_VERSION
        or value["contract_version"] != G3_CONTRACT_VERSION
        or value["task_registry_fingerprint"] != G3_TASK_REGISTRY_FINGERPRINT
        or value["planned_look_registry_fingerprint"]
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
        or value["passed"] is not True
        or value["generation_authorized"] is not False
    ):
        raise IntegrityError("G3 evidence contract identity is unsupported")
    bundle = value["evidence_bundle"]
    _exact_keys(bundle, {"identity", "fingerprint"}, "G3 persisted bundle")
    _required_fingerprint(bundle["fingerprint"], "persisted G3 bundle")
    _persisted_bundle(bundle)
    run = value["run"]
    _exact_keys(
        run,
        {
            "run_id",
            "completion_fingerprint",
            "completion_sequence",
            "preflight_fingerprint",
            "workers",
            "task_result_fingerprints",
            "checkpoint_fingerprints",
            "planned_look_fingerprints",
        },
        "G3 evidence run",
    )
    _required_fingerprint(run["run_id"], "evidence run ID")
    _required_fingerprint(run["completion_fingerprint"], "completion")
    _required_fingerprint(run["preflight_fingerprint"], "preflight")
    if (
        isinstance(run["completion_sequence"], bool)
        or not isinstance(run["completion_sequence"], int)
        or run["completion_sequence"] < 2
        or isinstance(run["workers"], bool)
        or not isinstance(run["workers"], int)
        or run["workers"] <= 0
    ):
        raise IntegrityError("G3 evidence completion fields are invalid")
    _fingerprint_map(
        run["task_result_fingerprints"],
        expected=G3_GATE_ORDER,
        label="evidence task-result",
    )
    _fingerprint_map(
        run["checkpoint_fingerprints"],
        expected=G3_GATE_ORDER,
        label="evidence checkpoint",
    )
    _look_fingerprint_map(
        run["planned_look_fingerprints"], label="evidence planned-look"
    )
    artifacts = value["artifacts"]
    _exact_keys(
        artifacts,
        {
            "design_artifact_fingerprint",
            "production_artifact_fingerprint",
            "pair_receipt_fingerprint",
        },
        "G3 evidence artifacts",
    )
    _required_fingerprint(artifacts["design_artifact_fingerprint"], "design artifact")
    _required_fingerprint(
        artifacts["production_artifact_fingerprint"], "production artifact"
    )
    _required_fingerprint(
        artifacts["pair_receipt_fingerprint"], "scientific artifact pair receipt"
    )
    provenance = value["provenance"]
    _exact_keys(
        provenance,
        {
            "execution_closure_schema",
            "execution_closure_manifest",
            "execution_closure_manifest_fingerprint",
            "source_code_fingerprint",
            "dependency_lock_fingerprint",
        },
        "G3 evidence provenance",
    )
    if (
        provenance["execution_closure_schema"] != G3_EXECUTION_CLOSURE_SCHEMA
        or _json_thaw(provenance["execution_closure_manifest"])
        != G3_EXECUTION_CLOSURE_MANIFEST.as_dict()
        or provenance["execution_closure_manifest_fingerprint"]
        != G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
    ):
        raise IntegrityError("G3 evidence execution-closure manifest is invalid")
    _required_fingerprint(provenance["source_code_fingerprint"], "source code")
    _required_fingerprint(provenance["dependency_lock_fingerprint"], "dependency lock")


def _persisted_bundle(value: object) -> G3EvidenceBundle:
    if not isinstance(value, Mapping):
        raise IntegrityError("persisted G3 bundle is invalid")
    _exact_keys(value, {"identity", "fingerprint"}, "persisted G3 bundle")
    identity = value["identity"]
    if not isinstance(identity, Mapping):
        raise IntegrityError("persisted G3 bundle identity is invalid")
    _exact_keys(
        identity,
        {
            "schema_version",
            "contract_version",
            "config_fingerprint",
            "processed_data_fingerprint",
            "calibration_input_fingerprint",
            "design_artifact_fingerprint",
            "production_artifact_fingerprint",
            "records",
            "passed",
            "generation_authorized",
        },
        "persisted G3 bundle identity",
    )
    records_value = identity["records"]
    if (
        identity["schema_version"] != G3_SCHEMA_VERSION
        or identity["contract_version"] != G3_CONTRACT_VERSION
        or identity["passed"] is not True
        or identity["generation_authorized"] is not False
    ):
        raise IntegrityError("persisted G3 bundle decision fields are invalid")
    for key, label in (
        ("config_fingerprint", "persisted G3 config"),
        ("processed_data_fingerprint", "persisted G3 processed data"),
        ("calibration_input_fingerprint", "persisted G3 calibration input"),
        ("design_artifact_fingerprint", "persisted G3 design artifact"),
        ("production_artifact_fingerprint", "persisted G3 production artifact"),
    ):
        _required_fingerprint(identity[key], label)
    if not isinstance(records_value, list | tuple) or len(records_value) != len(
        G3_GATE_ORDER
    ):
        raise IntegrityError("persisted G3 records are invalid")
    records: list[G3GateRecord] = []
    for expected_gate_id, item in zip(G3_GATE_ORDER, records_value, strict=True):
        if not isinstance(item, Mapping):
            raise IntegrityError("persisted G3 record is invalid")
        _exact_keys(
            item,
            {"gate_id", "status", "evidence_fingerprint", "summary"},
            "persisted G3 record",
        )
        summary = item["summary"]
        if (
            item["gate_id"] != expected_gate_id
            or item["status"] != "passed"
            or not isinstance(summary, Mapping)
            or not summary
        ):
            raise IntegrityError("persisted G3 record summary is invalid")
        _required_fingerprint(
            item["evidence_fingerprint"], "persisted G3 gate evidence"
        )
        _canonical_bytes(summary)
        records.append(
            G3GateRecord(
                gate_id=item["gate_id"],
                status=item["status"],
                evidence_fingerprint=item["evidence_fingerprint"],
                summary=MappingProxyType(dict(summary)),
            )
        )
    try:
        bundle = G3EvidenceBundle(
            schema_version=identity["schema_version"],
            contract_version=identity["contract_version"],
            config_fingerprint=identity["config_fingerprint"],
            processed_data_fingerprint=identity["processed_data_fingerprint"],
            calibration_input_fingerprint=(identity["calibration_input_fingerprint"]),
            design_artifact_fingerprint=identity["design_artifact_fingerprint"],
            production_artifact_fingerprint=(
                identity["production_artifact_fingerprint"]
            ),
            records=tuple(records),
            passed=identity["passed"],
            generation_authorized=identity["generation_authorized"],
            fingerprint=value["fingerprint"],
        )
    except (TypeError, ValueError) as error:
        raise IntegrityError("persisted G3 bundle fields are invalid") from error
    if (
        bundle.schema_version != G3_SCHEMA_VERSION
        or bundle.contract_version != G3_CONTRACT_VERSION
        or bundle.passed is not True
        or bundle.generation_authorized is not False
        or bundle.design_artifact_fingerprint is None
        or bundle.production_artifact_fingerprint is None
        or tuple(record.gate_id for record in bundle.records) != G3_GATE_ORDER
        or any(not record.passed for record in bundle.records)
    ):
        raise IntegrityError("persisted G3 bundle is not a complete passing decision")
    _required_fingerprint(bundle.fingerprint, "persisted G3 bundle")
    return bundle


def _completion_from_identity(run: Mapping[str, Any]) -> LaunchMarker:
    return LaunchMarker(
        run_id=run["run_id"],
        state="completed",
        sequence=run["completion_sequence"],
        preflight_fingerprint=run["preflight_fingerprint"],
        workers=run["workers"],
        fingerprint=run["completion_fingerprint"],
        path=Path(),
        reused=True,
    )


def _fingerprint_map(
    value: object, *, expected: tuple[str, ...], label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise IntegrityError(f"{label} mapping is not exact")
    result: dict[str, str] = {}
    for key in expected:
        result[key] = _required_fingerprint(value[key], f"{label} {key}")
    return result


def _look_fingerprint_map(value: object, *, label: str) -> dict[str, tuple[str, ...]]:
    expected = tuple(item.family_id for item in G3_PLANNED_LOOK_REGISTRY)
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise IntegrityError(f"{label} mapping is not exact")
    specs = {item.family_id: item for item in G3_PLANNED_LOOK_REGISTRY}
    result: dict[str, tuple[str, ...]] = {}
    for family_id in expected:
        sequence = value[family_id]
        if (
            not isinstance(sequence, list | tuple)
            or not sequence
            or len(sequence) > len(specs[family_id].sample_counts)
        ):
            raise IntegrityError(f"{label} sequence is invalid")
        result[family_id] = tuple(
            _required_fingerprint(item, f"{label} receipt") for item in sequence
        )
    return result


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise IntegrityError(f"{label} keys are invalid")


def _scientific_fingerprint(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _json_thaw(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise IntegrityError("G3 evidence contains noncanonical JSON values") from error


def _json_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_thaw(item) for item in value]
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prepare_directory(path: Path) -> None:
    _reject_symlink_components(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlink_components(path)
    if not path.is_dir():
        raise IntegrityError("G3 evidence store root is not a directory")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise IntegrityError("G3 evidence path cannot be inspected") from error
        if stat.S_ISLNK(info.st_mode):
            raise IntegrityError("G3 evidence path contains a symbolic link")


def _write_new_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive operating-system guard
                raise OSError("short write while publishing G3 evidence")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_stable_regular(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise IntegrityError("G3 evidence manifest is missing") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise IntegrityError("G3 evidence manifest is not a private regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError("G3 evidence manifest cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise IntegrityError(
            "G3 evidence manifest disappeared while reading"
        ) from error
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(opened, name) != getattr(after, name)
        or getattr(after, name) != getattr(current, name)
        for name in fields
    ):
        raise IntegrityError("G3 evidence manifest changed while reading")
    content = b"".join(chunks)
    if len(content) != after.st_size:
        raise IntegrityError("G3 evidence manifest byte count changed")
    return content


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value
