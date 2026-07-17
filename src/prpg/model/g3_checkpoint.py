"""Immutable, non-authorizing checkpoints for crash-resumable G3 execution.

Final scientific model artifacts are deliberately all-or-nothing.  This
module therefore stores intermediate task state in a physically and
semantically separate namespace.  It reuses the hardened, non-pickle generic
artifact container but wraps it in a closed G3 schema whose dependency chain,
selected state count, run identity, and preregistration fingerprints are
derived and reverified on every load.

A checkpoint is only durable computation state.  Its type exposes
``generation_authorized = False`` and no function in this module can create a
G3 evidence bundle or authorize a generator.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactReference, ModelArtifactStore
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
    task_spec,
)
from prpg.model.run_store import CalibrationRunStore

CHECKPOINT_SCHEMA_VERSION: Final = 1
CHECKPOINT_SCHEMA_ID: Final = "prpg-g3-calibration-checkpoint-v1"
CHECKPOINT_ARTIFACT_KIND: Final = "calibration_checkpoint"
CHECKPOINT_STATE_PATH: Final = "checkpoint/state.json"
CHECKPOINT_STORAGE_DIRECTORY: Final = "calibration-checkpoints"
_SELECTION_TASK: Final = "design_predictive_selection"
_SELECTION_POSITION: Final = G3_GATE_ORDER.index(_SELECTION_TASK)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CalibrationCheckpointBinding:
    """Immutable run/input identity shared by every checkpoint in one launch."""

    run_id: str
    config_fingerprint: str
    processed_data_fingerprint: str
    calibration_input_fingerprint: str
    preflight_fingerprint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "calibration_input_fingerprint": self.calibration_input_fingerprint,
            "preflight_fingerprint": self.preflight_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCheckpointReference:
    """Fully verified intermediate state that can never authorize generation."""

    artifact_kind: ClassVar[Literal["calibration_checkpoint"]] = (
        "calibration_checkpoint"
    )
    generation_authorized: ClassVar[Literal[False]] = False
    verification_scope: ClassVar[Literal["checkpoint_integrity_and_schema"]] = (
        "checkpoint_integrity_and_schema"
    )

    fingerprint: str
    path: Path
    task_id: str
    state_schema_id: str
    binding: CalibrationCheckpointBinding
    selected_n_states: int | None
    dependency_fingerprints: tuple[tuple[str, str], ...]
    arrays: Mapping[str, np.ndarray[Any, Any]]
    state: Mapping[str, Any]
    storage_reference: ModelArtifactReference
    reused: bool


class CalibrationCheckpointStore:
    """Publish checkpoints below a namespace distinct from final models."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.checkpoint_root = self.artifact_root / CHECKPOINT_STORAGE_DIRECTORY
        self._store = ModelArtifactStore(self.checkpoint_root)

    def create(
        self,
        *,
        task_id: str,
        binding: CalibrationCheckpointBinding,
        dependencies: Sequence[CalibrationCheckpointReference],
        selected_n_states: int | None,
        arrays: Mapping[str, npt.ArrayLike],
        state: Mapping[str, Any],
    ) -> CalibrationCheckpointReference:
        """Publish one exact passed-task checkpoint and typed-load it again."""

        spec = task_spec(task_id)
        checked_binding = _binding(binding, ModelError)
        checked_dependencies = self._validated_dependencies(
            task_id=task_id,
            expected=spec.dependencies,
            supplied=dependencies,
            binding=checked_binding,
        )
        checked_k = _selected_n_states_for_task(
            task_id,
            selected_n_states,
            checked_dependencies,
            ModelError,
        )
        if not isinstance(arrays, Mapping):
            raise ModelError("checkpoint arrays must be a mapping")
        stored_arrays: dict[str, npt.ArrayLike] = dict(arrays)
        if not stored_arrays:
            stored_arrays["checkpoint_marker"] = np.asarray([1], dtype=np.uint8)
        if not isinstance(state, Mapping) or not state:
            raise ModelError("checkpoint state must be a non-empty mapping")
        dependency_fingerprints = {
            item.task_id: item.fingerprint for item in checked_dependencies
        }
        state_schema_id = _state_schema_id(task_id)
        metadata = {
            "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_schema_id": state_schema_id,
            "task_id": task_id,
            "binding": checked_binding.as_dict(),
            "selected_n_states": checked_k,
            "dependency_fingerprints": dependency_fingerprints,
            "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
            "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
            "contract_version": G3_CONTRACT_VERSION,
        }
        document = {
            "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_schema_id": state_schema_id,
            "task_id": task_id,
            "binding": checked_binding.as_dict(),
            "selected_n_states": checked_k,
            "dependency_fingerprints": dependency_fingerprints,
            "state": dict(state),
        }
        provenance = {
            "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
            "producer": "prpg-g3-checkpoint-store-v1",
            "contract_version": G3_CONTRACT_VERSION,
            "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
            "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
        }
        reference = self._store.create(
            artifact_role=_storage_role(task_id),
            data_fingerprint=checked_binding.processed_data_fingerprint,
            model_metadata=metadata,
            arrays=stored_arrays,
            calibration_documents={CHECKPOINT_STATE_PATH: document},
            provenance=provenance,
        )
        loaded = self.load(reference.fingerprint)
        return replace(
            loaded,
            storage_reference=reference,
            reused=reference.reused,
        )

    def load(self, fingerprint: str) -> CalibrationCheckpointReference:
        """Load a checkpoint and recursively verify its complete DAG prefix."""

        return self._load(fingerprint, ancestry=frozenset())

    def load_from_task_result(
        self,
        run_store: CalibrationRunStore,
        *,
        run_id: str,
        task_id: str,
    ) -> CalibrationCheckpointReference:
        """Resolve only a passed task result's exact checkpoint reference."""

        if not isinstance(run_store, CalibrationRunStore):
            raise ModelError("checkpoint lookup requires a CalibrationRunStore")
        result = run_store.read_task_result(run_id, task_id)
        if result.status != "passed":
            raise ModelError("only a passed G3 task can reference a checkpoint")
        fingerprint = result.payload.get("checkpoint_fingerprint")
        if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
            raise IntegrityError("passed G3 task has no valid checkpoint fingerprint")
        checkpoint = self.load(fingerprint)
        run_reference = run_store.verify(run_id)
        launch = run_store.read_launch(run_id)
        if (
            checkpoint.task_id != result.task_id
            or checkpoint.binding.run_id != result.run_id
            or checkpoint.binding.config_fingerprint
            != run_reference.identity.config_fingerprint
            or checkpoint.binding.processed_data_fingerprint
            != run_reference.identity.processed_data_fingerprint
            or checkpoint.binding.calibration_input_fingerprint
            != run_reference.identity.calibration_input_fingerprint
            or checkpoint.binding.preflight_fingerprint != launch.preflight_fingerprint
        ):
            raise IntegrityError("G3 task result/checkpoint identity disagrees")
        for (
            dependency_task,
            dependency_checkpoint,
        ) in checkpoint.dependency_fingerprints:
            dependency_result = run_store.read_task_result(run_id, dependency_task)
            if (
                dependency_result.status != "passed"
                or dependency_result.payload.get("checkpoint_fingerprint")
                != dependency_checkpoint
            ):
                raise IntegrityError(
                    "checkpoint dependency does not match the G3 result chain"
                )
        return checkpoint

    def _load(
        self, fingerprint: str, *, ancestry: frozenset[str]
    ) -> CalibrationCheckpointReference:
        if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
            raise IntegrityError("calibration checkpoint fingerprint is invalid")
        if fingerprint in ancestry:
            raise IntegrityError("calibration checkpoint dependency cycle detected")
        reference = self._store.verify(fingerprint)
        identity = reference.manifest["identity"]
        metadata = identity.get("model_metadata")
        if not isinstance(metadata, Mapping):
            raise IntegrityError("checkpoint metadata is missing")
        _exact_keys(
            metadata,
            {
                "artifact_kind",
                "schema_id",
                "schema_version",
                "state_schema_id",
                "task_id",
                "binding",
                "selected_n_states",
                "dependency_fingerprints",
                "task_registry_fingerprint",
                "planned_look_registry_fingerprint",
                "contract_version",
            },
            "checkpoint metadata",
        )
        if (
            metadata["artifact_kind"] != CHECKPOINT_ARTIFACT_KIND
            or metadata["schema_id"] != CHECKPOINT_SCHEMA_ID
            or metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION
            or metadata["contract_version"] != G3_CONTRACT_VERSION
            or metadata["task_registry_fingerprint"] != G3_TASK_REGISTRY_FINGERPRINT
            or metadata["planned_look_registry_fingerprint"]
            != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
        ):
            raise IntegrityError("checkpoint contract identity is unsupported")
        task_id = metadata["task_id"]
        if not isinstance(task_id, str):
            raise IntegrityError("checkpoint task ID is invalid")
        spec = task_spec(task_id)
        state_schema_id = metadata["state_schema_id"]
        if state_schema_id != _state_schema_id(task_id):
            raise IntegrityError("checkpoint state schema is invalid")
        binding = _binding_from_mapping(metadata["binding"], IntegrityError)
        if identity.get("data_fingerprint") != binding.processed_data_fingerprint:
            raise IntegrityError("checkpoint processed-data bindings disagree")
        if identity.get("artifact_role") != _storage_role(task_id):
            raise IntegrityError("checkpoint storage role is inconsistent")
        dependencies = _dependency_mapping(
            metadata["dependency_fingerprints"],
            spec.dependencies,
            IntegrityError,
        )
        next_ancestry = ancestry | {fingerprint}
        loaded_dependencies = tuple(
            self._load(dependency_fingerprint, ancestry=next_ancestry)
            for _, dependency_fingerprint in dependencies
        )
        for expected_task, dependency in zip(
            spec.dependencies, loaded_dependencies, strict=True
        ):
            if dependency.task_id != expected_task or dependency.binding != binding:
                raise IntegrityError("checkpoint dependency identity is inconsistent")
        selected_k = _selected_n_states_for_task(
            task_id,
            metadata["selected_n_states"],
            loaded_dependencies,
            IntegrityError,
        )
        document_paths = tuple(
            record["relative_path"] for record in identity.get("documents", ())
        )
        if document_paths != (CHECKPOINT_STATE_PATH,):
            raise IntegrityError("checkpoint document allowlist is invalid")
        document = self._store.read_document(fingerprint, CHECKPOINT_STATE_PATH)
        _exact_keys(
            document,
            {
                "artifact_kind",
                "schema_id",
                "schema_version",
                "state_schema_id",
                "task_id",
                "binding",
                "selected_n_states",
                "dependency_fingerprints",
                "state",
            },
            "checkpoint state document",
        )
        expected_document_identity = {
            "artifact_kind": CHECKPOINT_ARTIFACT_KIND,
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_schema_id": state_schema_id,
            "task_id": task_id,
            "binding": binding.as_dict(),
            "selected_n_states": selected_k,
            "dependency_fingerprints": dict(dependencies),
        }
        for key, value in expected_document_identity.items():
            if document[key] != value:
                raise IntegrityError("checkpoint metadata/document bindings disagree")
        state = document["state"]
        if not isinstance(state, Mapping) or not state:
            raise IntegrityError("checkpoint state is empty or invalid")
        arrays = self._store.load_arrays(fingerprint)
        if not arrays:
            raise IntegrityError("checkpoint parameter arrays are empty")
        return CalibrationCheckpointReference(
            fingerprint=reference.fingerprint,
            path=reference.path,
            task_id=task_id,
            state_schema_id=state_schema_id,
            binding=binding,
            selected_n_states=selected_k,
            dependency_fingerprints=dependencies,
            arrays=arrays,
            state=state,
            storage_reference=reference,
            reused=reference.reused,
        )

    def _validated_dependencies(
        self,
        *,
        task_id: str,
        expected: tuple[str, ...],
        supplied: Sequence[CalibrationCheckpointReference],
        binding: CalibrationCheckpointBinding,
    ) -> tuple[CalibrationCheckpointReference, ...]:
        if not isinstance(supplied, Sequence):
            raise ModelError("checkpoint dependencies must be a sequence")
        values = tuple(supplied)
        if tuple(
            item.task_id
            for item in values
            if isinstance(item, CalibrationCheckpointReference)
        ) != expected or any(
            not isinstance(item, CalibrationCheckpointReference) for item in values
        ):
            raise ModelError(
                "checkpoint dependencies must match the exact task graph",
                details={"task_id": task_id, "expected": list(expected)},
            )
        verified: list[CalibrationCheckpointReference] = []
        for item in values:
            loaded = self.load(item.fingerprint)
            if loaded.task_id != item.task_id or loaded.binding != binding:
                raise ModelError("checkpoint dependency binding is inconsistent")
            verified.append(loaded)
        return tuple(verified)


def _binding(
    value: CalibrationCheckpointBinding,
    error_type: type[ModelError] | type[IntegrityError],
) -> CalibrationCheckpointBinding:
    if not isinstance(value, CalibrationCheckpointBinding):
        raise error_type("checkpoint binding has the wrong type")
    for label, fingerprint in value.as_dict().items():
        if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
            raise error_type(f"checkpoint {label.replace('_', ' ')} is invalid")
    return value


def _binding_from_mapping(
    value: object,
    error_type: type[ModelError] | type[IntegrityError],
) -> CalibrationCheckpointBinding:
    if not isinstance(value, Mapping):
        raise error_type("checkpoint binding is missing")
    _exact_keys(
        value,
        {
            "run_id",
            "config_fingerprint",
            "processed_data_fingerprint",
            "calibration_input_fingerprint",
            "preflight_fingerprint",
        },
        "checkpoint binding",
        error_type=error_type,
    )
    try:
        binding = CalibrationCheckpointBinding(
            run_id=value["run_id"],
            config_fingerprint=value["config_fingerprint"],
            processed_data_fingerprint=value["processed_data_fingerprint"],
            calibration_input_fingerprint=value["calibration_input_fingerprint"],
            preflight_fingerprint=value["preflight_fingerprint"],
        )
    except TypeError as error:  # pragma: no cover - exact keys guard construction.
        raise error_type("checkpoint binding is invalid") from error
    return _binding(binding, error_type)


def _dependency_mapping(
    value: object,
    expected: tuple[str, ...],
    error_type: type[ModelError] | type[IntegrityError],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise error_type("checkpoint dependency mapping is invalid")
    result: list[tuple[str, str]] = []
    for task_id in expected:
        fingerprint = value[task_id]
        if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
            raise error_type("checkpoint dependency fingerprint is invalid")
        result.append((task_id, fingerprint))
    return tuple(result)


def _selected_n_states_for_task(
    task_id: str,
    value: object,
    dependencies: Sequence[CalibrationCheckpointReference],
    error_type: type[ModelError] | type[IntegrityError],
) -> int | None:
    position = G3_GATE_ORDER.index(task_id)
    upstream = {
        dependency.selected_n_states
        for dependency in dependencies
        if dependency.selected_n_states is not None
    }
    if len(upstream) > 1:
        raise error_type("checkpoint dependencies disagree on selected K")
    if position < _SELECTION_POSITION:
        if value is not None:
            raise error_type("preselection checkpoint cannot contain selected K")
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (2, 3, 4, 5)
    ):
        raise error_type("checkpoint selected K must be one of 2,3,4,5")
    if task_id != _SELECTION_TASK and upstream != {value}:
        raise error_type("checkpoint cannot change or promote the selected K")
    return value


def _storage_role(task_id: str) -> Literal["design", "production"]:
    """Choose only the generic container field; checkpoint metadata is primary."""

    position = G3_GATE_ORDER.index(task_id)
    production_position = G3_GATE_ORDER.index("production_fit_same_k")
    return "production" if position >= production_position else "design"


def _state_schema_id(task_id: str) -> str:
    task_spec(task_id)
    return f"prpg-g3-{task_id}-checkpoint-state-v1"


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    *,
    error_type: type[ModelError] | type[IntegrityError] = IntegrityError,
) -> None:
    if set(value) != expected:
        raise error_type(
            f"{label} fields are invalid",
            details={
                "missing": sorted(expected - set(value)),
                "unexpected": sorted(set(value) - expected),
            },
        )


def checkpoint_dependency_map(
    value: CalibrationCheckpointReference,
) -> Mapping[str, str]:
    """Return a read-only canonical dependency mapping for evidence payloads."""

    if not isinstance(value, CalibrationCheckpointReference):
        raise ModelError("checkpoint dependency-map input has the wrong type")
    return MappingProxyType(dict(value.dependency_fingerprints))
