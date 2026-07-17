"""Sealed, store-derived scientific bindings for individual G3 tasks.

The G3 run manifest proves *which* executable/data contract was launched, but
an individual scientific computation also needs an exact statement of the
upstream checkpoints, durable sequential-look prefix, selected state count,
and scientific inputs it consumed.  This module supplies that statement
without launching a task or manufacturing scientific evidence.

Bindings are deliberately capability sealed.  Their JSON representation is
safe to embed in a calibration checkpoint, but deserializing bytes alone does
not restore authority: :func:`load_scientific_task_binding` re-verifies the
live source tree, persisted launch preflight, run store, checkpoint DAG, and
planned-look store before installing a process-local seal.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_preflight import (
    CanonicalG3Preflight,
    canonical_resume_preflight_fingerprint,
    read_persisted_canonical_g3_preflight,
    verify_compatible_canonical_g3_preflights,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
    task_spec,
)
from prpg.model.run_store import CalibrationRunIdentity, CalibrationRunStore

SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION: Final = 2
SCIENTIFIC_TASK_BINDING_SCHEMA_ID: Final = "prpg-g3-scientific-task-binding-v2"
SCIENTIFIC_TASK_SLOT_REGISTRY_SCHEMA_ID: Final = (
    "prpg-g3-scientific-task-slot-registry-v2"
)
TASK_SOURCE_MATERIALIZATION_SLOT: Final = "task_source_materialization"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SLOT = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_SELECTION_TASK: Final = "design_predictive_selection"
_SELECTION_POSITION: Final = G3_GATE_ORDER.index(_SELECTION_TASK)
_BINDING_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class ScientificTaskSlotSpec:
    """Closed fingerprint namespaces required by one registered G3 task."""

    task_id: str
    scientific_component_slots: tuple[str, ...]
    materialization_slots: tuple[str, ...]
    rng_slots: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "scientific_component_slots": list(self.scientific_component_slots),
            "materialization_slots": list(self.materialization_slots),
            "rng_slots": list(self.rng_slots),
        }


def _slot_spec(
    task_id: str,
    components: Sequence[str],
    materializations: Sequence[str],
    rng: Sequence[str],
) -> ScientificTaskSlotSpec:
    if TASK_SOURCE_MATERIALIZATION_SLOT in materializations:
        raise AssertionError("the universal source-materialization slot is implicit")
    return ScientificTaskSlotSpec(
        task_id,
        tuple(components),
        (TASK_SOURCE_MATERIALIZATION_SLOT, *tuple(materializations)),
        tuple(rng),
    )


# Slot names describe scientific identities, not implementation callables.  A
# fingerprint value normally comes from the typed executor/materialization
# that owns the named slot.  Deterministic consumers intentionally have an
# empty RNG namespace and bind their exact materialized parent instead.
SCIENTIFIC_TASK_SLOT_REGISTRY: Final = (
    _slot_spec(
        "input_integrity",
        ("input_validation_component",),
        ("calibration_input_materialization",),
        (),
    ),
    _slot_spec(
        "design_candidate_viability",
        ("candidate_viability_component",),
        (
            "design_candidate_fit_set",
            "design_candidate_suite",
            "design_candidate_common_materializations",
            "design_candidate_macro_bootstrap_materialization",
            "design_candidate_monthly_block_uniform_materialization",
            "design_candidate_daily_block_uniform_materialization",
            "design_fixed_point_viability_projection",
        ),
        (
            "design_candidate_fit_restarts",
            "design_candidate_main_and_rolling_fit_restarts",
            "design_candidate_all_registered_streams",
        ),
    ),
    _slot_spec(
        "design_predictive_selection",
        (
            "predictive_selection_component",
            "candidate_viability_parent_component",
        ),
        (
            "candidate_gate_parent_materialization",
            "rolling_score_bootstrap_materialization",
            "design_fixed_point_history_materialization",
            "design_fixed_point_final_selection_materialization",
            "design_fixed_point_final_block_parent_materialization",
        ),
        ("rolling_score_bootstrap_streams",),
    ),
    _slot_spec(
        "sealed_holdout_veto",
        (
            "sealed_holdout_veto_component",
            "predictive_selection_parent_component",
        ),
        (
            "selected_design_model_parent_materialization",
            "sealed_holdout_score_materialization",
        ),
        (),
    ),
    _slot_spec(
        "design_macro_stability",
        ("macro_stability_component",),
        ("design_model_parent_materialization", "design_macro_materialization"),
        ("design_macro_bootstrap_streams",),
    ),
    _slot_spec(
        "design_latent_correctness",
        ("latent_correctness_component",),
        ("design_model_parent_materialization",),
        ("design_latent_simulation_streams", "design_latent_bootstrap_streams"),
    ),
    _slot_spec(
        "design_refitted_adequacy",
        ("refitted_adequacy_component",),
        ("design_model_parent_materialization",),
        ("design_refit_adequacy_streams",),
    ),
    _slot_spec(
        "design_block_calibration",
        ("block_calibration_component",),
        (
            "design_model_parent_materialization",
            "design_fixed_point_final_block_parent_materialization",
            "design_block_uniform_materialization",
            "design_kernel_calibration_materialization",
            "design_transition_bootstrap_materialization",
        ),
        (
            "design_block_calibration_streams",
            "design_kernel_calibration_streams",
            "design_transition_bootstrap_streams",
        ),
    ),
    _slot_spec(
        "design_fragmentation",
        (
            "fragmentation_component",
            "design_block_calibration_parent_component",
        ),
        ("design_block_calibration_parent_materialization",),
        (),
    ),
    _slot_spec(
        "design_dependence",
        ("dependence_component",),
        (
            "design_block_calibration_parent_materialization",
            "design_dependence_uniform_materialization",
        ),
        ("design_dependence_streams",),
    ),
    _slot_spec(
        "production_fit_same_k",
        ("production_fit_same_k_component",),
        (
            "selected_design_model_parent_materialization",
            "full_calibration_input_materialization",
        ),
        ("production_fit_restarts",),
    ),
    _slot_spec(
        "production_macro_stability",
        ("macro_stability_component",),
        (
            "production_model_parent_materialization",
            "production_macro_materialization",
        ),
        ("production_macro_bootstrap_streams",),
    ),
    _slot_spec(
        "production_latent_correctness",
        ("latent_correctness_component",),
        ("production_model_parent_materialization",),
        (
            "production_latent_simulation_streams",
            "production_latent_bootstrap_streams",
        ),
    ),
    _slot_spec(
        "production_refitted_adequacy",
        ("refitted_adequacy_component",),
        ("production_model_parent_materialization",),
        ("production_refit_adequacy_streams",),
    ),
    _slot_spec(
        "production_block_calibration",
        ("block_calibration_component",),
        (
            "production_model_parent_materialization",
            "production_block_uniform_materialization",
            "production_kernel_calibration_materialization",
            "production_transition_bootstrap_materialization",
        ),
        (
            "production_block_calibration_streams",
            "production_kernel_calibration_streams",
            "production_transition_bootstrap_streams",
        ),
    ),
    _slot_spec(
        "production_fragmentation",
        (
            "fragmentation_component",
            "production_block_calibration_parent_component",
        ),
        ("production_block_calibration_parent_materialization",),
        (),
    ),
    _slot_spec(
        "production_dependence",
        ("dependence_component",),
        (
            "production_block_calibration_parent_materialization",
            "production_dependence_uniform_materialization",
        ),
        ("production_dependence_streams",),
    ),
    _slot_spec(
        "four_cell_adequacy_holm",
        (
            "adequacy_holm_component",
            "design_latent_parent_component",
            "design_refitted_parent_component",
            "production_latent_parent_component",
            "production_refitted_parent_component",
        ),
        ("four_cell_adequacy_parent_materialization",),
        (),
    ),
    _slot_spec(
        "four_cell_dependence_holm",
        (
            "dependence_holm_component",
            "design_dependence_parent_component",
            "production_dependence_parent_component",
        ),
        ("four_cell_dependence_parent_materialization",),
        (),
    ),
    _slot_spec(
        "bridge_resource_preflight",
        ("bridge_resource_component",),
        ("bridge_benchmark_materialization",),
        ("bridge_benchmark_pair_streams",),
    ),
    _slot_spec(
        "bridge_tier_a_model_dgp",
        ("bridge_tier_a_component",),
        ("bridge_tier_a_model_materialization",),
        ("bridge_tier_a_model_streams",),
    ),
    _slot_spec(
        "bridge_tier_a_nonparametric_dgp",
        ("bridge_tier_a_component",),
        ("bridge_tier_a_nonparametric_materialization",),
        ("bridge_tier_a_nonparametric_streams",),
    ),
    _slot_spec(
        "bridge_tier_b_model_dgp",
        ("bridge_tier_b_component",),
        ("bridge_tier_b_model_materialization",),
        ("bridge_tier_b_model_streams",),
    ),
    _slot_spec(
        "bridge_tier_b_nonparametric_dgp",
        ("bridge_tier_b_component",),
        ("bridge_tier_b_nonparametric_materialization",),
        ("bridge_tier_b_nonparametric_streams",),
    ),
    _slot_spec(
        "actual_artifact_bridge",
        ("actual_artifact_bridge_component",),
        ("actual_artifact_history_materialization",),
        (),
    ),
    _slot_spec(
        "artifact_integrity",
        (
            "artifact_integrity_component",
            "actual_artifact_bridge_parent_component",
        ),
        ("final_scientific_artifact_parent_materialization",),
        (),
    ),
)

SCIENTIFIC_TASK_SLOT_BY_ID: Final = MappingProxyType(
    {item.task_id: item for item in SCIENTIFIC_TASK_SLOT_REGISTRY}
)


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


SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT: Final = hashlib.sha256(
    _canonical_bytes(
        {
            "schema_id": SCIENTIFIC_TASK_SLOT_REGISTRY_SCHEMA_ID,
            "tasks": [item.as_dict() for item in SCIENTIFIC_TASK_SLOT_REGISTRY],
        }
    )
).hexdigest()


@dataclass(frozen=True, slots=True)
class ScientificTaskBinding:
    """Immutable exact task identity with non-serializable builder authority."""

    schema_id: str
    schema_version: int
    run_id: str
    run_identity: CalibrationRunIdentity
    launch_preflight_fingerprint: str
    persisted_preflight_receipt_fingerprint: str
    compatible_preflight_fingerprint: str
    task_id: str
    selected_n_states: int | None
    dependency_checkpoint_fingerprints: tuple[tuple[str, str], ...]
    owned_planned_look_fingerprints: tuple[tuple[str, tuple[str, ...]], ...]
    scientific_component_fingerprints: tuple[tuple[str, str], ...]
    materialization_fingerprints: tuple[tuple[str, str], ...]
    rng_fingerprints: tuple[tuple[str, str], ...]
    slot_registry_fingerprint: str
    fingerprint: str
    generation_authorized: Literal[False] = False
    _authority_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def identity_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe content addressed by ``fingerprint``."""

        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "contract_version": G3_CONTRACT_VERSION,
            "run_id": self.run_id,
            "run_identity": self.run_identity.as_dict(),
            "launch_preflight_fingerprint": self.launch_preflight_fingerprint,
            "persisted_preflight_receipt_fingerprint": (
                self.persisted_preflight_receipt_fingerprint
            ),
            "compatible_preflight_fingerprint": (self.compatible_preflight_fingerprint),
            "task_id": self.task_id,
            "selected_n_states": self.selected_n_states,
            "dependency_checkpoint_fingerprints": dict(
                self.dependency_checkpoint_fingerprints
            ),
            "owned_planned_look_fingerprints": {
                family_id: list(fingerprints)
                for family_id, fingerprints in self.owned_planned_look_fingerprints
            },
            "scientific_component_fingerprints": dict(
                self.scientific_component_fingerprints
            ),
            "materialization_fingerprints": dict(self.materialization_fingerprints),
            "rng_fingerprints": dict(self.rng_fingerprints),
            "slot_registry_fingerprint": self.slot_registry_fingerprint,
            "task_registry_fingerprint": self.run_identity.task_registry_fingerprint,
            "planned_look_registry_fingerprint": (
                self.run_identity.planned_look_registry_fingerprint
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the complete serialized representation for checkpoint embedding."""

        return {**self.identity_dict(), "binding_fingerprint": self.fingerprint}


def scientific_task_slot_spec(task_id: str) -> ScientificTaskSlotSpec:
    """Resolve a task's exact code-owned fingerprint namespaces."""

    task_spec(task_id)
    return SCIENTIFIC_TASK_SLOT_BY_ID[task_id]


def build_scientific_task_binding(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    run_id: str,
    task_id: str,
    live_preflight: CanonicalG3Preflight,
    scientific_component_fingerprints: Mapping[str, str],
    materialization_fingerprints: Mapping[str, str],
    rng_fingerprints: Mapping[str, str],
) -> ScientificTaskBinding:
    """Derive and seal one task binding from verified durable state.

    Dependency, selected-K, source/lock, registry, launch-preflight, and
    planned-look values are never accepted from the caller.  The caller only
    supplies fingerprints for the exact closed scientific slots owned by the
    registered task.
    """

    binding = _derive_binding(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        task_id=task_id,
        live_preflight=live_preflight,
        scientific_component_fingerprints=scientific_component_fingerprints,
        materialization_fingerprints=materialization_fingerprints,
        rng_fingerprints=rng_fingerprints,
    )
    object.__setattr__(binding, "_authority_seal", _BINDING_CAPABILITY)
    return binding


def verify_scientific_task_binding(
    value: ScientificTaskBinding,
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
) -> None:
    """Re-derive a sealed binding against fresh source and current stores."""

    if not isinstance(value, ScientificTaskBinding):
        raise ModelError("scientific task binding has the wrong type")
    if value._authority_seal is not _BINDING_CAPABILITY:
        raise ModelError("scientific task binding lacks builder authority")
    _verify_binding_hash(value, ModelError)
    expected = _derive_binding(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=value.run_id,
        task_id=value.task_id,
        live_preflight=live_preflight,
        scientific_component_fingerprints=dict(value.scientific_component_fingerprints),
        materialization_fingerprints=dict(value.materialization_fingerprints),
        rng_fingerprints=dict(value.rng_fingerprints),
    )
    if value.identity_dict() != expected.identity_dict():
        raise IntegrityError("scientific task binding differs from durable state")


def load_scientific_task_binding(
    value: Mapping[str, Any],
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
) -> ScientificTaskBinding:
    """Parse serialized bytes and restore authority only after full revalidation."""

    parsed = _binding_from_mapping(value)
    _verify_binding_hash(parsed, IntegrityError)
    expected = _derive_binding(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        run_id=parsed.run_id,
        task_id=parsed.task_id,
        live_preflight=live_preflight,
        scientific_component_fingerprints=dict(
            parsed.scientific_component_fingerprints
        ),
        materialization_fingerprints=dict(parsed.materialization_fingerprints),
        rng_fingerprints=dict(parsed.rng_fingerprints),
    )
    if parsed.identity_dict() != expected.identity_dict():
        raise IntegrityError("serialized scientific task binding is stale or forged")
    object.__setattr__(parsed, "_authority_seal", _BINDING_CAPABILITY)
    return parsed


def _derive_binding(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    run_id: str,
    task_id: str,
    live_preflight: CanonicalG3Preflight,
    scientific_component_fingerprints: Mapping[str, str],
    materialization_fingerprints: Mapping[str, str],
    rng_fingerprints: Mapping[str, str],
) -> ScientificTaskBinding:
    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("scientific task binding requires a CalibrationRunStore")
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("scientific task binding requires a checkpoint store")
    spec = task_spec(task_id)
    slot_spec = scientific_task_slot_spec(task_id)
    components = _fingerprint_slots(
        scientific_component_fingerprints,
        slot_spec.scientific_component_slots,
        "scientific component",
        ModelError,
    )
    materializations = _fingerprint_slots(
        materialization_fingerprints,
        slot_spec.materialization_slots,
        "materialization",
        ModelError,
    )
    rng = _fingerprint_slots(
        rng_fingerprints,
        slot_spec.rng_slots,
        "RNG",
        ModelError,
    )

    reference = run_store.verify(run_id)
    persisted = read_persisted_canonical_g3_preflight(run_store, run_id=run_id)
    launch_fingerprint = canonical_resume_preflight_fingerprint(
        run_store,
        run_id=run_id,
        live_preflight=live_preflight,
    )
    if launch_fingerprint != persisted.preflight.fingerprint:
        raise IntegrityError("persisted and launch preflight fingerprints disagree")
    verify_compatible_canonical_g3_preflights(
        persisted.preflight,
        live_preflight,
    )
    _verify_run_identity(reference.identity, live_preflight)

    dependencies = []
    selected_values: set[int] = set()
    for dependency_task in spec.dependencies:
        checkpoint = checkpoint_store.load_from_task_result(
            run_store,
            run_id=run_id,
            task_id=dependency_task,
        )
        dependencies.append((dependency_task, checkpoint.fingerprint))
        if checkpoint.selected_n_states is not None:
            selected_values.add(checkpoint.selected_n_states)
    selected_n_states = _selected_n_states(
        task_id,
        selected_values,
        has_dependencies=bool(spec.dependencies),
    )

    all_looks = run_store.verify_planned_look_receipts(
        run_id,
        require_terminal=False,
    )
    owned_looks = tuple(
        (
            look.family_id,
            tuple(item.fingerprint for item in all_looks.get(look.family_id, ())),
        )
        for look in G3_PLANNED_LOOK_REGISTRY
        if look.owner_task_id == task_id
    )
    compatible_preflight_fingerprint = _compatible_preflight_fingerprint(live_preflight)
    provisional = ScientificTaskBinding(
        schema_id=SCIENTIFIC_TASK_BINDING_SCHEMA_ID,
        schema_version=SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION,
        run_id=reference.run_id,
        run_identity=reference.identity,
        launch_preflight_fingerprint=launch_fingerprint,
        persisted_preflight_receipt_fingerprint=(persisted.receipt_fingerprint),
        compatible_preflight_fingerprint=compatible_preflight_fingerprint,
        task_id=task_id,
        selected_n_states=selected_n_states,
        dependency_checkpoint_fingerprints=tuple(dependencies),
        owned_planned_look_fingerprints=owned_looks,
        scientific_component_fingerprints=components,
        materialization_fingerprints=materializations,
        rng_fingerprints=rng,
        slot_registry_fingerprint=SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT,
        fingerprint="",
    )
    fingerprint = hashlib.sha256(
        _canonical_bytes(provisional.identity_dict())
    ).hexdigest()
    return ScientificTaskBinding(
        schema_id=provisional.schema_id,
        schema_version=provisional.schema_version,
        run_id=provisional.run_id,
        run_identity=provisional.run_identity,
        launch_preflight_fingerprint=provisional.launch_preflight_fingerprint,
        persisted_preflight_receipt_fingerprint=(
            provisional.persisted_preflight_receipt_fingerprint
        ),
        compatible_preflight_fingerprint=(provisional.compatible_preflight_fingerprint),
        task_id=provisional.task_id,
        selected_n_states=provisional.selected_n_states,
        dependency_checkpoint_fingerprints=(
            provisional.dependency_checkpoint_fingerprints
        ),
        owned_planned_look_fingerprints=(provisional.owned_planned_look_fingerprints),
        scientific_component_fingerprints=(
            provisional.scientific_component_fingerprints
        ),
        materialization_fingerprints=provisional.materialization_fingerprints,
        rng_fingerprints=provisional.rng_fingerprints,
        slot_registry_fingerprint=provisional.slot_registry_fingerprint,
        fingerprint=fingerprint,
    )


def _binding_from_mapping(value: Mapping[str, Any]) -> ScientificTaskBinding:
    if not isinstance(value, Mapping):
        raise IntegrityError("serialized scientific task binding must be a mapping")
    expected = {
        "schema_id",
        "schema_version",
        "contract_version",
        "run_id",
        "run_identity",
        "launch_preflight_fingerprint",
        "persisted_preflight_receipt_fingerprint",
        "compatible_preflight_fingerprint",
        "task_id",
        "selected_n_states",
        "dependency_checkpoint_fingerprints",
        "owned_planned_look_fingerprints",
        "scientific_component_fingerprints",
        "materialization_fingerprints",
        "rng_fingerprints",
        "slot_registry_fingerprint",
        "task_registry_fingerprint",
        "planned_look_registry_fingerprint",
        "binding_fingerprint",
    }
    _exact_keys(value, expected, "serialized scientific task binding")
    if (
        value["schema_id"] != SCIENTIFIC_TASK_BINDING_SCHEMA_ID
        or value["schema_version"] != SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION
        or value["contract_version"] != G3_CONTRACT_VERSION
        or value["slot_registry_fingerprint"]
        != SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT
        or value["task_registry_fingerprint"] != G3_TASK_REGISTRY_FINGERPRINT
        or value["planned_look_registry_fingerprint"]
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise IntegrityError("scientific task binding contract is unsupported")
    run_identity = _run_identity_from_mapping(value["run_identity"])
    if (
        value["task_registry_fingerprint"] != run_identity.task_registry_fingerprint
        or value["planned_look_registry_fingerprint"]
        != run_identity.planned_look_registry_fingerprint
    ):
        raise IntegrityError("scientific task binding registry identities disagree")
    task_id = value["task_id"]
    if not isinstance(task_id, str):
        raise IntegrityError("scientific task binding task ID is invalid")
    spec = task_spec(task_id)
    slot_spec = scientific_task_slot_spec(task_id)
    dependencies = _ordered_fingerprint_mapping(
        value["dependency_checkpoint_fingerprints"],
        spec.dependencies,
        "dependency checkpoint",
    )
    owned_family_ids = tuple(
        item.family_id
        for item in G3_PLANNED_LOOK_REGISTRY
        if item.owner_task_id == task_id
    )
    owned_looks = _look_mapping(
        value["owned_planned_look_fingerprints"],
        owned_family_ids,
    )
    components = _fingerprint_slots(
        value["scientific_component_fingerprints"],
        slot_spec.scientific_component_slots,
        "scientific component",
        IntegrityError,
    )
    materializations = _fingerprint_slots(
        value["materialization_fingerprints"],
        slot_spec.materialization_slots,
        "materialization",
        IntegrityError,
    )
    rng = _fingerprint_slots(
        value["rng_fingerprints"],
        slot_spec.rng_slots,
        "RNG",
        IntegrityError,
    )
    selected = value["selected_n_states"]
    if selected is not None and (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in {2, 3, 4, 5}
    ):
        raise IntegrityError("scientific task binding selected K is invalid")
    fingerprint_fields = (
        "run_id",
        "launch_preflight_fingerprint",
        "persisted_preflight_receipt_fingerprint",
        "compatible_preflight_fingerprint",
        "slot_registry_fingerprint",
        "binding_fingerprint",
    )
    if any(not _is_fingerprint(value[key]) for key in fingerprint_fields):
        raise IntegrityError("scientific task binding fingerprint field is invalid")
    return ScientificTaskBinding(
        schema_id=value["schema_id"],
        schema_version=value["schema_version"],
        run_id=value["run_id"],
        run_identity=run_identity,
        launch_preflight_fingerprint=value["launch_preflight_fingerprint"],
        persisted_preflight_receipt_fingerprint=(
            value["persisted_preflight_receipt_fingerprint"]
        ),
        compatible_preflight_fingerprint=value["compatible_preflight_fingerprint"],
        task_id=task_id,
        selected_n_states=selected,
        dependency_checkpoint_fingerprints=dependencies,
        owned_planned_look_fingerprints=owned_looks,
        scientific_component_fingerprints=components,
        materialization_fingerprints=materializations,
        rng_fingerprints=rng,
        slot_registry_fingerprint=value["slot_registry_fingerprint"],
        fingerprint=value["binding_fingerprint"],
    )


def _run_identity_from_mapping(value: object) -> CalibrationRunIdentity:
    if not isinstance(value, Mapping):
        raise IntegrityError("scientific task run identity is missing")
    expected = {
        "config_fingerprint",
        "processed_data_fingerprint",
        "calibration_input_fingerprint",
        "source_code_fingerprint",
        "dependency_lock_fingerprint",
        "contract_version",
        "task_registry_fingerprint",
        "planned_look_registry_fingerprint",
    }
    _exact_keys(value, expected, "scientific task run identity")
    try:
        identity = CalibrationRunIdentity(**dict(value))
    except TypeError as error:  # pragma: no cover - exact-key guard.
        raise IntegrityError("scientific task run identity is invalid") from error
    if (
        any(
            not _is_fingerprint(getattr(identity, field_name))
            for field_name in (
                "config_fingerprint",
                "processed_data_fingerprint",
                "calibration_input_fingerprint",
                "source_code_fingerprint",
                "dependency_lock_fingerprint",
                "task_registry_fingerprint",
                "planned_look_registry_fingerprint",
            )
        )
        or identity.contract_version != G3_CONTRACT_VERSION
    ):
        raise IntegrityError("scientific task run identity fields are invalid")
    return identity


def _verify_run_identity(
    identity: CalibrationRunIdentity,
    live_preflight: CanonicalG3Preflight,
) -> None:
    if (
        identity.config_fingerprint != live_preflight.config_fingerprint
        or identity.processed_data_fingerprint
        != live_preflight.processed_data_fingerprint
        or identity.calibration_input_fingerprint
        != live_preflight.calibration_input_fingerprint
        or identity.source_code_fingerprint != live_preflight.source_code_fingerprint
        or identity.dependency_lock_fingerprint
        != live_preflight.dependency_lock_fingerprint
        or identity.contract_version != G3_CONTRACT_VERSION
        or identity.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT
        or identity.planned_look_registry_fingerprint
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    ):
        raise IntegrityError("scientific task run/preflight identities disagree")


def _selected_n_states(
    task_id: str,
    dependency_values: set[int],
    *,
    has_dependencies: bool,
) -> int | None:
    """Return only K frozen on task entry; selection's newly chosen K is output."""

    position = G3_GATE_ORDER.index(task_id)
    if position <= _SELECTION_POSITION:
        if dependency_values:
            raise IntegrityError("preselection task received a selected K")
        return None
    if not has_dependencies or len(dependency_values) != 1:
        raise IntegrityError("postselection dependencies do not freeze one selected K")
    selected = next(iter(dependency_values))
    if selected not in {2, 3, 4, 5}:  # pragma: no cover - checkpoint loader guards.
        raise IntegrityError("postselection dependency selected K is invalid")
    return selected


def _compatible_preflight_fingerprint(value: CanonicalG3Preflight) -> str:
    identity = value.identity_dict()
    resources = identity.get("resources")
    if not isinstance(resources, Mapping):  # pragma: no cover - preflight guards.
        raise IntegrityError("canonical preflight resources are invalid")
    stable_resources = dict(resources)
    stable_resources.pop("disk_free_bytes", None)
    identity["resources"] = stable_resources
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _fingerprint_slots(
    value: object,
    expected: tuple[str, ...],
    label: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        missing = (
            sorted(set(expected) - set(value)) if isinstance(value, Mapping) else []
        )
        unexpected = (
            sorted(str(item) for item in set(value) - set(expected))
            if isinstance(value, Mapping)
            else []
        )
        raise error_type(
            f"scientific task {label} slots are invalid",
            details={"missing": missing, "unexpected": unexpected},
        )
    result = []
    for slot in expected:
        fingerprint = value[slot]
        if not _is_fingerprint(fingerprint):
            raise error_type(f"scientific task {label} fingerprint is invalid")
        result.append((slot, fingerprint))
    return tuple(result)


def _ordered_fingerprint_mapping(
    value: object,
    expected: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise IntegrityError(f"scientific task {label} mapping is invalid")
    result = []
    for key in expected:
        fingerprint = value[key]
        if not _is_fingerprint(fingerprint):
            raise IntegrityError(f"scientific task {label} fingerprint is invalid")
        result.append((key, fingerprint))
    return tuple(result)


def _look_mapping(
    value: object,
    expected_families: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, Mapping) or set(value) != set(expected_families):
        raise IntegrityError("scientific task planned-look mapping is invalid")
    result = []
    for family_id in expected_families:
        sequence = value[family_id]
        if (
            isinstance(sequence, str | bytes)
            or not isinstance(sequence, Sequence)
            or any(not _is_fingerprint(item) for item in sequence)
        ):
            raise IntegrityError("scientific task planned-look chain is invalid")
        result.append((family_id, tuple(sequence)))
    return tuple(result)


def _verify_binding_hash(
    value: ScientificTaskBinding,
    error_type: type[ModelError] | type[IntegrityError],
) -> None:
    if (
        value.schema_id != SCIENTIFIC_TASK_BINDING_SCHEMA_ID
        or value.schema_version != SCIENTIFIC_TASK_BINDING_SCHEMA_VERSION
        or value.slot_registry_fingerprint != SCIENTIFIC_TASK_SLOT_REGISTRY_FINGERPRINT
        or value.generation_authorized is not False
    ):
        raise error_type("scientific task binding contract fields are invalid")
    expected = hashlib.sha256(_canonical_bytes(value.identity_dict())).hexdigest()
    if value.fingerprint != expected:
        raise error_type("scientific task binding hash is inconsistent")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise IntegrityError(
            f"{label} fields are invalid",
            details={
                "missing": sorted(expected - set(value)),
                "unexpected": sorted(str(item) for item in set(value) - expected),
            },
        )


def _is_fingerprint(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_slot_registry() -> None:
    if tuple(item.task_id for item in SCIENTIFIC_TASK_SLOT_REGISTRY) != G3_GATE_ORDER:
        raise RuntimeError("scientific task slot registry differs from G3 task order")
    if len(SCIENTIFIC_TASK_SLOT_BY_ID) != len(G3_GATE_ORDER):
        raise RuntimeError("scientific task slot registry contains duplicate tasks")
    for item in SCIENTIFIC_TASK_SLOT_REGISTRY:
        namespaces = (
            item.scientific_component_slots,
            item.materialization_slots,
            item.rng_slots,
        )
        for namespace in namespaces:
            if len(namespace) != len(set(namespace)) or any(
                _SLOT.fullmatch(slot) is None for slot in namespace
            ):
                raise RuntimeError("scientific task slot registry is invalid")
        if not item.scientific_component_slots or not item.materialization_slots:
            raise RuntimeError(
                "every scientific task requires component/materialization"
            )
        if len(set().union(*map(set, namespaces))) != sum(map(len, namespaces)):
            raise RuntimeError("scientific task slot namespaces overlap")
    required_parent_slots = {
        "design_fragmentation": "design_block_calibration_parent_component",
        "production_fragmentation": "production_block_calibration_parent_component",
        "four_cell_adequacy_holm": "design_latent_parent_component",
        "four_cell_dependence_holm": "design_dependence_parent_component",
    }
    for task_id, parent_slot in required_parent_slots.items():
        if (
            parent_slot
            not in SCIENTIFIC_TASK_SLOT_BY_ID[task_id].scientific_component_slots
        ):
            raise RuntimeError("duplicated-source task lacks explicit parent component")


_validate_slot_registry()
