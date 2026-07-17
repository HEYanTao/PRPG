"""Sealed run/task views for the remaining Phase-3 publication boundaries.

Scientific executors produce the evidence records in this module before a
task binding exists.  A builder then revalidates the complete live evidence,
binds it to one run and one exact G3 task, and mints an object-specific view.
Compatibility bridge families remain report-only and cannot satisfy the
canonical type guard.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal, TypeAlias, TypeGuard

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    registered_block_calibration_identity,
    registered_block_source_episode_counts,
)
from prpg.model.bridge_driver import (
    BridgeDriverResult,
    CanonicalBridgeRequest,
    ReverifiedBridgeDriverEvidence,
    StagedBridgePlannedLookEvidence,
    StagedBridgeTaskEvidence,
    reverify_bridge_driver_result,
    staged_bridge_planned_look_identity,
    staged_bridge_task_evidence_identity,
    staged_bridge_task_planned_looks,
)
from prpg.model.g3_preflight import (
    CanonicalG3Preflight,
    verify_compatible_canonical_g3_preflights,
)
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    kernel_calibration_execution_identity,
)
from prpg.model.materialization_identity import scientific_array_fingerprint
from prpg.model.regime_policy import (
    OwnerFixedK4RecurringHistoryFailure,
    owner_fixed_k4_recurring_history_support,
)
from prpg.model.registered_fixed_point import (
    registered_final_block_projection_identity,
)
from prpg.model.scientific_artifact import (
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    is_verified_scientific_model_artifact,
)
from prpg.model.scientific_artifact_pair import (
    LoadedScientificArtifactPair,
    is_verified_scientific_artifact_pair,
    scientific_artifact_pair_identity,
)
from prpg.model.transition_bootstrap import (
    TransitionBootstrapAggregate,
    transition_bootstrap_aggregate_identity,
)

ArtifactRole: TypeAlias = Literal["design", "production"]
BridgeTaskId: TypeAlias = Literal[
    "bridge_resource_preflight",
    "bridge_tier_a_model_dgp",
    "bridge_tier_a_nonparametric_dgp",
    "bridge_tier_b_model_dgp",
    "bridge_tier_b_nonparametric_dgp",
    "actual_artifact_bridge",
]
BOUNDARY_VIEW_CONTRACT = "g3-boundary-task-views-v1"
BRIDGE_TASK_IDS: tuple[BridgeTaskId, ...] = (
    "bridge_resource_preflight",
    "bridge_tier_a_model_dgp",
    "bridge_tier_a_nonparametric_dgp",
    "bridge_tier_b_model_dgp",
    "bridge_tier_b_nonparametric_dgp",
    "actual_artifact_bridge",
)
_BRIDGE_DGP_IDS: tuple[Literal["model", "nonparametric"], ...] = (
    "model",
    "nonparametric",
)
_STAGED_BRIDGE_EXPECTED_PARENTS: Mapping[BridgeTaskId, tuple[BridgeTaskId, ...]] = (
    MappingProxyType(
        {
            "bridge_resource_preflight": (),
            "bridge_tier_a_model_dgp": ("bridge_resource_preflight",),
            "bridge_tier_a_nonparametric_dgp": ("bridge_resource_preflight",),
            "bridge_tier_b_model_dgp": (
                "bridge_resource_preflight",
                "bridge_tier_a_model_dgp",
                "bridge_tier_a_nonparametric_dgp",
            ),
            "bridge_tier_b_nonparametric_dgp": (
                "bridge_resource_preflight",
                "bridge_tier_a_model_dgp",
                "bridge_tier_a_nonparametric_dgp",
            ),
            "actual_artifact_bridge": (
                "bridge_tier_a_model_dgp",
                "bridge_tier_a_nonparametric_dgp",
                "bridge_tier_b_model_dgp",
                "bridge_tier_b_nonparametric_dgp",
            ),
        }
    )
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INPUT_CAPABILITY = object()
_BLOCK_CAPABILITY = object()
_ARTIFACT_CAPABILITY = object()
_BRIDGE_CAPABILITY = object()
_MINTED_INPUT: dict[int, InputIntegrityTaskView] = {}
_MINTED_BLOCK: dict[int, BlockCalibrationTaskView] = {}
_MINTED_ARTIFACT: dict[int, ArtifactIntegrityTaskView] = {}
_MINTED_BRIDGE: dict[int, BridgeDriverTaskView] = {}


@dataclass(frozen=True, slots=True)
class InputIntegrityDecisionEvidence:
    """Precomputed launch/live preflights and exact artifact binding."""

    launch_preflight: CanonicalG3Preflight
    live_preflight: CanonicalG3Preflight
    binding: ScientificArtifactBinding


@dataclass(frozen=True, slots=True)
class BlockCalibrationGateEvidence:
    """All block, kernel, and transition evidence for one artifact."""

    role: ArtifactRole
    selected_k: int
    monthly: RegisteredBlockCalibration
    daily: RegisteredBlockCalibration
    monthly_kernel: KernelCalibrationExecution
    daily_kernel: KernelCalibrationExecution
    transition: TransitionBootstrapAggregate


@dataclass(frozen=True, slots=True)
class ArtifactIntegrityDecisionEvidence:
    """The exact logical pair receipt checked by the final G3 task."""

    pair: LoadedScientificArtifactPair

    @property
    def design(self) -> LoadedScientificModelArtifact:
        return self.pair.design

    @property
    def production(self) -> LoadedScientificModelArtifact:
        return self.pair.production


@dataclass(frozen=True, slots=True, init=False)
class InputIntegrityTaskView:
    contract_version: str
    task_id: Literal["input_integrity"]
    run_id: str
    binding_fingerprint: str
    source: InputIntegrityDecisionEvidence
    launch_preflight_fingerprint: str
    live_preflight_fingerprint: str
    calibration_input_fingerprint: str
    source_fingerprint: str
    view_fingerprint: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("InputIntegrityTaskView is created only by its builder")


@dataclass(frozen=True, slots=True, init=False)
class BlockCalibrationTaskView:
    contract_version: str
    task_id: Literal["design_block_calibration", "production_block_calibration"]
    role: ArtifactRole
    run_id: str
    binding_fingerprint: str
    source: BlockCalibrationGateEvidence
    monthly_calibration_fingerprint: str
    daily_calibration_fingerprint: str
    monthly_kernel_fingerprint: str
    daily_kernel_fingerprint: str
    transition_fingerprint: str
    monthly_kernel_parent_fingerprint: str
    daily_kernel_parent_fingerprint: str
    monthly_kernel_materialization_fingerprint: str
    daily_kernel_materialization_fingerprint: str
    kernel_core_model_fingerprint: str
    kernel_calibration_rng_fingerprint: str
    transition_parent_model_fingerprint: str
    transition_materialization_fingerprint: str
    transition_rng_fingerprint: str
    design_fixed_point_fingerprint: str | None
    design_fixed_point_iteration_history_fingerprint: str | None
    design_fixed_point_final_selection_fingerprint: str | None
    design_fixed_point_final_block_parent_fingerprint: str | None
    parent_model_fingerprint: str
    block_uniform_materialization_fingerprint: str
    block_calibration_rng_fingerprint: str
    monthly_source_episode_counts: tuple[int, ...]
    daily_source_episode_counts: tuple[int, ...]
    recurring_history_support_fingerprint: str | None
    source_fingerprint: str
    view_fingerprint: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("BlockCalibrationTaskView is created only by its builder")


@dataclass(frozen=True, slots=True, init=False)
class ArtifactIntegrityTaskView:
    contract_version: str
    task_id: Literal["artifact_integrity"]
    run_id: str
    binding_fingerprint: str
    source: ArtifactIntegrityDecisionEvidence
    design_source_fingerprint: str
    production_source_fingerprint: str
    design_reference_fingerprint: str
    production_reference_fingerprint: str
    design_core_model_fingerprint: str
    production_core_model_fingerprint: str
    design_parameter_set_fingerprint: str
    production_parameter_set_fingerprint: str
    pair_receipt_fingerprint: str
    design_role_source_fingerprint: str
    production_role_source_fingerprint: str
    report_set_fingerprint: str
    artifact_binding_fingerprint: str
    provenance_fingerprint: str
    actual_bridge_parent_view_fingerprint: str
    final_artifact_parent_fingerprint: str
    source_fingerprint: str
    view_fingerprint: str
    _actual_bridge_parent: BridgeDriverTaskView = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("ArtifactIntegrityTaskView is created only by its builder")


@dataclass(frozen=True, slots=True, init=False)
class BridgeDriverTaskView:
    """One separately bound task view over exact staged bridge evidence."""

    contract_version: str
    task_id: BridgeTaskId
    run_id: str
    binding_fingerprint: str
    source: StagedBridgeTaskEvidence | ReverifiedBridgeDriverEvidence
    base_preflight_fingerprint: str
    selected_k: int
    shared_binding_fingerprint: str
    parent_result_fingerprints: tuple[tuple[str, str], ...]
    task_result_fingerprint: str
    source_lineage_fingerprint: str
    materialization_fingerprint: str
    known_trace_fingerprint: str | None
    rng_fingerprint: str | None
    planned_looks: tuple[StagedBridgePlannedLookEvidence, ...]
    planned_look_fingerprints: tuple[str, ...]
    view_fingerprint: str
    canonical: bool
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("BridgeDriverTaskView is created only by its builder")

    @property
    def generation_authorized(self) -> Literal[False]:
        return False


@dataclass(frozen=True, slots=True)
class BridgeDriverTaskViews:
    bridge_resource_preflight: BridgeDriverTaskView
    bridge_tier_a_model_dgp: BridgeDriverTaskView
    bridge_tier_a_nonparametric_dgp: BridgeDriverTaskView
    bridge_tier_b_model_dgp: BridgeDriverTaskView
    bridge_tier_b_nonparametric_dgp: BridgeDriverTaskView
    actual_artifact_bridge: BridgeDriverTaskView

    def as_mapping(self) -> Mapping[str, BridgeDriverTaskView]:
        return MappingProxyType(
            {task_id: getattr(self, task_id) for task_id in BRIDGE_TASK_IDS}
        )


def build_input_integrity_task_view(
    source: InputIntegrityDecisionEvidence,
    *,
    run_id: str,
    binding_fingerprint: str,
) -> InputIntegrityTaskView:
    _verify_input_source(source, run_id=run_id)
    return _mint_input(source, run_id, binding_fingerprint)


def build_block_calibration_task_view(
    source: BlockCalibrationGateEvidence,
    *,
    task_id: Literal["design_block_calibration", "production_block_calibration"],
    run_id: str,
    binding_fingerprint: str,
) -> BlockCalibrationTaskView:
    identities = _block_source_identities(source, task_id=task_id)
    _require_sha(run_id, "block task run ID")
    _require_sha(binding_fingerprint, "block task binding")
    payload = {
        "contract_version": BOUNDARY_VIEW_CONTRACT,
        "task_id": task_id,
        "role": source.role,
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        **identities,
    }
    view = object.__new__(BlockCalibrationTaskView)
    for name, value in (
        ("contract_version", BOUNDARY_VIEW_CONTRACT),
        ("task_id", task_id),
        ("role", source.role),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("source", source),
        *tuple(identities.items()),
        ("view_fingerprint", _fingerprint(payload)),
        ("_seal", _BLOCK_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_BLOCK[id(view)] = view
    return view


def build_artifact_integrity_task_view(
    source: ArtifactIntegrityDecisionEvidence,
    actual_bridge_parent: BridgeDriverTaskView,
    *,
    run_id: str,
    binding_fingerprint: str,
) -> ArtifactIntegrityTaskView:
    if (
        not is_bridge_driver_task_view(actual_bridge_parent)
        or actual_bridge_parent.task_id != "actual_artifact_bridge"
        or actual_bridge_parent.run_id != run_id
        or actual_bridge_parent.binding_fingerprint == binding_fingerprint
    ):
        raise ModelError("artifact integrity requires its exact actual-bridge parent")
    identities = _artifact_source_identities(
        source,
        run_id=run_id,
        actual_bridge_parent_view_fingerprint=(actual_bridge_parent.view_fingerprint),
    )
    _require_sha(binding_fingerprint, "artifact-integrity task binding")
    payload = {
        "contract_version": BOUNDARY_VIEW_CONTRACT,
        "task_id": "artifact_integrity",
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        **identities,
    }
    view = object.__new__(ArtifactIntegrityTaskView)
    for name, value in (
        ("contract_version", BOUNDARY_VIEW_CONTRACT),
        ("task_id", "artifact_integrity"),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("source", source),
        *tuple(identities.items()),
        ("view_fingerprint", _fingerprint(payload)),
        ("_actual_bridge_parent", actual_bridge_parent),
        ("_seal", _ARTIFACT_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_ARTIFACT[id(view)] = view
    return view


def build_bridge_driver_task_view(
    source: StagedBridgeTaskEvidence,
    *,
    task_id: BridgeTaskId,
    run_id: str,
    binding_fingerprint: str,
) -> BridgeDriverTaskView:
    if task_id not in BRIDGE_TASK_IDS:
        raise ModelError("bridge task ID is not registered")
    _require_sha(run_id, "bridge task run ID")
    _require_sha(binding_fingerprint, "bridge task binding")
    identities = _staged_bridge_source_identities(source, task_id=task_id)
    planned_looks = staged_bridge_task_planned_looks(source)
    return _mint_bridge(
        source,
        task_id=task_id,
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        identities=identities,
        planned_looks=planned_looks,
        canonical=True,
    )


def bridge_driver_task_views(
    result: BridgeDriverResult,
    request: CanonicalBridgeRequest,
) -> BridgeDriverTaskViews:
    """Return legacy all-at-once views that canonical guards reject."""

    source = reverify_bridge_driver_result(result, request)
    views = {
        task_id: _mint_bridge(
            source,
            task_id=task_id,
            run_id="0" * 64,
            binding_fingerprint=_fingerprint({"legacy_task_id": task_id}),
            identities=_legacy_bridge_source_identities(source, task_id=task_id),
            planned_looks=(),
            canonical=False,
        )
        for task_id in BRIDGE_TASK_IDS
    }
    return BridgeDriverTaskViews(
        bridge_resource_preflight=views["bridge_resource_preflight"],
        bridge_tier_a_model_dgp=views["bridge_tier_a_model_dgp"],
        bridge_tier_a_nonparametric_dgp=(views["bridge_tier_a_nonparametric_dgp"]),
        bridge_tier_b_model_dgp=views["bridge_tier_b_model_dgp"],
        bridge_tier_b_nonparametric_dgp=(views["bridge_tier_b_nonparametric_dgp"]),
        actual_artifact_bridge=views["actual_artifact_bridge"],
    )


def is_input_integrity_task_view(value: object) -> TypeGuard[InputIntegrityTaskView]:
    try:
        _verify_input_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_block_calibration_task_view(
    value: object,
) -> TypeGuard[BlockCalibrationTaskView]:
    try:
        _verify_block_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_artifact_integrity_task_view(
    value: object,
) -> TypeGuard[ArtifactIntegrityTaskView]:
    try:
        _verify_artifact_view(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def is_bridge_driver_task_view(value: object) -> TypeGuard[BridgeDriverTaskView]:
    try:
        _verify_bridge_view(value)
    except (IntegrityError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def _mint_input(
    source: InputIntegrityDecisionEvidence,
    run_id: str,
    binding_fingerprint: str,
) -> InputIntegrityTaskView:
    _require_sha(binding_fingerprint, "input-integrity task binding")
    source_fingerprint = _input_source_fingerprint(source)
    launch_preflight_fingerprint = source.launch_preflight.fingerprint
    live_preflight_fingerprint = source.live_preflight.fingerprint
    calibration_input_fingerprint = source.binding.calibration_input_fingerprint
    payload = {
        "contract_version": BOUNDARY_VIEW_CONTRACT,
        "task_id": "input_integrity",
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        "launch_preflight_fingerprint": launch_preflight_fingerprint,
        "live_preflight_fingerprint": live_preflight_fingerprint,
        "calibration_input_fingerprint": calibration_input_fingerprint,
        "source_fingerprint": source_fingerprint,
    }
    view = object.__new__(InputIntegrityTaskView)
    for name, value in (
        ("contract_version", BOUNDARY_VIEW_CONTRACT),
        ("task_id", "input_integrity"),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("source", source),
        ("launch_preflight_fingerprint", launch_preflight_fingerprint),
        ("live_preflight_fingerprint", live_preflight_fingerprint),
        ("calibration_input_fingerprint", calibration_input_fingerprint),
        ("source_fingerprint", source_fingerprint),
        ("view_fingerprint", _fingerprint(payload)),
        ("_seal", _INPUT_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_INPUT[id(view)] = view
    return view


def _mint_bridge(
    source: StagedBridgeTaskEvidence | ReverifiedBridgeDriverEvidence,
    *,
    task_id: BridgeTaskId,
    run_id: str,
    binding_fingerprint: str,
    identities: Mapping[str, object],
    planned_looks: tuple[StagedBridgePlannedLookEvidence, ...],
    canonical: bool,
) -> BridgeDriverTaskView:
    look_fingerprints = tuple(item.evidence_fingerprint for item in planned_looks)
    payload = {
        "contract_version": BOUNDARY_VIEW_CONTRACT,
        "task_id": task_id,
        "run_id": run_id,
        "binding_fingerprint": binding_fingerprint,
        **identities,
        "planned_look_fingerprints": list(look_fingerprints),
        "canonical": canonical,
    }
    view = object.__new__(BridgeDriverTaskView)
    for name, value in (
        ("contract_version", BOUNDARY_VIEW_CONTRACT),
        ("task_id", task_id),
        ("run_id", run_id),
        ("binding_fingerprint", binding_fingerprint),
        ("source", source),
        *tuple(identities.items()),
        ("planned_looks", planned_looks),
        ("planned_look_fingerprints", look_fingerprints),
        ("view_fingerprint", _fingerprint(payload)),
        ("canonical", canonical),
        ("_seal", _BRIDGE_CAPABILITY),
    ):
        object.__setattr__(view, name, value)
    _MINTED_BRIDGE[id(view)] = view
    return view


def _verify_input_source(source: object, *, run_id: str) -> None:
    if not isinstance(source, InputIntegrityDecisionEvidence):
        raise ModelError("input-integrity source has the wrong type")
    _require_sha(run_id, "input-integrity run ID")
    verify_compatible_canonical_g3_preflights(
        source.launch_preflight,
        source.live_preflight,
    )
    if (
        not isinstance(source.binding, ScientificArtifactBinding)
        or source.binding.calibration_run_fingerprint != run_id
        or source.binding.preflight_fingerprint != source.launch_preflight.fingerprint
    ):
        raise ModelError("input-integrity run/preflight binding is inconsistent")


def _verify_input_view(value: object) -> None:
    if (
        not isinstance(value, InputIntegrityTaskView)
        or value._seal is not _INPUT_CAPABILITY
        or _MINTED_INPUT.get(id(value)) is not value
        or value.contract_version != BOUNDARY_VIEW_CONTRACT
        or value.task_id != "input_integrity"
    ):
        raise ModelError("input-integrity view lacks builder authority")
    _verify_input_source(value.source, run_id=value.run_id)
    _require_sha(value.binding_fingerprint, "input-integrity task binding")
    source_fingerprint = _input_source_fingerprint(value.source)
    expected = _fingerprint(
        {
            "contract_version": BOUNDARY_VIEW_CONTRACT,
            "task_id": value.task_id,
            "run_id": value.run_id,
            "binding_fingerprint": value.binding_fingerprint,
            "launch_preflight_fingerprint": value.source.launch_preflight.fingerprint,
            "live_preflight_fingerprint": value.source.live_preflight.fingerprint,
            "calibration_input_fingerprint": (
                value.source.binding.calibration_input_fingerprint
            ),
            "source_fingerprint": source_fingerprint,
        }
    )
    if (
        value.launch_preflight_fingerprint != value.source.launch_preflight.fingerprint
        or value.live_preflight_fingerprint != value.source.live_preflight.fingerprint
        or value.calibration_input_fingerprint
        != value.source.binding.calibration_input_fingerprint
        or value.source_fingerprint != source_fingerprint
        or value.view_fingerprint != expected
    ):
        raise ModelError("input-integrity view identity changed")


def _input_source_fingerprint(source: InputIntegrityDecisionEvidence) -> str:
    return _fingerprint(
        {
            "schema_id": "prpg-g3-input-integrity-source-v1",
            "launch_preflight_fingerprint": source.launch_preflight.fingerprint,
            "live_preflight_fingerprint": source.live_preflight.fingerprint,
            "artifact_binding": source.binding.as_dict(),
        }
    )


def _block_source_identities(
    source: object,
    *,
    task_id: str,
) -> dict[str, object]:
    if not isinstance(source, BlockCalibrationGateEvidence):
        raise ModelError("block-calibration source has the wrong type")
    expected_role = "design" if task_id == "design_block_calibration" else "production"
    if task_id not in {"design_block_calibration", "production_block_calibration"}:
        raise ModelError("block-calibration task ID is not registered")
    if source.role != expected_role:
        raise ModelError("block-calibration source has cross-role evidence")
    monthly = registered_block_calibration_identity(source.monthly)
    daily = registered_block_calibration_identity(source.daily)
    monthly_kernel = kernel_calibration_execution_identity(source.monthly_kernel)
    daily_kernel = kernel_calibration_execution_identity(source.daily_kernel)
    transition = transition_bootstrap_aggregate_identity(source.transition)
    fixed_point = (
        registered_final_block_projection_identity(source.monthly, source.daily)
        if expected_role == "design"
        else None
    )
    if (
        (monthly.artifact, monthly.frequency) != (expected_role, "monthly")
        or (daily.artifact, daily.frequency) != (expected_role, "daily")
        or (monthly_kernel.artifact, monthly_kernel.frequency)
        != (expected_role, "monthly")
        or (daily_kernel.artifact, daily_kernel.frequency) != (expected_role, "daily")
        or transition.role != expected_role
    ):
        raise ModelError("block-calibration source lineage is inconsistent")
    if monthly.parent_model_fingerprint != daily.parent_model_fingerprint:
        raise ModelError("block frequencies do not share the exact parent model")
    if (
        monthly.master_seed != daily.master_seed
        or monthly.scientific_version != daily.scientific_version
    ):
        raise ModelError("block frequencies do not share one RNG execution contract")
    if (
        monthly_kernel.n_states != source.selected_k
        or daily_kernel.n_states != source.selected_k
        or transition.n_states != source.selected_k
        or monthly_kernel.selected_length
        != source.monthly.execution.policy_selection.selected_length
        or daily_kernel.selected_length
        != source.daily.execution.policy_selection.selected_length
        or monthly_kernel.parent_model_fingerprint != monthly.parent_model_fingerprint
        or daily_kernel.parent_model_fingerprint != daily.parent_model_fingerprint
        or transition.parent_model_fingerprint != monthly.parent_model_fingerprint
    ):
        raise ModelError("block/kernel/transition source lineage is inconsistent")
    if (
        source.monthly_kernel.model_artifact_fingerprint
        != source.daily_kernel.model_artifact_fingerprint
        or monthly_kernel.master_seed != monthly.master_seed
        or daily_kernel.master_seed != monthly.master_seed
        or transition.master_seed != monthly.master_seed
        or monthly_kernel.scientific_version != monthly.scientific_version
        or daily_kernel.scientific_version != monthly.scientific_version
        or transition.scientific_version != monthly.scientific_version
    ):
        raise ModelError(
            "block/kernel/transition sources do not share one RNG execution contract"
        )
    monthly_rng = source.monthly.rng_execution_fingerprint
    daily_rng = source.daily.rng_execution_fingerprint
    _require_sha(monthly_rng, "monthly block RNG execution fingerprint")
    _require_sha(daily_rng, "daily block RNG execution fingerprint")
    assert isinstance(monthly_rng, str)
    assert isinstance(daily_rng, str)
    monthly_episode_counts = registered_block_source_episode_counts(source.monthly)
    daily_episode_counts = registered_block_source_episode_counts(source.daily)
    recurring_history_support_fingerprint: str | None = None
    if expected_role == "production" and source.selected_k == 4:
        recurring_support = owner_fixed_k4_recurring_history_support(
            monthly_episode_counts=monthly_episode_counts,
            daily_run_counts=daily_episode_counts,
        )
        if not recurring_support.passed:
            raise OwnerFixedK4RecurringHistoryFailure(recurring_support)
        recurring_history_support_fingerprint = _fingerprint(
            recurring_support.as_dict()
        )
    identities: dict[str, object] = {
        "monthly_calibration_fingerprint": monthly.calibration_fingerprint,
        "daily_calibration_fingerprint": daily.calibration_fingerprint,
        "monthly_kernel_fingerprint": monthly_kernel.execution_fingerprint,
        "daily_kernel_fingerprint": daily_kernel.execution_fingerprint,
        "transition_fingerprint": transition.execution_fingerprint,
        "monthly_kernel_parent_fingerprint": (monthly_kernel.parent_model_fingerprint),
        "daily_kernel_parent_fingerprint": daily_kernel.parent_model_fingerprint,
        "monthly_kernel_materialization_fingerprint": (
            monthly_kernel.source_materialization_fingerprint
        ),
        "daily_kernel_materialization_fingerprint": (
            daily_kernel.source_materialization_fingerprint
        ),
        "kernel_core_model_fingerprint": (
            source.monthly_kernel.model_artifact_fingerprint
        ),
        "kernel_calibration_rng_fingerprint": _composite_fingerprint(
            "kernel_calibration_rng",
            {
                "monthly": monthly_kernel.rng_execution_fingerprint,
                "daily": daily_kernel.rng_execution_fingerprint,
            },
        ),
        "transition_parent_model_fingerprint": (transition.parent_model_fingerprint),
        "transition_materialization_fingerprint": (
            transition.source_materialization_fingerprint
        ),
        "transition_rng_fingerprint": transition.rng_execution_fingerprint,
        "design_fixed_point_fingerprint": (
            None if fixed_point is None else fixed_point.fixed_point_fingerprint
        ),
        "design_fixed_point_iteration_history_fingerprint": (
            None if fixed_point is None else fixed_point.iteration_history_fingerprint
        ),
        "design_fixed_point_final_selection_fingerprint": (
            None if fixed_point is None else fixed_point.final_selection_fingerprint
        ),
        "design_fixed_point_final_block_parent_fingerprint": (
            None
            if fixed_point is None
            else _composite_fingerprint(
                "design_fixed_point_final_block_parents",
                {
                    "monthly": fixed_point.monthly_block_core_fingerprint,
                    "daily": fixed_point.daily_block_core_fingerprint,
                },
            )
        ),
        "parent_model_fingerprint": monthly.parent_model_fingerprint,
        "block_uniform_materialization_fingerprint": _composite_fingerprint(
            "block_uniform_materialization",
            {
                "monthly": monthly.materialization_fingerprint,
                "daily": daily.materialization_fingerprint,
            },
        ),
        "block_calibration_rng_fingerprint": _composite_fingerprint(
            "block_calibration_rng",
            {"monthly": monthly_rng, "daily": daily_rng},
        ),
        "monthly_source_episode_counts": monthly_episode_counts,
        "daily_source_episode_counts": daily_episode_counts,
        "recurring_history_support_fingerprint": (
            recurring_history_support_fingerprint
        ),
    }
    if fixed_point is not None and (
        fixed_point.selected_n_states != source.selected_k
        or fixed_point.master_seed != monthly.master_seed
        or fixed_point.scientific_version != monthly.scientific_version
    ):
        raise ModelError("task-8 blocks differ from final fixed-point parents")
    identities["source_fingerprint"] = _fingerprint(
        {"role": source.role, "selected_k": source.selected_k, **identities}
    )
    return identities


def _verify_block_view(value: object) -> None:
    if (
        not isinstance(value, BlockCalibrationTaskView)
        or value._seal is not _BLOCK_CAPABILITY
        or _MINTED_BLOCK.get(id(value)) is not value
        or value.contract_version != BOUNDARY_VIEW_CONTRACT
    ):
        raise ModelError("block-calibration view lacks builder authority")
    _require_sha(value.run_id, "block task run ID")
    _require_sha(value.binding_fingerprint, "block task binding")
    identities = _block_source_identities(value.source, task_id=value.task_id)
    if any(getattr(value, key) != item for key, item in identities.items()):
        raise ModelError("block-calibration view source identity changed")
    expected = _fingerprint(
        {
            "contract_version": BOUNDARY_VIEW_CONTRACT,
            "task_id": value.task_id,
            "role": value.role,
            "run_id": value.run_id,
            "binding_fingerprint": value.binding_fingerprint,
            **identities,
        }
    )
    if value.view_fingerprint != expected:
        raise ModelError("block-calibration view fingerprint changed")


def _artifact_source_identities(
    source: object,
    *,
    run_id: str,
    actual_bridge_parent_view_fingerprint: str,
) -> dict[str, str]:
    if not isinstance(source, ArtifactIntegrityDecisionEvidence):
        raise ModelError("artifact-integrity source has the wrong type")
    _require_sha(run_id, "artifact-integrity run ID")
    pair = source.pair
    if (
        not is_verified_scientific_artifact_pair(pair)
        or not is_verified_scientific_model_artifact(source.design)
        or not is_verified_scientific_model_artifact(source.production)
        or source.design.role != "design"
        or source.production.role != "production"
        or source.design.binding.calibration_run_fingerprint != run_id
        or source.production.binding.calibration_run_fingerprint != run_id
    ):
        raise ModelError("artifact-integrity source lineage is inconsistent")
    _require_sha(
        actual_bridge_parent_view_fingerprint,
        "actual-bridge parent view fingerprint",
    )
    pair_identity = scientific_artifact_pair_identity(pair)
    if (
        pair.run_id != run_id
        or pair.actual_bridge_parent_view_fingerprint
        != actual_bridge_parent_view_fingerprint
        or pair_identity["actual_bridge_parent_view_fingerprint"]
        != actual_bridge_parent_view_fingerprint
    ):
        raise ModelError("artifact-integrity pair/bridge lineage is inconsistent")
    design_identity = _loaded_artifact_identity(source.design)
    production_identity = _loaded_artifact_identity(source.production)
    design = _fingerprint(design_identity)
    production = _fingerprint(production_identity)
    binding = _fingerprint(source.design.binding.as_dict())
    provenance = _fingerprint(source.design.provenance.as_dict())
    reports = _composite_fingerprint(
        "final_artifact_report_set",
        {
            "design": design_identity["report_set_fingerprint"],
            "production": production_identity["report_set_fingerprint"],
        },
    )
    final_parent = _composite_fingerprint(
        "final_scientific_artifact_parent",
        {
            "design": design,
            "production": production,
        },
    )
    return {
        "design_source_fingerprint": design,
        "production_source_fingerprint": production,
        "design_reference_fingerprint": source.design.reference.fingerprint,
        "production_reference_fingerprint": source.production.reference.fingerprint,
        "design_core_model_fingerprint": source.design.core_model_fingerprint,
        "production_core_model_fingerprint": source.production.core_model_fingerprint,
        "design_parameter_set_fingerprint": source.design.parameter_set_fingerprint,
        "production_parameter_set_fingerprint": (
            source.production.parameter_set_fingerprint
        ),
        "pair_receipt_fingerprint": pair.reference.fingerprint,
        "design_role_source_fingerprint": pair.design_role_source_fingerprint,
        "production_role_source_fingerprint": (pair.production_role_source_fingerprint),
        "report_set_fingerprint": reports,
        "artifact_binding_fingerprint": binding,
        "provenance_fingerprint": provenance,
        "actual_bridge_parent_view_fingerprint": (
            actual_bridge_parent_view_fingerprint
        ),
        "final_artifact_parent_fingerprint": final_parent,
        "source_fingerprint": _fingerprint(
            {
                "design": design,
                "production": production,
                "actual_bridge_parent": actual_bridge_parent_view_fingerprint,
            }
        ),
    }


def _verify_artifact_view(value: object) -> None:
    if (
        not isinstance(value, ArtifactIntegrityTaskView)
        or value._seal is not _ARTIFACT_CAPABILITY
        or _MINTED_ARTIFACT.get(id(value)) is not value
        or value.contract_version != BOUNDARY_VIEW_CONTRACT
        or value.task_id != "artifact_integrity"
    ):
        raise ModelError("artifact-integrity view lacks builder authority")
    _require_sha(value.binding_fingerprint, "artifact-integrity task binding")
    if (
        not is_bridge_driver_task_view(value._actual_bridge_parent)
        or value._actual_bridge_parent.task_id != "actual_artifact_bridge"
        or value._actual_bridge_parent.run_id != value.run_id
        or value._actual_bridge_parent.binding_fingerprint == value.binding_fingerprint
    ):
        raise ModelError("artifact-integrity parent view changed")
    identities = _artifact_source_identities(
        value.source,
        run_id=value.run_id,
        actual_bridge_parent_view_fingerprint=(
            value._actual_bridge_parent.view_fingerprint
        ),
    )
    if any(getattr(value, key) != item for key, item in identities.items()):
        raise ModelError("artifact-integrity view source identity changed")
    expected = _fingerprint(
        {
            "contract_version": BOUNDARY_VIEW_CONTRACT,
            "task_id": value.task_id,
            "run_id": value.run_id,
            "binding_fingerprint": value.binding_fingerprint,
            **identities,
        }
    )
    if value.view_fingerprint != expected:
        raise ModelError("artifact-integrity view fingerprint changed")


def _staged_bridge_source_identities(
    source: object,
    *,
    task_id: BridgeTaskId,
) -> dict[str, object]:
    if not isinstance(source, StagedBridgeTaskEvidence):
        raise ModelError("canonical bridge view requires staged task evidence")
    identity = staged_bridge_task_evidence_identity(source)
    if identity.task_id != task_id:
        raise ModelError("staged bridge evidence is bound to the wrong task")
    expected_parents = _STAGED_BRIDGE_EXPECTED_PARENTS[task_id]
    if tuple(parent for parent, _ in identity.parent_task_fingerprints) != (
        expected_parents
    ):
        raise ModelError("staged bridge evidence has the wrong parent lineage")
    _require_sha(identity.common_binding_fingerprint, "staged bridge common binding")
    _require_sha(identity.base_preflight_fingerprint, "staged bridge base preflight")
    if identity.selected_n_states not in {2, 3, 4, 5}:
        raise ModelError("staged bridge selected K is outside the registry")
    _require_sha(
        identity.source_materialization_fingerprint,
        "staged bridge source materialization",
    )
    _require_sha(identity.result_fingerprint, "staged bridge result")
    _require_sha(identity.evidence_fingerprint, "staged bridge evidence")
    _require_sha(identity.stage_receipt_fingerprint, "staged bridge stage receipt")
    for _parent, fingerprint in identity.parent_task_fingerprints:
        _require_sha(fingerprint, "staged bridge parent evidence")
    if task_id == "actual_artifact_bridge":
        if identity.rng_execution_fingerprint is not None:
            raise ModelError("actual-artifact bridge must have an empty RNG identity")
    else:
        _require_sha(identity.rng_execution_fingerprint, "staged bridge RNG identity")
    looks = staged_bridge_task_planned_looks(source)
    look_identities = tuple(staged_bridge_planned_look_identity(item) for item in looks)
    if task_id.startswith("bridge_tier_b_"):
        if (
            not look_identities
            or any(item.task_id != task_id for item in look_identities)
            or tuple(item.look_index for item in look_identities)
            != tuple(range(1, len(look_identities) + 1))
        ):
            raise ModelError("staged Tier-B view lacks its exact planned-look prefix")
    elif look_identities:
        raise ModelError("non-Tier-B staged evidence acquired planned looks")
    source_lineage = _fingerprint(
        {
            "schema_id": "prpg-g3-staged-bridge-task-source-v1",
            "task_id": task_id,
            "dgp": identity.dgp,
            "passed": identity.passed,
            "base_preflight_fingerprint": identity.base_preflight_fingerprint,
            "selected_n_states": identity.selected_n_states,
            "common_binding_fingerprint": identity.common_binding_fingerprint,
            "parent_task_fingerprints": [
                {"task_id": parent, "evidence_fingerprint": fingerprint}
                for parent, fingerprint in identity.parent_task_fingerprints
            ],
            "source_materialization_fingerprint": (
                identity.source_materialization_fingerprint
            ),
            "rng_execution_fingerprint": identity.rng_execution_fingerprint,
            "result_fingerprint": identity.result_fingerprint,
            "evidence_fingerprint": identity.evidence_fingerprint,
            "stage_receipt_fingerprint": identity.stage_receipt_fingerprint,
            "planned_looks": [
                {
                    "look_index": item.look_index,
                    "sample_count": item.sample_count,
                    "exceedances": item.exceedances,
                    "decision": item.decision,
                    "source_materialization_fingerprint": (
                        item.source_materialization_fingerprint
                    ),
                    "rng_execution_fingerprint": item.rng_execution_fingerprint,
                    "result_fingerprint": item.result_fingerprint,
                    "evidence_fingerprint": item.evidence_fingerprint,
                    "stage_receipt_fingerprint": item.stage_receipt_fingerprint,
                }
                for item in look_identities
            ],
        }
    )
    return {
        "base_preflight_fingerprint": identity.base_preflight_fingerprint,
        "selected_k": identity.selected_n_states,
        "shared_binding_fingerprint": identity.common_binding_fingerprint,
        "parent_result_fingerprints": identity.parent_task_fingerprints,
        "task_result_fingerprint": identity.evidence_fingerprint,
        "source_lineage_fingerprint": source_lineage,
        "materialization_fingerprint": identity.source_materialization_fingerprint,
        "known_trace_fingerprint": None,
        "rng_fingerprint": identity.rng_execution_fingerprint,
    }


def _legacy_bridge_source_identities(
    source: object,
    *,
    task_id: BridgeTaskId,
) -> dict[str, object]:
    if not isinstance(source, ReverifiedBridgeDriverEvidence):
        raise ModelError("bridge view requires reverified driver evidence")
    live = reverify_bridge_driver_result(source.driver_result, source.request)
    identities = _bridge_task_identities(source, task_id=task_id)
    if _bridge_task_identities(live, task_id=task_id) != identities:
        raise ModelError("bridge driver evidence changed after reverification")
    return identities


def _bridge_task_identities(
    source: ReverifiedBridgeDriverEvidence,
    *,
    task_id: BridgeTaskId,
) -> dict[str, object]:
    shared = source.binding.fingerprint
    _require_sha(shared, "bridge shared binding fingerprint")
    if (
        source.request.base_preflight.fingerprint
        != source.binding.base_preflight_fingerprint
    ):
        raise ModelError("bridge task request differs from its preregistered binding")
    purpose1 = dict(source.binding.purpose1_trace_fingerprints)
    if task_id == "bridge_resource_preflight":
        task_result = source.benchmark.receipt_fingerprint
        materialization = source.benchmark.receipt_fingerprint
        known_trace = None
        parents: tuple[tuple[str, str], ...] = ()
        task_lineage: dict[str, object] = {
            "benchmark_receipt_fingerprint": source.benchmark.receipt_fingerprint,
        }
    elif task_id == "actual_artifact_bridge":
        task_result = source.driver_result.scientific_fingerprint
        materialization = source.binding.actual_bridge_evidence_fingerprint
        known_trace = None
        parents = (
            (
                "bridge_tier_a_model_dgp",
                _bridge_tier_a_identity(source, "model"),
            ),
            (
                "bridge_tier_a_nonparametric_dgp",
                _bridge_tier_a_identity(source, "nonparametric"),
            ),
            (
                "bridge_tier_b_model_dgp",
                _bridge_tier_b_identity(source, "model"),
            ),
            (
                "bridge_tier_b_nonparametric_dgp",
                _bridge_tier_b_identity(source, "nonparametric"),
            ),
        )
        task_lineage = {
            "tier_a_parents": {
                dgp: _bridge_tier_a_identity(source, dgp) for dgp in _BRIDGE_DGP_IDS
            },
            "tier_b_parents": {
                dgp: _bridge_tier_b_identity(source, dgp) for dgp in _BRIDGE_DGP_IDS
            },
            "final_receipt_fingerprint": source.final_receipt_fingerprint,
        }
    else:
        dgp: Literal["model", "nonparametric"] = (
            "model" if task_id.endswith("model_dgp") else "nonparametric"
        )
        item = next((value for value in source.dgp_results if value.dgp == dgp), None)
        if item is None:
            raise ModelError("bridge task source lacks its registered DGP")
        if "tier_a" in task_id:
            task_result = _bridge_tier_a_identity(source, dgp)
            materialization = (
                task_result
                if item.tier_a is None
                else item.tier_a.scientific_fingerprint
            )
            task_lineage = {
                "benchmark_parent": source.benchmark.receipt_fingerprint,
                "tier_a_result": task_result,
            }
            parents = (
                ("bridge_resource_preflight", source.benchmark.receipt_fingerprint),
            )
        else:
            task_result = _bridge_tier_b_identity(source, dgp)
            materialization = task_result
            task_lineage = {
                "benchmark_parent": source.benchmark.receipt_fingerprint,
                "tier_a_parents": {
                    parent_dgp: _bridge_tier_a_identity(source, parent_dgp)
                    for parent_dgp in _BRIDGE_DGP_IDS
                },
                "tier_b_result": task_result,
            }
            parents = (
                ("bridge_resource_preflight", source.benchmark.receipt_fingerprint),
                (
                    "bridge_tier_a_model_dgp",
                    _bridge_tier_a_identity(source, "model"),
                ),
                (
                    "bridge_tier_a_nonparametric_dgp",
                    _bridge_tier_a_identity(source, "nonparametric"),
                ),
            )
        try:
            known_trace = purpose1[dgp]
        except KeyError as error:
            raise ModelError("bridge task source lacks its purpose-1 trace") from error
    _require_sha(task_result, "bridge task result fingerprint")
    _require_sha(materialization, "bridge task materialization fingerprint")
    if known_trace is not None:
        _require_sha(known_trace, "bridge task known trace fingerprint")
    source_fingerprint = _fingerprint(
        {
            "schema_id": "prpg-g3-task-specific-bridge-lineage-v1",
            "task_id": task_id,
            "shared_binding_fingerprint": shared,
            "base_preflight_fingerprint": source.binding.base_preflight_fingerprint,
            "task_result_fingerprint": task_result,
            "parent_result_fingerprints": dict(parents),
            "materialization_fingerprint": materialization,
            "known_trace_fingerprint": known_trace,
            "rng_fingerprint": None,
            "task_lineage": task_lineage,
        }
    )
    return {
        "base_preflight_fingerprint": source.binding.base_preflight_fingerprint,
        "selected_k": source.binding.selected_n_states,
        "shared_binding_fingerprint": shared,
        "parent_result_fingerprints": parents,
        "task_result_fingerprint": task_result,
        "source_lineage_fingerprint": source_fingerprint,
        "materialization_fingerprint": materialization,
        "known_trace_fingerprint": known_trace,
        "rng_fingerprint": None,
    }


def _bridge_tier_a_identity(
    source: ReverifiedBridgeDriverEvidence,
    dgp: Literal["model", "nonparametric"],
) -> str:
    item = next((value for value in source.dgp_results if value.dgp == dgp), None)
    if item is None:
        raise ModelError("bridge evidence lacks its registered DGP")
    gate = item.tier_a
    payload: dict[str, object] = {
        "schema_id": "prpg-g3-bridge-tier-a-result-v1",
        "dgp": dgp,
        "available": gate is not None,
    }
    if gate is not None:
        payload.update(
            {
                "pairs": gate.pairs,
                "successes": gate.successes,
                "lower_bound": gate.lower_bound,
                "passed": gate.passed,
                "scientific_fingerprint": gate.scientific_fingerprint,
                "receipt_fingerprint": gate.receipt_fingerprint,
            }
        )
    return _fingerprint(payload)


def _bridge_tier_b_identity(
    source: ReverifiedBridgeDriverEvidence,
    dgp: Literal["model", "nonparametric"],
) -> str:
    item = next((value for value in source.dgp_results if value.dgp == dgp), None)
    if item is None:
        raise ModelError("bridge evidence lacks its registered DGP")
    looks = source.tier_b_looks_by_dgp[dgp]
    return _fingerprint(
        {
            "schema_id": "prpg-g3-bridge-tier-b-result-v1",
            "dgp": dgp,
            "accelerator_eligible": item.accelerator_eligible,
            "decision": item.tier_b_decision,
            "pairs": item.tier_b_pairs,
            "passed": item.tier_b_passed,
            "looks": [
                {
                    "look_index": look.look_index,
                    "sample_count": look.sample_count,
                    "exceedances": look.exceedances,
                    "interval": _normalize(look.interval),
                    "decision": look.decision,
                    "receipt_fingerprint": look.receipt_fingerprint,
                }
                for look in looks
            ],
        }
    )


def _verify_bridge_view(value: object) -> None:
    if (
        not isinstance(value, BridgeDriverTaskView)
        or value._seal is not _BRIDGE_CAPABILITY
        or _MINTED_BRIDGE.get(id(value)) is not value
        or value.contract_version != BOUNDARY_VIEW_CONTRACT
        or value.task_id not in BRIDGE_TASK_IDS
        or not value.canonical
    ):
        raise ModelError("bridge task view lacks canonical builder authority")
    if not isinstance(value.source, StagedBridgeTaskEvidence):
        raise ModelError("canonical bridge task view lost its staged evidence")
    _require_sha(value.run_id, "bridge task run ID")
    _require_sha(value.binding_fingerprint, "bridge task binding")
    identities = _staged_bridge_source_identities(
        value.source,
        task_id=value.task_id,
    )
    planned_looks = staged_bridge_task_planned_looks(value.source)
    look_fingerprints = tuple(item.evidence_fingerprint for item in planned_looks)
    expected = _fingerprint(
        {
            "contract_version": BOUNDARY_VIEW_CONTRACT,
            "task_id": value.task_id,
            "run_id": value.run_id,
            "binding_fingerprint": value.binding_fingerprint,
            **identities,
            "planned_look_fingerprints": list(look_fingerprints),
            "canonical": True,
        }
    )
    if (
        any(getattr(value, key) != item for key, item in identities.items())
        or value.planned_looks != planned_looks
        or value.planned_look_fingerprints != look_fingerprints
        or value.view_fingerprint != expected
    ):
        raise ModelError("bridge task view identity changed")


def _loaded_artifact_identity(
    artifact: LoadedScientificModelArtifact,
) -> dict[str, str]:
    arrays = {
        name: scientific_array_fingerprint(value, role="g3_boundary_array")
        for name, value in sorted(artifact.arrays.items())
    }
    reports = {
        name: _fingerprint(_normalize(value))
        for name, value in sorted(artifact.reports.items())
    }
    report_set = _fingerprint(
        {
            "schema_id": "prpg-g3-artifact-report-set-v1",
            "reports": reports,
        }
    )
    identity = {
        "reference_fingerprint": artifact.reference.fingerprint,
        "role": artifact.role,
        "selected_k": str(artifact.selected_k),
        "binding_fingerprint": _fingerprint(artifact.binding.as_dict()),
        "provenance_fingerprint": _fingerprint(artifact.provenance.as_dict()),
        "generation_policy_fingerprint": _identity_fingerprint(
            artifact.generation_policy
        ),
        "geometry_fingerprint": _fingerprint(artifact.geometry.as_dict()),
        "array_set_fingerprint": _fingerprint(
            {"schema_id": "prpg-g3-artifact-array-set-v1", "arrays": arrays}
        ),
        "report_set_fingerprint": report_set,
        "core_model_fingerprint": artifact.core_model_fingerprint,
        "parameter_set_fingerprint": artifact.parameter_set_fingerprint,
    }
    for label, fingerprint in identity.items():
        if label not in {"role", "selected_k"}:
            _require_sha(fingerprint, f"artifact {label}")
    return identity


def _composite_fingerprint(label: str, values: Mapping[str, str]) -> str:
    for name, value in values.items():
        _require_sha(value, f"{label} {name} fingerprint")
    return _fingerprint(
        {
            "schema_id": "prpg-g3-boundary-composite-v1",
            "label": label,
            "fingerprints": dict(sorted(values.items())),
        }
    )


def _identity_fingerprint(value: object) -> str:
    return _fingerprint(_normalize(value))


def _normalize(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "array_fingerprint": scientific_array_fingerprint(
                value,
                role="g3_boundary_array",
            ),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Enum):
        return _normalize(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _normalize(getattr(value, item.name))
                for item in dataclasses.fields(value)
                if not item.name.startswith("_")
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ModelError("boundary identity mappings require string keys")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("boundary identity contains a non-finite number")
        return value
    raise ModelError(
        "boundary identity contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _require_sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"{label} must be a lowercase SHA-256 digest")
