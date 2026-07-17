"""Prebinding publication boundary for canonical G3 planned looks.

Planned-look receipts are scientific parents of their owning task binding.
They must therefore be projected from producer evidence and committed before
that binding exists.  This module supplies the cycle-free boundary for the
four kernel families and two staged Tier-B bridge families.

Publication is prefix-safe.  An interrupted call may leave a verified durable
prefix, but an exact resume first byte/evidence-compares every existing receipt
against a freshly rederived raw projection before appending anything.  The
returned registered prefix is available only after every owned family has a
terminal decision.  Unexpected projection and write errors are never changed
into scientific failures.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypeGuard, cast

from prpg.errors import IntegrityError, ModelError
from prpg.model import g3_boundary_views as boundary_view_module
from prpg.model.bridge_driver import StagedBridgeTaskEvidence
from prpg.model.g3_boundary_views import (
    BlockCalibrationGateEvidence,
    BlockCalibrationTaskView,
    BridgeDriverTaskView,
    BridgeTaskId,
)
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    PlannedLookSpec,
)
from prpg.model.g3_science_handlers import (
    CanonicalG3PlannedLook,
    canonical_planned_looks,
)
from prpg.model.g3_task_binding import (
    ScientificTaskBinding,
    verify_scientific_task_binding,
)
from prpg.model.g3_task_publication import bridge_executor_planned_looks
from prpg.model.kernel_execution import (
    CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
    CANONICAL_PREFIX_WINDOW_COUNTS,
    KernelCalibrationExecution,
    kernel_calibration_execution_identity,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    LaunchLease,
    LookDecision,
    PlannedLookReceipt,
)

PlannedLookOwnerTaskId: TypeAlias = Literal[
    "design_block_calibration",
    "production_block_calibration",
    "bridge_tier_b_model_dgp",
    "bridge_tier_b_nonparametric_dgp",
]
RawPlannedLookSource: TypeAlias = (
    BlockCalibrationGateEvidence | StagedBridgeTaskEvidence
)
FinalPlannedLookView: TypeAlias = BlockCalibrationTaskView | BridgeDriverTaskView

PLANNED_LOOK_EXECUTION_CONTRACT = "g3-raw-planned-look-publication-v1"
_PLANNED_LOOK_PREFIX_CAPABILITY = object()
_OWNER_TASKS: tuple[PlannedLookOwnerTaskId, ...] = (
    "design_block_calibration",
    "production_block_calibration",
    "bridge_tier_b_model_dgp",
    "bridge_tier_b_nonparametric_dgp",
)


@dataclass(frozen=True, slots=True, init=False)
class RegisteredPlannedLookPrefix:
    """Terminal durable look families descended from one exact raw source."""

    contract_version: str
    run_id: str
    task_id: PlannedLookOwnerTaskId
    family_ids: tuple[str, ...]
    look_count: int
    raw_projection_fingerprint: str
    receipt_fingerprints: tuple[tuple[str, tuple[str, ...]], ...]
    terminal_decisions: tuple[tuple[str, LookDecision], ...]
    publication_fingerprint: str
    generation_authorized: Literal[False]
    _raw_source: RawPlannedLookSource = field(repr=False, compare=False)
    _raw_projection: tuple[CanonicalG3PlannedLook, ...] = field(
        repr=False,
        compare=False,
    )
    _store: CalibrationRunStore = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredPlannedLookPrefix is created only by "
            "publish_raw_planned_look_prefix"
        )


@dataclass(frozen=True, slots=True)
class RegisteredPlannedLookPrefixIdentity:
    """Reverified raw projection and durable receipt-chain identity."""

    run_id: str
    task_id: PlannedLookOwnerTaskId
    family_ids: tuple[str, ...]
    look_count: int
    raw_projection_fingerprint: str
    receipt_fingerprints: tuple[tuple[str, tuple[str, ...]], ...]
    terminal_decisions: tuple[tuple[str, LookDecision], ...]
    publication_fingerprint: str


@dataclass(frozen=True, slots=True)
class PlannedLookParityIdentity:
    """One exact raw/store/binding/final-view four-way comparison."""

    run_id: str
    task_id: PlannedLookOwnerTaskId
    binding_fingerprint: str
    raw_projection_fingerprint: str
    receipt_fingerprints: tuple[tuple[str, tuple[str, ...]], ...]
    final_view_projection_fingerprint: str
    parity_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PrefixMintRecord:
    prefix: RegisteredPlannedLookPrefix
    source: RawPlannedLookSource
    projection: tuple[CanonicalG3PlannedLook, ...]
    store: CalibrationRunStore


_MINTED_PREFIXES: dict[int, _PrefixMintRecord] = {}


def project_raw_planned_looks(
    task_id: PlannedLookOwnerTaskId,
    source: RawPlannedLookSource,
) -> tuple[CanonicalG3PlannedLook, ...]:
    """Project a terminal look sequence directly from producer evidence."""

    checked_task = _owner_task(task_id)
    specs = _owned_specs(checked_task)
    if checked_task in {
        "design_block_calibration",
        "production_block_calibration",
    }:
        if not isinstance(source, BlockCalibrationGateEvidence):
            raise ModelError("kernel planned looks require raw block-gate evidence")
        # Reverify the entire block/kernel/transition parent graph, not merely
        # the public kernel fields, before projecting either frequency.
        boundary_view_module._block_source_identities(source, task_id=checked_task)
        kernels = (source.monthly_kernel, source.daily_kernel)
        if len(specs) != 2:
            raise AssertionError("block task lacks two planned-look families")
        projected: list[CanonicalG3PlannedLook] = []
        for spec, kernel in zip(specs, kernels, strict=True):
            projected.extend(_project_kernel_family(spec, kernel))
        result = tuple(projected)
    else:
        if not isinstance(source, StagedBridgeTaskEvidence):
            raise ModelError("Tier-B planned looks require staged task evidence")
        result = bridge_executor_planned_looks(
            source,
            task_id=cast(BridgeTaskId, checked_task),
        )
    _verify_terminal_projection(checked_task, result)
    return result


def publish_raw_planned_look_prefix(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    task_id: PlannedLookOwnerTaskId,
    source: RawPlannedLookSource,
) -> RegisteredPlannedLookPrefix:
    """Commit or exactly resume every raw planned-look family before binding."""

    checked_task = _owner_task(task_id)
    _verify_active_context(store, lease)
    _require_owner_dependencies(store, lease.run_id, checked_task)
    projection = project_raw_planned_looks(checked_task, source)
    specs = _owned_specs(checked_task)

    all_before = store.verify_planned_look_receipts(
        lease.run_id,
        require_terminal=False,
    )
    existing = _receipts_for_specs(specs, all_before)
    # Compare every existing family before appending any family.  A mixed
    # prefix can therefore never be partly extended before the mismatch is
    # discovered.
    _compare_receipts_to_projection(existing, projection, allow_prefix=True)
    prior_indices = {
        family_id: len(all_before.get(family_id, ())) for family_id in _spec_ids(specs)
    }
    for look in projection:
        if look.look_index <= prior_indices[look.family_id]:
            continue
        store.write_planned_look_receipt(
            lease,
            family_id=look.family_id,
            look_index=look.look_index,
            sample_count=look.sample_count,
            decision=look.decision,
            observed=look.observed,
        )
        prior_indices[look.family_id] = look.look_index

    all_after = store.verify_planned_look_receipts(
        lease.run_id,
        require_terminal=False,
    )
    receipts = _receipts_for_specs(specs, all_after)
    _compare_receipts_to_projection(receipts, projection, allow_prefix=False)
    receipt_fingerprints = _receipt_fingerprint_mapping(specs, all_after)
    terminal = _terminal_decisions(specs, all_after)
    projection_fingerprint = _projection_fingerprint(projection)
    publication_fingerprint = _publication_fingerprint(
        run_id=lease.run_id,
        task_id=checked_task,
        projection_fingerprint=projection_fingerprint,
        receipt_fingerprints=receipt_fingerprints,
        terminal_decisions=terminal,
    )
    value = object.__new__(RegisteredPlannedLookPrefix)
    for name, item in (
        ("contract_version", PLANNED_LOOK_EXECUTION_CONTRACT),
        ("run_id", lease.run_id),
        ("task_id", checked_task),
        ("family_ids", _spec_ids(specs)),
        ("look_count", len(projection)),
        ("raw_projection_fingerprint", projection_fingerprint),
        ("receipt_fingerprints", receipt_fingerprints),
        ("terminal_decisions", terminal),
        ("publication_fingerprint", publication_fingerprint),
        ("generation_authorized", False),
        ("_raw_source", source),
        ("_raw_projection", projection),
        ("_store", store),
        ("_seal", _PLANNED_LOOK_PREFIX_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_PREFIXES[id(value)] = _PrefixMintRecord(
        value,
        source,
        projection,
        store,
    )
    registered_planned_look_prefix_identity(value)
    return value


def registered_planned_look_prefix_identity(
    value: RegisteredPlannedLookPrefix,
) -> RegisteredPlannedLookPrefixIdentity:
    """Reproject raw evidence and reload every durable receipt."""

    if not isinstance(value, RegisteredPlannedLookPrefix):
        raise ModelError("planned-look prefix has the wrong type")
    record = _MINTED_PREFIXES.get(id(value))
    if (
        value._seal is not _PLANNED_LOOK_PREFIX_CAPABILITY
        or record is None
        or record.prefix is not value
        or record.source is not value._raw_source
        or record.projection is not value._raw_projection
        or record.store is not value._store
    ):
        raise ModelError("planned-look prefix lacks publisher authority")
    task_id = _owner_task(value.task_id)
    projection = project_raw_planned_looks(task_id, value._raw_source)
    if not _same_projection(projection, value._raw_projection):
        raise IntegrityError("raw planned-look projection changed after publication")
    specs = _owned_specs(task_id)
    all_receipts = value._store.verify_planned_look_receipts(
        value.run_id,
        require_terminal=False,
    )
    receipts = _receipts_for_specs(specs, all_receipts)
    _compare_receipts_to_projection(receipts, projection, allow_prefix=False)
    receipt_fingerprints = _receipt_fingerprint_mapping(specs, all_receipts)
    terminal = _terminal_decisions(specs, all_receipts)
    projection_fingerprint = _projection_fingerprint(projection)
    publication_fingerprint = _publication_fingerprint(
        run_id=value.run_id,
        task_id=task_id,
        projection_fingerprint=projection_fingerprint,
        receipt_fingerprints=receipt_fingerprints,
        terminal_decisions=terminal,
    )
    if (
        value.contract_version != PLANNED_LOOK_EXECUTION_CONTRACT
        or value.family_ids != _spec_ids(specs)
        or value.look_count != len(projection)
        or value.raw_projection_fingerprint != projection_fingerprint
        or value.receipt_fingerprints != receipt_fingerprints
        or value.terminal_decisions != terminal
        or value.publication_fingerprint != publication_fingerprint
        or value.generation_authorized is not False
    ):
        raise ModelError("registered planned-look prefix identity changed")
    return RegisteredPlannedLookPrefixIdentity(
        value.run_id,
        task_id,
        value.family_ids,
        value.look_count,
        projection_fingerprint,
        receipt_fingerprints,
        terminal,
        publication_fingerprint,
    )


def is_registered_planned_look_prefix(
    value: object,
) -> TypeGuard[RegisteredPlannedLookPrefix]:
    """Return whether this is the unchanged, fully durable prefix object."""

    if not isinstance(value, RegisteredPlannedLookPrefix):
        return False
    try:
        registered_planned_look_prefix_identity(value)
    except (AttributeError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def registered_planned_look_receipts(
    value: RegisteredPlannedLookPrefix,
) -> tuple[PlannedLookReceipt, ...]:
    """Reload exact terminal receipts after full prefix reverification."""

    identity = registered_planned_look_prefix_identity(value)
    specs = _owned_specs(identity.task_id)
    all_receipts = value._store.verify_planned_look_receipts(
        identity.run_id,
        require_terminal=False,
    )
    return _receipts_for_specs(specs, all_receipts)


def verify_planned_look_binding_view_parity(
    prefix: RegisteredPlannedLookPrefix,
    *,
    binding: ScientificTaskBinding,
    final_view: FinalPlannedLookView,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
) -> PlannedLookParityIdentity:
    """Verify raw projection == store == binding == guarded final view."""

    identity = registered_planned_look_prefix_identity(prefix)
    verify_scientific_task_binding(
        binding,
        run_store=prefix._store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
    )
    if binding.run_id != identity.run_id or binding.task_id != identity.task_id:
        raise IntegrityError("planned-look binding belongs to another run or task")
    if binding.owned_planned_look_fingerprints != identity.receipt_fingerprints:
        raise IntegrityError("binding-owned planned looks differ from raw receipts")
    if (
        not isinstance(final_view, BlockCalibrationTaskView | BridgeDriverTaskView)
        or final_view.run_id != identity.run_id
        or final_view.task_id != identity.task_id
        or final_view.binding_fingerprint != binding.fingerprint
        or final_view.source is not prefix._raw_source
    ):
        raise ModelError("final planned-look view has different exact ancestry")
    final_projection = canonical_planned_looks(identity.task_id, final_view)
    if not _same_projection(prefix._raw_projection, final_projection):
        raise IntegrityError("final view planned looks differ from raw projection")
    final_fingerprint = _projection_fingerprint(final_projection)
    parity_fingerprint = _json_fingerprint(
        {
            "schema_id": "prpg-g3-planned-look-four-way-parity-v1",
            "run_id": identity.run_id,
            "task_id": identity.task_id,
            "binding_fingerprint": binding.fingerprint,
            "raw_projection_fingerprint": identity.raw_projection_fingerprint,
            "receipt_fingerprints": _fingerprint_mapping_document(
                identity.receipt_fingerprints
            ),
            "final_view_projection_fingerprint": final_fingerprint,
        }
    )
    return PlannedLookParityIdentity(
        identity.run_id,
        identity.task_id,
        binding.fingerprint,
        identity.raw_projection_fingerprint,
        identity.receipt_fingerprints,
        final_fingerprint,
        parity_fingerprint,
    )


def _project_kernel_family(
    spec: PlannedLookSpec,
    kernel: KernelCalibrationExecution,
) -> tuple[CanonicalG3PlannedLook, ...]:
    identity = kernel_calibration_execution_identity(kernel)
    expected_role = "design" if "_design_" in spec.family_id else "production"
    expected_frequency = "monthly" if spec.family_id.endswith("monthly") else "daily"
    if (
        identity.artifact != expected_role
        or identity.frequency != expected_frequency
        or not kernel.strict_canonical
        or kernel.planned_looks != CANONICAL_PREFIX_WINDOW_COUNTS
        or spec.sample_counts != CANONICAL_PREFIX_WINDOW_COUNTS
    ):
        raise ModelError("kernel planned-look producer identity is noncanonical")
    completed = kernel.completed_looks
    if not completed or len(completed) > len(spec.sample_counts):
        raise ModelError("kernel planned-look sequence is empty or too long")
    projected: list[CanonicalG3PlannedLook] = []
    for index, look in enumerate(completed, start=1):
        passed = (
            look.maximum_annualized_half_width <= CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
        )
        if (
            look.look_index != index - 1
            or look.windows != spec.sample_counts[index - 1]
            or look.precision.windows != look.windows
            or look.passed is not passed
        ):
            raise ModelError("kernel planned-look evidence is inconsistent")
        is_last = index == len(completed)
        if look.passed and not is_last:
            raise ModelError("kernel execution retained work after a passing look")
        if not look.passed and is_last and index != len(spec.sample_counts):
            raise ModelError("kernel execution stopped before pass or maximum look")
        decision: LookDecision = (
            "pass" if is_last and look.passed else "fail" if is_last else "continue"
        )
        precision = look.precision
        projected.append(
            CanonicalG3PlannedLook(
                spec.family_id,
                index,
                look.windows,
                decision,
                MappingProxyType(
                    {
                        "calibration_stream_fingerprint": (
                            precision.calibration_stream_fingerprint
                        ),
                        "kernel_sample_fingerprint": (
                            precision.kernel_sample_fingerprint
                        ),
                        "maximum_annualized_half_width": (
                            look.maximum_annualized_half_width
                        ),
                        "model_artifact_fingerprint": (
                            precision.model_artifact_fingerprint
                        ),
                        "precision_passed": look.passed,
                        "source_window_fingerprint": (
                            precision.source_window_fingerprint
                        ),
                        "window_statistics_fingerprint": (
                            precision.window_statistics_fingerprint
                        ),
                    }
                ),
            )
        )
    if kernel.precision_passed is not completed[-1].passed:
        raise ModelError("kernel precision result differs from its terminal look")
    return tuple(projected)


def _verify_terminal_projection(
    task_id: PlannedLookOwnerTaskId,
    projection: Sequence[CanonicalG3PlannedLook],
) -> None:
    specs = _owned_specs(task_id)
    expected_family_ids = _spec_ids(specs)
    by_family: dict[str, list[CanonicalG3PlannedLook]] = {
        family_id: [] for family_id in expected_family_ids
    }
    encountered: list[str] = []
    for look in projection:
        if not isinstance(look, CanonicalG3PlannedLook):
            raise ModelError("raw planned-look projection has the wrong type")
        if look.family_id not in by_family:
            raise ModelError("raw projection contains an unowned planned-look family")
        if not encountered or encountered[-1] != look.family_id:
            if look.family_id in encountered:
                raise ModelError("raw planned-look families are interleaved")
            encountered.append(look.family_id)
        by_family[look.family_id].append(look)
    if tuple(encountered) != expected_family_ids:
        raise ModelError("raw projection omits or reorders a planned-look family")
    for spec in specs:
        family = by_family[spec.family_id]
        if not family or len(family) > len(spec.sample_counts):
            raise ModelError("planned-look family has no valid terminal prefix")
        for index, look in enumerate(family, start=1):
            if (
                look.look_index != index
                or look.sample_count != spec.sample_counts[index - 1]
                or look.decision not in {"continue", "pass", "fail"}
            ):
                raise ModelError("planned-look family schedule is inconsistent")
            if index < len(family) and look.decision != "continue":
                raise ModelError("terminal planned look is followed by more evidence")
        if family[-1].decision == "continue":
            raise ModelError("planned-look family lacks a terminal raw decision")


def _verify_active_context(store: CalibrationRunStore, lease: LaunchLease) -> None:
    if (
        not isinstance(store, CalibrationRunStore)
        or not isinstance(lease, LaunchLease)
        or lease.store is not store
        or not lease.active
    ):
        raise IntegrityError(
            "planned-look publication requires its active launch lease"
        )
    marker = store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != lease.preflight_fingerprint
        or marker.workers != lease.workers
    ):
        raise IntegrityError("planned-look lease differs from the running launch")


def _require_owner_dependencies(
    store: CalibrationRunStore,
    run_id: str,
    task_id: PlannedLookOwnerTaskId,
) -> None:
    owner_position = G3_GATE_ORDER.index(task_id)
    all_receipts = store.verify_planned_look_receipts(
        run_id,
        require_terminal=False,
    )
    future_families = {
        spec.family_id
        for spec in G3_PLANNED_LOOK_REGISTRY
        if G3_GATE_ORDER.index(spec.owner_task_id) > owner_position
    }
    unexpected_future = sorted(future_families.intersection(all_receipts))
    if unexpected_future:
        raise IntegrityError(
            "future-task planned looks precede the exact owner",
            details={"task_id": task_id, "future_families": unexpected_future},
        )

    results = store.verify_task_results(run_id, require_complete=False)
    expected_prior = G3_GATE_ORDER[:owner_position]
    if tuple(results) != expected_prior:
        raise ModelError(
            "planned-look owner is not the exact next registry task",
            details={
                "task_id": task_id,
                "expected_prior": list(expected_prior),
                "actual_results": list(results),
            },
        )
    nonpassing = [
        prior_task
        for prior_task in expected_prior
        if results[prior_task].status != "passed"
    ]
    if nonpassing:
        raise ModelError(
            "planned-look owner has a nonpassing prior registry task",
            details={"task_id": task_id, "nonpassing": nonpassing},
        )
    invalid_prior_looks: dict[str, str] = {}
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if G3_GATE_ORDER.index(spec.owner_task_id) >= owner_position:
            continue
        family = all_receipts.get(spec.family_id, ())
        if not family:
            invalid_prior_looks[spec.family_id] = "missing"
        elif family[-1].decision != "pass":
            invalid_prior_looks[spec.family_id] = family[-1].decision
    if invalid_prior_looks:
        raise IntegrityError(
            "prior-task planned-look receipts are missing or nonpassing",
            details={
                "task_id": task_id,
                "prior_families": invalid_prior_looks,
            },
        )


def _compare_receipts_to_projection(
    receipts: Sequence[PlannedLookReceipt],
    projection: Sequence[CanonicalG3PlannedLook],
    *,
    allow_prefix: bool,
) -> None:
    if (allow_prefix and len(receipts) > len(projection)) or (
        not allow_prefix and len(receipts) != len(projection)
    ):
        raise IntegrityError("durable planned-look sequence length changed")
    for receipt, look in zip(receipts, projection, strict=False):
        if (
            receipt.family_id != look.family_id
            or receipt.look_index != look.look_index
            or receipt.sample_count != look.sample_count
            or receipt.decision != look.decision
            or dict(receipt.observed) != _json_mapping(look.observed)
        ):
            raise IntegrityError(
                "durable planned-look bytes/evidence differ from raw projection"
            )


def _receipts_for_specs(
    specs: Sequence[PlannedLookSpec],
    all_receipts: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> tuple[PlannedLookReceipt, ...]:
    return tuple(
        receipt for spec in specs for receipt in all_receipts.get(spec.family_id, ())
    )


def _receipt_fingerprint_mapping(
    specs: Sequence[PlannedLookSpec],
    all_receipts: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (
            spec.family_id,
            tuple(item.fingerprint for item in all_receipts.get(spec.family_id, ())),
        )
        for spec in specs
    )


def _terminal_decisions(
    specs: Sequence[PlannedLookSpec],
    all_receipts: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> tuple[tuple[str, LookDecision], ...]:
    result: list[tuple[str, LookDecision]] = []
    for spec in specs:
        receipts = all_receipts.get(spec.family_id, ())
        if not receipts or receipts[-1].decision == "continue":
            raise IntegrityError("durable planned-look family lacks terminal evidence")
        result.append((spec.family_id, receipts[-1].decision))
    return tuple(result)


def _projection_fingerprint(projection: Sequence[CanonicalG3PlannedLook]) -> str:
    return _json_fingerprint(
        {
            "schema_id": "prpg-g3-raw-planned-look-projection-v1",
            "looks": [
                {
                    "family_id": look.family_id,
                    "look_index": look.look_index,
                    "sample_count": look.sample_count,
                    "decision": look.decision,
                    "observed": _json_mapping(look.observed),
                }
                for look in projection
            ],
        }
    )


def _publication_fingerprint(
    *,
    run_id: str,
    task_id: PlannedLookOwnerTaskId,
    projection_fingerprint: str,
    receipt_fingerprints: tuple[tuple[str, tuple[str, ...]], ...],
    terminal_decisions: tuple[tuple[str, LookDecision], ...],
) -> str:
    return _json_fingerprint(
        {
            "contract_version": PLANNED_LOOK_EXECUTION_CONTRACT,
            "run_id": run_id,
            "task_id": task_id,
            "raw_projection_fingerprint": projection_fingerprint,
            "receipt_fingerprints": _fingerprint_mapping_document(receipt_fingerprints),
            "terminal_decisions": dict(terminal_decisions),
            "generation_authorized": False,
        }
    )


def _same_projection(
    left: Sequence[CanonicalG3PlannedLook],
    right: Sequence[CanonicalG3PlannedLook],
) -> bool:
    return bool(
        len(left) == len(right)
        and all(
            first.family_id == second.family_id
            and first.look_index == second.look_index
            and first.sample_count == second.sample_count
            and first.decision == second.decision
            and _json_mapping(first.observed) == _json_mapping(second.observed)
            for first, second in zip(left, right, strict=True)
        )
    )


def _owned_specs(task_id: PlannedLookOwnerTaskId) -> tuple[PlannedLookSpec, ...]:
    result = tuple(
        item for item in G3_PLANNED_LOOK_REGISTRY if item.owner_task_id == task_id
    )
    if len(result) != (2 if "block_calibration" in task_id else 1):
        raise AssertionError("planned-look owner registry geometry changed")
    return result


def _spec_ids(specs: Sequence[PlannedLookSpec]) -> tuple[str, ...]:
    return tuple(item.family_id for item in specs)


def _owner_task(value: str) -> PlannedLookOwnerTaskId:
    if value not in _OWNER_TASKS:
        raise ModelError("task does not own a raw planned-look family")
    return cast(PlannedLookOwnerTaskId, value)


def _fingerprint_mapping_document(
    values: Sequence[tuple[str, tuple[str, ...]]],
) -> dict[str, list[str]]:
    return {family_id: list(fingerprints) for family_id, fingerprints in values}


def _json_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):  # pragma: no cover - type guarantees it.
        raise AssertionError("planned-look observation mapping changed type")
    return normalized


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("planned-look observations contain a non-finite value")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelError("planned-look observation keys must be strings")
            result[key] = _json_value(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    raise ModelError(
        "planned-look observations are not canonical JSON",
        details={"type": type(value).__qualname__},
    )


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
