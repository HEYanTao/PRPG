"""Closed, crash-resumable orchestration for the exact Phase-3 task graph.

This module exposes a callable-based smoke/test path and a separate canonical
publication path.  Smoke results are labelled in every durable task payload
and completion never means generation authorization.  Canonical publication
requires fresh code-owned preflight authority, the exact-field typed handler
bundle, and typed checkpoints bound to durable launch evidence; it fails
closed when any of those are absent.  Arbitrary callables and arbitrary JSON
receipts are not a canonical checkpoint contract.

The coordinator owns graph traversal, dependency blocking, planned-look
durability, selected-K freezing, and launch completion.  Smoke handlers own
only one unblocked task outcome; canonical typed sources are re-derived and
published by :mod:`prpg.model.g3_science_handlers`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from prpg.errors import IntegrityError, ModelError, NotImplementedStageError
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_preflight import (
    CanonicalG3Preflight,
    canonical_resume_preflight_fingerprint,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_TASK_REGISTRY,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    LaunchLease,
    LookDecision,
    PlannedLookReceipt,
)

if TYPE_CHECKING:
    from prpg.model.g3_science_handlers import (
        CanonicalG3ArtifactPublicationPrefix,
        CanonicalG3CoordinationResult,
        CanonicalG3FinalizationResult,
        CanonicalG3First25HandlerBundle,
        CanonicalG3HandlerBundle,
    )
    from prpg.model.scientific_artifact_pair import LoadedScientificArtifactPair

SmokeTaskStatus: TypeAlias = Literal["passed", "failed"]
_EXECUTION_MODE: Literal["smoke"] = "smoke"
_COORDINATOR_SCHEMA_VERSION = 1
_SELECTED_K_TASK = "design_predictive_selection"
_ALLOWED_STATE_COUNTS = frozenset({2, 3, 4, 5})
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class G3PlannedLookDecision:
    """One handler-produced decision at one registered sequential look."""

    family_id: str
    look_index: int
    sample_count: int
    decision: LookDecision
    observed: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class G3SmokeTaskOutcome:
    """A smoke handler outcome; ``blocked`` is coordinator-derived only."""

    status: SmokeTaskStatus
    evidence: Mapping[str, Any]
    selected_n_states: int | None
    planned_looks: tuple[G3PlannedLookDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class G3SmokeTaskContext:
    """Read-only state supplied to one smoke handler."""

    run_id: str
    task_id: str
    selected_n_states: int | None
    prior_results: Mapping[str, G3TaskResult]
    existing_planned_looks: Mapping[str, tuple[PlannedLookReceipt, ...]]


G3SmokeTaskHandler: TypeAlias = Callable[[G3SmokeTaskContext], G3SmokeTaskOutcome]


@dataclass(frozen=True, slots=True)
class G3SmokeHandlerSet:
    """An exact 26-handler set that is permanently restricted to smoke use."""

    handlers: Mapping[str, G3SmokeTaskHandler]

    def __post_init__(self) -> None:
        if not isinstance(self.handlers, Mapping):
            raise ModelError("G3 smoke handlers must be a mapping")
        copied = dict(self.handlers)
        if frozenset(copied) != frozenset(G3_GATE_ORDER):
            raise ModelError("G3 smoke handlers must contain the exact 26 task IDs")
        if any(not callable(copied[task_id]) for task_id in G3_GATE_ORDER):
            raise ModelError("every G3 smoke task handler must be callable")
        object.__setattr__(self, "handlers", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class G3SmokeCoordinationResult:
    """Terminal smoke result; it can never authorize path generation."""

    run_id: str
    task_results: tuple[G3TaskResult, ...]
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    selected_n_states: int | None
    completion_fingerprint: str
    all_tasks_passed: bool
    execution_mode: Literal["smoke"] = "smoke"
    generation_authorized: Literal[False] = False


@dataclass(frozen=True, slots=True)
class _ValidatedOutcome:
    status: SmokeTaskStatus
    evidence: Mapping[str, Any]
    selected_n_states: int | None
    planned_looks: tuple[G3PlannedLookDecision, ...]


def coordinate_g3_smoke_launch(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    handlers: G3SmokeHandlerSet,
) -> G3SmokeCoordinationResult:
    """Run or resume the exact 26-task DAG using smoke-only handlers.

    Existing task results are hash/dependency verified, checked for the exact
    smoke payload envelope, and skipped.  A handler returning planned looks
    must return the complete deterministic sequence for every family it owns;
    an existing durable prefix is byte-semantically compared and only its
    missing suffix is written.

    ``ModelError`` raised *by a handler* is a scientific task failure.  Other
    exceptions propagate as programming/operational errors and leave the
    launch RUNNING for an explicit resume.  Validation and store errors are
    likewise never converted into favorable or scientific outcomes.
    """

    _validate_active_lease(store, lease)
    if not isinstance(handlers, G3SmokeHandlerSet):
        raise ModelError("the G3 smoke coordinator requires a smoke handler set")

    existing_results = store.verify_task_results(lease.run_id, require_complete=False)
    _validate_existing_smoke_results(existing_results)
    store.verify_planned_look_receipts(lease.run_id, require_terminal=False)

    processed: dict[str, G3TaskResult] = {}
    selected_n_states: int | None = None

    for spec in G3_TASK_REGISTRY:
        prior = existing_results.get(spec.task_id)
        if prior is not None:
            processed[spec.task_id] = prior
            if spec.task_id == _SELECTED_K_TASK and prior.status == "passed":
                selected_n_states = _selected_k_from_payload(prior)
            continue

        blocked_by = tuple(
            dependency
            for dependency in spec.dependencies
            if processed[dependency].status != "passed"
        )
        if blocked_by:
            evidence = {
                "blocked_by": list(blocked_by),
                "reason": "upstream_dependency_not_passed",
            }
            result = store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="blocked",
                payload=_task_payload(
                    evidence=evidence,
                    selected_n_states=selected_n_states,
                ),
            )
            processed[spec.task_id] = result
            continue

        current_looks = store.verify_planned_look_receipts(
            lease.run_id, require_terminal=False
        )
        owned_looks = _owned_look_mapping(spec.task_id, current_looks)
        context = G3SmokeTaskContext(
            run_id=lease.run_id,
            task_id=spec.task_id,
            selected_n_states=selected_n_states,
            prior_results=MappingProxyType(dict(processed)),
            existing_planned_looks=owned_looks,
        )
        handler = handlers.handlers[spec.task_id]
        try:
            supplied = handler(context)
        except ModelError as error:
            supplied = _scientific_failure_outcome(
                task_id=spec.task_id,
                selected_n_states=selected_n_states,
                existing_looks=owned_looks,
                error=error,
            )

        validated = _validate_outcome(
            task_id=spec.task_id,
            supplied=supplied,
            frozen_selected_n_states=selected_n_states,
        )
        _commit_planned_looks(
            store=store,
            lease=lease,
            task_id=spec.task_id,
            decisions=validated.planned_looks,
        )
        result = store.write_task_result(
            lease,
            task_id=spec.task_id,
            status=validated.status,
            payload=_task_payload(
                evidence=validated.evidence,
                selected_n_states=validated.selected_n_states,
            ),
        )
        processed[spec.task_id] = result
        if spec.task_id == _SELECTED_K_TASK and validated.status == "passed":
            selected_n_states = validated.selected_n_states

    completion = store.complete_launch(lease)
    ordered = tuple(processed[task_id] for task_id in G3_GATE_ORDER)
    failed = tuple(item.task_id for item in ordered if item.status == "failed")
    blocked = tuple(item.task_id for item in ordered if item.status == "blocked")
    return G3SmokeCoordinationResult(
        run_id=lease.run_id,
        task_results=ordered,
        failed_task_ids=failed,
        blocked_task_ids=blocked,
        selected_n_states=selected_n_states,
        completion_fingerprint=completion.fingerprint,
        all_tasks_passed=not failed and not blocked,
    )


def coordinate_g3_canonical_launch(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
    handlers: CanonicalG3HandlerBundle | None = None,
) -> CanonicalG3CoordinationResult:
    """Run the code-owned typed canonical publisher, or fail closed if absent.

    Merely replaying hash-valid JSON task receipts is insufficient to restore
    model objects, arrays, RNG coordinates, and other typed scientific state.
    This entry therefore accepts only the exact-field
    :class:`CanonicalG3HandlerBundle`, never a task/callable mapping.  It
    verifies the active lease and a fresh code-owned preflight against durable
    evidence before delegating to the typed checkpoint publisher.  Omitting
    the bundle retains the historical fail-closed behavior and performs no
    launch mutation.
    """

    _validate_active_lease(store, lease)
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("canonical G3 coordination requires a checkpoint store")
    preflight_fingerprint = canonical_resume_preflight_fingerprint(
        store,
        run_id=lease.run_id,
        live_preflight=live_preflight,
    )
    if preflight_fingerprint != lease.preflight_fingerprint:
        raise IntegrityError("canonical preflight does not match the launch lease")
    if handlers is None:
        existing = store.verify_task_results(lease.run_id, require_complete=False)
        for task_id in G3_GATE_ORDER:
            result = existing.get(task_id)
            if result is not None and result.status == "passed":
                checkpoint_store.load_from_task_result(
                    store,
                    run_id=lease.run_id,
                    task_id=task_id,
                )
        raise NotImplementedStageError(
            "canonical G3 coordination requires a code-owned typed scientific "
            "handler and checkpoint-publisher bundle"
        )
    from prpg.model.g3_science_handlers import (
        CanonicalG3HandlerBundle,
        coordinate_canonical_g3_handlers,
    )

    if not isinstance(handlers, CanonicalG3HandlerBundle):
        raise ModelError("canonical G3 handlers have the wrong closed-bundle type")
    return coordinate_canonical_g3_handlers(
        store=store,
        lease=lease,
        live_preflight=live_preflight,
        checkpoint_store=checkpoint_store,
        handlers=handlers,
    )


def coordinate_g3_canonical_first_25(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
    handlers: CanonicalG3First25HandlerBundle,
) -> CanonicalG3ArtifactPublicationPrefix:
    """Canonical stage 1: publish exactly tasks 1--25, never artifacts."""

    from prpg.model.g3_science_handlers import (
        CanonicalG3First25HandlerBundle,
        coordinate_canonical_g3_first_25,
    )

    if not isinstance(handlers, CanonicalG3First25HandlerBundle):
        raise ModelError("canonical stage 1 requires the first-25 handler bundle")
    return coordinate_canonical_g3_first_25(
        store=store,
        lease=lease,
        live_preflight=live_preflight,
        checkpoint_store=checkpoint_store,
        handlers=handlers,
    )


def finalize_g3_canonical_artifacts(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    live_preflight: CanonicalG3Preflight,
    checkpoint_store: CalibrationCheckpointStore,
    prefix: CanonicalG3ArtifactPublicationPrefix,
    artifact_pair: LoadedScientificArtifactPair,
) -> CanonicalG3FinalizationResult:
    """Canonical stage 2: bind one receipt-backed pair and publish task 26."""

    from prpg.model.g3_science_handlers import finalize_canonical_g3_artifacts

    return finalize_canonical_g3_artifacts(
        store=store,
        lease=lease,
        live_preflight=live_preflight,
        checkpoint_store=checkpoint_store,
        prefix=prefix,
        artifact_pair=artifact_pair,
    )


def _validate_active_lease(store: CalibrationRunStore, lease: LaunchLease) -> None:
    if not isinstance(store, CalibrationRunStore):
        raise ModelError("G3 coordinator store has the wrong type")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not store
        or not lease.active
    ):
        raise IntegrityError("an active launch lease from this store is required")
    marker = store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != lease.preflight_fingerprint
        or marker.workers != lease.workers
    ):
        raise IntegrityError("launch lease does not match the running marker")


def _validate_existing_smoke_results(
    results: Mapping[str, G3TaskResult],
) -> None:
    selected_n_states: int | None = None
    for spec in G3_TASK_REGISTRY:
        result = results.get(spec.task_id)
        if result is None:
            continue
        payload = result.payload
        required = {
            "coordinator_schema_version",
            "evidence",
            "execution_mode",
            "selected_n_states",
        }
        if set(payload) != required:
            raise IntegrityError("existing task lacks the smoke coordinator envelope")
        if (
            payload["coordinator_schema_version"] != _COORDINATOR_SCHEMA_VERSION
            or payload["execution_mode"] != _EXECUTION_MODE
            or not isinstance(payload["evidence"], Mapping)
        ):
            raise IntegrityError("existing smoke task envelope is invalid")
        recorded_k = payload["selected_n_states"]
        if spec.task_id == _SELECTED_K_TASK:
            if result.status == "passed":
                try:
                    selected_n_states = _validated_state_count(
                        recorded_k, "existing selected state count"
                    )
                except ModelError as error:
                    raise IntegrityError(
                        "existing selected state count is invalid"
                    ) from error
            elif recorded_k is not None:
                raise IntegrityError("a failed selection cannot freeze a state count")
        elif selected_n_states is None:
            if recorded_k is not None:
                raise IntegrityError("state count appears before a passed selection")
        elif recorded_k != selected_n_states:
            raise IntegrityError(
                "existing task changed the frozen selected state count"
            )
        if result.status == "blocked":
            expected = {
                "blocked_by": list(result.blocked_by),
                "reason": "upstream_dependency_not_passed",
            }
            if dict(payload["evidence"]) != expected:
                raise IntegrityError(
                    "existing blocked result was not coordinator-derived"
                )


def _selected_k_from_payload(result: G3TaskResult) -> int:
    return _validated_state_count(
        result.payload["selected_n_states"], "selected state count"
    )


def _validate_outcome(
    *,
    task_id: str,
    supplied: G3SmokeTaskOutcome,
    frozen_selected_n_states: int | None,
) -> _ValidatedOutcome:
    if not isinstance(supplied, G3SmokeTaskOutcome):
        raise ModelError("G3 smoke handlers must return G3SmokeTaskOutcome")
    if supplied.status not in {"passed", "failed"}:
        raise ModelError("G3 smoke task outcome status is invalid")
    evidence = _normalized_mapping(supplied.evidence, "G3 smoke task evidence")

    if task_id == _SELECTED_K_TASK:
        if supplied.status == "passed":
            selected = _validated_state_count(
                supplied.selected_n_states, "selected state count"
            )
        elif supplied.selected_n_states is not None:
            raise ModelError("a failed selection cannot freeze a state count")
        else:
            selected = None
    elif frozen_selected_n_states is None:
        if supplied.selected_n_states is not None:
            raise ModelError("state count cannot appear before a passed selection")
        selected = None
    else:
        if supplied.selected_n_states != frozen_selected_n_states:
            raise ModelError("G3 task outcome changed the frozen selected state count")
        selected = frozen_selected_n_states

    planned = _validated_planned_looks(
        task_id=task_id,
        supplied=supplied.planned_looks,
        task_status=supplied.status,
    )
    return _ValidatedOutcome(
        supplied.status,
        MappingProxyType(evidence),
        selected,
        planned,
    )


def _validated_planned_looks(
    *,
    task_id: str,
    supplied: tuple[G3PlannedLookDecision, ...],
    task_status: SmokeTaskStatus,
) -> tuple[G3PlannedLookDecision, ...]:
    if not isinstance(supplied, tuple):
        raise ModelError("planned-look decisions must be an immutable tuple")
    specs = tuple(
        item for item in G3_PLANNED_LOOK_REGISTRY if item.owner_task_id == task_id
    )
    if not specs:
        if supplied:
            raise ModelError("task supplied a planned-look family it does not own")
        return ()

    by_family: dict[str, list[G3PlannedLookDecision]] = {
        spec.family_id: [] for spec in specs
    }
    family_positions = {spec.family_id: index for index, spec in enumerate(specs)}
    last_position = -1
    for item in supplied:
        if not isinstance(item, G3PlannedLookDecision):
            raise ModelError("planned looks must use G3PlannedLookDecision")
        position = family_positions.get(item.family_id)
        if position is None:
            raise ModelError("task supplied a planned-look family it does not own")
        if position < last_position:
            raise ModelError("planned-look families are outside registry order")
        last_position = position
        by_family[item.family_id].append(item)

    normalized: list[G3PlannedLookDecision] = []
    for spec in specs:
        family = by_family[spec.family_id]
        if not family:
            raise ModelError("every owned planned-look family must terminate")
        for expected_index, item in enumerate(family, start=1):
            if isinstance(item.look_index, bool) or item.look_index != expected_index:
                raise ModelError("planned looks must be sequential without gaps")
            if expected_index > len(spec.sample_counts):
                raise ModelError("planned-look sequence exceeds the closed schedule")
            if item.sample_count != spec.sample_counts[expected_index - 1]:
                raise ModelError("planned-look sample count differs from the registry")
            if item.decision not in {"continue", "pass", "fail"}:
                raise ModelError("planned-look decision is invalid")
            is_last = expected_index == len(family)
            if is_last and item.decision == "continue":
                raise ModelError("every owned planned-look family must terminate")
            if not is_last and item.decision != "continue":
                raise ModelError("no planned look may follow a terminal decision")
            observed = _normalized_mapping(
                item.observed,
                "G3 smoke planned-look observations",
            )
            normalized.append(
                G3PlannedLookDecision(
                    item.family_id,
                    item.look_index,
                    item.sample_count,
                    item.decision,
                    MappingProxyType(observed),
                )
            )
        if task_status == "passed" and family[-1].decision != "pass":
            raise ModelError("a passed task requires passing planned-look decisions")
    return tuple(normalized)


def _commit_planned_looks(
    *,
    store: CalibrationRunStore,
    lease: LaunchLease,
    task_id: str,
    decisions: tuple[G3PlannedLookDecision, ...],
) -> None:
    current = store.verify_planned_look_receipts(lease.run_id, require_terminal=False)
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if spec.owner_task_id != task_id:
            continue
        proposed = tuple(item for item in decisions if item.family_id == spec.family_id)
        existing = current.get(spec.family_id, ())
        if len(existing) > len(proposed):
            raise IntegrityError(
                "existing planned-look sequence exceeds handler output"
            )
        for prior, candidate in zip(existing, proposed, strict=False):
            if (
                prior.look_index != candidate.look_index
                or prior.sample_count != candidate.sample_count
                or prior.decision != candidate.decision
                or dict(prior.observed) != dict(candidate.observed)
            ):
                raise IntegrityError(
                    "existing planned-look prefix differs from handler output"
                )
        for candidate in proposed[len(existing) :]:
            store.write_planned_look_receipt(
                lease,
                family_id=candidate.family_id,
                look_index=candidate.look_index,
                sample_count=candidate.sample_count,
                decision=candidate.decision,
                observed=candidate.observed,
            )


def _scientific_failure_outcome(
    *,
    task_id: str,
    selected_n_states: int | None,
    existing_looks: Mapping[str, tuple[PlannedLookReceipt, ...]],
    error: ModelError,
) -> G3SmokeTaskOutcome:
    message = error.message
    if not isinstance(message, str) or not message:
        message = "model calculation failed"
    evidence = {
        "error_type": type(error).__name__,
        "exit_code": int(error.exit_code),
        "message": message[:1_000],
        "reason": "scientific_model_error",
    }
    decisions: list[G3PlannedLookDecision] = []
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if spec.owner_task_id != task_id:
            continue
        prior = existing_looks.get(spec.family_id, ())
        decisions.extend(
            G3PlannedLookDecision(
                item.family_id,
                item.look_index,
                item.sample_count,
                item.decision,
                item.observed,
            )
            for item in prior
        )
        if not prior or prior[-1].decision == "continue":
            next_index = len(prior) + 1
            decisions.append(
                G3PlannedLookDecision(
                    family_id=spec.family_id,
                    look_index=next_index,
                    sample_count=spec.sample_counts[next_index - 1],
                    decision="fail",
                    observed={
                        "coordinator_scientific_failure": True,
                        "error_type": type(error).__name__,
                        "exit_code": int(error.exit_code),
                    },
                )
            )
    return G3SmokeTaskOutcome(
        status="failed",
        evidence=evidence,
        selected_n_states=selected_n_states,
        planned_looks=tuple(decisions),
    )


def _owned_look_mapping(
    task_id: str,
    all_looks: Mapping[str, tuple[PlannedLookReceipt, ...]],
) -> Mapping[str, tuple[PlannedLookReceipt, ...]]:
    owned = {
        spec.family_id: all_looks.get(spec.family_id, ())
        for spec in G3_PLANNED_LOOK_REGISTRY
        if spec.owner_task_id == task_id
    }
    return MappingProxyType(owned)


def _task_payload(
    *,
    evidence: Mapping[str, Any],
    selected_n_states: int | None,
) -> dict[str, Any]:
    return {
        "coordinator_schema_version": _COORDINATOR_SCHEMA_VERSION,
        "evidence": dict(evidence),
        "execution_mode": _EXECUTION_MODE,
        "selected_n_states": selected_n_states,
    }


def _validated_state_count(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in _ALLOWED_STATE_COUNTS
    ):
        raise ModelError(f"{label} must be one of 2, 3, 4, or 5")
    return value


def _normalized_mapping(
    value: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ModelError(f"{label} must be a non-empty mapping")
    normalized = _normalized_json(dict(value), path=label)
    if not isinstance(normalized, dict):  # pragma: no cover - construction invariant
        raise AssertionError("mapping normalization returned a non-mapping")
    return normalized


def _normalized_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError(f"{path} contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ModelError(f"{path} contains an invalid key")
        result: dict[str, Any] = {}
        for key in sorted(keys):
            if any(marker in key.casefold() for marker in _SECRET_MARKERS):
                raise ModelError(f"{path} contains a secret-like key")
            result[key] = _normalized_json(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_normalized_json(item, path=f"{path}[]") for item in value]
    raise ModelError(f"{path} contains an unsupported JSON value")
