"""Exact dependency-block publication for canonical G3.

Passing and scientifically failing tasks are published from a producer-sealed
typed decision by :mod:`prpg.model.g3_task_publication`.  A downstream task
whose dependency did not pass has no scientific execution and therefore must
not receive a fabricated task binding, result, RNG identity, or checkpoint.

This module owns that distinct terminal state.  It publishes only the exact
next registry task, proves every direct dependency is itself an exact
canonical publication, binds the complete prior result prefix, and writes a
closed dependency-block certificate.  It deliberately catches no exception:
process, filesystem, integrity, and unexpected execution errors leave the
launch in ``running`` state so the same scientific run can be resumed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, TypeGuard

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
    task_spec,
)
from prpg.model.g3_task_publication import reverify_published_scientific_task
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    LaunchLease,
)

EXACT_BLOCKED_PUBLICATION_SCHEMA_ID: Final = (
    "prpg-g3-exact-dependency-block-publication-v1"
)
EXACT_BLOCKED_PUBLICATION_SCHEMA_VERSION: Final = 1

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BLOCKED_PUBLICATION_CAPABILITY = object()
_MINTED_BLOCKED_PUBLICATIONS: dict[int, ExactBlockedScientificTaskPublication] = {}
_ALLOWED_STATE_COUNTS: Final = frozenset({2, 3, 4, 5})
_SELECTION_TASK: Final = "design_predictive_selection"


@dataclass(frozen=True, slots=True, init=False)
class ExactBlockedScientificTaskPublication:
    """Producer-verified dependency-block result with no science authority."""

    run_id: str
    task_id: str
    selected_n_states: int | None
    blocked_by: tuple[str, ...]
    dependency_result_fingerprints: tuple[tuple[str, str], ...]
    prior_result_fingerprints: tuple[tuple[str, str], ...]
    certificate_fingerprint: str
    task_result: G3TaskResult
    generation_authorized: Literal[False]
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "ExactBlockedScientificTaskPublication is created only by its publisher"
        )


def publish_exact_next_blocked_scientific_task(
    *,
    run_store: CalibrationRunStore,
    lease: LaunchLease,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
) -> ExactBlockedScientificTaskPublication:
    """Publish the exact next task only when a direct dependency did not pass.

    No task ID, status, dependency list, selected K, payload, or checkpoint is
    accepted from the caller.  A next task whose dependencies all passed is a
    scientific execution responsibility and is rejected here.
    """

    _require_active_lease(run_store, lease)
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("blocked publication requires a checkpoint store")
    prior = _exact_result_prefix(run_store, lease.run_id)
    if len(prior) == len(G3_GATE_ORDER):
        raise IntegrityError("all canonical G3 tasks are already published")
    task_id = G3_GATE_ORDER[len(prior)]
    certificate, certificate_fingerprint = _derive_block_certificate(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        run_id=lease.run_id,
        task_id=task_id,
        prior=prior,
    )
    result = run_store.write_task_result(
        lease,
        task_id=task_id,
        status="blocked",
        payload=_blocked_payload(certificate, certificate_fingerprint),
    )
    verified = reverify_exact_blocked_scientific_task(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        run_id=lease.run_id,
        task_id=task_id,
    )
    if result.fingerprint != verified.task_result.fingerprint:
        raise IntegrityError("fresh blocked publication did not replay exactly")
    return verified


def reverify_exact_blocked_scientific_task(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    run_id: str,
    task_id: str,
) -> ExactBlockedScientificTaskPublication:
    """Reload and rederive one dependency-block certificate from live stores."""

    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("blocked publication replay requires a run store")
    if not isinstance(checkpoint_store, CalibrationCheckpointStore):
        raise ModelError("blocked publication replay requires a checkpoint store")
    spec = task_spec(task_id)
    position = G3_GATE_ORDER.index(spec.task_id)
    results = run_store.verify_task_results(run_id, require_complete=False)
    if tuple(results) != G3_GATE_ORDER[: len(results)]:
        raise IntegrityError("G3 task results are not the exact registry prefix")
    if task_id not in results:
        raise IntegrityError("blocked G3 task result is missing")
    prior = MappingProxyType({name: results[name] for name in G3_GATE_ORDER[:position]})
    result = results[task_id]
    if result.status != "blocked":
        raise IntegrityError("requested G3 task is not dependency-blocked")
    certificate, certificate_fingerprint = _derive_block_certificate(
        run_store=run_store,
        checkpoint_store=checkpoint_store,
        live_preflight=live_preflight,
        run_id=run_id,
        task_id=task_id,
        prior=prior,
    )
    expected_payload = _blocked_payload(certificate, certificate_fingerprint)
    if dict(result.payload) != expected_payload:
        raise IntegrityError("dependency-block task payload changed")
    blocked_by = tuple(certificate["blocked_by"])
    dependency_fingerprints = tuple(
        (name, fingerprint)
        for name, fingerprint in certificate["dependency_result_fingerprints"]
    )
    prior_fingerprints = tuple(
        (name, fingerprint)
        for name, fingerprint in certificate["prior_result_fingerprints"]
    )
    if result.blocked_by != blocked_by or result.dependency_fingerprints != dict(
        dependency_fingerprints
    ):
        raise IntegrityError("dependency-block result lineage changed")
    value = object.__new__(ExactBlockedScientificTaskPublication)
    for name, item in (
        ("run_id", run_id),
        ("task_id", task_id),
        ("selected_n_states", certificate["selected_n_states"]),
        ("blocked_by", blocked_by),
        ("dependency_result_fingerprints", dependency_fingerprints),
        ("prior_result_fingerprints", prior_fingerprints),
        ("certificate_fingerprint", certificate_fingerprint),
        ("task_result", result),
        ("generation_authorized", False),
        ("_seal", _BLOCKED_PUBLICATION_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_BLOCKED_PUBLICATIONS[id(value)] = value
    return value


def is_exact_blocked_scientific_task_publication(
    value: object,
) -> TypeGuard[ExactBlockedScientificTaskPublication]:
    """Return whether ``value`` is an unchanged publisher-minted receipt."""

    return (
        isinstance(value, ExactBlockedScientificTaskPublication)
        and value._seal is _BLOCKED_PUBLICATION_CAPABILITY
        and _MINTED_BLOCKED_PUBLICATIONS.get(id(value)) is value
        and value.generation_authorized is False
        and value.task_result.status == "blocked"
        and value.task_result.run_id == value.run_id
        and value.task_result.task_id == value.task_id
        and value.task_result.blocked_by == value.blocked_by
        and value.task_result.dependency_fingerprints
        == dict(value.dependency_result_fingerprints)
        and _SHA256.fullmatch(value.certificate_fingerprint) is not None
    )


def _derive_block_certificate(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    run_id: str,
    task_id: str,
    prior: Mapping[str, G3TaskResult],
) -> tuple[dict[str, Any], str]:
    spec = task_spec(task_id)
    position = G3_GATE_ORDER.index(task_id)
    if tuple(prior) != G3_GATE_ORDER[:position]:
        raise IntegrityError(
            "dependency-block publication lacks the exact prior prefix"
        )
    dependencies: list[tuple[str, str]] = []
    blocked_by: list[str] = []
    for dependency in spec.dependencies:
        result = prior.get(dependency)
        if result is None:
            raise IntegrityError("dependency-block publication lacks a dependency")
        dependencies.append((dependency, result.fingerprint))
        if result.status != "passed":
            blocked_by.append(dependency)
        _reverify_dependency_publication(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live_preflight,
            result=result,
        )
    if not blocked_by:
        raise ModelError(
            "the exact next G3 task has no failed dependency and must execute"
        )
    selected_n_states = _selected_n_states_from_prefix(prior)
    reference = run_store.verify(run_id)
    certificate: dict[str, Any] = {
        "schema_id": EXACT_BLOCKED_PUBLICATION_SCHEMA_ID,
        "schema_version": EXACT_BLOCKED_PUBLICATION_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity": reference.identity.as_dict(),
        "task_id": task_id,
        "selected_n_states": selected_n_states,
        "blocked_by": list(blocked_by),
        "dependency_result_fingerprints": [
            [name, fingerprint] for name, fingerprint in dependencies
        ],
        "prior_result_fingerprints": [
            [name, result.fingerprint] for name, result in prior.items()
        ],
        "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
        "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
        "reason": "upstream_dependency_not_passed",
    }
    return certificate, _fingerprint(certificate)


def _reverify_dependency_publication(
    *,
    run_store: CalibrationRunStore,
    checkpoint_store: CalibrationCheckpointStore,
    live_preflight: CanonicalG3Preflight,
    result: G3TaskResult,
) -> None:
    if result.status in {"passed", "failed"}:
        verified = reverify_published_scientific_task(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            run_id=result.run_id,
            task_id=result.task_id,
            live_preflight=live_preflight,
        )
        if verified.task_result.fingerprint != result.fingerprint:
            raise IntegrityError("dependency scientific publication changed")
        return
    if result.status == "blocked":
        verified_block = reverify_exact_blocked_scientific_task(
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            live_preflight=live_preflight,
            run_id=result.run_id,
            task_id=result.task_id,
        )
        if verified_block.task_result.fingerprint != result.fingerprint:
            raise IntegrityError("dependency-block publication changed")
        return
    raise IntegrityError("dependency task has an invalid terminal status")


def _selected_n_states_from_prefix(
    prior: Mapping[str, G3TaskResult],
) -> int | None:
    selection = prior.get(_SELECTION_TASK)
    if selection is None or selection.status != "passed":
        return None
    selected = selection.payload.get("selected_n_states")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in _ALLOWED_STATE_COUNTS
    ):
        raise IntegrityError("passed selection result lacks the frozen state count")
    for task_id, result in prior.items():
        if G3_GATE_ORDER.index(task_id) <= G3_GATE_ORDER.index(_SELECTION_TASK):
            continue
        reported = result.payload.get("selected_n_states")
        if reported != selected:
            raise IntegrityError(
                "canonical result prefix changed the frozen state count"
            )
    return selected


def _blocked_payload(
    certificate: Mapping[str, Any],
    certificate_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_id": EXACT_BLOCKED_PUBLICATION_SCHEMA_ID,
        "schema_version": EXACT_BLOCKED_PUBLICATION_SCHEMA_VERSION,
        "execution_mode": "canonical",
        "outcome_kind": "dependency_blocked",
        "checkpoint_fingerprint": None,
        "selected_n_states": certificate["selected_n_states"],
        "block_certificate": dict(certificate),
        "block_certificate_fingerprint": certificate_fingerprint,
        "summary": {
            "blocked_by": list(certificate["blocked_by"]),
            "reason": certificate["reason"],
        },
    }


def _require_active_lease(
    run_store: CalibrationRunStore,
    lease: LaunchLease,
) -> None:
    if not isinstance(run_store, CalibrationRunStore):
        raise ModelError("blocked publication requires a run store")
    if (
        not isinstance(lease, LaunchLease)
        or lease.store is not run_store
        or not lease.active
    ):
        raise IntegrityError("blocked publication requires its active launch lease")
    marker = run_store.read_launch(lease.run_id)
    if (
        marker.state != "running"
        or marker.preflight_fingerprint != lease.preflight_fingerprint
        or marker.workers != lease.workers
    ):
        raise IntegrityError("blocked publication launch identity changed")


def _exact_result_prefix(
    run_store: CalibrationRunStore,
    run_id: str,
) -> Mapping[str, G3TaskResult]:
    results = run_store.verify_task_results(run_id, require_complete=False)
    if tuple(results) != G3_GATE_ORDER[: len(results)]:
        raise IntegrityError("G3 task results are not the exact registry prefix")
    return MappingProxyType(results)


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
