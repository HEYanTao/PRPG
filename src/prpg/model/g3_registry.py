"""Closed Phase-3 task graph and sequential-look registry.

This module is deliberately declarative.  It contains no scientific runner and
cannot launch calibration.  The tuples below are the code-owned execution
contract used by the run store, preflight, and final G3 evidence bundle.
Adding, removing, reordering, or rewiring a record changes a content hash and
therefore creates a different scientific run identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from prpg.errors import ModelError


@dataclass(frozen=True, slots=True)
class G3TaskSpec:
    """One immutable G3 task and its direct pass dependencies."""

    task_id: str
    dependencies: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True, slots=True)
class PlannedLookSpec:
    """One closed sequential-look family owned by a G3 task."""

    family_id: str
    owner_task_id: str
    sample_counts: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "owner_task_id": self.owner_task_id,
            "sample_counts": list(self.sample_counts),
        }


# The order is also the exact 26-record order of the G3 evidence bundle.
G3_TASK_REGISTRY: Final = (
    G3TaskSpec("input_integrity", ()),
    G3TaskSpec("design_candidate_viability", ("input_integrity",)),
    G3TaskSpec("design_predictive_selection", ("design_candidate_viability",)),
    G3TaskSpec("sealed_holdout_veto", ("design_predictive_selection",)),
    G3TaskSpec("design_macro_stability", ("design_predictive_selection",)),
    G3TaskSpec("design_latent_correctness", ("design_predictive_selection",)),
    G3TaskSpec("design_refitted_adequacy", ("design_predictive_selection",)),
    G3TaskSpec("design_block_calibration", ("design_predictive_selection",)),
    G3TaskSpec("design_fragmentation", ("design_block_calibration",)),
    G3TaskSpec(
        "design_dependence",
        ("design_block_calibration", "design_fragmentation"),
    ),
    G3TaskSpec("production_fit_same_k", ("sealed_holdout_veto",)),
    G3TaskSpec("production_macro_stability", ("production_fit_same_k",)),
    G3TaskSpec("production_latent_correctness", ("production_fit_same_k",)),
    G3TaskSpec("production_refitted_adequacy", ("production_fit_same_k",)),
    G3TaskSpec("production_block_calibration", ("production_fit_same_k",)),
    G3TaskSpec("production_fragmentation", ("production_block_calibration",)),
    G3TaskSpec(
        "production_dependence",
        ("production_block_calibration", "production_fragmentation"),
    ),
    G3TaskSpec(
        "four_cell_adequacy_holm",
        (
            "design_latent_correctness",
            "design_refitted_adequacy",
            "production_latent_correctness",
            "production_refitted_adequacy",
        ),
    ),
    G3TaskSpec(
        "four_cell_dependence_holm",
        ("design_dependence", "production_dependence"),
    ),
    G3TaskSpec(
        "bridge_resource_preflight",
        (
            "design_macro_stability",
            "production_macro_stability",
            "design_fragmentation",
            "production_fragmentation",
            "four_cell_adequacy_holm",
            "four_cell_dependence_holm",
        ),
    ),
    G3TaskSpec("bridge_tier_a_model_dgp", ("bridge_resource_preflight",)),
    G3TaskSpec("bridge_tier_a_nonparametric_dgp", ("bridge_resource_preflight",)),
    # Both Tier-A reference laws must pass before either fixed-K Tier-B tail
    # experiment may run.  This mirrors the approved protocol and the bridge
    # driver's fail-closed execution boundary in the durable task DAG.
    G3TaskSpec(
        "bridge_tier_b_model_dgp",
        (
            "bridge_tier_a_model_dgp",
            "bridge_tier_a_nonparametric_dgp",
        ),
    ),
    G3TaskSpec(
        "bridge_tier_b_nonparametric_dgp",
        (
            "bridge_tier_a_model_dgp",
            "bridge_tier_a_nonparametric_dgp",
        ),
    ),
    G3TaskSpec(
        "actual_artifact_bridge",
        (
            "bridge_tier_a_model_dgp",
            "bridge_tier_a_nonparametric_dgp",
            "bridge_tier_b_model_dgp",
            "bridge_tier_b_nonparametric_dgp",
        ),
    ),
    G3TaskSpec("artifact_integrity", ("actual_artifact_bridge",)),
)

G3_GATE_ORDER: Final = tuple(item.task_id for item in G3_TASK_REGISTRY)
G3_TASK_BY_ID: Final = MappingProxyType(
    {item.task_id: item for item in G3_TASK_REGISTRY}
)

_KERNEL_LOOKS = (10_000, 20_000, 40_000, 80_000, 160_000, 320_000)
_BRIDGE_LOOKS = tuple(range(1_000, 10_001, 1_000))

G3_PLANNED_LOOK_REGISTRY: Final = (
    PlannedLookSpec("kernel_design_monthly", "design_block_calibration", _KERNEL_LOOKS),
    PlannedLookSpec("kernel_design_daily", "design_block_calibration", _KERNEL_LOOKS),
    PlannedLookSpec(
        "kernel_production_monthly", "production_block_calibration", _KERNEL_LOOKS
    ),
    PlannedLookSpec(
        "kernel_production_daily", "production_block_calibration", _KERNEL_LOOKS
    ),
    PlannedLookSpec(
        "bridge_tier_b_model_dgp", "bridge_tier_b_model_dgp", _BRIDGE_LOOKS
    ),
    PlannedLookSpec(
        "bridge_tier_b_nonparametric_dgp",
        "bridge_tier_b_nonparametric_dgp",
        _BRIDGE_LOOKS,
    ),
)
G3_PLANNED_LOOK_BY_ID: Final = MappingProxyType(
    {item.family_id: item for item in G3_PLANNED_LOOK_REGISTRY}
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


G3_TASK_REGISTRY_FINGERPRINT: Final = hashlib.sha256(
    _canonical_bytes([item.as_dict() for item in G3_TASK_REGISTRY])
).hexdigest()
G3_PLANNED_LOOK_REGISTRY_FINGERPRINT: Final = hashlib.sha256(
    _canonical_bytes([item.as_dict() for item in G3_PLANNED_LOOK_REGISTRY])
).hexdigest()


def task_spec(task_id: str) -> G3TaskSpec:
    """Resolve a task only from the closed registry."""

    try:
        return G3_TASK_BY_ID[task_id]
    except (KeyError, TypeError) as error:
        raise ModelError("G3 task ID is not in the closed registry") from error


def planned_look_spec(family_id: str) -> PlannedLookSpec:
    """Resolve a sequential-look family only from the closed registry."""

    try:
        return G3_PLANNED_LOOK_BY_ID[family_id]
    except (KeyError, TypeError) as error:
        raise ModelError(
            "G3 planned-look family is not in the closed registry"
        ) from error


def failure_propagation(failed_task_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return every downstream task that must be blocked by given failures."""

    failed = set(failed_task_ids)
    if not failed or len(failed) != len(failed_task_ids):
        raise ModelError("failed G3 task IDs must be non-empty and unique")
    for task_id in failed:
        task_spec(task_id)
    blocked: set[str] = set()
    changed = True
    while changed:
        changed = False
        unavailable = failed | blocked
        for item in G3_TASK_REGISTRY:
            if item.task_id in unavailable:
                continue
            if any(dependency in unavailable for dependency in item.dependencies):
                blocked.add(item.task_id)
                changed = True
    return tuple(task_id for task_id in G3_GATE_ORDER if task_id in blocked)


def _validate_registries() -> None:
    if len(G3_TASK_REGISTRY) != 26 or len(set(G3_GATE_ORDER)) != 26:
        raise RuntimeError("G3 registry must contain exactly 26 unique records")
    positions = {task_id: index for index, task_id in enumerate(G3_GATE_ORDER)}
    for index, task in enumerate(G3_TASK_REGISTRY):
        if not task.task_id or len(set(task.dependencies)) != len(task.dependencies):
            raise RuntimeError("G3 task registry contains an invalid record")
        if any(dependency not in positions for dependency in task.dependencies):
            raise RuntimeError("G3 task dependency is unregistered")
        if any(positions[dependency] >= index for dependency in task.dependencies):
            raise RuntimeError("G3 task registry is not a topological order")
    family_ids = tuple(look.family_id for look in G3_PLANNED_LOOK_REGISTRY)
    if len(set(family_ids)) != len(family_ids):
        raise RuntimeError("G3 planned-look family IDs must be unique")
    for look in G3_PLANNED_LOOK_REGISTRY:
        if look.owner_task_id not in positions:
            raise RuntimeError("G3 planned-look owner is unregistered")
        if (
            not look.sample_counts
            or tuple(sorted(set(look.sample_counts))) != look.sample_counts
            or look.sample_counts[0] <= 0
        ):
            raise RuntimeError("G3 planned-look counts must be positive and increasing")


_validate_registries()
