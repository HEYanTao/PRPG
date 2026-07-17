from __future__ import annotations

import hashlib
from typing import Literal, cast

import pytest

import prpg.model.g3_boundary_views as boundary_views
from prpg.model.bridge_driver import (
    StagedBridgePlannedLookEvidence,
    StagedBridgePlannedLookIdentity,
    StagedBridgeTaskEvidence,
    StagedBridgeTaskEvidenceIdentity,
    StagedBridgeTaskId,
)
from prpg.model.bridge_history import BridgeDGP
from prpg.model.g3_boundary_views import BridgeDriverTaskViews

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_D = "d" * 64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _task_evidence(
    task_id: StagedBridgeTaskId,
    *,
    dgp: BridgeDGP | None,
    parents: tuple[tuple[str, str], ...],
    event_index: int,
) -> StagedBridgeTaskEvidence:
    value = object.__new__(StagedBridgeTaskEvidence)
    for name, item in (
        ("task_id", task_id),
        ("dgp", dgp),
        ("passed", True),
        ("common_binding_fingerprint", _SHA_A),
        ("parent_task_fingerprints", parents),
        ("source_materialization_fingerprint", _sha(f"material:{task_id}")),
        (
            "rng_execution_fingerprint",
            None if task_id == "actual_artifact_bridge" else _sha(f"rng:{task_id}"),
        ),
        ("result_fingerprint", _sha(f"result:{task_id}")),
        ("evidence_fingerprint", _sha(f"evidence:{task_id}")),
        ("stage_receipt_fingerprint", _sha(f"receipt:{task_id}")),
        ("_session", cast(object, None)),
        ("_event_index", event_index),
        ("_seal", cast(object, None)),
    ):
        object.__setattr__(value, name, item)
    return value


def _planned_look(
    task: StagedBridgeTaskEvidence,
    *,
    dgp: BridgeDGP,
    event_index: int,
) -> StagedBridgePlannedLookEvidence:
    task_id = cast(
        Literal["bridge_tier_b_model_dgp", "bridge_tier_b_nonparametric_dgp"],
        task.task_id,
    )
    value = object.__new__(StagedBridgePlannedLookEvidence)
    for name, item in (
        ("task_id", task_id),
        ("dgp", dgp),
        ("look_index", 1),
        ("sample_count", 1_000),
        ("exceedances", 1_000),
        ("decision", "pass_above_tail"),
        ("common_binding_fingerprint", task.common_binding_fingerprint),
        ("parent_task_fingerprints", task.parent_task_fingerprints),
        ("source_materialization_fingerprint", task.source_materialization_fingerprint),
        ("rng_execution_fingerprint", cast(str, task.rng_execution_fingerprint)),
        ("result_fingerprint", task.result_fingerprint),
        ("evidence_fingerprint", _sha(f"look-evidence:{task_id}:1")),
        ("stage_receipt_fingerprint", _sha(f"look-receipt:{task_id}:1")),
        ("_session", cast(object, None)),
        ("_event_index", event_index),
        ("_seal", cast(object, None)),
    ):
        object.__setattr__(value, name, item)
    return value


def _task_identity(value: StagedBridgeTaskEvidence) -> StagedBridgeTaskEvidenceIdentity:
    return StagedBridgeTaskEvidenceIdentity(
        task_id=value.task_id,
        dgp=value.dgp,
        passed=value.passed,
        base_preflight_fingerprint=_SHA_B,
        selected_n_states=2,
        common_binding_fingerprint=value.common_binding_fingerprint,
        parent_task_fingerprints=value.parent_task_fingerprints,
        source_materialization_fingerprint=value.source_materialization_fingerprint,
        rng_execution_fingerprint=value.rng_execution_fingerprint,
        result_fingerprint=value.result_fingerprint,
        evidence_fingerprint=value.evidence_fingerprint,
        stage_receipt_fingerprint=value.stage_receipt_fingerprint,
    )


def _look_identity(
    value: StagedBridgePlannedLookEvidence,
) -> StagedBridgePlannedLookIdentity:
    return StagedBridgePlannedLookIdentity(
        task_id=value.task_id,
        dgp=value.dgp,
        look_index=value.look_index,
        sample_count=value.sample_count,
        exceedances=value.exceedances,
        decision=value.decision,
        common_binding_fingerprint=value.common_binding_fingerprint,
        parent_task_fingerprints=value.parent_task_fingerprints,
        source_materialization_fingerprint=value.source_materialization_fingerprint,
        rng_execution_fingerprint=value.rng_execution_fingerprint,
        result_fingerprint=value.result_fingerprint,
        evidence_fingerprint=value.evidence_fingerprint,
        stage_receipt_fingerprint=value.stage_receipt_fingerprint,
    )


@pytest.fixture
def passing_bridge_views(
    monkeypatch: pytest.MonkeyPatch,
) -> BridgeDriverTaskViews:
    """Return six canonical views over a compact synthetic staged prefix.

    The staged bridge driver tests exercise producer seals, append-only recovery,
    and exact-prefix replay.  Downstream G3 tests only need a deterministic,
    correctly shaped staged prefix, so this fixture substitutes the three live
    identity readers while retaining the real boundary builder and view guard.
    """

    resource = _task_evidence(
        "bridge_resource_preflight", dgp=None, parents=(), event_index=1
    )
    resource_parent = ((resource.task_id, resource.evidence_fingerprint),)
    tier_a_model = _task_evidence(
        "bridge_tier_a_model_dgp",
        dgp="model",
        parents=resource_parent,
        event_index=2,
    )
    tier_a_nonparametric = _task_evidence(
        "bridge_tier_a_nonparametric_dgp",
        dgp="nonparametric",
        parents=resource_parent,
        event_index=3,
    )
    tier_b_parents = (
        (resource.task_id, resource.evidence_fingerprint),
        (tier_a_model.task_id, tier_a_model.evidence_fingerprint),
        (tier_a_nonparametric.task_id, tier_a_nonparametric.evidence_fingerprint),
    )
    tier_b_model = _task_evidence(
        "bridge_tier_b_model_dgp",
        dgp="model",
        parents=tier_b_parents,
        event_index=5,
    )
    tier_b_nonparametric = _task_evidence(
        "bridge_tier_b_nonparametric_dgp",
        dgp="nonparametric",
        parents=tier_b_parents,
        event_index=7,
    )
    actual = _task_evidence(
        "actual_artifact_bridge",
        dgp=None,
        parents=(
            (tier_a_model.task_id, tier_a_model.evidence_fingerprint),
            (
                tier_a_nonparametric.task_id,
                tier_a_nonparametric.evidence_fingerprint,
            ),
            (tier_b_model.task_id, tier_b_model.evidence_fingerprint),
            (
                tier_b_nonparametric.task_id,
                tier_b_nonparametric.evidence_fingerprint,
            ),
        ),
        event_index=8,
    )
    task_sources = {
        item.task_id: item
        for item in (
            resource,
            tier_a_model,
            tier_a_nonparametric,
            tier_b_model,
            tier_b_nonparametric,
            actual,
        )
    }
    planned_looks = {
        tier_b_model.task_id: (
            _planned_look(tier_b_model, dgp="model", event_index=4),
        ),
        tier_b_nonparametric.task_id: (
            _planned_look(
                tier_b_nonparametric,
                dgp="nonparametric",
                event_index=6,
            ),
        ),
    }
    monkeypatch.setattr(
        boundary_views,
        "staged_bridge_task_evidence_identity",
        _task_identity,
    )
    monkeypatch.setattr(
        boundary_views,
        "staged_bridge_planned_look_identity",
        _look_identity,
    )
    monkeypatch.setattr(
        boundary_views,
        "staged_bridge_task_planned_looks",
        lambda value: planned_looks.get(value.task_id, ()),
    )

    views = {
        task_id: boundary_views.build_bridge_driver_task_view(
            task_sources[task_id],
            task_id=task_id,
            run_id=_SHA_D,
            binding_fingerprint=_sha(f"task-binding:{task_id}"),
        )
        for task_id in boundary_views.BRIDGE_TASK_IDS
    }
    return BridgeDriverTaskViews(
        bridge_resource_preflight=views["bridge_resource_preflight"],
        bridge_tier_a_model_dgp=views["bridge_tier_a_model_dgp"],
        bridge_tier_a_nonparametric_dgp=(views["bridge_tier_a_nonparametric_dgp"]),
        bridge_tier_b_model_dgp=views["bridge_tier_b_model_dgp"],
        bridge_tier_b_nonparametric_dgp=(views["bridge_tier_b_nonparametric_dgp"]),
        actual_artifact_bridge=views["actual_artifact_bridge"],
    )
