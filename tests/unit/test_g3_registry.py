from __future__ import annotations

import pytest

from prpg.errors import ModelError
from prpg.model.g3 import G3_GATE_ORDER
from prpg.model.g3_registry import (
    G3_PLANNED_LOOK_BY_ID,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_BY_ID,
    G3_TASK_REGISTRY,
    G3_TASK_REGISTRY_FINGERPRINT,
    failure_propagation,
    planned_look_spec,
    task_spec,
)


def test_exact_26_record_registry_is_topological_and_content_addressed() -> None:
    assert len(G3_TASK_REGISTRY) == 26
    assert tuple(item.task_id for item in G3_TASK_REGISTRY) == G3_GATE_ORDER
    positions = {task_id: index for index, task_id in enumerate(G3_GATE_ORDER)}
    assert all(
        positions[dependency] < positions[item.task_id]
        for item in G3_TASK_REGISTRY
        for dependency in item.dependencies
    )
    assert (
        G3_TASK_REGISTRY_FINGERPRINT
        == "f94a6521132ed7e233cd2bfb7e6daf20ad4e7f28e7c2f2473120d756963e63dd"
    )
    assert (
        G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
        == "8261570180a69ff3ede227c564dfc0410ca5688a60935921a064915ef6071e46"
    )


def test_registry_mappings_are_immutable_and_look_schedules_are_exact() -> None:
    with pytest.raises(TypeError):
        G3_TASK_BY_ID["new_task"] = task_spec("input_integrity")  # type: ignore[index]
    with pytest.raises(TypeError):
        G3_PLANNED_LOOK_BY_ID["new_look"] = planned_look_spec(  # type: ignore[index]
            "kernel_design_monthly"
        )
    assert planned_look_spec("kernel_design_daily").sample_counts == (
        10_000,
        20_000,
        40_000,
        80_000,
        160_000,
        320_000,
    )
    assert planned_look_spec("bridge_tier_b_model_dgp").sample_counts == tuple(
        range(1_000, 10_001, 1_000)
    )


def test_both_tier_b_tasks_require_both_tier_a_reference_laws() -> None:
    expected = (
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
    )
    assert task_spec("bridge_tier_b_model_dgp").dependencies == expected
    assert task_spec("bridge_tier_b_nonparametric_dgp").dependencies == expected
    blocked = failure_propagation(("bridge_tier_a_model_dgp",))
    assert "bridge_tier_b_model_dgp" in blocked
    assert "bridge_tier_b_nonparametric_dgp" in blocked


def test_failure_propagation_closes_every_downstream_branch() -> None:
    blocked = failure_propagation(("sealed_holdout_veto",))
    assert blocked[0] == "production_fit_same_k"
    assert "four_cell_adequacy_holm" in blocked
    assert "four_cell_dependence_holm" in blocked
    assert blocked[-2:] == ("actual_artifact_bridge", "artifact_integrity")
    assert "design_macro_stability" not in blocked

    adequacy_only = failure_propagation(("design_latent_correctness",))
    assert adequacy_only[0] == "four_cell_adequacy_holm"
    assert "four_cell_dependence_holm" not in adequacy_only


@pytest.mark.parametrize(
    "operation",
    [
        lambda: task_spec("unknown"),
        lambda: planned_look_spec("unknown"),
        lambda: failure_propagation(()),
        lambda: failure_propagation(("input_integrity", "input_integrity")),
    ],
)
def test_registry_queries_fail_closed(operation: object) -> None:
    with pytest.raises(ModelError):
        operation()  # type: ignore[operator]
