from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import prpg.model.g3_boundary_views as boundary
from prpg.errors import ModelError
from prpg.model.g3_assembly import (
    ArtifactIntegrityDecisionEvidence,
    BlockCalibrationGateEvidence,
    BridgeDriverTaskViews,
    derive_g3_gate_decision,
)
from prpg.model.g3_task_publication import (
    artifact_integrity_executor_source_slots,
    block_calibration_executor_source_slots,
    scientific_task_source_slots,
)
from prpg.model.scientific_artifact import (
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
)


@dataclass(frozen=True)
class _PolicySelectionStub:
    selected_length: int


@dataclass(frozen=True)
class _ExecutionStub:
    policy_selection: _PolicySelectionStub


@dataclass(frozen=True)
class _BlockStub:
    frequency: str
    rng_execution_fingerprint: str
    master_seed: int
    execution: _ExecutionStub


@dataclass(frozen=True)
class _KernelStub:
    artifact: str
    frequency: str
    n_states: int
    selected_length: int
    model_artifact_fingerprint: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class _TransitionStub:
    n_states: int
    values: tuple[float, ...]
    role: str = "design"


def test_block_boundary_rehashes_registered_and_live_component_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = "a" * 64
    core = "5" * 64

    def identity(value: _BlockStub) -> SimpleNamespace:
        marker = "b" if value.frequency == "monthly" else "c"
        return SimpleNamespace(
            artifact="design",
            frequency=value.frequency,
            master_seed=value.master_seed,
            scientific_version=1,
            materialization_fingerprint=marker * 64,
            parent_model_fingerprint=parent,
            calibration_fingerprint=("d" if value.frequency == "monthly" else "e") * 64,
        )

    def kernel_identity(value: _KernelStub) -> SimpleNamespace:
        marker = "6" if value.frequency == "monthly" else "7"
        return SimpleNamespace(
            artifact=value.artifact,
            frequency=value.frequency,
            master_seed=7,
            scientific_version=1,
            n_states=value.n_states,
            selected_length=value.selected_length,
            parent_model_fingerprint=parent,
            source_materialization_fingerprint=marker * 64,
            rng_execution_fingerprint=("8" if value.frequency == "monthly" else "9")
            * 64,
            execution_fingerprint=("0" if value.frequency == "monthly" else "1") * 64,
        )

    def transition_identity(value: _TransitionStub) -> SimpleNamespace:
        return SimpleNamespace(
            role=value.role,
            n_states=value.n_states,
            master_seed=7,
            scientific_version=1,
            parent_model_fingerprint=parent,
            source_materialization_fingerprint="2" * 64,
            rng_execution_fingerprint="3" * 64,
            execution_fingerprint="4" * 64,
        )

    monkeypatch.setattr(boundary, "registered_block_calibration_identity", identity)
    monkeypatch.setattr(
        boundary,
        "registered_block_source_episode_counts",
        lambda _value: (30, 30),
    )
    monkeypatch.setattr(
        boundary,
        "kernel_calibration_execution_identity",
        kernel_identity,
    )
    monkeypatch.setattr(
        boundary,
        "transition_bootstrap_aggregate_identity",
        transition_identity,
    )
    monkeypatch.setattr(
        boundary,
        "registered_final_block_projection_identity",
        lambda _monthly, _daily: SimpleNamespace(
            master_seed=7,
            scientific_version=1,
            selected_n_states=2,
            fixed_point_fingerprint="5" * 64,
            iteration_history_fingerprint="6" * 64,
            final_selection_fingerprint="7" * 64,
            monthly_block_core_fingerprint="8" * 64,
            daily_block_core_fingerprint="9" * 64,
        ),
    )
    source = BlockCalibrationGateEvidence(
        role="design",
        selected_k=2,
        monthly=cast(
            Any,
            _BlockStub("monthly", "f" * 64, 7, _ExecutionStub(_PolicySelectionStub(3))),
        ),
        daily=cast(
            Any,
            _BlockStub("daily", "1" * 64, 7, _ExecutionStub(_PolicySelectionStub(10))),
        ),
        monthly_kernel=cast(
            Any,
            _KernelStub("design", "monthly", 2, 3, core, (0.1,)),
        ),
        daily_kernel=cast(
            Any,
            _KernelStub("design", "daily", 2, 10, core, (0.2,)),
        ),
        transition=cast(Any, _TransitionStub(2, (0.3,))),
    )
    view = boundary.build_block_calibration_task_view(
        source,
        task_id="design_block_calibration",
        run_id="2" * 64,
        binding_fingerprint="3" * 64,
    )

    assert boundary.is_block_calibration_task_view(view)
    assert view.parent_model_fingerprint == parent
    assert len(view.block_uniform_materialization_fingerprint) == 64
    assert len(view.block_calibration_rng_fingerprint) == 64
    assert view.monthly_kernel_parent_fingerprint == parent
    assert view.daily_kernel_parent_fingerprint == parent
    assert view.kernel_core_model_fingerprint == core
    assert view.transition_parent_model_fingerprint == parent
    assert len(view.kernel_calibration_rng_fingerprint) == 64
    slots = scientific_task_source_slots(view)
    assert (
        block_calibration_executor_source_slots(
            source,
            task_id="design_block_calibration",
        )
        == slots
    )
    alternate_view = boundary.build_block_calibration_task_view(
        source,
        task_id="design_block_calibration",
        run_id="2" * 64,
        binding_fingerprint="4" * 64,
    )
    assert scientific_task_source_slots(alternate_view) == slots
    materializations = dict(slots.materialization_fingerprints)
    rng = dict(slots.rng_fingerprints)
    assert len(materializations["task_source_materialization"]) == 64
    assert materializations["design_transition_bootstrap_materialization"] == (
        view.transition_materialization_fingerprint
    )
    assert (
        materializations["design_fixed_point_final_block_parent_materialization"]
        == view.design_fixed_point_final_block_parent_fingerprint
    )
    assert rng == {
        "design_block_calibration_streams": (view.block_calibration_rng_fingerprint),
        "design_kernel_calibration_streams": (view.kernel_calibration_rng_fingerprint),
        "design_transition_bootstrap_streams": view.transition_rng_fingerprint,
    }
    assert not boundary.is_block_calibration_task_view(copy.copy(view))
    with pytest.raises(ModelError, match="cross-role"):
        boundary.build_block_calibration_task_view(
            replace(source, role="production"),
            task_id="design_block_calibration",
            run_id="2" * 64,
            binding_fingerprint="4" * 64,
        )
    with pytest.raises(ModelError, match="RNG execution contract"):
        boundary.build_block_calibration_task_view(
            replace(
                source,
                daily=replace(cast(_BlockStub, source.daily), master_seed=8),
            ),
            task_id="design_block_calibration",
            run_id="2" * 64,
            binding_fingerprint="4" * 64,
        )
    with pytest.raises(ModelError, match="block/kernel/transition"):
        boundary.build_block_calibration_task_view(
            replace(
                source,
                monthly_kernel=replace(
                    cast(_KernelStub, source.monthly_kernel), n_states=3
                ),
            ),
            task_id="design_block_calibration",
            run_id="2" * 64,
            binding_fingerprint="4" * 64,
        )
    with pytest.raises(ModelError, match="block/kernel/transition"):
        boundary.build_block_calibration_task_view(
            replace(
                source,
                monthly_kernel=replace(
                    cast(_KernelStub, source.monthly_kernel),
                    model_artifact_fingerprint="f" * 64,
                ),
            ),
            task_id="design_block_calibration",
            run_id="2" * 64,
            binding_fingerprint="4" * 64,
        )
    with pytest.raises(ModelError, match="lineage"):
        boundary.build_block_calibration_task_view(
            replace(
                source,
                transition=replace(
                    cast(_TransitionStub, source.transition),
                    role="production",
                ),
            ),
            task_id="design_block_calibration",
            run_id="2" * 64,
            binding_fingerprint="4" * 64,
        )

    original_block_identity = boundary.registered_block_calibration_identity
    with monkeypatch.context() as patch:

        def split_parent(value: Any) -> Any:
            result = original_block_identity(value)
            if value is source.daily:
                result.parent_model_fingerprint = "f" * 64
            return result

        patch.setattr(
            boundary,
            "registered_block_calibration_identity",
            split_parent,
        )
        with pytest.raises(ModelError, match="exact parent model"):
            boundary.build_block_calibration_task_view(
                source,
                task_id="design_block_calibration",
                run_id="2" * 64,
                binding_fingerprint="4" * 64,
            )

    original_fixed_point_identity = boundary.registered_final_block_projection_identity
    with monkeypatch.context() as patch:

        def changed_fixed_point(monthly_value: Any, daily_value: Any) -> Any:
            result = original_fixed_point_identity(monthly_value, daily_value)
            result.selected_n_states = 3
            return result

        patch.setattr(
            boundary,
            "registered_final_block_projection_identity",
            changed_fixed_point,
        )
        with pytest.raises(ModelError, match="final fixed-point parents"):
            boundary.build_block_calibration_task_view(
                source,
                task_id="design_block_calibration",
                run_id="2" * 64,
                binding_fingerprint="4" * 64,
            )

    original_monthly_fingerprint = view.monthly_calibration_fingerprint
    object.__setattr__(view, "monthly_calibration_fingerprint", "0" * 64)
    with pytest.raises(ModelError, match="source identity changed"):
        boundary._verify_block_view(view)
    object.__setattr__(
        view,
        "monthly_calibration_fingerprint",
        original_monthly_fingerprint,
    )

    object.__setattr__(view, "run_id", "5" * 64)
    assert not boundary.is_block_calibration_task_view(view)


@dataclass(frozen=True)
class _Reference:
    fingerprint: str


@dataclass(frozen=True)
class _Geometry:
    marker: str

    def as_dict(self) -> dict[str, str]:
        return {"marker": self.marker}


@dataclass(frozen=True)
class _Provenance:
    source_code_fingerprint: str
    dependency_lock_fingerprint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "producer": "test",
            "source_code_fingerprint": self.source_code_fingerprint,
            "dependency_lock_fingerprint": self.dependency_lock_fingerprint,
        }


@dataclass(frozen=True)
class _Policy:
    policy: str


@dataclass(frozen=True)
class _Artifact:
    reference: _Reference
    role: str
    selected_k: int
    geometry: _Geometry
    binding: ScientificArtifactBinding
    provenance: _Provenance
    generation_policy: _Policy
    arrays: dict[str, np.ndarray[Any, Any]]
    reports: dict[str, dict[str, object]]
    core_model_fingerprint: str
    parameter_set_fingerprint: str
    generation_authorized: bool = False


def test_artifact_boundary_binds_exact_bridge_parent_and_live_artifact_content(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    run_id = passing_bridge_views.actual_artifact_bridge.run_id
    binding = ScientificArtifactBinding(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        calibration_run_fingerprint=run_id,
        preflight_fingerprint="d" * 64,
    )
    provenance = _Provenance("e" * 64, "f" * 64)
    common = {
        "selected_k": 2,
        "geometry": _Geometry("canonical"),
        "binding": binding,
        "provenance": provenance,
        "generation_policy": _Policy("evidence-only"),
        "reports": {"report.json": {"result_fingerprint": "1" * 64}},
        "core_model_fingerprint": "2" * 64,
    }
    design = _Artifact(
        reference=_Reference("3" * 64),
        role="design",
        arrays={"weights": np.asarray([0.25, 0.75])},
        parameter_set_fingerprint="4" * 64,
        **common,
    )
    production = _Artifact(
        reference=_Reference("5" * 64),
        role="production",
        arrays={"weights": np.asarray([0.20, 0.80])},
        parameter_set_fingerprint="6" * 64,
        **common,
    )
    monkeypatch.setattr(
        boundary,
        "is_verified_scientific_model_artifact",
        lambda value: isinstance(value, _Artifact),
    )
    pair = SimpleNamespace(
        design=cast(LoadedScientificModelArtifact, design),
        production=cast(LoadedScientificModelArtifact, production),
        run_id=run_id,
        reference=_Reference("9" * 64),
        actual_bridge_parent_view_fingerprint=(
            passing_bridge_views.actual_artifact_bridge.view_fingerprint
        ),
        design_role_source_fingerprint="a" * 64,
        production_role_source_fingerprint="b" * 64,
        generation_authorized=False,
    )
    monkeypatch.setattr(
        boundary,
        "is_verified_scientific_artifact_pair",
        lambda value: value is pair,
    )
    monkeypatch.setattr(
        boundary,
        "scientific_artifact_pair_identity",
        lambda value: {
            "actual_bridge_parent_view_fingerprint": (
                value.actual_bridge_parent_view_fingerprint
            )
        },
    )
    source = ArtifactIntegrityDecisionEvidence(cast(Any, pair))
    view = boundary.build_artifact_integrity_task_view(
        source,
        passing_bridge_views.actual_artifact_bridge,
        run_id=run_id,
        binding_fingerprint="7" * 64,
    )

    assert boundary.is_artifact_integrity_task_view(view)
    assert derive_g3_gate_decision("artifact_integrity", view).passed
    assert (
        view.actual_bridge_parent_view_fingerprint
        == passing_bridge_views.actual_artifact_bridge.view_fingerprint
    )
    assert len(view.final_artifact_parent_fingerprint) == 64
    slots = scientific_task_source_slots(view)
    assert (
        artifact_integrity_executor_source_slots(
            source,
            passing_bridge_views.actual_artifact_bridge,
            run_id=run_id,
        )
        == slots
    )
    alternate_view = boundary.build_artifact_integrity_task_view(
        source,
        passing_bridge_views.actual_artifact_bridge,
        run_id=run_id,
        binding_fingerprint="8" * 64,
    )
    assert scientific_task_source_slots(alternate_view) == slots
    assert (
        dict(slots.materialization_fingerprints)[
            "final_scientific_artifact_parent_materialization"
        ]
        == view.final_artifact_parent_fingerprint
    )
    assert (
        len(dict(slots.materialization_fingerprints)["task_source_materialization"])
        == 64
    )
    assert not boundary.is_artifact_integrity_task_view(copy.copy(view))
    with pytest.raises(ModelError, match="actual-bridge parent"):
        boundary.build_artifact_integrity_task_view(
            source,
            passing_bridge_views.bridge_tier_a_model_dgp,
            run_id=run_id,
            binding_fingerprint="8" * 64,
        )

    unverified_pair = SimpleNamespace()
    with pytest.raises(ModelError, match="source lineage is inconsistent"):
        boundary._artifact_source_identities(
            ArtifactIntegrityDecisionEvidence(cast(Any, unverified_pair)),
            run_id=run_id,
            actual_bridge_parent_view_fingerprint=(
                passing_bridge_views.actual_artifact_bridge.view_fingerprint
            ),
        )

    foreign_pair = SimpleNamespace(**vars(pair))
    foreign_pair.run_id = "f" * 64
    monkeypatch.setattr(
        boundary,
        "is_verified_scientific_artifact_pair",
        lambda value: value is pair or value is foreign_pair,
    )
    with pytest.raises(ModelError, match="pair/bridge lineage is inconsistent"):
        boundary._artifact_source_identities(
            ArtifactIntegrityDecisionEvidence(cast(Any, foreign_pair)),
            run_id=run_id,
            actual_bridge_parent_view_fingerprint=(
                passing_bridge_views.actual_artifact_bridge.view_fingerprint
            ),
        )

    original_parent = view._actual_bridge_parent
    object.__setattr__(
        view,
        "_actual_bridge_parent",
        passing_bridge_views.bridge_tier_a_model_dgp,
    )
    with pytest.raises(ModelError, match="parent view changed"):
        boundary._verify_artifact_view(view)
    object.__setattr__(view, "_actual_bridge_parent", original_parent)

    original_view_fingerprint = view.view_fingerprint
    object.__setattr__(view, "view_fingerprint", "0" * 64)
    with pytest.raises(ModelError, match="view fingerprint changed"):
        boundary._verify_artifact_view(view)
    object.__setattr__(view, "view_fingerprint", original_view_fingerprint)

    design.arrays["weights"][0] += 0.01
    assert not boundary.is_artifact_integrity_task_view(view)


def test_boundary_view_constructors_and_simple_guards_fail_closed() -> None:
    for view_type in (
        boundary.InputIntegrityTaskView,
        boundary.BlockCalibrationTaskView,
        boundary.ArtifactIntegrityTaskView,
        boundary.BridgeDriverTaskView,
    ):
        with pytest.raises(TypeError, match="created only by its builder"):
            view_type()

    bridge = object.__new__(boundary.BridgeDriverTaskView)
    assert bridge.generation_authorized is False
    with pytest.raises(ModelError, match="task ID is not registered"):
        boundary.build_bridge_driver_task_view(
            cast(Any, object()),
            task_id=cast(Any, "not-a-bridge-task"),
            run_id="1" * 64,
            binding_fingerprint="2" * 64,
        )
    with pytest.raises(ModelError, match="input-integrity source has the wrong type"):
        boundary._verify_input_source(object(), run_id="1" * 64)
    with pytest.raises(ModelError, match="block-calibration source has the wrong type"):
        boundary._block_source_identities(
            object(),
            task_id="design_block_calibration",
        )
    malformed_block = BlockCalibrationGateEvidence(
        role="design",
        selected_k=2,
        monthly=cast(Any, object()),
        daily=cast(Any, object()),
        monthly_kernel=cast(Any, object()),
        daily_kernel=cast(Any, object()),
        transition=cast(Any, object()),
    )
    with pytest.raises(ModelError, match="task ID is not registered"):
        boundary._block_source_identities(malformed_block, task_id="not-a-task")
    with pytest.raises(
        ModelError, match="artifact-integrity source has the wrong type"
    ):
        boundary._artifact_source_identities(
            object(),
            run_id="1" * 64,
            actual_bridge_parent_view_fingerprint="2" * 64,
        )
    with pytest.raises(ModelError, match="requires staged task evidence"):
        boundary._staged_bridge_source_identities(
            object(),
            task_id="bridge_resource_preflight",
        )
    with pytest.raises(ModelError, match="requires reverified driver evidence"):
        boundary._legacy_bridge_source_identities(
            object(),
            task_id="bridge_resource_preflight",
        )


def test_bridge_and_identity_helper_edge_cases_are_explicit() -> None:
    missing_dgp = SimpleNamespace(dgp_results=())
    with pytest.raises(ModelError, match="lacks its registered DGP"):
        boundary._bridge_tier_a_identity(cast(Any, missing_dgp), "model")
    with pytest.raises(ModelError, match="lacks its registered DGP"):
        boundary._bridge_tier_b_identity(cast(Any, missing_dgp), "model")

    staged_view = object.__new__(boundary.BridgeDriverTaskView)
    for name, value in (
        ("contract_version", boundary.BOUNDARY_VIEW_CONTRACT),
        ("task_id", "bridge_resource_preflight"),
        ("canonical", True),
        ("source", object()),
        ("_seal", boundary._BRIDGE_CAPABILITY),
    ):
        object.__setattr__(staged_view, name, value)
    boundary._MINTED_BRIDGE[id(staged_view)] = staged_view
    try:
        with pytest.raises(ModelError, match="lost its staged evidence"):
            boundary._verify_bridge_view(staged_view)
    finally:
        boundary._MINTED_BRIDGE.pop(id(staged_view), None)

    class Marker(Enum):
        VALUE = "marker"

    array = np.asarray([[1.0, 2.0]], dtype=np.float64)
    normalized = boundary._normalize(array)
    assert normalized["shape"] == [1, 2]
    assert boundary._normalize(np.int64(3)) == 3
    assert boundary._normalize(Marker.VALUE) == "marker"
    assert boundary._normalize(_Policy("evidence-only"))["fields"] == {
        "policy": "evidence-only"
    }
    with pytest.raises(ModelError, match="string keys"):
        boundary._normalize({1: "value"})
    with pytest.raises(ModelError, match="non-finite"):
        boundary._normalize(float("inf"))
    with pytest.raises(ModelError, match="unsupported value"):
        boundary._normalize(object())
    with pytest.raises(ModelError, match="lowercase SHA-256"):
        boundary._require_sha("not-a-digest", "identity")


def test_input_source_and_view_identity_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "1" * 64
    launch = SimpleNamespace(fingerprint="2" * 64)
    live = SimpleNamespace(fingerprint="3" * 64)
    binding = ScientificArtifactBinding(
        config_fingerprint="4" * 64,
        processed_data_fingerprint="5" * 64,
        calibration_input_fingerprint="6" * 64,
        calibration_run_fingerprint=run_id,
        preflight_fingerprint=launch.fingerprint,
    )
    monkeypatch.setattr(
        boundary,
        "verify_compatible_canonical_g3_preflights",
        lambda _launch, _live: launch.fingerprint,
    )
    source = boundary.InputIntegrityDecisionEvidence(
        cast(Any, launch),
        cast(Any, live),
        binding,
    )
    view = boundary.build_input_integrity_task_view(
        source,
        run_id=run_id,
        binding_fingerprint="7" * 64,
    )
    assert boundary.is_input_integrity_task_view(view)

    wrong_binding = replace(binding, calibration_run_fingerprint="8" * 64)
    with pytest.raises(ModelError, match="run/preflight binding is inconsistent"):
        boundary._verify_input_source(
            replace(source, binding=wrong_binding),
            run_id=run_id,
        )

    original_source_fingerprint = view.source_fingerprint
    object.__setattr__(view, "source_fingerprint", "9" * 64)
    with pytest.raises(ModelError, match="view identity changed"):
        boundary._verify_input_view(view)
    object.__setattr__(view, "source_fingerprint", original_source_fingerprint)


def test_staged_bridge_identity_variants_reject_lineage_and_view_tamper(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    original_identity = boundary.staged_bridge_task_evidence_identity
    original_looks = boundary.staged_bridge_task_planned_looks
    tier_a = passing_bridge_views.bridge_tier_a_model_dgp.source
    resource = passing_bridge_views.bridge_resource_preflight.source
    tier_b = passing_bridge_views.bridge_tier_b_model_dgp.source
    actual = passing_bridge_views.actual_artifact_bridge.source

    with monkeypatch.context() as patch:
        patch.setattr(
            boundary,
            "staged_bridge_task_evidence_identity",
            lambda value: replace(
                original_identity(value),
                parent_task_fingerprints=(),
            ),
        )
        with pytest.raises(ModelError, match="wrong parent lineage"):
            boundary._staged_bridge_source_identities(
                tier_a,
                task_id="bridge_tier_a_model_dgp",
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            boundary,
            "staged_bridge_task_evidence_identity",
            lambda value: replace(original_identity(value), selected_n_states=6),
        )
        with pytest.raises(ModelError, match="selected K is outside"):
            boundary._staged_bridge_source_identities(
                resource,
                task_id="bridge_resource_preflight",
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            boundary,
            "staged_bridge_task_evidence_identity",
            lambda value: replace(
                original_identity(value),
                rng_execution_fingerprint="a" * 64,
            ),
        )
        with pytest.raises(ModelError, match="empty RNG identity"):
            boundary._staged_bridge_source_identities(
                actual,
                task_id="actual_artifact_bridge",
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            boundary,
            "staged_bridge_task_planned_looks",
            lambda _value: (),
        )
        with pytest.raises(ModelError, match="exact planned-look prefix"):
            boundary._staged_bridge_source_identities(
                tier_b,
                task_id="bridge_tier_b_model_dgp",
            )

    tier_b_looks = original_looks(tier_b)
    with monkeypatch.context() as patch:
        patch.setattr(
            boundary,
            "staged_bridge_task_planned_looks",
            lambda value: tier_b_looks if value is resource else original_looks(value),
        )
        with pytest.raises(ModelError, match="acquired planned looks"):
            boundary._staged_bridge_source_identities(
                resource,
                task_id="bridge_resource_preflight",
            )

    view = passing_bridge_views.bridge_resource_preflight
    original_view_fingerprint = view.view_fingerprint
    object.__setattr__(view, "view_fingerprint", "0" * 64)
    with pytest.raises(ModelError, match="view identity changed"):
        boundary._verify_bridge_view(view)
    object.__setattr__(view, "view_fingerprint", original_view_fingerprint)


def test_legacy_bridge_reverification_and_source_identity_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(
        *,
        request_fingerprint: str,
        binding_fingerprint: str = "b" * 64,
        dgp_results: tuple[object, ...] = (),
        purpose1: tuple[tuple[str, str], ...] = (),
    ) -> boundary.ReverifiedBridgeDriverEvidence:
        binding = SimpleNamespace(
            fingerprint="a" * 64,
            base_preflight_fingerprint=binding_fingerprint,
            purpose1_trace_fingerprints=purpose1,
            selected_n_states=2,
        )
        return boundary.ReverifiedBridgeDriverEvidence(
            driver_result=cast(Any, SimpleNamespace(binding=binding)),
            request=cast(
                Any,
                SimpleNamespace(
                    base_preflight=SimpleNamespace(fingerprint=request_fingerprint)
                ),
            ),
            benchmark=cast(Any, SimpleNamespace(receipt_fingerprint="c" * 64)),
            dgp_results=cast(Any, dgp_results),
            tier_b_looks=cast(Any, ((), ())),
            final_receipt_fingerprint="d" * 64,
        )

    original = source(request_fingerprint="b" * 64)
    live = source(request_fingerprint="b" * 64)
    with monkeypatch.context() as patch:
        patch.setattr(
            boundary,
            "reverify_bridge_driver_result",
            lambda _driver_result, _request: live,
        )
        patch.setattr(
            boundary,
            "_bridge_task_identities",
            lambda value, *, task_id: {
                "task_id": task_id,
                "origin": "original" if value is original else "live",
            },
        )
        with pytest.raises(ModelError, match="changed after reverification"):
            boundary._legacy_bridge_source_identities(
                original,
                task_id="bridge_resource_preflight",
            )

    mismatched_request = source(request_fingerprint="e" * 64)
    with pytest.raises(ModelError, match="preregistered binding"):
        boundary._bridge_task_identities(
            mismatched_request,
            task_id="bridge_resource_preflight",
        )

    missing_dgp = source(request_fingerprint="b" * 64)
    with pytest.raises(ModelError, match="lacks its registered DGP"):
        boundary._bridge_task_identities(
            missing_dgp,
            task_id="bridge_tier_a_model_dgp",
        )

    model_without_tier_a = SimpleNamespace(dgp="model", tier_a=None)
    missing_trace = source(
        request_fingerprint="b" * 64,
        dgp_results=(model_without_tier_a,),
    )
    with pytest.raises(ModelError, match="lacks its purpose-1 trace"):
        boundary._bridge_task_identities(
            missing_trace,
            task_id="bridge_tier_a_model_dgp",
        )
