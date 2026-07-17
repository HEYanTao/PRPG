from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import prpg.model.g3_assembly as assembly_module
import prpg.model.g3_preflight as preflight_module
from prpg.errors import ModelError
from prpg.model.adequacy_gate import (
    ADEQUACY_CELL_ORDER,
    AdequacyCellRecord,
    FourCellAdequacyGate,
)
from prpg.model.benchmarks import HoldoutBenchmarkScores, holdout_veto_decision
from prpg.model.bridge import (
    actual_bridge_gate,
    bridge_resource_preflight,
    fixed_k_bridge_statistic,
    tier_a_selection_gate,
    tier_b_tail_gate,
)
from prpg.model.bridge_driver import StagedBridgeTaskEvidence
from prpg.model.calibration import SealedHoldoutEvidence
from prpg.model.dependence_gate_execution import (
    DEPENDENCE_CELL_ORDER,
    DependenceLengthEvidence,
    DependenceTaskCellEvidence,
    FragmentationStateEvidence,
    FragmentationTaskCellEvidence,
)
from prpg.model.g3 import (
    G3_GATE_ORDER,
    build_g3_evidence_bundle,
    gate_record,
    verify_g3_evidence_bundle,
)
from prpg.model.g3_assembly import (
    ArtifactIntegrityDecisionEvidence,
    BridgeDriverTaskView,
    BridgeDriverTaskViews,
    DerivedG3GateDecision,
    G3TaskFailure,
    InputIntegrityDecisionEvidence,
    assemble_typed_g3_evidence_bundle,
    derive_g3_gate_decision,
    task_view_cross_binding_error,
)
from prpg.model.g3_boundary_views import (
    build_bridge_driver_task_view,
    build_input_integrity_task_view,
    is_bridge_driver_task_view,
    is_input_integrity_task_view,
)
from prpg.model.g3_preflight import (
    CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
    CANONICAL_CONFIG_FINGERPRINT,
    CANONICAL_PROCESSED_DATA_FINGERPRINT,
    CanonicalG3Preflight,
    canonical_g3_resource_estimate,
)
from prpg.model.g3_registry import (
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.inference import holm_correction
from prpg.model.scientific_artifact import ScientificArtifactBinding


def _binding(preflight_fingerprint: str = "e" * 64) -> ScientificArtifactBinding:
    return ScientificArtifactBinding(
        config_fingerprint=CANONICAL_CONFIG_FINGERPRINT,
        processed_data_fingerprint=CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_input_fingerprint=CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
        calibration_run_fingerprint="d" * 64,
        preflight_fingerprint=preflight_fingerprint,
    )


def _preflight() -> CanonicalG3Preflight:
    estimate = canonical_g3_resource_estimate()
    source = preflight_module.inspect_g3_execution_closure()
    toolchain_identity = {
        "python_version": preflight_module.CANONICAL_PYTHON_VERSION,
        "python_implementation": "CPython",
        "platform_system": preflight_module.CANONICAL_PLATFORM,
        "machine": preflight_module.CANONICAL_MACHINE,
        "dependency_versions": dict(preflight_module.EXPECTED_RUNTIME_VERSIONS),
    }
    provisional = CanonicalG3Preflight(
        config_fingerprint=CANONICAL_CONFIG_FINGERPRINT,
        processed_data_fingerprint=CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_input_fingerprint=CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
        contract_fingerprint=preflight_module._sha256(
            preflight_module._canonical_bytes(preflight_module._contract_identity())
        ),
        source_code_fingerprint=source.source_code_fingerprint,
        dependency_lock_fingerprint=source.dependency_lock_fingerprint,
        toolchain_fingerprint=preflight_module._toolchain_fingerprint(
            preflight_module._sha256(
                preflight_module._canonical_bytes(toolchain_identity)
            ),
            source,
        ),
        ppw_vendored_fingerprint=preflight_module.REFERENCE_SHA256,
        ppw_upstream_fingerprint=preflight_module.UPSTREAM_SHA256,
        task_registry_fingerprint=G3_TASK_REGISTRY_FINGERPRINT,
        planned_look_registry_fingerprint=G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
        logical_cpu_count=10,
        physical_memory_bytes=17_179_869_184,
        disk_free_bytes=80_000_000_000,
        workers=9,
        estimated_peak_bytes=estimate.estimated_peak_bytes,
        estimated_disk_bytes=estimate.estimated_disk_bytes,
        resource_estimate_fingerprint=estimate.fingerprint,
        checks=preflight_module._CANONICAL_PREFLIGHT_CHECKS,
        fingerprint="",
    )
    fingerprint = hashlib.sha256(
        (
            json.dumps(
                provisional.identity_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    result = replace(provisional, fingerprint=fingerprint)
    object.__setattr__(
        result,
        "_authority_seal",
        preflight_module._G3_PREFLIGHT_CAPABILITY,
    )
    return result


def _preflight_with_disk(
    value: CanonicalG3Preflight, disk_free_bytes: int
) -> CanonicalG3Preflight:
    provisional = replace(value, disk_free_bytes=disk_free_bytes, fingerprint="")
    fingerprint = hashlib.sha256(
        (
            json.dumps(
                provisional.identity_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    result = replace(provisional, fingerprint=fingerprint)
    object.__setattr__(
        result,
        "_authority_seal",
        preflight_module._G3_PREFLIGHT_CAPABILITY,
    )
    return result


def _input_view(
    launch: CanonicalG3Preflight,
    live: CanonicalG3Preflight,
    binding: ScientificArtifactBinding,
):
    return build_input_integrity_task_view(
        InputIntegrityDecisionEvidence(launch, live, binding),
        run_id=binding.calibration_run_fingerprint,
        binding_fingerprint="1" * 64,
    )


def _adequacy_gate(
    *, p_values: tuple[float, float, float, float]
) -> FourCellAdequacyGate:
    cells = (
        AdequacyCellRecord("design_latent", True, p_values[0], 10_000, 10_000, None),
        AdequacyCellRecord(
            "production_latent", True, p_values[1], 10_000, 10_000, None
        ),
        AdequacyCellRecord(
            "design_refitted_adequacy", True, p_values[2], 1_000, 950, None
        ),
        AdequacyCellRecord(
            "production_refitted_adequacy", True, p_values[3], 1_000, 950, None
        ),
    )
    holm = holm_correction(np.asarray(p_values), alpha=0.05)
    passed = not bool(holm.rejected.any())
    reasons = tuple(
        f"{cell.cell_id}:holm_rejected"
        for cell, rejected in zip(cells, holm.rejected, strict=True)
        if bool(rejected)
    )
    return FourCellAdequacyGate(
        ADEQUACY_CELL_ORDER,
        cells,
        holm,
        True,
        passed,
        reasons,
        True,
    )


class _RegisteredBlockStub:
    def __init__(self, execution: SimpleNamespace) -> None:
        self.execution = execution
        self.history_stream_sets = assembly_module.CANONICAL_HISTORY_COUNT


class _KernelExecutionStub(SimpleNamespace):
    pass


class _TransitionAggregateStub(SimpleNamespace):
    pass


class _BlockViewStub(SimpleNamespace):
    pass


class _DependenceViewStub(SimpleNamespace):
    pass


class _DependenceHolmViewStub(SimpleNamespace):
    pass


class _BridgeViewStub(SimpleNamespace):
    pass


class _ArtifactIntegrityViewStub(SimpleNamespace):
    pass


class _InputIntegrityViewStub(SimpleNamespace):
    pass


class _CrossBlockViewStub(SimpleNamespace):
    pass


class _CandidateViabilityViewStub(SimpleNamespace):
    pass


class _CandidateSelectionViewStub(SimpleNamespace):
    pass


def test_candidate_selection_assembly_treats_fixed_k4_cross_k_metrics_as_report_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assembly_module,
        "is_candidate_selection_task_view",
        lambda _value: True,
    )
    rows = tuple(
        SimpleNamespace(
            n_states=n_states,
            selected=n_states == 4,
            viability=SimpleNamespace(admissible=True),
            predictively_eligible=True,
            predictively_near_best=False,
            inside_bic_guard=False,
            disposition="selected" if n_states == 4 else "report_only_nonfixed_k",
        )
        for n_states in (2, 3, 4, 5)
    )
    selection = SimpleNamespace(
        selection_policy="owner_fixed_k4_v3",
        candidates=rows,
        rolling_max_z=None,
        bootstrap_replicates=0,
        max_z_critical_value=0.0,
        rolling_fold_count=6,
        rolling_score_count=72,
        selected_n_states=4,
        passed=True,
        failure_reason=None,
    )
    view = _CandidateSelectionViewStub(
        task_id="design_predictive_selection",
        role="design",
        run_id="1" * 64,
        binding_fingerprint="2" * 64,
        view_fingerprint="3" * 64,
        source_content_fingerprint="4" * 64,
        viability_result_fingerprint="5" * 64,
        fit_set_fingerprint="6" * 64,
        rolling_fingerprints=("7" * 64,) * 4,
        bootstrap_fingerprint="8" * 64,
        viability_view_fingerprint="9" * 64,
        selection_fingerprint="a" * 64,
        selected_n_states=4,
        passed=True,
        selection=selection,
        _viability_parent=SimpleNamespace(scientific_version=11),
    )

    passed, evidence, summary = assembly_module._candidate(
        "design_predictive_selection",
        view,
    )

    assert passed
    assert evidence["selection_policy"] == "owner_fixed_k4_v3"
    assert evidence["scientific_version"] == 11
    assert summary == {"selected_k": 4}

    selection.selection_policy = "predictive_bic_v1"
    selection.passed = False
    view.passed = False
    with pytest.raises(
        ModelError,
        match="scientific version 11 forbids ordinary cross-K selection",
    ):
        assembly_module._candidate(
            "design_predictive_selection",
            view,
        )

    # The legacy evaluator remains available only for replay/report callers
    # whose guarded parent carries a pre-v11 science identity.
    view._viability_parent.scientific_version = 10
    ordinary_passed, _, _ = assembly_module._candidate(
        "design_predictive_selection",
        view,
    )
    assert ordinary_passed is False


class _HoldoutViewStub(SimpleNamespace):
    pass


class _ProductionFitViewStub(SimpleNamespace):
    pass


class _MacroViewStub(SimpleNamespace):
    pass


class _AdequacyViewStub(SimpleNamespace):
    pass


class _AdequacyHolmViewStub(SimpleNamespace):
    pass


class _CrossDependenceViewStub(SimpleNamespace):
    pass


class _CrossDependenceHolmViewStub(SimpleNamespace):
    pass


@dataclass(frozen=True)
class _DictionaryStub:
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label}


@dataclass(frozen=True)
class _ProvenanceStub:
    label: str
    source_code_fingerprint: str = "a" * 64
    dependency_lock_fingerprint: str = "b" * 64

    def as_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "source_code_fingerprint": self.source_code_fingerprint,
            "dependency_lock_fingerprint": self.dependency_lock_fingerprint,
        }


def _install_block_gate_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assembly_module,
        "is_block_calibration_task_view",
        lambda value: isinstance(value, _BlockViewStub),
    )
    monkeypatch.setattr(
        assembly_module,
        "RegisteredBlockCalibration",
        _RegisteredBlockStub,
    )
    monkeypatch.setattr(
        assembly_module,
        "KernelCalibrationExecution",
        _KernelExecutionStub,
    )
    monkeypatch.setattr(
        assembly_module,
        "TransitionBootstrapAggregate",
        _TransitionAggregateStub,
    )


def _block_gate_view(role: str) -> _BlockViewStub:
    monthly_segments = (193,) if role == "design" else (210, 7)
    daily_segments = (4_049,) if role == "design" else (4_404, 143)
    selected_k = 4
    lengths = {"monthly": 4, "daily": 20}

    def registered(frequency: str) -> _RegisteredBlockStub:
        selected_length = lengths[frequency]
        execution = SimpleNamespace(
            geometry=SimpleNamespace(
                artifact=role,
                frequency=frequency,
                strict_canonical=True,
                monthly_segment_lengths=monthly_segments,
                target_segment_lengths=(
                    monthly_segments if frequency == "monthly" else daily_segments
                ),
            ),
            histories=assembly_module.CANONICAL_HISTORY_COUNT,
            policy_selection=SimpleNamespace(
                frequency=frequency,
                selected_length=selected_length,
            ),
            selected_fragmentation_gates=tuple(
                SimpleNamespace(
                    state=state,
                    intended_length=selected_length,
                    reasons=(),
                )
                for state in range(selected_k)
            ),
        )
        return _RegisteredBlockStub(execution)

    def kernel(frequency: str) -> _KernelExecutionStub:
        selected_length = lengths[frequency]
        stopped_windows = assembly_module.CANONICAL_PREFIX_WINDOW_COUNTS[0]
        return _KernelExecutionStub(
            artifact=role,
            frequency=frequency,
            n_states=selected_k,
            selected_length=selected_length,
            strict_canonical=True,
            planned_looks=assembly_module.CANONICAL_PREFIX_WINDOW_COUNTS,
            completed_looks=(SimpleNamespace(windows=stopped_windows),),
            stopped_windows=stopped_windows,
            precision_passed=True,
            verification_passed=True,
            offset_grid=np.zeros((1, selected_k, 3)),
            verification=SimpleNamespace(passed=True),
            final_precision=SimpleNamespace(
                annualized_simultaneous_half_widths=np.full(
                    (selected_k, 3),
                    assembly_module.CANONICAL_ANNUALIZED_HALF_WIDTH_MAX / 2,
                )
            ),
            passed=True,
            model_artifact_fingerprint="a" * 64,
        )

    transition = _TransitionAggregateStub(
        n_states=selected_k,
        attempts=assembly_module.MACRO_BOOTSTRAP_ATTEMPTS,
        successful_refits=assembly_module.MINIMUM_SUCCESSFUL_REFITS,
        failed_refits=(
            assembly_module.MACRO_BOOTSTRAP_ATTEMPTS
            - assembly_module.MINIMUM_SUCCESSFUL_REFITS
        ),
        edge_frequency_denominator=assembly_module.MACRO_BOOTSTRAP_ATTEMPTS,
        attempt_audits=tuple(range(assembly_module.MACRO_BOOTSTRAP_ATTEMPTS)),
        historical_transition_support=SimpleNamespace(passed=True),
    )
    source = SimpleNamespace(
        role=role,
        selected_k=selected_k,
        monthly=registered("monthly"),
        daily=registered("daily"),
        monthly_kernel=kernel("monthly"),
        daily_kernel=kernel("daily"),
        transition=transition,
    )
    task_id = f"{role}_block_calibration"
    return _BlockViewStub(
        task_id=task_id,
        run_id="b" * 64,
        binding_fingerprint="c" * 64,
        view_fingerprint="d" * 64,
        source_fingerprint="e" * 64,
        monthly_calibration_fingerprint="f" * 64,
        daily_calibration_fingerprint="1" * 64,
        monthly_kernel_fingerprint="2" * 64,
        daily_kernel_fingerprint="3" * 64,
        transition_fingerprint="4" * 64,
        monthly_source_episode_counts=(2, 2, 2, 3),
        daily_source_episode_counts=(2, 5, 7, 2),
        recurring_history_support_fingerprint=(
            "5" * 64 if role == "production" else None
        ),
        source=source,
    )


@pytest.mark.parametrize("role", ("design", "production"))
def test_block_gate_rederives_registered_geometry_kernel_and_transition(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    _install_block_gate_stubs(monkeypatch)
    view = _block_gate_view(role)

    passed, evidence, summary = assembly_module._block(view.task_id, view)

    assert passed
    assert evidence["role"] == role
    assert evidence["selected_k"] == 4
    assert evidence["block_lengths"] == {"monthly": 4, "daily": 20}
    assert evidence["block_ready"]
    assert evidence["fragmentation_passed"]
    assert evidence["kernel_passed"]
    assert evidence["transition_passed"]
    assert evidence["core_model_fingerprint"] == "a" * 64
    assert summary["core_model_fingerprint"] == "a" * 64


def test_block_gate_records_fragmentation_failure_without_forging_block_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_block_gate_stubs(monkeypatch)
    view = _block_gate_view("design")
    view.source.daily.execution.selected_fragmentation_gates = (
        SimpleNamespace(state=0, intended_length=20, reasons=("singleton_cap",)),
        SimpleNamespace(state=1, intended_length=20, reasons=()),
    )

    passed, evidence, _summary = assembly_module._block(view.task_id, view)

    assert passed
    assert not evidence["fragmentation_passed"]


def test_production_block_gate_requires_two_observed_runs_per_k4_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_block_gate_stubs(monkeypatch)
    view = _block_gate_view("production")
    view.monthly_source_episode_counts = (2, 1, 2, 3)
    view.recurring_history_support_fingerprint = None

    passed, evidence, _summary = assembly_module._block(view.task_id, view)

    assert not passed
    assert evidence["recurring_history_support_passed"] is False
    assert evidence["monthly_source_episode_counts"] == [2, 1, 2, 3]


@pytest.mark.parametrize(
    "tamper",
    (
        "geometry",
        "history-count",
        "selected-length",
        "kernel-geometry",
        "kernel-precision",
        "kernel-model-mismatch",
        "kernel-model-invalid",
        "transition",
    ),
)
def test_block_gate_fails_closed_when_registered_science_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _install_block_gate_stubs(monkeypatch)
    view = _block_gate_view("design")
    if tamper == "geometry":
        view.source.monthly.execution.geometry.target_segment_lengths = (192,)
    elif tamper == "history-count":
        view.source.daily.history_stream_sets -= 1
    elif tamper == "selected-length":
        view.source.monthly.execution.policy_selection.selected_length = 0
    elif tamper == "kernel-geometry":
        view.source.monthly_kernel.frequency = "daily"
    elif tamper == "kernel-precision":
        view.source.daily_kernel.final_precision.annualized_simultaneous_half_widths = (
            np.full(
                (2, 3),
                assembly_module.CANONICAL_ANNUALIZED_HALF_WIDTH_MAX * 2,
            )
        )
    elif tamper == "kernel-model-mismatch":
        view.source.daily_kernel.model_artifact_fingerprint = "b" * 64
    elif tamper == "kernel-model-invalid":
        view.source.daily_kernel.model_artifact_fingerprint = "invalid"
    elif tamper == "transition":
        view.source.transition.successful_refits -= 1
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(f"unsupported tamper: {tamper}")

    passed, evidence, _summary = assembly_module._block(view.task_id, view)

    assert not passed
    assert not (
        evidence["block_ready"]
        and evidence["kernel_passed"]
        and evidence["transition_passed"]
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("view", "sealed block-calibration task view"),
        ("role", "role/K binding"),
        ("registered", "registered execution results"),
        ("kernel", "registered kernel executions"),
        ("kernel-pass", "pass flag is inconsistent"),
        ("transition", "transition-bootstrap aggregate"),
    ),
)
def test_block_gate_rejects_substitution_and_inconsistent_pass_flags(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    _install_block_gate_stubs(monkeypatch)
    view = _block_gate_view("design")
    if tamper == "view":
        view.task_id = "production_block_calibration"
    elif tamper == "role":
        view.source.role = "production"
    elif tamper == "registered":
        view.source.monthly = object()
    elif tamper == "kernel":
        view.source.daily_kernel = object()
    elif tamper == "kernel-pass":
        view.source.monthly_kernel.passed = False
    elif tamper == "transition":
        view.source.transition = object()
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(f"unsupported tamper: {tamper}")

    with pytest.raises(ModelError, match=message):
        assembly_module._block("design_block_calibration", view)


def _fragmentation_cell(
    role: str,
    frequency: str,
    *,
    histories: int | None = None,
    common_draw_reused: bool = True,
    reasons: tuple[str, ...] = (),
) -> FragmentationTaskCellEvidence:
    selected_length = 4 if frequency == "monthly" else 20
    return FragmentationTaskCellEvidence(
        cell_id=f"{role}_{frequency}",  # type: ignore[arg-type]
        artifact=role,  # type: ignore[arg-type]
        frequency=frequency,  # type: ignore[arg-type]
        histories=(
            assembly_module.CANONICAL_HISTORY_COUNT if histories is None else histories
        ),
        selected_length=selected_length,
        common_draw_reused=common_draw_reused,
        fragmentation=(
            FragmentationStateEvidence(
                state=0,
                intended_length=selected_length,
                conditioned_count=100,
                conditioned_mean=float(selected_length),
                conditioned_median=float(selected_length),
                conditioned_singleton_fraction=0.0,
                ideal_count=100,
                ideal_median=float(selected_length),
                ideal_singleton_fraction=0.0,
                reasons=reasons,
            ),
        ),
        input_fingerprint="1" * 64,
        model_fingerprint="2" * 64,
        materialization_fingerprint="3" * 64,
        source_fingerprint="4" * 64,
        source_evidence_fingerprint="5" * 64,
        selected_k=2,
        starting_length=selected_length,
        parent_block_core_fingerprint="6" * 64,
        fragmentation_evidence_fingerprint="7" * 64,
    )


def _dependence_cell(
    role: str,
    frequency: str,
    *,
    histories: int | None = None,
    endpoint_count: int | None = None,
    common_draw_reused: bool = True,
    repetitions: int | None = None,
    decision_role: str = "selected",
    observed_statistic: float = 0.5,
    critical_value: float = 1.0,
    p_value: float = 0.5,
) -> DependenceTaskCellEvidence:
    selected_length = 4 if frequency == "monthly" else 20
    return DependenceTaskCellEvidence(
        cell_id=f"{role}_{frequency}",  # type: ignore[arg-type]
        artifact=role,  # type: ignore[arg-type]
        frequency=frequency,  # type: ignore[arg-type]
        histories=(
            assembly_module.CANONICAL_HISTORY_COUNT if histories is None else histories
        ),
        selected_length=selected_length,
        endpoint_count=(
            (87 if frequency == "monthly" else 31)
            if endpoint_count is None
            else endpoint_count
        ),
        common_draw_reused=common_draw_reused,
        selected_result=DependenceLengthEvidence(
            intended_length=selected_length,
            decision_role=decision_role,  # type: ignore[arg-type]
            observed_statistic=observed_statistic,
            critical_value_95=critical_value,
            p_value=p_value,
            active_components=3,
            repetitions=(
                assembly_module.CANONICAL_HISTORY_COUNT
                if repetitions is None
                else repetitions
            ),
        ),
        loyo_fingerprint="8" * 64,
        input_fingerprint="1" * 64,
        model_fingerprint="2" * 64,
        materialization_fingerprint="3" * 64,
        source_fingerprint="4" * 64,
        source_evidence_fingerprint="5" * 64,
        selected_k=2,
        starting_length=selected_length,
        parent_block_core_fingerprint="6" * 64,
        purpose4_materialization_fingerprint="7" * 64,
        purpose4_stream_fingerprint="8" * 64,
        rng_execution_fingerprint="9" * 64,
        dependence_result_fingerprint="a" * 64,
    )


def _dependence_view(
    role: str,
    kind: str,
    *,
    cells: tuple[object, object] | None = None,
    passed: bool = True,
) -> _DependenceViewStub:
    if cells is None:
        cells = (
            (
                _fragmentation_cell(role, "monthly")
                if kind == "fragmentation"
                else _dependence_cell(role, "monthly")
            ),
            (
                _fragmentation_cell(role, "daily")
                if kind == "fragmentation"
                else _dependence_cell(role, "daily")
            ),
        )
    task_id = f"{role}_{kind}"
    return _DependenceViewStub(
        task_id=task_id,
        strict_canonical=True,
        role=role,
        task_kind=kind,
        cells=cells,
        passed=passed,
        run_id="b" * 64,
        binding_fingerprint="c" * 64,
        source_execution_fingerprint="d" * 64,
        parent_block_core_fingerprints=("e" * 64, "f" * 64),
        fragmentation_parent_view_fingerprint=(
            None if kind == "fragmentation" else "1" * 64
        ),
        view_fingerprint="2" * 64,
    )


def _install_dependence_gate_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assembly_module,
        "is_registered_dependence_task_view",
        lambda value: isinstance(value, _DependenceViewStub),
    )
    monkeypatch.setattr(
        assembly_module,
        "is_registered_four_cell_dependence_holm_view",
        lambda value: isinstance(value, _DependenceHolmViewStub),
    )


@pytest.mark.parametrize(
    ("role", "kind"),
    (
        ("design", "fragmentation"),
        ("production", "fragmentation"),
        ("design", "dependence"),
        ("production", "dependence"),
    ),
)
def test_dependence_gate_rederives_each_task_from_canonical_cells(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    kind: str,
) -> None:
    _install_dependence_gate_stubs(monkeypatch)
    view = _dependence_view(role, kind)

    passed, evidence, summary = assembly_module._dependence(view.task_id, view)

    assert passed
    assert evidence["role"] == role
    assert evidence["task_kind"] == kind
    assert len(evidence["cells"]) == 2
    assert summary == {"role": role, "cell_count": 2}


@pytest.mark.parametrize(
    "tamper",
    (
        "histories",
        "common-draw",
        "fragmentation",
        "endpoint-count",
        "repetitions",
        "decision-role",
        "predictive-envelope",
    ),
)
def test_dependence_gate_records_scientific_nonpass_without_false_authority(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _install_dependence_gate_stubs(monkeypatch)
    if tamper in {"histories", "common-draw", "fragmentation"}:
        first = _fragmentation_cell(
            "design",
            "monthly",
            histories=(9_999 if tamper == "histories" else None),
            common_draw_reused=tamper != "common-draw",
            reasons=(("singleton_cap",) if tamper == "fragmentation" else ()),
        )
        view = _dependence_view(
            "design",
            "fragmentation",
            cells=(first, _fragmentation_cell("design", "daily")),
            passed=False,
        )
    else:
        first = _dependence_cell(
            "design",
            "monthly",
            endpoint_count=(86 if tamper == "endpoint-count" else None),
            repetitions=(9_999 if tamper == "repetitions" else None),
            decision_role=(
                "report_only_neighbor" if tamper == "decision-role" else "selected"
            ),
            observed_statistic=(2.0 if tamper == "predictive-envelope" else 0.5),
        )
        view = _dependence_view(
            "design",
            "dependence",
            cells=(first, _dependence_cell("design", "daily")),
            passed=False,
        )

    passed, evidence, _summary = assembly_module._dependence(view.task_id, view)

    assert not passed
    assert len(evidence["cells"]) == 2


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("role", "wrong role or kind"),
        ("kind", "wrong role or kind"),
        ("fragmentation-cell-type", "exposes dependence evidence"),
        ("dependence-cell-type", "exposes fragmentation evidence"),
        ("pass-flag", "decision is inconsistent"),
    ),
)
def test_dependence_gate_rejects_cross_role_or_future_evidence_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    _install_dependence_gate_stubs(monkeypatch)
    view = _dependence_view("design", "fragmentation")
    if tamper == "role":
        view.role = "production"
    elif tamper == "kind":
        view.task_kind = "dependence"
    elif tamper == "fragmentation-cell-type":
        view.cells = (
            _dependence_cell("design", "monthly"),
            _fragmentation_cell("design", "daily"),
        )
    elif tamper == "dependence-cell-type":
        view = _dependence_view(
            "design",
            "dependence",
            cells=(
                _fragmentation_cell("design", "monthly"),
                _dependence_cell("design", "daily"),
            ),
        )
    elif tamper == "pass-flag":
        view.passed = False
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(tamper)

    with pytest.raises(ModelError, match=message):
        assembly_module._dependence(view.task_id, view)


def _dependence_holm_view(
    *,
    p_values: tuple[float, float, float, float] = (0.5, 0.6, 0.7, 0.8),
    common_draw_reused: bool = True,
    passed: bool | None = None,
) -> _DependenceHolmViewStub:
    cells = tuple(
        _dependence_cell(
            "design" if index < 2 else "production",
            "monthly" if index % 2 == 0 else "daily",
            common_draw_reused=common_draw_reused,
            p_value=p_value,
        )
        for index, p_value in enumerate(p_values)
    )
    holm = holm_correction(np.asarray(p_values), alpha=0.05)
    expected_passed = not bool(holm.rejected.any()) and common_draw_reused
    return _DependenceHolmViewStub(
        task_id="four_cell_dependence_holm",
        cell_order=DEPENDENCE_CELL_ORDER,
        strict_canonical=True,
        cells=cells,
        holm=SimpleNamespace(holm=holm),
        passed=expected_passed if passed is None else passed,
        run_id="b" * 64,
        binding_fingerprint="c" * 64,
        view_fingerprint="d" * 64,
        source_views=(
            SimpleNamespace(binding_fingerprint="e" * 64),
            SimpleNamespace(binding_fingerprint="f" * 64),
        ),
        source_view_fingerprints=("1" * 64, "2" * 64),
        failure_reasons=(),
    )


def test_four_cell_dependence_holm_is_recomputed_and_can_scientifically_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dependence_gate_stubs(monkeypatch)
    passing = _dependence_holm_view()
    passed, evidence, summary = assembly_module._dependence(
        "four_cell_dependence_holm",
        passing,
    )
    assert passed
    assert evidence["cell_order"] == list(DEPENDENCE_CELL_ORDER)
    assert summary == {"holm_passed": True}

    failing = _dependence_holm_view(
        p_values=(0.001, 0.6, 0.7, 0.8),
    )
    passed, evidence, summary = assembly_module._dependence(
        "four_cell_dependence_holm",
        failing,
    )
    assert not passed
    assert any(evidence["rejected"])
    assert summary == {"holm_passed": False}

    no_common_draw = _dependence_holm_view(common_draw_reused=False)
    passed, _evidence, _summary = assembly_module._dependence(
        "four_cell_dependence_holm",
        no_common_draw,
    )
    assert not passed


def test_four_cell_dependence_holm_rejects_tampered_reduction_or_pass_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dependence_gate_stubs(monkeypatch)
    view = _dependence_holm_view()
    view.holm = SimpleNamespace(holm=holm_correction(np.full(4, 0.01), alpha=0.05))
    with pytest.raises(ModelError, match="Holm result is inconsistent"):
        assembly_module._dependence("four_cell_dependence_holm", view)

    view = _dependence_holm_view(passed=False)
    with pytest.raises(ModelError, match="aggregate pass flag is inconsistent"):
        assembly_module._dependence("four_cell_dependence_holm", view)


def test_public_boolean_records_cannot_authorize_even_when_all_pass() -> None:
    records = tuple(
        gate_record(
            gate_id,
            passed=True,
            evidence={"caller_claim": "pass"},
            summary={"status": "claimed"},
        )
        for gate_id in G3_GATE_ORDER
    )
    bundle = build_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        design_artifact_fingerprint="d" * 64,
        production_artifact_fingerprint="e" * 64,
        records=records,
    )

    assert bundle.passed
    assert not bundle.generation_authorized
    verify_g3_evidence_bundle(bundle)


def test_derived_decision_has_no_public_constructor_or_pass_argument() -> None:
    with pytest.raises(TypeError):
        DerivedG3GateDecision(  # type: ignore[call-arg]
            gate_id="input_integrity",
            passed=True,
            evidence={},
            summary={},
        )
    with pytest.raises(ModelError, match="requires its sealed task view"):
        derive_g3_gate_decision("input_integrity", True)


def test_input_integrity_is_derived_from_cross_fingerprints() -> None:
    preflight = _preflight()
    binding = _binding(preflight.fingerprint)
    good = derive_g3_gate_decision(
        "input_integrity",
        _input_view(preflight, preflight, binding),
    )
    input_view = _input_view(preflight, preflight, binding)
    assert is_input_integrity_task_view(input_view)
    assert not is_input_integrity_task_view(copy.copy(input_view))
    with pytest.raises(ModelError, match="sealed task view"):
        derive_g3_gate_decision(
            "input_integrity",
            InputIntegrityDecisionEvidence(preflight, preflight, binding),
        )
    different_run_binding = replace(binding, calibration_run_fingerprint="9" * 64)
    bad = derive_g3_gate_decision(
        "input_integrity",
        _input_view(preflight, preflight, different_run_binding),
    )

    assert good.passed
    # Calibration-run identity is not a preflight output and therefore does
    # not alter this gate; it is bound later by the typed artifact pair.
    assert bad.passed
    mismatched = derive_g3_gate_decision(
        "input_integrity",
        _input_view(
            preflight,
            preflight,
            replace(binding, config_fingerprint="9" * 64),
        ),
    )
    assert not mismatched.passed
    with pytest.raises(ModelError, match="builder authority"):
        derive_g3_gate_decision(
            "input_integrity",
            _input_view(preflight, replace(preflight), binding),
        )


def test_input_integrity_is_resume_stable_across_free_disk_observations() -> None:
    launch = _preflight()
    persisted_launch = replace(launch)
    live = _preflight_with_disk(launch, launch.disk_free_bytes + 1_000_000)
    binding = _binding(launch.fingerprint)

    initial = derive_g3_gate_decision(
        "input_integrity",
        _input_view(launch, launch, binding),
    )
    resumed = derive_g3_gate_decision(
        "input_integrity",
        _input_view(persisted_launch, live, binding),
    )

    assert live.fingerprint != launch.fingerprint
    assert initial.passed and resumed.passed
    assert resumed.evidence != initial.evidence
    volatile_identity_fields = {
        "live_preflight_fingerprint",
        "source_fingerprint",
        "view_fingerprint",
    }
    assert {
        key: value
        for key, value in resumed.evidence.items()
        if key not in volatile_identity_fields
    } == {
        key: value
        for key, value in initial.evidence.items()
        if key not in volatile_identity_fields
    }
    assert resumed.summary == initial.summary


def test_raw_holdout_aggregate_is_rejected_at_task_boundary() -> None:
    hmm = np.full(24, 2.0)
    benchmarks = HoldoutBenchmarkScores(np.ones(24), np.full(24, 1.5))
    evidence = SealedHoldoutEvidence(
        (17, 7), hmm, benchmarks, holdout_veto_decision(hmm, benchmarks)
    )
    with pytest.raises(ModelError, match="sealed executed task view"):
        derive_g3_gate_decision("sealed_holdout_veto", evidence)


def test_raw_four_cell_adequacy_aggregate_is_rejected_at_task_boundary() -> None:
    passing = _adequacy_gate(p_values=(0.5, 0.6, 0.7, 0.8))
    for task in (
        "design_latent_correctness",
        "production_latent_correctness",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
        "four_cell_adequacy_holm",
    ):
        with pytest.raises(ModelError, match="sealed|Holm view"):
            derive_g3_gate_decision(task, passing)


def test_bridge_decisions_require_one_sealed_checkpoint_rederived_driver(
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    preflight = bridge_resource_preflight(
        benchmark_pair_seconds=np.ones(100),
        projected_pair_fits=10_000,
        workers=9,
        peak_memory_fraction=0.5,
    )
    tier_a = tier_a_selection_gate(np.ones(250, dtype=np.bool_))
    tier_b = tier_b_tail_gate(0.0, np.ones(1_000))
    fixed = fixed_k_bridge_statistic(
        centroid_shifts=[0.1],
        transition_cell_shifts=[0.01],
        occupancy_shifts=[0.01],
        relative_median_dwell_shifts=[0.01],
        dwell_survival_shifts=[0.01],
    )
    actual = actual_bridge_gate(
        design_n_states=2,
        production_n_states=2,
        same_aligned_order=True,
        fixed_k=fixed,
        design_monthly_block_length=3,
        production_monthly_block_length=3,
        design_daily_block_length=10,
        production_daily_block_length=10,
        design_monthly_state_counts=[100, 100],
        production_monthly_state_counts=[100, 100],
        design_daily_state_counts=[1_000, 1_000],
        production_daily_state_counts=[1_000, 1_000],
        monthly_kernel_offset_differences=np.zeros((4, 2, 3)),
        daily_kernel_offset_differences=np.zeros((4, 2, 3)),
    )
    for task_id, bare in (
        ("bridge_resource_preflight", preflight),
        ("bridge_tier_a_model_dgp", tier_a),
        ("bridge_tier_b_nonparametric_dgp", tier_b),
        ("actual_artifact_bridge", actual),
    ):
        with pytest.raises(ModelError, match="sealed bridge-driver task view"):
            derive_g3_gate_decision(task_id, bare)

    for task_id, view in passing_bridge_views.as_mapping().items():
        decision = derive_g3_gate_decision(task_id, view)
        staged_source = cast(StagedBridgeTaskEvidence, view.source)
        assert decision.passed
        assert decision.evidence["generation_authorized"] is False
        assert (
            decision.evidence["bridge_stage_receipt_fingerprint"]
            == staged_source.stage_receipt_fingerprint
        )
        assert (
            decision.evidence["bridge_evidence_fingerprint"]
            == staged_source.evidence_fingerprint
        )
        if task_id == "actual_artifact_bridge":
            assert decision.evidence["bridge_task_rng_fingerprint"] is None
            assert len(decision.evidence["bridge_parent_result_fingerprints"]) == 4
        else:
            assert decision.evidence["bridge_task_rng_fingerprint"] is not None
    assert (
        len(
            {
                view.binding_fingerprint
                for view in passing_bridge_views.as_mapping().values()
            }
        )
        == 6
    )
    assert not is_bridge_driver_task_view(
        copy.copy(passing_bridge_views.bridge_resource_preflight)
    )

    source = cast(
        StagedBridgeTaskEvidence,
        passing_bridge_views.bridge_tier_a_model_dgp.source,
    )
    mixed_run = build_bridge_driver_task_view(
        source,
        task_id="bridge_tier_a_model_dgp",
        run_id="9" * 64,
        binding_fingerprint="8" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "bridge_resource_preflight": (
                    passing_bridge_views.bridge_resource_preflight
                ),
                "bridge_tier_a_model_dgp": mixed_run,
            }
        )
        == "task-scoped scientific views mix calibration runs"
    )

    with pytest.raises(ModelError, match="sealed bridge-driver task view"):
        derive_g3_gate_decision(
            "bridge_tier_a_model_dgp",
            passing_bridge_views.bridge_tier_a_nonparametric_dgp,
        )
    with pytest.raises(TypeError):
        BridgeDriverTaskView()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        replace(
            passing_bridge_views.bridge_tier_a_model_dgp,
            task_id="bridge_tier_a_nonparametric_dgp",
        )


def _install_bridge_gate_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assembly_module,
        "is_bridge_driver_task_view",
        lambda value: isinstance(value, _BridgeViewStub),
    )


def _bridge_source_with(
    source: StagedBridgeTaskEvidence,
    **overrides: object,
) -> StagedBridgeTaskEvidence:
    result = copy.copy(source)
    for name, value in overrides.items():
        object.__setattr__(result, name, value)
    return result


def _bridge_view_stub(
    source_view: BridgeDriverTaskView,
    *,
    source: object | None = None,
    planned_looks: tuple[object, ...] | None = None,
    **overrides: object,
) -> _BridgeViewStub:
    values: dict[str, object] = {
        "task_id": source_view.task_id,
        "run_id": source_view.run_id,
        "binding_fingerprint": source_view.binding_fingerprint,
        "view_fingerprint": source_view.view_fingerprint,
        "selected_k": source_view.selected_k,
        "shared_binding_fingerprint": source_view.shared_binding_fingerprint,
        "base_preflight_fingerprint": source_view.base_preflight_fingerprint,
        "task_result_fingerprint": source_view.task_result_fingerprint,
        "parent_result_fingerprints": source_view.parent_result_fingerprints,
        "source_lineage_fingerprint": source_view.source_lineage_fingerprint,
        "materialization_fingerprint": source_view.materialization_fingerprint,
        "rng_fingerprint": source_view.rng_fingerprint,
        "known_trace_fingerprint": source_view.known_trace_fingerprint,
        "planned_look_fingerprints": source_view.planned_look_fingerprints,
        "planned_looks": (
            source_view.planned_looks if planned_looks is None else planned_looks
        ),
        "source": source_view.source if source is None else source,
    }
    values.update(overrides)
    return _BridgeViewStub(**values)


def _bridge_look(
    *,
    dgp: str = "model",
    look_index: int = 1,
    sample_count: int = 1_000,
    exceedances: int = 1_000,
    decision: str = "pass_above_tail",
    receipt: str = "a" * 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        dgp=dgp,
        look_index=look_index,
        sample_count=sample_count,
        exceedances=exceedances,
        decision=decision,
        stage_receipt_fingerprint=receipt,
    )


@pytest.mark.parametrize(
    ("task_id", "tamper"),
    (
        ("bridge_resource_preflight", "dgp"),
        ("bridge_resource_preflight", "looks"),
        ("bridge_tier_a_model_dgp", "dgp"),
        ("bridge_tier_a_model_dgp", "looks"),
        ("actual_artifact_bridge", "dgp"),
        ("actual_artifact_bridge", "looks"),
        ("actual_artifact_bridge", "rng"),
    ),
)
def test_bridge_stage_identity_rejects_wrong_dgp_look_or_rng_scope(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
    task_id: str,
    tamper: str,
) -> None:
    _install_bridge_gate_stubs(monkeypatch)
    original = passing_bridge_views.as_mapping()[task_id]
    source = cast(StagedBridgeTaskEvidence, original.source)
    if tamper == "dgp":
        source = _bridge_source_with(
            source,
            dgp=(None if source.dgp is not None else "model"),
        )
    view = _bridge_view_stub(
        original,
        source=source,
        planned_looks=(
            (_bridge_look(),) if tamper == "looks" else original.planned_looks
        ),
        **({"rng_fingerprint": "f" * 64} if tamper == "rng" else {}),
    )

    with pytest.raises(ModelError, match="invalid|inconsistent"):
        assembly_module._evaluate(task_id, view)


def test_bridge_stage_rejects_nonstaged_source(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _install_bridge_gate_stubs(monkeypatch)
    view = _bridge_view_stub(
        passing_bridge_views.bridge_resource_preflight,
        source=object(),
    )
    with pytest.raises(ModelError, match="lacks staged task evidence"):
        assembly_module._bridge_preflight(view)


@pytest.mark.parametrize(
    "tamper",
    (
        "wrong-source-dgp",
        "missing-looks",
        "too-many-looks",
        "look-dgp",
        "look-index",
        "sample-count",
        "negative-exceedances",
        "excess-exceedances",
        "receipt",
    ),
)
def test_tier_b_rejects_invalid_registered_look_geometry(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
    tamper: str,
) -> None:
    _install_bridge_gate_stubs(monkeypatch)
    original = passing_bridge_views.bridge_tier_b_model_dgp
    source = cast(StagedBridgeTaskEvidence, original.source)
    look = _bridge_look()
    looks: tuple[object, ...] = (look,)
    if tamper == "wrong-source-dgp":
        source = _bridge_source_with(source, dgp="nonparametric")
    elif tamper == "missing-looks":
        looks = ()
    elif tamper == "too-many-looks":
        looks = tuple(_bridge_look() for _ in range(11))
    elif tamper == "look-dgp":
        look.dgp = "nonparametric"
    elif tamper == "look-index":
        look.look_index = 2
    elif tamper == "sample-count":
        look.sample_count = 999
    elif tamper == "negative-exceedances":
        look.exceedances = -1
    elif tamper == "excess-exceedances":
        look.exceedances = 1_001
    elif tamper == "receipt":
        look.stage_receipt_fingerprint = "bad"
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(tamper)
    view = _bridge_view_stub(original, source=source, planned_looks=looks)

    with pytest.raises(ModelError, match="DGP/looks|too long|schedule/counts"):
        assembly_module._tier_b("bridge_tier_b_model_dgp", view)


def test_tier_b_rejects_inconsistent_or_nonterminal_local_decisions(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _install_bridge_gate_stubs(monkeypatch)
    original = passing_bridge_views.bridge_tier_b_model_dgp

    inconsistent = _bridge_view_stub(
        original,
        planned_looks=(_bridge_look(exceedances=15, decision="pass_above_tail"),),
    )
    with pytest.raises(ModelError, match="local look decision is inconsistent"):
        assembly_module._tier_b("bridge_tier_b_model_dgp", inconsistent)

    nonterminal = _bridge_view_stub(
        original,
        planned_looks=(_bridge_look(exceedances=15, decision="continue"),),
    )
    with pytest.raises(ModelError, match="sequence is nonterminal"):
        assembly_module._tier_b("bridge_tier_b_model_dgp", nonterminal)

    continued_after_decision = _bridge_view_stub(
        original,
        planned_looks=(
            _bridge_look(),
            _bridge_look(look_index=2, sample_count=2_000),
        ),
    )
    with pytest.raises(ModelError, match="continue after a decision"):
        assembly_module._tier_b(
            "bridge_tier_b_model_dgp",
            continued_after_decision,
        )


def test_tier_b_rederives_failed_tail_and_maximum_crossing_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _install_bridge_gate_stubs(monkeypatch)
    original = passing_bridge_views.bridge_tier_b_model_dgp
    source = cast(StagedBridgeTaskEvidence, original.source)

    failed_tail = _bridge_view_stub(
        original,
        source=_bridge_source_with(source, passed=False),
        planned_looks=(_bridge_look(exceedances=0, decision="fail_below_tail"),),
    )
    passed, evidence, summary = assembly_module._tier_b(
        "bridge_tier_b_model_dgp",
        failed_tail,
    )
    assert not passed
    assert evidence["decision"] == "fail_below_tail"
    assert summary["decision"] == "fail_below_tail"

    crossing_looks = tuple(
        _bridge_look(
            look_index=index,
            sample_count=index * 1_000,
            exceedances=index * 25,
            decision=("continue" if index < 10 else "fail_closed_at_maximum"),
        )
        for index in range(1, 11)
    )
    maximum = _bridge_view_stub(
        original,
        source=_bridge_source_with(source, passed=False),
        planned_looks=crossing_looks,
    )
    passed, evidence, summary = assembly_module._tier_b(
        "bridge_tier_b_model_dgp",
        maximum,
    )
    assert not passed
    assert evidence["pairs_used"] == 10_000
    assert summary["decision"] == "fail_closed_at_maximum"


def test_tier_b_rejects_source_pass_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _install_bridge_gate_stubs(monkeypatch)
    original = passing_bridge_views.bridge_tier_b_model_dgp
    view = _bridge_view_stub(
        original,
        planned_looks=(_bridge_look(exceedances=0, decision="fail_below_tail"),),
    )
    with pytest.raises(ModelError, match="internally inconsistent"):
        assembly_module._tier_b("bridge_tier_b_model_dgp", view)


def _artifact_integrity_view() -> _ArtifactIntegrityViewStub:
    binding = _binding()
    provenance = _ProvenanceStub("shared")
    design = SimpleNamespace(
        role="design",
        selected_k=2,
        binding=binding,
        provenance=provenance,
        reference=SimpleNamespace(fingerprint="d" * 64),
        generation_authorized=False,
        parameter_set_fingerprint="1" * 64,
        core_model_fingerprint="2" * 64,
    )
    production = SimpleNamespace(
        role="production",
        selected_k=2,
        binding=binding,
        provenance=provenance,
        reference=SimpleNamespace(fingerprint="e" * 64),
        generation_authorized=False,
        parameter_set_fingerprint="3" * 64,
        core_model_fingerprint="4" * 64,
    )
    bridge_parent = "5" * 64
    pair_receipt = "6" * 64
    pair = SimpleNamespace(
        design=design,
        production=production,
        reference=SimpleNamespace(fingerprint=pair_receipt),
        actual_bridge_parent_view_fingerprint=bridge_parent,
        generation_authorized=False,
    )
    source = SimpleNamespace(pair=pair, design=design, production=production)
    return _ArtifactIntegrityViewStub(
        task_id="artifact_integrity",
        run_id=binding.calibration_run_fingerprint,
        binding_fingerprint="7" * 64,
        view_fingerprint="8" * 64,
        source_fingerprint="9" * 64,
        source=source,
        actual_bridge_parent_view_fingerprint=bridge_parent,
        final_artifact_parent_fingerprint="a" * 64,
        report_set_fingerprint="b" * 64,
        artifact_binding_fingerprint="c" * 64,
        provenance_fingerprint="d" * 64,
        pair_receipt_fingerprint=pair_receipt,
        design_role_source_fingerprint="e" * 64,
        production_role_source_fingerprint="f" * 64,
    )


def _install_artifact_integrity_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assembly_module,
        "is_artifact_integrity_task_view",
        lambda value: isinstance(value, _ArtifactIntegrityViewStub),
    )


def test_artifact_integrity_rederives_exact_pair_and_withholds_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_artifact_integrity_stub(monkeypatch)
    view = _artifact_integrity_view()

    passed, evidence, summary = assembly_module._artifact_integrity(view)

    assert passed
    assert evidence["typed_schema_verified"]
    assert evidence["selected_k"] == 2
    assert summary["design_artifact_fingerprint"] == "d" * 64
    assert summary["production_artifact_fingerprint"] == "e" * 64


@pytest.mark.parametrize(
    "tamper",
    (
        "design-role",
        "production-role",
        "selected-k",
        "binding",
        "provenance",
        "duplicate-fingerprint",
        "pair-design-substitution",
        "pair-production-substitution",
        "pair-receipt",
        "bridge-parent",
        "design-generation",
        "production-generation",
        "pair-generation",
    ),
)
def test_artifact_integrity_fails_closed_for_each_pair_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _install_artifact_integrity_stub(monkeypatch)
    view = _artifact_integrity_view()
    source = view.source
    design = source.design
    production = source.production
    pair = source.pair
    if tamper == "design-role":
        design.role = "production"
    elif tamper == "production-role":
        production.role = "design"
    elif tamper == "selected-k":
        production.selected_k = 3
    elif tamper == "binding":
        production.binding = replace(_binding(), calibration_run_fingerprint="0" * 64)
    elif tamper == "provenance":
        production.provenance = _ProvenanceStub("different")
    elif tamper == "duplicate-fingerprint":
        production.reference.fingerprint = design.reference.fingerprint
    elif tamper == "pair-design-substitution":
        pair.design = copy.copy(design)
    elif tamper == "pair-production-substitution":
        pair.production = copy.copy(production)
    elif tamper == "pair-receipt":
        pair.reference.fingerprint = "0" * 64
    elif tamper == "bridge-parent":
        pair.actual_bridge_parent_view_fingerprint = "0" * 64
    elif tamper == "design-generation":
        design.generation_authorized = True
    elif tamper == "production-generation":
        production.generation_authorized = True
    elif tamper == "pair-generation":
        pair.generation_authorized = True
    else:  # pragma: no cover - closed parametrization above
        raise AssertionError(tamper)

    passed, evidence, _summary = assembly_module._artifact_integrity(view)

    assert not passed
    assert not evidence["typed_schema_verified"]


def _decision_stub(
    source: object,
    *,
    passed: bool = True,
    summary: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        _source=source,
        passed=passed,
        summary=({} if summary is None else summary),
    )


def _cross_binding_kwargs() -> dict[str, str]:
    return {
        "config_fingerprint": CANONICAL_CONFIG_FINGERPRINT,
        "processed_data_fingerprint": CANONICAL_PROCESSED_DATA_FINGERPRINT,
        "calibration_input_fingerprint": (CANONICAL_CALIBRATION_INPUT_FINGERPRINT),
    }


def _install_cross_binding_stub_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assembly_module,
        "ArtifactIntegrityTaskView",
        _ArtifactIntegrityViewStub,
    )
    monkeypatch.setattr(
        assembly_module,
        "BridgeDriverTaskView",
        _BridgeViewStub,
    )
    monkeypatch.setattr(
        assembly_module,
        "InputIntegrityTaskView",
        _InputIntegrityViewStub,
    )


def test_cross_binding_propagates_structural_error_and_selected_k_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assembly_module,
        "task_view_cross_binding_error",
        lambda _sources: "synthetic structural lineage error",
    )
    assert (
        assembly_module._cross_binding_error({}, **_cross_binding_kwargs())
        == "synthetic structural lineage error"
    )

    monkeypatch.setattr(
        assembly_module,
        "task_view_cross_binding_error",
        lambda _sources: None,
    )
    decisions = {
        "first": _decision_stub(object(), summary={"selected_k": 2}),
        "second": _decision_stub(object(), summary={"selected_k": 3}),
    }
    assert (
        assembly_module._cross_binding_error(
            decisions,  # type: ignore[arg-type]
            **_cross_binding_kwargs(),
        )
        == "typed decisions disagree on selected K"
    )


def test_cross_binding_rejects_mixed_staged_bridge_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cross_binding_stub_types(monkeypatch)
    monkeypatch.setattr(
        assembly_module,
        "task_view_cross_binding_error",
        lambda _sources: None,
    )
    decisions = {
        "bridge_resource_preflight": _decision_stub(
            _BridgeViewStub(shared_binding_fingerprint="a" * 64)
        ),
        "bridge_tier_a_model_dgp": _decision_stub(
            _BridgeViewStub(shared_binding_fingerprint="b" * 64)
        ),
    }
    assert (
        assembly_module._cross_binding_error(
            decisions,  # type: ignore[arg-type]
            **_cross_binding_kwargs(),
        )
        == "bridge tasks mix distinct staged executions"
    )


@pytest.mark.parametrize(
    ("tamper", "expected"),
    (
        ("artifact-bindings", "design and production artifact bindings differ"),
        ("config", "artifact config fingerprint differs from bundle"),
        ("processed", "artifact processed-data fingerprint differs from bundle"),
        ("calibration", "artifact calibration-input fingerprint differs from bundle"),
        ("bridge-k", "bridge selected K differs from typed artifacts"),
        (
            "design-block-core",
            "design_block_calibration kernel core fingerprint differs from artifact",
        ),
        (
            "production-block-core",
            "production_block_calibration kernel core fingerprint differs "
            "from artifact",
        ),
        ("input-binding", "preflight and artifact bindings differ"),
        ("provenance", "artifact provenance differs from launch preflight"),
        ("bridge-preflight", "bridge launch preflight differs from input integrity"),
    ),
)
def test_cross_binding_rejects_artifact_bundle_and_launch_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    expected: str,
) -> None:
    _install_cross_binding_stub_types(monkeypatch)
    monkeypatch.setattr(
        assembly_module,
        "task_view_cross_binding_error",
        lambda _sources: None,
    )
    artifact = _artifact_integrity_view()
    design = artifact.source.design
    production = artifact.source.production
    binding = design.binding
    decisions: dict[str, SimpleNamespace] = {
        "artifact_integrity": _decision_stub(
            artifact,
            summary={"selected_k": 2},
        )
    }
    if tamper == "artifact-bindings":
        production.binding = replace(binding, calibration_run_fingerprint="0" * 64)
    elif tamper in {"config", "processed", "calibration"}:
        changes = {
            "config": {"config_fingerprint": "0" * 64},
            "processed": {"processed_data_fingerprint": "0" * 64},
            "calibration": {"calibration_input_fingerprint": "0" * 64},
        }[tamper]
        changed_binding = replace(binding, **changes)
        design.binding = changed_binding
        production.binding = changed_binding
    elif tamper == "bridge-k":
        decisions["bridge_resource_preflight"] = _decision_stub(
            _BridgeViewStub(
                shared_binding_fingerprint="a" * 64,
                selected_k=3,
                base_preflight_fingerprint=binding.preflight_fingerprint,
            ),
            summary={"selected_k": 2},
        )
    elif tamper in {"design-block-core", "production-block-core"}:
        task_id = (
            "design_block_calibration"
            if tamper.startswith("design")
            else "production_block_calibration"
        )
        decisions[task_id] = _decision_stub(
            object(),
            summary={"core_model_fingerprint": "0" * 64},
        )
    else:
        bridge = _BridgeViewStub(
            shared_binding_fingerprint="a" * 64,
            selected_k=2,
            base_preflight_fingerprint=binding.preflight_fingerprint,
        )
        if tamper == "bridge-preflight":
            bridge.base_preflight_fingerprint = "0" * 64
        if tamper == "bridge-preflight":
            decisions["bridge_resource_preflight"] = _decision_stub(
                bridge,
                summary={"selected_k": 2},
            )
        launch = SimpleNamespace(
            source_code_fingerprint=design.provenance.source_code_fingerprint,
            dependency_lock_fingerprint=(design.provenance.dependency_lock_fingerprint),
            fingerprint=binding.preflight_fingerprint,
        )
        input_binding = binding
        if tamper == "input-binding":
            input_binding = replace(binding, calibration_run_fingerprint="0" * 64)
        elif tamper == "provenance":
            launch.source_code_fingerprint = "0" * 64
        input_view = _InputIntegrityViewStub(
            source=SimpleNamespace(binding=input_binding, launch_preflight=launch)
        )
        decisions["input_integrity"] = _decision_stub(input_view)

    assert (
        assembly_module._cross_binding_error(
            decisions,  # type: ignore[arg-type]
            **_cross_binding_kwargs(),
        )
        == expected
    )


def test_cross_binding_accepts_consistent_artifact_launch_and_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_cross_binding_stub_types(monkeypatch)
    monkeypatch.setattr(
        assembly_module,
        "task_view_cross_binding_error",
        lambda _sources: None,
    )
    artifact = _artifact_integrity_view()
    design = artifact.source.design
    binding = design.binding
    bridge = _BridgeViewStub(
        shared_binding_fingerprint="a" * 64,
        selected_k=2,
        base_preflight_fingerprint=binding.preflight_fingerprint,
    )
    launch = SimpleNamespace(
        source_code_fingerprint=design.provenance.source_code_fingerprint,
        dependency_lock_fingerprint=design.provenance.dependency_lock_fingerprint,
        fingerprint=binding.preflight_fingerprint,
    )
    input_view = _InputIntegrityViewStub(
        source=SimpleNamespace(binding=binding, launch_preflight=launch)
    )
    decisions = {
        "artifact_integrity": _decision_stub(
            artifact,
            summary={"selected_k": 2},
        ),
        "bridge_resource_preflight": _decision_stub(
            bridge,
            summary={"selected_k": 2},
        ),
        "design_block_calibration": _decision_stub(
            object(),
            summary={"core_model_fingerprint": design.core_model_fingerprint},
        ),
        "production_block_calibration": _decision_stub(
            object(),
            summary={
                "core_model_fingerprint": (
                    artifact.source.production.core_model_fingerprint
                )
            },
        ),
        "input_integrity": _decision_stub(input_view),
    }
    assert (
        assembly_module._cross_binding_error(
            decisions,  # type: ignore[arg-type]
            **_cross_binding_kwargs(),
        )
        is None
    )


def _install_task_lineage_stub_types(monkeypatch: pytest.MonkeyPatch) -> None:
    replacements = {
        "InputIntegrityTaskView": _InputIntegrityViewStub,
        "CandidateViabilityTaskView": _CandidateViabilityViewStub,
        "CandidateSelectionTaskView": _CandidateSelectionViewStub,
        "SealedHoldoutVetoView": _HoldoutViewStub,
        "MacroStabilityTaskView": _MacroViewStub,
        "ProductionFitSameKView": _ProductionFitViewStub,
        "AdequacyCellTaskView": _AdequacyViewStub,
        "FourCellAdequacyHolmView": _AdequacyHolmViewStub,
        "DependenceTaskView": _CrossDependenceViewStub,
        "FourCellDependenceHolmView": _CrossDependenceHolmViewStub,
        "BlockCalibrationTaskView": _CrossBlockViewStub,
        "BridgeDriverTaskView": _BridgeViewStub,
        "ArtifactIntegrityTaskView": _ArtifactIntegrityViewStub,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(assembly_module, name, value)


def _lineage_view(
    view_type: type[SimpleNamespace],
    task_id: str,
    *,
    run_id: str = "d" * 64,
    binding_fingerprint: str | None = None,
    **values: object,
) -> SimpleNamespace:
    return view_type(
        task_id=task_id,
        run_id=run_id,
        binding_fingerprint=(
            hashlib.sha256(task_id.encode()).hexdigest()
            if binding_fingerprint is None
            else binding_fingerprint
        ),
        **values,
    )


def _lineage_bridge(
    task_id: str,
    *,
    parents: tuple[tuple[str, str], ...],
    result_fingerprint: str | None = None,
    shared_binding_fingerprint: str = "a" * 64,
    **values: object,
) -> _BridgeViewStub:
    return cast(
        _BridgeViewStub,
        _lineage_view(
            _BridgeViewStub,
            task_id,
            shared_binding_fingerprint=shared_binding_fingerprint,
            parent_result_fingerprints=parents,
            task_result_fingerprint=(
                hashlib.sha256(f"result:{task_id}".encode()).hexdigest()
                if result_fingerprint is None
                else result_fingerprint
            ),
            **values,
        ),
    )


def test_task_lineage_rejects_run_launch_binding_and_task_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    assert task_view_cross_binding_error({"opaque": object()}) is None

    first = _lineage_bridge("bridge_resource_preflight", parents=())
    second = _lineage_bridge(
        "bridge_tier_a_model_dgp",
        parents=(("bridge_resource_preflight", first.task_result_fingerprint),),
        run_id="e" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "bridge_resource_preflight": first,
                "bridge_tier_a_model_dgp": second,
            }
        )
        == "task-scoped scientific views mix calibration runs"
    )

    input_view = _lineage_view(
        _InputIntegrityViewStub,
        "input_integrity",
        source=SimpleNamespace(
            binding=SimpleNamespace(calibration_run_fingerprint="e" * 64)
        ),
    )
    assert (
        task_view_cross_binding_error({"input_integrity": input_view})
        == "task-scoped scientific views differ from the input launch"
    )

    second.run_id = first.run_id
    second.binding_fingerprint = first.binding_fingerprint
    assert (
        task_view_cross_binding_error(
            {
                "bridge_resource_preflight": first,
                "bridge_tier_a_model_dgp": second,
            }
        )
        == "task-scoped scientific views reuse a task binding"
    )

    wrong_task = _lineage_bridge("bridge_resource_preflight", parents=())
    assert (
        task_view_cross_binding_error({"bridge_tier_a_model_dgp": wrong_task})
        == "task-scoped scientific view is bound to the wrong task"
    )


def test_task_lineage_rejects_bridge_binding_parent_and_artifact_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    resource = _lineage_bridge("bridge_resource_preflight", parents=())
    tier_a = _lineage_bridge(
        "bridge_tier_a_model_dgp",
        parents=(("bridge_resource_preflight", resource.task_result_fingerprint),),
        shared_binding_fingerprint="b" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "bridge_resource_preflight": resource,
                "bridge_tier_a_model_dgp": tier_a,
            }
        )
        == "bridge tasks mix distinct preregistered bindings"
    )

    wrong_parent = _lineage_bridge("bridge_tier_a_model_dgp", parents=())
    assert (
        task_view_cross_binding_error({"bridge_tier_a_model_dgp": wrong_parent})
        == "bridge_tier_a_model_dgp has the wrong bridge parent lineage"
    )

    inconsistent_parent = _lineage_bridge(
        "bridge_tier_a_model_dgp",
        parents=(("bridge_resource_preflight", "0" * 64),),
    )
    assert (
        task_view_cross_binding_error(
            {
                "bridge_resource_preflight": resource,
                "bridge_tier_a_model_dgp": inconsistent_parent,
            }
        )
        == "bridge_tier_a_model_dgp bridge parent result is inconsistent"
    )

    actual = _lineage_bridge(
        "actual_artifact_bridge",
        parents=(
            ("bridge_tier_a_model_dgp", "1" * 64),
            ("bridge_tier_a_nonparametric_dgp", "2" * 64),
            ("bridge_tier_b_model_dgp", "3" * 64),
            ("bridge_tier_b_nonparametric_dgp", "4" * 64),
        ),
    )
    artifact = _lineage_view(
        _ArtifactIntegrityViewStub,
        "artifact_integrity",
        _actual_bridge_parent=object(),
    )
    assert (
        task_view_cross_binding_error(
            {
                "actual_artifact_bridge": actual,
                "artifact_integrity": artifact,
            }
        )
        == "artifact integrity does not descend from the exact bridge view"
    )


def _candidate_lineage_views() -> tuple[
    _CandidateViabilityViewStub,
    _CandidateSelectionViewStub,
]:
    viability = cast(
        _CandidateViabilityViewStub,
        _lineage_view(
            _CandidateViabilityViewStub,
            "design_candidate_viability",
            view_fingerprint="1" * 64,
            source_content_fingerprint="2" * 64,
            fit_set_fingerprint="3" * 64,
            viability_result_fingerprint="4" * 64,
        ),
    )
    selection = cast(
        _CandidateSelectionViewStub,
        _lineage_view(
            _CandidateSelectionViewStub,
            "design_predictive_selection",
            _viability_parent=viability,
            viability_view_fingerprint=viability.view_fingerprint,
            source_content_fingerprint=viability.source_content_fingerprint,
            fit_set_fingerprint=viability.fit_set_fingerprint,
            viability_result_fingerprint=viability.viability_result_fingerprint,
            view_fingerprint="5" * 64,
            selected_n_states=2,
        ),
    )
    return viability, selection


def test_task_lineage_rejects_candidate_holdout_and_production_parent_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    viability, selection = _candidate_lineage_views()
    selection._viability_parent = object()
    assert (
        task_view_cross_binding_error(
            {
                "design_candidate_viability": viability,
                "design_predictive_selection": selection,
            }
        )
        == "candidate selection does not descend from viability"
    )

    viability, selection = _candidate_lineage_views()
    holdout = _lineage_view(
        _HoldoutViewStub,
        "sealed_holdout_veto",
        _selection_source=object(),
        selection_view_fingerprint=selection.view_fingerprint,
        selected_model_fingerprint="6" * 64,
        selected_n_states=2,
    )
    assert (
        task_view_cross_binding_error(
            {
                "design_predictive_selection": selection,
                "sealed_holdout_veto": holdout,
            }
        )
        == "sealed holdout does not descend from predictive selection"
    )

    production = _lineage_view(
        _ProductionFitViewStub,
        "production_fit_same_k",
        _selection_source=object(),
        selection_view_fingerprint=selection.view_fingerprint,
        design_selected_k=2,
        design_model_fingerprint="6" * 64,
        production_fit_fingerprint="7" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "design_predictive_selection": selection,
                "production_fit_same_k": production,
            }
        )
        == "production fit does not descend from predictive selection"
    )

    holdout._selection_source = selection
    production._selection_source = selection
    production.design_model_fingerprint = "0" * 64
    assert (
        task_view_cross_binding_error(
            {
                "design_predictive_selection": selection,
                "sealed_holdout_veto": holdout,
                "production_fit_same_k": production,
            }
        )
        == "design-model lineage differs between holdout and production fit"
    )


def test_task_lineage_rejects_block_and_macro_model_parent_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    holdout = _lineage_view(
        _HoldoutViewStub,
        "sealed_holdout_veto",
        selected_model_fingerprint="6" * 64,
        selected_n_states=2,
    )
    block = _lineage_view(
        _CrossBlockViewStub,
        "design_block_calibration",
        parent_model_fingerprint="0" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "sealed_holdout_veto": holdout,
                "design_block_calibration": block,
            }
        )
        == "design_block_calibration parent model fingerprint is inconsistent"
    )

    wrong_role = _lineage_view(
        _MacroViewStub,
        "design_macro_stability",
        role="production",
    )
    assert (
        task_view_cross_binding_error({"design_macro_stability": wrong_role})
        == "design_macro_stability has cross-role macro evidence"
    )

    _viability, selection = _candidate_lineage_views()
    wrong_parent = _lineage_view(
        _MacroViewStub,
        "design_macro_stability",
        role="design",
        _parent_view=object(),
        parent_view_fingerprint=selection.view_fingerprint,
        parent_model_fingerprint="6" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "design_predictive_selection": selection,
                "design_macro_stability": wrong_parent,
            }
        )
        == "design_macro_stability does not descend from its exact parent view"
    )

    wrong_model = _lineage_view(
        _MacroViewStub,
        "design_macro_stability",
        role="design",
        parent_model_fingerprint="0" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "sealed_holdout_veto": holdout,
                "design_macro_stability": wrong_model,
            }
        )
        == "design_macro_stability parent model fingerprint is inconsistent"
    )


def test_task_lineage_rejects_adequacy_model_and_holm_parent_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    holdout = _lineage_view(
        _HoldoutViewStub,
        "sealed_holdout_veto",
        selected_model_fingerprint="6" * 64,
        selected_n_states=2,
    )
    adequacy = _lineage_view(
        _AdequacyViewStub,
        "design_latent_correctness",
        parent_model_fingerprint="0" * 64,
    )
    assert (
        task_view_cross_binding_error(
            {
                "sealed_holdout_veto": holdout,
                "design_latent_correctness": adequacy,
            }
        )
        == "design_latent_correctness parent model fingerprint is inconsistent"
    )

    adequacy.parent_model_fingerprint = holdout.selected_model_fingerprint
    holm = _lineage_view(
        _AdequacyHolmViewStub,
        "four_cell_adequacy_holm",
        source_views=(object(), object(), object(), object()),
    )
    assert (
        task_view_cross_binding_error(
            {
                "sealed_holdout_veto": holdout,
                "design_latent_correctness": adequacy,
                "four_cell_adequacy_holm": holm,
            }
        )
        == "adequacy Holm does not retain the exact prerequisite views"
    )


def _cross_dependence_view(
    task_id: str,
    *,
    role: str,
    fragmentation_parent: str | None,
    parent_cores: tuple[str, str] = ("1" * 64, "2" * 64),
    model_fingerprint: str = "6" * 64,
) -> _CrossDependenceViewStub:
    return cast(
        _CrossDependenceViewStub,
        _lineage_view(
            _CrossDependenceViewStub,
            task_id,
            role=role,
            fragmentation_parent_view_fingerprint=fragmentation_parent,
            parent_block_core_fingerprints=parent_cores,
            view_fingerprint=hashlib.sha256(f"view:{task_id}".encode()).hexdigest(),
            cells=(SimpleNamespace(model_fingerprint=model_fingerprint),),
        ),
    )


def test_task_lineage_rejects_dependence_temporal_and_model_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    fragmentation = _cross_dependence_view(
        "design_fragmentation",
        role="design",
        fragmentation_parent="future",
    )
    assert (
        task_view_cross_binding_error({"design_fragmentation": fragmentation})
        == "design fragmentation contains future lineage"
    )

    fragmentation = _cross_dependence_view(
        "design_fragmentation",
        role="design",
        fragmentation_parent=None,
    )
    dependence = _cross_dependence_view(
        "design_dependence",
        role="design",
        fragmentation_parent=fragmentation.view_fingerprint,
        parent_cores=("3" * 64, "4" * 64),
    )
    assert (
        task_view_cross_binding_error(
            {
                "design_fragmentation": fragmentation,
                "design_dependence": dependence,
            }
        )
        == "design dependence block-core lineage differs"
    )

    dependence.parent_block_core_fingerprints = (
        fragmentation.parent_block_core_fingerprints
    )
    dependence.fragmentation_parent_view_fingerprint = "0" * 64
    assert (
        task_view_cross_binding_error(
            {
                "design_fragmentation": fragmentation,
                "design_dependence": dependence,
            }
        )
        == "design dependence does not descend from fragmentation"
    )

    holdout = _lineage_view(
        _HoldoutViewStub,
        "sealed_holdout_veto",
        selected_model_fingerprint="6" * 64,
        selected_n_states=2,
    )
    fragmentation.cells = (SimpleNamespace(model_fingerprint="0" * 64),)
    assert (
        task_view_cross_binding_error(
            {
                "sealed_holdout_veto": holdout,
                "design_fragmentation": fragmentation,
            }
        )
        == "design_fragmentation parent model fingerprint is inconsistent"
    )


def test_task_lineage_rejects_dependence_holm_parent_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_task_lineage_stub_types(monkeypatch)
    design = _cross_dependence_view(
        "design_dependence",
        role="design",
        fragmentation_parent="1" * 64,
    )
    production = _cross_dependence_view(
        "production_dependence",
        role="production",
        fragmentation_parent="2" * 64,
    )
    holm = _lineage_view(
        _CrossDependenceHolmViewStub,
        "four_cell_dependence_holm",
        source_views=(object(), production),
    )
    assert (
        task_view_cross_binding_error(
            {
                "design_dependence": design,
                "production_dependence": production,
                "four_cell_dependence_holm": holm,
            }
        )
        == "dependence Holm does not retain the exact prerequisite views"
    )


@pytest.mark.parametrize(
    "gate_id",
    [
        "design_candidate_viability",
        "design_predictive_selection",
        "design_macro_stability",
        "production_macro_stability",
        "design_block_calibration",
        "design_fragmentation",
        "production_block_calibration",
        "production_fragmentation",
        "production_fit_same_k",
        "design_dependence",
        "production_dependence",
        "four_cell_dependence_holm",
        "artifact_integrity",
    ],
)
def test_each_remaining_evaluator_rejects_untyped_input(gate_id: str) -> None:
    with pytest.raises(ModelError, match="requires|evidence"):
        derive_g3_gate_decision(gate_id, object())


def test_missing_or_explicit_failure_creates_complete_non_authorizing_bundle() -> None:
    empty = assemble_typed_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        decisions=(),
    )
    assert len(empty.records) == 26
    assert empty.records[0].status == "failed"
    assert all(record.status == "failed" for record in empty.records)
    assert not empty.passed
    assert not empty.generation_authorized
    assert empty.design_artifact_fingerprint is None
    verify_g3_evidence_bundle(empty)

    failed = assemble_typed_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        decisions=(),
        failures=(G3TaskFailure("input_integrity", "IntegrityError", "bad input"),),
    )
    assert failed.records[0].summary["failure_type"] == "IntegrityError"
    assert failed.records[1].summary["status"] == "blocked"
    assert not failed.generation_authorized


@dataclass(frozen=True)
class _ReferenceStub:
    fingerprint: str


@dataclass(frozen=True)
class _ArtifactStub:
    reference: _ReferenceStub


@dataclass(frozen=True)
class _ArtifactViewStub:
    source: ArtifactIntegrityDecisionEvidence


@dataclass(frozen=True)
class _ArtifactPairStub:
    design: _ArtifactStub
    production: _ArtifactStub


def test_complete_sealed_26_decision_path_is_passed_but_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate(
        gate_id: str, result: object
    ) -> tuple[bool, dict[str, object], dict[str, object]]:
        return True, {"typed_result": gate_id}, {"selected_k": 2}

    monkeypatch.setattr(assembly_module, "_evaluate", fake_evaluate)
    monkeypatch.setattr(assembly_module, "_cross_binding_error", lambda *a, **k: None)
    pair = ArtifactIntegrityDecisionEvidence(
        _ArtifactPairStub(
            design=_ArtifactStub(_ReferenceStub("d" * 64)),
            production=_ArtifactStub(_ReferenceStub("e" * 64)),
        )  # type: ignore[arg-type]
    )
    artifact_view = _ArtifactViewStub(pair)
    monkeypatch.setattr(assembly_module, "ArtifactIntegrityTaskView", _ArtifactViewStub)
    decisions = tuple(
        derive_g3_gate_decision(
            gate_id,
            artifact_view if gate_id == "artifact_integrity" else {"typed": gate_id},
        )
        for gate_id in G3_GATE_ORDER
    )
    bundle = assemble_typed_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        decisions=decisions,
    )

    assert bundle.passed
    assert bundle.generation_authorized is False
    assert bundle.design_artifact_fingerprint == "d" * 64
    assert bundle.production_artifact_fingerprint == "e" * 64
    verify_g3_evidence_bundle(bundle)

    copied = replace(bundle)
    assert copied.generation_authorized is False
    verify_g3_evidence_bundle(copied)


def test_assembler_rejects_duplicates_unsealed_failures_and_bad_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assembly_module,
        "_evaluate",
        lambda gate_id, result: (True, {"gate": gate_id}, {"gate": gate_id}),
    )
    decision = derive_g3_gate_decision("input_integrity", object())
    with pytest.raises(ModelError, match="duplicate"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(decision, decision),
        )
    with pytest.raises(ModelError, match="duplicate decision/failure"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(decision,),
            failures=(G3TaskFailure("input_integrity", "Failure", "failed"),),
        )
    with pytest.raises(ModelError, match="fingerprint"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="bad",
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(),
        )
    with pytest.raises(ModelError, match="unregistered"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(),
            failures=(G3TaskFailure("unknown", "Failure", "failed"),),
        )


def test_closed_registry_and_assembler_reject_unsealed_or_relabelled_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelError, match="closed registry"):
        derive_g3_gate_decision("unknown", object())
    with pytest.raises(ModelError, match="sealed typed decisions"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(object(),),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        assembly_module,
        "_evaluate",
        lambda gate_id, _result: (True, {"task_id": gate_id}, {}),
    )
    decision = derive_g3_gate_decision("input_integrity", object())
    object.__setattr__(decision, "gate_id", "unknown")
    with pytest.raises(ModelError, match="unregistered task"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(decision,),
        )


@pytest.mark.parametrize(
    "failure",
    (
        G3TaskFailure("input_integrity", "", "message"),
        G3TaskFailure("input_integrity", "x" * 129, "message"),
        G3TaskFailure("input_integrity", "Failure", ""),
        G3TaskFailure("input_integrity", "Failure", "x" * 513),
    ),
)
def test_task_failure_text_must_be_present_and_bounded(
    failure: G3TaskFailure,
) -> None:
    with pytest.raises(ModelError, match="empty or unbounded"):
        assemble_typed_g3_evidence_bundle(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            decisions=(),
            failures=(failure,),
        )


def test_artifact_record_surfaces_cross_binding_failure_without_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assembly_module,
        "_evaluate",
        lambda gate_id, _result: (
            True,
            {"typed_result": gate_id},
            {"selected_k": 2},
        ),
    )
    monkeypatch.setattr(
        assembly_module,
        "_cross_binding_error",
        lambda *_args, **_kwargs: "synthetic cross-binding mismatch",
    )
    decision = derive_g3_gate_decision("artifact_integrity", object())
    bundle = assemble_typed_g3_evidence_bundle(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        decisions=(decision,),
    )
    artifact = next(
        record for record in bundle.records if record.gate_id == "artifact_integrity"
    )

    assert artifact.status == "failed"
    assert artifact.summary["cross_binding_failure"] == (
        "synthetic cross-binding mismatch"
    )
    assert bundle.design_artifact_fingerprint is None
    assert bundle.production_artifact_fingerprint is None
