from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import prpg.model.g3_success_bridge_stage as subject
from prpg.errors import ModelError
from prpg.model.bridge_driver import CanonicalBridgeRequest, StagedBridgeTaskEvidence
from prpg.model.bridge_fixed_k import ActualBridgeExecution
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_success_production_stage import CanonicalG3ProductionStageResult
from prpg.model.g3_success_runner import CanonicalG3DesignStageResult
from prpg.model.g3_task_publication import ScientificTaskSourceSlots
from prpg.model.run_store import CalibrationRunStore, LaunchLease


def _immutable(values: list[int] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    array = np.asarray(values)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


class _RunnerStore:
    def __init__(self, root: Path) -> None:
        self.run_path = root / "calibration" / ("1" * 64)

    def verify(self, run_id: str) -> object:
        assert run_id == "1" * 64
        return SimpleNamespace(path=self.run_path)


def _preflight() -> CanonicalG3Preflight:
    return cast(CanonicalG3Preflight, SimpleNamespace(workers=9, fingerprint="a" * 64))


def _runner_inputs(
    tmp_path: Path,
) -> tuple[
    CanonicalG3DesignStageResult,
    CanonicalG3ProductionStageResult,
    CanonicalG3Preflight,
    CalibrationRunStore,
    CalibrationCheckpointStore,
    LaunchLease,
]:
    return (
        cast(CanonicalG3DesignStageResult, object()),
        cast(CanonicalG3ProductionStageResult, object()),
        _preflight(),
        cast(CalibrationRunStore, _RunnerStore(tmp_path)),
        cast(CalibrationCheckpointStore, object()),
        cast(LaunchLease, SimpleNamespace(run_id="1" * 64)),
    )


def _patch_runner_scaffolding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str],
    actual_bridge: ActualBridgeExecution,
    request: CanonicalBridgeRequest,
    session: object,
) -> None:
    monkeypatch.setattr(subject, "_verify_stage_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject,
        "_assemble_actual_bridge_from_stages",
        lambda *_args, **_kwargs: actual_bridge,
    )
    monkeypatch.setattr(
        subject,
        "_canonical_bridge_request",
        lambda *_args, **_kwargs: request,
    )

    def prepare(_request: object, root: Path) -> object:
        events.append(f"prepare:{root.name}")
        return session

    monkeypatch.setattr(subject, "prepare_canonical_staged_bridge", prepare)
    monkeypatch.setattr(subject, "_verify_staged_completion", lambda *_args: None)
    monkeypatch.setattr(
        subject,
        "_verify_returned_publications",
        lambda *_args: None,
    )


def test_closed_bridge_stage_orders_six_tasks_and_tier_b_four_way_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    design, production, preflight, store, checkpoints, lease = _runner_inputs(tmp_path)
    actual_bridge = cast(ActualBridgeExecution, object())
    request = cast(CanonicalBridgeRequest, object())
    session = object()
    _patch_runner_scaffolding(
        monkeypatch,
        events=events,
        actual_bridge=actual_bridge,
        request=request,
        session=session,
    )

    evidence_by_task = {
        task_id: cast(
            StagedBridgeTaskEvidence,
            SimpleNamespace(task_id=task_id, passed=True),
        )
        for task_id in G3_GATE_ORDER[19:25]
    }

    def resource(_session: object) -> StagedBridgeTaskEvidence:
        events.append("execute:bridge_resource_preflight")
        return evidence_by_task["bridge_resource_preflight"]

    def tier_a(_session: object, dgp: str) -> StagedBridgeTaskEvidence:
        task_id = f"bridge_tier_a_{dgp}_dgp"
        events.append(f"execute:{task_id}")
        return evidence_by_task[task_id]

    def tier_b_look(_session: object, dgp: str) -> object:
        task_id = f"bridge_tier_b_{dgp}_dgp"
        events.append(f"look:{task_id}")
        return SimpleNamespace(decision="pass_above_tail")

    def tier_b_complete(_session: object, dgp: str) -> StagedBridgeTaskEvidence:
        task_id = f"bridge_tier_b_{dgp}_dgp"
        events.append(f"complete:{task_id}")
        return evidence_by_task[task_id]

    def actual(_session: object) -> StagedBridgeTaskEvidence:
        events.append("execute:actual_artifact_bridge")
        return evidence_by_task["actual_artifact_bridge"]

    monkeypatch.setattr(subject, "execute_staged_bridge_resource_benchmark", resource)
    monkeypatch.setattr(subject, "execute_staged_bridge_tier_a", tier_a)
    monkeypatch.setattr(subject, "execute_staged_bridge_tier_b_next_look", tier_b_look)
    monkeypatch.setattr(subject, "complete_staged_bridge_tier_b_task", tier_b_complete)
    monkeypatch.setattr(subject, "execute_staged_actual_artifact_bridge", actual)

    def slots(evidence: object, *, task_id: str) -> ScientificTaskSourceSlots:
        assert cast(Any, evidence).task_id == task_id
        events.append(f"slots:{task_id}")
        return ScientificTaskSourceSlots(task_id, (), (), ())

    monkeypatch.setattr(subject, "bridge_executor_source_slots", slots)

    def bind(**kwargs: object) -> object:
        task_id = cast(str, kwargs["task_id"])
        events.append(f"bind:{task_id}")
        return SimpleNamespace(task_id=task_id, fingerprint=task_id[0] * 64)

    monkeypatch.setattr(subject, "build_scientific_task_binding", bind)

    def view(
        evidence: object,
        *,
        task_id: str,
        run_id: str,
        binding_fingerprint: str,
    ) -> object:
        del evidence, run_id, binding_fingerprint
        events.append(f"view:{task_id}")
        return SimpleNamespace(task_id=task_id)

    monkeypatch.setattr(subject, "build_bridge_driver_task_view", view)

    def raw_looks(**kwargs: object) -> object:
        task_id = cast(str, kwargs["task_id"])
        events.append(f"raw-looks:{task_id}")
        return SimpleNamespace(task_id=task_id)

    monkeypatch.setattr(subject, "publish_raw_planned_look_prefix", raw_looks)

    def parity(prefix: object, **_kwargs: object) -> object:
        task_id = cast(Any, prefix).task_id
        events.append(f"parity:{task_id}")
        return SimpleNamespace(task_id=task_id)

    monkeypatch.setattr(subject, "verify_planned_look_binding_view_parity", parity)

    def publish(**kwargs: object) -> object:
        binding = cast(Any, kwargs["binding"])
        events.append(f"publish:{binding.task_id}")
        return SimpleNamespace(
            binding=binding,
            decision=SimpleNamespace(passed=True),
            checkpoint=object(),
            task_result=SimpleNamespace(status="passed"),
        )

    monkeypatch.setattr(subject, "publish_exact_next_scientific_task", publish)

    result = subject.run_canonical_g3_bridge_stage(
        design,
        production,
        launch_preflight=preflight,
        live_preflight=preflight,
        run_store=store,
        checkpoint_store=checkpoints,
        lease=lease,
        master_seed=20260714,
        scientific_version=1,
    )

    expected = G3_GATE_ORDER[19:25]
    assert tuple(item.binding.task_id for item in result.publications) == expected
    assert result.actual_bridge is actual_bridge
    assert result.request is request
    assert result.session is session
    assert [
        event.removeprefix("publish:")
        for event in events
        if event.startswith("publish:")
    ] == list(expected)
    for task_id in expected:
        assert events.index(f"slots:{task_id}") < events.index(f"bind:{task_id}")
        assert events.index(f"bind:{task_id}") < events.index(f"view:{task_id}")
        assert events.index(f"view:{task_id}") < events.index(f"publish:{task_id}")
    for task_id in expected[3:5]:
        assert events.index(f"complete:{task_id}") < events.index(
            f"raw-looks:{task_id}"
        )
        assert events.index(f"raw-looks:{task_id}") < events.index(f"bind:{task_id}")
        assert events.index(f"view:{task_id}") < events.index(f"parity:{task_id}")
        assert events.index(f"parity:{task_id}") < events.index(f"publish:{task_id}")
    assert "prepare:canonical-bridge-execution-v1" in events


def test_actual_bridge_uses_exact_lengths_counts_and_four_offset_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_seed = 7
    version = 1
    design_blocks = (object(), object())
    production_blocks = (object(), object())
    design_offsets = (
        _immutable(np.zeros((4, 2, 3), dtype=np.float64)),
        _immutable(np.ones((4, 2, 3), dtype=np.float64)),
    )
    production_offsets = (
        _immutable(np.full((4, 2, 3), 2.0, dtype=np.float64)),
        _immutable(np.full((4, 2, 3), 3.0, dtype=np.float64)),
    )
    design_grids = (
        SimpleNamespace(
            frequency="monthly",
            model_artifact_fingerprint="d" * 64,
            offsets=design_offsets[0],
        ),
        SimpleNamespace(
            frequency="daily",
            model_artifact_fingerprint="d" * 64,
            offsets=design_offsets[1],
        ),
    )
    production_grids = (
        SimpleNamespace(
            frequency="monthly",
            model_artifact_fingerprint="p" * 64,
            offsets=production_offsets[0],
        ),
        SimpleNamespace(
            frequency="daily",
            model_artifact_fingerprint="p" * 64,
            offsets=production_offsets[1],
        ),
    )
    design_kernels = tuple(
        SimpleNamespace(offset_grid=grid, passed=True) for grid in design_grids
    )
    production_kernels = tuple(
        SimpleNamespace(offset_grid=grid, passed=True) for grid in production_grids
    )
    design_fit = SimpleNamespace(n_states=2)
    production_fit = SimpleNamespace(n_states=2)
    design = cast(
        CanonicalG3DesignStageResult,
        SimpleNamespace(
            block_parents=design_blocks,
            design_fit=design_fit,
            monthly_kernel=design_kernels[0],
            daily_kernel=design_kernels[1],
            monthly_kernel_grid=design_grids[0],
            daily_kernel_grid=design_grids[1],
            core_model_fingerprint="d" * 64,
            decoded_return_pools=SimpleNamespace(
                monthly_states=_immutable(np.asarray([0, 1, 1], dtype=np.int64)),
                daily_states=_immutable(np.asarray([1, 1, 0, 1], dtype=np.int64)),
            ),
        ),
    )
    production = cast(
        CanonicalG3ProductionStageResult,
        SimpleNamespace(
            block_parents=production_blocks,
            production_fit=production_fit,
            monthly_kernel=production_kernels[0],
            daily_kernel=production_kernels[1],
            monthly_kernel_grid=production_grids[0],
            daily_kernel_grid=production_grids[1],
            core_model_fingerprint="p" * 64,
            decoded_pools=SimpleNamespace(
                monthly_states=_immutable(np.asarray([0, 0, 1, 0], dtype=np.int64)),
                daily_states=_immutable(np.asarray([0, 1, 0, 1, 1], dtype=np.int64)),
            ),
        ),
    )
    cores = {
        design_blocks[0]: SimpleNamespace(
            artifact="design",
            frequency="monthly",
            selected_k=2,
            master_seed=master_seed,
            scientific_version=version,
            selected_length=5,
        ),
        design_blocks[1]: SimpleNamespace(
            artifact="design",
            frequency="daily",
            selected_k=2,
            master_seed=master_seed,
            scientific_version=version,
            selected_length=20,
        ),
        production_blocks[0]: SimpleNamespace(
            artifact="production",
            frequency="monthly",
            selected_k=2,
            master_seed=master_seed,
            scientific_version=version,
            selected_length=7,
        ),
        production_blocks[1]: SimpleNamespace(
            artifact="production",
            frequency="daily",
            selected_k=2,
            master_seed=master_seed,
            scientific_version=version,
            selected_length=22,
        ),
    }
    monkeypatch.setattr(
        subject,
        "registered_block_core_identity",
        lambda value: cores[value],
    )

    kernel_identities = {
        id(design_kernels[0]): SimpleNamespace(
            artifact="design",
            frequency="monthly",
            selected_length=5,
            n_states=2,
            master_seed=master_seed,
            scientific_version=version,
        ),
        id(design_kernels[1]): SimpleNamespace(
            artifact="design",
            frequency="daily",
            selected_length=20,
            n_states=2,
            master_seed=master_seed,
            scientific_version=version,
        ),
        id(production_kernels[0]): SimpleNamespace(
            artifact="production",
            frequency="monthly",
            selected_length=7,
            n_states=2,
            master_seed=master_seed,
            scientific_version=version,
        ),
        id(production_kernels[1]): SimpleNamespace(
            artifact="production",
            frequency="daily",
            selected_length=22,
            n_states=2,
            master_seed=master_seed,
            scientific_version=version,
        ),
    }
    monkeypatch.setattr(
        subject,
        "kernel_calibration_execution_identity",
        lambda value: kernel_identities[id(value)],
    )
    captured: dict[str, object] = {}
    actual = cast(ActualBridgeExecution, object())

    def assemble(**kwargs: object) -> ActualBridgeExecution:
        captured.update(kwargs)
        return actual

    monkeypatch.setattr(subject, "assemble_actual_bridge", assemble)

    result = subject._assemble_actual_bridge_from_stages(
        design,
        production,
        master_seed=master_seed,
        scientific_version=version,
    )

    assert result is actual
    assert captured["design_monthly_block_length"] == 5
    assert captured["production_monthly_block_length"] == 7
    assert captured["design_daily_block_length"] == 20
    assert captured["production_daily_block_length"] == 22
    assert cast(np.ndarray, captured["design_monthly_state_counts"]).tolist() == [1, 2]
    assert cast(np.ndarray, captured["production_monthly_state_counts"]).tolist() == [
        3,
        1,
    ]
    assert cast(np.ndarray, captured["design_daily_state_counts"]).tolist() == [1, 3]
    assert cast(np.ndarray, captured["production_daily_state_counts"]).tolist() == [
        2,
        3,
    ]
    assert captured["design_monthly_kernel_offsets"] is design_offsets[0]
    assert captured["production_monthly_kernel_offsets"] is production_offsets[0]
    assert captured["design_daily_kernel_offsets"] is design_offsets[1]
    assert captured["production_daily_kernel_offsets"] is production_offsets[1]


def test_bridge_request_uses_standardized_design_history_and_registered_grid_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _immutable(np.arange(193 * 4, dtype=np.float64).reshape(193, 4))
    parameters = object()
    grids = tuple(
        SimpleNamespace(frequency=frequency)
        for frequency in ("monthly", "daily", "monthly", "daily")
    )
    design = cast(
        CanonicalG3DesignStageResult,
        SimpleNamespace(
            design_fit=SimpleNamespace(
                n_states=2,
                lengths=(193,),
                parameters=parameters,
            ),
            design_features=SimpleNamespace(values=history, lengths=(193,)),
            macro_evidence=SimpleNamespace(macro_block_length=6),
            monthly_kernel_grid=grids[0],
            daily_kernel_grid=grids[1],
        ),
    )
    production = cast(
        CanonicalG3ProductionStageResult,
        SimpleNamespace(
            monthly_kernel_grid=grids[2],
            daily_kernel_grid=grids[3],
        ),
    )
    monkeypatch.setattr(
        subject,
        "registered_macro_stability_execution_identity",
        lambda _value: SimpleNamespace(
            role="design",
            n_states=2,
            master_seed=11,
            scientific_version=1,
        ),
    )
    actual = cast(ActualBridgeExecution, object())
    preflight = _preflight()

    request = subject._canonical_bridge_request(
        design,
        production,
        actual_bridge=actual,
        live_preflight=preflight,
        master_seed=11,
        scientific_version=1,
    )

    assert request.base_preflight is preflight
    assert request.design_parameters is parameters
    assert request.design_standardized_history is history
    assert request.design_continuation.shape == (193,)
    assert not request.design_continuation[0]
    assert request.design_continuation[1:].all()
    assert not request.design_continuation.flags.writeable
    assert request.macro_block_length == 6
    assert request.actual_bridge is actual
    assert request.alpha_offset_grids == grids


def test_bridge_stage_rejects_any_stale_durable_task_in_exact_1_to_19_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "1" * 64
    publications = tuple(
        SimpleNamespace(
            binding=SimpleNamespace(task_id=task_id, run_id=run_id),
            decision=SimpleNamespace(passed=True),
            checkpoint=object(),
            task_result=SimpleNamespace(status="passed", fingerprint=str(index) * 64),
        )
        for index, task_id in enumerate(G3_GATE_ORDER[:19], start=1)
    )

    class _Stage:
        def __init__(
            self,
            values: tuple[object, ...],
            *,
            design_stage: object | None = None,
        ) -> None:
            self.publications = values
            self.design_stage = design_stage

    class _Store:
        def __init__(self) -> None:
            self.durable = {
                item.binding.task_id: SimpleNamespace(
                    fingerprint=item.task_result.fingerprint
                )
                for item in publications
            }

        def read_launch(self, _run_id: str) -> object:
            return SimpleNamespace(
                state="running",
                preflight_fingerprint="a" * 64,
                workers=9,
            )

        def verify_task_results(
            self,
            _run_id: str,
            *,
            require_complete: bool,
        ) -> dict[str, object]:
            assert not require_complete
            return self.durable

    store = _Store()
    lease = SimpleNamespace(
        store=store,
        run_id=run_id,
        active=True,
        preflight_fingerprint="a" * 64,
        workers=9,
    )
    design = _Stage(publications[:10])
    production = _Stage(publications[10:], design_stage=design)
    preflight = SimpleNamespace(workers=9, fingerprint="a" * 64)
    monkeypatch.setattr(subject, "CanonicalG3DesignStageResult", _Stage)
    monkeypatch.setattr(subject, "CanonicalG3ProductionStageResult", _Stage)
    monkeypatch.setattr(subject, "LaunchLease", SimpleNamespace)
    monkeypatch.setattr(
        subject,
        "verify_compatible_canonical_g3_preflights",
        lambda *_args: "a" * 64,
    )

    wrong_parent = _Stage(publications[10:], design_stage=object())
    with pytest.raises(ModelError, match="different exact design stage"):
        subject._verify_stage_inputs(
            cast(CanonicalG3DesignStageResult, design),
            cast(CanonicalG3ProductionStageResult, wrong_parent),
            launch_preflight=cast(CanonicalG3Preflight, preflight),
            live_preflight=cast(CanonicalG3Preflight, preflight),
            run_store=cast(CalibrationRunStore, store),
            lease=cast(LaunchLease, lease),
        )

    later_live = SimpleNamespace(workers=9, fingerprint="b" * 64)
    with pytest.raises(ModelError, match="live-authorized launch preflight"):
        subject._verify_stage_inputs(
            cast(CanonicalG3DesignStageResult, design),
            cast(CanonicalG3ProductionStageResult, production),
            launch_preflight=cast(CanonicalG3Preflight, preflight),
            live_preflight=cast(CanonicalG3Preflight, later_live),
            run_store=cast(CalibrationRunStore, store),
            lease=cast(LaunchLease, lease),
        )

    subject._verify_stage_inputs(
        cast(CanonicalG3DesignStageResult, design),
        cast(CanonicalG3ProductionStageResult, production),
        launch_preflight=cast(CanonicalG3Preflight, preflight),
        live_preflight=cast(CanonicalG3Preflight, preflight),
        run_store=cast(CalibrationRunStore, store),
        lease=cast(LaunchLease, lease),
    )
    store.durable[G3_GATE_ORDER[8]] = SimpleNamespace(fingerprint="f" * 64)
    with pytest.raises(ModelError, match="durable tasks 1--19"):
        subject._verify_stage_inputs(
            cast(CanonicalG3DesignStageResult, design),
            cast(CanonicalG3ProductionStageResult, production),
            launch_preflight=cast(CanonicalG3Preflight, preflight),
            live_preflight=cast(CanonicalG3Preflight, preflight),
            run_store=cast(CalibrationRunStore, store),
            lease=cast(LaunchLease, lease),
        )


def test_false_resource_pass_cannot_reach_tier_a_and_unexpected_error_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    design, production, preflight, store, checkpoints, lease = _runner_inputs(tmp_path)
    _patch_runner_scaffolding(
        monkeypatch,
        events=events,
        actual_bridge=cast(ActualBridgeExecution, object()),
        request=cast(CanonicalBridgeRequest, object()),
        session=object(),
    )
    resource = cast(
        StagedBridgeTaskEvidence,
        SimpleNamespace(task_id="bridge_resource_preflight", passed=False),
    )
    monkeypatch.setattr(
        subject,
        "execute_staged_bridge_resource_benchmark",
        lambda _session: resource,
    )
    monkeypatch.setattr(
        subject,
        "bridge_executor_source_slots",
        lambda *_args, **_kwargs: ScientificTaskSourceSlots(
            "bridge_resource_preflight", (), (), ()
        ),
    )
    monkeypatch.setattr(
        subject,
        "build_scientific_task_binding",
        lambda **_kwargs: SimpleNamespace(
            task_id="bridge_resource_preflight",
            fingerprint="b" * 64,
        ),
    )
    monkeypatch.setattr(
        subject,
        "build_bridge_driver_task_view",
        lambda *_args, **_kwargs: SimpleNamespace(task_id="bridge_resource_preflight"),
    )
    monkeypatch.setattr(
        subject,
        "publish_exact_next_scientific_task",
        lambda **_kwargs: SimpleNamespace(
            binding=SimpleNamespace(task_id="bridge_resource_preflight"),
            decision=SimpleNamespace(passed=False),
            checkpoint=None,
            task_result=SimpleNamespace(status="failed"),
        ),
    )

    def forbidden_tier_a(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Tier A ran after a failed resource gate")

    monkeypatch.setattr(subject, "execute_staged_bridge_tier_a", forbidden_tier_a)
    with pytest.raises(ModelError, match="did not pass and checkpoint"):
        subject.run_canonical_g3_bridge_stage(
            design,
            production,
            launch_preflight=preflight,
            live_preflight=preflight,
            run_store=store,
            checkpoint_store=checkpoints,
            lease=lease,
            master_seed=1,
            scientific_version=1,
        )

    sentinel = RuntimeError("unexpected bridge worker fault")

    def explode(_session: object) -> object:
        raise sentinel

    monkeypatch.setattr(subject, "execute_staged_bridge_resource_benchmark", explode)
    with pytest.raises(RuntimeError) as caught:
        subject.run_canonical_g3_bridge_stage(
            design,
            production,
            launch_preflight=preflight,
            live_preflight=preflight,
            run_store=store,
            checkpoint_store=checkpoints,
            lease=lease,
            master_seed=1,
            scientific_version=1,
        )
    assert caught.value is sentinel


def test_tier_b_loop_has_no_runner_cap_before_registered_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    calls = 0

    def next_look(_session: object, _dgp: object) -> object:
        nonlocal calls
        calls += 1
        events.append(f"look:{calls}")
        return SimpleNamespace(decision="continue" if calls < 8 else "pass_above_tail")

    evidence = cast(
        StagedBridgeTaskEvidence,
        SimpleNamespace(task_id="bridge_tier_b_model_dgp", passed=True),
    )
    monkeypatch.setattr(subject, "execute_staged_bridge_tier_b_next_look", next_look)
    monkeypatch.setattr(
        subject,
        "complete_staged_bridge_tier_b_task",
        lambda *_args: events.append("complete") or evidence,
    )
    monkeypatch.setattr(
        subject,
        "publish_raw_planned_look_prefix",
        lambda **_kwargs: events.append("raw") or object(),
    )
    monkeypatch.setattr(
        subject,
        "bridge_executor_source_slots",
        lambda *_args, **_kwargs: events.append("slots")
        or ScientificTaskSourceSlots("bridge_tier_b_model_dgp", (), (), ()),
    )
    monkeypatch.setattr(
        subject,
        "_build_binding",
        lambda *_args: events.append("bind") or SimpleNamespace(fingerprint="b" * 64),
    )
    monkeypatch.setattr(
        subject,
        "build_bridge_driver_task_view",
        lambda *_args, **_kwargs: events.append("view") or object(),
    )
    monkeypatch.setattr(
        subject,
        "verify_planned_look_binding_view_parity",
        lambda *_args, **_kwargs: events.append("parity") or object(),
    )
    monkeypatch.setattr(
        subject,
        "_publish_pass",
        lambda *_args: events.append("publish") or object(),
    )
    context = cast(
        Any,
        SimpleNamespace(
            run_store=object(),
            checkpoint_store=object(),
            lease=object(),
            live_preflight=object(),
            run_id="1" * 64,
        ),
    )

    subject._execute_tier_b_success(
        context,
        cast(Any, object()),
        dgp="model",
        task_id="bridge_tier_b_model_dgp",
    )

    assert calls == 8
    assert events[-7:] == [
        "complete",
        "raw",
        "slots",
        "bind",
        "view",
        "parity",
        "publish",
    ]


def test_bridge_stage_public_surface_has_no_injected_execution_graph() -> None:
    signature = inspect.signature(subject.run_canonical_g3_bridge_stage)
    assert tuple(signature.parameters) == (
        "design_stage",
        "production_stage",
        "launch_preflight",
        "live_preflight",
        "run_store",
        "checkpoint_store",
        "lease",
        "master_seed",
        "scientific_version",
    )
