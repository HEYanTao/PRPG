"""Focused orchestration tests for canonical G3 tasks 11 through 19."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import prpg.model.g3_success_production_stage as stage_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_success_production_stage import (
    CanonicalG3ProductionStageResult,
    run_canonical_g3_production_stage,
)
from prpg.model.g3_success_runner import CanonicalG3DesignStageResult
from prpg.model.hmm import summarize_state_path
from prpg.model.input import CalibrationInput
from prpg.model.regime_policy import OwnerFixedK4RecurringHistoryFailure

_RUN = "a" * 64
_TASKS = (
    "production_fit_same_k",
    "production_macro_stability",
    "production_latent_correctness",
    "production_refitted_adequacy",
    "production_block_calibration",
    "production_fragmentation",
    "production_dependence",
    "four_cell_adequacy_holm",
    "four_cell_dependence_holm",
)


@dataclass(frozen=True, slots=True)
class _Slots:
    task_id: str


def _publication(task_id: str) -> object:
    return SimpleNamespace(
        binding=SimpleNamespace(task_id=task_id, run_id=_RUN),
        decision=SimpleNamespace(passed=True),
        checkpoint=object(),
        task_result=SimpleNamespace(status="passed", fingerprint=task_id),
    )


def _full_slice() -> object:
    return SimpleNamespace(
        macro_features=np.zeros((2, 4), dtype=np.float64),
        monthly_period_ordinals=np.asarray([600, 601], dtype=np.int64),
        monthly_continuation=np.asarray([False, True]),
        monthly_returns=np.zeros((2, 3), dtype=np.float64),
        daily_session_ns=np.asarray([0, 1, 2, 3], dtype=np.int64),
        daily_continuation=np.asarray([False, True, True, True]),
        daily_returns=np.zeros((4, 3), dtype=np.float64),
    )


def test_closed_stage_executes_11_through_19_in_order_with_exact_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    worker_calls: dict[str, list[int]] = {}
    selection = object()
    design_fit = object()
    design_latent = object()
    design_refitted = object()
    design_dependence = object()
    design_stage = SimpleNamespace(
        candidate_selection=selection,
        design_fit=design_fit,
        latent_correctness=design_latent,
        refitted_adequacy=design_refitted,
        dependence=design_dependence,
    )
    full = _full_slice()
    calibration_input = SimpleNamespace(full=full)
    features = object()
    production_execution = object()
    production_fit = SimpleNamespace(n_states=4)
    production_view = object()
    preliminary_outcome = object()
    preliminary = SimpleNamespace(
        monthly=SimpleNamespace(combined=SimpleNamespace(starting_length=4)),
        daily=SimpleNamespace(combined=SimpleNamespace(starting_length=20)),
        macro=SimpleNamespace(combined=SimpleNamespace(starting_length=3)),
    )
    macro_materialization = object()
    macro_evidence = object()
    macro_view = object()
    latent_evidence = object()
    latent_view = object()
    refitted_evidence = object()
    refitted_view = object()
    decoded_pools = SimpleNamespace(
        monthly_states=np.asarray([0, 1]),
        daily_states=np.asarray([0, 0, 1, 1]),
        monthly_diagnostics=summarize_state_path(
            [0, 1, 2, 3, 0, 1, 2, 3],
            4,
        ),
        daily_diagnostics=summarize_state_path(
            [0, 1, 2, 3, 0, 1, 2, 3],
            4,
        ),
    )
    monthly_materialization = object()
    daily_materialization = object()
    monthly_block = object()
    daily_block = object()
    monthly_replay = SimpleNamespace(model=production_fit)
    daily_replay = SimpleNamespace(model=production_fit)
    monthly_grid = object()
    daily_grid = object()
    monthly_kernel = SimpleNamespace(offset_grid=monthly_grid)
    daily_kernel = SimpleNamespace(offset_grid=daily_grid)
    transition = object()
    look_prefix = object()
    look_parity = object()
    block_view = object()
    fragmentation_view = object()
    dependence_execution = object()
    dependence_view = object()
    adequacy_holm_view = object()
    dependence_holm_view = object()
    monthly_reference = object()
    daily_reference = object()
    core_fingerprint = "c" * 64

    monkeypatch.setattr(stage_module, "_verify_stage_inputs", lambda *_a, **_k: None)
    monkeypatch.setattr(
        stage_module,
        "_prepare_production_features",
        lambda _value: features,
    )

    def production_fit_executor(*args: object, **kwargs: object) -> object:
        assert args == (selection, features)
        worker_calls.setdefault("fit", []).append(cast(int, kwargs["workers"]))
        return production_execution

    monkeypatch.setattr(
        stage_module,
        "execute_registered_production_fit_same_k",
        production_fit_executor,
    )
    monkeypatch.setattr(
        stage_module,
        "is_registered_production_fit_same_k",
        lambda value: value is production_execution,
    )
    monkeypatch.setattr(
        stage_module,
        "registered_production_fit_same_k_sources",
        lambda _value: (selection, design_fit, features, production_fit),
    )
    monkeypatch.setattr(
        stage_module,
        "execute_registered_preliminary_calibrations",
        lambda source, **kwargs: (
            preliminary_outcome
            if source is full and kwargs == {"role": "production"}
            else pytest.fail("wrong preliminary source")
        ),
    )
    monkeypatch.setattr(
        stage_module,
        "registered_preliminary_calibrations",
        lambda value: (
            preliminary
            if value is preliminary_outcome
            else pytest.fail("wrong preliminary outcome")
        ),
    )
    monkeypatch.setattr(
        stage_module,
        "materialize_registered_macro_bootstrap",
        lambda *_a, **_k: macro_materialization,
    )

    def macro_executor(*_args: object, **kwargs: object) -> object:
        worker_calls.setdefault("macro", []).append(cast(int, kwargs["workers"]))
        assert kwargs["source_materialization"] is macro_materialization
        assert kwargs["main_fit"] is production_fit
        return macro_evidence

    monkeypatch.setattr(
        stage_module,
        "execute_registered_macro_stability",
        macro_executor,
    )

    def latent_executor(*args: object, **kwargs: object) -> object:
        assert args == (production_fit,)
        assert "workers" not in kwargs
        events.append("latent_serial")
        return latent_evidence

    monkeypatch.setattr(stage_module, "run_registered_latent_cell", latent_executor)

    def refit_executor(**kwargs: object) -> object:
        worker_calls.setdefault("refit", []).append(cast(int, kwargs["workers"]))
        assert kwargs["original_fit"] is production_fit
        assert kwargs["artifact_features"] is features
        return refitted_evidence

    monkeypatch.setattr(
        stage_module,
        "run_registered_refitted_adequacy_cell",
        refit_executor,
    )
    monkeypatch.setattr(
        stage_module,
        "decoded_return_pools",
        lambda fit, source: (
            decoded_pools
            if fit is production_fit and source is full
            else pytest.fail("wrong decoded pool source")
        ),
    )
    materializations = iter((monthly_materialization, daily_materialization))
    monkeypatch.setattr(
        stage_module,
        "materialize_registered_block_uniforms",
        lambda **_kwargs: next(materializations),
    )

    def block_outcome(materialization: object, **kwargs: object) -> object:
        assert kwargs["model"] is production_fit
        if materialization is monthly_materialization:
            assert kwargs["starting_length"] == 4
            return monthly_block
        assert materialization is daily_materialization
        assert kwargs["starting_length"] == 20
        return daily_block

    monkeypatch.setattr(
        stage_module,
        "run_registered_materialized_block_calibration_outcome",
        block_outcome,
    )
    monkeypatch.setattr(
        stage_module,
        "_require_successful_block_pair",
        lambda monthly, daily: (
            (monthly_block, daily_block)
            if monthly is monthly_block and daily is daily_block
            else pytest.fail("wrong block pair")
        ),
    )
    monkeypatch.setattr(
        stage_module,
        "registered_block_replay_source",
        lambda value: monthly_replay if value is monthly_block else daily_replay,
    )
    monkeypatch.setattr(
        stage_module,
        "gaussian_hmm_fit_fingerprint",
        lambda value: "p" * 64 if value is production_fit else pytest.fail("wrong fit"),
    )
    monkeypatch.setattr(
        stage_module,
        "fit_dependence_reference",
        lambda _returns, _continuation, *, frequency: (
            monthly_reference if frequency == "monthly" else daily_reference
        ),
    )

    def core(**kwargs: object) -> str:
        assert kwargs["fit"] is production_fit
        assert kwargs["pools"] is decoded_pools
        assert kwargs["monthly_dependence_reference"] is monthly_reference
        assert kwargs["daily_dependence_reference"] is daily_reference
        return core_fingerprint

    monkeypatch.setattr(stage_module, "scientific_core_model_fingerprint", core)

    def kernel(**kwargs: object) -> object:
        worker_calls.setdefault("kernel", []).append(cast(int, kwargs["workers"]))
        assert kwargs["core_model_fingerprint"] == core_fingerprint
        return monthly_kernel if kwargs["frequency"] == "monthly" else daily_kernel

    monkeypatch.setattr(stage_module, "_run_production_kernel", kernel)
    monkeypatch.setattr(
        stage_module,
        "aggregate_transition_bootstrap",
        lambda fit, macro: (
            transition
            if fit is production_fit and macro is macro_evidence
            else pytest.fail("wrong transition parents")
        ),
    )

    def raw_looks(**kwargs: object) -> object:
        events.append("raw_look_prefix")
        assert kwargs["task_id"] == "production_block_calibration"
        return look_prefix

    monkeypatch.setattr(stage_module, "publish_raw_planned_look_prefix", raw_looks)

    def parity(prefix: object, **kwargs: object) -> object:
        events.append("planned_look_parity")
        assert prefix is look_prefix
        assert kwargs["final_view"] is block_view
        return look_parity

    monkeypatch.setattr(
        stage_module,
        "verify_planned_look_binding_view_parity",
        parity,
    )
    monkeypatch.setattr(
        stage_module,
        "_production_dependence_cells",
        lambda *_a, **_k: (object(), object()),
    )
    monkeypatch.setattr(
        stage_module,
        "execute_registered_artifact_dependence_role",
        lambda *_a, **kwargs: (
            dependence_execution
            if kwargs["fragmentation_parent"] is fragmentation_view
            else pytest.fail("wrong fragmentation parent")
        ),
    )

    fixed_slots = {
        "production_fit_executor_source_slots": "production_fit_same_k",
        "macro_stability_executor_source_slots": "production_macro_stability",
        "block_calibration_executor_source_slots": "production_block_calibration",
        "fragmentation_executor_source_slots": "production_fragmentation",
        "dependence_executor_source_slots": "production_dependence",
        "adequacy_holm_executor_source_slots": "four_cell_adequacy_holm",
        "dependence_holm_executor_source_slots": "four_cell_dependence_holm",
    }
    for name, task_id in fixed_slots.items():
        monkeypatch.setattr(
            stage_module,
            name,
            lambda *_args, _task_id=task_id, **_kwargs: _Slots(_task_id),
        )
    monkeypatch.setattr(
        stage_module,
        "adequacy_cell_executor_source_slots",
        lambda *_args, **kwargs: _Slots(cast(str, kwargs["task_id"])),
    )

    def build_binding(_context: object, slots: _Slots) -> object:
        events.append(f"bind:{slots.task_id}")
        return SimpleNamespace(
            task_id=slots.task_id,
            run_id=_RUN,
            fingerprint=(slots.task_id[0] * 64),
        )

    monkeypatch.setattr(stage_module, "_build_binding", build_binding)

    views = {
        "build_production_fit_same_k_view": production_view,
        "build_macro_stability_task_view": macro_view,
        "build_latent_adequacy_task_view": latent_view,
        "build_refitted_adequacy_task_view": refitted_view,
        "build_block_calibration_task_view": block_view,
        "build_fragmentation_task_view": fragmentation_view,
        "build_registered_dependence_task_view": dependence_view,
    }
    for name, value in views.items():
        monkeypatch.setattr(
            stage_module,
            name,
            lambda *_args, _value=value, **_kwargs: _value,
        )

    def adequacy_holm(**kwargs: object) -> object:
        assert tuple(
            kwargs[name]
            for name in (
                "design_latent",
                "production_latent",
                "design_refitted",
                "production_refitted",
            )
        ) == (design_latent, latent_view, design_refitted, refitted_view)
        return adequacy_holm_view

    monkeypatch.setattr(
        stage_module,
        "assemble_four_cell_adequacy_holm_view",
        adequacy_holm,
    )

    def dependence_holm(
        design: object, production: object, **_kwargs: object
    ) -> object:
        assert (design, production) == (design_dependence, dependence_view)
        return dependence_holm_view

    monkeypatch.setattr(
        stage_module,
        "assemble_four_cell_dependence_holm_view",
        dependence_holm,
    )

    def publish(_context: object, binding: object, _source: object) -> object:
        task_id = cast(Any, binding).task_id
        events.append(f"publish:{task_id}")
        return _publication(task_id)

    monkeypatch.setattr(stage_module, "_publish_pass", publish)

    result = run_canonical_g3_production_stage(
        cast(CanonicalG3DesignStageResult, design_stage),
        cast(CalibrationInput, calibration_input),
        launch_preflight=cast(Any, object()),
        live_preflight=cast(Any, SimpleNamespace(workers=9)),
        run_store=cast(Any, object()),
        checkpoint_store=cast(Any, object()),
        lease=cast(Any, SimpleNamespace(run_id=_RUN)),
        master_seed=17,
        scientific_version=1,
    )

    assert isinstance(result, CanonicalG3ProductionStageResult)
    assert tuple(item.binding.task_id for item in result.publications) == _TASKS
    assert worker_calls == {
        "fit": [9],
        "macro": [9],
        "refit": [9],
        "kernel": [9, 9],
    }
    assert result.design_stage is design_stage
    assert result.production_fit_execution is production_execution
    assert result.production_fit is production_fit
    assert result.production_features is features
    assert result.preliminary_outcome is preliminary_outcome
    assert result.macro_materialization is macro_materialization
    assert result.block_parents == (monthly_block, daily_block)
    assert result.core_model_fingerprint == core_fingerprint
    assert result.monthly_dependence_reference is monthly_reference
    assert result.daily_dependence_reference is daily_reference
    assert result.block_planned_look_parity is look_parity
    assert result.adequacy_holm is adequacy_holm_view
    assert result.dependence_holm is dependence_holm_view
    assert not result.generation_authorized
    assert events.index("raw_look_prefix") < events.index("planned_look_parity")
    assert events.index("planned_look_parity") < events.index(
        "publish:production_block_calibration"
    )


def test_production_recurring_history_support_fails_before_block_work() -> None:
    pools = SimpleNamespace(
        monthly_diagnostics=summarize_state_path(
            [0, 1, 2, 3, 0, 2, 3],
            4,
        ),
        daily_diagnostics=summarize_state_path(
            [0, 1, 2, 3, 0, 1, 3],
            4,
        ),
    )

    with pytest.raises(OwnerFixedK4RecurringHistoryFailure) as caught:
        stage_module._require_production_recurring_history_support(cast(Any, pools))

    assert caught.value.report.rejection_reasons == (
        "state_1_monthly_observed_episodes_below_two",
        "state_2_daily_runs_below_two",
    )


def test_production_kernel_binds_scientific_core_not_fit_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    replay = SimpleNamespace(
        model=model,
        core_identity=SimpleNamespace(
            frequency="monthly",
            artifact="production",
            parent_model_fingerprint="p" * 64,
            selected_length=4,
        ),
        historical_source_states=np.asarray([0, 1]),
        source_continuation_links=np.asarray([True]),
    )
    source = SimpleNamespace(monthly_returns=np.zeros((2, 3)))
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(
        stage_module,
        "scientific_array_fingerprint",
        lambda *_args, **_kwargs: "t" * 64,
    )

    def kernel(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        stage_module,
        "run_registered_kernel_calibration_cell",
        kernel,
    )
    result = stage_module._run_production_kernel(
        replay=cast(Any, replay),
        source_slice=cast(Any, source),
        frequency="monthly",
        model=cast(Any, model),
        parent_model_fingerprint="p" * 64,
        core_model_fingerprint="c" * 64,
        master_seed=17,
        scientific_version=1,
        workers=9,
    )

    assert result is sentinel
    assert captured["model_artifact_fingerprint"] == "c" * 64
    assert captured["target_history_fingerprint"] == "t" * 64
    assert captured["workers"] == 9
    assert captured["strict_canonical"] is True


def test_stage_rejects_cross_stage_seed_or_version_before_production_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = object.__new__(CanonicalG3DesignStageResult)
    object.__setattr__(design, "fixed_point", object())
    calibration_input = object.__new__(CalibrationInput)
    object.__setattr__(calibration_input, "fingerprint", "f" * 64)
    monkeypatch.setattr(
        stage_module,
        "registered_design_fixed_point_identity",
        lambda _value: SimpleNamespace(
            master_seed=17,
            scientific_version=1,
            passed=True,
        ),
    )

    with pytest.raises(ModelError, match="seed/version"):
        stage_module._verify_stage_inputs(
            design,
            calibration_input,
            launch_preflight=cast(Any, object()),
            live_preflight=cast(Any, object()),
            run_store=cast(Any, object()),
            lease=cast(Any, object()),
            master_seed=18,
            scientific_version=1,
        )


@pytest.mark.parametrize(
    "error",
    (
        ModelError("model fault"),
        IntegrityError("integrity fault"),
        RuntimeError("runtime fault"),
        OSError("filesystem fault"),
        MemoryError("memory fault"),
    ),
    ids=("model", "integrity", "runtime", "filesystem", "memory"),
)
def test_unexpected_stage_faults_escape_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(stage_module, "_verify_stage_inputs", lambda *_a, **_k: None)

    def fail(_value: object) -> object:
        raise error

    monkeypatch.setattr(stage_module, "_prepare_production_features", fail)
    with pytest.raises(type(error), match=str(error)):
        run_canonical_g3_production_stage(
            cast(Any, object()),
            cast(Any, object()),
            launch_preflight=cast(Any, object()),
            live_preflight=cast(Any, SimpleNamespace(workers=9)),
            run_store=cast(Any, object()),
            checkpoint_store=cast(Any, object()),
            lease=cast(Any, SimpleNamespace(run_id=_RUN)),
            master_seed=17,
            scientific_version=1,
        )


def test_public_production_stage_has_no_callbacks_maps_or_reduced_geometry() -> None:
    parameters = inspect.signature(run_canonical_g3_production_stage).parameters

    assert tuple(parameters) == (
        "design_stage",
        "calibration_input",
        "launch_preflight",
        "live_preflight",
        "run_store",
        "checkpoint_store",
        "lease",
        "master_seed",
        "scientific_version",
    )
    assert all(
        forbidden not in parameters
        for forbidden in (
            "handler_map",
            "callbacks",
            "fit_function",
            "history_count",
            "strict_canonical",
            "recover",
        )
    )
