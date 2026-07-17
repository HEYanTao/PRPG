from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import prpg.model.g3_success_runner as subject
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_task_publication import ScientificTaskSourceSlots
from prpg.model.input import CalibrationInput, CalibrationSlice
from prpg.model.run_store import CalibrationRunStore, LaunchLease


def _slice(role: str) -> CalibrationSlice:
    return CalibrationSlice(
        role=cast(Any, role),
        monthly_returns=np.zeros((1, 3), dtype=np.float64),
        daily_returns=np.zeros((1, 3), dtype=np.float64),
        macro_features=np.zeros((1, 4), dtype=np.float64),
        monthly_period_ordinals=np.asarray([0], dtype=np.int64),
        daily_session_ns=np.asarray([0], dtype=np.int64),
        monthly_continuation=np.asarray([False], dtype=np.bool_),
        daily_continuation=np.asarray([False], dtype=np.bool_),
    )


def _input() -> CalibrationInput:
    return CalibrationInput(
        fingerprint="a" * 64,
        provenance=cast(Any, None),
        geometry=cast(Any, None),
        full=_slice("full"),
        design=_slice("design"),
        holdout=_slice("holdout"),
    )


def _preflight() -> CanonicalG3Preflight:
    return cast(
        CanonicalG3Preflight,
        SimpleNamespace(
            config_fingerprint="b" * 64,
            processed_data_fingerprint="c" * 64,
            calibration_input_fingerprint="a" * 64,
            fingerprint="d" * 64,
            task_registry_fingerprint="e" * 64,
            planned_look_registry_fingerprint="f" * 64,
            workers=9,
        ),
    )


class _FreshRunStore:
    def verify_task_results(
        self,
        _run_id: str,
        *,
        require_complete: bool,
    ) -> dict[str, object]:
        assert require_complete is False
        return {}


def test_closed_design_stage_orders_bindings_looks_parity_and_publications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    objects: dict[str, object] = {}
    calibration_input = _input()
    launch = _preflight()
    live = _preflight()
    store = cast(CalibrationRunStore, _FreshRunStore())
    checkpoint_store = cast(CalibrationCheckpointStore, object())
    lease = cast(LaunchLease, SimpleNamespace(run_id="1" * 64))

    monkeypatch.setattr(
        subject,
        "verify_compatible_canonical_g3_preflights",
        lambda *_args: events.append("preflight"),
    )

    def slots(task_id: str) -> ScientificTaskSourceSlots:
        events.append(f"slots:{task_id}")
        return ScientificTaskSourceSlots(task_id, (), (), ())

    monkeypatch.setattr(
        subject,
        "input_integrity_executor_source_slots",
        lambda *_args, **_kwargs: slots("input_integrity"),
    )
    monkeypatch.setattr(
        subject,
        "candidate_viability_executor_source_slots",
        lambda *_args, **_kwargs: slots("design_candidate_viability"),
    )
    monkeypatch.setattr(
        subject,
        "candidate_selection_executor_source_slots",
        lambda *_args, **_kwargs: slots("design_predictive_selection"),
    )
    monkeypatch.setattr(
        subject,
        "sealed_holdout_executor_source_slots",
        lambda *_args, **_kwargs: slots("sealed_holdout_veto"),
    )
    monkeypatch.setattr(
        subject,
        "macro_stability_executor_source_slots",
        lambda *_args, **kwargs: slots(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "adequacy_cell_executor_source_slots",
        lambda *_args, **kwargs: slots(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "block_calibration_executor_source_slots",
        lambda *_args, **kwargs: slots(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "fragmentation_executor_source_slots",
        lambda *_args, **kwargs: slots(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "dependence_executor_source_slots",
        lambda *_args, **kwargs: slots(kwargs["task_id"]),
    )

    def bind(**kwargs: object) -> object:
        task_id = cast(str, kwargs["task_id"])
        events.append(f"bind:{task_id}")
        return SimpleNamespace(task_id=task_id, fingerprint=task_id[0] * 64)

    monkeypatch.setattr(subject, "build_scientific_task_binding", bind)

    def view(task_id: str, **fields: object) -> object:
        events.append(f"view:{task_id}")
        result = SimpleNamespace(task_id=task_id, **fields)
        objects[task_id] = result
        return result

    monkeypatch.setattr(
        subject,
        "build_input_integrity_task_view",
        lambda *_args, **_kwargs: view("input_integrity"),
    )
    monkeypatch.setattr(
        subject,
        "build_candidate_viability_task_view",
        lambda *_args, **_kwargs: view("design_candidate_viability"),
    )
    monkeypatch.setattr(
        subject,
        "build_candidate_selection_task_view",
        lambda *_args, **_kwargs: view(
            "design_predictive_selection",
            selected_n_states=2,
        ),
    )
    monkeypatch.setattr(
        subject,
        "build_sealed_holdout_veto_view",
        lambda *_args, **_kwargs: view("sealed_holdout_veto"),
    )
    monkeypatch.setattr(
        subject,
        "build_macro_stability_task_view",
        lambda *_args, **kwargs: view(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "build_latent_adequacy_task_view",
        lambda *_args, **kwargs: view(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "build_refitted_adequacy_task_view",
        lambda *_args, **kwargs: view(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "build_block_calibration_task_view",
        lambda *_args, **kwargs: view(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "build_fragmentation_task_view",
        lambda *_args, **kwargs: view(kwargs["task_id"]),
    )
    monkeypatch.setattr(
        subject,
        "build_registered_dependence_task_view",
        lambda *_args, **kwargs: view(kwargs["task_id"]),
    )

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

    fixed_point = object()
    final_viability = object()
    final_selection = object()
    macro_evidence = object()
    decoded_pools = object()
    final_suite = SimpleNamespace(
        rows=(
            SimpleNamespace(
                n_states=2,
                macro_stability=macro_evidence,
                decoded_return_pools=decoded_pools,
            ),
        )
    )
    monkeypatch.setattr(
        subject,
        "execute_registered_design_fixed_point",
        lambda *_args, **kwargs: (
            events.append(f"fixed_point_workers:{kwargs['workers']}") or fixed_point
        ),
    )
    monkeypatch.setattr(
        subject,
        "registered_design_fixed_point_sources",
        lambda _value: (((final_suite, final_viability, final_selection),), None, None),
    )
    monkeypatch.setattr(
        subject,
        "registered_design_candidate_suite_identity",
        lambda _value: events.append("suite_identity"),
    )

    fit = object()
    holdout_execution = SimpleNamespace(failure_code=None)
    holdout_evidence = object()
    monkeypatch.setattr(
        subject,
        "execute_registered_sealed_holdout",
        lambda *_args: holdout_execution,
    )
    monkeypatch.setattr(
        subject,
        "is_registered_sealed_holdout_evaluation",
        lambda value: value is holdout_execution,
    )
    monkeypatch.setattr(
        subject,
        "is_registered_sealed_holdout_invalid_benchmark",
        lambda _value: False,
    )
    monkeypatch.setattr(
        subject,
        "registered_sealed_holdout_evaluation_sources",
        lambda _value: (
            objects["design_predictive_selection"],
            fit,
            calibration_input.design,
            calibration_input.holdout,
            holdout_evidence,
        ),
    )
    monkeypatch.setattr(subject, "selected_design_fit_from_view", lambda _view: fit)
    features = object()
    monkeypatch.setattr(subject, "_prepare_design_features", lambda _input: features)

    latent = object()
    refitted = object()
    monkeypatch.setattr(
        subject,
        "run_registered_latent_cell",
        lambda *_args, **_kwargs: latent,
    )

    def refit(**kwargs: object) -> object:
        events.append(f"refit_workers:{kwargs['workers']}")
        return refitted

    monkeypatch.setattr(subject, "run_registered_refitted_adequacy_cell", refit)

    block_parents = (object(), object())
    monkeypatch.setattr(
        subject,
        "registered_final_selected_blocks",
        lambda _selection: block_parents,
    )
    replays = (object(), object())
    replay_index = iter(replays)
    monkeypatch.setattr(
        subject,
        "registered_block_replay_source",
        lambda _parent: next(replay_index),
    )
    grids = (object(), object())
    kernels = (
        SimpleNamespace(offset_grid=grids[0]),
        SimpleNamespace(offset_grid=grids[1]),
    )

    def kernel(**kwargs: object) -> object:
        frequency = cast(str, kwargs["frequency"])
        assert kwargs["parent_model_fingerprint"] == "9" * 64
        assert kwargs["core_model_fingerprint"] == "8" * 64
        events.append(f"kernel:{frequency}:workers:{kwargs['workers']}")
        return kernels[0] if frequency == "monthly" else kernels[1]

    monkeypatch.setattr(subject, "_run_design_kernel", kernel)
    monkeypatch.setattr(
        subject,
        "gaussian_hmm_fit_fingerprint",
        lambda _fit: "9" * 64,
    )
    dependence_references = (object(), object())
    reference_index = iter(dependence_references)
    monkeypatch.setattr(
        subject,
        "fit_dependence_reference",
        lambda *_args, **_kwargs: next(reference_index),
    )
    monkeypatch.setattr(
        subject,
        "scientific_core_model_fingerprint",
        lambda **_kwargs: "8" * 64,
    )
    transition = object()
    monkeypatch.setattr(
        subject,
        "aggregate_transition_bootstrap",
        lambda *_args: transition,
    )

    look_prefix = object()

    def looks(**_kwargs: object) -> object:
        events.append("looks:design_block_calibration")
        return look_prefix

    monkeypatch.setattr(subject, "publish_raw_planned_look_prefix", looks)
    look_parity = object()

    def parity(*_args: object, **_kwargs: object) -> object:
        events.append("parity:design_block_calibration")
        return look_parity

    monkeypatch.setattr(subject, "verify_planned_look_binding_view_parity", parity)

    dependence_cells = (object(), object())
    dependence_execution = object()
    monkeypatch.setattr(
        subject,
        "_design_dependence_cells",
        lambda *_args, **_kwargs: dependence_cells,
    )
    monkeypatch.setattr(
        subject,
        "execute_registered_artifact_dependence_role",
        lambda *_args, **_kwargs: dependence_execution,
    )

    result = subject.run_canonical_g3_design_stage(
        calibration_input,
        launch_preflight=launch,
        live_preflight=live,
        run_store=store,
        checkpoint_store=checkpoint_store,
        lease=lease,
        master_seed=123,
        scientific_version=1,
    )

    expected = G3_GATE_ORDER[:10]
    assert tuple(item.binding.task_id for item in result.publications) == expected
    assert [
        event.removeprefix("publish:")
        for event in events
        if event.startswith("publish:")
    ] == list(expected)
    for task_id in expected:
        assert events.index(f"slots:{task_id}") < events.index(f"bind:{task_id}")
        assert events.index(f"bind:{task_id}") < events.index(f"view:{task_id}")
        assert events.index(f"view:{task_id}") < events.index(f"publish:{task_id}")
    assert events.index("looks:design_block_calibration") < events.index(
        "bind:design_block_calibration"
    )
    assert events.index("view:design_block_calibration") < events.index(
        "parity:design_block_calibration"
    )
    assert events.index("parity:design_block_calibration") < events.index(
        "publish:design_block_calibration"
    )
    assert "fixed_point_workers:9" in events
    assert "refit_workers:9" in events
    assert "kernel:monthly:workers:9" in events
    assert "kernel:daily:workers:9" in events
    assert result.block_planned_look_prefix is look_prefix
    assert result.block_planned_look_parity is look_parity
    assert result.monthly_kernel_grid is grids[0]
    assert result.daily_kernel_grid is grids[1]
    assert result.decoded_return_pools is decoded_pools
    assert result.monthly_dependence_reference is dependence_references[0]
    assert result.daily_dependence_reference is dependence_references[1]
    assert result.core_model_fingerprint == "8" * 64


def test_design_stage_public_surface_has_no_injected_execution_graph() -> None:
    signature = inspect.signature(subject.run_canonical_g3_design_stage)
    assert tuple(signature.parameters) == (
        "calibration_input",
        "launch_preflight",
        "live_preflight",
        "run_store",
        "checkpoint_store",
        "lease",
        "master_seed",
        "scientific_version",
    )


def test_calendar_year_projection_is_exact_and_immutable() -> None:
    monthly = subject._monthly_calendar_years(
        np.asarray([-1, 0, 11, 12, 647], dtype=np.int64)
    )
    daily = subject._daily_calendar_years(
        np.asarray(
            [
                np.datetime64("1999-12-31", "ns").astype(np.int64),
                np.datetime64("2000-01-01", "ns").astype(np.int64),
                np.datetime64("2024-07-14", "ns").astype(np.int64),
            ],
            dtype=np.int64,
        )
    )

    assert monthly.tolist() == [1969, 1970, 1970, 1971, 2023]
    assert daily.tolist() == [1999, 2000, 2024]
    assert not monthly.flags.writeable
    assert not daily.flags.writeable
