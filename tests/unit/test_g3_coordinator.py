from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import prpg.model.g3_coordinator as g3_coordinator
from prpg.errors import IntegrityError, ModelError, NotImplementedStageError
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_coordinator import (
    G3PlannedLookDecision,
    G3SmokeHandlerSet,
    G3SmokeTaskContext,
    G3SmokeTaskOutcome,
    coordinate_g3_canonical_launch,
    coordinate_g3_smoke_launch,
    finalize_g3_canonical_artifacts,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    failure_propagation,
)
from prpg.model.run_store import (
    CalibrationRunStore,
    LaunchLease,
    default_run_identity,
)


def _planned_store(
    tmp_path: Path,
) -> tuple[CalibrationRunStore, str, str]:
    store = CalibrationRunStore(tmp_path / "runs")
    identity = default_run_identity(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="e" * 64,
        dependency_lock_fingerprint="f" * 64,
        workers=9,
    )
    run_id = store.initialize(identity).run_id
    preflight = "d" * 64
    store.plan_launch(run_id, preflight_fingerprint=preflight, workers=9)
    return store, run_id, preflight


def _looks_for(
    task_id: str,
    *,
    terminal: str = "pass",
    two_looks: bool = False,
    marker: str = "stable",
) -> tuple[G3PlannedLookDecision, ...]:
    result: list[G3PlannedLookDecision] = []
    for spec in G3_PLANNED_LOOK_REGISTRY:
        if spec.owner_task_id != task_id:
            continue
        if two_looks:
            result.append(
                G3PlannedLookDecision(
                    spec.family_id,
                    1,
                    spec.sample_counts[0],
                    "continue",
                    {"marker": marker, "precision_passed": False},
                )
            )
            result.append(
                G3PlannedLookDecision(
                    spec.family_id,
                    2,
                    spec.sample_counts[1],
                    terminal,  # type: ignore[arg-type]
                    {"marker": marker, "precision_passed": terminal == "pass"},
                )
            )
        else:
            result.append(
                G3PlannedLookDecision(
                    spec.family_id,
                    1,
                    spec.sample_counts[0],
                    terminal,  # type: ignore[arg-type]
                    {"marker": marker, "precision_passed": terminal == "pass"},
                )
            )
    return tuple(result)


def _passing_handlers(
    calls: list[str],
    *,
    selected_k: int = 3,
    overrides: dict[str, Callable[[G3SmokeTaskContext], G3SmokeTaskOutcome]]
    | None = None,
    two_look_task: str | None = None,
    look_marker: str = "stable",
) -> G3SmokeHandlerSet:
    replacements = overrides or {}
    handlers: dict[str, Callable[[G3SmokeTaskContext], G3SmokeTaskOutcome]] = {}
    for task_id in G3_GATE_ORDER:

        def handler(
            context: G3SmokeTaskContext,
            *,
            current: str = task_id,
        ) -> G3SmokeTaskOutcome:
            calls.append(current)
            replacement = replacements.get(current)
            if replacement is not None:
                return replacement(context)
            state_count = (
                selected_k
                if current == "design_predictive_selection"
                else (context.selected_n_states)
            )
            return G3SmokeTaskOutcome(
                status="passed",
                evidence={
                    "metric": -0.0,
                    "task_id": current,
                    "values": [1, None],
                    "verified": True,
                },
                selected_n_states=state_count,
                planned_looks=_looks_for(
                    current,
                    two_looks=current == two_look_task,
                    marker=look_marker,
                ),
            )

        handlers[task_id] = handler
    return G3SmokeHandlerSet(handlers)


def _start(
    store: CalibrationRunStore,
    run_id: str,
    preflight: str,
    *,
    resume: bool = False,
) -> LaunchLease:
    return store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
        resume=resume,
    )


def test_smoke_coordinator_executes_exact_graph_and_never_authorizes_generation(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    calls: list[str] = []
    with _start(store, run_id, preflight) as lease:
        result = coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers(calls),
        )

        assert calls == list(G3_GATE_ORDER)
        assert tuple(item.task_id for item in result.task_results) == G3_GATE_ORDER
        assert all(item.status == "passed" for item in result.task_results)
        assert result.failed_task_ids == ()
        assert result.blocked_task_ids == ()
        assert result.selected_n_states == 3
        assert result.all_tasks_passed
        assert not result.generation_authorized
        assert result.execution_mode == "smoke"
        assert store.read_launch(run_id).state == "completed"
        looks = store.verify_planned_look_receipts(
            run_id,
            require_terminal=True,
            task_results=store.verify_task_results(run_id, require_complete=True),
        )
        assert tuple(looks) == tuple(
            item.family_id for item in G3_PLANNED_LOOK_REGISTRY
        )
        assert all(receipts[-1].decision == "pass" for receipts in looks.values())


def test_failure_propagation_is_derived_and_independent_branches_still_run(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    calls: list[str] = []

    def fail_holdout(context: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return G3SmokeTaskOutcome(
            "failed",
            {"holdout_veto": True},
            context.selected_n_states,
        )

    handlers = _passing_handlers(
        calls,
        overrides={"sealed_holdout_veto": fail_holdout},
    )
    with _start(store, run_id, preflight) as lease:
        result = coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=handlers,
        )

    expected_blocked = failure_propagation(("sealed_holdout_veto",))
    assert result.failed_task_ids == ("sealed_holdout_veto",)
    assert result.blocked_task_ids == expected_blocked
    assert "design_dependence" in calls
    assert "production_fit_same_k" not in calls
    assert all(task_id not in calls for task_id in expected_blocked)
    blocked = store.read_task_result(run_id, "production_fit_same_k")
    assert blocked.status == "blocked"
    assert blocked.payload["evidence"] == {
        "blocked_by": ["sealed_holdout_veto"],
        "reason": "upstream_dependency_not_passed",
    }
    look_families = store.verify_planned_look_receipts(run_id, require_terminal=True)
    assert "kernel_design_monthly" in look_families
    assert "kernel_production_monthly" not in look_families


def test_handler_model_error_is_terminal_scientific_failure_with_failed_looks(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    calls: list[str] = []

    def fail_scientifically(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        raise ModelError("registered kernel precision failed")

    with _start(store, run_id, preflight) as lease:
        result = coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers(
                calls,
                overrides={"design_block_calibration": fail_scientifically},
            ),
        )

    assert result.failed_task_ids == ("design_block_calibration",)
    failure = store.read_task_result(run_id, "design_block_calibration")
    assert failure.payload["evidence"]["reason"] == "scientific_model_error"
    assert failure.payload["evidence"]["error_type"] == "ModelError"
    looks = store.verify_planned_look_receipts(run_id, require_terminal=True)
    for family_id in ("kernel_design_monthly", "kernel_design_daily"):
        assert len(looks[family_id]) == 1
        assert looks[family_id][-1].decision == "fail"
        assert looks[family_id][-1].observed["coordinator_scientific_failure"]
    assert "design_fragmentation" not in calls
    assert "production_fit_same_k" in calls


def test_programming_error_propagates_then_resume_skips_durable_tasks(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    first_calls: list[str] = []

    def programming_error(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        raise RuntimeError("bug in handler")

    lease = _start(store, run_id, preflight)
    with pytest.raises(RuntimeError, match="bug in handler"):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers(
                first_calls,
                overrides={"design_macro_stability": programming_error},
            ),
        )
    assert store.read_launch(run_id).state == "running"
    assert tuple(store.verify_task_results(run_id, require_complete=False)) == (
        "input_integrity",
        "design_candidate_viability",
        "design_predictive_selection",
        "sealed_holdout_veto",
    )
    lease.close()

    resumed_calls: list[str] = []
    with _start(store, run_id, preflight, resume=True) as resumed:
        result = coordinate_g3_smoke_launch(
            store=store,
            lease=resumed,
            handlers=_passing_handlers(resumed_calls),
        )
    assert resumed_calls[0] == "design_macro_stability"
    assert all(task_id not in resumed_calls for task_id in G3_GATE_ORDER[:4])
    assert result.all_tasks_passed
    assert result.selected_n_states == 3


def test_resume_reuses_only_an_exact_full_planned_look_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    original_write = store.write_task_result
    interrupted = False

    def interrupt_after_look_commit(*args: object, **kwargs: object) -> object:
        nonlocal interrupted
        if kwargs.get("task_id") == "design_block_calibration" and not interrupted:
            interrupted = True
            raise RuntimeError("crash after look commit")
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "write_task_result", interrupt_after_look_commit)
    lease = _start(store, run_id, preflight)
    with pytest.raises(RuntimeError, match="crash after look commit"):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers([], two_look_task="design_block_calibration"),
        )
    look_prefix = store.verify_planned_look_receipts(run_id, require_terminal=False)
    assert len(look_prefix["kernel_design_monthly"]) == 2
    assert "design_block_calibration" not in store.verify_task_results(
        run_id, require_complete=False
    )
    lease.close()
    monkeypatch.setattr(store, "write_task_result", original_write)

    with (
        _start(store, run_id, preflight, resume=True) as mismatched,
        pytest.raises(IntegrityError, match="prefix differs"),
    ):
        coordinate_g3_smoke_launch(
            store=store,
            lease=mismatched,
            handlers=_passing_handlers(
                [],
                two_look_task="design_block_calibration",
                look_marker="changed",
            ),
        )

    calls: list[str] = []
    with _start(store, run_id, preflight, resume=True) as resumed:
        result = coordinate_g3_smoke_launch(
            store=store,
            lease=resumed,
            handlers=_passing_handlers(calls, two_look_task="design_block_calibration"),
        )
    assert calls[0] == "design_block_calibration"
    assert result.all_tasks_passed
    assert all(
        len(receipts) == 2
        for family_id, receipts in store.verify_planned_look_receipts(
            run_id, require_terminal=True
        ).items()
        if family_id.startswith("kernel_design")
    )


def test_resume_rejects_foreign_json_task_receipts(tmp_path: Path) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    lease = _start(store, run_id, preflight)
    store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload={"arbitrary": "json"},
    )
    lease.close()

    with _start(store, run_id, preflight, resume=True) as resumed:
        with pytest.raises(IntegrityError, match="smoke coordinator envelope"):
            coordinate_g3_smoke_launch(
                store=store,
                lease=resumed,
                handlers=_passing_handlers([]),
            )
        assert store.read_launch(run_id).state == "running"


def test_frozen_selected_k_cannot_be_replaced_by_a_runner_up(tmp_path: Path) -> None:
    store, run_id, preflight = _planned_store(tmp_path)

    def promote_runner_up(context: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        assert context.selected_n_states == 3
        return G3SmokeTaskOutcome(
            "passed",
            {"attempted_promotion": 4},
            4,
        )

    with _start(store, run_id, preflight) as lease:
        with pytest.raises(ModelError, match="changed the frozen"):
            coordinate_g3_smoke_launch(
                store=store,
                lease=lease,
                handlers=_passing_handlers(
                    [], overrides={"sealed_holdout_veto": promote_runner_up}
                ),
            )
        assert "sealed_holdout_veto" not in store.verify_task_results(
            run_id, require_complete=False
        )
        assert (
            store.read_task_result(run_id, "design_predictive_selection").payload[
                "selected_n_states"
            ]
            == 3
        )


def test_failed_selection_freezes_no_k_and_promotes_no_runner_up(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    calls: list[str] = []

    def no_selection(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return G3SmokeTaskOutcome(
            "failed",
            {"reason": "no_candidate_passed_registered_selection"},
            None,
        )

    with _start(store, run_id, preflight) as lease:
        result = coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers(
                calls,
                overrides={"design_predictive_selection": no_selection},
            ),
        )
    assert result.selected_n_states is None
    assert result.failed_task_ids == ("design_predictive_selection",)
    assert result.blocked_task_ids == failure_propagation(
        ("design_predictive_selection",)
    )
    assert calls == list(G3_GATE_ORDER[:3])


@pytest.mark.parametrize(
    "bad_outcome, message",
    [
        (
            G3SmokeTaskOutcome("passed", {"ok": True}, 3, ()),
            "every owned planned-look family",
        ),
        (
            G3SmokeTaskOutcome(
                "passed",
                {"ok": True},
                3,
                _looks_for("design_block_calibration", terminal="fail"),
            ),
            "passing planned-look decisions",
        ),
    ],
)
def test_owned_planned_looks_must_be_exact_and_terminal(
    tmp_path: Path,
    bad_outcome: G3SmokeTaskOutcome,
    message: str,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)

    def invalid(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return bad_outcome

    with _start(store, run_id, preflight) as lease:
        with pytest.raises(ModelError, match=message):
            coordinate_g3_smoke_launch(
                store=store,
                lease=lease,
                handlers=_passing_handlers(
                    [], overrides={"design_block_calibration": invalid}
                ),
            )
        assert not store.verify_planned_look_receipts(run_id, require_terminal=False)


def test_handler_set_is_exact_and_canonical_entry_point_fails_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = dict(_passing_handlers([]).handlers)
    complete.pop("artifact_integrity")
    with pytest.raises(ModelError, match="exact 26"):
        G3SmokeHandlerSet(complete)

    store, run_id, preflight = _planned_store(tmp_path)
    with _start(store, run_id, preflight) as lease:
        monkeypatch.setattr(
            g3_coordinator,
            "canonical_resume_preflight_fingerprint",
            lambda *_args, **_kwargs: preflight,
        )
        before = store.read_launch(run_id)
        with pytest.raises(NotImplementedStageError, match="typed scientific"):
            coordinate_g3_canonical_launch(
                store=store,
                lease=lease,
                live_preflight=object(),  # type: ignore[arg-type]
                checkpoint_store=CalibrationCheckpointStore(tmp_path / "artifacts"),
            )
        after = store.read_launch(run_id)
        assert after.fingerprint == before.fingerprint
        assert store.verify_task_results(run_id, require_complete=False) == {}
        assert store.verify_planned_look_receipts(run_id, require_terminal=False) == {}


def test_handler_set_and_coordinator_runtime_guards(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="must be a mapping"):
        G3SmokeHandlerSet(None)  # type: ignore[arg-type]
    noncallable = dict(_passing_handlers([]).handlers)
    noncallable["input_integrity"] = None  # type: ignore[assignment]
    with pytest.raises(ModelError, match="must be callable"):
        G3SmokeHandlerSet(noncallable)

    store, run_id, preflight = _planned_store(tmp_path)
    lease = _start(store, run_id, preflight)
    with pytest.raises(ModelError, match="wrong type"):
        coordinate_g3_smoke_launch(
            store=object(),  # type: ignore[arg-type]
            lease=lease,
            handlers=_passing_handlers([]),
        )
    with pytest.raises(ModelError, match="requires a smoke handler"):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=object(),  # type: ignore[arg-type]
        )
    lease.workers = 8
    with pytest.raises(IntegrityError, match="does not match"):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers([]),
        )
    lease.workers = 9
    lease.close()
    with pytest.raises(IntegrityError, match="active launch lease"):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers([]),
        )


def _replace_first_look(
    **changes: object,
) -> tuple[G3PlannedLookDecision, ...]:
    looks = list(_looks_for("design_block_calibration"))
    first = looks[0]
    values: dict[str, object] = {
        "family_id": first.family_id,
        "look_index": first.look_index,
        "sample_count": first.sample_count,
        "decision": first.decision,
        "observed": first.observed,
    }
    values.update(changes)
    looks[0] = G3PlannedLookDecision(**values)  # type: ignore[arg-type]
    return tuple(looks)


@pytest.mark.parametrize(
    "planned_looks, message",
    [
        ([], "immutable tuple"),
        ((object(),), "G3PlannedLookDecision"),
        (_replace_first_look(family_id="unknown"), "does not own"),
        (tuple(reversed(_looks_for("design_block_calibration"))), "registry order"),
        (_replace_first_look(look_index=True), "sequential"),
        (_replace_first_look(sample_count=9_999), "differs from the registry"),
        (_replace_first_look(decision="unknown"), "decision is invalid"),
        (_replace_first_look(decision="continue"), "must terminate"),
        (_replace_first_look(observed={}), "non-empty mapping"),
        (_replace_first_look(observed={"value": float("nan")}), "non-finite"),
        (_replace_first_look(observed={1: "bad"}), "invalid key"),
        (_replace_first_look(observed={"api_token": "bad"}), "secret-like"),
        (_replace_first_look(observed={"value": {1, 2}}), "unsupported JSON"),
    ],
)
def test_malformed_planned_look_contract_fails_before_any_look_write(
    tmp_path: Path,
    planned_looks: object,
    message: str,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)

    def malformed(context: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return G3SmokeTaskOutcome(
            "passed",
            {"ok": True},
            context.selected_n_states,
            planned_looks,  # type: ignore[arg-type]
        )

    with _start(store, run_id, preflight) as lease:
        with pytest.raises(ModelError, match=message):
            coordinate_g3_smoke_launch(
                store=store,
                lease=lease,
                handlers=_passing_handlers(
                    [], overrides={"design_block_calibration": malformed}
                ),
            )
        assert not store.verify_planned_look_receipts(run_id, require_terminal=False)


def test_unowned_look_and_non_outcome_return_fail_the_handler_contract(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path / "unowned")

    def unowned(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return G3SmokeTaskOutcome(
            "passed",
            {"ok": True},
            None,
            (
                G3PlannedLookDecision(
                    "kernel_design_monthly",
                    1,
                    10_000,
                    "pass",
                    {"ok": True},
                ),
            ),
        )

    with (
        _start(store, run_id, preflight) as lease,
        pytest.raises(ModelError, match="does not own"),
    ):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers([], overrides={"input_integrity": unowned}),
        )

    store, run_id, preflight = _planned_store(tmp_path / "wrong-return")

    def wrong_return(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return object()  # type: ignore[return-value]

    with (
        _start(store, run_id, preflight) as lease,
        pytest.raises(ModelError, match="must return"),
    ):
        coordinate_g3_smoke_launch(
            store=store,
            lease=lease,
            handlers=_passing_handlers([], overrides={"input_integrity": wrong_return}),
        )


@pytest.mark.parametrize(
    "task_id, outcome, message",
    [
        (
            "input_integrity",
            G3SmokeTaskOutcome(
                "blocked",  # type: ignore[arg-type]
                {"ok": False},
                None,
            ),
            "status is invalid",
        ),
        (
            "input_integrity",
            G3SmokeTaskOutcome("passed", {"ok": True}, 3),
            "before a passed selection",
        ),
        (
            "design_predictive_selection",
            G3SmokeTaskOutcome("failed", {"ok": False}, 3),
            "failed selection",
        ),
        (
            "design_predictive_selection",
            G3SmokeTaskOutcome("passed", {"ok": True}, 6),
            "one of 2, 3, 4, or 5",
        ),
        (
            "input_integrity",
            G3SmokeTaskOutcome("passed", {}, None),
            "non-empty mapping",
        ),
    ],
)
def test_invalid_task_outcomes_fail_without_becoming_scientific_failures(
    tmp_path: Path,
    task_id: str,
    outcome: G3SmokeTaskOutcome,
    message: str,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)

    def invalid(_: G3SmokeTaskContext) -> G3SmokeTaskOutcome:
        return outcome

    with _start(store, run_id, preflight) as lease:
        with pytest.raises(ModelError, match=message):
            coordinate_g3_smoke_launch(
                store=store,
                lease=lease,
                handlers=_passing_handlers([], overrides={task_id: invalid}),
            )
        assert task_id not in store.verify_task_results(run_id, require_complete=False)


def _coordinator_payload(
    *,
    selected_n_states: int | None,
    evidence: dict[str, object] | None = None,
    execution_mode: str = "smoke",
) -> dict[str, object]:
    return {
        "coordinator_schema_version": 1,
        "evidence": evidence or {"manual": True},
        "execution_mode": execution_mode,
        "selected_n_states": selected_n_states,
    }


def test_resume_rejects_invalid_smoke_envelopes_and_selected_k_state(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path / "mode")
    lease = _start(store, run_id, preflight)
    store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload=_coordinator_payload(
            selected_n_states=None, execution_mode="canonical"
        ),
    )
    lease.close()
    with (
        _start(store, run_id, preflight, resume=True) as resumed,
        pytest.raises(IntegrityError, match="envelope is invalid"),
    ):
        coordinate_g3_smoke_launch(
            store=store, lease=resumed, handlers=_passing_handlers([])
        )

    store, run_id, preflight = _planned_store(tmp_path / "early-k")
    lease = _start(store, run_id, preflight)
    store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload=_coordinator_payload(selected_n_states=3),
    )
    lease.close()
    with (
        _start(store, run_id, preflight, resume=True) as resumed,
        pytest.raises(IntegrityError, match="before a passed selection"),
    ):
        coordinate_g3_smoke_launch(
            store=store, lease=resumed, handlers=_passing_handlers([])
        )

    store, run_id, preflight = _planned_store(tmp_path / "failed-selection")
    lease = _start(store, run_id, preflight)
    for task_id in G3_GATE_ORDER[:2]:
        store.write_task_result(
            lease,
            task_id=task_id,
            status="passed",
            payload=_coordinator_payload(selected_n_states=None),
        )
    store.write_task_result(
        lease,
        task_id="design_predictive_selection",
        status="failed",
        payload=_coordinator_payload(selected_n_states=3),
    )
    lease.close()
    with (
        _start(store, run_id, preflight, resume=True) as resumed,
        pytest.raises(IntegrityError, match="failed selection"),
    ):
        coordinate_g3_smoke_launch(
            store=store, lease=resumed, handlers=_passing_handlers([])
        )

    store, run_id, preflight = _planned_store(tmp_path / "invalid-selection-k")
    lease = _start(store, run_id, preflight)
    for task_id in G3_GATE_ORDER[:2]:
        store.write_task_result(
            lease,
            task_id=task_id,
            status="passed",
            payload=_coordinator_payload(selected_n_states=None),
        )
    store.write_task_result(
        lease,
        task_id="design_predictive_selection",
        status="passed",
        payload=_coordinator_payload(selected_n_states=9),
    )
    lease.close()
    with (
        _start(store, run_id, preflight, resume=True) as resumed,
        pytest.raises(IntegrityError, match="selected state count is invalid"),
    ):
        coordinate_g3_smoke_launch(
            store=store, lease=resumed, handlers=_passing_handlers([])
        )

    store, run_id, preflight = _planned_store(tmp_path / "changed-k")
    lease = _start(store, run_id, preflight)
    for task_id in G3_GATE_ORDER[:2]:
        store.write_task_result(
            lease,
            task_id=task_id,
            status="passed",
            payload=_coordinator_payload(selected_n_states=None),
        )
    store.write_task_result(
        lease,
        task_id="design_predictive_selection",
        status="passed",
        payload=_coordinator_payload(selected_n_states=3),
    )
    store.write_task_result(
        lease,
        task_id="sealed_holdout_veto",
        status="passed",
        payload=_coordinator_payload(selected_n_states=4),
    )
    lease.close()
    with (
        _start(store, run_id, preflight, resume=True) as resumed,
        pytest.raises(IntegrityError, match="changed the frozen"),
    ):
        coordinate_g3_smoke_launch(
            store=store, lease=resumed, handlers=_passing_handlers([])
        )


def test_resume_rejects_blocked_receipt_not_derived_by_coordinator(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    lease = _start(store, run_id, preflight)
    store.write_task_result(
        lease,
        task_id="input_integrity",
        status="failed",
        payload=_coordinator_payload(selected_n_states=None),
    )
    store.write_task_result(
        lease,
        task_id="design_candidate_viability",
        status="blocked",
        payload=_coordinator_payload(
            selected_n_states=None,
            evidence={"reason": "caller_claimed_block"},
        ),
    )
    lease.close()
    with (
        _start(store, run_id, preflight, resume=True) as resumed,
        pytest.raises(IntegrityError, match="not coordinator-derived"),
    ):
        coordinate_g3_smoke_launch(
            store=store, lease=resumed, handlers=_passing_handlers([])
        )


def test_canonical_resume_requires_checkpoint_fingerprint_on_every_passed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    lease = _start(store, run_id, preflight)
    store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload={"scientific_evidence": True},
    )
    monkeypatch.setattr(
        g3_coordinator,
        "canonical_resume_preflight_fingerprint",
        lambda *_args, **_kwargs: preflight,
    )
    with pytest.raises(IntegrityError, match="no valid checkpoint fingerprint"):
        coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=object(),  # type: ignore[arg-type]
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "checkpoints"),
        )
    assert store.read_launch(run_id).state == "running"
    lease.close()


def test_canonical_finalization_wrapper_forwards_only_artifact_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prpg.model.g3_science_handlers as handlers_module

    observed: dict[str, object] = {}
    expected = object()

    def finalize(**kwargs: object) -> object:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(handlers_module, "finalize_canonical_g3_artifacts", finalize)
    pair = object()
    result = finalize_g3_canonical_artifacts(
        store=object(),  # type: ignore[arg-type]
        lease=object(),  # type: ignore[arg-type]
        live_preflight=object(),  # type: ignore[arg-type]
        checkpoint_store=object(),  # type: ignore[arg-type]
        prefix=object(),  # type: ignore[arg-type]
        artifact_pair=pair,  # type: ignore[arg-type]
    )

    assert result is expected
    assert observed["artifact_pair"] is pair
    assert "design_artifact" not in observed
    assert "production_artifact" not in observed
