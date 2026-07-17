from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_bridge_driver import (
    _execute_staged_tier_a_pair,
    _prepare_staged,
)
from test_g3_science_handlers import _block_source, _kernel

from prpg.errors import IntegrityError, ModelError
from prpg.model import bridge_driver as driver
from prpg.model import g3_boundary_views as boundary
from prpg.model import g3_planned_look_execution as planned
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_registry import G3_GATE_ORDER, G3_PLANNED_LOOK_REGISTRY
from prpg.model.g3_science_handlers import canonical_planned_looks
from prpg.model.g3_task_binding import ScientificTaskBinding
from prpg.model.kernel_execution import CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
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
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        workers=9,
    )
    run_id = store.initialize(identity).run_id
    preflight = "f" * 64
    store.plan_launch(run_id, preflight_fingerprint=preflight, workers=9)
    return store, run_id, preflight


def _seed_passed_prefix(
    store: CalibrationRunStore,
    lease: LaunchLease,
    *,
    stop_before: str,
) -> None:
    """Publish an exact passed registry prefix, including owner look receipts."""

    owner_specs: dict[str, list[Any]] = {}
    for spec in G3_PLANNED_LOOK_REGISTRY:
        owner_specs.setdefault(spec.owner_task_id, []).append(spec)
    stop = G3_GATE_ORDER.index(stop_before)
    for task_id in G3_GATE_ORDER[:stop]:
        for spec in owner_specs.get(task_id, []):
            store.write_planned_look_receipt(
                lease,
                family_id=spec.family_id,
                look_index=1,
                sample_count=spec.sample_counts[0],
                decision="pass",
                observed={"seeded_for_prior_owner": task_id},
            )
        store.write_task_result(
            lease,
            task_id=task_id,
            status="passed",
            payload={"passed": True},
        )


def _allow_synthetic_block_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    # The helper objects exercise this module's mechanics without pretending
    # that test-built kernel/block objects possess production mint authority.
    monkeypatch.setattr(
        boundary,
        "_block_source_identities",
        lambda _source, *, task_id: {},
    )
    monkeypatch.setattr(
        planned,
        "kernel_calibration_execution_identity",
        lambda value: SimpleNamespace(
            artifact=value.artifact,
            frequency=value.frequency,
        ),
    )


def _synthetic_block_source(
    task_id: str,
    *,
    monthly_widths: tuple[float, ...] | None = None,
    daily_widths: tuple[float, ...] | None = None,
) -> Any:
    limit = CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
    monthly = _kernel("monthly", monthly_widths or (limit * 2.0, limit * 0.5))
    daily = _kernel("daily", daily_widths or (limit * 0.5,))
    source = _block_source(monthly, daily)
    if task_id == "production_block_calibration":
        monthly = replace(monthly, artifact="production")
        daily = replace(daily, artifact="production")
        source = replace(
            source,
            role="production",
            monthly_kernel=monthly,
            daily_kernel=daily,
        )
    return source


@pytest.mark.parametrize(
    ("task_id", "family_prefix"),
    [
        ("design_block_calibration", "kernel_design_"),
        ("production_block_calibration", "kernel_production_"),
    ],
)
def test_projects_all_four_kernel_families_from_terminal_raw_evidence(
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    family_prefix: str,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source(task_id)

    result = planned.project_raw_planned_looks(cast(Any, task_id), source)

    assert tuple(item.family_id for item in result) == (
        f"{family_prefix}monthly",
        f"{family_prefix}monthly",
        f"{family_prefix}daily",
    )
    assert tuple(item.look_index for item in result) == (1, 2, 1)
    assert tuple(item.decision for item in result) == ("continue", "pass", "pass")


def test_kernel_projection_accepts_max_look_failure_and_rejects_early_nonpass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    limit = CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
    maximum_failure = _synthetic_block_source(
        "design_block_calibration",
        monthly_widths=(limit * 2.0,) * 6,
        daily_widths=(limit * 2.0,) * 6,
    )
    projected = planned.project_raw_planned_looks(
        "design_block_calibration",
        maximum_failure,
    )
    assert len(projected) == 12
    assert projected[5].decision == "fail"
    assert projected[-1].decision == "fail"

    early_nonpass = _synthetic_block_source(
        "design_block_calibration",
        monthly_widths=(limit * 2.0,),
    )
    with pytest.raises(ModelError, match="stopped before pass or maximum"):
        planned.project_raw_planned_looks(
            "design_block_calibration",
            early_nonpass,
        )


def test_projects_and_publishes_both_tier_b_early_pass_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request, _binding, _harness = _prepare_staged(tmp_path, monkeypatch)
    _execute_staged_tier_a_pair(session)
    driver.execute_staged_bridge_tier_b_next_look(session, "model")
    model = driver.complete_staged_bridge_tier_b_task(session, "model")
    driver.execute_staged_bridge_tier_b_next_look(session, "nonparametric")
    nonparametric = driver.complete_staged_bridge_tier_b_task(
        session,
        "nonparametric",
    )

    for index, (task_id, source) in enumerate(
        (
            ("bridge_tier_b_model_dgp", model),
            ("bridge_tier_b_nonparametric_dgp", nonparametric),
        )
    ):
        projection = planned.project_raw_planned_looks(
            cast(Any, task_id),
            source,
        )
        assert len(projection) == 1
        assert projection[0].family_id == task_id
        assert projection[0].decision == "pass"

        store, run_id, preflight = _planned_store(tmp_path / f"publish-{index}")
        with store.start_launch(
            run_id,
            preflight_fingerprint=preflight,
            workers=9,
        ) as lease:
            _seed_passed_prefix(store, lease, stop_before=task_id)
            prefix = planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id=cast(Any, task_id),
                source=source,
            )
            assert prefix.receipt_fingerprints[0][0] == task_id
            assert prefix.terminal_decisions == ((task_id, "pass"),)


def test_projects_tier_b_fail_closed_at_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def statistic(_dgp: Any, pair: int) -> float:
        return 1.0 if pair % 1_000 < 25 else 0.0

    session, _request, _binding, _harness = _prepare_staged(
        tmp_path,
        monkeypatch,
        statistic_for_pair=statistic,
    )
    _execute_staged_tier_a_pair(session)
    terminal = None
    while terminal is None or terminal.decision == "continue":
        terminal = driver.execute_staged_bridge_tier_b_next_look(session, "model")
    assert terminal.look_index == 10
    assert terminal.decision == "fail_closed_at_maximum"
    source = driver.complete_staged_bridge_tier_b_task(session, "model")

    projection = planned.project_raw_planned_looks(
        "bridge_tier_b_model_dgp",
        source,
    )
    assert len(projection) == 10
    assert projection[-1].decision == "fail"


def test_publication_is_crash_resumable_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    limit = CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
    source = _synthetic_block_source(
        "design_block_calibration",
        daily_widths=(limit * 2.0, limit * 0.5),
    )
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        _seed_passed_prefix(
            store,
            lease,
            stop_before="design_block_calibration",
        )
        original_write = store.write_planned_look_receipt
        calls = 0

        def interrupt_after_cross_family_prefix(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            # Leave the first family terminal and the second family with one
            # durable continuing look, the strongest cross-family resume case.
            if calls == 4:
                raise RuntimeError("simulated receipt interruption")
            return original_write(*args, **kwargs)

        monkeypatch.setattr(
            store,
            "write_planned_look_receipt",
            interrupt_after_cross_family_prefix,
        )
        with pytest.raises(RuntimeError, match="simulated receipt interruption"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="design_block_calibration",
                source=source,
            )
        partial = store.verify_planned_look_receipts(
            run_id,
            require_terminal=False,
        )
        assert len(partial["kernel_design_monthly"]) == 2
        assert partial["kernel_design_monthly"][-1].decision == "pass"
        assert len(partial["kernel_design_daily"]) == 1
        assert partial["kernel_design_daily"][-1].decision == "continue"

        monkeypatch.setattr(store, "write_planned_look_receipt", original_write)
        resumed = planned.publish_raw_planned_look_prefix(
            store=store,
            lease=lease,
            task_id="design_block_calibration",
            source=source,
        )
        repeated = planned.publish_raw_planned_look_prefix(
            store=store,
            lease=lease,
            task_id="design_block_calibration",
            source=source,
        )
        assert repeated.receipt_fingerprints == resumed.receipt_fingerprints
        assert repeated.publication_fingerprint == resumed.publication_fingerprint


def test_publication_rejects_mixed_existing_prefix_before_appending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("design_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        _seed_passed_prefix(
            store,
            lease,
            stop_before="design_block_calibration",
        )
        store.write_planned_look_receipt(
            lease,
            family_id="kernel_design_monthly",
            look_index=1,
            sample_count=10_000,
            decision="continue",
            observed={"wrong_source": True},
        )
        with pytest.raises(IntegrityError, match="bytes/evidence differ"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="design_block_calibration",
                source=source,
            )
        receipts = store.verify_planned_look_receipts(
            run_id,
            require_terminal=False,
        )
        assert tuple(receipts) == ("kernel_design_monthly",)
        assert len(receipts["kernel_design_monthly"]) == 1


def test_publication_propagates_tamper_and_projection_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("design_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        _seed_passed_prefix(
            store,
            lease,
            stop_before="design_block_calibration",
        )
        projection = planned.project_raw_planned_looks(
            "design_block_calibration",
            source,
        )
        first = projection[0]
        receipt = store.write_planned_look_receipt(
            lease,
            family_id=first.family_id,
            look_index=first.look_index,
            sample_count=first.sample_count,
            decision=first.decision,
            observed=first.observed,
        )
        serialized = json.loads(receipt.path.read_text())
        serialized["observed"]["precision_passed"] = True
        receipt.path.write_text(
            json.dumps(serialized, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with pytest.raises(IntegrityError, match="hash is inconsistent"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="design_block_calibration",
                source=source,
            )

    clean_store, clean_run, clean_preflight = _planned_store(tmp_path / "fault")
    with clean_store.start_launch(
        clean_run,
        preflight_fingerprint=clean_preflight,
        workers=9,
    ) as clean_lease:
        _seed_passed_prefix(
            clean_store,
            clean_lease,
            stop_before="design_block_calibration",
        )

        def projection_fault(_task_id: Any, _source: Any) -> Any:
            raise RuntimeError("unexpected projection fault")

        monkeypatch.setattr(planned, "project_raw_planned_looks", projection_fault)
        with pytest.raises(RuntimeError, match="unexpected projection fault"):
            planned.publish_raw_planned_look_prefix(
                store=clean_store,
                lease=clean_lease,
                task_id="design_block_calibration",
                source=source,
            )
        receipts = clean_store.verify_planned_look_receipts(
            clean_run,
            require_terminal=False,
        )
        assert not receipts


def test_owner_must_be_exact_next_task_even_when_direct_dependency_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("design_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        for task_id in G3_GATE_ORDER[:3]:
            store.write_task_result(
                lease,
                task_id=task_id,
                status="passed",
                payload={"passed": True},
            )
        with pytest.raises(ModelError, match="not the exact next registry task"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="design_block_calibration",
                source=source,
            )


def test_orphan_prior_owner_receipt_cannot_bypass_exact_registry_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("production_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        for task_id in G3_GATE_ORDER[:3]:
            store.write_task_result(
                lease,
                task_id=task_id,
                status="passed",
                payload={"passed": True},
            )
        store.write_planned_look_receipt(
            lease,
            family_id="kernel_design_monthly",
            look_index=1,
            sample_count=10_000,
            decision="pass",
            observed={"orphan_prior_owner": True},
        )
        store.write_task_result(
            lease,
            task_id="sealed_holdout_veto",
            status="passed",
            payload={"passed": True},
        )
        store.write_task_result(
            lease,
            task_id="production_fit_same_k",
            status="passed",
            payload={"passed": True},
        )

        with pytest.raises(ModelError, match="not the exact next registry task"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="production_block_calibration",
                source=source,
            )


def test_exact_prior_owner_receipts_must_remain_terminal_and_passing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("production_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        _seed_passed_prefix(
            store,
            lease,
            stop_before="production_block_calibration",
        )
        receipts = store.verify_planned_look_receipts(
            run_id,
            require_terminal=False,
        )
        receipts["kernel_design_daily"][0].path.unlink()

        with pytest.raises(
            IntegrityError,
            match="prior-task planned-look receipts are missing or nonpassing",
        ):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="production_block_calibration",
                source=source,
            )


def test_owner_rejects_nonpassing_exact_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("design_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        for index, task_id in enumerate(
            G3_GATE_ORDER[: G3_GATE_ORDER.index("design_block_calibration")]
        ):
            store.write_task_result(
                lease,
                task_id=task_id,
                status="failed" if index == 0 else "blocked",
                payload={"passed": False},
            )
        with pytest.raises(ModelError, match="nonpassing prior registry task"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="design_block_calibration",
                source=source,
            )


def test_owner_rejects_receipt_owned_by_any_future_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("design_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        for task_id in G3_GATE_ORDER[:4]:
            store.write_task_result(
                lease,
                task_id=task_id,
                status="passed",
                payload={"passed": True},
            )
        store.write_task_result(
            lease,
            task_id="production_fit_same_k",
            status="passed",
            payload={"passed": True},
        )
        store.write_planned_look_receipt(
            lease,
            family_id="kernel_production_monthly",
            look_index=1,
            sample_count=10_000,
            decision="pass",
            observed={"future": True},
        )

        with pytest.raises(IntegrityError, match="future-task planned looks"):
            planned.publish_raw_planned_look_prefix(
                store=store,
                lease=lease,
                task_id="design_block_calibration",
                source=source,
            )


def test_registered_prefix_is_object_led_and_parity_checks_all_four_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_block_evidence(monkeypatch)
    source = _synthetic_block_source("design_block_calibration")
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        _seed_passed_prefix(
            store,
            lease,
            stop_before="design_block_calibration",
        )
        prefix = planned.publish_raw_planned_look_prefix(
            store=store,
            lease=lease,
            task_id="design_block_calibration",
            source=source,
        )
        assert planned.is_registered_planned_look_prefix(prefix)
        assert not planned.is_registered_planned_look_prefix(copy.copy(prefix))
        with pytest.raises(TypeError, match="created only"):
            planned.RegisteredPlannedLookPrefix()

        original_count = prefix.look_count
        object.__setattr__(prefix, "look_count", original_count + 1)
        assert not planned.is_registered_planned_look_prefix(prefix)
        object.__setattr__(prefix, "look_count", original_count)
        assert planned.is_registered_planned_look_prefix(prefix)

        precision = source.monthly_kernel.completed_looks[0].precision
        original_model_fingerprint = precision.model_artifact_fingerprint
        object.__setattr__(precision, "model_artifact_fingerprint", "0" * 64)
        assert not planned.is_registered_planned_look_prefix(prefix)
        object.__setattr__(
            precision,
            "model_artifact_fingerprint",
            original_model_fingerprint,
        )
        assert planned.is_registered_planned_look_prefix(prefix)

        monkeypatch.setattr(
            planned,
            "verify_scientific_task_binding",
            lambda *_args, **_kwargs: None,
        )
        binding_fingerprint = "9" * 64
        binding = ScientificTaskBinding(
            schema_id="test-binding",
            schema_version=2,
            run_id=run_id,
            run_identity=store.verify(run_id).identity,
            launch_preflight_fingerprint=preflight,
            persisted_preflight_receipt_fingerprint="1" * 64,
            compatible_preflight_fingerprint="2" * 64,
            task_id="design_block_calibration",
            selected_n_states=2,
            dependency_checkpoint_fingerprints=(),
            owned_planned_look_fingerprints=prefix.receipt_fingerprints,
            scientific_component_fingerprints=(),
            materialization_fingerprints=(),
            rng_fingerprints=(),
            slot_registry_fingerprint="3" * 64,
            fingerprint=binding_fingerprint,
        )
        final_view = boundary.build_block_calibration_task_view(
            source,
            task_id="design_block_calibration",
            run_id=run_id,
            binding_fingerprint=binding_fingerprint,
        )
        parity = planned.verify_planned_look_binding_view_parity(
            prefix,
            binding=binding,
            final_view=final_view,
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "checkpoints"),
            live_preflight=cast(Any, object()),
        )
        assert parity.binding_fingerprint == binding_fingerprint
        assert parity.raw_projection_fingerprint == (
            parity.final_view_projection_fingerprint
        )

        wrong_binding = replace(binding, owned_planned_look_fingerprints=())
        with pytest.raises(IntegrityError, match="differ from raw receipts"):
            planned.verify_planned_look_binding_view_parity(
                prefix,
                binding=wrong_binding,
                final_view=final_view,
                checkpoint_store=CalibrationCheckpointStore(
                    tmp_path / "wrong-checkpoints"
                ),
                live_preflight=cast(Any, object()),
            )

        equivalent_source = replace(source)
        equivalent_view = boundary.build_block_calibration_task_view(
            equivalent_source,
            task_id="design_block_calibration",
            run_id=run_id,
            binding_fingerprint=binding_fingerprint,
        )
        with pytest.raises(ModelError, match="different exact ancestry"):
            planned.verify_planned_look_binding_view_parity(
                prefix,
                binding=binding,
                final_view=equivalent_view,
                checkpoint_store=CalibrationCheckpointStore(
                    tmp_path / "equivalent-checkpoints"
                ),
                live_preflight=cast(Any, object()),
            )

        original_projection = canonical_planned_looks
        monkeypatch.setattr(
            planned,
            "canonical_planned_looks",
            lambda task_id, view: original_projection(task_id, view)[:-1],
        )
        with pytest.raises(IntegrityError, match="differ from raw projection"):
            planned.verify_planned_look_binding_view_parity(
                prefix,
                binding=binding,
                final_view=final_view,
                checkpoint_store=CalibrationCheckpointStore(
                    tmp_path / "altered-checkpoints"
                ),
                live_preflight=cast(Any, object()),
            )
