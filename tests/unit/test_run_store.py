from __future__ import annotations

import json
from pathlib import Path

import pytest

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_registry import G3_PLANNED_LOOK_REGISTRY, G3_TASK_REGISTRY
from prpg.model.run_store import (
    COMPLETE_MARKER_NAME,
    FAILED_PLANNED_LOOK_DISPOSITION_SCHEMA_ID,
    CalibrationRunIdentity,
    CalibrationRunStore,
    default_run_identity,
)


def _store(tmp_path: Path) -> tuple[CalibrationRunStore, str]:
    store = CalibrationRunStore(tmp_path / "runs")
    identity = default_run_identity(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        workers=9,
    )
    return store, store.initialize(identity).run_id


def _planned_store(
    tmp_path: Path,
) -> tuple[CalibrationRunStore, str, str]:
    store, run_id = _store(tmp_path)
    preflight = "d" * 64
    store.plan_launch(run_id, preflight_fingerprint=preflight, workers=9)
    return store, run_id, preflight


def _rewrite_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _not_started(fingerprint: str) -> dict[str, str]:
    return {
        "disposition": "not_started",
        "failure_certificate_fingerprint": fingerprint,
    }


def _terminal(*fingerprints: str) -> dict[str, object]:
    return {
        "disposition": "terminal_receipts",
        "receipt_fingerprints": list(fingerprints),
    }


def _failed_look_dispositions(
    families: dict[str, dict[str, object] | dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_id": FAILED_PLANNED_LOOK_DISPOSITION_SCHEMA_ID,
        "families": families,
    }


def test_run_and_receipts_are_reusable_only_for_identical_bytes(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    reference = store.verify(run_id)
    receipt = store.write_receipt(
        run_id,
        stage="candidate_fit",
        task_id="k2",
        payload={"passed": True, "bic": 123.5},
    )
    reused = store.write_receipt(
        run_id,
        stage="candidate_fit",
        task_id="k2",
        payload={"bic": 123.5, "passed": True},
    )

    assert reference.run_id == run_id
    assert not receipt.reused
    assert reused.reused
    assert (
        store.read_receipt(run_id, stage="candidate_fit", task_id="k2").fingerprint
        == receipt.fingerprint
    )
    with pytest.raises(IntegrityError, match="differs"):
        store.write_receipt(
            run_id,
            stage="candidate_fit",
            task_id="k2",
            payload={"passed": False, "bic": 123.5},
        )


def test_complete_requires_and_hashes_every_expected_receipt(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    for task in ("k2", "k3"):
        store.write_receipt(
            run_id,
            stage="candidate_fit",
            task_id=task,
            payload={"task": task},
        )
    marker = store.complete(
        run_id,
        expected_tasks={"candidate_fit": ("k2", "k3")},
    )

    assert not marker.reused
    assert len(marker.receipt_fingerprints) == 2
    assert marker.path.name == COMPLETE_MARKER_NAME
    assert store.complete(
        run_id,
        expected_tasks={"candidate_fit": ("k2", "k3")},
    ).reused
    with pytest.raises(IntegrityError, match="cannot be opened"):
        store.complete(
            run_id,
            expected_tasks={"candidate_fit": ("k2", "k3", "k4")},
        )


def test_tampered_receipt_and_manifest_fail_verification(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    receipt = store.write_receipt(
        run_id,
        stage="candidate_fit",
        task_id="k2",
        payload={"passed": True},
    )
    value = json.loads(receipt.path.read_text())
    value["payload"]["passed"] = False
    receipt.path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(IntegrityError, match="hash"):
        store.read_receipt(run_id, stage="candidate_fit", task_id="k2")


def test_secret_nonfinite_and_unsafe_coordinates_fail(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    with pytest.raises(ModelError, match="secret-like"):
        store.write_receipt(
            run_id,
            stage="candidate_fit",
            task_id="k2",
            payload={"api_token": "bad"},
        )
    with pytest.raises(ModelError, match="non-finite"):
        store.write_receipt(
            run_id,
            stage="candidate_fit",
            task_id="k2",
            payload={"value": float("nan")},
        )
    with pytest.raises(ModelError, match="safe canonical"):
        store.write_receipt(
            run_id,
            stage="../escape",
            task_id="k2",
            payload={"ok": True},
        )


def test_symbolic_link_run_root_fails_closed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    store = CalibrationRunStore(linked)
    identity = default_run_identity(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        workers=9,
    )
    with pytest.raises(IntegrityError, match="symbolic-link ancestor"):
        store.initialize(identity)


def test_scientific_run_id_is_independent_of_worker_count(tmp_path: Path) -> None:
    store = CalibrationRunStore(tmp_path / "runs")
    one = default_run_identity(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        workers=1,
    )
    nine = default_run_identity(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        source_code_fingerprint="d" * 64,
        dependency_lock_fingerprint="e" * 64,
        workers=9,
    )

    first = store.initialize(one)
    second = store.initialize(nine)

    assert first.run_id == second.run_id
    assert first.identity.as_dict() == second.identity.as_dict()
    assert "workers" not in first.identity.as_dict()
    assert second.reused


def test_scientific_run_id_changes_with_source_or_dependency_lock(
    tmp_path: Path,
) -> None:
    store = CalibrationRunStore(tmp_path / "runs")
    base = {
        "config_fingerprint": "a" * 64,
        "processed_data_fingerprint": "b" * 64,
        "calibration_input_fingerprint": "c" * 64,
        "source_code_fingerprint": "d" * 64,
        "dependency_lock_fingerprint": "e" * 64,
        "workers": 9,
    }
    original = store.initialize(default_run_identity(**base))
    changed_source = store.initialize(
        default_run_identity(**{**base, "source_code_fingerprint": "f" * 64})
    )
    changed_lock = store.initialize(
        default_run_identity(**{**base, "dependency_lock_fingerprint": "1" * 64})
    )

    assert len({original.run_id, changed_source.run_id, changed_lock.run_id}) == 3
    assert original.identity.source_code_fingerprint == "d" * 64
    assert original.identity.dependency_lock_fingerprint == "e" * 64


def test_launch_lock_is_exclusive_and_completed_launch_cannot_rerun(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    lease = store.start_launch(run_id, preflight_fingerprint=preflight, workers=9)
    assert store.read_launch(run_id).state == "running"
    with pytest.raises(IntegrityError, match="holds the lock"):
        store.start_launch(
            run_id,
            preflight_fingerprint=preflight,
            workers=9,
            resume=True,
        )
    lease.close()
    with pytest.raises(IntegrityError, match="already running"):
        store.start_launch(run_id, preflight_fingerprint=preflight, workers=9)

    resumed = store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
        resume=True,
    )
    store.write_task_result(
        resumed,
        task_id="input_integrity",
        status="failed",
        payload={"reason": "fixture"},
    )
    for spec in G3_TASK_REGISTRY[1:]:
        store.write_task_result(
            resumed,
            task_id=spec.task_id,
            status="blocked",
            payload={"reason": "upstream_failure"},
        )
    completed = store.complete_launch(resumed)
    assert completed.state == "completed"
    assert completed.sequence == 3
    resumed.close()
    with pytest.raises(IntegrityError, match="cannot be rerun"):
        store.start_launch(
            run_id,
            preflight_fingerprint=preflight,
            workers=9,
            resume=True,
        )


def test_task_results_enforce_dependencies_and_failure_propagation(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        with pytest.raises(ModelError, match="dependencies are incomplete"):
            store.write_task_result(
                lease,
                task_id="design_candidate_viability",
                status="passed",
                payload={"ok": True},
            )
        first = store.write_task_result(
            lease,
            task_id="input_integrity",
            status="failed",
            payload={"ok": False},
        )
        assert not first.reused
        assert store.write_task_result(
            lease,
            task_id="input_integrity",
            status="failed",
            payload={"ok": False},
        ).reused
        with pytest.raises(ModelError, match="must propagate"):
            store.write_task_result(
                lease,
                task_id="design_candidate_viability",
                status="passed",
                payload={"ok": False},
            )
        blocked = store.write_task_result(
            lease,
            task_id="design_candidate_viability",
            status="blocked",
            payload={"ok": False},
        )
        assert blocked.blocked_by == ("input_integrity",)


def test_all_passed_launch_requires_and_verifies_terminal_planned_looks(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    owner_families: dict[str, list[str]] = {}
    for look in G3_PLANNED_LOOK_REGISTRY:
        owner_families.setdefault(look.owner_task_id, []).append(look.family_id)

    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        for spec in G3_TASK_REGISTRY:
            for family_id in owner_families.get(spec.task_id, []):
                receipt = store.write_planned_look_receipt(
                    lease,
                    family_id=family_id,
                    look_index=1,
                    sample_count=(1_000 if family_id.startswith("bridge") else 10_000),
                    decision="pass",
                    observed={"precision_passed": True},
                )
                assert store.write_planned_look_receipt(
                    lease,
                    family_id=family_id,
                    look_index=1,
                    sample_count=receipt.sample_count,
                    decision="pass",
                    observed={"precision_passed": True},
                ).reused
                with pytest.raises(IntegrityError, match="differs from requested"):
                    store.write_planned_look_receipt(
                        lease,
                        family_id=family_id,
                        look_index=1,
                        sample_count=receipt.sample_count,
                        decision="pass",
                        observed={"precision_passed": False},
                    )
            store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="passed",
                payload={"passed": True},
            )
            owned = owner_families.get(spec.task_id, [])
            if owned:
                family_id = owned[0]
                with pytest.raises(ModelError, match="owner task result"):
                    store.write_planned_look_receipt(
                        lease,
                        family_id=family_id,
                        look_index=1,
                        sample_count=(
                            1_000 if family_id.startswith("bridge") else 10_000
                        ),
                        decision="pass",
                        observed={"precision_passed": True},
                    )
        completed = store.complete_launch(lease)
        assert completed.state == "completed"
        assert len(store.verify_task_results(run_id, require_complete=True)) == 26
        looks = store.verify_planned_look_receipts(
            run_id,
            require_terminal=True,
            task_results=store.verify_task_results(run_id, require_complete=True),
        )
        assert len(looks) == len(G3_PLANNED_LOOK_REGISTRY)


def test_failed_look_owners_complete_with_not_started_and_mixed_dispositions(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        for spec in G3_TASK_REGISTRY[:7]:
            store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="passed",
                payload={"passed": True},
            )
        store.write_task_result(
            lease,
            task_id="design_block_calibration",
            status="failed",
            payload={
                "failed": True,
                "planned_look_dispositions": _failed_look_dispositions(
                    {
                        "kernel_design_monthly": _not_started("1" * 64),
                        "kernel_design_daily": _not_started("2" * 64),
                    }
                ),
            },
        )
        for task_id in ("design_fragmentation", "design_dependence"):
            store.write_task_result(
                lease,
                task_id=task_id,
                status="blocked",
                payload={"blocked": True},
            )
        for task_id in (
            "production_fit_same_k",
            "production_macro_stability",
            "production_latent_correctness",
            "production_refitted_adequacy",
        ):
            store.write_task_result(
                lease,
                task_id=task_id,
                status="passed",
                payload={"passed": True},
            )
        monthly = store.write_planned_look_receipt(
            lease,
            family_id="kernel_production_monthly",
            look_index=1,
            sample_count=10_000,
            decision="pass",
            observed={"precision_passed": True},
        )
        store.write_task_result(
            lease,
            task_id="production_block_calibration",
            status="failed",
            payload={
                "failed": True,
                "planned_look_dispositions": _failed_look_dispositions(
                    {
                        "kernel_production_monthly": _terminal(monthly.fingerprint),
                        "kernel_production_daily": _not_started("3" * 64),
                    }
                ),
            },
        )
        for task_id in ("production_fragmentation", "production_dependence"):
            store.write_task_result(
                lease,
                task_id=task_id,
                status="blocked",
                payload={"blocked": True},
            )
        store.write_task_result(
            lease,
            task_id="four_cell_adequacy_holm",
            status="passed",
            payload={"passed": True},
        )
        for spec in G3_TASK_REGISTRY[18:]:
            store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="blocked",
                payload={"blocked": True},
            )

        completed = store.complete_launch(lease)
        assert completed.state == "completed"
        looks = store.verify_planned_look_receipts(
            run_id,
            require_terminal=True,
            task_results=store.verify_task_results(run_id, require_complete=True),
        )
        assert tuple(looks) == ("kernel_production_monthly",)


@pytest.mark.parametrize(
    "case",
    (
        "missing_document",
        "wrong_schema",
        "incomplete_families",
        "not_started_after_receipt",
        "terminal_without_receipt",
        "nonterminal_receipt",
    ),
)
def test_failed_look_dispositions_reject_missing_forged_and_partial_evidence(
    tmp_path: Path,
    case: str,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path / case)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        for spec in G3_TASK_REGISTRY[:7]:
            store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="passed",
                payload={"passed": True},
            )
        receipt = None
        if case in {"not_started_after_receipt", "nonterminal_receipt"}:
            receipt = store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=1,
                sample_count=10_000,
                decision=("continue" if case == "nonterminal_receipt" else "pass"),
                observed={"precision_passed": False},
            )

        payload: dict[str, object] = {"failed": True}
        families: dict[str, dict[str, object] | dict[str, str]] = {
            "kernel_design_monthly": _not_started("1" * 64),
            "kernel_design_daily": _not_started("2" * 64),
        }
        if case == "wrong_schema":
            document = _failed_look_dispositions(families)
            document["schema_id"] = "wrong"
            payload["planned_look_dispositions"] = document
        elif case == "incomplete_families":
            payload["planned_look_dispositions"] = _failed_look_dispositions(
                {"kernel_design_monthly": _not_started("1" * 64)}
            )
        elif case == "not_started_after_receipt":
            payload["planned_look_dispositions"] = _failed_look_dispositions(families)
        elif case == "terminal_without_receipt":
            families["kernel_design_monthly"] = _terminal("4" * 64)
            payload["planned_look_dispositions"] = _failed_look_dispositions(families)
        elif case == "nonterminal_receipt":
            assert receipt is not None
            families["kernel_design_monthly"] = _terminal(receipt.fingerprint)
            payload["planned_look_dispositions"] = _failed_look_dispositions(families)

        with pytest.raises(ModelError, match="planned-look|look disposition"):
            store.write_task_result(
                lease,
                task_id="design_block_calibration",
                status="failed",
                payload=payload,
            )


def test_planned_looks_cannot_skip_continue_at_final_or_follow_terminal(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        for task_id in (
            "input_integrity",
            "design_candidate_viability",
            "design_predictive_selection",
        ):
            store.write_task_result(
                lease,
                task_id=task_id,
                status="passed",
                payload={"passed": True},
            )
        with pytest.raises(ModelError, match="sequentially"):
            store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=2,
                sample_count=20_000,
                decision="continue",
                observed={"passed": False},
            )
        store.write_planned_look_receipt(
            lease,
            family_id="kernel_design_monthly",
            look_index=1,
            sample_count=10_000,
            decision="pass",
            observed={"passed": True},
        )
        with pytest.raises(ModelError, match="after a terminal"):
            store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=2,
                sample_count=20_000,
                decision="pass",
                observed={"passed": True},
            )
        with pytest.raises(ModelError, match="final planned look"):
            store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_daily",
                look_index=6,
                sample_count=320_000,
                decision="continue",
                observed={"passed": False},
            )


def test_resume_reverifies_task_hashes_before_returning_a_lease(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    lease = store.start_launch(run_id, preflight_fingerprint=preflight, workers=9)
    result = store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload={"passed": True},
    )
    lease.close()
    value = json.loads(result.path.read_text())
    value["payload"]["passed"] = False
    result.path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(IntegrityError, match="hash"):
        store.start_launch(
            run_id,
            preflight_fingerprint=preflight,
            workers=9,
            resume=True,
        )


def test_launch_plan_resume_and_lease_coordinates_fail_closed(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    preflight = "d" * 64
    with pytest.raises(ModelError, match="fingerprint"):
        store.plan_launch(run_id, preflight_fingerprint="bad", workers=9)
    with pytest.raises(ModelError, match="positive integer"):
        store.plan_launch(run_id, preflight_fingerprint=preflight, workers=0)
    planned = store.plan_launch(run_id, preflight_fingerprint=preflight, workers=9)
    assert not planned.reused
    assert store.plan_launch(run_id, preflight_fingerprint=preflight, workers=9).reused
    with pytest.raises(IntegrityError, match="differs"):
        store.plan_launch(run_id, preflight_fingerprint="e" * 64, workers=9)
    with pytest.raises(IntegrityError, match="resume requires"):
        store.start_launch(
            run_id,
            preflight_fingerprint=preflight,
            workers=9,
            resume=True,
        )
    with pytest.raises(IntegrityError, match="differs from the planned"):
        store.start_launch(run_id, preflight_fingerprint="e" * 64, workers=9)
    lease = store.start_launch(run_id, preflight_fingerprint=preflight, workers=9)
    lease.close()
    lease.close()
    assert not lease.active
    with pytest.raises(IntegrityError, match="already closed"):
        lease.__enter__()
    with pytest.raises(IntegrityError, match="active launch lease"):
        store.write_task_result(
            lease,
            task_id="input_integrity",
            status="passed",
            payload={"passed": True},
        )


def test_exact_task_and_look_inputs_are_closed(tmp_path: Path) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        with pytest.raises(ModelError, match="closed registry"):
            store.write_task_result(
                lease,
                task_id="unknown",
                status="passed",
                payload={"passed": True},
            )
        with pytest.raises(ModelError, match="status"):
            store.write_task_result(
                lease,
                task_id="input_integrity",
                status="unknown",  # type: ignore[arg-type]
                payload={"passed": True},
            )
        with pytest.raises(ModelError, match="non-empty"):
            store.write_task_result(
                lease,
                task_id="input_integrity",
                status="passed",
                payload={},
            )
        with pytest.raises(ModelError, match="cannot be blocked"):
            store.write_task_result(
                lease,
                task_id="input_integrity",
                status="blocked",
                payload={"passed": False},
            )
        with pytest.raises(IntegrityError, match="missing"):
            store.read_task_result(run_id, "input_integrity")
        with pytest.raises(IntegrityError, match="incomplete"):
            store.verify_task_results(run_id, require_complete=True)

        for operation in (
            lambda: store.write_planned_look_receipt(
                lease,
                family_id="unknown",
                look_index=1,
                sample_count=1,
                decision="pass",
                observed={"passed": True},
            ),
            lambda: store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=True,  # type: ignore[arg-type]
                sample_count=10_000,
                decision="pass",
                observed={"passed": True},
            ),
            lambda: store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=0,
                sample_count=10_000,
                decision="pass",
                observed={"passed": True},
            ),
            lambda: store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=1,
                sample_count=9_999,
                decision="pass",
                observed={"passed": True},
            ),
            lambda: store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=1,
                sample_count=10_000,
                decision="bad",  # type: ignore[arg-type]
                observed={"passed": True},
            ),
        ):
            with pytest.raises(ModelError):
                operation()
        with pytest.raises(ModelError, match="dependencies have not passed"):
            store.write_planned_look_receipt(
                lease,
                family_id="kernel_design_monthly",
                look_index=1,
                sample_count=10_000,
                decision="pass",
                observed={"passed": True},
            )


def test_task_and_look_allowlists_and_hashes_are_reverified(tmp_path: Path) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    lease = store.start_launch(run_id, preflight_fingerprint=preflight, workers=9)
    result = store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload={"passed": True},
    )
    task_directory = result.path.parent
    extra = task_directory / "unregistered.json"
    extra.write_text("{}\n")
    with pytest.raises(IntegrityError, match="unregistered entry"):
        store.verify_task_results(run_id, require_complete=False)
    extra.unlink()

    for task_id in (
        "design_candidate_viability",
        "design_predictive_selection",
    ):
        store.write_task_result(
            lease,
            task_id=task_id,
            status="passed",
            payload={"passed": True},
        )
    look = store.write_planned_look_receipt(
        lease,
        family_id="kernel_design_monthly",
        look_index=1,
        sample_count=10_000,
        decision="continue",
        observed={"passed": False},
    )
    look.path.rename(look.path.parent / "03.json")
    with pytest.raises(IntegrityError, match="sequence gap"):
        store.verify_planned_look_receipts(run_id, require_terminal=False)
    lease.close()


def test_tampered_launch_and_task_record_fields_fail_verification(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    marker = store.read_launch(run_id)
    marker_value = json.loads(marker.path.read_text())
    marker_value["workers"] = 8
    marker.path.write_text(
        json.dumps(marker_value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(IntegrityError, match="hash"):
        store.read_launch(run_id)


def test_identity_registry_and_worker_validation_are_closed(tmp_path: Path) -> None:
    store = CalibrationRunStore(tmp_path / "runs")
    base = {
        "config_fingerprint": "a" * 64,
        "processed_data_fingerprint": "b" * 64,
        "calibration_input_fingerprint": "c" * 64,
        "source_code_fingerprint": "d" * 64,
        "dependency_lock_fingerprint": "e" * 64,
        "contract_version": G3_CONTRACT_VERSION,
    }
    with pytest.raises(ModelError, match="wrong type"):
        store.initialize(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="task registry"):
        store.initialize(
            CalibrationRunIdentity(
                **base,
                task_registry_fingerprint="d" * 64,
            )
        )
    with pytest.raises(ModelError, match="planned-look registry"):
        store.initialize(
            CalibrationRunIdentity(
                **base,
                planned_look_registry_fingerprint="d" * 64,
            )
        )
    with pytest.raises(ModelError, match="positive integer"):
        store.initialize(CalibrationRunIdentity(**base, workers=0))
    detached = CalibrationRunIdentity(**base)
    with pytest.raises(ModelError, match="not attached"):
        detached.execution_dict()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("keys", "keys are invalid"),
        ("schema", "schema version"),
        ("run_id", "ID is inconsistent"),
        ("identity_type", "identity is invalid"),
        ("identity_field", "identity is invalid"),
        ("identity_hash", "identity hash"),
    ],
)
def test_run_manifest_fields_are_reverified(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    store, run_id = _store(tmp_path)
    path = store.root / run_id / "run-manifest.json"
    value = json.loads(path.read_text())
    if mutation == "keys":
        value["extra"] = True
    elif mutation == "schema":
        value["schema_version"] = 1
    elif mutation == "run_id":
        value["run_id"] = "f" * 64
    elif mutation == "identity_type":
        value["identity"] = []
    elif mutation == "identity_field":
        value["identity"]["contract_version"] = "wrong"
    else:
        value["identity"]["config_fingerprint"] = "f" * 64
    _rewrite_json(path, value)
    with pytest.raises(IntegrityError, match=message):
        store.verify(run_id)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("keys", "keys are invalid"),
        ("unknown_task", "outside the closed registry"),
        ("status", "status is invalid"),
        ("identity", "identity is inconsistent"),
        ("payload", "payload is invalid"),
        ("hash", "hash is inconsistent"),
    ],
)
def test_exact_task_result_fields_are_reverified(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        result = store.write_task_result(
            lease,
            task_id="input_integrity",
            status="passed",
            payload={"passed": True},
        )
        value = json.loads(result.path.read_text())
        if mutation == "keys":
            value["extra"] = True
        elif mutation == "unknown_task":
            value["task_id"] = "unknown"
        elif mutation == "status":
            value["status"] = "unknown"
        elif mutation == "identity":
            value["run_id"] = "f" * 64
        elif mutation == "payload":
            value["payload"] = {"secret_token": "bad"}
        else:
            value["result_fingerprint"] = "f" * 64
        _rewrite_json(result.path, value)
        with pytest.raises(IntegrityError, match=message):
            store.verify_task_results(run_id, require_complete=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("keys", "keys are invalid"),
        ("identity", "identity is inconsistent"),
        ("observed", "observations are invalid"),
        ("hash", "hash is inconsistent"),
    ],
)
def test_planned_look_receipt_fields_are_reverified(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        for task_id in (
            "input_integrity",
            "design_candidate_viability",
            "design_predictive_selection",
        ):
            store.write_task_result(
                lease,
                task_id=task_id,
                status="passed",
                payload={"passed": True},
            )
        receipt = store.write_planned_look_receipt(
            lease,
            family_id="kernel_design_monthly",
            look_index=1,
            sample_count=10_000,
            decision="continue",
            observed={"passed": False},
        )
        value = json.loads(receipt.path.read_text())
        if mutation == "keys":
            value["extra"] = True
        elif mutation == "identity":
            value["sample_count"] = 9_999
        elif mutation == "observed":
            value["observed"] = {"secret_token": "bad"}
        else:
            value["look_fingerprint"] = "f" * 64
        _rewrite_json(receipt.path, value)
        with pytest.raises(IntegrityError, match=message):
            store.verify_planned_look_receipts(run_id, require_terminal=False)


def test_partial_results_and_missing_terminal_looks_cannot_complete(
    tmp_path: Path,
) -> None:
    store, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight, workers=9
    ) as lease:
        store.write_task_result(
            lease,
            task_id="input_integrity",
            status="passed",
            payload={"passed": True},
        )
        with pytest.raises(IntegrityError, match="incomplete"):
            store.verify_task_results(run_id, require_complete=True)
        for spec in G3_TASK_REGISTRY[1:7]:
            store.write_task_result(
                lease,
                task_id=spec.task_id,
                status="passed",
                payload={"passed": True},
            )
        with pytest.raises(ModelError, match="owned planned look"):
            store.write_task_result(
                lease,
                task_id="design_block_calibration",
                status="passed",
                payload={"passed": True},
            )


def test_generic_receipt_and_completion_negative_records_fail_closed(
    tmp_path: Path,
) -> None:
    store, run_id = _store(tmp_path)
    with pytest.raises(ModelError, match="non-empty"):
        store.complete(run_id, expected_tasks={})
    with pytest.raises(ModelError, match="unique"):
        store.complete(run_id, expected_tasks={"stage": ("task", "task")})
    with pytest.raises(ModelError, match="unsupported JSON"):
        store.write_receipt(
            run_id,
            stage="stage",
            task_id="task",
            payload={"bad": {1, 2}},
        )
    receipt = store.write_receipt(
        run_id,
        stage="stage",
        task_id="task",
        payload={"passed": True},
    )
    value = json.loads(receipt.path.read_text())
    value["payload"] = [True]
    _rewrite_json(receipt.path, value)
    with pytest.raises(IntegrityError, match="payload is invalid"):
        store.read_receipt(run_id, stage="stage", task_id="task")


def test_missing_run_and_noncanonical_json_fail_closed(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    with pytest.raises(IntegrityError, match="directory is missing"):
        store.verify("f" * 64)
    manifest = store.root / run_id / "run-manifest.json"
    manifest.write_text("{not-json}\n")
    with pytest.raises(IntegrityError, match="not valid JSON"):
        store.verify(run_id)
