from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tests.unit.test_g3_task_publication import (
    _input_binding,
    _input_evidence,
    _input_source,
    _stores,
)

import prpg.model.g3_failure_publication as failure_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_failure_publication import (
    ExactBlockedScientificTaskPublication,
    is_exact_blocked_scientific_task_publication,
    publish_exact_next_blocked_scientific_task,
    reverify_exact_blocked_scientific_task,
)
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_task_publication import publish_exact_next_scientific_task
from prpg.model.run_store import CalibrationRunStore, default_run_identity


def _planned_store(
    tmp_path: Path,
) -> tuple[CalibrationRunStore, CalibrationCheckpointStore, str, str]:
    store = CalibrationRunStore(tmp_path / "runs")
    run = store.initialize(
        default_run_identity(
            config_fingerprint="a" * 64,
            processed_data_fingerprint="b" * 64,
            calibration_input_fingerprint="c" * 64,
            source_code_fingerprint="d" * 64,
            dependency_lock_fingerprint="e" * 64,
            workers=9,
        )
    )
    preflight = "f" * 64
    store.plan_launch(run.run_id, preflight_fingerprint=preflight, workers=9)
    return (
        store,
        CalibrationCheckpointStore(tmp_path / "checkpoints"),
        run.run_id,
        preflight,
    )


def _unused_live_preflight() -> CanonicalG3Preflight:
    return cast(CanonicalG3Preflight, object())


def _trust_direct_scientific_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replay(**kwargs: object) -> SimpleNamespace:
        store = cast(CalibrationRunStore, kwargs["run_store"])
        run_id = cast(str, kwargs["run_id"])
        task_id = cast(str, kwargs["task_id"])
        return SimpleNamespace(task_result=store.read_task_result(run_id, task_id))

    monkeypatch.setattr(
        failure_module,
        "reverify_published_scientific_task",
        replay,
    )


def test_exact_blocked_publisher_closes_a_failed_input_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, checkpoints, run_id, preflight = _planned_store(tmp_path)
    _trust_direct_scientific_dependencies(monkeypatch)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        store.write_task_result(
            lease,
            task_id="input_integrity",
            status="failed",
            payload={"synthetic_typed_failure": True},
        )
        publications = []
        for expected_task_id in G3_GATE_ORDER[1:]:
            publication = publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=checkpoints,
                live_preflight=_unused_live_preflight(),
            )
            assert publication.task_id == expected_task_id
            assert publication.task_result.status == "blocked"
            assert publication.task_result.blocked_by == publication.blocked_by
            assert publication.generation_authorized is False
            assert is_exact_blocked_scientific_task_publication(publication)
            publications.append(publication)

        completed = store.complete_launch(lease)

    assert completed.state == "completed"
    assert tuple(store.verify_task_results(run_id, require_complete=True)) == (
        G3_GATE_ORDER
    )
    assert not is_exact_blocked_scientific_task_publication(copy.copy(publications[-1]))
    replayed = reverify_exact_blocked_scientific_task(
        run_store=store,
        checkpoint_store=checkpoints,
        live_preflight=_unused_live_preflight(),
        run_id=run_id,
        task_id="artifact_integrity",
    )
    assert replayed.certificate_fingerprint == (
        publications[-1].certificate_fingerprint
    )


def test_blocked_publisher_rejects_a_task_whose_dependencies_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, checkpoints, run_id, preflight = _planned_store(tmp_path)
    _trust_direct_scientific_dependencies(monkeypatch)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        store.write_task_result(
            lease,
            task_id="input_integrity",
            status="passed",
            payload={"synthetic_typed_pass": True},
        )
        with pytest.raises(ModelError, match="must execute"):
            publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=checkpoints,
                live_preflight=_unused_live_preflight(),
            )
    assert tuple(store.verify_task_results(run_id, require_complete=False)) == (
        "input_integrity",
    )


def test_arbitrary_failed_receipt_cannot_authorize_a_blocked_publication(
    tmp_path: Path,
) -> None:
    store, checkpoints, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        store.write_task_result(
            lease,
            task_id="input_integrity",
            status="failed",
            payload={"caller_claim": "failed"},
        )
        with pytest.raises(IntegrityError, match="binding-aware"):
            publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=checkpoints,
                live_preflight=_unused_live_preflight(),
            )
    assert tuple(store.verify_task_results(run_id, require_complete=False)) == (
        "input_integrity",
    )


def test_unexpected_write_error_leaves_the_exact_prefix_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, checkpoints, run_id, preflight = _planned_store(tmp_path)
    _trust_direct_scientific_dependencies(monkeypatch)
    original_write = store.write_task_result
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        original_write(
            lease,
            task_id="input_integrity",
            status="failed",
            payload={"synthetic_typed_failure": True},
        )

        def crash(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("synthetic filesystem failure")

        monkeypatch.setattr(store, "write_task_result", crash)
        with pytest.raises(RuntimeError, match="filesystem"):
            publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=checkpoints,
                live_preflight=_unused_live_preflight(),
            )

    assert tuple(store.verify_task_results(run_id, require_complete=False)) == (
        "input_integrity",
    )
    monkeypatch.setattr(store, "write_task_result", original_write)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
        resume=True,
    ) as lease:
        recovered = publish_exact_next_blocked_scientific_task(
            run_store=store,
            lease=lease,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
        )
    assert recovered.task_id == "design_candidate_viability"


def test_real_binding_aware_typed_failure_authorizes_only_its_blocked_child(
    tmp_path: Path,
) -> None:
    """Exercise the actual failed-task publisher rather than a replay stub."""

    store, checkpoints, run_id, launch, live = _stores(tmp_path)
    evidence = _input_evidence(launch, live, run_id, passing=False)
    binding = _input_binding(store, checkpoints, run_id, live, evidence)
    source = _input_source(evidence, run_id, binding)

    with store.start_launch(
        run_id,
        preflight_fingerprint=launch.fingerprint,
        workers=launch.workers,
    ) as lease:
        failed = publish_exact_next_scientific_task(
            run_store=store,
            lease=lease,
            checkpoint_store=checkpoints,
            live_preflight=live,
            binding=binding,
            source=source,
        )
        assert failed.task_result.task_id == "input_integrity"
        assert failed.task_result.status == "failed"
        assert failed.checkpoint is None

        blocked = publish_exact_next_blocked_scientific_task(
            run_store=store,
            lease=lease,
            checkpoint_store=checkpoints,
            live_preflight=live,
        )
        assert blocked.task_id == "design_candidate_viability"
        assert blocked.blocked_by == ("input_integrity",)
        assert blocked.task_result.status == "blocked"

    replayed_failure = failure_module.reverify_published_scientific_task(
        run_store=store,
        checkpoint_store=checkpoints,
        live_preflight=live,
        run_id=run_id,
        task_id="input_integrity",
    )
    assert replayed_failure.task_result.fingerprint == failed.task_result.fingerprint
    replayed_block = reverify_exact_blocked_scientific_task(
        run_store=store,
        checkpoint_store=checkpoints,
        live_preflight=live,
        run_id=run_id,
        task_id="design_candidate_viability",
    )
    assert replayed_block.certificate_fingerprint == blocked.certificate_fingerprint


def test_blocked_publication_authority_is_constructor_closed() -> None:
    with pytest.raises(TypeError, match="created only by its publisher"):
        ExactBlockedScientificTaskPublication()


@pytest.mark.parametrize("selected", (2, 3, 4, 5))
def test_selected_state_count_is_frozen_across_the_result_prefix(selected: int) -> None:
    before = SimpleNamespace(status="passed", payload={"selected_n_states": selected})
    selection = SimpleNamespace(
        status="passed",
        payload={"selected_n_states": selected},
    )
    after = SimpleNamespace(status="blocked", payload={"selected_n_states": selected})
    prior = {
        "input_integrity": before,
        "design_predictive_selection": selection,
        "sealed_holdout_veto": after,
    }

    assert failure_module._selected_n_states_from_prefix(prior) == selected
    assert failure_module._selected_n_states_from_prefix({}) is None
    assert (
        failure_module._selected_n_states_from_prefix(
            {
                "design_predictive_selection": SimpleNamespace(
                    status="failed",
                    payload={"selected_n_states": selected},
                )
            }
        )
        is None
    )


@pytest.mark.parametrize("selected", (True, None, 1, 6, "2"))
def test_selected_state_count_rejects_invalid_or_changed_prefix(
    selected: object,
) -> None:
    selection = SimpleNamespace(
        status="passed",
        payload={"selected_n_states": selected},
    )
    with pytest.raises(IntegrityError, match="lacks the frozen state count"):
        failure_module._selected_n_states_from_prefix(
            {"design_predictive_selection": selection}
        )

    valid_selection = SimpleNamespace(
        status="passed",
        payload={"selected_n_states": 2},
    )
    changed = SimpleNamespace(status="blocked", payload={"selected_n_states": 3})
    with pytest.raises(IntegrityError, match="changed the frozen state count"):
        failure_module._selected_n_states_from_prefix(
            {
                "design_predictive_selection": valid_selection,
                "sealed_holdout_veto": changed,
            }
        )


@pytest.mark.parametrize("status", ("passed", "failed"))
def test_dependency_reverification_requires_exact_scientific_publication(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    result = SimpleNamespace(
        status=status,
        run_id="a" * 64,
        task_id="input_integrity",
        fingerprint="b" * 64,
    )
    monkeypatch.setattr(
        failure_module,
        "reverify_published_scientific_task",
        lambda **_kwargs: SimpleNamespace(task_result=result),
    )
    failure_module._reverify_dependency_publication(
        run_store=cast(CalibrationRunStore, object()),
        checkpoint_store=cast(CalibrationCheckpointStore, object()),
        live_preflight=_unused_live_preflight(),
        result=result,
    )

    changed = SimpleNamespace(fingerprint="c" * 64)
    monkeypatch.setattr(
        failure_module,
        "reverify_published_scientific_task",
        lambda **_kwargs: SimpleNamespace(task_result=changed),
    )
    with pytest.raises(IntegrityError, match="scientific publication changed"):
        failure_module._reverify_dependency_publication(
            run_store=cast(CalibrationRunStore, object()),
            checkpoint_store=cast(CalibrationCheckpointStore, object()),
            live_preflight=_unused_live_preflight(),
            result=result,
        )


def test_dependency_reverification_covers_blocked_and_invalid_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        status="blocked",
        run_id="a" * 64,
        task_id="design_candidate_viability",
        fingerprint="b" * 64,
    )
    monkeypatch.setattr(
        failure_module,
        "reverify_exact_blocked_scientific_task",
        lambda **_kwargs: SimpleNamespace(task_result=result),
    )
    failure_module._reverify_dependency_publication(
        run_store=cast(CalibrationRunStore, object()),
        checkpoint_store=cast(CalibrationCheckpointStore, object()),
        live_preflight=_unused_live_preflight(),
        result=result,
    )

    monkeypatch.setattr(
        failure_module,
        "reverify_exact_blocked_scientific_task",
        lambda **_kwargs: SimpleNamespace(
            task_result=SimpleNamespace(fingerprint="c" * 64)
        ),
    )
    with pytest.raises(IntegrityError, match="dependency-block publication changed"):
        failure_module._reverify_dependency_publication(
            run_store=cast(CalibrationRunStore, object()),
            checkpoint_store=cast(CalibrationCheckpointStore, object()),
            live_preflight=_unused_live_preflight(),
            result=result,
        )

    with pytest.raises(IntegrityError, match="invalid terminal status"):
        failure_module._reverify_dependency_publication(
            run_store=cast(CalibrationRunStore, object()),
            checkpoint_store=cast(CalibrationCheckpointStore, object()),
            live_preflight=_unused_live_preflight(),
            result=SimpleNamespace(status="running"),
        )


def test_blocked_replay_and_prefix_helpers_reject_substituted_stores() -> None:
    with pytest.raises(ModelError, match="replay requires a run store"):
        reverify_exact_blocked_scientific_task(
            run_store=cast(CalibrationRunStore, object()),
            checkpoint_store=cast(CalibrationCheckpointStore, object()),
            live_preflight=_unused_live_preflight(),
            run_id="a" * 64,
            task_id="input_integrity",
        )
    store = SimpleNamespace(
        verify_task_results=lambda *_args, **_kwargs: {
            G3_GATE_ORDER[1]: object(),
        }
    )
    with pytest.raises(IntegrityError, match="exact registry prefix"):
        failure_module._exact_result_prefix(store, "a" * 64)


def test_blocked_publisher_rejects_checkpoint_store_and_completed_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _checkpoints, run_id, preflight = _planned_store(tmp_path)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        with pytest.raises(ModelError, match="requires a checkpoint store"):
            publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=cast(CalibrationCheckpointStore, object()),
                live_preflight=_unused_live_preflight(),
            )
        monkeypatch.setattr(
            failure_module,
            "_exact_result_prefix",
            lambda *_args, **_kwargs: {task_id: object() for task_id in G3_GATE_ORDER},
        )
        with pytest.raises(IntegrityError, match="already published"):
            publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=_checkpoints,
                live_preflight=_unused_live_preflight(),
            )


def test_blocked_publisher_rejects_fresh_result_that_does_not_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, checkpoints, run_id, preflight = _planned_store(tmp_path)
    _trust_direct_scientific_dependencies(monkeypatch)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        store.write_task_result(
            lease,
            task_id="input_integrity",
            status="failed",
            payload={"synthetic_typed_failure": True},
        )
        monkeypatch.setattr(
            failure_module,
            "reverify_exact_blocked_scientific_task",
            lambda **_kwargs: SimpleNamespace(
                task_result=SimpleNamespace(fingerprint="0" * 64)
            ),
        )
        with pytest.raises(IntegrityError, match="did not replay exactly"):
            publish_exact_next_blocked_scientific_task(
                run_store=store,
                lease=lease,
                checkpoint_store=checkpoints,
                live_preflight=_unused_live_preflight(),
            )


def test_blocked_replay_rejects_checkpoint_prefix_status_payload_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, checkpoints, run_id, _preflight = _planned_store(tmp_path)
    with pytest.raises(ModelError, match="requires a checkpoint store"):
        reverify_exact_blocked_scientific_task(
            run_store=store,
            checkpoint_store=cast(CalibrationCheckpointStore, object()),
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="input_integrity",
        )

    monkeypatch.setattr(
        store,
        "verify_task_results",
        lambda *_args, **_kwargs: {G3_GATE_ORDER[1]: object()},
    )
    with pytest.raises(IntegrityError, match="exact registry prefix"):
        reverify_exact_blocked_scientific_task(
            run_store=store,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="input_integrity",
        )

    passed = SimpleNamespace(status="passed")
    monkeypatch.setattr(
        store,
        "verify_task_results",
        lambda *_args, **_kwargs: {"input_integrity": passed},
    )
    with pytest.raises(IntegrityError, match="not dependency-blocked"):
        reverify_exact_blocked_scientific_task(
            run_store=store,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="input_integrity",
        )
    with pytest.raises(IntegrityError, match="result is missing"):
        reverify_exact_blocked_scientific_task(
            run_store=store,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="design_candidate_viability",
        )

    certificate = {
        "selected_n_states": None,
        "blocked_by": ["input_integrity"],
        "dependency_result_fingerprints": [["input_integrity", "a" * 64]],
        "prior_result_fingerprints": [],
        "reason": "upstream_dependency_not_passed",
    }
    fingerprint = failure_module._fingerprint(certificate)
    monkeypatch.setattr(
        failure_module,
        "_derive_block_certificate",
        lambda **_kwargs: (certificate, fingerprint),
    )
    blocked = SimpleNamespace(
        status="blocked",
        payload={},
        blocked_by=("input_integrity",),
        dependency_fingerprints={"input_integrity": "a" * 64},
    )
    monkeypatch.setattr(
        store,
        "verify_task_results",
        lambda *_args, **_kwargs: {"input_integrity": blocked},
    )
    with pytest.raises(IntegrityError, match="payload changed"):
        reverify_exact_blocked_scientific_task(
            run_store=store,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="input_integrity",
        )
    blocked.payload = failure_module._blocked_payload(certificate, fingerprint)
    blocked.blocked_by = ()
    with pytest.raises(IntegrityError, match="result lineage changed"):
        reverify_exact_blocked_scientific_task(
            run_store=store,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="input_integrity",
        )


def test_block_certificate_and_lease_require_exact_prefix_and_live_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, checkpoints, run_id, preflight = _planned_store(tmp_path)
    with pytest.raises(IntegrityError, match="exact prior prefix"):
        failure_module._derive_block_certificate(
            run_store=store,
            checkpoint_store=checkpoints,
            live_preflight=_unused_live_preflight(),
            run_id=run_id,
            task_id="design_candidate_viability",
            prior={},
        )
    with pytest.raises(ModelError, match="requires a run store"):
        failure_module._require_active_lease(
            cast(CalibrationRunStore, object()),
            cast(Any, object()),
        )
    with pytest.raises(IntegrityError, match="active launch lease"):
        failure_module._require_active_lease(store, cast(Any, object()))

    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight,
        workers=9,
    ) as lease:
        monkeypatch.setattr(
            store,
            "read_launch",
            lambda _run_id: SimpleNamespace(
                state="running",
                preflight_fingerprint="0" * 64,
                workers=lease.workers,
            ),
        )
        with pytest.raises(IntegrityError, match="launch identity changed"):
            failure_module._require_active_lease(store, lease)
