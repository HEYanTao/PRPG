"""Focused tests for the closed canonical G3 26-task composition root."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

import prpg.model.g3_success_execution as module
from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_assembly import DerivedG3GateDecision
from prpg.model.g3_checkpoint import (
    CalibrationCheckpointReference,
    CalibrationCheckpointStore,
)
from prpg.model.g3_evidence_store import G3EvidenceStore, LoadedG3EvidenceAuthority
from prpg.model.g3_exact_v2_finalization import ExactV2G3FinalizationResult
from prpg.model.g3_preflight import CANONICAL_MASTER_SEED, CanonicalG3Preflight
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.g3_success_bridge_stage import CanonicalG3BridgeStageResult
from prpg.model.g3_success_execution import (
    CanonicalG3SuccessExecutionResult,
    run_canonical_g3_success_execution,
)
from prpg.model.g3_success_production_stage import (
    CanonicalG3ProductionStageResult,
)
from prpg.model.g3_success_runner import CanonicalG3DesignStageResult
from prpg.model.g3_task_binding import ScientificTaskBinding
from prpg.model.g3_task_publication import ScientificTaskPublication
from prpg.model.input import CalibrationInput
from prpg.model.run_store import (
    CalibrationRunStore,
    G3TaskResult,
    LaunchLease,
    LaunchMarker,
)
from prpg.model.scientific_artifact import CANONICAL_SCIENTIFIC_VERSION
from prpg.model.scientific_artifact_pair import ScientificArtifactPairStore

_RUN = "a" * 64
_PREFLIGHT = "b" * 64


def _partial(value_type: type[Any], **values: object) -> Any:
    value = object.__new__(value_type)
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def _placeholder_publications(count: int) -> tuple[ScientificTaskPublication, ...]:
    return cast(
        tuple[ScientificTaskPublication, ...], tuple(object() for _ in range(count))
    )


def _call_kwargs() -> dict[str, object]:
    return {
        "preflight": object(),
        "run_store": object(),
        "checkpoint_store": object(),
        "lease": SimpleNamespace(run_id=_RUN),
        "master_seed": CANONICAL_MASTER_SEED,
        "scientific_version": CANONICAL_SCIENTIFIC_VERSION,
        "model_store": object(),
        "pair_store": object(),
        "evidence_store": object(),
    }


def _fresh_guard_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    run_store = CalibrationRunStore(tmp_path / "runs")
    preflight = _partial(
        CanonicalG3Preflight,
        fingerprint=_PREFLIGHT,
        calibration_input_fingerprint="c" * 64,
        workers=9,
    )
    calibration_input = _partial(CalibrationInput, fingerprint="c" * 64)
    lease = _partial(
        LaunchLease,
        store=run_store,
        run_id=_RUN,
        preflight_fingerprint=_PREFLIGHT,
        workers=9,
        _closed=False,
    )
    state: dict[str, object] = {
        "persisted_fingerprint": _PREFLIGHT,
        "marker": LaunchMarker(
            _RUN,
            "running",
            1,
            _PREFLIGHT,
            9,
            "d" * 64,
            tmp_path / "launch-state.json",
            False,
        ),
        "task_results": {},
    }
    monkeypatch.setattr(
        module,
        "verify_live_canonical_g3_preflight",
        lambda _preflight: None,
    )
    monkeypatch.setattr(
        module,
        "read_persisted_canonical_g3_preflight",
        lambda _store, *, run_id: SimpleNamespace(
            preflight=SimpleNamespace(fingerprint=state["persisted_fingerprint"])
        ),
    )
    monkeypatch.setattr(
        CalibrationRunStore,
        "read_launch",
        lambda _store, _run_id: state["marker"],
    )
    monkeypatch.setattr(
        CalibrationRunStore,
        "verify_task_results",
        lambda _store, _run_id, *, require_complete: state["task_results"],
    )
    arguments = {
        "calibration_input": calibration_input,
        "preflight": preflight,
        "run_store": run_store,
        "checkpoint_store": CalibrationCheckpointStore(tmp_path / "artifacts"),
        "lease": lease,
        "master_seed": CANONICAL_MASTER_SEED,
        "scientific_version": CANONICAL_SCIENTIFIC_VERSION,
        "model_store": ModelArtifactStore(tmp_path / "artifacts"),
        "pair_store": ScientificArtifactPairStore(tmp_path / "artifacts"),
        "evidence_store": G3EvidenceStore(tmp_path / "artifacts"),
    }
    return arguments, state


def _invoke_fresh_guard(arguments: dict[str, object]) -> None:
    module._verify_fresh_execution_inputs(
        cast(Any, arguments["calibration_input"]),
        preflight=cast(Any, arguments["preflight"]),
        run_store=cast(Any, arguments["run_store"]),
        checkpoint_store=cast(Any, arguments["checkpoint_store"]),
        lease=cast(Any, arguments["lease"]),
        master_seed=cast(Any, arguments["master_seed"]),
        scientific_version=cast(Any, arguments["scientific_version"]),
        model_store=cast(Any, arguments["model_store"]),
        pair_store=cast(Any, arguments["pair_store"]),
        evidence_store=cast(Any, arguments["evidence_store"]),
    )


def test_closed_executor_calls_exact_stages_in_order_and_propagates_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _call_kwargs()
    calibration_input = object()
    preflight = values["preflight"]
    design_publications = _placeholder_publications(10)
    production_publications = _placeholder_publications(9)
    bridge_publications = _placeholder_publications(6)
    artifact_publication = cast(ScientificTaskPublication, object())
    design = _partial(
        CanonicalG3DesignStageResult,
        publications=design_publications,
        candidate_selection=SimpleNamespace(selected_n_states=4),
        design_fit=SimpleNamespace(n_states=4),
    )
    production = _partial(
        CanonicalG3ProductionStageResult,
        publications=production_publications,
        design_stage=design,
        production_fit_view=SimpleNamespace(design_selected_k=4),
        production_fit=SimpleNamespace(n_states=4),
    )
    bridge = _partial(
        CanonicalG3BridgeStageResult,
        publications=bridge_publications,
    )
    first_25 = (*design_publications, *production_publications, *bridge_publications)
    finalization = _partial(
        ExactV2G3FinalizationResult,
        prefix=SimpleNamespace(selected_n_states=4),
        artifact_pair=SimpleNamespace(
            selected_k=4,
            design=SimpleNamespace(selected_k=4),
            production=SimpleNamespace(selected_k=4),
        ),
    )
    all_publications = (*first_25, artifact_publication)
    events: list[object] = []

    monkeypatch.setattr(
        module,
        "_verify_fresh_execution_inputs",
        lambda source, **kwargs: events.append(("inputs", source, kwargs)),
    )

    def verify_stage(
        publications: tuple[ScientificTaskPublication, ...],
        *,
        expected_tasks: tuple[str, ...],
        run_id: str,
    ) -> tuple[ScientificTaskPublication, ...]:
        expected = cast(tuple[str, ...], expected_tasks)
        events.append(("prefix", expected))
        assert run_id == _RUN
        return publications

    monkeypatch.setattr(module, "_verified_stage_publications", verify_stage)

    def design_stage(source: object, **kwargs: object) -> object:
        events.append("design")
        assert source is calibration_input
        _assert_common_stage_kwargs(kwargs, values, preflight=preflight)
        return design

    def production_stage(*args: object, **kwargs: object) -> object:
        events.append("production")
        assert args == (design, calibration_input)
        _assert_common_stage_kwargs(kwargs, values, preflight=preflight)
        return production

    def bridge_stage(*args: object, **kwargs: object) -> object:
        events.append("bridge")
        assert args == (design, production)
        _assert_common_stage_kwargs(kwargs, values, preflight=preflight)
        return bridge

    def finalize(**kwargs: object) -> object:
        events.append("finalize")
        assert kwargs == {
            "run_store": values["run_store"],
            "lease": values["lease"],
            "checkpoint_store": values["checkpoint_store"],
            "live_preflight": preflight,
            "publications": first_25,
            "model_store": values["model_store"],
            "pair_store": values["pair_store"],
            "evidence_store": values["evidence_store"],
        }
        return finalization

    monkeypatch.setattr(module, "run_canonical_g3_design_stage", design_stage)
    monkeypatch.setattr(module, "run_canonical_g3_production_stage", production_stage)
    monkeypatch.setattr(module, "run_canonical_g3_bridge_stage", bridge_stage)
    monkeypatch.setattr(module, "finalize_exact_v2_all_success", finalize)

    def verify_final(
        value: object,
        *,
        first_25: tuple[ScientificTaskPublication, ...],
        run_store: object,
        lease: object,
        preflight: object,
    ) -> tuple[ScientificTaskPublication, ...]:
        events.append("verify_finalization")
        assert value is finalization
        assert first_25 == all_publications[:25]
        assert all(
            left is right
            for left, right in zip(first_25, all_publications[:25], strict=True)
        )
        assert run_store is values["run_store"]
        assert lease is values["lease"]
        assert preflight is values["preflight"]
        return all_publications

    monkeypatch.setattr(
        module,
        "_verified_finalization_publications",
        verify_final,
    )

    result = run_canonical_g3_success_execution(
        cast(Any, calibration_input),
        **cast(Any, values),
    )

    assert isinstance(result, CanonicalG3SuccessExecutionResult)
    assert result.design_stage is design
    assert result.production_stage is production
    assert result.bridge_stage is bridge
    assert result.finalization is finalization
    assert result.first_25_publications == first_25
    assert result.publications == all_publications
    assert not result.generation_authorized
    assert events[1:] == [
        "design",
        ("prefix", G3_GATE_ORDER[:10]),
        "production",
        ("prefix", G3_GATE_ORDER[:19]),
        "bridge",
        ("prefix", G3_GATE_ORDER[:25]),
        "finalize",
        "verify_finalization",
    ]


def _assert_common_stage_kwargs(
    kwargs: dict[str, object],
    expected: dict[str, object],
    *,
    preflight: object,
) -> None:
    assert kwargs["launch_preflight"] is preflight
    assert kwargs["live_preflight"] is preflight
    for name in (
        "run_store",
        "checkpoint_store",
        "lease",
        "master_seed",
        "scientific_version",
    ):
        assert kwargs[name] is expected[name] or kwargs[name] == expected[name]


def test_public_executor_has_no_injection_or_recovery_surface() -> None:
    parameters = inspect.signature(run_canonical_g3_success_execution).parameters

    assert tuple(parameters) == (
        "calibration_input",
        "preflight",
        "run_store",
        "checkpoint_store",
        "lease",
        "master_seed",
        "scientific_version",
        "model_store",
        "pair_store",
        "evidence_store",
    )
    assert all(
        name not in parameters
        for name in (
            "launch_preflight",
            "live_preflight",
            "callbacks",
            "handler_map",
            "stage_map",
            "history_count",
            "strict_canonical",
            "recover",
            "resume",
        )
    )


def test_fresh_input_guard_accepts_exact_active_empty_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments, _state = _fresh_guard_arguments(monkeypatch, tmp_path)

    _invoke_fresh_guard(arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("master_seed", CANONICAL_MASTER_SEED + 1, "approved master seed 20260714"),
        ("master_seed", -1, "approved master seed 20260714"),
        (
            "scientific_version",
            CANONICAL_SCIENTIFIC_VERSION + 1,
            "canonical scientific version 11",
        ),
        ("scientific_version", -1, "canonical scientific version 11"),
    ),
    ids=(
        "wrong-positive-seed",
        "negative-seed",
        "wrong-positive-version",
        "negative-version",
    ),
)
def test_fresh_input_guard_rejects_noncanonical_seed_and_version_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    arguments, _state = _fresh_guard_arguments(monkeypatch, tmp_path)
    arguments[field] = value

    with pytest.raises(ModelError, match=message):
        _invoke_fresh_guard(arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("calibration_input", object(), "requires CalibrationInput"),
        ("run_store", object(), "requires CalibrationRunStore"),
        ("checkpoint_store", object(), "requires CalibrationCheckpointStore"),
        ("model_store", object(), "requires ModelArtifactStore"),
        ("pair_store", object(), "requires ScientificArtifactPairStore"),
        ("evidence_store", object(), "requires G3EvidenceStore"),
        ("master_seed", True, "approved master seed 20260714"),
        ("scientific_version", 0, "canonical scientific version 11"),
    ),
)
def test_fresh_input_guard_rejects_substituted_types_and_invalid_rng_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    arguments, _state = _fresh_guard_arguments(monkeypatch, tmp_path)
    arguments[field] = value

    with pytest.raises(ModelError, match=message):
        _invoke_fresh_guard(arguments)


@pytest.mark.parametrize("mismatch", ("input", "workers"))
def test_fresh_input_guard_rejects_input_or_worker_geometry_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    arguments, _state = _fresh_guard_arguments(monkeypatch, tmp_path)
    if mismatch == "input":
        object.__setattr__(arguments["calibration_input"], "fingerprint", "9" * 64)
    else:
        object.__setattr__(arguments["preflight"], "workers", 8)

    with pytest.raises(ModelError, match="input or worker geometry"):
        _invoke_fresh_guard(arguments)


def test_fresh_input_guard_requires_exact_active_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments, _state = _fresh_guard_arguments(monkeypatch, tmp_path)
    object.__setattr__(arguments["lease"], "_closed", True)

    with pytest.raises(ModelError, match="exact active launch"):
        _invoke_fresh_guard(arguments)


def test_fresh_input_guard_rejects_changed_persisted_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments, state = _fresh_guard_arguments(monkeypatch, tmp_path)
    state["persisted_fingerprint"] = "9" * 64

    with pytest.raises(IntegrityError, match="persisted launch parent"):
        _invoke_fresh_guard(arguments)


def test_fresh_input_guard_requires_exact_running_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments, state = _fresh_guard_arguments(monkeypatch, tmp_path)
    state["marker"] = LaunchMarker(
        _RUN,
        "completed",
        2,
        _PREFLIGHT,
        9,
        "d" * 64,
        tmp_path / "launch-state.json",
        False,
    )

    with pytest.raises(ModelError, match="exact running launch"):
        _invoke_fresh_guard(arguments)


def test_fresh_input_guard_rejects_nonempty_task_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments, state = _fresh_guard_arguments(monkeypatch, tmp_path)
    state["task_results"] = {G3_GATE_ORDER[0]: object()}

    with pytest.raises(ModelError, match="fresh task prefix"):
        _invoke_fresh_guard(arguments)


@pytest.mark.parametrize(
    "fault",
    (
        "selection",
        "design_fit",
        "production_design_parent",
        "production_fit",
    ),
)
def test_v3_pre_finalization_boundary_rejects_every_non_k4_lineage(
    fault: str,
) -> None:
    design = _partial(
        CanonicalG3DesignStageResult,
        candidate_selection=SimpleNamespace(selected_n_states=4),
        design_fit=SimpleNamespace(n_states=4),
    )
    production = _partial(
        CanonicalG3ProductionStageResult,
        production_fit_view=SimpleNamespace(design_selected_k=4),
        production_fit=SimpleNamespace(n_states=4),
    )
    targets = {
        "selection": (design.candidate_selection, "selected_n_states"),
        "design_fit": (design.design_fit, "n_states"),
        "production_design_parent": (
            production.production_fit_view,
            "design_selected_k",
        ),
        "production_fit": (production.production_fit, "n_states"),
    }
    target, field = targets[fault]
    setattr(target, field, 3)

    with pytest.raises(IntegrityError, match="requires owner-approved K=4"):
        module._require_owner_fixed_k4_before_finalization(design, production)


@pytest.mark.parametrize(
    "fault",
    ("prefix", "pair", "design_artifact", "production_artifact"),
)
def test_v3_post_finalization_boundary_rejects_every_non_k4_output(
    fault: str,
) -> None:
    prefix = SimpleNamespace(selected_n_states=4)
    pair = SimpleNamespace(
        selected_k=4,
        design=SimpleNamespace(selected_k=4),
        production=SimpleNamespace(selected_k=4),
    )
    finalization = _partial(
        ExactV2G3FinalizationResult,
        prefix=prefix,
        artifact_pair=pair,
    )
    targets = {
        "prefix": (prefix, "selected_n_states"),
        "pair": (pair, "selected_k"),
        "design_artifact": (pair.design, "selected_k"),
        "production_artifact": (pair.production, "selected_k"),
    }
    target, field = targets[fault]
    setattr(target, field, 3)

    with pytest.raises(IntegrityError, match="require owner-approved K=4"):
        module._require_owner_fixed_k4_after_finalization(finalization)


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("design_type", "design stage returned a substituted result type"),
        ("production_type", "production stage returned a substituted result type"),
        ("production_parent", "production stage changed its exact design parent"),
        ("bridge_type", "bridge stage returned a substituted result type"),
    ),
)
def test_closed_executor_rejects_substituted_stage_results_and_parentage(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    message: str,
) -> None:
    design = _partial(CanonicalG3DesignStageResult, publications=())
    production = _partial(
        CanonicalG3ProductionStageResult,
        publications=(),
        design_stage=design if fault != "production_parent" else object(),
    )
    bridge = _partial(CanonicalG3BridgeStageResult, publications=())
    monkeypatch.setattr(module, "_verify_fresh_execution_inputs", lambda *a, **k: None)
    monkeypatch.setattr(
        module,
        "_verified_stage_publications",
        lambda publications, **kwargs: publications,
    )
    monkeypatch.setattr(
        module,
        "run_canonical_g3_design_stage",
        lambda *a, **k: object() if fault == "design_type" else design,
    )
    monkeypatch.setattr(
        module,
        "run_canonical_g3_production_stage",
        lambda *a, **k: object() if fault == "production_type" else production,
    )
    monkeypatch.setattr(
        module,
        "run_canonical_g3_bridge_stage",
        lambda *a, **k: object() if fault == "bridge_type" else bridge,
    )

    with pytest.raises((ModelError, IntegrityError), match=message):
        run_canonical_g3_success_execution(
            cast(Any, object()),
            **cast(Any, _call_kwargs()),
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
def test_child_stage_fault_identity_escapes_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(
        module,
        "_verify_fresh_execution_inputs",
        lambda *_args, **_kwargs: None,
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(module, "run_canonical_g3_design_stage", fail)
    with pytest.raises(type(error)) as captured:
        run_canonical_g3_success_execution(
            cast(Any, object()),
            **cast(Any, _call_kwargs()),
        )

    assert captured.value is error


def test_reordered_stage_publications_are_refused() -> None:
    first = _scientific_publication(G3_GATE_ORDER[0], 0)
    second = _scientific_publication(G3_GATE_ORDER[1], 1)

    with pytest.raises(IntegrityError, match="reordered or substituted"):
        module._verified_stage_publications(
            (second, first),
            expected_tasks=G3_GATE_ORDER[:2],
            run_id=_RUN,
        )


@pytest.mark.parametrize(
    "publications",
    (
        [],
        (),
        (object(),),
    ),
    ids=("non_tuple", "wrong_count", "substituted_item"),
)
def test_stage_publication_verifier_rejects_shape_and_type_substitution(
    publications: object,
) -> None:
    with pytest.raises(IntegrityError, match="substituted publication"):
        module._verified_stage_publications(
            cast(Any, publications),
            expected_tasks=(G3_GATE_ORDER[0],),
            run_id=_RUN,
        )


def test_finalization_verifier_rejects_substituted_result_type() -> None:
    with pytest.raises(ModelError, match="substituted result type"):
        module._verified_finalization_publications(
            cast(Any, object()),
            first_25=(),
            run_store=cast(Any, object()),
            lease=cast(Any, object()),
            preflight=cast(Any, object()),
        )


def test_finalizer_must_return_the_exact_first_25_objects() -> None:
    first_25 = tuple(
        _scientific_publication(task_id, position)
        for position, task_id in enumerate(G3_GATE_ORDER[:25])
    )
    substituted = (
        _scientific_publication(G3_GATE_ORDER[0], 0),
        *first_25[1:],
    )
    finalization = _partial(
        ExactV2G3FinalizationResult,
        first_25_publications=substituted,
    )

    with pytest.raises(IntegrityError, match="object identity"):
        module._verified_finalization_publications(
            finalization,
            first_25=first_25,
            run_store=cast(Any, object()),
            lease=cast(Any, SimpleNamespace(run_id=_RUN)),
            preflight=cast(Any, object()),
        )


def test_finalization_verifier_requires_task_26_and_evidence_only_result() -> None:
    first_25 = tuple(
        _scientific_publication(task_id, position)
        for position, task_id in enumerate(G3_GATE_ORDER[:25])
    )
    durable = _partial(LoadedG3EvidenceAuthority)
    finalization = _partial(
        ExactV2G3FinalizationResult,
        first_25_publications=first_25,
        artifact_integrity_publication=_scientific_publication(
            "actual_artifact_bridge",
            25,
        ),
        evidence_bundle=SimpleNamespace(
            passed=True,
            generation_authorized=False,
        ),
        durable_evidence=durable,
    )

    with pytest.raises(IntegrityError, match="reordered or substituted"):
        module._verified_finalization_publications(
            finalization,
            first_25=first_25,
            run_store=cast(Any, object()),
            lease=cast(Any, SimpleNamespace(run_id=_RUN)),
            preflight=cast(Any, object()),
        )


def test_completed_verifier_rejects_a_noncompleted_launch() -> None:
    publications = tuple(
        _scientific_publication(task_id, position)
        for position, task_id in enumerate(G3_GATE_ORDER[:25])
    )
    durable = _partial(
        LoadedG3EvidenceAuthority,
        run_id=_RUN,
        completion_fingerprint="e" * 64,
    )
    completed = LaunchMarker(
        _RUN,
        "completed",
        2,
        _PREFLIGHT,
        9,
        "e" * 64,
        Path("/completion.json"),
        False,
    )
    finalization = _partial(
        ExactV2G3FinalizationResult,
        first_25_publications=publications,
        artifact_integrity_publication=_scientific_publication(
            G3_GATE_ORDER[-1],
            25,
        ),
        evidence_bundle=SimpleNamespace(
            passed=True,
            generation_authorized=False,
        ),
        completion_marker=completed,
        durable_evidence=durable,
    )
    running = LaunchMarker(
        _RUN,
        "running",
        1,
        _PREFLIGHT,
        9,
        "f" * 64,
        Path("/completion.json"),
        False,
    )
    run_store = SimpleNamespace(
        read_launch=lambda _run_id: running,
    )

    with pytest.raises(IntegrityError, match="durable evidence-only"):
        module._verified_finalization_publications(
            finalization,
            first_25=publications,
            run_store=cast(Any, run_store),
            lease=cast(Any, SimpleNamespace(run_id=_RUN)),
            preflight=cast(Any, SimpleNamespace(fingerprint=_PREFLIGHT)),
        )


def test_completed_verifier_returns_exact_26_publication_objects() -> None:
    first_25 = tuple(
        _scientific_publication(task_id, position)
        for position, task_id in enumerate(G3_GATE_ORDER[:25])
    )
    artifact_publication = _scientific_publication(G3_GATE_ORDER[-1], 25)
    completion = LaunchMarker(
        _RUN,
        "completed",
        2,
        _PREFLIGHT,
        9,
        "e" * 64,
        Path("/completion.json"),
        False,
    )
    finalization = _partial(
        ExactV2G3FinalizationResult,
        first_25_publications=first_25,
        artifact_integrity_publication=artifact_publication,
        evidence_bundle=SimpleNamespace(
            passed=True,
            generation_authorized=False,
        ),
        completion_marker=completion,
        durable_evidence=_partial(
            LoadedG3EvidenceAuthority,
            run_id=_RUN,
            completion_fingerprint=completion.fingerprint,
        ),
    )
    run_store = SimpleNamespace(read_launch=lambda _run_id: completion)

    publications = module._verified_finalization_publications(
        finalization,
        first_25=first_25,
        run_store=cast(Any, run_store),
        lease=cast(Any, SimpleNamespace(run_id=_RUN)),
        preflight=cast(Any, SimpleNamespace(fingerprint=_PREFLIGHT)),
    )

    assert len(publications) == 26
    assert all(
        actual is expected
        for actual, expected in zip(publications[:25], first_25, strict=True)
    )
    assert publications[-1] is artifact_publication


def _scientific_publication(task_id: str, position: int) -> ScientificTaskPublication:
    binding = _partial(
        ScientificTaskBinding,
        task_id=task_id,
        run_id=_RUN,
        fingerprint=f"{position + 1:064x}",
    )
    decision = _partial(
        DerivedG3GateDecision,
        gate_id=task_id,
        passed=True,
    )
    checkpoint_fingerprint = f"{position + 101:064x}"
    checkpoint = _partial(
        CalibrationCheckpointReference,
        task_id=task_id,
        binding=SimpleNamespace(run_id=_RUN),
        fingerprint=checkpoint_fingerprint,
    )
    result = G3TaskResult(
        run_id=_RUN,
        task_id=task_id,
        status="passed",
        payload=MappingProxyType({"checkpoint_fingerprint": checkpoint_fingerprint}),
        dependency_fingerprints=MappingProxyType({}),
        blocked_by=(),
        fingerprint=f"{position + 201:064x}",
        path=Path(f"/{task_id}.json"),
        reused=False,
    )
    return ScientificTaskPublication(
        binding=binding,
        decision=decision,
        checkpoint=checkpoint,
        task_result=result,
        serialized_binding_sha256=f"{position + 301:064x}",
    )
