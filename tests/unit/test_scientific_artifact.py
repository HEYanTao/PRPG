from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import prpg.model.scientific_artifact as scientific_artifact_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.calibration import DecodedReturnPools
from prpg.model.dependence import DependenceReference
from prpg.model.g3_registry import G3_GATE_ORDER, G3_PLANNED_LOOK_REGISTRY
from prpg.model.hmm import GaussianHMMFit, HMMParameters
from prpg.model.kernel_execution import KernelCalibrationExecution
from prpg.model.scientific_artifact import (
    CANONICAL_KERNEL_LOOKS,
    CANONICAL_SCIENTIFIC_VERSION,
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    ScientificModelArtifactDraft,
    ScientificReportPayload,
    draft_scientific_model_artifact_from_phase3_results,
    load_scientific_model_artifact,
    publish_scientific_model_artifact,
    required_scientific_arrays,
    required_scientific_documents,
    scientific_core_model_fingerprint,
    scientific_report_payload,
    scientific_report_planned_look_families,
    scientific_report_task_ids,
    verify_scientific_artifact_run_evidence,
)
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificGenerationPolicy,
)


def _binding() -> ScientificArtifactBinding:
    return ScientificArtifactBinding(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        calibration_run_fingerprint="d" * 64,
        preflight_fingerprint="e" * 64,
    )


def _provenance() -> ScientificArtifactProvenance:
    return ScientificArtifactProvenance(
        source_code_fingerprint="f" * 64,
        dependency_lock_fingerprint="1" * 64,
    )


def _policy(*, monthly: int = 4, daily: int = 20) -> ScientificGenerationPolicy:
    return ScientificGenerationPolicy(
        schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=monthly,
        daily_block_length=daily,
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
        neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


def _continuation(segments: tuple[int, ...]) -> np.ndarray[Any, Any]:
    result = np.ones(sum(segments), dtype=np.bool_)
    start = 0
    for length in segments:
        result[start] = False
        start += length
    return result


def _arrays(role: str, k: int = 2) -> dict[str, np.ndarray[Any, Any]]:
    monthly_segments = (193,) if role == "design" else (210, 7)
    daily_segments = (4_049,) if role == "design" else (4_404, 143)
    monthly_states = np.arange(sum(monthly_segments), dtype=np.int64) % k
    daily_states = np.arange(sum(daily_segments), dtype=np.int64) % k
    start = np.full(k, 1.0 / k, dtype=np.float64)
    transition = np.full((k, k), 0.1 / (k - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.9)
    means = np.arange(k * 4, dtype=np.float64).reshape(k, 4) / 10.0
    lambdas = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    return {
        "hmm_start_probabilities": start,
        "hmm_transition_matrix": transition,
        "hmm_state_means": means,
        "hmm_tied_covariance": np.eye(4, dtype=np.float64),
        "hmm_scaler_means": np.arange(4, dtype=np.float64),
        "hmm_scaler_standard_deviations": np.ones(4, dtype=np.float64),
        "canonical_to_original": np.arange(k, dtype=np.int64),
        "original_to_canonical": np.arange(k, dtype=np.int64),
        "monthly_source_states": monthly_states,
        "daily_source_states": daily_states,
        "monthly_continuation": _continuation(monthly_segments),
        "daily_continuation": _continuation(daily_segments),
        "monthly_state_counts": np.bincount(monthly_states, minlength=k),
        "daily_state_counts": np.bincount(daily_states, minlength=k),
        "monthly_pc1_loadings": np.asarray([1.0, 0.0, 0.0]),
        "daily_pc1_loadings": np.asarray([1.0, 0.0, 0.0]),
        "monthly_kernel_lambdas": lambdas,
        "daily_kernel_lambdas": lambdas,
        "monthly_kernel_means": np.zeros((k, 3), dtype=np.float64),
        "daily_kernel_means": np.zeros((k, 3), dtype=np.float64),
        "monthly_kernel_half_widths": np.full((k, 3), 0.0009),
        "daily_kernel_half_widths": np.full((k, 3), 0.0009),
        "monthly_kernel_state_counts": np.full(k, 10_000, dtype=np.int64),
        "daily_kernel_state_counts": np.full(k, 10_000, dtype=np.int64),
        "monthly_kernel_offsets": np.zeros((4, k, 3), dtype=np.float64),
        "daily_kernel_offsets": np.zeros((4, k, 3), dtype=np.float64),
    }


def _phase3_fit(k: int = 2) -> GaussianHMMFit:
    arrays = _arrays("design", k)
    fit = object.__new__(GaussianHMMFit)
    fields: dict[str, object] = {
        "parameters": HMMParameters(
            start_probabilities=arrays["hmm_start_probabilities"],
            transition_matrix=arrays["hmm_transition_matrix"],
            means=arrays["hmm_state_means"],
            tied_covariance=arrays["hmm_tied_covariance"],
        ),
        "decoded_states": arrays["monthly_source_states"],
        "canonical_to_original": arrays["canonical_to_original"],
        "original_to_canonical": arrays["original_to_canonical"],
        "n_observations": 193,
        "n_features": 4,
        "n_states": k,
        "lengths": (193,),
        "scaler_scope": "design_training",
        "scaler_means": arrays["hmm_scaler_means"],
        "scaler_standard_deviations": arrays["hmm_scaler_standard_deviations"],
    }
    for name, value in fields.items():
        object.__setattr__(fit, name, value)
    return fit


def _phase3_inputs(
    k: int = 2,
) -> tuple[
    GaussianHMMFit,
    DecodedReturnPools,
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    DependenceReference,
    DependenceReference,
]:
    arrays = _arrays("design", k)
    fit = _phase3_fit(k)
    pools = DecodedReturnPools(
        monthly_states=arrays["monthly_source_states"],
        daily_states=arrays["daily_source_states"],
        monthly_diagnostics=object(),
        daily_diagnostics=object(),
    )

    def reference(frequency: str) -> DependenceReference:
        return DependenceReference(
            frequency=frequency,  # type: ignore[arg-type]
            asset_center=np.zeros(3),
            asset_scale=np.ones(3),
            pc1_loadings=arrays[f"{frequency}_pc1_loadings"],
            labels=("component",),
            raw_values=np.zeros(1),
            transformed_values=np.zeros(1),
        )

    return (
        fit,
        pools,
        arrays["monthly_continuation"],
        arrays["daily_continuation"],
        reference("monthly"),
        reference("daily"),
    )


def _phase3_kernel(
    frequency: str,
    core_fingerprint: str,
    *,
    selected_length: int,
    k: int = 2,
) -> KernelCalibrationExecution:
    arrays = _arrays("design", k)
    kernel = object.__new__(KernelCalibrationExecution)
    fields: dict[str, object] = {
        "artifact": "design",
        "frequency": frequency,
        "n_states": k,
        "selected_length": selected_length,
        "model_artifact_fingerprint": core_fingerprint,
        "passed": True,
        "final_precision": SimpleNamespace(
            means=arrays[f"{frequency}_kernel_means"],
            annualized_simultaneous_half_widths=arrays[
                f"{frequency}_kernel_half_widths"
            ],
            state_observation_counts=arrays[f"{frequency}_kernel_state_counts"],
        ),
        "offset_grid": SimpleNamespace(
            lambdas=arrays[f"{frequency}_kernel_lambdas"],
            offsets=arrays[f"{frequency}_kernel_offsets"],
        ),
        "verification": SimpleNamespace(passed=True),
    }
    for name, value in fields.items():
        object.__setattr__(kernel, name, value)
    return kernel


def _authorize_phase3_kernel_test_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Limit schema-unit doubles to tests that explicitly waive executor setup."""

    def identity(kernel: KernelCalibrationExecution) -> SimpleNamespace:
        return SimpleNamespace(
            artifact=kernel.artifact,
            frequency=kernel.frequency,
            n_states=kernel.n_states,
            selected_length=kernel.selected_length,
        )

    monkeypatch.setattr(
        scientific_artifact_module,
        "kernel_calibration_execution_identity",
        identity,
    )


def _counts(document_type: str, role: str) -> dict[str, int]:
    rows = 193 if role == "design" else 217
    values = {
        "model_fit": {
            "observations": rows,
            "features": 4,
            "ordinary_restarts": 50,
        },
        "macro_stability": {"attempts": 100, "minimum_successes": 90},
        "latent_correctness": {
            "chains": 10_000,
            "measured_months": 600,
            "extension_months": 2_000,
            "cluster_bootstrap_replicates": 10_000,
            "cluster_resample_size": 10_000,
        },
        "refitted_adequacy": {"attempts": 1_000, "minimum_successes": 950},
        "transition_support": {"attempts": 100},
        "block_calibration": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "fragmentation": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "dependence": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "kernel_calibration": {
            "monthly_windows": 10_000,
            "daily_windows": 20_000,
            "monthly_verification_windows": 10_000,
            "daily_verification_windows": 20_000,
            "planned_looks": len(CANONICAL_KERNEL_LOOKS),
        },
        "adequacy_holm": {"cells": 4},
        "dependence_holm": {"cells": 4},
        "design_production_bridge": {
            "resource_benchmark_pairs": 100,
            "tier_a_pairs_per_dgp": 250,
            "tier_b_maximum_pairs_per_dgp": 10_000,
            "accelerator_audit_pairs_per_dgp": 100,
            "dgp_families": 2,
        },
        "candidate_selection": {
            "candidates": 4,
            "rolling_folds": 6,
            "rolling_scores_per_candidate": 72,
            "rolling_bootstrap_replicates": 10_000,
            "ordinary_restarts_per_fit": 50,
        },
        "sealed_holdout_veto": {"observations": 24, "segments": 2},
        "production_fit_same_k": {
            "observations": 217,
            "ordinary_restarts": 50,
        },
    }
    return values[document_type]


def _reports(role: str) -> dict[str, ScientificReportPayload]:
    result: dict[str, ScientificReportPayload] = {}
    for path, document_type in required_scientific_documents(role).items():
        tasks = scientific_report_task_ids(document_type, role)  # type: ignore[arg-type]
        families = scientific_report_planned_look_families(
            document_type,
            role,  # type: ignore[arg-type]
        )
        result[path] = ScientificReportPayload(
            document_type=document_type,
            counts=_counts(document_type, role),
            payload=scientific_report_payload(
                document_type=document_type,
                role=role,  # type: ignore[arg-type]
                task_result_fingerprints=dict.fromkeys(tasks, "2" * 64),
                checkpoint_fingerprints=dict.fromkeys(tasks, "3" * 64),
                planned_look_fingerprints=dict.fromkeys(families, ("4" * 64,)),
                metrics={"recorded": True},
            ),
        )
    return result


def _run_evidence() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, tuple[str, ...]],
]:
    return (
        dict.fromkeys(G3_GATE_ORDER, "2" * 64),
        dict.fromkeys(G3_GATE_ORDER, "3" * 64),
        dict.fromkeys(
            (item.family_id for item in G3_PLANNED_LOOK_REGISTRY), ("4" * 64,)
        ),
    )


def _draft(role: str = "design", k: int = 2) -> ScientificModelArtifactDraft:
    return ScientificModelArtifactDraft(
        role=role,  # type: ignore[arg-type]
        selected_k=k,
        binding=_binding(),
        provenance=_provenance(),
        generation_policy=_policy(),
        arrays=_arrays(role, k),
        reports=_reports(role),
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw(item) for item in value]
    return value


def _generic_clone(
    store: ModelArtifactStore,
    loaded: LoadedScientificModelArtifact,
    *,
    metadata_mutator: Callable[[dict[str, Any]], None] | None = None,
    arrays_mutator: Callable[[dict[str, np.ndarray[Any, Any]]], None] | None = None,
    documents_mutator: Callable[[dict[str, dict[str, Any]]], None] | None = None,
    provenance_mutator: Callable[[dict[str, Any]], None] | None = None,
    data_fingerprint: str | None = None,
) -> str:
    identity = loaded.reference.manifest["identity"]
    metadata = _thaw(identity["model_metadata"])
    arrays = {name: value.copy() for name, value in loaded.arrays.items()}
    documents = {path: _thaw(value) for path, value in loaded.reports.items()}
    provenance = _thaw(identity["provenance"])
    if metadata_mutator is not None:
        metadata_mutator(metadata)
    if arrays_mutator is not None:
        arrays_mutator(arrays)
    if documents_mutator is not None:
        documents_mutator(documents)
    if provenance_mutator is not None:
        provenance_mutator(provenance)
    reference = store.create(
        artifact_role=loaded.role,
        data_fingerprint=data_fingerprint or loaded.binding.processed_data_fingerprint,
        model_metadata=metadata,
        arrays=arrays,
        calibration_documents=documents,
        provenance=provenance,
    )
    return reference.fingerprint


@pytest.mark.parametrize("role,k", [("design", 2), ("production", 5)])
def test_typed_round_trip_has_exact_role_schema_and_never_authorizes(
    tmp_path: Path, role: str, k: int
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft(role, k))
    replay = load_scientific_model_artifact(store, loaded.reference.fingerprint)

    assert loaded.role == role
    assert loaded.selected_k == k
    assert set(loaded.arrays) == required_scientific_arrays()
    assert set(loaded.reports) == set(required_scientific_documents(role))
    assert loaded.geometry.monthly_rows == (193 if role == "design" else 217)
    assert loaded.geometry.daily_rows == (4_049 if role == "design" else 4_547)
    assert loaded.generation_policy == _policy()
    assert loaded.core_model_fingerprint != loaded.parameter_set_fingerprint
    assert loaded.generation_authorized is False
    assert replay.generation_authorized is False
    assert all(not array.flags.writeable for array in replay.arrays.values())
    assert replay.reference.fingerprint == loaded.reference.fingerprint
    assert (
        replay.reports["reports/model-fit.json"]["binding"]
        == (loaded.reference.manifest["identity"]["model_metadata"]["binding"])
    )
    assert (
        replay.reports["reports/model-fit.json"]["generation_policy"]
        == loaded.reference.manifest["identity"]["model_metadata"]["generation_policy"]
    )


def test_phase3_builder_derives_policy_from_exact_kernel_pair_and_alpha_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorize_phase3_kernel_test_doubles(monkeypatch)
    fit, pools, monthly_links, daily_links, monthly_reference, daily_reference = (
        _phase3_inputs()
    )
    core = scientific_core_model_fingerprint(
        role="design",
        fit=fit,
        pools=pools,
        monthly_continuation=monthly_links,
        daily_continuation=daily_links,
        monthly_dependence_reference=monthly_reference,
        daily_dependence_reference=daily_reference,
    )
    monthly = _phase3_kernel("monthly", core, selected_length=4)
    daily = _phase3_kernel("daily", core, selected_length=20)
    draft = draft_scientific_model_artifact_from_phase3_results(
        role="design",
        binding=_binding(),
        provenance=_provenance(),
        fit=fit,
        pools=pools,
        monthly_continuation=monthly_links,
        daily_continuation=daily_links,
        monthly_dependence_reference=monthly_reference,
        daily_dependence_reference=daily_reference,
        monthly_kernel=monthly,
        daily_kernel=daily,
        alpha_policy="kernel_mean_neutral",
        selected_neutral_lambda=0.5,
        reports=_reports("design"),
    )

    assert draft.generation_policy.monthly_block_length == 4
    assert draft.generation_policy.daily_block_length == 20
    assert draft.generation_policy.alpha_policy == "kernel_mean_neutral"
    assert draft.generation_policy.selected_neutral_lambda == 0.5
    loaded = publish_scientific_model_artifact(ModelArtifactStore(tmp_path), draft)
    assert loaded.generation_policy == draft.generation_policy

    with pytest.raises(ModelError, match="requires a registered lambda"):
        draft_scientific_model_artifact_from_phase3_results(
            role="design",
            binding=_binding(),
            provenance=_provenance(),
            fit=fit,
            pools=pools,
            monthly_continuation=monthly_links,
            daily_continuation=daily_links,
            monthly_dependence_reference=monthly_reference,
            daily_dependence_reference=daily_reference,
            monthly_kernel=monthly,
            daily_kernel=daily,
            alpha_policy="kernel_mean_neutral",
            selected_neutral_lambda=None,
            reports=_reports("design"),
        )


def test_public_artifact_api_and_builder_type_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ModelArtifactStore(tmp_path)
    with pytest.raises(ModelError, match="requires ModelArtifactStore"):
        publish_scientific_model_artifact(object(), _draft())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="requires a typed draft"):
        publish_scientific_model_artifact(store, object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="requires ModelArtifactStore"):
        load_scientific_model_artifact(object(), "a" * 64)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="type/role task binding"):
        scientific_report_task_ids("unregistered", "design")

    fit, pools, monthly_links, daily_links, monthly_reference, daily_reference = (
        _phase3_inputs()
    )
    core = scientific_core_model_fingerprint(
        role="design",
        fit=fit,
        pools=pools,
        monthly_continuation=monthly_links,
        daily_continuation=daily_links,
        monthly_dependence_reference=monthly_reference,
        daily_dependence_reference=daily_reference,
    )
    daily = _phase3_kernel("daily", core, selected_length=20)
    common: dict[str, Any] = {
        "role": "design",
        "binding": _binding(),
        "provenance": _provenance(),
        "fit": fit,
        "pools": pools,
        "monthly_continuation": monthly_links,
        "daily_continuation": daily_links,
        "monthly_dependence_reference": monthly_reference,
        "daily_dependence_reference": daily_reference,
        "daily_kernel": daily,
        "alpha_policy": "historical_vector",
        "selected_neutral_lambda": None,
        "reports": _reports("design"),
    }
    with pytest.raises(ModelError, match="lacks registered execution authority"):
        draft_scientific_model_artifact_from_phase3_results(
            **common,
            monthly_kernel=_phase3_kernel("monthly", core, selected_length=4),
        )
    with pytest.raises(ModelError, match="kernel result is not typed"):
        draft_scientific_model_artifact_from_phase3_results(
            **common,
            monthly_kernel=object(),
        )
    _authorize_phase3_kernel_test_doubles(monkeypatch)
    with pytest.raises(ModelError, match="role/K/core/result binding"):
        draft_scientific_model_artifact_from_phase3_results(
            **common,
            monthly_kernel=_phase3_kernel("daily", core, selected_length=4),
        )
    with pytest.raises(ModelError, match="requires GaussianHMMFit"):
        scientific_core_model_fingerprint(
            role="design",
            fit=object(),  # type: ignore[arg-type]
            pools=pools,
            monthly_continuation=monthly_links,
            daily_continuation=daily_links,
            monthly_dependence_reference=monthly_reference,
            daily_dependence_reference=daily_reference,
        )
    with pytest.raises(ModelError, match="requires DecodedReturnPools"):
        scientific_core_model_fingerprint(
            role="design",
            fit=fit,
            pools=object(),  # type: ignore[arg-type]
            monthly_continuation=monthly_links,
            daily_continuation=daily_links,
            monthly_dependence_reference=monthly_reference,
            daily_dependence_reference=daily_reference,
        )
    wrong_geometry_fit = _phase3_fit()
    object.__setattr__(wrong_geometry_fit, "n_observations", 192)
    with pytest.raises(ModelError, match="HMM role/K/geometry"):
        scientific_core_model_fingerprint(
            role="design",
            fit=wrong_geometry_fit,
            pools=pools,
            monthly_continuation=monthly_links,
            daily_continuation=daily_links,
            monthly_dependence_reference=monthly_reference,
            daily_dependence_reference=daily_reference,
        )
    with pytest.raises(ModelError, match="dependence references"):
        scientific_core_model_fingerprint(
            role="design",
            fit=fit,
            pools=pools,
            monthly_continuation=monthly_links,
            daily_continuation=daily_links,
            monthly_dependence_reference=daily_reference,
            daily_dependence_reference=monthly_reference,
        )


def test_loaded_scientific_artifact_has_no_public_forgeable_constructor() -> None:
    with pytest.raises(TypeError):
        LoadedScientificModelArtifact(  # type: ignore[call-arg]
            reference=None,
            role="design",
            selected_k=2,
            geometry=None,
            binding=None,
            provenance=None,
            arrays={},
            reports={},
            core_model_fingerprint="a" * 64,
            parameter_set_fingerprint="b" * 64,
        )


def test_loaded_reports_cross_bind_exact_completed_run_evidence(tmp_path: Path) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())
    tasks, checkpoints, looks = _run_evidence()

    verify_scientific_artifact_run_evidence(
        loaded,
        binding=_binding(),
        provenance=_provenance(),
        task_result_fingerprints=tasks,
        checkpoint_fingerprints=checkpoints,
        planned_look_fingerprints=looks,
    )

    changed_tasks = {**tasks, "design_predictive_selection": "9" * 64}
    with pytest.raises(IntegrityError, match="task-result bindings"):
        verify_scientific_artifact_run_evidence(
            loaded,
            binding=_binding(),
            provenance=_provenance(),
            task_result_fingerprints=changed_tasks,
            checkpoint_fingerprints=checkpoints,
            planned_look_fingerprints=looks,
        )
    with pytest.raises(IntegrityError, match="provenance"):
        verify_scientific_artifact_run_evidence(
            loaded,
            binding=_binding(),
            provenance=replace(_provenance(), source_code_fingerprint="9" * 64),
            task_result_fingerprints=tasks,
            checkpoint_fingerprints=checkpoints,
            planned_look_fingerprints=looks,
        )
    with pytest.raises(ModelError, match="requires a loaded artifact"):
        verify_scientific_artifact_run_evidence(
            object(),  # type: ignore[arg-type]
            binding=_binding(),
            provenance=_provenance(),
            task_result_fingerprints=tasks,
            checkpoint_fingerprints=checkpoints,
            planned_look_fingerprints=looks,
        )
    with pytest.raises(IntegrityError, match="binding differs"):
        verify_scientific_artifact_run_evidence(
            loaded,
            binding=replace(_binding(), config_fingerprint="9" * 64),
            provenance=_provenance(),
            task_result_fingerprints=tasks,
            checkpoint_fingerprints=checkpoints,
            planned_look_fingerprints=looks,
        )
    with pytest.raises(IntegrityError, match="checkpoint bindings"):
        verify_scientific_artifact_run_evidence(
            loaded,
            binding=_binding(),
            provenance=_provenance(),
            task_result_fingerprints=tasks,
            checkpoint_fingerprints={
                **checkpoints,
                "design_predictive_selection": "9" * 64,
            },
            planned_look_fingerprints=looks,
        )
    with pytest.raises(IntegrityError, match="planned-look bindings"):
        verify_scientific_artifact_run_evidence(
            loaded,
            binding=_binding(),
            provenance=_provenance(),
            task_result_fingerprints=tasks,
            checkpoint_fingerprints=checkpoints,
            planned_look_fingerprints={
                **looks,
                "kernel_design_monthly": ("9" * 64,),
            },
        )


@pytest.mark.parametrize("missing", sorted(required_scientific_arrays()))
def test_serializer_rejects_every_missing_required_array(
    tmp_path: Path, missing: str
) -> None:
    draft = _draft()
    arrays = dict(draft.arrays)
    del arrays[missing]
    with pytest.raises(ModelError, match="array allowlist"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path), replace(draft, arrays=arrays)
        )


@pytest.mark.parametrize(
    "role",
    ["design", "production"],
)
def test_serializer_rejects_every_missing_role_specific_document(
    tmp_path: Path, role: str
) -> None:
    draft = _draft(role)
    for missing in required_scientific_documents(role):
        reports = dict(draft.reports)
        del reports[missing]
        with pytest.raises(ModelError, match="report allowlist"):
            publish_scientific_model_artifact(
                ModelArtifactStore(tmp_path / missing.replace("/", "-")),
                replace(draft, reports=reports),
            )


def test_serializer_rejects_extra_or_wrong_typed_reports(tmp_path: Path) -> None:
    draft = _draft()
    extra = dict(draft.reports)
    extra["reports/extra.json"] = next(iter(extra.values()))
    with pytest.raises(ModelError, match="report allowlist"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "extra"), replace(draft, reports=extra)
        )
    wrong = dict(draft.reports)
    wrong["reports/model-fit.json"] = replace(
        wrong["reports/model-fit.json"], document_type="fragmentation"
    )
    with pytest.raises(ModelError, match="type does not match"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "wrong"), replace(draft, reports=wrong)
        )
    untyped: dict[str, Any] = dict(draft.reports)
    untyped["reports/model-fit.json"] = {"counts": {}}
    with pytest.raises(ModelError, match="not typed"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "untyped"),
            replace(draft, reports=untyped),
        )


def test_serializer_rejects_noncanonical_counts_and_empty_payload(
    tmp_path: Path,
) -> None:
    draft = _draft()
    reports = dict(draft.reports)
    report = reports["reports/model-fit.json"]
    reports["reports/model-fit.json"] = replace(
        report, counts={**report.counts, "observations": 192}
    )
    with pytest.raises(ModelError, match="counts are noncanonical"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "counts"), replace(draft, reports=reports)
        )
    reports = dict(draft.reports)
    reports["reports/model-fit.json"] = replace(
        reports["reports/model-fit.json"], payload={}
    )
    with pytest.raises(ModelError, match="payload fields are invalid"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "payload"), replace(draft, reports=reports)
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (object(), "payload is invalid"),
        (
            {
                "schema_version": 1,
                "status": "failed",
                "task_result_fingerprints": {
                    "design_candidate_viability": "2" * 64,
                    "design_predictive_selection": "2" * 64,
                },
                "checkpoint_fingerprints": {
                    "design_candidate_viability": "3" * 64,
                    "design_predictive_selection": "3" * 64,
                },
                "planned_look_fingerprints": {},
                "metrics": {"recorded": True},
                "result_fingerprint": "4" * 64,
            },
            "payload status",
        ),
    ],
)
def test_serializer_rejects_nonmapping_or_failed_report_payload(
    tmp_path: Path, payload: object, message: str
) -> None:
    draft = _draft()
    reports = dict(draft.reports)
    reports["reports/model-fit.json"] = replace(
        reports["reports/model-fit.json"],
        payload=payload,  # type: ignore[arg-type]
    )
    with pytest.raises(ModelError, match=message):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path), replace(draft, reports=reports)
        )


def test_serializer_rejects_empty_metrics_bad_result_hash_and_bad_counts(
    tmp_path: Path,
) -> None:
    draft = _draft()
    for name, mutation, message in (
        (
            "empty-metrics",
            lambda payload: payload.update({"metrics": {}}),
            "metrics must be non-empty",
        ),
        (
            "result-hash",
            lambda payload: payload.update({"result_fingerprint": "9" * 64}),
            "result fingerprint is inconsistent",
        ),
    ):
        reports = dict(draft.reports)
        report = reports["reports/model-fit.json"]
        payload = _thaw(report.payload)
        mutation(payload)
        reports["reports/model-fit.json"] = replace(report, payload=payload)
        with pytest.raises(ModelError, match=message):
            publish_scientific_model_artifact(
                ModelArtifactStore(tmp_path / name), replace(draft, reports=reports)
            )

    reports = dict(draft.reports)
    reports["reports/model-fit.json"] = replace(
        reports["reports/model-fit.json"],
        counts={"observations": 193.0},  # type: ignore[dict-item]
    )
    with pytest.raises(ModelError, match="counts are invalid"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "count-type"),
            replace(draft, reports=reports),
        )

    reports = dict(draft.reports)
    kernel_report = reports["reports/kernel-calibration.json"]
    reports["reports/kernel-calibration.json"] = replace(
        kernel_report, counts={"monthly_windows": 10_000}
    )
    with pytest.raises(ModelError, match="count fields are invalid"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "kernel-fields"),
            replace(draft, reports=reports),
        )

    reports = dict(draft.reports)
    reports["reports/kernel-calibration.json"] = replace(
        kernel_report,
        counts={**kernel_report.counts, "planned_looks": 5},
    )
    with pytest.raises(ModelError, match="counts are noncanonical"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "kernel-values"),
            replace(draft, reports=reports),
        )


def test_serializer_requires_an_exact_typed_generation_policy(tmp_path: Path) -> None:
    draft = _draft()
    with pytest.raises(ModelError, match="generation policy is not typed"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "untyped"),
            replace(draft, generation_policy=object()),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="monthly block length"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "invalid"),
            replace(
                draft,
                generation_policy=replace(
                    draft.generation_policy, monthly_block_length=13
                ),
            ),
        )


@pytest.mark.parametrize(
    "name,mutator,match",
    [
        (
            "hmm_start_probabilities",
            lambda value: np.asarray([0.8, 0.3]),
            "start probabilities",
        ),
        (
            "hmm_transition_matrix",
            lambda value: np.asarray([[0.8, 0.1], [0.2, 0.8]]),
            "transition matrix",
        ),
        (
            "hmm_tied_covariance",
            lambda value: np.diag([1e-8, 1.0, 1.0, 1.0]),
            "covariance floor",
        ),
        (
            "hmm_tied_covariance",
            lambda value: np.asarray(
                [
                    [1.0, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "not symmetric",
        ),
        (
            "hmm_scaler_standard_deviations",
            lambda value: np.asarray([1.0, 0.0, 1.0, 1.0]),
            "scaler standard",
        ),
        (
            "original_to_canonical",
            lambda value: np.asarray([0, 0]),
            "inverse permutations",
        ),
        (
            "monthly_source_states",
            lambda value: np.full(193, 2),
            "outside selected K",
        ),
        (
            "monthly_state_counts",
            lambda value: np.asarray([1, 192]),
            "counts are inconsistent",
        ),
        (
            "monthly_continuation",
            lambda value: np.ones(193, dtype=np.bool_),
            "continuation geometry",
        ),
        (
            "monthly_continuation",
            lambda value: np.ones(193, dtype=np.int64),
            "must be boolean",
        ),
        (
            "monthly_pc1_loadings",
            lambda value: np.asarray([-1.0, 0.0, 0.0]),
            "PC1 loading",
        ),
        (
            "monthly_kernel_lambdas",
            lambda value: np.asarray([0.2, 0.5, 0.75, 1.0]),
            "lambda grid",
        ),
        (
            "monthly_kernel_half_widths",
            lambda value: np.full((2, 3), 0.0011),
            "precision gate",
        ),
        (
            "monthly_kernel_state_counts",
            lambda value: np.asarray([0, 1]),
            "kernel state counts",
        ),
        (
            "monthly_kernel_means",
            lambda value: np.full((2, 3), np.nan),
            "non-finite",
        ),
        (
            "monthly_kernel_means",
            lambda value: np.full((2, 3), "invalid", dtype=object),
            "must be numeric",
        ),
    ],
)
def test_serializer_rejects_semantically_corrupt_arrays(
    tmp_path: Path,
    name: str,
    mutator: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    match: str,
) -> None:
    draft = _draft()
    arrays = dict(draft.arrays)
    arrays[name] = mutator(np.asarray(arrays[name]))
    with pytest.raises(ModelError, match=match):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path), replace(draft, arrays=arrays)
        )


def test_serializer_rejects_shape_kind_role_k_and_binding_errors(
    tmp_path: Path,
) -> None:
    draft = _draft()
    arrays = dict(draft.arrays)
    arrays["hmm_state_means"] = np.zeros((2, 3))
    with pytest.raises(ModelError, match="shape"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "shape"), replace(draft, arrays=arrays)
        )
    arrays = dict(draft.arrays)
    arrays["monthly_source_states"] = np.full(193, 0.5)
    with pytest.raises(ModelError, match="integer dtype"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "kind"), replace(draft, arrays=arrays)
        )
    with pytest.raises(ModelError, match="selected K"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "k"), replace(draft, selected_k=6)
        )
    with pytest.raises(ModelError, match="role"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "role"),
            replace(draft, role="control"),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="contract binding"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "contract"),
            replace(
                draft,
                binding=replace(draft.binding, scientific_version=1),
            ),
        )
    with pytest.raises(ModelError, match="producer"):
        publish_scientific_model_artifact(
            ModelArtifactStore(tmp_path / "producer"),
            replace(draft, provenance=replace(draft.provenance, producer="custom")),
        )


def test_version_1_artifact_identity_cannot_be_mixed_into_version_2(
    tmp_path: Path,
) -> None:
    draft = _draft()

    assert draft.binding.scientific_version == CANONICAL_SCIENTIFIC_VERSION == 11
    for suffix, binding in (
        (
            "v1-contract",
            replace(
                draft.binding,
                contract_version="sections-4.5-4.6-owner-approved-20260714-v1",
            ),
        ),
        ("v1-scientific-version", replace(draft.binding, scientific_version=1)),
    ):
        with pytest.raises(ModelError, match="contract binding"):
            publish_scientific_model_artifact(
                ModelArtifactStore(tmp_path / suffix),
                replace(draft, binding=binding),
            )


def test_typed_loader_rejects_reauthenticated_array_schema_corruption(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())

    missing = _generic_clone(
        store,
        loaded,
        arrays_mutator=lambda arrays: arrays.pop("daily_kernel_offsets"),
    )
    with pytest.raises(IntegrityError, match="array allowlist"):
        load_scientific_model_artifact(store, missing)

    def wrong_dtype(arrays: dict[str, np.ndarray[Any, Any]]) -> None:
        arrays["hmm_state_means"] = arrays["hmm_state_means"].astype(np.float32)

    dtype = _generic_clone(store, loaded, arrays_mutator=wrong_dtype)
    with pytest.raises(IntegrityError, match="dtype"):
        load_scientific_model_artifact(store, dtype)


def test_typed_loader_rejects_every_reauthenticated_missing_document(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())
    for path in required_scientific_documents("design"):
        fingerprint = _generic_clone(
            store,
            loaded,
            documents_mutator=lambda documents, selected=path: documents.pop(selected),
        )
        with pytest.raises(IntegrityError, match="report allowlist"):
            load_scientific_model_artifact(store, fingerprint)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda document: document.pop("counts"),
            "fields are invalid",
        ),
        (
            lambda document: document["counts"].update({"observations": 192}),
            "counts are noncanonical",
        ),
        (
            lambda document: document["binding"].update(
                {"config_fingerprint": "9" * 64}
            ),
            "cross-fingerprint",
        ),
        (
            lambda document: document.update({"document_type": "fragmentation"}),
            "path/type binding",
        ),
        (
            lambda document: document.update({"schema_version": 1}),
            "report schema is unsupported",
        ),
        (
            lambda document: document.update({"selected_k": 3}),
            "role/K binding",
        ),
        (
            lambda document: document.update({"parameter_set_fingerprint": "9" * 64}),
            "parameter binding",
        ),
    ],
)
def test_typed_loader_rejects_reauthenticated_report_corruption(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    match: str,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())

    def mutate(documents: dict[str, dict[str, Any]]) -> None:
        mutation(documents["reports/model-fit.json"])

    fingerprint = _generic_clone(store, loaded, documents_mutator=mutate)
    with pytest.raises(IntegrityError, match=match):
        load_scientific_model_artifact(store, fingerprint)


def test_typed_loader_rejects_metadata_data_and_provenance_cross_corruption(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())

    def bad_metadata(metadata: dict[str, Any]) -> None:
        metadata["parameter_set_fingerprint"] = "9" * 64

    parameter = _generic_clone(store, loaded, metadata_mutator=bad_metadata)
    with pytest.raises(IntegrityError, match="parameter-set fingerprint"):
        load_scientific_model_artifact(store, parameter)

    data = _generic_clone(store, loaded, data_fingerprint="9" * 64)
    with pytest.raises(IntegrityError, match="processed-data bindings"):
        load_scientific_model_artifact(store, data)

    provenance = _generic_clone(
        store,
        loaded,
        provenance_mutator=lambda value: value.update({"producer": "custom"}),
    )
    with pytest.raises(IntegrityError, match="producer"):
        load_scientific_model_artifact(store, provenance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata.update({"artifact_role": "production"}),
            "role bindings disagree",
        ),
        (
            lambda metadata: metadata.update(
                {"feature_names": list(reversed(metadata["feature_names"]))}
            ),
            "feature order",
        ),
        (
            lambda metadata: metadata.update(
                {"asset_names": list(reversed(metadata["asset_names"]))}
            ),
            "asset order",
        ),
        (
            lambda metadata: metadata.update({"core_model_fingerprint": "9" * 64}),
            "core-model fingerprint",
        ),
        (
            lambda metadata: metadata["geometry"].update({"monthly_rows": 192}),
            "geometry is noncanonical",
        ),
        (
            lambda metadata: metadata.update({"binding": None}),
            "binding is missing",
        ),
        (
            lambda metadata: metadata.update({"generation_policy": None}),
            "policy is missing",
        ),
    ],
)
def test_typed_loader_rejects_reauthenticated_metadata_semantic_corruption(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())
    fingerprint = _generic_clone(store, loaded, metadata_mutator=mutation)
    with pytest.raises(IntegrityError, match=message):
        load_scientific_model_artifact(store, fingerprint)


def test_typed_loader_rejects_old_tampered_and_swapped_generation_policy(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())

    old = _generic_clone(
        store,
        loaded,
        metadata_mutator=lambda metadata: metadata.pop("generation_policy"),
    )
    with pytest.raises(IntegrityError, match="metadata fields are invalid"):
        load_scientific_model_artifact(store, old)

    stale = _generic_clone(
        store,
        loaded,
        metadata_mutator=lambda metadata: metadata.update({"schema_version": 1}),
    )
    with pytest.raises(IntegrityError, match="schema is unsupported"):
        load_scientific_model_artifact(store, stale)

    def tamper(metadata: dict[str, Any]) -> None:
        metadata["generation_policy"]["selected_neutral_lambda"] = 0.5

    tampered = _generic_clone(store, loaded, metadata_mutator=tamper)
    with pytest.raises(IntegrityError, match="cannot select a lambda"):
        load_scientific_model_artifact(store, tampered)

    def swap_frequencies(metadata: dict[str, Any]) -> None:
        policy = metadata["generation_policy"]
        policy["monthly_block_length"] = 8
        policy["daily_block_length"] = 4

    swapped = _generic_clone(store, loaded, metadata_mutator=swap_frequencies)
    with pytest.raises(IntegrityError, match="generation-policy binding"):
        load_scientific_model_artifact(store, swapped)


def test_typed_loader_rejects_stale_or_tampered_report_policy_binding(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())

    def stale_report(documents: dict[str, dict[str, Any]]) -> None:
        report = documents["reports/model-fit.json"]
        report["generation_policy"]["alpha_policy"] = "kernel_mean_neutral"
        report["generation_policy"]["selected_neutral_lambda"] = 0.5

    stale = _generic_clone(store, loaded, documents_mutator=stale_report)
    with pytest.raises(IntegrityError, match="generation-policy binding"):
        load_scientific_model_artifact(store, stale)

    def tampered_report(documents: dict[str, dict[str, Any]]) -> None:
        documents["reports/model-fit.json"]["generation_policy"].pop(
            "neutral_lambda_grid"
        )

    tampered = _generic_clone(store, loaded, documents_mutator=tampered_report)
    with pytest.raises(IntegrityError, match="policy keys are invalid"):
        load_scientific_model_artifact(store, tampered)


def test_generic_integrity_corruption_is_still_rejected_before_typed_loading(
    tmp_path: Path,
) -> None:
    store = ModelArtifactStore(tmp_path)
    loaded = publish_scientific_model_artifact(store, _draft())
    parameter_path = loaded.reference.path / "model-parameters.npz"
    parameter_path.write_bytes(parameter_path.read_bytes() + b"tamper")
    with pytest.raises(IntegrityError, match="integrity"):
        load_scientific_model_artifact(store, loaded.reference.fingerprint)
