from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import tests.unit.test_g3_calibration_views as calibration_fixtures

import prpg.model.adequacy_gate as gate_module
import prpg.model.g3_assembly as assembly_module
import prpg.model.g3_calibration_views as calibration_views_module
import prpg.model.production_fit_execution as production_execution_module
import prpg.model.refit_adequacy_execution as refit_execution
from prpg.errors import ModelError
from prpg.model.adequacy import EndpointDirection, EndpointVector
from prpg.model.adequacy_execution import (
    RegisteredLatentCell,
    run_registered_latent_cell,
)
from prpg.model.adequacy_gate import (
    AdequacyCellTaskView,
    FourCellAdequacyHolmView,
    assemble_four_cell_adequacy_gate,
    assemble_four_cell_adequacy_holm_view,
    build_latent_adequacy_task_view,
    build_refitted_adequacy_task_view,
    is_adequacy_cell_task_view,
    is_four_cell_adequacy_gate,
    is_four_cell_adequacy_holm_view,
)
from prpg.model.g3_assembly import (
    derive_g3_gate_decision,
    task_view_cross_binding_error,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    ProductionFitSameKView,
    build_candidate_selection_task_view,
    build_candidate_viability_task_view,
    build_production_fit_same_k_view,
    production_fit_sources_from_view,
    selected_design_fit_from_view,
)
from prpg.model.g3_task_publication import scientific_task_source_slots
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
    forward_backward,
)
from prpg.model.production_fit_execution import (
    RegisteredProductionFitSameK,
    execute_registered_production_fit_same_k,
)
from prpg.model.refit_adequacy import RefittedAdequacyEndpoint
from prpg.model.refit_adequacy_execution import (
    RegisteredRefittedAdequacyBatch,
    run_registered_refitted_adequacy_cell,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    ModelFitScope,
)


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _make_parents(
    run_id: str,
    suffix: str,
) -> tuple[CandidateSelectionTaskView, ProductionFitSameKView]:
    viability_result, selection_result = calibration_fixtures._candidate_stages()
    viability = build_candidate_viability_task_view(
        viability_result,
        master_seed=calibration_fixtures._MASTER_SEED,
        scientific_version=calibration_fixtures._SCIENTIFIC_VERSION,
        run_id=run_id,
        binding_fingerprint=_fp(f"{run_id}-{suffix}-viability-parent"),
    )
    selection = build_candidate_selection_task_view(
        viability,
        selection_result,
        run_id=run_id,
        binding_fingerprint=_fp(f"{run_id}-{suffix}-selection-parent"),
    )
    production_features = calibration_fixtures._production_features()

    def exact_production_fit(
        features: HMMFeatureMatrix,
        *,
        selected_n_states: int,
        master_seed: int,
        scientific_version: int,
    ) -> GaussianHMMFit:
        base = calibration_fixtures._registered_fake_production_fit(
            features,
            selected_n_states=selected_n_states,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )
        decoded = decode_viterbi(
            features.values,
            base.parameters,
            lengths=features.lengths,
        )
        posterior = forward_backward(
            features.values,
            base.parameters,
            lengths=features.lengths,
        )
        return replace(
            base,
            decoded_states=decoded,
            log_likelihood=posterior.total_log_likelihood,
            forward_backward=posterior,
        )

    original = production_execution_module.fit_production_selected_k
    production_execution_module.fit_production_selected_k = exact_production_fit
    try:
        execution = execute_registered_production_fit_same_k(
            selection,
            production_features,
            master_seed=calibration_fixtures._MASTER_SEED,
            scientific_version=calibration_fixtures._SCIENTIFIC_VERSION,
        )
        assert isinstance(execution, RegisteredProductionFitSameK)
        production = build_production_fit_same_k_view(
            execution,
            run_id=run_id,
            binding_fingerprint=_fp(f"{run_id}-{suffix}-production-parent"),
        )
    finally:
        production_execution_module.fit_production_selected_k = original
    return selection, production


@lru_cache(maxsize=16)
def _parents(
    run_id: str,
) -> tuple[CandidateSelectionTaskView, ProductionFitSameKView]:
    return _make_parents(run_id, "canonical")


@pytest.fixture(autouse=True)
def _authorize_legacy_selection_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Stand in only for fixed-point wrapping in these focused adequacy fixtures."""

    _parents.cache_clear()
    monkeypatch.setattr(
        calibration_views_module,
        "is_registered_candidate_selection_task_view",
        calibration_views_module.is_candidate_selection_task_view,
    )
    yield
    _parents.cache_clear()


def _parent_for_role(
    role: str,
    *,
    run_id: str | None = None,
) -> CandidateSelectionTaskView | ProductionFitSameKView:
    parents = _parents(_fp("run-a") if run_id is None else run_id)
    return parents[0] if role == "design" else parents[1]


def _latent(
    artifact: CalibrationArtifact,
    variant: float,
    *,
    parent: CandidateSelectionTaskView | ProductionFitSameKView | None = None,
) -> RegisteredLatentCell:
    role = (
        "design" if artifact is CalibrationArtifact.DESIGN_OR_CONTROL else "production"
    )
    source_parent = _parent_for_role(role) if parent is None else parent
    fit = (
        selected_design_fit_from_view(source_parent)
        if isinstance(source_parent, CandidateSelectionTaskView)
        else production_fit_sources_from_view(source_parent)[0]
    )
    return run_registered_latent_cell(
        fit,
        master_seed=calibration_fixtures._MASTER_SEED + round(variant * 1_000),
        scientific_version=calibration_fixtures._SCIENTIFIC_VERSION,
        artifact=artifact,
        chain_count=20,
        measured_months=25,
        extension_months=100,
        bootstrap_replicates=30,
        resample_size=20,
        survival_months=5,
    )


def _test_refit_fit(*_args: object, **_kwargs: object) -> GaussianHMMFit:
    raise AssertionError("registered refit fixture replaces the one-attempt executor")


def _pvalue_geometry(p_value: float) -> tuple[np.ndarray, float]:
    for denominator in (10, 100, 1_000):
        raw_count = p_value * denominator - 1.0
        count = round(raw_count)
        if abs(raw_count - count) <= 1e-12 and 0 <= count < denominator:
            attempts = denominator - 1
            break
    else:  # pragma: no cover - fixture inputs are closed in this module.
        raise AssertionError("fixture p-value is not an empirical-grid value")
    values = np.random.default_rng(4_001 + attempts).normal(size=attempts)
    values += np.linspace(-3.0, 5.0, attempts, dtype=np.float64)
    center = float(values.mean())
    deviations = np.sort(np.abs(values - center))[::-1]
    if count == 0:
        threshold = float(deviations[0] + 1.0)
    else:
        threshold = float((deviations[count - 1] + deviations[count]) / 2.0)
    return values, center + threshold


def _refit(
    role: str,
    p_value: float,
    *,
    parent: CandidateSelectionTaskView | ProductionFitSameKView | None = None,
) -> RegisteredRefittedAdequacyBatch:
    source_parent = _parent_for_role(role) if parent is None else parent
    if role == "design":
        if not isinstance(source_parent, CandidateSelectionTaskView):
            raise AssertionError("design fixture requires selection parent")
        fit = selected_design_fit_from_view(source_parent)
        source_states = np.resize(
            np.repeat(np.asarray([0, 1], dtype=np.int64), 10),
            fit.n_observations,
        )
        values = fit.parameters.means[source_states] + np.random.default_rng(
            fit.n_observations
        ).normal(
            0.0,
            0.05,
            size=(fit.n_observations, fit.n_features),
        )
        features = HMMFeatureMatrix(
            values,
            fit.lengths,
            fit.feature_names,
            tuple(f"design-{row}" for row in range(fit.n_observations)),
            np.asarray(fit.scaler_means),
            np.asarray(fit.scaler_standard_deviations),
            "design_training",
        )
        scope = ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY
    else:
        if not isinstance(source_parent, ProductionFitSameKView):
            raise AssertionError("production fixture requires production parent")
        fit, features = production_fit_sources_from_view(source_parent)
        scope = ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY
    replicate_values, observed_value = _pvalue_geometry(p_value)
    attempt_count = len(replicate_values)
    first_seed_to_attempt = {
        deterministic_hmm_restart_seeds(
            master_seed=calibration_fixtures._MASTER_SEED,
            scientific_version=calibration_fixtures._SCIENTIFIC_VERSION,
            scope=scope,
            replicate=attempt,
            n_states=fit.n_states,
        )[0]: attempt
        for attempt in range(attempt_count)
    }

    def observed(*_args: object, **_kwargs: object) -> RefittedAdequacyEndpoint:
        values = np.asarray([observed_value], dtype=np.float64)
        values.setflags(write=False)
        return RefittedAdequacyEndpoint(
            EndpointVector(("fixture/metric",), values),
            (EndpointDirection.TWO_SIDED,),
            np.zeros(fit.n_observations, dtype=np.int64),
            np.zeros((fit.n_states, fit.n_features, 1), dtype=np.int64),
        )

    def attempt(
        _fit: GaussianHMMFit,
        _features: HMMFeatureMatrix,
        _uniforms: np.ndarray,
        _normals: np.ndarray,
        *,
        scope: ModelFitScope,
        seeds: Sequence[int],
        fit_function: object,
    ) -> object:
        del scope
        assert fit_function is _test_refit_fit
        index = first_seed_to_attempt[int(seeds[0])]
        values = np.asarray([replicate_values[index]], dtype=np.float64)
        endpoint = RefittedAdequacyEndpoint(
            EndpointVector(("fixture/metric",), values),
            (EndpointDirection.TWO_SIDED,),
            np.zeros(fit.n_observations, dtype=np.int64),
            np.zeros((fit.n_states, fit.n_features, 1), dtype=np.int64),
        )
        return SimpleNamespace(
            endpoint=endpoint,
            refit=SimpleNamespace(best_restart_index=0, best_seed=int(seeds[0])),
            alignment=SimpleNamespace(
                candidate_to_reference=np.arange(fit.n_states, dtype=np.int64),
                selected_total_cost=0.0,
            ),
        )

    original_observed = refit_execution.build_refitted_adequacy_endpoint
    original_attempt = refit_execution.run_refitted_adequacy_attempt
    refit_execution.build_refitted_adequacy_endpoint = observed
    refit_execution.run_refitted_adequacy_attempt = attempt
    try:
        result = run_registered_refitted_adequacy_cell(
            role=role,  # type: ignore[arg-type]
            original_fit=fit,
            artifact_features=features,
            master_seed=calibration_fixtures._MASTER_SEED,
            scientific_version=calibration_fixtures._SCIENTIFIC_VERSION,
            attempt_count=attempt_count,
            minimum_successes=2,
            strict_canonical=False,
            fit_function=_test_refit_fit,
        )
    finally:
        refit_execution.build_refitted_adequacy_endpoint = original_observed
        refit_execution.run_refitted_adequacy_attempt = original_attempt
    assert result.max_statistic_cell is not None
    assert result.max_statistic_cell.p_value == p_value
    return result


def _cell_views(
    *,
    run_id: str | None = None,
    p_values: tuple[float, float, float, float] = (0.8, 0.7, 0.6, 0.5),
    strict_canonical: bool = False,
) -> tuple[
    AdequacyCellTaskView,
    AdequacyCellTaskView,
    AdequacyCellTaskView,
    AdequacyCellTaskView,
]:
    bound_run = _fp("run-a") if run_id is None else run_id
    design_parent, production_parent = _parents(bound_run)
    return (
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=_latent(
                CalibrationArtifact.DESIGN_OR_CONTROL,
                p_values[0],
                parent=design_parent,
            ),
            parent=design_parent,
            run_id=bound_run,
            binding_fingerprint=_fp("binding-design-latent"),
            strict_canonical=strict_canonical,
        ),
        build_latent_adequacy_task_view(
            task_id="production_latent_correctness",
            evidence=_latent(
                CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
                p_values[1],
                parent=production_parent,
            ),
            parent=production_parent,
            run_id=bound_run,
            binding_fingerprint=_fp("binding-production-latent"),
            strict_canonical=strict_canonical,
        ),
        build_refitted_adequacy_task_view(
            task_id="design_refitted_adequacy",
            evidence=_refit("design", p_values[2], parent=design_parent),
            parent=design_parent,
            run_id=bound_run,
            binding_fingerprint=_fp("binding-design-refit"),
            strict_canonical=strict_canonical,
        ),
        build_refitted_adequacy_task_view(
            task_id="production_refitted_adequacy",
            evidence=_refit("production", p_values[3], parent=production_parent),
            parent=production_parent,
            run_id=bound_run,
            binding_fingerprint=_fp("binding-production-refit"),
            strict_canonical=strict_canonical,
        ),
    )


def _holm_view(
    views: tuple[
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
        AdequacyCellTaskView,
    ],
) -> FourCellAdequacyHolmView:
    return assemble_four_cell_adequacy_holm_view(
        design_latent=views[0],
        production_latent=views[1],
        design_refitted=views[2],
        production_refitted=views[3],
        binding_fingerprint=_fp("binding-holm"),
        strict_canonical=False,
    )


def test_v2_publication_slots_bind_split_latent_rng_and_holm_without_own_seal() -> None:
    views = _cell_views(strict_canonical=False)
    for view in views:
        slots = scientific_task_source_slots(view)
        materializations = dict(slots.materialization_fingerprints)
        assert len(materializations["task_source_materialization"]) == 64
        assert materializations["task_source_materialization"] != (
            view.content_fingerprint
        )
        rng = dict(slots.rng_fingerprints)
        if "latent_correctness" in view.task_id:
            assert rng == {
                f"{view.role}_latent_simulation_streams": (
                    view.simulation_rng_execution_fingerprint
                ),
                f"{view.role}_latent_bootstrap_streams": (
                    view.bootstrap_rng_execution_fingerprint
                ),
            }
        else:
            assert rng == {
                f"{view.role}_refit_adequacy_streams": (view.rng_execution_fingerprint)
            }

    holm = _holm_view(views)
    slots = scientific_task_source_slots(holm)
    materializations = dict(slots.materialization_fingerprints)
    assert len(materializations["four_cell_adequacy_parent_materialization"]) == 64
    assert materializations["task_source_materialization"] != holm.content_fingerprint
    with pytest.raises(ModelError, match="code-owned binding-slot derivation"):
        scientific_task_source_slots(copy.copy(holm))


def test_g3_assembly_uses_one_exact_adequacy_cell_per_task_then_holm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_views = _cell_views(strict_canonical=False)
    canonical: list[AdequacyCellTaskView] = []
    for view in raw_views:
        latent = "latent" in view.task_id
        repetitions = 10_000 if latent else 1_000
        successes = 10_000 if latent else 950
        canonical.append(
            replace(
                view,
                cell=replace(
                    view.cell,
                    ready=True,
                    repetitions=repetitions,
                    successful_repetitions=successes,
                    failure_reason=None,
                ),
                canonical_count_ready=True,
                ready=True,
                passed=True,
                strict_canonical=True,
            )
        )
    source_holm = _holm_view(raw_views)
    canonical_holm = replace(
        source_holm,
        source_views=tuple(canonical),
        source_content_fingerprints=tuple(
            view.content_fingerprint for view in canonical
        ),
        ready=True,
        passed=True,
        failure_reasons=(),
        strict_canonical=True,
    )
    monkeypatch.setattr(
        assembly_module,
        "is_adequacy_cell_task_view",
        lambda value: any(value is item for item in canonical),
    )
    monkeypatch.setattr(
        assembly_module,
        "is_four_cell_adequacy_holm_view",
        lambda value: value is canonical_holm,
    )

    sources = {view.task_id: view for view in canonical} | {
        "four_cell_adequacy_holm": canonical_holm
    }
    for task_id, source in sources.items():
        decision = derive_g3_gate_decision(task_id, source)
        assert decision.passed
        assert decision.evidence["run_id"] == source.run_id
        assert decision.evidence["binding_fingerprint"] == (source.binding_fingerprint)
    assert task_view_cross_binding_error(sources) is None

    with pytest.raises(ModelError, match="sealed adequacy cell task view"):
        derive_g3_gate_decision(
            "design_latent_correctness",
            copy.copy(canonical[0]),
        )
    with pytest.raises(ModelError, match="sealed adequacy cell task view"):
        derive_g3_gate_decision(
            "design_latent_correctness",
            canonical[1],
        )

    wrong_holm = replace(
        canonical_holm,
        source_views=(canonical[1], canonical[0], canonical[2], canonical[3]),
    )
    assert "exact prerequisite" in (
        task_view_cross_binding_error(
            {**sources, "four_cell_adequacy_holm": wrong_holm}
        )
        or ""
    )


def test_four_cell_gate_rejects_copied_unregistered_execution_evidence() -> None:
    latent = _latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8)
    refit = _refit("design", 0.8)

    with pytest.raises(ModelError, match="not registered"):
        assemble_four_cell_adequacy_gate(
            design_latent=replace(latent),
            production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.8),
            design_refitted=refit,
            production_refitted=_refit("production", 0.8),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="not registered"):
        assemble_four_cell_adequacy_gate(
            design_latent=latent,
            production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.8),
            design_refitted=replace(refit),
            production_refitted=_refit("production", 0.8),
            strict_canonical=False,
        )


def test_four_cell_gate_preserves_order_and_passes_when_holm_rejects_none() -> None:
    design_latent = _latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8)
    production_latent = _latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.7)
    result = assemble_four_cell_adequacy_gate(
        design_latent=design_latent,
        production_latent=production_latent,
        design_refitted=_refit("design", 0.6),
        production_refitted=_refit("production", 0.5),
        strict_canonical=False,
    )

    assert result.ready
    assert result.passed
    assert result.holm is not None
    assert result.cell_order == (
        "design_latent",
        "production_latent",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )
    np.testing.assert_allclose(
        result.holm.raw_p_values,
        [
            design_latent.bootstrap.p_value,
            production_latent.bootstrap.p_value,
            0.6,
            0.5,
        ],
    )


def test_low_cell_pvalue_is_holm_rejected_and_fails_gate() -> None:
    result = assemble_four_cell_adequacy_gate(
        design_latent=_latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8),
        production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.7),
        design_refitted=_refit("design", 0.001),
        production_refitted=_refit("production", 0.5),
        strict_canonical=False,
    )
    assert result.ready
    assert not result.passed
    assert result.failure_reasons == ("design_refitted_adequacy:holm_rejected",)


def test_incomplete_refit_withholds_holm_evaluation() -> None:
    refit = replace(
        _refit("design", 0.6),
        ready_for_holm=False,
        max_statistic_cell=None,
    )
    object.__setattr__(
        refit,
        "_execution_seal",
        refit_execution._REGISTERED_REFIT_CAPABILITY,
    )
    with pytest.raises(ModelError, match="not registered execution evidence"):
        assemble_four_cell_adequacy_gate(
            design_latent=_latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8),
            production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.7),
            design_refitted=refit,
            production_refitted=_refit("production", 0.5),
            strict_canonical=False,
        )


def test_strict_mode_rejects_reduced_fixture_counts_without_raising() -> None:
    result = assemble_four_cell_adequacy_gate(
        design_latent=_latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8),
        production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.7),
        design_refitted=_refit("design", 0.6),
        production_refitted=_refit("production", 0.5),
    )
    assert not result.ready
    assert len(result.failure_reasons) == 4


def test_independent_cell_views_expose_only_one_exact_role_and_binding() -> None:
    views = _cell_views()

    assert tuple(view.task_id for view in views) == (
        "design_latent_correctness",
        "production_latent_correctness",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )
    assert tuple(view.role for view in views) == (
        "design",
        "production",
        "design",
        "production",
    )
    assert tuple(view.cell.cell_id for view in views) == (
        "design_latent",
        "production_latent",
        "design_refitted_adequacy",
        "production_refitted_adequacy",
    )
    assert all(is_adequacy_cell_task_view(view) for view in views)
    assert all(view.ready and view.passed for view in views)
    assert all(not view.canonical_count_ready for view in views)
    assert all(len(view.content_fingerprint) == 64 for view in views)
    assert all(len(view.source_evidence_fingerprint) == 64 for view in views)
    assert all(not hasattr(view, "evidence") for view in views)
    assert len({view.binding_fingerprint for view in views}) == 4
    assert len({view.parent_model_fingerprint for view in views}) == 2
    assert len({view.parent_view_fingerprint for view in views}) == 2
    assert len({view.rng_execution_fingerprint for view in views}) == 4
    for view in views[:2]:
        assert len(view.simulation_rng_execution_fingerprint or "") == 64
        assert len(view.bootstrap_rng_execution_fingerprint or "") == 64
        assert (
            view.simulation_rng_execution_fingerprint
            != view.bootstrap_rng_execution_fingerprint
        )
    for view in views[2:]:
        assert view.simulation_rng_execution_fingerprint is None
        assert view.bootstrap_rng_execution_fingerprint is None


def test_design_views_reject_future_production_evidence() -> None:
    run_id = _fp("run")
    design_parent, production_parent = _parents(run_id)
    with pytest.raises(ModelError, match="parent|cross-role"):
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=_latent(
                CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
                0.5,
                parent=production_parent,
            ),
            parent=design_parent,
            run_id=run_id,
            binding_fingerprint=_fp("binding"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="parent|cross-role"):
        build_refitted_adequacy_task_view(
            task_id="design_refitted_adequacy",
            evidence=_refit("production", 0.5, parent=production_parent),
            parent=design_parent,
            run_id=run_id,
            binding_fingerprint=_fp("binding"),
            strict_canonical=False,
        )


def test_cell_views_reject_same_run_equivalent_parent_substitution() -> None:
    run_id = _fp("same-run-parent-substitution")
    original_design, original_production = _make_parents(run_id, "original")
    alternate_design, alternate_production = _make_parents(run_id, "alternate")

    design_latent = _latent(
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0.8,
        parent=original_design,
    )
    with pytest.raises(ModelError, match="exact parent|descend"):
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=design_latent,
            parent=alternate_design,
            run_id=run_id,
            binding_fingerprint=_fp("substituted-design-latent"),
            strict_canonical=False,
        )

    production_refit = _refit(
        "production",
        0.5,
        parent=original_production,
    )
    with pytest.raises(ModelError, match="exact parent"):
        build_refitted_adequacy_task_view(
            task_id="production_refitted_adequacy",
            evidence=production_refit,
            parent=alternate_production,
            run_id=run_id,
            binding_fingerprint=_fp("substituted-production-refit"),
            strict_canonical=False,
        )


def test_holm_view_rejects_cross_run_and_cross_role_mixing() -> None:
    run_a = _cell_views(run_id=_fp("run-a"))
    run_b = _cell_views(run_id=_fp("run-b"))

    with pytest.raises(ModelError, match="different runs"):
        assemble_four_cell_adequacy_holm_view(
            design_latent=run_a[0],
            production_latent=run_b[1],
            design_refitted=run_a[2],
            production_refitted=run_a[3],
            binding_fingerprint=_fp("binding-holm"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="roles/order"):
        assemble_four_cell_adequacy_holm_view(
            design_latent=run_a[1],
            production_latent=run_a[0],
            design_refitted=run_a[2],
            production_refitted=run_a[3],
            binding_fingerprint=_fp("binding-holm"),
            strict_canonical=False,
        )


def test_holm_view_rejects_duplicate_or_reused_task_binding() -> None:
    views = _cell_views()
    production_parent = _parents(views[0].run_id)[1]
    duplicate = build_latent_adequacy_task_view(
        task_id="production_latent_correctness",
        evidence=_latent(
            CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
            0.7,
            parent=production_parent,
        ),
        parent=production_parent,
        run_id=views[0].run_id,
        binding_fingerprint=views[0].binding_fingerprint,
        strict_canonical=False,
    )
    with pytest.raises(ModelError, match="distinct task bindings"):
        assemble_four_cell_adequacy_holm_view(
            design_latent=views[0],
            production_latent=duplicate,
            design_refitted=views[2],
            production_refitted=views[3],
            binding_fingerprint=_fp("binding-holm"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="distinct from source"):
        assemble_four_cell_adequacy_holm_view(
            design_latent=views[0],
            production_latent=views[1],
            design_refitted=views[2],
            production_refitted=views[3],
            binding_fingerprint=views[0].binding_fingerprint,
            strict_canonical=False,
        )


def test_process_authority_rejects_copy_replace_and_stolen_seal_forgery() -> None:
    source = _cell_views()[0]
    copied = copy.copy(source)
    replaced = replace(source)
    forged = replace(source)
    object.__setattr__(
        forged,
        "_execution_seal",
        gate_module._ADEQUACY_TASK_VIEW_CAPABILITY,
    )

    assert copied is not source
    assert not is_adequacy_cell_task_view(copied)
    assert not is_adequacy_cell_task_view(replaced)
    assert not is_adequacy_cell_task_view(forged)

    holm = _holm_view(_cell_views())
    assert not is_four_cell_adequacy_holm_view(copy.copy(holm))
    assert not is_four_cell_adequacy_holm_view(replace(holm))

    compatibility = assemble_four_cell_adequacy_gate(
        design_latent=_latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8),
        production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.7),
        design_refitted=_refit("design", 0.6),
        production_refitted=_refit("production", 0.5),
        strict_canonical=False,
    )
    assert is_four_cell_adequacy_gate(compatibility)
    assert not is_four_cell_adequacy_gate(copy.copy(compatibility))
    assert not is_four_cell_adequacy_gate(replace(compatibility))


def test_content_validation_detects_in_place_view_and_holm_mutation() -> None:
    source = _cell_views()[0]
    object.__setattr__(source, "passed", False)
    assert not is_adequacy_cell_task_view(source)

    holm = _holm_view(_cell_views())
    assert holm.holm is not None
    object.__setattr__(
        holm.holm,
        "raw_p_values",
        np.asarray([0.9, 0.9, 0.9, 0.9]),
    )
    assert not is_four_cell_adequacy_holm_view(holm)


def test_holm_is_recomputed_from_pvalues_and_never_trusts_cell_pass_flags() -> None:
    views = _cell_views(p_values=(0.8, 0.7, 0.001, 0.5))
    assert all(view.passed for view in views)

    result = _holm_view(views)

    assert is_four_cell_adequacy_holm_view(result)
    assert result.ready
    assert not result.passed
    assert result.holm is not None
    np.testing.assert_allclose(
        result.holm.raw_p_values,
        [views[0].cell.p_value, views[1].cell.p_value, 0.001, 0.5],
    )
    assert result.failure_reasons == ("design_refitted_adequacy:holm_rejected",)


def test_task_view_content_fingerprints_are_deterministic_and_input_sensitive() -> None:
    first = _cell_views()
    second = _cell_views()
    changed = _cell_views(p_values=(0.81, 0.7, 0.6, 0.5))

    assert tuple(view.content_fingerprint for view in first) == tuple(
        view.content_fingerprint for view in second
    )
    assert tuple(view.source_evidence_fingerprint for view in first) == tuple(
        view.source_evidence_fingerprint for view in second
    )
    assert (
        first[0].source_evidence_fingerprint != changed[0].source_evidence_fingerprint
    )
    assert first[0].content_fingerprint != changed[0].content_fingerprint


def test_noncanonical_mode_is_explicit_and_strict_mode_fails_closed() -> None:
    run_id = _fp("run")
    parent = _parents(run_id)[0]
    raw = _latent(
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0.8,
        parent=parent,
    )
    test_view = build_latent_adequacy_task_view(
        task_id="design_latent_correctness",
        evidence=raw,
        parent=parent,
        run_id=run_id,
        binding_fingerprint=_fp("binding"),
        strict_canonical=False,
    )
    strict_view = build_latent_adequacy_task_view(
        task_id="design_latent_correctness",
        evidence=raw,
        parent=parent,
        run_id=run_id,
        binding_fingerprint=_fp("binding"),
        strict_canonical=True,
    )

    assert not test_view.canonical_count_ready
    assert test_view.ready and test_view.passed
    assert not strict_view.canonical_count_ready
    assert not strict_view.ready and not strict_view.passed
    assert (
        strict_view.cell.failure_reason == "latent_canonical_geometry_or_count_mismatch"
    )


def test_incomplete_forged_task_source_cannot_cross_the_view_guard() -> None:
    raw = replace(
        _refit("design", 0.6),
        ready_for_holm=False,
        max_statistic_cell=None,
    )
    object.__setattr__(
        raw,
        "_execution_seal",
        refit_execution._REGISTERED_REFIT_CAPABILITY,
    )
    views = _cell_views()
    with pytest.raises(ModelError, match="not registered execution evidence"):
        build_refitted_adequacy_task_view(
            task_id="design_refitted_adequacy",
            evidence=raw,
            parent=_parents(views[0].run_id)[0],
            run_id=views[0].run_id,
            binding_fingerprint=views[2].binding_fingerprint,
            strict_canonical=False,
        )


@pytest.mark.parametrize("field", ["run_id", "binding_fingerprint"])
def test_task_view_rejects_malformed_external_binding(field: str) -> None:
    run_id = "not-a-sha256" if field == "run_id" else _fp("run")
    binding = "not-a-sha256" if field == "binding_fingerprint" else _fp("binding")
    parent = _parent_for_role("design")
    with pytest.raises(ModelError, match="must be SHA-256"):
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=_latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8),
            parent=parent,
            run_id=run_id,
            binding_fingerprint=binding,
            strict_canonical=False,
        )


def test_private_boundary_validators_reject_every_structural_mismatch() -> None:
    views = _cell_views()
    design_parent, _production_parent = _parents(views[0].run_id)

    with pytest.raises(ModelError, match="parent role is invalid"):
        gate_module._parent_sources("invalid", design_parent)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="production adequacy requires"):
        gate_module._parent_sources("production", design_parent)
    with pytest.raises(ModelError, match="cross parent run"):
        gate_module._validate_parent_task_binding(
            design_parent,
            run_id=_fp("different-run"),
            child_binding_fingerprint=_fp("child"),
        )
    with pytest.raises(ModelError, match="must differ"):
        gate_module._validate_parent_task_binding(
            design_parent,
            run_id=design_parent.run_id,
            child_binding_fingerprint=design_parent.binding_fingerprint,
        )
    with pytest.raises(ModelError, match="task ID is not registered"):
        gate_module._validate_view_inputs(
            "not-a-task",
            _fp("run"),
            _fp("binding"),
            False,
        )
    with pytest.raises(ModelError, match="strict_canonical must be boolean"):
        gate_module._validate_view_inputs(
            "design_latent_correctness",
            _fp("run"),
            _fp("binding"),
            1,  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="non-latent task ID"):
        build_latent_adequacy_task_view(
            task_id="design_refitted_adequacy",  # type: ignore[arg-type]
            evidence=object(),  # type: ignore[arg-type]
            parent=design_parent,
            run_id=design_parent.run_id,
            binding_fingerprint=_fp("child-latent"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="not registered execution evidence"):
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=object(),  # type: ignore[arg-type]
            parent=design_parent,
            run_id=design_parent.run_id,
            binding_fingerprint=_fp("child-latent"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="non-refitted task ID"):
        build_refitted_adequacy_task_view(
            task_id="design_latent_correctness",  # type: ignore[arg-type]
            evidence=object(),  # type: ignore[arg-type]
            parent=design_parent,
            run_id=design_parent.run_id,
            binding_fingerprint=_fp("child-refit"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="strict_canonical must be boolean"):
        assemble_four_cell_adequacy_holm_view(
            design_latent=views[0],
            production_latent=views[1],
            design_refitted=views[2],
            production_refitted=views[3],
            binding_fingerprint=_fp("holm"),
            strict_canonical=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="strict_canonical must be boolean"):
        assemble_four_cell_adequacy_gate(
            design_latent=object(),  # type: ignore[arg-type]
            production_latent=object(),  # type: ignore[arg-type]
            design_refitted=object(),  # type: ignore[arg-type]
            production_refitted=object(),  # type: ignore[arg-type]
            strict_canonical=1,  # type: ignore[arg-type]
        )
    assert not is_four_cell_adequacy_gate(object())
    with pytest.raises(ModelError, match="wrong type"):
        gate_module._cell_payload(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="not canonical JSON"):
        gate_module._canonical_json_bytes({object()})


def test_cell_record_validation_covers_role_artifact_and_consistency_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate_module, "is_registered_latent_cell", lambda _value: True)
    latent = SimpleNamespace(
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        chain_streams=3,
        bootstrap_streams=2,
        state_uniforms_per_chain=3,
        resampled_ids_per_bootstrap=2,
        simulation=SimpleNamespace(
            chain_count=2,
            measured_months=12,
            extension_months=24,
            uniforms_per_chain=3,
        ),
        bootstrap=SimpleNamespace(
            chain_count=2,
            bootstrap_replicates=2,
            resample_size=2,
            measured_months=12,
            extension_months=24,
            survival_months=3,
            p_value=0.5,
        ),
    )
    with pytest.raises(ModelError, match="wrong registered artifact"):
        gate_module._latent_record(
            "design_latent",
            latent,
            expected_artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
            strict_canonical=False,
        )
    record = gate_module._latent_record(
        "design_latent",
        latent,
        expected_artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        strict_canonical=False,
    )
    assert not record.ready
    assert record.failure_reason == "latent_evidence_inconsistent"

    monkeypatch.setattr(
        gate_module,
        "is_registered_refitted_adequacy_batch",
        lambda _value: True,
    )
    refit = SimpleNamespace(
        role="production",
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        successful_attempts=1,
        failed_attempts=0,
        attempts=2,
        attempt_audits=(object(), object()),
        successful_attempt_indices=(0,),
        successful_endpoint_matrix=np.zeros((1, 1)),
        minimum_successes=1,
        success_requirement_passed=True,
        ready_for_holm=True,
        max_statistic_cell=SimpleNamespace(successful_replicates=1, p_value=0.5),
        cell_failure=None,
        strict_canonical=False,
    )
    with pytest.raises(ModelError, match="wrong artifact role"):
        gate_module._refit_record(
            "design_refitted_adequacy",
            refit,
            expected_role="design",
            strict_canonical=False,
        )
    refit.role = "design"
    with pytest.raises(ModelError, match="wrong registered artifact"):
        gate_module._refit_record(
            "design_refitted_adequacy",
            refit,
            expected_role="design",
            strict_canonical=False,
        )
    refit.artifact = CalibrationArtifact.DESIGN_OR_CONTROL
    record = gate_module._refit_record(
        "design_refitted_adequacy",
        refit,
        expected_role="design",
        strict_canonical=False,
    )
    assert not record.ready
    assert record.failure_reason == "refitted_adequacy_not_ready"


def test_view_builders_reject_cross_artifact_and_feature_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = _fp("builder-source-guards")
    design_parent, production_parent = _parents(run_id)
    design_evidence = _latent(
        CalibrationArtifact.DESIGN_OR_CONTROL,
        0.5,
        parent=design_parent,
    )
    design_fit = selected_design_fit_from_view(design_parent)
    design_identity = gate_module.registered_latent_cell_identity(design_evidence)
    cross_artifact = SimpleNamespace(
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    monkeypatch.setattr(gate_module, "is_registered_latent_cell", lambda _value: True)
    monkeypatch.setattr(
        gate_module,
        "registered_latent_cell_identity",
        lambda _value: design_identity,
    )
    monkeypatch.setattr(
        gate_module,
        "registered_latent_cell_source_fit",
        lambda _value: design_fit,
    )
    with pytest.raises(ModelError, match="cross-role latent evidence"):
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=cross_artifact,  # type: ignore[arg-type]
            parent=design_parent,
            run_id=run_id,
            binding_fingerprint=_fp("cross-artifact"),
            strict_canonical=False,
        )

    evidence = _refit("production", 0.5, parent=production_parent)
    parent_fit, parent_features = production_fit_sources_from_view(production_parent)
    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_sources",
        lambda _evidence: (parent_fit, copy.copy(parent_features)),
    )
    with pytest.raises(ModelError, match="exact parent features"):
        build_refitted_adequacy_task_view(
            task_id="production_refitted_adequacy",
            evidence=evidence,
            parent=production_parent,
            run_id=run_id,
            binding_fingerprint=_fp("feature-substitution"),
            strict_canonical=False,
        )


def test_cell_view_minter_requires_task_specific_split_rng_contract() -> None:
    record = gate_module.AdequacyCellRecord(
        "design_latent",
        True,
        0.5,
        10,
        10,
        None,
    )
    common = {
        "run_id": _fp("run"),
        "binding_fingerprint": _fp("binding"),
        "parent_view_fingerprint": _fp("parent-view"),
        "parent_model_fingerprint": _fp("parent-model"),
        "scientific_input_fingerprint": _fp("input"),
        "rng_execution_fingerprint": _fp("rng"),
        "source_evidence_fingerprint": _fp("source"),
    }
    with pytest.raises(ModelError, match="simulation RNG fingerprint is missing"):
        gate_module._mint_cell_task_view(
            task_id="design_latent_correctness",
            role="design",
            simulation_rng_execution_fingerprint=None,
            bootstrap_rng_execution_fingerprint=_fp("bootstrap"),
            cell=record,
            canonical_count_ready=True,
            strict_canonical=False,
            source_evidence=object(),  # type: ignore[arg-type]
            parent_view=object(),  # type: ignore[arg-type]
            **common,
        )
    with pytest.raises(ModelError, match="cannot expose latent RNG"):
        gate_module._mint_cell_task_view(
            task_id="design_refitted_adequacy",
            role="design",
            simulation_rng_execution_fingerprint=_fp("simulation"),
            bootstrap_rng_execution_fingerprint=None,
            cell=replace(record, cell_id="design_refitted_adequacy"),
            canonical_count_ready=True,
            strict_canonical=False,
            source_evidence=object(),  # type: ignore[arg-type]
            parent_view=object(),  # type: ignore[arg-type]
            **common,
        )


def test_authoritative_holm_rejects_order_alpha_and_missing_pvalue() -> None:
    cells = tuple(view.cell for view in _cell_views())
    with pytest.raises(ModelError, match="preregistered order"):
        gate_module._authoritative_holm(
            (cells[1], cells[0], cells[2], cells[3]),
            source_passed=(True, True, True, True),
            alpha=0.05,
        )
    with pytest.raises(ModelError, match="alpha must be finite"):
        gate_module._authoritative_holm(
            cells,
            source_passed=(True, True, True, True),
            alpha=float("nan"),
        )
    with pytest.raises(ModelError, match="missing its p-value"):
        gate_module._authoritative_holm(
            (replace(cells[0], p_value=None), cells[1], cells[2], cells[3]),
            source_passed=(True, True, True, True),
            alpha=0.05,
        )


def test_minted_view_verifiers_reject_each_late_boundary_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _cell_views()
    cell = views[0]

    def rejected(value: object, field: str, changed: object, verifier: object) -> None:
        original = getattr(value, field)
        object.__setattr__(value, field, changed)
        try:
            with pytest.raises(ModelError):
                verifier(value)  # type: ignore[operator]
        finally:
            object.__setattr__(value, field, original)

    rejected(cell, "contract_version", "wrong", gate_module._verify_cell_task_view)
    rejected(
        cell,
        "cell",
        replace(cell.cell, cell_id="production_latent"),
        gate_module._verify_cell_task_view,
    )
    rejected(
        cell,
        "parent_view_fingerprint",
        _fp("changed-parent"),
        gate_module._verify_cell_task_view,
    )
    rejected(
        cell,
        "content_fingerprint",
        _fp("changed-content"),
        gate_module._verify_cell_task_view,
    )

    holm = _holm_view(views)
    rejected(holm, "contract_version", "wrong", gate_module._verify_holm_view)
    rejected(holm, "strict_canonical", 1, gate_module._verify_holm_view)
    rejected(holm, "source_views", holm.source_views[:3], gate_module._verify_holm_view)
    rejected(
        holm,
        "source_views",
        (holm.source_views[1], holm.source_views[0], *holm.source_views[2:]),
        gate_module._verify_holm_view,
    )
    rejected(
        holm,
        "source_content_fingerprints",
        (_fp("changed-source"), *holm.source_content_fingerprints[1:]),
        gate_module._verify_holm_view,
    )
    rejected(holm, "passed", not holm.passed, gate_module._verify_holm_view)
    rejected(
        holm,
        "content_fingerprint",
        _fp("changed-holm"),
        gate_module._verify_holm_view,
    )

    compatibility = assemble_four_cell_adequacy_gate(
        design_latent=_latent(CalibrationArtifact.DESIGN_OR_CONTROL, 0.8),
        production_latent=_latent(CalibrationArtifact.PRODUCTION_OR_CANDIDATE, 0.7),
        design_refitted=_refit("design", 0.6),
        production_refitted=_refit("production", 0.5),
        strict_canonical=False,
    )
    rejected(
        compatibility,
        "cell_order",
        tuple(reversed(compatibility.cell_order)),
        gate_module._verify_compatibility_gate,
    )
    rejected(
        compatibility,
        "cells",
        (
            replace(compatibility.cells[0], cell_id="production_latent"),
            *compatibility.cells[1:],
        ),
        gate_module._verify_compatibility_gate,
    )
    rejected(
        compatibility,
        "strict_canonical",
        1,
        gate_module._verify_compatibility_gate,
    )
    rejected(
        compatibility,
        "passed",
        not compatibility.passed,
        gate_module._verify_compatibility_gate,
    )
    rejected(
        compatibility,
        "content_fingerprint",
        _fp("changed-gate"),
        gate_module._verify_compatibility_gate,
    )

    source_identity = gate_module._reverify_cell_sources(cell)
    monkeypatch.setattr(
        gate_module,
        "_reverify_cell_sources",
        lambda _value: source_identity,
    )
    rejected(
        cell,
        "simulation_rng_execution_fingerprint",
        None,
        gate_module._verify_cell_task_view,
    )
    rejected(cell, "canonical_count_ready", 1, gate_module._verify_cell_task_view)
    rejected(cell, "ready", not cell.ready, gate_module._verify_cell_task_view)


def test_registry_and_registered_source_reverification_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _cell_views()
    original_registry = gate_module._TASK_CELL_ROLE

    latent_registry = dict(original_registry)
    latent_registry["design_latent_correctness"] = ("wrong_cell", "design")
    monkeypatch.setattr(gate_module, "_TASK_CELL_ROLE", latent_registry)
    with pytest.raises(AssertionError, match="latent task registry"):
        build_latent_adequacy_task_view(
            task_id="design_latent_correctness",
            evidence=views[0]._source_evidence,
            parent=views[0]._parent_view,
            run_id=views[0].run_id,
            binding_fingerprint=_fp("registry-latent"),
            strict_canonical=False,
        )

    refit_registry = dict(original_registry)
    refit_registry["design_refitted_adequacy"] = ("wrong_cell", "design")
    monkeypatch.setattr(gate_module, "_TASK_CELL_ROLE", refit_registry)
    with pytest.raises(AssertionError, match="refitted task registry"):
        build_refitted_adequacy_task_view(
            task_id="design_refitted_adequacy",
            evidence=views[2]._source_evidence,
            parent=views[2]._parent_view,
            run_id=views[2].run_id,
            binding_fingerprint=_fp("registry-refit"),
            strict_canonical=False,
        )
    monkeypatch.setattr(gate_module, "_TASK_CELL_ROLE", original_registry)

    latent = views[0]
    original_latent_source = latent._source_evidence
    object.__setattr__(latent, "_source_evidence", object())
    try:
        with pytest.raises(ModelError, match="lost its registered source"):
            gate_module._reverify_cell_sources(latent)
    finally:
        object.__setattr__(latent, "_source_evidence", original_latent_source)

    original_latent_source_fit = gate_module.registered_latent_cell_source_fit
    monkeypatch.setattr(
        gate_module,
        "registered_latent_cell_source_fit",
        lambda _evidence: object(),
    )
    with pytest.raises(ModelError, match="no longer matches"):
        gate_module._reverify_cell_sources(latent)
    monkeypatch.setattr(
        gate_module,
        "registered_latent_cell_source_fit",
        original_latent_source_fit,
    )

    refit = views[2]
    original_refit_source = refit._source_evidence
    object.__setattr__(refit, "_source_evidence", object())
    try:
        with pytest.raises(ModelError, match="lost its registered source"):
            gate_module._reverify_cell_sources(refit)
    finally:
        object.__setattr__(refit, "_source_evidence", original_refit_source)


def test_refit_source_identity_and_holm_relationship_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _cell_views()
    design_refit = views[2]
    production_refit = views[3]

    original_sources = gate_module.registered_refitted_adequacy_sources
    source_fit, source_features = original_sources(design_refit._source_evidence)
    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_sources",
        lambda _evidence: (object(), source_features),
    )
    with pytest.raises(ModelError, match="retains its parent fit"):
        gate_module._reverify_cell_sources(design_refit)

    production_fit, production_features = original_sources(
        production_refit._source_evidence
    )
    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_sources",
        lambda _evidence: (production_fit, copy.copy(production_features)),
    )
    with pytest.raises(ModelError, match="retains its parent features"):
        gate_module._reverify_cell_sources(production_refit)

    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_sources",
        original_sources,
    )
    original_identity = gate_module.registered_refitted_adequacy_batch_identity
    identity = original_identity(design_refit._source_evidence)
    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_batch_identity",
        lambda _evidence: replace(
            identity,
            parent_model_fingerprint=_fp("changed-parent-model"),
        ),
    )
    with pytest.raises(ModelError, match="no longer matches"):
        gate_module._reverify_cell_sources(design_refit)

    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_batch_identity",
        original_identity,
    )
    holm = _holm_view(views)

    def rejected(field: str, changed: object) -> None:
        original = getattr(holm, field)
        object.__setattr__(holm, field, changed)
        try:
            with pytest.raises(ModelError):
                gate_module._verify_holm_view(holm)
        finally:
            object.__setattr__(holm, field, original)

    rejected("run_id", _fp("different-holm-run"))
    rejected("strict_canonical", True)
    rejected("binding_fingerprint", views[0].binding_fingerprint)


def test_late_cell_view_shape_guards_use_reverified_dynamic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _cell_views()
    latent = views[0]
    refit = views[2]
    original_reverify = gate_module._reverify_cell_sources
    latent_identity = original_reverify(latent)
    refit_identity = original_reverify(refit)
    monkeypatch.setattr(
        gate_module,
        "_reverify_cell_sources",
        lambda value: (
            *latent_identity[:4],
            value.simulation_rng_execution_fingerprint,
            value.bootstrap_rng_execution_fingerprint,
            latent_identity[6],
            latent_identity[7],
            value.canonical_count_ready,
        ),
    )

    original_simulation = latent.simulation_rng_execution_fingerprint
    object.__setattr__(latent, "simulation_rng_execution_fingerprint", None)
    try:
        with pytest.raises(ModelError, match="lacks split RNG"):
            gate_module._verify_cell_task_view(latent)
    finally:
        object.__setattr__(
            latent,
            "simulation_rng_execution_fingerprint",
            original_simulation,
        )

    original_ready = latent.canonical_count_ready
    object.__setattr__(latent, "canonical_count_ready", 1)
    try:
        with pytest.raises(ModelError, match="readiness must be boolean"):
            gate_module._verify_cell_task_view(latent)
    finally:
        object.__setattr__(latent, "canonical_count_ready", original_ready)

    monkeypatch.setattr(
        gate_module,
        "_reverify_cell_sources",
        lambda value: (
            *refit_identity[:4],
            value.simulation_rng_execution_fingerprint,
            value.bootstrap_rng_execution_fingerprint,
            refit_identity[6],
            refit_identity[7],
            value.canonical_count_ready,
        ),
    )
    object.__setattr__(refit, "simulation_rng_execution_fingerprint", _fp("extra-rng"))
    try:
        with pytest.raises(ModelError, match="contains latent RNG"):
            gate_module._verify_cell_task_view(refit)
    finally:
        object.__setattr__(refit, "simulation_rng_execution_fingerprint", None)


def test_refit_builder_identity_and_holm_strict_mode_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _cell_views()
    design_refit = views[2]
    original_identity = gate_module.registered_refitted_adequacy_batch_identity
    identity = original_identity(design_refit._source_evidence)
    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_batch_identity",
        lambda _evidence: replace(
            identity,
            artifact_feature_fingerprint=_fp("changed-feature-identity"),
        ),
    )
    with pytest.raises(ModelError, match="source identities differ"):
        build_refitted_adequacy_task_view(
            task_id="design_refitted_adequacy",
            evidence=design_refit._source_evidence,
            parent=design_refit._parent_view,
            run_id=design_refit.run_id,
            binding_fingerprint=_fp("identity-mismatch"),
            strict_canonical=False,
        )
    monkeypatch.setattr(
        gate_module,
        "registered_refitted_adequacy_batch_identity",
        original_identity,
    )
    with pytest.raises(ModelError, match="strict mode differs"):
        assemble_four_cell_adequacy_holm_view(
            design_latent=views[0],
            production_latent=views[1],
            design_refitted=views[2],
            production_refitted=views[3],
            binding_fingerprint=_fp("strict-mismatch"),
            strict_canonical=True,
        )
