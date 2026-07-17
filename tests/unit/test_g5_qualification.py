from __future__ import annotations

import hashlib

import numpy as np
import pytest

from prpg.errors import ModelError, ValidationError
from prpg.validation.adversaries import (
    fit_g5_adversary_model,
    score_g5_adversary_path,
)
from prpg.validation.meta_validation import (
    G5_DEFECT_REGISTRY,
    apply_registered_return_defect,
    fixed_hmm_meta_means,
    perturb_transition_matrix,
    project_g5_meta_resources,
    run_g5_meta_validation,
)
from prpg.validation.qualification import (
    G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    G5_PATH_OBSERVABLE_INDEX,
    G5CandidateBundle,
    G5CandidatePathView,
    activate_g5_candidate_generation,
    aggregate_g5_cluster_matrices,
    aggregate_g5_cluster_observables,
    build_g5_family_null_calibration,
    build_g5_latent_chain_reference,
    build_g5_reference_data,
    compute_g5_latent_chain_cap_ratio,
    compute_g5_path_observables,
    compute_g5_reference_observables,
    evaluate_g5_candidate,
    pack_g5_path_observables,
    project_g5_candidate_resources,
    project_g5_null_resources,
    run_g5_family_null_calibration,
)
from prpg.validation.sobol import registered_sobol_directions
from prpg.validation.statistics import POWER_CLAIM_NAMES


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _path(
    *,
    stage: str,
    family: str,
    entity: int,
    observations: int,
    seed: int,
) -> G5CandidatePathView:
    rng = np.random.default_rng(seed)
    covariance = np.asarray(((1.0, 0.2, 0.1), (0.2, 0.7, 0.25), (0.1, 0.25, 0.8)))
    values = rng.multivariate_normal(np.zeros(3), covariance, observations) / 100.0
    values[0, 0] += entity * 1e-8
    prefix = "D" if stage == "development" else "Q"
    path_id = f"{prefix}{family}{entity:06d}"
    source = _fp(f"source-{stage}-{family}-{entity}-{seed}")
    return G5CandidatePathView(
        path_id=path_id,
        family=family,  # type: ignore[arg-type]
        log_returns=values,
        target_states=np.arange(observations, dtype=np.int64) % 4,
        source_fingerprint=source,
        fingerprint=_fp(f"view-{source}"),
    )


def _bundle(
    stage: str,
    *,
    lf_count: int = 4,
    hf_count: int = 4,
    elapsed: float = 1.0,
) -> G5CandidateBundle:
    lf = tuple(
        _path(
            stage=stage,
            family="LF",
            entity=index + 1,
            observations=96,
            seed=1000 + index + (0 if stage == "development" else 100),
        )
        for index in range(lf_count)
    )
    hf = tuple(
        _path(
            stage=stage,
            family="HF",
            entity=index + 1,
            observations=128,
            seed=2000 + index + (0 if stage == "development" else 100),
        )
        for index in range(hf_count)
    )
    return G5CandidateBundle(
        schema_id="prpg-g5-candidate-qualification-v1",
        candidate_stage=stage,  # type: ignore[arg-type]
        generation_policy_fingerprint=_fp("policy"),
        lf_paths=lf,
        hf_paths=hf,
        elapsed_seconds=elapsed,
        exact_canonical_counts=False,
        generation_authorized=False,
        fingerprint=_fp(f"bundle-{stage}-{lf_count}-{hf_count}"),
    )


def test_family_null_canonical_record_uses_exact_50000_higher_ranks() -> None:
    grid = np.linspace(0.0, 2.0, 50_000)
    calibration = build_g5_family_null_calibration(
        {
            "g5_a": grid,
            "g5_b": grid + 0.01,
            "g5_c": grid + 0.02,
            "g5_d": grid + 0.03,
        },
        threshold_set_id="g5.fixed_policy.v1",
        binding_fingerprints={"qualification_design": _fp("design")},
        sobol_direction_fingerprint=_fp("sobol"),
    )

    assert calibration.fixed_sample_canonical
    assert calibration.canonical_thresholds is not None
    assert calibration.canonical_thresholds.precision_passed
    assert {
        item.sample_count for item in calibration.canonical_thresholds.thresholds
    } == {50_000}
    assert {
        item.one_based_rank for item in calibration.canonical_thresholds.thresholds
    } == {47_501}


def test_callback_null_calibrator_is_reduced_until_exactly_50000() -> None:
    run = run_g5_family_null_calibration(
        lambda replicate: {
            "g5_a": replicate / 10.0,
            "g5_b": replicate / 9.0,
            "g5_c": replicate / 8.0,
            "g5_d": replicate / 7.0,
        },
        threshold_set_id="g5.callback.pilot",
        binding_fingerprints={"qualification_design": _fp("design")},
        sobol_direction_fingerprint=_fp("sobol"),
        replicate_count=10,
    )

    assert run.calibration.replicate_count == 10
    assert not run.calibration.fixed_sample_canonical
    assert run.calibration.canonical_thresholds is None
    projection = project_g5_null_resources(run)
    assert projection.target_replicates == 50_000
    assert projection.projected_seconds == pytest.approx(run.elapsed_seconds * 5_000)


def test_reduced_candidate_assessment_is_deterministic_but_nonauthorizing() -> None:
    monthly_values = _path(
        stage="reference", family="LF", entity=1, observations=72, seed=77
    ).log_returns
    daily_values = _path(
        stage="reference", family="HF", entity=1, observations=64, seed=88
    ).log_returns
    monthly = build_g5_reference_data(
        monthly_values, np.arange(72) % 4, frequency="monthly"
    )
    daily = build_g5_reference_data(daily_values, np.arange(64) % 4, frequency="daily")
    directions = registered_sobol_directions(master_seed=123, scientific_version=11)
    null = build_g5_family_null_calibration(
        {
            name: np.linspace(0.0, 100.0, 20)
            for name in ("g5_a", "g5_b", "g5_c", "g5_d")
        },
        threshold_set_id="g5.reduced",
        binding_fingerprints={"qualification_design": _fp("design")},
        sobol_direction_fingerprint=directions.fingerprint,
    )
    development = _bundle("development")
    qualification = _bundle("qualification")

    first = evaluate_g5_candidate(
        development,
        qualification,
        monthly_reference=monthly,
        daily_reference=daily,
        sobol_directions=directions,
        null_calibration=null,
        result_id="g5.candidate.historical_vector",
    )
    second = evaluate_g5_candidate(
        development,
        qualification,
        monthly_reference=monthly,
        daily_reference=daily,
        sobol_directions=directions,
        null_calibration=null,
        result_id="g5.candidate.historical_vector",
    )

    assert first.fingerprint == second.fingerprint
    assert first.canonical_results is None
    assert not first.exact_canonical_inputs
    assert not first.passed
    with pytest.raises(ModelError, match="canonical thresholds"):
        activate_g5_candidate_generation(
            g3_evidence=object(),  # type: ignore[arg-type]
            null_calibration=null,
            validation_store=object(),  # type: ignore[arg-type]
            evidence_store=object(),  # type: ignore[arg-type]
            seed_registry_fingerprint=_fp("seed"),
            g5_source_code_fingerprint=_fp("source"),
            g5_dependency_lock_fingerprint=_fp("lock"),
        )


def test_candidate_rows_are_reference_independent_and_matrix_contract_is_exact() -> (
    None
):
    rng = np.random.default_rng(91)
    monthly_data = rng.normal(0.0, 0.01, (72, 3))
    daily_data = rng.normal(0.0, 0.005, (64, 3))
    monthly_reference = compute_g5_reference_observables(
        build_g5_reference_data(monthly_data, np.arange(72) % 4, frequency="monthly")
    )
    daily_reference = compute_g5_reference_observables(
        build_g5_reference_data(daily_data, np.arange(64) % 4, frequency="daily")
    )
    lf_rows = tuple(
        compute_g5_path_observables(
            rng.normal(0.0, 0.01, (72, 3)),
            frequency="monthly",
            alpha_sharpes=np.zeros(9),
            selection_uniform=(index + 0.5) / 4,
            path_rank=index + 1,
        )
        for index in range(4)
    )
    hf_rows = tuple(
        compute_g5_path_observables(
            rng.normal(0.0, 0.005, (64, 3)),
            frequency="daily",
            alpha_sharpes=np.zeros(9),
            selection_uniform=(index + 0.5) / 4,
            path_rank=index + 1,
        )
        for index in range(4)
    )
    directions = registered_sobol_directions(master_seed=123, scientific_version=11)

    typed = aggregate_g5_cluster_observables(
        lf_rows,
        hf_rows,
        monthly_reference,
        daily_reference,
        sobol_directions=directions,
    )
    matrix = aggregate_g5_cluster_matrices(
        pack_g5_path_observables(lf_rows, frequency="monthly"),
        pack_g5_path_observables(hf_rows, frequency="daily"),
        monthly_reference,
        daily_reference,
        sobol_directions=directions,
    )

    assert typed.fingerprint == matrix.fingerprint
    assert typed.estimator_contract_fingerprint == G5_ESTIMATOR_CONTRACT_FINGERPRINT
    assert all(
        row.estimator_contract_fingerprint == G5_ESTIMATOR_CONTRACT_FINGERPRINT
        for row in lf_rows
    )


def test_covariance_uses_mean_of_per_window_matrices_and_latent_is_separate() -> None:
    rng = np.random.default_rng(92)
    monthly = compute_g5_reference_observables(
        build_g5_reference_data(
            rng.normal(0.0, 0.01, (72, 3)), np.arange(72) % 4, frequency="monthly"
        )
    )
    daily = compute_g5_reference_observables(
        build_g5_reference_data(
            rng.normal(0.0, 0.005, (64, 3)), np.arange(64) % 4, frequency="daily"
        )
    )
    lf = np.tile(monthly.values, (2, 1))
    hf = np.tile(daily.values, (2, 1))
    covariance_indices = [
        G5_PATH_OBSERVABLE_INDEX[f"covariance.{row}.{column}"]
        for row in range(3)
        for column in range(3)
    ]
    reference_covariance = monthly.values[covariance_indices]
    lf[0, covariance_indices] = reference_covariance * 0.5
    lf[1, covariance_indices] = reference_covariance * 1.5
    for matrix in (lf, hf):
        matrix[:, G5_PATH_OBSERVABLE_INDEX["meta.selection_uniform"]] = (0.25, 0.75)
        matrix[:, G5_PATH_OBSERVABLE_INDEX["meta.path_rank"]] = (1.0, 2.0)
    directions = registered_sobol_directions(master_seed=123, scientific_version=11)

    ordinary = aggregate_g5_cluster_matrices(
        lf, hf, monthly, daily, sobol_directions=directions
    )
    latent_failure = aggregate_g5_cluster_matrices(
        lf,
        hf,
        monthly,
        daily,
        sobol_directions=directions,
        latent_chain_cap_ratio=9.0,
        latent_chain_evaluated=True,
    )

    assert dict(ordinary.endpoint_values)[
        "monthly.covariance_correlation"
    ] == pytest.approx(0.0, abs=1e-12)
    assert ordinary.family_statistics == latent_failure.family_statistics
    assert latent_failure.latent_chain_cap_ratio == 9.0
    assert latent_failure.latent_chain_evaluated


def test_latent_chain_compares_to_fitted_k4_law_not_decoded_history() -> None:
    transition = np.full((4, 4), 0.1)
    np.fill_diagonal(transition, 0.7)
    reference = build_g5_latent_chain_reference(np.full(4, 0.25), transition)
    rng = np.random.default_rng(93)
    paths = []
    for _ in range(40):
        states = np.empty(600, dtype=np.int64)
        states[0] = rng.choice(4, p=np.full(4, 0.25))
        for index in range(1, states.size):
            states[index] = rng.choice(4, p=transition[states[index - 1]])
        paths.append(states)

    assert compute_g5_latent_chain_cap_ratio(paths, reference) < 1.0


def test_candidate_resource_projection_is_linear_and_under_small_memory_budget() -> (
    None
):
    projection = project_g5_candidate_resources(
        _bundle("development", elapsed=2.0), physical_memory_bytes=16 * 1024**3
    )

    assert projection.target_rows == 3_132_000
    assert projection.projected_generation_seconds > 2.0
    assert projection.estimated_peak_memory_bytes < 1024**3
    assert projection.memory_passed


def test_exact_meta_validation_passes_only_50_clean_and_all_power_detections() -> None:
    def null_suite(replicate: int) -> dict[str, float]:
        return {
            "g5_a": 0.001 if replicate < 50 else 0.5,
            "g5_b": 0.5,
            "g5_c": 0.5,
            "g5_d": 0.5,
        }

    result = run_g5_meta_validation(
        null_suite,
        lambda defect, replicate: True,
        threshold_fingerprint=_fp("threshold"),
        qualification_results_fingerprint=_fp("qualification"),
        result_id="g5.meta.historical_vector",
    )

    assert result.false_rejections == 50
    assert result.detection_counts == tuple((name, 1000) for name in POWER_CLAIM_NAMES)
    assert result.type_i_audit is not None and result.type_i_audit.passed
    assert result.power_audit is not None and result.power_audit.passed
    assert result.canonical_results is not None
    assert result.canonical_results.passed
    assert result.passed
    assert project_g5_meta_resources(result).suite_calls_per_replicate == 13


def test_meta_validation_enforces_37_and_925_boundaries() -> None:
    failing_claim = POWER_CLAIM_NAMES[0]

    result = run_g5_meta_validation(
        lambda replicate: {
            "g5_a": 0.001 if replicate < 36 else 0.5,
            "g5_b": 0.5,
            "g5_c": 0.5,
            "g5_d": 0.5,
        },
        lambda defect, replicate: not (
            defect.claim_name == failing_claim and replicate >= 924
        ),
        threshold_fingerprint=_fp("threshold"),
        qualification_results_fingerprint=_fp("qualification"),
        result_id="g5.meta.failure",
    )

    assert result.type_i_audit is not None and not result.type_i_audit.passed
    assert result.power_audit is not None and not result.power_audit.passed
    assert dict(result.detection_counts)[failing_claim] == 924
    assert result.canonical_results is not None
    assert not result.canonical_results.passed
    assert not result.passed


def test_reduced_meta_run_cannot_create_canonical_results() -> None:
    result = run_g5_meta_validation(
        lambda replicate: dict.fromkeys(("g5_a", "g5_b", "g5_c", "g5_d"), 0.5),
        lambda defect, replicate: True,
        threshold_fingerprint=_fp("threshold"),
        qualification_results_fingerprint=_fp("qualification"),
        result_id="g5.meta.reduced",
        replicate_count=10,
    )
    assert result.canonical_results is None
    assert not result.passed


def test_material_numeric_defects_hit_oracles_without_mutation() -> None:
    rng = np.random.default_rng(44)
    development = tuple(rng.normal(0.0, 0.01, (120, 3)) for _ in range(4))
    qualification = tuple(rng.normal(0.0, 0.01, (120, 3)) for _ in range(4))
    pristine = tuple(item.copy() for item in (*development, *qualification))

    linear = apply_registered_return_defect(
        "linear_alpha_sharpe", development, qualification, annualization=12
    )
    covariance = apply_registered_return_defect(
        "normalized_covariance", development, qualification, annualization=12
    )
    correlation = apply_registered_return_defect(
        "maximum_correlation", development, qualification, annualization=12
    )

    assert linear.oracle_value == pytest.approx(0.10, abs=1e-12)
    assert covariance.oracle_value == pytest.approx(0.10, abs=1e-10)
    assert correlation.oracle_value == pytest.approx(0.05, abs=1e-10)
    assert all(
        np.array_equal(observed, expected)
        for observed, expected in zip(
            (*development, *qualification), pristine, strict=True
        )
    )


def test_fixed_adversaries_detect_linear_quadratic_and_exact_memory_defects() -> None:
    rng = np.random.default_rng(94)
    development = tuple(rng.normal(0.0, 0.01, (600, 3)) for _ in range(8))
    qualification = tuple(rng.normal(0.0, 0.01, (600, 3)) for _ in range(20))

    linear = apply_registered_return_defect(
        "linear_alpha_sharpe", development, qualification, annualization=12
    )
    linear_model = fit_g5_adversary_model(linear.development_paths, frequency="monthly")
    linear_scores = np.vstack(
        [
            score_g5_adversary_path(linear_model, path)
            for path in linear.qualification_paths
        ]
    )
    assert (
        np.mean(linear_scores[:, 0])
        + 2.0 * np.std(linear_scores[:, 0], ddof=1) / np.sqrt(linear_scores.shape[0])
        > 0.10
    )

    quadratic = apply_registered_return_defect(
        "nonlinear_monthly", development, qualification, annualization=12
    )
    quadratic_model = fit_g5_adversary_model(
        quadratic.development_paths, frequency="monthly"
    )
    quadratic_scores = np.vstack(
        [
            score_g5_adversary_path(quadratic_model, path)
            for path in quadratic.qualification_paths
        ]
    )
    assert (
        np.mean(quadratic_scores[:, 3])
        + 2.0
        * np.std(quadratic_scores[:, 3], ddof=1)
        / np.sqrt(quadratic_scores.shape[0])
        > 0.10
    )

    memorized = apply_registered_return_defect(
        "subsequence_monthly",
        development,
        qualification,
        annualization=12,
        selected_development_paths=(0, 1),
        selected_qualification_paths=(0, 1, 2, 3, 4),
    )
    memory_model = fit_g5_adversary_model(
        memorized.development_paths, frequency="monthly"
    )
    memory_scores = np.vstack(
        [
            score_g5_adversary_path(memory_model, path)
            for path in memorized.qualification_paths
        ]
    )
    assert (
        np.max(np.quantile(np.abs(memory_scores[:, 6:]), 0.95, axis=0, method="higher"))
        > 0.10
    )


def test_desynchronization_clipping_and_subsequence_are_exact_copy_transforms() -> None:
    rng = np.random.default_rng(55)
    development = tuple(rng.normal(0.0, 0.01, (900, 3)) for _ in range(4))
    qualification = tuple(rng.normal(0.0, 0.01, (900, 3)) for _ in range(4))

    desynchronized = apply_registered_return_defect(
        "one_session_desynchronization",
        development,
        qualification,
        annualization=252,
    )
    assert np.array_equal(
        desynchronized.development_paths[0][1:, 2], development[0][:-1, 2]
    )
    clipped = apply_registered_return_defect(
        "tail_clipping", development, qualification, annualization=252
    )
    assert clipped.oracle_value == 0.02
    subsequence = apply_registered_return_defect(
        "subsequence_daily",
        development,
        qualification,
        annualization=252,
        selected_development_paths=(2,),
        selected_qualification_paths=(1,),
    )
    assert np.array_equal(
        subsequence.development_paths[2][756 : 756 + 64],
        development[2][692:756],
    )
    assert np.array_equal(subsequence.development_paths[0], development[0])


def test_transition_hmm_and_registry_are_exact() -> None:
    transition = np.asarray(((0.8, 0.2), (0.3, 0.7)))
    changed = perturb_transition_matrix(transition)
    assert changed == pytest.approx(
        np.asarray(((0.9 / 1.1, 0.2 / 1.1), (0.3 / 1.1, 0.8 / 1.1)))
    )
    assert np.allclose(changed.sum(axis=1), 1.0)
    assert fixed_hmm_meta_means("hmm_merged_states")[:, 0] == pytest.approx(
        (-1.0, -1.0, 1.0)
    )
    assert tuple(item.claim_name for item in G5_DEFECT_REGISTRY) == POWER_CLAIM_NAMES
    with pytest.raises(ValidationError, match="registered quarter"):
        apply_registered_return_defect(
            "subsequence_monthly",
            tuple(np.zeros((100, 3)) for _ in range(4)),
            tuple(np.zeros((100, 3)) for _ in range(4)),
            annualization=12,
        )
