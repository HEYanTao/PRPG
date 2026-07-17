from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from prpg.errors import IntegrityError, ModelError, ValidationError
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.g5_generation_authority import (
    G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING,
    G5_THRESHOLD_ESTIMATOR_BINDING,
    G5_THRESHOLD_POLICY_BINDING,
    G5_THRESHOLD_POLICY_STREAM_BINDING,
    g5_policy_stream_fingerprint,
)
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificGenerationPolicy,
)
from prpg.validation.adversaries import (
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    fit_g5_adversary_pair,
)
from prpg.validation.meta_validation import G5_DEFECT_REGISTRY
from prpg.validation.qualification import (
    G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    build_g5_family_null_calibration,
    build_g5_reference_data,
)
from prpg.validation.size_aware_null import (
    G5_ADVERSARY_FIT_BINDING,
    G5_NULL_RESULT_MATRIX_NAME,
    G5SizeAwareNullRun,
    build_g5_size_aware_null_subject,
    build_nonauthorizing_g5_meta_callbacks,
    compare_g5_size_aware_worker_parity,
    compare_literal_and_proposed_g5_null_designs,
    estimate_g5_size_aware_reservoir_bytes,
    load_g5_size_aware_null_result,
    materialize_g5_size_aware_null_reservoir,
    project_g5_size_aware_null_resources,
    run_g5_size_aware_null,
    verify_g5_size_aware_null_run,
)
from prpg.validation.sobol import registered_sobol_directions


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _policy() -> ScientificGenerationPolicy:
    return ScientificGenerationPolicy(
        schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=4,
        daily_block_length=20,
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
        neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


def _returns(seed: int, observations: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    covariance = np.asarray(((1.0, 0.20, 0.10), (0.20, 0.70, 0.25), (0.10, 0.25, 0.80)))
    values = rng.multivariate_normal(np.zeros(3), covariance, observations) / 100.0
    values[:, 0] += np.linspace(-1e-5, 1e-5, observations)
    return np.asarray(values, dtype=np.float64)


def _fixture() -> dict[str, object]:
    policy = _policy()
    monthly_values = _returns(101, 24)
    daily_values = _returns(202, 300)
    monthly = build_g5_reference_data(
        monthly_values, np.arange(24) % 4, frequency="monthly"
    )
    daily = build_g5_reference_data(daily_values, np.arange(300) % 4, frequency="daily")
    fit = fit_g5_adversary_pair(
        tuple(_returns(300 + index, 24) for index in range(4)),
        tuple(_returns(400 + index, 300) for index in range(4)),
    )
    policy_fingerprint = hashlib.sha256(
        (
            __import__("json").dumps(
                policy.as_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    bindings = {
        G5_THRESHOLD_ESTIMATOR_BINDING: G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        G5_THRESHOLD_POLICY_BINDING: policy_fingerprint,
        G5_THRESHOLD_POLICY_STREAM_BINDING: g5_policy_stream_fingerprint(policy),
        G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING: (
            G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        ),
        G5_ADVERSARY_FIT_BINDING: fit.fingerprint,
    }
    return {
        "policy": policy,
        "monthly": monthly,
        "daily": daily,
        "monthly_links": np.ones(23, dtype=np.bool_),
        "daily_links": np.ones(299, dtype=np.bool_),
        "fit": fit,
        "bindings": bindings,
    }


def _materialize(
    root: Path,
    fixture: dict[str, object],
    *,
    workers: int,
    replicate_count: int = 2,
):
    directions = registered_sobol_directions(master_seed=9182, scientific_version=1)
    reservoir = materialize_g5_size_aware_null_reservoir(
        root / "reservoir",
        mode="rehearsal",
        generation_policy=fixture["policy"],  # type: ignore[arg-type]
        monthly_reference=fixture["monthly"],  # type: ignore[arg-type]
        daily_reference=fixture["daily"],  # type: ignore[arg-type]
        monthly_continuation_links=fixture["monthly_links"],
        daily_continuation_links=fixture["daily_links"],
        monthly_mean_block_length=4.0,
        daily_mean_block_length=20.0,
        adversary_fit=fixture["fit"],  # type: ignore[arg-type]
        sobol_directions=directions,
        master_seed=9182,
        stream_scientific_version=1,
        threshold_set_id="g5.rehearsal.size-aware",
        binding_fingerprints=fixture["bindings"],  # type: ignore[arg-type]
        replicate_count=replicate_count,
        lf_cluster_count=3,
        hf_cluster_count=3,
        workers=workers,
        batch_size=1,
    )
    run = run_g5_size_aware_null(
        reservoir,
        root / "result",
        sobol_directions=directions,
        workers=workers,
        batch_size=1,
    )
    return directions, reservoir, run


def test_rehearsal_materializes_exact_cluster_and_durable_matrix(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    directions, reservoir, run = _materialize(tmp_path, fixture, workers=1)

    assert reservoir.mode == "rehearsal"
    assert reservoir.replicate_count == 2
    assert reservoir.lf_cluster_count == reservoir.hf_cluster_count == 3
    assert not reservoir.canonical_eligible
    assert run.calibration.family_statistics.shape == (2, 4)
    assert not run.loaded_from_store
    assert not run.generation_authorized
    assert np.load(
        tmp_path / "reservoir" / "lf-cluster-indices.npy", allow_pickle=False
    ).shape == (2, 3)

    subject = build_g5_size_aware_null_subject(
        generation_policy=fixture["policy"],  # type: ignore[arg-type]
        monthly_reference=fixture["monthly"],  # type: ignore[arg-type]
        daily_reference=fixture["daily"],  # type: ignore[arg-type]
        monthly_continuation_links=fixture["monthly_links"],
        daily_continuation_links=fixture["daily_links"],
        monthly_mean_block_length=4.0,
        daily_mean_block_length=20.0,
        adversary_fit=fixture["fit"],  # type: ignore[arg-type]
    )
    loaded = load_g5_size_aware_null_result(
        tmp_path / "result",
        expected_subject_fingerprint=subject.fingerprint,
        expected_policy_stream_fingerprint=g5_policy_stream_fingerprint(
            fixture["policy"]  # type: ignore[arg-type]
        ),
        expected_adversary_fit_fingerprint=fixture["fit"].fingerprint,  # type: ignore[union-attr]
        expected_sobol_direction_fingerprint=directions.fingerprint,
        expected_binding_fingerprints=fixture["bindings"],  # type: ignore[arg-type]
    )
    assert loaded.loaded_from_store
    assert loaded.family_statistics_sha256 == run.family_statistics_sha256
    assert not loaded.canonical_eligible
    assert loaded.calibration.canonical_thresholds is None


def test_rehearsal_cannot_touch_selected_policy_stream(tmp_path: Path) -> None:
    fixture = _fixture()
    directions = registered_sobol_directions(master_seed=9182, scientific_version=11)

    with pytest.raises(ModelError, match="cannot inspect"):
        materialize_g5_size_aware_null_reservoir(
            tmp_path / "forbidden",
            mode="rehearsal",
            generation_policy=fixture["policy"],  # type: ignore[arg-type]
            monthly_reference=fixture["monthly"],  # type: ignore[arg-type]
            daily_reference=fixture["daily"],  # type: ignore[arg-type]
            monthly_continuation_links=fixture["monthly_links"],
            daily_continuation_links=fixture["daily_links"],
            monthly_mean_block_length=4.0,
            daily_mean_block_length=20.0,
            adversary_fit=fixture["fit"],  # type: ignore[arg-type]
            sobol_directions=directions,
            master_seed=9182,
            stream_scientific_version=11,
            threshold_set_id="g5.forbidden",
            binding_fingerprints=fixture["bindings"],  # type: ignore[arg-type]
            replicate_count=2,
            lf_cluster_count=3,
            hf_cluster_count=3,
        )
    assert not (tmp_path / "forbidden").exists()


def test_reservoir_has_no_canonical_execution_mode(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    directions = registered_sobol_directions(master_seed=9182, scientific_version=11)

    with pytest.raises(ValidationError, match="execution mode"):
        materialize_g5_size_aware_null_reservoir(
            tmp_path / "canonical",
            mode="canonical",  # type: ignore[arg-type]
            generation_policy=fixture["policy"],  # type: ignore[arg-type]
            monthly_reference=fixture["monthly"],  # type: ignore[arg-type]
            daily_reference=fixture["daily"],  # type: ignore[arg-type]
            monthly_continuation_links=fixture["monthly_links"],
            daily_continuation_links=fixture["daily_links"],
            monthly_mean_block_length=4.0,
            daily_mean_block_length=20.0,
            adversary_fit=fixture["fit"],  # type: ignore[arg-type]
            sobol_directions=directions,
            master_seed=9182,
            stream_scientific_version=11,
            threshold_set_id="g5.canonical",
            binding_fingerprints=fixture["bindings"],  # type: ignore[arg-type]
            replicate_count=50_000,
        )
    assert not (tmp_path / "canonical").exists()

    retired_directions = registered_sobol_directions(
        master_seed=9182, scientific_version=1
    )
    with pytest.raises(ModelError, match="canonical count"):
        materialize_g5_size_aware_null_reservoir(
            tmp_path / "fifty-thousand-proposed",
            mode="rehearsal",
            generation_policy=fixture["policy"],  # type: ignore[arg-type]
            monthly_reference=fixture["monthly"],  # type: ignore[arg-type]
            daily_reference=fixture["daily"],  # type: ignore[arg-type]
            monthly_continuation_links=fixture["monthly_links"],
            daily_continuation_links=fixture["daily_links"],
            monthly_mean_block_length=4.0,
            daily_mean_block_length=20.0,
            adversary_fit=fixture["fit"],  # type: ignore[arg-type]
            sobol_directions=retired_directions,
            master_seed=9182,
            stream_scientific_version=1,
            threshold_set_id="g5.proposed.forbidden-50k",
            binding_fingerprints=fixture["bindings"],  # type: ignore[arg-type]
            replicate_count=50_000,
        )
    assert not (tmp_path / "fifty-thousand-proposed").exists()


def test_result_is_no_overwrite_and_tamper_evident(tmp_path: Path) -> None:
    fixture = _fixture()
    directions, reservoir, run = _materialize(tmp_path, fixture, workers=1)
    with pytest.raises(ModelError, match="already exists"):
        run_g5_size_aware_null(
            reservoir,
            tmp_path / "result",
            sobol_directions=directions,
        )

    matrix = tmp_path / "result" / G5_NULL_RESULT_MATRIX_NAME
    with matrix.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes((byte[0] ^ 1,)))
    with pytest.raises(IntegrityError, match="matrix file changed"):
        verify_g5_size_aware_null_run(run)
    with pytest.raises(TypeError, match="runner/loader"):
        G5SizeAwareNullRun()


def test_one_vs_nine_is_exact_and_projection_is_a_hard_typed_gate(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    _, serial_reservoir, serial_run = _materialize(
        tmp_path / "serial", fixture, workers=1, replicate_count=1
    )
    with patch("concurrent.futures.process.os.sysconf", return_value=-1):
        _, parallel_reservoir, parallel_run = _materialize(
            tmp_path / "parallel", fixture, workers=9, replicate_count=1
        )

    parity = compare_g5_size_aware_worker_parity(
        serial_reservoir, serial_run, parallel_reservoir, parallel_run
    )
    assert parity.passed
    projection = project_g5_size_aware_null_resources(
        parallel_reservoir,
        parallel_run,
        parity,
        host_fingerprint=_fp("same-host"),
        physical_memory_bytes=16 * 1024**3,
        disk_free_bytes=80 * 1024**3,
        maximum_projected_hours=1_000_000.0,
    )
    assert projection.passed
    assert projection.target_replicates == 50_000
    assert projection.target_workers == 9
    assert projection.projected_total_seconds > 0.0
    assert projection.scientific_cost_complete is False
    assert projection.canonical_authorized is False


def test_resource_formula_is_bounded_on_target_mac_geometry() -> None:
    baseline = estimate_g5_size_aware_reservoir_bytes(
        replicate_count=50_000,
        monthly_observations=217,
        daily_observations=4_547,
        lf_cluster_count=500,
        hf_cluster_count=200,
        has_raw_references=False,
    )
    neutral = estimate_g5_size_aware_reservoir_bytes(
        replicate_count=50_000,
        monthly_observations=217,
        daily_observations=4_547,
        lf_cluster_count=500,
        hf_cluster_count=200,
        has_raw_references=True,
    )

    assert 1_000_000_000 < baseline < 2_000_000_000
    assert baseline < neutral < 3_000_000_000


def test_literal_vs_proposed_workload_is_explicit_and_nonauthorizing() -> None:
    comparison = compare_literal_and_proposed_g5_null_designs(
        monthly_observations=217,
        daily_observations=4_547,
    )

    assert comparison.literal_candidate_histories == 35_000_000
    assert comparison.literal_common_reference_histories == 100_000
    assert comparison.literal_adversary_scored_histories == 35_100_000
    assert comparison.literal_history_rows == 51_133_200_000
    assert comparison.proposed_candidate_histories == 100_000
    assert comparison.proposed_common_reference_histories == 100_000
    assert comparison.proposed_adversary_scored_histories == 200_000
    assert comparison.proposed_history_rows == 476_400_000
    assert comparison.adversary_scoring_reduction_factor == 175.5
    assert comparison.literal_projected_hours is None
    assert comparison.eta_basis == "unavailable_pending_exact_adversary_registry"
    assert comparison.proposed_design_authorized is False


def test_meta_callbacks_compute_designated_holm_and_never_authorize() -> None:
    null_values = np.linspace(0.0, 1.0, 100)
    calibration = build_g5_family_null_calibration(
        dict.fromkeys(("g5_a", "g5_b", "g5_c", "g5_d"), null_values),
        threshold_set_id="g5.meta-pilot",
        binding_fingerprints={"pilot": _fp("pilot")},
        sobol_direction_fingerprint=_fp("sobol"),
    )
    clean = np.full((2, 4), 0.5)
    return_claims = {
        item.claim_name
        for item in G5_DEFECT_REGISTRY
        if item.designated_family != "hmm_identification"
    }
    hmm_claims = {
        item.claim_name
        for item in G5_DEFECT_REGISTRY
        if item.designated_family == "hmm_identification"
    }
    defects = {name: np.full((2, 4), 0.5) for name in return_claims}
    defects["linear_alpha_sharpe"][0, 0] = 10.0
    callbacks = build_nonauthorizing_g5_meta_callbacks(
        calibration,
        clean_statistics=clean,
        defect_statistics=defects,
        hmm_identification_decisions={
            name: np.asarray((True, False), dtype=np.bool_) for name in hmm_claims
        },
        materialization_fingerprints={"actual-suite-producer": _fp("actual")},
    )
    linear = next(
        item for item in G5_DEFECT_REGISTRY if item.claim_name == "linear_alpha_sharpe"
    )
    merged = next(
        item for item in G5_DEFECT_REGISTRY if item.claim_name == "hmm_merged_states"
    )

    assert callbacks.power_suite(linear, 0)
    assert callbacks.power_suite(merged, 0)
    assert not callbacks.power_suite(merged, 1)
    assert set(callbacks.null_suite(0)) == {"g5_a", "g5_b", "g5_c", "g5_d"}
    assert callbacks.canonical_authorized is False

    wrong_family = {name: matrix.copy() for name, matrix in defects.items()}
    wrong_family["linear_alpha_sharpe"][0] = (0.5, 10.0, 0.5, 0.5)
    corrupted = build_nonauthorizing_g5_meta_callbacks(
        calibration,
        clean_statistics=clean,
        defect_statistics=wrong_family,
        hmm_identification_decisions={
            name: np.asarray((True, False), dtype=np.bool_) for name in hmm_claims
        },
        materialization_fingerprints={"actual-suite-producer": _fp("actual")},
    )
    assert not corrupted.power_suite(linear, 0)
