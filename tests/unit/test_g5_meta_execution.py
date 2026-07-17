from __future__ import annotations

import copy
import hashlib
import pickle

import numpy as np
import pytest

from prpg.errors import IntegrityError, ValidationError
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.hmm import validate_markov_chain
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificGenerationPolicy,
)
from prpg.simulation.serial import (
    G5MetaSerialNumericalInput,
    _mint_g5_meta_serial_numerical_input,
    materialize_g5_meta_rehearsal_base_path,
    verify_g5_meta_serial_numerical_input,
)
from prpg.validation.meta_execution import (
    G5_RETURN_META_BLOCKED_PREREQUISITE,
    G5_RETURN_META_CANONICAL_GEOMETRY,
    G5_RETURN_META_DEFECT_CLAIMS,
    build_g5_return_meta_pilot_geometry,
    project_g5_return_meta_resources,
    run_g5_return_meta_infrastructure,
    verify_g5_return_meta_infrastructure_run,
)


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _numerical_input() -> G5MetaSerialNumericalInput:
    rng = np.random.default_rng(71)
    transition = np.full((4, 4), 0.05, dtype=np.float64)
    np.fill_diagonal(transition, 0.85)
    monthly_rows = 320
    daily_rows = 1_200
    monthly_states = np.arange(monthly_rows, dtype=np.int64) % 4
    daily_states = np.arange(daily_rows, dtype=np.int64) % 4
    policy = ScientificGenerationPolicy(
        schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=6,
        daily_block_length=60,
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
        neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
        initial_state="stationary",
        synchronized_return_vectors=True,
    )
    grid = np.asarray(CANONICAL_NEUTRAL_LAMBDA_GRID, dtype=np.float64)
    return _mint_g5_meta_serial_numerical_input(
        source_generation_input_fingerprint=_fp("source"),
        master_seed=90210,
        scientific_version=11,
        selected_k=4,
        configured_years=50,
        generation_policy=policy,
        stationary_distribution=validate_markov_chain(
            transition
        ).stationary_distribution,
        transition_matrix=transition,
        monthly_returns=rng.normal(0.001, 0.02, (monthly_rows, 3)),
        daily_returns=rng.normal(0.0001, 0.007, (daily_rows, 3)),
        monthly_source_states=monthly_states,
        daily_source_states=daily_states,
        monthly_continuation_links=np.ones(monthly_rows - 1, dtype=np.bool_),
        daily_continuation_links=np.ones(daily_rows - 1, dtype=np.bool_),
        monthly_kernel_lambdas=grid,
        daily_kernel_lambdas=grid,
        monthly_kernel_offsets=np.zeros((grid.size, 4, 3), dtype=np.float64),
        daily_kernel_offsets=np.zeros((grid.size, 4, 3), dtype=np.float64),
    )


def _pilot_geometry():
    return build_g5_return_meta_pilot_geometry(
        development_lf_paths=4,
        development_hf_paths=4,
        qualification_lf_paths=4,
        qualification_hf_paths=4,
        years=6,
    )


def test_purpose_12_13_infrastructure_is_complete_and_nonauthorizing() -> None:
    result = run_g5_return_meta_infrastructure(
        _numerical_input(),
        replicate_count=1,
        workers=1,
        geometry=_pilot_geometry(),
    )

    assert verify_g5_return_meta_infrastructure_run(result) is result
    assert result.generation_authorized is False
    assert result.scientific_meta_ready is False
    assert result.blocked_prerequisite == G5_RETURN_META_BLOCKED_PREREQUISITE
    assert not result.exact_registered_replicates
    assert not result.canonical_geometry
    receipt = result.replicate_receipts[0]
    assert len(receipt.purpose_12_key_fingerprints) == 6
    assert len(receipt.purpose_13_key_fingerprints) == 2
    assert tuple(item.claim_name for item in receipt.fixture_receipts) == (
        G5_RETURN_META_DEFECT_CLAIMS
    )
    transition = next(
        item
        for item in receipt.fixture_receipts
        if item.claim_name == "transition_perturbation"
    )
    assert dict(transition.oracle_values)["P_prime_maximum_formula_error"] <= 1e-12
    assert all(item.oracle_values for item in receipt.fixture_receipts)


def test_pristine_and_fixture_fingerprints_are_equal_for_one_vs_nine_workers() -> None:
    numerical = _numerical_input()
    geometry = _pilot_geometry()
    serial = run_g5_return_meta_infrastructure(
        numerical, replicate_count=9, workers=1, geometry=geometry
    )
    try:
        parallel = run_g5_return_meta_infrastructure(
            numerical, replicate_count=9, workers=9, geometry=geometry
        )
    except PermissionError:
        pytest.skip("sandbox forbids the host semaphore capacity query")

    assert serial.fingerprint == parallel.fingerprint
    assert tuple(item.fingerprint for item in serial.replicate_receipts) == tuple(
        item.fingerprint for item in parallel.replicate_receipts
    )
    assert serial.execution_workers == 1
    assert parallel.execution_workers == 9


def test_transition_wrapper_reuses_draws_but_changes_registered_path() -> None:
    numerical = _numerical_input()
    rows = 6 * 12
    rng = np.random.default_rng(8)
    state = np.full(rows, 0.856, dtype=np.float64)
    state[0] = 0.10
    restart = rng.random(rows)
    source = rng.random(rows)
    pristine = materialize_g5_meta_rehearsal_base_path(
        numerical,
        family="LF",
        path_entity=1,
        years=6,
        state_uniforms=state,
        restart_uniforms=restart,
        source_rank_uniforms=source,
    )
    changed_transition = numerical.transition_matrix.copy()
    changed_transition[np.diag_indices_from(changed_transition)] += 0.10
    changed_transition /= 1.10
    changed = materialize_g5_meta_rehearsal_base_path(
        numerical,
        family="LF",
        path_entity=1,
        years=6,
        state_uniforms=state,
        restart_uniforms=restart,
        source_rank_uniforms=source,
        transition_matrix=changed_transition,
    )

    assert pristine.generation_authorized is False
    assert changed.generation_authorized is False
    assert pristine.fingerprint != changed.fingerprint
    assert pristine.state_uniforms_consumed == changed.state_uniforms_consumed == rows


def test_spawn_safe_numerical_input_detects_content_mutation() -> None:
    numerical = pickle.loads(pickle.dumps(_numerical_input()))
    assert verify_g5_meta_serial_numerical_input(numerical) is numerical
    numerical.monthly_returns[0, 0] += 1.0
    with pytest.raises(IntegrityError, match="fingerprint"):
        verify_g5_meta_serial_numerical_input(numerical)


def test_sealed_run_rejects_copy_and_projection_is_infrastructure_only() -> None:
    result = run_g5_return_meta_infrastructure(
        _numerical_input(),
        replicate_count=1,
        workers=1,
        geometry=_pilot_geometry(),
    )
    with pytest.raises(IntegrityError, match="not minted"):
        verify_g5_return_meta_infrastructure_run(copy.copy(result))

    projection = project_g5_return_meta_resources(
        result,
        target_workers=9,
        physical_memory_bytes=64 * 1024**3,
    )
    assert projection.target_replicates == 1_000
    assert projection.target_workers == 9
    assert projection.projected_seconds > result.elapsed_seconds
    assert projection.memory_passed


def test_geometry_and_replicate_bounds_fail_before_execution() -> None:
    assert G5_RETURN_META_CANONICAL_GEOMETRY.base_rows_per_replicate == 3_132_000
    with pytest.raises(ValidationError, match="noncanonical"):
        build_g5_return_meta_pilot_geometry(
            development_lf_paths=100,
            development_hf_paths=20,
            qualification_lf_paths=500,
            qualification_hf_paths=200,
            years=50,
        )
    with pytest.raises(ValidationError, match="exceeds 1,000"):
        run_g5_return_meta_infrastructure(
            _numerical_input(),
            replicate_count=1_001,
            workers=1,
            geometry=_pilot_geometry(),
        )
