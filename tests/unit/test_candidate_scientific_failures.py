"""Narrow scientific-failure authority for preliminary and per-K blocks."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
from test_g3_calibration_views import (
    _registered_iteration_suite,
    _registered_selection_bootstrap,
)

import prpg.model.block_execution_registered as block_registered
import prpg.model.calibration as calibration_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.block_execution import (
    BlockCalibrationSupportFailure,
    BlockStateModel,
    BlockSupportFailureEvidence,
)
from prpg.model.block_execution_registered import (
    RegisteredBlockSupportFailure,
    is_registered_block_support_failure,
    materialize_registered_block_uniforms,
    registered_block_support_failure_identity,
    run_registered_materialized_block_calibration_outcome,
)
from prpg.model.calibration import (
    RegisteredPreliminaryCalibrationOutcome,
    execute_registered_preliminary_calibrations,
    is_registered_preliminary_calibration_outcome,
    registered_preliminary_calibration_identity,
)
from prpg.model.candidate_gate import (
    execute_registered_candidate_viability,
    run_owner_fixed_k4_selection,
)
from prpg.model.candidate_suite import (
    assemble_registered_design_candidate_suite,
    registered_design_candidate_suite_identity,
)
from prpg.model.input import CalibrationSlice
from prpg.model.registered_fixed_point import (
    assemble_registered_design_fixed_point,
)


def _preliminary_slice() -> CalibrationSlice:
    monthly_rows = 60
    daily_rows = monthly_rows * 21
    monthly_continuation = np.ones(monthly_rows, dtype=np.bool_)
    monthly_continuation[0] = False
    daily_continuation = np.ones(daily_rows, dtype=np.bool_)
    daily_continuation[0] = False
    rng = np.random.default_rng(901)
    return CalibrationSlice(
        role="design",
        monthly_returns=np.ones((monthly_rows, 3), dtype=np.float64),
        daily_returns=rng.normal(size=(daily_rows, 3)),
        macro_features=rng.normal(size=(monthly_rows, 4)),
        monthly_period_ordinals=np.arange(monthly_rows, dtype=np.int64),
        daily_session_ns=np.arange(daily_rows, dtype=np.int64),
        monthly_continuation=monthly_continuation,
        daily_continuation=daily_continuation,
    )


def _block_failure_sources() -> tuple[object, BlockStateModel, np.ndarray, np.ndarray]:
    materialization = materialize_registered_block_uniforms(
        master_seed=123,
        scientific_version=3,
        artifact="design",
        frequency="monthly",
        history_count=7,
        strict_canonical=False,
        monthly_segment_lengths=(12,),
    )
    model = BlockStateModel(
        stationary_distribution=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )
    states = np.arange(60, dtype=np.int64) % 2
    links = np.ones(len(states) - 1, dtype=np.bool_)
    return materialization, model, states, links


def test_preliminary_scientific_nonattainment_is_exact_and_tamper_evident() -> None:
    source = _preliminary_slice()
    outcome = execute_registered_preliminary_calibrations(source, role="design")
    identity = registered_preliminary_calibration_identity(outcome)

    assert isinstance(outcome, RegisteredPreliminaryCalibrationOutcome)
    assert is_registered_preliminary_calibration_outcome(outcome)
    assert not identity.passed
    assert identity.failure_stage == "monthly"
    assert identity.failure_code == "numerical_failure"
    assert identity.failure_type == "PreliminaryPPWNumericalFailure"
    assert outcome.calibrations is None
    assert not is_registered_preliminary_calibration_outcome(copy.copy(outcome))

    original = outcome.failure_message
    object.__setattr__(outcome, "failure_message", "favorable rewrite")
    try:
        assert not is_registered_preliminary_calibration_outcome(outcome)
    finally:
        object.__setattr__(outcome, "failure_message", original)
    assert is_registered_preliminary_calibration_outcome(outcome)

    equivalent = execute_registered_preliminary_calibrations(
        replace(source),
        role="design",
    )
    assert equivalent is not outcome
    assert (
        registered_preliminary_calibration_identity(equivalent).source_fingerprint
        == identity.source_fingerprint
    )


@pytest.mark.parametrize(
    "fault",
    [
        ModelError("generic model fault"),
        IntegrityError("integrity fault"),
        RuntimeError("runtime fault"),
        OSError("filesystem fault"),
        MemoryError("memory fault"),
    ],
)
def test_preliminary_executor_propagates_every_nonregistered_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault: BaseException,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise fault

    monkeypatch.setattr(calibration_module, "estimate_frequency_start", fail)
    with pytest.raises(type(fault), match="fault"):
        execute_registered_preliminary_calibrations(
            _preliminary_slice(),
            role="design",
        )


def test_block_support_failure_binds_seed_geometry_sources_and_tamper() -> None:
    materialization, model, states, links = _block_failure_sources()
    outcome = run_registered_materialized_block_calibration_outcome(
        materialization,  # type: ignore[arg-type]
        model=model,
        starting_length=2,
        historical_source_states=states,
        source_continuation_links=links,
    )
    assert isinstance(outcome, RegisteredBlockSupportFailure)
    identity = registered_block_support_failure_identity(outcome)
    assert is_registered_block_support_failure(outcome)
    assert identity.master_seed == 123
    assert identity.scientific_version == 3
    assert identity.artifact == "design"
    assert identity.frequency == "monthly"
    assert identity.selected_k == 2
    assert identity.starting_length == 2
    assert outcome.evidence.geometry.monthly_segment_lengths == (12,)
    assert outcome.evidence.geometry.target_segment_lengths == (12,)
    assert outcome.evidence.failure_code == "historical_support_unattained"
    assert not is_registered_block_support_failure(copy.copy(outcome))

    evidence = outcome.evidence
    object.__setattr__(
        outcome,
        "evidence",
        replace(evidence, failure_message="favorable rewrite"),
    )
    try:
        assert not is_registered_block_support_failure(outcome)
    finally:
        object.__setattr__(outcome, "evidence", evidence)
    assert is_registered_block_support_failure(outcome)

    retained_states = outcome._source_states
    object.__setattr__(outcome, "_source_states", retained_states.copy())
    try:
        assert not is_registered_block_support_failure(outcome)
    finally:
        object.__setattr__(outcome, "_source_states", retained_states)
    assert is_registered_block_support_failure(outcome)


@pytest.mark.parametrize(
    "fault",
    [
        ModelError("generic model fault"),
        IntegrityError("integrity fault"),
        RuntimeError("runtime fault"),
        OSError("filesystem fault"),
        MemoryError("memory fault"),
    ],
)
def test_block_outcome_executor_propagates_every_nonregistered_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault: BaseException,
) -> None:
    materialization, model, states, links = _block_failure_sources()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise fault

    monkeypatch.setattr(
        block_registered,
        "run_registered_materialized_block_calibration",
        fail,
    )
    with pytest.raises(type(fault), match="fault"):
        run_registered_materialized_block_calibration_outcome(
            materialization,  # type: ignore[arg-type]
            model=model,
            starting_length=2,
            historical_source_states=states,
            source_continuation_links=links,
        )


def test_per_k_block_failure_becomes_inadmissible_without_aborting_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successful = _registered_iteration_suite(monkeypatch)
    row = successful.rows[0]
    fit = successful._fit_set.attempts[0].fit
    assert fit is not None
    assert row.decoded_return_pools is not None
    assert row.daily_block is not None
    evidence = BlockSupportFailureEvidence(
        geometry=successful._monthly_materialization.geometry,
        histories=successful._monthly_materialization.history_count,
        starting_length=successful.monthly_input_start,
        structurally_feasible_lengths=(),
        simulations=(),
        failure_code="historical_support_unattained",
        failure_message="controlled code-owned support nonattainment",
        failure_details=(("frequency", "monthly"),),
    )
    success_executor = block_registered.run_registered_materialized_block_calibration

    def scientific_failure(*_args: object, **_kwargs: object) -> object:
        raise BlockCalibrationSupportFailure(evidence)

    monkeypatch.setattr(
        block_registered,
        "run_registered_materialized_block_calibration",
        scientific_failure,
    )
    monthly_failure = run_registered_materialized_block_calibration_outcome(
        successful._monthly_materialization,
        model=fit,
        starting_length=successful.monthly_input_start,
        historical_source_states=row.decoded_return_pools.monthly_states,
        source_continuation_links=successful._design_slice.monthly_continuation[1:],
    )
    monkeypatch.setattr(
        block_registered,
        "run_registered_materialized_block_calibration",
        success_executor,
    )
    assert isinstance(monthly_failure, RegisteredBlockSupportFailure)

    suite = assemble_registered_design_candidate_suite(
        successful._design_slice,
        successful._design_features,
        successful._fit_set,
        preliminary_outcome=successful._preliminary_outcome,
        macro_materialization=successful._macro_materialization,
        monthly_materialization=successful._monthly_materialization,
        daily_materialization=successful._daily_materialization,
        rolling_by_k=tuple(item.rolling for item in successful.rows),
        macro_by_k=tuple(item.macro_stability for item in successful.rows),
        monthly_blocks_by_k=(monthly_failure, None, None, None),
        daily_blocks_by_k=(row.daily_block, None, None, None),
        master_seed=successful.master_seed,
        scientific_version=successful.scientific_version,
    )
    suite_identity = registered_design_candidate_suite_identity(suite)
    assert suite.rows[0].monthly_block is None
    assert suite.rows[0].monthly_block_failure is monthly_failure
    assert suite_identity.block_core_fingerprints[0][0] is None
    assert suite_identity.block_failure_fingerprints[0][0] == (
        monthly_failure.failure_fingerprint
    )

    viability = execute_registered_candidate_viability(suite)
    row_audit = viability.audit_rows[0]
    assert row_audit.fit_succeeded
    assert not row_audit.viability.admissible
    assert any(
        reason.code == "preselection_check_failed"
        and reason.check_id == "monthly_block_support"
        for reason in row_audit.viability.rejection_reasons
    )
    selection = run_owner_fixed_k4_selection(
        viability,
        _registered_selection_bootstrap(),
    )
    fixed_point = assemble_registered_design_fixed_point(
        ((suite, viability, selection),)
    )
    assert fixed_point.outcome == "no_admissible_candidates"
    assert not fixed_point.passed
