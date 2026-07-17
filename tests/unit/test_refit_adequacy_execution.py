from __future__ import annotations

import copy
import hashlib
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model import refit_adequacy_execution as execution
from prpg.model.adequacy import (
    AdequacyEndpointSupportFailure,
    EndpointDirection,
    EndpointVector,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMCandidateInadmissibility,
    HMMCandidateInadmissibilityEvidence,
    HMMFeatureMatrix,
    HMMParameters,
    NoNumericallyValidHMMRestarts,
    RestartDiagnostic,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
    forward_backward,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
)
from prpg.model.refit_adequacy import (
    MACRO_FEATURE_NAMES,
    RefittedAdequacyEndpoint,
    RefittedAdequacyEndpointSupportFailure,
)
from prpg.model.refit_adequacy_execution import (
    CANONICAL_MINIMUM_SUCCESSES,
    CANONICAL_REFIT_ATTEMPTS,
    run_registered_refitted_adequacy_cell,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    ModelFitScope,
    Stage,
    calibration_entity,
    calibration_rng_key,
    hmm_emission_standard_normals,
)


def _unused_fit(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("outer execution test should inject the attempt result")


def _restart_diagnostic(index: int, *, valid: bool) -> RestartDiagnostic:
    return RestartDiagnostic(
        restart_index=index,
        seed=index,
        converged=valid,
        iterations=10 if valid else 0,
        log_likelihood=-100.0 if valid else None,
        valid=valid,
        raw_minimum_eigenvalue=1.0 if valid else None,
        maximum_floor_relative_change=0.0 if valid else None,
        covariance_condition_number=1.0 if valid else None,
        maximum_likelihood_decrease=0.0 if valid else None,
        preliminary_chain_passed=valid,
        failure_type=None if valid else "NonConvergence",
        failure_message=None if valid else "synthetic numerical non-pass",
    )


def _no_valid_restarts() -> NoNumericallyValidHMMRestarts:
    return NoNumericallyValidHMMRestarts(
        n_states=2,
        restart_diagnostics=tuple(
            _restart_diagnostic(index, valid=False) for index in range(50)
        ),
    )


def _candidate_inadmissibility() -> HMMCandidateInadmissibility:
    reasons = ("state_1_posterior_effective_mass_below_minimum",)
    gate_details = tuple(
        sorted(
            {
                "assigned_state_mean_posteriors": (0.9, 0.9),
                "mean_maximum_posterior": 0.9,
                "minimum_pairwise_mahalanobis_distance": 2.5,
                "minimum_required_effective_mass": 20.0,
                "normalized_posterior_entropy": 0.2,
                "passed": False,
                "posterior_effective_masses": (100.0, 10.0),
                "rejection_reasons": reasons,
            }.items()
        )
    )
    diagnostics = tuple(_restart_diagnostic(index, valid=True) for index in range(50))
    return HMMCandidateInadmissibility(
        HMMCandidateInadmissibilityEvidence(
            n_states=2,
            failure_code="posterior_identification",
            failure_message="synthetic candidate inadmissibility",
            rejection_reasons=reasons,
            gate_details=gate_details,
            best_restart_index=0,
            best_seed=0,
            restart_diagnostics=diagnostics,
        )
    )


def _inputs(
    role: str = "design",
    rows: int = 12,
) -> tuple[GaussianHMMFit, HMMFeatureMatrix]:
    scope = "design_training" if role == "design" else "production_full"
    lengths = (rows,) if role == "design" else (rows - 5, 5)
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        means=np.asarray([[-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0]]),
        tied_covariance=np.eye(4),
    )
    scaler_means = np.zeros(4, dtype=np.float64)
    scaler_scales = np.ones(4, dtype=np.float64)
    values = np.zeros((rows, 4), dtype=np.float64)
    decoded = decode_viterbi(values, parameters, lengths=lengths)
    posterior = forward_backward(values, parameters, lengths=lengths)
    fit = GaussianHMMFit(
        parameters=parameters,
        decoded_states=decoded,
        canonical_to_original=np.asarray([0, 1], dtype=np.int64),
        original_to_canonical=np.asarray([0, 1], dtype=np.int64),
        log_likelihood=posterior.total_log_likelihood,
        bic=100.0,
        parameter_count=25,
        n_observations=rows,
        n_features=4,
        n_states=2,
        lengths=lengths,
        best_restart_index=0,
        best_seed=1,
        restart_diagnostics=(),
        covariance_diagnostic=None,  # type: ignore[arg-type]
        state_path_diagnostics=None,  # type: ignore[arg-type]
        chain_diagnostics=None,  # type: ignore[arg-type]
        forward_backward=posterior,
        identification=None,  # type: ignore[arg-type]
        historical_transition_support=None,  # type: ignore[arg-type]
        scaler_scope=scope,
        scaler_means=scaler_means,
        scaler_standard_deviations=scaler_scales,
        feature_names=MACRO_FEATURE_NAMES,
    )
    features = HMMFeatureMatrix(
        values=values,
        lengths=lengths,
        feature_names=MACRO_FEATURE_NAMES,
        row_labels=tuple(f"row-{row}" for row in range(rows)),
        scaler_means=scaler_means,
        scaler_standard_deviations=scaler_scales,
        scaler_scope=scope,  # type: ignore[arg-type]
    )
    return fit, features


def _observed_endpoint(*_args: object, **_kwargs: object) -> RefittedAdequacyEndpoint:
    values = np.asarray([0.5, 0.5], dtype=np.float64)
    values.setflags(write=False)
    decoded = np.zeros(12, dtype=np.int64)
    counts = np.full((2, 4, 3), 10, dtype=np.int64)
    return RefittedAdequacyEndpoint(
        EndpointVector(("metric/two_sided", "metric/upper_tail"), values),
        (EndpointDirection.TWO_SIDED, EndpointDirection.UPPER_TAIL),
        decoded,
        counts,
    )


def _install_attempt_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    master_seed: int,
    scientific_version: int,
    scope: ModelFitScope,
    failure_attempts: set[int] | None = None,
) -> dict[int, tuple[ModelFitScope, tuple[int, ...]]]:
    first_seed_to_attempt = {
        deterministic_hmm_restart_seeds(
            master_seed=master_seed,
            scientific_version=scientific_version,
            scope=scope,
            replicate=attempt,
            n_states=2,
        )[0]: attempt
        for attempt in range(20)
    }
    calls: dict[int, tuple[ModelFitScope, tuple[int, ...]]] = {}
    failures = failure_attempts or set()

    def fake_attempt(
        _original_fit: object,
        _artifact_features: object,
        state_uniforms: np.ndarray,
        normals: np.ndarray,
        *,
        scope: ModelFitScope,
        seeds: Sequence[int],
        fit_function: object,
    ) -> object:
        assert fit_function is _unused_fit
        attempt = first_seed_to_attempt[int(seeds[0])]
        calls[attempt] = (scope, tuple(int(seed) for seed in seeds))
        if attempt in failures:
            raise RefittedAdequacyEndpointSupportFailure(
                "failure " + "x" * 700,
                details={f"detail-{index}": "y" * 300 for index in range(12)},
            )
        values = np.asarray(
            [float(state_uniforms[0]), abs(float(normals[0, 0]))],
            dtype=np.float64,
        )
        endpoint = RefittedAdequacyEndpoint(
            EndpointVector(
                ("metric/two_sided", "metric/upper_tail"),
                values,
            ),
            (EndpointDirection.TWO_SIDED, EndpointDirection.UPPER_TAIL),
            np.zeros(len(state_uniforms), dtype=np.int64),
            np.full((2, 4, 3), 10, dtype=np.int64),
        )
        return SimpleNamespace(
            endpoint=endpoint,
            refit=SimpleNamespace(
                best_restart_index=attempt % 50,
                best_seed=int(seeds[attempt % 50]),
            ),
            alignment=SimpleNamespace(
                candidate_to_reference=np.asarray([0, 1], dtype=np.int64),
                selected_total_cost=float(attempt) / 10.0,
            ),
        )

    monkeypatch.setattr(
        execution, "build_refitted_adequacy_endpoint", _observed_endpoint
    )
    monkeypatch.setattr(execution, "run_refitted_adequacy_attempt", fake_attempt)
    return calls


def _serial_map(
    function: Callable[[object], object],
    items: Iterable[object],
    *,
    workers: int,
    chunksize: int,
) -> tuple[object, ...]:
    del workers, chunksize
    return tuple(function(item) for item in items)


def _reverse_schedule_map(
    function: Callable[[object], object],
    items: Iterable[object],
    *,
    workers: int,
    chunksize: int,
) -> tuple[object, ...]:
    del workers, chunksize
    jobs = tuple(items)
    results = {
        job.attempts: function(job)  # type: ignore[attr-defined]
        for job in reversed(jobs)
    }
    return tuple(results[job.attempts] for job in jobs)  # type: ignore[attr-defined]


def _run(
    *,
    role: str = "design",
    attempts: int = 5,
    minimum: int = 3,
    workers: int = 1,
    attempt_batch_size: int = 2,
) -> execution.RegisteredRefittedAdequacyBatch:
    fit, features = _inputs(role)
    return run_registered_refitted_adequacy_cell(
        role=role,  # type: ignore[arg-type]
        original_fit=fit,
        artifact_features=features,
        master_seed=123,
        scientific_version=1,
        attempt_count=attempts,
        minimum_successes=minimum,
        workers=workers,
        attempt_batch_size=attempt_batch_size,
        process_chunksize=1,
        strict_canonical=False,
        fit_function=_unused_fit,  # type: ignore[arg-type]
    )


def test_defaults_freeze_canonical_attempt_and_success_counts() -> None:
    parameters = inspect.signature(run_registered_refitted_adequacy_cell).parameters
    assert parameters["attempt_count"].default == CANONICAL_REFIT_ATTEMPTS == 1_000
    assert parameters["minimum_successes"].default == CANONICAL_MINIMUM_SUCCESSES == 950
    assert parameters["strict_canonical"].default is True


def test_registered_streams_scopes_seeds_and_fixed_order_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_attempt_stub(
        monkeypatch,
        master_seed=123,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
        failure_attempts={2},
    )
    seen_execution: list[tuple[int, int]] = []

    def mapped(
        function: Callable[[object], object],
        items: Iterable[object],
        *,
        workers: int,
        chunksize: int,
    ) -> tuple[object, ...]:
        seen_execution.append((workers, chunksize))
        return _serial_map(function, items, workers=workers, chunksize=chunksize)

    monkeypatch.setattr(execution, "deterministic_process_map", mapped)
    result = _run(workers=3)

    assert seen_execution == [(3, 1)]
    assert result.attempts == 5
    assert result.successful_attempts == 4
    assert result.failed_attempts == 1
    assert result.successful_attempt_indices == (0, 1, 3, 4)
    assert result.success_requirement_passed
    assert result.ready_for_holm
    assert result.max_statistic_cell is not None
    assert result.unadjusted_95_max_statistic_quantile is not None
    assert result.successful_endpoint_matrix.shape == (4, 2)
    assert not result.successful_endpoint_matrix.flags.writeable
    assert not result.observed_endpoint_values.flags.writeable
    assert result.state_streams == result.emission_streams == 5
    assert result.restart_seed_sets == 5
    source_fit, source_features = execution.registered_refitted_adequacy_sources(result)
    assert result.parent_model_fingerprint == gaussian_hmm_fit_fingerprint(source_fit)
    assert result.artifact_feature_fingerprint == hmm_feature_matrix_fingerprint(
        source_features
    )
    assert len(result.source_evidence_fingerprint) == 64
    assert len(result.execution_fingerprint) == 64
    assert len(result.rng_execution_fingerprint) == 64
    assert result.model_n_states == 2
    assert result.observation_count == 12
    assert execution.is_registered_refitted_adequacy_batch(result)
    assert tuple(calls) == (0, 1, 2, 3, 4)
    retained_names = {field.name for field in fields(result)}
    assert "refit" not in retained_names
    assert "successful_attempts_full" not in retained_names

    for attempt, audit in enumerate(result.attempt_audits):
        expected_seeds = deterministic_hmm_restart_seeds(
            master_seed=123,
            scientific_version=1,
            scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            replicate=attempt,
            n_states=2,
        )
        assert calls[attempt] == (
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            expected_seeds,
        )
        assert audit.attempt == attempt
        assert audit.artifact is CalibrationArtifact.DESIGN_OR_CONTROL
        assert audit.scope is ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY
        assert audit.state_stream_stage is Stage.STATE_CHAIN
        assert audit.emission_stream_stage is Stage.OPTIONAL_RESIDUAL_NOISE
        expected_entity = calibration_entity(
            CalibrationPurpose.PARAMETRIC_ADEQUACY,
            CalibrationArtifact.DESIGN_OR_CONTROL,
            0,
            attempt,
        )
        assert audit.state_stream_entity == expected_entity
        assert audit.emission_stream_entity == expected_entity
        expected_seed_hash = hashlib.sha256(
            np.asarray(expected_seeds, dtype="<u4").tobytes()
        ).hexdigest()
        assert audit.restart_seed_count == 50
        assert audit.restart_seed_sha256 == expected_seed_hash
    failure = result.attempt_audits[2]
    assert not failure.successful
    assert failure.endpoint_row is None
    assert failure.failure is not None
    assert len(failure.failure.message) == 500
    assert len(failure.failure.details) == 8


def test_registered_batch_authority_is_object_specific_and_content_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_attempt_stub(
        monkeypatch,
        master_seed=123,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)
    result = _run(attempts=4, minimum=3)
    identity = execution.registered_refitted_adequacy_batch_identity(result)
    assert identity.execution_fingerprint == result.execution_fingerprint
    assert identity.source_evidence_fingerprint == result.source_evidence_fingerprint

    copied = copy.copy(result)
    replaced = replace(result)
    object.__setattr__(
        copied, "_execution_seal", execution._REGISTERED_REFIT_CAPABILITY
    )
    object.__setattr__(
        replaced,
        "_execution_seal",
        execution._REGISTERED_REFIT_CAPABILITY,
    )
    assert not execution.is_registered_refitted_adequacy_batch(copied)
    assert not execution.is_registered_refitted_adequacy_batch(replaced)

    forged = replace(result)
    object.__setattr__(forged, "successful_attempts", 3)
    object.__setattr__(
        forged,
        "_execution_seal",
        execution._REGISTERED_REFIT_CAPABILITY,
    )
    assert not execution.is_registered_refitted_adequacy_batch(forged)


def test_registered_batch_rejects_result_audit_and_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_attempt_stub(
        monkeypatch,
        master_seed=123,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)

    changed_result = _run(attempts=4, minimum=3)
    changed_matrix = changed_result.successful_endpoint_matrix.copy()
    changed_matrix[0, 0] += 0.25
    changed_matrix.setflags(write=False)
    object.__setattr__(
        changed_result,
        "successful_endpoint_matrix",
        changed_matrix,
    )
    assert not execution.is_registered_refitted_adequacy_batch(changed_result)

    changed_audit = _run(attempts=4, minimum=3)
    first = replace(changed_audit.attempt_audits[0], state_uniform_sha256="0" * 64)
    object.__setattr__(
        changed_audit,
        "attempt_audits",
        (first, *changed_audit.attempt_audits[1:]),
    )
    assert not execution.is_registered_refitted_adequacy_batch(changed_audit)

    changed_source = _run(attempts=4, minimum=3)
    source_fit, _source_features = execution.registered_refitted_adequacy_sources(
        changed_source
    )
    source_fit.parameters.transition_matrix[0, 0] -= 0.01
    assert not execution.is_registered_refitted_adequacy_batch(changed_source)


def test_stream_hashes_replay_and_attempt_batch_schedule_do_not_change_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_attempt_stub(
        monkeypatch,
        master_seed=123,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)
    serial = _run(attempts=6, minimum=3, attempt_batch_size=1)
    monkeypatch.setattr(execution, "deterministic_process_map", _reverse_schedule_map)
    reversed_schedule = _run(
        attempts=6,
        minimum=3,
        workers=4,
        attempt_batch_size=4,
    )

    np.testing.assert_array_equal(
        serial.successful_endpoint_matrix,
        reversed_schedule.successful_endpoint_matrix,
    )
    assert (
        serial.successful_attempt_indices
        == reversed_schedule.successful_attempt_indices
    )
    assert [item.state_uniform_sha256 for item in serial.attempt_audits] == [
        item.state_uniform_sha256 for item in reversed_schedule.attempt_audits
    ]
    assert [item.emission_normal_sha256 for item in serial.attempt_audits] == [
        item.emission_normal_sha256 for item in reversed_schedule.attempt_audits
    ]
    assert serial.max_statistic_cell is not None
    assert reversed_schedule.max_statistic_cell is not None
    assert (
        serial.max_statistic_cell.p_value
        == reversed_schedule.max_statistic_cell.p_value
    )
    assert (
        serial.rng_execution_fingerprint == reversed_schedule.rng_execution_fingerprint
    )

    state_key = calibration_rng_key(
        master_seed=123,
        scientific_version=1,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        variant=0,
        replicate=0,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.STATE_CHAIN,
    )
    emission_key = calibration_rng_key(
        master_seed=123,
        scientific_version=1,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        variant=0,
        replicate=0,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.OPTIONAL_RESIDUAL_NOISE,
    )
    state = state_key.generator().random(12)
    normals = hmm_emission_standard_normals(emission_key, 12)
    expected_state_hash = hashlib.sha256(
        np.asarray(state, dtype="<f8").tobytes()
    ).hexdigest()
    expected_normal_hash = hashlib.sha256(
        np.asarray(normals, dtype="<f8").tobytes()
    ).hexdigest()
    assert serial.attempt_audits[0].state_uniform_sha256 == expected_state_hash
    assert serial.attempt_audits[0].emission_normal_sha256 == expected_normal_hash


def test_failures_are_never_replaced_and_below_minimum_retains_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_attempt_stub(
        monkeypatch,
        master_seed=123,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
        failure_attempts={1, 3},
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)
    result = _run(attempts=4, minimum=3)

    assert result.attempts == 4
    assert result.successful_attempts == 2
    assert result.failed_attempts == 2
    assert result.successful_attempt_indices == (0, 2)
    assert [audit.attempt for audit in result.attempt_audits] == [0, 1, 2, 3]
    assert [audit.successful for audit in result.attempt_audits] == [
        True,
        False,
        True,
        False,
    ]
    assert not result.success_requirement_passed
    assert not result.ready_for_holm
    assert result.max_statistic_cell is None
    assert result.cell_failure is not None
    assert result.cell_failure.failure_type == "InsufficientSuccessfulRefits"
    assert result.successful_endpoint_matrix.shape == (2, 2)


def test_unexpected_programming_error_is_not_reclassified_as_refit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution, "build_refitted_adequacy_endpoint", _observed_endpoint
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected implementation defect")

    monkeypatch.setattr(execution, "run_refitted_adequacy_attempt", unexpected)
    with pytest.raises(RuntimeError, match="implementation defect"):
        _run(attempts=2, minimum=2)


def test_unexpected_generic_model_error_aborts_the_registered_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution, "build_refitted_adequacy_endpoint", _observed_endpoint
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise ModelError("unexpected generic model fault")

    monkeypatch.setattr(execution, "run_refitted_adequacy_attempt", unexpected)
    with pytest.raises(ModelError, match="unexpected generic model fault"):
        _run(attempts=2, minimum=2)


@pytest.mark.parametrize(
    ("failure_factory", "failure_type"),
    [
        (_no_valid_restarts, "NoNumericallyValidHMMRestarts"),
        (_candidate_inadmissibility, "HMMCandidateInadmissibility"),
        (
            lambda: RefittedAdequacyEndpointSupportFailure(
                "synthetic endpoint support non-pass"
            ),
            "RefittedAdequacyEndpointSupportFailure",
        ),
        (
            lambda: AdequacyEndpointSupportFailure(
                "synthetic shared endpoint support non-pass"
            ),
            "AdequacyEndpointSupportFailure",
        ),
    ],
)
def test_explicit_scientific_non_pass_types_retain_fixed_attempts(
    monkeypatch: pytest.MonkeyPatch,
    failure_factory: Callable[[], ModelError],
    failure_type: str,
) -> None:
    monkeypatch.setattr(
        execution, "build_refitted_adequacy_endpoint", _observed_endpoint
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)

    def scientific_non_pass(*_args: object, **_kwargs: object) -> object:
        raise failure_factory()

    monkeypatch.setattr(
        execution,
        "run_refitted_adequacy_attempt",
        scientific_non_pass,
    )
    result = _run(attempts=2, minimum=2)

    assert result.attempts == 2
    assert result.successful_attempts == 0
    assert result.failed_attempts == 2
    assert tuple(audit.attempt for audit in result.attempt_audits) == (0, 1)
    assert all(
        audit.failure is not None and audit.failure.failure_type == failure_type
        for audit in result.attempt_audits
    )


def test_successful_outcome_endpoint_contract_drift_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution, "build_refitted_adequacy_endpoint", _observed_endpoint
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)

    def inconsistent(*_args: object, **_kwargs: object) -> object:
        endpoint = _observed_endpoint()
        return SimpleNamespace(
            endpoint=replace(
                endpoint,
                vector=EndpointVector(
                    ("metric/drifted", "metric/upper_tail"),
                    endpoint.vector.values,
                ),
            ),
            refit=SimpleNamespace(best_restart_index=0, best_seed=0),
            alignment=SimpleNamespace(
                candidate_to_reference=np.asarray([0, 1], dtype=np.int64),
                selected_total_cost=0.0,
            ),
        )

    monkeypatch.setattr(execution, "run_refitted_adequacy_attempt", inconsistent)
    with pytest.raises(ModelError, match="inconsistent endpoint contract"):
        _run(attempts=2, minimum=2)


def test_production_role_maps_to_exact_artifact_and_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_attempt_stub(
        monkeypatch,
        master_seed=123,
        scientific_version=1,
        scope=ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY,
    )
    monkeypatch.setattr(execution, "deterministic_process_map", _serial_map)
    result = _run(role="production", attempts=3, minimum=2)
    assert result.artifact is CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    assert result.scope is ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY
    assert all(
        scope is ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY
        for scope, _seeds in calls.values()
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"attempt_count": 3}, "strict canonical"),
        (
            {"attempt_count": 1_001, "strict_canonical": False},
            "registered attempt range",
        ),
        (
            {
                "attempt_count": 3,
                "minimum_successes": 1,
                "strict_canonical": False,
            },
            "minimum_successes",
        ),
        (
            {
                "attempt_count": 3,
                "minimum_successes": 4,
                "strict_canonical": False,
            },
            "minimum_successes",
        ),
    ],
)
def test_count_contracts_fail_before_execution(
    updates: dict[str, object], message: str
) -> None:
    fit, features = _inputs()
    kwargs: dict[str, object] = {
        "role": "design",
        "original_fit": fit,
        "artifact_features": features,
        "master_seed": 123,
        "scientific_version": 1,
        "attempt_count": 1_000,
        "minimum_successes": 950,
        "strict_canonical": True,
    }
    kwargs.update(updates)
    with pytest.raises(ModelError, match=message):
        run_registered_refitted_adequacy_cell(**kwargs)  # type: ignore[arg-type]


def test_role_and_scaler_mapping_fail_closed() -> None:
    fit, features = _inputs("production")
    with pytest.raises(ModelError, match="scaler scope"):
        run_registered_refitted_adequacy_cell(
            role="design",
            original_fit=fit,
            artifact_features=features,
            master_seed=123,
            scientific_version=1,
            attempt_count=3,
            minimum_successes=2,
            strict_canonical=False,
        )


def test_parallel_pickling_and_geometry_preflights_fail_closed() -> None:
    fit, features = _inputs()
    common = {
        "role": "design",
        "original_fit": fit,
        "artifact_features": features,
        "master_seed": 123,
        "scientific_version": 1,
    }
    with pytest.raises(ModelError, match="pickleable"):
        run_registered_refitted_adequacy_cell(
            **common,  # type: ignore[arg-type]
            attempt_count=3,
            minimum_successes=2,
            workers=2,
            strict_canonical=False,
            fit_function=lambda *_args: None,  # type: ignore[arg-type]
        )

    mismatched = HMMFeatureMatrix(
        values=features.values,
        lengths=(7, 5),
        feature_names=MACRO_FEATURE_NAMES,
        row_labels=features.row_labels,
        scaler_means=features.scaler_means,
        scaler_standard_deviations=features.scaler_standard_deviations,
        scaler_scope=features.scaler_scope,
    )
    with pytest.raises(ModelError, match="geometry is inconsistent"):
        run_registered_refitted_adequacy_cell(
            role="design",
            original_fit=fit,
            artifact_features=mismatched,
            master_seed=123,
            scientific_version=1,
            attempt_count=3,
            minimum_successes=2,
            strict_canonical=False,
        )

    with pytest.raises(ModelError, match="wrong segment geometry"):
        run_registered_refitted_adequacy_cell(
            **common,  # type: ignore[arg-type]
        )

    with pytest.raises(ModelError, match="does not accept an injected fitter"):
        run_registered_refitted_adequacy_cell(
            **common,  # type: ignore[arg-type]
            fit_function=_unused_fit,  # type: ignore[arg-type]
        )


def test_non_integer_execution_controls_fail_closed() -> None:
    fit, features = _inputs()
    with pytest.raises(ModelError, match="workers must be a positive integer"):
        run_registered_refitted_adequacy_cell(
            role="design",
            original_fit=fit,
            artifact_features=features,
            master_seed=123,
            scientific_version=1,
            attempt_count=3,
            minimum_successes=2,
            workers=True,  # type: ignore[arg-type]
            strict_canonical=False,
        )


def test_invalid_role_fails_closed() -> None:
    design_fit, design_features = _inputs()
    with pytest.raises(ModelError, match="design or production"):
        run_registered_refitted_adequacy_cell(
            role="other",  # type: ignore[arg-type]
            original_fit=design_fit,
            artifact_features=design_features,
            master_seed=123,
            scientific_version=1,
            attempt_count=3,
            minimum_successes=2,
            strict_canonical=False,
        )
