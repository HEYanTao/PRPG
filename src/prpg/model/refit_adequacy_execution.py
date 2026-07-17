"""Registered, deterministic execution of the two refitted-adequacy cells.

The pure one-attempt implementation lives in :mod:`prpg.model.refit_adequacy`.
This layer fixes the purpose-3 stream and model-fit namespaces, preserves the
attempt denominator, executes bounded attempt chunks through
``prpg.execution``, and retains only endpoint rows plus compact audit records.
It never retains the fitted attempt objects in the returned batch evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal, TypeAlias, TypeGuard

import numpy as np
from numpy.typing import NDArray

from prpg.errors import ModelError
from prpg.execution import deterministic_process_map
from prpg.model.adequacy import (
    AdequacyEndpointSupportFailure,
    EndpointDirection,
    MaxStatisticCell,
    mixed_max_statistic_cell,
)
from prpg.model.hmm import (
    ForwardBackwardResult,
    GaussianHMMFit,
    HMMCandidateInadmissibility,
    HMMFeatureMatrix,
    NoNumericallyValidHMMRestarts,
    decode_viterbi,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_restarts,
    forward_backward,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    rng_execution_fingerprint,
)
from prpg.model.refit_adequacy import (
    FitFunction,
    RefittedAdequacyEndpointSupportFailure,
    build_refitted_adequacy_endpoint,
    run_refitted_adequacy_attempt,
)
from prpg.model.regime_policy import owner_fixed_k4_policy_for_scientific_version
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    ModelFitScope,
    Stage,
    calibration_rng_key,
    hmm_emission_standard_normals,
)

FloatArray: TypeAlias = NDArray[np.float64]
ArtifactRole: TypeAlias = Literal["design", "production"]
ProcessMap: TypeAlias = Callable[..., tuple[object, ...]]

CANONICAL_REFIT_ATTEMPTS = 1_000
CANONICAL_MINIMUM_SUCCESSES = 950
CANONICAL_DESIGN_LENGTHS = (193,)
CANONICAL_PRODUCTION_LENGTHS = (210, 7)
DEFAULT_ATTEMPT_BATCH_SIZE = 5
DEFAULT_PROCESS_CHUNKSIZE = 1
FAILURE_MESSAGE_LIMIT = 500
FAILURE_DETAIL_LIMIT = 8
FAILURE_DETAIL_VALUE_LIMIT = 160
_REGISTERED_REFIT_CAPABILITY = object()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class BoundedFailure:
    """A compact failure record safe to retain for all fixed attempts."""

    failure_type: str
    message: str
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RefitAttemptAudit:
    """Compact replay evidence for one zero-based registered attempt."""

    attempt: int
    successful: bool
    endpoint_row: int | None
    artifact: CalibrationArtifact
    scope: ModelFitScope
    state_stream_entity: int
    state_stream_stage: Stage
    emission_stream_entity: int
    emission_stream_stage: Stage
    state_uniform_sha256: str
    emission_normal_sha256: str
    restart_seed_count: int
    restart_seed_sha256: str
    best_restart_index: int | None
    best_seed: int | None
    candidate_to_reference: tuple[int, ...] | None
    alignment_total_cost: float | None
    failure: BoundedFailure | None


@dataclass(frozen=True, slots=True)
class RegisteredRefittedAdequacyBatch:
    """Bounded evidence for one design or production refitted-adequacy cell."""

    role: ArtifactRole
    artifact: CalibrationArtifact
    scope: ModelFitScope
    master_seed: int
    scientific_version: int
    parent_model_fingerprint: str
    artifact_feature_fingerprint: str
    rng_execution_fingerprint: str
    model_n_states: int
    observation_count: int
    attempts: int
    minimum_successes: int
    successful_attempts: int
    failed_attempts: int
    success_requirement_passed: bool
    ready_for_holm: bool
    successful_attempt_indices: tuple[int, ...]
    endpoint_labels: tuple[str, ...]
    directions: tuple[EndpointDirection, ...]
    observed_endpoint_values: FloatArray
    successful_endpoint_matrix: FloatArray
    max_statistic_cell: MaxStatisticCell | None
    unadjusted_95_max_statistic_quantile: float | None
    cell_failure: BoundedFailure | None
    attempt_audits: tuple[RefitAttemptAudit, ...]
    state_streams: int
    emission_streams: int
    restart_seed_sets: int
    workers: int
    attempt_batch_size: int
    process_chunksize: int
    strict_canonical: bool
    source_evidence_fingerprint: str = field(default="", init=False)
    execution_fingerprint: str = field(default="", init=False)
    _source_original_fit: GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_artifact_features: HMMFeatureMatrix | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RegisteredRefittedAdequacyBatchIdentity:
    """Reverified source, output, and registered-stream execution identity."""

    role: ArtifactRole
    artifact: CalibrationArtifact
    scope: ModelFitScope
    master_seed: int
    scientific_version: int
    parent_model_fingerprint: str
    artifact_feature_fingerprint: str
    rng_execution_fingerprint: str
    source_evidence_fingerprint: str
    execution_fingerprint: str


_MINTED_REGISTERED_REFIT_BATCHES: dict[int, RegisteredRefittedAdequacyBatch] = {}


def registered_refitted_adequacy_batch_identity(
    value: RegisteredRefittedAdequacyBatch,
) -> RegisteredRefittedAdequacyBatchIdentity:
    """Revalidate one exact object minted by the registered executor."""

    if (
        not isinstance(value, RegisteredRefittedAdequacyBatch)
        or value._execution_seal is not _REGISTERED_REFIT_CAPABILITY
        or _MINTED_REGISTERED_REFIT_BATCHES.get(id(value)) is not value
    ):
        raise ModelError("registered refitted-adequacy batch lacks executor authority")
    source_fit, source_features = _registered_sources(value)
    _verify_source_contract(value.role, source_fit, source_features)
    parent_fingerprint = gaussian_hmm_fit_fingerprint(source_fit)
    feature_fingerprint = hmm_feature_matrix_fingerprint(source_features)
    expected_rng = _refit_rng_execution_fingerprint(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        artifact=value.artifact,
        scope=value.scope,
        attempts=value.attempts,
        minimum_successes=value.minimum_successes,
        n_states=value.model_n_states,
        row_count=value.observation_count,
    )
    _verify_batch_result(value)
    source_evidence_fingerprint = _refit_source_evidence_fingerprint(value)
    execution_fingerprint = _refit_execution_fingerprint(
        value,
        parent_model_fingerprint=parent_fingerprint,
        artifact_feature_fingerprint=feature_fingerprint,
        rng_fingerprint=expected_rng,
        source_evidence_fingerprint=source_evidence_fingerprint,
    )
    if (
        value.parent_model_fingerprint != parent_fingerprint
        or value.artifact_feature_fingerprint != feature_fingerprint
        or value.rng_execution_fingerprint != expected_rng
        or value.source_evidence_fingerprint != source_evidence_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("registered refitted-adequacy batch identity changed")
    return RegisteredRefittedAdequacyBatchIdentity(
        value.role,
        value.artifact,
        value.scope,
        value.master_seed,
        value.scientific_version,
        parent_fingerprint,
        feature_fingerprint,
        expected_rng,
        source_evidence_fingerprint,
        execution_fingerprint,
    )


def registered_refitted_adequacy_sources(
    value: RegisteredRefittedAdequacyBatch,
) -> tuple[GaussianHMMFit, HMMFeatureMatrix]:
    """Return exact retained typed sources after full executor reverification."""

    registered_refitted_adequacy_batch_identity(value)
    return _registered_sources(value)


def is_registered_refitted_adequacy_batch(
    value: object,
) -> TypeGuard[RegisteredRefittedAdequacyBatch]:
    """Return whether this unchanged object came from the exact executor."""

    try:
        registered_refitted_adequacy_batch_identity(value)  # type: ignore[arg-type]
    except (
        AttributeError,
        ModelError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class _AttemptChunkJob:
    attempts: tuple[int, ...]
    master_seed: int
    scientific_version: int
    artifact: CalibrationArtifact
    scope: ModelFitScope
    original_fit: GaussianHMMFit
    artifact_features: HMMFeatureMatrix
    fit_function: FitFunction


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    audit: RefitAttemptAudit
    labels: tuple[str, ...] | None
    directions: tuple[EndpointDirection, ...] | None
    values: FloatArray | None


def run_registered_refitted_adequacy_cell(
    *,
    role: ArtifactRole,
    original_fit: GaussianHMMFit,
    artifact_features: HMMFeatureMatrix,
    master_seed: int,
    scientific_version: int,
    attempt_count: int = CANONICAL_REFIT_ATTEMPTS,
    minimum_successes: int = CANONICAL_MINIMUM_SUCCESSES,
    workers: int = 1,
    attempt_batch_size: int = DEFAULT_ATTEMPT_BATCH_SIZE,
    process_chunksize: int = DEFAULT_PROCESS_CHUNKSIZE,
    strict_canonical: bool = True,
    fit_function: FitFunction = fit_gaussian_hmm_restarts,
) -> RegisteredRefittedAdequacyBatch:
    """Run one fixed-denominator refitted-adequacy cell.

    Canonical execution is exactly 1,000 zero-based attempts with at least 950
    successes.  Small explicit counts are available only with
    ``strict_canonical=False`` for tests and dry-run integration.  Every
    attempt owns purpose-3 stage-2 and stage-5 keys and the exact 50 model-fit
    seeds for its artifact-specific scope.  Failed attempts are never replaced.
    """

    if not isinstance(original_fit, GaussianHMMFit):
        raise ModelError("refitted adequacy requires a typed GaussianHMMFit source")
    if not isinstance(artifact_features, HMMFeatureMatrix):
        raise ModelError("refitted adequacy requires a typed HMMFeatureMatrix source")
    attempts = _positive_integer(attempt_count, "attempt_count")
    required = _positive_integer(minimum_successes, "minimum_successes")
    worker_count = _positive_integer(workers, "workers")
    batch_size = _positive_integer(attempt_batch_size, "attempt_batch_size")
    map_chunksize = _positive_integer(process_chunksize, "process_chunksize")
    if attempts > CANONICAL_REFIT_ATTEMPTS:
        raise ModelError("refitted adequacy exceeds the registered attempt range")
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if strict_canonical and fit_function is not fit_gaussian_hmm_restarts:
        raise ModelError(
            "strict canonical refitted adequacy does not accept an injected fitter"
        )
    if strict_canonical and (
        attempts != CANONICAL_REFIT_ATTEMPTS or required != CANONICAL_MINIMUM_SUCCESSES
    ):
        raise ModelError(
            "strict canonical refitted adequacy requires exactly "
            "1,000 attempts and 950 successes"
        )
    registered_fit_function = fit_function
    if fit_function is fit_gaussian_hmm_restarts:
        policy = owner_fixed_k4_policy_for_scientific_version(scientific_version)
        if policy is not None:
            registered_fit_function = partial(
                fit_gaussian_hmm_restarts,
                owner_fixed_k4_policy=policy,
            )
    if required < 2 or required > attempts:
        raise ModelError("minimum_successes must lie in [2, attempt_count]")
    if worker_count > 1:
        try:
            pickle.dumps(registered_fit_function)
        except (pickle.PickleError, TypeError, AttributeError) as error:
            raise ModelError(
                "parallel refitted-adequacy fit_function must be pickleable"
            ) from error
    artifact, scope, canonical_lengths, _scaler_scope = _role_contract(role)
    _verify_source_contract(role, original_fit, artifact_features)
    if strict_canonical and original_fit.lengths != canonical_lengths:
        raise ModelError(
            "strict canonical refitted adequacy has the wrong segment geometry"
        )

    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(original_fit)
    artifact_feature_fingerprint = hmm_feature_matrix_fingerprint(artifact_features)
    observed = build_refitted_adequacy_endpoint(
        artifact_features.values,
        original_fit.parameters,
        original_fit.lengths,
        feature_names=artifact_features.feature_names,
        owner_fixed_k4_disposition=original_fit.owner_fixed_k4_disposition,
    )
    _preflight_registered_attempt(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        scope=scope,
        n_states=original_fit.n_states,
        row_count=original_fit.n_observations,
    )
    jobs = tuple(
        _AttemptChunkJob(
            attempts=tuple(range(start, min(start + batch_size, attempts))),
            master_seed=master_seed,
            scientific_version=scientific_version,
            artifact=artifact,
            scope=scope,
            original_fit=original_fit,
            artifact_features=artifact_features,
            fit_function=registered_fit_function,
        )
        for start in range(0, attempts, batch_size)
    )
    chunk_results = deterministic_process_map(
        _run_attempt_chunk,
        jobs,
        workers=worker_count,
        chunksize=map_chunksize,
    )
    outcomes = tuple(item for chunk in chunk_results for item in chunk)
    if tuple(outcome.audit.attempt for outcome in outcomes) != tuple(range(attempts)):
        raise ModelError("refitted-adequacy process results are not in attempt order")

    labels = observed.vector.labels
    directions = observed.directions
    successful_indices: list[int] = []
    rows: list[FloatArray] = []
    audits: list[RefitAttemptAudit] = []
    for outcome in outcomes:
        if not outcome.audit.successful:
            audits.append(outcome.audit)
            continue
        if (
            outcome.labels != labels
            or outcome.directions != directions
            or outcome.values is None
            or outcome.values.shape != observed.vector.values.shape
            or not np.isfinite(outcome.values).all()
        ):
            raise ModelError(
                "successful attempt returned an inconsistent endpoint contract",
                details={
                    "attempt": outcome.audit.attempt,
                    "labels_match": outcome.labels == labels,
                    "directions_match": outcome.directions == directions,
                    "values_present": outcome.values is not None,
                },
            )
        endpoint_row = len(rows)
        successful_indices.append(outcome.audit.attempt)
        rows.append(np.asarray(outcome.values, dtype=np.float64).copy())
        audits.append(replace(outcome.audit, endpoint_row=endpoint_row))

    if rows:
        matrix = np.vstack(rows).astype(np.float64, copy=False)
    else:
        matrix = np.empty((0, len(labels)), dtype=np.float64)
    success_count = len(rows)
    success_passed = success_count >= required
    cell: MaxStatisticCell | None = None
    cell_failure: BoundedFailure | None = None
    quantile: float | None = None
    if success_passed:
        try:
            cell = mixed_max_statistic_cell(
                observed.vector.values,
                matrix,
                directions,
                minimum_successes=required,
            )
            quantile = float(
                np.quantile(
                    cell.replicate_statistics,
                    0.95,
                    method="higher",
                )
            )
        except ModelError as error:
            cell_failure = _bounded_failure(error)
    else:
        cell_failure = BoundedFailure(
            "InsufficientSuccessfulRefits",
            "fixed refitted-adequacy attempts did not meet the success minimum",
            (
                ("actual", str(success_count)),
                ("required", str(required)),
            ),
        )

    observed_values = observed.vector.values.astype(np.float64, copy=True)
    matrix = matrix.astype(np.float64, copy=True)
    _readonly(observed_values, matrix)
    rng_fingerprint = _refit_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        scope=scope,
        attempts=attempts,
        minimum_successes=required,
        n_states=original_fit.n_states,
        row_count=original_fit.n_observations,
    )
    result = RegisteredRefittedAdequacyBatch(
        role=role,
        artifact=artifact,
        scope=scope,
        master_seed=master_seed,
        scientific_version=scientific_version,
        parent_model_fingerprint=parent_model_fingerprint,
        artifact_feature_fingerprint=artifact_feature_fingerprint,
        rng_execution_fingerprint=rng_fingerprint,
        model_n_states=original_fit.n_states,
        observation_count=original_fit.n_observations,
        attempts=attempts,
        minimum_successes=required,
        successful_attempts=success_count,
        failed_attempts=attempts - success_count,
        success_requirement_passed=success_passed,
        ready_for_holm=cell is not None,
        successful_attempt_indices=tuple(successful_indices),
        endpoint_labels=labels,
        directions=directions,
        observed_endpoint_values=observed_values,
        successful_endpoint_matrix=matrix,
        max_statistic_cell=cell,
        unadjusted_95_max_statistic_quantile=quantile,
        cell_failure=cell_failure,
        attempt_audits=tuple(audits),
        state_streams=attempts,
        emission_streams=attempts,
        restart_seed_sets=attempts,
        workers=worker_count,
        attempt_batch_size=batch_size,
        process_chunksize=map_chunksize,
        strict_canonical=strict_canonical,
    )
    object.__setattr__(result, "_source_original_fit", original_fit)
    object.__setattr__(result, "_source_artifact_features", artifact_features)
    if (
        gaussian_hmm_fit_fingerprint(original_fit) != parent_model_fingerprint
        or hmm_feature_matrix_fingerprint(artifact_features)
        != artifact_feature_fingerprint
    ):
        raise ModelError("refitted-adequacy source changed during execution")
    source_evidence_fingerprint = _refit_source_evidence_fingerprint(result)
    execution_fingerprint = _refit_execution_fingerprint(
        result,
        parent_model_fingerprint=parent_model_fingerprint,
        artifact_feature_fingerprint=artifact_feature_fingerprint,
        rng_fingerprint=rng_fingerprint,
        source_evidence_fingerprint=source_evidence_fingerprint,
    )
    object.__setattr__(
        result,
        "source_evidence_fingerprint",
        source_evidence_fingerprint,
    )
    object.__setattr__(result, "execution_fingerprint", execution_fingerprint)
    _verify_batch_result(result)
    object.__setattr__(result, "_execution_seal", _REGISTERED_REFIT_CAPABILITY)
    _MINTED_REGISTERED_REFIT_BATCHES[id(result)] = result
    return result


def _registered_sources(
    value: RegisteredRefittedAdequacyBatch,
) -> tuple[GaussianHMMFit, HMMFeatureMatrix]:
    source_fit = value._source_original_fit
    source_features = value._source_artifact_features
    if not isinstance(source_fit, GaussianHMMFit) or not isinstance(
        source_features, HMMFeatureMatrix
    ):
        raise ModelError("registered refitted-adequacy sources are unavailable")
    return source_fit, source_features


def _verify_source_contract(
    role: ArtifactRole,
    original_fit: GaussianHMMFit,
    artifact_features: HMMFeatureMatrix,
) -> None:
    _artifact, _scope, _canonical_lengths, scaler_scope = _role_contract(role)
    if original_fit.scaler_scope != scaler_scope:
        raise ModelError("refitted-adequacy role does not match artifact scaler scope")
    if artifact_features.scaler_scope != scaler_scope:
        raise ModelError("refitted-adequacy feature scaler scope is inconsistent")
    values = np.asarray(artifact_features.values)
    if values.ndim != 2 or values.shape != (
        original_fit.n_observations,
        original_fit.n_features,
    ):
        raise ModelError("refitted-adequacy artifact geometry is inconsistent")
    if (
        original_fit.lengths != artifact_features.lengths
        or original_fit.feature_names != artifact_features.feature_names
        or len(artifact_features.row_labels) != original_fit.n_observations
        or original_fit.parameters.n_states != original_fit.n_states
        or original_fit.parameters.n_features != original_fit.n_features
    ):
        raise ModelError("refitted-adequacy artifact geometry is inconsistent")
    if not np.array_equal(
        original_fit.scaler_means,
        artifact_features.scaler_means,
    ) or not np.array_equal(
        original_fit.scaler_standard_deviations,
        artifact_features.scaler_standard_deviations,
    ):
        raise ModelError("refitted-adequacy fit and feature scalers differ")
    decoded = decode_viterbi(
        artifact_features.values,
        original_fit.parameters,
        lengths=artifact_features.lengths,
    )
    posterior = forward_backward(
        artifact_features.values,
        original_fit.parameters,
        lengths=artifact_features.lengths,
    )
    retained = original_fit.forward_backward
    if not isinstance(retained, ForwardBackwardResult) or (
        not np.array_equal(decoded, original_fit.decoded_states)
        or retained.lengths != posterior.lengths
        or retained.sequence_log_likelihoods != posterior.sequence_log_likelihoods
        or retained.total_log_likelihood != posterior.total_log_likelihood
        or not np.array_equal(
            retained.filtered_probabilities,
            posterior.filtered_probabilities,
        )
        or not np.array_equal(
            retained.posterior_probabilities,
            posterior.posterior_probabilities,
        )
        or not np.array_equal(
            retained.expected_transition_counts,
            posterior.expected_transition_counts,
        )
    ):
        raise ModelError(
            "refitted-adequacy features do not reproduce the retained parent fit"
        )
    # These public fingerprint functions also enforce finite values and the
    # complete nested source geometry.  Calling them here makes malformed
    # typed fixtures report a model-boundary error before any parallel work.
    gaussian_hmm_fit_fingerprint(original_fit)
    hmm_feature_matrix_fingerprint(artifact_features)


def _verify_batch_result(value: RegisteredRefittedAdequacyBatch) -> None:
    source_fit, _source_features = _registered_sources(value)
    artifact, scope, canonical_lengths, _scaler_scope = _role_contract(value.role)
    if value.artifact is not artifact or value.scope is not scope:
        raise ModelError("registered refitted-adequacy role mapping changed")
    for control, label in (
        (value.attempts, "attempts"),
        (value.minimum_successes, "minimum_successes"),
        (value.workers, "workers"),
        (value.attempt_batch_size, "attempt_batch_size"),
        (value.process_chunksize, "process_chunksize"),
    ):
        _positive_integer(control, label)
    if value.attempts > CANONICAL_REFIT_ATTEMPTS:
        raise ModelError("registered refitted-adequacy attempt range changed")
    if not 2 <= value.minimum_successes <= value.attempts:
        raise ModelError("registered refitted-adequacy success threshold changed")
    if not isinstance(value.strict_canonical, bool):
        raise ModelError("registered refitted-adequacy strict mode is invalid")
    if value.strict_canonical and (
        value.attempts != CANONICAL_REFIT_ATTEMPTS
        or value.minimum_successes != CANONICAL_MINIMUM_SUCCESSES
        or source_fit.lengths != canonical_lengths
    ):
        raise ModelError("registered refitted-adequacy canonical contract changed")
    if (
        value.model_n_states != source_fit.n_states
        or value.observation_count != source_fit.n_observations
        or value.state_streams != value.attempts
        or value.emission_streams != value.attempts
        or value.restart_seed_sets != value.attempts
    ):
        raise ModelError("registered refitted-adequacy source/count metadata changed")

    if (
        not value.endpoint_labels
        or len(set(value.endpoint_labels)) != len(value.endpoint_labels)
        or not all(
            isinstance(label, str) and bool(label) for label in value.endpoint_labels
        )
        or len(value.directions) != len(value.endpoint_labels)
        or not all(isinstance(item, EndpointDirection) for item in value.directions)
        or value.observed_endpoint_values.shape != (len(value.endpoint_labels),)
        or not bool(np.isfinite(value.observed_endpoint_values).all())
    ):
        raise ModelError("registered refitted-adequacy observed endpoint changed")
    if value.observed_endpoint_values.flags.writeable:
        raise ModelError("registered refitted-adequacy observed endpoint is writable")
    matrix = np.asarray(value.successful_endpoint_matrix)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(value.endpoint_labels)
        or not bool(np.isfinite(matrix).all())
        or matrix.flags.writeable
    ):
        raise ModelError("registered refitted-adequacy endpoint matrix changed")
    if len(value.attempt_audits) != value.attempts:
        raise ModelError("registered refitted-adequacy audit count changed")

    successful_indices: list[int] = []
    endpoint_rows: list[int] = []
    for attempt, audit in enumerate(value.attempt_audits):
        _verify_attempt_audit(value, audit, attempt)
        if audit.successful:
            successful_indices.append(attempt)
            if audit.endpoint_row is None:
                raise ModelError("successful refit audit lacks an endpoint row")
            endpoint_rows.append(audit.endpoint_row)
    success_count = len(successful_indices)
    if (
        value.successful_attempt_indices != tuple(successful_indices)
        or endpoint_rows != list(range(success_count))
        or matrix.shape[0] != success_count
        or value.successful_attempts != success_count
        or value.failed_attempts != value.attempts - success_count
    ):
        raise ModelError("registered refitted-adequacy success accounting changed")
    success_passed = success_count >= value.minimum_successes
    if value.success_requirement_passed is not success_passed:
        raise ModelError("registered refitted-adequacy success decision changed")

    expected_cell: MaxStatisticCell | None = None
    expected_failure: BoundedFailure | None = None
    expected_quantile: float | None = None
    if success_passed:
        try:
            expected_cell = mixed_max_statistic_cell(
                value.observed_endpoint_values,
                value.successful_endpoint_matrix,
                value.directions,
                minimum_successes=value.minimum_successes,
            )
            expected_quantile = float(
                np.quantile(
                    expected_cell.replicate_statistics,
                    0.95,
                    method="higher",
                )
            )
        except ModelError as error:
            expected_failure = _bounded_failure(error)
    else:
        expected_failure = BoundedFailure(
            "InsufficientSuccessfulRefits",
            "fixed refitted-adequacy attempts did not meet the success minimum",
            (
                ("actual", str(success_count)),
                ("required", str(value.minimum_successes)),
            ),
        )
    if (
        value.ready_for_holm is not (expected_cell is not None)
        or not _same_max_statistic_cell(value.max_statistic_cell, expected_cell)
        or value.unadjusted_95_max_statistic_quantile != expected_quantile
        or value.cell_failure != expected_failure
    ):
        raise ModelError("registered refitted-adequacy cell summary changed")


def _verify_attempt_audit(
    batch: RegisteredRefittedAdequacyBatch,
    audit: RefitAttemptAudit,
    attempt: int,
) -> None:
    if not isinstance(audit, RefitAttemptAudit) or audit.attempt != attempt:
        raise ModelError("registered refitted-adequacy audit order changed")
    state_key = calibration_rng_key(
        master_seed=batch.master_seed,
        scientific_version=batch.scientific_version,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=batch.artifact,
        variant=0,
        replicate=attempt,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.STATE_CHAIN,
    )
    emission_key = calibration_rng_key(
        master_seed=batch.master_seed,
        scientific_version=batch.scientific_version,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=batch.artifact,
        variant=0,
        replicate=attempt,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.OPTIONAL_RESIDUAL_NOISE,
    )
    state_uniforms = state_key.generator().random(batch.observation_count)
    normals = hmm_emission_standard_normals(emission_key, batch.observation_count)
    seeds = deterministic_hmm_restart_seeds(
        master_seed=batch.master_seed,
        scientific_version=batch.scientific_version,
        scope=batch.scope,
        replicate=attempt,
        n_states=batch.model_n_states,
    )
    if (
        audit.artifact is not batch.artifact
        or audit.scope is not batch.scope
        or audit.state_stream_entity != state_key.entity
        or audit.state_stream_stage is not Stage.STATE_CHAIN
        or audit.emission_stream_entity != emission_key.entity
        or audit.emission_stream_stage is not Stage.OPTIONAL_RESIDUAL_NOISE
        or audit.state_uniform_sha256 != _array_sha256(state_uniforms)
        or audit.emission_normal_sha256 != _array_sha256(normals)
        or audit.restart_seed_count != len(seeds)
        or len(seeds) != 50
        or audit.restart_seed_sha256 != _seed_sha256(seeds)
    ):
        raise ModelError("registered refitted-adequacy audit RNG lineage changed")
    if audit.successful:
        restart = audit.best_restart_index
        if (
            audit.failure is not None
            or audit.endpoint_row is None
            or isinstance(restart, bool)
            or not isinstance(restart, int)
            or not 0 <= restart < len(seeds)
            or audit.best_seed != seeds[restart]
            or audit.candidate_to_reference is None
            or tuple(sorted(audit.candidate_to_reference))
            != tuple(range(batch.model_n_states))
            or audit.alignment_total_cost is None
            or not math.isfinite(audit.alignment_total_cost)
            or audit.alignment_total_cost < 0.0
        ):
            raise ModelError("registered successful-refit audit changed")
    elif (
        audit.endpoint_row is not None
        or audit.best_restart_index is not None
        or audit.best_seed is not None
        or audit.candidate_to_reference is not None
        or audit.alignment_total_cost is not None
        or not isinstance(audit.failure, BoundedFailure)
    ):
        raise ModelError("registered failed-refit audit changed")
    _verify_bounded_failure(audit.failure)


def _same_max_statistic_cell(
    left: MaxStatisticCell | None,
    right: MaxStatisticCell | None,
) -> bool:
    if left is None or right is None:
        return left is right
    arrays = (
        (left.replicate_statistics, right.replicate_statistics),
        (left.center, right.center),
        (left.scale, right.scale),
        (left.active_components, right.active_components),
    )
    return bool(
        isinstance(left, MaxStatisticCell)
        and left.observed_statistic == right.observed_statistic
        and left.p_value == right.p_value
        and left.successful_replicates == right.successful_replicates
        and all(np.array_equal(actual, expected) for actual, expected in arrays)
        and all(not actual.flags.writeable for actual, _expected in arrays)
    )


def _verify_bounded_failure(value: BoundedFailure | None) -> None:
    if value is None:
        return
    if (
        not isinstance(value, BoundedFailure)
        or not isinstance(value.failure_type, str)
        or not value.failure_type
        or not isinstance(value.message, str)
        or not value.message
        or len(value.message) > FAILURE_MESSAGE_LIMIT
        or len(value.details) > FAILURE_DETAIL_LIMIT
    ):
        raise ModelError("registered refitted-adequacy failure record changed")
    for item in value.details:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
            or any(len(part) > FAILURE_DETAIL_VALUE_LIMIT for part in item)
        ):
            raise ModelError("registered refitted-adequacy failure detail changed")


def _refit_source_evidence_fingerprint(
    value: RegisteredRefittedAdequacyBatch,
) -> str:
    max_cell = value.max_statistic_cell
    return _json_fingerprint(
        {
            "schema_id": "prpg-registered-refit-source-evidence-v2",
            "role": value.role,
            "artifact": int(value.artifact),
            "scope": int(value.scope),
            "master_seed": value.master_seed,
            "scientific_version": value.scientific_version,
            "parent_model_fingerprint": value.parent_model_fingerprint,
            "artifact_feature_fingerprint": value.artifact_feature_fingerprint,
            "rng_execution_fingerprint": value.rng_execution_fingerprint,
            "model_n_states": value.model_n_states,
            "observation_count": value.observation_count,
            "attempts": value.attempts,
            "minimum_successes": value.minimum_successes,
            "successful_attempts": value.successful_attempts,
            "failed_attempts": value.failed_attempts,
            "success_requirement_passed": value.success_requirement_passed,
            "ready_for_holm": value.ready_for_holm,
            "successful_attempt_indices": list(value.successful_attempt_indices),
            "endpoint_labels": list(value.endpoint_labels),
            "directions": [direction.value for direction in value.directions],
            "observed_endpoint_values": _array_identity(
                value.observed_endpoint_values,
                "observed_endpoint_values",
            ),
            "successful_endpoint_matrix": _array_identity(
                value.successful_endpoint_matrix,
                "successful_endpoint_matrix",
            ),
            "max_statistic_cell": None
            if max_cell is None
            else {
                "observed_statistic": max_cell.observed_statistic,
                "replicate_statistics": _array_identity(
                    max_cell.replicate_statistics,
                    "max_replicate_statistics",
                ),
                "p_value": max_cell.p_value,
                "center": _array_identity(max_cell.center, "max_center"),
                "scale": _array_identity(max_cell.scale, "max_scale"),
                "active_components": _array_identity(
                    max_cell.active_components,
                    "max_active_components",
                ),
                "successful_replicates": max_cell.successful_replicates,
            },
            "unadjusted_95_max_statistic_quantile": (
                value.unadjusted_95_max_statistic_quantile
            ),
            "cell_failure": _failure_payload(value.cell_failure),
            "attempt_audits": [_audit_payload(audit) for audit in value.attempt_audits],
            "state_streams": value.state_streams,
            "emission_streams": value.emission_streams,
            "restart_seed_sets": value.restart_seed_sets,
            "workers": value.workers,
            "attempt_batch_size": value.attempt_batch_size,
            "process_chunksize": value.process_chunksize,
            "strict_canonical": value.strict_canonical,
        }
    )


def _refit_execution_fingerprint(
    value: RegisteredRefittedAdequacyBatch,
    *,
    parent_model_fingerprint: str,
    artifact_feature_fingerprint: str,
    rng_fingerprint: str,
    source_evidence_fingerprint: str,
) -> str:
    return _json_fingerprint(
        {
            "schema_id": "prpg-registered-refit-execution-v2",
            "role": value.role,
            "artifact": int(value.artifact),
            "scope": int(value.scope),
            "master_seed": value.master_seed,
            "scientific_version": value.scientific_version,
            "parent_model_fingerprint": parent_model_fingerprint,
            "artifact_feature_fingerprint": artifact_feature_fingerprint,
            "rng_execution_fingerprint": rng_fingerprint,
            "source_evidence_fingerprint": source_evidence_fingerprint,
        }
    )


def _audit_payload(audit: RefitAttemptAudit) -> dict[str, object]:
    return {
        "attempt": audit.attempt,
        "successful": audit.successful,
        "endpoint_row": audit.endpoint_row,
        "artifact": int(audit.artifact),
        "scope": int(audit.scope),
        "state_stream_entity": audit.state_stream_entity,
        "state_stream_stage": int(audit.state_stream_stage),
        "emission_stream_entity": audit.emission_stream_entity,
        "emission_stream_stage": int(audit.emission_stream_stage),
        "state_uniform_sha256": audit.state_uniform_sha256,
        "emission_normal_sha256": audit.emission_normal_sha256,
        "restart_seed_count": audit.restart_seed_count,
        "restart_seed_sha256": audit.restart_seed_sha256,
        "best_restart_index": audit.best_restart_index,
        "best_seed": audit.best_seed,
        "candidate_to_reference": None
        if audit.candidate_to_reference is None
        else list(audit.candidate_to_reference),
        "alignment_total_cost": audit.alignment_total_cost,
        "failure": _failure_payload(audit.failure),
    }


def _failure_payload(value: BoundedFailure | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "failure_type": value.failure_type,
        "message": value.message,
        "details": [list(item) for item in value.details],
    }


def _array_identity(values: Any, role: str) -> str:
    source = np.asarray(values)
    if source.dtype.fields is not None:
        raise ModelError(f"{role} cannot use a structured dtype")
    kind = source.dtype.kind
    if kind == "f":
        if not bool(np.isfinite(source).all()):
            raise ModelError(f"{role} contains non-finite values")
        target: np.dtype[Any] = np.dtype("<f8")
    elif kind == "i":
        target = np.dtype("<i8")
    elif kind == "u":
        target = np.dtype("<u8")
    elif kind == "b":
        target = np.dtype("?")
    else:
        raise ModelError(f"{role} has an unsupported dtype")
    canonical = np.ascontiguousarray(source.astype(target, copy=False))
    header = _canonical_json_bytes(
        {
            "schema_id": "prpg-registered-refit-array-v1",
            "role": role,
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
            "nbytes": canonical.nbytes,
        }
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    content = canonical.tobytes(order="C")
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
    return digest.hexdigest()


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "registered refitted-adequacy identity is not canonical"
        ) from error


def _run_attempt_chunk(job: _AttemptChunkJob) -> tuple[_AttemptOutcome, ...]:
    return tuple(_run_one_attempt(job, attempt) for attempt in job.attempts)


def _run_one_attempt(job: _AttemptChunkJob, attempt: int) -> _AttemptOutcome:
    state_key = calibration_rng_key(
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=job.artifact,
        variant=0,
        replicate=attempt,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.STATE_CHAIN,
    )
    emission_key = calibration_rng_key(
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=job.artifact,
        variant=0,
        replicate=attempt,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.OPTIONAL_RESIDUAL_NOISE,
    )
    state_uniforms = state_key.generator().random(job.original_fit.n_observations)
    normals = hmm_emission_standard_normals(
        emission_key,
        job.original_fit.n_observations,
    )
    seeds = deterministic_hmm_restart_seeds(
        master_seed=job.master_seed,
        scientific_version=job.scientific_version,
        scope=job.scope,
        replicate=attempt,
        n_states=job.original_fit.n_states,
    )
    state_digest = _array_sha256(state_uniforms)
    emission_digest = _array_sha256(normals)
    seed_digest = _seed_sha256(seeds)
    try:
        result = run_refitted_adequacy_attempt(
            job.original_fit,
            job.artifact_features,
            state_uniforms,
            normals,
            scope=job.scope,
            seeds=seeds,
            fit_function=job.fit_function,
        )
        audit = RefitAttemptAudit(
            attempt=attempt,
            successful=True,
            endpoint_row=None,
            artifact=job.artifact,
            scope=job.scope,
            state_stream_entity=state_key.entity,
            state_stream_stage=Stage.STATE_CHAIN,
            emission_stream_entity=emission_key.entity,
            emission_stream_stage=Stage.OPTIONAL_RESIDUAL_NOISE,
            state_uniform_sha256=state_digest,
            emission_normal_sha256=emission_digest,
            restart_seed_count=len(seeds),
            restart_seed_sha256=seed_digest,
            best_restart_index=result.refit.best_restart_index,
            best_seed=result.refit.best_seed,
            candidate_to_reference=tuple(
                int(value) for value in result.alignment.candidate_to_reference
            ),
            alignment_total_cost=float(result.alignment.selected_total_cost),
            failure=None,
        )
        values = result.endpoint.vector.values.astype(np.float64, copy=True)
        _readonly(values)
        return _AttemptOutcome(
            audit,
            result.endpoint.vector.labels,
            result.endpoint.directions,
            values,
        )
    except (
        NoNumericallyValidHMMRestarts,
        HMMCandidateInadmissibility,
        AdequacyEndpointSupportFailure,
        RefittedAdequacyEndpointSupportFailure,
    ) as error:
        audit = RefitAttemptAudit(
            attempt=attempt,
            successful=False,
            endpoint_row=None,
            artifact=job.artifact,
            scope=job.scope,
            state_stream_entity=state_key.entity,
            state_stream_stage=Stage.STATE_CHAIN,
            emission_stream_entity=emission_key.entity,
            emission_stream_stage=Stage.OPTIONAL_RESIDUAL_NOISE,
            state_uniform_sha256=state_digest,
            emission_normal_sha256=emission_digest,
            restart_seed_count=len(seeds),
            restart_seed_sha256=seed_digest,
            best_restart_index=None,
            best_seed=None,
            candidate_to_reference=None,
            alignment_total_cost=None,
            failure=_bounded_failure(error),
        )
        return _AttemptOutcome(audit, None, None, None)


def _preflight_registered_attempt(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    scope: ModelFitScope,
    n_states: int,
    row_count: int,
) -> None:
    """Validate both closed streams and all 50 model seeds before parallelism."""

    for stage in (Stage.STATE_CHAIN, Stage.OPTIONAL_RESIDUAL_NOISE):
        calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
            artifact=artifact,
            variant=0,
            replicate=0,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=Family.NOT_APPLICABLE,
            stage=stage,
        )
    if row_count <= 0:
        raise ModelError("refitted-adequacy artifact row count must be positive")
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=scope,
        replicate=0,
        n_states=n_states,
    )
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise ModelError(
            "registered refitted-adequacy seeds are not 50 distinct values"
        )


def _refit_rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    scope: ModelFitScope,
    attempts: int,
    minimum_successes: int,
    n_states: int,
    row_count: int,
) -> str:
    """Bind all registered purpose-3 streams and restart namespaces."""

    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="refitted_adequacy",
        contract={
            "artifact": int(artifact),
            "attempts": attempts,
            "emission_stage": int(Stage.OPTIONAL_RESIDUAL_NOISE),
            "minimum_successes": minimum_successes,
            "model_n_states": n_states,
            "model_restart_count": 50,
            "model_scope": int(scope),
            "observation_count": row_count,
            "purpose": int(CalibrationPurpose.PARAMETRIC_ADEQUACY),
            "state_stage": int(Stage.STATE_CHAIN),
        },
    )


def _role_contract(
    role: ArtifactRole,
) -> tuple[
    CalibrationArtifact,
    ModelFitScope,
    tuple[int, ...],
    Literal["design_training", "production_full"],
]:
    if role == "design":
        return (
            CalibrationArtifact.DESIGN_OR_CONTROL,
            ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY,
            CANONICAL_DESIGN_LENGTHS,
            "design_training",
        )
    if role == "production":
        return (
            CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
            ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY,
            CANONICAL_PRODUCTION_LENGTHS,
            "production_full",
        )
    raise ModelError("refitted-adequacy role must be design or production")


def _bounded_failure(error: Exception) -> BoundedFailure:
    if isinstance(error, ModelError):
        message = error.message
        raw_details = error.details
    else:
        message = str(error)
        raw_details = {}
    bounded_message = " ".join(message.split())[:FAILURE_MESSAGE_LIMIT]
    if not bounded_message:
        bounded_message = "<no message>"
    details = tuple(
        (
            str(key)[:FAILURE_DETAIL_VALUE_LIMIT],
            _bounded_detail_value(value),
        )
        for key, value in sorted(raw_details.items(), key=lambda item: str(item[0]))[
            :FAILURE_DETAIL_LIMIT
        ]
    )
    return BoundedFailure(type(error).__name__, bounded_message, details)


def _bounded_detail_value(value: object) -> str:
    if isinstance(value, dict | list | tuple | set):
        return f"<{type(value).__name__} length={len(value)}>"
    return " ".join(repr(value).split())[:FAILURE_DETAIL_VALUE_LIMIT]


def _array_sha256(values: FloatArray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _seed_sha256(seeds: Sequence[int]) -> str:
    values = np.asarray(seeds, dtype="<u4")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
