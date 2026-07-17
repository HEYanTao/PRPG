"""Registered-RNG boundary for return-block calibration execution.

The pure block executor consumes caller-owned uniforms so it can be tested
without hidden randomness.  Canonical execution must additionally prove that
every paired history owns exactly the three purpose-4 streams registered in
Section 4.6: monthly target-state, target-frequency restart, and source-rank
uniforms.  This module creates those streams in bounded row chunks and passes
them to the pure executor without retaining the complete experiment in RAM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard, cast

import numpy as np

from prpg.errors import ModelError
from prpg.model.block_execution import (
    CANONICAL_HISTORY_COUNT,
    DEFAULT_TRACE_CHUNK_SIZE,
    Artifact,
    BlockCalibrationExecution,
    BlockCalibrationNumericalFailure,
    BlockCalibrationSupportFailure,
    BlockStateModel,
    BlockSupportFailureEvidence,
    CalibrationGeometry,
    PairedTraceChunk,
    PooledBlockStatistics,
    PooledFragmentationGate,
    RegisteredUniformChunk,
    RegisteredUniformMatrices,
    execute_block_calibration,
    iter_paired_trace_chunks,
    resolve_calibration_geometry,
    uniform_chunks_from_matrices,
)
from prpg.model.block_length import BlockPolicySelection, Frequency
from prpg.model.hmm import GaussianHMMFit
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    rng_execution_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)

_UNIFORM_MATERIALIZATION_CAPABILITY = object()
_BLOCK_CALIBRATION_CAPABILITY = object()
_BLOCK_SUPPORT_FAILURE_CAPABILITY = object()
_MINTED_UNIFORM_MATERIALIZATIONS: dict[int, RegisteredBlockUniformMaterialization] = {}
_MINTED_BLOCK_CALIBRATIONS: dict[int, RegisteredBlockCalibration] = {}
_MINTED_BLOCK_SUPPORT_FAILURES: dict[int, _BlockSupportFailureMintRecord] = {}


@dataclass(frozen=True, slots=True)
class RegisteredBlockCalibration:
    """Block decision plus exact purpose-4 stream-consumption accounting."""

    execution: BlockCalibrationExecution
    history_stream_sets: int
    state_uniforms_per_history: int
    restart_uniforms_per_history: int
    source_rank_uniforms_per_history: int
    uniform_chunk_size: int
    master_seed: int | None = None
    scientific_version: int | None = None
    rng_namespace: str | None = None
    rng_execution_fingerprint: str | None = None
    materialization_fingerprint: str | None = None
    parent_model_fingerprint: str | None = None
    source_history_fingerprint: str | None = None
    source_parent_fingerprint: str | None = None
    execution_fingerprint: str | None = None
    calibration_fingerprint: str | None = None
    _source_model: BlockStateModel | GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_states: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_continuation_links: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_materialization: RegisteredBlockUniformMaterialization | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True, init=False)
class RegisteredBlockSupportFailure:
    """Producer-sealed terminal per-K block support/numerical failure."""

    contract_version: str
    evidence: BlockSupportFailureEvidence
    master_seed: int
    scientific_version: int
    artifact: Artifact
    frequency: Frequency
    selected_k: int
    starting_length: int
    materialization_fingerprint: str
    parent_model_fingerprint: str
    source_history_fingerprint: str
    source_parent_fingerprint: str
    failure_fingerprint: str
    _source_model: BlockStateModel | GaussianHMMFit = field(
        repr=False,
        compare=False,
    )
    _source_states: np.ndarray = field(repr=False, compare=False)
    _source_continuation_links: np.ndarray = field(repr=False, compare=False)
    _source_materialization: RegisteredBlockUniformMaterialization = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("RegisteredBlockSupportFailure is created only by its executor")


@dataclass(frozen=True, slots=True)
class RegisteredBlockSupportFailureIdentity:
    """Reverified exact parent and failure evidence identity."""

    master_seed: int
    scientific_version: int
    artifact: Artifact
    frequency: Frequency
    selected_k: int
    starting_length: int
    failure_code: str
    materialization_fingerprint: str
    parent_model_fingerprint: str
    source_history_fingerprint: str
    source_parent_fingerprint: str
    failure_fingerprint: str


@dataclass(frozen=True, slots=True)
class _BlockSupportFailureMintRecord:
    value: RegisteredBlockSupportFailure
    evidence: BlockSupportFailureEvidence
    model: BlockStateModel | GaussianHMMFit
    states: np.ndarray
    links: np.ndarray
    materialization: RegisteredBlockUniformMaterialization
    failure_fingerprint: str


REGISTERED_BLOCK_SUPPORT_FAILURE_CONTRACT = "registered-block-support-failure-v1"


@dataclass(frozen=True, slots=True)
class RegisteredBlockUniformMaterialization:
    """One common purpose-4 draw set reusable across K, L, and diagnostics."""

    geometry: CalibrationGeometry
    matrices: RegisteredUniformMatrices
    history_count: int
    allocated_bytes: int
    master_seed: int | None = None
    scientific_version: int | None = None
    rng_namespace: str | None = None
    rng_execution_fingerprint: str | None = None
    stream_fingerprint: str | None = None
    materialization_fingerprint: str | None = None
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RegisteredBlockUniformIdentity:
    """Revalidated purpose-4 key, geometry, and immutable-byte identity."""

    master_seed: int
    scientific_version: int
    artifact: Artifact
    frequency: Frequency
    history_count: int
    rng_namespace: str
    rng_execution_fingerprint: str
    stream_fingerprint: str
    materialization_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredBlockCalibrationIdentity:
    """Revalidated block result and every exact scientific parent identity."""

    master_seed: int
    scientific_version: int
    artifact: Artifact
    frequency: Frequency
    materialization_fingerprint: str
    parent_model_fingerprint: str
    source_history_fingerprint: str
    source_parent_fingerprint: str
    execution_fingerprint: str
    calibration_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredBlockCoreIdentity:
    """Task-safe identity of block selection before fragmentation/dependence.

    The core fingerprint deliberately excludes ``selected_fragmentation_gates``
    and every dependence result.  It binds the exact producer-authorized
    calibration, model/history parents, purpose-4 draw set, RNG registration,
    role/frequency/K, and selected block length.
    """

    master_seed: int
    scientific_version: int
    artifact: Artifact
    frequency: Frequency
    selected_k: int
    histories: int
    starting_length: int
    selected_length: int
    rng_namespace: str
    rng_execution_fingerprint: str
    uniform_stream_fingerprint: str
    materialization_fingerprint: str
    parent_model_fingerprint: str
    source_history_fingerprint: str
    source_parent_fingerprint: str
    core_execution_fingerprint: str
    core_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredBlockFragmentationIdentity:
    """Exact selected-L fragmentation evidence kept outside block core."""

    artifact: Artifact
    frequency: Frequency
    selected_k: int
    selected_length: int
    parent_core_fingerprint: str
    fragmentation_evidence_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredBlockReplaySource:
    """Producer-reverified references needed for later purpose-4 replay.

    This object is not an authority token.  Callers receive it only after the
    registered calibration and its live sources have been reverified, and must
    reverify the parent calibration again after replay.
    """

    core_identity: RegisteredBlockCoreIdentity
    uniform_identity: RegisteredBlockUniformIdentity
    materialization: RegisteredBlockUniformMaterialization
    model: BlockStateModel | GaussianHMMFit
    historical_source_states: np.ndarray
    source_continuation_links: np.ndarray


def registered_block_uniform_identity(
    value: RegisteredBlockUniformMaterialization,
) -> RegisteredBlockUniformIdentity:
    """Re-hash one code-minted common draw set and return its exact identity."""

    if (
        not isinstance(value, RegisteredBlockUniformMaterialization)
        or value._execution_seal is not _UNIFORM_MATERIALIZATION_CAPABILITY
        or _MINTED_UNIFORM_MATERIALIZATIONS.get(id(value)) is not value
    ):
        raise ModelError("block uniform materialization lacks producer authority")
    master_seed = value.master_seed
    scientific_version = value.scientific_version
    if master_seed is None or scientific_version is None:
        raise ModelError("block uniform materialization RNG identity is incomplete")
    geometry = value.geometry
    expected_geometry = resolve_calibration_geometry(
        artifact=geometry.artifact,
        frequency=geometry.frequency,
        strict_canonical=geometry.strict_canonical,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if geometry.frequency == "daily" else None
        ),
    )
    if geometry != expected_geometry:
        raise ModelError("block uniform materialization geometry is inconsistent")
    state, restart, rank = _verified_immutable_uniform_matrices(value)
    allocated = state.nbytes + restart.nbytes + rank.nbytes
    expected_allocated = registered_block_uniform_bytes(
        artifact=geometry.artifact,
        frequency=geometry.frequency,
        history_count=value.history_count,
        strict_canonical=geometry.strict_canonical,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if geometry.frequency == "daily" else None
        ),
    )
    if value.allocated_bytes != allocated or allocated != expected_allocated:
        raise ModelError("block uniform materialization byte accounting changed")
    namespace = _registered_rng_namespace(geometry.artifact, geometry.frequency)
    rng_fingerprint = _block_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        geometry=geometry,
        history_count=value.history_count,
    )
    stream_fingerprint = _block_uniform_stream_fingerprint(
        geometry=geometry,
        history_count=value.history_count,
        matrices=value.matrices,
    )
    materialization_fingerprint = _block_uniform_materialization_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        geometry=geometry,
        history_count=value.history_count,
        allocated_bytes=allocated,
        rng_namespace=namespace,
        rng_fingerprint=rng_fingerprint,
        matrices=value.matrices,
    )
    if (
        value.rng_namespace != namespace
        or value.rng_execution_fingerprint != rng_fingerprint
        or value.stream_fingerprint != stream_fingerprint
        or value.materialization_fingerprint != materialization_fingerprint
    ):
        raise ModelError("block uniform materialization identity changed")
    return RegisteredBlockUniformIdentity(
        master_seed,
        scientific_version,
        geometry.artifact,
        geometry.frequency,
        value.history_count,
        namespace,
        rng_fingerprint,
        stream_fingerprint,
        materialization_fingerprint,
    )


def is_registered_block_uniform_materialization(
    value: object,
) -> TypeGuard[RegisteredBlockUniformMaterialization]:
    """Return whether ``value`` is the unchanged code-minted draw set."""

    if not isinstance(value, RegisteredBlockUniformMaterialization):
        return False
    try:
        registered_block_uniform_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_block_calibration_identity(
    value: RegisteredBlockCalibration,
) -> RegisteredBlockCalibrationIdentity:
    """Revalidate a block decision against live materialization and parents."""

    if (
        not isinstance(value, RegisteredBlockCalibration)
        or value._execution_seal is not _BLOCK_CALIBRATION_CAPABILITY
        or _MINTED_BLOCK_CALIBRATIONS.get(id(value)) is not value
    ):
        raise ModelError("registered block calibration lacks executor authority")
    materialization = value._source_materialization
    model = value._source_model
    states = value._source_states
    links = value._source_continuation_links
    if materialization is None or model is None or states is None or links is None:
        raise ModelError("registered block calibration source identity is incomplete")
    materialization_identity = registered_block_uniform_identity(materialization)
    geometry = materialization.geometry
    if (
        value.execution.geometry != geometry
        or value.execution.histories != materialization.history_count
        or value.history_stream_sets != materialization.history_count
        or value.state_uniforms_per_history != geometry.monthly_observations
        or value.restart_uniforms_per_history != geometry.target_observations
        or value.source_rank_uniforms_per_history != geometry.target_observations
        or _positive_integer(value.uniform_chunk_size, "uniform chunk size")
        != value.uniform_chunk_size
    ):
        raise ModelError("registered block calibration accounting is inconsistent")
    _require_immutable_bytes_array(states, "historical source states")
    _require_immutable_bytes_array(links, "source continuation links")
    parent_model_fingerprint = _block_parent_model_fingerprint(model)
    source_history_fingerprint = _block_source_history_fingerprint(states, links)
    source_parent_fingerprint = _block_source_parent_fingerprint(
        parent_model_fingerprint,
        states,
        links,
    )
    execution_fingerprint = block_calibration_execution_fingerprint(value.execution)
    calibration_fingerprint = _registered_calibration_fingerprint(
        materialization_fingerprint=materialization_identity.materialization_fingerprint,
        parent_model_fingerprint=parent_model_fingerprint,
        source_history_fingerprint=source_history_fingerprint,
        source_parent_fingerprint=source_parent_fingerprint,
        execution_fingerprint=execution_fingerprint,
        execution=value.execution,
    )
    if (
        value.master_seed != materialization_identity.master_seed
        or value.scientific_version != materialization_identity.scientific_version
        or value.rng_namespace != materialization_identity.rng_namespace
        or value.rng_execution_fingerprint
        != materialization_identity.rng_execution_fingerprint
        or value.materialization_fingerprint
        != materialization_identity.materialization_fingerprint
        or value.parent_model_fingerprint != parent_model_fingerprint
        or value.source_history_fingerprint != source_history_fingerprint
        or value.source_parent_fingerprint != source_parent_fingerprint
        or value.execution_fingerprint != execution_fingerprint
        or value.calibration_fingerprint != calibration_fingerprint
    ):
        raise ModelError("registered block calibration identity changed")
    return RegisteredBlockCalibrationIdentity(
        materialization_identity.master_seed,
        materialization_identity.scientific_version,
        geometry.artifact,
        geometry.frequency,
        materialization_identity.materialization_fingerprint,
        parent_model_fingerprint,
        source_history_fingerprint,
        source_parent_fingerprint,
        execution_fingerprint,
        calibration_fingerprint,
    )


def registered_block_source_episode_counts(
    value: RegisteredBlockCalibration,
) -> tuple[int, ...]:
    """Return source-history run counts after revalidating the full lineage."""

    registered_block_calibration_identity(value)
    model = value._source_model
    states = value._source_states
    links = value._source_continuation_links
    if model is None or states is None or links is None:
        raise ModelError("registered block calibration source identity is incomplete")
    n_states = model.n_states
    labels = np.asarray(states, dtype=np.int64)
    continuation = np.asarray(links, dtype=np.bool_)
    if labels.ndim != 1 or continuation.shape != (len(labels) - 1,):
        raise ModelError("registered block source episode geometry is invalid")
    starts = np.ones(len(labels), dtype=np.bool_)
    if len(labels) > 1:
        starts[1:] = (~continuation) | (labels[1:] != labels[:-1])
    counts = np.bincount(labels[starts], minlength=n_states)
    if counts.shape != (n_states,):
        raise ModelError("registered block source episode counts are incomplete")
    return tuple(int(count) for count in counts)


def registered_block_support_failure_identity(
    value: RegisteredBlockSupportFailure,
) -> RegisteredBlockSupportFailureIdentity:
    """Revalidate one exact failed block cell and every live source parent."""

    if not isinstance(value, RegisteredBlockSupportFailure):
        raise ModelError("registered block support failure has the wrong type")
    record = _MINTED_BLOCK_SUPPORT_FAILURES.get(id(value))
    if (
        value._execution_seal is not _BLOCK_SUPPORT_FAILURE_CAPABILITY
        or record is None
        or record.value is not value
        or value.evidence is not record.evidence
        or value._source_model is not record.model
        or value._source_states is not record.states
        or value._source_continuation_links is not record.links
        or value._source_materialization is not record.materialization
    ):
        raise ModelError("block support failure lacks exact producer authority")
    materialization = registered_block_uniform_identity(record.materialization)
    _require_immutable_bytes_array(record.states, "historical source states")
    _require_immutable_bytes_array(record.links, "source continuation links")
    parent_model_fingerprint = _block_parent_model_fingerprint(record.model)
    source_history_fingerprint = _block_source_history_fingerprint(
        record.states,
        record.links,
    )
    source_parent_fingerprint = _block_source_parent_fingerprint(
        parent_model_fingerprint,
        record.states,
        record.links,
    )
    selected_k = record.model.n_states
    failure_fingerprint = block_support_failure_evidence_fingerprint(record.evidence)
    if (
        value.contract_version != REGISTERED_BLOCK_SUPPORT_FAILURE_CONTRACT
        or record.evidence.geometry != record.materialization.geometry
        or record.evidence.histories != record.materialization.history_count
        or value.master_seed != materialization.master_seed
        or value.scientific_version != materialization.scientific_version
        or value.artifact != materialization.artifact
        or value.frequency != materialization.frequency
        or value.selected_k != selected_k
        or value.starting_length != record.evidence.starting_length
        or value.materialization_fingerprint
        != materialization.materialization_fingerprint
        or value.parent_model_fingerprint != parent_model_fingerprint
        or value.source_history_fingerprint != source_history_fingerprint
        or value.source_parent_fingerprint != source_parent_fingerprint
        or value.failure_fingerprint != failure_fingerprint
        or record.failure_fingerprint != failure_fingerprint
    ):
        raise ModelError("registered block support failure identity changed")
    return RegisteredBlockSupportFailureIdentity(
        master_seed=materialization.master_seed,
        scientific_version=materialization.scientific_version,
        artifact=materialization.artifact,
        frequency=materialization.frequency,
        selected_k=selected_k,
        starting_length=record.evidence.starting_length,
        failure_code=record.evidence.failure_code,
        materialization_fingerprint=materialization.materialization_fingerprint,
        parent_model_fingerprint=parent_model_fingerprint,
        source_history_fingerprint=source_history_fingerprint,
        source_parent_fingerprint=source_parent_fingerprint,
        failure_fingerprint=failure_fingerprint,
    )


def is_registered_block_support_failure(
    value: object,
) -> TypeGuard[RegisteredBlockSupportFailure]:
    if not isinstance(value, RegisteredBlockSupportFailure):
        return False
    try:
        registered_block_support_failure_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_block_core_identity(
    value: RegisteredBlockCalibration,
) -> RegisteredBlockCoreIdentity:
    """Reverify and return the task-safe pre-fragmentation block identity."""

    legacy_identity = registered_block_calibration_identity(value)
    materialization = value._source_materialization
    model = value._source_model
    if materialization is None or model is None:
        raise ModelError("registered block core source identity is incomplete")
    uniform = registered_block_uniform_identity(materialization)
    execution = value.execution
    selected_k = model.n_states
    if isinstance(selected_k, bool) or selected_k <= 0:
        raise ModelError("registered block core has an invalid selected K")
    core_execution_fingerprint = block_calibration_core_execution_fingerprint(execution)
    core_fingerprint = _registered_block_core_fingerprint(
        selected_k=selected_k,
        uniform=uniform,
        parent_model_fingerprint=legacy_identity.parent_model_fingerprint,
        source_history_fingerprint=legacy_identity.source_history_fingerprint,
        source_parent_fingerprint=legacy_identity.source_parent_fingerprint,
        core_execution_fingerprint=core_execution_fingerprint,
        execution=execution,
    )
    return RegisteredBlockCoreIdentity(
        master_seed=uniform.master_seed,
        scientific_version=uniform.scientific_version,
        artifact=uniform.artifact,
        frequency=uniform.frequency,
        selected_k=selected_k,
        histories=execution.histories,
        starting_length=execution.starting_length,
        selected_length=execution.selected_length,
        rng_namespace=uniform.rng_namespace,
        rng_execution_fingerprint=uniform.rng_execution_fingerprint,
        uniform_stream_fingerprint=uniform.stream_fingerprint,
        materialization_fingerprint=uniform.materialization_fingerprint,
        parent_model_fingerprint=legacy_identity.parent_model_fingerprint,
        source_history_fingerprint=legacy_identity.source_history_fingerprint,
        source_parent_fingerprint=legacy_identity.source_parent_fingerprint,
        core_execution_fingerprint=core_execution_fingerprint,
        core_fingerprint=core_fingerprint,
    )


def registered_block_fragmentation_identity(
    value: RegisteredBlockCalibration,
) -> RegisteredBlockFragmentationIdentity:
    """Return exact fragmentation evidence under its task-safe core parent."""

    core = registered_block_core_identity(value)
    fingerprint = scientific_materialization_fingerprint(
        schema_id="registered_block_fragmentation_evidence",
        metadata={
            "artifact": core.artifact,
            "frequency": core.frequency,
            "selected_k": core.selected_k,
            "selected_length": core.selected_length,
            "parent_core_fingerprint": core.core_fingerprint,
            "fragmentation_gates": [
                _fragmentation_gate_identity(item)
                for item in value.execution.selected_fragmentation_gates
            ],
        },
        arrays={
            "selected_identity": np.asarray(
                [core.selected_k, core.selected_length], dtype=np.int64
            )
        },
    )
    return RegisteredBlockFragmentationIdentity(
        artifact=core.artifact,
        frequency=core.frequency,
        selected_k=core.selected_k,
        selected_length=core.selected_length,
        parent_core_fingerprint=core.core_fingerprint,
        fragmentation_evidence_fingerprint=fingerprint,
    )


def registered_block_replay_source(
    value: RegisteredBlockCalibration,
) -> RegisteredBlockReplaySource:
    """Return exact live sources for later replay without drawing new uniforms."""

    core = registered_block_core_identity(value)
    materialization = value._source_materialization
    model = value._source_model
    states = value._source_states
    links = value._source_continuation_links
    if materialization is None or model is None or states is None or links is None:
        raise ModelError("registered block replay source identity is incomplete")
    uniform = registered_block_uniform_identity(materialization)
    if uniform.materialization_fingerprint != core.materialization_fingerprint:
        raise ModelError("registered block replay materialization changed")
    return RegisteredBlockReplaySource(
        core_identity=core,
        uniform_identity=uniform,
        materialization=materialization,
        model=model,
        historical_source_states=states,
        source_continuation_links=links,
    )


def is_registered_block_calibration(
    value: object,
) -> TypeGuard[RegisteredBlockCalibration]:
    """Return whether ``value`` is an unchanged code-executed calibration."""

    if not isinstance(value, RegisteredBlockCalibration):
        return False
    try:
        registered_block_calibration_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_block_uniform_bytes(
    *,
    artifact: Artifact,
    frequency: Frequency,
    history_count: int = CANONICAL_HISTORY_COUNT,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
) -> int:
    """Return the exact binary64 allocation before any draw is materialized."""

    histories = _positive_integer(history_count, "history count")
    if strict_canonical and histories != CANONICAL_HISTORY_COUNT:
        raise ModelError("strict canonical block RNG requires exactly 10,000 histories")
    geometry = resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    values = histories * (
        geometry.monthly_observations + 2 * geometry.target_observations
    )
    return values * np.dtype(np.float64).itemsize


def materialize_registered_block_uniforms(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: Artifact,
    frequency: Frequency,
    history_count: int = CANONICAL_HISTORY_COUNT,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
) -> RegisteredBlockUniformMaterialization:
    """Create the common draw bytes once for all candidate comparisons.

    The returned arrays are immutable.  A coordinator may iterate over them
    repeatedly for K-specific block calibration and then for the selected-L
    dependence reduction without creating a second stream materialization.
    """

    histories = _positive_integer(history_count, "history count")
    geometry = resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    allocated = registered_block_uniform_bytes(
        artifact=artifact,
        frequency=frequency,
        history_count=histories,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    state = np.empty((histories, geometry.monthly_observations), dtype=np.float64)
    restart = np.empty((histories, geometry.target_observations), dtype=np.float64)
    rank = np.empty((histories, geometry.target_observations), dtype=np.float64)
    for chunk in iter_registered_block_uniform_chunks(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        frequency=frequency,
        history_count=histories,
        chunk_size=chunk_size,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    ):
        start = chunk.history_start
        rows = np.asarray(chunk.monthly_state_uniforms).shape[0]
        stop = start + rows
        state[start:stop] = chunk.monthly_state_uniforms
        restart[start:stop] = chunk.restart_uniforms
        rank[start:stop] = chunk.source_rank_uniforms
    if state.nbytes + restart.nbytes + rank.nbytes != allocated:
        raise AssertionError("registered block uniform byte accounting is inconsistent")
    # Freeze one matrix at a time and promptly release its mutable allocation.
    # This matters for canonical daily geometry, where retaining all three source
    # arrays beside all three immutable copies would nearly double peak memory.
    immutable_state = _immutable_bytes_array(state, dtype=np.dtype("<f8"))
    del state
    immutable_restart = _immutable_bytes_array(restart, dtype=np.dtype("<f8"))
    del restart
    immutable_rank = _immutable_bytes_array(rank, dtype=np.dtype("<f8"))
    del rank
    matrices = RegisteredUniformMatrices(
        immutable_state,
        immutable_restart,
        immutable_rank,
    )
    namespace = _registered_rng_namespace(artifact, frequency)
    rng_fingerprint = _block_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        geometry=geometry,
        history_count=histories,
    )
    stream_fingerprint = _block_uniform_stream_fingerprint(
        geometry=geometry,
        history_count=histories,
        matrices=matrices,
    )
    materialization_fingerprint = _block_uniform_materialization_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        geometry=geometry,
        history_count=histories,
        allocated_bytes=allocated,
        rng_namespace=namespace,
        rng_fingerprint=rng_fingerprint,
        matrices=matrices,
    )
    result = RegisteredBlockUniformMaterialization(
        geometry=geometry,
        matrices=matrices,
        history_count=histories,
        allocated_bytes=allocated,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_namespace=namespace,
        rng_execution_fingerprint=rng_fingerprint,
        stream_fingerprint=stream_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
    )
    object.__setattr__(
        result,
        "_execution_seal",
        _UNIFORM_MATERIALIZATION_CAPABILITY,
    )
    _MINTED_UNIFORM_MATERIALIZATIONS[id(result)] = result
    return result


def run_materialized_block_calibration(
    materialization: RegisteredBlockUniformMaterialization,
    *,
    model: BlockStateModel | GaussianHMMFit,
    starting_length: int,
    historical_source_states: np.typing.ArrayLike,
    source_continuation_links: np.typing.ArrayLike,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    trace_consumer: Callable[[PairedTraceChunk], None] | None = None,
) -> BlockCalibrationExecution:
    """Run one comparison from an existing common immutable draw set."""

    registered_block_uniform_identity(materialization)
    chunk = _positive_integer(chunk_size, "uniform chunk size")
    geometry = materialization.geometry
    uniform_chunks = uniform_chunks_from_matrices(
        materialization.matrices,
        chunk_size=chunk,
    )
    return execute_block_calibration(
        model=model,
        artifact=geometry.artifact,
        frequency=geometry.frequency,
        starting_length=starting_length,
        historical_source_states=historical_source_states,
        source_continuation_links=source_continuation_links,
        uniform_chunks=uniform_chunks,
        history_count=materialization.history_count,
        strict_canonical=geometry.strict_canonical,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if geometry.frequency == "daily" else None
        ),
        trace_consumer=trace_consumer,
    )


def iter_materialized_block_trace_chunks(
    materialization: RegisteredBlockUniformMaterialization,
    *,
    model: BlockStateModel | GaussianHMMFit,
    historical_source_states: np.typing.ArrayLike,
    source_continuation_links: np.typing.ArrayLike,
    intended_lengths: Sequence[int],
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
) -> Iterator[PairedTraceChunk]:
    """Replay only selected/neighbor L traces from the one common draw set.

    This avoids a second all-length histogram pass after calibration has
    frozen the selected length.  Consumers such as dependence and kernel
    diagnostics receive the exact same materialized purpose-4 draws.
    """

    registered_block_uniform_identity(materialization)
    chunk = _positive_integer(chunk_size, "uniform chunk size")
    lengths = tuple(intended_lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ModelError("trace replay lengths must be non-empty and unique")
    geometry = materialization.geometry
    uniform_chunks = uniform_chunks_from_matrices(
        materialization.matrices,
        chunk_size=chunk,
    )
    yield from iter_paired_trace_chunks(
        model=model,
        geometry=geometry,
        historical_source_states=historical_source_states,
        source_continuation_links=source_continuation_links,
        intended_lengths=lengths,
        uniform_chunks=uniform_chunks,
        expected_history_count=materialization.history_count,
    )


def iter_registered_block_uniform_chunks(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: Artifact,
    frequency: Frequency,
    history_count: int,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
) -> Iterator[RegisteredUniformChunk]:
    """Materialize one bounded chunk from three distinct keys per history."""

    histories = _positive_integer(history_count, "history count")
    chunk = _positive_integer(chunk_size, "uniform chunk size")
    if strict_canonical and histories != CANONICAL_HISTORY_COUNT:
        raise ModelError(
            "strict canonical block RNG requires exactly 10,000 histories",
            details={"actual": histories, "expected": CANONICAL_HISTORY_COUNT},
        )
    geometry = resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    registered_artifact = _registered_artifact(artifact)
    family = _registered_family(frequency)
    for start in range(0, histories, chunk):
        stop = min(histories, start + chunk)
        rows = stop - start
        state = np.empty((rows, geometry.monthly_observations), dtype=np.float64)
        restart = np.empty((rows, geometry.target_observations), dtype=np.float64)
        rank = np.empty((rows, geometry.target_observations), dtype=np.float64)
        for row, replicate in enumerate(range(start, stop)):
            state[row] = _uniforms_for_registered_stage(
                master_seed=master_seed,
                scientific_version=scientific_version,
                artifact=registered_artifact,
                family=family,
                stage=Stage.STATE_CHAIN,
                replicate=replicate,
                count=geometry.monthly_observations,
            )
            restart[row] = _uniforms_for_registered_stage(
                master_seed=master_seed,
                scientific_version=scientific_version,
                artifact=registered_artifact,
                family=family,
                stage=Stage.RESTART_UNIFORMS,
                replicate=replicate,
                count=geometry.target_observations,
            )
            rank[row] = _uniforms_for_registered_stage(
                master_seed=master_seed,
                scientific_version=scientific_version,
                artifact=registered_artifact,
                family=family,
                stage=Stage.SOURCE_RANK_UNIFORMS,
                replicate=replicate,
                count=geometry.target_observations,
            )
        state.setflags(write=False)
        restart.setflags(write=False)
        rank.setflags(write=False)
        yield RegisteredUniformChunk(start, state, restart, rank)


def run_registered_block_calibration(
    *,
    model: BlockStateModel | GaussianHMMFit,
    master_seed: int,
    scientific_version: int,
    artifact: Artifact,
    frequency: Frequency,
    starting_length: int,
    historical_source_states: np.typing.ArrayLike,
    source_continuation_links: np.typing.ArrayLike,
    history_count: int = CANONICAL_HISTORY_COUNT,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
    trace_consumer: Callable[[PairedTraceChunk], None] | None = None,
) -> RegisteredBlockCalibration:
    """Execute one cell from its closed purpose-4 RNG namespace."""

    histories = _positive_integer(history_count, "history count")
    chunk = _positive_integer(chunk_size, "uniform chunk size")
    materialization = materialize_registered_block_uniforms(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        frequency=frequency,
        history_count=histories,
        chunk_size=chunk,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    return run_registered_materialized_block_calibration(
        materialization,
        model=model,
        starting_length=starting_length,
        historical_source_states=historical_source_states,
        source_continuation_links=source_continuation_links,
        chunk_size=chunk,
        trace_consumer=trace_consumer,
    )


def run_registered_materialized_block_calibration(
    materialization: RegisteredBlockUniformMaterialization,
    *,
    model: BlockStateModel | GaussianHMMFit,
    starting_length: int,
    historical_source_states: np.typing.ArrayLike,
    source_continuation_links: np.typing.ArrayLike,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    trace_consumer: Callable[[PairedTraceChunk], None] | None = None,
) -> RegisteredBlockCalibration:
    """Execute one model/K/L comparison from a shared purpose-4 draw set.

    Repeated calls may bind different candidate models or starting lengths to
    the same ``materialization_fingerprint``.  No call creates or consumes a
    second RNG namespace, preserving the registered one-materialization-
    across-L contract.
    """

    materialization_identity = registered_block_uniform_identity(materialization)
    chunk = _positive_integer(chunk_size, "uniform chunk size")
    states = _immutable_bytes_array(
        historical_source_states,
        dtype=np.dtype("<i8"),
    )
    links = _immutable_bytes_array(
        source_continuation_links,
        dtype=np.dtype("?"),
    )
    parent_model_fingerprint = _block_parent_model_fingerprint(model)
    source_history_fingerprint = _block_source_history_fingerprint(states, links)
    source_parent_fingerprint = _block_source_parent_fingerprint(
        parent_model_fingerprint,
        states,
        links,
    )
    execution = run_materialized_block_calibration(
        materialization,
        model=model,
        starting_length=starting_length,
        historical_source_states=states,
        source_continuation_links=links,
        chunk_size=chunk,
        trace_consumer=trace_consumer,
    )
    if _block_parent_model_fingerprint(model) != parent_model_fingerprint:
        raise ModelError("block calibration parent model changed during execution")
    execution_fingerprint = block_calibration_execution_fingerprint(execution)
    calibration_fingerprint = _registered_calibration_fingerprint(
        materialization_fingerprint=(
            materialization_identity.materialization_fingerprint
        ),
        parent_model_fingerprint=parent_model_fingerprint,
        source_history_fingerprint=source_history_fingerprint,
        source_parent_fingerprint=source_parent_fingerprint,
        execution_fingerprint=execution_fingerprint,
        execution=execution,
    )
    geometry = materialization.geometry
    result = RegisteredBlockCalibration(
        execution=execution,
        history_stream_sets=materialization.history_count,
        state_uniforms_per_history=geometry.monthly_observations,
        restart_uniforms_per_history=geometry.target_observations,
        source_rank_uniforms_per_history=geometry.target_observations,
        uniform_chunk_size=chunk,
        master_seed=materialization_identity.master_seed,
        scientific_version=materialization_identity.scientific_version,
        rng_namespace=materialization_identity.rng_namespace,
        rng_execution_fingerprint=(materialization_identity.rng_execution_fingerprint),
        materialization_fingerprint=(
            materialization_identity.materialization_fingerprint
        ),
        parent_model_fingerprint=parent_model_fingerprint,
        source_history_fingerprint=source_history_fingerprint,
        source_parent_fingerprint=source_parent_fingerprint,
        execution_fingerprint=execution_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
    )
    object.__setattr__(result, "_source_model", model)
    object.__setattr__(result, "_source_states", states)
    object.__setattr__(result, "_source_continuation_links", links)
    object.__setattr__(result, "_source_materialization", materialization)
    object.__setattr__(result, "_execution_seal", _BLOCK_CALIBRATION_CAPABILITY)
    _MINTED_BLOCK_CALIBRATIONS[id(result)] = result
    return result


def run_registered_materialized_block_calibration_outcome(
    materialization: RegisteredBlockUniformMaterialization,
    *,
    model: BlockStateModel | GaussianHMMFit,
    starting_length: int,
    historical_source_states: np.typing.ArrayLike,
    source_continuation_links: np.typing.ArrayLike,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
) -> RegisteredBlockCalibration | RegisteredBlockSupportFailure:
    """Return a success or sealed terminal per-K scientific failure.

    The catch list is deliberately closed.  Structural ``ModelError`` values,
    integrity faults, callback/API faults, resource failures, and unexpected
    exceptions propagate without creating a scientific outcome.
    """

    materialization_identity = registered_block_uniform_identity(materialization)
    chunk = _positive_integer(chunk_size, "uniform chunk size")
    states = _immutable_bytes_array(
        historical_source_states,
        dtype=np.dtype("<i8"),
    )
    links = _immutable_bytes_array(
        source_continuation_links,
        dtype=np.dtype("?"),
    )
    parent_model_fingerprint = _block_parent_model_fingerprint(model)
    source_history_fingerprint = _block_source_history_fingerprint(states, links)
    source_parent_fingerprint = _block_source_parent_fingerprint(
        parent_model_fingerprint,
        states,
        links,
    )
    try:
        return run_registered_materialized_block_calibration(
            materialization,
            model=model,
            starting_length=starting_length,
            historical_source_states=states,
            source_continuation_links=links,
            chunk_size=chunk,
        )
    except (
        BlockCalibrationSupportFailure,
        BlockCalibrationNumericalFailure,
    ) as error:
        evidence = error.evidence
    if _block_parent_model_fingerprint(model) != parent_model_fingerprint:
        raise ModelError("block calibration parent model changed during execution")
    if registered_block_uniform_identity(materialization) != materialization_identity:
        raise ModelError("block materialization changed during failed execution")
    failure_fingerprint = block_support_failure_evidence_fingerprint(evidence)
    value = object.__new__(RegisteredBlockSupportFailure)
    for name, item in (
        ("contract_version", REGISTERED_BLOCK_SUPPORT_FAILURE_CONTRACT),
        ("evidence", evidence),
        ("master_seed", materialization_identity.master_seed),
        ("scientific_version", materialization_identity.scientific_version),
        ("artifact", materialization_identity.artifact),
        ("frequency", materialization_identity.frequency),
        ("selected_k", model.n_states),
        ("starting_length", evidence.starting_length),
        (
            "materialization_fingerprint",
            materialization_identity.materialization_fingerprint,
        ),
        ("parent_model_fingerprint", parent_model_fingerprint),
        ("source_history_fingerprint", source_history_fingerprint),
        ("source_parent_fingerprint", source_parent_fingerprint),
        ("failure_fingerprint", failure_fingerprint),
        ("_source_model", model),
        ("_source_states", states),
        ("_source_continuation_links", links),
        ("_source_materialization", materialization),
        ("_execution_seal", _BLOCK_SUPPORT_FAILURE_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    _MINTED_BLOCK_SUPPORT_FAILURES[id(value)] = _BlockSupportFailureMintRecord(
        value=value,
        evidence=evidence,
        model=model,
        states=states,
        links=links,
        materialization=materialization,
        failure_fingerprint=failure_fingerprint,
    )
    try:
        registered_block_support_failure_identity(value)
    except Exception:
        _MINTED_BLOCK_SUPPORT_FAILURES.pop(id(value), None)
        raise
    return value


def block_support_failure_evidence_fingerprint(
    value: BlockSupportFailureEvidence,
) -> str:
    """Fingerprint every compact statistic in one terminal failed block cell."""

    if not isinstance(value, BlockSupportFailureEvidence):
        raise ModelError("block support failure evidence has the wrong type")
    if value.failure_code not in {
        "historical_support_unattained",
        "effective_support_unattained",
        "effective_length_numerical_failure",
    }:
        raise ModelError("block support failure code is invalid")
    simulation_lengths = tuple(item.intended_length for item in value.simulations)
    if simulation_lengths != value.structurally_feasible_lengths:
        raise ModelError("block support failure simulation order is invalid")
    if value.failure_code == "historical_support_unattained":
        if value.structurally_feasible_lengths or value.simulations:
            raise ModelError("historical support failure retained simulations")
    elif not value.structurally_feasible_lengths:
        raise ModelError("post-simulation block failure lacks simulations")
    return scientific_materialization_fingerprint(
        schema_id="registered_block_support_failure_evidence_v1",
        metadata={
            "geometry": _geometry_identity(value.geometry),
            "histories": value.histories,
            "starting_length": value.starting_length,
            "structurally_feasible_lengths": list(value.structurally_feasible_lengths),
            "simulations": [
                {
                    "intended_length": item.intended_length,
                    "histories": item.histories,
                    "conditioned": [
                        _pooled_statistics_identity(statistic)
                        for statistic in item.conditioned
                    ],
                    "ideal": [
                        _pooled_statistics_identity(statistic)
                        for statistic in item.ideal
                    ],
                }
                for item in value.simulations
            ],
            "failure_code": value.failure_code,
            "failure_message": value.failure_message,
            "failure_details": value.failure_details,
        },
        arrays={
            "length_identity": np.asarray(
                [value.starting_length, *value.structurally_feasible_lengths],
                dtype=np.int64,
            )
        },
    )


def block_calibration_execution_fingerprint(
    value: BlockCalibrationExecution,
) -> str:
    """Fingerprint every compact statistic and decision in one block result."""

    if not isinstance(value, BlockCalibrationExecution):
        raise ModelError("block execution fingerprint requires a block execution")
    geometry = value.geometry
    metadata = {
        "geometry": _geometry_identity(geometry),
        "histories": value.histories,
        "starting_length": value.starting_length,
        "structurally_feasible_lengths": list(value.structurally_feasible_lengths),
        "simulations": [
            {
                "intended_length": item.intended_length,
                "histories": item.histories,
                "conditioned": [
                    _pooled_statistics_identity(statistic)
                    for statistic in item.conditioned
                ],
                "ideal": [
                    _pooled_statistics_identity(statistic) for statistic in item.ideal
                ],
            }
            for item in value.simulations
        ],
        "policy_selection": _block_policy_identity(value.policy_selection),
        "dependence_reuse_lengths": list(value.dependence_reuse_lengths),
        "selected_fragmentation_gates": [
            _fragmentation_gate_identity(item)
            for item in value.selected_fragmentation_gates
        ],
    }
    return scientific_materialization_fingerprint(
        schema_id="registered_block_calibration_execution",
        metadata=metadata,
        arrays={
            "length_identity": np.asarray(
                [value.starting_length, value.selected_length],
                dtype=np.int64,
            )
        },
    )


def block_calibration_core_execution_fingerprint(
    value: BlockCalibrationExecution,
) -> str:
    """Fingerprint block selection evidence with no later-stage outcomes.

    Raw paired calibration summaries remain core scientific evidence because
    they determine the registered selected L.  The selected-L fragmentation
    gates and the dependence replay schedule are intentionally absent.
    """

    if not isinstance(value, BlockCalibrationExecution):
        raise ModelError("block core fingerprint requires a block execution")
    metadata = {
        "geometry": _geometry_identity(value.geometry),
        "histories": value.histories,
        "starting_length": value.starting_length,
        "structurally_feasible_lengths": list(value.structurally_feasible_lengths),
        "simulations": [
            {
                "intended_length": item.intended_length,
                "histories": item.histories,
                "conditioned": [
                    _pooled_statistics_identity(statistic)
                    for statistic in item.conditioned
                ],
                "ideal": [
                    _pooled_statistics_identity(statistic) for statistic in item.ideal
                ],
            }
            for item in value.simulations
        ],
        "policy_selection": _block_policy_identity(value.policy_selection),
        "fragmentation_evidence_excluded": True,
        "dependence_evidence_excluded": True,
    }
    return scientific_materialization_fingerprint(
        schema_id="registered_block_calibration_core_execution",
        metadata=metadata,
        arrays={
            "length_identity": np.asarray(
                [value.starting_length, value.selected_length], dtype=np.int64
            )
        },
    )


def _registered_rng_namespace(artifact: Artifact, frequency: Frequency) -> str:
    _registered_artifact(artifact)
    _registered_family(frequency)
    return f"purpose4_conditioned_block_{artifact}_{frequency}"


def _block_rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    geometry: CalibrationGeometry,
    history_count: int,
) -> str:
    histories = _positive_integer(history_count, "history count")
    registered_artifact = _registered_artifact(geometry.artifact)
    family = _registered_family(geometry.frequency)
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace=_registered_rng_namespace(
            geometry.artifact,
            geometry.frequency,
        ),
        contract={
            "artifact": int(registered_artifact),
            "domain": int(Domain.BLOCK_ALPHA_CALIBRATION),
            "family": int(family),
            "first_replicate": 0,
            "history_count": histories,
            "last_replicate": histories - 1,
            "monthly_uniforms_per_history": geometry.monthly_observations,
            "purpose": int(CalibrationPurpose.CONDITIONED_BLOCK),
            "stages": [
                int(Stage.STATE_CHAIN),
                int(Stage.RESTART_UNIFORMS),
                int(Stage.SOURCE_RANK_UNIFORMS),
            ],
            "target_uniforms_per_history_per_stage": (geometry.target_observations),
            "variant": 0,
        },
    )


def _block_uniform_stream_fingerprint(
    *,
    geometry: CalibrationGeometry,
    history_count: int,
    matrices: RegisteredUniformMatrices,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_block_uniform_stream",
        metadata={
            "geometry": _geometry_identity(geometry),
            "history_count": history_count,
            "shared_across_candidate_k_and_l": True,
        },
        arrays={
            "monthly_state_uniforms": matrices.monthly_state_uniforms,
            "restart_uniforms": matrices.restart_uniforms,
            "source_rank_uniforms": matrices.source_rank_uniforms,
        },
    )


def _block_uniform_materialization_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    geometry: CalibrationGeometry,
    history_count: int,
    allocated_bytes: int,
    rng_namespace: str,
    rng_fingerprint: str,
    matrices: RegisteredUniformMatrices,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_block_uniform_materialization",
        metadata={
            "master_seed": master_seed,
            "scientific_version": scientific_version,
            "geometry": _geometry_identity(geometry),
            "history_count": history_count,
            "allocated_bytes": allocated_bytes,
            "rng_namespace": rng_namespace,
            "rng_execution_fingerprint": rng_fingerprint,
            "shared_across_candidate_k_and_l": True,
        },
        arrays={
            "monthly_state_uniforms": matrices.monthly_state_uniforms,
            "restart_uniforms": matrices.restart_uniforms,
            "source_rank_uniforms": matrices.source_rank_uniforms,
        },
    )


def _verified_immutable_uniform_matrices(
    value: RegisteredBlockUniformMaterialization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = value.geometry
    expected_shapes = (
        (value.history_count, geometry.monthly_observations),
        (value.history_count, geometry.target_observations),
        (value.history_count, geometry.target_observations),
    )
    arrays = tuple(
        np.asarray(item)
        for item in (
            value.matrices.monthly_state_uniforms,
            value.matrices.restart_uniforms,
            value.matrices.source_rank_uniforms,
        )
    )
    for array, expected, label in zip(
        arrays,
        expected_shapes,
        ("monthly state", "restart", "source-rank"),
        strict=True,
    ):
        if (
            array.shape != expected
            or array.dtype != np.dtype("<f8")
            or not bool(np.isfinite(array).all())
            or bool(((array < 0.0) | (array >= 1.0)).any())
        ):
            raise ModelError(f"registered {label} uniforms have invalid content")
        _require_immutable_bytes_array(array, f"registered {label} uniforms")
    return cast(tuple[np.ndarray, np.ndarray, np.ndarray], arrays)


def _immutable_bytes_array(
    value: np.typing.ArrayLike,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "registered block source cannot be converted to an array"
        ) from error
    if source.ndim == 0 or source.size == 0:
        raise ModelError("registered block source array must be nonempty")
    if dtype.kind == "f":
        allowed = source.dtype.kind == "f" and bool(np.isfinite(source).all())
    elif dtype.kind == "i":
        allowed = source.dtype.kind in "iu"
    elif dtype.kind == "b":
        allowed = source.dtype.kind == "b"
    else:  # pragma: no cover - all internal callers use the three cases above.
        raise AssertionError("unsupported immutable block-array dtype")
    if not allowed:
        raise ModelError("registered block source array has an invalid dtype or value")
    canonical = np.ascontiguousarray(source.astype(dtype, copy=False))
    immutable = np.frombuffer(canonical.tobytes(order="C"), dtype=dtype).reshape(
        canonical.shape
    )
    _require_immutable_bytes_array(immutable, "registered block source array")
    return immutable


def _require_immutable_bytes_array(value: np.ndarray, label: str) -> None:
    array = np.asarray(value)
    if array.flags.writeable:
        raise ModelError(f"{label} must be read-only")
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        raise ModelError(f"{label} must be backed by immutable bytes")


def _block_parent_model_fingerprint(
    model: BlockStateModel | GaussianHMMFit,
) -> str:
    if isinstance(model, GaussianHMMFit):
        return gaussian_hmm_fit_fingerprint(model)
    if not isinstance(model, BlockStateModel):
        raise ModelError("registered block calibration parent model is invalid")
    return scientific_materialization_fingerprint(
        schema_id="registered_block_state_model",
        metadata={"n_states": model.n_states},
        arrays={
            "stationary_distribution": model.stationary_distribution,
            "transition_matrix": model.transition_matrix,
        },
    )


def _block_source_history_fingerprint(
    states: np.ndarray,
    links: np.ndarray,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_block_source_history",
        metadata={"observations": len(states)},
        arrays={
            "source_continuation_links": links,
            "source_states": states,
        },
    )


def _block_source_parent_fingerprint(
    model_fingerprint: str,
    states: np.ndarray,
    links: np.ndarray,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_block_source_parent",
        metadata={"parent_model_fingerprint": model_fingerprint},
        arrays={
            "source_continuation_links": links,
            "source_states": states,
        },
    )


def _registered_calibration_fingerprint(
    *,
    materialization_fingerprint: str,
    parent_model_fingerprint: str,
    source_history_fingerprint: str,
    source_parent_fingerprint: str,
    execution_fingerprint: str,
    execution: BlockCalibrationExecution,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_block_calibration",
        metadata={
            "geometry": _geometry_identity(execution.geometry),
            "histories": execution.histories,
            "materialization_fingerprint": materialization_fingerprint,
            "parent_model_fingerprint": parent_model_fingerprint,
            "source_history_fingerprint": source_history_fingerprint,
            "source_parent_fingerprint": source_parent_fingerprint,
            "execution_fingerprint": execution_fingerprint,
            "schedule_independent": True,
        },
        arrays={
            "length_identity": np.asarray(
                [execution.starting_length, execution.selected_length],
                dtype=np.int64,
            )
        },
    )


def _registered_block_core_fingerprint(
    *,
    selected_k: int,
    uniform: RegisteredBlockUniformIdentity,
    parent_model_fingerprint: str,
    source_history_fingerprint: str,
    source_parent_fingerprint: str,
    core_execution_fingerprint: str,
    execution: BlockCalibrationExecution,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_block_calibration_core",
        metadata={
            "artifact": uniform.artifact,
            "frequency": uniform.frequency,
            "selected_k": selected_k,
            "histories": execution.histories,
            "starting_length": execution.starting_length,
            "selected_length": execution.selected_length,
            "master_seed": uniform.master_seed,
            "scientific_version": uniform.scientific_version,
            "rng_namespace": uniform.rng_namespace,
            "rng_execution_fingerprint": uniform.rng_execution_fingerprint,
            "uniform_stream_fingerprint": uniform.stream_fingerprint,
            "materialization_fingerprint": uniform.materialization_fingerprint,
            "parent_model_fingerprint": parent_model_fingerprint,
            "source_history_fingerprint": source_history_fingerprint,
            "source_parent_fingerprint": source_parent_fingerprint,
            "core_execution_fingerprint": core_execution_fingerprint,
            "fragmentation_evidence_excluded": True,
            "dependence_evidence_excluded": True,
            "schedule_independent": True,
        },
        arrays={
            "selected_identity": np.asarray(
                [selected_k, execution.starting_length, execution.selected_length],
                dtype=np.int64,
            )
        },
    )


def _geometry_identity(value: CalibrationGeometry) -> dict[str, object]:
    return {
        "artifact": value.artifact,
        "frequency": value.frequency,
        "monthly_segment_lengths": list(value.monthly_segment_lengths),
        "target_segment_lengths": list(value.target_segment_lengths),
        "strict_canonical": value.strict_canonical,
    }


def _pooled_statistics_identity(
    value: PooledBlockStatistics,
) -> dict[str, object]:
    return {
        "state": value.state,
        "count": value.count,
        "mean": value.mean,
        "minimum": value.minimum,
        "maximum": value.maximum,
        "median": value.median,
        "singleton_fraction": value.singleton_fraction,
        "quantile_05": value.quantile_05,
        "quantile_95": value.quantile_95,
        "length_histogram": list(value.length_histogram),
    }


def _block_policy_identity(value: BlockPolicySelection) -> dict[str, object]:
    return {
        "frequency": value.frequency,
        "selected_length": value.selected_length,
        "restart_probability": value.restart_probability,
        "trials": [
            {
                "intended_length": trial.intended_length,
                "required_eligible_starts": trial.required_eligible_starts,
                "support": [
                    {
                        "state": support.state,
                        "intended_length": support.intended_length,
                        "completed_episodes": support.completed_episodes,
                        "eligible_start_indices": list(support.eligible_start_indices),
                    }
                    for support in trial.support
                ],
                "realized_means": [list(item) for item in trial.realized_means],
                "reasons": list(trial.reasons),
            }
            for trial in value.trials
        ],
    }


def _fragmentation_gate_identity(
    value: PooledFragmentationGate,
) -> dict[str, object]:
    return {
        "state": value.state,
        "intended_length": value.intended_length,
        "conditioned": _pooled_statistics_identity(value.conditioned),
        "ideal": _pooled_statistics_identity(value.ideal),
        "conditioned_mean_minimum": value.conditioned_mean_minimum,
        "conditioned_median_minimum": value.conditioned_median_minimum,
        "singleton_fraction_maximum_from_ideal": (
            value.singleton_fraction_maximum_from_ideal
        ),
        "absolute_singleton_fraction_maximum": (
            value.absolute_singleton_fraction_maximum
        ),
        "reasons": list(value.reasons),
    }


def _uniforms_for_registered_stage(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    family: Family,
    stage: Stage,
    replicate: int,
    count: int,
) -> np.typing.NDArray[np.float64]:
    key = calibration_rng_key(
        master_seed=master_seed,
        scientific_version=scientific_version,
        purpose=CalibrationPurpose.CONDITIONED_BLOCK,
        artifact=artifact,
        variant=0,
        replicate=replicate,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=family,
        stage=stage,
    )
    return key.generator().random(count)


def _registered_artifact(artifact: Artifact) -> CalibrationArtifact:
    if artifact == "design":
        return CalibrationArtifact.DESIGN_OR_CONTROL
    if artifact == "production":
        return CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    raise ModelError("block RNG artifact must be design or production")


def _registered_family(frequency: Frequency) -> Family:
    if frequency == "monthly":
        return Family.LF
    if frequency == "daily":
        return Family.HF
    raise ModelError("block RNG frequency must be monthly or daily")


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value
