"""Streaming execution for the approved return-block calibration experiment.

This module joins the already-frozen Markov-state, conditioned-bootstrap, and
block-support primitives without owning an RNG or writing an artifact.  All
uniforms are supplied by the caller.  A uniform chunk is materialized once,
then reused for every unique intended block length before it is released.  The
result is deterministic for a fixed model, history, geometry, and uniform
stream, irrespective of the chosen chunk size.

The pooled block-length distributions are represented by exact integer
histograms.  This keeps peak working memory proportional to one history chunk,
not to 10,000 times the daily history length.  The public trace iterator also
lets the later dependence diagnostic consume source-index histories one chunk
at a time instead of retaining a multi-gigabyte tensor.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from prpg.errors import ModelError
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
    BlockPolicySelection,
    BlockSupportNumericalFailure,
    Frequency,
    NoSupportedBlockLength,
    select_supported_length,
    state_support,
)
from prpg.model.hmm import (
    PROBABILITY_ATOL,
    STATIONARY_RESIDUAL_ATOL,
    GaussianHMMFit,
    validate_markov_chain,
)

Artifact: TypeAlias = Literal["design", "production"]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

CANONICAL_HISTORY_COUNT = 10_000
DEFAULT_TRACE_CHUNK_SIZE = 512
MINIMUM_COMPLETED_EPISODES = 0
MINIMUM_EFFECTIVE_FRACTION = 0.50

_CANONICAL_MONTHLY_GEOMETRY: dict[Artifact, tuple[int, ...]] = {
    "design": (193,),
    "production": (210, 7),
}
_CANONICAL_DAILY_GEOMETRY: dict[Artifact, tuple[int, ...]] = {
    "design": (4_049,),
    "production": (4_404, 143),
}
_FREQUENCY_BOUNDS: dict[Frequency, tuple[int, int]] = {
    "monthly": MONTHLY_BLOCK_LENGTH_BOUNDS,
    "daily": DAILY_BLOCK_LENGTH_BOUNDS,
}


@dataclass(frozen=True, slots=True)
class BlockStateModel:
    """The fitted-model probabilities required by block calibration."""

    stationary_distribution: FloatArray
    transition_matrix: FloatArray

    def __post_init__(self) -> None:
        transition, stationary = _validated_model_probabilities(
            self.transition_matrix, self.stationary_distribution
        )
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "stationary_distribution", stationary)

    @classmethod
    def from_fitted_hmm(cls, fitted_model: GaussianHMMFit) -> BlockStateModel:
        """Extract the canonical fitted HMM transition law without refitting."""

        return cls(
            stationary_distribution=(
                fitted_model.chain_diagnostics.stationary_distribution
            ),
            transition_matrix=fitted_model.parameters.transition_matrix,
        )

    @property
    def n_states(self) -> int:
        return int(self.transition_matrix.shape[0])


@dataclass(frozen=True, slots=True)
class CalibrationGeometry:
    """Segment geometry for one artifact/frequency calibration cell."""

    artifact: Artifact
    frequency: Frequency
    monthly_segment_lengths: tuple[int, ...]
    target_segment_lengths: tuple[int, ...]
    strict_canonical: bool

    @property
    def monthly_observations(self) -> int:
        return sum(self.monthly_segment_lengths)

    @property
    def target_observations(self) -> int:
        return sum(self.target_segment_lengths)

    @property
    def target_continuation_links(self) -> BoolArray:
        return _segment_continuation_links(self.target_segment_lengths)


@dataclass(frozen=True, slots=True)
class RegisteredUniformMatrices:
    """Caller-owned registered draws for one complete calibration cell.

    State uniforms have one column per macro month.  Restart and source-rank
    uniforms have one column per target-frequency observation.  The matrices
    may be memory mapped; this module never mutates them.
    """

    monthly_state_uniforms: npt.ArrayLike
    restart_uniforms: npt.ArrayLike
    source_rank_uniforms: npt.ArrayLike


@dataclass(frozen=True, slots=True)
class RegisteredUniformChunk:
    """A contiguous history-row slice of the three registered matrices."""

    history_start: int
    monthly_state_uniforms: npt.ArrayLike
    restart_uniforms: npt.ArrayLike
    source_rank_uniforms: npt.ArrayLike


@dataclass(frozen=True, slots=True)
class PairedTraceChunk:
    """One intended-L trace chunk suitable for streaming diagnostics."""

    history_start: int
    intended_length: int
    target_states: IntArray
    source_indices: IntArray
    conditioned_restart_flags: BoolArray
    ideal_restart_flags: BoolArray
    target_continuation_links: BoolArray

    @property
    def history_count(self) -> int:
        return int(self.target_states.shape[0])


@dataclass(frozen=True, slots=True)
class PooledBlockStatistics:
    """Exact histogram summary of realized blocks pooled across histories."""

    state: int
    count: int
    mean: float
    minimum: int
    maximum: int
    median: float
    singleton_fraction: float
    quantile_05: float
    quantile_95: float
    length_histogram: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PooledFragmentationGate:
    """Selected-L gate over compact pooled conditioned/ideal histograms."""

    state: int
    intended_length: int
    conditioned: PooledBlockStatistics
    ideal: PooledBlockStatistics
    conditioned_mean_minimum: float
    conditioned_median_minimum: float
    singleton_fraction_maximum_from_ideal: float
    absolute_singleton_fraction_maximum: float
    reasons: tuple[str, ...]

    @property
    def conditioned_mean_passed(self) -> bool:
        return self.conditioned.mean >= self.conditioned_mean_minimum

    @property
    def conditioned_median_passed(self) -> bool:
        return self.conditioned.median >= self.conditioned_median_minimum

    @property
    def relative_singleton_passed(self) -> bool:
        return (
            self.conditioned.singleton_fraction
            <= self.singleton_fraction_maximum_from_ideal
        )

    @property
    def absolute_singleton_passed(self) -> bool:
        return (
            self.conditioned.singleton_fraction
            <= self.absolute_singleton_fraction_maximum
        )

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class LengthSimulationSummary:
    """Pooled paired results for one intended block length."""

    intended_length: int
    histories: int
    conditioned: tuple[PooledBlockStatistics, ...]
    ideal: tuple[PooledBlockStatistics, ...]

    @property
    def conditioned_means(self) -> dict[int, float]:
        return {item.state: item.mean for item in self.conditioned}


@dataclass(frozen=True, slots=True)
class BlockCalibrationExecution:
    """Complete in-memory decision for one return-block calibration cell."""

    geometry: CalibrationGeometry
    histories: int
    starting_length: int
    structurally_feasible_lengths: tuple[int, ...]
    simulations: tuple[LengthSimulationSummary, ...]
    policy_selection: BlockPolicySelection
    dependence_reuse_lengths: tuple[int, ...]
    selected_fragmentation_gates: tuple[PooledFragmentationGate, ...]

    @property
    def selected_length(self) -> int:
        return self.policy_selection.selected_length

    @property
    def fragmentation_passed(self) -> bool:
        return all(gate.passed for gate in self.selected_fragmentation_gates)

    @property
    def selected_simulation(self) -> LengthSimulationSummary:
        for item in self.simulations:
            if item.intended_length == self.selected_length:
                return item
        raise AssertionError("selected simulation is missing")


@dataclass(frozen=True, slots=True)
class BlockSupportFailureEvidence:
    """Complete code-derived evidence for one terminal block-cell failure."""

    geometry: CalibrationGeometry
    histories: int
    starting_length: int
    structurally_feasible_lengths: tuple[int, ...]
    simulations: tuple[LengthSimulationSummary, ...]
    failure_code: Literal[
        "historical_support_unattained",
        "effective_support_unattained",
        "effective_length_numerical_failure",
    ]
    failure_message: str
    failure_details: tuple[tuple[str, object], ...]


class BlockCalibrationSupportFailure(ModelError):
    """Code-owned terminal support failure with complete compact evidence."""

    def __init__(self, evidence: BlockSupportFailureEvidence) -> None:
        self.evidence = evidence
        super().__init__(
            evidence.failure_message,
            details=dict(evidence.failure_details),
        )


class BlockCalibrationNumericalFailure(ModelError):
    """Code-owned terminal block numerical failure with compact evidence."""

    def __init__(self, evidence: BlockSupportFailureEvidence) -> None:
        self.evidence = evidence
        super().__init__(
            evidence.failure_message,
            details=dict(evidence.failure_details),
        )


class _BlockHistogramPool:
    """Mutable exact accumulator kept private to one execution."""

    def __init__(self, *, n_states: int, maximum_length: int) -> None:
        self.conditioned = np.zeros((n_states, maximum_length + 1), dtype=np.int64)
        self.ideal = np.zeros((n_states, maximum_length + 1), dtype=np.int64)
        self.histories = 0

    def add(self, chunk: PairedTraceChunk, source_links: BoolArray) -> None:
        conditioned_boundaries = _conditioned_boundaries(chunk, source_links)
        self._add_boundaries(
            self.conditioned, chunk.target_states, conditioned_boundaries
        )
        self._add_boundaries(self.ideal, chunk.target_states, chunk.ideal_restart_flags)
        self.histories += chunk.history_count

    @staticmethod
    def _add_boundaries(
        histogram: IntArray, target_states: IntArray, boundaries: BoolArray
    ) -> None:
        observations = target_states.shape[1]
        for row in range(target_states.shape[0]):
            starts = np.flatnonzero(boundaries[row])
            stops = np.append(starts[1:], observations)
            lengths = stops - starts
            block_states = target_states[row, starts]
            for state in np.unique(block_states):
                state_lengths = lengths[block_states == state]
                histogram[int(state)] += np.bincount(
                    state_lengths, minlength=histogram.shape[1]
                )


def resolve_calibration_geometry(
    *,
    artifact: Artifact,
    frequency: Frequency,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
) -> CalibrationGeometry:
    """Resolve and validate exact or explicitly reduced test geometry."""

    if artifact not in _CANONICAL_MONTHLY_GEOMETRY:
        raise ModelError("block-calibration artifact must be design or production")
    if frequency not in _FREQUENCY_BOUNDS:
        raise ModelError("block-calibration frequency must be monthly or daily")
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict canonical mode must be Boolean")

    canonical_monthly = _CANONICAL_MONTHLY_GEOMETRY[artifact]
    canonical_daily = _CANONICAL_DAILY_GEOMETRY[artifact]
    monthly = _positive_lengths(
        (
            canonical_monthly
            if monthly_segment_lengths is None
            else monthly_segment_lengths
        ),
        label="monthly segment lengths",
    )
    if strict_canonical and monthly != canonical_monthly:
        raise ModelError(
            "strict canonical monthly geometry does not match the artifact",
            details={"actual": list(monthly), "expected": list(canonical_monthly)},
        )

    if frequency == "monthly":
        if daily_segment_lengths is not None:
            raise ModelError("monthly calibration cannot accept daily segment lengths")
        target = monthly
    else:
        target = _positive_lengths(
            canonical_daily if daily_segment_lengths is None else daily_segment_lengths,
            label="daily segment lengths",
        )
        _validate_daily_expansion(monthly, target)
        if strict_canonical and target != canonical_daily:
            raise ModelError(
                "strict canonical daily geometry does not match the artifact",
                details={"actual": list(target), "expected": list(canonical_daily)},
            )
    return CalibrationGeometry(
        artifact=artifact,
        frequency=frequency,
        monthly_segment_lengths=monthly,
        target_segment_lengths=target,
        strict_canonical=strict_canonical,
    )


def uniform_chunks_from_matrices(
    matrices: RegisteredUniformMatrices,
    *,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
) -> Iterator[RegisteredUniformChunk]:
    """Yield non-copying row views from caller-owned uniform matrices."""

    size = _positive_integer(chunk_size, "uniform chunk size")
    state = _raw_matrix(matrices.monthly_state_uniforms, "monthly state uniforms")
    restart = _raw_matrix(matrices.restart_uniforms, "restart uniforms")
    rank = _raw_matrix(matrices.source_rank_uniforms, "source-rank uniforms")
    rows = state.shape[0]
    if restart.shape[0] != rows or rank.shape[0] != rows:
        raise ModelError(
            "registered uniform matrices must have the same history count",
            details={
                "state": int(rows),
                "restart": int(restart.shape[0]),
                "source_rank": int(rank.shape[0]),
            },
        )
    for start in range(0, rows, size):
        stop = min(rows, start + size)
        yield RegisteredUniformChunk(
            history_start=start,
            monthly_state_uniforms=state[start:stop],
            restart_uniforms=restart[start:stop],
            source_rank_uniforms=rank[start:stop],
        )


def iter_paired_trace_chunks(
    *,
    model: BlockStateModel | GaussianHMMFit,
    geometry: CalibrationGeometry,
    historical_source_states: npt.ArrayLike,
    source_continuation_links: npt.ArrayLike,
    intended_lengths: Sequence[int],
    uniform_chunks: Iterable[RegisteredUniformChunk],
    expected_history_count: int = CANONICAL_HISTORY_COUNT,
) -> Iterator[PairedTraceChunk]:
    """Stream exact conditioned/ideal traces while reusing every draw chunk.

    Output is ordered first by input chunk and then by the caller's unique
    intended-length order.  Consequently a downstream dependence accumulator
    can consume all requested L values with peak memory bounded by one output
    chunk.  The iterator validates contiguous history numbering and exact
    total count before normal exhaustion.
    """

    state_model = _coerce_state_model(model)
    source, source_links = _validated_source_history(
        historical_source_states,
        source_continuation_links,
        n_states=state_model.n_states,
    )
    _validate_strict_source_geometry(geometry, source, source_links)
    lengths = _validated_intended_lengths(
        intended_lengths, frequency=geometry.frequency
    )
    expected = _positive_integer(expected_history_count, "expected history count")
    target_links = geometry.target_continuation_links
    eligible = tuple(
        np.flatnonzero(source == state).astype(np.int64, copy=False)
        for state in range(state_model.n_states)
    )

    next_history = 0
    for raw_chunk in uniform_chunks:
        if raw_chunk.history_start != next_history:
            raise ModelError(
                "uniform chunks must be contiguous and zero based",
                details={
                    "actual_history_start": raw_chunk.history_start,
                    "expected_history_start": next_history,
                },
            )
        state_uniforms = _uniform_matrix(
            raw_chunk.monthly_state_uniforms,
            columns=geometry.monthly_observations,
            label="monthly state uniforms",
        )
        restart_uniforms = _uniform_matrix(
            raw_chunk.restart_uniforms,
            columns=geometry.target_observations,
            label="restart uniforms",
        )
        rank_uniforms = _uniform_matrix(
            raw_chunk.source_rank_uniforms,
            columns=geometry.target_observations,
            label="source-rank uniforms",
        )
        rows = state_uniforms.shape[0]
        if rows == 0:
            raise ModelError("uniform chunks must contain at least one history")
        if restart_uniforms.shape[0] != rows or rank_uniforms.shape[0] != rows:
            raise ModelError(
                "all matrices in a uniform chunk must have the same row count"
            )
        if next_history + rows > expected:
            raise ModelError("uniform chunks exceed the expected history count")

        monthly_states = _simulate_segmented_state_matrix(
            state_model, state_uniforms, geometry.monthly_segment_lengths
        )
        target_states = _expand_target_states(monthly_states, geometry)
        for intended_length in lengths:
            source_indices, conditioned_restarts = _conditioned_trace_matrix(
                target_states=target_states,
                source_states=source,
                source_links=source_links,
                target_links=target_links,
                restart_uniforms=restart_uniforms,
                rank_uniforms=rank_uniforms,
                restart_probability=1.0 / intended_length,
                eligible=eligible,
            )
            ideal_restarts = _ideal_restart_matrix(
                target_states,
                target_links,
                restart_uniforms,
                restart_probability=1.0 / intended_length,
            )
            for array in (
                target_states,
                source_indices,
                conditioned_restarts,
                ideal_restarts,
                target_links,
            ):
                array.setflags(write=False)
            yield PairedTraceChunk(
                history_start=next_history,
                intended_length=intended_length,
                target_states=target_states,
                source_indices=source_indices,
                conditioned_restart_flags=conditioned_restarts,
                ideal_restart_flags=ideal_restarts,
                target_continuation_links=target_links,
            )
        next_history += rows
    if next_history != expected:
        raise ModelError(
            "uniform chunks do not contain the expected number of histories",
            details={"actual": next_history, "expected": expected},
        )


def execute_block_calibration(
    *,
    model: BlockStateModel | GaussianHMMFit,
    artifact: Artifact,
    frequency: Frequency,
    starting_length: int,
    historical_source_states: npt.ArrayLike,
    source_continuation_links: npt.ArrayLike,
    uniform_chunks: Iterable[RegisteredUniformChunk],
    history_count: int = CANONICAL_HISTORY_COUNT,
    strict_canonical: bool = True,
    monthly_segment_lengths: Sequence[int] | None = None,
    daily_segment_lengths: Sequence[int] | None = None,
    trace_consumer: Callable[[PairedTraceChunk], None] | None = None,
) -> BlockCalibrationExecution:
    """Run support search, paired simulations, and selected-L hard gates.

    In strict canonical mode the geometry and exactly 10,000 histories are
    mandatory.  Explicit smaller values are accepted only with
    ``strict_canonical=False`` for tests and non-canonical diagnostics.
    Structural support is evaluated before simulation; all unique supported
    lengths from two through ``starting_length + 1`` are then evaluated in one
    pass over the injected uniform stream.  Including the extra upper neighbor
    ensures the selected L's complete ``{L-1,L,L+1}`` sensitivity set is
    available whenever historically supported.  An optional deterministic
    ``trace_consumer`` receives each immutable source-index chunk during this
    sole pass, allowing dependence diagnostics to reduce the common histories
    without retaining them in memory.
    """

    histories = _positive_integer(history_count, "history count")
    if trace_consumer is not None and not callable(trace_consumer):
        raise ModelError("trace consumer must be callable")
    if strict_canonical and histories != CANONICAL_HISTORY_COUNT:
        raise ModelError(
            "strict canonical block calibration requires exactly 10,000 histories",
            details={"actual": histories, "expected": CANONICAL_HISTORY_COUNT},
        )
    geometry = resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=monthly_segment_lengths,
        daily_segment_lengths=daily_segment_lengths,
    )
    state_model = _coerce_state_model(model)
    source, source_links = _validated_source_history(
        historical_source_states,
        source_continuation_links,
        n_states=state_model.n_states,
    )
    _validate_strict_source_geometry(geometry, source, source_links)
    lower, upper = _FREQUENCY_BOUNDS[frequency]
    start = _positive_integer(starting_length, "starting block length")
    if not lower <= start <= upper:
        raise ModelError(
            "starting block length is outside the approved frequency bounds",
            details={"frequency": frequency, "starting_length": start},
        )

    simulation_lengths: list[int] = []
    for intended in range(min(upper, start + 1), lower - 1, -1):
        support = state_support(
            source,
            source_links,
            intended_length=intended,
            selected_states=tuple(range(state_model.n_states)),
        )
        required = _required_eligible_starts(frequency, intended)
        if all(
            item.completed_episodes >= MINIMUM_COMPLETED_EPISODES
            and item.eligible_start_count >= required
            for item in support
        ):
            simulation_lengths.append(intended)
    if not simulation_lengths:
        raise BlockCalibrationSupportFailure(
            BlockSupportFailureEvidence(
                geometry=geometry,
                histories=histories,
                starting_length=start,
                structurally_feasible_lengths=(),
                simulations=(),
                failure_code="historical_support_unattained",
                failure_message=(
                    "no intended length has the required episode and "
                    "source-start support"
                ),
                failure_details=(
                    ("frequency", frequency),
                    ("starting_length", start),
                ),
            )
        )

    pools = {
        intended: _BlockHistogramPool(
            n_states=state_model.n_states,
            maximum_length=geometry.target_observations,
        )
        for intended in simulation_lengths
    }
    traces = iter_paired_trace_chunks(
        model=state_model,
        geometry=geometry,
        historical_source_states=source,
        source_continuation_links=source_links,
        intended_lengths=simulation_lengths,
        uniform_chunks=uniform_chunks,
        expected_history_count=histories,
    )
    for trace in traces:
        pools[trace.intended_length].add(trace, source_links)
        if trace_consumer is not None:
            trace_consumer(trace)

    summaries: list[LengthSimulationSummary] = []
    realized_means: dict[int, dict[int, float]] = {}
    for intended in simulation_lengths:
        pool = pools[intended]
        if pool.histories != histories:
            raise AssertionError("trace history accounting is inconsistent")
        conditioned = _summarize_histograms(pool.conditioned)
        ideal = _summarize_histograms(pool.ideal)
        summary = LengthSimulationSummary(
            intended_length=intended,
            histories=histories,
            conditioned=conditioned,
            ideal=ideal,
        )
        summaries.append(summary)
        realized_means[intended] = summary.conditioned_means

    try:
        selection = select_supported_length(
            starting_length=start,
            frequency=frequency,
            states=source,
            continuation=source_links,
            realized_means_by_length=realized_means,
            selected_states=tuple(range(state_model.n_states)),
            minimum_completed_episodes=MINIMUM_COMPLETED_EPISODES,
            minimum_effective_fraction=MINIMUM_EFFECTIVE_FRACTION,
        )
    except NoSupportedBlockLength as error:
        raise BlockCalibrationSupportFailure(
            BlockSupportFailureEvidence(
                geometry=geometry,
                histories=histories,
                starting_length=start,
                structurally_feasible_lengths=tuple(simulation_lengths),
                simulations=tuple(summaries),
                failure_code="effective_support_unattained",
                failure_message=error.message,
                failure_details=_frozen_failure_details(error.details),
            )
        ) from error
    except BlockSupportNumericalFailure as error:
        raise BlockCalibrationNumericalFailure(
            BlockSupportFailureEvidence(
                geometry=geometry,
                histories=histories,
                starting_length=start,
                structurally_feasible_lengths=tuple(simulation_lengths),
                simulations=tuple(summaries),
                failure_code="effective_length_numerical_failure",
                failure_message=error.message,
                failure_details=_frozen_failure_details(error.details),
            )
        ) from error
    selected_summary = next(
        item for item in summaries if item.intended_length == selection.selected_length
    )
    gates = pooled_fragmentation_gates(
        selected_summary.conditioned,
        selected_summary.ideal,
        intended_length=selection.selected_length,
    )
    available = {item.intended_length for item in summaries}
    reuse = tuple(
        candidate
        for candidate in (
            selection.selected_length - 1,
            selection.selected_length,
            selection.selected_length + 1,
        )
        if candidate in available
    )
    return BlockCalibrationExecution(
        geometry=geometry,
        histories=histories,
        starting_length=start,
        structurally_feasible_lengths=tuple(simulation_lengths),
        simulations=tuple(summaries),
        policy_selection=selection,
        dependence_reuse_lengths=reuse,
        selected_fragmentation_gates=gates,
    )


def _coerce_state_model(
    model: BlockStateModel | GaussianHMMFit,
) -> BlockStateModel:
    if isinstance(model, BlockStateModel):
        return model
    if not isinstance(model, GaussianHMMFit):
        raise ModelError("block calibration requires a fitted HMM or state model")
    return BlockStateModel.from_fitted_hmm(model)


def _validated_model_probabilities(
    transition_matrix: npt.ArrayLike, stationary_distribution: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    try:
        transition = np.asarray(transition_matrix, dtype=np.float64)
        stationary = np.asarray(stationary_distribution, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("block-state model probabilities must be numeric") from error
    chain = validate_markov_chain(transition)
    if stationary.ndim != 1 or stationary.size != transition.shape[0]:
        raise ModelError("stationary distribution has incompatible shape")
    if not np.isfinite(stationary).all() or bool(
        ((stationary < 0.0) | (stationary > 1.0)).any()
    ):
        raise ModelError("stationary distribution must be finite and in [0, 1]")
    if abs(float(stationary.sum()) - 1.0) > PROBABILITY_ATOL:
        raise ModelError("stationary distribution must sum to one")
    residual = float(np.max(np.abs(stationary @ transition - stationary)))
    if residual > STATIONARY_RESIDUAL_ATOL:
        raise ModelError(
            "stationary distribution is inconsistent with the transition matrix",
            details={"residual": residual},
        )
    if (
        float(np.max(np.abs(stationary - chain.stationary_distribution)))
        > PROBABILITY_ATOL
    ):
        raise ModelError(
            "stationary distribution differs from the validated Markov chain"
        )
    transition = np.array(transition, dtype=np.float64, copy=True)
    stationary = np.array(stationary, dtype=np.float64, copy=True)
    transition.setflags(write=False)
    stationary.setflags(write=False)
    return transition, stationary


def _validated_source_history(
    states: npt.ArrayLike,
    continuation_links: npt.ArrayLike,
    *,
    n_states: int,
) -> tuple[IntArray, BoolArray]:
    raw_states = np.asarray(states)
    if (
        raw_states.ndim != 1
        or raw_states.size == 0
        or raw_states.dtype.kind
        not in {
            "i",
            "u",
        }
    ):
        raise ModelError("historical source states must be a non-empty integer vector")
    result = raw_states.astype(np.int64, copy=True)
    if bool(((result < 0) | (result >= n_states)).any()):
        raise ModelError(
            "historical source states contain an invalid model-state label"
        )
    missing = [state for state in range(n_states) if not bool(np.any(result == state))]
    if missing:
        raise ModelError(
            "historical source states omit fitted model states",
            details={"missing_states": missing},
        )
    raw_links = np.asarray(continuation_links)
    if (
        raw_links.ndim != 1
        or raw_links.size != result.size - 1
        or raw_links.dtype.kind != "b"
    ):
        raise ModelError(
            "source continuation links must be Boolean with history length minus one"
        )
    links = raw_links.astype(np.bool_, copy=True)
    result.setflags(write=False)
    links.setflags(write=False)
    return result, links


def _validate_strict_source_geometry(
    geometry: CalibrationGeometry, states: IntArray, links: BoolArray
) -> None:
    if not geometry.strict_canonical:
        return
    expected_links = geometry.target_continuation_links
    if states.size != geometry.target_observations:
        raise ModelError(
            "strict canonical source history has the wrong exact geometry",
            details={
                "actual_observations": int(states.size),
                "expected_observations": geometry.target_observations,
            },
        )
    if not np.array_equal(links, expected_links):
        raise ModelError(
            "strict canonical source continuation mask has the wrong segments",
            details={"expected_segment_lengths": list(geometry.target_segment_lengths)},
        )


def _simulate_segmented_state_matrix(
    model: BlockStateModel,
    uniforms: FloatArray,
    segment_lengths: tuple[int, ...],
) -> IntArray:
    rows = uniforms.shape[0]
    states = np.empty(uniforms.shape, dtype=np.int64)
    transition_cdf = np.cumsum(model.transition_matrix, axis=1)
    stationary_cdf = np.cumsum(model.stationary_distribution)
    offset = 0
    for length in segment_lengths:
        states[:, offset] = _categorical_columns(
            uniforms[:, offset], np.broadcast_to(stationary_cdf, (rows, model.n_states))
        )
        for position in range(offset + 1, offset + length):
            probabilities = transition_cdf[states[:, position - 1]]
            states[:, position] = _categorical_columns(
                uniforms[:, position], probabilities
            )
        offset += length
    return states


def _categorical_columns(uniforms: FloatArray, cumulative: FloatArray) -> IntArray:
    selected = np.asarray(
        np.sum(uniforms[:, None] >= cumulative, axis=1, dtype=np.int64),
        dtype=np.int64,
    )
    clipped: IntArray = np.asarray(
        np.minimum(selected, cumulative.shape[1] - 1), dtype=np.int64
    )
    return clipped


def _expand_target_states(
    monthly_states: IntArray, geometry: CalibrationGeometry
) -> IntArray:
    if geometry.frequency == "monthly":
        return monthly_states
    indices: list[IntArray] = []
    offset = 0
    for monthly_length, daily_length in zip(
        geometry.monthly_segment_lengths,
        geometry.target_segment_lengths,
        strict=True,
    ):
        local = np.arange(daily_length, dtype=np.int64) // 21
        indices.append(local + offset)
        offset += monthly_length
    expansion = np.concatenate(indices)
    return np.asarray(monthly_states[:, expansion], dtype=np.int64)


def _conditioned_trace_matrix(
    *,
    target_states: IntArray,
    source_states: IntArray,
    source_links: BoolArray,
    target_links: BoolArray,
    restart_uniforms: FloatArray,
    rank_uniforms: FloatArray,
    restart_probability: float,
    eligible: tuple[IntArray, ...],
) -> tuple[IntArray, BoolArray]:
    rows, observations = target_states.shape
    indices = np.empty((rows, observations), dtype=np.int64)
    restarts = np.zeros((rows, observations), dtype=np.bool_)
    restarts[:, 0] = True
    _assign_ranked_sources(
        indices[:, 0],
        target_states[:, 0],
        rank_uniforms[:, 0],
        np.ones(rows, dtype=np.bool_),
        eligible,
    )
    source_link_with_terminal = np.append(source_links, False)
    for position in range(1, observations):
        previous = indices[:, position - 1]
        next_source = previous + 1
        target_forced = (not bool(target_links[position - 1])) | (
            target_states[:, position] != target_states[:, position - 1]
        )
        source_end = next_source >= source_states.size
        safe_next = np.minimum(next_source, source_states.size - 1)
        source_gap = ~source_link_with_terminal[previous]
        source_mismatch = source_states[safe_next] != target_states[:, position]
        forced = target_forced | source_end | source_gap | source_mismatch
        random_restart = (~forced) & (
            restart_uniforms[:, position] < restart_probability
        )
        restarted = forced | random_restart
        restarts[:, position] = restarted
        indices[:, position] = safe_next
        _assign_ranked_sources(
            indices[:, position],
            target_states[:, position],
            rank_uniforms[:, position],
            restarted,
            eligible,
        )
    return indices, restarts


def _assign_ranked_sources(
    output: IntArray,
    target_states: IntArray,
    rank_uniforms: FloatArray,
    selected_rows: BoolArray,
    eligible: tuple[IntArray, ...],
) -> None:
    for state, candidates in enumerate(eligible):
        mask = selected_rows & (target_states == state)
        if not bool(np.any(mask)):
            continue
        ranks = np.floor(rank_uniforms[mask] * candidates.size).astype(np.int64)
        output[mask] = candidates[ranks]


def _ideal_restart_matrix(
    target_states: IntArray,
    target_links: BoolArray,
    restart_uniforms: FloatArray,
    *,
    restart_probability: float,
) -> BoolArray:
    result = np.zeros(target_states.shape, dtype=np.bool_)
    result[:, 0] = True
    for position in range(1, target_states.shape[1]):
        forced = (not bool(target_links[position - 1])) | (
            target_states[:, position] != target_states[:, position - 1]
        )
        result[:, position] = forced | (
            (~forced) & (restart_uniforms[:, position] < restart_probability)
        )
    return result


def _conditioned_boundaries(
    chunk: PairedTraceChunk, source_links: BoolArray
) -> BoolArray:
    target = chunk.target_states
    source = chunk.source_indices
    result = np.zeros(target.shape, dtype=np.bool_)
    result[:, 0] = True
    previous = source[:, :-1]
    adjacent = source[:, 1:] == previous + 1
    valid_source_link = np.append(source_links, False)[previous]
    target_boundary = (~chunk.target_continuation_links)[None, :] | (
        target[:, 1:] != target[:, :-1]
    )
    result[:, 1:] = target_boundary | ~adjacent | ~valid_source_link
    return result


def _summarize_histograms(histograms: IntArray) -> tuple[PooledBlockStatistics, ...]:
    result: list[PooledBlockStatistics] = []
    for state in range(histograms.shape[0]):
        counts = histograms[state]
        count = int(np.sum(counts))
        if count <= 0:
            raise ModelError(
                "pooled block histories contain no realized block for a model state",
                details={"state": state},
            )
        support = np.flatnonzero(counts)
        lengths = np.arange(counts.size, dtype=np.int64)
        result.append(
            PooledBlockStatistics(
                state=state,
                count=count,
                mean=float(np.dot(lengths, counts) / count),
                minimum=int(support[0]),
                maximum=int(support[-1]),
                median=float(_higher_histogram_quantile(counts, 0.50)),
                singleton_fraction=float(counts[1] / count),
                quantile_05=float(_higher_histogram_quantile(counts, 0.05)),
                quantile_95=float(_higher_histogram_quantile(counts, 0.95)),
                length_histogram=tuple(int(value) for value in counts),
            )
        )
    return tuple(result)


def pooled_fragmentation_gates(
    conditioned: Sequence[PooledBlockStatistics],
    ideal: Sequence[PooledBlockStatistics],
    *,
    intended_length: int,
) -> tuple[PooledFragmentationGate, ...]:
    """Apply all four frozen fragmentation gates to exact pooled histograms."""

    intended = _positive_integer(intended_length, "intended block length")
    if intended < 2:
        raise ModelError("fragmentation gates require intended length at least two")
    conditioned_by_state = _validated_pooled_by_state(
        conditioned, label="conditioned pooled statistics"
    )
    ideal_by_state = _validated_pooled_by_state(ideal, label="ideal pooled statistics")
    if set(conditioned_by_state) != set(ideal_by_state):
        raise ModelError(
            "pooled conditioned and ideal statistics require matching states"
        )
    result: list[PooledFragmentationGate] = []
    for state in sorted(conditioned_by_state):
        observed = conditioned_by_state[state]
        counterfactual = ideal_by_state[state]
        mean_minimum = 0.50 * intended
        median_minimum = 0.50 * counterfactual.median
        relative_singleton_maximum = counterfactual.singleton_fraction + 0.10
        absolute_singleton_maximum = 0.75
        reasons: list[str] = []
        if observed.mean < mean_minimum:
            reasons.append("conditioned_mean_below_half_intended_length")
        if observed.median < median_minimum:
            reasons.append("conditioned_median_below_half_ideal_median")
        if observed.singleton_fraction > relative_singleton_maximum:
            reasons.append("conditioned_singleton_exceeds_ideal_plus_0.10")
        if observed.singleton_fraction > absolute_singleton_maximum:
            reasons.append("conditioned_singleton_exceeds_0.75")
        result.append(
            PooledFragmentationGate(
                state=state,
                intended_length=intended,
                conditioned=observed,
                ideal=counterfactual,
                conditioned_mean_minimum=mean_minimum,
                conditioned_median_minimum=median_minimum,
                singleton_fraction_maximum_from_ideal=(relative_singleton_maximum),
                absolute_singleton_fraction_maximum=absolute_singleton_maximum,
                reasons=tuple(reasons),
            )
        )
    return tuple(result)


def _validated_pooled_by_state(
    values: Sequence[PooledBlockStatistics], *, label: str
) -> dict[int, PooledBlockStatistics]:
    result: dict[int, PooledBlockStatistics] = {}
    for item in values:
        if (
            isinstance(item.state, bool)
            or not isinstance(item.state, int | np.integer)
            or item.state < 0
            or item.state in result
        ):
            raise ModelError(f"{label} contain invalid or duplicate states")
        histogram_raw = np.asarray(item.length_histogram)
        if (
            histogram_raw.ndim != 1
            or histogram_raw.size < 2
            or histogram_raw.dtype.kind not in {"i", "u"}
            or bool((histogram_raw < 0).any())
            or int(histogram_raw[0]) != 0
        ):
            raise ModelError(f"{label} contain an invalid length histogram")
        histogram = histogram_raw.astype(np.int64, copy=False)
        support = np.flatnonzero(histogram)
        if support.size == 0:
            raise ModelError(f"{label} contain an empty length histogram")
        count = int(np.sum(histogram))
        lengths = np.arange(histogram.size, dtype=np.int64)
        expected = (
            float(np.dot(lengths, histogram) / count),
            float(_higher_histogram_quantile(histogram, 0.50)),
            float(histogram[1] / count),
            float(_higher_histogram_quantile(histogram, 0.05)),
            float(_higher_histogram_quantile(histogram, 0.95)),
        )
        actual = (
            item.mean,
            item.median,
            item.singleton_fraction,
            item.quantile_05,
            item.quantile_95,
        )
        if (
            item.count != count
            or item.minimum != int(support[0])
            or item.maximum != int(support[-1])
            or not all(math.isfinite(value) for value in actual)
            or not 0.0 <= item.singleton_fraction <= 1.0
            or any(
                not math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12)
                for observed, target in zip(actual, expected, strict=True)
            )
        ):
            raise ModelError(f"{label} are inconsistent with their histogram")
        result[int(item.state)] = item
    if not result:
        raise ModelError(f"{label} must be non-empty")
    return result


def _higher_histogram_quantile(counts: IntArray, probability: float) -> int:
    total = int(np.sum(counts))
    rank = math.ceil(probability * (total - 1))
    return int(np.searchsorted(np.cumsum(counts), rank, side="right"))


def _required_eligible_starts(frequency: Frequency, intended_length: int) -> int:
    if frequency == "monthly":
        return max(8, intended_length)
    return max(50, 2 * intended_length)


def _validated_intended_lengths(
    values: Sequence[int], *, frequency: Frequency
) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ModelError("intended lengths must be a non-empty sequence") from error
    if not raw:
        raise ModelError("intended lengths must be a non-empty sequence")
    if any(
        isinstance(value, bool) or not isinstance(value, int | np.integer)
        for value in raw
    ):
        raise ModelError("intended lengths must contain integers")
    result = tuple(int(value) for value in raw)
    if len(set(result)) != len(result):
        raise ModelError("intended lengths must be unique")
    lower, upper = _FREQUENCY_BOUNDS[frequency]
    if any(not lower <= value <= upper for value in result):
        raise ModelError("intended length is outside the approved frequency bounds")
    return result


def _uniform_matrix(values: npt.ArrayLike, *, columns: int, label: str) -> FloatArray:
    raw = _raw_matrix(values, label)
    if raw.shape[1] != columns:
        raise ModelError(
            f"{label} has the wrong observation count",
            details={"actual": int(raw.shape[1]), "expected": columns},
        )
    if not np.isfinite(raw).all() or bool(((raw < 0.0) | (raw >= 1.0)).any()):
        raise ModelError(f"{label} must be finite and in [0, 1)")
    return np.asarray(raw, dtype=np.float64)


def _raw_matrix(values: npt.ArrayLike, label: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ModelError(f"{label} must be a two-dimensional matrix")
    if raw.dtype.kind == "b":
        raise ModelError(f"{label} must be numeric, not Boolean")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error


def _positive_lengths(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ModelError(f"{label} must be a sequence of positive integers") from error
    if not raw or any(
        isinstance(value, bool)
        or not isinstance(value, int | np.integer)
        or int(value) <= 0
        for value in raw
    ):
        raise ModelError(f"{label} must be a sequence of positive integers")
    return tuple(int(value) for value in raw)


def _validate_daily_expansion(
    monthly_lengths: tuple[int, ...], daily_lengths: tuple[int, ...]
) -> None:
    if len(monthly_lengths) != len(daily_lengths):
        raise ModelError("monthly and daily geometry must have the same segments")
    for segment, (monthly, daily) in enumerate(
        zip(monthly_lengths, daily_lengths, strict=True)
    ):
        minimum = (monthly - 1) * 21 + 1
        maximum = monthly * 21
        if not minimum <= daily <= maximum:
            raise ModelError(
                "daily geometry is not a terminal 21-session expansion",
                details={
                    "segment": segment,
                    "daily": daily,
                    "allowed": [minimum, maximum],
                },
            )


def _segment_continuation_links(lengths: tuple[int, ...]) -> BoolArray:
    result = np.ones(max(0, sum(lengths) - 1), dtype=np.bool_)
    offset = 0
    for length in lengths[:-1]:
        offset += length
        result[offset - 1] = False
    result.setflags(write=False)
    return result


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ModelError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return result


def _frozen_failure_details(
    values: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, _frozen_metadata(value)) for key, value in sorted(values.items())
    )


def _frozen_metadata(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _frozen_metadata(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list | tuple):
        return tuple(_frozen_metadata(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ModelError("block failure details contain an unsupported value")
