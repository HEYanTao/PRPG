"""Bounded-memory reduction of block traces into dependence-gate cells.

The block executor owns source-index simulation; this module only consumes its
immutable chunks.  For each explicitly requested selected/neighbor block
length, it fills exact replicate rows in a preallocated transformed endpoint
matrix.  Histories may arrive out of order, but overlaps, omissions, wrong
geometry, and unrequested lengths fail closed.

Endpoint evaluation is vectorized across every history in a chunk while
preserving the frozen PC1 transform, endpoint order, gap-aware available-pair
ACFs, and Fisher transformation implemented by :mod:`prpg.model.dependence`.
No RNG, trace replay, file I/O, or adaptive reference fitting occurs here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from prpg.errors import ModelError
from prpg.model.block_execution import (
    CANONICAL_HISTORY_COUNT,
    Artifact,
    CalibrationGeometry,
    PairedTraceChunk,
    resolve_calibration_geometry,
)
from prpg.model.dependence import (
    CONSTANT_TOLERANCE,
    CORRELATION_CLIP,
    DAILY_LAGS,
    DIAGNOSTIC_NAMES,
    MONTHLY_LAGS,
    PAIR_NAMES,
    DependenceCell,
    DependenceReference,
    Frequency,
    dependence_max_statistic_cell,
    evaluate_dependence_vector,
    feasible_sensitivity_lengths,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DependenceLengthExecution:
    """One requested L's complete transformed matrix and max-stat cell."""

    intended_length: int
    transformed_replicates: FloatArray
    cell: DependenceCell


@dataclass(frozen=True, slots=True)
class DependenceExecution:
    """Finalized results for one artifact/frequency trace stream."""

    geometry: CalibrationGeometry
    selected_length: int
    requested_lengths: tuple[int, ...]
    histories: int
    labels: tuple[str, ...]
    results: tuple[DependenceLengthExecution, ...]

    def result_for_length(self, intended_length: int) -> DependenceLengthExecution:
        """Return one explicitly requested result or fail closed."""

        for result in self.results:
            if result.intended_length == intended_length:
                return result
        raise ModelError(
            "dependence execution does not contain the requested block length",
            details={"intended_length": intended_length},
        )


class DependenceTraceReducer:
    """Stateful, deterministic consumer for immutable block-trace chunks.

    The only retained simulation state is one ``histories x endpoint`` matrix
    and one fill bitmap per requested L.  Source-index paths and return tensors
    are released after each ``consume`` call.
    """

    def __init__(
        self,
        *,
        artifact: Artifact,
        selected_length: int,
        requested_lengths: Sequence[int],
        historical_returns: npt.ArrayLike,
        reference: DependenceReference,
        history_count: int = CANONICAL_HISTORY_COUNT,
        strict_canonical: bool = True,
        monthly_segment_lengths: Sequence[int] | None = None,
        daily_segment_lengths: Sequence[int] | None = None,
    ) -> None:
        histories = _positive_integer(history_count, "history count")
        if not isinstance(strict_canonical, bool):
            raise ModelError("strict canonical mode must be Boolean")
        if strict_canonical and histories != CANONICAL_HISTORY_COUNT:
            raise ModelError(
                "strict canonical dependence execution requires exactly "
                "10,000 histories",
                details={"actual": histories, "expected": CANONICAL_HISTORY_COUNT},
            )
        frequency = _reference_frequency(reference)
        geometry = resolve_calibration_geometry(
            artifact=artifact,
            frequency=frequency,
            strict_canonical=strict_canonical,
            monthly_segment_lengths=monthly_segment_lengths,
            daily_segment_lengths=daily_segment_lengths,
        )
        selected = _positive_integer(selected_length, "selected block length")
        requested = _requested_lengths(
            requested_lengths,
            selected_length=selected,
            frequency=frequency,
            require_all_feasible=strict_canonical,
        )
        source_returns = _historical_returns(
            historical_returns, expected_rows=geometry.target_observations
        )
        _validate_reference(reference, source_returns, geometry)
        components = len(reference.labels)

        self._geometry = geometry
        self._selected_length = selected
        self._requested_lengths = requested
        self._historical_returns = source_returns
        self._reference = reference
        self._histories = histories
        self._replicates = {
            intended: np.empty((histories, components), dtype=np.float64)
            for intended in requested
        }
        self._filled = {
            intended: np.zeros(histories, dtype=np.bool_) for intended in requested
        }
        self._finalized: DependenceExecution | None = None

    @property
    def geometry(self) -> CalibrationGeometry:
        return self._geometry

    @property
    def requested_lengths(self) -> tuple[int, ...]:
        return self._requested_lengths

    @property
    def histories_filled(self) -> dict[int, int]:
        """Return defensive fill counts for audit/progress reporting."""

        return {
            intended: int(np.count_nonzero(self._filled[intended]))
            for intended in self._requested_lengths
        }

    def __call__(self, chunk: PairedTraceChunk) -> None:
        self.consume(chunk)

    def consume(self, chunk: PairedTraceChunk) -> None:
        """Validate and reduce one exact history slice transactionally."""

        if self._finalized is not None:
            raise ModelError("finalized dependence reducer cannot consume more traces")
        start, stop = _validate_trace_chunk(
            chunk,
            geometry=self._geometry,
            requested_lengths=self._requested_lengths,
            history_count=self._histories,
            source_rows=self._historical_returns.shape[0],
        )
        filled = self._filled[chunk.intended_length]
        if bool(filled[start:stop].any()):
            overlap = np.flatnonzero(filled[start:stop]) + start
            raise ModelError(
                "dependence trace histories overlap previously filled rows",
                details={
                    "intended_length": chunk.intended_length,
                    "first_overlapping_history": int(overlap[0]),
                    "overlap_count": int(overlap.size),
                },
            )
        simulated_returns = self._historical_returns[chunk.source_indices]
        transformed = _vectorized_dependence_values(
            simulated_returns,
            chunk.target_continuation_links,
            self._reference,
        )
        expected_shape = (stop - start, len(self._reference.labels))
        if transformed.shape != expected_shape:
            raise AssertionError("vectorized dependence result has an invalid shape")
        self._replicates[chunk.intended_length][start:stop] = transformed
        filled[start:stop] = True

    def finalize(self) -> DependenceExecution:
        """Require complete rows, calculate every cell, and freeze outputs."""

        if self._finalized is not None:
            return self._finalized
        for intended in self._requested_lengths:
            missing = np.flatnonzero(~self._filled[intended])
            if missing.size:
                raise ModelError(
                    "dependence trace stream has missing history rows",
                    details={
                        "intended_length": intended,
                        "first_missing_history": int(missing[0]),
                        "missing_count": int(missing.size),
                    },
                )
        results: list[DependenceLengthExecution] = []
        for intended in self._requested_lengths:
            replicates = self._replicates[intended]
            if not np.isfinite(replicates).all():
                raise AssertionError(
                    "filled dependence matrix contains non-finite values"
                )
            cell = dependence_max_statistic_cell(
                self._reference.transformed_values,
                replicates,
                expected_repetitions=self._histories,
            )
            replicates.setflags(write=False)
            results.append(DependenceLengthExecution(intended, replicates, cell))
        execution = DependenceExecution(
            geometry=self._geometry,
            selected_length=self._selected_length,
            requested_lengths=self._requested_lengths,
            histories=self._histories,
            labels=self._reference.labels,
            results=tuple(results),
        )
        self._finalized = execution
        return execution


def _vectorized_dependence_values(
    returns: FloatArray,
    continuation_links: BoolArray,
    reference: DependenceReference,
) -> FloatArray:
    histories, observations, assets = returns.shape
    if assets != 3:
        raise AssertionError("dependence return tensor must have exactly three assets")
    center = np.asarray(reference.asset_center, dtype=np.float64)
    scale = np.asarray(reference.asset_scale, dtype=np.float64)
    loadings = np.asarray(reference.pc1_loadings, dtype=np.float64)
    scaled_loadings = loadings / scale
    pc1_offset = float(np.dot(center / scale, loadings))
    pc1 = np.asarray(
        returns @ scaled_loadings - pc1_offset,
        dtype=np.float64,
    )
    lags = MONTHLY_LAGS if reference.frequency == "monthly" else DAILY_LAGS
    result = np.empty(
        (histories, len(PAIR_NAMES) + len(DIAGNOSTIC_NAMES) * len(lags)),
        dtype=np.float64,
    )
    column = 0
    for left, right in ((0, 1), (0, 2), (1, 2)):
        result[:, column] = _vectorized_correlations(
            returns[:, :, left], returns[:, :, right], column=column
        )
        column += 1

    row_continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), continuation_links)
    )
    segment_ids = np.cumsum(~row_continuation, dtype=np.int64) - 1
    for diagnostic, name in enumerate(DIAGNOSTIC_NAMES):
        series = _diagnostic_series(returns, pc1, diagnostic)
        centered = series - series.mean(axis=1, keepdims=True)
        for lag in lags:
            valid = segment_ids[lag:] == segment_ids[:-lag]
            if not bool(valid.any()):
                raise ModelError(
                    "dependence ACF has no available within-segment pairs",
                    details={"diagnostic": name, "lag": lag},
                )
            current = centered[:, lag:]
            previous = centered[:, :-lag]
            weights = valid.astype(np.float64, copy=False)
            numerator = np.einsum(
                "hi,i,hi->h", current, weights, previous, optimize=False
            )
            denominator = np.sqrt(
                np.einsum("hi,i,hi->h", current, weights, current, optimize=False)
                * np.einsum("hi,i,hi->h", previous, weights, previous, optimize=False)
            )
            invalid = ~np.isfinite(denominator) | (denominator <= CONSTANT_TOLERANCE)
            if bool(invalid.any()):
                failed = np.flatnonzero(invalid)
                raise ModelError(
                    "dependence ACF denominator is unavailable",
                    details={
                        "diagnostic": name,
                        "lag": lag,
                        "first_chunk_history": int(failed[0]),
                        "failed_histories": int(failed.size),
                    },
                )
            result[:, column] = np.clip(numerator / denominator, -1.0, 1.0)
            column += 1
    if column != result.shape[1] or observations <= 1:
        raise AssertionError("dependence endpoint order is inconsistent")
    transformed = np.arctanh(
        np.clip(
            result,
            -1.0 + CORRELATION_CLIP,
            1.0 - CORRELATION_CLIP,
        )
    ).astype(np.float64, copy=False)
    if not np.isfinite(transformed).all():
        raise ModelError("dependence vector contains a non-finite endpoint")
    return transformed


def _diagnostic_series(
    returns: FloatArray, pc1: FloatArray, diagnostic: int
) -> FloatArray:
    if diagnostic < 3:
        return returns[:, :, diagnostic]
    if diagnostic == 3:
        return pc1
    if diagnostic == 4:
        return np.asarray(returns[:, :, 2] - returns[:, :, 1], dtype=np.float64)
    if diagnostic == 5:
        return np.abs(pc1).astype(np.float64, copy=False)
    if diagnostic == 6:
        return np.square(pc1).astype(np.float64, copy=False)
    raise AssertionError("dependence diagnostic index is inconsistent")


def _vectorized_correlations(
    left: FloatArray, right: FloatArray, *, column: int
) -> FloatArray:
    centered_left = left - left.mean(axis=1, keepdims=True)
    centered_right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_left * centered_right, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(centered_left), axis=1)
        * np.sum(np.square(centered_right), axis=1)
    )
    invalid = ~np.isfinite(denominator) | (denominator <= CONSTANT_TOLERANCE)
    if bool(invalid.any()):
        failed = np.flatnonzero(invalid)
        raise ModelError(
            "contemporaneous correlation denominator is unavailable",
            details={
                "component": column,
                "first_chunk_history": int(failed[0]),
                "failed_histories": int(failed.size),
            },
        )
    result: FloatArray = np.asarray(
        np.clip(numerator / denominator, -1.0, 1.0), dtype=np.float64
    )
    return result


def _validate_trace_chunk(
    chunk: PairedTraceChunk,
    *,
    geometry: CalibrationGeometry,
    requested_lengths: tuple[int, ...],
    history_count: int,
    source_rows: int,
) -> tuple[int, int]:
    if not isinstance(chunk, PairedTraceChunk):
        raise ModelError("dependence reducer requires a PairedTraceChunk")
    if chunk.intended_length not in requested_lengths:
        raise ModelError(
            "dependence trace has an unrequested intended length",
            details={"intended_length": chunk.intended_length},
        )
    if (
        isinstance(chunk.history_start, bool)
        or not isinstance(chunk.history_start, int | np.integer)
        or chunk.history_start < 0
    ):
        raise ModelError("dependence trace history start must be non-negative")
    target = np.asarray(chunk.target_states)
    source = np.asarray(chunk.source_indices)
    conditioned = np.asarray(chunk.conditioned_restart_flags)
    ideal = np.asarray(chunk.ideal_restart_flags)
    links = np.asarray(chunk.target_continuation_links)
    expected_columns = geometry.target_observations
    if (
        target.ndim != 2
        or target.shape[0] == 0
        or target.shape[1] != expected_columns
        or target.dtype.kind not in {"i", "u"}
    ):
        raise ModelError("dependence trace target states have the wrong exact geometry")
    if source.shape != target.shape or source.dtype.kind not in {"i", "u"}:
        raise ModelError(
            "dependence trace source indices have the wrong exact geometry"
        )
    if (
        conditioned.shape != target.shape
        or conditioned.dtype.kind != "b"
        or ideal.shape != target.shape
        or ideal.dtype.kind != "b"
    ):
        raise ModelError("dependence trace restart flags have the wrong exact geometry")
    expected_links = geometry.target_continuation_links
    if (
        links.shape != expected_links.shape
        or links.dtype.kind != "b"
        or not np.array_equal(links, expected_links)
    ):
        raise ModelError(
            "dependence trace continuation mask has the wrong exact geometry"
        )
    for name, array in (
        ("target states", target),
        ("source indices", source),
        ("conditioned restart flags", conditioned),
        ("ideal restart flags", ideal),
        ("target continuation links", links),
    ):
        if array.flags.writeable:
            raise ModelError(f"dependence trace {name} must be immutable")
    if bool((target < 0).any()):
        raise ModelError("dependence trace target states must be non-negative")
    if bool(((source < 0) | (source >= source_rows)).any()):
        raise ModelError("dependence trace source index is outside return history")
    if not bool(conditioned[:, 0].all()) or not bool(ideal[:, 0].all()):
        raise ModelError("dependence trace first observation must be a restart")
    start = int(chunk.history_start)
    stop = start + target.shape[0]
    if stop > history_count:
        raise ModelError(
            "dependence trace history rows exceed the exact replicate count",
            details={"history_start": start, "history_stop": stop},
        )
    return start, stop


def _validate_reference(
    reference: DependenceReference,
    historical_returns: FloatArray,
    geometry: CalibrationGeometry,
) -> None:
    expected_components = 87 if reference.frequency == "monthly" else 31
    arrays = (
        reference.asset_center,
        reference.asset_scale,
        reference.pc1_loadings,
        reference.raw_values,
        reference.transformed_values,
    )
    if (
        np.asarray(reference.asset_center).shape != (3,)
        or np.asarray(reference.asset_scale).shape != (3,)
        or np.asarray(reference.pc1_loadings).shape != (3,)
        or np.asarray(reference.raw_values).shape != (expected_components,)
        or np.asarray(reference.transformed_values).shape != (expected_components,)
        or len(reference.labels) != expected_components
    ):
        raise ModelError("frozen dependence reference has an invalid exact shape")
    if any(np.asarray(array).flags.writeable for array in arrays):
        raise ModelError("dependence reference arrays must be frozen and immutable")
    if any(
        not np.isfinite(np.asarray(array, dtype=np.float64)).all() for array in arrays
    ):
        raise ModelError("dependence reference arrays must be finite")
    row_continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), geometry.target_continuation_links)
    )
    evaluated = evaluate_dependence_vector(
        historical_returns, row_continuation, reference
    )
    if (
        evaluated.labels != reference.labels
        or not np.allclose(
            evaluated.raw_values,
            reference.raw_values,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            evaluated.transformed_values,
            reference.transformed_values,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ModelError(
            "dependence reference is inconsistent with historical returns and geometry"
        )


def _requested_lengths(
    values: Sequence[int],
    *,
    selected_length: int,
    frequency: Frequency,
    require_all_feasible: bool,
) -> tuple[int, ...]:
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise ModelError(
            "requested dependence lengths must be an explicit sequence"
        ) from error
    if not supplied:
        raise ModelError("requested dependence lengths must be non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int | np.integer)
        for value in supplied
    ):
        raise ModelError("requested dependence lengths must contain integers")
    normalized = tuple(int(value) for value in supplied)
    if len(set(normalized)) != len(normalized):
        raise ModelError("requested dependence lengths must be unique")
    allowed = feasible_sensitivity_lengths(
        selected_length,
        frequency=frequency,
    )
    if selected_length not in normalized or any(
        intended not in allowed for intended in normalized
    ):
        raise ModelError(
            "requested dependence lengths must be selected L and feasible neighbors",
            details={"allowed": list(allowed), "supplied": list(normalized)},
        )
    supplied_set = set(normalized)
    requested = tuple(intended for intended in allowed if intended in supplied_set)
    if require_all_feasible and requested != allowed:
        raise ModelError(
            "strict canonical dependence execution requires every feasible neighbor",
            details={"required": list(allowed), "supplied": list(normalized)},
        )
    return requested


def _reference_frequency(reference: DependenceReference) -> Frequency:
    if not isinstance(reference, DependenceReference):
        raise ModelError("dependence execution requires a DependenceReference")
    if reference.frequency not in {"monthly", "daily"}:
        raise ModelError("dependence reference frequency must be monthly or daily")
    return reference.frequency


def _historical_returns(values: npt.ArrayLike, *, expected_rows: int) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("historical returns must be numeric") from error
    if result.shape != (expected_rows, 3):
        raise ModelError(
            "historical returns have the wrong exact geometry",
            details={"actual": list(result.shape), "expected": [expected_rows, 3]},
        )
    if not np.isfinite(result).all():
        raise ModelError("historical returns contain non-finite values")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ModelError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return result
