"""Finite, observable three-asset return-path metrics.

Inputs are synchronized log-return vectors in the fixed order ``(equity,
muni_bond, taxable_bond)``.  The module intentionally reports descriptive
values only; calibrated thresholds and pass/fail orchestration live elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError

ASSET_COUNT: Final = 3
ASSET_NAMES: Final = ("equity", "muni_bond", "taxable_bond")
METRICS_SCHEMA_ID: Final = "prpg-base-frequency-metrics-v1"
COMPOUNDING_SCHEMA_ID: Final = "prpg-log-compounding-audit-v1"
DIVERSITY_SCHEMA_ID: Final = "prpg-path-diversity-v1"
DEFAULT_QUANTILE_PROBABILITIES: Final = (0.01, 0.05, 0.50, 0.95, 0.99)

BaseFrequency: TypeAlias = Literal["daily", "monthly"]
Vector3: TypeAlias = tuple[float, float, float]
BoolVector3: TypeAlias = tuple[bool, bool, bool]
Matrix3: TypeAlias = tuple[Vector3, Vector3, Vector3]
VectorRows: TypeAlias = tuple[Vector3, ...]
FloatArray = NDArray[np.float64]

_ANNUALIZATION: Final[dict[BaseFrequency, int]] = {
    "daily": 252,
    "monthly": 12,
}
_RAW_ACF_LAGS: Final[dict[BaseFrequency, tuple[int, ...]]] = {
    "daily": tuple(range(1, 21)),
    "monthly": tuple(range(1, 13)),
}
_TRANSFORMED_ACF_LAGS: Final[dict[BaseFrequency, tuple[int, ...]]] = {
    "daily": tuple(range(1, 61)),
    "monthly": tuple(range(1, 13)),
}


@dataclass(frozen=True, slots=True)
class BaseFrequencyMetrics:
    """Canonical descriptive metrics for one finite synchronized return sample."""

    schema_id: str
    frequency: BaseFrequency
    observations: int
    annualization_factor: int
    annualized_log_mean: Vector3
    annualized_volatility: Vector3
    base_frequency_covariance: Matrix3
    annualized_covariance: Matrix3
    correlation: Matrix3
    constant_assets: BoolVector3
    quantile_probabilities: tuple[float, ...]
    marginal_quantiles: VectorRows
    lower_tail_probabilities: tuple[float, ...]
    lower_tail_means: VectorRows
    upper_tail_probabilities: tuple[float, ...]
    upper_tail_means: VectorRows
    raw_acf_lags: tuple[int, ...]
    raw_acf: VectorRows
    absolute_acf_lags: tuple[int, ...]
    absolute_acf: VectorRows
    squared_acf_lags: tuple[int, ...]
    squared_acf: VectorRows
    maximum_drawdown: Vector3
    maximum_drawdown_duration: tuple[int, int, int]
    fingerprint: str

    def identity_dict(self) -> dict[str, object]:
        """Return all fingerprint-bearing values in canonical field order."""

        return {
            "schema_id": self.schema_id,
            "frequency": self.frequency,
            "observations": self.observations,
            "annualization_factor": self.annualization_factor,
            "annualized_log_mean": list(self.annualized_log_mean),
            "annualized_volatility": list(self.annualized_volatility),
            "base_frequency_covariance": _lists(self.base_frequency_covariance),
            "annualized_covariance": _lists(self.annualized_covariance),
            "correlation": _lists(self.correlation),
            "constant_assets": list(self.constant_assets),
            "quantile_probabilities": list(self.quantile_probabilities),
            "marginal_quantiles": _lists(self.marginal_quantiles),
            "lower_tail_probabilities": list(self.lower_tail_probabilities),
            "lower_tail_means": _lists(self.lower_tail_means),
            "upper_tail_probabilities": list(self.upper_tail_probabilities),
            "upper_tail_means": _lists(self.upper_tail_means),
            "raw_acf_lags": list(self.raw_acf_lags),
            "raw_acf": _lists(self.raw_acf),
            "absolute_acf_lags": list(self.absolute_acf_lags),
            "absolute_acf": _lists(self.absolute_acf),
            "squared_acf_lags": list(self.squared_acf_lags),
            "squared_acf": _lists(self.squared_acf),
            "maximum_drawdown": list(self.maximum_drawdown),
            "maximum_drawdown_duration": list(self.maximum_drawdown_duration),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical JSON-compatible record."""

        result = self.identity_dict()
        result["fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True, slots=True)
class CrossFrequencyCompoundingAudit:
    """Exact or tolerance-bounded equality of derived and summed log returns."""

    schema_id: str
    base_observations: int
    aggregate_observations: int
    group_boundaries: tuple[int, ...]
    absolute_tolerance: float
    maximum_absolute_error: Vector3
    expected_sha256: str
    observed_sha256: str
    exact_match: bool
    passed: bool
    fingerprint: str

    def identity_dict(self) -> dict[str, object]:
        """Return all fingerprint-bearing values."""

        return {
            "schema_id": self.schema_id,
            "base_observations": self.base_observations,
            "aggregate_observations": self.aggregate_observations,
            "group_boundaries": list(self.group_boundaries),
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_absolute_error": list(self.maximum_absolute_error),
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "exact_match": self.exact_match,
            "passed": self.passed,
        }

    def as_dict(self) -> dict[str, object]:
        result = self.identity_dict()
        result["fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True, slots=True)
class PathDiversityMetrics:
    """Exact complete-path duplicate counts from canonical matrix bytes."""

    schema_id: str
    path_count: int
    periods_per_path: int
    unique_path_count: int
    duplicate_copies: int
    duplicate_group_count: int
    paths_in_duplicate_groups: int
    maximum_multiplicity: int
    unique_fraction: float
    duplicate_hashes: tuple[str, ...]
    fingerprint: str

    def identity_dict(self) -> dict[str, object]:
        """Return all fingerprint-bearing values."""

        return {
            "schema_id": self.schema_id,
            "path_count": self.path_count,
            "periods_per_path": self.periods_per_path,
            "unique_path_count": self.unique_path_count,
            "duplicate_copies": self.duplicate_copies,
            "duplicate_group_count": self.duplicate_group_count,
            "paths_in_duplicate_groups": self.paths_in_duplicate_groups,
            "maximum_multiplicity": self.maximum_multiplicity,
            "unique_fraction": self.unique_fraction,
            "duplicate_hashes": list(self.duplicate_hashes),
        }

    def as_dict(self) -> dict[str, object]:
        result = self.identity_dict()
        result["fingerprint"] = self.fingerprint
        return result


class PathDiversityAccumulator:
    """Streaming complete-path hash counter with constant per-path work memory."""

    def __init__(self) -> None:
        self._periods: int | None = None
        self._path_count = 0
        self._counts: Counter[str] = Counter()

    @property
    def path_count(self) -> int:
        """Return the number of accepted complete paths."""

        return self._path_count

    def update(self, log_returns: ArrayLike) -> None:
        """Consume one finite ``periods x 3`` complete base-frequency path."""

        values = _return_matrix(log_returns, minimum_observations=1)
        periods = int(values.shape[0])
        if self._periods is None:
            self._periods = periods
        elif periods != self._periods:
            raise ValidationError("path-diversity inputs use different horizons")
        digest = hashlib.sha256(_canonical_array_bytes(values)).hexdigest()
        self._counts[digest] += 1
        self._path_count += 1

    def finalize(self) -> PathDiversityMetrics:
        """Freeze duplicate and diversity counts after at least one path."""

        if self._path_count == 0 or self._periods is None:
            raise ValidationError("path diversity requires at least one path")
        duplicate_hashes = tuple(
            sorted(digest for digest, count in self._counts.items() if count > 1)
        )
        unique_count = len(self._counts)
        unique_fraction = _zero(float(unique_count / self._path_count))
        in_duplicate_groups = sum(self._counts[item] for item in duplicate_hashes)
        maximum = max(self._counts.values())
        identity: dict[str, object] = {
            "schema_id": DIVERSITY_SCHEMA_ID,
            "path_count": self._path_count,
            "periods_per_path": self._periods,
            "unique_path_count": unique_count,
            "duplicate_copies": self._path_count - unique_count,
            "duplicate_group_count": len(duplicate_hashes),
            "paths_in_duplicate_groups": in_duplicate_groups,
            "maximum_multiplicity": maximum,
            "unique_fraction": unique_fraction,
            "duplicate_hashes": list(duplicate_hashes),
        }
        return PathDiversityMetrics(
            schema_id=DIVERSITY_SCHEMA_ID,
            path_count=self._path_count,
            periods_per_path=self._periods,
            unique_path_count=unique_count,
            duplicate_copies=self._path_count - unique_count,
            duplicate_group_count=len(duplicate_hashes),
            paths_in_duplicate_groups=in_duplicate_groups,
            maximum_multiplicity=maximum,
            unique_fraction=unique_fraction,
            duplicate_hashes=duplicate_hashes,
            fingerprint=_fingerprint(identity),
        )


def compute_base_frequency_metrics(
    log_returns: ArrayLike,
    *,
    frequency: BaseFrequency,
    quantile_probabilities: Sequence[float] = DEFAULT_QUANTILE_PROBABILITIES,
    raw_acf_lags: Sequence[int] | None = None,
    transformed_acf_lags: Sequence[int] | None = None,
) -> BaseFrequencyMetrics:
    """Compute finite base-frequency summaries without pooling path metadata.

    Sample covariance and volatility use ``ddof=1``.  Positive-lag ACF uses
    the full-sample mean and denominator; a constant transformed series is
    represented by finite zero ACF and is separately disclosed through
    ``constant_assets`` for raw returns.
    """

    checked_frequency = _frequency(frequency)
    values = _return_matrix(log_returns, minimum_observations=2)
    probabilities = _probabilities(quantile_probabilities)
    raw_lags = _lags(
        _RAW_ACF_LAGS[checked_frequency] if raw_acf_lags is None else raw_acf_lags,
        values.shape[0],
        "raw ACF lags",
    )
    transformed_lags = _lags(
        _TRANSFORMED_ACF_LAGS[checked_frequency]
        if transformed_acf_lags is None
        else transformed_acf_lags,
        values.shape[0],
        "transformed ACF lags",
    )
    annualization = _ANNUALIZATION[checked_frequency]

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(values, axis=0, dtype=np.float64)
        covariance = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=np.float64)
        constant = np.ptp(values, axis=0) == 0.0
        # np.cov can leave roundoff-sized nonzero cells for a nonzero constant
        # column.  The exact observable condition is known, so canonicalize
        # those mathematically zero covariance rows and columns.
        covariance[constant, :] = 0.0
        covariance[:, constant] = 0.0
        annualized_mean = np.asarray(mean * annualization, dtype=np.float64)
        annualized_covariance = np.asarray(covariance * annualization, dtype=np.float64)
        diagonal = np.maximum(np.diag(covariance), 0.0)
        annualized_volatility = np.sqrt(diagonal * annualization)
    _require_finite(
        annualized_mean,
        covariance,
        annualized_covariance,
        annualized_volatility,
        label="mean/covariance metrics",
    )
    correlation = _finite_correlation(covariance, diagonal)

    quantiles = np.asarray(
        np.quantile(values, probabilities, axis=0, method="higher"),
        dtype=np.float64,
    )
    lower_indices = tuple(
        index for index, value in enumerate(probabilities) if value < 0.5
    )
    upper_indices = tuple(
        index for index, value in enumerate(probabilities) if value > 0.5
    )
    lower_tail_means = _tail_means(values, quantiles, lower_indices, lower=True)
    upper_tail_means = _tail_means(values, quantiles, upper_indices, lower=False)

    raw_acf = _autocorrelations(values, raw_lags)
    absolute_acf = _autocorrelations(np.abs(values), transformed_lags)
    squared_acf = _autocorrelations(values * values, transformed_lags)
    drawdown, duration = _drawdown_metrics(values)
    constant_assets = _bool_vector3(constant)
    drawdown_duration = _int_vector3(duration)
    _require_finite(
        quantiles,
        lower_tail_means,
        upper_tail_means,
        raw_acf,
        absolute_acf,
        squared_acf,
        drawdown,
        label="tail/ACF/drawdown metrics",
    )

    identity: dict[str, object] = {
        "schema_id": METRICS_SCHEMA_ID,
        "frequency": checked_frequency,
        "observations": int(values.shape[0]),
        "annualization_factor": annualization,
        "annualized_log_mean": _vector(annualized_mean),
        "annualized_volatility": _vector(annualized_volatility),
        "base_frequency_covariance": _matrix(covariance),
        "annualized_covariance": _matrix(annualized_covariance),
        "correlation": _matrix(correlation),
        "constant_assets": constant_assets,
        "quantile_probabilities": probabilities,
        "marginal_quantiles": _rows(quantiles),
        "lower_tail_probabilities": tuple(
            probabilities[index] for index in lower_indices
        ),
        "lower_tail_means": _rows(lower_tail_means),
        "upper_tail_probabilities": tuple(
            probabilities[index] for index in upper_indices
        ),
        "upper_tail_means": _rows(upper_tail_means),
        "raw_acf_lags": raw_lags,
        "raw_acf": _rows(raw_acf),
        "absolute_acf_lags": transformed_lags,
        "absolute_acf": _rows(absolute_acf),
        "squared_acf_lags": transformed_lags,
        "squared_acf": _rows(squared_acf),
        "maximum_drawdown": _vector(drawdown),
        "maximum_drawdown_duration": drawdown_duration,
    }
    return BaseFrequencyMetrics(
        schema_id=METRICS_SCHEMA_ID,
        frequency=checked_frequency,
        observations=int(values.shape[0]),
        annualization_factor=annualization,
        annualized_log_mean=_vector(annualized_mean),
        annualized_volatility=_vector(annualized_volatility),
        base_frequency_covariance=_matrix(covariance),
        annualized_covariance=_matrix(annualized_covariance),
        correlation=_matrix(correlation),
        constant_assets=constant_assets,
        quantile_probabilities=probabilities,
        marginal_quantiles=_rows(quantiles),
        lower_tail_probabilities=tuple(probabilities[index] for index in lower_indices),
        lower_tail_means=_rows(lower_tail_means),
        upper_tail_probabilities=tuple(probabilities[index] for index in upper_indices),
        upper_tail_means=_rows(upper_tail_means),
        raw_acf_lags=raw_lags,
        raw_acf=_rows(raw_acf),
        absolute_acf_lags=transformed_lags,
        absolute_acf=_rows(absolute_acf),
        squared_acf_lags=transformed_lags,
        squared_acf=_rows(squared_acf),
        maximum_drawdown=_vector(drawdown),
        maximum_drawdown_duration=drawdown_duration,
        fingerprint=_fingerprint(identity),
    )


def cross_frequency_log_compounding_identity(
    base_log_returns: ArrayLike,
    aggregate_log_returns: ArrayLike,
    group_boundaries: Sequence[int],
    *,
    absolute_tolerance: float = 0.0,
) -> CrossFrequencyCompoundingAudit:
    """Compare aggregates with sums over explicit contiguous base segments.

    Boundaries include both zero and the final base-row count.  This supports
    fixed months/quarters and irregular model-week boundaries with one API.
    The default is an exact floating-value identity; a positive tolerance must
    be explicit and remains fingerprint-bearing.
    """

    base = _return_matrix(base_log_returns, minimum_observations=1)
    aggregate = _return_matrix(aggregate_log_returns, minimum_observations=1)
    boundaries = _boundaries(group_boundaries, base.shape[0])
    if aggregate.shape[0] != len(boundaries) - 1:
        raise ValidationError("aggregate rows do not match compounding groups")
    tolerance = _nonnegative_finite(absolute_tolerance, "absolute tolerance")
    expected = np.empty_like(aggregate)
    for index, (start, stop) in enumerate(pairwise(boundaries)):
        expected[index] = np.sum(base[start:stop], axis=0, dtype=np.float64)
    _require_finite(expected, label="compounded log returns")
    errors = np.abs(expected - aggregate)
    maximum = np.max(errors, axis=0)
    exact = bool(np.array_equal(expected, aggregate))
    passed = bool(np.all(errors <= tolerance))
    identity: dict[str, object] = {
        "schema_id": COMPOUNDING_SCHEMA_ID,
        "base_observations": int(base.shape[0]),
        "aggregate_observations": int(aggregate.shape[0]),
        "group_boundaries": boundaries,
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": _vector(maximum),
        "expected_sha256": hashlib.sha256(_canonical_array_bytes(expected)).hexdigest(),
        "observed_sha256": hashlib.sha256(
            _canonical_array_bytes(aggregate)
        ).hexdigest(),
        "exact_match": exact,
        "passed": passed,
    }
    return CrossFrequencyCompoundingAudit(
        schema_id=COMPOUNDING_SCHEMA_ID,
        base_observations=int(base.shape[0]),
        aggregate_observations=int(aggregate.shape[0]),
        group_boundaries=boundaries,
        absolute_tolerance=tolerance,
        maximum_absolute_error=_vector(maximum),
        expected_sha256=str(identity["expected_sha256"]),
        observed_sha256=str(identity["observed_sha256"]),
        exact_match=exact,
        passed=passed,
        fingerprint=_fingerprint(identity),
    )


def fixed_group_boundaries(
    observations: int, periods_per_group: int
) -> tuple[int, ...]:
    """Return exact boundaries for evenly sized contiguous aggregates."""

    rows = _positive_integer(observations, "observations")
    group = _positive_integer(periods_per_group, "periods per group")
    if rows % group != 0:
        raise ValidationError("observations are not divisible by periods per group")
    return tuple(range(0, rows + 1, group))


def compute_path_diversity(paths: Iterable[ArrayLike]) -> PathDiversityMetrics:
    """Stream any iterable of complete paths into exact duplicate metrics."""

    if not isinstance(paths, Iterable):
        raise ValidationError("paths must be iterable")
    accumulator = PathDiversityAccumulator()
    for path in paths:
        accumulator.update(path)
    return accumulator.finalize()


def _autocorrelations(values: FloatArray, lags: tuple[int, ...]) -> FloatArray:
    centered = values - np.mean(values, axis=0, dtype=np.float64)
    denominator = np.sum(centered * centered, axis=0, dtype=np.float64)
    result = np.zeros((len(lags), ASSET_COUNT), dtype=np.float64)
    positive = (denominator > 0.0) & (np.ptp(values, axis=0) > 0.0)
    for index, lag in enumerate(lags):
        numerator = np.sum(centered[:-lag] * centered[lag:], axis=0, dtype=np.float64)
        np.divide(numerator, denominator, out=result[index], where=positive)
    return result


def _finite_correlation(covariance: FloatArray, diagonal: FloatArray) -> FloatArray:
    scales = np.sqrt(diagonal)
    denominator = np.outer(scales, scales)
    result = np.zeros((ASSET_COUNT, ASSET_COUNT), dtype=np.float64)
    np.divide(covariance, denominator, out=result, where=denominator > 0.0)
    np.fill_diagonal(result, 1.0)
    result = np.clip(result, -1.0, 1.0)
    _require_finite(result, label="correlation")
    return result


def _tail_means(
    values: FloatArray,
    quantiles: FloatArray,
    indices: tuple[int, ...],
    *,
    lower: bool,
) -> FloatArray:
    result = np.empty((len(indices), ASSET_COUNT), dtype=np.float64)
    for output_index, quantile_index in enumerate(indices):
        for asset in range(ASSET_COUNT):
            threshold = quantiles[quantile_index, asset]
            selected = (
                values[:, asset] <= threshold
                if lower
                else values[:, asset] >= threshold
            )
            # Higher empirical quantiles and finite non-empty input guarantee
            # at least one selected observation in either tail.
            result[output_index, asset] = np.mean(
                values[selected, asset], dtype=np.float64
            )
    return result


def _drawdown_metrics(values: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
    with np.errstate(over="ignore", invalid="ignore"):
        cumulative = np.vstack(
            (
                np.zeros((1, ASSET_COUNT), dtype=np.float64),
                np.cumsum(values, axis=0, dtype=np.float64),
            )
        )
    _require_finite(cumulative, label="cumulative log wealth")
    running_peak = np.maximum.accumulate(cumulative, axis=0)
    drawdown_log = cumulative - running_peak
    depths = -np.expm1(np.min(drawdown_log, axis=0))
    durations = np.zeros(ASSET_COUNT, dtype=np.int64)
    current = np.zeros(ASSET_COUNT, dtype=np.int64)
    for row in range(1, cumulative.shape[0]):
        below_peak = cumulative[row] < running_peak[row]
        current = np.where(below_peak, current + 1, 0)
        durations = np.maximum(durations, current)
    return np.asarray(depths, dtype=np.float64), durations


def _return_matrix(value: ArrayLike, *, minimum_observations: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (ASSET_COUNT,):
        raise ValidationError("log returns must have shape (observations, 3)")
    if result.shape[0] < minimum_observations:
        raise ValidationError(
            f"log returns require at least {minimum_observations} observations"
        )
    if not bool(np.isfinite(result).all()):
        raise ValidationError("log returns must all be finite")
    return result


def _frequency(value: object) -> BaseFrequency:
    if value == "daily":
        return "daily"
    if value == "monthly":
        return "monthly"
    raise ValidationError("base frequency must be daily or monthly")


def _probabilities(values: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValidationError("quantile probabilities must be a non-empty sequence")
    result = tuple(_open_probability(value, "quantile probability") for value in values)
    if tuple(sorted(set(result))) != result:
        raise ValidationError("quantile probabilities must be unique and increasing")
    return result


def _lags(values: Sequence[int], observations: int, label: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValidationError(f"{label} must be a non-empty sequence")
    result = tuple(_positive_integer(value, label) for value in values)
    if tuple(sorted(set(result))) != result:
        raise ValidationError(f"{label} must be unique and increasing")
    if result[-1] >= observations:
        raise ValidationError(f"{label} must be smaller than observations")
    return result


def _boundaries(values: Sequence[int], observations: int) -> tuple[int, ...]:
    if not isinstance(values, Sequence):
        raise ValidationError("group boundaries must be a sequence")
    result = tuple(_nonnegative_integer(value, "group boundary") for value in values)
    if len(result) < 2 or result[0] != 0 or result[-1] != observations:
        raise ValidationError("group boundaries must span zero through all base rows")
    if any(left >= right for left, right in pairwise(result)):
        raise ValidationError("group boundaries must be strictly increasing")
    return result


def _vector(values: ArrayLike) -> Vector3:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (ASSET_COUNT,) or not bool(np.isfinite(array).all()):
        raise ValidationError("three-asset metric vector is invalid")
    return (_zero(float(array[0])), _zero(float(array[1])), _zero(float(array[2])))


def _rows(values: ArrayLike) -> VectorRows:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (ASSET_COUNT,):
        raise ValidationError("metric rows must have three assets")
    return tuple(_vector(row) for row in array)


def _matrix(values: ArrayLike) -> Matrix3:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (ASSET_COUNT, ASSET_COUNT):
        raise ValidationError("three-asset metric matrix is invalid")
    rows = _rows(array)
    return (rows[0], rows[1], rows[2])


def _bool_vector3(values: ArrayLike) -> BoolVector3:
    array = np.asarray(values, dtype=np.bool_)
    if array.shape != (ASSET_COUNT,):
        raise ValidationError("three-asset boolean vector is invalid")
    return (bool(array[0]), bool(array[1]), bool(array[2]))


def _int_vector3(values: ArrayLike) -> tuple[int, int, int]:
    array = np.asarray(values, dtype=np.int64)
    if array.shape != (ASSET_COUNT,):
        raise ValidationError("three-asset integer vector is invalid")
    return (int(array[0]), int(array[1]), int(array[2]))


def _lists(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in values]


def _canonical_array_bytes(values: FloatArray) -> bytes:
    normalized = np.asarray(values, dtype="<f8", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    return normalized.tobytes(order="C")


def _require_finite(*values: FloatArray, label: str) -> None:
    if any(not bool(np.isfinite(value).all()) for value in values):
        raise ValidationError(f"{label} must be finite")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _open_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{label} must be finite and in (0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValidationError(f"{label} must be finite and in (0, 1)")
    return result


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{label} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{label} must be finite and non-negative")
    return _zero(result)


def _zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
