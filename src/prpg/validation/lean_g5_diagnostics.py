"""Observable-only path diagnostics for the lean G5 plausibility checks.

The module deliberately stops at deterministic, compact measurements.  It
does not calibrate thresholds, make a G5 decision, publish evidence, or grant
generation authority.  Inputs are synchronized three-asset log returns in the
fixed order ``(equity, muni_bond, taxable_bond)``.

Hidden states are optional audit inputs from the generator.  They are never
inferred from, or exposed with, consumer return CSVs.  A state can legitimately
be absent from a finite path or even a small path collection; undefined
transition, dwell, and conditional-mean entries are therefore represented by
``None`` rather than silently filled with zero.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError
from prpg.validation.metrics import compute_base_frequency_metrics

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Frequency: TypeAlias = Literal["monthly", "daily"]
OptionalVector4: TypeAlias = tuple[
    float | None, float | None, float | None, float | None
]
OptionalMatrix4: TypeAlias = tuple[
    OptionalVector4,
    OptionalVector4,
    OptionalVector4,
    OptionalVector4,
]
OptionalStateMeans: TypeAlias = tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]

LEAN_G5_DIAGNOSTICS_SCHEMA_ID: Final = "prpg-g5-lean-path-diagnostics-v1"
LEAN_G5_STATE_BEHAVIOR_SCHEMA_ID: Final = "prpg-g5-lean-state-behavior-v1"
LEAN_G5_DIVERSITY_SCHEMA_ID: Final = "prpg-g5-lean-return-diversity-v1"
LEAN_G5_STATE_COUNT: Final = 4
LEAN_G5_ASSET_NAMES: Final = ("equity", "muni_bond", "taxable_bond")
LEAN_G5_QUANTILE_PROBABILITIES: Final = (0.01, 0.05, 0.50, 0.95, 0.99)

_ANNUALIZATION: Final[dict[Frequency, int]] = {"monthly": 12, "daily": 252}
_RAW_ACF_LAGS: Final[dict[Frequency, tuple[int, ...]]] = {
    "monthly": tuple(range(1, 13)),
    "daily": tuple(range(1, 21)),
}
_TRANSFORMED_ACF_LAGS: Final[dict[Frequency, tuple[int, ...]]] = {
    "monthly": tuple(range(1, 13)),
    "daily": tuple(range(1, 61)),
}
_REPEATED_SUBSEQUENCE_WINDOWS: Final[dict[Frequency, int]] = {
    "monthly": 6,
    "daily": 63,
}
_TAIL_COEXCEEDANCE_PROBABILITY: Final[dict[Frequency, float]] = {
    "monthly": 0.10,
    "daily": 0.05,
}


@dataclass(frozen=True, slots=True)
class LeanG5PathDiagnostics:
    """One compact row of return-observable G5-B/C/D measurements."""

    schema_id: str
    frequency: Frequency
    observations: int
    names: tuple[str, ...]
    values: tuple[float, ...]

    def value(self, name: str) -> float:
        """Return one named endpoint, failing clearly on a misspelled name."""

        try:
            return self.values[self.names.index(name)]
        except ValueError as error:
            raise ValidationError(
                f"unknown lean G5 diagnostic endpoint: {name}"
            ) from error


@dataclass(frozen=True, slots=True)
class LeanG5StateBehavior:
    """Aggregate K=4 occupancy, transition, dwell, and conditional-return inputs."""

    schema_id: str
    path_count: int
    observations: int
    state_count: int
    supported_states: tuple[bool, bool, bool, bool]
    occupancy_counts: tuple[int, int, int, int]
    occupancy: tuple[float, float, float, float]
    departure_counts: tuple[int, int, int, int]
    transition_probabilities: OptionalMatrix4
    episode_counts: tuple[int, int, int, int]
    mean_dwell: OptionalVector4
    median_dwell: OptionalVector4
    one_period_spell_fraction: OptionalVector4
    state_conditional_log_mean: OptionalStateMeans


@dataclass(frozen=True, slots=True)
class LeanG5ReturnDiversity:
    """Exact-return duplicate and fixed-window repeated-subsequence counts."""

    schema_id: str
    frequency: Frequency
    path_count: int
    periods_per_path: int
    unique_complete_paths: int
    complete_path_duplicate_copies: int
    complete_path_duplicate_groups: int
    maximum_complete_path_multiplicity: int
    subsequence_window: int
    subsequence_windows: int
    unique_subsequences: int
    repeated_subsequence_copies: int
    repeated_subsequence_groups: int
    maximum_subsequence_multiplicity: int


def compute_lean_g5_path_diagnostics(
    log_returns: ArrayLike,
    *,
    frequency: Frequency,
) -> LeanG5PathDiagnostics:
    """Measure one equal-history path without using hidden generator fields.

    The row retains the established 12/20 raw-return ACF lags and 12/60
    absolute/squared-return ACF lags.  It also includes lag-one cross-asset
    correlation, marginal and joint downside-tail summaries, drawdowns, and
    the annual-horizon taxable-minus-municipal spread.  Canonical equal-history
    paths are much longer than the minimum 13 monthly or 253 daily rows.
    """

    checked_frequency = _frequency(frequency)
    minimum = _ANNUALIZATION[checked_frequency] + 1
    values = _return_matrix(log_returns, minimum_observations=minimum)
    metrics = compute_base_frequency_metrics(
        values,
        frequency=checked_frequency,
        quantile_probabilities=LEAN_G5_QUANTILE_PROBABILITIES,
        raw_acf_lags=_RAW_ACF_LAGS[checked_frequency],
        transformed_acf_lags=_TRANSFORMED_ACF_LAGS[checked_frequency],
    )

    names: list[str] = []
    endpoints: list[float] = []

    def add(name: str, value: float | int | np.float64) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValidationError(f"lean G5 endpoint {name} is non-finite")
        names.append(name)
        endpoints.append(0.0 if numeric == 0.0 else numeric)

    for asset, asset_name in enumerate(LEAN_G5_ASSET_NAMES):
        add(f"annualized_log_mean.{asset_name}", metrics.annualized_log_mean[asset])
        add(f"annualized_volatility.{asset_name}", metrics.annualized_volatility[asset])
        add(f"constant_asset.{asset_name}", int(metrics.constant_assets[asset]))

    covariance = np.asarray(metrics.base_frequency_covariance, dtype=np.float64)
    correlation = np.asarray(metrics.correlation, dtype=np.float64)
    for left in range(3):
        for right in range(left, 3):
            add(f"covariance.{left}.{right}", covariance[left, right])
    for left, right in ((0, 1), (0, 2), (1, 2)):
        add(f"correlation.{left}.{right}", correlation[left, right])

    quantiles = np.asarray(metrics.marginal_quantiles, dtype=np.float64)
    for slot, probability in enumerate(LEAN_G5_QUANTILE_PROBABILITIES):
        for asset, asset_name in enumerate(LEAN_G5_ASSET_NAMES):
            add(f"quantile.{probability:.2f}.{asset_name}", quantiles[slot, asset])

    for probability, row in zip(
        metrics.lower_tail_probabilities, metrics.lower_tail_means, strict=True
    ):
        for asset, asset_name in enumerate(LEAN_G5_ASSET_NAMES):
            add(f"lower_tail_mean.{probability:.2f}.{asset_name}", row[asset])
    for probability, row in zip(
        metrics.upper_tail_probabilities, metrics.upper_tail_means, strict=True
    ):
        for asset, asset_name in enumerate(LEAN_G5_ASSET_NAMES):
            add(f"upper_tail_mean.{probability:.2f}.{asset_name}", row[asset])

    for prefix, lags, rows in (
        ("raw_acf", metrics.raw_acf_lags, metrics.raw_acf),
        ("absolute_acf", metrics.absolute_acf_lags, metrics.absolute_acf),
        ("squared_acf", metrics.squared_acf_lags, metrics.squared_acf),
    ):
        for lag, row in zip(lags, rows, strict=True):
            for asset, asset_name in enumerate(LEAN_G5_ASSET_NAMES):
                add(f"{prefix}.{lag}.{asset_name}", row[asset])

    cross_lag = _cross_lag_correlation(values)
    for left in range(3):
        for right in range(3):
            add(f"cross_lag.{left}.{right}", cross_lag[left, right])

    for asset, asset_name in enumerate(LEAN_G5_ASSET_NAMES):
        add(f"drawdown_depth.{asset_name}", metrics.maximum_drawdown[asset])
        add(
            f"drawdown_duration_fraction.{asset_name}",
            metrics.maximum_drawdown_duration[asset] / values.shape[0],
        )

    spread = values[:, 2] - values[:, 1]
    spread_quantiles = np.quantile(
        spread, LEAN_G5_QUANTILE_PROBABILITIES, method="higher"
    )
    rolling_spread = _rolling_sums(spread, _ANNUALIZATION[checked_frequency])
    rolling_quantiles = np.quantile(rolling_spread, (0.10, 0.50, 0.90), method="higher")
    add("taxable_muni_spread.mean", float(np.mean(spread)))
    add("taxable_muni_spread.volatility", float(np.std(spread, ddof=1)))
    for probability, value in zip(
        LEAN_G5_QUANTILE_PROBABILITIES, spread_quantiles, strict=True
    ):
        add(f"taxable_muni_spread.quantile.{probability:.2f}", value)
    for probability, value in zip((0.10, 0.50, 0.90), rolling_quantiles, strict=True):
        add(f"taxable_muni_spread.annual_sum_quantile.{probability:.2f}", value)
    add(
        "taxable_muni_spread.annual_sum_volatility",
        float(np.std(rolling_spread, ddof=1)),
    )

    tail_probability = _TAIL_COEXCEEDANCE_PROBABILITY[checked_frequency]
    thresholds = np.quantile(values, tail_probability, axis=0, method="higher")
    downside = values <= thresholds
    for left, right in ((0, 1), (0, 2), (1, 2)):
        add(
            f"downside_coexceedance.{left}.{right}",
            np.mean(downside[:, left] & downside[:, right]),
        )

    if len(names) != len(set(names)):
        raise ValidationError("lean G5 endpoint names are not unique")
    return LeanG5PathDiagnostics(
        schema_id=LEAN_G5_DIAGNOSTICS_SCHEMA_ID,
        frequency=checked_frequency,
        observations=int(values.shape[0]),
        names=tuple(names),
        values=tuple(endpoints),
    )


def compute_lean_g5_state_behavior(
    return_paths: Sequence[ArrayLike],
    state_paths: Sequence[ArrayLike],
) -> LeanG5StateBehavior:
    """Aggregate audit-only K=4 state behavior without crossing path borders."""

    if not isinstance(return_paths, Sequence) or not return_paths:
        raise ValidationError("lean G5 state behavior requires return paths")
    if not isinstance(state_paths, Sequence) or len(state_paths) != len(return_paths):
        raise ValidationError("lean G5 return/state path counts differ")

    occupancy = np.zeros(LEAN_G5_STATE_COUNT, dtype=np.int64)
    transition = np.zeros((LEAN_G5_STATE_COUNT, LEAN_G5_STATE_COUNT), dtype=np.int64)
    conditional_sum = np.zeros((LEAN_G5_STATE_COUNT, 3), dtype=np.float64)
    dwell: list[list[int]] = [[] for _ in range(LEAN_G5_STATE_COUNT)]
    observations = 0

    for raw_returns, raw_states in zip(return_paths, state_paths, strict=True):
        values = _return_matrix(raw_returns, minimum_observations=2)
        states = _state_vector(raw_states, expected_observations=values.shape[0])
        observations += int(values.shape[0])
        occupancy += np.bincount(states, minlength=LEAN_G5_STATE_COUNT)
        np.add.at(transition, (states[:-1], states[1:]), 1)
        for state in range(LEAN_G5_STATE_COUNT):
            conditional_sum[state] += np.sum(
                values[states == state], axis=0, dtype=np.float64
            )

        boundaries = np.flatnonzero(states[1:] != states[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), boundaries))
        stops = np.concatenate((boundaries, np.asarray([states.size])))
        for start, stop in zip(starts, stops, strict=True):
            dwell[int(states[start])].append(int(stop - start))

    supported = tuple(bool(value > 0) for value in occupancy)
    departure = transition.sum(axis=1)
    transition_rows: list[OptionalVector4] = []
    means: list[tuple[float, float, float] | None] = []
    mean_dwell: list[float | None] = []
    median_dwell: list[float | None] = []
    one_period: list[float | None] = []
    for state in range(LEAN_G5_STATE_COUNT):
        if departure[state] == 0:
            transition_rows.append((None, None, None, None))
        else:
            row = transition[state] / departure[state]
            transition_rows.append(tuple(float(value) for value in row))  # type: ignore[arg-type]
        if occupancy[state] == 0:
            means.append(None)
        else:
            row = conditional_sum[state] / occupancy[state]
            means.append(tuple(float(value) for value in row))  # type: ignore[arg-type]
        if not dwell[state]:
            mean_dwell.append(None)
            median_dwell.append(None)
            one_period.append(None)
        else:
            episodes = np.asarray(dwell[state], dtype=np.float64)
            mean_dwell.append(float(np.mean(episodes)))
            median_dwell.append(float(np.median(episodes)))
            one_period.append(float(np.mean(episodes == 1.0)))

    return LeanG5StateBehavior(
        schema_id=LEAN_G5_STATE_BEHAVIOR_SCHEMA_ID,
        path_count=len(return_paths),
        observations=observations,
        state_count=LEAN_G5_STATE_COUNT,
        supported_states=supported,  # type: ignore[arg-type]
        occupancy_counts=tuple(int(value) for value in occupancy),  # type: ignore[arg-type]
        occupancy=tuple(float(value / observations) for value in occupancy),  # type: ignore[arg-type]
        departure_counts=tuple(int(value) for value in departure),  # type: ignore[arg-type]
        transition_probabilities=tuple(transition_rows),  # type: ignore[arg-type]
        episode_counts=tuple(len(values) for values in dwell),  # type: ignore[arg-type]
        mean_dwell=tuple(mean_dwell),  # type: ignore[arg-type]
        median_dwell=tuple(median_dwell),  # type: ignore[arg-type]
        one_period_spell_fraction=tuple(one_period),  # type: ignore[arg-type]
        state_conditional_log_mean=tuple(means),  # type: ignore[arg-type]
    )


def compute_lean_g5_return_diversity(
    paths: Iterable[ArrayLike],
    *,
    frequency: Frequency,
    subsequence_window: int | None = None,
) -> LeanG5ReturnDiversity:
    """Count complete duplicates and repeated exact synchronized-return windows.

    The scan hashes canonical return-matrix bytes only.  It never reads path
    IDs, dates, source indices, restart flags, macro data, or hidden states.
    The default six-month/63-session window is an observable metric whose
    eventual release threshold must be calibrated; bootstrap paths are
    expected to share some historical blocks, so mere nonzero repetition is
    not itself a failure.
    """

    checked_frequency = _frequency(frequency)
    checked_paths = tuple(
        _return_matrix(path, minimum_observations=1) for path in paths
    )
    if not checked_paths:
        raise ValidationError("lean G5 diversity requires at least one path")
    periods = int(checked_paths[0].shape[0])
    if any(path.shape[0] != periods for path in checked_paths[1:]):
        raise ValidationError("lean G5 diversity paths use different horizons")
    window = (
        _REPEATED_SUBSEQUENCE_WINDOWS[checked_frequency]
        if subsequence_window is None
        else _positive_integer(subsequence_window, "subsequence window")
    )
    if window > periods:
        raise ValidationError("lean G5 subsequence window exceeds the path horizon")

    complete_counts: Counter[bytes] = Counter()
    windows_per_path = periods - window + 1
    digest_rows = np.empty(len(checked_paths) * windows_per_path, dtype="V32")
    cursor = 0
    for path in checked_paths:
        normalized = _canonical_return_matrix(path)
        complete_counts[hashlib.sha256(normalized.tobytes(order="C")).digest()] += 1
        for start in range(windows_per_path):
            payload = normalized[start : start + window].tobytes(order="C")
            digest_rows[cursor] = np.void(hashlib.sha256(payload).digest())
            cursor += 1

    _, subsequence_counts = np.unique(digest_rows, return_counts=True)
    unique_subsequences = int(subsequence_counts.size)
    repeated_groups = int(np.count_nonzero(subsequence_counts > 1))
    complete_multiplicities = np.fromiter(
        complete_counts.values(), dtype=np.int64, count=len(complete_counts)
    )
    complete_groups = int(np.count_nonzero(complete_multiplicities > 1))
    return LeanG5ReturnDiversity(
        schema_id=LEAN_G5_DIVERSITY_SCHEMA_ID,
        frequency=checked_frequency,
        path_count=len(checked_paths),
        periods_per_path=periods,
        unique_complete_paths=len(complete_counts),
        complete_path_duplicate_copies=len(checked_paths) - len(complete_counts),
        complete_path_duplicate_groups=complete_groups,
        maximum_complete_path_multiplicity=int(np.max(complete_multiplicities)),
        subsequence_window=window,
        subsequence_windows=int(digest_rows.size),
        unique_subsequences=unique_subsequences,
        repeated_subsequence_copies=int(digest_rows.size) - unique_subsequences,
        repeated_subsequence_groups=repeated_groups,
        maximum_subsequence_multiplicity=int(np.max(subsequence_counts)),
    )


def _frequency(value: object) -> Frequency:
    if value == "monthly":
        return "monthly"
    if value == "daily":
        return "daily"
    raise ValidationError("lean G5 diagnostics frequency must be monthly or daily")


def _return_matrix(value: ArrayLike, *, minimum_observations: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError("lean G5 returns are not numeric") from error
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValidationError("lean G5 returns must have shape (observations, 3)")
    if result.shape[0] < minimum_observations:
        raise ValidationError(
            f"lean G5 returns require at least {minimum_observations} observations"
        )
    if not bool(np.isfinite(result).all()):
        raise ValidationError("lean G5 returns must be finite")
    return np.asarray(result, dtype=np.float64)


def _state_vector(value: ArrayLike, *, expected_observations: int) -> IntArray:
    raw = np.asarray(value)
    if (
        raw.shape != (expected_observations,)
        or raw.dtype.kind not in {"i", "u"}
        or bool(((raw < 0) | (raw >= LEAN_G5_STATE_COUNT)).any())
    ):
        raise ValidationError("lean G5 state path is not an aligned K=4 sequence")
    return np.asarray(raw, dtype=np.int64)


def _cross_lag_correlation(values: FloatArray) -> FloatArray:
    left = values[:-1]
    right = values[1:]
    result = np.zeros((3, 3), dtype=np.float64)
    for row in range(3):
        x = left[:, row] - np.mean(left[:, row])
        x_sum = float(np.dot(x, x))
        for column in range(3):
            y = right[:, column] - np.mean(right[:, column])
            denominator = math.sqrt(x_sum * float(np.dot(y, y)))
            result[row, column] = (
                0.0 if denominator == 0.0 else float(np.dot(x, y) / denominator)
            )
    if not bool(np.isfinite(result).all()):
        raise ValidationError("lean G5 cross-lag correlation is non-finite")
    return result


def _rolling_sums(values: FloatArray, window: int) -> FloatArray:
    cumulative = np.concatenate(
        (np.asarray([0.0]), np.cumsum(values, dtype=np.float64))
    )
    result = cumulative[window:] - cumulative[:-window]
    if result.size < 2 or not bool(np.isfinite(result).all()):
        raise ValidationError("lean G5 rolling spread has insufficient support")
    return np.asarray(result, dtype=np.float64)


def _canonical_return_matrix(values: FloatArray) -> FloatArray:
    normalized = np.asarray(values, dtype="<f8", order="C").copy()
    normalized[normalized == 0.0] = 0.0
    return normalized


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"lean G5 {label} must be a positive integer")
    return value
