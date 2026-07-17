"""Lean inverse-volatility timing diagnostic for G5.

The timer observes only prior three-asset returns.  Monthly scoring uses a
six-row volatility window and rebalances every row; daily scoring uses a
63-session window and holds each decision for 21 sessions.  Historical and
simulated paths intentionally share the same scoring API.

This module stops at compact path-level improvement rows.  A later statistics
layer compares the simulated and historical row samples with a bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError

FloatArray: TypeAlias = NDArray[np.float64]
Frequency: TypeAlias = Literal["monthly", "daily"]

_ASSET_COUNT: Final = 3
_MINIMUM_SCALE: Final = 1.0e-12


@dataclass(frozen=True, slots=True)
class LeanG5InverseVolatilityModel:
    """Frozen timer geometry and development-only static comparator."""

    frequency: Frequency
    lookback_periods: int
    rebalance_periods: int
    annualization: int
    comparator_weights: FloatArray
    development_paths: int
    development_weight_rows: int


@dataclass(frozen=True, slots=True)
class LeanG5InverseVolatilityWeights:
    """Timer weights aligned to the return rows on which they are used."""

    first_return_index: int
    values: FloatArray


@dataclass(frozen=True, slots=True)
class LeanG5InverseVolatilityScore:
    """One path's timer-minus-static annualized zero-cash Sharpe."""

    frequency: Frequency
    observations: int
    timer_sharpe: float
    comparator_sharpe: float
    improvement: float


@dataclass(frozen=True, slots=True)
class LeanG5InverseVolatilityRows:
    """Compact path observations retained for a later two-sample bootstrap."""

    frequency: Frequency
    improvements: FloatArray

    @property
    def paths(self) -> int:
        return int(self.improvements.size)


@dataclass(frozen=True, slots=True)
class _FrequencySpec:
    frequency: Frequency
    lookback_periods: int
    rebalance_periods: int
    annualization: int
    development_paths: int


_MONTHLY_SPEC: Final = _FrequencySpec("monthly", 6, 1, 12, 100)
_DAILY_SPEC: Final = _FrequencySpec("daily", 63, 21, 252, 20)


def fit_lean_g5_inverse_volatility_model(
    development_paths: Sequence[ArrayLike],
    *,
    frequency: Frequency,
) -> LeanG5InverseVolatilityModel:
    """Fit only the development-average static timer weight.

    The registered design uses the same 100 monthly or 20 daily development
    paths as the six price-only probes.  Each eligible return-row weight is
    represented in the average, including a final partial daily holding span.
    """

    spec = _frequency_spec(frequency)
    paths = _development_paths(development_paths, spec)
    weight_sum = np.zeros(_ASSET_COUNT, dtype=np.float64)
    weight_rows = 0
    for path in paths:
        values = _timer_weights(path, spec)
        weight_sum += np.sum(values, axis=0, dtype=np.float64)
        weight_rows += values.shape[0]
    if weight_rows <= 0:
        raise ValidationError("lean G5 inverse-volatility development fit is empty")
    comparator = weight_sum / weight_rows
    comparator /= np.sum(comparator, dtype=np.float64)
    comparator = _validated_weights(
        comparator, label="development comparator", expected_rows=None
    )
    return LeanG5InverseVolatilityModel(
        frequency=spec.frequency,
        lookback_periods=spec.lookback_periods,
        rebalance_periods=spec.rebalance_periods,
        annualization=spec.annualization,
        comparator_weights=_freeze(comparator),
        development_paths=len(paths),
        development_weight_rows=weight_rows,
    )


def lean_g5_inverse_volatility_weights(
    model: LeanG5InverseVolatilityModel,
    log_returns: ArrayLike,
) -> LeanG5InverseVolatilityWeights:
    """Return long-only timer weights using no current or future return."""

    checked_model = _model(model)
    spec = _frequency_spec(checked_model.frequency)
    path = _path(
        log_returns,
        label="score",
        minimum_rows=spec.lookback_periods + 1,
    )
    values = _timer_weights(path, spec)
    return LeanG5InverseVolatilityWeights(
        first_return_index=spec.lookback_periods,
        values=_freeze(values),
    )


def score_lean_g5_inverse_volatility_path(
    model: LeanG5InverseVolatilityModel,
    log_returns: ArrayLike,
) -> LeanG5InverseVolatilityScore:
    """Score one historical or simulated path with the identical frozen rule."""

    checked_model = _model(model)
    path = _path(
        log_returns,
        label="score",
        minimum_rows=checked_model.lookback_periods + 2,
    )
    timed = lean_g5_inverse_volatility_weights(checked_model, path)
    simple_returns = _simple_returns(path)[timed.first_return_index :]
    timer_returns = np.einsum("ta,ta->t", timed.values, simple_returns, optimize=True)
    comparator_returns = simple_returns @ checked_model.comparator_weights
    timer_sharpe = _annualized_zero_cash_sharpe(
        timer_returns, checked_model.annualization, "timer"
    )
    comparator_sharpe = _annualized_zero_cash_sharpe(
        comparator_returns, checked_model.annualization, "comparator"
    )
    improvement = timer_sharpe - comparator_sharpe
    if not math.isfinite(improvement):
        raise ValidationError("lean G5 inverse-volatility improvement is non-finite")
    return LeanG5InverseVolatilityScore(
        frequency=checked_model.frequency,
        observations=int(simple_returns.shape[0]),
        timer_sharpe=timer_sharpe,
        comparator_sharpe=comparator_sharpe,
        improvement=improvement,
    )


def score_lean_g5_inverse_volatility_paths(
    model: LeanG5InverseVolatilityModel,
    paths: Sequence[ArrayLike],
) -> LeanG5InverseVolatilityRows:
    """Retain one improvement row per path for later bootstrap inference."""

    checked_model = _model(model)
    if not isinstance(paths, Sequence) or not paths:
        raise ValidationError("lean G5 inverse-volatility score paths are required")
    improvements = np.asarray(
        tuple(
            score_lean_g5_inverse_volatility_path(checked_model, path).improvement
            for path in paths
        ),
        dtype=np.float64,
    )
    if improvements.ndim != 1 or not bool(np.isfinite(improvements).all()):
        raise ValidationError("lean G5 inverse-volatility rows are invalid")
    return LeanG5InverseVolatilityRows(
        frequency=checked_model.frequency,
        improvements=_freeze(improvements),
    )


def _development_paths(
    values: Sequence[ArrayLike], spec: _FrequencySpec
) -> tuple[FloatArray, ...]:
    if not isinstance(values, Sequence) or len(values) != spec.development_paths:
        raise ValidationError(
            "lean G5 inverse-volatility development path count is invalid"
        )
    paths = tuple(
        _path(
            value,
            label="development",
            minimum_rows=spec.lookback_periods + 1,
        )
        for value in values
    )
    if len({path.shape[0] for path in paths}) != 1:
        raise ValidationError(
            "lean G5 inverse-volatility development paths need equal history"
        )
    return paths


def _timer_weights(path: FloatArray, spec: _FrequencySpec) -> FloatArray:
    simple_returns = _simple_returns(path)
    rows = path.shape[0] - spec.lookback_periods
    weights = np.empty((rows, _ASSET_COUNT), dtype=np.float64)
    for target in range(spec.lookback_periods, path.shape[0], spec.rebalance_periods):
        trailing = simple_returns[target - spec.lookback_periods : target]
        volatility = np.std(trailing, axis=0, ddof=1, dtype=np.float64)
        if (
            volatility.shape != (_ASSET_COUNT,)
            or not bool(np.isfinite(volatility).all())
            or bool((volatility <= _MINIMUM_SCALE).any())
        ):
            raise ValidationError(
                "lean G5 inverse-volatility trailing volatility is degenerate"
            )
        inverse = 1.0 / volatility
        decision = inverse / np.sum(inverse, dtype=np.float64)
        start = target - spec.lookback_periods
        stop = min(start + spec.rebalance_periods, rows)
        weights[start:stop] = decision
    return _validated_weights(weights, label="timer", expected_rows=rows)


def _annualized_zero_cash_sharpe(
    returns: FloatArray, annualization: int, label: str
) -> float:
    if returns.ndim != 1 or returns.size < 2 or not bool(np.isfinite(returns).all()):
        raise ValidationError(f"lean G5 inverse-volatility {label} returns are invalid")
    scale = float(np.std(returns, ddof=1, dtype=np.float64))
    if not math.isfinite(scale) or scale <= _MINIMUM_SCALE:
        raise ValidationError(
            f"lean G5 inverse-volatility {label} Sharpe denominator is degenerate"
        )
    result = (
        math.sqrt(annualization) * float(np.mean(returns, dtype=np.float64)) / scale
    )
    if not math.isfinite(result):
        raise ValidationError(
            f"lean G5 inverse-volatility {label} Sharpe is non-finite"
        )
    return result


def _model(value: object) -> LeanG5InverseVolatilityModel:
    if not isinstance(value, LeanG5InverseVolatilityModel):
        raise ValidationError("lean G5 inverse-volatility scoring needs a fitted model")
    spec = _frequency_spec(value.frequency)
    if (
        value.lookback_periods != spec.lookback_periods
        or value.rebalance_periods != spec.rebalance_periods
        or value.annualization != spec.annualization
        or value.development_paths != spec.development_paths
        or isinstance(value.development_weight_rows, bool)
        or not isinstance(value.development_weight_rows, int)
        or value.development_weight_rows <= 0
    ):
        raise ValidationError("lean G5 inverse-volatility model geometry is invalid")
    _validated_weights(
        value.comparator_weights,
        label="model comparator",
        expected_rows=None,
    )
    return value


def _validated_weights(
    value: ArrayLike, *, label: str, expected_rows: int | None
) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"lean G5 inverse-volatility {label} weights must be numeric"
        ) from error
    expected_shape = (
        (_ASSET_COUNT,) if expected_rows is None else (expected_rows, _ASSET_COUNT)
    )
    if (
        result.shape != expected_shape
        or not bool(np.isfinite(result).all())
        or bool((result <= 0.0).any())
    ):
        raise ValidationError(f"lean G5 inverse-volatility {label} weights are invalid")
    totals = np.sum(result, axis=-1, dtype=np.float64)
    if not bool(np.all(np.abs(totals - 1.0) <= 1.0e-12)):
        raise ValidationError(f"lean G5 inverse-volatility {label} weights are invalid")
    return result


def _frequency_spec(value: object) -> _FrequencySpec:
    if value == "monthly":
        return _MONTHLY_SPEC
    if value == "daily":
        return _DAILY_SPEC
    raise ValidationError("lean G5 inverse-volatility frequency is invalid")


def _path(value: ArrayLike, *, label: str, minimum_rows: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"lean G5 inverse-volatility {label} path must be numeric"
        ) from error
    if (
        result.ndim != 2
        or result.shape[1] != _ASSET_COUNT
        or result.shape[0] < minimum_rows
    ):
        raise ValidationError(
            f"lean G5 inverse-volatility {label} path has the wrong shape"
        )
    if not bool(np.isfinite(result).all()):
        raise ValidationError(f"lean G5 inverse-volatility {label} path must be finite")
    return result


def _simple_returns(path: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.expm1(path)
    if not bool(np.isfinite(result).all()) or bool((result <= -1.0).any()):
        raise ValidationError(
            "lean G5 inverse-volatility simple returns are non-finite"
        )
    return np.asarray(result, dtype=np.float64)


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
