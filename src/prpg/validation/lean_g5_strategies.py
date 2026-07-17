"""Lean, price-only strategy probes for the G5 predictability check.

This module implements the six strategy families approved in Section 4.13.2.
It deliberately stops at fitting, deterministic path weights, and the
exposure-controlled alpha score.  Sampling banks, bootstrap decisions,
release authority, and command-line orchestration belong to later layers.

Inputs are synchronized log-return rows in the fixed order
``(equity, muni_bond, taxable_bond)``.  No API accepts macro data, hidden
states, source indices, or generator audit fields.  A weight applied to row
``t`` is computed only from rows before ``t``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError

FloatArray: TypeAlias = NDArray[np.float64]
Frequency: TypeAlias = Literal["monthly", "daily"]

LEAN_G5_STRATEGY_SCHEMA_ID: Final = "prpg-g5-lean-six-strategy-fit-v1"
LEAN_G5_RIDGE_PENALTY: Final = 1.0
LEAN_G5_ASSET_NAMES: Final = ("equity", "muni_bond", "taxable_bond")
LEAN_G5_STRATEGY_NAMES: Final = (
    "momentum_6m",
    "trend_12m",
    "reversal_1m",
    "taxable_vs_muni_6m",
    "linear_ridge",
    "quadratic_ridge",
)
LEAN_G5_FEATURE_NAMES: Final = (
    "return_1m_equity",
    "return_1m_muni_bond",
    "return_1m_taxable_bond",
    "momentum_6m_equity",
    "momentum_6m_muni_bond",
    "momentum_6m_taxable_bond",
    "trend_12m_equity",
    "trend_12m_muni_bond",
    "trend_12m_taxable_bond",
    "spread_6m_taxable_minus_muni",
)

_ASSET_COUNT: Final = 3
_STRATEGY_COUNT: Final = 6
_BASE_FEATURE_COUNT: Final = 10
_QUADRATIC_FEATURE_COUNT: Final = 65


@dataclass(frozen=True, slots=True)
class LeanG5StrategyModel:
    """One immutable development fit reused for every scored path."""

    frequency: Frequency
    periods_per_month: int
    rebalance_periods: int
    warmup_periods: int
    feature_center: FloatArray
    feature_scale: FloatArray
    linear_coefficients: FloatArray
    quadratic_coefficients: FloatArray
    development_average_weights: FloatArray
    development_paths: int
    development_examples: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class LeanG5PathWeights:
    """Six strategy weights aligned to eligible return rows."""

    first_return_index: int
    strategy_names: tuple[str, ...]
    values: FloatArray


@dataclass(frozen=True, slots=True)
class LeanG5PathScore:
    """Exposure-controlled alpha Sharpes for one historical or simulated path."""

    frequency: Frequency
    strategy_names: tuple[str, ...]
    observations: int
    alpha_sharpes: FloatArray
    static_exposure_only: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _FrequencySpec:
    frequency: Frequency
    periods_per_month: int
    rebalance_periods: int
    annualization: int

    @property
    def warmup_periods(self) -> int:
        return 12 * self.periods_per_month


_MONTHLY_SPEC: Final = _FrequencySpec("monthly", 1, 1, 12)
_DAILY_SPEC: Final = _FrequencySpec("daily", 21, 21, 252)


def fit_lean_g5_strategy_model(
    development_paths: Sequence[ArrayLike],
    *,
    frequency: Frequency,
) -> LeanG5StrategyModel:
    """Fit the linear and quadratic ridge probes exactly once.

    The normalized ridge objective is ``mean(SSE)/2 + ||beta||^2/2`` with an
    unpenalized intercept.  Each feature at decision row ``t`` uses only rows
    before ``t``; its target is the following one-month compounded simple
    return (one monthly row or 21 daily rows).
    """

    spec = _frequency_spec(frequency)
    paths = _development_paths(development_paths, spec)
    feature_batches: list[FloatArray] = []
    outcome_batches: list[FloatArray] = []
    for path in paths:
        features: list[FloatArray] = []
        path_outcomes: list[FloatArray] = []
        prefix = _prefix_sums(path)
        stop = path.shape[0] - spec.periods_per_month + 1
        for target in range(spec.warmup_periods, stop, spec.rebalance_periods):
            features.append(_base_features(prefix, target, spec))
            path_outcomes.append(
                np.expm1(prefix[target + spec.periods_per_month] - prefix[target])
            )
        feature_batches.append(np.vstack(features))
        outcome_batches.append(np.vstack(path_outcomes))

    base_features = np.vstack(feature_batches)
    development_outcomes = np.vstack(outcome_batches)
    center = np.mean(base_features, axis=0, dtype=np.float64)
    scale = np.std(base_features, axis=0, ddof=0, dtype=np.float64)
    if (
        center.shape != (_BASE_FEATURE_COUNT,)
        or scale.shape != (_BASE_FEATURE_COUNT,)
        or not bool(np.isfinite(center).all())
        or not bool(np.isfinite(scale).all())
        or bool((scale <= 1.0e-12).any())
    ):
        raise ValidationError("lean G5 development features are degenerate")
    standardized = (base_features - center) / scale
    linear_coefficients = _fit_normalized_ridge(standardized, development_outcomes)
    quadratic_coefficients = _fit_normalized_ridge(
        _quadratic_features(standardized), development_outcomes
    )

    center = _freeze(center)
    scale = _freeze(scale)
    linear_coefficients = _freeze(linear_coefficients)
    quadratic_coefficients = _freeze(quadratic_coefficients)
    all_weights = np.concatenate(
        tuple(
            _strategy_weights(
                path,
                spec,
                center,
                scale,
                linear_coefficients,
                quadratic_coefficients,
            )
            for path in paths
        ),
        axis=0,
    )
    average_weights = _freeze(np.mean(all_weights, axis=0, dtype=np.float64))
    identity = {
        "schema_id": LEAN_G5_STRATEGY_SCHEMA_ID,
        "frequency": spec.frequency,
        "periods_per_month": spec.periods_per_month,
        "rebalance_periods": spec.rebalance_periods,
        "warmup_periods": spec.warmup_periods,
        "ridge_penalty": LEAN_G5_RIDGE_PENALTY,
        "feature_names": LEAN_G5_FEATURE_NAMES,
        "strategy_names": LEAN_G5_STRATEGY_NAMES,
        "feature_center_sha256": _array_sha256(center),
        "feature_scale_sha256": _array_sha256(scale),
        "linear_coefficients_sha256": _array_sha256(linear_coefficients),
        "quadratic_coefficients_sha256": _array_sha256(quadratic_coefficients),
        "development_average_weights_sha256": _array_sha256(average_weights),
        "development_paths": len(paths),
        "development_examples": int(base_features.shape[0]),
    }
    return LeanG5StrategyModel(
        frequency=spec.frequency,
        periods_per_month=spec.periods_per_month,
        rebalance_periods=spec.rebalance_periods,
        warmup_periods=spec.warmup_periods,
        feature_center=center,
        feature_scale=scale,
        linear_coefficients=linear_coefficients,
        quadratic_coefficients=quadratic_coefficients,
        development_average_weights=average_weights,
        development_paths=len(paths),
        development_examples=int(base_features.shape[0]),
        fingerprint=_fingerprint(identity),
    )


def lean_g5_strategy_weights(
    model: LeanG5StrategyModel,
    log_returns: ArrayLike,
) -> LeanG5PathWeights:
    """Produce long-only weights without observing the return they are applied to."""

    checked_model = _model(model)
    path = _path(
        log_returns,
        label="score",
        minimum_rows=checked_model.warmup_periods + 1,
    )
    spec = _frequency_spec(checked_model.frequency)
    values = _strategy_weights(
        path,
        spec,
        checked_model.feature_center,
        checked_model.feature_scale,
        checked_model.linear_coefficients,
        checked_model.quadratic_coefficients,
    )
    return LeanG5PathWeights(
        first_return_index=spec.warmup_periods,
        strategy_names=LEAN_G5_STRATEGY_NAMES,
        values=_freeze(values),
    )


def score_lean_g5_path(
    model: LeanG5StrategyModel,
    log_returns: ArrayLike,
) -> LeanG5PathScore:
    """Score one path with the frozen development comparator.

    There is intentionally no historical/simulated switch.  Both kinds of
    path call this same function, and both receive a calculated score.  The
    active return for strategy ``s`` is
    ``(weight[t,s] - development_average_weight[s]) @ simple_return[t]``.
    It is regressed on an intercept and all three contemporaneous asset simple
    returns before annualizing the intercept by residual volatility.
    """

    checked_model = _model(model)
    path = _path(
        log_returns,
        label="score",
        minimum_rows=checked_model.warmup_periods + 5,
    )
    path_weights = lean_g5_strategy_weights(checked_model, path)
    simple_returns = np.expm1(path[path_weights.first_return_index :])
    design = np.column_stack(
        (np.ones(simple_returns.shape[0], dtype=np.float64), simple_returns)
    )
    if design.shape[0] <= design.shape[1] or np.linalg.matrix_rank(design) != 4:
        raise ValidationError("lean G5 alpha regression is rank-degenerate")
    active_returns = np.einsum(
        "tsa,ta->ts",
        path_weights.values - checked_model.development_average_weights,
        simple_returns,
        optimize=True,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, active_returns, rcond=None)
    residuals = active_returns - design @ coefficients
    residual_scale = np.sqrt(
        np.sum(residuals * residuals, axis=0, dtype=np.float64)
        / (design.shape[0] - design.shape[1])
    )
    if not bool(np.isfinite(residual_scale).all()):
        raise ValidationError("lean G5 alpha regression residual is non-finite")
    annualization = _frequency_spec(checked_model.frequency).annualization
    alpha_values = np.empty(_STRATEGY_COUNT, dtype=np.float64)
    static_exposure_only: list[bool] = []
    for strategy in range(_STRATEGY_COUNT):
        if residual_scale[strategy] > 1.0e-12:
            alpha_values[strategy] = (
                math.sqrt(annualization)
                * coefficients[0, strategy]
                / residual_scale[strategy]
            )
            static_exposure_only.append(False)
            continue

        # A strategy that holds one exactly constant weight vector has no
        # timing variation after contemporaneous asset exposures are removed.
        # Its alpha estimand is therefore zero, not an operational failure.
        # This narrow branch is deliberately unavailable to a dynamic strategy
        # or to any numerically non-zero intercept/residual degeneracy.
        weights = path_weights.values[:, strategy]
        is_static = bool(np.all(weights == weights[:1]))
        active_scale = max(
            1.0,
            float(np.max(np.abs(active_returns[:, strategy]))),
        )
        tolerance = (
            256.0 * np.finfo(np.float64).eps * active_scale * math.sqrt(design.shape[0])
        )
        residual_norm = float(np.linalg.norm(residuals[:, strategy]))
        if (
            not is_static
            or abs(float(coefficients[0, strategy])) > tolerance
            or residual_norm > tolerance
        ):
            raise ValidationError("lean G5 alpha regression residual is degenerate")
        alpha_values[strategy] = 0.0
        static_exposure_only.append(True)

    alpha_sharpes = _freeze(alpha_values)
    if alpha_sharpes.shape != (_STRATEGY_COUNT,) or not bool(
        np.isfinite(alpha_sharpes).all()
    ):
        raise ValidationError("lean G5 alpha score is invalid")
    return LeanG5PathScore(
        frequency=checked_model.frequency,
        strategy_names=LEAN_G5_STRATEGY_NAMES,
        observations=int(design.shape[0]),
        alpha_sharpes=alpha_sharpes,
        static_exposure_only=tuple(static_exposure_only),
    )


def _development_paths(
    values: Sequence[ArrayLike], spec: _FrequencySpec
) -> tuple[FloatArray, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValidationError("lean G5 development paths are required")
    paths = tuple(
        _path(
            value,
            label="development",
            minimum_rows=spec.warmup_periods + spec.periods_per_month,
        )
        for value in values
    )
    lengths = {path.shape[0] for path in paths}
    if len(lengths) != 1:
        raise ValidationError("lean G5 development paths must have equal history")
    return paths


def _strategy_weights(
    path: FloatArray,
    spec: _FrequencySpec,
    center: FloatArray,
    scale: FloatArray,
    linear_coefficients: FloatArray,
    quadratic_coefficients: FloatArray,
) -> FloatArray:
    prefix = _prefix_sums(path)
    weights = np.empty(
        (path.shape[0] - spec.warmup_periods, _STRATEGY_COUNT, _ASSET_COUNT),
        dtype=np.float64,
    )
    for target in range(spec.warmup_periods, path.shape[0], spec.rebalance_periods):
        feature = _base_features(prefix, target, spec)
        standardized = (feature - center) / scale
        traditional = (
            _winner_weights(feature[3:6]),
            _winner_weights(feature[6:9]),
            _winner_weights(-feature[0:3]),
            _bond_switch_weights(float(feature[9])),
        )
        linear_prediction = _ridge_prediction(standardized, linear_coefficients)
        quadratic_prediction = _ridge_prediction(
            _quadratic_features(standardized[np.newaxis, :])[0],
            quadratic_coefficients,
        )
        decision = np.vstack(
            (
                *traditional,
                _winner_weights(linear_prediction),
                _winner_weights(quadratic_prediction),
            )
        )
        start = target - spec.warmup_periods
        stop = min(start + spec.rebalance_periods, weights.shape[0])
        weights[start:stop] = decision
    if (
        not bool(np.isfinite(weights).all())
        or bool((weights < 0.0).any())
        or not bool(np.all(np.sum(weights, axis=2) == 1.0))
    ):
        raise ValidationError("lean G5 strategy weights are invalid")
    return weights


def _base_features(prefix: FloatArray, target: int, spec: _FrequencySpec) -> FloatArray:
    one = prefix[target] - prefix[target - spec.periods_per_month]
    six = prefix[target] - prefix[target - 6 * spec.periods_per_month]
    horizon = 12 * spec.periods_per_month
    levels = prefix[1:]
    trend = levels[target - 1] - np.mean(
        levels[target - horizon : target], axis=0, dtype=np.float64
    )
    result = np.asarray(
        np.concatenate((one, six, trend, (six[2] - six[1],))), dtype=np.float64
    )
    if result.shape != (_BASE_FEATURE_COUNT,) or not bool(np.isfinite(result).all()):
        raise ValidationError("lean G5 feature vector is invalid")
    return result


def _quadratic_features(standardized: FloatArray) -> FloatArray:
    if standardized.ndim != 2 or standardized.shape[1] != _BASE_FEATURE_COUNT:
        raise ValidationError("lean G5 quadratic feature input has the wrong shape")
    columns: list[FloatArray] = [standardized, standardized * standardized - 1.0]
    columns.extend(
        standardized[:, left] * standardized[:, right]
        for left in range(_BASE_FEATURE_COUNT)
        for right in range(left + 1, _BASE_FEATURE_COUNT)
    )
    result = np.column_stack(columns)
    if result.shape[1] != _QUADRATIC_FEATURE_COUNT or not bool(
        np.isfinite(result).all()
    ):
        raise ValidationError("lean G5 quadratic feature vector is invalid")
    return result


def _fit_normalized_ridge(features: FloatArray, outcomes: FloatArray) -> FloatArray:
    design = np.column_stack((np.ones(features.shape[0], dtype=np.float64), features))
    observations = design.shape[0]
    gram = (design.T @ design) / observations
    penalty = np.eye(gram.shape[0], dtype=np.float64) * LEAN_G5_RIDGE_PENALTY
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(
            gram + penalty, (design.T @ outcomes) / observations
        )
    except np.linalg.LinAlgError as error:
        raise ValidationError("lean G5 ridge fit failed") from error
    if not bool(np.isfinite(coefficients).all()):
        raise ValidationError("lean G5 ridge fit is non-finite")
    return coefficients


def _ridge_prediction(features: FloatArray, coefficients: FloatArray) -> FloatArray:
    design = np.concatenate(((1.0,), features))
    prediction = design @ coefficients
    if prediction.shape != (_ASSET_COUNT,) or not bool(np.isfinite(prediction).all()):
        raise ValidationError("lean G5 ridge prediction is invalid")
    return prediction


def _winner_weights(scores: FloatArray) -> FloatArray:
    if scores.shape != (_ASSET_COUNT,) or not bool(np.isfinite(scores).all()):
        raise ValidationError("lean G5 winner scores are invalid")
    winners = scores == np.max(scores)
    return np.asarray(
        winners.astype(np.float64) / int(np.sum(winners)), dtype=np.float64
    )


def _bond_switch_weights(spread: float) -> FloatArray:
    if not math.isfinite(spread):
        raise ValidationError("lean G5 bond-switch score is invalid")
    if spread > 0.0:
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if spread < 0.0:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    return np.asarray((0.0, 0.5, 0.5), dtype=np.float64)


def _model(value: object) -> LeanG5StrategyModel:
    if not isinstance(value, LeanG5StrategyModel):
        raise ValidationError("lean G5 scoring requires a fitted strategy model")
    spec = _frequency_spec(value.frequency)
    if (
        value.periods_per_month != spec.periods_per_month
        or value.rebalance_periods != spec.rebalance_periods
        or value.warmup_periods != spec.warmup_periods
        or value.feature_center.shape != (_BASE_FEATURE_COUNT,)
        or value.feature_scale.shape != (_BASE_FEATURE_COUNT,)
        or value.linear_coefficients.shape != (_BASE_FEATURE_COUNT + 1, _ASSET_COUNT)
        or value.quadratic_coefficients.shape
        != (_QUADRATIC_FEATURE_COUNT + 1, _ASSET_COUNT)
        or value.development_average_weights.shape != (_STRATEGY_COUNT, _ASSET_COUNT)
    ):
        raise ValidationError("lean G5 strategy model has invalid geometry")
    return value


def _frequency_spec(value: object) -> _FrequencySpec:
    if value == "monthly":
        return _MONTHLY_SPEC
    if value == "daily":
        return _DAILY_SPEC
    raise ValidationError("lean G5 frequency is invalid")


def _path(value: ArrayLike, *, label: str, minimum_rows: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"lean G5 {label} path must be numeric") from error
    if (
        result.ndim != 2
        or result.shape[1] != _ASSET_COUNT
        or result.shape[0] < minimum_rows
    ):
        raise ValidationError(f"lean G5 {label} path has the wrong shape")
    if not bool(np.isfinite(result).all()):
        raise ValidationError(f"lean G5 {label} path must be finite")
    return np.asarray(result, dtype=np.float64)


def _prefix_sums(path: FloatArray) -> FloatArray:
    return np.vstack(
        (
            np.zeros((_ASSET_COUNT,), dtype=np.float64),
            np.cumsum(path, axis=0, dtype=np.float64),
        )
    )


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _array_sha256(value: FloatArray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _fingerprint(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
