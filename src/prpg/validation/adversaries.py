"""Fixed development-trained adversaries for the G5-A qualification family.

The model is deliberately small: two fixed-ridge predictors plus one exact
history lookup.  Fitting sees development paths only.  Scoring is walk-forward
on one qualification path and never tunes a feature, ridge value, or exposure
using qualification outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError

FloatArray: TypeAlias = NDArray[np.float64]
Frequency: TypeAlias = Literal["monthly", "daily"]

G5_ADVERSARY_SCHEMA_ID: Final = "prpg-g5-fixed-adversaries-v1"
G5_RIDGE_PENALTY: Final = 1.0e-3
G5_MEMORIZATION_HORIZON: Final = {"monthly": 6, "daily": 63}


@dataclass(frozen=True, slots=True)
class G5AdversaryModel:
    """One immutable frequency-specific development fit."""

    frequency: Frequency
    linear_center: FloatArray
    linear_scale: FloatArray
    linear_coefficients: FloatArray
    quadratic_center: FloatArray
    quadratic_scale: FloatArray
    quadratic_coefficients: FloatArray
    outcome_scale: FloatArray
    memorization_horizon: int
    memorized_outcomes: Mapping[bytes, tuple[float, float, float]]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5AdversaryFitPair:
    """The exact development fit reused for untouched qualification and G7."""

    monthly: G5AdversaryModel
    daily: G5AdversaryModel
    specification_fingerprint: str
    fingerprint: str


def fit_g5_adversary_pair(
    monthly_development_paths: Sequence[ArrayLike],
    daily_development_paths: Sequence[ArrayLike],
) -> G5AdversaryFitPair:
    """Fit and bind the two base-frequency adversary models once."""

    monthly = fit_g5_adversary_model(monthly_development_paths, frequency="monthly")
    daily = fit_g5_adversary_model(daily_development_paths, frequency="daily")
    identity = {
        "specification_fingerprint": G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
        "monthly_model_fingerprint": monthly.fingerprint,
        "daily_model_fingerprint": daily.fingerprint,
    }
    return G5AdversaryFitPair(
        monthly=monthly,
        daily=daily,
        specification_fingerprint=G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
        fingerprint=_fingerprint(identity),
    )


def fit_g5_adversary_model(
    development_paths: Sequence[ArrayLike],
    *,
    frequency: Frequency,
) -> G5AdversaryModel:
    """Fit all registered adversaries from development histories only."""

    checked_frequency = _frequency(frequency)
    paths = _paths(development_paths, "development")
    lagged = np.vstack([path[:-1] for path in paths])
    outcomes = np.vstack([path[1:] for path in paths])
    linear = _linear_features(lagged)
    quadratic = _quadratic_features(lagged)
    linear_center, linear_scale, linear_coefficients = _fit_ridge(linear, outcomes)
    quadratic_center, quadratic_scale, quadratic_coefficients = _fit_ridge(
        quadratic, outcomes
    )
    outcome_scale = np.std(outcomes, axis=0, ddof=1)
    if bool((outcome_scale <= 1e-12).any()) or not bool(
        np.isfinite(outcome_scale).all()
    ):
        raise ValidationError("G5 adversary development outcomes are degenerate")
    horizon = G5_MEMORIZATION_HORIZON[checked_frequency]
    memorized = _fit_memorization(paths, horizon)
    frozen_outcome_scale = _freeze(outcome_scale)
    identity = {
        "schema_id": G5_ADVERSARY_SCHEMA_ID,
        "frequency": checked_frequency,
        "ridge_penalty": G5_RIDGE_PENALTY,
        "linear_center_sha256": _array_sha256(linear_center),
        "linear_scale_sha256": _array_sha256(linear_scale),
        "linear_coefficients_sha256": _array_sha256(linear_coefficients),
        "quadratic_center_sha256": _array_sha256(quadratic_center),
        "quadratic_scale_sha256": _array_sha256(quadratic_scale),
        "quadratic_coefficients_sha256": _array_sha256(quadratic_coefficients),
        "outcome_scale_sha256": _array_sha256(frozen_outcome_scale),
        "memorization_horizon": horizon,
        "memorization_digest": _memorization_digest(memorized),
    }
    return G5AdversaryModel(
        frequency=checked_frequency,
        linear_center=linear_center,
        linear_scale=linear_scale,
        linear_coefficients=linear_coefficients,
        quadratic_center=quadratic_center,
        quadratic_scale=quadratic_scale,
        quadratic_coefficients=quadratic_coefficients,
        outcome_scale=frozen_outcome_scale,
        memorization_horizon=horizon,
        memorized_outcomes=MappingProxyType(memorized),
        fingerprint=_fingerprint(identity),
    )


def score_g5_adversary_path(
    model: G5AdversaryModel,
    qualification_path: ArrayLike,
) -> FloatArray:
    """Return linear, quadratic, then memorization Sharpes (three each)."""

    if not isinstance(model, G5AdversaryModel):
        raise ValidationError("G5 adversary score requires a fitted model")
    path = _path(qualification_path, "qualification")
    lagged = path[:-1]
    outcomes = path[1:]
    linear_prediction = _predict(
        _linear_features(lagged),
        model.linear_center,
        model.linear_scale,
        model.linear_coefficients,
    )
    quadratic_prediction = _predict(
        _quadratic_features(lagged),
        model.quadratic_center,
        model.quadratic_scale,
        model.quadratic_coefficients,
    )
    linear = _payoff_sharpes(
        _controlled_exposure(linear_prediction, model.outcome_scale),
        outcomes,
        model.frequency,
    )
    quadratic = _payoff_sharpes(
        _controlled_exposure(quadratic_prediction, model.outcome_scale),
        outcomes,
        model.frequency,
    )
    memorization_exposure = _walk_forward_memorization_exposure(model, path)
    memorization = _payoff_sharpes(memorization_exposure[1:], outcomes, model.frequency)
    result = np.concatenate((linear, quadratic, memorization))
    if result.shape != (9,) or not bool(np.isfinite(result).all()):
        raise ValidationError("G5 adversary score is invalid")
    return _freeze(result)


def _fit_ridge(
    features: FloatArray, outcomes: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - center) / scale
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * G5_RIDGE_PENALTY
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(gram + penalty, design.T @ outcomes)
    except np.linalg.LinAlgError as error:
        raise ValidationError("G5 adversary ridge fit failed") from error
    if not bool(np.isfinite(coefficients).all()):
        raise ValidationError("G5 adversary ridge fit is non-finite")
    return _freeze(center), _freeze(scale), _freeze(coefficients)


def _predict(
    features: FloatArray,
    center: FloatArray,
    scale: FloatArray,
    coefficients: FloatArray,
) -> FloatArray:
    standardized = (features - center) / scale
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    prediction = design @ coefficients
    if (
        prediction.ndim != 2
        or prediction.shape[1] != 3
        or not bool(np.isfinite(prediction).all())
    ):
        raise ValidationError("G5 adversary prediction is invalid")
    return prediction


def _quadratic_features(lagged: FloatArray) -> FloatArray:
    signs = np.where(lagged >= 0.0, 1.0, -1.0)
    return np.column_stack(
        (
            signs[:, 0] * signs[:, 1],
            signs[:, 0] * signs[:, 2],
            signs[:, 1] * signs[:, 2],
        )
    )


def _linear_features(lagged: FloatArray) -> FloatArray:
    return np.where(lagged >= 0.0, 1.0, -1.0)


def _controlled_exposure(prediction: FloatArray, scale: FloatArray) -> FloatArray:
    result = np.tanh(prediction / scale)
    if not bool(np.isfinite(result).all()) or bool((np.abs(result) > 1.0).any()):
        raise ValidationError("G5 adversary exposure control failed")
    return result


def _payoff_sharpes(
    exposure: FloatArray,
    outcomes: FloatArray,
    frequency: Frequency,
) -> FloatArray:
    if exposure.shape != outcomes.shape:
        raise ValidationError("G5 adversary exposure/outcome geometry differs")
    payoff = exposure * outcomes
    scale = np.std(payoff, axis=0, ddof=1)
    mean = np.mean(payoff, axis=0)
    sharpes = np.zeros(3, dtype=np.float64)
    valid = scale > 1e-15
    annualization = 12.0 if frequency == "monthly" else 252.0
    sharpes[valid] = math.sqrt(annualization) * mean[valid] / scale[valid]
    return sharpes


def _fit_memorization(
    paths: tuple[FloatArray, ...], horizon: int
) -> dict[bytes, tuple[float, float, float]]:
    learned: dict[bytes, tuple[float, float, float]] = {}
    ambiguous: set[bytes] = set()
    for path in paths:
        if path.shape[0] <= horizon:
            raise ValidationError("development path is too short for memorization")
        for stop in range(horizon, path.shape[0]):
            key = _history_key(path[stop - horizon : stop])
            outcome = (
                float(path[stop, 0]),
                float(path[stop, 1]),
                float(path[stop, 2]),
            )
            prior = learned.get(key)
            if prior is None and key not in ambiguous:
                learned[key] = outcome
            elif prior != outcome:
                learned.pop(key, None)
                ambiguous.add(key)
    return learned


def _walk_forward_memorization_exposure(
    model: G5AdversaryModel, path: FloatArray
) -> FloatArray:
    horizon = model.memorization_horizon
    learned: dict[bytes, tuple[float, float, float]] = {}
    ambiguous: set[bytes] = set()
    exposure = np.zeros_like(path)
    for stop in range(horizon, path.shape[0]):
        key = _history_key(path[stop - horizon : stop])
        prediction = learned.get(key)
        if prediction is None and key not in ambiguous:
            prediction = model.memorized_outcomes.get(key)
        if prediction is not None:
            exposure[stop] = np.tanh(
                np.asarray(prediction, dtype=np.float64) / model.outcome_scale
            )
        outcome = (
            float(path[stop, 0]),
            float(path[stop, 1]),
            float(path[stop, 2]),
        )
        prior = learned.get(key)
        if prior is None and key not in ambiguous:
            learned[key] = outcome
        elif prior != outcome:
            learned.pop(key, None)
            ambiguous.add(key)
    return exposure


def _history_key(values: FloatArray) -> bytes:
    little = np.asarray(values, dtype="<f8", order="C")
    return hashlib.blake2b(little.tobytes(order="C"), digest_size=16).digest()


def _memorization_digest(
    values: Mapping[bytes, tuple[float, float, float]],
) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key)
        digest.update(np.asarray(values[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def _paths(values: Sequence[ArrayLike], label: str) -> tuple[FloatArray, ...]:
    if not isinstance(values, Sequence) or not values:
        raise ValidationError(f"G5 adversary {label} paths are required")
    return tuple(_path(value, label) for value in values)


def _path(value: ArrayLike, label: str) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"G5 adversary {label} path must be numeric") from error
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < 8:
        raise ValidationError(f"G5 adversary {label} path has the wrong shape")
    if not bool(np.isfinite(result).all()):
        raise ValidationError(f"G5 adversary {label} path must be finite")
    return np.asarray(result, dtype=np.float64)


def _frequency(value: object) -> Frequency:
    if value not in {"monthly", "daily"}:
        raise ValidationError("G5 adversary frequency is invalid")
    return value  # type: ignore[return-value]


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


G5_ADVERSARY_SPECIFICATION_FINGERPRINT: Final = _fingerprint(
    {
        "schema_id": G5_ADVERSARY_SCHEMA_ID,
        "ridge_penalty": G5_RIDGE_PENALTY,
        "linear_features": "intercept_plus_three_fixed_lag1_return_signs",
        "quadratic_features": ("three_fixed_pairwise_sign_interactions"),
        "exposure_control": "componentwise_tanh_by_development_outcome_scale",
        "memorization_horizons": G5_MEMORIZATION_HORIZON,
        "memorization": "development_exact_history_plus_own_prefix_walk_forward",
        "outputs": "linear3_quadratic3_memorization3_annualized_payoff_sharpe",
    }
)
