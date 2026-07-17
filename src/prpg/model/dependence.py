"""Frozen-reference dependence diagnostics for the Phase-3 block gate.

This module implements the Section 4.6 Point-6 endpoint contract.  It is
deliberately independent of return generation: callers supply historical or
simulated return histories and the exact row continuation mask.  A reference
fit freezes the historical asset centers, scales, and correlation-PC1 loading;
every simulated history is evaluated with those values unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.inference import HolmResult, higher_quantile, holm_correction

Frequency: TypeAlias = Literal["monthly", "daily"]
FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

ASSET_NAMES = ("equity", "muni_bond", "taxable_bond")
DIAGNOSTIC_NAMES = (
    *ASSET_NAMES,
    "pc1",
    "taxable_minus_muni",
    "abs_pc1",
    "squared_pc1",
)
PAIR_NAMES = (
    "corr_equity_muni_bond",
    "corr_equity_taxable_bond",
    "corr_muni_bond_taxable_bond",
)
MONTHLY_LAGS = tuple(range(1, 13))
DAILY_LAGS = (1, 5, 21, 63)
CONSTANT_TOLERANCE = 1e-12
CORRELATION_CLIP = 2.0**-53


@dataclass(frozen=True, slots=True)
class DependenceReference:
    """Historical transformation and endpoint vector frozen for simulation."""

    frequency: Frequency
    asset_center: FloatArray
    asset_scale: FloatArray
    pc1_loadings: FloatArray
    labels: tuple[str, ...]
    raw_values: FloatArray
    transformed_values: FloatArray


@dataclass(frozen=True, slots=True)
class DependenceVector:
    """One dependence vector evaluated under a frozen reference transform."""

    frequency: Frequency
    labels: tuple[str, ...]
    raw_values: FloatArray
    transformed_values: FloatArray


@dataclass(frozen=True, slots=True)
class DependenceCell:
    """One exact two-sided max-z dependence-preservation cell."""

    observed_statistic: float
    replicate_statistics: FloatArray
    p_value: float
    critical_value_95: float
    center: FloatArray
    scale: FloatArray
    active_components: BoolArray
    repetitions: int


@dataclass(frozen=True, slots=True)
class DependenceGate:
    """Holm decision for the four selected-L artifact-frequency cells."""

    cell_order: tuple[str, ...]
    holm: HolmResult
    passed: bool


def fit_dependence_reference(
    returns: ArrayLike,
    continuation: ArrayLike,
    *,
    frequency: Frequency,
) -> DependenceReference:
    """Freeze historical transforms and calculate the historical endpoint."""

    values = _returns(returns)
    mask = _continuation(continuation, len(values))
    checked_frequency, lags = _frequency_lags(frequency)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or bool((scale <= CONSTANT_TOLERANCE).any()):
        raise ModelError("historical return scale is at or below the binding floor")
    standardized = (values - center) / scale
    correlation = standardized.T @ standardized / (len(values) - 1)
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigengap = float(eigenvalues[-1] - eigenvalues[-2])
    threshold = 1e-10 * max(1.0, abs(float(eigenvalues[-1])))
    if not np.isfinite(eigenvalues).all() or eigengap <= threshold:
        raise ModelError(
            "return correlation PC1 is not identified",
            details={"eigengap": eigengap, "threshold": threshold},
        )
    loadings = eigenvectors[:, -1].astype(np.float64, copy=True)
    if abs(float(loadings[0])) <= CONSTANT_TOLERANCE:
        raise ModelError("return correlation PC1 has no usable equity sign anchor")
    if loadings[0] < 0.0:
        loadings *= -1.0
    raw, transformed, labels = _dependence_values(
        values,
        mask,
        center.astype(np.float64),
        scale.astype(np.float64),
        loadings,
        lags,
    )
    center = center.astype(np.float64, copy=True)
    scale = scale.astype(np.float64, copy=True)
    _readonly(center, scale, loadings, raw, transformed)
    return DependenceReference(
        checked_frequency,
        center,
        scale,
        loadings,
        labels,
        raw,
        transformed,
    )


def evaluate_dependence_vector(
    returns: ArrayLike,
    continuation: ArrayLike,
    reference: DependenceReference,
) -> DependenceVector:
    """Evaluate a history without adapting the historical PC1 transform."""

    values = _returns(returns)
    mask = _continuation(continuation, len(values))
    frequency, lags = _frequency_lags(reference.frequency)
    center = _frozen_vector(reference.asset_center, "asset center")
    scale = _frozen_vector(reference.asset_scale, "asset scale")
    loadings = _frozen_vector(reference.pc1_loadings, "PC1 loadings")
    if center.shape != (3,) or scale.shape != (3,) or loadings.shape != (3,):
        raise ModelError("frozen dependence transform must contain three assets")
    if bool((scale <= CONSTANT_TOLERANCE).any()):
        raise ModelError("frozen dependence scale is at or below the binding floor")
    raw, transformed, labels = _dependence_values(
        values, mask, center, scale, loadings, lags
    )
    if labels != reference.labels:
        raise ModelError("frozen dependence endpoint order is inconsistent")
    _readonly(raw, transformed)
    return DependenceVector(frequency, labels, raw, transformed)


def dependence_max_statistic_cell(
    observed: ArrayLike,
    replicates: ArrayLike,
    *,
    expected_repetitions: int = 10_000,
    constant_tolerance: float = CONSTANT_TOLERANCE,
) -> DependenceCell:
    """Apply the exact two-sided max-z and plus-one empirical p-value."""

    _positive_integer(expected_repetitions, "expected_repetitions")
    if not np.isfinite(constant_tolerance) or constant_tolerance < 0.0:
        raise ModelError("constant tolerance must be finite and non-negative")
    target = np.asarray(observed, dtype=np.float64)
    samples = np.asarray(replicates, dtype=np.float64)
    if target.ndim != 1 or target.size == 0 or not np.isfinite(target).all():
        raise ModelError("observed dependence vector must be finite and non-empty")
    if samples.shape != (expected_repetitions, target.size):
        raise ModelError(
            "dependence replicate matrix has the wrong exact shape",
            details={
                "expected": [expected_repetitions, int(target.size)],
                "actual": list(samples.shape),
            },
        )
    if not np.isfinite(samples).all():
        raise ModelError("dependence replicate matrix contains non-finite values")
    center = samples.mean(axis=0)
    scale = samples.std(axis=0, ddof=1)
    active = scale > constant_tolerance
    for component in np.flatnonzero(~active):
        spread = float(np.max(np.abs(samples[:, component] - center[component])))
        if spread > constant_tolerance:
            raise ModelError(
                "constant dependence component has excessive replicate spread",
                details={"component": int(component), "spread": spread},
            )
        if abs(float(target[component] - center[component])) > constant_tolerance:
            raise ModelError(
                "historical dependence value fails a constant component",
                details={"component": int(component)},
            )
    if bool(active.any()):
        standardized = np.abs((samples[:, active] - center[active]) / scale[active])
        replicate_statistics = standardized.max(axis=1)
        observed_statistic = float(
            np.max(np.abs((target[active] - center[active]) / scale[active]))
        )
    else:
        replicate_statistics = np.zeros(expected_repetitions, dtype=np.float64)
        observed_statistic = 0.0
    p_value = float(
        (1 + np.count_nonzero(replicate_statistics >= observed_statistic))
        / (expected_repetitions + 1)
    )
    critical = higher_quantile(replicate_statistics, 0.95)
    center = center.astype(np.float64, copy=True)
    scale = scale.astype(np.float64, copy=True)
    active = active.astype(np.bool_, copy=True)
    replicate_statistics = replicate_statistics.astype(np.float64, copy=True)
    _readonly(center, scale, active, replicate_statistics)
    return DependenceCell(
        observed_statistic,
        replicate_statistics,
        p_value,
        critical,
        center,
        scale,
        active,
        expected_repetitions,
    )


def dependence_holm_gate(
    cells: tuple[DependenceCell, DependenceCell, DependenceCell, DependenceCell],
    *,
    alpha: float = 0.05,
) -> DependenceGate:
    """Holm-correct exactly design/production by monthly/daily cells."""

    if not isinstance(cells, tuple) or len(cells) != 4:
        raise ModelError("dependence gate requires exactly four ordered cells")
    cell_order = (
        "design_monthly",
        "design_daily",
        "production_monthly",
        "production_daily",
    )
    result = holm_correction([cell.p_value for cell in cells], alpha=alpha)
    return DependenceGate(cell_order, result, not bool(result.rejected.any()))


def feasible_sensitivity_lengths(
    selected_length: int, *, frequency: Frequency
) -> tuple[int, ...]:
    """Return unique feasible values from ``{L-1,L,L+1}`` in fixed order."""

    _positive_integer(selected_length, "selected_length")
    checked_frequency, _ = _frequency_lags(frequency)
    lower, upper = (
        MONTHLY_BLOCK_LENGTH_BOUNDS
        if checked_frequency == "monthly"
        else DAILY_BLOCK_LENGTH_BOUNDS
    )
    if not lower <= selected_length <= upper:
        raise ModelError("selected block length is outside the approved bounds")
    return tuple(
        value
        for value in (selected_length - 1, selected_length, selected_length + 1)
        if lower <= value <= upper
    )


def _dependence_values(
    values: FloatArray,
    continuation: BoolArray,
    center: FloatArray,
    scale: FloatArray,
    loadings: FloatArray,
    lags: tuple[int, ...],
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    standardized_assets = (values - center) / scale
    pc1 = standardized_assets @ loadings
    diagnostics = np.column_stack(
        (
            values,
            pc1,
            values[:, 2] - values[:, 1],
            np.abs(pc1),
            np.square(pc1),
        )
    )
    raw_values: list[float] = []
    labels: list[str] = list(PAIR_NAMES)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        raw_values.append(_correlation(values[:, left], values[:, right]))
    segment_ids = np.cumsum(~continuation, dtype=np.int64) - 1
    for diagnostic, name in enumerate(DIAGNOSTIC_NAMES):
        series = diagnostics[:, diagnostic]
        for lag in lags:
            labels.append(f"acf_{name}_lag_{lag}")
            raw_values.append(_available_pair_acf(series, segment_ids, lag))
    raw = np.asarray(raw_values, dtype=np.float64)
    transformed = np.arctanh(
        np.clip(raw, -1.0 + CORRELATION_CLIP, 1.0 - CORRELATION_CLIP)
    )
    if not np.isfinite(raw).all() or not np.isfinite(transformed).all():
        raise ModelError("dependence vector contains a non-finite endpoint")
    expected = 3 + len(DIAGNOSTIC_NAMES) * len(lags)
    if raw.shape != (expected,) or len(labels) != expected:
        raise AssertionError("dependence endpoint order is inconsistent")
    return raw, transformed.astype(np.float64), tuple(labels)


def _available_pair_acf(
    series: FloatArray, segment_ids: NDArray[np.int64], lag: int
) -> float:
    centered = series - float(np.mean(series))
    valid = segment_ids[lag:] == segment_ids[:-lag]
    if not bool(valid.any()):
        raise ModelError(
            "dependence ACF has no available within-segment pairs",
            details={"lag": lag},
        )
    current = centered[lag:][valid]
    previous = centered[:-lag][valid]
    denominator = float(np.sqrt(np.dot(current, current) * np.dot(previous, previous)))
    if not np.isfinite(denominator) or denominator <= CONSTANT_TOLERANCE:
        raise ModelError(
            "dependence ACF denominator is unavailable",
            details={"lag": lag},
        )
    return float(np.clip(np.dot(current, previous) / denominator, -1.0, 1.0))


def _correlation(left: FloatArray, right: FloatArray) -> float:
    x = left - float(np.mean(left))
    y = right - float(np.mean(right))
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if not np.isfinite(denominator) or denominator <= CONSTANT_TOLERANCE:
        raise ModelError("contemporaneous correlation denominator is unavailable")
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _returns(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < 2:
        raise ModelError("returns must be a matrix with three columns and two rows")
    if not np.isfinite(result).all():
        raise ModelError("returns contain non-finite values")
    return result


def _continuation(values: ArrayLike, observations: int) -> BoolArray:
    result = np.asarray(values)
    if result.shape != (observations,) or result.dtype.kind != "b":
        raise ModelError("continuation must be a row boolean mask matching returns")
    if bool(result[0]):
        raise ModelError("the first continuation value must be false")
    return result.astype(np.bool_, copy=False)


def _frequency_lags(frequency: str) -> tuple[Frequency, tuple[int, ...]]:
    if frequency == "monthly":
        return "monthly", MONTHLY_LAGS
    if frequency == "daily":
        return "daily", DAILY_LAGS
    raise ModelError("dependence frequency must be monthly or daily")


def _frozen_vector(values: ArrayLike, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ModelError(f"frozen {label} must be a finite vector")
    return result


def _positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
