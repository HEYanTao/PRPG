"""Small, deterministic statistics layer for the lean G5 alpha gate.

The inputs are already-scored paths: six monthly alpha columns, six daily
alpha columns, per-path inverse-volatility improvements, and identically
scored historical controls.  This module does not generate paths, fit
strategies, derive seeds, define G5-B/C/D metrics, or grant release authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from prpg.errors import ValidationError
from prpg.validation.lean_g5_inverse_volatility import LeanG5InverseVolatilityRows
from prpg.validation.lean_g5_strategies import LEAN_G5_STRATEGY_NAMES
from prpg.validation.statistics import HolmAudit, holm_decisions

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Mode: TypeAlias = Literal["production", "pilot", "test"]
Frequency: TypeAlias = Literal["monthly", "daily"]

LEAN_G5_MONTHLY_PATHS: Final = 500
LEAN_G5_DAILY_PATHS: Final = 200
LEAN_G5_RESAMPLES: Final = 10_000
LEAN_G5_COMPONENTS_PER_FREQUENCY: Final = 6
LEAN_G5_EQUIVALENCE_CONFIDENCE: Final = 0.90
LEAN_G5_EQUIVALENCE_MARGIN: Final = 0.10
LEAN_G5_LEAKAGE_CONFIDENCE: Final = 0.95
LEAN_G5_LEAKAGE_CAP: Final = 0.10
LEAN_G5_INVERSE_VOL_CAP: Final = 0.20
LEAN_G5_HOLM_FAMILIES: Final = ("G5-A", "G5-B", "G5-C", "G5-D")
LEAN_G5_COMPONENT_NAMES: Final = tuple(
    f"{frequency}.{strategy}"
    for frequency in ("monthly", "daily")
    for strategy in LEAN_G5_STRATEGY_NAMES
)

_PILOT_COUNTS: Final = (100, 40, 1_000)
_PRODUCTION_COUNTS: Final = (500, 200, 10_000)
_FREQUENCIES: Final[tuple[Frequency, Frequency]] = ("monthly", "daily")
_BATCH: Final = 128


@dataclass(frozen=True, slots=True)
class LeanG5Geometry:
    """Registered counts with an explicit canonicality label."""

    monthly_paths: int
    daily_paths: int
    resamples: int
    mode: Mode

    def __post_init__(self) -> None:
        counts = (self.monthly_paths, self.daily_paths, self.resamples)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in counts
        ):
            raise ValidationError("lean G5 geometry counts must be integers >= 2")
        expected = {
            "production": _PRODUCTION_COUNTS,
            "pilot": _PILOT_COUNTS,
        }.get(self.mode)
        if expected is not None and counts != expected:
            formatted = "/".join(f"{value:,}" for value in expected)
            raise ValidationError(
                f"lean G5 {self.mode} geometry must be exactly {formatted}"
            )
        if self.mode not in ("production", "pilot", "test"):
            raise ValidationError("lean G5 geometry mode is invalid")

    @property
    def canonical(self) -> bool:
        return self.mode == "production"

    @classmethod
    def production(cls) -> LeanG5Geometry:
        return cls(*_PRODUCTION_COUNTS, mode="production")

    @classmethod
    def pilot(cls) -> LeanG5Geometry:
        return cls(*_PILOT_COUNTS, mode="pilot")

    @classmethod
    def test(
        cls, *, monthly_paths: int, daily_paths: int, resamples: int
    ) -> LeanG5Geometry:
        return cls(monthly_paths, daily_paths, resamples, mode="test")


LEAN_G5_PRODUCTION_GEOMETRY: Final = LeanG5Geometry.production()
LEAN_G5_PILOT_GEOMETRY: Final = LeanG5Geometry.pilot()


@dataclass(frozen=True, slots=True)
class LeanG5AlphaRows:
    """Pre-scored rows in ``LEAN_G5_STRATEGY_NAMES`` order."""

    monthly: ArrayLike
    daily: ArrayLike


@dataclass(frozen=True, slots=True)
class LeanG5RegisteredResampleSeeds:
    """Four seeds registered by the caller; no seed is derived here."""

    simulated_monthly: int
    simulated_daily: int
    historical_monthly: int
    historical_daily: int


@dataclass(frozen=True, slots=True)
class LeanG5ResampleIndices:
    simulated_monthly: ArrayLike
    simulated_daily: ArrayLike
    historical_monthly: ArrayLike
    historical_daily: ArrayLike


@dataclass(frozen=True, slots=True)
class HigherEmpiricalQuantile:
    probability: float
    sample_count: int
    method: Literal["higher"]
    zero_based_index: int
    one_based_rank: int
    value: float


@dataclass(frozen=True, slots=True)
class PlusOnePValue:
    observed: float
    exceedances: int
    resamples: int
    p_value: float


@dataclass(frozen=True, slots=True)
class AlphaEquivalenceAudit:
    geometry: LeanG5Geometry
    confidence: float
    margin: float
    critical_value: HigherEmpiricalQuantile
    names: tuple[str, ...]
    estimates: tuple[float, ...]
    standard_errors: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    component_passed: tuple[bool, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class LeakageGapAudit:
    geometry: LeanG5Geometry
    confidence: float
    cap: float
    critical_value: HigherEmpiricalQuantile
    names: tuple[str, ...]
    simulated_estimates: tuple[float, ...]
    historical_estimates: tuple[float, ...]
    gaps: tuple[float, ...]
    standard_errors: tuple[float, ...]
    upper: tuple[float, ...]
    component_passed: tuple[bool, ...]
    passed: bool

    @property
    def maximum_upper(self) -> float:
        return max(self.upper)


@dataclass(frozen=True, slots=True)
class InverseVolatilityGapAudit:
    geometry: LeanG5Geometry
    confidence: float
    cap: float
    critical_value: HigherEmpiricalQuantile
    frequencies: tuple[Frequency, Frequency]
    simulated_estimates: tuple[float, float]
    historical_estimates: tuple[float, float]
    gaps: tuple[float, float]
    standard_errors: tuple[float, float]
    upper: tuple[float, float]
    component_passed: tuple[bool, bool]
    passed: bool

    @property
    def maximum_upper(self) -> float:
        return max(self.upper)


@dataclass(frozen=True, slots=True)
class FixedClopperPearsonAudit:
    exceedances: int
    trials: int
    estimate: float
    confidence: float
    lower: float
    upper: float
    maximum_distance: float
    permitted_distance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class LeanG5AlphaAudit:
    geometry: LeanG5Geometry
    equivalence: AlphaEquivalenceAudit
    leakage_gap: LeakageGapAudit
    inverse_volatility: InverseVolatilityGapAudit
    passed: bool


def resample_indices_from_registered_seeds(
    seeds: LeanG5RegisteredResampleSeeds,
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> LeanG5ResampleIndices:
    """Draw indices with PCG64DXSM directly from four registered seeds."""

    seed_values = tuple(
        _seed(value)
        for value in (
            seeds.simulated_monthly,
            seeds.simulated_daily,
            seeds.historical_monthly,
            seeds.historical_daily,
        )
    )
    if len(set(seed_values)) != len(seed_values):
        raise ValidationError("lean G5 registered resample seeds must be distinct")

    def draw(seed: int, paths: int) -> IntArray:
        generator = np.random.Generator(np.random.PCG64DXSM(seed))
        result = generator.integers(
            0, paths, size=(geometry.resamples, paths), dtype=np.int64
        )
        result.setflags(write=False)
        return result

    return LeanG5ResampleIndices(
        draw(seed_values[0], geometry.monthly_paths),
        draw(seed_values[1], geometry.daily_paths),
        draw(seed_values[2], geometry.monthly_paths),
        draw(seed_values[3], geometry.daily_paths),
    )


def higher_empirical_quantile(
    statistics: ArrayLike,
    probability: float,
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> HigherEmpiricalQuantile:
    """Return the exact ``method='higher'`` registered order statistic."""

    values = _vector(statistics, geometry.resamples)
    checked_probability = _number(probability, "probability")
    if not 0.0 < checked_probability < 1.0:
        raise ValidationError("lean G5 probability must be between zero and one")
    index = math.ceil(checked_probability * (values.size - 1))
    value = float(np.partition(values.copy(), index)[index])
    return HigherEmpiricalQuantile(
        checked_probability, int(values.size), "higher", index, index + 1, value
    )


def plus_one_upper_tail_p_value(
    null_statistics: ArrayLike,
    observed: float,
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> PlusOnePValue:
    """Return ``(count(null >= observed) + 1) / (B + 1)``, including ties."""

    values = _vector(null_statistics, geometry.resamples)
    checked_observed = _number(observed, "observed statistic")
    exceedances = int(np.count_nonzero(values >= checked_observed))
    return PlusOnePValue(
        checked_observed,
        exceedances,
        geometry.resamples,
        (exceedances + 1) / (geometry.resamples + 1),
    )


def four_family_holm(p_values: Mapping[str, float]) -> HolmAudit:
    """Apply 5% Holm correction to exactly G5-A/B/C/D."""

    if not isinstance(p_values, Mapping) or set(p_values) != set(LEAN_G5_HOLM_FAMILIES):
        raise ValidationError("lean G5 Holm requires exactly G5-A/B/C/D")
    return holm_decisions(p_values)


def fixed_clopper_pearson_precision_audit() -> FixedClopperPearsonAudit:
    """Audit the registered 500 exceedances in 10,000 draws once."""

    lower = float(beta.ppf(0.025, 500, 9_501))
    upper = float(beta.ppf(0.975, 501, 9_500))
    distance = max(0.05 - lower, upper - 0.05)
    return FixedClopperPearsonAudit(
        500,
        10_000,
        0.05,
        0.95,
        lower,
        upper,
        distance,
        0.005,
        distance < 0.005,
    )


def simultaneous_alpha_equivalence(
    simulated: LeanG5AlphaRows,
    indices: LeanG5ResampleIndices,
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> AlphaEquivalenceAudit:
    """Compute one simultaneous 90% max-t interval for all twelve alphas."""

    monthly, daily = _rows(simulated, geometry, "simulated")
    monthly_indices = _indices(
        indices.simulated_monthly, geometry.resamples, geometry.monthly_paths
    )
    daily_indices = _indices(
        indices.simulated_daily, geometry.resamples, geometry.daily_paths
    )
    monthly_mean, monthly_se = _mean_se(monthly)
    daily_mean, daily_se = _mean_se(daily)
    monthly_boot_mean, monthly_boot_se = _bootstrap_mean_se(monthly, monthly_indices)
    daily_boot_mean, daily_boot_se = _bootstrap_mean_se(daily, daily_indices)
    estimates = np.concatenate((monthly_mean, daily_mean))
    standard_errors = np.concatenate((monthly_se, daily_se))
    t_values = np.concatenate(
        (
            (monthly_boot_mean - monthly_mean) / monthly_boot_se,
            (daily_boot_mean - daily_mean) / daily_boot_se,
        ),
        axis=1,
    )
    critical = higher_empirical_quantile(
        np.max(np.abs(t_values), axis=1),
        LEAN_G5_EQUIVALENCE_CONFIDENCE,
        geometry=geometry,
    )
    lower = estimates - critical.value * standard_errors
    upper = estimates + critical.value * standard_errors
    component_passed = tuple(
        bool(low >= -LEAN_G5_EQUIVALENCE_MARGIN and high <= LEAN_G5_EQUIVALENCE_MARGIN)
        for low, high in zip(lower, upper, strict=True)
    )
    return AlphaEquivalenceAudit(
        geometry,
        LEAN_G5_EQUIVALENCE_CONFIDENCE,
        LEAN_G5_EQUIVALENCE_MARGIN,
        critical,
        LEAN_G5_COMPONENT_NAMES,
        _floats(estimates),
        _floats(standard_errors),
        _floats(lower),
        _floats(upper),
        component_passed,
        all(component_passed),
    )


def simultaneous_leakage_gap(
    simulated: LeanG5AlphaRows,
    historical: LeanG5AlphaRows,
    indices: LeanG5ResampleIndices,
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> LeakageGapAudit:
    """Compute simultaneous 95% upper bounds for simulated-control gaps."""

    simulated_rows = _rows(simulated, geometry, "simulated")
    historical_rows = _rows(historical, geometry, "historical")
    index_matrices = (
        _indices(indices.simulated_monthly, geometry.resamples, geometry.monthly_paths),
        _indices(indices.simulated_daily, geometry.resamples, geometry.daily_paths),
        _indices(
            indices.historical_monthly, geometry.resamples, geometry.monthly_paths
        ),
        _indices(indices.historical_daily, geometry.resamples, geometry.daily_paths),
    )
    matrices = (*simulated_rows, *historical_rows)
    point = tuple(_mean_se(matrix) for matrix in matrices)
    boot = tuple(
        _bootstrap_mean_se(matrix, index)
        for matrix, index in zip(matrices, index_matrices, strict=True)
    )
    simulated_mean = np.concatenate((point[0][0], point[1][0]))
    historical_mean = np.concatenate((point[2][0], point[3][0]))
    gaps = simulated_mean - historical_mean
    standard_errors = np.sqrt(
        np.concatenate((point[0][1] ** 2, point[1][1] ** 2))
        + np.concatenate((point[2][1] ** 2, point[3][1] ** 2))
    )
    boot_simulated = np.concatenate((boot[0][0], boot[1][0]), axis=1)
    boot_historical = np.concatenate((boot[2][0], boot[3][0]), axis=1)
    boot_se = np.sqrt(
        np.concatenate((boot[0][1] ** 2, boot[1][1] ** 2), axis=1)
        + np.concatenate((boot[2][1] ** 2, boot[3][1] ** 2), axis=1)
    )
    t_values = (boot_simulated - boot_historical - gaps) / boot_se
    critical = higher_empirical_quantile(
        np.max(t_values, axis=1), LEAN_G5_LEAKAGE_CONFIDENCE, geometry=geometry
    )
    upper = gaps + critical.value * standard_errors
    component_passed = tuple(bool(value < LEAN_G5_LEAKAGE_CAP) for value in upper)
    return LeakageGapAudit(
        geometry,
        LEAN_G5_LEAKAGE_CONFIDENCE,
        LEAN_G5_LEAKAGE_CAP,
        critical,
        LEAN_G5_COMPONENT_NAMES,
        _floats(simulated_mean),
        _floats(historical_mean),
        _floats(gaps),
        _floats(standard_errors),
        _floats(upper),
        component_passed,
        all(component_passed),
    )


def evaluate_lean_g5_alpha(
    simulated: LeanG5AlphaRows,
    historical: LeanG5AlphaRows,
    indices: LeanG5ResampleIndices,
    simulated_inverse_volatility: Mapping[str, LeanG5InverseVolatilityRows],
    historical_inverse_volatility: Mapping[str, LeanG5InverseVolatilityRows],
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> LeanG5AlphaAudit:
    """Evaluate this layer's alpha and inverse-volatility controls."""

    equivalence = simultaneous_alpha_equivalence(simulated, indices, geometry=geometry)
    leakage = simultaneous_leakage_gap(
        simulated, historical, indices, geometry=geometry
    )
    inverse_volatility = simultaneous_inverse_volatility_gap(
        simulated_inverse_volatility,
        historical_inverse_volatility,
        indices,
        geometry=geometry,
    )
    passed = equivalence.passed and leakage.passed and inverse_volatility.passed
    return LeanG5AlphaAudit(geometry, equivalence, leakage, inverse_volatility, passed)


def simultaneous_inverse_volatility_gap(
    simulated: Mapping[str, LeanG5InverseVolatilityRows],
    historical: Mapping[str, LeanG5InverseVolatilityRows],
    indices: LeanG5ResampleIndices,
    *,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> InverseVolatilityGapAudit:
    """Bound both simulated-minus-historical improvement gaps together."""

    simulated_rows = _inverse_volatility_rows(simulated, geometry, "simulated")
    historical_rows = _inverse_volatility_rows(historical, geometry, "historical")
    matrices = tuple(
        values[:, np.newaxis] for values in (*simulated_rows, *historical_rows)
    )
    index_matrices = (
        _indices(indices.simulated_monthly, geometry.resamples, geometry.monthly_paths),
        _indices(indices.simulated_daily, geometry.resamples, geometry.daily_paths),
        _indices(
            indices.historical_monthly, geometry.resamples, geometry.monthly_paths
        ),
        _indices(indices.historical_daily, geometry.resamples, geometry.daily_paths),
    )
    point = tuple(_mean_se(matrix) for matrix in matrices)
    boot = tuple(
        _bootstrap_mean_se(matrix, index)
        for matrix, index in zip(matrices, index_matrices, strict=True)
    )
    simulated_mean = np.concatenate((point[0][0], point[1][0]))
    historical_mean = np.concatenate((point[2][0], point[3][0]))
    gaps = simulated_mean - historical_mean
    standard_errors = np.sqrt(
        np.concatenate((point[0][1] ** 2, point[1][1] ** 2))
        + np.concatenate((point[2][1] ** 2, point[3][1] ** 2))
    )
    boot_simulated = np.concatenate((boot[0][0], boot[1][0]), axis=1)
    boot_historical = np.concatenate((boot[2][0], boot[3][0]), axis=1)
    boot_se = np.sqrt(
        np.concatenate((boot[0][1] ** 2, boot[1][1] ** 2), axis=1)
        + np.concatenate((boot[2][1] ** 2, boot[3][1] ** 2), axis=1)
    )
    t_values = (boot_simulated - boot_historical - gaps) / boot_se
    critical = higher_empirical_quantile(
        np.max(t_values, axis=1), LEAN_G5_LEAKAGE_CONFIDENCE, geometry=geometry
    )
    upper = gaps + critical.value * standard_errors
    component_passed = tuple(bool(value < LEAN_G5_INVERSE_VOL_CAP) for value in upper)
    return InverseVolatilityGapAudit(
        geometry=geometry,
        confidence=LEAN_G5_LEAKAGE_CONFIDENCE,
        cap=LEAN_G5_INVERSE_VOL_CAP,
        critical_value=critical,
        frequencies=_FREQUENCIES,
        simulated_estimates=(float(simulated_mean[0]), float(simulated_mean[1])),
        historical_estimates=(float(historical_mean[0]), float(historical_mean[1])),
        gaps=(float(gaps[0]), float(gaps[1])),
        standard_errors=(
            float(standard_errors[0]),
            float(standard_errors[1]),
        ),
        upper=(float(upper[0]), float(upper[1])),
        component_passed=(component_passed[0], component_passed[1]),
        passed=all(component_passed),
    )


def _rows(
    value: object, geometry: LeanG5Geometry, label: str
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(value, LeanG5AlphaRows):
        raise ValidationError(f"lean G5 {label} alpha rows have the wrong type")
    return (
        _matrix(value.monthly, geometry.monthly_paths, f"{label} monthly"),
        _matrix(value.daily, geometry.daily_paths, f"{label} daily"),
    )


def _matrix(value: ArrayLike, paths: int, label: str) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"lean G5 {label} scores must be numeric") from exc
    expected = (paths, LEAN_G5_COMPONENTS_PER_FREQUENCY)
    if result.shape != expected:
        raise ValidationError(f"lean G5 {label} scores have the wrong shape")
    if not bool(np.isfinite(result).all()):
        raise ValidationError(f"lean G5 {label} scores must be finite")
    return result


def _indices(value: ArrayLike, resamples: int, paths: int) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValidationError("lean G5 resample indices must be integers")
    result = np.asarray(raw, dtype=np.int64)
    if result.shape != (resamples, paths):
        raise ValidationError("lean G5 resample indices have the wrong shape")
    if bool((result < 0).any()) or bool((result >= paths).any()):
        raise ValidationError("lean G5 resample indices are out of range")
    return result


def _mean_se(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = np.asarray(np.mean(values, axis=0), dtype=np.float64)
    standard_error = np.asarray(
        np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0]),
        dtype=np.float64,
    )
    if bool((standard_error <= 0.0).any()) or not bool(
        np.isfinite(standard_error).all()
    ):
        raise ValidationError("lean G5 score standard error is degenerate")
    return mean, standard_error


def _bootstrap_mean_se(
    values: FloatArray, indices: IntArray
) -> tuple[FloatArray, FloatArray]:
    means = np.empty((indices.shape[0], values.shape[1]), dtype=np.float64)
    standard_errors = np.empty_like(means)
    denominator = math.sqrt(values.shape[0])
    for start in range(0, indices.shape[0], _BATCH):
        stop = min(start + _BATCH, indices.shape[0])
        selected = values[indices[start:stop]]
        means[start:stop] = np.mean(selected, axis=1)
        standard_errors[start:stop] = np.std(selected, axis=1, ddof=1) / denominator
    if bool((standard_errors <= 0.0).any()) or not bool(
        np.isfinite(standard_errors).all()
    ):
        raise ValidationError("lean G5 resampled standard error is degenerate")
    return means, standard_errors


def _inverse_volatility_rows(
    values: object,
    geometry: LeanG5Geometry,
    label: str,
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(values, Mapping) or set(values) != set(_FREQUENCIES):
        raise ValidationError(
            f"lean G5 {label} inverse-volatility rows require monthly and daily"
        )
    rows: list[FloatArray] = []
    for frequency, paths in zip(
        _FREQUENCIES,
        (geometry.monthly_paths, geometry.daily_paths),
        strict=True,
    ):
        item = values[frequency]
        if (
            not isinstance(item, LeanG5InverseVolatilityRows)
            or item.frequency != frequency
        ):
            raise ValidationError(
                f"lean G5 {label} inverse-volatility frequency is invalid"
            )
        array = np.asarray(item.improvements, dtype=np.float64)
        if array.shape != (paths,) or not bool(np.isfinite(array).all()):
            raise ValidationError(
                f"lean G5 {label} inverse-volatility rows are invalid"
            )
        rows.append(array)
    return rows[0], rows[1]


def _vector(value: ArrayLike, expected: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError("lean G5 statistics must be numeric") from exc
    if result.shape != (expected,) or not bool(np.isfinite(result).all()):
        raise ValidationError(
            "lean G5 statistics do not match the registered resample count"
        )
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"lean G5 {label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"lean G5 {label} must be finite")
    return result


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 2**64 - 1
    ):
        raise ValidationError("lean G5 seed must be an unsigned 64-bit integer")
    return value


def _floats(values: FloatArray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
