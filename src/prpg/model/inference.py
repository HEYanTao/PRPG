"""Small deterministic inference primitives shared by Phase-3/5 gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from prpg.errors import ModelError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HolmResult:
    """Original-order Holm step-down decisions and adjusted p-values."""

    raw_p_values: FloatArray
    adjusted_p_values: FloatArray
    rejected: BoolArray
    alpha: float


@dataclass(frozen=True, slots=True)
class BinomialInterval:
    """Ordinary two-sided Clopper-Pearson interval."""

    successes: int
    trials: int
    confidence: float
    estimate: float
    lower: float
    upper: float
    conservative_half_width: float


def holm_correction(p_values: ArrayLike, *, alpha: float = 0.05) -> HolmResult:
    """Apply Holm correction in stable ``(p,index)`` order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ModelError("Holm p-values must be a non-empty vector")
    if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ModelError("Holm p-values must be finite values in [0, 1]")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ModelError("Holm alpha must be finite and in (0, 1)")
    order = np.asarray(
        sorted(range(values.size), key=lambda index: (values[index], index)),
        dtype=np.int64,
    )
    rejected = np.zeros(values.size, dtype=np.bool_)
    still_rejecting = True
    for rank, index in enumerate(order):
        threshold = alpha / (values.size - rank)
        if still_rejecting and values[index] <= threshold:
            rejected[index] = True
        else:
            still_rejecting = False
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(values.size, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    raw = values.copy()
    _readonly(raw, adjusted, rejected)
    return HolmResult(raw, adjusted, rejected, float(alpha))


def plus_one_upper_tail_pvalue(observed: float, replicates: ArrayLike) -> float:
    """Return ``(1 + count(rep >= observed)) / (B + 1)`` with ties included."""

    if not np.isfinite(observed):
        raise ModelError("observed statistic must be finite")
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ModelError("replicate statistics must be a non-empty finite vector")
    return float((1 + np.count_nonzero(values >= observed)) / (values.size + 1))


def higher_quantile(values: ArrayLike, probability: float) -> float:
    """Return NumPy's frozen empirical quantile with method ``higher``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ModelError("quantile values must be a non-empty finite vector")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ModelError("quantile probability must be finite and in [0, 1]")
    return float(np.quantile(array, probability, method="higher"))


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> BinomialInterval:
    """Calculate the ordinary exact two-sided binomial interval."""

    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ModelError("binomial trials must be a positive integer")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or not 0 <= successes <= trials
    ):
        raise ModelError("binomial successes must be an integer in [0, trials]")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ModelError("binomial confidence must be finite and in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    estimate = successes / trials
    half_width = max(estimate - lower, upper - estimate)
    return BinomialInterval(
        successes=successes,
        trials=trials,
        confidence=float(confidence),
        estimate=float(estimate),
        lower=lower,
        upper=upper,
        conservative_half_width=float(half_width),
    )


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
