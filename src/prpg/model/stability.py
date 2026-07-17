"""Pure alignment, splitting, and simultaneous-envelope utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from prpg.errors import ModelError


@dataclass(frozen=True, slots=True)
class LabelAlignment:
    """Hungarian mapping from candidate labels onto reference labels."""

    candidate_to_reference: NDArray[np.int64]
    aligned_states: NDArray[np.int64]
    contingency: NDArray[np.int64]
    adjusted_rand_index: float


@dataclass(frozen=True, slots=True)
class SimultaneousEnvelope:
    """Max-studentized two-sided predictive envelope."""

    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    critical_value: float
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    observed: NDArray[np.float64]
    component_passed: NDArray[np.bool_]
    passed: bool
    confidence: float
    repetitions: int


@dataclass(frozen=True, slots=True)
class RollingFold:
    """One exact expanding fold with source-segment validation geometry."""

    fold: int
    training: slice
    validation: slice
    training_rows: int
    validation_rows: int
    validation_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MacroStabilitySummary:
    """Binding 100-attempt macro-bootstrap stability decision."""

    successful_attempts: int
    median_ari: float
    tenth_percentile_ari: float
    median_jaccard_by_state: NDArray[np.float64]
    tenth_percentile_jaccard_by_state: NDArray[np.float64]
    passed: bool


@dataclass(frozen=True, slots=True)
class RollingStabilitySummary:
    """Binding six-fold rolling-model stability decision."""

    successful_fits: int
    median_ari: float
    minimum_successful_ari: float
    median_jaccard_by_state: NDArray[np.float64]
    passed: bool


def sequence_lengths(continuation: NDArray[np.bool_]) -> tuple[int, ...]:
    """Convert a false-at-segment-start mask into positive HMM lengths."""

    values = np.asarray(continuation)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind != "b":
        raise ModelError("continuation mask must be a non-empty boolean vector")
    if bool(values[0]):
        raise ModelError("first continuation-mask value must be false")
    starts = np.flatnonzero(~values)
    ends = np.r_[starts[1:], len(values)]
    lengths = tuple(int(value) for value in ends - starts)
    if any(value <= 0 for value in lengths) or sum(lengths) != len(values):
        raise ModelError("continuation mask produced invalid sequence lengths")
    return lengths


def align_state_labels(
    reference: NDArray[np.integer],
    candidate: NDArray[np.integer],
    n_states: int,
) -> LabelAlignment:
    """Hungarian-align labels decoded on the same reference observations."""

    first = _state_vector(reference, n_states, "reference")
    second = _state_vector(candidate, n_states, "candidate")
    if len(first) != len(second):
        raise ModelError("state sequences require equal lengths for alignment")
    contingency = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(contingency, (first, second), 1)
    rows, columns = linear_sum_assignment(-contingency)
    mapping = np.full(n_states, -1, dtype=np.int64)
    mapping[columns] = rows
    if bool((mapping < 0).any()):
        raise ModelError("state-label assignment is incomplete")
    aligned = mapping[second]
    return LabelAlignment(
        candidate_to_reference=mapping,
        aligned_states=aligned,
        contingency=contingency,
        adjusted_rand_index=float(adjusted_rand_score(first, aligned)),
    )


def aligned_state_jaccards(
    reference: NDArray[np.integer],
    aligned_candidate: NDArray[np.integer],
    n_states: int,
) -> NDArray[np.float64]:
    """Calculate one Jaccard overlap per canonical reference state."""

    first = _state_vector(reference, n_states, "reference")
    second = _state_vector(aligned_candidate, n_states, "aligned candidate")
    if len(first) != len(second):
        raise ModelError("state sequences require equal lengths for Jaccard overlap")
    result = np.zeros(n_states, dtype=np.float64)
    for state in range(n_states):
        first_mask = first == state
        second_mask = second == state
        union = int(np.count_nonzero(first_mask | second_mask))
        if union:
            result[state] = np.count_nonzero(first_mask & second_mask) / union
    result.setflags(write=False)
    return result


def rolling_origin_splits(
    observations: int,
    *,
    folds: int,
    validation_size: int,
    minimum_training_size: int,
) -> tuple[tuple[slice, slice], ...]:
    """Create fixed final contiguous folds with strictly expanding training."""

    for name, value in (
        ("observations", observations),
        ("folds", folds),
        ("validation_size", validation_size),
        ("minimum_training_size", minimum_training_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelError(f"{name} must be a positive integer")
    first_end = observations - folds * validation_size
    if first_end < minimum_training_size:
        raise ModelError("rolling-origin contract leaves too little initial training")
    result: list[tuple[slice, slice]] = []
    for fold in range(folds):
        training_end = first_end + fold * validation_size
        validation_end = training_end + validation_size
        result.append((slice(0, training_end), slice(training_end, validation_end)))
    return tuple(result)


def canonical_rolling_folds(
    continuation: NDArray[np.bool_],
    *,
    folds: int = 6,
    validation_size: int = 12,
) -> tuple[RollingFold, ...]:
    """Build the approved folds and retain exact validation segment lengths."""

    mask = np.asarray(continuation)
    # sequence_lengths performs all shape/dtype/first-row checks.
    sequence_lengths(mask)
    splits = rolling_origin_splits(
        len(mask),
        folds=folds,
        validation_size=validation_size,
        minimum_training_size=1,
    )
    result: list[RollingFold] = []
    for fold, (training, validation) in enumerate(splits):
        local = mask[validation].astype(np.bool_, copy=True)
        local[0] = False
        lengths = sequence_lengths(local)
        result.append(
            RollingFold(
                fold=fold,
                training=training,
                validation=validation,
                training_rows=int(training.stop or 0),
                validation_rows=int((validation.stop or 0) - (validation.start or 0)),
                validation_lengths=lengths,
            )
        )
    return tuple(result)


def summarize_macro_bootstrap_stability(
    successful: NDArray[np.bool_],
    adjusted_rand_indices: NDArray[np.floating],
    jaccards: NDArray[np.floating],
) -> MacroStabilitySummary:
    """Apply the exact failure-as-zero 100-attempt macro stability gates."""

    success, ari, overlap = _stability_inputs(
        successful,
        adjusted_rand_indices,
        jaccards,
        expected_attempts=100,
    )
    successful_attempts = int(np.count_nonzero(success))
    effective_ari = np.where(success, ari, 0.0)
    effective_overlap = np.where(success[:, None], overlap, 0.0)
    median_ari = float(np.quantile(effective_ari, 0.5, method="higher"))
    tenth_ari = float(np.quantile(effective_ari, 0.1, method="higher"))
    median_jaccard = np.quantile(
        effective_overlap, 0.5, axis=0, method="higher"
    ).astype(np.float64)
    tenth_jaccard = np.quantile(effective_overlap, 0.1, axis=0, method="higher").astype(
        np.float64
    )
    passed = (
        successful_attempts >= 90
        and median_ari >= 0.80
        and tenth_ari >= 0.50
        and bool((median_jaccard >= 0.70).all())
        and bool((tenth_jaccard >= 0.50).all())
    )
    median_jaccard.setflags(write=False)
    tenth_jaccard.setflags(write=False)
    return MacroStabilitySummary(
        successful_attempts,
        median_ari,
        tenth_ari,
        median_jaccard,
        tenth_jaccard,
        passed,
    )


def summarize_rolling_stability(
    successful: NDArray[np.bool_],
    adjusted_rand_indices: NDArray[np.floating],
    jaccards: NDArray[np.floating],
) -> RollingStabilitySummary:
    """Apply the distinct six-fold rolling stability gates."""

    success, ari, overlap = _stability_inputs(
        successful,
        adjusted_rand_indices,
        jaccards,
        expected_attempts=6,
    )
    successful_fits = int(np.count_nonzero(success))
    effective_ari = np.where(success, ari, 0.0)
    effective_overlap = np.where(success[:, None], overlap, 0.0)
    median_ari = float(np.quantile(effective_ari, 0.5, method="higher"))
    minimum_successful = float(np.min(ari[success])) if successful_fits else -math.inf
    median_jaccard = np.quantile(
        effective_overlap, 0.5, axis=0, method="higher"
    ).astype(np.float64)
    passed = (
        successful_fits >= 5
        and median_ari >= 0.70
        and minimum_successful >= 0.40
        and bool((median_jaccard >= 0.60).all())
    )
    median_jaccard.setflags(write=False)
    return RollingStabilitySummary(
        successful_fits,
        median_ari,
        minimum_successful,
        median_jaccard,
        passed,
    )


def map_monthly_states_to_daily(
    monthly_period_ordinals: NDArray[np.integer],
    monthly_states: NDArray[np.integer],
    daily_session_ns: NDArray[np.integer],
) -> NDArray[np.int64]:
    """Assign each daily return the decoded state of its return-ending month."""

    ordinals = np.asarray(monthly_period_ordinals, dtype=np.int64)
    states = np.asarray(monthly_states, dtype=np.int64)
    sessions = np.asarray(daily_session_ns, dtype=np.int64)
    if ordinals.ndim != 1 or states.ndim != 1 or sessions.ndim != 1:
        raise ModelError("state-mapping inputs must be one-dimensional")
    if len(ordinals) != len(states) or len(ordinals) == 0 or len(sessions) == 0:
        raise ModelError("state-mapping inputs have incompatible or empty lengths")
    if not bool(np.all(np.diff(ordinals) > 0)):
        raise ModelError("monthly period ordinals must be strictly increasing")
    month_periods = pd.PeriodIndex.from_ordinals(ordinals, freq="M")
    daily_periods = pd.DatetimeIndex(sessions).to_period("M")
    lookup = pd.Series(states, index=month_periods)
    mapped = lookup.reindex(daily_periods)
    if bool(mapped.isna().any()):
        missing = sorted({str(value) for value in daily_periods[mapped.isna()]})
        raise ModelError(
            "daily sessions include months without decoded states",
            details={"missing_months": missing},
        )
    result: NDArray[np.int64] = np.asarray(mapped.to_numpy(), dtype=np.int64)
    return result


def simultaneous_predictive_envelope(
    replicates: NDArray[np.floating],
    observed: NDArray[np.floating],
    *,
    confidence: float,
    constant_tolerance: float = 1e-12,
) -> SimultaneousEnvelope:
    """Create an empirical max-studentized family-wise predictive envelope."""

    samples = np.asarray(replicates, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    if samples.ndim != 2 or target.ndim != 1 or samples.shape[1] != len(target):
        raise ModelError("predictive-envelope dimensions are incompatible")
    if samples.shape[0] < 2:
        raise ModelError("predictive envelope requires at least two repetitions")
    if not np.isfinite(samples).all() or not np.isfinite(target).all():
        raise ModelError("predictive envelope inputs must be finite")
    if not 0.0 < confidence < 1.0:
        raise ModelError("predictive-envelope confidence must be in (0, 1)")
    if not np.isfinite(constant_tolerance) or constant_tolerance < 0.0:
        raise ModelError("constant tolerance must be finite and non-negative")

    center = samples.mean(axis=0)
    scale = samples.std(axis=0, ddof=1)
    varying = scale > constant_tolerance
    standardized = np.zeros_like(samples)
    standardized[:, varying] = np.abs(
        (samples[:, varying] - center[varying]) / scale[varying]
    )
    maxima = standardized.max(axis=1)
    critical = float(np.quantile(maxima, confidence, method="higher"))
    lower = center - critical * scale
    upper = center + critical * scale
    constant_passed = np.abs(target - center) <= constant_tolerance
    component = np.where(
        varying,
        (target >= lower) & (target <= upper),
        constant_passed,
    )
    return SimultaneousEnvelope(
        center=center,
        scale=scale,
        critical_value=critical,
        lower=lower,
        upper=upper,
        observed=target,
        component_passed=component,
        passed=bool(component.all()),
        confidence=confidence,
        repetitions=samples.shape[0],
    )


def _stability_inputs(
    successful: NDArray[np.bool_],
    adjusted_rand_indices: NDArray[np.floating],
    jaccards: NDArray[np.floating],
    *,
    expected_attempts: int,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    success = np.asarray(successful)
    ari = np.asarray(adjusted_rand_indices, dtype=np.float64)
    overlap = np.asarray(jaccards, dtype=np.float64)
    if success.shape != (expected_attempts,) or success.dtype.kind != "b":
        raise ModelError(
            "stability success vector must contain exactly "
            f"{expected_attempts} booleans"
        )
    if ari.shape != (expected_attempts,) or not np.isfinite(ari).all():
        raise ModelError("stability ARI vector has invalid shape or values")
    if (
        overlap.ndim != 2
        or overlap.shape[0] != expected_attempts
        or overlap.shape[1] == 0
        or not np.isfinite(overlap).all()
    ):
        raise ModelError("stability Jaccard matrix has invalid shape or values")
    if bool(((overlap < 0.0) | (overlap > 1.0)).any()):
        raise ModelError("stability Jaccards must be in [0, 1]")
    return success.astype(np.bool_, copy=False), ari, overlap


def _state_vector(
    values: NDArray[np.integer], n_states: int, label: str
) -> NDArray[np.int64]:
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states <= 0:
        raise ModelError("n_states must be a positive integer")
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ModelError(f"{label} states must be a non-empty integer vector")
    result = array.astype(np.int64, copy=False)
    if bool(((result < 0) | (result >= n_states)).any()):
        raise ModelError(f"{label} states fall outside the registered state range")
    return result
