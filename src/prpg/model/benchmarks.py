"""Frozen holdout benchmarks for the Phase-3 HMM veto.

The Gaussian and ridge-VAR(1) models in this module are deliberately small,
closed-form references.  They have no tuning loop and cannot select or promote
an HMM candidate.  They only determine whether the already-selected design HMM
has non-negative holdout log-score differences against both registered
benchmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov

from prpg.errors import ModelError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

EIGENVALUE_FLOOR = 1e-6
MAXIMUM_CONDITION_NUMBER = 1e6
MAXIMUM_VAR_SPECTRAL_RADIUS = 0.995

HoldoutBenchmarkFailureCode: TypeAlias = Literal[
    "gaussian_covariance_invalid",
    "gaussian_score_invalid",
    "var_ridge_solve_failed",
    "var_parameters_nonfinite",
    "var_eigenvalues_nonfinite",
    "var_spectral_radius_invalid",
    "var_residual_covariance_invalid",
    "var_stationary_law_invalid",
    "var_stationary_covariance_invalid",
    "var_score_invalid",
]


@dataclass(frozen=True, slots=True)
class InvalidHoldoutBenchmarkEvidence:
    """Compact code-owned evidence for one invalid frozen benchmark."""

    failure_code: HoldoutBenchmarkFailureCode
    failure_message: str
    failure_details: tuple[tuple[str, object], ...]


class InvalidHoldoutBenchmark(ModelError):
    """Narrow scientific signal for a valid input whose benchmark is invalid.

    Only the closed numerical/stability checks below raise this subtype.  Bad
    shapes, non-finite caller inputs, invalid continuation masks, and generic
    operational or programming failures remain ordinary exceptions and must
    leave a canonical launch resumable.
    """

    def __init__(self, evidence: InvalidHoldoutBenchmarkEvidence) -> None:
        if not isinstance(evidence, InvalidHoldoutBenchmarkEvidence):
            raise ModelError("invalid holdout benchmark signal requires typed evidence")
        self.evidence = evidence
        super().__init__(
            evidence.failure_message,
            details=dict(evidence.failure_details),
        )


@dataclass(frozen=True, slots=True)
class GaussianBenchmark:
    """Maximum-likelihood one-state Gaussian reference."""

    mean: FloatArray
    covariance: FloatArray
    condition_number: float


@dataclass(frozen=True, slots=True)
class RidgeVAR1Benchmark:
    """Fixed-penalty VAR(1) and its stationary Gaussian law."""

    intercept: FloatArray
    coefficients: FloatArray
    residual_covariance: FloatArray
    stationary_mean: FloatArray
    stationary_covariance: FloatArray
    residual_condition_number: float
    stationary_condition_number: float
    spectral_radius: float
    training_links: int


@dataclass(frozen=True, slots=True)
class HoldoutBenchmarkScores:
    """Per-row scores for both frozen benchmark families."""

    gaussian: FloatArray
    ridge_var1: FloatArray

    @property
    def n_observations(self) -> int:
        return int(self.gaussian.size)


@dataclass(frozen=True, slots=True)
class HoldoutVetoDecision:
    """Two-benchmark veto evidence for one already-selected HMM."""

    hmm_mean_log_score: float
    gaussian_mean_log_score: float
    ridge_var1_mean_log_score: float
    hmm_minus_gaussian: float
    hmm_minus_ridge_var1: float
    passed: bool


def fit_gaussian_benchmark(values: ArrayLike) -> GaussianBenchmark:
    """Fit the exact training ML Gaussian with the canonical eigenvalue floor."""

    matrix = _matrix(values, "Gaussian benchmark training values")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    covariance = centered.T @ centered / len(matrix)
    floored, condition = _floor_covariance(
        covariance,
        label="Gaussian benchmark covariance",
        failure_code="gaussian_covariance_invalid",
    )
    _make_read_only(mean, floored)
    return GaussianBenchmark(mean, floored, condition)


def fit_ridge_var1_benchmark(
    values: ArrayLike,
    continuation: ArrayLike,
) -> RidgeVAR1Benchmark:
    """Fit the fixed unit-ridge VAR(1) on valid within-segment links only."""

    matrix = _matrix(values, "VAR training values")
    links = _continuation(continuation, len(matrix))
    valid_rows = np.flatnonzero(links)
    if valid_rows.size == 0:
        raise ModelError("VAR benchmark has no valid within-segment training links")
    x = matrix[valid_rows - 1]
    y = matrix[valid_rows]
    n_links = len(valid_rows)
    design = np.column_stack([np.ones(n_links, dtype=np.float64), x])
    penalty = np.diag(
        np.concatenate(
            [np.zeros(1, dtype=np.float64), np.full(matrix.shape[1], n_links)]
        )
    )
    normal = design.T @ design + penalty
    try:
        solution = np.linalg.solve(normal, design.T @ y)
    except np.linalg.LinAlgError as error:
        raise _invalid_benchmark(
            "var_ridge_solve_failed",
            "VAR benchmark ridge solve failed",
        ) from error
    intercept = np.asarray(solution[0], dtype=np.float64)
    coefficients = np.asarray(solution[1:].T, dtype=np.float64)
    residuals = y - intercept - x @ coefficients.T
    raw_covariance = residuals.T @ residuals / n_links
    dimension = matrix.shape[1]
    shrunk = 0.90 * raw_covariance + (
        0.10 * float(np.trace(raw_covariance)) / dimension
    ) * np.eye(dimension)
    residual_covariance, residual_condition = _floor_covariance(
        shrunk,
        label="VAR residual covariance",
        failure_code="var_residual_covariance_invalid",
    )
    if not np.isfinite(intercept).all() or not np.isfinite(coefficients).all():
        raise _invalid_benchmark(
            "var_parameters_nonfinite",
            "VAR benchmark parameters are non-finite",
        )
    try:
        eigenvalues = np.linalg.eigvals(coefficients)
    except np.linalg.LinAlgError as error:
        raise _invalid_benchmark(
            "var_eigenvalues_nonfinite",
            "VAR benchmark eigenvalue computation failed",
        ) from error
    if not np.isfinite(eigenvalues).all():
        raise _invalid_benchmark(
            "var_eigenvalues_nonfinite",
            "VAR benchmark eigenvalues are non-finite",
        )
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if spectral_radius >= MAXIMUM_VAR_SPECTRAL_RADIUS:
        raise _invalid_benchmark(
            "var_spectral_radius_invalid",
            "VAR benchmark spectral radius is not below 0.995",
            details={"spectral_radius": spectral_radius},
        )
    try:
        stationary_mean = np.asarray(
            np.linalg.solve(np.eye(dimension) - coefficients, intercept),
            dtype=np.float64,
        )
        stationary_covariance_raw = solve_discrete_lyapunov(
            coefficients, residual_covariance
        )
    except (np.linalg.LinAlgError, ValueError) as error:
        raise _invalid_benchmark(
            "var_stationary_law_invalid",
            "VAR stationary-law solve failed",
        ) from error
    stationary_covariance, stationary_condition = _validated_covariance(
        stationary_covariance_raw,
        label="VAR stationary covariance",
        allow_floor=False,
        failure_code="var_stationary_covariance_invalid",
    )
    if not np.isfinite(stationary_mean).all():
        raise _invalid_benchmark(
            "var_stationary_law_invalid",
            "VAR stationary mean is non-finite",
        )
    residual = coefficients @ stationary_covariance @ coefficients.T
    residual += residual_covariance
    lyapunov_error = float(np.max(np.abs(residual - stationary_covariance)))
    if not math.isfinite(lyapunov_error) or lyapunov_error > 1e-10:
        raise _invalid_benchmark(
            "var_stationary_law_invalid",
            "VAR stationary covariance fails the Lyapunov identity",
            details={"maximum_absolute_error": lyapunov_error},
        )
    _make_read_only(
        intercept,
        coefficients,
        residual_covariance,
        stationary_mean,
        stationary_covariance,
    )
    return RidgeVAR1Benchmark(
        intercept=intercept,
        coefficients=coefficients,
        residual_covariance=residual_covariance,
        stationary_mean=stationary_mean,
        stationary_covariance=stationary_covariance,
        residual_condition_number=residual_condition,
        stationary_condition_number=stationary_condition,
        spectral_radius=spectral_radius,
        training_links=n_links,
    )


def score_holdout_benchmarks(
    training_values: ArrayLike,
    holdout_values: ArrayLike,
    holdout_continuation: ArrayLike,
) -> HoldoutBenchmarkScores:
    """Score holdout rows with terminal conditioning and stationary gap resets."""

    training = _matrix(training_values, "benchmark training values")
    holdout = _matrix(holdout_values, "benchmark holdout values")
    if training.shape[1] != holdout.shape[1]:
        raise ModelError("benchmark training and holdout feature dimensions differ")
    continuation = _continuation(holdout_continuation, len(holdout))
    if bool(continuation[0]):
        raise ModelError("holdout continuation must be false at its first row")
    gaussian = fit_gaussian_benchmark(training)

    # The design fit is one uninterrupted 193-row segment in version 1.  The
    # generic helper still accepts explicit all-true links after row zero.
    training_continuation = np.ones(len(training), dtype=np.bool_)
    training_continuation[0] = False
    var = fit_ridge_var1_benchmark(training, training_continuation)
    gaussian_scores = _gaussian_logpdf_rows(
        holdout,
        gaussian.mean,
        gaussian.covariance,
        failure_code="gaussian_score_invalid",
    )
    var_scores = np.empty(len(holdout), dtype=np.float64)
    for row in range(len(holdout)):
        if row == 0:
            mean = var.intercept + var.coefficients @ training[-1]
            covariance = var.residual_covariance
        elif bool(continuation[row]):
            mean = var.intercept + var.coefficients @ holdout[row - 1]
            covariance = var.residual_covariance
        else:
            mean = var.stationary_mean
            covariance = var.stationary_covariance
        var_scores[row] = _gaussian_logpdf_rows(
            holdout[row : row + 1],
            mean,
            covariance,
            failure_code="var_score_invalid",
        )[0]
    _make_read_only(gaussian_scores, var_scores)
    return HoldoutBenchmarkScores(gaussian_scores, var_scores)


def holdout_veto_decision(
    hmm_log_scores: ArrayLike,
    benchmark_scores: HoldoutBenchmarkScores,
) -> HoldoutVetoDecision:
    """Require non-negative selected-HMM score differences against both models."""

    hmm = _vector(hmm_log_scores, "HMM holdout log scores")
    gaussian = _vector(benchmark_scores.gaussian, "Gaussian holdout log scores")
    var = _vector(benchmark_scores.ridge_var1, "VAR holdout log scores")
    if not (hmm.shape == gaussian.shape == var.shape):
        raise ModelError("holdout score vectors must have identical shapes")
    hmm_mean = float(hmm.mean())
    gaussian_mean = float(gaussian.mean())
    var_mean = float(var.mean())
    gaussian_difference = hmm_mean - gaussian_mean
    var_difference = hmm_mean - var_mean
    return HoldoutVetoDecision(
        hmm_mean_log_score=hmm_mean,
        gaussian_mean_log_score=gaussian_mean,
        ridge_var1_mean_log_score=var_mean,
        hmm_minus_gaussian=gaussian_difference,
        hmm_minus_ridge_var1=var_difference,
        passed=gaussian_difference >= 0.0 and var_difference >= 0.0,
    )


def _floor_covariance(
    covariance: ArrayLike,
    *,
    label: str,
    failure_code: HoldoutBenchmarkFailureCode,
) -> tuple[FloatArray, float]:
    return _validated_covariance(
        covariance,
        label=label,
        allow_floor=True,
        failure_code=failure_code,
    )


def _validated_covariance(
    covariance: ArrayLike,
    *,
    label: str,
    allow_floor: bool,
    failure_code: HoldoutBenchmarkFailureCode,
) -> tuple[FloatArray, float]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise _invalid_benchmark(
            failure_code,
            f"{label} must be a non-empty square matrix",
        )
    if not np.isfinite(matrix).all():
        raise _invalid_benchmark(failure_code, f"{label} is non-finite")
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as error:
        raise _invalid_benchmark(
            failure_code,
            f"{label} eigendecomposition failed",
        ) from error
    if allow_floor:
        eigenvalues = np.maximum(eigenvalues, EIGENVALUE_FLOOR)
        symmetric = (eigenvectors * eigenvalues) @ eigenvectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    elif float(eigenvalues.min()) <= 0.0:
        raise _invalid_benchmark(
            failure_code,
            f"{label} is not positive definite",
        )
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    condition = math.inf if minimum <= 0.0 else maximum / minimum
    if not np.isfinite(condition) or condition > MAXIMUM_CONDITION_NUMBER:
        raise _invalid_benchmark(
            failure_code,
            f"{label} condition number exceeds 1e6",
            details={"condition_number": condition},
        )
    return np.asarray(symmetric, dtype=np.float64), float(condition)


def _gaussian_logpdf_rows(
    values: FloatArray,
    mean: FloatArray,
    covariance: FloatArray,
    *,
    failure_code: HoldoutBenchmarkFailureCode,
) -> FloatArray:
    try:
        cholesky = np.linalg.cholesky(covariance)
        solved = np.linalg.solve(cholesky, (values - mean).T).T
    except np.linalg.LinAlgError as error:
        raise _invalid_benchmark(
            failure_code,
            "benchmark Gaussian covariance solve failed",
        ) from error
    log_determinant = 2.0 * float(np.log(np.diag(cholesky)).sum())
    quadratic = np.einsum("ij,ij->i", solved, solved)
    result = -0.5 * (
        values.shape[1] * math.log(2.0 * math.pi) + log_determinant + quadratic
    )
    if not np.isfinite(result).all():
        raise _invalid_benchmark(
            failure_code,
            "benchmark Gaussian scores are non-finite",
        )
    return np.asarray(result, dtype=np.float64)


def _invalid_benchmark(
    failure_code: HoldoutBenchmarkFailureCode,
    failure_message: str,
    *,
    details: dict[str, object] | None = None,
) -> InvalidHoldoutBenchmark:
    frozen_details = tuple(sorted((details or {}).items()))
    return InvalidHoldoutBenchmark(
        InvalidHoldoutBenchmarkEvidence(
            failure_code=failure_code,
            failure_message=failure_message,
            failure_details=frozen_details,
        )
    )


def _matrix(values: ArrayLike, label: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ModelError(f"{label} must have at least two rows and one column")
    if not np.isfinite(matrix).all():
        raise ModelError(f"{label} contains non-finite values")
    return matrix


def _vector(values: ArrayLike, label: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ModelError(f"{label} must be a non-empty finite vector")
    return vector


def _continuation(values: ArrayLike, length: int) -> BoolArray:
    raw = np.asarray(values)
    if raw.shape != (length,) or raw.dtype.kind != "b":
        raise ModelError("continuation must be a boolean vector matching rows")
    result = raw.astype(np.bool_, copy=True)
    if bool(result[0]):
        raise ModelError("continuation must be false at row zero")
    return result


def _make_read_only(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
