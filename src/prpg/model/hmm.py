"""Deterministic Gaussian-HMM fitting and diagnostic primitives.

This module implements the numerical core frozen in design Sections 8.2 and
8.3.  It deliberately does not select a state count, calibrate block lengths,
or publish artifacts.  Inputs and outputs are ordinary scalars and NumPy
arrays suitable for audited JSON metadata plus ``allow_pickle=False`` NPZ/NPY
storage; fitted Python objects are never serialized.

Missing calendar months split the macro history into independent sequences.
The split is honored by fitting, likelihood scoring, Viterbi decoding, and
historical path diagnostics, so no transition is estimated across a source
gap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from numpy.typing import NDArray
from scipy.special import logsumexp

from prpg.data.transform import StandardizedFeatures
from prpg.errors import ModelError
from prpg.execution import deterministic_process_map
from prpg.model.regime_policy import (
    OwnerFixedK4Disposition,
    OwnerFixedK4Policy,
    effective_identification_report_passed,
    effective_transition_report_passed,
    owner_fixed_k4_disposition,
    validate_owner_fixed_k4_disposition,
    validate_owner_fixed_k4_policy,
)
from prpg.simulation.rng import ModelFitScope, hmm_library_seed

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ArrayScalar = TypeVar("ArrayScalar", bound=np.generic)
ScalerScope = Literal["design_training", "production_full"]

HMM_RESTARTS = 50
PROBABILITY_ATOL = 1e-12
STATIONARY_RESIDUAL_ATOL = 1e-12
DEFAULT_EIGENVALUE_FLOOR = 1e-6
DEFAULT_MAX_COVARIANCE_CONDITION = 1e6
RAW_COVARIANCE_EIGENVALUE_TOLERANCE = -1e-10
MAXIMUM_COVARIANCE_FLOOR_RELATIVE_CHANGE = 0.01
LIKELIHOOD_DECREASE_RELATIVE_TOLERANCE = 1e-8
LIKELIHOOD_TIE_RELATIVE_TOLERANCE = 1e-10
NEGATIVE_PROBABILITY_TOLERANCE = 1e-15
NUMERICAL_EDGE_THRESHOLD = 1e-12
MAXIMUM_SELF_TRANSITION = 0.98
MIXING_TOTAL_VARIATION_THRESHOLD = 0.01
MAXIMUM_MIXING_MONTHS = 120

MAXIMUM_NORMALIZED_POSTERIOR_ENTROPY = 0.70
MINIMUM_MEAN_MAXIMUM_POSTERIOR = 0.65
MINIMUM_POSTERIOR_EFFECTIVE_MASS = 24.0
MINIMUM_POSTERIOR_EFFECTIVE_FRACTION = 0.10
MINIMUM_ASSIGNED_STATE_POSTERIOR = 0.60
MINIMUM_PAIRWISE_MAHALANOBIS_DISTANCE = 0.75


@dataclass(frozen=True, slots=True)
class HMMFeatureMatrix:
    """A scaler-scoped, finite fitting matrix and its source sequences."""

    values: FloatArray
    lengths: tuple[int, ...]
    feature_names: tuple[str, ...]
    row_labels: tuple[str, ...]
    scaler_means: FloatArray
    scaler_standard_deviations: FloatArray
    scaler_scope: ScalerScope


@dataclass(frozen=True, slots=True)
class HMMParameters:
    """JSON/NPZ-ready parameters of a tied-covariance Gaussian HMM."""

    start_probabilities: FloatArray
    transition_matrix: FloatArray
    means: FloatArray
    tied_covariance: FloatArray

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])


@dataclass(frozen=True, slots=True)
class CanonicalizedHMM:
    """Parameters and states after deterministic centroid label ordering."""

    parameters: HMMParameters
    canonical_to_original: IntArray
    original_to_canonical: IntArray
    states: IntArray | None


@dataclass(frozen=True, slots=True)
class RestartDiagnostic:
    """Complete redacted outcome of one deterministic library restart."""

    restart_index: int
    seed: int
    converged: bool
    iterations: int
    log_likelihood: float | None
    valid: bool
    raw_minimum_eigenvalue: float | None
    maximum_floor_relative_change: float | None
    covariance_condition_number: float | None
    maximum_likelihood_decrease: float | None
    preliminary_chain_passed: bool
    failure_type: str | None
    failure_message: str | None

    def as_dict(self) -> dict[str, int | float | bool | str | None]:
        """Return primitive audit metadata without a fitted model object."""

        return {
            "restart_index": self.restart_index,
            "seed": self.seed,
            "converged": self.converged,
            "iterations": self.iterations,
            "log_likelihood": self.log_likelihood,
            "valid": self.valid,
            "raw_minimum_eigenvalue": self.raw_minimum_eigenvalue,
            "maximum_floor_relative_change": self.maximum_floor_relative_change,
            "covariance_condition_number": self.covariance_condition_number,
            "maximum_likelihood_decrease": self.maximum_likelihood_decrease,
            "preliminary_chain_passed": self.preliminary_chain_passed,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


_NUMERICAL_RESTART_FAILURE_TYPES = frozenset(
    {
        "CovarianceAuditMissing",
        "CovarianceFloorPerturbation",
        "InsufficientIterations",
        "LikelihoodDecrease",
        "LikelihoodEvaluationFailure",
        "NonConvergence",
        "NonFiniteLikelihood",
        "NumericalRestartFailure",
        "ParameterAuditFailure",
        "PreliminaryMarkovFailure",
        "RawCovarianceEigenvalue",
    }
)


def is_numerical_restart_failure_diagnostic(value: object) -> bool:
    """Return whether one diagnostic is a closed numerical invalid outcome."""

    return (
        isinstance(value, RestartDiagnostic)
        and value.valid is False
        and value.failure_type in _NUMERICAL_RESTART_FAILURE_TYPES
        and isinstance(value.failure_message, str)
        and bool(value.failure_message)
    )


class HMMRestartNumericalFailure(ModelError):
    """One explicitly classified numerical restart failure.

    Only this narrow subtype (plus NumPy's two numerical exception classes)
    may be converted into an invalid restart diagnostic.  Generic
    ``ModelError`` instances and operational/programming exceptions must
    propagate so a canonical run remains resumable rather than acquiring a
    fabricated scientific outcome.
    """


class NoNumericallyValidHMMRestarts(HMMRestartNumericalFailure):
    """Exact terminal signal for fifty completed but invalid HMM restarts."""

    n_states: int
    restart_diagnostics: tuple[RestartDiagnostic, ...]

    def __init__(
        self,
        *,
        n_states: int,
        restart_diagnostics: Sequence[RestartDiagnostic],
    ) -> None:
        diagnostics = tuple(restart_diagnostics)
        if (
            isinstance(n_states, bool)
            or not isinstance(n_states, int | np.integer)
            or int(n_states) <= 0
            or len(diagnostics) != HMM_RESTARTS
            or any(
                not isinstance(item, RestartDiagnostic)
                or item.restart_index != index
                or not is_numerical_restart_failure_diagnostic(item)
                for index, item in enumerate(diagnostics)
            )
        ):
            raise ModelError(
                "no-valid-restarts signal requires exactly 50 ordered invalid "
                "numerical diagnostics"
            )
        seeds = tuple(item.seed for item in diagnostics)
        if len(set(seeds)) != HMM_RESTARTS or any(
            isinstance(seed, bool)
            or not isinstance(seed, int | np.integer)
            or not 0 <= int(seed) <= np.iinfo(np.uint32).max
            for seed in seeds
        ):
            raise ModelError(
                "no-valid-restarts signal diagnostics require distinct uint32 seeds"
            )
        self.n_states = int(n_states)
        self.restart_diagnostics = diagnostics
        super().__init__(
            "no numerically valid tied Gaussian-HMM restart",
            details={
                "n_states": self.n_states,
                "restart_diagnostics": [item.as_dict() for item in diagnostics],
            },
        )


HMMCandidateInadmissibilityCode = Literal[
    "posterior_identification",
    "historical_transition_support",
]


@dataclass(frozen=True, slots=True)
class HMMCandidateInadmissibilityEvidence:
    """Immutable evidence for one scientifically ineligible post-fit K.

    This type is deliberately limited to the two approved post-fit screens
    that belong to candidate-family viability.  Numerical/input-contract
    defects and operational failures do not have a representation here and
    therefore cannot be converted into a scientific candidate rejection.
    """

    n_states: int
    failure_code: HMMCandidateInadmissibilityCode
    failure_message: str
    rejection_reasons: tuple[str, ...]
    gate_details: tuple[tuple[str, object], ...]
    best_restart_index: int
    best_seed: int
    restart_diagnostics: tuple[RestartDiagnostic, ...]

    def as_details(self) -> dict[str, object]:
        """Return a detached structured representation for error logging."""

        return {
            "n_states": self.n_states,
            "failure_code": self.failure_code,
            "rejection_reasons": self.rejection_reasons,
            "gate_details": self.gate_details,
            "best_restart_index": self.best_restart_index,
            "best_seed": self.best_seed,
            "restart_diagnostics": tuple(
                tuple(sorted(item.as_dict().items()))
                for item in self.restart_diagnostics
            ),
        }


class HMMCandidateInadmissibility(ModelError):
    """Narrow scientific signal for one completed but inadmissible HMM fit."""

    evidence: HMMCandidateInadmissibilityEvidence

    def __init__(self, evidence: HMMCandidateInadmissibilityEvidence) -> None:
        if not _valid_candidate_inadmissibility_evidence(evidence):
            raise ModelError(
                "candidate inadmissibility signal requires complete immutable evidence"
            )
        self.evidence = evidence
        super().__init__(evidence.failure_message, details=evidence.as_details())


@dataclass(frozen=True, slots=True)
class CovarianceDiagnostic:
    """Numerical diagnostics for the shared tied emission covariance."""

    eigenvalues: FloatArray
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    symmetry_max_absolute_error: float
    symmetric: bool
    positive_definite: bool
    eigenvalue_floor_passed: bool
    condition_number_passed: bool


@dataclass(frozen=True, slots=True)
class StateEpisode:
    """One contiguous decoded state spell within one source segment."""

    state: int
    start: int
    end_exclusive: int
    length: int
    segment: int
    left_censored: bool
    right_censored: bool


@dataclass(frozen=True, slots=True)
class StatePathDiagnostic:
    """Occupancy, episode, transition, and dwell evidence for a state path."""

    state_counts: IntArray
    occupancy: FloatArray
    transition_counts: IntArray
    episode_counts: IntArray
    completed_episode_counts: IntArray
    censored_episode_counts: IntArray
    all_one_period_episode_counts: IntArray
    one_period_episode_counts: IntArray
    all_dwell_lengths_by_state: tuple[IntArray, ...]
    dwell_lengths_by_state: tuple[IntArray, ...]
    episodes: tuple[StateEpisode, ...]
    lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MarkovChainDiagnostic:
    """Validated stationary and mixing diagnostics for a transition matrix."""

    stationary_distribution: FloatArray
    stationary_residual: float
    row_sum_max_absolute_error: float
    irreducible: bool
    aperiodic: bool
    period: int
    unique_stationary_distribution: bool
    eigenvalue_moduli: FloatArray
    second_largest_eigenvalue_modulus: float
    spectral_gap: float
    implied_relaxation_time: float
    mixing_time_to_one_percent: int
    maximum_total_variation_at_mixing: float
    self_transition_probabilities: FloatArray
    near_absorbing_states: tuple[int, ...]
    near_absorbing_threshold: float


@dataclass(frozen=True, slots=True)
class ForwardBackwardResult:
    """Segmented filtering, smoothing, and expected-transition evidence."""

    filtered_probabilities: FloatArray
    posterior_probabilities: FloatArray
    expected_transition_counts: FloatArray
    sequence_log_likelihoods: tuple[float, ...]
    total_log_likelihood: float
    lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PosteriorIdentificationReport:
    """Exact Section 4.5/4.6 posterior-identification gate evidence."""

    normalized_posterior_entropy: float
    mean_maximum_posterior: float
    posterior_effective_masses: FloatArray
    minimum_required_effective_mass: float
    assigned_state_mean_posteriors: FloatArray
    minimum_pairwise_mahalanobis_distance: float
    passed: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "normalized_posterior_entropy": self.normalized_posterior_entropy,
            "mean_maximum_posterior": self.mean_maximum_posterior,
            "posterior_effective_masses": self.posterior_effective_masses.tolist(),
            "minimum_required_effective_mass": self.minimum_required_effective_mass,
            "assigned_state_mean_posteriors": [
                None if not math.isfinite(float(value)) else float(value)
                for value in self.assigned_state_mean_posteriors
            ],
            "minimum_pairwise_mahalanobis_distance": (
                self.minimum_pairwise_mahalanobis_distance
            ),
            "passed": self.passed,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class PredictiveScoreResult:
    """Sequential one-step predictive scores within exact source segments."""

    log_scores: FloatArray
    predictive_state_probabilities: FloatArray
    filtered_probabilities: FloatArray
    terminal_filtered_probabilities: FloatArray
    total_log_score: float
    lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HistoricalTransitionSupportReport:
    """Numerical arrays and decision for the historically supported graph."""

    probabilities: FloatArray
    viterbi_counts: IntArray
    expected_counts: FloatArray
    supported_cells: BoolArray
    supported_off_diagonal_edges: BoolArray
    supported_self_loop_states: tuple[int, ...]
    decoded_entries: IntArray
    decoded_exits: IntArray
    bootstrap_edge_frequencies: FloatArray | None
    bootstrap_probability_quantiles: FloatArray | None
    strongly_connected: bool
    aperiodic: bool
    passed: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GaussianHMMFit:
    """Best converged restart represented without a pickle-only model object."""

    parameters: HMMParameters
    decoded_states: IntArray
    canonical_to_original: IntArray
    original_to_canonical: IntArray
    log_likelihood: float
    bic: float
    parameter_count: int
    n_observations: int
    n_features: int
    n_states: int
    lengths: tuple[int, ...]
    best_restart_index: int
    best_seed: int
    restart_diagnostics: tuple[RestartDiagnostic, ...]
    covariance_diagnostic: CovarianceDiagnostic
    state_path_diagnostics: StatePathDiagnostic
    chain_diagnostics: MarkovChainDiagnostic
    forward_backward: ForwardBackwardResult
    identification: PosteriorIdentificationReport
    historical_transition_support: HistoricalTransitionSupportReport
    scaler_scope: ScalerScope
    scaler_means: FloatArray
    scaler_standard_deviations: FloatArray
    feature_names: tuple[str, ...]
    owner_fixed_k4_disposition: OwnerFixedK4Disposition | None = None

    def __post_init__(self) -> None:
        validate_owner_fixed_k4_disposition(self)


@dataclass(frozen=True, slots=True)
class BridgeAcceleratedHMMFit:
    """Ten-initialization fixed-K bridge fit with explicit source mapping.

    Diagnostic index zero is the deterministic parent-parameter warm start.
    Diagnostic indices one through nine correspond to registered random
    restart IDs zero through eight.  The warm start reuses the first random
    seed only as a deterministic ``hmmlearn`` ``random_state`` placeholder;
    with ``init_params=''`` it introduces no additional random initialization.
    """

    fit: GaussianHMMFit
    parent_warm_diagnostic_index: int
    random_restart_ids: tuple[int, ...]
    random_diagnostic_indices: tuple[int, ...]
    best_initialization: str
    warm_random_state_placeholder_seed: int


@dataclass(frozen=True, slots=True)
class _HMMInitialization:
    diagnostic_index: int
    seed: int
    parent_parameters: HMMParameters | None


@dataclass(frozen=True, slots=True)
class _HMMInitializationJob:
    """Pickle-safe input for one independently auditable restart."""

    values: FloatArray
    lengths: tuple[int, ...]
    n_states: int
    initialization: _HMMInitialization
    model_factory: ModelFactory


@dataclass(frozen=True, slots=True)
class _HMMInitializationOutcome:
    """Plain scientific output returned from a restart worker to its parent."""

    diagnostic: RestartDiagnostic
    parameters: HMMParameters | None
    covariance: CovarianceDiagnostic | None
    chain: MarkovChainDiagnostic | None


class GaussianHMMLike(Protocol):
    """Small hmmlearn surface used by the deterministic restart runner."""

    monitor_: Any
    startprob_: FloatArray
    transmat_: FloatArray
    means_: FloatArray
    covars_: FloatArray
    _covars_: FloatArray
    raw_covariance_min_eigenvalues_: list[float]
    covariance_floor_relative_changes_: list[float]

    def fit(
        self, values: FloatArray, lengths: Sequence[int] | None = None
    ) -> GaussianHMMLike: ...

    def score(
        self, values: FloatArray, lengths: Sequence[int] | None = None
    ) -> float: ...


class AuditedTiedGaussianHMM(GaussianHMM):  # type: ignore[misc]
    """hmmlearn tied HMM with an audited floor after every M-step."""

    raw_covariance_min_eigenvalues_: list[float]
    covariance_floor_relative_changes_: list[float]
    _covars_: FloatArray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.covariance_type != "tied":
            raise HMMRestartNumericalFailure(
                "AuditedTiedGaussianHMM requires covariance_type='tied'"
            )
        self.raw_covariance_min_eigenvalues_ = []
        self.covariance_floor_relative_changes_ = []

    def _do_mstep(self, stats: dict[str, Any]) -> None:
        super()._do_mstep(stats)
        raw = np.asarray(self._covars_, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
            raise HMMRestartNumericalFailure(
                "tied covariance M-step did not produce a square matrix"
            )
        if not np.isfinite(raw).all():
            raise HMMRestartNumericalFailure(
                "tied covariance M-step produced non-finite values"
            )
        symmetric = 0.5 * (raw + raw.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        minimum = float(eigenvalues[0])
        floored = (eigenvectors * np.maximum(eigenvalues, self.min_covar)) @ (
            eigenvectors.T
        )
        floored = 0.5 * (floored + floored.T)
        denominator = float(np.linalg.norm(raw, ord="fro"))
        numerator = float(np.linalg.norm(floored - raw, ord="fro"))
        relative_change = math.inf if denominator == 0.0 else numerator / denominator
        self.raw_covariance_min_eigenvalues_.append(minimum)
        self.covariance_floor_relative_changes_.append(relative_change)
        self._covars_ = floored


ModelFactory = Callable[..., GaussianHMMLike]


def deterministic_hmm_restart_seeds(
    *,
    master_seed: int,
    scientific_version: int,
    scope: int | ModelFitScope,
    replicate: int,
    n_states: int,
    restart_count: int = HMM_RESTARTS,
) -> tuple[int, ...]:
    """Return explicit seeds for one deterministic model-fit restart library.

    The historical contracts use the default 50 restarts.  A caller building a
    new, noncanonical model may request a different positive count without
    changing how any existing seed is derived.
    """

    _require_positive_integer(n_states, "n_states")
    _require_positive_integer(restart_count, "restart_count")
    return tuple(
        hmm_library_seed(
            master_seed=master_seed,
            scientific_version=scientific_version,
            scope=scope,
            replicate=replicate,
            k=n_states,
            restart_index=restart,
        )
        for restart in range(restart_count)
    )


def sequence_lengths_from_continuation(
    continuation: Sequence[bool] | BoolArray,
) -> tuple[int, ...]:
    """Convert the processed continuation mask into positive segment lengths.

    ``False`` marks the first row and every row whose predecessor is not the
    immediately preceding eligible calendar month.  Thus ``[F,T,F,T,T]``
    becomes two independent sequences of lengths ``(2, 3)``.
    """

    mask = np.asarray(continuation)
    if mask.ndim != 1 or mask.size == 0:
        raise ModelError("continuation mask must be non-empty and one-dimensional")
    if mask.dtype.kind != "b":
        raise ModelError("continuation mask must have boolean dtype")
    if bool(mask[0]):
        raise ModelError("the first continuation-mask value must be false")
    starts = np.flatnonzero(~mask)
    boundaries = np.append(starts, mask.size)
    lengths = tuple(int(value) for value in np.diff(boundaries))
    _validate_lengths(lengths, int(mask.size))
    return lengths


def consume_standardized_features(
    standardized: StandardizedFeatures,
    *,
    continuation: Sequence[bool] | BoolArray,
    scaler_scope: ScalerScope,
    fit_index: pd.Index | None = None,
    zscore_tolerance: float = 1e-10,
) -> HMMFeatureMatrix:
    """Create the exact training-only or full-data HMM fitting matrix.

    Design fitting requires an explicit, proper subset and verifies that the
    selected rows—not the full frame—have population mean zero and standard
    deviation one.  This catches accidental full-sample scaler leakage.  A
    production fit rejects a row subset and consumes every standardized row.
    Source continuation breaks are retained, and selecting non-adjacent rows
    introduces additional breaks rather than inventing transitions.
    """

    if scaler_scope not in {"design_training", "production_full"}:
        raise ModelError(
            "scaler scope is invalid", details={"scaler_scope": scaler_scope}
        )
    if not np.isfinite(zscore_tolerance) or zscore_tolerance <= 0.0:
        raise ModelError("z-score verification tolerance must be finite and positive")
    values = standardized.values
    if not isinstance(values, pd.DataFrame) or values.empty:
        raise ModelError("standardized feature table is empty or invalid")
    if not values.index.is_unique or not values.columns.is_unique:
        raise ModelError("standardized feature rows and columns must be unique")
    if not standardized.means.index.equals(values.columns) or not (
        standardized.standard_deviations.index.equals(values.columns)
    ):
        raise ModelError("scaler parameters do not align with feature columns")
    full_mask = np.asarray(continuation)
    if full_mask.ndim != 1 or len(full_mask) != len(values):
        raise ModelError(
            "continuation mask length does not match standardized features",
            details={"mask_rows": len(full_mask), "feature_rows": len(values)},
        )
    # Validate the source mask even when a training subset later adds breaks.
    sequence_lengths_from_continuation(full_mask)

    if scaler_scope == "production_full":
        if fit_index is not None:
            raise ModelError("production-full scaler consumption forbids a row subset")
        selected = values
        selected_mask = full_mask.astype(np.bool_, copy=True)
    else:
        if fit_index is None:
            raise ModelError("design-training scaler consumption requires fit_index")
        requested = pd.Index(fit_index)
        if requested.empty or not requested.is_unique:
            raise ModelError("design-training fit_index must be non-empty and unique")
        positions = values.index.get_indexer(requested)
        if bool((positions < 0).any()):
            missing = [str(requested[index]) for index in np.flatnonzero(positions < 0)]
            raise ModelError(
                "design-training fit_index contains unknown rows",
                details={"missing_rows": missing},
            )
        if bool((np.diff(positions) <= 0).any()):
            raise ModelError("design-training fit_index must preserve source order")
        if len(positions) >= len(values):
            raise ModelError("design-training fit_index must be a proper subset")
        selected = values.iloc[positions]
        selected_mask = np.zeros(len(positions), dtype=np.bool_)
        if len(positions) > 1:
            selected_mask[1:] = (np.diff(positions) == 1) & full_mask[
                positions[1:]
            ].astype(bool)

    matrix = selected.to_numpy(dtype=np.float64, copy=True)
    means = standardized.means.to_numpy(dtype=np.float64, copy=True)
    standard_deviations = standardized.standard_deviations.to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.isfinite(matrix).all():
        raise ModelError("standardized HMM fitting matrix is non-finite")
    if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
        raise ModelError("feature scaler parameters are non-finite")
    if bool((standard_deviations <= 0.0).any()):
        raise ModelError("feature scaler standard deviations must be positive")
    fitted_means = matrix.mean(axis=0)
    fitted_scales = matrix.std(axis=0, ddof=0)
    if not np.allclose(fitted_means, 0.0, rtol=0.0, atol=zscore_tolerance) or (
        not np.allclose(fitted_scales, 1.0, rtol=0.0, atol=zscore_tolerance)
    ):
        raise ModelError(
            "selected HMM rows are inconsistent with the declared scaler scope",
            details={
                "maximum_absolute_mean": float(np.max(np.abs(fitted_means))),
                "maximum_absolute_scale_error": float(
                    np.max(np.abs(fitted_scales - 1.0))
                ),
            },
        )
    lengths = sequence_lengths_from_continuation(selected_mask)
    return HMMFeatureMatrix(
        values=matrix,
        lengths=lengths,
        feature_names=tuple(str(column) for column in values.columns),
        row_labels=tuple(str(index) for index in selected.index),
        scaler_means=means,
        scaler_standard_deviations=standard_deviations,
        scaler_scope=scaler_scope,
    )


def gaussian_hmm_parameter_count(n_states: int, n_features: int) -> int:
    """Return the exact tied-covariance free-parameter count."""

    _require_positive_integer(n_states, "n_states")
    _require_positive_integer(n_features, "n_features")
    return (
        (n_states - 1)
        + n_states * (n_states - 1)
        + n_states * n_features
        + n_features * (n_features + 1) // 2
    )


def gaussian_hmm_bic(
    log_likelihood: float,
    n_observations: int,
    n_states: int,
    n_features: int,
) -> float:
    """Calculate ``-2 log(L) + q log(T)`` using the exact ``q`` above."""

    if not np.isfinite(log_likelihood):
        raise ModelError("HMM log likelihood must be finite for BIC")
    _require_positive_integer(n_observations, "n_observations")
    count = gaussian_hmm_parameter_count(n_states, n_features)
    return float(-2.0 * log_likelihood + count * math.log(n_observations))


def canonicalize_hmm(
    parameters: HMMParameters,
    *,
    states: Sequence[int] | IntArray | None = None,
) -> CanonicalizedHMM:
    """Canonicalize labels by centroid lexicographic order.

    Feature zero is the primary centroid key, feature one the next key, and so
    on.  Exact centroid ties are broken by the original integer state ID.  The
    resulting ``canonical_to_original`` permutation is applied to start
    probabilities, both axes of the transition matrix, means, and every
    supplied decoded state.  The tied covariance is label-invariant.
    """

    start, transition, means, tied_covariance = _validated_parameter_arrays(parameters)
    n_states = len(start)
    order = np.asarray(
        sorted(
            range(n_states),
            key=lambda state: (*means[state].tolist(), state),
        ),
        dtype=np.int64,
    )
    inverse = np.empty(n_states, dtype=np.int64)
    inverse[order] = np.arange(n_states, dtype=np.int64)
    remapped_states: IntArray | None = None
    if states is not None:
        raw_states = _validated_states(states, n_states)
        remapped_states = inverse[raw_states]
    canonical = HMMParameters(
        start_probabilities=start[order].copy(),
        transition_matrix=transition[np.ix_(order, order)].copy(),
        means=means[order].copy(),
        tied_covariance=tied_covariance.copy(),
    )
    return CanonicalizedHMM(canonical, order, inverse, remapped_states)


def decode_viterbi(
    values: Sequence[Sequence[float]] | FloatArray,
    parameters: HMMParameters,
    *,
    lengths: Sequence[int] | None = None,
) -> IntArray:
    """Decode the unsmoothed maximum-probability path for each source segment."""

    matrix = _validated_matrix(values, "Viterbi feature matrix")
    start, transition, means, tied_covariance = _validated_parameter_arrays(parameters)
    if matrix.shape[1] != means.shape[1]:
        raise ModelError("Viterbi feature dimension does not match HMM means")
    sequence_lengths = _validated_lengths_or_single(lengths, len(matrix))
    emissions = _gaussian_log_emissions(matrix, means, tied_covariance)
    log_start = _safe_log_probabilities(start)
    log_transition = _safe_log_probabilities(transition)
    decoded = np.empty(len(matrix), dtype=np.int64)
    offset = 0
    for length in sequence_lengths:
        segment_emissions = emissions[offset : offset + length]
        scores = np.empty((length, len(start)), dtype=np.float64)
        predecessors = np.zeros((length, len(start)), dtype=np.int64)
        scores[0] = log_start + segment_emissions[0]
        if not np.isfinite(scores[0]).any():
            raise ModelError("Viterbi segment has no reachable initial state")
        for row in range(1, length):
            candidates = scores[row - 1][:, None] + log_transition
            predecessors[row] = np.argmax(candidates, axis=0)
            scores[row] = (
                candidates[predecessors[row], np.arange(len(start))]
                + segment_emissions[row]
            )
            if not np.isfinite(scores[row]).any():
                raise ModelError(
                    "Viterbi segment has no reachable path",
                    details={"segment_offset": offset, "row_in_segment": row},
                )
        state = int(np.argmax(scores[-1]))
        decoded[offset + length - 1] = state
        for row in range(length - 1, 0, -1):
            state = int(predecessors[row, state])
            decoded[offset + row - 1] = state
        offset += length
    return decoded


def tied_covariance_diagnostic(
    covariance: Sequence[Sequence[float]] | FloatArray,
    *,
    eigenvalue_floor: float = DEFAULT_EIGENVALUE_FLOOR,
    maximum_condition_number: float = DEFAULT_MAX_COVARIANCE_CONDITION,
    symmetry_tolerance: float = 1e-12,
) -> CovarianceDiagnostic:
    """Return eigenvalue, symmetry, and condition diagnostics for tied Sigma."""

    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
        raise ModelError("tied covariance must be a non-empty square matrix")
    if not np.isfinite(values).all():
        raise ModelError("tied covariance contains non-finite values")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor < 0.0:
        raise ModelError("covariance eigenvalue floor must be finite and non-negative")
    if not np.isfinite(maximum_condition_number) or maximum_condition_number <= 0.0:
        raise ModelError("maximum covariance condition number must be positive")
    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance < 0.0:
        raise ModelError("covariance symmetry tolerance must be non-negative")
    symmetry_error = float(np.max(np.abs(values - values.T)))
    symmetric = symmetry_error <= symmetry_tolerance
    symmetric_covariance = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_covariance)
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    condition = math.inf if minimum <= 0.0 else float(maximum / minimum)
    return CovarianceDiagnostic(
        eigenvalues=eigenvalues.astype(np.float64, copy=True),
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        condition_number=condition,
        symmetry_max_absolute_error=symmetry_error,
        symmetric=symmetric,
        positive_definite=symmetric and minimum > 0.0,
        eigenvalue_floor_passed=symmetric and minimum >= eigenvalue_floor,
        condition_number_passed=(symmetric and condition <= maximum_condition_number),
    )


def covariance_diagnostics(
    covariance: Sequence[Sequence[float]] | FloatArray,
    **kwargs: float,
) -> tuple[CovarianceDiagnostic, ...]:
    """Compatibility wrapper returning the one tied-covariance diagnostic."""

    return (tied_covariance_diagnostic(covariance, **kwargs),)


def summarize_state_path(
    states: Sequence[int] | IntArray,
    n_states: int,
    *,
    lengths: Sequence[int] | None = None,
) -> StatePathDiagnostic:
    """Summarize states without counting transitions across source gaps."""

    _require_positive_integer(n_states, "n_states")
    path = _validated_states(states, n_states)
    sequence_lengths = _validated_lengths_or_single(lengths, len(path))
    counts = np.bincount(path, minlength=n_states).astype(np.int64)
    occupancy = counts.astype(np.float64) / len(path)
    transitions = np.zeros((n_states, n_states), dtype=np.int64)
    episodes: list[StateEpisode] = []
    offset = 0
    for segment, length in enumerate(sequence_lengths):
        segment_path = path[offset : offset + length]
        if length > 1:
            np.add.at(transitions, (segment_path[:-1], segment_path[1:]), 1)
        start = 0
        for position in range(1, length + 1):
            if position == length or segment_path[position] != segment_path[start]:
                episodes.append(
                    StateEpisode(
                        state=int(segment_path[start]),
                        start=offset + start,
                        end_exclusive=offset + position,
                        length=position - start,
                        segment=segment,
                        left_censored=start == 0,
                        right_censored=position == length,
                    )
                )
                start = position
        offset += length
    episode_counts = np.zeros(n_states, dtype=np.int64)
    completed_counts = np.zeros(n_states, dtype=np.int64)
    all_one_period_counts = np.zeros(n_states, dtype=np.int64)
    completed_one_period_counts = np.zeros(n_states, dtype=np.int64)
    all_dwell_lists: list[list[int]] = [[] for _ in range(n_states)]
    completed_dwell_lists: list[list[int]] = [[] for _ in range(n_states)]
    for episode in episodes:
        episode_counts[episode.state] += 1
        all_one_period_counts[episode.state] += int(episode.length == 1)
        all_dwell_lists[episode.state].append(episode.length)
        if not episode.left_censored and not episode.right_censored:
            completed_counts[episode.state] += 1
            completed_one_period_counts[episode.state] += int(episode.length == 1)
            completed_dwell_lists[episode.state].append(episode.length)
    return StatePathDiagnostic(
        state_counts=counts,
        occupancy=occupancy,
        transition_counts=transitions,
        episode_counts=episode_counts,
        completed_episode_counts=completed_counts,
        censored_episode_counts=episode_counts - completed_counts,
        all_one_period_episode_counts=all_one_period_counts,
        one_period_episode_counts=completed_one_period_counts,
        all_dwell_lengths_by_state=tuple(
            np.asarray(values, dtype=np.int64) for values in all_dwell_lists
        ),
        dwell_lengths_by_state=tuple(
            np.asarray(values, dtype=np.int64) for values in completed_dwell_lists
        ),
        episodes=tuple(episodes),
        lengths=sequence_lengths,
    )


def validate_numerical_chain(
    transition_matrix: Sequence[Sequence[float]] | FloatArray,
) -> MarkovChainDiagnostic:
    """Apply the frozen numerical graph, stationarity, and exact-TV gates."""

    raw = np.asarray(transition_matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
        raise HMMRestartNumericalFailure(
            "transition matrix must be non-empty and square"
        )
    if not np.isfinite(raw).all():
        raise HMMRestartNumericalFailure("transition matrix contains non-finite values")
    minimum = float(raw.min())
    maximum = float(raw.max())
    if minimum < -NEGATIVE_PROBABILITY_TOLERANCE or (
        maximum > 1.0 + NEGATIVE_PROBABILITY_TOLERANCE
    ):
        raise HMMRestartNumericalFailure(
            "transition probabilities exceed numerical bounds",
            details={"minimum": minimum, "maximum": maximum},
        )
    row_error = float(np.max(np.abs(raw.sum(axis=1) - 1.0)))
    if row_error > PROBABILITY_ATOL:
        raise HMMRestartNumericalFailure(
            "transition matrix is not row-stochastic",
            details={
                "maximum_absolute_row_sum_error": row_error,
                "tolerance": PROBABILITY_ATOL,
            },
        )

    # Values in [-1e-15, 0) are accepted numerical noise.  Clip and renormalize
    # only for linear algebra; the reported row error remains that of the input.
    matrix = np.maximum(raw, 0.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    adjacency = matrix > NUMERICAL_EDGE_THRESHOLD
    if not _is_irreducible(adjacency):
        raise HMMRestartNumericalFailure(
            "transition matrix numerical graph is reducible"
        )
    period = _chain_period(adjacency)
    if period != 1:
        raise HMMRestartNumericalFailure(
            "transition matrix numerical graph is periodic", details={"period": period}
        )

    self_probabilities = np.diag(matrix).astype(np.float64, copy=True)
    near_absorbing = tuple(
        int(state)
        for state in np.flatnonzero(self_probabilities >= MAXIMUM_SELF_TRANSITION)
    )
    if near_absorbing:
        raise HMMRestartNumericalFailure(
            "transition matrix has a self-transition at or above 0.98",
            details={"states": list(near_absorbing)},
        )

    stationary = _stationary_distribution(matrix)
    if bool((stationary < -NEGATIVE_PROBABILITY_TOLERANCE).any()) or not (
        np.isfinite(stationary).all()
    ):
        raise HMMRestartNumericalFailure(
            "stationary distribution is not finite and non-negative"
        )
    stationary = np.maximum(stationary, 0.0)
    stationary = stationary / stationary.sum()
    residual = float(np.max(np.abs(stationary @ matrix - stationary)))
    if residual > STATIONARY_RESIDUAL_ATOL:
        raise HMMRestartNumericalFailure(
            "stationary distribution residual exceeds tolerance",
            details={"residual": residual, "tolerance": STATIONARY_RESIDUAL_ATOL},
        )

    moduli = np.sort(np.abs(np.linalg.eigvals(matrix)).astype(np.float64))[::-1]
    second = float(moduli[1]) if len(moduli) > 1 else 0.0
    second = min(max(second, 0.0), 1.0)
    gap = float(1.0 - second)
    relaxation = math.inf if gap <= 0.0 else float(1.0 / gap)

    power = np.eye(len(matrix), dtype=np.float64)
    mixing_time: int | None = None
    mixing_tv = math.inf
    for month in range(1, MAXIMUM_MIXING_MONTHS + 1):
        power = power @ matrix
        distances = 0.5 * np.sum(np.abs(power - stationary[None, :]), axis=1)
        mixing_tv = float(np.max(distances))
        if mixing_tv <= MIXING_TOTAL_VARIATION_THRESHOLD:
            mixing_time = month
            break
    if mixing_time is None:
        raise HMMRestartNumericalFailure(
            "transition matrix does not mix within 120 months",
            details={"total_variation_at_120": mixing_tv},
        )

    return MarkovChainDiagnostic(
        stationary_distribution=_readonly(stationary),
        stationary_residual=residual,
        row_sum_max_absolute_error=row_error,
        irreducible=True,
        aperiodic=True,
        period=period,
        unique_stationary_distribution=True,
        eigenvalue_moduli=_readonly(moduli),
        second_largest_eigenvalue_modulus=second,
        spectral_gap=gap,
        implied_relaxation_time=relaxation,
        mixing_time_to_one_percent=mixing_time,
        maximum_total_variation_at_mixing=mixing_tv,
        self_transition_probabilities=_readonly(self_probabilities),
        near_absorbing_states=(),
        near_absorbing_threshold=MAXIMUM_SELF_TRANSITION,
    )


def validate_markov_chain(
    transition_matrix: Sequence[Sequence[float]] | FloatArray,
) -> MarkovChainDiagnostic:
    """Backward-compatible name for the canonical numerical-chain gate."""

    return validate_numerical_chain(transition_matrix)


def forward_backward(
    values: Sequence[Sequence[float]] | FloatArray,
    parameters: HMMParameters,
    *,
    lengths: Sequence[int] | None = None,
) -> ForwardBackwardResult:
    """Run exact segmented log-space filtering, smoothing, and xi accumulation."""

    matrix = _validated_matrix(values, "forward-backward feature matrix")
    start, transition, means, tied_covariance = _validated_parameter_arrays(parameters)
    if matrix.shape[1] != means.shape[1]:
        raise ModelError("forward-backward feature dimension does not match HMM means")
    sequence_lengths = _validated_lengths_or_single(lengths, len(matrix))
    emissions = _gaussian_log_emissions(matrix, means, tied_covariance)
    log_start = _safe_log_probabilities(start)
    log_transition = _safe_log_probabilities(transition)
    filtered = np.empty((len(matrix), len(start)), dtype=np.float64)
    posterior = np.empty_like(filtered)
    expected_counts = np.zeros((len(start), len(start)), dtype=np.float64)
    sequence_likelihoods: list[float] = []
    offset = 0
    for segment, length in enumerate(sequence_lengths):
        segment_emissions = emissions[offset : offset + length]
        log_alpha = np.empty((length, len(start)), dtype=np.float64)
        log_beta = np.zeros_like(log_alpha)
        log_alpha[0] = log_start + segment_emissions[0]
        if not np.isfinite(log_alpha[0]).any():
            raise ModelError(
                "forward-backward segment has no reachable initial state",
                details={"segment": segment},
            )
        filtered[offset] = np.exp(log_alpha[0] - logsumexp(log_alpha[0]))
        for row in range(1, length):
            log_alpha[row] = segment_emissions[row] + logsumexp(
                log_alpha[row - 1][:, None] + log_transition, axis=0
            )
            if not np.isfinite(log_alpha[row]).any():
                raise ModelError(
                    "forward-backward segment has no reachable path",
                    details={"segment": segment, "row_in_segment": row},
                )
            filtered[offset + row] = np.exp(log_alpha[row] - logsumexp(log_alpha[row]))
        segment_likelihood = float(logsumexp(log_alpha[-1]))
        if not math.isfinite(segment_likelihood):
            raise ModelError("forward-backward segment likelihood is non-finite")
        sequence_likelihoods.append(segment_likelihood)
        for row in range(length - 2, -1, -1):
            log_beta[row] = logsumexp(
                log_transition
                + segment_emissions[row + 1][None, :]
                + log_beta[row + 1][None, :],
                axis=1,
            )
        segment_posterior = np.exp(log_alpha + log_beta - segment_likelihood)
        row_sums = segment_posterior.sum(axis=1, keepdims=True)
        if bool((row_sums <= 0.0).any()) or not np.isfinite(row_sums).all():
            raise ModelError("forward-backward posterior normalization failed")
        segment_posterior = segment_posterior / row_sums
        posterior[offset : offset + length] = segment_posterior
        for row in range(length - 1):
            log_xi = (
                log_alpha[row][:, None]
                + log_transition
                + segment_emissions[row + 1][None, :]
                + log_beta[row + 1][None, :]
                - segment_likelihood
            )
            xi = np.exp(log_xi)
            xi_sum = float(xi.sum())
            if not math.isfinite(xi_sum) or xi_sum <= 0.0:
                raise ModelError("forward-backward xi normalization failed")
            expected_counts += xi / xi_sum
        offset += length
    total = float(math.fsum(sequence_likelihoods))
    if not math.isfinite(total):
        raise ModelError("forward-backward total likelihood is non-finite")
    return ForwardBackwardResult(
        filtered_probabilities=_readonly(filtered),
        posterior_probabilities=_readonly(posterior),
        expected_transition_counts=_readonly(expected_counts),
        sequence_log_likelihoods=tuple(sequence_likelihoods),
        total_log_likelihood=total,
        lengths=sequence_lengths,
    )


def posterior_identification_report(
    posterior_probabilities: Sequence[Sequence[float]] | FloatArray,
    decoded_states: Sequence[int] | IntArray,
    parameters: HMMParameters,
) -> PosteriorIdentificationReport:
    """Calculate every binding posterior-identification screen exactly."""

    start, _transition, means, tied_covariance = _validated_parameter_arrays(parameters)
    gamma = np.asarray(posterior_probabilities, dtype=np.float64)
    if gamma.ndim != 2 or gamma.shape[1] != len(start) or gamma.shape[0] == 0:
        raise ModelError("posterior probability matrix shape is inconsistent")
    if not np.isfinite(gamma).all() or bool((gamma < 0.0).any()):
        raise ModelError("posterior probabilities must be finite and non-negative")
    row_error = float(np.max(np.abs(gamma.sum(axis=1) - 1.0)))
    if row_error > PROBABILITY_ATOL:
        raise ModelError(
            "posterior probability rows must sum to one",
            details={"maximum_absolute_row_sum_error": row_error},
        )
    states = _validated_states(decoded_states, len(start))
    if len(states) != len(gamma):
        raise ModelError("decoded states and posterior rows must have equal lengths")

    positive = gamma > 0.0
    entropy_terms = np.zeros_like(gamma)
    entropy_terms[positive] = gamma[positive] * np.log(gamma[positive])
    if len(start) == 1:
        normalized_entropy = 0.0
    else:
        normalized_entropy = float(
            -entropy_terms.sum() / (len(gamma) * math.log(len(start)))
        )
    mean_maximum = float(np.max(gamma, axis=1).mean())
    effective_masses = gamma.sum(axis=0)
    required_mass = float(
        max(
            MINIMUM_POSTERIOR_EFFECTIVE_MASS,
            MINIMUM_POSTERIOR_EFFECTIVE_FRACTION * len(gamma),
        )
    )
    assigned = np.full(len(start), np.nan, dtype=np.float64)
    reasons: list[str] = []
    for state in range(len(start)):
        mask = states == state
        if bool(mask.any()):
            assigned[state] = float(gamma[mask, state].mean())
        else:
            reasons.append(f"state_{state}_has_no_viterbi_assigned_row")

    minimum_distance = math.inf
    try:
        for first in range(len(start)):
            for second in range(first + 1, len(start)):
                difference = means[first] - means[second]
                solved = np.linalg.solve(tied_covariance, difference)
                squared = float(difference @ solved)
                if not math.isfinite(squared) or squared < -1e-12:
                    raise np.linalg.LinAlgError
                minimum_distance = min(minimum_distance, math.sqrt(max(squared, 0.0)))
    except np.linalg.LinAlgError as error:
        raise ModelError("Mahalanobis separation solve failed") from error

    if normalized_entropy > MAXIMUM_NORMALIZED_POSTERIOR_ENTROPY:
        reasons.append("normalized_posterior_entropy_above_maximum")
    if mean_maximum < MINIMUM_MEAN_MAXIMUM_POSTERIOR:
        reasons.append("mean_maximum_posterior_below_minimum")
    for state in np.flatnonzero(effective_masses < required_mass):
        reasons.append(f"state_{int(state)}_posterior_effective_mass_below_minimum")
    for state, value in enumerate(assigned):
        if math.isfinite(float(value)) and value < MINIMUM_ASSIGNED_STATE_POSTERIOR:
            reasons.append(f"state_{state}_assigned_posterior_below_minimum")
    if minimum_distance < MINIMUM_PAIRWISE_MAHALANOBIS_DISTANCE:
        reasons.append("pairwise_mahalanobis_distance_below_minimum")
    return PosteriorIdentificationReport(
        normalized_posterior_entropy=normalized_entropy,
        mean_maximum_posterior=mean_maximum,
        posterior_effective_masses=_readonly(effective_masses.astype(np.float64)),
        minimum_required_effective_mass=required_mass,
        assigned_state_mean_posteriors=_readonly(assigned),
        minimum_pairwise_mahalanobis_distance=minimum_distance,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


def one_step_predictive_log_scores(
    values: Sequence[Sequence[float]] | FloatArray,
    parameters: HMMParameters,
    *,
    lengths: Sequence[int] | None = None,
    initial_state_probabilities: Sequence[Sequence[float]] | FloatArray | None = None,
) -> PredictiveScoreResult:
    """Score each row using only prior rows in its exact source segment.

    ``initial_state_probabilities`` are predictive weights immediately before
    each segment's first emission.  A contiguous holdout therefore supplies
    ``terminal_filter @ P``; a post-gap segment supplies the stationary law.
    """

    matrix = _validated_matrix(values, "predictive-score feature matrix")
    start, transition, means, tied_covariance = _validated_parameter_arrays(parameters)
    if matrix.shape[1] != means.shape[1]:
        raise ModelError("predictive-score feature dimension does not match HMM means")
    sequence_lengths = _validated_lengths_or_single(lengths, len(matrix))
    initial = _validated_initial_state_probabilities(
        initial_state_probabilities,
        n_segments=len(sequence_lengths),
        n_states=len(start),
        default=start,
    )
    emissions = _gaussian_log_emissions(matrix, means, tied_covariance)
    scores = np.empty(len(matrix), dtype=np.float64)
    priors = np.empty((len(matrix), len(start)), dtype=np.float64)
    filtered = np.empty_like(priors)
    terminal = np.empty((len(sequence_lengths), len(start)), dtype=np.float64)
    offset = 0
    for segment, length in enumerate(sequence_lengths):
        weights = initial[segment].copy()
        for row in range(length):
            absolute_row = offset + row
            priors[absolute_row] = weights
            joint = _safe_log_probabilities(weights) + emissions[absolute_row]
            score = float(logsumexp(joint))
            if not math.isfinite(score):
                raise ModelError(
                    "one-step predictive density is non-finite",
                    details={"segment": segment, "row_in_segment": row},
                )
            scores[absolute_row] = score
            posterior = np.exp(joint - score)
            posterior = posterior / posterior.sum()
            filtered[absolute_row] = posterior
            weights = posterior @ transition
        terminal[segment] = filtered[offset + length - 1]
        offset += length
    return PredictiveScoreResult(
        log_scores=_readonly(scores),
        predictive_state_probabilities=_readonly(priors),
        filtered_probabilities=_readonly(filtered),
        terminal_filtered_probabilities=_readonly(terminal),
        total_log_score=float(math.fsum(float(value) for value in scores)),
        lengths=sequence_lengths,
    )


def historical_transition_support_report(
    transition_matrix: Sequence[Sequence[float]] | FloatArray,
    viterbi_transition_counts: Sequence[Sequence[int]] | IntArray,
    expected_transition_counts: Sequence[Sequence[float]] | FloatArray,
    *,
    bootstrap_edge_frequencies: Sequence[Sequence[float]] | FloatArray | None = None,
    bootstrap_probability_quantiles: FloatArray | None = None,
) -> HistoricalTransitionSupportReport:
    """Build the exact supported-cell graph and decoded entry/exit evidence."""

    probabilities = np.asarray(transition_matrix, dtype=np.float64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or (probabilities.shape[0] != probabilities.shape[1])
    ):
        raise ModelError("historical transition probabilities must be square")
    if not np.isfinite(probabilities).all():
        raise ModelError("historical transition probabilities must be finite")
    shape = probabilities.shape
    raw_viterbi = np.asarray(viterbi_transition_counts)
    if raw_viterbi.shape != shape or raw_viterbi.dtype.kind not in {"i", "u"}:
        raise ModelError("Viterbi transition counts must be a matching integer matrix")
    viterbi = raw_viterbi.astype(np.int64, copy=True)
    if bool((viterbi < 0).any()):
        raise ModelError("Viterbi transition counts must be non-negative")
    expected = np.asarray(expected_transition_counts, dtype=np.float64)
    if (
        expected.shape != shape
        or not np.isfinite(expected).all()
        or bool((expected < 0.0).any())
    ):
        raise ModelError("expected transition counts must be finite and non-negative")
    frequencies = _optional_probability_matrix(
        bootstrap_edge_frequencies, shape, "bootstrap edge frequencies"
    )
    quantiles = _optional_transition_quantiles(bootstrap_probability_quantiles, shape)

    supported = (probabilities >= 0.01) & (expected >= 2.0) & (viterbi >= 1)
    off_diagonal = supported.copy()
    np.fill_diagonal(off_diagonal, False)
    supported_self = tuple(int(value) for value in np.flatnonzero(np.diag(supported)))
    strongly_connected = _is_irreducible(off_diagonal)
    decoded_exits = np.sum(np.where(off_diagonal, viterbi, 0), axis=1).astype(np.int64)
    decoded_entries = np.sum(np.where(off_diagonal, viterbi, 0), axis=0).astype(
        np.int64
    )
    reasons: list[str] = []
    if not strongly_connected:
        reasons.append("supported_off_diagonal_graph_not_strongly_connected")
    if not supported_self:
        reasons.append("no_historically_supported_self_loop")
    for state in np.flatnonzero(decoded_entries < 3):
        reasons.append(f"state_{int(state)}_decoded_entries_below_three")
    for state in np.flatnonzero(decoded_exits < 3):
        reasons.append(f"state_{int(state)}_decoded_exits_below_three")
    return HistoricalTransitionSupportReport(
        probabilities=_readonly(probabilities.copy()),
        viterbi_counts=_readonly(viterbi),
        expected_counts=_readonly(expected.copy()),
        supported_cells=_readonly(supported),
        supported_off_diagonal_edges=_readonly(off_diagonal),
        supported_self_loop_states=supported_self,
        decoded_entries=_readonly(decoded_entries),
        decoded_exits=_readonly(decoded_exits),
        bootstrap_edge_frequencies=frequencies,
        bootstrap_probability_quantiles=quantiles,
        strongly_connected=strongly_connected,
        aperiodic=strongly_connected and bool(supported_self),
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


def validate_historical_transition_support(
    transition_matrix: Sequence[Sequence[float]] | FloatArray,
    viterbi_transition_counts: Sequence[Sequence[int]] | IntArray,
    expected_transition_counts: Sequence[Sequence[float]] | FloatArray,
    *,
    bootstrap_edge_frequencies: Sequence[Sequence[float]] | FloatArray | None = None,
    bootstrap_probability_quantiles: FloatArray | None = None,
) -> HistoricalTransitionSupportReport:
    """Require the approved historically supported transition graph."""

    report = historical_transition_support_report(
        transition_matrix,
        viterbi_transition_counts,
        expected_transition_counts,
        bootstrap_edge_frequencies=bootstrap_edge_frequencies,
        bootstrap_probability_quantiles=bootstrap_probability_quantiles,
    )
    if not report.passed:
        raise ModelError(
            "historically supported transition graph failed",
            details={"rejection_reasons": list(report.rejection_reasons)},
        )
    return report


def _post_fit_candidate_inadmissibility(
    *,
    n_states: int,
    failure_code: HMMCandidateInadmissibilityCode,
    failure_message: str,
    rejection_reasons: tuple[str, ...],
    gate_details: dict[str, object],
    best_restart_index: int,
    best_seed: int,
    restart_diagnostics: Sequence[RestartDiagnostic],
) -> HMMCandidateInadmissibility:
    """Build the closed typed signal from code-owned post-fit gate evidence."""

    return HMMCandidateInadmissibility(
        HMMCandidateInadmissibilityEvidence(
            n_states=n_states,
            failure_code=failure_code,
            failure_message=failure_message,
            rejection_reasons=rejection_reasons,
            gate_details=_frozen_candidate_gate_details(gate_details),
            best_restart_index=best_restart_index,
            best_seed=best_seed,
            restart_diagnostics=tuple(restart_diagnostics),
        )
    )


def fit_gaussian_hmm_restarts(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
    *,
    model_factory: ModelFactory = AuditedTiedGaussianHMM,
    workers: int = 1,
    owner_fixed_k4_policy: OwnerFixedK4Policy | None = None,
) -> GaussianHMMFit:
    """Fit, audit, and rank exactly 50 binding tied-HMM restarts.

    Restart jobs are independent and their ordered plain-data outcomes are
    returned to this parent process.  The parent alone selects the winner and
    constructs the authoritative :class:`GaussianHMMFit`.
    """

    _require_positive_integer(workers, "workers")
    if len(seeds) != HMM_RESTARTS:
        raise ModelError(
            "exactly 50 deterministic HMM seeds are required",
            details={"actual": len(seeds), "required": HMM_RESTARTS},
        )
    checked_seeds = tuple(_validated_seed(seed) for seed in seeds)
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ModelError("deterministic HMM restart seeds must be distinct")
    initializations = tuple(
        _HMMInitialization(index, seed, None)
        for index, seed in enumerate(checked_seeds)
    )
    return _fit_gaussian_hmm_initializations(
        features,
        n_states,
        initializations,
        model_factory=model_factory,
        workers=int(workers),
        owner_fixed_k4_policy=owner_fixed_k4_policy,
    )


def fit_gaussian_hmm_reduced_diagnostic_restarts(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
    *,
    model_factory: ModelFactory = AuditedTiedGaussianHMM,
    workers: int = 1,
    owner_fixed_k4_policy: OwnerFixedK4Policy | None = None,
) -> GaussianHMMFit:
    """Run a strict subset of ordinary starts for a noncanonical diagnostic.

    The numerical path is exactly the ordinary audited HMM engine, but this
    entry point requires fewer than the 50 binding starts.  Its output is
    therefore suitable only for explicitly noncanonical rehearsals and
    diagnostics; canonical executors continue to call
    :func:`fit_gaussian_hmm_restarts` and require all 50 starts.
    """

    _require_positive_integer(workers, "workers")
    if not 1 <= len(seeds) < HMM_RESTARTS:
        raise ModelError(
            "reduced HMM diagnostic requires between 1 and 49 seeds",
            details={"actual": len(seeds), "canonical": HMM_RESTARTS},
        )
    checked_seeds = tuple(_validated_seed(seed) for seed in seeds)
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ModelError("reduced HMM diagnostic seeds must be distinct")
    initializations = tuple(
        _HMMInitialization(index, seed, None)
        for index, seed in enumerate(checked_seeds)
    )
    return _fit_gaussian_hmm_initializations(
        features,
        n_states,
        initializations,
        model_factory=model_factory,
        workers=int(workers),
        owner_fixed_k4_policy=owner_fixed_k4_policy,
    )


def fit_gaussian_hmm_bridge_accelerated(
    features: HMMFeatureMatrix,
    n_states: int,
    random_seeds: Sequence[int],
    parent_parameters: HMMParameters,
    *,
    model_factory: ModelFactory = AuditedTiedGaussianHMM,
    workers: int = 1,
    owner_fixed_k4_policy: OwnerFixedK4Policy | None = None,
) -> BridgeAcceleratedHMMFit:
    """Fit the predeclared parent-warm plus random-start 0..8 bridge set."""

    _require_positive_integer(workers, "workers")
    if len(random_seeds) != 9:
        raise ModelError(
            "accelerated bridge fit requires exactly nine registered random seeds"
        )
    checked_seeds = tuple(_validated_seed(seed) for seed in random_seeds)
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ModelError("accelerated bridge random seeds must be distinct")
    start, transition, means, covariance = _validated_parameter_arrays(
        parent_parameters
    )
    if means.shape != (n_states, np.asarray(features.values).shape[1]):
        raise ModelError("bridge parent parameters do not match requested K/features")
    parent = HMMParameters(
        start.copy(),
        transition.copy(),
        means.copy(),
        covariance.copy(),
    )
    # The warm initialization owns no RNG coordinate.  A fixed registered seed
    # is supplied only because the library accepts a random_state argument.
    initializations = (
        _HMMInitialization(0, checked_seeds[0], parent),
        *tuple(
            _HMMInitialization(index + 1, seed, None)
            for index, seed in enumerate(checked_seeds)
        ),
    )
    fit = _fit_gaussian_hmm_initializations(
        features,
        n_states,
        initializations,
        model_factory=model_factory,
        workers=int(workers),
        owner_fixed_k4_policy=owner_fixed_k4_policy,
    )
    best = (
        "parent_warm"
        if fit.best_restart_index == 0
        else f"registered_random_{fit.best_restart_index - 1}"
    )
    return BridgeAcceleratedHMMFit(
        fit,
        0,
        tuple(range(9)),
        tuple(range(1, 10)),
        best,
        checked_seeds[0],
    )


def _fit_gaussian_hmm_initializations(
    features: HMMFeatureMatrix,
    n_states: int,
    initializations: Sequence[_HMMInitialization],
    *,
    model_factory: ModelFactory,
    workers: int,
    owner_fixed_k4_policy: OwnerFixedK4Policy | None,
) -> GaussianHMMFit:
    """Common audited runner for ordinary and bridge-only initializations."""

    _require_positive_integer(n_states, "n_states")
    policy = validate_owner_fixed_k4_policy(owner_fixed_k4_policy)
    values = _validated_matrix(features.values, "HMM fitting matrix")
    _validate_feature_metadata(features, values.shape)
    if len(values) < n_states:
        raise ModelError("HMM fitting observations must be at least n_states")
    lengths = _validated_lengths_or_single(features.lengths, len(values))
    checked_initializations = tuple(initializations)
    if not checked_initializations or tuple(
        item.diagnostic_index for item in checked_initializations
    ) != tuple(range(len(checked_initializations))):
        raise ModelError("HMM initialization diagnostics must be contiguous from zero")

    jobs = tuple(
        _HMMInitializationJob(
            values=values,
            lengths=lengths,
            n_states=n_states,
            initialization=initialization,
            model_factory=model_factory,
        )
        for initialization in checked_initializations
    )
    outcomes = deterministic_process_map(
        _run_hmm_initialization,
        jobs,
        workers=workers,
        chunksize=1,
    )
    if tuple(item.diagnostic.restart_index for item in outcomes) != tuple(
        range(len(checked_initializations))
    ):
        raise ModelError("HMM restart process results are not in diagnostic order")

    diagnostics: list[RestartDiagnostic] = []
    eligible: list[
        tuple[
            int,
            int,
            float,
            HMMParameters,
            CovarianceDiagnostic,
            MarkovChainDiagnostic,
        ]
    ] = []
    for outcome in outcomes:
        diagnostic = outcome.diagnostic
        diagnostics.append(diagnostic)
        if diagnostic.valid:
            if (
                outcome.parameters is None
                or outcome.covariance is None
                or outcome.chain is None
            ):
                raise ModelError("valid restart audit omitted required evidence")
            if diagnostic.log_likelihood is None:
                raise ModelError("valid restart audit omitted likelihood")
            eligible.append(
                (
                    diagnostic.restart_index,
                    diagnostic.seed,
                    diagnostic.log_likelihood,
                    outcome.parameters,
                    outcome.covariance,
                    outcome.chain,
                )
            )

    if not eligible:
        if len(checked_initializations) == HMM_RESTARTS and all(
            item.parent_parameters is None for item in checked_initializations
        ):
            raise NoNumericallyValidHMMRestarts(
                n_states=n_states,
                restart_diagnostics=diagnostics,
            )
        raise ModelError(
            "no numerically valid tied Gaussian-HMM restart",
            details={
                "n_states": n_states,
                "restart_diagnostics": [item.as_dict() for item in diagnostics],
            },
        )
    greatest_likelihood = max(item[2] for item in eligible)
    likelihood_tolerance = LIKELIHOOD_TIE_RELATIVE_TOLERANCE * max(
        1.0, abs(greatest_likelihood)
    )
    best = min(
        (
            item
            for item in eligible
            if greatest_likelihood - item[2] <= likelihood_tolerance
        ),
        key=lambda item: item[0],
    )
    (
        best_restart,
        best_seed,
        best_likelihood,
        best_parameters,
        _preliminary_covariance,
        _preliminary_chain,
    ) = best
    try:
        canonical = canonicalize_hmm(best_parameters)
        decoded = decode_viterbi(values, canonical.parameters, lengths=lengths)
        covariance_report = tied_covariance_diagnostic(
            canonical.parameters.tied_covariance
        )
        chain_report = validate_numerical_chain(canonical.parameters.transition_matrix)
        path_report = summarize_state_path(decoded, n_states, lengths=lengths)
        posterior = forward_backward(values, canonical.parameters, lengths=lengths)
        identification = posterior_identification_report(
            posterior.posterior_probabilities,
            decoded,
            canonical.parameters,
        )
        transition_support = historical_transition_support_report(
            canonical.parameters.transition_matrix,
            path_report.transition_counts,
            posterior.expected_transition_counts,
        )
        disposition = (
            None
            if policy is None
            else owner_fixed_k4_disposition(
                n_states=n_states,
                identification=identification,
                transition=transition_support,
            )
        )
        if not effective_identification_report_passed(
            identification,
            n_states=n_states,
            disposition=disposition,
        ):
            raise _post_fit_candidate_inadmissibility(
                n_states=n_states,
                failure_code="posterior_identification",
                failure_message="HMM posterior-identification gates failed",
                rejection_reasons=identification.rejection_reasons,
                gate_details=identification.as_dict(),
                best_restart_index=best_restart,
                best_seed=best_seed,
                restart_diagnostics=diagnostics,
            )
        if not effective_transition_report_passed(
            transition_support,
            n_states=n_states,
            disposition=disposition,
        ):
            raise _post_fit_candidate_inadmissibility(
                n_states=n_states,
                failure_code="historical_transition_support",
                failure_message="historically supported transition graph failed",
                rejection_reasons=transition_support.rejection_reasons,
                gate_details={
                    "probabilities": transition_support.probabilities.tolist(),
                    "viterbi_counts": transition_support.viterbi_counts.tolist(),
                    "expected_counts": transition_support.expected_counts.tolist(),
                    "supported_cells": transition_support.supported_cells.tolist(),
                    "supported_off_diagonal_edges": (
                        transition_support.supported_off_diagonal_edges.tolist()
                    ),
                    "supported_self_loop_states": (
                        transition_support.supported_self_loop_states
                    ),
                    "decoded_entries": transition_support.decoded_entries.tolist(),
                    "decoded_exits": transition_support.decoded_exits.tolist(),
                    "strongly_connected": transition_support.strongly_connected,
                    "aperiodic": transition_support.aperiodic,
                    "passed": transition_support.passed,
                    "rejection_reasons": transition_support.rejection_reasons,
                },
                best_restart_index=best_restart,
                best_seed=best_seed,
                restart_diagnostics=diagnostics,
            )
    except HMMCandidateInadmissibility:
        raise
    except ModelError as error:
        raise ModelError(
            error.message,
            details={
                **error.details,
                "best_restart_index": best_restart,
                "best_seed": best_seed,
                "restart_diagnostics": [item.as_dict() for item in diagnostics],
            },
        ) from error
    parameter_count = gaussian_hmm_parameter_count(n_states, values.shape[1])
    return GaussianHMMFit(
        parameters=canonical.parameters,
        decoded_states=decoded,
        canonical_to_original=canonical.canonical_to_original,
        original_to_canonical=canonical.original_to_canonical,
        log_likelihood=best_likelihood,
        bic=gaussian_hmm_bic(best_likelihood, len(values), n_states, values.shape[1]),
        parameter_count=parameter_count,
        n_observations=len(values),
        n_features=values.shape[1],
        n_states=n_states,
        lengths=lengths,
        best_restart_index=best_restart,
        best_seed=best_seed,
        restart_diagnostics=tuple(diagnostics),
        covariance_diagnostic=covariance_report,
        state_path_diagnostics=path_report,
        chain_diagnostics=chain_report,
        forward_backward=posterior,
        identification=identification,
        historical_transition_support=transition_support,
        scaler_scope=features.scaler_scope,
        scaler_means=features.scaler_means.copy(),
        scaler_standard_deviations=features.scaler_standard_deviations.copy(),
        feature_names=features.feature_names,
        owner_fixed_k4_disposition=disposition,
    )


def _run_hmm_initialization(job: _HMMInitializationJob) -> _HMMInitializationOutcome:
    """Fit and audit one restart without constructing a final HMM fit."""

    initialization = job.initialization
    restart_index = initialization.diagnostic_index
    seed = _validated_seed(initialization.seed)
    model: GaussianHMMLike | None = None
    try:
        model = job.model_factory(
            n_components=job.n_states,
            covariance_type="tied",
            min_covar=DEFAULT_EIGENVALUE_FLOOR,
            startprob_prior=1.0,
            transmat_prior=1.0,
            means_weight=0.0,
            covars_prior=0.0,
            covars_weight=0.0,
            n_iter=1_000,
            tol=1e-6,
            random_state=seed,
            algorithm="viterbi",
            implementation="log",
            params="stmc",
            init_params=("stmc" if initialization.parent_parameters is None else ""),
        )
        if initialization.parent_parameters is not None:
            _apply_parent_initialization(model, initialization.parent_parameters)
        model.fit(job.values, lengths=job.lengths)
        diagnostic, parameters, covariance, chain = _audit_fitted_restart(
            model,
            restart_index=restart_index,
            seed=seed,
            values=job.values,
            lengths=job.lengths,
        )
        return _HMMInitializationOutcome(
            diagnostic,
            parameters,
            covariance,
            chain,
        )
    except (
        HMMRestartNumericalFailure,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        return _HMMInitializationOutcome(
            _exception_restart_diagnostic(
                model,
                restart_index=restart_index,
                seed=seed,
                error=error,
            ),
            None,
            None,
            None,
        )


def _apply_parent_initialization(
    model: GaussianHMMLike,
    parameters: HMMParameters,
) -> None:
    """Install validated tied-HMM parameters before a warm-start EM fit."""

    start, transition, means, covariance = _validated_parameter_arrays(parameters)
    model.startprob_ = start.copy()
    model.transmat_ = transition.copy()
    model.means_ = means.copy()
    # hmmlearn stores the canonical tied matrix in the private backing field;
    # its public getter expands it once per component.  Writing the backing
    # field avoids accidentally installing a component-expanded covariance.
    model._covars_ = covariance.copy()


def _audit_fitted_restart(
    model: GaussianHMMLike,
    *,
    restart_index: int,
    seed: int,
    values: FloatArray,
    lengths: tuple[int, ...],
) -> tuple[
    RestartDiagnostic,
    HMMParameters | None,
    CovarianceDiagnostic | None,
    MarkovChainDiagnostic | None,
]:
    monitor = getattr(model, "monitor_", None)
    converged = bool(getattr(monitor, "converged", False))
    iterations = int(getattr(monitor, "iter", 0) or 0)
    likelihood_failure: str | None = None
    try:
        likelihood = float(model.score(values, lengths=lengths))
    except (
        HMMRestartNumericalFailure,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        likelihood = math.nan
        likelihood_failure = _bounded_failure_message(error)
    history_raw = tuple(getattr(monitor, "history", ()))
    try:
        history = tuple(float(value) for value in history_raw)
    except (TypeError, ValueError, OverflowError):
        history = ()
    maximum_decrease = 0.0
    monotonic = len(history) == iterations and all(math.isfinite(v) for v in history)
    if monotonic:
        for previous, current in pairwise(history):
            decrease = previous - current
            maximum_decrease = max(maximum_decrease, decrease)
            allowed = LIKELIHOOD_DECREASE_RELATIVE_TOLERANCE * max(1.0, abs(previous))
            if decrease > allowed:
                monotonic = False

    raw_minima_value = getattr(model, "raw_covariance_min_eigenvalues_", None)
    floor_changes_value = getattr(model, "covariance_floor_relative_changes_", None)
    covariance_audit_present = isinstance(raw_minima_value, Sequence) and isinstance(
        floor_changes_value, Sequence
    )
    raw_minima: tuple[float, ...] = ()
    floor_changes: tuple[float, ...] = ()
    if covariance_audit_present:
        try:
            raw_minima = tuple(
                float(value) for value in cast(Sequence[Any], raw_minima_value)
            )
            floor_changes = tuple(
                float(value) for value in cast(Sequence[Any], floor_changes_value)
            )
        except (TypeError, ValueError, OverflowError):
            covariance_audit_present = False
    covariance_audit_present = (
        covariance_audit_present
        and len(raw_minima) == iterations
        and len(floor_changes) == iterations
        and all(math.isfinite(value) for value in raw_minima)
        and all(math.isfinite(value) for value in floor_changes)
    )
    raw_minimum = min(raw_minima) if raw_minima else None
    maximum_floor_change = max(floor_changes) if floor_changes else None

    parameters: HMMParameters | None = None
    covariance: CovarianceDiagnostic | None = None
    chain: MarkovChainDiagnostic | None = None
    parameter_failure: str | None = None
    chain_failure: str | None = None
    try:
        parameters = _extract_tied_parameters(model)
        covariance = tied_covariance_diagnostic(parameters.tied_covariance)
    except (
        HMMRestartNumericalFailure,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        parameter_failure = _bounded_failure_message(error)
        parameters = None
        covariance = None
    else:
        if not covariance.eigenvalue_floor_passed:
            parameter_failure = "final tied covariance is below the 1e-6 floor"
        elif not covariance.condition_number_passed:
            parameter_failure = "final tied covariance condition number exceeds 1e6"
    if parameters is not None:
        try:
            chain = validate_numerical_chain(parameters.transition_matrix)
        except HMMRestartNumericalFailure as error:
            chain_failure = _bounded_failure_message(error)
            chain = None

    failure_type: str | None = None
    failure_message: str | None = None
    likelihood_value: float | None
    if likelihood_failure is not None:
        likelihood_value = None
        failure_type = "LikelihoodEvaluationFailure"
        failure_message = likelihood_failure
    elif not math.isfinite(likelihood):
        likelihood_value = None
        failure_type = "NonFiniteLikelihood"
        failure_message = "score returned a non-finite value"
    else:
        likelihood_value = likelihood
    if failure_type is None and iterations < 2:
        failure_type = "InsufficientIterations"
        failure_message = "fewer than two EM iterations occurred"
    if failure_type is None and not converged:
        failure_type = "NonConvergence"
        failure_message = "hmmlearn convergence monitor did not converge"
    if failure_type is None and not monotonic:
        failure_type = "LikelihoodDecrease"
        failure_message = "EM likelihood history violated the monotonicity tolerance"
    if failure_type is None and not covariance_audit_present:
        failure_type = "CovarianceAuditMissing"
        failure_message = "per-M-step tied covariance audit is missing or incomplete"
    if (
        failure_type is None
        and raw_minimum is not None
        and (raw_minimum < RAW_COVARIANCE_EIGENVALUE_TOLERANCE)
    ):
        failure_type = "RawCovarianceEigenvalue"
        failure_message = "raw tied covariance eigenvalue is below -1e-10"
    if (
        failure_type is None
        and maximum_floor_change is not None
        and (maximum_floor_change > MAXIMUM_COVARIANCE_FLOOR_RELATIVE_CHANGE)
    ):
        failure_type = "CovarianceFloorPerturbation"
        failure_message = "tied covariance flooring changed an M-step by more than 1%"
    if failure_type is None and parameter_failure is not None:
        failure_type = "ParameterAuditFailure"
        failure_message = parameter_failure
    if failure_type is None and chain_failure is not None:
        failure_type = "PreliminaryMarkovFailure"
        failure_message = chain_failure

    diagnostic = RestartDiagnostic(
        restart_index=restart_index,
        seed=seed,
        converged=converged,
        iterations=iterations,
        log_likelihood=likelihood_value,
        valid=failure_type is None,
        raw_minimum_eigenvalue=raw_minimum,
        maximum_floor_relative_change=maximum_floor_change,
        covariance_condition_number=(
            None if covariance is None else covariance.condition_number
        ),
        maximum_likelihood_decrease=(maximum_decrease if history else None),
        preliminary_chain_passed=chain is not None,
        failure_type=failure_type,
        failure_message=failure_message,
    )
    return diagnostic, parameters, covariance, chain


def _exception_restart_diagnostic(
    model: GaussianHMMLike | None,
    *,
    restart_index: int,
    seed: int,
    error: Exception,
) -> RestartDiagnostic:
    monitor = getattr(model, "monitor_", None)
    iterations = int(getattr(monitor, "iter", 0) or 0)
    converged = bool(getattr(monitor, "converged", False))
    return RestartDiagnostic(
        restart_index=restart_index,
        seed=seed,
        converged=converged,
        iterations=iterations,
        log_likelihood=None,
        valid=False,
        raw_minimum_eigenvalue=None,
        maximum_floor_relative_change=None,
        covariance_condition_number=None,
        maximum_likelihood_decrease=None,
        preliminary_chain_passed=False,
        failure_type="NumericalRestartFailure",
        failure_message=(f"{type(error).__name__}: {_bounded_failure_message(error)}"),
    )


def _extract_tied_parameters(model: GaussianHMMLike) -> HMMParameters:
    means = np.asarray(model.means_, dtype=np.float64)
    raw_covariance = getattr(model, "_covars_", None)
    if raw_covariance is None:
        raw_covariance = model.covars_
    covariance = np.asarray(raw_covariance, dtype=np.float64)
    if covariance.ndim == 3:
        if covariance.shape[0] == 0 or not np.allclose(
            covariance,
            covariance[0][None, :, :],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ModelError("hmmlearn tied covariance expansion is inconsistent")
        covariance = covariance[0]
    parameters = HMMParameters(
        start_probabilities=np.asarray(model.startprob_, dtype=np.float64),
        transition_matrix=np.asarray(model.transmat_, dtype=np.float64),
        means=means,
        tied_covariance=covariance,
    )
    start, transition, checked_means, checked_covariance = _validated_parameter_arrays(
        parameters
    )
    return HMMParameters(
        start_probabilities=start,
        transition_matrix=transition,
        means=checked_means.copy(),
        tied_covariance=checked_covariance.copy(),
    )


def _validated_parameter_arrays(
    parameters: HMMParameters,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    start = np.asarray(parameters.start_probabilities, dtype=np.float64)
    transition = np.asarray(parameters.transition_matrix, dtype=np.float64)
    means = np.asarray(parameters.means, dtype=np.float64)
    tied_covariance = np.asarray(parameters.tied_covariance, dtype=np.float64)
    if start.ndim != 1 or start.size == 0:
        raise ModelError("HMM start probabilities must be one-dimensional")
    n_states = len(start)
    if transition.shape != (n_states, n_states):
        raise ModelError("HMM transition matrix shape is inconsistent")
    if means.ndim != 2 or means.shape[0] != n_states or means.shape[1] == 0:
        raise ModelError("HMM means shape is inconsistent")
    if tied_covariance.shape != (means.shape[1], means.shape[1]):
        raise ModelError("HMM tied covariance shape is inconsistent")
    for name, array in (
        ("start probabilities", start),
        ("transition matrix", transition),
        ("means", means),
        ("tied covariance", tied_covariance),
    ):
        if not np.isfinite(array).all():
            raise ModelError(f"HMM {name} contain non-finite values")
    start = _validate_probability_vector(start, "HMM start probabilities")
    transition = np.vstack(
        [_validate_probability_vector(row, "HMM transition row") for row in transition]
    )
    tied_covariance_diagnostic(tied_covariance, eigenvalue_floor=0.0)
    return start, transition, means, tied_covariance


def _validate_feature_metadata(
    features: HMMFeatureMatrix, shape: tuple[int, int]
) -> None:
    n_observations, n_features = shape
    if features.scaler_scope not in {"design_training", "production_full"}:
        raise ModelError("HMM feature scaler scope is invalid")
    names_match = len(features.feature_names) == n_features
    names_are_unique = len(set(features.feature_names)) == n_features
    if not names_match or not names_are_unique:
        raise ModelError("HMM feature names do not match matrix columns")
    if len(features.row_labels) != n_observations:
        raise ModelError("HMM row labels do not match matrix rows")
    means = np.asarray(features.scaler_means, dtype=np.float64)
    scales = np.asarray(features.scaler_standard_deviations, dtype=np.float64)
    if means.shape != (n_features,) or scales.shape != (n_features,):
        raise ModelError("HMM scaler metadata does not match matrix columns")
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ModelError("HMM scaler metadata contains non-finite values")
    if bool((scales <= 0.0).any()):
        raise ModelError("HMM scaler standard deviations must be positive")


def _validate_probability_vector(values: FloatArray, label: str) -> FloatArray:
    if bool((values < -NEGATIVE_PROBABILITY_TOLERANCE).any()) or bool(
        (values > 1.0 + NEGATIVE_PROBABILITY_TOLERANCE).any()
    ):
        raise ModelError(f"{label} must be within numerical probability bounds")
    error = abs(float(values.sum()) - 1.0)
    if error > PROBABILITY_ATOL:
        raise ModelError(
            f"{label} must sum to one",
            details={"absolute_sum_error": error},
        )
    sanitized = np.maximum(values, 0.0).astype(np.float64, copy=True)
    sanitized /= sanitized.sum()
    return sanitized


def _validated_states(states: Sequence[int] | IntArray, n_states: int) -> IntArray:
    raw = np.asarray(states)
    if raw.ndim != 1 or raw.size == 0:
        raise ModelError("state path must be non-empty and one-dimensional")
    if raw.dtype.kind not in {"i", "u"}:
        raise ModelError("state path must have integer dtype")
    path = raw.astype(np.int64, copy=True)
    if bool((path < 0).any()) or bool((path >= n_states).any()):
        raise ModelError("state path contains an out-of-range label")
    return path


def _validated_matrix(
    values: Sequence[Sequence[float]] | FloatArray, label: str
) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ModelError(f"{label} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ModelError(f"{label} contains non-finite values")
    return matrix


def _validated_lengths_or_single(
    lengths: Sequence[int] | None, n_observations: int
) -> tuple[int, ...]:
    if lengths is None:
        return (n_observations,)
    result = tuple(lengths)
    _validate_lengths(result, n_observations)
    return tuple(int(value) for value in result)


def _validate_lengths(lengths: Sequence[int], n_observations: int) -> None:
    if not lengths:
        raise ModelError("HMM sequence lengths must be non-empty")
    total = 0
    for value in lengths:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ModelError("HMM sequence lengths must be positive integers")
        if int(value) <= 0:
            raise ModelError("HMM sequence lengths must be positive integers")
        total += int(value)
    if total != n_observations:
        raise ModelError(
            "HMM sequence lengths do not sum to observation count",
            details={"length_sum": total, "n_observations": n_observations},
        )


def _gaussian_log_emissions(
    values: FloatArray, means: FloatArray, tied_covariance: FloatArray
) -> FloatArray:
    result = np.empty((len(values), len(means)), dtype=np.float64)
    constant = means.shape[1] * math.log(2.0 * math.pi)
    sign, log_determinant = np.linalg.slogdet(tied_covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ModelError("tied covariance is not positive definite")
    for state, mean in enumerate(means):
        difference = values - mean
        try:
            solved = np.linalg.solve(tied_covariance, difference.T).T
        except np.linalg.LinAlgError as error:
            raise ModelError(
                "tied covariance solve failed", details={"state": state}
            ) from error
        quadratic = np.einsum("ij,ij->i", difference, solved)
        result[:, state] = -0.5 * (constant + log_determinant + quadratic)
    if not np.isfinite(result).all():
        raise ModelError("Viterbi emission likelihoods are non-finite")
    return result


def _safe_log_probabilities(values: FloatArray) -> FloatArray:
    result = np.full(values.shape, -math.inf, dtype=np.float64)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


def _validated_initial_state_probabilities(
    values: Sequence[Sequence[float]] | FloatArray | None,
    *,
    n_segments: int,
    n_states: int,
    default: FloatArray,
) -> FloatArray:
    if values is None:
        return np.tile(default, (n_segments, 1))
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("initial state probabilities must be numeric") from error
    if array.ndim == 1 and n_segments == 1:
        array = array[None, :]
    if array.shape != (n_segments, n_states):
        raise ModelError(
            "initial state probabilities have incompatible shape",
            details={"required": [n_segments, n_states], "actual": list(array.shape)},
        )
    return np.vstack(
        [
            _validate_probability_vector(row, "initial state probabilities")
            for row in array
        ]
    )


def _optional_probability_matrix(
    values: Sequence[Sequence[float]] | FloatArray | None,
    shape: tuple[int, int],
    label: str,
) -> FloatArray | None:
    if values is None:
        return None
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if (
        matrix.shape != shape
        or not np.isfinite(matrix).all()
        or bool(((matrix < 0.0) | (matrix > 1.0)).any())
    ):
        raise ModelError(f"{label} must be a matching finite probability matrix")
    return _readonly(matrix.copy())


def _optional_transition_quantiles(
    values: FloatArray | None,
    shape: tuple[int, int],
) -> FloatArray | None:
    if values is None:
        return None
    try:
        quantiles = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("bootstrap transition quantiles must be numeric") from error
    expected_shape = (*shape, 3)
    if quantiles.shape != expected_shape or not np.isfinite(quantiles).all():
        raise ModelError("bootstrap transition quantiles must have shape (K, K, 3)")
    if bool(((quantiles < 0.0) | (quantiles > 1.0)).any()) or bool(
        (np.diff(quantiles, axis=2) < 0.0).any()
    ):
        raise ModelError("bootstrap transition quantiles are invalid or unordered")
    return _readonly(quantiles.copy())


def _readonly(array: NDArray[ArrayScalar]) -> NDArray[ArrayScalar]:
    array.setflags(write=False)
    return array


def _frozen_candidate_gate_details(
    values: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    """Recursively freeze one small post-fit scientific gate document."""

    return tuple(
        (key, _frozen_candidate_gate_value(value))
        for key, value in sorted(values.items())
    )


def _frozen_candidate_gate_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _frozen_candidate_gate_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list | tuple):
        return tuple(_frozen_candidate_gate_value(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ModelError("candidate inadmissibility evidence has an unsupported value")


def _valid_candidate_inadmissibility_evidence(value: object) -> bool:
    if not isinstance(value, HMMCandidateInadmissibilityEvidence):
        return False
    if (
        isinstance(value.n_states, bool)
        or not isinstance(value.n_states, int)
        or value.n_states <= 0
        or value.failure_code
        not in {"posterior_identification", "historical_transition_support"}
        or not isinstance(value.failure_message, str)
        or not value.failure_message
        or not value.rejection_reasons
        or any(
            not isinstance(reason, str) or not reason
            for reason in value.rejection_reasons
        )
        or not value.restart_diagnostics
        or any(
            not isinstance(item, RestartDiagnostic) or item.restart_index != index
            for index, item in enumerate(value.restart_diagnostics)
        )
        or isinstance(value.best_restart_index, bool)
        or not isinstance(value.best_restart_index, int)
        or not 0 <= value.best_restart_index < len(value.restart_diagnostics)
        or isinstance(value.best_seed, bool)
        or not isinstance(value.best_seed, int)
        or not 0 <= value.best_seed <= np.iinfo(np.uint32).max
    ):
        return False
    best = value.restart_diagnostics[value.best_restart_index]
    if not best.valid or best.seed != value.best_seed:
        return False
    if (
        not isinstance(value.gate_details, tuple)
        or not value.gate_details
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not _is_frozen_candidate_gate_value(item[1])
            for item in value.gate_details
        )
        or tuple(key for key, _ in value.gate_details)
        != tuple(sorted(key for key, _ in value.gate_details))
    ):
        return False
    details = dict(value.gate_details)
    expected_keys = (
        {
            "assigned_state_mean_posteriors",
            "mean_maximum_posterior",
            "minimum_pairwise_mahalanobis_distance",
            "minimum_required_effective_mass",
            "normalized_posterior_entropy",
            "passed",
            "posterior_effective_masses",
            "rejection_reasons",
        }
        if value.failure_code == "posterior_identification"
        else {
            "aperiodic",
            "decoded_entries",
            "decoded_exits",
            "expected_counts",
            "passed",
            "probabilities",
            "rejection_reasons",
            "strongly_connected",
            "supported_cells",
            "supported_off_diagonal_edges",
            "supported_self_loop_states",
            "viterbi_counts",
        }
    )
    return bool(
        set(details) == expected_keys
        and details["passed"] is False
        and details["rejection_reasons"] == value.rejection_reasons
    )


def _is_frozen_candidate_gate_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, tuple):
        return all(_is_frozen_candidate_gate_value(item) for item in value)
    return False


def _is_irreducible(adjacency: BoolArray) -> bool:
    for start in range(len(adjacency)):
        visited = {start}
        stack = [start]
        while stack:
            state = stack.pop()
            for target in np.flatnonzero(adjacency[state]):
                target_int = int(target)
                if target_int not in visited:
                    visited.add(target_int)
                    stack.append(target_int)
        if len(visited) != len(adjacency):
            return False
    return True


def _chain_period(adjacency: BoolArray) -> int:
    depth = np.full(len(adjacency), -1, dtype=np.int64)
    depth[0] = 0
    stack = [0]
    while stack:
        state = stack.pop()
        for target in np.flatnonzero(adjacency[state]):
            if depth[target] < 0:
                depth[target] = depth[state] + 1
                stack.append(int(target))
    period = 0
    for state in range(len(adjacency)):
        for target in np.flatnonzero(adjacency[state]):
            period = math.gcd(period, int(depth[state] + 1 - depth[target]))
    return abs(period)


def _stationary_distribution(matrix: FloatArray) -> FloatArray:
    n_states = len(matrix)
    system = np.vstack([matrix.T - np.eye(n_states), np.ones(n_states)])
    target = np.concatenate([np.zeros(n_states), np.ones(1)])
    try:
        solution, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    except np.linalg.LinAlgError as error:
        raise HMMRestartNumericalFailure(
            "stationary distribution solve failed"
        ) from error
    if rank < n_states:
        raise HMMRestartNumericalFailure("stationary distribution is non-unique")
    return np.asarray(solution, dtype=np.float64)


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise ModelError("HMM restart seeds must be uint32 integers")
    result = int(seed)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ModelError("HMM restart seeds must be uint32 integers")
    return result


def _require_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ModelError(f"{label} must be a positive integer")
    if int(value) <= 0:
        raise ModelError(f"{label} must be a positive integer")


def _bounded_failure_message(error: Exception, limit: int = 500) -> str:
    text = " ".join(str(error).split())
    return text[:limit] if text else "<no message>"
