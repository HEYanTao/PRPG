"""Exact statistical primitives for the four-cell Phase-3 adequacy gate.

Simulation/orchestration lives elsewhere.  This module freezes endpoint order,
pooled latent-chain estimators, predictive-mixture diagnostics, mixed-direction
max statistics, and the final four-cell Holm decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import ndtr

from prpg.errors import ModelError
from prpg.model.inference import HolmResult, holm_correction

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
StateArray = NDArray[np.integer[Any]]

CONSTANT_TOLERANCE = 1e-12
PIT_CLIP = 2.0**-53
LATENT_CHAIN_COUNT = 10_000
LATENT_MEASURED_MONTHS = 600
LATENT_EXTENSION_MONTHS = 2_000
LATENT_SURVIVAL_MONTHS = 12
LATENT_BOOTSTRAP_REPLICATES = 10_000
LATENT_BOOTSTRAP_RESAMPLE_SIZE = 10_000
LATENT_GENERATION_CHUNK_SIZE = 1_024
LATENT_BOOTSTRAP_CHUNK_SIZE = 32


class EndpointDirection(StrEnum):
    """Direction used by a max-stat component."""

    TWO_SIDED = "two_sided"
    UPPER_TAIL = "upper_tail"


class AdequacyEndpointSupportFailure(ModelError):
    """A generated replicate lacks data-dependent support for an endpoint.

    This narrow subtype is reserved for otherwise well-formed endpoint inputs
    whose realized values make a registered statistic undefined.  Geometry,
    dtype, non-finite arithmetic, and other contract failures remain ordinary
    :class:`ModelError` instances and must not be counted as scientific
    replicate non-passes.
    """


@dataclass(frozen=True, slots=True)
class EndpointVector:
    """One canonically ordered named endpoint vector."""

    labels: tuple[str, ...]
    values: FloatArray


@dataclass(frozen=True, slots=True)
class MaxStatisticCell:
    """One empirical mixed-direction max-stat cell."""

    observed_statistic: float
    replicate_statistics: FloatArray
    p_value: float
    center: FloatArray
    scale: FloatArray
    active_components: BoolArray
    successful_replicates: int


@dataclass(frozen=True, slots=True)
class PredictiveMixtureDiagnostics:
    """One-step mixture residual and PIT evidence in fixed feature order."""

    predictive_means: FloatArray
    predictive_residuals: FloatArray
    predictive_covariance_error_upper: FloatArray
    pits: FloatArray
    pit_cramer_von_mises: FloatArray


@dataclass(frozen=True, slots=True)
class ACFDiagnostics:
    """Available-pair cosine ACFs and pair counts."""

    lags: tuple[int, ...]
    raw: FloatArray
    squared: FloatArray
    raw_pair_counts: IntArray
    squared_pair_counts: IntArray


@dataclass(frozen=True, slots=True)
class LatentChainSimulation:
    """Audit evidence for one exact sentinel-prefixed latent-chain sample.

    ``chains`` has columns ``S[-1], S[0], ..., S[M+E-1]``.  The compact
    unsigned state dtype keeps the binding 10,000 by 2,601 default geometry
    practical without changing any state value.
    """

    transition_matrix: FloatArray
    stationary_distribution: FloatArray
    chains: StateArray
    chain_count: int
    measured_months: int
    extension_months: int
    uniforms_per_chain: int


@dataclass(frozen=True, slots=True)
class LatentClusterBootstrapEvidence:
    """Complete evidence for one exact latent-cell cluster-bootstrap test."""

    target: EndpointVector
    pooled_estimate: EndpointVector
    bootstrap_estimates: FloatArray
    standard_errors: FloatArray
    active_components: BoolArray
    bootstrap_max_statistics: FloatArray
    observed_max_statistic: float
    p_value: float
    chain_count: int
    resample_size: int
    bootstrap_replicates: int
    measured_months: int
    extension_months: int
    survival_months: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class _LatentClusterSufficientStatistics:
    """Per-chain linear counts and sparse complete-dwell histograms."""

    occupancy_and_transitions: IntArray
    dwell_histogram: csr_matrix
    n_states: int
    total_months: int


def generate_latent_chains(
    transition_matrix: ArrayLike,
    *,
    chain_count: int = LATENT_CHAIN_COUNT,
    measured_months: int = LATENT_MEASURED_MONTHS,
    extension_months: int = LATENT_EXTENSION_MONTHS,
    uniforms: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    chunk_size: int = LATENT_GENERATION_CHUNK_SIZE,
) -> LatentChainSimulation:
    """Generate the exact stationary-sentinel latent-chain geometry.

    Exactly one randomness source is mandatory.  Passing ``uniforms`` makes
    the function mathematically pure; passing a caller-owned ``Generator``
    consumes the same row-major uniform trace in bounded chunks.  Every row
    consumes one sentinel uniform followed by one transition uniform for each
    measured or extension month.
    """

    transition = _transition_matrix(transition_matrix).astype(np.float64, copy=True)
    _positive_integer(chain_count, "latent chain count")
    _positive_integer(measured_months, "measured_months")
    _positive_integer(extension_months, "extension_months")
    _positive_integer(chunk_size, "latent generation chunk_size")
    if (uniforms is None) == (rng is None):
        raise ModelError("latent generation requires exactly one of uniforms or rng")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ModelError("latent generation rng must be a numpy Generator")

    total_months = measured_months + extension_months
    draw_count = 1 + total_months
    uniform_values: FloatArray | None = None
    if uniforms is not None:
        uniform_values = _uniform_matrix(
            uniforms,
            expected_shape=(chain_count, draw_count),
            label="latent generation uniforms",
        )

    stationary = _stationary_distribution(transition)
    stationary_cdf = np.cumsum(stationary)
    transition_cdf = np.cumsum(transition, axis=1)
    # Row sums are accepted to 1e-12.  Closing the final interval at exactly
    # one prevents that tolerance from creating an impossible state index.
    stationary_cdf[-1] = 1.0
    transition_cdf[:, -1] = 1.0
    n_states = len(transition)
    if n_states <= np.iinfo(np.uint8).max + 1:
        state_dtype: np.dtype[Any] = np.dtype(np.uint8)
    elif n_states <= np.iinfo(np.uint16).max + 1:
        state_dtype = np.dtype(np.uint16)
    elif n_states <= np.iinfo(np.uint32).max + 1:
        state_dtype = np.dtype(np.uint32)
    else:  # pragma: no cover - an array this large cannot be materialized.
        raise ModelError("transition matrix has too many states")
    chains = np.empty((chain_count, draw_count), dtype=state_dtype)

    for start in range(0, chain_count, chunk_size):
        stop = min(start + chunk_size, chain_count)
        if uniform_values is not None:
            draws = uniform_values[start:stop]
        else:
            if rng is None:  # pragma: no cover - narrowed by validation above.
                raise AssertionError("validated latent RNG is unavailable")
            draws = _uniform_matrix(
                rng.random((stop - start, draw_count)),
                expected_shape=(stop - start, draw_count),
                label="latent generation RNG output",
            )
        output = chains[start:stop]
        output[:, 0] = np.searchsorted(stationary_cdf, draws[:, 0], side="right")
        for month in range(total_months):
            previous = output[:, month].astype(np.intp, copy=False)
            row_cdf = transition_cdf[previous]
            output[:, month + 1] = np.count_nonzero(
                draws[:, month + 1, None] >= row_cdf, axis=1
            )
        if bool((output >= n_states).any()):
            raise ModelError("latent generation produced an out-of-range state")

    stationary = stationary.astype(np.float64, copy=True)
    _readonly(transition, stationary, chains)
    return LatentChainSimulation(
        transition_matrix=transition,
        stationary_distribution=stationary,
        chains=chains,
        chain_count=chain_count,
        measured_months=measured_months,
        extension_months=extension_months,
        uniforms_per_chain=draw_count,
    )


def latent_cluster_bootstrap(
    simulation: LatentChainSimulation,
    *,
    expected_chain_count: int = LATENT_CHAIN_COUNT,
    bootstrap_replicates: int = LATENT_BOOTSTRAP_REPLICATES,
    resample_size: int = LATENT_BOOTSTRAP_RESAMPLE_SIZE,
    survival_months: int = LATENT_SURVIVAL_MONTHS,
    resample_indices: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    chunk_size: int = LATENT_BOOTSTRAP_CHUNK_SIZE,
    constant_tolerance: float = CONSTANT_TOLERANCE,
) -> LatentClusterBootstrapEvidence:
    """Run the exact whole-chain latent-cell cluster bootstrap.

    The binding defaults require 10,000 input chains, 10,000 draws per
    replicate, and 10,000 replicates.  Smaller audit fixtures must override
    all three exact counts explicitly.  Integer sufficient statistics and a
    sparse per-chain dwell histogram make results invariant to ``chunk_size``
    while bounding the working set; only the final endpoint matrix is retained.
    """

    checked = _validated_latent_simulation(simulation)
    _positive_integer(expected_chain_count, "expected latent chain count")
    _positive_integer(bootstrap_replicates, "latent bootstrap replicate count")
    _positive_integer(resample_size, "latent bootstrap resample size")
    _positive_integer(survival_months, "survival_months")
    _positive_integer(chunk_size, "latent bootstrap chunk_size")
    if bootstrap_replicates < 2:
        raise ModelError("latent bootstrap requires at least two replicates")
    if checked.chain_count != expected_chain_count:
        raise ModelError(
            "latent chain count does not match the exact bootstrap contract",
            details={
                "actual": checked.chain_count,
                "expected": expected_chain_count,
            },
        )
    if resample_size != expected_chain_count:
        raise ModelError("latent bootstrap must resample exactly one full chain-count")
    total_months = checked.measured_months + checked.extension_months
    if survival_months > total_months:
        raise ModelError("survival_months exceeds the simulated chain horizon")
    if not np.isfinite(constant_tolerance) or constant_tolerance < 0.0:
        raise ModelError("constant tolerance must be finite and non-negative")
    if (resample_indices is None) == (rng is None):
        raise ModelError(
            "latent bootstrap requires exactly one of resample_indices or rng"
        )
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ModelError("latent bootstrap rng must be a numpy Generator")

    supplied_indices: NDArray[np.generic] | None = None
    if resample_indices is not None:
        supplied_indices = _bootstrap_index_matrix(
            resample_indices,
            expected_shape=(bootstrap_replicates, resample_size),
            chain_count=expected_chain_count,
            label="latent bootstrap resample indices",
        )

    n_states = len(checked.transition_matrix)
    statistics = _latent_cluster_sufficient_statistics(
        checked.chains,
        n_states=n_states,
        measured_months=checked.measured_months,
        extension_months=checked.extension_months,
    )
    pooled_linear = statistics.occupancy_and_transitions.sum(axis=0, dtype=np.int64)[
        None, :
    ]
    pooled_histogram = np.asarray(
        statistics.dwell_histogram.sum(axis=0), dtype=np.int64
    )
    pooled_values = _latent_estimates_from_counts(
        pooled_linear,
        pooled_histogram,
        n_states=n_states,
        measured_months=checked.measured_months,
        resample_size=expected_chain_count,
        survival_months=survival_months,
        total_months=total_months,
    )[0]
    labels = _latent_labels(n_states, survival_months)
    pooled_values = pooled_values.astype(np.float64, copy=True)
    _readonly(pooled_values)
    pooled_estimate = EndpointVector(labels, pooled_values)
    target = latent_analytic_target(
        checked.transition_matrix, survival_months=survival_months
    )

    endpoint_count = len(labels)
    estimates = np.empty((bootstrap_replicates, endpoint_count), dtype=np.float64)
    for start in range(0, bootstrap_replicates, chunk_size):
        stop = min(start + chunk_size, bootstrap_replicates)
        if supplied_indices is not None:
            indices = supplied_indices[start:stop]
        else:
            if rng is None:  # pragma: no cover - narrowed by validation above.
                raise AssertionError("validated bootstrap RNG is unavailable")
            indices = _bootstrap_index_matrix(
                rng.integers(
                    0,
                    expected_chain_count,
                    size=(stop - start, resample_size),
                    dtype=np.int64,
                ),
                expected_shape=(stop - start, resample_size),
                chain_count=expected_chain_count,
                label="latent bootstrap RNG output",
            )
        weights = _cluster_resample_weights(
            indices,
            chain_count=expected_chain_count,
            resample_size=resample_size,
        )
        linear_counts = weights @ statistics.occupancy_and_transitions
        dwell_counts = np.asarray(weights @ statistics.dwell_histogram, dtype=np.int64)
        estimates[start:stop] = _latent_estimates_from_counts(
            linear_counts,
            dwell_counts,
            n_states=n_states,
            measured_months=checked.measured_months,
            resample_size=resample_size,
            survival_months=survival_months,
            total_months=total_months,
        )

    if not np.isfinite(estimates).all():
        raise ModelError("latent bootstrap endpoint matrix is non-finite")
    standard_errors = estimates.std(axis=0, ddof=1)
    active = standard_errors > constant_tolerance
    for component in np.flatnonzero(~active):
        bootstrap_spread = float(
            np.max(np.abs(estimates[:, component] - pooled_values[component]))
        )
        if bootstrap_spread > constant_tolerance:
            raise ModelError(
                "constant latent component has excessive bootstrap spread",
                details={
                    "component": int(component),
                    "spread": bootstrap_spread,
                },
            )
        target_error = abs(float(pooled_values[component] - target.values[component]))
        if target_error > constant_tolerance:
            raise ModelError(
                "pooled latent estimate fails a constant target component",
                details={
                    "component": int(component),
                    "error": target_error,
                },
            )

    if bool(active.any()):
        centered = (
            estimates[:, active] - pooled_values[None, active]
        ) / standard_errors[None, active]
        bootstrap_max = np.max(np.abs(centered), axis=1)
        observed_max = float(
            np.max(
                np.abs(
                    (pooled_values[active] - target.values[active])
                    / standard_errors[active]
                )
            )
        )
    else:
        bootstrap_max = np.zeros(bootstrap_replicates, dtype=np.float64)
        observed_max = 0.0
    if not np.isfinite(bootstrap_max).all() or not np.isfinite(observed_max):
        raise ModelError("latent bootstrap max statistic is non-finite")
    p_value = float(
        (1 + np.count_nonzero(bootstrap_max >= observed_max))
        / (bootstrap_replicates + 1)
    )

    standard_errors = standard_errors.astype(np.float64, copy=True)
    active = active.astype(np.bool_, copy=True)
    bootstrap_max = bootstrap_max.astype(np.float64, copy=True)
    _readonly(estimates, standard_errors, active, bootstrap_max)
    return LatentClusterBootstrapEvidence(
        target=target,
        pooled_estimate=pooled_estimate,
        bootstrap_estimates=estimates,
        standard_errors=standard_errors,
        active_components=active,
        bootstrap_max_statistics=bootstrap_max,
        observed_max_statistic=observed_max,
        p_value=p_value,
        chain_count=expected_chain_count,
        resample_size=resample_size,
        bootstrap_replicates=bootstrap_replicates,
        measured_months=checked.measured_months,
        extension_months=checked.extension_months,
        survival_months=survival_months,
        chunk_size=chunk_size,
    )


def latent_analytic_target(
    transition_matrix: ArrayLike, *, survival_months: int = 12
) -> EndpointVector:
    """Create the exact analytic latent endpoint vector in canonical order."""

    transition = _transition_matrix(transition_matrix)
    _positive_integer(survival_months, "survival_months")
    stationary = _stationary_distribution(transition)
    labels = _latent_labels(len(transition), survival_months)
    values: list[float] = []
    values.extend(float(value) for value in stationary)
    for source in range(len(transition)):
        for target in range(len(transition)):
            if source != target:
                values.append(float(transition[source, target]))
    values.extend(
        float(stationary[state] * (1.0 - transition[state, state]))
        for state in range(len(transition))
    )
    values.extend(
        float(1.0 - transition[state, state]) for state in range(len(transition))
    )
    for state in range(len(transition)):
        self_probability = float(transition[state, state])
        if self_probability >= 1.0:
            raise ModelError("latent dwell target requires P_ss below one")
        median = (
            1
            if self_probability == 0.0
            else math.ceil(math.log(0.5) / math.log(self_probability))
        )
        values.append(float(median))
    for state in range(len(transition)):
        self_probability = float(transition[state, state])
        values.extend(
            self_probability ** (month - 1) for month in range(1, survival_months + 1)
        )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(labels),):
        raise AssertionError("latent target endpoint order is inconsistent")
    _readonly(result)
    return EndpointVector(labels, result)


def latent_pooled_estimate(
    chains: ArrayLike,
    n_states: int,
    *,
    measured_months: int = 600,
    extension_months: int = 2_000,
    survival_months: int = 12,
) -> EndpointVector:
    """Pool exact latent sufficient statistics across sentinel-prefixed chains.

    Each row is ``S[-1], S[0], ..., S[measured+extension-1]``.  Only the
    measured window contributes occupancy/transitions/episode starts; the
    extension is used solely to complete spells that begin in that window.
    """

    _positive_integer(n_states, "n_states")
    _positive_integer(measured_months, "measured_months")
    _positive_integer(extension_months, "extension_months")
    _positive_integer(survival_months, "survival_months")
    values = _state_chains(
        chains,
        n_states=n_states,
        expected_columns=1 + measured_months + extension_months,
    )
    occupancy = np.zeros(n_states, dtype=np.int64)
    transitions = np.zeros((n_states, n_states), dtype=np.int64)
    departures = np.zeros(n_states, dtype=np.int64)
    dwell: list[list[int]] = [[] for _ in range(n_states)]
    for chain_index, chain in enumerate(values):
        measured = chain[1 : measured_months + 1]
        occupancy += np.bincount(measured, minlength=n_states)
        for time in range(measured_months):
            previous = int(chain[time])
            current = int(chain[time + 1])
            transitions[previous, current] += 1
            departures[previous] += 1
            if previous == current:
                continue
            end = time + 2
            while end < len(chain) and int(chain[end]) == current:
                end += 1
            if end == len(chain):
                raise ModelError(
                    "latent spell remains incomplete after the fixed extension",
                    details={"chain": chain_index, "measured_time": time},
                )
            dwell[current].append(end - (time + 1))
    if bool((departures == 0).any()):
        raise ModelError("latent transition denominator is zero")
    if any(len(items) == 0 for items in dwell):
        raise ModelError("latent episode count is zero")
    denominator = len(values) * measured_months
    result_values: list[float] = []
    result_values.extend(float(value / denominator) for value in occupancy)
    for source in range(n_states):
        for target in range(n_states):
            if source != target:
                result_values.append(
                    float(transitions[source, target] / departures[source])
                )
    result_values.extend(float(len(items) / denominator) for items in dwell)
    result_values.extend(
        float(np.count_nonzero(np.asarray(items) == 1) / len(items)) for items in dwell
    )
    result_values.extend(
        float(np.quantile(items, 0.5, method="higher")) for items in dwell
    )
    for items in dwell:
        array = np.asarray(items, dtype=np.int64)
        result_values.extend(
            float(np.count_nonzero(array >= month) / len(array))
            for month in range(1, survival_months + 1)
        )
    labels = _latent_labels(n_states, survival_months)
    result = np.asarray(result_values, dtype=np.float64)
    if result.shape != (len(labels),):
        raise AssertionError("latent estimate endpoint order is inconsistent")
    _readonly(result)
    return EndpointVector(labels, result)


def mixed_max_statistic_cell(
    observed: ArrayLike,
    replicates: ArrayLike,
    directions: tuple[EndpointDirection | str, ...],
    *,
    minimum_successes: int = 1,
    constant_tolerance: float = CONSTANT_TOLERANCE,
) -> MaxStatisticCell:
    """Calculate the approved mixed two-sided/upper-tail empirical cell."""

    x = _vector(observed, "observed adequacy vector")
    reference = np.asarray(replicates, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != x.size:
        raise ModelError("adequacy replicate matrix shape does not match observed")
    if not np.isfinite(reference).all():
        raise ModelError("adequacy replicate matrix contains non-finite values")
    _positive_integer(minimum_successes, "minimum_successes")
    if len(reference) < minimum_successes:
        raise ModelError(
            "insufficient successful adequacy replicates",
            details={"actual": len(reference), "required": minimum_successes},
        )
    if len(reference) < 2:
        raise ModelError("at least two adequacy replicates are required")
    if len(directions) != x.size:
        raise ModelError("adequacy endpoint directions do not match components")
    checked_directions = tuple(EndpointDirection(value) for value in directions)
    if not np.isfinite(constant_tolerance) or constant_tolerance < 0.0:
        raise ModelError("constant tolerance must be finite and non-negative")
    center = reference.mean(axis=0)
    scale = reference.std(axis=0, ddof=1)
    active = scale > constant_tolerance
    for component in np.flatnonzero(~active):
        spread = float(np.max(np.abs(reference[:, component] - center[component])))
        if spread > constant_tolerance:
            raise ModelError(
                "constant adequacy component has excessive replicate spread",
                details={"component": int(component), "spread": spread},
            )
        if checked_directions[component] is EndpointDirection.TWO_SIDED:
            passes = abs(float(x[component] - center[component])) <= constant_tolerance
        else:
            passes = float(x[component]) <= center[component] + constant_tolerance
        if not passes:
            raise ModelError(
                "observed adequacy value fails a constant component",
                details={"component": int(component)},
            )
    if bool(active.any()):
        replicate_statistics = np.full(len(reference), -math.inf, dtype=np.float64)
        observed_statistic = -math.inf
    else:
        replicate_statistics = np.zeros(len(reference), dtype=np.float64)
        observed_statistic = 0.0
    for component in np.flatnonzero(active):
        standardized_reference = (reference[:, component] - center[component]) / scale[
            component
        ]
        standardized_observed = float(
            (x[component] - center[component]) / scale[component]
        )
        if checked_directions[component] is EndpointDirection.TWO_SIDED:
            standardized_reference = np.abs(standardized_reference)
            standardized_observed = abs(standardized_observed)
        replicate_statistics = np.maximum(replicate_statistics, standardized_reference)
        observed_statistic = max(observed_statistic, standardized_observed)
    p_value = float(
        (1 + np.count_nonzero(replicate_statistics >= observed_statistic))
        / (len(reference) + 1)
    )
    center = center.astype(np.float64, copy=True)
    scale = scale.astype(np.float64, copy=True)
    active = active.astype(np.bool_, copy=True)
    _readonly(center, scale, active, replicate_statistics)
    return MaxStatisticCell(
        observed_statistic=float(observed_statistic),
        replicate_statistics=replicate_statistics,
        p_value=p_value,
        center=center,
        scale=scale,
        active_components=active,
        successful_replicates=len(reference),
    )


def four_cell_holm(p_values: ArrayLike, *, alpha: float = 0.05) -> HolmResult:
    """Apply Holm to exactly design/production latent/adequacy cells."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.shape != (4,):
        raise ModelError("the adequacy family must contain exactly four cells")
    return holm_correction(values, alpha=alpha)


def predictive_mixture_diagnostics(
    values: ArrayLike,
    predictive_weights: ArrayLike,
    state_means: ArrayLike,
    tied_covariance: ArrayLike,
) -> PredictiveMixtureDiagnostics:
    """Compute one-step mixture residuals, covariance error, PITs, and CvM."""

    x = _matrix(values, "predictive values")
    weights = np.asarray(predictive_weights, dtype=np.float64)
    means = _matrix(state_means, "state means")
    covariance = _covariance(tied_covariance, means.shape[1])
    if weights.shape != (len(x), len(means)):
        raise ModelError("predictive weight shape is inconsistent")
    if not np.isfinite(weights).all() or bool((weights < 0.0).any()):
        raise ModelError("predictive weights must be finite and non-negative")
    row_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    if row_error > 1e-12:
        raise ModelError("predictive weights must sum to one")
    if x.shape[1] != means.shape[1]:
        raise ModelError("predictive value and state-mean dimensions differ")
    predictive_means = weights @ means
    residuals = np.empty_like(x)
    for row in range(len(x)):
        difference = means - predictive_means[row]
        predictive_covariance = covariance.copy()
        predictive_covariance += np.einsum(
            "s,si,sj->ij", weights[row], difference, difference
        )
        try:
            cholesky = np.linalg.cholesky(predictive_covariance)
            residuals[row] = np.linalg.solve(cholesky, x[row] - predictive_means[row])
        except np.linalg.LinAlgError as error:
            raise ModelError("predictive covariance Cholesky failed") from error
    covariance_error = residuals.T @ residuals / len(residuals)
    covariance_error -= np.eye(residuals.shape[1])
    upper = covariance_error[np.triu_indices(residuals.shape[1])]
    standard_deviation = np.sqrt(np.diag(covariance))
    standardized = (x[:, None, :] - means[None, :, :]) / standard_deviation
    pits = np.einsum("ts,tsj->tj", weights, ndtr(standardized))
    pits = np.clip(pits, PIT_CLIP, 1.0 - PIT_CLIP)
    cvm = np.asarray(
        [_cramer_von_mises(pits[:, feature]) for feature in range(pits.shape[1])],
        dtype=np.float64,
    )
    predictive_means = predictive_means.astype(np.float64, copy=True)
    residuals = residuals.astype(np.float64, copy=True)
    upper = upper.astype(np.float64, copy=True)
    pits = pits.astype(np.float64, copy=True)
    _readonly(predictive_means, residuals, upper, pits, cvm)
    return PredictiveMixtureDiagnostics(
        predictive_means,
        residuals,
        upper,
        pits,
        cvm,
    )


def residual_acf_diagnostics(
    residuals: ArrayLike,
    continuation: ArrayLike,
    lags: tuple[int, ...],
    *,
    states: ArrayLike | None = None,
    minimum_pairs: int = 10,
) -> ACFDiagnostics:
    """Compute raw/squared available-pair cosine ACFs without crossing gaps/runs."""

    values = _matrix(residuals, "residual matrix")
    links = _continuation(continuation, len(values))
    if not lags:
        raise ModelError("residual ACF lags must be non-empty")
    for lag in lags:
        _positive_integer(lag, "residual ACF lag")
    _positive_integer(minimum_pairs, "minimum_pairs")
    run_ids: IntArray | None = None
    if states is not None:
        state_values = _state_vector(states, len(values))
        run_ids = np.zeros(len(values), dtype=np.int64)
        current_run = 0
        for row in range(1, len(values)):
            if not bool(links[row]) or state_values[row] != state_values[row - 1]:
                current_run += 1
            run_ids[row] = current_run
    raw = np.empty((values.shape[1], len(lags)), dtype=np.float64)
    squared = np.empty_like(raw)
    raw_counts = np.empty(raw.shape, dtype=np.int64)
    squared_counts = np.empty(raw.shape, dtype=np.int64)
    centered = values - values.mean(axis=0)
    centered_squared = values * values - (values * values).mean(axis=0)
    for lag_index, lag in enumerate(lags):
        eligible = np.ones(len(values) - lag, dtype=np.bool_)
        for step in range(1, lag + 1):
            eligible &= links[lag - step + 1 : len(values) - step + 1]
        if run_ids is not None:
            eligible &= run_ids[lag:] == run_ids[:-lag]
        count = int(np.count_nonzero(eligible))
        if count < minimum_pairs:
            raise ModelError(
                "residual ACF has insufficient valid pairs",
                details={"lag": lag, "actual": count, "required": minimum_pairs},
            )
        for feature in range(values.shape[1]):
            raw[feature, lag_index] = _cosine_pair_correlation(
                centered[lag:, feature][eligible],
                centered[:-lag, feature][eligible],
            )
            squared[feature, lag_index] = _cosine_pair_correlation(
                centered_squared[lag:, feature][eligible],
                centered_squared[:-lag, feature][eligible],
            )
            raw_counts[feature, lag_index] = count
            squared_counts[feature, lag_index] = count
    _readonly(raw, squared, raw_counts, squared_counts)
    return ACFDiagnostics(lags, raw, squared, raw_counts, squared_counts)


def marginal_moments(values: ArrayLike) -> FloatArray:
    """Return population mean, variance, skewness, and non-excess kurtosis."""

    matrix = _matrix(values, "marginal-moment values")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    second = np.mean(centered**2, axis=0)
    if not np.isfinite(second).all():
        raise ModelError("marginal variance is non-finite")
    if bool((second <= 1e-12).any()):
        raise AdequacyEndpointSupportFailure("marginal variance is at or below 1e-12")
    third = np.mean(centered**3, axis=0) / second**1.5
    fourth = np.mean(centered**4, axis=0) / second**2
    result = np.column_stack([mean, second, third, fourth]).astype(
        np.float64, copy=False
    )
    _readonly(result)
    return result


def _uniform_matrix(
    values: ArrayLike, *, expected_shape: tuple[int, int], label: str
) -> FloatArray:
    raw = np.asarray(values)
    if raw.shape != expected_shape:
        raise ModelError(
            f"{label} has the wrong exact shape",
            details={"actual": raw.shape, "expected": expected_shape},
        )
    if raw.dtype.kind == "b":
        raise ModelError(f"{label} must be numeric non-boolean values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{label} must be numeric") from error
    if not np.isfinite(result).all():
        raise ModelError(f"{label} contains non-finite values")
    if bool((result < 0.0).any()) or bool((result >= 1.0).any()):
        raise ModelError(f"{label} must lie in the half-open interval [0, 1)")
    return result


def _bootstrap_index_matrix(
    values: ArrayLike,
    *,
    expected_shape: tuple[int, int],
    chain_count: int,
    label: str,
) -> NDArray[np.generic]:
    raw = np.asarray(values)
    if raw.shape != expected_shape:
        raise ModelError(
            f"{label} has the wrong exact shape",
            details={"actual": raw.shape, "expected": expected_shape},
        )
    if raw.dtype.kind not in {"i", "u"}:
        raise ModelError(f"{label} must have integer dtype")
    if bool((raw < 0).any()) or bool((raw >= chain_count).any()):
        raise ModelError(f"{label} contains an out-of-range chain ID")
    return raw


def _validated_latent_simulation(
    simulation: LatentChainSimulation,
) -> LatentChainSimulation:
    if not isinstance(simulation, LatentChainSimulation):
        raise ModelError("latent bootstrap input must be LatentChainSimulation")
    _positive_integer(simulation.chain_count, "latent simulation chain_count")
    _positive_integer(simulation.measured_months, "latent simulation measured_months")
    _positive_integer(simulation.extension_months, "latent simulation extension_months")
    transition = _transition_matrix(simulation.transition_matrix)
    expected_stationary = _stationary_distribution(transition)
    stationary = np.asarray(simulation.stationary_distribution, dtype=np.float64)
    if stationary.shape != (len(transition),) or not np.isfinite(stationary).all():
        raise ModelError("latent simulation stationary distribution is invalid")
    if float(np.max(np.abs(stationary - expected_stationary))) > 1e-12:
        raise ModelError(
            "latent simulation stationary distribution does not match transition"
        )
    expected_columns = 1 + simulation.measured_months + simulation.extension_months
    chains = _state_chains(
        simulation.chains,
        n_states=len(transition),
        expected_columns=expected_columns,
    )
    if len(chains) != simulation.chain_count:
        raise ModelError("latent simulation chain_count metadata is inconsistent")
    if simulation.uniforms_per_chain != expected_columns:
        raise ModelError(
            "latent simulation uniform-consumption metadata is inconsistent"
        )
    return simulation


def _latent_cluster_sufficient_statistics(
    chains: ArrayLike,
    *,
    n_states: int,
    measured_months: int,
    extension_months: int,
) -> _LatentClusterSufficientStatistics:
    total_months = measured_months + extension_months
    values = _state_chains(
        chains,
        n_states=n_states,
        expected_columns=1 + total_months,
    )
    chain_count = len(values)
    previous = values[:, :measured_months]
    current = values[:, 1 : measured_months + 1]
    linear = np.empty((chain_count, n_states + n_states * n_states), dtype=np.int64)
    for state in range(n_states):
        linear[:, state] = np.count_nonzero(current == state, axis=1)
    transition_offset = n_states
    for source in range(n_states):
        source_rows = previous == source
        for target in range(n_states):
            linear[:, transition_offset + source * n_states + target] = (
                np.count_nonzero(source_rows & (current == target), axis=1)
            )

    row_parts: list[IntArray] = []
    column_parts: list[IntArray] = []
    data_parts: list[IntArray] = []
    for chain_index, chain in enumerate(values):
        changes = np.flatnonzero(chain[1:] != chain[:-1]).astype(np.int64, copy=False)
        changes += 1
        measured_change_count = int(
            np.searchsorted(changes, measured_months, side="right")
        )
        if measured_change_count == 0:
            continue
        if measured_change_count >= len(changes):
            raise ModelError(
                "latent spell remains incomplete after the fixed extension",
                details={
                    "chain": chain_index,
                    "measured_start_column": int(changes[measured_change_count - 1]),
                },
            )
        starts = changes[:measured_change_count]
        ends = changes[1 : measured_change_count + 1]
        lengths = ends - starts
        states = np.asarray(chain[starts], dtype=np.int64)
        if bool((lengths <= 0).any()) or bool((lengths > total_months).any()):
            raise ModelError("latent complete dwell length is invalid")
        columns = states * total_months + lengths - 1
        unique_columns, column_counts = np.unique(columns, return_counts=True)
        row_parts.append(np.full(len(unique_columns), chain_index, dtype=np.int64))
        column_parts.append(unique_columns.astype(np.int64, copy=False))
        data_parts.append(column_counts.astype(np.int64, copy=False))

    histogram_shape = (chain_count, n_states * total_months)
    if row_parts:
        rows = np.concatenate(row_parts)
        columns = np.concatenate(column_parts)
        data = np.concatenate(data_parts)
        dwell_histogram = coo_matrix(
            (data, (rows, columns)), shape=histogram_shape, dtype=np.int64
        ).tocsr()
        dwell_histogram.sum_duplicates()
        dwell_histogram.sort_indices()
    else:
        dwell_histogram = csr_matrix(histogram_shape, dtype=np.int64)
    _readonly(linear)
    return _LatentClusterSufficientStatistics(
        occupancy_and_transitions=linear,
        dwell_histogram=dwell_histogram,
        n_states=n_states,
        total_months=total_months,
    )


def _cluster_resample_weights(
    indices: NDArray[np.generic], *, chain_count: int, resample_size: int
) -> IntArray:
    weights = np.empty((len(indices), chain_count), dtype=np.int64)
    for row, selected in enumerate(indices):
        weights[row] = np.bincount(
            selected.astype(np.int64, copy=False), minlength=chain_count
        )
    if bool((weights.sum(axis=1) != resample_size).any()):
        raise ModelError("latent bootstrap RNG/index shape consumed a wrong count")
    return weights


def _latent_estimates_from_counts(
    linear_counts: ArrayLike,
    dwell_counts: ArrayLike,
    *,
    n_states: int,
    measured_months: int,
    resample_size: int,
    survival_months: int,
    total_months: int,
) -> FloatArray:
    linear = np.asarray(linear_counts, dtype=np.int64)
    dwell = np.asarray(dwell_counts, dtype=np.int64)
    expected_linear_columns = n_states + n_states * n_states
    if (
        linear.ndim != 2
        or linear.shape[1] != expected_linear_columns
        or dwell.shape != (len(linear), n_states * total_months)
    ):
        raise ModelError("latent sufficient-statistic geometry is inconsistent")
    if bool((linear < 0).any()) or bool((dwell < 0).any()):
        raise ModelError("latent sufficient statistics contain negative counts")
    occupancy = linear[:, :n_states]
    transitions = linear[:, n_states:].reshape(-1, n_states, n_states)
    departures = transitions.sum(axis=2, dtype=np.int64)
    if bool((departures == 0).any()):
        raise ModelError("latent transition denominator is zero")
    dwell_by_state = dwell.reshape(-1, n_states, total_months)
    episode_counts = dwell_by_state.sum(axis=2, dtype=np.int64)
    if bool((episode_counts == 0).any()):
        raise ModelError("latent episode count is zero")
    cumulative = np.cumsum(dwell_by_state, axis=2, dtype=np.int64)
    median_threshold = episode_counts // 2 + 1
    median_dwell = (
        np.count_nonzero(cumulative < median_threshold[:, :, None], axis=2) + 1
    )
    survival_counts = np.empty((len(linear), n_states, survival_months), dtype=np.int64)
    survival_counts[:, :, 0] = episode_counts
    if survival_months > 1:
        survival_counts[:, :, 1:] = (
            episode_counts[:, :, None] - cumulative[:, :, : survival_months - 1]
        )

    denominator = resample_size * measured_months
    off_diagonal: FloatArray = np.asarray(
        np.column_stack(
            [
                transitions[:, source, target] / departures[:, source]
                for source in range(n_states)
                for target in range(n_states)
                if source != target
            ]
        ),
        dtype=np.float64,
    )
    result: FloatArray = np.asarray(
        np.concatenate(
            [
                occupancy / denominator,
                off_diagonal,
                episode_counts / denominator,
                dwell_by_state[:, :, 0] / episode_counts,
                median_dwell.astype(np.float64),
                (survival_counts / episode_counts[:, :, None]).reshape(len(linear), -1),
            ],
            axis=1,
        ),
        dtype=np.float64,
    )
    expected_endpoints = len(_latent_labels(n_states, survival_months))
    if result.shape != (len(linear), expected_endpoints):
        raise AssertionError("latent bootstrap endpoint order is inconsistent")
    if not np.isfinite(result).all():
        raise ModelError("latent sufficient-statistic ratios are non-finite")
    return result


def _latent_labels(n_states: int, survival_months: int) -> tuple[str, ...]:
    labels: list[str] = [f"occupancy/{state}" for state in range(n_states)]
    labels.extend(
        f"transition/{source}/{target}"
        for source in range(n_states)
        for target in range(n_states)
        if source != target
    )
    labels.extend(f"episode_rate/{state}" for state in range(n_states))
    labels.extend(f"one_period/{state}" for state in range(n_states))
    labels.extend(f"median_dwell/{state}" for state in range(n_states))
    labels.extend(
        f"dwell_survival/{state}/{month}"
        for state in range(n_states)
        for month in range(1, survival_months + 1)
    )
    return tuple(labels)


def _cramer_von_mises(values: FloatArray) -> float:
    ordered = np.sort(values)
    count = len(ordered)
    target = (2 * np.arange(1, count + 1) - 1) / (2 * count)
    return float(1.0 / (12 * count) + np.sum((ordered - target) ** 2))


def _cosine_pair_correlation(left: FloatArray, right: FloatArray) -> float:
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if not np.isfinite(denominator):
        raise ModelError("residual ACF denominator is non-finite")
    if denominator <= 1e-12:
        raise AdequacyEndpointSupportFailure("residual ACF denominator is unavailable")
    value = float(left @ right) / denominator
    if not np.isfinite(value):
        raise ModelError("residual ACF is non-finite")
    return value


def _transition_matrix(values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ModelError("transition matrix must be non-empty and square")
    if not np.isfinite(matrix).all() or bool((matrix < 0.0).any()):
        raise ModelError("transition matrix must be finite and non-negative")
    if float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) > 1e-12:
        raise ModelError("transition matrix rows must sum to one")
    return matrix


def _stationary_distribution(transition: FloatArray) -> FloatArray:
    size = len(transition)
    system = np.vstack([transition.T - np.eye(size), np.ones(size)])
    target = np.concatenate([np.zeros(size), np.ones(1)])
    try:
        solution, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    except np.linalg.LinAlgError as error:
        raise ModelError("stationary distribution solve failed") from error
    if rank < size or not np.isfinite(solution).all():
        raise ModelError("stationary distribution is unavailable or non-unique")
    if bool((solution < -1e-12).any()):
        raise ModelError("stationary distribution is negative")
    solution = np.maximum(solution, 0.0)
    solution /= solution.sum()
    return np.asarray(solution, dtype=np.float64)


def _state_chains(
    values: ArrayLike, *, n_states: int, expected_columns: int
) -> StateArray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != expected_columns:
        raise ModelError("latent chains have the wrong sentinel/measurement geometry")
    if raw.dtype.kind not in {"i", "u"}:
        raise ModelError("latent chains must have integer dtype")
    if bool((raw < 0).any()) or bool((raw >= n_states).any()):
        raise ModelError("latent chains contain an out-of-range state")
    return raw


def _matrix(values: ArrayLike, label: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ModelError(f"{label} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ModelError(f"{label} contains non-finite values")
    return matrix


def _vector(values: ArrayLike, label: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ModelError(f"{label} must be a non-empty finite vector")
    return vector


def _covariance(values: ArrayLike, dimension: int) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ModelError("tied covariance shape or values are invalid")
    symmetric = 0.5 * (matrix + matrix.T)
    if float(np.linalg.eigvalsh(symmetric).min()) <= 0.0:
        raise ModelError("tied covariance must be positive definite")
    return symmetric


def _continuation(values: ArrayLike, length: int) -> BoolArray:
    raw = np.asarray(values)
    if raw.shape != (length,) or raw.dtype.kind != "b":
        raise ModelError("continuation must be a boolean vector matching rows")
    result = raw.astype(np.bool_, copy=False)
    if bool(result[0]):
        raise ModelError("continuation must be false at row zero")
    return result


def _state_vector(values: ArrayLike, length: int) -> IntArray:
    raw = np.asarray(values)
    if raw.shape != (length,) or raw.dtype.kind not in {"i", "u"}:
        raise ModelError("states must be an integer vector matching rows")
    return raw.astype(np.int64, copy=False)


def _positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)
