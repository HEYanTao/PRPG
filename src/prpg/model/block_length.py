"""Version-2 stationary-bootstrap block-length calibration primitives.

The automatic selector implements the flat-top spectral plug-in method from
Politis and White (2004), with the corrected stationary-bootstrap variance
constant ``D_SB = 2 * g_hat(0)^2`` from Patton, Politis, and White (2009).
Version 2 retains the version-1 estimator unchanged and prospectively raises
only the daily mean intended block-length hard cap from 60 to 126 sessions.
It freezes available-pair autocorrelations for the bandwidth search,
``gamma(k)`` with divisor ``n``, the conservative band
``2*sqrt(log10(n)/n)``, and the finite-sample bound from Patton's revised
reference implementation.

Primary references:

* D. N. Politis and H. White, "Automatic Block-Length Selection for the
  Dependent Bootstrap", Econometric Reviews 23(1), 53-70 (2004),
  https://doi.org/10.1081/ETC-120028836.
* A. J. Patton, D. N. Politis, and H. White, "Correction to Automatic
  Block-Length Selection for the Dependent Bootstrap", Econometric Reviews
  28(4), 372-375 (2009), https://doi.org/10.1080/07474930802459016.
* Corrected reference code:
  https://public.econ.duke.edu/~ap172/opt_block_length_REV_dec07.txt.

This module is deterministic and side-effect free.  It does not fit an HMM,
simulate calibration traces, or write artifacts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from prpg.errors import ModelError

Frequency: TypeAlias = Literal["monthly", "daily"]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


class PreliminaryPPWInsufficientSupport(ModelError):
    """A valid preliminary series cannot support the registered PPW rule."""


class PreliminaryPPWNumericalFailure(ModelError):
    """A code-owned PPW/PCA numerical gate is not attained."""


class PreliminaryBlockLengthCapFailure(ModelError):
    """A valid preliminary estimate exceeds a binding scientific hard cap."""


class NoSupportedBlockLength(ModelError):
    """No candidate block length attains the registered support gates."""


class BlockSupportNumericalFailure(ModelError):
    """A code-derived effective block statistic is numerically invalid."""


BLOCK_POLICY_VERSION = "ppw2009-stationary-frequency-wide-v3"
MONTHLY_BLOCK_LENGTH_BOUNDS: Final = (2, 12)
DAILY_BLOCK_LENGTH_BOUNDS: Final = (2, 126)
DIAGNOSTIC_NAMES = (
    "equity_return",
    "muni_return",
    "taxable_return",
    "pc1",
    "taxable_minus_muni",
    "abs_pc1",
    "squared_pc1",
)
MACRO_FEATURE_NAMES = (
    "industrial_production_growth",
    "inflation_yoy",
    "yield_curve_spread",
    "credit_spread",
)
MACRO_DIAGNOSTIC_NAMES = (
    *MACRO_FEATURE_NAMES,
    "macro_pc1",
    "abs_macro_pc1",
    "squared_macro_pc1",
)
_BOUNDS: dict[Frequency, tuple[int, int]] = {
    "monthly": MONTHLY_BLOCK_LENGTH_BOUNDS,
    "daily": DAILY_BLOCK_LENGTH_BOUNDS,
}


@dataclass(frozen=True, slots=True)
class DiagnosticSeriesSet:
    """The exact seven raw and standardized diagnostic series."""

    names: tuple[str, ...]
    raw: FloatArray
    standardized: FloatArray
    pc1_loadings: FloatArray


@dataclass(frozen=True, slots=True)
class BlockLengthEstimate:
    """One corrected PPW stationary-bootstrap estimate."""

    series_name: str
    observations: int
    consecutive_insignificant_lags: int
    maximum_lag: int
    selected_bandwidth: int
    correlation_band: float
    g_hat: float | None
    long_run_variance: float | None
    variance_constant: float | None
    unconstrained_length: float
    finite_sample_upper_bound: float
    stationary_length: float
    finite_sample_truncated: bool
    m_hat: int = -1


@dataclass(frozen=True, slots=True)
class FrequencyWideLength:
    """Ceiling/max combination and the approved absolute bound application."""

    frequency: Frequency
    diagnostic_lengths: tuple[tuple[str, float], ...]
    controlling_series: tuple[str, ...]
    raw_ceiling_length: int
    starting_length: int
    lower_bound: int
    upper_bound: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrequencyCalibration:
    """Seven-series selector output before state-support reduction."""

    policy_version: str
    diagnostics: DiagnosticSeriesSet
    estimates: tuple[BlockLengthEstimate, ...]
    combined: FrequencyWideLength


@dataclass(frozen=True, slots=True)
class MacroWideLength:
    """Ceiling/max combination and both binding macro upper bounds."""

    diagnostic_lengths: tuple[tuple[str, float], ...]
    controlling_series: tuple[str, ...]
    raw_ceiling_length: int
    starting_length: int
    lower_bound: int
    absolute_upper_bound: int
    sample_support_upper_bound: int


@dataclass(frozen=True, slots=True)
class MacroCalibration:
    """Seven-series macro-bootstrap selector output."""

    policy_version: str
    diagnostics: DiagnosticSeriesSet
    estimates: tuple[BlockLengthEstimate, ...]
    combined: MacroWideLength


@dataclass(frozen=True, slots=True)
class StateSupport:
    """Support evidence for one state at one intended block length."""

    state: int
    intended_length: int
    completed_episodes: int
    eligible_start_indices: tuple[int, ...]

    @property
    def eligible_start_count(self) -> int:
        return len(self.eligible_start_indices)


@dataclass(frozen=True, slots=True)
class LengthTrial:
    """One downward-search candidate and every rejection reason."""

    intended_length: int
    required_eligible_starts: int
    support: tuple[StateSupport, ...]
    realized_means: tuple[tuple[int, float], ...]
    reasons: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class BlockPolicySelection:
    """Selected supported length and all tried larger lengths."""

    frequency: Frequency
    selected_length: int
    restart_probability: float
    trials: tuple[LengthTrial, ...]


@dataclass(frozen=True, slots=True)
class RealizedBlockStats:
    """Realized effective block-length distribution for one target state."""

    state: int
    lengths: tuple[int, ...]
    mean: float
    minimum: int
    maximum: int
    median: float = math.nan
    singleton_fraction: float = math.nan
    quantile_05: float = math.nan
    quantile_95: float = math.nan
    count: int = -1

    def __post_init__(self) -> None:
        if self.count == -1:
            object.__setattr__(self, "count", len(self.lengths))


@dataclass(frozen=True, slots=True)
class PairedFragmentationGate:
    """Binding conditioned-versus-ideal fragmentation decision for one state."""

    state: int
    intended_length: int
    conditioned: RealizedBlockStats
    ideal: RealizedBlockStats
    conditioned_mean_minimum: float
    conditioned_median_minimum: float
    singleton_fraction_maximum_from_ideal: float
    absolute_singleton_fraction_maximum: float
    reasons: tuple[str, ...]

    @property
    def conditioned_mean_passed(self) -> bool:
        return self.conditioned.mean >= self.conditioned_mean_minimum

    @property
    def conditioned_median_passed(self) -> bool:
        return self.conditioned.median >= self.conditioned_median_minimum

    @property
    def relative_singleton_passed(self) -> bool:
        return (
            self.conditioned.singleton_fraction
            <= self.singleton_fraction_maximum_from_ideal
        )

    @property
    def absolute_singleton_passed(self) -> bool:
        return (
            self.conditioned.singleton_fraction
            <= self.absolute_singleton_fraction_maximum
        )

    @property
    def passed(self) -> bool:
        return not self.reasons


def build_diagnostic_series(returns: npt.ArrayLike) -> DiagnosticSeriesSet:
    """Build the seven binding return diagnostics using correlation PC1.

    Columns are ordered equity, municipal bond, taxable bond.  PC1 is the top
    eigenvector of their sample correlation matrix, applied to sample-
    standardized columns.  The top eigenvalue must be identified and the PC1
    sign is fixed by requiring a positive equity loading.
    """

    values = _as_float_matrix(returns, columns=3, name="returns")
    if values.shape[0] < 3:
        raise PreliminaryPPWInsufficientSupport(
            "at least three synchronized returns are required for PCA",
            details={"observations": values.shape[0]},
        )
    pc1, loadings = _correlation_pc1(
        values,
        name="return correlation PC1",
        sign_anchor=0,
        anchor_name="equity",
    )
    raw = np.column_stack(
        (
            values[:, 0],
            values[:, 1],
            values[:, 2],
            pc1,
            values[:, 2] - values[:, 1],
            np.abs(pc1),
            np.square(pc1),
        )
    )
    standardized = np.empty_like(raw, dtype=np.float64)
    for column, name in enumerate(DIAGNOSTIC_NAMES):
        standardized[:, column] = _standardize(raw[:, column], name=name)
    return DiagnosticSeriesSet(
        names=DIAGNOSTIC_NAMES,
        raw=_readonly(raw),
        standardized=_readonly(standardized),
        pc1_loadings=_readonly(loadings),
    )


def build_macro_diagnostic_series(
    macro_features: npt.ArrayLike,
) -> DiagnosticSeriesSet:
    """Build the four standardized macro features and three PC1 diagnostics."""

    values = _as_float_matrix(macro_features, columns=4, name="macro features")
    if values.shape[0] < 3:
        raise PreliminaryPPWInsufficientSupport(
            "at least three synchronized macro observations are required for PCA",
            details={"observations": values.shape[0]},
        )
    standardized_features = np.column_stack(
        tuple(
            _standardize(values[:, column], name=name)
            for column, name in enumerate(MACRO_FEATURE_NAMES)
        )
    )
    pc1, loadings = _correlation_pc1(
        standardized_features,
        name="macro correlation PC1",
        sign_anchor=None,
        anchor_name=None,
    )
    raw = np.column_stack(
        (
            standardized_features,
            pc1,
            np.abs(pc1),
            np.square(pc1),
        )
    )
    standardized = np.empty_like(raw, dtype=np.float64)
    for column, name in enumerate(MACRO_DIAGNOSTIC_NAMES):
        standardized[:, column] = _standardize(raw[:, column], name=name)
    return DiagnosticSeriesSet(
        names=MACRO_DIAGNOSTIC_NAMES,
        raw=_readonly(raw),
        standardized=_readonly(standardized),
        pc1_loadings=_readonly(loadings),
    )


def stationary_bootstrap_length(
    series: npt.ArrayLike,
    *,
    series_name: str = "series",
    continuation: npt.ArrayLike | None = None,
) -> BlockLengthEstimate:
    """Estimate the binding corrected, gap-aware PPW stationary block length.

    ``continuation`` may use forward links (length ``n-1``) or the project row
    representation (length ``n``, false at every segment start).  Omitting it
    declares one gap-free segment.
    """

    x = _as_float_vector(series, name=series_name)
    nobs = x.size
    if nobs < 20:
        raise PreliminaryPPWInsufficientSupport(
            "PPW block-length selection requires at least 20 observations",
            details={"series": series_name, "observations": nobs},
        )
    centered = x - float(np.mean(x))
    variance = float(np.mean(np.square(centered)))
    if not math.isfinite(variance) or variance <= 1e-12:
        raise PreliminaryPPWNumericalFailure(
            "PPW diagnostic variance is at or below the binding floor",
            details={"series": series_name, "variance": variance, "floor": 1e-12},
        )
    eps = centered / math.sqrt(variance)
    links = _selector_continuation_links(continuation, nobs=nobs)
    segment_ids, maximum_segment_length = _segment_geometry(links, nobs=nobs)
    kn = max(5, math.ceil(math.sqrt(math.log10(nobs))))
    maximum_lag = min(
        maximum_segment_length - 1,
        math.ceil(math.sqrt(nobs)) + kn,
    )
    if maximum_lag < kn:
        raise PreliminaryPPWInsufficientSupport(
            "PPW diagnostic has insufficient within-segment lag support",
            details={
                "series": series_name,
                "maximum_segment_length": maximum_segment_length,
                "maximum_lag": maximum_lag,
                "required_lags": kn,
            },
        )
    correlation_band = 2.0 * math.sqrt(math.log10(nobs) / nobs)

    autocovariances = np.empty(maximum_lag + 1, dtype=np.float64)
    correlations = np.empty(maximum_lag + 1, dtype=np.float64)
    autocovariances[0] = float(np.dot(eps, eps) / nobs)
    correlations[0] = 1.0
    for lag in range(1, maximum_lag + 1):
        available = segment_ids[lag:] == segment_ids[:-lag]
        left = eps[lag:][available]
        right = eps[:-lag][available]
        cross = float(np.dot(left, right))
        autocovariances[lag] = cross / nobs
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise PreliminaryPPWNumericalFailure(
                "PPW available-pair correlation denominator is unavailable",
                details={"series": series_name, "lag": lag},
            )
        correlations[lag] = cross / denominator

    first_quiet_lag: int | None = None
    last_start = maximum_lag - kn + 1
    for start in range(1, last_start + 1):
        if bool(np.all(np.abs(correlations[start : start + kn]) < correlation_band)):
            first_quiet_lag = start
            break
    significant = np.flatnonzero(np.abs(correlations[1:]) >= correlation_band).astype(
        np.int64
    )
    if significant.size == 0:
        # Make the approved no-significant-lag b=1 branch explicit.  Without
        # this precedence, the all-quiet case would be swallowed by m=1.
        m_hat = 0
    elif first_quiet_lag is not None:
        m_hat = first_quiet_lag
    else:
        m_hat = int(significant[-1]) + 1

    finite_sample_upper = float(math.ceil(min(3.0 * math.sqrt(nobs), nobs / 3.0)))
    if m_hat == 0:
        return BlockLengthEstimate(
            series_name=series_name,
            observations=nobs,
            consecutive_insignificant_lags=kn,
            maximum_lag=maximum_lag,
            selected_bandwidth=0,
            correlation_band=correlation_band,
            g_hat=None,
            long_run_variance=None,
            variance_constant=None,
            unconstrained_length=1.0,
            finite_sample_upper_bound=finite_sample_upper,
            stationary_length=1.0,
            finite_sample_truncated=False,
            m_hat=0,
        )

    selected_bandwidth = min(2 * m_hat, maximum_lag)
    g_hat = 0.0
    long_run_variance = float(autocovariances[0])
    for lag in range(1, selected_bandwidth + 1):
        ratio = lag / selected_bandwidth
        weight = 1.0 if ratio <= 0.5 else 2.0 * (1.0 - ratio)
        g_hat += 2.0 * weight * lag * float(autocovariances[lag])
        long_run_variance += 2.0 * weight * float(autocovariances[lag])

    variance_constant = 2.0 * long_run_variance**2
    numerator = 2.0 * g_hat**2
    if (
        not math.isfinite(g_hat)
        or not math.isfinite(long_run_variance)
        or not math.isfinite(variance_constant)
        or variance_constant <= 1e-12
    ):
        raise PreliminaryPPWNumericalFailure(
            "PPW stationary block-length estimate is invalid",
            details={
                "series": series_name,
                "m_hat": m_hat,
                "bandwidth": selected_bandwidth,
                "g_hat": g_hat,
                "long_run_variance": long_run_variance,
                "variance_constant": variance_constant,
            },
        )
    unconstrained = ((numerator / variance_constant) * nobs) ** (1.0 / 3.0)
    if not math.isfinite(unconstrained) or unconstrained <= 0.0:
        raise PreliminaryPPWNumericalFailure(
            "PPW stationary block-length estimate is invalid",
            details={
                "series": series_name,
                "m_hat": m_hat,
                "bandwidth": selected_bandwidth,
                "g_hat": g_hat,
                "long_run_variance": long_run_variance,
                "variance_constant": variance_constant,
                "unconstrained_length": unconstrained,
            },
        )
    stationary_length = min(unconstrained, finite_sample_upper)
    return BlockLengthEstimate(
        series_name=series_name,
        observations=nobs,
        consecutive_insignificant_lags=kn,
        maximum_lag=maximum_lag,
        selected_bandwidth=selected_bandwidth,
        correlation_band=correlation_band,
        g_hat=g_hat,
        long_run_variance=long_run_variance,
        variance_constant=variance_constant,
        unconstrained_length=unconstrained,
        finite_sample_upper_bound=finite_sample_upper,
        stationary_length=stationary_length,
        finite_sample_truncated=unconstrained > finite_sample_upper,
        m_hat=m_hat,
    )


def estimate_frequency_start(
    returns: npt.ArrayLike,
    *,
    frequency: Frequency,
    continuation: npt.ArrayLike | None = None,
) -> FrequencyCalibration:
    """Estimate all seven lengths and combine them for one base frequency."""

    diagnostics = build_diagnostic_series(returns)
    estimates = tuple(
        stationary_bootstrap_length(
            diagnostics.standardized[:, column],
            series_name=name,
            continuation=continuation,
        )
        for column, name in enumerate(diagnostics.names)
    )
    return FrequencyCalibration(
        policy_version=BLOCK_POLICY_VERSION,
        diagnostics=diagnostics,
        estimates=estimates,
        combined=combine_diagnostic_lengths(
            {estimate.series_name: estimate for estimate in estimates},
            frequency=frequency,
        ),
    )


def estimate_macro_start(
    macro_features: npt.ArrayLike,
    *,
    continuation: npt.ArrayLike | None = None,
) -> MacroCalibration:
    """Estimate the separate seven-diagnostic macro-bootstrap start length."""

    diagnostics = build_macro_diagnostic_series(macro_features)
    estimates = tuple(
        stationary_bootstrap_length(
            diagnostics.standardized[:, column],
            series_name=name,
            continuation=continuation,
        )
        for column, name in enumerate(diagnostics.names)
    )
    return MacroCalibration(
        policy_version=BLOCK_POLICY_VERSION,
        diagnostics=diagnostics,
        estimates=estimates,
        combined=combine_macro_diagnostic_lengths(
            {estimate.series_name: estimate for estimate in estimates},
            observations=diagnostics.raw.shape[0],
        ),
    )


def combine_diagnostic_lengths(
    estimates: Mapping[str, BlockLengthEstimate | float], *, frequency: Frequency
) -> FrequencyWideLength:
    """Apply ``ceil(max(L_j))`` and the approved monthly/daily bounds."""

    lower, upper = _frequency_bounds(frequency)
    supplied = set(estimates)
    expected = set(DIAGNOSTIC_NAMES)
    if supplied != expected:
        raise ModelError(
            "frequency-wide combination requires exactly seven diagnostics",
            details={
                "missing": sorted(expected - supplied),
                "unexpected": sorted(supplied - expected),
            },
        )
    values: list[tuple[str, float]] = []
    for name in DIAGNOSTIC_NAMES:
        item = estimates[name]
        value = (
            item.stationary_length if isinstance(item, BlockLengthEstimate) else item
        )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ModelError(
                "diagnostic block length is non-finite or non-positive",
                details={"series": name, "estimate": numeric},
            )
        values.append((name, numeric))
    maximum = max(value for _, value in values)
    unfloored = math.ceil(maximum)
    raw = max(lower, unfloored)
    if raw > upper:
        raise PreliminaryBlockLengthCapFailure(
            "raw return block length exceeds the binding absolute hard cap",
            details={
                "frequency": frequency,
                "raw_ceiling_length": raw,
                "upper_bound": upper,
            },
        )
    starting = raw
    reasons: list[str] = []
    if unfloored < lower:
        reasons.append("raised_to_absolute_lower_bound")
    controlling = tuple(
        name
        for name, value in values
        if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-15)
    )
    return FrequencyWideLength(
        frequency=frequency,
        diagnostic_lengths=tuple(values),
        controlling_series=controlling,
        raw_ceiling_length=raw,
        starting_length=starting,
        lower_bound=lower,
        upper_bound=upper,
        reasons=tuple(reasons),
    )


def combine_macro_diagnostic_lengths(
    estimates: Mapping[str, BlockLengthEstimate | float],
    *,
    observations: int,
) -> MacroWideLength:
    """Apply the binding macro lower bound and both hard upper bounds."""

    if isinstance(observations, bool) or not isinstance(observations, int | np.integer):
        raise ModelError("macro observations must be an integer")
    observations = int(observations)
    if observations < 20:
        raise PreliminaryPPWInsufficientSupport(
            "macro block-length combination requires at least 20 observations",
            details={"observations": observations},
        )
    values = _validated_length_values(estimates, expected=MACRO_DIAGNOSTIC_NAMES)
    maximum = max(value for _, value in values)
    raw = max(2, math.ceil(maximum))
    absolute_upper = 24
    sample_upper = math.floor(observations / 8)
    if raw > absolute_upper or raw > sample_upper:
        raise PreliminaryBlockLengthCapFailure(
            "macro block length exceeds a binding hard cap",
            details={
                "raw_ceiling_length": raw,
                "absolute_upper_bound": absolute_upper,
                "sample_support_upper_bound": sample_upper,
                "observations": observations,
            },
        )
    return MacroWideLength(
        diagnostic_lengths=values,
        controlling_series=tuple(
            name
            for name, value in values
            if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-15)
        ),
        raw_ceiling_length=raw,
        starting_length=raw,
        lower_bound=2,
        absolute_upper_bound=absolute_upper,
        sample_support_upper_bound=sample_upper,
    )


def state_support(
    states: npt.ArrayLike,
    continuation: npt.ArrayLike,
    *,
    intended_length: int,
    selected_states: Sequence[int] | None = None,
) -> tuple[StateSupport, ...]:
    """Count completed episodes and same-state continuation-eligible starts.

    ``continuation[i]`` is true only when source row ``i+1`` may follow row
    ``i`` without crossing a source gap.  A completed episode is bounded on
    both sides by observed, continuation-valid transitions; sample/gap-censored
    edge episodes are not counted as completed.
    """

    labels, links, selected = _state_inputs(states, continuation, selected_states)
    if intended_length < 2:
        raise ModelError(
            "support diagnostics require an intended length of at least two",
            details={"intended_length": intended_length},
        )
    completed = _completed_episode_counts(labels, links, selected)
    result: list[StateSupport] = []
    stop = labels.size - intended_length + 1
    for state in selected:
        starts: list[int] = []
        for start in range(max(0, stop)):
            end = start + intended_length
            if bool(np.all(labels[start:end] == state)) and bool(
                np.all(links[start : end - 1])
            ):
                starts.append(start)
        result.append(
            StateSupport(
                state=state,
                intended_length=intended_length,
                completed_episodes=completed[state],
                eligible_start_indices=tuple(starts),
            )
        )
    return tuple(result)


def select_supported_length(
    *,
    starting_length: int,
    frequency: Frequency,
    states: npt.ArrayLike,
    continuation: npt.ArrayLike,
    realized_means_by_length: Mapping[int, Mapping[int, float]],
    selected_states: Sequence[int] | None = None,
    minimum_completed_episodes: int = 0,
    minimum_effective_fraction: float = 0.5,
) -> BlockPolicySelection:
    """Search downward and return the largest support/effectiveness-feasible L."""

    lower, upper = _frequency_bounds(frequency)
    if not lower <= starting_length <= upper:
        raise ModelError(
            "starting block length is outside the approved frequency bounds",
            details={
                "frequency": frequency,
                "starting_length": starting_length,
                "bounds": [lower, upper],
            },
        )
    if minimum_completed_episodes < 0 or not 0.0 < minimum_effective_fraction <= 1.0:
        raise ModelError("block support thresholds are invalid")
    labels, links, selected = _state_inputs(states, continuation, selected_states)
    trials: list[LengthTrial] = []
    for intended in range(starting_length, lower - 1, -1):
        support = state_support(
            labels,
            links,
            intended_length=intended,
            selected_states=selected,
        )
        required = max(8, intended) if frequency == "monthly" else max(50, 2 * intended)
        means_input = realized_means_by_length.get(intended)
        reasons: list[str] = []
        means: list[tuple[int, float]] = []
        for item in support:
            if item.completed_episodes < minimum_completed_episodes:
                reasons.append(f"state_{item.state}:completed_episodes")
            if item.eligible_start_count < required:
                reasons.append(f"state_{item.state}:eligible_starts")
            if means_input is None or item.state not in means_input:
                reasons.append(f"state_{item.state}:missing_realized_mean")
                continue
            mean = float(means_input[item.state])
            if not math.isfinite(mean) or mean <= 0.0:
                raise BlockSupportNumericalFailure(
                    "realized effective block length is invalid",
                    details={
                        "intended_length": intended,
                        "state": item.state,
                        "realized_mean": mean,
                    },
                )
            means.append((item.state, mean))
            if mean < minimum_effective_fraction * intended:
                reasons.append(f"state_{item.state}:effective_mean_below_fraction")
        trial = LengthTrial(
            intended_length=intended,
            required_eligible_starts=required,
            support=support,
            realized_means=tuple(means),
            reasons=tuple(reasons),
        )
        trials.append(trial)
        if trial.feasible:
            return BlockPolicySelection(
                frequency=frequency,
                selected_length=intended,
                restart_probability=1.0 / intended,
                trials=tuple(trials),
            )
    raise NoSupportedBlockLength(
        "no block length satisfies state support and realized-length rules",
        details={
            "frequency": frequency,
            "starting_length": starting_length,
            "trials": [
                {"length": trial.intended_length, "reasons": list(trial.reasons)}
                for trial in trials
            ],
        },
    )


def realized_block_statistics(
    target_states: npt.ArrayLike,
    restart_flags: npt.ArrayLike,
    *,
    source_indices: npt.ArrayLike | None = None,
    source_continuation_links: npt.ArrayLike | None = None,
    target_continuation_links: npt.ArrayLike | None = None,
    ideal: bool = False,
) -> tuple[RealizedBlockStats, ...]:
    """Summarize binding realized conditioned or paired-ideal blocks.

    Conditioned mode requires source indices and source continuation links.
    A restart selecting the immediately following valid source row does not
    split a conditioned block.  Ideal mode has no source provenance and splits
    exactly at restart flags (which must already encode first row, target-state
    changes, target-segment starts, and Bernoulli restarts).
    """

    if not isinstance(ideal, bool):
        raise ModelError("ideal mode flag must be Boolean")
    states = _as_state_vector(target_states)
    restarts_raw = np.asarray(restart_flags)
    if (
        restarts_raw.ndim != 1
        or restarts_raw.size != states.size
        or restarts_raw.dtype.kind != "b"
    ):
        raise ModelError(
            "restart flags must be a Boolean vector matching target states",
            details={"states": states.size, "restart_flags": restarts_raw.size},
        )
    restarts = np.asarray(restarts_raw, dtype=np.bool_)
    if not bool(restarts[0]):
        raise ModelError("the first realized observation must be a restart")
    transitions = states[1:] != states[:-1]
    if bool(np.any(transitions & ~restarts[1:])):
        raise ModelError("every target-state transition must force a restart")

    target_links = _target_links_for_trace(
        target_continuation_links, observations=states.size
    )
    if bool(np.any(~target_links & ~restarts[1:])):
        raise ModelError("every target-segment start must force a restart")
    boundaries = np.array(restarts, dtype=np.bool_, copy=True)
    if ideal:
        if source_indices is not None or source_continuation_links is not None:
            raise ModelError("ideal realized blocks cannot accept source provenance")
    else:
        if source_indices is None or source_continuation_links is None:
            raise ModelError(
                "conditioned realized blocks require source indices and continuation"
            )
        sources = _source_index_vector(source_indices, expected_size=states.size)
        source_links = _source_links_for_trace(source_continuation_links, sources)
        boundaries[1:] = transitions | ~target_links
        for index in range(1, states.size):
            previous = int(sources[index - 1])
            adjacent = int(sources[index]) == previous + 1
            valid_link = previous < source_links.size and bool(source_links[previous])
            if not adjacent or not valid_link:
                if not bool(restarts[index]):
                    raise ModelError(
                        "every conditioned source discontinuity must be a restart",
                        details={"position": index},
                    )
                boundaries[index] = True

    by_state: defaultdict[int, list[int]] = defaultdict(list)
    block_start = 0
    for index in range(1, states.size):
        if bool(boundaries[index]):
            by_state[int(states[block_start])].append(index - block_start)
            block_start = index
    by_state[int(states[block_start])].append(states.size - block_start)
    result: list[RealizedBlockStats] = []
    for state, lengths in sorted(by_state.items()):
        values = np.asarray(lengths, dtype=np.int64)
        result.append(
            RealizedBlockStats(
                state=state,
                lengths=tuple(lengths),
                mean=float(np.mean(values)),
                minimum=int(np.min(values)),
                maximum=int(np.max(values)),
                median=float(np.quantile(values, 0.50, method="higher")),
                singleton_fraction=float(np.mean(values == 1)),
                quantile_05=float(np.quantile(values, 0.05, method="higher")),
                quantile_95=float(np.quantile(values, 0.95, method="higher")),
                count=len(lengths),
            )
        )
    return tuple(result)


def paired_fragmentation_gates(
    conditioned_statistics: Sequence[RealizedBlockStats],
    ideal_statistics: Sequence[RealizedBlockStats],
    *,
    intended_length: int,
) -> tuple[PairedFragmentationGate, ...]:
    """Apply all four frozen per-state conditioned/ideal fragmentation gates."""

    if isinstance(intended_length, bool) or not isinstance(
        intended_length, int | np.integer
    ):
        raise ModelError("intended block length must be an integer")
    intended_length = int(intended_length)
    if intended_length < 2:
        raise ModelError(
            "fragmentation gates require an intended length of at least two"
        )
    conditioned = _statistics_by_state(
        conditioned_statistics, label="conditioned statistics"
    )
    ideal = _statistics_by_state(ideal_statistics, label="ideal statistics")
    if set(conditioned) != set(ideal):
        raise ModelError(
            "conditioned and ideal statistics must contain the same states",
            details={
                "conditioned_states": sorted(conditioned),
                "ideal_states": sorted(ideal),
            },
        )

    result: list[PairedFragmentationGate] = []
    for state in sorted(conditioned):
        observed = conditioned[state]
        counterfactual = ideal[state]
        mean_minimum = 0.50 * intended_length
        median_minimum = 0.50 * counterfactual.median
        relative_singleton_maximum = counterfactual.singleton_fraction + 0.10
        absolute_singleton_maximum = 0.75
        reasons: list[str] = []
        if observed.mean < mean_minimum:
            reasons.append("conditioned_mean_below_half_intended_length")
        if observed.median < median_minimum:
            reasons.append("conditioned_median_below_half_ideal_median")
        if observed.singleton_fraction > relative_singleton_maximum:
            reasons.append("conditioned_singleton_exceeds_ideal_plus_0.10")
        if observed.singleton_fraction > absolute_singleton_maximum:
            reasons.append("conditioned_singleton_exceeds_0.75")
        result.append(
            PairedFragmentationGate(
                state=state,
                intended_length=intended_length,
                conditioned=observed,
                ideal=counterfactual,
                conditioned_mean_minimum=mean_minimum,
                conditioned_median_minimum=median_minimum,
                singleton_fraction_maximum_from_ideal=relative_singleton_maximum,
                absolute_singleton_fraction_maximum=absolute_singleton_maximum,
                reasons=tuple(reasons),
            )
        )
    return tuple(result)


def realized_mean_mapping(
    statistics: Sequence[RealizedBlockStats],
) -> dict[int, float]:
    """Return the state-to-mean form consumed by ``select_supported_length``."""

    result: dict[int, float] = {}
    for item in statistics:
        if item.state in result:
            raise ModelError(
                "realized block statistics contain a duplicate state",
                details={"state": item.state},
            )
        if not item.lengths or not math.isfinite(item.mean) or item.mean <= 0.0:
            raise ModelError(
                "realized block statistics are invalid",
                details={"state": item.state},
            )
        result[item.state] = item.mean
    if not result:
        raise ModelError("realized block statistics are empty")
    return result


def _correlation_pc1(
    values: FloatArray,
    *,
    name: str,
    sign_anchor: int | None,
    anchor_name: str | None,
) -> tuple[FloatArray, FloatArray]:
    """Return identified correlation-PC1 scores and deterministic loadings."""

    centered = values - np.mean(values, axis=0)
    scales = np.std(centered, axis=0, ddof=1)
    if not bool(np.all(np.isfinite(scales))) or bool(np.any(scales <= 0.0)):
        raise PreliminaryPPWNumericalFailure(
            "PC1 input contains a constant or invalid column",
            details={"diagnostic": name, "sample_standard_deviations": scales.tolist()},
        )
    standardized = centered / scales
    correlation = standardized.T @ standardized / (values.shape[0] - 1)
    correlation = 0.5 * (correlation + correlation.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    except np.linalg.LinAlgError as error:
        raise PreliminaryPPWNumericalFailure(
            "correlation-PC1 eigendecomposition failed",
            details={"diagnostic": name, "error_type": type(error).__name__},
        ) from error
    largest = float(eigenvalues[-1])
    second = float(eigenvalues[-2])
    eigengap = largest - second
    eigengap_floor = 1e-10 * max(1.0, abs(largest))
    if not math.isfinite(eigengap) or eigengap <= eigengap_floor:
        raise PreliminaryPPWNumericalFailure(
            "correlation-PC1 top eigenvalue is not identified",
            details={
                "diagnostic": name,
                "largest_eigenvalue": largest,
                "second_eigenvalue": second,
                "eigengap": eigengap,
                "required_strictly_above": eigengap_floor,
            },
        )
    loadings = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    if sign_anchor is None:
        anchor = int(np.argmax(np.abs(loadings)))
        effective_anchor_name = f"column_{anchor}"
    else:
        anchor = sign_anchor
        effective_anchor_name = anchor_name or f"column_{anchor}"
    anchor_loading = float(loadings[anchor])
    if abs(anchor_loading) <= 1e-12:
        raise PreliminaryPPWNumericalFailure(
            "correlation-PC1 sign anchor loading is not identified",
            details={
                "diagnostic": name,
                "anchor": effective_anchor_name,
                "loading": anchor_loading,
            },
        )
    if anchor_loading < 0.0:
        loadings = -loadings
    scores = standardized @ loadings
    if not bool(np.all(np.isfinite(scores))):
        raise PreliminaryPPWNumericalFailure(
            "correlation-PC1 scores are non-finite", details={"diagnostic": name}
        )
    return np.asarray(scores, dtype=np.float64), loadings


def _selector_continuation_links(
    continuation: npt.ArrayLike | None, *, nobs: int
) -> BoolArray:
    if continuation is None:
        return np.ones(max(0, nobs - 1), dtype=np.bool_)
    raw = np.asarray(continuation)
    if raw.ndim != 1 or raw.dtype.kind != "b":
        raise ModelError("PPW continuation must be a one-dimensional Boolean vector")
    if raw.size == nobs - 1:
        return np.asarray(raw, dtype=np.bool_)
    if raw.size == nobs:
        if bool(raw[0]):
            raise ModelError("row continuation must be false at the first observation")
        return np.asarray(raw[1:], dtype=np.bool_)
    raise ModelError(
        "PPW continuation must have length n-1 or n",
        details={"observations": nobs, "continuation": int(raw.size)},
    )


def _segment_geometry(links: BoolArray, *, nobs: int) -> tuple[IntArray, int]:
    segment_ids = np.zeros(nobs, dtype=np.int64)
    if nobs > 1:
        segment_ids[1:] = np.cumsum(~links, dtype=np.int64)
    counts = np.bincount(segment_ids)
    return segment_ids, int(np.max(counts))


def _validated_length_values(
    estimates: Mapping[str, BlockLengthEstimate | float],
    *,
    expected: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    supplied = set(estimates)
    expected_set = set(expected)
    if supplied != expected_set:
        raise ModelError(
            "block-length combination requires exactly the registered diagnostics",
            details={
                "missing": sorted(expected_set - supplied),
                "unexpected": sorted(supplied - expected_set),
            },
        )
    values: list[tuple[str, float]] = []
    for name in expected:
        item = estimates[name]
        value = (
            item.stationary_length if isinstance(item, BlockLengthEstimate) else item
        )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ModelError(
                "diagnostic block length is non-finite or non-positive",
                details={"series": name, "estimate": numeric},
            )
        values.append((name, numeric))
    return tuple(values)


def _source_index_vector(values: npt.ArrayLike, *, expected_size: int) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != expected_size or raw.dtype.kind not in {"i", "u"}:
        raise ModelError(
            "source indices must be an integer vector matching target states",
            details={"expected": expected_size, "actual": int(raw.size)},
        )
    result = np.asarray(raw, dtype=np.int64)
    if bool(np.any(result < 0)):
        raise ModelError("source indices must be non-negative")
    return result


def _source_links_for_trace(
    values: npt.ArrayLike, source_indices: IntArray
) -> BoolArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind != "b":
        raise ModelError(
            "source continuation links must be a one-dimensional Boolean vector"
        )
    links = np.asarray(raw, dtype=np.bool_)
    if bool(np.any(source_indices > links.size)):
        raise ModelError(
            "source index is outside the history represented by continuation links",
            details={
                "maximum_source_index": int(np.max(source_indices)),
                "source_observations": int(links.size + 1),
            },
        )
    return links


def _target_links_for_trace(
    values: npt.ArrayLike | None, *, observations: int
) -> BoolArray:
    if values is None:
        return np.ones(max(0, observations - 1), dtype=np.bool_)
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != max(0, observations - 1) or raw.dtype.kind != "b":
        raise ModelError(
            "target continuation links must be a Boolean vector of length n-1",
            details={"observations": observations, "actual": int(raw.size)},
        )
    return np.asarray(raw, dtype=np.bool_)


def _statistics_by_state(
    statistics: Sequence[RealizedBlockStats], *, label: str
) -> dict[int, RealizedBlockStats]:
    result: dict[int, RealizedBlockStats] = {}
    for item in statistics:
        if (
            isinstance(item.state, bool)
            or not isinstance(item.state, int | np.integer)
            or item.state < 0
        ):
            raise ModelError(f"{label} contain an invalid state label")
        if item.state in result:
            raise ModelError(f"{label} contain a duplicate state")
        if any(
            isinstance(length, bool)
            or not isinstance(length, int | np.integer)
            or length < 1
            for length in item.lengths
        ):
            raise ModelError(f"{label} are invalid", details={"state": item.state})
        numeric = (
            item.mean,
            item.median,
            item.singleton_fraction,
            item.quantile_05,
            item.quantile_95,
        )
        if (
            not item.lengths
            or not all(math.isfinite(value) for value in numeric)
            or not 0.0 <= item.singleton_fraction <= 1.0
        ):
            raise ModelError(f"{label} are invalid", details={"state": item.state})
        lengths = np.asarray(item.lengths, dtype=np.int64)
        expected = (
            float(np.mean(lengths)),
            float(np.quantile(lengths, 0.50, method="higher")),
            float(np.mean(lengths == 1)),
            float(np.quantile(lengths, 0.05, method="higher")),
            float(np.quantile(lengths, 0.95, method="higher")),
        )
        if (
            item.count != len(item.lengths)
            or item.minimum != int(np.min(lengths))
            or item.maximum != int(np.max(lengths))
            or any(
                not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12)
                for actual, target in zip(numeric, expected, strict=True)
            )
        ):
            raise ModelError(
                f"{label} are inconsistent with their block lengths",
                details={"state": item.state},
            )
        result[item.state] = item
    if not result:
        raise ModelError(f"{label} are empty")
    return result


def _frequency_bounds(frequency: Frequency) -> tuple[int, int]:
    try:
        return _BOUNDS[frequency]
    except KeyError as error:
        raise ModelError(
            "frequency must be monthly or daily", details={"frequency": frequency}
        ) from error


def _completed_episode_counts(
    states: IntArray, continuation: BoolArray, selected: tuple[int, ...]
) -> dict[int, int]:
    counts = dict.fromkeys(selected, 0)
    start = 0
    for index in range(states.size):
        at_end = index == states.size - 1
        continues_run = (
            not at_end
            and bool(continuation[index])
            and states[index + 1] == states[index]
        )
        if continues_run:
            continue
        state = int(states[start])
        left_complete = (
            start > 0
            and bool(continuation[start - 1])
            and states[start - 1] != states[start]
        )
        right_complete = (
            not at_end
            and bool(continuation[index])
            and states[index + 1] != states[index]
        )
        if state in counts and left_complete and right_complete:
            counts[state] += 1
        start = index + 1
    return counts


def _state_inputs(
    states: npt.ArrayLike,
    continuation: npt.ArrayLike,
    selected_states: Sequence[int] | None,
) -> tuple[IntArray, BoolArray, tuple[int, ...]]:
    labels = _as_state_vector(states)
    links_raw = np.asarray(continuation)
    if (
        links_raw.ndim != 1
        or links_raw.size != max(0, labels.size - 1)
        or links_raw.dtype.kind != "b"
    ):
        raise ModelError(
            "continuation must be a Boolean vector of length n-1",
            details={"states": labels.size, "continuation": links_raw.size},
        )
    links = np.asarray(links_raw, dtype=np.bool_)
    available = {int(value) for value in np.unique(labels)}
    if selected_states is None:
        selected = tuple(sorted(available))
    else:
        selected = tuple(int(value) for value in selected_states)
        if len(set(selected)) != len(selected) or not selected:
            raise ModelError("selected states must be non-empty and unique")
        missing = set(selected) - available
        if missing:
            raise ModelError(
                "selected state is absent from the history",
                details={"missing_states": sorted(missing)},
            )
    return labels, links, selected


def _as_state_vector(values: npt.ArrayLike) -> IntArray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ModelError("states must be a non-empty one-dimensional integer vector")
    labels = np.asarray(array, dtype=np.int64)
    if bool(np.any(labels < 0)):
        raise ModelError("state labels must be non-negative integers")
    return labels


def _as_float_matrix(values: npt.ArrayLike, *, columns: int, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(
            f"{name} cannot be converted to a numeric array",
            details={"error_type": type(error).__name__},
        ) from error
    if array.ndim != 2 or array.shape[1] != columns:
        raise ModelError(
            f"{name} must have shape (n, {columns})",
            details={"shape": list(array.shape)},
        )
    if not bool(np.all(np.isfinite(array))):
        raise ModelError(f"{name} contains NaN or infinity")
    return np.array(array, dtype=np.float64, copy=True)


def _as_float_vector(values: npt.ArrayLike, *, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError(
            f"{name} cannot be converted to a numeric vector",
            details={"error_type": type(error).__name__},
        ) from error
    if array.ndim != 1 or array.size == 0:
        raise ModelError(
            f"{name} must be a non-empty one-dimensional vector",
            details={"shape": list(array.shape)},
        )
    if not bool(np.all(np.isfinite(array))):
        raise ModelError(f"{name} contains NaN or infinity")
    return np.array(array, dtype=np.float64, copy=True)


def _standardize(values: FloatArray, *, name: str) -> FloatArray:
    centered = values - float(np.mean(values))
    scale = float(np.sqrt(np.mean(np.square(centered))))
    numerical_floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values))))
    if not math.isfinite(scale) or scale <= numerical_floor:
        raise PreliminaryPPWNumericalFailure(
            "diagnostic series is constant or numerically constant",
            details={"series": name, "scale": scale},
        )
    result = centered / scale
    if not bool(np.all(np.isfinite(result))):
        raise PreliminaryPPWNumericalFailure(
            "standardized diagnostic series is non-finite", details={"series": name}
        )
    return np.asarray(result, dtype=np.float64)


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
