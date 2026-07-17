"""Strict, immutable configuration and canonical hashing.

Loading a configuration performs no network calls and writes no files.
Unknown keys and invalid cross-field combinations fail closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from prpg.errors import ConfigurationError

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
UnitFloat = Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
StrictFloat = Annotated[float, Field(strict=True)]
Ticker = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,14}$")]
SeriesId = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,31}$")]


class StrictModel(BaseModel):
    """Frozen base model shared by every configuration section."""

    # Container coercion is intentionally permitted so ordinary YAML lists and
    # path strings resolve to immutable tuples/Paths. Scientific scalar fields
    # carry explicit ``strict=True`` constraints below, so e.g. ``"5000"`` is
    # never accepted as a path count.
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
    master_seed: Annotated[int, Field(strict=True, ge=0, le=(1 << 128) - 1)]
    output_root: Path


class AssetConfig(StrictModel):
    equity: Ticker
    muni_bond: Ticker
    taxable_bond: Ticker

    @model_validator(mode="after")
    def roles_are_distinct(self) -> AssetConfig:
        if len({self.equity, self.muni_bond, self.taxable_bond}) != 3:
            raise ValueError("the three semantic asset roles require distinct tickers")
        return self


class MacroConfig(StrictModel):
    industrial_production: SeriesId
    inflation: SeriesId
    yield_curve_spread: SeriesId
    credit_spread: SeriesId

    @model_validator(mode="after")
    def series_are_distinct(self) -> MacroConfig:
        values = (
            self.industrial_production,
            self.inflation,
            self.yield_curve_spread,
            self.credit_spread,
        )
        if len(set(values)) != len(values):
            raise ValueError("macro feature roles require distinct FRED series")
        return self


class YFinanceConfig(StrictModel):
    start: date
    end_exclusive: date
    interval: Literal["1d"]
    auto_adjust: Literal[False]
    actions: Literal[True]
    repair: Literal[False]
    keepna: Literal[True]
    prepost: Literal[False]
    threads: Literal[False]
    timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=120)]
    timezone: Literal["America/New_York"]
    max_rounds: Annotated[int, Field(strict=True, ge=1, le=3)]
    library_retries_per_round: Annotated[int, Field(strict=True, ge=0, le=2)]
    ticker_wall_clock_seconds: Annotated[int, Field(strict=True, ge=1, le=600)]
    http_429_min_cooldown_seconds: Annotated[int, Field(strict=True, ge=60)]

    @model_validator(mode="after")
    def end_follows_start(self) -> YFinanceConfig:
        if self.end_exclusive <= self.start:
            raise ValueError("yfinance end_exclusive must be after start")
        return self


class FredConfig(StrictModel):
    transport: Literal["official_csv"]
    max_attempts: Literal[5]
    backoff_caps_seconds: tuple[Literal[2], Literal[5], Literal[15], Literal[45]]
    daily_month_coverage_fraction: Annotated[float, Field(strict=True, ge=0.80, le=1.0)]


class DataQualityConfig(StrictModel):
    equity_abs_simple_return_flag: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    bond_abs_simple_return_flag: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    adjustment_factor_move_flag: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    minimum_common_monthly_vectors: PositiveInt
    minimum_common_daily_vectors: PositiveInt


class DataConfig(StrictModel):
    snapshot_mode: Literal["immutable"]
    artifact_root: Path
    cutoff: date
    macro_vintage_policy: Literal["current_vintage_retrospective"]
    assets: AssetConfig
    macro: MacroConfig
    common_history_only: Literal[True]
    forbid_price_forward_fill: Literal[True]
    yfinance: YFinanceConfig
    fred: FredConfig
    quality: DataQualityConfig

    @model_validator(mode="after")
    def request_covers_cutoff(self) -> DataConfig:
        if self.yfinance.end_exclusive <= self.cutoff:
            raise ValueError("exclusive provider end must be later than data cutoff")
        if (self.yfinance.end_exclusive - self.cutoff).days != 1:
            raise ValueError("canonical provider end must be cutoff plus one day")
        return self


class ModelConfig(StrictModel):
    regime_candidates: tuple[PositiveInt, ...]
    regime_selection_policy: Literal["owner_fixed_k4_v3"]
    fixed_regime_count: Literal[4]
    covariance_type: Literal["tied"]
    deterministic_restarts: PositiveInt
    alpha_policy_candidates: tuple[
        Literal["historical_vector", "kernel_mean_neutral"], ...
    ]
    # G3 registers the candidate set but does not choose a generation policy.
    # The sealed G5 authority alone supplies both selected values.
    selected_alpha_policy: None
    selected_neutral_lambda: None
    neutral_lambda_grid: tuple[float, ...]
    initial_state: Literal["stationary"]
    design_holdout_months: PositiveInt
    production_refit: Literal["full_data_mandatory"]

    @model_validator(mode="after")
    def validate_scientific_choices(self) -> ModelConfig:
        if tuple(self.regime_candidates) != (2, 3, 4, 5):
            raise ValueError("diagnostic regime_candidates must be [2, 3, 4, 5]")
        if self.regime_selection_policy != "owner_fixed_k4_v3":
            raise ValueError("regime selection policy must be owner_fixed_k4_v3")
        if self.fixed_regime_count != 4:
            raise ValueError("owner-approved fixed regime count must be 4")
        expected_policies = ("historical_vector", "kernel_mean_neutral")
        if tuple(self.alpha_policy_candidates) != expected_policies:
            raise ValueError("version 1 alpha policy candidates/order is frozen")
        expected_grid = (0.25, 0.50, 0.75, 1.00)
        if tuple(self.neutral_lambda_grid) != expected_grid:
            raise ValueError("version 1 neutral lambda grid is frozen")
        return self


class FrequencyBootstrapConfig(StrictModel):
    block_selector: Literal["politis_white_multivariate"]


class BootstrapConfig(StrictModel):
    monthly: FrequencyBootstrapConfig
    daily: FrequencyBootstrapConfig


class LowFrequencyConfig(StrictModel):
    paths: Annotated[int, Field(strict=True, gt=0, le=999_999)]
    years: PositiveInt
    base_frequency: Literal["monthly"]


class HighFrequencyConfig(StrictModel):
    paths: Annotated[int, Field(strict=True, gt=0, le=999_999)]
    years: PositiveInt
    base_frequency: Literal["daily"]


class SimulationConfig(StrictModel):
    calendar: Literal["SC252-52-v1"]
    scientific_version: Annotated[int, Field(strict=True, ge=0, le=(1 << 32) - 1)]
    low_frequency: LowFrequencyConfig
    high_frequency: HighFrequencyConfig

    @model_validator(mode="after")
    def horizons_match(self) -> SimulationConfig:
        if self.low_frequency.years != self.high_frequency.years:
            raise ValueError("version 1 LF and HF horizons must match")
        return self


class ExportConfig(StrictModel):
    format: Literal["csv"]
    return_representation: Literal["log_total_decimal"]
    low_paths_per_shard: PositiveInt
    high_paths_per_shard: PositiveInt


class ExecutionConfig(StrictModel):
    workers: Literal["auto"] | PositiveInt
    reserve_physical_cores: NonNegativeInt
    memory_fraction_limit: Annotated[float, Field(strict=True, gt=0.0, le=0.70)]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class KernelMeanCalibrationConfig(StrictModel):
    prefix_window_counts: tuple[
        Literal[10_000],
        Literal[20_000],
        Literal[40_000],
        Literal[80_000],
        Literal[160_000],
        Literal[320_000],
    ]
    annualized_simultaneous_half_width_max: StrictFloat
    confidence: StrictFloat
    cluster_unit: Literal["window"]
    multiplicity: Literal["bonferroni_all_cells_and_six_looks"]

    @model_validator(mode="after")
    def values_are_frozen(self) -> KernelMeanCalibrationConfig:
        if self.annualized_simultaneous_half_width_max != 0.001:
            raise ValueError("kernel mean half-width maximum is frozen at 0.001")
        if self.confidence != 0.95:
            raise ValueError("kernel mean confidence is frozen at 0.95")
        return self


class RawDistortionConfig(StrictModel):
    comparison_rule: Literal["smaller_of_raw_vs_raw_95pct_critical_and_practical_cap"]
    mean_shift_over_raw_volatility: StrictFloat
    absolute_annualized_mean_shift: StrictFloat
    relative_volatility_change: StrictFloat
    w1_over_raw_sigma: StrictFloat
    eligible_quantile_shift_over_raw_sigma: StrictFloat
    normalized_covariance_frobenius_distance: StrictFloat
    maximum_absolute_pairwise_correlation_change: StrictFloat
    standardized_sliced_wasserstein_distance: StrictFloat
    standardized_energy_distance: StrictFloat
    eligible_tail_downside_coexceedance_probability_change: StrictFloat
    taxable_minus_muni_mean_shift_over_spread_volatility: StrictFloat
    taxable_minus_muni_absolute_annualized_mean_shift: StrictFloat
    spread_relative_volatility_change: StrictFloat
    spread_w1_over_spread_sigma: StrictFloat
    eligible_spread_quantile_shift_over_spread_sigma: StrictFloat
    daily_quantiles: tuple[
        StrictFloat, StrictFloat, StrictFloat, StrictFloat, StrictFloat
    ]
    monthly_quantiles: tuple[StrictFloat, StrictFloat, StrictFloat]

    @model_validator(mode="after")
    def practical_caps_are_frozen(self) -> RawDistortionConfig:
        expected = {
            "mean_shift_over_raw_volatility": 0.05,
            "absolute_annualized_mean_shift": 0.01,
            "relative_volatility_change": 0.05,
            "w1_over_raw_sigma": 0.10,
            "eligible_quantile_shift_over_raw_sigma": 0.15,
            "normalized_covariance_frobenius_distance": 0.10,
            "maximum_absolute_pairwise_correlation_change": 0.05,
            "standardized_sliced_wasserstein_distance": 0.10,
            "standardized_energy_distance": 0.10,
            "eligible_tail_downside_coexceedance_probability_change": 0.02,
            "taxable_minus_muni_mean_shift_over_spread_volatility": 0.05,
            "taxable_minus_muni_absolute_annualized_mean_shift": 0.005,
            "spread_relative_volatility_change": 0.05,
            "spread_w1_over_spread_sigma": 0.10,
            "eligible_spread_quantile_shift_over_spread_sigma": 0.15,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"version 1 raw-distortion cap {field_name} is frozen at "
                    f"{expected_value}"
                )
        if self.daily_quantiles != (0.01, 0.05, 0.50, 0.95, 0.99):
            raise ValueError("version 1 daily distortion quantiles are frozen")
        if self.monthly_quantiles != (0.10, 0.50, 0.90):
            raise ValueError("version 1 monthly distortion quantiles are frozen")
        return self


class VolatilityTimingConfig(StrictModel):
    gap_definition: Literal[
        "simulated_improvement_minus_historical_control_improvement"
    ]
    simultaneous_familywise_one_sided_confidence: StrictFloat
    annualized_sharpe_upper_bound_exclusive: StrictFloat

    @model_validator(mode="after")
    def values_are_frozen(self) -> VolatilityTimingConfig:
        if self.simultaneous_familywise_one_sided_confidence != 0.95:
            raise ValueError("volatility-timing confidence is frozen at 0.95")
        if self.annualized_sharpe_upper_bound_exclusive != 0.20:
            raise ValueError("volatility-timing Sharpe upper bound is frozen at 0.20")
        return self


class ValidationConfig(StrictModel):
    profile: Literal["smoke", "release"]
    directional_alpha_margin: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    familywise_alpha: Annotated[float, Field(strict=True, gt=0.0, lt=1.0)]
    tost_confidence: Annotated[float, Field(strict=True, gt=0.0, lt=1.0)]
    csv_log_sum_abs_tolerance: Annotated[float, Field(strict=True, gt=0.0)]
    simple_return_identity_abs_tolerance: Annotated[float, Field(strict=True, gt=0.0)]
    kernel_mean_calibration: KernelMeanCalibrationConfig
    raw_distortion: RawDistortionConfig
    volatility_timing: VolatilityTimingConfig


class PRPGConfig(StrictModel):
    """Complete version-1 study/run configuration."""

    schema_version: Literal[1]
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    bootstrap: BootstrapConfig
    simulation: SimulationConfig
    export: ExportConfig
    execution: ExecutionConfig
    validation: ValidationConfig

    @model_validator(mode="after")
    def validate_shards_and_profile(self) -> PRPGConfig:
        if self.export.low_paths_per_shard > self.simulation.low_frequency.paths:
            raise ValueError("LF paths per shard cannot exceed LF path count")
        if self.export.high_paths_per_shard > self.simulation.high_frequency.paths:
            raise ValueError("HF paths per shard cannot exceed HF path count")
        canonical_counts = (
            self.simulation.low_frequency.paths == 5_000
            and self.simulation.high_frequency.paths == 1_000
            and self.simulation.low_frequency.years == 50
        )
        if self.validation.profile == "release" and not canonical_counts:
            raise ValueError("release profile requires canonical 5000/1000/50 counts")
        return self

    def canonical_json(self) -> str:
        """Serialize the fully resolved model with stable ordering and spacing."""

        payload = self.model_dump(mode="json", exclude_none=False)
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def fingerprint(self) -> str:
        """Return the lowercase SHA-256 hash of :meth:`canonical_json`."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


def redact_mapping(value: Any) -> Any:
    """Recursively redact values whose key names may identify secrets."""

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if any(marker in normalized for marker in _SECRET_MARKERS):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_mapping(item)
        return redacted
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_mapping(item) for item in value)
    return value


def _validation_details(error: ValidationError) -> dict[str, Any]:
    """Return useful Pydantic diagnostics without echoing input values."""

    return {
        "errors": [
            {
                "location": [str(part) for part in item["loc"]],
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors(include_url=False, include_input=False)
        ]
    }


def parse_config(data: Mapping[str, Any]) -> PRPGConfig:
    """Validate an already-decoded mapping as a version-1 configuration."""

    try:
        return PRPGConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigurationError(
            "configuration validation failed", details=_validation_details(error)
        ) from error


def load_config(path: str | Path) -> PRPGConfig:
    """Safely load and validate one YAML configuration from ``path``."""

    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(
            "configuration file could not be read",
            details={"path": str(config_path), "reason": type(error).__name__},
        ) from error
    try:
        decoded = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            "configuration YAML is malformed", details={"path": str(config_path)}
        ) from error
    if not isinstance(decoded, Mapping):
        raise ConfigurationError(
            "configuration root must be a mapping",
            details={"path": str(config_path)},
        )
    return parse_config(decoded)


def config_json_schema() -> dict[str, Any]:
    """Return the machine-readable version-1 configuration JSON Schema."""

    return PRPGConfig.model_json_schema()
