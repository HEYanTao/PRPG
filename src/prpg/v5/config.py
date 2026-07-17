"""Strict prospective configuration for PRPG delivery version 5.

The legacy configuration remains unchanged for the immutable version-4
delivery.  This module contains only the settings needed by the synchronized
calendar-month design and its future daily-authoritative generator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from prpg.config import AssetConfig, YFinanceConfig
from prpg.errors import ConfigurationError

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PathCount = Annotated[int, Field(strict=True, gt=0, le=999_999)]
StrictFloat = Annotated[float, Field(strict=True)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
ProbabilityTolerance = Annotated[
    float, Field(strict=True, gt=0.0, lt=1.0, allow_inf_nan=False)
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V5RunConfig(_StrictModel):
    name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
    master_seed: Annotated[int, Field(strict=True, ge=0, le=(1 << 128) - 1)]
    artifact_root: Path
    output_root: Path
    seed_registry: Path


class V5QualityFlagConfig(_StrictModel):
    """Diagnostic-only provider-return flags, not scientific release gates."""

    equity_abs_simple_return: Annotated[float, Field(strict=True, gt=0.0)]
    bond_abs_simple_return: Annotated[float, Field(strict=True, gt=0.0)]
    adjustment_factor_move: Annotated[float, Field(strict=True, gt=0.0)]


class V5DataConfig(_StrictModel):
    cutoff: date
    calendar: Literal["XNYS"]
    assets: AssetConfig
    common_history_only: Literal[True]
    complete_months_only: Literal[True]
    forbid_imputation: Literal[True]
    yfinance: YFinanceConfig
    quality_flags: V5QualityFlagConfig

    @model_validator(mode="after")
    def request_ends_after_cutoff(self) -> V5DataConfig:
        if (self.yfinance.end_exclusive - self.cutoff).days != 1:
            raise ValueError("provider end must be exactly cutoff plus one day")
        return self


class V5HMMConfig(_StrictModel):
    scientific_version: PositiveInt
    feature_order: tuple[
        Literal["equity"], Literal["taxable_bond"], Literal["muni_bond"]
    ]
    states: Literal[4]
    covariance_type: Literal["diag"]
    deterministic_restarts: PositiveInt
    variance_floor: PositiveFloat
    start_probability_prior: PositiveFloat
    transition_probability_prior: PositiveFloat
    means_prior: FiniteFloat
    means_weight: NonNegativeFloat
    covariance_prior: NonNegativeFloat
    covariance_weight: NonNegativeFloat
    maximum_iterations: PositiveInt
    tolerance: PositiveFloat
    standard_deviation_ddof: Annotated[int, Field(strict=True, ge=0, le=1)]
    initial_state: Literal["stationary"]
    minimum_state_months: PositiveInt
    ranking: Literal["greatest_finite_training_log_likelihood"]
    canonical_state_order: Literal["lexicographic_standardized_centroid"]
    probability_tolerance: ProbabilityTolerance
    stationary_residual_tolerance: ProbabilityTolerance

    @model_validator(mode="after")
    def fixed_k4_architecture(self) -> V5HMMConfig:
        """Keep the model family fixed while allowing numerical tuning."""

        expected: dict[str, object] = {
            "feature_order": ("equity", "taxable_bond", "muni_bond"),
            "states": 4,
            "covariance_type": "diag",
            "initial_state": "stationary",
            "ranking": "greatest_finite_training_log_likelihood",
            "canonical_state_order": "lexicographic_standardized_centroid",
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"version-5 K4 architecture {field_name} is fixed at "
                    f"{expected_value!r}"
                )
        return self


class V5SplitConfig(_StrictModel):
    training: PositiveInt
    validation: PositiveInt
    test: PositiveInt


class V5SimulationConfig(_StrictModel):
    paths: PathCount
    horizon_months: PositiveInt
    split: V5SplitConfig
    output_frequencies: tuple[
        Literal["daily"], Literal["monthly"], Literal["quarterly"], Literal["annual"]
    ]
    rl_decision_frequency: Literal["quarterly"]
    return_representation: Literal["log_total_decimal"]
    paths_per_shard: PositiveInt

    @model_validator(mode="after")
    def geometry_is_consistent(self) -> V5SimulationConfig:
        if self.horizon_months % 12:
            raise ValueError("version-5 horizon must contain complete model years")
        split_total = self.split.training + self.split.validation + self.split.test
        if split_total != self.paths:
            raise ValueError("training/validation/test counts must sum to paths")
        expected = ("daily", "monthly", "quarterly", "annual")
        if self.output_frequencies != expected:
            raise ValueError(
                "version-5 outputs must be daily, monthly, quarterly, and annual"
            )
        return self


class V5ExecutionConfig(_StrictModel):
    workers: PositiveInt
    native_threads_per_worker: PositiveInt


class V5ValidationConfig(_StrictModel):
    volatility_ratio_minimum: StrictFloat
    volatility_ratio_maximum: StrictFloat
    correlation_error_maximum_exclusive: StrictFloat
    covariance_relative_error_maximum: StrictFloat
    occupancy_absolute_floor: StrictFloat
    transition_cell_absolute_floor: StrictFloat
    csv_log_sum_absolute_tolerance: StrictFloat
    aggregate_branch_coverage_minimum_percent: StrictFloat

    @model_validator(mode="after")
    def prospective_thresholds_are_frozen(self) -> V5ValidationConfig:
        expected = {
            "volatility_ratio_minimum": 0.90,
            "volatility_ratio_maximum": 1.10,
            "correlation_error_maximum_exclusive": 0.05,
            "covariance_relative_error_maximum": 0.15,
            "occupancy_absolute_floor": 0.005,
            "transition_cell_absolute_floor": 0.01,
            "csv_log_sum_absolute_tolerance": 1e-12,
            "aggregate_branch_coverage_minimum_percent": 80.0,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"version-5 validation {field_name} is frozen at {expected_value!r}"
                )
        return self


class V5Config(_StrictModel):
    schema_version: Literal[2]
    run: V5RunConfig
    data: V5DataConfig
    model: V5HMMConfig
    simulation: V5SimulationConfig
    execution: V5ExecutionConfig
    validation: V5ValidationConfig


class V5HMMSeedConfig(_StrictModel):
    domain: Literal["model_fit"]
    family: Literal["not_applicable"]
    stage: Literal["hmm_library_seed"]
    scope: Literal["production_main_and_folds"]
    replicate: Literal[0]
    states: Literal[4]
    restart_first: Literal[0]
    restart_last: NonNegativeInt


class V5PathSeedConfig(_StrictModel):
    domain: Literal["production"]
    family: Literal["not_applicable"]
    entity: Literal["one_based_path_id"]
    stages: tuple[Literal["state_chain"], Literal["source_rank_uniforms"]]
    training_path_ids: tuple[PathCount, PathCount]
    validation_path_ids: tuple[PathCount, PathCount]
    test_path_ids: tuple[PathCount, PathCount]


class V5SeedRegistry(_StrictModel):
    schema_version: Literal[1]
    rng_contract_version: Literal[1]
    master_seed: Annotated[int, Field(strict=True, ge=0, le=(1 << 128) - 1)]
    scientific_version: PositiveInt
    hmm_fit: V5HMMSeedConfig
    path_generation: V5PathSeedConfig


@dataclass(frozen=True, slots=True)
class ResolvedV5Inputs:
    """Validated config/seed inputs plus stable identities and resolved paths."""

    config_path: Path
    config: V5Config
    config_fingerprint: str
    seed_registry_path: Path
    seed_registry: V5SeedRegistry
    seed_registry_fingerprint: str
    artifact_root: Path
    output_root: Path


def load_v5_inputs(path: str | Path) -> ResolvedV5Inputs:
    """Load and cross-check the version-5 config and seed registry offline."""

    config_path = Path(path).expanduser().resolve()
    config_mapping = _load_yaml_mapping(config_path, "version-5 config")
    try:
        config = V5Config.model_validate(config_mapping)
    except ValidationError as error:
        raise ConfigurationError(
            "version-5 configuration is invalid",
            details={"path": str(config_path), "errors": error.errors()},
        ) from error

    registry_path = (config_path.parent / config.run.seed_registry).resolve()
    registry_mapping = _load_yaml_mapping(registry_path, "version-5 seed registry")
    try:
        registry = V5SeedRegistry.model_validate(registry_mapping)
    except ValidationError as error:
        raise ConfigurationError(
            "version-5 seed registry is invalid",
            details={"path": str(registry_path), "errors": error.errors()},
        ) from error
    if registry.master_seed != config.run.master_seed:
        raise ConfigurationError("version-5 config and seed master seeds differ")
    if registry.scientific_version != config.model.scientific_version:
        raise ConfigurationError("version-5 config and seed scientific versions differ")
    if registry.hmm_fit.states != config.model.states or (
        registry.hmm_fit.restart_last - registry.hmm_fit.restart_first + 1
        != config.model.deterministic_restarts
    ):
        raise ConfigurationError("version-5 HMM seed namespace is inconsistent")
    split = config.simulation.split
    expected_path_ranges = (
        (1, split.training),
        (split.training + 1, split.training + split.validation),
        (split.training + split.validation + 1, config.simulation.paths),
    )
    actual_path_ranges = (
        registry.path_generation.training_path_ids,
        registry.path_generation.validation_path_ids,
        registry.path_generation.test_path_ids,
    )
    if actual_path_ranges != expected_path_ranges:
        raise ConfigurationError(
            "version-5 path seed ranges differ from the configured role split",
            details={
                "expected": expected_path_ranges,
                "actual": actual_path_ranges,
            },
        )

    return ResolvedV5Inputs(
        config_path=config_path,
        config=config,
        config_fingerprint=_fingerprint_model(config),
        seed_registry_path=registry_path,
        seed_registry=registry,
        seed_registry_fingerprint=_fingerprint_model(registry),
        artifact_root=(config_path.parent / config.run.artifact_root).resolve(),
        output_root=(config_path.parent / config.run.output_root).resolve(),
    )


def _load_yaml_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"{label} could not be read",
            details={"path": str(path), "error_type": type(error).__name__},
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"{label} root must be a mapping", details={"path": str(path)}
        )
    return value


def _fingerprint_model(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
