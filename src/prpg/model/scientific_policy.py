"""Typed generation policy frozen into each Phase-3 scientific artifact.

Model parameters alone are insufficient to reproduce a return path.  The
generator also needs the selected monthly/daily restart probabilities and the
approved one-shot alpha policy.  This module defines that small immutable
contract separately from report prose so a future generator never has to
interpret free-form metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, TypeAlias

from prpg.errors import IntegrityError, ModelError
from prpg.model.block_length import (
    BLOCK_POLICY_VERSION,
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.kernel_execution import KernelCalibrationExecution

AlphaPolicy: TypeAlias = Literal["historical_vector", "kernel_mean_neutral"]
PolicyError: TypeAlias = type[ModelError] | type[IntegrityError]

SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION: Final = 1
SCIENTIFIC_GENERATOR_ALGORITHM: Final = "conditioned_stationary_bootstrap_v1"
CANONICAL_NEUTRAL_LAMBDA_GRID: Final = (0.25, 0.50, 0.75, 1.00)
SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID: Final = (
    "prpg-g3-unqualified-alpha-candidate-set-v1"
)
SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class ScientificCandidatePolicySet:
    """G3's exact, deliberately unqualified alpha-policy candidate set.

    G3 calibrates and bridges both the historical vector and every registered
    kernel-neutral offset.  It does not select a policy: selection and
    qualification require G5 outputs that do not yet exist.  Consequently
    this record is descriptive model content and can never authorize
    generation by itself.
    """

    generation_authorized: ClassVar[Literal[False]] = False
    schema_id: str
    schema_version: int
    generator_algorithm: str
    block_policy_version: str
    monthly_block_length: int
    daily_block_length: int
    candidate_policies: tuple[Literal["historical_vector", "kernel_mean_neutral"], ...]
    neutral_lambda_candidates: tuple[float, float, float, float]
    selected_policy: None
    selected_neutral_lambda: None
    qualification_gate: Literal["G5"]
    initial_state: Literal["stationary"]
    synchronized_return_vectors: Literal[True]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "generator_algorithm": self.generator_algorithm,
            "block_policy_version": self.block_policy_version,
            "monthly_block_length": self.monthly_block_length,
            "daily_block_length": self.daily_block_length,
            "candidate_policies": list(self.candidate_policies),
            "neutral_lambda_candidates": list(self.neutral_lambda_candidates),
            "selected_policy": self.selected_policy,
            "selected_neutral_lambda": self.selected_neutral_lambda,
            "qualification_gate": self.qualification_gate,
            "initial_state": self.initial_state,
            "synchronized_return_vectors": self.synchronized_return_vectors,
        }


def candidate_policy_set_from_kernel_results(
    *,
    monthly: KernelCalibrationExecution,
    daily: KernelCalibrationExecution,
) -> ScientificCandidatePolicySet:
    """Derive the exact unqualified G3 candidate set from both kernel cells."""

    if not isinstance(monthly, KernelCalibrationExecution) or not isinstance(
        daily, KernelCalibrationExecution
    ):
        raise ModelError("candidate policy set requires typed kernel executions")
    if (
        monthly.frequency != "monthly"
        or daily.frequency != "daily"
        or monthly.artifact != daily.artifact
        or monthly.n_states != daily.n_states
        or monthly.model_artifact_fingerprint != daily.model_artifact_fingerprint
        or not monthly.passed
        or not daily.passed
        or monthly.offset_grid is None
        or daily.offset_grid is None
        or tuple(float(value) for value in monthly.offset_grid.lambdas)
        != CANONICAL_NEUTRAL_LAMBDA_GRID
        or tuple(float(value) for value in daily.offset_grid.lambdas)
        != CANONICAL_NEUTRAL_LAMBDA_GRID
    ):
        raise ModelError("candidate policy set kernel pair is inconsistent")
    return _validated_candidate_policy_set(
        ScientificCandidatePolicySet(
            schema_id=SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID,
            schema_version=SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION,
            generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
            block_policy_version=BLOCK_POLICY_VERSION,
            monthly_block_length=monthly.selected_length,
            daily_block_length=daily.selected_length,
            candidate_policies=("historical_vector", "kernel_mean_neutral"),
            neutral_lambda_candidates=CANONICAL_NEUTRAL_LAMBDA_GRID,
            selected_policy=None,
            selected_neutral_lambda=None,
            qualification_gate="G5",
            initial_state="stationary",
            synchronized_return_vectors=True,
        ),
        ModelError,
    )


def candidate_policy_set_from_mapping(
    value: object,
    *,
    error_type: PolicyError = IntegrityError,
) -> ScientificCandidatePolicySet:
    """Parse and validate the exact unqualified G3 policy-set schema."""

    if not isinstance(value, Mapping):
        raise error_type("scientific candidate policy set is missing")
    expected = {
        "schema_id",
        "schema_version",
        "generator_algorithm",
        "block_policy_version",
        "monthly_block_length",
        "daily_block_length",
        "candidate_policies",
        "neutral_lambda_candidates",
        "selected_policy",
        "selected_neutral_lambda",
        "qualification_gate",
        "initial_state",
        "synchronized_return_vectors",
    }
    if set(value) != expected:
        raise error_type("scientific candidate policy set keys are invalid")
    policies = value["candidate_policies"]
    lambdas = value["neutral_lambda_candidates"]
    if not isinstance(policies, list | tuple) or not isinstance(lambdas, list | tuple):
        raise error_type("scientific candidate policy set values are invalid")
    try:
        result = ScientificCandidatePolicySet(
            schema_id=value["schema_id"],
            schema_version=value["schema_version"],
            generator_algorithm=value["generator_algorithm"],
            block_policy_version=value["block_policy_version"],
            monthly_block_length=value["monthly_block_length"],
            daily_block_length=value["daily_block_length"],
            candidate_policies=tuple(policies),
            neutral_lambda_candidates=tuple(lambdas),
            selected_policy=value["selected_policy"],
            selected_neutral_lambda=value["selected_neutral_lambda"],
            qualification_gate=value["qualification_gate"],
            initial_state=value["initial_state"],
            synchronized_return_vectors=value["synchronized_return_vectors"],
        )
    except (TypeError, ValueError) as error:
        raise error_type(
            "scientific candidate policy set values are invalid"
        ) from error
    return _validated_candidate_policy_set(result, error_type)


def _validated_candidate_policy_set(
    value: ScientificCandidatePolicySet,
    error_type: PolicyError,
) -> ScientificCandidatePolicySet:
    if not isinstance(value, ScientificCandidatePolicySet):
        raise error_type("scientific candidate policy set has the wrong type")
    if (
        value.schema_id != SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID
        or value.schema_version != SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION
        or value.generator_algorithm != SCIENTIFIC_GENERATOR_ALGORITHM
        or value.block_policy_version != BLOCK_POLICY_VERSION
        or value.candidate_policies != ("historical_vector", "kernel_mean_neutral")
        or tuple(value.neutral_lambda_candidates) != CANONICAL_NEUTRAL_LAMBDA_GRID
        or value.selected_policy is not None
        or value.selected_neutral_lambda is not None
        or value.qualification_gate != "G5"
        or value.initial_state != "stationary"
        or value.synchronized_return_vectors is not True
    ):
        raise error_type("scientific candidate policy set is noncanonical")
    for length, (lower, upper) in (
        (value.monthly_block_length, MONTHLY_BLOCK_LENGTH_BOUNDS),
        (value.daily_block_length, DAILY_BLOCK_LENGTH_BOUNDS),
    ):
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not lower <= length <= upper
        ):
            raise error_type("scientific candidate policy block length is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ScientificGenerationPolicy:
    """All non-array choices needed to interpret one scientific artifact."""

    schema_version: int
    generator_algorithm: str
    block_policy_version: str
    monthly_block_length: int
    daily_block_length: int
    alpha_policy: AlphaPolicy
    selected_neutral_lambda: float | None
    neutral_lambda_grid: tuple[float, float, float, float]
    initial_state: Literal["stationary"]
    synchronized_return_vectors: Literal[True]

    @property
    def monthly_restart_probability(self) -> float:
        return 1.0 / self.monthly_block_length

    @property
    def daily_restart_probability(self) -> float:
        return 1.0 / self.daily_block_length

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_algorithm": self.generator_algorithm,
            "block_policy_version": self.block_policy_version,
            "monthly_block_length": self.monthly_block_length,
            "daily_block_length": self.daily_block_length,
            "alpha_policy": self.alpha_policy,
            "selected_neutral_lambda": self.selected_neutral_lambda,
            "neutral_lambda_grid": list(self.neutral_lambda_grid),
            "initial_state": self.initial_state,
            "synchronized_return_vectors": self.synchronized_return_vectors,
        }


def generation_policy_from_kernel_results(
    *,
    monthly: KernelCalibrationExecution,
    daily: KernelCalibrationExecution,
    alpha_policy: AlphaPolicy,
    selected_neutral_lambda: float | None,
) -> ScientificGenerationPolicy:
    """Derive the policy only from two passed same-artifact kernel results."""

    if not isinstance(monthly, KernelCalibrationExecution) or not isinstance(
        daily, KernelCalibrationExecution
    ):
        raise ModelError("generation policy requires typed kernel executions")
    if (
        monthly.frequency != "monthly"
        or daily.frequency != "daily"
        or monthly.artifact != daily.artifact
        or monthly.n_states != daily.n_states
        or monthly.model_artifact_fingerprint != daily.model_artifact_fingerprint
        or not monthly.passed
        or not daily.passed
        or monthly.offset_grid is None
        or daily.offset_grid is None
    ):
        raise ModelError("generation policy kernel pair is inconsistent")
    if tuple(float(value) for value in monthly.offset_grid.lambdas) != (
        CANONICAL_NEUTRAL_LAMBDA_GRID
    ) or tuple(float(value) for value in daily.offset_grid.lambdas) != (
        CANONICAL_NEUTRAL_LAMBDA_GRID
    ):
        raise ModelError("generation policy kernel lambda grid is invalid")
    return _validated_policy(
        ScientificGenerationPolicy(
            schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
            generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
            block_policy_version=BLOCK_POLICY_VERSION,
            monthly_block_length=monthly.selected_length,
            daily_block_length=daily.selected_length,
            alpha_policy=alpha_policy,
            selected_neutral_lambda=selected_neutral_lambda,
            neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
            initial_state="stationary",
            synchronized_return_vectors=True,
        ),
        ModelError,
    )


def generation_policy_from_mapping(
    value: object,
    *,
    error_type: PolicyError = IntegrityError,
) -> ScientificGenerationPolicy:
    """Parse exact artifact metadata into the typed closed policy."""

    if not isinstance(value, Mapping):
        raise error_type("scientific generation policy is missing")
    expected = {
        "schema_version",
        "generator_algorithm",
        "block_policy_version",
        "monthly_block_length",
        "daily_block_length",
        "alpha_policy",
        "selected_neutral_lambda",
        "neutral_lambda_grid",
        "initial_state",
        "synchronized_return_vectors",
    }
    if set(value) != expected:
        raise error_type("scientific generation policy keys are invalid")
    grid = value["neutral_lambda_grid"]
    if not isinstance(grid, list) or len(grid) != 4:
        raise error_type("scientific generation policy lambda grid is invalid")
    try:
        policy = ScientificGenerationPolicy(
            schema_version=value["schema_version"],
            generator_algorithm=value["generator_algorithm"],
            block_policy_version=value["block_policy_version"],
            monthly_block_length=value["monthly_block_length"],
            daily_block_length=value["daily_block_length"],
            alpha_policy=value["alpha_policy"],
            selected_neutral_lambda=value["selected_neutral_lambda"],
            neutral_lambda_grid=tuple(grid),
            initial_state=value["initial_state"],
            synchronized_return_vectors=value["synchronized_return_vectors"],
        )
    except (TypeError, ValueError) as error:
        raise error_type("scientific generation policy values are invalid") from error
    return _validated_policy(policy, error_type)


def _validated_policy(
    value: ScientificGenerationPolicy,
    error_type: PolicyError,
) -> ScientificGenerationPolicy:
    if not isinstance(value, ScientificGenerationPolicy):
        raise error_type("scientific generation policy has the wrong type")
    if (
        value.schema_version != SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION
        or value.generator_algorithm != SCIENTIFIC_GENERATOR_ALGORITHM
        or value.block_policy_version != BLOCK_POLICY_VERSION
        or value.initial_state != "stationary"
        or value.synchronized_return_vectors is not True
        or tuple(value.neutral_lambda_grid) != CANONICAL_NEUTRAL_LAMBDA_GRID
    ):
        raise error_type("scientific generation policy contract is unsupported")
    for length, (lower, upper), label in (
        (value.monthly_block_length, MONTHLY_BLOCK_LENGTH_BOUNDS, "monthly"),
        (value.daily_block_length, DAILY_BLOCK_LENGTH_BOUNDS, "daily"),
    ):
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not lower <= length <= upper
        ):
            raise error_type(
                f"scientific generation policy {label} block length is invalid"
            )
    if value.alpha_policy == "historical_vector":
        if value.selected_neutral_lambda is not None:
            raise error_type("historical-vector policy cannot select a lambda")
    elif value.alpha_policy == "kernel_mean_neutral":
        selected = value.selected_neutral_lambda
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int | float)
            or not math.isfinite(float(selected))
            or float(selected) not in CANONICAL_NEUTRAL_LAMBDA_GRID
        ):
            raise error_type("kernel-neutral policy requires a registered lambda")
    else:
        raise error_type("scientific generation alpha policy is invalid")
    return value
