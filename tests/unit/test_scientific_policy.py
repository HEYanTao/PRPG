from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prpg.errors import IntegrityError, ModelError
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.kernel_execution import KernelCalibrationExecution
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
    candidate_policy_set_from_kernel_results,
    candidate_policy_set_from_mapping,
    generation_policy_from_kernel_results,
    generation_policy_from_mapping,
)


def _kernel(
    frequency: str,
    *,
    artifact: str = "production",
    selected_length: int | None = None,
    passed: bool = True,
    model_fingerprint: str = "a" * 64,
    lambdas: tuple[float, ...] = CANONICAL_NEUTRAL_LAMBDA_GRID,
) -> KernelCalibrationExecution:
    value = object.__new__(KernelCalibrationExecution)
    fields = {
        "frequency": frequency,
        "artifact": artifact,
        "n_states": 3,
        "model_artifact_fingerprint": model_fingerprint,
        "passed": passed,
        "selected_length": (
            selected_length
            if selected_length is not None
            else (4 if frequency == "monthly" else 20)
        ),
        "offset_grid": type("Grid", (), {"lambdas": np.asarray(lambdas)})(),
    }
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    return value


def test_generation_policy_is_derived_from_exact_kernel_pair_and_roundtrips() -> None:
    policy = generation_policy_from_kernel_results(
        monthly=_kernel("monthly"),
        daily=_kernel("daily"),
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
    )

    assert policy.monthly_block_length == 4
    assert policy.daily_block_length == 20
    assert policy.monthly_restart_probability == 0.25
    assert policy.daily_restart_probability == 0.05
    assert generation_policy_from_mapping(policy.as_dict()) == policy

    neutral = generation_policy_from_kernel_results(
        monthly=_kernel("monthly"),
        daily=_kernel("daily"),
        alpha_policy="kernel_mean_neutral",
        selected_neutral_lambda=0.5,
    )
    assert neutral.selected_neutral_lambda == 0.5


@pytest.mark.parametrize("daily_length", [108, 126])
def test_v2_generation_policy_accepts_registered_daily_boundary_values(
    daily_length: int,
) -> None:
    policy = generation_policy_from_kernel_results(
        monthly=_kernel("monthly"),
        daily=_kernel("daily", selected_length=daily_length),
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
    )

    assert policy.block_policy_version == BLOCK_POLICY_VERSION
    assert policy.daily_block_length == daily_length
    assert generation_policy_from_mapping(policy.as_dict()) == policy


def test_g3_candidate_policy_set_is_explicit_unselected_and_roundtrips() -> None:
    policy_set = candidate_policy_set_from_kernel_results(
        monthly=_kernel("monthly", artifact="design", selected_length=5),
        daily=_kernel("daily", artifact="design", selected_length=21),
    )

    assert isinstance(policy_set, ScientificCandidatePolicySet)
    assert policy_set.candidate_policies == (
        "historical_vector",
        "kernel_mean_neutral",
    )
    assert policy_set.neutral_lambda_candidates == CANONICAL_NEUTRAL_LAMBDA_GRID
    assert policy_set.selected_policy is None
    assert policy_set.selected_neutral_lambda is None
    assert policy_set.qualification_gate == "G5"
    assert not policy_set.generation_authorized
    assert candidate_policy_set_from_mapping(policy_set.as_dict()) == policy_set

    malformed = policy_set.as_dict()
    malformed["selected_policy"] = "historical_vector"
    with pytest.raises(IntegrityError, match="noncanonical"):
        candidate_policy_set_from_mapping(malformed)


@pytest.mark.parametrize(
    ("monthly", "daily", "message"),
    [
        (object(), _kernel("daily"), "typed"),
        (_kernel("daily"), _kernel("daily"), "inconsistent"),
        (_kernel("monthly", artifact="design"), _kernel("daily"), "inconsistent"),
        (_kernel("monthly", passed=False), _kernel("daily"), "inconsistent"),
        (
            _kernel("monthly", lambdas=(0.25, 0.5, 0.75, 0.9)),
            _kernel("daily"),
            "lambda grid",
        ),
    ],
)
def test_kernel_pair_contract_fails_closed(
    monthly: object, daily: object, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        generation_policy_from_kernel_results(
            monthly=monthly,  # type: ignore[arg-type]
            daily=daily,  # type: ignore[arg-type]
            alpha_policy="historical_vector",
            selected_neutral_lambda=None,
        )


def _base_policy() -> ScientificGenerationPolicy:
    return ScientificGenerationPolicy(
        schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=4,
        daily_block_length=20,
        alpha_policy="historical_vector",
        selected_neutral_lambda=None,
        neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"generator_algorithm": "other"},
        {"block_policy_version": "ppw2009-stationary-frequency-wide-v1"},
        {"monthly_block_length": 1},
        {"daily_block_length": 127},
        {"alpha_policy": "other"},
        {"selected_neutral_lambda": 0.5},
        {
            "alpha_policy": "kernel_mean_neutral",
            "selected_neutral_lambda": None,
        },
        {
            "alpha_policy": "kernel_mean_neutral",
            "selected_neutral_lambda": float("inf"),
        },
    ],
)
def test_policy_values_fail_closed(changes: dict[str, object]) -> None:
    malformed = replace(_base_policy(), **changes)  # type: ignore[arg-type]
    with pytest.raises(IntegrityError):
        generation_policy_from_mapping(malformed.as_dict())


def test_mapping_shape_and_custom_error_type_fail_closed() -> None:
    with pytest.raises(IntegrityError, match="missing"):
        generation_policy_from_mapping(None)
    with pytest.raises(IntegrityError, match="keys"):
        generation_policy_from_mapping({"schema_version": 1})
    malformed = _base_policy().as_dict()
    malformed["neutral_lambda_grid"] = [0.25]
    with pytest.raises(ModelError, match="lambda grid"):
        generation_policy_from_mapping(malformed, error_type=ModelError)
