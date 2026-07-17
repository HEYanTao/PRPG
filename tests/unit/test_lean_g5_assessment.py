from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pytest

import prpg.validation.lean_g5_assessment as assessment
from prpg.errors import ValidationError
from prpg.validation.lean_g5_assessment import (
    LEAN_G5_ASSESSMENT_SCHEMA_ID,
    LeanG5AssessmentReport,
    LeanG5SuppliedControlBank,
    assess_lean_g5,
)
from prpg.validation.lean_g5_control_banks import (
    BankRole,
    draw_lean_g5_historical_control_path,
)
from prpg.validation.lean_g5_statistics import (
    LeanG5Geometry,
    LeanG5RegisteredResampleSeeds,
    LeanG5ResampleIndices,
    resample_indices_from_registered_seeds,
)
from prpg.validation.qualification import (
    CandidateStage,
    G5CandidateBundle,
    G5CandidatePathView,
)

Family = Literal["LF", "HF"]


@dataclass(frozen=True, slots=True)
class _Inputs:
    development: G5CandidateBundle
    qualification: G5CandidateBundle
    controls: tuple[LeanG5SuppliedControlBank, ...]
    indices: LeanG5ResampleIndices
    geometry: LeanG5Geometry


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path_view(
    *,
    family: Family,
    observations: int,
    seed: int,
    prefix: str,
) -> G5CandidatePathView:
    rng = np.random.default_rng(seed)
    if family == "LF":
        mean = np.asarray((0.0008, 0.0003, 0.0004))
        scale = np.asarray((0.0180, 0.0070, 0.0080))
        cycle_scale = 0.0020
    else:
        mean = np.asarray((0.0008, 0.0003, 0.0004)) / 21.0
        scale = np.asarray((0.0180, 0.0070, 0.0080)) / np.sqrt(21.0)
        cycle_scale = 0.0004
    innovations = rng.normal(mean, scale, size=(observations, 3))
    returns = np.empty_like(innovations)
    returns[0] = innovations[0]
    direction = np.asarray((1.0, -0.4, 0.6))
    for row in range(1, observations):
        returns[row] = (
            innovations[row]
            + 0.22 * returns[row - 1]
            + cycle_scale * math.sin((row + seed % 29) / 7.0) * direction
        )
    states = rng.integers(0, 4, size=observations, dtype=np.int64)
    identity = _digest(f"{prefix}-{family}-{seed}")
    return G5CandidatePathView(
        path_id=f"{prefix}-{family}-{seed}",
        family=family,
        log_returns=np.asarray(returns, dtype=np.float64),
        target_states=states,
        source_fingerprint=identity,
        fingerprint=identity,
        path_entity=seed,
    )


def _views(
    count: int, *, family: Family, observations: int, seed_start: int, prefix: str
) -> tuple[G5CandidatePathView, ...]:
    return tuple(
        _path_view(
            family=family,
            observations=observations,
            seed=seed_start + index,
            prefix=prefix,
        )
        for index in range(count)
    )


def _bundle(
    *,
    stage: CandidateStage,
    lf_paths: tuple[G5CandidatePathView, ...],
    hf_paths: tuple[G5CandidatePathView, ...],
    exact: bool,
) -> G5CandidateBundle:
    return G5CandidateBundle(
        schema_id="test-g5-bundle",
        candidate_stage=stage,
        generation_policy_fingerprint=_digest("policy-v11"),
        lf_paths=lf_paths,
        hf_paths=hf_paths,
        elapsed_seconds=1.0,
        exact_canonical_counts=exact,
        generation_authorized=False,
        fingerprint=_digest(f"{stage}-{len(lf_paths)}-{len(hf_paths)}"),
        policy_scientific_version=11,
    )


def _control_bank(
    role: BankRole,
    *,
    monthly_source: G5CandidatePathView,
    daily_source: G5CandidatePathView,
    paths: int,
) -> LeanG5SuppliedControlBank:
    monthly_links = np.ones(monthly_source.log_returns.shape[0] - 1, dtype=np.bool_)
    daily_links = np.ones(daily_source.log_returns.shape[0] - 1, dtype=np.bool_)
    monthly_draws = tuple(
        draw_lean_g5_historical_control_path(
            monthly_source.log_returns,
            monthly_links,
            bank_role=role,
            frequency="monthly",
            path_index=index,
            mean_block_length=12.0,
            master_seed=20260715,
            scientific_version=11,
        )
        for index in range(paths)
    )
    daily_draws = tuple(
        draw_lean_g5_historical_control_path(
            daily_source.log_returns,
            daily_links,
            bank_role=role,
            frequency="daily",
            path_index=index,
            mean_block_length=60.0,
            master_seed=20260715,
            scientific_version=11,
        )
        for index in range(paths)
    )
    return LeanG5SuppliedControlBank(
        bank_role=role,
        monthly_draws=monthly_draws,
        daily_draws=daily_draws,
        monthly_source_states=monthly_source.target_states,
        daily_source_states=daily_source.target_states,
    )


@pytest.fixture(scope="module")
def inputs() -> _Inputs:
    geometry = LeanG5Geometry.test(
        monthly_paths=12,
        daily_paths=12,
        resamples=128,
    )
    development = _bundle(
        stage="development",
        lf_paths=_views(
            100, family="LF", observations=217, seed_start=1_000, prefix="D"
        ),
        hf_paths=_views(
            20, family="HF", observations=4_547, seed_start=2_000, prefix="D"
        ),
        exact=True,
    )
    qualification = _bundle(
        stage="qualification",
        lf_paths=_views(
            12, family="LF", observations=217, seed_start=3_000, prefix="Q"
        ),
        hf_paths=_views(
            12, family="HF", observations=4_547, seed_start=4_000, prefix="Q"
        ),
        exact=False,
    )
    roles: tuple[tuple[BankRole, int], ...] = (
        ("A", 5_000),
        ("B", 6_000),
        ("C", 7_000),
    )
    monthly_source = _path_view(family="LF", observations=360, seed=5_000, prefix="HM")
    daily_source = _path_view(family="HF", observations=5_200, seed=6_000, prefix="HD")
    controls = tuple(
        _control_bank(
            role,
            monthly_source=monthly_source,
            daily_source=daily_source,
            paths=12,
        )
        for role, _start in roles
    )
    indices = resample_indices_from_registered_seeds(
        LeanG5RegisteredResampleSeeds(11, 12, 13, 14), geometry=geometry
    )
    return _Inputs(development, qualification, controls, indices, geometry)


@pytest.fixture(scope="module")
def report(inputs: _Inputs) -> LeanG5AssessmentReport:
    return assess_lean_g5(
        inputs.development,
        inputs.qualification,
        inputs.controls,
        inputs.indices,
        geometry=inputs.geometry,
    )


def test_assessment_returns_one_typed_nonauthorizing_report(
    report: LeanG5AssessmentReport,
) -> None:
    assert report.schema_id == LEAN_G5_ASSESSMENT_SCHEMA_ID
    assert report.generation_authorized is False
    assert len(report.fingerprint) == 64
    assert len(report.resample_indices_fingerprint) == 64
    assert report.control_mean_block_lengths == (("monthly", 12.0), ("daily", 60.0))
    assert tuple(item.family for item in report.grouped_comparisons) == (
        "G5-A",
        "G5-B",
        "G5-C",
        "G5-D",
    )
    assert report.alpha.equivalence.names[0] == "monthly.momentum_6m"
    assert report.alpha.leakage_gap.historical_estimates != (0.0,) * 12
    assert report.states.qualification_monthly.state_count == 4
    assert report.states.qualification_daily.state_count == 4
    assert report.diversity.complete_duplicates_passed


def test_assessment_is_deterministic(
    inputs: _Inputs, report: LeanG5AssessmentReport
) -> None:
    repeated = assess_lean_g5(
        inputs.development,
        inputs.qualification,
        inputs.controls,
        inputs.indices,
        geometry=inputs.geometry,
    )
    assert repeated.fingerprint == report.fingerprint
    assert repeated.reasons == report.reasons


def test_exact_duplicate_is_a_direct_non_pass(inputs: _Inputs) -> None:
    copied_lf = list(inputs.qualification.lf_paths)
    copied_lf[1] = replace(
        copied_lf[1],
        log_returns=copied_lf[0].log_returns.copy(),
        target_states=copied_lf[0].target_states.copy(),
    )
    defective = replace(inputs.qualification, lf_paths=tuple(copied_lf))
    result = assess_lean_g5(
        inputs.development,
        defective,
        inputs.controls,
        inputs.indices,
        geometry=inputs.geometry,
    )
    assert result.decision == "non_pass"
    assert not result.diversity.complete_duplicates_passed
    assert any("duplicate qualification" in reason for reason in result.reasons)


def test_development_count_fails_before_fitting(inputs: _Inputs) -> None:
    defective = replace(
        inputs.development,
        lf_paths=inputs.development.lf_paths[:-1],
    )
    with pytest.raises(ValidationError, match="path count"):
        assess_lean_g5(
            defective,
            inputs.qualification,
            inputs.controls,
            inputs.indices,
            geometry=inputs.geometry,
        )


def test_control_draw_ordinal_and_registered_rng_identity_are_bound(
    inputs: _Inputs,
) -> None:
    first = inputs.controls[0]
    reordered = list(first.monthly_draws)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    defective_first = replace(first, monthly_draws=tuple(reordered))
    defective_controls = (defective_first, *inputs.controls[1:])

    with pytest.raises(ValidationError, match="draw identity"):
        assess_lean_g5(
            inputs.development,
            inputs.qualification,
            defective_controls,
            inputs.indices,
            geometry=inputs.geometry,
        )

    changed_length = list(first.monthly_draws)
    changed_length[0] = replace(changed_length[0], mean_block_length=13.0)
    bad_length_first = replace(first, monthly_draws=tuple(changed_length))
    with pytest.raises(ValidationError, match="source, block, and RNG"):
        assess_lean_g5(
            inputs.development,
            inputs.qualification,
            (bad_length_first, *inputs.controls[1:]),
            inputs.indices,
            geometry=inputs.geometry,
        )


def test_group_bootstrap_centers_ab_without_shifting_independent_qc() -> None:
    geometry = LeanG5Geometry.test(monthly_paths=2, daily_paths=2, resamples=4)
    parts = assessment._ComparisonParts(
        observed=np.asarray((0.0,)),
        null=np.asarray(((9.0,), (11.0,), (8.0,), (12.0,))),
        center=np.asarray((10.0,)),
    )
    result = assessment._finish_comparison("G5-B", (parts,), geometry)

    assert result.observed_statistic == 0.0
    assert result.p_value is not None
    assert result.p_value.p_value == 1.0


def test_resample_index_identity_changes_report_fingerprint(
    inputs: _Inputs, report: LeanG5AssessmentReport
) -> None:
    changed_monthly = np.asarray(inputs.indices.simulated_monthly).copy()
    changed_monthly[0, 0] = (changed_monthly[0, 0] + 1) % inputs.geometry.monthly_paths
    changed_indices = replace(
        inputs.indices,
        simulated_monthly=changed_monthly,
    )
    changed = assess_lean_g5(
        inputs.development,
        inputs.qualification,
        inputs.controls,
        changed_indices,
        geometry=inputs.geometry,
    )

    assert changed.resample_indices_fingerprint != report.resample_indices_fingerprint
    assert changed.fingerprint != report.fingerprint


def test_invalid_resample_index_tampering_fails_before_model_fit(
    inputs: _Inputs,
) -> None:
    invalid_monthly = np.asarray(inputs.indices.simulated_monthly).copy()
    invalid_monthly[0, 0] = -1
    invalid_indices = replace(
        inputs.indices,
        simulated_monthly=invalid_monthly,
    )

    with pytest.raises(ValidationError, match="resample indices"):
        assess_lean_g5(
            inputs.development,
            inputs.qualification,
            inputs.controls,
            invalid_indices,
            geometry=inputs.geometry,
        )


def test_policy_versions_bind_development_qualification_and_controls(
    inputs: _Inputs,
) -> None:
    with pytest.raises(ValidationError, match="development bundle"):
        assess_lean_g5(
            replace(inputs.development, policy_scientific_version=12),
            inputs.qualification,
            inputs.controls,
            inputs.indices,
            geometry=inputs.geometry,
        )
    with pytest.raises(ValidationError, match="differs from qualification"):
        assess_lean_g5(
            inputs.development,
            replace(inputs.qualification, policy_scientific_version=12),
            inputs.controls,
            inputs.indices,
            geometry=inputs.geometry,
        )


def test_large_repeated_subsequence_injection_is_a_direct_non_pass(
    inputs: _Inputs,
) -> None:
    copied_lf = list(inputs.qualification.lf_paths)
    template = copied_lf[0].log_returns[:6].copy()
    for index, path in enumerate(copied_lf):
        altered = path.log_returns.copy()
        for start in range(20, 200, 9):
            altered[start : start + 6] = template
        copied_lf[index] = replace(path, log_returns=altered)
    defective = replace(inputs.qualification, lf_paths=tuple(copied_lf))
    result = assess_lean_g5(
        inputs.development,
        defective,
        inputs.controls,
        inputs.indices,
        geometry=inputs.geometry,
    )
    assert result.diversity.complete_duplicates_passed
    assert result.diversity.repeated_subsequence_status == "non_pass"
    assert any("two-times-A/B/C-control" in reason for reason in result.reasons)
