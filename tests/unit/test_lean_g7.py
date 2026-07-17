from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from prpg.validation.lean_g5_assessment import LeanG5SuppliedControlBank
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
from prpg.validation.lean_g7 import (
    LEAN_G7_PRODUCTION_GEOMETRY,
    LeanG7Geometry,
    LeanG7StructuralHook,
)
from prpg.validation.structural import StructuralPathView

FloatArray = NDArray[np.float64]
Frequency = Literal["monthly", "daily"]


@dataclass(frozen=True, slots=True)
class _Fixture:
    geometry: LeanG7Geometry
    controls: tuple[LeanG5SuppliedControlBank, ...]
    indices: LeanG5ResampleIndices
    monthly: tuple[FloatArray, ...]
    daily: tuple[FloatArray, ...]


def _source(rows: int, *, seed: int, daily: bool) -> FloatArray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(21.0) if daily else 1.0
    covariance = (
        np.asarray(
            (
                (0.035**2, 0.00012, 0.00020),
                (0.00012, 0.013**2, 0.00011),
                (0.00020, 0.00011, 0.017**2),
            )
        )
        * scale**2
    )
    innovations = rng.multivariate_normal(
        np.asarray((0.004, 0.0015, 0.0020)) / (21.0 if daily else 1.0),
        covariance,
        size=rows,
    )
    result = np.empty_like(innovations)
    result[0] = innovations[0]
    for row in range(1, rows):
        result[row] = innovations[row] + 0.18 * result[row - 1]
    return np.asarray(result, dtype=np.float64)


def _control_bank(
    role: BankRole,
    *,
    monthly_source: FloatArray,
    daily_source: FloatArray,
    paths: int,
    monthly_history: int,
    daily_history: int,
) -> LeanG5SuppliedControlBank:
    monthly_links = np.ones(monthly_source.shape[0] - 1, dtype=np.bool_)
    daily_links = np.ones(daily_source.shape[0] - 1, dtype=np.bool_)
    monthly = tuple(
        draw_lean_g5_historical_control_path(
            monthly_source,
            monthly_links,
            bank_role=role,
            frequency="monthly",
            path_index=index,
            history_length=monthly_history,
            mean_block_length=7.0,
            master_seed=20260715,
            scientific_version=11,
            noncanonical_use="test",
        )
        for index in range(paths)
    )
    daily = tuple(
        draw_lean_g5_historical_control_path(
            daily_source,
            daily_links,
            bank_role=role,
            frequency="daily",
            path_index=index,
            history_length=daily_history,
            mean_block_length=20.0,
            master_seed=20260715,
            scientific_version=11,
            noncanonical_use="test",
        )
        for index in range(paths)
    )
    return LeanG5SuppliedControlBank(
        bank_role=role,
        monthly_draws=monthly,
        daily_draws=daily,
        monthly_source_states=np.zeros(monthly_source.shape[0], dtype=np.int64),
        daily_source_states=np.zeros(daily_source.shape[0], dtype=np.int64),
    )


@pytest.fixture(scope="module")
def fixture() -> _Fixture:
    subset = 8
    geometry = LeanG7Geometry.test(
        lf_paths=10,
        hf_paths=10,
        scientific_lf_paths=subset,
        scientific_hf_paths=subset,
        resamples=256,
        monthly_path_length=24,
        daily_path_length=260,
        monthly_history_length=24,
        daily_history_length=260,
    )
    monthly_source = _source(480, seed=101, daily=False)
    daily_source = _source(780, seed=202, daily=True)
    controls = tuple(
        _control_bank(
            role,
            monthly_source=monthly_source,
            daily_source=daily_source,
            paths=subset,
            monthly_history=24,
            daily_history=260,
        )
        for role in ("A", "B", "C")
    )
    indices = resample_indices_from_registered_seeds(
        LeanG5RegisteredResampleSeeds(31, 32, 33, 34),
        geometry=LeanG5Geometry.test(
            monthly_paths=subset,
            daily_paths=subset,
            resamples=256,
        ),
    )
    control_c = controls[2]
    monthly = (
        *(
            np.asarray(draw.log_returns, dtype=np.float64).copy()
            for draw in control_c.monthly_draws
        ),
        _source(24, seed=901, daily=False),
        _source(24, seed=902, daily=False),
    )
    daily = (
        *(
            np.asarray(draw.log_returns, dtype=np.float64).copy()
            for draw in control_c.daily_draws
        ),
        _source(260, seed=903, daily=True),
        _source(260, seed=904, daily=True),
    )
    return _Fixture(geometry, controls, indices, monthly, daily)


def _hook(fixture: _Fixture) -> LeanG7StructuralHook:
    return LeanG7StructuralHook(
        fixture.controls,
        fixture.indices,
        g5_assessment_fingerprint="a" * 64,
        g5_authority_fingerprint="b" * 64,
        geometry=fixture.geometry,
    )


def _observe(
    hook: LeanG7StructuralHook,
    fixture: _Fixture,
    *,
    monthly_transform: Callable[[FloatArray, int], FloatArray] | None = None,
    duplicate: bool = False,
) -> dict[str, Any]:
    first_monthly: FloatArray | None = None
    for entity, original in enumerate(fixture.monthly, start=1):
        values = original.copy()
        if (
            monthly_transform is not None
            and entity <= fixture.geometry.scientific_lf_paths
        ):
            values = monthly_transform(values, entity)
        if entity == 1:
            first_monthly = values.copy()
        elif entity == 2 and duplicate:
            assert first_monthly is not None
            values = first_monthly.copy()
        hook.observe_path(
            StructuralPathView(
                family="LF",
                frequency="monthly",
                path_entity=entity,
                path_id=f"LF{entity:06d}",
                log_returns=values,
            )
        )
    for entity, values in enumerate(fixture.daily, start=1):
        hook.observe_path(
            StructuralPathView(
                family="HF",
                frequency="daily",
                path_entity=entity,
                path_id=f"HF{entity:06d}",
                log_returns=values,
            )
        )
    return dict(hook.finalize())


def _group(report: dict[str, Any], family: str) -> dict[str, Any]:
    return next(item for item in report["grouped_p_values"] if item["family"] == family)


def test_canonical_geometry_is_exact_and_clean_fixture_passes(
    fixture: _Fixture,
) -> None:
    canonical = LEAN_G7_PRODUCTION_GEOMETRY
    assert (
        canonical.lf_paths,
        canonical.hf_paths,
        canonical.scientific_lf_paths,
        canonical.scientific_hf_paths,
        canonical.resamples,
    ) == (5_000, 1_000, 500, 200, 10_000)

    report = _observe(_hook(fixture), fixture)

    assert report["decision"] == "pass"
    assert report["counts"] == {
        "observed_lf_paths": 10,
        "observed_hf_paths": 10,
        "scientific_lf_paths": 8,
        "scientific_hf_paths": 8,
        "complete_unique_paths": 20,
        "complete_duplicate_copies": 0,
    }
    assert all(not item["rejected"] for item in report["grouped_p_values"])
    assert len(report["fingerprint"]) == 64


def _covariance(values: FloatArray, _entity: int) -> FloatArray:
    values[:, 0] *= 8.0
    return values


def _correlation(values: FloatArray, entity: int) -> FloatArray:
    values[:, 2] = values[:, 0] + entity * 1.0e-8
    return values


def _tail(values: FloatArray, _entity: int) -> FloatArray:
    values[[0, 4, 8, 12], 0] -= 0.75
    return values


def _drawdown(values: FloatArray, _entity: int) -> FloatArray:
    values[:16, 0] = -0.09
    values[16:, 0] = 0.01
    return values


def _spread(values: FloatArray, _entity: int) -> FloatArray:
    values[:, 2] += 0.12
    return values


@pytest.mark.parametrize(
    ("transform", "family"),
    (
        (_covariance, "G5-B"),
        (_correlation, "G5-B"),
        (_tail, "G5-C"),
        (_drawdown, "G5-D"),
        (_spread, "G5-C"),
    ),
)
def test_material_observable_distortions_are_non_passes(
    fixture: _Fixture,
    transform: Callable[[FloatArray, int], FloatArray],
    family: str,
) -> None:
    report = _observe(_hook(fixture), fixture, monthly_transform=transform)

    assert report["decision"] == "non_pass"
    assert _group(report, family)["rejected"] is True


def test_complete_production_path_duplicate_is_a_direct_non_pass(
    fixture: _Fixture,
) -> None:
    report = _observe(_hook(fixture), fixture, duplicate=True)

    assert report["decision"] == "non_pass"
    assert report["counts"]["complete_duplicate_copies"] == 1
    assert any("complete production" in reason for reason in report["reasons"])


def _repeat(values: FloatArray, entity: int) -> FloatArray:
    pattern = np.arange(18, dtype=np.float64).reshape(6, 3) * 1.0e-5
    values[:18] = np.tile(pattern, (3, 1))
    values[-1, 0] += entity * 1.0e-7
    return values


def test_repeated_subsequence_excess_is_a_direct_non_pass(
    fixture: _Fixture,
) -> None:
    report = _observe(_hook(fixture), fixture, monthly_transform=_repeat)
    monthly = next(
        item
        for item in report["repeated_subsequences"]
        if item["frequency"] == "monthly"
    )

    assert report["decision"] == "non_pass"
    assert monthly["passed"] is False
    assert any("repeated subsequences" in reason for reason in report["reasons"])


def test_quarterly_annual_and_weekly_views_are_ignored(fixture: _Fixture) -> None:
    hook = _hook(fixture)
    bad = np.asarray([[np.nan, np.nan, np.nan]])
    for frequency, family in (
        ("quarterly", "LF"),
        ("annual", "LF"),
        ("weekly", "HF"),
    ):
        hook.observe_path(
            StructuralPathView(
                family=family,
                frequency=frequency,
                path_entity=999_999,
                path_id="ignored",
                log_returns=bad,
            )
        )

    report = _observe(hook, fixture)

    assert report["decision"] == "pass"
    assert report["counts"]["observed_lf_paths"] == 10
    assert report["counts"]["observed_hf_paths"] == 10
