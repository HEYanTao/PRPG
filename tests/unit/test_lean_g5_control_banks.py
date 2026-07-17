from __future__ import annotations

import math
from collections.abc import Sequence
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import prpg.validation.lean_g5_control_banks as control
from prpg.errors import ValidationError
from prpg.validation.lean_g5_control_banks import (
    LEAN_G5_CONTROL_BANK_ROLES,
    LEAN_G5_HF_HISTORY_LENGTH,
    LEAN_G5_HF_PATHS_PER_BANK,
    LEAN_G5_LF_HISTORY_LENGTH,
    LEAN_G5_LF_PATHS_PER_BANK,
    build_scored_lean_g5_control_bank,
    build_scored_lean_g5_control_banks,
    draw_lean_g5_historical_control_path,
    lean_g5_control_bank_geometry,
    lean_g5_control_rng_identity,
)
from prpg.validation.lean_g5_strategies import (
    LEAN_G5_STRATEGY_NAMES,
    Frequency,
    LeanG5StrategyModel,
    fit_lean_g5_strategy_model,
    score_lean_g5_path,
)

FloatArray = NDArray[np.float64]


def _return_paths(
    seed: int,
    *,
    paths: int = 6,
    observations: int = 217,
) -> tuple[FloatArray, ...]:
    generator = np.random.default_rng(seed)
    result: list[FloatArray] = []
    for path_index in range(paths):
        innovations = generator.normal(
            loc=np.asarray((0.0008, 0.0003, 0.0004)),
            scale=np.asarray((0.018, 0.007, 0.008)),
            size=(observations, 3),
        )
        values = np.empty_like(innovations)
        values[0] = innovations[0]
        for row in range(1, observations):
            values[row] = (
                innovations[row]
                + 0.22 * values[row - 1]
                + 0.002
                * math.sin((row + 3 * path_index) / 7.0)
                * np.asarray((1.0, -0.4, 0.6))
            )
        result.append(values)
    return tuple(result)


@pytest.fixture(scope="module")
def monthly_model() -> LeanG5StrategyModel:
    return fit_lean_g5_strategy_model(_return_paths(100), frequency="monthly")


@pytest.fixture(scope="module")
def daily_model() -> LeanG5StrategyModel:
    return fit_lean_g5_strategy_model(
        _return_paths(101, paths=4, observations=600), frequency="daily"
    )


@pytest.fixture(scope="module")
def monthly_source() -> tuple[FloatArray, NDArray[np.bool_]]:
    source = _return_paths(200, paths=1, observations=500)[0]
    links = np.ones(source.shape[0] - 1, dtype=np.bool_)
    links[[73, 241, 388]] = False
    return source, links


def test_public_production_geometry_is_exact_for_every_bank_role() -> None:
    monthly = lean_g5_control_bank_geometry("monthly")
    daily = lean_g5_control_bank_geometry("daily")

    assert LEAN_G5_CONTROL_BANK_ROLES == ("A", "B", "C")
    assert (
        (monthly.path_count, monthly.history_length)
        == (
            LEAN_G5_LF_PATHS_PER_BANK,
            LEAN_G5_LF_HISTORY_LENGTH,
        )
        == (500, 217)
    )
    assert (
        (daily.path_count, daily.history_length)
        == (
            LEAN_G5_HF_PATHS_PER_BANK,
            LEAN_G5_HF_HISTORY_LENGTH,
        )
        == (200, 4_547)
    )


@pytest.mark.parametrize("bank_role", LEAN_G5_CONTROL_BANK_ROLES)
@pytest.mark.parametrize(
    ("frequency", "expected_count", "expected_length"),
    (("monthly", 500, 217), ("daily", 200, 4_547)),
)
def test_omitted_overrides_reach_exact_production_jobs_without_materializing_them(
    monkeypatch: pytest.MonkeyPatch,
    monthly_model: LeanG5StrategyModel,
    daily_model: LeanG5StrategyModel,
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
    bank_role: control.BankRole,
    frequency: Frequency,
    expected_count: int,
    expected_length: int,
) -> None:
    source, links = monthly_source
    model = monthly_model if frequency == "monthly" else daily_model

    def fake_map(
        function: object,
        items: Sequence[control._ScoredControlWork],
        *,
        workers: int,
        chunksize: int,
    ) -> tuple[control._ScoredAlphaChunk, ...]:
        del function, workers, chunksize
        jobs = tuple(items)
        assert jobs[0].start == 0
        assert jobs[-1].stop == expected_count
        assert all(job.history_length == expected_length for job in jobs)
        return tuple(
            control._ScoredAlphaChunk(
                job.start,
                job.stop,
                expected_length - model.warmup_periods,
                np.full((job.stop - job.start, 6), 0.01, dtype=np.float64),
            )
            for job in jobs
        )

    monkeypatch.setattr(control, "deterministic_process_map", fake_map)
    bank = build_scored_lean_g5_control_bank(
        source,
        links,
        model,
        bank_role=bank_role,
        frequency=frequency,
        mean_block_length=8.0,
        master_seed=20260715,
        scientific_version=11,
    )

    assert bank.path_count == expected_count
    assert bank.history_length == expected_length
    assert bank.production_geometry
    assert bank.noncanonical_use is None
    assert bank.alpha_rows.shape == (expected_count, 6)


def test_reduced_geometry_requires_an_explicit_label_and_cannot_expand(
    monthly_model: LeanG5StrategyModel,
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
) -> None:
    source, links = monthly_source
    common = {
        "bank_role": "A",
        "frequency": "monthly",
        "mean_block_length": 4.0,
        "master_seed": 123,
        "scientific_version": 11,
        "workers": 1,
    }
    with pytest.raises(ValidationError, match="labelled pilot or test"):
        build_scored_lean_g5_control_bank(
            source,
            links,
            monthly_model,
            path_count=4,
            history_length=100,
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="only reduce"):
        build_scored_lean_g5_control_bank(
            source,
            links,
            monthly_model,
            path_count=501,
            noncanonical_use="test",
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="cannot be relabelled"):
        build_scored_lean_g5_control_bank(
            source,
            links,
            monthly_model,
            noncanonical_use="test",
            **common,  # type: ignore[arg-type]
        )


def test_bank_rng_streams_are_registered_disjoint_and_reproducible() -> None:
    frequencies: tuple[Frequency, ...] = ("monthly", "daily")
    identities = tuple(
        lean_g5_control_rng_identity(
            bank_role=role,
            frequency=frequency,
            path_index=index,
            master_seed=20260715,
            scientific_version=11,
        )
        for role in LEAN_G5_CONTROL_BANK_ROLES
        for frequency in frequencies
        for index in range(3)
    )

    assert len({item.spawn_key for item in identities}) == len(identities)
    assert len({item.fingerprint for item in identities}) == len(identities)
    repeated = lean_g5_control_rng_identity(
        bank_role="C",
        frequency="daily",
        path_index=2,
        master_seed=20260715,
        scientific_version=11,
    )
    assert repeated == identities[-1]
    role_coordinates = {
        role: (item.artifact, item.replicate)
        for role in LEAN_G5_CONTROL_BANK_ROLES
        for item in identities
        if item.bank_role == role
        and item.frequency == "monthly"
        and item.path_index == 0
    }
    assert role_coordinates == {"A": (0, 0), "B": (1, 0), "C": (0, 10_000)}


def test_draw_is_gap_safe_non_circular_and_exactly_reproducible(
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
) -> None:
    source, links = monthly_source
    first = draw_lean_g5_historical_control_path(
        source,
        links,
        bank_role="C",
        frequency="monthly",
        path_index=7,
        history_length=180,
        mean_block_length=30.0,
        master_seed=20260715,
        scientific_version=11,
        noncanonical_use="test",
    )
    repeated = draw_lean_g5_historical_control_path(
        source,
        links,
        bank_role="C",
        frequency="monthly",
        path_index=7,
        history_length=180,
        mean_block_length=30.0,
        master_seed=20260715,
        scientific_version=11,
        noncanonical_use="test",
    )

    assert first.fingerprint == repeated.fingerprint
    np.testing.assert_array_equal(first.source_indices, repeated.source_indices)
    np.testing.assert_array_equal(first.log_returns, repeated.log_returns)
    assert first.source_indices.shape == (180,)
    assert first.log_returns.shape == (180, 3)
    assert first.mean_block_length == 30.0
    assert first.restart_flags[0]
    for position in range(1, first.history_length):
        previous = int(first.source_indices[position - 1])
        current = int(first.source_indices[position])
        if not first.restart_flags[position]:
            assert previous < source.shape[0] - 1
            assert links[previous]
            assert current == previous + 1
    for array in (first.source_indices, first.restart_flags, first.log_returns):
        assert not array.flags.writeable


def test_bank_uses_identical_nonzero_historical_scoring(
    monthly_model: LeanG5StrategyModel,
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
) -> None:
    source, links = monthly_source
    bank = build_scored_lean_g5_control_bank(
        source,
        links,
        monthly_model,
        bank_role="A",
        frequency="monthly",
        mean_block_length=4.0,
        master_seed=20260715,
        scientific_version=11,
        workers=1,
        path_count=3,
        history_length=217,
        noncanonical_use="test",
        batch_size=2,
    )
    first_path = draw_lean_g5_historical_control_path(
        source,
        links,
        bank_role="A",
        frequency="monthly",
        path_index=0,
        mean_block_length=4.0,
        master_seed=20260715,
        scientific_version=11,
    )
    direct = score_lean_g5_path(monthly_model, first_path.log_returns)

    assert bank.strategy_names == LEAN_G5_STRATEGY_NAMES
    assert bank.alpha_rows.shape == (3, 6)
    assert not bank.alpha_rows.flags.writeable
    assert not np.all(bank.alpha_rows == 0.0)
    np.testing.assert_array_equal(bank.alpha_rows[0], direct.alpha_sharpes)


def test_all_three_small_banks_bind_distinct_role_and_rng_identity(
    monthly_model: LeanG5StrategyModel,
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
) -> None:
    source, links = monthly_source
    banks = build_scored_lean_g5_control_banks(
        source,
        links,
        monthly_model,
        frequency="monthly",
        mean_block_length=4.0,
        master_seed=20260715,
        scientific_version=11,
        workers=1,
        path_count=3,
        history_length=217,
        noncanonical_use="pilot",
        batch_size=2,
    )

    assert tuple(bank.bank_role for bank in banks) == LEAN_G5_CONTROL_BANK_ROLES
    assert all(not bank.production_geometry for bank in banks)
    assert all(bank.noncanonical_use == "pilot" for bank in banks)
    assert len({bank.rng_identity_fingerprint for bank in banks}) == 3
    assert len({bank.alpha_rows_sha256 for bank in banks}) == 3
    assert len({bank.fingerprint for bank in banks}) == 3
    assert len({bank.source_fingerprint for bank in banks}) == 1
    assert len({bank.model_fingerprint for bank in banks}) == 1


def test_small_fixture_is_byte_identical_with_one_and_nine_workers(
    monthly_model: LeanG5StrategyModel,
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
) -> None:
    source, links = monthly_source
    arguments = {
        "bank_role": "B",
        "frequency": "monthly",
        "mean_block_length": 4.0,
        "master_seed": 20260715,
        "scientific_version": 11,
        "path_count": 9,
        "history_length": 217,
        "noncanonical_use": "test",
        "batch_size": 1,
    }
    serial = build_scored_lean_g5_control_bank(
        source,
        links,
        monthly_model,
        workers=1,
        **arguments,  # type: ignore[arg-type]
    )
    # macOS may report semaphore capacity as indeterminate.  This preserves
    # actual spawn execution while avoiding an irrelevant stdlib warning path.
    with patch("concurrent.futures.process.os.sysconf", return_value=-1):
        parallel = build_scored_lean_g5_control_bank(
            source,
            links,
            monthly_model,
            workers=9,
            **arguments,  # type: ignore[arg-type]
        )

    assert serial.alpha_rows.tobytes(order="C") == parallel.alpha_rows.tobytes(
        order="C"
    )
    assert serial.alpha_rows_sha256 == parallel.alpha_rows_sha256
    assert serial.rng_identity_fingerprint == parallel.rng_identity_fingerprint
    assert serial.fingerprint == parallel.fingerprint


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"bank_role": "D"}, "role"),
        ({"frequency": "weekly"}, "frequency"),
        ({"mean_block_length": 0.5}, "block length"),
        ({"workers": 10}, "nine"),
        ({"history_length": 16, "noncanonical_use": "test"}, "too short"),
    ),
)
def test_invalid_bank_inputs_fail_before_returning_scores(
    monthly_model: LeanG5StrategyModel,
    monthly_source: tuple[FloatArray, NDArray[np.bool_]],
    overrides: dict[str, object],
    message: str,
) -> None:
    source, links = monthly_source
    arguments: dict[str, object] = {
        "bank_role": "A",
        "frequency": "monthly",
        "mean_block_length": 4.0,
        "master_seed": 123,
        "scientific_version": 11,
        "workers": 1,
        "path_count": 1,
        "history_length": 100,
        "noncanonical_use": "test",
    }
    arguments.update(overrides)
    with pytest.raises(ValidationError, match=message):
        build_scored_lean_g5_control_bank(
            source,
            links,
            monthly_model,
            **arguments,  # type: ignore[arg-type]
        )
