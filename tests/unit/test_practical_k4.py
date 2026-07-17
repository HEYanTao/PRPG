from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prpg.config import load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.model.input import load_calibration_input
from prpg.model.practical_k4 import (
    OWNER_APPROVED_DESIGN_DAILY_SESSIONS,
    OWNER_APPROVED_DESIGN_MONTHS,
    OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT,
    OWNER_APPROVED_STATE_COUNTS,
    PracticalK4ArtifactStore,
    expected_owner_approved_monthly_states,
    reproduce_owner_approved_k4,
)
from prpg.simulation.serial import (
    build_practical_k4_serial_generation_input,
    generate_configured_serial_base_path,
    is_sealed_serial_generation_input,
)

ROOT = Path(__file__).resolve().parents[2]


def test_owner_approved_month_ranges_are_complete_and_exact() -> None:
    months = pd.period_range("2008-04", "2024-04", freq="M")
    states = expected_owner_approved_monthly_states(
        months.asi8.astype(np.int64, copy=False)
    )

    assert len(states) == OWNER_APPROVED_DESIGN_MONTHS
    assert tuple(np.bincount(states, minlength=4)) == OWNER_APPROVED_STATE_COUNTS
    assert not states.flags.writeable


def test_owner_fit_persists_and_drives_smoke_paths(tmp_path: Path) -> None:
    processed = (
        ROOT
        / "artifacts/data/processed/sha256"
        / OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT
        / "processed-manifest.json"
    )
    if not processed.is_file():
        pytest.skip("immutable owner-approved data snapshot is not installed")

    canonical = load_config(ROOT / "configs/canonical.yaml")
    canonical_input = load_calibration_input(
        ProcessedSnapshotStore(ROOT / canonical.data.artifact_root),
        OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=canonical.fingerprint(),
        holdout_months=canonical.model.design_holdout_months,
    )
    artifact = reproduce_owner_approved_k4(canonical_input)
    store = PracticalK4ArtifactStore(tmp_path)
    reference = store.persist(artifact)
    loaded = store.load(reference.fingerprint)

    assert loaded.fingerprint == artifact.fingerprint
    assert len(loaded.monthly_source_states) == OWNER_APPROVED_DESIGN_MONTHS
    assert len(loaded.daily_source_states) == OWNER_APPROVED_DESIGN_DAILY_SESSIONS
    assert loaded.generation_policy.monthly_block_length == 2
    assert loaded.generation_policy.daily_block_length == 62

    smoke = load_config(ROOT / "configs/smoke.yaml")
    smoke_input = load_calibration_input(
        ProcessedSnapshotStore(ROOT / smoke.data.artifact_root),
        OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=smoke.fingerprint(),
        holdout_months=smoke.model.design_holdout_months,
    )
    generation_input = build_practical_k4_serial_generation_input(
        authority=loaded,
        config=smoke,
        calibration_input=smoke_input,
    )
    assert is_sealed_serial_generation_input(generation_input)

    lf = generate_configured_serial_base_path(
        generation_input, family="LF", path_entity=1
    )
    hf = generate_configured_serial_base_path(
        generation_input, family="HF", path_entity=1
    )
    repeated = generate_configured_serial_base_path(
        generation_input, family="LF", path_entity=1
    )
    assert lf.log_returns.shape == (24, 3)
    assert hf.log_returns.shape == (504, 3)
    np.testing.assert_array_equal(lf.log_returns, repeated.log_returns)
