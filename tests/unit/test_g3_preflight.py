from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prpg.config import load_config
from prpg.errors import ModelError
from prpg.execution import ResourceSnapshot
from prpg.model import g3_preflight as preflight_module
from prpg.model.bridge import bridge_resource_preflight
from prpg.model.g3_preflight import (
    CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
    CANONICAL_CONFIG_FINGERPRINT,
    CANONICAL_LOGICAL_CPUS,
    CANONICAL_MACHINE,
    CANONICAL_PHYSICAL_MEMORY_BYTES,
    CANONICAL_PLATFORM,
    CANONICAL_PYTHON_VERSION,
    EXPECTED_RUNTIME_VERSIONS,
    RuntimeEnvironment,
    canonical_bridge_resource_preflight,
    canonical_g3_preflight,
    canonical_g3_resource_estimate,
    canonical_resume_preflight_fingerprint,
    inspect_runtime_environment,
    persist_canonical_g3_preflight,
    read_persisted_canonical_g3_preflight,
    verify_live_canonical_g3_preflight,
)
from prpg.model.input import (
    ASSET_COLUMNS,
    MACRO_COLUMNS,
    CalibrationInput,
    CalibrationProvenance,
    CalibrationSlice,
    SourceIdentity,
    SplitGeometry,
)
from prpg.model.run_store import CalibrationRunStore, default_run_identity
from prpg.model.scientific_artifact import CANONICAL_SCIENTIFIC_VERSION
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
)

ROOT = Path(__file__).parents[2]


def test_version_3_canonical_config_and_calibration_identities_are_frozen() -> None:
    config = load_config(ROOT / "configs/canonical.yaml")

    assert CANONICAL_SCIENTIFIC_VERSION == 11
    assert config.simulation.scientific_version == CANONICAL_SCIENTIFIC_VERSION
    assert config.fingerprint() == CANONICAL_CONFIG_FINGERPRINT
    assert (
        CANONICAL_CONFIG_FINGERPRINT
        == "572e0a843a540b0a6e307cfb72c3c9c43d2fb6cafc8c2c52f6691f408c38ae58"
    )
    assert (
        CANONICAL_CALIBRATION_INPUT_FINGERPRINT
        == "17d61d99a5b2618abeab5e0898ffe34d6bc1ed35a309ea9be3ad7455b45fd38f"
    )


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _mask(size: int, starts: tuple[int, ...]) -> np.ndarray:
    value = np.ones(size, dtype=np.bool_)
    value[list(starts)] = False
    return _immutable(value)


def _slice(
    role: str,
    monthly_returns: np.ndarray,
    daily_returns: np.ndarray,
    macro_features: np.ndarray,
    monthly_ordinals: np.ndarray,
    daily_ns: np.ndarray,
    monthly_mask: np.ndarray,
    daily_mask: np.ndarray,
) -> CalibrationSlice:
    return CalibrationSlice(
        role=role,  # type: ignore[arg-type]
        monthly_returns=_immutable(monthly_returns),
        daily_returns=_immutable(daily_returns),
        macro_features=_immutable(macro_features),
        monthly_period_ordinals=_immutable(monthly_ordinals),
        daily_session_ns=_immutable(daily_ns),
        monthly_continuation=_immutable(monthly_mask),
        daily_continuation=_immutable(daily_mask),
    )


def _calibration_input(config_fingerprint: str) -> CalibrationInput:
    monthly_ordinals = np.r_[np.arange(459, 669), np.arange(670, 677)].astype(np.int64)
    design_daily = np.linspace(
        1_207_008_000_000_000_000,
        1_714_435_200_000_000_000,
        4_049,
        dtype=np.int64,
    )
    holdout_daily = np.linspace(
        1_714_521_600_000_000_000,
        1_780_012_800_000_000_000,
        498,
        dtype=np.int64,
    )
    daily_ns = np.r_[design_daily, holdout_daily]
    monthly_returns = np.arange(217 * 3, dtype=np.float64).reshape(217, 3) / 10_000
    daily_returns = np.arange(4_547 * 3, dtype=np.float64).reshape(4_547, 3) / 100_000
    macro_features = np.arange(217 * 4, dtype=np.float64).reshape(217, 4) / 100
    full_monthly_mask = _mask(217, (0, 210))
    full_daily_mask = _mask(4_547, (0, 4_404))
    design = _slice(
        "design",
        monthly_returns[:193],
        daily_returns[:4_049],
        macro_features[:193],
        monthly_ordinals[:193],
        daily_ns[:4_049],
        _mask(193, (0,)),
        _mask(4_049, (0,)),
    )
    holdout = _slice(
        "holdout",
        monthly_returns[193:],
        daily_returns[4_049:],
        macro_features[193:],
        monthly_ordinals[193:],
        daily_ns[4_049:],
        _mask(24, (0, 17)),
        _mask(498, (0, 355)),
    )
    full = _slice(
        "full",
        monthly_returns,
        daily_returns,
        macro_features,
        monthly_ordinals,
        daily_ns,
        full_monthly_mask,
        full_daily_mask,
    )
    geometry = SplitGeometry(
        holdout_months=24,
        full_monthly_rows=217,
        full_daily_rows=4_547,
        design_monthly_rows=193,
        design_daily_rows=4_049,
        holdout_monthly_rows=24,
        holdout_daily_rows=498,
        full_first_month_ordinal=459,
        full_last_month_ordinal=676,
        design_last_month_ordinal=651,
        holdout_first_month_ordinal=652,
        full_first_daily_session_ns=1_207_008_000_000_000_000,
        full_last_daily_session_ns=1_780_012_800_000_000_000,
        design_last_daily_session_ns=1_714_435_200_000_000_000,
        holdout_first_daily_session_ns=1_714_521_600_000_000_000,
    )
    source_identities = (
        SourceIdentity("asset:equity", "yfinance", "ACWI"),
        SourceIdentity("asset:muni_bond", "yfinance", "MUB"),
        SourceIdentity("asset:taxable_bond", "yfinance", "AGG"),
        SourceIdentity("macro:industrial_production", "fred", "INDPRO"),
        SourceIdentity("macro:inflation", "fred", "CPIAUCSL"),
        SourceIdentity("macro:yield_curve_spread", "fred", "T10Y3M"),
        SourceIdentity("macro:credit_spread", "fred", "BAA10Y"),
    )
    provenance = CalibrationProvenance(
        processed_data_fingerprint=(
            preflight_module.CANONICAL_PROCESSED_DATA_FINGERPRINT
        ),
        raw_snapshot_fingerprint=preflight_module.CANONICAL_RAW_SNAPSHOT_FINGERPRINT,
        processed_config_fingerprint=config_fingerprint,
        calibration_config_fingerprint=config_fingerprint,
        raw_acquisition_contract_sha256="a" * 64,
        data_dictionary_sha256="b" * 64,
        asset_columns=ASSET_COLUMNS,
        macro_columns=MACRO_COLUMNS,
        source_identities=source_identities,
        source_files=(),
    )
    provisional = CalibrationInput(
        fingerprint="0" * 64,
        provenance=provenance,
        geometry=geometry,
        full=full,
        design=design,
        holdout=holdout,
    )
    content = (
        json.dumps(
            provisional.identity_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return replace(provisional, fingerprint=hashlib.sha256(content).hexdigest())


def _runtime() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version=CANONICAL_PYTHON_VERSION,
        python_implementation="CPython",
        platform_system=CANONICAL_PLATFORM,
        machine=CANONICAL_MACHINE,
        dependency_versions=tuple(EXPECTED_RUNTIME_VERSIONS.items()),
        file_descriptor_soft_limit=1_024,
    )


def _rehash(value: CalibrationInput) -> CalibrationInput:
    content = (
        json.dumps(value.identity_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return replace(value, fingerprint=hashlib.sha256(content).hexdigest())


def _resources(**updates: int) -> ResourceSnapshot:
    values = {
        "logical_cpu_count": CANONICAL_LOGICAL_CPUS,
        "physical_memory_bytes": CANONICAL_PHYSICAL_MEMORY_BYTES,
        "disk_free_bytes": 20_000_000_000,
        "resolved_workers": 9,
        "reserve_cores": 1,
    }
    values.update(updates)
    return ResourceSnapshot(**values)


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[object, CalibrationInput]:
    config = load_config(ROOT / "configs/canonical.yaml")
    calibration_input = _calibration_input(config.fingerprint())
    monkeypatch.setattr(
        preflight_module,
        "CANONICAL_CALIBRATION_INPUT_FINGERPRINT",
        calibration_input.fingerprint,
    )
    return config, calibration_input


def test_canonical_preflight_hashes_all_contract_and_resource_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)

    result = canonical_g3_preflight(
        config,  # type: ignore[arg-type]
        calibration_input,
        _resources(),
        runtime=_runtime(),
    )

    assert len(result.checks) == 9
    assert result.workers == 9
    assert result.calibration_input_fingerprint == calibration_input.fingerprint
    assert len(result.contract_fingerprint) == 64
    assert len(result.source_code_fingerprint) == 64
    assert len(result.dependency_lock_fingerprint) == 64
    assert len(result.toolchain_fingerprint) == 64
    estimate = canonical_g3_resource_estimate()
    assert result.estimated_peak_bytes == estimate.estimated_peak_bytes
    assert result.estimated_disk_bytes == estimate.estimated_disk_bytes
    assert result.resource_estimate_fingerprint == estimate.fingerprint
    identity = result.identity_dict()
    assert identity["schema_version"] == 2
    assert identity["execution_closure_schema"] == G3_EXECUTION_CLOSURE_SCHEMA
    assert identity["execution_closure_manifest"] == (
        G3_EXECUTION_CLOSURE_MANIFEST.as_dict()
    )
    assert identity["execution_closure_manifest_fingerprint"] == (
        G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
    )
    assert (
        hashlib.sha256(
            (
                json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest()
        == result.fingerprint
    )
    bridge = canonical_bridge_resource_preflight(
        result,
        bridge_resource_preflight(
            benchmark_pair_seconds=np.full(100, 30.0),
            projected_pair_fits=20_500,
            workers=9,
            peak_memory_fraction=0.69,
        ),
    )
    assert bridge.benchmark_pairs == 100
    assert bridge.workers == 9
    assert len(bridge.fingerprint) == 64


def test_live_preflight_reverification_uses_scoped_g3_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    preflight = canonical_g3_preflight(
        config,  # type: ignore[arg-type]
        calibration_input,
        _resources(),
        runtime=_runtime(),
    )
    observed = preflight_module.inspect_g3_execution_closure()
    changed = replace(observed, source_code_fingerprint="f" * 64)
    monkeypatch.setattr(
        preflight_module,
        "inspect_g3_execution_closure",
        lambda: changed,
    )

    with pytest.raises(ModelError, match="contract fields"):
        verify_live_canonical_g3_preflight(preflight)


def test_bridge_stage_preflight_rejects_tamper_and_failed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    base = canonical_g3_preflight(
        config,  # type: ignore[arg-type]
        calibration_input,
        _resources(),
        runtime=_runtime(),
    )
    passed = bridge_resource_preflight(
        benchmark_pair_seconds=np.full(100, 30.0),
        projected_pair_fits=20_500,
        workers=9,
        peak_memory_fraction=0.69,
    )
    failed = bridge_resource_preflight(
        benchmark_pair_seconds=np.full(100, 200.0),
        projected_pair_fits=20_500,
        workers=9,
        peak_memory_fraction=0.71,
    )
    with pytest.raises(ModelError, match="hash is inconsistent"):
        canonical_bridge_resource_preflight(replace(base, fingerprint="f" * 64), passed)
    with pytest.raises(ModelError, match="failed closed"):
        canonical_bridge_resource_preflight(base, failed)
    with pytest.raises(ModelError, match="base-preflight"):
        canonical_bridge_resource_preflight(object(), passed)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="BridgeResourcePreflight"):
        canonical_bridge_resource_preflight(base, object())  # type: ignore[arg-type]


def test_live_runtime_inventory_matches_the_pinned_test_environment() -> None:
    runtime = inspect_runtime_environment()
    assert runtime.python_version == CANONICAL_PYTHON_VERSION
    assert runtime.python_implementation == "CPython"
    assert runtime.platform_system == CANONICAL_PLATFORM
    assert runtime.machine == CANONICAL_MACHINE
    assert runtime.dependency_versions == tuple(EXPECTED_RUNTIME_VERSIONS.items())
    assert dict(runtime.dependency_versions)["joblib"] == "1.5.3"
    assert runtime.file_descriptor_soft_limit > 0


def test_preflight_rejects_transitive_joblib_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    dependencies = dict(_runtime().dependency_versions)
    dependencies["joblib"] = "0.0.0"
    wrong_joblib = replace(_runtime(), dependency_versions=tuple(dependencies.items()))

    with pytest.raises(ModelError, match="pinned target-Mac"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            calibration_input,
            _resources(),
            runtime=wrong_joblib,
        )


def test_toolchain_fingerprint_binds_complete_locked_inventory() -> None:
    source = preflight_module.inspect_g3_execution_closure()
    runtime_fingerprint = "a" * 64
    changed_versions = tuple(
        (name, "0.0.0" if name == "joblib" else value)
        for name, value in source.dependency_versions
    )
    changed = replace(source, dependency_versions=changed_versions)

    assert preflight_module._toolchain_fingerprint(
        runtime_fingerprint, source
    ) != preflight_module._toolchain_fingerprint(runtime_fingerprint, changed)


def test_preflight_rejects_writable_or_nonsealed_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    writable = np.array(calibration_input.full.monthly_returns, copy=True)
    bad_full = replace(calibration_input.full, monthly_returns=writable)
    bad_input = replace(calibration_input, full=bad_full)

    with pytest.raises(ModelError, match="read-only"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            bad_input,
            _resources(),
            runtime=_runtime(),
        )

    changed = np.array(calibration_input.holdout.monthly_returns, copy=True)
    changed[0, 0] += 1.0
    sealed_bad = replace(
        calibration_input,
        holdout=replace(
            calibration_input.holdout,
            monthly_returns=_immutable(changed),
        ),
    )
    with pytest.raises(ModelError, match="exact sealed"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            sealed_bad,
            _resources(),
            runtime=_runtime(),
        )


@pytest.mark.parametrize(
    ("runtime", "resources", "message"),
    [
        (
            replace(_runtime(), python_version="3.11.14"),
            _resources(),
            "pinned target-Mac",
        ),
        (
            _runtime(),
            _resources(resolved_workers=8),
            "nine workers",
        ),
        (
            _runtime(),
            _resources(disk_free_bytes=12_000_000_000),
            "three times",
        ),
    ],
)
def test_preflight_environment_and_resource_failures_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    runtime: RuntimeEnvironment,
    resources: ResourceSnapshot,
    message: str,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    with pytest.raises(ModelError, match=message):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            calibration_input,
            resources,
            runtime=runtime,
        )


def test_code_owned_resource_estimate_fails_the_memory_gate_when_bound_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    monkeypatch.setattr(
        preflight_module,
        "_RESOURCE_CONTINGENCY_BYTES",
        5 * (1 << 30),
    )
    with pytest.raises(ModelError, match="70%"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            calibration_input,
            _resources(),
            runtime=_runtime(),
        )


def test_preflight_rejects_wrong_types_fingerprints_and_registered_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    resources = _resources()
    arguments = {"runtime": _runtime()}
    with pytest.raises(ModelError, match="resolved PRPGConfig"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            object(), calibration_input, resources, **arguments
        )
    with pytest.raises(ModelError, match="CalibrationInput"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, object(), resources, **arguments
        )
    with pytest.raises(ModelError, match="resource snapshot"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, calibration_input, object(), **arguments
        )

    wrong_lineage = replace(
        calibration_input,
        provenance=replace(
            calibration_input.provenance,
            processed_data_fingerprint="f" * 64,
        ),
    )
    with pytest.raises(ModelError, match="lineage"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            wrong_lineage,
            resources,
            **arguments,
        )

    monkeypatch.setattr(preflight_module, "RNG_CONTRACT_VERSION", 2)
    with pytest.raises(ModelError, match="RNG enum"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            calibration_input,
            resources,
            **arguments,
        )
    monkeypatch.setattr(preflight_module, "RNG_CONTRACT_VERSION", 1)
    monkeypatch.setattr(preflight_module, "HMM_RESTARTS", 49)
    with pytest.raises(ModelError, match="scientific constants"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            calibration_input,
            resources,
            **arguments,
        )


@pytest.mark.parametrize(
    ("constant", "wrong_value"),
    (
        ("OWNER_FIXED_K4_FIXED_POINT_STATE_CONTRACT", "wrong-state-contract"),
        ("MAXIMUM_FIXED_POINT_ITERATIONS", 4),
        ("BRIDGE_FIXED_K4_TIER_A_POLICY", "cross_k_rolling_v1"),
        ("BRIDGE_FIXED_K4_TIER_A_FITS_PER_MEMBER", 2),
        ("BRIDGE_FIXED_K4_TIER_A_ROLLING_FOLDS", 6),
        ("BRIDGE_FIXED_K4_TIER_A_USES_CROSS_K_SELECTION", True),
        ("BRIDGE_FIXED_K4_TIER_A_USES_BIC", True),
        ("BRIDGE_FIXED_K4_TIER_A_RESTARTS_PER_PAIR", 2_800),
        ("G3_RESOURCE_PROJECTION_VERSION", "stale-projection-v2"),
    ),
)
def test_preflight_binds_version_3_fixed_k4_contract_constants(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    wrong_value: object,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    monkeypatch.setattr(preflight_module, constant, wrong_value)

    with pytest.raises(ModelError, match="scientific constants"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            calibration_input,
            _resources(),
            runtime=_runtime(),
        )


def test_preflight_rejects_geometry_dtype_mask_index_and_memory_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, value = _fixture(monkeypatch)
    kwargs = {"runtime": _runtime()}

    bad_geometry = _rehash(
        replace(
            value,
            geometry=replace(value.geometry, full_monthly_rows=216),
        )
    )
    monkeypatch.setattr(
        preflight_module,
        "CANONICAL_CALIBRATION_INPUT_FINGERPRINT",
        bad_geometry.fingerprint,
    )
    with pytest.raises(ModelError, match="exact G2 geometry"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, bad_geometry, _resources(), **kwargs
        )
    monkeypatch.setattr(
        preflight_module,
        "CANONICAL_CALIBRATION_INPUT_FINGERPRINT",
        value.fingerprint,
    )

    float32 = _immutable(value.full.monthly_returns.astype(np.float32))
    bad_dtype = replace(value, full=replace(value.full, monthly_returns=float32))
    with pytest.raises(ModelError, match="binary64"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, bad_dtype, _resources(), **kwargs
        )

    nonfinite = np.array(value.full.monthly_returns, copy=True)
    nonfinite[0, 0] = np.nan
    bad_finite = replace(
        value,
        full=replace(value.full, monthly_returns=_immutable(nonfinite)),
    )
    with pytest.raises(ModelError, match="non-finite"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, bad_finite, _resources(), **kwargs
        )

    bad_mask = np.array(value.full.monthly_continuation, copy=True)
    bad_mask[1] = False
    bad_segments = replace(
        value,
        full=replace(value.full, monthly_continuation=_immutable(bad_mask)),
    )
    with pytest.raises(ModelError, match="segments"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, bad_segments, _resources(), **kwargs
        )

    bad_index = np.array(value.full.monthly_period_ordinals, copy=True)
    bad_index[1] = bad_index[0]
    nonmonotone = replace(
        value,
        full=replace(value.full, monthly_period_ordinals=_immutable(bad_index)),
    )
    with pytest.raises(ModelError, match="strictly increasing"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, nonmonotone, _resources(), **kwargs
        )

    shared_design = replace(
        value.design,
        monthly_returns=value.full.monthly_returns[:193],
    )
    aliased = replace(value, design=shared_design)
    with pytest.raises(ModelError, match="detached"):
        canonical_g3_preflight(  # type: ignore[arg-type]
            config, aliased, _resources(), **kwargs
        )


def test_resource_estimate_is_exact_fingerprinted_and_closed_to_nine_workers() -> None:
    estimate = canonical_g3_resource_estimate()
    assert estimate.workers == 9
    assert estimate.process_count == 10
    assert estimate.largest_registered_kernel_arrays_bytes == 559_243_008
    assert estimate.estimated_peak_bytes == sum(
        (
            estimate.parent_serial_workspace_bytes,
            estimate.python_process_allowance_bytes,
            estimate.largest_registered_kernel_arrays_bytes,
            estimate.nonkernel_stage_workspace_bytes,
            estimate.contingency_bytes,
        )
    )
    assert estimate.estimated_disk_bytes == 4 * (1 << 30)
    assert len(estimate.fingerprint) == 64
    with pytest.raises(ModelError, match="nine workers"):
        canonical_g3_resource_estimate(workers=8)
    with pytest.raises(ModelError, match="integer"):
        canonical_g3_resource_estimate(workers=True)  # type: ignore[arg-type]


def test_live_preflight_is_sealed_persisted_and_resume_stable_across_free_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    first = canonical_g3_preflight(
        config,  # type: ignore[arg-type]
        calibration_input,
        _resources(disk_free_bytes=20_000_000_000),
        runtime=_runtime(),
    )
    verify_live_canonical_g3_preflight(first)
    with pytest.raises(ModelError, match="builder authority"):
        verify_live_canonical_g3_preflight(replace(first))

    store = CalibrationRunStore(tmp_path / "runs")
    run = store.initialize(
        default_run_identity(
            config_fingerprint=first.config_fingerprint,
            processed_data_fingerprint=first.processed_data_fingerprint,
            calibration_input_fingerprint=first.calibration_input_fingerprint,
            source_code_fingerprint=first.source_code_fingerprint,
            dependency_lock_fingerprint=first.dependency_lock_fingerprint,
            workers=first.workers,
        )
    )
    receipt = persist_canonical_g3_preflight(
        store,
        run_id=run.run_id,
        preflight=first,
    )
    assert not receipt.reused
    stored = read_persisted_canonical_g3_preflight(store, run_id=run.run_id)
    assert stored.preflight.fingerprint == first.fingerprint
    assert stored.receipt_fingerprint == receipt.fingerprint
    with pytest.raises(ModelError, match="builder authority"):
        verify_live_canonical_g3_preflight(stored.preflight)

    store.plan_launch(
        run.run_id,
        preflight_fingerprint=first.fingerprint,
        workers=first.workers,
    )
    second = canonical_g3_preflight(
        config,  # type: ignore[arg-type]
        calibration_input,
        _resources(disk_free_bytes=21_000_000_000),
        runtime=_runtime(),
    )
    assert second.fingerprint != first.fingerprint
    assert (
        canonical_resume_preflight_fingerprint(
            store,
            run_id=run.run_id,
            live_preflight=second,
        )
        == first.fingerprint
    )


def test_preflight_persistence_rejects_run_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, calibration_input = _fixture(monkeypatch)
    preflight = canonical_g3_preflight(
        config,  # type: ignore[arg-type]
        calibration_input,
        _resources(),
        runtime=_runtime(),
    )
    store = CalibrationRunStore(tmp_path / "runs")
    run = store.initialize(
        default_run_identity(
            config_fingerprint="f" * 64,
            processed_data_fingerprint=preflight.processed_data_fingerprint,
            calibration_input_fingerprint=preflight.calibration_input_fingerprint,
            source_code_fingerprint=preflight.source_code_fingerprint,
            dependency_lock_fingerprint=preflight.dependency_lock_fingerprint,
            workers=9,
        )
    )
    with pytest.raises(ModelError, match="bindings disagree"):
        persist_canonical_g3_preflight(
            store,
            run_id=run.run_id,
            preflight=preflight,
        )


def test_preflight_rejects_insufficient_descriptor_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, value = _fixture(monkeypatch)
    with pytest.raises(ModelError, match="descriptor"):
        canonical_g3_preflight(
            config,  # type: ignore[arg-type]
            value,
            _resources(),
            runtime=replace(_runtime(), file_descriptor_soft_limit=99),
        )
