from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prpg.simulation.serial as serial_module
import prpg.validation.structural as structural_module
from prpg.errors import GenerationError, IntegrityError
from prpg.simulation.aggregation import (
    aggregate_hf_weekly_log_returns,
    aggregate_lf_log_returns,
)
from prpg.simulation.serial import (
    SERIAL_BASE_PATH_SCHEMA_VERSION,
    SERIAL_GENERATOR_VERSION,
    RestartAudit,
    RestartReason,
    SerialBasePath,
    _array_fingerprint,
    _immutable_array,
    _record_fingerprint,
    is_sealed_serial_base_path,
)
from prpg.storage.group_commit import (
    StorageCommitBinding,
    commit_hf_serial_work_unit,
    commit_lf_serial_work_unit,
    serial_work_unit_id,
)
from prpg.validation.structural import (
    CANONICAL_STRUCTURAL_PLAN,
    StructuralPathView,
    StructuralValidationPlan,
    estimate_canonical_scan_seconds,
    is_sealed_structural_validation_report,
    publish_consumer_manifest_from_receipts,
    validate_and_publish_data_validated,
    validate_consumer_structure,
)


def _sealed_path(family: str, entity: int, years: int) -> SerialBasePath:
    monthly_count = 12 * years
    count = monthly_count if family == "LF" else 252 * years
    frequency = "monthly" if family == "LF" else "daily"
    values = (
        np.arange(count * 3, dtype=np.float64).reshape(count, 3) + entity * 1_000
    ) * 1e-7
    monthly_states = np.zeros(monthly_count, dtype=np.int64)
    target_states = monthly_states if family == "LF" else np.repeat(monthly_states, 21)
    restart_flags = np.zeros(count, dtype=np.bool_)
    restart_flags[0] = True
    restart_used = np.ones(count, dtype=np.bool_)
    restart_used[0] = False
    rank_used = np.zeros(count, dtype=np.bool_)
    rank_used[0] = True
    reasons = (
        RestartReason.FIRST_OBSERVATION,
        *(RestartReason.CONTINUATION for _ in range(count - 1)),
    )
    audit = RestartAudit(
        restart_flags=_immutable_array(
            restart_flags, dtype=np.bool_, label="restart flags"
        ),
        reasons=reasons,
        restart_uniform_used=_immutable_array(
            restart_used, dtype=np.bool_, label="restart use"
        ),
        source_rank_uniform_used=_immutable_array(
            rank_used, dtype=np.bool_, label="rank use"
        ),
        restart_uniforms_consumed=count,
        source_rank_uniforms_consumed=count,
    )
    arrays = {
        "log_returns": _immutable_array(values, dtype=np.float64, label="log returns"),
        "target_monthly_states": _immutable_array(
            monthly_states, dtype=np.int64, label="monthly states"
        ),
        "target_states": _immutable_array(
            target_states, dtype=np.int64, label="target states"
        ),
        "source_indices": _immutable_array(
            np.zeros(count, dtype=np.int64),
            dtype=np.int64,
            label="source indices",
        ),
        "restart_flags": audit.restart_flags,
        "restart_uniform_used": audit.restart_uniform_used,
        "source_rank_uniform_used": audit.source_rank_uniform_used,
    }
    metadata = {
        "schema_version": SERIAL_BASE_PATH_SCHEMA_VERSION,
        "generator_version": SERIAL_GENERATOR_VERSION,
        "execution_mode": "canonical",
        "generation_authorized": True,
        "generation_input_fingerprint": "a" * 64,
        "family": family,
        "base_frequency": frequency,
        "path_entity": entity,
        "years": years,
        "state_uniforms_consumed": monthly_count,
        "restart_uniforms_consumed": count,
        "source_rank_uniforms_consumed": count,
        "restart_reasons": [value.value for value in reasons],
    }
    fingerprint = _record_fingerprint(
        metadata,
        {
            name: _array_fingerprint(value, name)
            for name, value in sorted(arrays.items())
        },
    )
    result = object.__new__(SerialBasePath)
    for name, value in (
        ("schema_version", SERIAL_BASE_PATH_SCHEMA_VERSION),
        ("generator_version", SERIAL_GENERATOR_VERSION),
        ("execution_mode", "canonical"),
        ("generation_authorized", True),
        ("generation_input_fingerprint", "a" * 64),
        ("family", family),
        ("base_frequency", frequency),
        ("path_entity", entity),
        ("years", years),
        ("log_returns", arrays["log_returns"]),
        ("target_monthly_states", arrays["target_monthly_states"]),
        ("target_states", arrays["target_states"]),
        ("source_indices", arrays["source_indices"]),
        ("restart_audit", audit),
        ("state_uniforms_consumed", monthly_count),
        ("fingerprint", fingerprint),
        ("_seal", serial_module._BASE_PATH_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    assert is_sealed_serial_base_path(result)
    return result


_SMALL_PLAN = StructuralValidationPlan(
    profile_id="test-1y",
    years=1,
    lf_paths=2,
    hf_paths=1,
    lf_paths_per_shard=2,
    hf_paths_per_shard=1,
)
_BINDING = StorageCommitBinding(
    simulation_fingerprint="1" * 64,
    materialization_fingerprint="2" * 64,
    toolchain_fingerprint="3" * 64,
    run_fingerprint="4" * 64,
)


def _make_small_bundle(root: Path) -> None:
    lf = tuple(_sealed_path("LF", entity, 1) for entity in (1, 2))
    lf_aggregates = tuple(aggregate_lf_log_returns(path) for path in lf)
    hf = (_sealed_path("HF", 1, 1),)
    hf_weekly = (aggregate_hf_weekly_log_returns(hf[0]),)
    commit_lf_serial_work_unit(
        root,
        binding=_BINDING,
        work_unit_id=serial_work_unit_id("LF", 0, 1, 2),
        worker_identity="test-worker",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=2,
        expected_path_count=2,
        base_paths=lf,
        aggregates=lf_aggregates,
    )
    commit_hf_serial_work_unit(
        root,
        binding=_BINDING,
        work_unit_id=serial_work_unit_id("HF", 0, 1, 1),
        worker_identity="test-worker",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        expected_path_count=1,
        base_paths=hf,
        weekly_returns=hf_weekly,
    )
    publish_consumer_manifest_from_receipts(root, plan=_SMALL_PLAN, binding=_BINDING)


class _CountingHook:
    name = "counting"

    def __init__(self) -> None:
        self.paths = 0
        self.rows = 0
        self.frequencies: set[str] = set()

    def observe_path(self, path: StructuralPathView) -> None:
        assert not path.log_returns.flags.writeable
        self.paths += 1
        self.rows += path.log_returns.shape[0]
        self.frequencies.add(path.frequency)

    def finalize(self) -> dict[str, Any]:
        return {
            "paths": self.paths,
            "rows": self.rows,
            "frequencies": sorted(self.frequencies),
        }


def test_canonical_exact_plan_is_frozen() -> None:
    plan = CANONICAL_STRUCTURAL_PLAN
    assert plan.years == 50
    assert plan.lf_paths == 5_000
    assert plan.hf_paths == 1_000
    assert plan.lf_paths_per_shard == 100
    assert plan.hf_paths_per_shard == 50
    assert plan.work_unit_count == 70
    assert plan.csv_shard_count == 190
    assert plan.total_rows == 19_450_000
    assert plan.total_values == 58_350_000
    assert len(plan.fingerprint) == 64
    scopes = structural_module._work_unit_scopes(plan)
    assert (
        scopes[0].family,
        scopes[0].shard_index,
        scopes[0].first,
        scopes[0].last,
    ) == (
        "LF",
        0,
        1,
        100,
    )
    assert (
        scopes[49].family,
        scopes[49].shard_index,
        scopes[49].first,
        scopes[49].last,
    ) == ("LF", 49, 4_901, 5_000)
    assert (
        scopes[50].family,
        scopes[50].shard_index,
        scopes[50].first,
        scopes[50].last,
    ) == ("HF", 0, 1, 50)
    assert (
        scopes[-1].family,
        scopes[-1].shard_index,
        scopes[-1].first,
        scopes[-1].last,
    ) == ("HF", 19, 951, 1_000)
    assert structural_module._dataset_summaries(plan) == [
        {
            "family": "LF",
            "frequency": "monthly",
            "paths": 5_000,
            "periods_per_path": 600,
            "rows": 3_000_000,
            "values": 9_000_000,
            "shards": 50,
        },
        {
            "family": "LF",
            "frequency": "quarterly",
            "paths": 5_000,
            "periods_per_path": 200,
            "rows": 1_000_000,
            "values": 3_000_000,
            "shards": 50,
        },
        {
            "family": "LF",
            "frequency": "annual",
            "paths": 5_000,
            "periods_per_path": 50,
            "rows": 250_000,
            "values": 750_000,
            "shards": 50,
        },
        {
            "family": "HF",
            "frequency": "daily",
            "paths": 1_000,
            "periods_per_path": 12_600,
            "rows": 12_600_000,
            "values": 37_800_000,
            "shards": 20,
        },
        {
            "family": "HF",
            "frequency": "weekly",
            "paths": 1_000,
            "periods_per_path": 2_600,
            "rows": 2_600_000,
            "values": 7_800_000,
            "shards": 20,
        },
    ]


def test_small_bundle_streams_all_rows_aggregations_and_metric_hooks(
    tmp_path: Path,
) -> None:
    _make_small_bundle(tmp_path)
    hook = _CountingHook()

    report = validate_consumer_structure(tmp_path, plan=_SMALL_PLAN, hooks=(hook,))

    assert is_sealed_structural_validation_report(report)
    assert report.total_rows == 338
    assert report.total_values == 1_014
    assert report.roundtrip_values_checked == 1_014
    assert report.csv_shards == 5
    assert report.work_units == 2
    assert report.max_abs_log_aggregation_error <= 1e-12
    assert report.max_abs_simple_compounding_error <= 1e-10
    assert report.max_abs_log_simple_roundtrip_error <= 1e-10
    assert hook.paths == 8
    assert hook.rows == 338
    assert hook.frequencies == {"monthly", "quarterly", "annual", "daily", "weekly"}
    assert report.hook_metrics["counting"]["paths"] == 8  # type: ignore[index]
    assert report.rows_per_second > 0.0
    estimate = estimate_canonical_scan_seconds(report)
    assert math.isfinite(estimate) and estimate > 0.0
    with pytest.raises(FrozenInstanceError):
        report.total_rows = 1  # type: ignore[misc]


def test_hidden_consumer_file_and_truncated_shard_fail_closed(tmp_path: Path) -> None:
    _make_small_bundle(tmp_path)
    hidden = tmp_path / "consumer" / ".DS_Store"
    hidden.write_bytes(b"hidden")
    with pytest.raises(IntegrityError, match="allowlist"):
        validate_consumer_structure(tmp_path, plan=_SMALL_PLAN)
    hidden.unlink()

    manifest = json.loads(
        (tmp_path / "consumer" / "consumer-manifest.json").read_bytes()
    )
    shard = tmp_path / manifest["shards"][0]["relative_path"]
    shard.write_bytes(shard.read_bytes()[:-8])
    with pytest.raises(IntegrityError):
        validate_consumer_structure(tmp_path, plan=_SMALL_PLAN)


def test_independent_cross_frequency_recomputation_rejects_false_derived_rows() -> None:
    lf = _sealed_path("LF", 1, 1)
    lf_derived = aggregate_lf_log_returns(lf)
    quarterly = lf_derived.quarterly_log_returns.copy()
    quarterly[0, 0] += 2e-12
    errors = structural_module._ErrorAccumulator()
    with pytest.raises(IntegrityError, match="log sum"):
        structural_module._validate_lf_path(
            lf.log_returns,
            quarterly,
            lf_derived.annual_log_returns,
            1,
            errors,
        )

    hf = _sealed_path("HF", 1, 1)
    weekly = aggregate_hf_weekly_log_returns(hf).weekly_log_returns.copy()
    weekly[0, 1] += 2e-12
    with pytest.raises(IntegrityError, match="log sum"):
        structural_module._validate_hf_path(
            hf.log_returns, weekly, 1, structural_module._ErrorAccumulator()
        )


@pytest.mark.parametrize(
    "replacement",
    [
        b"LF000001,1,1,1,-0,0,0\n",
        b"LF000001,1,1,1,NaN,0,0\n",
        b"LF000001,2,1,2,0,0,0\n",
        b"LF000001,1,1,1,1.0,0,0\n",
    ],
)
def test_row_parser_rejects_signed_zero_nonfinite_bad_index_and_float_text(
    tmp_path: Path, replacement: bytes
) -> None:
    relative = "consumer/returns/low/monthly/part-00000-path-LF000001-LF000001.csv"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    content = structural_module.CSV_HEADER_BYTES + replacement
    path.write_bytes(content)
    record: dict[str, structural_module.JSONValue] = {
        "relative_path": relative,
        "family": "LF",
        "frequency": "monthly",
        "periods_per_path": 1,
        "rows": 1,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    stream = structural_module._CSVShardStream(tmp_path, record)
    try:
        with pytest.raises(IntegrityError):
            stream.read_path(1)
    finally:
        stream.close()


def test_untyped_data_validated_publication_is_disabled(tmp_path: Path) -> None:
    _make_small_bundle(tmp_path)
    with pytest.raises(GenerationError, match="coordinated typed G7"):
        validate_and_publish_data_validated(
            tmp_path,
            plan=_SMALL_PLAN,
            scientific_pass_fingerprint="e" * 64,
        )
    assert not (tmp_path / "DATA_VALIDATED").exists()


def test_invalid_science_fingerprint_creates_nothing(tmp_path: Path) -> None:
    _make_small_bundle(tmp_path)
    with pytest.raises(GenerationError, match="coordinated typed G7"):
        validate_and_publish_data_validated(
            tmp_path,
            plan=CANONICAL_STRUCTURAL_PLAN,
            scientific_pass_fingerprint="not-a-fingerprint",
        )
    assert not (tmp_path / "validation").exists()
    assert not (tmp_path / "DATA_VALIDATED").exists()
