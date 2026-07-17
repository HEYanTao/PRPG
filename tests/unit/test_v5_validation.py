from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prpg.errors import IntegrityError
from prpg.simulation.rng import Domain, Family, RNGKey, Stage, uniforms
from prpg.storage.csv_v1 import format_csv_float64
from prpg.v5.config import load_v5_inputs
from prpg.v5.validation import (
    COARSE_HEADER,
    DAILY_HEADER,
    _compounding_error,
    _risk_gate,
    _ValidationReference,
    validate_v5_run,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = load_v5_inputs(ROOT / "configs" / "v5-canonical.yaml")
ZERO_FINGERPRINT = "0" * 64
STATE_POOLS = tuple(
    np.asarray(np.arange(state, 219, 4), dtype=np.int64) for state in range(4)
)


@dataclass(frozen=True)
class _Selection:
    states: np.ndarray[Any, Any]
    source_month_indices: np.ndarray[Any, Any]


@dataclass(frozen=True)
class _Path:
    selection: _Selection
    month_lengths: np.ndarray[Any, Any]
    daily_log_returns: np.ndarray[Any, Any]
    monthly_log_returns: np.ndarray[Any, Any]
    quarterly_log_returns: np.ndarray[Any, Any]
    annual_log_returns: np.ndarray[Any, Any]


def _path(entity: int, *, duplicate: bool = False) -> _Path:
    months = 600
    lengths = np.full(months, 2, dtype=np.int64)
    row = np.arange(months * 2, dtype=np.float64)
    offset = 0.0 if duplicate else entity * 1.0e-8
    daily = np.column_stack(
        (
            np.sin(row / 17.0) * 0.003 + offset,
            np.cos(row / 23.0) * 0.002 - offset,
            np.sin(row / 31.0 + 0.4) * 0.0025 + 0.5 * offset,
        )
    )
    monthly = daily.reshape(months, 2, 3).sum(axis=1)
    quarterly = monthly.reshape(-1, 3, 3).sum(axis=1)
    annual = monthly.reshape(-1, 12, 3).sum(axis=1)
    states = np.arange(months, dtype=np.int64) % 4
    key = RNGKey(
        master_seed=INPUTS.seed_registry.master_seed,
        scientific_version=INPUTS.seed_registry.scientific_version,
        domain=Domain.PRODUCTION,
        family=Family.NOT_APPLICABLE,
        entity=entity,
        stage=Stage.SOURCE_RANK_UNIFORMS,
    )
    sources = np.empty(months, dtype=np.int64)
    for month, (state, draw) in enumerate(
        zip(states, uniforms(key, months), strict=True)
    ):
        pool = STATE_POOLS[int(state)]
        sources[month] = pool[min(int(float(draw) * len(pool)), len(pool) - 1)]
    return _Path(
        _Selection(states, sources),
        lengths,
        daily,
        monthly,
        quarterly,
        annual,
    )


def _write_csv(path: Path, frequency: str, values: list[_Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = bytearray(DAILY_HEADER if frequency == "daily" else COARSE_HEADER)
    periods_per_year = {"monthly": 12, "quarterly": 4, "annual": 1}
    for entity, materialized in enumerate(values, start=1):
        path_id = f"P{entity:06d}"
        if frequency == "daily":
            cursor = 0
            period = 1
            for month, length in enumerate(materialized.month_lengths):
                for session in range(int(length)):
                    vector = materialized.daily_log_returns[cursor]
                    fields = (
                        path_id,
                        str(period),
                        str(month // 12 + 1),
                        str(month % 12 + 1),
                        str(session + 1),
                        *(format_csv_float64(item) for item in vector),
                    )
                    content.extend((",".join(fields) + "\n").encode())
                    cursor += 1
                    period += 1
            continue
        array = getattr(materialized, f"{frequency}_log_returns")
        per_year = periods_per_year[frequency]
        for index, vector in enumerate(array):
            fields = (
                path_id,
                str(index + 1),
                str(index // per_year + 1),
                str(index % per_year + 1),
                *(format_csv_float64(item) for item in vector),
            )
            content.extend((",".join(fields) + "\n").encode())
    path.write_bytes(bytes(content))


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _publish_bundle(root: Path, values: list[_Path]) -> None:
    count = len(values)
    files: list[dict[str, object]] = []
    by_frequency: dict[str, dict[str, int]] = {}
    for frequency in ("daily", "monthly", "quarterly", "annual"):
        relative = (
            f"consumer/returns/training/{frequency}/"
            f"part-00000-path-P000001-P{count:06d}.csv"
        )
        path = root / relative
        _write_csv(path, frequency, values)
        content = path.read_bytes()
        rows = content.count(b"\n") - 1
        record = {
            "frequency": frequency,
            "relative_path": relative,
            "rows": rows,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        files.append(record)
        by_frequency[frequency] = {"files": 1, "rows": rows, "bytes": len(content)}
    checksums = "".join(
        f"{item['sha256']}  {item['relative_path']}\n"
        for item in sorted(files, key=lambda row: str(row["relative_path"]))
    ).encode()
    (root / "checksums.sha256").write_bytes(checksums)
    daily_counts = [len(item.daily_log_returns) for item in values]
    plan = {
        "profile": "smoke",
        "horizon_months": 600,
        "paths_per_shard": count,
        "path_count": count,
        "path_ranges": [
            {
                "role": "training",
                "first_path": 1,
                "last_path": count,
                "path_count": count,
            }
        ],
        "work_unit_count": 1,
        "csv_file_count": 4,
    }
    identity = {
        "schema_id": "prpg-v5-generation-manifest-v1",
        "status": "complete",
        "config_fingerprint": INPUTS.config_fingerprint,
        "seed_registry_fingerprint": INPUTS.seed_registry_fingerprint,
        "processed_data_fingerprint": ZERO_FINGERPRINT,
        "model_fingerprint": ZERO_FINGERPRINT,
        "approved_model_fingerprint": ZERO_FINGERPRINT,
        "plan": plan,
        "work_units": [
            {
                "work_unit_id": f"training-00000-p000001-p{count:06d}",
                "role": "training",
                "shard_index": 0,
                "first_path": 1,
                "last_path": count,
                "path_count": count,
                "daily_rows_minimum": min(daily_counts),
                "daily_rows_maximum": max(daily_counts),
                "daily_rows_sum": sum(daily_counts),
                "files": files,
            }
        ],
        "totals": {
            "paths": count,
            "work_units": 1,
            "csv_files": 4,
            "rows": sum(int(item["rows"]) for item in files),
            "bytes": sum(int(item["bytes"]) for item in files),
            "by_frequency": by_frequency,
            "daily_rows_per_path": {
                "minimum": min(daily_counts),
                "maximum": max(daily_counts),
                "mean": sum(daily_counts) / count,
            },
        },
        "checksums": {
            "relative_path": "checksums.sha256",
            "entries": 4,
            "sha256": hashlib.sha256(checksums).hexdigest(),
        },
    }
    manifest = {
        **identity,
        "manifest_fingerprint": hashlib.sha256(_canonical(identity)).hexdigest(),
    }
    (root / "manifest.json").write_bytes(_canonical(manifest))


def _reference(values: _Path) -> _ValidationReference:
    transition = np.roll(np.eye(4, dtype=np.float64), -1, axis=1)
    return _ValidationReference(
        values.daily_log_returns,
        transition,
        np.full(4, 0.25, dtype=np.float64),
        STATE_POOLS,
    )


def test_streaming_smoke_bundle_passes_and_reports_consumer_counts(
    tmp_path: Path,
) -> None:
    values = [_path(1)]
    _publish_bundle(tmp_path, values)

    report = validate_v5_run(
        INPUTS,
        tmp_path,
        expected_paths=1,
        _replay_path=lambda _entity: values[0],
        _reference=_reference(values[0]),
    )

    assert report.status == "passed"
    assert report.paths == 1
    assert report.csv_files == 4
    assert report.maximum_source_replay_error <= 1.0e-12
    assert report.maximum_log_compounding_error <= 1.0e-12
    assert report.as_dict()["report_only_deferred"] == [
        "tails_drawdowns_acf",
        "source_use_and_repeated_subsequences",
        "quarterly_price_only_strategy_diagnostics",
    ]


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    values = [_path(1)]
    _publish_bundle(tmp_path, values)
    monthly = next((tmp_path / "consumer").rglob("monthly/*.csv"))
    monthly.write_bytes(monthly.read_bytes() + b"0")

    with pytest.raises(IntegrityError, match="checksum|excess"):
        validate_v5_run(
            INPUTS,
            tmp_path,
            expected_paths=1,
            _replay_path=lambda _entity: values[0],
            _reference=_reference(values[0]),
        )


def test_wrong_aggregate_and_science_gate_fail_directly() -> None:
    values = _path(1)
    wrong = values.monthly_log_returns.copy()
    wrong[0, 0] += 0.01
    assert (
        _compounding_error(
            values.daily_log_returns,
            values.month_lengths,
            wrong,
            values.quarterly_log_returns,
            values.annual_log_returns,
        )
        > 1.0e-12
    )
    with pytest.raises(IntegrityError, match="covariance plausibility"):
        _risk_gate(
            INPUTS,
            values.daily_log_returns,
            np.cov(values.daily_log_returns, rowvar=False) * 4.0,
            enforce=True,
        )


def test_future_source_field_header_is_rejected(tmp_path: Path) -> None:
    values = [_path(1)]
    _publish_bundle(tmp_path, values)
    daily = next((tmp_path / "consumer").rglob("daily/*.csv"))
    lines = daily.read_bytes().splitlines(keepends=True)
    lines[0] = lines[0].rstrip(b"\n") + b",source_month\n"
    daily.write_bytes(b"".join(lines))

    with pytest.raises(IntegrityError, match="header"):
        validate_v5_run(
            INPUTS,
            tmp_path,
            expected_paths=1,
            _replay_path=lambda _entity: values[0],
            _reference=_reference(values[0]),
        )


def test_historical_successor_source_sequence_is_rejected(tmp_path: Path) -> None:
    valid = _path(1)
    successor_sources = valid.selection.source_month_indices.copy()
    for month in range(1, len(successor_sources)):
        successor_sources[month] = (successor_sources[month - 1] + 1) % 219
    injected = replace(
        valid,
        selection=_Selection(valid.selection.states, successor_sources),
    )
    _publish_bundle(tmp_path, [injected])

    with pytest.raises(IntegrityError, match="source-month selection"):
        validate_v5_run(
            INPUTS,
            tmp_path,
            expected_paths=1,
            _replay_path=lambda _entity: injected,
            _reference=_reference(valid),
        )
