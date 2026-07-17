from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prpg.config import PRPGConfig, load_config
from prpg.data.processed import (
    MANIFEST_NAME,
    ProcessedSnapshotStore,
    _canonical_json_bytes,
    _json_value,
    build_processed_snapshot,
    load_processed_arrays,
)
from prpg.data.providers import AcquisitionAttempt, ProviderPayload
from prpg.data.snapshot import SnapshotStore
from prpg.data.transform import expected_xnys_sessions
from prpg.errors import DataValidationError, IntegrityError

ROOT = Path(__file__).parents[2]


def _tiny_config(tmp_path: Path) -> PRPGConfig:
    config = load_config(ROOT / "configs" / "canonical.yaml")
    quality = config.data.quality.model_copy(
        update={
            "minimum_common_monthly_vectors": 1,
            "minimum_common_daily_vectors": 1,
        }
    )
    yfinance = config.data.yfinance.model_copy(
        update={
            "start": date(2023, 1, 1),
            "end_exclusive": date(2024, 4, 1),
        }
    )
    data = config.data.model_copy(
        update={
            "artifact_root": tmp_path / "artifacts",
            "cutoff": date(2024, 3, 31),
            "quality": quality,
            "yfinance": yfinance,
        }
    )
    model = config.model.model_copy(update={"design_holdout_months": 1})
    return config.model_copy(update={"data": data, "model": model})


def _attempt() -> tuple[AcquisitionAttempt, ...]:
    return (
        AcquisitionAttempt(
            1,
            "2024-04-01T00:00:00+00:00",
            "2024-04-01T00:00:01+00:00",
            "success",
        ),
    )


def _yahoo_payload(ticker: str, sessions: pd.DatetimeIndex) -> ProviderPayload:
    prices = 100.0 * np.exp(0.001 * np.arange(len(sessions)))
    frame = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Adj Close": prices,
            "Volume": 1000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=sessions.tz_localize("America/New_York"),
    )
    frame.index.name = "Date"
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index_label="Date",
        lineterminator="\n",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.17g",
    )
    return ProviderPayload(
        "yfinance",
        ticker,
        frame,
        buffer.getvalue().encode(),
        "2024-04-01T00:00:01+00:00",
        {"ticker": ticker},
        "test",
        _attempt(),
    )


def _fred_payload(
    series_id: str, dates: pd.DatetimeIndex, values: np.ndarray
) -> ProviderPayload:
    frame = pd.DataFrame({series_id: values}, index=dates)
    frame.index.name = "DATE"
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index_label="observation_date", lineterminator="\n")
    return ProviderPayload(
        "fred",
        series_id,
        frame,
        buffer.getvalue().encode(),
        "2024-04-01T00:00:01+00:00",
        {"series_id": series_id},
        "test",
        _attempt(),
    )


def _raw_snapshot(config: PRPGConfig) -> str:
    sessions = expected_xnys_sessions(date(2024, 1, 1), date(2024, 3, 31))
    months = pd.date_range("2023-01-01", "2024-03-01", freq="MS")
    weekdays = pd.bdate_range("2023-01-01", "2024-03-31")
    payloads = [
        *[_yahoo_payload(ticker, sessions) for ticker in ("ACWI", "MUB", "AGG")],
        _fred_payload("INDPRO", months, 100.0 * np.exp(0.01 * np.arange(15))),
        _fred_payload("CPIAUCSL", months, 200.0 * np.exp(0.005 * np.arange(15))),
        _fred_payload("T10Y3M", weekdays, np.linspace(0.5, 1.0, len(weekdays))),
        _fred_payload("BAA10Y", weekdays, np.linspace(2.0, 2.5, len(weekdays))),
    ]
    reference = SnapshotStore(config.data.artifact_root).create(
        payloads,
        acquisition_contract={"test": "processed"},
        acquisition_group_id="test-group",
    )
    return reference.fingerprint


def test_processed_store_is_content_addressed_and_timestamp_independent(
    tmp_path: Path,
) -> None:
    files = {"tables/a.csv": b"x\n1\n", "arrays/a.npy": b"npy-bytes"}
    first = ProcessedSnapshotStore(
        tmp_path,
        now=lambda: datetime(2024, 1, 1, tzinfo=UTC),
    ).create(
        raw_snapshot_fingerprint="a" * 64,
        transform_contract={"version": 1},
        files=files,
        qc_report={"status": "passed"},
        audit_metadata={"config": "one"},
    )
    second = ProcessedSnapshotStore(
        tmp_path,
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    ).create(
        raw_snapshot_fingerprint="a" * 64,
        transform_contract={"version": 1},
        files=files,
        qc_report={"status": "passed"},
        audit_metadata={"config": "two"},
    )

    assert not first.reused
    assert second.reused
    assert first.fingerprint == second.fingerprint
    assert second.manifest["audit_metadata"] == {"config": "one"}


def test_processed_store_detects_tamper_and_unmanifested_file(tmp_path: Path) -> None:
    store = ProcessedSnapshotStore(tmp_path)
    reference = store.create(
        raw_snapshot_fingerprint="a" * 64,
        transform_contract={"version": 1},
        files={"tables/a.csv": b"x\n1\n"},
        qc_report={"status": "passed"},
        audit_metadata={},
    )
    table = reference.path / "tables" / "a.csv"
    table.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="integrity check failed"):
        store.verify(reference.fingerprint)

    table.write_bytes(b"x\n1\n")
    (reference.path / "extra").write_text("unexpected", encoding="utf-8")
    with pytest.raises(IntegrityError, match="unmanifested"):
        store.verify(reference.fingerprint)


def test_processed_store_rejects_empty_inputs_and_bad_lookup(tmp_path: Path) -> None:
    store = ProcessedSnapshotStore(tmp_path)
    common = {
        "raw_snapshot_fingerprint": "a" * 64,
        "transform_contract": {},
        "qc_report": {},
        "audit_metadata": {},
    }
    with pytest.raises(DataValidationError, match="requires files"):
        store.create(files={}, **common)
    with pytest.raises(DataValidationError, match="file is empty"):
        store.create(files={"table.csv": b""}, **common)
    with pytest.raises(IntegrityError, match="fingerprint is invalid"):
        store.verify("not-a-fingerprint")
    with pytest.raises(IntegrityError, match="directory is missing"):
        store.verify("f" * 64)


def _basic_processed(store: ProcessedSnapshotStore):
    return store.create(
        raw_snapshot_fingerprint="a" * 64,
        transform_contract={"version": 1},
        files={"tables/a.csv": b"x\n1\n"},
        qc_report={"status": "passed"},
        audit_metadata={},
    )


def _write_manifest(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: [], "root is invalid"),
        (
            lambda manifest: {**manifest, "schema_version": 999},
            "identity fields are invalid",
        ),
        (lambda manifest: {**manifest, "identity": None}, "identity is missing"),
        (
            lambda manifest: {
                **manifest,
                "identity": {
                    **manifest["identity"],
                    "raw_snapshot_fingerprint": "b" * 64,
                },
            },
            "identity hash mismatch",
        ),
    ],
)
def test_processed_manifest_structural_corruption_fails_closed(
    tmp_path: Path, mutation: object, message: str
) -> None:
    store = ProcessedSnapshotStore(tmp_path)
    reference = _basic_processed(store)
    manifest_path = reference.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = mutation(manifest)  # type: ignore[operator]
    _write_manifest(manifest_path, changed)

    with pytest.raises(IntegrityError, match=message):
        store.verify(reference.fingerprint)


def test_processed_manifest_unreadable_and_missing_fail_closed(tmp_path: Path) -> None:
    store = ProcessedSnapshotStore(tmp_path)
    reference = _basic_processed(store)
    manifest_path = reference.path / MANIFEST_NAME
    manifest_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(IntegrityError, match="cannot be read"):
        store.verify(reference.fingerprint)

    manifest_path.unlink()
    with pytest.raises(IntegrityError, match="manifest is missing"):
        store.verify(reference.fingerprint)


def _relocate_for_identity(
    store: ProcessedSnapshotStore,
    reference: object,
    identity: object,
) -> tuple[str, Path]:
    old_path = reference.path  # type: ignore[attr-defined]
    manifest_path = old_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_bytes = (
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    fingerprint = hashlib.sha256(identity_bytes).hexdigest()
    manifest["identity"] = identity
    manifest["data_fingerprint"] = fingerprint
    _write_manifest(manifest_path, manifest)
    new_path = store.store_root / fingerprint
    old_path.rename(new_path)
    return fingerprint, new_path


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (None, "file list is invalid"),
        (["not-a-record"], "file record is invalid"),
        ([{"relative_path": 4}], "relative path is invalid"),
        ([{"relative_path": "../escape"}], "relative path is unsafe"),
    ],
)
def test_processed_identity_file_records_fail_closed(
    tmp_path: Path, files: object, message: str
) -> None:
    store = ProcessedSnapshotStore(tmp_path)
    reference = _basic_processed(store)
    identity = dict(reference.manifest["identity"])
    identity["files"] = files
    fingerprint, _ = _relocate_for_identity(store, reference, identity)

    with pytest.raises(IntegrityError, match=message):
        store.verify(fingerprint)


def test_processed_missing_payload_and_parent_symlink_fail_closed(
    tmp_path: Path,
) -> None:
    store = ProcessedSnapshotStore(tmp_path)
    reference = _basic_processed(store)
    payload = reference.path / "tables" / "a.csv"
    payload.unlink()
    with pytest.raises(IntegrityError, match="file is missing"):
        store.verify(reference.fingerprint)

    payload.parent.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.csv").write_bytes(b"x\n1\n")
    (reference.path / "tables").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError, match="escapes root"):
        store.verify(reference.fingerprint)


def test_json_normalization_handles_models_paths_dates_and_fallback(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs" / "canonical.yaml")

    normalized = _json_value(
        {
            "model": config.data.assets,
            "items": (Path("a/b"), date(2024, 1, 1), object()),
        }
    )

    assert normalized["model"]["equity"] == "ACWI"
    assert normalized["items"][0] == "a/b"
    assert normalized["items"][1] == "2024-01-01"
    assert normalized["items"][2].startswith("<object object at")
    assert _canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'


@pytest.mark.parametrize("path", ["../escape", "/absolute", r"bad\windows"])
def test_processed_store_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(DataValidationError, match="unsafe"):
        ProcessedSnapshotStore(tmp_path).create(
            raw_snapshot_fingerprint="a" * 64,
            transform_contract={},
            files={path: b"value"},
            qc_report={},
            audit_metadata={},
        )


def test_offline_processed_build_has_exact_aligned_arrays_and_replays(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    raw_fingerprint = _raw_snapshot(config)

    first = build_processed_snapshot(config, raw_fingerprint)
    second = build_processed_snapshot(config, raw_fingerprint)
    arrays = load_processed_arrays(first)

    assert not first.reused
    assert second.reused
    assert second.fingerprint == first.fingerprint
    assert arrays.monthly_returns.shape == (2, 3)
    assert arrays.daily_returns.shape[1] == 3
    assert arrays.macro_features.shape == (2, 4)
    assert arrays.monthly_period_ordinals.shape == (2,)
    assert arrays.daily_session_ns.shape == (len(arrays.daily_returns),)
    assert arrays.monthly_continuation.tolist() == [False, True]
    assert not arrays.daily_continuation[0]
    assert arrays.daily_continuation[1:].all()
    qc = first.manifest["qc_report"]
    assert qc["status"] == "passed"
    assert qc["aligned"]["first_month"] == "2024-02"
    assert qc["aligned"]["last_month"] == "2024-03"
    assert qc["minimum_gates"]["monthly_passed"]
    assert qc["minimum_gates"]["daily_passed"]
