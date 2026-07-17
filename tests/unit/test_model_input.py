from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prpg.data.processed import (
    PROCESSED_SCHEMA_VERSION,
    TRANSFORM_VERSION,
    ProcessedSnapshotStore,
)
from prpg.data.providers import AcquisitionAttempt, ProviderPayload
from prpg.data.snapshot import MANIFEST_NAME as RAW_MANIFEST_NAME
from prpg.data.snapshot import SnapshotStore
from prpg.data.transform import expected_xnys_sessions
from prpg.errors import IntegrityError, ModelError
from prpg.model.input import (
    ASSET_COLUMNS,
    CALIBRATION_INPUT_SCHEMA_VERSION,
    DATA_DICTIONARY_PATH,
    MACRO_COLUMNS,
    forward_links_from_row_start_mask,
    load_calibration_input,
)

CONFIG_FINGERPRINT = "c" * 64
CALIBRATION_FINGERPRINT = "d" * 64


def _payload(provider: str, identifier: str) -> ProviderPayload:
    retrieved = "2026-07-14T12:00:00+00:00"
    frame = pd.DataFrame(
        {"value": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02"], name="date"),
    )
    return ProviderPayload(
        provider=provider,
        identifier=identifier,
        frame=frame,
        raw_bytes=f"date,{identifier}\n2024-01-02,1\n".encode(),
        retrieved_utc=retrieved,
        request={"identifier": identifier},
        library_version="synthetic-1",
        attempts=(
            AcquisitionAttempt(
                number=1,
                started_utc=retrieved,
                finished_utc=retrieved,
                outcome="success",
                http_status=200,
            ),
        ),
    )


def _raw_snapshot(root: Path) -> tuple[SnapshotStore, str]:
    store = SnapshotStore(root)
    payloads = [
        *(_payload("yfinance", ticker) for ticker in ("ACWI", "MUB", "AGG")),
        *(
            _payload("fred", series)
            for series in ("INDPRO", "CPIAUCSL", "T10Y3M", "BAA10Y")
        ),
    ]
    reference = store.create(
        payloads,
        acquisition_contract={"schema_version": 1, "synthetic": True},
        acquisition_group_id="synthetic-input-test",
        audit_metadata={"resolved_config_fingerprint": CONFIG_FINGERPRINT},
    )
    return store, reference.fingerprint


def _base_arrays() -> dict[str, np.ndarray]:
    month_index = pd.PeriodIndex(
        ["2024-01", "2024-02", "2024-03", "2024-05", "2024-06", "2024-07"],
        freq="M",
    )
    monthly_returns = np.array(
        [
            [0.010, 0.004, 0.003],
            [-0.020, 0.002, 0.001],
            [0.015, -0.001, 0.002],
            [0.005, 0.003, -0.002],
            [0.008, 0.001, 0.004],
            [-0.003, 0.002, 0.005],
        ],
        dtype=np.float64,
    )
    sessions: list[pd.Timestamp] = []
    daily_rows: list[np.ndarray] = []
    for ordinal, month in enumerate(month_index):
        start = date(month.year, month.month, 1)
        end = (month + 1).start_time.date()
        month_sessions = expected_xnys_sessions(start, end)[:2]
        sessions.extend(month_sessions)
        daily_rows.extend(
            [monthly_returns[ordinal] * 0.4, monthly_returns[ordinal] * 0.6]
        )
    session_index = pd.DatetimeIndex(sessions)
    calendar = expected_xnys_sessions(session_index[0].date(), session_index[-1].date())
    positions = calendar.get_indexer(session_index)
    daily_continuation = np.zeros(len(session_index), dtype=np.bool_)
    daily_continuation[1:] = np.diff(positions) == 1
    monthly_continuation = np.zeros(len(month_index), dtype=np.bool_)
    monthly_continuation[1:] = np.diff(month_index.asi8) == 1
    return {
        "monthly_returns": monthly_returns,
        "daily_returns": np.asarray(daily_rows, dtype=np.float64),
        "macro_features": np.arange(24, dtype=np.float64).reshape(6, 4) / 10.0,
        "monthly_period_ordinals": month_index.asi8.astype(np.int64),
        "daily_session_ns": session_index.asi8.astype(np.int64),
        "monthly_continuation": monthly_continuation,
        "daily_continuation": daily_continuation,
    }


def _transform_contract() -> dict[str, object]:
    return {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "calendar": "XNYS",
        "return_representation": "log_total_decimal",
        "common_history_only": True,
        "forbid_price_forward_fill": True,
        "assets": {
            "equity": "ACWI",
            "muni_bond": "MUB",
            "taxable_bond": "AGG",
        },
        "macro": {
            "industrial_production": "INDPRO",
            "inflation": "CPIAUCSL",
            "yield_curve_spread": "T10Y3M",
            "credit_spread": "BAA10Y",
        },
    }


def _dictionary(contract: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "return_semantics": "nominal total log return in decimal units",
        "asset_columns": {
            "equity": "ACWI adjusted-close total-return proxy",
            "muni_bond": "MUB adjusted-close total-return proxy",
            "taxable_bond": "AGG broad taxable-bond adjusted-close proxy",
        },
        "macro_columns": {
            "industrial_production_growth": "100 * log(INDPRO_t / INDPRO_t-1)",
            "inflation_yoy": "100 * log(CPIAUCSL_t / CPIAUCSL_t-12)",
            "yield_curve_spread": ("within-month finite-observation mean of T10Y3M"),
            "credit_spread": "within-month finite-observation mean of BAA10Y",
        },
        "index_semantics": {
            "monthly": "calendar month YYYY-MM; return ends in that month",
            "daily": "normalized America/New_York XNYS session date",
        },
        "continuation_masks": (
            "False at the first row and whenever the source index is not the "
            "immediately following eligible calendar/session observation; blocks "
            "must restart when False"
        ),
        "transform_contract": contract,
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    # The one adversarial object-array fixture must be serializable so the
    # production loader, rather than the test helper, proves pickle stays off.
    np.save(buffer, array, allow_pickle=array.dtype.hasobject)
    return buffer.getvalue()


def _processed_snapshot(
    root: Path,
    raw_fingerprint: str,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    contract: dict[str, object] | None = None,
    dictionary: dict[str, object] | None = None,
    dictionary_bytes: bytes | None = None,
    qc_updates: dict[str, object] | None = None,
    audit_metadata: dict[str, object] | None = None,
    omit_file: str | None = None,
) -> tuple[ProcessedSnapshotStore, str]:
    values = arrays or _base_arrays()
    transform = contract or _transform_contract()
    semantic_dictionary = dictionary or _dictionary(transform)
    files = {f"arrays/{name}.npy": _npy_bytes(array) for name, array in values.items()}
    files[DATA_DICTIONARY_PATH] = dictionary_bytes or _json_bytes(semantic_dictionary)
    if omit_file is not None:
        files.pop(omit_file)
    qc = {
        "status": "passed",
        "raw_snapshot_fingerprint": raw_fingerprint,
        "raw_source_count": 7,
        "outlier_resolution_complete": True,
        "retrospective_not_realtime": True,
        "secret_scan_passed": True,
    }
    qc.update(qc_updates or {})
    store = ProcessedSnapshotStore(root)
    reference = store.create(
        raw_snapshot_fingerprint=raw_fingerprint,
        transform_contract=transform,
        files=files,
        qc_report=qc,
        audit_metadata=(
            {"resolved_config_fingerprint": CONFIG_FINGERPRINT}
            if audit_metadata is None
            else audit_metadata
        ),
    )
    return store, reference.fingerprint


def _fixture(
    tmp_path: Path,
    **processed_options: object,
) -> tuple[ProcessedSnapshotStore, SnapshotStore, str]:
    root = tmp_path / "artifacts"
    raw_store, raw_fingerprint = _raw_snapshot(root)
    processed_store, fingerprint = _processed_snapshot(
        root,
        raw_fingerprint,
        **processed_options,  # type: ignore[arg-type]
    )
    return processed_store, raw_store, fingerprint


def _load(
    processed_store: ProcessedSnapshotStore,
    raw_store: SnapshotStore,
    fingerprint: str,
    *,
    config_fingerprint: str = CALIBRATION_FINGERPRINT,
    holdout_months: int = 2,
):
    return load_calibration_input(
        processed_store,
        fingerprint,
        calibration_config_fingerprint=config_fingerprint,
        holdout_months=holdout_months,
        raw_store=raw_store,
    )


def test_load_binds_verified_provenance_semantics_and_split(tmp_path: Path) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path)

    result = _load(processed_store, raw_store, fingerprint)

    assert result.provenance.processed_data_fingerprint == fingerprint
    assert result.provenance.processed_config_fingerprint == CONFIG_FINGERPRINT
    assert result.provenance.calibration_config_fingerprint == CALIBRATION_FINGERPRINT
    assert result.provenance.asset_columns == ASSET_COLUMNS
    assert result.provenance.macro_columns == MACRO_COLUMNS
    assert len(result.provenance.source_identities) == 7
    assert (
        len([item for item in result.provenance.source_files if item.layer == "raw"])
        == 7
    )
    assert (
        len(
            [
                item
                for item in result.provenance.source_files
                if item.layer == "processed"
            ]
        )
        == 8
    )
    assert result.geometry.holdout_months == 2
    assert result.geometry.design_monthly_rows == 4
    assert result.geometry.holdout_monthly_rows == 2
    assert result.geometry.design_daily_rows == 8
    assert result.geometry.holdout_daily_rows == 4
    assert result.full.role == "full"
    assert result.design.role == "design"
    assert result.holdout.role == "holdout"
    assert result.full.monthly_returns.shape == (6, 3)
    assert result.design.macro_features.shape == (4, 4)
    assert result.holdout.daily_returns.shape == (4, 3)
    assert not result.holdout.monthly_continuation[0]
    assert not result.holdout.daily_continuation[0]

    encoded = _json_bytes(result.identity_dict())
    assert result.identity_dict()["schema_version"] == CALIBRATION_INPUT_SCHEMA_VERSION
    assert hashlib.sha256(encoded).hexdigest() == result.fingerprint


def test_loaded_arrays_are_detached_and_irreversibly_read_only(tmp_path: Path) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path)
    result = _load(processed_store, raw_store, fingerprint)
    arrays = (
        result.full.monthly_returns,
        result.full.daily_returns,
        result.full.macro_features,
        result.full.monthly_period_ordinals,
        result.full.daily_session_ns,
        result.full.monthly_continuation,
        result.full.daily_continuation,
        result.design.monthly_returns,
        result.holdout.monthly_returns,
    )

    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError):
        result.full.monthly_returns[0, 0] = 99.0
    assert not np.shares_memory(
        result.full.monthly_returns, result.design.monthly_returns
    )
    assert not np.shares_memory(
        result.full.monthly_returns, result.holdout.monthly_returns
    )


def test_fingerprint_is_deterministic_and_binds_config_and_geometry(
    tmp_path: Path,
) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path)

    first = _load(processed_store, raw_store, fingerprint)
    replay = _load(processed_store, raw_store, fingerprint)
    changed_config = _load(
        processed_store,
        raw_store,
        fingerprint,
        config_fingerprint="e" * 64,
    )
    changed_split = _load(
        processed_store,
        raw_store,
        fingerprint,
        holdout_months=1,
    )

    assert first.fingerprint == replay.fingerprint
    assert first.fingerprint != changed_config.fingerprint
    assert first.fingerprint != changed_split.fingerprint


def test_daily_split_is_derived_from_month_ordinals(tmp_path: Path) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path)

    result = _load(processed_store, raw_store, fingerprint, holdout_months=3)

    design_months = pd.DatetimeIndex(result.design.daily_session_ns).to_period("M")
    holdout_months = pd.DatetimeIndex(result.holdout.daily_session_ns).to_period("M")
    assert set(design_months.astype(str)) == {"2024-01", "2024-02", "2024-03"}
    assert set(holdout_months.astype(str)) == {"2024-05", "2024-06", "2024-07"}
    assert result.geometry.design_daily_rows == 6


def test_forward_link_adapter_has_exact_offset_and_immutable_result() -> None:
    mask = np.array([False, True, False, True], dtype=np.bool_)

    result = forward_links_from_row_start_mask(mask)
    singleton = forward_links_from_row_start_mask(np.array([False], dtype=np.bool_))

    assert result.tolist() == [True, False, True]
    assert singleton.shape == (0,)
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.array([], dtype=np.bool_), "non-empty bool"),
        (np.array([[False]], dtype=np.bool_), "non-empty bool"),
        (np.array([0, 1], dtype=np.int8), "non-empty bool"),
        (np.array([True, False], dtype=np.bool_), "false at row zero"),
    ],
)
def test_forward_link_adapter_rejects_ambiguous_masks(
    mask: np.ndarray, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        forward_links_from_row_start_mask(mask)


ArrayMutation = Callable[[dict[str, np.ndarray]], None]


def _monthly_float32(values: dict[str, np.ndarray]) -> None:
    values["monthly_returns"] = values["monthly_returns"].astype(np.float32)


def _monthly_wrong_shape(values: dict[str, np.ndarray]) -> None:
    values["monthly_returns"] = values["monthly_returns"][:, :2]


def _macro_nonfinite(values: dict[str, np.ndarray]) -> None:
    values["macro_features"][0, 0] = np.nan


def _macro_wrong_rows(values: dict[str, np.ndarray]) -> None:
    values["macro_features"] = values["macro_features"][:-1]


def _ordinal_int32(values: dict[str, np.ndarray]) -> None:
    values["monthly_period_ordinals"] = values["monthly_period_ordinals"].astype(
        np.int32
    )


def _duplicate_ordinal(values: dict[str, np.ndarray]) -> None:
    values["monthly_period_ordinals"][1] = values["monthly_period_ordinals"][0]


def _reversed_daily_time(values: dict[str, np.ndarray]) -> None:
    values["daily_session_ns"][[0, 1]] = values["daily_session_ns"][[1, 0]]


def _monthly_mask_wrong(values: dict[str, np.ndarray]) -> None:
    values["monthly_continuation"][3] = True


def _daily_mask_wrong(values: dict[str, np.ndarray]) -> None:
    values["daily_continuation"][1] = False


def _daily_mask_wrong_length(values: dict[str, np.ndarray]) -> None:
    values["daily_continuation"] = values["daily_continuation"][:-1]


def _daily_non_session(values: dict[str, np.ndarray]) -> None:
    timestamps = pd.DatetimeIndex(values["daily_session_ns"])
    timestamps = timestamps.where(
        np.arange(len(timestamps)) != 2,
        pd.Timestamp("2024-01-06"),  # between Jan 3 and Feb 1, but a Saturday
    )
    values["daily_session_ns"] = timestamps.asi8.astype(np.int64)


def _daily_not_normalized(values: dict[str, np.ndarray]) -> None:
    values["daily_session_ns"][0] += 3_600_000_000_000


def _daily_month_missing(values: dict[str, np.ndarray]) -> None:
    keep = np.arange(len(values["daily_returns"])) < 10
    for name in ("daily_returns", "daily_session_ns", "daily_continuation"):
        values[name] = values[name][keep]


def _monthly_sum_mismatch(values: dict[str, np.ndarray]) -> None:
    values["monthly_returns"][0, 0] += 0.001


def _empty_monthly_ordinals(values: dict[str, np.ndarray]) -> None:
    values["monthly_period_ordinals"] = np.array([], dtype=np.int64)


def _monthly_mask_int(values: dict[str, np.ndarray]) -> None:
    values["monthly_continuation"] = values["monthly_continuation"].astype(np.int8)


def _monthly_mask_matrix(values: dict[str, np.ndarray]) -> None:
    values["monthly_continuation"] = values["monthly_continuation"].reshape(2, 3)


def _monthly_mask_first_true(values: dict[str, np.ndarray]) -> None:
    values["monthly_continuation"][0] = True


def _monthly_mask_short(values: dict[str, np.ndarray]) -> None:
    values["monthly_continuation"] = values["monthly_continuation"][:-1]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_monthly_float32, "exact float64"),
        (_monthly_wrong_shape, "n-by-3"),
        (_macro_nonfinite, "finite"),
        (_macro_wrong_rows, "misaligned"),
        (_ordinal_int32, "exact int64"),
        (_duplicate_ordinal, "strictly increasing"),
        (_reversed_daily_time, "strictly increasing"),
        (_monthly_mask_wrong, "source chronology"),
        (_daily_mask_wrong, "source chronology"),
        (_daily_mask_wrong_length, "misaligned"),
        (_daily_non_session, "strictly increasing|non-XNYS"),
        (_daily_not_normalized, "normalized"),
        (_daily_month_missing, "different month sets"),
        (_monthly_sum_mismatch, "do not match monthly"),
        (_empty_monthly_ordinals, "non-empty vector"),
        (_monthly_mask_int, "exact bool"),
        (_monthly_mask_matrix, "non-empty vector"),
        (_monthly_mask_first_true, "false at row zero"),
        (_monthly_mask_short, "length is misaligned"),
    ],
)
def test_array_contract_fails_closed(
    tmp_path: Path, mutation: ArrayMutation, message: str
) -> None:
    arrays = _base_arrays()
    mutation(arrays)
    processed_store, raw_store, fingerprint = _fixture(tmp_path, arrays=arrays)

    with pytest.raises(ModelError, match=message):
        _load(processed_store, raw_store, fingerprint)


def test_pickle_requiring_array_is_rejected_safely(tmp_path: Path) -> None:
    arrays = _base_arrays()
    arrays["macro_features"] = np.array([[object()]], dtype=object)
    processed_store, raw_store, fingerprint = _fixture(tmp_path, arrays=arrays)

    with pytest.raises(ModelError, match="cannot be loaded safely"):
        _load(processed_store, raw_store, fingerprint)


def test_dictionary_must_be_canonical_and_semantically_exact(tmp_path: Path) -> None:
    contract = _transform_contract()
    changed = _dictionary(contract)
    changed["return_semantics"] = "simple return"
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path / "semantic", contract=contract, dictionary=changed
    )
    with pytest.raises(ModelError, match="semantics are incompatible"):
        _load(processed_store, raw_store, fingerprint)

    ordinary_json = json.dumps(_dictionary(contract), indent=2).encode()
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path / "canonical", contract=contract, dictionary_bytes=ordinary_json
    )
    with pytest.raises(ModelError, match="canonical JSON"):
        _load(processed_store, raw_store, fingerprint)


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"{broken", "cannot be read"), (b"[]\n", "root must be an object")],
)
def test_dictionary_must_be_readable_object(
    tmp_path: Path, content: bytes, message: str
) -> None:
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path, dictionary_bytes=content
    )

    with pytest.raises(ModelError, match=message):
        _load(processed_store, raw_store, fingerprint)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calendar", "weekday"),
        ("return_representation", "simple"),
        ("common_history_only", False),
        ("schema_version", True),
    ],
)
def test_transform_contract_scientific_semantics_are_strict(
    tmp_path: Path, field: str, value: object
) -> None:
    contract = _transform_contract()
    contract[field] = value
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path, contract=contract, dictionary=_dictionary(contract)
    )

    with pytest.raises(ModelError, match="transform contract is incompatible"):
        _load(processed_store, raw_store, fingerprint)


def test_raw_sources_must_match_processed_roles(tmp_path: Path) -> None:
    contract = _transform_contract()
    assets = dict(contract["assets"])  # type: ignore[arg-type]
    assets["equity"] = "SPY"
    contract["assets"] = assets
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path, contract=contract, dictionary=_dictionary(contract)
    )

    with pytest.raises(ModelError, match="sources do not match"):
        _load(processed_store, raw_store, fingerprint)


@pytest.mark.parametrize(
    ("assets", "message"),
    [
        ({"equity": "ACWI", "muni_bond": "MUB"}, "role mapping is not exact"),
        (
            {"equity": "ACWI", "muni_bond": "MUB", "taxable_bond": 7},
            "identifier is invalid",
        ),
        (
            {"equity": "ACWI", "muni_bond": "ACWI", "taxable_bond": "AGG"},
            "identifiers are not distinct",
        ),
    ],
)
def test_semantic_role_mapping_must_be_exact(
    tmp_path: Path, assets: dict[str, object], message: str
) -> None:
    contract = _transform_contract()
    contract["assets"] = assets
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path, contract=contract, dictionary=_dictionary(contract)
    )

    with pytest.raises(ModelError, match=message):
        _load(processed_store, raw_store, fingerprint)


def test_raw_manifest_bytes_are_cross_checked_against_raw_identity(
    tmp_path: Path,
) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path)
    processed_reference = processed_store.verify(fingerprint)
    raw_fingerprint = processed_reference.manifest["identity"][
        "raw_snapshot_fingerprint"
    ]
    raw_reference = raw_store.verify(raw_fingerprint)
    manifest_path = raw_reference.path / RAW_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["files"][0]
    payload_path = raw_reference.path / record["relative_path"]
    changed = b"changed,raw,bytes\n"
    payload_path.write_bytes(changed)
    record["bytes"] = len(changed)
    record["sha256"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert raw_store.verify(raw_fingerprint)

    with pytest.raises(ModelError, match="disagrees with raw identity"):
        _load(processed_store, raw_store, fingerprint)


@pytest.mark.parametrize(
    ("qc_updates", "message"),
    [
        ({"status": "failed"}, "not calibration-eligible"),
        ({"raw_source_count": 6}, "not calibration-eligible"),
        ({"secret_scan_passed": False}, "not calibration-eligible"),
    ],
)
def test_processed_qc_must_be_passed_and_exact(
    tmp_path: Path, qc_updates: dict[str, object], message: str
) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path, qc_updates=qc_updates)

    with pytest.raises(ModelError, match=message):
        _load(processed_store, raw_store, fingerprint)


@pytest.mark.parametrize(
    "audit_metadata",
    [{}, {"resolved_config_fingerprint": "BAD"}],
)
def test_processed_config_provenance_is_required(
    tmp_path: Path, audit_metadata: dict[str, object]
) -> None:
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path, audit_metadata=audit_metadata
    )

    with pytest.raises(ModelError, match="processed config fingerprint"):
        _load(processed_store, raw_store, fingerprint)


@pytest.mark.parametrize("fingerprint", ["BAD", "A" * 64])
def test_calibration_config_fingerprint_is_strict(
    tmp_path: Path, fingerprint: str
) -> None:
    processed_store, raw_store, processed_fingerprint = _fixture(tmp_path)

    with pytest.raises(ModelError, match="calibration config fingerprint"):
        _load(
            processed_store,
            raw_store,
            processed_fingerprint,
            config_fingerprint=fingerprint,
        )


def test_processed_fingerprint_is_validated_before_store_access(tmp_path: Path) -> None:
    processed_store = ProcessedSnapshotStore(tmp_path)

    with pytest.raises(ModelError, match="processed data fingerprint"):
        load_calibration_input(
            processed_store,
            "BAD",
            calibration_config_fingerprint=CALIBRATION_FINGERPRINT,
            holdout_months=2,
        )


def test_default_raw_store_uses_processed_artifact_root(tmp_path: Path) -> None:
    processed_store, _, fingerprint = _fixture(tmp_path)

    result = load_calibration_input(
        processed_store,
        fingerprint,
        calibration_config_fingerprint=CALIBRATION_FINGERPRINT,
        holdout_months=2,
    )

    assert result.provenance.processed_data_fingerprint == fingerprint


@pytest.mark.parametrize("holdout", [True, 0, -1, 6, 7])
def test_holdout_geometry_must_leave_both_partitions(
    tmp_path: Path, holdout: int
) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path)

    with pytest.raises(ModelError, match="holdout_months"):
        _load(processed_store, raw_store, fingerprint, holdout_months=holdout)


def test_missing_required_processed_file_is_rejected_before_array_load(
    tmp_path: Path,
) -> None:
    processed_store, raw_store, fingerprint = _fixture(
        tmp_path, omit_file="arrays/daily_continuation.npy"
    )

    with pytest.raises(ModelError, match="lacks required calibration files"):
        _load(processed_store, raw_store, fingerprint)


def test_processed_and_raw_tamper_remain_integrity_failures(tmp_path: Path) -> None:
    processed_store, raw_store, fingerprint = _fixture(tmp_path / "processed")
    processed_reference = processed_store.verify(fingerprint)
    (processed_reference.path / "arrays/monthly_returns.npy").write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="integrity check failed"):
        _load(processed_store, raw_store, fingerprint)

    processed_store, raw_store, fingerprint = _fixture(tmp_path / "raw")
    processed_reference = processed_store.verify(fingerprint)
    raw_fingerprint = processed_reference.manifest["identity"][
        "raw_snapshot_fingerprint"
    ]
    raw_reference = raw_store.verify(raw_fingerprint)
    raw_file = raw_reference.path / raw_reference.manifest["files"][0]["relative_path"]
    raw_file.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="integrity check failed"):
        _load(processed_store, raw_store, fingerprint)
