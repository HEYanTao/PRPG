"""Assets-only immutable data preparation for delivery version 5.

This is a thin coordinator over the existing provider, content-addressed
stores, and synchronized complete-month transform.  It deliberately excludes
FRED data, holdouts, proxy splices, and legacy model orchestration.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from prpg.config import DataQualityConfig
from prpg.data.processed import ProcessedSnapshotReference, ProcessedSnapshotStore
from prpg.data.providers import (
    ProviderPayload,
    YFinanceProvider,
    parse_yfinance_csv_bytes,
)
from prpg.data.snapshot import SnapshotReference, SnapshotStore
from prpg.data.transform import transform_assets
from prpg.errors import DataValidationError, IntegrityError
from prpg.v5.config import ResolvedV5Inputs

MONTH_LIBRARY_SCHEMA = "prpg-v5-complete-calendar-month-library-v1"
FEATURE_ROLES = ("equity", "taxable_bond", "muni_bond")


class AssetProvider(Protocol):
    def fetch(self, ticker: str, config: object) -> ProviderPayload: ...


@dataclass(frozen=True, slots=True)
class V5MonthLibrary:
    """Verified in-memory view of the immutable complete-month library."""

    reference: ProcessedSnapshotReference
    feature_names: tuple[str, str, str]
    daily_returns: np.ndarray[Any, np.dtype[np.float64]]
    daily_session_ns: np.ndarray[Any, np.dtype[np.int64]]
    monthly_returns: np.ndarray[Any, np.dtype[np.float64]]
    monthly_period_ordinals: np.ndarray[Any, np.dtype[np.int64]]
    month_start_offsets: np.ndarray[Any, np.dtype[np.int64]]
    monthly_continuation: np.ndarray[Any, np.dtype[np.bool_]]

    @property
    def month_count(self) -> int:
        return len(self.monthly_returns)

    @property
    def daily_count(self) -> int:
        return len(self.daily_returns)


def acquire_v5_raw_snapshot(
    inputs: ResolvedV5Inputs,
    *,
    provider: AssetProvider | None = None,
    now: Callable[[], datetime] | None = None,
) -> SnapshotReference:
    """Acquire the configured three tickers and publish only after all succeed."""

    fetcher = provider or YFinanceProvider()
    ticker_order = _feature_tickers(inputs)
    payloads = tuple(
        fetcher.fetch(ticker, inputs.config.data.yfinance) for ticker in ticker_order
    )
    actual = tuple(payload.identifier for payload in payloads)
    if actual != ticker_order or any(
        payload.provider != "yfinance" for payload in payloads
    ):
        raise DataValidationError(
            "version-5 provider returned the wrong asset set or order",
            details={"expected": ticker_order, "actual": actual},
        )
    clock = now or (lambda: datetime.now(UTC))
    timestamp = clock().astimezone(UTC)
    group_id = f"v5-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    return SnapshotStore(inputs.artifact_root, now=clock).create(
        payloads,
        acquisition_contract=_raw_acquisition_contract(inputs),
        acquisition_group_id=group_id,
        audit_metadata={"resolved_config_fingerprint": inputs.config_fingerprint},
    )


def build_v5_month_library(
    inputs: ResolvedV5Inputs,
    raw_snapshot_fingerprint: str,
) -> ProcessedSnapshotReference:
    """Build one offline synchronized complete-month library from raw bytes."""

    raw_store = SnapshotStore(inputs.artifact_root)
    raw_reference = raw_store.verify(raw_snapshot_fingerprint)
    _verify_raw_asset_set(raw_reference, _feature_tickers(inputs))
    _verify_raw_acquisition_contract(raw_reference, inputs)
    frames = {
        ticker: parse_yfinance_csv_bytes(
            raw_store.read_payload(raw_snapshot_fingerprint, "yfinance", ticker),
            ticker,
        )
        for ticker in _feature_tickers(inputs)
    }
    flags = inputs.config.data.quality_flags
    quality = DataQualityConfig(
        equity_abs_simple_return_flag=flags.equity_abs_simple_return,
        bond_abs_simple_return_flag=flags.bond_abs_simple_return,
        adjustment_factor_move_flag=flags.adjustment_factor_move,
        # These legacy fields are not used by transform_assets. Version 5
        # records all complete support rather than imposing another minimum.
        minimum_common_monthly_vectors=1,
        minimum_common_daily_vectors=1,
    )
    transformed = transform_assets(
        frames,
        inputs.config.data.assets,
        quality,
        request_start=inputs.config.data.yfinance.start,
        cutoff=inputs.config.data.cutoff,
    )
    role_to_ticker = inputs.config.data.assets.model_dump()
    ticker_order = _feature_tickers(inputs)
    daily = _ordered_ticker_frame(
        transformed.daily_returns, role_to_ticker, ticker_order
    )
    monthly = _ordered_ticker_frame(
        transformed.monthly_returns, role_to_ticker, ticker_order
    )
    direct_monthly = _ordered_ticker_frame(
        transformed.direct_monthly_returns, role_to_ticker, ticker_order
    )
    adjusted_prices = _ordered_ticker_frame(
        transformed.adjusted_prices, role_to_ticker, ticker_order
    )
    if monthly.empty or daily.empty:
        raise DataValidationError("version-5 complete-month library is empty")
    crosscheck = (monthly - direct_monthly).abs()
    maximum_crosscheck_error = float(crosscheck.to_numpy().max(initial=0.0))
    tolerance = inputs.config.validation.csv_log_sum_absolute_tolerance
    if maximum_crosscheck_error > tolerance:
        raise DataValidationError(
            "version-5 monthly return crosscheck failed",
            details={
                "maximum_absolute_error": maximum_crosscheck_error,
                "tolerance": tolerance,
            },
        )

    month_ordinals = monthly.index.asi8.astype(np.int64)
    daily_months = daily.index.to_period("M")
    lengths = np.asarray(
        [(daily_months == month).sum() for month in monthly.index], dtype=np.int64
    )
    if bool((lengths <= 0).any()) or int(lengths.sum()) != len(daily):
        raise DataValidationError("version-5 month block lengths are inconsistent")
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    continuation = np.zeros(len(month_ordinals), dtype=np.bool_)
    if len(continuation) > 1:
        continuation[1:] = np.diff(month_ordinals) == 1

    reconstructed = np.vstack(
        [
            daily.iloc[int(offsets[index]) : int(offsets[index + 1])]
            .sum(axis=0)
            .to_numpy(dtype=np.float64)
            for index in range(len(monthly))
        ]
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed - monthly.to_numpy(dtype=np.float64)))
    )
    if reconstruction_error > tolerance:
        raise DataValidationError(
            "version-5 block offsets do not reconstruct monthly returns",
            details={
                "maximum_absolute_error": reconstruction_error,
                "tolerance": tolerance,
            },
        )

    eligibility = _eligibility_table(transformed.month_coverage)
    transform_contract = _processed_transform_contract(inputs, ticker_order)
    files = {
        "tables/adjusted-prices.csv": _frame_csv_bytes(
            adjusted_prices, index_label="session"
        ),
        "tables/daily-returns.csv": _frame_csv_bytes(daily, index_label="session"),
        "tables/monthly-returns.csv": _frame_csv_bytes(monthly, index_label="month"),
        "evidence/month-eligibility.csv": _frame_csv_bytes(
            eligibility, index_label="month"
        ),
        "evidence/outlier-flags.csv": _frame_csv_bytes(
            transformed.outlier_flags, include_index=False
        ),
        "metadata/data-dictionary.json": _canonical_json_bytes(
            {
                "schema": MONTH_LIBRARY_SCHEMA,
                "return_semantics": "nominal adjusted-close total log return",
                "feature_order": list(ticker_order),
                "semantic_roles": {
                    role: role_to_ticker[role] for role in FEATURE_ROLES
                },
                "daily_membership": (
                    "month_start_offsets[i]:month_start_offsets[i+1] contains "
                    "every synchronized XNYS return vector for source month i"
                ),
                "diagnostic_flags_are_release_gates": False,
            }
        ),
        "arrays/daily_returns.npy": _npy_bytes(daily.to_numpy(dtype=np.float64)),
        "arrays/daily_session_ns.npy": _npy_bytes(daily.index.asi8.astype(np.int64)),
        "arrays/monthly_returns.npy": _npy_bytes(monthly.to_numpy(dtype=np.float64)),
        "arrays/monthly_period_ordinals.npy": _npy_bytes(month_ordinals),
        "arrays/month_start_offsets.npy": _npy_bytes(offsets),
        "arrays/monthly_continuation.npy": _npy_bytes(continuation),
    }
    qc_report = {
        "schema": MONTH_LIBRARY_SCHEMA,
        "raw_snapshot_fingerprint": raw_snapshot_fingerprint,
        "expected_calendar_months": len(eligibility),
        "eligible_complete_months": len(monthly),
        "excluded_months": int((~eligibility["eligible"]).sum()),
        "eligible_daily_vectors": len(daily),
        "first_eligible_month": str(monthly.index[0]),
        "last_eligible_month": str(monthly.index[-1]),
        "contiguous_segments": int((~continuation).sum()),
        "maximum_direct_monthly_crosscheck_error": maximum_crosscheck_error,
        "maximum_offset_reconstruction_error": reconstruction_error,
        "diagnostic_outlier_flags": len(transformed.outlier_flags),
        "diagnostic_unresolved_flags": int(
            (
                transformed.outlier_flags.get("resolution")
                == "unresolved_release_blocker"
            ).sum()
            if not transformed.outlier_flags.empty
            else 0
        ),
    }
    reference = ProcessedSnapshotStore(inputs.artifact_root).create(
        raw_snapshot_fingerprint=raw_snapshot_fingerprint,
        transform_contract=transform_contract,
        files=files,
        qc_report=qc_report,
        audit_metadata={
            "resolved_config_fingerprint": inputs.config_fingerprint,
            "diagnostic_note": (
                "outlier flags are inspection prompts, not silent release gates"
            ),
        },
    )
    # Close P9-G1 with a full offline reload, not merely a successful write.
    load_v5_month_library(inputs, reference.fingerprint)
    return reference


def load_v5_month_library(
    inputs: ResolvedV5Inputs,
    fingerprint: str,
) -> V5MonthLibrary:
    """Verify and load the exact complete-month arrays entirely offline."""

    reference = ProcessedSnapshotStore(inputs.artifact_root).verify(fingerprint)
    identity = reference.manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise IntegrityError("version-5 processed identity is missing")
    contract = identity.get("transform_contract")
    expected_contract = _processed_transform_contract(inputs, _feature_tickers(inputs))
    if contract != expected_contract:
        raise IntegrityError("version-5 processed transform contract changed")
    feature_names = tuple(contract["feature_order"])
    if feature_names != _feature_tickers(inputs):
        raise IntegrityError("version-5 processed feature order changed")

    arrays = {
        name: _load_array(reference.path / "arrays" / f"{name}.npy")
        for name in (
            "daily_returns",
            "daily_session_ns",
            "monthly_returns",
            "monthly_period_ordinals",
            "month_start_offsets",
            "monthly_continuation",
        )
    }
    daily = _require_matrix(arrays["daily_returns"], "daily returns", 3)
    monthly = _require_matrix(arrays["monthly_returns"], "monthly returns", 3)
    daily_ns = _require_vector(arrays["daily_session_ns"], "daily session labels", "i")
    month_ordinals = _require_vector(
        arrays["monthly_period_ordinals"], "monthly period labels", "i"
    )
    offsets = _require_vector(arrays["month_start_offsets"], "month start offsets", "i")
    continuation = _require_vector(
        arrays["monthly_continuation"], "monthly continuation", "b"
    )
    if len(daily_ns) != len(daily) or len(month_ordinals) != len(monthly):
        raise IntegrityError("version-5 data label counts do not match returns")
    if (
        len(offsets) != len(monthly) + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(daily)
        or bool((np.diff(offsets) <= 0).any())
    ):
        raise IntegrityError("version-5 month offsets are invalid")
    if (
        len(continuation) != len(monthly)
        or bool(continuation[0])
        or not np.array_equal(continuation[1:], np.diff(month_ordinals) == 1)
    ):
        raise IntegrityError("version-5 monthly continuation mask is invalid")
    if bool((np.diff(daily_ns) <= 0).any()) or bool(
        (np.diff(month_ordinals) <= 0).any()
    ):
        raise IntegrityError("version-5 source labels are not strictly increasing")

    source_months = pd.DatetimeIndex(pd.to_datetime(daily_ns)).to_period("M").asi8
    reconstructed = np.empty_like(monthly)
    for index in range(len(monthly)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if not bool((source_months[start:stop] == month_ordinals[index]).all()):
            raise IntegrityError("version-5 daily block belongs to the wrong month")
        reconstructed[index] = daily[start:stop].sum(axis=0)
    maximum_error = float(np.max(np.abs(reconstructed - monthly)))
    tolerance = inputs.config.validation.csv_log_sum_absolute_tolerance
    if maximum_error > tolerance:
        raise IntegrityError(
            "version-5 stored daily/monthly identity failed",
            details={"maximum_absolute_error": maximum_error, "tolerance": tolerance},
        )
    return V5MonthLibrary(
        reference=reference,
        feature_names=(feature_names[0], feature_names[1], feature_names[2]),
        daily_returns=daily,
        daily_session_ns=daily_ns,
        monthly_returns=monthly,
        monthly_period_ordinals=month_ordinals,
        month_start_offsets=offsets,
        monthly_continuation=continuation,
    )


def _feature_tickers(inputs: ResolvedV5Inputs) -> tuple[str, str, str]:
    assets = inputs.config.data.assets
    mapping = assets.model_dump()
    values = tuple(str(mapping[role]) for role in inputs.config.model.feature_order)
    return values[0], values[1], values[2]


def _raw_acquisition_contract(inputs: ResolvedV5Inputs) -> dict[str, object]:
    data = inputs.config.data
    return {
        "schema": "prpg-v5-three-etf-raw-v1",
        "assets": data.assets.model_dump(mode="json"),
        "feature_order": list(_feature_tickers(inputs)),
        "cutoff": data.cutoff.isoformat(),
        "calendar": data.calendar,
        "common_history_only": data.common_history_only,
        "complete_months_only": data.complete_months_only,
        "forbid_imputation": data.forbid_imputation,
        "provider": "yfinance",
        "request": data.yfinance.model_dump(mode="json"),
    }


def _processed_transform_contract(
    inputs: ResolvedV5Inputs,
    feature_names: Sequence[str],
) -> dict[str, object]:
    return {
        "schema": MONTH_LIBRARY_SCHEMA,
        "raw_acquisition_contract_sha256": _sha256_json(
            _raw_acquisition_contract(inputs)
        ),
        "cutoff": inputs.config.data.cutoff.isoformat(),
        "calendar": inputs.config.data.calendar,
        "feature_order": list(feature_names),
        "semantic_roles": inputs.config.data.assets.model_dump(mode="json"),
        "quality_flags": inputs.config.data.quality_flags.model_dump(mode="json"),
        "complete_months_only": True,
        "common_history_only": True,
        "forbid_imputation": True,
        "return_representation": "log_total_decimal",
        "price_field": "Adj Close",
        "first_return_requires_preceding_expected_session": True,
        "exchange_calendars_version": version("exchange-calendars"),
        "numpy_version": version("numpy"),
        "pandas_version": version("pandas"),
        "yfinance_version": version("yfinance"),
    }


def _verify_raw_asset_set(
    reference: SnapshotReference,
    expected_tickers: Sequence[str],
) -> None:
    files = reference.manifest.get("files")
    if not isinstance(files, list):
        raise IntegrityError("version-5 raw snapshot file list is invalid")
    actual = sorted(
        (item.get("provider"), item.get("identifier"))
        for item in files
        if isinstance(item, Mapping)
    )
    expected = sorted(("yfinance", ticker) for ticker in expected_tickers)
    if actual != expected:
        raise IntegrityError(
            "version-5 raw snapshot must contain exactly the configured tickers",
            details={"expected": expected, "actual": actual},
        )


def _verify_raw_acquisition_contract(
    reference: SnapshotReference,
    inputs: ResolvedV5Inputs,
) -> None:
    identity = reference.manifest.get("identity")
    if not isinstance(identity, Mapping) or (
        identity.get("acquisition_contract") != _raw_acquisition_contract(inputs)
    ):
        raise IntegrityError(
            "version-5 raw acquisition contract does not match the active config"
        )


def _ordered_ticker_frame(
    frame: pd.DataFrame,
    role_to_ticker: Mapping[str, str],
    ticker_order: Sequence[str],
) -> pd.DataFrame:
    ordered_roles = [
        next(role for role, ticker in role_to_ticker.items() if ticker == target)
        for target in ticker_order
    ]
    result = frame.loc[:, ordered_roles].rename(columns=role_to_ticker).copy()
    if tuple(result.columns) != tuple(ticker_order):
        raise DataValidationError("version-5 asset feature order is inconsistent")
    return result


def _eligibility_table(coverage: pd.DataFrame) -> pd.DataFrame:
    result = coverage.copy()
    result["eligible"] = result["complete"].astype(bool)
    missing = result["expected_sessions"] - result["valid_common_return_sessions"]
    result["excluded_session_count"] = missing.astype(int)
    result["reason"] = [
        "eligible"
        if bool(eligible)
        else f"missing_{int(count)}_synchronized_return_sessions"
        for eligible, count in zip(result["eligible"], missing, strict=True)
    ]
    return result


def _frame_csv_bytes(
    frame: pd.DataFrame,
    *,
    index_label: str | None = None,
    include_index: bool = True,
) -> bytes:
    exported = frame.copy()
    if include_index and isinstance(exported.index, pd.PeriodIndex):
        exported.index = exported.index.astype(str)
    buffer = io.StringIO(newline="")
    exported.to_csv(
        buffer,
        index=include_index,
        index_label=index_label,
        lineterminator="\n",
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def _npy_bytes(array: np.ndarray[Any, Any]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _load_array(path: Path) -> np.ndarray[Any, Any]:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise IntegrityError(
            "version-5 processed array is unreadable",
            details={"path": str(path), "error_type": type(error).__name__},
        ) from error
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _require_matrix(
    value: np.ndarray[Any, Any], label: str, columns: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != columns or result.shape[0] == 0:
        raise IntegrityError(f"version-5 {label} shape is invalid")
    if not np.isfinite(result).all():
        raise IntegrityError(f"version-5 {label} contains non-finite values")
    result.setflags(write=False)
    return result


def _require_vector(
    value: np.ndarray[Any, Any], label: str, kind: str
) -> np.ndarray[Any, Any]:
    result = np.asarray(value)
    if result.ndim != 1 or result.size == 0 or result.dtype.kind != kind:
        raise IntegrityError(f"version-5 {label} is invalid")
    result.setflags(write=False)
    return result


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
