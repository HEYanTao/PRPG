"""Offline construction and verification of the processed data snapshot."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from prpg.config import PRPGConfig, redact_mapping
from prpg.data.providers import parse_fred_csv_bytes, parse_yfinance_csv_bytes
from prpg.data.snapshot import SnapshotStore
from prpg.data.transform import (
    AlignedHistoricalData,
    AssetTransformResult,
    MacroTransformResult,
    align_historical_data,
    transform_assets,
    transform_macro,
)
from prpg.errors import DataValidationError, IntegrityError

PROCESSED_SCHEMA_VERSION = 1
TRANSFORM_VERSION = "asset-macro-transform-v1"
MANIFEST_NAME = "processed-manifest.json"


@dataclass(frozen=True, slots=True)
class ProcessedSnapshotReference:
    """A verified processed-data artifact."""

    fingerprint: str
    path: Path
    manifest: Mapping[str, Any]
    reused: bool


@dataclass(frozen=True, slots=True)
class ProcessedArrays:
    """Safe numerical inputs and exact source-continuation masks."""

    monthly_returns: np.ndarray
    daily_returns: np.ndarray
    macro_features: np.ndarray
    monthly_period_ordinals: np.ndarray
    daily_session_ns: np.ndarray
    monthly_continuation: np.ndarray
    daily_continuation: np.ndarray


class ProcessedSnapshotStore:
    """Publish immutable processed artifacts under a content address."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.store_root = self.artifact_root / "data" / "processed" / "sha256"
        self._now = now or (lambda: datetime.now(UTC))

    def create(
        self,
        *,
        raw_snapshot_fingerprint: str,
        transform_contract: Mapping[str, Any],
        files: Mapping[str, bytes],
        qc_report: Mapping[str, Any],
        audit_metadata: Mapping[str, Any],
    ) -> ProcessedSnapshotReference:
        if not files:
            raise DataValidationError("processed snapshot requires files")
        records: list[dict[str, Any]] = []
        safe_files: dict[str, bytes] = {}
        for relative_path, content in sorted(files.items()):
            _validate_relative_path(relative_path)
            if not content:
                raise DataValidationError(
                    "processed snapshot file is empty",
                    details={"relative_path": relative_path},
                )
            safe_files[relative_path] = content
            records.append(
                {
                    "relative_path": relative_path,
                    "bytes": len(content),
                    "sha256": _sha256_bytes(content),
                }
            )
        identity = {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "raw_snapshot_fingerprint": raw_snapshot_fingerprint,
            "transform_contract": _json_value(transform_contract),
            "files": records,
            "qc_report": _json_value(qc_report),
        }
        fingerprint = _sha256_bytes(_canonical_json_bytes(identity))
        self.store_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self.store_root / fingerprint
        if destination.exists():
            reference = self.verify(fingerprint, reused=True)
            if _canonical_json_bytes(reference.manifest["identity"]) != (
                _canonical_json_bytes(identity)
            ):
                raise IntegrityError(
                    "processed snapshot fingerprint collision",
                    details={"fingerprint": fingerprint},
                )
            return reference

        stage = Path(
            tempfile.mkdtemp(prefix=f".stage-{fingerprint[:12]}-", dir=self.store_root)
        )
        try:
            for relative_path, content in safe_files.items():
                target = stage.joinpath(*PurePosixPath(relative_path).parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                _write_new(target, content)
            manifest = {
                "schema_version": PROCESSED_SCHEMA_VERSION,
                "data_fingerprint": fingerprint,
                "created_utc": self._now().astimezone(UTC).isoformat(),
                "identity": identity,
                "qc_report": _json_value(qc_report),
                "audit_metadata": _json_value(audit_metadata),
            }
            _write_new(stage / MANIFEST_NAME, _canonical_json_bytes(manifest))
            _fsync_directory(stage)
            try:
                os.rename(stage, destination)
                _fsync_directory(self.store_root)
            except OSError as error:
                if destination.exists():
                    reference = self.verify(fingerprint, reused=True)
                    if _canonical_json_bytes(reference.manifest["identity"]) != (
                        _canonical_json_bytes(identity)
                    ):
                        raise IntegrityError(
                            "processed snapshot fingerprint collision",
                            details={"fingerprint": fingerprint},
                        ) from error
                    return reference
                raise
            return self.verify(fingerprint, reused=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def verify(
        self, fingerprint: str, *, reused: bool = True
    ) -> ProcessedSnapshotReference:
        """Verify the identity and every allowlisted processed file offline."""

        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise IntegrityError(
                "processed fingerprint is invalid",
                details={"fingerprint": fingerprint},
            )
        root = self.store_root / fingerprint
        if not root.is_dir() or root.is_symlink():
            raise IntegrityError(
                "processed snapshot directory is missing or unsafe",
                details={"fingerprint": fingerprint},
            )
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise IntegrityError(
                "processed manifest is missing or unsafe",
                details={"fingerprint": fingerprint},
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "processed manifest cannot be read",
                details={
                    "fingerprint": fingerprint,
                    "error_type": type(error).__name__,
                },
            ) from error
        if not isinstance(manifest, dict):
            raise IntegrityError("processed manifest root is invalid")
        if (
            manifest.get("schema_version") != PROCESSED_SCHEMA_VERSION
            or manifest.get("data_fingerprint") != fingerprint
        ):
            raise IntegrityError(
                "processed manifest identity fields are invalid",
                details={"fingerprint": fingerprint},
            )
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise IntegrityError("processed snapshot identity is missing")
        computed = _sha256_bytes(_canonical_json_bytes(identity))
        if computed != fingerprint:
            raise IntegrityError(
                "processed snapshot identity hash mismatch",
                details={"expected": fingerprint, "computed": computed},
            )
        if _canonical_json_bytes(manifest.get("qc_report")) != _canonical_json_bytes(
            identity.get("qc_report")
        ):
            raise IntegrityError("processed QC report does not match its identity")
        records = identity.get("files")
        if not isinstance(records, list) or not records:
            raise IntegrityError("processed snapshot file list is invalid")
        allowed = {MANIFEST_NAME}
        for record in records:
            if not isinstance(record, dict):
                raise IntegrityError("processed snapshot file record is invalid")
            relative_path = record.get("relative_path")
            if not isinstance(relative_path, str):
                raise IntegrityError("processed relative path is invalid")
            _validate_relative_path(relative_path, error_type=IntegrityError)
            path = root.joinpath(*PurePosixPath(relative_path).parts)
            if not path.is_file() or path.is_symlink():
                raise IntegrityError(
                    "processed snapshot file is missing or unsafe",
                    details={"relative_path": relative_path},
                )
            if not path.resolve().is_relative_to(root.resolve()):
                raise IntegrityError(
                    "processed snapshot path escapes root",
                    details={"relative_path": relative_path},
                )
            actual = path.read_bytes()
            if len(actual) != record.get("bytes") or _sha256_bytes(
                actual
            ) != record.get("sha256"):
                raise IntegrityError(
                    "processed snapshot file integrity check failed",
                    details={"relative_path": relative_path},
                )
            allowed.add(relative_path)
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_files != allowed:
            raise IntegrityError(
                "processed snapshot has unmanifested or missing files",
                details={
                    "unexpected": sorted(actual_files - allowed),
                    "missing": sorted(allowed - actual_files),
                },
            )
        return ProcessedSnapshotReference(fingerprint, root, manifest, reused)


def build_processed_snapshot(
    config: PRPGConfig,
    raw_snapshot_fingerprint: str,
    *,
    store: ProcessedSnapshotStore | None = None,
) -> ProcessedSnapshotReference:
    """Rebuild the complete processed data artifact using only local bytes."""

    raw_store = SnapshotStore(config.data.artifact_root)
    raw_reference = raw_store.verify(raw_snapshot_fingerprint)
    if redact_mapping(raw_reference.manifest) != raw_reference.manifest:
        raise DataValidationError(
            "raw snapshot manifest contains an unredacted secret-like field"
        )
    asset_frames = {
        ticker: parse_yfinance_csv_bytes(
            raw_store.read_payload(raw_snapshot_fingerprint, "yfinance", ticker),
            ticker,
        )
        for ticker in (
            config.data.assets.equity,
            config.data.assets.muni_bond,
            config.data.assets.taxable_bond,
        )
    }
    macro_frames = {
        series_id: parse_fred_csv_bytes(
            raw_store.read_payload(raw_snapshot_fingerprint, "fred", series_id),
            series_id,
        )
        for series_id in (
            config.data.macro.industrial_production,
            config.data.macro.inflation,
            config.data.macro.yield_curve_spread,
            config.data.macro.credit_spread,
        )
    }
    assets = transform_assets(
        asset_frames,
        config.data.assets,
        config.data.quality,
        request_start=config.data.yfinance.start,
        cutoff=config.data.cutoff,
    )
    macro = transform_macro(
        macro_frames,
        config.data.macro,
        cutoff=config.data.cutoff,
        daily_coverage_fraction=config.data.fred.daily_month_coverage_fraction,
    )
    aligned = align_historical_data(assets, macro, config.data.quality)
    transform_contract = _transform_contract(config)
    files = _processed_files(
        assets,
        macro,
        aligned,
        transform_contract,
        holdout_months=config.model.design_holdout_months,
    )
    qc_report = _qc_report(config, raw_reference.manifest, assets, macro, aligned)
    processed_store = store or ProcessedSnapshotStore(config.data.artifact_root)
    return processed_store.create(
        raw_snapshot_fingerprint=raw_snapshot_fingerprint,
        transform_contract=transform_contract,
        files=files,
        qc_report=qc_report,
        audit_metadata={"resolved_config_fingerprint": config.fingerprint()},
    )


def load_processed_arrays(
    reference: ProcessedSnapshotReference,
) -> ProcessedArrays:
    """Load the seven safe ``allow_pickle=False`` calibration arrays."""

    names = (
        "monthly_returns",
        "daily_returns",
        "macro_features",
        "monthly_period_ordinals",
        "daily_session_ns",
        "monthly_continuation",
        "daily_continuation",
    )
    arrays = tuple(
        np.load(reference.path / "arrays" / f"{name}.npy", allow_pickle=False)
        for name in names
    )
    return ProcessedArrays(*arrays)


def _transform_contract(config: PRPGConfig) -> dict[str, Any]:
    return {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "calendar": "XNYS",
        "calendar_library_version": version("exchange-calendars"),
        "numpy_version": version("numpy"),
        "pandas_version": version("pandas"),
        "cutoff": config.data.cutoff.isoformat(),
        "macro_vintage_policy": config.data.macro_vintage_policy,
        "common_history_only": config.data.common_history_only,
        "forbid_price_forward_fill": config.data.forbid_price_forward_fill,
        "assets": config.data.assets.model_dump(mode="json"),
        "macro": config.data.macro.model_dump(mode="json"),
        "quality": config.data.quality.model_dump(mode="json"),
        "daily_macro_coverage_fraction": (
            config.data.fred.daily_month_coverage_fraction
        ),
        "return_representation": config.export.return_representation,
    }


def _processed_files(
    assets: AssetTransformResult,
    macro: MacroTransformResult,
    aligned: AlignedHistoricalData,
    transform_contract: Mapping[str, Any],
    *,
    holdout_months: int,
) -> dict[str, bytes]:
    eligibility = _model_month_eligibility(assets, macro)
    monthly_continuation = _monthly_continuation(aligned.monthly_returns.index)
    daily_continuation = _daily_continuation(
        aligned.daily_returns.index, assets.gap_mask.index
    )
    dictionary = {
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
            "yield_curve_spread": "within-month finite-observation mean of T10Y3M",
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
        "transform_contract": _json_value(transform_contract),
    }
    return {
        "tables/adjusted-prices.csv": _frame_csv_bytes(
            assets.adjusted_prices, index_label="session"
        ),
        "tables/daily-returns.csv": _frame_csv_bytes(
            aligned.daily_returns, index_label="session"
        ),
        "tables/monthly-returns.csv": _frame_csv_bytes(
            aligned.monthly_returns, index_label="month"
        ),
        "tables/macro-features.csv": _frame_csv_bytes(
            aligned.macro_features, index_label="month"
        ),
        "evidence/source-gap-mask.csv": _frame_csv_bytes(
            assets.gap_mask, index_label="session"
        ),
        "evidence/asset-month-coverage.csv": _frame_csv_bytes(
            assets.month_coverage, index_label="month"
        ),
        "evidence/macro-daily-coverage.csv": _frame_csv_bytes(
            macro.daily_coverage, index_label="month"
        ),
        "evidence/outlier-flags.csv": _frame_csv_bytes(
            assets.outlier_flags, include_index=False
        ),
        "evidence/model-month-eligibility.csv": _frame_csv_bytes(
            eligibility, index_label="month"
        ),
        "metadata/data-dictionary.json": _canonical_json_bytes(dictionary),
        "metadata/descriptive-statistics.json": _canonical_json_bytes(
            _descriptive_statistics(aligned, holdout_months=holdout_months)
        ),
        "arrays/monthly_returns.npy": _npy_bytes(
            aligned.monthly_returns.to_numpy(dtype=np.float64)
        ),
        "arrays/daily_returns.npy": _npy_bytes(
            aligned.daily_returns.to_numpy(dtype=np.float64)
        ),
        "arrays/macro_features.npy": _npy_bytes(
            aligned.macro_features.to_numpy(dtype=np.float64)
        ),
        "arrays/monthly_period_ordinals.npy": _npy_bytes(
            aligned.monthly_returns.index.asi8.astype(np.int64)
        ),
        "arrays/daily_session_ns.npy": _npy_bytes(
            aligned.daily_returns.index.asi8.astype(np.int64)
        ),
        "arrays/monthly_continuation.npy": _npy_bytes(monthly_continuation),
        "arrays/daily_continuation.npy": _npy_bytes(daily_continuation),
    }


def _model_month_eligibility(
    assets: AssetTransformResult, macro: MacroTransformResult
) -> pd.DataFrame:
    months = assets.month_coverage.index.union(macro.features.index).sort_values()
    asset_complete = assets.month_coverage["complete"].reindex(months, fill_value=False)
    macro_complete = pd.Series(months.isin(macro.features.index), index=months)
    eligible = asset_complete.astype(bool) & macro_complete.astype(bool)
    reasons = np.select(
        [
            (~asset_complete.astype(bool)) & (~macro_complete.astype(bool)),
            ~asset_complete.astype(bool),
            ~macro_complete.astype(bool),
        ],
        ["asset_and_macro_incomplete", "asset_incomplete", "macro_incomplete"],
        default="eligible",
    )
    result = pd.DataFrame(
        {
            "asset_complete": asset_complete.astype(bool),
            "macro_complete": macro_complete.astype(bool),
            "model_eligible": eligible.astype(bool),
            "reason": reasons,
        },
        index=months,
    )
    result.index.name = "month"
    return result


def _monthly_continuation(index: pd.PeriodIndex) -> np.ndarray:
    ordinals = index.asi8
    result = np.zeros(len(index), dtype=np.bool_)
    if len(index) > 1:
        result[1:] = np.diff(ordinals) == 1
    return result


def _daily_continuation(
    index: pd.DatetimeIndex, expected_sessions: pd.DatetimeIndex
) -> np.ndarray:
    positions = expected_sessions.get_indexer(index)
    if bool((positions < 0).any()):
        raise DataValidationError(
            "eligible daily return has no expected XNYS session position"
        )
    result = np.zeros(len(index), dtype=np.bool_)
    if len(index) > 1:
        result[1:] = np.diff(positions) == 1
    return result


def _descriptive_statistics(
    aligned: AlignedHistoricalData, *, holdout_months: int
) -> dict[str, Any]:
    if holdout_months <= 0 or holdout_months >= len(aligned.monthly_returns):
        raise DataValidationError(
            "design holdout leaves no training observations",
            details={
                "holdout_months": holdout_months,
                "available_months": len(aligned.monthly_returns),
            },
        )
    design_months = aligned.monthly_returns.index[:-holdout_months]
    holdout = aligned.monthly_returns.index[-holdout_months:]
    training_daily = aligned.daily_returns.loc[
        aligned.daily_returns.index.to_period("M").isin(design_months)
    ]
    return {
        "schema_version": 1,
        "monthly_returns": _matrix_statistics(
            aligned.monthly_returns, annualization=12
        ),
        "daily_returns": _matrix_statistics(aligned.daily_returns, annualization=252),
        "macro_features_unstandardized": _matrix_statistics(
            aligned.macro_features, annualization=None
        ),
        "taxable_minus_muni_spread": {
            "monthly": _series_statistics(
                aligned.monthly_returns["taxable_bond"]
                - aligned.monthly_returns["muni_bond"],
                annualization=12,
            ),
            "daily": _series_statistics(
                aligned.daily_returns["taxable_bond"]
                - aligned.daily_returns["muni_bond"],
                annualization=252,
            ),
        },
        "design_split": {
            "policy": "sealed_last_complete_months",
            "holdout_months": holdout_months,
            "training_months": len(design_months),
            "training_daily_vectors": len(training_daily),
            "training_first_month": str(design_months.min()),
            "training_last_month": str(design_months.max()),
            "holdout_first_month": str(holdout.min()),
            "holdout_last_month": str(holdout.max()),
        },
    }


def _matrix_statistics(
    frame: pd.DataFrame, *, annualization: int | None
) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "columns": {
            str(column): _series_statistics(frame[column], annualization=annualization)
            for column in frame.columns
        },
        "correlation": {
            str(row): {str(column): float(value) for column, value in values.items()}
            for row, values in frame.corr().to_dict(orient="index").items()
        },
    }


def _series_statistics(
    series: pd.Series, *, annualization: int | None
) -> dict[str, Any]:
    clean = series.astype(float)
    result: dict[str, Any] = {
        "count": len(clean),
        "mean": float(clean.mean()),
        "sample_standard_deviation": float(clean.std(ddof=1)),
        "minimum": float(clean.min()),
        "quantiles": {
            f"{quantile:.2f}": float(clean.quantile(quantile))
            for quantile in (0.01, 0.05, 0.50, 0.95, 0.99)
        },
        "maximum": float(clean.max()),
    }
    if annualization is not None:
        result["annualized_log_mean"] = annualization * result["mean"]
        result["annualized_volatility"] = (
            np.sqrt(annualization) * result["sample_standard_deviation"]
        )
    return result


def _qc_report(
    config: PRPGConfig,
    raw_manifest: Mapping[str, Any],
    assets: AssetTransformResult,
    macro: MacroTransformResult,
    aligned: AlignedHistoricalData,
) -> dict[str, Any]:
    gap_counts = {
        column.removesuffix("_gap"): int(assets.gap_mask[column].sum())
        for column in assets.gap_mask.columns
        if column.endswith("_gap")
    }
    spread_pass_columns = [
        column
        for column in macro.daily_coverage.columns
        if str(column).startswith("passed")
    ]
    failed_coverage = {
        column: int((~macro.daily_coverage[column].fillna(False).astype(bool)).sum())
        for column in spread_pass_columns
    }
    return {
        "status": "passed",
        "raw_snapshot_fingerprint": raw_manifest["snapshot_fingerprint"],
        "raw_source_count": len(raw_manifest["files"]),
        "asset": {
            **dict(assets.diagnostics),
            "ticker_gap_counts_within_observed_range": gap_counts,
            "incomplete_months": int((~assets.month_coverage["complete"]).sum()),
        },
        "macro": {
            **dict(macro.diagnostics),
            "failed_daily_coverage_months": failed_coverage,
        },
        "aligned": {
            **dict(aligned.diagnostics),
            "monthly_source_segments": int(
                (~_monthly_continuation(aligned.monthly_returns.index)).sum()
            ),
            "daily_source_segments": int(
                (
                    ~_daily_continuation(
                        aligned.daily_returns.index, assets.gap_mask.index
                    )
                ).sum()
            ),
        },
        "minimum_gates": {
            "monthly_required": config.data.quality.minimum_common_monthly_vectors,
            "monthly_actual": len(aligned.monthly_returns),
            "monthly_passed": (
                len(aligned.monthly_returns)
                >= config.data.quality.minimum_common_monthly_vectors
            ),
            "daily_required": config.data.quality.minimum_common_daily_vectors,
            "daily_actual": len(aligned.daily_returns),
            "daily_passed": (
                len(aligned.daily_returns)
                >= config.data.quality.minimum_common_daily_vectors
            ),
        },
        "outlier_resolution_complete": assets.outlier_flags.empty
        or bool(
            (assets.outlier_flags["resolution"] != "unresolved_release_blocker").all()
        ),
        "latest_modeling_month": str(aligned.eligible_months.max()),
        "retrospective_not_realtime": True,
        "secret_scan_passed": True,
    }


def _frame_csv_bytes(
    frame: pd.DataFrame,
    *,
    index_label: str | None = None,
    include_index: bool = True,
) -> bytes:
    copy = frame.copy(deep=False)
    if include_index and isinstance(copy.index, pd.PeriodIndex):
        copy = copy.copy()
        copy.index = copy.index.astype(str)
    buffer = io.StringIO(newline="")
    copy.to_csv(
        buffer,
        index=include_index,
        index_label=index_label if include_index else None,
        lineterminator="\n",
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _validate_relative_path(
    value: str,
    *,
    error_type: type[DataValidationError | IntegrityError] = DataValidationError,
) -> None:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in value
    ):
        raise error_type(
            "processed relative path is unsafe", details={"relative_path": value}
        )


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
