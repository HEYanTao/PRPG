"""Deterministic transformation of immutable provider tables.

This module is deliberately network-free.  It turns raw provider tables into
auditable, synchronized return and macro matrices while retaining the evidence
needed to explain every excluded date.  In particular, ticker gaps are detected
against the expected XNYS calendar *before* the three assets are intersected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from prpg.config import AssetConfig, DataQualityConfig, MacroConfig
from prpg.errors import DataValidationError

ASSET_ROLES: Final = ("equity", "muni_bond", "taxable_bond")
MACRO_FEATURES: Final = (
    "industrial_production_growth",
    "inflation_yoy",
    "yield_curve_spread",
    "credit_spread",
)
OUTLIER_RESOLUTIONS: Final = frozenset(
    {
        "valid_market_event",
        "provider_adjustment_reconciled",
        "provider_error_rejected",
        "unresolved_release_blocker",
    }
)
OutlierResolution = Literal[
    "valid_market_event",
    "provider_adjustment_reconciled",
    "provider_error_rejected",
    "unresolved_release_blocker",
]


@dataclass(frozen=True, slots=True)
class AssetTransformResult:
    """Processed asset matrices plus retained gap/outlier evidence."""

    adjusted_prices: pd.DataFrame
    daily_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    direct_monthly_returns: pd.DataFrame
    gap_mask: pd.DataFrame
    month_coverage: pd.DataFrame
    outlier_flags: pd.DataFrame
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MacroTransformResult:
    """Unstandardized monthly macro features and coverage evidence."""

    features: pd.DataFrame
    daily_coverage: pd.DataFrame
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AlignedHistoricalData:
    """Final common historical matrices eligible for model calibration."""

    monthly_returns: pd.DataFrame
    daily_returns: pd.DataFrame
    macro_features: pd.DataFrame
    eligible_months: pd.PeriodIndex
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StandardizedFeatures:
    """Z-scores and the exact training-only location/scale parameters."""

    values: pd.DataFrame
    means: pd.Series
    standard_deviations: pd.Series


def expected_xnys_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """Return pinned XNYS session labels, inclusive, as timezone-naive dates."""

    if end < start:
        raise DataValidationError(
            "XNYS session range is reversed",
            details={"start": start.isoformat(), "end": end.isoformat()},
        )
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    result = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
    result.name = "session"
    return result


def transform_assets(
    frames: Mapping[str, pd.DataFrame],
    assets: AssetConfig,
    quality: DataQualityConfig,
    *,
    request_start: date,
    cutoff: date,
    resolutions: Mapping[str, OutlierResolution] | None = None,
) -> AssetTransformResult:
    """Build synchronized one-session and complete-calendar-month returns.

    ``frames`` is keyed by provider ticker.  No missing price is filled.  A
    daily return is valid only when both prices occur on adjacent expected XNYS
    sessions.  A month enters either source pool only when every expected
    return-ending session has a synchronized three-asset vector.
    """

    role_to_ticker = _role_to_ticker(assets)
    missing_tickers = sorted(set(role_to_ticker.values()).difference(frames))
    if missing_tickers:
        raise DataValidationError(
            "asset provider tables are missing",
            details={"missing_tickers": missing_tickers},
        )

    sessions = expected_xnys_sessions(request_start, cutoff)
    normalized: dict[str, pd.DataFrame] = {}
    prices: dict[str, pd.Series] = {}
    ticker_returns: dict[str, pd.Series] = {}
    gap_columns: dict[str, pd.Series] = {}
    flags: list[dict[str, object]] = []
    resolution_map = dict(resolutions or {})

    for role in ASSET_ROLES:
        ticker = role_to_ticker[role]
        table = _normalize_asset_table(frames[ticker], ticker, cutoff)
        normalized[ticker] = table
        aligned = table.reindex(sessions)
        adjusted = pd.to_numeric(aligned["Adj Close"], errors="coerce")
        close = pd.to_numeric(aligned["Close"], errors="coerce")
        observed = adjusted.notna() & np.isfinite(adjusted) & (adjusted > 0.0)
        finite_positions = np.flatnonzero(observed.to_numpy())
        if finite_positions.size == 0:
            raise DataValidationError(
                "ticker has no finite positive adjusted prices",
                details={"ticker": ticker},
            )
        first_position = int(finite_positions[0])
        last_position = int(finite_positions[-1])
        within_observed_range = pd.Series(False, index=sessions, dtype=bool)
        within_observed_range.iloc[first_position : last_position + 1] = True
        gap = within_observed_range & ~observed
        gap_columns[f"{ticker}_observed"] = observed.astype(bool)
        gap_columns[f"{ticker}_within_observed_range"] = within_observed_range
        gap_columns[f"{ticker}_gap"] = gap.astype(bool)

        adjusted = adjusted.where(observed).astype(float)
        price_log = np.log(adjusted)
        one_session = price_log.diff().where(
            observed & observed.shift(1, fill_value=False)
        )
        one_session.name = role
        prices[role] = adjusted.rename(role)
        ticker_returns[role] = one_session

        action = _action_mask(aligned)
        adjustment_factor = adjusted / close.where(
            close.notna() & np.isfinite(close) & (close > 0.0)
        )
        adjustment_move = adjustment_factor.pct_change(fill_method=None)
        simple_return = np.expm1(one_session)
        threshold = (
            quality.equity_abs_simple_return_flag
            if role == "equity"
            else quality.bond_abs_simple_return_flag
        )
        large = simple_return.abs() > threshold
        unexplained_adjustment = (
            adjustment_move.abs() > quality.adjustment_factor_move_flag
        ) & ~action
        event_dates = sessions[
            large.fillna(False) | unexplained_adjustment.fillna(False)
        ]
        for session in event_dates:
            is_large = bool(large.loc[session])
            is_adjustment = bool(unexplained_adjustment.loc[session])
            key = _outlier_key(ticker, session, is_large, is_adjustment)
            proposed: OutlierResolution
            if key in resolution_map:
                proposed = resolution_map[key]
            elif bool(action.loc[session]):
                proposed = "provider_adjustment_reconciled"
            else:
                proposed = "unresolved_release_blocker"
            if proposed not in OUTLIER_RESOLUTIONS:
                raise DataValidationError(
                    "outlier resolution is invalid",
                    details={"event_key": key, "resolution": proposed},
                )
            flags.append(
                {
                    "event_key": key,
                    "session": session,
                    "role": role,
                    "ticker": ticker,
                    "large_return_flag": is_large,
                    "adjustment_factor_flag": is_adjustment,
                    "log_return": float(one_session.loc[session]),
                    "simple_return": float(simple_return.loc[session]),
                    "adjustment_factor_move": _optional_float(
                        adjustment_move.loc[session]
                    ),
                    "dividend": _optional_float(aligned.loc[session, "Dividends"]),
                    "stock_split": _optional_float(
                        aligned.loc[session, "Stock Splits"]
                    ),
                    "resolution": proposed,
                }
            )

    adjusted_prices = pd.DataFrame(prices, index=sessions)
    all_returns = pd.DataFrame(ticker_returns, index=sessions)
    common_daily = all_returns.dropna(how="any")
    gap_mask = pd.DataFrame(gap_columns, index=sessions)
    gap_mask["common_observed"] = adjusted_prices.notna().all(axis=1)
    gap_mask["common_return_valid"] = all_returns.notna().all(axis=1)

    coverage, complete_months = _asset_month_coverage(sessions, common_daily.index)
    daily_periods = common_daily.index.to_period("M")
    daily_returns = common_daily.loc[daily_periods.isin(complete_months)].copy()
    daily_returns.index.name = "session"
    monthly_returns = daily_returns.groupby(daily_returns.index.to_period("M")).sum()
    monthly_returns.index.name = "month"
    direct_monthly = _direct_monthly_returns(adjusted_prices, sessions, complete_months)
    if not monthly_returns.index.equals(direct_monthly.index):
        raise DataValidationError("monthly return derivations have different periods")
    monthly_error = (monthly_returns - direct_monthly).abs()
    max_monthly_error = float(monthly_error.to_numpy().max(initial=0.0))
    if max_monthly_error > 1e-12:
        raise DataValidationError(
            "summed daily logs do not match direct monthly price ratios",
            details={"maximum_absolute_error": max_monthly_error},
        )

    outlier_flags = pd.DataFrame.from_records(flags, columns=_outlier_columns())
    if not outlier_flags.empty:
        outlier_flags = outlier_flags.sort_values(
            ["session", "ticker", "event_key"]
        ).reset_index(drop=True)
    diagnostics: dict[str, object] = {
        "calendar": "XNYS",
        "request_start": request_start.isoformat(),
        "cutoff": cutoff.isoformat(),
        "expected_sessions": len(sessions),
        "complete_months": len(complete_months),
        "eligible_daily_vectors_before_macro_alignment": len(daily_returns),
        "eligible_monthly_vectors_before_macro_alignment": len(monthly_returns),
        "maximum_monthly_crosscheck_error": max_monthly_error,
        "outlier_flag_count": len(outlier_flags),
        "unresolved_outlier_count": int(
            (outlier_flags.get("resolution") == "unresolved_release_blocker").sum()
            if not outlier_flags.empty
            else 0
        ),
        "ticker_rows_after_cutoff": {
            ticker: len(normalized[ticker]) for ticker in sorted(normalized)
        },
    }
    return AssetTransformResult(
        adjusted_prices=adjusted_prices,
        daily_returns=daily_returns,
        monthly_returns=monthly_returns,
        direct_monthly_returns=direct_monthly,
        gap_mask=gap_mask,
        month_coverage=coverage,
        outlier_flags=outlier_flags,
        diagnostics=diagnostics,
    )


def transform_macro(
    frames: Mapping[str, pd.DataFrame],
    macro: MacroConfig,
    *,
    cutoff: date,
    daily_coverage_fraction: float,
) -> MacroTransformResult:
    """Create the approved retrospective four-feature monthly macro matrix."""

    role_to_series = _macro_role_to_series(macro)
    missing = sorted(set(role_to_series.values()).difference(frames))
    if missing:
        raise DataValidationError(
            "macro provider tables are missing", details={"missing_series": missing}
        )
    if not 0.0 < daily_coverage_fraction <= 1.0:
        raise DataValidationError(
            "daily macro coverage fraction is invalid",
            details={"daily_coverage_fraction": daily_coverage_fraction},
        )

    indpro = _monthly_source(
        frames[macro.industrial_production], macro.industrial_production, cutoff
    )
    cpi = _monthly_source(frames[macro.inflation], macro.inflation, cutoff)
    yield_mean, yield_coverage = _daily_monthly_mean(
        frames[macro.yield_curve_spread],
        macro.yield_curve_spread,
        cutoff,
        daily_coverage_fraction,
    )
    credit_mean, credit_coverage = _daily_monthly_mean(
        frames[macro.credit_spread],
        macro.credit_spread,
        cutoff,
        daily_coverage_fraction,
    )

    features = pd.concat(
        [
            (100.0 * np.log(indpro / indpro.shift(1))).rename(
                "industrial_production_growth"
            ),
            (100.0 * np.log(cpi / cpi.shift(12))).rename("inflation_yoy"),
            yield_mean.rename("yield_curve_spread"),
            credit_mean.rename("credit_spread"),
        ],
        axis=1,
        join="outer",
    ).sort_index()
    features = features.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    features.index.name = "month"
    if features.empty:
        raise DataValidationError("macro transformation produced no complete months")

    coverage = yield_coverage.join(
        credit_coverage,
        how="outer",
        lsuffix=f"_{macro.yield_curve_spread}",
        rsuffix=f"_{macro.credit_spread}",
    ).sort_index()
    coverage.index.name = "month"
    diagnostics: dict[str, object] = {
        "vintage_policy": "current_vintage_retrospective",
        "cutoff": cutoff.isoformat(),
        "daily_coverage_fraction": daily_coverage_fraction,
        "complete_feature_months": len(features),
        "first_complete_month": str(features.index.min()),
        "last_complete_month": str(features.index.max()),
    }
    return MacroTransformResult(
        features=features,
        daily_coverage=coverage,
        diagnostics=diagnostics,
    )


def align_historical_data(
    assets: AssetTransformResult,
    macro: MacroTransformResult,
    quality: DataQualityConfig,
) -> AlignedHistoricalData:
    """Intersect complete asset months with complete retrospective features."""

    eligible = assets.monthly_returns.index.intersection(macro.features.index)
    eligible = pd.PeriodIndex(eligible, freq="M").sort_values()
    monthly = assets.monthly_returns.loc[eligible].copy()
    features = macro.features.loc[eligible].copy()
    daily_months = assets.daily_returns.index.to_period("M")
    daily = assets.daily_returns.loc[daily_months.isin(eligible)].copy()

    _require_finite_matrix(monthly, "aligned monthly returns")
    _require_finite_matrix(daily, "aligned daily returns")
    _require_finite_matrix(features, "aligned macro features")
    if not monthly.index.equals(features.index):
        raise DataValidationError("monthly return and macro indices are not identical")
    if len(monthly) < quality.minimum_common_monthly_vectors:
        raise DataValidationError(
            "common monthly support is below the frozen minimum",
            details={
                "actual": len(monthly),
                "required": quality.minimum_common_monthly_vectors,
            },
        )
    if len(daily) < quality.minimum_common_daily_vectors:
        raise DataValidationError(
            "common daily support is below the frozen minimum",
            details={
                "actual": len(daily),
                "required": quality.minimum_common_daily_vectors,
            },
        )
    unresolved = assets.outlier_flags.get("resolution")
    unresolved_count = int(
        (unresolved == "unresolved_release_blocker").sum()
        if unresolved is not None
        else 0
    )
    if unresolved_count:
        raise DataValidationError(
            "asset outlier reconciliation is incomplete",
            details={"unresolved_release_blockers": unresolved_count},
        )

    diagnostics: dict[str, object] = {
        "monthly_vectors": len(monthly),
        "daily_vectors": len(daily),
        "first_month": str(eligible.min()),
        "last_month": str(eligible.max()),
        "excluded_asset_complete_months_missing_macro": len(
            assets.monthly_returns.index.difference(eligible)
        ),
        "macro_complete_months_without_asset_month": len(
            macro.features.index.difference(eligible)
        ),
    }
    return AlignedHistoricalData(
        monthly_returns=monthly,
        daily_returns=daily,
        macro_features=features,
        eligible_months=eligible,
        diagnostics=diagnostics,
    )


def standardize_features(
    features: pd.DataFrame,
    *,
    training_index: pd.Index | None = None,
) -> StandardizedFeatures:
    """Fit population z-score parameters on an explicit training subset."""

    _require_finite_matrix(features, "macro features for standardization")
    training = features if training_index is None else features.loc[training_index]
    if training.empty:
        raise DataValidationError("feature-scaler training subset is empty")
    means = training.mean(axis=0)
    standard_deviations = training.std(axis=0, ddof=0)
    invalid = (~np.isfinite(standard_deviations)) | (standard_deviations <= 0.0)
    if bool(invalid.any()):
        raise DataValidationError(
            "feature scaler has a non-positive or non-finite standard deviation",
            details={"columns": list(standard_deviations.index[invalid])},
        )
    values = (features - means) / standard_deviations
    _require_finite_matrix(values, "standardized macro features")
    return StandardizedFeatures(
        values=values,
        means=means,
        standard_deviations=standard_deviations,
    )


def _normalize_asset_table(
    frame: pd.DataFrame, ticker: str, cutoff: date
) -> pd.DataFrame:
    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("asset table is empty", details={"ticker": ticker})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(
            "asset table schema is missing required columns",
            details={"ticker": ticker, "missing_columns": missing},
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(
            "asset table index is not datetime", details={"ticker": ticker}
        )
    index = frame.index
    if index.tz is not None:
        index = index.tz_convert("America/New_York").tz_localize(None)
    index = index.normalize()
    if not index.is_unique or not index.is_monotonic_increasing:
        raise DataValidationError(
            "normalized asset session dates are not strictly increasing and unique",
            details={"ticker": ticker},
        )
    result = frame.copy(deep=True)
    result.index = index
    result.index.name = "session"
    result = result.loc[result.index <= pd.Timestamp(cutoff)]
    for column in ("Adj Close", "Close"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        invalid = numeric.notna() & (~np.isfinite(numeric) | (numeric <= 0.0))
        if bool(invalid.any()):
            raise DataValidationError(
                "asset table contains non-positive or non-finite prices",
                details={
                    "ticker": ticker,
                    "column": column,
                    "count": int(invalid.sum()),
                },
            )
    return result


def _asset_month_coverage(
    sessions: pd.DatetimeIndex, valid_return_sessions: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.PeriodIndex]:
    expected_counts = (
        pd.Series(1, index=sessions).groupby(sessions.to_period("M")).sum()
    )
    valid_counts = (
        pd.Series(1, index=valid_return_sessions)
        .groupby(valid_return_sessions.to_period("M"))
        .sum()
    )
    coverage = pd.DataFrame({"expected_sessions": expected_counts})
    coverage["valid_common_return_sessions"] = valid_counts.reindex(
        coverage.index, fill_value=0
    ).astype(int)
    coverage["complete"] = (
        coverage["valid_common_return_sessions"] == coverage["expected_sessions"]
    )
    coverage.index.name = "month"
    complete = pd.PeriodIndex(coverage.index[coverage["complete"]], freq="M")
    return coverage, complete


def _direct_monthly_returns(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    complete_months: pd.PeriodIndex,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    labels: list[pd.Period] = []
    session_positions = pd.Series(np.arange(len(sessions)), index=sessions)
    periods = sessions.to_period("M")
    for month in complete_months:
        month_sessions = sessions[periods == month]
        first_position = int(session_positions.loc[month_sessions[0]])
        if first_position == 0:
            continue
        prior_session = sessions[first_position - 1]
        last_session = month_sessions[-1]
        endpoints = prices.loc[[prior_session, last_session]]
        if endpoints.isna().any(axis=None):
            raise DataValidationError(
                "complete month lacks direct-return price endpoints",
                details={"month": str(month)},
            )
        rows.append(np.log(endpoints.iloc[-1] / endpoints.iloc[0]))
        labels.append(month)
    result = pd.DataFrame(
        rows,
        index=pd.PeriodIndex(labels, freq="M"),
        columns=prices.columns,
    )
    result.index.name = "month"
    return result


def _monthly_source(frame: pd.DataFrame, series_id: str, cutoff: date) -> pd.Series:
    series = _validated_macro_series(frame, series_id, cutoff)
    periods = series.index.to_period("M")
    if periods.duplicated().any():
        raise DataValidationError(
            "monthly macro source has duplicate observations within a month",
            details={"series_id": series_id},
        )
    result = pd.Series(series.to_numpy(dtype=float), index=periods, name=series_id)
    positive = result.notna() & (result <= 0.0)
    if bool(positive.any()):
        raise DataValidationError(
            "log-transformed macro source contains non-positive values",
            details={"series_id": series_id, "count": int(positive.sum())},
        )
    return result


def _daily_monthly_mean(
    frame: pd.DataFrame,
    series_id: str,
    cutoff: date,
    minimum_fraction: float,
) -> tuple[pd.Series, pd.DataFrame]:
    series = _validated_macro_series(frame, series_id, cutoff)
    first_month = series.index.min().to_period("M")
    last_month = pd.Timestamp(cutoff).to_period("M")
    months = pd.period_range(first_month, last_month, freq="M")
    means: dict[pd.Period, float] = {}
    records: list[dict[str, object]] = []
    for month in months:
        start = month.start_time.normalize()
        end = min(month.end_time.normalize(), pd.Timestamp(cutoff))
        weekdays = pd.bdate_range(start, end)
        values = series.reindex(weekdays)
        finite = values.notna() & np.isfinite(values)
        expected = len(weekdays)
        observed = int(finite.sum())
        fraction = observed / expected if expected else 0.0
        passed = fraction >= minimum_fraction
        if passed:
            means[month] = float(values.loc[finite].mean())
        records.append(
            {
                "month": month,
                "expected_weekdays": expected,
                "finite_observations": observed,
                "coverage_fraction": fraction,
                "passed": passed,
            }
        )
    mean_series = pd.Series(means, dtype=float, name=series_id).sort_index()
    mean_series.index = pd.PeriodIndex(mean_series.index, freq="M")
    coverage = pd.DataFrame.from_records(records).set_index("month")
    coverage.index = pd.PeriodIndex(coverage.index, freq="M")
    return mean_series, coverage


def _validated_macro_series(
    frame: pd.DataFrame, series_id: str, cutoff: date
) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError(
            "macro source table is empty", details={"series_id": series_id}
        )
    if series_id not in frame.columns or len(frame.columns) != 1:
        raise DataValidationError(
            "macro source schema is invalid",
            details={
                "series_id": series_id,
                "columns": [str(c) for c in frame.columns],
            },
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(
            "macro source index is not datetime", details={"series_id": series_id}
        )
    index = frame.index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = index.normalize()
    if not index.is_unique or not index.is_monotonic_increasing:
        raise DataValidationError(
            "macro dates are not strictly increasing and unique",
            details={"series_id": series_id},
        )
    values = pd.to_numeric(frame[series_id], errors="coerce")
    result = pd.Series(values.to_numpy(), index=index, name=series_id, dtype=float)
    result = result.loc[result.index <= pd.Timestamp(cutoff)]
    return result.replace([np.inf, -np.inf], np.nan)


def _action_mask(aligned: pd.DataFrame) -> pd.Series:
    dividends = pd.to_numeric(aligned["Dividends"], errors="coerce").fillna(0.0)
    splits = pd.to_numeric(aligned["Stock Splits"], errors="coerce").fillna(0.0)
    return (dividends != 0.0) | (splits != 0.0)


def _role_to_ticker(assets: AssetConfig) -> dict[str, str]:
    return {
        "equity": assets.equity,
        "muni_bond": assets.muni_bond,
        "taxable_bond": assets.taxable_bond,
    }


def _macro_role_to_series(macro: MacroConfig) -> dict[str, str]:
    return {
        "industrial_production_growth": macro.industrial_production,
        "inflation_yoy": macro.inflation,
        "yield_curve_spread": macro.yield_curve_spread,
        "credit_spread": macro.credit_spread,
    }


def _outlier_key(
    ticker: str, session: pd.Timestamp, large: bool, adjustment: bool
) -> str:
    kinds: list[str] = []
    if large:
        kinds.append("large_return")
    if adjustment:
        kinds.append("adjustment_factor")
    return f"{ticker}:{session.date().isoformat()}:{'+'.join(kinds)}"


def _outlier_columns() -> list[str]:
    return [
        "event_key",
        "session",
        "role",
        "ticker",
        "large_return_flag",
        "adjustment_factor_flag",
        "log_return",
        "simple_return",
        "adjustment_factor_move",
        "dividend",
        "stock_split",
        "resolution",
    ]


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _require_finite_matrix(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise DataValidationError(f"{label} is empty")
    array = frame.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise DataValidationError(f"{label} contains NA or infinite values")
