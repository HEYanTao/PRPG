from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prpg.config import DataQualityConfig, load_config
from prpg.data.transform import (
    AssetTransformResult,
    MacroTransformResult,
    align_historical_data,
    expected_xnys_sessions,
    standardize_features,
    transform_assets,
    transform_macro,
)
from prpg.errors import DataValidationError

ROOT = Path(__file__).parents[2]
CONFIG = load_config(ROOT / "configs" / "canonical.yaml")


def _quality(*, monthly: int = 1, daily: int = 1) -> DataQualityConfig:
    return CONFIG.data.quality.model_copy(
        update={
            "minimum_common_monthly_vectors": monthly,
            "minimum_common_daily_vectors": daily,
        }
    )


def _asset_frame(
    sessions: pd.DatetimeIndex,
    log_returns: np.ndarray | None = None,
) -> pd.DataFrame:
    if log_returns is None:
        log_returns = np.full(len(sessions), 0.001)
        log_returns[0] = 0.0
    adjusted = 100.0 * np.exp(np.cumsum(log_returns))
    frame = pd.DataFrame(
        {
            "Open": adjusted,
            "High": adjusted,
            "Low": adjusted,
            "Close": adjusted,
            "Adj Close": adjusted,
            "Volume": np.full(len(sessions), 1_000_000),
            "Dividends": np.zeros(len(sessions)),
            "Stock Splits": np.zeros(len(sessions)),
        },
        index=sessions.tz_localize("America/New_York"),
    )
    frame.index.name = "Date"
    return frame


def _asset_frames(
    start: date, cutoff: date
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    sessions = expected_xnys_sessions(start, cutoff)
    return sessions, {
        "ACWI": _asset_frame(sessions),
        "MUB": _asset_frame(sessions),
        "AGG": _asset_frame(sessions),
    }


def _macro_frame(
    series_id: str, dates: pd.DatetimeIndex, values: object
) -> pd.DataFrame:
    return pd.DataFrame({series_id: values}, index=dates)


def test_expected_calendar_uses_actual_xnys_sessions() -> None:
    sessions = expected_xnys_sessions(date(2024, 1, 1), date(2024, 1, 5))

    assert sessions.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert sessions.tz is None


def test_asset_transform_keeps_only_complete_months_and_crosschecks_logs() -> None:
    start = date(2024, 1, 1)
    cutoff = date(2024, 3, 31)
    sessions, frames = _asset_frames(start, cutoff)

    result = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=start,
        cutoff=cutoff,
    )

    # January is incomplete because the first provider price has no prior
    # expected-session endpoint. February and March are complete.
    assert result.monthly_returns.index.tolist() == [
        pd.Period("2024-02", freq="M"),
        pd.Period("2024-03", freq="M"),
    ]
    expected_daily = sessions[
        sessions.to_period("M").isin(result.monthly_returns.index)
    ]
    assert result.daily_returns.index.equals(expected_daily)
    pd.testing.assert_frame_equal(result.monthly_returns, result.direct_monthly_returns)
    assert result.diagnostics["maximum_monthly_crosscheck_error"] < 1e-14


def test_ticker_gap_is_visible_before_intersection_and_invalidates_both_edges() -> None:
    start = date(2024, 1, 1)
    cutoff = date(2024, 3, 31)
    sessions, frames = _asset_frames(start, cutoff)
    missing_session = pd.Timestamp("2024-02-12")
    next_session = sessions[sessions.get_loc(missing_session) + 1]
    frames["MUB"] = frames["MUB"].drop(missing_session.tz_localize("America/New_York"))

    result = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=start,
        cutoff=cutoff,
    )

    assert bool(result.gap_mask.loc[missing_session, "MUB_gap"])
    assert not bool(result.gap_mask.loc[missing_session, "common_return_valid"])
    assert not bool(result.gap_mask.loc[next_session, "common_return_valid"])
    assert pd.Period("2024-02", freq="M") not in result.monthly_returns.index
    assert missing_session not in result.daily_returns.index
    assert next_session not in result.daily_returns.index


def test_unresolved_large_return_is_retained_and_blocks_final_alignment() -> None:
    start = date(2024, 1, 1)
    cutoff = date(2024, 3, 31)
    sessions, frames = _asset_frames(start, cutoff)
    jump = sessions.get_loc(pd.Timestamp("2024-02-15"))
    log_returns = np.full(len(sessions), 0.001)
    log_returns[0] = 0.0
    log_returns[jump] = np.log1p(0.25)
    frames["ACWI"] = _asset_frame(sessions, log_returns)
    assets = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=start,
        cutoff=cutoff,
    )
    macro = MacroTransformResult(
        features=pd.DataFrame(
            np.ones((len(assets.monthly_returns), 4)),
            index=assets.monthly_returns.index,
            columns=["a", "b", "c", "d"],
        ),
        daily_coverage=pd.DataFrame(),
        diagnostics={},
    )

    assert len(assets.outlier_flags) == 1
    assert assets.outlier_flags.loc[0, "resolution"] == "unresolved_release_blocker"
    with pytest.raises(DataValidationError, match="reconciliation is incomplete"):
        align_historical_data(assets, macro, _quality())


def test_explicit_outlier_resolution_allows_alignment_without_deleting_return() -> None:
    start = date(2024, 1, 1)
    cutoff = date(2024, 3, 31)
    sessions, frames = _asset_frames(start, cutoff)
    event_date = pd.Timestamp("2024-02-15")
    jump = sessions.get_loc(event_date)
    log_returns = np.full(len(sessions), 0.001)
    log_returns[0] = 0.0
    log_returns[jump] = np.log1p(0.25)
    frames["ACWI"] = _asset_frame(sessions, log_returns)
    event_key = "ACWI:2024-02-15:large_return"
    assets = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=start,
        cutoff=cutoff,
        resolutions={event_key: "valid_market_event"},
    )
    features = pd.DataFrame(
        {
            "g": np.arange(len(assets.monthly_returns), dtype=float),
            "i": 1.0,
            "y": 2.0,
            "c": 3.0,
        },
        index=assets.monthly_returns.index,
    )
    macro = MacroTransformResult(features, pd.DataFrame(), {})

    aligned = align_historical_data(assets, macro, _quality())

    assert assets.outlier_flags.loc[0, "resolution"] == "valid_market_event"
    assert aligned.daily_returns.loc[event_date, "equity"] == pytest.approx(
        np.log1p(0.25)
    )


def test_macro_transform_applies_exact_growth_inflation_and_coverage_rules() -> None:
    monthly_dates = pd.date_range("2022-01-01", "2024-03-01", freq="MS")
    indpro_values = 100.0 * np.exp(0.01 * np.arange(len(monthly_dates)))
    cpi_values = 200.0 * np.exp(0.005 * np.arange(len(monthly_dates)))
    weekdays = pd.bdate_range("2022-01-01", "2024-03-31")
    yield_values = np.linspace(0.5, 1.5, len(weekdays))
    credit_values = np.linspace(2.0, 3.0, len(weekdays))
    frames = {
        "INDPRO": _macro_frame("INDPRO", monthly_dates, indpro_values),
        "CPIAUCSL": _macro_frame("CPIAUCSL", monthly_dates, cpi_values),
        "T10Y3M": _macro_frame("T10Y3M", weekdays, yield_values),
        "BAA10Y": _macro_frame("BAA10Y", weekdays, credit_values),
    }

    result = transform_macro(
        frames,
        CONFIG.data.macro,
        cutoff=date(2024, 3, 31),
        daily_coverage_fraction=0.80,
    )

    assert result.features.index.min() == pd.Period("2023-01", freq="M")
    assert result.features.loc[
        pd.Period("2024-03"), "industrial_production_growth"
    ] == pytest.approx(1.0)
    assert result.features.loc[pd.Period("2024-03"), "inflation_yoy"] == pytest.approx(
        6.0
    )
    assert result.daily_coverage.filter(like="passed").to_numpy().all()
    assert result.diagnostics["vintage_policy"] == "current_vintage_retrospective"


def test_macro_month_below_eighty_percent_is_not_imputed() -> None:
    monthly_dates = pd.date_range("2022-01-01", "2024-03-01", freq="MS")
    weekdays = pd.bdate_range("2022-01-01", "2024-03-31")
    missing_month = weekdays[weekdays.to_period("M") == pd.Period("2024-02")]
    frames = {
        "INDPRO": _macro_frame(
            "INDPRO", monthly_dates, np.arange(len(monthly_dates)) + 100.0
        ),
        "CPIAUCSL": _macro_frame(
            "CPIAUCSL", monthly_dates, np.arange(len(monthly_dates)) + 200.0
        ),
        "T10Y3M": _macro_frame("T10Y3M", weekdays, np.ones(len(weekdays))),
        "BAA10Y": _macro_frame("BAA10Y", weekdays, np.ones(len(weekdays)) * 2.0),
    }
    frames["T10Y3M"].loc[missing_month[:5], "T10Y3M"] = np.nan

    result = transform_macro(
        frames,
        CONFIG.data.macro,
        cutoff=date(2024, 3, 31),
        daily_coverage_fraction=0.80,
    )

    assert pd.Period("2024-02", freq="M") not in result.features.index
    assert not bool(result.daily_coverage.loc[pd.Period("2024-02"), "passed_T10Y3M"])


def test_align_filters_daily_pool_to_same_macro_eligible_months() -> None:
    months = pd.period_range("2020-01", periods=3, freq="M")
    returns = pd.DataFrame(
        np.arange(9, dtype=float).reshape(3, 3),
        index=months,
        columns=["equity", "muni_bond", "taxable_bond"],
    )
    daily_index = pd.DatetimeIndex(["2020-01-02", "2020-02-03", "2020-03-02"])
    daily = pd.DataFrame(np.ones((3, 3)), index=daily_index, columns=returns.columns)
    assets = AssetTransformResult(
        adjusted_prices=pd.DataFrame(),
        daily_returns=daily,
        monthly_returns=returns,
        direct_monthly_returns=returns,
        gap_mask=pd.DataFrame(),
        month_coverage=pd.DataFrame(),
        outlier_flags=pd.DataFrame(),
        diagnostics={},
    )
    macro_months = months[[0, 2]]
    macro = MacroTransformResult(
        features=pd.DataFrame(
            np.ones((2, 4)), index=macro_months, columns=list("abcd")
        ),
        daily_coverage=pd.DataFrame(),
        diagnostics={},
    )

    result = align_historical_data(assets, macro, _quality())

    assert result.eligible_months.equals(macro_months)
    assert result.daily_returns.index.tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-03-02"),
    ]


def test_standardizer_fits_only_explicit_training_rows() -> None:
    features = pd.DataFrame(
        {"a": [0.0, 2.0, 100.0], "b": [10.0, 14.0, -500.0]},
        index=pd.period_range("2020-01", periods=3, freq="M"),
    )

    result = standardize_features(features, training_index=features.index[:2])

    assert result.means.to_dict() == {"a": 1.0, "b": 12.0}
    assert result.standard_deviations.to_dict() == {"a": 1.0, "b": 2.0}
    assert result.values.iloc[2].to_dict() == {"a": 99.0, "b": -256.0}


def test_standardizer_rejects_constant_training_feature() -> None:
    features = pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, 2.0]})

    with pytest.raises(DataValidationError, match="standard deviation"):
        standardize_features(features)
