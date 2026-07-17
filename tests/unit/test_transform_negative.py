from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import prpg.data.transform as transform_module
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
    *,
    timezone: bool = True,
) -> pd.DataFrame:
    if log_returns is None:
        log_returns = np.full(len(sessions), 0.001)
        log_returns[0] = 0.0
    adjusted = 100.0 * np.exp(np.cumsum(log_returns))
    index = sessions.tz_localize("America/New_York") if timezone else sessions
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
        index=index,
    )
    frame.index.name = "Date"
    return frame


def _asset_frames(
    start: date = date(2024, 1, 1), cutoff: date = date(2024, 3, 31)
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


def _macro_frames(
    *,
    start: str = "2022-01-01",
    end: str = "2024-03-31",
) -> dict[str, pd.DataFrame]:
    monthly = pd.date_range(start, end, freq="MS")
    weekdays = pd.bdate_range(start, end)
    return {
        "INDPRO": _macro_frame(
            "INDPRO", monthly, 100.0 * np.exp(0.001 * np.arange(len(monthly)))
        ),
        "CPIAUCSL": _macro_frame(
            "CPIAUCSL", monthly, 200.0 * np.exp(0.002 * np.arange(len(monthly)))
        ),
        "T10Y3M": _macro_frame(
            "T10Y3M", weekdays, np.linspace(0.25, 1.25, len(weekdays))
        ),
        "BAA10Y": _macro_frame(
            "BAA10Y", weekdays, np.linspace(1.5, 2.5, len(weekdays))
        ),
    }


def _aligned_inputs(
    *, months: int = 2, daily_rows_per_month: int = 1
) -> tuple[AssetTransformResult, MacroTransformResult]:
    month_index = pd.period_range("2024-01", periods=months, freq="M")
    columns = ["equity", "muni_bond", "taxable_bond"]
    monthly = pd.DataFrame(0.01, index=month_index, columns=columns)
    daily_dates: list[pd.Timestamp] = []
    for month in month_index:
        for offset in range(daily_rows_per_month):
            daily_dates.append(month.start_time + pd.offsets.BDay(offset + 1))
    daily = pd.DataFrame(0.001, index=pd.DatetimeIndex(daily_dates), columns=columns)
    assets = AssetTransformResult(
        adjusted_prices=pd.DataFrame(),
        daily_returns=daily,
        monthly_returns=monthly,
        direct_monthly_returns=monthly.copy(),
        gap_mask=pd.DataFrame(),
        month_coverage=pd.DataFrame(),
        outlier_flags=pd.DataFrame(),
        diagnostics={},
    )
    macro = MacroTransformResult(
        features=pd.DataFrame(
            np.arange(months * 4, dtype=float).reshape(months, 4),
            index=month_index,
            columns=list("abcd"),
        ),
        daily_coverage=pd.DataFrame(),
        diagnostics={},
    )
    return assets, macro


def test_reversed_exchange_calendar_fails_with_requested_range() -> None:
    with pytest.raises(DataValidationError, match="range is reversed") as caught:
        expected_xnys_sessions(date(2024, 2, 1), date(2024, 1, 1))

    assert caught.value.details == {"start": "2024-02-01", "end": "2024-01-01"}


def test_asset_transform_lists_all_missing_provider_tables() -> None:
    _, frames = _asset_frames()
    del frames["MUB"]
    del frames["AGG"]

    with pytest.raises(DataValidationError, match="tables are missing") as caught:
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )

    assert caught.value.details["missing_tickers"] == ["AGG", "MUB"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[0:0], "asset table is empty"),
        (lambda _frame: cast(Any, []), "asset table is empty"),
        (
            lambda frame: frame.drop(columns="Adj Close"),
            "schema is missing required columns",
        ),
        (lambda frame: frame.reset_index(drop=True), "index is not datetime"),
        (
            lambda frame: pd.concat([frame.iloc[[0]], frame]),
            "not strictly increasing and unique",
        ),
        (lambda frame: frame.iloc[::-1], "not strictly increasing and unique"),
    ],
    ids=["empty", "wrong-type", "missing-column", "index", "duplicate", "reverse"],
)
def test_asset_table_schema_and_timestamp_failures(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    _, frames = _asset_frames()
    frames["ACWI"] = mutate(frames["ACWI"])

    with pytest.raises(DataValidationError, match=message):
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )


def test_normalization_rejects_distinct_times_collapsing_to_one_session() -> None:
    _, frames = _asset_frames()
    first = frames["ACWI"].iloc[[0]].copy()
    first.index = first.index + pd.Timedelta(hours=12)
    frames["ACWI"] = pd.concat(
        [frames["ACWI"].iloc[[0]], first, frames["ACWI"].iloc[1:]]
    )

    with pytest.raises(DataValidationError, match="not strictly increasing and unique"):
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )


@pytest.mark.parametrize("column", ["Adj Close", "Close"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.inf, -np.inf])
def test_asset_transform_rejects_nonpositive_and_nonfinite_prices(
    column: str,
    bad_value: float,
) -> None:
    _, frames = _asset_frames()
    frames["ACWI"].iloc[2, frames["ACWI"].columns.get_loc(column)] = bad_value

    with pytest.raises(
        DataValidationError, match="non-positive or non-finite"
    ) as caught:
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )

    assert caught.value.details["column"] == column
    assert caught.value.details["count"] == 1


def test_ticker_with_only_post_cutoff_rows_has_no_observed_prices() -> None:
    _, frames = _asset_frames()
    future_sessions = expected_xnys_sessions(date(2025, 1, 1), date(2025, 3, 31))
    frames["ACWI"] = _asset_frame(future_sessions)

    with pytest.raises(DataValidationError, match="no finite positive adjusted"):
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )


def test_timezone_naive_asset_tables_follow_same_deterministic_path() -> None:
    sessions, _ = _asset_frames()
    frames = {
        ticker: _asset_frame(sessions, timezone=False)
        for ticker in ("ACWI", "MUB", "AGG")
    }

    result = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=date(2024, 1, 1),
        cutoff=date(2024, 3, 31),
    )

    assert not result.monthly_returns.empty
    assert result.adjusted_prices.index.tz is None


def test_large_return_with_action_is_automatically_reconciled() -> None:
    sessions, frames = _asset_frames()
    event = pd.Timestamp("2024-02-15")
    position = sessions.get_loc(event)
    log_returns = np.full(len(sessions), 0.001)
    log_returns[0] = 0.0
    log_returns[position] = np.log1p(0.25)
    frames["ACWI"] = _asset_frame(sessions, log_returns)
    source_event = event.tz_localize("America/New_York")
    frames["ACWI"].loc[source_event, "Dividends"] = 0.50

    result = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=date(2024, 1, 1),
        cutoff=date(2024, 3, 31),
    )

    flag = result.outlier_flags.iloc[0]
    assert flag["event_key"] == "ACWI:2024-02-15:large_return"
    assert flag["resolution"] == "provider_adjustment_reconciled"
    assert flag["dividend"] == pytest.approx(0.5)
    assert result.diagnostics["unresolved_outlier_count"] == 0


def test_adjustment_only_outlier_records_factor_edges_and_optional_values() -> None:
    sessions, frames = _asset_frames()
    event = pd.Timestamp("2024-02-15")
    source_event = event.tz_localize("America/New_York")
    frames["ACWI"].loc[source_event, "Close"] /= 1.10
    frames["ACWI"]["Dividends"] = frames["ACWI"]["Dividends"].astype(object)
    frames["ACWI"].loc[source_event, "Dividends"] = "not-numeric"

    result = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=date(2024, 1, 1),
        cutoff=date(2024, 3, 31),
    )

    flags = result.outlier_flags[result.outlier_flags["ticker"] == "ACWI"]
    assert len(flags) == 2
    assert not flags["large_return_flag"].any()
    assert flags["adjustment_factor_flag"].all()
    assert flags.iloc[0]["event_key"].endswith(":adjustment_factor")
    assert pd.isna(flags.iloc[0]["dividend"])


def test_both_outlier_kinds_accept_explicit_provider_error_resolution() -> None:
    sessions, frames = _asset_frames()
    event = pd.Timestamp("2024-02-15")
    position = sessions.get_loc(event)
    log_returns = np.full(len(sessions), 0.001)
    log_returns[0] = 0.0
    log_returns[position] = np.log1p(0.25)
    frames["ACWI"] = _asset_frame(sessions, log_returns)
    source_event = event.tz_localize("America/New_York")
    frames["ACWI"].loc[source_event, "Close"] /= 1.10
    key = "ACWI:2024-02-15:large_return+adjustment_factor"

    result = transform_assets(
        frames,
        CONFIG.data.assets,
        _quality(),
        request_start=date(2024, 1, 1),
        cutoff=date(2024, 3, 31),
        resolutions={key: "provider_error_rejected"},
    )

    matching = result.outlier_flags[result.outlier_flags["event_key"] == key]
    assert len(matching) == 1
    assert matching.iloc[0]["resolution"] == "provider_error_rejected"


def test_invalid_outlier_override_fails_closed() -> None:
    sessions, frames = _asset_frames()
    event = pd.Timestamp("2024-02-15")
    position = sessions.get_loc(event)
    log_returns = np.full(len(sessions), 0.001)
    log_returns[0] = 0.0
    log_returns[position] = np.log1p(0.25)
    frames["ACWI"] = _asset_frame(sessions, log_returns)
    invalid = cast(Any, "silently_delete")

    with pytest.raises(DataValidationError, match="resolution is invalid") as caught:
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
            resolutions={"ACWI:2024-02-15:large_return": invalid},
        )

    assert caught.value.details["resolution"] == "silently_delete"


def test_monthly_period_mismatch_guard_rejects_inconsistent_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frames = _asset_frames()
    original = transform_module._direct_monthly_returns

    def missing_period(*args: Any, **kwargs: Any) -> pd.DataFrame:
        result = original(*args, **kwargs)
        return result.iloc[:-1]

    monkeypatch.setattr(transform_module, "_direct_monthly_returns", missing_period)
    with pytest.raises(DataValidationError, match="different periods"):
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )


def test_monthly_numeric_crosscheck_guard_rejects_inconsistent_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frames = _asset_frames()
    original = transform_module._direct_monthly_returns

    def changed_value(*args: Any, **kwargs: Any) -> pd.DataFrame:
        result = original(*args, **kwargs)
        return result + 1e-6

    monkeypatch.setattr(transform_module, "_direct_monthly_returns", changed_value)
    with pytest.raises(DataValidationError, match="do not match") as caught:
        transform_assets(
            frames,
            CONFIG.data.assets,
            _quality(),
            request_start=date(2024, 1, 1),
            cutoff=date(2024, 3, 31),
        )

    assert caught.value.details["maximum_absolute_error"] == pytest.approx(1e-6)


def test_direct_monthly_first_calendar_month_has_no_prior_endpoint() -> None:
    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        100.0,
        index=sessions,
        columns=["equity", "muni_bond", "taxable_bond"],
    )

    result = transform_module._direct_monthly_returns(
        prices, sessions, pd.PeriodIndex(["2024-01"], freq="M")
    )

    assert result.empty
    assert result.columns.tolist() == prices.columns.tolist()


def test_direct_monthly_endpoint_guard_rejects_missing_price() -> None:
    sessions = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-02-02"])
    prices = pd.DataFrame(
        100.0,
        index=sessions,
        columns=["equity", "muni_bond", "taxable_bond"],
    )
    prices.loc[pd.Timestamp("2024-01-31"), "equity"] = np.nan

    with pytest.raises(DataValidationError, match="lacks direct-return"):
        transform_module._direct_monthly_returns(
            prices, sessions, pd.PeriodIndex(["2024-02"], freq="M")
        )


def test_macro_transform_lists_all_missing_sources() -> None:
    frames = _macro_frames()
    del frames["INDPRO"]
    del frames["BAA10Y"]

    with pytest.raises(DataValidationError, match="tables are missing") as caught:
        transform_macro(
            frames,
            CONFIG.data.macro,
            cutoff=date(2024, 3, 31),
            daily_coverage_fraction=0.8,
        )

    assert caught.value.details["missing_series"] == ["BAA10Y", "INDPRO"]


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.0001])
def test_macro_coverage_fraction_must_be_in_open_closed_unit_interval(
    fraction: float,
) -> None:
    with pytest.raises(DataValidationError, match="coverage fraction is invalid"):
        transform_macro(
            _macro_frames(),
            CONFIG.data.macro,
            cutoff=date(2024, 3, 31),
            daily_coverage_fraction=fraction,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[0:0], "source table is empty"),
        (lambda _frame: cast(Any, []), "source table is empty"),
        (lambda frame: frame.assign(EXTRA=1.0), "source schema is invalid"),
        (lambda frame: frame.rename(columns={"INDPRO": "WRONG"}), "schema is invalid"),
        (lambda frame: frame.reset_index(drop=True), "index is not datetime"),
        (
            lambda frame: pd.concat([frame.iloc[[0]], frame]),
            "not strictly increasing and unique",
        ),
        (lambda frame: frame.iloc[::-1], "not strictly increasing and unique"),
    ],
    ids=[
        "empty",
        "wrong-type",
        "extra-column",
        "wrong-column",
        "index",
        "duplicate",
        "reverse",
    ],
)
def test_macro_source_schema_and_timestamp_failures(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    frames = _macro_frames()
    frames["INDPRO"] = mutate(frames["INDPRO"])

    with pytest.raises(DataValidationError, match=message):
        transform_macro(
            frames,
            CONFIG.data.macro,
            cutoff=date(2024, 3, 31),
            daily_coverage_fraction=0.8,
        )


def test_monthly_macro_rejects_two_distinct_observations_in_one_month() -> None:
    frames = _macro_frames()
    extra = frames["INDPRO"].iloc[[0]].copy()
    extra.index = pd.DatetimeIndex(["2022-01-15"])
    frames["INDPRO"] = pd.concat([frames["INDPRO"], extra]).sort_index()

    with pytest.raises(DataValidationError, match="duplicate observations"):
        transform_macro(
            frames,
            CONFIG.data.macro,
            cutoff=date(2024, 3, 31),
            daily_coverage_fraction=0.8,
        )


@pytest.mark.parametrize("series_id", ["INDPRO", "CPIAUCSL"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_log_transformed_macro_sources_must_be_positive(
    series_id: str,
    bad_value: float,
) -> None:
    frames = _macro_frames()
    frames[series_id].iloc[3, 0] = bad_value

    with pytest.raises(DataValidationError, match="non-positive") as caught:
        transform_macro(
            frames,
            CONFIG.data.macro,
            cutoff=date(2024, 3, 31),
            daily_coverage_fraction=0.8,
        )

    assert caught.value.details["series_id"] == series_id


def test_timezone_aware_macro_sources_are_normalized_to_utc_dates() -> None:
    frames = _macro_frames()
    for frame in frames.values():
        frame.index = frame.index.tz_localize("UTC")

    result = transform_macro(
        frames,
        CONFIG.data.macro,
        cutoff=date(2024, 3, 31),
        daily_coverage_fraction=0.8,
    )

    assert not result.features.empty
    assert isinstance(result.features.index, pd.PeriodIndex)


def test_macro_transform_fails_when_no_complete_feature_month_exists() -> None:
    frames = _macro_frames(start="2024-01-01", end="2024-03-31")

    with pytest.raises(DataValidationError, match="no complete months"):
        transform_macro(
            frames,
            CONFIG.data.macro,
            cutoff=date(2024, 3, 31),
            daily_coverage_fraction=0.8,
        )


@pytest.mark.parametrize(
    ("missing_count", "expected_fraction", "expected_pass", "expected_feature"),
    [(4, 0.80, True, True), (5, 0.75, False, False)],
)
def test_daily_macro_coverage_has_exact_eighty_percent_boundary(
    missing_count: int,
    expected_fraction: float,
    expected_pass: bool,
    expected_feature: bool,
) -> None:
    frames = _macro_frames(start="2023-01-01", end="2024-06-30")
    june = frames["T10Y3M"].index.to_period("M") == pd.Period("2024-06")
    june_dates = frames["T10Y3M"].index[june]
    assert len(june_dates) == 20
    frames["T10Y3M"].loc[june_dates[:missing_count], "T10Y3M"] = np.nan

    result = transform_macro(
        frames,
        CONFIG.data.macro,
        cutoff=date(2024, 6, 30),
        daily_coverage_fraction=0.8,
    )

    target = pd.Period("2024-06", freq="M")
    coverage = result.daily_coverage.loc[target]
    assert coverage["coverage_fraction_T10Y3M"] == pytest.approx(expected_fraction)
    assert bool(coverage["passed_T10Y3M"]) is expected_pass
    assert (target in result.features.index) is expected_feature


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("monthly", "aligned monthly returns contains"),
        ("daily", "aligned daily returns contains"),
        ("macro", "aligned macro features contains"),
    ],
)
def test_alignment_rejects_nonfinite_input_matrices(
    target: str,
    message: str,
) -> None:
    assets, macro = _aligned_inputs()
    if target == "monthly":
        assets.monthly_returns.iloc[0, 0] = np.nan
    elif target == "daily":
        assets.daily_returns.iloc[0, 0] = np.inf
    else:
        macro.features.iloc[0, 0] = -np.inf

    with pytest.raises(DataValidationError, match=message):
        align_historical_data(assets, macro, _quality())


def test_alignment_rejects_empty_common_month_intersection() -> None:
    assets, macro = _aligned_inputs()
    macro.features.index = pd.period_range("2030-01", periods=2, freq="M")

    with pytest.raises(DataValidationError, match="aligned monthly returns is empty"):
        align_historical_data(assets, macro, _quality())


def test_alignment_rejects_duplicate_macro_month_index() -> None:
    assets, macro = _aligned_inputs()
    duplicate = pd.concat([macro.features.iloc[[0]], macro.features]).sort_index()
    macro = MacroTransformResult(duplicate, pd.DataFrame(), {})

    with pytest.raises(DataValidationError, match="indices are not identical"):
        align_historical_data(assets, macro, _quality())


def test_alignment_enforces_monthly_support_minimum() -> None:
    assets, macro = _aligned_inputs(months=2, daily_rows_per_month=2)

    with pytest.raises(DataValidationError, match="monthly support") as caught:
        align_historical_data(assets, macro, _quality(monthly=3, daily=1))

    assert caught.value.details == {"actual": 2, "required": 3}


def test_alignment_enforces_daily_support_minimum_after_monthly_passes() -> None:
    assets, macro = _aligned_inputs(months=2, daily_rows_per_month=1)

    with pytest.raises(DataValidationError, match="daily support") as caught:
        align_historical_data(assets, macro, _quality(monthly=2, daily=3))

    assert caught.value.details == {"actual": 2, "required": 3}


def test_alignment_accepts_support_exactly_at_both_minima() -> None:
    assets, macro = _aligned_inputs(months=2, daily_rows_per_month=2)

    result = align_historical_data(assets, macro, _quality(monthly=2, daily=4))

    assert len(result.monthly_returns) == 2
    assert len(result.daily_returns) == 4


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_standardization_rejects_nonfinite_source_matrix(value: float) -> None:
    features = pd.DataFrame({"a": [0.0, value], "b": [1.0, 2.0]})

    with pytest.raises(DataValidationError, match="NA or infinite"):
        standardize_features(features)


def test_standardization_rejects_empty_source_and_empty_training_subset() -> None:
    with pytest.raises(DataValidationError, match="is empty"):
        standardize_features(pd.DataFrame(columns=["a", "b"]))

    features = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
    with pytest.raises(DataValidationError, match="training subset is empty"):
        standardize_features(features, training_index=features.index[:0])


def test_standardization_rejects_nonfinite_scale_from_finite_overflow() -> None:
    features = pd.DataFrame({"a": [1e308, -1e308], "b": [0.0, 1.0]}, index=["x", "y"])

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(DataValidationError, match="standard deviation") as caught,
    ):
        standardize_features(features)

    assert "a" in caught.value.details["columns"]


def test_standardization_rejects_overflow_in_nontraining_holdout() -> None:
    features = pd.DataFrame(
        {"a": [0.0, 1.0, 1e308], "b": [2.0, 4.0, 6.0]}, index=["a", "b", "c"]
    )

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(DataValidationError, match="standardized.*infinite"),
    ):
        standardize_features(features, training_index=features.index[:2])
