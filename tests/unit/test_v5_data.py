from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prpg.data.providers import AcquisitionAttempt, ProviderPayload
from prpg.data.transform import expected_xnys_sessions
from prpg.errors import DataAcquisitionError, IntegrityError
from prpg.v5.config import ResolvedV5Inputs, load_v5_inputs
from prpg.v5.data import (
    acquire_v5_raw_snapshot,
    build_v5_month_library,
    load_v5_month_library,
)

ROOT = Path(__file__).resolve().parents[2]


class _FakeProvider:
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        fail_ticker: str | None = None,
    ) -> None:
        self.frames = frames
        self.fail_ticker = fail_ticker
        self.calls: list[str] = []

    def fetch(self, ticker: str, config: object) -> ProviderPayload:
        self.calls.append(ticker)
        if ticker == self.fail_ticker:
            raise DataAcquisitionError("injected provider failure")
        frame = self.frames[ticker]
        buffer = io.StringIO(newline="")
        frame.to_csv(
            buffer,
            index=True,
            index_label="Date",
            lineterminator="\n",
            date_format="%Y-%m-%dT%H:%M:%S%z",
            float_format="%.17g",
        )
        return ProviderPayload(
            provider="yfinance",
            identifier=ticker,
            frame=frame,
            raw_bytes=buffer.getvalue().encode("utf-8"),
            retrieved_utc="2026-07-16T12:00:00+00:00",
            request={"ticker": ticker},
            library_version="test",
            attempts=(
                AcquisitionAttempt(
                    number=1,
                    started_utc="2026-07-16T12:00:00+00:00",
                    finished_utc="2026-07-16T12:00:01+00:00",
                    outcome="success",
                ),
            ),
        )


def _test_inputs(tmp_path: Path) -> ResolvedV5Inputs:
    resolved = load_v5_inputs(ROOT / "configs" / "v5-canonical.yaml")
    yfinance = resolved.config.data.yfinance.model_copy(
        update={
            "start": date(2019, 12, 31),
            "end_exclusive": date(2020, 3, 1),
        }
    )
    data = resolved.config.data.model_copy(
        update={"cutoff": date(2020, 2, 29), "yfinance": yfinance}
    )
    config = resolved.config.model_copy(update={"data": data})
    config_payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return replace(
        resolved,
        config=config,
        config_fingerprint=hashlib.sha256(config_payload).hexdigest(),
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "runs",
    )


def _frames(*, drop_lqd_session: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    sessions = expected_xnys_sessions(date(2019, 12, 31), date(2020, 2, 29))
    frames: dict[str, pd.DataFrame] = {}
    for ticker, base in (("ACWI", 70.0), ("LQD", 120.0), ("MUB", 110.0)):
        index = sessions
        if ticker == "LQD" and drop_lqd_session is not None:
            index = index[index != drop_lqd_session]
        values = base * np.exp(np.arange(len(index), dtype=np.float64) * 0.001)
        frame = pd.DataFrame(
            {
                "Open": values,
                "High": values,
                "Low": values,
                "Close": values,
                "Adj Close": values,
                "Volume": np.full(len(index), 1_000, dtype=np.int64),
                "Dividends": np.zeros(len(index)),
                "Stock Splits": np.zeros(len(index)),
                "Capital Gains": np.zeros(len(index)),
            },
            index=index.tz_localize("America/New_York"),
        )
        frame.index.name = "Date"
        frames[ticker] = frame
    return frames


def test_joint_acquisition_publishes_nothing_when_one_asset_fails(
    tmp_path: Path,
) -> None:
    inputs = _test_inputs(tmp_path)
    provider = _FakeProvider(_frames(), fail_ticker="LQD")

    with pytest.raises(DataAcquisitionError, match="injected provider failure"):
        acquire_v5_raw_snapshot(inputs, provider=provider)

    assert provider.calls == ["ACWI", "LQD"]
    assert not (inputs.artifact_root / "data" / "sha256").exists()


def test_complete_month_library_is_ordered_and_replays_offline(tmp_path: Path) -> None:
    inputs = _test_inputs(tmp_path)
    missing_session = pd.Timestamp("2020-02-18")
    provider = _FakeProvider(_frames(drop_lqd_session=missing_session))

    def now() -> datetime:
        return datetime(2026, 7, 16, 12, 0, tzinfo=UTC)

    raw = acquire_v5_raw_snapshot(inputs, provider=provider, now=now)
    processed = build_v5_month_library(inputs, raw.fingerprint)
    library = load_v5_month_library(inputs, processed.fingerprint)

    assert provider.calls == ["ACWI", "LQD", "MUB"]
    assert library.feature_names == ("ACWI", "LQD", "MUB")
    assert library.month_count == 1
    month = pd.PeriodIndex.from_ordinals(library.monthly_period_ordinals, freq="M")
    assert list(month.astype(str)) == ["2020-01"]
    assert library.month_start_offsets.tolist() == [0, library.daily_count]
    np.testing.assert_allclose(
        library.daily_returns.sum(axis=0),
        library.monthly_returns[0],
        rtol=0.0,
        atol=1e-12,
    )
    eligibility = pd.read_csv(processed.path / "evidence" / "month-eligibility.csv")
    february = eligibility.loc[eligibility["month"] == "2020-02"].iloc[0]
    assert not bool(february["eligible"])
    assert int(february["excluded_session_count"]) == 2


def test_alternate_proxy_tickers_and_date_window_flow_through_month_library(
    tmp_path: Path,
) -> None:
    inputs = _test_inputs(tmp_path)
    assets = inputs.config.data.assets.model_copy(
        update={"equity": "SPY", "taxable_bond": "VCIT", "muni_bond": "VTEB"}
    )
    data = inputs.config.data.model_copy(update={"assets": assets})
    config = inputs.config.model_copy(update={"data": data})
    config_payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    changed = replace(
        inputs,
        config=config,
        config_fingerprint=hashlib.sha256(config_payload).hexdigest(),
    )
    canonical_frames = _frames()
    proxy_frames = {
        "SPY": canonical_frames["ACWI"],
        "VCIT": canonical_frames["LQD"],
        "VTEB": canonical_frames["MUB"],
    }
    provider = _FakeProvider(proxy_frames)

    raw = acquire_v5_raw_snapshot(changed, provider=provider)
    processed = build_v5_month_library(changed, raw.fingerprint)
    library = load_v5_month_library(changed, processed.fingerprint)

    assert provider.calls == ["SPY", "VCIT", "VTEB"]
    assert library.feature_names == ("SPY", "VCIT", "VTEB")
    months = pd.PeriodIndex.from_ordinals(
        library.monthly_period_ordinals, freq="M"
    ).astype(str)
    assert list(months) == ["2020-01", "2020-02"]


def test_raw_snapshot_request_contract_must_match_active_config(
    tmp_path: Path,
) -> None:
    inputs = _test_inputs(tmp_path)
    raw = acquire_v5_raw_snapshot(inputs, provider=_FakeProvider(_frames()))
    changed_yfinance = inputs.config.data.yfinance.model_copy(
        update={"timeout_seconds": 29}
    )
    changed_data = inputs.config.data.model_copy(update={"yfinance": changed_yfinance})
    changed_inputs = replace(
        inputs,
        config=inputs.config.model_copy(update={"data": changed_data}),
    )

    with pytest.raises(IntegrityError, match="acquisition contract"):
        build_v5_month_library(changed_inputs, raw.fingerprint)


def test_quality_flag_settings_rotate_processed_identity(tmp_path: Path) -> None:
    inputs = _test_inputs(tmp_path)
    raw = acquire_v5_raw_snapshot(inputs, provider=_FakeProvider(_frames()))
    original = build_v5_month_library(inputs, raw.fingerprint)
    changed_flags = inputs.config.data.quality_flags.model_copy(
        update={"equity_abs_simple_return": 0.21}
    )
    changed_data = inputs.config.data.model_copy(
        update={"quality_flags": changed_flags}
    )
    changed_inputs = replace(
        inputs,
        config=inputs.config.model_copy(update={"data": changed_data}),
    )

    changed = build_v5_month_library(changed_inputs, raw.fingerprint)

    assert changed.fingerprint != original.fingerprint
    transform = changed.manifest["identity"]["transform_contract"]
    assert transform["quality_flags"]["equity_abs_simple_return"] == 0.21
