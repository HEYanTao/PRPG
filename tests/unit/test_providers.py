from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yfinance as yf
from yfinance import utils as yf_utils
from yfinance.exceptions import YFDataException, YFPricesMissingError, YFRateLimitError

from prpg.config import FredConfig, YFinanceConfig, load_config
from prpg.data.providers import (
    FRED_CSV_URL,
    AcquisitionAttempt,
    FredCsvProvider,
    ProviderPayload,
    YFinanceProvider,
)
from prpg.errors import DataAcquisitionError, DataValidationError

ROOT = Path(__file__).parents[2]
FIXED_NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def yfinance_config() -> YFinanceConfig:
    return load_config(ROOT / "configs" / "canonical.yaml").data.yfinance


@pytest.fixture(scope="module")
def fred_config() -> FredConfig:
    return load_config(ROOT / "configs" / "canonical.yaml").data.fred


@pytest.fixture
def valid_history() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2026-06-26 00:00", "2026-06-29 00:00"],
        tz="America/New_York",
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_100_000],
            "Dividends": [0.0, 0.25],
            "Stock Splits": [0.0, 0.0],
        },
        index=index,
    )


class _TickerFactory:
    def __init__(
        self,
        outcomes: list[pd.DataFrame | BaseException],
        *,
        metadata: dict[str, Any] | BaseException | None = None,
        before_history: Callable[[], None] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.metadata = metadata if metadata is not None else {"exchange": "NMS"}
        self.before_history = before_history
        self.tickers: list[str] = []
        self.history_calls: list[dict[str, Any]] = []
        self.config_states: list[tuple[int, bool, bool]] = []
        self.metadata_calls = 0

    def __call__(self, ticker: str) -> _FakeTicker:
        self.tickers.append(ticker)
        return _FakeTicker(self)


class _FakeTicker:
    def __init__(self, owner: _TickerFactory) -> None:
        self.owner = owner

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.owner.history_calls.append(kwargs)
        self.owner.config_states.append(
            (
                int(yf.config.network.retries),
                bool(yf.config.debug.hide_exceptions),
                bool(yf.config.debug.logging),
            )
        )
        if self.owner.before_history is not None:
            self.owner.before_history()
        outcome = self.owner.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def get_history_metadata(self) -> dict[str, Any]:
        self.owner.metadata_calls += 1
        if isinstance(self.owner.metadata, BaseException):
            raise self.owner.metadata
        return dict(self.owner.metadata)


def _fixed_now() -> datetime:
    return FIXED_NOW


def _http_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _with_duplicate_yfinance_column(frame: pd.DataFrame) -> pd.DataFrame:
    duplicate = frame.copy(deep=True)
    duplicate["Extra"] = 1.0
    duplicate.columns = [*frame.columns, "Close"]
    return duplicate


def _http_status_error(
    status: int, *, retry_after: str | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://query.invalid.test/history")
    headers = {"retry-after": retry_after} if retry_after is not None else None
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError(
        "provider response body is intentionally absent",
        request=request,
        response=response,
    )


def test_yfinance_success_uses_exact_frozen_kwargs_and_request_record(
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    factory = _TickerFactory([valid_history])
    provider = YFinanceProvider(
        ticker_factory=factory,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    )

    payload = provider.fetch("ACWI", yfinance_config)

    assert factory.tickers == ["ACWI"]
    assert factory.history_calls == [
        {
            "start": yfinance_config.start,
            "end": yfinance_config.end_exclusive,
            "interval": "1d",
            "auto_adjust": False,
            "back_adjust": False,
            "actions": True,
            "repair": False,
            "keepna": True,
            "prepost": False,
            "rounding": False,
            "timeout": 30,
            "raise_errors": True,
        }
    ]
    assert payload.request["api_method"] == "Ticker.history"
    assert payload.request["start"] == "2007-01-01"
    assert payload.request["end_exclusive"] == "2026-07-01"
    assert payload.request["threads"] is False
    assert payload.request["timezone"] == "America/New_York"
    assert payload.request["library_retries_per_round"] == 2
    assert payload.request["max_rounds"] == 3
    assert payload.metadata == {"exchange": "NMS"}
    assert payload.raw_kind == "unmodified_provider_table_csv"
    assert [(item.number, item.outcome) for item in payload.attempts] == [
        (1, "success")
    ]


def test_yfinance_table_bytes_are_stable_complete_and_hashed(
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    first = YFinanceProvider(
        ticker_factory=_TickerFactory([valid_history.copy(deep=True)]),
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)
    second = YFinanceProvider(
        ticker_factory=_TickerFactory([valid_history.copy(deep=True)]),
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert first.raw_bytes == second.raw_bytes
    assert first.raw_bytes.startswith(
        b"Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits\n"
    )
    assert b"2026-06-26T00:00:00-0400" in first.raw_bytes
    assert first.sha256 == hashlib.sha256(first.raw_bytes).hexdigest()
    assert list(first.frame.columns) == list(valid_history.columns)
    assert first.frame.index.equals(valid_history.index)

    record = first.manifest_record(relative_path="raw/yfinance/ACWI.csv")
    assert record["sha256"] == first.sha256
    assert record["bytes"] == len(first.raw_bytes)
    assert record["rows"] == 2
    assert record["actual_start"] == "2026-06-26T00:00:00-04:00"
    assert record["actual_end"] == "2026-06-29T00:00:00-04:00"


def test_yfinance_metadata_failure_is_nonfatal_and_redacted_to_type(
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    factory = _TickerFactory(
        [valid_history], metadata=RuntimeError("cookie=DO-NOT-STORE")
    )

    payload = YFinanceProvider(
        ticker_factory=factory,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert payload.metadata == {}
    assert payload.warnings == ("history_metadata_unavailable:RuntimeError",)
    assert "DO-NOT-STORE" not in json.dumps(
        payload.manifest_record(relative_path="ACWI.csv")
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.drop(columns="Adj Close"),
        lambda frame: frame.reset_index(drop=True),
        lambda frame: frame.iloc[::-1],
        lambda frame: frame.iloc[0:0],
        _with_duplicate_yfinance_column,
    ],
    ids=[
        "missing-column",
        "non-datetime-index",
        "unsorted",
        "empty",
        "duplicate-column",
    ],
)
def test_yfinance_schema_failures_are_never_retried(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    bad_frame = mutate(valid_history.copy(deep=True))
    factory = _TickerFactory([bad_frame, valid_history])
    sleeps: list[float] = []

    with pytest.raises(DataValidationError) as caught:
        YFinanceProvider(
            ticker_factory=factory,
            sleep=sleeps.append,
            now=_fixed_now,
            monotonic=lambda: 0.0,
        ).fetch("ACWI", yfinance_config)

    assert len(factory.history_calls) == 1
    assert sleeps == []
    assert caught.value.details["attempts"] == 1
    assert caught.value.details["attempt_log"][0]["outcome"] == "validation_failed"


def test_yfinance_transient_retry_enters_redacted_diagnostic_mode(
    caplog: pytest.LogCaptureFixture,
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    call_number = 0

    def emit_provider_debug() -> None:
        nonlocal call_number
        call_number += 1
        if call_number == 2:
            yf_utils.get_yf_logger().debug("crumb = 'TOP-SECRET-CRUMB'")

    factory = _TickerFactory(
        [TimeoutError("signed_url=TOP-SECRET-URL"), valid_history],
        before_history=emit_provider_debug,
    )
    sleeps: list[float] = []
    caplog.set_level(logging.DEBUG, logger="yfinance")

    payload = YFinanceProvider(
        ticker_factory=factory,
        sleep=sleeps.append,
        uniform=lambda _low, _high: 1.25,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert factory.config_states == [(2, True, False), (2, False, True)]
    assert sleeps == [1.25]
    assert [item.outcome for item in payload.attempts] == ["retry", "success"]
    assert [item.diagnostic_mode for item in payload.attempts] == [False, True]
    assert payload.attempts[0].error_type == "TimeoutError"
    assert "TOP-SECRET" not in caplog.text
    assert "yfinance_provider_debug_event:<redacted>" in caplog.text


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [("30", 60.0), ("75", 75.0)],
)
def test_yfinance_429_honors_minimum_cooldown_and_retry_after(
    retry_after: str,
    expected_delay: float,
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    rate_limit = _http_status_error(429, retry_after=retry_after)
    factory = _TickerFactory([rate_limit, valid_history])
    sleeps: list[float] = []

    payload = YFinanceProvider(
        ticker_factory=factory,
        sleep=sleeps.append,
        uniform=lambda _low, _high: 0.0,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert sleeps == [expected_delay]
    assert payload.attempts[0].http_status == 429
    assert payload.attempts[0].delay_before_next_seconds == expected_delay


@pytest.mark.parametrize("status", [408, 500, 503, 599])
def test_yfinance_retries_only_transient_http_statuses(
    status: int,
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    factory = _TickerFactory([_http_status_error(status), valid_history])
    sleeps: list[float] = []

    payload = YFinanceProvider(
        ticker_factory=factory,
        sleep=sleeps.append,
        uniform=lambda _low, _high: 2.0,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert len(factory.history_calls) == 2
    assert sleeps == [2.0]
    assert payload.attempts[0].http_status == status


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_yfinance_does_not_retry_nontransient_http_statuses(
    status: int,
    yfinance_config: YFinanceConfig,
) -> None:
    factory = _TickerFactory([_http_status_error(status), TimeoutError()])
    sleeps: list[float] = []

    with pytest.raises(DataAcquisitionError) as caught:
        YFinanceProvider(
            ticker_factory=factory,
            sleep=sleeps.append,
            now=_fixed_now,
            monotonic=lambda: 0.0,
        ).fetch("ACWI", yfinance_config)

    assert len(factory.history_calls) == 1
    assert sleeps == []
    assert caught.value.details["http_status"] == status
    assert caught.value.details["transient"] is False


def test_yfinance_recovers_status_from_wrapped_history_error(
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    wrapped = YFPricesMissingError(
        "ACWI", "(1d 2007-01-01 -> 2026-07-01)(Yahoo status_code = 503)"
    )
    factory = _TickerFactory([wrapped, valid_history])
    sleeps: list[float] = []

    payload = YFinanceProvider(
        ticker_factory=factory,
        sleep=sleeps.append,
        uniform=lambda _low, _high: 0.25,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert sleeps == [0.25]
    assert payload.attempts[0].http_status == 503


def test_yfinance_retries_explicit_temporary_down_signal(
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    factory = _TickerFactory([YFDataException("Yahoo temporarily down"), valid_history])

    payload = YFinanceProvider(
        ticker_factory=factory,
        sleep=lambda _delay: None,
        uniform=lambda _low, _high: 0.0,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert [attempt.outcome for attempt in payload.attempts] == ["retry", "success"]


def test_yfinance_native_rate_limit_maps_to_429_and_minimum_cooldown(
    yfinance_config: YFinanceConfig,
    valid_history: pd.DataFrame,
) -> None:
    factory = _TickerFactory([YFRateLimitError(), valid_history])
    sleeps: list[float] = []

    payload = YFinanceProvider(
        ticker_factory=factory,
        sleep=sleeps.append,
        now=_fixed_now,
        monotonic=lambda: 0.0,
    ).fetch("ACWI", yfinance_config)

    assert sleeps == [60.0]
    assert payload.attempts[0].http_status == 429


def test_yfinance_exhausts_exact_outer_budget_with_redacted_attempt_log(
    yfinance_config: YFinanceConfig,
) -> None:
    secret = "SIGNED-QUERY-AND-COOKIE"
    factory = _TickerFactory(
        [TimeoutError(secret), TimeoutError(secret), TimeoutError(secret)]
    )
    sleeps: list[float] = []
    caps: list[tuple[float, float]] = []

    def at_cap(low: float, high: float) -> float:
        caps.append((low, high))
        return high

    with pytest.raises(DataAcquisitionError) as caught:
        YFinanceProvider(
            ticker_factory=factory,
            sleep=sleeps.append,
            uniform=at_cap,
            now=_fixed_now,
            monotonic=lambda: 0.0,
        ).fetch("ACWI", yfinance_config)

    assert len(factory.history_calls) == 3
    assert caps == [(0.0, 5.0), (0.0, 10.0)]
    assert sleeps == [5.0, 10.0]
    assert caught.value.details["attempts"] == 3
    assert [item["outcome"] for item in caught.value.details["attempt_log"]] == [
        "retry",
        "retry",
        "failed",
    ]
    serialized = json.dumps(caught.value.as_event())
    assert secret not in serialized
    assert "message" not in serialized.split('"attempt_log"', maxsplit=1)[1]


def test_yfinance_deterministic_exception_is_not_retried(
    yfinance_config: YFinanceConfig,
) -> None:
    factory = _TickerFactory([ValueError("invalid parameter SECRET"), TimeoutError()])
    sleeps: list[float] = []

    with pytest.raises(DataAcquisitionError) as caught:
        YFinanceProvider(
            ticker_factory=factory,
            sleep=sleeps.append,
            now=_fixed_now,
            monotonic=lambda: 0.0,
        ).fetch("ACWI", yfinance_config)

    assert len(factory.history_calls) == 1
    assert sleeps == []
    assert caught.value.details["transient"] is False
    assert caught.value.details["attempt_log"][0]["error_type"] == "ValueError"
    assert "SECRET" not in json.dumps(caught.value.as_event())


def test_yfinance_wall_clock_cap_prevents_truncated_429_cooldown(
    yfinance_config: YFinanceConfig,
) -> None:
    factory = _TickerFactory([YFRateLimitError()])
    monotonic_values = iter([0.0, 550.0])
    sleeps: list[float] = []

    with pytest.raises(DataAcquisitionError, match="wall-clock") as caught:
        YFinanceProvider(
            ticker_factory=factory,
            sleep=sleeps.append,
            now=_fixed_now,
            monotonic=lambda: next(monotonic_values),
        ).fetch("ACWI", yfinance_config)

    assert sleeps == []
    assert caught.value.details["http_status"] == 429
    assert caught.value.details["attempt_log"][0]["outcome"] == "failed"


def test_yfinance_restores_global_config_and_logger_state_after_failure(
    yfinance_config: YFinanceConfig,
) -> None:
    logger = logging.getLogger("yfinance")
    old_config = (
        yf.config.network.retries,
        yf.config.debug.hide_exceptions,
        yf.config.debug.logging,
    )
    old_logger_state = (
        logger.level,
        tuple(logger.handlers),
        tuple(logger.filters),
        logger.propagate,
        logger.disabled,
        yf_utils.yf_logger,
        yf_utils.yf_log_indented,
    )

    with pytest.raises(DataAcquisitionError):
        YFinanceProvider(
            ticker_factory=_TickerFactory([TimeoutError(), ValueError("stop")]),
            sleep=lambda _delay: None,
            uniform=lambda _low, _high: 0.0,
            now=_fixed_now,
            monotonic=lambda: 0.0,
        ).fetch("ACWI", yfinance_config)

    assert (
        yf.config.network.retries,
        yf.config.debug.hide_exceptions,
        yf.config.debug.logging,
    ) == old_config
    assert (
        logger.level,
        tuple(logger.handlers),
        tuple(logger.filters),
        logger.propagate,
        logger.disabled,
        yf_utils.yf_logger,
        yf_utils.yf_log_indented,
    ) == old_logger_state


def test_manifest_recursively_redacts_secret_like_request_and_metadata_keys(
    valid_history: pd.DataFrame,
) -> None:
    payload = ProviderPayload(
        provider="test",
        identifier="ACWI",
        frame=valid_history,
        raw_bytes=b"raw",
        retrieved_utc=FIXED_NOW.isoformat(),
        request={"params": {"api_key": "SECRET", "id": "ACWI"}},
        library_version="1",
        attempts=(
            AcquisitionAttempt(
                number=1,
                started_utc=FIXED_NOW.isoformat(),
                finished_utc=FIXED_NOW.isoformat(),
                outcome="success",
            ),
        ),
        metadata={"session": {"cookie": "SECRET", "exchange": "NMS"}},
    )

    record = payload.manifest_record(relative_path="raw.csv")

    assert record["request"]["params"]["api_key"] == "<redacted>"
    assert record["metadata"]["session"]["cookie"] == "<redacted>"
    assert "SECRET" not in json.dumps(record)


@pytest.mark.parametrize("date_header", ["observation_date", "DATE"])
def test_fred_success_preserves_exact_bytes_and_parses_supported_date_headers(
    date_header: str,
    fred_config: FredConfig,
) -> None:
    raw = f"{date_header},INDPRO\n2026-04-01,100.25\n2026-05-01,.\n".encode()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=raw,
            headers={"content-type": "text/csv;charset=UTF-8"},
        )

    with _http_client(handler) as client:
        payload = FredCsvProvider(client=client, now=_fixed_now).fetch(
            "INDPRO", fred_config, start="2007-01-01", end="2026-06-30"
        )

    assert len(requests) == 1
    assert str(requests[0].url.copy_with(query=None)) == FRED_CSV_URL
    assert dict(requests[0].url.params) == {
        "id": "INDPRO",
        "cosd": "2007-01-01",
        "coed": "2026-06-30",
    }
    assert payload.raw_bytes == raw
    assert payload.sha256 == hashlib.sha256(raw).hexdigest()
    assert payload.frame.index.name == "DATE"
    assert payload.frame.iloc[0, 0] == pytest.approx(100.25)
    assert pd.isna(payload.frame.iloc[1, 0])
    assert payload.request["transport"] == "official_csv"
    assert payload.request["timeout_seconds"] == 30.0
    assert payload.request["max_attempts"] == 5
    assert payload.request["backoff_caps_seconds"] == [2, 5, 15, 45]
    assert payload.metadata["current_vintage_retrospective"] is True
    assert payload.attempts[0].http_status == 200


@pytest.mark.parametrize(
    "raw",
    [
        b"wrong_date,INDPRO\n2026-01-01,1\n",
        b"observation_date,WRONG\n2026-01-01,1\n",
        b"observation_date,INDPRO,EXTRA\n2026-01-01,1,2\n",
        b"observation_date,INDPRO\nnot-a-date,1\n",
        b"observation_date,INDPRO\n2026-01-01,not-a-number\n",
        b"",
    ],
    ids=[
        "date-column",
        "series-column",
        "extra-column",
        "bad-date",
        "bad-value",
        "empty",
    ],
)
def test_fred_schema_failures_are_never_retried(
    raw: bytes,
    fred_config: FredConfig,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=raw)

    sleeps: list[float] = []
    with (
        _http_client(handler) as client,
        pytest.raises(DataValidationError) as caught,
    ):
        FredCsvProvider(client=client, sleep=sleeps.append, now=_fixed_now).fetch(
            "INDPRO", fred_config, start="2007-01-01", end="2026-06-30"
        )

    assert calls == 1
    assert sleeps == []
    assert caught.value.details["attempts"] == 1
    assert caught.value.details["attempt_log"][0]["outcome"] == "validation_failed"


def test_fred_retries_transient_status_and_transport_then_succeeds(
    fred_config: FredConfig,
) -> None:
    calls = 0
    raw = b"observation_date,T10Y3M\n2026-06-02,0.42\n"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"retry-after": "3"})
        if calls == 2:
            raise httpx.ReadTimeout("SECRET BODY", request=request)
        return httpx.Response(200, content=raw)

    sleeps: list[float] = []
    jitter = iter([1.0, 4.0])
    with _http_client(handler) as client:
        payload = FredCsvProvider(
            client=client,
            sleep=sleeps.append,
            uniform=lambda _low, _high: next(jitter),
            now=_fixed_now,
        ).fetch("T10Y3M", fred_config, start="2007-01-01", end="2026-06-30")

    assert calls == 3
    assert sleeps == [3.0, 4.0]
    assert [attempt.outcome for attempt in payload.attempts] == [
        "retry",
        "retry",
        "success",
    ]
    assert [attempt.http_status for attempt in payload.attempts] == [503, None, 200]


def test_fred_retry_after_http_date_is_honored(fred_config: FredConfig) -> None:
    calls = 0
    raw = b"observation_date,BAA10Y\n2026-06-02,1.1\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "Tue, 14 Jul 2026 12:00:10 GMT"},
            )
        return httpx.Response(200, content=raw)

    sleeps: list[float] = []
    with _http_client(handler) as client:
        FredCsvProvider(
            client=client,
            sleep=sleeps.append,
            uniform=lambda _low, _high: 0.5,
            now=_fixed_now,
        ).fetch("BAA10Y", fred_config, start="2007-01-01", end="2026-06-30")

    assert sleeps == [10.0]


def test_fred_exhausts_exact_five_attempt_budget_with_redacted_log(
    fred_config: FredConfig,
) -> None:
    secret = "RESPONSE-BODY-SECRET"
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, content=secret.encode())

    sleeps: list[float] = []
    with (
        _http_client(handler) as client,
        pytest.raises(DataAcquisitionError) as caught,
    ):
        FredCsvProvider(
            client=client,
            sleep=sleeps.append,
            uniform=lambda _low, high: high,
            now=_fixed_now,
        ).fetch("INDPRO", fred_config, start="2007-01-01", end="2026-06-30")

    assert calls == 5
    assert sleeps == [2.0, 5.0, 15.0, 45.0]
    assert [item["outcome"] for item in caught.value.details["attempt_log"]] == [
        "retry",
        "retry",
        "retry",
        "retry",
        "failed",
    ]
    assert [item["http_status"] for item in caught.value.details["attempt_log"]] == [
        500,
        500,
        500,
        500,
        500,
    ]
    assert secret not in json.dumps(caught.value.as_event())


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_fred_nontransient_http_failures_are_not_retried(
    status: int,
    fred_config: FredConfig,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, content=b"SECRET")

    sleeps: list[float] = []
    with (
        _http_client(handler) as client,
        pytest.raises(DataAcquisitionError) as caught,
    ):
        FredCsvProvider(client=client, sleep=sleeps.append, now=_fixed_now).fetch(
            "INDPRO", fred_config, start="2007-01-01", end="2026-06-30"
        )

    assert calls == 1
    assert sleeps == []
    assert caught.value.details["transient"] is False
    assert caught.value.details["attempt_log"][0]["http_status"] == status
    assert "SECRET" not in json.dumps(caught.value.as_event())
