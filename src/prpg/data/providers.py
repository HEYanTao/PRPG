"""Bounded provider adapters for the approved version-1 data contract.

Network access is isolated here.  The adapters return both an unmodified
provider table and deterministic bytes suitable for an immutable snapshot.
They never write files and never log response bodies, cookies, or credentials.
"""

from __future__ import annotations

import hashlib
import io
import logging
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from importlib.metadata import version
from typing import Any, Protocol

import httpx
import pandas as pd
import yfinance as yf
from yfinance import utils as yf_utils

from prpg.config import FredConfig, YFinanceConfig, redact_mapping
from prpg.errors import DataAcquisitionError, DataValidationError

LOGGER = logging.getLogger(__name__)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_TIMEOUT_SECONDS = 30.0
_YFINANCE_CONFIG_LOCK = threading.RLock()
YFINANCE_REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
)


class _YFinanceRedactionFilter(logging.Filter):
    """Suppress provider debug text while retaining event visibility.

    yfinance 1.5.1 can include its session crumb in DEBUG records.  Diagnostic
    rounds therefore expose the fact and level of each provider event, but not
    its untrusted message or arguments.  PRPG's own structured retry record
    carries the safe exception type and HTTP status.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = "yfinance_provider_debug_event:<redacted>"
        record.args = ()
        return True


@dataclass(frozen=True, slots=True)
class AcquisitionAttempt:
    """One redacted provider-attempt record."""

    number: int
    started_utc: str
    finished_utc: str
    outcome: str
    error_type: str | None = None
    http_status: int | None = None
    diagnostic_mode: bool = False
    delay_before_next_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderPayload:
    """A successful provider result before snapshot materialization."""

    provider: str
    identifier: str
    frame: pd.DataFrame = field(compare=False, repr=False)
    raw_bytes: bytes = field(compare=False, repr=False)
    retrieved_utc: str
    request: Mapping[str, Any]
    library_version: str
    attempts: tuple[AcquisitionAttempt, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    raw_kind: str = "provider_response"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    def manifest_record(self, *, relative_path: str) -> dict[str, Any]:
        """Return a JSON-safe manifest entry without embedding the table."""

        return {
            "provider": self.provider,
            "identifier": self.identifier,
            "relative_path": relative_path,
            "sha256": self.sha256,
            "bytes": len(self.raw_bytes),
            "rows": len(self.frame),
            "columns": [str(column) for column in self.frame.columns],
            "actual_start": _date_text(self.frame.index.min()),
            "actual_end": _date_text(self.frame.index.max()),
            "retrieved_utc": self.retrieved_utc,
            "request": _json_safe(redact_mapping(dict(self.request))),
            "library_version": self.library_version,
            "attempts": [_json_safe(asdict(attempt)) for attempt in self.attempts],
            "metadata": _json_safe(redact_mapping(dict(self.metadata))),
            "warnings": list(self.warnings),
            "raw_kind": self.raw_kind,
        }


class TickerLike(Protocol):
    def history(self, **kwargs: Any) -> pd.DataFrame: ...

    def get_history_metadata(self) -> Mapping[str, Any]: ...


class YFinanceProvider:
    """Fetch one ticker at a time with bounded, recorded retries."""

    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], TickerLike] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        uniform: Callable[[float, float], float] = random.uniform,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._ticker_factory = ticker_factory or yf.Ticker
        self._sleep = sleep
        self._monotonic = monotonic
        self._uniform = uniform
        self._now = now

    def fetch(self, ticker: str, config: YFinanceConfig) -> ProviderPayload:
        """Fetch the approved daily unadjusted/action-rich history."""

        # yfinance exposes retry and diagnostic switches as process globals.
        # Serializing this small acquisition section prevents sibling ticker
        # calls from observing or restoring one another's temporary settings.
        with _YFINANCE_CONFIG_LOCK:
            return self._fetch_locked(ticker, config)

    def _fetch_locked(self, ticker: str, config: YFinanceConfig) -> ProviderPayload:
        """Execute a fetch while holding the yfinance global-config lock."""

        request = {
            "api_method": "Ticker.history",
            "ticker": ticker,
            "start": config.start.isoformat(),
            "end_exclusive": config.end_exclusive.isoformat(),
            "interval": config.interval,
            "auto_adjust": config.auto_adjust,
            "actions": config.actions,
            "repair": config.repair,
            "keepna": config.keepna,
            "prepost": config.prepost,
            "threads": config.threads,
            "timeout_seconds": config.timeout_seconds,
            "timezone": config.timezone,
            "library_retries_per_round": config.library_retries_per_round,
            "outer_retry_caps_seconds": [5.0, 10.0],
            "max_rounds": config.max_rounds,
            "ticker_wall_clock_seconds": config.ticker_wall_clock_seconds,
            "http_429_min_cooldown_seconds": (config.http_429_min_cooldown_seconds),
        }
        started = self._monotonic()
        attempts: list[AcquisitionAttempt] = []
        old_retries = int(yf.config.network.retries)
        old_hide = bool(yf.config.debug.hide_exceptions)
        old_logging = bool(yf.config.debug.logging)
        provider_logger = logging.getLogger("yfinance")
        old_logger_level = provider_logger.level
        old_logger_handlers = tuple(provider_logger.handlers)
        old_logger_filters = tuple(provider_logger.filters)
        old_logger_propagate = provider_logger.propagate
        old_logger_disabled = provider_logger.disabled
        old_yf_logger = yf_utils.yf_logger
        old_yf_log_indented = yf_utils.yf_log_indented
        redaction_filter = _YFinanceRedactionFilter()
        provider_logger.addFilter(redaction_filter)
        yf.config.network.retries = config.library_retries_per_round
        try:
            for round_number in range(1, config.max_rounds + 1):
                diagnostic = round_number > 1
                yf.config.debug.hide_exceptions = not diagnostic
                yf.config.debug.logging = diagnostic
                attempt_start = self._now().isoformat()
                try:
                    provider_ticker = self._ticker_factory(ticker)
                    frame = provider_ticker.history(
                        start=config.start,
                        end=config.end_exclusive,
                        interval=config.interval,
                        auto_adjust=config.auto_adjust,
                        back_adjust=False,
                        actions=config.actions,
                        repair=config.repair,
                        keepna=config.keepna,
                        prepost=config.prepost,
                        rounding=False,
                        timeout=config.timeout_seconds,
                        raise_errors=True,
                    )
                    _validate_yfinance_frame(frame, ticker)
                    warnings: list[str] = []
                    try:
                        metadata = dict(provider_ticker.get_history_metadata())
                    except Exception as metadata_error:
                        # Provider metadata is useful but auxiliary.
                        metadata = {}
                        warnings.append(
                            "history_metadata_unavailable:"
                            f"{type(metadata_error).__name__}"
                        )
                    attempts.append(
                        AcquisitionAttempt(
                            number=round_number,
                            started_utc=attempt_start,
                            finished_utc=self._now().isoformat(),
                            outcome="success",
                            diagnostic_mode=diagnostic,
                        )
                    )
                    return ProviderPayload(
                        provider="yfinance",
                        identifier=ticker,
                        frame=frame.copy(deep=True),
                        raw_bytes=_frame_csv_bytes(frame),
                        retrieved_utc=self._now().isoformat(),
                        request=request,
                        library_version=version("yfinance"),
                        attempts=tuple(attempts),
                        metadata=metadata,
                        warnings=tuple(warnings),
                        raw_kind="unmodified_provider_table_csv",
                    )
                except DataValidationError as error:
                    attempts.append(
                        AcquisitionAttempt(
                            number=round_number,
                            started_utc=attempt_start,
                            finished_utc=self._now().isoformat(),
                            outcome="validation_failed",
                            error_type=type(error).__name__,
                            diagnostic_mode=diagnostic,
                        )
                    )
                    raise DataValidationError(
                        error.message,
                        details={
                            **error.details,
                            "provider": "yfinance",
                            "ticker": ticker,
                            "attempts": len(attempts),
                            "attempt_log": _attempt_records(attempts),
                        },
                    ) from error
                except Exception as error:
                    status = _status_code(error)
                    elapsed = self._monotonic() - started
                    transient = _yfinance_transient(error, status)
                    if (
                        not transient
                        or round_number >= config.max_rounds
                        or elapsed >= config.ticker_wall_clock_seconds
                    ):
                        attempts.append(
                            AcquisitionAttempt(
                                number=round_number,
                                started_utc=attempt_start,
                                finished_utc=self._now().isoformat(),
                                outcome="failed",
                                error_type=type(error).__name__,
                                http_status=status,
                                diagnostic_mode=diagnostic,
                            )
                        )
                        raise DataAcquisitionError(
                            "yfinance retry budget exhausted",
                            details={
                                "provider": "yfinance",
                                "ticker": ticker,
                                "attempts": len(attempts),
                                "last_error_type": type(error).__name__,
                                "http_status": status,
                                "transient": transient,
                                "attempt_log": _attempt_records(attempts),
                            },
                        ) from error
                    if status == 429:
                        delay = max(
                            float(config.http_429_min_cooldown_seconds),
                            _retry_after_seconds(error, now=self._now()) or 0.0,
                        )
                    else:
                        cap = min(5.0 * (2 ** (round_number - 1)), 60.0)
                        delay = float(self._uniform(0.0, cap))
                    if elapsed + delay >= config.ticker_wall_clock_seconds:
                        attempts.append(
                            AcquisitionAttempt(
                                number=round_number,
                                started_utc=attempt_start,
                                finished_utc=self._now().isoformat(),
                                outcome="failed",
                                error_type=type(error).__name__,
                                http_status=status,
                                diagnostic_mode=diagnostic,
                            )
                        )
                        raise DataAcquisitionError(
                            "yfinance wall-clock retry budget exhausted",
                            details={
                                "provider": "yfinance",
                                "ticker": ticker,
                                "attempts": len(attempts),
                                "last_error_type": type(error).__name__,
                                "http_status": status,
                                "transient": transient,
                                "attempt_log": _attempt_records(attempts),
                            },
                        ) from error
                    attempts.append(
                        AcquisitionAttempt(
                            number=round_number,
                            started_utc=attempt_start,
                            finished_utc=self._now().isoformat(),
                            outcome="retry",
                            error_type=type(error).__name__,
                            http_status=status,
                            diagnostic_mode=diagnostic,
                            delay_before_next_seconds=delay,
                        )
                    )
                    LOGGER.warning(
                        "yfinance transient failure provider=%s ticker=%s "
                        "round=%d error=%s status=%s diagnostic_mode=%s",
                        "yfinance",
                        ticker,
                        round_number,
                        type(error).__name__,
                        status,
                        diagnostic,
                    )
                    self._sleep(delay)
        finally:
            yf.config.network.retries = old_retries
            yf.config.debug.hide_exceptions = old_hide
            yf.config.debug.logging = old_logging
            # A diagnostic provider call can mutate both logging state and
            # yfinance's module-level logger cache.  Restore the exact incoming
            # state so acquisition has no process-global aftereffects.
            added_handlers = [
                handler
                for handler in provider_logger.handlers
                if handler not in old_logger_handlers
            ]
            provider_logger.handlers[:] = old_logger_handlers
            provider_logger.filters[:] = old_logger_filters
            provider_logger.setLevel(old_logger_level)
            provider_logger.propagate = old_logger_propagate
            provider_logger.disabled = old_logger_disabled
            yf_utils.yf_logger = old_yf_logger
            yf_utils.yf_log_indented = old_yf_log_indented
            for handler in added_handlers:
                handler.close()
        raise AssertionError("unreachable yfinance retry state")


class FredCsvProvider:
    """Fetch exact current-vintage FRED CSV response bytes."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        uniform: Callable[[float, float], float] = random.uniform,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._sleep = sleep
        self._uniform = uniform
        self._now = now

    def fetch(
        self,
        series_id: str,
        config: FredConfig,
        *,
        start: str,
        end: str,
    ) -> ProviderPayload:
        """Fetch one official FRED series using the frozen retry policy."""

        params = {"id": series_id, "cosd": start, "coed": end}
        request = {
            "url": FRED_CSV_URL,
            "params": params,
            "transport": config.transport,
            "timeout_seconds": FRED_TIMEOUT_SECONDS,
            "max_attempts": config.max_attempts,
            "backoff_caps_seconds": list(config.backoff_caps_seconds),
        }
        attempts: list[AcquisitionAttempt] = []
        owns_client = self._client is None
        client = self._client or httpx.Client(
            follow_redirects=True, timeout=FRED_TIMEOUT_SECONDS
        )
        try:
            for attempt_number in range(1, config.max_attempts + 1):
                attempt_start = self._now().isoformat()
                try:
                    response = client.get(FRED_CSV_URL, params=params)
                    response_status = int(response.status_code)
                    if response_status >= 400:
                        response.raise_for_status()
                    raw = bytes(response.content)
                    frame = parse_fred_csv_bytes(raw, series_id)
                    attempts.append(
                        AcquisitionAttempt(
                            number=attempt_number,
                            started_utc=attempt_start,
                            finished_utc=self._now().isoformat(),
                            outcome="success",
                            http_status=response_status,
                        )
                    )
                    return ProviderPayload(
                        provider="fred",
                        identifier=series_id,
                        frame=frame,
                        raw_bytes=raw,
                        retrieved_utc=self._now().isoformat(),
                        request=request,
                        library_version=version("httpx"),
                        attempts=tuple(attempts),
                        metadata={
                            "content_type": response.headers.get("content-type"),
                            "current_vintage_retrospective": True,
                        },
                    )
                except DataValidationError as error:
                    attempts.append(
                        AcquisitionAttempt(
                            number=attempt_number,
                            started_utc=attempt_start,
                            finished_utc=self._now().isoformat(),
                            outcome="validation_failed",
                            error_type=type(error).__name__,
                        )
                    )
                    raise DataValidationError(
                        error.message,
                        details={
                            **error.details,
                            "provider": "fred",
                            "series_id": series_id,
                            "attempts": len(attempts),
                            "attempt_log": _attempt_records(attempts),
                        },
                    ) from error
                except Exception as error:
                    error_status = _status_code(error)
                    transient = _fred_transient(error, error_status)
                    if not transient or attempt_number >= config.max_attempts:
                        attempts.append(
                            AcquisitionAttempt(
                                number=attempt_number,
                                started_utc=attempt_start,
                                finished_utc=self._now().isoformat(),
                                outcome="failed",
                                error_type=type(error).__name__,
                                http_status=error_status,
                            )
                        )
                        raise DataAcquisitionError(
                            "FRED acquisition failed",
                            details={
                                "provider": "fred",
                                "series_id": series_id,
                                "attempts": len(attempts),
                                "last_error_type": type(error).__name__,
                                "http_status": error_status,
                                "transient": transient,
                                "attempt_log": _attempt_records(attempts),
                            },
                        ) from error
                    retry_after = _retry_after_seconds(error, now=self._now())
                    cap = float(config.backoff_caps_seconds[attempt_number - 1])
                    delay = max(retry_after or 0.0, self._uniform(0.0, cap))
                    attempts.append(
                        AcquisitionAttempt(
                            number=attempt_number,
                            started_utc=attempt_start,
                            finished_utc=self._now().isoformat(),
                            outcome="retry",
                            error_type=type(error).__name__,
                            http_status=error_status,
                            delay_before_next_seconds=delay,
                        )
                    )
                    self._sleep(delay)
        finally:
            if owns_client:
                client.close()
        raise AssertionError("unreachable FRED retry state")


def _validate_yfinance_frame(frame: pd.DataFrame, ticker: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError(
            "yfinance returned an empty table",
            details={"provider": "yfinance", "ticker": ticker},
        )
    missing = sorted(set(YFINANCE_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise DataValidationError(
            "yfinance schema is missing required columns",
            details={"ticker": ticker, "missing_columns": missing},
        )
    if not frame.columns.is_unique:
        raise DataValidationError(
            "yfinance schema contains duplicate columns", details={"ticker": ticker}
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(
            "yfinance history index is not datetime",
            details={"ticker": ticker},
        )
    if (
        frame.index.hasnans
        or not frame.index.is_monotonic_increasing
        or not frame.index.is_unique
    ):
        raise DataValidationError(
            "yfinance timestamps are not strictly increasing and unique",
            details={"ticker": ticker},
        )


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=True,
        index_label=frame.index.name or "Date",
        lineterminator="\n",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def parse_yfinance_csv_bytes(raw: bytes, ticker: str) -> pd.DataFrame:
    """Reconstruct one snapshotted yfinance provider table offline."""

    if not raw:
        raise DataValidationError(
            "snapshotted yfinance table is empty",
            details={"provider": "yfinance", "ticker": ticker},
        )
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as error:
        raise DataValidationError(
            "snapshotted yfinance CSV could not be parsed",
            details={"ticker": ticker, "error_type": type(error).__name__},
        ) from error
    if frame.empty or frame.columns.empty:
        raise DataValidationError(
            "snapshotted yfinance CSV is empty", details={"ticker": ticker}
        )
    date_column = str(frame.columns[0])
    dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    if dates.isna().any():
        raise DataValidationError(
            "snapshotted yfinance CSV contains invalid dates",
            details={"ticker": ticker},
        )
    parsed = frame.drop(columns=[date_column])
    parsed.index = pd.DatetimeIndex(dates)
    parsed.index.name = date_column
    _validate_yfinance_frame(parsed, ticker)
    return parsed


def parse_fred_csv_bytes(raw: bytes, series_id: str) -> pd.DataFrame:
    """Parse one exact snapshotted current-vintage FRED CSV offline."""

    if not raw:
        raise DataValidationError(
            "FRED returned an empty response",
            details={"provider": "fred", "series_id": series_id},
        )
    try:
        frame = pd.read_csv(io.BytesIO(raw), na_values=".")
    except Exception as error:
        raise DataValidationError(
            "FRED CSV could not be parsed",
            details={"series_id": series_id, "error_type": type(error).__name__},
        ) from error
    if frame.empty or len(frame.columns) != 2:
        raise DataValidationError(
            "FRED CSV schema is invalid",
            details={"series_id": series_id, "column_count": len(frame.columns)},
        )
    date_column = str(frame.columns[0])
    value_column = str(frame.columns[1])
    if date_column not in {"DATE", "observation_date"}:
        raise DataValidationError(
            "FRED CSV date column is invalid",
            details={"series_id": series_id, "returned_column": date_column},
        )
    if value_column != series_id:
        raise DataValidationError(
            "FRED CSV series column does not match request",
            details={"series_id": series_id, "returned_column": value_column},
        )
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if dates.isna().any():
        raise DataValidationError(
            "FRED CSV contains invalid dates", details={"series_id": series_id}
        )
    invalid_values = values.isna() & frame[value_column].notna()
    if invalid_values.any():
        raise DataValidationError(
            "FRED CSV contains invalid numeric values",
            details={"series_id": series_id},
        )
    parsed = pd.DataFrame({series_id: values.to_numpy()}, index=pd.DatetimeIndex(dates))
    parsed.index.name = "DATE"
    if not parsed.index.is_unique or not parsed.index.is_monotonic_increasing:
        raise DataValidationError(
            "FRED dates are not strictly increasing and unique",
            details={"series_id": series_id},
        )
    return parsed


def _status_code(error: Exception) -> int | None:
    if type(error).__name__ == "YFRateLimitError":
        return 429
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return int(status)
    # Ticker.history can wrap a Yahoo HTTP response in YFPricesMissingError.
    # Extract only the numeric status from its private diagnostic attribute;
    # never persist the untrusted diagnostic text itself.
    debug_info = getattr(error, "debug_info", None)
    if isinstance(debug_info, str):
        match = re.search(r"Yahoo status_code\s*=\s*(\d{3})", debug_info)
        if match is not None:
            return int(match.group(1))
    return None


def _fred_transient(error: Exception, status: int | None) -> bool:
    if status is not None:
        return status in {408, 429} or 500 <= status <= 599
    return isinstance(error, httpx.TransportError)


def _yfinance_transient(error: Exception, status: int | None) -> bool:
    """Classify only frozen transient categories as retryable."""

    if status is not None:
        return status in {408, 429} or 500 <= status <= 599
    if isinstance(error, TimeoutError | ConnectionError | httpx.TransportError):
        return True
    if type(error).__name__ == "YFDataException":
        # In the approved Ticker.history route this denotes Yahoo's explicit
        # temporary-down response. Other deterministic yfinance exceptions
        # (invalid period, missing ticker/timezone) remain non-retryable.
        return True
    # yfinance's pinned curl transport exceptions do not inherit Python's
    # ConnectionError.  Match the small reviewed transient class-name set
    # across its MRO without treating every OSError as retryable.
    transient_names = {
        "ChunkedEncodingError",
        "ConnectionError",
        "DNSError",
        "ProxyError",
        "RetryError",
        "Timeout",
    }
    return any(cls.__name__ in transient_names for cls in type(error).__mro__)


def _retry_after_seconds(
    error: Exception, *, now: datetime | None = None
) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    if raw is None:
        raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            parsed_retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not isinstance(parsed_retry_at, datetime):
            return None
        retry_at = parsed_retry_at
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        current = now or datetime.now(UTC)
        seconds = (retry_at.astimezone(UTC) - current.astimezone(UTC)).total_seconds()
        return max(0.0, seconds)


def _attempt_records(attempts: list[AcquisitionAttempt]) -> list[dict[str, Any]]:
    """Return JSON-safe attempt evidence containing no exception messages."""

    return [_json_safe(asdict(attempt)) for attempt in attempts]


def _date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
