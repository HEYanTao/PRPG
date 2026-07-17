from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

import prpg.data.pipeline as pipeline
from prpg.config import PRPGConfig, load_config
from prpg.data.pipeline import (
    AcquisitionJournal,
    _json_value,
    acquire_raw_snapshot,
    acquisition_contract,
)
from prpg.data.providers import AcquisitionAttempt, ProviderPayload
from prpg.data.snapshot import SnapshotStore
from prpg.errors import DataAcquisitionError, DataValidationError

ROOT = Path(__file__).parents[2]


def _config(tmp_path: Path) -> PRPGConfig:
    config = load_config(ROOT / "configs" / "canonical.yaml")
    data = config.data.model_copy(update={"artifact_root": tmp_path / "artifacts"})
    return config.model_copy(update={"data": data})


def _payload(provider: str, identifier: str) -> ProviderPayload:
    if provider == "yfinance":
        index = pd.DatetimeIndex(["2026-06-29", "2026-06-30"], name="Date")
        frame = pd.DataFrame({"Adj Close": [100.0, 101.0]}, index=index)
        raw = b"Date,Adj Close\n2026-06-29,100\n2026-06-30,101\n"
    else:
        index = pd.DatetimeIndex(["2026-06-01"], name="DATE")
        frame = pd.DataFrame({identifier: [1.0]}, index=index)
        raw = f"DATE,{identifier}\n2026-06-01,1\n".encode()
    attempt = AcquisitionAttempt(
        number=1,
        started_utc="2026-07-14T12:00:00+00:00",
        finished_utc="2026-07-14T12:00:01+00:00",
        outcome="success",
    )
    return ProviderPayload(
        provider=provider,
        identifier=identifier,
        frame=frame,
        raw_bytes=raw,
        retrieved_utc="2026-07-14T12:00:01+00:00",
        request={"identifier": identifier},
        library_version="test",
        attempts=(attempt,),
    )


class FakeYahoo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, ticker: str, config: object) -> ProviderPayload:
        self.calls.append(ticker)
        return _payload("yfinance", ticker)


class FakeFred:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.fail_on = fail_on

    def fetch(
        self,
        series_id: str,
        config: object,
        *,
        start: str,
        end: str,
    ) -> ProviderPayload:
        self.calls.append((series_id, start, end))
        if series_id == self.fail_on:
            raise DataAcquisitionError(
                "test provider failed",
                details={"provider": "fred", "series_id": series_id},
            )
        return _payload("fred", series_id)


def test_complete_group_is_journaled_then_promoted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    yahoo = FakeYahoo()
    fred = FakeFred()

    result = acquire_raw_snapshot(
        config,
        yfinance_provider=yahoo,
        fred_provider=fred,
        now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    )

    assert yahoo.calls == ["ACWI", "MUB", "AGG"]
    assert fred.calls == [
        ("INDPRO", "2007-01-01", "2026-06-30"),
        ("CPIAUCSL", "2007-01-01", "2026-06-30"),
        ("T10Y3M", "2007-01-01", "2026-06-30"),
        ("BAA10Y", "2007-01-01", "2026-06-30"),
    ]
    assert result.source_count == 7
    assert (result.journal_path / "PROMOTED.json").is_file()
    assert not (result.journal_path / "FAILED.json").exists()
    assert len(tuple((result.journal_path / "events").glob("*.json"))) == 8
    assert len(result.snapshot.manifest["files"]) == 7
    assert (
        result.snapshot.manifest["acquisition_group_id"] == result.acquisition_group_id
    )


def test_successful_siblings_survive_a_later_provider_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(DataAcquisitionError) as captured:
        acquire_raw_snapshot(
            config,
            yfinance_provider=FakeYahoo(),
            fred_provider=FakeFred(fail_on="INDPRO"),
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        )

    details = captured.value.details
    journal = Path(details["journal_path"])
    assert details["completed_sources"] == 3
    assert (journal / "FAILED.json").is_file()
    assert sorted(path.name for path in (journal / "raw" / "yfinance").iterdir()) == [
        "ACWI.csv",
        "AGG.csv",
        "MUB.csv",
    ]
    events = sorted((journal / "events").glob("*.json"))
    assert len(events) == 4
    failure = json.loads(events[-1].read_text(encoding="utf-8"))
    assert failure["event"] == "provider_failed"
    assert "test provider failed" not in "".join(
        path.read_text(encoding="utf-8")
        for path in (journal / "raw" / "yfinance").iterdir()
    )
    snapshot_root = config.data.artifact_root / "data" / "sha256"
    assert not snapshot_root.exists() or not tuple(snapshot_root.iterdir())


def test_wrong_adapter_identity_fails_closed_and_is_not_promoted(
    tmp_path: Path,
) -> None:
    class WrongYahoo(FakeYahoo):
        def fetch(self, ticker: str, config: object) -> ProviderPayload:
            return _payload("yfinance", "WRONG")

    with pytest.raises(DataAcquisitionError) as captured:
        acquire_raw_snapshot(
            _config(tmp_path),
            yfinance_provider=WrongYahoo(),
            fred_provider=FakeFred(),
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        )

    assert captured.value.details["completed_sources"] == 0
    assert captured.value.details["cause"]["error_type"] == "DataValidationError"


def test_acquisition_contract_is_location_independent() -> None:
    canonical = load_config(ROOT / "configs" / "canonical.yaml")
    relocated = canonical.model_copy(
        update={
            "data": canonical.data.model_copy(
                update={"artifact_root": Path("elsewhere")}
            )
        }
    )

    first = acquisition_contract(canonical)
    second = acquisition_contract(relocated)

    assert first["data"]["cutoff"] == "2026-06-30"
    assert "artifact_root" not in first["data"]
    assert first == second


def test_journal_rejects_invalid_group_before_writing(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="identifier is invalid") as captured:
        AcquisitionJournal(
            tmp_path,
            "../not-a-valid-group",
            contract={"schema": 1},
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        )

    assert captured.value.details == {"acquisition_group_id": "../not-a-valid-group"}
    assert not (tmp_path / "data").exists()


def test_unexpected_provider_exception_is_wrapped_and_journaled(tmp_path: Path) -> None:
    class ExplodingYahoo(FakeYahoo):
        def fetch(self, ticker: str, config: object) -> ProviderPayload:
            raise RuntimeError(f"unexpected failure for {ticker}")

    with pytest.raises(DataAcquisitionError) as captured:
        acquire_raw_snapshot(
            _config(tmp_path),
            yfinance_provider=ExplodingYahoo(),
            fred_provider=FakeFred(),
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        )

    details = captured.value.details
    assert details["completed_sources"] == 0
    assert details["cause"]["error_type"] == "DataAcquisitionError"
    assert details["cause"]["details"] == {"error_type": "RuntimeError"}
    journal = Path(details["journal_path"])
    assert (journal / "FAILED.json").is_file()
    event = json.loads((journal / "events" / "001.json").read_text(encoding="utf-8"))
    assert event["event"] == "provider_failed"
    assert event["details"] == {"error_type": "RuntimeError"}


def test_wrong_fred_identity_preserves_completed_etf_payloads(tmp_path: Path) -> None:
    class WrongFred(FakeFred):
        def fetch(
            self,
            series_id: str,
            config: object,
            *,
            start: str,
            end: str,
        ) -> ProviderPayload:
            return _payload("fred", "WRONG")

    with pytest.raises(DataAcquisitionError) as captured:
        acquire_raw_snapshot(
            _config(tmp_path),
            yfinance_provider=FakeYahoo(),
            fred_provider=WrongFred(),
            now=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        )

    assert captured.value.details["completed_sources"] == 3
    cause = captured.value.details["cause"]
    assert cause["error_type"] == "DataValidationError"
    assert cause["details"]["expected_identifier"] == "INDPRO"
    assert cause["details"]["actual_identifier"] == "WRONG"


def test_default_adapter_factories_and_explicit_store_are_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yahoo = FakeYahoo()
    fred = FakeFred()
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "YFinanceProvider", lambda: yahoo)
    monkeypatch.setattr(pipeline, "FredCsvProvider", lambda: fred)

    result = acquire_raw_snapshot(
        config,
        snapshot_store=SnapshotStore(config.data.artifact_root),
    )

    assert result.source_count == 7
    assert yahoo.calls == ["ACWI", "MUB", "AGG"]
    assert len(fred.calls) == 4


@dataclass(frozen=True)
class _HelperRecord:
    value: int


class _StringOnly:
    def __str__(self) -> str:
        return "string-only"


def test_json_value_handles_all_supported_helper_shapes(tmp_path: Path) -> None:
    run_model = _config(tmp_path).run
    instant = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)

    assert _json_value(run_model)["name"] == "canonical-prpg"
    assert _json_value((Path("a/b"), instant)) == [
        "a/b",
        "2026-07-14T12:00:00+00:00",
    ]
    assert _json_value(date(2026, 7, 14)) == "2026-07-14"
    assert _json_value(_HelperRecord(value=7)) == {"value": 7}
    assert _json_value(_StringOnly()) == "string-only"
