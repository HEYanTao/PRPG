"""Orchestration for the only network-enabled PRPG command.

Provider adapters remain independent and sequential because yfinance exposes
process-global retry/debug configuration.  Each successful response is written
to a noncanonical acquisition journal immediately; only a complete seven-source
group can be promoted to the immutable content-addressed snapshot store.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from prpg import __version__
from prpg.config import FredConfig, PRPGConfig, YFinanceConfig
from prpg.data.providers import FredCsvProvider, ProviderPayload, YFinanceProvider
from prpg.data.snapshot import SnapshotReference, SnapshotStore
from prpg.errors import DataAcquisitionError, DataValidationError, PRPGError

_GROUP_PATTERN = re.compile(r"acq-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}\Z")


class YFinanceFetcher(Protocol):
    def fetch(self, ticker: str, config: YFinanceConfig) -> ProviderPayload: ...


class FredFetcher(Protocol):
    def fetch(
        self,
        series_id: str,
        config: FredConfig,
        *,
        start: str,
        end: str,
    ) -> ProviderPayload: ...


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """A complete raw snapshot and its noncanonical acquisition journal."""

    snapshot: SnapshotReference
    acquisition_group_id: str
    journal_path: Path
    source_count: int


class AcquisitionJournal:
    """Append-only local evidence for one bounded acquisition group."""

    def __init__(
        self,
        artifact_root: Path,
        group_id: str,
        *,
        contract: Mapping[str, Any],
        now: datetime,
    ) -> None:
        if not _GROUP_PATTERN.fullmatch(group_id):
            raise DataValidationError(
                "acquisition group identifier is invalid",
                details={"acquisition_group_id": group_id},
            )
        self.group_id = group_id
        self.path = artifact_root / "data" / "acquisitions" / group_id
        self.path.mkdir(parents=True, exist_ok=False, mode=0o700)
        contract_value = _json_value(contract)
        contract_bytes = _canonical_json_bytes(contract_value)
        header = {
            "schema_version": 1,
            "acquisition_group_id": group_id,
            "created_utc": now.astimezone(UTC).isoformat(),
            "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "contract": contract_value,
            "status": "in_progress",
        }
        _write_new(self.path / "acquisition.json", _canonical_json_bytes(header))
        (self.path / "events").mkdir(mode=0o700)
        (self.path / "raw").mkdir(mode=0o700)
        self._event_number = 0

    def record_payload(self, payload: ProviderPayload) -> None:
        """Persist a successful payload before the next provider is called."""

        provider_dir = self.path / "raw" / payload.provider
        provider_dir.mkdir(exist_ok=True, mode=0o700)
        raw_path = provider_dir / f"{payload.identifier}.csv"
        _write_new(raw_path, payload.raw_bytes)
        record = payload.manifest_record(
            relative_path=raw_path.relative_to(self.path).as_posix()
        )
        self._event(
            {
                "event": "provider_succeeded",
                "provider": payload.provider,
                "identifier": payload.identifier,
                "payload": record,
            }
        )

    def record_failure(self, error: PRPGError) -> None:
        """Persist redacted structured failure evidence without response bodies."""

        self._event(
            {
                "event": "provider_failed",
                "error_type": type(error).__name__,
                "exit_code": int(error.exit_code),
                "message": error.message,
                "details": error.details,
            }
        )
        self._marker(
            "FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "exit_code": int(error.exit_code),
            },
        )

    def record_promotion(self, reference: SnapshotReference) -> None:
        """Mark the journal as promoted without changing its initial header."""

        self._event(
            {
                "event": "snapshot_promoted",
                "snapshot_fingerprint": reference.fingerprint,
                "snapshot_path": str(reference.path),
                "reused": reference.reused,
            }
        )
        self._marker(
            "PROMOTED.json",
            {
                "status": "promoted",
                "snapshot_fingerprint": reference.fingerprint,
            },
        )

    def _event(self, value: Mapping[str, Any]) -> None:
        self._event_number += 1
        payload = {
            "schema_version": 1,
            "sequence": self._event_number,
            **dict(value),
        }
        path = self.path / "events" / f"{self._event_number:03d}.json"
        _write_new(path, _canonical_json_bytes(_json_value(payload)))

    def _marker(self, name: str, value: Mapping[str, Any]) -> None:
        _write_new(self.path / name, _canonical_json_bytes(_json_value(value)))


def acquire_raw_snapshot(
    config: PRPGConfig,
    *,
    yfinance_provider: YFinanceFetcher | None = None,
    fred_provider: FredFetcher | None = None,
    snapshot_store: SnapshotStore | None = None,
    now: datetime | None = None,
) -> AcquisitionResult:
    """Acquire all seven approved sources and promote one immutable snapshot."""

    timestamp = now or datetime.now(UTC)
    group_id = _new_group_id(timestamp)
    contract = acquisition_contract(config)
    journal = AcquisitionJournal(
        config.data.artifact_root,
        group_id,
        contract=contract,
        now=timestamp,
    )
    yahoo = yfinance_provider or YFinanceProvider()
    fred = fred_provider or FredCsvProvider()
    store = snapshot_store or SnapshotStore(config.data.artifact_root)
    payloads: list[ProviderPayload] = []
    try:
        for ticker in (
            config.data.assets.equity,
            config.data.assets.muni_bond,
            config.data.assets.taxable_bond,
        ):
            payload = yahoo.fetch(ticker, config.data.yfinance)
            _require_payload_identity(payload, "yfinance", ticker)
            journal.record_payload(payload)
            payloads.append(payload)
        for series_id in (
            config.data.macro.industrial_production,
            config.data.macro.inflation,
            config.data.macro.yield_curve_spread,
            config.data.macro.credit_spread,
        ):
            payload = fred.fetch(
                series_id,
                config.data.fred,
                start=config.data.yfinance.start.isoformat(),
                end=config.data.cutoff.isoformat(),
            )
            _require_payload_identity(payload, "fred", series_id)
            journal.record_payload(payload)
            payloads.append(payload)
    except PRPGError as error:
        journal.record_failure(error)
        raise DataAcquisitionError(
            "acquisition group did not produce a promotable snapshot",
            details={
                "acquisition_group_id": group_id,
                "journal_path": str(journal.path),
                "completed_sources": len(payloads),
                "cause": error.as_event(),
            },
        ) from error
    except Exception as error:
        wrapped = DataAcquisitionError(
            "unexpected provider orchestration failure",
            details={"error_type": type(error).__name__},
        )
        journal.record_failure(wrapped)
        raise DataAcquisitionError(
            "acquisition group did not produce a promotable snapshot",
            details={
                "acquisition_group_id": group_id,
                "journal_path": str(journal.path),
                "completed_sources": len(payloads),
                "cause": wrapped.as_event(),
            },
        ) from error

    reference = store.create(
        payloads,
        acquisition_contract=contract,
        acquisition_group_id=group_id,
        audit_metadata={"resolved_config_fingerprint": config.fingerprint()},
    )
    journal.record_promotion(reference)
    return AcquisitionResult(
        snapshot=reference,
        acquisition_group_id=group_id,
        journal_path=journal.path,
        source_count=len(payloads),
    )


def acquisition_contract(config: PRPGConfig) -> dict[str, Any]:
    """Return the complete frozen acquisition identity for snapshot hashing."""

    data_contract = config.data.model_dump(mode="json")
    # Storage roots are operational bindings, not scientific source identity.
    data_contract.pop("artifact_root")
    encoded = _canonical_json_bytes(data_contract)
    return {
        "contract_schema_version": 1,
        "application_version": __version__,
        "data_contract_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "data": data_contract,
    }


def _new_group_id(now: datetime) -> str:
    timestamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"acq-{timestamp}-{uuid4().hex[:12]}"


def _require_payload_identity(
    payload: ProviderPayload, provider: str, identifier: str
) -> None:
    if payload.provider != provider or payload.identifier != identifier:
        raise DataValidationError(
            "provider adapter returned the wrong source identity",
            details={
                "expected_provider": provider,
                "expected_identifier": identifier,
                "actual_provider": payload.provider,
                "actual_identifier": payload.identifier,
            },
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
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return asdict(value) if hasattr(value, "__dataclass_fields__") else str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
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
