"""Exact CSV schema-v1 serialization and serial atomic shard writing.

The writer accepts only process-sealed Phase-4 path or aggregate objects.  It
streams canonical rows to a same-directory temporary file, flushes and fsyncs
the bytes, then publishes with the repository's no-replace hard-link pattern.
There is intentionally no worker scheduling, manifest, resume, or release
state in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, TypeAlias, TypeGuard

import numpy as np
import numpy.typing as npt

from prpg.errors import GenerationError, IntegrityError
from prpg.simulation.aggregation import (
    HFWeeklyReturns,
    LFAggregatedReturns,
    is_sealed_hf_weekly_returns,
    is_sealed_lf_aggregated_returns,
)
from prpg.simulation.serial import (
    SerialBasePath,
    is_sealed_serial_base_path,
)

CSV_SCHEMA_VERSION: Final = 1
CSV_SERIALIZER_VERSION: Final = "prpg-csv-v1-cpython-17g"
CSV_HEADER: Final = (
    "path_id,period_index,simulation_year,period_in_year,"
    "equity_log_return,muni_bond_log_return,taxable_bond_log_return"
)
CSV_HEADER_BYTES: Final = (CSV_HEADER + "\n").encode("utf-8")
MAX_PATH_ENTITY: Final = 999_999
MAX_SHARD_INDEX: Final = 99_999
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPT_SEAL = object()

CSVFrequency: TypeAlias = Literal["monthly", "quarterly", "annual", "daily", "weekly"]
CSVFamily: TypeAlias = Literal["LF", "HF"]
ShardValue: TypeAlias = SerialBasePath | LFAggregatedReturns | HFWeeklyReturns


@dataclass(frozen=True, slots=True, init=False)
class CSVShardReceipt:
    """Content and range identity of one atomically published CSV shard."""

    schema_version: int
    serializer_version: str
    relative_path: str
    frequency: CSVFrequency
    family: CSVFamily
    shard_index: int
    first_path_entity: int
    last_path_entity: int
    first_path_id: str
    last_path_id: str
    path_count: int
    years: int
    rows: int
    bytes: int
    sha256: str
    source_fingerprints: tuple[str, ...]
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _FrequencySpec:
    frequency: CSVFrequency
    family: CSVFamily
    family_directory: Literal["low", "high"]
    periods_per_year: int


_FREQUENCIES: Final[Mapping[CSVFrequency, _FrequencySpec]] = {
    "monthly": _FrequencySpec("monthly", "LF", "low", 12),
    "quarterly": _FrequencySpec("quarterly", "LF", "low", 4),
    "annual": _FrequencySpec("annual", "LF", "low", 1),
    "daily": _FrequencySpec("daily", "HF", "high", 252),
    "weekly": _FrequencySpec("weekly", "HF", "high", 52),
}


class _DigestingWriter:
    """Minimal binary sink that counts and hashes exactly written bytes."""

    def __init__(self, handle: object) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, content: bytes) -> None:
        write = getattr(self._handle, "write", None)
        if not callable(write):  # pragma: no cover - os.fdopen invariant
            raise IntegrityError("CSV temporary file is not writable")
        written = write(content)
        if written != len(content):
            raise IntegrityError("CSV temporary file write was incomplete")
        self._digest.update(content)
        self.bytes_written += len(content)

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def format_csv_float64(value: float | np.floating[Any]) -> str:
    """Return the frozen finite ``.17g`` grammar with signed zero removed."""

    if isinstance(value, bool) or not isinstance(value, float | np.floating):
        raise GenerationError("CSV return value must be float64-compatible")
    result = float(value)
    if not math.isfinite(result):
        raise GenerationError("CSV return value must be finite")
    if result == 0.0:
        return "0"
    text = format(result, ".17g")
    if "E" in text:  # pragma: no cover - CPython currently emits lowercase
        text = text.lower()
    return text


def write_serial_csv_shard(
    output_root: str | Path,
    *,
    frequency: CSVFrequency,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
    values: Iterable[ShardValue],
) -> CSVShardReceipt:
    """Stream one exact contiguous path range to a no-overwrite CSV shard.

    ``values`` must yield each expected one-based path entity exactly once in
    ascending order.  A shard cannot mix horizons, families, or frequencies.
    The caller-supplied root is operational only and never enters CSV bytes.
    """

    spec = _frequency_spec(frequency)
    index = _shard_index(shard_index)
    first = _path_entity(first_path_entity, "first path entity")
    last = _path_entity(last_path_entity, "last path entity")
    if first > last:
        raise GenerationError("CSV shard first path entity exceeds last")
    if not isinstance(values, Iterable):
        raise GenerationError("CSV shard values must be iterable")
    root = _prepare_output_root(output_root)
    first_id = _path_id(spec.family, first)
    last_id = _path_id(spec.family, last)
    filename = f"part-{index:05d}-path-{first_id}-{last_id}.csv"
    relative = PurePosixPath(
        "consumer",
        "returns",
        spec.family_directory,
        spec.frequency,
        filename,
    )
    parent = _prepare_relative_directory(root, relative.parent.parts)
    destination = parent / filename
    _reject_existing_destination(destination)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    rows = 0
    paths = 0
    source_fingerprints: list[str] = []
    years: int | None = None
    digest = ""
    byte_count = 0
    expected_entity = first
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            writer = _DigestingWriter(handle)
            writer.write(CSV_HEADER_BYTES)
            for value in values:
                if expected_entity > last:
                    raise GenerationError("CSV shard contains more paths than declared")
                entity, item_years, returns, source_fingerprint = _frequency_values(
                    value, spec
                )
                if entity != expected_entity:
                    raise GenerationError(
                        "CSV shard paths are duplicated, missing, or out of order",
                        details={"expected": expected_entity, "actual": entity},
                    )
                if years is None:
                    years = item_years
                elif item_years != years:
                    raise GenerationError("CSV shard cannot mix path horizons")
                expected_rows = item_years * spec.periods_per_year
                if returns.shape != (expected_rows, 3):
                    raise IntegrityError("sealed CSV source has invalid row geometry")
                path_id = _path_id(spec.family, entity)
                for zero_index, vector in enumerate(returns):
                    period_index = zero_index + 1
                    simulation_year = zero_index // spec.periods_per_year + 1
                    period_in_year = zero_index % spec.periods_per_year + 1
                    writer.write(
                        encode_csv_row(
                            path_id,
                            period_index,
                            simulation_year,
                            period_in_year,
                            vector,
                        )
                    )
                    rows += 1
                paths += 1
                source_fingerprints.append(source_fingerprint)
                expected_entity += 1
            if expected_entity != last + 1:
                raise GenerationError(
                    "CSV shard path range is incomplete",
                    details={"next_expected": expected_entity, "last": last},
                )
            handle.flush()
            os.fsync(handle.fileno())
            digest = writer.sha256
            byte_count = writer.bytes_written
        os.chmod(temporary, 0o600)
        _atomic_publish_no_replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()

    if years is None:  # pragma: no cover - complete positive range prevents this
        raise AssertionError("completed CSV shard has no horizon")
    sources = tuple(source_fingerprints)
    identity = _receipt_identity(
        relative_path=relative.as_posix(),
        frequency=spec.frequency,
        family=spec.family,
        shard_index=index,
        first_path_entity=first,
        last_path_entity=last,
        first_path_id=first_id,
        last_path_id=last_id,
        path_count=paths,
        years=years,
        rows=rows,
        bytes_count=byte_count,
        sha256=digest,
        source_fingerprints=sources,
    )
    receipt = object.__new__(CSVShardReceipt)
    for name, field_value in (
        *tuple(identity.items()),
        ("fingerprint", _sha256_json(identity)),
        ("_seal", _RECEIPT_SEAL),
    ):
        object.__setattr__(receipt, name, field_value)
    object.__setattr__(receipt, "_owner_id", id(receipt))
    _verify_receipt(receipt)
    return receipt


def is_sealed_csv_shard_receipt(value: object) -> TypeGuard[CSVShardReceipt]:
    """Return whether a receipt is the original untampered minted instance."""

    try:
        _verify_receipt(value)
    except (GenerationError, IntegrityError, TypeError, ValueError):
        return False
    return True


def _frequency_spec(value: object) -> _FrequencySpec:
    if not isinstance(value, str) or value not in _FREQUENCIES:
        raise GenerationError("CSV frequency is not registered in schema v1")
    return _FREQUENCIES[value]  # type: ignore[index]


def _frequency_values(
    value: object, spec: _FrequencySpec
) -> tuple[int, int, npt.NDArray[np.float64], str]:
    if spec.frequency in {"monthly", "daily"}:
        if not is_sealed_serial_base_path(value):
            raise GenerationError("base-frequency CSV requires a sealed base path")
        if value.family != spec.family or value.base_frequency != spec.frequency:
            raise GenerationError("base path family/frequency does not match CSV shard")
        return value.path_entity, value.years, value.log_returns, value.fingerprint
    if spec.frequency in {"quarterly", "annual"}:
        if not is_sealed_lf_aggregated_returns(value):
            raise GenerationError("LF derived CSV requires a sealed LF aggregate")
        returns = (
            value.quarterly_log_returns
            if spec.frequency == "quarterly"
            else value.annual_log_returns
        )
        return value.path_entity, value.years, returns, value.fingerprint
    if not is_sealed_hf_weekly_returns(value):
        raise GenerationError("weekly CSV requires a sealed HF weekly aggregate")
    return (
        value.path_entity,
        value.years,
        value.weekly_log_returns,
        value.fingerprint,
    )


def encode_csv_row(
    path_id: str,
    period_index: int,
    simulation_year: int,
    period_in_year: int,
    vector: npt.NDArray[np.float64],
) -> bytes:
    """Encode one row using the exact schema-v1 integer and float grammar."""

    if re.fullmatch(r"(?:LF|HF)[0-9]{6}", path_id) is None or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (period_index, simulation_year, period_in_year)
    ):
        raise GenerationError("CSV row identity fields are invalid")
    if vector.shape != (3,) or vector.dtype != np.dtype(np.float64):
        raise IntegrityError("sealed CSV return vector has invalid geometry or dtype")
    fields = (
        path_id,
        str(period_index),
        str(simulation_year),
        str(period_in_year),
        *(format_csv_float64(value) for value in vector),
    )
    return (",".join(fields) + "\n").encode("utf-8")


def _path_entity(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_PATH_ENTITY
    ):
        raise GenerationError(f"{label} must be in [1, {MAX_PATH_ENTITY}]")
    return value


def _shard_index(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_SHARD_INDEX
    ):
        raise GenerationError(f"CSV shard index must be in [0, {MAX_SHARD_INDEX}]")
    return value


def _path_id(family: CSVFamily, entity: int) -> str:
    return f"{family}{entity:06d}"


def _prepare_output_root(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("CSV output root must be a path")
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise GenerationError("CSV output root must be an absolute traversal-free path")
    _reject_symlink_components(root)
    try:
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as error:
        raise IntegrityError("CSV output root cannot be created") from error
    _reject_symlink_components(root)
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError("CSV output root is not a safe directory")
    return root


def _prepare_relative_directory(root: Path, parts: tuple[str, ...]) -> Path:
    current = root
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise IntegrityError("CSV relative output path contains traversal")
        current = current / part
        if current.exists() or current.is_symlink():
            if not current.is_dir() or current.is_symlink():
                raise IntegrityError("CSV output directory path is unsafe")
        else:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                # Another production worker may have created this shared
                # frequency directory after our existence check.  Treat that
                # race as success and let the safety check below reject a
                # symlink or non-directory winner.
                pass
            except OSError as error:
                raise IntegrityError(
                    "CSV output directory cannot be created"
                ) from error
        if current.is_symlink() or not current.is_dir():
            raise IntegrityError("CSV output directory changed during creation")
    return current


def _reject_symlink_components(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise IntegrityError("CSV output path contains a symbolic link")


def _reject_existing_destination(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError("CSV shard destination already exists")


def _atomic_publish_no_replace(temporary: Path, destination: Path) -> None:
    """Publish atomically with the no-replace equivalent of an atomic rename.

    A same-filesystem hard link makes the fully fsynced temporary inode visible
    at ``destination`` in one namespace operation and, unlike ``os.rename``,
    fails atomically if that name already exists.  The private temporary name
    is removed by the caller afterward.
    """

    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise IntegrityError("CSV shard destination appeared concurrently") from error
    except OSError as error:
        raise IntegrityError("CSV shard cannot be atomically published") from error
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        raise IntegrityError(
            "CSV output directory cannot be opened for fsync"
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise IntegrityError("CSV output directory fsync failed") from error
    finally:
        os.close(descriptor)


def _receipt_identity(
    *,
    relative_path: str,
    frequency: CSVFrequency,
    family: CSVFamily,
    shard_index: int,
    first_path_entity: int,
    last_path_entity: int,
    first_path_id: str,
    last_path_id: str,
    path_count: int,
    years: int,
    rows: int,
    bytes_count: int,
    sha256: str,
    source_fingerprints: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": CSV_SCHEMA_VERSION,
        "serializer_version": CSV_SERIALIZER_VERSION,
        "relative_path": relative_path,
        "frequency": frequency,
        "family": family,
        "shard_index": shard_index,
        "first_path_entity": first_path_entity,
        "last_path_entity": last_path_entity,
        "first_path_id": first_path_id,
        "last_path_id": last_path_id,
        "path_count": path_count,
        "years": years,
        "rows": rows,
        "bytes": bytes_count,
        "sha256": sha256,
        "source_fingerprints": source_fingerprints,
    }


def _verify_receipt(value: object) -> None:
    if type(value) is not CSVShardReceipt:
        raise GenerationError("CSV receipt was not minted by the shard writer")
    if value._seal is not _RECEIPT_SEAL or value._owner_id != id(value):
        raise IntegrityError("CSV receipt process seal is invalid")
    if (
        value.schema_version != CSV_SCHEMA_VERSION
        or value.serializer_version != CSV_SERIALIZER_VERSION
        or value.frequency not in _FREQUENCIES
        or value.family != _FREQUENCIES[value.frequency].family
        or value.path_count != value.last_path_entity - value.first_path_entity + 1
        or value.path_count != len(value.source_fingerprints)
        or value.first_path_id != _path_id(value.family, value.first_path_entity)
        or value.last_path_id != _path_id(value.family, value.last_path_entity)
        or value.years <= 0
        or value.rows
        != value.path_count
        * value.years
        * _FREQUENCIES[value.frequency].periods_per_year
        or value.bytes <= len(CSV_HEADER_BYTES)
        or _FINGERPRINT.fullmatch(value.sha256) is None
        or any(
            _FINGERPRINT.fullmatch(item) is None for item in value.source_fingerprints
        )
    ):
        raise IntegrityError("CSV receipt identity fields are invalid")
    expected_relative = PurePosixPath(
        "consumer",
        "returns",
        _FREQUENCIES[value.frequency].family_directory,
        value.frequency,
        (
            f"part-{value.shard_index:05d}-path-"
            f"{value.first_path_id}-{value.last_path_id}.csv"
        ),
    ).as_posix()
    if value.relative_path != expected_relative:
        raise IntegrityError("CSV receipt relative path is invalid")
    identity = _receipt_identity(
        relative_path=value.relative_path,
        frequency=value.frequency,
        family=value.family,
        shard_index=value.shard_index,
        first_path_entity=value.first_path_entity,
        last_path_entity=value.last_path_entity,
        first_path_id=value.first_path_id,
        last_path_id=value.last_path_id,
        path_count=value.path_count,
        years=value.years,
        rows=value.rows,
        bytes_count=value.bytes,
        sha256=value.sha256,
        source_fingerprints=value.source_fingerprints,
    )
    if value.fingerprint != _sha256_json(identity):
        raise IntegrityError("CSV receipt fingerprint is invalid")


def _sha256_json(value: Mapping[str, object]) -> str:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(content).hexdigest()
