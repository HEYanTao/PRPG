"""Pure immutable cross-frequency aggregation for serial base paths."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, TypeGuard

import numpy as np
import numpy.typing as npt

from prpg.data.calendar import WEEK_BOUNDARIES, WEEKS_PER_YEAR
from prpg.errors import GenerationError, IntegrityError, ModelError
from prpg.simulation.serial import (
    ASSET_COUNT,
    MONTHS_PER_YEAR,
    SESSIONS_PER_YEAR,
    FloatArray,
    SerialBasePath,
    _array_fingerprint,
    _has_bytes_owner,
    _immutable_array,
    _record_fingerprint,
    _verify_base_path,
)

AGGREGATION_SCHEMA_VERSION: Final = 1
AGGREGATION_VERSION: Final = "prpg-cross-frequency-aggregation-v1"
QUARTERS_PER_YEAR: Final = 4
_LF_AGGREGATE_SEAL = object()
_HF_WEEKLY_SEAL = object()
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True, init=False)
class LFAggregatedReturns:
    """Quarterly and annual exact sums derived from one LF monthly path."""

    schema_version: int
    aggregation_version: str
    base_path_fingerprint: str
    path_entity: int
    years: int
    quarterly_log_returns: FloatArray
    annual_log_returns: FloatArray
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True, init=False)
class HFWeeklyReturns:
    """SC252-52-v1 weekly exact sums derived from one HF daily path."""

    schema_version: int
    aggregation_version: str
    base_path_fingerprint: str
    path_entity: int
    years: int
    weekly_log_returns: FloatArray
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)


def aggregate_lf_log_returns(path: SerialBasePath) -> LFAggregatedReturns:
    """Derive consecutive 3-month quarters and 12-month years from LF."""

    _verify_base_path(path)
    if path.family != "LF" or path.base_frequency != "monthly":
        raise GenerationError("LF aggregation requires an LF monthly base path")
    monthly = path.log_returns.reshape(path.years, MONTHS_PER_YEAR, ASSET_COUNT)
    quarterly = monthly.reshape(path.years, QUARTERS_PER_YEAR, 3, ASSET_COUNT).sum(
        axis=2
    )
    annual = monthly.sum(axis=1)
    quarterly_frozen = _immutable_array(
        quarterly.reshape(path.years * QUARTERS_PER_YEAR, ASSET_COUNT),
        dtype=np.float64,
        label="LF quarterly returns",
    )
    annual_frozen = _immutable_array(
        annual,
        dtype=np.float64,
        label="LF annual returns",
    )
    metadata = _metadata(path)
    arrays = {
        "quarterly_log_returns": quarterly_frozen,
        "annual_log_returns": annual_frozen,
    }
    fingerprint = _aggregate_fingerprint(metadata, arrays)
    result = object.__new__(LFAggregatedReturns)
    for name, value in (
        ("schema_version", AGGREGATION_SCHEMA_VERSION),
        ("aggregation_version", AGGREGATION_VERSION),
        ("base_path_fingerprint", path.fingerprint),
        ("path_entity", path.path_entity),
        ("years", path.years),
        ("quarterly_log_returns", quarterly_frozen),
        ("annual_log_returns", annual_frozen),
        ("fingerprint", fingerprint),
        ("_seal", _LF_AGGREGATE_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _verify_lf_aggregate(result)
    return result


def aggregate_hf_weekly_log_returns(path: SerialBasePath) -> HFWeeklyReturns:
    """Derive exactly 52 boundary-defined model weeks per HF model year."""

    _verify_base_path(path)
    if path.family != "HF" or path.base_frequency != "daily":
        raise GenerationError("HF weekly aggregation requires an HF daily base path")
    weekly = np.empty((path.years * WEEKS_PER_YEAR, ASSET_COUNT), dtype=np.float64)
    output = 0
    for year in range(path.years):
        offset = year * SESSIONS_PER_YEAR
        for week in range(WEEKS_PER_YEAR):
            start = offset + WEEK_BOUNDARIES[week]
            stop = offset + WEEK_BOUNDARIES[week + 1]
            weekly[output] = path.log_returns[start:stop].sum(axis=0)
            output += 1
    weekly_frozen = _immutable_array(
        weekly,
        dtype=np.float64,
        label="HF weekly returns",
    )
    metadata = _metadata(path)
    arrays = {"weekly_log_returns": weekly_frozen}
    fingerprint = _aggregate_fingerprint(metadata, arrays)
    result = object.__new__(HFWeeklyReturns)
    for name, value in (
        ("schema_version", AGGREGATION_SCHEMA_VERSION),
        ("aggregation_version", AGGREGATION_VERSION),
        ("base_path_fingerprint", path.fingerprint),
        ("path_entity", path.path_entity),
        ("years", path.years),
        ("weekly_log_returns", weekly_frozen),
        ("fingerprint", fingerprint),
        ("_seal", _HF_WEEKLY_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _verify_hf_weekly(result)
    return result


def is_sealed_lf_aggregated_returns(value: object) -> TypeGuard[LFAggregatedReturns]:
    """Return whether ``value`` is the original untampered LF aggregate."""

    try:
        _verify_lf_aggregate(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_sealed_hf_weekly_returns(value: object) -> TypeGuard[HFWeeklyReturns]:
    """Return whether ``value`` is the original untampered HF aggregate."""

    try:
        _verify_hf_weekly(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def _metadata(path: SerialBasePath) -> dict[str, object]:
    return {
        "schema_version": AGGREGATION_SCHEMA_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "base_path_fingerprint": path.fingerprint,
        "path_entity": path.path_entity,
        "years": path.years,
    }


def _aggregate_fingerprint(
    metadata: dict[str, object], arrays: Mapping[str, npt.NDArray[np.generic]]
) -> str:
    return _record_fingerprint(
        metadata,
        {
            name: _array_fingerprint(value, name)
            for name, value in sorted(arrays.items())
        },
    )


def _verify_lf_aggregate(value: object) -> None:
    if type(value) is not LFAggregatedReturns:
        raise GenerationError("LF aggregate was not minted by the serial aggregator")
    _verify_aggregate_identity(value)
    if (
        value._seal is not _LF_AGGREGATE_SEAL
        or value._owner_id != id(value)
        or value.schema_version != AGGREGATION_SCHEMA_VERSION
        or value.aggregation_version != AGGREGATION_VERSION
        or value.quarterly_log_returns.shape
        != (value.years * QUARTERS_PER_YEAR, ASSET_COUNT)
        or value.annual_log_returns.shape != (value.years, ASSET_COUNT)
    ):
        raise IntegrityError("LF aggregate process seal or geometry is invalid")
    arrays = {
        "quarterly_log_returns": value.quarterly_log_returns,
        "annual_log_returns": value.annual_log_returns,
    }
    _verify_aggregate_arrays(arrays)
    metadata = {
        "schema_version": value.schema_version,
        "aggregation_version": value.aggregation_version,
        "base_path_fingerprint": value.base_path_fingerprint,
        "path_entity": value.path_entity,
        "years": value.years,
    }
    if value.fingerprint != _aggregate_fingerprint(metadata, arrays):
        raise IntegrityError("LF aggregate fingerprint changed")


def _verify_hf_weekly(value: object) -> None:
    if type(value) is not HFWeeklyReturns:
        raise GenerationError("HF aggregate was not minted by the serial aggregator")
    _verify_aggregate_identity(value)
    if (
        value._seal is not _HF_WEEKLY_SEAL
        or value._owner_id != id(value)
        or value.schema_version != AGGREGATION_SCHEMA_VERSION
        or value.aggregation_version != AGGREGATION_VERSION
        or value.weekly_log_returns.shape != (value.years * WEEKS_PER_YEAR, ASSET_COUNT)
    ):
        raise IntegrityError("HF weekly process seal or geometry is invalid")
    arrays = {"weekly_log_returns": value.weekly_log_returns}
    _verify_aggregate_arrays(arrays)
    metadata = {
        "schema_version": value.schema_version,
        "aggregation_version": value.aggregation_version,
        "base_path_fingerprint": value.base_path_fingerprint,
        "path_entity": value.path_entity,
        "years": value.years,
    }
    if value.fingerprint != _aggregate_fingerprint(metadata, arrays):
        raise IntegrityError("HF weekly fingerprint changed")


def _verify_aggregate_arrays(
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> None:
    for value in arrays.values():
        if (
            value.dtype != np.dtype(np.float64)
            or value.flags.writeable
            or not _has_bytes_owner(value)
            or not bool(np.isfinite(value).all())
        ):
            raise IntegrityError("aggregate array is not finite immutable float64")


def _verify_aggregate_identity(value: LFAggregatedReturns | HFWeeklyReturns) -> None:
    if (
        isinstance(value.path_entity, bool)
        or not isinstance(value.path_entity, int)
        or not 1 <= value.path_entity <= (1 << 32) - 1
        or isinstance(value.years, bool)
        or not isinstance(value.years, int)
        or value.years <= 0
        or not isinstance(value.base_path_fingerprint, str)
        or _FINGERPRINT.fullmatch(value.base_path_fingerprint) is None
        or not isinstance(value.fingerprint, str)
        or _FINGERPRINT.fullmatch(value.fingerprint) is None
    ):
        raise IntegrityError("aggregate identity fields are invalid")
