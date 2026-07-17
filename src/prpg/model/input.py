"""Verified, immutable processed-data boundary for model calibration.

The processed snapshot store proves that bytes match their content address.  A
model needs a stronger contract: the arrays must have the exact semantic
columns, dtypes, chronology, continuation convention, and monthly/daily
alignment expected by calibration.  This module performs that fail-closed
translation without fitting a model or choosing any scientific default.

``load_calibration_input`` verifies both the processed snapshot and its parent
raw snapshot, cross-checks the seven provider identities, and returns detached
arrays backed by immutable ``bytes``.  The resulting fingerprint binds the raw
and processed content addresses, both configuration fingerprints, every source
file hash, semantic identities, and the exact design/holdout geometry.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from prpg.data.processed import (
    PROCESSED_SCHEMA_VERSION,
    TRANSFORM_VERSION,
    ProcessedArrays,
    ProcessedSnapshotReference,
    ProcessedSnapshotStore,
    load_processed_arrays,
)
from prpg.data.snapshot import SnapshotReference, SnapshotStore
from prpg.data.transform import ASSET_ROLES, MACRO_FEATURES, expected_xnys_sessions
from prpg.errors import ModelError

CALIBRATION_INPUT_SCHEMA_VERSION = 1
MONTHLY_AGGREGATION_TOLERANCE: Final = 1e-12
DATA_DICTIONARY_PATH: Final = "metadata/data-dictionary.json"

AssetColumns: TypeAlias = tuple[str, str, str]
MacroColumns: TypeAlias = tuple[str, str, str, str]
SplitRole: TypeAlias = Literal["full", "design", "holdout"]
SourceLayer: TypeAlias = Literal["raw", "processed"]

ASSET_COLUMNS: Final[AssetColumns] = cast(AssetColumns, tuple(ASSET_ROLES))
MACRO_COLUMNS: Final[MacroColumns] = cast(MacroColumns, tuple(MACRO_FEATURES))

_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_RETURN_SEMANTICS = "nominal total log return in decimal units"
_ASSET_DICTIONARY: Final = {
    "equity": "ACWI adjusted-close total-return proxy",
    "muni_bond": "MUB adjusted-close total-return proxy",
    "taxable_bond": "AGG broad taxable-bond adjusted-close proxy",
}
_MACRO_DICTIONARY: Final = {
    "industrial_production_growth": "100 * log(INDPRO_t / INDPRO_t-1)",
    "inflation_yoy": "100 * log(CPIAUCSL_t / CPIAUCSL_t-12)",
    "yield_curve_spread": "within-month finite-observation mean of T10Y3M",
    "credit_spread": "within-month finite-observation mean of BAA10Y",
}
_INDEX_DICTIONARY: Final = {
    "monthly": "calendar month YYYY-MM; return ends in that month",
    "daily": "normalized America/New_York XNYS session date",
}
_CONTINUATION_SEMANTICS = (
    "False at the first row and whenever the source index is not the "
    "immediately following eligible calendar/session observation; blocks "
    "must restart when False"
)
_REQUIRED_ARRAY_PATHS: Final = frozenset(
    {
        "arrays/monthly_returns.npy",
        "arrays/daily_returns.npy",
        "arrays/macro_features.npy",
        "arrays/monthly_period_ordinals.npy",
        "arrays/daily_session_ns.npy",
        "arrays/monthly_continuation.npy",
        "arrays/daily_continuation.npy",
    }
)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """One exact semantic role and the provider identifier that supplies it."""

    role: str
    provider: Literal["yfinance", "fred"]
    identifier: str

    def as_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "provider": self.provider,
            "identifier": self.identifier,
        }


@dataclass(frozen=True, slots=True)
class SourceFileProvenance:
    """Identity-bearing hash record for one verified source file."""

    layer: SourceLayer
    relative_path: str
    bytes: int
    sha256: str
    provider: str | None = None
    identifier: str | None = None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "layer": self.layer,
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "provider": self.provider,
            "identifier": self.identifier,
        }


@dataclass(frozen=True, slots=True)
class SplitGeometry:
    """Exact row counts and time boundaries of the sealed split."""

    holdout_months: int
    full_monthly_rows: int
    full_daily_rows: int
    design_monthly_rows: int
    design_daily_rows: int
    holdout_monthly_rows: int
    holdout_daily_rows: int
    full_first_month_ordinal: int
    full_last_month_ordinal: int
    design_last_month_ordinal: int
    holdout_first_month_ordinal: int
    full_first_daily_session_ns: int
    full_last_daily_session_ns: int
    design_last_daily_session_ns: int
    holdout_first_daily_session_ns: int

    def as_dict(self) -> dict[str, int]:
        return {
            "holdout_months": self.holdout_months,
            "full_monthly_rows": self.full_monthly_rows,
            "full_daily_rows": self.full_daily_rows,
            "design_monthly_rows": self.design_monthly_rows,
            "design_daily_rows": self.design_daily_rows,
            "holdout_monthly_rows": self.holdout_monthly_rows,
            "holdout_daily_rows": self.holdout_daily_rows,
            "full_first_month_ordinal": self.full_first_month_ordinal,
            "full_last_month_ordinal": self.full_last_month_ordinal,
            "design_last_month_ordinal": self.design_last_month_ordinal,
            "holdout_first_month_ordinal": self.holdout_first_month_ordinal,
            "full_first_daily_session_ns": self.full_first_daily_session_ns,
            "full_last_daily_session_ns": self.full_last_daily_session_ns,
            "design_last_daily_session_ns": self.design_last_daily_session_ns,
            "holdout_first_daily_session_ns": self.holdout_first_daily_session_ns,
        }


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Canonical upstream identities retained by every calibration input."""

    processed_data_fingerprint: str
    raw_snapshot_fingerprint: str
    processed_config_fingerprint: str
    calibration_config_fingerprint: str
    raw_acquisition_contract_sha256: str
    data_dictionary_sha256: str
    asset_columns: AssetColumns
    macro_columns: MacroColumns
    source_identities: tuple[SourceIdentity, ...]
    source_files: tuple[SourceFileProvenance, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "raw_snapshot_fingerprint": self.raw_snapshot_fingerprint,
            "processed_config_fingerprint": self.processed_config_fingerprint,
            "calibration_config_fingerprint": self.calibration_config_fingerprint,
            "raw_acquisition_contract_sha256": (self.raw_acquisition_contract_sha256),
            "data_dictionary_sha256": self.data_dictionary_sha256,
            "asset_columns": list(self.asset_columns),
            "macro_columns": list(self.macro_columns),
            "source_identities": [item.as_dict() for item in self.source_identities],
            "source_files": [item.as_dict() for item in self.source_files],
        }


@dataclass(frozen=True, slots=True)
class CalibrationSlice:
    """One detached, read-only full/design/holdout data view."""

    role: SplitRole
    monthly_returns: NDArray[np.float64]
    daily_returns: NDArray[np.float64]
    macro_features: NDArray[np.float64]
    monthly_period_ordinals: NDArray[np.int64]
    daily_session_ns: NDArray[np.int64]
    monthly_continuation: NDArray[np.bool_]
    daily_continuation: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CalibrationInput:
    """Verified model input whose fingerprint covers provenance and geometry."""

    fingerprint: str
    provenance: CalibrationProvenance
    geometry: SplitGeometry
    full: CalibrationSlice
    design: CalibrationSlice
    holdout: CalibrationSlice

    def identity_dict(self) -> dict[str, object]:
        """Return the exact JSON-ready identity hashed by ``fingerprint``."""

        return _input_identity(self.provenance, self.geometry)


def forward_links_from_row_start_mask(
    continuation: NDArray[np.bool_] | np.ndarray[Any, Any],
) -> NDArray[np.bool_]:
    """Convert an ``n``-row false-at-start mask to ``n-1`` forward links.

    Row ``i`` in a continuation mask says whether row ``i`` immediately
    follows row ``i-1``.  A forward-link vector instead assigns that same fact
    to edge ``i-1 -> i``.  Centralizing this one-position conversion prevents
    downstream block/HMM code from silently mixing the two conventions.
    """

    values = np.asarray(continuation)
    if values.ndim != 1 or values.size == 0 or values.dtype != np.dtype(np.bool_):
        raise ModelError("row-start continuation mask must be a non-empty bool array")
    if bool(values[0]):
        raise ModelError("row-start continuation mask must be false at row zero")
    return _immutable(values[1:].astype(np.bool_, copy=True))


def load_calibration_input(
    processed_store: ProcessedSnapshotStore,
    processed_data_fingerprint: str,
    *,
    calibration_config_fingerprint: str,
    holdout_months: int,
    raw_store: SnapshotStore | None = None,
) -> CalibrationInput:
    """Verify and detach one processed snapshot for later calibration.

    This function is network-free and performs no fit, bootstrap, random draw,
    or scientific selection.  ``holdout_months`` is required explicitly and is
    identity-bearing; the caller is responsible for supplying it from the
    resolved calibration configuration represented by
    ``calibration_config_fingerprint``.
    """

    _require_fingerprint(processed_data_fingerprint, label="processed data fingerprint")
    _require_fingerprint(
        calibration_config_fingerprint, label="calibration config fingerprint"
    )
    if (
        isinstance(holdout_months, bool)
        or not isinstance(holdout_months, int)
        or holdout_months <= 0
    ):
        raise ModelError("holdout_months must be a positive integer")

    processed_reference = processed_store.verify(processed_data_fingerprint)
    processed_identity = _processed_identity(processed_reference)
    raw_fingerprint = _required_string(
        processed_identity.get("raw_snapshot_fingerprint"),
        "processed raw snapshot fingerprint",
    )
    _require_fingerprint(raw_fingerprint, label="raw snapshot fingerprint")
    parent_store = raw_store or SnapshotStore(processed_store.artifact_root)
    raw_reference = parent_store.verify(raw_fingerprint)

    transform_contract = _transform_contract(processed_identity)
    dictionary, dictionary_sha = _load_data_dictionary(
        processed_reference, transform_contract
    )
    del dictionary  # Validation is the purpose; its exact hash is retained below.
    source_identities = _source_identities(transform_contract)
    raw_files = _raw_source_files(raw_reference, source_identities)
    processed_files = _processed_source_files(processed_identity)
    processed_config_fingerprint = _processed_config_fingerprint(processed_reference)
    _validate_processed_qc(processed_reference, raw_fingerprint)

    arrays = _load_arrays(processed_reference)
    normalized, daily_month_ordinals = _validate_arrays(arrays)
    full, design, holdout, geometry = _split_arrays(
        normalized,
        daily_month_ordinals=daily_month_ordinals,
        holdout_months=holdout_months,
    )

    raw_identity = raw_reference.manifest.get("identity")
    if not isinstance(raw_identity, dict):  # SnapshotStore.verify also checks this.
        raise ModelError("verified raw snapshot identity is unavailable")
    acquisition_contract = raw_identity.get("acquisition_contract")
    if not isinstance(acquisition_contract, dict):
        raise ModelError("raw acquisition contract is missing or invalid")
    provenance = CalibrationProvenance(
        processed_data_fingerprint=processed_data_fingerprint,
        raw_snapshot_fingerprint=raw_fingerprint,
        processed_config_fingerprint=processed_config_fingerprint,
        calibration_config_fingerprint=calibration_config_fingerprint,
        raw_acquisition_contract_sha256=_sha256(
            _canonical_json_bytes(acquisition_contract)
        ),
        data_dictionary_sha256=dictionary_sha,
        asset_columns=ASSET_COLUMNS,
        macro_columns=MACRO_COLUMNS,
        source_identities=source_identities,
        source_files=(*raw_files, *processed_files),
    )
    identity = _input_identity(provenance, geometry)
    return CalibrationInput(
        fingerprint=_sha256(_canonical_json_bytes(identity)),
        provenance=provenance,
        geometry=geometry,
        full=full,
        design=design,
        holdout=holdout,
    )


def _processed_identity(reference: ProcessedSnapshotReference) -> dict[str, Any]:
    manifest = reference.manifest
    if manifest.get("schema_version") != PROCESSED_SCHEMA_VERSION:
        raise ModelError("processed manifest schema is unsupported for calibration")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ModelError("processed calibration identity is missing")
    return identity


def _transform_contract(identity: dict[str, Any]) -> dict[str, Any]:
    contract = identity.get("transform_contract")
    if not isinstance(contract, dict):
        raise ModelError("processed transform contract is missing")
    required = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "calendar": "XNYS",
        "return_representation": "log_total_decimal",
        "common_history_only": True,
        "forbid_price_forward_fill": True,
    }
    for key, expected in required.items():
        if (
            type(contract.get(key)) is not type(expected)
            or contract.get(key) != expected
        ):
            raise ModelError(
                "processed transform contract is incompatible with calibration",
                details={
                    "field": key,
                    "expected": expected,
                    "actual": contract.get(key),
                },
            )
    return contract


def _load_data_dictionary(
    reference: ProcessedSnapshotReference,
    transform_contract: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = reference.path / DATA_DICTIONARY_PATH
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelError(
            "processed data dictionary cannot be read",
            details={"error_type": type(error).__name__},
        ) from error
    if not isinstance(value, dict):
        raise ModelError("processed data dictionary root must be an object")
    if content != _canonical_json_bytes(value):
        raise ModelError("processed data dictionary must be canonical JSON")
    expected = {
        "schema_version": 1,
        "return_semantics": _RETURN_SEMANTICS,
        "asset_columns": _ASSET_DICTIONARY,
        "macro_columns": _MACRO_DICTIONARY,
        "index_semantics": _INDEX_DICTIONARY,
        "continuation_masks": _CONTINUATION_SEMANTICS,
        "transform_contract": transform_contract,
    }
    if _canonical_json_bytes(value) != _canonical_json_bytes(expected):
        raise ModelError("processed data dictionary semantics are incompatible")
    return value, _sha256(content)


def _source_identities(
    transform_contract: dict[str, Any],
) -> tuple[SourceIdentity, ...]:
    assets = transform_contract.get("assets")
    macro = transform_contract.get("macro")
    asset_ids = _exact_identifier_mapping(assets, ASSET_COLUMNS, "asset")
    macro_roles = (
        "industrial_production",
        "inflation",
        "yield_curve_spread",
        "credit_spread",
    )
    macro_ids = _exact_identifier_mapping(macro, macro_roles, "macro")
    values = [
        *(
            SourceIdentity(f"asset:{role}", "yfinance", asset_ids[role])
            for role in ASSET_COLUMNS
        ),
        *(
            SourceIdentity(f"macro:{role}", "fred", macro_ids[role])
            for role in macro_roles
        ),
    ]
    identifiers = {(value.provider, value.identifier) for value in values}
    if len(identifiers) != len(values):
        raise ModelError("processed source identifiers are not unique")
    return tuple(values)


def _exact_identifier_mapping(
    value: object,
    roles: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(roles):
        raise ModelError(f"processed {label} role mapping is not exact")
    result: dict[str, str] = {}
    for role in roles:
        identifier = value[role]
        if not isinstance(identifier, str) or not identifier:
            raise ModelError(f"processed {label} identifier is invalid")
        result[role] = identifier
    if len(set(result.values())) != len(result):
        raise ModelError(f"processed {label} identifiers are not distinct")
    return result


def _raw_source_files(
    reference: SnapshotReference,
    expected_identities: tuple[SourceIdentity, ...],
) -> tuple[SourceFileProvenance, ...]:
    files = reference.manifest.get("files")
    identity = reference.manifest.get("identity")
    payloads = identity.get("payloads") if isinstance(identity, dict) else None
    if not isinstance(files, list) or not isinstance(payloads, list):
        raise ModelError("raw snapshot source records are missing")

    expected = {
        (value.provider, value.identifier): value for value in expected_identities
    }
    manifest_records = _indexed_raw_records(files, "raw manifest")
    identity_records = _indexed_raw_records(payloads, "raw identity")
    if set(manifest_records) != set(expected) or set(identity_records) != set(expected):
        raise ModelError(
            "raw snapshot sources do not match processed semantic identities",
            details={
                "expected": sorted(expected),
                "manifest": sorted(manifest_records),
                "identity": sorted(identity_records),
            },
        )

    result: list[SourceFileProvenance] = []
    for provider, identifier in sorted(expected):
        manifest_record = manifest_records[(provider, identifier)]
        identity_record = identity_records[(provider, identifier)]
        size = _required_positive_int(manifest_record.get("bytes"), "raw source bytes")
        digest = _required_hash(manifest_record.get("sha256"), "raw source sha256")
        if (
            identity_record.get("bytes") != size
            or identity_record.get("sha256") != digest
        ):
            raise ModelError("raw manifest source record disagrees with raw identity")
        relative_path = _required_string(
            manifest_record.get("relative_path"), "raw source relative path"
        )
        result.append(
            SourceFileProvenance(
                layer="raw",
                relative_path=relative_path,
                bytes=size,
                sha256=digest,
                provider=provider,
                identifier=identifier,
            )
        )
    return tuple(result)


def _indexed_raw_records(
    values: list[object], label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ModelError(f"{label} source record is invalid")
        provider = value.get("provider")
        identifier = value.get("identifier")
        if not isinstance(provider, str) or not isinstance(identifier, str):
            raise ModelError(f"{label} source identity is invalid")
        key = (provider, identifier)
        if key in result:
            raise ModelError(f"{label} contains a duplicate source identity")
        result[key] = value
    return result


def _processed_source_files(
    identity: dict[str, Any],
) -> tuple[SourceFileProvenance, ...]:
    files = identity.get("files")
    if not isinstance(files, list) or not files:
        raise ModelError("processed source file records are missing")
    result: list[SourceFileProvenance] = []
    seen: set[str] = set()
    for value in files:
        if not isinstance(value, dict):
            raise ModelError("processed source file record is invalid")
        relative_path = _required_string(
            value.get("relative_path"), "processed source relative path"
        )
        if relative_path in seen:
            raise ModelError("processed source file path is duplicated")
        seen.add(relative_path)
        result.append(
            SourceFileProvenance(
                layer="processed",
                relative_path=relative_path,
                bytes=_required_positive_int(
                    value.get("bytes"), "processed source bytes"
                ),
                sha256=_required_hash(value.get("sha256"), "processed source sha256"),
            )
        )
    missing = (_REQUIRED_ARRAY_PATHS | {DATA_DICTIONARY_PATH}).difference(seen)
    if missing:
        raise ModelError(
            "processed snapshot lacks required calibration files",
            details={"missing": sorted(missing)},
        )
    return tuple(sorted(result, key=lambda item: item.relative_path))


def _processed_config_fingerprint(reference: ProcessedSnapshotReference) -> str:
    audit = reference.manifest.get("audit_metadata")
    if not isinstance(audit, dict):
        raise ModelError("processed configuration provenance is missing")
    value = _required_string(
        audit.get("resolved_config_fingerprint"),
        "processed config fingerprint",
    )
    _require_fingerprint(value, label="processed config fingerprint")
    return value


def _validate_processed_qc(
    reference: ProcessedSnapshotReference, raw_fingerprint: str
) -> None:
    qc = reference.manifest.get("qc_report")
    if not isinstance(qc, dict):
        raise ModelError("processed QC report is missing")
    required = {
        "status": "passed",
        "raw_snapshot_fingerprint": raw_fingerprint,
        "raw_source_count": 7,
        "outlier_resolution_complete": True,
        "retrospective_not_realtime": True,
        "secret_scan_passed": True,
    }
    for key, expected in required.items():
        if type(qc.get(key)) is not type(expected) or qc.get(key) != expected:
            raise ModelError(
                "processed QC report is not calibration-eligible",
                details={"field": key, "expected": expected, "actual": qc.get(key)},
            )


def _load_arrays(reference: ProcessedSnapshotReference) -> ProcessedArrays:
    try:
        return load_processed_arrays(reference)
    except (OSError, ValueError, TypeError) as error:
        raise ModelError(
            "processed calibration arrays cannot be loaded safely",
            details={"error_type": type(error).__name__},
        ) from error


def _validate_arrays(
    arrays: ProcessedArrays,
) -> tuple[ProcessedArrays, NDArray[np.int64]]:
    monthly_returns = _float_matrix(arrays.monthly_returns, "monthly returns", 3)
    daily_returns = _float_matrix(arrays.daily_returns, "daily returns", 3)
    macro_features = _float_matrix(arrays.macro_features, "macro features", 4)
    monthly_ordinals = _integer_vector(
        arrays.monthly_period_ordinals, "monthly period ordinals"
    )
    daily_ns = _integer_vector(arrays.daily_session_ns, "daily session timestamps")
    monthly_continuation = _boolean_vector(
        arrays.monthly_continuation, "monthly continuation mask"
    )
    daily_continuation = _boolean_vector(
        arrays.daily_continuation, "daily continuation mask"
    )

    monthly_rows = monthly_returns.shape[0]
    daily_rows = daily_returns.shape[0]
    if macro_features.shape[0] != monthly_rows or monthly_ordinals.size != monthly_rows:
        raise ModelError("monthly returns, macro features, and ordinals are misaligned")
    if monthly_continuation.size != monthly_rows:
        raise ModelError("monthly continuation mask length is misaligned")
    if daily_ns.size != daily_rows or daily_continuation.size != daily_rows:
        raise ModelError("daily returns, timestamps, and continuation are misaligned")
    if not bool(np.all(np.diff(monthly_ordinals) > 0)):
        raise ModelError("monthly period ordinals must be strictly increasing")
    if not bool(np.all(np.diff(daily_ns) > 0)):
        raise ModelError("daily session timestamps must be strictly increasing")

    expected_monthly = np.zeros(monthly_rows, dtype=np.bool_)
    expected_monthly[1:] = np.diff(monthly_ordinals) == 1
    _require_exact_continuation(
        monthly_continuation, expected_monthly, "monthly continuation mask"
    )

    sessions = pd.DatetimeIndex(daily_ns)
    if sessions.hasnans or not bool((sessions == sessions.normalize()).all()):
        raise ModelError("daily timestamps must be normalized finite session dates")
    calendar = expected_xnys_sessions(sessions[0].date(), sessions[-1].date())
    positions = calendar.get_indexer(sessions)
    if bool((positions < 0).any()):
        raise ModelError("daily timestamps include a non-XNYS session")
    expected_daily = np.zeros(daily_rows, dtype=np.bool_)
    expected_daily[1:] = np.diff(positions) == 1
    _require_exact_continuation(
        daily_continuation, expected_daily, "daily continuation mask"
    )

    daily_month_ordinals = np.asarray(sessions.to_period("M").asi8, dtype=np.int64)
    if set(daily_month_ordinals.tolist()) != set(monthly_ordinals.tolist()):
        raise ModelError("daily and monthly histories cover different month sets")
    _validate_monthly_daily_aggregation(
        monthly_returns,
        monthly_ordinals,
        daily_returns,
        daily_month_ordinals,
    )
    normalized = ProcessedArrays(
        monthly_returns,
        daily_returns,
        macro_features,
        monthly_ordinals,
        daily_ns,
        monthly_continuation,
        daily_continuation,
    )
    return normalized, daily_month_ordinals


def _float_matrix(value: object, label: str, columns: int) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise ModelError(f"{label} must have exact float64 dtype")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != columns:
        raise ModelError(f"{label} must be a non-empty n-by-{columns} matrix")
    if not bool(np.isfinite(array).all()):
        raise ModelError(f"{label} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _integer_vector(value: object, label: str) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64):
        raise ModelError(f"{label} must have exact int64 dtype")
    if array.ndim != 1 or array.size == 0:
        raise ModelError(f"{label} must be a non-empty vector")
    return np.ascontiguousarray(array, dtype=np.int64)


def _boolean_vector(value: object, label: str) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_):
        raise ModelError(f"{label} must have exact bool dtype")
    if array.ndim != 1 or array.size == 0:
        raise ModelError(f"{label} must be a non-empty vector")
    if bool(array[0]):
        raise ModelError(f"{label} must be false at row zero")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _require_exact_continuation(
    actual: NDArray[np.bool_], expected: NDArray[np.bool_], label: str
) -> None:
    if not np.array_equal(actual, expected):
        mismatches = np.flatnonzero(actual != expected)
        raise ModelError(
            f"{label} disagrees with source chronology",
            details={"first_mismatch": int(mismatches[0])},
        )


def _validate_monthly_daily_aggregation(
    monthly_returns: NDArray[np.float64],
    monthly_ordinals: NDArray[np.int64],
    daily_returns: NDArray[np.float64],
    daily_month_ordinals: NDArray[np.int64],
) -> None:
    for row, ordinal in enumerate(monthly_ordinals):
        observed = daily_returns[daily_month_ordinals == ordinal].sum(axis=0)
        error = float(np.max(np.abs(observed - monthly_returns[row])))
        if not np.isfinite(error) or error > MONTHLY_AGGREGATION_TOLERANCE:
            raise ModelError(
                "summed daily returns do not match monthly returns",
                details={
                    "month_ordinal": int(ordinal),
                    "maximum_absolute_error": error,
                    "tolerance": MONTHLY_AGGREGATION_TOLERANCE,
                },
            )


def _split_arrays(
    arrays: ProcessedArrays,
    *,
    daily_month_ordinals: NDArray[np.int64],
    holdout_months: int,
) -> tuple[CalibrationSlice, CalibrationSlice, CalibrationSlice, SplitGeometry]:
    monthly_rows = arrays.monthly_returns.shape[0]
    if holdout_months >= monthly_rows:
        raise ModelError("holdout_months leaves no design observations")
    design_monthly_rows = monthly_rows - holdout_months
    first_holdout_ordinal = int(arrays.monthly_period_ordinals[design_monthly_rows])
    design_daily_rows = int(
        np.count_nonzero(daily_month_ordinals < first_holdout_ordinal)
    )
    if design_daily_rows <= 0 or design_daily_rows >= arrays.daily_returns.shape[0]:
        raise ModelError("sealed monthly split leaves an empty daily partition")
    if not bool(
        np.all(daily_month_ordinals[:design_daily_rows] < first_holdout_ordinal)
        and np.all(daily_month_ordinals[design_daily_rows:] >= first_holdout_ordinal)
    ):
        raise ModelError("daily rows are not chronologically separable by month")

    full = _make_slice("full", arrays, slice(None), slice(None))
    design = _make_slice(
        "design",
        arrays,
        slice(0, design_monthly_rows),
        slice(0, design_daily_rows),
    )
    holdout = _make_slice(
        "holdout",
        arrays,
        slice(design_monthly_rows, None),
        slice(design_daily_rows, None),
    )
    geometry = SplitGeometry(
        holdout_months=holdout_months,
        full_monthly_rows=monthly_rows,
        full_daily_rows=arrays.daily_returns.shape[0],
        design_monthly_rows=design_monthly_rows,
        design_daily_rows=design_daily_rows,
        holdout_monthly_rows=holdout_months,
        holdout_daily_rows=arrays.daily_returns.shape[0] - design_daily_rows,
        full_first_month_ordinal=int(arrays.monthly_period_ordinals[0]),
        full_last_month_ordinal=int(arrays.monthly_period_ordinals[-1]),
        design_last_month_ordinal=int(
            arrays.monthly_period_ordinals[design_monthly_rows - 1]
        ),
        holdout_first_month_ordinal=first_holdout_ordinal,
        full_first_daily_session_ns=int(arrays.daily_session_ns[0]),
        full_last_daily_session_ns=int(arrays.daily_session_ns[-1]),
        design_last_daily_session_ns=int(
            arrays.daily_session_ns[design_daily_rows - 1]
        ),
        holdout_first_daily_session_ns=int(arrays.daily_session_ns[design_daily_rows]),
    )
    return full, design, holdout, geometry


def _make_slice(
    role: SplitRole,
    arrays: ProcessedArrays,
    monthly_slice: slice,
    daily_slice: slice,
) -> CalibrationSlice:
    monthly_continuation = np.array(
        arrays.monthly_continuation[monthly_slice], dtype=np.bool_, copy=True
    )
    daily_continuation = np.array(
        arrays.daily_continuation[daily_slice], dtype=np.bool_, copy=True
    )
    monthly_continuation[0] = False
    daily_continuation[0] = False
    return CalibrationSlice(
        role=role,
        monthly_returns=_immutable(
            np.array(arrays.monthly_returns[monthly_slice], dtype=np.float64, copy=True)
        ),
        daily_returns=_immutable(
            np.array(arrays.daily_returns[daily_slice], dtype=np.float64, copy=True)
        ),
        macro_features=_immutable(
            np.array(arrays.macro_features[monthly_slice], dtype=np.float64, copy=True)
        ),
        monthly_period_ordinals=_immutable(
            np.array(
                arrays.monthly_period_ordinals[monthly_slice],
                dtype=np.int64,
                copy=True,
            )
        ),
        daily_session_ns=_immutable(
            np.array(arrays.daily_session_ns[daily_slice], dtype=np.int64, copy=True)
        ),
        monthly_continuation=_immutable(monthly_continuation),
        daily_continuation=_immutable(daily_continuation),
    )


def _immutable(array: NDArray[Any]) -> NDArray[Any]:
    contiguous = np.ascontiguousarray(array)
    # A bytes-backed ndarray cannot be made writable again by a caller.
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return result.reshape(contiguous.shape)


def _input_identity(
    provenance: CalibrationProvenance, geometry: SplitGeometry
) -> dict[str, object]:
    return {
        "schema_version": CALIBRATION_INPUT_SCHEMA_VERSION,
        "provenance": provenance.as_dict(),
        "split_geometry": geometry.as_dict(),
    }


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelError(f"{label} is missing or invalid")
    return value


def _required_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} is missing or invalid")
    return value


def _required_hash(value: object, label: str) -> str:
    result = _required_string(value, label)
    _require_fingerprint(result, label=label)
    return result


def _require_fingerprint(value: str, *, label: str) -> None:
    if not _FINGERPRINT_PATTERN.fullmatch(value):
        raise ModelError(f"{label} must be a lowercase SHA-256 fingerprint")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise ModelError(
            "calibration input identity is not canonical JSON",
            details={"error_type": type(error).__name__},
        ) from error
