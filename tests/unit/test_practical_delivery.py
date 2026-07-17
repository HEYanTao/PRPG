from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from prpg.data.calendar import WEEK_BOUNDARIES
from prpg.errors import IntegrityError
from prpg.simulation.production import ProductionPlan, production_work_units
from prpg.storage.csv_v1 import CSV_HEADER_BYTES, format_csv_float64
from prpg.validation.practical_delivery import validate_practical_delivery

FloatArray = npt.NDArray[np.float64]

_PLAN = ProductionPlan(
    profile_id="practical-validator-test",
    years=1,
    lf_paths=2,
    hf_paths=1,
    lf_paths_per_shard=2,
    hf_paths_per_shard=1,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _returns(*, duplicate_lf: bool = False) -> dict[tuple[str, int, str], FloatArray]:
    result: dict[tuple[str, int, str], FloatArray] = {}
    for entity in (1, 2):
        source_entity = 1 if duplicate_lf else entity
        monthly = (
            np.arange(36, dtype=np.float64).reshape(12, 3) * 1e-5 + source_entity * 1e-3
        )
        result[("LF", entity, "monthly")] = monthly
        result[("LF", entity, "quarterly")] = monthly.reshape(4, 3, 3).sum(axis=1)
        result[("LF", entity, "annual")] = monthly.sum(axis=0).reshape(1, 3)

    daily = np.arange(756, dtype=np.float64).reshape(252, 3) * 1e-6 + 2e-3
    weekly = np.empty((52, 3), dtype=np.float64)
    for week in range(52):
        weekly[week] = daily[WEEK_BOUNDARIES[week] : WEEK_BOUNDARIES[week + 1]].sum(
            axis=0
        )
    result[("HF", 1, "daily")] = daily
    result[("HF", 1, "weekly")] = weekly
    return result


def _relative_path(
    family: str,
    shard_index: int,
    first: int,
    last: int,
    frequency: str,
) -> str:
    directory = "low" if family == "LF" else "high"
    filename = f"part-{shard_index:05d}-path-{family}{first:06d}-{family}{last:06d}.csv"
    return PurePosixPath(
        "consumer", "returns", directory, frequency, filename
    ).as_posix()


def _csv_bytes(
    family: str,
    first: int,
    last: int,
    frequency: str,
    values: dict[tuple[str, int, str], FloatArray],
) -> bytes:
    periods_per_year = {
        "monthly": 12,
        "quarterly": 4,
        "annual": 1,
        "daily": 252,
        "weekly": 52,
    }[frequency]
    result = bytearray(CSV_HEADER_BYTES)
    for entity in range(first, last + 1):
        array = values[(family, entity, frequency)]
        for zero_index, vector in enumerate(array):
            fields = (
                f"{family}{entity:06d}",
                str(zero_index + 1),
                str(zero_index // periods_per_year + 1),
                str(zero_index % periods_per_year + 1),
                *(format_csv_float64(value) for value in vector),
            )
            result.extend((",".join(fields) + "\n").encode("utf-8"))
    return bytes(result)


def _publish_manifest(root: Path, manifest: dict[str, Any]) -> None:
    identity = dict(manifest)
    identity.pop("manifest_fingerprint", None)
    manifest["manifest_fingerprint"] = _fingerprint(identity)
    (root / "manifest.json").write_bytes(_canonical_bytes(manifest))


def _publish_checksums(root: Path, manifest: dict[str, Any]) -> None:
    entries = sorted(
        (
            sibling["relative_path"],
            sibling["sha256"],
        )
        for unit in manifest["work_units"]
        for sibling in unit["siblings"]
    )
    (root / "checksums.sha256").write_text(
        "".join(f"{sha256}  {relative}\n" for relative, sha256 in entries),
        encoding="utf-8",
        newline="",
    )


def _make_bundle(root: Path, *, duplicate_lf: bool = False) -> None:
    values = _returns(duplicate_lf=duplicate_lf)
    work_units: list[dict[str, Any]] = []
    total_bytes = 0
    for scope in production_work_units(_PLAN):
        frequencies = (
            ("monthly", "quarterly", "annual")
            if scope.family == "LF"
            else ("daily", "weekly")
        )
        siblings: list[dict[str, Any]] = []
        for frequency in frequencies:
            relative = _relative_path(
                scope.family,
                scope.shard_index,
                scope.first_path_entity,
                scope.last_path_entity,
                frequency,
            )
            content = _csv_bytes(
                scope.family,
                scope.first_path_entity,
                scope.last_path_entity,
                frequency,
                values,
            )
            path = root.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            rows = len(content.splitlines()) - 1
            siblings.append(
                {
                    "frequency": frequency,
                    "relative_path": relative,
                    "rows": rows,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            total_bytes += len(content)
        work_units.append(
            {
                "work_unit_id": scope.work_unit_id,
                "receipt_fingerprint": "9" * 64,
                "siblings": siblings,
            }
        )
    manifest: dict[str, Any] = {
        "schema_id": "prpg-practical-manifest-v1",
        "status": "complete",
        "artifact_fingerprint": "1" * 64,
        "processed_data_fingerprint": "2" * 64,
        "config_fingerprint": "3" * 64,
        "plan": _PLAN.identity(),
        "binding": {"schema_id": "prpg-practical-binding-v1"},
        "workers": 1,
        "totals": {
            "paths": 3,
            "work_units": 2,
            "csv_shards": 5,
            "rows": _PLAN.total_rows,
            "bytes": total_bytes,
        },
        "work_units": work_units,
    }
    _publish_manifest(root, manifest)
    _publish_checksums(root, manifest)


def _load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_bytes())


def _reseal_bundle(root: Path) -> None:
    manifest = _load_manifest(root)
    total_bytes = 0
    for unit in manifest["work_units"]:
        for sibling in unit["siblings"]:
            content = root.joinpath(
                *PurePosixPath(sibling["relative_path"]).parts
            ).read_bytes()
            sibling["rows"] = len(content.splitlines()) - 1
            sibling["bytes"] = len(content)
            sibling["sha256"] = hashlib.sha256(content).hexdigest()
            total_bytes += len(content)
    manifest["totals"]["bytes"] = total_bytes
    _publish_manifest(root, manifest)
    _publish_checksums(root, manifest)


def _change_first_return(root: Path, relative: str, replacement: str) -> None:
    path = root.joinpath(*PurePosixPath(relative).parts)
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[4] = replacement
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def test_valid_practical_bundle_returns_compact_report(tmp_path: Path) -> None:
    _make_bundle(tmp_path)

    report = validate_practical_delivery(tmp_path)

    assert report.status == "passed"
    assert report.total_rows == 338
    assert report.work_units == 2
    assert report.csv_shards == 5
    assert report.checksums_verified == 5
    assert report.aggregation_paths_checked == 3
    assert report.aggregation_values_checked == 186
    assert report.duplicate_base_paths == 0
    assert sum(item.rows for item in report.datasets) == 338


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    manifest = _load_manifest(tmp_path)
    manifest["status"] = "running"
    _publish_manifest(tmp_path, manifest)

    with pytest.raises(IntegrityError, match="not complete"):
        validate_practical_delivery(tmp_path)


def test_unlisted_csv_is_rejected(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    extra = tmp_path / "consumer" / "returns" / "extra.csv"
    extra.write_bytes(CSV_HEADER_BYTES)

    with pytest.raises(IntegrityError, match="file set differs"):
        validate_practical_delivery(tmp_path)


def test_row_key_discontinuity_is_rejected_even_when_resealed(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    relative = _load_manifest(tmp_path)["work_units"][0]["siblings"][0]["relative_path"]
    path = tmp_path.joinpath(*PurePosixPath(relative).parts)
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[1] = "2"
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    _reseal_bundle(tmp_path)

    with pytest.raises(IntegrityError, match="sequence is not contiguous"):
        validate_practical_delivery(tmp_path)


def test_nonfinite_return_is_rejected_even_when_resealed(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    relative = _load_manifest(tmp_path)["work_units"][0]["siblings"][0]["relative_path"]
    _change_first_return(tmp_path, relative, "nan")
    _reseal_bundle(tmp_path)

    with pytest.raises(IntegrityError, match="return is not finite"):
        validate_practical_delivery(tmp_path)


def test_cross_frequency_mismatch_is_rejected_even_when_resealed(
    tmp_path: Path,
) -> None:
    _make_bundle(tmp_path)
    relative = _load_manifest(tmp_path)["work_units"][0]["siblings"][1]["relative_path"]
    _change_first_return(tmp_path, relative, "0.123")
    _reseal_bundle(tmp_path)

    with pytest.raises(IntegrityError, match="cross-frequency log sums"):
        validate_practical_delivery(tmp_path)


def test_duplicate_complete_base_path_is_rejected(tmp_path: Path) -> None:
    _make_bundle(tmp_path, duplicate_lf=True)

    with pytest.raises(IntegrityError, match="duplicate complete base path"):
        validate_practical_delivery(tmp_path)


def test_csv_content_checksum_is_verified(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    relative = _load_manifest(tmp_path)["work_units"][0]["siblings"][0]["relative_path"]
    path = tmp_path.joinpath(*PurePosixPath(relative).parts)
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    original = float(fields[4])
    fields[4] = fields[4] + "0"
    assert float(fields[4]) == original
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")

    with pytest.raises(IntegrityError, match="row, byte, or checksum"):
        validate_practical_delivery(tmp_path)
