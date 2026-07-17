from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from prpg.errors import GenerationError, IntegrityError
from prpg.simulation.aggregation import (
    aggregate_hf_weekly_log_returns,
    aggregate_lf_log_returns,
)
from prpg.simulation.serial import (
    SERIAL_BASE_PATH_SCHEMA_VERSION,
    SERIAL_GENERATOR_VERSION,
    RestartAudit,
    RestartReason,
    SerialBasePath,
    _array_fingerprint,
    _immutable_array,
    _record_fingerprint,
    is_sealed_serial_base_path,
)
from prpg.storage.csv_v1 import (
    CSV_HEADER,
    CSV_SCHEMA_VERSION,
    CSVShardReceipt,
    format_csv_float64,
    is_sealed_csv_shard_receipt,
    write_serial_csv_shard,
)
from prpg.storage.group_commit import (
    LoadedWorkUnitReceipt,
    StorageCommitBinding,
    commit_hf_serial_work_unit,
    commit_lf_serial_work_unit,
    is_verified_work_unit_receipt,
    quarantine_serial_work_unit,
    serial_work_unit_id,
    verify_serial_work_unit,
)


def _sealed_path(
    family: str,
    entity: int,
    years: int,
    *,
    golden_first_row: bool = False,
) -> SerialBasePath:
    import prpg.simulation.serial as module

    monthly_count = 12 * years
    count = monthly_count if family == "LF" else 252 * years
    frequency = "monthly" if family == "LF" else "daily"
    values = (
        np.arange(count * 3, dtype=np.float64).reshape(count, 3) + entity * 1_000
    ) * 1e-7
    if golden_first_row:
        values[0] = [-0.0, np.nextafter(0.0, 1.0), -1.25]
    monthly_states = np.zeros(monthly_count, dtype=np.int64)
    target_states = monthly_states if family == "LF" else np.repeat(monthly_states, 21)
    restart_flags = np.zeros(count, dtype=np.bool_)
    restart_flags[0] = True
    restart_used = np.ones(count, dtype=np.bool_)
    restart_used[0] = False
    rank_used = np.zeros(count, dtype=np.bool_)
    rank_used[0] = True
    reasons = (
        RestartReason.FIRST_OBSERVATION,
        *(RestartReason.CONTINUATION for _ in range(count - 1)),
    )
    audit = RestartAudit(
        restart_flags=_immutable_array(
            restart_flags, dtype=np.bool_, label="restart flags"
        ),
        reasons=reasons,
        restart_uniform_used=_immutable_array(
            restart_used, dtype=np.bool_, label="restart use"
        ),
        source_rank_uniform_used=_immutable_array(
            rank_used, dtype=np.bool_, label="rank use"
        ),
        restart_uniforms_consumed=count,
        source_rank_uniforms_consumed=count,
    )
    arrays = {
        "log_returns": _immutable_array(values, dtype=np.float64, label="log returns"),
        "target_monthly_states": _immutable_array(
            monthly_states, dtype=np.int64, label="monthly states"
        ),
        "target_states": _immutable_array(
            target_states, dtype=np.int64, label="target states"
        ),
        "source_indices": _immutable_array(
            np.zeros(count, dtype=np.int64),
            dtype=np.int64,
            label="source indices",
        ),
        "restart_flags": audit.restart_flags,
        "restart_uniform_used": audit.restart_uniform_used,
        "source_rank_uniform_used": audit.source_rank_uniform_used,
    }
    metadata = {
        "schema_version": SERIAL_BASE_PATH_SCHEMA_VERSION,
        "generator_version": SERIAL_GENERATOR_VERSION,
        "execution_mode": "canonical",
        "generation_authorized": True,
        "generation_input_fingerprint": "a" * 64,
        "family": family,
        "base_frequency": frequency,
        "path_entity": entity,
        "years": years,
        "state_uniforms_consumed": monthly_count,
        "restart_uniforms_consumed": count,
        "source_rank_uniforms_consumed": count,
        "restart_reasons": [value.value for value in reasons],
    }
    fingerprint = _record_fingerprint(
        metadata,
        {
            name: _array_fingerprint(value, name)
            for name, value in sorted(arrays.items())
        },
    )
    result = object.__new__(SerialBasePath)
    for name, value in (
        ("schema_version", SERIAL_BASE_PATH_SCHEMA_VERSION),
        ("generator_version", SERIAL_GENERATOR_VERSION),
        ("execution_mode", "canonical"),
        ("generation_authorized", True),
        ("generation_input_fingerprint", "a" * 64),
        ("family", family),
        ("base_frequency", frequency),
        ("path_entity", entity),
        ("years", years),
        ("log_returns", arrays["log_returns"]),
        ("target_monthly_states", arrays["target_monthly_states"]),
        ("target_states", arrays["target_states"]),
        ("source_indices", arrays["source_indices"]),
        ("restart_audit", audit),
        ("state_uniforms_consumed", monthly_count),
        ("fingerprint", fingerprint),
        ("_seal", module._BASE_PATH_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    assert is_sealed_serial_base_path(result)
    return result


def _destination(root: Path, receipt: CSVShardReceipt) -> Path:
    return root / receipt.relative_path


def test_float_grammar_golden_values_and_rejections() -> None:
    assert format_csv_float64(0.0) == "0"
    assert format_csv_float64(-0.0) == "0"
    assert format_csv_float64(1.2345678901234567) == "1.2345678901234567"
    assert format_csv_float64(np.nextafter(0.0, 1.0)) == ("4.9406564584124654e-324")
    assert format_csv_float64(-1.25) == "-1.25"
    assert format_csv_float64(1e20) == "1e+20"
    for value in (True, 1, np.nan, np.inf, -np.inf):
        with pytest.raises(GenerationError):
            format_csv_float64(cast(Any, value))


def test_monthly_golden_bytes_order_receipt_and_round_trip(tmp_path: Path) -> None:
    first = _sealed_path("LF", 1, 1, golden_first_row=True)
    second = _sealed_path("LF", 2, 1)

    receipt = write_serial_csv_shard(
        tmp_path,
        frequency="monthly",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=2,
        values=(value for value in (first, second)),
    )

    path = _destination(tmp_path, receipt)
    content = path.read_bytes()
    lines = content.decode("utf-8").splitlines()
    assert lines[0] == CSV_HEADER
    assert lines[1] == ("LF000001,1,1,1,0,4.9406564584124654e-324,-1.25")
    assert lines[12].startswith("LF000001,12,1,12,")
    assert lines[13].startswith("LF000002,1,1,1,")
    assert lines[-1].startswith("LF000002,12,1,12,")
    assert len(lines) == 25
    assert b"\r" not in content and b"\n\n" not in content
    assert content.endswith(b"\n") and b"-0" not in content
    assert receipt.schema_version == CSV_SCHEMA_VERSION
    assert receipt.relative_path == (
        "consumer/returns/low/monthly/part-00000-path-LF000001-LF000002.csv"
    )
    assert receipt.path_count == 2
    assert receipt.rows == 24
    assert receipt.bytes == len(content)
    assert receipt.sha256 == hashlib.sha256(content).hexdigest()
    assert receipt.source_fingerprints == (first.fingerprint, second.fingerprint)
    assert len(receipt.fingerprint) == 64
    assert is_sealed_csv_shard_receipt(receipt)
    assert receipt.first_path_id == "LF000001"
    assert receipt.last_path_id == "LF000002"
    assert os.stat(path).st_mode & 0o777 == 0o600

    reconstructed = np.asarray(
        [[float(field) for field in row.split(",")[4:]] for row in lines[1:13]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(reconstructed, first.log_returns)


@pytest.mark.parametrize(
    ("frequency", "periods_per_year", "family"),
    [
        ("monthly", 12, "LF"),
        ("quarterly", 4, "LF"),
        ("annual", 1, "LF"),
        ("daily", 252, "HF"),
        ("weekly", 52, "HF"),
    ],
)
def test_all_five_frequency_contracts_for_arbitrary_horizon(
    tmp_path: Path,
    frequency: str,
    periods_per_year: int,
    family: str,
) -> None:
    years = 3
    lf = _sealed_path("LF", 7, years)
    hf = _sealed_path("HF", 7, years)
    lf_derived = aggregate_lf_log_returns(lf)
    hf_weekly = aggregate_hf_weekly_log_returns(hf)
    sources: dict[str, object] = {
        "monthly": lf,
        "quarterly": lf_derived,
        "annual": lf_derived,
        "daily": hf,
        "weekly": hf_weekly,
    }

    receipt = write_serial_csv_shard(
        tmp_path,
        frequency=cast(Any, frequency),
        shard_index=12,
        first_path_entity=7,
        last_path_entity=7,
        values=[cast(Any, sources[frequency])],
    )

    lines = _destination(tmp_path, receipt).read_text(encoding="utf-8").splitlines()
    assert receipt.family == family
    assert receipt.source_fingerprints == (cast(Any, sources[frequency]).fingerprint,)
    assert receipt.years == years
    assert receipt.rows == years * periods_per_year
    first_fields = lines[1].split(",")
    last_fields = lines[-1].split(",")
    assert first_fields[:4] == [f"{family}000007", "1", "1", "1"]
    assert last_fields[:4] == [
        f"{family}000007",
        str(years * periods_per_year),
        str(years),
        str(periods_per_year),
    ]
    expected_directory = "low" if family == "LF" else "high"
    assert f"returns/{expected_directory}/{frequency}/" in receipt.relative_path
    assert "part-00012" in receipt.relative_path


@pytest.mark.parametrize(
    "values",
    [
        lambda: [_sealed_path("LF", 1, 1), _sealed_path("LF", 3, 1)],
        lambda: [_sealed_path("LF", 1, 1), _sealed_path("LF", 1, 1)],
        lambda: [_sealed_path("LF", 2, 1), _sealed_path("LF", 1, 1)],
    ],
)
def test_rejects_gap_duplicate_and_out_of_order_paths(
    tmp_path: Path, values: Any
) -> None:
    with pytest.raises(GenerationError, match="duplicated, missing, or out of order"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=2,
            values=values(),
        )


def test_rejects_incomplete_excess_and_mixed_horizon_ranges(tmp_path: Path) -> None:
    cases = (
        ("incomplete", 1, 2, [_sealed_path("LF", 1, 1)]),
        (
            "more paths",
            1,
            1,
            [_sealed_path("LF", 1, 1), _sealed_path("LF", 2, 1)],
        ),
        (
            "mix path horizons",
            1,
            2,
            [_sealed_path("LF", 1, 1), _sealed_path("LF", 2, 2)],
        ),
    )
    for index, (message, first, last, values) in enumerate(cases):
        with pytest.raises(GenerationError, match=message):
            write_serial_csv_shard(
                tmp_path / f"case-{index}",
                frequency="monthly",
                shard_index=0,
                first_path_entity=first,
                last_path_entity=last,
                values=values,
            )


def test_rejects_wrong_family_frequency_unsealed_and_noniterable(
    tmp_path: Path,
) -> None:
    lf = _sealed_path("LF", 1, 1)
    hf = _sealed_path("HF", 1, 1)
    lf_derived = aggregate_lf_log_returns(lf)
    cases = (
        ("daily", [lf]),
        ("monthly", [hf]),
        ("quarterly", [lf]),
        ("weekly", [lf_derived]),
        ("monthly", [copy.copy(lf)]),
    )
    for index, (frequency, values) in enumerate(cases):
        with pytest.raises(GenerationError):
            write_serial_csv_shard(
                tmp_path / f"wrong-{index}",
                frequency=cast(Any, frequency),
                shard_index=0,
                first_path_entity=1,
                last_path_entity=1,
                values=cast(Any, values),
            )
    with pytest.raises(GenerationError, match="iterable"):
        write_serial_csv_shard(
            tmp_path / "noniterable",
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=cast(Any, object()),
        )


@pytest.mark.parametrize("frequency", ["", "Monthly", "monthly/../daily", 1])
def test_rejects_unregistered_frequency(tmp_path: Path, frequency: object) -> None:
    with pytest.raises(GenerationError, match="frequency"):
        write_serial_csv_shard(
            tmp_path,
            frequency=cast(Any, frequency),
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=[],
        )


@pytest.mark.parametrize("shard_index", [-1, True, 100_000, "0"])
def test_rejects_invalid_shard_index(tmp_path: Path, shard_index: object) -> None:
    with pytest.raises(GenerationError, match="shard index"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=cast(Any, shard_index),
            first_path_entity=1,
            last_path_entity=1,
            values=[],
        )


@pytest.mark.parametrize("entity", [0, -1, True, 1_000_000, "1"])
def test_rejects_invalid_path_range(tmp_path: Path, entity: object) -> None:
    with pytest.raises(GenerationError, match="path entity"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=cast(Any, entity),
            last_path_entity=1,
            values=[],
        )
    with pytest.raises(GenerationError, match="path entity"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=cast(Any, entity),
            values=[],
        )


def test_rejects_descending_range_relative_traversal_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(GenerationError, match="exceeds"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=2,
            last_path_entity=1,
            values=[],
        )
    with pytest.raises(GenerationError, match="absolute"):
        write_serial_csv_shard(
            Path("relative"),
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=[_sealed_path("LF", 1, 1)],
        )
    traversal = tmp_path / "inside" / ".." / "outside"
    with pytest.raises(GenerationError, match="traversal"):
        write_serial_csv_shard(
            traversal,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=[_sealed_path("LF", 1, 1)],
        )
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="symbolic link"):
        write_serial_csv_shard(
            linked,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=[_sealed_path("LF", 1, 1)],
        )


def test_no_overwrite_concurrent_publish_and_receipt_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.csv_v1 as module

    path = _sealed_path("LF", 1, 1)
    arguments = {
        "frequency": "monthly",
        "shard_index": 0,
        "first_path_entity": 1,
        "last_path_entity": 1,
        "values": [path],
    }
    receipt = write_serial_csv_shard(tmp_path, **cast(Any, arguments))
    destination = _destination(tmp_path, receipt)
    original = destination.read_bytes()
    with pytest.raises(IntegrityError, match="already exists"):
        write_serial_csv_shard(tmp_path, **cast(Any, arguments))
    assert destination.read_bytes() == original
    with pytest.raises(FrozenInstanceError):
        receipt.rows = 0  # type: ignore[misc]
    assert not is_sealed_csv_shard_receipt(copy.copy(receipt))
    monkeypatch.setattr(module, "_reject_existing_destination", lambda _path: None)
    with pytest.raises(IntegrityError, match="appeared concurrently"):
        write_serial_csv_shard(tmp_path, **cast(Any, arguments))
    assert destination.read_bytes() == original


def test_relative_directory_accepts_concurrent_directory_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.csv_v1 as module

    root = tmp_path.resolve()
    raced_directory = root / "consumer"
    original_mkdir = Path.mkdir

    def mkdir_with_concurrent_winner(
        self: Path, *args: object, **kwargs: object
    ) -> None:
        if self == raced_directory and not self.exists():
            original_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]
            raise FileExistsError("simulated concurrent directory creation")
        original_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", mkdir_with_concurrent_winner)

    prepared = module._prepare_relative_directory(root, ("consumer", "returns"))

    assert prepared == root / "consumer" / "returns"
    assert prepared.is_dir()


def test_failure_cleans_temporary_file_and_publishes_nothing(tmp_path: Path) -> None:
    first = _sealed_path("LF", 1, 1)

    def failed_stream() -> Any:
        yield first
        raise RuntimeError("injected stream failure")

    with pytest.raises(RuntimeError, match="injected"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=4,
            first_path_entity=1,
            last_path_entity=2,
            values=failed_stream(),
        )
    directory = tmp_path / "consumer/returns/low/monthly"
    assert not any(directory.iterdir())


def test_rejects_symlink_in_fixed_consumer_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    consumer = tmp_path / "consumer"
    consumer.symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError, match="unsafe"):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=[_sealed_path("LF", 1, 1)],
        )


def _storage_binding(*, alternate: bool = False) -> StorageCommitBinding:
    values = ("1", "2", "3", "4") if not alternate else ("5", "6", "7", "8")
    return StorageCommitBinding(
        simulation_fingerprint=values[0] * 64,
        materialization_fingerprint=values[1] * 64,
        toolchain_fingerprint=values[2] * 64,
        run_fingerprint=values[3] * 64,
    )


def _canonical_id(
    family: str, *, first: int = 1, last: int = 1, shard_index: int = 0
) -> str:
    return serial_work_unit_id(cast(Any, family), shard_index, first, last)


def _lf_group_sources(
    first: int = 1, last: int = 1, years: int = 1
) -> tuple[tuple[SerialBasePath, ...], tuple[Any, ...]]:
    bases = tuple(
        _sealed_path("LF", entity, years) for entity in range(first, last + 1)
    )
    return bases, tuple(aggregate_lf_log_returns(item) for item in bases)


def _hf_group_sources(
    first: int = 1, last: int = 1, years: int = 1
) -> tuple[tuple[SerialBasePath, ...], tuple[Any, ...]]:
    bases = tuple(
        _sealed_path("HF", entity, years) for entity in range(first, last + 1)
    )
    return bases, tuple(aggregate_hf_weekly_log_returns(item) for item in bases)


def _commit_lf_group(
    root: Path,
    *,
    work_unit_id: str | None = None,
    first: int = 1,
    last: int = 1,
    years: int = 1,
) -> LoadedWorkUnitReceipt:
    bases, aggregates = _lf_group_sources(first, last, years)
    return commit_lf_serial_work_unit(
        root,
        binding=_storage_binding(),
        work_unit_id=(
            _canonical_id("LF", first=first, last=last)
            if work_unit_id is None
            else work_unit_id
        ),
        worker_identity="serial-test",
        shard_index=0,
        first_path_entity=first,
        last_path_entity=last,
        expected_path_count=last - first + 1,
        base_paths=bases,
        aggregates=aggregates,
    )


def _commit_hf_group(
    root: Path,
    *,
    work_unit_id: str | None = None,
) -> LoadedWorkUnitReceipt:
    bases, weekly = _hf_group_sources()
    return commit_hf_serial_work_unit(
        root,
        binding=_storage_binding(),
        work_unit_id=_canonical_id("HF") if work_unit_id is None else work_unit_id,
        worker_identity="serial-test",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        expected_path_count=1,
        base_paths=bases,
        weekly_returns=weekly,
    )


def _group_receipt_path(root: Path, work_unit_id: str) -> Path:
    return root / "receipts" / "storage" / f"{work_unit_id}.json"


def _receipt_document(root: Path, work_unit_id: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(_group_receipt_path(root, work_unit_id).read_bytes()),
    )


def _rewrite_group_receipt(
    root: Path,
    work_unit_id: str,
    mutate: Any,
    *,
    resign: bool = True,
    canonical: bool = True,
) -> None:
    import prpg.storage.group_commit as module

    document = _receipt_document(root, work_unit_id)
    mutate(document)
    if resign:
        document["receipt_fingerprint"] = module._sha256_canonical(document["identity"])
    content = (
        module._canonical_bytes(document)
        if canonical
        else json.dumps(document, indent=2).encode("utf-8")
    )
    _group_receipt_path(root, work_unit_id).write_bytes(content)


def _resign_changed_sibling(root: Path, work_unit_id: str, sibling_index: int) -> None:
    import prpg.storage.group_commit as module

    document = _receipt_document(root, work_unit_id)
    identity = document["identity"]
    sibling = identity["siblings"][sibling_index]
    content = (root / sibling["relative_path"]).read_bytes()
    sibling["bytes"] = len(content)
    sibling["sha256"] = hashlib.sha256(content).hexdigest()
    csv_identity = module._receipt_identity(
        relative_path=sibling["relative_path"],
        frequency=sibling["frequency"],
        family=identity["family"],
        shard_index=identity["shard_index"],
        first_path_entity=identity["first_path_entity"],
        last_path_entity=identity["last_path_entity"],
        first_path_id=module._path_id(
            identity["family"], identity["first_path_entity"]
        ),
        last_path_id=module._path_id(identity["family"], identity["last_path_entity"]),
        path_count=identity["expected_path_count"],
        years=identity["years"],
        rows=sibling["rows"],
        bytes_count=sibling["bytes"],
        sha256=sibling["sha256"],
        source_fingerprints=tuple(sibling["source_fingerprints"]),
    )
    sibling["csv_receipt_fingerprint"] = module._sha256_json(csv_identity)
    document["receipt_fingerprint"] = module._sha256_canonical(identity)
    _group_receipt_path(root, work_unit_id).write_bytes(
        module._canonical_bytes(document)
    )


def test_lf_group_commit_receipt_is_canonical_durable_and_sealed(
    tmp_path: Path,
) -> None:
    import prpg.storage.group_commit as module

    receipt = _commit_lf_group(tmp_path, first=7, last=8, years=2)

    assert receipt.family == "LF"
    assert receipt.first_path_entity == 7
    assert receipt.last_path_entity == 8
    assert len(receipt.sibling_paths) == 3
    assert [item["frequency"] for item in receipt.identity["siblings"]] == [
        "monthly",
        "quarterly",
        "annual",
    ]
    assert receipt.identity["binding"] == _storage_binding().as_dict()
    assert receipt.identity["csv_schema_version"] == CSV_SCHEMA_VERSION
    assert is_verified_work_unit_receipt(receipt)
    assert not is_verified_work_unit_receipt(copy.copy(receipt))
    with pytest.raises(FrozenInstanceError):
        receipt.family = "HF"  # type: ignore[misc]

    for relative in receipt.sibling_paths:
        assert (tmp_path / relative).is_file()
        assert os.stat(tmp_path / relative).st_nlink == 1
    content = receipt.path.read_bytes()
    document = json.loads(content)
    assert content == module._canonical_bytes(document)
    assert document["receipt_fingerprint"] == module._sha256_canonical(
        document["identity"]
    )
    replay = verify_serial_work_unit(
        tmp_path,
        binding=_storage_binding(),
        work_unit_id=receipt.work_unit_id,
    )
    assert replay.fingerprint == receipt.fingerprint
    assert replay.sibling_paths == receipt.sibling_paths


def test_hf_group_commit_binds_daily_weekly_and_reverifies(tmp_path: Path) -> None:
    receipt = _commit_hf_group(tmp_path)

    assert receipt.family == "HF"
    assert len(receipt.sibling_paths) == 2
    assert [item["frequency"] for item in receipt.identity["siblings"]] == [
        "daily",
        "weekly",
    ]
    assert (
        receipt.identity["derived_base_path_fingerprints"]
        == receipt.identity["base_path_fingerprints"]
    )
    assert (
        verify_serial_work_unit(
            tmp_path, binding=_storage_binding(), work_unit_id=receipt.work_unit_id
        ).fingerprint
        == receipt.fingerprint
    )


def test_valid_duplicate_group_is_rejected_without_quarantine_or_mutation(
    tmp_path: Path,
) -> None:
    receipt = _commit_lf_group(tmp_path)
    before = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in (
            *receipt.sibling_paths,
            f"receipts/storage/{receipt.work_unit_id}.json",
        )
    }

    with pytest.raises(GenerationError, match="reuse is forbidden"):
        _commit_lf_group(tmp_path)

    after = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in before
    }
    assert after == before
    assert not (tmp_path / "quarantine").exists()


def test_cross_id_same_scope_cannot_quarantine_or_mask_valid_group(
    tmp_path: Path,
) -> None:
    receipt = _commit_lf_group(tmp_path)
    before = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in (*receipt.sibling_paths, receipt.path.relative_to(tmp_path))
    }

    with pytest.raises(GenerationError, match="canonical.*scope"):
        _commit_lf_group(tmp_path, work_unit_id="lf-wrong-scope-id")
    with pytest.raises(GenerationError, match="canonical.*scope"):
        quarantine_serial_work_unit(
            tmp_path,
            work_unit_id="lf-wrong-scope-id",
            family="LF",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
        )

    after = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in before
    }
    assert after == before
    assert (
        verify_serial_work_unit(
            tmp_path,
            binding=_storage_binding(),
            work_unit_id=receipt.work_unit_id,
        ).fingerprint
        == receipt.fingerprint
    )
    assert not (tmp_path / "quarantine").exists()


def test_all_siblings_are_staged_and_verified_before_first_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    real_writer = module.write_serial_csv_shard
    real_publish = module._atomic_publish_no_replace
    writes: list[Path] = []
    publication_write_counts: list[int] = []

    def recording_writer(root: Path, **kwargs: Any) -> CSVShardReceipt:
        writes.append(root)
        assert root != tmp_path
        assert not list((tmp_path / "consumer").rglob("*.csv"))
        return real_writer(root, **kwargs)

    def recording_publish(source: Path, destination: Path) -> None:
        publication_write_counts.append(len(writes))
        real_publish(source, destination)

    monkeypatch.setattr(module, "write_serial_csv_shard", recording_writer)
    monkeypatch.setattr(module, "_atomic_publish_no_replace", recording_publish)
    receipt = _commit_lf_group(tmp_path)

    assert len(writes) == 3
    assert len(set(writes)) == 1
    assert publication_write_counts[:3] == [3, 3, 3]
    assert all((tmp_path / relative).is_file() for relative in receipt.sibling_paths)
    assert not list(tmp_path.glob(f".prpg-stage-{receipt.work_unit_id}-*"))


def test_receipt_binds_validated_worker_and_measured_monotonic_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    real_clock = module.time.monotonic_ns
    ticks = iter((1_000, 1_075))
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: next(ticks))
    receipt = _commit_lf_group(tmp_path)

    assert receipt.worker_identity == "serial-test"
    assert receipt.identity["worker_identity"] == "serial-test"
    assert receipt.duration_monotonic_ns == 75
    assert receipt.identity["duration_monotonic_ns"] == 75

    monkeypatch.setattr(module.time, "monotonic_ns", real_clock)
    bases, aggregates = _lf_group_sources()
    with pytest.raises(GenerationError, match="worker identity"):
        commit_lf_serial_work_unit(
            tmp_path / "invalid-worker",
            binding=_storage_binding(),
            work_unit_id=_canonical_id("LF"),
            worker_identity="worker identity with spaces",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            expected_path_count=1,
            base_paths=bases,
            aggregates=aggregates,
        )


def test_verifier_rejects_mixed_external_binding_without_changing_group(
    tmp_path: Path,
) -> None:
    receipt = _commit_lf_group(tmp_path)
    with pytest.raises(IntegrityError, match="binding differs"):
        verify_serial_work_unit(
            tmp_path,
            binding=_storage_binding(alternate=True),
            work_unit_id=receipt.work_unit_id,
        )
    assert all((tmp_path / relative).is_file() for relative in receipt.sibling_paths)


def test_injected_staging_failure_publishes_nothing_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    real_writer = module.write_serial_csv_shard
    calls = 0

    def fail_second_write(*args: Any, **kwargs: Any) -> CSVShardReceipt:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected sibling failure")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(module, "write_serial_csv_shard", fail_second_write)
    with pytest.raises(RuntimeError, match="injected sibling failure"):
        _commit_lf_group(tmp_path)

    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.glob(f".prpg-stage-{_canonical_id('LF')}-*"))
    assert not (tmp_path / "quarantine").exists()


def test_unsealed_staged_receipt_is_rejected_before_any_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    real_writer = module.write_serial_csv_shard

    def copied_receipt(*args: Any, **kwargs: Any) -> CSVShardReceipt:
        return copy.copy(real_writer(*args, **kwargs))

    monkeypatch.setattr(module, "write_serial_csv_shard", copied_receipt)
    with pytest.raises(IntegrityError, match="unsealed"):
        _commit_lf_group(tmp_path)
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.glob(".prpg-stage-*"))


def test_staging_creation_and_backwards_clock_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    monkeypatch.setattr(
        module.tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("injected mkdtemp")),
    )
    with pytest.raises(IntegrityError, match="staging directory"):
        _commit_lf_group(tmp_path / "stage-error")
    assert not list((tmp_path / "stage-error").rglob("*.csv"))

    monkeypatch.undo()
    clock_root = tmp_path / "clock-error"
    ticks = iter((100, 99))
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(IntegrityError, match="clock moved backwards"):
        _commit_lf_group(clock_root)
    assert not list((clock_root / "consumer").rglob("*.csv"))
    assert len(list((clock_root / "quarantine").rglob("*.csv"))) == 3


def test_injected_partial_publication_quarantines_every_canonical_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    real_publish = module._atomic_publish_no_replace
    calls = 0

    def fail_second_publish(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication failure")
        real_publish(source, destination)

    monkeypatch.setattr(module, "_atomic_publish_no_replace", fail_second_publish)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        _commit_lf_group(tmp_path)

    assert not list((tmp_path / "consumer").rglob("*.csv"))
    attempts = list(
        (tmp_path / "quarantine/storage" / _canonical_id("LF")).glob("attempt-*")
    )
    assert len(attempts) == 1
    moved = list(attempts[0].rglob("*.csv"))
    assert len(moved) == 1
    assert "monthly" in moved[0].parts


def test_sigkill_hardlink_window_is_safely_reconciled_then_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    real_discard = module._discard_stage_root

    def simulate_sigkill(_path: Path) -> None:
        raise RuntimeError("simulated SIGKILL window")

    monkeypatch.setattr(module, "_discard_stage_root", simulate_sigkill)
    with pytest.raises(RuntimeError, match="SIGKILL window"):
        _commit_lf_group(tmp_path)

    sibling_paths = module._sibling_relative_paths("LF", 0, 1, 1)
    stages = list(tmp_path.glob(f".prpg-stage-{_canonical_id('LF')}-*"))
    assert len(stages) == 1
    for relative in sibling_paths:
        staged = stages[0] / relative
        canonical = tmp_path / relative
        assert staged.is_file() and canonical.is_file()
        assert os.stat(staged).st_ino == os.stat(canonical).st_ino
        assert os.stat(canonical).st_nlink == 2

    monkeypatch.setattr(module, "_discard_stage_root", real_discard)
    with pytest.raises(IntegrityError, match="without a valid"):
        _commit_lf_group(tmp_path)

    assert not list(tmp_path.glob(f".prpg-stage-{_canonical_id('LF')}-*"))
    assert not list((tmp_path / "consumer").rglob("*.csv"))
    quarantined = list((tmp_path / "quarantine").rglob("*.csv"))
    assert len(quarantined) == 3
    assert all(os.stat(path).st_nlink == 1 for path in quarantined)

    receipt = _commit_lf_group(tmp_path)
    assert all(os.stat(tmp_path / path).st_nlink == 1 for path in receipt.sibling_paths)
    assert (
        verify_serial_work_unit(
            tmp_path,
            binding=_storage_binding(),
            work_unit_id=receipt.work_unit_id,
        ).fingerprint
        == receipt.fingerprint
    )


def test_stale_unpublished_stage_is_discarded_before_fresh_generation(
    tmp_path: Path,
) -> None:
    import prpg.storage.group_commit as module

    stage = module._create_stage_root(tmp_path, _canonical_id("LF"))
    base = _sealed_path("LF", 1, 1)
    staged = write_serial_csv_shard(
        stage,
        frequency="monthly",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        values=[base],
    )
    assert (stage / staged.relative_path).is_file()

    receipt = _commit_lf_group(tmp_path)
    assert not stage.exists()
    assert all((tmp_path / path).is_file() for path in receipt.sibling_paths)


def test_stale_stage_reconciliation_rejects_unowned_or_unsafe_aliases(
    tmp_path: Path,
) -> None:
    import prpg.storage.group_commit as module

    unsafe_root = tmp_path / "unsafe-root"
    unsafe_root.mkdir()
    (unsafe_root / f".prpg-stage-{_canonical_id('LF')}-bad").mkdir()
    with pytest.raises(IntegrityError, match="staging root is unsafe"):
        _commit_lf_group(unsafe_root)

    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    symlink_stage = symlink_root / f".prpg-stage-{_canonical_id('LF')}-abcdefgh"
    symlink_stage.mkdir()
    target = symlink_root / "target"
    target.write_text("target", encoding="utf-8")
    (symlink_stage / "link").symlink_to(target)
    with pytest.raises(IntegrityError, match="symbolic link"):
        _commit_lf_group(symlink_root)

    alias_root = tmp_path / "alias-root"
    alias_root.mkdir()
    alias_stage = module._create_stage_root(alias_root, _canonical_id("LF"))
    alias_receipt = write_serial_csv_shard(
        alias_stage,
        frequency="monthly",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        values=[_sealed_path("LF", 1, 1)],
    )
    os.link(alias_stage / alias_receipt.relative_path, alias_root / "unknown-alias")
    with pytest.raises(IntegrityError, match="unknown hard-link alias"):
        _commit_lf_group(alias_root)

    ownership_root = tmp_path / "ownership-root"
    ownership_root.mkdir()
    ownership_stage = module._create_stage_root(ownership_root, _canonical_id("LF"))
    staged_receipt = write_serial_csv_shard(
        ownership_stage,
        frequency="monthly",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        values=[_sealed_path("LF", 1, 1)],
    )
    canonical_receipt = write_serial_csv_shard(
        ownership_root,
        frequency="monthly",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        values=[_sealed_path("LF", 1, 1, golden_first_row=True)],
    )
    staged_path = ownership_stage / staged_receipt.relative_path
    canonical_path = ownership_root / canonical_receipt.relative_path
    os.link(staged_path, ownership_root / "staged-alias")
    os.link(canonical_path, ownership_root / "canonical-alias")
    with pytest.raises(IntegrityError, match="ownership or content differs"):
        _commit_lf_group(ownership_root)


def test_stage_cleanup_and_loaded_receipt_type_guard_fail_closed(
    tmp_path: Path,
) -> None:
    import prpg.storage.group_commit as module

    module._discard_stage_root(tmp_path / "missing-stage")
    target = tmp_path / "target-stage"
    target.mkdir()
    stage_link = tmp_path / "stage-link"
    stage_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="changed before cleanup"):
        module._discard_stage_root(stage_link)

    stage_tree = tmp_path / "stage-tree"
    stage_tree.mkdir()
    (stage_tree / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="contains a symbolic link"):
        module._discard_stage_root(stage_tree)

    assert not is_verified_work_unit_receipt(object())
    bad_field = _commit_lf_group(tmp_path / "bad-field")
    object.__setattr__(bad_field, "fingerprint", "bad")
    assert not is_verified_work_unit_receipt(bad_field)
    bad_identity = _commit_lf_group(tmp_path / "bad-identity")
    object.__setattr__(bad_identity, "family", "HF")
    assert not is_verified_work_unit_receipt(bad_identity)


def test_orphan_sibling_is_quarantined_before_new_commit(tmp_path: Path) -> None:
    base = _sealed_path("LF", 1, 1)
    orphan = write_serial_csv_shard(
        tmp_path,
        frequency="monthly",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
        values=[base],
    )

    with pytest.raises(IntegrityError, match="without a valid"):
        _commit_lf_group(tmp_path)

    assert not (tmp_path / orphan.relative_path).exists()
    moved = list((tmp_path / "quarantine").rglob("*.csv"))
    assert len(moved) == 1


def test_invalid_existing_receipt_quarantines_entire_group(tmp_path: Path) -> None:
    receipt = _commit_lf_group(tmp_path)
    path = receipt.path
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(IntegrityError, match="canonical"):
        _commit_lf_group(tmp_path)

    assert not path.exists()
    assert all(not (tmp_path / relative).exists() for relative in receipt.sibling_paths)
    quarantined = list((tmp_path / "quarantine").rglob("*"))
    assert sum(item.is_file() for item in quarantined) == 4


def test_corrupt_sibling_invalidates_then_quarantines_whole_lf_group(
    tmp_path: Path,
) -> None:
    receipt = _commit_lf_group(tmp_path)
    corrupt = tmp_path / receipt.sibling_paths[0]
    content = corrupt.read_bytes()
    corrupt.write_bytes(b"wrong-header\n" + content.split(b"\n", 1)[1])

    with pytest.raises(IntegrityError, match="header"):
        verify_serial_work_unit(
            tmp_path, binding=_storage_binding(), work_unit_id=receipt.work_unit_id
        )
    moved = quarantine_serial_work_unit(
        tmp_path,
        work_unit_id=receipt.work_unit_id,
        family="LF",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
    )
    assert set(moved.moved_relative_paths) == {
        *receipt.sibling_paths,
        f"receipts/storage/{receipt.work_unit_id}.json",
    }
    assert all(not (tmp_path / relative).exists() for relative in receipt.sibling_paths)


def test_missing_hf_sibling_quarantines_remaining_set_and_receipt(
    tmp_path: Path,
) -> None:
    receipt = _commit_hf_group(tmp_path)
    (tmp_path / receipt.sibling_paths[1]).unlink()
    with pytest.raises(IntegrityError, match="cannot be opened"):
        verify_serial_work_unit(
            tmp_path, binding=_storage_binding(), work_unit_id=receipt.work_unit_id
        )

    moved = quarantine_serial_work_unit(
        tmp_path,
        work_unit_id=receipt.work_unit_id,
        family="HF",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
    )
    assert moved.moved_relative_paths == (
        receipt.sibling_paths[0],
        f"receipts/storage/{receipt.work_unit_id}.json",
    )


def test_hardlink_and_symlink_surprises_fail_closed_before_quarantine(
    tmp_path: Path,
) -> None:
    hard_root = tmp_path / "hard"
    hard_receipt = _commit_lf_group(hard_root)
    hard_path = hard_root / hard_receipt.sibling_paths[0]
    os.link(hard_path, hard_root / "unexpected-alias.csv")
    with pytest.raises(IntegrityError, match="single-link"):
        verify_serial_work_unit(
            hard_root,
            binding=_storage_binding(),
            work_unit_id=hard_receipt.work_unit_id,
        )
    with pytest.raises(IntegrityError, match="single-link"):
        quarantine_serial_work_unit(
            hard_root,
            work_unit_id=hard_receipt.work_unit_id,
            family="LF",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
        )
    assert hard_receipt.path.exists()

    link_root = tmp_path / "link"
    link_receipt = _commit_lf_group(link_root)
    link_path = link_root / link_receipt.sibling_paths[0]
    target = link_root / "unexpected-target.csv"
    target.write_bytes(link_path.read_bytes())
    link_path.unlink()
    link_path.symlink_to(target)
    with pytest.raises(IntegrityError, match="symbolic link"):
        verify_serial_work_unit(
            link_root,
            binding=_storage_binding(),
            work_unit_id=link_receipt.work_unit_id,
        )
    with pytest.raises(IntegrityError, match="symbolic link"):
        quarantine_serial_work_unit(
            link_root,
            work_unit_id=link_receipt.work_unit_id,
            family="LF",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
        )
    assert link_receipt.path.exists()


@pytest.mark.parametrize(
    ("case", "mutate", "message"),
    [
        (
            "outer-schema",
            lambda doc: doc.__setitem__("schema_version", 2),
            "schema is unsupported",
        ),
        (
            "csv-schema",
            lambda doc: doc["identity"].__setitem__("csv_schema_version", 2),
            "schema is unsupported",
        ),
        (
            "serializer",
            lambda doc: doc["identity"].__setitem__("csv_serializer_version", "other"),
            "schema is unsupported",
        ),
        (
            "work-id",
            lambda doc: doc["identity"].__setitem__("work_unit_id", "different"),
            "semantic fields",
        ),
        (
            "bad-binding",
            lambda doc: doc["identity"]["binding"].__setitem__(
                "simulation_fingerprint", "bad"
            ),
            "semantic fields",
        ),
        (
            "binding-shape",
            lambda doc: doc["identity"]["binding"].pop("run_fingerprint"),
            "binding is invalid",
        ),
        (
            "bad-worker",
            lambda doc: doc["identity"].__setitem__("worker_identity", "bad worker"),
            "semantic fields",
        ),
        (
            "bad-duration",
            lambda doc: doc["identity"].__setitem__("duration_monotonic_ns", -1),
            "semantic fields",
        ),
        (
            "lineage",
            lambda doc: doc["identity"].__setitem__(
                "derived_base_path_fingerprints", ["f" * 64]
            ),
            "lineage differs",
        ),
        (
            "base-count",
            lambda doc: doc["identity"].__setitem__("base_path_fingerprints", []),
            "fingerprint count",
        ),
        (
            "range-count",
            lambda doc: doc["identity"].__setitem__("expected_path_count", 2),
            "count/range differs",
        ),
        (
            "sibling-count",
            lambda doc: doc["identity"]["siblings"].pop(),
            "receipt count",
        ),
        (
            "sibling-frequency",
            lambda doc: doc["identity"]["siblings"][0].__setitem__(
                "frequency", "daily"
            ),
            "frequency/path",
        ),
        (
            "sibling-extra",
            lambda doc: doc["identity"]["siblings"][0].__setitem__("unexpected", True),
            "sibling identity keys",
        ),
        (
            "sibling-sha",
            lambda doc: doc["identity"]["siblings"][0].__setitem__("sha256", "bad"),
            "sibling sha256",
        ),
        (
            "sibling-source",
            lambda doc: doc["identity"]["siblings"][0].__setitem__(
                "source_fingerprints", ["e" * 64]
            ),
            "source lineage",
        ),
        (
            "embedded-receipt",
            lambda doc: doc["identity"]["siblings"][0].__setitem__(
                "csv_receipt_fingerprint", "e" * 64
            ),
            "embedded CSV receipt",
        ),
        (
            "rows",
            lambda doc: doc["identity"]["siblings"][0].__setitem__("rows", 13),
            "row count",
        ),
        (
            "identity-extra",
            lambda doc: doc["identity"].__setitem__("unexpected", True),
            "identity keys",
        ),
        (
            "document-extra",
            lambda doc: doc.__setitem__("unexpected", True),
            "document keys",
        ),
        (
            "identity-nondict",
            lambda doc: doc.__setitem__("identity", []),
            "identity is invalid",
        ),
    ],
)
def test_durable_verifier_rejects_resigned_receipt_mutations(
    tmp_path: Path, case: str, mutate: Any, message: str
) -> None:
    root = tmp_path / case
    _commit_lf_group(root)
    work_unit_id = _canonical_id("LF")
    _rewrite_group_receipt(root, work_unit_id, mutate)

    with pytest.raises(IntegrityError, match=message):
        verify_serial_work_unit(
            root, binding=_storage_binding(), work_unit_id=work_unit_id
        )


def test_verifier_rejects_invalid_json_and_unresigned_receipt_content(
    tmp_path: Path,
) -> None:
    import prpg.storage.group_commit as module

    invalid_root = tmp_path / "invalid-json"
    invalid = _commit_lf_group(invalid_root)
    invalid.path.write_bytes(b"{")
    with pytest.raises(IntegrityError, match="invalid JSON"):
        verify_serial_work_unit(
            invalid_root,
            binding=_storage_binding(),
            work_unit_id=invalid.work_unit_id,
        )

    hash_root = tmp_path / "content-hash"
    receipt = _commit_lf_group(hash_root)
    document = _receipt_document(hash_root, receipt.work_unit_id)
    document["receipt_fingerprint"] = "e" * 64
    receipt.path.write_bytes(module._canonical_bytes(document))
    with pytest.raises(IntegrityError, match="content hash"):
        verify_serial_work_unit(
            hash_root,
            binding=_storage_binding(),
            work_unit_id=receipt.work_unit_id,
        )


@pytest.mark.parametrize(
    ("case", "transform", "message"),
    [
        (
            "header",
            lambda content: b"wrong\n" + content.split(b"\n", 1)[1],
            "header",
        ),
        (
            "path-key",
            lambda content: content.replace(b"LF000001,1,", b"LF000002,1,", 1),
            "row key",
        ),
        (
            "noncanonical-float",
            lambda content: content.replace(
                b",9.9999999999999991e-05,", b",09.9999999999999991e-05,", 1
            ),
            "noncanonical",
        ),
        (
            "crlf",
            lambda content: content.split(b"\n", 1)[0]
            + b"\n"
            + content.split(b"\n", 1)[1].replace(b"\n", b"\r\n", 1),
            "line ending",
        ),
        (
            "non-utf8",
            lambda content: content.replace(b"LF000001,1,", b"\xffF000001,1,", 1),
            "UTF-8",
        ),
        (
            "non-decimal",
            lambda content: content.replace(
                b",9.9999999999999991e-05,", b",not-decimal,", 1
            ),
            "not decimal",
        ),
        (
            "nonfinite",
            lambda content: content.replace(b",9.9999999999999991e-05,", b",nan,", 1),
            "non-finite",
        ),
        (
            "extra-field",
            lambda content: content.replace(
                b"LF000001,1,1,1,", b"LF000001,1,1,1,x,", 1
            ),
            "field contract",
        ),
        (
            "excess-row",
            lambda content: content + content.splitlines(keepends=True)[1],
            "excess path rows",
        ),
    ],
)
def test_csv_semantics_reject_tampering_even_after_all_hashes_are_resigned(
    tmp_path: Path, case: str, transform: Any, message: str
) -> None:
    root = tmp_path / case
    receipt = _commit_lf_group(root)
    monthly = root / receipt.sibling_paths[0]
    monthly.write_bytes(transform(monthly.read_bytes()))
    _resign_changed_sibling(root, receipt.work_unit_id, 0)

    with pytest.raises(IntegrityError, match=message):
        verify_serial_work_unit(
            root, binding=_storage_binding(), work_unit_id=receipt.work_unit_id
        )


def test_verifier_rejects_semantically_valid_csv_bytes_with_stale_hash(
    tmp_path: Path,
) -> None:
    receipt = _commit_lf_group(tmp_path)
    monthly = tmp_path / receipt.sibling_paths[0]
    content = monthly.read_bytes()
    monthly.write_bytes(
        content.replace(b",9.9999999999999991e-05,", b",8.9999999999999992e-05,", 1)
    )

    with pytest.raises(IntegrityError, match="hash/size/row contract"):
        verify_serial_work_unit(
            tmp_path,
            binding=_storage_binding(),
            work_unit_id=receipt.work_unit_id,
        )


def test_source_scope_and_horizon_validation_happens_before_any_write(
    tmp_path: Path,
) -> None:
    base = _sealed_path("LF", 1, 1)
    other = _sealed_path("LF", 1, 1, golden_first_row=True)
    cases = [
        {
            "expected_path_count": 2,
            "base_paths": (base,),
            "aggregates": (aggregate_lf_log_returns(base),),
        },
        {
            "expected_path_count": 1,
            "base_paths": (base,),
            "aggregates": (aggregate_lf_log_returns(other),),
        },
        {
            "expected_path_count": 1,
            "base_paths": (copy.copy(base),),
            "aggregates": (aggregate_lf_log_returns(base),),
        },
    ]
    for index, values in enumerate(cases):
        root = tmp_path / f"source-{index}"
        with pytest.raises((GenerationError, IntegrityError)):
            commit_lf_serial_work_unit(
                root,
                binding=_storage_binding(),
                work_unit_id=f"lf-source-{index}",
                worker_identity="serial-test",
                shard_index=0,
                first_path_entity=1,
                last_path_entity=1,
                expected_path_count=cast(int, values["expected_path_count"]),
                base_paths=cast(Any, values["base_paths"]),
                aggregates=cast(Any, values["aggregates"]),
            )
        assert not root.exists()


def test_source_collections_mixed_horizons_and_hf_lineage_fail_prewrite(
    tmp_path: Path,
) -> None:
    first = _sealed_path("LF", 1, 1)
    second = _sealed_path("LF", 2, 2)
    lf_cases = (
        (cast(Any, object()), (aggregate_lf_log_returns(first),), 1, 1),
        ((), (), 1, 1),
        (
            (first, second),
            (aggregate_lf_log_returns(first), aggregate_lf_log_returns(second)),
            1,
            2,
        ),
    )
    for index, (bases, aggregates, first_entity, last_entity) in enumerate(lf_cases):
        root = tmp_path / f"lf-collection-{index}"
        with pytest.raises(GenerationError):
            commit_lf_serial_work_unit(
                root,
                binding=_storage_binding(),
                work_unit_id=_canonical_id("LF", first=first_entity, last=last_entity),
                worker_identity="serial-test",
                shard_index=0,
                first_path_entity=first_entity,
                last_path_entity=last_entity,
                expected_path_count=last_entity - first_entity + 1,
                base_paths=cast(Any, bases),
                aggregates=cast(Any, aggregates),
            )
        assert not root.exists()

    hf = _sealed_path("HF", 1, 1)
    other_hf = _sealed_path("HF", 1, 1, golden_first_row=True)
    with pytest.raises(IntegrityError, match="HF work-unit source"):
        commit_hf_serial_work_unit(
            tmp_path / "hf-lineage",
            binding=_storage_binding(),
            work_unit_id=_canonical_id("HF"),
            worker_identity="serial-test",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            expected_path_count=1,
            base_paths=(hf,),
            weekly_returns=(aggregate_hf_weekly_log_returns(other_hf),),
        )
    assert not (tmp_path / "hf-lineage").exists()


def test_registered_scope_and_boundary_validators_reject_invalid_inputs(
    tmp_path: Path,
) -> None:
    for arguments in (
        ("XX", 0, 1, 1),
        ("LF", -1, 1, 1),
        ("LF", 100_000, 1, 1),
        ("LF", 0, 0, 1),
        ("LF", 0, 1_000_000, 1_000_000),
        ("LF", 0, 2, 1),
    ):
        with pytest.raises(GenerationError):
            serial_work_unit_id(*cast(Any, arguments))

    receipt = _commit_lf_group(tmp_path)
    with pytest.raises(GenerationError, match="binding must be typed"):
        verify_serial_work_unit(
            tmp_path,
            binding=cast(Any, object()),
            work_unit_id=receipt.work_unit_id,
        )
    with pytest.raises(GenerationError, match="output root must be a path"):
        verify_serial_work_unit(
            cast(Any, object()),
            binding=_storage_binding(),
            work_unit_id=receipt.work_unit_id,
        )


@pytest.mark.parametrize("work_unit_id", ["../escape", "UPPER", "", "a/b", "a" * 64])
def test_group_rejects_unsafe_work_unit_ids(tmp_path: Path, work_unit_id: str) -> None:
    bases, aggregates = _lf_group_sources()
    with pytest.raises(GenerationError, match="work-unit ID"):
        commit_lf_serial_work_unit(
            tmp_path / work_unit_id.replace("/", "-"),
            binding=_storage_binding(),
            work_unit_id=work_unit_id,
            worker_identity="serial-test",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            expected_path_count=1,
            base_paths=bases,
            aggregates=aggregates,
        )


def test_group_rejects_relative_symlink_roots_and_empty_quarantine(
    tmp_path: Path,
) -> None:
    with pytest.raises(GenerationError, match="absolute"):
        verify_serial_work_unit(
            Path("relative"),
            binding=_storage_binding(),
            work_unit_id=_canonical_id("LF"),
        )
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="unsafe"):
        verify_serial_work_unit(
            linked, binding=_storage_binding(), work_unit_id=_canonical_id("LF")
        )
    with pytest.raises(IntegrityError, match="no canonical"):
        quarantine_serial_work_unit(
            target,
            work_unit_id=_canonical_id("LF"),
            family="LF",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
        )


def test_quarantine_allocates_new_attempt_and_wraps_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prpg.storage.group_commit as module

    base = _sealed_path("LF", 1, 1)

    def write_orphan() -> None:
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=[base],
        )

    write_orphan()
    first = quarantine_serial_work_unit(
        tmp_path,
        work_unit_id=_canonical_id("LF"),
        family="LF",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
    )
    assert first.quarantine_relative_path.endswith("attempt-00000")

    write_orphan()
    second = quarantine_serial_work_unit(
        tmp_path,
        work_unit_id=_canonical_id("LF"),
        family="LF",
        shard_index=0,
        first_path_entity=1,
        last_path_entity=1,
    )
    assert second.quarantine_relative_path.endswith("attempt-00001")

    write_orphan()
    monkeypatch.setattr(
        module.os,
        "rename",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("injected rename")),
    )
    with pytest.raises(IntegrityError, match="quarantine move failed"):
        quarantine_serial_work_unit(
            tmp_path,
            work_unit_id=_canonical_id("LF"),
            family="LF",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
        )
