from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.unit.test_g3_evidence_store import (
    _build_fixture,
    _counterfeit_manifest,
    _Fixture,
)

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_artifact_handoff import handoff_completed_g3
from prpg.model.g3_evidence_store import G3_EVIDENCE_MANIFEST_NAME
from prpg.model.scientific_artifact_pair import (
    SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY,
    SCIENTIFIC_ARTIFACT_PAIR_RECEIPT,
)


@pytest.fixture(scope="module")
def completed_g3(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_Fixture]:
    monkeypatch = pytest.MonkeyPatch()
    try:
        fixture = _build_fixture(
            tmp_path_factory.mktemp("g3-handoff-completed"), monkeypatch
        )
        pair_root = (
            fixture.evidence_store.artifact_root
            / SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY
            / "sha256"
            / ("3" * 64)
        )
        pair_root.mkdir(parents=True)
        (pair_root / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT).write_bytes(b"fixture")
        yield fixture
    finally:
        monkeypatch.undo()


def test_handoff_copies_only_required_private_bytes_and_returns_receipt(
    tmp_path: Path,
    completed_g3: _Fixture,
) -> None:
    source = completed_g3.evidence_store.artifact_root.parent
    destination = tmp_path / "downstream-runtime"

    receipt = handoff_completed_g3(
        source,
        destination,
        evidence_fingerprint=completed_g3.evidence_fingerprint,
    )

    assert receipt.run_id == completed_g3.run_id
    assert receipt.evidence_fingerprint == completed_g3.evidence_fingerprint
    assert receipt.completion_fingerprint == completed_g3.completion.fingerprint
    assert receipt.design_artifact_fingerprint == (
        completed_g3.design.reference.fingerprint
    )
    assert receipt.production_artifact_fingerprint == (
        completed_g3.production.reference.fingerprint
    )
    manifest = completed_g3.evidence_store.verify(
        completed_g3.evidence_fingerprint
    ).manifest
    assert (
        receipt.pair_receipt_fingerprint
        == (manifest["identity"]["artifacts"]["pair_receipt_fingerprint"])
    )
    assert receipt.checkpoint_count == 26
    assert receipt.copied_file_count > 26
    assert receipt.copied_bytes > 0
    assert not (destination / "artifacts" / "data").exists()
    for path in destination.rglob("*"):
        information = path.lstat()
        assert not stat.S_ISLNK(information.st_mode)
        if stat.S_ISREG(information.st_mode):
            assert information.st_nlink == 1


def test_handoff_refuses_existing_destination_without_changing_it(
    tmp_path: Path,
    completed_g3: _Fixture,
) -> None:
    source = completed_g3.evidence_store.artifact_root.parent
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "owner.txt"
    marker.write_text("keep")

    with pytest.raises(ModelError, match="overwrite refused"):
        handoff_completed_g3(
            source,
            destination,
            evidence_fingerprint=completed_g3.evidence_fingerprint,
        )

    assert marker.read_text() == "keep"


def test_handoff_refuses_missing_and_incomplete_g3_without_destination(
    tmp_path: Path,
    completed_g3: _Fixture,
) -> None:
    missing = tmp_path / "missing-evidence"
    missing.mkdir()
    with pytest.raises(IntegrityError, match="missing|unsafe"):
        handoff_completed_g3(
            missing,
            tmp_path / "missing-output",
            evidence_fingerprint="1" * 64,
        )
    assert not (tmp_path / "missing-output").exists()

    incomplete = tmp_path / "incomplete"
    evidence_source = (
        completed_g3.evidence_store.store_root / completed_g3.evidence_fingerprint
    )
    evidence_destination = (
        incomplete
        / "artifacts"
        / "g3-evidence"
        / "sha256"
        / completed_g3.evidence_fingerprint
    )
    evidence_destination.parent.mkdir(parents=True)
    shutil.copytree(evidence_source, evidence_destination)
    with pytest.raises(IntegrityError, match="run directory is missing"):
        handoff_completed_g3(
            incomplete,
            tmp_path / "incomplete-output",
            evidence_fingerprint=completed_g3.evidence_fingerprint,
        )
    assert not (tmp_path / "incomplete-output").exists()


def test_handoff_refuses_readdressed_nonpassing_evidence(
    tmp_path: Path,
    completed_g3: _Fixture,
) -> None:
    source, fingerprint = _counterfeit_manifest(
        completed_g3,
        tmp_path / "nonpassing",
        lambda value: value["identity"].__setitem__("passed", False),
    )
    source_root = source.artifact_root.parent
    assert (source.store_root / fingerprint / G3_EVIDENCE_MANIFEST_NAME).is_file()

    with pytest.raises(IntegrityError, match="contract identity is unsupported"):
        handoff_completed_g3(
            source_root,
            tmp_path / "nonpassing-output",
            evidence_fingerprint=fingerprint,
        )
    assert not (tmp_path / "nonpassing-output").exists()


def test_handoff_never_accepts_a_hardlink_copy(
    tmp_path: Path,
    completed_g3: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prpg.model.g3_artifact_handoff as module

    source = completed_g3.evidence_store.artifact_root.parent
    destination = tmp_path / "hardlinked-output"
    monkeypatch.setattr(module.shutil, "copyfile", os.link)

    with pytest.raises(IntegrityError, match="private regular storage"):
        handoff_completed_g3(
            source,
            destination,
            evidence_fingerprint=completed_g3.evidence_fingerprint,
        )
    assert not destination.exists()
