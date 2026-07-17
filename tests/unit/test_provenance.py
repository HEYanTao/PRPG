from __future__ import annotations

import os
from pathlib import Path

import pytest

import prpg.provenance as provenance
from prpg.errors import IntegrityError, ModelError
from prpg.provenance import (
    EXECUTABLE_SOURCE_SCHEMA,
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
    inspect_executable_source,
    inspect_g3_execution_closure,
    parse_dependency_lock_versions,
    verify_executable_source_identity,
    verify_g3_execution_closure_identity,
)

ROOT = Path(__file__).parents[2]


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src" / "prpg" / "model").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / ".python-version").write_text("3.11.15\n")
    (root / "pyproject.toml").write_text("[project]\nname='prpg'\n")
    (root / "configs" / "seed-registry.yaml").write_text("version: 1\n")
    (root / "requirements.lock").write_text("numpy==2.3.5\n")
    (root / "src" / "prpg" / "__init__.py").write_text("VERSION = 1\n")
    (root / "src" / "prpg" / "model" / "core.py").write_text("VALUE = 1\n")
    return root


def _g3_tree(tmp_path: Path) -> Path:
    root = tmp_path / "g3-project"
    manifest = G3_EXECUTION_CLOSURE_MANIFEST
    for relative_path in (
        *manifest.module_paths,
        *manifest.resource_static_paths,
        manifest.dependency_lock_path,
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "numpy==2.3.5\n"
            if relative_path == manifest.dependency_lock_path
            else f"frozen:{relative_path}\n"
        )
        path.write_text(content)
    return root


def test_source_and_lock_have_separate_deterministic_fingerprints(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    first = inspect_executable_source(root)
    second = inspect_executable_source(root)

    assert first == second
    assert first.schema_id == EXECUTABLE_SOURCE_SCHEMA
    assert tuple(item.relative_path for item in first.source_files) == tuple(
        sorted(item.relative_path for item in first.source_files)
    )
    assert "requirements.lock" not in {
        item.relative_path for item in first.source_files
    }
    assert len(first.source_code_fingerprint) == 64
    assert len(first.dependency_lock_fingerprint) == 64
    verify_executable_source_identity(first, project_root=root)

    (root / "src" / "prpg" / "model" / "core.py").write_text("VALUE = 2\n")
    source_changed = inspect_executable_source(root)
    assert source_changed.source_code_fingerprint != first.source_code_fingerprint
    assert (
        source_changed.dependency_lock_fingerprint == first.dependency_lock_fingerprint
    )
    with pytest.raises(IntegrityError, match="changed"):
        verify_executable_source_identity(first, project_root=root)

    (root / "requirements.lock").write_text("numpy==2.3.6\n")
    lock_changed = inspect_executable_source(root)
    assert (
        lock_changed.source_code_fingerprint == source_changed.source_code_fingerprint
    )
    assert (
        lock_changed.dependency_lock_fingerprint
        != source_changed.dependency_lock_fingerprint
    )


def test_caches_are_excluded_but_new_package_bytes_are_identity_bearing(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    first = inspect_executable_source(root)
    cache = root / "src" / "prpg" / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"cache")
    assert inspect_executable_source(root) == first

    (root / "src" / "prpg" / "contract.json").write_text('{"v":1}\n')
    assert inspect_executable_source(root).source_code_fingerprint != (
        first.source_code_fingerprint
    )


def test_source_symlink_and_hardlink_fail_closed(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    target = root / "outside.py"
    target.write_text("VALUE = 3\n")
    (root / "src" / "prpg" / "linked.py").symlink_to(target)
    with pytest.raises(IntegrityError, match="symbolic link"):
        inspect_executable_source(root)
    (root / "src" / "prpg" / "linked.py").unlink()

    os.link(target, root / "src" / "prpg" / "hardlinked.py")
    with pytest.raises(IntegrityError, match="private regular"):
        inspect_executable_source(root)


def test_missing_static_input_and_wrong_verifier_type_fail_closed(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    (root / "configs" / "seed-registry.yaml").unlink()
    with pytest.raises(IntegrityError, match="missing"):
        inspect_executable_source(root)
    with pytest.raises(ModelError, match="wrong type"):
        verify_executable_source_identity(object(), project_root=root)  # type: ignore[arg-type]


def test_live_workspace_source_identity_is_read_only_and_complete() -> None:
    identity = inspect_executable_source(ROOT)
    paths = {item.relative_path for item in identity.source_files}

    assert "src/prpg/cli.py" in paths
    assert "src/prpg/model/g3_preflight.py" in paths
    assert "src/prpg/vendor/patton_politis_white/LICENSE" in paths
    assert "pyproject.toml" in paths
    assert "configs/seed-registry.yaml" in paths
    assert all("tests/" not in path for path in paths)
    assert all("__pycache__" not in path for path in paths)


def test_installed_package_falls_back_to_complete_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path)
    installed = tmp_path / "venv/lib/python3.11/site-packages/prpg/provenance.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("# installed wheel location\n")
    monkeypatch.setattr(provenance, "__file__", str(installed))
    monkeypatch.chdir(root)

    assert provenance.inspect_dependency_lock_versions() == (("numpy", "2.3.5"),)


def test_g3_closure_uses_and_persists_the_exact_ordered_manifest(
    tmp_path: Path,
) -> None:
    root = _g3_tree(tmp_path)
    identity = inspect_g3_execution_closure(root)

    assert identity.schema_id == G3_EXECUTION_CLOSURE_SCHEMA
    assert identity.manifest == G3_EXECUTION_CLOSURE_MANIFEST
    assert identity.manifest_fingerprint == (
        "f5e0afa8e444b6a4aefaf4bbc33bd022926ee8c2865121a8b7c6e3da1253e436"
    )
    assert identity.manifest_fingerprint == G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
    assert identity.dependency_versions == (("numpy", "2.3.5"),)
    assert identity.source_identity_dict()["manifest"] == (
        G3_EXECUTION_CLOSURE_MANIFEST.as_dict()
    )
    assert tuple(item.relative_path for item in identity.source_files) == (
        G3_EXECUTION_CLOSURE_MANIFEST.module_paths
        + G3_EXECUTION_CLOSURE_MANIFEST.resource_static_paths
    )
    verify_g3_execution_closure_identity(identity, project_root=root)


def test_downstream_only_mutation_does_not_change_g3_closure(
    tmp_path: Path,
) -> None:
    root = _g3_tree(tmp_path)
    first = inspect_g3_execution_closure(root)
    downstream = root / "src/prpg/validation/release.py"
    downstream.parent.mkdir(parents=True)
    downstream.write_text("RELEASE_VERSION = 1\n")
    assert inspect_g3_execution_closure(root) == first
    downstream.write_text("RELEASE_VERSION = 2\n")
    assert inspect_g3_execution_closure(root) == first


@pytest.mark.parametrize(
    "relative_path",
    (
        "src/prpg/model/hmm.py",
        "configs/seed-registry.yaml",
        "src/prpg/vendor/patton_politis_white/opt_block_length_REV_dec07.txt",
        "scripts/run_v3_g3_rehearsal.py",
        "requirements.lock",
    ),
)
def test_g3_code_resource_launcher_and_lock_mutations_change_identity(
    tmp_path: Path,
    relative_path: str,
) -> None:
    root = _g3_tree(tmp_path)
    first = inspect_g3_execution_closure(root)
    changed_content = (
        "numpy==2.3.6\n"
        if relative_path == G3_EXECUTION_CLOSURE_MANIFEST.dependency_lock_path
        else f"changed:{relative_path}\n"
    )
    (root / relative_path).write_text(changed_content)
    changed = inspect_g3_execution_closure(root)

    assert changed != first
    if relative_path == G3_EXECUTION_CLOSURE_MANIFEST.dependency_lock_path:
        assert changed.source_code_fingerprint == first.source_code_fingerprint
        assert changed.dependency_lock_fingerprint != first.dependency_lock_fingerprint
    else:
        assert changed.source_code_fingerprint != first.source_code_fingerprint
        assert changed.dependency_lock_fingerprint == first.dependency_lock_fingerprint


def test_dependency_lock_parser_accepts_complete_exact_inventory() -> None:
    versions = parse_dependency_lock_versions((ROOT / "requirements.lock").read_bytes())

    assert len(versions) == 67
    assert ("joblib", "1.5.3") in versions
    assert ("python-dateutil", "2.9.0.post0") in versions
    assert ("pytz", "2026.2") in versions


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"numpy>=2.3.5\n", "exact name==version"),
        (b"numpy\n", "exact name==version"),
        (b"numpy==2.3.5 ; python_version > '3'\n", "exact name==version"),
        (b"numpy==2.3.5\nNumPy==2.3.5\n", "duplicate distribution"),
        (b"# comments only\n", "no exact pins"),
        (b"numpy==2.3.5\xff\n", "not UTF-8"),
    ),
)
def test_dependency_lock_parser_fails_closed(content: bytes, message: str) -> None:
    with pytest.raises(IntegrityError, match=message):
        parse_dependency_lock_versions(content)
