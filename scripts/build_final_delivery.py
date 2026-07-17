#!/usr/bin/env python3
"""Assemble the standalone PRPG v5 final-delivery directory.

The script deliberately copies an explicit allowlist.  It includes the current
application and tests, only the canonical v5 research artifacts, and only the
latest v5 production return bundle.  Development runs, superseded artifacts,
caches, the active virtual environment, and the source repository's Git
metadata are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RAW_FINGERPRINT = "d791b153b3e72fb7325e1d24f43da63a9a4da4c36d35c13b0d0338c4edfc579b"
PROCESSED_FINGERPRINT = (
    "e2caa7756ec5f6b37806e6ef58e44fe827f741ed544e82b132a0cd2f1bbe1760"
)
MODEL_FINGERPRINT = "be5df93d382a499dbf21c00b1e0be9d47bfda08bd387d0ac70b36d47ba88d6c7"
RUN_NAME = "v5-production-5000-20260716"
CANONICAL_GENERATOR_COMMIT = "07806dfba940be4f19f9d783e989f1c5354ea123"


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"required source file is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {".DS_Store", "__pycache__", ".pytest_cache", ".mypy_cache"}
        or name.endswith((".pyc", ".pyo"))
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"required source directory is missing or unsafe: {source}")
    shutil.copytree(source, destination, symlinks=False, ignore=_ignore)


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("source Git commit is invalid")
    return commit


def _require_clean_source(root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.stdout:
        raise RuntimeError(
            "source worktree must be clean so the recorded commit matches delivery"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"delivery contains a forbidden symlink: {path}")
        if path.is_file() and path.name != "DELIVERY-CHECKSUMS.sha256":
            yield path


def _inventory(root: Path) -> dict[str, int]:
    files = list(_files(root))
    return {
        "content_files_excluding_manifest_and_checksum": len(files),
        "content_bytes_excluding_manifest_and_checksum": sum(
            path.stat().st_size for path in files
        ),
    }


def _manifest(
    root: Path, *, source_commit: str, application_version: str
) -> dict[str, Any]:
    run = root / "runs" / "v5" / RUN_NAME
    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (run / "validation-report.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (run / "diagnostics-report.json").read_text(encoding="utf-8")
    )
    return {
        "schema_id": "prpg-final-delivery-manifest-v1",
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "application_version": application_version,
        "delivery_source_commit": source_commit,
        "canonical_generator_commit": CANONICAL_GENERATOR_COMMIT,
        "platform_target": "CPython 3.11 / macOS arm64",
        "contents": {
            "complete_source": "src/prpg",
            "tests": "tests",
            "configuration": "configs",
            "documentation": "doc",
            "onboarding": "ONBOARDING.md",
            "offline_dependencies": "wheelhouse",
            "application_wheel": "dist",
            "canonical_run": f"runs/v5/{RUN_NAME}",
            "raw_data_fingerprint": RAW_FINGERPRINT,
            "processed_data_fingerprint": PROCESSED_FINGERPRINT,
            "model_fingerprint": MODEL_FINGERPRINT,
        },
        "canonical_run": {
            "run_fingerprint": run_manifest["run_fingerprint"],
            "paths": validation["paths"],
            "csv_files": validation["csv_files"],
            "rows": validation["rows"],
            "bytes": validation["bytes"],
            "hard_validation_status": validation["status"],
            "diagnostics_status": diagnostics["status"],
            "consumer_checksum_list_sha256": _sha256(run / "checksums.sha256"),
            "manifest_sha256": _sha256(run / "manifest.json"),
            "validation_report_sha256": _sha256(run / "validation-report.json"),
            "diagnostics_report_sha256": _sha256(run / "diagnostics-report.json"),
        },
        "scope_notes": [
            "Only the latest canonical v5 return run is included.",
            "Superseded v5 processed artifacts and development runs are excluded.",
            "The wheelhouse makes installation offline on the stated platform target.",
            "Provider data and generated paths are for private local research use.",
        ],
    }


def _write_checksums(root: Path) -> None:
    rows = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in _files(root)
    ]
    (root / "DELIVERY-CHECKSUMS.sha256").write_text("".join(rows), encoding="utf-8")


def assemble(source_root: Path, output_root: Path) -> None:
    source = source_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"delivery output must be fresh: {output}")
    _require_clean_source(source)
    source_commit = _git_commit(source)
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    application_version = str(project["project"]["version"])
    output.mkdir(parents=True, mode=0o700)

    for directory in ("src", "tests", "configs", "scripts", "doc"):
        _copy_tree(source / directory, output / directory)
    for filename in (
        ".gitignore",
        ".python-version",
        "README.md",
        "CHANGELOG.md",
        "ONBOARDING.md",
        "LICENSE.txt",
        "pyproject.toml",
        "requirements.lock",
    ):
        _copy_file(source / filename, output / filename)
    _copy_tree(source / "reports", output / "reports")
    _copy_tree(source / "evidence" / "v5", output / "evidence" / "v5")
    _copy_tree(source / "wheelhouse", output / "wheelhouse")
    _copy_tree(source / "dist", output / "dist")

    selected_artifacts = (
        (
            source / "artifacts" / "v5" / "data" / "sha256" / RAW_FINGERPRINT,
            output / "artifacts" / "v5" / "data" / "sha256" / RAW_FINGERPRINT,
        ),
        (
            source
            / "artifacts"
            / "v5"
            / "data"
            / "processed"
            / "sha256"
            / PROCESSED_FINGERPRINT,
            output
            / "artifacts"
            / "v5"
            / "data"
            / "processed"
            / "sha256"
            / PROCESSED_FINGERPRINT,
        ),
        (
            source / "artifacts" / "v5" / "model" / "sha256" / MODEL_FINGERPRINT,
            output / "artifacts" / "v5" / "model" / "sha256" / MODEL_FINGERPRINT,
        ),
    )
    for artifact_source, artifact_destination in selected_artifacts:
        _copy_tree(artifact_source, artifact_destination)
    _copy_tree(
        source / "runs" / "v5" / RUN_NAME,
        output / "runs" / "v5" / RUN_NAME,
    )

    manifest = _manifest(
        output,
        source_commit=source_commit,
        application_version=application_version,
    )
    manifest["inventory"] = _inventory(output)
    (output / "DELIVERY-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("delivery/PRPG-v5-final-20260717"),
    )
    arguments = parser.parse_args(argv)
    assemble(arguments.source_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
