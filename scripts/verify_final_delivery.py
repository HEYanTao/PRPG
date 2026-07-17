#!/usr/bin/env python3
"""Verify PRPG final-delivery completeness, isolation, and file integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

RUN_NAME = "v5-production-5000-20260716"
RAW_FINGERPRINT = "d791b153b3e72fb7325e1d24f43da63a9a4da4c36d35c13b0d0338c4edfc579b"
PROCESSED_FINGERPRINT = (
    "e2caa7756ec5f6b37806e6ef58e44fe827f741ed544e82b132a0cd2f1bbe1760"
)
MODEL_FINGERPRINT = "be5df93d382a499dbf21c00b1e0be9d47bfda08bd387d0ac70b36d47ba88d6c7"
SUPERSEDED_PROCESSED = (
    "dad2ce8acbacfcfb0164b4cd88b3941192cd9eba3b6df90671c02fcd3cbc2f00"
)
LOCAL_PATH_PATTERN = re.compile(
    r"/(?:Users|home)/[^\s'\"`]+|/(?:private/tmp|var/folders)/[^\s'\"`]+"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _delivery_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"forbidden symlink: {relative}")
        if path.is_file() and path.name != "DELIVERY-CHECKSUMS.sha256":
            yield path


def _required(root: Path) -> None:
    paths = (
        "README.md",
        "ONBOARDING.md",
        "LICENSE.txt",
        "DELIVERY-MANIFEST.json",
        "DELIVERY-CHECKSUMS.sha256",
        "pyproject.toml",
        "requirements.lock",
        "src/prpg/v5/config.py",
        "src/prpg/v5/data.py",
        "src/prpg/v5/hmm.py",
        "src/prpg/v5/generation.py",
        "src/prpg/v5/validation.py",
        "doc/PRPG-plan.md",
        "doc/PRPG-technical-design-and-implementation-log.md",
        "doc/PRPG-v5-historical-vs-simulated-summary.md",
        "reports/v5-historical-vs-simulated-summary.json",
        f"runs/v5/{RUN_NAME}/manifest.json",
        f"runs/v5/{RUN_NAME}/checksums.sha256",
        f"artifacts/v5/data/sha256/{RAW_FINGERPRINT}/snapshot-manifest.json",
        (
            "artifacts/v5/data/processed/sha256/"
            f"{PROCESSED_FINGERPRINT}/processed-manifest.json"
        ),
        f"artifacts/v5/model/sha256/{MODEL_FINGERPRINT}/model-manifest.json",
    )
    missing = [relative for relative in paths if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"delivery is missing required files: {missing}")
    if (root / ".venv").exists():
        raise RuntimeError("delivery must not contain a nonrelocatable virtualenv")
    if (root / "artifacts/v5/data/processed/sha256" / SUPERSEDED_PROCESSED).exists():
        raise RuntimeError("delivery contains the superseded processed artifact")
    runs = sorted(path.name for path in (root / "runs/v5").iterdir())
    if runs != [RUN_NAME]:
        raise RuntimeError(f"delivery contains unexpected v5 runs: {runs}")
    if not any((root / "dist").glob("prpg-*.whl")):
        raise RuntimeError("delivery application wheel is missing")
    if len(list((root / "wheelhouse").glob("*"))) < 60:
        raise RuntimeError("delivery offline wheelhouse is incomplete")


def _internal_references(root: Path) -> None:
    scan_roots = (root / "src", root / "configs", root / "scripts")
    extensions = {".py", ".yaml", ".yml", ".toml", ".sh"}
    violations: list[str] = []
    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix not in extensions:
                continue
            text = path.read_text(encoding="utf-8")
            match = LOCAL_PATH_PATTERN.search(text)
            if match:
                violations.append(f"{path.relative_to(root)}: {match.group(0)}")
    if violations:
        raise RuntimeError(
            f"executable material has external local paths: {violations}"
        )

    config = root / "configs/v5-canonical.yaml"
    values: dict[str, str] = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for key in ("artifact_root", "output_root", "seed_registry"):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                values[key] = stripped.removeprefix(prefix).strip()
    for key in ("artifact_root", "output_root"):
        target = (config.parent / values[key]).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"{key} escapes the delivery root: {target}")
    seed = (config.parent / values["seed_registry"]).resolve()
    if not seed.is_relative_to(root) or not seed.is_file():
        raise RuntimeError("seed registry is not internal to the delivery")


def _checksum_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or relative in entries:
            raise RuntimeError(f"invalid checksum-list row: {line!r}")
        entries[relative] = digest
    return entries


def _verify_delivery_hashes(root: Path) -> int:
    entries = _checksum_entries(root / "DELIVERY-CHECKSUMS.sha256")
    actual = {path.relative_to(root).as_posix(): path for path in _delivery_files(root)}
    if set(entries) != set(actual):
        missing = sorted(set(actual) - set(entries))
        extra = sorted(set(entries) - set(actual))
        raise RuntimeError(
            f"delivery checksum allowlist differs; missing={missing}, extra={extra}"
        )
    for relative, path in actual.items():
        if _sha256(path) != entries[relative]:
            raise RuntimeError(f"delivery checksum mismatch: {relative}")
    return len(entries)


def _verify_consumer_hashes(root: Path) -> int:
    run = root / "runs/v5" / RUN_NAME
    entries = _checksum_entries(run / "checksums.sha256")
    if len(entries) != 400:
        raise RuntimeError("canonical consumer checksum list must contain 400 rows")
    for relative, expected in entries.items():
        path = run / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"canonical consumer checksum mismatch: {relative}")
    return len(entries)


def verify(root: Path, *, hashes: bool) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    _required(resolved)
    _internal_references(resolved)
    files = list(_delivery_files(resolved))
    manifest = json.loads(
        (resolved / "DELIVERY-MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "complete":
        raise RuntimeError("delivery manifest is not complete")
    package_hashes = None
    consumer_hashes = None
    if hashes:
        package_hashes = _verify_delivery_hashes(resolved)
        consumer_hashes = _verify_consumer_hashes(resolved)
    return {
        "status": "passed",
        "delivery_root": str(resolved),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "external_local_references": 0,
        "symlinks": 0,
        "delivery_hashes_verified": package_hashes,
        "consumer_hashes_verified": consumer_hashes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Run structural/isolation checks without rehashing the 5.8 GB bundle.",
    )
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            verify(arguments.root, hashes=not arguments.skip_hashes),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
