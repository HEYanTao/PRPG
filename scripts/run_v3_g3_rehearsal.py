"""Run and archive the sole version-3 noncanonical G3 rehearsal.

This is a thin command wrapper around the code-owned rehearsal.  It cannot
create a canonical launch, model artifact, G3 evidence, or generation
authority.  A real file and ``__main__`` guard are required by macOS's
``spawn`` multiprocessing mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prpg.config import load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.model.g3_preflight import CANONICAL_PROCESSED_DATA_FINGERPRINT
from prpg.model.g3_rehearsal import (
    run_reduced_g3_rehearsal,
    verify_noncanonical_g3_rehearsal_report,
)
from prpg.model.input import load_calibration_input
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed version-3 noncanonical G3 rehearsal once."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical.yaml"),
        help="Configuration path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON evidence path; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help=(
            "Existing separate runtime directory for relative artifacts and "
            "output; the hashed launcher itself continues to execute in place."
        ),
    )
    return parser.parse_args()


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_new_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"STOP: rehearsal output already exists: {path}")
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    try:
        with pending.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, path)
        except FileExistsError as error:
            raise SystemExit(
                f"STOP: rehearsal output appeared during execution: {path}"
            ) from error
    finally:
        pending.unlink(missing_ok=True)


def main() -> int:
    args = _arguments()
    runtime_root = (
        ROOT
        if args.runtime_root is None
        else args.runtime_root.expanduser().resolve(strict=False)
    )
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise SystemExit(f"STOP: runtime root is missing or unsafe: {runtime_root}")
    config_path = _rooted(args.config, ROOT).resolve()
    output_path = _rooted(args.output, runtime_root).resolve()
    os.chdir(runtime_root)

    config = load_config(config_path)
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(config.data.artifact_root),
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=config.fingerprint(),
        holdout_months=config.model.design_holdout_months,
    )
    report = run_reduced_g3_rehearsal(config, calibration_input)
    verify_noncanonical_g3_rehearsal_report(report)
    encoded = report.to_json_bytes()
    _write_new_atomic(output_path, encoded)

    projection = report.resource_projection
    if projection is None:
        raise SystemExit("STOP: rehearsal report omitted the resource projection")
    summary = {
        "event": "g3_noncanonical_rehearsal_completed",
        "report_path": str(output_path),
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "report_fingerprint": report.report_fingerprint,
        "execution_closure_schema": G3_EXECUTION_CLOSURE_SCHEMA,
        "execution_closure_manifest_fingerprint": (
            G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
        ),
        "execution_closure_fingerprint": report.source_code_fingerprint,
        "task_count": len(report.tasks),
        "selection_policy": report.selection_policy,
        "selected_n_states": report.selected_n_states,
        "scientific_version": report.scientific_version,
        "worker_parity_families": list(report.worker_parity_families),
        "workers": projection.workers,
        "projection_version": projection.projection_version,
        "projection_passed": projection.passed,
        "projection_reasons": list(projection.reasons),
        "projected_total_seconds": projection.projected_total_seconds,
        "maximum_wall_seconds": projection.maximum_wall_seconds,
        "projected_memory_fraction": projection.projected_memory_fraction,
        "maximum_memory_fraction": projection.maximum_memory_fraction,
        "disk_free_bytes": projection.disk_free_bytes,
        "required_free_disk_bytes": projection.required_free_disk_bytes,
        "native_thread_cap_verified": projection.native_thread_cap_verified,
        "worker_parity_verified": projection.worker_parity_verified,
        "g3_passed": report.g3_passed,
        "canonical_authority": report.canonical_authority,
        "generation_authorized": report.generation_authorized,
        "releasable": report.releasable,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if projection.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
