"""Dedicated, phase-scoped command line for canonical G3.

This module is the only supported production composition root for the G3
preflight and one-shot launch.  It deliberately imports no G4--G8 generator,
validation, qualification, or release module, allowing G3 provenance to be
frozen independently while downstream implementation continues.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prpg.config import load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.errors import PRPGError
from prpg.execution import resource_snapshot
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_evidence_store import G3EvidenceStore
from prpg.model.g3_preflight import (
    CANONICAL_PROCESSED_DATA_FINGERPRINT,
    canonical_g3_preflight,
    persist_canonical_g3_preflight,
)
from prpg.model.g3_success_execution import run_canonical_g3_success_execution
from prpg.model.input import load_calibration_input
from prpg.model.run_store import CalibrationRunStore, default_run_identity
from prpg.model.scientific_artifact_pair import ScientificArtifactPairStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or execute the dedicated canonical G3 path."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the G3 launch boundary without fitting or writing.",
    )
    mode.add_argument(
        "--canonical-launch",
        action="store_true",
        help="Execute one fresh, non-resumable canonical G3 launch.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical.yaml"),
        help="Approved canonical YAML configuration.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable JSON event.",
    )
    return parser


def execute_g3_launch(*, config_path: Path, canonical_launch: bool) -> dict[str, Any]:
    """Execute the scoped preflight and optionally the one-shot G3 launch."""

    resolved = load_config(config_path)
    config_fingerprint = resolved.fingerprint()
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(resolved.data.artifact_root),
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=config_fingerprint,
        holdout_months=resolved.model.design_holdout_months,
    )
    resources = resource_snapshot(
        resolved.run.output_root,
        requested_workers=resolved.execution.workers,
        reserve_cores=resolved.execution.reserve_physical_cores,
    )
    preflight = canonical_g3_preflight(resolved, calibration_input, resources)
    if not canonical_launch:
        return {
            "event": "g3_preflight_passed",
            "preflight_fingerprint": preflight.fingerprint,
            "config_fingerprint": preflight.config_fingerprint,
            "processed_data_fingerprint": preflight.processed_data_fingerprint,
            "calibration_input_fingerprint": preflight.calibration_input_fingerprint,
            "source_code_fingerprint": preflight.source_code_fingerprint,
            "dependency_lock_fingerprint": preflight.dependency_lock_fingerprint,
            "toolchain_fingerprint": preflight.toolchain_fingerprint,
            "workers": preflight.workers,
            "estimated_peak_bytes": preflight.estimated_peak_bytes,
            "physical_memory_bytes": preflight.physical_memory_bytes,
            "estimated_memory_fraction": (
                preflight.estimated_peak_bytes / preflight.physical_memory_bytes
            ),
            "estimated_disk_bytes": preflight.estimated_disk_bytes,
            "disk_free_bytes": preflight.disk_free_bytes,
            "checks": list(preflight.checks),
            "launch_created": False,
            "scientific_result_observed": False,
        }

    run_store = CalibrationRunStore(resolved.run.output_root)
    checkpoint_store = CalibrationCheckpointStore(resolved.data.artifact_root)
    model_store = ModelArtifactStore(resolved.data.artifact_root)
    pair_store = ScientificArtifactPairStore(resolved.data.artifact_root)
    evidence_store = G3EvidenceStore(resolved.data.artifact_root)
    reference = run_store.initialize(
        default_run_identity(
            config_fingerprint=preflight.config_fingerprint,
            processed_data_fingerprint=preflight.processed_data_fingerprint,
            calibration_input_fingerprint=preflight.calibration_input_fingerprint,
            source_code_fingerprint=preflight.source_code_fingerprint,
            dependency_lock_fingerprint=preflight.dependency_lock_fingerprint,
            workers=preflight.workers,
        )
    )
    preflight_receipt = persist_canonical_g3_preflight(
        run_store,
        run_id=reference.run_id,
        preflight=preflight,
    )
    run_store.plan_launch(
        reference.run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=preflight.workers,
    )
    with run_store.start_launch(
        reference.run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=preflight.workers,
        resume=False,
    ) as lease:
        result = run_canonical_g3_success_execution(
            calibration_input,
            preflight=preflight,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            lease=lease,
            master_seed=resolved.run.master_seed,
            scientific_version=resolved.simulation.scientific_version,
            model_store=model_store,
            pair_store=pair_store,
            evidence_store=evidence_store,
        )
    finalization = result.finalization
    artifact_pair = finalization.artifact_pair
    durable_evidence = finalization.durable_evidence
    return {
        "event": "g3_canonical_launch_completed",
        "run_id": reference.run_id,
        "run_path": str(reference.path),
        "preflight_fingerprint": preflight.fingerprint,
        "preflight_receipt_fingerprint": preflight_receipt.fingerprint,
        "completion_fingerprint": finalization.completion_marker.fingerprint,
        "workers": preflight.workers,
        "task_count": len(result.publications),
        "selected_n_states": artifact_pair.selected_k,
        "artifact_pair_fingerprint": artifact_pair.reference.fingerprint,
        "artifact_pair_path": str(artifact_pair.reference.path),
        "design_artifact_fingerprint": artifact_pair.design.reference.fingerprint,
        "design_artifact_path": str(artifact_pair.design.reference.path),
        "production_artifact_fingerprint": (
            artifact_pair.production.reference.fingerprint
        ),
        "production_artifact_path": str(artifact_pair.production.reference.path),
        "g3_evidence_fingerprint": durable_evidence.reference.fingerprint,
        "g3_evidence_path": str(durable_evidence.reference.path),
        "generation_authorized": result.generation_authorized,
    }


def _emit(value: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)
        return
    if value.get("event") == "error":
        print(f"ERROR [{value['exit_code']}]: {value['message']}", flush=True)
        details = value.get("details")
        if details:
            print(json.dumps(details, indent=2, sort_keys=True), flush=True)
        return
    for key, item in value.items():
        print(f"{key}: {item}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dedicated G3 CLI and return a stable process exit code."""

    arguments = _parser().parse_args(argv)
    try:
        event = execute_g3_launch(
            config_path=arguments.config,
            canonical_launch=arguments.canonical_launch,
        )
    except PRPGError as error:
        _emit(error.as_event(), json_output=arguments.json)
        return int(error.exit_code)
    _emit(event, json_output=arguments.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
