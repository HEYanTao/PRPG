"""Command-line interface for the staged PRPG application."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from prpg import __version__
from prpg.config import PRPGConfig, load_config
from prpg.data.pipeline import acquire_raw_snapshot
from prpg.data.processed import ProcessedSnapshotStore, build_processed_snapshot
from prpg.errors import (
    ConfigurationError,
    ExitCode,
    GenerationError,
    NotImplementedStageError,
    PRPGError,
)
from prpg.execution import resource_snapshot
from prpg.model.artifact import ModelArtifactStore
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_evidence_store import G3EvidenceStore
from prpg.model.g3_preflight import (
    CANONICAL_PROCESSED_DATA_FINGERPRINT,
)
from prpg.model.input import load_calibration_input
from prpg.model.practical_k4 import (
    PracticalK4ArtifactStore,
    reproduce_owner_approved_k4,
)
from prpg.model.run_store import CalibrationRunStore
from prpg.simulation.g4_reference import run_g4_reference_fixture
from prpg.simulation.practical_production import (
    build_practical_production_loader,
    resolve_practical_workers,
    run_practical_production,
)
from prpg.simulation.production import (
    ProductionPlan,
    benchmark_production_worker_counts,
    build_production_binding,
    build_production_loader_spec,
    load_generated_production_run,
    production_resource_estimate,
    replay_representative_work_units,
    run_production_generation,
)
from prpg.v5.config import load_v5_inputs
from prpg.v5.data import acquire_v5_raw_snapshot, build_v5_month_library
from prpg.v5.diagnostics import run_v5_report_diagnostics
from prpg.v5.generation import V5GenerationProfile, run_v5_generation
from prpg.v5.hmm import fit_and_publish_v5_k4
from prpg.v5.validation import validate_v5_run
from prpg.validation.practical_delivery import validate_practical_delivery
from prpg.validation.release import (
    ReleaseEvidence,
    ReleaseFinalizationPlan,
    build_external_release_attestation,
    finalize_complete,
    publish_released,
    verify_released,
)

app = typer.Typer(
    name="prpg",
    help="Auditable plausible-return path generator.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Resolve and validate immutable configuration.")
data_app = typer.Typer(help="Acquire or validate immutable source data.")
validation_app = typer.Typer(help="Calibrate immutable validation thresholds.")
release_app = typer.Typer(
    help="Finalize and independently attest an authorized release.",
    invoke_without_command=True,
)
v5_app = typer.Typer(help="Delivery-version-5 calendar-month workflow.")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(validation_app, name="validation")
app.add_typer(release_app, name="release")
app.add_typer(v5_app, name="v5")

_EXPECTED_RUNTIME_VERSIONS = {
    "exchange-calendars": "4.13.2",
    "hmmlearn": "0.3.3",
    "httpx": "0.28.1",
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "pandas-datareader": "0.11.1",
    "psutil": "7.1.0",
    "pydantic": "2.12.4",
    "PyYAML": "6.0.3",
    "requests": "2.32.5",
    "scikit-learn": "1.7.2",
    "scipy": "1.16.3",
    "threadpoolctl": "3.6.0",
    "typer": "0.20.0",
    "yfinance": "1.5.1",
}


class _V5Profile(str, Enum):
    """Bounded run geometries exposed by the version-5 CLI."""

    smoke = "smoke"
    parity = "parity"
    pilot = "pilot"
    configured = "configured"
    canonical = "canonical"


def _v5_profile_value(profile: _V5Profile) -> V5GenerationProfile:
    return profile.value


def _v5_expected_paths(profile: _V5Profile, configured_paths: int) -> int:
    return {
        _V5Profile.smoke: 1,
        _V5Profile.parity: 90,
        _V5Profile.pilot: 500,
        _V5Profile.configured: configured_paths,
        _V5Profile.canonical: 5_000,
    }[profile]


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    if payload.get("event") == "error":
        typer.echo(f"ERROR [{payload['exit_code']}]: {payload['message']}", err=True)
        details = payload.get("details")
        if details:
            typer.echo(json.dumps(details, indent=2, sort_keys=True), err=True)
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def _exit_for_error(error: PRPGError, *, json_output: bool) -> None:
    _emit(error.as_event(), json_output=json_output)
    raise typer.Exit(code=int(error.exit_code))


def _not_implemented(command: str) -> None:
    error = NotImplementedStageError(
        f"{command} is not implemented in the current qualified phase",
        details={"command": command, "qualified_phase": 2},
    )
    _emit(error.as_event(), json_output=False)
    raise typer.Exit(code=int(error.exit_code))


def _resolve_practical_config_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except OSError as error:
        raise GenerationError("practical configuration path is invalid") from error


@v5_app.command("config")
def v5_config(
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Validate the prospective version-5 config and seed registry."""

    try:
        inputs = load_v5_inputs(path)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "v5_config_validated",
            "config_path": str(inputs.config_path),
            "config_fingerprint": inputs.config_fingerprint,
            "seed_registry_path": str(inputs.seed_registry_path),
            "seed_registry_fingerprint": inputs.seed_registry_fingerprint,
            "artifact_root": str(inputs.artifact_root),
        },
        json_output=json_output,
    )


@v5_app.command("acquire-data")
def v5_acquire_data(
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Acquire one joint immutable three-asset raw snapshot."""

    try:
        inputs = load_v5_inputs(path)
        reference = acquire_v5_raw_snapshot(inputs)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "v5_raw_data_acquired",
            "raw_snapshot_fingerprint": reference.fingerprint,
            "snapshot_path": str(reference.path),
            "reused": reference.reused,
        },
        json_output=json_output,
    )


@v5_app.command("prepare-data")
def v5_prepare_data(
    raw_snapshot_fingerprint: Annotated[
        str,
        typer.Option("--raw", help="Joint three-asset raw fingerprint."),
    ],
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Build and offline-verify the complete synchronized month library."""

    try:
        inputs = load_v5_inputs(path)
        reference = build_v5_month_library(inputs, raw_snapshot_fingerprint)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "v5_month_library_prepared",
            "raw_snapshot_fingerprint": raw_snapshot_fingerprint,
            "processed_data_fingerprint": reference.fingerprint,
            "processed_path": str(reference.path),
            "qc_report": dict(reference.manifest["qc_report"]),
            "reused": reference.reused,
        },
        json_output=json_output,
    )


@v5_app.command("fit-k4")
def v5_fit_k4(
    processed_data_fingerprint: Annotated[
        str,
        typer.Option("--processed", help="Verified v5 processed-data fingerprint."),
    ],
    source_commit: Annotated[
        str,
        typer.Option("--source-commit", help="Full immutable source commit SHA-1."),
    ],
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    review_directory: Annotated[
        Path,
        typer.Option("--review-dir", help="Owner-facing review output directory."),
    ] = Path("evidence/v5"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Fit the single registered K4 model and publish its owner-review packet."""

    try:
        inputs = load_v5_inputs(path)
        publication = fit_and_publish_v5_k4(
            inputs,
            processed_data_fingerprint,
            source_commit=source_commit,
            review_directory=review_directory,
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "v5_k4_owner_review_ready",
            "model_artifact_fingerprint": publication.reference.fingerprint,
            "model_artifact_path": str(publication.reference.path),
            "selected_restart_index": publication.selected_restart_index,
            "selected_seed": publication.selected_seed,
            "state_counts": list(publication.state_counts),
            "minimum_state_months_passed": publication.pool_floor_passed,
            "review_markdown_path": str(publication.review_markdown_path),
            "month_assignments_path": str(publication.month_assignments_path),
            "generation_authorized": False,
        },
        json_output=json_output,
    )


@v5_app.command("generate")
def v5_generate(
    model_fingerprint: Annotated[
        str,
        typer.Option("--model", help="Approved version-5 K4 model fingerprint."),
    ],
    processed_data_fingerprint: Annotated[
        str,
        typer.Option("--processed", help="Verified v5 processed-data fingerprint."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output", help="New exclusive version-5 run directory."),
    ],
    run_name: Annotated[
        str,
        typer.Option("--run-name", help="Human-readable immutable run name."),
    ],
    profile: Annotated[
        _V5Profile,
        typer.Option(
            "--profile",
            case_sensitive=False,
            help=("Run geometry: smoke, parity, pilot, configured, or canonical."),
        ),
    ],
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            min=1,
            help="Worker processes; omit to use execution.workers from config.",
        ),
    ] = None,
    path_generation_seed: Annotated[
        int | None,
        typer.Option(
            "--path-seed",
            min=0,
            help=(
                "Optional path RNG seed; reuses the fitted model and leaves its "
                "calibration seed unchanged."
            ),
        ),
    ] = None,
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Generate one deterministic four-frequency version-5 run."""

    try:
        inputs = load_v5_inputs(path)
        worker_count = inputs.config.execution.workers if workers is None else workers
        result = run_v5_generation(
            inputs,
            model_fingerprint=model_fingerprint,
            processed_data_fingerprint=processed_data_fingerprint,
            output_root=output_root,
            run_name=run_name,
            profile=_v5_profile_value(profile),
            workers=worker_count,
            path_generation_seed=path_generation_seed,
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    payload = dict(result.as_dict())
    payload["event"] = "v5_generation_completed"
    _emit(payload, json_output=json_output)


@v5_app.command("validate")
def v5_validate(
    run_root: Annotated[
        Path,
        typer.Argument(help="Completed version-5 run directory."),
    ],
    profile: Annotated[
        _V5Profile,
        typer.Option(
            "--profile",
            case_sensitive=False,
            help=("Expected smoke, parity, pilot, configured, or canonical geometry."),
        ),
    ],
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Stream-validate one version-5 run and emit its result."""

    try:
        inputs = load_v5_inputs(path)
        report = validate_v5_run(
            inputs,
            run_root,
            expected_paths=_v5_expected_paths(profile, inputs.config.simulation.paths),
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    payload = dict(report.as_dict())
    payload["event"] = "v5_run_validated"
    _emit(payload, json_output=json_output)


@v5_app.command("diagnose")
def v5_diagnose(
    run_root: Annotated[
        Path,
        typer.Argument(help="Completed canonical version-5 run directory."),
    ],
    path: Annotated[
        Path,
        typer.Option("--config", help="Version-5 YAML configuration."),
    ] = Path("configs/v5-canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Write bounded descriptive diagnostics; never make a release decision."""

    try:
        inputs = load_v5_inputs(path)
        result = run_v5_report_diagnostics(inputs, run_root)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    payload = dict(result.as_dict())
    payload["event"] = "v5_report_only_diagnostics_written"
    _emit(payload, json_output=json_output)


@app.callback()
def root(
    version_requested: bool = typer.Option(
        False,
        "--version",
        help="Print the application version and exit.",
        is_eager=True,
    ),
) -> None:
    """Auditable plausible-return path generator."""

    if version_requested:
        typer.echo(__version__)
        raise typer.Exit()


@config_app.command("validate")
def config_validate(
    path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            dir_okay=False,
            readable=False,
            help="YAML configuration to validate.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Validate and fingerprint a configuration without side effects."""

    try:
        resolved = load_config(path)
    except ConfigurationError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "config_validated",
            "schema_version": resolved.schema_version,
            "config_fingerprint": resolved.fingerprint(),
            "path": str(path),
        },
        json_output=json_output,
    )


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Inspect the local runtime without revealing credential values."""

    python_supported = sys.version_info[:2] == (3, 11)
    dependency_versions: dict[str, str | None] = {}
    for distribution in _EXPECTED_RUNTIME_VERSIONS:
        try:
            dependency_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            dependency_versions[distribution] = None
    dependency_versions_match = all(
        dependency_versions[name] == expected
        for name, expected in _EXPECTED_RUNTIME_VERSIONS.items()
    )
    environment_supported = python_supported and dependency_versions_match
    disk = shutil.disk_usage(Path.cwd())
    payload: dict[str, Any] = {
        "event": "doctor",
        "status": "ok" if environment_supported else "unsupported_environment",
        "python": platform.python_version(),
        "python_supported": python_supported,
        "platform": platform.system(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "disk_free_bytes": disk.free,
        "fred_api_key_present": bool(os.environ.get("FRED_API_KEY")),
        "dependency_versions": dependency_versions,
        "expected_dependency_versions": _EXPECTED_RUNTIME_VERSIONS,
        "dependency_versions_match": dependency_versions_match,
    }
    _emit(payload, json_output=json_output)
    if not environment_supported:
        raise typer.Exit(code=int(ExitCode.CONFIGURATION))


def _physical_memory_bytes() -> int | None:
    """Read physical RAM using portable sysconf names when available."""

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * page_count


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _resource_plan(config: PRPGConfig) -> dict[str, Any]:
    low = config.simulation.low_frequency
    high = config.simulation.high_frequency
    rows = {
        "low_monthly": low.paths * low.years * 12,
        "low_quarterly": low.paths * low.years * 4,
        "low_annual": low.paths * low.years,
        "high_daily": high.paths * high.years * 252,
        "high_weekly": high.paths * high.years * 52,
    }
    low_groups = _ceil_div(low.paths, config.export.low_paths_per_shard)
    high_groups = _ceil_div(high.paths, config.export.high_paths_per_shard)
    return {
        "event": "resource_plan",
        "config_fingerprint": config.fingerprint(),
        "rows": rows,
        "total_rows": sum(rows.values()),
        "return_values": 3 * sum(rows.values()),
        "csv_shards": {
            "low_monthly": low_groups,
            "low_quarterly": low_groups,
            "low_annual": low_groups,
            "high_daily": high_groups,
            "high_weekly": high_groups,
            "total": 3 * low_groups + 2 * high_groups,
        },
    }


@app.command("plan")
def plan(
    path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            dir_okay=False,
            readable=False,
            help="YAML configuration to plan.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Dry-run exact structural row, value, and CSV shard counts."""

    try:
        resolved = load_config(path)
    except ConfigurationError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(_resource_plan(resolved), json_output=json_output)


@data_app.command("fetch")
def data_fetch(
    path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            dir_okay=False,
            readable=False,
            help="Approved YAML configuration for the acquisition contract.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Fetch seven approved sources and create an immutable raw snapshot."""

    try:
        resolved = load_config(path)
        result = acquire_raw_snapshot(resolved)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "raw_snapshot_created",
            "config_fingerprint": resolved.fingerprint(),
            "snapshot_fingerprint": result.snapshot.fingerprint,
            "snapshot_path": str(result.snapshot.path),
            "snapshot_reused": result.snapshot.reused,
            "acquisition_group_id": result.acquisition_group_id,
            "acquisition_journal_path": str(result.journal_path),
            "source_count": result.source_count,
        },
        json_output=json_output,
    )


@data_app.command("validate")
def data_validate(
    raw_snapshot_fingerprint: Annotated[
        str,
        typer.Argument(help="Exact immutable raw snapshot SHA-256 fingerprint."),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Approved YAML configuration for transformations and gates.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Build and verify the processed snapshot entirely offline."""

    try:
        resolved = load_config(path)
        reference = build_processed_snapshot(resolved, raw_snapshot_fingerprint)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    qc = reference.manifest["qc_report"]
    _emit(
        {
            "event": "processed_snapshot_validated",
            "config_fingerprint": resolved.fingerprint(),
            "raw_snapshot_fingerprint": raw_snapshot_fingerprint,
            "data_fingerprint": reference.fingerprint,
            "processed_snapshot_path": str(reference.path),
            "snapshot_reused": reference.reused,
            "monthly_vectors": qc["aligned"]["monthly_vectors"],
            "daily_vectors": qc["aligned"]["daily_vectors"],
            "first_month": qc["aligned"]["first_month"],
            "last_month": qc["aligned"]["last_month"],
            "status": qc["status"],
        },
        json_output=json_output,
    )


@app.command("calibrate")
def calibrate(
    preflight_only: bool = typer.Option(
        False,
        "--preflight-only",
        help="Verify the canonical G3 launch boundary without fitting or writing.",
    ),
    canonical_launch: bool = typer.Option(
        False,
        "--canonical-launch",
        help="Execute one fresh, non-resumable canonical G3 launch.",
    ),
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Approved canonical YAML configuration.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Reject the retired whole-package G3 composition root."""

    if preflight_only and canonical_launch:
        _exit_for_error(
            ConfigurationError(
                "calibration modes are mutually exclusive",
                details={
                    "modes": ["--preflight-only", "--canonical-launch"],
                },
            ),
            json_output=json_output,
        )
    if not preflight_only and not canonical_launch:
        _not_implemented("prpg calibrate")
    _exit_for_error(
        ConfigurationError(
            "G3 uses the dedicated execution-closure launcher",
            details={
                "required_command": "python -m prpg.g3_launch",
                "requested_mode": (
                    "canonical_launch" if canonical_launch else "preflight_only"
                ),
                "config": str(path),
            },
        ),
        json_output=json_output,
    )


@validation_app.command("calibrate")
def validation_calibrate() -> None:
    """Calibrate immutable null thresholds (Phase 5)."""

    _not_implemented("prpg validation calibrate")


@app.command("benchmark")
def benchmark(
    g3_evidence_fingerprint: Annotated[
        str,
        typer.Argument(help="Exact immutable passed G3 evidence SHA-256."),
    ],
    g5_authority_fingerprint: Annotated[
        str,
        typer.Argument(help="Exact immutable passed G5 authority SHA-256."),
    ],
    output_parent: Annotated[
        Path,
        typer.Option("--output", help="New parent for G6 worker-count runs."),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Approved canonical YAML configuration.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Benchmark workers, shards, and CSV throughput (Phase 6)."""

    try:
        project_root = Path(__file__).resolve().parents[2]
        config_path = path.expanduser().resolve(strict=False)
        resolved = load_config(config_path)
        loader = build_production_loader_spec(
            project_root=project_root,
            config_path=config_path,
            processed_data_fingerprint=CANONICAL_PROCESSED_DATA_FINGERPRINT,
            g3_evidence_fingerprint=g3_evidence_fingerprint,
            g5_authority_fingerprint=g5_authority_fingerprint,
        )
        benchmark_plan = ProductionPlan(
            profile_id="g6-benchmark-50y-v1",
            years=resolved.simulation.low_frequency.years,
            lf_paths=700,
            hf_paths=150,
            lf_paths_per_shard=resolved.export.low_paths_per_shard,
            hf_paths_per_shard=resolved.export.high_paths_per_shard,
        )
        binding = build_production_binding(
            loader,
            plan=benchmark_plan,
            run_name="g6-benchmark-base",
        )
        report = benchmark_production_worker_counts(
            loader=loader,
            plan=benchmark_plan,
            base_binding=binding,
            output_parent=output_parent.expanduser().resolve(strict=False),
            worker_counts=(1, 2, 4, 9),
            run_name_prefix="g6-benchmark",
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "g6_production_benchmark_completed",
            "report_fingerprint": report.fingerprint,
            "benchmark_plan_fingerprint": report.benchmark_plan_fingerprint,
            "benchmark_rows": benchmark_plan.total_rows,
            "worker_counts": list(report.worker_counts),
            "observations": [
                {
                    "workers": item.workers,
                    "wall_seconds": item.wall_seconds,
                    "rows_per_second": item.rows_per_second,
                    "parallel_efficiency": item.parallel_efficiency,
                    "peak_process_tree_rss_bytes": (item.peak_process_tree_rss_bytes),
                    "csv_parity_with_one_worker": (item.csv_parity_with_one_worker),
                    "run_root": str(item.run_root),
                }
                for item in report.observations
            ],
            "selected_workers": report.selected_workers,
            "selected_rows_per_second": report.selected_rows_per_second,
            "canonical_projected_wall_seconds": (
                report.canonical_projected_wall_seconds
            ),
            "canonical_projected_wall_hours": (report.canonical_projected_wall_hours),
            "all_csv_parity_passed": report.all_csv_parity_passed,
            "minimum_parallel_efficiency": report.minimum_parallel_efficiency,
            "benchmark_report_path": str(
                output_parent.expanduser().resolve(strict=False)
                / "benchmark-report.json"
            ),
        },
        json_output=json_output,
    )


@app.command("g4-reference")
def g4_reference(
    g3_evidence_fingerprint: Annotated[
        str,
        typer.Argument(help="Exact immutable passed G3 evidence SHA-256."),
    ],
    output_root: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New isolated directory for noncanonical reference CSVs/report.",
        ),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Approved canonical YAML configuration.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Create the bounded, explicitly noncanonical G4 reference fixture."""

    try:
        resolved = load_config(path)
        config_fingerprint = resolved.fingerprint()
        calibration_input = load_calibration_input(
            ProcessedSnapshotStore(resolved.data.artifact_root),
            CANONICAL_PROCESSED_DATA_FINGERPRINT,
            calibration_config_fingerprint=config_fingerprint,
            holdout_months=resolved.model.design_holdout_months,
        )
        run_store = CalibrationRunStore(resolved.run.output_root)
        checkpoint_store = CalibrationCheckpointStore(resolved.data.artifact_root)
        model_store = ModelArtifactStore(resolved.data.artifact_root)
        evidence = G3EvidenceStore(resolved.data.artifact_root).load(
            g3_evidence_fingerprint,
            run_store=run_store,
            checkpoint_store=checkpoint_store,
            model_store=model_store,
        )
        report = run_g4_reference_fixture(
            g3_evidence=evidence,
            config=resolved,
            calibration_input=calibration_input,
            output_root=output_root.expanduser().resolve(strict=False),
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    fixture = report.identity["fixture"]
    assert isinstance(fixture, dict)
    _emit(
        {
            "event": "g4_reference_completed_noncanonical",
            "report_fingerprint": report.fingerprint,
            "report_sha256": report.sha256,
            "report_path": str(report.path),
            "g3_evidence_fingerprint": g3_evidence_fingerprint,
            "structural_validation_passed": (report.structural_validation.passed),
            "paths": fixture["base_paths"],
            "rows": fixture["rows"],
            "files": fixture["files"],
            "g4_passed": report.g4_passed,
            "canonical_authority": report.canonical_authority,
            "generation_authorized": report.generation_authorized,
            "releasable": report.releasable,
        },
        json_output=json_output,
    )


@app.command("generate")
def generate(
    g3_evidence_fingerprint: Annotated[
        str,
        typer.Argument(help="Exact immutable passed G3 evidence SHA-256."),
    ],
    g5_authority_fingerprint: Annotated[
        str,
        typer.Argument(help="Exact immutable passed G5 authority SHA-256."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output", help="New or resumable production run root."),
    ],
    run_name: str = typer.Option(
        "canonical-prpg-v1",
        "--run-name",
        help="Stable content identity for this production run.",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Worker count; omit to use the configured auto policy.",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume an identical run."),
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Approved canonical YAML configuration.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Generate staging shards from frozen artifacts (Phase 6)."""

    try:
        project_root = Path(__file__).resolve().parents[2]
        config_path = path.expanduser().resolve(strict=False)
        resolved = load_config(config_path)
        output = output_root.expanduser().resolve(strict=False)
        loader = build_production_loader_spec(
            project_root=project_root,
            config_path=config_path,
            processed_data_fingerprint=CANONICAL_PROCESSED_DATA_FINGERPRINT,
            g3_evidence_fingerprint=g3_evidence_fingerprint,
            g5_authority_fingerprint=g5_authority_fingerprint,
        )
        plan = ProductionPlan.from_config(resolved)
        binding = build_production_binding(loader, plan=plan, run_name=run_name)
        resources = resource_snapshot(
            output,
            requested_workers=(
                resolved.execution.workers if workers is None else workers
            ),
            reserve_cores=resolved.execution.reserve_physical_cores,
        )
        estimate = production_resource_estimate(
            plan=plan,
            config=resolved,
            output_root=output,
            requested_workers=resources.resolved_workers,
        )
        result = run_production_generation(
            loader=loader,
            plan=plan,
            binding=binding,
            output_root=output,
            run_name=run_name,
            workers=resources.resolved_workers,
            resume=resume,
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "production_generation_completed",
            "run_root": str(result.run_root),
            "run_fingerprint": result.binding.run_fingerprint,
            "simulation_fingerprint": result.binding.simulation_fingerprint,
            "qualification_fingerprint": (result.binding.qualification_fingerprint),
            "materialization_fingerprint": (result.binding.materialization_fingerprint),
            "result_fingerprint": result.fingerprint,
            "workers": result.workers,
            "generated_work_units": result.generated_work_units,
            "reused_work_units": result.reused_work_units,
            "csv_shards": plan.csv_shard_count,
            "total_rows": plan.total_rows,
            "total_values": plan.total_values,
            "wall_seconds": result.wall_seconds,
            "rows_per_second": result.rows_per_second,
            "peak_process_tree_rss_bytes": result.peak_process_tree_rss_bytes,
            "estimated_peak_memory_bytes": estimate.estimated_peak_memory_bytes,
            "required_free_disk_bytes": estimate.required_free_disk_bytes,
            "generation_complete": True,
            "data_validated": False,
            "released": False,
        },
        json_output=json_output,
    )


@app.command("prepare-k4")
def prepare_k4(
    processed_data_fingerprint: str = typer.Option(
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        "--processed-data",
        help="Immutable processed-data SHA-256 used by the approved fit.",
    ),
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Configuration used to load the immutable calibration input.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Reproduce and persist the exact owner-approved practical K=4 artifact."""

    try:
        project_root = Path(__file__).resolve().parents[2]
        config_path = _resolve_practical_config_path(path)
        resolved = load_config(config_path)
        artifact_root = Path(resolved.data.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = project_root / artifact_root
        calibration_input = load_calibration_input(
            ProcessedSnapshotStore(artifact_root),
            processed_data_fingerprint,
            calibration_config_fingerprint=resolved.fingerprint(),
            holdout_months=resolved.model.design_holdout_months,
        )
        artifact = reproduce_owner_approved_k4(calibration_input)
        reference = PracticalK4ArtifactStore(artifact_root).persist(artifact)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "practical_k4_artifact_prepared",
            "artifact_fingerprint": artifact.fingerprint,
            "model_fingerprint": artifact.model_fingerprint,
            "artifact_directory": str(reference.directory),
            "state_counts": np.bincount(
                artifact.monthly_source_states, minlength=4
            ).tolist(),
            "iterations": artifact.iterations,
            "log_likelihood": artifact.log_likelihood,
        },
        json_output=json_output,
    )


@app.command("validate-k4")
def validate_k4(
    run_root: Annotated[
        Path,
        typer.Argument(help="Completed practical K=4 run directory."),
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Stream-validate schemas, counts, checksums, compounding, and uniqueness."""

    try:
        report = validate_practical_delivery(run_root)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "practical_k4_delivery_validated",
            "status": report.status,
            "profile_id": report.profile_id,
            "years": report.years,
            "lf_paths": report.lf_paths,
            "hf_paths": report.hf_paths,
            "work_units": report.work_units,
            "csv_shards": report.csv_shards,
            "total_rows": report.total_rows,
            "total_bytes": report.total_bytes,
            "checksums_verified": report.checksums_verified,
            "aggregation_paths_checked": report.aggregation_paths_checked,
            "aggregation_values_checked": report.aggregation_values_checked,
            "duplicate_base_paths": report.duplicate_base_paths,
            "manifest_sha256": report.manifest_sha256,
            "elapsed_seconds": report.elapsed_seconds,
            "rows_per_second": report.rows_per_second,
            "datasets": [
                {
                    "family": item.family,
                    "frequency": item.frequency,
                    "paths": item.paths,
                    "periods_per_path": item.periods_per_path,
                    "shards": item.shards,
                    "rows": item.rows,
                }
                for item in report.datasets
            ],
        },
        json_output=json_output,
    )


@app.command("generate-k4")
def generate_k4(
    artifact_fingerprint: Annotated[
        str,
        typer.Argument(help="Owner-approved practical K=4 artifact SHA-256."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output", help="New output directory for final CSV files."),
    ],
    processed_data_fingerprint: str = typer.Option(
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        "--processed-data",
        help="Immutable processed-data SHA-256 used by the artifact.",
    ),
    run_name: str = typer.Option(
        "practical-k4-delivery-v4",
        "--run-name",
        help="Stable identifier for this generation run.",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Worker count; default uses available capacity up to nine.",
    ),
    lf_paths: int | None = typer.Option(
        None,
        "--lf-paths",
        min=1,
        help="Optional LF path-count override within the selected config.",
    ),
    hf_paths: int | None = typer.Option(
        None,
        "--hf-paths",
        min=1,
        help="Optional HF path-count override within the selected config.",
    ),
    lf_shard_size: int | None = typer.Option(
        None,
        "--lf-shard-size",
        min=1,
        help="Optional LF paths per CSV shard.",
    ),
    hf_shard_size: int | None = typer.Option(
        None,
        "--hf-shard-size",
        min=1,
        help="Optional HF paths per CSV shard.",
    ),
    path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            readable=False,
            help="Use canonical.yaml for delivery or smoke.yaml for a short run.",
        ),
    ] = Path("configs/canonical.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Generate practical K=4 CSVs with the lean delivery path."""

    try:
        project_root = Path(__file__).resolve().parents[2]
        config_path = _resolve_practical_config_path(path)
        resolved = load_config(config_path)
        artifact_root = Path(resolved.data.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = project_root / artifact_root
        loader = build_practical_production_loader(
            project_root=project_root,
            config_path=config_path,
            artifact_root=artifact_root,
            artifact_fingerprint=artifact_fingerprint,
            processed_data_fingerprint=processed_data_fingerprint,
        )
        if any(
            value is not None
            for value in (lf_paths, hf_paths, lf_shard_size, hf_shard_size)
        ):
            resolved_lf_paths = lf_paths or resolved.simulation.low_frequency.paths
            resolved_hf_paths = hf_paths or resolved.simulation.high_frequency.paths
            plan = ProductionPlan(
                profile_id="practical-override-v1",
                years=resolved.simulation.low_frequency.years,
                lf_paths=resolved_lf_paths,
                hf_paths=resolved_hf_paths,
                lf_paths_per_shard=(
                    lf_shard_size
                    or min(resolved.export.low_paths_per_shard, resolved_lf_paths)
                ),
                hf_paths_per_shard=(
                    hf_shard_size
                    or min(resolved.export.high_paths_per_shard, resolved_hf_paths)
                ),
            )
        else:
            plan = ProductionPlan.from_config(resolved)
        resolved_workers = resolve_practical_workers(
            resolved, requested_workers=workers
        )
        result = run_practical_production(
            loader=loader,
            plan=plan,
            output_root=output_root,
            run_name=run_name,
            workers=resolved_workers,
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "practical_k4_generation_completed",
            "run_root": str(result.run_root),
            "manifest": str(result.manifest_path),
            "checksums": str(result.checksum_path),
            "manifest_fingerprint": result.manifest_fingerprint,
            "workers": result.workers,
            "paths": plan.lf_paths + plan.hf_paths,
            "work_units": result.work_units,
            "csv_shards": result.csv_shards,
            "total_rows": result.total_rows,
            "wall_seconds": result.wall_seconds,
        },
        json_output=json_output,
    )


@app.command("validate")
def validate() -> None:
    """Run structural and scientific validation gates (Phase 7)."""

    _not_implemented("prpg validate")


@app.command("qualify")
def qualify() -> None:
    """Consume frozen thresholds on untouched streams (Phase 7)."""

    _not_implemented("prpg qualify")


@release_app.callback()
def release(ctx: typer.Context) -> None:
    """Finalize checksums and write COMPLETE after authorization (Phase 8)."""

    if ctx.invoked_subcommand is None:
        _not_implemented("prpg release")


def _release_plan_from_json(path: Path) -> ReleaseFinalizationPlan:
    """Load the small prospective G8 allowlist document."""

    try:
        content = path.expanduser().resolve(strict=True).read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            "release finalization plan could not be read",
            details={"path": str(path), "reason": type(error).__name__},
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError("release finalization plan must be a JSON object")
    expected = {
        "run_id",
        "generation_commit",
        "release_commit",
        "release_tag",
        "finalizer_identity",
        "immutable_audit_files",
        "evidence",
    }
    if set(value) != expected:
        raise ConfigurationError("release finalization plan keys are invalid")
    audit_files = value["immutable_audit_files"]
    evidence_values = value["evidence"]
    if not isinstance(audit_files, list) or not all(
        isinstance(item, str) for item in audit_files
    ):
        raise ConfigurationError("release audit-file allowlist is invalid")
    if not isinstance(evidence_values, list):
        raise ConfigurationError("release evidence list is invalid")
    evidence: list[ReleaseEvidence] = []
    for item in evidence_values:
        if not isinstance(item, dict) or set(item) != {
            "evidence_id",
            "purpose",
            "relative_path",
            "sha256",
        }:
            raise ConfigurationError("release evidence entry is invalid")
        evidence.append(
            ReleaseEvidence(
                evidence_id=item["evidence_id"],
                purpose=item["purpose"],
                relative_path=item["relative_path"],
                sha256=item["sha256"],
            )
        )
    try:
        return ReleaseFinalizationPlan(
            run_id=value["run_id"],
            generation_commit=value["generation_commit"],
            release_commit=value["release_commit"],
            release_tag=value["release_tag"],
            finalizer_identity=value["finalizer_identity"],
            immutable_audit_files=tuple(audit_files),
            evidence=tuple(evidence),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "release finalization plan values are invalid"
        ) from error


@release_app.command("finalize")
def release_finalize(
    run_root: Annotated[Path, typer.Argument(help="Canonical G7 run root.")],
    plan_path: Annotated[
        Path,
        typer.Option("--plan", help="Prospective G8 finalization-plan JSON."),
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Finalize checksums and publish COMPLETE from an explicit allowlist."""

    try:
        plan_value = _release_plan_from_json(plan_path)
        result = finalize_complete(run_root, plan=plan_value)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "release_complete_finalized",
            "run_id": result.run_id,
            "root_sha256sums_sha256": result.root_sha256sums_sha256,
            "complete_sha256": result.complete_sha256,
            "run_manifest_sha256": result.run_manifest_sha256,
            "covered_file_count": result.covered_file_count,
            "released": False,
        },
        json_output=json_output,
    )


@release_app.command("attest")
def release_attest(
    run_root: Annotated[Path, typer.Argument(help="Completed canonical run root.")],
    output_path: Annotated[
        Path,
        typer.Option("--output", help="New external attestation JSON path."),
    ],
    verifier_identity: str = typer.Option(
        ..., "--verifier", help="Identity distinct from the release finalizer."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Independently verify COMPLETE and write external attestation bytes."""

    try:
        content = build_external_release_attestation(
            run_root, verifier_identity=verifier_identity
        )
        output = output_path.expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    except (FileExistsError, OSError) as error:
        _exit_for_error(
            ConfigurationError(
                "external release attestation could not be published",
                details={"path": str(output_path), "reason": type(error).__name__},
            ),
            json_output=json_output,
        )
    _emit(
        {
            "event": "external_release_attestation_created",
            "path": str(output),
            "sha256": hashlib.sha256(content).hexdigest(),
            "verifier_identity": verifier_identity,
        },
        json_output=json_output,
    )


@release_app.command("publish")
def release_publish(
    run_root: Annotated[Path, typer.Argument(help="Completed canonical run root.")],
    attestation_path: Annotated[
        Path,
        typer.Option("--attestation", help="External attestation JSON path."),
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Verify external attestation, publish RELEASED, and reverify the root."""

    try:
        result = publish_released(run_root, attestation_path=attestation_path)
        verified = verify_released(run_root, attestation_path=attestation_path)
        if result != verified:
            raise ConfigurationError("released verification changed after publication")
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "release_published",
            "run_id": result.complete.run_id,
            "root_sha256sums_sha256": result.complete.root_sha256sums_sha256,
            "attestation_sha256": result.attestation_sha256,
            "verifier_identity": result.verifier_identity,
            "released_sha256": result.released_sha256,
            "released": True,
        },
        json_output=json_output,
    )


@app.command("inspect")
def inspect_run(
    run_root: Annotated[Path, typer.Argument(help="Generated production run root.")],
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Summarize run lineage, progress, metrics, and hashes."""

    try:
        loaded = load_generated_production_run(run_root)
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    markers = {
        name: (loaded.run_root / name).is_file()
        for name in ("DATA_VALIDATED", "COMPLETE", "RELEASED")
    }
    _emit(
        {
            "event": "production_run_inspected",
            "run_root": str(loaded.run_root),
            "result_fingerprint": loaded.result_fingerprint,
            "run_fingerprint": loaded.binding.run_fingerprint,
            "simulation_fingerprint": loaded.binding.simulation_fingerprint,
            "qualification_fingerprint": loaded.binding.qualification_fingerprint,
            "materialization_fingerprint": (loaded.binding.materialization_fingerprint),
            "g3_evidence_fingerprint": loaded.loader.g3_evidence_fingerprint,
            "g5_authority_fingerprint": loaded.loader.g5_authority_fingerprint,
            "workers": loaded.workers,
            "paths": {
                "low_frequency": loaded.plan.lf_paths,
                "high_frequency": loaded.plan.hf_paths,
            },
            "years": loaded.plan.years,
            "total_rows": loaded.plan.total_rows,
            "csv_shards": loaded.plan.csv_shard_count,
            "data_validated": markers["DATA_VALIDATED"],
            "complete": markers["COMPLETE"],
            "released": markers["RELEASED"],
        },
        json_output=json_output,
    )


@app.command("reproduce")
def reproduce(
    run_root: Annotated[Path, typer.Argument(help="Generated production run root.")],
    output_root: Annotated[
        Path,
        typer.Option("--output", help="New isolated clean-replay root."),
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON event."
    ),
) -> None:
    """Replay a run from frozen artifacts and its manifest."""

    try:
        report = replay_representative_work_units(
            run_root,
            output_root=output_root.expanduser().resolve(strict=False),
        )
    except PRPGError as error:
        _exit_for_error(error, json_output=json_output)
    _emit(
        {
            "event": "production_clean_replay_completed",
            "source_run_fingerprint": report.source_run_fingerprint,
            "replay_run_fingerprint": report.replay_run_fingerprint,
            "source_result_fingerprint": report.source_result_fingerprint,
            "work_unit_ids": list(report.work_unit_ids),
            "csv_shards_compared": report.csv_shards_compared,
            "elapsed_seconds": report.elapsed_seconds,
            "identical": report.identical,
            "report_fingerprint": report.fingerprint,
            "report_path": str(
                output_root.expanduser().resolve(strict=False)
                / "clean-replay-report.json"
            ),
        },
        json_output=json_output,
    )


@app.command("run")
def run_pipeline() -> None:
    """Orchestrate the approved pipeline stages."""

    _not_implemented("prpg run")


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
