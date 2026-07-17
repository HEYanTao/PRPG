from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import prpg.model.g3_rehearsal as subject
from prpg.config import load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.errors import ModelError
from prpg.execution import deterministic_process_map
from prpg.model.g3_preflight import CANONICAL_PROCESSED_DATA_FINGERPRINT
from prpg.model.g3_registry import G3_TASK_REGISTRY
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    forward_backward,
    gaussian_hmm_bic,
    gaussian_hmm_parameter_count,
    historical_transition_support_report,
    posterior_identification_report,
    summarize_state_path,
    tied_covariance_diagnostic,
    validate_numerical_chain,
)
from prpg.model.input import load_calibration_input

ROOT = Path(__file__).resolve().parents[2]


def _noncanonical_parent(features: HMMFeatureMatrix, n_states: int) -> GaussianHMMFit:
    """Build a deterministic test parent; it can never cross the projection seam."""

    assert n_states == 4
    transition = np.full((n_states, n_states), 0.2 / (n_states - 1))
    np.fill_diagonal(transition, 0.8)
    levels = np.linspace(-1.5, 1.5, n_states)
    parameters = HMMParameters(
        start_probabilities=np.full(n_states, 1.0 / n_states),
        transition_matrix=transition,
        means=np.column_stack((levels, 0.5 * levels, 0.25 * levels, 0.1 * levels)),
        tied_covariance=np.eye(4),
    )
    states = np.resize(np.repeat(np.arange(n_states), 24), len(features.values))
    path = summarize_state_path(states, n_states, lengths=features.lengths)
    posterior = forward_backward(
        features.values,
        parameters,
        lengths=features.lengths,
    )
    return GaussianHMMFit(
        parameters=parameters,
        decoded_states=states,
        canonical_to_original=np.arange(n_states, dtype=np.int64),
        original_to_canonical=np.arange(n_states, dtype=np.int64),
        log_likelihood=posterior.total_log_likelihood,
        bic=gaussian_hmm_bic(
            posterior.total_log_likelihood,
            len(states),
            n_states,
            4,
        ),
        parameter_count=gaussian_hmm_parameter_count(n_states, 4),
        n_observations=len(states),
        n_features=4,
        n_states=n_states,
        lengths=features.lengths,
        best_restart_index=0,
        best_seed=1,
        restart_diagnostics=(),
        covariance_diagnostic=tied_covariance_diagnostic(parameters.tied_covariance),
        state_path_diagnostics=path,
        chain_diagnostics=validate_numerical_chain(parameters.transition_matrix),
        forward_backward=posterior,
        identification=posterior_identification_report(
            posterior.posterior_probabilities,
            states,
            parameters,
        ),
        historical_transition_support=historical_transition_support_report(
            parameters.transition_matrix,
            path.transition_counts,
            posterior.expected_transition_counts,
        ),
        scaler_scope=features.scaler_scope,
        scaler_means=features.scaler_means,
        scaler_standard_deviations=features.scaler_standard_deviations,
        feature_names=features.feature_names,
    )


def test_private_small_rehearsal_runs_real_families_but_cannot_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/canonical.yaml")
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(config.data.artifact_root),
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=config.fingerprint(),
        holdout_months=config.model.design_holdout_months,
    )
    geometry = subject._ConcreteRehearsalGeometry(
        workers=2,
        hmm_state_counts=(4,),
        hmm_restarts_per_k=2,
        block_histories=2,
        kernel_windows_per_cell=4,
        latent_chains_per_role=32,
        latent_bootstrap_replicates_per_role=32,
        latent_bootstrap_resample_size=32,
        bridge_coordinates=1,
        latent_measured_months=24,
        latent_extension_months=2_000,
        projection_eligible=False,
    )

    block_starts: list[int] = []
    original_block = subject.run_materialized_block_calibration

    def record_block(*args: Any, **kwargs: Any) -> Any:
        block_starts.append(int(kwargs["starting_length"]))
        return original_block(*args, **kwargs)

    bridge_job_counts: list[int] = []
    original_map = deterministic_process_map

    def record_map(
        function: Callable[[Any], Any],
        items: Sequence[Any] | Iterable[Any],
        *,
        workers: int,
        chunksize: int = 1,
    ) -> tuple[Any, ...]:
        materialized = tuple(items)
        if function is subject._run_benchmark_job:
            bridge_job_counts.append(len(materialized))
            assert all(isinstance(item, subject._BenchmarkJob) for item in materialized)
        return original_map(
            function,
            materialized,
            workers=workers,
            chunksize=chunksize,
        )

    monkeypatch.setattr(subject, "run_materialized_block_calibration", record_block)
    monkeypatch.setattr(subject, "deterministic_process_map", record_map)
    monkeypatch.setattr(
        subject,
        "_fit_rehearsal_hmm",
        lambda features, n_states, seeds, *, workers: _noncanonical_parent(
            features,
            n_states,
        ),
    )

    workload, telemetry = subject._run_test_only_rehearsal_workload(
        config,
        calibration_input,
        geometry,
    )

    assert block_starts == [10, 108, 12, 126]
    assert bridge_job_counts == [2, 2]
    assert workload.rehearsal_geometry is None
    assert workload.worker_parity_verified
    assert workload.selection_policy == "owner_fixed_k4_v3"
    assert workload.selected_n_states == 4
    assert workload.bridge_benchmark_n_states == 4
    assert telemetry.peak_process_tree_rss_bytes > 0
    assert telemetry.process_tree_cpu_seconds > 0.0
    assert telemetry.rss_telemetry_complete
    with pytest.raises(ModelError, match="exact fixed rehearsal geometry"):
        subject.assemble_noncanonical_g3_rehearsal_report(workload, telemetry)

    fingerprints: dict[str, str] = {}
    families: list[str] = []
    cells: list[tuple[str, str]] = []
    for spec in G3_TASK_REGISTRY:
        dependencies = tuple(
            (task_id, fingerprints[task_id]) for task_id in spec.dependencies
        )
        probe = workload.execute_task(spec.task_id, dependencies)
        fingerprints[spec.task_id] = probe.result_fingerprint
        families.extend(item.family for item in probe.timings)
        cells.extend((item.family, item.cell) for item in probe.timings)

    assert set(families) == {
        "hmm",
        "block_materialization",
        "block_call",
        "kernel",
        "dependence",
        "latent_generation",
        "latent_bootstrap",
        "bridge_benchmark",
        "lightweight",
    }
    assert [cell for family, cell in cells if family == "block_call"] == [
        "design_monthly",
        "design_daily",
        "production_monthly",
        "production_daily",
    ]
    assert ("bridge_benchmark", "total") in cells
    assert ("bridge_benchmark", "qmax") in cells


def test_resource_block_starts_bind_measured_design_and_production_maxima() -> None:
    assert subject._rehearsal_block_start_geometry(10, 108) == (
        (10, 108),
        (12, 126),
    )

    with pytest.raises(ModelError, match="daily.*outside policy"):
        subject._rehearsal_block_start_geometry(10, 127)


def test_bridge_jobs_and_qmax_cover_both_dgps_in_exact_coordinate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/canonical.yaml")
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(config.data.artifact_root),
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=config.fingerprint(),
        holdout_months=config.model.design_holdout_months,
    )
    design_features = subject.prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="design",
        design_rows=len(calibration_input.design.macro_features),
    )
    parent = SimpleNamespace(
        n_states=5,
        parameters=HMMParameters(
            np.full(5, 0.2),
            np.full((5, 5), 0.2),
            np.arange(20, dtype=np.float64).reshape(5, 4) / 10.0,
            np.eye(4),
        ),
    )
    monkeypatch.setattr(
        subject,
        "gaussian_hmm_fit_fingerprint",
        lambda _fit: "a" * 64,
    )

    jobs = subject._bridge_benchmark_jobs(
        config,
        parent,
        design_features,
        macro_block_length=20,
        coordinates=3,
    )

    assert tuple((item.dgp, item.pair) for item in jobs) == (
        ("model", 0),
        ("model", 1),
        ("model", 2),
        ("nonparametric", 0),
        ("nonparametric", 1),
        ("nonparametric", 2),
    )
    assert all(isinstance(item, subject._BenchmarkJob) for item in jobs)
    assert all(item.selected_n_states == 5 for item in jobs)
    assert all(item.design_history.shape == (193, 4) for item in jobs)

    records = tuple(
        SimpleNamespace(dgp=dgp, pair=pair, elapsed_seconds=seconds)
        for dgp, elapsed in (
            ("model", (1.0, 4.0, 2.0)),
            ("nonparametric", (3.0, 2.0, 5.0)),
        )
        for pair, seconds in enumerate(elapsed)
    )
    maxima, qmax = subject._bridge_elapsed_by_coordinate(records, coordinates=3)
    assert maxima == (3.0, 4.0, 5.0)
    assert qmax == 5.0

    with pytest.raises(ModelError, match="both DGPs in exact order"):
        subject._bridge_elapsed_by_coordinate(records[::-1], coordinates=3)
