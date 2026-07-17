from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from prpg.errors import IntegrityError, ModelError
from prpg.execution import (
    ExecutionResourceMeasurement,
    MeasuredProcessMapResult,
    MemoryPreflight,
)
from prpg.model import bridge_driver as driver
from prpg.model.alpha import AlphaOffsetGrid
from prpg.model.bridge import AcceleratorResult, BridgeResourcePreflight
from prpg.model.bridge_driver import BridgeDecision
from prpg.model.bridge_fixed_k import (
    ActualBridgeExecution,
    FitMode,
    assemble_actual_bridge,
)
from prpg.model.bridge_history import BridgeDGP
from prpg.model.calibration import RegisteredRollingBootstrapMaterialization
from prpg.model.g3_preflight import (
    CanonicalBridgeResourcePreflight,
    CanonicalG3Preflight,
)
from prpg.model.g3_science_handlers import canonical_planned_looks
from prpg.model.g3_task_publication import (
    bridge_executor_planned_looks,
    bridge_executor_source_slots,
    scientific_task_source_slots,
)
from prpg.model.hmm import GaussianHMMFit, HMMParameters, summarize_state_path
from prpg.model.inference import BinomialInterval
from prpg.model.input import MACRO_COLUMNS
from prpg.simulation.rng import CalibrationArtifact

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _binding(*, actual_statistic: float = 0.5) -> driver.BridgeScientificBinding:
    grids = tuple(
        driver.BoundAlphaOffsetGrid(
            role,
            "monthly" if "monthly" in role else "daily",
            SHA_A if role.startswith("design") else SHA_B,
            SHA_C,
            SHA_D,
            SHA_E,
        )
        for role in driver.BRIDGE_ALPHA_GRID_ROLES
    )
    provisional = driver.BridgeScientificBinding(
        SHA_A,
        20250308,
        1,
        2,
        SHA_B,
        SHA_C,
        3,
        SHA_D,
        actual_statistic,
        grids,
        (("model", SHA_D), ("nonparametric", SHA_E)),
        "",
    )
    return replace(
        provisional,
        fingerprint=driver._sha256_json(provisional.identity_dict()),
    )


def _tier_b_pair(
    binding: driver.BridgeScientificBinding,
    *,
    dgp: BridgeDGP,
    pair: int,
    mode: FitMode,
    statistic: float,
) -> driver.BridgeTierBPairEvidence:
    history = f"{(pair + (0 if dgp == 'model' else 20_000)):064x}"
    item = driver.BridgeTierBPairEvidence(
        dgp,
        pair,
        binding.selected_n_states,
        mode,
        history,
        True,
        statistic,
        None,
        None,
        binding.fingerprint,
        "",
    )
    return driver._with_tier_b_fingerprint(item)


def _benchmark_pair(
    binding: driver.BridgeScientificBinding,
    job: driver._BenchmarkJob,
    *,
    statistic: float,
    elapsed: float,
    agreement: bool = True,
) -> driver.BridgeBenchmarkPairEvidence:
    accelerated = _tier_b_pair(
        binding,
        dgp=job.dgp,
        pair=job.pair,
        mode="accelerated_10",
        statistic=statistic,
    )
    ordinary = _tier_b_pair(
        binding,
        dgp=job.dgp,
        pair=job.pair,
        mode="ordinary_50",
        statistic=statistic,
    )
    delta = 0.0 if agreement else 1.0
    item = driver.BridgeBenchmarkPairEvidence(
        job.dgp,
        job.pair,
        binding.selected_n_states,
        accelerated.history_sha256,
        accelerated,
        ordinary,
        (True, True),
        (True, True),
        (delta, delta),
        (delta, delta),
        (delta, delta),
        (agreement, agreement),
        elapsed,
        binding.fingerprint,
        "",
        "",
    )
    return driver._with_benchmark_fingerprints(item)


def _tier_a_batch(
    binding: driver.BridgeScientificBinding,
    job: driver._TierABatchJob,
    *,
    failed_pair: tuple[BridgeDGP, int] | None = None,
) -> tuple[driver.BridgeTierAPairEvidence, ...]:
    output: list[driver.BridgeTierAPairEvidence] = []
    for pair in range(job.pair_start, job.pair_stop):
        success = failed_pair not in {(job.dgp, pair), (job.dgp, -1)}
        item = driver.BridgeTierAPairEvidence(
            job.dgp,
            pair,
            binding.selected_n_states,
            f"{(40_000 + pair + (0 if job.dgp == 'model' else 1_000)):064x}",
            binding.purpose1_by_dgp[job.dgp],
            binding.selected_n_states if success else None,
            binding.selected_n_states,
            success,
            True,
            () if success else ("prefix:ModelError",),
            success,
            binding.fingerprint,
            "",
        )
        output.append(driver._with_tier_a_fingerprint(item))
    return tuple(output)


def _measurement(
    *, elapsed: float = 1.0, peak_rss: int = 100
) -> ExecutionResourceMeasurement:
    physical = 10_000
    preflight = MemoryPreflight(1_000, physical, 0.70, 0.10, True)
    return ExecutionResourceMeasurement(
        9,
        200,
        elapsed,
        peak_rss,
        peak_rss / physical,
        0,
        0,
        0,
        True,
        True,
        preflight,
        True,
    )


def _canonical_resource(
    base: object,
    projection: BridgeResourcePreflight,
) -> CanonicalBridgeResourcePreflight:
    provisional = CanonicalBridgeResourcePreflight(
        cast(str, getattr(base, "fingerprint", SHA_A)),
        projection.benchmark_pairs,
        projection.workers,
        projection.projected_worst_case_hours,
        projection.peak_memory_fraction,
        projection.maximum_hours,
        projection.maximum_memory_fraction,
        "",
    )
    return replace(
        provisional,
        fingerprint=driver._sha256_json(provisional.identity_dict()),
    )


class _Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        binding: driver.BridgeScientificBinding,
        *,
        statistic_for_pair: Callable[[BridgeDGP, int], float] | None = None,
        elapsed: float = 0.001,
        failed_tier_a_pair: tuple[BridgeDGP, int] | None = None,
        accelerator_disagreements: int = 0,
    ) -> None:
        self.binding = binding
        self.fixed_coordinates: list[tuple[BridgeDGP, int]] = []
        self.tier_a_coordinates: list[tuple[BridgeDGP, int, int]] = []
        self.benchmark_calls = 0
        self.statistic_for_pair = statistic_for_pair or (lambda _dgp, _pair: 1.0)
        self.failed_tier_a_pair = failed_tier_a_pair
        traces = {
            "model": np.zeros((1,), dtype=np.int64),
            "nonparametric": np.zeros((1,), dtype=np.int64),
        }
        for value in traces.values():
            value.setflags(write=False)
        monkeypatch.setattr(
            driver,
            "_prepare_request",
            lambda _request: (binding, MappingProxyType(traces)),
        )
        monkeypatch.setattr(
            driver,
            "canonical_bridge_resource_preflight",
            _canonical_resource,
        )

        def measured(
            function: object,
            items: Sequence[driver._BenchmarkJob],
            **kwargs: object,
        ) -> MeasuredProcessMapResult[driver.BridgeBenchmarkPairEvidence]:
            assert function is driver._run_benchmark_job
            assert kwargs["workers"] == 9
            self.benchmark_calls += 1
            outputs = tuple(
                _benchmark_pair(
                    binding,
                    job,
                    statistic=self.statistic_for_pair(job.dgp, job.pair),
                    elapsed=elapsed,
                    agreement=job.pair >= accelerator_disagreements,
                )
                for job in items
            )
            return MeasuredProcessMapResult(outputs, _measurement())

        def process(
            function: object,
            items: Sequence[Any],
            **kwargs: object,
        ) -> tuple[Any, ...]:
            assert kwargs["workers"] == 9
            if function is driver._run_tier_a_batch:
                output = []
                for tier_a_job in cast(Sequence[driver._TierABatchJob], items):
                    self.tier_a_coordinates.append(
                        (
                            tier_a_job.dgp,
                            tier_a_job.pair_start,
                            tier_a_job.pair_stop,
                        )
                    )
                    output.append(
                        _tier_a_batch(
                            binding,
                            tier_a_job,
                            failed_pair=self.failed_tier_a_pair,
                        )
                    )
                return tuple(output)
            if function is driver._run_fixed_k_job:
                output_pairs = []
                for fixed_job in cast(Sequence[driver._FixedKJob], items):
                    self.fixed_coordinates.append((fixed_job.dgp, fixed_job.pair))
                    output_pairs.append(
                        _tier_b_pair(
                            binding,
                            dgp=fixed_job.dgp,
                            pair=fixed_job.pair,
                            mode=fixed_job.mode,
                            statistic=self.statistic_for_pair(
                                fixed_job.dgp, fixed_job.pair
                            ),
                        )
                    )
                return tuple(output_pairs)
            raise AssertionError("unexpected process-map function")

        monkeypatch.setattr(driver, "measured_process_map", measured)
        monkeypatch.setattr(driver, "deterministic_process_map", process)


def _request() -> driver.CanonicalBridgeRequest:
    base = SimpleNamespace(
        estimated_peak_bytes=1_000,
        physical_memory_bytes=10_000,
        fingerprint=SHA_A,
    )
    return cast(
        driver.CanonicalBridgeRequest,
        SimpleNamespace(
            base_preflight=base,
            master_seed=1,
            scientific_version=1,
            design_parameters=None,
            design_standardized_history=None,
            design_continuation=None,
            macro_block_length=3,
        ),
    )


def _base_preflight() -> CanonicalG3Preflight:
    provisional = CanonicalG3Preflight(
        SHA_A,
        SHA_B,
        SHA_C,
        SHA_D,
        SHA_E,
        SHA_A,
        SHA_B,
        SHA_C,
        SHA_D,
        SHA_E,
        SHA_A,
        10,
        10_000,
        1_000_000,
        9,
        1_000,
        1_000,
        SHA_B,
        ("ok",),
        "",
    )
    return replace(
        provisional,
        fingerprint=driver._sha256_json(provisional.identity_dict()),
    )


def _parameters() -> HMMParameters:
    return HMMParameters(
        np.array([0.5, 0.5]),
        np.array([[0.8, 0.2], [0.2, 0.8]]),
        np.array([[-0.5, 0.2, 0.1, 0.4], [0.5, -0.2, -0.1, -0.4]]),
        np.eye(4),
    )


def _actual_fit(
    rows: int,
    lengths: tuple[int, ...],
    *,
    production: bool,
) -> GaussianHMMFit:
    parameters = _parameters()
    if production:
        shifted_means = np.asarray(parameters.means, dtype=np.float64).copy()
        shifted_means[:, 0] += 0.375
        parameters = HMMParameters(
            np.asarray(parameters.start_probabilities).copy(),
            np.asarray(parameters.transition_matrix).copy(),
            shifted_means,
            np.asarray(parameters.tied_covariance).copy(),
        )
    decoded = np.resize(np.repeat(np.arange(2), 5), rows).astype(np.int64)
    order = np.arange(2, dtype=np.int64)
    return GaussianHMMFit(
        parameters,
        decoded,
        order,
        order.copy(),
        -float(rows),
        200.0,
        20,
        rows,
        4,
        2,
        lengths,
        1,
        12345 if production else 54321,
        (),
        cast(Any, SimpleNamespace()),
        summarize_state_path(decoded, 2, lengths=lengths),
        cast(Any, SimpleNamespace(stationary_distribution=np.full(2, 0.5))),
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        "production_full" if production else "design_training",
        np.zeros(4),
        np.ones(4),
        tuple(MACRO_COLUMNS),
    )


def _minted_actual_bridge() -> ActualBridgeExecution:
    design = _actual_fit(193, (193,), production=False)
    production = _actual_fit(217, (210, 7), production=True)
    zeros = np.zeros((4, 2, 3), dtype=np.float64)
    return assemble_actual_bridge(
        design_fit=design,
        production_fit=production,
        design_monthly_block_length=5,
        production_monthly_block_length=5,
        design_daily_block_length=20,
        production_daily_block_length=20,
        design_monthly_state_counts=design.state_path_diagnostics.state_counts,
        production_monthly_state_counts=(
            production.state_path_diagnostics.state_counts
        ),
        design_daily_state_counts=np.array([2_025, 2_024]),
        production_daily_state_counts=np.array([2_274, 2_273]),
        design_monthly_kernel_offsets=zeros,
        production_monthly_kernel_offsets=zeros,
        design_daily_kernel_offsets=zeros,
        production_daily_kernel_offsets=zeros,
    )


@lru_cache(maxsize=2)
def _cached_rolling(
    artifact: CalibrationArtifact,
) -> RegisteredRollingBootstrapMaterialization:
    return driver.materialize_registered_rolling_bootstrap(
        master_seed=20250308,
        scientific_version=1,
        artifact=artifact,
    )


def _staged_request_and_binding() -> tuple[
    driver.CanonicalBridgeRequest,
    driver.BridgeScientificBinding,
]:
    base = _base_preflight()
    parameters = _parameters()
    history = np.zeros((193, 4), dtype=np.float64)
    continuation = np.r_[False, np.ones(192, dtype=np.bool_)]
    actual = _minted_actual_bridge()
    zeros = np.zeros((4, 2, 3), dtype=np.float64)
    lambdas = np.asarray(driver.BRIDGE_ALPHA_LAMBDAS, dtype=np.float64)
    grid_values: list[AlphaOffsetGrid] = []
    for index in range(4):
        grid_lambdas = lambdas.copy()
        grid_offsets = zeros.copy()
        grid_lambdas.setflags(write=False)
        grid_offsets.setflags(write=False)
        provisional_grid = AlphaOffsetGrid(
            "monthly" if index % 2 == 0 else "daily",
            SHA_A if index < 2 else SHA_B,
            SHA_C,
            SHA_D,
            "",
            grid_lambdas,
            grid_offsets,
        )
        grid_values.append(
            replace(
                provisional_grid,
                grid_fingerprint=driver._alpha_grid_fingerprint(
                    provisional_grid,
                    grid_lambdas,
                    grid_offsets,
                ),
            )
        )
    grids = cast(
        tuple[AlphaOffsetGrid, AlphaOffsetGrid, AlphaOffsetGrid, AlphaOffsetGrid],
        tuple(grid_values),
    )
    request = driver.CanonicalBridgeRequest(
        base,
        20250308,
        1,
        parameters,
        history,
        continuation,
        3,
        actual,
        grids,
    )
    actual_statistic, actual_fingerprint = driver._actual_bridge_binding(
        actual,
        parameters,
        grids,
        2,
    )
    traces = tuple(
        (
            dgp,
            driver._purpose1_trace_sha256(_cached_rolling(artifact).indices),
        )
        for dgp, artifact in zip(
            driver.BRIDGE_DGPS,
            (
                CalibrationArtifact.BRIDGE_MODEL,
                CalibrationArtifact.BRIDGE_NONPARAMETRIC,
            ),
            strict=True,
        )
    )
    bound_grids = driver._bound_alpha_grids(grids, 2)
    provisional = driver.BridgeScientificBinding(
        base.fingerprint,
        request.master_seed,
        request.scientific_version,
        2,
        driver.bridge_parameters_sha256(parameters),
        driver._design_history_sha256(history, continuation),
        request.macro_block_length,
        actual_fingerprint,
        actual_statistic,
        bound_grids,
        traces,
        "",
    )
    binding = replace(
        provisional,
        fingerprint=driver._sha256_json(provisional.identity_dict()),
    )
    return request, binding


def _prepare_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    statistic_for_pair: Callable[[BridgeDGP, int], float] | None = None,
    failed_tier_a_pair: tuple[BridgeDGP, int] | None = None,
    elapsed: float = 0.001,
) -> tuple[
    driver.CanonicalStagedBridgeSession,
    driver.CanonicalBridgeRequest,
    driver.BridgeScientificBinding,
    _Harness,
]:
    request, binding = _staged_request_and_binding()
    harness = _Harness(
        monkeypatch,
        binding,
        statistic_for_pair=statistic_for_pair,
        failed_tier_a_pair=failed_tier_a_pair,
        elapsed=elapsed,
    )
    monkeypatch.setattr(
        driver,
        "materialize_registered_rolling_bootstrap",
        lambda **arguments: _cached_rolling(arguments["artifact"]),
    )
    session = driver.prepare_canonical_staged_bridge(request, tmp_path)
    return session, request, binding, harness


def _execute_staged_tier_a_pair(
    session: driver.CanonicalStagedBridgeSession,
) -> tuple[
    driver.StagedBridgeTaskEvidence,
    driver.StagedBridgeTaskEvidence,
    driver.StagedBridgeTaskEvidence,
]:
    resource = driver.execute_staged_bridge_resource_benchmark(session)
    model = driver.execute_staged_bridge_tier_a(session, "model")
    nonparametric = driver.execute_staged_bridge_tier_a(session, "nonparametric")
    return resource, model, nonparametric


def _rewrite_staged_event_with_valid_receipt(
    path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    value.pop("receipt_fingerprint")
    mutate(value)
    value["receipt_fingerprint"] = driver._sha256_bytes(driver._canonical_bytes(value))
    path.write_bytes(driver._canonical_bytes(value))


def _rewrite_staged_evidence_with_valid_hashes(
    path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    value.pop("receipt_fingerprint")
    evidence = cast(dict[str, Any], value["evidence"])
    evidence.pop("evidence_fingerprint")
    mutate(evidence)
    evidence["evidence_fingerprint"] = driver._sha256_json(evidence)
    value["receipt_fingerprint"] = driver._sha256_bytes(driver._canonical_bytes(value))
    path.write_bytes(driver._canonical_bytes(value))


def test_staged_bridge_enforces_full_order_and_mints_exact_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, _binding_value, harness = _prepare_staged(
        tmp_path,
        monkeypatch,
    )
    initial = driver.reverify_staged_bridge_prefix(session)
    assert initial.next_action == "bridge_resource_preflight"
    assert not initial.terminal
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_bridge_tier_a(session, "model")

    resource = driver.execute_staged_bridge_resource_benchmark(session)
    assert resource.passed
    assert resource.parent_task_fingerprints == ()
    assert resource.rng_execution_fingerprint is not None
    assert driver.is_staged_bridge_task_evidence(resource)
    assert not driver.is_staged_bridge_task_evidence(copy.copy(resource))
    with pytest.raises(TypeError, match="minted only"):
        driver.StagedBridgeTaskEvidence()

    model_a = driver.execute_staged_bridge_tier_a(session, "model")
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_bridge_tier_b_next_look(session, "model")
    nonparametric_a = driver.execute_staged_bridge_tier_a(
        session,
        "nonparametric",
    )
    expected_a_parents = ((resource.task_id, resource.evidence_fingerprint),)
    assert model_a.parent_task_fingerprints == expected_a_parents
    assert nonparametric_a.parent_task_fingerprints == expected_a_parents
    assert model_a.evidence_fingerprint != nonparametric_a.evidence_fingerprint
    assert model_a.rng_execution_fingerprint != (
        nonparametric_a.rng_execution_fingerprint
    )

    model_look = driver.execute_staged_bridge_tier_b_next_look(session, "model")
    assert model_look.look_index == 1
    assert model_look.sample_count == 1_000
    assert model_look.decision == "pass_above_tail"
    assert driver.is_staged_bridge_planned_look_evidence(model_look)
    assert not driver.is_staged_bridge_planned_look_evidence(copy.copy(model_look))
    with pytest.raises(TypeError, match="minted only"):
        driver.StagedBridgePlannedLookEvidence()
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_bridge_tier_b_next_look(session, "nonparametric")
    model_b = driver.complete_staged_bridge_tier_b_task(session, "model")
    assert tuple(
        item.evidence_fingerprint
        for item in driver.staged_bridge_task_planned_looks(model_b)
    ) == (model_look.evidence_fingerprint,)

    nonparametric_look = driver.execute_staged_bridge_tier_b_next_look(
        session,
        "nonparametric",
    )
    nonparametric_b = driver.complete_staged_bridge_tier_b_task(
        session,
        "nonparametric",
    )
    actual = driver.execute_staged_actual_artifact_bridge(session)
    assert actual.rng_execution_fingerprint is None
    assert actual.parent_task_fingerprints == (
        (model_a.task_id, model_a.evidence_fingerprint),
        (nonparametric_a.task_id, nonparametric_a.evidence_fingerprint),
        (model_b.task_id, model_b.evidence_fingerprint),
        (nonparametric_b.task_id, nonparametric_b.evidence_fingerprint),
    )
    assert nonparametric_look.parent_task_fingerprints == (
        (resource.task_id, resource.evidence_fingerprint),
        (model_a.task_id, model_a.evidence_fingerprint),
        (nonparametric_a.task_id, nonparametric_a.evidence_fingerprint),
    )
    assert driver.staged_bridge_task_planned_looks(actual) == ()
    terminal = driver.reverify_staged_bridge_prefix(session)
    assert terminal.terminal
    assert terminal.next_action is None
    assert [item.task_id for item in terminal.task_evidence] == [
        "bridge_resource_preflight",
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
        "actual_artifact_bridge",
    ]
    assert harness.benchmark_calls == 1
    assert len(harness.tier_a_coordinates) == 100
    assert len(harness.fixed_coordinates) == 1_800
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_actual_artifact_bridge(session)

    from prpg.model import g3_boundary_views as boundary

    staged_by_task = {
        item.task_id: item
        for item in (
            resource,
            model_a,
            nonparametric_a,
            model_b,
            nonparametric_b,
            actual,
        )
    }
    views = {
        task_id: boundary.build_bridge_driver_task_view(
            staged_by_task[task_id],
            task_id=task_id,
            run_id=SHA_D,
            binding_fingerprint=driver._sha256_json({"task_id": task_id}),
        )
        for task_id in boundary.BRIDGE_TASK_IDS
    }
    assert all(boundary.is_bridge_driver_task_view(item) for item in views.values())
    assert views["actual_artifact_bridge"].rng_fingerprint is None
    assert all(
        views[task_id].rng_fingerprint is not None
        for task_id in boundary.BRIDGE_TASK_IDS
        if task_id != "actual_artifact_bridge"
    )
    assert views["bridge_tier_b_model_dgp"].planned_look_fingerprints == (
        model_look.evidence_fingerprint,
    )
    assert tuple(
        parent
        for parent, _ in views["actual_artifact_bridge"].parent_result_fingerprints
    ) == (
        "bridge_tier_a_model_dgp",
        "bridge_tier_a_nonparametric_dgp",
        "bridge_tier_b_model_dgp",
        "bridge_tier_b_nonparametric_dgp",
    )
    assert not boundary.is_bridge_driver_task_view(
        copy.copy(views["actual_artifact_bridge"])
    )

    for task_id in boundary.BRIDGE_TASK_IDS:
        staged = staged_by_task[task_id]
        prebinding = bridge_executor_source_slots(staged, task_id=task_id)
        final = scientific_task_source_slots(views[task_id])
        assert prebinding == final
        materializations = dict(final.materialization_fingerprints)
        assert materializations["task_source_materialization"]
        assert views[task_id].materialization_fingerprint in materializations.values()
        if task_id == "actual_artifact_bridge":
            assert final.rng_fingerprints == ()
            assert len(staged.parent_task_fingerprints) == 4
        else:
            assert tuple(dict(final.rng_fingerprints).values()) == (
                cast(str, staged.rng_execution_fingerprint),
            )

        alternate = boundary.build_bridge_driver_task_view(
            staged,
            task_id=task_id,
            run_id="e" * 64,
            binding_fingerprint=driver._sha256_json(
                {"alternate_task_binding": task_id}
            ),
        )
        assert scientific_task_source_slots(alternate) == final

        projected = bridge_executor_planned_looks(staged, task_id=task_id)
        assert projected == canonical_planned_looks(task_id, views[task_id])
        if task_id.startswith("bridge_tier_b_"):
            assert projected
        else:
            assert projected == ()

    with pytest.raises(ModelError, match="wrong task"):
        bridge_executor_source_slots(
            model_a,
            task_id="bridge_tier_a_nonparametric_dgp",
        )
    with pytest.raises(ModelError, match="producer authority"):
        bridge_executor_source_slots(
            copy.copy(actual),
            task_id="actual_artifact_bridge",
        )
    original_parents = actual.parent_task_fingerprints
    object.__setattr__(actual, "parent_task_fingerprints", original_parents[:-1])
    with pytest.raises(IntegrityError, match="content changed"):
        bridge_executor_source_slots(actual, task_id="actual_artifact_bridge")
    object.__setattr__(actual, "parent_task_fingerprints", original_parents)

    benchmark_receipt = session._operational_store.benchmark_receipt(
        session._legacy_binding
    )
    assert benchmark_receipt is not None
    benchmark_bytes = benchmark_receipt.path.read_bytes()
    benchmark_receipt.path.unlink()
    with pytest.raises(IntegrityError, match="lost its operational receipt"):
        driver.reverify_staged_bridge_prefix(session)
    benchmark_receipt.path.write_bytes(benchmark_bytes)

    tier_a_receipt = session._operational_store.tier_a_receipt(
        session._legacy_binding, "model"
    )
    assert tier_a_receipt is not None
    tier_a_bytes = tier_a_receipt.path.read_bytes()
    tier_a_receipt.path.unlink()
    with pytest.raises(IntegrityError, match="lost its aggregate receipt"):
        driver.reverify_staged_bridge_prefix(session)
    tier_a_receipt.path.write_bytes(tier_a_bytes)

    model_look_receipt = session._operational_store.tier_b_look_receipts(
        session._legacy_binding, "model"
    )[0]
    model_look_bytes = model_look_receipt.path.read_bytes()
    model_look_receipt.path.unlink()
    with pytest.raises(IntegrityError, match="lost its operational receipt"):
        driver.reverify_staged_bridge_prefix(session)
    model_look_receipt.path.write_bytes(model_look_bytes)


def test_staged_bridge_emits_each_registered_look_and_reverifies_old_look(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def statistic(_dgp: BridgeDGP, pair: int) -> float:
        return 1.0 if pair < 25 or 1_000 <= pair < 1_075 else 0.0

    session, _request_value, _binding_value, harness = _prepare_staged(
        tmp_path,
        monkeypatch,
        statistic_for_pair=statistic,
    )
    _execute_staged_tier_a_pair(session)

    first = driver.execute_staged_bridge_tier_b_next_look(session, "model")
    assert (first.look_index, first.sample_count, first.exceedances) == (1, 1_000, 25)
    assert first.decision == "continue"
    assert driver.reverify_staged_bridge_prefix(session).next_action == (
        "bridge_tier_b_model_dgp:look:2"
    )
    first_event_path = session.checkpoint_path / "event-003.json"
    original_first_event = first_event_path.read_bytes()
    _rewrite_staged_evidence_with_valid_hashes(
        first_event_path,
        lambda evidence: evidence.__setitem__(
            "exceedances", cast(int, evidence["exceedances"]) + 1
        ),
    )
    with pytest.raises(IntegrityError, match="differs from its live execution"):
        driver.reverify_staged_bridge_prefix(session)
    first_event_path.write_bytes(original_first_event)

    second = driver.execute_staged_bridge_tier_b_next_look(session, "model")
    assert (second.look_index, second.sample_count, second.exceedances) == (
        2,
        2_000,
        100,
    )
    assert second.decision == "pass_above_tail"
    assert second.rng_execution_fingerprint != first.rng_execution_fingerprint
    assert second.result_fingerprint != first.result_fingerprint
    assert driver.staged_bridge_planned_look_identity(first).look_index == 1
    assert driver.staged_bridge_planned_look_identity(second).look_index == 2
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_bridge_tier_b_next_look(session, "model")
    task = driver.complete_staged_bridge_tier_b_task(session, "model")
    from prpg.model import g3_boundary_views as boundary

    view = boundary.build_bridge_driver_task_view(
        task,
        task_id="bridge_tier_b_model_dgp",
        run_id=SHA_D,
        binding_fingerprint=driver._sha256_json({"two-look-tier-b": True}),
    )
    assert bridge_executor_planned_looks(
        task,
        task_id="bridge_tier_b_model_dgp",
    ) == canonical_planned_looks("bridge_tier_b_model_dgp", view)
    assert len(scientific_task_source_slots(view).materialization_fingerprints) == 2
    assert harness.fixed_coordinates[:2] == [("model", 100), ("model", 101)]
    assert harness.fixed_coordinates.count(("model", 100)) == 1
    assert harness.fixed_coordinates.count(("model", 1_000)) == 1
    with pytest.raises(IntegrityError, match="exceeds the staged prefix"):
        driver._execute_or_recover_staged_tier_b_look(
            session,
            "model",
            expected_look_index=1,
        )


def test_staged_bridge_runs_both_tier_a_tasks_before_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, _binding_value, harness = _prepare_staged(
        tmp_path,
        monkeypatch,
        failed_tier_a_pair=("model", -1),
    )
    resource = driver.execute_staged_bridge_resource_benchmark(session)
    model = driver.execute_staged_bridge_tier_a(session, "model")
    assert resource.passed and not model.passed
    assert driver.reverify_staged_bridge_prefix(session).next_action == (
        "bridge_tier_a_nonparametric_dgp"
    )
    nonparametric = driver.execute_staged_bridge_tier_a(session, "nonparametric")
    assert nonparametric.passed
    prefix = driver.reverify_staged_bridge_prefix(session)
    assert prefix.terminal and prefix.next_action is None
    assert len(harness.tier_a_coordinates) == 100
    assert not harness.fixed_coordinates
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_bridge_tier_b_next_look(session, "model")


def test_staged_bridge_resource_failure_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, _binding_value, harness = _prepare_staged(
        tmp_path,
        monkeypatch,
        elapsed=100.0,
    )
    resource = driver.execute_staged_bridge_resource_benchmark(session)
    assert not resource.passed
    prefix = driver.reverify_staged_bridge_prefix(session)
    assert prefix.terminal and prefix.next_action is None
    assert not harness.tier_a_coordinates
    with pytest.raises(ModelError, match="out of order"):
        driver.execute_staged_bridge_tier_a(session, "model")


def test_staged_bridge_resume_reuses_exact_prefix_and_rejects_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, request, _binding_value, harness = _prepare_staged(
        tmp_path,
        monkeypatch,
    )
    driver.execute_staged_bridge_resource_benchmark(session)
    driver.execute_staged_bridge_tier_a(session, "model")
    before = (
        harness.benchmark_calls,
        tuple(harness.tier_a_coordinates),
        tuple(harness.fixed_coordinates),
    )

    resumed = driver.prepare_canonical_staged_bridge(request, tmp_path)
    resumed_prefix = driver.reverify_staged_bridge_prefix(resumed)
    assert resumed_prefix.next_action == "bridge_tier_a_nonparametric_dgp"
    assert before == (
        harness.benchmark_calls,
        tuple(harness.tier_a_coordinates),
        tuple(harness.fixed_coordinates),
    )

    driver.execute_staged_bridge_tier_a(resumed, "nonparametric")
    assert len(harness.tier_a_coordinates) == 100
    with pytest.raises(ModelError, match="producer authority"):
        driver.reverify_staged_bridge_prefix(copy.copy(resumed))
    with pytest.raises(TypeError, match="created only"):
        driver.CanonicalStagedBridgeSession()

    original_request = resumed._request
    object.__setattr__(
        resumed,
        "_request",
        replace(original_request, master_seed=original_request.master_seed + 1),
    )
    with pytest.raises(IntegrityError):
        driver.reverify_staged_bridge_prefix(resumed)
    object.__setattr__(resumed, "_request", original_request)

    changed_actual = copy.copy(original_request.actual_bridge)
    object.__setattr__(
        changed_actual,
        "production_scaler_means_in_design_coordinates",
        np.ones(4),
    )
    object.__setattr__(
        resumed,
        "_request",
        replace(original_request, actual_bridge=changed_actual),
    )
    with pytest.raises(ModelError, match="producer authority|object-specifically"):
        driver.reverify_staged_bridge_prefix(resumed)
    object.__setattr__(resumed, "_request", original_request)

    original_materializations = resumed._rolling_materializations
    object.__setattr__(
        resumed,
        "_rolling_materializations",
        (original_materializations[0], original_materializations[0]),
    )
    with pytest.raises(IntegrityError, match="rolling materialization"):
        driver.reverify_staged_bridge_prefix(resumed)
    object.__setattr__(
        resumed,
        "_rolling_materializations",
        original_materializations,
    )
    with pytest.raises(IntegrityError, match="DGP"):
        driver.execute_staged_bridge_tier_b_next_look(
            resumed,
            cast(Any, "wrong-dgp"),
        )


def test_staged_bridge_receipts_are_future_free_and_reject_validly_rehashed_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, binding, _harness = _prepare_staged(
        tmp_path,
        monkeypatch,
    )
    resource = driver.execute_staged_bridge_resource_benchmark(session)
    model = driver.execute_staged_bridge_tier_a(session, "model")
    nonparametric = driver.execute_staged_bridge_tier_a(session, "nonparametric")
    assert model.parent_task_fingerprints == nonparametric.parent_task_fingerprints
    binding_bytes = (session.checkpoint_path / "binding.json").read_text()
    resource_path = session.checkpoint_path / "event-000.json"
    resource_bytes = resource_path.read_text()
    assert binding.actual_bridge_evidence_fingerprint not in binding_bytes
    assert binding.actual_bridge_evidence_fingerprint not in resource_bytes
    assert "actual_bridge" not in binding_bytes
    assert "actual_bridge" not in resource_bytes
    assert resource.evidence_fingerprint in resource_bytes

    _rewrite_staged_event_with_valid_receipt(
        resource_path,
        lambda value: cast(dict[str, Any], value["evidence"]).__setitem__(
            "future_result_fingerprint",
            SHA_E,
        ),
    )
    with pytest.raises(IntegrityError, match="unexpected or missing fields"):
        driver.reverify_staged_bridge_prefix(session)


def test_exact_driver_passes_both_dgps_and_resume_does_no_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    harness = _Harness(monkeypatch, binding)

    first = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert first.bridge_gate_passed
    assert not first.generation_authorized
    assert driver.is_verified_bridge_driver_result(first)
    assert not driver.is_verified_bridge_driver_result(
        replace(first, reasons=("forged",))
    )
    with pytest.raises(ModelError, match="sealed bridge-driver"):
        driver.reverify_bridge_driver_result(
            replace(first, reasons=("forged",)),
            _request(),
        )
    assert tuple(item.dgp for item in first.dgp_results) == driver.BRIDGE_DGPS
    assert all(item.passed and item.tier_b_pairs == 1_000 for item in first.dgp_results)
    assert harness.benchmark_calls == 1
    assert len(harness.tier_a_coordinates) == 100
    assert len(harness.fixed_coordinates) == 1_800
    assert harness.fixed_coordinates[:2] == [("model", 100), ("model", 101)]
    typed_looks = driver.BridgeCheckpointStore(tmp_path).verified_tier_b_look_results(
        first.binding, "model"
    )
    assert len(typed_looks) == 1
    assert typed_looks[0].sample_count == 1_000
    assert typed_looks[0].decision == "pass_above_tail"
    reverified = driver.reverify_bridge_driver_result(first, _request())
    assert not reverified.generation_authorized
    assert reverified.driver_result is first
    assert reverified.final_receipt_fingerprint == first.final_receipt_fingerprint
    assert tuple(reverified.tier_b_looks_by_dgp) == driver.BRIDGE_DGPS
    assert reverified.tier_b_looks_by_dgp["model"] == typed_looks
    from prpg.model import g3_boundary_views as boundary

    legacy_views = boundary.bridge_driver_task_views(first, _request())
    assert not any(
        boundary.is_bridge_driver_task_view(item)
        for item in legacy_views.as_mapping().values()
    )
    before = (
        harness.benchmark_calls,
        tuple(harness.tier_a_coordinates),
        tuple(harness.fixed_coordinates),
    )

    resumed = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert resumed.scientific_fingerprint == first.scientific_fingerprint
    assert driver.is_verified_bridge_driver_result(resumed)
    assert resumed.final_receipt_fingerprint == first.final_receipt_fingerprint
    assert before == (
        harness.benchmark_calls,
        tuple(harness.tier_a_coordinates),
        tuple(harness.fixed_coordinates),
    )

    original_binding = resumed.binding
    object.__setattr__(
        resumed,
        "binding",
        replace(
            original_binding,
            actual_statistic=original_binding.actual_statistic + 0.1,
        ),
    )
    with pytest.raises(IntegrityError, match="canonical request"):
        driver.reverify_bridge_driver_result(resumed, _request())
    object.__setattr__(resumed, "binding", original_binding)

    original_checkpoint = resumed.checkpoint_path
    object.__setattr__(resumed, "checkpoint_path", original_checkpoint.parent)
    with pytest.raises(IntegrityError, match="wrong binding"):
        driver.reverify_bridge_driver_result(resumed, _request())
    object.__setattr__(resumed, "checkpoint_path", original_checkpoint)

    store = driver.BridgeCheckpointStore(tmp_path)
    benchmark_receipt = store.benchmark_receipt(original_binding)
    assert benchmark_receipt is not None
    benchmark_bytes = benchmark_receipt.path.read_bytes()
    benchmark_receipt.path.unlink()
    with pytest.raises(IntegrityError, match="missing its benchmark receipt"):
        driver.reverify_bridge_driver_result(resumed, _request())
    benchmark_receipt.path.write_bytes(benchmark_bytes)

    original_benchmark = resumed.benchmark
    object.__setattr__(resumed, "benchmark", None)
    with pytest.raises(IntegrityError, match="benchmark result differs"):
        driver.reverify_bridge_driver_result(resumed, _request())
    object.__setattr__(resumed, "benchmark", original_benchmark)

    batch = store.tier_a_batch_receipts(original_binding, "model")[-1]
    batch_bytes = batch.path.read_bytes()
    batch.path.unlink()
    with pytest.raises(IntegrityError, match="Tier-A batch checkpoint chain"):
        driver.reverify_bridge_driver_result(resumed, _request())
    batch.path.write_bytes(batch_bytes)

    aggregate = store.tier_a_receipt(original_binding, "model")
    assert aggregate is not None
    aggregate_bytes = aggregate.path.read_bytes()
    aggregate.path.unlink()
    with pytest.raises(IntegrityError, match="missing its Tier-A aggregate"):
        driver.reverify_bridge_driver_result(resumed, _request())
    aggregate.path.write_bytes(aggregate_bytes)

    _rewrite_staged_event_with_valid_receipt(
        aggregate.path,
        lambda value: cast(list[dict[str, Any]], value["pairs"])[0].__setitem__(
            "success", False
        ),
    )
    with pytest.raises(IntegrityError, match="differs from its exact batch chain"):
        driver.reverify_bridge_driver_result(resumed, _request())
    aggregate.path.write_bytes(aggregate_bytes)

    original_dgp_results = resumed.dgp_results
    object.__setattr__(resumed, "dgp_results", tuple(reversed(original_dgp_results)))
    with pytest.raises(IntegrityError, match="DGP results differ"):
        driver.reverify_bridge_driver_result(resumed, _request())
    object.__setattr__(resumed, "dgp_results", original_dgp_results)

    object.__setattr__(resumed, "bridge_gate_passed", False)
    with pytest.raises(IntegrityError, match="terminal reduction"):
        driver.reverify_bridge_driver_result(resumed, _request())
    object.__setattr__(resumed, "bridge_gate_passed", True)

    final_receipt = store.final_receipt(original_binding)
    assert final_receipt is not None
    final_bytes = final_receipt.path.read_bytes()
    final_receipt.path.unlink()
    with pytest.raises(IntegrityError, match="final receipt is missing"):
        driver.reverify_bridge_driver_result(resumed, _request())
    final_receipt.path.write_bytes(final_bytes)

    _rewrite_staged_event_with_valid_receipt(
        final_receipt.path,
        lambda value: value.__setitem__("generation_authorized", True),
    )
    changed_final = store.final_receipt(original_binding)
    assert changed_final is not None
    original_final_fingerprint = resumed.final_receipt_fingerprint
    object.__setattr__(
        resumed,
        "final_receipt_fingerprint",
        changed_final.fingerprint,
    )
    with pytest.raises(IntegrityError, match="content is inconsistent"):
        driver.reverify_bridge_driver_result(resumed, _request())
    object.__setattr__(
        resumed,
        "final_receipt_fingerprint",
        original_final_fingerprint,
    )
    final_receipt.path.write_bytes(final_bytes)


def test_resource_failure_prevents_tier_a_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    harness = _Harness(monkeypatch, binding, elapsed=100.0)

    result = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert not result.bridge_gate_passed
    assert result.benchmark is not None and not result.benchmark.passed
    assert not harness.tier_a_coordinates
    assert not harness.fixed_coordinates
    assert all(item.tier_a is None for item in result.dgp_results)
    assert (
        not driver.reverify_bridge_driver_result(result, _request())
        .dgp_results[0]
        .passed
    )


def test_tier_a_failure_stops_before_remaining_tier_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    harness = _Harness(
        monkeypatch,
        binding,
        failed_tier_a_pair=("nonparametric", -1),
    )

    result = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert not result.bridge_gate_passed
    assert len(harness.tier_a_coordinates) == 100
    assert not harness.fixed_coordinates
    assert result.benchmark is not None and result.benchmark.passed
    assert (
        not driver.reverify_bridge_driver_result(result, _request())
        .dgp_results[1]
        .passed
    )


def test_ineligible_accelerator_falls_back_to_ordinary_for_both_dgps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    _Harness(monkeypatch, binding, accelerator_disagreements=2)

    result = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert result.bridge_gate_passed
    assert result.benchmark is not None
    assert all(
        not item.result.eligible
        and item.retained_mode == "ordinary_50"
        and all(pair.mode == "ordinary_50" for pair in item.retained_pairs)
        for item in result.benchmark.accelerators
    )


def test_both_dgps_execute_independently_and_fail_below_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    harness = _Harness(
        monkeypatch,
        binding,
        statistic_for_pair=lambda _dgp, _pair: 0.0,
    )

    result = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert not result.bridge_gate_passed
    assert not result.generation_authorized
    assert [item.tier_b_decision for item in result.dgp_results] == [
        "fail_below_tail",
        "fail_below_tail",
    ]
    assert len(harness.fixed_coordinates) == 1_800
    reverified = driver.reverify_bridge_driver_result(result, _request())
    assert all(not item.passed for item in reverified.dgp_results)


def test_resume_from_first_continue_look_never_repeats_completed_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()

    def statistic(_dgp: BridgeDGP, pair: int) -> float:
        # 25/1000 crosses 0.025; 100/2000 is wholly above it.
        return 1.0 if pair < 25 or 1_000 <= pair < 1_075 else 0.0

    harness = _Harness(monkeypatch, binding, statistic_for_pair=statistic)
    original = driver.BridgeCheckpointStore.write_tier_b_look
    interrupted = False

    def write_then_interrupt(
        self: driver.BridgeCheckpointStore,
        binding_value: driver.BridgeScientificBinding,
        *,
        dgp: BridgeDGP,
        look_index: int,
        pairs: Sequence[driver.BridgeTierBPairEvidence],
        prior_receipts: Sequence[driver._Receipt],
        cumulative_statistics: Sequence[float],
        decision: BridgeDecision,
        interval: BinomialInterval,
        exceedances: int,
    ) -> driver._Receipt:
        nonlocal interrupted
        receipt = original(
            self,
            binding_value,
            dgp=dgp,
            look_index=look_index,
            pairs=pairs,
            prior_receipts=prior_receipts,
            cumulative_statistics=cumulative_statistics,
            decision=decision,
            interval=interval,
            exceedances=exceedances,
        )
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated process interruption")
        return receipt

    monkeypatch.setattr(
        driver.BridgeCheckpointStore,
        "write_tier_b_look",
        write_then_interrupt,
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        driver.run_canonical_bridge_driver(_request(), tmp_path)
    first_coordinates = tuple(harness.fixed_coordinates)
    assert ("model", 999) in first_coordinates

    monkeypatch.setattr(driver.BridgeCheckpointStore, "write_tier_b_look", original)
    result = driver.run_canonical_bridge_driver(_request(), tmp_path)

    assert result.bridge_gate_passed
    resumed = harness.fixed_coordinates[len(first_coordinates) :]
    assert ("model", 100) not in resumed
    assert ("model", 999) not in resumed
    assert ("model", 1_000) in resumed
    assert all(item.tier_b_pairs == 2_000 for item in result.dgp_results)


def test_tier_a_batch_checkpoint_resumes_without_repeating_persisted_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    harness = _Harness(monkeypatch, binding)
    original = driver.BridgeCheckpointStore.write_tier_a_batch
    interrupted = False

    def write_then_interrupt(
        self: driver.BridgeCheckpointStore,
        binding_value: driver.BridgeScientificBinding,
        *,
        dgp: BridgeDGP,
        batch_index: int,
        pairs: Sequence[driver.BridgeTierAPairEvidence],
        prior_receipts: Sequence[driver._Receipt],
    ) -> driver._Receipt:
        nonlocal interrupted
        receipt = original(
            self,
            binding_value,
            dgp=dgp,
            batch_index=batch_index,
            pairs=pairs,
            prior_receipts=prior_receipts,
        )
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated Tier-A interruption")
        return receipt

    monkeypatch.setattr(
        driver.BridgeCheckpointStore,
        "write_tier_a_batch",
        write_then_interrupt,
    )
    with pytest.raises(RuntimeError, match="Tier-A interruption"):
        driver.run_canonical_bridge_driver(_request(), tmp_path)
    prior = tuple(harness.tier_a_coordinates)
    assert ("model", 0, 5) in prior

    monkeypatch.setattr(driver.BridgeCheckpointStore, "write_tier_a_batch", original)
    result = driver.run_canonical_bridge_driver(_request(), tmp_path)

    resumed = harness.tier_a_coordinates[len(prior) :]
    assert result.bridge_gate_passed
    assert ("model", 0, 5) not in resumed
    assert ("model", 5, 10) in resumed


def test_tampered_look_is_rejected_before_resume_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    harness = _Harness(monkeypatch, binding)
    result = driver.run_canonical_bridge_driver(_request(), tmp_path)
    path = result.checkpoint_path / "tier-b-model" / "look-01.json"
    value = json.loads(path.read_text())
    value["pairs"][0]["statistic"] = 999.0
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    prior_work = tuple(harness.fixed_coordinates)

    with pytest.raises(IntegrityError, match="fingerprint"):
        driver.reverify_bridge_driver_result(result, _request())
    with pytest.raises(IntegrityError, match="fingerprint"):
        driver.run_canonical_bridge_driver(_request(), tmp_path)
    assert tuple(harness.fixed_coordinates) == prior_work


def test_reverify_rejects_a_valid_but_nonterminal_local_look_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()

    def statistic(_dgp: BridgeDGP, pair: int) -> float:
        return 1.0 if pair < 25 or 1_000 <= pair < 1_075 else 0.0

    _Harness(monkeypatch, binding, statistic_for_pair=statistic)
    result = driver.run_canonical_bridge_driver(_request(), tmp_path)
    model_looks = result.checkpoint_path / "tier-b-model"
    (model_looks / "look-02.json").unlink()

    with pytest.raises(IntegrityError, match="missing or nonterminal"):
        driver.reverify_bridge_driver_result(result, _request())


def test_worker_timing_does_not_change_scientific_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _binding()
    _Harness(monkeypatch, binding, elapsed=0.001)
    first = driver.run_canonical_bridge_driver(_request(), tmp_path / "a")

    _Harness(monkeypatch, binding, elapsed=0.002)
    second = driver.run_canonical_bridge_driver(_request(), tmp_path / "b")

    assert first.scientific_fingerprint == second.scientific_fingerprint
    assert first.final_receipt_fingerprint != second.final_receipt_fingerprint
    assert (
        first.benchmark is not None
        and second.benchmark is not None
        and first.benchmark.receipt_fingerprint != second.benchmark.receipt_fingerprint
    )


def test_projection_covers_tier_a_and_mode_dependent_remaining_tier_b() -> None:
    binding = _binding()
    pair_agreement = np.ones(100, dtype=np.bool_)
    pair_agreement.setflags(write=False)
    eligible = AcceleratorResult(100, 100, True, pair_agreement)
    ineligible = AcceleratorResult(98, 100, False, pair_agreement)
    empty: tuple[driver.BridgeTierBPairEvidence, ...] = ()
    decisions = (
        driver.BridgeAcceleratorDecision(
            "model", eligible, "accelerated_10", empty, SHA_A
        ),
        driver.BridgeAcceleratorDecision(
            "nonparametric", ineligible, "ordinary_50", empty, SHA_B
        ),
    )
    expected_restarts = (
        2 * 250 * driver.BRIDGE_LEGACY_TIER_A_RESTARTS_PER_PAIR
        + 9_900 * driver.BRIDGE_ACCELERATED_RESTARTS_PER_PAIR
        + 9_900 * driver.BRIDGE_ORDINARY_RESTARTS_PER_PAIR
    )
    tier_a_weighted = (
        2
        * 250
        * driver.BRIDGE_LEGACY_TIER_A_RESTARTS_PER_PAIR
        * (5 / binding.selected_n_states) ** driver.BRIDGE_TIER_A_STATE_COMPLEXITY_POWER
    )
    tier_b_restarts = expected_restarts - (
        2 * 250 * driver.BRIDGE_LEGACY_TIER_A_RESTARTS_PER_PAIR
    )
    assert driver._projected_benchmark_pair_units(
        decisions,
        selected_n_states=binding.selected_n_states,
    ) == int(
        np.ceil(
            driver.BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR
            * (tier_a_weighted + tier_b_restarts)
            / driver.BRIDGE_BENCHMARK_RESTARTS_PER_PAIR
        )
    )

    fixed_k4_restarts = (
        2 * 250 * driver.BRIDGE_TIER_A_RESTARTS_PER_PAIR + tier_b_restarts
    )
    assert driver._projected_benchmark_pair_units(
        decisions,
        selected_n_states=4,
        scientific_version=driver.CANONICAL_SCIENTIFIC_VERSION,
    ) == int(
        np.ceil(
            driver.BRIDGE_RESOURCE_PROJECTION_SAFETY_FACTOR
            * fixed_k4_restarts
            / driver.BRIDGE_BENCHMARK_RESTARTS_PER_PAIR
        )
    )
    assert binding.selected_n_states == 2


def test_alpha_grid_content_tamper_is_rejected() -> None:
    lambdas = np.asarray(driver.BRIDGE_ALPHA_LAMBDAS, dtype=np.float64)
    offsets = np.zeros((4, 2, 3), dtype=np.float64)
    lambdas.setflags(write=False)
    offsets.setflags(write=False)
    provisional = AlphaOffsetGrid(
        "monthly",
        SHA_A,
        SHA_B,
        SHA_C,
        "",
        lambdas,
        offsets,
    )
    valid = replace(
        provisional,
        grid_fingerprint=driver._alpha_grid_fingerprint(provisional, lambdas, offsets),
    )
    invalid = replace(valid, grid_fingerprint=SHA_D)
    grids = (
        invalid,
        replace(valid, frequency="daily"),
        valid,
        replace(valid, frequency="daily"),
    )

    with pytest.raises(ModelError, match="fingerprint"):
        driver._bound_alpha_grids(grids, 2)


def test_tier_a_pair_cannot_substitute_a_different_purpose1_trace(
    tmp_path: Path,
) -> None:
    binding = _binding()
    store = driver.BridgeCheckpointStore(tmp_path)
    store.initialize(binding)
    job = cast(
        driver._TierABatchJob,
        SimpleNamespace(dgp="model", pair_start=0, pair_stop=5),
    )
    pairs = list(_tier_a_batch(binding, job))
    pairs[0] = driver._with_tier_a_fingerprint(
        replace(pairs[0], purpose1_trace_sha256=SHA_A, evidence_fingerprint="")
    )

    with pytest.raises(ModelError, match="identity"):
        store.write_tier_a_batch(
            binding,
            dgp="model",
            batch_index=0,
            pairs=pairs,
            prior_receipts=(),
        )


def test_actual_bridge_binding_rejects_untyped_evidence() -> None:
    with pytest.raises(ModelError, match="producer authority"):
        driver._actual_bridge_binding(
            object(),  # type: ignore[arg-type]
            cast(Any, object()),
            (),
            2,
        )


def test_prepare_request_builds_official_two_trace_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = _parameters()
    values = np.zeros((193, 4), dtype=np.float64)
    continuation = np.r_[False, np.ones(192, dtype=np.bool_)]
    trace = np.zeros((10_000, 6, 12), dtype=np.int64)
    trace.setflags(write=False)
    alpha_bound = _binding().alpha_offset_grids
    monkeypatch.setattr(driver, "_bound_alpha_grids", lambda _grids, _k: alpha_bound)
    monkeypatch.setattr(
        driver,
        "_actual_bridge_binding",
        lambda _actual, _parameters, _grids, _k: (0.5, SHA_D),
    )
    artifacts: list[object] = []

    def materialize(**kwargs: object) -> np.ndarray:
        artifacts.append(kwargs["artifact"])
        return trace

    monkeypatch.setattr(driver, "materialize_rolling_bootstrap_indices", materialize)
    request = driver.CanonicalBridgeRequest(
        _base_preflight(),
        123,
        1,
        parameters,
        values,
        continuation,
        3,
        cast(Any, object()),
        cast(Any, (None, None, None, None)),
    )

    binding, traces = driver._prepare_request(request)

    assert binding.selected_n_states == 2
    assert binding.actual_statistic == 0.5
    assert tuple(traces) == driver.BRIDGE_DGPS
    assert len(artifacts) == 2
    assert all(not item.flags.writeable for item in traces.values())

    with pytest.raises(
        ModelError,
        match="version-3 canonical bridge requires fixed K=4",
    ):
        driver._prepare_request(
            replace(
                request,
                scientific_version=driver.CANONICAL_SCIENTIFIC_VERSION,
            )
        )


def test_compact_worker_adapters_reduce_fit_rich_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    parameters = _parameters()
    history = object()
    selection = SimpleNamespace(selected_n_states=2, passed=True)
    tier_a_result = SimpleNamespace(
        dgp="model",
        pair=0,
        actual_selected_k=2,
        history_sha256=SHA_A,
        rolling_bootstrap_sha256=binding.purpose1_by_dgp["model"],
        prefix=SimpleNamespace(selection=selection),
        full=SimpleNamespace(selection=selection),
        member_failures=(),
        success=True,
    )
    monkeypatch.setattr(
        driver, "generate_registered_bridge_history", lambda **_kw: history
    )
    monkeypatch.setattr(
        driver, "run_bridge_tier_a_pair", lambda *_a, **_kw: tier_a_result
    )
    trace = np.zeros((1,), dtype=np.int64)
    job = driver._TierABatchJob(
        "model",
        0,
        1,
        binding.fingerprint,
        2,
        1,
        1,
        parameters,
        np.zeros((193, 4)),
        np.r_[False, np.ones(192, dtype=np.bool_)],
        3,
        trace,
        binding.purpose1_by_dgp["model"],
    )

    tier_a = driver._run_tier_a_batch(job)

    assert len(tier_a) == 1 and tier_a[0].success
    fixed_result = SimpleNamespace(
        dgp="model",
        pair=7,
        n_states=2,
        mode="accelerated_10",
        history_sha256=SHA_B,
        valid=True,
        statistic=0.75,
        failure_type=None,
        failure_message=None,
    )
    monkeypatch.setattr(
        driver, "run_bridge_fixed_k_pair", lambda *_a, **_kw: fixed_result
    )
    fixed_job = driver._FixedKJob(
        "model",
        7,
        binding.fingerprint,
        2,
        1,
        1,
        parameters,
        np.zeros((193, 4)),
        np.r_[False, np.ones(192, dtype=np.bool_)],
        3,
        "accelerated_10",
    )

    fixed = driver._run_fixed_k_job(fixed_job)

    assert fixed.valid and fixed.statistic == 0.75
    assert fixed.evidence_fingerprint == driver._sha256_json(fixed.identity_dict())


def test_version3_tier_a_worker_uses_fixed_k4_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transition = np.full((4, 4), 0.05)
    np.fill_diagonal(transition, 0.85)
    parameters = HMMParameters(
        np.full(4, 0.25),
        transition,
        np.arange(16, dtype=np.float64).reshape(4, 4),
        np.eye(4),
    )
    member = SimpleNamespace(
        fit=object(),
        passed=True,
        rejection_reasons=(),
        observed_episode_counts=(2, 2, 2, 2),
        reference_to_candidate=(0, 1, 2, 3),
    )
    fixed_result = SimpleNamespace(
        dgp="model",
        pair=0,
        actual_selected_k=4,
        history_sha256=SHA_A,
        prefix=member,
        full=member,
        success=True,
    )
    monkeypatch.setattr(
        driver, "generate_registered_bridge_history", lambda **_kw: object()
    )
    monkeypatch.setattr(
        driver,
        "run_bridge_fixed_k4_tier_a_pair",
        lambda *_a, **_kw: fixed_result,
    )

    def legacy_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("version-3 Tier A invoked legacy order selection")

    monkeypatch.setattr(driver, "run_bridge_tier_a_pair", legacy_forbidden)
    trace = np.zeros((1,), dtype=np.int64)
    job = driver._TierABatchJob(
        "model",
        0,
        1,
        SHA_B,
        4,
        1,
        driver.CANONICAL_SCIENTIFIC_VERSION,
        parameters,
        np.zeros((193, 4)),
        np.r_[False, np.ones(192, dtype=np.bool_)],
        3,
        trace,
        SHA_C,
    )

    evidence = driver._run_tier_a_batch(job)

    assert len(evidence) == 1
    assert evidence[0].success
    assert evidence[0].tier_a_policy == driver.BRIDGE_FIXED_K4_TIER_A_POLICY
    assert evidence[0].prefix_observed_episode_counts == (2, 2, 2, 2)
    assert evidence[0].prefix_reference_to_candidate == (0, 1, 2, 3)


def test_version3_tier_a_rng_identity_states_exact_k4_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fingerprint(**kwargs: object) -> str:
        captured.update(kwargs)
        return SHA_A

    monkeypatch.setattr(driver, "rng_execution_fingerprint", fingerprint)
    session = SimpleNamespace(
        binding=SimpleNamespace(
            master_seed=123,
            scientific_version=driver.CANONICAL_SCIENTIFIC_VERSION,
        )
    )
    materialization = cast(RegisteredRollingBootstrapMaterialization, object())

    result = driver._staged_tier_a_rng_fingerprint(
        cast(driver.CanonicalStagedBridgeSession, session),
        "model",
        materialization,
    )

    assert result == SHA_A
    assert captured["namespace"] == "bridge_tier_a_model_fixed_k4_composite"
    contract = cast(dict[str, object], captured["contract"])
    assert contract["tier_a_policy"] == driver.BRIDGE_FIXED_K4_TIER_A_POLICY
    assert contract["rolling_score_bootstrap_used"] is False
    assert contract["cross_k_selection_used"] is False
    assert contract["bic_used"] is False
    fit_contracts = cast(list[dict[str, object]], contract["fixed_k4_fit_contracts"])
    assert len(fit_contracts) == 2
    assert all(item["n_states"] == 4 for item in fit_contracts)
    assert all(item["fit_coordinates_per_member"] == 1 for item in fit_contracts)
    assert all(item["rolling_fold_ids"] == [] for item in fit_contracts)
    assert all(item["restart_ids"] == list(range(50)) for item in fit_contracts)


def test_benchmark_worker_reduces_member_and_component_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    fit_fast = SimpleNamespace(log_likelihood=-100.0, n_observations=193)
    fit_full = SimpleNamespace(log_likelihood=-100.00001, n_observations=193)
    components_fast = SimpleNamespace(
        normalized_coordinates=np.array([0.1, 0.2]),
        coordinate_family=("a", "b"),
        coordinate_names=("x", "y"),
        statistic=SimpleNamespace(maximum=0.2),
        component_decisions=np.array([True, True]),
    )
    components_full = SimpleNamespace(
        normalized_coordinates=np.array([0.1, 0.20001]),
        coordinate_family=("a", "b"),
        coordinate_names=("x", "y"),
        statistic=SimpleNamespace(maximum=0.20001),
        component_decisions=np.array([True, True]),
    )

    def rich(mode: FitMode) -> SimpleNamespace:
        fast = mode == "accelerated_10"
        return SimpleNamespace(
            dgp="model",
            pair=0,
            n_states=2,
            mode=mode,
            history_sha256=SHA_A,
            valid=True,
            statistic=0.2 if fast else 0.20001,
            failure_type=None,
            failure_message=None,
            prefix=SimpleNamespace(valid=True, fit=fit_fast if fast else fit_full),
            full=SimpleNamespace(valid=True, fit=fit_fast if fast else fit_full),
            components=components_fast if fast else components_full,
        )

    monkeypatch.setattr(driver, "_run_fixed_k_full", lambda job: rich(job.mode))
    job = driver._BenchmarkJob(
        "model",
        0,
        binding.fingerprint,
        2,
        1,
        1,
        _parameters(),
        np.zeros((193, 4)),
        np.r_[False, np.ones(192, dtype=np.bool_)],
        3,
    )

    result = driver._run_benchmark_job(job)

    assert result.accelerated_valid == (True, True)
    assert result.ordinary_valid == (True, True)
    assert max(result.maximum_component_difference) == pytest.approx(1e-5)
    assert all(result.component_decisions_identical)
    assert result.elapsed_seconds > 0.0


def test_actual_bridge_typed_binding_hashes_components_and_grids() -> None:
    request, _binding_value = _staged_request_and_binding()
    actual = request.actual_bridge

    maximum, fingerprint = driver._actual_bridge_binding(
        actual,
        request.design_parameters,
        request.alpha_offset_grids,
        2,
    )

    assert maximum == actual.components.statistic.maximum
    assert len(fingerprint) == 64
    with pytest.raises(ModelError, match="object-specifically"):
        driver._actual_bridge_binding(
            copy.copy(actual),
            request.design_parameters,
            request.alpha_offset_grids,
            2,
        )


def test_staged_checkpoint_store_rejects_gaps_extras_and_bad_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, _binding_value, _harness = _prepare_staged(
        tmp_path,
        monkeypatch,
    )
    store = session._stage_store
    binding = session.binding
    binding_path = session.checkpoint_path / "binding.json"
    original_binding = binding_path.read_bytes()

    _rewrite_staged_event_with_valid_receipt(
        binding_path,
        lambda value: value.__setitem__("binding_fingerprint", SHA_E),
    )
    with pytest.raises(IntegrityError, match="differs from the request"):
        store.verify_binding(binding)
    binding_path.write_bytes(original_binding)

    extra = session.checkpoint_path / "unexpected.json"
    extra.write_text("{}\n")
    with pytest.raises(IntegrityError, match="extra file"):
        store.event_receipts(binding)
    extra.unlink()

    gap = session.checkpoint_path / "event-001.json"
    gap.write_text("{}\n")
    with pytest.raises(IntegrityError, match="contains a gap"):
        store.event_receipts(binding)
    gap.unlink()

    with pytest.raises(ModelError, match="exact next prefix"):
        store.write_event(
            binding,
            {
                "event_index": 1,
                "previous_receipt_fingerprint": None,
                "common_binding_fingerprint": binding.fingerprint,
            },
        )

    event = session.checkpoint_path / "event-000.json"
    event.write_bytes(b"{\n")
    with pytest.raises(IntegrityError, match="not valid JSON"):
        store.event_receipts(binding)
    event.write_bytes(b"{}\n\n")
    with pytest.raises(IntegrityError, match="not canonical"):
        store.event_receipts(binding)
    event.write_bytes(b"{}\n")
    with pytest.raises(IntegrityError, match="receipt fingerprint"):
        store.event_receipts(binding)
    event.unlink()

    store.write_event(
        binding,
        {
            "schema_version": driver.STAGED_BRIDGE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "unregistered_stage",
            "event_index": 0,
            "common_binding_fingerprint": binding.fingerprint,
            "previous_receipt_fingerprint": None,
            "evidence": {},
        },
    )
    with pytest.raises(IntegrityError, match="kind is not registered"):
        driver.reverify_staged_bridge_prefix(session)


def test_legacy_checkpoint_store_rejects_namespace_gaps_and_wrong_batches(
    tmp_path: Path,
) -> None:
    binding = _binding()
    store = driver.BridgeCheckpointStore(tmp_path)
    checkpoint = store.initialize(binding)
    binding_path = checkpoint / "binding.json"
    original_binding = binding_path.read_bytes()
    _rewrite_staged_event_with_valid_receipt(
        binding_path,
        lambda value: value.__setitem__("binding_fingerprint", SHA_E),
    )
    with pytest.raises(IntegrityError, match="differs from the request"):
        store.verify_binding(binding)
    binding_path.write_bytes(original_binding)

    tier_a_directory = checkpoint / "tier-a-model"
    tier_a_directory.mkdir()
    extra_batch = tier_a_directory / "unexpected.json"
    extra_batch.write_text("{}\n")
    with pytest.raises(IntegrityError, match="extra file"):
        store.tier_a_batch_receipts(binding, "model")
    extra_batch.unlink()
    gap_batch = tier_a_directory / "batch-01.json"
    gap_batch.write_text("{}\n")
    with pytest.raises(IntegrityError, match="sequence gap"):
        store.tier_a_batch_receipts(binding, "model")
    gap_batch.unlink()

    with pytest.raises(ModelError, match="next registered batch"):
        store.write_tier_a_batch(
            binding,
            dgp="model",
            batch_index=1,
            pairs=(),
            prior_receipts=(),
        )
    short_job = cast(
        driver._TierABatchJob,
        SimpleNamespace(dgp="model", pair_start=0, pair_stop=4),
    )
    with pytest.raises(ModelError, match="wrong exact pair interval"):
        store.write_tier_a_batch(
            binding,
            dgp="model",
            batch_index=0,
            pairs=_tier_a_batch(binding, short_job),
            prior_receipts=(),
        )
    with pytest.raises(ModelError, match="exact ordered pairs"):
        store.write_tier_a(binding, "model", ())

    tier_b_directory = checkpoint / "tier-b-model"
    tier_b_directory.mkdir()
    extra_look = tier_b_directory / "unexpected.json"
    extra_look.write_text("{}\n")
    with pytest.raises(IntegrityError, match="extra file"):
        store.tier_b_look_receipts(binding, "model")
    extra_look.unlink()
    gap_look = tier_b_directory / "look-02.json"
    gap_look.write_text("{}\n")
    with pytest.raises(IntegrityError, match="sequence gap"):
        store.tier_b_look_receipts(binding, "model")
    gap_look.unlink()

    interval = driver.clopper_pearson_interval(
        0,
        1_000,
        confidence=1.0 - driver.TIER_B_INTERVAL_ERROR,
    )
    with pytest.raises(ModelError, match="next registered look"):
        store.write_tier_b_look(
            binding,
            dgp="model",
            look_index=0,
            pairs=(),
            prior_receipts=(),
            cumulative_statistics=(),
            decision="continue",
            interval=interval,
            exceedances=0,
        )
    terminal = driver._Receipt(
        tmp_path / "terminal.json",
        {"decision": "pass_above_tail"},
        SHA_A,
        False,
    )
    with pytest.raises(ModelError, match="after a terminal decision"):
        store.write_tier_b_look(
            binding,
            dgp="model",
            look_index=2,
            pairs=(),
            prior_receipts=(terminal,),
            cumulative_statistics=(),
            decision="continue",
            interval=interval,
            exceedances=0,
        )
    with pytest.raises(ModelError, match="exact ordered 1,000-pair batch"):
        store.write_tier_b_look(
            binding,
            dgp="model",
            look_index=1,
            pairs=(),
            prior_receipts=(),
            cumulative_statistics=(),
            decision="continue",
            interval=interval,
            exceedances=0,
        )


def test_staged_evidence_decoders_reject_forged_task_and_look_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, _binding_value, _harness = _prepare_staged(
        tmp_path,
        monkeypatch,
    )

    task_payload = driver._staged_task_payload(
        task_id="bridge_resource_preflight",
        dgp=None,
        passed=True,
        common_binding_fingerprint=session.binding.fingerprint,
        parent_task_fingerprints=(),
        source_materialization_fingerprint=SHA_A,
        rng_execution_fingerprint=SHA_B,
        result_fingerprint=SHA_C,
    )
    task_payload["evidence_fingerprint"] = driver._sha256_json(task_payload)
    task_receipt = driver._Receipt(
        tmp_path / "task.json",
        {"evidence": task_payload},
        SHA_D,
        False,
    )
    task = driver._mint_staged_task_from_receipt(session, task_receipt, 0)
    assert task.task_id == "bridge_resource_preflight"

    wrong_schema = copy.deepcopy(task_payload)
    wrong_schema["schema_id"] = "unknown"
    with pytest.raises(IntegrityError, match="schema is not registered"):
        driver._mint_staged_task_from_receipt(
            session,
            replace(task_receipt, identity={"evidence": wrong_schema}),
            0,
        )

    wrong_dgp = copy.deepcopy(task_payload)
    wrong_dgp["dgp"] = "model"
    with pytest.raises(IntegrityError, match="ID/DGP identity"):
        driver._mint_staged_task_from_receipt(
            session,
            replace(task_receipt, identity={"evidence": wrong_dgp}),
            0,
        )

    wrong_fingerprint = copy.deepcopy(task_payload)
    wrong_fingerprint["evidence_fingerprint"] = SHA_E
    with pytest.raises(IntegrityError, match="evidence fingerprint is invalid"):
        driver._mint_staged_task_from_receipt(
            session,
            replace(task_receipt, identity={"evidence": wrong_fingerprint}),
            0,
        )

    look_payload = driver._staged_look_payload(
        task_id="bridge_tier_b_model_dgp",
        dgp="model",
        look_index=1,
        sample_count=1_000,
        exceedances=50,
        decision="continue",
        common_binding_fingerprint=session.binding.fingerprint,
        parent_task_fingerprints=(("bridge_resource_preflight", SHA_A),),
        source_materialization_fingerprint=SHA_B,
        rng_execution_fingerprint=SHA_C,
        result_fingerprint=SHA_D,
    )
    look_payload["evidence_fingerprint"] = driver._sha256_json(look_payload)
    look_receipt = driver._Receipt(
        tmp_path / "look.json",
        {"evidence": look_payload},
        SHA_E,
        False,
    )
    look = driver._mint_staged_look_from_receipt(session, look_receipt, 0)
    assert (look.dgp, look.look_index, look.decision) == ("model", 1, "continue")

    wrong_schema = copy.deepcopy(look_payload)
    wrong_schema["schema_id"] = "unknown"
    with pytest.raises(IntegrityError, match="schema is not registered"):
        driver._mint_staged_look_from_receipt(
            session,
            replace(look_receipt, identity={"evidence": wrong_schema}),
            0,
        )

    wrong_task = copy.deepcopy(look_payload)
    wrong_task["task_id"] = "bridge_tier_b_nonparametric_dgp"
    with pytest.raises(IntegrityError, match="task/DGP identity"):
        driver._mint_staged_look_from_receipt(
            session,
            replace(look_receipt, identity={"evidence": wrong_task}),
            0,
        )

    wrong_fingerprint = copy.deepcopy(look_payload)
    wrong_fingerprint["evidence_fingerprint"] = SHA_A
    with pytest.raises(IntegrityError, match="evidence fingerprint is invalid"):
        driver._mint_staged_look_from_receipt(
            session,
            replace(look_receipt, identity={"evidence": wrong_fingerprint}),
            0,
        )

    duplicated_parent = [
        {"task_id": "bridge_resource_preflight", "evidence_fingerprint": SHA_A},
        {"task_id": "bridge_resource_preflight", "evidence_fingerprint": SHA_B},
    ]
    with pytest.raises(IntegrityError, match="contain a duplicate"):
        driver._parent_fingerprints(duplicated_parent)


def test_staged_session_reverification_rejects_live_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _request_value, _binding_value, _harness = _prepare_staged(
        tmp_path,
        monkeypatch,
    )

    original_binding = session.binding
    object.__setattr__(
        session,
        "binding",
        replace(original_binding, master_seed=original_binding.master_seed + 1),
    )
    with pytest.raises(IntegrityError, match="request/common binding changed"):
        driver.reverify_staged_bridge_prefix(session)
    object.__setattr__(session, "binding", original_binding)

    original_path = session.checkpoint_path
    object.__setattr__(session, "checkpoint_path", original_path.parent)
    with pytest.raises(IntegrityError, match="checkpoint path changed"):
        driver.reverify_staged_bridge_prefix(session)
    object.__setattr__(session, "checkpoint_path", original_path)

    original_materializations = session._rolling_materializations
    object.__setattr__(
        session,
        "_rolling_materializations",
        original_materializations[:1],
    )
    with pytest.raises(IntegrityError, match="lost a rolling materialization"):
        driver.reverify_staged_bridge_prefix(session)
    object.__setattr__(
        session,
        "_rolling_materializations",
        original_materializations,
    )


def test_prepare_staged_bridge_rejects_unregistered_or_mismatched_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, binding = _staged_request_and_binding()
    traces = MappingProxyType(
        {
            "model": np.zeros((1,), dtype=np.int64),
            "nonparametric": np.zeros((1,), dtype=np.int64),
        }
    )
    monkeypatch.setattr(driver, "_prepare_request", lambda _request: (binding, traces))
    monkeypatch.setattr(
        driver,
        "materialize_registered_rolling_bootstrap",
        lambda **_arguments: object(),
    )
    with pytest.raises(ModelError, match="not registered"):
        driver.prepare_canonical_staged_bridge(request, tmp_path / "unregistered")

    monkeypatch.setattr(
        driver,
        "materialize_registered_rolling_bootstrap",
        lambda **arguments: _cached_rolling(arguments["artifact"]),
    )
    mismatched = replace(
        binding,
        purpose1_trace_fingerprints=(("model", SHA_A), ("nonparametric", SHA_B)),
    )
    monkeypatch.setattr(
        driver,
        "_prepare_request",
        lambda _request: (mismatched, traces),
    )
    with pytest.raises(ModelError, match="differs from its binding"):
        driver.prepare_canonical_staged_bridge(request, tmp_path / "mismatched")


def test_compact_evidence_validators_fail_closed_on_semantic_drift() -> None:
    binding = _binding()
    tier_a_job = cast(
        driver._TierABatchJob,
        SimpleNamespace(dgp="model", pair_start=0, pair_stop=1),
    )
    tier_a = _tier_a_batch(binding, tier_a_job)[0]
    assert driver._validated_tier_a_pair(tier_a, binding, "model") is tier_a
    with pytest.raises(ModelError, match="wrong type"):
        driver._validated_tier_a_pair(cast(Any, object()), binding, "model")
    with pytest.raises(ModelError, match="identity is inconsistent"):
        driver._validated_tier_a_pair(
            driver._with_tier_a_fingerprint(
                replace(tier_a, pair=driver.TIER_A_PAIRS, evidence_fingerprint="")
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="outside K=2..5"):
        driver._validated_tier_a_pair(
            driver._with_tier_a_fingerprint(
                replace(
                    tier_a,
                    prefix_selected_n_states=9,
                    evidence_fingerprint="",
                )
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="failure evidence is malformed"):
        driver._validated_tier_a_pair(
            driver._with_tier_a_fingerprint(
                replace(tier_a, failure_types=("",), evidence_fingerprint="")
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="success flag is inconsistent"):
        driver._validated_tier_a_pair(
            driver._with_tier_a_fingerprint(
                replace(tier_a, success=False, evidence_fingerprint="")
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="evidence fingerprint is inconsistent"):
        driver._validated_tier_a_pair(
            replace(tier_a, evidence_fingerprint=SHA_E),
            binding,
            "model",
        )

    tier_b = _tier_b_pair(
        binding,
        dgp="model",
        pair=0,
        mode="accelerated_10",
        statistic=0.5,
    )
    assert driver._validated_tier_b_pair(tier_b, binding, "model") is tier_b
    with pytest.raises(ModelError, match="wrong type"):
        driver._validated_tier_b_pair(cast(Any, object()), binding, "model")
    with pytest.raises(ModelError, match="history fingerprint"):
        driver._validated_tier_b_pair(
            driver._with_tier_b_fingerprint(
                replace(tier_b, history_sha256="bad", evidence_fingerprint="")
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="valid Tier-B compact evidence"):
        driver._validated_tier_b_pair(
            driver._with_tier_b_fingerprint(
                replace(tier_b, statistic=None, evidence_fingerprint="")
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="omitted its failure"):
        driver._validated_tier_b_pair(
            driver._with_tier_b_fingerprint(
                replace(
                    tier_b,
                    valid=False,
                    statistic=None,
                    failure_type=None,
                    failure_message=None,
                    evidence_fingerprint="",
                )
            ),
            binding,
            "model",
        )
    with pytest.raises(ModelError, match="evidence fingerprint is inconsistent"):
        driver._validated_tier_b_pair(
            replace(tier_b, evidence_fingerprint=SHA_E),
            binding,
            "model",
        )

    benchmark_job = cast(
        driver._BenchmarkJob,
        SimpleNamespace(dgp="model", pair=0),
    )
    benchmark = _benchmark_pair(
        binding,
        benchmark_job,
        statistic=0.5,
        elapsed=1.0,
    )
    assert driver._validated_benchmark_pair(benchmark, binding) is benchmark
    with pytest.raises(ModelError, match="wrong type"):
        driver._validated_benchmark_pair(cast(Any, object()), binding)
    with pytest.raises(ModelError, match="must contain two booleans"):
        driver._validated_benchmark_pair(
            driver._with_benchmark_fingerprints(
                replace(
                    benchmark,
                    accelerated_valid=cast(Any, (True,)),
                    scientific_fingerprint="",
                    evidence_fingerprint="",
                )
            ),
            binding,
        )
    with pytest.raises(ModelError, match="two nonnegative values"):
        driver._validated_benchmark_pair(
            driver._with_benchmark_fingerprints(
                replace(
                    benchmark,
                    statistic_difference=(-1.0, 0.0),
                    scientific_fingerprint="",
                    evidence_fingerprint="",
                )
            ),
            binding,
        )
    with pytest.raises(ModelError, match="elapsed time"):
        driver._validated_benchmark_pair(
            driver._with_benchmark_fingerprints(
                replace(
                    benchmark,
                    elapsed_seconds=0.0,
                    scientific_fingerprint="",
                    evidence_fingerprint="",
                )
            ),
            binding,
        )
    with pytest.raises(ModelError, match="scientific fingerprint"):
        driver._validated_benchmark_pair(
            replace(benchmark, scientific_fingerprint=SHA_E),
            binding,
        )
    with pytest.raises(ModelError, match="evidence fingerprint"):
        driver._validated_benchmark_pair(
            replace(benchmark, evidence_fingerprint=SHA_E),
            binding,
        )


def test_serializers_round_trip_and_reject_inconsistent_resource_arithmetic() -> None:
    binding = _binding()
    tier_a_job = cast(
        driver._TierABatchJob,
        SimpleNamespace(dgp="model", pair_start=0, pair_stop=1),
    )
    tier_a = _tier_a_batch(binding, tier_a_job)[0]
    assert driver._tier_a_pair_from_dict(tier_a.as_dict()) == tier_a

    tier_b = _tier_b_pair(
        binding,
        dgp="model",
        pair=0,
        mode="accelerated_10",
        statistic=0.5,
    )
    assert driver._tier_b_pair_from_dict(tier_b.as_dict()) == tier_b
    bad_mode = tier_b.as_dict()
    bad_mode["mode"] = "fast"
    with pytest.raises(IntegrityError, match="fit mode is invalid"):
        driver._tier_b_pair_from_dict(bad_mode)

    benchmark_job = cast(
        driver._BenchmarkJob,
        SimpleNamespace(dgp="model", pair=0),
    )
    benchmark = _benchmark_pair(
        binding,
        benchmark_job,
        statistic=0.5,
        elapsed=1.0,
    )
    assert driver._benchmark_pair_from_dict(benchmark.as_dict()) == benchmark
    bad_nested = benchmark.as_dict()
    bad_nested["accelerated_evidence_fingerprint"] = SHA_E
    with pytest.raises(IntegrityError, match="nested evidence fingerprints"):
        driver._benchmark_pair_from_dict(bad_nested)

    measurement = _measurement()
    measurement_dict = driver._measurement_dict(measurement)
    assert driver._measurement_from_dict(measurement_dict) == measurement
    with pytest.raises(ModelError, match="measurement has the wrong type"):
        driver._measurement_dict(cast(Any, object()))

    bad_preflight = copy.deepcopy(measurement_dict)
    cast(dict[str, Any], bad_preflight["preflight"])["estimated_fraction"] = 0.2
    with pytest.raises(IntegrityError, match="preflight arithmetic"):
        driver._measurement_from_dict(bad_preflight)
    bad_measurement = copy.deepcopy(measurement_dict)
    bad_measurement["workers"] = 8
    with pytest.raises(IntegrityError, match="measurement arithmetic"):
        driver._measurement_from_dict(bad_measurement)

    projection = BridgeResourcePreflight(200, 9, 1.0, 0.01, 72.0, 0.70, True, ())
    projection_dict = driver._projection_dict(projection)
    assert driver._projection_from_dict(projection_dict) == projection
    with pytest.raises(ModelError, match="projection has the wrong type"):
        driver._projection_dict(cast(Any, object()))

    canonical = _canonical_resource(SimpleNamespace(fingerprint=SHA_A), projection)
    canonical_dict = {**canonical.identity_dict(), "fingerprint": canonical.fingerprint}
    assert driver._canonical_bridge_from_dict(canonical_dict) == canonical
    bad_schema = {**canonical_dict, "schema_version": 2}
    with pytest.raises(IntegrityError, match="schema is unsupported"):
        driver._canonical_bridge_from_dict(bad_schema)
    bad_fingerprint = {**canonical_dict, "fingerprint": SHA_E}
    with pytest.raises(IntegrityError, match="fingerprint is inconsistent"):
        driver._canonical_bridge_from_dict(bad_fingerprint)


def test_clean_scientific_values_and_closed_json_decoders_are_deterministic() -> None:
    class Example(Enum):
        VALUE = "value"

    array = np.array([-0.0, 1.0], dtype=np.float64)
    cleaned = cast(
        dict[str, Any],
        driver._clean_scientific_value(
            {
                "array": array,
                "scalar": np.float64(2.0),
                "enum": Example.VALUE,
                "path": Path("ignored"),
                "sequence": (1, True, None),
                "binding_fingerprint": SHA_A,
            }
        ),
    )
    assert cleaned["array"] == {
        "dtype": array.dtype.str,
        "shape": [2],
        "sha256": driver._array_sha256(array),
    }
    assert cleaned["scalar"] == 2.0
    assert cleaned["enum"] == "value"
    assert cleaned["path"] is None
    assert cleaned["sequence"] == [1, True, None]
    assert "binding_fingerprint" not in cleaned
    assert driver._clean_scientific_value(_measurement())["type"].endswith(
        "ExecutionResourceMeasurement"
    )

    with pytest.raises(ModelError, match="string keys"):
        driver._clean_scientific_value({1: "not allowed"})
    with pytest.raises(ModelError, match="non-finite scalar"):
        driver._clean_scientific_value(float("inf"))
    with pytest.raises(ModelError, match="unsupported value"):
        driver._clean_scientific_value(object())
    with pytest.raises(ModelError, match="unsupported array dtype"):
        driver._array_sha256(np.array(["x"], dtype=object))
    with pytest.raises(ModelError, match="not canonical JSON"):
        driver._canonical_bytes({"value": float("nan")})

    with pytest.raises(IntegrityError, match="string-keyed mapping"):
        driver._mapping([], "mapping")
    with pytest.raises(IntegrityError, match="JSON array"):
        driver._sequence((), "sequence")
    with pytest.raises(IntegrityError, match="only strings"):
        driver._string_sequence(["ok", 1], "strings")
    with pytest.raises(IntegrityError, match="unexpected or missing fields"):
        driver._exact_keys({"a": 1}, {"b"}, "object")
    with pytest.raises(IntegrityError, match="closed registry"):
        driver._dgp("other")
    with pytest.raises(IntegrityError, match="must be a string"):
        driver._string(1, "string")
    with pytest.raises(IntegrityError, match="must be boolean"):
        driver._boolean(1, "boolean")
    with pytest.raises(IntegrityError, match="must be an integer"):
        driver._integer(True, "integer")
    with pytest.raises(IntegrityError, match="must be numeric"):
        driver._float(True, "float")
    with pytest.raises(IntegrityError, match="must be finite"):
        driver._float(float("inf"), "float")
    with pytest.raises(IntegrityError, match="exactly two values"):
        driver._bool_pair([True], "pair")
    with pytest.raises(IntegrityError, match="must be boolean"):
        driver._bool_pair([True, 1], "pair")
    with pytest.raises(IntegrityError, match="exactly two values"):
        driver._float_pair([1.0], "pair")
    with pytest.raises(IntegrityError, match="is invalid"):
        driver._require_sha256("xyz", "digest", IntegrityError)
    with pytest.raises(ModelError, match="positive integer"):
        driver._positive_integer(0, "positive")
    with pytest.raises(ModelError, match="non-negative integer"):
        driver._nonnegative_integer(-1, "nonnegative")


def test_canonical_request_and_binding_validation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _binding_value = _staged_request_and_binding()
    with pytest.raises(ModelError, match="wrong type"):
        driver._prepare_request(cast(Any, object()))
    with pytest.raises(ModelError, match="canonical G3 preflight"):
        driver._prepare_request(replace(request, base_preflight=cast(Any, object())))
    with pytest.raises(ModelError, match="base preflight fingerprint"):
        driver._prepare_request(
            replace(
                request,
                base_preflight=replace(request.base_preflight, fingerprint="bad"),
            )
        )
    with pytest.raises(ModelError, match="evidence hash"):
        driver._prepare_request(
            replace(
                request,
                base_preflight=replace(request.base_preflight, fingerprint=SHA_A),
            )
        )

    unsealed_base = replace(request.base_preflight, workers=8, fingerprint="")
    wrong_worker_base = replace(
        unsealed_base,
        fingerprint=driver._sha256_json(unsealed_base.identity_dict()),
    )
    with pytest.raises(ModelError, match="bind nine workers"):
        driver._prepare_request(replace(request, base_preflight=wrong_worker_base))
    with pytest.raises(ModelError, match="non-negative integer"):
        driver._prepare_request(replace(request, master_seed=-1))
    with pytest.raises(ModelError, match="positive integer"):
        driver._prepare_request(replace(request, scientific_version=0))

    one_state = HMMParameters(
        np.ones(1),
        np.ones((1, 1)),
        np.zeros((1, 4)),
        np.eye(4),
    )
    with pytest.raises(ModelError, match="selected K"):
        driver._prepare_request(replace(request, design_parameters=one_state))
    with pytest.raises(ModelError, match="exact gap-free"):
        driver._prepare_request(
            replace(request, design_standardized_history=np.zeros((192, 4)))
        )
    with pytest.raises(ModelError, match="exceeds design history"):
        driver._prepare_request(
            replace(request, macro_block_length=driver.BRIDGE_PREFIX_ROWS + 1)
        )

    malformed_trace = np.zeros((1,), dtype=np.int64)
    malformed_trace.setflags(write=False)
    monkeypatch.setattr(
        driver,
        "materialize_rolling_bootstrap_indices",
        lambda **_arguments: malformed_trace,
    )
    with pytest.raises(ModelError, match="purpose-1 bridge trace is malformed"):
        driver._prepare_request(request)

    binding = _binding()
    driver._validate_binding(binding)
    with pytest.raises(ModelError, match="wrong type"):
        driver._validate_binding(cast(Any, object()))
    with pytest.raises(ModelError, match="non-negative integer"):
        driver._validate_binding(replace(binding, master_seed=-1))
    with pytest.raises(ModelError, match="positive integer"):
        driver._validate_binding(replace(binding, scientific_version=0))
    with pytest.raises(ModelError, match="selected K"):
        driver._validate_binding(replace(binding, selected_n_states=9))
    with pytest.raises(ModelError, match="actual statistic"):
        driver._validate_binding(replace(binding, actual_statistic=float("inf")))
    with pytest.raises(ModelError, match="alpha-grid order"):
        driver._validate_binding(
            replace(
                binding, alpha_offset_grids=tuple(reversed(binding.alpha_offset_grids))
            )
        )
    with pytest.raises(ModelError, match="trace order"):
        driver._validate_binding(
            replace(
                binding,
                purpose1_trace_fingerprints=tuple(
                    reversed(binding.purpose1_trace_fingerprints)
                ),
            )
        )
    with pytest.raises(ModelError, match="trace fingerprint"):
        driver._validate_binding(
            replace(
                binding,
                purpose1_trace_fingerprints=(
                    ("model", "bad"),
                    ("nonparametric", SHA_E),
                ),
            )
        )
    with pytest.raises(ModelError, match="binding fingerprint"):
        driver._validate_binding(replace(binding, fingerprint=SHA_E))


def test_bound_alpha_grids_reject_role_shape_and_cross_frequency_drift() -> None:
    request, _binding_value = _staged_request_and_binding()
    grids = request.alpha_offset_grids
    with pytest.raises(ModelError, match="exactly four"):
        driver._bound_alpha_grids(grids[:3], 2)
    with pytest.raises(ModelError, match="wrong role/frequency"):
        driver._bound_alpha_grids(
            (replace(grids[0], frequency="daily"), *grids[1:]),
            2,
        )
    with pytest.raises(ModelError, match="model artifact fingerprint"):
        driver._bound_alpha_grids(
            (replace(grids[0], model_artifact_fingerprint="bad"), *grids[1:]),
            2,
        )

    mutable_offsets = np.asarray(grids[0].offsets).copy()
    with pytest.raises(ModelError, match="immutable exact"):
        driver._bound_alpha_grids(
            (replace(grids[0], offsets=mutable_offsets), *grids[1:]),
            2,
        )

    changed_daily = replace(
        grids[1],
        model_artifact_fingerprint=SHA_E,
        grid_fingerprint="",
    )
    changed_daily = replace(
        changed_daily,
        grid_fingerprint=driver._alpha_grid_fingerprint(
            changed_daily,
            np.asarray(changed_daily.lambdas),
            np.asarray(changed_daily.offsets),
        ),
    )
    with pytest.raises(ModelError, match="different model artifacts"):
        driver._bound_alpha_grids(
            (grids[0], changed_daily, grids[2], grids[3]),
            2,
        )
