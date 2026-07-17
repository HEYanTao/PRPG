from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import prpg.model.g3_assembly as assembly_module
import prpg.model.g3_boundary_views as boundary_module
import prpg.model.g3_coordinator as coordinator_module
import prpg.model.g3_science_handlers as science_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.alpha import KernelMeanPrecision
from prpg.model.benchmarks import HoldoutBenchmarkScores, holdout_veto_decision
from prpg.model.block_execution import (
    BlockCalibrationExecution,
    CalibrationGeometry,
    PooledBlockStatistics,
    PooledFragmentationGate,
)
from prpg.model.block_execution_registered import RegisteredBlockCalibration
from prpg.model.block_length import BlockPolicySelection, select_supported_length
from prpg.model.bridge import TIER_B_INTERVAL_ERROR
from prpg.model.calibration import (
    CandidateFitAttempt,
    CandidateFitSet,
    DecodedReturnPools,
    MacroBootstrapAttempt,
    MacroStabilityEvidence,
    RollingCandidateEvidence,
    RollingFitAttempt,
    SealedHoldoutEvidence,
)
from prpg.model.candidate_gate import (
    CANONICAL_CANDIDATES,
    CandidateGateEvidence,
    CandidateGateResult,
    run_candidate_gate,
)
from prpg.model.g3_assembly import (
    ArtifactIntegrityDecisionEvidence,
    BlockCalibrationGateEvidence,
    BridgeDriverTaskViews,
    G3TaskFailure,
    InputIntegrityDecisionEvidence,
)
from prpg.model.g3_checkpoint import CalibrationCheckpointStore
from prpg.model.g3_coordinator import coordinate_g3_canonical_launch
from prpg.model.g3_preflight import CanonicalG3Preflight
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.g3_science_handlers import (
    CanonicalG3First25HandlerBundle,
    CanonicalG3HandlerBundle,
    canonical_g3_scientific_report_payload,
    canonical_planned_looks,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HistoricalTransitionSupportReport,
    summarize_state_path,
)
from prpg.model.inference import clopper_pearson_interval
from prpg.model.kernel_execution import (
    CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
    KernelCalibrationExecution,
    KernelPrecisionLook,
)
from prpg.model.run_store import CalibrationRunStore, default_run_identity
from prpg.model.scientific_artifact import (
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    required_scientific_documents,
)
from prpg.model.stability import MacroStabilitySummary, RollingStabilitySummary
from prpg.model.transition_bootstrap import (
    TransitionBootstrapAggregate,
    TransitionBootstrapAttemptAudit,
)
from prpg.simulation.rng import ModelFitScope


class _Covariance:
    symmetric = True
    positive_definite = True
    eigenvalue_floor_passed = True
    condition_number_passed = True
    condition_number = 1_000_000.0


class _Chain:
    irreducible = True
    aperiodic = True
    unique_stationary_distribution = True
    period = 1
    stationary_residual = 1e-12
    near_absorbing_states: tuple[int, ...] = ()
    mixing_time_to_one_percent = 120


class _Passed:
    passed = True


class _Fit:
    def __init__(self, n_states: int, states: NDArray[np.int64], bic: float) -> None:
        self.n_states = n_states
        self.n_features = 4
        self.n_observations = len(states)
        self.bic = bic
        self.log_likelihood = -100.0
        self.decoded_states = states
        self.covariance_diagnostic = _Covariance()
        self.chain_diagnostics = _Chain()
        self.identification = _Passed()
        self.historical_transition_support = _Passed()


def _preflight() -> CanonicalG3Preflight:
    provisional = CanonicalG3Preflight(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        contract_fingerprint="1" * 64,
        source_code_fingerprint="2" * 64,
        dependency_lock_fingerprint="3" * 64,
        toolchain_fingerprint="4" * 64,
        ppw_vendored_fingerprint="5" * 64,
        ppw_upstream_fingerprint="6" * 64,
        task_registry_fingerprint=G3_TASK_REGISTRY_FINGERPRINT,
        planned_look_registry_fingerprint=G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
        logical_cpu_count=10,
        physical_memory_bytes=17_179_869_184,
        disk_free_bytes=80_000_000_000,
        workers=9,
        estimated_peak_bytes=1_000_000,
        estimated_disk_bytes=1_000_000,
        resource_estimate_fingerprint="7" * 64,
        checks=("contract", "input", "resources"),
        fingerprint="",
    )
    fingerprint = hashlib.sha256(
        (
            json.dumps(
                provisional.identity_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return replace(provisional, fingerprint=fingerprint)


def _changed_disk_preflight(
    launch: CanonicalG3Preflight,
) -> CanonicalG3Preflight:
    provisional = replace(
        launch,
        disk_free_bytes=launch.disk_free_bytes - 1_000_000,
        fingerprint="",
    )
    fingerprint = hashlib.sha256(
        (
            json.dumps(
                provisional.identity_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return replace(provisional, fingerprint=fingerprint)


def _run(
    tmp_path: Path,
) -> tuple[CalibrationRunStore, str, CanonicalG3Preflight]:
    preflight = _preflight()
    store = CalibrationRunStore(tmp_path / "runs")
    identity = default_run_identity(
        config_fingerprint=preflight.config_fingerprint,
        processed_data_fingerprint=preflight.processed_data_fingerprint,
        calibration_input_fingerprint=preflight.calibration_input_fingerprint,
        source_code_fingerprint=preflight.source_code_fingerprint,
        dependency_lock_fingerprint=preflight.dependency_lock_fingerprint,
        workers=9,
    )
    run_id = store.initialize(identity).run_id
    store.plan_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    return store, run_id, preflight


def _input(
    launch_preflight: CanonicalG3Preflight,
    run_id: str,
    *,
    live_preflight: CanonicalG3Preflight | None = None,
) -> boundary_module.InputIntegrityTaskView:
    binding = ScientificArtifactBinding(
        config_fingerprint=launch_preflight.config_fingerprint,
        processed_data_fingerprint=launch_preflight.processed_data_fingerprint,
        calibration_input_fingerprint=launch_preflight.calibration_input_fingerprint,
        calibration_run_fingerprint=run_id,
        preflight_fingerprint=launch_preflight.fingerprint,
    )
    return boundary_module.build_input_integrity_task_view(
        InputIntegrityDecisionEvidence(
            launch_preflight,
            live_preflight or launch_preflight,
            binding,
        ),
        run_id=run_id,
        binding_fingerprint=hashlib.sha256(
            f"test-input-integrity:{run_id}".encode()
        ).hexdigest(),
    )


def _failure(task_id: str) -> G3TaskFailure:
    return G3TaskFailure(task_id, "SyntheticScientificFailure", f"{task_id} failed")


def _handlers(
    input_source: object,
    **replacements: object,
) -> CanonicalG3HandlerBundle:
    values: dict[str, object] = {
        task_id: _failure(task_id) for task_id in G3_GATE_ORDER
    }
    values["input_integrity"] = input_source
    values.update(replacements)
    return CanonicalG3HandlerBundle(**values)  # type: ignore[arg-type]


def _first_25_handlers(
    input_source: object,
    **replacements: object,
) -> CanonicalG3First25HandlerBundle:
    values: dict[str, object] = {
        task_id: _failure(task_id) for task_id in G3_GATE_ORDER[:-1]
    }
    values["input_integrity"] = input_source
    values.update(replacements)
    return CanonicalG3First25HandlerBundle(**values)  # type: ignore[arg-type]


def _states(n_states: int, rows: int, run_length: int) -> NDArray[np.int64]:
    return np.resize(np.repeat(np.arange(n_states), run_length), rows).astype(np.int64)


def _policy(
    *,
    frequency: str,
    states: NDArray[np.int64],
    n_states: int,
) -> BlockPolicySelection:
    length = 6 if frequency == "monthly" else 60
    return select_supported_length(
        starting_length=length,
        frequency=frequency,  # type: ignore[arg-type]
        states=states,
        continuation=np.ones(len(states) - 1, dtype=np.bool_),
        realized_means_by_length={length: dict.fromkeys(range(n_states), length / 2)},
        selected_states=tuple(range(n_states)),
    )


def _candidate_gate() -> CandidateGateResult:
    attempts: list[CandidateFitAttempt] = []
    evidence: list[CandidateGateEvidence] = []
    for n_states in CANONICAL_CANDIDATES:
        monthly = _states(n_states, 193, 10)
        daily = _states(n_states, 4_049, 100)
        fit = cast(
            GaussianHMMFit,
            _Fit(n_states, monthly, 100.0 + 4.0 * (n_states - 2)),
        )
        pools = DecodedReturnPools(
            monthly,
            daily,
            summarize_state_path(monthly, n_states, lengths=(193,)),
            summarize_state_path(daily, n_states, lengths=(4_049,)),
        )
        rolling_attempts = tuple(
            RollingFitAttempt(
                fold=fold,
                training_rows=121 + 12 * fold,
                validation_rows=12,
                validation_lengths=(12,),
                fit=fit,
                scores=(2.0 - n_states / 10.0,) * 12,
                adjusted_rand_index=0.80,
                jaccards=np.full(n_states, 0.70),
                failure_type=None,
                failure_message=None,
            )
            for fold in range(6)
        )
        rolling = RollingCandidateEvidence(
            n_states=n_states,
            role="design",
            attempts=rolling_attempts,
            stability=RollingStabilitySummary(
                successful_fits=6,
                median_ari=0.80,
                minimum_successful_ari=0.50,
                median_jaccard_by_state=np.full(n_states, 0.70),
                passed=True,
            ),
        )
        macro_attempts = tuple(
            MacroBootstrapAttempt(
                attempt=index,
                source_indices=np.arange(193, dtype=np.int64),
                fit=fit,
                adjusted_rand_index=0.85,
                jaccards=np.full(n_states, 0.75),
                failure_type=None,
                failure_message=None,
            )
            for index in range(100)
        )
        macro = MacroStabilityEvidence(
            n_states=n_states,
            role="design",
            macro_block_length=2,
            attempts=macro_attempts,
            summary=MacroStabilitySummary(
                successful_attempts=100,
                median_ari=0.85,
                tenth_percentile_ari=0.60,
                median_jaccard_by_state=np.full(n_states, 0.75),
                tenth_percentile_jaccard_by_state=np.full(n_states, 0.60),
                passed=True,
            ),
        )
        attempts.append(CandidateFitAttempt(n_states, fit, None, None, None))
        evidence.append(
            CandidateGateEvidence(
                n_states=n_states,
                decoded_return_pools=pools,
                rolling=rolling,
                macro_stability=macro,
                monthly_block_policy=_policy(
                    frequency="monthly", states=monthly, n_states=n_states
                ),
                daily_block_policy=_policy(
                    frequency="daily", states=daily, n_states=n_states
                ),
            )
        )
    indices = np.broadcast_to(np.arange(12, dtype=np.int64), (10_000, 6, 12)).copy()
    return run_candidate_gate(
        CandidateFitSet(ModelFitScope.DESIGN_MAIN, 0, tuple(attempts)),
        tuple(evidence),
        role="design",
        registered_bootstrap_indices=indices,
    )


def _holdout() -> SealedHoldoutEvidence:
    hmm = np.full(24, 2.0)
    benchmarks = HoldoutBenchmarkScores(np.ones(24), np.full(24, 1.5))
    return SealedHoldoutEvidence(
        (17, 7), hmm, benchmarks, holdout_veto_decision(hmm, benchmarks)
    )


def _kernel(
    frequency: str,
    widths: tuple[float, ...],
) -> KernelCalibrationExecution:
    looks: list[KernelPrecisionLook] = []
    schedule = (10_000, 20_000, 40_000, 80_000, 160_000, 320_000)
    for index, width in enumerate(widths):
        windows = schedule[index]
        precision = KernelMeanPrecision(
            frequency=frequency,  # type: ignore[arg-type]
            model_artifact_fingerprint="a" * 64,
            calibration_stream_fingerprint="b" * 64,
            source_window_fingerprint="c" * 64,
            window_statistics_fingerprint="d" * 64,
            kernel_sample_fingerprint="e" * 64,
            sample_stage="unadjusted_kernel",
            windows=windows,
            periods_per_year=12 if frequency == "monthly" else 252,
            confidence=0.95,
            degrees_freedom=windows - 1,
            critical_value=2.0,
            simultaneous_comparisons=6,
            state_window_counts=np.full(2, windows, dtype=np.int64),
            state_observation_counts=np.full(2, windows, dtype=np.int64),
            means=np.zeros((2, 3)),
            cluster_standard_errors=np.zeros((2, 3)),
            annualized_means=np.zeros((2, 3)),
            annualized_simultaneous_half_widths=np.full((2, 3), width),
        )
        looks.append(
            KernelPrecisionLook(
                look_index=index,
                windows=windows,
                precision=precision,
                maximum_annualized_half_width=width,
                passed=width <= CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
            )
        )
    terminal_pass = looks[-1].passed
    return KernelCalibrationExecution(
        artifact="design",
        frequency=frequency,  # type: ignore[arg-type]
        geometry=CalibrationGeometry(
            "design",
            frequency,  # type: ignore[arg-type]
            (193,),
            (193,),
            True,
        ),
        n_states=2,
        selected_length=2,
        model_artifact_fingerprint="a" * 64,
        target_history_fingerprint="f" * 64,
        calibration_stream_fingerprint="b" * 64,
        draw_order=(
            "monthly_state_uniforms",
            "restart_uniforms",
            "source_rank_uniforms",
        ),
        draws_per_window=579,
        planned_looks=schedule,
        completed_looks=tuple(looks),
        stopped_windows=looks[-1].windows,
        precision_passed=terminal_pass,
        verification_passed=terminal_pass,
        passed=terminal_pass,
        final_precision=looks[-1].precision,
        offset_grid=None,
        verification=None,
        failure_reason=None if terminal_pass else "precision_failed",
        purpose_7_streams=looks[-1].windows,
        purpose_8_streams=0,
        first_purpose_7_entity=1,
        last_purpose_7_entity=looks[-1].windows,
        workers=9,
        statistics_batch_windows=128,
        process_chunksize=1,
        strict_canonical=True,
    )


def _block_source(
    monthly: KernelCalibrationExecution,
    daily: KernelCalibrationExecution,
) -> BlockCalibrationGateEvidence:
    return BlockCalibrationGateEvidence(
        role="design",
        selected_k=2,
        monthly=cast(Any, object()),
        daily=cast(Any, object()),
        monthly_kernel=monthly,
        daily_kernel=daily,
        transition=cast(Any, object()),
    )


def _failed_design_block_source() -> BlockCalibrationGateEvidence:
    """Typed terminal kernel failure with exact registered look counts."""

    registered: dict[str, RegisteredBlockCalibration] = {}
    kernels: dict[str, KernelCalibrationExecution] = {}
    for frequency, selected_length, target_lengths in (
        ("monthly", 6, (193,)),
        ("daily", 60, (4_049,)),
    ):
        policy = BlockPolicySelection(
            frequency=frequency,  # type: ignore[arg-type]
            selected_length=selected_length,
            restart_probability=1.0 / selected_length,
            trials=(),
        )
        gates: list[PooledFragmentationGate] = []
        for state in range(2):
            statistics = PooledBlockStatistics(
                state=state,
                count=100,
                mean=float(selected_length),
                minimum=1,
                maximum=selected_length,
                median=float(selected_length),
                singleton_fraction=0.0,
                quantile_05=1.0,
                quantile_95=float(selected_length),
                length_histogram=(0, 100),
            )
            gates.append(
                PooledFragmentationGate(
                    state=state,
                    intended_length=selected_length,
                    conditioned=statistics,
                    ideal=statistics,
                    conditioned_mean_minimum=1.0,
                    conditioned_median_minimum=1.0,
                    singleton_fraction_maximum_from_ideal=0.1,
                    absolute_singleton_fraction_maximum=0.1,
                    reasons=(),
                )
            )
        execution = BlockCalibrationExecution(
            geometry=CalibrationGeometry(
                "design",
                frequency,  # type: ignore[arg-type]
                (193,),
                target_lengths,
                True,
            ),
            histories=10_000,
            starting_length=selected_length,
            structurally_feasible_lengths=(selected_length,),
            simulations=(),
            policy_selection=policy,
            dependence_reuse_lengths=(selected_length,),
            selected_fragmentation_gates=tuple(gates),
        )
        registered[frequency] = RegisteredBlockCalibration(
            execution=execution,
            history_stream_sets=10_000,
            state_uniforms_per_history=193,
            restart_uniforms_per_history=sum(target_lengths),
            source_rank_uniforms_per_history=sum(target_lengths),
            uniform_chunk_size=128,
        )
        kernels[frequency] = replace(
            _kernel(
                frequency,
                (2.0 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,) * 6,
            ),
            selected_length=selected_length,
        )

    support = HistoricalTransitionSupportReport(
        probabilities=np.full((2, 2), 0.5),
        viterbi_counts=np.ones((2, 2), dtype=np.int64),
        expected_counts=np.full((2, 2), 2.0),
        supported_cells=np.ones((2, 2), dtype=np.bool_),
        supported_off_diagonal_edges=np.ones((2, 2), dtype=np.bool_),
        supported_self_loop_states=(0, 1),
        decoded_entries=np.ones(2, dtype=np.int64),
        decoded_exits=np.ones(2, dtype=np.int64),
        bootstrap_edge_frequencies=np.ones((2, 2)),
        bootstrap_probability_quantiles=np.full((2, 2, 3), 0.5),
        strongly_connected=True,
        aperiodic=True,
        passed=True,
        rejection_reasons=(),
    )
    transition = TransitionBootstrapAggregate(
        n_states=2,
        attempts=100,
        successful_refits=90,
        failed_refits=10,
        edge_frequency_denominator=100,
        quantile_probabilities=(0.05, 0.50, 0.95),
        quantile_method="higher",
        bootstrap_edge_frequencies=np.ones((2, 2)),
        bootstrap_probability_quantiles=np.full((2, 2, 3), 0.5),
        historical_transition_support=support,
        attempt_audits=tuple(
            TransitionBootstrapAttemptAudit(index, False, None, 0, None)
            for index in range(100)
        ),
    )
    return BlockCalibrationGateEvidence(
        role="design",
        selected_k=2,
        monthly=registered["monthly"],
        daily=registered["daily"],
        monthly_kernel=kernels["monthly"],
        daily_kernel=kernels["daily"],
        transition=transition,
    )


def _permit_test_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        boundary_module,
        "verify_compatible_canonical_g3_preflights",
        lambda launch, _live: launch.fingerprint,
    )
    monkeypatch.setattr(
        science_module,
        "verify_live_canonical_g3_preflight",
        lambda _value: None,
    )
    monkeypatch.setattr(
        coordinator_module,
        "canonical_resume_preflight_fingerprint",
        lambda *_args, **_kwargs: _args[0]
        .read_launch(_kwargs["run_id"])
        .preflight_fingerprint,
    )
    monkeypatch.setattr(
        science_module,
        "canonical_resume_preflight_fingerprint",
        lambda *_args, **_kwargs: _args[0]
        .read_launch(_kwargs["run_id"])
        .preflight_fingerprint,
    )


def _synthetic_passing_first_25(
    input_source: boundary_module.InputIntegrityTaskView,
    passing_bridge_views: BridgeDriverTaskViews,
) -> CanonicalG3First25HandlerBundle:
    shared_candidate = object()
    shared_adequacy = object()
    shared_dependence = object()
    design_block = _block_source(
        _kernel("monthly", (0.5 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
        _kernel("daily", (0.5 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
    )
    production_block = _block_source(
        _kernel("monthly", (0.5 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
        _kernel("daily", (0.5 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
    )
    bridge_sources: dict[str, Any] = {
        task_id: view.source
        for task_id, view in passing_bridge_views.as_mapping().items()
    }
    aligned_bridge: dict[str, object] = {
        task_id: boundary_module.build_bridge_driver_task_view(
            bridge_sources[task_id],
            task_id=cast(Any, task_id),
            run_id=input_source.run_id,
            binding_fingerprint=hashlib.sha256(
                f"synthetic-bridge:{input_source.run_id}:{task_id}".encode()
            ).hexdigest(),
        )
        for task_id in boundary_module.BRIDGE_TASK_IDS
    }
    values: dict[str, object] = {task_id: object() for task_id in G3_GATE_ORDER[:-1]}
    values.update(
        {
            "input_integrity": input_source,
            "design_candidate_viability": shared_candidate,
            "design_predictive_selection": shared_candidate,
            "design_block_calibration": design_block,
            "design_fragmentation": design_block,
            "production_block_calibration": production_block,
            "production_fragmentation": production_block,
            "design_latent_correctness": shared_adequacy,
            "design_refitted_adequacy": shared_adequacy,
            "production_latent_correctness": shared_adequacy,
            "production_refitted_adequacy": shared_adequacy,
            "four_cell_adequacy_holm": shared_adequacy,
            "design_dependence": shared_dependence,
            "production_dependence": shared_dependence,
            "four_cell_dependence_holm": shared_dependence,
            **aligned_bridge,
        }
    )
    return CanonicalG3First25HandlerBundle(**values)  # type: ignore[arg-type]


def _permit_synthetic_passing_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    original_planned_looks = science_module.canonical_planned_looks

    def planned_looks(
        task_id: str,
        source: object,
    ) -> tuple[science_module.CanonicalG3PlannedLook, ...]:
        if task_id in {
            "design_block_calibration",
            "production_block_calibration",
        } and isinstance(source, BlockCalibrationGateEvidence):
            specs = tuple(
                item
                for item in science_module.G3_PLANNED_LOOK_REGISTRY
                if item.owner_task_id == task_id
            )
            result: list[science_module.CanonicalG3PlannedLook] = []
            for spec, kernel in zip(
                specs,
                (source.monthly_kernel, source.daily_kernel),
                strict=True,
            ):
                result.extend(
                    science_module._kernel_looks(
                        spec.family_id,
                        spec.sample_counts,
                        kernel,
                    )
                )
            return tuple(result)
        return original_planned_looks(task_id, cast(Any, source))

    monkeypatch.setattr(
        science_module,
        "_SOURCE_TYPE_BY_TASK",
        MappingProxyType(dict.fromkeys(G3_GATE_ORDER, object)),
    )
    monkeypatch.setattr(
        assembly_module,
        "_evaluate",
        lambda gate_id, _source: (
            True,
            {"synthetic_task": gate_id},
            {"selected_k": 2},
        ),
    )
    monkeypatch.setattr(
        assembly_module,
        "_cross_binding_error",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        science_module,
        "_is_valid_task_scoped_view",
        lambda _task_id, _source: True,
    )
    monkeypatch.setattr(
        science_module,
        "task_view_cross_binding_error",
        lambda _sources: None,
    )
    monkeypatch.setattr(science_module, "canonical_planned_looks", planned_looks)


def _permit_legacy_coordinator_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep old compact doubles scoped to coordinator-only regression tests."""

    original = assembly_module._evaluate
    original_planned_looks = science_module.canonical_planned_looks

    def evaluate(
        task_id: str, source: object
    ) -> tuple[bool, dict[str, object], dict[str, object]]:
        if isinstance(source, CandidateGateResult):
            selected = source.selection.selected_n_states
            passed = (
                any(row.viability.admissible for row in source.audit_rows)
                if task_id == "design_candidate_viability"
                else source.selection.passed
            )
            summary: dict[str, object] = {"status": "passed" if passed else "failed"}
            if task_id == "design_predictive_selection":
                summary["selected_k"] = selected
            return bool(passed), {"legacy_test_double": task_id}, summary
        if isinstance(source, SealedHoldoutEvidence):
            return (
                source.decision.passed,
                {"legacy_test_double": task_id},
                {
                    "minimum_score_advantage": min(
                        source.decision.hmm_minus_gaussian,
                        source.decision.hmm_minus_ridge_var1,
                    )
                },
            )
        if task_id in {
            "design_block_calibration",
            "production_block_calibration",
        } and isinstance(source, BlockCalibrationGateEvidence):
            passed = source.monthly_kernel.passed and source.daily_kernel.passed
            return (
                passed,
                {"legacy_test_double": task_id},
                {"selected_k": source.selected_k},
            )
        if task_id in {
            "design_fragmentation",
            "production_fragmentation",
        } and isinstance(source, BlockCalibrationGateEvidence):
            return (
                True,
                {"legacy_test_double": task_id},
                {"selected_k": source.selected_k},
            )
        return original(task_id, source)

    def planned_looks(
        task_id: str,
        source: object,
    ) -> tuple[science_module.CanonicalG3PlannedLook, ...]:
        if task_id in {
            "design_block_calibration",
            "production_block_calibration",
        } and isinstance(source, BlockCalibrationGateEvidence):
            specs = tuple(
                item
                for item in science_module.G3_PLANNED_LOOK_REGISTRY
                if item.owner_task_id == task_id
            )
            result: list[science_module.CanonicalG3PlannedLook] = []
            for spec, kernel in zip(
                specs,
                (source.monthly_kernel, source.daily_kernel),
                strict=True,
            ):
                result.extend(
                    science_module._kernel_looks(
                        spec.family_id,
                        spec.sample_counts,
                        kernel,
                    )
                )
            return tuple(result)
        return original_planned_looks(task_id, cast(Any, source))

    monkeypatch.setattr(
        science_module,
        "_SOURCE_TYPE_BY_TASK",
        MappingProxyType(dict.fromkeys(G3_GATE_ORDER, object)),
    )
    monkeypatch.setattr(
        science_module,
        "_is_valid_task_scoped_view",
        lambda _task_id, _source: True,
    )
    monkeypatch.setattr(assembly_module, "_evaluate", evaluate)
    monkeypatch.setattr(science_module, "canonical_planned_looks", planned_looks)


def _synthetic_loaded_artifact(
    prefix: science_module.CanonicalG3ArtifactPublicationPrefix,
    role: str,
) -> SimpleNamespace:
    reports: dict[str, object] = {}
    for path, document_type in required_scientific_documents(cast(Any, role)).items():
        reports[path] = {
            "payload": canonical_g3_scientific_report_payload(
                prefix,
                document_type=document_type,
                role=cast(Any, role),
                metrics={"synthetic": 1},
            )
        }
    return SimpleNamespace(
        role=role,
        selected_k=prefix.selected_n_states,
        binding=prefix.binding,
        provenance=prefix.provenance,
        geometry=SimpleNamespace(as_dict=lambda: {"role": role}),
        generation_policy={"policy": "synthetic"},
        arrays={"weights": np.asarray([0.25, 0.75])},
        reports=reports,
        reference=SimpleNamespace(fingerprint=("d" if role == "design" else "e") * 64),
        core_model_fingerprint=("a" if role == "design" else "b") * 64,
        parameter_set_fingerprint=("c" if role == "design" else "f") * 64,
        generation_authorized=False,
    )


def test_closed_handlers_publish_checkpoint_for_every_passing_prefix_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_legacy_coordinator_fixtures(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    gate = _candidate_gate()
    failed_block = _failed_design_block_source()
    handlers = _handlers(
        _input(preflight, run_id),
        design_candidate_viability=gate,
        design_predictive_selection=gate,
        sealed_holdout_veto=_holdout(),
        design_block_calibration=failed_block,
        design_fragmentation=failed_block,
    )
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")

    with store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    ) as lease:
        result = coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )

    assert tuple(item.task_id for item in result.task_results) == G3_GATE_ORDER
    assert [item.status for item in result.task_results[:5]] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "failed",
    ]
    assert tuple(item.task_id for item in result.checkpoints) == G3_GATE_ORDER[:4]
    assert result.selected_n_states == gate.selection.selected_n_states == 2
    assert result.failed_task_ids == (
        "design_macro_stability",
        "design_latent_correctness",
        "design_refitted_adequacy",
        "design_block_calibration",
        "production_fit_same_k",
    )
    assert not result.generation_authorized
    assert not result.evidence_bundle.generation_authorized
    for checkpoint in result.checkpoints:
        receipt = store.read_task_result(run_id, checkpoint.task_id)
        assert receipt.payload["checkpoint_fingerprint"] == checkpoint.fingerprint
        assert receipt.payload["execution_mode"] == "canonical"
        assert not checkpoint.generation_authorized
    looks = store.verify_planned_look_receipts(run_id, require_terminal=True)
    assert set(looks) == {"kernel_design_monthly", "kernel_design_daily"}
    assert all(receipts[-1].decision == "fail" for receipts in looks.values())


def test_two_stage_rejects_early_artifacts_and_failed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    full = _handlers(_input(preflight, run_id))
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    with pytest.raises(ModelError, match="first-25 handler bundle"):
        science_module.coordinate_canonical_g3_first_25(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=full,  # type: ignore[arg-type]
        )
    assert store.verify_task_results(run_id, require_complete=False) == {}

    prefix = science_module.coordinate_canonical_g3_first_25(
        store=store,
        lease=lease,
        live_preflight=preflight,
        checkpoint_store=checkpoint_store,
        handlers=_first_25_handlers(_input(preflight, run_id)),
    )
    assert store.read_launch(run_id).state == "running"
    assert not prefix.artifact_publication_ready
    assert prefix.failed_task_ids == ("design_candidate_viability",)
    assert prefix.blocked_task_ids == G3_GATE_ORDER[2:-1]
    with pytest.raises(ModelError, match="failed or blocked"):
        canonical_g3_scientific_report_payload(
            prefix,
            document_type="model_fit",
            role="design",
            metrics={"never": True},
        )
    with pytest.raises(ModelError, match="failed or blocked"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=cast(Any, object()),
        )
    assert "artifact_integrity" not in store.verify_task_results(
        run_id,
        require_complete=False,
    )
    lease.close()


def test_two_stage_report_binding_finalization_and_completed_prefix_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_synthetic_passing_decisions(monkeypatch)
    monkeypatch.setattr(
        science_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "is_verified_scientific_artifact_pair",
        lambda _pair: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "scientific_artifact_pair_identity",
        lambda pair: {
            "actual_bridge_parent_view_fingerprint": (
                pair.actual_bridge_parent_view_fingerprint
            )
        },
    )
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    handlers = _synthetic_passing_first_25(
        _input(preflight, run_id),
        passing_bridge_views,
    )
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    prefix = science_module.coordinate_canonical_g3_first_25(
        store=store,
        lease=lease,
        live_preflight=preflight,
        checkpoint_store=checkpoint_store,
        handlers=handlers,
    )
    assert prefix.artifact_publication_ready
    assert len(prefix.task_results) == len(prefix.checkpoints) == 25
    assert store.read_launch(run_id).state == "running"
    assert (
        tuple(store.verify_task_results(run_id, require_complete=False))
        == (G3_GATE_ORDER[:-1])
    )

    design = _synthetic_loaded_artifact(prefix, "design")
    production = _synthetic_loaded_artifact(prefix, "production")

    def artifact_pair(design_value: object, production_value: object) -> Any:
        return SimpleNamespace(
            _prefix=prefix,
            design=design_value,
            production=production_value,
            run_id=prefix.run_id,
            selected_k=prefix.selected_n_states,
            prefix_fingerprint=prefix.prefix_fingerprint,
            reference=SimpleNamespace(fingerprint="9" * 64),
            actual_bridge_parent_view_fingerprint=(
                prefix._handlers.actual_artifact_bridge.view_fingerprint
            ),
            design_role_source_fingerprint="a" * 64,
            production_role_source_fingerprint="b" * 64,
            generation_authorized=False,
        )

    def pair_identity(pair: object) -> Mapping[str, object]:
        return {
            "task_result_fingerprints": dict(prefix.task_result_fingerprints),
            "checkpoint_fingerprints": dict(prefix.checkpoint_fingerprints),
            "actual_bridge_parent_view_fingerprint": cast(
                Any, pair
            ).actual_bridge_parent_view_fingerprint,
        }

    monkeypatch.setattr(
        science_module,
        "is_verified_scientific_artifact_pair",
        lambda _pair: True,
    )
    monkeypatch.setattr(
        science_module, "scientific_artifact_pair_identity", pair_identity
    )
    monkeypatch.setattr(
        boundary_module, "scientific_artifact_pair_identity", pair_identity
    )
    bad_design = _synthetic_loaded_artifact(prefix, "design")
    model_fit = dict(bad_design.reports["reports/model-fit.json"])  # type: ignore[arg-type]
    payload = dict(cast(Mapping[str, Any], model_fit["payload"]))
    wrong_tasks = dict(payload["task_result_fingerprints"])
    wrong_tasks["design_candidate_viability"] = "0" * 64
    payload["task_result_fingerprints"] = wrong_tasks
    model_fit["payload"] = payload
    bad_reports = dict(bad_design.reports)
    bad_reports["reports/model-fit.json"] = model_fit
    bad_design.reports = bad_reports
    with pytest.raises(IntegrityError, match="task-result bindings"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=artifact_pair(bad_design, production),
        )
    with pytest.raises(IntegrityError, match="role/K/binding/provenance"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=artifact_pair(production, design),
        )
    stale_design = _synthetic_loaded_artifact(prefix, "design")
    stale_design.binding = replace(
        prefix.binding,
        calibration_run_fingerprint="9" * 64,
    )
    with pytest.raises(IntegrityError, match="role/K/binding/provenance"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=artifact_pair(stale_design, production),
        )
    assert (
        tuple(store.verify_task_results(run_id, require_complete=False))
        == (G3_GATE_ORDER[:-1])
    )

    original_complete = store.complete_launch
    interrupted = False

    def interrupt_before_completion(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("interrupted after durable task 26")
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(store, "complete_launch", interrupt_before_completion)
    with pytest.raises(RuntimeError, match="durable task 26"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=artifact_pair(design, production),
        )
    assert tuple(store.verify_task_results(run_id, require_complete=False)) == (
        G3_GATE_ORDER
    )
    assert store.read_launch(run_id).state == "running"

    monkeypatch.setattr(store, "complete_launch", original_complete)
    final = science_module.finalize_canonical_g3_artifacts(
        store=store,
        lease=lease,
        live_preflight=preflight,
        checkpoint_store=checkpoint_store,
        prefix=prefix,
        artifact_pair=artifact_pair(design, production),
    )
    assert final.generation_authorized is False
    assert final.coordination.evidence_bundle.passed
    assert final.completion_marker.state == "completed"
    assert tuple(final.checkpoint_map) == G3_GATE_ORDER
    assert store.read_task_result(run_id, "artifact_integrity").status == "passed"
    with pytest.raises(IntegrityError, match="launch lease does not match"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=artifact_pair(design, production),
        )
    lease.close()


def test_two_stage_resumes_a_durable_task_25_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_synthetic_passing_decisions(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    handlers = _synthetic_passing_first_25(
        _input(preflight, run_id),
        passing_bridge_views,
    )
    original_write = store.write_task_result
    interrupted = False

    def interrupt_after_task_25(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        result = original_write(*args, **kwargs)
        if kwargs.get("task_id") == "actual_artifact_bridge" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted after durable task 25")
        return result

    monkeypatch.setattr(store, "write_task_result", interrupt_after_task_25)
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    with pytest.raises(RuntimeError, match="durable task 25"):
        science_module.coordinate_canonical_g3_first_25(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    assert (
        tuple(store.verify_task_results(run_id, require_complete=False))
        == (G3_GATE_ORDER[:-1])
    )
    lease.close()

    monkeypatch.setattr(store, "write_task_result", original_write)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
        resume=True,
    ) as resumed:
        prefix = science_module.coordinate_canonical_g3_first_25(
            store=store,
            lease=resumed,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
        assert prefix.artifact_publication_ready
        assert all(checkpoint.reused for checkpoint in prefix.checkpoints)
        assert store.read_launch(run_id).state == "running"


def test_crash_after_checkpoint_publish_resumes_by_exact_checkpoint_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    handlers = _handlers(_input(preflight, run_id))
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    original = store.write_task_result
    interrupted = False

    def crash_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        if kwargs.get("task_id") == "input_integrity" and not interrupted:
            interrupted = True
            raise RuntimeError("crash after checkpoint")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "write_task_result", crash_once)
    lease = store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    )
    with pytest.raises(RuntimeError, match="after checkpoint"):
        coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    lease.close()
    stored = tuple(checkpoint_store.checkpoint_root.iterdir())
    assert len(stored) == 1

    monkeypatch.setattr(store, "write_task_result", original)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
        resume=True,
    ) as resumed:
        result = coordinate_g3_canonical_launch(
            store=store,
            lease=resumed,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    assert result.task_results[0].status == "passed"
    assert result.checkpoints[0].reused
    assert tuple(checkpoint_store.checkpoint_root.iterdir()) == stored


def test_resume_rejects_changed_exact_live_preflight_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, launch_preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    original_write = store.write_task_result

    def stop_after_input(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("task_id") == "design_candidate_viability":
            raise RuntimeError("planned interruption after input")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(store, "write_task_result", stop_after_input)
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=launch_preflight.fingerprint,
        workers=9,
    )
    with pytest.raises(RuntimeError, match="after input"):
        coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=launch_preflight,
            checkpoint_store=checkpoint_store,
            handlers=_handlers(_input(launch_preflight, run_id)),
        )
    original_input = store.read_task_result(run_id, "input_integrity")
    lease.close()

    live_preflight = _changed_disk_preflight(launch_preflight)
    monkeypatch.setattr(store, "write_task_result", original_write)
    with (
        store.start_launch(
            run_id,
            preflight_fingerprint=launch_preflight.fingerprint,
            workers=9,
            resume=True,
        ) as resumed,
        pytest.raises(IntegrityError, match="payload differs from typed source"),
    ):
        coordinate_g3_canonical_launch(
            store=store,
            lease=resumed,
            live_preflight=live_preflight,
            checkpoint_store=checkpoint_store,
            handlers=_handlers(
                _input(
                    launch_preflight,
                    run_id,
                    live_preflight=live_preflight,
                )
            ),
        )
    assert (
        store.read_task_result(run_id, "input_integrity").fingerprint
        == original_input.fingerprint
    )
    assert live_preflight.fingerprint != launch_preflight.fingerprint


def test_resume_rejects_foreign_canonical_json_without_typed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    lease = store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    )
    store.write_task_result(
        lease,
        task_id="input_integrity",
        status="passed",
        payload={"execution_mode": "canonical", "caller_claim": True},
    )
    lease.close()
    with (
        store.start_launch(
            run_id,
            preflight_fingerprint=preflight.fingerprint,
            workers=9,
            resume=True,
        ) as resumed,
        pytest.raises(IntegrityError, match="payload differs"),
    ):
        coordinate_g3_canonical_launch(
            store=store,
            lease=resumed,
            live_preflight=preflight,
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "artifacts"),
            handlers=_handlers(_input(preflight, run_id)),
        )


def test_handler_bundle_rejects_callables_raw_aggregates_and_split_bridge_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _permit_test_preflight(monkeypatch)
    _, run_id, preflight = _run(tmp_path)
    input_view = _input(preflight, run_id)
    with pytest.raises(ModelError, match="cannot contain callables"):
        _handlers(lambda: True)
    with pytest.raises(ModelError, match="wrong typed scientific source"):
        _handlers(input_view.source)
    with pytest.raises(ModelError, match="wrong typed scientific source"):
        _handlers(
            input_view,
            design_block_calibration=_block_source(
                _kernel("monthly", (CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
                _kernel("daily", (CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
            ),
        )
    with pytest.raises(ModelError, match="wrong typed scientific source"):
        _handlers(
            input_view,
            artifact_integrity=ArtifactIntegrityDecisionEvidence(cast(Any, object())),
        )

    with pytest.raises(ModelError, match="wrong task"):
        _handlers(
            input_view,
            design_candidate_viability=_failure("sealed_holdout_veto"),
        )

    gate = _candidate_gate()
    with pytest.raises(ModelError, match="wrong typed scientific source"):
        _handlers(
            input_view,
            design_candidate_viability=gate,
            design_predictive_selection=replace(gate),
        )

    second_source = cast(
        Any,
        passing_bridge_views.bridge_tier_a_model_dgp.source,
    )
    original_identity = cast(Any, boundary_module).staged_bridge_task_evidence_identity

    def split_identity(value: Any) -> Any:
        identity = original_identity(value)
        if value is second_source:
            return replace(identity, common_binding_fingerprint="f" * 64)
        return identity

    monkeypatch.setattr(
        boundary_module,
        "staged_bridge_task_evidence_identity",
        split_identity,
    )
    second_view = boundary_module.build_bridge_driver_task_view(
        second_source,
        task_id="bridge_tier_a_model_dgp",
        run_id=passing_bridge_views.bridge_tier_a_model_dgp.run_id,
        binding_fingerprint="9" * 64,
    )
    monkeypatch.setattr(
        science_module,
        "task_view_cross_binding_error",
        lambda _sources: None,
    )
    with pytest.raises(ModelError, match="share one staged execution"):
        _handlers(
            input_view,
            bridge_resource_preflight=(passing_bridge_views.bridge_resource_preflight),
            bridge_tier_a_model_dgp=second_view,
        )


def test_input_source_must_bind_the_exact_run_and_live_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    wrong = _input(preflight, "f" * 64)
    with (
        store.start_launch(
            run_id, preflight_fingerprint=preflight.fingerprint, workers=9
        ) as lease,
        pytest.raises(IntegrityError, match="not bound"),
    ):
        coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "artifacts"),
            handlers=_handlers(wrong),
        )
    assert store.verify_task_results(run_id, require_complete=False) == {}


def test_typed_failed_holdout_is_failed_without_promoting_a_runner_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_legacy_coordinator_fixtures(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    gate = _candidate_gate()
    hmm = np.zeros(24)
    benchmarks = HoldoutBenchmarkScores(np.ones(24), np.full(24, 0.5))
    failed_holdout = SealedHoldoutEvidence(
        (17, 7),
        hmm,
        benchmarks,
        holdout_veto_decision(hmm, benchmarks),
    )
    failed_block = _failed_design_block_source()
    handlers = _handlers(
        _input(preflight, run_id),
        design_candidate_viability=gate,
        design_predictive_selection=gate,
        sealed_holdout_veto=failed_holdout,
        design_block_calibration=failed_block,
        design_fragmentation=failed_block,
    )
    with store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    ) as lease:
        result = coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "artifacts"),
            handlers=handlers,
        )

    assert store.read_task_result(run_id, "sealed_holdout_veto").status == "failed"
    assert store.read_task_result(run_id, "production_fit_same_k").status == "blocked"
    assert result.selected_n_states == gate.selection.selected_n_states
    assert result.task_results[2].payload["selected_n_states"] == (
        gate.selection.selected_n_states
    )


def test_resume_revalidates_existing_failure_and_coordinator_blocked_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    handlers = _handlers(_input(preflight, run_id))
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    original_write = store.write_task_result

    def interrupt_on_later_block(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("task_id") == "design_latent_correctness":
            raise RuntimeError("interrupt after failed and blocked prefix")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(store, "write_task_result", interrupt_on_later_block)
    lease = store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    )
    with pytest.raises(RuntimeError, match="failed and blocked"):
        coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    lease.close()
    assert store.read_task_result(run_id, "design_candidate_viability").status == (
        "failed"
    )
    assert store.read_task_result(run_id, "design_macro_stability").status == "blocked"

    monkeypatch.setattr(store, "write_task_result", original_write)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
        resume=True,
    ) as resumed:
        result = coordinate_g3_canonical_launch(
            store=store,
            lease=resumed,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    assert result.failed_task_ids == ("design_candidate_viability",)
    assert result.blocked_task_ids == G3_GATE_ORDER[2:]


def test_resume_reuses_exact_failed_planned_look_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_legacy_coordinator_fixtures(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    gate = _candidate_gate()
    failed_block = _failed_design_block_source()
    handlers = _handlers(
        _input(preflight, run_id),
        design_candidate_viability=gate,
        design_predictive_selection=gate,
        sealed_holdout_veto=_holdout(),
        design_block_calibration=failed_block,
        design_fragmentation=failed_block,
    )
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    original_write = store.write_task_result

    def interrupt_after_looks(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("task_id") == "design_block_calibration":
            raise RuntimeError("interrupt after planned looks")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(store, "write_task_result", interrupt_after_looks)
    lease = store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    )
    with pytest.raises(RuntimeError, match="after planned looks"):
        coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    before = store.verify_planned_look_receipts(run_id, require_terminal=False)
    assert len(before) == 2
    lease.close()

    monkeypatch.setattr(store, "write_task_result", original_write)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
        resume=True,
    ) as resumed:
        result = coordinate_g3_canonical_launch(
            store=store,
            lease=resumed,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    after = store.verify_planned_look_receipts(run_id, require_terminal=True)
    assert {
        family: tuple(item.fingerprint for item in receipts)
        for family, receipts in after.items()
    } == {
        family: tuple(item.fingerprint for item in receipts)
        for family, receipts in before.items()
    }
    assert "design_block_calibration" in result.failed_task_ids


def test_tier_b_planned_look_republishes_exact_verified_local_receipt(
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    with pytest.raises(ModelError, match="typed terminal look evidence"):
        canonical_planned_looks(
            "bridge_tier_b_model_dgp",
            _failure("bridge_tier_b_model_dgp"),
        )

    source = passing_bridge_views.bridge_tier_b_model_dgp
    looks = canonical_planned_looks("bridge_tier_b_model_dgp", source)
    local = source.planned_looks[0]
    interval = clopper_pearson_interval(
        local.exceedances,
        local.sample_count,
        confidence=1.0 - TIER_B_INTERVAL_ERROR,
    )

    assert len(looks) == 1
    assert looks[0].sample_count == 1_000
    assert looks[0].decision == "pass"
    assert looks[0].observed["exceedances"] == 1_000
    assert looks[0].observed["interval_lower"] == interval.lower
    assert (
        looks[0].observed["local_bridge_look_receipt_fingerprint"]
        == local.stage_receipt_fingerprint
    )
    assert (
        looks[0].observed["bridge_task_result_fingerprint"]
        == source.task_result_fingerprint
    )
    assert looks[0].observed["look_evidence_fingerprint"] == local.evidence_fingerprint
    assert looks[0].observed["look_rng_fingerprint"] == local.rng_execution_fingerprint
    nonparametric = canonical_planned_looks(
        "bridge_tier_b_nonparametric_dgp",
        passing_bridge_views.bridge_tier_b_nonparametric_dgp,
    )
    assert nonparametric[0].observed["dgp"] == "nonparametric"

    with pytest.raises(ModelError, match="sealed driver task view"):
        canonical_planned_looks(
            "bridge_tier_b_model_dgp",
            passing_bridge_views.bridge_tier_b_nonparametric_dgp,
        )


def test_bridge_artifact_cross_binding_uses_staged_selected_k(
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    monkeypatch.setattr(
        boundary_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "is_verified_scientific_artifact_pair",
        lambda _pair: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "scientific_artifact_pair_identity",
        lambda pair: {
            "actual_bridge_parent_view_fingerprint": (
                pair.actual_bridge_parent_view_fingerprint
            )
        },
    )
    binding = ScientificArtifactBinding(
        config_fingerprint="a" * 64,
        processed_data_fingerprint="b" * 64,
        calibration_input_fingerprint="c" * 64,
        calibration_run_fingerprint="d" * 64,
        preflight_fingerprint="e" * 64,
    )
    provenance = ScientificArtifactProvenance("f" * 64, "1" * 64)
    cores = {"design": "8" * 64, "production": "9" * 64}

    def artifact(
        role: str,
        outer: str,
        core: str,
        *,
        selected_k: int = 2,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            role=role,
            selected_k=selected_k,
            binding=binding,
            provenance=provenance,
            reference=SimpleNamespace(fingerprint=outer * 64),
            geometry=SimpleNamespace(as_dict=lambda: {"role": role}),
            generation_policy={"policy": "synthetic"},
            arrays={"weights": np.asarray([0.25, 0.75])},
            reports={"report.json": {"result_fingerprint": "6" * 64}},
            core_model_fingerprint=core,
            parameter_set_fingerprint=("2" if role == "design" else "3") * 64,
            generation_authorized=False,
        )

    design = artifact("design", "4", cores["design"])
    production = artifact("production", "5", cores["production"])

    def artifact_pair(design_value: object, production_value: object) -> Any:
        return SimpleNamespace(
            design=design_value,
            production=production_value,
            run_id="d" * 64,
            reference=SimpleNamespace(fingerprint="6" * 64),
            actual_bridge_parent_view_fingerprint=(
                passing_bridge_views.actual_artifact_bridge.view_fingerprint
            ),
            design_role_source_fingerprint="a" * 64,
            production_role_source_fingerprint="b" * 64,
            generation_authorized=False,
        )

    sources = {
        task_id: assembly_module.derive_g3_gate_decision(task_id, view)
        for task_id, view in passing_bridge_views.as_mapping().items()
    }
    sources["artifact_integrity"] = assembly_module.derive_g3_gate_decision(
        "artifact_integrity",
        boundary_module.build_artifact_integrity_task_view(
            ArtifactIntegrityDecisionEvidence(artifact_pair(design, production)),
            passing_bridge_views.actual_artifact_bridge,
            run_id="d" * 64,
            binding_fingerprint="7" * 64,
        ),
    )
    assert (
        assembly_module._cross_binding_error(
            sources,
            config_fingerprint=binding.config_fingerprint,
            processed_data_fingerprint=binding.processed_data_fingerprint,
            calibration_input_fingerprint=binding.calibration_input_fingerprint,
        )
        is None
    )

    stale_design = artifact("design", "4", cores["design"], selected_k=3)
    stale_production = artifact(
        "production",
        "5",
        cores["production"],
        selected_k=3,
    )
    sources["artifact_integrity"] = assembly_module.derive_g3_gate_decision(
        "artifact_integrity",
        boundary_module.build_artifact_integrity_task_view(
            ArtifactIntegrityDecisionEvidence(
                artifact_pair(stale_design, stale_production)
            ),
            passing_bridge_views.actual_artifact_bridge,
            run_id="d" * 64,
            binding_fingerprint="7" * 64,
        ),
    )
    assert (
        assembly_module._cross_binding_error(
            sources,
            config_fingerprint=binding.config_fingerprint,
            processed_data_fingerprint=binding.processed_data_fingerprint,
            calibration_input_fingerprint=binding.calibration_input_fingerprint,
        )
        == "typed decisions disagree on selected K"
    )


def test_kernel_planned_looks_bind_exact_prefixes_and_terminal_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _block_source(
        _kernel(
            "monthly",
            (
                2.0 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
                0.9 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
            ),
        ),
        _kernel("daily", (0.8 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,)),
    )
    with pytest.raises(ModelError, match="lacks block-calibration evidence"):
        canonical_planned_looks("design_block_calibration", source)
    monkeypatch.setattr(
        science_module,
        "is_block_calibration_task_view",
        lambda value: isinstance(value, SimpleNamespace)
        and value.task_id == "design_block_calibration",
    )

    def view(value: BlockCalibrationGateEvidence) -> Any:
        return SimpleNamespace(
            task_id="design_block_calibration",
            source=value,
        )

    looks = canonical_planned_looks("design_block_calibration", view(source))

    assert [(item.family_id, item.sample_count, item.decision) for item in looks] == [
        ("kernel_design_monthly", 10_000, "continue"),
        ("kernel_design_monthly", 20_000, "pass"),
        ("kernel_design_daily", 10_000, "pass"),
    ]
    assert looks[1].observed["maximum_annualized_half_width"] == (
        0.9 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX
    )
    assert looks[1].observed["kernel_sample_fingerprint"] == "e" * 64

    monthly = source.monthly_kernel
    bad_index = replace(
        monthly,
        completed_looks=(replace(monthly.completed_looks[0], look_index=4),),
        precision_passed=False,
    )
    with pytest.raises(ModelError, match="inconsistent"):
        canonical_planned_looks(
            "design_block_calibration",
            view(_block_source(bad_index, source.daily_kernel)),
        )

    early_failure = _kernel("monthly", (2.0 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,))
    with pytest.raises(ModelError, match="before pass or maximum"):
        canonical_planned_looks(
            "design_block_calibration",
            view(_block_source(early_failure, source.daily_kernel)),
        )

    mismatched_terminal = replace(monthly, precision_passed=False)
    with pytest.raises(ModelError, match="differs from its terminal"):
        canonical_planned_looks(
            "design_block_calibration",
            view(_block_source(mismatched_terminal, source.daily_kernel)),
        )


def test_canonical_direct_entry_requires_live_preflight_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, run_id, preflight = _run(tmp_path)
    monkeypatch.setattr(
        boundary_module,
        "verify_compatible_canonical_g3_preflights",
        lambda launch, _live: launch.fingerprint,
    )
    with (
        store.start_launch(
            run_id, preflight_fingerprint=preflight.fingerprint, workers=9
        ) as lease,
        pytest.raises(ModelError, match="contract fields|live builder authority"),
    ):
        science_module.coordinate_canonical_g3_handlers(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "artifacts"),
            handlers=_handlers(_input(preflight, run_id)),
        )


def test_canonical_handler_and_restore_runtime_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    handlers = _handlers(_input(preflight, run_id))
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    lease = store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    )
    kwargs: dict[str, Any] = {
        "store": store,
        "lease": lease,
        "live_preflight": preflight,
        "checkpoint_store": checkpoint_store,
        "handlers": handlers,
    }
    with pytest.raises(ModelError, match="store has the wrong type"):
        science_module.coordinate_canonical_g3_handlers(**{**kwargs, "store": object()})
    with pytest.raises(IntegrityError, match="active canonical launch lease"):
        science_module.coordinate_canonical_g3_handlers(**{**kwargs, "lease": object()})
    with pytest.raises(ModelError, match="typed live preflight"):
        science_module.coordinate_canonical_g3_handlers(
            **{**kwargs, "live_preflight": object()}
        )
    with pytest.raises(ModelError, match="checkpoint store"):
        science_module.coordinate_canonical_g3_handlers(
            **{**kwargs, "checkpoint_store": object()}
        )
    with pytest.raises(ModelError, match="closed handler bundle"):
        science_module.coordinate_canonical_g3_handlers(
            **{**kwargs, "handlers": object()}
        )

    monkeypatch.setattr(
        science_module,
        "canonical_resume_preflight_fingerprint",
        lambda *_args, **_kwargs: "f" * 64,
    )
    with pytest.raises(IntegrityError, match="persisted preflight"):
        science_module.coordinate_canonical_g3_handlers(**kwargs)
    _permit_test_preflight(monkeypatch)

    with pytest.raises(ModelError, match="store has the wrong type"):
        science_module.restore_canonical_g3_state(
            store=object(),  # type: ignore[arg-type]
            lease=lease,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    with pytest.raises(IntegrityError, match="active canonical launch lease"):
        science_module.restore_canonical_g3_state(
            store=store,
            lease=object(),  # type: ignore[arg-type]
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    with pytest.raises(ModelError, match="checkpoint store"):
        science_module.restore_canonical_g3_state(
            store=store,
            lease=lease,
            checkpoint_store=object(),  # type: ignore[arg-type]
            handlers=handlers,
        )
    with pytest.raises(ModelError, match="closed handler bundle"):
        science_module.restore_canonical_g3_state(
            store=store,
            lease=lease,
            checkpoint_store=checkpoint_store,
            handlers=object(),  # type: ignore[arg-type]
        )
    lease.close()


def test_private_integrity_guards_reject_changed_checkpoint_bundle_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    with store.start_launch(
        run_id, preflight_fingerprint=preflight.fingerprint, workers=9
    ) as lease:
        result = coordinate_g3_canonical_launch(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=CalibrationCheckpointStore(tmp_path / "artifacts"),
            handlers=_handlers(_input(preflight, run_id)),
        )
    checkpoint = result.checkpoints[0]
    state = checkpoint.state
    decision = cast(dict[str, Any], state["decision"])
    decision_fingerprint = cast(str, state["decision_fingerprint"])
    looks = cast(dict[str, list[str]], state["planned_look_fingerprints"])

    with pytest.raises(IntegrityError, match="scientific state"):
        science_module._verify_checkpoint_state(
            replace(checkpoint, state={"changed": True}),
            decision_document=decision,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints=looks,
        )
    with pytest.raises(IntegrityError, match="digest arrays are invalid"):
        science_module._verify_checkpoint_state(
            replace(checkpoint, arrays={"decision_sha256": np.zeros(32)}),
            decision_document=decision,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints=looks,
        )
    changed_arrays = dict(checkpoint.arrays)
    changed_arrays["decision_sha256"] = np.zeros(32, dtype=np.uint8)
    with pytest.raises(IntegrityError, match="digest arrays disagree"):
        science_module._verify_checkpoint_state(
            replace(checkpoint, arrays=changed_arrays),
            decision_document=decision,
            decision_fingerprint=decision_fingerprint,
            planned_look_fingerprints=looks,
        )

    with pytest.raises(IntegrityError, match="graph is incomplete"):
        science_module._verify_bundle_matches_results(result.evidence_bundle, {})
    reversed_bundle = replace(
        result.evidence_bundle,
        records=tuple(reversed(result.evidence_bundle.records)),
    )
    result_mapping = {item.task_id: item for item in result.task_results}
    with pytest.raises(IntegrityError, match="order differs"):
        science_module._verify_bundle_matches_results(reversed_bundle, result_mapping)
    first = result.evidence_bundle.records[0]
    changed_record = replace(first, status="failed")
    changed_bundle = replace(
        result.evidence_bundle,
        records=(changed_record, *result.evidence_bundle.records[1:]),
    )
    with pytest.raises(IntegrityError, match="decision differs"):
        science_module._verify_bundle_matches_results(changed_bundle, result_mapping)
    with pytest.raises(IntegrityError, match="passed decision differs"):
        science_module._verify_bundle_matches_results(
            replace(
                result.evidence_bundle,
                passed=not result.evidence_bundle.passed,
            ),
            result_mapping,
        )
    with pytest.raises(IntegrityError, match="evidence-only"):
        science_module._verify_bundle_matches_results(
            replace(result.evidence_bundle, generation_authorized=True),
            result_mapping,
        )

    assert science_module._json_value(np.bool_(True), "x") is True
    assert science_module._json_value(np.int64(2), "x") == 2
    assert science_module._json_value(np.asarray([1, 2]), "x") == [1, 2]
    with pytest.raises(ModelError, match="non-finite"):
        science_module._json_value(float("nan"), "x")
    with pytest.raises(ModelError, match="invalid mapping key"):
        science_module._json_value({1: "bad"}, "x")
    with pytest.raises(ModelError, match="non-JSON"):
        science_module._json_value({1, 2}, "x")
    with pytest.raises(ModelError, match="empty or unbounded"):
        science_module._validated_failure(
            "input_integrity", G3TaskFailure("input_integrity", "", "")
        )
    with pytest.raises(ModelError, match="one of 2,3,4,5"):
        science_module._state_count(True)


def _tampered_dataclass(value: Any, **changes: object) -> Any:
    """Clone an init-closed evidence object while simulating durable tamper."""

    clone = object.__new__(type(value))
    for item in fields(value):
        name = item.name
        object.__setattr__(clone, name, changes.get(name, getattr(value, name)))
    return clone


def _unverified_task_result(
    run_id: str,
    task_id: str,
    *,
    status: str,
    payload: Mapping[str, object],
) -> science_module.G3TaskResult:
    """Build a store-boundary test double whose payload restoration must reject."""

    return science_module.G3TaskResult(
        run_id=run_id,
        task_id=task_id,
        status=cast(Any, status),
        payload=MappingProxyType(dict(payload)),
        dependency_fingerprints=MappingProxyType({}),
        blocked_by=(),
        fingerprint=hashlib.sha256(f"tampered:{task_id}:{status}".encode()).hexdigest(),
        path=Path("/untrusted/task-result.json"),
        reused=False,
    )


def test_closed_task_scope_dispatch_and_handler_order_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = (
        ("input_integrity", "is_input_integrity_task_view"),
        ("design_candidate_viability", "is_candidate_viability_task_view"),
        ("design_predictive_selection", "is_candidate_selection_task_view"),
        ("sealed_holdout_veto", "is_sealed_holdout_veto_view"),
        ("design_macro_stability", "is_macro_stability_task_view"),
        ("production_macro_stability", "is_macro_stability_task_view"),
        ("production_fit_same_k", "is_production_fit_same_k_view"),
        ("design_latent_correctness", "is_adequacy_cell_task_view"),
        ("design_refitted_adequacy", "is_adequacy_cell_task_view"),
        ("production_latent_correctness", "is_adequacy_cell_task_view"),
        ("production_refitted_adequacy", "is_adequacy_cell_task_view"),
        ("four_cell_adequacy_holm", "is_four_cell_adequacy_holm_view"),
        ("design_fragmentation", "is_registered_dependence_task_view"),
        ("design_dependence", "is_registered_dependence_task_view"),
        ("production_fragmentation", "is_registered_dependence_task_view"),
        ("production_dependence", "is_registered_dependence_task_view"),
        (
            "four_cell_dependence_holm",
            "is_registered_four_cell_dependence_holm_view",
        ),
        ("design_block_calibration", "is_block_calibration_task_view"),
        ("production_block_calibration", "is_block_calibration_task_view"),
        ("bridge_resource_preflight", "is_bridge_driver_task_view"),
        ("bridge_tier_a_model_dgp", "is_bridge_driver_task_view"),
        ("actual_artifact_bridge", "is_bridge_driver_task_view"),
        ("artifact_integrity", "is_artifact_integrity_task_view"),
    )
    for task_id, validator_name in dispatch:
        calls: list[object] = []
        monkeypatch.setattr(
            science_module,
            validator_name,
            lambda value, calls=calls: not calls.append(value),
        )
        source = SimpleNamespace(task_id=task_id)
        assert science_module._is_valid_task_scoped_view(task_id, source)
        assert calls == [source]

    assert not science_module._is_valid_task_scoped_view(
        "design_block_calibration",
        SimpleNamespace(task_id="production_block_calibration"),
    )
    assert not science_module._is_valid_task_scoped_view(
        "bridge_tier_a_model_dgp",
        SimpleNamespace(task_id="bridge_tier_a_nonparametric_dgp"),
    )
    assert not science_module._is_valid_task_scoped_view("unknown", object())

    reversed_failures = {
        task_id: _failure(task_id) for task_id in reversed(G3_GATE_ORDER)
    }
    with pytest.raises(AssertionError, match="fields differ"):
        science_module._validate_handler_sources(
            cast(Any, reversed_failures),
            expected_order=G3_GATE_ORDER,
        )


def test_failure_and_selected_k_derivation_cannot_promote_or_change_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    _, run_id, preflight = _run(tmp_path)
    handlers = _handlers(_input(preflight, run_id))
    decisions, failures = science_module._derived_sources(handlers)
    assert tuple(item.gate_id for item in decisions) == ("input_integrity",)
    assert decisions[0].passed
    assert tuple(item.task_id for item in failures) == G3_GATE_ORDER[1:]

    def decision(
        task_id: str,
        *,
        passed: bool,
        selected_k: object | None = None,
    ) -> assembly_module.DerivedG3GateDecision:
        value = object.__new__(assembly_module.DerivedG3GateDecision)
        summary = {} if selected_k is None else {"selected_k": selected_k}
        for name, item in (
            ("gate_id", task_id),
            ("passed", passed),
            ("evidence", MappingProxyType({})),
            ("summary", MappingProxyType(summary)),
            ("_source", object()),
            ("_seal", object()),
        ):
            object.__setattr__(value, name, item)
        return value

    assert (
        science_module._next_selected_n_states(
            task_id="design_predictive_selection",
            decision=decision("design_predictive_selection", passed=False),
            frozen=None,
        )
        is None
    )
    assert (
        science_module._next_selected_n_states(
            task_id="design_predictive_selection",
            decision=decision("design_predictive_selection", passed=True, selected_k=3),
            frozen=None,
        )
        == 3
    )
    assert (
        science_module._next_selected_n_states(
            task_id="design_candidate_viability",
            decision=decision("design_candidate_viability", passed=True, selected_k=3),
            frozen=None,
        )
        is None
    )
    with pytest.raises(ModelError, match="without a passed design selection"):
        science_module._next_selected_n_states(
            task_id="sealed_holdout_veto",
            decision=decision("sealed_holdout_veto", passed=True, selected_k=3),
            frozen=None,
        )
    with pytest.raises(ModelError, match="change or promote"):
        science_module._next_selected_n_states(
            task_id="sealed_holdout_veto",
            decision=decision("sealed_holdout_veto", passed=True, selected_k=3),
            frozen=2,
        )
    assert (
        science_module._next_selected_n_states(
            task_id="sealed_holdout_veto",
            decision=decision("sealed_holdout_veto", passed=True, selected_k=2),
            frozen=2,
        )
        == 2
    )
    for invalid in (2.0, 6):
        with pytest.raises(ModelError, match="one of 2,3,4,5"):
            science_module._state_count(invalid)


def test_input_binding_requires_failure_or_sealed_exact_launch_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    _, run_id, preflight = _run(tmp_path)
    source = _input(preflight, run_id)
    science_module._validate_input_source_binding(
        _failure("input_integrity"),
        run_id=run_id,
        persisted_preflight_fingerprint=preflight.fingerprint,
        live_preflight=preflight,
    )
    science_module._validate_input_source_binding(
        source,
        run_id=run_id,
        persisted_preflight_fingerprint=preflight.fingerprint,
        live_preflight=preflight,
    )
    with monkeypatch.context() as patch:
        unexpected_owner = replace(
            science_module.G3_PLANNED_LOOK_REGISTRY[0],
            owner_task_id="input_integrity",
        )
        patch.setattr(
            science_module,
            "G3_PLANNED_LOOK_REGISTRY",
            (unexpected_owner,),
        )
        with pytest.raises(AssertionError, match="no code-owned publisher"):
            canonical_planned_looks("input_integrity", source)
    monkeypatch.setattr(
        science_module,
        "is_input_integrity_task_view",
        lambda _value: False,
    )
    with pytest.raises(ModelError, match="sealed task authority"):
        science_module._validate_input_source_binding(
            source,
            run_id=run_id,
            persisted_preflight_fingerprint=preflight.fingerprint,
            live_preflight=preflight,
        )


def test_planned_look_sequences_fail_closed_on_type_schedule_and_replay_tamper(
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    with pytest.raises(ModelError, match="typed execution evidence"):
        science_module._kernel_looks("kernel", (10_000,), cast(Any, object()))
    empty_kernel = replace(_kernel("monthly", (0.01,)), completed_looks=())
    with pytest.raises(ModelError, match="empty or too long"):
        science_module._kernel_looks("kernel", (10_000,), empty_kernel)
    retained_after_pass = _kernel(
        "monthly",
        (
            0.5 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
            0.5 * CANONICAL_ANNUALIZED_HALF_WIDTH_MAX,
        ),
    )
    with pytest.raises(ModelError, match="retained work"):
        science_module._kernel_looks(
            "kernel",
            (10_000, 20_000),
            retained_after_pass,
        )

    source = passing_bridge_views.bridge_tier_b_model_dgp
    local = source.planned_looks[0]
    empty_source = _tampered_dataclass(source, planned_looks=())
    with pytest.raises(ModelError, match="empty or too long"):
        science_module._tier_b_looks("tier-b", (1_000,), empty_source)
    wrong_dgp = _tampered_dataclass(local, dgp="nonparametric")
    with pytest.raises(ModelError, match="schedule is inconsistent"):
        science_module._tier_b_looks(
            "tier-b",
            (1_000,),
            _tampered_dataclass(source, planned_looks=(wrong_dgp,)),
        )
    first_terminal = _tampered_dataclass(local, decision="pass_above_tail")
    second_terminal = _tampered_dataclass(
        local,
        look_index=2,
        sample_count=5_000,
        decision="pass_above_tail",
    )
    with pytest.raises(ModelError, match="terminal before its last look"):
        science_module._tier_b_looks(
            "tier-b",
            (1_000, 5_000),
            _tampered_dataclass(
                source,
                planned_looks=(first_terminal, second_terminal),
            ),
        )
    nonterminal = _tampered_dataclass(local, decision="continue")
    with pytest.raises(ModelError, match="nonterminal"):
        science_module._tier_b_looks(
            "tier-b",
            (1_000,),
            _tampered_dataclass(source, planned_looks=(nonterminal,)),
        )

    proposed = science_module.CanonicalG3PlannedLook(
        "family",
        1,
        1_000,
        "pass",
        MappingProxyType({"estimate": 1.0}),
    )
    receipt = science_module.PlannedLookReceipt(
        run_id="a" * 64,
        family_id="family",
        look_index=1,
        sample_count=1_000,
        decision="pass",
        observed=MappingProxyType({"estimate": 1.0}),
        previous_fingerprint=None,
        fingerprint="b" * 64,
        path=Path("/verified/look.json"),
        reused=False,
    )
    science_module._compare_receipts_to_looks((receipt,), (proposed,))
    with pytest.raises(IntegrityError, match="sequence length changed"):
        science_module._compare_receipts_to_looks((), (proposed,))
    with pytest.raises(IntegrityError, match="sequence length changed"):
        science_module._compare_receipts_to_looks(
            (receipt,),
            (),
            allow_prefix=True,
        )
    with pytest.raises(IntegrityError, match="evidence changed"):
        science_module._compare_receipts_to_looks(
            (replace(receipt, observed=MappingProxyType({"estimate": 0.0})),),
            (proposed,),
        )


def test_sealed_prefix_report_and_restored_state_reject_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_bridge_views: BridgeDriverTaskViews,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_synthetic_passing_decisions(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    handlers = _synthetic_passing_first_25(
        _input(preflight, run_id),
        passing_bridge_views,
    )
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    prefix = science_module.coordinate_canonical_g3_first_25(
        store=store,
        lease=lease,
        live_preflight=preflight,
        checkpoint_store=checkpoint_store,
        handlers=handlers,
    )
    science_module._require_artifact_ready_prefix(prefix)
    with pytest.raises(ModelError, match="sealed first-25 prefix"):
        science_module._require_artifact_ready_prefix(cast(Any, object()))
    with pytest.raises(IntegrityError, match="prefix identity changed"):
        science_module._require_artifact_ready_prefix(
            _tampered_dataclass(prefix, prefix_fingerprint="0" * 64)
        )

    processed = {item.task_id: item for item in prefix.task_results}
    checkpoints = {item.task_id: item for item in prefix.checkpoints}
    with pytest.raises(IntegrityError, match="exactly 25 tasks"):
        science_module._mint_artifact_publication_prefix(
            run_id=run_id,
            processed={
                key: value
                for key, value in processed.items()
                if key != G3_GATE_ORDER[0]
            },
            checkpoints=checkpoints,
            selected_n_states=prefix.selected_n_states,
            binding=prefix.binding,
            provenance=prefix.provenance,
            handlers=handlers,
            store=store,
        )
    with pytest.raises(IntegrityError, match="lacks exact checkpoints"):
        science_module._mint_artifact_publication_prefix(
            run_id=run_id,
            processed=processed,
            checkpoints={
                key: value
                for key, value in checkpoints.items()
                if key != G3_GATE_ORDER[0]
            },
            selected_n_states=prefix.selected_n_states,
            binding=prefix.binding,
            provenance=prefix.provenance,
            handlers=handlers,
            store=store,
        )

    monkeypatch.setattr(
        science_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: False,
    )
    artifact = _synthetic_loaded_artifact(prefix, "design")
    with pytest.raises(ModelError, match="typed loaded scientific artifacts"):
        science_module._verify_artifact_prefix_reports(
            cast(Any, artifact),
            expected_role="design",
            prefix=prefix,
        )
    monkeypatch.setattr(
        science_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: True,
    )

    bad_checkpoint = _synthetic_loaded_artifact(prefix, "design")
    for path, report in bad_checkpoint.reports.items():
        payload = dict(
            cast(Mapping[str, Any], cast(Mapping[str, Any], report)["payload"])
        )
        checkpoint_bindings = dict(payload["checkpoint_fingerprints"])
        if checkpoint_bindings:
            checkpoint_bindings[next(iter(checkpoint_bindings))] = "0" * 64
            payload["checkpoint_fingerprints"] = checkpoint_bindings
            bad_checkpoint.reports[path] = {"payload": payload}
            break
    with pytest.raises(IntegrityError, match="checkpoint bindings"):
        science_module._verify_artifact_prefix_reports(
            cast(Any, bad_checkpoint),
            expected_role="design",
            prefix=prefix,
        )

    bad_looks = _synthetic_loaded_artifact(prefix, "design")
    for path, report in bad_looks.reports.items():
        payload = dict(
            cast(Mapping[str, Any], cast(Mapping[str, Any], report)["payload"])
        )
        if payload["planned_look_fingerprints"]:
            payload["planned_look_fingerprints"] = {}
            bad_looks.reports[path] = {"payload": payload}
            break
    with pytest.raises(IntegrityError, match="planned-look bindings"):
        science_module._verify_artifact_prefix_reports(
            cast(Any, bad_looks),
            expected_role="design",
            prefix=prefix,
        )

    original_scientific_report_task_ids = science_module.scientific_report_task_ids
    monkeypatch.setattr(
        science_module,
        "scientific_report_task_ids",
        lambda _document_type, _role: ("artifact_integrity",),
    )
    with pytest.raises(AssertionError, match="depends on task 26"):
        canonical_g3_scientific_report_payload(
            prefix,
            document_type="model_fit",
            role="design",
            metrics={},
        )
    monkeypatch.setattr(
        science_module,
        "scientific_report_task_ids",
        original_scientific_report_task_ids,
    )

    state = science_module.CanonicalG3State(
        run_id=run_id,
        task_results=MappingProxyType(processed),
        checkpoints=MappingProxyType(checkpoints),
        selected_n_states=prefix.selected_n_states,
    )
    science_module._verify_restored_prefix(prefix, state, store=store)
    with pytest.raises(IntegrityError, match="task prefix is incomplete"):
        science_module._verify_restored_prefix(
            prefix,
            replace(
                state,
                task_results=MappingProxyType(
                    {
                        key: value
                        for key, value in processed.items()
                        if key != G3_GATE_ORDER[-2]
                    }
                ),
            ),
            store=store,
        )
    changed_results = dict(processed)
    changed_results[G3_GATE_ORDER[0]] = replace(
        changed_results[G3_GATE_ORDER[0]],
        fingerprint="0" * 64,
    )
    with pytest.raises(IntegrityError, match="task receipts changed"):
        science_module._verify_restored_prefix(
            prefix,
            replace(state, task_results=MappingProxyType(changed_results)),
            store=store,
        )
    with pytest.raises(IntegrityError, match="checkpoints changed"):
        science_module._verify_restored_prefix(
            prefix,
            replace(
                state,
                checkpoints=MappingProxyType(
                    {
                        key: value
                        for key, value in checkpoints.items()
                        if key != G3_GATE_ORDER[0]
                    }
                ),
            ),
            store=store,
        )
    empty_look_store = cast(
        CalibrationRunStore,
        SimpleNamespace(verify_planned_look_receipts=lambda *_args, **_kwargs: {}),
    )
    with pytest.raises(IntegrityError, match="planned looks changed"):
        science_module._verify_restored_prefix(
            prefix,
            state,
            store=empty_look_store,
        )
    with pytest.raises(IntegrityError, match="selected K changed"):
        science_module._verify_restored_prefix(
            prefix,
            replace(state, selected_n_states=3),
            store=store,
        )

    def rehashed_prefix(**changes: object) -> Any:
        altered = _tampered_dataclass(prefix, **changes)
        identity = {
            "binding": altered.binding.as_dict(),
            "checkpoint_fingerprints": dict(altered.checkpoint_fingerprints),
            "planned_look_fingerprints": {
                key: list(value)
                for key, value in altered.planned_look_fingerprints.items()
            },
            "provenance": altered.provenance.as_dict(),
            "run_id": altered.run_id,
            "selected_n_states": altered.selected_n_states,
            "task_result_fingerprints": dict(altered.task_result_fingerprints),
        }
        return _tampered_dataclass(
            altered,
            prefix_fingerprint=science_module._fingerprint(identity),
        )

    foreign_binding = replace(
        prefix.binding,
        calibration_run_fingerprint="f" * 64,
    )
    with pytest.raises(IntegrityError, match="belongs to another launch"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=rehashed_prefix(binding=foreign_binding),
            artifact_pair=cast(Any, object()),
        )
    stale_provenance = ScientificArtifactProvenance(
        "f" * 64,
        prefix.provenance.dependency_lock_fingerprint,
    )
    with pytest.raises(IntegrityError, match="prefix provenance is stale"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=rehashed_prefix(provenance=stale_provenance),
            artifact_pair=cast(Any, object()),
        )

    with pytest.raises(ModelError, match="loaded scientific pair receipt"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=cast(Any, object()),
        )
    monkeypatch.setattr(
        science_module,
        "is_verified_scientific_artifact_pair",
        lambda _pair: True,
    )
    monkeypatch.setattr(
        science_module,
        "scientific_artifact_pair_identity",
        lambda _pair: {
            "task_result_fingerprints": dict(prefix.task_result_fingerprints),
            "checkpoint_fingerprints": dict(prefix.checkpoint_fingerprints),
        },
    )
    mismatched_pair = SimpleNamespace(
        _prefix=object(),
        run_id=prefix.run_id,
        prefix_fingerprint=prefix.prefix_fingerprint,
        selected_k=prefix.selected_n_states,
    )
    with pytest.raises(IntegrityError, match="differs from first-25 prefix"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=cast(Any, mismatched_pair),
        )

    design = _synthetic_loaded_artifact(prefix, "design")
    production = _synthetic_loaded_artifact(prefix, "production")
    good_pair = SimpleNamespace(
        _prefix=prefix,
        design=design,
        production=production,
        run_id=prefix.run_id,
        selected_k=prefix.selected_n_states,
        prefix_fingerprint=prefix.prefix_fingerprint,
        reference=SimpleNamespace(fingerprint="9" * 64),
        actual_bridge_parent_view_fingerprint=(
            prefix._handlers.actual_artifact_bridge.view_fingerprint
        ),
        design_role_source_fingerprint="a" * 64,
        production_role_source_fingerprint="b" * 64,
        generation_authorized=False,
    )
    bridge_guard = science_module.is_bridge_driver_task_view
    monkeypatch.setattr(
        science_module,
        "is_bridge_driver_task_view",
        lambda _value: False,
    )
    with pytest.raises(IntegrityError, match="lacks its sealed bridge parent"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=cast(Any, good_pair),
        )

    monkeypatch.setattr(science_module, "is_bridge_driver_task_view", bridge_guard)
    monkeypatch.setattr(
        boundary_module,
        "is_verified_scientific_model_artifact",
        lambda _artifact: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "is_verified_scientific_artifact_pair",
        lambda _pair: True,
    )
    monkeypatch.setattr(
        boundary_module,
        "scientific_artifact_pair_identity",
        lambda _pair: {
            "actual_bridge_parent_view_fingerprint": (
                prefix._handlers.actual_artifact_bridge.view_fingerprint
            )
        },
    )
    original_derive_g3_gate_decision = science_module.derive_g3_gate_decision
    monkeypatch.setattr(
        science_module,
        "derive_g3_gate_decision",
        lambda _task_id, _source: SimpleNamespace(passed=False),
    )
    with pytest.raises(ModelError, match="artifact integrity did not pass"):
        science_module.finalize_canonical_g3_artifacts(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            prefix=prefix,
            artifact_pair=cast(Any, good_pair),
        )
    monkeypatch.setattr(
        science_module,
        "derive_g3_gate_decision",
        original_derive_g3_gate_decision,
    )

    invalid_results = {
        key: value for key, value in processed.items() if key != G3_GATE_ORDER[-2]
    }
    invalid_state = replace(
        state,
        task_results=MappingProxyType(invalid_results),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            science_module,
            "restore_canonical_g3_state",
            lambda **_kwargs: invalid_state,
        )
        patch.setattr(
            science_module,
            "_verify_restored_prefix",
            lambda *_args, **_kwargs: None,
        )
        with pytest.raises(IntegrityError, match="found an invalid prefix"):
            science_module.finalize_canonical_g3_artifacts(
                store=store,
                lease=lease,
                live_preflight=preflight,
                checkpoint_store=checkpoint_store,
                prefix=prefix,
                artifact_pair=cast(Any, good_pair),
            )

    full_results = {
        **processed,
        "artifact_integrity": _unverified_task_result(
            run_id,
            "artifact_integrity",
            status="passed",
            payload={},
        ),
    }
    full_checkpoints = {
        **checkpoints,
        "artifact_integrity": prefix.checkpoints[-1],
    }
    full_state = replace(
        state,
        task_results=MappingProxyType(full_results),
        checkpoints=MappingProxyType(full_checkpoints),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            science_module,
            "restore_canonical_g3_state",
            lambda **_kwargs: full_state,
        )
        patch.setattr(
            science_module,
            "_verify_restored_prefix",
            lambda *_args, **_kwargs: None,
        )
        patch.setattr(science_module, "_derived_sources", lambda _handlers: ((), ()))
        patch.setattr(
            science_module,
            "assemble_typed_g3_evidence_bundle",
            lambda **_kwargs: SimpleNamespace(
                passed=False,
                generation_authorized=False,
            ),
        )
        patch.setattr(
            science_module,
            "_verify_bundle_matches_results",
            lambda *_args, **_kwargs: None,
        )
        with pytest.raises(IntegrityError, match="did not produce passed evidence"):
            science_module.finalize_canonical_g3_artifacts(
                store=store,
                lease=lease,
                live_preflight=preflight,
                checkpoint_store=checkpoint_store,
                prefix=prefix,
                artifact_pair=cast(Any, good_pair),
            )
    lease.close()


def test_launch_context_and_legacy_entry_reject_tampered_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    handlers = _handlers(_input(preflight, run_id))
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    kwargs: dict[str, Any] = {
        "store": store,
        "lease": lease,
        "live_preflight": preflight,
        "checkpoint_store": checkpoint_store,
    }
    science_module._canonical_launch_context(**kwargs)
    with pytest.raises(ModelError, match="store has the wrong type"):
        science_module._canonical_launch_context(**{**kwargs, "store": object()})
    with pytest.raises(IntegrityError, match="active canonical launch lease"):
        science_module._canonical_launch_context(**{**kwargs, "lease": object()})
    with pytest.raises(ModelError, match="typed live preflight"):
        science_module._canonical_launch_context(
            **{**kwargs, "live_preflight": object()}
        )
    with pytest.raises(ModelError, match="checkpoint store"):
        science_module._canonical_launch_context(
            **{**kwargs, "checkpoint_store": object()}
        )
    with monkeypatch.context() as patch:
        patch.setattr(
            science_module,
            "canonical_resume_preflight_fingerprint",
            lambda *_args, **_kwargs: "f" * 64,
        )
        with pytest.raises(IntegrityError, match="persisted preflight"):
            science_module._canonical_launch_context(**kwargs)

    stale_preflight = replace(preflight, config_fingerprint="f" * 64)
    with pytest.raises(IntegrityError, match="identities disagree"):
        science_module._canonical_launch_context(
            **{**kwargs, "live_preflight": stale_preflight}
        )

    first_25 = _first_25_handlers(_input(preflight, run_id))
    with monkeypatch.context() as patch:
        patch.setattr(
            store,
            "verify_task_results",
            lambda *_args, **_kwargs: dict.fromkeys(G3_GATE_ORDER, object()),
        )
        with pytest.raises(IntegrityError, match="already finalized"):
            science_module.coordinate_canonical_g3_first_25(
                **kwargs,
                handlers=first_25,
            )
    with monkeypatch.context() as patch:
        patch.setattr(
            store,
            "verify_task_results",
            lambda *_args, **_kwargs: {"design_candidate_viability": object()},
        )
        with pytest.raises(IntegrityError, match="not a first-25 prefix"):
            science_module.coordinate_canonical_g3_first_25(
                **kwargs,
                handlers=first_25,
            )

    original_read_launch = store.read_launch
    with monkeypatch.context() as patch:
        read_count = 0

        def completed_during_first_stage(run_id_value: str) -> Any:
            nonlocal read_count
            read_count += 1
            marker = original_read_launch(run_id_value)
            return replace(marker, state="completed") if read_count >= 3 else marker

        patch.setattr(store, "read_launch", completed_during_first_stage)
        patch.setattr(
            science_module,
            "_advance_canonical_tasks",
            lambda **_kwargs: ({}, {}, None),
        )
        with pytest.raises(IntegrityError, match="unexpectedly completed"):
            science_module.coordinate_canonical_g3_first_25(
                **kwargs,
                handlers=first_25,
            )

    object.__setattr__(
        handlers,
        "artifact_integrity",
        handlers.input_integrity,
    )
    with pytest.raises(ModelError, match="prebuilt artifact-integrity"):
        science_module.coordinate_canonical_g3_handlers(
            **kwargs,
            handlers=handlers,
        )
    untampered = _handlers(_input(preflight, run_id))
    with pytest.raises(IntegrityError, match="identities disagree"):
        science_module.coordinate_canonical_g3_handlers(
            **{**kwargs, "live_preflight": stale_preflight},
            handlers=untampered,
        )
    lease.close()


def test_advance_derives_failed_selection_without_retaining_or_promoting_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "a" * 64
    existing = {
        task_id: _unverified_task_result(
            run_id,
            task_id,
            status="passed",
            payload={},
        )
        for task_id in G3_GATE_ORDER[:2]
    }
    state = science_module.CanonicalG3State(
        run_id=run_id,
        task_results=MappingProxyType(existing),
        checkpoints=MappingProxyType({}),
        selected_n_states=None,
    )
    failed_selection = object.__new__(assembly_module.DerivedG3GateDecision)
    for name, value in (
        ("gate_id", "design_predictive_selection"),
        ("passed", False),
        ("evidence", MappingProxyType({"selection": "failed"})),
        ("summary", MappingProxyType({"selected_k": 2})),
        ("_source", object()),
        ("_seal", object()),
    ):
        object.__setattr__(failed_selection, name, value)
    monkeypatch.setattr(
        science_module,
        "derive_g3_gate_decision",
        lambda _task_id, _source: failed_selection,
    )

    def write_result(
        _lease: object,
        *,
        task_id: str,
        status: str,
        payload: Mapping[str, object],
    ) -> science_module.G3TaskResult:
        return _unverified_task_result(
            run_id,
            task_id,
            status=status,
            payload=payload,
        )

    fake_store = cast(
        CalibrationRunStore,
        SimpleNamespace(
            verify_planned_look_receipts=lambda *_args, **_kwargs: {},
            write_task_result=write_result,
        ),
    )
    binding = science_module.CalibrationCheckpointBinding(
        run_id=run_id,
        config_fingerprint="b" * 64,
        processed_data_fingerprint="c" * 64,
        calibration_input_fingerprint="d" * 64,
        preflight_fingerprint="e" * 64,
    )
    processed, checkpoints, selected_n_states = science_module._advance_canonical_tasks(
        store=fake_store,
        lease=cast(Any, SimpleNamespace(run_id=run_id)),
        checkpoint_store=cast(Any, object()),
        binding=binding,
        task_specs=science_module.G3_TASK_REGISTRY[:3],
        sources={"design_predictive_selection": object()},
        state=state,
    )
    assert processed["design_predictive_selection"].status == "failed"
    assert processed["design_predictive_selection"].payload["selected_n_states"] is None
    assert checkpoints == {}
    assert selected_n_states is None

    failed_input = _tampered_dataclass(
        failed_selection,
        gate_id="input_integrity",
        summary=MappingProxyType({}),
    )
    monkeypatch.setattr(
        science_module,
        "derive_g3_gate_decision",
        lambda _task_id, _source: failed_input,
    )
    input_processed, _, _ = science_module._advance_canonical_tasks(
        store=fake_store,
        lease=cast(Any, SimpleNamespace(run_id=run_id)),
        checkpoint_store=cast(Any, object()),
        binding=binding,
        task_specs=science_module.G3_TASK_REGISTRY[:1],
        sources={"input_integrity": object()},
        state=science_module.CanonicalG3State(
            run_id=run_id,
            task_results=MappingProxyType({}),
            checkpoints=MappingProxyType({}),
            selected_n_states=None,
        ),
    )
    assert input_processed["input_integrity"].status == "failed"


def test_failed_selection_resume_rederives_none_without_promoting_runner_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    _permit_legacy_coordinator_fixtures(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    gate = _candidate_gate()
    failed_gate = replace(
        gate,
        selection=replace(
            gate.selection,
            selected_n_states=None,
            passed=False,
            failure_reason="no_predictively_eligible_candidate",
        ),
    )
    handlers = _handlers(
        _input(preflight, run_id),
        design_candidate_viability=failed_gate,
        design_predictive_selection=failed_gate,
    )
    original_write = store.write_task_result

    def interrupt_after_selection(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("task_id") == "sealed_holdout_veto":
            raise RuntimeError("interrupt after failed selection")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(store, "write_task_result", interrupt_after_selection)
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )
    with pytest.raises(RuntimeError, match="after failed selection"):
        science_module.coordinate_canonical_g3_handlers(
            store=store,
            lease=lease,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    lease.close()
    selection = store.read_task_result(run_id, "design_predictive_selection")
    assert selection.status == "failed"
    assert selection.payload["selected_n_states"] is None

    monkeypatch.setattr(store, "write_task_result", original_write)
    with store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
        resume=True,
    ) as resumed:
        result = science_module.coordinate_canonical_g3_handlers(
            store=store,
            lease=resumed,
            live_preflight=preflight,
            checkpoint_store=checkpoint_store,
            handlers=handlers,
        )
    assert result.selected_n_states is None
    assert result.failed_task_ids == ("design_predictive_selection",)
    assert result.blocked_task_ids == G3_GATE_ORDER[3:]


def test_durable_publication_requires_sealed_consistent_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prpg.model.g3_evidence_store import G3EvidenceStore

    run_id = "a" * 64
    completion_fingerprint = "b" * 64
    design = object()
    production = object()
    pair = SimpleNamespace(design=design, production=production)
    coordination = SimpleNamespace(
        evidence_bundle=SimpleNamespace(passed=True),
        run_id=run_id,
        completion_fingerprint=completion_fingerprint,
    )

    def finalization(marker_state: str) -> science_module.CanonicalG3FinalizationResult:
        value = object.__new__(science_module.CanonicalG3FinalizationResult)
        for name, item in (
            ("coordination", coordination),
            (
                "completion_marker",
                SimpleNamespace(
                    state=marker_state,
                    run_id=run_id,
                    fingerprint=completion_fingerprint,
                ),
            ),
            (
                "checkpoint_map",
                MappingProxyType(dict.fromkeys(G3_GATE_ORDER, object())),
            ),
            ("artifact_pair", pair),
            ("design_artifact", design),
            ("production_artifact", production),
            ("_seal", science_module._FINALIZATION_SEAL),
        ):
            object.__setattr__(value, name, item)
        return value

    call_kwargs = {
        "run_store": cast(Any, object()),
        "checkpoint_store": cast(Any, object()),
        "live_preflight": cast(Any, object()),
        "model_store": cast(Any, object()),
    }
    with pytest.raises(ModelError, match="sealed finalization"):
        science_module.publish_finalized_canonical_g3_evidence(
            cast(Any, object()),
            evidence_store=cast(Any, object()),
            **call_kwargs,
        )

    sealed = finalization("completed")
    with pytest.raises(ModelError, match="requires G3EvidenceStore"):
        science_module.publish_finalized_canonical_g3_evidence(
            sealed,
            evidence_store=cast(Any, object()),
            **call_kwargs,
        )

    evidence_store = G3EvidenceStore(tmp_path / "evidence")
    with pytest.raises(IntegrityError, match="completion evidence changed"):
        science_module.publish_finalized_canonical_g3_evidence(
            finalization("running"),
            evidence_store=evidence_store,
            **call_kwargs,
        )

    sentinel = object()
    monkeypatch.setattr(
        science_module,
        "is_verified_scientific_artifact_pair",
        lambda _pair: True,
    )
    monkeypatch.setattr(
        G3EvidenceStore,
        "publish",
        lambda _self, **_kwargs: sentinel,
    )
    assert (
        science_module.publish_finalized_canonical_g3_evidence(
            sealed,
            evidence_store=evidence_store,
            **call_kwargs,
        )
        is sentinel
    )


def test_restore_rederives_and_rejects_nonprefix_or_mutated_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _permit_test_preflight(monkeypatch)
    store, run_id, preflight = _run(tmp_path)
    checkpoint_store = CalibrationCheckpointStore(tmp_path / "artifacts")
    input_view = _input(preflight, run_id)
    typed_handlers = _handlers(input_view)
    failure_handlers = _handlers(_failure("input_integrity"))
    lease = store.start_launch(
        run_id,
        preflight_fingerprint=preflight.fingerprint,
        workers=9,
    )

    def restore(
        existing: Mapping[str, science_module.G3TaskResult],
        handlers: CanonicalG3HandlerBundle,
    ) -> science_module.CanonicalG3State:
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "verify_task_results",
                lambda *_args, **_kwargs: dict(existing),
            )
            patch.setattr(
                store,
                "verify_planned_look_receipts",
                lambda *_args, **_kwargs: {},
            )
            return science_module.restore_canonical_g3_state(
                store=store,
                lease=lease,
                checkpoint_store=checkpoint_store,
                handlers=handlers,
            )

    second_only = _unverified_task_result(
        run_id,
        "design_candidate_viability",
        status="failed",
        payload={},
    )
    with pytest.raises(IntegrityError, match="not a registry prefix"):
        restore({"design_candidate_viability": second_only}, typed_handlers)

    blocked_input = _unverified_task_result(
        run_id,
        "input_integrity",
        status="blocked",
        payload={},
    )
    with pytest.raises(IntegrityError, match="blocked despite passed dependencies"):
        restore({"input_integrity": blocked_input}, typed_handlers)

    changed_failure = _unverified_task_result(
        run_id,
        "input_integrity",
        status="failed",
        payload={"caller_claim": "scientific failure"},
    )
    with pytest.raises(IntegrityError, match="scientific-failure receipt changed"):
        restore({"input_integrity": changed_failure}, failure_handlers)

    changed_decision = _unverified_task_result(
        run_id,
        "input_integrity",
        status="passed",
        payload={"caller_claim": "passed"},
    )
    with pytest.raises(IntegrityError, match="payload differs from typed source"):
        restore({"input_integrity": changed_decision}, typed_handlers)

    input_decision = assembly_module.derive_g3_gate_decision(
        "input_integrity",
        input_view,
    )
    document = science_module._decision_document(input_decision, input_view)
    exact_payload = science_module._decision_payload(
        checkpoint_fingerprint=None,
        decision_fingerprint=science_module._fingerprint(document),
        decision=input_decision,
        selected_n_states=None,
        planned_look_fingerprints={},
    )
    wrong_status = _unverified_task_result(
        run_id,
        "input_integrity",
        status="failed",
        payload=exact_payload,
    )
    with pytest.raises(IntegrityError, match="status differs from typed decision"):
        restore({"input_integrity": wrong_status}, typed_handlers)

    failed_input_decision = _tampered_dataclass(
        input_decision,
        passed=False,
        evidence=MappingProxyType({"integrity": "failed"}),
        summary=MappingProxyType({}),
    )
    failed_document = science_module._decision_document(
        failed_input_decision,
        input_view,
    )
    typed_failure = _unverified_task_result(
        run_id,
        "input_integrity",
        status="failed",
        payload=science_module._decision_payload(
            checkpoint_fingerprint=None,
            decision_fingerprint=science_module._fingerprint(failed_document),
            decision=failed_input_decision,
            selected_n_states=None,
            planned_look_fingerprints={},
        ),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            science_module,
            "derive_g3_gate_decision",
            lambda _task_id, _source: failed_input_decision,
        )
        restored_failure = restore(
            {"input_integrity": typed_failure},
            typed_handlers,
        )
    assert restored_failure.selected_n_states is None
    assert restored_failure.task_results["input_integrity"].status == "failed"

    exact_failure = _unverified_task_result(
        run_id,
        "input_integrity",
        status="failed",
        payload=science_module._failure_payload(
            _failure("input_integrity"),
            selected_n_states=None,
            planned_look_fingerprints={},
        ),
    )
    wrong_blocked = _unverified_task_result(
        run_id,
        "design_candidate_viability",
        status="failed",
        payload={},
    )
    with pytest.raises(
        IntegrityError, match="blocked result is not coordinator-derived"
    ):
        restore(
            {
                "input_integrity": exact_failure,
                "design_candidate_viability": wrong_blocked,
            },
            failure_handlers,
        )
    lease.close()
