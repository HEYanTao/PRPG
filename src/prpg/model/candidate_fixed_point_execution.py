"""Closed numerical execution of the registered design fixed point.

The authority types in :mod:`prpg.model.registered_fixed_point` validate a
complete iteration history but deliberately do not run scientific work.  This
module supplies the missing launch path.  It prepares every common registered
parent exactly once, reruns only the block-length-dependent candidate evidence
at each iteration, and hands the first scientifically terminal history to the
registered assembler.

The public entry point exposes no scientific callback, handler map, reduced
geometry, iteration limit, or favorable-history selector.  Module-private
stage functions exist only to keep the orchestration auditable and to permit
reduced registered fixtures in tests.  The prospective owner policy fixes
``K=4``: ordinary main fits for ``K=2,3,5`` remain descriptive diagnostics,
while only ``K=4`` receives rolling, macro-stability, and block-calibration
work.  Production exceptions are not caught: only the narrow producer-owned
scientific outcomes represented by registered parents can become candidate or
fixed-point outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

from prpg.errors import ModelError
from prpg.model.block_execution import CANONICAL_HISTORY_COUNT
from prpg.model.block_execution_registered import (
    RegisteredBlockUniformMaterialization,
    materialize_registered_block_uniforms,
    registered_block_core_identity,
    run_registered_materialized_block_calibration_outcome,
)
from prpg.model.calibration import (
    CandidateFitSet,
    DecodedReturnPools,
    MacroStabilityEvidence,
    PreliminaryCalibrations,
    RegisteredMacroBootstrapMaterialization,
    RegisteredPreliminaryCalibrationOutcome,
    RegisteredRollingBootstrapMaterialization,
    RollingCandidateEvidence,
    decoded_return_pools,
    execute_registered_hmm_candidate_set,
    execute_registered_macro_stability,
    execute_registered_preliminary_calibrations,
    execute_registered_rolling_candidate,
    materialize_registered_macro_bootstrap,
    materialize_registered_rolling_bootstrap,
    prepare_hmm_feature_matrix,
    registered_preliminary_calibrations,
)
from prpg.model.candidate_gate import (
    CandidateGateResult,
    CandidateViabilityResult,
    execute_registered_candidate_viability,
    run_owner_fixed_k4_selection,
)
from prpg.model.candidate_suite import (
    RegisteredBlockOutcome,
    RegisteredDesignCandidateSuite,
    assemble_registered_design_candidate_suite,
)
from prpg.model.fixed_point import MAXIMUM_FIXED_POINT_ITERATIONS, FixedPointState
from prpg.model.hmm import HMMFeatureMatrix
from prpg.model.input import CalibrationInput, CalibrationSlice
from prpg.model.registered_fixed_point import (
    IterationSource,
    RegisteredDesignFixedPoint,
    assemble_registered_design_fixed_point,
)
from prpg.model.selection import OWNER_FIXED_N_STATES
from prpg.simulation.rng import CalibrationArtifact


@dataclass(frozen=True, slots=True)
class _RegisteredCommonSources:
    """Exact parents reused unchanged across every numerical iteration."""

    design_slice: CalibrationSlice
    design_features: HMMFeatureMatrix
    preliminary_outcome: RegisteredPreliminaryCalibrationOutcome
    preliminary: PreliminaryCalibrations
    fit_set: CandidateFitSet
    rolling_bootstrap: RegisteredRollingBootstrapMaterialization
    macro_materialization: RegisteredMacroBootstrapMaterialization
    monthly_materialization: RegisteredBlockUniformMaterialization
    daily_materialization: RegisteredBlockUniformMaterialization
    decoded_pools_by_k: tuple[DecodedReturnPools | None, ...]
    rolling_by_k: tuple[RollingCandidateEvidence | None, ...]
    macro_by_k: tuple[MacroStabilityEvidence | None, ...]
    master_seed: int
    scientific_version: int


@dataclass(frozen=True, slots=True)
class _RegisteredIterationExecution:
    """One exact iteration source plus the state needed to drive the loop."""

    source: IterationSource
    state: FixedPointState | None


def execute_registered_design_fixed_point(
    calibration_input: CalibrationInput,
    *,
    master_seed: int,
    scientific_version: int,
    workers: int = 1,
) -> RegisteredDesignFixedPoint:
    """Run and mint the complete design candidate/block fixed point.

    Iteration one starts from the exact registered PPW decisions.  Every later
    iteration starts from the immediately preceding selected candidate's
    finalized monthly and daily block lengths.  The function stops only for a
    terminal viability/selection result, adjacent state equality,
    non-adjacent repetition, or the binding fifth iteration.  The registered
    assembler independently rederives that termination before minting the
    returned authority.
    """

    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("design fixed-point execution requires CalibrationInput")
    worker_count = _validated_workers(workers)
    common = _prepare_common_registered_sources(
        calibration_input,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=worker_count,
    )
    return _execute_registered_iteration_loop(common)


def _prepare_common_registered_sources(
    calibration_input: CalibrationInput,
    *,
    master_seed: int,
    scientific_version: int,
    workers: int,
) -> _RegisteredCommonSources:
    """Prepare all L-independent registered sources and families once."""

    design = calibration_input.design
    features = _prepare_design_features(calibration_input)
    preliminary_outcome = execute_registered_preliminary_calibrations(
        design,
        role="design",
    )
    preliminary = registered_preliminary_calibrations(preliminary_outcome)
    fit_set = execute_registered_hmm_candidate_set(
        features,
        master_seed=master_seed,
        scientific_version=scientific_version,
        workers=workers,
    )
    rolling_bootstrap = materialize_registered_rolling_bootstrap(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
    )
    macro_materialization = materialize_registered_macro_bootstrap(
        design.macro_features,
        design.monthly_period_ordinals,
        design.monthly_continuation,
        role="design",
        macro_block_length=preliminary.macro.combined.starting_length,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )

    decoded: list[DecodedReturnPools | None] = []
    rolling: list[RollingCandidateEvidence | None] = []
    macro: list[MacroStabilityEvidence | None] = []
    for attempt in fit_set.attempts:
        fit = attempt.fit
        if fit is None:
            decoded.append(None)
            rolling.append(None)
            macro.append(None)
            continue
        decoded.append(decoded_return_pools(fit, design))
        if attempt.n_states != OWNER_FIXED_N_STATES:
            # These fits are retained for descriptive fit/timing evidence only.
            # They must not expand the registered K4 stability workload or
            # influence the fixed-point controller.
            rolling.append(None)
            macro.append(None)
            continue
        rolling.append(
            execute_registered_rolling_candidate(
                design.macro_features,
                design.monthly_period_ordinals,
                design.monthly_continuation,
                role="design",
                main_fit=fit,
                master_seed=master_seed,
                scientific_version=scientific_version,
                workers=workers,
            )
        )
        macro.append(
            execute_registered_macro_stability(
                design.macro_features,
                design.monthly_period_ordinals,
                design.monthly_continuation,
                role="design",
                main_fit=fit,
                macro_block_length=preliminary.macro.combined.starting_length,
                master_seed=master_seed,
                scientific_version=scientific_version,
                source_materialization=macro_materialization,
                workers=workers,
            )
        )

    # Canonical materializations deliberately expose no reduced geometry or
    # count through the public driver.  Tests replace this private preparation
    # stage with already registered reduced fixtures.
    monthly_materialization = materialize_registered_block_uniforms(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact="design",
        frequency="monthly",
        history_count=CANONICAL_HISTORY_COUNT,
        strict_canonical=True,
    )
    daily_materialization = materialize_registered_block_uniforms(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact="design",
        frequency="daily",
        history_count=CANONICAL_HISTORY_COUNT,
        strict_canonical=True,
    )
    return _RegisteredCommonSources(
        design_slice=design,
        design_features=features,
        preliminary_outcome=preliminary_outcome,
        preliminary=preliminary,
        fit_set=fit_set,
        rolling_bootstrap=rolling_bootstrap,
        macro_materialization=macro_materialization,
        monthly_materialization=monthly_materialization,
        daily_materialization=daily_materialization,
        decoded_pools_by_k=tuple(decoded),
        rolling_by_k=tuple(rolling),
        macro_by_k=tuple(macro),
        master_seed=master_seed,
        scientific_version=scientific_version,
    )


def _validated_workers(value: int) -> int:
    """Validate the single local HMM restart-pool width for this execution."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError("design fixed-point workers must be a positive integer")
    return value


def _prepare_design_features(calibration_input: CalibrationInput) -> HMMFeatureMatrix:
    """Fit the scaler on the exact design prefix and return only that prefix."""

    design_rows = len(calibration_input.design.macro_features)
    return prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="design",
        design_rows=design_rows,
    )


def _execute_registered_iteration_loop(
    common: _RegisteredCommonSources,
) -> RegisteredDesignFixedPoint:
    """Execute until the earliest binding terminal condition, never sooner."""

    history: list[IterationSource] = []
    seen: set[FixedPointState] = set()
    previous: FixedPointState | None = None
    monthly_start = common.preliminary.monthly.combined.starting_length
    daily_start = common.preliminary.daily.combined.starting_length

    for iteration in range(1, MAXIMUM_FIXED_POINT_ITERATIONS + 1):
        current = _execute_registered_iteration(
            common,
            iteration=iteration,
            monthly_start=monthly_start,
            daily_start=daily_start,
        )
        history.append(current.source)
        state = current.state
        if state is None:
            break
        if previous is not None and state == previous:
            break
        if state in seen:
            break
        seen.add(state)
        previous = state
        monthly_start = state.monthly_length
        daily_start = state.daily_length

    # This is intentionally the only mint point.  The assembler independently
    # rejects favorable truncation, records after termination, a false
    # convergence, or any mismatch between the driving state and exact parents.
    return assemble_registered_design_fixed_point(tuple(history))


def _execute_registered_iteration(
    common: _RegisteredCommonSources,
    *,
    iteration: int,
    monthly_start: int,
    daily_start: int,
) -> _RegisteredIterationExecution:
    """Rerun only L-dependent block outcomes and deterministic reductions."""

    monthly_blocks, daily_blocks = _execute_registered_block_outcomes(
        common,
        monthly_start=monthly_start,
        daily_start=daily_start,
    )
    suite = assemble_registered_design_candidate_suite(
        common.design_slice,
        common.design_features,
        common.fit_set,
        preliminary_outcome=common.preliminary_outcome,
        macro_materialization=common.macro_materialization,
        monthly_materialization=common.monthly_materialization,
        daily_materialization=common.daily_materialization,
        rolling_by_k=common.rolling_by_k,
        macro_by_k=common.macro_by_k,
        monthly_blocks_by_k=monthly_blocks,
        daily_blocks_by_k=daily_blocks,
        master_seed=common.master_seed,
        scientific_version=common.scientific_version,
        iteration=iteration,
        monthly_input_start=(None if iteration == 1 else monthly_start),
        daily_input_start=(None if iteration == 1 else daily_start),
    )
    viability = execute_registered_candidate_viability(suite)
    selection = run_owner_fixed_k4_selection(viability, common.rolling_bootstrap)
    source = (suite, viability, selection)
    return _RegisteredIterationExecution(
        source=source,
        state=_derive_iteration_state(suite, viability, selection),
    )


def _execute_registered_block_outcomes(
    common: _RegisteredCommonSources,
    *,
    monthly_start: int,
    daily_start: int,
) -> tuple[
    tuple[RegisteredBlockOutcome | None, ...], tuple[RegisteredBlockOutcome | None, ...]
]:
    """Execute both frequency outcomes only for the fixed owner model K=4."""

    monthly: list[RegisteredBlockOutcome | None] = []
    daily: list[RegisteredBlockOutcome | None] = []
    monthly_links = common.design_slice.monthly_continuation[1:]
    daily_links = common.design_slice.daily_continuation[1:]
    for attempt, pools in zip(
        common.fit_set.attempts,
        common.decoded_pools_by_k,
        strict=True,
    ):
        fit = attempt.fit
        if fit is None:
            if pools is not None:
                raise ModelError("failed candidate fit retained decoded pools")
            monthly.append(None)
            daily.append(None)
            continue
        if pools is None:
            raise ModelError("successful candidate fit lacks decoded pools")
        if attempt.n_states != OWNER_FIXED_N_STATES:
            monthly.append(None)
            daily.append(None)
            continue
        monthly.append(
            run_registered_materialized_block_calibration_outcome(
                common.monthly_materialization,
                model=fit,
                starting_length=monthly_start,
                historical_source_states=pools.monthly_states,
                source_continuation_links=monthly_links,
            )
        )
        daily.append(
            run_registered_materialized_block_calibration_outcome(
                common.daily_materialization,
                model=fit,
                starting_length=daily_start,
                historical_source_states=pools.daily_states,
                source_continuation_links=daily_links,
            )
        )
    return tuple(monthly), tuple(daily)


def _derive_iteration_state(
    suite: RegisteredDesignCandidateSuite,
    viability: CandidateViabilityResult,
    selection: CandidateGateResult,
) -> FixedPointState | None:
    """Derive only the controller state; the authority assembler rederives it."""

    admissible = tuple(
        row.n_states for row in viability.audit_rows if row.viability.admissible
    )
    selected = selection.selection.selected_n_states
    if not admissible:
        if selection.selection.passed or selected is not None:
            raise ModelError("empty candidate viability has a passed selection")
        return None
    if not selection.selection.passed:
        if selected is not None:
            raise ModelError("failed candidate selection retained a selected K")
        return None
    if selected is None or selected not in admissible:
        raise ModelError("passed candidate selection lacks an admissible K")
    row = next((item for item in suite.rows if item.n_states == selected), None)
    if row is None or row.monthly_block is None or row.daily_block is None:
        raise ModelError("selected candidate lacks successful block outcomes")
    monthly = registered_block_core_identity(row.monthly_block)
    daily = registered_block_core_identity(row.daily_block)
    return FixedPointState(
        selected,
        monthly.selected_length,
        daily.selected_length,
    )
