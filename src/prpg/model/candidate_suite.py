"""Producer-sealed evidence suite for one design fixed-point iteration.

The ordinary candidate gate intentionally accepts compact caller-assembled
evidence for exploratory reports.  Canonical G3 needs a stronger boundary:
every successful K must descend from the registered design fit family.  Under
the prospective owner-fixed-K4 policy, successful K=2,3,5 fits may be retained
as report-only main-fit diagnostics with no downstream parents; K=4 alone
must descend from the same exact source history, purpose-0 macro draw set, and
purpose-4 block draw sets.  Legacy complete non-K rows remain verifiable for
old report fixtures but are never produced by the fixed-K4 executor.  This
module verifies the graph, derives decoded pools and ``CandidateGateEvidence``
internally, and seals the resulting composite.  It does not claim that this
iteration is the converged final iteration; a separate registered fixed-point
wrapper must retain the complete history and grant the task-2/task-3 authority
described by Section 8.2.

The suite does not draw randomness or accept an injectable scientific
callable.  Its parents are already producer-authorized executor results.  A
process-local mint record retains their exact object identities so a copied
suite, a content-equivalent substitute parent, or a private-field rewrite
cannot acquire authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, TypeGuard, cast

import numpy as np
import pandas as pd

from prpg.errors import ModelError
from prpg.model.block_execution import CANONICAL_HISTORY_COUNT
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    RegisteredBlockSupportFailure,
    RegisteredBlockUniformMaterialization,
    registered_block_core_identity,
    registered_block_replay_source,
    registered_block_support_failure_identity,
    registered_block_uniform_identity,
)
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.calibration import (
    VERSION_1_CANDIDATES,
    CandidateFitSet,
    DecodedReturnPools,
    MacroStabilityEvidence,
    PreliminaryCalibrations,
    RegisteredMacroBootstrapMaterialization,
    RegisteredPreliminaryCalibrationOutcome,
    RollingCandidateEvidence,
    decoded_return_pools,
    preliminary_calibrations_fingerprint,
    registered_candidate_fit_set_identity,
    registered_macro_bootstrap_materialization_identity,
    registered_macro_stability_execution_identity,
    registered_preliminary_calibration_identity,
    registered_preliminary_calibrations,
    registered_rolling_candidate_identity,
)
from prpg.model.calibration_identity import (
    calibration_slice_fingerprint,
    candidate_fit_set_fingerprint,
    decoded_return_pools_fingerprint,
)
from prpg.model.candidate_gate import (
    CandidateGateEvidence,
    candidate_gate_source_evidence_fingerprint,
)
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    sequence_lengths_from_continuation,
)
from prpg.model.input import CalibrationSlice
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.model.selection import OWNER_FIXED_N_STATES

_SUITE_CAPABILITY = object()
RegisteredBlockOutcome: TypeAlias = (
    RegisteredBlockCalibration | RegisteredBlockSupportFailure
)


@dataclass(frozen=True, slots=True)
class RegisteredDesignCandidateSuiteRow:
    """One K row; failed fits have an exact empty downstream evidence set."""

    n_states: int
    decoded_return_pools: DecodedReturnPools | None
    rolling: RollingCandidateEvidence | None
    macro_stability: MacroStabilityEvidence | None
    monthly_block: RegisteredBlockCalibration | None
    daily_block: RegisteredBlockCalibration | None
    monthly_block_failure: RegisteredBlockSupportFailure | None
    daily_block_failure: RegisteredBlockSupportFailure | None


@dataclass(frozen=True, slots=True)
class RegisteredDesignCandidateSuite:
    """Exact K=2..5 evidence graph for one registered design iteration."""

    contract_version: str
    iteration: int
    master_seed: int
    scientific_version: int
    preliminary: PreliminaryCalibrations
    monthly_input_start: int
    daily_input_start: int
    rows: tuple[
        RegisteredDesignCandidateSuiteRow,
        RegisteredDesignCandidateSuiteRow,
        RegisteredDesignCandidateSuiteRow,
        RegisteredDesignCandidateSuiteRow,
    ]
    strict_canonical_geometry: bool
    suite_fingerprint: str
    _design_slice: CalibrationSlice = field(repr=False, compare=False)
    _design_features: HMMFeatureMatrix = field(repr=False, compare=False)
    _fit_set: CandidateFitSet = field(repr=False, compare=False)
    _preliminary_outcome: RegisteredPreliminaryCalibrationOutcome = field(
        repr=False,
        compare=False,
    )
    _macro_materialization: RegisteredMacroBootstrapMaterialization = field(
        repr=False,
        compare=False,
    )
    _monthly_materialization: RegisteredBlockUniformMaterialization = field(
        repr=False,
        compare=False,
    )
    _daily_materialization: RegisteredBlockUniformMaterialization = field(
        repr=False,
        compare=False,
    )
    _gate_evidence: tuple[
        CandidateGateEvidence,
        CandidateGateEvidence,
        CandidateGateEvidence,
        CandidateGateEvidence,
    ] = field(repr=False, compare=False)
    _execution_seal: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class RegisteredDesignCandidateSuiteIdentity:
    """Reverified component, materialization, and RNG identity of one suite."""

    suite_fingerprint: str
    iteration: int
    design_slice_fingerprint: str
    design_feature_fingerprint: str
    preliminary_fingerprint: str
    preliminary_outcome_fingerprint: str
    fit_set_fingerprint: str
    source_content_fingerprint: str
    main_fit_rng_execution_fingerprint: str
    rolling_fit_rng_execution_fingerprints: tuple[str | None, ...]
    combined_fit_rng_execution_fingerprint: str
    macro_materialization_fingerprint: str
    macro_materialization_rng_execution_fingerprint: str
    macro_fit_rng_execution_fingerprints: tuple[str | None, ...]
    monthly_block_materialization_fingerprint: str
    monthly_block_rng_execution_fingerprint: str
    daily_block_materialization_fingerprint: str
    daily_block_rng_execution_fingerprint: str
    block_core_fingerprints: tuple[tuple[str | None, str | None], ...]
    block_failure_fingerprints: tuple[tuple[str | None, str | None], ...]
    combined_materialization_fingerprint: str
    combined_rng_execution_fingerprint: str
    master_seed: int
    scientific_version: int
    strict_canonical_geometry: bool
    monthly_input_start: int
    daily_input_start: int


@dataclass(frozen=True, slots=True)
class _SuiteMintRecord:
    suite: RegisteredDesignCandidateSuite
    design_slice: CalibrationSlice
    design_features: HMMFeatureMatrix
    fit_set: CandidateFitSet
    preliminary_outcome: RegisteredPreliminaryCalibrationOutcome
    macro_materialization: RegisteredMacroBootstrapMaterialization
    monthly_materialization: RegisteredBlockUniformMaterialization
    daily_materialization: RegisteredBlockUniformMaterialization
    rows: object
    gate_evidence: object
    rolling_parents: tuple[RollingCandidateEvidence | None, ...]
    macro_parents: tuple[MacroStabilityEvidence | None, ...]
    monthly_block_parents: tuple[RegisteredBlockOutcome | None, ...]
    daily_block_parents: tuple[RegisteredBlockOutcome | None, ...]
    suite_fingerprint: str


_MINTED_SUITES: dict[int, _SuiteMintRecord] = {}
REGISTERED_DESIGN_CANDIDATE_SUITE_CONTRACT = "registered-design-candidate-suite-v2"


def assemble_registered_design_candidate_suite(
    design_slice: CalibrationSlice,
    design_features: HMMFeatureMatrix,
    fit_set: CandidateFitSet,
    *,
    preliminary_outcome: RegisteredPreliminaryCalibrationOutcome,
    macro_materialization: RegisteredMacroBootstrapMaterialization,
    monthly_materialization: RegisteredBlockUniformMaterialization,
    daily_materialization: RegisteredBlockUniformMaterialization,
    rolling_by_k: Sequence[RollingCandidateEvidence | None],
    macro_by_k: Sequence[MacroStabilityEvidence | None],
    monthly_blocks_by_k: Sequence[RegisteredBlockOutcome | None],
    daily_blocks_by_k: Sequence[RegisteredBlockOutcome | None],
    master_seed: int,
    scientific_version: int,
    iteration: int = 1,
    monthly_input_start: int | None = None,
    daily_input_start: int | None = None,
) -> RegisteredDesignCandidateSuite:
    """Verify registered parents and mint one non-injectable composite suite."""

    if not isinstance(design_slice, CalibrationSlice) or design_slice.role != "design":
        raise ModelError("candidate suite requires the exact design slice")
    _validate_seed_version(master_seed, scientific_version)
    checked_iteration = _fixed_point_iteration(iteration)
    fit_identity = registered_candidate_fit_set_identity(fit_set)
    if (
        fit_identity.master_seed != master_seed
        or fit_identity.scientific_version != scientific_version
        or fit_set._features is not design_features
        or fit_identity.feature_fingerprint
        != hmm_feature_matrix_fingerprint(design_features)
    ):
        raise ModelError("candidate suite fit-family ancestry is inconsistent")
    _verify_design_features(design_slice, design_features)
    preliminary_identity = registered_preliminary_calibration_identity(
        preliminary_outcome
    )
    if (
        not preliminary_identity.passed
        or preliminary_identity.role != "design"
        or preliminary_outcome._source_slice is not design_slice
    ):
        raise ModelError("candidate suite requires its exact passed preliminary parent")
    preliminary = registered_preliminary_calibrations(preliminary_outcome)
    if checked_iteration > 1 and (
        monthly_input_start is None or daily_input_start is None
    ):
        raise ModelError("later candidate suites require explicit prior-final starts")
    monthly_start = _iteration_input_length(
        monthly_input_start,
        preliminary=preliminary.monthly.combined.starting_length,
        frequency="monthly",
    )
    daily_start = _iteration_input_length(
        daily_input_start,
        preliminary=preliminary.daily.combined.starting_length,
        frequency="daily",
    )
    macro_source = registered_macro_bootstrap_materialization_identity(
        macro_materialization
    )
    monthly_source = registered_block_uniform_identity(monthly_materialization)
    daily_source = registered_block_uniform_identity(daily_materialization)
    if (
        macro_source.role != "design"
        or macro_source.macro_block_length != preliminary.macro.combined.starting_length
        or macro_source.master_seed != master_seed
        or macro_source.scientific_version != scientific_version
        or monthly_source.artifact != "design"
        or monthly_source.frequency != "monthly"
        or daily_source.artifact != "design"
        or daily_source.frequency != "daily"
        or monthly_source.master_seed != master_seed
        or daily_source.master_seed != master_seed
        or monthly_source.scientific_version != scientific_version
        or daily_source.scientific_version != scientific_version
    ):
        raise ModelError("candidate suite common materialization ancestry differs")
    expected_macro_input = scientific_materialization_fingerprint(
        schema_id="macro_stability_input",
        metadata={"role": "design"},
        arrays={
            "continuation": design_slice.monthly_continuation,
            "macro_features": design_slice.macro_features,
            "month_ordinals": design_slice.monthly_period_ordinals,
        },
    )
    if macro_source.source_input_fingerprint != expected_macro_input:
        raise ModelError("candidate suite macro materialization has another history")

    rolling = cast(
        tuple[RollingCandidateEvidence | None, ...],
        _ordered_parent_tuple(rolling_by_k, "rolling"),
    )
    macro = cast(
        tuple[MacroStabilityEvidence | None, ...],
        _ordered_parent_tuple(macro_by_k, "macro"),
    )
    monthly_blocks = cast(
        tuple[RegisteredBlockOutcome | None, ...],
        _ordered_parent_tuple(monthly_blocks_by_k, "monthly block"),
    )
    daily_blocks = cast(
        tuple[RegisteredBlockOutcome | None, ...],
        _ordered_parent_tuple(daily_blocks_by_k, "daily block"),
    )
    rows: list[RegisteredDesignCandidateSuiteRow] = []
    evidence: list[CandidateGateEvidence] = []
    for index, attempt in enumerate(fit_set.attempts):
        n_states = attempt.n_states
        if attempt.fit is None:
            if any(
                item is not None
                for item in (
                    rolling[index],
                    macro[index],
                    monthly_blocks[index],
                    daily_blocks[index],
                )
            ):
                raise ModelError(
                    "failed candidate fits require exact empty downstream evidence"
                )
            row = RegisteredDesignCandidateSuiteRow(
                n_states,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            gate_row = CandidateGateEvidence(
                n_states,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            downstream = (
                rolling[index],
                macro[index],
                monthly_blocks[index],
                daily_blocks[index],
            )
            if n_states != OWNER_FIXED_N_STATES and all(
                item is None for item in downstream
            ):
                # The prospective fixed-K4 executor retains ordinary non-K4
                # fits as descriptive diagnostics only.  Decoded pools are a
                # deterministic view of the main fit, not a stochastic heavy
                # family, and make the report-only ancestry auditable.
                pools = decoded_return_pools(attempt.fit, design_slice)
                row = RegisteredDesignCandidateSuiteRow(
                    n_states,
                    pools,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                gate_row = CandidateGateEvidence(
                    n_states,
                    pools,
                    None,
                    None,
                    None,
                    None,
                )
                rows.append(row)
                evidence.append(gate_row)
                continue
            if any(item is None for item in downstream):
                raise ModelError(
                    "successful candidate fits require every registered evidence parent"
                )
            pools = decoded_return_pools(attempt.fit, design_slice)
            rolling_parent = cast(RollingCandidateEvidence, rolling[index])
            macro_parent = cast(MacroStabilityEvidence, macro[index])
            monthly_outcome = monthly_blocks[index]
            daily_outcome = daily_blocks[index]
            if not isinstance(
                monthly_outcome,
                RegisteredBlockCalibration | RegisteredBlockSupportFailure,
            ) or not isinstance(
                daily_outcome,
                RegisteredBlockCalibration | RegisteredBlockSupportFailure,
            ):
                raise ModelError(
                    "successful candidate fits require exact block outcomes"
                )
            _verify_successful_candidate_parents(
                n_states=n_states,
                fit=attempt.fit,
                design_slice=design_slice,
                pools=pools,
                monthly_input_start=monthly_start,
                daily_input_start=daily_start,
                macro_materialization=macro_materialization,
                monthly_materialization=monthly_materialization,
                daily_materialization=daily_materialization,
                rolling=rolling_parent,
                macro=macro_parent,
                monthly_block=monthly_outcome,
                daily_block=daily_outcome,
                master_seed=master_seed,
                scientific_version=scientific_version,
            )
            monthly_parent = (
                monthly_outcome
                if isinstance(monthly_outcome, RegisteredBlockCalibration)
                else None
            )
            daily_parent = (
                daily_outcome
                if isinstance(daily_outcome, RegisteredBlockCalibration)
                else None
            )
            monthly_failure = (
                monthly_outcome
                if isinstance(monthly_outcome, RegisteredBlockSupportFailure)
                else None
            )
            daily_failure = (
                daily_outcome
                if isinstance(daily_outcome, RegisteredBlockSupportFailure)
                else None
            )
            row = RegisteredDesignCandidateSuiteRow(
                n_states,
                pools,
                rolling_parent,
                macro_parent,
                monthly_parent,
                daily_parent,
                monthly_failure,
                daily_failure,
            )
            gate_row = CandidateGateEvidence(
                n_states,
                pools,
                rolling_parent,
                macro_parent,
                (
                    None
                    if monthly_parent is None
                    else monthly_parent.execution.policy_selection
                ),
                (
                    None
                    if daily_parent is None
                    else daily_parent.execution.policy_selection
                ),
                (
                    None
                    if monthly_parent is None
                    else monthly_parent.execution.fragmentation_passed
                ),
                (
                    None
                    if daily_parent is None
                    else daily_parent.execution.fragmentation_passed
                ),
            )
        rows.append(row)
        evidence.append(gate_row)

    checked_rows = cast(
        tuple[
            RegisteredDesignCandidateSuiteRow,
            RegisteredDesignCandidateSuiteRow,
            RegisteredDesignCandidateSuiteRow,
            RegisteredDesignCandidateSuiteRow,
        ],
        tuple(rows),
    )
    checked_evidence = cast(
        tuple[
            CandidateGateEvidence,
            CandidateGateEvidence,
            CandidateGateEvidence,
            CandidateGateEvidence,
        ],
        tuple(evidence),
    )
    strict = bool(
        monthly_materialization.geometry.strict_canonical
        and daily_materialization.geometry.strict_canonical
        and monthly_materialization.history_count == CANONICAL_HISTORY_COUNT
        and daily_materialization.history_count == CANONICAL_HISTORY_COUNT
    )
    provisional = RegisteredDesignCandidateSuite(
        contract_version=REGISTERED_DESIGN_CANDIDATE_SUITE_CONTRACT,
        iteration=checked_iteration,
        master_seed=master_seed,
        scientific_version=scientific_version,
        preliminary=preliminary,
        monthly_input_start=monthly_start,
        daily_input_start=daily_start,
        rows=checked_rows,
        strict_canonical_geometry=strict,
        suite_fingerprint="",
        _design_slice=design_slice,
        _design_features=design_features,
        _fit_set=fit_set,
        _preliminary_outcome=preliminary_outcome,
        _macro_materialization=macro_materialization,
        _monthly_materialization=monthly_materialization,
        _daily_materialization=daily_materialization,
        _gate_evidence=checked_evidence,
    )
    identity = _derive_suite_identity(provisional)
    result = RegisteredDesignCandidateSuite(
        contract_version=provisional.contract_version,
        iteration=checked_iteration,
        master_seed=master_seed,
        scientific_version=scientific_version,
        preliminary=preliminary,
        monthly_input_start=monthly_start,
        daily_input_start=daily_start,
        rows=checked_rows,
        strict_canonical_geometry=strict,
        suite_fingerprint=identity.suite_fingerprint,
        _design_slice=design_slice,
        _design_features=design_features,
        _fit_set=fit_set,
        _preliminary_outcome=preliminary_outcome,
        _macro_materialization=macro_materialization,
        _monthly_materialization=monthly_materialization,
        _daily_materialization=daily_materialization,
        _gate_evidence=checked_evidence,
    )
    object.__setattr__(result, "_execution_seal", _SUITE_CAPABILITY)
    _MINTED_SUITES[id(result)] = _SuiteMintRecord(
        suite=result,
        design_slice=design_slice,
        design_features=design_features,
        fit_set=fit_set,
        preliminary_outcome=preliminary_outcome,
        macro_materialization=macro_materialization,
        monthly_materialization=monthly_materialization,
        daily_materialization=daily_materialization,
        rows=result.rows,
        gate_evidence=result._gate_evidence,
        rolling_parents=rolling,
        macro_parents=macro,
        monthly_block_parents=monthly_blocks,
        daily_block_parents=daily_blocks,
        suite_fingerprint=result.suite_fingerprint,
    )
    registered_design_candidate_suite_identity(result)
    return result


def registered_design_candidate_suite_identity(
    value: RegisteredDesignCandidateSuite,
) -> RegisteredDesignCandidateSuiteIdentity:
    """Reverify the exact live parent graph and return its composite identity."""

    if not isinstance(value, RegisteredDesignCandidateSuite):
        raise ModelError("candidate suite has the wrong type")
    record = _MINTED_SUITES.get(id(value))
    if (
        value._execution_seal is not _SUITE_CAPABILITY
        or record is None
        or record.suite is not value
        or value._design_slice is not record.design_slice
        or value._design_features is not record.design_features
        or value._fit_set is not record.fit_set
        or value._preliminary_outcome is not record.preliminary_outcome
        or value._macro_materialization is not record.macro_materialization
        or value._monthly_materialization is not record.monthly_materialization
        or value._daily_materialization is not record.daily_materialization
        or value.rows is not record.rows
        or value._gate_evidence is not record.gate_evidence
    ):
        raise ModelError("candidate suite lacks exact producer authority")
    if (
        value.contract_version != REGISTERED_DESIGN_CANDIDATE_SUITE_CONTRACT
        or value.iteration != _fixed_point_iteration(value.iteration)
        or value.master_seed != record.fit_set._master_seed
        or value.scientific_version != record.fit_set._scientific_version
    ):
        raise ModelError("candidate suite contract identity changed")
    for row, rolling, macro, monthly, daily in zip(
        value.rows,
        record.rolling_parents,
        record.macro_parents,
        record.monthly_block_parents,
        record.daily_block_parents,
        strict=True,
    ):
        if (
            row.rolling is not rolling
            or row.macro_stability is not macro
            or not _row_block_parent_matches(row, monthly, frequency="monthly")
            or not _row_block_parent_matches(row, daily, frequency="daily")
        ):
            raise ModelError("candidate suite row parent substitution detected")
    current = _derive_suite_identity(value)
    if (
        current.suite_fingerprint != value.suite_fingerprint
        or current.suite_fingerprint != record.suite_fingerprint
    ):
        raise ModelError("candidate suite sources or outputs changed")
    return current


def is_registered_design_candidate_suite(
    value: object,
) -> TypeGuard[RegisteredDesignCandidateSuite]:
    """Return whether a suite is the unchanged object-specific minted instance."""

    if not isinstance(value, RegisteredDesignCandidateSuite):
        return False
    try:
        registered_design_candidate_suite_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_design_candidate_gate_evidence(
    value: RegisteredDesignCandidateSuite,
) -> tuple[
    CandidateGateEvidence,
    CandidateGateEvidence,
    CandidateGateEvidence,
    CandidateGateEvidence,
]:
    """Return the internally derived evidence only after full reverification."""

    registered_design_candidate_suite_identity(value)
    return value._gate_evidence


def registered_design_candidate_fit_set(
    value: RegisteredDesignCandidateSuite,
) -> CandidateFitSet:
    """Return the exact registered fit-family parent after reverification."""

    registered_design_candidate_suite_identity(value)
    return value._fit_set


def _derive_suite_identity(
    value: RegisteredDesignCandidateSuite,
) -> RegisteredDesignCandidateSuiteIdentity:
    _validate_seed_version(value.master_seed, value.scientific_version)
    _fixed_point_iteration(value.iteration)
    slice_fingerprint = calibration_slice_fingerprint(value._design_slice)
    feature_fingerprint = hmm_feature_matrix_fingerprint(value._design_features)
    _verify_design_features(value._design_slice, value._design_features)
    fit = registered_candidate_fit_set_identity(value._fit_set)
    if (
        fit.master_seed != value.master_seed
        or fit.scientific_version != value.scientific_version
        or fit.feature_fingerprint != feature_fingerprint
        or value._fit_set._features is not value._design_features
    ):
        raise ModelError("candidate suite live fit-family ancestry changed")
    preliminary_identity = registered_preliminary_calibration_identity(
        value._preliminary_outcome
    )
    if (
        not preliminary_identity.passed
        or preliminary_identity.role != "design"
        or value._preliminary_outcome._source_slice is not value._design_slice
    ):
        raise ModelError("candidate suite preliminary parent changed")
    recomputed_preliminary = registered_preliminary_calibrations(
        value._preliminary_outcome
    )
    preliminary_fingerprint = _preliminary_fingerprint(recomputed_preliminary)
    if _preliminary_fingerprint(value.preliminary) != preliminary_fingerprint:
        raise ModelError("candidate suite preliminary starts changed")
    monthly_start = _iteration_input_length(
        value.monthly_input_start,
        preliminary=recomputed_preliminary.monthly.combined.starting_length,
        frequency="monthly",
    )
    daily_start = _iteration_input_length(
        value.daily_input_start,
        preliminary=recomputed_preliminary.daily.combined.starting_length,
        frequency="daily",
    )
    macro_source = registered_macro_bootstrap_materialization_identity(
        value._macro_materialization
    )
    monthly_source = registered_block_uniform_identity(value._monthly_materialization)
    daily_source = registered_block_uniform_identity(value._daily_materialization)
    if (
        macro_source.role != "design"
        or macro_source.master_seed != value.master_seed
        or macro_source.scientific_version != value.scientific_version
        or macro_source.macro_block_length
        != recomputed_preliminary.macro.combined.starting_length
        or monthly_source.artifact != "design"
        or monthly_source.frequency != "monthly"
        or daily_source.artifact != "design"
        or daily_source.frequency != "daily"
        or monthly_source.master_seed != value.master_seed
        or daily_source.master_seed != value.master_seed
        or monthly_source.scientific_version != value.scientific_version
        or daily_source.scientific_version != value.scientific_version
    ):
        raise ModelError("candidate suite live common materializations changed")
    strict = bool(
        value._monthly_materialization.geometry.strict_canonical
        and value._daily_materialization.geometry.strict_canonical
        and value._monthly_materialization.history_count == CANONICAL_HISTORY_COUNT
        and value._daily_materialization.history_count == CANONICAL_HISTORY_COUNT
    )
    if value.strict_canonical_geometry is not strict:
        raise ModelError("candidate suite strict-geometry identity changed")

    rolling_rng: list[str | None] = []
    macro_rng: list[str | None] = []
    block_cores: list[tuple[str | None, str | None]] = []
    block_failures: list[tuple[str | None, str | None]] = []
    derived_evidence: list[CandidateGateEvidence] = []
    for attempt, row in zip(value._fit_set.attempts, value.rows, strict=True):
        if row.n_states != attempt.n_states:
            raise ModelError("candidate suite row ordering changed")
        if attempt.fit is None:
            if any(
                item is not None
                for item in (
                    row.decoded_return_pools,
                    row.rolling,
                    row.macro_stability,
                    row.monthly_block,
                    row.daily_block,
                    row.monthly_block_failure,
                    row.daily_block_failure,
                )
            ):
                raise ModelError("failed candidate suite row is not empty")
            rolling_rng.append(None)
            macro_rng.append(None)
            block_cores.append((None, None))
            block_failures.append((None, None))
            derived_evidence.append(
                CandidateGateEvidence(
                    attempt.n_states,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        if any(
            item is None
            for item in (
                row.decoded_return_pools,
                row.rolling,
                row.macro_stability,
            )
        ):
            if (
                attempt.n_states != OWNER_FIXED_N_STATES
                and row.decoded_return_pools is not None
                and all(
                    item is None
                    for item in (
                        row.rolling,
                        row.macro_stability,
                        row.monthly_block,
                        row.daily_block,
                        row.monthly_block_failure,
                        row.daily_block_failure,
                    )
                )
            ):
                pools = decoded_return_pools(attempt.fit, value._design_slice)
                stored_pools = row.decoded_return_pools
                if decoded_return_pools_fingerprint(stored_pools) != (
                    decoded_return_pools_fingerprint(pools)
                ):
                    raise ModelError("candidate suite decoded pools changed")
                rolling_rng.append(None)
                macro_rng.append(None)
                block_cores.append((None, None))
                block_failures.append((None, None))
                derived_evidence.append(
                    CandidateGateEvidence(
                        attempt.n_states,
                        stored_pools,
                        None,
                        None,
                        None,
                        None,
                    )
                )
                continue
            raise ModelError("successful candidate suite row is incomplete")
        monthly_outcome = _row_block_outcome(row, frequency="monthly")
        daily_outcome = _row_block_outcome(row, frequency="daily")
        pools = decoded_return_pools(attempt.fit, value._design_slice)
        stored_pools = cast(DecodedReturnPools, row.decoded_return_pools)
        if decoded_return_pools_fingerprint(stored_pools) != (
            decoded_return_pools_fingerprint(pools)
        ):
            raise ModelError("candidate suite decoded pools changed")
        rolling = cast(RollingCandidateEvidence, row.rolling)
        macro = cast(MacroStabilityEvidence, row.macro_stability)
        _verify_successful_candidate_parents(
            n_states=attempt.n_states,
            fit=attempt.fit,
            design_slice=value._design_slice,
            pools=stored_pools,
            monthly_input_start=monthly_start,
            daily_input_start=daily_start,
            macro_materialization=value._macro_materialization,
            monthly_materialization=value._monthly_materialization,
            daily_materialization=value._daily_materialization,
            rolling=rolling,
            macro=macro,
            monthly_block=monthly_outcome,
            daily_block=daily_outcome,
            master_seed=value.master_seed,
            scientific_version=value.scientific_version,
        )
        rolling_identity = registered_rolling_candidate_identity(rolling)
        macro_identity = registered_macro_stability_execution_identity(macro)
        monthly_core = (
            registered_block_core_identity(monthly_outcome)
            if isinstance(monthly_outcome, RegisteredBlockCalibration)
            else None
        )
        daily_core = (
            registered_block_core_identity(daily_outcome)
            if isinstance(daily_outcome, RegisteredBlockCalibration)
            else None
        )
        monthly_failure = (
            registered_block_support_failure_identity(monthly_outcome)
            if isinstance(monthly_outcome, RegisteredBlockSupportFailure)
            else None
        )
        daily_failure = (
            registered_block_support_failure_identity(daily_outcome)
            if isinstance(daily_outcome, RegisteredBlockSupportFailure)
            else None
        )
        monthly_success = (
            monthly_outcome
            if isinstance(monthly_outcome, RegisteredBlockCalibration)
            else None
        )
        daily_success = (
            daily_outcome
            if isinstance(daily_outcome, RegisteredBlockCalibration)
            else None
        )
        rolling_rng.append(rolling_identity.rng_execution_fingerprint)
        macro_rng.append(macro_identity.rng_execution_fingerprint)
        block_cores.append(
            (
                None if monthly_core is None else monthly_core.core_fingerprint,
                None if daily_core is None else daily_core.core_fingerprint,
            )
        )
        block_failures.append(
            (
                (
                    None
                    if monthly_failure is None
                    else monthly_failure.failure_fingerprint
                ),
                (None if daily_failure is None else daily_failure.failure_fingerprint),
            )
        )
        derived_evidence.append(
            CandidateGateEvidence(
                attempt.n_states,
                stored_pools,
                rolling,
                macro,
                (
                    None
                    if monthly_success is None
                    else monthly_success.execution.policy_selection
                ),
                (
                    None
                    if daily_success is None
                    else daily_success.execution.policy_selection
                ),
                (
                    None
                    if monthly_success is None
                    else monthly_success.execution.fragmentation_passed
                ),
                (
                    None
                    if daily_success is None
                    else daily_success.execution.fragmentation_passed
                ),
            )
        )
    checked_derived = tuple(derived_evidence)
    if candidate_gate_source_evidence_fingerprint(checked_derived) != (
        candidate_gate_source_evidence_fingerprint(value._gate_evidence)
    ):
        raise ModelError("candidate suite derived gate evidence changed")
    source_content_fingerprint = candidate_gate_source_evidence_fingerprint(
        value._gate_evidence
    )
    fit_set_fingerprint = candidate_fit_set_fingerprint(value._fit_set)
    combined_fit_rng = _composite_fingerprint(
        "design_candidate_main_and_rolling_fit_rng",
        {
            "main": fit.rng_execution_fingerprint,
            "rolling": rolling_rng,
        },
    )
    combined_materialization = _composite_fingerprint(
        "design_candidate_common_materializations",
        {
            "macro": macro_source.materialization_fingerprint,
            "monthly_block": monthly_source.materialization_fingerprint,
            "daily_block": daily_source.materialization_fingerprint,
        },
    )
    combined_rng = _composite_fingerprint(
        "design_candidate_all_rng_executions",
        {
            "main_fit": fit.rng_execution_fingerprint,
            "rolling_fits": rolling_rng,
            "macro_source": macro_source.rng_execution_fingerprint,
            "macro_fits": macro_rng,
            "monthly_block": monthly_source.rng_execution_fingerprint,
            "daily_block": daily_source.rng_execution_fingerprint,
        },
    )
    suite_fingerprint = scientific_materialization_fingerprint(
        schema_id="registered_design_candidate_suite_v2",
        metadata={
            "contract_version": value.contract_version,
            "iteration": value.iteration,
            "master_seed": value.master_seed,
            "scientific_version": value.scientific_version,
            "design_slice_fingerprint": slice_fingerprint,
            "design_feature_fingerprint": feature_fingerprint,
            "preliminary_fingerprint": preliminary_fingerprint,
            "preliminary_outcome_fingerprint": (
                preliminary_identity.outcome_fingerprint
            ),
            "monthly_input_start": monthly_start,
            "daily_input_start": daily_start,
            "fit_set_fingerprint": fit_set_fingerprint,
            "source_content_fingerprint": source_content_fingerprint,
            "main_fit_rng_execution_fingerprint": fit.rng_execution_fingerprint,
            "rolling_fit_rng_execution_fingerprints": rolling_rng,
            "macro_materialization_fingerprint": (
                macro_source.materialization_fingerprint
            ),
            "macro_materialization_rng_execution_fingerprint": (
                macro_source.rng_execution_fingerprint
            ),
            "macro_fit_rng_execution_fingerprints": macro_rng,
            "monthly_block_materialization_fingerprint": (
                monthly_source.materialization_fingerprint
            ),
            "monthly_block_rng_execution_fingerprint": (
                monthly_source.rng_execution_fingerprint
            ),
            "daily_block_materialization_fingerprint": (
                daily_source.materialization_fingerprint
            ),
            "daily_block_rng_execution_fingerprint": (
                daily_source.rng_execution_fingerprint
            ),
            "block_core_fingerprints": block_cores,
            "block_failure_fingerprints": block_failures,
            "combined_fit_rng_execution_fingerprint": combined_fit_rng,
            "combined_materialization_fingerprint": combined_materialization,
            "combined_rng_execution_fingerprint": combined_rng,
            "strict_canonical_geometry": strict,
        },
        arrays={"candidate_order": np.asarray(VERSION_1_CANDIDATES)},
    )
    return RegisteredDesignCandidateSuiteIdentity(
        suite_fingerprint=suite_fingerprint,
        iteration=value.iteration,
        design_slice_fingerprint=slice_fingerprint,
        design_feature_fingerprint=feature_fingerprint,
        preliminary_fingerprint=preliminary_fingerprint,
        preliminary_outcome_fingerprint=(preliminary_identity.outcome_fingerprint),
        fit_set_fingerprint=fit_set_fingerprint,
        source_content_fingerprint=source_content_fingerprint,
        main_fit_rng_execution_fingerprint=fit.rng_execution_fingerprint,
        rolling_fit_rng_execution_fingerprints=tuple(rolling_rng),
        combined_fit_rng_execution_fingerprint=combined_fit_rng,
        macro_materialization_fingerprint=macro_source.materialization_fingerprint,
        macro_materialization_rng_execution_fingerprint=(
            macro_source.rng_execution_fingerprint
        ),
        macro_fit_rng_execution_fingerprints=tuple(macro_rng),
        monthly_block_materialization_fingerprint=(
            monthly_source.materialization_fingerprint
        ),
        monthly_block_rng_execution_fingerprint=(
            monthly_source.rng_execution_fingerprint
        ),
        daily_block_materialization_fingerprint=(
            daily_source.materialization_fingerprint
        ),
        daily_block_rng_execution_fingerprint=daily_source.rng_execution_fingerprint,
        block_core_fingerprints=tuple(block_cores),
        block_failure_fingerprints=tuple(block_failures),
        combined_materialization_fingerprint=combined_materialization,
        combined_rng_execution_fingerprint=combined_rng,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        strict_canonical_geometry=strict,
        monthly_input_start=monthly_start,
        daily_input_start=daily_start,
    )


def _verify_successful_candidate_parents(
    *,
    n_states: int,
    fit: GaussianHMMFit,
    design_slice: CalibrationSlice,
    pools: DecodedReturnPools,
    monthly_input_start: int,
    daily_input_start: int,
    macro_materialization: RegisteredMacroBootstrapMaterialization,
    monthly_materialization: RegisteredBlockUniformMaterialization,
    daily_materialization: RegisteredBlockUniformMaterialization,
    rolling: RollingCandidateEvidence,
    macro: MacroStabilityEvidence,
    monthly_block: RegisteredBlockOutcome,
    daily_block: RegisteredBlockOutcome,
    master_seed: int,
    scientific_version: int,
) -> None:
    fit_fingerprint = gaussian_hmm_fit_fingerprint(fit)
    rolling_identity = registered_rolling_candidate_identity(rolling)
    macro_identity = registered_macro_stability_execution_identity(macro)
    if (
        rolling.n_states != n_states
        or rolling._parent_model is not fit
        or rolling_identity.role != "design"
        or rolling_identity.parent_model_fingerprint != fit_fingerprint
        or rolling_identity.master_seed != master_seed
        or rolling_identity.scientific_version != scientific_version
        or macro.n_states != n_states
        or macro._parent_model is not fit
        or macro._source_materialization is not macro_materialization
        or macro_identity.role != "design"
        or macro_identity.parent_model_fingerprint != fit_fingerprint
        or macro_identity.master_seed != master_seed
        or macro_identity.scientific_version != scientific_version
        or macro_identity.materialization_fingerprint
        != macro_materialization.materialization_fingerprint
    ):
        raise ModelError("candidate suite rolling/macro parent ancestry differs")
    _verify_candidate_block_outcome(
        outcome=monthly_block,
        frequency="monthly",
        fit=fit,
        n_states=n_states,
        input_start=monthly_input_start,
        materialization=monthly_materialization,
        source_states=pools.monthly_states,
        source_links=np.asarray(design_slice.monthly_continuation[1:]),
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    _verify_candidate_block_outcome(
        outcome=daily_block,
        frequency="daily",
        fit=fit,
        n_states=n_states,
        input_start=daily_input_start,
        materialization=daily_materialization,
        source_states=pools.daily_states,
        source_links=np.asarray(design_slice.daily_continuation[1:]),
        master_seed=master_seed,
        scientific_version=scientific_version,
    )


def _verify_candidate_block_outcome(
    *,
    outcome: RegisteredBlockOutcome,
    frequency: str,
    fit: GaussianHMMFit,
    n_states: int,
    input_start: int,
    materialization: RegisteredBlockUniformMaterialization,
    source_states: np.ndarray,
    source_links: np.ndarray,
    master_seed: int,
    scientific_version: int,
) -> None:
    if isinstance(outcome, RegisteredBlockCalibration):
        source = registered_block_replay_source(outcome)
        artifact = source.core_identity.artifact
        identity_frequency = source.core_identity.frequency
        selected_k = source.core_identity.selected_k
        starting_length = source.core_identity.starting_length
        identity_seed = source.core_identity.master_seed
        identity_version = source.core_identity.scientific_version
        exact = (
            source.materialization is materialization
            and source.model is fit
            and np.array_equal(source.historical_source_states, source_states)
            and np.array_equal(source.source_continuation_links, source_links)
        )
    else:
        failure = registered_block_support_failure_identity(outcome)
        artifact = failure.artifact
        identity_frequency = failure.frequency
        selected_k = failure.selected_k
        starting_length = failure.starting_length
        identity_seed = failure.master_seed
        identity_version = failure.scientific_version
        exact = (
            outcome._source_materialization is materialization
            and outcome._source_model is fit
            and np.array_equal(outcome._source_states, source_states)
            and np.array_equal(outcome._source_continuation_links, source_links)
        )
    if (
        not exact
        or artifact != "design"
        or identity_frequency != frequency
        or selected_k != n_states
        or starting_length != input_start
        or identity_seed != master_seed
        or identity_version != scientific_version
    ):
        raise ModelError("candidate suite block parent ancestry differs")


def _verify_design_features(
    design_slice: CalibrationSlice,
    features: HMMFeatureMatrix,
) -> None:
    if not isinstance(features, HMMFeatureMatrix):
        raise ModelError("candidate suite requires HMMFeatureMatrix")
    source = np.asarray(design_slice.macro_features, dtype=np.float64)
    means = source.mean(axis=0)
    scales = source.std(axis=0, ddof=0)
    expected = (source - means) / scales
    expected_labels = tuple(
        str(value)
        for value in pd.PeriodIndex.from_ordinals(
            design_slice.monthly_period_ordinals,
            freq="M",
        )
    )
    if (
        features.scaler_scope != "design_training"
        or features.feature_names
        != (
            "industrial_production_growth",
            "inflation_yoy",
            "yield_curve_spread",
            "credit_spread",
        )
        or features.lengths
        != sequence_lengths_from_continuation(design_slice.monthly_continuation)
        or features.row_labels != expected_labels
        or not np.array_equal(np.asarray(features.values), expected)
        or not np.array_equal(np.asarray(features.scaler_means), means)
        or not np.array_equal(
            np.asarray(features.scaler_standard_deviations),
            scales,
        )
    ):
        raise ModelError("candidate suite design feature materialization differs")


def _row_block_outcome(
    row: RegisteredDesignCandidateSuiteRow,
    *,
    frequency: str,
) -> RegisteredBlockOutcome:
    if frequency == "monthly":
        success = row.monthly_block
        failure = row.monthly_block_failure
    elif frequency == "daily":
        success = row.daily_block
        failure = row.daily_block_failure
    else:  # pragma: no cover - closed internal callers
        raise AssertionError("unsupported candidate block frequency")
    if (success is None) is (failure is None):
        raise ModelError(
            f"successful candidate requires exactly one {frequency} block outcome"
        )
    return (
        success if success is not None else cast(RegisteredBlockSupportFailure, failure)
    )


def _row_block_parent_matches(
    row: RegisteredDesignCandidateSuiteRow,
    parent: RegisteredBlockOutcome | None,
    *,
    frequency: str,
) -> bool:
    if frequency == "monthly":
        success = row.monthly_block
        failure = row.monthly_block_failure
    else:
        success = row.daily_block
        failure = row.daily_block_failure
    if parent is None:
        return success is None and failure is None
    if isinstance(parent, RegisteredBlockCalibration):
        return success is parent and failure is None
    return failure is parent and success is None


def _ordered_parent_tuple(
    values: Sequence[object | None],
    label: str,
) -> tuple[object | None, object | None, object | None, object | None]:
    if len(values) != len(VERSION_1_CANDIDATES):
        raise ModelError(f"candidate suite {label} parents require K=2,3,4,5")
    return cast(
        tuple[object | None, object | None, object | None, object | None],
        tuple(values),
    )


def _preliminary_fingerprint(value: PreliminaryCalibrations) -> str:
    if not isinstance(value, PreliminaryCalibrations) or value.role != "design":
        raise ModelError("candidate suite preliminary calibration is invalid")
    return preliminary_calibrations_fingerprint(value)


def _composite_fingerprint(label: str, values: object) -> str:
    return scientific_materialization_fingerprint(
        schema_id=label,
        metadata={"values": values},
        arrays={"candidate_order": np.asarray(VERSION_1_CANDIDATES)},
    )


def _iteration_input_length(
    value: int | None,
    *,
    preliminary: int,
    frequency: str,
) -> int:
    selected = preliminary if value is None else value
    bounds = (
        MONTHLY_BLOCK_LENGTH_BOUNDS
        if frequency == "monthly"
        else DAILY_BLOCK_LENGTH_BOUNDS
    )
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or not bounds[0] <= selected <= bounds[1]
        or selected > preliminary
    ):
        raise ModelError(f"candidate iteration {frequency} input start is invalid")
    return selected


def _validate_seed_version(master_seed: int, scientific_version: int) -> None:
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or master_seed < 0
    ):
        raise ModelError("candidate suite master seed is invalid")
    if (
        isinstance(scientific_version, bool)
        or not isinstance(scientific_version, int)
        or scientific_version <= 0
    ):
        raise ModelError("candidate suite scientific version is invalid")


def _fixed_point_iteration(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ModelError("candidate suite fixed-point iteration must be in [1, 5]")
    return value
