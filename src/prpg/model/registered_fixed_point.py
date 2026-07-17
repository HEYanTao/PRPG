"""Producer-sealed design candidate/block fixed-point authority.

``candidate_suite`` owns one complete candidate-support iteration.  This
module owns the stronger cross-iteration statement required by G3: iteration
one starts at the exact design PPW lengths, every later iteration starts at
the immediately preceding selected block lengths, and the complete state

``(fixed K=4, monthly L, daily L)``

is equal in two adjacent iterations no later than iteration five.  A
non-adjacent repeat is an oscillation, a five-look drift is fatal, and records
after the first adjacent equality are forbidden.

Authority is object-specific.  The process-local ledger retains every exact
suite, viability result, selection result, and selected monthly/daily block
parent.  Public verification re-derives every upstream identity and every
fingerprint.  Dataclass construction, copying, an equivalent substitute, or
writing the private staging flags in :mod:`prpg.model.candidate_gate` cannot
mint final authority.

Task 2 receives only :class:`RegisteredFinalViabilityProjectionIdentity`.
That projection contains no selected K, selection result, selection
fingerprint, selected-block fingerprint, or full-history fingerprint.  Task 3
may consume :class:`RegisteredFinalSelectionProjectionIdentity`, which binds
the complete converged history and exact final selected-block parents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias, TypeGuard

import numpy as np

from prpg.errors import ModelError
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    registered_block_core_identity,
)
from prpg.model.candidate_gate import (
    CANONICAL_CANDIDATES,
    CandidateGateResult,
    CandidateViabilityResult,
    verified_candidate_gate_execution_identity,
    verified_candidate_viability_execution_identity,
)
from prpg.model.candidate_suite import (
    RegisteredDesignCandidateSuite,
    registered_design_candidate_suite_identity,
)
from prpg.model.fixed_point import MAXIMUM_FIXED_POINT_ITERATIONS, FixedPointState
from prpg.model.materialization_identity import scientific_materialization_fingerprint
from prpg.model.selection import OWNER_FIXED_K4_SELECTION_POLICY, OWNER_FIXED_N_STATES

REGISTERED_DESIGN_FIXED_POINT_CONTRACT: Final = "registered-design-fixed-point-v2"
_FIXED_POINT_CAPABILITY = object()

IterationSource: TypeAlias = tuple[
    RegisteredDesignCandidateSuite,
    CandidateViabilityResult,
    CandidateGateResult,
]
FixedPointOutcome: TypeAlias = Literal[
    "converged",
    "no_admissible_candidates",
    "candidate_selection_failed",
    "oscillation",
    "maximum_iterations_exceeded",
]


@dataclass(frozen=True, slots=True, init=False)
class RegisteredDesignFixedPointIteration:
    """One exact retained suite/viability/selection reduction."""

    iteration: int
    input_state: FixedPointState | None
    state: FixedPointState | None
    suite_fingerprint: str
    viability_result_fingerprint: str
    selection_fingerprint: str
    monthly_block_core_fingerprint: str | None
    daily_block_core_fingerprint: str | None
    iteration_fingerprint: str
    _suite: RegisteredDesignCandidateSuite = field(repr=False, compare=False)
    _viability: CandidateViabilityResult = field(repr=False, compare=False)
    _selection: CandidateGateResult = field(repr=False, compare=False)
    _monthly_block: RegisteredBlockCalibration | None = field(
        repr=False,
        compare=False,
    )
    _daily_block: RegisteredBlockCalibration | None = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredDesignFixedPointIteration is created only by "
            "assemble_registered_design_fixed_point"
        )


@dataclass(frozen=True, slots=True, init=False)
class RegisteredDesignFixedPoint:
    """Converged final authority and its exact full iteration history."""

    contract_version: str
    master_seed: int
    scientific_version: int
    outcome: FixedPointOutcome
    passed: bool
    failure_reason: str | None
    state: FixedPointState | None
    iterations: tuple[RegisteredDesignFixedPointIteration, ...]
    converged_at_iteration: int
    maximum_iterations: int
    viability_projection_fingerprint: str
    iteration_history_fingerprint: str
    fixed_point_fingerprint: str
    generation_authorized: bool
    _final_viability: CandidateViabilityResult = field(repr=False, compare=False)
    _final_selection: CandidateGateResult = field(repr=False, compare=False)
    _final_monthly_block: RegisteredBlockCalibration | None = field(
        repr=False,
        compare=False,
    )
    _final_daily_block: RegisteredBlockCalibration | None = field(
        repr=False,
        compare=False,
    )
    _execution_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "RegisteredDesignFixedPoint is created only by "
            "assemble_registered_design_fixed_point"
        )


@dataclass(frozen=True, slots=True)
class RegisteredFinalViabilityProjectionIdentity:
    """Task-2-safe proof of the exact final viability projection.

    Deliberately absent: selected K, the selection fingerprint/result, selected
    block identities, and the complete fixed-point history fingerprint.
    """

    contract_version: str
    master_seed: int
    scientific_version: int
    final_iteration: int
    final_suite_fingerprint: str
    final_viability_result_fingerprint: str
    viability_projection_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredFinalSelectionProjectionIdentity:
    """Task-3 proof binding the full history and final selected parents."""

    contract_version: str
    master_seed: int
    scientific_version: int
    outcome: FixedPointOutcome
    passed: bool
    failure_reason: str | None
    state: FixedPointState | None
    converged_at_iteration: int
    iteration_fingerprints: tuple[str, ...]
    iteration_history_fingerprint: str
    fixed_point_fingerprint: str
    final_suite_fingerprint: str
    final_viability_result_fingerprint: str
    final_selection_fingerprint: str
    final_monthly_block_core_fingerprint: str | None
    final_daily_block_core_fingerprint: str | None
    viability_projection_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredFinalBlockProjectionIdentity:
    """Task-8 proof for the exact final monthly/daily block objects."""

    master_seed: int
    scientific_version: int
    selected_n_states: int
    fixed_point_fingerprint: str
    iteration_history_fingerprint: str
    final_selection_fingerprint: str
    monthly_block_core_fingerprint: str
    daily_block_core_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegisteredDesignFixedPointIdentity:
    """Fully rederived producer identity, including both task projections."""

    master_seed: int
    scientific_version: int
    outcome: FixedPointOutcome
    passed: bool
    failure_reason: str | None
    state: FixedPointState | None
    converged_at_iteration: int
    iteration_fingerprints: tuple[str, ...]
    viability_projection: RegisteredFinalViabilityProjectionIdentity
    final_selection_projection: RegisteredFinalSelectionProjectionIdentity


@dataclass(frozen=True, slots=True)
class _FixedPointMintRecord:
    value: RegisteredDesignFixedPoint
    iterations: object
    iteration_objects: tuple[RegisteredDesignFixedPointIteration, ...]
    sources: tuple[IterationSource, ...]
    final_viability: CandidateViabilityResult
    final_selection: CandidateGateResult
    final_monthly_block: RegisteredBlockCalibration | None
    final_daily_block: RegisteredBlockCalibration | None
    viability_projection_fingerprint: str
    iteration_history_fingerprint: str
    fixed_point_fingerprint: str


@dataclass(frozen=True, slots=True)
class _DerivedIteration:
    source: IterationSource
    state: FixedPointState | None
    suite_fingerprint: str
    viability_result_fingerprint: str
    selection_fingerprint: str
    monthly_block: RegisteredBlockCalibration | None
    daily_block: RegisteredBlockCalibration | None
    monthly_block_core_fingerprint: str | None
    daily_block_core_fingerprint: str | None
    iteration_fingerprint: str


@dataclass(frozen=True, slots=True)
class _DerivedHistory:
    iterations: tuple[_DerivedIteration, ...]
    outcome: FixedPointOutcome
    failure_reason: str | None


_MINTED_FIXED_POINTS: dict[int, _FixedPointMintRecord] = {}
_FINAL_VIABILITY_LEDGER: dict[int, RegisteredDesignFixedPoint] = {}
_FINAL_SELECTION_LEDGER: dict[int, RegisteredDesignFixedPoint] = {}
_FINAL_BLOCK_PAIR_LEDGER: dict[tuple[int, int], RegisteredDesignFixedPoint] = {}


def assemble_registered_design_fixed_point(
    iterations: Sequence[IterationSource],
) -> RegisteredDesignFixedPoint:
    """Validate and mint one exact converged design fixed point.

    ``iterations`` must contain the complete ordered records from iteration 1
    through the first adjacent equality.  The assembler derives iteration
    numbers and selected block parents; callers cannot supply either.
    """

    sources = _normalize_sources(iterations)
    history = _derive_source_history(sources)
    derived = history.iterations
    records = tuple(
        _mint_iteration_record(
            index,
            item,
            derived[index - 2].state if index > 1 else None,
        )
        for index, item in enumerate(derived, start=1)
    )
    final = derived[-1]
    viability_projection_fingerprint = _viability_projection_fingerprint(derived)
    history_fingerprint = _history_fingerprint(derived)
    fixed_point_fingerprint = _fixed_point_fingerprint(
        master_seed=derived[0].source[0].master_seed,
        scientific_version=derived[0].source[0].scientific_version,
        outcome=history.outcome,
        failure_reason=history.failure_reason,
        state=final.state,
        converged_at_iteration=len(derived),
        viability_projection_fingerprint=viability_projection_fingerprint,
        iteration_history_fingerprint=history_fingerprint,
    )
    value = object.__new__(RegisteredDesignFixedPoint)
    for name, item in (
        ("contract_version", REGISTERED_DESIGN_FIXED_POINT_CONTRACT),
        ("master_seed", derived[0].source[0].master_seed),
        ("scientific_version", derived[0].source[0].scientific_version),
        ("outcome", history.outcome),
        ("passed", history.outcome == "converged"),
        ("failure_reason", history.failure_reason),
        ("state", final.state),
        ("iterations", records),
        ("converged_at_iteration", len(records)),
        ("maximum_iterations", MAXIMUM_FIXED_POINT_ITERATIONS),
        ("viability_projection_fingerprint", viability_projection_fingerprint),
        ("iteration_history_fingerprint", history_fingerprint),
        ("fixed_point_fingerprint", fixed_point_fingerprint),
        ("generation_authorized", False),
        ("_final_viability", final.source[1]),
        ("_final_selection", final.source[2]),
        ("_final_monthly_block", final.monthly_block),
        ("_final_daily_block", final.daily_block),
        ("_execution_seal", _FIXED_POINT_CAPABILITY),
    ):
        object.__setattr__(value, name, item)

    final_block_key = (
        None
        if final.monthly_block is None or final.daily_block is None
        else (id(final.monthly_block), id(final.daily_block))
    )
    if (
        id(final.source[1]) in _FINAL_VIABILITY_LEDGER
        or id(final.source[2]) in _FINAL_SELECTION_LEDGER
        or (final_block_key is not None and final_block_key in _FINAL_BLOCK_PAIR_LEDGER)
    ):
        raise ModelError(
            "final candidate/block result already belongs to a fixed point"
        )
    record = _FixedPointMintRecord(
        value=value,
        iterations=value.iterations,
        iteration_objects=records,
        sources=sources,
        final_viability=final.source[1],
        final_selection=final.source[2],
        final_monthly_block=final.monthly_block,
        final_daily_block=final.daily_block,
        viability_projection_fingerprint=viability_projection_fingerprint,
        iteration_history_fingerprint=history_fingerprint,
        fixed_point_fingerprint=fixed_point_fingerprint,
    )
    _MINTED_FIXED_POINTS[id(value)] = record
    _FINAL_VIABILITY_LEDGER[id(final.source[1])] = value
    _FINAL_SELECTION_LEDGER[id(final.source[2])] = value
    if final_block_key is not None:
        _FINAL_BLOCK_PAIR_LEDGER[final_block_key] = value
    object.__setattr__(final.source[1], "_final_fixed_point_authority", True)
    object.__setattr__(
        final.source[1], "_final_fixed_point_seal", _FIXED_POINT_CAPABILITY
    )
    object.__setattr__(final.source[2], "_final_fixed_point_authority", True)
    object.__setattr__(
        final.source[2], "_final_fixed_point_seal", _FIXED_POINT_CAPABILITY
    )
    try:
        registered_design_fixed_point_identity(value)
    except Exception:
        _MINTED_FIXED_POINTS.pop(id(value), None)
        _FINAL_VIABILITY_LEDGER.pop(id(final.source[1]), None)
        _FINAL_SELECTION_LEDGER.pop(id(final.source[2]), None)
        if final_block_key is not None:
            _FINAL_BLOCK_PAIR_LEDGER.pop(final_block_key, None)
        object.__setattr__(final.source[1], "_final_fixed_point_authority", False)
        object.__setattr__(final.source[1], "_final_fixed_point_seal", None)
        object.__setattr__(final.source[2], "_final_fixed_point_authority", False)
        object.__setattr__(final.source[2], "_final_fixed_point_seal", None)
        raise
    return value


def registered_design_fixed_point_identity(
    value: RegisteredDesignFixedPoint,
) -> RegisteredDesignFixedPointIdentity:
    """Revalidate the external object ledger and complete live parent graph."""

    if not isinstance(value, RegisteredDesignFixedPoint):
        raise ModelError("design fixed point has the wrong type")
    ledger = _MINTED_FIXED_POINTS.get(id(value))
    if (
        value._execution_seal is not _FIXED_POINT_CAPABILITY
        or ledger is None
        or ledger.value is not value
        or value.iterations is not ledger.iterations
        or len(value.iterations) != len(ledger.iteration_objects)
        or any(
            supplied is not retained
            for supplied, retained in zip(
                value.iterations,
                ledger.iteration_objects,
                strict=True,
            )
        )
        or value._final_viability is not ledger.final_viability
        or value._final_selection is not ledger.final_selection
        or value._final_monthly_block is not ledger.final_monthly_block
        or value._final_daily_block is not ledger.final_daily_block
    ):
        raise ModelError("design fixed point lacks exact producer authority")
    if (
        value.contract_version != REGISTERED_DESIGN_FIXED_POINT_CONTRACT
        or value.maximum_iterations != MAXIMUM_FIXED_POINT_ITERATIONS
        or value.generation_authorized is not False
        or _FINAL_VIABILITY_LEDGER.get(id(ledger.final_viability)) is not value
        or _FINAL_SELECTION_LEDGER.get(id(ledger.final_selection)) is not value
        or (
            ledger.final_monthly_block is not None
            and ledger.final_daily_block is not None
            and _FINAL_BLOCK_PAIR_LEDGER.get(
                (id(ledger.final_monthly_block), id(ledger.final_daily_block))
            )
            is not value
        )
        or ledger.final_viability._final_fixed_point_authority is not True
        or ledger.final_viability._final_fixed_point_seal is not _FIXED_POINT_CAPABILITY
        or ledger.final_selection._final_fixed_point_authority is not True
        or ledger.final_selection._final_fixed_point_seal is not _FIXED_POINT_CAPABILITY
    ):
        raise ModelError("design fixed-point authority or contract changed")

    history = _derive_source_history(ledger.sources)
    derived = history.iterations
    if len(derived) != len(value.iterations):
        raise ModelError("design fixed-point history length changed")
    for index, (stored, current, source) in enumerate(
        zip(value.iterations, derived, ledger.sources, strict=True),
        start=1,
    ):
        expected_input = derived[index - 2].state if index > 1 else None
        if (
            stored._execution_seal is not _FIXED_POINT_CAPABILITY
            or stored._suite is not source[0]
            or stored._viability is not source[1]
            or stored._selection is not source[2]
            or stored._monthly_block is not current.monthly_block
            or stored._daily_block is not current.daily_block
            or stored.iteration != index
            or stored.input_state != expected_input
            or stored.state != current.state
            or stored.suite_fingerprint != current.suite_fingerprint
            or stored.viability_result_fingerprint
            != current.viability_result_fingerprint
            or stored.selection_fingerprint != current.selection_fingerprint
            or stored.monthly_block_core_fingerprint
            != current.monthly_block_core_fingerprint
            or stored.daily_block_core_fingerprint
            != current.daily_block_core_fingerprint
            or stored.iteration_fingerprint != current.iteration_fingerprint
        ):
            raise ModelError("design fixed-point iteration changed or was substituted")

    final = derived[-1]
    viability_projection_fingerprint = _viability_projection_fingerprint(derived)
    history_fingerprint = _history_fingerprint(derived)
    fixed_point_fingerprint = _fixed_point_fingerprint(
        master_seed=derived[0].source[0].master_seed,
        scientific_version=derived[0].source[0].scientific_version,
        outcome=history.outcome,
        failure_reason=history.failure_reason,
        state=final.state,
        converged_at_iteration=len(derived),
        viability_projection_fingerprint=viability_projection_fingerprint,
        iteration_history_fingerprint=history_fingerprint,
    )
    if (
        value.master_seed != derived[0].source[0].master_seed
        or value.scientific_version != derived[0].source[0].scientific_version
        or value.outcome != history.outcome
        or value.passed is not (history.outcome == "converged")
        or value.failure_reason != history.failure_reason
        or value.state != final.state
        or value.converged_at_iteration != len(derived)
        or value.viability_projection_fingerprint != viability_projection_fingerprint
        or value.iteration_history_fingerprint != history_fingerprint
        or value.fixed_point_fingerprint != fixed_point_fingerprint
        or ledger.viability_projection_fingerprint != viability_projection_fingerprint
        or ledger.iteration_history_fingerprint != history_fingerprint
        or ledger.fixed_point_fingerprint != fixed_point_fingerprint
        or value._final_viability is not final.source[1]
        or value._final_selection is not final.source[2]
        or value._final_monthly_block is not final.monthly_block
        or value._final_daily_block is not final.daily_block
    ):
        raise ModelError("design fixed-point sources or fingerprints changed")

    viability_projection = RegisteredFinalViabilityProjectionIdentity(
        contract_version=value.contract_version,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        final_iteration=len(derived),
        final_suite_fingerprint=final.suite_fingerprint,
        final_viability_result_fingerprint=final.viability_result_fingerprint,
        viability_projection_fingerprint=viability_projection_fingerprint,
    )
    selection_projection = RegisteredFinalSelectionProjectionIdentity(
        contract_version=value.contract_version,
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        outcome=history.outcome,
        passed=history.outcome == "converged",
        failure_reason=history.failure_reason,
        state=final.state,
        converged_at_iteration=len(derived),
        iteration_fingerprints=tuple(item.iteration_fingerprint for item in derived),
        iteration_history_fingerprint=history_fingerprint,
        fixed_point_fingerprint=fixed_point_fingerprint,
        final_suite_fingerprint=final.suite_fingerprint,
        final_viability_result_fingerprint=final.viability_result_fingerprint,
        final_selection_fingerprint=final.selection_fingerprint,
        final_monthly_block_core_fingerprint=(final.monthly_block_core_fingerprint),
        final_daily_block_core_fingerprint=final.daily_block_core_fingerprint,
        viability_projection_fingerprint=viability_projection_fingerprint,
    )
    return RegisteredDesignFixedPointIdentity(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        outcome=history.outcome,
        passed=history.outcome == "converged",
        failure_reason=history.failure_reason,
        state=final.state,
        converged_at_iteration=len(derived),
        iteration_fingerprints=selection_projection.iteration_fingerprints,
        viability_projection=viability_projection,
        final_selection_projection=selection_projection,
    )


def is_registered_design_fixed_point(
    value: object,
) -> TypeGuard[RegisteredDesignFixedPoint]:
    """Return whether ``value`` is the exact unchanged minted fixed point."""

    if not isinstance(value, RegisteredDesignFixedPoint):
        return False
    try:
        registered_design_fixed_point_identity(value)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def has_registered_final_candidate_authority(
    value: CandidateViabilityResult | CandidateGateResult,
) -> bool:
    """Ledger-backed final authority check used by candidate task guards."""

    if isinstance(value, CandidateViabilityResult):
        fixed_point = _FINAL_VIABILITY_LEDGER.get(id(value))
        exact = fixed_point is not None and fixed_point._final_viability is value
    else:
        fixed_point = _FINAL_SELECTION_LEDGER.get(id(value))
        exact = fixed_point is not None and fixed_point._final_selection is value
    if not exact or fixed_point is None:
        return False
    try:
        registered_design_fixed_point_identity(fixed_point)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_final_viability_projection_identity(
    value: CandidateViabilityResult,
) -> RegisteredFinalViabilityProjectionIdentity:
    """Return the task-2-safe projection for the exact final viability object."""

    fixed_point = _FINAL_VIABILITY_LEDGER.get(id(value))
    if fixed_point is None or fixed_point._final_viability is not value:
        raise ModelError("candidate viability lacks final fixed-point authority")
    return registered_design_fixed_point_identity(fixed_point).viability_projection


def registered_final_selection_projection_identity(
    value: CandidateGateResult,
) -> RegisteredFinalSelectionProjectionIdentity:
    """Return full task-3 history for the exact final selection object."""

    fixed_point = _FINAL_SELECTION_LEDGER.get(id(value))
    if fixed_point is None or fixed_point._final_selection is not value:
        raise ModelError("candidate selection lacks final fixed-point authority")
    return registered_design_fixed_point_identity(
        fixed_point
    ).final_selection_projection


def registered_design_fixed_point_sources(
    value: RegisteredDesignFixedPoint,
) -> tuple[
    tuple[IterationSource, ...],
    RegisteredBlockCalibration | None,
    RegisteredBlockCalibration | None,
]:
    """Return the full exact source history and final selected block parents."""

    registered_design_fixed_point_identity(value)
    ledger = _MINTED_FIXED_POINTS[id(value)]
    return (
        ledger.sources,
        ledger.final_monthly_block,
        ledger.final_daily_block,
    )


def registered_final_selected_blocks(
    value: CandidateGateResult,
) -> tuple[RegisteredBlockCalibration, RegisteredBlockCalibration]:
    """Return task-8's exact final monthly/daily parents after reverification."""

    fixed_point = _FINAL_SELECTION_LEDGER.get(id(value))
    if fixed_point is None or fixed_point._final_selection is not value:
        raise ModelError("candidate selection lacks final fixed-point authority")
    identity = registered_design_fixed_point_identity(fixed_point)
    monthly = fixed_point._final_monthly_block
    daily = fixed_point._final_daily_block
    if not identity.passed or monthly is None or daily is None:
        raise ModelError("failed design fixed point has no selected block parents")
    return monthly, daily


def registered_final_block_projection_identity(
    monthly: RegisteredBlockCalibration,
    daily: RegisteredBlockCalibration,
) -> RegisteredFinalBlockProjectionIdentity:
    """Return task 8's proof for one exact converged final block pair.

    Equivalent/copy block objects and blocks retained by a scientifically
    terminal non-converged history are deliberately outside this authority.
    """

    fixed_point = _FINAL_BLOCK_PAIR_LEDGER.get((id(monthly), id(daily)))
    if (
        fixed_point is None
        or fixed_point._final_monthly_block is not monthly
        or fixed_point._final_daily_block is not daily
    ):
        raise ModelError("design blocks lack exact final fixed-point authority")
    identity = registered_design_fixed_point_identity(fixed_point)
    projection = identity.final_selection_projection
    state = identity.state
    if (
        not identity.passed
        or state is None
        or projection.final_monthly_block_core_fingerprint is None
        or projection.final_daily_block_core_fingerprint is None
    ):
        raise ModelError("non-converged fixed point cannot authorize task 8")
    return RegisteredFinalBlockProjectionIdentity(
        master_seed=identity.master_seed,
        scientific_version=identity.scientific_version,
        selected_n_states=state.selected_n_states,
        fixed_point_fingerprint=projection.fixed_point_fingerprint,
        iteration_history_fingerprint=projection.iteration_history_fingerprint,
        final_selection_fingerprint=projection.final_selection_fingerprint,
        monthly_block_core_fingerprint=(
            projection.final_monthly_block_core_fingerprint
        ),
        daily_block_core_fingerprint=projection.final_daily_block_core_fingerprint,
    )


def _normalize_sources(
    iterations: Sequence[IterationSource],
) -> tuple[IterationSource, ...]:
    supplied = tuple(iterations)
    if not 1 <= len(supplied) <= MAXIMUM_FIXED_POINT_ITERATIONS:
        raise ModelError("design fixed point requires one through five iterations")
    normalized: list[IterationSource] = []
    suite_ids: set[int] = set()
    viability_ids: set[int] = set()
    selection_ids: set[int] = set()
    for item in supplied:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ModelError(
                "each fixed-point iteration requires suite, viability, and selection"
            )
        suite, viability, selection = item
        if (
            not isinstance(suite, RegisteredDesignCandidateSuite)
            or not isinstance(viability, CandidateViabilityResult)
            or not isinstance(selection, CandidateGateResult)
        ):
            raise ModelError("fixed-point iteration source types are invalid")
        if (
            id(suite) in suite_ids
            or id(viability) in viability_ids
            or id(selection) in selection_ids
        ):
            raise ModelError("fixed-point iteration records must be object-unique")
        suite_ids.add(id(suite))
        viability_ids.add(id(viability))
        selection_ids.add(id(selection))
        normalized.append((suite, viability, selection))
    return tuple(normalized)


def _derive_source_history(
    sources: tuple[IterationSource, ...],
) -> _DerivedHistory:
    derived: list[_DerivedIteration] = []
    seen: dict[FixedPointState, int] = {}
    outcome: FixedPointOutcome | None = None
    failure_reason: str | None = None
    first_suite = sources[0][0]
    common_selection_materialization = sources[0][
        2
    ]._registered_bootstrap_materialization
    for index, source in enumerate(sources, start=1):
        if outcome is not None:
            raise ModelError("fixed-point history contains records after termination")
        suite, viability, selection = source
        suite_identity = registered_design_candidate_suite_identity(suite)
        viability_identity = verified_candidate_viability_execution_identity(viability)
        selection_identity = verified_candidate_gate_execution_identity(selection)
        if (
            suite.iteration != index
            or suite_identity.iteration != index
            or not viability._canonical_suite_execution
            or viability._source_suite is not suite
            or selection.viability_result is not viability
            or viability_identity.role != "design"
            or viability_identity.suite_fingerprint != suite_identity.suite_fingerprint
            or selection_identity.role != "design"
            or selection_identity.viability_result_fingerprint
            != viability_identity.viability_result_fingerprint
            or suite.master_seed != first_suite.master_seed
            or suite.scientific_version != first_suite.scientific_version
        ):
            raise ModelError("fixed-point viability/selection ancestry differs")
        _verify_common_iteration_sources(first_suite, suite)
        if (
            selection._registered_bootstrap_materialization
            is not common_selection_materialization
        ):
            raise ModelError(
                "fixed-point iterations require one exact selection materialization"
            )
        if index == 1:
            if (
                suite.monthly_input_start
                != suite.preliminary.monthly.combined.starting_length
                or suite.daily_input_start
                != suite.preliminary.daily.combined.starting_length
            ):
                raise ModelError(
                    "first fixed-point iteration must use exact PPW starts"
                )
        else:
            previous = derived[-1].state
            if previous is None:
                raise ModelError(
                    "fixed-point history continued after terminal selection"
                )
            if (
                suite.monthly_input_start != previous.monthly_length
                or suite.daily_input_start != previous.daily_length
            ):
                raise ModelError(
                    "fixed-point iteration did not start at prior finalized lengths"
                )
        admissible = tuple(
            row.n_states for row in viability.audit_rows if row.viability.admissible
        )
        if selection.selection.selection_policy != OWNER_FIXED_K4_SELECTION_POLICY:
            raise ModelError("fixed-point selection must use owner_fixed_k4_v3")
        selected = selection.selection.selected_n_states
        if selected is not None and selected != OWNER_FIXED_N_STATES:
            raise ModelError("owner-fixed selection cannot select a non-K4 candidate")
        if tuple(sorted(set(admissible))) != admissible or any(
            item not in CANONICAL_CANDIDATES for item in admissible
        ):
            raise ModelError("fixed-point admissible K tuple is invalid")

        state: FixedPointState | None = None
        monthly: RegisteredBlockCalibration | None = None
        daily: RegisteredBlockCalibration | None = None
        monthly_core_fingerprint: str | None = None
        daily_core_fingerprint: str | None = None
        if not admissible:
            if selection.selection.passed or selected is not None:
                raise ModelError("empty viability set has a contradictory selection")
            outcome = "no_admissible_candidates"
            failure_reason = (
                selection.selection.failure_reason or "no_admissible_candidates"
            )
        elif not selection.selection.passed:
            if selected is not None:
                raise ModelError("failed candidate selection retained a selected K")
            outcome = "candidate_selection_failed"
            failure_reason = (
                selection.selection.failure_reason or "candidate_selection_failed"
            )
        else:
            if selected is None or selected not in admissible:
                raise ModelError("passed selection lacks an admissible selected K")
            row = next((item for item in suite.rows if item.n_states == selected), None)
            if row is None or row.monthly_block is None or row.daily_block is None:
                raise ModelError("fixed-point selected K lacks exact block parents")
            monthly = row.monthly_block
            daily = row.daily_block
            monthly_core = registered_block_core_identity(monthly)
            daily_core = registered_block_core_identity(daily)
            if (
                monthly_core.artifact != "design"
                or daily_core.artifact != "design"
                or monthly_core.frequency != "monthly"
                or daily_core.frequency != "daily"
                or monthly_core.selected_k != selected
                or daily_core.selected_k != selected
                or monthly_core.starting_length != suite.monthly_input_start
                or daily_core.starting_length != suite.daily_input_start
                or monthly_core.master_seed != suite.master_seed
                or daily_core.master_seed != suite.master_seed
                or monthly_core.scientific_version != suite.scientific_version
                or daily_core.scientific_version != suite.scientific_version
            ):
                raise ModelError("fixed-point selected block ancestry differs")
            state = FixedPointState(
                selected,
                monthly_core.selected_length,
                daily_core.selected_length,
            )
            monthly_core_fingerprint = monthly_core.core_fingerprint
            daily_core_fingerprint = daily_core.core_fingerprint
            if index > 1:
                previous = derived[-1].state
                assert previous is not None
                if state == previous:
                    outcome = "converged"
                elif state in seen:
                    outcome = "oscillation"
                    failure_reason = "fixed_point_oscillation"
            if outcome is None:
                seen.setdefault(state, index)

        if outcome is not None and index != len(sources):
            raise ModelError(
                "fixed-point history contains records after scientific termination"
            )
        iteration_fingerprint = _iteration_fingerprint(
            iteration=index,
            suite_fingerprint=suite_identity.suite_fingerprint,
            viability_result_fingerprint=viability_identity.viability_result_fingerprint,
            selection_fingerprint=selection_identity.selection_fingerprint,
            state=state,
            monthly_block_core_fingerprint=monthly_core_fingerprint,
            daily_block_core_fingerprint=daily_core_fingerprint,
        )
        derived.append(
            _DerivedIteration(
                source=source,
                state=state,
                suite_fingerprint=suite_identity.suite_fingerprint,
                viability_result_fingerprint=(
                    viability_identity.viability_result_fingerprint
                ),
                selection_fingerprint=selection_identity.selection_fingerprint,
                monthly_block=monthly,
                daily_block=daily,
                monthly_block_core_fingerprint=monthly_core_fingerprint,
                daily_block_core_fingerprint=daily_core_fingerprint,
                iteration_fingerprint=iteration_fingerprint,
            )
        )
    if outcome is None:
        if len(derived) == MAXIMUM_FIXED_POINT_ITERATIONS:
            outcome = "maximum_iterations_exceeded"
            failure_reason = "fixed_point_maximum_iterations"
        else:
            raise ModelError("design fixed-point history is favorably truncated")
    if outcome == "converged":
        failure_reason = None
    return _DerivedHistory(tuple(derived), outcome, failure_reason)


def _verify_common_iteration_sources(
    first: RegisteredDesignCandidateSuite,
    current: RegisteredDesignCandidateSuite,
) -> None:
    if (
        current._design_slice is not first._design_slice
        or current._design_features is not first._design_features
        or current._fit_set is not first._fit_set
        or current._preliminary_outcome is not first._preliminary_outcome
        or current._macro_materialization is not first._macro_materialization
        or current._monthly_materialization is not first._monthly_materialization
        or current._daily_materialization is not first._daily_materialization
    ):
        raise ModelError("fixed-point iterations substituted a common parent")
    for first_row, current_row in zip(first.rows, current.rows, strict=True):
        if (
            current_row.n_states != first_row.n_states
            or current_row.rolling is not first_row.rolling
            or current_row.macro_stability is not first_row.macro_stability
        ):
            raise ModelError("fixed-point iterations changed rolling/macro evidence")


def _mint_iteration_record(
    iteration: int,
    derived: _DerivedIteration,
    input_state: FixedPointState | None,
) -> RegisteredDesignFixedPointIteration:
    value = object.__new__(RegisteredDesignFixedPointIteration)
    suite, viability, selection = derived.source
    for name, item in (
        ("iteration", iteration),
        ("input_state", input_state),
        ("state", derived.state),
        ("suite_fingerprint", derived.suite_fingerprint),
        ("viability_result_fingerprint", derived.viability_result_fingerprint),
        ("selection_fingerprint", derived.selection_fingerprint),
        (
            "monthly_block_core_fingerprint",
            derived.monthly_block_core_fingerprint,
        ),
        ("daily_block_core_fingerprint", derived.daily_block_core_fingerprint),
        ("iteration_fingerprint", derived.iteration_fingerprint),
        ("_suite", suite),
        ("_viability", viability),
        ("_selection", selection),
        ("_monthly_block", derived.monthly_block),
        ("_daily_block", derived.daily_block),
        ("_execution_seal", _FIXED_POINT_CAPABILITY),
    ):
        object.__setattr__(value, name, item)
    return value


def _iteration_fingerprint(
    *,
    iteration: int,
    suite_fingerprint: str,
    viability_result_fingerprint: str,
    selection_fingerprint: str,
    state: FixedPointState | None,
    monthly_block_core_fingerprint: str | None,
    daily_block_core_fingerprint: str | None,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_design_fixed_point_iteration_v2",
        metadata={
            "contract_version": REGISTERED_DESIGN_FIXED_POINT_CONTRACT,
            "iteration": iteration,
            "suite_fingerprint": suite_fingerprint,
            "viability_result_fingerprint": viability_result_fingerprint,
            "selection_fingerprint": selection_fingerprint,
            "state": None if state is None else _state_dict(state),
            "monthly_block_core_fingerprint": monthly_block_core_fingerprint,
            "daily_block_core_fingerprint": daily_block_core_fingerprint,
        },
        arrays={"candidate_order": np.asarray(CANONICAL_CANDIDATES)},
    )


def _viability_projection_fingerprint(
    derived: Sequence[_DerivedIteration],
) -> str:
    # This intentionally exposes only the final viability projection.  It
    # excludes prior-iteration identities as well as every state, selected K,
    # selection fingerprint, selected-block fingerprint, and full-history
    # fingerprint.  Task 3 alone owns the complete history.
    final = derived[-1]
    return scientific_materialization_fingerprint(
        schema_id="registered_design_fixed_point_viability_projection_v2",
        metadata={
            "contract_version": REGISTERED_DESIGN_FIXED_POINT_CONTRACT,
            "final_iteration": len(derived),
            "final_suite_fingerprint": final.suite_fingerprint,
            "final_viability_result_fingerprint": (final.viability_result_fingerprint),
        },
        arrays={"final_iteration": np.asarray([len(derived)], dtype=np.int64)},
    )


def _history_fingerprint(derived: Sequence[_DerivedIteration]) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_design_fixed_point_history_v2",
        metadata={
            "contract_version": REGISTERED_DESIGN_FIXED_POINT_CONTRACT,
            "iterations": [item.iteration_fingerprint for item in derived],
            "states": [
                None if item.state is None else _state_dict(item.state)
                for item in derived
            ],
        },
        arrays={"iteration_order": np.arange(1, len(derived) + 1, dtype=np.int64)},
    )


def _fixed_point_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    outcome: FixedPointOutcome,
    failure_reason: str | None,
    state: FixedPointState | None,
    converged_at_iteration: int,
    viability_projection_fingerprint: str,
    iteration_history_fingerprint: str,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="registered_design_fixed_point_v2",
        metadata={
            "contract_version": REGISTERED_DESIGN_FIXED_POINT_CONTRACT,
            "master_seed": master_seed,
            "scientific_version": scientific_version,
            "outcome": outcome,
            "failure_reason": failure_reason,
            "state": None if state is None else _state_dict(state),
            "converged_at_iteration": converged_at_iteration,
            "maximum_iterations": MAXIMUM_FIXED_POINT_ITERATIONS,
            "viability_projection_fingerprint": viability_projection_fingerprint,
            "iteration_history_fingerprint": iteration_history_fingerprint,
        },
        arrays={"candidate_order": np.asarray(CANONICAL_CANDIDATES)},
    )


def _state_dict(value: FixedPointState) -> dict[str, object]:
    return {
        "selected_n_states": value.selected_n_states,
        "monthly_length": value.monthly_length,
        "daily_length": value.daily_length,
    }
