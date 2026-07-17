"""Closed execution contract for the Phase-3 dependence/fragmentation gate.

The lower-level block and dependence modules intentionally own no orchestration.
This module closes that gap without leaking future evidence backward through
the G3 DAG: each artifact executes its monthly and daily cells in isolation,
then exact run-bound dependence views feed the single authoritative four-cell
Holm task.  Every cell reuses one immutable purpose-4 materialization for block
calibration and selected/neighbor replay.  Neighbor and leave-one-calendar-
year-out (LOYO) results remain report-only and never enter the decision.

The returned evidence contains only scalar summaries, short tuples, and frozen
Holm arrays.  In particular, it does not retain source-index histories, return
tensors, uniform matrices, or the 10,000-by-endpoint dependence matrices.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias, TypeGuard, cast

import numpy as np
import numpy.typing as npt

from prpg.errors import ModelError
from prpg.model.block_execution import (
    CANONICAL_HISTORY_COUNT,
    DEFAULT_TRACE_CHUNK_SIZE,
    Artifact,
    BlockStateModel,
    CalibrationGeometry,
    PooledFragmentationGate,
)
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    RegisteredBlockCoreIdentity,
    RegisteredBlockFragmentationIdentity,
    RegisteredBlockReplaySource,
    RegisteredBlockUniformMaterialization,
    iter_materialized_block_trace_chunks,
    registered_block_core_identity,
    registered_block_fragmentation_identity,
    registered_block_replay_source,
    registered_block_uniform_bytes,
    registered_block_uniform_identity,
    run_materialized_block_calibration,
)
from prpg.model.block_length import (
    BLOCK_POLICY_VERSION,
    Frequency,
    estimate_frequency_start,
)
from prpg.model.dependence import (
    DependenceGate,
    feasible_sensitivity_lengths,
    fit_dependence_reference,
)
from prpg.model.dependence_execution import DependenceTraceReducer
from prpg.model.hmm import GaussianHMMFit
from prpg.model.inference import holm_correction
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_array_fingerprint,
)

DependenceCellId: TypeAlias = Literal[
    "design_monthly",
    "design_daily",
    "production_monthly",
    "production_daily",
]
DecisionRole: TypeAlias = Literal["selected", "report_only_neighbor"]
DependenceTaskId: TypeAlias = Literal[
    "design_fragmentation",
    "design_dependence",
    "production_fragmentation",
    "production_dependence",
]
MaterializationProducer: TypeAlias = Callable[
    ["DependenceGateCellInput"], RegisteredBlockUniformMaterialization
]

DEPENDENCE_CELL_ORDER: tuple[DependenceCellId, ...] = (
    "design_monthly",
    "design_daily",
    "production_monthly",
    "production_daily",
)
DEPENDENCE_GATE_CONTRACT = "sections-4.5-4.6-dependence-v1"
ARTIFACT_DEPENDENCE_EXECUTION_CONTRACT = (
    "sections-4.5-4.6-artifact-dependence-execution-v1"
)
DEPENDENCE_TASK_VIEW_CONTRACT = "sections-4.5-4.6-dependence-task-view-v1"
STAGED_DEPENDENCE_EXECUTION_CONTRACT = "sections-4.5-4.6-staged-dependence-execution-v2"
STAGED_DEPENDENCE_TASK_VIEW_CONTRACT = "sections-4.5-4.6-staged-dependence-task-view-v2"
REPORT_ONLY_DEPENDENCE_TASK_VIEW_CONTRACT = (
    "sections-4.5-4.6-all-at-once-report-only-v1"
)
FOUR_CELL_DEPENDENCE_HOLM_VIEW_CONTRACT = (
    "sections-4.5-4.6-four-cell-dependence-holm-view-v1"
)
_REGISTERED_DEPENDENCE_CAPABILITY = object()
_REGISTERED_DEPENDENCE_CELL_CAPABILITY = object()
_REGISTERED_ARTIFACT_DEPENDENCE_CAPABILITY = object()
_REGISTERED_DEPENDENCE_TASK_VIEW_CAPABILITY = object()
_REGISTERED_DEPENDENCE_HOLM_VIEW_CAPABILITY = object()
_REGISTERED_STAGED_DEPENDENCE_CELL_CAPABILITY = object()
_REGISTERED_STAGED_DEPENDENCE_EXECUTION_CAPABILITY = object()
_REPORT_ONLY_DEPENDENCE_TASK_VIEW_CAPABILITY = object()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# A shallow copy can preserve an ``init=False`` capability field.  These
# process-local ledgers make each authorization object-specific.  Strong
# references are bounded by the handful of G3 evidence objects in one run and
# prevent Python from reusing an authorized object ID after garbage collection.
_MINTED_DEPENDENCE_CELLS: dict[int, object] = {}
_MINTED_ARTIFACT_DEPENDENCE_EXECUTIONS: dict[int, object] = {}
_MINTED_DEPENDENCE_TASK_VIEWS: dict[int, object] = {}
_MINTED_DEPENDENCE_HOLM_VIEWS: dict[int, object] = {}
_MINTED_FOUR_CELL_DEPENDENCE_GATES: dict[int, object] = {}
_MINTED_STAGED_DEPENDENCE_CELLS: dict[int, object] = {}
_MINTED_STAGED_DEPENDENCE_EXECUTIONS: dict[int, object] = {}
_MINTED_REPORT_ONLY_DEPENDENCE_TASK_VIEWS: dict[int, object] = {}

_CELL_ROLE: dict[DependenceCellId, tuple[Artifact, Frequency]] = {
    "design_monthly": ("design", "monthly"),
    "design_daily": ("design", "daily"),
    "production_monthly": ("production", "monthly"),
    "production_daily": ("production", "daily"),
}

_ARTIFACT_CELL_ORDER: dict[Artifact, tuple[DependenceCellId, DependenceCellId]] = {
    "design": ("design_monthly", "design_daily"),
    "production": ("production_monthly", "production_daily"),
}

_TASK_ROLE_AND_KIND: dict[
    DependenceTaskId, tuple[Artifact, Literal["fragmentation", "dependence"]]
] = {
    "design_fragmentation": ("design", "fragmentation"),
    "design_dependence": ("design", "dependence"),
    "production_fragmentation": ("production", "fragmentation"),
    "production_dependence": ("production", "dependence"),
}


@dataclass(frozen=True, slots=True)
class DependenceGateCellInput:
    """Observed inputs needed to execute one artifact/frequency cell.

    ``source_continuation_links`` uses the forward-link convention and must
    therefore have one fewer element than the return/state histories.
    ``calendar_years`` is the already-audited calendar-year label aligned to
    each return row; it is used only for deterministic LOYO selector evidence.
    """

    cell_id: DependenceCellId
    artifact: Artifact
    frequency: Frequency
    model: BlockStateModel | GaussianHMMFit
    historical_returns: npt.ArrayLike
    historical_source_states: npt.ArrayLike
    source_continuation_links: npt.ArrayLike
    calendar_years: npt.ArrayLike


@dataclass(frozen=True, slots=True)
class LOYOSelectorRecord:
    """One deterministic selector rerun with a calendar year removed."""

    omitted_year: int
    omitted_observations: int
    remaining_observations: int
    starting_length: int | None
    raw_ceiling_length: int | None
    controlling_series: tuple[str, ...]
    failure_type: str | None
    failure_message: str | None

    @property
    def succeeded(self) -> bool:
        return self.starting_length is not None


@dataclass(frozen=True, slots=True)
class LOYOSelectorRobustness:
    """Full-sample selector plus every deterministic LOYO rerun."""

    frequency: Frequency
    policy_version: str
    observations: int
    full_sample_starting_length: int
    full_sample_raw_ceiling_length: int
    full_sample_controlling_series: tuple[str, ...]
    records: tuple[LOYOSelectorRecord, ...]
    fingerprint: str
    report_only: bool = True


@dataclass(frozen=True, slots=True)
class FragmentationStateEvidence:
    """Compact selected-L conditioned/ideal evidence for one HMM state."""

    state: int
    intended_length: int
    conditioned_count: int
    conditioned_mean: float
    conditioned_median: float
    conditioned_singleton_fraction: float
    ideal_count: int
    ideal_median: float
    ideal_singleton_fraction: float
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class DependenceLengthEvidence:
    """Compact max-z result for selected L or one report-only neighbor."""

    intended_length: int
    decision_role: DecisionRole
    observed_statistic: float
    critical_value_95: float
    p_value: float
    active_components: int
    repetitions: int

    @property
    def envelope_passed(self) -> bool:
        return self.observed_statistic <= self.critical_value_95


@dataclass(frozen=True, slots=True)
class DependenceCellEvidence:
    """Legacy report-only combined fragmentation/dependence cell."""

    cell_id: DependenceCellId
    artifact: Artifact
    frequency: Frequency
    histories: int
    monthly_segment_lengths: tuple[int, ...]
    target_segment_lengths: tuple[int, ...]
    starting_length: int
    selected_length: int
    requested_lengths: tuple[int, ...]
    endpoint_count: int
    endpoint_labels_fingerprint: str
    stream_namespace: str
    common_draw_reused: bool
    fragmentation: tuple[FragmentationStateEvidence, ...]
    length_results: tuple[DependenceLengthEvidence, ...]
    loyo: LOYOSelectorRobustness
    input_fingerprint: str
    model_fingerprint: str
    materialization_fingerprint: str
    source_fingerprint: str
    evidence_fingerprint: str
    strict_canonical: bool
    report_only: bool = True
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def selected_result(self) -> DependenceLengthEvidence:
        matches = tuple(
            result
            for result in self.length_results
            if result.intended_length == self.selected_length
        )
        if len(matches) != 1:
            raise ModelError("dependence cell has no unique selected-L result")
        return matches[0]

    @property
    def selected_envelope_passed(self) -> bool:
        return self.selected_result.envelope_passed

    @property
    def fragmentation_passed(self) -> bool:
        return bool(self.fragmentation) and all(
            item.passed for item in self.fragmentation
        )


@dataclass(frozen=True, slots=True)
class ArtifactDependenceExecution:
    """Legacy report-only monthly-plus-daily combined role execution.

    A design instance cannot retain or reveal production inputs: its exact
    cell order is ``design_monthly, design_daily`` and its content fingerprint
    is computed solely from those two registered cells.  Production uses the
    symmetric isolated contract.
    """

    contract_version: str
    role: Artifact
    cell_order: tuple[DependenceCellId, DependenceCellId]
    cells: tuple[DependenceCellEvidence, DependenceCellEvidence]
    source_execution_fingerprint: str
    strict_canonical: bool
    neighbors_report_only: bool = True
    loyo_report_only: bool = True
    report_only: bool = True
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RegisteredDependenceCellExecution:
    """Dependence-only replay under one exact registered block parent."""

    cell_id: DependenceCellId
    artifact: Artifact
    frequency: Frequency
    histories: int
    starting_length: int
    selected_length: int
    requested_lengths: tuple[int, ...]
    endpoint_count: int
    endpoint_labels_fingerprint: str
    common_draw_reused: bool
    length_results: tuple[DependenceLengthEvidence, ...]
    loyo: LOYOSelectorRobustness
    input_fingerprint: str
    model_fingerprint: str
    parent_block_core_fingerprint: str
    purpose4_materialization_fingerprint: str
    purpose4_stream_fingerprint: str
    rng_execution_fingerprint: str
    dependence_result_fingerprint: str
    source_fingerprint: str
    evidence_fingerprint: str
    strict_canonical: bool
    neighbors_report_only: bool = True
    loyo_report_only: bool = True
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def selected_result(self) -> DependenceLengthEvidence:
        matches = tuple(
            item
            for item in self.length_results
            if item.intended_length == self.selected_length
        )
        if len(matches) != 1:
            raise ModelError("staged dependence cell has no unique selected result")
        return matches[0]


@dataclass(frozen=True, slots=True)
class RegisteredArtifactDependenceExecution:
    """Role-isolated dependence replay after an exact fragmentation view."""

    contract_version: str
    role: Artifact
    cell_order: tuple[DependenceCellId, DependenceCellId]
    cells: tuple[RegisteredDependenceCellExecution, RegisteredDependenceCellExecution]
    parent_block_core_fingerprints: tuple[str, str]
    fragmentation_parent_view_fingerprint: str
    source_execution_fingerprint: str
    strict_canonical: bool
    neighbors_report_only: bool = True
    loyo_report_only: bool = True
    _block_parents: (
        tuple[RegisteredBlockCalibration, RegisteredBlockCalibration] | None
    ) = field(default=None, init=False, repr=False, compare=False)
    _fragmentation_parent: DependenceTaskView | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class FragmentationTaskCellEvidence:
    """Task-scoped cell evidence that exposes no dependence outcome."""

    cell_id: DependenceCellId
    artifact: Artifact
    frequency: Frequency
    histories: int
    selected_length: int
    common_draw_reused: bool
    fragmentation: tuple[FragmentationStateEvidence, ...]
    input_fingerprint: str
    model_fingerprint: str
    materialization_fingerprint: str
    source_fingerprint: str
    source_evidence_fingerprint: str
    selected_k: int
    starting_length: int
    parent_block_core_fingerprint: str
    fragmentation_evidence_fingerprint: str

    @property
    def fragmentation_passed(self) -> bool:
        return bool(self.fragmentation) and all(
            item.passed for item in self.fragmentation
        )


@dataclass(frozen=True, slots=True)
class DependenceTaskCellEvidence:
    """Task-scoped selected-L evidence that exposes no fragmentation detail."""

    cell_id: DependenceCellId
    artifact: Artifact
    frequency: Frequency
    histories: int
    selected_length: int
    endpoint_count: int
    common_draw_reused: bool
    selected_result: DependenceLengthEvidence
    loyo_fingerprint: str
    input_fingerprint: str
    model_fingerprint: str
    materialization_fingerprint: str
    source_fingerprint: str
    source_evidence_fingerprint: str
    selected_k: int
    starting_length: int
    parent_block_core_fingerprint: str
    purpose4_materialization_fingerprint: str
    purpose4_stream_fingerprint: str
    rng_execution_fingerprint: str
    dependence_result_fingerprint: str


@dataclass(frozen=True, slots=True)
class DependenceTaskView:
    """Run-bound view of one role execution for one exact G3 task."""

    contract_version: str
    task_id: DependenceTaskId
    role: Artifact
    task_kind: Literal["fragmentation", "dependence"]
    run_id: str
    binding_fingerprint: str
    source_execution_fingerprint: str
    cells: (
        tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence]
        | tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence]
    )
    passed: bool
    failure_reasons: tuple[str, ...]
    view_fingerprint: str
    strict_canonical: bool
    parent_block_core_fingerprints: tuple[str, str]
    fragmentation_parent_view_fingerprint: str | None
    canonical_stage: bool
    report_only: bool
    neighbors_report_only: bool = True
    loyo_report_only: bool = True
    _source_block_parents: (
        tuple[RegisteredBlockCalibration, RegisteredBlockCalibration] | None
    ) = field(default=None, init=False, repr=False, compare=False)
    _source_fragmentation_view: DependenceTaskView | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_staged_execution: RegisteredArtifactDependenceExecution | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _view_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class FourCellDependenceHolmView:
    """Run-bound authoritative Holm view over two exact dependence views."""

    contract_version: str
    task_id: Literal["four_cell_dependence_holm"]
    run_id: str
    binding_fingerprint: str
    source_view_fingerprints: tuple[str, str]
    source_views: tuple[DependenceTaskView, DependenceTaskView]
    cell_order: tuple[DependenceCellId, ...]
    holm: DependenceGate
    passed: bool
    failure_reasons: tuple[str, ...]
    view_fingerprint: str
    strict_canonical: bool
    neighbors_report_only: bool = True
    loyo_report_only: bool = True
    _view_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def cells(
        self,
    ) -> tuple[
        DependenceTaskCellEvidence,
        DependenceTaskCellEvidence,
        DependenceTaskCellEvidence,
        DependenceTaskCellEvidence,
    ]:
        """Return the exact fixed-order selected-L task evidence."""

        design, production = self.source_views
        design_cells = _dependence_task_cells(design)
        production_cells = _dependence_task_cells(production)
        return (*design_cells, *production_cells)


@dataclass(frozen=True, slots=True)
class FourCellDependenceGateExecution:
    """Legacy report-only all-at-once four-cell decision."""

    contract_version: str
    cell_order: tuple[DependenceCellId, ...]
    cells: tuple[
        DependenceCellEvidence,
        DependenceCellEvidence,
        DependenceCellEvidence,
        DependenceCellEvidence,
    ]
    holm: DependenceGate
    passed: bool
    failure_reasons: tuple[str, ...]
    strict_canonical: bool
    neighbors_report_only: bool = True
    loyo_report_only: bool = True
    report_only: bool = True
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


def is_registered_four_cell_dependence_execution(
    value: object,
) -> TypeGuard[FourCellDependenceGateExecution]:
    """Reject the legacy all-at-once gate at the canonical boundary."""

    return False


def is_report_only_four_cell_dependence_execution(
    value: object,
) -> TypeGuard[FourCellDependenceGateExecution]:
    """Return whether a legacy all-at-once gate retains report authority."""

    if not isinstance(value, FourCellDependenceGateExecution):
        return False
    try:
        _verify_four_cell_gate(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return value._execution_seal is _REGISTERED_DEPENDENCE_CAPABILITY


def is_registered_dependence_cell_evidence(
    value: object,
) -> TypeGuard[DependenceCellEvidence]:
    """Reject a legacy cell containing two scientific stages."""

    return False


def is_report_only_dependence_cell_evidence(
    value: object,
) -> TypeGuard[DependenceCellEvidence]:
    """Return whether a legacy combined cell retains report authority."""

    if not isinstance(value, DependenceCellEvidence):
        return False
    try:
        _verify_registered_cell(value, strict_canonical=value.strict_canonical)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


def is_registered_artifact_dependence_execution(
    value: object,
) -> TypeGuard[ArtifactDependenceExecution]:
    """Reject a legacy role execution containing future task evidence."""

    return False


def is_report_only_artifact_dependence_execution(
    value: object,
) -> TypeGuard[ArtifactDependenceExecution]:
    """Return whether a legacy role execution retains report authority."""

    if not isinstance(value, ArtifactDependenceExecution):
        return False
    try:
        _verify_artifact_execution(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


def is_registered_dependence_task_view(
    value: object,
) -> TypeGuard[DependenceTaskView]:
    """Return whether a task view follows the staged canonical topology."""

    if not isinstance(value, DependenceTaskView):
        return False
    try:
        _verify_task_view(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


def is_report_only_dependence_task_view(
    value: object,
) -> TypeGuard[DependenceTaskView]:
    """Return whether ``value`` is the sealed legacy all-at-once projection."""

    if not isinstance(value, DependenceTaskView):
        return False
    try:
        _verify_report_only_task_view(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


def is_registered_staged_dependence_cell(
    value: object,
) -> TypeGuard[RegisteredDependenceCellExecution]:
    """Return whether a dependence-only cell retains producer authority."""

    if not isinstance(value, RegisteredDependenceCellExecution):
        return False
    try:
        _verify_staged_dependence_cell(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


def is_registered_staged_dependence_execution(
    value: object,
) -> TypeGuard[RegisteredArtifactDependenceExecution]:
    """Return whether a role replay has exact block/fragmentation parents."""

    if not isinstance(value, RegisteredArtifactDependenceExecution):
        return False
    try:
        _verify_staged_dependence_execution(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


def is_registered_four_cell_dependence_holm_view(
    value: object,
) -> TypeGuard[FourCellDependenceHolmView]:
    """Return whether the Holm view was recomputed from exact sealed views."""

    if not isinstance(value, FourCellDependenceHolmView):
        return False
    try:
        _verify_holm_view(value)
    except (ModelError, TypeError, ValueError, AttributeError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class _PreparedCell:
    values: npt.NDArray[np.float64]
    states: npt.NDArray[np.int64]
    links: npt.NDArray[np.bool_]
    years: npt.NDArray[np.int64]


def deterministic_loyo_selector_robustness(
    returns: npt.ArrayLike,
    continuation_links: npt.ArrayLike,
    calendar_years: npt.ArrayLike,
    *,
    frequency: Frequency,
) -> LOYOSelectorRobustness:
    """Run the full selector and every calendar-year deletion without RNG.

    Removing a year never splices the rows on its two sides together.  A link
    in a reduced sample is true only when the retained rows were adjacent in
    the original history and their original forward link was true.  A
    scientifically invalid reduced selector is recorded, not raised, because
    the preregistration makes LOYO report-only.  Invalid full-sample inputs
    still fail closed.
    """

    values, links, years = _selector_inputs(
        returns, continuation_links, calendar_years, frequency=frequency
    )
    full = estimate_frequency_start(values, frequency=frequency, continuation=links)
    records: list[LOYOSelectorRecord] = []
    for omitted_year in tuple(int(year) for year in np.unique(years)):
        keep = years != omitted_year
        retained = np.flatnonzero(keep)
        reduced_links = _retained_links(retained, links)
        try:
            calibration = estimate_frequency_start(
                values[keep],
                frequency=frequency,
                continuation=reduced_links,
            )
            combined = calibration.combined
            record = LOYOSelectorRecord(
                omitted_year=omitted_year,
                omitted_observations=int(np.count_nonzero(~keep)),
                remaining_observations=int(np.count_nonzero(keep)),
                starting_length=combined.starting_length,
                raw_ceiling_length=combined.raw_ceiling_length,
                controlling_series=combined.controlling_series,
                failure_type=None,
                failure_message=None,
            )
        except ModelError as error:
            record = LOYOSelectorRecord(
                omitted_year=omitted_year,
                omitted_observations=int(np.count_nonzero(~keep)),
                remaining_observations=int(np.count_nonzero(keep)),
                starting_length=None,
                raw_ceiling_length=None,
                controlling_series=(),
                failure_type=type(error).__name__,
                failure_message=_bounded(error.message),
            )
        records.append(record)
    combined = full.combined
    result = LOYOSelectorRobustness(
        frequency=frequency,
        policy_version=BLOCK_POLICY_VERSION,
        observations=len(values),
        full_sample_starting_length=combined.starting_length,
        full_sample_raw_ceiling_length=combined.raw_ceiling_length,
        full_sample_controlling_series=combined.controlling_series,
        records=tuple(records),
        fingerprint="",
    )
    return _with_loyo_fingerprint(result)


def build_fragmentation_task_view(
    block_parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
    *,
    task_id: Literal["design_fragmentation", "production_fragmentation"],
    run_id: str,
    binding_fingerprint: str,
) -> DependenceTaskView:
    """Build fragmentation solely from two registered block parents.

    No dependence execution, dependence result, dependence RNG, calendar-year
    evidence, return matrix, or production-role evidence is accepted by this
    boundary.  The two exact code-minted parents are retained privately so the
    view can reverify them on every guard call.
    """

    expected = _TASK_ROLE_AND_KIND.get(task_id)
    if expected is None or expected[1] != "fragmentation":
        raise ModelError("fragmentation view requires a registered task ID")
    role = expected[0]
    cores, fragmentations = _validated_block_parent_pair(block_parents, role=role)
    _require_sha256(run_id, "fragmentation task run ID")
    _require_sha256(binding_fingerprint, "fragmentation task binding fingerprint")
    cells = _fragmentation_cells_from_block_parents(
        block_parents, cores=cores, fragmentations=fragmentations
    )
    reasons = _task_view_failure_reasons(cells, kind="fragmentation")
    source_fingerprint = _fragmentation_source_fingerprint(
        role=role,
        cores=cores,
        fragmentations=fragmentations,
    )
    core_fingerprints = (cores[0].core_fingerprint, cores[1].core_fingerprint)
    provisional = DependenceTaskView(
        contract_version=STAGED_DEPENDENCE_TASK_VIEW_CONTRACT,
        task_id=task_id,
        role=role,
        task_kind="fragmentation",
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        source_execution_fingerprint=source_fingerprint,
        cells=cells,
        passed=not reasons,
        failure_reasons=reasons,
        view_fingerprint="",
        strict_canonical=block_parents[0].execution.geometry.strict_canonical,
        parent_block_core_fingerprints=core_fingerprints,
        fragmentation_parent_view_fingerprint=None,
        canonical_stage=True,
        report_only=False,
    )
    result = replace(provisional, view_fingerprint=_task_view_fingerprint(provisional))
    object.__setattr__(result, "_source_block_parents", block_parents)
    object.__setattr__(
        result, "_view_seal", _REGISTERED_DEPENDENCE_TASK_VIEW_CAPABILITY
    )
    _MINTED_DEPENDENCE_TASK_VIEWS[id(result)] = result
    return result


def execute_registered_artifact_dependence_role(
    cells: tuple[DependenceGateCellInput, DependenceGateCellInput],
    *,
    role: Artifact,
    block_parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
    fragmentation_parent: DependenceTaskView,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    strict_canonical: bool = True,
) -> RegisteredArtifactDependenceExecution:
    """Replay dependence after fragmentation using the exact block draw sets.

    This canonical path has no materialization producer argument: each cell
    must reuse the immutable materialization already retained by its registered
    block parent.  Consequently a later stage cannot silently draw or rebind a
    second purpose-4 stream.
    """

    cores, _ = _validate_staged_role_inputs(
        cells,
        role=role,
        block_parents=block_parents,
        fragmentation_parent=fragmentation_parent,
        chunk_size=chunk_size,
        strict_canonical=strict_canonical,
    )
    completed = tuple(
        _execute_staged_dependence_cell(
            cell_input,
            parent,
            chunk_size=chunk_size,
            strict_canonical=strict_canonical,
        )
        for cell_input, parent in zip(cells, block_parents, strict=True)
    )
    checked = cast(
        tuple[RegisteredDependenceCellExecution, RegisteredDependenceCellExecution],
        completed,
    )
    core_fingerprints = (cores[0].core_fingerprint, cores[1].core_fingerprint)
    source_fingerprint = _staged_artifact_execution_fingerprint(
        role=role,
        cells=checked,
        parent_core_fingerprints=core_fingerprints,
        fragmentation_parent_view_fingerprint=(fragmentation_parent.view_fingerprint),
        strict_canonical=strict_canonical,
    )
    result = RegisteredArtifactDependenceExecution(
        contract_version=STAGED_DEPENDENCE_EXECUTION_CONTRACT,
        role=role,
        cell_order=_ARTIFACT_CELL_ORDER[role],
        cells=checked,
        parent_block_core_fingerprints=core_fingerprints,
        fragmentation_parent_view_fingerprint=(fragmentation_parent.view_fingerprint),
        source_execution_fingerprint=source_fingerprint,
        strict_canonical=strict_canonical,
    )
    object.__setattr__(result, "_block_parents", block_parents)
    object.__setattr__(result, "_fragmentation_parent", fragmentation_parent)
    object.__setattr__(
        result,
        "_execution_seal",
        _REGISTERED_STAGED_DEPENDENCE_EXECUTION_CAPABILITY,
    )
    _MINTED_STAGED_DEPENDENCE_EXECUTIONS[id(result)] = result
    return result


def build_registered_dependence_task_view(
    source: RegisteredArtifactDependenceExecution,
    *,
    task_id: Literal["design_dependence", "production_dependence"],
    run_id: str,
    binding_fingerprint: str,
) -> DependenceTaskView:
    """Build the later dependence view from one staged role execution."""

    _verify_staged_dependence_execution(source)
    expected = _TASK_ROLE_AND_KIND.get(task_id)
    if expected is None or expected != (source.role, "dependence"):
        raise ModelError("dependence task ID does not match staged execution role")
    _require_sha256(run_id, "dependence task run ID")
    _require_sha256(binding_fingerprint, "dependence task binding fingerprint")
    fragmentation_parent = source._fragmentation_parent
    block_parents = source._block_parents
    if fragmentation_parent is None or block_parents is None:
        raise ModelError("staged dependence source parents are unavailable")
    if fragmentation_parent.run_id != run_id:
        raise ModelError("dependence task and fragmentation parent use different runs")
    if fragmentation_parent.binding_fingerprint == binding_fingerprint:
        raise ModelError("dependence task binding must differ from fragmentation")
    cells = _dependence_cells_from_staged_execution(source.cells, block_parents)
    reasons = _task_view_failure_reasons(cells, kind="dependence")
    provisional = DependenceTaskView(
        contract_version=STAGED_DEPENDENCE_TASK_VIEW_CONTRACT,
        task_id=task_id,
        role=source.role,
        task_kind="dependence",
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        source_execution_fingerprint=source.source_execution_fingerprint,
        cells=cells,
        passed=not reasons,
        failure_reasons=reasons,
        view_fingerprint="",
        strict_canonical=source.strict_canonical,
        parent_block_core_fingerprints=source.parent_block_core_fingerprints,
        fragmentation_parent_view_fingerprint=(fragmentation_parent.view_fingerprint),
        canonical_stage=True,
        report_only=False,
    )
    result = replace(provisional, view_fingerprint=_task_view_fingerprint(provisional))
    object.__setattr__(result, "_source_block_parents", block_parents)
    object.__setattr__(result, "_source_fragmentation_view", fragmentation_parent)
    object.__setattr__(result, "_source_staged_execution", source)
    object.__setattr__(
        result, "_view_seal", _REGISTERED_DEPENDENCE_TASK_VIEW_CAPABILITY
    )
    _MINTED_DEPENDENCE_TASK_VIEWS[id(result)] = result
    return result


def execute_artifact_dependence_role(
    cells: tuple[DependenceGateCellInput, DependenceGateCellInput],
    *,
    role: Artifact,
    materializations: Mapping[DependenceCellId, RegisteredBlockUniformMaterialization]
    | None = None,
    materialization_producer: MaterializationProducer | None = None,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    strict_canonical: bool = True,
) -> ArtifactDependenceExecution:
    """Execute one role without accepting or inspecting the other role.

    The two inputs are exactly monthly then daily for ``role``.  This is the
    topological primitive used to finish design evidence before production
    fitting or production inputs exist.
    """

    checked_order = _validate_role_execution_options(
        cells,
        role=role,
        materializations=materializations,
        materialization_producer=materialization_producer,
        chunk_size=chunk_size,
        strict_canonical=strict_canonical,
    )
    if materializations is not None:
        _validate_no_cross_cell_aliasing(materializations, checked_order)

    completed: list[DependenceCellEvidence] = []
    for cell_input in cells:
        materialization = (
            materializations[cell_input.cell_id]
            if materializations is not None
            else _produce_materialization(materialization_producer, cell_input)
        )
        completed.append(
            _execute_dependence_cell(
                cell_input,
                materialization,
                chunk_size=chunk_size,
                strict_canonical=strict_canonical,
            )
        )
        del materialization
    pair = cast(tuple[DependenceCellEvidence, DependenceCellEvidence], tuple(completed))
    return assemble_artifact_dependence_execution(
        pair,
        role=role,
        strict_canonical=strict_canonical,
    )


def assemble_artifact_dependence_execution(
    cells: tuple[DependenceCellEvidence, DependenceCellEvidence],
    *,
    role: Artifact,
    strict_canonical: bool,
) -> ArtifactDependenceExecution:
    """Assemble an isolated role execution from two registered cells only."""

    if role not in _ARTIFACT_CELL_ORDER:
        raise ModelError("artifact dependence role must be design or production")
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if not isinstance(cells, tuple) or len(cells) != 2:
        raise ModelError("artifact dependence execution requires exactly two cells")
    expected_order = _ARTIFACT_CELL_ORDER[role]
    actual_order = tuple(cell.cell_id for cell in cells)
    if actual_order != expected_order:
        raise ModelError(
            "artifact dependence cells have the wrong role-scoped order",
            details={"expected": list(expected_order), "actual": list(actual_order)},
        )
    for cell in cells:
        _verify_registered_cell(cell, strict_canonical=strict_canonical)
    if cells[0].model_fingerprint != cells[1].model_fingerprint:
        raise ModelError("monthly and daily dependence cells use different models")
    if cells[0].materialization_fingerprint == cells[1].materialization_fingerprint:
        raise ModelError(
            "monthly and daily dependence materializations are not distinct"
        )
    source_fingerprint = _artifact_execution_fingerprint(
        role=role,
        cells=cells,
        strict_canonical=strict_canonical,
    )
    result = ArtifactDependenceExecution(
        contract_version=ARTIFACT_DEPENDENCE_EXECUTION_CONTRACT,
        role=role,
        cell_order=expected_order,
        cells=cells,
        source_execution_fingerprint=source_fingerprint,
        strict_canonical=strict_canonical,
    )
    object.__setattr__(
        result, "_execution_seal", _REGISTERED_ARTIFACT_DEPENDENCE_CAPABILITY
    )
    _MINTED_ARTIFACT_DEPENDENCE_EXECUTIONS[id(result)] = result
    return result


def build_dependence_task_view(
    source: ArtifactDependenceExecution,
    *,
    task_id: DependenceTaskId,
    run_id: str,
    binding_fingerprint: str,
) -> DependenceTaskView:
    """Create a sealed **report-only** all-at-once compatibility projection.

    This legacy source already contains fragmentation and later dependence
    outcomes.  Canonical guards intentionally reject the returned view; use
    ``build_fragmentation_task_view`` followed by
    ``execute_registered_artifact_dependence_role`` and
    ``build_registered_dependence_task_view`` for scientific publication.
    """

    _verify_artifact_execution(source)
    expected = _TASK_ROLE_AND_KIND.get(task_id)
    if expected is None:
        raise ModelError("dependence task view has an unregistered task ID")
    role, kind = expected
    if role != source.role:
        raise ModelError("dependence task view role does not match its execution")
    _require_sha256(run_id, "dependence task run ID")
    _require_sha256(binding_fingerprint, "dependence task binding fingerprint")
    scoped_cells = _task_scoped_cells(source.cells, kind=kind)
    reasons = _task_view_failure_reasons(scoped_cells, kind=kind)
    provisional = DependenceTaskView(
        contract_version=REPORT_ONLY_DEPENDENCE_TASK_VIEW_CONTRACT,
        task_id=task_id,
        role=role,
        task_kind=kind,
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        source_execution_fingerprint=source.source_execution_fingerprint,
        cells=scoped_cells,
        passed=not reasons,
        failure_reasons=reasons,
        view_fingerprint="",
        strict_canonical=source.strict_canonical,
        parent_block_core_fingerprints=(
            source.cells[0].source_fingerprint,
            source.cells[1].source_fingerprint,
        ),
        fragmentation_parent_view_fingerprint=None,
        canonical_stage=False,
        report_only=True,
    )
    fingerprint = _task_view_fingerprint(provisional)
    result = DependenceTaskView(
        contract_version=provisional.contract_version,
        task_id=provisional.task_id,
        role=provisional.role,
        task_kind=provisional.task_kind,
        run_id=provisional.run_id,
        binding_fingerprint=provisional.binding_fingerprint,
        source_execution_fingerprint=provisional.source_execution_fingerprint,
        cells=provisional.cells,
        passed=provisional.passed,
        failure_reasons=provisional.failure_reasons,
        view_fingerprint=fingerprint,
        strict_canonical=provisional.strict_canonical,
        parent_block_core_fingerprints=(provisional.parent_block_core_fingerprints),
        fragmentation_parent_view_fingerprint=None,
        canonical_stage=False,
        report_only=True,
    )
    object.__setattr__(
        result, "_view_seal", _REPORT_ONLY_DEPENDENCE_TASK_VIEW_CAPABILITY
    )
    _MINTED_REPORT_ONLY_DEPENDENCE_TASK_VIEWS[id(result)] = result
    return result


def assemble_four_cell_dependence_holm_view(
    design: DependenceTaskView,
    production: DependenceTaskView,
    *,
    run_id: str,
    binding_fingerprint: str,
    alpha: float = 0.05,
) -> FourCellDependenceHolmView:
    """Recompute the authoritative Holm decision from two exact task views."""

    _verify_task_view(design)
    _verify_task_view(production)
    if design.task_id != "design_dependence":
        raise ModelError("Holm design source must be design_dependence")
    if production.task_id != "production_dependence":
        raise ModelError("Holm production source must be production_dependence")
    _require_sha256(run_id, "dependence Holm run ID")
    _require_sha256(binding_fingerprint, "dependence Holm binding fingerprint")
    if design.run_id != run_id or production.run_id != run_id:
        raise ModelError("dependence Holm sources belong to different runs")
    source_bindings = (design.binding_fingerprint, production.binding_fingerprint)
    if len(set(source_bindings)) != 2:
        raise ModelError("dependence Holm requires distinct source task bindings")
    if binding_fingerprint in source_bindings:
        raise ModelError("dependence Holm binding must differ from source tasks")
    if design.strict_canonical != production.strict_canonical:
        raise ModelError("dependence Holm sources use different canonical modes")
    if design.source_execution_fingerprint == production.source_execution_fingerprint:
        raise ModelError("dependence Holm sources must be distinct role executions")
    cells = (*_dependence_task_cells(design), *_dependence_task_cells(production))
    holm, reasons = _recomputed_task_view_holm(
        cells,
        alpha=alpha,
        strict_canonical=design.strict_canonical,
    )
    provisional = FourCellDependenceHolmView(
        contract_version=FOUR_CELL_DEPENDENCE_HOLM_VIEW_CONTRACT,
        task_id="four_cell_dependence_holm",
        run_id=run_id,
        binding_fingerprint=binding_fingerprint,
        source_view_fingerprints=(
            design.view_fingerprint,
            production.view_fingerprint,
        ),
        source_views=(design, production),
        cell_order=DEPENDENCE_CELL_ORDER,
        holm=holm,
        passed=not reasons,
        failure_reasons=reasons,
        view_fingerprint="",
        strict_canonical=design.strict_canonical,
    )
    fingerprint = _holm_view_fingerprint(provisional)
    result = FourCellDependenceHolmView(
        contract_version=provisional.contract_version,
        task_id=provisional.task_id,
        run_id=provisional.run_id,
        binding_fingerprint=provisional.binding_fingerprint,
        source_view_fingerprints=provisional.source_view_fingerprints,
        source_views=provisional.source_views,
        cell_order=provisional.cell_order,
        holm=provisional.holm,
        passed=provisional.passed,
        failure_reasons=provisional.failure_reasons,
        view_fingerprint=fingerprint,
        strict_canonical=provisional.strict_canonical,
    )
    object.__setattr__(
        result, "_view_seal", _REGISTERED_DEPENDENCE_HOLM_VIEW_CAPABILITY
    )
    _MINTED_DEPENDENCE_HOLM_VIEWS[id(result)] = result
    return result


def execute_four_cell_dependence_gate(
    cells: tuple[
        DependenceGateCellInput,
        DependenceGateCellInput,
        DependenceGateCellInput,
        DependenceGateCellInput,
    ],
    *,
    materializations: Mapping[DependenceCellId, RegisteredBlockUniformMaterialization]
    | None = None,
    materialization_producer: MaterializationProducer | None = None,
    chunk_size: int = DEFAULT_TRACE_CHUNK_SIZE,
    alpha: float = 0.05,
    strict_canonical: bool = True,
) -> FourCellDependenceGateExecution:
    """Execute all four cells sequentially and apply the authoritative gate.

    Exactly one materialization source is required.  A producer is invoked
    once per cell in fixed order and permits canonical execution to release a
    cell's large uniform matrices before producing the next cell.
    """

    _validate_execution_options(
        cells,
        materializations=materializations,
        materialization_producer=materialization_producer,
        chunk_size=chunk_size,
        strict_canonical=strict_canonical,
    )
    if materializations is not None:
        _validate_no_cross_cell_aliasing(materializations, DEPENDENCE_CELL_ORDER)

    design_materializations = (
        {
            cell_id: materializations[cell_id]
            for cell_id in _ARTIFACT_CELL_ORDER["design"]
        }
        if materializations is not None
        else None
    )
    production_materializations = (
        {
            cell_id: materializations[cell_id]
            for cell_id in _ARTIFACT_CELL_ORDER["production"]
        }
        if materializations is not None
        else None
    )
    design = execute_artifact_dependence_role(
        (cells[0], cells[1]),
        role="design",
        materializations=design_materializations,
        materialization_producer=materialization_producer,
        chunk_size=chunk_size,
        strict_canonical=strict_canonical,
    )
    production = execute_artifact_dependence_role(
        (cells[2], cells[3]),
        role="production",
        materializations=production_materializations,
        materialization_producer=materialization_producer,
        chunk_size=chunk_size,
        strict_canonical=strict_canonical,
    )
    completed_tuple = (
        design.cells[0],
        design.cells[1],
        production.cells[0],
        production.cells[1],
    )
    return assemble_four_cell_dependence_gate(
        completed_tuple,
        alpha=alpha,
        strict_canonical=strict_canonical,
    )


def assemble_four_cell_dependence_gate(
    cells: tuple[
        DependenceCellEvidence,
        DependenceCellEvidence,
        DependenceCellEvidence,
        DependenceCellEvidence,
    ],
    *,
    alpha: float = 0.05,
    strict_canonical: bool = True,
) -> FourCellDependenceGateExecution:
    """Recompute the legacy compact gate from four registered cells only."""

    holm, reasons = _recomputed_holm(
        cells,
        alpha=alpha,
        strict_canonical=strict_canonical,
    )
    result = FourCellDependenceGateExecution(
        contract_version=DEPENDENCE_GATE_CONTRACT,
        cell_order=DEPENDENCE_CELL_ORDER,
        cells=cells,
        holm=holm,
        passed=not reasons,
        failure_reasons=reasons,
        strict_canonical=strict_canonical,
    )
    object.__setattr__(
        result,
        "_execution_seal",
        _REGISTERED_DEPENDENCE_CAPABILITY,
    )
    _MINTED_FOUR_CELL_DEPENDENCE_GATES[id(result)] = result
    return result


def _execute_dependence_cell(
    cell_input: DependenceGateCellInput,
    materialization: RegisteredBlockUniformMaterialization,
    *,
    chunk_size: int,
    strict_canonical: bool,
) -> DependenceCellEvidence:
    geometry = _validate_materialization(
        materialization,
        cell_input=cell_input,
        strict_canonical=strict_canonical,
    )
    prepared = _prepare_cell(cell_input, geometry)
    input_fingerprint = _dependence_input_fingerprint(cell_input, prepared)
    model_fingerprint = _dependence_model_fingerprint(cell_input.model)
    materialization_fingerprint = _dependence_materialization_fingerprint(
        materialization
    )
    loyo = deterministic_loyo_selector_robustness(
        prepared.values,
        prepared.links,
        prepared.years,
        frequency=cell_input.frequency,
    )

    matrices_before = (
        materialization.matrices.monthly_state_uniforms,
        materialization.matrices.restart_uniforms,
        materialization.matrices.source_rank_uniforms,
    )
    block = run_materialized_block_calibration(
        materialization,
        model=cell_input.model,
        starting_length=loyo.full_sample_starting_length,
        historical_source_states=prepared.states,
        source_continuation_links=prepared.links,
        chunk_size=chunk_size,
    )
    requested = feasible_sensitivity_lengths(
        block.selected_length, frequency=cell_input.frequency
    )
    row_continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), prepared.links)
    )
    reference = fit_dependence_reference(
        prepared.values,
        row_continuation,
        frequency=cell_input.frequency,
    )
    reducer = DependenceTraceReducer(
        artifact=cell_input.artifact,
        selected_length=block.selected_length,
        requested_lengths=requested,
        historical_returns=prepared.values,
        reference=reference,
        history_count=materialization.history_count,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if cell_input.frequency == "daily" else None
        ),
    )
    for trace in iter_materialized_block_trace_chunks(
        materialization,
        model=cell_input.model,
        historical_source_states=prepared.states,
        source_continuation_links=prepared.links,
        intended_lengths=requested,
        chunk_size=chunk_size,
    ):
        reducer.consume(trace)
    dependence = reducer.finalize()
    matrices_after = (
        materialization.matrices.monthly_state_uniforms,
        materialization.matrices.restart_uniforms,
        materialization.matrices.source_rank_uniforms,
    )
    common_draw_reused = all(
        before is after
        for before, after in zip(matrices_before, matrices_after, strict=True)
    )

    length_results = tuple(
        DependenceLengthEvidence(
            intended_length=result.intended_length,
            decision_role=(
                "selected"
                if result.intended_length == block.selected_length
                else "report_only_neighbor"
            ),
            observed_statistic=result.cell.observed_statistic,
            critical_value_95=result.cell.critical_value_95,
            p_value=result.cell.p_value,
            active_components=int(np.count_nonzero(result.cell.active_components)),
            repetitions=result.cell.repetitions,
        )
        for result in dependence.results
    )
    if model_fingerprint != _dependence_model_fingerprint(cell_input.model):
        raise ModelError("dependence model changed during registered execution")
    source_fingerprint = _dependence_cell_source_fingerprint(
        cell_id=cell_input.cell_id,
        artifact=cell_input.artifact,
        frequency=cell_input.frequency,
        input_fingerprint=input_fingerprint,
        model_fingerprint=model_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
        strict_canonical=strict_canonical,
    )
    provisional = DependenceCellEvidence(
        cell_id=cell_input.cell_id,
        artifact=cell_input.artifact,
        frequency=cell_input.frequency,
        histories=materialization.history_count,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        target_segment_lengths=geometry.target_segment_lengths,
        starting_length=block.starting_length,
        selected_length=block.selected_length,
        requested_lengths=requested,
        endpoint_count=len(reference.labels),
        endpoint_labels_fingerprint=_labels_fingerprint(reference.labels),
        stream_namespace=_stream_namespace(
            cell_input.artifact,
            cell_input.frequency,
            materialization.history_count,
        ),
        common_draw_reused=common_draw_reused,
        fragmentation=_fragmentation_evidence(block.selected_fragmentation_gates),
        length_results=length_results,
        loyo=loyo,
        input_fingerprint=input_fingerprint,
        model_fingerprint=model_fingerprint,
        materialization_fingerprint=materialization_fingerprint,
        source_fingerprint=source_fingerprint,
        evidence_fingerprint="",
        strict_canonical=strict_canonical,
    )
    result = replace(
        provisional,
        evidence_fingerprint=_dependence_cell_evidence_fingerprint(provisional),
    )
    object.__setattr__(
        result, "_execution_seal", _REGISTERED_DEPENDENCE_CELL_CAPABILITY
    )
    _MINTED_DEPENDENCE_CELLS[id(result)] = result
    return result


def _validated_block_parent_pair(
    parents: object,
    *,
    role: Artifact,
) -> tuple[
    tuple[RegisteredBlockCoreIdentity, RegisteredBlockCoreIdentity],
    tuple[
        RegisteredBlockFragmentationIdentity,
        RegisteredBlockFragmentationIdentity,
    ],
]:
    if role not in _ARTIFACT_CELL_ORDER:
        raise ModelError("registered block parent role must be design or production")
    if (
        not isinstance(parents, tuple)
        or len(parents) != 2
        or any(not isinstance(item, RegisteredBlockCalibration) for item in parents)
    ):
        raise ModelError("staged task requires exactly two registered block parents")
    checked = cast(
        tuple[RegisteredBlockCalibration, RegisteredBlockCalibration], parents
    )
    cores = cast(
        tuple[RegisteredBlockCoreIdentity, RegisteredBlockCoreIdentity],
        tuple(registered_block_core_identity(item) for item in checked),
    )
    fragmentations = cast(
        tuple[
            RegisteredBlockFragmentationIdentity,
            RegisteredBlockFragmentationIdentity,
        ],
        tuple(registered_block_fragmentation_identity(item) for item in checked),
    )
    expected_roles = tuple(_CELL_ROLE[item] for item in _ARTIFACT_CELL_ORDER[role])
    actual_roles = tuple((item.artifact, item.frequency) for item in cores)
    if actual_roles != expected_roles:
        raise ModelError("registered block parents have the wrong role/frequency order")
    if cores[0].selected_k != cores[1].selected_k:
        raise ModelError("monthly and daily block parents select different K")
    if cores[0].parent_model_fingerprint != cores[1].parent_model_fingerprint:
        raise ModelError("monthly and daily block parents use different models")
    if (
        cores[0].master_seed != cores[1].master_seed
        or cores[0].scientific_version != cores[1].scientific_version
    ):
        raise ModelError("monthly and daily block parents use different RNG registry")
    if cores[0].materialization_fingerprint == cores[1].materialization_fingerprint:
        raise ModelError("monthly and daily block parents reuse one materialization")
    strict_modes = tuple(item.execution.geometry.strict_canonical for item in checked)
    if strict_modes[0] != strict_modes[1]:
        raise ModelError("monthly and daily block parents use different modes")
    for core, fragmentation in zip(cores, fragmentations, strict=True):
        if (
            fragmentation.parent_core_fingerprint != core.core_fingerprint
            or fragmentation.selected_k != core.selected_k
            or fragmentation.selected_length != core.selected_length
        ):
            raise ModelError("fragmentation identity does not match its block core")
    return cores, fragmentations


def _fragmentation_cells_from_block_parents(
    parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
    *,
    cores: tuple[RegisteredBlockCoreIdentity, RegisteredBlockCoreIdentity],
    fragmentations: tuple[
        RegisteredBlockFragmentationIdentity,
        RegisteredBlockFragmentationIdentity,
    ],
) -> tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence]:
    cells = tuple(
        FragmentationTaskCellEvidence(
            cell_id=cast(DependenceCellId, f"{core.artifact}_{core.frequency}"),
            artifact=core.artifact,
            frequency=core.frequency,
            histories=core.histories,
            selected_length=core.selected_length,
            common_draw_reused=True,
            fragmentation=_fragmentation_evidence(
                parent.execution.selected_fragmentation_gates
            ),
            input_fingerprint=core.source_history_fingerprint,
            model_fingerprint=core.parent_model_fingerprint,
            materialization_fingerprint=core.materialization_fingerprint,
            source_fingerprint=core.core_fingerprint,
            source_evidence_fingerprint=(
                fragmentation.fragmentation_evidence_fingerprint
            ),
            selected_k=core.selected_k,
            starting_length=core.starting_length,
            parent_block_core_fingerprint=core.core_fingerprint,
            fragmentation_evidence_fingerprint=(
                fragmentation.fragmentation_evidence_fingerprint
            ),
        )
        for parent, core, fragmentation in zip(
            parents, cores, fragmentations, strict=True
        )
    )
    return cast(
        tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence], cells
    )


def _fragmentation_source_fingerprint(
    *,
    role: Artifact,
    cores: tuple[RegisteredBlockCoreIdentity, RegisteredBlockCoreIdentity],
    fragmentations: tuple[
        RegisteredBlockFragmentationIdentity,
        RegisteredBlockFragmentationIdentity,
    ],
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-registered-fragmentation-source-v2",
            "role": role,
            "block_core_fingerprints": [item.core_fingerprint for item in cores],
            "fragmentation_evidence_fingerprints": [
                item.fragmentation_evidence_fingerprint for item in fragmentations
            ],
            "dependence_evidence_excluded": True,
        }
    )


def _validate_staged_role_inputs(
    cells: object,
    *,
    role: Artifact,
    block_parents: object,
    fragmentation_parent: object,
    chunk_size: int,
    strict_canonical: bool,
) -> tuple[
    tuple[RegisteredBlockCoreIdentity, RegisteredBlockCoreIdentity],
    tuple[RegisteredBlockReplaySource, RegisteredBlockReplaySource],
]:
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int | np.integer)
        or int(chunk_size) <= 0
    ):
        raise ModelError("dependence chunk size must be a positive integer")
    if (
        not isinstance(cells, tuple)
        or len(cells) != 2
        or any(not isinstance(item, DependenceGateCellInput) for item in cells)
    ):
        raise ModelError("staged dependence requires exactly two cell inputs")
    cores, _ = _validated_block_parent_pair(block_parents, role=role)
    parents = cast(
        tuple[RegisteredBlockCalibration, RegisteredBlockCalibration], block_parents
    )
    expected_order = _ARTIFACT_CELL_ORDER[role]
    actual_order = tuple(item.cell_id for item in cells)
    if actual_order != expected_order:
        raise ModelError("staged dependence inputs have the wrong role-scoped order")
    for item in cells:
        if (item.artifact, item.frequency) != _CELL_ROLE[item.cell_id]:
            raise ModelError(f"{item.cell_id} has the wrong artifact/frequency role")
    if not isinstance(fragmentation_parent, DependenceTaskView):
        raise ModelError("staged dependence requires a fragmentation task view")
    _verify_task_view(fragmentation_parent)
    expected_fragmentation_task = cast(DependenceTaskId, f"{role}_fragmentation")
    if (
        fragmentation_parent.task_id != expected_fragmentation_task
        or fragmentation_parent.role != role
        or fragmentation_parent.task_kind != "fragmentation"
    ):
        raise ModelError("staged dependence has the wrong fragmentation parent")
    if fragmentation_parent.strict_canonical != strict_canonical:
        raise ModelError("fragmentation and dependence canonical modes differ")
    exact_fragmentation_parents = fragmentation_parent._source_block_parents
    if exact_fragmentation_parents is None or any(
        left is not right
        for left, right in zip(exact_fragmentation_parents, parents, strict=True)
    ):
        raise ModelError("dependence must reuse the exact fragmentation block parents")
    core_fingerprints = tuple(item.core_fingerprint for item in cores)
    if fragmentation_parent.parent_block_core_fingerprints != core_fingerprints:
        raise ModelError("fragmentation parent block-core lineage changed")
    replays = cast(
        tuple[RegisteredBlockReplaySource, RegisteredBlockReplaySource],
        tuple(registered_block_replay_source(item) for item in parents),
    )
    return cores, replays


def _execute_staged_dependence_cell(
    cell_input: DependenceGateCellInput,
    parent: RegisteredBlockCalibration,
    *,
    chunk_size: int,
    strict_canonical: bool,
) -> RegisteredDependenceCellExecution:
    replay = registered_block_replay_source(parent)
    core = replay.core_identity
    uniform_before = replay.uniform_identity
    geometry = _validate_materialization(
        replay.materialization,
        cell_input=cell_input,
        strict_canonical=strict_canonical,
    )
    if (core.artifact, core.frequency) != (
        cell_input.artifact,
        cell_input.frequency,
    ):
        raise ModelError("dependence cell does not match its exact block parent")
    prepared = _prepare_cell(cell_input, geometry)
    if _dependence_model_fingerprint(cell_input.model) != _dependence_model_fingerprint(
        replay.model
    ):
        raise ModelError("dependence cell model differs from its block parent")
    if not np.array_equal(
        prepared.states, replay.historical_source_states
    ) or not np.array_equal(prepared.links, replay.source_continuation_links):
        raise ModelError("dependence history differs from its block parent")
    input_fingerprint = _dependence_input_fingerprint(cell_input, prepared)
    loyo = deterministic_loyo_selector_robustness(
        prepared.values,
        prepared.links,
        prepared.years,
        frequency=cell_input.frequency,
    )
    if loyo.full_sample_starting_length != core.starting_length:
        raise ModelError("dependence selector differs from registered block parent")
    requested = feasible_sensitivity_lengths(
        core.selected_length, frequency=core.frequency
    )
    row_continuation = np.concatenate(
        (np.array([False], dtype=np.bool_), prepared.links)
    )
    reference = fit_dependence_reference(
        prepared.values, row_continuation, frequency=core.frequency
    )
    reducer = DependenceTraceReducer(
        artifact=core.artifact,
        selected_length=core.selected_length,
        requested_lengths=requested,
        historical_returns=prepared.values,
        reference=reference,
        history_count=core.histories,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if core.frequency == "daily" else None
        ),
    )
    matrices_before = (
        replay.materialization.matrices.monthly_state_uniforms,
        replay.materialization.matrices.restart_uniforms,
        replay.materialization.matrices.source_rank_uniforms,
    )
    for trace in iter_materialized_block_trace_chunks(
        replay.materialization,
        model=replay.model,
        historical_source_states=replay.historical_source_states,
        source_continuation_links=replay.source_continuation_links,
        intended_lengths=requested,
        chunk_size=chunk_size,
    ):
        reducer.consume(trace)
    dependence = reducer.finalize()
    matrices_after = (
        replay.materialization.matrices.monthly_state_uniforms,
        replay.materialization.matrices.restart_uniforms,
        replay.materialization.matrices.source_rank_uniforms,
    )
    uniform_after = registered_block_uniform_identity(replay.materialization)
    core_after = registered_block_core_identity(parent)
    common_draw_reused = uniform_before == uniform_after and all(
        before is after
        for before, after in zip(matrices_before, matrices_after, strict=True)
    )
    if not common_draw_reused:
        raise ModelError("dependence did not reuse the exact block materialization")
    if core_after != core:
        raise ModelError("registered block parent changed during dependence replay")
    length_results = tuple(
        DependenceLengthEvidence(
            intended_length=item.intended_length,
            decision_role=(
                "selected"
                if item.intended_length == core.selected_length
                else "report_only_neighbor"
            ),
            observed_statistic=item.cell.observed_statistic,
            critical_value_95=item.cell.critical_value_95,
            p_value=item.cell.p_value,
            active_components=int(np.count_nonzero(item.cell.active_components)),
            repetitions=item.cell.repetitions,
        )
        for item in dependence.results
    )
    result_fingerprint = _staged_dependence_result_fingerprint(
        selected_length=core.selected_length,
        requested_lengths=requested,
        endpoint_count=len(reference.labels),
        endpoint_labels_fingerprint=_labels_fingerprint(reference.labels),
        length_results=length_results,
        loyo=loyo,
    )
    source_fingerprint = _staged_dependence_source_fingerprint(
        cell_id=cell_input.cell_id,
        input_fingerprint=input_fingerprint,
        parent_block_core_fingerprint=core.core_fingerprint,
        materialization_fingerprint=uniform_before.materialization_fingerprint,
        stream_fingerprint=uniform_before.stream_fingerprint,
        rng_fingerprint=uniform_before.rng_execution_fingerprint,
        strict_canonical=strict_canonical,
    )
    provisional = RegisteredDependenceCellExecution(
        cell_id=cell_input.cell_id,
        artifact=core.artifact,
        frequency=core.frequency,
        histories=core.histories,
        starting_length=core.starting_length,
        selected_length=core.selected_length,
        requested_lengths=requested,
        endpoint_count=len(reference.labels),
        endpoint_labels_fingerprint=_labels_fingerprint(reference.labels),
        common_draw_reused=True,
        length_results=length_results,
        loyo=loyo,
        input_fingerprint=input_fingerprint,
        model_fingerprint=core.parent_model_fingerprint,
        parent_block_core_fingerprint=core.core_fingerprint,
        purpose4_materialization_fingerprint=uniform_before.materialization_fingerprint,
        purpose4_stream_fingerprint=uniform_before.stream_fingerprint,
        rng_execution_fingerprint=uniform_before.rng_execution_fingerprint,
        dependence_result_fingerprint=result_fingerprint,
        source_fingerprint=source_fingerprint,
        evidence_fingerprint="",
        strict_canonical=strict_canonical,
    )
    result = replace(
        provisional,
        evidence_fingerprint=_staged_dependence_cell_fingerprint(provisional),
    )
    object.__setattr__(
        result, "_execution_seal", _REGISTERED_STAGED_DEPENDENCE_CELL_CAPABILITY
    )
    _MINTED_STAGED_DEPENDENCE_CELLS[id(result)] = result
    return result


def _dependence_cells_from_staged_execution(
    cells: tuple[RegisteredDependenceCellExecution, RegisteredDependenceCellExecution],
    parents: tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
) -> tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence]:
    cores = tuple(registered_block_core_identity(item) for item in parents)
    result = tuple(
        DependenceTaskCellEvidence(
            cell_id=cell.cell_id,
            artifact=cell.artifact,
            frequency=cell.frequency,
            histories=cell.histories,
            selected_length=cell.selected_length,
            endpoint_count=cell.endpoint_count,
            common_draw_reused=cell.common_draw_reused,
            selected_result=cell.selected_result,
            loyo_fingerprint=cell.loyo.fingerprint,
            input_fingerprint=cell.input_fingerprint,
            model_fingerprint=cell.model_fingerprint,
            materialization_fingerprint=(cell.purpose4_materialization_fingerprint),
            source_fingerprint=cell.source_fingerprint,
            source_evidence_fingerprint=cell.evidence_fingerprint,
            selected_k=core.selected_k,
            starting_length=cell.starting_length,
            parent_block_core_fingerprint=cell.parent_block_core_fingerprint,
            purpose4_materialization_fingerprint=(
                cell.purpose4_materialization_fingerprint
            ),
            purpose4_stream_fingerprint=cell.purpose4_stream_fingerprint,
            rng_execution_fingerprint=cell.rng_execution_fingerprint,
            dependence_result_fingerprint=cell.dependence_result_fingerprint,
        )
        for cell, core in zip(cells, cores, strict=True)
    )
    return cast(tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence], result)


def _validate_execution_options(
    cells: object,
    *,
    materializations: Mapping[DependenceCellId, RegisteredBlockUniformMaterialization]
    | None,
    materialization_producer: MaterializationProducer | None,
    chunk_size: int,
    strict_canonical: bool,
) -> None:
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int | np.integer)
        or int(chunk_size) <= 0
    ):
        raise ModelError("dependence chunk size must be a positive integer")
    if not isinstance(cells, tuple) or len(cells) != 4:
        raise ModelError("dependence execution requires exactly four cell inputs")
    if any(not isinstance(cell, DependenceGateCellInput) for cell in cells):
        raise ModelError("dependence execution received an invalid cell input")
    actual = tuple(cell.cell_id for cell in cells)
    if actual != DEPENDENCE_CELL_ORDER:
        raise ModelError("dependence input cells must use the fixed order")
    for cell in cells:
        expected = _CELL_ROLE[cell.cell_id]
        if (cell.artifact, cell.frequency) != expected:
            raise ModelError(f"{cell.cell_id} has the wrong artifact/frequency role")
    if (materializations is None) == (materialization_producer is None):
        raise ModelError(
            "provide exactly one of materializations or materialization_producer"
        )
    if materializations is not None:
        if set(materializations) != set(DEPENDENCE_CELL_ORDER):
            raise ModelError(
                "materialization mapping must contain the exact four cells"
            )
        if any(
            not isinstance(item, RegisteredBlockUniformMaterialization)
            for item in materializations.values()
        ):
            raise ModelError("materialization mapping contains an invalid value")
    if materialization_producer is not None and not callable(materialization_producer):
        raise ModelError("materialization producer must be callable")


def _validate_role_execution_options(
    cells: object,
    *,
    role: Artifact,
    materializations: Mapping[DependenceCellId, RegisteredBlockUniformMaterialization]
    | None,
    materialization_producer: MaterializationProducer | None,
    chunk_size: int,
    strict_canonical: bool,
) -> tuple[DependenceCellId, DependenceCellId]:
    if role not in _ARTIFACT_CELL_ORDER:
        raise ModelError("artifact dependence role must be design or production")
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int | np.integer)
        or int(chunk_size) <= 0
    ):
        raise ModelError("dependence chunk size must be a positive integer")
    if not isinstance(cells, tuple) or len(cells) != 2:
        raise ModelError("artifact dependence execution requires exactly two inputs")
    if any(not isinstance(cell, DependenceGateCellInput) for cell in cells):
        raise ModelError("artifact dependence execution received an invalid input")
    expected_order = _ARTIFACT_CELL_ORDER[role]
    actual_order = tuple(cell.cell_id for cell in cells)
    if actual_order != expected_order:
        raise ModelError("artifact dependence inputs have the wrong role-scoped order")
    for cell in cells:
        expected = _CELL_ROLE[cell.cell_id]
        if (cell.artifact, cell.frequency) != expected:
            raise ModelError(f"{cell.cell_id} has the wrong artifact/frequency role")
    if (materializations is None) == (materialization_producer is None):
        raise ModelError(
            "provide exactly one of materializations or materialization_producer"
        )
    if materializations is not None:
        if set(materializations) != set(expected_order):
            raise ModelError(
                "role materialization mapping must contain the exact two cells"
            )
        if any(
            not isinstance(item, RegisteredBlockUniformMaterialization)
            for item in materializations.values()
        ):
            raise ModelError("materialization mapping contains an invalid value")
    if materialization_producer is not None and not callable(materialization_producer):
        raise ModelError("materialization producer must be callable")
    return expected_order


def _produce_materialization(
    producer: MaterializationProducer | None,
    cell_input: DependenceGateCellInput,
) -> RegisteredBlockUniformMaterialization:
    if producer is None:  # pragma: no cover - guarded by option validation.
        raise AssertionError("materialization producer is unexpectedly absent")
    result = producer(cell_input)
    if not isinstance(result, RegisteredBlockUniformMaterialization):
        raise ModelError(
            "materialization producer returned the wrong evidence type",
            details={"cell_id": cell_input.cell_id},
        )
    return result


def _validate_materialization(
    materialization: RegisteredBlockUniformMaterialization,
    *,
    cell_input: DependenceGateCellInput,
    strict_canonical: bool,
) -> CalibrationGeometry:
    if not isinstance(materialization, RegisteredBlockUniformMaterialization):
        raise ModelError("dependence execution requires a registered materialization")
    geometry = materialization.geometry
    if (
        geometry.artifact != cell_input.artifact
        or geometry.frequency != cell_input.frequency
    ):
        raise ModelError(
            "registered materialization has the wrong artifact/frequency identity"
        )
    if geometry.strict_canonical != strict_canonical:
        raise ModelError("registered materialization canonical mode is inconsistent")
    histories = materialization.history_count
    if (
        isinstance(histories, bool)
        or not isinstance(histories, int | np.integer)
        or int(histories) <= 0
    ):
        raise ModelError("registered materialization history count is invalid")
    if strict_canonical and histories != CANONICAL_HISTORY_COUNT:
        raise ModelError(
            "canonical dependence materialization requires 10,000 histories"
        )

    arrays = tuple(
        np.asarray(value)
        for value in (
            materialization.matrices.monthly_state_uniforms,
            materialization.matrices.restart_uniforms,
            materialization.matrices.source_rank_uniforms,
        )
    )
    expected_shapes = (
        (histories, geometry.monthly_observations),
        (histories, geometry.target_observations),
        (histories, geometry.target_observations),
    )
    for array, shape in zip(arrays, expected_shapes, strict=True):
        if array.shape != shape or array.dtype != np.dtype(np.float64):
            raise ModelError(
                "registered materialization matrix has the wrong exact shape"
            )
        if array.flags.writeable:
            raise ModelError("registered materialization matrices must be immutable")
    if any(
        np.shares_memory(arrays[left], arrays[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ModelError("registered purpose-4 stage matrices must not alias")
    expected_bytes = registered_block_uniform_bytes(
        artifact=cell_input.artifact,
        frequency=cell_input.frequency,
        history_count=histories,
        strict_canonical=strict_canonical,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if cell_input.frequency == "daily" else None
        ),
    )
    actual_bytes = sum(array.nbytes for array in arrays)
    if (
        actual_bytes != expected_bytes
        or materialization.allocated_bytes != expected_bytes
    ):
        raise ModelError("registered materialization byte accounting is inconsistent")
    return geometry


def _validate_no_cross_cell_aliasing(
    materializations: Mapping[DependenceCellId, RegisteredBlockUniformMaterialization],
    order: Sequence[DependenceCellId] = DEPENDENCE_CELL_ORDER,
) -> None:
    ordered = tuple(materializations[cell_id] for cell_id in order)
    if len({id(item) for item in ordered}) != len(order):
        raise ModelError("distinct dependence cells cannot share a materialization")
    matrices = tuple(
        tuple(
            np.asarray(value)
            for value in (
                item.matrices.monthly_state_uniforms,
                item.matrices.restart_uniforms,
                item.matrices.source_rank_uniforms,
            )
        )
        for item in ordered
    )
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            if any(
                np.shares_memory(left_array, right_array)
                for left_array in matrices[left]
                for right_array in matrices[right]
            ):
                raise ModelError("distinct dependence stream namespaces cannot alias")


def _prepare_cell(
    cell_input: DependenceGateCellInput, geometry: CalibrationGeometry
) -> _PreparedCell:
    try:
        values = np.asarray(cell_input.historical_returns, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("historical dependence returns must be numeric") from error
    states_raw = np.asarray(cell_input.historical_source_states)
    links_raw = np.asarray(cell_input.source_continuation_links)
    years_raw = np.asarray(cell_input.calendar_years)
    rows = geometry.target_observations
    if values.shape != (rows, 3) or not np.isfinite(values).all():
        raise ModelError("historical dependence returns have the wrong exact geometry")
    if states_raw.shape != (rows,) or states_raw.dtype.kind not in {"i", "u"}:
        raise ModelError("historical dependence states have the wrong exact geometry")
    if links_raw.shape != (rows - 1,) or links_raw.dtype.kind != "b":
        raise ModelError("historical dependence links have the wrong exact geometry")
    links = links_raw.astype(np.bool_, copy=True)
    if not np.array_equal(links, geometry.target_continuation_links):
        raise ModelError("historical dependence links do not match artifact segments")
    if years_raw.shape != (rows,) or years_raw.dtype.kind not in {"i", "u"}:
        raise ModelError("calendar years must be an integer vector matching returns")
    years = years_raw.astype(np.int64, copy=True)
    if (
        bool((years < 1).any())
        or bool((years > 9_999).any())
        or bool((years[1:] < years[:-1]).any())
    ):
        raise ModelError("calendar years must be ordered values in [1, 9999]")
    values = np.array(values, dtype=np.float64, copy=True)
    states = states_raw.astype(np.int64, copy=True)
    for array in (values, states, links, years):
        array.setflags(write=False)
    return _PreparedCell(values, states, links, years)


def _selector_inputs(
    returns: npt.ArrayLike,
    continuation_links: npt.ArrayLike,
    calendar_years: npt.ArrayLike,
    *,
    frequency: Frequency,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    if frequency not in {"monthly", "daily"}:
        raise ModelError("LOYO frequency must be monthly or daily")
    try:
        values = np.asarray(returns, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ModelError("LOYO returns must be numeric") from error
    links_raw = np.asarray(continuation_links)
    years_raw = np.asarray(calendar_years)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ModelError("LOYO returns must be a finite three-column matrix")
    if links_raw.shape != (len(values) - 1,) or links_raw.dtype.kind != "b":
        raise ModelError("LOYO continuation must be a Boolean forward-link vector")
    if years_raw.shape != (len(values),) or years_raw.dtype.kind not in {"i", "u"}:
        raise ModelError("LOYO years must be an integer vector matching returns")
    years = years_raw.astype(np.int64, copy=False)
    if (
        bool((years < 1).any())
        or bool((years > 9_999).any())
        or bool((years[1:] < years[:-1]).any())
    ):
        raise ModelError("LOYO years must be ordered values in [1, 9999]")
    return values, links_raw.astype(np.bool_, copy=False), years


def _retained_links(
    retained: npt.NDArray[np.int64], original_links: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    if retained.size <= 1:
        return np.zeros(0, dtype=np.bool_)
    adjacent = retained[1:] == retained[:-1] + 1
    links = adjacent & original_links[retained[:-1]]
    return np.asarray(links, dtype=np.bool_)


def _fragmentation_evidence(
    gates: Sequence[PooledFragmentationGate],
) -> tuple[FragmentationStateEvidence, ...]:
    result = tuple(
        FragmentationStateEvidence(
            state=gate.state,
            intended_length=gate.intended_length,
            conditioned_count=gate.conditioned.count,
            conditioned_mean=gate.conditioned.mean,
            conditioned_median=gate.conditioned.median,
            conditioned_singleton_fraction=gate.conditioned.singleton_fraction,
            ideal_count=gate.ideal.count,
            ideal_median=gate.ideal.median,
            ideal_singleton_fraction=gate.ideal.singleton_fraction,
            reasons=gate.reasons,
        )
        for gate in gates
    )
    if not result or tuple(item.state for item in result) != tuple(
        sorted(item.state for item in result)
    ):
        raise ModelError("selected fragmentation evidence has invalid state order")
    return result


def _validate_compact_cell(
    cell: DependenceCellEvidence, *, strict_canonical: bool
) -> None:
    if not isinstance(cell, DependenceCellEvidence):
        raise ModelError("dependence gate received an invalid compact cell")
    if not isinstance(cell.strict_canonical, bool):
        raise ModelError("dependence cell canonical mode must be boolean")
    if not cell.report_only:
        raise ModelError("combined dependence cells must remain report-only")
    if cell.strict_canonical != strict_canonical:
        raise ModelError("dependence cell canonical mode is inconsistent")
    expected_artifact, expected_frequency = _CELL_ROLE[cell.cell_id]
    if (cell.artifact, cell.frequency) != (expected_artifact, expected_frequency):
        raise ModelError(f"{cell.cell_id} compact role is inconsistent")
    expected_lengths = feasible_sensitivity_lengths(
        cell.selected_length, frequency=cell.frequency
    )
    if cell.requested_lengths != expected_lengths:
        raise ModelError(f"{cell.cell_id} omits a feasible sensitivity neighbor")
    if (
        tuple(result.intended_length for result in cell.length_results)
        != expected_lengths
    ):
        raise ModelError(f"{cell.cell_id} dependence result order is inconsistent")
    for result in cell.length_results:
        expected_role: DecisionRole = (
            "selected"
            if result.intended_length == cell.selected_length
            else "report_only_neighbor"
        )
        if result.decision_role != expected_role:
            raise ModelError(f"{cell.cell_id} dependence result role is inconsistent")
        if (
            not math.isfinite(result.observed_statistic)
            or not math.isfinite(result.critical_value_95)
            or not math.isfinite(result.p_value)
            or not 0.0 <= result.p_value <= 1.0
            or not 0 <= result.active_components <= cell.endpoint_count
            or result.repetitions != cell.histories
        ):
            raise ModelError(f"{cell.cell_id} has an invalid dependence result")
    if cell.endpoint_count != (87 if cell.frequency == "monthly" else 31):
        raise ModelError(f"{cell.cell_id} has the wrong endpoint count")
    if _SHA256.fullmatch(cell.endpoint_labels_fingerprint) is None:
        raise ModelError(f"{cell.cell_id} has an invalid endpoint-label fingerprint")
    if cell.stream_namespace != _stream_namespace(
        cell.artifact, cell.frequency, cell.histories
    ):
        raise ModelError(f"{cell.cell_id} stream namespace is inconsistent")
    if not cell.fragmentation or tuple(
        item.state for item in cell.fragmentation
    ) != tuple(sorted(item.state for item in cell.fragmentation)):
        raise ModelError(f"{cell.cell_id} fragmentation state order is inconsistent")
    if any(item.intended_length != cell.selected_length for item in cell.fragmentation):
        raise ModelError(f"{cell.cell_id} fragmentation length is inconsistent")
    _verify_loyo(cell.loyo)
    if (
        cell.loyo.frequency != cell.frequency
        or cell.loyo.observations != sum(cell.target_segment_lengths)
        or cell.loyo.full_sample_starting_length != cell.starting_length
    ):
        raise ModelError(f"{cell.cell_id} LOYO selector identity is inconsistent")
    if strict_canonical:
        if cell.histories != CANONICAL_HISTORY_COUNT:
            raise ModelError(f"{cell.cell_id} does not contain 10,000 histories")
        geometry = _canonical_geometry(cell.artifact, cell.frequency)
        if (
            cell.monthly_segment_lengths != geometry.monthly_segment_lengths
            or cell.target_segment_lengths != geometry.target_segment_lengths
        ):
            raise ModelError(f"{cell.cell_id} has noncanonical geometry")


def _verify_registered_cell(
    cell: DependenceCellEvidence, *, strict_canonical: bool
) -> None:
    if not isinstance(cell, DependenceCellEvidence):
        raise ModelError("dependence execution requires registered compact cells")
    _validate_compact_cell(cell, strict_canonical=strict_canonical)
    if (
        cell._execution_seal is not _REGISTERED_DEPENDENCE_CELL_CAPABILITY
        or _MINTED_DEPENDENCE_CELLS.get(id(cell)) is not cell
    ):
        raise ModelError("dependence compact cell lacks registered execution authority")
    for value, label in (
        (cell.input_fingerprint, "input"),
        (cell.model_fingerprint, "model"),
        (cell.materialization_fingerprint, "materialization"),
        (cell.source_fingerprint, "source"),
        (cell.evidence_fingerprint, "evidence"),
    ):
        _require_sha256(value, f"dependence cell {label} fingerprint")
    expected_source = _dependence_cell_source_fingerprint(
        cell_id=cell.cell_id,
        artifact=cell.artifact,
        frequency=cell.frequency,
        input_fingerprint=cell.input_fingerprint,
        model_fingerprint=cell.model_fingerprint,
        materialization_fingerprint=cell.materialization_fingerprint,
        strict_canonical=cell.strict_canonical,
    )
    if cell.source_fingerprint != expected_source:
        raise ModelError("dependence compact cell source fingerprint is inconsistent")
    if cell.evidence_fingerprint != _dependence_cell_evidence_fingerprint(cell):
        raise ModelError("dependence compact cell evidence fingerprint is inconsistent")


def _verify_staged_dependence_cell(
    cell: RegisteredDependenceCellExecution,
) -> None:
    if not isinstance(cell, RegisteredDependenceCellExecution):
        raise ModelError("staged dependence cell has the wrong type")
    if (
        cell._execution_seal is not _REGISTERED_STAGED_DEPENDENCE_CELL_CAPABILITY
        or _MINTED_STAGED_DEPENDENCE_CELLS.get(id(cell)) is not cell
    ):
        raise ModelError("staged dependence cell lacks producer authority")
    expected_role = _CELL_ROLE.get(cell.cell_id)
    if expected_role != (cell.artifact, cell.frequency):
        raise ModelError("staged dependence cell role is inconsistent")
    if not isinstance(cell.strict_canonical, bool):
        raise ModelError("staged dependence canonical mode must be boolean")
    if not cell.neighbors_report_only or not cell.loyo_report_only:
        raise ModelError("staged dependence robustness evidence must be report-only")
    if (
        isinstance(cell.histories, bool)
        or not isinstance(cell.histories, int)
        or cell.histories <= 0
        or isinstance(cell.starting_length, bool)
        or not isinstance(cell.starting_length, int)
        or cell.starting_length <= 0
        or isinstance(cell.selected_length, bool)
        or not isinstance(cell.selected_length, int)
        or cell.selected_length <= 0
        or not cell.common_draw_reused
    ):
        raise ModelError("staged dependence cell geometry is invalid")
    if cell.strict_canonical and cell.histories != CANONICAL_HISTORY_COUNT:
        raise ModelError("canonical staged dependence requires 10,000 histories")
    expected_lengths = feasible_sensitivity_lengths(
        cell.selected_length, frequency=cell.frequency
    )
    if (
        cell.requested_lengths != expected_lengths
        or tuple(item.intended_length for item in cell.length_results)
        != expected_lengths
    ):
        raise ModelError("staged dependence sensitivity lengths are inconsistent")
    if cell.endpoint_count != (87 if cell.frequency == "monthly" else 31):
        raise ModelError("staged dependence endpoint count is inconsistent")
    for item in cell.length_results:
        expected_decision_role: DecisionRole = (
            "selected"
            if item.intended_length == cell.selected_length
            else "report_only_neighbor"
        )
        if (
            item.decision_role != expected_decision_role
            or item.repetitions != cell.histories
            or not math.isfinite(item.observed_statistic)
            or not math.isfinite(item.critical_value_95)
            or not math.isfinite(item.p_value)
            or not 0.0 <= item.p_value <= 1.0
            or not 0 <= item.active_components <= cell.endpoint_count
        ):
            raise ModelError("staged dependence result is invalid")
    _verify_loyo(cell.loyo)
    if (
        cell.loyo.frequency != cell.frequency
        or cell.loyo.full_sample_starting_length != cell.starting_length
    ):
        raise ModelError("staged dependence LOYO identity is inconsistent")
    for fingerprint, label in (
        (cell.endpoint_labels_fingerprint, "endpoint labels"),
        (cell.input_fingerprint, "input"),
        (cell.model_fingerprint, "model"),
        (cell.parent_block_core_fingerprint, "parent block core"),
        (cell.purpose4_materialization_fingerprint, "purpose-4 materialization"),
        (cell.purpose4_stream_fingerprint, "purpose-4 stream"),
        (cell.rng_execution_fingerprint, "RNG execution"),
        (cell.dependence_result_fingerprint, "dependence result"),
        (cell.source_fingerprint, "source"),
        (cell.evidence_fingerprint, "evidence"),
    ):
        _require_sha256(fingerprint, f"staged dependence {label} fingerprint")
    expected_result = _staged_dependence_result_fingerprint(
        selected_length=cell.selected_length,
        requested_lengths=cell.requested_lengths,
        endpoint_count=cell.endpoint_count,
        endpoint_labels_fingerprint=cell.endpoint_labels_fingerprint,
        length_results=cell.length_results,
        loyo=cell.loyo,
    )
    if cell.dependence_result_fingerprint != expected_result:
        raise ModelError("staged dependence result fingerprint is inconsistent")
    expected_source = _staged_dependence_source_fingerprint(
        cell_id=cell.cell_id,
        input_fingerprint=cell.input_fingerprint,
        parent_block_core_fingerprint=cell.parent_block_core_fingerprint,
        materialization_fingerprint=cell.purpose4_materialization_fingerprint,
        stream_fingerprint=cell.purpose4_stream_fingerprint,
        rng_fingerprint=cell.rng_execution_fingerprint,
        strict_canonical=cell.strict_canonical,
    )
    if cell.source_fingerprint != expected_source:
        raise ModelError("staged dependence source fingerprint is inconsistent")
    if cell.evidence_fingerprint != _staged_dependence_cell_fingerprint(cell):
        raise ModelError("staged dependence evidence fingerprint is inconsistent")


def _verify_staged_dependence_execution(
    value: RegisteredArtifactDependenceExecution,
) -> None:
    if not isinstance(value, RegisteredArtifactDependenceExecution):
        raise ModelError("staged dependence execution has the wrong type")
    if (
        value._execution_seal is not _REGISTERED_STAGED_DEPENDENCE_EXECUTION_CAPABILITY
        or _MINTED_STAGED_DEPENDENCE_EXECUTIONS.get(id(value)) is not value
    ):
        raise ModelError("staged dependence execution lacks producer authority")
    if value.contract_version != STAGED_DEPENDENCE_EXECUTION_CONTRACT:
        raise ModelError("staged dependence execution contract is inconsistent")
    if value.role not in _ARTIFACT_CELL_ORDER:
        raise ModelError("staged dependence execution has an invalid role")
    if not isinstance(value.strict_canonical, bool):
        raise ModelError("staged dependence canonical mode must be boolean")
    if not value.neighbors_report_only or not value.loyo_report_only:
        raise ModelError("staged dependence robustness evidence must be report-only")
    parents = value._block_parents
    fragmentation_parent = value._fragmentation_parent
    if parents is None or fragmentation_parent is None:
        raise ModelError("staged dependence execution source parents are unavailable")
    cores, _ = _validated_block_parent_pair(parents, role=value.role)
    _verify_task_view(fragmentation_parent)
    expected_task = cast(DependenceTaskId, f"{value.role}_fragmentation")
    if fragmentation_parent.task_id != expected_task:
        raise ModelError("staged dependence fragmentation role is inconsistent")
    if fragmentation_parent._source_block_parents is None or any(
        left is not right
        for left, right in zip(
            fragmentation_parent._source_block_parents, parents, strict=True
        )
    ):
        raise ModelError("staged dependence does not reuse fragmentation parents")
    expected_order = _ARTIFACT_CELL_ORDER[value.role]
    if (
        value.cell_order != expected_order
        or tuple(item.cell_id for item in value.cells) != expected_order
    ):
        raise ModelError("staged dependence cell order is inconsistent")
    expected_core_fingerprints = tuple(item.core_fingerprint for item in cores)
    if (
        value.parent_block_core_fingerprints != expected_core_fingerprints
        or fragmentation_parent.parent_block_core_fingerprints
        != expected_core_fingerprints
    ):
        raise ModelError("staged dependence block-core lineage is inconsistent")
    if value.fragmentation_parent_view_fingerprint != (
        fragmentation_parent.view_fingerprint
    ):
        raise ModelError("staged dependence fragmentation lineage is inconsistent")
    if value.strict_canonical != fragmentation_parent.strict_canonical:
        raise ModelError("staged dependence source modes are inconsistent")
    for cell, parent, core in zip(value.cells, parents, cores, strict=True):
        _verify_staged_dependence_cell(cell)
        replay = registered_block_replay_source(parent)
        if (
            cell.parent_block_core_fingerprint != core.core_fingerprint
            or cell.model_fingerprint != core.parent_model_fingerprint
            or cell.starting_length != core.starting_length
            or cell.selected_length != core.selected_length
            or cell.histories != core.histories
            or cell.purpose4_materialization_fingerprint
            != replay.uniform_identity.materialization_fingerprint
            or cell.purpose4_stream_fingerprint
            != replay.uniform_identity.stream_fingerprint
            or cell.rng_execution_fingerprint
            != replay.uniform_identity.rng_execution_fingerprint
            or cell.strict_canonical != value.strict_canonical
        ):
            raise ModelError("staged dependence cell does not match block parent")
    if (
        value.cells[0].purpose4_materialization_fingerprint
        == value.cells[1].purpose4_materialization_fingerprint
    ):
        raise ModelError("staged dependence role reuses one cell materialization")
    expected_fingerprint = _staged_artifact_execution_fingerprint(
        role=value.role,
        cells=value.cells,
        parent_core_fingerprints=value.parent_block_core_fingerprints,
        fragmentation_parent_view_fingerprint=(
            value.fragmentation_parent_view_fingerprint
        ),
        strict_canonical=value.strict_canonical,
    )
    if value.source_execution_fingerprint != expected_fingerprint:
        raise ModelError("staged dependence execution fingerprint is inconsistent")


def _verify_artifact_execution(value: ArtifactDependenceExecution) -> None:
    if not isinstance(value, ArtifactDependenceExecution):
        raise ModelError("artifact dependence execution has the wrong type")
    if (
        value._execution_seal is not _REGISTERED_ARTIFACT_DEPENDENCE_CAPABILITY
        or _MINTED_ARTIFACT_DEPENDENCE_EXECUTIONS.get(id(value)) is not value
    ):
        raise ModelError("artifact dependence execution lacks registered authority")
    if value.contract_version != ARTIFACT_DEPENDENCE_EXECUTION_CONTRACT:
        raise ModelError("artifact dependence execution contract is inconsistent")
    if not value.neighbors_report_only or not value.loyo_report_only:
        raise ModelError("artifact dependence robustness evidence must be report-only")
    if not value.report_only:
        raise ModelError("combined artifact dependence execution must be report-only")
    if value.role not in _ARTIFACT_CELL_ORDER:
        raise ModelError("artifact dependence execution has an invalid role")
    if not isinstance(value.strict_canonical, bool):
        raise ModelError("artifact dependence canonical mode must be boolean")
    expected_order = _ARTIFACT_CELL_ORDER[value.role]
    if (
        value.cell_order != expected_order
        or tuple(cell.cell_id for cell in value.cells) != expected_order
    ):
        raise ModelError("artifact dependence execution cell order is inconsistent")
    if not isinstance(value.cells, tuple) or len(value.cells) != 2:
        raise ModelError("artifact dependence execution must contain exactly two cells")
    for cell in value.cells:
        _verify_registered_cell(cell, strict_canonical=value.strict_canonical)
    if value.cells[0].model_fingerprint != value.cells[1].model_fingerprint:
        raise ModelError("artifact dependence execution mixes fitted models")
    if (
        value.cells[0].materialization_fingerprint
        == value.cells[1].materialization_fingerprint
    ):
        raise ModelError("artifact dependence execution reuses one materialization")
    expected_fingerprint = _artifact_execution_fingerprint(
        role=value.role,
        cells=value.cells,
        strict_canonical=value.strict_canonical,
    )
    if value.source_execution_fingerprint != expected_fingerprint:
        raise ModelError("artifact dependence source fingerprint is inconsistent")


def _verify_task_view(value: DependenceTaskView) -> None:
    if not isinstance(value, DependenceTaskView):
        raise ModelError("dependence task view has the wrong type")
    if (
        value._view_seal is not _REGISTERED_DEPENDENCE_TASK_VIEW_CAPABILITY
        or _MINTED_DEPENDENCE_TASK_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("dependence task view lacks registered authority")
    if value.contract_version != STAGED_DEPENDENCE_TASK_VIEW_CONTRACT:
        raise ModelError("staged dependence task view contract is inconsistent")
    if not value.canonical_stage or value.report_only:
        raise ModelError("canonical dependence guard rejects report-only views")
    if not value.neighbors_report_only or not value.loyo_report_only:
        raise ModelError("dependence task robustness evidence must be report-only")
    expected = _TASK_ROLE_AND_KIND.get(value.task_id)
    if expected is None or expected != (value.role, value.task_kind):
        raise ModelError("dependence task ID, role, and kind are inconsistent")
    _require_sha256(value.run_id, "dependence task run ID")
    _require_sha256(value.binding_fingerprint, "dependence task binding fingerprint")
    _require_sha256(
        value.source_execution_fingerprint,
        "dependence task source execution fingerprint",
    )
    _require_sha256(value.view_fingerprint, "dependence task view fingerprint")
    _verify_task_scoped_cells(value)
    parents = value._source_block_parents
    if parents is None:
        raise ModelError("dependence task view has no exact block parents")
    cores, fragmentations = _validated_block_parent_pair(parents, role=value.role)
    expected_core_fingerprints = (
        cores[0].core_fingerprint,
        cores[1].core_fingerprint,
    )
    if value.parent_block_core_fingerprints != expected_core_fingerprints:
        raise ModelError("dependence task view block-core lineage is inconsistent")
    if value.strict_canonical != parents[0].execution.geometry.strict_canonical:
        raise ModelError("dependence task view parent mode is inconsistent")
    expected_cells: (
        tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence]
        | tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence]
    )
    if value.task_kind == "fragmentation":
        if (
            value.fragmentation_parent_view_fingerprint is not None
            or value._source_fragmentation_view is not None
            or value._source_staged_execution is not None
        ):
            raise ModelError("fragmentation task view contains future evidence")
        expected_cells = _fragmentation_cells_from_block_parents(
            parents, cores=cores, fragmentations=fragmentations
        )
        expected_source = _fragmentation_source_fingerprint(
            role=value.role,
            cores=cores,
            fragmentations=fragmentations,
        )
    else:
        fragmentation_parent = value._source_fragmentation_view
        staged_execution = value._source_staged_execution
        if fragmentation_parent is None or staged_execution is None:
            raise ModelError("dependence task view lacks its staged parents")
        _verify_task_view(fragmentation_parent)
        _verify_staged_dependence_execution(staged_execution)
        if (
            staged_execution._block_parents is None
            or any(
                left is not right
                for left, right in zip(
                    staged_execution._block_parents, parents, strict=True
                )
            )
            or staged_execution._fragmentation_parent is not fragmentation_parent
        ):
            raise ModelError("dependence task view source objects changed")
        if (
            fragmentation_parent.run_id != value.run_id
            or fragmentation_parent.role != value.role
            or fragmentation_parent.task_kind != "fragmentation"
        ):
            raise ModelError("dependence task fragmentation topology is inconsistent")
        if fragmentation_parent.binding_fingerprint == value.binding_fingerprint:
            raise ModelError("dependence and fragmentation bindings must differ")
        if value.fragmentation_parent_view_fingerprint != (
            fragmentation_parent.view_fingerprint
        ):
            raise ModelError("dependence task fragmentation fingerprint changed")
        expected_cells = _dependence_cells_from_staged_execution(
            staged_execution.cells, parents
        )
        expected_source = staged_execution.source_execution_fingerprint
    if value.cells != expected_cells:
        raise ModelError("dependence task view cells changed from exact parents")
    if value.source_execution_fingerprint != expected_source:
        raise ModelError("dependence task source fingerprint is inconsistent")
    reasons = _task_view_failure_reasons(value.cells, kind=value.task_kind)
    if value.failure_reasons != reasons or value.passed != (not reasons):
        raise ModelError("dependence task view decision is inconsistent")
    if value.view_fingerprint != _task_view_fingerprint(value):
        raise ModelError("dependence task view fingerprint is inconsistent")


def _verify_report_only_task_view(value: DependenceTaskView) -> None:
    if not isinstance(value, DependenceTaskView):
        raise ModelError("report-only dependence task view has the wrong type")
    if (
        value._view_seal is not _REPORT_ONLY_DEPENDENCE_TASK_VIEW_CAPABILITY
        or _MINTED_REPORT_ONLY_DEPENDENCE_TASK_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("report-only dependence task view lacks authority")
    if (
        value.contract_version != REPORT_ONLY_DEPENDENCE_TASK_VIEW_CONTRACT
        or value.canonical_stage
        or not value.report_only
    ):
        raise ModelError("report-only dependence task contract is inconsistent")
    expected = _TASK_ROLE_AND_KIND.get(value.task_id)
    if expected is None or expected != (value.role, value.task_kind):
        raise ModelError("report-only dependence task identity is inconsistent")
    _require_sha256(value.run_id, "report-only dependence run ID")
    _require_sha256(
        value.binding_fingerprint, "report-only dependence binding fingerprint"
    )
    _verify_task_scoped_cells(value)
    reasons = _task_view_failure_reasons(value.cells, kind=value.task_kind)
    if value.failure_reasons != reasons or value.passed != (not reasons):
        raise ModelError("report-only dependence decision is inconsistent")
    if value.view_fingerprint != _task_view_fingerprint(value):
        raise ModelError("report-only dependence fingerprint is inconsistent")


def _verify_holm_view(value: FourCellDependenceHolmView) -> None:
    if not isinstance(value, FourCellDependenceHolmView):
        raise ModelError("dependence Holm view has the wrong type")
    if (
        value._view_seal is not _REGISTERED_DEPENDENCE_HOLM_VIEW_CAPABILITY
        or _MINTED_DEPENDENCE_HOLM_VIEWS.get(id(value)) is not value
    ):
        raise ModelError("dependence Holm view lacks registered authority")
    if (
        value.contract_version != FOUR_CELL_DEPENDENCE_HOLM_VIEW_CONTRACT
        or value.task_id != "four_cell_dependence_holm"
        or value.cell_order != DEPENDENCE_CELL_ORDER
    ):
        raise ModelError("dependence Holm view contract/order is inconsistent")
    if not value.neighbors_report_only or not value.loyo_report_only:
        raise ModelError("dependence Holm robustness evidence must be report-only")
    _require_sha256(value.run_id, "dependence Holm run ID")
    _require_sha256(value.binding_fingerprint, "dependence Holm binding fingerprint")
    _require_sha256(value.view_fingerprint, "dependence Holm view fingerprint")
    if not isinstance(value.source_views, tuple) or len(value.source_views) != 2:
        raise ModelError("dependence Holm view requires exactly two source views")
    design, production = value.source_views
    _verify_task_view(design)
    _verify_task_view(production)
    if (
        design.task_id != "design_dependence"
        or production.task_id != "production_dependence"
    ):
        raise ModelError("dependence Holm source task roles are inconsistent")
    if design.run_id != value.run_id or production.run_id != value.run_id:
        raise ModelError("dependence Holm source runs are inconsistent")
    source_bindings = (design.binding_fingerprint, production.binding_fingerprint)
    if len(set(source_bindings)) != 2 or value.binding_fingerprint in source_bindings:
        raise ModelError("dependence Holm task bindings are inconsistent")
    if design.strict_canonical != production.strict_canonical or (
        value.strict_canonical != design.strict_canonical
    ):
        raise ModelError("dependence Holm source canonical modes are inconsistent")
    if design.source_execution_fingerprint == production.source_execution_fingerprint:
        raise ModelError("dependence Holm must use distinct role executions")
    expected_view_fingerprints = (
        design.view_fingerprint,
        production.view_fingerprint,
    )
    if value.source_view_fingerprints != expected_view_fingerprints:
        raise ModelError("dependence Holm source fingerprints are inconsistent")
    holm, reasons = _recomputed_task_view_holm(
        value.cells,
        alpha=value.holm.holm.alpha,
        strict_canonical=value.strict_canonical,
    )
    if not _same_dependence_gate(value.holm, holm):
        raise ModelError("dependence Holm result was not recomputed exactly")
    if value.failure_reasons != reasons or value.passed != (not reasons):
        raise ModelError("dependence Holm decision is inconsistent")
    if value.view_fingerprint != _holm_view_fingerprint(value):
        raise ModelError("dependence Holm view fingerprint is inconsistent")


def _verify_four_cell_gate(value: FourCellDependenceGateExecution) -> None:
    if (
        value._execution_seal is not _REGISTERED_DEPENDENCE_CAPABILITY
        or _MINTED_FOUR_CELL_DEPENDENCE_GATES.get(id(value)) is not value
    ):
        raise ModelError("four-cell dependence gate lacks registered authority")
    if (
        value.contract_version != DEPENDENCE_GATE_CONTRACT
        or value.cell_order != DEPENDENCE_CELL_ORDER
        or not value.neighbors_report_only
        or not value.loyo_report_only
        or not value.report_only
        or not isinstance(value.strict_canonical, bool)
    ):
        raise ModelError("four-cell dependence gate contract is inconsistent")
    holm, reasons = _recomputed_holm(
        value.cells,
        alpha=value.holm.holm.alpha,
        strict_canonical=value.strict_canonical,
    )
    if not _same_dependence_gate(value.holm, holm):
        raise ModelError("four-cell dependence Holm evidence is inconsistent")
    if value.failure_reasons != reasons or value.passed != (not reasons):
        raise ModelError("four-cell dependence decision is inconsistent")


def _recomputed_holm(
    cells: object,
    *,
    alpha: float,
    strict_canonical: bool,
) -> tuple[DependenceGate, tuple[str, ...]]:
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if not isinstance(cells, tuple) or len(cells) != 4:
        raise ModelError("dependence gate requires exactly four compact cells")
    if any(not isinstance(cell, DependenceCellEvidence) for cell in cells):
        raise ModelError("dependence gate received an invalid compact cell")
    checked_cells = cast(
        tuple[
            DependenceCellEvidence,
            DependenceCellEvidence,
            DependenceCellEvidence,
            DependenceCellEvidence,
        ],
        cells,
    )
    actual_order = tuple(cell.cell_id for cell in checked_cells)
    if actual_order != DEPENDENCE_CELL_ORDER:
        raise ModelError(
            "dependence cells must use the fixed authoritative order",
            details={"expected": list(DEPENDENCE_CELL_ORDER), "actual": actual_order},
        )
    for cell in checked_cells:
        _verify_registered_cell(cell, strict_canonical=strict_canonical)
    if strict_canonical and alpha != 0.05:
        raise ModelError("canonical dependence Holm alpha must equal 0.05")
    selected_p_values = [cell.selected_result.p_value for cell in checked_cells]
    holm_result = holm_correction(selected_p_values, alpha=alpha)
    holm = DependenceGate(
        cell_order=DEPENDENCE_CELL_ORDER,
        holm=holm_result,
        passed=not bool(holm_result.rejected.any()),
    )
    reasons: list[str] = []
    for cell, rejected in zip(checked_cells, holm_result.rejected, strict=True):
        if not cell.common_draw_reused:
            reasons.append(f"{cell.cell_id}:common_purpose4_draw_not_reused")
        if not cell.fragmentation_passed:
            reasons.append(f"{cell.cell_id}:selected_L_fragmentation_failed")
        if not cell.selected_envelope_passed:
            reasons.append(f"{cell.cell_id}:selected_L_predictive_envelope_failed")
        if bool(rejected):
            reasons.append(f"{cell.cell_id}:holm_rejected")
    return holm, tuple(reasons)


def _task_scoped_cells(
    cells: tuple[DependenceCellEvidence, DependenceCellEvidence],
    *,
    kind: Literal["fragmentation", "dependence"],
) -> (
    tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence]
    | tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence]
):
    if kind == "fragmentation":
        return cast(
            tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence],
            tuple(
                FragmentationTaskCellEvidence(
                    cell_id=cell.cell_id,
                    artifact=cell.artifact,
                    frequency=cell.frequency,
                    histories=cell.histories,
                    selected_length=cell.selected_length,
                    common_draw_reused=cell.common_draw_reused,
                    fragmentation=cell.fragmentation,
                    input_fingerprint=cell.input_fingerprint,
                    model_fingerprint=cell.model_fingerprint,
                    materialization_fingerprint=cell.materialization_fingerprint,
                    source_fingerprint=cell.source_fingerprint,
                    source_evidence_fingerprint=cell.evidence_fingerprint,
                    selected_k=len(cell.fragmentation),
                    starting_length=cell.starting_length,
                    parent_block_core_fingerprint=cell.source_fingerprint,
                    fragmentation_evidence_fingerprint=(cell.evidence_fingerprint),
                )
                for cell in cells
            ),
        )
    return cast(
        tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence],
        tuple(
            DependenceTaskCellEvidence(
                cell_id=cell.cell_id,
                artifact=cell.artifact,
                frequency=cell.frequency,
                histories=cell.histories,
                selected_length=cell.selected_length,
                endpoint_count=cell.endpoint_count,
                common_draw_reused=cell.common_draw_reused,
                selected_result=cell.selected_result,
                loyo_fingerprint=cell.loyo.fingerprint,
                input_fingerprint=cell.input_fingerprint,
                model_fingerprint=cell.model_fingerprint,
                materialization_fingerprint=cell.materialization_fingerprint,
                source_fingerprint=cell.source_fingerprint,
                source_evidence_fingerprint=cell.evidence_fingerprint,
                selected_k=len(cell.fragmentation),
                starting_length=cell.starting_length,
                parent_block_core_fingerprint=cell.source_fingerprint,
                purpose4_materialization_fingerprint=(cell.materialization_fingerprint),
                purpose4_stream_fingerprint=cell.materialization_fingerprint,
                rng_execution_fingerprint=cell.materialization_fingerprint,
                dependence_result_fingerprint=cell.evidence_fingerprint,
            )
            for cell in cells
        ),
    )


def _verify_task_scoped_cells(value: DependenceTaskView) -> None:
    if not isinstance(value.cells, tuple) or len(value.cells) != 2:
        raise ModelError("dependence task view requires exactly two cells")
    expected_order = _ARTIFACT_CELL_ORDER[value.role]
    if tuple(item.cell_id for item in value.cells) != expected_order:
        raise ModelError("dependence task view cell order is inconsistent")
    expected_type: (
        type[FragmentationTaskCellEvidence] | type[DependenceTaskCellEvidence]
    ) = (
        FragmentationTaskCellEvidence
        if value.task_kind == "fragmentation"
        else DependenceTaskCellEvidence
    )
    if any(not isinstance(item, expected_type) for item in value.cells):
        raise ModelError("dependence task view exposes evidence for the wrong task")
    for item in value.cells:
        expected_role = _CELL_ROLE[item.cell_id]
        if (item.artifact, item.frequency) != expected_role:
            raise ModelError("dependence task cell role is inconsistent")
        if (
            isinstance(item.histories, bool)
            or not isinstance(item.histories, int)
            or item.histories <= 0
            or isinstance(item.selected_length, bool)
            or not isinstance(item.selected_length, int)
            or item.selected_length <= 0
            or not isinstance(item.common_draw_reused, bool)
            or isinstance(item.selected_k, bool)
            or not isinstance(item.selected_k, int)
            or item.selected_k <= 0
            or isinstance(item.starting_length, bool)
            or not isinstance(item.starting_length, int)
            or item.starting_length <= 0
        ):
            raise ModelError("dependence task cell geometry is invalid")
        if value.strict_canonical and item.histories != CANONICAL_HISTORY_COUNT:
            raise ModelError("canonical dependence task cell requires 10,000 histories")
        for fingerprint, label in (
            (item.input_fingerprint, "input"),
            (item.model_fingerprint, "model"),
            (item.materialization_fingerprint, "materialization"),
            (item.source_fingerprint, "source"),
            (item.source_evidence_fingerprint, "source evidence"),
            (item.parent_block_core_fingerprint, "parent block core"),
        ):
            _require_sha256(fingerprint, f"dependence task cell {label} fingerprint")
        if isinstance(item, FragmentationTaskCellEvidence):
            _require_sha256(
                item.fragmentation_evidence_fingerprint,
                "fragmentation task evidence fingerprint",
            )
            if not item.fragmentation or tuple(
                part.state for part in item.fragmentation
            ) != tuple(sorted(part.state for part in item.fragmentation)):
                raise ModelError("fragmentation task state order is inconsistent")
            if any(
                part.intended_length != item.selected_length
                for part in item.fragmentation
            ):
                raise ModelError("fragmentation task selected length is inconsistent")
        else:
            for fingerprint, label in (
                (
                    item.purpose4_materialization_fingerprint,
                    "purpose-4 materialization",
                ),
                (item.purpose4_stream_fingerprint, "purpose-4 stream"),
                (item.rng_execution_fingerprint, "RNG execution"),
                (item.dependence_result_fingerprint, "dependence result"),
            ):
                _require_sha256(fingerprint, f"dependence task {label} fingerprint")
            result = item.selected_result
            if (
                item.endpoint_count != (87 if item.frequency == "monthly" else 31)
                or result.decision_role != "selected"
                or result.intended_length != item.selected_length
                or result.repetitions != item.histories
                or not math.isfinite(result.observed_statistic)
                or not math.isfinite(result.critical_value_95)
                or not math.isfinite(result.p_value)
                or not 0.0 <= result.p_value <= 1.0
            ):
                raise ModelError("dependence task selected-L evidence is invalid")
            _require_sha256(item.loyo_fingerprint, "dependence task LOYO fingerprint")
    if value.cells[0].model_fingerprint != value.cells[1].model_fingerprint:
        raise ModelError("dependence task view mixes fitted models")
    if value.cells[0].selected_k != value.cells[1].selected_k:
        raise ModelError("dependence task view mixes selected K")
    if (
        value.cells[0].materialization_fingerprint
        == value.cells[1].materialization_fingerprint
    ):
        raise ModelError("dependence task view reuses one materialization")


def _dependence_task_cells(
    value: DependenceTaskView,
) -> tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence]:
    if value.task_kind != "dependence" or any(
        not isinstance(item, DependenceTaskCellEvidence) for item in value.cells
    ):
        raise ModelError("Holm source is not a dependence task view")
    return cast(
        tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence], value.cells
    )


def _task_view_failure_reasons(
    cells: (
        tuple[FragmentationTaskCellEvidence, FragmentationTaskCellEvidence]
        | tuple[DependenceTaskCellEvidence, DependenceTaskCellEvidence]
    ),
    *,
    kind: Literal["fragmentation", "dependence"],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for cell in cells:
        if not cell.common_draw_reused:
            reasons.append(f"{cell.cell_id}:common_purpose4_draw_not_reused")
        if (
            kind == "fragmentation"
            and isinstance(cell, FragmentationTaskCellEvidence)
            and not cell.fragmentation_passed
        ):
            reasons.append(f"{cell.cell_id}:selected_L_fragmentation_failed")
        if (
            kind == "dependence"
            and isinstance(cell, DependenceTaskCellEvidence)
            and not cell.selected_result.envelope_passed
        ):
            reasons.append(f"{cell.cell_id}:selected_L_predictive_envelope_failed")
    return tuple(reasons)


def _recomputed_task_view_holm(
    cells: object,
    *,
    alpha: float,
    strict_canonical: bool,
) -> tuple[DependenceGate, tuple[str, ...]]:
    if not isinstance(strict_canonical, bool):
        raise ModelError("strict_canonical must be boolean")
    if (
        not isinstance(cells, tuple)
        or len(cells) != 4
        or any(not isinstance(item, DependenceTaskCellEvidence) for item in cells)
    ):
        raise ModelError("dependence Holm requires four selected-L task cells")
    checked = cast(
        tuple[
            DependenceTaskCellEvidence,
            DependenceTaskCellEvidence,
            DependenceTaskCellEvidence,
            DependenceTaskCellEvidence,
        ],
        cells,
    )
    if tuple(item.cell_id for item in checked) != DEPENDENCE_CELL_ORDER:
        raise ModelError("dependence Holm task cells have the wrong exact order")
    if strict_canonical and any(
        item.histories != CANONICAL_HISTORY_COUNT for item in checked
    ):
        raise ModelError("canonical dependence Holm requires 10,000 histories")
    if strict_canonical and alpha != 0.05:
        raise ModelError("canonical dependence Holm alpha must equal 0.05")
    p_values = [item.selected_result.p_value for item in checked]
    holm_result = holm_correction(p_values, alpha=alpha)
    holm = DependenceGate(
        cell_order=DEPENDENCE_CELL_ORDER,
        holm=holm_result,
        passed=not bool(holm_result.rejected.any()),
    )
    reasons: list[str] = []
    for item, rejected in zip(checked, holm_result.rejected, strict=True):
        if not item.common_draw_reused:
            reasons.append(f"{item.cell_id}:common_purpose4_draw_not_reused")
        if not item.selected_result.envelope_passed:
            reasons.append(f"{item.cell_id}:selected_L_predictive_envelope_failed")
        if bool(rejected):
            reasons.append(f"{item.cell_id}:holm_rejected")
    return holm, tuple(reasons)


def _same_dependence_gate(left: DependenceGate, right: DependenceGate) -> bool:
    return bool(
        isinstance(left, DependenceGate)
        and left.cell_order == right.cell_order
        and left.passed == right.passed
        and left.holm.alpha == right.holm.alpha
        and np.array_equal(left.holm.raw_p_values, right.holm.raw_p_values)
        and np.array_equal(left.holm.adjusted_p_values, right.holm.adjusted_p_values)
        and np.array_equal(left.holm.rejected, right.holm.rejected)
    )


def _dependence_input_fingerprint(
    cell_input: DependenceGateCellInput,
    prepared: _PreparedCell,
) -> str:
    parts = {
        "returns": scientific_array_fingerprint(
            prepared.values, role="dependence_historical_returns"
        ),
        "source_states": scientific_array_fingerprint(
            prepared.states, role="dependence_source_states"
        ),
        "continuation_links": scientific_array_fingerprint(
            prepared.links, role="dependence_continuation_links"
        ),
        "calendar_years": scientific_array_fingerprint(
            prepared.years, role="dependence_calendar_years"
        ),
    }
    return _sha256_json(
        {
            "schema_id": "prpg-dependence-cell-input-v1",
            "cell_id": cell_input.cell_id,
            "artifact": cell_input.artifact,
            "frequency": cell_input.frequency,
            "array_fingerprints": parts,
        }
    )


def _dependence_model_fingerprint(model: BlockStateModel | GaussianHMMFit) -> str:
    if isinstance(model, GaussianHMMFit):
        return gaussian_hmm_fit_fingerprint(model)
    if not isinstance(model, BlockStateModel):
        raise ModelError("dependence execution has an invalid fitted model")
    return _sha256_json(
        {
            "schema_id": "prpg-dependence-block-state-model-v1",
            "stationary_distribution": scientific_array_fingerprint(
                model.stationary_distribution,
                role="dependence_stationary_distribution",
            ),
            "transition_matrix": scientific_array_fingerprint(
                model.transition_matrix, role="dependence_transition_matrix"
            ),
        }
    )


def _dependence_materialization_fingerprint(
    value: RegisteredBlockUniformMaterialization,
) -> str:
    if not isinstance(value, RegisteredBlockUniformMaterialization):
        raise ModelError("dependence materialization fingerprint has the wrong type")
    geometry = value.geometry
    return _sha256_json(
        {
            "schema_id": "prpg-dependence-purpose4-materialization-v1",
            "artifact": geometry.artifact,
            "frequency": geometry.frequency,
            "monthly_segment_lengths": list(geometry.monthly_segment_lengths),
            "target_segment_lengths": list(geometry.target_segment_lengths),
            "strict_canonical": geometry.strict_canonical,
            "history_count": value.history_count,
            "allocated_bytes": value.allocated_bytes,
            "monthly_state_uniforms": scientific_array_fingerprint(
                value.matrices.monthly_state_uniforms,
                role="dependence_monthly_state_uniforms",
            ),
            "restart_uniforms": scientific_array_fingerprint(
                value.matrices.restart_uniforms,
                role="dependence_restart_uniforms",
            ),
            "source_rank_uniforms": scientific_array_fingerprint(
                value.matrices.source_rank_uniforms,
                role="dependence_source_rank_uniforms",
            ),
        }
    )


def _dependence_cell_source_fingerprint(
    *,
    cell_id: DependenceCellId,
    artifact: Artifact,
    frequency: Frequency,
    input_fingerprint: str,
    model_fingerprint: str,
    materialization_fingerprint: str,
    strict_canonical: bool,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-dependence-cell-source-v1",
            "cell_id": cell_id,
            "artifact": artifact,
            "frequency": frequency,
            "input_fingerprint": input_fingerprint,
            "model_fingerprint": model_fingerprint,
            "materialization_fingerprint": materialization_fingerprint,
            "strict_canonical": strict_canonical,
        }
    )


def _dependence_cell_evidence_fingerprint(value: DependenceCellEvidence) -> str:
    return _sha256_json(_dependence_cell_identity(value))


def _dependence_cell_identity(value: DependenceCellEvidence) -> dict[str, Any]:
    return {
        "schema_id": "prpg-dependence-cell-evidence-v1",
        "cell_id": value.cell_id,
        "artifact": value.artifact,
        "frequency": value.frequency,
        "histories": value.histories,
        "monthly_segment_lengths": list(value.monthly_segment_lengths),
        "target_segment_lengths": list(value.target_segment_lengths),
        "starting_length": value.starting_length,
        "selected_length": value.selected_length,
        "requested_lengths": list(value.requested_lengths),
        "endpoint_count": value.endpoint_count,
        "endpoint_labels_fingerprint": value.endpoint_labels_fingerprint,
        "stream_namespace": value.stream_namespace,
        "common_draw_reused": value.common_draw_reused,
        "fragmentation": [
            {
                "state": item.state,
                "intended_length": item.intended_length,
                "conditioned_count": item.conditioned_count,
                "conditioned_mean": item.conditioned_mean,
                "conditioned_median": item.conditioned_median,
                "conditioned_singleton_fraction": item.conditioned_singleton_fraction,
                "ideal_count": item.ideal_count,
                "ideal_median": item.ideal_median,
                "ideal_singleton_fraction": item.ideal_singleton_fraction,
                "reasons": list(item.reasons),
            }
            for item in value.fragmentation
        ],
        "length_results": [
            {
                "intended_length": item.intended_length,
                "decision_role": item.decision_role,
                "observed_statistic": item.observed_statistic,
                "critical_value_95": item.critical_value_95,
                "p_value": item.p_value,
                "active_components": item.active_components,
                "repetitions": item.repetitions,
            }
            for item in value.length_results
        ],
        "loyo": _loyo_identity(value.loyo),
        "input_fingerprint": value.input_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "materialization_fingerprint": value.materialization_fingerprint,
        "source_fingerprint": value.source_fingerprint,
        "strict_canonical": value.strict_canonical,
        "report_only": value.report_only,
    }


def _artifact_execution_fingerprint(
    *,
    role: Artifact,
    cells: tuple[DependenceCellEvidence, DependenceCellEvidence],
    strict_canonical: bool,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-artifact-dependence-execution-v1",
            "contract_version": ARTIFACT_DEPENDENCE_EXECUTION_CONTRACT,
            "role": role,
            "cell_order": list(_ARTIFACT_CELL_ORDER[role]),
            "cell_source_fingerprints": [cell.source_fingerprint for cell in cells],
            "cell_evidence_fingerprints": [cell.evidence_fingerprint for cell in cells],
            "strict_canonical": strict_canonical,
            "neighbors_report_only": True,
            "loyo_report_only": True,
            "report_only": True,
        }
    )


def _task_view_fingerprint(value: DependenceTaskView) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-dependence-task-view-v1",
            "contract_version": value.contract_version,
            "task_id": value.task_id,
            "role": value.role,
            "task_kind": value.task_kind,
            "run_id": value.run_id,
            "binding_fingerprint": value.binding_fingerprint,
            "source_execution_fingerprint": value.source_execution_fingerprint,
            "parent_block_core_fingerprints": list(
                value.parent_block_core_fingerprints
            ),
            "fragmentation_parent_view_fingerprint": (
                value.fragmentation_parent_view_fingerprint
            ),
            "cells": [_task_cell_identity(item) for item in value.cells],
            "passed": value.passed,
            "failure_reasons": list(value.failure_reasons),
            "strict_canonical": value.strict_canonical,
            "canonical_stage": value.canonical_stage,
            "report_only": value.report_only,
            "neighbors_report_only": value.neighbors_report_only,
            "loyo_report_only": value.loyo_report_only,
        }
    )


def _task_cell_identity(
    value: FragmentationTaskCellEvidence | DependenceTaskCellEvidence,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "cell_id": value.cell_id,
        "artifact": value.artifact,
        "frequency": value.frequency,
        "histories": value.histories,
        "selected_length": value.selected_length,
        "common_draw_reused": value.common_draw_reused,
        "input_fingerprint": value.input_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "materialization_fingerprint": value.materialization_fingerprint,
        "source_fingerprint": value.source_fingerprint,
        "source_evidence_fingerprint": value.source_evidence_fingerprint,
        "selected_k": value.selected_k,
        "starting_length": value.starting_length,
        "parent_block_core_fingerprint": value.parent_block_core_fingerprint,
    }
    if isinstance(value, FragmentationTaskCellEvidence):
        return {
            **common,
            "task_kind": "fragmentation",
            "fragmentation": [
                {
                    "state": item.state,
                    "intended_length": item.intended_length,
                    "conditioned_count": item.conditioned_count,
                    "conditioned_mean": item.conditioned_mean,
                    "conditioned_median": item.conditioned_median,
                    "conditioned_singleton_fraction": (
                        item.conditioned_singleton_fraction
                    ),
                    "ideal_count": item.ideal_count,
                    "ideal_median": item.ideal_median,
                    "ideal_singleton_fraction": item.ideal_singleton_fraction,
                    "reasons": list(item.reasons),
                }
                for item in value.fragmentation
            ],
            "fragmentation_evidence_fingerprint": (
                value.fragmentation_evidence_fingerprint
            ),
        }
    return {
        **common,
        "task_kind": "dependence",
        "endpoint_count": value.endpoint_count,
        "selected_result": {
            "intended_length": value.selected_result.intended_length,
            "decision_role": value.selected_result.decision_role,
            "observed_statistic": value.selected_result.observed_statistic,
            "critical_value_95": value.selected_result.critical_value_95,
            "p_value": value.selected_result.p_value,
            "active_components": value.selected_result.active_components,
            "repetitions": value.selected_result.repetitions,
        },
        "loyo_fingerprint": value.loyo_fingerprint,
        "purpose4_materialization_fingerprint": (
            value.purpose4_materialization_fingerprint
        ),
        "purpose4_stream_fingerprint": value.purpose4_stream_fingerprint,
        "rng_execution_fingerprint": value.rng_execution_fingerprint,
        "dependence_result_fingerprint": value.dependence_result_fingerprint,
    }


def _staged_dependence_result_fingerprint(
    *,
    selected_length: int,
    requested_lengths: tuple[int, ...],
    endpoint_count: int,
    endpoint_labels_fingerprint: str,
    length_results: tuple[DependenceLengthEvidence, ...],
    loyo: LOYOSelectorRobustness,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-staged-dependence-result-v2",
            "selected_length": selected_length,
            "requested_lengths": list(requested_lengths),
            "endpoint_count": endpoint_count,
            "endpoint_labels_fingerprint": endpoint_labels_fingerprint,
            "length_results": [
                {
                    "intended_length": item.intended_length,
                    "decision_role": item.decision_role,
                    "observed_statistic": item.observed_statistic,
                    "critical_value_95": item.critical_value_95,
                    "p_value": item.p_value,
                    "active_components": item.active_components,
                    "repetitions": item.repetitions,
                }
                for item in length_results
            ],
            "loyo": _loyo_identity(loyo),
        }
    )


def _staged_dependence_source_fingerprint(
    *,
    cell_id: DependenceCellId,
    input_fingerprint: str,
    parent_block_core_fingerprint: str,
    materialization_fingerprint: str,
    stream_fingerprint: str,
    rng_fingerprint: str,
    strict_canonical: bool,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-staged-dependence-source-v2",
            "cell_id": cell_id,
            "input_fingerprint": input_fingerprint,
            "parent_block_core_fingerprint": parent_block_core_fingerprint,
            "purpose4_materialization_fingerprint": materialization_fingerprint,
            "purpose4_stream_fingerprint": stream_fingerprint,
            "rng_execution_fingerprint": rng_fingerprint,
            "strict_canonical": strict_canonical,
        }
    )


def _staged_dependence_cell_identity(
    value: RegisteredDependenceCellExecution,
) -> dict[str, Any]:
    return {
        "schema_id": "prpg-staged-dependence-cell-v2",
        "cell_id": value.cell_id,
        "artifact": value.artifact,
        "frequency": value.frequency,
        "histories": value.histories,
        "starting_length": value.starting_length,
        "selected_length": value.selected_length,
        "requested_lengths": list(value.requested_lengths),
        "endpoint_count": value.endpoint_count,
        "endpoint_labels_fingerprint": value.endpoint_labels_fingerprint,
        "common_draw_reused": value.common_draw_reused,
        "input_fingerprint": value.input_fingerprint,
        "model_fingerprint": value.model_fingerprint,
        "parent_block_core_fingerprint": value.parent_block_core_fingerprint,
        "purpose4_materialization_fingerprint": (
            value.purpose4_materialization_fingerprint
        ),
        "purpose4_stream_fingerprint": value.purpose4_stream_fingerprint,
        "rng_execution_fingerprint": value.rng_execution_fingerprint,
        "dependence_result_fingerprint": value.dependence_result_fingerprint,
        "source_fingerprint": value.source_fingerprint,
        "strict_canonical": value.strict_canonical,
        "neighbors_report_only": value.neighbors_report_only,
        "loyo_report_only": value.loyo_report_only,
    }


def _staged_dependence_cell_fingerprint(
    value: RegisteredDependenceCellExecution,
) -> str:
    return _sha256_json(_staged_dependence_cell_identity(value))


def _staged_artifact_execution_fingerprint(
    *,
    role: Artifact,
    cells: tuple[RegisteredDependenceCellExecution, RegisteredDependenceCellExecution],
    parent_core_fingerprints: tuple[str, str],
    fragmentation_parent_view_fingerprint: str,
    strict_canonical: bool,
) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-staged-artifact-dependence-execution-v2",
            "contract_version": STAGED_DEPENDENCE_EXECUTION_CONTRACT,
            "role": role,
            "cell_order": list(_ARTIFACT_CELL_ORDER[role]),
            "parent_block_core_fingerprints": list(parent_core_fingerprints),
            "fragmentation_parent_view_fingerprint": (
                fragmentation_parent_view_fingerprint
            ),
            "cell_source_fingerprints": [item.source_fingerprint for item in cells],
            "dependence_result_fingerprints": [
                item.dependence_result_fingerprint for item in cells
            ],
            "cell_evidence_fingerprints": [item.evidence_fingerprint for item in cells],
            "strict_canonical": strict_canonical,
            "fragmentation_evidence_excluded": True,
        }
    )


def _holm_view_fingerprint(value: FourCellDependenceHolmView) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-four-cell-dependence-holm-view-v1",
            "contract_version": value.contract_version,
            "task_id": value.task_id,
            "run_id": value.run_id,
            "binding_fingerprint": value.binding_fingerprint,
            "source_view_fingerprints": list(value.source_view_fingerprints),
            "cell_order": list(value.cell_order),
            "holm": {
                "raw_p_values": value.holm.holm.raw_p_values.tolist(),
                "adjusted_p_values": value.holm.holm.adjusted_p_values.tolist(),
                "rejected": value.holm.holm.rejected.tolist(),
                "alpha": value.holm.holm.alpha,
                "passed": value.holm.passed,
            },
            "passed": value.passed,
            "failure_reasons": list(value.failure_reasons),
            "strict_canonical": value.strict_canonical,
            "neighbors_report_only": value.neighbors_report_only,
            "loyo_report_only": value.loyo_report_only,
        }
    )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _sha256_json(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError("dependence identity is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_geometry(
    artifact: Artifact, frequency: Frequency
) -> CalibrationGeometry:
    # Delay the import cycle-free public resolver until validation is needed.
    from prpg.model.block_execution import resolve_calibration_geometry

    return resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=True,
    )


def _with_loyo_fingerprint(
    value: LOYOSelectorRobustness,
) -> LOYOSelectorRobustness:
    identity = _loyo_identity(value)
    fingerprint = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return LOYOSelectorRobustness(
        frequency=value.frequency,
        policy_version=value.policy_version,
        observations=value.observations,
        full_sample_starting_length=value.full_sample_starting_length,
        full_sample_raw_ceiling_length=value.full_sample_raw_ceiling_length,
        full_sample_controlling_series=value.full_sample_controlling_series,
        records=value.records,
        fingerprint=fingerprint,
        report_only=True,
    )


def _verify_loyo(value: LOYOSelectorRobustness) -> None:
    if not isinstance(value, LOYOSelectorRobustness) or not value.report_only:
        raise ModelError(
            "LOYO selector evidence must be immutable report-only evidence"
        )
    if value.policy_version != BLOCK_POLICY_VERSION:
        raise ModelError("LOYO selector policy version is inconsistent")
    if tuple(record.omitted_year for record in value.records) != tuple(
        sorted(record.omitted_year for record in value.records)
    ):
        raise ModelError("LOYO selector years are not in deterministic order")
    if len({record.omitted_year for record in value.records}) != len(value.records):
        raise ModelError("LOYO selector years are duplicated")
    if any(
        record.omitted_observations <= 0
        or record.remaining_observations + record.omitted_observations
        != value.observations
        or (
            record.succeeded
            and (
                record.raw_ceiling_length is None
                or record.failure_type is not None
                or record.failure_message is not None
            )
        )
        or (
            not record.succeeded
            and (
                record.raw_ceiling_length is not None
                or record.controlling_series
                or record.failure_type is None
                or record.failure_message is None
            )
        )
        for record in value.records
    ):
        raise ModelError("LOYO selector record is inconsistent")
    expected = _with_loyo_fingerprint(
        LOYOSelectorRobustness(
            frequency=value.frequency,
            policy_version=value.policy_version,
            observations=value.observations,
            full_sample_starting_length=value.full_sample_starting_length,
            full_sample_raw_ceiling_length=value.full_sample_raw_ceiling_length,
            full_sample_controlling_series=value.full_sample_controlling_series,
            records=value.records,
            fingerprint="",
        )
    ).fingerprint
    if value.fingerprint != expected:
        raise ModelError("LOYO selector fingerprint is inconsistent")


def _loyo_identity(value: LOYOSelectorRobustness) -> dict[str, object]:
    return {
        "frequency": value.frequency,
        "policy_version": value.policy_version,
        "observations": value.observations,
        "full_sample_starting_length": value.full_sample_starting_length,
        "full_sample_raw_ceiling_length": value.full_sample_raw_ceiling_length,
        "full_sample_controlling_series": list(value.full_sample_controlling_series),
        "records": [
            {
                "omitted_year": record.omitted_year,
                "omitted_observations": record.omitted_observations,
                "remaining_observations": record.remaining_observations,
                "starting_length": record.starting_length,
                "raw_ceiling_length": record.raw_ceiling_length,
                "controlling_series": list(record.controlling_series),
                "failure_type": record.failure_type,
                "failure_message": record.failure_message,
            }
            for record in value.records
        ],
        "report_only": True,
    }


def _labels_fingerprint(labels: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(labels).encode("utf-8")).hexdigest()


def _stream_namespace(artifact: Artifact, frequency: Frequency, histories: int) -> str:
    return (
        f"purpose4-v0/{artifact}/{frequency}/"
        "state_chain+restart_uniforms+source_rank_uniforms/"
        f"replicates=0..{histories - 1}"
    )


def _bounded(value: str, *, limit: int = 240) -> str:
    normalized = " ".join(str(value).split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."
