"""Fixed-K accelerator audit and Tier-B bridge execution.

Every fixed-K pseudo-fit uses the closed bridge scope for its DGP/member and
pair ID.  Ordinary execution uses registered random starts 0..49.  Accelerated
execution uses one parent-parameter warm start plus registered random starts
0..8.  The first 100 pairs are evaluated both ways and reduced through the
frozen accelerator audit before either result family may be retained.

The module also aligns prefix/full fits to the parent design coordinate system,
forms every hard-cap-normalized continuous bridge component, and enforces the
fixed 1,000-pair Tier-B look schedule.  It performs no history generation and
no file I/O; planned-look durability belongs to the run-control layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import SimpleNamespace
from typing import ClassVar, Literal, TypeAlias, TypeGuard

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
)
from prpg.model.bridge import (
    AcceleratorResult,
    ActualBridgeGate,
    FixedKBridgeStatistic,
    TierBResult,
    accelerator_eligibility,
    actual_bridge_gate,
    fixed_k_bridge_statistic,
    tier_b_tail_gate,
)
from prpg.model.bridge_history import (
    BRIDGE_FULL_LENGTHS,
    BRIDGE_FULL_ROWS,
    BRIDGE_PREFIX_ROWS,
    BridgeDGP,
    BridgePseudoHistory,
    bridge_fit_input_sha256,
    bridge_parameters_sha256,
    validate_bridge_pseudo_history,
)
from prpg.model.bridge_selection import BridgeMember, bridge_feature_matrix
from prpg.model.hmm import (
    BridgeAcceleratedHMMFit,
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    StatePathDiagnostic,
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_bridge_accelerated,
    fit_gaussian_hmm_restarts,
    summarize_state_path,
    validate_numerical_chain,
)
from prpg.model.input import MACRO_COLUMNS
from prpg.model.materialization_identity import gaussian_hmm_fit_fingerprint
from prpg.model.refit_adequacy import AlignedRefit, align_refit_to_artifact
from prpg.model.regime_policy import owner_fixed_k4_policy_for_scientific_version
from prpg.simulation.rng import ModelFitScope

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
FitMode: TypeAlias = Literal["ordinary_50", "accelerated_10"]
OrdinaryFitFunction: TypeAlias = Callable[
    [HMMFeatureMatrix, int, Sequence[int]], GaussianHMMFit
]
AcceleratedFitFunction: TypeAlias = Callable[
    [HMMFeatureMatrix, int, Sequence[int], HMMParameters],
    BridgeAcceleratedHMMFit,
]

DWELL_SURVIVAL_MONTHS = 12
ACCELERATOR_AUDIT_PAIRS = 100
BRIDGE_KERNEL_LAMBDAS = (0.25, 0.50, 0.75, 1.00)
ACTUAL_BRIDGE_EXECUTION_CONTRACT = "producer-sealed-actual-bridge-execution-v1"
_ACTUAL_BRIDGE_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class BridgeFixedKMemberFit:
    """One prefix/full fixed-K fit or retained scientific failure."""

    dgp: BridgeDGP
    member: BridgeMember
    pair: int
    n_states: int
    scope: ModelFitScope
    mode: FitMode
    random_restart_ids: tuple[int, ...]
    fit: GaussianHMMFit | None
    accelerated: BridgeAcceleratedHMMFit | None
    failure_type: str | None
    failure_message: str | None

    @property
    def valid(self) -> bool:
        return self.fit is not None


@dataclass(frozen=True, slots=True)
class BridgeFixedKComponents:
    """Aligned hard-cap families and their coordinate-level audit."""

    prefix_alignment: AlignedRefit
    full_alignment: AlignedRefit
    prefix_path: StatePathDiagnostic
    full_path: StatePathDiagnostic
    statistic: FixedKBridgeStatistic
    normalized_coordinates: FloatArray
    component_decisions: BoolArray
    coordinate_family: tuple[str, ...]
    coordinate_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BridgeFixedKPairEvaluation:
    """One pseudo-pair under either accelerated or ordinary fitting."""

    dgp: BridgeDGP
    pair: int
    n_states: int
    mode: FitMode
    history_sha256: str
    prefix: BridgeFixedKMemberFit
    full: BridgeFixedKMemberFit
    components: BridgeFixedKComponents | None
    failure_type: str | None
    failure_message: str | None

    @property
    def valid(self) -> bool:
        return self.components is not None

    @property
    def statistic(self) -> float | None:
        return None if self.components is None else self.components.statistic.maximum


@dataclass(frozen=True, slots=True)
class BridgeAcceleratorAudit:
    """First-100 comparison matrices and frozen eligibility result."""

    dgp: BridgeDGP
    n_states: int
    accelerated_valid: BoolArray
    ordinary_valid: BoolArray
    likelihood_difference_per_observation: FloatArray
    maximum_component_difference: FloatArray
    statistic_difference: FloatArray
    component_decisions_identical: BoolArray
    history_sha256: tuple[str, ...]
    result: AcceleratorResult

    @property
    def eligible(self) -> bool:
        return self.result.eligible


@dataclass(frozen=True, slots=True)
class BridgeTierBExecution:
    """One DGP's exact retained pairs and planned-look tail decision."""

    dgp: BridgeDGP
    n_states: int
    accelerator: BridgeAcceleratorAudit
    pairs: tuple[BridgeFixedKPairEvaluation, ...]
    pseudo_statistics: FloatArray
    gate: TierBResult


@dataclass(frozen=True, slots=True)
class ActualBridgeExecutionIdentity:
    """Source-reverified identity for one actual-artifact bridge execution."""

    contract_version: str
    n_states: int
    passed: bool
    design_fit_fingerprint: str
    production_fit_fingerprint: str
    source_materialization_fingerprint: str
    execution_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ActualBridgeSources:
    design_fit: GaussianHMMFit
    production_fit: GaussianHMMFit
    design_monthly_block_length: int
    production_monthly_block_length: int
    design_daily_block_length: int
    production_daily_block_length: int
    design_monthly_state_counts: NDArray[np.int64]
    production_monthly_state_counts: NDArray[np.int64]
    design_daily_state_counts: NDArray[np.int64]
    production_daily_state_counts: NDArray[np.int64]
    design_monthly_kernel_offsets: FloatArray
    production_monthly_kernel_offsets: FloatArray
    design_daily_kernel_offsets: FloatArray
    production_daily_kernel_offsets: FloatArray


@dataclass(frozen=True, slots=True)
class _ActualBridgeComputation:
    n_states: int
    components: BridgeFixedKComponents
    production_scaler_means_in_design_coordinates: FloatArray
    production_scaler_scales_in_design_coordinates: FloatArray
    kernel_lambdas: FloatArray
    monthly_kernel_offset_differences: FloatArray
    daily_kernel_offset_differences: FloatArray
    gate: ActualBridgeGate


@dataclass(frozen=True, slots=True, init=False)
class ActualBridgeExecution:
    """Producer-sealed actual design-to-production bridge and source ledger."""

    generation_authorized: ClassVar[Literal[False]] = False

    n_states: int
    components: BridgeFixedKComponents
    production_scaler_means_in_design_coordinates: FloatArray
    production_scaler_scales_in_design_coordinates: FloatArray
    kernel_lambdas: FloatArray
    monthly_kernel_offset_differences: FloatArray
    daily_kernel_offset_differences: FloatArray
    gate: ActualBridgeGate
    design_fit_fingerprint: str
    production_fit_fingerprint: str
    source_materialization_fingerprint: str
    execution_fingerprint: str
    _sources: _ActualBridgeSources = field(repr=False, compare=False)
    _producer_seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "ActualBridgeExecution is created only by assemble_actual_bridge"
        )


_MINTED_ACTUAL_BRIDGE_EXECUTIONS: dict[
    int,
    tuple[str, int, bool, str, str, str, str],
] = {}


def bridge_fixed_k_scope(
    dgp: BridgeDGP,
    member: BridgeMember,
) -> ModelFitScope:
    """Return the closed fixed-K bridge scope."""

    mapping = {
        ("model", "prefix"): ModelFitScope.BRIDGE_MODEL_FIXED_K_PREFIX,
        ("model", "full"): ModelFitScope.BRIDGE_MODEL_FIXED_K_FULL,
        (
            "nonparametric",
            "prefix",
        ): ModelFitScope.BRIDGE_NONPARAMETRIC_FIXED_K_PREFIX,
        (
            "nonparametric",
            "full",
        ): ModelFitScope.BRIDGE_NONPARAMETRIC_FIXED_K_FULL,
    }
    try:
        return mapping[(dgp, member)]
    except KeyError as error:
        raise ModelError("fixed-K bridge DGP/member is outside the registry") from error


def parent_parameters_in_fit_coordinates(
    parent: HMMParameters,
    scaler_means_in_parent_coordinates: ArrayLike,
    scaler_standard_deviations_in_parent_coordinates: ArrayLike,
) -> HMMParameters:
    """Map parent parameters into a pseudo-member's recomputed z-score space."""

    means = np.asarray(scaler_means_in_parent_coordinates, dtype=np.float64)
    scales = np.asarray(
        scaler_standard_deviations_in_parent_coordinates,
        dtype=np.float64,
    )
    parent_means = np.asarray(parent.means, dtype=np.float64)
    covariance = np.asarray(parent.tied_covariance, dtype=np.float64)
    if (
        parent_means.ndim != 2
        or means.shape != (parent_means.shape[1],)
        or scales.shape != means.shape
        or covariance.shape != (len(means), len(means))
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or bool((scales <= 0.0).any())
    ):
        raise ModelError("bridge parent/scaler coordinates are inconsistent")
    transformed_means = (parent_means - means) / scales
    transformed_covariance = covariance / (scales[:, None] * scales[None, :])
    for array in (transformed_means, transformed_covariance):
        if not np.isfinite(array).all():
            raise ModelError("bridge parent transformation is non-finite")
    return HMMParameters(
        np.asarray(parent.start_probabilities, dtype=np.float64).copy(),
        np.asarray(parent.transition_matrix, dtype=np.float64).copy(),
        transformed_means.astype(np.float64, copy=True),
        transformed_covariance.astype(np.float64, copy=True),
    )


def run_bridge_fixed_k_pair(
    history: BridgePseudoHistory,
    *,
    n_states: int,
    parent_parameters: HMMParameters,
    master_seed: int,
    scientific_version: int,
    mode: FitMode,
    ordinary_fit_function: OrdinaryFitFunction = fit_gaussian_hmm_restarts,
    accelerated_fit_function: AcceleratedFitFunction = (
        fit_gaussian_hmm_bridge_accelerated
    ),
) -> BridgeFixedKPairEvaluation:
    """Fit and align both nested members under one declared fit mode."""

    selected = _selected_k(n_states)
    validate_bridge_pseudo_history(
        history,
        expected_tier="fixed_k",
        expected_n_states=selected,
        expected_master_seed=master_seed,
        expected_scientific_version=scientific_version,
    )
    if bridge_parameters_sha256(parent_parameters) != (
        history.design_parameters_sha256
    ):
        raise ModelError("Tier-B parent parameters do not match history reference")
    history_sha256 = bridge_fit_input_sha256(history)
    prefix = _run_member(
        history.prefix_values,
        history.prefix_continuation,
        dgp=history.dgp,
        member="prefix",
        pair=history.pair,
        n_states=selected,
        parent_parameters=parent_parameters,
        master_seed=master_seed,
        scientific_version=scientific_version,
        mode=mode,
        ordinary_fit_function=ordinary_fit_function,
        accelerated_fit_function=accelerated_fit_function,
    )
    full = _run_member(
        history.values,
        history.continuation,
        dgp=history.dgp,
        member="full",
        pair=history.pair,
        n_states=selected,
        parent_parameters=parent_parameters,
        master_seed=master_seed,
        scientific_version=scientific_version,
        mode=mode,
        ordinary_fit_function=ordinary_fit_function,
        accelerated_fit_function=accelerated_fit_function,
    )
    if not prefix.valid or not full.valid:
        return BridgeFixedKPairEvaluation(
            history.dgp,
            history.pair,
            selected,
            mode,
            history_sha256,
            prefix,
            full,
            None,
            "MemberFitFailure",
            "one or both fixed-K member fits failed",
        )
    if prefix.fit is None or full.fit is None:  # pragma: no cover - property invariant
        raise AssertionError("valid bridge member omitted its fit")
    try:
        components = fixed_k_pair_components(
            parent_parameters,
            prefix.fit,
            full.fit,
        )
    except ModelError as error:
        return BridgeFixedKPairEvaluation(
            history.dgp,
            history.pair,
            selected,
            mode,
            history_sha256,
            prefix,
            full,
            None,
            type(error).__name__,
            _bounded(error.message),
        )
    return BridgeFixedKPairEvaluation(
        history.dgp,
        history.pair,
        selected,
        mode,
        history_sha256,
        prefix,
        full,
        components,
        None,
        None,
    )


def fixed_k_pair_components(
    parent_parameters: HMMParameters,
    prefix_fit: GaussianHMMFit,
    full_fit: GaussianHMMFit,
    *,
    prefix_scaler_means_in_parent_coordinates: ArrayLike | None = None,
    prefix_scaler_scales_in_parent_coordinates: ArrayLike | None = None,
    full_scaler_means_in_parent_coordinates: ArrayLike | None = None,
    full_scaler_scales_in_parent_coordinates: ArrayLike | None = None,
) -> BridgeFixedKComponents:
    """Align two same-K fits and form every continuous hard-cap coordinate."""

    n_states = int(np.asarray(parent_parameters.means).shape[0])
    if prefix_fit.n_states != n_states or full_fit.n_states != n_states:
        raise ModelError("fixed-K bridge member state counts do not match parent K")
    if (
        prefix_fit.n_observations != BRIDGE_PREFIX_ROWS
        or prefix_fit.lengths != (BRIDGE_PREFIX_ROWS,)
        or full_fit.n_observations != BRIDGE_FULL_ROWS
        or full_fit.lengths != BRIDGE_FULL_LENGTHS
    ):
        raise ModelError("fixed-K bridge fits have inconsistent prefix/full geometry")
    prefix_alignment = _alignment(
        parent_parameters,
        prefix_fit,
        prefix_scaler_means_in_parent_coordinates,
        prefix_scaler_scales_in_parent_coordinates,
    )
    full_alignment = _alignment(
        parent_parameters,
        full_fit,
        full_scaler_means_in_parent_coordinates,
        full_scaler_scales_in_parent_coordinates,
    )
    if (
        prefix_alignment.aligned_decoded_states is None
        or full_alignment.aligned_decoded_states is None
    ):
        raise ModelError("fixed-K bridge alignment omitted decoded states")
    prefix_path = summarize_state_path(
        prefix_alignment.aligned_decoded_states,
        n_states,
        lengths=prefix_fit.lengths,
    )
    full_path = summarize_state_path(
        full_alignment.aligned_decoded_states,
        n_states,
        lengths=full_fit.lengths,
    )
    prefix_parameters = prefix_alignment.parameters_in_original_coordinates
    full_parameters = full_alignment.parameters_in_original_coordinates
    centroid = np.linalg.norm(
        full_parameters.means - prefix_parameters.means,
        axis=1,
    ).astype(np.float64)
    transition = np.abs(
        full_parameters.transition_matrix - prefix_parameters.transition_matrix
    )
    prefix_stationary = validate_numerical_chain(
        prefix_parameters.transition_matrix
    ).stationary_distribution
    full_stationary = validate_numerical_chain(
        full_parameters.transition_matrix
    ).stationary_distribution
    occupancy = np.abs(full_stationary - prefix_stationary)
    prefix_median, prefix_survival = _dwell_summary(prefix_path)
    full_median, full_survival = _dwell_summary(full_path)
    if bool((prefix_median <= 0.0).any()):  # pragma: no cover - dwell validation
        raise ModelError("prefix median dwell contains a zero denominator")
    relative_median = np.abs(full_median - prefix_median) / prefix_median
    survival = np.abs(full_survival - prefix_survival)
    statistic = fixed_k_bridge_statistic(
        centroid_shifts=centroid,
        transition_cell_shifts=transition,
        occupancy_shifts=occupancy,
        relative_median_dwell_shifts=relative_median,
        dwell_survival_shifts=survival,
    )
    families = (
        ("centroid", centroid, 0.75),
        ("transition_cell", transition, 0.10),
        ("occupancy", occupancy, 0.10),
        ("relative_median_dwell", relative_median, 0.25),
        ("dwell_survival", survival, 0.10),
    )
    normalized = np.concatenate(
        [
            np.asarray(values, dtype=np.float64).ravel() / cap
            for _, values, cap in families
        ]
    ).astype(np.float64, copy=False)
    labels = tuple(
        name for name, values, _cap in families for _ in range(np.asarray(values).size)
    )
    coordinate_names = (
        tuple(f"centroid/{state}" for state in range(n_states))
        + tuple(
            f"transition_cell/{source}/{target}"
            for source in range(n_states)
            for target in range(n_states)
        )
        + tuple(f"stationary_occupancy/{state}" for state in range(n_states))
        + tuple(f"relative_median_dwell/{state}" for state in range(n_states))
        + tuple(
            f"dwell_survival/{state}/{month}"
            for state in range(n_states)
            for month in range(1, DWELL_SURVIVAL_MONTHS + 1)
        )
    )
    if len(coordinate_names) != len(labels):  # pragma: no cover - fixed construction
        raise AssertionError("fixed-K bridge coordinate names are incomplete")
    decisions = (normalized <= 1.0).astype(np.bool_)
    if not np.isfinite(normalized).all():
        raise ModelError("fixed-K normalized bridge components are non-finite")
    normalized.setflags(write=False)
    decisions.setflags(write=False)
    return BridgeFixedKComponents(
        prefix_alignment,
        full_alignment,
        prefix_path,
        full_path,
        statistic,
        normalized,
        decisions,
        labels,
        coordinate_names,
    )


def audit_bridge_accelerator(
    accelerated: Sequence[BridgeFixedKPairEvaluation],
    ordinary: Sequence[BridgeFixedKPairEvaluation],
    *,
    dgp: BridgeDGP,
    n_states: int,
) -> BridgeAcceleratorAudit:
    """Apply the first-100 two-member ten-versus-fifty-start comparison."""

    selected = _selected_k(n_states)
    fast = tuple(accelerated)
    full = tuple(ordinary)
    if len(fast) != ACCELERATOR_AUDIT_PAIRS or len(full) != ACCELERATOR_AUDIT_PAIRS:
        raise ModelError("accelerator audit requires exactly 100 paired evaluations")
    shape = (ACCELERATOR_AUDIT_PAIRS, 2)
    fast_valid = np.zeros(shape, dtype=np.bool_)
    full_valid = np.zeros(shape, dtype=np.bool_)
    likelihood = np.zeros(shape, dtype=np.float64)
    component = np.zeros(shape, dtype=np.float64)
    statistic = np.zeros(shape, dtype=np.float64)
    decisions = np.ones(shape, dtype=np.bool_)
    history_hashes: list[str] = []
    for pair, (fast_result, full_result) in enumerate(zip(fast, full, strict=True)):
        _audit_pair_identity(fast_result, dgp, selected, pair, "accelerated_10")
        _audit_pair_identity(full_result, dgp, selected, pair, "ordinary_50")
        if fast_result.history_sha256 != full_result.history_sha256:
            raise ModelError("accelerator audit did not reuse identical history bytes")
        history_hashes.append(fast_result.history_sha256)
        for member_index, member in enumerate(("prefix", "full")):
            fast_member = getattr(fast_result, member)
            full_member = getattr(full_result, member)
            fast_valid[pair, member_index] = fast_member.valid
            full_valid[pair, member_index] = full_member.valid
            if fast_member.valid and full_member.valid:
                if fast_member.fit is None or full_member.fit is None:
                    raise AssertionError("valid accelerator member omitted fit")
                if fast_member.fit.n_observations != full_member.fit.n_observations:
                    raise ModelError("accelerator member observation counts differ")
                likelihood[pair, member_index] = (
                    abs(fast_member.fit.log_likelihood - full_member.fit.log_likelihood)
                    / full_member.fit.n_observations
                )
        if fast_result.valid and full_result.valid:
            if fast_result.components is None or full_result.components is None:
                raise AssertionError("valid accelerator pair omitted components")
            first = fast_result.components
            second = full_result.components
            if (
                first.normalized_coordinates.shape
                != second.normalized_coordinates.shape
                or first.coordinate_family != second.coordinate_family
                or first.coordinate_names != second.coordinate_names
            ):
                raise ModelError("accelerator component coordinates differ")
            difference = float(
                np.max(
                    np.abs(first.normalized_coordinates - second.normalized_coordinates)
                )
            )
            statistic_difference = abs(
                first.statistic.maximum - second.statistic.maximum
            )
            identical = bool(
                np.array_equal(first.component_decisions, second.component_decisions)
            )
        else:
            # The member-level validity matrix and any comparable-member
            # likelihood already decide agreement.  Pair-level aligned
            # components do not exist when either member is invalid.
            difference = 0.0
            statistic_difference = 0.0
            identical = True
        component[pair, :] = difference
        statistic[pair, :] = statistic_difference
        decisions[pair, :] = identical
    result = accelerator_eligibility(
        accelerated_valid=fast_valid,
        ordinary_valid=full_valid,
        likelihood_difference_per_observation=likelihood,
        maximum_component_difference=component,
        statistic_difference=statistic,
        component_decisions_identical=decisions,
    )
    for array in (fast_valid, full_valid, likelihood, component, statistic, decisions):
        array.setflags(write=False)
    return BridgeAcceleratorAudit(
        dgp,
        selected,
        fast_valid,
        full_valid,
        likelihood,
        component,
        statistic,
        decisions,
        tuple(history_hashes),
        result,
    )


def retain_audited_bridge_pair(
    accelerated: BridgeFixedKPairEvaluation,
    ordinary: BridgeFixedKPairEvaluation,
    audit: BridgeAcceleratorAudit,
) -> BridgeFixedKPairEvaluation:
    """Retain the preregistered first-100 result family after the audit."""

    _validate_accelerator_audit(audit)
    if accelerated.pair >= ACCELERATOR_AUDIT_PAIRS or ordinary.pair != accelerated.pair:
        raise ModelError("audited pair retention requires a matching first-100 pair")
    _audit_pair_identity(
        accelerated,
        audit.dgp,
        audit.n_states,
        accelerated.pair,
        "accelerated_10",
    )
    _audit_pair_identity(
        ordinary,
        audit.dgp,
        audit.n_states,
        ordinary.pair,
        "ordinary_50",
    )
    if accelerated.history_sha256 != ordinary.history_sha256:
        raise ModelError("audited pair retention requires identical history bytes")
    if audit.history_sha256[accelerated.pair] != accelerated.history_sha256:
        raise ModelError("accelerator audit does not belong to the retained pair")
    return accelerated if audit.eligible else ordinary


def assemble_bridge_tier_b(
    retained_pairs: Sequence[BridgeFixedKPairEvaluation],
    *,
    dgp: BridgeDGP,
    n_states: int,
    actual_statistic: float,
    accelerator: BridgeAcceleratorAudit,
) -> BridgeTierBExecution:
    """Assemble one or more fixed 1,000-pair looks with no post-stop work."""

    selected = _selected_k(n_states)
    _validate_accelerator_audit(accelerator)
    if accelerator.dgp != dgp or accelerator.n_states != selected:
        raise ModelError("Tier-B accelerator identity does not match the DGP/K")
    pairs = tuple(retained_pairs)
    if not pairs or len(pairs) > 10_000 or len(pairs) % 1_000 != 0:
        raise ModelError("Tier B requires fixed 1,000-pair batches through 10,000")
    expected_mode: FitMode = "accelerated_10" if accelerator.eligible else "ordinary_50"
    values = np.empty(len(pairs), dtype=np.float64)
    for pair, result in enumerate(pairs):
        _audit_pair_identity(result, dgp, selected, pair, expected_mode)
        if (
            pair < ACCELERATOR_AUDIT_PAIRS
            and result.history_sha256 != accelerator.history_sha256[pair]
        ):
            raise ModelError(
                "Tier-B first-100 history does not match accelerator audit"
            )
        if not result.valid or result.statistic is None:
            raise ModelError(
                "Tier-B retained pseudo-pair is invalid",
                details={"pair": pair},
            )
        values[pair] = result.statistic
    gate = tier_b_tail_gate(actual_statistic, values)
    if gate.pairs_used != len(pairs):
        raise ModelError("Tier-B work continued beyond its first binding decision")
    values.setflags(write=False)
    return BridgeTierBExecution(
        dgp,
        selected,
        accelerator,
        pairs,
        values,
        gate,
    )


def assemble_actual_bridge(
    *,
    design_fit: GaussianHMMFit,
    production_fit: GaussianHMMFit,
    design_monthly_block_length: int,
    production_monthly_block_length: int,
    design_daily_block_length: int,
    production_daily_block_length: int,
    design_monthly_state_counts: ArrayLike,
    production_monthly_state_counts: ArrayLike,
    design_daily_state_counts: ArrayLike,
    production_daily_state_counts: ArrayLike,
    design_monthly_kernel_offsets: ArrayLike,
    production_monthly_kernel_offsets: ArrayLike,
    design_daily_kernel_offsets: ArrayLike,
    production_daily_kernel_offsets: ArrayLike,
) -> ActualBridgeExecution:
    """Build and producer-seal the exact-source actual-artifact bridge."""

    sources = _retain_actual_bridge_sources(
        design_fit=design_fit,
        production_fit=production_fit,
        design_monthly_block_length=design_monthly_block_length,
        production_monthly_block_length=production_monthly_block_length,
        design_daily_block_length=design_daily_block_length,
        production_daily_block_length=production_daily_block_length,
        design_monthly_state_counts=design_monthly_state_counts,
        production_monthly_state_counts=production_monthly_state_counts,
        design_daily_state_counts=design_daily_state_counts,
        production_daily_state_counts=production_daily_state_counts,
        design_monthly_kernel_offsets=design_monthly_kernel_offsets,
        production_monthly_kernel_offsets=production_monthly_kernel_offsets,
        design_daily_kernel_offsets=design_daily_kernel_offsets,
        production_daily_kernel_offsets=production_daily_kernel_offsets,
    )
    computation = _compute_actual_bridge(sources)
    return _mint_actual_bridge_execution(sources, computation)


def _compute_actual_bridge(
    sources: _ActualBridgeSources,
) -> _ActualBridgeComputation:
    """Recompute the complete actual bridge from its retained exact sources."""

    design_fit = sources.design_fit
    production_fit = sources.production_fit
    design_monthly_block_length = sources.design_monthly_block_length
    production_monthly_block_length = sources.production_monthly_block_length
    design_daily_block_length = sources.design_daily_block_length
    production_daily_block_length = sources.production_daily_block_length
    design_monthly_state_counts = sources.design_monthly_state_counts
    production_monthly_state_counts = sources.production_monthly_state_counts
    design_daily_state_counts = sources.design_daily_state_counts
    production_daily_state_counts = sources.production_daily_state_counts
    design_monthly_kernel_offsets = sources.design_monthly_kernel_offsets
    production_monthly_kernel_offsets = sources.production_monthly_kernel_offsets
    design_daily_kernel_offsets = sources.design_daily_kernel_offsets
    production_daily_kernel_offsets = sources.production_daily_kernel_offsets

    if design_fit.n_states != production_fit.n_states:
        raise ModelError("actual bridge requires production to retain design K")
    n_states = _selected_k(design_fit.n_states)
    if (
        design_fit.n_observations != 193
        or design_fit.lengths != (193,)
        or production_fit.n_observations != 217
        or production_fit.lengths != (210, 7)
    ):
        raise ModelError("actual bridge HMM geometry is not design/production exact")
    if (
        design_fit.scaler_scope != "design_training"
        or production_fit.scaler_scope != "production_full"
        or tuple(design_fit.feature_names) != tuple(MACRO_COLUMNS)
        or tuple(production_fit.feature_names) != tuple(MACRO_COLUMNS)
    ):
        raise ModelError("actual bridge HMM role/feature metadata is inconsistent")
    design_means = _scaler(design_fit.scaler_means, "design scaler means")
    design_scales = _positive_scaler(
        design_fit.scaler_standard_deviations,
        "design scaler standard deviations",
    )
    production_means = _scaler(
        production_fit.scaler_means,
        "production scaler means",
    )
    production_scales = _positive_scaler(
        production_fit.scaler_standard_deviations,
        "production scaler standard deviations",
    )
    production_means_in_design = (production_means - design_means) / design_scales
    production_scales_in_design = production_scales / design_scales
    production_means_in_design = production_means_in_design.astype(
        np.float64, copy=True
    )
    production_scales_in_design = production_scales_in_design.astype(
        np.float64, copy=True
    )
    production_means_in_design.setflags(write=False)
    production_scales_in_design.setflags(write=False)
    components = fixed_k_pair_components(
        design_fit.parameters,
        design_fit,
        production_fit,
        prefix_scaler_means_in_parent_coordinates=np.zeros(4),
        prefix_scaler_scales_in_parent_coordinates=np.ones(4),
        full_scaler_means_in_parent_coordinates=production_means_in_design,
        full_scaler_scales_in_parent_coordinates=production_scales_in_design,
    )
    expected_order = np.arange(n_states, dtype=np.int64)
    same_order = bool(
        np.array_equal(
            components.prefix_alignment.reference_to_candidate,
            expected_order,
        )
        and np.array_equal(
            components.full_alignment.reference_to_candidate,
            expected_order,
        )
    )
    design_monthly_kernel = _kernel_offsets(
        design_monthly_kernel_offsets,
        n_states,
        "design monthly kernel offsets",
    )
    production_monthly_kernel = _kernel_offsets(
        production_monthly_kernel_offsets,
        n_states,
        "production monthly kernel offsets",
    )
    design_daily_kernel = _kernel_offsets(
        design_daily_kernel_offsets,
        n_states,
        "design daily kernel offsets",
    )
    production_daily_kernel = _kernel_offsets(
        production_daily_kernel_offsets,
        n_states,
        "production daily kernel offsets",
    )
    design_monthly_counts = _exact_counts(
        design_monthly_state_counts,
        n_states,
        193,
        "design monthly state counts",
    )
    production_monthly_counts = _exact_counts(
        production_monthly_state_counts,
        n_states,
        217,
        "production monthly state counts",
    )
    _require_decoded_count_identity(
        design_monthly_counts,
        design_fit.decoded_states,
        n_states,
        "design monthly state counts",
    )
    _require_decoded_count_identity(
        production_monthly_counts,
        production_fit.decoded_states,
        n_states,
        "production monthly state counts",
    )
    design_daily_counts = _exact_counts(
        design_daily_state_counts,
        n_states,
        4_049,
        "design daily state counts",
    )
    production_daily_counts = _exact_counts(
        production_daily_state_counts,
        n_states,
        4_547,
        "production daily state counts",
    )
    reference_to_production = components.full_alignment.reference_to_candidate
    aligned_production_monthly_counts = production_monthly_counts[
        reference_to_production
    ]
    aligned_production_daily_counts = production_daily_counts[reference_to_production]
    aligned_production_monthly_kernel = production_monthly_kernel[
        :, reference_to_production, :
    ]
    aligned_production_daily_kernel = production_daily_kernel[
        :, reference_to_production, :
    ]
    monthly_kernel_differences = np.asarray(
        aligned_production_monthly_kernel - design_monthly_kernel,
        dtype=np.float64,
    )
    daily_kernel_differences = np.asarray(
        aligned_production_daily_kernel - design_daily_kernel,
        dtype=np.float64,
    )
    kernel_lambdas = np.asarray(BRIDGE_KERNEL_LAMBDAS, dtype=np.float64)
    for array in (
        monthly_kernel_differences,
        daily_kernel_differences,
        kernel_lambdas,
    ):
        array.setflags(write=False)
    gate = actual_bridge_gate(
        design_n_states=n_states,
        production_n_states=n_states,
        same_aligned_order=same_order,
        fixed_k=components.statistic,
        design_monthly_block_length=design_monthly_block_length,
        production_monthly_block_length=production_monthly_block_length,
        design_daily_block_length=design_daily_block_length,
        production_daily_block_length=production_daily_block_length,
        design_monthly_state_counts=design_monthly_counts,
        production_monthly_state_counts=aligned_production_monthly_counts,
        design_daily_state_counts=design_daily_counts,
        production_daily_state_counts=aligned_production_daily_counts,
        monthly_kernel_offset_differences=monthly_kernel_differences,
        daily_kernel_offset_differences=daily_kernel_differences,
    )
    return _ActualBridgeComputation(
        n_states,
        components,
        production_means_in_design,
        production_scales_in_design,
        kernel_lambdas,
        monthly_kernel_differences,
        daily_kernel_differences,
        gate,
    )


def actual_bridge_execution_identity(
    value: ActualBridgeExecution,
) -> ActualBridgeExecutionIdentity:
    """Recompute one producer-minted actual bridge from every retained source."""

    if (
        not isinstance(value, ActualBridgeExecution)
        or value._producer_seal is not _ACTUAL_BRIDGE_CAPABILITY
    ):
        raise ModelError("actual bridge execution lacks producer authority")
    registered = _MINTED_ACTUAL_BRIDGE_EXECUTIONS.get(id(value))
    if registered is None:
        raise ModelError("actual bridge execution is not object-specifically minted")
    if not isinstance(value._sources, _ActualBridgeSources):
        raise ModelError("actual bridge execution lost its retained sources")
    design_fingerprint = gaussian_hmm_fit_fingerprint(value._sources.design_fit)
    production_fingerprint = gaussian_hmm_fit_fingerprint(value._sources.production_fit)
    source_fingerprint = _actual_bridge_source_fingerprint(value._sources)
    recomputed = _compute_actual_bridge(value._sources)
    expected_execution = _actual_bridge_output_fingerprint(
        recomputed,
        design_fit_fingerprint=design_fingerprint,
        production_fit_fingerprint=production_fingerprint,
        source_materialization_fingerprint=source_fingerprint,
    )
    live_execution = _actual_bridge_output_fingerprint(
        value,
        design_fit_fingerprint=design_fingerprint,
        production_fit_fingerprint=production_fingerprint,
        source_materialization_fingerprint=source_fingerprint,
    )
    expected_registered = (
        ACTUAL_BRIDGE_EXECUTION_CONTRACT,
        recomputed.n_states,
        recomputed.gate.passed,
        design_fingerprint,
        production_fingerprint,
        source_fingerprint,
        expected_execution,
    )
    if (
        registered != expected_registered
        or value.n_states != recomputed.n_states
        or value.gate.passed != recomputed.gate.passed
        or value.design_fit_fingerprint != design_fingerprint
        or value.production_fit_fingerprint != production_fingerprint
        or value.source_materialization_fingerprint != source_fingerprint
        or value.execution_fingerprint != expected_execution
        or live_execution != expected_execution
    ):
        raise ModelError("actual bridge execution differs from its exact sources")
    return ActualBridgeExecutionIdentity(*expected_registered)


def is_actual_bridge_execution(
    value: object,
) -> TypeGuard[ActualBridgeExecution]:
    """Return whether an actual bridge remains producer-sealed and source-exact."""

    if not isinstance(value, ActualBridgeExecution):
        return False
    try:
        actual_bridge_execution_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def _retain_actual_bridge_sources(
    *,
    design_fit: GaussianHMMFit,
    production_fit: GaussianHMMFit,
    design_monthly_block_length: int,
    production_monthly_block_length: int,
    design_daily_block_length: int,
    production_daily_block_length: int,
    design_monthly_state_counts: ArrayLike,
    production_monthly_state_counts: ArrayLike,
    design_daily_state_counts: ArrayLike,
    production_daily_state_counts: ArrayLike,
    design_monthly_kernel_offsets: ArrayLike,
    production_monthly_kernel_offsets: ArrayLike,
    design_daily_kernel_offsets: ArrayLike,
    production_daily_kernel_offsets: ArrayLike,
) -> _ActualBridgeSources:
    if not isinstance(design_fit, GaussianHMMFit) or not isinstance(
        production_fit, GaussianHMMFit
    ):
        raise ModelError("actual bridge requires exact GaussianHMMFit parents")
    return _ActualBridgeSources(
        design_fit,
        production_fit,
        _source_block_length(
            design_monthly_block_length,
            "design monthly block length",
            MONTHLY_BLOCK_LENGTH_BOUNDS,
        ),
        _source_block_length(
            production_monthly_block_length,
            "production monthly block length",
            MONTHLY_BLOCK_LENGTH_BOUNDS,
        ),
        _source_block_length(
            design_daily_block_length,
            "design daily block length",
            DAILY_BLOCK_LENGTH_BOUNDS,
        ),
        _source_block_length(
            production_daily_block_length,
            "production daily block length",
            DAILY_BLOCK_LENGTH_BOUNDS,
        ),
        _immutable_source_counts(
            design_monthly_state_counts,
            "design monthly state counts",
        ),
        _immutable_source_counts(
            production_monthly_state_counts,
            "production monthly state counts",
        ),
        _immutable_source_counts(
            design_daily_state_counts,
            "design daily state counts",
        ),
        _immutable_source_counts(
            production_daily_state_counts,
            "production daily state counts",
        ),
        _immutable_source_floats(
            design_monthly_kernel_offsets,
            "design monthly kernel offsets",
        ),
        _immutable_source_floats(
            production_monthly_kernel_offsets,
            "production monthly kernel offsets",
        ),
        _immutable_source_floats(
            design_daily_kernel_offsets,
            "design daily kernel offsets",
        ),
        _immutable_source_floats(
            production_daily_kernel_offsets,
            "production daily kernel offsets",
        ),
    )


def _mint_actual_bridge_execution(
    sources: _ActualBridgeSources,
    computation: _ActualBridgeComputation,
) -> ActualBridgeExecution:
    design_fingerprint = gaussian_hmm_fit_fingerprint(sources.design_fit)
    production_fingerprint = gaussian_hmm_fit_fingerprint(sources.production_fit)
    source_fingerprint = _actual_bridge_source_fingerprint(sources)
    execution_fingerprint = _actual_bridge_output_fingerprint(
        computation,
        design_fit_fingerprint=design_fingerprint,
        production_fit_fingerprint=production_fingerprint,
        source_materialization_fingerprint=source_fingerprint,
    )
    result = object.__new__(ActualBridgeExecution)
    for name, item in (
        ("n_states", computation.n_states),
        ("components", computation.components),
        (
            "production_scaler_means_in_design_coordinates",
            computation.production_scaler_means_in_design_coordinates,
        ),
        (
            "production_scaler_scales_in_design_coordinates",
            computation.production_scaler_scales_in_design_coordinates,
        ),
        ("kernel_lambdas", computation.kernel_lambdas),
        (
            "monthly_kernel_offset_differences",
            computation.monthly_kernel_offset_differences,
        ),
        (
            "daily_kernel_offset_differences",
            computation.daily_kernel_offset_differences,
        ),
        ("gate", computation.gate),
        ("design_fit_fingerprint", design_fingerprint),
        ("production_fit_fingerprint", production_fingerprint),
        ("source_materialization_fingerprint", source_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("_sources", sources),
        ("_producer_seal", _ACTUAL_BRIDGE_CAPABILITY),
    ):
        object.__setattr__(result, name, item)
    _MINTED_ACTUAL_BRIDGE_EXECUTIONS[id(result)] = (
        ACTUAL_BRIDGE_EXECUTION_CONTRACT,
        computation.n_states,
        computation.gate.passed,
        design_fingerprint,
        production_fingerprint,
        source_fingerprint,
        execution_fingerprint,
    )
    return result


def _actual_bridge_source_fingerprint(sources: _ActualBridgeSources) -> str:
    return _identity_fingerprint(
        {
            "schema_id": "prpg-actual-bridge-exact-source-v1",
            "contract_version": ACTUAL_BRIDGE_EXECUTION_CONTRACT,
            "design_fit_materialization_fingerprint": (
                gaussian_hmm_fit_fingerprint(sources.design_fit)
            ),
            "production_fit_materialization_fingerprint": (
                gaussian_hmm_fit_fingerprint(sources.production_fit)
            ),
            "retained_sources": sources,
        }
    )


def _actual_bridge_output_fingerprint(
    value: ActualBridgeExecution | _ActualBridgeComputation,
    *,
    design_fit_fingerprint: str,
    production_fit_fingerprint: str,
    source_materialization_fingerprint: str,
) -> str:
    return _identity_fingerprint(
        {
            "schema_id": "prpg-actual-bridge-execution-v1",
            "contract_version": ACTUAL_BRIDGE_EXECUTION_CONTRACT,
            "design_fit_fingerprint": design_fit_fingerprint,
            "production_fit_fingerprint": production_fit_fingerprint,
            "source_materialization_fingerprint": (source_materialization_fingerprint),
            "n_states": value.n_states,
            "components": value.components,
            "production_scaler_means_in_design_coordinates": (
                value.production_scaler_means_in_design_coordinates
            ),
            "production_scaler_scales_in_design_coordinates": (
                value.production_scaler_scales_in_design_coordinates
            ),
            "kernel_lambdas": value.kernel_lambdas,
            "monthly_kernel_offset_differences": (
                value.monthly_kernel_offset_differences
            ),
            "daily_kernel_offset_differences": (value.daily_kernel_offset_differences),
            "gate": value.gate,
        }
    )


def _identity_fingerprint(value: object) -> str:
    encoded = json.dumps(
        _identity_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _identity_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            canonical = np.ascontiguousarray(value, dtype=np.dtype("u1"))
        elif value.dtype.kind == "i":
            canonical = np.ascontiguousarray(value, dtype=np.dtype("<i8"))
        elif value.dtype.kind == "u":
            canonical = np.ascontiguousarray(value, dtype=np.dtype("<u8"))
        elif value.dtype.kind == "f":
            canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
        else:
            raise ModelError("actual bridge identity contains an unsupported array")
        return {
            "array_dtype": canonical.dtype.str,
            "array_shape": list(canonical.shape),
            "array_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
            "writeable": bool(value.flags.writeable),
        }
    if isinstance(value, np.generic):
        return _identity_value(value.item())
    if isinstance(value, Enum):
        return _identity_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "record_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _identity_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, SimpleNamespace):
        return {
            "record_type": "types.SimpleNamespace",
            "fields": {
                key: _identity_value(item) for key, item in sorted(vars(value).items())
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ModelError("actual bridge identity mappings require string keys")
        return {key: _identity_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple | list):
        return [_identity_value(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return {"float64_hex": value.hex()}
    raise ModelError(
        "actual bridge identity contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _source_block_length(
    value: int,
    label: str,
    bounds: tuple[int, int],
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not bounds[0] <= value <= bounds[1]
    ):
        raise ModelError(
            f"{label} is outside the approved bounds",
            details={
                "lower_bound": bounds[0],
                "upper_bound": bounds[1],
                "value": value,
            },
        )
    return value


def _immutable_source_counts(values: ArrayLike, label: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u"}:
        raise ModelError(f"{label} must contain integers")
    result = np.ascontiguousarray(raw, dtype=np.dtype("<i8")).copy()
    result.setflags(write=False)
    return result


def _immutable_source_floats(values: ArrayLike, label: str) -> FloatArray:
    raw = np.asarray(values, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ModelError(f"{label} must contain finite values")
    result = np.ascontiguousarray(raw, dtype=np.dtype("<f8")).copy()
    result.setflags(write=False)
    return result


def _run_member(
    values: ArrayLike,
    continuation: ArrayLike,
    *,
    dgp: BridgeDGP,
    member: BridgeMember,
    pair: int,
    n_states: int,
    parent_parameters: HMMParameters,
    master_seed: int,
    scientific_version: int,
    mode: FitMode,
    ordinary_fit_function: OrdinaryFitFunction,
    accelerated_fit_function: AcceleratedFitFunction,
) -> BridgeFixedKMemberFit:
    scope = bridge_fixed_k_scope(dgp, member)
    if mode not in {"ordinary_50", "accelerated_10"}:
        raise ModelError("fixed-K bridge fit mode is invalid")
    features = bridge_feature_matrix(values, continuation)
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=scope,
        replicate=pair,
        n_states=n_states,
    )
    try:
        policy = owner_fixed_k4_policy_for_scientific_version(scientific_version)
        if mode == "ordinary_50":
            if (
                ordinary_fit_function is fit_gaussian_hmm_restarts
                and policy is not None
            ):
                fit = fit_gaussian_hmm_restarts(
                    features,
                    n_states,
                    seeds,
                    owner_fixed_k4_policy=policy,
                )
            else:
                fit = ordinary_fit_function(features, n_states, seeds)
            accelerated = None
            restart_ids = tuple(range(50))
        else:
            parent = parent_parameters_in_fit_coordinates(
                parent_parameters,
                features.scaler_means,
                features.scaler_standard_deviations,
            )
            if (
                accelerated_fit_function is fit_gaussian_hmm_bridge_accelerated
                and policy is not None
            ):
                accelerated = fit_gaussian_hmm_bridge_accelerated(
                    features,
                    n_states,
                    seeds[:9],
                    parent,
                    owner_fixed_k4_policy=policy,
                )
            else:
                accelerated = accelerated_fit_function(
                    features,
                    n_states,
                    seeds[:9],
                    parent,
                )
            if not isinstance(accelerated, BridgeAcceleratedHMMFit):
                raise ModelError("accelerated fit returned the wrong record type")
            _validate_accelerated_fit_record(accelerated, seeds)
            fit = accelerated.fit
            restart_ids = accelerated.random_restart_ids
        if (
            fit.n_states != n_states
            or fit.n_observations != len(features.values)
            or fit.lengths != features.lengths
        ):
            raise ModelError("fixed-K bridge fit returned inconsistent geometry")
        return BridgeFixedKMemberFit(
            dgp,
            member,
            pair,
            n_states,
            scope,
            mode,
            restart_ids,
            fit,
            accelerated,
            None,
            None,
        )
    except ModelError as error:
        return BridgeFixedKMemberFit(
            dgp,
            member,
            pair,
            n_states,
            scope,
            mode,
            tuple(range(50 if mode == "ordinary_50" else 9)),
            None,
            None,
            type(error).__name__,
            _bounded(error.message),
        )


def _alignment(
    parent: HMMParameters,
    fit: GaussianHMMFit,
    scaler_means: ArrayLike | None,
    scaler_scales: ArrayLike | None,
) -> AlignedRefit:
    if (scaler_means is None) != (scaler_scales is None):
        raise ModelError("bridge scaler-coordinate overrides must be paired")
    means = fit.scaler_means if scaler_means is None else scaler_means
    scales = fit.scaler_standard_deviations if scaler_scales is None else scaler_scales
    return align_refit_to_artifact(
        parent,
        fit.parameters,
        means,
        scales,
        decoded_states=fit.decoded_states,
    )


def _dwell_summary(path: StatePathDiagnostic) -> tuple[FloatArray, FloatArray]:
    n_states = len(path.dwell_lengths_by_state)
    median = np.empty(n_states, dtype=np.float64)
    survival = np.empty((n_states, DWELL_SURVIVAL_MONTHS), dtype=np.float64)
    for state, values in enumerate(path.dwell_lengths_by_state):
        dwell = np.asarray(values)
        if (
            dwell.ndim != 1
            or dwell.dtype.kind not in {"i", "u"}
            or len(dwell) == 0
            or bool((dwell <= 0).any())
        ):
            raise ModelError(
                "fixed-K bridge requires completed dwell support in every state",
                details={"state": state},
            )
        median[state] = float(np.quantile(dwell, 0.5, method="higher"))
        for month in range(1, DWELL_SURVIVAL_MONTHS + 1):
            survival[state, month - 1] = np.count_nonzero(dwell >= month) / len(dwell)
    median.setflags(write=False)
    survival.setflags(write=False)
    return median, survival


def _scaler(values: ArrayLike, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ModelError(f"{label} must contain four finite values")
    return result


def _positive_scaler(values: ArrayLike, label: str) -> FloatArray:
    result = _scaler(values, label)
    if bool((result <= 0.0).any()):
        raise ModelError(f"{label} must be positive")
    return result


def _kernel_offsets(values: ArrayLike, n_states: int, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (4, n_states, 3) or not np.isfinite(result).all():
        raise ModelError(f"{label} must have finite exact shape (4,K,3)")
    return result


def _exact_counts(
    values: ArrayLike,
    n_states: int,
    total: int,
    label: str,
) -> FloatArray:
    raw = np.asarray(values)
    if (
        raw.shape != (n_states,)
        or raw.dtype.kind not in {"i", "u"}
        or bool((raw <= 0).any())
        or int(raw.sum()) != total
    ):
        raise ModelError(
            f"{label} must be positive integer K-vector summing to {total}"
        )
    return raw.astype(np.float64)


def _require_decoded_count_identity(
    supplied: FloatArray,
    decoded_states: ArrayLike,
    n_states: int,
    label: str,
) -> None:
    decoded = np.asarray(decoded_states)
    if (
        decoded.ndim != 1
        or decoded.dtype.kind not in {"i", "u"}
        or bool(((decoded < 0) | (decoded >= n_states)).any())
    ):
        raise ModelError(f"{label} cannot be verified against decoded states")
    expected = np.bincount(decoded.astype(np.int64), minlength=n_states)
    if not np.array_equal(supplied, expected.astype(np.float64)):
        raise ModelError(f"{label} do not match the fitted decoded path")


def _audit_pair_identity(
    result: BridgeFixedKPairEvaluation,
    dgp: BridgeDGP,
    n_states: int,
    pair: int,
    mode: FitMode,
) -> None:
    if (
        not isinstance(result, BridgeFixedKPairEvaluation)
        or result.dgp != dgp
        or result.n_states != n_states
        or result.pair != pair
        or result.mode != mode
        or not _is_sha256(result.history_sha256)
    ):
        raise ModelError("fixed-K bridge pair identity/order is inconsistent")
    member_names: tuple[BridgeMember, ...] = ("prefix", "full")
    for expected_member in member_names:
        member_fit = result.prefix if expected_member == "prefix" else result.full
        if (
            not isinstance(member_fit, BridgeFixedKMemberFit)
            or member_fit.dgp != dgp
            or member_fit.member != expected_member
            or member_fit.pair != pair
            or member_fit.n_states != n_states
            or member_fit.scope is not bridge_fixed_k_scope(dgp, expected_member)
            or member_fit.mode != mode
        ):
            raise ModelError("fixed-K bridge member identity/order is inconsistent")
    if result.components is not None and (
        not result.prefix.valid or not result.full.valid
    ):
        raise ModelError("fixed-K components cannot exist after a member failure")


def _selected_k(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in {2, 3, 4, 5}
    ):
        raise ModelError("fixed-K bridge state count must be one of 2,3,4,5")
    return value


def _bounded(value: str, limit: int = 500) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _validate_accelerated_fit_record(
    result: BridgeAcceleratedHMMFit,
    registered_seeds: Sequence[int],
) -> None:
    expected_best = (
        "parent_warm"
        if result.fit.best_restart_index == 0
        else f"registered_random_{result.fit.best_restart_index - 1}"
    )
    if (
        result.parent_warm_diagnostic_index != 0
        or result.random_restart_ids != tuple(range(9))
        or result.random_diagnostic_indices != tuple(range(1, 10))
        or result.warm_random_state_placeholder_seed != int(registered_seeds[0])
        or result.fit.best_restart_index not in range(10)
        or result.best_initialization != expected_best
    ):
        raise ModelError(
            "accelerated bridge fit initialization mapping is inconsistent"
        )


def _validate_accelerator_audit(audit: BridgeAcceleratorAudit) -> None:
    if not isinstance(audit, BridgeAcceleratorAudit):
        raise ModelError("Tier-B accelerator audit has the wrong record type")
    if len(audit.history_sha256) != ACCELERATOR_AUDIT_PAIRS or not all(
        _is_sha256(value) for value in audit.history_sha256
    ):
        raise ModelError("Tier-B accelerator audit history identities are incomplete")
    recomputed = accelerator_eligibility(
        accelerated_valid=audit.accelerated_valid,
        ordinary_valid=audit.ordinary_valid,
        likelihood_difference_per_observation=(
            audit.likelihood_difference_per_observation
        ),
        maximum_component_difference=audit.maximum_component_difference,
        statistic_difference=audit.statistic_difference,
        component_decisions_identical=audit.component_decisions_identical,
    )
    if (
        audit.result.agreeing_pairs != recomputed.agreeing_pairs
        or audit.result.audited_pairs != recomputed.audited_pairs
        or audit.result.eligible != recomputed.eligible
        or not np.array_equal(
            audit.result.pair_agreement,
            recomputed.pair_agreement,
        )
    ):
        raise ModelError("Tier-B accelerator decision is inconsistent with its audit")


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
