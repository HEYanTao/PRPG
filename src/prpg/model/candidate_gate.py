"""Fail-closed orchestration for the version-1 HMM candidate gate.

This module is the boundary between the expensive candidate-specific evidence
builders and :mod:`prpg.model.selection`.  It deliberately does not fit a
model, generate a bootstrap trace, or run a selected-only adequacy/holdout
gate.  Its responsibilities are narrower and auditable:

* retain exactly one row for every ordinary candidate ``K=2,3,4,5``;
* translate decoded return pools and finalized block policies into the exact
  pre-BIC state-support thresholds;
* retain failed fits and missing downstream evidence as rejections rather
  than silently dropping their K;
* pass exactly 72 fold-major scores and caller-materialized common bootstrap
  indices to the frozen selector; and
* keep selected-only vetoes separate so a failed selected model can never
  promote a runner-up.

The functions are pure and perform no file I/O.  Canonical staged selection
accepts only the producer-sealed registered bootstrap materialization.  The
one-shot compatibility entry point may accept a raw precomputed array, but
that result is report-only and cannot mint a canonical task view.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Literal, TypeAlias, TypeGuard

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.block_length import (
    DAILY_BLOCK_LENGTH_BOUNDS,
    MONTHLY_BLOCK_LENGTH_BOUNDS,
    BlockPolicySelection,
    LengthTrial,
)
from prpg.model.calibration import (
    ArtifactRole,
    CandidateFitAttempt,
    CandidateFitSet,
    DecodedReturnPools,
    MacroStabilityEvidence,
    RegisteredRollingBootstrapMaterialization,
    RollingCandidateEvidence,
    is_registered_rolling_bootstrap_materialization,
)
from prpg.model.calibration_identity import (
    candidate_fit_set_fingerprint,
    decoded_return_pools_fingerprint,
    macro_stability_evidence_fingerprint,
    rolling_candidate_evidence_fingerprint,
)
from prpg.model.hmm import GaussianHMMFit, StatePathDiagnostic
from prpg.model.materialization_identity import (
    scientific_array_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.model.regime_policy import (
    OWNER_FIXED_K4_MINIMUM_OBSERVED_EPISODES,
    effective_identification_passed,
    effective_transition_support_passed,
    owner_fixed_k4_recurring_history_support,
)
from prpg.model.selection import (
    ORDINARY_SELECTION_POLICY,
    OWNER_FIXED_K4_SELECTION_POLICY,
    ROLLING_BOOTSTRAP_REPLICATES,
    ROLLING_FOLD_COUNT,
    ROLLING_FOLD_SIZE,
    ROLLING_SCORE_COUNT,
    CandidateSelectionInput,
    CandidateSelectionReport,
    CandidateViabilityEvidence,
    CandidateViabilityRecord,
    PreselectionCheck,
    StateViabilityEvidence,
    ViabilityThresholds,
    evaluate_candidate_viability,
    select_candidate,
    select_owner_fixed_k4,
)
from prpg.simulation.rng import CalibrationArtifact

IntArray: TypeAlias = NDArray[np.int64]
CANONICAL_CANDIDATES = (2, 3, 4, 5)

MINIMUM_MONTHLY_OBSERVATIONS = 24
MONTHLY_BLOCK_SUPPORT_MULTIPLE = 5
MINIMUM_STATE_OCCUPANCY = 0.10
MINIMUM_COMPLETED_MONTHLY_EPISODES = 3
MINIMUM_DAILY_OBSERVATIONS = 252
DAILY_BLOCK_SUPPORT_MULTIPLE = 5
MINIMUM_DAILY_RUNS = 3
OWNER_FIXED_N_STATES = 4
OWNER_FIXED_MINIMUM_OBSERVATIONS = 1
OWNER_FIXED_MINIMUM_OCCUPANCY = 0.0
# Backward-compatible public spelling used by the bridge/preflight layers.
# The policy module is the single semantic owner of this threshold.
OWNER_FIXED_MINIMUM_OBSERVED_EPISODES = OWNER_FIXED_K4_MINIMUM_OBSERVED_EPISODES
MINIMUM_BLOCK_COMPLETED_EPISODES = 0
MAXIMUM_TIED_COVARIANCE_CONDITION = 1_000_000.0
MINIMUM_MACRO_BOOTSTRAP_MEDIAN_ARI = 0.80
MINIMUM_ROLLING_MEDIAN_ARI = 0.70

_CHECK_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_CANDIDATE_VIABILITY_CAPABILITY = object()
_CANDIDATE_SELECTION_CAPABILITY = object()
_CANDIDATE_COMPATIBILITY_CAPABILITY = object()
_CANDIDATE_FIXED_POINT_CAPABILITY = object()
_MINTED_CANDIDATE_VIABILITY_RESULTS: dict[int, CandidateViabilityResult] = {}
_MINTED_CANDIDATE_GATE_RESULTS: dict[int, CandidateGateResult] = {}


@dataclass(frozen=True, slots=True)
class CandidateGateEvidence:
    """Candidate-specific artifacts available after ordinary fitting.

    Every K has a row, including a failed fit.  Fields may therefore be
    ``None``.  Missing evidence for a successful fit becomes an explicit
    failed preselection check.  Fragmentation outcomes are optional because
    some order-selection scopes do not run that family; when supplied, they
    are binding preselection checks.
    """

    n_states: int
    decoded_return_pools: DecodedReturnPools | None
    rolling: RollingCandidateEvidence | None
    macro_stability: MacroStabilityEvidence | None
    monthly_block_policy: BlockPolicySelection | None
    daily_block_policy: BlockPolicySelection | None
    monthly_fragmentation_passed: bool | None = None
    daily_fragmentation_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class CandidateGateAuditRow:
    """One complete, model-object-free row in the candidate-gate audit."""

    n_states: int
    fit_succeeded: bool
    fit_failure_type: str | None
    fit_failure_message: str | None
    fit_failure_details: Mapping[str, object] | None
    monthly_block_length: int | None
    daily_block_length: int | None
    invalid_rolling_folds: tuple[int, ...]
    rolling_score_count: int
    viability: CandidateViabilityRecord

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready audit record without fitted Python objects."""

        return {
            "n_states": self.n_states,
            "fit_succeeded": self.fit_succeeded,
            "fit_failure_type": self.fit_failure_type,
            "fit_failure_message": self.fit_failure_message,
            "fit_failure_details": (
                None
                if self.fit_failure_details is None
                else dict(self.fit_failure_details)
            ),
            "monthly_block_length": self.monthly_block_length,
            "daily_block_length": self.daily_block_length,
            "invalid_rolling_folds": list(self.invalid_rolling_folds),
            "rolling_score_count": self.rolling_score_count,
            "viability": self.viability.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class CandidateViabilityResult:
    """Sealed pre-selection result with no bootstrap array or K selection."""

    fit_set: CandidateFitSet
    audit_rows: tuple[CandidateGateAuditRow, ...]
    selection_inputs: tuple[CandidateSelectionInput, ...]
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_role: ArtifactRole | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_evidence: tuple[CandidateGateEvidence, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _fit_set_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _rolling_fingerprints: tuple[str | None, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _source_content_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _result_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_suite: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _canonical_suite_execution: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    # Minted only by the forthcoming producer-sealed fixed-point wrapper.
    # A registered iteration suite alone deliberately leaves this false.
    _final_fixed_point_authority: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _final_fixed_point_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def as_dict(self) -> dict[str, object]:
        """Return pre-selection audit metadata without any selection outcome."""

        return {
            "candidate_state_counts": [row.n_states for row in self.audit_rows],
            "audit_rows": [row.as_dict() for row in self.audit_rows],
        }


@dataclass(frozen=True, slots=True)
class CandidateGateResult:
    """Frozen selector result descended from one exact viability result."""

    viability_result: CandidateViabilityResult
    selection: CandidateSelectionReport
    _registered_bootstrap_indices: IntArray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _registered_bootstrap_materialization: (
        RegisteredRollingBootstrapMaterialization | None
    ) = field(default=None, init=False, repr=False, compare=False)
    _bootstrap_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _bootstrap_rng_execution_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _selection_fingerprint: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _canonical_staged_execution: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _final_fixed_point_authority: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _final_fixed_point_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def fit_set(self) -> CandidateFitSet:
        return self.viability_result.fit_set

    @property
    def audit_rows(self) -> tuple[CandidateGateAuditRow, ...]:
        return self.viability_result.audit_rows

    @property
    def selection_inputs(self) -> tuple[CandidateSelectionInput, ...]:
        return self.viability_result.selection_inputs

    @property
    def selected_fit(self) -> GaussianHMMFit | None:
        """Return only the design-selected fit; never search for a fallback."""

        selected = self.selection.selected_n_states
        if selected is None:
            return None
        for attempt in self.fit_set.attempts:
            if attempt.n_states == selected:
                return attempt.fit
        raise ModelError(  # pragma: no cover - guarded by canonical assembly
            "selected K is absent from the canonical fit set"
        )

    def as_dict(self) -> dict[str, object]:
        """Return stable audit metadata for the entire gate."""

        return {
            "candidate_state_counts": [row.n_states for row in self.audit_rows],
            "audit_rows": [row.as_dict() for row in self.audit_rows],
            "selection": self.selection.as_dict(),
        }


def is_candidate_gate_result(value: object) -> TypeGuard[CandidateGateResult]:
    """Return whether the result came from the closed candidate evaluator."""

    return (
        isinstance(value, CandidateGateResult)
        and value._execution_seal
        in {_CANDIDATE_SELECTION_CAPABILITY, _CANDIDATE_COMPATIBILITY_CAPABILITY}
        and _MINTED_CANDIDATE_GATE_RESULTS.get(id(value)) is value
    )


def is_candidate_viability_result(
    value: object,
) -> TypeGuard[CandidateViabilityResult]:
    """Return whether the exact object came from the pre-selection executor."""

    return (
        isinstance(value, CandidateViabilityResult)
        and value._execution_seal is _CANDIDATE_VIABILITY_CAPABILITY
        and _MINTED_CANDIDATE_VIABILITY_RESULTS.get(id(value)) is value
    )


def has_registered_final_fixed_point_candidate_authority(
    value: CandidateViabilityResult | CandidateGateResult,
) -> bool:
    """Consult the fixed-point ledger and fully revalidate its live history.

    The local staging fields are intentionally ignored.  They are retained so
    views can record producer state, but changing both the flag and seal on a
    nonfinal result, copying a final result, or constructing an equivalent
    substitute cannot create an entry in the external mint ledger.
    """

    try:
        from prpg.model.registered_fixed_point import (
            has_registered_final_candidate_authority,
        )

        return has_registered_final_candidate_authority(value)
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True, slots=True)
class CandidateGateExecutionIdentity:
    """Exact execution-time sources retained by an authorizing task view.

    The ordinary candidate gate remains useful as a pure report builder in
    small synthetic tests.  Canonical G3 publication is stricter: this
    identity is available only when all scientific sources admit their full
    platform-normalized content fingerprints and those sources are unchanged
    since :func:`run_candidate_selection` returned.
    """

    role: ArtifactRole
    fit_set_fingerprint: str
    rolling_fingerprints: tuple[str | None, ...]
    bootstrap_materialization_fingerprint: str
    bootstrap_rng_execution_fingerprint: str
    bootstrap_master_seed: int
    bootstrap_scientific_version: int
    bootstrap_artifact: CalibrationArtifact
    source_content_fingerprint: str
    viability_result_fingerprint: str
    selection_fingerprint: str
    fixed_point_outcome: str | None = None
    fixed_point_failure_reason: str | None = None
    fixed_point_converged_at_iteration: int | None = None
    fixed_point_iteration_fingerprints: tuple[str, ...] = ()
    fixed_point_iteration_history_fingerprint: str | None = None
    fixed_point_fingerprint: str | None = None
    fixed_point_final_selection_fingerprint: str | None = None
    fixed_point_final_monthly_block_core_fingerprint: str | None = None
    fixed_point_final_daily_block_core_fingerprint: str | None = None
    fixed_point_viability_projection_fingerprint: str | None = None

    @property
    def bootstrap_fingerprint(self) -> str:
        """Compatibility alias for the explicit materialization identity."""

        return self.bootstrap_materialization_fingerprint


@dataclass(frozen=True, slots=True)
class CandidateViabilityExecutionIdentity:
    """Pre-selection identity that deliberately excludes bootstrap material."""

    role: ArtifactRole
    fit_set_fingerprint: str
    rolling_fingerprints: tuple[str | None, ...]
    source_content_fingerprint: str
    viability_result_fingerprint: str
    suite_fingerprint: str | None = None
    suite_combined_materialization_fingerprint: str | None = None
    suite_combined_fit_rng_execution_fingerprint: str | None = None
    suite_combined_rng_execution_fingerprint: str | None = None
    suite_macro_materialization_fingerprint: str | None = None
    suite_monthly_block_materialization_fingerprint: str | None = None
    suite_daily_block_materialization_fingerprint: str | None = None
    suite_strict_canonical_geometry: bool = False
    suite_master_seed: int | None = None
    suite_scientific_version: int | None = None
    fixed_point_final_iteration: int | None = None
    fixed_point_viability_projection_fingerprint: str | None = None


def verified_candidate_viability_execution_identity(
    value: CandidateViabilityResult,
) -> CandidateViabilityExecutionIdentity:
    """Re-fingerprint only sources consumed by the pre-BIC viability task."""

    if not is_candidate_viability_result(value):
        raise ModelError("candidate viability result lacks pre-selection authority")
    role = value._source_role
    evidence = value._source_evidence
    expected_fit = value._fit_set_fingerprint
    expected_rolling = value._rolling_fingerprints
    expected_source = value._source_content_fingerprint
    expected_result = value._result_fingerprint
    if (
        role not in {"design", "production"}
        or len(evidence) != len(CANONICAL_CANDIDATES)
        or expected_fit is None
        or len(expected_rolling) != len(CANONICAL_CANDIDATES)
        or expected_source is None
        or expected_result is None
    ):
        raise ModelError("candidate gate lacks canonical viability-source fingerprints")
    current_fit = candidate_fit_set_fingerprint(value.fit_set)
    current_rolling = _rolling_source_fingerprints(evidence)
    current_source = candidate_gate_source_evidence_fingerprint(evidence)
    current_result = _candidate_viability_result_fingerprint(value)
    if (
        current_fit != expected_fit
        or current_rolling != expected_rolling
        or current_source != expected_source
        or current_result != expected_result
    ):
        raise ModelError("candidate viability sources changed after evaluation")
    suite_fingerprint: str | None = None
    suite_combined_materialization: str | None = None
    suite_combined_fit_rng: str | None = None
    suite_combined_rng: str | None = None
    suite_macro: str | None = None
    suite_monthly: str | None = None
    suite_daily: str | None = None
    suite_strict = False
    suite_master_seed: int | None = None
    suite_scientific_version: int | None = None
    if value._canonical_suite_execution:
        from prpg.model.candidate_suite import (
            RegisteredDesignCandidateSuite,
            registered_design_candidate_suite_identity,
        )

        suite = value._source_suite
        if not isinstance(suite, RegisteredDesignCandidateSuite):
            raise ModelError("candidate viability suite ancestry is incomplete")
        suite_identity = registered_design_candidate_suite_identity(suite)
        # The identity guard above has just reverified the complete exact
        # suite ledger.  Reading its retained private parents directly here
        # avoids hashing the same potentially large materializations two more
        # times inside this single viability guard.
        suite_evidence = suite._gate_evidence
        if (
            suite._fit_set is not value.fit_set
            or len(suite_evidence) != len(evidence)
            or any(
                supplied is not retained
                for supplied, retained in zip(
                    suite_evidence,
                    evidence,
                    strict=True,
                )
            )
            or suite_identity.fit_set_fingerprint != current_fit
            or suite_identity.source_content_fingerprint != current_source
        ):
            raise ModelError("candidate viability differs from its registered suite")
        suite_fingerprint = suite_identity.suite_fingerprint
        suite_combined_materialization = (
            suite_identity.combined_materialization_fingerprint
        )
        suite_combined_fit_rng = suite_identity.combined_fit_rng_execution_fingerprint
        suite_combined_rng = suite_identity.combined_rng_execution_fingerprint
        suite_macro = suite_identity.macro_materialization_fingerprint
        suite_monthly = suite_identity.monthly_block_materialization_fingerprint
        suite_daily = suite_identity.daily_block_materialization_fingerprint
        suite_strict = suite_identity.strict_canonical_geometry
        suite_master_seed = suite_identity.master_seed
        suite_scientific_version = suite_identity.scientific_version
    elif value._source_suite is not None:
        raise ModelError("report-only candidate viability retained a suite parent")
    return CandidateViabilityExecutionIdentity(
        role=role,
        fit_set_fingerprint=current_fit,
        rolling_fingerprints=current_rolling,
        source_content_fingerprint=current_source,
        viability_result_fingerprint=current_result,
        suite_fingerprint=suite_fingerprint,
        suite_combined_materialization_fingerprint=(suite_combined_materialization),
        suite_combined_fit_rng_execution_fingerprint=suite_combined_fit_rng,
        suite_combined_rng_execution_fingerprint=suite_combined_rng,
        suite_macro_materialization_fingerprint=suite_macro,
        suite_monthly_block_materialization_fingerprint=suite_monthly,
        suite_daily_block_materialization_fingerprint=suite_daily,
        suite_strict_canonical_geometry=suite_strict,
        suite_master_seed=suite_master_seed,
        suite_scientific_version=suite_scientific_version,
    )


def verified_registered_candidate_viability_execution_identity(
    value: CandidateViabilityResult,
    *,
    require_strict_canonical_geometry: bool = False,
) -> CandidateViabilityExecutionIdentity:
    """Return viability identity only for an exact registered suite descent."""

    identity = verified_candidate_viability_execution_identity(value)
    if (
        not value._canonical_suite_execution
        or identity.suite_fingerprint is None
        or identity.suite_combined_materialization_fingerprint is None
        or identity.suite_combined_fit_rng_execution_fingerprint is None
        or identity.suite_combined_rng_execution_fingerprint is None
        or identity.suite_macro_materialization_fingerprint is None
        or identity.suite_monthly_block_materialization_fingerprint is None
        or identity.suite_daily_block_materialization_fingerprint is None
        or identity.suite_master_seed is None
        or identity.suite_scientific_version is None
    ):
        raise ModelError(
            "canonical candidate viability requires a registered design suite"
        )
    if require_strict_canonical_geometry and not (
        identity.suite_strict_canonical_geometry
    ):
        raise ModelError("canonical candidate viability requires strict block geometry")
    if has_registered_final_fixed_point_candidate_authority(value):
        from prpg.model.registered_fixed_point import (
            registered_final_viability_projection_identity,
        )

        projection = registered_final_viability_projection_identity(value)
        if (
            projection.final_suite_fingerprint != identity.suite_fingerprint
            or projection.final_viability_result_fingerprint
            != identity.viability_result_fingerprint
            or projection.master_seed != identity.suite_master_seed
            or projection.scientific_version != identity.suite_scientific_version
        ):
            raise ModelError("final viability projection differs from suite ancestry")
        identity = replace(
            identity,
            fixed_point_final_iteration=projection.final_iteration,
            fixed_point_viability_projection_fingerprint=(
                projection.viability_projection_fingerprint
            ),
        )
    return identity


def verified_candidate_gate_execution_identity(
    value: CandidateGateResult,
) -> CandidateGateExecutionIdentity:
    """Re-fingerprint and return the exact sources used by a sealed gate."""

    if not is_candidate_gate_result(value):
        raise ModelError("candidate gate result lacks closed-executor authority")
    if not value._canonical_staged_execution:
        raise ModelError("candidate selection was not executed as a staged task")
    viability = value.viability_result
    viability_identity = verified_candidate_viability_execution_identity(viability)
    materialization = value._registered_bootstrap_materialization
    indices = value._registered_bootstrap_indices
    expected_bootstrap = value._bootstrap_fingerprint
    expected_bootstrap_rng = value._bootstrap_rng_execution_fingerprint
    expected_selection = value._selection_fingerprint
    if (
        materialization is None
        or not is_registered_rolling_bootstrap_materialization(materialization)
        or indices is not materialization.indices
        or expected_bootstrap is None
        or expected_bootstrap_rng is None
        or expected_selection is None
    ):
        raise ModelError("candidate gate lacks canonical execution-source fingerprints")
    expected_artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if viability_identity.role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    current_bootstrap = materialization.materialization_fingerprint
    current_bootstrap_rng = materialization.rng_execution_fingerprint
    current_selection = _candidate_selection_fingerprint(value.selection)
    if (
        materialization.artifact is not expected_artifact
        or current_bootstrap != expected_bootstrap
        or current_bootstrap_rng != expected_bootstrap_rng
        or current_selection != expected_selection
    ):
        raise ModelError("candidate gate execution sources changed after evaluation")
    return CandidateGateExecutionIdentity(
        role=viability_identity.role,
        fit_set_fingerprint=viability_identity.fit_set_fingerprint,
        rolling_fingerprints=viability_identity.rolling_fingerprints,
        bootstrap_materialization_fingerprint=current_bootstrap,
        bootstrap_rng_execution_fingerprint=current_bootstrap_rng,
        bootstrap_master_seed=materialization.master_seed,
        bootstrap_scientific_version=materialization.scientific_version,
        bootstrap_artifact=materialization.artifact,
        source_content_fingerprint=viability_identity.source_content_fingerprint,
        viability_result_fingerprint=(viability_identity.viability_result_fingerprint),
        selection_fingerprint=current_selection,
    )


def verified_registered_candidate_gate_execution_identity(
    value: CandidateGateResult,
    *,
    require_strict_canonical_geometry: bool = False,
) -> CandidateGateExecutionIdentity:
    """Return a selection identity only when viability came from a suite."""

    result = verified_candidate_gate_execution_identity(value)
    verified_registered_candidate_viability_execution_identity(
        value.viability_result,
        require_strict_canonical_geometry=require_strict_canonical_geometry,
    )
    if has_registered_final_fixed_point_candidate_authority(value):
        from prpg.model.registered_fixed_point import (
            registered_final_selection_projection_identity,
        )

        projection = registered_final_selection_projection_identity(value)
        if (
            projection.final_selection_fingerprint != result.selection_fingerprint
            or projection.final_viability_result_fingerprint
            != result.viability_result_fingerprint
        ):
            raise ModelError("final selection projection differs from gate ancestry")
        result = replace(
            result,
            fixed_point_outcome=projection.outcome,
            fixed_point_failure_reason=projection.failure_reason,
            fixed_point_converged_at_iteration=projection.converged_at_iteration,
            fixed_point_iteration_fingerprints=projection.iteration_fingerprints,
            fixed_point_iteration_history_fingerprint=(
                projection.iteration_history_fingerprint
            ),
            fixed_point_fingerprint=projection.fixed_point_fingerprint,
            fixed_point_final_selection_fingerprint=(
                projection.final_selection_fingerprint
            ),
            fixed_point_final_monthly_block_core_fingerprint=(
                projection.final_monthly_block_core_fingerprint
            ),
            fixed_point_final_daily_block_core_fingerprint=(
                projection.final_daily_block_core_fingerprint
            ),
            fixed_point_viability_projection_fingerprint=(
                projection.viability_projection_fingerprint
            ),
        )
    return result


def candidate_gate_source_evidence_fingerprint(
    evidence: Sequence[CandidateGateEvidence],
) -> str:
    """Fingerprint all four exact preselection source rows.

    The digest covers decoded pools, rolling and macro executions, both block
    policies (including every failed trial), and optional fragmentation
    outcomes.  The candidate fit set and common rolling-bootstrap array have
    separate identities because they occupy distinct G3 materialization/RNG
    slots.
    """

    rows = _canonical_evidence_rows(evidence)
    metadata: list[dict[str, object]] = []
    for row in rows:
        metadata.append(
            {
                "n_states": row.n_states,
                "decoded_return_pools_fingerprint": (
                    None
                    if row.decoded_return_pools is None
                    else decoded_return_pools_fingerprint(row.decoded_return_pools)
                ),
                "rolling_fingerprint": (
                    None
                    if row.rolling is None
                    else rolling_candidate_evidence_fingerprint(row.rolling)
                ),
                "macro_stability_fingerprint": (
                    None
                    if row.macro_stability is None
                    else macro_stability_evidence_fingerprint(row.macro_stability)
                ),
                "monthly_block_policy": _block_policy_identity(
                    row.monthly_block_policy
                ),
                "daily_block_policy": _block_policy_identity(row.daily_block_policy),
                "monthly_fragmentation_passed": (row.monthly_fragmentation_passed),
                "daily_fragmentation_passed": row.daily_fragmentation_passed,
            }
        )
    return scientific_materialization_fingerprint(
        schema_id="candidate_gate_source_evidence",
        metadata={"rows": metadata},
        arrays={"candidate_order": np.asarray(CANONICAL_CANDIDATES)},
    )


def _candidate_viability_result_fingerprint(
    value: CandidateViabilityResult,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="candidate_viability_result_v1",
        metadata={
            "role": value._source_role,
            "audit_rows": [row.as_dict() for row in value.audit_rows],
            "selection_inputs": [
                {
                    "viability": item.viability.as_dict(),
                    "rolling_scores": list(item.rolling_scores),
                }
                for item in value.selection_inputs
            ],
        },
        arrays={"candidate_order": np.asarray(CANONICAL_CANDIDATES)},
    )


def _candidate_selection_fingerprint(
    value: CandidateSelectionReport,
) -> str:
    return scientific_materialization_fingerprint(
        schema_id="candidate_selection_result_v1",
        metadata={"selection": value.as_dict()},
        arrays={"candidate_order": np.asarray(CANONICAL_CANDIDATES)},
    )


@dataclass(frozen=True, slots=True)
class SelectedOnlyVeto:
    """One downstream decision applied only after K has been frozen."""

    veto_id: str
    passed: bool


@dataclass(frozen=True, slots=True)
class SelectedCandidateAuthorization:
    """Authorization result that cannot contain a promoted runner-up."""

    selected_n_states: int | None
    authorized_n_states: int | None
    vetoes: tuple[SelectedOnlyVeto, ...]
    passed: bool
    failure_reason: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready selected-only authorization record."""

        return {
            "selected_n_states": self.selected_n_states,
            "authorized_n_states": self.authorized_n_states,
            "vetoes": [
                {"veto_id": veto.veto_id, "passed": veto.passed} for veto in self.vetoes
            ],
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


def candidate_viability_thresholds(
    monthly_block_length: int | None,
    daily_block_length: int | None,
    *,
    n_states: int | None = None,
) -> ViabilityThresholds:
    """Return ordinary thresholds or the prospective owner-fixed K4 policy.

    A failed candidate can lack finalized policies.  In that case the fixed
    absolute floors are retained for audit display while the separately named
    block-support checks fail, so the candidate cannot become admissible.
    """

    monthly = _optional_block_length(monthly_block_length, "monthly")
    daily = _optional_block_length(daily_block_length, "daily")
    if n_states is not None and n_states not in CANONICAL_CANDIDATES:
        raise ModelError("candidate threshold state count must be 2, 3, 4, or 5")
    if n_states == OWNER_FIXED_N_STATES:
        return ViabilityThresholds(
            minimum_monthly_observations=OWNER_FIXED_MINIMUM_OBSERVATIONS,
            minimum_occupancy=OWNER_FIXED_MINIMUM_OCCUPANCY,
            minimum_completed_monthly_episodes=0,
            minimum_daily_observations=OWNER_FIXED_MINIMUM_OBSERVATIONS,
            minimum_daily_runs=0,
            maximum_covariance_condition_number=(MAXIMUM_TIED_COVARIANCE_CONDITION),
            minimum_bootstrap_median_ari=(MINIMUM_MACRO_BOOTSTRAP_MEDIAN_ARI),
            minimum_rolling_median_ari=MINIMUM_ROLLING_MEDIAN_ARI,
        )
    return ViabilityThresholds(
        minimum_monthly_observations=max(
            MINIMUM_MONTHLY_OBSERVATIONS,
            MONTHLY_BLOCK_SUPPORT_MULTIPLE * (monthly or 0),
        ),
        minimum_occupancy=MINIMUM_STATE_OCCUPANCY,
        minimum_completed_monthly_episodes=(MINIMUM_COMPLETED_MONTHLY_EPISODES),
        minimum_daily_observations=max(
            MINIMUM_DAILY_OBSERVATIONS,
            DAILY_BLOCK_SUPPORT_MULTIPLE * (daily or 0),
        ),
        minimum_daily_runs=MINIMUM_DAILY_RUNS,
        maximum_covariance_condition_number=(MAXIMUM_TIED_COVARIANCE_CONDITION),
        minimum_bootstrap_median_ari=(MINIMUM_MACRO_BOOTSTRAP_MEDIAN_ARI),
        minimum_rolling_median_ari=MINIMUM_ROLLING_MEDIAN_ARI,
    )


def run_candidate_viability(
    fit_set: CandidateFitSet,
    evidence: Sequence[CandidateGateEvidence],
    *,
    role: ArtifactRole,
) -> CandidateViabilityResult:
    """Build report-only pre-BIC viability from caller-assembled evidence.

    This compatibility path deliberately cannot authorize a canonical G3 task
    view.  Use :func:`execute_registered_candidate_viability` with a
    producer-sealed design suite for canonical execution.
    """

    return _execute_candidate_viability(
        fit_set,
        evidence,
        role=role,
        source_suite=None,
    )


def execute_registered_candidate_viability(
    suite: object,
) -> CandidateViabilityResult:
    """Execute one registered fixed-point-iteration viability reduction.

    This result proves its exact iteration suite but is not, by itself, the
    final-iteration authority needed by canonical G3 task 2.
    """

    from prpg.model.candidate_suite import (
        RegisteredDesignCandidateSuite,
        registered_design_candidate_fit_set,
        registered_design_candidate_gate_evidence,
        registered_design_candidate_suite_identity,
    )

    if not isinstance(suite, RegisteredDesignCandidateSuite):
        raise ModelError("canonical candidate viability requires a design suite")
    registered_design_candidate_suite_identity(suite)
    fit_set = registered_design_candidate_fit_set(suite)
    evidence = registered_design_candidate_gate_evidence(suite)
    before = registered_design_candidate_suite_identity(suite)
    result = _execute_candidate_viability(
        fit_set,
        evidence,
        role="design",
        source_suite=suite,
    )
    after = registered_design_candidate_suite_identity(suite)
    if before != after:
        raise ModelError("candidate suite changed during viability execution")
    verified_registered_candidate_viability_execution_identity(result)
    return result


def _execute_candidate_viability(
    fit_set: CandidateFitSet,
    evidence: Sequence[CandidateGateEvidence],
    *,
    role: ArtifactRole,
    source_suite: object | None,
) -> CandidateViabilityResult:
    """Shared deterministic reduction for report-only and suite execution."""

    if role not in {"design", "production"}:
        raise ModelError("candidate gate role must be design or production")
    attempts = _canonical_fit_attempts(fit_set)
    rows = _canonical_evidence_rows(evidence)

    audit_rows: list[CandidateGateAuditRow] = []
    selection_inputs: list[CandidateSelectionInput] = []
    for attempt, supplied in zip(attempts, rows, strict=True):
        audit, selection_input = _assemble_candidate(
            attempt,
            supplied,
            role=role,
        )
        audit_rows.append(audit)
        selection_inputs.append(selection_input)

    result = CandidateViabilityResult(
        fit_set=fit_set,
        audit_rows=tuple(audit_rows),
        selection_inputs=tuple(selection_inputs),
    )
    object.__setattr__(
        result,
        "_execution_seal",
        _CANDIDATE_VIABILITY_CAPABILITY,
    )
    object.__setattr__(result, "_source_role", role)
    object.__setattr__(result, "_source_evidence", rows)
    object.__setattr__(result, "_source_suite", source_suite)
    object.__setattr__(
        result,
        "_canonical_suite_execution",
        source_suite is not None,
    )
    _MINTED_CANDIDATE_VIABILITY_RESULTS[id(result)] = result
    try:
        fit_set_fingerprint = candidate_fit_set_fingerprint(fit_set)
        rolling_fingerprints = _rolling_source_fingerprints(rows)
        source_content_fingerprint = candidate_gate_source_evidence_fingerprint(rows)
        result_fingerprint = _candidate_viability_result_fingerprint(result)
    except ModelError:
        # Report-only callers may use deliberately compact test doubles.  Such
        # a result remains a sealed ordinary gate result, but the stricter G3
        # task-view builder rejects it because these fields remain unset.
        return result
    object.__setattr__(result, "_fit_set_fingerprint", fit_set_fingerprint)
    object.__setattr__(result, "_rolling_fingerprints", rolling_fingerprints)
    object.__setattr__(
        result,
        "_source_content_fingerprint",
        source_content_fingerprint,
    )
    object.__setattr__(result, "_result_fingerprint", result_fingerprint)
    return result


def run_candidate_selection(
    viability: CandidateViabilityResult,
    registered_bootstrap: RegisteredRollingBootstrapMaterialization,
) -> CandidateGateResult:
    """Run the ordinary report-only cross-K selector on a sealed trace."""

    return _run_registered_candidate_selection(
        viability,
        registered_bootstrap,
        selection_policy=ORDINARY_SELECTION_POLICY,
    )


def run_owner_fixed_k4_selection(
    viability: CandidateViabilityResult,
    registered_bootstrap: RegisteredRollingBootstrapMaterialization,
) -> CandidateGateResult:
    """Run the canonical owner-fixed K4 selector with sealed ancestry."""

    return _run_registered_candidate_selection(
        viability,
        registered_bootstrap,
        selection_policy=OWNER_FIXED_K4_SELECTION_POLICY,
    )


def _run_registered_candidate_selection(
    viability: CandidateViabilityResult,
    registered_bootstrap: RegisteredRollingBootstrapMaterialization,
    *,
    selection_policy: Literal["predictive_bic_v1", "owner_fixed_k4_v3"],
) -> CandidateGateResult:
    """Apply one closed selector while retaining the registered trace parent."""

    if not is_registered_rolling_bootstrap_materialization(registered_bootstrap):
        raise ModelError(
            "canonical candidate selection requires a registered rolling "
            "bootstrap materialization"
        )
    if not is_candidate_viability_result(viability) or viability._source_role not in {
        "design",
        "production",
    }:
        raise ModelError("candidate selection requires sealed viability evidence")
    expected_artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if viability._source_role == "design"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    if registered_bootstrap.artifact is not expected_artifact:
        raise ModelError("candidate selection rolling bootstrap has the wrong artifact")
    before = verified_candidate_viability_execution_identity(viability)
    result = _execute_candidate_selection(
        viability,
        registered_bootstrap.indices,
        canonical_staged=True,
        registered_bootstrap=registered_bootstrap,
        selection_policy=selection_policy,
    )
    if not is_registered_rolling_bootstrap_materialization(registered_bootstrap):
        raise ModelError("rolling bootstrap materialization changed during selection")
    after = verified_candidate_viability_execution_identity(viability)
    if before != after:
        raise ModelError("candidate viability changed during selection")
    return result


def run_candidate_gate(
    fit_set: CandidateFitSet,
    evidence: Sequence[CandidateGateEvidence],
    *,
    role: ArtifactRole,
    registered_bootstrap_indices: ArrayLike,
) -> CandidateGateResult:
    """Compatibility one-shot wrapper; its result cannot enter canonical G3."""

    viability = run_candidate_viability(fit_set, evidence, role=role)
    return _execute_candidate_selection(
        viability,
        _registered_bootstrap_indices(registered_bootstrap_indices),
        canonical_staged=False,
        registered_bootstrap=None,
        selection_policy=ORDINARY_SELECTION_POLICY,
    )


def _execute_candidate_selection(
    viability: CandidateViabilityResult,
    indices: IntArray,
    *,
    canonical_staged: bool,
    registered_bootstrap: RegisteredRollingBootstrapMaterialization | None,
    selection_policy: Literal["predictive_bic_v1", "owner_fixed_k4_v3"],
) -> CandidateGateResult:
    if not is_candidate_viability_result(viability):
        raise ModelError("candidate selection requires sealed viability evidence")
    if canonical_staged is (registered_bootstrap is None):
        raise AssertionError("candidate selection materialization mode is inconsistent")
    if selection_policy == ORDINARY_SELECTION_POLICY:
        selection = select_candidate(
            viability.selection_inputs,
            bootstrap_indices=indices,
        )
    elif selection_policy == OWNER_FIXED_K4_SELECTION_POLICY:
        selection = select_owner_fixed_k4(viability.selection_inputs)
    else:
        raise AssertionError("candidate selection policy is not closed")
    result = CandidateGateResult(viability, selection)
    object.__setattr__(result, "_registered_bootstrap_indices", indices)
    object.__setattr__(
        result,
        "_registered_bootstrap_materialization",
        registered_bootstrap,
    )
    object.__setattr__(
        result,
        "_bootstrap_fingerprint",
        (
            registered_bootstrap.materialization_fingerprint
            if registered_bootstrap is not None
            else scientific_array_fingerprint(
                indices,
                role="rolling_bootstrap_indices",
            )
        ),
    )
    object.__setattr__(
        result,
        "_bootstrap_rng_execution_fingerprint",
        (
            None
            if registered_bootstrap is None
            else registered_bootstrap.rng_execution_fingerprint
        ),
    )
    object.__setattr__(
        result,
        "_selection_fingerprint",
        _candidate_selection_fingerprint(selection),
    )
    object.__setattr__(result, "_canonical_staged_execution", canonical_staged)
    object.__setattr__(
        result,
        "_execution_seal",
        (
            _CANDIDATE_SELECTION_CAPABILITY
            if canonical_staged
            else _CANDIDATE_COMPATIBILITY_CAPABILITY
        ),
    )
    _MINTED_CANDIDATE_GATE_RESULTS[id(result)] = result
    return result


def apply_selected_only_vetoes(
    gate: CandidateGateResult,
    vetoes: Sequence[SelectedOnlyVeto],
) -> SelectedCandidateAuthorization:
    """Apply adequacy/holdout-style vetoes without rerunning K selection.

    A failed veto withholds authorization for the already selected K.  The
    function has no alternative-candidate input and therefore cannot promote
    a runner-up.
    """

    normalized = _normalize_vetoes(vetoes)
    selected = gate.selection.selected_n_states
    if not gate.selection.passed or selected is None:
        return SelectedCandidateAuthorization(
            selected_n_states=None,
            authorized_n_states=None,
            vetoes=normalized,
            passed=False,
            failure_reason="candidate_selection_failed",
        )
    failed = tuple(veto.veto_id for veto in normalized if not veto.passed)
    passed = not failed
    return SelectedCandidateAuthorization(
        selected_n_states=selected,
        authorized_n_states=selected if passed else None,
        vetoes=normalized,
        passed=passed,
        failure_reason=(None if passed else "selected_candidate_veto_failed"),
    )


def _assemble_candidate(
    attempt: CandidateFitAttempt,
    supplied: CandidateGateEvidence,
    *,
    role: ArtifactRole,
) -> tuple[CandidateGateAuditRow, CandidateSelectionInput]:
    n_states = attempt.n_states
    fit = attempt.fit
    fit_succeeded = _fit_is_consistent(attempt)

    monthly_length = _policy_length(supplied.monthly_block_policy, "monthly")
    daily_length = _policy_length(supplied.daily_block_policy, "daily")
    thresholds = candidate_viability_thresholds(
        monthly_length,
        daily_length,
        n_states=n_states,
    )

    states, pools_passed = _state_evidence(fit, supplied.decoded_return_pools)
    observed_episode_support_passed = _observed_episode_support_passed(
        fit,
        supplied.decoded_return_pools,
    )
    numerical_passed = _numerical_checks_passed(fit)
    identification_passed = bool(
        fit is not None and effective_identification_passed(fit)
    )
    historical_passed = bool(
        fit is not None and effective_transition_support_passed(fit)
    )
    monthly_support_passed = _block_support_passed(
        supplied.monthly_block_policy,
        frequency="monthly",
        n_states=n_states,
    )
    daily_support_passed = _block_support_passed(
        supplied.daily_block_policy,
        frequency="daily",
        n_states=n_states,
    )
    macro_median, macro_passed = _macro_stability(
        supplied.macro_stability,
        n_states=n_states,
        role=role,
    )
    rolling_scores, invalid_folds, rolling_median, rolling_passed = (
        _rolling_stability_and_scores(
            supplied.rolling,
            n_states=n_states,
            role=role,
        )
    )

    checks = [
        PreselectionCheck("fit_succeeded", fit_succeeded),
        PreselectionCheck("numerical_chain", numerical_passed),
        PreselectionCheck("posterior_identification", identification_passed),
        PreselectionCheck("historical_transition_support", historical_passed),
        PreselectionCheck("decoded_return_pools", pools_passed),
        PreselectionCheck("macro_stability_full_gate", macro_passed),
        PreselectionCheck("rolling_stability_full_gate", rolling_passed),
        PreselectionCheck("monthly_block_support", monthly_support_passed),
        PreselectionCheck("daily_block_support", daily_support_passed),
    ]
    if n_states == OWNER_FIXED_N_STATES:
        checks.append(
            PreselectionCheck(
                "owner_fixed_k4_observed_episode_support",
                observed_episode_support_passed,
            )
        )
    _append_fragmentation_check(
        checks,
        "monthly_fragmentation",
        supplied.monthly_fragmentation_passed,
    )
    _append_fragmentation_check(
        checks,
        "daily_fragmentation",
        supplied.daily_fragmentation_passed,
    )

    viability = evaluate_candidate_viability(
        CandidateViabilityEvidence(
            n_states=n_states,
            bic=float(fit.bic) if fit_succeeded and fit is not None else None,
            states=states,
            bootstrap_median_ari=macro_median,
            rolling_median_ari=rolling_median,
            preselection_checks=tuple(checks),
        ),
        thresholds=thresholds,
    )
    # Inadmissible rows are allowed to have no scores; admissible rows always
    # have exactly 72 because rolling evidence is a named hard check.
    scores_for_selection = rolling_scores if rolling_scores else ()
    selection_input = CandidateSelectionInput(viability, scores_for_selection)
    audit = CandidateGateAuditRow(
        n_states=n_states,
        fit_succeeded=fit_succeeded,
        fit_failure_type=attempt.failure_type,
        fit_failure_message=attempt.failure_message,
        fit_failure_details=attempt.failure_details,
        monthly_block_length=monthly_length,
        daily_block_length=daily_length,
        invalid_rolling_folds=invalid_folds,
        rolling_score_count=len(rolling_scores),
        viability=viability,
    )
    return audit, selection_input


def _canonical_fit_attempts(
    fit_set: CandidateFitSet,
) -> tuple[CandidateFitAttempt, ...]:
    if not isinstance(fit_set, CandidateFitSet):
        raise ModelError("candidate gate requires a CandidateFitSet")
    attempts = fit_set.attempts
    if any(not isinstance(attempt, CandidateFitAttempt) for attempt in attempts):
        raise ModelError("candidate fit set contains an invalid attempt record")
    if tuple(attempt.n_states for attempt in attempts) != CANONICAL_CANDIDATES:
        raise ModelError("candidate fit set must contain exact ordered K=2,3,4,5")
    return attempts


def _canonical_evidence_rows(
    evidence: Sequence[CandidateGateEvidence],
) -> tuple[CandidateGateEvidence, ...]:
    if len(evidence) != len(CANONICAL_CANDIDATES):
        raise ModelError("candidate gate evidence must contain exact K=2,3,4,5")
    by_state: dict[int, CandidateGateEvidence] = {}
    for row in evidence:
        if not isinstance(row, CandidateGateEvidence):
            raise ModelError("candidate gate evidence has an invalid record type")
        if row.n_states not in CANONICAL_CANDIDATES:
            raise ModelError("candidate gate evidence K must be 2, 3, 4, or 5")
        if row.n_states in by_state:
            raise ModelError("candidate gate evidence K values must be unique")
        by_state[row.n_states] = row
    return tuple(by_state[n_states] for n_states in CANONICAL_CANDIDATES)


def _fit_is_consistent(attempt: CandidateFitAttempt) -> bool:
    fit = attempt.fit
    if fit is None:
        return False
    return bool(
        fit.n_states == attempt.n_states
        and fit.n_features == 4
        and fit.n_observations > 0
        and math.isfinite(float(fit.bic))
        and math.isfinite(float(fit.log_likelihood))
        and attempt.failure_type is None
        and attempt.failure_message is None
        and attempt.failure_details is None
    )


def _numerical_checks_passed(fit: GaussianHMMFit | None) -> bool:
    if fit is None:
        return False
    covariance = fit.covariance_diagnostic
    chain = fit.chain_diagnostics
    return bool(
        covariance.symmetric
        and covariance.positive_definite
        and covariance.eigenvalue_floor_passed
        and covariance.condition_number_passed
        and math.isfinite(covariance.condition_number)
        and covariance.condition_number <= MAXIMUM_TIED_COVARIANCE_CONDITION
        and chain.irreducible
        and chain.aperiodic
        and chain.unique_stationary_distribution
        and chain.period == 1
        and chain.stationary_residual <= 1e-12
        and not chain.near_absorbing_states
        and chain.mixing_time_to_one_percent <= 120
    )


def _state_evidence(
    fit: GaussianHMMFit | None,
    pools: DecodedReturnPools | None,
) -> tuple[tuple[StateViabilityEvidence, ...], bool]:
    if fit is None or pools is None or not isinstance(pools, DecodedReturnPools):
        return (), False
    monthly = pools.monthly_diagnostics
    daily = pools.daily_diagnostics
    if not isinstance(monthly, StatePathDiagnostic) or not isinstance(
        daily, StatePathDiagnostic
    ):
        return (), False
    n_states = fit.n_states
    if not _path_diagnostic_consistent(
        monthly,
        pools.monthly_states,
        n_states,
    ) or not _path_diagnostic_consistent(daily, pools.daily_states, n_states):
        return (), False
    fit_states = np.asarray(fit.decoded_states)
    monthly_states = np.asarray(pools.monthly_states)
    if fit_states.shape != monthly_states.shape or not np.array_equal(
        fit_states, monthly_states
    ):
        return (), False
    condition = float(fit.covariance_diagnostic.condition_number)
    return (
        tuple(
            StateViabilityEvidence(
                state=state,
                monthly_observations=int(monthly.state_counts[state]),
                occupancy=float(monthly.occupancy[state]),
                completed_monthly_episodes=int(monthly.completed_episode_counts[state]),
                daily_observations=int(daily.state_counts[state]),
                daily_runs=int(daily.episode_counts[state]),
                covariance_condition_number=condition,
            )
            for state in range(n_states)
        ),
        True,
    )


def _observed_episode_support_passed(
    fit: GaussianHMMFit | None,
    pools: DecodedReturnPools | None,
) -> bool:
    """Require two observed monthly episodes and daily runs in every K4 state."""

    if fit is None or fit.n_states != OWNER_FIXED_N_STATES or pools is None:
        return False
    monthly = pools.monthly_diagnostics
    daily = pools.daily_diagnostics
    if not isinstance(monthly, StatePathDiagnostic) or not isinstance(
        daily, StatePathDiagnostic
    ):
        return False
    monthly_counts = np.asarray(monthly.episode_counts)
    daily_counts = np.asarray(daily.episode_counts)
    if monthly_counts.shape != (OWNER_FIXED_N_STATES,) or daily_counts.shape != (
        OWNER_FIXED_N_STATES,
    ):
        return False
    try:
        report = owner_fixed_k4_recurring_history_support(
            monthly_episode_counts=tuple(int(value) for value in monthly_counts),
            daily_run_counts=tuple(int(value) for value in daily_counts),
        )
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return report.passed


def _path_diagnostic_consistent(
    diagnostic: StatePathDiagnostic,
    states: ArrayLike,
    n_states: int,
) -> bool:
    raw_states = np.asarray(states)
    if (
        raw_states.ndim != 1
        or raw_states.size == 0
        or raw_states.dtype.kind not in "iu"
        or bool(((raw_states < 0) | (raw_states >= n_states)).any())
    ):
        return False
    expected_shape = (n_states,)
    vectors = (
        diagnostic.state_counts,
        diagnostic.occupancy,
        diagnostic.episode_counts,
        diagnostic.completed_episode_counts,
    )
    if any(np.asarray(vector).shape != expected_shape for vector in vectors):
        return False
    counts = np.bincount(raw_states.astype(np.int64), minlength=n_states)
    occupancy = counts.astype(np.float64) / len(raw_states)
    return bool(
        np.array_equal(counts, diagnostic.state_counts)
        and np.allclose(occupancy, diagnostic.occupancy, rtol=0.0, atol=1e-15)
        and sum(diagnostic.lengths) == len(raw_states)
    )


def _macro_stability(
    evidence: MacroStabilityEvidence | None,
    *,
    n_states: int,
    role: ArtifactRole,
) -> tuple[float | None, bool]:
    if evidence is None or not isinstance(evidence, MacroStabilityEvidence):
        return None, False
    summary = evidence.summary
    actual_successes = sum(attempt.fit is not None for attempt in evidence.attempts)
    attempts_match_candidate = all(
        (attempt.fit is None or attempt.fit.n_states == n_states)
        and np.asarray(attempt.jaccards).shape == (n_states,)
        for attempt in evidence.attempts
    )
    shape_ok = np.asarray(summary.median_jaccard_by_state).shape == (
        n_states,
    ) and np.asarray(summary.tenth_percentile_jaccard_by_state).shape == (n_states,)
    exact_pass = bool(
        evidence.n_states == n_states
        and evidence.role == role
        and len(evidence.attempts) == 100
        and tuple(attempt.attempt for attempt in evidence.attempts) == tuple(range(100))
        and 2 <= evidence.macro_block_length <= 24
        and shape_ok
        and attempts_match_candidate
        and summary.successful_attempts == actual_successes
        and summary.successful_attempts >= 90
        and summary.median_ari >= MINIMUM_MACRO_BOOTSTRAP_MEDIAN_ARI
        and summary.tenth_percentile_ari >= 0.50
        and bool((summary.median_jaccard_by_state >= 0.70).all())
        and bool((summary.tenth_percentile_jaccard_by_state >= 0.50).all())
        and summary.passed
    )
    median = (
        float(summary.median_ari)
        if evidence.n_states == n_states and math.isfinite(summary.median_ari)
        else None
    )
    return median, exact_pass


def _rolling_stability_and_scores(
    evidence: RollingCandidateEvidence | None,
    *,
    n_states: int,
    role: ArtifactRole,
) -> tuple[tuple[float | None, ...], tuple[int, ...], float | None, bool]:
    if evidence is None or not isinstance(evidence, RollingCandidateEvidence):
        return (), (), None, False
    if (
        evidence.n_states != n_states
        or evidence.role != role
        or len(evidence.attempts) != ROLLING_FOLD_COUNT
        or tuple(attempt.fold for attempt in evidence.attempts)
        != tuple(range(ROLLING_FOLD_COUNT))
    ):
        return (), (), None, False
    scores: list[float | None] = []
    invalid: list[int] = []
    records_valid = True
    for attempt in evidence.attempts:
        if len(attempt.scores) != ROLLING_FOLD_SIZE:
            raise ModelError("each rolling fold must contain exactly 12 scores")
        missing = tuple(value is None for value in attempt.scores)
        if attempt.fit is None:
            if not all(missing):
                raise ModelError(
                    "a failed rolling fold must contain twelve None scores"
                )
            invalid.append(attempt.fold)
        elif any(missing):
            raise ModelError("a successful rolling fold cannot contain missing scores")
        for value in attempt.scores:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float | np.integer | np.floating)
                or not math.isfinite(float(value))
            ):
                raise ModelError("rolling scores must be finite numeric values or None")
        if attempt.validation_rows != 12 or sum(attempt.validation_lengths) != 12:
            records_valid = False
        if (attempt.fit is not None and attempt.fit.n_states != n_states) or np.asarray(
            attempt.jaccards
        ).shape != (n_states,):
            records_valid = False
        scores.extend(attempt.scores)
    if len(scores) != ROLLING_SCORE_COUNT:  # pragma: no cover - loop invariant
        raise ModelError("rolling candidate did not produce exactly 72 scores")

    summary = evidence.stability
    actual_successes = sum(attempt.fit is not None for attempt in evidence.attempts)
    shape_ok = np.asarray(summary.median_jaccard_by_state).shape == (n_states,)
    exact_pass = bool(
        records_valid
        and shape_ok
        and summary.successful_fits == actual_successes
        and summary.successful_fits >= 5
        and summary.median_ari >= MINIMUM_ROLLING_MEDIAN_ARI
        and summary.minimum_successful_ari >= 0.40
        and bool((summary.median_jaccard_by_state >= 0.60).all())
        and summary.passed
    )
    median = float(summary.median_ari) if math.isfinite(summary.median_ari) else None
    return tuple(scores), tuple(invalid), median, exact_pass


def _policy_length(
    policy: BlockPolicySelection | None,
    frequency: Literal["monthly", "daily"],
) -> int | None:
    if policy is None or not isinstance(policy, BlockPolicySelection):
        return None
    if policy.frequency != frequency:
        return None
    try:
        return _optional_block_length(policy.selected_length, frequency)
    except ModelError:
        return None


def _optional_block_length(
    value: int | None,
    frequency: Literal["monthly", "daily"],
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"{frequency} block length must be an integer")
    lower, upper = (
        MONTHLY_BLOCK_LENGTH_BOUNDS
        if frequency == "monthly"
        else DAILY_BLOCK_LENGTH_BOUNDS
    )
    if not lower <= value <= upper:
        raise ModelError(f"{frequency} block length must be in [{lower}, {upper}]")
    return value


def _block_support_passed(
    policy: BlockPolicySelection | None,
    *,
    frequency: Literal["monthly", "daily"],
    n_states: int,
) -> bool:
    length = _policy_length(policy, frequency)
    if policy is None or length is None or not policy.trials:
        return False
    if not math.isfinite(policy.restart_probability) or not math.isclose(
        policy.restart_probability,
        1.0 / length,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return False
    trial_lengths = tuple(trial.intended_length for trial in policy.trials)
    if trial_lengths[-1] != length or any(
        first - second != 1 for first, second in pairwise(trial_lengths)
    ):
        return False
    if any(trial.feasible for trial in policy.trials[:-1]):
        return False
    return _selected_trial_passed(
        policy.trials[-1],
        frequency=frequency,
        n_states=n_states,
    )


def _selected_trial_passed(
    trial: LengthTrial,
    *,
    frequency: Literal["monthly", "daily"],
    n_states: int,
) -> bool:
    length = trial.intended_length
    required = max(8, length) if frequency == "monthly" else max(50, 2 * length)
    if trial.reasons or trial.required_eligible_starts != required:
        return False
    support_by_state = {item.state: item for item in trial.support}
    mean_by_state = dict(trial.realized_means)
    expected = set(range(n_states))
    if set(support_by_state) != expected or set(mean_by_state) != expected:
        return False
    for state in range(n_states):
        support = support_by_state[state]
        mean = float(mean_by_state[state])
        if (
            support.intended_length != length
            or support.completed_episodes < MINIMUM_BLOCK_COMPLETED_EPISODES
            or support.eligible_start_count < required
            or not math.isfinite(mean)
            or mean < 0.50 * length
        ):
            return False
    return True


def _rolling_source_fingerprints(
    evidence: Sequence[CandidateGateEvidence],
) -> tuple[str | None, ...]:
    return tuple(
        None
        if row.rolling is None
        else rolling_candidate_evidence_fingerprint(row.rolling)
        for row in _canonical_evidence_rows(evidence)
    )


def _block_policy_identity(
    policy: BlockPolicySelection | None,
) -> dict[str, object] | None:
    if policy is None:
        return None
    if not isinstance(policy, BlockPolicySelection):
        raise ModelError("candidate gate block policy has the wrong type")
    return {
        "frequency": policy.frequency,
        "selected_length": policy.selected_length,
        "restart_probability": policy.restart_probability,
        "trials": [
            {
                "intended_length": trial.intended_length,
                "required_eligible_starts": trial.required_eligible_starts,
                "support": [
                    {
                        "state": support.state,
                        "intended_length": support.intended_length,
                        "completed_episodes": support.completed_episodes,
                        "eligible_start_indices": list(support.eligible_start_indices),
                    }
                    for support in trial.support
                ],
                "realized_means": [list(item) for item in trial.realized_means],
                "reasons": list(trial.reasons),
            }
            for trial in policy.trials
        ],
    }


def _append_fragmentation_check(
    checks: list[PreselectionCheck],
    check_id: str,
    value: bool | None,
) -> None:
    if value is None:
        return
    if not isinstance(value, bool):
        raise ModelError(f"{check_id} outcome must be Boolean when supplied")
    checks.append(PreselectionCheck(check_id, value))


def _registered_bootstrap_indices(values: ArrayLike) -> IntArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "registered rolling bootstrap indices must be an integer array"
        ) from error
    expected = (
        ROLLING_BOOTSTRAP_REPLICATES,
        ROLLING_FOLD_COUNT,
        ROLLING_FOLD_SIZE,
    )
    if raw.shape != expected or raw.dtype.kind not in "iu":
        raise ModelError(
            "registered rolling bootstrap indices must have integer shape "
            "(10000, 6, 12)"
        )
    if bool(((raw < 0) | (raw >= ROLLING_FOLD_SIZE)).any()):
        raise ModelError("registered rolling bootstrap indices must be in [0, 11]")
    result = raw.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _normalize_vetoes(
    vetoes: Sequence[SelectedOnlyVeto],
) -> tuple[SelectedOnlyVeto, ...]:
    if not vetoes:
        raise ModelError("selected-only authorization requires at least one veto")
    result: list[SelectedOnlyVeto] = []
    seen: set[str] = set()
    for veto in vetoes:
        if not isinstance(veto, SelectedOnlyVeto):
            raise ModelError("selected-only veto has an invalid record type")
        if not _CHECK_ID.fullmatch(veto.veto_id) or veto.veto_id in seen:
            raise ModelError("selected-only veto IDs must be unique lower snake_case")
        if not isinstance(veto.passed, bool):
            raise ModelError("selected-only veto outcome must be Boolean")
        seen.add(veto.veto_id)
        result.append(veto)
    return tuple(sorted(result, key=lambda veto: veto.veto_id))
