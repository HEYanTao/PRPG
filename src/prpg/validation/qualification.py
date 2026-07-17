"""Bounded G5 candidate generation and scientific qualification.

This module intentionally implements one small success path.  It generates the
registered development and untouched qualification bundles, computes a frozen
set of observable G5-A/B/C/D statistics on equal-history windows, applies one
fixed-sample null calibration plus Holm, and publishes only passed canonical
records.  Reduced counts are useful for rehearsals but can never mint evidence
or production authority.

The inferential unit remains a path.  Each path contributes one deterministic
window with the same number of observations as its historical reference; the
95th percentile across paths (NumPy ``method='higher'``) is the declared
cluster aggregate.  No qualification value is used to tune a policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.stats import norm, wasserstein_distance

from prpg.errors import IntegrityError, ModelError, ValidationError
from prpg.model.g3_evidence_store import LoadedG3EvidenceAuthority
from prpg.model.g5_evidence_store import G5EvidenceStore
from prpg.model.g5_generation_authority import (
    G5_DEVELOPMENT_HF_PATH_COUNT,
    G5_DEVELOPMENT_LF_PATH_COUNT,
    G5_QUALIFICATION_HF_PATH_COUNT,
    G5_QUALIFICATION_LF_PATH_COUNT,
    G5DevelopmentPolicySelection,
    G5MetaValidationEvidence,
    G5QualificationEvidence,
    G5QualificationGenerationAuthority,
    G5QualifiedProductionAuthority,
    G5ThresholdEvidence,
    g5_policy_scientific_version,
    g5_policy_stream_fingerprint,
    produce_g5_meta_validation_evidence,
    produce_g5_qualification_evidence,
    produce_g5_qualification_generation_authority,
    produce_g5_qualified_production_authority,
    produce_g5_threshold_evidence,
    verify_g5_qualification_generation_authority,
)
from prpg.model.scientific_policy import (
    ScientificGenerationPolicy,
    generation_policy_from_mapping,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)
from prpg.simulation.serial import (
    G5CandidateSerialBasePath,
    G5CandidateSerialGenerationInput,
    generate_g5_candidate_serial_base_path,
    verify_g5_candidate_serial_base_path,
    verify_g5_candidate_serial_generation_input,
)
from prpg.validation.adversaries import (
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    G5AdversaryFitPair,
    fit_g5_adversary_pair,
    score_g5_adversary_path,
)
from prpg.validation.metrics import (
    PathDiversityAccumulator,
    compute_base_frequency_metrics,
)
from prpg.validation.records import (
    CanonicalThresholds,
    CanonicalValidationResults,
    ValidationDecision,
    build_canonical_results,
    build_canonical_thresholds,
)
from prpg.validation.sobol import SobolDirectionSet
from prpg.validation.statistics import (
    OUTER_NULL_REPLICATES,
    EmpiricalCriticalValue,
    HolmAudit,
    empirical_higher_critical_value,
    holm_decisions,
    pointwise_outer_null_audit,
)
from prpg.validation.store import ValidationArtifactStore

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
CandidateStage: TypeAlias = Literal["development", "qualification"]
Frequency: TypeAlias = Literal["monthly", "daily"]

G5_QUALIFICATION_SCHEMA_ID: Final = "prpg-g5-candidate-qualification-v1"
G5_NULL_CALIBRATION_SCHEMA_ID: Final = "prpg-g5-family-null-calibration-v1"
G5_FAMILY_NAMES: Final = ("g5_a", "g5_b", "g5_c", "g5_d")
G5_CLUSTER_QUANTILE: Final = 0.95
G5_DIRECT_ALPHA_SHARPE_MARGIN: Final = 0.10
G5_SERIAL_DIFFERENCE_CAP: Final = 0.10
G5_CROSS_LAG_DIFFERENCE_CAP: Final = 0.10
G5_SLICED_WASSERSTEIN_CAP: Final = 0.10
G5_ENERGY_DISTANCE_CAP: Final = 0.10
G5_DRAWDOWN_DIFFERENCE_CAP: Final = 0.10
G5_STATE_DIFFERENCE_CAP: Final = 0.10
G5_JOINT_SAMPLE_ROWS: Final = 256
G5_ESTIMATOR_SCHEMA_ID: Final = "prpg-g5-primary-estimator-v2"
G5_THRESHOLD_ESTIMATOR_BINDING: Final = "g5_estimator_contract"
G5_THRESHOLD_POLICY_BINDING: Final = "g5_generation_policy"
G5_THRESHOLD_POLICY_STREAM_BINDING: Final = "g5_policy_stream"
G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING: Final = "g5_adversary_specification"
G5_FIXED_STATE_COUNT: Final = 4
G5_ALPHA_ADVERSARY_NAMES: Final = (
    "linear.equity",
    "linear.muni_bond",
    "linear.taxable_bond",
    "quadratic.equity",
    "quadratic.muni_bond",
    "quadratic.taxable_bond",
    "memorization.equity",
    "memorization.muni_bond",
    "memorization.taxable_bond",
)
G5_ASSET_NAMES: Final = ("equity", "muni_bond", "taxable_bond")
G5_PATH_OBSERVABLE_NAMES: Final = (
    "meta.selection_uniform",
    "meta.path_rank",
    "meta.joint_row_index",
    *(f"alpha.{name}" for name in G5_ALPHA_ADVERSARY_NAMES),
    *(f"mean.{asset}" for asset in G5_ASSET_NAMES),
    *(f"covariance.{row}.{column}" for row in range(3) for column in range(3)),
    *(f"quantile.{slot}.{asset}" for slot in range(5) for asset in G5_ASSET_NAMES),
    *(f"skew.{asset}" for asset in G5_ASSET_NAMES),
    *(f"kurtosis.{asset}" for asset in G5_ASSET_NAMES),
    *(f"raw_acf.{lag}.{asset}" for lag in range(1, 21) for asset in G5_ASSET_NAMES),
    *(
        f"absolute_acf.{lag}.{asset}"
        for lag in range(1, 61)
        for asset in G5_ASSET_NAMES
    ),
    *(f"squared_acf.{lag}.{asset}" for lag in range(1, 61) for asset in G5_ASSET_NAMES),
    *(f"cross_lag.{left}.{right}" for left in range(3) for right in range(3)),
    *(f"drawdown_depth.{asset}" for asset in G5_ASSET_NAMES),
    *(f"drawdown_duration.{asset}" for asset in G5_ASSET_NAMES),
    "spread.mean",
    "spread.volatility",
    *(f"spread.quantile.{slot}" for slot in range(5)),
    *(f"spread.rolling_quantile.{slot}" for slot in range(3)),
    "spread.rolling_volatility",
    *(f"tail_coexceedance.{left}.{right}" for left, right in ((0, 1), (0, 2), (1, 2))),
    *(f"joint_vector.{asset}" for asset in G5_ASSET_NAMES),
    "source_concentration",
    "repeated_subsequence",
)
G5_PATH_OBSERVABLE_INDEX: Final = {
    name: index for index, name in enumerate(G5_PATH_OBSERVABLE_NAMES)
}
G5_ESTIMATOR_CONTRACT_FINGERPRINT: Final = hashlib.sha256(
    (
        json.dumps(
            {
                "schema_id": G5_ESTIMATOR_SCHEMA_ID,
                "path_observable_names": list(G5_PATH_OBSERVABLE_NAMES),
                "cluster_aggregation": "path_mean_with_mean_per_window_covariance_v1",
                "window_offset": "registered_purpose_14_variant_1_v1",
                "latent_state": "fitted_chain_not_decoded_history_v1",
                "adversary_specification_fingerprint": (
                    G5_ADVERSARY_SPECIFICATION_FINGERPRINT
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class G5ReferenceData:
    """One immutable base-frequency historical qualification reference."""

    frequency: Frequency
    log_returns: FloatArray
    decoded_states: IntArray
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5CandidatePathView:
    """The minimum sealed-path view consumed by statistical qualification."""

    path_id: str
    family: Literal["LF", "HF"]
    log_returns: FloatArray
    target_states: IntArray
    source_fingerprint: str
    fingerprint: str
    source_indices: IntArray | None = None
    path_entity: int = 0
    evaluation_uniform: float = 0.0
    evaluation_key_fingerprint: str | None = None
    registered_evaluation_window: bool = False
    target_monthly_states: IntArray | None = None


@dataclass(frozen=True, slots=True)
class G5CandidateBundle:
    """One policy/stage bundle with explicit canonical-count eligibility."""

    schema_id: str
    candidate_stage: CandidateStage
    generation_policy_fingerprint: str
    lf_paths: tuple[G5CandidatePathView, ...]
    hf_paths: tuple[G5CandidatePathView, ...]
    elapsed_seconds: float
    exact_canonical_counts: bool
    generation_authorized: Literal[False]
    fingerprint: str
    latent_stationary_distribution: FloatArray | None = None
    latent_transition_matrix: FloatArray | None = None
    latent_reference_fingerprint: str | None = None
    policy_scientific_version: int | None = None


@dataclass(frozen=True, slots=True)
class G5FamilyNullCalibration:
    """One fixed-sample matrix containing the four complete-suite null stats."""

    schema_id: str
    replicate_count: int
    family_names: tuple[str, ...]
    family_statistics: FloatArray
    statistics_sha256: str
    binding_fingerprints: tuple[tuple[str, str], ...]
    sobol_direction_fingerprint: str
    fixed_sample_canonical: bool
    canonical_thresholds: CanonicalThresholds | None
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5ObservedMetrics:
    """Observable candidate endpoints and their four family max statistics."""

    endpoint_values: tuple[tuple[str, float], ...]
    family_statistics: tuple[tuple[str, float], ...]
    direct_alpha_cap_ratio: float
    raw_hard_cap_ratio: float
    latent_chain_cap_ratio: float
    latent_chain_evaluated: bool
    duplicate_copies: int
    estimator_contract_fingerprint: str
    adversary_fit_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5PathObservables:
    """One fixed-order, columnar-compatible history-window statistic row."""

    frequency: Frequency
    evaluation_offset: int
    selection_uniform: float
    path_rank: int
    joint_row_index: int
    observable_names: tuple[str, ...]
    values: FloatArray
    estimator_contract_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5ReferenceObservables:
    """Candidate-independent common-reference summaries and balanced vectors."""

    frequency: Frequency
    observations: int
    observable_names: tuple[str, ...]
    values: FloatArray
    joint_sample: FloatArray
    estimator_contract_fingerprint: str
    source_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5LatentChainReference:
    """Fitted-chain target used only for latent generator correctness."""

    stationary_distribution: FloatArray
    transition_matrix: FloatArray
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5QualificationAssessment:
    """Diagnostic or canonical G5 candidate assessment."""

    development_bundle_fingerprint: str
    qualification_bundle_fingerprint: str
    null_calibration_fingerprint: str
    generation_policy_fingerprint: str | None
    policy_scientific_version: int | None
    metrics: G5ObservedMetrics
    family_p_values: tuple[tuple[str, float], ...]
    holm: HolmAudit
    decisions: tuple[ValidationDecision, ...]
    exact_canonical_inputs: bool
    passed: bool
    canonical_results: CanonicalValidationResults | None
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5CandidateActivation:
    """Published thresholds plus bounded candidate-only generation authority."""

    threshold_evidence: G5ThresholdEvidence
    qualification_authority: G5QualificationGenerationAuthority
    threshold_record_path: Path


@dataclass(frozen=True, slots=True)
class G5QualificationPublication:
    """One passed immutable result and its lifecycle evidence."""

    qualification_evidence: G5QualificationEvidence
    result_record_path: Path


@dataclass(frozen=True, slots=True)
class G5FinalPublication:
    """Passed meta evidence and final production authority."""

    meta_validation_evidence: G5MetaValidationEvidence
    production_authority: G5QualifiedProductionAuthority
    meta_record_path: Path


@dataclass(frozen=True, slots=True)
class G5ResourceProjection:
    """Linear same-host projection from a measured reduced bundle."""

    pilot_rows: int
    target_rows: int
    pilot_elapsed_seconds: float
    projected_generation_seconds: float
    projected_generation_minutes: float
    estimated_peak_memory_bytes: int
    physical_memory_bytes: int
    estimated_memory_fraction: float
    memory_passed: bool


@dataclass(frozen=True, slots=True)
class G5NullCalibrationRun:
    """Measured callback-driven null calibration run."""

    calibration: G5FamilyNullCalibration
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class G5NullResourceProjection:
    """Linear projection from a reduced matched-estimator null pilot."""

    pilot_replicates: int
    pilot_elapsed_seconds: float
    target_replicates: int
    projected_seconds: float
    projected_hours: float


FamilyNullSuite = Callable[[int], Mapping[str, float]]


def build_g5_reference_data(
    log_returns: ArrayLike,
    decoded_states: ArrayLike,
    *,
    frequency: Frequency,
) -> G5ReferenceData:
    """Validate and freeze one synchronized three-asset/state reference."""

    checked_frequency = _frequency(frequency)
    returns = _frozen_returns(log_returns, minimum=13 if frequency == "monthly" else 61)
    states = _frozen_states(decoded_states, returns.shape[0])
    identity = {
        "frequency": checked_frequency,
        "returns_sha256": _array_sha256(returns),
        "states_sha256": _array_sha256(states),
    }
    return G5ReferenceData(
        frequency=checked_frequency,
        log_returns=returns,
        decoded_states=states,
        fingerprint=_fingerprint(identity),
    )


def build_g5_latent_chain_reference(
    stationary_distribution: ArrayLike,
    transition_matrix: ArrayLike,
) -> G5LatentChainReference:
    """Freeze the fitted K=4 chain target without using decoded history."""

    stationary = _finite_vector(stationary_distribution, "latent stationary law")
    transition = np.asarray(transition_matrix, dtype=np.float64)
    if stationary.shape != (G5_FIXED_STATE_COUNT,) or transition.shape != (
        G5_FIXED_STATE_COUNT,
        G5_FIXED_STATE_COUNT,
    ):
        raise ValidationError("G5 latent reference requires one K=4 Markov chain")
    if bool((stationary < 0.0).any()) or bool((transition < 0.0).any()):
        raise ValidationError("G5 latent reference probabilities must be non-negative")
    if not math.isclose(float(stationary.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValidationError("G5 latent stationary law must sum to one")
    if not np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValidationError("G5 latent transition rows must sum to one")
    frozen_stationary = _frozen_float_array(stationary)
    frozen_transition = _frozen_float_array(transition)
    identity = {
        "stationary_sha256": _array_sha256(frozen_stationary),
        "transition_sha256": _array_sha256(frozen_transition),
        "state_count": G5_FIXED_STATE_COUNT,
    }
    return G5LatentChainReference(
        stationary_distribution=frozen_stationary,
        transition_matrix=frozen_transition,
        fingerprint=_fingerprint(identity),
    )


def compute_g5_path_observables(
    path_window: ArrayLike,
    *,
    frequency: Frequency,
    alpha_sharpes: ArrayLike | None = None,
    source_indices: ArrayLike | None = None,
    evaluation_offset: int = 0,
    selection_uniform: float = 0.0,
    path_rank: int = 1,
) -> G5PathObservables:
    """Compute one candidate-only sufficient-statistic row.

    No historical or pseudo-reference enters this function.  A path row can
    therefore be reused against the one common reference selected for a null
    replicate without silently changing the reference across paths.
    """

    checked_frequency = _frequency(frequency)
    values = _frozen_returns(path_window, minimum=2)
    offset = _nonnegative_int(evaluation_offset, "evaluation offset")
    uniform = _unit_interval(selection_uniform, "evaluation-window uniform")
    rank = _positive_int(path_rank, "canonical path rank")
    joint_row_index = min(values.shape[0] - 1, math.floor(uniform * values.shape[0]))
    if alpha_sharpes is None:
        alpha = _directional_alpha_sharpes(values, checked_frequency)
    else:
        alpha = _finite_vector(alpha_sharpes, "G5 alpha adversary statistics")
    if alpha.shape != (len(G5_ALPHA_ADVERSARY_NAMES),):
        raise ValidationError(
            "G5 path requires exactly nine alpha adversary statistics"
        )
    row = _raw_path_observable_values(
        values,
        frequency=checked_frequency,
        alpha_sharpes=alpha,
        source_indices=source_indices,
        selection_uniform=uniform,
        path_rank=rank,
        joint_row_index=joint_row_index,
    )
    identity = {
        "frequency": checked_frequency,
        "evaluation_offset": offset,
        "selection_uniform": uniform,
        "path_rank": rank,
        "joint_row_index": joint_row_index,
        "observable_names": list(G5_PATH_OBSERVABLE_NAMES),
        "values_sha256": _array_sha256(row),
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    }
    return G5PathObservables(
        frequency=checked_frequency,
        evaluation_offset=offset,
        selection_uniform=uniform,
        path_rank=rank,
        joint_row_index=joint_row_index,
        observable_names=G5_PATH_OBSERVABLE_NAMES,
        values=row,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        fingerprint=_fingerprint(identity),
    )


def compute_g5_reference_observables(
    reference: G5ReferenceData,
) -> G5ReferenceObservables:
    """Materialize one compact candidate-independent common-reference row."""

    if not isinstance(reference, G5ReferenceData):
        raise ValidationError("G5 reference observables require typed reference data")
    values = _raw_path_observable_values(
        reference.log_returns,
        frequency=reference.frequency,
        alpha_sharpes=np.zeros(len(G5_ALPHA_ADVERSARY_NAMES), dtype=np.float64),
        source_indices=None,
        selection_uniform=0.0,
        path_rank=1,
        joint_row_index=0,
    )
    size = min(G5_JOINT_SAMPLE_ROWS, reference.log_returns.shape[0])
    joint_sample = _frozen_float_array(
        reference.log_returns[_even_indices(reference.log_returns.shape[0], size)]
    )
    identity = {
        "frequency": reference.frequency,
        "observations": int(reference.log_returns.shape[0]),
        "observable_names": list(G5_PATH_OBSERVABLE_NAMES),
        "values_sha256": _array_sha256(values),
        "joint_sample_sha256": _array_sha256(joint_sample),
        "source_fingerprint": reference.fingerprint,
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    }
    return G5ReferenceObservables(
        frequency=reference.frequency,
        observations=int(reference.log_returns.shape[0]),
        observable_names=G5_PATH_OBSERVABLE_NAMES,
        values=values,
        joint_sample=joint_sample,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        source_fingerprint=reference.fingerprint,
        fingerprint=_fingerprint(identity),
    )


def pack_g5_path_observables(
    values: Sequence[G5PathObservables],
    *,
    frequency: Frequency,
) -> FloatArray:
    """Pack typed rows into the exact numeric matrix used by vectorized nulls."""

    checked_frequency = _frequency(frequency)
    if not isinstance(values, Sequence) or not values:
        raise ValidationError("G5 observable packing requires at least one row")
    rows: list[FloatArray] = []
    for value in values:
        if (
            not isinstance(value, G5PathObservables)
            or value.frequency != checked_frequency
            or value.observable_names != G5_PATH_OBSERVABLE_NAMES
            or value.estimator_contract_fingerprint != G5_ESTIMATOR_CONTRACT_FINGERPRINT
            or value.values.shape != (len(G5_PATH_OBSERVABLE_NAMES),)
        ):
            raise ValidationError("G5 path observable row has the wrong contract")
        rows.append(value.values)
    return _frozen_float_array(np.vstack(rows))


def aggregate_g5_cluster_observables(
    lf_rows: Sequence[G5PathObservables],
    hf_rows: Sequence[G5PathObservables],
    monthly_reference: G5ReferenceObservables,
    daily_reference: G5ReferenceObservables,
    *,
    sobol_directions: SobolDirectionSet,
    expected_lf: int | None = None,
    expected_hf: int | None = None,
    latent_chain_cap_ratio: float = 0.0,
    latent_chain_evaluated: bool = False,
    adversary_fit_fingerprint: str = G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    duplicate_copies: int = 0,
    raw_monthly_reference: G5ReferenceObservables | None = None,
    raw_daily_reference: G5ReferenceObservables | None = None,
) -> G5ObservedMetrics:
    """Typed wrapper around the exact columnar cluster aggregator."""

    return aggregate_g5_cluster_matrices(
        pack_g5_path_observables(lf_rows, frequency="monthly"),
        pack_g5_path_observables(hf_rows, frequency="daily"),
        monthly_reference,
        daily_reference,
        sobol_directions=sobol_directions,
        expected_lf=expected_lf,
        expected_hf=expected_hf,
        latent_chain_cap_ratio=latent_chain_cap_ratio,
        latent_chain_evaluated=latent_chain_evaluated,
        adversary_fit_fingerprint=adversary_fit_fingerprint,
        duplicate_copies=duplicate_copies,
        raw_monthly_reference=raw_monthly_reference,
        raw_daily_reference=raw_daily_reference,
    )


def aggregate_g5_cluster_matrices(
    lf_matrix: ArrayLike,
    hf_matrix: ArrayLike,
    monthly_reference: G5ReferenceObservables,
    daily_reference: G5ReferenceObservables,
    *,
    sobol_directions: SobolDirectionSet,
    expected_lf: int | None = None,
    expected_hf: int | None = None,
    latent_chain_cap_ratio: float = 0.0,
    latent_chain_evaluated: bool = False,
    adversary_fit_fingerprint: str = G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    duplicate_copies: int = 0,
    raw_monthly_reference: G5ReferenceObservables | None = None,
    raw_daily_reference: G5ReferenceObservables | None = None,
) -> G5ObservedMetrics:
    """Pure fixed-order path-cluster estimator shared by candidate and null.

    Optional raw matrices are required for a neutral policy and encode the same
    candidate windows against the untransformed historical references.  They
    are combined with policy-fidelity metrics without pooling the references.
    """

    if (
        not isinstance(sobol_directions, SobolDirectionSet)
        or monthly_reference.frequency != "monthly"
        or daily_reference.frequency != "daily"
    ):
        raise ValidationError("G5 cluster references use the wrong frequencies")
    _validate_reference_observables(monthly_reference)
    _validate_reference_observables(daily_reference)
    lf = _observable_matrix(lf_matrix, "LF")
    hf = _observable_matrix(hf_matrix, "HF")
    if expected_lf is not None and lf.shape[0] != _positive_int(
        expected_lf, "LF count"
    ):
        raise ValidationError("G5 LF observable count differs from the registered size")
    if expected_hf is not None and hf.shape[0] != _positive_int(
        expected_hf, "HF count"
    ):
        raise ValidationError("G5 HF observable count differs from the registered size")
    latent = _nonnegative_finite(latent_chain_cap_ratio, "latent-chain cap ratio")
    if type(latent_chain_evaluated) is not bool:
        raise ValidationError("latent-chain evaluated flag must be boolean")
    adversary_fit = _sha256(adversary_fit_fingerprint, "G5 adversary fit")
    duplicates = _nonnegative_int(duplicate_copies, "duplicate copies")

    monthly = _aggregate_frequency_observables(
        lf, monthly_reference, sobol_directions.directions
    )
    daily = _aggregate_frequency_observables(
        hf, daily_reference, sobol_directions.directions
    )
    primary_endpoints = {
        **{f"monthly.{name}": value for name, value in monthly.items()},
        **{f"daily.{name}": value for name, value in daily.items()},
    }
    endpoints = dict(primary_endpoints)
    raw_hard_cap_ratio = 0.0
    if (raw_monthly_reference is None) is not (raw_daily_reference is None):
        raise ValidationError("neutral raw references must be supplied together")
    if raw_monthly_reference is not None and raw_daily_reference is not None:
        _validate_reference_observables(raw_monthly_reference)
        _validate_reference_observables(raw_daily_reference)
        raw_monthly = _aggregate_frequency_observables(
            lf, raw_monthly_reference, sobol_directions.directions
        )
        raw_daily = _aggregate_frequency_observables(
            hf, raw_daily_reference, sobol_directions.directions
        )
        raw_endpoints = {
            **{f"monthly.raw.{name}": value for name, value in raw_monthly.items()},
            **{f"daily.raw.{name}": value for name, value in raw_daily.items()},
        }
        endpoints.update(raw_endpoints)
        raw_hard_cap_ratio = max(
            value
            for name, value in raw_endpoints.items()
            if not name.endswith(
                (
                    "directional_alpha",
                    "source_concentration",
                    "repeated_subsequence",
                )
            )
        )

    # Only the selected policy's primary reference enters the calibrated family
    # null.  A neutral policy additionally faces the untransformed historical
    # hard cap above; that second reference never changes the null statistic.
    family_values = _family_values(primary_endpoints)
    direct_alpha_cap_ratio = max(
        monthly["directional_alpha"], daily["directional_alpha"]
    )
    identity = {
        "endpoint_values": endpoints,
        "family_statistics": family_values,
        "direct_alpha_cap_ratio": direct_alpha_cap_ratio,
        "raw_hard_cap_ratio": raw_hard_cap_ratio,
        "latent_chain_cap_ratio": latent,
        "latent_chain_evaluated": latent_chain_evaluated,
        "duplicate_copies": duplicates,
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        "adversary_fit_fingerprint": adversary_fit,
    }
    return G5ObservedMetrics(
        endpoint_values=tuple(sorted(endpoints.items())),
        family_statistics=tuple(
            (name, family_values[name]) for name in G5_FAMILY_NAMES
        ),
        direct_alpha_cap_ratio=direct_alpha_cap_ratio,
        raw_hard_cap_ratio=raw_hard_cap_ratio,
        latent_chain_cap_ratio=latent,
        latent_chain_evaluated=latent_chain_evaluated,
        duplicate_copies=duplicates,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        adversary_fit_fingerprint=adversary_fit,
        fingerprint=_fingerprint(identity),
    )


def compute_g5_latent_chain_cap_ratio(
    state_paths: Sequence[ArrayLike],
    reference: G5LatentChainReference,
) -> float:
    """Compare generated latent chains with their fitted-chain target only."""

    if not isinstance(reference, G5LatentChainReference):
        raise ValidationError("G5 latent correctness requires a fitted-chain reference")
    if not isinstance(state_paths, Sequence) or not state_paths:
        raise ValidationError("G5 latent correctness requires state paths")
    occupancy = np.zeros(G5_FIXED_STATE_COUNT, dtype=np.float64)
    transitions = np.zeros(
        (G5_FIXED_STATE_COUNT, G5_FIXED_STATE_COUNT), dtype=np.float64
    )
    dwell_lists: list[list[int]] = [[] for _ in range(G5_FIXED_STATE_COUNT)]
    total = 0
    for raw in state_paths:
        states = np.asarray(raw)
        if (
            states.ndim != 1
            or states.size < 2
            or states.dtype.kind not in {"i", "u"}
            or bool(((states < 0) | (states >= G5_FIXED_STATE_COUNT)).any())
        ):
            raise ValidationError("G5 latent state path is invalid")
        checked = np.asarray(states, dtype=np.int64)
        occupancy += np.bincount(checked, minlength=G5_FIXED_STATE_COUNT)
        np.add.at(transitions, (checked[:-1], checked[1:]), 1.0)
        total += int(checked.size)
        start = 0
        for stop in range(1, checked.size + 1):
            if stop == checked.size or checked[stop] != checked[start]:
                dwell_lists[int(checked[start])].append(stop - start)
                start = stop
    observed_occupancy = occupancy / total
    departure = transitions.sum(axis=1, keepdims=True)
    if bool((departure <= 0.0).any()) or any(not values for values in dwell_lists):
        raise ValidationError("G5 latent paths have insufficient state support")
    observed_transition = transitions / departure
    discrepancy = max(
        float(np.max(np.abs(observed_occupancy - reference.stationary_distribution))),
        float(np.max(np.abs(observed_transition - reference.transition_matrix))),
    )
    for state, dwell in enumerate(dwell_lists):
        observed = np.asarray(dwell, dtype=np.float64)
        persistence = float(reference.transition_matrix[state, state])
        discrepancy = max(
            discrepancy,
            abs(float(np.mean(observed == 1.0)) - (1.0 - persistence)),
            max(
                abs(float(np.mean(observed >= month)) - persistence ** (month - 1))
                for month in range(1, 13)
            ),
        )
    return discrepancy / G5_STATE_DIFFERENCE_CAP


def generate_g5_candidate_bundle(
    generation_input: G5CandidateSerialGenerationInput,
    *,
    lf_path_count: int | None = None,
    hf_path_count: int | None = None,
) -> G5CandidateBundle:
    """Generate a bounded bundle through the public candidate serial API.

    Omitting counts generates the exact stage budget (100/20 or 500/200).
    Smaller positive values are permitted only as visibly noncanonical
    rehearsals.  Values above the stage budget fail before any path is drawn.
    """

    checked = verify_g5_candidate_serial_generation_input(generation_input)
    lf_count = _bounded_count(lf_path_count, checked.maximum_lf_paths, "LF paths")
    hf_count = _bounded_count(hf_path_count, checked.maximum_hf_paths, "HF paths")
    started = time.monotonic()
    lf_paths = tuple(
        _candidate_path_view(
            generate_g5_candidate_serial_base_path(
                checked, family="LF", path_entity=entity
            )
        )
        for entity in range(1, lf_count + 1)
    )
    hf_paths = tuple(
        _candidate_path_view(
            generate_g5_candidate_serial_base_path(
                checked, family="HF", path_entity=entity
            )
        )
        for entity in range(1, hf_count + 1)
    )
    elapsed = time.monotonic() - started
    latent_reference = build_g5_latent_chain_reference(
        checked.numerical_input.stationary_distribution,
        checked.numerical_input.transition_matrix,
    )
    exact = (
        lf_count == checked.maximum_lf_paths and hf_count == checked.maximum_hf_paths
    )
    identity = {
        "schema_id": G5_QUALIFICATION_SCHEMA_ID,
        "candidate_stage": checked.candidate_stage,
        "generation_policy_fingerprint": checked.generation_policy_fingerprint,
        "lf_path_fingerprints": [item.fingerprint for item in lf_paths],
        "hf_path_fingerprints": [item.fingerprint for item in hf_paths],
        "exact_canonical_counts": exact,
        "generation_authorized": False,
        "latent_reference_fingerprint": latent_reference.fingerprint,
        "policy_scientific_version": checked.policy_scientific_version,
    }
    return G5CandidateBundle(
        schema_id=G5_QUALIFICATION_SCHEMA_ID,
        candidate_stage=checked.candidate_stage,
        generation_policy_fingerprint=checked.generation_policy_fingerprint,
        lf_paths=lf_paths,
        hf_paths=hf_paths,
        elapsed_seconds=elapsed,
        exact_canonical_counts=exact,
        generation_authorized=False,
        fingerprint=_fingerprint(identity),
        latent_stationary_distribution=latent_reference.stationary_distribution,
        latent_transition_matrix=latent_reference.transition_matrix,
        latent_reference_fingerprint=latent_reference.fingerprint,
        policy_scientific_version=checked.policy_scientific_version,
    )


def build_g5_family_null_calibration(
    family_statistics: Mapping[str, ArrayLike],
    *,
    threshold_set_id: str,
    binding_fingerprints: Mapping[str, str],
    sobol_direction_fingerprint: str,
) -> G5FamilyNullCalibration:
    """Freeze equal-length four-family null arrays and, at 50k, thresholds."""

    if set(family_statistics) != set(G5_FAMILY_NAMES):
        raise ValidationError("G5 null calibration requires exactly G5-A/B/C/D")
    columns: list[FloatArray] = []
    count: int | None = None
    for name in G5_FAMILY_NAMES:
        column = _finite_vector(family_statistics[name], f"{name} null statistics")
        if bool((column < 0.0).any()):
            raise ValidationError("G5 null family statistics must be non-negative")
        if count is None:
            count = int(column.size)
        elif column.size != count:
            raise ValidationError("G5 null family arrays must have equal lengths")
        columns.append(column)
    assert count is not None
    matrix = np.column_stack(columns)
    matrix = _frozen_float_array(matrix)
    bindings = _fingerprint_mapping(binding_fingerprints, "null binding")
    sobol = _sha256(sobol_direction_fingerprint, "Sobol direction")
    canonical = count == OUTER_NULL_REPLICATES
    thresholds: CanonicalThresholds | None = None
    if canonical:
        critical_values: dict[str, EmpiricalCriticalValue] = {
            f"{name}.max": empirical_higher_critical_value(matrix[:, index], 0.95)
            for index, name in enumerate(G5_FAMILY_NAMES)
        }
        audits = {
            f"{name}.practical_cap": pointwise_outer_null_audit(
                int(np.count_nonzero(matrix[:, index] > 1.0))
            )
            for index, name in enumerate(G5_FAMILY_NAMES)
        }
        thresholds = build_canonical_thresholds(
            threshold_set_id=threshold_set_id,
            binding_fingerprints=dict(bindings),
            sobol_direction_fingerprint=sobol,
            critical_values=critical_values,
            audit_intervals=audits,
        )
    digest = _array_sha256(matrix)
    identity = {
        "schema_id": G5_NULL_CALIBRATION_SCHEMA_ID,
        "replicate_count": count,
        "family_names": list(G5_FAMILY_NAMES),
        "statistics_sha256": digest,
        "binding_fingerprints": dict(bindings),
        "sobol_direction_fingerprint": sobol,
        "fixed_sample_canonical": canonical,
        "canonical_threshold_fingerprint": (
            None if thresholds is None else thresholds.fingerprint
        ),
    }
    return G5FamilyNullCalibration(
        schema_id=G5_NULL_CALIBRATION_SCHEMA_ID,
        replicate_count=count,
        family_names=G5_FAMILY_NAMES,
        family_statistics=matrix,
        statistics_sha256=digest,
        binding_fingerprints=bindings,
        sobol_direction_fingerprint=sobol,
        fixed_sample_canonical=canonical,
        canonical_thresholds=thresholds,
        fingerprint=_fingerprint(identity),
    )


def run_g5_family_null_calibration(
    null_suite: FamilyNullSuite,
    *,
    threshold_set_id: str,
    binding_fingerprints: Mapping[str, str],
    sobol_direction_fingerprint: str,
    replicate_count: int = OUTER_NULL_REPLICATES,
) -> G5NullCalibrationRun:
    """Run one matched-estimator callback per registered null replicate.

    The callback must generate the complete pseudo development/qualification
    design and return the same four cluster statistics used on the candidate.
    This function intentionally does not accept raw-base outer-null output as
    a substitute.  Reduced counts are pilots; only exactly 50,000 callbacks
    create a canonical threshold record.
    """

    count = _positive_int(replicate_count, "G5 null replicate count")
    if count > OUTER_NULL_REPLICATES:
        raise ValidationError("G5 null replicate count exceeds the fixed 50,000")
    matrix = np.empty((count, len(G5_FAMILY_NAMES)), dtype=np.float64)
    started = time.monotonic()
    for replicate in range(count):
        values = null_suite(replicate)
        if not isinstance(values, Mapping) or set(values) != set(G5_FAMILY_NAMES):
            raise ValidationError("G5 null callback must return exactly G5-A/B/C/D")
        for index, name in enumerate(G5_FAMILY_NAMES):
            value = values[name]
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValidationError("G5 null callback returned a non-finite value")
            if float(value) < 0.0:
                raise ValidationError("G5 null callback returned a negative statistic")
            matrix[replicate, index] = float(value)
    elapsed = time.monotonic() - started
    calibration = build_g5_family_null_calibration(
        {name: matrix[:, index] for index, name in enumerate(G5_FAMILY_NAMES)},
        threshold_set_id=threshold_set_id,
        binding_fingerprints=binding_fingerprints,
        sobol_direction_fingerprint=sobol_direction_fingerprint,
    )
    return G5NullCalibrationRun(calibration=calibration, elapsed_seconds=elapsed)


def project_g5_null_resources(
    pilot: G5NullCalibrationRun,
) -> G5NullResourceProjection:
    """Project the fixed 50,000 callbacks from a same-host reduced pilot."""

    count = pilot.calibration.replicate_count
    if count <= 0 or pilot.elapsed_seconds <= 0.0:
        raise ValidationError("G5 null pilot requires positive measured work")
    projected = pilot.elapsed_seconds * OUTER_NULL_REPLICATES / count
    return G5NullResourceProjection(
        pilot_replicates=count,
        pilot_elapsed_seconds=pilot.elapsed_seconds,
        target_replicates=OUTER_NULL_REPLICATES,
        projected_seconds=projected,
        projected_hours=projected / 3600.0,
    )


def evaluate_g5_candidate(
    development_bundle: G5CandidateBundle,
    qualification_bundle: G5CandidateBundle,
    *,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
    sobol_directions: SobolDirectionSet,
    null_calibration: G5FamilyNullCalibration,
    result_id: str,
    generation_policy: ScientificGenerationPolicy | None = None,
    raw_monthly_reference: G5ReferenceData | None = None,
    raw_daily_reference: G5ReferenceData | None = None,
    adversary_fit: G5AdversaryFitPair | None = None,
) -> G5QualificationAssessment:
    """Compute frozen endpoints and apply max-stat/Holm qualification rules."""

    _validate_bundle_pair(development_bundle, qualification_bundle)
    if monthly_reference.frequency != "monthly" or daily_reference.frequency != "daily":
        raise ValidationError("G5 references use the wrong base frequencies")
    if sobol_directions.fingerprint != null_calibration.sobol_direction_fingerprint:
        raise IntegrityError("G5 Sobol directions differ from null calibration")
    policy_fingerprint: str | None = None
    expected_policy_stream: int | None = None
    expected_policy_stream_fingerprint: str | None = None
    policy_bound = False
    neutral_policy = False
    if generation_policy is not None:
        policy_fingerprint = _generation_policy_fingerprint(generation_policy)
        expected_policy_stream = g5_policy_scientific_version(generation_policy)
        expected_policy_stream_fingerprint = g5_policy_stream_fingerprint(
            generation_policy
        )
        neutral_policy = generation_policy.alpha_policy == "kernel_mean_neutral"
        policy_bound = (
            development_bundle.generation_policy_fingerprint == policy_fingerprint
            and qualification_bundle.generation_policy_fingerprint == policy_fingerprint
            and qualification_bundle.policy_scientific_version == expected_policy_stream
            and (
                development_bundle.policy_scientific_version == 11
                or development_bundle.policy_scientific_version
                == expected_policy_stream
            )
        )
    raw_reference_contract = (
        raw_monthly_reference is not None and raw_daily_reference is not None
        if neutral_policy
        else raw_monthly_reference is None and raw_daily_reference is None
    )
    metrics = compute_g5_observed_metrics(
        development_bundle,
        qualification_bundle,
        monthly_reference=monthly_reference,
        daily_reference=daily_reference,
        sobol_directions=sobol_directions,
        raw_monthly_reference=raw_monthly_reference,
        raw_daily_reference=raw_daily_reference,
        adversary_fit=adversary_fit,
    )
    observed = dict(metrics.family_statistics)
    p_values = {
        name: _upper_empirical_p_value(
            null_calibration.family_statistics[:, index], observed[name]
        )
        for index, name in enumerate(G5_FAMILY_NAMES)
    }
    holm = holm_decisions(p_values)
    holm_by_name = {item.name: item for item in holm.decisions}
    exact_counts = _exact_bundle_counts(development_bundle, qualification_bundle)
    bindings = dict(null_calibration.binding_fingerprints)
    estimator_bound = (
        bindings.get(G5_THRESHOLD_ESTIMATOR_BINDING)
        == G5_ESTIMATOR_CONTRACT_FINGERPRINT
    )
    adversary_specification_bound = (
        bindings.get(G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING)
        == G5_ADVERSARY_SPECIFICATION_FINGERPRINT
    )
    threshold_policy_bound = (
        policy_fingerprint is not None
        and expected_policy_stream is not None
        and bindings.get(G5_THRESHOLD_POLICY_BINDING) == policy_fingerprint
        and bindings.get(G5_THRESHOLD_POLICY_STREAM_BINDING)
        == expected_policy_stream_fingerprint
    )
    registered_windows = _registered_evaluation_windows(qualification_bundle)
    canonical_inputs = (
        exact_counts
        and null_calibration.fixed_sample_canonical
        and null_calibration.canonical_thresholds is not None
        and null_calibration.canonical_thresholds.precision_passed
        and policy_bound
        and threshold_policy_bound
        and estimator_bound
        and adversary_specification_bound
        and raw_reference_contract
        and registered_windows
        and metrics.latent_chain_evaluated
    )
    decisions = [
        ValidationDecision(
            name="candidate_counts_exact",
            required=True,
            passed=exact_counts,
            observed=exact_counts,
            criterion="development=100/20 and qualification=500/200",
        ),
        ValidationDecision(
            name="generation_policy_and_stream_bound",
            required=True,
            passed=policy_bound and threshold_policy_bound,
            observed=expected_policy_stream,
            criterion=(
                "bundle and frozen thresholds bind the selected policy and "
                "registered scientific stream"
            ),
        ),
        ValidationDecision(
            name="primary_estimator_contract_bound",
            required=True,
            passed=estimator_bound and adversary_specification_bound,
            observed=metrics.estimator_contract_fingerprint,
            criterion="thresholds bind the exact candidate/null estimator contract",
        ),
        ValidationDecision(
            name="registered_evaluation_windows",
            required=True,
            passed=registered_windows,
            observed=registered_windows,
            criterion="every qualification window uses purpose 14 variant 1",
        ),
        ValidationDecision(
            name="g5_a_direct_alpha_equivalence",
            required=True,
            passed=metrics.direct_alpha_cap_ratio <= 1.0,
            observed=metrics.direct_alpha_cap_ratio,
            criterion="maximum direct alpha-Sharpe cap ratio <= 1",
        ),
        ValidationDecision(
            name="g5_b_practical_cap",
            required=True,
            passed=observed["g5_b"] <= 1.0,
            observed=observed["g5_b"],
            criterion="all G5-B practical cap ratios <= 1",
        ),
        ValidationDecision(
            name="g5_c_practical_cap",
            required=True,
            passed=observed["g5_c"] <= 1.0,
            observed=observed["g5_c"],
            criterion="all G5-C practical cap ratios <= 1",
        ),
        ValidationDecision(
            name="g5_d_practical_cap",
            required=True,
            passed=observed["g5_d"] <= 1.0,
            observed=observed["g5_d"],
            criterion="all G5-D practical cap ratios <= 1",
        ),
        ValidationDecision(
            name="neutral_raw_historical_cap",
            required=neutral_policy,
            passed=raw_reference_contract and metrics.raw_hard_cap_ratio <= 1.0,
            observed=metrics.raw_hard_cap_ratio,
            criterion=(
                "neutral candidate remains within all untransformed historical "
                "B/C/D hard caps"
            ),
        ),
        ValidationDecision(
            name="latent_chain_correctness",
            required=True,
            passed=(
                metrics.latent_chain_evaluated and metrics.latent_chain_cap_ratio <= 1.0
            ),
            observed=metrics.latent_chain_cap_ratio,
            criterion="generated latent chain matches fitted K=4 pi/P/dwell caps",
        ),
        ValidationDecision(
            name="g5_d_complete_path_diversity",
            required=True,
            passed=metrics.duplicate_copies == 0,
            observed=metrics.duplicate_copies,
            criterion="zero duplicate complete base-frequency paths",
        ),
    ]
    for name in G5_FAMILY_NAMES:
        decision = holm_by_name[name]
        decisions.append(
            ValidationDecision(
                name=f"{name}_holm",
                required=True,
                passed=not decision.rejected,
                observed=decision.adjusted_p_value,
                criterion="Holm-adjusted upper-tail p-value > 0.05",
            )
        )
    decisions_tuple = tuple(decisions)
    diagnostic_pass = all(item.passed for item in decisions_tuple if item.required)
    canonical_results: CanonicalValidationResults | None = None
    if canonical_inputs:
        assert null_calibration.canonical_thresholds is not None
        endpoint_fingerprints = {
            name: _fingerprint({"name": name, "value": value})
            for name, value in metrics.endpoint_values
        }
        endpoint_fingerprints["observed_metrics"] = metrics.fingerprint
        endpoint_fingerprints["adversary_fit"] = metrics.adversary_fit_fingerprint
        canonical_results = build_canonical_results(
            result_id=result_id,
            threshold_fingerprint=(null_calibration.canonical_thresholds.fingerprint),
            subject_fingerprint=qualification_bundle.fingerprint,
            metric_fingerprints=endpoint_fingerprints,
            decisions=decisions_tuple,
        )
    passed = canonical_results is not None and canonical_results.passed
    identity = {
        "development_bundle_fingerprint": development_bundle.fingerprint,
        "qualification_bundle_fingerprint": qualification_bundle.fingerprint,
        "null_calibration_fingerprint": null_calibration.fingerprint,
        "generation_policy_fingerprint": policy_fingerprint,
        "policy_scientific_version": expected_policy_stream,
        "metrics_fingerprint": metrics.fingerprint,
        "family_p_values": p_values,
        "decisions": [item.as_dict() for item in decisions_tuple],
        "exact_canonical_inputs": canonical_inputs,
        "diagnostic_passed": diagnostic_pass,
        "passed": passed,
        "canonical_results_fingerprint": (
            None if canonical_results is None else canonical_results.fingerprint
        ),
    }
    return G5QualificationAssessment(
        development_bundle_fingerprint=development_bundle.fingerprint,
        qualification_bundle_fingerprint=qualification_bundle.fingerprint,
        null_calibration_fingerprint=null_calibration.fingerprint,
        generation_policy_fingerprint=policy_fingerprint,
        policy_scientific_version=expected_policy_stream,
        metrics=metrics,
        family_p_values=tuple(sorted(p_values.items())),
        holm=holm,
        decisions=decisions_tuple,
        exact_canonical_inputs=canonical_inputs,
        passed=passed,
        canonical_results=canonical_results,
        fingerprint=_fingerprint(identity),
    )


def compute_g5_observed_metrics(
    development_bundle: G5CandidateBundle,
    qualification_bundle: G5CandidateBundle,
    *,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
    sobol_directions: SobolDirectionSet,
    raw_monthly_reference: G5ReferenceData | None = None,
    raw_daily_reference: G5ReferenceData | None = None,
    adversary_fit: G5AdversaryFitPair | None = None,
) -> G5ObservedMetrics:
    """Compute the exact four-family estimator used by null and candidate runs."""

    _validate_bundle_pair(development_bundle, qualification_bundle)
    if qualification_bundle.candidate_stage != "qualification":
        raise ValidationError(
            "G5 observed metrics require a qualification-stage bundle"
        )
    if monthly_reference.frequency != "monthly" or daily_reference.frequency != "daily":
        raise ValidationError("G5 observed metric references use wrong frequencies")
    return _observed_metrics(
        development_bundle,
        qualification_bundle,
        monthly_reference=monthly_reference,
        daily_reference=daily_reference,
        sobol_directions=sobol_directions,
        raw_monthly_reference=raw_monthly_reference,
        raw_daily_reference=raw_daily_reference,
        adversary_fit=adversary_fit,
    )


def fit_g5_candidate_adversaries(
    development_bundle: G5CandidateBundle,
    *,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
) -> G5AdversaryFitPair:
    """Fit once on registered development windows for qualification/G7 reuse."""

    if (
        not isinstance(development_bundle, G5CandidateBundle)
        or development_bundle.candidate_stage != "development"
        or monthly_reference.frequency != "monthly"
        or daily_reference.frequency != "daily"
    ):
        raise ValidationError("G5 adversary fit inputs are invalid")
    monthly_windows = [
        _evaluation_window(path, monthly_reference.log_returns.shape[0])[0]
        for path in development_bundle.lf_paths
    ]
    daily_windows = [
        _evaluation_window(path, daily_reference.log_returns.shape[0])[0]
        for path in development_bundle.hf_paths
    ]
    return fit_g5_adversary_pair(monthly_windows, daily_windows)


def baseline_allows_g5_neutral_remedy(
    results: CanonicalValidationResults,
) -> bool:
    """Return true only for the approved sole baseline alpha-gate failure."""

    if not isinstance(results, CanonicalValidationResults) or results.passed:
        return False
    required_failures = {
        decision.name
        for decision in results.decisions
        if decision.required and not decision.passed
    }
    allowed = {"g5_a_direct_alpha_equivalence", "g5_a_holm"}
    return bool(required_failures) and required_failures <= allowed


def activate_g5_candidate_generation(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    null_calibration: G5FamilyNullCalibration,
    validation_store: ValidationArtifactStore,
    evidence_store: G5EvidenceStore,
    seed_registry_fingerprint: str,
    g5_source_code_fingerprint: str,
    g5_dependency_lock_fingerprint: str,
    generation_policy: ScientificGenerationPolicy | None = None,
    development_selection: G5DevelopmentPolicySelection | None = None,
) -> G5CandidateActivation:
    """Publish passed 50k thresholds, then mint bounded candidate authority."""

    thresholds = null_calibration.canonical_thresholds
    if (
        not null_calibration.fixed_sample_canonical
        or thresholds is None
        or not thresholds.precision_passed
    ):
        raise ModelError("G5 candidate generation requires passed canonical thresholds")
    if generation_policy is None:
        raise ModelError("G5 threshold activation requires the selected policy")
    stored = validation_store.publish_thresholds(thresholds)
    evidence = produce_g5_threshold_evidence(
        g3_evidence=g3_evidence,
        generation_policy=generation_policy,
        thresholds=thresholds,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        seed_registry_fingerprint=seed_registry_fingerprint,
        g5_source_code_fingerprint=g5_source_code_fingerprint,
        g5_dependency_lock_fingerprint=g5_dependency_lock_fingerprint,
        development_selection=development_selection,
    )
    evidence_store.publish_threshold_evidence(evidence)
    authority = produce_g5_qualification_generation_authority(
        threshold_evidence=evidence
    )
    return G5CandidateActivation(evidence, authority, stored.path)


def publish_g5_qualification(
    assessment: G5QualificationAssessment,
    *,
    activation: G5CandidateActivation,
    generation_policy: ScientificGenerationPolicy,
    adversary_fit: G5AdversaryFitPair,
    validation_store: ValidationArtifactStore,
    evidence_store: G5EvidenceStore,
) -> G5QualificationPublication:
    """Publish and seal one passed exact-count candidate assessment."""

    verify_g5_qualification_generation_authority(activation.qualification_authority)
    if not isinstance(assessment, G5QualificationAssessment):
        raise ValidationError("G5 publication requires a typed assessment")
    results = assessment.canonical_results
    if not assessment.passed or results is None or not results.passed:
        raise ModelError("G5 qualification evidence requires passed canonical results")
    if (
        results.threshold_fingerprint
        != activation.threshold_evidence.threshold_registry_fingerprint
    ):
        raise IntegrityError("G5 qualification uses a different threshold record")
    policy_fingerprint = _generation_policy_fingerprint(generation_policy)
    expected_stream = g5_policy_scientific_version(generation_policy)
    fitted = _validate_adversary_fit(adversary_fit)
    if (
        assessment.generation_policy_fingerprint != policy_fingerprint
        or assessment.policy_scientific_version != expected_stream
        or results.subject_fingerprint != assessment.qualification_bundle_fingerprint
        or fitted.fingerprint != assessment.metrics.adversary_fit_fingerprint
    ):
        raise IntegrityError(
            "G5 qualification policy, stream, or result subject differs"
        )
    stored = validation_store.publish_results(results)
    evidence = produce_g5_qualification_evidence(
        threshold_evidence=activation.threshold_evidence,
        generation_policy=generation_policy,
        qualification_results=results,
        adversary_fit=fitted,
    )
    evidence_store.publish_qualification_evidence(evidence)
    return G5QualificationPublication(evidence, stored.path)


def publish_g5_meta_and_finalize(
    meta_results: CanonicalValidationResults,
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    activation: G5CandidateActivation,
    qualification: G5QualificationPublication,
    generation_policy: ScientificGenerationPolicy,
    validation_store: ValidationArtifactStore,
    evidence_store: G5EvidenceStore,
) -> G5FinalPublication:
    """Publish passed 1,000-run meta results and mint final authority."""

    if (
        not isinstance(meta_results, CanonicalValidationResults)
        or not meta_results.passed
    ):
        raise ModelError("G5 finalization requires passed canonical meta-validation")
    if (
        meta_results.threshold_fingerprint
        != activation.threshold_evidence.threshold_registry_fingerprint
    ):
        raise IntegrityError("G5 meta-validation uses a different threshold record")
    qualification_evidence = qualification.qualification_evidence
    policy_fingerprint = _generation_policy_fingerprint(generation_policy)
    if (
        meta_results.subject_fingerprint
        != qualification_evidence.qualification_results_fingerprint
        or qualification_evidence.threshold_evidence_fingerprint
        != activation.threshold_evidence.fingerprint
        or qualification_evidence.generation_policy_fingerprint != policy_fingerprint
    ):
        raise IntegrityError(
            "G5 meta-validation subject, threshold parent, or policy differs"
        )
    stored = validation_store.publish_results(meta_results)
    meta = produce_g5_meta_validation_evidence(
        threshold_evidence=activation.threshold_evidence,
        qualification_evidence=qualification.qualification_evidence,
        meta_validation_results=meta_results,
    )
    evidence_store.publish_meta_validation_evidence(meta)
    authority = produce_g5_qualified_production_authority(
        g3_evidence=g3_evidence,
        generation_policy=generation_policy,
        threshold_evidence=activation.threshold_evidence,
        qualification_evidence=qualification.qualification_evidence,
        meta_validation_evidence=meta,
    )
    evidence_store.publish_final_authority(authority)
    return G5FinalPublication(meta, authority, stored.path)


def project_g5_candidate_resources(
    pilot_bundle: G5CandidateBundle,
    *,
    physical_memory_bytes: int,
) -> G5ResourceProjection:
    """Project exact development+qualification generation from a reduced pilot."""

    memory = _positive_int(physical_memory_bytes, "physical memory bytes")
    pilot_rows = _bundle_rows(pilot_bundle)
    if pilot_rows <= 0 or pilot_bundle.elapsed_seconds <= 0.0:
        raise ValidationError("G5 resource pilot requires positive rows and duration")
    target_rows = (
        G5_DEVELOPMENT_LF_PATH_COUNT + G5_QUALIFICATION_LF_PATH_COUNT
    ) * 600 + (G5_DEVELOPMENT_HF_PATH_COUNT + G5_QUALIFICATION_HF_PATH_COUNT) * 12_600
    projected = pilot_bundle.elapsed_seconds * target_rows / pilot_rows
    # Returns, target/source indices, restart audit, fingerprints, and bounded
    # Python object overhead, plus a conservative 256 MiB coordinator reserve.
    peak = int(target_rows * 72 + 256 * 1024 * 1024)
    fraction = peak / memory
    return G5ResourceProjection(
        pilot_rows=pilot_rows,
        target_rows=target_rows,
        pilot_elapsed_seconds=pilot_bundle.elapsed_seconds,
        projected_generation_seconds=projected,
        projected_generation_minutes=projected / 60.0,
        estimated_peak_memory_bytes=peak,
        physical_memory_bytes=memory,
        estimated_memory_fraction=fraction,
        memory_passed=fraction <= 0.70,
    )


def _observed_metrics(
    development_bundle: G5CandidateBundle,
    bundle: G5CandidateBundle,
    *,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
    sobol_directions: SobolDirectionSet,
    raw_monthly_reference: G5ReferenceData | None = None,
    raw_daily_reference: G5ReferenceData | None = None,
    adversary_fit: G5AdversaryFitPair | None = None,
) -> G5ObservedMetrics:
    monthly_observables = compute_g5_reference_observables(monthly_reference)
    daily_observables = compute_g5_reference_observables(daily_reference)
    if (raw_monthly_reference is None) is not (raw_daily_reference is None):
        raise ValidationError("neutral raw references must be supplied together")
    raw_monthly_observables = (
        None
        if raw_monthly_reference is None
        else compute_g5_reference_observables(raw_monthly_reference)
    )
    raw_daily_observables = (
        None
        if raw_daily_reference is None
        else compute_g5_reference_observables(raw_daily_reference)
    )

    fitted = (
        fit_g5_candidate_adversaries(
            development_bundle,
            monthly_reference=monthly_reference,
            daily_reference=daily_reference,
        )
        if adversary_fit is None
        else _validate_adversary_fit(adversary_fit)
    )
    adversaries = {"monthly": fitted.monthly, "daily": fitted.daily}

    lf_rows: list[G5PathObservables] = []
    hf_rows: list[G5PathObservables] = []
    groups: tuple[
        tuple[Frequency, tuple[G5CandidatePathView, ...], G5ReferenceData], ...
    ] = (
        ("monthly", bundle.lf_paths, monthly_reference),
        ("daily", bundle.hf_paths, daily_reference),
    )
    for frequency, paths, reference in groups:
        for ordinal, path in enumerate(paths, start=1):
            window, _states, source, offset = _evaluation_window(
                path, reference.log_returns.shape[0]
            )
            alpha = score_g5_adversary_path(adversaries[frequency], window)
            observable = compute_g5_path_observables(
                window,
                frequency=frequency,
                alpha_sharpes=alpha,
                source_indices=source,
                evaluation_offset=offset,
                selection_uniform=path.evaluation_uniform,
                path_rank=path.path_entity if path.path_entity > 0 else ordinal,
            )
            (lf_rows if frequency == "monthly" else hf_rows).append(observable)

    lf_diversity = PathDiversityAccumulator()
    hf_diversity = PathDiversityAccumulator()
    for path in bundle.lf_paths:
        lf_diversity.update(path.log_returns)
    for path in bundle.hf_paths:
        hf_diversity.update(path.log_returns)
    duplicate_copies = (
        lf_diversity.finalize().duplicate_copies
        + hf_diversity.finalize().duplicate_copies
    )
    latent_cap_ratio = 0.0
    latent_evaluated = False
    if (
        bundle.latent_stationary_distribution is not None
        and bundle.latent_transition_matrix is not None
        and bundle.latent_reference_fingerprint is not None
    ):
        latent_reference = build_g5_latent_chain_reference(
            bundle.latent_stationary_distribution,
            bundle.latent_transition_matrix,
        )
        if bundle.latent_reference_fingerprint != latent_reference.fingerprint:
            raise IntegrityError("G5 bundle latent reference fingerprint changed")
        state_paths: list[IntArray] = []
        for path in (*bundle.lf_paths, *bundle.hf_paths):
            states = path.target_monthly_states
            if states is None and path.family == "LF":
                states = path.target_states
            if states is None:
                state_paths = []
                break
            state_paths.append(states)
        if state_paths:
            latent_cap_ratio = compute_g5_latent_chain_cap_ratio(
                state_paths, latent_reference
            )
            latent_evaluated = True
    return aggregate_g5_cluster_observables(
        lf_rows,
        hf_rows,
        monthly_observables,
        daily_observables,
        sobol_directions=sobol_directions,
        expected_lf=len(bundle.lf_paths),
        expected_hf=len(bundle.hf_paths),
        latent_chain_cap_ratio=latent_cap_ratio,
        latent_chain_evaluated=latent_evaluated,
        adversary_fit_fingerprint=fitted.fingerprint,
        duplicate_copies=duplicate_copies,
        raw_monthly_reference=raw_monthly_observables,
        raw_daily_reference=raw_daily_observables,
    )


def _candidate_path_view(value: G5CandidateSerialBasePath) -> G5CandidatePathView:
    path = verify_g5_candidate_serial_base_path(value)
    prefix = "D" if path.candidate_stage == "development" else "Q"
    path_id = f"{prefix}{path.family}{path.path_entity:06d}"
    family = Family.LF if path.family == "LF" else Family.HF
    artifact = (
        CalibrationArtifact.DESIGN_OR_CONTROL
        if path.candidate_stage == "development"
        else CalibrationArtifact.PRODUCTION_OR_CANDIDATE
    )
    key = calibration_rng_key(
        master_seed=path.generation_input.numerical_input.master_seed,
        # Development deliberately remains on the common version-11 stream,
        # while untouched qualification must follow the selected policy's
        # registered version (11 for baseline, 12--15 for neutral candidates).
        scientific_version=path.generation_input.policy_scientific_version,
        purpose=CalibrationPurpose.SOBOL_OR_STRATEGY,
        artifact=artifact,
        variant=1,
        replicate=path.path_entity - 1,
        domain=Domain.QUALIFICATION,
        family=family,
        stage=Stage.VALIDATION_SAMPLING_OR_STRATEGY,
    )
    evaluation_uniform = float(key.generator().random())
    evaluation_key_fingerprint = _fingerprint(
        {
            "master_seed": key.master_seed,
            "spawn_key": list(key.spawn_key),
        }
    )
    identity = {
        "path_id": path_id,
        "family": path.family,
        "source_fingerprint": path.fingerprint,
        "path_entity": path.path_entity,
        "evaluation_uniform": evaluation_uniform,
        "evaluation_key_fingerprint": evaluation_key_fingerprint,
    }
    return G5CandidatePathView(
        path_id=path_id,
        family=path.family,
        log_returns=path.log_returns,
        target_states=path.target_states,
        source_fingerprint=path.fingerprint,
        fingerprint=_fingerprint(identity),
        source_indices=path.source_indices,
        path_entity=path.path_entity,
        evaluation_uniform=evaluation_uniform,
        evaluation_key_fingerprint=evaluation_key_fingerprint,
        registered_evaluation_window=True,
        target_monthly_states=path.target_monthly_states,
    )


def _evaluation_window(
    path: G5CandidatePathView, observations: int
) -> tuple[FloatArray, IntArray, IntArray | None, int]:
    if path.log_returns.shape[0] < observations:
        raise ValidationError("candidate path is shorter than its historical reference")
    choices = path.log_returns.shape[0] - observations + 1
    if not 0.0 <= path.evaluation_uniform < 1.0:
        raise ValidationError("candidate evaluation-window uniform is invalid")
    offset = min(choices - 1, math.floor(path.evaluation_uniform * choices))
    returns = path.log_returns[offset : offset + observations]
    states = path.target_states[offset : offset + observations]
    if states.shape != (observations,):
        raise ValidationError("candidate state window is incomplete")
    source: IntArray | None = None
    if path.source_indices is not None:
        source = np.asarray(
            path.source_indices[offset : offset + observations], dtype=np.int64
        )
        if source.shape != (observations,):
            raise ValidationError("candidate source-index window is incomplete")
    return returns, states, source, offset


def _directional_alpha_sharpes(values: FloatArray, frequency: Frequency) -> FloatArray:
    predictors = np.where(values[:-1] >= 0.0, 1.0, -1.0)
    outcomes = values[1:]
    payoffs = predictors[:, :, np.newaxis] * outcomes[:, np.newaxis, :]
    flattened = payoffs.reshape(payoffs.shape[0], -1)
    scale = np.std(flattened, axis=0, ddof=1)
    valid = scale > 1e-12
    if not bool(valid.any()):
        raise ValidationError("directional-alpha endpoint is degenerate")
    annualization = 12.0 if frequency == "monthly" else 252.0
    sharpes = np.zeros(scale.shape, dtype=np.float64)
    sharpes[valid] = (
        math.sqrt(annualization) * np.mean(flattened[:, valid], axis=0) / scale[valid]
    )
    return sharpes


def _cross_lag(values: FloatArray) -> float:
    left = values[:-1]
    right = values[1:]
    joined = np.corrcoef(np.column_stack((left, right)), rowvar=False)
    block = joined[:3, 3:]
    if not bool(np.isfinite(block).all()):
        raise ValidationError("cross-lag endpoint is non-finite")
    return float(np.max(np.abs(block)))


def _joint_distances(
    candidate: FloatArray, reference: FloatArray, directions: FloatArray
) -> tuple[float, float]:
    scale = np.std(reference, axis=0, ddof=1)
    if bool((scale <= 1e-12).any()) or not bool(np.isfinite(scale).all()):
        raise ValidationError("joint-distance reference scale is degenerate")
    center = np.mean(reference, axis=0)
    candidate_standard = (candidate - center) / scale
    reference_standard = (reference - center) / scale
    size = min(G5_JOINT_SAMPLE_ROWS, candidate.shape[0], reference.shape[0])
    candidate_sample = candidate_standard[_even_indices(candidate.shape[0], size)]
    reference_sample = reference_standard[_even_indices(reference.shape[0], size)]
    candidate_projection = np.sort(candidate_sample @ directions.T, axis=0)
    reference_projection = np.sort(reference_sample @ directions.T, axis=0)
    sliced = float(np.mean(np.abs(candidate_projection - reference_projection)))
    cross = float(np.mean(cdist(candidate_sample, reference_sample)))
    within_candidate = float(np.mean(cdist(candidate_sample, candidate_sample)))
    within_reference = float(np.mean(cdist(reference_sample, reference_sample)))
    energy_squared = max(0.0, 2.0 * cross - within_candidate - within_reference)
    energy = math.sqrt(energy_squared)
    if not math.isfinite(sliced) or not math.isfinite(energy):
        raise ValidationError("joint-distance endpoint is non-finite")
    return sliced, energy


def _joint_distances_from_samples(
    candidate: FloatArray,
    reference: FloatArray,
    directions: FloatArray,
    *,
    center: FloatArray,
    scale: FloatArray,
) -> tuple[float, float]:
    """Compute joint distances from one already-balanced sample pair."""

    if (
        candidate.ndim != 2
        or reference.ndim != 2
        or candidate.shape != reference.shape
        or candidate.shape[0] < 2
        or candidate.shape[1] != 3
        or directions.ndim != 2
        or directions.shape[1] != 3
        or directions.shape[0] == 0
        or center.shape != (3,)
        or scale.shape != (3,)
        or bool((scale <= 1e-12).any())
        or not bool(
            np.isfinite(candidate).all()
            and np.isfinite(reference).all()
            and np.isfinite(directions).all()
            and np.isfinite(center).all()
            and np.isfinite(scale).all()
        )
    ):
        raise ValidationError("balanced G5 joint samples are invalid")
    candidate_standard = (candidate - center) / scale
    reference_standard = (reference - center) / scale
    candidate_projection = np.sort(candidate_standard @ directions.T, axis=0)
    reference_projection = np.sort(reference_standard @ directions.T, axis=0)
    sliced = float(np.mean(np.abs(candidate_projection - reference_projection)))
    cross = float(np.mean(cdist(candidate_standard, reference_standard)))
    within_candidate = float(np.mean(cdist(candidate_standard, candidate_standard)))
    within_reference = float(np.mean(cdist(reference_standard, reference_standard)))
    energy = math.sqrt(max(0.0, 2.0 * cross - within_candidate - within_reference))
    if not math.isfinite(sliced) or not math.isfinite(energy):
        raise ValidationError("G5 joint-distance endpoint is non-finite")
    return sliced, energy


def _cross_lag_matrix(values: FloatArray) -> FloatArray:
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        raise ValidationError("G5 cross-lag endpoint has insufficient support")
    joined = np.corrcoef(np.column_stack((values[:-1], values[1:])), rowvar=False)
    result = np.asarray(joined[:3, 3:], dtype=np.float64)
    if result.shape != (3, 3) or not bool(np.isfinite(result).all()):
        raise ValidationError("G5 cross-lag endpoint is non-finite")
    return result


def _raw_path_observable_values(
    values: FloatArray,
    *,
    frequency: Frequency,
    alpha_sharpes: FloatArray,
    source_indices: ArrayLike | None,
    selection_uniform: float,
    path_rank: int,
    joint_row_index: int,
) -> FloatArray:
    probabilities = (
        (0.10, 0.50, 0.90) if frequency == "monthly" else (0.01, 0.05, 0.50, 0.95, 0.99)
    )
    metrics = compute_base_frequency_metrics(
        values,
        frequency=frequency,
        quantile_probabilities=probabilities,
    )
    annualization = 12 if frequency == "monthly" else 252
    centered = values - np.mean(values, axis=0)
    second = np.mean(centered**2, axis=0)
    if bool((second <= 1e-12).any()):
        raise ValidationError("G5 path has degenerate marginal moments")
    skew = np.mean(centered**3, axis=0) / np.power(second, 1.5)
    kurtosis = np.mean(centered**4, axis=0) / np.square(second)
    cross_lag = _cross_lag_matrix(values)
    spread = values[:, 2] - values[:, 1]
    spread_quantiles = np.quantile(spread, probabilities, method="higher")
    # Reduced rehearsals may use a shorter-than-one-year history; canonical
    # references retain the registered 12/252-period rolling horizon.
    rolling = _rolling_sums(spread, min(annualization, values.shape[0] - 1))
    rolling_quantiles = np.quantile(rolling, (0.10, 0.50, 0.90), method="higher")
    tail_probability = 0.10 if frequency == "monthly" else 0.05
    tail_threshold = np.quantile(values, tail_probability, axis=0, method="higher")
    down = values <= tail_threshold
    tail_coexceedance = np.asarray(
        [
            np.mean(down[:, left] & down[:, right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ],
        dtype=np.float64,
    )
    source_concentration, repeated_subsequence = _source_diversity_observables(
        source_indices,
        observations=values.shape[0],
        frequency=frequency,
    )
    by_name = dict.fromkeys(G5_PATH_OBSERVABLE_NAMES, 0.0)
    by_name["meta.selection_uniform"] = selection_uniform
    by_name["meta.path_rank"] = float(path_rank)
    by_name["meta.joint_row_index"] = float(joint_row_index)
    for index, name in enumerate(G5_ALPHA_ADVERSARY_NAMES):
        by_name[f"alpha.{name}"] = float(alpha_sharpes[index])
    base_mean = np.asarray(metrics.annualized_log_mean) / annualization
    for asset, name in enumerate(G5_ASSET_NAMES):
        by_name[f"mean.{name}"] = float(base_mean[asset])
        by_name[f"skew.{name}"] = float(skew[asset])
        by_name[f"kurtosis.{name}"] = float(kurtosis[asset])
        by_name[f"drawdown_depth.{name}"] = float(metrics.maximum_drawdown[asset])
        by_name[f"drawdown_duration.{name}"] = (
            float(metrics.maximum_drawdown_duration[asset]) / values.shape[0]
        )
        by_name[f"joint_vector.{name}"] = float(values[joint_row_index, asset])
    covariance = np.asarray(metrics.base_frequency_covariance)
    for row in range(3):
        for column in range(3):
            by_name[f"covariance.{row}.{column}"] = float(covariance[row, column])
            by_name[f"cross_lag.{row}.{column}"] = float(cross_lag[row, column])
    quantiles = np.asarray(metrics.marginal_quantiles)
    for slot in range(len(probabilities)):
        for asset, name in enumerate(G5_ASSET_NAMES):
            by_name[f"quantile.{slot}.{name}"] = float(quantiles[slot, asset])
    for row, lag in enumerate(metrics.raw_acf_lags):
        for asset, name in enumerate(G5_ASSET_NAMES):
            by_name[f"raw_acf.{lag}.{name}"] = float(metrics.raw_acf[row][asset])
    for prefix, lags, rows in (
        ("absolute_acf", metrics.absolute_acf_lags, metrics.absolute_acf),
        ("squared_acf", metrics.squared_acf_lags, metrics.squared_acf),
    ):
        for row, lag in enumerate(lags):
            for asset, name in enumerate(G5_ASSET_NAMES):
                by_name[f"{prefix}.{lag}.{name}"] = float(rows[row][asset])
    by_name["spread.mean"] = float(np.mean(spread))
    by_name["spread.volatility"] = float(np.std(spread, ddof=1))
    for slot, value in enumerate(spread_quantiles):
        by_name[f"spread.quantile.{slot}"] = float(value)
    for slot, value in enumerate(rolling_quantiles):
        by_name[f"spread.rolling_quantile.{slot}"] = float(value)
    by_name["spread.rolling_volatility"] = float(np.std(rolling, ddof=1))
    for index, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        by_name[f"tail_coexceedance.{left}.{right}"] = float(tail_coexceedance[index])
    by_name["source_concentration"] = source_concentration
    by_name["repeated_subsequence"] = repeated_subsequence
    path_row = np.asarray([by_name[name] for name in G5_PATH_OBSERVABLE_NAMES])
    if path_row.shape != (len(G5_PATH_OBSERVABLE_NAMES),) or not bool(
        np.isfinite(path_row).all()
    ):
        raise ValidationError("G5 raw path observables are invalid")
    return _frozen_float_array(path_row)


def _observable_matrix(values: ArrayLike, label: str) -> FloatArray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} observable matrix must be numeric") from error
    if matrix.ndim != 2 or matrix.shape[1] != len(G5_PATH_OBSERVABLE_NAMES):
        raise ValidationError(f"{label} observable matrix has the wrong shape")
    if matrix.shape[0] == 0 or not bool(np.isfinite(matrix).all()):
        raise ValidationError(f"{label} observable matrix must be finite and non-empty")
    return matrix


def _validate_reference_observables(value: G5ReferenceObservables) -> None:
    if (
        not isinstance(value, G5ReferenceObservables)
        or value.frequency not in {"monthly", "daily"}
        or isinstance(value.observations, bool)
        or not isinstance(value.observations, int)
        or value.observations < 2
        or value.observable_names != G5_PATH_OBSERVABLE_NAMES
        or value.values.shape != (len(G5_PATH_OBSERVABLE_NAMES),)
        or value.joint_sample.shape
        != (min(G5_JOINT_SAMPLE_ROWS, value.observations), 3)
        or value.estimator_contract_fingerprint != G5_ESTIMATOR_CONTRACT_FINGERPRINT
        or not bool(
            np.isfinite(value.values).all() and np.isfinite(value.joint_sample).all()
        )
    ):
        raise ValidationError("G5 reference observables have the wrong contract")
    _sha256(value.source_fingerprint, "G5 reference source")
    expected = _fingerprint(
        {
            "frequency": value.frequency,
            "observations": value.observations,
            "observable_names": list(value.observable_names),
            "values_sha256": _array_sha256(value.values),
            "joint_sample_sha256": _array_sha256(value.joint_sample),
            "source_fingerprint": value.source_fingerprint,
            "estimator_contract_fingerprint": (value.estimator_contract_fingerprint),
        }
    )
    if value.fingerprint != expected:
        raise IntegrityError("G5 reference observable fingerprint changed")


def _aggregate_frequency_observables(
    matrix: FloatArray,
    reference: G5ReferenceObservables,
    directions: FloatArray,
) -> dict[str, float]:
    observed = matrix
    reference_values = reference.values

    def columns(names: Sequence[str]) -> FloatArray:
        indices = [G5_PATH_OBSERVABLE_INDEX[name] for name in names]
        return observed[:, indices]

    def reference_vector(names: Sequence[str]) -> FloatArray:
        indices = [G5_PATH_OBSERVABLE_INDEX[name] for name in names]
        return reference_values[indices]

    alpha_names = [f"alpha.{name}" for name in G5_ALPHA_ADVERSARY_NAMES]
    covariance_names = [
        f"covariance.{row}.{column}" for row in range(3) for column in range(3)
    ]
    mean_names = [f"mean.{asset}" for asset in G5_ASSET_NAMES]
    mean_covariance = np.mean(columns(covariance_names), axis=0).reshape(3, 3)
    reference_covariance = reference_vector(covariance_names).reshape(3, 3)
    denominator = float(np.linalg.norm(reference_covariance, ord="fro"))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValidationError("G5 covariance reference is degenerate")
    covariance_ratio = (
        float(np.linalg.norm(mean_covariance - reference_covariance, ord="fro"))
        / denominator
        / 0.10
    )
    candidate_correlation = _covariance_to_correlation(mean_covariance)
    reference_correlation = _covariance_to_correlation(reference_covariance)
    correlation_ratio = (
        float(
            np.max(
                np.abs(candidate_correlation - reference_correlation)[
                    np.triu_indices(3, k=1)
                ]
            )
        )
        / 0.05
    )
    annualization = 12 if reference.frequency == "monthly" else 252
    candidate_mean = np.mean(columns(mean_names), axis=0)
    reference_mean = reference_vector(mean_names)
    reference_base_vol = np.sqrt(np.diag(reference_covariance))
    reference_annual_vol = reference_base_vol * math.sqrt(annualization)
    location_scale = max(
        float(
            np.max(
                np.abs(annualization * (candidate_mean - reference_mean))
                / reference_annual_vol
            )
        )
        / 0.05,
        float(np.max(np.abs(annualization * (candidate_mean - reference_mean)))) / 0.01,
        float(
            np.max(
                np.abs(np.sqrt(np.diag(mean_covariance)) - reference_base_vol)
                / reference_base_vol
            )
        )
        / 0.05,
    )
    probability_slots = 3 if reference.frequency == "monthly" else 5
    quantile_names = [
        f"quantile.{slot}.{asset}"
        for slot in range(probability_slots)
        for asset in G5_ASSET_NAMES
    ]
    marginal_tail = float(
        np.max(
            np.abs(
                np.mean(columns(quantile_names), axis=0)
                - reference_vector(quantile_names)
            )
            / np.tile(reference_base_vol, probability_slots)
        )
        / 0.15
    )
    shape_names = [
        *(f"skew.{asset}" for asset in G5_ASSET_NAMES),
        *(f"kurtosis.{asset}" for asset in G5_ASSET_NAMES),
    ]
    marginal_shape = float(
        np.max(
            np.abs(
                np.mean(columns(shape_names), axis=0) - reference_vector(shape_names)
            )
        )
    )
    raw_limit = 12 if reference.frequency == "monthly" else 20
    transformed_limit = 12 if reference.frequency == "monthly" else 60
    acf_names = [
        *(
            f"raw_acf.{lag}.{asset}"
            for lag in range(1, raw_limit + 1)
            for asset in G5_ASSET_NAMES
        ),
        *(
            f"absolute_acf.{lag}.{asset}"
            for lag in range(1, transformed_limit + 1)
            for asset in G5_ASSET_NAMES
        ),
        *(
            f"squared_acf.{lag}.{asset}"
            for lag in range(1, transformed_limit + 1)
            for asset in G5_ASSET_NAMES
        ),
    ]
    serial_dependence = float(
        np.max(
            np.abs(np.mean(columns(acf_names), axis=0) - reference_vector(acf_names))
        )
        / G5_SERIAL_DIFFERENCE_CAP
    )
    cross_names = [
        f"cross_lag.{left}.{right}" for left in range(3) for right in range(3)
    ]
    cross_lag = float(
        np.max(
            np.abs(
                np.mean(columns(cross_names), axis=0) - reference_vector(cross_names)
            )
        )
        / G5_CROSS_LAG_DIFFERENCE_CAP
    )
    uniform = columns(("meta.selection_uniform",))[:, 0]
    path_rank = columns(("meta.path_rank",))[:, 0]
    sample_size = min(
        G5_JOINT_SAMPLE_ROWS, matrix.shape[0], reference.joint_sample.shape[0]
    )
    selected = np.lexsort((path_rank, uniform))[:sample_size]
    joint_names = [f"joint_vector.{asset}" for asset in G5_ASSET_NAMES]
    candidate_joint = columns(joint_names)[selected]
    reference_joint = reference.joint_sample[
        _even_indices(reference.joint_sample.shape[0], sample_size)
    ]
    sliced, energy = _joint_distances_from_samples(
        candidate_joint,
        reference_joint,
        directions,
        center=reference_mean,
        scale=reference_base_vol,
    )
    spread_mean_name = ("spread.mean",)
    spread_vol_name = ("spread.volatility",)
    spread_quantile_names = [
        f"spread.quantile.{slot}" for slot in range(probability_slots)
    ]
    candidate_spread_mean = float(np.mean(columns(spread_mean_name)))
    reference_spread_mean = float(reference_vector(spread_mean_name)[0])
    candidate_spread_vol = float(np.mean(columns(spread_vol_name)))
    reference_spread_vol = float(reference_vector(spread_vol_name)[0])
    if reference_spread_vol <= 1e-12:
        raise ValidationError("G5 reference spread is degenerate")
    spread = max(
        annualization
        * abs(candidate_spread_mean - reference_spread_mean)
        / (reference_spread_vol * math.sqrt(annualization))
        / 0.05,
        annualization * abs(candidate_spread_mean - reference_spread_mean) / 0.005,
        abs(candidate_spread_vol - reference_spread_vol) / reference_spread_vol / 0.05,
        float(
            np.max(
                np.abs(
                    np.mean(columns(spread_quantile_names), axis=0)
                    - reference_vector(spread_quantile_names)
                )
            )
        )
        / reference_spread_vol
        / 0.15,
    )
    candidate_spread_sample = candidate_joint[:, 2] - candidate_joint[:, 1]
    reference_spread_sample = reference_joint[:, 2] - reference_joint[:, 1]
    spread_wasserstein = (
        float(wasserstein_distance(candidate_spread_sample, reference_spread_sample))
        / reference_spread_vol
        / 0.10
    )
    rolling_names = [f"spread.rolling_quantile.{slot}" for slot in range(3)]
    rolling_scale = float(
        reference_values[G5_PATH_OBSERVABLE_INDEX["spread.rolling_volatility"]]
    )
    if rolling_scale <= 1e-12:
        raise ValidationError("G5 reference rolling spread is degenerate")
    spread_rolling = float(
        np.max(
            np.abs(
                np.mean(columns(rolling_names), axis=0)
                - reference_vector(rolling_names)
            )
        )
        / rolling_scale
        / 0.15
    )
    tail_names = [
        f"tail_coexceedance.{left}.{right}" for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    tail_dependence = float(
        np.max(
            np.abs(np.mean(columns(tail_names), axis=0) - reference_vector(tail_names))
        )
        / 0.02
    )
    drawdown_names = [
        *(f"drawdown_depth.{asset}" for asset in G5_ASSET_NAMES),
        *(f"drawdown_duration.{asset}" for asset in G5_ASSET_NAMES),
    ]
    drawdown = float(
        np.max(
            np.abs(
                np.mean(columns(drawdown_names), axis=0)
                - reference_vector(drawdown_names)
            )
        )
        / G5_DRAWDOWN_DIFFERENCE_CAP
    )
    output = {
        "directional_alpha": _simultaneous_alpha_cap_ratio(columns(alpha_names)),
        "covariance_correlation": max(covariance_ratio, correlation_ratio),
        "serial_dependence": serial_dependence,
        "cross_lag": cross_lag,
        "sliced_wasserstein": sliced / G5_SLICED_WASSERSTEIN_CAP,
        "energy_distance": energy / G5_ENERGY_DISTANCE_CAP,
        "spread": spread,
        "spread_wasserstein": spread_wasserstein,
        "spread_rolling": spread_rolling,
        "location_scale": location_scale,
        "marginal_tail": marginal_tail,
        "marginal_shape": marginal_shape,
        "tail_dependence": tail_dependence,
        "drawdown": drawdown,
        "source_concentration": float(np.mean(columns(("source_concentration",)))),
        "repeated_subsequence": float(np.mean(columns(("repeated_subsequence",)))),
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in output.values()):
        raise ValidationError("G5 cluster aggregation produced an invalid endpoint")
    return output


def _simultaneous_alpha_cap_ratio(values: FloatArray) -> float:
    if values.ndim != 2 or values.shape[1] != len(G5_ALPHA_ADVERSARY_NAMES):
        raise ValidationError("G5 alpha cluster has the wrong shape")
    if values.shape[0] < 2:
        raise ValidationError("G5 alpha cluster requires at least two paths")
    mean = np.mean(values, axis=0, dtype=np.float64)
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    # The 90% TOST family spans nine adversaries at both base frequencies.
    critical = float(norm.ppf(1.0 - 0.05 / (2 * len(G5_ALPHA_ADVERSARY_NAMES))))
    simultaneous = np.abs(mean) + critical * standard_error
    # An exact copied subsequence is intentionally injected into one quarter
    # of paths in the registered power test.  Preserve that path-cluster
    # signal instead of averaging it away; random exact-history hits are null.
    memorization_upper = np.quantile(
        np.abs(values[:, 6:]), G5_CLUSTER_QUANTILE, axis=0, method="higher"
    )
    numerator = max(float(np.max(simultaneous)), float(np.max(memorization_upper)))
    return numerator / G5_DIRECT_ALPHA_SHARPE_MARGIN


def _family_values(
    endpoints: Mapping[str, float],
) -> dict[str, float]:
    groups = {
        "g5_a": ("directional_alpha",),
        "g5_b": ("covariance_correlation", "serial_dependence", "cross_lag"),
        "g5_c": (
            "sliced_wasserstein",
            "energy_distance",
            "spread",
            "spread_wasserstein",
            "spread_rolling",
            "tail_dependence",
        ),
        "g5_d": (
            "location_scale",
            "marginal_tail",
            "marginal_shape",
            "drawdown",
            "source_concentration",
            "repeated_subsequence",
        ),
    }
    result = {
        family: _endpoint_max(endpoints, suffixes)
        for family, suffixes in groups.items()
    }
    return result


def _covariance_to_correlation(covariance: FloatArray) -> FloatArray:
    diagonal = np.diag(covariance)
    if bool((diagonal <= 0.0).any()) or not bool(np.isfinite(diagonal).all()):
        raise ValidationError("G5 covariance has a non-positive marginal variance")
    scale = np.sqrt(diagonal)
    result = covariance / np.outer(scale, scale)
    if not bool(np.isfinite(result).all()):
        raise ValidationError("G5 covariance-to-correlation transform failed")
    return np.asarray(np.clip(result, -1.0, 1.0), dtype=np.float64)


def _spread_distribution_distances(
    candidate: FloatArray,
    reference: FloatArray,
    frequency: Frequency,
) -> tuple[float, float]:
    candidate_spread = candidate[:, 2] - candidate[:, 1]
    reference_spread = reference[:, 2] - reference[:, 1]
    reference_scale = float(np.std(reference_spread, ddof=1))
    if not math.isfinite(reference_scale) or reference_scale <= 1e-12:
        raise ValidationError("G5 spread reference is degenerate")
    wasserstein = (
        float(np.mean(np.abs(np.sort(candidate_spread) - np.sort(reference_spread))))
        / reference_scale
        / 0.10
    )
    horizon = 12 if frequency == "monthly" else 252
    candidate_rolling = _rolling_sums(candidate_spread, horizon)
    reference_rolling = _rolling_sums(reference_spread, horizon)
    rolling_scale = float(np.std(reference_rolling, ddof=1))
    if not math.isfinite(rolling_scale) or rolling_scale <= 1e-12:
        raise ValidationError("G5 rolling-spread reference is degenerate")
    probabilities = (0.10, 0.50, 0.90)
    rolling = (
        float(
            np.max(
                np.abs(
                    np.quantile(candidate_rolling, probabilities, method="higher")
                    - np.quantile(reference_rolling, probabilities, method="higher")
                )
            )
        )
        / rolling_scale
        / 0.15
    )
    return wasserstein, rolling


def _rolling_sums(values: FloatArray, horizon: int) -> FloatArray:
    if values.size < horizon + 1:
        raise ValidationError("G5 rolling endpoint has insufficient support")
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))
    result = cumulative[horizon:] - cumulative[:-horizon]
    if result.size < 2 or not bool(np.isfinite(result).all()):
        raise ValidationError("G5 rolling endpoint is invalid")
    return result


def _marginal_shape_difference(candidate: FloatArray, reference: FloatArray) -> float:
    def moments(values: FloatArray) -> tuple[FloatArray, FloatArray]:
        centered = values - np.mean(values, axis=0)
        second = np.mean(centered**2, axis=0)
        if bool((second <= 1e-12).any()):
            raise ValidationError("G5 marginal-shape reference is degenerate")
        skew = np.mean(centered**3, axis=0) / np.power(second, 1.5)
        kurtosis = np.mean(centered**4, axis=0) / np.square(second)
        return skew, kurtosis

    candidate_skew, candidate_kurtosis = moments(candidate)
    reference_skew, reference_kurtosis = moments(reference)
    return float(
        max(
            np.max(np.abs(candidate_skew - reference_skew)),
            np.max(np.abs(candidate_kurtosis - reference_kurtosis)),
        )
    )


def _tail_dependence_difference(
    candidate: FloatArray,
    reference: FloatArray,
    frequency: Frequency,
) -> float:
    probability = 0.10 if frequency == "monthly" else 0.05
    threshold = np.quantile(reference, probability, axis=0, method="higher")
    candidate_down = candidate <= threshold
    reference_down = reference <= threshold
    difference = max(
        abs(
            float(np.mean(candidate_down[:, left] & candidate_down[:, right]))
            - float(np.mean(reference_down[:, left] & reference_down[:, right]))
        )
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    return difference / 0.02


def _cross_lag_difference(candidate: FloatArray, reference: FloatArray) -> float:
    def block(values: FloatArray) -> FloatArray:
        joined = np.corrcoef(np.column_stack((values[:-1], values[1:])), rowvar=False)
        result = np.asarray(joined[:3, 3:], dtype=np.float64)
        if not bool(np.isfinite(result).all()):
            raise ValidationError("G5 cross-lag endpoint is non-finite")
        return result

    return float(np.max(np.abs(block(candidate) - block(reference))))


def _source_diversity_observables(
    source_indices: ArrayLike | None,
    *,
    observations: int,
    frequency: Frequency,
) -> tuple[float, float]:
    if source_indices is None:
        return 0.0, 0.0
    raw = np.asarray(source_indices)
    if (
        raw.shape != (observations,)
        or raw.dtype.kind not in {"i", "u"}
        or bool((raw < 0).any())
    ):
        raise ValidationError("G5 source-index window is invalid")
    indices = np.asarray(raw, dtype=np.int64)
    _, counts = np.unique(indices, return_counts=True)
    concentration = float(np.max(counts) / observations)
    horizon = 6 if frequency == "monthly" else 63
    starts: list[int] = []
    run_start = 0
    for stop in range(1, observations + 1):
        if stop == observations or indices[stop] != indices[stop - 1] + 1:
            run_length = stop - run_start
            if run_length >= horizon:
                starts.extend(
                    int(value) for value in indices[run_start : stop - horizon + 1]
                )
            run_start = stop
    if not starts:
        repeated = 0.0
    else:
        _, repeated_counts = np.unique(np.asarray(starts), return_counts=True)
        repeated = float((len(starts) - repeated_counts.size) / len(starts))
    return concentration, repeated


def _state_difference(candidate: IntArray, reference: IntArray) -> float:
    maximum_state = max(int(np.max(candidate)), int(np.max(reference)))
    size = maximum_state + 1
    candidate_occupancy = np.bincount(candidate, minlength=size) / candidate.size
    reference_occupancy = np.bincount(reference, minlength=size) / reference.size
    occupancy = float(np.max(np.abs(candidate_occupancy - reference_occupancy)))
    candidate_transition = _transition_frequency(candidate, size)
    reference_transition = _transition_frequency(reference, size)
    transition = float(np.max(np.abs(candidate_transition - reference_transition)))
    return max(occupancy, transition)


def _transition_frequency(states: IntArray, size: int) -> FloatArray:
    counts = np.zeros((size, size), dtype=np.float64)
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    result = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0.0)
    return np.asarray(result, dtype=np.float64)


def _validate_bundle_pair(
    development: G5CandidateBundle, qualification: G5CandidateBundle
) -> None:
    if not isinstance(development, G5CandidateBundle) or not isinstance(
        qualification, G5CandidateBundle
    ):
        raise ValidationError("G5 qualification requires candidate bundles")
    if (
        development.candidate_stage != "development"
        or qualification.candidate_stage != "qualification"
    ):
        raise ValidationError("G5 candidate stages are missing or reversed")
    if (
        development.generation_policy_fingerprint
        != qualification.generation_policy_fingerprint
    ):
        raise IntegrityError("development and qualification policies differ")
    development_sources = {
        item.source_fingerprint
        for item in (*development.lf_paths, *development.hf_paths)
    }
    qualification_sources = {
        item.source_fingerprint
        for item in (*qualification.lf_paths, *qualification.hf_paths)
    }
    if development_sources & qualification_sources:
        raise IntegrityError("development and qualification paths overlap")


def _generation_policy_fingerprint(value: ScientificGenerationPolicy) -> str:
    if not isinstance(value, ScientificGenerationPolicy):
        raise ValidationError("G5 qualification requires a typed generation policy")
    checked = generation_policy_from_mapping(value.as_dict())
    if checked != value:
        raise IntegrityError("G5 generation policy changed during validation")
    return _fingerprint(checked.as_dict())


def _validate_adversary_fit(value: G5AdversaryFitPair) -> G5AdversaryFitPair:
    if (
        not isinstance(value, G5AdversaryFitPair)
        or value.monthly.frequency != "monthly"
        or value.daily.frequency != "daily"
        or value.specification_fingerprint != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
    ):
        raise ValidationError("G5 adversary fit pair has the wrong contract")
    expected = _fingerprint(
        {
            "specification_fingerprint": value.specification_fingerprint,
            "monthly_model_fingerprint": value.monthly.fingerprint,
            "daily_model_fingerprint": value.daily.fingerprint,
        }
    )
    if value.fingerprint != expected:
        raise IntegrityError("G5 adversary fit-pair fingerprint changed")
    return value


def _registered_evaluation_windows(bundle: G5CandidateBundle) -> bool:
    for path in (*bundle.lf_paths, *bundle.hf_paths):
        if (
            path.registered_evaluation_window is not True
            or path.path_entity <= 0
            or path.evaluation_key_fingerprint is None
            or not 0.0 <= path.evaluation_uniform < 1.0
        ):
            return False
        try:
            _sha256(path.evaluation_key_fingerprint, "evaluation-window RNG key")
        except ValidationError:
            return False
    return True


def _exact_bundle_counts(
    development: G5CandidateBundle, qualification: G5CandidateBundle
) -> bool:
    return (
        development.exact_canonical_counts
        and qualification.exact_canonical_counts
        and len(development.lf_paths) == G5_DEVELOPMENT_LF_PATH_COUNT
        and len(development.hf_paths) == G5_DEVELOPMENT_HF_PATH_COUNT
        and len(qualification.lf_paths) == G5_QUALIFICATION_LF_PATH_COUNT
        and len(qualification.hf_paths) == G5_QUALIFICATION_HF_PATH_COUNT
    )


def _upper_empirical_p_value(null: FloatArray, observed: float) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def _cluster_upper(values: Sequence[float]) -> float:
    array = _finite_vector(values, "cluster endpoint values")
    return float(np.quantile(array, G5_CLUSTER_QUANTILE, method="higher"))


def _endpoint_max(endpoints: Mapping[str, float], suffixes: Sequence[str]) -> float:
    selected = [
        value
        for name, value in endpoints.items()
        if any(name.endswith(f".{suffix}") for suffix in suffixes)
    ]
    if not selected:
        raise ValidationError("G5 family has no observable endpoints")
    return max(selected)


def _append(values: dict[str, list[float]], name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValidationError("G5 endpoint is invalid")
    values.setdefault(name, []).append(value)


def _even_indices(observations: int, size: int) -> NDArray[np.intp]:
    if size == observations:
        return np.arange(observations, dtype=np.intp)
    return np.linspace(0, observations - 1, num=size, dtype=np.intp)


def _bounded_count(value: int | None, maximum: int, label: str) -> int:
    if value is None:
        return maximum
    result = _positive_int(value, label)
    if result > maximum:
        raise ValidationError(f"{label} exceeds its G5 stage budget")
    return result


def _bundle_rows(bundle: G5CandidateBundle) -> int:
    return sum(
        item.log_returns.shape[0] for item in (*bundle.lf_paths, *bundle.hf_paths)
    )


def _frequency(value: object) -> Frequency:
    if value not in {"monthly", "daily"}:
        raise ValidationError("G5 reference frequency is invalid")
    return value  # type: ignore[return-value]


def _frozen_returns(value: ArrayLike, *, minimum: int) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError("G5 returns must be numeric") from error
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < minimum:
        raise ValidationError("G5 returns have insufficient three-asset support")
    if not bool(np.isfinite(array).all()):
        raise ValidationError("G5 returns must be finite")
    return _frozen_float_array(array)


def _frozen_states(value: ArrayLike, observations: int) -> IntArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValidationError("G5 states must be numeric") from error
    if raw.shape != (observations,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValidationError("G5 states have the wrong geometry or type")
    array = np.asarray(raw, dtype="<i8", order="C")
    if bool((array < 0).any()):
        raise ValidationError("G5 states must be non-negative")
    return np.frombuffer(array.tobytes(order="C"), dtype="<i8")


def _finite_vector(value: ArrayLike, label: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} must be numeric") from error
    if array.ndim != 1 or array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValidationError(f"{label} must be a non-empty finite vector")
    return np.asarray(array, dtype=np.float64)


def _frozen_float_array(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _fingerprint_mapping(
    value: Mapping[str, str], label: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"{label} fingerprints are required")
    return tuple((name, _sha256(item, label)) for name, item in sorted(value.items()))


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} fingerprint is invalid")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValidationError(f"{label} fingerprint is invalid") from error
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float | np.number)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValidationError(f"{label} must be finite and non-negative")
    return float(value)


def _unit_interval(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float | np.number)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValidationError(f"{label} must be in [0,1)")
    return float(value)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
