"""Proposed size-aware, matched-estimator G5 outer-null rehearsal.

The registered literal estimator compares one exact 500-LF/200-HF pseudo-candidate
cluster with one common, independent, equal-length pseudo-historical reference
at each frequency.  Constructing every history afresh for every cluster would
visit roughly fifty-one billion history rows.  This module implements the
proposed conditional empirical approximation instead:

* purpose-10 artifact 0 supplies one common reference history per replicate;
* purpose-10 artifact 1 supplies one candidate-history reservoir row, one
  outcome-independent joint-sample uniform, and exact with-replacement cluster
  indices per replicate;
* one development-trained current proxy fit scores every reservoir history for
  implementation testing; this scorer is not the approved qualification/G7
  adversary registry; and
* conditional on the materialized candidate reservoir, final rows use
  independent registered reference/cluster draws.

This finite-reservoir scheme is not the owner-approved literal design of
35,000,000 fresh candidate histories and has not been approved as a substitute.
It cannot create canonical thresholds or generation authority.  It exists to
measure the approximation and make the exact-versus-proposed workload
difference explicit.

The large reservoir is a bounded implementation detail.  A rehearsal result
is a no-overwrite float64 matrix plus canonical-JSON manifest and loader;
"canonical" here describes JSON serialization only.  State-chain correctness and
macro-emission/Viterbi adequacy remain separate G3/G5 gates and are not folded
into the raw-return G5-A/B/C/D null.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import ArrayLike, NDArray

from prpg.errors import IntegrityError, ModelError, ValidationError
from prpg.execution import deterministic_process_map
from prpg.model.g5_generation_authority import (
    G5_HISTORICAL_POLICY_SCIENTIFIC_VERSION,
    G5_QUALIFICATION_HF_PATH_COUNT,
    G5_QUALIFICATION_LF_PATH_COUNT,
    G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING,
    G5_THRESHOLD_ESTIMATOR_BINDING,
    G5_THRESHOLD_POLICY_BINDING,
    G5_THRESHOLD_POLICY_STREAM_BINDING,
    g5_policy_scientific_version,
    g5_policy_stream_fingerprint,
)
from prpg.model.scientific_policy import ScientificGenerationPolicy
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)
from prpg.validation.adversaries import (
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    G5AdversaryFitPair,
    score_g5_adversary_path,
)
from prpg.validation.qualification import (
    G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    G5_FAMILY_NAMES,
    G5_JOINT_SAMPLE_ROWS,
    G5_PATH_OBSERVABLE_NAMES,
    G5FamilyNullCalibration,
    G5ReferenceData,
    G5ReferenceObservables,
    aggregate_g5_cluster_matrices,
    build_g5_family_null_calibration,
    build_g5_reference_data,
    compute_g5_path_observables,
    compute_g5_reference_observables,
)
from prpg.validation.sobol import SobolDirectionSet
from prpg.validation.statistics import OUTER_NULL_REPLICATES, holm_decisions

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
Frequency: TypeAlias = Literal["monthly", "daily"]
ExecutionMode: TypeAlias = Literal["rehearsal"]

G5_SIZE_AWARE_NULL_SCHEMA_ID: Final = "prpg-g5-size-aware-null-v1"
G5_SIZE_AWARE_NULL_SCHEMA_VERSION: Final = 1
G5_SIZE_AWARE_RESERVOIR_SCHEMA_ID: Final = "prpg-g5-null-reservoir-v1"
G5_SIZE_AWARE_RESULT_SCHEMA_ID: Final = "prpg-g5-null-result-v1"
G5_SIZE_AWARE_METHOD_ID: Final = (
    "conditional-empirical-common-reference-cluster-bootstrap-v1"
)
G5_NULL_RESULT_MATRIX_NAME: Final = "family-statistics.npy"
G5_NULL_RESULT_MANIFEST_NAME: Final = "null-result-manifest.json"
G5_NULL_RESERVOIR_MANIFEST_NAME: Final = "reservoir-manifest.json"
G5_ADVERSARY_FIT_BINDING: Final = "g5_adversary_fit"
G5_CURRENT_ADVERSARY_SCHEMA_STATUS: Final = (
    "simplified-nine-score-proxy-nonconforming-to-approved-registry"
)

_OBSERVABLE_COUNT: Final = len(G5_PATH_OBSERVABLE_NAMES)
_FINGERPRINT_LENGTH: Final = 64
_MAXIMUM_MEMORY_FRACTION: Final = 0.70
_MAXIMUM_DISK_FRACTION: Final = 0.80
_DEFAULT_SAFETY_FACTOR: Final = 1.25
_RESERVOIR_SEAL: Final = object()
_RUN_SEAL: Final = object()


@dataclass(frozen=True, slots=True)
class G5NullPolicyScope:
    """Selected policy and its distinct G5 scientific stream identity."""

    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    scientific_version: int
    stream_fingerprint: str
    g3_parent_scientific_version: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5SizeAwareNullSubject:
    """Stream-independent subject for the proposed reservoir rehearsal."""

    policy_scope: G5NullPolicyScope
    monthly_reference_fingerprint: str
    daily_reference_fingerprint: str
    raw_monthly_reference_fingerprint: str | None
    raw_daily_reference_fingerprint: str | None
    monthly_continuation_sha256: str
    daily_continuation_sha256: str
    monthly_mean_block_length: float
    daily_mean_block_length: float
    adversary_fit_fingerprint: str
    estimator_contract_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class G5SizeAwareNullReservoir:
    """Producer-minted, immutable manifest for one materialized reservoir."""

    root: Path
    mode: ExecutionMode
    subject: G5SizeAwareNullSubject
    stream_scientific_version: int
    master_seed: int
    replicate_count: int
    lf_cluster_count: int
    hf_cluster_count: int
    monthly_observations: int
    daily_observations: int
    threshold_set_id: str
    binding_fingerprints: tuple[tuple[str, str], ...]
    sobol_direction_fingerprint: str
    files: tuple[tuple[str, str], ...]
    canonical_eligible: bool
    elapsed_seconds: float
    workers: int
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("G5SizeAwareNullReservoir is created only by its producer")


@dataclass(frozen=True, slots=True, init=False)
class G5SizeAwareNullRun:
    """Durable rehearsal matrix plus reconstructed diagnostic calibration."""

    root: Path
    mode: ExecutionMode
    subject_fingerprint: str
    reservoir_fingerprint: str
    policy_stream_fingerprint: str
    adversary_fit_fingerprint: str
    estimator_contract_fingerprint: str
    replicate_count: int
    lf_cluster_count: int
    hf_cluster_count: int
    threshold_set_id: str
    binding_fingerprints: tuple[tuple[str, str], ...]
    sobol_direction_fingerprint: str
    family_statistics_sha256: str
    calibration: G5FamilyNullCalibration
    canonical_eligible: bool
    loaded_from_store: bool
    generation_authorized: Literal[False]
    elapsed_seconds: float
    workers: int
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("G5SizeAwareNullRun is created only by its runner/loader")


@dataclass(frozen=True, slots=True)
class G5SizeAwareWorkerParity:
    """Exact serial/nine-worker equality evidence from a rehearsal stream."""

    serial_reservoir_fingerprint: str
    parallel_reservoir_fingerprint: str
    serial_run_fingerprint: str
    parallel_run_fingerprint: str
    serial_statistics_sha256: str
    parallel_statistics_sha256: str
    serial_workers: int
    parallel_workers: int
    passed: bool
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5SizeAwareResourceProjection:
    """Same-host projection for the unapproved reservoir approximation."""

    subject_fingerprint: str
    host_fingerprint: str
    pilot_replicates: int
    pilot_lf_cluster_count: int
    pilot_hf_cluster_count: int
    target_replicates: int
    target_lf_cluster_count: int
    target_hf_cluster_count: int
    target_workers: int
    pilot_materialization_seconds: float
    pilot_aggregation_seconds: float
    projected_materialization_seconds: float
    projected_aggregation_seconds: float
    safety_factor: float
    projected_total_seconds: float
    projected_total_hours: float
    maximum_projected_hours: float
    estimated_reservoir_bytes: int
    estimated_result_bytes: int
    estimated_peak_memory_bytes: int
    physical_memory_bytes: int
    disk_free_bytes: int
    estimated_memory_fraction: float
    estimated_disk_fraction: float
    worker_parity_fingerprint: str
    worker_parity_passed: bool
    memory_passed: bool
    disk_passed: bool
    time_passed: bool
    passed: bool
    scientific_cost_complete: Literal[False]
    canonical_authorized: Literal[False]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5NullDesignComparison:
    """Literal-exact versus proposed-reservoir workload and optional ETA."""

    outer_replicates: int
    lf_cluster_count: int
    hf_cluster_count: int
    literal_candidate_histories: int
    literal_common_reference_histories: int
    literal_adversary_scored_histories: int
    literal_history_rows: int
    proposed_candidate_histories: int
    proposed_common_reference_histories: int
    proposed_adversary_scored_histories: int
    proposed_history_rows: int
    history_row_reduction_factor: float
    adversary_scoring_reduction_factor: float
    literal_projected_seconds: float | None
    literal_projected_hours: float | None
    proposed_projected_seconds: float | None
    proposed_projected_hours: float | None
    literal_persistent_result_bytes: int
    proposed_reservoir_bytes: int
    proposed_persistent_result_bytes: int
    eta_basis: Literal["unavailable_pending_exact_adversary_registry"]
    proposed_design_authorized: Literal[False]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5MetaStatisticalCallbacks:
    """Nonauthorizing adapter over actual, separately produced suite statistics.

    This type computes empirical p-values and designated-family Holm decisions;
    it cannot certify that a caller actually generated the purpose-12/13 or K3
    suites.  Consequently it is deliberately ineligible for canonical meta
    evidence until a separate typed producer supplies those inputs.
    """

    null_calibration: G5FamilyNullCalibration
    clean_statistics: FloatArray
    defect_statistics: tuple[tuple[str, FloatArray], ...]
    hmm_identification_decisions: tuple[tuple[str, BoolArray], ...]
    replicate_count: int
    materialization_fingerprints: tuple[tuple[str, str], ...]
    canonical_authorized: Literal[False]
    fingerprint: str

    def null_suite(self, replicate: int) -> Mapping[str, float]:
        """Return four empirical p-values for one actual clean statistic row."""

        index = _meta_index(replicate, self.replicate_count)
        return _family_p_values(self.null_calibration, self.clean_statistics[index])

    def power_suite(self, defect: object, replicate: int) -> bool:
        """Evaluate only the registered defect's designated family."""

        from prpg.validation.meta_validation import RegisteredDefect

        if not isinstance(defect, RegisteredDefect):
            raise ValidationError("meta power callback requires a registered defect")
        index = _meta_index(replicate, self.replicate_count)
        if defect.designated_family == "hmm_identification":
            decisions = dict(self.hmm_identification_decisions)
            if defect.claim_name not in decisions:
                raise ValidationError("HMM meta decision is missing")
            return bool(decisions[defect.claim_name][index])
        matrices = dict(self.defect_statistics)
        if defect.claim_name not in matrices:
            raise ValidationError("return meta statistic is missing")
        p_values = _family_p_values(
            self.null_calibration, matrices[defect.claim_name][index]
        )
        holm_by_name = {
            item.name: item.rejected for item in holm_decisions(p_values).decisions
        }
        return bool(holm_by_name[defect.designated_family])


@dataclass(frozen=True, slots=True)
class _ArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class _MaterializationWork:
    root: str
    frequency: Frequency
    source: FloatArray
    raw_source: FloatArray | None
    continuation_links: BoolArray
    mean_block_length: float
    master_seed: int
    stream_scientific_version: int
    replicate_count: int
    cluster_count: int
    start: int
    stop: int
    adversary_fit: G5AdversaryFitPair


@dataclass(frozen=True, slots=True)
class _AggregationWork:
    root: str
    start: int
    stop: int
    replicate_count: int
    lf_cluster_count: int
    hf_cluster_count: int
    monthly_observations: int
    daily_observations: int
    sobol_directions: SobolDirectionSet
    adversary_fit_fingerprint: str
    has_raw_references: bool


@dataclass(frozen=True, slots=True)
class _ChunkReceipt:
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _StatisticsChunk:
    start: int
    stop: int
    statistics: FloatArray


_MINTED_RESERVOIRS: dict[int, G5SizeAwareNullReservoir] = {}
_MINTED_RUNS: dict[int, G5SizeAwareNullRun] = {}


def g5_null_policy_scope(policy: ScientificGenerationPolicy) -> G5NullPolicyScope:
    """Build the exact selected-policy/G3-parent version separation."""

    if not isinstance(policy, ScientificGenerationPolicy):
        raise ValidationError("G5 null policy must be a selected policy")
    version = g5_policy_scientific_version(policy)
    policy_fingerprint = _fingerprint(policy.as_dict())
    stream_fingerprint = g5_policy_stream_fingerprint(policy)
    identity = {
        "generation_policy_fingerprint": policy_fingerprint,
        "scientific_version": version,
        "stream_fingerprint": stream_fingerprint,
        "g3_parent_scientific_version": (G5_HISTORICAL_POLICY_SCIENTIFIC_VERSION),
    }
    return G5NullPolicyScope(
        generation_policy=policy,
        generation_policy_fingerprint=policy_fingerprint,
        scientific_version=version,
        stream_fingerprint=stream_fingerprint,
        g3_parent_scientific_version=G5_HISTORICAL_POLICY_SCIENTIFIC_VERSION,
        fingerprint=_fingerprint(identity),
    )


def build_g5_size_aware_null_subject(
    *,
    generation_policy: ScientificGenerationPolicy,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
    monthly_continuation_links: ArrayLike,
    daily_continuation_links: ArrayLike,
    monthly_mean_block_length: float,
    daily_mean_block_length: float,
    adversary_fit: G5AdversaryFitPair,
    raw_monthly_reference: G5ReferenceData | None = None,
    raw_daily_reference: G5ReferenceData | None = None,
) -> G5SizeAwareNullSubject:
    """Freeze the stream-independent proposed-rehearsal subject."""

    scope = g5_null_policy_scope(generation_policy)
    monthly = _reference(monthly_reference, "monthly")
    daily = _reference(daily_reference, "daily")
    monthly_links = _continuation_links(
        monthly_continuation_links, monthly.log_returns.shape[0]
    )
    daily_links = _continuation_links(
        daily_continuation_links, daily.log_returns.shape[0]
    )
    monthly_block = _mean_block_length(
        monthly_mean_block_length, "monthly mean block length"
    )
    daily_block = _mean_block_length(daily_mean_block_length, "daily mean block length")
    fitted = _adversary_fit(adversary_fit)
    raw_monthly, raw_daily = _raw_references(
        scope,
        monthly,
        daily,
        raw_monthly_reference,
        raw_daily_reference,
    )
    identity = {
        "policy_scope_fingerprint": scope.fingerprint,
        "monthly_reference_fingerprint": monthly.fingerprint,
        "daily_reference_fingerprint": daily.fingerprint,
        "raw_monthly_reference_fingerprint": (
            None if raw_monthly is None else raw_monthly.fingerprint
        ),
        "raw_daily_reference_fingerprint": (
            None if raw_daily is None else raw_daily.fingerprint
        ),
        "monthly_continuation_sha256": _array_sha256(monthly_links),
        "daily_continuation_sha256": _array_sha256(daily_links),
        "monthly_mean_block_length": monthly_block,
        "daily_mean_block_length": daily_block,
        "adversary_fit_fingerprint": fitted.fingerprint,
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    }
    return G5SizeAwareNullSubject(
        policy_scope=scope,
        monthly_reference_fingerprint=monthly.fingerprint,
        daily_reference_fingerprint=daily.fingerprint,
        raw_monthly_reference_fingerprint=(
            None if raw_monthly is None else raw_monthly.fingerprint
        ),
        raw_daily_reference_fingerprint=(
            None if raw_daily is None else raw_daily.fingerprint
        ),
        monthly_continuation_sha256=_array_sha256(monthly_links),
        daily_continuation_sha256=_array_sha256(daily_links),
        monthly_mean_block_length=monthly_block,
        daily_mean_block_length=daily_block,
        adversary_fit_fingerprint=fitted.fingerprint,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        fingerprint=_fingerprint(identity),
    )


def materialize_g5_size_aware_null_reservoir(
    output_directory: str | Path,
    *,
    mode: ExecutionMode,
    generation_policy: ScientificGenerationPolicy,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
    monthly_continuation_links: ArrayLike,
    daily_continuation_links: ArrayLike,
    monthly_mean_block_length: float,
    daily_mean_block_length: float,
    adversary_fit: G5AdversaryFitPair,
    sobol_directions: SobolDirectionSet,
    master_seed: int,
    stream_scientific_version: int,
    threshold_set_id: str,
    binding_fingerprints: Mapping[str, str],
    replicate_count: int,
    lf_cluster_count: int = G5_QUALIFICATION_LF_PATH_COUNT,
    hf_cluster_count: int = G5_QUALIFICATION_HF_PATH_COUNT,
    workers: int = 1,
    batch_size: int | None = None,
    raw_monthly_reference: G5ReferenceData | None = None,
    raw_daily_reference: G5ReferenceData | None = None,
) -> G5SizeAwareNullReservoir:
    """Create one no-overwrite candidate/reference reservoir.

    Rehearsals are forced onto a nonselected scientific-version stream and
    below 50,000 replicates.  The proposed reservoir path cannot inspect the
    selected-policy purpose-10 stream.
    """

    checked_mode = _mode(mode)
    count = _replicate_count(replicate_count)
    lf_count = _positive_int(lf_cluster_count, "LF cluster count")
    hf_count = _positive_int(hf_cluster_count, "HF cluster count")
    worker_count = _positive_int(workers, "workers")
    stream_version = _uint32(stream_scientific_version, "stream scientific version")
    seed = _uint128(master_seed, "master seed")
    threshold_id = _identifier(threshold_set_id, "threshold set ID")
    bindings = _binding_fingerprints(binding_fingerprints)
    fitted = _adversary_fit(adversary_fit)
    subject = build_g5_size_aware_null_subject(
        generation_policy=generation_policy,
        monthly_reference=monthly_reference,
        daily_reference=daily_reference,
        monthly_continuation_links=monthly_continuation_links,
        daily_continuation_links=daily_continuation_links,
        monthly_mean_block_length=monthly_mean_block_length,
        daily_mean_block_length=daily_mean_block_length,
        adversary_fit=fitted,
        raw_monthly_reference=raw_monthly_reference,
        raw_daily_reference=raw_daily_reference,
    )
    _validate_binding_contract(bindings, subject)
    _validate_execution_scope(
        checked_mode,
        stream_version,
        subject,
        count,
        lf_count,
        hf_count,
        bindings,
    )
    if (
        not isinstance(sobol_directions, SobolDirectionSet)
        or sobol_directions.scientific_version != stream_version
    ):
        raise ValidationError("Sobol directions use the wrong execution stream")

    monthly = _reference(monthly_reference, "monthly")
    daily = _reference(daily_reference, "daily")
    monthly_links = _continuation_links(
        monthly_continuation_links, monthly.log_returns.shape[0]
    )
    daily_links = _continuation_links(
        daily_continuation_links, daily.log_returns.shape[0]
    )
    raw_monthly, raw_daily = _raw_references(
        subject.policy_scope,
        monthly,
        daily,
        raw_monthly_reference,
        raw_daily_reference,
    )
    worker_fit = _picklable_adversary_fit(fitted)
    root = Path(output_directory).expanduser().resolve()
    _create_exclusive_directory(root)
    specs = _reservoir_specs(
        count,
        monthly.log_returns.shape[0],
        daily.log_returns.shape[0],
        lf_count,
        hf_count,
        has_raw=raw_monthly is not None,
    )
    try:
        _create_arrays(root, specs)
        size = (
            max(1, math.ceil(count / worker_count))
            if batch_size is None
            else _positive_int(batch_size, "reservoir batch size")
        )
        jobs: list[_MaterializationWork] = []
        for frequency, source, raw, links, block, cluster in (
            (
                "monthly",
                monthly.log_returns,
                None if raw_monthly is None else raw_monthly.log_returns,
                monthly_links,
                subject.monthly_mean_block_length,
                lf_count,
            ),
            (
                "daily",
                daily.log_returns,
                None if raw_daily is None else raw_daily.log_returns,
                daily_links,
                subject.daily_mean_block_length,
                hf_count,
            ),
        ):
            jobs.extend(
                _MaterializationWork(
                    root=str(root),
                    frequency=_frequency(frequency),
                    source=np.asarray(source, dtype=np.float64),
                    raw_source=(
                        None if raw is None else np.asarray(raw, dtype=np.float64)
                    ),
                    continuation_links=links,
                    mean_block_length=block,
                    master_seed=seed,
                    stream_scientific_version=stream_version,
                    replicate_count=count,
                    cluster_count=cluster,
                    start=start,
                    stop=min(start + size, count),
                    adversary_fit=worker_fit,
                )
                for start in range(0, count, size)
            )
        started = time.monotonic()
        receipts = deterministic_process_map(
            _materialize_chunk, jobs, workers=worker_count, chunksize=1
        )
        elapsed = time.monotonic() - started
        _validate_receipts(receipts, jobs, count)
        file_records = _file_records(root, specs)
        manifest = _reservoir_manifest(
            mode=checked_mode,
            subject=subject,
            stream_scientific_version=stream_version,
            master_seed=seed,
            replicate_count=count,
            lf_cluster_count=lf_count,
            hf_cluster_count=hf_count,
            monthly_observations=monthly.log_returns.shape[0],
            daily_observations=daily.log_returns.shape[0],
            threshold_set_id=threshold_id,
            binding_fingerprints=bindings,
            sobol_direction_fingerprint=sobol_directions.fingerprint,
            file_records=file_records,
            canonical_eligible=False,
        )
        _write_new(root / G5_NULL_RESERVOIR_MANIFEST_NAME, _canonical_bytes(manifest))
        result = _mint_reservoir(
            root=root,
            manifest=manifest,
            subject=subject,
            elapsed_seconds=elapsed,
            workers=worker_count,
        )
        return verify_g5_size_aware_null_reservoir(result)
    except Exception:
        # Deliberately no broad recovery/cleanup machinery: an incomplete
        # directory has no manifest and cannot load or authorize a result.
        raise


def run_g5_size_aware_null(
    reservoir: G5SizeAwareNullReservoir,
    output_directory: str | Path,
    *,
    sobol_directions: SobolDirectionSet,
    workers: int = 1,
    batch_size: int | None = None,
) -> G5SizeAwareNullRun:
    """Aggregate one proposed rehearsal matrix and publish it no-overwrite."""

    checked = verify_g5_size_aware_null_reservoir(reservoir)
    worker_count = _positive_int(workers, "workers")
    if (
        not isinstance(sobol_directions, SobolDirectionSet)
        or sobol_directions.fingerprint != checked.sobol_direction_fingerprint
        or sobol_directions.scientific_version != checked.stream_scientific_version
    ):
        raise IntegrityError("null aggregation Sobol directions changed")
    result_root = Path(output_directory).expanduser().resolve()
    _create_exclusive_directory(result_root)
    size = (
        max(1, math.ceil(checked.replicate_count / worker_count))
        if batch_size is None
        else _positive_int(batch_size, "aggregation batch size")
    )
    jobs = tuple(
        _AggregationWork(
            root=str(checked.root),
            start=start,
            stop=min(start + size, checked.replicate_count),
            replicate_count=checked.replicate_count,
            lf_cluster_count=checked.lf_cluster_count,
            hf_cluster_count=checked.hf_cluster_count,
            monthly_observations=checked.monthly_observations,
            daily_observations=checked.daily_observations,
            sobol_directions=sobol_directions,
            adversary_fit_fingerprint=checked.subject.adversary_fit_fingerprint,
            has_raw_references=(
                checked.subject.raw_monthly_reference_fingerprint is not None
            ),
        )
        for start in range(0, checked.replicate_count, size)
    )
    started = time.monotonic()
    chunks = deterministic_process_map(
        _aggregate_chunk, jobs, workers=worker_count, chunksize=1
    )
    elapsed = time.monotonic() - started
    statistics = np.empty((checked.replicate_count, len(G5_FAMILY_NAMES)), dtype="<f8")
    expected = 0
    for chunk in chunks:
        if (
            chunk.start != expected
            or chunk.stop <= chunk.start
            or chunk.statistics.shape
            != (
                chunk.stop - chunk.start,
                len(G5_FAMILY_NAMES),
            )
        ):
            raise IntegrityError("null aggregation chunks are incomplete or reordered")
        statistics[chunk.start : chunk.stop] = chunk.statistics
        expected = chunk.stop
    if expected != checked.replicate_count or not bool(np.isfinite(statistics).all()):
        raise IntegrityError("null aggregation did not produce one finite fixed matrix")
    matrix_path = result_root / G5_NULL_RESULT_MATRIX_NAME
    stored = open_memmap(  # type: ignore[no-untyped-call]
        matrix_path,
        mode="w+",
        dtype="<f8",
        shape=statistics.shape,
        fortran_order=False,
    )
    stored[:] = statistics
    stored.flush()
    del stored
    matrix_file_sha = _file_sha256(matrix_path)
    matrix_values_sha = _array_sha256(statistics)
    bindings = dict(checked.binding_fingerprints)
    calibration = build_g5_family_null_calibration(
        {name: statistics[:, index] for index, name in enumerate(G5_FAMILY_NAMES)},
        threshold_set_id=checked.threshold_set_id,
        binding_fingerprints=bindings,
        sobol_direction_fingerprint=checked.sobol_direction_fingerprint,
    )
    if (
        calibration.fixed_sample_canonical
        or calibration.canonical_thresholds is not None
    ):
        raise IntegrityError(
            "proposed reservoir rehearsal created canonical thresholds"
        )
    manifest = _result_manifest(
        reservoir=checked,
        matrix_file_sha256=matrix_file_sha,
        matrix_values_sha256=matrix_values_sha,
        calibration=calibration,
    )
    _write_new(result_root / G5_NULL_RESULT_MANIFEST_NAME, _canonical_bytes(manifest))
    result = _mint_run(
        root=result_root,
        manifest=manifest,
        calibration=calibration,
        elapsed_seconds=elapsed,
        workers=worker_count,
        loaded_from_store=False,
    )
    return verify_g5_size_aware_null_run(result)


def load_g5_size_aware_null_result(
    directory: str | Path,
    *,
    expected_subject_fingerprint: str,
    expected_policy_stream_fingerprint: str,
    expected_adversary_fit_fingerprint: str,
    expected_sobol_direction_fingerprint: str,
    expected_binding_fingerprints: Mapping[str, str],
) -> G5SizeAwareNullRun:
    """Load the compact matrix for later-process rehearsal inspection only."""

    root = Path(directory).expanduser().resolve()
    manifest = _read_manifest(root / G5_NULL_RESULT_MANIFEST_NAME)
    if manifest.get("schema_id") != G5_SIZE_AWARE_RESULT_SCHEMA_ID:
        raise IntegrityError("G5 null result manifest schema is invalid")
    for actual, expected, label in (
        (manifest.get("subject_fingerprint"), expected_subject_fingerprint, "subject"),
        (
            manifest.get("policy_stream_fingerprint"),
            expected_policy_stream_fingerprint,
            "policy stream",
        ),
        (
            manifest.get("adversary_fit_fingerprint"),
            expected_adversary_fit_fingerprint,
            "adversary fit",
        ),
        (
            manifest.get("sobol_direction_fingerprint"),
            expected_sobol_direction_fingerprint,
            "Sobol directions",
        ),
        (
            manifest.get("estimator_contract_fingerprint"),
            G5_ESTIMATOR_CONTRACT_FINGERPRINT,
            "estimator",
        ),
    ):
        if actual != _sha256(expected, label):
            raise IntegrityError(f"G5 null result {label} differs from expected")
    expected_bindings = _binding_fingerprints(expected_binding_fingerprints)
    manifest_bindings = _binding_fingerprints(
        dict(_pairs(manifest.get("binding_fingerprints"), "result bindings"))
    )
    if manifest_bindings != expected_bindings:
        raise IntegrityError("G5 null result binding subjects changed")
    matrix_path = root / G5_NULL_RESULT_MATRIX_NAME
    if _file_sha256(matrix_path) != manifest.get("matrix_file_sha256"):
        raise IntegrityError("G5 null result matrix file changed")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    count = _positive_int(manifest.get("replicate_count"), "replicate count")
    if matrix.shape != (count, len(G5_FAMILY_NAMES)) or matrix.dtype.str != "<f8":
        raise IntegrityError("G5 null result matrix geometry changed")
    if _array_sha256(matrix) != manifest.get("family_statistics_sha256"):
        raise IntegrityError("G5 null result matrix values changed")
    calibration = build_g5_family_null_calibration(
        {name: matrix[:, index] for index, name in enumerate(G5_FAMILY_NAMES)},
        threshold_set_id=_identifier(manifest.get("threshold_set_id"), "threshold ID"),
        binding_fingerprints=dict(manifest_bindings),
        sobol_direction_fingerprint=_sha256(
            manifest.get("sobol_direction_fingerprint"), "Sobol directions"
        ),
    )
    if calibration.fingerprint != manifest.get("calibration_fingerprint"):
        raise IntegrityError("G5 null reconstructed calibration changed")
    result = _mint_run(
        root=root,
        manifest=manifest,
        calibration=calibration,
        elapsed_seconds=0.0,
        workers=0,
        loaded_from_store=True,
    )
    return verify_g5_size_aware_null_run(result)


def compare_g5_size_aware_worker_parity(
    serial_reservoir: G5SizeAwareNullReservoir,
    serial_run: G5SizeAwareNullRun,
    parallel_reservoir: G5SizeAwareNullReservoir,
    parallel_run: G5SizeAwareNullRun,
) -> G5SizeAwareWorkerParity:
    """Require exact one-versus-nine equality on a disjoint rehearsal stream."""

    first_reservoir = verify_g5_size_aware_null_reservoir(serial_reservoir)
    second_reservoir = verify_g5_size_aware_null_reservoir(parallel_reservoir)
    first_run = verify_g5_size_aware_null_run(serial_run)
    second_run = verify_g5_size_aware_null_run(parallel_run)
    if (
        first_reservoir.mode != "rehearsal"
        or second_reservoir.mode != "rehearsal"
        or first_reservoir.workers != 1
        or second_reservoir.workers != 9
        or first_run.workers != 1
        or second_run.workers != 9
    ):
        raise ValidationError("worker parity requires one- and nine-worker rehearsals")
    passed = (
        first_reservoir.fingerprint == second_reservoir.fingerprint
        and first_run.fingerprint == second_run.fingerprint
        and first_run.family_statistics_sha256 == second_run.family_statistics_sha256
    )
    identity = {
        "serial_reservoir_fingerprint": first_reservoir.fingerprint,
        "parallel_reservoir_fingerprint": second_reservoir.fingerprint,
        "serial_run_fingerprint": first_run.fingerprint,
        "parallel_run_fingerprint": second_run.fingerprint,
        "serial_statistics_sha256": first_run.family_statistics_sha256,
        "parallel_statistics_sha256": second_run.family_statistics_sha256,
        "serial_workers": 1,
        "parallel_workers": 9,
        "passed": passed,
    }
    return G5SizeAwareWorkerParity(
        serial_reservoir_fingerprint=first_reservoir.fingerprint,
        parallel_reservoir_fingerprint=second_reservoir.fingerprint,
        serial_run_fingerprint=first_run.fingerprint,
        parallel_run_fingerprint=second_run.fingerprint,
        serial_statistics_sha256=first_run.family_statistics_sha256,
        parallel_statistics_sha256=second_run.family_statistics_sha256,
        serial_workers=1,
        parallel_workers=9,
        passed=passed,
        fingerprint=_fingerprint(identity),
    )


def project_g5_size_aware_null_resources(
    pilot_reservoir: G5SizeAwareNullReservoir,
    pilot_run: G5SizeAwareNullRun,
    worker_parity: G5SizeAwareWorkerParity,
    *,
    host_fingerprint: str,
    physical_memory_bytes: int,
    disk_free_bytes: int,
    target_workers: int = 9,
    maximum_projected_hours: float = 72.0,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
) -> G5SizeAwareResourceProjection:
    """Project the proposed 50k reservoir from a retired-stream host pilot."""

    reservoir = verify_g5_size_aware_null_reservoir(pilot_reservoir)
    run = verify_g5_size_aware_null_run(pilot_run)
    if reservoir.mode != "rehearsal" or run.mode != "rehearsal":
        raise ValidationError("resource projection requires a noncanonical rehearsal")
    if run.reservoir_fingerprint != reservoir.fingerprint:
        raise IntegrityError("resource pilot run uses a different reservoir")
    if not worker_parity.passed:
        raise ModelError("resource projection requires passed one-vs-nine equality")
    memory = _positive_int(physical_memory_bytes, "physical memory bytes")
    disk = _positive_int(disk_free_bytes, "disk free bytes")
    workers = _positive_int(target_workers, "target workers")
    maximum_hours = _positive_finite(maximum_projected_hours, "maximum hours")
    factor = _positive_finite(safety_factor, "safety factor")
    if factor < 1.0:
        raise ValidationError("resource safety factor must be at least one")
    if reservoir.elapsed_seconds <= 0.0 or run.elapsed_seconds <= 0.0:
        raise ValidationError("resource pilot requires measured positive durations")
    count_scale = OUTER_NULL_REPLICATES / reservoir.replicate_count
    pilot_cost = _aggregation_cost(
        reservoir.lf_cluster_count,
        reservoir.hf_cluster_count,
        reservoir.monthly_observations,
        reservoir.daily_observations,
    )
    target_cost = _aggregation_cost(
        G5_QUALIFICATION_LF_PATH_COUNT,
        G5_QUALIFICATION_HF_PATH_COUNT,
        reservoir.monthly_observations,
        reservoir.daily_observations,
    )
    projected_materialization = reservoir.elapsed_seconds * count_scale
    projected_aggregation = run.elapsed_seconds * count_scale * target_cost / pilot_cost
    projected_total = factor * (projected_materialization + projected_aggregation)
    has_raw = reservoir.subject.raw_monthly_reference_fingerprint is not None
    reservoir_bytes = estimate_g5_size_aware_reservoir_bytes(
        replicate_count=OUTER_NULL_REPLICATES,
        monthly_observations=reservoir.monthly_observations,
        daily_observations=reservoir.daily_observations,
        lf_cluster_count=G5_QUALIFICATION_LF_PATH_COUNT,
        hf_cluster_count=G5_QUALIFICATION_HF_PATH_COUNT,
        has_raw_references=has_raw,
    )
    result_bytes = OUTER_NULL_REPLICATES * len(G5_FAMILY_NAMES) * 8 + 64 * 1024
    # Mmaps bound persistent storage; resident working-set allowance includes a
    # 2-GiB file-cache window, one 256-MiB scientific worker allowance, and a
    # 512-MiB coordinator/BLAS/Python reserve.
    peak = min(reservoir_bytes, 2 * 1024**3) + workers * 256 * 1024**2 + 512 * 1024**2
    memory_fraction = peak / memory
    disk_fraction = (reservoir_bytes + result_bytes) / disk
    memory_passed = memory_fraction <= _MAXIMUM_MEMORY_FRACTION
    disk_passed = disk_fraction <= _MAXIMUM_DISK_FRACTION
    time_passed = projected_total / 3600.0 <= maximum_hours
    passed = worker_parity.passed and memory_passed and disk_passed and time_passed
    identity = {
        "subject_fingerprint": reservoir.subject.fingerprint,
        "host_fingerprint": _sha256(host_fingerprint, "host"),
        "pilot_replicates": reservoir.replicate_count,
        "pilot_lf_cluster_count": reservoir.lf_cluster_count,
        "pilot_hf_cluster_count": reservoir.hf_cluster_count,
        "target_replicates": OUTER_NULL_REPLICATES,
        "target_lf_cluster_count": G5_QUALIFICATION_LF_PATH_COUNT,
        "target_hf_cluster_count": G5_QUALIFICATION_HF_PATH_COUNT,
        "target_workers": workers,
        "pilot_materialization_seconds": reservoir.elapsed_seconds,
        "pilot_aggregation_seconds": run.elapsed_seconds,
        "projected_materialization_seconds": projected_materialization,
        "projected_aggregation_seconds": projected_aggregation,
        "safety_factor": factor,
        "projected_total_seconds": projected_total,
        "projected_total_hours": projected_total / 3600.0,
        "maximum_projected_hours": maximum_hours,
        "estimated_reservoir_bytes": reservoir_bytes,
        "estimated_result_bytes": result_bytes,
        "estimated_peak_memory_bytes": peak,
        "physical_memory_bytes": memory,
        "disk_free_bytes": disk,
        "estimated_memory_fraction": memory_fraction,
        "estimated_disk_fraction": disk_fraction,
        "worker_parity_fingerprint": worker_parity.fingerprint,
        "worker_parity_passed": worker_parity.passed,
        "memory_passed": memory_passed,
        "disk_passed": disk_passed,
        "time_passed": time_passed,
        "passed": passed,
        "scientific_cost_complete": False,
        "canonical_authorized": False,
    }
    return G5SizeAwareResourceProjection(
        subject_fingerprint=reservoir.subject.fingerprint,
        host_fingerprint=_sha256(host_fingerprint, "host"),
        pilot_replicates=reservoir.replicate_count,
        pilot_lf_cluster_count=reservoir.lf_cluster_count,
        pilot_hf_cluster_count=reservoir.hf_cluster_count,
        target_replicates=OUTER_NULL_REPLICATES,
        target_lf_cluster_count=G5_QUALIFICATION_LF_PATH_COUNT,
        target_hf_cluster_count=G5_QUALIFICATION_HF_PATH_COUNT,
        target_workers=workers,
        pilot_materialization_seconds=reservoir.elapsed_seconds,
        pilot_aggregation_seconds=run.elapsed_seconds,
        projected_materialization_seconds=projected_materialization,
        projected_aggregation_seconds=projected_aggregation,
        safety_factor=factor,
        projected_total_seconds=projected_total,
        projected_total_hours=projected_total / 3600.0,
        maximum_projected_hours=maximum_hours,
        estimated_reservoir_bytes=reservoir_bytes,
        estimated_result_bytes=result_bytes,
        estimated_peak_memory_bytes=peak,
        physical_memory_bytes=memory,
        disk_free_bytes=disk,
        estimated_memory_fraction=memory_fraction,
        estimated_disk_fraction=disk_fraction,
        worker_parity_fingerprint=worker_parity.fingerprint,
        worker_parity_passed=worker_parity.passed,
        memory_passed=memory_passed,
        disk_passed=disk_passed,
        time_passed=time_passed,
        passed=passed,
        scientific_cost_complete=False,
        canonical_authorized=False,
        fingerprint=_fingerprint(identity),
    )


def estimate_g5_size_aware_reservoir_bytes(
    *,
    replicate_count: int,
    monthly_observations: int,
    daily_observations: int,
    lf_cluster_count: int,
    hf_cluster_count: int,
    has_raw_references: bool,
) -> int:
    """Return the exact numeric payload plus conservative ``.npy`` headers."""

    count = _positive_int(replicate_count, "replicate count")
    monthly = _positive_int(monthly_observations, "monthly observations")
    daily = _positive_int(daily_observations, "daily observations")
    lf = _positive_int(lf_cluster_count, "LF cluster count")
    hf = _positive_int(hf_cluster_count, "HF cluster count")
    if type(has_raw_references) is not bool:
        raise ValidationError("raw-reference flag must be boolean")
    primary_observable = 4 * count * _OBSERVABLE_COUNT * 8
    primary_joint = (
        count
        * (min(G5_JOINT_SAMPLE_ROWS, monthly) + min(G5_JOINT_SAMPLE_ROWS, daily))
        * 3
        * 8
    )
    indices = count * (lf + hf) * 2
    source_fingerprints = 2 * count * _FINGERPRINT_LENGTH
    raw = 0
    if has_raw_references:
        raw = 2 * count * _OBSERVABLE_COUNT * 8 + primary_joint + source_fingerprints
    # At most twelve arrays plus one small manifest.
    return (
        primary_observable
        + primary_joint
        + indices
        + source_fingerprints
        + raw
        + 128 * 1024
    )


def compare_literal_and_proposed_g5_null_designs(
    *,
    monthly_observations: int,
    daily_observations: int,
    has_raw_references: bool = False,
) -> G5NullDesignComparison:
    """Compare exact workloads without timing the nonconforming adversary proxy.

    Literal exact mode uses one already-frozen development fit, then exactly
    50,000 * (500 LF + 200 HF) fresh candidate histories plus one common LF
    and one common HF reference per replicate.  No per-replicate development
    refit is counted.  Wall-clock ETA remains unavailable until the approved
    adversary registry replaces the current simplified nine-score proxy and a
    same-host retired-stream pilot measures that exact scorer.
    """

    monthly = _positive_int(monthly_observations, "monthly observations")
    daily = _positive_int(daily_observations, "daily observations")
    if type(has_raw_references) is not bool:
        raise ValidationError("raw-reference flag must be boolean")
    literal_candidates = OUTER_NULL_REPLICATES * (
        G5_QUALIFICATION_LF_PATH_COUNT + G5_QUALIFICATION_HF_PATH_COUNT
    )
    literal_references = 2 * OUTER_NULL_REPLICATES
    literal_scored = literal_candidates + literal_references
    literal_rows = OUTER_NULL_REPLICATES * (
        G5_QUALIFICATION_LF_PATH_COUNT * monthly
        + G5_QUALIFICATION_HF_PATH_COUNT * daily
        + monthly
        + daily
    )
    proposed_candidates = 2 * OUTER_NULL_REPLICATES
    proposed_references = 2 * OUTER_NULL_REPLICATES
    proposed_scored = proposed_candidates + proposed_references
    proposed_rows = 2 * OUTER_NULL_REPLICATES * (monthly + daily)
    result_bytes = OUTER_NULL_REPLICATES * len(G5_FAMILY_NAMES) * 8 + 64 * 1024
    reservoir_bytes = estimate_g5_size_aware_reservoir_bytes(
        replicate_count=OUTER_NULL_REPLICATES,
        monthly_observations=monthly,
        daily_observations=daily,
        lf_cluster_count=G5_QUALIFICATION_LF_PATH_COUNT,
        hf_cluster_count=G5_QUALIFICATION_HF_PATH_COUNT,
        has_raw_references=has_raw_references,
    )
    identity = {
        "outer_replicates": OUTER_NULL_REPLICATES,
        "lf_cluster_count": G5_QUALIFICATION_LF_PATH_COUNT,
        "hf_cluster_count": G5_QUALIFICATION_HF_PATH_COUNT,
        "literal_candidate_histories": literal_candidates,
        "literal_common_reference_histories": literal_references,
        "literal_adversary_scored_histories": literal_scored,
        "literal_history_rows": literal_rows,
        "proposed_candidate_histories": proposed_candidates,
        "proposed_common_reference_histories": proposed_references,
        "proposed_adversary_scored_histories": proposed_scored,
        "proposed_history_rows": proposed_rows,
        "history_row_reduction_factor": literal_rows / proposed_rows,
        "adversary_scoring_reduction_factor": (literal_scored / proposed_scored),
        "literal_projected_seconds": None,
        "literal_projected_hours": None,
        "proposed_projected_seconds": None,
        "proposed_projected_hours": None,
        "literal_persistent_result_bytes": result_bytes,
        "proposed_reservoir_bytes": reservoir_bytes,
        "proposed_persistent_result_bytes": result_bytes,
        "eta_basis": "unavailable_pending_exact_adversary_registry",
        "proposed_design_authorized": False,
    }
    return G5NullDesignComparison(
        outer_replicates=OUTER_NULL_REPLICATES,
        lf_cluster_count=G5_QUALIFICATION_LF_PATH_COUNT,
        hf_cluster_count=G5_QUALIFICATION_HF_PATH_COUNT,
        literal_candidate_histories=literal_candidates,
        literal_common_reference_histories=literal_references,
        literal_adversary_scored_histories=literal_scored,
        literal_history_rows=literal_rows,
        proposed_candidate_histories=proposed_candidates,
        proposed_common_reference_histories=proposed_references,
        proposed_adversary_scored_histories=proposed_scored,
        proposed_history_rows=proposed_rows,
        history_row_reduction_factor=literal_rows / proposed_rows,
        adversary_scoring_reduction_factor=(literal_scored / proposed_scored),
        literal_projected_seconds=None,
        literal_projected_hours=None,
        proposed_projected_seconds=None,
        proposed_projected_hours=None,
        literal_persistent_result_bytes=result_bytes,
        proposed_reservoir_bytes=reservoir_bytes,
        proposed_persistent_result_bytes=result_bytes,
        eta_basis="unavailable_pending_exact_adversary_registry",
        proposed_design_authorized=False,
        fingerprint=_fingerprint(identity),
    )


def build_nonauthorizing_g5_meta_callbacks(
    null_calibration: G5FamilyNullCalibration,
    *,
    clean_statistics: ArrayLike,
    defect_statistics: Mapping[str, ArrayLike],
    hmm_identification_decisions: Mapping[str, ArrayLike],
    materialization_fingerprints: Mapping[str, str],
) -> G5MetaStatisticalCallbacks:
    """Build concrete p-value/Holm callbacks without fabricating producer proof."""

    from prpg.validation.meta_validation import G5_DEFECT_REGISTRY

    if not isinstance(null_calibration, G5FamilyNullCalibration):
        raise ValidationError("meta callbacks require typed G5 null calibration")
    clean = _statistics_matrix(clean_statistics, "clean meta statistics")
    count = clean.shape[0]
    expected_return = {
        item.claim_name
        for item in G5_DEFECT_REGISTRY
        if item.designated_family != "hmm_identification"
    }
    expected_hmm = {
        item.claim_name
        for item in G5_DEFECT_REGISTRY
        if item.designated_family == "hmm_identification"
    }
    if set(defect_statistics) != expected_return:
        raise ValidationError("meta return-statistic claims are incomplete")
    if set(hmm_identification_decisions) != expected_hmm:
        raise ValidationError("meta HMM-decision claims are incomplete")
    defects = tuple(
        (name, _statistics_matrix(defect_statistics[name], f"{name} statistics"))
        for name in sorted(expected_return)
    )
    if any(matrix.shape[0] != count for _, matrix in defects):
        raise ValidationError("meta return-statistic replicate counts differ")
    hmm: list[tuple[str, BoolArray]] = []
    for name in sorted(expected_hmm):
        values = np.asarray(hmm_identification_decisions[name])
        if values.shape != (count,) or values.dtype.kind != "b":
            raise ValidationError("meta HMM decisions must be boolean vectors")
        frozen = np.frombuffer(
            np.asarray(values, dtype=np.bool_).tobytes(), dtype=np.bool_
        )
        hmm.append((name, frozen))
    materializations = _binding_fingerprints(materialization_fingerprints)
    identity = {
        "null_calibration_fingerprint": null_calibration.fingerprint,
        "clean_statistics_sha256": _array_sha256(clean),
        "defect_statistics_sha256": {
            name: _array_sha256(matrix) for name, matrix in defects
        },
        "hmm_decision_sha256": {name: _array_sha256(values) for name, values in hmm},
        "replicate_count": count,
        "materialization_fingerprints": list(materializations),
        "canonical_authorized": False,
    }
    return G5MetaStatisticalCallbacks(
        null_calibration=null_calibration,
        clean_statistics=clean,
        defect_statistics=defects,
        hmm_identification_decisions=tuple(hmm),
        replicate_count=count,
        materialization_fingerprints=materializations,
        canonical_authorized=False,
        fingerprint=_fingerprint(identity),
    )


def verify_g5_size_aware_null_reservoir(
    value: object,
) -> G5SizeAwareNullReservoir:
    """Rehash an original producer-minted reservoir and every persisted array."""

    if (
        type(value) is not G5SizeAwareNullReservoir
        or value._seal is not _RESERVOIR_SEAL
        or value._owner_id != id(value)
        or _MINTED_RESERVOIRS.get(id(value)) is not value
    ):
        raise ModelError("G5 null reservoir lacks producer provenance")
    manifest = _read_manifest(value.root / G5_NULL_RESERVOIR_MANIFEST_NAME)
    if _fingerprint(manifest) != value.fingerprint:
        raise IntegrityError("G5 null reservoir manifest changed")
    records = _manifest_file_records(manifest)
    for name, digest in records:
        if _file_sha256(value.root / name) != digest:
            raise IntegrityError(f"G5 null reservoir array changed: {name}")
    return value


def verify_g5_size_aware_null_run(value: object) -> G5SizeAwareNullRun:
    """Rehash one original or canonical-JSON-loaded rehearsal result."""

    if (
        type(value) is not G5SizeAwareNullRun
        or value._seal is not _RUN_SEAL
        or value._owner_id != id(value)
        or _MINTED_RUNS.get(id(value)) is not value
    ):
        raise ModelError("G5 size-aware null result lacks loader provenance")
    manifest = _read_manifest(value.root / G5_NULL_RESULT_MANIFEST_NAME)
    if _fingerprint(manifest) != value.fingerprint:
        raise IntegrityError("G5 null result manifest changed")
    matrix_path = value.root / G5_NULL_RESULT_MATRIX_NAME
    if _file_sha256(matrix_path) != manifest.get("matrix_file_sha256"):
        raise IntegrityError("G5 null result matrix file changed")
    return value


def _materialize_chunk(work: _MaterializationWork) -> _ChunkReceipt:
    root = Path(work.root)
    prefix = "lf" if work.frequency == "monthly" else "hf"
    candidate = np.load(root / f"{prefix}-candidate-observables.npy", mmap_mode="r+")
    reference_values = np.load(
        root / f"{prefix}-reference-observables.npy", mmap_mode="r+"
    )
    reference_joint = np.load(root / f"{prefix}-reference-joint.npy", mmap_mode="r+")
    reference_source = np.load(
        root / f"{prefix}-reference-source-fingerprints.npy", mmap_mode="r+"
    )
    cluster_indices = np.load(root / f"{prefix}-cluster-indices.npy", mmap_mode="r+")
    raw_values: NDArray[Any] | None = None
    raw_joint: NDArray[Any] | None = None
    raw_source: NDArray[Any] | None = None
    if work.raw_source is not None:
        raw_values = np.load(
            root / f"{prefix}-raw-reference-observables.npy", mmap_mode="r+"
        )
        raw_joint = np.load(root / f"{prefix}-raw-reference-joint.npy", mmap_mode="r+")
        raw_source = np.load(
            root / f"{prefix}-raw-reference-source-fingerprints.npy", mmap_mode="r+"
        )
    family = Family.LF if work.frequency == "monthly" else Family.HF
    model = (
        work.adversary_fit.monthly
        if work.frequency == "monthly"
        else work.adversary_fit.daily
    )
    observations = work.source.shape[0]
    probability = 1.0 / work.mean_block_length
    for replicate in range(work.start, work.stop):
        histories: list[IntArray] = []
        candidate_suffix: FloatArray | None = None
        for artifact in (
            CalibrationArtifact.DESIGN_OR_CONTROL,
            CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        ):
            key = calibration_rng_key(
                master_seed=work.master_seed,
                scientific_version=work.stream_scientific_version,
                purpose=CalibrationPurpose.G5_OUTER_NULL,
                artifact=artifact,
                variant=0,
                replicate=replicate,
                domain=Domain.THRESHOLD_CALIBRATION,
                family=family,
                stage=Stage.REFERENCE_RESAMPLING,
            )
            suffix = 0
            if artifact is CalibrationArtifact.PRODUCTION_OR_CANDIDATE:
                suffix = 1 + work.cluster_count
            draws = np.asarray(
                key.generator().random(observations * 2 + suffix), dtype=np.float64
            )
            paired = draws[: observations * 2].reshape(observations, 2)
            histories.append(
                _bootstrap_indices(
                    observations,
                    work.continuation_links,
                    paired[:, 0],
                    paired[:, 1],
                    probability,
                )
            )
            if suffix:
                candidate_suffix = draws[observations * 2 :]
        assert candidate_suffix is not None
        reference_history = work.source[histories[0]]
        candidate_history = work.source[histories[1]]
        reference_data = build_g5_reference_data(
            reference_history,
            np.zeros(observations, dtype=np.int64),
            frequency=work.frequency,
        )
        reference_observable = compute_g5_reference_observables(reference_data)
        alpha = score_g5_adversary_path(model, candidate_history)
        candidate_observable = compute_g5_path_observables(
            candidate_history,
            frequency=work.frequency,
            alpha_sharpes=alpha,
            source_indices=histories[1],
            selection_uniform=float(candidate_suffix[0]),
            path_rank=replicate + 1,
        )
        candidate[replicate] = candidate_observable.values
        reference_values[replicate] = reference_observable.values
        reference_joint[replicate] = reference_observable.joint_sample
        reference_source[replicate] = reference_observable.source_fingerprint.encode()
        cluster_indices[replicate] = np.floor(
            candidate_suffix[1:] * work.replicate_count
        ).astype(np.uint16)
        if (
            work.raw_source is not None
            and raw_values is not None
            and raw_joint is not None
            and raw_source is not None
        ):
            raw_data = build_g5_reference_data(
                work.raw_source[histories[0]],
                np.zeros(observations, dtype=np.int64),
                frequency=work.frequency,
            )
            raw_observable = compute_g5_reference_observables(raw_data)
            raw_values[replicate] = raw_observable.values
            raw_joint[replicate] = raw_observable.joint_sample
            raw_source[replicate] = raw_observable.source_fingerprint.encode()
    for value in (
        candidate,
        reference_values,
        reference_joint,
        reference_source,
        cluster_indices,
        raw_values,
        raw_joint,
        raw_source,
    ):
        if value is not None and hasattr(value, "flush"):
            value.flush()
    return _ChunkReceipt(work.start, work.stop)


def _aggregate_chunk(work: _AggregationWork) -> _StatisticsChunk:
    root = Path(work.root)
    lf_candidate = np.load(root / "lf-candidate-observables.npy", mmap_mode="r")
    hf_candidate = np.load(root / "hf-candidate-observables.npy", mmap_mode="r")
    lf_indices = np.load(root / "lf-cluster-indices.npy", mmap_mode="r")
    hf_indices = np.load(root / "hf-cluster-indices.npy", mmap_mode="r")
    lf_reference = np.load(root / "lf-reference-observables.npy", mmap_mode="r")
    hf_reference = np.load(root / "hf-reference-observables.npy", mmap_mode="r")
    lf_joint = np.load(root / "lf-reference-joint.npy", mmap_mode="r")
    hf_joint = np.load(root / "hf-reference-joint.npy", mmap_mode="r")
    lf_source = np.load(root / "lf-reference-source-fingerprints.npy", mmap_mode="r")
    hf_source = np.load(root / "hf-reference-source-fingerprints.npy", mmap_mode="r")
    raw_arrays: tuple[NDArray[Any], ...] | None = None
    if work.has_raw_references:
        raw_arrays = (
            np.load(root / "lf-raw-reference-observables.npy", mmap_mode="r"),
            np.load(root / "hf-raw-reference-observables.npy", mmap_mode="r"),
            np.load(root / "lf-raw-reference-joint.npy", mmap_mode="r"),
            np.load(root / "hf-raw-reference-joint.npy", mmap_mode="r"),
            np.load(root / "lf-raw-reference-source-fingerprints.npy", mmap_mode="r"),
            np.load(root / "hf-raw-reference-source-fingerprints.npy", mmap_mode="r"),
        )
    output = np.empty((work.stop - work.start, len(G5_FAMILY_NAMES)), dtype=np.float64)
    for row, replicate in enumerate(range(work.start, work.stop)):
        monthly_reference = _reference_observable_from_arrays(
            "monthly",
            work.monthly_observations,
            lf_reference[replicate],
            lf_joint[replicate],
            bytes(lf_source[replicate]).decode(),
        )
        daily_reference = _reference_observable_from_arrays(
            "daily",
            work.daily_observations,
            hf_reference[replicate],
            hf_joint[replicate],
            bytes(hf_source[replicate]).decode(),
        )
        raw_monthly: G5ReferenceObservables | None = None
        raw_daily: G5ReferenceObservables | None = None
        if raw_arrays is not None:
            raw_monthly = _reference_observable_from_arrays(
                "monthly",
                work.monthly_observations,
                raw_arrays[0][replicate],
                raw_arrays[2][replicate],
                bytes(raw_arrays[4][replicate]).decode(),
            )
            raw_daily = _reference_observable_from_arrays(
                "daily",
                work.daily_observations,
                raw_arrays[1][replicate],
                raw_arrays[3][replicate],
                bytes(raw_arrays[5][replicate]).decode(),
            )
        metrics = aggregate_g5_cluster_matrices(
            lf_candidate[np.asarray(lf_indices[replicate], dtype=np.int64)],
            hf_candidate[np.asarray(hf_indices[replicate], dtype=np.int64)],
            monthly_reference,
            daily_reference,
            sobol_directions=work.sobol_directions,
            expected_lf=work.lf_cluster_count,
            expected_hf=work.hf_cluster_count,
            latent_chain_cap_ratio=0.0,
            latent_chain_evaluated=False,
            adversary_fit_fingerprint=work.adversary_fit_fingerprint,
            duplicate_copies=0,
            raw_monthly_reference=raw_monthly,
            raw_daily_reference=raw_daily,
        )
        values = dict(metrics.family_statistics)
        output[row] = [values[name] for name in G5_FAMILY_NAMES]
    output.setflags(write=False)
    return _StatisticsChunk(work.start, work.stop, output)


def _reservoir_specs(
    count: int,
    monthly_observations: int,
    daily_observations: int,
    lf_cluster_count: int,
    hf_cluster_count: int,
    *,
    has_raw: bool,
) -> tuple[_ArraySpec, ...]:
    specs: list[_ArraySpec] = []
    for prefix, observations, cluster in (
        ("lf", monthly_observations, lf_cluster_count),
        ("hf", daily_observations, hf_cluster_count),
    ):
        specs.extend(
            (
                _ArraySpec(
                    f"{prefix}-candidate-observables.npy",
                    (count, _OBSERVABLE_COUNT),
                    "<f8",
                ),
                _ArraySpec(
                    f"{prefix}-reference-observables.npy",
                    (count, _OBSERVABLE_COUNT),
                    "<f8",
                ),
                _ArraySpec(
                    f"{prefix}-reference-joint.npy",
                    (count, min(G5_JOINT_SAMPLE_ROWS, observations), 3),
                    "<f8",
                ),
                _ArraySpec(
                    f"{prefix}-reference-source-fingerprints.npy",
                    (count,),
                    "|S64",
                ),
                _ArraySpec(f"{prefix}-cluster-indices.npy", (count, cluster), "<u2"),
            )
        )
        if has_raw:
            specs.extend(
                (
                    _ArraySpec(
                        f"{prefix}-raw-reference-observables.npy",
                        (count, _OBSERVABLE_COUNT),
                        "<f8",
                    ),
                    _ArraySpec(
                        f"{prefix}-raw-reference-joint.npy",
                        (count, min(G5_JOINT_SAMPLE_ROWS, observations), 3),
                        "<f8",
                    ),
                    _ArraySpec(
                        f"{prefix}-raw-reference-source-fingerprints.npy",
                        (count,),
                        "|S64",
                    ),
                )
            )
    return tuple(specs)


def _create_arrays(root: Path, specs: Sequence[_ArraySpec]) -> None:
    for spec in specs:
        value = open_memmap(  # type: ignore[no-untyped-call]
            root / spec.name,
            mode="w+",
            dtype=spec.dtype,
            shape=spec.shape,
            fortran_order=False,
        )
        value.flush()
        del value


def _file_records(
    root: Path, specs: Sequence[_ArraySpec]
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": spec.name,
            "shape": list(spec.shape),
            "dtype": np.dtype(spec.dtype).str,
            "sha256": _file_sha256(root / spec.name),
        }
        for spec in specs
    )


def _reservoir_manifest(
    *,
    mode: ExecutionMode,
    subject: G5SizeAwareNullSubject,
    stream_scientific_version: int,
    master_seed: int,
    replicate_count: int,
    lf_cluster_count: int,
    hf_cluster_count: int,
    monthly_observations: int,
    daily_observations: int,
    threshold_set_id: str,
    binding_fingerprints: tuple[tuple[str, str], ...],
    sobol_direction_fingerprint: str,
    file_records: tuple[dict[str, object], ...],
    canonical_eligible: bool,
) -> dict[str, object]:
    return {
        "schema_id": G5_SIZE_AWARE_RESERVOIR_SCHEMA_ID,
        "schema_version": G5_SIZE_AWARE_NULL_SCHEMA_VERSION,
        "method_id": G5_SIZE_AWARE_METHOD_ID,
        "mode": mode,
        "subject": _subject_dict(subject),
        "subject_fingerprint": subject.fingerprint,
        "stream_scientific_version": stream_scientific_version,
        "master_seed": master_seed,
        "replicate_count": replicate_count,
        "lf_cluster_count": lf_cluster_count,
        "hf_cluster_count": hf_cluster_count,
        "monthly_observations": monthly_observations,
        "daily_observations": daily_observations,
        "threshold_set_id": threshold_set_id,
        "binding_fingerprints": [list(item) for item in binding_fingerprints],
        "sobol_direction_fingerprint": sobol_direction_fingerprint,
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        "adversary_fit_fingerprint": subject.adversary_fit_fingerprint,
        "adversary_schema_status": G5_CURRENT_ADVERSARY_SCHEMA_STATUS,
        "approved_endpoint_compatible": False,
        "files": list(file_records),
        "conditional_inference": True,
        "common_reference_per_frequency": True,
        "candidate_resampling": "with_replacement_from_fixed_reservoir",
        "state_families_included": False,
        "canonical_eligible": canonical_eligible,
        "generation_authorized": False,
    }


def _result_manifest(
    *,
    reservoir: G5SizeAwareNullReservoir,
    matrix_file_sha256: str,
    matrix_values_sha256: str,
    calibration: G5FamilyNullCalibration,
) -> dict[str, object]:
    return {
        "schema_id": G5_SIZE_AWARE_RESULT_SCHEMA_ID,
        "schema_version": G5_SIZE_AWARE_NULL_SCHEMA_VERSION,
        "method_id": G5_SIZE_AWARE_METHOD_ID,
        "mode": reservoir.mode,
        "subject_fingerprint": reservoir.subject.fingerprint,
        "reservoir_fingerprint": reservoir.fingerprint,
        "policy_stream_fingerprint": reservoir.subject.policy_scope.stream_fingerprint,
        "stream_scientific_version": reservoir.stream_scientific_version,
        "adversary_fit_fingerprint": reservoir.subject.adversary_fit_fingerprint,
        "adversary_schema_status": G5_CURRENT_ADVERSARY_SCHEMA_STATUS,
        "approved_endpoint_compatible": False,
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        "replicate_count": reservoir.replicate_count,
        "lf_cluster_count": reservoir.lf_cluster_count,
        "hf_cluster_count": reservoir.hf_cluster_count,
        "threshold_set_id": reservoir.threshold_set_id,
        "binding_fingerprints": [list(item) for item in reservoir.binding_fingerprints],
        "sobol_direction_fingerprint": reservoir.sobol_direction_fingerprint,
        "family_names": list(G5_FAMILY_NAMES),
        "matrix_name": G5_NULL_RESULT_MATRIX_NAME,
        "matrix_shape": [reservoir.replicate_count, len(G5_FAMILY_NAMES)],
        "matrix_dtype": "<f8",
        "matrix_file_sha256": matrix_file_sha256,
        "family_statistics_sha256": matrix_values_sha256,
        "calibration_fingerprint": calibration.fingerprint,
        "canonical_threshold_fingerprint": None,
        "canonical_eligible": reservoir.canonical_eligible,
        "state_families_included": False,
        "generation_authorized": False,
    }


def _mint_reservoir(
    *,
    root: Path,
    manifest: Mapping[str, object],
    subject: G5SizeAwareNullSubject,
    elapsed_seconds: float,
    workers: int,
) -> G5SizeAwareNullReservoir:
    value = object.__new__(G5SizeAwareNullReservoir)
    attributes = {
        "root": root,
        "mode": manifest["mode"],
        "subject": subject,
        "stream_scientific_version": manifest["stream_scientific_version"],
        "master_seed": manifest["master_seed"],
        "replicate_count": manifest["replicate_count"],
        "lf_cluster_count": manifest["lf_cluster_count"],
        "hf_cluster_count": manifest["hf_cluster_count"],
        "monthly_observations": manifest["monthly_observations"],
        "daily_observations": manifest["daily_observations"],
        "threshold_set_id": manifest["threshold_set_id"],
        "binding_fingerprints": _pairs(
            manifest["binding_fingerprints"], "reservoir bindings"
        ),
        "sobol_direction_fingerprint": manifest["sobol_direction_fingerprint"],
        "files": _manifest_file_records(manifest),
        "canonical_eligible": manifest["canonical_eligible"],
        "elapsed_seconds": elapsed_seconds,
        "workers": workers,
        "fingerprint": _fingerprint(manifest),
        "_seal": _RESERVOIR_SEAL,
        "_owner_id": id(value),
    }
    for name, attribute in attributes.items():
        object.__setattr__(value, name, attribute)
    _MINTED_RESERVOIRS[id(value)] = value
    return value


def _mint_run(
    *,
    root: Path,
    manifest: Mapping[str, object],
    calibration: G5FamilyNullCalibration,
    elapsed_seconds: float,
    workers: int,
    loaded_from_store: bool,
) -> G5SizeAwareNullRun:
    value = object.__new__(G5SizeAwareNullRun)
    attributes = {
        "root": root,
        "mode": manifest["mode"],
        "subject_fingerprint": manifest["subject_fingerprint"],
        "reservoir_fingerprint": manifest["reservoir_fingerprint"],
        "policy_stream_fingerprint": manifest["policy_stream_fingerprint"],
        "adversary_fit_fingerprint": manifest["adversary_fit_fingerprint"],
        "estimator_contract_fingerprint": manifest["estimator_contract_fingerprint"],
        "replicate_count": manifest["replicate_count"],
        "lf_cluster_count": manifest["lf_cluster_count"],
        "hf_cluster_count": manifest["hf_cluster_count"],
        "threshold_set_id": manifest["threshold_set_id"],
        "binding_fingerprints": _pairs(
            manifest["binding_fingerprints"], "result bindings"
        ),
        "sobol_direction_fingerprint": manifest["sobol_direction_fingerprint"],
        "family_statistics_sha256": manifest["family_statistics_sha256"],
        "calibration": calibration,
        "canonical_eligible": manifest["canonical_eligible"],
        "loaded_from_store": loaded_from_store,
        "generation_authorized": False,
        "elapsed_seconds": elapsed_seconds,
        "workers": workers,
        "fingerprint": _fingerprint(manifest),
        "_seal": _RUN_SEAL,
        "_owner_id": id(value),
    }
    for name, attribute in attributes.items():
        object.__setattr__(value, name, attribute)
    _MINTED_RUNS[id(value)] = value
    return value


def _reference_observable_from_arrays(
    frequency: Frequency,
    observations: int,
    values: ArrayLike,
    joint_sample: ArrayLike,
    source_fingerprint: str,
) -> G5ReferenceObservables:
    row = np.asarray(values, dtype=np.float64)
    joint = np.asarray(joint_sample, dtype=np.float64)
    source = _sha256(source_fingerprint, "pseudo-reference source")
    identity = {
        "frequency": frequency,
        "observations": observations,
        "observable_names": list(G5_PATH_OBSERVABLE_NAMES),
        "values_sha256": _array_sha256(row),
        "joint_sample_sha256": _array_sha256(joint),
        "source_fingerprint": source,
        "estimator_contract_fingerprint": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    }
    return G5ReferenceObservables(
        frequency=frequency,
        observations=observations,
        observable_names=G5_PATH_OBSERVABLE_NAMES,
        values=row,
        joint_sample=joint,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        source_fingerprint=source,
        fingerprint=_fingerprint(identity),
    )


def _bootstrap_indices(
    source_observations: int,
    continuation_links: BoolArray,
    restart_uniforms: FloatArray,
    source_uniforms: FloatArray,
    restart_probability: float,
) -> IntArray:
    """Reason-free equivalent of the audited outer-null trace constructor."""

    indices = np.empty(restart_uniforms.size, dtype=np.int64)
    for position in range(restart_uniforms.size):
        restart = position == 0
        if position > 0:
            next_source = int(indices[position - 1]) + 1
            restart = (
                next_source >= source_observations
                or not continuation_links[next_source - 1]
                or restart_uniforms[position] < restart_probability
            )
        if restart:
            indices[position] = math.floor(
                float(source_uniforms[position]) * source_observations
            )
        else:
            indices[position] = indices[position - 1] + 1
    return indices


def _validate_execution_scope(
    mode: ExecutionMode,
    stream_version: int,
    subject: G5SizeAwareNullSubject,
    replicate_count: int,
    lf_cluster_count: int,
    hf_cluster_count: int,
    bindings: tuple[tuple[str, str], ...],
) -> None:
    del mode, lf_cluster_count, hf_cluster_count, bindings
    selected = subject.policy_scope.scientific_version
    if stream_version == selected:
        raise ModelError(
            "reservoir rehearsal cannot inspect the selected-policy purpose-10 stream"
        )
    if replicate_count == OUTER_NULL_REPLICATES:
        raise ModelError("reservoir rehearsal cannot consume the canonical count")


def _validate_binding_contract(
    bindings: tuple[tuple[str, str], ...], subject: G5SizeAwareNullSubject
) -> None:
    values = dict(bindings)
    required = {
        G5_THRESHOLD_ESTIMATOR_BINDING: G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        G5_THRESHOLD_POLICY_BINDING: (
            subject.policy_scope.generation_policy_fingerprint
        ),
        G5_THRESHOLD_POLICY_STREAM_BINDING: subject.policy_scope.stream_fingerprint,
        G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING: (
            G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        ),
        G5_ADVERSARY_FIT_BINDING: subject.adversary_fit_fingerprint,
    }
    for name, expected in required.items():
        if values.get(name) != expected:
            raise ModelError(f"G5 null binding is missing or wrong: {name}")


def _validate_receipts(
    receipts: Sequence[_ChunkReceipt],
    jobs: Sequence[_MaterializationWork],
    count: int,
) -> None:
    for frequency in ("monthly", "daily"):
        expected = 0
        selected = [
            receipt
            for receipt, job in zip(receipts, jobs, strict=True)
            if job.frequency == frequency
        ]
        for receipt in selected:
            if receipt.start != expected or receipt.stop <= receipt.start:
                raise IntegrityError("reservoir chunks are incomplete or reordered")
            expected = receipt.stop
        if expected != count:
            raise IntegrityError("reservoir frequency is incomplete")


def _manifest_file_records(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    raw = manifest.get("files")
    if not isinstance(raw, list):
        raise IntegrityError("reservoir file records are missing")
    result: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "shape",
            "dtype",
            "sha256",
        }:
            raise IntegrityError("reservoir file record is invalid")
        name = item["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise IntegrityError("reservoir file name is unsafe")
        result.append((name, _sha256(item["sha256"], "reservoir file")))
    return tuple(result)


def _subject_dict(value: G5SizeAwareNullSubject) -> dict[str, object]:
    return {
        "policy_scope_fingerprint": value.policy_scope.fingerprint,
        "generation_policy_fingerprint": (
            value.policy_scope.generation_policy_fingerprint
        ),
        "policy_scientific_version": value.policy_scope.scientific_version,
        "policy_stream_fingerprint": value.policy_scope.stream_fingerprint,
        "g3_parent_scientific_version": (
            value.policy_scope.g3_parent_scientific_version
        ),
        "monthly_reference_fingerprint": value.monthly_reference_fingerprint,
        "daily_reference_fingerprint": value.daily_reference_fingerprint,
        "raw_monthly_reference_fingerprint": value.raw_monthly_reference_fingerprint,
        "raw_daily_reference_fingerprint": value.raw_daily_reference_fingerprint,
        "monthly_continuation_sha256": value.monthly_continuation_sha256,
        "daily_continuation_sha256": value.daily_continuation_sha256,
        "monthly_mean_block_length": value.monthly_mean_block_length,
        "daily_mean_block_length": value.daily_mean_block_length,
        "adversary_fit_fingerprint": value.adversary_fit_fingerprint,
        "estimator_contract_fingerprint": value.estimator_contract_fingerprint,
        "fingerprint": value.fingerprint,
    }


def _aggregation_cost(
    lf_count: int, hf_count: int, monthly_observations: int, daily_observations: int
) -> float:
    result = 0.0
    for count, observations in (
        (lf_count, monthly_observations),
        (hf_count, daily_observations),
    ):
        sample = min(G5_JOINT_SAMPLE_ROWS, count, observations)
        result += count * _OBSERVABLE_COUNT + sample * 512 * 3 + sample * sample * 3
    return result


def _family_p_values(
    calibration: G5FamilyNullCalibration, observed: ArrayLike
) -> dict[str, float]:
    values = np.asarray(observed, dtype=np.float64)
    if values.shape != (len(G5_FAMILY_NAMES),) or not bool(np.isfinite(values).all()):
        raise ValidationError("meta family statistic row is invalid")
    count = calibration.family_statistics.shape[0]
    return {
        name: (
            1.0
            + float(
                np.count_nonzero(
                    calibration.family_statistics[:, index] >= values[index]
                )
            )
        )
        / (count + 1.0)
        for index, name in enumerate(G5_FAMILY_NAMES)
    }


def _statistics_matrix(value: ArrayLike, label: str) -> FloatArray:
    try:
        matrix = np.asarray(value, dtype="<f8", order="C")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} must be numeric") from error
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] != len(G5_FAMILY_NAMES)
        or not bool(np.isfinite(matrix).all())
        or bool((matrix < 0.0).any())
    ):
        raise ValidationError(f"{label} has the wrong finite nonnegative shape")
    return np.frombuffer(matrix.tobytes(order="C"), dtype="<f8").reshape(matrix.shape)


def _meta_index(value: object, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise ValidationError("meta replicate is outside its materialization")
    return value


def _reference(value: object, frequency: Frequency) -> G5ReferenceData:
    if not isinstance(value, G5ReferenceData) or value.frequency != frequency:
        raise ValidationError(f"G5 null requires a typed {frequency} reference")
    return value


def _raw_references(
    scope: G5NullPolicyScope,
    monthly: G5ReferenceData,
    daily: G5ReferenceData,
    raw_monthly: G5ReferenceData | None,
    raw_daily: G5ReferenceData | None,
) -> tuple[G5ReferenceData | None, G5ReferenceData | None]:
    neutral = scope.generation_policy.alpha_policy == "kernel_mean_neutral"
    if neutral != (raw_monthly is not None and raw_daily is not None):
        raise ValidationError(
            "neutral policy requires both raw references; baseline requires neither"
        )
    if not neutral:
        return None, None
    checked_monthly = _reference(raw_monthly, "monthly")
    checked_daily = _reference(raw_daily, "daily")
    if (
        checked_monthly.log_returns.shape != monthly.log_returns.shape
        or checked_daily.log_returns.shape != daily.log_returns.shape
    ):
        raise ValidationError("neutral raw/reference histories are not row-aligned")
    return checked_monthly, checked_daily


def _adversary_fit(value: object) -> G5AdversaryFitPair:
    if (
        not isinstance(value, G5AdversaryFitPair)
        or value.monthly.frequency != "monthly"
        or value.daily.frequency != "daily"
        or value.specification_fingerprint != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        or value.fingerprint
        != _fingerprint(
            {
                "specification_fingerprint": G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
                "monthly_model_fingerprint": value.monthly.fingerprint,
                "daily_model_fingerprint": value.daily.fingerprint,
            }
        )
    ):
        raise ValidationError("G5 null adversary fit is invalid")
    return value


def _picklable_adversary_fit(value: G5AdversaryFitPair) -> G5AdversaryFitPair:
    """Replace read-only mapping proxies with equivalent worker-safe mappings."""

    monthly = replace(
        value.monthly, memorized_outcomes=dict(value.monthly.memorized_outcomes)
    )
    daily = replace(
        value.daily, memorized_outcomes=dict(value.daily.memorized_outcomes)
    )
    return replace(value, monthly=monthly, daily=daily)


def _continuation_links(value: ArrayLike, observations: int) -> BoolArray:
    array = np.asarray(value)
    if array.shape != (observations - 1,) or array.dtype.kind != "b":
        raise ValidationError("G5 null continuation links have the wrong shape")
    return np.asarray(array, dtype=np.bool_)


def _mean_block_length(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 1.0
    ):
        raise ValidationError(f"{label} must be finite and greater than one")
    return float(value)


def _binding_fingerprints(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError("G5 null binding fingerprints are required")
    result: list[tuple[str, str]] = []
    for name, fingerprint in value.items():
        if not isinstance(name, str) or not name:
            raise ValidationError("G5 null binding name is invalid")
        result.append((name, _sha256(fingerprint, f"binding {name}")))
    return tuple(sorted(result))


def _pairs(value: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list | tuple):
        raise IntegrityError(f"{label} are invalid")
    result: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise IntegrityError(f"{label} are invalid")
        result.append((item[0], _sha256(item[1], label)))
    if tuple(sorted(result)) != tuple(result):
        raise IntegrityError(f"{label} are not sorted")
    return tuple(result)


def _mode(value: object) -> ExecutionMode:
    if value != "rehearsal":
        raise ValidationError("G5 null execution mode is invalid")
    return value  # type: ignore[return-value]


def _frequency(value: object) -> Frequency:
    if value not in {"monthly", "daily"}:
        raise ValidationError("G5 null frequency is invalid")
    return value  # type: ignore[return-value]


def _replicate_count(value: object) -> int:
    count = _positive_int(value, "replicate count")
    if count > OUTER_NULL_REPLICATES:
        raise ValidationError("G5 null replicate count exceeds 50,000")
    return count


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _uint32(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 32
    ):
        raise ValidationError(f"{label} must be an unsigned 32-bit integer")
    return value


def _uint128(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 128
    ):
        raise ValidationError(f"{label} must be an unsigned 128-bit integer")
    return value


def _positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValidationError(f"{label} must be positive and finite")
    return float(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValidationError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} fingerprint is invalid")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValidationError(f"{label} fingerprint is invalid") from error
    return value


def _array_sha256(value: ArrayLike) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise IntegrityError(f"cannot read G5 null artifact: {path.name}") from error
    return digest.hexdigest()


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


def _create_exclusive_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ModelError(f"G5 null output already exists: {path}") from error
    except OSError as error:
        raise ModelError(f"cannot create G5 null output: {path}") from error


def _write_new(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ModelError(f"G5 null artifact already exists: {path}") from error
    except OSError as error:
        raise ModelError(f"cannot publish G5 null artifact: {path}") from error


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"cannot load G5 null manifest: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise IntegrityError("G5 null manifest is not canonical JSON")
    return value
