"""Lean scientific revalidation of canonical production return CSVs.

G7 does not recalibrate thresholds, refit adversaries, or call the generator.
It consumes immutable G5 identities, the frozen development fit, the exact
50,000-replicate family null, and base-frequency production paths delivered by
the streaming structural validator.  G5 and G7 use the same typed Section-14
path-row and cluster estimator.

The primary inferential subset is fixed at ``LF000001..LF000500`` and
``HF000001..HF000200``.  Metrics over all 5,000/1,000 paths are secondary
summaries and never rescue a primary failure.  Latent-state and source-index
endpoints are absent from consumer CSVs, so G7 excludes them from recomputed
metrics and binds their exact passed G5/G3 evidence.  They are never zero-filled.

The standalone publisher can create only ``g7-scientific-pass.json``.  The
coordinated publisher verifies that this typed scientific pass and the typed
structural report came from the same scan, then writes both reports and creates
``DATA_VALIDATED`` last.  This module never creates ``COMPLETE`` or ``RELEASED``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from prpg.errors import IntegrityError, ValidationError
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Stage,
    calibration_rng_key,
)
from prpg.simulation.rng import (
    Family as RngFamily,
)
from prpg.validation.adversaries import (
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    G5AdversaryFitPair,
    score_g5_adversary_path,
)
from prpg.validation.metrics import PathDiversityAccumulator
from prpg.validation.qualification import (
    G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    G5_FAMILY_NAMES,
    G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING,
    G5_THRESHOLD_ESTIMATOR_BINDING,
    G5FamilyNullCalibration,
    G5PathObservables,
    G5ReferenceData,
    G5ReferenceObservables,
    aggregate_g5_cluster_observables,
    build_g5_family_null_calibration,
    build_g5_reference_data,
    compute_g5_path_observables,
    compute_g5_reference_observables,
)
from prpg.validation.records import (
    CanonicalThresholds,
    CanonicalValidationResults,
    ValidationDecision,
    build_canonical_results,
    canonical_results_from_dict,
    canonical_thresholds_from_dict,
)
from prpg.validation.sobol import (
    SOBOL_DIMENSION,
    SOBOL_DIRECTION_COUNT,
    SOBOL_METHOD,
    SobolDirectionSet,
)
from prpg.validation.statistics import OUTER_NULL_REPLICATES, holm_decisions
from prpg.validation.structural import (
    CANONICAL_STRUCTURAL_PLAN,
    DATA_VALIDATED_SCHEMA_ID,
    DATA_VALIDATED_SCHEMA_VERSION,
    JSONValue,
    StructuralPathView,
    StructuralValidationReport,
    is_sealed_structural_validation_report,
)

FloatArray: TypeAlias = NDArray[np.float64]
Frequency: TypeAlias = Literal["monthly", "daily"]
Family: TypeAlias = Literal["LF", "HF"]

G7_SCIENTIFIC_CONTRACT_SCHEMA_ID: Final = "prpg-g7-scientific-contract-v1"
G7_SCIENTIFIC_PLAN_SCHEMA_ID: Final = "prpg-g7-scientific-plan-v1"
G7_SCIENTIFIC_ASSESSMENT_SCHEMA_ID: Final = "prpg-g7-scientific-assessment-v1"
G7_SCIENTIFIC_PASS_SCHEMA_ID: Final = "prpg-g7-scientific-pass-v1"
G7_SCIENTIFIC_SCHEMA_VERSION: Final = 1
G7_PRIMARY_LF_PATHS: Final = 500
G7_PRIMARY_HF_PATHS: Final = 200
G7_ALL_LF_PATHS: Final = 5_000
G7_ALL_HF_PATHS: Final = 1_000
G7_SCIENTIFIC_HOOK_NAME: Final = "g7_scientific"
G7_SCIENTIFIC_REPORT_FILENAME: Final = "g7-scientific-pass.json"
G7_ESTIMATOR_CONTRACT_ID: Final = G5_ESTIMATOR_CONTRACT_FINGERPRINT
G7_INHERITED_STATE_SOURCE_SCHEMA_ID: Final = (
    "prpg-g7-inherited-g5-state-source-evidence-v1"
)
G7_CSV_EXCLUDED_ENDPOINTS: Final = (
    "source_concentration",
    "repeated_subsequence",
    "latent_chain_correctness",
)
G7_THRESHOLD_ADVERSARY_FIT_BINDING: Final = "g5_adversary_fit"

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT_SEAL = object()
_PASS_SEAL = object()


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


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValidationError(f"G7 {label} fingerprint is invalid")
    return value


def _validated_run_root(value: str | Path) -> Path:
    root = Path(value)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise IntegrityError("G7 run root is missing or unsafe")
    for component in (root, *root.parents):
        if component.is_symlink():
            raise IntegrityError("G7 run root contains a symbolic link")
    return root


def _scientific_report_bytes(report: G7ScientificPassReport) -> bytes:
    return _canonical_bytes(
        {"fingerprint": report.fingerprint, "identity": report.identity_dict()}
    )


def _publish_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            written = handle.write(content)
            if written != len(content):
                raise IntegrityError("G7 publication write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise IntegrityError("G7 publication overwrite refused") from error


@dataclass(frozen=True, slots=True, init=False)
class G7FrozenScientificContract:
    """Verified G5 identities and numerical inputs reused without tuning."""

    schema_id: str
    schema_version: int
    production_authority_fingerprint: str
    thresholds: CanonicalThresholds
    qualification_results: CanonicalValidationResults
    null_calibration: G5FamilyNullCalibration
    monthly_reference: G5ReferenceData
    daily_reference: G5ReferenceData
    monthly_reference_observables: G5ReferenceObservables
    daily_reference_observables: G5ReferenceObservables
    raw_monthly_reference_observables: G5ReferenceObservables | None
    raw_daily_reference_observables: G5ReferenceObservables | None
    sobol_directions: SobolDirectionSet
    adversary_fit: G5AdversaryFitPair
    estimator_contract_fingerprint: str
    evaluation_master_seed: int
    policy_scientific_version: int
    qualification_observed_metrics_fingerprint: str
    g3_production_latent_task_fingerprint: str
    g3_production_latent_checkpoint_fingerprint: str
    g3_production_adequacy_task_fingerprint: str
    g3_production_adequacy_checkpoint_fingerprint: str
    inherited_state_source_evidence_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "production_authority_fingerprint": (self.production_authority_fingerprint),
            "threshold_fingerprint": self.thresholds.fingerprint,
            "qualification_results_fingerprint": (
                self.qualification_results.fingerprint
            ),
            "null_calibration_fingerprint": self.null_calibration.fingerprint,
            "monthly_reference_fingerprint": self.monthly_reference.fingerprint,
            "daily_reference_fingerprint": self.daily_reference.fingerprint,
            "monthly_reference_observables_fingerprint": (
                self.monthly_reference_observables.fingerprint
            ),
            "daily_reference_observables_fingerprint": (
                self.daily_reference_observables.fingerprint
            ),
            "raw_monthly_reference_observables_fingerprint": (
                None
                if self.raw_monthly_reference_observables is None
                else self.raw_monthly_reference_observables.fingerprint
            ),
            "raw_daily_reference_observables_fingerprint": (
                None
                if self.raw_daily_reference_observables is None
                else self.raw_daily_reference_observables.fingerprint
            ),
            "sobol_direction_fingerprint": self.sobol_directions.fingerprint,
            "adversary_fit_fingerprint": self.adversary_fit.fingerprint,
            "estimator_contract_fingerprint": self.estimator_contract_fingerprint,
            "evaluation_master_seed": self.evaluation_master_seed,
            "policy_scientific_version": self.policy_scientific_version,
            "qualification_observed_metrics_fingerprint": (
                self.qualification_observed_metrics_fingerprint
            ),
            "g3_production_latent_task_fingerprint": (
                self.g3_production_latent_task_fingerprint
            ),
            "g3_production_latent_checkpoint_fingerprint": (
                self.g3_production_latent_checkpoint_fingerprint
            ),
            "g3_production_adequacy_task_fingerprint": (
                self.g3_production_adequacy_task_fingerprint
            ),
            "g3_production_adequacy_checkpoint_fingerprint": (
                self.g3_production_adequacy_checkpoint_fingerprint
            ),
            "inherited_state_source_evidence_fingerprint": (
                self.inherited_state_source_evidence_fingerprint
            ),
        }


@dataclass(frozen=True, slots=True)
class G7ScientificValidationPlan:
    """Expected stream geometry; reduced plans are test/rehearsal-only."""

    profile_id: str
    expected_lf_paths: int
    expected_hf_paths: int
    primary_lf_paths: int
    primary_hf_paths: int
    collect_all_path_secondary: bool = True
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", self.profile_id) is None
        ):
            raise ValidationError("G7 scientific profile ID is invalid")
        for name in (
            "expected_lf_paths",
            "expected_hf_paths",
            "primary_lf_paths",
            "primary_hf_paths",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"G7 scientific {name} must be positive")
        if (
            self.primary_lf_paths > self.expected_lf_paths
            or self.primary_hf_paths > self.expected_hf_paths
        ):
            raise ValidationError("G7 primary subset exceeds the available paths")
        if type(self.collect_all_path_secondary) is not bool:
            raise ValidationError("G7 secondary-summary flag must be boolean")
        object.__setattr__(self, "fingerprint", _fingerprint(self.identity_dict()))

    @property
    def canonical_profile(self) -> bool:
        return self == CANONICAL_G7_SCIENTIFIC_PLAN

    @property
    def canonical_primary_counts(self) -> bool:
        return (
            self.primary_lf_paths == G7_PRIMARY_LF_PATHS
            and self.primary_hf_paths == G7_PRIMARY_HF_PATHS
        )

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": G7_SCIENTIFIC_PLAN_SCHEMA_ID,
            "profile_id": self.profile_id,
            "expected_lf_paths": self.expected_lf_paths,
            "expected_hf_paths": self.expected_hf_paths,
            "primary_lf_paths": self.primary_lf_paths,
            "primary_hf_paths": self.primary_hf_paths,
            "collect_all_path_secondary": self.collect_all_path_secondary,
        }


CANONICAL_G7_SCIENTIFIC_PLAN: Final = G7ScientificValidationPlan(
    profile_id="canonical-50y-v1",
    expected_lf_paths=G7_ALL_LF_PATHS,
    expected_hf_paths=G7_ALL_HF_PATHS,
    primary_lf_paths=G7_PRIMARY_LF_PATHS,
    primary_hf_paths=G7_PRIMARY_HF_PATHS,
)


def reduced_g7_scientific_plan(
    *,
    expected_lf_paths: int,
    expected_hf_paths: int,
    primary_lf_paths: int,
    primary_hf_paths: int,
    collect_all_path_secondary: bool = True,
) -> G7ScientificValidationPlan:
    """Create a visibly noncanonical plan for fast fixture/rehearsal tests."""

    plan = G7ScientificValidationPlan(
        profile_id="reduced-fixture-v1",
        expected_lf_paths=expected_lf_paths,
        expected_hf_paths=expected_hf_paths,
        primary_lf_paths=primary_lf_paths,
        primary_hf_paths=primary_hf_paths,
        collect_all_path_secondary=collect_all_path_secondary,
    )
    if plan.canonical_profile or plan.canonical_primary_counts:
        raise ValidationError("a reduced G7 plan cannot use canonical primary counts")
    return plan


@dataclass(frozen=True, slots=True)
class G7ObservableMetrics:
    """Cluster summaries over one fixed production-path population."""

    lf_paths: int
    hf_paths: int
    endpoint_values: tuple[tuple[str, float], ...]
    family_statistics: tuple[tuple[str, float], ...]
    direct_alpha_cap_ratio: float
    raw_hard_cap_ratio: float
    duplicate_copies: int
    csv_excluded_endpoints: tuple[str, ...]
    inherited_state_source_evidence_fingerprint: str
    estimator_contract_fingerprint: str
    adversary_fit_fingerprint: str
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "lf_paths": self.lf_paths,
            "hf_paths": self.hf_paths,
            "endpoint_values": dict(self.endpoint_values),
            "family_statistics": dict(self.family_statistics),
            "direct_alpha_cap_ratio": self.direct_alpha_cap_ratio,
            "raw_hard_cap_ratio": self.raw_hard_cap_ratio,
            "duplicate_copies": self.duplicate_copies,
            "csv_excluded_endpoints": list(self.csv_excluded_endpoints),
            "inherited_state_source_evidence_fingerprint": (
                self.inherited_state_source_evidence_fingerprint
            ),
            "estimator_contract_fingerprint": self.estimator_contract_fingerprint,
            "adversary_fit_fingerprint": self.adversary_fit_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class G7ScientificAssessment:
    """Deterministic diagnostic assessment; only canonical passes are publishable."""

    schema_id: str
    schema_version: int
    contract_fingerprint: str
    plan_fingerprint: str
    primary: G7ObservableMetrics
    all_path_secondary: G7ObservableMetrics | None
    family_p_values: tuple[tuple[str, float], ...]
    family_critical_values: tuple[tuple[str, float], ...]
    decisions: tuple[ValidationDecision, ...]
    estimator_contract_id: str
    estimator_contract_ready: bool
    zero_nonfinite_or_data_issues: bool
    canonical_profile: bool
    primary_passed: bool
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "contract_fingerprint": self.contract_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "primary": self.primary.as_dict(),
            "all_path_secondary": (
                None
                if self.all_path_secondary is None
                else self.all_path_secondary.as_dict()
            ),
            "family_p_values": dict(self.family_p_values),
            "family_critical_values": dict(self.family_critical_values),
            "decisions": [item.as_dict() for item in self.decisions],
            "estimator_contract_id": self.estimator_contract_id,
            "estimator_contract_ready": self.estimator_contract_ready,
            "zero_nonfinite_or_data_issues": self.zero_nonfinite_or_data_issues,
            "canonical_profile": self.canonical_profile,
            "primary_passed": self.primary_passed,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True, init=False)
class G7ScientificPassReport:
    """Canonical immutable G7 scientific pass bound to one structural scan."""

    schema_id: str
    schema_version: int
    production_authority_fingerprint: str
    g5_threshold_fingerprint: str
    g5_qualification_results_fingerprint: str
    null_calibration_fingerprint: str
    consumer_manifest_sha256: str
    structural_report_fingerprint: str
    assessment: G7ScientificAssessment
    canonical_results: CanonicalValidationResults
    latent_state_evidence_scope: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "production_authority_fingerprint": (self.production_authority_fingerprint),
            "g5_threshold_fingerprint": self.g5_threshold_fingerprint,
            "g5_qualification_results_fingerprint": (
                self.g5_qualification_results_fingerprint
            ),
            "null_calibration_fingerprint": self.null_calibration_fingerprint,
            "consumer_manifest_sha256": self.consumer_manifest_sha256,
            "structural_report_fingerprint": self.structural_report_fingerprint,
            "assessment": self.assessment.as_dict(),
            "canonical_results": self.canonical_results.as_dict(),
            "latent_state_evidence_scope": self.latent_state_evidence_scope,
        }


@dataclass(frozen=True, slots=True)
class G7ValidatedPublication:
    """Paths and identities published by the coordinated G7 boundary."""

    structural_report_path: Path
    scientific_report_path: Path
    data_validated_path: Path
    structural_report_fingerprint: str
    scientific_pass_fingerprint: str


class _MetricAccumulator:
    def __init__(self, contract: G7FrozenScientificContract) -> None:
        self.contract = contract
        self.lf_rows: list[G5PathObservables] = []
        self.hf_rows: list[G5PathObservables] = []
        self.lf_diversity = PathDiversityAccumulator()
        self.hf_diversity = PathDiversityAccumulator()
        self.lf_paths = 0
        self.hf_paths = 0

    def update(
        self,
        *,
        frequency: Frequency,
        full_path: FloatArray,
        row: G5PathObservables,
    ) -> None:
        if frequency == "monthly":
            self.lf_rows.append(row)
            self.lf_diversity.update(full_path)
            self.lf_paths += 1
        else:
            self.hf_rows.append(row)
            self.hf_diversity.update(full_path)
            self.hf_paths += 1

    def finalize(self) -> G7ObservableMetrics:
        if self.lf_paths <= 0 or self.hf_paths <= 0:
            raise ValidationError("G7 metrics require both LF and HF paths")
        duplicate_copies = (
            self.lf_diversity.finalize().duplicate_copies
            + self.hf_diversity.finalize().duplicate_copies
        )
        observed = aggregate_g5_cluster_observables(
            tuple(sorted(self.lf_rows, key=lambda item: item.path_rank)),
            tuple(sorted(self.hf_rows, key=lambda item: item.path_rank)),
            self.contract.monthly_reference_observables,
            self.contract.daily_reference_observables,
            sobol_directions=self.contract.sobol_directions,
            expected_lf=self.lf_paths,
            expected_hf=self.hf_paths,
            adversary_fit_fingerprint=self.contract.adversary_fit.fingerprint,
            duplicate_copies=duplicate_copies,
            raw_monthly_reference=(self.contract.raw_monthly_reference_observables),
            raw_daily_reference=self.contract.raw_daily_reference_observables,
        )
        endpoints = {
            name: value
            for name, value in observed.endpoint_values
            if not name.endswith(("source_concentration", "repeated_subsequence"))
        }
        primary_endpoints = {
            name: value for name, value in endpoints.items() if ".raw." not in name
        }
        family_values = _observable_family_values(primary_endpoints)
        identity = {
            "lf_paths": self.lf_paths,
            "hf_paths": self.hf_paths,
            "endpoint_values": endpoints,
            "family_statistics": family_values,
            "direct_alpha_cap_ratio": observed.direct_alpha_cap_ratio,
            "raw_hard_cap_ratio": observed.raw_hard_cap_ratio,
            "duplicate_copies": duplicate_copies,
            "csv_excluded_endpoints": G7_CSV_EXCLUDED_ENDPOINTS,
            "inherited_state_source_evidence_fingerprint": (
                self.contract.inherited_state_source_evidence_fingerprint
            ),
            "estimator_contract_fingerprint": (observed.estimator_contract_fingerprint),
            "adversary_fit_fingerprint": observed.adversary_fit_fingerprint,
        }
        return G7ObservableMetrics(
            lf_paths=self.lf_paths,
            hf_paths=self.hf_paths,
            endpoint_values=tuple(sorted(endpoints.items())),
            family_statistics=tuple(
                (name, family_values[name]) for name in G5_FAMILY_NAMES
            ),
            direct_alpha_cap_ratio=observed.direct_alpha_cap_ratio,
            raw_hard_cap_ratio=observed.raw_hard_cap_ratio,
            duplicate_copies=duplicate_copies,
            csv_excluded_endpoints=G7_CSV_EXCLUDED_ENDPOINTS,
            inherited_state_source_evidence_fingerprint=(
                self.contract.inherited_state_source_evidence_fingerprint
            ),
            estimator_contract_fingerprint=(observed.estimator_contract_fingerprint),
            adversary_fit_fingerprint=observed.adversary_fit_fingerprint,
            fingerprint=_fingerprint(identity),
        )


class G7ScientificMetricHook:
    """Bounded structural hook for primary and optional all-path G7 metrics."""

    name: Final = G7_SCIENTIFIC_HOOK_NAME

    def __init__(
        self,
        contract: G7FrozenScientificContract,
        *,
        plan: G7ScientificValidationPlan = CANONICAL_G7_SCIENTIFIC_PLAN,
    ) -> None:
        self.contract = _verify_contract(contract)
        if not isinstance(plan, G7ScientificValidationPlan):
            raise ValidationError("G7 scientific plan has the wrong type")
        self.plan = plan
        self._primary = _MetricAccumulator(self.contract)
        self._secondary = (
            _MetricAccumulator(self.contract)
            if plan.collect_all_path_secondary
            else None
        )
        self._seen: dict[Family, set[int]] = {"LF": set(), "HF": set()}
        self._assessment: G7ScientificAssessment | None = None

    @property
    def assessment(self) -> G7ScientificAssessment:
        if self._assessment is None:
            raise ValidationError("G7 scientific hook has not been finalized")
        return self._assessment

    def observe_path(self, path: StructuralPathView) -> None:
        """Consume only monthly/daily paths after structural row validation."""

        if self._assessment is not None:
            raise ValidationError("G7 scientific hook was already finalized")
        if not isinstance(path, StructuralPathView):
            raise ValidationError("G7 scientific hook received the wrong path type")
        if path.frequency not in {"monthly", "daily"}:
            return
        family: Family = "LF" if path.frequency == "monthly" else "HF"
        if path.family != family:
            raise IntegrityError("G7 base-frequency path uses the wrong family")
        expected = (
            self.plan.expected_lf_paths
            if family == "LF"
            else self.plan.expected_hf_paths
        )
        if not 1 <= path.path_entity <= expected:
            raise IntegrityError("G7 base-frequency path is outside the fixed range")
        if path.path_entity in self._seen[family]:
            raise IntegrityError("G7 base-frequency path was observed more than once")
        expected_id = f"{family}{path.path_entity:06d}"
        if path.path_id != expected_id:
            raise IntegrityError("G7 base-frequency path ID is noncanonical")
        self._seen[family].add(path.path_entity)
        reference = (
            self.contract.monthly_reference
            if path.frequency == "monthly"
            else self.contract.daily_reference
        )
        values = _return_matrix(path.log_returns)
        frequency = cast(Frequency, path.frequency)
        selection_uniform = _registered_evaluation_uniform(
            self.contract,
            family=family,
            path_entity=path.path_entity,
        )
        window, offset = _equal_history_window(
            values,
            observations=reference.log_returns.shape[0],
            selection_uniform=selection_uniform,
        )
        fitted = (
            self.contract.adversary_fit.monthly
            if frequency == "monthly"
            else self.contract.adversary_fit.daily
        )
        alpha_sharpes = score_g5_adversary_path(fitted, window)
        row = compute_g5_path_observables(
            window,
            frequency=frequency,
            alpha_sharpes=alpha_sharpes,
            source_indices=None,
            evaluation_offset=offset,
            selection_uniform=selection_uniform,
            path_rank=path.path_entity,
        )
        if self._secondary is not None:
            self._secondary.update(
                frequency=frequency,
                full_path=values,
                row=row,
            )
        primary_limit = (
            self.plan.primary_lf_paths if family == "LF" else self.plan.primary_hf_paths
        )
        if path.path_entity <= primary_limit:
            self._primary.update(
                frequency=frequency,
                full_path=values,
                row=row,
            )

    def finalize(self) -> Mapping[str, JSONValue]:
        """Finalize deterministic summaries; incomplete streams abort."""

        if self._assessment is None:
            self._require_complete_stream()
            primary = self._primary.finalize()
            secondary = None if self._secondary is None else self._secondary.finalize()
            self._assessment = _assess(
                self.contract,
                plan=self.plan,
                primary=primary,
                secondary=secondary,
            )
        return cast(Mapping[str, JSONValue], self._assessment.as_dict())

    def _require_complete_stream(self) -> None:
        expected = {
            "LF": set(range(1, self.plan.expected_lf_paths + 1)),
            "HF": set(range(1, self.plan.expected_hf_paths + 1)),
        }
        if self._seen != expected:
            raise IntegrityError("G7 scientific base-frequency stream is incomplete")


def build_g7_frozen_scientific_contract(
    *,
    production_authority_fingerprint: str,
    thresholds: CanonicalThresholds,
    qualification_results: CanonicalValidationResults,
    null_calibration: G5FamilyNullCalibration,
    monthly_reference: G5ReferenceData,
    daily_reference: G5ReferenceData,
    sobol_directions: SobolDirectionSet,
    adversary_fit: G5AdversaryFitPair,
    estimator_contract_fingerprint: str,
    evaluation_master_seed: int,
    policy_scientific_version: int,
    g3_production_latent_task_fingerprint: str,
    g3_production_latent_checkpoint_fingerprint: str,
    g3_production_adequacy_task_fingerprint: str,
    g3_production_adequacy_checkpoint_fingerprint: str,
    raw_monthly_reference: G5ReferenceData | None = None,
    raw_daily_reference: G5ReferenceData | None = None,
) -> G7FrozenScientificContract:
    """Verify and freeze the exact passed G5 inputs reused by G7."""

    authority = _required_fingerprint(
        production_authority_fingerprint, "production authority"
    )
    checked_thresholds = _verified_thresholds(thresholds)
    checked_results = _verified_qualification_results(
        qualification_results, checked_thresholds.fingerprint
    )
    checked_null = _verified_null_calibration(null_calibration, checked_thresholds)
    checked_monthly = _verified_reference(monthly_reference, "monthly")
    checked_daily = _verified_reference(daily_reference, "daily")
    checked_raw_monthly, checked_raw_daily = _verified_raw_reference_pair(
        raw_monthly_reference, raw_daily_reference
    )
    _verify_sobol(sobol_directions, checked_null.sobol_direction_fingerprint)
    estimator = _required_fingerprint(
        estimator_contract_fingerprint, "estimator contract"
    )
    if estimator != G5_ESTIMATOR_CONTRACT_FINGERPRINT:
        raise IntegrityError("G7 estimator differs from the frozen G5 estimator")
    fitted = _verified_adversary_fit(adversary_fit)
    bindings = dict(checked_thresholds.binding_fingerprints)
    if (
        bindings.get(G5_THRESHOLD_ESTIMATOR_BINDING) != estimator
        or bindings.get(G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING)
        != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        or bindings.get(G7_THRESHOLD_ADVERSARY_FIT_BINDING) != fitted.fingerprint
        or dict(checked_results.metric_fingerprints).get("adversary_fit")
        != fitted.fingerprint
    ):
        raise IntegrityError("G7 fit/specification differs from frozen G5 evidence")
    if (
        isinstance(evaluation_master_seed, bool)
        or not isinstance(evaluation_master_seed, int)
        or evaluation_master_seed < 0
        or isinstance(policy_scientific_version, bool)
        or not isinstance(policy_scientific_version, int)
        or policy_scientific_version not in {11, 12, 13, 14, 15}
    ):
        raise ValidationError("G7 evaluation RNG identity is invalid")
    observed_metrics = _required_fingerprint(
        dict(checked_results.metric_fingerprints).get("observed_metrics"),
        "qualification observed metrics",
    )
    latent_task_parent = _required_fingerprint(
        g3_production_latent_task_fingerprint,
        "G3 production latent-correctness task",
    )
    latent_checkpoint_parent = _required_fingerprint(
        g3_production_latent_checkpoint_fingerprint,
        "G3 production latent-correctness checkpoint",
    )
    adequacy_task_parent = _required_fingerprint(
        g3_production_adequacy_task_fingerprint,
        "G3 production refitted-adequacy task",
    )
    adequacy_checkpoint_parent = _required_fingerprint(
        g3_production_adequacy_checkpoint_fingerprint,
        "G3 production refitted-adequacy checkpoint",
    )
    inherited_state_source = _inherited_state_source_fingerprint(
        qualification_results=checked_results,
        observed_metrics_fingerprint=observed_metrics,
        latent_task_fingerprint=latent_task_parent,
        latent_checkpoint_fingerprint=latent_checkpoint_parent,
        adequacy_task_fingerprint=adequacy_task_parent,
        adequacy_checkpoint_fingerprint=adequacy_checkpoint_parent,
    )
    monthly_observables = compute_g5_reference_observables(checked_monthly)
    daily_observables = compute_g5_reference_observables(checked_daily)
    raw_monthly_observables = (
        None
        if checked_raw_monthly is None
        else compute_g5_reference_observables(checked_raw_monthly)
    )
    raw_daily_observables = (
        None
        if checked_raw_daily is None
        else compute_g5_reference_observables(checked_raw_daily)
    )
    identity = {
        "schema_id": G7_SCIENTIFIC_CONTRACT_SCHEMA_ID,
        "schema_version": G7_SCIENTIFIC_SCHEMA_VERSION,
        "production_authority_fingerprint": authority,
        "threshold_fingerprint": checked_thresholds.fingerprint,
        "qualification_results_fingerprint": checked_results.fingerprint,
        "null_calibration_fingerprint": checked_null.fingerprint,
        "monthly_reference_fingerprint": checked_monthly.fingerprint,
        "daily_reference_fingerprint": checked_daily.fingerprint,
        "monthly_reference_observables_fingerprint": monthly_observables.fingerprint,
        "daily_reference_observables_fingerprint": daily_observables.fingerprint,
        "raw_monthly_reference_observables_fingerprint": (
            None
            if raw_monthly_observables is None
            else raw_monthly_observables.fingerprint
        ),
        "raw_daily_reference_observables_fingerprint": (
            None if raw_daily_observables is None else raw_daily_observables.fingerprint
        ),
        "sobol_direction_fingerprint": sobol_directions.fingerprint,
        "adversary_fit_fingerprint": fitted.fingerprint,
        "estimator_contract_fingerprint": estimator,
        "evaluation_master_seed": evaluation_master_seed,
        "policy_scientific_version": policy_scientific_version,
        "qualification_observed_metrics_fingerprint": observed_metrics,
        "g3_production_latent_task_fingerprint": latent_task_parent,
        "g3_production_latent_checkpoint_fingerprint": (latent_checkpoint_parent),
        "g3_production_adequacy_task_fingerprint": adequacy_task_parent,
        "g3_production_adequacy_checkpoint_fingerprint": (adequacy_checkpoint_parent),
        "inherited_state_source_evidence_fingerprint": inherited_state_source,
    }
    result = object.__new__(G7FrozenScientificContract)
    for name, value in (
        ("schema_id", G7_SCIENTIFIC_CONTRACT_SCHEMA_ID),
        ("schema_version", G7_SCIENTIFIC_SCHEMA_VERSION),
        ("production_authority_fingerprint", authority),
        ("thresholds", checked_thresholds),
        ("qualification_results", checked_results),
        ("null_calibration", checked_null),
        ("monthly_reference", checked_monthly),
        ("daily_reference", checked_daily),
        ("monthly_reference_observables", monthly_observables),
        ("daily_reference_observables", daily_observables),
        ("raw_monthly_reference_observables", raw_monthly_observables),
        ("raw_daily_reference_observables", raw_daily_observables),
        ("sobol_directions", sobol_directions),
        ("adversary_fit", fitted),
        ("estimator_contract_fingerprint", estimator),
        ("evaluation_master_seed", evaluation_master_seed),
        ("policy_scientific_version", policy_scientific_version),
        ("qualification_observed_metrics_fingerprint", observed_metrics),
        (
            "g3_production_latent_task_fingerprint",
            latent_task_parent,
        ),
        (
            "g3_production_latent_checkpoint_fingerprint",
            latent_checkpoint_parent,
        ),
        ("g3_production_adequacy_task_fingerprint", adequacy_task_parent),
        (
            "g3_production_adequacy_checkpoint_fingerprint",
            adequacy_checkpoint_parent,
        ),
        ("inherited_state_source_evidence_fingerprint", inherited_state_source),
        ("fingerprint", _fingerprint(identity)),
        ("_seal", _CONTRACT_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    return result


def build_g7_scientific_pass_report(
    hook: G7ScientificMetricHook,
    structural_report: StructuralValidationReport,
    *,
    result_id: str,
) -> G7ScientificPassReport:
    """Mint a pass only from the canonical hook-bound all-row scan."""

    if not isinstance(hook, G7ScientificMetricHook):
        raise ValidationError("G7 pass requires the scientific metric hook")
    contract = _verify_contract(hook.contract)
    assessment = hook.assessment
    if not is_sealed_structural_validation_report(structural_report):
        raise IntegrityError("G7 pass requires a sealed structural report")
    if (
        hook.plan != CANONICAL_G7_SCIENTIFIC_PLAN
        or not assessment.canonical_profile
        or not assessment.primary_passed
        or not assessment.zero_nonfinite_or_data_issues
        or not structural_report.canonical_profile
        or structural_report.plan_fingerprint != CANONICAL_STRUCTURAL_PLAN.fingerprint
    ):
        raise ValidationError("G7 scientific pass requires the canonical clean profile")
    hook_payload = structural_report.hook_metrics.get(hook.name)
    if hook_payload != hook.finalize():
        raise IntegrityError("G7 assessment is not bound to the structural scan")
    metric_fingerprints = {
        "g7_assessment": assessment.fingerprint,
        "g7_primary_observables": assessment.primary.fingerprint,
        "g7_structural_scan": structural_report.fingerprint,
    }
    if assessment.all_path_secondary is not None:
        metric_fingerprints["g7_all_path_secondary"] = (
            assessment.all_path_secondary.fingerprint
        )
    results = build_canonical_results(
        result_id=result_id,
        threshold_fingerprint=contract.thresholds.fingerprint,
        subject_fingerprint=structural_report.manifest_sha256,
        metric_fingerprints=metric_fingerprints,
        decisions=assessment.decisions,
    )
    if not results.passed:
        raise ValidationError("G7 canonical results contain a required failure")
    identity = {
        "schema_id": G7_SCIENTIFIC_PASS_SCHEMA_ID,
        "schema_version": G7_SCIENTIFIC_SCHEMA_VERSION,
        "production_authority_fingerprint": (contract.production_authority_fingerprint),
        "g5_threshold_fingerprint": contract.thresholds.fingerprint,
        "g5_qualification_results_fingerprint": (
            contract.qualification_results.fingerprint
        ),
        "null_calibration_fingerprint": contract.null_calibration.fingerprint,
        "consumer_manifest_sha256": structural_report.manifest_sha256,
        "structural_report_fingerprint": structural_report.fingerprint,
        "assessment": assessment.as_dict(),
        "canonical_results": results.as_dict(),
        "latent_state_evidence_scope": (
            "latent_and_source_inherited_from_exact_passed_g5_g3_evidence_"
            "and_excluded_from_csv_recomputation"
        ),
    }
    report = object.__new__(G7ScientificPassReport)
    for name, value in (
        ("schema_id", G7_SCIENTIFIC_PASS_SCHEMA_ID),
        ("schema_version", G7_SCIENTIFIC_SCHEMA_VERSION),
        (
            "production_authority_fingerprint",
            contract.production_authority_fingerprint,
        ),
        ("g5_threshold_fingerprint", contract.thresholds.fingerprint),
        (
            "g5_qualification_results_fingerprint",
            contract.qualification_results.fingerprint,
        ),
        ("null_calibration_fingerprint", contract.null_calibration.fingerprint),
        ("consumer_manifest_sha256", structural_report.manifest_sha256),
        ("structural_report_fingerprint", structural_report.fingerprint),
        ("assessment", assessment),
        ("canonical_results", results),
        (
            "latent_state_evidence_scope",
            (
                "latent_and_source_inherited_from_exact_passed_g5_g3_evidence_"
                "and_excluded_from_csv_recomputation"
            ),
        ),
        ("fingerprint", _fingerprint(identity)),
        ("_seal", _PASS_SEAL),
    ):
        object.__setattr__(report, name, value)
    object.__setattr__(report, "_owner_id", id(report))
    return report


def publish_g7_scientific_pass_report(
    run_root: str | Path, report: G7ScientificPassReport
) -> Path:
    """Write one canonical pass with exclusive creation and no lifecycle marker."""

    checked = _verify_pass_report(report)
    root = _validated_run_root(run_root)
    directory = root / "validation"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise IntegrityError("G7 validation directory is unsafe")
    path = directory / G7_SCIENTIFIC_REPORT_FILENAME
    _publish_exclusive(path, _scientific_report_bytes(checked))
    return path


def publish_g7_validated_pair(
    run_root: str | Path,
    *,
    scientific_report: G7ScientificPassReport,
    structural_report: StructuralValidationReport,
) -> G7ValidatedPublication:
    """Publish one shared-scan report pair, then ``DATA_VALIDATED`` last.

    Each report is an exclusive durable file.  A crash or write error before
    the final marker leaves a visibly non-passing run; there is no repair or
    overwrite path.  This function performs no second CSV scan.
    """

    science = _verify_pass_report(scientific_report)
    if not is_sealed_structural_validation_report(structural_report):
        raise IntegrityError("G7 publication requires a sealed structural report")
    if (
        not structural_report.canonical_profile
        or structural_report.plan_fingerprint != CANONICAL_STRUCTURAL_PLAN.fingerprint
        or science.structural_report_fingerprint != structural_report.fingerprint
        or science.consumer_manifest_sha256 != structural_report.manifest_sha256
        or science.assessment.plan_fingerprint
        != CANONICAL_G7_SCIENTIFIC_PLAN.fingerprint
        or science.fingerprint
        in {
            structural_report.fingerprint,
            structural_report.manifest_sha256,
            structural_report.plan_fingerprint,
        }
    ):
        raise IntegrityError(
            "G7 scientific and structural reports do not share one scan"
        )
    root = _validated_run_root(run_root)
    for lifecycle_marker in (
        "RUNNING",
        "INTERRUPTED",
        "FAILED",
        "DATA_VALIDATED",
        "COMPLETE",
        "RELEASED",
    ):
        path = root / lifecycle_marker
        if path.exists() or path.is_symlink():
            raise IntegrityError("run lifecycle is not clean for G7 publication")
    validation = root / "validation"
    validation.mkdir(mode=0o700, exist_ok=True)
    if validation.is_symlink() or not validation.is_dir():
        raise IntegrityError("G7 validation directory is unsafe")
    structural_path = validation / "structural-report.json"
    scientific_path = validation / G7_SCIENTIFIC_REPORT_FILENAME
    data_validated_path = root / "DATA_VALIDATED"
    for path in (structural_path, scientific_path, data_validated_path):
        if path.exists() or path.is_symlink():
            raise IntegrityError("G7 publication destination already exists")

    structural_content = _canonical_bytes(
        {
            "schema_version": structural_report.schema_version,
            "report_fingerprint": structural_report.fingerprint,
            "identity": structural_report.identity(),
        }
    )
    scientific_content = _scientific_report_bytes(science)
    marker = {
        "schema_id": DATA_VALIDATED_SCHEMA_ID,
        "schema_version": DATA_VALIDATED_SCHEMA_VERSION,
        "scientific_pass_fingerprint": science.fingerprint,
        "structural_report_fingerprint": structural_report.fingerprint,
        "structural_report_sha256": hashlib.sha256(structural_content).hexdigest(),
    }
    _publish_exclusive(structural_path, structural_content)
    _publish_exclusive(scientific_path, scientific_content)
    _publish_exclusive(data_validated_path, _canonical_bytes(marker))
    return G7ValidatedPublication(
        structural_report_path=structural_path,
        scientific_report_path=scientific_path,
        data_validated_path=data_validated_path,
        structural_report_fingerprint=structural_report.fingerprint,
        scientific_pass_fingerprint=science.fingerprint,
    )


def _assess(
    contract: G7FrozenScientificContract,
    *,
    plan: G7ScientificValidationPlan,
    primary: G7ObservableMetrics,
    secondary: G7ObservableMetrics | None,
) -> G7ScientificAssessment:
    observed = dict(primary.family_statistics)
    null = contract.null_calibration.family_statistics
    p_values = {
        name: float(
            (1 + np.count_nonzero(null[:, index] >= observed[name]))
            / (null.shape[0] + 1)
        )
        for index, name in enumerate(G5_FAMILY_NAMES)
    }
    holm = holm_decisions(p_values)
    holm_by_name = {item.name: item for item in holm.decisions}
    critical = {
        item.name.removesuffix(".max"): item.critical_value
        for item in contract.thresholds.thresholds
    }
    estimator_ready = (
        contract.estimator_contract_fingerprint == G5_ESTIMATOR_CONTRACT_FINGERPRINT
        and primary.estimator_contract_fingerprint == G5_ESTIMATOR_CONTRACT_FINGERPRINT
        and contract.adversary_fit.specification_fingerprint
        == G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        and primary.adversary_fit_fingerprint == contract.adversary_fit.fingerprint
        and primary.inherited_state_source_evidence_fingerprint
        == contract.inherited_state_source_evidence_fingerprint
    )
    decisions = [
        ValidationDecision(
            name="shared_section14_estimator_contract",
            required=True,
            passed=estimator_ready,
            observed=G7_ESTIMATOR_CONTRACT_ID,
            criterion=(
                "corrected shared G5/G7 estimator and registered purpose-14/v1 "
                "offset contract installed"
            ),
        ),
        ValidationDecision(
            name="inherited_state_source_evidence_bound",
            required=True,
            passed=(
                primary.inherited_state_source_evidence_fingerprint
                == contract.inherited_state_source_evidence_fingerprint
                and primary.csv_excluded_endpoints == G7_CSV_EXCLUDED_ENDPOINTS
            ),
            observed=primary.inherited_state_source_evidence_fingerprint,
            criterion=(
                "CSV-unavailable source/latent endpoints are excluded and bound "
                "to exact passed G5/G3 evidence"
            ),
        ),
        ValidationDecision(
            name="canonical_primary_subset",
            required=True,
            passed=(
                plan.canonical_primary_counts
                and primary.lf_paths == G7_PRIMARY_LF_PATHS
                and primary.hf_paths == G7_PRIMARY_HF_PATHS
            ),
            observed=f"{primary.lf_paths}/{primary.hf_paths}",
            criterion="LF000001..LF000500 and HF000001..HF000200",
        ),
        ValidationDecision(
            name="g5_a_direct_alpha_equivalence",
            required=True,
            passed=primary.direct_alpha_cap_ratio <= 1.0,
            observed=primary.direct_alpha_cap_ratio,
            criterion="maximum direct alpha-Sharpe cap ratio <= 1",
        ),
        ValidationDecision(
            name="g5_b_practical_cap",
            required=True,
            passed=observed["g5_b"] <= 1.0,
            observed=observed["g5_b"],
            criterion="all CSV-recomputed G5-B practical cap ratios <= 1",
        ),
        ValidationDecision(
            name="g5_c_practical_cap",
            required=True,
            passed=observed["g5_c"] <= 1.0,
            observed=observed["g5_c"],
            criterion="all CSV-recomputed G5-C practical cap ratios <= 1",
        ),
        ValidationDecision(
            name="g5_d_practical_cap",
            required=True,
            passed=observed["g5_d"] <= 1.0,
            observed=observed["g5_d"],
            criterion="all CSV-recomputed G5-D practical cap ratios <= 1",
        ),
        ValidationDecision(
            name="neutral_raw_historical_cap",
            required=contract.raw_monthly_reference_observables is not None,
            passed=primary.raw_hard_cap_ratio <= 1.0,
            observed=primary.raw_hard_cap_ratio,
            criterion="neutral production remains within raw-history hard caps",
        ),
        ValidationDecision(
            name="g5_d_complete_path_diversity",
            required=True,
            passed=primary.duplicate_copies == 0,
            observed=primary.duplicate_copies,
            criterion="zero duplicate complete base-frequency primary paths",
        ),
        ValidationDecision(
            name="zero_nonfinite_or_data_issues",
            required=True,
            passed=True,
            observed=0,
            criterion="zero nonfinite values or path/data contract issues",
        ),
    ]
    for name in G5_FAMILY_NAMES:
        decisions.extend(
            (
                ValidationDecision(
                    name=f"{name}_fixed_threshold",
                    required=True,
                    passed=observed[name] <= critical[name],
                    observed=observed[name],
                    criterion=f"observed <= frozen {name}.max critical value",
                ),
                ValidationDecision(
                    name=f"{name}_holm",
                    required=True,
                    passed=not holm_by_name[name].rejected,
                    observed=holm_by_name[name].adjusted_p_value,
                    criterion="Holm-adjusted upper-tail p-value > 0.05",
                ),
            )
        )
    decisions_tuple = tuple(decisions)
    passed = all(item.passed for item in decisions_tuple if item.required)
    identity = {
        "schema_id": G7_SCIENTIFIC_ASSESSMENT_SCHEMA_ID,
        "schema_version": G7_SCIENTIFIC_SCHEMA_VERSION,
        "contract_fingerprint": contract.fingerprint,
        "plan_fingerprint": plan.fingerprint,
        "primary_fingerprint": primary.fingerprint,
        "all_path_secondary_fingerprint": (
            None if secondary is None else secondary.fingerprint
        ),
        "family_p_values": p_values,
        "family_critical_values": critical,
        "decisions": [item.as_dict() for item in decisions_tuple],
        "estimator_contract_id": G7_ESTIMATOR_CONTRACT_ID,
        "estimator_contract_ready": estimator_ready,
        "zero_nonfinite_or_data_issues": True,
        "canonical_profile": plan.canonical_profile,
        "primary_passed": passed,
    }
    return G7ScientificAssessment(
        schema_id=G7_SCIENTIFIC_ASSESSMENT_SCHEMA_ID,
        schema_version=G7_SCIENTIFIC_SCHEMA_VERSION,
        contract_fingerprint=contract.fingerprint,
        plan_fingerprint=plan.fingerprint,
        primary=primary,
        all_path_secondary=secondary,
        family_p_values=tuple(sorted(p_values.items())),
        family_critical_values=tuple(sorted(critical.items())),
        decisions=decisions_tuple,
        estimator_contract_id=G7_ESTIMATOR_CONTRACT_ID,
        estimator_contract_ready=estimator_ready,
        zero_nonfinite_or_data_issues=True,
        canonical_profile=plan.canonical_profile,
        primary_passed=passed,
        fingerprint=_fingerprint(identity),
    )


def _equal_history_window(
    values: FloatArray,
    *,
    observations: int,
    selection_uniform: float,
) -> tuple[FloatArray, int]:
    if values.shape[0] < observations:
        raise ValidationError("production path is shorter than historical reference")
    choices = values.shape[0] - observations + 1
    if not 0.0 <= selection_uniform < 1.0:
        raise ValidationError("G7 registered evaluation uniform is invalid")
    offset = min(choices - 1, int(selection_uniform * choices))
    return values[offset : offset + observations], offset


def _registered_evaluation_uniform(
    contract: G7FrozenScientificContract,
    *,
    family: Family,
    path_entity: int,
) -> float:
    key = calibration_rng_key(
        master_seed=contract.evaluation_master_seed,
        scientific_version=contract.policy_scientific_version,
        purpose=CalibrationPurpose.SOBOL_OR_STRATEGY,
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        variant=1,
        replicate=path_entity - 1,
        domain=Domain.QUALIFICATION,
        family=RngFamily.LF if family == "LF" else RngFamily.HF,
        stage=Stage.VALIDATION_SAMPLING_OR_STRATEGY,
    )
    return float(key.generator().random())


def _observable_family_values(endpoints: Mapping[str, float]) -> dict[str, float]:
    return {
        "g5_a": _endpoint_max(endpoints, ("directional_alpha",)),
        "g5_b": _endpoint_max(
            endpoints,
            ("covariance_correlation", "serial_dependence", "cross_lag"),
        ),
        "g5_c": _endpoint_max(
            endpoints,
            (
                "sliced_wasserstein",
                "energy_distance",
                "spread",
                "spread_wasserstein",
                "spread_rolling",
                "tail_dependence",
            ),
        ),
        "g5_d": _endpoint_max(
            endpoints,
            ("location_scale", "marginal_tail", "marginal_shape", "drawdown"),
        ),
    }


def _verified_thresholds(value: CanonicalThresholds) -> CanonicalThresholds:
    if not isinstance(value, CanonicalThresholds):
        raise ValidationError("G7 requires canonical G5 thresholds")
    rebuilt = canonical_thresholds_from_dict(value.as_dict())
    names = {item.name for item in rebuilt.thresholds}
    expected = {f"{name}.max" for name in G5_FAMILY_NAMES}
    if names != expected or not rebuilt.precision_passed:
        raise ValidationError("G7 thresholds are not the passed G5 family set")
    return rebuilt


def _verified_qualification_results(
    value: CanonicalValidationResults, threshold_fingerprint: str
) -> CanonicalValidationResults:
    if not isinstance(value, CanonicalValidationResults):
        raise ValidationError("G7 requires canonical G5 qualification results")
    rebuilt = canonical_results_from_dict(value.as_dict())
    required = {
        "candidate_counts_exact",
        "generation_policy_and_stream_bound",
        "primary_estimator_contract_bound",
        "registered_evaluation_windows",
        "g5_a_direct_alpha_equivalence",
        "g5_b_practical_cap",
        "g5_c_practical_cap",
        "g5_d_practical_cap",
        "latent_chain_correctness",
        "g5_d_complete_path_diversity",
        *(f"{name}_holm" for name in G5_FAMILY_NAMES),
    }
    decisions = {item.name: item for item in rebuilt.decisions}
    if (
        not rebuilt.passed
        or rebuilt.threshold_fingerprint != threshold_fingerprint
        or not required.issubset(decisions)
        or "adversary_fit" not in dict(rebuilt.metric_fingerprints)
        or "observed_metrics" not in dict(rebuilt.metric_fingerprints)
        or any(
            not decisions[name].required or not decisions[name].passed
            for name in required
        )
    ):
        raise ValidationError("G7 requires the exact passed G5 qualification gates")
    return rebuilt


def _verified_null_calibration(
    value: G5FamilyNullCalibration, thresholds: CanonicalThresholds
) -> G5FamilyNullCalibration:
    if not isinstance(value, G5FamilyNullCalibration):
        raise ValidationError("G7 requires the G5 family null calibration")
    if (
        value.replicate_count != OUTER_NULL_REPLICATES
        or not value.fixed_sample_canonical
        or value.canonical_thresholds is None
        or value.canonical_thresholds.fingerprint != thresholds.fingerprint
    ):
        raise ValidationError("G7 requires the exact canonical 50,000-run calibration")
    rebuilt = build_g5_family_null_calibration(
        {
            name: value.family_statistics[:, index]
            for index, name in enumerate(G5_FAMILY_NAMES)
        },
        threshold_set_id=thresholds.threshold_set_id,
        binding_fingerprints=dict(value.binding_fingerprints),
        sobol_direction_fingerprint=value.sobol_direction_fingerprint,
    )
    if rebuilt.fingerprint != value.fingerprint:
        raise IntegrityError("G5 family null calibration identity is inconsistent")
    return rebuilt


def _verified_reference(
    value: G5ReferenceData, frequency: Frequency
) -> G5ReferenceData:
    if not isinstance(value, G5ReferenceData) or value.frequency != frequency:
        raise ValidationError("G7 historical reference has the wrong frequency")
    rebuilt = build_g5_reference_data(
        value.log_returns, value.decoded_states, frequency=frequency
    )
    if rebuilt.fingerprint != value.fingerprint:
        raise IntegrityError("G7 historical reference identity is inconsistent")
    return rebuilt


def _verified_raw_reference_pair(
    monthly: G5ReferenceData | None,
    daily: G5ReferenceData | None,
) -> tuple[G5ReferenceData | None, G5ReferenceData | None]:
    if (monthly is None) is not (daily is None):
        raise ValidationError("G7 raw historical references must be supplied together")
    if monthly is None or daily is None:
        return None, None
    return _verified_reference(monthly, "monthly"), _verified_reference(daily, "daily")


def _verified_adversary_fit(value: G5AdversaryFitPair) -> G5AdversaryFitPair:
    if (
        not isinstance(value, G5AdversaryFitPair)
        or value.monthly.frequency != "monthly"
        or value.daily.frequency != "daily"
        or value.specification_fingerprint != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
    ):
        raise ValidationError("G7 requires the exact frozen G5 adversary fit")
    expected = _fingerprint(
        {
            "specification_fingerprint": value.specification_fingerprint,
            "monthly_model_fingerprint": value.monthly.fingerprint,
            "daily_model_fingerprint": value.daily.fingerprint,
        }
    )
    if value.fingerprint != expected:
        raise IntegrityError("G7 adversary fit identity is inconsistent")
    return value


def _inherited_state_source_fingerprint(
    *,
    qualification_results: CanonicalValidationResults,
    observed_metrics_fingerprint: str,
    latent_task_fingerprint: str,
    latent_checkpoint_fingerprint: str,
    adequacy_task_fingerprint: str,
    adequacy_checkpoint_fingerprint: str,
) -> str:
    decisions = {item.name: item for item in qualification_results.decisions}
    inherited = {
        "latent_chain_correctness",
        "g5_d_complete_path_diversity",
        "g5_d_practical_cap",
        "g5_d_holm",
    }
    if any(
        name not in decisions
        or not decisions[name].required
        or not decisions[name].passed
        for name in inherited
    ):
        raise ValidationError("G7 inherited state/source evidence did not pass G5")
    return _fingerprint(
        {
            "schema_id": G7_INHERITED_STATE_SOURCE_SCHEMA_ID,
            "qualification_results_fingerprint": qualification_results.fingerprint,
            "qualification_subject_fingerprint": (
                qualification_results.subject_fingerprint
            ),
            "observed_metrics_fingerprint": observed_metrics_fingerprint,
            "g3_production_latent_task_fingerprint": latent_task_fingerprint,
            "g3_production_latent_checkpoint_fingerprint": (
                latent_checkpoint_fingerprint
            ),
            "g3_production_adequacy_task_fingerprint": adequacy_task_fingerprint,
            "g3_production_adequacy_checkpoint_fingerprint": (
                adequacy_checkpoint_fingerprint
            ),
            "inherited_decisions": sorted(inherited),
            "csv_excluded_endpoints": list(G7_CSV_EXCLUDED_ENDPOINTS),
        }
    )


def _verify_sobol(value: SobolDirectionSet, expected_fingerprint: str) -> None:
    if (
        not isinstance(value, SobolDirectionSet)
        or value.fingerprint != expected_fingerprint
        or value.fingerprint != _fingerprint(value.identity_dict())
        or value.method != SOBOL_METHOD
        or value.count != SOBOL_DIRECTION_COUNT
        or value.dimension != SOBOL_DIMENSION
        or value.directions.shape != (SOBOL_DIRECTION_COUNT, SOBOL_DIMENSION)
        or value.directions.flags.writeable
        or not bool(np.isfinite(value.directions).all())
        or hashlib.sha256(value.directions.tobytes(order="C")).hexdigest()
        != value.directions_sha256
        or not np.allclose(
            np.linalg.norm(value.directions, axis=1),
            1.0,
            rtol=0.0,
            atol=4.0 * np.finfo(np.float64).eps,
        )
    ):
        raise IntegrityError("G7 Sobol direction identity is inconsistent")


def _verify_contract(value: object) -> G7FrozenScientificContract:
    if type(value) is not G7FrozenScientificContract:
        raise ValidationError("G7 scientific contract was not built by its producer")
    contract = value
    if (
        contract._seal is not _CONTRACT_SEAL
        or contract._owner_id != id(contract)
        or contract.schema_id != G7_SCIENTIFIC_CONTRACT_SCHEMA_ID
        or contract.schema_version != G7_SCIENTIFIC_SCHEMA_VERSION
        or contract.fingerprint != _fingerprint(contract.identity_dict())
    ):
        raise IntegrityError("G7 scientific contract identity is inconsistent")
    return contract


def _verify_pass_report(value: object) -> G7ScientificPassReport:
    if type(value) is not G7ScientificPassReport:
        raise ValidationError("G7 scientific pass was not built by its producer")
    report = value
    if (
        report._seal is not _PASS_SEAL
        or report._owner_id != id(report)
        or report.schema_id != G7_SCIENTIFIC_PASS_SCHEMA_ID
        or report.schema_version != G7_SCIENTIFIC_SCHEMA_VERSION
        or not report.assessment.primary_passed
        or not report.assessment.canonical_profile
        or not report.canonical_results.passed
        or report.fingerprint != _fingerprint(report.identity_dict())
    ):
        raise IntegrityError("G7 scientific pass identity is inconsistent")
    return report


def _return_matrix(value: object) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError("G7 path returns must be numeric") from error
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
        raise ValidationError("G7 path returns have the wrong geometry")
    if not bool(np.isfinite(array).all()):
        raise ValidationError("G7 path returns contain NaN or infinity")
    return array


def _endpoint_max(endpoints: Mapping[str, float], suffixes: tuple[str, ...]) -> float:
    selected = [
        value
        for name, value in endpoints.items()
        if any(name.endswith(f".{suffix}") for suffix in suffixes)
    ]
    if not selected:
        raise ValidationError("G7 family has no observable endpoints")
    return max(selected)
