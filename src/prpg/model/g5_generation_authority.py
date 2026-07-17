"""Sealed G5 candidate-generation and final production authorities.

G3 supplies immutable scientific evidence and an unqualified candidate-policy
set.  G5 has two deliberately different authority boundaries:

* a threshold-passed authority may generate only bounded development and
  untouched qualification candidates; and
* final production authority exists only after immutable threshold,
  qualification-result, and meta-validation evidence all pass and agree.

The public producers in this module are the narrow coordinator boundary for
the evolving statistical implementation.  They accept exact final
fingerprints plus explicit pass decisions, but no statistical estimators.
All returned objects are immutable, process-original capabilities; copying or
direct dataclass construction cannot manufacture authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, TypeGuard

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_evidence_store import (
    LoadedG3EvidenceAuthority,
    verify_loaded_g3_evidence,
)
from prpg.model.scientific_artifact import (
    LoadedScientificModelArtifact,
    is_verified_scientific_model_artifact,
)
from prpg.model.scientific_policy import (
    AlphaPolicy,
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
    generation_policy_from_mapping,
)
from prpg.validation.adversaries import (
    G5_ADVERSARY_SCHEMA_ID,
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    G5_MEMORIZATION_HORIZON,
    G5_RIDGE_PENALTY,
    G5AdversaryFitPair,
    G5AdversaryModel,
)
from prpg.validation.records import (
    CanonicalThresholds,
    CanonicalValidationResults,
    canonical_results_from_dict,
    canonical_thresholds_from_dict,
)

G5_GENERATION_AUTHORITY_SCHEMA_ID: Final = "prpg-g5-generation-authority-v3"
G5_GENERATION_AUTHORITY_SCHEMA_VERSION: Final = 3
G5_QUALIFICATION_AUTHORITY_SCHEMA_ID: Final = (
    "prpg-g5-qualification-generation-authority-v3"
)
G5_QUALIFICATION_AUTHORITY_SCHEMA_VERSION: Final = 3
G5_EVIDENCE_SCHEMA_VERSION: Final = 2
G5_THRESHOLD_EVIDENCE_SCHEMA_ID: Final = "prpg-g5-threshold-evidence-v2"
G5_QUALIFICATION_EVIDENCE_SCHEMA_ID: Final = "prpg-g5-qualification-evidence-v2"
G5_META_VALIDATION_EVIDENCE_SCHEMA_ID: Final = "prpg-g5-meta-validation-evidence-v2"
G5_DEVELOPMENT_SELECTION_SCHEMA_ID: Final = "prpg-g5-development-selection-v1"
G5_DEVELOPMENT_SELECTION_SCHEMA_VERSION: Final = 1
G5_POLICY_STREAM_SCHEMA_ID: Final = "prpg-g5-policy-stream-v1"
G5_QUALIFICATION_NAMESPACE: Final = "g5_candidate_qualification_v1"
G5_THRESHOLD_ESTIMATOR_BINDING: Final = "g5_estimator_contract"
G5_THRESHOLD_POLICY_BINDING: Final = "g5_generation_policy"
G5_THRESHOLD_POLICY_STREAM_BINDING: Final = "g5_policy_stream"
G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING: Final = "g5_adversary_specification"
G5_THRESHOLD_ADVERSARY_FIT_BINDING: Final = "g5_adversary_fit"
G5_BASELINE_DIRECTIONAL_FAILURES: Final = frozenset(
    {"g5_a_direct_alpha_equivalence", "g5_a_holm"}
)
G5_BASELINE_REQUIRED_DECISIONS: Final = frozenset(
    {
        "candidate_counts_exact",
        "g5_a_direct_alpha_equivalence",
        "g5_d_complete_path_diversity",
        "g5_a_holm",
        "g5_b_holm",
        "g5_c_holm",
        "g5_d_holm",
    }
)
G5_PRODUCTION_LATENT_TASK_ID: Final = "production_latent_correctness"
G5_PRODUCTION_ADEQUACY_TASK_ID: Final = "production_refitted_adequacy"
G5_HISTORICAL_POLICY_SCIENTIFIC_VERSION: Final = 11
G5_NEUTRAL_POLICY_SCIENTIFIC_VERSIONS: Final = (
    (0.25, 12),
    (0.50, 13),
    (0.75, 14),
    (1.00, 15),
)

# These are path budgets.  The 50,000 outer-null simulations are a distinct
# threshold/meta-validation sample count and never occupy a path-count field.
G5_DEVELOPMENT_LF_PATH_COUNT: Final = 100
G5_DEVELOPMENT_HF_PATH_COUNT: Final = 20
G5_QUALIFICATION_LF_PATH_COUNT: Final = 500
G5_QUALIFICATION_HF_PATH_COUNT: Final = 200
G5_OUTER_NULL_REPLICATES: Final = 50_000

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_THRESHOLD_EVIDENCE_SEAL = object()
_QUALIFICATION_EVIDENCE_SEAL = object()
_META_VALIDATION_EVIDENCE_SEAL = object()
_DEVELOPMENT_SELECTION_SEAL = object()
_G5_AUTHORITY_SEAL = object()
_G5_QUALIFICATION_AUTHORITY_SEAL = object()
_G5_NONCANONICAL_TEST_CAPABILITY = object()
_G5_NONCANONICAL_TEST_OBJECTS: set[int] = set()


@dataclass(frozen=True, slots=True)
class G5DevelopmentPolicyDecision:
    """One frozen-rule neutral development row in registered lambda order."""

    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    neutral_lambda: float
    policy_scientific_version: int
    evaluated: bool
    development_result_fingerprint: str | None
    passed: bool | None

    def identity_dict(self) -> dict[str, object]:
        return {
            "generation_policy": self.generation_policy.as_dict(),
            "generation_policy_fingerprint": self.generation_policy_fingerprint,
            "neutral_lambda": self.neutral_lambda,
            "policy_scientific_version": self.policy_scientific_version,
            "evaluated": self.evaluated,
            "development_result_fingerprint": self.development_result_fingerprint,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True, init=False)
class G5DevelopmentPolicySelection:
    """Nonauthorizing proof that neutral activation chose the first pass."""

    generation_authorized: ClassVar[Literal[False]] = False
    schema_id: str
    schema_version: int
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    baseline_threshold_evidence: G5ThresholdEvidence
    baseline_threshold_evidence_fingerprint: str
    baseline_qualification_results: CanonicalValidationResults
    baseline_qualification_results_fingerprint: str
    decisions: tuple[G5DevelopmentPolicyDecision, ...]
    selected_policy: ScientificGenerationPolicy
    selected_policy_fingerprint: str
    selected_neutral_lambda: float
    selected_policy_scientific_version: int
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5DevelopmentPolicySelection is created only by its public producer"
        )

    def identity_dict(self) -> dict[str, object]:
        return _development_selection_identity(
            baseline_threshold_evidence=self.baseline_threshold_evidence,
            baseline_results=self.baseline_qualification_results,
            decisions=self.decisions,
            selected_policy=self.selected_policy,
        )


@dataclass(frozen=True, slots=True, init=False)
class G5ThresholdEvidence:
    """Passed fixed threshold calibration bound to exact G3 and provenance."""

    schema_id: str
    schema_version: int
    passed: Literal[True]
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    g3_source_code_fingerprint: str
    g3_dependency_lock_fingerprint: str
    g5_source_code_fingerprint: str
    g5_dependency_lock_fingerprint: str
    seed_registry_fingerprint: str
    thresholds: CanonicalThresholds
    threshold_registry_fingerprint: str
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    selected_alpha_policy: AlphaPolicy
    selected_neutral_lambda: float | None
    policy_scientific_version: int
    policy_stream_fingerprint: str
    estimator_contract_fingerprint: str
    adversary_specification_fingerprint: str
    adversary_fit: G5AdversaryFitPair
    adversary_fit_fingerprint: str
    threshold_subject_fingerprint: str
    development_selection: G5DevelopmentPolicySelection | None
    development_selection_fingerprint: str | None
    baseline_qualification_results_fingerprint: str | None
    outer_null_replicates: int
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("G5ThresholdEvidence is created only by its public producer")

    def identity_dict(self) -> dict[str, object]:
        return _threshold_evidence_identity(
            evidence=self.g3_evidence,
            g5_source_code_fingerprint=self.g5_source_code_fingerprint,
            g5_dependency_lock_fingerprint=self.g5_dependency_lock_fingerprint,
            seed_registry_fingerprint=self.seed_registry_fingerprint,
            thresholds=self.thresholds,
            policy=self.generation_policy,
            estimator_contract_fingerprint=self.estimator_contract_fingerprint,
            adversary_fit=self.adversary_fit,
            development_selection=self.development_selection,
            outer_null_replicates=self.outer_null_replicates,
        )


@dataclass(frozen=True, slots=True, init=False)
class G5QualificationEvidence:
    """Passed candidate qualification results for one registered policy."""

    schema_id: str
    schema_version: int
    passed: Literal[True]
    threshold_evidence: G5ThresholdEvidence
    threshold_evidence_fingerprint: str
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    selected_alpha_policy: AlphaPolicy
    selected_neutral_lambda: float | None
    seed_registry_fingerprint: str
    qualification_results: CanonicalValidationResults
    qualification_results_fingerprint: str
    qualification_subject_fingerprint: str
    adversary_fit: G5AdversaryFitPair
    adversary_fit_fingerprint: str
    development_lf_path_count: int
    development_hf_path_count: int
    qualification_lf_path_count: int
    qualification_hf_path_count: int
    g5_source_code_fingerprint: str
    g5_dependency_lock_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5QualificationEvidence is created only by its public producer"
        )

    def identity_dict(self) -> dict[str, object]:
        return _qualification_evidence_identity(
            threshold_evidence=self.threshold_evidence,
            policy=self.generation_policy,
            qualification_results=self.qualification_results,
            adversary_fit=self.adversary_fit,
        )


@dataclass(frozen=True, slots=True, init=False)
class G5MetaValidationEvidence:
    """Passed G5 meta-validation bound to threshold and qualification evidence."""

    schema_id: str
    schema_version: int
    passed: Literal[True]
    threshold_evidence: G5ThresholdEvidence
    threshold_evidence_fingerprint: str
    qualification_evidence: G5QualificationEvidence
    qualification_evidence_fingerprint: str
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    generation_policy_fingerprint: str
    seed_registry_fingerprint: str
    meta_validation_results: CanonicalValidationResults
    meta_validation_fingerprint: str
    meta_validation_subject_fingerprint: str
    outer_null_replicates: int
    g5_source_code_fingerprint: str
    g5_dependency_lock_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5MetaValidationEvidence is created only by its public producer"
        )

    def identity_dict(self) -> dict[str, object]:
        return _meta_validation_evidence_identity(
            threshold_evidence=self.threshold_evidence,
            qualification_evidence=self.qualification_evidence,
            meta_validation_results=self.meta_validation_results,
            outer_null_replicates=self.outer_null_replicates,
        )


@dataclass(frozen=True, slots=True, init=False)
class G5QualificationGenerationAuthority:
    """Restricted authority for bounded G5 candidate paths only."""

    generation_authorized: ClassVar[Literal[False]] = False
    schema_id: str
    schema_version: int
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    pair_receipt_fingerprint: str
    production_artifact: LoadedScientificModelArtifact
    production_artifact_fingerprint: str
    parameter_set_fingerprint: str
    candidate_policy_set: ScientificCandidatePolicySet
    candidate_policy_set_fingerprint: str
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    selected_alpha_policy: AlphaPolicy
    selected_neutral_lambda: float | None
    policy_scientific_version: int
    policy_stream_fingerprint: str
    adversary_fit: G5AdversaryFitPair
    adversary_fit_fingerprint: str
    qualification_namespace: str
    seed_registry_fingerprint: str
    threshold_registry_fingerprint: str
    threshold_evidence: G5ThresholdEvidence
    threshold_evidence_fingerprint: str
    development_lf_path_count: int
    development_hf_path_count: int
    qualification_lf_path_count: int
    qualification_hf_path_count: int
    g5_source_code_fingerprint: str
    g5_dependency_lock_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5QualificationGenerationAuthority is created only by the public "
            "G5 qualification coordinator"
        )

    @property
    def qualification_generation_authorized(self) -> Literal[True]:
        """Authorize only registered, bounded G5 candidate generation."""

        verify_g5_qualification_generation_authority(self)
        return True


@dataclass(frozen=True, slots=True, init=False)
class G5QualifiedProductionAuthority:
    """Final G5 authority bound to all three passed evidence records."""

    verification_scope: ClassVar[
        Literal["g5_thresholds_results_meta_policy_counts_provenance_and_exact_g3"]
    ] = "g5_thresholds_results_meta_policy_counts_provenance_and_exact_g3"

    schema_id: str
    schema_version: int
    qualification_gate: Literal["G5"]
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    g3_bundle_fingerprint: str
    pair_receipt_fingerprint: str
    g3_source_code_fingerprint: str
    g3_dependency_lock_fingerprint: str
    production_artifact: LoadedScientificModelArtifact
    production_artifact_fingerprint: str
    core_model_fingerprint: str
    parameter_set_fingerprint: str
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    selected_alpha_policy: AlphaPolicy
    selected_neutral_lambda: float | None
    policy_scientific_version: int
    policy_stream_fingerprint: str
    estimator_contract_fingerprint: str
    adversary_specification_fingerprint: str
    adversary_fit_fingerprint: str
    seed_registry_fingerprint: str
    development_lf_path_count: int
    development_hf_path_count: int
    qualification_lf_path_count: int
    qualification_hf_path_count: int
    outer_null_replicates: int
    g5_source_code_fingerprint: str
    g5_dependency_lock_fingerprint: str
    threshold_evidence: G5ThresholdEvidence
    threshold_evidence_fingerprint: str
    qualification_evidence: G5QualificationEvidence
    qualification_evidence_fingerprint: str
    meta_validation_evidence: G5MetaValidationEvidence
    meta_validation_evidence_fingerprint: str
    threshold_registry_fingerprint: str
    threshold_subject_fingerprint: str
    qualification_results_fingerprint: str
    qualification_subject_fingerprint: str
    meta_validation_fingerprint: str
    meta_validation_subject_fingerprint: str
    development_selection_fingerprint: str | None
    baseline_qualification_results_fingerprint: str | None
    g3_production_latent_task_fingerprint: str
    g3_production_latent_checkpoint_fingerprint: str
    g3_production_adequacy_task_fingerprint: str
    g3_production_adequacy_checkpoint_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5QualifiedProductionAuthority is created only by the public G5 producer"
        )

    @property
    def generation_authorized(self) -> Literal[True]:
        """Return true only after revalidating all live authority bindings."""

        verify_g5_qualified_production_authority(self)
        return True

    def identity_dict(self) -> dict[str, object]:
        """Return the immutable, JSON-safe authority identity."""

        return _authority_identity(
            evidence=self.g3_evidence,
            artifact=self.production_artifact,
            policy=self.generation_policy,
            threshold_evidence=self.threshold_evidence,
            qualification_evidence=self.qualification_evidence,
            meta_validation_evidence=self.meta_validation_evidence,
        )


_MINTED_THRESHOLD_EVIDENCE: dict[int, G5ThresholdEvidence] = {}
_MINTED_QUALIFICATION_EVIDENCE: dict[int, G5QualificationEvidence] = {}
_MINTED_META_VALIDATION_EVIDENCE: dict[int, G5MetaValidationEvidence] = {}
_MINTED_DEVELOPMENT_SELECTIONS: dict[int, G5DevelopmentPolicySelection] = {}
_MINTED_G5_AUTHORITIES: dict[int, G5QualifiedProductionAuthority] = {}
_MINTED_G5_QUALIFICATION_AUTHORITIES: dict[int, G5QualificationGenerationAuthority] = {}


def g5_policy_scientific_version(policy: ScientificGenerationPolicy) -> int:
    """Return the registered v3 stream code for one exact selected policy."""

    checked = _validated_selected_policy(policy)
    if checked.alpha_policy == "historical_vector":
        return G5_HISTORICAL_POLICY_SCIENTIFIC_VERSION
    for neutral_lambda, version in G5_NEUTRAL_POLICY_SCIENTIFIC_VERSIONS:
        if checked.selected_neutral_lambda == neutral_lambda:
            return version
    raise ModelError("G5 neutral policy is outside the registered v3 stream map")


def g5_policy_stream_fingerprint(policy: ScientificGenerationPolicy) -> str:
    """Bind one selected policy fingerprint to its registered RNG version."""

    checked = _validated_selected_policy(policy)
    return _fingerprint(
        {
            "schema_id": G5_POLICY_STREAM_SCHEMA_ID,
            "generation_policy_fingerprint": _fingerprint(checked.as_dict()),
            "scientific_version": g5_policy_scientific_version(checked),
        }
    )


def produce_g5_development_policy_selection(
    *,
    baseline_threshold_evidence: G5ThresholdEvidence,
    baseline_qualification_results: CanonicalValidationResults,
    decisions: Sequence[G5DevelopmentPolicyDecision],
) -> G5DevelopmentPolicySelection:
    """Seal the first passing neutral-development row after baseline failure."""

    baseline_threshold = verify_g5_threshold_evidence(baseline_threshold_evidence)
    baseline_results = _verified_results(baseline_qualification_results)
    if (
        baseline_threshold.selected_alpha_policy != "historical_vector"
        or baseline_threshold.policy_scientific_version
        != G5_HISTORICAL_POLICY_SCIENTIFIC_VERSION
        or baseline_results.passed
        or baseline_results.threshold_fingerprint
        != baseline_threshold.threshold_registry_fingerprint
    ):
        raise ModelError("G5 neutral activation requires the failed v11 baseline")
    failures = {
        item.name
        for item in baseline_results.decisions
        if item.required and not item.passed
    }
    required_names = {item.name for item in baseline_results.decisions if item.required}
    if (
        not required_names >= G5_BASELINE_REQUIRED_DECISIONS
        or not failures
        or not failures <= G5_BASELINE_DIRECTIONAL_FAILURES
    ):
        raise ModelError(
            "G5 neutral activation requires only registered G5-A/directional failures"
        )
    rows = tuple(decisions)
    if len(rows) != len(G5_NEUTRAL_POLICY_SCIENTIFIC_VERSIONS):
        raise ModelError("G5 development selection requires all four lambda rows")
    selected: G5DevelopmentPolicyDecision | None = None
    reached_unevaluated = False
    for row, (expected_lambda, expected_version) in zip(
        rows, G5_NEUTRAL_POLICY_SCIENTIFIC_VERSIONS, strict=True
    ):
        if not isinstance(row, G5DevelopmentPolicyDecision):
            raise ModelError("G5 development selection row has the wrong type")
        policy = _validated_selected_policy(row.generation_policy)
        _validate_policy_is_g3_candidate(
            policy, baseline_threshold.g3_evidence.production_artifact
        )
        expected_policy_fingerprint = _fingerprint(policy.as_dict())
        if (
            policy.alpha_policy != "kernel_mean_neutral"
            or policy.selected_neutral_lambda != expected_lambda
            or row.neutral_lambda != expected_lambda
            or row.policy_scientific_version != expected_version
            or row.generation_policy_fingerprint != expected_policy_fingerprint
        ):
            raise ModelError("G5 development selection row identity is invalid")
        if row.evaluated:
            if reached_unevaluated or selected is not None:
                raise ModelError("G5 development selection evaluated after first pass")
            _required_fingerprint(
                row.development_result_fingerprint, "G5 development result"
            )
            if type(row.passed) is not bool:
                raise ModelError("G5 evaluated development row lacks a decision")
            if row.passed:
                selected = row
        else:
            reached_unevaluated = True
            if row.development_result_fingerprint is not None or row.passed is not None:
                raise ModelError("G5 unevaluated development row contains a result")
    if selected is None:
        raise ModelError("G5 neutral development has no passing lambda")
    identity = _development_selection_identity(
        baseline_threshold_evidence=baseline_threshold,
        baseline_results=baseline_results,
        decisions=rows,
        selected_policy=selected.generation_policy,
    )
    result = object.__new__(G5DevelopmentPolicySelection)
    _set_fields(
        result,
        (
            ("schema_id", G5_DEVELOPMENT_SELECTION_SCHEMA_ID),
            ("schema_version", G5_DEVELOPMENT_SELECTION_SCHEMA_VERSION),
            ("g3_evidence", baseline_threshold.g3_evidence),
            ("g3_evidence_fingerprint", baseline_threshold.g3_evidence_fingerprint),
            ("baseline_threshold_evidence", baseline_threshold),
            ("baseline_threshold_evidence_fingerprint", baseline_threshold.fingerprint),
            ("baseline_qualification_results", baseline_results),
            (
                "baseline_qualification_results_fingerprint",
                baseline_results.fingerprint,
            ),
            ("decisions", rows),
            ("selected_policy", selected.generation_policy),
            ("selected_policy_fingerprint", selected.generation_policy_fingerprint),
            ("selected_neutral_lambda", selected.neutral_lambda),
            ("selected_policy_scientific_version", selected.policy_scientific_version),
            ("fingerprint", _fingerprint(identity)),
            ("_seal", _DEVELOPMENT_SELECTION_SEAL),
        ),
    )
    _MINTED_DEVELOPMENT_SELECTIONS[id(result)] = result
    return verify_g5_development_policy_selection(result)


def produce_g5_threshold_evidence(
    **_: object,
) -> G5ThresholdEvidence:
    """Fail closed until an approved literal fresh-history null is sealed.

    The currently implemented finite candidate-reservoir bootstrap is an
    explicitly nonauthorizing empirical approximation.  Threshold authority
    must not be minted from bare canonical-threshold values or from that
    approximation.
    """

    raise ModelError(
        "G5 threshold authority awaits approved sealed literal fresh-history "
        "outer-null evidence"
    )


def _produce_g5_threshold_evidence_for_testing(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    generation_policy: ScientificGenerationPolicy,
    thresholds: CanonicalThresholds,
    estimator_contract_fingerprint: str,
    adversary_fit: G5AdversaryFitPair,
    seed_registry_fingerprint: str,
    g5_source_code_fingerprint: str,
    g5_dependency_lock_fingerprint: str,
    development_selection: G5DevelopmentPolicySelection | None = None,
    _test_capability: object,
) -> G5ThresholdEvidence:
    """Mint a visibly registered noncanonical threshold fixture for tests."""

    if _test_capability is not _G5_NONCANONICAL_TEST_CAPABILITY:
        raise ModelError("G5 noncanonical threshold fixture lacks test capability")

    evidence = verify_loaded_g3_evidence(g3_evidence)
    _verify_g3_production_science_parents(evidence)
    policy = _validated_selected_policy(generation_policy)
    _validate_policy_is_g3_candidate(policy, evidence.production_artifact)
    record = _verified_thresholds(thresholds)
    if not record.precision_passed:
        raise ModelError("G5 threshold evidence requires passed precision audits")
    version = g5_policy_scientific_version(policy)
    policy_fingerprint = _fingerprint(policy.as_dict())
    stream_fingerprint = g5_policy_stream_fingerprint(policy)
    estimator = _required_fingerprint(
        estimator_contract_fingerprint, "G5 estimator contract"
    )
    if estimator != _canonical_estimator_contract_fingerprint():
        raise ModelError("G5 threshold uses a noncanonical estimator contract")
    fit = _verified_adversary_fit(adversary_fit)
    bindings = dict(record.binding_fingerprints)
    required_bindings = {
        G5_THRESHOLD_ESTIMATOR_BINDING: estimator,
        G5_THRESHOLD_POLICY_BINDING: policy_fingerprint,
        G5_THRESHOLD_POLICY_STREAM_BINDING: stream_fingerprint,
        G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING: (
            G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        ),
        G5_THRESHOLD_ADVERSARY_FIT_BINDING: fit.fingerprint,
    }
    if any(bindings.get(name) != value for name, value in required_bindings.items()):
        raise ModelError("G5 threshold record has the wrong estimator/policy stream")
    selection: G5DevelopmentPolicySelection | None = None
    if policy.alpha_policy == "historical_vector":
        if development_selection is not None:
            raise ModelError("G5 historical baseline cannot claim neutral selection")
    else:
        selection = verify_g5_development_policy_selection(development_selection)
        if (
            selection.g3_evidence is not evidence
            or selection.selected_policy != policy
            or selection.selected_policy_scientific_version != version
        ):
            raise ModelError("G5 neutral threshold differs from development selection")
    for value, label in (
        (record.fingerprint, "G5 threshold registry"),
        (seed_registry_fingerprint, "G5 seed registry"),
        (g5_source_code_fingerprint, "G5 source code"),
        (g5_dependency_lock_fingerprint, "G5 dependency lock"),
    ):
        _required_fingerprint(value, label)
    identity = _threshold_evidence_identity(
        evidence=evidence,
        g5_source_code_fingerprint=g5_source_code_fingerprint,
        g5_dependency_lock_fingerprint=g5_dependency_lock_fingerprint,
        seed_registry_fingerprint=seed_registry_fingerprint,
        thresholds=record,
        policy=policy,
        estimator_contract_fingerprint=estimator,
        adversary_fit=fit,
        development_selection=selection,
        outer_null_replicates=G5_OUTER_NULL_REPLICATES,
    )
    result = object.__new__(G5ThresholdEvidence)
    _set_fields(
        result,
        (
            ("schema_id", G5_THRESHOLD_EVIDENCE_SCHEMA_ID),
            ("schema_version", G5_EVIDENCE_SCHEMA_VERSION),
            ("passed", True),
            ("g3_evidence", evidence),
            ("g3_evidence_fingerprint", evidence.reference.fingerprint),
            ("g3_source_code_fingerprint", evidence.g3_source_code_fingerprint),
            (
                "g3_dependency_lock_fingerprint",
                evidence.g3_dependency_lock_fingerprint,
            ),
            ("g5_source_code_fingerprint", g5_source_code_fingerprint),
            ("g5_dependency_lock_fingerprint", g5_dependency_lock_fingerprint),
            ("seed_registry_fingerprint", seed_registry_fingerprint),
            ("thresholds", record),
            ("threshold_registry_fingerprint", record.fingerprint),
            ("generation_policy", policy),
            ("generation_policy_fingerprint", policy_fingerprint),
            ("selected_alpha_policy", policy.alpha_policy),
            ("selected_neutral_lambda", policy.selected_neutral_lambda),
            ("policy_scientific_version", version),
            ("policy_stream_fingerprint", stream_fingerprint),
            ("estimator_contract_fingerprint", estimator),
            (
                "adversary_specification_fingerprint",
                G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
            ),
            ("adversary_fit", fit),
            ("adversary_fit_fingerprint", fit.fingerprint),
            ("threshold_subject_fingerprint", policy_fingerprint),
            ("development_selection", selection),
            (
                "development_selection_fingerprint",
                None if selection is None else selection.fingerprint,
            ),
            (
                "baseline_qualification_results_fingerprint",
                None
                if selection is None
                else selection.baseline_qualification_results_fingerprint,
            ),
            ("outer_null_replicates", G5_OUTER_NULL_REPLICATES),
            ("fingerprint", _fingerprint(identity)),
            ("_seal", _THRESHOLD_EVIDENCE_SEAL),
        ),
    )
    _MINTED_THRESHOLD_EVIDENCE[id(result)] = result
    _G5_NONCANONICAL_TEST_OBJECTS.add(id(result))
    return verify_g5_threshold_evidence(result)


def produce_g5_qualification_evidence(
    **_: object,
) -> G5QualificationEvidence:
    """Fail closed while canonical threshold authority is unavailable."""

    raise ModelError("G5 qualification authority awaits sealed threshold evidence")


def _produce_g5_qualification_evidence_for_testing(
    *,
    threshold_evidence: G5ThresholdEvidence,
    generation_policy: ScientificGenerationPolicy,
    qualification_results: CanonicalValidationResults,
    _test_capability: object,
) -> G5QualificationEvidence:
    """Mint a registered noncanonical qualification fixture for tests."""

    if _test_capability is not _G5_NONCANONICAL_TEST_CAPABILITY:
        raise ModelError("G5 noncanonical qualification fixture lacks capability")
    if id(threshold_evidence) not in _G5_NONCANONICAL_TEST_OBJECTS:
        raise ModelError("G5 test qualification requires a test threshold fixture")

    threshold = verify_g5_threshold_evidence(threshold_evidence)
    policy = _validated_selected_policy(generation_policy)
    _validate_policy_is_g3_candidate(policy, threshold.g3_evidence.production_artifact)
    results = _verified_results(qualification_results)
    fit = _verified_adversary_fit(threshold.adversary_fit)
    if not results.passed:
        raise ModelError("G5 qualification evidence requires a passed canonical result")
    if (
        policy != threshold.generation_policy
        or results.threshold_fingerprint != threshold.threshold_registry_fingerprint
        or dict(results.metric_fingerprints).get("adversary_fit") != fit.fingerprint
        or fit.specification_fingerprint
        != threshold.adversary_specification_fingerprint
    ):
        raise ModelError("G5 qualification policy/threshold parent disagrees")
    identity = _qualification_evidence_identity(
        threshold_evidence=threshold,
        policy=policy,
        qualification_results=results,
        adversary_fit=fit,
    )
    result = object.__new__(G5QualificationEvidence)
    _set_fields(
        result,
        (
            ("schema_id", G5_QUALIFICATION_EVIDENCE_SCHEMA_ID),
            ("schema_version", G5_EVIDENCE_SCHEMA_VERSION),
            ("passed", True),
            ("threshold_evidence", threshold),
            ("threshold_evidence_fingerprint", threshold.fingerprint),
            ("g3_evidence", threshold.g3_evidence),
            ("g3_evidence_fingerprint", threshold.g3_evidence_fingerprint),
            ("generation_policy", policy),
            ("generation_policy_fingerprint", _fingerprint(policy.as_dict())),
            ("selected_alpha_policy", policy.alpha_policy),
            ("selected_neutral_lambda", policy.selected_neutral_lambda),
            ("seed_registry_fingerprint", threshold.seed_registry_fingerprint),
            ("qualification_results", results),
            ("qualification_results_fingerprint", results.fingerprint),
            ("qualification_subject_fingerprint", results.subject_fingerprint),
            ("adversary_fit", fit),
            ("adversary_fit_fingerprint", fit.fingerprint),
            ("development_lf_path_count", G5_DEVELOPMENT_LF_PATH_COUNT),
            ("development_hf_path_count", G5_DEVELOPMENT_HF_PATH_COUNT),
            ("qualification_lf_path_count", G5_QUALIFICATION_LF_PATH_COUNT),
            ("qualification_hf_path_count", G5_QUALIFICATION_HF_PATH_COUNT),
            (
                "g5_source_code_fingerprint",
                threshold.g5_source_code_fingerprint,
            ),
            (
                "g5_dependency_lock_fingerprint",
                threshold.g5_dependency_lock_fingerprint,
            ),
            ("fingerprint", _fingerprint(identity)),
            ("_seal", _QUALIFICATION_EVIDENCE_SEAL),
        ),
    )
    _MINTED_QUALIFICATION_EVIDENCE[id(result)] = result
    _G5_NONCANONICAL_TEST_OBJECTS.add(id(result))
    return verify_g5_qualification_evidence(result)


def produce_g5_meta_validation_evidence(
    *,
    threshold_evidence: G5ThresholdEvidence,
    qualification_evidence: G5QualificationEvidence,
    meta_validation_results: CanonicalValidationResults,
) -> G5MetaValidationEvidence:
    """Reject authority until the concrete sealed meta executor is integrated."""

    del threshold_evidence, qualification_evidence, meta_validation_results
    raise ModelError(
        "G5 meta authority awaits sealed purpose-12/13 and K3 execution evidence"
    )


def _produce_g5_meta_validation_evidence_for_testing(
    *,
    threshold_evidence: G5ThresholdEvidence,
    qualification_evidence: G5QualificationEvidence,
    meta_validation_results: CanonicalValidationResults,
    _test_capability: object,
) -> G5MetaValidationEvidence:
    """Test-only fixture mint; production finalization cannot call this helper."""

    if _test_capability is not _G5_NONCANONICAL_TEST_CAPABILITY:
        raise ModelError("G5 noncanonical meta fixture lacks test capability")
    if (
        id(threshold_evidence) not in _G5_NONCANONICAL_TEST_OBJECTS
        or id(qualification_evidence) not in _G5_NONCANONICAL_TEST_OBJECTS
    ):
        raise ModelError("G5 test meta requires test-only exact parents")
    threshold = verify_g5_threshold_evidence(threshold_evidence)
    qualification = verify_g5_qualification_evidence(qualification_evidence)
    if qualification.threshold_evidence is not threshold:
        raise ModelError("G5 meta-validation evidence has a different threshold parent")
    results = _verified_results(meta_validation_results)
    if not results.passed:
        raise ModelError("G5 meta-validation evidence requires a passed result")
    if (
        results.threshold_fingerprint != threshold.threshold_registry_fingerprint
        or results.subject_fingerprint
        != qualification.qualification_results_fingerprint
    ):
        raise ModelError("G5 meta-validation result has the wrong exact parents")
    identity = _meta_validation_evidence_identity(
        threshold_evidence=threshold,
        qualification_evidence=qualification,
        meta_validation_results=results,
        outer_null_replicates=G5_OUTER_NULL_REPLICATES,
    )
    result = object.__new__(G5MetaValidationEvidence)
    _set_fields(
        result,
        (
            ("schema_id", G5_META_VALIDATION_EVIDENCE_SCHEMA_ID),
            ("schema_version", G5_EVIDENCE_SCHEMA_VERSION),
            ("passed", True),
            ("threshold_evidence", threshold),
            ("threshold_evidence_fingerprint", threshold.fingerprint),
            ("qualification_evidence", qualification),
            ("qualification_evidence_fingerprint", qualification.fingerprint),
            ("g3_evidence", threshold.g3_evidence),
            ("g3_evidence_fingerprint", threshold.g3_evidence_fingerprint),
            (
                "generation_policy_fingerprint",
                qualification.generation_policy_fingerprint,
            ),
            ("seed_registry_fingerprint", threshold.seed_registry_fingerprint),
            ("meta_validation_results", results),
            ("meta_validation_fingerprint", results.fingerprint),
            ("meta_validation_subject_fingerprint", results.subject_fingerprint),
            ("outer_null_replicates", G5_OUTER_NULL_REPLICATES),
            (
                "g5_source_code_fingerprint",
                threshold.g5_source_code_fingerprint,
            ),
            (
                "g5_dependency_lock_fingerprint",
                threshold.g5_dependency_lock_fingerprint,
            ),
            ("fingerprint", _fingerprint(identity)),
            ("_seal", _META_VALIDATION_EVIDENCE_SEAL),
        ),
    )
    _MINTED_META_VALIDATION_EVIDENCE[id(result)] = result
    _G5_NONCANONICAL_TEST_OBJECTS.add(id(result))
    return verify_g5_meta_validation_evidence(result)


def produce_g5_qualification_generation_authority(
    **_: object,
) -> G5QualificationGenerationAuthority:
    """Fail closed while canonical threshold authority is unavailable."""

    raise ModelError("G5 candidate qualification awaits sealed threshold evidence")


def _produce_g5_qualification_generation_authority_for_testing(
    *,
    threshold_evidence: G5ThresholdEvidence,
    _test_capability: object,
) -> G5QualificationGenerationAuthority:
    """Mint bounded noncanonical candidate authority for serial unit tests."""

    if _test_capability is not _G5_NONCANONICAL_TEST_CAPABILITY:
        raise ModelError("G5 noncanonical candidate fixture lacks test capability")
    if id(threshold_evidence) not in _G5_NONCANONICAL_TEST_OBJECTS:
        raise ModelError("G5 test candidate authority requires a test threshold")

    threshold = verify_g5_threshold_evidence(threshold_evidence)
    evidence = threshold.g3_evidence
    artifact = evidence.production_artifact
    candidates = artifact.generation_policy
    if not isinstance(candidates, ScientificCandidatePolicySet):
        raise ModelError("G5 qualification requires the registered candidate set")
    identity = _qualification_authority_identity(
        evidence=evidence,
        artifact=artifact,
        candidates=candidates,
        threshold_evidence=threshold,
    )
    result = object.__new__(G5QualificationGenerationAuthority)
    _set_fields(
        result,
        (
            ("schema_id", G5_QUALIFICATION_AUTHORITY_SCHEMA_ID),
            ("schema_version", G5_QUALIFICATION_AUTHORITY_SCHEMA_VERSION),
            ("g3_evidence", evidence),
            ("g3_evidence_fingerprint", evidence.reference.fingerprint),
            ("pair_receipt_fingerprint", evidence.pair_receipt_fingerprint),
            ("production_artifact", artifact),
            ("production_artifact_fingerprint", artifact.reference.fingerprint),
            ("parameter_set_fingerprint", artifact.parameter_set_fingerprint),
            ("candidate_policy_set", candidates),
            ("candidate_policy_set_fingerprint", _fingerprint(candidates.as_dict())),
            ("generation_policy", threshold.generation_policy),
            (
                "generation_policy_fingerprint",
                threshold.generation_policy_fingerprint,
            ),
            ("selected_alpha_policy", threshold.selected_alpha_policy),
            ("selected_neutral_lambda", threshold.selected_neutral_lambda),
            ("policy_scientific_version", threshold.policy_scientific_version),
            ("policy_stream_fingerprint", threshold.policy_stream_fingerprint),
            ("adversary_fit", threshold.adversary_fit),
            ("adversary_fit_fingerprint", threshold.adversary_fit_fingerprint),
            ("qualification_namespace", G5_QUALIFICATION_NAMESPACE),
            ("seed_registry_fingerprint", threshold.seed_registry_fingerprint),
            (
                "threshold_registry_fingerprint",
                threshold.threshold_registry_fingerprint,
            ),
            ("threshold_evidence", threshold),
            ("threshold_evidence_fingerprint", threshold.fingerprint),
            ("development_lf_path_count", G5_DEVELOPMENT_LF_PATH_COUNT),
            ("development_hf_path_count", G5_DEVELOPMENT_HF_PATH_COUNT),
            ("qualification_lf_path_count", G5_QUALIFICATION_LF_PATH_COUNT),
            ("qualification_hf_path_count", G5_QUALIFICATION_HF_PATH_COUNT),
            (
                "g5_source_code_fingerprint",
                threshold.g5_source_code_fingerprint,
            ),
            (
                "g5_dependency_lock_fingerprint",
                threshold.g5_dependency_lock_fingerprint,
            ),
            ("fingerprint", _fingerprint(identity)),
            ("_seal", _G5_QUALIFICATION_AUTHORITY_SEAL),
        ),
    )
    _MINTED_G5_QUALIFICATION_AUTHORITIES[id(result)] = result
    _G5_NONCANONICAL_TEST_OBJECTS.add(id(result))
    return verify_g5_qualification_generation_authority(result)


def produce_g5_qualified_production_authority(
    **_: object,
) -> G5QualifiedProductionAuthority:
    """Fail closed pending approved null and combined meta evidence."""

    raise ModelError(
        "G5 final authority awaits approved literal-null and sealed combined "
        "meta-validation evidence"
    )


def _produce_g5_qualified_production_authority_for_testing(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    generation_policy: ScientificGenerationPolicy,
    threshold_evidence: G5ThresholdEvidence,
    qualification_evidence: G5QualificationEvidence,
    meta_validation_evidence: G5MetaValidationEvidence,
    _test_capability: object,
) -> G5QualifiedProductionAuthority:
    """Mint a registered noncanonical final fixture for serial unit tests."""

    if _test_capability is not _G5_NONCANONICAL_TEST_CAPABILITY:
        raise ModelError("G5 noncanonical final fixture lacks test capability")
    if any(
        id(parent) not in _G5_NONCANONICAL_TEST_OBJECTS
        for parent in (
            threshold_evidence,
            qualification_evidence,
            meta_validation_evidence,
        )
    ):
        raise ModelError("G5 test final authority requires test-only exact parents")

    evidence = verify_loaded_g3_evidence(g3_evidence)
    threshold = verify_g5_threshold_evidence(threshold_evidence)
    qualification = verify_g5_qualification_evidence(qualification_evidence)
    meta = verify_g5_meta_validation_evidence(meta_validation_evidence)
    policy = _validated_selected_policy(generation_policy)
    artifact = evidence.production_artifact
    _validate_policy_is_g3_candidate(policy, artifact)
    if (
        threshold.g3_evidence is not evidence
        or qualification.threshold_evidence is not threshold
        or meta.threshold_evidence is not threshold
        or meta.qualification_evidence is not qualification
        or qualification.generation_policy != policy
        or threshold.generation_policy != policy
        or threshold.generation_policy != policy
        or qualification.generation_policy_fingerprint != _fingerprint(policy.as_dict())
    ):
        raise ModelError("G5 final evidence parents or selected policy disagree")
    identity = _authority_identity(
        evidence=evidence,
        artifact=artifact,
        policy=policy,
        threshold_evidence=threshold,
        qualification_evidence=qualification,
        meta_validation_evidence=meta,
    )
    result = object.__new__(G5QualifiedProductionAuthority)
    _set_fields(
        result,
        (
            ("schema_id", G5_GENERATION_AUTHORITY_SCHEMA_ID),
            ("schema_version", G5_GENERATION_AUTHORITY_SCHEMA_VERSION),
            ("qualification_gate", "G5"),
            ("g3_evidence", evidence),
            ("g3_evidence_fingerprint", evidence.reference.fingerprint),
            ("g3_bundle_fingerprint", evidence.evidence_bundle.fingerprint),
            ("pair_receipt_fingerprint", evidence.pair_receipt_fingerprint),
            ("g3_source_code_fingerprint", evidence.g3_source_code_fingerprint),
            (
                "g3_dependency_lock_fingerprint",
                evidence.g3_dependency_lock_fingerprint,
            ),
            ("production_artifact", artifact),
            ("production_artifact_fingerprint", artifact.reference.fingerprint),
            ("core_model_fingerprint", artifact.core_model_fingerprint),
            ("parameter_set_fingerprint", artifact.parameter_set_fingerprint),
            ("generation_policy", policy),
            ("generation_policy_fingerprint", _fingerprint(policy.as_dict())),
            ("selected_alpha_policy", policy.alpha_policy),
            ("selected_neutral_lambda", policy.selected_neutral_lambda),
            ("policy_scientific_version", threshold.policy_scientific_version),
            ("policy_stream_fingerprint", threshold.policy_stream_fingerprint),
            (
                "estimator_contract_fingerprint",
                threshold.estimator_contract_fingerprint,
            ),
            (
                "adversary_specification_fingerprint",
                threshold.adversary_specification_fingerprint,
            ),
            (
                "adversary_fit_fingerprint",
                qualification.adversary_fit_fingerprint,
            ),
            ("seed_registry_fingerprint", threshold.seed_registry_fingerprint),
            ("development_lf_path_count", G5_DEVELOPMENT_LF_PATH_COUNT),
            ("development_hf_path_count", G5_DEVELOPMENT_HF_PATH_COUNT),
            ("qualification_lf_path_count", G5_QUALIFICATION_LF_PATH_COUNT),
            ("qualification_hf_path_count", G5_QUALIFICATION_HF_PATH_COUNT),
            ("outer_null_replicates", G5_OUTER_NULL_REPLICATES),
            (
                "g5_source_code_fingerprint",
                threshold.g5_source_code_fingerprint,
            ),
            (
                "g5_dependency_lock_fingerprint",
                threshold.g5_dependency_lock_fingerprint,
            ),
            ("threshold_evidence", threshold),
            ("threshold_evidence_fingerprint", threshold.fingerprint),
            ("qualification_evidence", qualification),
            ("qualification_evidence_fingerprint", qualification.fingerprint),
            ("meta_validation_evidence", meta),
            ("meta_validation_evidence_fingerprint", meta.fingerprint),
            (
                "threshold_registry_fingerprint",
                threshold.threshold_registry_fingerprint,
            ),
            (
                "threshold_subject_fingerprint",
                threshold.threshold_subject_fingerprint,
            ),
            (
                "qualification_results_fingerprint",
                qualification.qualification_results_fingerprint,
            ),
            (
                "qualification_subject_fingerprint",
                qualification.qualification_subject_fingerprint,
            ),
            ("meta_validation_fingerprint", meta.meta_validation_fingerprint),
            (
                "meta_validation_subject_fingerprint",
                meta.meta_validation_subject_fingerprint,
            ),
            (
                "development_selection_fingerprint",
                threshold.development_selection_fingerprint,
            ),
            (
                "baseline_qualification_results_fingerprint",
                threshold.baseline_qualification_results_fingerprint,
            ),
            (
                "g3_production_latent_task_fingerprint",
                evidence.task_result_fingerprints[G5_PRODUCTION_LATENT_TASK_ID],
            ),
            (
                "g3_production_latent_checkpoint_fingerprint",
                evidence.checkpoint_fingerprints[G5_PRODUCTION_LATENT_TASK_ID],
            ),
            (
                "g3_production_adequacy_task_fingerprint",
                evidence.task_result_fingerprints[G5_PRODUCTION_ADEQUACY_TASK_ID],
            ),
            (
                "g3_production_adequacy_checkpoint_fingerprint",
                evidence.checkpoint_fingerprints[G5_PRODUCTION_ADEQUACY_TASK_ID],
            ),
            ("fingerprint", _fingerprint(identity)),
            ("_seal", _G5_AUTHORITY_SEAL),
        ),
    )
    _MINTED_G5_AUTHORITIES[id(result)] = result
    _G5_NONCANONICAL_TEST_OBJECTS.add(id(result))
    return verify_g5_qualified_production_authority(result)


def verify_g5_development_policy_selection(
    value: object,
) -> G5DevelopmentPolicySelection:
    """Revalidate one original baseline-triggered neutral selection."""

    if (
        type(value) is not G5DevelopmentPolicySelection
        or value._seal is not _DEVELOPMENT_SELECTION_SEAL
        or value._owner_id != id(value)
        or _MINTED_DEVELOPMENT_SELECTIONS.get(id(value)) is not value
    ):
        raise ModelError("G5 development selection lacks producer provenance")
    baseline_threshold = verify_g5_threshold_evidence(value.baseline_threshold_evidence)
    baseline_results = _verified_results(value.baseline_qualification_results)
    selected_policy = _validated_selected_policy(value.selected_policy)
    for item, label in (
        (value.g3_evidence_fingerprint, "G3 evidence"),
        (value.baseline_threshold_evidence_fingerprint, "G5 baseline threshold"),
        (
            value.baseline_qualification_results_fingerprint,
            "G5 baseline qualification result",
        ),
        (value.selected_policy_fingerprint, "G5 selected policy"),
        (value.fingerprint, "G5 development selection"),
    ):
        _required_fingerprint(item, label)
    failures = {
        item.name
        for item in baseline_results.decisions
        if item.required and not item.passed
    }
    required_names = {item.name for item in baseline_results.decisions if item.required}
    selected_rows = [
        row for row in value.decisions if row.evaluated and row.passed is True
    ]
    identity = _development_selection_identity(
        baseline_threshold_evidence=baseline_threshold,
        baseline_results=baseline_results,
        decisions=value.decisions,
        selected_policy=selected_policy,
    )
    if (
        value.schema_id != G5_DEVELOPMENT_SELECTION_SCHEMA_ID
        or value.schema_version != G5_DEVELOPMENT_SELECTION_SCHEMA_VERSION
        or value.g3_evidence is not baseline_threshold.g3_evidence
        or value.g3_evidence_fingerprint != baseline_threshold.g3_evidence_fingerprint
        or baseline_threshold.selected_alpha_policy != "historical_vector"
        or baseline_results.passed
        or baseline_results.threshold_fingerprint
        != baseline_threshold.threshold_registry_fingerprint
        or not required_names >= G5_BASELINE_REQUIRED_DECISIONS
        or not failures
        or not failures <= G5_BASELINE_DIRECTIONAL_FAILURES
        or len(selected_rows) != 1
        or selected_rows[0].generation_policy != selected_policy
        or value.selected_policy_fingerprint != _fingerprint(selected_policy.as_dict())
        or value.selected_neutral_lambda != selected_policy.selected_neutral_lambda
        or value.selected_policy_scientific_version
        != g5_policy_scientific_version(selected_policy)
        or value.baseline_threshold_evidence_fingerprint
        != baseline_threshold.fingerprint
        or value.baseline_qualification_results_fingerprint
        != baseline_results.fingerprint
        or value.fingerprint != _fingerprint(identity)
    ):
        raise IntegrityError("G5 development selection identity changed")
    # Reuse the producer's full row/order/evaluation validation without
    # minting a substitute: the deterministic identity check above plus this
    # explicit prefix rule closes post-mint field substitution.
    passed_seen = False
    unevaluated_seen = False
    for row, (neutral_lambda, version) in zip(
        value.decisions, G5_NEUTRAL_POLICY_SCIENTIFIC_VERSIONS, strict=True
    ):
        policy = _validated_selected_policy(row.generation_policy)
        if (
            row.neutral_lambda != neutral_lambda
            or policy.selected_neutral_lambda != neutral_lambda
            or policy.alpha_policy != "kernel_mean_neutral"
            or row.policy_scientific_version != version
            or row.generation_policy_fingerprint != _fingerprint(policy.as_dict())
        ):
            raise IntegrityError("G5 development selection row changed")
        if row.evaluated:
            if unevaluated_seen or passed_seen or type(row.passed) is not bool:
                raise IntegrityError("G5 development selection order changed")
            _required_fingerprint(
                row.development_result_fingerprint, "G5 development result"
            )
            passed_seen = row.passed
        else:
            unevaluated_seen = True
            if row.development_result_fingerprint is not None or row.passed is not None:
                raise IntegrityError("G5 unevaluated selection row changed")
    return value


def verify_g5_threshold_evidence(value: object) -> G5ThresholdEvidence:
    """Revalidate one original, passed threshold-evidence object."""

    if (
        type(value) is not G5ThresholdEvidence
        or value._seal is not _THRESHOLD_EVIDENCE_SEAL
        or value._owner_id != id(value)
        or _MINTED_THRESHOLD_EVIDENCE.get(id(value)) is not value
    ):
        raise ModelError("G5 threshold evidence lacks producer provenance")
    evidence = verify_loaded_g3_evidence(value.g3_evidence)
    _verify_g3_production_science_parents(evidence)
    thresholds = _verified_thresholds(value.thresholds)
    policy = _validated_selected_policy(value.generation_policy)
    fit = _verified_adversary_fit(value.adversary_fit)
    _validate_policy_is_g3_candidate(policy, evidence.production_artifact)
    selection = value.development_selection
    if selection is not None:
        selection = verify_g5_development_policy_selection(selection)
    for item, label in (
        (value.g3_evidence_fingerprint, "G3 evidence"),
        (value.g3_source_code_fingerprint, "G3 source code"),
        (value.g3_dependency_lock_fingerprint, "G3 dependency lock"),
        (value.g5_source_code_fingerprint, "G5 source code"),
        (value.g5_dependency_lock_fingerprint, "G5 dependency lock"),
        (value.seed_registry_fingerprint, "G5 seed registry"),
        (value.threshold_registry_fingerprint, "G5 threshold registry"),
        (value.generation_policy_fingerprint, "G5 generation policy"),
        (value.policy_stream_fingerprint, "G5 policy stream"),
        (value.estimator_contract_fingerprint, "G5 estimator contract"),
        (
            value.adversary_specification_fingerprint,
            "G5 adversary specification",
        ),
        (value.adversary_fit_fingerprint, "G5 adversary fit"),
        (value.threshold_subject_fingerprint, "G5 threshold subject"),
        (value.fingerprint, "G5 threshold evidence"),
    ):
        _required_fingerprint(item, label)
    identity = value.identity_dict()
    if (
        value.schema_id != G5_THRESHOLD_EVIDENCE_SCHEMA_ID
        or value.schema_version != G5_EVIDENCE_SCHEMA_VERSION
        or value.passed is not True
        or value.outer_null_replicates != G5_OUTER_NULL_REPLICATES
        or not thresholds.precision_passed
        or value.g3_evidence_fingerprint != evidence.reference.fingerprint
        or value.g3_source_code_fingerprint != evidence.g3_source_code_fingerprint
        or value.g3_dependency_lock_fingerprint
        != evidence.g3_dependency_lock_fingerprint
        or value.threshold_registry_fingerprint != thresholds.fingerprint
        or value.generation_policy_fingerprint != _fingerprint(policy.as_dict())
        or value.selected_alpha_policy != policy.alpha_policy
        or value.selected_neutral_lambda != policy.selected_neutral_lambda
        or value.policy_scientific_version != g5_policy_scientific_version(policy)
        or value.policy_stream_fingerprint != g5_policy_stream_fingerprint(policy)
        or value.threshold_subject_fingerprint != value.generation_policy_fingerprint
        or dict(thresholds.binding_fingerprints).get(G5_THRESHOLD_ESTIMATOR_BINDING)
        != value.estimator_contract_fingerprint
        or dict(thresholds.binding_fingerprints).get(G5_THRESHOLD_POLICY_BINDING)
        != value.generation_policy_fingerprint
        or dict(thresholds.binding_fingerprints).get(G5_THRESHOLD_POLICY_STREAM_BINDING)
        != value.policy_stream_fingerprint
        or value.adversary_specification_fingerprint
        != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        or dict(thresholds.binding_fingerprints).get(
            G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING
        )
        != value.adversary_specification_fingerprint
        or value.adversary_fit_fingerprint != fit.fingerprint
        or dict(thresholds.binding_fingerprints).get(G5_THRESHOLD_ADVERSARY_FIT_BINDING)
        != fit.fingerprint
        or (policy.alpha_policy == "historical_vector" and selection is not None)
        or (policy.alpha_policy == "kernel_mean_neutral" and selection is None)
        or (
            selection is not None
            and (
                selection.g3_evidence is not evidence
                or selection.selected_policy != policy
                or value.development_selection_fingerprint != selection.fingerprint
                or value.baseline_qualification_results_fingerprint
                != selection.baseline_qualification_results_fingerprint
            )
        )
        or (
            selection is None
            and (
                value.development_selection_fingerprint is not None
                or value.baseline_qualification_results_fingerprint is not None
            )
        )
        or value.fingerprint != _fingerprint(identity)
    ):
        raise IntegrityError("G5 threshold evidence identity changed")
    return value


def verify_g5_qualification_evidence(value: object) -> G5QualificationEvidence:
    """Revalidate one original, passed qualification-result object."""

    if (
        type(value) is not G5QualificationEvidence
        or value._seal is not _QUALIFICATION_EVIDENCE_SEAL
        or value._owner_id != id(value)
        or _MINTED_QUALIFICATION_EVIDENCE.get(id(value)) is not value
    ):
        raise ModelError("G5 qualification evidence lacks producer provenance")
    threshold = verify_g5_threshold_evidence(value.threshold_evidence)
    policy = _validated_selected_policy(value.generation_policy)
    _validate_policy_is_g3_candidate(policy, threshold.g3_evidence.production_artifact)
    results = _verified_results(value.qualification_results)
    fit = _verified_adversary_fit(value.adversary_fit)
    for item, label in (
        (value.threshold_evidence_fingerprint, "G5 threshold evidence"),
        (value.g3_evidence_fingerprint, "G3 evidence"),
        (value.generation_policy_fingerprint, "G5 generation policy"),
        (value.seed_registry_fingerprint, "G5 seed registry"),
        (value.qualification_results_fingerprint, "G5 qualification results"),
        (value.qualification_subject_fingerprint, "G5 qualification subject"),
        (value.adversary_fit_fingerprint, "G5 adversary fit"),
        (value.g5_source_code_fingerprint, "G5 source code"),
        (value.g5_dependency_lock_fingerprint, "G5 dependency lock"),
        (value.fingerprint, "G5 qualification evidence"),
    ):
        _required_fingerprint(item, label)
    if (
        value.schema_id != G5_QUALIFICATION_EVIDENCE_SCHEMA_ID
        or value.schema_version != G5_EVIDENCE_SCHEMA_VERSION
        or value.passed is not True
        or value.threshold_evidence_fingerprint != threshold.fingerprint
        or value.g3_evidence is not threshold.g3_evidence
        or value.g3_evidence_fingerprint != threshold.g3_evidence_fingerprint
        or value.generation_policy_fingerprint != _fingerprint(policy.as_dict())
        or policy != threshold.generation_policy
        or value.selected_alpha_policy != policy.alpha_policy
        or value.selected_neutral_lambda != policy.selected_neutral_lambda
        or value.seed_registry_fingerprint != threshold.seed_registry_fingerprint
        or not results.passed
        or results.threshold_fingerprint != threshold.threshold_registry_fingerprint
        or value.qualification_results_fingerprint != results.fingerprint
        or value.qualification_subject_fingerprint != results.subject_fingerprint
        or value.adversary_fit_fingerprint != fit.fingerprint
        or value.adversary_fit is not threshold.adversary_fit
        or dict(results.metric_fingerprints).get("adversary_fit") != fit.fingerprint
        or fit.specification_fingerprint
        != threshold.adversary_specification_fingerprint
        or value.development_lf_path_count != G5_DEVELOPMENT_LF_PATH_COUNT
        or value.development_hf_path_count != G5_DEVELOPMENT_HF_PATH_COUNT
        or value.qualification_lf_path_count != G5_QUALIFICATION_LF_PATH_COUNT
        or value.qualification_hf_path_count != G5_QUALIFICATION_HF_PATH_COUNT
        or value.g5_source_code_fingerprint != threshold.g5_source_code_fingerprint
        or value.g5_dependency_lock_fingerprint
        != threshold.g5_dependency_lock_fingerprint
        or value.fingerprint != _fingerprint(value.identity_dict())
    ):
        raise IntegrityError("G5 qualification evidence identity changed")
    return value


def verify_g5_meta_validation_evidence(value: object) -> G5MetaValidationEvidence:
    """Revalidate one original, passed meta-validation object."""

    if (
        type(value) is not G5MetaValidationEvidence
        or value._seal is not _META_VALIDATION_EVIDENCE_SEAL
        or value._owner_id != id(value)
        or _MINTED_META_VALIDATION_EVIDENCE.get(id(value)) is not value
    ):
        raise ModelError("G5 meta-validation evidence lacks producer provenance")
    threshold = verify_g5_threshold_evidence(value.threshold_evidence)
    qualification = verify_g5_qualification_evidence(value.qualification_evidence)
    results = _verified_results(value.meta_validation_results)
    for item, label in (
        (value.threshold_evidence_fingerprint, "G5 threshold evidence"),
        (value.qualification_evidence_fingerprint, "G5 qualification evidence"),
        (value.g3_evidence_fingerprint, "G3 evidence"),
        (value.generation_policy_fingerprint, "G5 generation policy"),
        (value.seed_registry_fingerprint, "G5 seed registry"),
        (value.meta_validation_fingerprint, "G5 meta-validation"),
        (value.meta_validation_subject_fingerprint, "G5 meta subject"),
        (value.g5_source_code_fingerprint, "G5 source code"),
        (value.g5_dependency_lock_fingerprint, "G5 dependency lock"),
        (value.fingerprint, "G5 meta-validation evidence"),
    ):
        _required_fingerprint(item, label)
    if (
        value.schema_id != G5_META_VALIDATION_EVIDENCE_SCHEMA_ID
        or value.schema_version != G5_EVIDENCE_SCHEMA_VERSION
        or value.passed is not True
        or value.threshold_evidence_fingerprint != threshold.fingerprint
        or qualification.threshold_evidence is not threshold
        or value.qualification_evidence_fingerprint != qualification.fingerprint
        or value.g3_evidence is not threshold.g3_evidence
        or value.g3_evidence_fingerprint != threshold.g3_evidence_fingerprint
        or value.generation_policy_fingerprint
        != qualification.generation_policy_fingerprint
        or value.seed_registry_fingerprint != threshold.seed_registry_fingerprint
        or not results.passed
        or results.threshold_fingerprint != threshold.threshold_registry_fingerprint
        or results.subject_fingerprint
        != qualification.qualification_results_fingerprint
        or value.meta_validation_fingerprint != results.fingerprint
        or value.meta_validation_subject_fingerprint != results.subject_fingerprint
        or value.outer_null_replicates != G5_OUTER_NULL_REPLICATES
        or value.g5_source_code_fingerprint != threshold.g5_source_code_fingerprint
        or value.g5_dependency_lock_fingerprint
        != threshold.g5_dependency_lock_fingerprint
        or value.fingerprint != _fingerprint(value.identity_dict())
    ):
        raise IntegrityError("G5 meta-validation evidence identity changed")
    return value


def verify_g5_qualification_generation_authority(
    value: object,
) -> G5QualificationGenerationAuthority:
    """Revalidate bounded qualification-only authority and exact G3 parents."""

    if (
        type(value) is not G5QualificationGenerationAuthority
        or value._seal is not _G5_QUALIFICATION_AUTHORITY_SEAL
        or value._owner_id != id(value)
        or _MINTED_G5_QUALIFICATION_AUTHORITIES.get(id(value)) is not value
    ):
        raise ModelError("G5 qualification generation lacks sealed authority")
    threshold = verify_g5_threshold_evidence(value.threshold_evidence)
    evidence = verify_loaded_g3_evidence(value.g3_evidence)
    artifact = evidence.production_artifact
    candidates = artifact.generation_policy
    policy = _validated_selected_policy(value.generation_policy)
    if (
        threshold.g3_evidence is not evidence
        or artifact is not value.production_artifact
        or not is_verified_scientific_model_artifact(artifact)
        or artifact.role != "production"
        or not isinstance(candidates, ScientificCandidatePolicySet)
        or value.candidate_policy_set != candidates
        or policy != threshold.generation_policy
    ):
        raise IntegrityError("G5 qualification candidate/artifact binding changed")
    for item, label in (
        (value.g3_evidence_fingerprint, "G3 evidence"),
        (value.pair_receipt_fingerprint, "scientific artifact pair receipt"),
        (value.production_artifact_fingerprint, "production artifact"),
        (value.parameter_set_fingerprint, "parameter set"),
        (value.candidate_policy_set_fingerprint, "candidate policy set"),
        (value.generation_policy_fingerprint, "selected generation policy"),
        (value.policy_stream_fingerprint, "selected policy stream"),
        (value.adversary_fit_fingerprint, "G5 adversary fit"),
        (value.seed_registry_fingerprint, "G5 seed registry"),
        (value.threshold_registry_fingerprint, "G5 threshold registry"),
        (value.threshold_evidence_fingerprint, "G5 threshold evidence"),
        (value.g5_source_code_fingerprint, "G5 source code"),
        (value.g5_dependency_lock_fingerprint, "G5 dependency lock"),
        (value.fingerprint, "G5 qualification authority"),
    ):
        _required_fingerprint(item, label)
    expected = _qualification_authority_identity(
        evidence=evidence,
        artifact=artifact,
        candidates=candidates,
        threshold_evidence=threshold,
    )
    if (
        value.schema_id != G5_QUALIFICATION_AUTHORITY_SCHEMA_ID
        or value.schema_version != G5_QUALIFICATION_AUTHORITY_SCHEMA_VERSION
        or value.qualification_namespace != G5_QUALIFICATION_NAMESPACE
        or value.g3_evidence_fingerprint != evidence.reference.fingerprint
        or value.pair_receipt_fingerprint != evidence.pair_receipt_fingerprint
        or value.production_artifact_fingerprint != artifact.reference.fingerprint
        or value.parameter_set_fingerprint != artifact.parameter_set_fingerprint
        or value.candidate_policy_set_fingerprint != _fingerprint(candidates.as_dict())
        or value.generation_policy_fingerprint
        != threshold.generation_policy_fingerprint
        or value.selected_alpha_policy != policy.alpha_policy
        or value.selected_neutral_lambda != policy.selected_neutral_lambda
        or value.policy_scientific_version != threshold.policy_scientific_version
        or value.policy_stream_fingerprint != threshold.policy_stream_fingerprint
        or value.adversary_fit is not threshold.adversary_fit
        or value.adversary_fit_fingerprint != threshold.adversary_fit_fingerprint
        or value.seed_registry_fingerprint != threshold.seed_registry_fingerprint
        or value.threshold_registry_fingerprint
        != threshold.threshold_registry_fingerprint
        or value.threshold_evidence_fingerprint != threshold.fingerprint
        or value.development_lf_path_count != G5_DEVELOPMENT_LF_PATH_COUNT
        or value.development_hf_path_count != G5_DEVELOPMENT_HF_PATH_COUNT
        or value.qualification_lf_path_count != G5_QUALIFICATION_LF_PATH_COUNT
        or value.qualification_hf_path_count != G5_QUALIFICATION_HF_PATH_COUNT
        or value.g5_source_code_fingerprint != threshold.g5_source_code_fingerprint
        or value.g5_dependency_lock_fingerprint
        != threshold.g5_dependency_lock_fingerprint
        or value.fingerprint != _fingerprint(expected)
    ):
        raise IntegrityError("G5 qualification generation identity changed")
    return value


def verify_g5_qualified_production_authority(
    value: object,
) -> G5QualifiedProductionAuthority:
    """Revalidate exact final authority, passed evidence, and live G3 parents."""

    if (
        type(value) is not G5QualifiedProductionAuthority
        or value._seal is not _G5_AUTHORITY_SEAL
        or value._owner_id != id(value)
        or _MINTED_G5_AUTHORITIES.get(id(value)) is not value
    ):
        raise ModelError("generation requires an original sealed G5 authority")
    evidence = verify_loaded_g3_evidence(value.g3_evidence)
    threshold = verify_g5_threshold_evidence(value.threshold_evidence)
    qualification = verify_g5_qualification_evidence(value.qualification_evidence)
    meta = verify_g5_meta_validation_evidence(value.meta_validation_evidence)
    artifact = value.production_artifact
    policy = _validated_selected_policy(value.generation_policy)
    _validate_policy_is_g3_candidate(policy, artifact)
    if (
        artifact is not evidence.production_artifact
        or not is_verified_scientific_model_artifact(artifact)
        or artifact.role != "production"
        or evidence.evidence_bundle.generation_authorized is not False
        or not evidence.evidence_bundle.passed
        or threshold.g3_evidence is not evidence
        or qualification.threshold_evidence is not threshold
        or meta.threshold_evidence is not threshold
        or meta.qualification_evidence is not qualification
        or qualification.generation_policy != policy
    ):
        raise IntegrityError("G5 authority production/evidence binding changed")
    for item, label in (
        (value.g3_evidence_fingerprint, "G3 evidence"),
        (value.g3_bundle_fingerprint, "G3 bundle"),
        (value.pair_receipt_fingerprint, "scientific artifact pair receipt"),
        (value.g3_source_code_fingerprint, "G3 source code"),
        (value.g3_dependency_lock_fingerprint, "G3 dependency lock"),
        (value.production_artifact_fingerprint, "production artifact"),
        (value.core_model_fingerprint, "core model"),
        (value.parameter_set_fingerprint, "parameter set"),
        (value.generation_policy_fingerprint, "generation policy"),
        (value.policy_stream_fingerprint, "generation policy stream"),
        (value.estimator_contract_fingerprint, "G5 estimator contract"),
        (
            value.adversary_specification_fingerprint,
            "G5 adversary specification",
        ),
        (value.adversary_fit_fingerprint, "G5 adversary fit"),
        (value.seed_registry_fingerprint, "G5 seed registry"),
        (value.g5_source_code_fingerprint, "G5 source code"),
        (value.g5_dependency_lock_fingerprint, "G5 dependency lock"),
        (value.threshold_evidence_fingerprint, "G5 threshold evidence"),
        (value.qualification_evidence_fingerprint, "G5 qualification evidence"),
        (value.meta_validation_evidence_fingerprint, "G5 meta evidence"),
        (value.threshold_registry_fingerprint, "G5 threshold registry"),
        (value.threshold_subject_fingerprint, "G5 threshold subject"),
        (value.qualification_results_fingerprint, "G5 qualification results"),
        (value.qualification_subject_fingerprint, "G5 qualification subject"),
        (value.meta_validation_fingerprint, "G5 meta-validation"),
        (value.meta_validation_subject_fingerprint, "G5 meta subject"),
        (
            value.g3_production_latent_task_fingerprint,
            "G3 production latent task",
        ),
        (
            value.g3_production_latent_checkpoint_fingerprint,
            "G3 production latent checkpoint",
        ),
        (
            value.g3_production_adequacy_task_fingerprint,
            "G3 production adequacy task",
        ),
        (
            value.g3_production_adequacy_checkpoint_fingerprint,
            "G3 production adequacy checkpoint",
        ),
        (value.fingerprint, "G5 authority"),
    ):
        _required_fingerprint(item, label)
    expected = _authority_identity(
        evidence=evidence,
        artifact=artifact,
        policy=policy,
        threshold_evidence=threshold,
        qualification_evidence=qualification,
        meta_validation_evidence=meta,
    )
    if (
        value.schema_id != G5_GENERATION_AUTHORITY_SCHEMA_ID
        or value.schema_version != G5_GENERATION_AUTHORITY_SCHEMA_VERSION
        or value.qualification_gate != "G5"
        or value.g3_evidence_fingerprint != evidence.reference.fingerprint
        or value.g3_bundle_fingerprint != evidence.evidence_bundle.fingerprint
        or value.pair_receipt_fingerprint != evidence.pair_receipt_fingerprint
        or value.g3_source_code_fingerprint != evidence.g3_source_code_fingerprint
        or value.g3_dependency_lock_fingerprint
        != evidence.g3_dependency_lock_fingerprint
        or value.production_artifact_fingerprint != artifact.reference.fingerprint
        or value.core_model_fingerprint != artifact.core_model_fingerprint
        or value.parameter_set_fingerprint != artifact.parameter_set_fingerprint
        or value.generation_policy_fingerprint != _fingerprint(policy.as_dict())
        or value.selected_alpha_policy != policy.alpha_policy
        or value.selected_neutral_lambda != policy.selected_neutral_lambda
        or value.policy_scientific_version != threshold.policy_scientific_version
        or value.policy_stream_fingerprint != threshold.policy_stream_fingerprint
        or value.estimator_contract_fingerprint
        != threshold.estimator_contract_fingerprint
        or value.adversary_specification_fingerprint
        != threshold.adversary_specification_fingerprint
        or value.adversary_fit_fingerprint != qualification.adversary_fit_fingerprint
        or value.seed_registry_fingerprint != threshold.seed_registry_fingerprint
        or value.development_lf_path_count != G5_DEVELOPMENT_LF_PATH_COUNT
        or value.development_hf_path_count != G5_DEVELOPMENT_HF_PATH_COUNT
        or value.qualification_lf_path_count != G5_QUALIFICATION_LF_PATH_COUNT
        or value.qualification_hf_path_count != G5_QUALIFICATION_HF_PATH_COUNT
        or value.outer_null_replicates != G5_OUTER_NULL_REPLICATES
        or value.g5_source_code_fingerprint != threshold.g5_source_code_fingerprint
        or value.g5_dependency_lock_fingerprint
        != threshold.g5_dependency_lock_fingerprint
        or value.threshold_evidence_fingerprint != threshold.fingerprint
        or value.qualification_evidence_fingerprint != qualification.fingerprint
        or value.meta_validation_evidence_fingerprint != meta.fingerprint
        or value.threshold_registry_fingerprint
        != threshold.threshold_registry_fingerprint
        or value.threshold_subject_fingerprint
        != threshold.threshold_subject_fingerprint
        or value.qualification_results_fingerprint
        != qualification.qualification_results_fingerprint
        or value.qualification_subject_fingerprint
        != qualification.qualification_subject_fingerprint
        or value.meta_validation_fingerprint != meta.meta_validation_fingerprint
        or value.meta_validation_subject_fingerprint
        != meta.meta_validation_subject_fingerprint
        or value.development_selection_fingerprint
        != threshold.development_selection_fingerprint
        or value.baseline_qualification_results_fingerprint
        != threshold.baseline_qualification_results_fingerprint
        or value.g3_production_latent_task_fingerprint
        != evidence.task_result_fingerprints[G5_PRODUCTION_LATENT_TASK_ID]
        or value.g3_production_latent_checkpoint_fingerprint
        != evidence.checkpoint_fingerprints[G5_PRODUCTION_LATENT_TASK_ID]
        or value.g3_production_adequacy_task_fingerprint
        != evidence.task_result_fingerprints[G5_PRODUCTION_ADEQUACY_TASK_ID]
        or value.g3_production_adequacy_checkpoint_fingerprint
        != evidence.checkpoint_fingerprints[G5_PRODUCTION_ADEQUACY_TASK_ID]
        or value.fingerprint != _fingerprint(expected)
    ):
        raise IntegrityError("G5 generation authority identity changed")
    return value


def is_g5_threshold_evidence(value: object) -> TypeGuard[G5ThresholdEvidence]:
    try:
        verify_g5_threshold_evidence(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g5_development_policy_selection(
    value: object,
) -> TypeGuard[G5DevelopmentPolicySelection]:
    try:
        verify_g5_development_policy_selection(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g5_qualification_evidence(
    value: object,
) -> TypeGuard[G5QualificationEvidence]:
    try:
        verify_g5_qualification_evidence(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g5_meta_validation_evidence(
    value: object,
) -> TypeGuard[G5MetaValidationEvidence]:
    try:
        verify_g5_meta_validation_evidence(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g5_qualification_generation_authority(
    value: object,
) -> TypeGuard[G5QualificationGenerationAuthority]:
    try:
        verify_g5_qualification_generation_authority(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g5_qualified_production_authority(
    value: object,
) -> TypeGuard[G5QualifiedProductionAuthority]:
    try:
        verify_g5_qualified_production_authority(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g5_noncanonical_test_fixture(value: object) -> bool:
    """Return whether ``value`` was minted through a test-only G5 capability."""

    return _G5_NONCANONICAL_TEST_OBJECTS.__contains__(id(value))


def _development_selection_identity(
    *,
    baseline_threshold_evidence: G5ThresholdEvidence,
    baseline_results: CanonicalValidationResults,
    decisions: tuple[G5DevelopmentPolicyDecision, ...],
    selected_policy: ScientificGenerationPolicy,
) -> dict[str, object]:
    return {
        "schema_id": G5_DEVELOPMENT_SELECTION_SCHEMA_ID,
        "schema_version": G5_DEVELOPMENT_SELECTION_SCHEMA_VERSION,
        "generation_authorized": False,
        "g3_evidence_fingerprint": (
            baseline_threshold_evidence.g3_evidence_fingerprint
        ),
        "baseline_threshold_evidence_fingerprint": (
            baseline_threshold_evidence.fingerprint
        ),
        "baseline_qualification_results": baseline_results.as_dict(),
        "baseline_qualification_results_fingerprint": baseline_results.fingerprint,
        "decisions": [item.identity_dict() for item in decisions],
        "selected_policy": selected_policy.as_dict(),
        "selected_policy_fingerprint": _fingerprint(selected_policy.as_dict()),
        "selected_neutral_lambda": selected_policy.selected_neutral_lambda,
        "selected_policy_scientific_version": (
            g5_policy_scientific_version(selected_policy)
        ),
    }


def _threshold_evidence_identity(
    *,
    evidence: LoadedG3EvidenceAuthority,
    g5_source_code_fingerprint: str,
    g5_dependency_lock_fingerprint: str,
    seed_registry_fingerprint: str,
    thresholds: CanonicalThresholds,
    policy: ScientificGenerationPolicy,
    estimator_contract_fingerprint: str,
    adversary_fit: G5AdversaryFitPair,
    development_selection: G5DevelopmentPolicySelection | None,
    outer_null_replicates: int,
) -> dict[str, object]:
    policy_fingerprint = _fingerprint(policy.as_dict())
    return {
        "schema_id": G5_THRESHOLD_EVIDENCE_SCHEMA_ID,
        "schema_version": G5_EVIDENCE_SCHEMA_VERSION,
        "passed": True,
        "g3_evidence_fingerprint": evidence.reference.fingerprint,
        "g3_source_code_fingerprint": evidence.g3_source_code_fingerprint,
        "g3_dependency_lock_fingerprint": evidence.g3_dependency_lock_fingerprint,
        "g5_source_code_fingerprint": g5_source_code_fingerprint,
        "g5_dependency_lock_fingerprint": g5_dependency_lock_fingerprint,
        "seed_registry_fingerprint": seed_registry_fingerprint,
        "thresholds": thresholds.as_dict(),
        "threshold_registry_fingerprint": thresholds.fingerprint,
        "generation_policy": policy.as_dict(),
        "generation_policy_fingerprint": policy_fingerprint,
        "selected_alpha_policy": policy.alpha_policy,
        "selected_neutral_lambda": policy.selected_neutral_lambda,
        "policy_scientific_version": g5_policy_scientific_version(policy),
        "policy_stream_fingerprint": g5_policy_stream_fingerprint(policy),
        "estimator_contract_fingerprint": estimator_contract_fingerprint,
        "adversary_specification_fingerprint": (G5_ADVERSARY_SPECIFICATION_FINGERPRINT),
        "adversary_fit_fingerprint": adversary_fit.fingerprint,
        "threshold_subject_fingerprint": policy_fingerprint,
        "development_selection_fingerprint": (
            None if development_selection is None else development_selection.fingerprint
        ),
        "baseline_qualification_results_fingerprint": (
            None
            if development_selection is None
            else development_selection.baseline_qualification_results_fingerprint
        ),
        "outer_null_replicates": outer_null_replicates,
    }


def _qualification_evidence_identity(
    *,
    threshold_evidence: G5ThresholdEvidence,
    policy: ScientificGenerationPolicy,
    qualification_results: CanonicalValidationResults,
    adversary_fit: G5AdversaryFitPair,
) -> dict[str, object]:
    return {
        "schema_id": G5_QUALIFICATION_EVIDENCE_SCHEMA_ID,
        "schema_version": G5_EVIDENCE_SCHEMA_VERSION,
        "passed": True,
        "threshold_evidence_fingerprint": threshold_evidence.fingerprint,
        "g3_evidence_fingerprint": threshold_evidence.g3_evidence_fingerprint,
        "generation_policy": policy.as_dict(),
        "generation_policy_fingerprint": _fingerprint(policy.as_dict()),
        "selected_alpha_policy": policy.alpha_policy,
        "selected_neutral_lambda": policy.selected_neutral_lambda,
        "seed_registry_fingerprint": threshold_evidence.seed_registry_fingerprint,
        "qualification_results": qualification_results.as_dict(),
        "qualification_results_fingerprint": qualification_results.fingerprint,
        "qualification_subject_fingerprint": qualification_results.subject_fingerprint,
        "adversary_fit_fingerprint": adversary_fit.fingerprint,
        "development_lf_path_count": G5_DEVELOPMENT_LF_PATH_COUNT,
        "development_hf_path_count": G5_DEVELOPMENT_HF_PATH_COUNT,
        "qualification_lf_path_count": G5_QUALIFICATION_LF_PATH_COUNT,
        "qualification_hf_path_count": G5_QUALIFICATION_HF_PATH_COUNT,
        "g5_source_code_fingerprint": (threshold_evidence.g5_source_code_fingerprint),
        "g5_dependency_lock_fingerprint": (
            threshold_evidence.g5_dependency_lock_fingerprint
        ),
    }


def _meta_validation_evidence_identity(
    *,
    threshold_evidence: G5ThresholdEvidence,
    qualification_evidence: G5QualificationEvidence,
    meta_validation_results: CanonicalValidationResults,
    outer_null_replicates: int,
) -> dict[str, object]:
    return {
        "schema_id": G5_META_VALIDATION_EVIDENCE_SCHEMA_ID,
        "schema_version": G5_EVIDENCE_SCHEMA_VERSION,
        "passed": True,
        "threshold_evidence_fingerprint": threshold_evidence.fingerprint,
        "qualification_evidence_fingerprint": qualification_evidence.fingerprint,
        "g3_evidence_fingerprint": threshold_evidence.g3_evidence_fingerprint,
        "generation_policy_fingerprint": (
            qualification_evidence.generation_policy_fingerprint
        ),
        "seed_registry_fingerprint": threshold_evidence.seed_registry_fingerprint,
        "meta_validation_results": meta_validation_results.as_dict(),
        "meta_validation_fingerprint": meta_validation_results.fingerprint,
        "meta_validation_subject_fingerprint": (
            meta_validation_results.subject_fingerprint
        ),
        "outer_null_replicates": outer_null_replicates,
        "g5_source_code_fingerprint": (threshold_evidence.g5_source_code_fingerprint),
        "g5_dependency_lock_fingerprint": (
            threshold_evidence.g5_dependency_lock_fingerprint
        ),
    }


def _qualification_authority_identity(
    *,
    evidence: LoadedG3EvidenceAuthority,
    artifact: LoadedScientificModelArtifact,
    candidates: ScientificCandidatePolicySet,
    threshold_evidence: G5ThresholdEvidence,
) -> dict[str, object]:
    return {
        "schema_id": G5_QUALIFICATION_AUTHORITY_SCHEMA_ID,
        "schema_version": G5_QUALIFICATION_AUTHORITY_SCHEMA_VERSION,
        "g3_evidence_fingerprint": evidence.reference.fingerprint,
        "pair_receipt_fingerprint": evidence.pair_receipt_fingerprint,
        "production_artifact_fingerprint": artifact.reference.fingerprint,
        "parameter_set_fingerprint": artifact.parameter_set_fingerprint,
        "candidate_policy_set": candidates.as_dict(),
        "candidate_policy_set_fingerprint": _fingerprint(candidates.as_dict()),
        "generation_policy": threshold_evidence.generation_policy.as_dict(),
        "generation_policy_fingerprint": (
            threshold_evidence.generation_policy_fingerprint
        ),
        "selected_alpha_policy": threshold_evidence.selected_alpha_policy,
        "selected_neutral_lambda": threshold_evidence.selected_neutral_lambda,
        "policy_scientific_version": (threshold_evidence.policy_scientific_version),
        "policy_stream_fingerprint": threshold_evidence.policy_stream_fingerprint,
        "adversary_fit_fingerprint": threshold_evidence.adversary_fit_fingerprint,
        "qualification_namespace": G5_QUALIFICATION_NAMESPACE,
        "seed_registry_fingerprint": threshold_evidence.seed_registry_fingerprint,
        "threshold_registry_fingerprint": (
            threshold_evidence.threshold_registry_fingerprint
        ),
        "threshold_evidence_fingerprint": threshold_evidence.fingerprint,
        "development_lf_path_count": G5_DEVELOPMENT_LF_PATH_COUNT,
        "development_hf_path_count": G5_DEVELOPMENT_HF_PATH_COUNT,
        "qualification_lf_path_count": G5_QUALIFICATION_LF_PATH_COUNT,
        "qualification_hf_path_count": G5_QUALIFICATION_HF_PATH_COUNT,
        "g5_source_code_fingerprint": (threshold_evidence.g5_source_code_fingerprint),
        "g5_dependency_lock_fingerprint": (
            threshold_evidence.g5_dependency_lock_fingerprint
        ),
        "generation_authorized": False,
    }


def _authority_identity(
    *,
    evidence: LoadedG3EvidenceAuthority,
    artifact: LoadedScientificModelArtifact,
    policy: ScientificGenerationPolicy,
    threshold_evidence: G5ThresholdEvidence,
    qualification_evidence: G5QualificationEvidence,
    meta_validation_evidence: G5MetaValidationEvidence,
) -> dict[str, object]:
    return {
        "schema_id": G5_GENERATION_AUTHORITY_SCHEMA_ID,
        "schema_version": G5_GENERATION_AUTHORITY_SCHEMA_VERSION,
        "qualification_gate": "G5",
        "g3_evidence_fingerprint": evidence.reference.fingerprint,
        "g3_bundle_fingerprint": evidence.evidence_bundle.fingerprint,
        "pair_receipt_fingerprint": evidence.pair_receipt_fingerprint,
        "g3_source_code_fingerprint": evidence.g3_source_code_fingerprint,
        "g3_dependency_lock_fingerprint": evidence.g3_dependency_lock_fingerprint,
        "production_artifact_fingerprint": artifact.reference.fingerprint,
        "core_model_fingerprint": artifact.core_model_fingerprint,
        "parameter_set_fingerprint": artifact.parameter_set_fingerprint,
        "generation_policy": policy.as_dict(),
        "generation_policy_fingerprint": _fingerprint(policy.as_dict()),
        "selected_alpha_policy": policy.alpha_policy,
        "selected_neutral_lambda": policy.selected_neutral_lambda,
        "policy_scientific_version": threshold_evidence.policy_scientific_version,
        "policy_stream_fingerprint": threshold_evidence.policy_stream_fingerprint,
        "estimator_contract_fingerprint": (
            threshold_evidence.estimator_contract_fingerprint
        ),
        "adversary_specification_fingerprint": (
            threshold_evidence.adversary_specification_fingerprint
        ),
        "adversary_fit_fingerprint": (qualification_evidence.adversary_fit_fingerprint),
        "seed_registry_fingerprint": threshold_evidence.seed_registry_fingerprint,
        "development_lf_path_count": G5_DEVELOPMENT_LF_PATH_COUNT,
        "development_hf_path_count": G5_DEVELOPMENT_HF_PATH_COUNT,
        "qualification_lf_path_count": G5_QUALIFICATION_LF_PATH_COUNT,
        "qualification_hf_path_count": G5_QUALIFICATION_HF_PATH_COUNT,
        "outer_null_replicates": G5_OUTER_NULL_REPLICATES,
        "g5_source_code_fingerprint": (threshold_evidence.g5_source_code_fingerprint),
        "g5_dependency_lock_fingerprint": (
            threshold_evidence.g5_dependency_lock_fingerprint
        ),
        "threshold_evidence_fingerprint": threshold_evidence.fingerprint,
        "qualification_evidence_fingerprint": qualification_evidence.fingerprint,
        "meta_validation_evidence_fingerprint": meta_validation_evidence.fingerprint,
        "threshold_registry_fingerprint": (
            threshold_evidence.threshold_registry_fingerprint
        ),
        "threshold_subject_fingerprint": (
            threshold_evidence.threshold_subject_fingerprint
        ),
        "qualification_results_fingerprint": (
            qualification_evidence.qualification_results_fingerprint
        ),
        "qualification_subject_fingerprint": (
            qualification_evidence.qualification_subject_fingerprint
        ),
        "meta_validation_fingerprint": (
            meta_validation_evidence.meta_validation_fingerprint
        ),
        "meta_validation_subject_fingerprint": (
            meta_validation_evidence.meta_validation_subject_fingerprint
        ),
        "development_selection_fingerprint": (
            threshold_evidence.development_selection_fingerprint
        ),
        "baseline_qualification_results_fingerprint": (
            threshold_evidence.baseline_qualification_results_fingerprint
        ),
        "g3_production_latent_task_fingerprint": (
            evidence.task_result_fingerprints[G5_PRODUCTION_LATENT_TASK_ID]
        ),
        "g3_production_latent_checkpoint_fingerprint": (
            evidence.checkpoint_fingerprints[G5_PRODUCTION_LATENT_TASK_ID]
        ),
        "g3_production_adequacy_task_fingerprint": (
            evidence.task_result_fingerprints[G5_PRODUCTION_ADEQUACY_TASK_ID]
        ),
        "g3_production_adequacy_checkpoint_fingerprint": (
            evidence.checkpoint_fingerprints[G5_PRODUCTION_ADEQUACY_TASK_ID]
        ),
    }


def _validated_selected_policy(value: object) -> ScientificGenerationPolicy:
    if not isinstance(value, ScientificGenerationPolicy):
        raise ModelError("G5 qualification requires a selected scientific policy")
    try:
        rebuilt = generation_policy_from_mapping(value.as_dict())
    except (IntegrityError, ModelError) as error:
        raise ModelError("G5 selected scientific policy is invalid") from error
    if rebuilt != value:
        raise ModelError("G5 selected scientific policy is noncanonical")
    return rebuilt


def _verify_g3_production_science_parents(
    evidence: LoadedG3EvidenceAuthority,
) -> None:
    decisions = {
        record.gate_id: record.passed for record in evidence.evidence_bundle.records
    }
    for task_id in (
        G5_PRODUCTION_LATENT_TASK_ID,
        G5_PRODUCTION_ADEQUACY_TASK_ID,
    ):
        if decisions.get(task_id) is not True:
            raise IntegrityError("G5 authority requires passed G3 production science")
        _required_fingerprint(
            evidence.task_result_fingerprints.get(task_id),
            f"G3 {task_id} result",
        )
        _required_fingerprint(
            evidence.checkpoint_fingerprints.get(task_id),
            f"G3 {task_id} checkpoint",
        )


def _validate_policy_is_g3_candidate(
    policy: ScientificGenerationPolicy,
    artifact: LoadedScientificModelArtifact,
) -> None:
    candidates = artifact.generation_policy
    if not isinstance(candidates, ScientificCandidatePolicySet):
        raise ModelError("G5 requires an unqualified G3 candidate-policy artifact")
    if (
        policy.generator_algorithm != candidates.generator_algorithm
        or policy.block_policy_version != candidates.block_policy_version
        or policy.monthly_block_length != candidates.monthly_block_length
        or policy.daily_block_length != candidates.daily_block_length
        or policy.alpha_policy not in candidates.candidate_policies
        or tuple(policy.neutral_lambda_grid)
        != tuple(candidates.neutral_lambda_candidates)
        or policy.initial_state != candidates.initial_state
        or policy.synchronized_return_vectors
        is not candidates.synchronized_return_vectors
        or (
            policy.alpha_policy == "kernel_mean_neutral"
            and policy.selected_neutral_lambda
            not in candidates.neutral_lambda_candidates
        )
    ):
        raise ModelError("G5 selected policy is outside the exact G3 candidate set")


def _verified_thresholds(value: object) -> CanonicalThresholds:
    if not isinstance(value, CanonicalThresholds):
        raise ModelError("G5 threshold evidence requires canonical thresholds")
    try:
        rebuilt = canonical_thresholds_from_dict(value.as_dict())
    except (IntegrityError, ModelError, TypeError, ValueError) as error:
        raise ModelError("G5 threshold record is invalid") from error
    if rebuilt != value:
        raise ModelError("G5 threshold record is noncanonical")
    return rebuilt


def _canonical_estimator_contract_fingerprint() -> str:
    # Imported lazily because qualification owns the estimator and imports this
    # authority module for its publication boundary.
    from prpg.validation.qualification import G5_ESTIMATOR_CONTRACT_FINGERPRINT

    return _required_fingerprint(
        G5_ESTIMATOR_CONTRACT_FINGERPRINT, "canonical G5 estimator contract"
    )


def _verified_results(value: object) -> CanonicalValidationResults:
    if not isinstance(value, CanonicalValidationResults):
        raise ModelError("G5 evidence requires canonical validation results")
    try:
        rebuilt = canonical_results_from_dict(value.as_dict())
    except (IntegrityError, ModelError, TypeError, ValueError) as error:
        raise ModelError("G5 validation-result record is invalid") from error
    if rebuilt != value:
        raise ModelError("G5 validation-result record is noncanonical")
    return rebuilt


def _verified_adversary_fit(value: object) -> G5AdversaryFitPair:
    if type(value) is not G5AdversaryFitPair:
        raise ModelError("G5 qualification requires a typed adversary fit pair")
    monthly = _verified_adversary_model(value.monthly, "monthly")
    daily = _verified_adversary_model(value.daily, "daily")
    identity = {
        "specification_fingerprint": G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
        "monthly_model_fingerprint": monthly.fingerprint,
        "daily_model_fingerprint": daily.fingerprint,
    }
    if (
        value.specification_fingerprint != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        or value.fingerprint != _fingerprint(identity)
    ):
        raise ModelError("G5 adversary fit-pair identity is invalid")
    return value


def _verified_adversary_model(
    value: object, frequency: Literal["monthly", "daily"]
) -> G5AdversaryModel:
    if type(value) is not G5AdversaryModel or value.frequency != frequency:
        raise ModelError("G5 adversary model frequency/type is invalid")
    arrays = {
        "linear_center": (value.linear_center, (3,)),
        "linear_scale": (value.linear_scale, (3,)),
        "linear_coefficients": (value.linear_coefficients, (4, 3)),
        "quadratic_center": (value.quadratic_center, (3,)),
        "quadratic_scale": (value.quadratic_scale, (3,)),
        "quadratic_coefficients": (value.quadratic_coefficients, (4, 3)),
        "outcome_scale": (value.outcome_scale, (3,)),
    }
    digests: dict[str, str] = {}
    for name, (array, shape) in arrays.items():
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.dtype("<f8")
            or array.shape != shape
            or not array.flags.c_contiguous
            or array.flags.writeable
            or not bool(np.isfinite(array).all())
        ):
            raise ModelError("G5 adversary coefficient array is invalid")
        digests[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    expected_horizon = G5_MEMORIZATION_HORIZON[frequency]
    if value.memorization_horizon != expected_horizon or not isinstance(
        value.memorized_outcomes, Mapping
    ):
        raise ModelError("G5 adversary memorization contract is invalid")
    memo_digest = hashlib.sha256()
    for key in sorted(value.memorized_outcomes):
        outcome = value.memorized_outcomes[key]
        if (
            not isinstance(key, bytes)
            or len(key) != 16
            or not isinstance(outcome, tuple)
            or len(outcome) != 3
            or any(
                not isinstance(item, float) or not np.isfinite(item) for item in outcome
            )
        ):
            raise ModelError("G5 adversary memorized outcome is invalid")
        memo_digest.update(key)
        memo_digest.update(np.asarray(outcome, dtype="<f8").tobytes())
    identity = {
        "schema_id": G5_ADVERSARY_SCHEMA_ID,
        "frequency": frequency,
        "ridge_penalty": G5_RIDGE_PENALTY,
        "linear_center_sha256": digests["linear_center"],
        "linear_scale_sha256": digests["linear_scale"],
        "linear_coefficients_sha256": digests["linear_coefficients"],
        "quadratic_center_sha256": digests["quadratic_center"],
        "quadratic_scale_sha256": digests["quadratic_scale"],
        "quadratic_coefficients_sha256": digests["quadratic_coefficients"],
        "outcome_scale_sha256": digests["outcome_scale"],
        "memorization_horizon": expected_horizon,
        "memorization_digest": memo_digest.hexdigest(),
    }
    if value.fingerprint != _fingerprint(identity):
        raise ModelError("G5 adversary model fingerprint is invalid")
    return value


def _set_fields(value: object, fields: tuple[tuple[str, object], ...]) -> None:
    for name, item in fields:
        object.__setattr__(value, name, item)
    object.__setattr__(value, "_owner_id", id(value))


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ModelError(f"{label} fingerprint is invalid")
    return value


def _fingerprint(value: object) -> str:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(content).hexdigest()
