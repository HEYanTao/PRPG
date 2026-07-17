"""Small durable production authority for the delivery-first lean G5 gate.

The retired G5 authority graph remains available for legacy runs.  This module
is the production boundary for the lean contract: one canonical, passing
``LeanG5AssessmentReport`` and its exact qualification permit are reduced to a
compact content-addressed JSON identity.  Loading that identity requires an
already reverified G3 evidence object and never rebuilds the old threshold or
meta-validation graph.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeGuard

from prpg.errors import IntegrityError, ModelError, ValidationError
from prpg.model.g3_evidence_store import (
    LoadedG3EvidenceAuthority,
    verify_loaded_g3_evidence,
)
from prpg.model.g5_generation_authority import g5_policy_scientific_version
from prpg.model.lean_g5_qualification import (
    LEAN_G5_QUALIFICATION_CONTRACT_ID,
    LeanG5QualificationPermit,
    _validated_candidate_policy,
    verify_lean_g5_qualification_permit,
)
from prpg.model.scientific_artifact import LoadedScientificModelArtifact
from prpg.model.scientific_policy import (
    ScientificGenerationPolicy,
    generation_policy_from_mapping,
)
from prpg.validation.lean_g5_assessment import (
    LEAN_G5_ASSESSMENT_SCHEMA_ID,
    LEAN_G5_GROUP_NAMES,
    LeanG5AssessmentReport,
    _jsonable,
)
from prpg.validation.lean_g5_statistics import (
    LEAN_G5_PILOT_GEOMETRY,
    LEAN_G5_PRODUCTION_GEOMETRY,
    LeanG5Geometry,
)

LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID: Final = "prpg-lean-g5-production-authority-v1"
LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_VERSION: Final = 1
LEAN_G5_PRODUCTION_AUTHORITY_DIRECTORY: Final = "lean-g5-production-authorities"
LEAN_G5_PRODUCTION_AUTHORITY_MANIFEST: Final = "authority-manifest.json"
LEAN_G5_DOWNSTREAM_IDENTITY_SCHEMA_ID: Final = "prpg-lean-g5-downstream-v1"
LEAN_G5_VERIFICATION_RECEIPT_SCHEMA_ID: Final = "prpg-lean-g5-verification-receipt-v1"
LEAN_G5_REQUIRED_DEFECTS: Final = (
    "linear_alpha",
    "quadratic_interaction_alpha",
    "covariance",
    "correlation",
    "desynchronization",
    "clipping",
    "transition_diagonal",
    "copied_path",
    "repeated_subsequence",
)

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_AUTHORITY_SEAL = object()
_VERIFICATION_RECEIPT_SEAL = object()
_LIVE_AUTHORITIES: dict[int, LeanG5ProductionAuthority] = {}
_LIVE_VERIFICATION_RECEIPTS: dict[int, LeanG5VerificationReceipt] = {}


@dataclass(frozen=True, slots=True, init=False)
class LeanG5VerificationReceipt:
    """Compact proof that the bounded pre-canonical checks all passed."""

    schema_id: str
    pilot_monthly_paths: int
    pilot_daily_paths: int
    pilot_resamples: int
    pilot_assessment_fingerprint: str
    generation_policy_fingerprint: str
    one_worker_fingerprint: str
    nine_worker_fingerprint: str
    worker_parity_passed: Literal[True]
    resource_projection_fingerprint: str
    resource_projection_passed: Literal[True]
    pristine_assessment_fingerprint: str
    pristine_passed: Literal[True]
    defect_results: tuple[tuple[str, Literal[True]], ...]
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "LeanG5VerificationReceipt is created only by its public builder"
        )


def build_lean_g5_verification_receipt(
    *,
    pilot_assessment: LeanG5AssessmentReport,
    one_worker_fingerprint: str,
    nine_worker_fingerprint: str,
    worker_parity_passed: bool,
    resource_projection_fingerprint: str,
    resource_projection_passed: bool,
    pristine_assessment_fingerprint: str,
    pristine_passed: bool,
    defect_results: tuple[tuple[str, bool], ...],
) -> LeanG5VerificationReceipt:
    """Bind the one pilot/parity/resource/targeted-defect verification run."""

    pilot = _validated_passing_assessment(
        pilot_assessment, expected_geometry=LEAN_G5_PILOT_GEOMETRY
    )
    expected_defects = tuple((name, True) for name in LEAN_G5_REQUIRED_DEFECTS)
    if (
        worker_parity_passed is not True
        or resource_projection_passed is not True
        or pristine_passed is not True
        or defect_results != expected_defects
    ):
        raise ValidationError("lean G5 bounded verification did not fully pass")
    one_worker = _required_fingerprint(one_worker_fingerprint, "one-worker pilot")
    nine_worker = _required_fingerprint(nine_worker_fingerprint, "nine-worker pilot")
    if one_worker != nine_worker:
        raise ValidationError("lean G5 one-worker/nine-worker fingerprints differ")
    identity = _verification_receipt_identity(
        pilot_assessment_fingerprint=pilot.fingerprint,
        generation_policy_fingerprint=pilot.generation_policy_fingerprint,
        one_worker_fingerprint=one_worker,
        nine_worker_fingerprint=nine_worker,
        resource_projection_fingerprint=_required_fingerprint(
            resource_projection_fingerprint, "resource projection"
        ),
        pristine_assessment_fingerprint=_required_fingerprint(
            pristine_assessment_fingerprint, "pristine assessment"
        ),
        defect_results=expected_defects,
    )
    result = object.__new__(LeanG5VerificationReceipt)
    for name, item in (
        ("schema_id", LEAN_G5_VERIFICATION_RECEIPT_SCHEMA_ID),
        ("pilot_monthly_paths", 100),
        ("pilot_daily_paths", 40),
        ("pilot_resamples", 1_000),
        ("pilot_assessment_fingerprint", identity["pilot_assessment_fingerprint"]),
        (
            "generation_policy_fingerprint",
            identity["generation_policy_fingerprint"],
        ),
        ("one_worker_fingerprint", one_worker),
        ("nine_worker_fingerprint", nine_worker),
        ("worker_parity_passed", True),
        (
            "resource_projection_fingerprint",
            identity["resource_projection_fingerprint"],
        ),
        ("resource_projection_passed", True),
        (
            "pristine_assessment_fingerprint",
            identity["pristine_assessment_fingerprint"],
        ),
        ("pristine_passed", True),
        ("defect_results", expected_defects),
        ("fingerprint", _fingerprint(identity)),
        ("_seal", _VERIFICATION_RECEIPT_SEAL),
    ):
        object.__setattr__(result, name, item)
    object.__setattr__(result, "_owner_id", id(result))
    _LIVE_VERIFICATION_RECEIPTS[id(result)] = result
    return verify_lean_g5_verification_receipt(result)


def verify_lean_g5_verification_receipt(
    value: object,
) -> LeanG5VerificationReceipt:
    if (
        type(value) is not LeanG5VerificationReceipt
        or value._seal is not _VERIFICATION_RECEIPT_SEAL
        or value._owner_id != id(value)
        or _LIVE_VERIFICATION_RECEIPTS.get(id(value)) is not value
    ):
        raise ModelError("lean G5 verification receipt lacks builder provenance")
    expected = tuple((name, True) for name in LEAN_G5_REQUIRED_DEFECTS)
    identity = _verification_receipt_identity(
        pilot_assessment_fingerprint=value.pilot_assessment_fingerprint,
        generation_policy_fingerprint=value.generation_policy_fingerprint,
        one_worker_fingerprint=value.one_worker_fingerprint,
        nine_worker_fingerprint=value.nine_worker_fingerprint,
        resource_projection_fingerprint=value.resource_projection_fingerprint,
        pristine_assessment_fingerprint=value.pristine_assessment_fingerprint,
        defect_results=value.defect_results,
    )
    if (
        value.schema_id != LEAN_G5_VERIFICATION_RECEIPT_SCHEMA_ID
        or (value.pilot_monthly_paths, value.pilot_daily_paths, value.pilot_resamples)
        != (100, 40, 1_000)
        or value.one_worker_fingerprint != value.nine_worker_fingerprint
        or _required_fingerprint(
            value.generation_policy_fingerprint, "pilot generation policy"
        )
        != value.generation_policy_fingerprint
        or value.worker_parity_passed is not True
        or value.resource_projection_passed is not True
        or value.pristine_passed is not True
        or value.defect_results != expected
        or value.fingerprint != _fingerprint(identity)
    ):
        raise IntegrityError("lean G5 verification receipt identity changed")
    return value


@dataclass(frozen=True, slots=True, init=False)
class LeanG5ProductionAuthority:
    """Process-original authority for one exact lean-G5 pass."""

    schema_id: str
    schema_version: int
    contract_id: str
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    production_artifact_fingerprint: str
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    policy_scientific_version: int
    assessment_schema_id: str
    assessment_fingerprint: str
    qualification_permit_fingerprint: str
    verification_receipt_fingerprint: str
    downstream_source_code_fingerprint: str
    downstream_dependency_lock_fingerprint: str
    downstream_identity_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "LeanG5ProductionAuthority is created only by its mint or store loader"
        )

    @property
    def generation_authorized(self) -> Literal[True]:
        verify_lean_g5_production_authority(self)
        return True

    @property
    def production_artifact(self) -> LoadedScientificModelArtifact:
        return self.g3_evidence.production_artifact

    def identity_dict(self) -> dict[str, object]:
        return _authority_identity(
            g3_evidence_fingerprint=self.g3_evidence_fingerprint,
            production_artifact_fingerprint=self.production_artifact_fingerprint,
            generation_policy=self.generation_policy,
            policy_scientific_version=self.policy_scientific_version,
            assessment_fingerprint=self.assessment_fingerprint,
            qualification_permit_fingerprint=self.qualification_permit_fingerprint,
            verification_receipt_fingerprint=(self.verification_receipt_fingerprint),
            downstream_source_code_fingerprint=(
                self.downstream_source_code_fingerprint
            ),
            downstream_dependency_lock_fingerprint=(
                self.downstream_dependency_lock_fingerprint
            ),
        )


def mint_lean_g5_production_authority(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    generation_policy: ScientificGenerationPolicy,
    assessment: LeanG5AssessmentReport,
    qualification_permit: LeanG5QualificationPermit,
    verification_receipt: LeanG5VerificationReceipt,
    downstream_source_code_fingerprint: str,
    downstream_dependency_lock_fingerprint: str,
) -> LeanG5ProductionAuthority:
    """Mint only from the canonical passing lean-G5 report and exact permit."""

    evidence = verify_loaded_g3_evidence(g3_evidence)
    permit = verify_lean_g5_qualification_permit(qualification_permit)
    if permit.g3_evidence is not evidence:
        raise ModelError("lean G5 authority permit belongs to different G3 evidence")
    policy = _validated_candidate_policy(generation_policy, evidence)
    if policy != permit.generation_policy:
        raise ModelError("lean G5 authority policy differs from qualification permit")
    report = _validated_passing_assessment(assessment)
    verification = verify_lean_g5_verification_receipt(verification_receipt)
    policy_fingerprint = _fingerprint(policy.as_dict())
    if (
        report.generation_policy_fingerprint != policy_fingerprint
        or permit.generation_policy_fingerprint != policy_fingerprint
        or verification.generation_policy_fingerprint != policy_fingerprint
    ):
        raise ValidationError(
            "lean G5 report/permit/verification policy fingerprint differs"
        )
    if report.control_mean_block_lengths != (
        ("monthly", float(policy.monthly_block_length)),
        ("daily", float(policy.daily_block_length)),
    ):
        raise ValidationError(
            "lean G5 historical-control block lengths differ from selected policy"
        )
    version = g5_policy_scientific_version(policy)
    if permit.policy_scientific_version != version:
        raise ModelError("lean G5 qualification stream version changed")
    source = _required_fingerprint(
        downstream_source_code_fingerprint, "downstream source code"
    )
    dependency = _required_fingerprint(
        downstream_dependency_lock_fingerprint, "downstream dependency lock"
    )
    identity = _authority_identity(
        g3_evidence_fingerprint=evidence.reference.fingerprint,
        production_artifact_fingerprint=(
            evidence.production_artifact.reference.fingerprint
        ),
        generation_policy=policy,
        policy_scientific_version=version,
        assessment_fingerprint=report.fingerprint,
        qualification_permit_fingerprint=permit.fingerprint,
        verification_receipt_fingerprint=verification.fingerprint,
        downstream_source_code_fingerprint=source,
        downstream_dependency_lock_fingerprint=dependency,
    )
    return _new_authority(identity, evidence)


def verify_lean_g5_production_authority(
    value: object,
) -> LeanG5ProductionAuthority:
    """Reverify one original minted or store-loaded lean-G5 authority."""

    if (
        type(value) is not LeanG5ProductionAuthority
        or value._seal is not _AUTHORITY_SEAL
        or value._owner_id != id(value)
        or _LIVE_AUTHORITIES.get(id(value)) is not value
    ):
        raise ModelError("lean G5 production authority lacks loader/mint provenance")
    evidence = verify_loaded_g3_evidence(value.g3_evidence)
    policy = _validated_candidate_policy(value.generation_policy, evidence)
    identity = _authority_identity(
        g3_evidence_fingerprint=evidence.reference.fingerprint,
        production_artifact_fingerprint=(
            evidence.production_artifact.reference.fingerprint
        ),
        generation_policy=policy,
        policy_scientific_version=g5_policy_scientific_version(policy),
        assessment_fingerprint=value.assessment_fingerprint,
        qualification_permit_fingerprint=value.qualification_permit_fingerprint,
        verification_receipt_fingerprint=value.verification_receipt_fingerprint,
        downstream_source_code_fingerprint=(value.downstream_source_code_fingerprint),
        downstream_dependency_lock_fingerprint=(
            value.downstream_dependency_lock_fingerprint
        ),
    )
    if (
        value.schema_id != LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID
        or value.schema_version != LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_VERSION
        or value.contract_id != LEAN_G5_QUALIFICATION_CONTRACT_ID
        or value.g3_evidence_fingerprint != evidence.reference.fingerprint
        or value.production_artifact_fingerprint
        != evidence.production_artifact.reference.fingerprint
        or value.generation_policy_fingerprint != _fingerprint(policy.as_dict())
        or value.policy_scientific_version != g5_policy_scientific_version(policy)
        or value.assessment_schema_id != LEAN_G5_ASSESSMENT_SCHEMA_ID
        or value.downstream_identity_fingerprint
        != _downstream_identity_fingerprint(
            value.downstream_source_code_fingerprint,
            value.downstream_dependency_lock_fingerprint,
        )
        or value.fingerprint != _fingerprint(identity)
    ):
        raise IntegrityError("lean G5 production authority identity changed")
    return value


def is_lean_g5_production_authority(
    value: object,
) -> TypeGuard[LeanG5ProductionAuthority]:
    try:
        verify_lean_g5_production_authority(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


class LeanG5ProductionAuthorityStore:
    """Content-addressed canonical-JSON store for lean production authority."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.root = (
            Path(artifact_root).expanduser().resolve()
            / LEAN_G5_PRODUCTION_AUTHORITY_DIRECTORY
            / "sha256"
        )

    def contains(self, fingerprint: str) -> bool:
        return self._manifest_path(fingerprint).is_file()

    def store(self, authority: LeanG5ProductionAuthority) -> Path:
        checked = verify_lean_g5_production_authority(authority)
        manifest = {
            "schema_version": LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_VERSION,
            "authority_fingerprint": checked.fingerprint,
            "identity": checked.identity_dict(),
        }
        payload = _canonical_bytes(manifest)
        path = self._manifest_path(checked.fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise IntegrityError("stored lean G5 authority conflicts with identity")
            return path
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=".authority-", delete=False
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return path

    def load(
        self,
        fingerprint: str,
        *,
        g3_evidence: LoadedG3EvidenceAuthority,
    ) -> LeanG5ProductionAuthority:
        expected = _required_fingerprint(fingerprint, "lean G5 authority")
        evidence = verify_loaded_g3_evidence(g3_evidence)
        path = self._manifest_path(expected)
        if not path.is_file() or path.is_symlink():
            raise IntegrityError("lean G5 authority manifest is missing")
        payload = path.read_bytes()
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "lean G5 authority manifest is invalid JSON"
            ) from error
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "schema_version",
            "authority_fingerprint",
            "identity",
        }:
            raise IntegrityError("lean G5 authority manifest keys are invalid")
        if payload != _canonical_bytes(manifest):
            raise IntegrityError("lean G5 authority manifest is not canonical JSON")
        identity = manifest["identity"]
        if (
            manifest["schema_version"] != LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_VERSION
            or manifest["authority_fingerprint"] != expected
            or not isinstance(identity, Mapping)
            or _fingerprint(identity) != expected
        ):
            raise IntegrityError("lean G5 authority manifest fingerprint is invalid")
        return _new_authority(_parsed_identity(identity), evidence)

    def _manifest_path(self, fingerprint: str) -> Path:
        checked = _required_fingerprint(fingerprint, "lean G5 authority")
        return self.root / checked / LEAN_G5_PRODUCTION_AUTHORITY_MANIFEST


def _validated_passing_assessment(
    value: object,
    *,
    expected_geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> LeanG5AssessmentReport:
    if type(value) is not LeanG5AssessmentReport:
        raise ValidationError("lean G5 production decision requires its typed report")
    report = value
    identity = _assessment_identity(report)
    comparisons = report.grouped_comparisons
    holm = report.holm
    for fingerprint in (
        report.generation_policy_fingerprint,
        report.development_bundle_fingerprint,
        report.qualification_bundle_fingerprint,
        report.resample_indices_fingerprint,
        *(item for _, item in report.control_bank_fingerprints),
        *(item for _, item in report.model_fingerprints),
    ):
        _required_fingerprint(fingerprint, "lean G5 assessment component")
    if (
        report.schema_id != LEAN_G5_ASSESSMENT_SCHEMA_ID
        or report.geometry != expected_geometry
        or report.generation_authorized is not False
        or _required_fingerprint(report.fingerprint, "lean G5 assessment")
        != _fingerprint(identity)
    ):
        raise IntegrityError("lean G5 assessment identity is invalid")
    if (
        report.decision != "pass"
        or tuple(name for name, _ in report.control_bank_fingerprints)
        != ("A", "B", "C")
        or tuple(name for name, _ in report.model_fingerprints)
        != (
            "strategy.monthly",
            "strategy.daily",
            "inverse_volatility.monthly",
            "inverse_volatility.daily",
        )
        or report.alpha.passed is not True
        or report.alpha.geometry != report.geometry
        or report.alpha.equivalence.geometry != report.geometry
        or report.alpha.leakage_gap.geometry != report.geometry
        or report.alpha.inverse_volatility.geometry != report.geometry
        or report.alpha.equivalence.passed is not True
        or report.alpha.leakage_gap.passed is not True
        or report.alpha.inverse_volatility.passed is not True
        or tuple(item.family for item in comparisons) != LEAN_G5_GROUP_NAMES
        or any(
            item.status == "ambiguous" or item.p_value is None for item in comparisons
        )
        or holm is None
        or holm.familywise_alpha != 0.05
        or tuple(item.name for item in holm.decisions) != LEAN_G5_GROUP_NAMES
        or holm.any_rejected is not False
        or any(item.rejected for item in holm.decisions)
        or report.diversity.complete_duplicates_passed is not True
        or report.diversity.repeated_subsequence_status != "pass"
    ):
        raise ValidationError("lean G5 assessment did not pass every release control")
    return report


def _assessment_identity(value: LeanG5AssessmentReport) -> dict[str, object]:
    return {
        "schema_id": value.schema_id,
        "geometry": _jsonable(value.geometry),
        "generation_policy_fingerprint": value.generation_policy_fingerprint,
        "development_bundle_fingerprint": value.development_bundle_fingerprint,
        "qualification_bundle_fingerprint": value.qualification_bundle_fingerprint,
        "control_bank_fingerprints": value.control_bank_fingerprints,
        "control_mean_block_lengths": value.control_mean_block_lengths,
        "model_fingerprints": value.model_fingerprints,
        "resample_indices_fingerprint": value.resample_indices_fingerprint,
        "alpha": _jsonable(value.alpha),
        "grouped_comparisons": _jsonable(value.grouped_comparisons),
        "holm": _jsonable(value.holm),
        "states": _jsonable(value.states),
        "diversity": _jsonable(value.diversity),
        "decision": value.decision,
        "reasons": value.reasons,
        "generation_authorized": value.generation_authorized,
    }


def _verification_receipt_identity(
    *,
    pilot_assessment_fingerprint: str,
    generation_policy_fingerprint: str,
    one_worker_fingerprint: str,
    nine_worker_fingerprint: str,
    resource_projection_fingerprint: str,
    pristine_assessment_fingerprint: str,
    defect_results: tuple[tuple[str, bool], ...],
) -> dict[str, object]:
    return {
        "schema_id": LEAN_G5_VERIFICATION_RECEIPT_SCHEMA_ID,
        "pilot_geometry": {
            "monthly_paths": 100,
            "daily_paths": 40,
            "resamples": 1_000,
        },
        "pilot_assessment_fingerprint": _required_fingerprint(
            pilot_assessment_fingerprint, "pilot assessment"
        ),
        "generation_policy_fingerprint": _required_fingerprint(
            generation_policy_fingerprint, "pilot generation policy"
        ),
        "one_worker_fingerprint": _required_fingerprint(
            one_worker_fingerprint, "one-worker pilot"
        ),
        "nine_worker_fingerprint": _required_fingerprint(
            nine_worker_fingerprint, "nine-worker pilot"
        ),
        "worker_parity_passed": True,
        "resource_projection_fingerprint": _required_fingerprint(
            resource_projection_fingerprint, "resource projection"
        ),
        "resource_projection_passed": True,
        "pristine_assessment_fingerprint": _required_fingerprint(
            pristine_assessment_fingerprint, "pristine assessment"
        ),
        "pristine_passed": True,
        "defect_results": defect_results,
    }


def _authority_identity(
    *,
    g3_evidence_fingerprint: str,
    production_artifact_fingerprint: str,
    generation_policy: ScientificGenerationPolicy,
    policy_scientific_version: int,
    assessment_fingerprint: str,
    qualification_permit_fingerprint: str,
    verification_receipt_fingerprint: str,
    downstream_source_code_fingerprint: str,
    downstream_dependency_lock_fingerprint: str,
) -> dict[str, object]:
    source = _required_fingerprint(
        downstream_source_code_fingerprint, "downstream source code"
    )
    dependency = _required_fingerprint(
        downstream_dependency_lock_fingerprint, "downstream dependency lock"
    )
    return {
        "schema_id": LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID,
        "schema_version": LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_VERSION,
        "generation_authorized": True,
        "contract_id": LEAN_G5_QUALIFICATION_CONTRACT_ID,
        "g3_evidence_fingerprint": _required_fingerprint(
            g3_evidence_fingerprint, "G3 evidence"
        ),
        "production_artifact_fingerprint": _required_fingerprint(
            production_artifact_fingerprint, "production artifact"
        ),
        "generation_policy": generation_policy.as_dict(),
        "generation_policy_fingerprint": _fingerprint(generation_policy.as_dict()),
        "policy_scientific_version": policy_scientific_version,
        "assessment": {
            "schema_id": LEAN_G5_ASSESSMENT_SCHEMA_ID,
            "fingerprint": _required_fingerprint(
                assessment_fingerprint, "lean G5 assessment"
            ),
            "geometry": {"monthly_paths": 500, "daily_paths": 200, "resamples": 10_000},
            "decision": "pass",
            "alpha_controls_passed": True,
            "inverse_volatility_passed": True,
            "holm_complete": True,
            "holm_any_rejected": False,
            "complete_duplicates_passed": True,
        },
        "qualification_permit_fingerprint": _required_fingerprint(
            qualification_permit_fingerprint, "lean G5 qualification permit"
        ),
        "verification_receipt_fingerprint": _required_fingerprint(
            verification_receipt_fingerprint, "lean G5 verification receipt"
        ),
        "downstream": {
            "schema_id": LEAN_G5_DOWNSTREAM_IDENTITY_SCHEMA_ID,
            "source_code_fingerprint": source,
            "dependency_lock_fingerprint": dependency,
            "fingerprint": _downstream_identity_fingerprint(source, dependency),
        },
    }


def _parsed_identity(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema_id",
        "schema_version",
        "generation_authorized",
        "contract_id",
        "g3_evidence_fingerprint",
        "production_artifact_fingerprint",
        "generation_policy",
        "generation_policy_fingerprint",
        "policy_scientific_version",
        "assessment",
        "qualification_permit_fingerprint",
        "verification_receipt_fingerprint",
        "downstream",
    }
    if set(value) != expected:
        raise IntegrityError("lean G5 authority identity keys are invalid")
    policy = generation_policy_from_mapping(value["generation_policy"])
    assessment = value["assessment"]
    downstream = value["downstream"]
    if not isinstance(assessment, Mapping) or not isinstance(downstream, Mapping):
        raise IntegrityError("lean G5 authority nested identity is invalid")
    rebuilt = _authority_identity(
        g3_evidence_fingerprint=value["g3_evidence_fingerprint"],  # type: ignore[arg-type]
        production_artifact_fingerprint=value["production_artifact_fingerprint"],  # type: ignore[arg-type]
        generation_policy=policy,
        policy_scientific_version=value["policy_scientific_version"],  # type: ignore[arg-type]
        assessment_fingerprint=assessment.get("fingerprint"),  # type: ignore[arg-type]
        qualification_permit_fingerprint=value["qualification_permit_fingerprint"],  # type: ignore[arg-type]
        verification_receipt_fingerprint=value["verification_receipt_fingerprint"],  # type: ignore[arg-type]
        downstream_source_code_fingerprint=downstream.get("source_code_fingerprint"),  # type: ignore[arg-type]
        downstream_dependency_lock_fingerprint=downstream.get(
            "dependency_lock_fingerprint"
        ),  # type: ignore[arg-type]
    )
    if dict(value) != rebuilt:
        raise IntegrityError("lean G5 authority identity is noncanonical")
    return rebuilt


def _new_authority(
    identity: Mapping[str, object], evidence: LoadedG3EvidenceAuthority
) -> LeanG5ProductionAuthority:
    checked_evidence = verify_loaded_g3_evidence(evidence)
    if identity["g3_evidence_fingerprint"] != checked_evidence.reference.fingerprint:
        raise IntegrityError("lean G5 authority belongs to different G3 evidence")
    if (
        identity["production_artifact_fingerprint"]
        != checked_evidence.production_artifact.reference.fingerprint
    ):
        raise IntegrityError("lean G5 authority production artifact differs from G3")
    policy = generation_policy_from_mapping(identity["generation_policy"])
    policy = _validated_candidate_policy(policy, checked_evidence)
    assessment = identity["assessment"]
    downstream = identity["downstream"]
    assert isinstance(assessment, Mapping) and isinstance(downstream, Mapping)
    result = object.__new__(LeanG5ProductionAuthority)
    fields = (
        ("schema_id", identity["schema_id"]),
        ("schema_version", identity["schema_version"]),
        ("contract_id", identity["contract_id"]),
        ("g3_evidence", checked_evidence),
        ("g3_evidence_fingerprint", identity["g3_evidence_fingerprint"]),
        (
            "production_artifact_fingerprint",
            identity["production_artifact_fingerprint"],
        ),
        ("generation_policy", policy),
        ("generation_policy_fingerprint", identity["generation_policy_fingerprint"]),
        ("policy_scientific_version", identity["policy_scientific_version"]),
        ("assessment_schema_id", assessment["schema_id"]),
        ("assessment_fingerprint", assessment["fingerprint"]),
        (
            "qualification_permit_fingerprint",
            identity["qualification_permit_fingerprint"],
        ),
        (
            "verification_receipt_fingerprint",
            identity["verification_receipt_fingerprint"],
        ),
        (
            "downstream_source_code_fingerprint",
            downstream["source_code_fingerprint"],
        ),
        (
            "downstream_dependency_lock_fingerprint",
            downstream["dependency_lock_fingerprint"],
        ),
        ("downstream_identity_fingerprint", downstream["fingerprint"]),
        ("fingerprint", _fingerprint(identity)),
        ("_seal", _AUTHORITY_SEAL),
    )
    for name, item in fields:
        object.__setattr__(result, name, item)
    object.__setattr__(result, "_owner_id", id(result))
    _LIVE_AUTHORITIES[id(result)] = result
    return verify_lean_g5_production_authority(result)


def _downstream_identity_fingerprint(source: str, dependency: str) -> str:
    return _fingerprint(
        {
            "schema_id": LEAN_G5_DOWNSTREAM_IDENTITY_SCHEMA_ID,
            "source_code_fingerprint": _required_fingerprint(
                source, "downstream source code"
            ),
            "dependency_lock_fingerprint": _required_fingerprint(
                dependency, "downstream dependency lock"
            ),
        }
    )


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} fingerprint is invalid")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise IntegrityError(
            "lean G5 authority identity is not canonical JSON"
        ) from error
    return (payload + "\n").encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()
