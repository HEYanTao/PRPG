"""Minimal bounded qualification permit for the lean G5 contract.

The permit binds a verified completed G3 result, one registered candidate
policy, and the frozen development decision that selected that policy.  It
authorizes only the lean untouched qualification sample; no production API
accepts this type.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, TypeGuard

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_evidence_store import (
    LoadedG3EvidenceAuthority,
    verify_loaded_g3_evidence,
)
from prpg.model.g5_generation_authority import g5_policy_scientific_version
from prpg.model.scientific_policy import (
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
    generation_policy_from_mapping,
)

LEAN_G5_QUALIFICATION_CONTRACT_ID: Final = "g5-lean-six-family-three-bank-v1"
LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_ID: Final = "prpg-lean-g5-qualification-permit-v1"
LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_VERSION: Final = 1

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_PERMIT_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class LeanG5QualificationPermit:
    """Process-original permit for one untouched bounded G5 sample."""

    production_generation_authorized: ClassVar[Literal[False]] = False

    schema_id: str
    schema_version: int
    qualification_generation_permitted: Literal[True]
    contract_id: str
    g3_evidence: LoadedG3EvidenceAuthority
    g3_evidence_fingerprint: str
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    policy_scientific_version: int
    development_decision_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "LeanG5QualificationPermit is created only by its public builder"
        )


_MINTED_PERMITS: dict[int, LeanG5QualificationPermit] = {}


def build_lean_g5_qualification_permit(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    generation_policy: ScientificGenerationPolicy,
    development_decision_fingerprint: str,
) -> LeanG5QualificationPermit:
    """Bind the mandatory baseline qualification to exact G3 evidence.

    Neutral-policy qualification remains closed until a typed development
    selector proves the owner-approved baseline-first/smallest-lambda rule.
    A caller-supplied digest alone is intentionally insufficient for that
    policy choice.
    """

    evidence = verify_loaded_g3_evidence(g3_evidence)
    policy = _validated_candidate_policy(generation_policy, evidence)
    decision = _required_fingerprint(
        development_decision_fingerprint, "lean G5 development decision"
    )
    policy_fingerprint = _fingerprint(policy.as_dict())
    version = g5_policy_scientific_version(policy)
    identity = _permit_identity(
        g3_evidence_fingerprint=evidence.reference.fingerprint,
        generation_policy=policy,
        policy_scientific_version=version,
        development_decision_fingerprint=decision,
    )
    result = object.__new__(LeanG5QualificationPermit)
    for name, value in (
        ("schema_id", LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_ID),
        ("schema_version", LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_VERSION),
        ("qualification_generation_permitted", True),
        ("contract_id", LEAN_G5_QUALIFICATION_CONTRACT_ID),
        ("g3_evidence", evidence),
        ("g3_evidence_fingerprint", evidence.reference.fingerprint),
        ("generation_policy", policy),
        ("generation_policy_fingerprint", policy_fingerprint),
        ("policy_scientific_version", version),
        ("development_decision_fingerprint", decision),
        ("fingerprint", _fingerprint(identity)),
        ("_seal", _PERMIT_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _MINTED_PERMITS[id(result)] = result
    return verify_lean_g5_qualification_permit(result)


def verify_lean_g5_qualification_permit(
    value: object,
) -> LeanG5QualificationPermit:
    """Recheck one original live lean-G5 qualification permit."""

    if (
        type(value) is not LeanG5QualificationPermit
        or value._seal is not _PERMIT_SEAL
        or value._owner_id != id(value)
        or _MINTED_PERMITS.get(id(value)) is not value
    ):
        raise ModelError("lean G5 qualification permit lacks builder provenance")
    evidence = verify_loaded_g3_evidence(value.g3_evidence)
    policy = _validated_candidate_policy(value.generation_policy, evidence)
    decision = _required_fingerprint(
        value.development_decision_fingerprint, "lean G5 development decision"
    )
    policy_fingerprint = _fingerprint(policy.as_dict())
    version = g5_policy_scientific_version(policy)
    identity = _permit_identity(
        g3_evidence_fingerprint=evidence.reference.fingerprint,
        generation_policy=policy,
        policy_scientific_version=version,
        development_decision_fingerprint=decision,
    )
    if (
        value.schema_id != LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_ID
        or value.schema_version != LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_VERSION
        or value.qualification_generation_permitted is not True
        or value.contract_id != LEAN_G5_QUALIFICATION_CONTRACT_ID
        or value.g3_evidence_fingerprint != evidence.reference.fingerprint
        or value.generation_policy_fingerprint != policy_fingerprint
        or value.policy_scientific_version != version
        or value.fingerprint != _fingerprint(identity)
    ):
        raise IntegrityError("lean G5 qualification permit identity changed")
    return value


def is_lean_g5_qualification_permit(
    value: object,
) -> TypeGuard[LeanG5QualificationPermit]:
    """Return whether ``value`` is an intact original lean-G5 permit."""

    try:
        verify_lean_g5_qualification_permit(value)
    except (IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def _validated_candidate_policy(
    value: object,
    evidence: LoadedG3EvidenceAuthority,
) -> ScientificGenerationPolicy:
    if not isinstance(value, ScientificGenerationPolicy):
        raise ModelError("lean G5 qualification policy has the wrong type")
    try:
        policy = generation_policy_from_mapping(value.as_dict())
    except (IntegrityError, ModelError, TypeError, ValueError) as error:
        raise ModelError("lean G5 qualification policy is invalid") from error
    if policy != value:
        raise ModelError("lean G5 qualification policy is noncanonical")
    if policy.alpha_policy != "historical_vector":
        raise ModelError(
            "lean G5 neutral qualification requires a typed development selection"
        )
    candidates = evidence.production_artifact.generation_policy
    if not isinstance(candidates, ScientificCandidatePolicySet) or (
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
    ):
        raise ModelError("lean G5 policy is outside the verified G3 candidate set")
    return policy


def _permit_identity(
    *,
    g3_evidence_fingerprint: str,
    generation_policy: ScientificGenerationPolicy,
    policy_scientific_version: int,
    development_decision_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_id": LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_ID,
        "schema_version": LEAN_G5_QUALIFICATION_PERMIT_SCHEMA_VERSION,
        "qualification_generation_permitted": True,
        "production_generation_authorized": False,
        "contract_id": LEAN_G5_QUALIFICATION_CONTRACT_ID,
        "g3_evidence_fingerprint": g3_evidence_fingerprint,
        "generation_policy": generation_policy.as_dict(),
        "generation_policy_fingerprint": _fingerprint(generation_policy.as_dict()),
        "policy_scientific_version": policy_scientific_version,
        "development_decision_fingerprint": development_decision_fingerprint,
    }


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ModelError(f"{label} fingerprint is invalid")
    return value


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256((payload + "\n").encode()).hexdigest()
