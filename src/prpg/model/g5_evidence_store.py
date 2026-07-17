"""Small no-overwrite store for G5 lifecycle evidence and final authority.

The store persists only canonical JSON identities.  Numerical threshold and
validation artifacts remain in their owning statistical stores and are bound
here by exact fingerprints.  Reloading reconstructs sealed typed evidence and
re-mints final authority against a freshly verified historical G3 object; it
does not require today's source tree to match the immutable G3 source bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_evidence_store import LoadedG3EvidenceAuthority
from prpg.model.g5_generation_authority import (
    G5DevelopmentPolicyDecision,
    G5DevelopmentPolicySelection,
    G5MetaValidationEvidence,
    G5QualificationEvidence,
    G5QualifiedProductionAuthority,
    G5ThresholdEvidence,
    is_g5_noncanonical_test_fixture,
    produce_g5_development_policy_selection,
    produce_g5_meta_validation_evidence,
    produce_g5_qualification_evidence,
    produce_g5_qualified_production_authority,
    produce_g5_threshold_evidence,
    verify_g5_development_policy_selection,
    verify_g5_meta_validation_evidence,
    verify_g5_qualification_evidence,
    verify_g5_qualified_production_authority,
    verify_g5_threshold_evidence,
)
from prpg.model.scientific_policy import generation_policy_from_mapping
from prpg.validation.adversaries import (
    G5_ADVERSARY_SCHEMA_ID,
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    G5_MEMORIZATION_HORIZON,
    G5_RIDGE_PENALTY,
    G5AdversaryFitPair,
    G5AdversaryModel,
)
from prpg.validation.records import (
    canonical_results_from_dict,
    canonical_thresholds_from_dict,
)

G5_EVIDENCE_STORE_SCHEMA_VERSION: Final = 1
G5_EVIDENCE_DIRECTORY: Final = "g5-evidence"

G5EvidenceKind: TypeAlias = Literal[
    "development-selection",
    "adversary-fit",
    "threshold",
    "qualification",
    "meta-validation",
    "final-authority",
]

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_ADVERSARY_ARRAY_SHAPES: Final = {
    "linear_center": (3,),
    "linear_scale": (3,),
    "linear_coefficients": (4, 3),
    "quadratic_center": (3,),
    "quadratic_scale": (3,),
    "quadratic_coefficients": (4, 3),
    "outcome_scale": (3,),
}


@dataclass(frozen=True, slots=True)
class StoredG5EvidenceRecord:
    """Location and identity of one newly published immutable G5 manifest."""

    kind: G5EvidenceKind
    fingerprint: str
    path: Path
    bytes: int


class G5EvidenceStore:
    """Publish and reload small content-addressed G5 lifecycle manifests."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store_root = self.artifact_root / G5_EVIDENCE_DIRECTORY

    def publish_development_selection(
        self, value: G5DevelopmentPolicySelection
    ) -> StoredG5EvidenceRecord:
        """Publish the nonauthorizing neutral-selection receipt."""

        checked = verify_g5_development_policy_selection(value)
        self._require_existing(
            "threshold",
            checked.baseline_threshold_evidence_fingerprint,
            checked.baseline_threshold_evidence.identity_dict(),
        )
        return self._publish(
            "development-selection", checked.fingerprint, checked.identity_dict()
        )

    def publish_adversary_fit(
        self, value: G5AdversaryFitPair
    ) -> StoredG5EvidenceRecord:
        """Persist exact G5 development-fit coefficients and memorized outcomes."""

        payload = _adversary_fit_payload(value)
        fingerprint = _string(payload, "fingerprint")
        directory = self._directory("adversary-fit", create=True)
        destination = directory / f"{fingerprint}.json"
        content = _canonical_bytes(payload)
        _publish_content_no_replace(destination, content)
        loaded = self.load_adversary_fit(fingerprint)
        if loaded.fingerprint != value.fingerprint:
            raise IntegrityError("stored G5 adversary fit differs after publication")
        return StoredG5EvidenceRecord(
            "adversary-fit", fingerprint, destination, len(content)
        )

    def publish_threshold_evidence(
        self, value: G5ThresholdEvidence
    ) -> StoredG5EvidenceRecord:
        """Publish one passed threshold-evidence manifest without overwrite."""

        checked = verify_g5_threshold_evidence(value)
        if is_g5_noncanonical_test_fixture(checked):
            raise ModelError("G5 store rejects noncanonical test threshold evidence")
        self.load_adversary_fit(checked.adversary_fit_fingerprint)
        if checked.development_selection is not None:
            self._require_existing(
                "development-selection",
                checked.development_selection.fingerprint,
                checked.development_selection.identity_dict(),
            )
        return self._publish("threshold", checked.fingerprint, checked.identity_dict())

    def publish_qualification_evidence(
        self, value: G5QualificationEvidence
    ) -> StoredG5EvidenceRecord:
        """Publish one passed candidate-result manifest without overwrite."""

        checked = verify_g5_qualification_evidence(value)
        if is_g5_noncanonical_test_fixture(checked):
            raise ModelError(
                "G5 store rejects noncanonical test qualification evidence"
            )
        self._require_existing(
            "threshold",
            checked.threshold_evidence_fingerprint,
            checked.threshold_evidence.identity_dict(),
        )
        self.load_adversary_fit(checked.adversary_fit_fingerprint)
        return self._publish(
            "qualification", checked.fingerprint, checked.identity_dict()
        )

    def publish_meta_validation_evidence(
        self, value: G5MetaValidationEvidence
    ) -> StoredG5EvidenceRecord:
        """Publish one passed meta-validation manifest without overwrite."""

        checked = verify_g5_meta_validation_evidence(value)
        if is_g5_noncanonical_test_fixture(checked):
            raise ModelError("G5 store rejects noncanonical test meta evidence")
        self._require_existing(
            "threshold",
            checked.threshold_evidence_fingerprint,
            checked.threshold_evidence.identity_dict(),
        )
        self._require_existing(
            "qualification",
            checked.qualification_evidence_fingerprint,
            checked.qualification_evidence.identity_dict(),
        )
        return self._publish(
            "meta-validation", checked.fingerprint, checked.identity_dict()
        )

    def publish_final_authority(
        self, value: G5QualifiedProductionAuthority
    ) -> StoredG5EvidenceRecord:
        """Publish final authority only after all referenced manifests exist."""

        checked = verify_g5_qualified_production_authority(value)
        if is_g5_noncanonical_test_fixture(checked):
            raise ModelError("G5 store rejects noncanonical test final authority")
        parents: tuple[tuple[G5EvidenceKind, str, Mapping[str, object]], ...] = (
            (
                "threshold",
                checked.threshold_evidence_fingerprint,
                checked.threshold_evidence.identity_dict(),
            ),
            (
                "qualification",
                checked.qualification_evidence_fingerprint,
                checked.qualification_evidence.identity_dict(),
            ),
            (
                "meta-validation",
                checked.meta_validation_evidence_fingerprint,
                checked.meta_validation_evidence.identity_dict(),
            ),
        )
        for kind, fingerprint, identity in parents:
            self._require_existing(kind, fingerprint, identity)
        self.load_adversary_fit(checked.adversary_fit_fingerprint)
        return self._publish(
            "final-authority", checked.fingerprint, checked.identity_dict()
        )

    def load_development_selection(
        self,
        fingerprint: str,
        *,
        g3_evidence: LoadedG3EvidenceAuthority,
    ) -> G5DevelopmentPolicySelection:
        """Reload a neutral selection and its exact historical threshold parent."""

        identity = self._load_identity("development-selection", fingerprint)
        baseline_threshold = self.load_threshold_evidence(
            _string(identity, "baseline_threshold_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        baseline_results = canonical_results_from_dict(
            dict(_mapping(identity, "baseline_qualification_results"))
        )
        raw_decisions = identity.get("decisions")
        if not isinstance(raw_decisions, list):
            raise IntegrityError("G5 selection decisions are not a list")
        decisions: list[G5DevelopmentPolicyDecision] = []
        for raw in raw_decisions:
            if not isinstance(raw, Mapping):
                raise IntegrityError("G5 selection decision is not an object")
            evaluated = _boolean(raw, "evaluated")
            passed_value = raw.get("passed")
            result_fingerprint = raw.get("development_result_fingerprint")
            if passed_value is not None and type(passed_value) is not bool:
                raise IntegrityError("G5 selection pass decision is invalid")
            if result_fingerprint is not None:
                _required_fingerprint(result_fingerprint, "G5 development result")
            neutral_lambda = _finite_float(raw, "neutral_lambda")
            decisions.append(
                G5DevelopmentPolicyDecision(
                    generation_policy=generation_policy_from_mapping(
                        _mapping(raw, "generation_policy")
                    ),
                    generation_policy_fingerprint=_string(
                        raw, "generation_policy_fingerprint"
                    ),
                    neutral_lambda=neutral_lambda,
                    policy_scientific_version=_integer(
                        raw, "policy_scientific_version"
                    ),
                    evaluated=evaluated,
                    development_result_fingerprint=cast(str | None, result_fingerprint),
                    passed=passed_value,
                )
            )
        rebuilt = produce_g5_development_policy_selection(
            baseline_threshold_evidence=baseline_threshold,
            baseline_qualification_results=baseline_results,
            decisions=decisions,
        )
        self._require_rebuilt(
            fingerprint, identity, rebuilt.fingerprint, rebuilt.identity_dict()
        )
        return rebuilt

    def load_adversary_fit(self, fingerprint: str) -> G5AdversaryFitPair:
        """Reload and fully hash-verify one exact development adversary fit."""

        checked = _required_fingerprint(fingerprint, "G5 adversary fit")
        path = self._directory("adversary-fit", create=False) / f"{checked}.json"
        content = _read_stable_regular(path)
        try:
            value: Any = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError("G5 adversary fit is not valid JSON") from error
        if not isinstance(value, dict) or content != _canonical_bytes(value):
            raise IntegrityError("G5 adversary fit is not canonical JSON")
        rebuilt = _adversary_fit_from_payload(value)
        if rebuilt.fingerprint != checked:
            raise IntegrityError("G5 adversary fit address differs from content")
        return rebuilt

    def load_threshold_evidence(
        self,
        fingerprint: str,
        *,
        g3_evidence: LoadedG3EvidenceAuthority,
    ) -> G5ThresholdEvidence:
        """Reload and re-seal passed threshold evidence against exact G3."""

        identity = self._load_identity("threshold", fingerprint)
        thresholds = canonical_thresholds_from_dict(
            dict(_mapping(identity, "thresholds"))
        )
        policy = generation_policy_from_mapping(_mapping(identity, "generation_policy"))
        adversary_fit = self.load_adversary_fit(
            _string(identity, "adversary_fit_fingerprint")
        )
        selection_fingerprint = identity.get("development_selection_fingerprint")
        if selection_fingerprint is None:
            selection = None
        else:
            selection = self.load_development_selection(
                _required_fingerprint(
                    selection_fingerprint, "G5 development selection"
                ),
                g3_evidence=g3_evidence,
            )
        rebuilt = produce_g5_threshold_evidence(
            g3_evidence=g3_evidence,
            generation_policy=policy,
            thresholds=thresholds,
            estimator_contract_fingerprint=_string(
                identity, "estimator_contract_fingerprint"
            ),
            adversary_fit=adversary_fit,
            seed_registry_fingerprint=_string(identity, "seed_registry_fingerprint"),
            g5_source_code_fingerprint=_string(identity, "g5_source_code_fingerprint"),
            g5_dependency_lock_fingerprint=_string(
                identity, "g5_dependency_lock_fingerprint"
            ),
            development_selection=selection,
        )
        self._require_rebuilt(
            fingerprint, identity, rebuilt.fingerprint, rebuilt.identity_dict()
        )
        return rebuilt

    def load_qualification_evidence(
        self,
        fingerprint: str,
        *,
        g3_evidence: LoadedG3EvidenceAuthority,
    ) -> G5QualificationEvidence:
        """Reload passed qualification evidence and its exact threshold parent."""

        identity = self._load_identity("qualification", fingerprint)
        threshold = self.load_threshold_evidence(
            _string(identity, "threshold_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        policy = generation_policy_from_mapping(_mapping(identity, "generation_policy"))
        results = canonical_results_from_dict(
            dict(_mapping(identity, "qualification_results"))
        )
        adversary_fit = self.load_adversary_fit(
            _string(identity, "adversary_fit_fingerprint")
        )
        if adversary_fit.fingerprint != threshold.adversary_fit_fingerprint:
            raise IntegrityError("G5 qualification adversary-fit parent differs")
        rebuilt = produce_g5_qualification_evidence(
            threshold_evidence=threshold,
            generation_policy=policy,
            qualification_results=results,
        )
        self._require_rebuilt(
            fingerprint, identity, rebuilt.fingerprint, rebuilt.identity_dict()
        )
        return rebuilt

    def load_meta_validation_evidence(
        self,
        fingerprint: str,
        *,
        g3_evidence: LoadedG3EvidenceAuthority,
    ) -> G5MetaValidationEvidence:
        """Reload meta-validation and both exact upstream G5 records."""

        identity = self._load_identity("meta-validation", fingerprint)
        threshold = self.load_threshold_evidence(
            _string(identity, "threshold_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        qualification = self.load_qualification_evidence(
            _string(identity, "qualification_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        # Both loads independently reconstruct threshold evidence.  Identity
        # equality is sufficient here; the coordinator requires object-parent
        # identity, so rebuild qualification with the threshold instance used
        # for meta-validation.
        if qualification.threshold_evidence_fingerprint != threshold.fingerprint:
            raise IntegrityError("G5 meta-validation threshold parents disagree")
        qualification = produce_g5_qualification_evidence(
            threshold_evidence=threshold,
            generation_policy=qualification.generation_policy,
            qualification_results=qualification.qualification_results,
        )
        meta_results = canonical_results_from_dict(
            dict(_mapping(identity, "meta_validation_results"))
        )
        rebuilt = produce_g5_meta_validation_evidence(
            threshold_evidence=threshold,
            qualification_evidence=qualification,
            meta_validation_results=meta_results,
        )
        self._require_rebuilt(
            fingerprint, identity, rebuilt.fingerprint, rebuilt.identity_dict()
        )
        return rebuilt

    def load_final_authority(
        self,
        fingerprint: str,
        *,
        g3_evidence: LoadedG3EvidenceAuthority,
    ) -> G5QualifiedProductionAuthority:
        """Reload final authority without a live-source equality gate."""

        identity = self._load_identity("final-authority", fingerprint)
        threshold = self.load_threshold_evidence(
            _string(identity, "threshold_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        qualification_loaded = self.load_qualification_evidence(
            _string(identity, "qualification_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        if qualification_loaded.threshold_evidence_fingerprint != threshold.fingerprint:
            raise IntegrityError("G5 final authority threshold parents disagree")
        qualification = produce_g5_qualification_evidence(
            threshold_evidence=threshold,
            generation_policy=qualification_loaded.generation_policy,
            qualification_results=qualification_loaded.qualification_results,
        )
        meta_loaded = self.load_meta_validation_evidence(
            _string(identity, "meta_validation_evidence_fingerprint"),
            g3_evidence=g3_evidence,
        )
        if (
            meta_loaded.threshold_evidence_fingerprint != threshold.fingerprint
            or meta_loaded.qualification_evidence_fingerprint
            != qualification.fingerprint
        ):
            raise IntegrityError("G5 final authority evidence parents disagree")
        meta = produce_g5_meta_validation_evidence(
            threshold_evidence=threshold,
            qualification_evidence=qualification,
            meta_validation_results=meta_loaded.meta_validation_results,
        )
        policy = generation_policy_from_mapping(_mapping(identity, "generation_policy"))
        rebuilt = produce_g5_qualified_production_authority(
            g3_evidence=g3_evidence,
            generation_policy=policy,
            threshold_evidence=threshold,
            qualification_evidence=qualification,
            meta_validation_evidence=meta,
        )
        self._require_rebuilt(
            fingerprint, identity, rebuilt.fingerprint, rebuilt.identity_dict()
        )
        return rebuilt

    def _publish(
        self,
        kind: G5EvidenceKind,
        fingerprint: str,
        identity: Mapping[str, object],
    ) -> StoredG5EvidenceRecord:
        checked = _required_fingerprint(fingerprint, "G5 evidence")
        manifest = {
            "schema_version": G5_EVIDENCE_STORE_SCHEMA_VERSION,
            "fingerprint": checked,
            "identity": dict(identity),
        }
        content = _canonical_bytes(manifest)
        if _sha256(identity) != checked:
            raise IntegrityError("G5 evidence address differs from its identity")
        directory = self._directory(kind, create=True)
        destination = directory / f"{checked}.json"
        if destination.exists() or destination.is_symlink():
            raise IntegrityError("G5 evidence already exists; overwrite refused")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checked}.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                if handle.write(content) != len(content):
                    raise IntegrityError("G5 evidence write was incomplete")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise IntegrityError(
                    "G5 evidence already exists; overwrite refused"
                ) from error
            _fsync_directory(directory)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        self._load_identity(kind, checked)
        return StoredG5EvidenceRecord(kind, checked, destination, len(content))

    def _load_identity(
        self, kind: G5EvidenceKind, fingerprint: str
    ) -> dict[str, object]:
        checked = _required_fingerprint(fingerprint, "G5 evidence")
        path = self._directory(kind, create=False) / f"{checked}.json"
        content = _read_stable_regular(path)
        try:
            value: Any = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError("G5 evidence is not valid JSON") from error
        if not isinstance(value, dict) or content != _canonical_bytes(value):
            raise IntegrityError("G5 evidence is not canonical JSON")
        if set(value) != {"schema_version", "fingerprint", "identity"}:
            raise IntegrityError("G5 evidence manifest fields are invalid")
        identity = value["identity"]
        if (
            value["schema_version"] != G5_EVIDENCE_STORE_SCHEMA_VERSION
            or value["fingerprint"] != checked
            or not isinstance(identity, dict)
            or _sha256(identity) != checked
        ):
            raise IntegrityError("G5 evidence manifest identity is invalid")
        return cast(dict[str, object], identity)

    def _require_existing(
        self,
        kind: G5EvidenceKind,
        fingerprint: str,
        identity: Mapping[str, object],
    ) -> None:
        observed = self._load_identity(kind, fingerprint)
        if observed != dict(identity):
            raise IntegrityError("stored G5 evidence differs from its live parent")

    @staticmethod
    def _require_rebuilt(
        expected_fingerprint: str,
        observed_identity: Mapping[str, object],
        rebuilt_fingerprint: str,
        rebuilt_identity: Mapping[str, object],
    ) -> None:
        if rebuilt_fingerprint != expected_fingerprint or dict(
            rebuilt_identity
        ) != dict(observed_identity):
            raise IntegrityError("stored G5 evidence failed typed reconstruction")

    def _directory(self, kind: G5EvidenceKind, *, create: bool) -> Path:
        directory = self.store_root / kind / "sha256"
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory.is_dir() or directory.is_symlink():
            raise IntegrityError("G5 evidence directory is missing or unsafe")
        for parent in (self.store_root, directory.parent):
            if parent.is_symlink():
                raise IntegrityError("G5 evidence parent directory is unsafe")
        return directory


def _adversary_fit_payload(value: G5AdversaryFitPair) -> dict[str, object]:
    if type(value) is not G5AdversaryFitPair:
        raise ModelError("G5 adversary fit has the wrong type")
    monthly = _adversary_model_payload(value.monthly, "monthly")
    daily = _adversary_model_payload(value.daily, "daily")
    expected = _sha256(
        {
            "specification_fingerprint": G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
            "monthly_model_fingerprint": value.monthly.fingerprint,
            "daily_model_fingerprint": value.daily.fingerprint,
        }
    )
    if (
        value.specification_fingerprint != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
        or value.fingerprint != expected
    ):
        raise ModelError("G5 adversary fit identity is invalid")
    return {
        "schema_id": "prpg-g5-adversary-fit-store-v1",
        "schema_version": 1,
        "fingerprint": value.fingerprint,
        "specification_fingerprint": value.specification_fingerprint,
        "monthly": monthly,
        "daily": daily,
    }


def _adversary_model_payload(
    value: G5AdversaryModel, frequency: Literal["monthly", "daily"]
) -> dict[str, object]:
    if type(value) is not G5AdversaryModel or value.frequency != frequency:
        raise ModelError("G5 adversary model frequency/type is invalid")
    arrays = {
        name: _adversary_array(getattr(value, name), name)
        for name in (
            "linear_center",
            "linear_scale",
            "linear_coefficients",
            "quadratic_center",
            "quadratic_scale",
            "quadratic_coefficients",
            "outcome_scale",
        )
    }
    memorized: list[dict[str, object]] = []
    if not isinstance(value.memorized_outcomes, Mapping):
        raise ModelError("G5 adversary memorized outcomes are invalid")
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
        memorized.append({"key_hex": key.hex(), "outcome": list(outcome)})
    if value.memorization_horizon != G5_MEMORIZATION_HORIZON[frequency]:
        raise ModelError("G5 adversary memorization horizon is invalid")
    expected = _adversary_model_fingerprint(
        frequency=frequency,
        arrays=arrays,
        memorized_outcomes=value.memorized_outcomes,
        memorization_horizon=value.memorization_horizon,
    )
    if value.fingerprint != expected:
        raise ModelError("G5 adversary model fingerprint is invalid")
    return {
        "frequency": frequency,
        "arrays": {name: array.tolist() for name, array in arrays.items()},
        "memorization_horizon": value.memorization_horizon,
        "memorized_outcomes": memorized,
        "fingerprint": value.fingerprint,
    }


def _adversary_fit_from_payload(value: Mapping[str, object]) -> G5AdversaryFitPair:
    if set(value) != {
        "schema_id",
        "schema_version",
        "fingerprint",
        "specification_fingerprint",
        "monthly",
        "daily",
    }:
        raise IntegrityError("G5 adversary-fit store fields are invalid")
    if (
        value["schema_id"] != "prpg-g5-adversary-fit-store-v1"
        or value["schema_version"] != 1
        or value["specification_fingerprint"] != G5_ADVERSARY_SPECIFICATION_FINGERPRINT
    ):
        raise IntegrityError("G5 adversary-fit store schema is invalid")
    monthly = _adversary_model_from_payload(_mapping(value, "monthly"), "monthly")
    daily = _adversary_model_from_payload(_mapping(value, "daily"), "daily")
    fingerprint = _string(value, "fingerprint")
    expected = _sha256(
        {
            "specification_fingerprint": G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
            "monthly_model_fingerprint": monthly.fingerprint,
            "daily_model_fingerprint": daily.fingerprint,
        }
    )
    if fingerprint != expected:
        raise IntegrityError("G5 adversary fit fingerprint is invalid")
    return G5AdversaryFitPair(
        monthly=monthly,
        daily=daily,
        specification_fingerprint=G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
        fingerprint=fingerprint,
    )


def _adversary_model_from_payload(
    value: Mapping[str, object], frequency: Literal["monthly", "daily"]
) -> G5AdversaryModel:
    if (
        set(value)
        != {
            "frequency",
            "arrays",
            "memorization_horizon",
            "memorized_outcomes",
            "fingerprint",
        }
        or value["frequency"] != frequency
    ):
        raise IntegrityError("G5 stored adversary model fields are invalid")
    raw_arrays = _mapping(value, "arrays")
    expected_names = {
        "linear_center",
        "linear_scale",
        "linear_coefficients",
        "quadratic_center",
        "quadratic_scale",
        "quadratic_coefficients",
        "outcome_scale",
    }
    if set(raw_arrays) != expected_names:
        raise IntegrityError("G5 stored adversary arrays are incomplete")
    arrays = {name: _adversary_array(raw_arrays[name], name) for name in expected_names}
    raw_memorized = value["memorized_outcomes"]
    if not isinstance(raw_memorized, list):
        raise IntegrityError("G5 stored memorized outcomes are invalid")
    memorized: dict[bytes, tuple[float, float, float]] = {}
    prior_key: bytes | None = None
    for item in raw_memorized:
        if not isinstance(item, Mapping) or set(item) != {"key_hex", "outcome"}:
            raise IntegrityError("G5 stored memorized row is invalid")
        key_hex = _string(item, "key_hex")
        try:
            key = bytes.fromhex(key_hex)
        except ValueError as error:
            raise IntegrityError("G5 stored memorized key is invalid") from error
        raw_outcome = item["outcome"]
        if (
            len(key) != 16
            or (prior_key is not None and key <= prior_key)
            or not isinstance(raw_outcome, list)
            or len(raw_outcome) != 3
        ):
            raise IntegrityError("G5 stored memorized row order/shape is invalid")
        outcome = tuple(
            _finite_number(entry, "memorized outcome") for entry in raw_outcome
        )
        memorized[key] = cast(tuple[float, float, float], outcome)
        prior_key = key
    horizon = _integer(value, "memorization_horizon")
    fingerprint = _string(value, "fingerprint")
    expected = _adversary_model_fingerprint(
        frequency=frequency,
        arrays=arrays,
        memorized_outcomes=memorized,
        memorization_horizon=horizon,
    )
    if fingerprint != expected:
        raise IntegrityError("G5 stored adversary model fingerprint is invalid")
    return G5AdversaryModel(
        frequency=frequency,
        linear_center=arrays["linear_center"],
        linear_scale=arrays["linear_scale"],
        linear_coefficients=arrays["linear_coefficients"],
        quadratic_center=arrays["quadratic_center"],
        quadratic_scale=arrays["quadratic_scale"],
        quadratic_coefficients=arrays["quadratic_coefficients"],
        outcome_scale=arrays["outcome_scale"],
        memorization_horizon=horizon,
        memorized_outcomes=MappingProxyType(memorized),
        fingerprint=fingerprint,
    )


def _adversary_model_fingerprint(
    *,
    frequency: Literal["monthly", "daily"],
    arrays: Mapping[str, np.ndarray[Any, Any]],
    memorized_outcomes: Mapping[bytes, tuple[float, float, float]],
    memorization_horizon: int,
) -> str:
    if memorization_horizon != G5_MEMORIZATION_HORIZON[frequency]:
        raise IntegrityError("G5 adversary memorization horizon changed")
    digest = hashlib.sha256()
    for key in sorted(memorized_outcomes):
        digest.update(key)
        digest.update(np.asarray(memorized_outcomes[key], dtype="<f8").tobytes())
    return _sha256(
        {
            "schema_id": G5_ADVERSARY_SCHEMA_ID,
            "frequency": frequency,
            "ridge_penalty": G5_RIDGE_PENALTY,
            **{
                f"{name}_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()
                for name, array in arrays.items()
            },
            "memorization_horizon": memorization_horizon,
            "memorization_digest": digest.hexdigest(),
        }
    )


def _adversary_array(value: object, label: str) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(value, dtype="<f8", order="C")
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"G5 adversary {label} is not numeric") from error
    if array.shape != _ADVERSARY_ARRAY_SHAPES.get(label) or not bool(
        np.isfinite(array).all()
    ):
        raise IntegrityError(f"G5 adversary {label} shape/values are invalid")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    return frozen


def _publish_content_no_replace(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError("G5 evidence already exists; overwrite refused")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(content) != len(content):
                raise IntegrityError("G5 evidence write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise IntegrityError(
                "G5 evidence already exists; overwrite refused"
            ) from error
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IntegrityError(f"G5 evidence {label} is not numeric")
    result = float(value)
    if not np.isfinite(result):
        raise IntegrityError(f"G5 evidence {label} is non-finite")
    return result


def _finite_float(identity: Mapping[str, object], key: str) -> float:
    return _finite_number(identity.get(key), key)


def _string(identity: Mapping[str, object], key: str) -> str:
    value = identity.get(key)
    if not isinstance(value, str):
        raise IntegrityError(f"G5 evidence {key} is not a string")
    return value


def _integer(identity: Mapping[str, object], key: str) -> int:
    value = identity.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"G5 evidence {key} is not an integer")
    return value


def _boolean(identity: Mapping[str, object], key: str) -> bool:
    value = identity.get(key)
    if type(value) is not bool:
        raise IntegrityError(f"G5 evidence {key} is not a boolean")
    return value


def _mapping(identity: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = identity.get(key)
    if not isinstance(value, Mapping):
        raise IntegrityError(f"G5 evidence {key} is not an object")
    return cast(Mapping[str, object], value)


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ModelError(f"{label} fingerprint is invalid")
    return value


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
    ).encode()


def _sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_stable_regular(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise IntegrityError("G5 evidence is missing") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise IntegrityError("G5 evidence is not a regular file")
    try:
        content = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise IntegrityError("G5 evidence cannot be read") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise IntegrityError("G5 evidence changed while being read")
    return content


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
