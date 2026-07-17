"""Canonical G3 report production and logical two-artifact publication.

The generic scientific-artifact serializer intentionally remains useful for
report-only and post-G5 compatibility work.  Canonical G3 publication is more
restrictive: callers cannot supply report metrics, arrays, a selected alpha
policy, or either artifact independently.  This module reconstructs both
role-specific artifacts from the exact producer-sealed first-25 task prefix,
publishes their content-addressed directories, and writes one pair receipt
last.  Artifact directories left by an interrupted attempt are harmless and
are byte-verified/reused on retry; only the final receipt denotes a complete
logical pair, and even that receipt never authorizes generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, TypeAlias, TypeGuard, cast

import numpy as np

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import ModelArtifactStore
from prpg.model.block_execution_registered import registered_block_replay_source
from prpg.model.calibration import DecodedReturnPools
from prpg.model.calibration_identity import decoded_return_pools_fingerprint
from prpg.model.dependence import DependenceReference, fit_dependence_reference
from prpg.model.g3_registry import G3_GATE_ORDER
from prpg.model.hmm import GaussianHMMFit, summarize_state_path
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    kernel_calibration_execution_identity,
    kernel_calibration_replay_source,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.model.scientific_artifact import (
    ArtifactRole,
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    ScientificArtifactProvenance,
    ScientificModelArtifactDraft,
    ScientificReportPayload,
    draft_scientific_model_artifact_from_phase3_results,
    is_verified_scientific_model_artifact,
    load_scientific_model_artifact,
    publish_scientific_model_artifact,
    required_scientific_documents,
    scientific_report_payload,
    scientific_report_planned_look_families,
    scientific_report_task_ids,
)
from prpg.model.scientific_policy import (
    ScientificCandidatePolicySet,
    candidate_policy_set_from_kernel_results,
)

SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID: Final = "prpg-g3-scientific-artifact-pair-v1"
SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION: Final = 1
SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY: Final = "scientific-artifact-pairs"
SCIENTIFIC_ARTIFACT_PAIR_RECEIPT: Final = "pair-receipt.json"
CLOSED_REPORT_METRICS_SCHEMA_ID: Final = "prpg-g3-closed-report-metrics-v1"
CLOSED_REPORT_METRICS_SCHEMA_VERSION: Final = 1

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DRAFT_CAPABILITY = object()
_PAIR_CAPABILITY = object()
_MINTED_DRAFTS: dict[int, CanonicalScientificModelArtifactDraft] = {}
_MINTED_PAIRS: dict[int, LoadedScientificArtifactPair] = {}

PairRole: TypeAlias = Literal["design", "production"]


@dataclass(frozen=True, slots=True, init=False)
class CanonicalScientificModelArtifactDraft:
    """One role draft derived only from exact first-25 producer parents."""

    generation_authorized: ClassVar[Literal[False]] = False
    role: PairRole
    selected_k: int
    run_id: str
    prefix_fingerprint: str
    actual_bridge_parent_view_fingerprint: str
    role_source_fingerprint: str
    report_set_fingerprint: str
    candidate_policy_set: ScientificCandidatePolicySet
    draft: ScientificModelArtifactDraft
    _prefix: object = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "CanonicalScientificModelArtifactDraft is created only from a "
            "sealed first-25 prefix"
        )


@dataclass(frozen=True, slots=True)
class ScientificArtifactPairReference:
    """Integrity-verified reference to one immutable pair receipt."""

    fingerprint: str
    path: Path
    manifest: Mapping[str, Any]
    reused: bool


@dataclass(frozen=True, slots=True, init=False)
class LoadedScientificArtifactPair:
    """Reloaded pair tied to one exact live first-25 prefix.

    This object proves report, role-source, artifact, and run lineage.  It is
    deliberately not a generation capability; G5 has not selected or
    qualified an alpha policy at this stage.
    """

    generation_authorized: ClassVar[Literal[False]] = False
    verification_scope: ClassVar[
        Literal["g3_pair_receipt_artifacts_reports_and_first25_lineage"]
    ] = "g3_pair_receipt_artifacts_reports_and_first25_lineage"

    reference: ScientificArtifactPairReference
    design: LoadedScientificModelArtifact
    production: LoadedScientificModelArtifact
    selected_k: int
    run_id: str
    binding: ScientificArtifactBinding
    provenance: ScientificArtifactProvenance
    prefix_fingerprint: str
    actual_bridge_parent_view_fingerprint: str
    design_role_source_fingerprint: str
    production_role_source_fingerprint: str
    report_pair_fingerprint: str
    _prefix: object = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("LoadedScientificArtifactPair is created only by its store")


@dataclass(frozen=True, slots=True)
class _RoleInputs:
    role: PairRole
    block_view: object
    fit: GaussianHMMFit
    pools: DecodedReturnPools
    monthly_continuation: np.ndarray[Any, Any]
    daily_continuation: np.ndarray[Any, Any]
    monthly_reference: DependenceReference
    daily_reference: DependenceReference
    monthly_kernel: KernelCalibrationExecution
    daily_kernel: KernelCalibrationExecution
    role_source_fingerprint: str


class ScientificArtifactPairStore:
    """Publish and reload pair receipts below one artifact root."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.store_root = (
            self.artifact_root / SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY / "sha256"
        )

    def publish(
        self,
        *,
        model_store: ModelArtifactStore,
        design: CanonicalScientificModelArtifactDraft,
        production: CanonicalScientificModelArtifactDraft,
    ) -> LoadedScientificArtifactPair:
        """Publish both artifacts, then atomically publish the pair receipt."""

        self._require_same_root(model_store)
        _verify_canonical_draft(design, expected_role="design")
        _verify_canonical_draft(production, expected_role="production")
        if design._prefix is not production._prefix:
            raise ModelError(
                "canonical artifact drafts use different first-25 prefixes"
            )
        if (
            design.run_id != production.run_id
            or design.selected_k != production.selected_k
            or design.prefix_fingerprint != production.prefix_fingerprint
            or design.actual_bridge_parent_view_fingerprint
            != production.actual_bridge_parent_view_fingerprint
            or design.draft.binding != production.draft.binding
            or design.draft.provenance != production.draft.provenance
        ):
            raise ModelError("canonical artifact drafts do not form one exact pair")

        # These two immutable creates are intentionally ordered before the
        # receipt.  A crash at either point leaves no observable complete pair.
        loaded_design = publish_scientific_model_artifact(model_store, design.draft)
        _verify_loaded_against_draft(loaded_design, design)
        loaded_production = publish_scientific_model_artifact(
            model_store, production.draft
        )
        _verify_loaded_against_draft(loaded_production, production)

        identity = _pair_identity(
            design=loaded_design,
            production=loaded_production,
            design_draft=design,
            production_draft=production,
        )
        fingerprint = _sha256_bytes(_canonical_bytes(identity))
        manifest = {
            "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
            "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
            "pair_receipt_fingerprint": fingerprint,
            "identity": identity,
        }
        reused = self._write_or_reuse(fingerprint, _canonical_bytes(manifest))
        return self.load(
            fingerprint,
            model_store=model_store,
            prefix=design._prefix,
            reused=reused,
        )

    def load(
        self,
        fingerprint: str,
        *,
        model_store: ModelArtifactStore,
        prefix: object,
        reused: bool = True,
    ) -> LoadedScientificArtifactPair:
        """Reload and rederive both artifacts from the supplied live prefix."""

        self._require_same_root(model_store)
        checked_prefix = _checked_prefix(prefix)
        reference = self.verify(fingerprint, reused=reused)
        identity = reference.manifest["identity"]
        artifacts = identity.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise IntegrityError("scientific pair artifact identities are missing")
        design_identity = artifacts.get("design")
        production_identity = artifacts.get("production")
        if not isinstance(design_identity, Mapping) or not isinstance(
            production_identity, Mapping
        ):
            raise IntegrityError("scientific pair role identities are invalid")
        design_loaded = load_scientific_model_artifact(
            model_store,
            _required_sha(
                design_identity.get("artifact_fingerprint"),
                "design artifact fingerprint",
                IntegrityError,
            ),
        )
        production_loaded = load_scientific_model_artifact(
            model_store,
            _required_sha(
                production_identity.get("artifact_fingerprint"),
                "production artifact fingerprint",
                IntegrityError,
            ),
        )
        design_draft = build_canonical_scientific_model_artifact_draft(
            checked_prefix,
            role="design",
        )
        production_draft = build_canonical_scientific_model_artifact_draft(
            checked_prefix,
            role="production",
        )
        _verify_loaded_against_draft(design_loaded, design_draft)
        _verify_loaded_against_draft(production_loaded, production_draft)
        expected = _pair_identity(
            design=design_loaded,
            production=production_loaded,
            design_draft=design_draft,
            production_draft=production_draft,
        )
        if _canonical_bytes(
            _json_value(identity, "scientific_pair_receipt.identity")
        ) != _canonical_bytes(expected):
            raise IntegrityError(
                "scientific artifact pair receipt differs from live first-25 lineage"
            )
        pair = object.__new__(LoadedScientificArtifactPair)
        for name, value in (
            ("reference", reference),
            ("design", design_loaded),
            ("production", production_loaded),
            ("selected_k", design_draft.selected_k),
            ("run_id", design_draft.run_id),
            ("binding", design_draft.draft.binding),
            ("provenance", design_draft.draft.provenance),
            ("prefix_fingerprint", design_draft.prefix_fingerprint),
            (
                "actual_bridge_parent_view_fingerprint",
                design_draft.actual_bridge_parent_view_fingerprint,
            ),
            ("design_role_source_fingerprint", design_draft.role_source_fingerprint),
            (
                "production_role_source_fingerprint",
                production_draft.role_source_fingerprint,
            ),
            ("report_pair_fingerprint", str(expected["report_pair_fingerprint"])),
            ("_prefix", checked_prefix),
            ("_seal", _PAIR_CAPABILITY),
        ):
            object.__setattr__(pair, name, value)
        _MINTED_PAIRS[id(pair)] = pair
        return pair

    def verify(
        self, fingerprint: str, *, reused: bool = True
    ) -> ScientificArtifactPairReference:
        """Verify receipt bytes and content address without granting authority."""

        checked = _required_sha(
            fingerprint, "scientific artifact pair fingerprint", IntegrityError
        )
        _reject_symlink_ancestors(self.store_root)
        root = self.store_root / checked
        if not root.is_dir() or root.is_symlink():
            raise IntegrityError(
                "scientific artifact pair directory is missing or unsafe"
            )
        names = {entry.name for entry in root.iterdir()}
        if names != {SCIENTIFIC_ARTIFACT_PAIR_RECEIPT}:
            raise IntegrityError(
                "scientific artifact pair directory allowlist is invalid"
            )
        receipt_path = root / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise IntegrityError("scientific artifact pair receipt is unsafe")
        content = _read_stable_regular_file(receipt_path)
        try:
            parsed = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "scientific artifact pair receipt is invalid JSON"
            ) from error
        if not isinstance(parsed, dict) or content != _canonical_bytes(parsed):
            raise IntegrityError(
                "scientific artifact pair receipt is not canonical JSON"
            )
        if set(parsed) != {
            "schema_id",
            "schema_version",
            "pair_receipt_fingerprint",
            "identity",
        }:
            raise IntegrityError("scientific artifact pair receipt fields are invalid")
        if (
            parsed["schema_id"] != SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID
            or parsed["schema_version"] != SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION
            or parsed["pair_receipt_fingerprint"] != checked
            or not isinstance(parsed["identity"], Mapping)
            or _sha256_bytes(_canonical_bytes(parsed["identity"])) != checked
        ):
            raise IntegrityError("scientific artifact pair receipt identity is invalid")
        frozen = _freeze(parsed)
        assert isinstance(frozen, Mapping)
        return ScientificArtifactPairReference(checked, root, frozen, reused)

    def _require_same_root(self, model_store: ModelArtifactStore) -> None:
        if not isinstance(model_store, ModelArtifactStore):
            raise ModelError("scientific pair publication requires ModelArtifactStore")
        if model_store.artifact_root.resolve() != self.artifact_root.resolve():
            raise ModelError("scientific pair and model stores use different roots")

    def _write_or_reuse(self, fingerprint: str, content: bytes) -> bool:
        self._prepare_root()
        destination = self.store_root / fingerprint
        if destination.exists() or destination.is_symlink():
            existing = self.verify(fingerprint)
            if (
                _read_stable_regular_file(
                    existing.path / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT
                )
                != content
            ):
                raise IntegrityError("immutable scientific pair receipt collision")
            return True
        stage = Path(
            tempfile.mkdtemp(prefix=f".stage-{fingerprint[:12]}-", dir=self.store_root)
        )
        try:
            target = stage / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                with suppress(OSError):
                    os.close(descriptor)
                raise
            _fsync_directory(stage)
            try:
                os.rename(stage, destination)
            except OSError as error:
                if destination.exists() or destination.is_symlink():
                    existing = self.verify(fingerprint)
                    if (
                        _read_stable_regular_file(
                            existing.path / SCIENTIFIC_ARTIFACT_PAIR_RECEIPT
                        )
                        != content
                    ):
                        raise IntegrityError(
                            "concurrent scientific pair receipt differs"
                        ) from error
                    return True
                raise
            _fsync_directory(self.store_root)
            self.verify(fingerprint, reused=False)
            return False
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _prepare_root(self) -> None:
        _reject_symlink_ancestors(self.artifact_root)
        if self.artifact_root.exists() and (
            self.artifact_root.is_symlink() or not self.artifact_root.is_dir()
        ):
            raise IntegrityError("scientific artifact pair root is unsafe")
        self.artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        pair_root = self.artifact_root / SCIENTIFIC_ARTIFACT_PAIR_DIRECTORY
        if pair_root.exists() and (pair_root.is_symlink() or not pair_root.is_dir()):
            raise IntegrityError("scientific artifact pair store path is unsafe")
        pair_root.mkdir(exist_ok=True, mode=0o700)
        if self.store_root.exists() and (
            self.store_root.is_symlink() or not self.store_root.is_dir()
        ):
            raise IntegrityError("scientific artifact pair store path is unsafe")
        self.store_root.mkdir(exist_ok=True, mode=0o700)
        _reject_symlink_ancestors(self.store_root)
        if any(
            path.is_symlink() or not path.is_dir()
            for path in (self.artifact_root, pair_root, self.store_root)
        ):
            raise IntegrityError("scientific artifact pair store path is unsafe")


def build_canonical_scientific_model_artifact_draft(
    prefix: object,
    *,
    role: ArtifactRole,
) -> CanonicalScientificModelArtifactDraft:
    """Derive one role draft from exact first-25 sources and no caller metrics."""

    checked_prefix = _checked_prefix(prefix)
    checked_role = _role(role)
    inputs = _role_inputs(checked_prefix, checked_role)
    policy_set = candidate_policy_set_from_kernel_results(
        monthly=inputs.monthly_kernel,
        daily=inputs.daily_kernel,
    )
    reports = _closed_reports(checked_prefix, inputs)
    draft = draft_scientific_model_artifact_from_phase3_results(
        role=checked_role,
        binding=checked_prefix.binding,
        provenance=checked_prefix.provenance,
        fit=inputs.fit,
        pools=inputs.pools,
        monthly_continuation=inputs.monthly_continuation,
        daily_continuation=inputs.daily_continuation,
        monthly_dependence_reference=inputs.monthly_reference,
        daily_dependence_reference=inputs.daily_reference,
        monthly_kernel=inputs.monthly_kernel,
        daily_kernel=inputs.daily_kernel,
        candidate_policy_set=policy_set,
        reports=reports,
    )
    report_fingerprint = _report_set_fingerprint(reports)
    actual_bridge = checked_prefix._handlers.actual_artifact_bridge
    actual_bridge_fingerprint = getattr(actual_bridge, "view_fingerprint", None)
    _required_sha(
        actual_bridge_fingerprint,
        "actual bridge parent view fingerprint",
        ModelError,
    )
    result = object.__new__(CanonicalScientificModelArtifactDraft)
    for name, value in (
        ("role", checked_role),
        ("selected_k", checked_prefix.selected_n_states),
        ("run_id", checked_prefix.run_id),
        ("prefix_fingerprint", checked_prefix.prefix_fingerprint),
        ("actual_bridge_parent_view_fingerprint", actual_bridge_fingerprint),
        ("role_source_fingerprint", inputs.role_source_fingerprint),
        ("report_set_fingerprint", report_fingerprint),
        ("candidate_policy_set", policy_set),
        ("draft", draft),
        ("_prefix", checked_prefix),
        ("_seal", _DRAFT_CAPABILITY),
    ):
        object.__setattr__(result, name, value)
    _MINTED_DRAFTS[id(result)] = result
    return result


def is_verified_scientific_artifact_pair(
    value: object,
) -> TypeGuard[LoadedScientificArtifactPair]:
    """Return whether a pair is unchanged and still descends from its prefix."""

    if not isinstance(value, LoadedScientificArtifactPair):
        return False
    try:
        scientific_artifact_pair_identity(value)
    except (IntegrityError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def scientific_artifact_pair_identity(
    value: LoadedScientificArtifactPair,
) -> Mapping[str, Any]:
    """Recompute an in-memory pair identity against exact live role sources."""

    if (
        not isinstance(value, LoadedScientificArtifactPair)
        or getattr(value, "_seal", None) is not _PAIR_CAPABILITY
        or _MINTED_PAIRS.get(id(value)) is not value
    ):
        raise ModelError("scientific artifact pair lacks store-loader authority")
    prefix = _checked_prefix(value._prefix)
    design_draft = build_canonical_scientific_model_artifact_draft(
        prefix, role="design"
    )
    production_draft = build_canonical_scientific_model_artifact_draft(
        prefix, role="production"
    )
    _verify_loaded_against_draft(value.design, design_draft)
    _verify_loaded_against_draft(value.production, production_draft)
    expected = _pair_identity(
        design=value.design,
        production=value.production,
        design_draft=design_draft,
        production_draft=production_draft,
    )
    if (
        value.reference.fingerprint != _sha256_bytes(_canonical_bytes(expected))
        or _canonical_bytes(
            _json_value(
                value.reference.manifest["identity"],
                "scientific_pair_receipt.identity",
            )
        )
        != _canonical_bytes(expected)
        or value.selected_k != design_draft.selected_k
        or value.run_id != design_draft.run_id
        or value.binding != design_draft.draft.binding
        or value.provenance != design_draft.draft.provenance
        or value.prefix_fingerprint != design_draft.prefix_fingerprint
        or value.actual_bridge_parent_view_fingerprint
        != design_draft.actual_bridge_parent_view_fingerprint
        or value.design_role_source_fingerprint != design_draft.role_source_fingerprint
        or value.production_role_source_fingerprint
        != production_draft.role_source_fingerprint
        or value.report_pair_fingerprint != expected["report_pair_fingerprint"]
    ):
        raise ModelError("scientific artifact pair in-memory identity changed")
    return MappingProxyType(expected)


def verify_durable_scientific_artifact_pair_receipt(
    *,
    pair_store: ScientificArtifactPairStore,
    model_store: ModelArtifactStore,
    fingerprint: str,
    design: LoadedScientificModelArtifact,
    production: LoadedScientificModelArtifact,
    run_id: str,
    selected_k: int,
    binding: ScientificArtifactBinding,
    provenance: ScientificArtifactProvenance,
    task_result_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    planned_look_fingerprints: Mapping[str, tuple[str, ...]],
    decision_documents: Mapping[str, Mapping[str, Any]],
    actual_bridge_parent_view_fingerprint: str,
    design_role_source_fingerprint: str,
    production_role_source_fingerprint: str,
) -> ScientificArtifactPairReference:
    """Reverify a completed-run receipt without live task-26 producer objects.

    Durable G3 loading no longer has the live numerical task-source objects.
    It does have their immutable first-25 task/checkpoint/look receipts and
    exact decision documents.  This verifier therefore checks the pair and
    every closed report against those durable records while the ordinary
    artifact loader independently rechecks all bytes and schemas.
    """

    if not isinstance(pair_store, ScientificArtifactPairStore):
        raise ModelError("durable pair verification requires its receipt store")
    pair_store._require_same_root(model_store)
    reference = pair_store.verify(fingerprint)
    identity = reference.manifest["identity"]
    if not isinstance(identity, Mapping):
        raise IntegrityError("durable scientific artifact pair identity is invalid")

    first_25 = G3_GATE_ORDER[:-1]
    if set(task_result_fingerprints) != set(first_25) or set(
        checkpoint_fingerprints
    ) != set(first_25):
        raise IntegrityError("durable scientific pair receipt maps are not exact")
    expected_tasks = {
        task_id: _required_sha(
            task_result_fingerprints[task_id],
            f"{task_id} durable task-result fingerprint",
            IntegrityError,
        )
        for task_id in first_25
    }
    expected_checkpoints = {
        task_id: _required_sha(
            checkpoint_fingerprints[task_id],
            f"{task_id} durable checkpoint fingerprint",
            IntegrityError,
        )
        for task_id in first_25
    }
    expected_looks: dict[str, list[str]] = {}
    for family_id, values in planned_look_fingerprints.items():
        if (
            not isinstance(family_id, str)
            or not family_id
            or not isinstance(values, tuple)
        ):
            raise IntegrityError("durable scientific pair planned-look map is invalid")
        expected_looks[family_id] = [
            _required_sha(
                value,
                f"{family_id} durable planned-look fingerprint",
                IntegrityError,
            )
            for value in values
        ]

    checked_run_id = _required_sha(
        run_id, "durable pair run fingerprint", IntegrityError
    )
    checked_bridge_parent = _required_sha(
        actual_bridge_parent_view_fingerprint,
        "durable pair bridge-parent fingerprint",
        IntegrityError,
    )
    checked_role_sources = {
        "design": _required_sha(
            design_role_source_fingerprint,
            "durable design role-source fingerprint",
            IntegrityError,
        ),
        "production": _required_sha(
            production_role_source_fingerprint,
            "durable production role-source fingerprint",
            IntegrityError,
        ),
    }
    if (
        not is_verified_scientific_model_artifact(design)
        or not is_verified_scientific_model_artifact(production)
        or design.role != "design"
        or production.role != "production"
    ):
        raise IntegrityError("durable scientific pair artifacts are not typed roles")

    expected_artifacts = {"design": design, "production": production}
    report_fingerprints: dict[str, str] = {}
    artifact_identities: dict[str, dict[str, object]] = {}
    candidate_policy_sets: dict[str, dict[str, object]] = {}
    for role, artifact in expected_artifacts.items():
        if (
            artifact.selected_k != selected_k
            or artifact.binding != binding
            or artifact.provenance != provenance
            or not isinstance(artifact.generation_policy, ScientificCandidatePolicySet)
        ):
            raise IntegrityError("durable scientific pair role identity is invalid")
        _verify_closed_reports_from_durable_decisions(
            artifact,
            task_result_fingerprints=task_result_fingerprints,
            checkpoint_fingerprints=checkpoint_fingerprints,
            planned_look_fingerprints=planned_look_fingerprints,
            decision_documents=decision_documents,
        )
        report_fingerprint = _loaded_report_set_fingerprint(artifact)
        policy_set = artifact.generation_policy.as_dict()
        report_fingerprints[role] = report_fingerprint
        candidate_policy_sets[role] = policy_set
        artifact_identities[role] = {
            "role": role,
            "artifact_fingerprint": artifact.reference.fingerprint,
            "core_model_fingerprint": artifact.core_model_fingerprint,
            "parameter_set_fingerprint": artifact.parameter_set_fingerprint,
            "role_source_fingerprint": checked_role_sources[role],
            "report_set_fingerprint": report_fingerprint,
            "candidate_policy_set_fingerprint": _sha256_bytes(
                _canonical_bytes(policy_set)
            ),
        }

    bridge_decision = decision_documents.get("actual_artifact_bridge")
    if not isinstance(bridge_decision, Mapping):
        raise IntegrityError("durable scientific pair lacks bridge decision")
    bridge_evidence = bridge_decision.get("evidence")
    if (
        not isinstance(bridge_evidence, Mapping)
        or bridge_evidence.get("view_fingerprint") != checked_bridge_parent
    ):
        raise IntegrityError("durable scientific pair bridge ancestry changed")

    prefix_identity = {
        "binding": binding.as_dict(),
        "checkpoint_fingerprints": expected_checkpoints,
        "planned_look_fingerprints": expected_looks,
        "provenance": provenance.as_dict(),
        "run_id": checked_run_id,
        "selected_n_states": selected_k,
        "task_result_fingerprints": expected_tasks,
    }
    report_pair_fingerprint = _sha256_bytes(
        _canonical_bytes(
            {
                "schema_id": "prpg-g3-scientific-report-pair-v1",
                "design": report_fingerprints["design"],
                "production": report_fingerprints["production"],
            }
        )
    )
    expected_identity = {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "run_id": checked_run_id,
        "selected_k": selected_k,
        "prefix_fingerprint": _sha256_bytes(_canonical_bytes(prefix_identity)),
        "binding": binding.as_dict(),
        "provenance": provenance.as_dict(),
        "task_result_fingerprints": expected_tasks,
        "checkpoint_fingerprints": expected_checkpoints,
        "planned_look_fingerprints": expected_looks,
        "actual_bridge_parent_view_fingerprint": checked_bridge_parent,
        "candidate_policy_sets": candidate_policy_sets,
        "artifacts": artifact_identities,
        "report_pair_fingerprint": report_pair_fingerprint,
        "generation_authorized": False,
    }
    if _canonical_bytes(
        _json_value(identity, "durable_scientific_pair_receipt.identity")
    ) != _canonical_bytes(expected_identity):
        raise IntegrityError(
            "durable scientific pair receipt differs from reconstructed lineage"
        )
    return reference


def _checked_prefix(value: object) -> Any:
    # Local import avoids a module cycle: the handler module itself imports
    # artifact types and exposes the prefix minting boundary.
    from prpg.model.g3_science_handlers import (
        CanonicalG3ArtifactPublicationPrefix,
        _require_artifact_ready_prefix,
    )

    if not isinstance(value, CanonicalG3ArtifactPublicationPrefix):
        raise ModelError("canonical artifact publication requires first-25 prefix")
    _require_artifact_ready_prefix(value)
    return value


def _role_inputs(prefix: Any, role: PairRole) -> _RoleInputs:
    from prpg.model.g3_boundary_views import is_block_calibration_task_view

    task_id = f"{role}_block_calibration"
    view = prefix._handlers.as_mapping()[task_id]
    if (
        not is_block_calibration_task_view(view)
        or view.role != role
        or view.run_id != prefix.run_id
        or view.source.selected_k != prefix.selected_n_states
    ):
        raise ModelError("canonical artifact role lacks exact block parent")
    source = view.source
    monthly_block = registered_block_replay_source(source.monthly)
    daily_block = registered_block_replay_source(source.daily)
    monthly_kernel = kernel_calibration_replay_source(source.monthly_kernel)
    daily_kernel = kernel_calibration_replay_source(source.daily_kernel)
    if not isinstance(monthly_kernel.model, GaussianHMMFit) or not isinstance(
        daily_kernel.model, GaussianHMMFit
    ):
        raise ModelError("canonical artifact kernel parents must retain fitted HMMs")
    fit = monthly_kernel.model
    fit_fingerprint = gaussian_hmm_fit_fingerprint(fit)
    if not isinstance(monthly_block.model, GaussianHMMFit) or not isinstance(
        daily_block.model, GaussianHMMFit
    ):
        raise ModelError("canonical artifact block parents must retain fitted HMMs")
    if (
        gaussian_hmm_fit_fingerprint(daily_kernel.model) != fit_fingerprint
        or gaussian_hmm_fit_fingerprint(monthly_block.model) != fit_fingerprint
        or gaussian_hmm_fit_fingerprint(daily_block.model) != fit_fingerprint
        or monthly_kernel.identity.parent_model_fingerprint != fit_fingerprint
        or daily_kernel.identity.parent_model_fingerprint != fit_fingerprint
        or monthly_block.core_identity.parent_model_fingerprint != fit_fingerprint
        or daily_block.core_identity.parent_model_fingerprint != fit_fingerprint
    ):
        raise ModelError("canonical artifact role sources use different fitted models")
    for kernel_source, block_source, frequency in (
        (monthly_kernel, monthly_block, "monthly"),
        (daily_kernel, daily_block, "daily"),
    ):
        if (
            not np.array_equal(
                kernel_source.historical_states,
                block_source.historical_source_states,
            )
            or not np.array_equal(
                kernel_source.continuation_links,
                block_source.source_continuation_links,
            )
            or kernel_source.identity.artifact != role
            or kernel_source.identity.frequency != frequency
        ):
            raise ModelError("canonical artifact block/kernel live sources disagree")
    if not np.array_equal(fit.decoded_states, monthly_kernel.historical_states):
        raise ModelError("canonical artifact monthly states differ from fitted path")

    monthly_continuation = _continuation_mask(monthly_kernel.continuation_links)
    daily_continuation = _continuation_mask(daily_kernel.continuation_links)
    pools = DecodedReturnPools(
        monthly_kernel.historical_states,
        daily_kernel.historical_states,
        summarize_state_path(
            monthly_kernel.historical_states,
            fit.n_states,
            lengths=source.monthly.execution.geometry.target_segment_lengths,
        ),
        summarize_state_path(
            daily_kernel.historical_states,
            fit.n_states,
            lengths=source.daily.execution.geometry.target_segment_lengths,
        ),
    )
    monthly_reference = fit_dependence_reference(
        monthly_kernel.historical_returns,
        monthly_continuation,
        frequency="monthly",
    )
    daily_reference = fit_dependence_reference(
        daily_kernel.historical_returns,
        daily_continuation,
        frequency="daily",
    )
    role_source_fingerprint = scientific_materialization_fingerprint(
        schema_id="canonical_scientific_artifact_role_source",
        metadata={
            "role": role,
            "run_id": prefix.run_id,
            "selected_k": prefix.selected_n_states,
            "block_view_fingerprint": view.view_fingerprint,
            "monthly_block_core_fingerprint": (
                monthly_block.core_identity.core_fingerprint
            ),
            "daily_block_core_fingerprint": daily_block.core_identity.core_fingerprint,
            "monthly_kernel_execution_fingerprint": (
                monthly_kernel.identity.execution_fingerprint
            ),
            "daily_kernel_execution_fingerprint": (
                daily_kernel.identity.execution_fingerprint
            ),
            "fit_fingerprint": fit_fingerprint,
            "pool_fingerprint": decoded_return_pools_fingerprint(pools),
        },
        arrays={
            "monthly_returns": monthly_kernel.historical_returns,
            "monthly_states": monthly_kernel.historical_states,
            "monthly_continuation": monthly_continuation,
            "monthly_pc1_loadings": monthly_reference.pc1_loadings,
            "daily_returns": daily_kernel.historical_returns,
            "daily_states": daily_kernel.historical_states,
            "daily_continuation": daily_continuation,
            "daily_pc1_loadings": daily_reference.pc1_loadings,
        },
    )
    # Reverify registered parents after every derived transformation.
    kernel_calibration_execution_identity(source.monthly_kernel)
    kernel_calibration_execution_identity(source.daily_kernel)
    registered_block_replay_source(source.monthly)
    registered_block_replay_source(source.daily)
    return _RoleInputs(
        role,
        view,
        fit,
        pools,
        monthly_continuation,
        daily_continuation,
        monthly_reference,
        daily_reference,
        source.monthly_kernel,
        source.daily_kernel,
        role_source_fingerprint,
    )


def _closed_reports(
    prefix: Any,
    inputs: _RoleInputs,
) -> Mapping[str, ScientificReportPayload]:
    result: dict[str, ScientificReportPayload] = {}
    for path, document_type in required_scientific_documents(inputs.role).items():
        task_ids = scientific_report_task_ids(document_type, inputs.role)
        families = scientific_report_planned_look_families(document_type, inputs.role)
        payload = scientific_report_payload(
            document_type=document_type,
            role=inputs.role,
            task_result_fingerprints={
                task_id: prefix.task_result_fingerprints[task_id]
                for task_id in task_ids
            },
            checkpoint_fingerprints={
                task_id: prefix.checkpoint_fingerprints[task_id] for task_id in task_ids
            },
            planned_look_fingerprints={
                family_id: prefix.planned_look_fingerprints[family_id]
                for family_id in families
            },
            metrics=_closed_report_metrics(
                prefix,
                document_type=document_type,
                role=inputs.role,
            ),
        )
        result[path] = ScientificReportPayload(
            document_type=document_type,
            counts=_canonical_report_counts(document_type, inputs),
            payload=payload,
        )
    return MappingProxyType(result)


def _closed_report_metrics(
    prefix: Any,
    *,
    document_type: str,
    role: PairRole,
) -> Mapping[str, Any]:
    from prpg.model.g3_assembly import derive_g3_gate_decision

    entries: list[dict[str, Any]] = []
    sources = prefix._handlers.as_mapping()
    for task_id in scientific_report_task_ids(document_type, role):
        source = sources[task_id]
        decision = derive_g3_gate_decision(task_id, source)
        if not decision.passed:
            raise ModelError("a failed first-25 decision cannot produce a report")
        evidence = _json_value(decision.evidence, f"{task_id}.evidence")
        summary = _json_value(decision.summary, f"{task_id}.summary")
        source_view_fingerprint = _closed_report_source_fingerprint(task_id, source)
        decision_identity = {
            "gate_id": task_id,
            "passed": True,
            "evidence": evidence,
            "summary": summary,
        }
        entries.append(
            {
                **decision_identity,
                "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
                "source_view_fingerprint": source_view_fingerprint,
                "decision_fingerprint": _sha256_bytes(
                    _canonical_bytes(decision_identity)
                ),
            }
        )
    set_fingerprint = _sha256_bytes(
        _canonical_bytes(
            {
                "schema_id": "prpg-g3-report-source-decision-set-v1",
                "source_decisions": entries,
            }
        )
    )
    return MappingProxyType(
        {
            "schema_id": CLOSED_REPORT_METRICS_SCHEMA_ID,
            "schema_version": CLOSED_REPORT_METRICS_SCHEMA_VERSION,
            "document_type": document_type,
            "artifact_role": role,
            "selected_k": prefix.selected_n_states,
            "source_decisions": entries,
            "source_decision_set_fingerprint": set_fingerprint,
        }
    )


def _closed_report_source_fingerprint(task_id: str, source: object) -> str:
    """Return one exact sealed task source's report-lineage fingerprint.

    Most G3 task sources expose a ``view_fingerprint``.  Adequacy cell and
    Holm sources instead expose their equivalent sealed ``content_fingerprint``.
    Resolve that representational difference only after the source's code-owned
    type guard has reverified process authority and content identity.
    """

    from prpg.model.adequacy_gate import (
        is_adequacy_cell_task_view,
        is_four_cell_adequacy_holm_view,
    )
    from prpg.model.dependence_gate_execution import (
        is_registered_dependence_task_view,
        is_registered_four_cell_dependence_holm_view,
    )
    from prpg.model.g3_boundary_views import (
        is_block_calibration_task_view,
        is_bridge_driver_task_view,
        is_input_integrity_task_view,
    )
    from prpg.model.g3_calibration_views import (
        is_candidate_selection_task_view,
        is_candidate_viability_task_view,
        is_macro_stability_task_view,
        is_production_fit_same_k_view,
        is_sealed_holdout_veto_view,
    )

    if is_adequacy_cell_task_view(source) or is_four_cell_adequacy_holm_view(source):
        return _task_bound_source_fingerprint(
            task_id=task_id,
            source_task_id=source.task_id,
            fingerprint=source.content_fingerprint,
        )
    if (
        is_input_integrity_task_view(source)
        or is_candidate_viability_task_view(source)
        or is_candidate_selection_task_view(source)
        or is_sealed_holdout_veto_view(source)
        or is_macro_stability_task_view(source)
        or is_production_fit_same_k_view(source)
        or is_block_calibration_task_view(source)
        or is_registered_dependence_task_view(source)
        or is_registered_four_cell_dependence_holm_view(source)
        or is_bridge_driver_task_view(source)
    ):
        return _task_bound_source_fingerprint(
            task_id=task_id,
            source_task_id=source.task_id,
            fingerprint=source.view_fingerprint,
        )
    raise ModelError(
        "closed report source lacks exact sealed task-source authority",
        details={"task_id": task_id, "source_type": type(source).__name__},
    )


def _task_bound_source_fingerprint(
    *,
    task_id: str,
    source_task_id: object,
    fingerprint: object,
) -> str:
    if source_task_id != task_id:
        raise ModelError(
            "closed report source is bound to a different task",
            details={"expected_task_id": task_id, "actual_task_id": source_task_id},
        )
    return _required_sha(
        fingerprint,
        f"{task_id} source view fingerprint",
        ModelError,
    )


def _canonical_report_counts(
    document_type: str,
    inputs: _RoleInputs,
) -> Mapping[str, int]:
    rows = 193 if inputs.role == "design" else 217
    fixed: dict[str, dict[str, int]] = {
        "model_fit": {"observations": rows, "features": 4, "ordinary_restarts": 50},
        "macro_stability": {"attempts": 100, "minimum_successes": 90},
        "latent_correctness": {
            "chains": 10_000,
            "measured_months": 600,
            "extension_months": 2_000,
            "cluster_bootstrap_replicates": 10_000,
            "cluster_resample_size": 10_000,
        },
        "refitted_adequacy": {"attempts": 1_000, "minimum_successes": 950},
        "transition_support": {"attempts": 100},
        "block_calibration": {"monthly_histories": 10_000, "daily_histories": 10_000},
        "fragmentation": {"monthly_histories": 10_000, "daily_histories": 10_000},
        "dependence": {"monthly_histories": 10_000, "daily_histories": 10_000},
        "adequacy_holm": {"cells": 4},
        "dependence_holm": {"cells": 4},
        "design_production_bridge": {
            "resource_benchmark_pairs": 100,
            "tier_a_pairs_per_dgp": 250,
            "tier_b_maximum_pairs_per_dgp": 10_000,
            "accelerator_audit_pairs_per_dgp": 100,
            "dgp_families": 2,
        },
        "candidate_selection": {
            "candidates": 4,
            "rolling_folds": 6,
            "rolling_scores_per_candidate": 72,
            "rolling_bootstrap_replicates": 10_000,
            "ordinary_restarts_per_fit": 50,
        },
        "sealed_holdout_veto": {"observations": 24, "segments": 2},
        "production_fit_same_k": {"observations": 217, "ordinary_restarts": 50},
    }
    if document_type == "kernel_calibration":
        return MappingProxyType(
            {
                "monthly_windows": inputs.monthly_kernel.stopped_windows,
                "daily_windows": inputs.daily_kernel.stopped_windows,
                "monthly_verification_windows": cast(
                    Any, inputs.monthly_kernel.verification
                ).windows,
                "daily_verification_windows": cast(
                    Any, inputs.daily_kernel.verification
                ).windows,
                "planned_looks": len(inputs.monthly_kernel.planned_looks),
            }
        )
    try:
        return MappingProxyType(fixed[document_type])
    except KeyError as error:
        raise ModelError("closed report type has no code-owned count schema") from error


def _pair_identity(
    *,
    design: LoadedScientificModelArtifact,
    production: LoadedScientificModelArtifact,
    design_draft: CanonicalScientificModelArtifactDraft,
    production_draft: CanonicalScientificModelArtifactDraft,
) -> dict[str, Any]:
    prefix = _checked_prefix(design_draft._prefix)
    if production_draft._prefix is not prefix:
        raise ModelError("scientific pair identity uses different prefixes")
    artifact_identities = {
        "design": _artifact_identity(design, design_draft),
        "production": _artifact_identity(production, production_draft),
    }
    report_pair_fingerprint = _sha256_bytes(
        _canonical_bytes(
            {
                "schema_id": "prpg-g3-scientific-report-pair-v1",
                "design": design_draft.report_set_fingerprint,
                "production": production_draft.report_set_fingerprint,
            }
        )
    )
    return {
        "schema_id": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_PAIR_SCHEMA_VERSION,
        "run_id": prefix.run_id,
        "selected_k": prefix.selected_n_states,
        "prefix_fingerprint": prefix.prefix_fingerprint,
        "binding": prefix.binding.as_dict(),
        "provenance": prefix.provenance.as_dict(),
        "task_result_fingerprints": dict(prefix.task_result_fingerprints),
        "checkpoint_fingerprints": dict(prefix.checkpoint_fingerprints),
        "planned_look_fingerprints": {
            key: list(value) for key, value in prefix.planned_look_fingerprints.items()
        },
        "actual_bridge_parent_view_fingerprint": (
            design_draft.actual_bridge_parent_view_fingerprint
        ),
        "candidate_policy_sets": {
            "design": design_draft.candidate_policy_set.as_dict(),
            "production": production_draft.candidate_policy_set.as_dict(),
        },
        "artifacts": artifact_identities,
        "report_pair_fingerprint": report_pair_fingerprint,
        "generation_authorized": False,
    }


def _artifact_identity(
    artifact: LoadedScientificModelArtifact,
    draft: CanonicalScientificModelArtifactDraft,
) -> dict[str, Any]:
    return {
        "role": draft.role,
        "artifact_fingerprint": artifact.reference.fingerprint,
        "core_model_fingerprint": artifact.core_model_fingerprint,
        "parameter_set_fingerprint": artifact.parameter_set_fingerprint,
        "role_source_fingerprint": draft.role_source_fingerprint,
        "report_set_fingerprint": draft.report_set_fingerprint,
        "candidate_policy_set_fingerprint": _sha256_bytes(
            _canonical_bytes(draft.candidate_policy_set.as_dict())
        ),
    }


def _verify_canonical_draft(
    value: CanonicalScientificModelArtifactDraft,
    *,
    expected_role: PairRole,
) -> None:
    if (
        not isinstance(value, CanonicalScientificModelArtifactDraft)
        or getattr(value, "_seal", None) is not _DRAFT_CAPABILITY
        or _MINTED_DRAFTS.get(id(value)) is not value
    ):
        raise ModelError("canonical scientific artifact draft lacks producer authority")
    prefix = _checked_prefix(value._prefix)
    expected = build_canonical_scientific_model_artifact_draft(
        prefix, role=expected_role
    )
    if (
        value.role != expected_role
        or value.selected_k != expected.selected_k
        or value.run_id != expected.run_id
        or value.prefix_fingerprint != expected.prefix_fingerprint
        or value.actual_bridge_parent_view_fingerprint
        != expected.actual_bridge_parent_view_fingerprint
        or value.role_source_fingerprint != expected.role_source_fingerprint
        or value.report_set_fingerprint != expected.report_set_fingerprint
        or value.candidate_policy_set != expected.candidate_policy_set
        or not _drafts_equal(value.draft, expected.draft)
    ):
        raise ModelError("canonical scientific artifact draft changed after minting")


def _verify_loaded_against_draft(
    loaded: LoadedScientificModelArtifact,
    expected: CanonicalScientificModelArtifactDraft,
) -> None:
    if not is_verified_scientific_model_artifact(loaded):
        raise ModelError("scientific pair requires typed loaded artifacts")
    draft = expected.draft
    if (
        loaded.role != expected.role
        or loaded.selected_k != expected.selected_k
        or loaded.binding != draft.binding
        or loaded.provenance != draft.provenance
        or loaded.generation_policy != draft.generation_policy
        or set(loaded.arrays) != set(draft.arrays)
        or any(
            not np.array_equal(loaded.arrays[name], np.asarray(draft.arrays[name]))
            for name in loaded.arrays
        )
        or set(loaded.reports) != set(draft.reports)
    ):
        raise IntegrityError("loaded scientific artifact differs from canonical draft")
    for path, supplied in draft.reports.items():
        envelope = loaded.reports[path]
        if (
            envelope["document_type"] != supplied.document_type
            or dict(envelope["counts"]) != dict(supplied.counts)
            or _canonical_bytes(envelope["payload"])
            != _canonical_bytes(supplied.payload)
            or _canonical_bytes(envelope["generation_policy"])
            != _canonical_bytes(draft.generation_policy.as_dict())
        ):
            raise IntegrityError(
                "loaded scientific report differs from closed producer"
            )


def _drafts_equal(
    left: ScientificModelArtifactDraft,
    right: ScientificModelArtifactDraft,
) -> bool:
    return (
        left.role == right.role
        and left.selected_k == right.selected_k
        and left.binding == right.binding
        and left.provenance == right.provenance
        and left.generation_policy == right.generation_policy
        and set(left.arrays) == set(right.arrays)
        and all(
            np.array_equal(
                np.asarray(left.arrays[name]), np.asarray(right.arrays[name])
            )
            for name in left.arrays
        )
        and _report_set_fingerprint(left.reports)
        == _report_set_fingerprint(right.reports)
    )


def _report_set_fingerprint(
    reports: Mapping[str, ScientificReportPayload],
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_id": "prpg-g3-closed-scientific-report-set-v1",
                "reports": {
                    path: {
                        "document_type": report.document_type,
                        "counts": dict(report.counts),
                        "payload": report.payload,
                    }
                    for path, report in sorted(reports.items())
                },
            }
        )
    )


def _loaded_report_set_fingerprint(
    artifact: LoadedScientificModelArtifact,
) -> str:
    reports = {
        path: ScientificReportPayload(
            document_type=str(envelope["document_type"]),
            counts=cast(Mapping[str, int], envelope["counts"]),
            payload=cast(Mapping[str, Any], envelope["payload"]),
        )
        for path, envelope in artifact.reports.items()
    }
    return _report_set_fingerprint(reports)


def _verify_closed_reports_from_durable_decisions(
    artifact: LoadedScientificModelArtifact,
    *,
    task_result_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    planned_look_fingerprints: Mapping[str, tuple[str, ...]],
    decision_documents: Mapping[str, Mapping[str, Any]],
) -> None:
    for path, document_type in required_scientific_documents(artifact.role).items():
        report = artifact.reports[path]
        payload = report["payload"]
        if not isinstance(payload, Mapping):
            raise IntegrityError("durable closed report payload is invalid")
        task_ids = scientific_report_task_ids(document_type, artifact.role)
        families = scientific_report_planned_look_families(document_type, artifact.role)
        if (
            _canonical_bytes(payload.get("task_result_fingerprints"))
            != _canonical_bytes(
                {task_id: task_result_fingerprints[task_id] for task_id in task_ids}
            )
            or _canonical_bytes(payload.get("checkpoint_fingerprints"))
            != _canonical_bytes(
                {task_id: checkpoint_fingerprints[task_id] for task_id in task_ids}
            )
            or _canonical_bytes(payload.get("planned_look_fingerprints"))
            != _canonical_bytes(
                {
                    family_id: list(planned_look_fingerprints[family_id])
                    for family_id in families
                }
            )
        ):
            raise IntegrityError("durable closed report receipt bindings changed")
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "schema_id",
            "schema_version",
            "document_type",
            "artifact_role",
            "selected_k",
            "source_decisions",
            "source_decision_set_fingerprint",
        }:
            raise IntegrityError("durable closed report metric schema is invalid")
        if (
            metrics["schema_id"] != CLOSED_REPORT_METRICS_SCHEMA_ID
            or metrics["schema_version"] != CLOSED_REPORT_METRICS_SCHEMA_VERSION
            or metrics["document_type"] != document_type
            or metrics["artifact_role"] != artifact.role
            or metrics["selected_k"] != artifact.selected_k
        ):
            raise IntegrityError("durable closed report metric identity is invalid")
        entries = metrics["source_decisions"]
        if not isinstance(entries, Sequence) or len(entries) != len(task_ids):
            raise IntegrityError("durable closed report source decisions are invalid")
        normalized_entries: list[object] = []
        for task_id, entry in zip(task_ids, entries, strict=True):
            decision = decision_documents.get(task_id)
            if not isinstance(entry, Mapping) or not isinstance(decision, Mapping):
                raise IntegrityError("durable closed report decision is missing")
            expected_identity = {
                "gate_id": task_id,
                "passed": True,
                "evidence": decision.get("evidence"),
                "summary": decision.get("summary"),
            }
            if (
                set(entry)
                != {
                    "gate_id",
                    "passed",
                    "evidence",
                    "summary",
                    "source_type",
                    "source_view_fingerprint",
                    "decision_fingerprint",
                }
                or decision.get("gate_id") != task_id
                or decision.get("passed") is not True
                or entry.get("gate_id") != task_id
                or entry.get("passed") is not True
                or entry.get("source_type") != decision.get("source_type")
                or _canonical_bytes(entry.get("evidence"))
                != _canonical_bytes(decision.get("evidence"))
                or _canonical_bytes(entry.get("summary"))
                != _canonical_bytes(decision.get("summary"))
                or entry.get("decision_fingerprint")
                != _sha256_bytes(_canonical_bytes(expected_identity))
                or not isinstance(entry.get("source_view_fingerprint"), str)
                or _SHA256.fullmatch(str(entry.get("source_view_fingerprint"))) is None
            ):
                raise IntegrityError("durable closed report decision lineage changed")
            normalized_entries.append(dict(entry))
        expected_set = _sha256_bytes(
            _canonical_bytes(
                {
                    "schema_id": "prpg-g3-report-source-decision-set-v1",
                    "source_decisions": normalized_entries,
                }
            )
        )
        if metrics["source_decision_set_fingerprint"] != expected_set:
            raise IntegrityError("durable closed report decision set changed")


def _continuation_mask(links: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    source = np.asarray(links)
    if source.ndim != 1 or source.dtype.kind != "b":
        raise ModelError("canonical artifact continuation links are invalid")
    mask = np.empty(source.size + 1, dtype=np.bool_)
    mask[0] = False
    mask[1:] = source
    result = np.frombuffer(mask.tobytes(), dtype=np.bool_)
    result.setflags(write=False)
    return result


def _role(value: object) -> PairRole:
    if value not in {"design", "production"}:
        raise ModelError("scientific pair artifact role is invalid")
    return cast(PairRole, value)


def _json_value(value: object, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError(f"{path} contains a non-finite value")
        return 0.0 if value == 0.0 else value
    if isinstance(value, np.generic):
        return _json_value(value.item(), path)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path)
    if isinstance(value, Enum):
        return _json_value(value.value, path)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ModelError(f"{path} contains an invalid mapping key")
        return {key: _json_value(value[key], f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item, f"{path}[]") for item in value]
    raise ModelError(
        f"{path} contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _required_sha(
    value: object,
    label: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise error_type(f"{label} is invalid")
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _reject_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    for candidate in (current, *current.parents):
        if candidate.exists() and candidate.is_symlink():
            raise IntegrityError("scientific artifact pair path has a symlink ancestor")


def _read_stable_regular_file(path: Path) -> bytes:
    """Read one receipt through a no-follow descriptor with stability checks."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError(
            "scientific artifact pair receipt cannot be opened"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise IntegrityError(
                "scientific artifact pair receipt is not a single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise IntegrityError("scientific artifact pair receipt changed during read")
        try:
            current = path.lstat()
        except OSError as error:
            raise IntegrityError(
                "scientific artifact pair receipt path changed during read"
            ) from error
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
        ) != (after.st_dev, after.st_ino, after.st_mode, after.st_nlink):
            raise IntegrityError(
                "scientific artifact pair receipt path changed during read"
            )
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise IntegrityError(
                "scientific artifact pair receipt size changed during read"
            )
        return content
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
