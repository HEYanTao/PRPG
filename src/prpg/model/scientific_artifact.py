"""Typed, fail-closed scientific schema for canonical Phase-3 artifacts.

The lower-level :mod:`prpg.model.artifact` store guarantees byte integrity for
arbitrary, safe JSON/NPZ content.  This module adds the scientific contract:
the exact role-specific report and array allowlists, canonical geometry and
replicate counts, cross-document provenance bindings, and semantic array
checks needed before a stored object can be called a Phase-3 model artifact.

Publication derives all metadata and document envelopes itself.  A caller may
provide report payloads, but cannot choose their role, geometry, selected K,
cross-fingerprints, or count contract.  Loading repeats every check after the
generic store has verified the authenticated bytes.  Neither operation is a
G3 decision or a generation authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, TypeAlias, TypeGuard

import numpy as np
import numpy.typing as npt

from prpg.errors import IntegrityError, ModelError
from prpg.model.artifact import (
    ModelArtifactReference,
    ModelArtifactStore,
)
from prpg.model.calibration import DecodedReturnPools
from prpg.model.dependence import DependenceReference
from prpg.model.g3 import G3_CONTRACT_VERSION
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.hmm import GaussianHMMFit
from prpg.model.input import ASSET_COLUMNS, MACRO_COLUMNS
from prpg.model.kernel_execution import (
    KernelCalibrationExecution,
    kernel_calibration_execution_identity,
)
from prpg.model.scientific_policy import (
    AlphaPolicy,
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
    candidate_policy_set_from_kernel_results,
    candidate_policy_set_from_mapping,
    generation_policy_from_kernel_results,
    generation_policy_from_mapping,
)
from prpg.simulation.rng import RNG_CONTRACT_VERSION

SCIENTIFIC_ARTIFACT_SCHEMA_VERSION: Final = 3
SCIENTIFIC_ARTIFACT_SCHEMA_ID: Final = "prpg-phase3-scientific-model-v3"
SCIENTIFIC_REPORT_SCHEMA_VERSION: Final = 2
CANONICAL_SCIENTIFIC_VERSION: Final = 11
CANONICAL_KERNEL_LAMBDAS: Final = (0.25, 0.50, 0.75, 1.00)
CANONICAL_KERNEL_LOOKS: Final = (10_000, 20_000, 40_000, 80_000, 160_000, 320_000)

ArtifactRole: TypeAlias = Literal["design", "production"]
ScientificStoredPolicy: TypeAlias = (
    ScientificGenerationPolicy | ScientificCandidatePolicySet
)
ReportCountValue: TypeAlias = int
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_LOADED_ARTIFACT_SEAL = object()
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)

_DESIGN_MONTHLY_SEGMENTS = (193,)
_DESIGN_DAILY_SEGMENTS = (4_049,)
_PRODUCTION_MONTHLY_SEGMENTS = (210, 7)
_PRODUCTION_DAILY_SEGMENTS = (4_404, 143)

_COMMON_REPORTS: Final = MappingProxyType(
    {
        "reports/model-fit.json": "model_fit",
        "reports/macro-stability.json": "macro_stability",
        "reports/latent-correctness.json": "latent_correctness",
        "reports/refitted-adequacy.json": "refitted_adequacy",
        "reports/transition-support.json": "transition_support",
        "reports/block-calibration.json": "block_calibration",
        "reports/fragmentation.json": "fragmentation",
        "reports/dependence.json": "dependence",
        "reports/kernel-calibration.json": "kernel_calibration",
        "reports/adequacy-holm.json": "adequacy_holm",
        "reports/dependence-holm.json": "dependence_holm",
        "reports/design-production-bridge.json": "design_production_bridge",
    }
)
_DESIGN_ONLY_REPORTS: Final = MappingProxyType(
    {
        "reports/candidate-selection.json": "candidate_selection",
        "reports/sealed-holdout-veto.json": "sealed_holdout_veto",
    }
)
_PRODUCTION_ONLY_REPORTS: Final = MappingProxyType(
    {"reports/production-fit-same-k.json": "production_fit_same_k"}
)

_ROLE_REPORT_TASKS: Final[Mapping[str, Mapping[ArtifactRole, tuple[str, ...]]]] = (
    MappingProxyType(
        {
            "model_fit": {
                "design": ("design_candidate_viability", "design_predictive_selection"),
                "production": ("production_fit_same_k",),
            },
            "macro_stability": {
                "design": ("design_macro_stability",),
                "production": ("production_macro_stability",),
            },
            "latent_correctness": {
                "design": ("design_latent_correctness",),
                "production": ("production_latent_correctness",),
            },
            "refitted_adequacy": {
                "design": ("design_refitted_adequacy",),
                "production": ("production_refitted_adequacy",),
            },
            "transition_support": {
                "design": ("design_macro_stability",),
                "production": ("production_macro_stability",),
            },
            "block_calibration": {
                "design": ("design_block_calibration",),
                "production": ("production_block_calibration",),
            },
            "fragmentation": {
                "design": ("design_block_calibration", "design_fragmentation"),
                "production": (
                    "production_block_calibration",
                    "production_fragmentation",
                ),
            },
            "dependence": {
                "design": (
                    "design_block_calibration",
                    "design_fragmentation",
                    "design_dependence",
                ),
                "production": (
                    "production_block_calibration",
                    "production_fragmentation",
                    "production_dependence",
                ),
            },
            "kernel_calibration": {
                "design": ("design_block_calibration",),
                "production": ("production_block_calibration",),
            },
            "adequacy_holm": {
                "design": (
                    "design_latent_correctness",
                    "design_refitted_adequacy",
                    "production_latent_correctness",
                    "production_refitted_adequacy",
                    "four_cell_adequacy_holm",
                ),
                "production": (
                    "design_latent_correctness",
                    "design_refitted_adequacy",
                    "production_latent_correctness",
                    "production_refitted_adequacy",
                    "four_cell_adequacy_holm",
                ),
            },
            "dependence_holm": {
                "design": (
                    "design_dependence",
                    "production_dependence",
                    "four_cell_dependence_holm",
                ),
                "production": (
                    "design_dependence",
                    "production_dependence",
                    "four_cell_dependence_holm",
                ),
            },
            "design_production_bridge": {
                "design": (
                    "bridge_resource_preflight",
                    "bridge_tier_a_model_dgp",
                    "bridge_tier_a_nonparametric_dgp",
                    "bridge_tier_b_model_dgp",
                    "bridge_tier_b_nonparametric_dgp",
                    "actual_artifact_bridge",
                ),
                "production": (
                    "bridge_resource_preflight",
                    "bridge_tier_a_model_dgp",
                    "bridge_tier_a_nonparametric_dgp",
                    "bridge_tier_b_model_dgp",
                    "bridge_tier_b_nonparametric_dgp",
                    "actual_artifact_bridge",
                ),
            },
            "candidate_selection": {
                "design": ("design_candidate_viability", "design_predictive_selection"),
            },
            "sealed_holdout_veto": {"design": ("sealed_holdout_veto",)},
            "production_fit_same_k": {"production": ("production_fit_same_k",)},
        }
    )
)


def required_scientific_documents(role: ArtifactRole) -> Mapping[str, str]:
    """Return the immutable exact report-path/type allowlist for ``role``."""

    checked = _role(role, ModelError)
    additions = (
        _DESIGN_ONLY_REPORTS if checked == "design" else _PRODUCTION_ONLY_REPORTS
    )
    return MappingProxyType(dict(sorted({**_COMMON_REPORTS, **additions}.items())))


def scientific_report_task_ids(
    document_type: str,
    role: ArtifactRole,
) -> tuple[str, ...]:
    """Return the exact completed G3 tasks a report must reference."""

    checked_role = _role(role, ModelError)
    try:
        by_role = _ROLE_REPORT_TASKS[document_type]
        return by_role[checked_role]
    except (KeyError, TypeError) as error:
        raise ModelError(
            "scientific report type/role task binding is invalid"
        ) from error


def scientific_report_planned_look_families(
    document_type: str,
    role: ArtifactRole,
) -> tuple[str, ...]:
    """Return exact sequential-look families inherited by one report."""

    tasks = frozenset(scientific_report_task_ids(document_type, role))
    return tuple(
        item.family_id
        for item in G3_PLANNED_LOOK_REGISTRY
        if item.owner_task_id in tasks
    )


def scientific_report_payload(
    *,
    document_type: str,
    role: ArtifactRole,
    task_result_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    planned_look_fingerprints: Mapping[str, tuple[str, ...]],
    metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build one canonical passed-report payload from exact durable evidence."""

    provisional: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "task_result_fingerprints": dict(task_result_fingerprints),
        "checkpoint_fingerprints": dict(checkpoint_fingerprints),
        "planned_look_fingerprints": {
            key: list(value) for key, value in planned_look_fingerprints.items()
        },
        "metrics": dict(metrics),
    }
    normalized = _report_payload(
        provisional,
        document_type=document_type,
        role=role,
        error_type=ModelError,
        require_result_fingerprint=False,
    )
    result_fingerprint = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    return MappingProxyType({**normalized, "result_fingerprint": result_fingerprint})


_ARRAY_NAMES: Final = frozenset(
    {
        "hmm_start_probabilities",
        "hmm_transition_matrix",
        "hmm_state_means",
        "hmm_tied_covariance",
        "hmm_scaler_means",
        "hmm_scaler_standard_deviations",
        "canonical_to_original",
        "original_to_canonical",
        "monthly_source_states",
        "daily_source_states",
        "monthly_continuation",
        "daily_continuation",
        "monthly_state_counts",
        "daily_state_counts",
        "monthly_pc1_loadings",
        "daily_pc1_loadings",
        "monthly_kernel_lambdas",
        "daily_kernel_lambdas",
        "monthly_kernel_means",
        "daily_kernel_means",
        "monthly_kernel_half_widths",
        "daily_kernel_half_widths",
        "monthly_kernel_state_counts",
        "daily_kernel_state_counts",
        "monthly_kernel_offsets",
        "daily_kernel_offsets",
    }
)
_CORE_ARRAY_NAMES: Final = frozenset(
    {
        "hmm_start_probabilities",
        "hmm_transition_matrix",
        "hmm_state_means",
        "hmm_tied_covariance",
        "hmm_scaler_means",
        "hmm_scaler_standard_deviations",
        "canonical_to_original",
        "original_to_canonical",
        "monthly_source_states",
        "daily_source_states",
        "monthly_continuation",
        "daily_continuation",
        "monthly_state_counts",
        "daily_state_counts",
        "monthly_pc1_loadings",
        "daily_pc1_loadings",
    }
)


def required_scientific_arrays() -> frozenset[str]:
    """Return the exact v1 NPZ member-name allowlist."""

    return _ARRAY_NAMES


@dataclass(frozen=True, slots=True)
class ScientificArtifactGeometry:
    """Exact source-pool segment geometry for one artifact role."""

    monthly_segment_lengths: tuple[int, ...]
    daily_segment_lengths: tuple[int, ...]

    @property
    def monthly_rows(self) -> int:
        return sum(self.monthly_segment_lengths)

    @property
    def daily_rows(self) -> int:
        return sum(self.daily_segment_lengths)

    def as_dict(self) -> dict[str, object]:
        return {
            "monthly_segment_lengths": list(self.monthly_segment_lengths),
            "daily_segment_lengths": list(self.daily_segment_lengths),
            "monthly_rows": self.monthly_rows,
            "daily_rows": self.daily_rows,
        }

    @classmethod
    def for_role(cls, role: ArtifactRole) -> ScientificArtifactGeometry:
        checked = _role(role, ModelError)
        if checked == "design":
            return cls(_DESIGN_MONTHLY_SEGMENTS, _DESIGN_DAILY_SEGMENTS)
        return cls(_PRODUCTION_MONTHLY_SEGMENTS, _PRODUCTION_DAILY_SEGMENTS)


@dataclass(frozen=True, slots=True)
class ScientificArtifactBinding:
    """Identity shared exactly by metadata and every calibration report."""

    config_fingerprint: str
    processed_data_fingerprint: str
    calibration_input_fingerprint: str
    calibration_run_fingerprint: str
    preflight_fingerprint: str
    contract_version: str = G3_CONTRACT_VERSION
    task_registry_fingerprint: str = G3_TASK_REGISTRY_FINGERPRINT
    planned_look_registry_fingerprint: str = G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
    rng_contract_version: int = RNG_CONTRACT_VERSION
    scientific_version: int = CANONICAL_SCIENTIFIC_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "processed_data_fingerprint": self.processed_data_fingerprint,
            "calibration_input_fingerprint": self.calibration_input_fingerprint,
            "calibration_run_fingerprint": self.calibration_run_fingerprint,
            "preflight_fingerprint": self.preflight_fingerprint,
            "contract_version": self.contract_version,
            "task_registry_fingerprint": self.task_registry_fingerprint,
            "planned_look_registry_fingerprint": (
                self.planned_look_registry_fingerprint
            ),
            "rng_contract_version": self.rng_contract_version,
            "scientific_version": self.scientific_version,
        }


@dataclass(frozen=True, slots=True)
class ScientificArtifactProvenance:
    """Exact producer provenance; operational timestamps are intentionally absent."""

    source_code_fingerprint: str
    dependency_lock_fingerprint: str
    producer: str = "prpg-model-calibrate-v1"

    def as_dict(self) -> dict[str, str]:
        return {
            "producer": self.producer,
            "source_code_fingerprint": self.source_code_fingerprint,
            "dependency_lock_fingerprint": self.dependency_lock_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ScientificReportPayload:
    """Typed input for one report whose identity envelope is code-derived."""

    document_type: str
    counts: Mapping[str, ReportCountValue]
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScientificModelArtifactDraft:
    """Inputs accepted by the strict serializer."""

    role: ArtifactRole
    selected_k: int
    binding: ScientificArtifactBinding
    provenance: ScientificArtifactProvenance
    generation_policy: ScientificStoredPolicy
    arrays: Mapping[str, npt.ArrayLike]
    reports: Mapping[str, ScientificReportPayload]


@dataclass(frozen=True, slots=True, init=False)
class LoadedScientificModelArtifact:
    """Fully integrity- and schema-verified Phase-3 artifact.

    Scientific validity is retained as evidence in the reports.  It is not
    inferred by this loader and the object can never authorize generation.
    """

    generation_authorized: ClassVar[Literal[False]] = False
    verification_scope: ClassVar[Literal["integrity_and_schema"]] = (
        "integrity_and_schema"
    )

    reference: ModelArtifactReference
    role: ArtifactRole
    selected_k: int
    geometry: ScientificArtifactGeometry
    binding: ScientificArtifactBinding
    provenance: ScientificArtifactProvenance
    generation_policy: ScientificStoredPolicy
    arrays: Mapping[str, np.ndarray[Any, Any]]
    reports: Mapping[str, Mapping[str, Any]]
    core_model_fingerprint: str
    parameter_set_fingerprint: str
    _schema_seal: object = field(repr=False, compare=False)


def scientific_core_model_fingerprint(
    *,
    role: ArtifactRole,
    fit: GaussianHMMFit,
    pools: DecodedReturnPools,
    monthly_continuation: npt.ArrayLike,
    daily_continuation: npt.ArrayLike,
    monthly_dependence_reference: DependenceReference,
    daily_dependence_reference: DependenceReference,
) -> str:
    """Return the pre-kernel core fingerprint used by purpose-7 execution.

    This solves the otherwise circular identity problem: kernel executions
    bind to the HMM/scaler/pool/PC1 core, while the final outer artifact also
    binds the resulting kernel arrays and reports.
    """

    arrays, selected_k, geometry = _core_arrays_from_phase3_results(
        role=role,
        fit=fit,
        pools=pools,
        monthly_continuation=monthly_continuation,
        daily_continuation=daily_continuation,
        monthly_dependence_reference=monthly_dependence_reference,
        daily_dependence_reference=daily_dependence_reference,
    )
    normalized = _scientific_arrays(
        _with_placeholder_kernel_arrays(arrays, selected_k),
        selected_k=selected_k,
        geometry=geometry,
        error_type=ModelError,
    )
    return _array_set_fingerprint(normalized, _CORE_ARRAY_NAMES)


def draft_scientific_model_artifact_from_phase3_results(
    *,
    role: ArtifactRole,
    binding: ScientificArtifactBinding,
    provenance: ScientificArtifactProvenance,
    fit: GaussianHMMFit,
    pools: DecodedReturnPools,
    monthly_continuation: npt.ArrayLike,
    daily_continuation: npt.ArrayLike,
    monthly_dependence_reference: DependenceReference,
    daily_dependence_reference: DependenceReference,
    monthly_kernel: KernelCalibrationExecution,
    daily_kernel: KernelCalibrationExecution,
    alpha_policy: AlphaPolicy | None = None,
    selected_neutral_lambda: float | None = None,
    candidate_policy_set: ScientificCandidatePolicySet | None = None,
    reports: Mapping[str, ScientificReportPayload],
) -> ScientificModelArtifactDraft:
    """Build a strict draft directly from implemented Phase-3 result types.

    ``candidate_policy_set`` is the canonical G3 form: it records every
    calibrated alpha candidate without selecting one.  Supplying
    ``alpha_policy`` retains the legacy/report-only selected-policy form used
    by compatibility tests and by future post-G5 qualification code.
    """

    arrays, selected_k, geometry = _core_arrays_from_phase3_results(
        role=role,
        fit=fit,
        pools=pools,
        monthly_continuation=monthly_continuation,
        daily_continuation=daily_continuation,
        monthly_dependence_reference=monthly_dependence_reference,
        daily_dependence_reference=daily_dependence_reference,
    )
    normalized_core = _scientific_arrays(
        _with_placeholder_kernel_arrays(arrays, selected_k),
        selected_k=selected_k,
        geometry=geometry,
        error_type=ModelError,
    )
    core_fingerprint = _array_set_fingerprint(normalized_core, _CORE_ARRAY_NAMES)
    arrays = {name: normalized_core[name] for name in _CORE_ARRAY_NAMES}
    for expected_frequency, kernel in (
        ("monthly", monthly_kernel),
        ("daily", daily_kernel),
    ):
        if not isinstance(kernel, KernelCalibrationExecution):
            raise ModelError("scientific draft kernel result is not typed")
        try:
            kernel_identity = kernel_calibration_execution_identity(kernel)
        except ModelError as exc:
            raise ModelError(
                "scientific draft kernel lacks registered execution authority"
            ) from exc
        if (
            kernel_identity.artifact != role
            or kernel_identity.frequency != expected_frequency
            or kernel_identity.n_states != selected_k
            or kernel.artifact != kernel_identity.artifact
            or kernel.frequency != kernel_identity.frequency
            or kernel.n_states != kernel_identity.n_states
            or kernel.model_artifact_fingerprint != core_fingerprint
            or not kernel.passed
            or kernel.offset_grid is None
            or kernel.verification is None
            or not kernel.verification.passed
        ):
            raise ModelError(
                "scientific draft kernel role/K/core/result binding is invalid"
            )
        prefix = expected_frequency
        arrays[f"{prefix}_kernel_lambdas"] = kernel.offset_grid.lambdas
        arrays[f"{prefix}_kernel_means"] = kernel.final_precision.means
        arrays[f"{prefix}_kernel_half_widths"] = (
            kernel.final_precision.annualized_simultaneous_half_widths
        )
        arrays[f"{prefix}_kernel_state_counts"] = (
            kernel.final_precision.state_observation_counts
        )
        arrays[f"{prefix}_kernel_offsets"] = kernel.offset_grid.offsets
    normalized = _scientific_arrays(
        arrays,
        selected_k=selected_k,
        geometry=geometry,
        error_type=ModelError,
    )
    if candidate_policy_set is not None:
        if alpha_policy is not None or selected_neutral_lambda is not None:
            raise ModelError(
                "unqualified G3 candidate set cannot contain a selected alpha policy"
            )
        expected_policy_set = candidate_policy_set_from_kernel_results(
            monthly=monthly_kernel,
            daily=daily_kernel,
        )
        if candidate_policy_set != expected_policy_set:
            raise ModelError("scientific draft candidate policy set is inconsistent")
        generation_policy: ScientificStoredPolicy = expected_policy_set
    else:
        if alpha_policy is None:
            raise ModelError(
                "scientific draft requires an unqualified candidate set or policy"
            )
        generation_policy = generation_policy_from_kernel_results(
            monthly=monthly_kernel,
            daily=daily_kernel,
            alpha_policy=alpha_policy,
            selected_neutral_lambda=selected_neutral_lambda,
        )
    return ScientificModelArtifactDraft(
        role=role,
        selected_k=selected_k,
        binding=_binding(binding, ModelError),
        provenance=_provenance(provenance, ModelError),
        generation_policy=generation_policy,
        arrays=MappingProxyType(normalized),
        reports=reports,
    )


def publish_scientific_model_artifact(
    store: ModelArtifactStore,
    draft: ScientificModelArtifactDraft,
) -> LoadedScientificModelArtifact:
    """Validate, serialize, publish, and typed-load one scientific artifact."""

    if not isinstance(store, ModelArtifactStore):
        raise ModelError("scientific artifact publication requires ModelArtifactStore")
    if not isinstance(draft, ScientificModelArtifactDraft):
        raise ModelError("scientific artifact publication requires a typed draft")
    role = _role(draft.role, ModelError)
    selected_k = _selected_k(draft.selected_k, ModelError)
    binding = _binding(draft.binding, ModelError)
    provenance = _provenance(draft.provenance, ModelError)
    generation_policy = _generation_policy(draft.generation_policy, ModelError)
    geometry = ScientificArtifactGeometry.for_role(role)
    arrays = _scientific_arrays(
        draft.arrays,
        selected_k=selected_k,
        geometry=geometry,
        error_type=ModelError,
    )
    core_fingerprint = _array_set_fingerprint(arrays, _CORE_ARRAY_NAMES)
    parameter_fingerprint = _array_set_fingerprint(arrays, _ARRAY_NAMES)
    metadata = _metadata(
        role,
        selected_k,
        geometry,
        binding,
        generation_policy,
        core_fingerprint,
        parameter_fingerprint,
    )
    documents = _documents(
        draft.reports,
        role=role,
        selected_k=selected_k,
        geometry=geometry,
        binding=binding,
        generation_policy=generation_policy,
        core_model_fingerprint=core_fingerprint,
        parameter_set_fingerprint=parameter_fingerprint,
        error_type=ModelError,
    )
    reference = store.create(
        artifact_role=role,
        data_fingerprint=binding.processed_data_fingerprint,
        model_metadata=metadata,
        arrays=arrays,
        calibration_documents=documents,
        provenance=provenance.as_dict(),
    )
    return load_scientific_model_artifact(store, reference.fingerprint)


def load_scientific_model_artifact(
    store: ModelArtifactStore,
    fingerprint: str,
) -> LoadedScientificModelArtifact:
    """Verify generic integrity, then reconstruct and validate the typed schema."""

    if not isinstance(store, ModelArtifactStore):
        raise ModelError("scientific artifact loading requires ModelArtifactStore")
    reference = store.verify(fingerprint)
    identity = reference.manifest["identity"]
    role = _role(identity.get("artifact_role"), IntegrityError)
    if identity.get("data_fingerprint") is None:
        raise IntegrityError("scientific artifact data fingerprint is missing")
    metadata = identity.get("model_metadata")
    if not isinstance(metadata, Mapping):
        raise IntegrityError("scientific artifact metadata is missing")
    expected_metadata_keys = {
        "schema_id",
        "schema_version",
        "artifact_role",
        "selected_k",
        "feature_names",
        "asset_names",
        "geometry",
        "binding",
        "generation_policy",
        "core_model_fingerprint",
        "parameter_set_fingerprint",
    }
    _exact_keys(metadata, expected_metadata_keys, "scientific artifact metadata")
    if (
        metadata["schema_id"] != SCIENTIFIC_ARTIFACT_SCHEMA_ID
        or metadata["schema_version"] != SCIENTIFIC_ARTIFACT_SCHEMA_VERSION
    ):
        raise IntegrityError("scientific artifact schema is unsupported")
    if metadata["artifact_role"] != role:
        raise IntegrityError("scientific artifact role bindings disagree")
    selected_k = _selected_k(metadata["selected_k"], IntegrityError)
    if tuple(metadata["feature_names"]) != tuple(MACRO_COLUMNS):
        raise IntegrityError("scientific artifact feature order is invalid")
    if tuple(metadata["asset_names"]) != tuple(ASSET_COLUMNS):
        raise IntegrityError("scientific artifact asset order is invalid")
    geometry = _geometry(metadata["geometry"], role, IntegrityError)
    binding = _binding_from_mapping(metadata["binding"], IntegrityError)
    generation_policy = _generation_policy_from_artifact_mapping(
        metadata["generation_policy"], error_type=IntegrityError
    )
    if identity["data_fingerprint"] != binding.processed_data_fingerprint:
        raise IntegrityError("scientific artifact processed-data bindings disagree")
    provenance = _provenance_from_mapping(identity["provenance"], IntegrityError)
    arrays = _scientific_arrays(
        store.load_arrays(fingerprint),
        selected_k=selected_k,
        geometry=geometry,
        error_type=IntegrityError,
        require_exact_dtype=True,
    )
    core_fingerprint = _array_set_fingerprint(arrays, _CORE_ARRAY_NAMES)
    parameter_fingerprint = _array_set_fingerprint(arrays, _ARRAY_NAMES)
    if metadata["core_model_fingerprint"] != core_fingerprint:
        raise IntegrityError("scientific artifact core-model fingerprint is invalid")
    if metadata["parameter_set_fingerprint"] != parameter_fingerprint:
        raise IntegrityError("scientific artifact parameter-set fingerprint is invalid")

    report_paths = tuple(record["relative_path"] for record in identity["documents"])
    if set(report_paths) != set(required_scientific_documents(role)):
        raise IntegrityError(
            "scientific artifact report allowlist is invalid",
            details={
                "missing": sorted(
                    set(required_scientific_documents(role)) - set(report_paths)
                ),
                "unexpected": sorted(
                    set(report_paths) - set(required_scientific_documents(role))
                ),
            },
        )
    reports: dict[str, Mapping[str, Any]] = {}
    for path in sorted(report_paths):
        report = store.read_document(fingerprint, path)
        _validate_document_envelope(
            path,
            report,
            role=role,
            selected_k=selected_k,
            geometry=geometry,
            binding=binding,
            generation_policy=generation_policy,
            core_model_fingerprint=core_fingerprint,
            parameter_set_fingerprint=parameter_fingerprint,
            error_type=IntegrityError,
        )
        reports[path] = report
    loaded = object.__new__(LoadedScientificModelArtifact)
    for name, value in (
        ("reference", reference),
        ("role", role),
        ("selected_k", selected_k),
        ("geometry", geometry),
        ("binding", binding),
        ("provenance", provenance),
        ("generation_policy", generation_policy),
        ("arrays", MappingProxyType(arrays)),
        ("reports", MappingProxyType(reports)),
        ("core_model_fingerprint", core_fingerprint),
        ("parameter_set_fingerprint", parameter_fingerprint),
        ("_schema_seal", _LOADED_ARTIFACT_SEAL),
    ):
        object.__setattr__(loaded, name, value)
    return loaded


def is_verified_scientific_model_artifact(
    value: object,
) -> TypeGuard[LoadedScientificModelArtifact]:
    """Return whether ``value`` came from the typed integrity loader."""

    return (
        isinstance(value, LoadedScientificModelArtifact)
        and value._schema_seal is _LOADED_ARTIFACT_SEAL
    )


def verify_scientific_artifact_run_evidence(
    artifact: LoadedScientificModelArtifact,
    *,
    binding: ScientificArtifactBinding,
    provenance: ScientificArtifactProvenance,
    task_result_fingerprints: Mapping[str, str],
    checkpoint_fingerprints: Mapping[str, str],
    planned_look_fingerprints: Mapping[str, tuple[str, ...]],
) -> None:
    """Cross-check every report against one completed canonical G3 run."""

    if not is_verified_scientific_model_artifact(artifact):
        raise ModelError("scientific run-evidence verifier requires a loaded artifact")
    expected_binding = _binding(binding, ModelError)
    expected_provenance = _provenance(provenance, ModelError)
    if artifact.binding != expected_binding:
        raise IntegrityError("scientific artifact binding differs from completed run")
    if artifact.provenance != expected_provenance:
        raise IntegrityError("scientific artifact provenance differs from preflight")
    tasks = _fingerprint_mapping(
        task_result_fingerprints,
        expected_keys=frozenset(G3_GATE_ORDER),
        label="global task-result",
        error_type=IntegrityError,
    )
    checkpoints = _fingerprint_mapping(
        checkpoint_fingerprints,
        expected_keys=frozenset(G3_GATE_ORDER),
        label="global checkpoint",
        error_type=IntegrityError,
    )
    family_ids = tuple(item.family_id for item in G3_PLANNED_LOOK_REGISTRY)
    looks = _planned_look_fingerprint_mapping(
        planned_look_fingerprints,
        family_ids=family_ids,
        error_type=IntegrityError,
    )
    for path, document_type in required_scientific_documents(artifact.role).items():
        report = artifact.reports[path]
        payload = _report_payload(
            report["payload"],
            document_type=document_type,
            role=artifact.role,
            error_type=IntegrityError,
        )
        report_tasks = scientific_report_task_ids(document_type, artifact.role)
        if payload["task_result_fingerprints"] != {
            task_id: tasks[task_id] for task_id in report_tasks
        }:
            raise IntegrityError(
                "scientific report task-result bindings differ from completed run"
            )
        if payload["checkpoint_fingerprints"] != {
            task_id: checkpoints[task_id] for task_id in report_tasks
        }:
            raise IntegrityError(
                "scientific report checkpoint bindings differ from completed run"
            )
        report_families = scientific_report_planned_look_families(
            document_type, artifact.role
        )
        if payload["planned_look_fingerprints"] != {
            family_id: looks[family_id] for family_id in report_families
        }:
            raise IntegrityError(
                "scientific report planned-look bindings differ from completed run"
            )


def _metadata(
    role: ArtifactRole,
    selected_k: int,
    geometry: ScientificArtifactGeometry,
    binding: ScientificArtifactBinding,
    generation_policy: ScientificStoredPolicy,
    core_fingerprint: str,
    parameter_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_id": SCIENTIFIC_ARTIFACT_SCHEMA_ID,
        "schema_version": SCIENTIFIC_ARTIFACT_SCHEMA_VERSION,
        "artifact_role": role,
        "selected_k": selected_k,
        "feature_names": list(MACRO_COLUMNS),
        "asset_names": list(ASSET_COLUMNS),
        "geometry": geometry.as_dict(),
        "binding": binding.as_dict(),
        "generation_policy": generation_policy.as_dict(),
        "core_model_fingerprint": core_fingerprint,
        "parameter_set_fingerprint": parameter_fingerprint,
    }


def _documents(
    supplied: Mapping[str, ScientificReportPayload],
    *,
    role: ArtifactRole,
    selected_k: int,
    geometry: ScientificArtifactGeometry,
    binding: ScientificArtifactBinding,
    generation_policy: ScientificStoredPolicy,
    core_model_fingerprint: str,
    parameter_set_fingerprint: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(supplied, Mapping):
        raise error_type("scientific artifact reports must be a mapping")
    required = required_scientific_documents(role)
    if set(supplied) != set(required):
        raise error_type(
            "scientific artifact report allowlist is invalid",
            details={
                "missing": sorted(set(required) - set(supplied)),
                "unexpected": sorted(set(supplied) - set(required)),
            },
        )
    documents: dict[str, Mapping[str, Any]] = {}
    for path in sorted(required):
        value = supplied[path]
        if not isinstance(value, ScientificReportPayload):
            raise error_type("scientific artifact report is not typed")
        if value.document_type != required[path]:
            raise error_type("scientific artifact report type does not match its path")
        counts = _report_counts(value.document_type, value.counts, role, error_type)
        payload = _report_payload(
            value.payload,
            document_type=value.document_type,
            role=role,
            error_type=error_type,
        )
        envelope: dict[str, Any] = {
            "schema_version": SCIENTIFIC_REPORT_SCHEMA_VERSION,
            "document_type": value.document_type,
            "artifact_role": role,
            "selected_k": selected_k,
            "geometry": geometry.as_dict(),
            "binding": binding.as_dict(),
            "generation_policy": generation_policy.as_dict(),
            "core_model_fingerprint": core_model_fingerprint,
            "parameter_set_fingerprint": parameter_set_fingerprint,
            "counts": counts,
            "payload": payload,
        }
        documents[path] = envelope
    return documents


def _validate_document_envelope(
    path: str,
    value: Mapping[str, Any],
    *,
    role: ArtifactRole,
    selected_k: int,
    geometry: ScientificArtifactGeometry,
    binding: ScientificArtifactBinding,
    generation_policy: ScientificStoredPolicy,
    core_model_fingerprint: str,
    parameter_set_fingerprint: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "document_type",
            "artifact_role",
            "selected_k",
            "geometry",
            "binding",
            "generation_policy",
            "core_model_fingerprint",
            "parameter_set_fingerprint",
            "counts",
            "payload",
        },
        "scientific calibration report",
        error_type=error_type,
    )
    required = required_scientific_documents(role)
    if path not in required or value["document_type"] != required[path]:
        raise error_type("scientific calibration report path/type binding is invalid")
    if value["schema_version"] != SCIENTIFIC_REPORT_SCHEMA_VERSION:
        raise error_type("scientific calibration report schema is unsupported")
    if value["artifact_role"] != role or value["selected_k"] != selected_k:
        raise error_type("scientific calibration report role/K binding is invalid")
    report_geometry = _geometry(value["geometry"], role, error_type)
    if report_geometry != geometry:  # pragma: no cover - role fixes the geometry
        raise error_type("scientific calibration report geometry binding is invalid")
    report_binding = _binding_from_mapping(value["binding"], error_type)
    if report_binding != binding:
        raise error_type(
            "scientific calibration report cross-fingerprint binding is invalid"
        )
    report_generation_policy = _generation_policy_from_artifact_mapping(
        value["generation_policy"], error_type=error_type
    )
    if report_generation_policy != generation_policy:
        raise error_type(
            "scientific calibration report generation-policy binding is invalid"
        )
    if (
        value["core_model_fingerprint"] != core_model_fingerprint
        or value["parameter_set_fingerprint"] != parameter_set_fingerprint
    ):
        raise error_type("scientific calibration report parameter binding is invalid")
    _report_counts(str(value["document_type"]), value["counts"], role, error_type)
    _report_payload(
        value["payload"],
        document_type=str(value["document_type"]),
        role=role,
        error_type=error_type,
    )


def _report_payload(
    supplied: object,
    *,
    document_type: str,
    role: ArtifactRole,
    error_type: type[ModelError] | type[IntegrityError],
    require_result_fingerprint: bool = True,
) -> dict[str, Any]:
    if not isinstance(supplied, Mapping):
        raise error_type("scientific calibration report payload is invalid")
    expected_keys = {
        "schema_version",
        "status",
        "task_result_fingerprints",
        "checkpoint_fingerprints",
        "planned_look_fingerprints",
        "metrics",
    }
    if require_result_fingerprint:
        expected_keys.add("result_fingerprint")
    if set(supplied) != expected_keys:
        raise error_type("scientific calibration report payload fields are invalid")
    if supplied["schema_version"] != 1 or supplied["status"] != "passed":
        raise error_type("scientific calibration report payload status is invalid")
    task_ids = scientific_report_task_ids(document_type, role)
    tasks = _fingerprint_mapping(
        supplied["task_result_fingerprints"],
        expected_keys=frozenset(task_ids),
        label="task-result",
        error_type=error_type,
    )
    checkpoints = _fingerprint_mapping(
        supplied["checkpoint_fingerprints"],
        expected_keys=frozenset(task_ids),
        label="checkpoint",
        error_type=error_type,
    )
    family_ids = scientific_report_planned_look_families(document_type, role)
    looks = _planned_look_fingerprint_mapping(
        supplied["planned_look_fingerprints"],
        family_ids=family_ids,
        error_type=error_type,
    )
    metrics = _canonical_report_value(supplied["metrics"], "report.metrics", error_type)
    if not isinstance(metrics, dict) or not metrics:
        raise error_type("scientific calibration report metrics must be non-empty")
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "task_result_fingerprints": tasks,
        "checkpoint_fingerprints": checkpoints,
        "planned_look_fingerprints": looks,
        "metrics": metrics,
    }
    if require_result_fingerprint:
        result_fingerprint = supplied["result_fingerprint"]
        expected = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
        if result_fingerprint != expected:
            raise error_type(
                "scientific calibration report result fingerprint is inconsistent"
            )
        normalized["result_fingerprint"] = result_fingerprint
    return normalized


def _fingerprint_mapping(
    supplied: object,
    *,
    expected_keys: frozenset[str],
    label: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> dict[str, str]:
    if (
        not isinstance(supplied, Mapping)
        or set(supplied) != expected_keys
        or any(
            not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None
            for value in supplied.values()
        )
    ):
        raise error_type(f"scientific calibration report {label} bindings are invalid")
    return {key: str(supplied[key]) for key in sorted(expected_keys)}


def _planned_look_fingerprint_mapping(
    supplied: object,
    *,
    family_ids: tuple[str, ...],
    error_type: type[ModelError] | type[IntegrityError],
) -> dict[str, list[str]]:
    if not isinstance(supplied, Mapping) or set(supplied) != set(family_ids):
        raise error_type(
            "scientific calibration report planned-look families are invalid"
        )
    result: dict[str, list[str]] = {}
    look_specs = {item.family_id: item for item in G3_PLANNED_LOOK_REGISTRY}
    for family_id in family_ids:
        fingerprints = supplied[family_id]
        if (
            not isinstance(fingerprints, list | tuple)
            or not fingerprints
            or len(fingerprints) > len(look_specs[family_id].sample_counts)
            or any(
                not isinstance(item, str) or _FINGERPRINT.fullmatch(item) is None
                for item in fingerprints
            )
        ):
            raise error_type(
                "scientific calibration report planned-look fingerprints are invalid"
            )
        result[family_id] = list(fingerprints)
    return result


def _canonical_report_value(
    value: object,
    path: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{path} contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise error_type(f"{path} contains an invalid key")
            if any(marker in key.casefold() for marker in _SECRET_MARKERS):
                raise error_type(f"{path} contains a secret-like key")
            result[key] = _canonical_report_value(
                value[key], f"{path}.{key}", error_type
            )
        return result
    if isinstance(value, list | tuple):
        return [
            _canonical_report_value(item, f"{path}[]", error_type) for item in value
        ]
    raise error_type(f"{path} contains a non-JSON value")


def _canonical_json_bytes(value: object) -> bytes:
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


def _report_counts(
    document_type: str,
    supplied: object,
    role: ArtifactRole,
    error_type: type[ModelError] | type[IntegrityError],
) -> dict[str, int]:
    if not isinstance(supplied, Mapping) or any(
        not isinstance(key, str) or type(value) is not int
        for key, value in supplied.items()
    ):
        raise error_type("scientific calibration report counts are invalid")
    counts = {str(key): int(value) for key, value in supplied.items()}
    rows = 193 if role == "design" else 217
    fixed: dict[str, dict[str, int]] = {
        "model_fit": {
            "observations": rows,
            "features": 4,
            "ordinary_restarts": 50,
        },
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
        "block_calibration": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "fragmentation": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
        "dependence": {
            "monthly_histories": 10_000,
            "daily_histories": 10_000,
        },
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
        "production_fit_same_k": {
            "observations": 217,
            "ordinary_restarts": 50,
        },
    }
    if document_type == "kernel_calibration":
        expected_keys = {
            "monthly_windows",
            "daily_windows",
            "monthly_verification_windows",
            "daily_verification_windows",
            "planned_looks",
        }
        if set(counts) != expected_keys:
            raise error_type("kernel calibration report count fields are invalid")
        if (
            counts["monthly_windows"] not in CANONICAL_KERNEL_LOOKS
            or counts["daily_windows"] not in CANONICAL_KERNEL_LOOKS
            or counts["monthly_verification_windows"] != counts["monthly_windows"]
            or counts["daily_verification_windows"] != counts["daily_windows"]
            or counts["planned_looks"] != len(CANONICAL_KERNEL_LOOKS)
        ):
            raise error_type("kernel calibration report counts are noncanonical")
        return counts
    expected = fixed.get(document_type)
    if expected is None:
        raise error_type("scientific calibration report type is unregistered")
    if counts != expected:
        raise error_type(
            "scientific calibration report counts are noncanonical",
            details={
                "document_type": document_type,
                "expected": expected,
                "actual": counts,
            },
        )
    if document_type in _DESIGN_ONLY_REPORTS.values() and role != "design":
        raise error_type("design-only calibration report is in a production artifact")
    if document_type in _PRODUCTION_ONLY_REPORTS.values() and role != "production":
        raise error_type("production-only calibration report is in a design artifact")
    return counts


def _core_arrays_from_phase3_results(
    *,
    role: ArtifactRole,
    fit: GaussianHMMFit,
    pools: DecodedReturnPools,
    monthly_continuation: npt.ArrayLike,
    daily_continuation: npt.ArrayLike,
    monthly_dependence_reference: DependenceReference,
    daily_dependence_reference: DependenceReference,
) -> tuple[dict[str, npt.ArrayLike], int, ScientificArtifactGeometry]:
    checked_role = _role(role, ModelError)
    if not isinstance(fit, GaussianHMMFit):
        raise ModelError("scientific core requires GaussianHMMFit")
    if not isinstance(pools, DecodedReturnPools):
        raise ModelError("scientific core requires DecodedReturnPools")
    expected_scope = (
        "design_training" if checked_role == "design" else "production_full"
    )
    geometry = ScientificArtifactGeometry.for_role(checked_role)
    expected_lengths = geometry.monthly_segment_lengths
    if (
        fit.n_states not in {2, 3, 4, 5}
        or fit.n_features != 4
        or fit.n_observations != geometry.monthly_rows
        or fit.lengths != expected_lengths
        or fit.scaler_scope != expected_scope
    ):
        raise ModelError("scientific core HMM role/K/geometry binding is invalid")
    if (
        not isinstance(monthly_dependence_reference, DependenceReference)
        or monthly_dependence_reference.frequency != "monthly"
        or not isinstance(daily_dependence_reference, DependenceReference)
        or daily_dependence_reference.frequency != "daily"
    ):
        raise ModelError("scientific core dependence references are invalid")
    monthly_states = np.asarray(pools.monthly_states)
    daily_states = np.asarray(pools.daily_states)
    arrays: dict[str, npt.ArrayLike] = {
        "hmm_start_probabilities": fit.parameters.start_probabilities,
        "hmm_transition_matrix": fit.parameters.transition_matrix,
        "hmm_state_means": fit.parameters.means,
        "hmm_tied_covariance": fit.parameters.tied_covariance,
        "hmm_scaler_means": fit.scaler_means,
        "hmm_scaler_standard_deviations": fit.scaler_standard_deviations,
        "canonical_to_original": fit.canonical_to_original,
        "original_to_canonical": fit.original_to_canonical,
        "monthly_source_states": monthly_states,
        "daily_source_states": daily_states,
        "monthly_continuation": monthly_continuation,
        "daily_continuation": daily_continuation,
        "monthly_state_counts": np.bincount(
            monthly_states.astype(np.int64, copy=False), minlength=fit.n_states
        ),
        "daily_state_counts": np.bincount(
            daily_states.astype(np.int64, copy=False), minlength=fit.n_states
        ),
        "monthly_pc1_loadings": monthly_dependence_reference.pc1_loadings,
        "daily_pc1_loadings": daily_dependence_reference.pc1_loadings,
    }
    return arrays, fit.n_states, geometry


def _with_placeholder_kernel_arrays(
    core: Mapping[str, npt.ArrayLike], selected_k: int
) -> dict[str, npt.ArrayLike]:
    arrays = dict(core)
    for frequency in ("monthly", "daily"):
        arrays[f"{frequency}_kernel_lambdas"] = np.asarray(
            CANONICAL_KERNEL_LAMBDAS, dtype=np.float64
        )
        arrays[f"{frequency}_kernel_means"] = np.zeros(
            (selected_k, 3), dtype=np.float64
        )
        arrays[f"{frequency}_kernel_half_widths"] = np.zeros(
            (selected_k, 3), dtype=np.float64
        )
        arrays[f"{frequency}_kernel_state_counts"] = np.ones(selected_k, dtype=np.int64)
        arrays[f"{frequency}_kernel_offsets"] = np.zeros(
            (4, selected_k, 3), dtype=np.float64
        )
    return arrays


def _scientific_arrays(
    supplied: Mapping[str, npt.ArrayLike],
    *,
    selected_k: int,
    geometry: ScientificArtifactGeometry,
    error_type: type[ModelError] | type[IntegrityError],
    require_exact_dtype: bool = False,
) -> dict[str, np.ndarray[Any, Any]]:
    if not isinstance(supplied, Mapping) or set(supplied) != _ARRAY_NAMES:
        actual = set(supplied) if isinstance(supplied, Mapping) else set()
        raise error_type(
            "scientific artifact array allowlist is invalid",
            details={
                "missing": sorted(_ARRAY_NAMES - actual),
                "unexpected": sorted(actual - _ARRAY_NAMES),
            },
        )
    k = selected_k
    m = geometry.monthly_rows
    d = geometry.daily_rows
    shapes: dict[str, tuple[int, ...]] = {
        "hmm_start_probabilities": (k,),
        "hmm_transition_matrix": (k, k),
        "hmm_state_means": (k, 4),
        "hmm_tied_covariance": (4, 4),
        "hmm_scaler_means": (4,),
        "hmm_scaler_standard_deviations": (4,),
        "canonical_to_original": (k,),
        "original_to_canonical": (k,),
        "monthly_source_states": (m,),
        "daily_source_states": (d,),
        "monthly_continuation": (m,),
        "daily_continuation": (d,),
        "monthly_state_counts": (k,),
        "daily_state_counts": (k,),
        "monthly_pc1_loadings": (3,),
        "daily_pc1_loadings": (3,),
        "monthly_kernel_lambdas": (4,),
        "daily_kernel_lambdas": (4,),
        "monthly_kernel_means": (k, 3),
        "daily_kernel_means": (k, 3),
        "monthly_kernel_half_widths": (k, 3),
        "daily_kernel_half_widths": (k, 3),
        "monthly_kernel_state_counts": (k,),
        "daily_kernel_state_counts": (k,),
        "monthly_kernel_offsets": (4, k, 3),
        "daily_kernel_offsets": (4, k, 3),
    }
    integer_names = {
        "canonical_to_original",
        "original_to_canonical",
        "monthly_source_states",
        "daily_source_states",
        "monthly_state_counts",
        "daily_state_counts",
        "monthly_kernel_state_counts",
        "daily_kernel_state_counts",
    }
    boolean_names = {"monthly_continuation", "daily_continuation"}
    result: dict[str, np.ndarray[Any, Any]] = {}
    for name in sorted(_ARRAY_NAMES):
        try:
            array = np.asarray(supplied[name])
        except (TypeError, ValueError) as error:
            raise error_type(
                "scientific artifact array cannot be converted",
                details={"name": name, "error_type": type(error).__name__},
            ) from error
        if array.shape != shapes[name]:
            raise error_type(
                "scientific artifact array shape is invalid",
                details={"name": name, "expected": shapes[name], "actual": array.shape},
            )
        expected_dtype = np.dtype(
            np.bool_
            if name in boolean_names
            else np.int64
            if name in integer_names
            else np.float64
        )
        if require_exact_dtype and array.dtype != expected_dtype:
            raise error_type(
                "scientific artifact array dtype is invalid",
                details={
                    "name": name,
                    "expected": expected_dtype.str,
                    "actual": array.dtype.str,
                },
            )
        if name in boolean_names:
            if array.dtype.kind != "b":
                raise error_type("scientific continuation arrays must be boolean")
        elif name in integer_names:
            if array.dtype.kind not in {"i", "u"}:
                raise error_type("scientific integer arrays must have integer dtype")
            if (
                array.dtype.kind == "u"
                and array.size
                and int(np.max(array)) > np.iinfo(np.int64).max
            ):
                raise error_type("scientific integer array exceeds int64")
        elif array.dtype.kind not in {"f", "i", "u"}:
            raise error_type("scientific floating arrays must be numeric")
        normalized = np.ascontiguousarray(array.astype(expected_dtype, copy=False))
        if normalized.dtype.kind == "f" and not np.isfinite(normalized).all():
            raise error_type("scientific artifact array contains non-finite values")
        normalized.setflags(write=False)
        result[name] = normalized
    _validate_array_semantics(result, k, geometry, error_type)
    return result


def _validate_array_semantics(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    k: int,
    geometry: ScientificArtifactGeometry,
    error_type: type[ModelError] | type[IntegrityError],
) -> None:
    start = arrays["hmm_start_probabilities"]
    transition = arrays["hmm_transition_matrix"]
    if bool((start < 0.0).any()) or not math.isclose(
        float(start.sum()), 1.0, abs_tol=1e-12
    ):
        raise error_type("HMM start probabilities are not stochastic")
    if bool((transition < -1e-15).any()) or not np.allclose(
        transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise error_type("HMM transition matrix is not stochastic")
    covariance = arrays["hmm_tied_covariance"]
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise error_type("HMM tied covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if (
        float(eigenvalues[0]) < 1e-6 - 1e-12
        or float(eigenvalues[-1] / eigenvalues[0]) > 1e6 + 1e-6
    ):
        raise error_type("HMM tied covariance floor/condition contract failed")
    if bool((arrays["hmm_scaler_standard_deviations"] <= 0.0).any()):
        raise error_type("HMM scaler standard deviations must be positive")
    expected_permutation = np.arange(k, dtype=np.int64)
    c2o = arrays["canonical_to_original"]
    o2c = arrays["original_to_canonical"]
    if (
        not np.array_equal(np.sort(c2o), expected_permutation)
        or not np.array_equal(np.sort(o2c), expected_permutation)
        or not np.array_equal(o2c[c2o], expected_permutation)
    ):
        raise error_type("HMM canonical-state mappings are not inverse permutations")
    for frequency, segments in (
        ("monthly", geometry.monthly_segment_lengths),
        ("daily", geometry.daily_segment_lengths),
    ):
        states = arrays[f"{frequency}_source_states"]
        if bool((states < 0).any()) or bool((states >= k).any()):
            raise error_type("scientific source state is outside selected K")
        counts = arrays[f"{frequency}_state_counts"]
        expected_counts = np.bincount(states, minlength=k).astype(np.int64)
        if not np.array_equal(counts, expected_counts) or bool((counts <= 0).any()):
            raise error_type("scientific source-state counts are inconsistent")
        continuation = arrays[f"{frequency}_continuation"]
        expected_continuation = np.ones(len(states), dtype=np.bool_)
        start_index = 0
        for length in segments:
            expected_continuation[start_index] = False
            start_index += length
        if not np.array_equal(continuation, expected_continuation):
            raise error_type("scientific continuation geometry is inconsistent")
        loadings = arrays[f"{frequency}_pc1_loadings"]
        if (
            not math.isclose(float(np.linalg.norm(loadings)), 1.0, abs_tol=1e-12)
            or loadings[0] <= 1e-12
        ):
            raise error_type("scientific PC1 loading convention is invalid")
        lambdas = arrays[f"{frequency}_kernel_lambdas"]
        if not np.array_equal(lambdas, np.asarray(CANONICAL_KERNEL_LAMBDAS)):
            raise error_type("scientific kernel lambda grid is noncanonical")
        half_widths = arrays[f"{frequency}_kernel_half_widths"]
        if (
            bool((half_widths < 0.0).any())
            or float(np.max(half_widths)) > 0.001 + 1e-15
        ):
            raise error_type("scientific kernel precision gate failed")
        kernel_counts = arrays[f"{frequency}_kernel_state_counts"]
        if bool((kernel_counts <= 0).any()):
            raise error_type("scientific kernel state counts must be positive")


def _array_set_fingerprint(
    arrays: Mapping[str, np.ndarray[Any, Any]], names: frozenset[str]
) -> str:
    records: list[dict[str, object]] = []
    for name in sorted(names):
        array = np.ascontiguousarray(arrays[name])
        records.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    content = json.dumps(
        {"schema": "prpg-scientific-array-set-v1", "records": records},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _binding(
    value: object, error_type: type[ModelError] | type[IntegrityError]
) -> ScientificArtifactBinding:
    if not isinstance(value, ScientificArtifactBinding):
        raise error_type("scientific artifact binding is not typed")
    for name in (
        "config_fingerprint",
        "processed_data_fingerprint",
        "calibration_input_fingerprint",
        "calibration_run_fingerprint",
        "preflight_fingerprint",
        "task_registry_fingerprint",
        "planned_look_registry_fingerprint",
    ):
        _fingerprint(getattr(value, name), name, error_type)
    if (
        value.contract_version != G3_CONTRACT_VERSION
        or value.task_registry_fingerprint != G3_TASK_REGISTRY_FINGERPRINT
        or value.planned_look_registry_fingerprint
        != G3_PLANNED_LOOK_REGISTRY_FINGERPRINT
        or value.rng_contract_version != RNG_CONTRACT_VERSION
        or value.scientific_version != CANONICAL_SCIENTIFIC_VERSION
    ):
        raise error_type("scientific artifact contract binding is noncanonical")
    return value


def _binding_from_mapping(
    value: object, error_type: type[ModelError] | type[IntegrityError]
) -> ScientificArtifactBinding:
    if not isinstance(value, Mapping):
        raise error_type("scientific artifact binding is missing")
    expected = set(ScientificArtifactBinding.__dataclass_fields__)
    _exact_keys(value, expected, "scientific artifact binding", error_type=error_type)
    try:
        binding = ScientificArtifactBinding(**dict(value))
    except TypeError as error:
        raise error_type("scientific artifact binding fields are invalid") from error
    return _binding(binding, error_type)


def _generation_policy(
    value: object,
    error_type: type[ModelError] | type[IntegrityError],
) -> ScientificStoredPolicy:
    if isinstance(value, ScientificCandidatePolicySet):
        normalized_candidate = candidate_policy_set_from_mapping(
            value.as_dict(), error_type=error_type
        )
        if normalized_candidate != value:
            raise error_type("scientific artifact candidate policy set is noncanonical")
        return normalized_candidate
    if not isinstance(value, ScientificGenerationPolicy):
        raise error_type("scientific artifact generation policy is not typed")
    normalized = generation_policy_from_mapping(value.as_dict(), error_type=error_type)
    if normalized != value:  # pragma: no cover - exact mapping preserves valid values
        raise error_type("scientific artifact generation policy is not canonical")
    return normalized


def _generation_policy_from_artifact_mapping(
    value: object,
    *,
    error_type: type[ModelError] | type[IntegrityError],
) -> ScientificStoredPolicy:
    """Rehydrate the generic store's immutable JSON sequence representation."""

    if not isinstance(value, Mapping):
        return generation_policy_from_mapping(value, error_type=error_type)
    normalized = dict(value)
    if normalized.get("schema_id") is not None:
        candidates = normalized.get("candidate_policies")
        lambdas = normalized.get("neutral_lambda_candidates")
        if isinstance(candidates, tuple):
            normalized["candidate_policies"] = list(candidates)
        if isinstance(lambdas, tuple):
            normalized["neutral_lambda_candidates"] = list(lambdas)
        return candidate_policy_set_from_mapping(normalized, error_type=error_type)
    grid = normalized.get("neutral_lambda_grid")
    if isinstance(grid, tuple):
        normalized["neutral_lambda_grid"] = list(grid)
    return generation_policy_from_mapping(normalized, error_type=error_type)


def _provenance(
    value: object, error_type: type[ModelError] | type[IntegrityError]
) -> ScientificArtifactProvenance:
    if not isinstance(value, ScientificArtifactProvenance):
        raise error_type("scientific artifact provenance is not typed")
    if value.producer != "prpg-model-calibrate-v1":
        raise error_type("scientific artifact producer is invalid")
    _fingerprint(value.source_code_fingerprint, "source code", error_type)
    _fingerprint(value.dependency_lock_fingerprint, "dependency lock", error_type)
    return value


def _provenance_from_mapping(
    value: object, error_type: type[ModelError] | type[IntegrityError]
) -> ScientificArtifactProvenance:
    if not isinstance(value, Mapping):
        raise error_type("scientific artifact provenance is missing")
    _exact_keys(
        value,
        set(ScientificArtifactProvenance.__dataclass_fields__),
        "scientific artifact provenance",
        error_type=error_type,
    )
    try:
        provenance = ScientificArtifactProvenance(**dict(value))
    except TypeError as error:
        raise error_type("scientific artifact provenance fields are invalid") from error
    return _provenance(provenance, error_type)


def _geometry(
    value: object,
    role: ArtifactRole,
    error_type: type[ModelError] | type[IntegrityError],
) -> ScientificArtifactGeometry:
    if not isinstance(value, Mapping):
        raise error_type("scientific artifact geometry is missing")
    _exact_keys(
        value,
        {
            "monthly_segment_lengths",
            "daily_segment_lengths",
            "monthly_rows",
            "daily_rows",
        },
        "scientific artifact geometry",
        error_type=error_type,
    )
    expected = ScientificArtifactGeometry.for_role(role)
    monthly = value["monthly_segment_lengths"]
    daily = value["daily_segment_lengths"]
    if (
        not isinstance(monthly, list | tuple)
        or not isinstance(daily, list | tuple)
        or tuple(monthly) != expected.monthly_segment_lengths
        or tuple(daily) != expected.daily_segment_lengths
        or value["monthly_rows"] != expected.monthly_rows
        or value["daily_rows"] != expected.daily_rows
    ):
        raise error_type("scientific artifact geometry is noncanonical")
    return expected


def _role(
    value: object, error_type: type[ModelError] | type[IntegrityError]
) -> ArtifactRole:
    if value not in {"design", "production"}:
        raise error_type("scientific artifact role is invalid")
    return value  # type: ignore[return-value]


def _selected_k(
    value: object, error_type: type[ModelError] | type[IntegrityError]
) -> int:
    if type(value) is not int or value not in {2, 3, 4, 5}:
        raise error_type("scientific artifact selected K must be one of 2,3,4,5")
    return value


def _fingerprint(
    value: object,
    label: str,
    error_type: type[ModelError] | type[IntegrityError],
) -> None:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise error_type(f"scientific artifact {label} fingerprint is invalid")


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    *,
    error_type: type[ModelError] | type[IntegrityError] = IntegrityError,
) -> None:
    actual = set(value)
    if actual != expected:
        raise error_type(
            f"{label} fields are invalid",
            details={
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
            },
        )
