from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from prpg.config import PRPGConfig, load_config
from prpg.errors import GenerationError, IntegrityError, ModelError, ValidationError
from prpg.model.artifact import ModelArtifactReference
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.g3 import G3EvidenceBundle, build_g3_evidence_bundle, gate_record
from prpg.model.g3_evidence_store import (
    G3_EVIDENCE_SCHEMA_ID,
    G3_EVIDENCE_SCHEMA_VERSION,
    G3EvidenceReference,
    LoadedG3EvidenceAuthority,
)
from prpg.model.g3_registry import (
    G3_GATE_ORDER,
    G3_PLANNED_LOOK_REGISTRY,
    G3_PLANNED_LOOK_REGISTRY_FINGERPRINT,
    G3_TASK_REGISTRY_FINGERPRINT,
)
from prpg.model.g5_evidence_store import G5EvidenceStore
from prpg.model.g5_generation_authority import (
    _G5_NONCANONICAL_TEST_CAPABILITY,
    G5_DEVELOPMENT_HF_PATH_COUNT,
    G5_DEVELOPMENT_LF_PATH_COUNT,
    G5_OUTER_NULL_REPLICATES,
    G5_QUALIFICATION_HF_PATH_COUNT,
    G5_QUALIFICATION_LF_PATH_COUNT,
    G5DevelopmentPolicyDecision,
    G5QualificationGenerationAuthority,
    G5QualifiedProductionAuthority,
    _produce_g5_meta_validation_evidence_for_testing,
    _produce_g5_qualification_evidence_for_testing,
    _produce_g5_qualification_generation_authority_for_testing,
    _produce_g5_qualified_production_authority_for_testing,
    _produce_g5_threshold_evidence_for_testing,
    g5_policy_scientific_version,
    g5_policy_stream_fingerprint,
    is_g5_qualification_generation_authority,
    is_g5_qualified_production_authority,
    produce_g5_development_policy_selection,
    produce_g5_qualification_generation_authority,
    produce_g5_qualified_production_authority,
    produce_g5_threshold_evidence,
)
from prpg.model.input import (
    CalibrationInput,
    CalibrationProvenance,
    CalibrationSlice,
    SourceFileProvenance,
    SplitGeometry,
)
from prpg.model.lean_g5_generation_authority import (
    LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID,
    LEAN_G5_REQUIRED_DEFECTS,
    LeanG5ProductionAuthority,
    LeanG5ProductionAuthorityStore,
    _assessment_identity,
    build_lean_g5_verification_receipt,
    is_lean_g5_production_authority,
    mint_lean_g5_production_authority,
    verify_lean_g5_production_authority,
)
from prpg.model.lean_g5_generation_authority import (
    _fingerprint as _lean_authority_fingerprint,
)
from prpg.model.lean_g5_qualification import (
    LEAN_G5_QUALIFICATION_CONTRACT_ID,
    LeanG5QualificationPermit,
    build_lean_g5_qualification_permit,
    is_lean_g5_qualification_permit,
    verify_lean_g5_qualification_permit,
)
from prpg.model.scientific_artifact import (
    LoadedScientificModelArtifact,
    ScientificArtifactBinding,
    ScientificArtifactGeometry,
    ScientificArtifactProvenance,
)
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID,
    SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
)
from prpg.provenance import (
    G3_EXECUTION_CLOSURE_MANIFEST,
    G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT,
    G3_EXECUTION_CLOSURE_SCHEMA,
)
from prpg.simulation.aggregation import (
    aggregate_hf_weekly_log_returns,
    aggregate_lf_log_returns,
    is_sealed_hf_weekly_returns,
    is_sealed_lf_aggregated_returns,
)
from prpg.simulation.rng import Domain, Stage
from prpg.simulation.serial import (
    G5CandidateSerialBasePath,
    G5CandidateSerialGenerationInput,
    RestartReason,
    SerialGenerationInput,
    _build_g5_candidate_serial_generation_input_for_testing,
    _build_noncanonical_g5_serial_generation_input_for_testing,
    _build_rehearsal_serial_generation_input_for_testing,
    _generate_rehearsal_serial_base_path_for_testing,
    build_g4_reference_generation_input,
    build_g5_candidate_serial_generation_input,
    build_lean_g5_qualification_serial_generation_input,
    build_serial_generation_input,
    generate_configured_serial_base_path,
    generate_g4_reference_base_path,
    generate_g5_candidate_serial_base_path,
    generate_serial_base_path,
    is_g4_reference_base_path,
    is_g4_reference_generation_input,
    is_g5_candidate_serial_base_path,
    is_g5_candidate_serial_generation_input,
    is_rehearsal_serial_base_path,
    is_rehearsal_serial_generation_input,
    is_sealed_serial_base_path,
    is_sealed_serial_generation_input,
    verify_g5_candidate_serial_generation_input,
)
from prpg.validation.adversaries import (
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    fit_g5_adversary_pair,
)
from prpg.validation.lean_g5_assessment import (
    LEAN_G5_ASSESSMENT_SCHEMA_ID,
    LeanG5AssessmentReport,
    LeanG5DiversityAudit,
    LeanG5GroupedComparison,
    LeanG5StateAudit,
)
from prpg.validation.lean_g5_diagnostics import (
    LEAN_G5_DIVERSITY_SCHEMA_ID,
    LEAN_G5_STATE_BEHAVIOR_SCHEMA_ID,
    LeanG5ReturnDiversity,
    LeanG5StateBehavior,
)
from prpg.validation.lean_g5_statistics import (
    LEAN_G5_COMPONENT_NAMES,
    LEAN_G5_PILOT_GEOMETRY,
    LEAN_G5_PRODUCTION_GEOMETRY,
    AlphaEquivalenceAudit,
    HigherEmpiricalQuantile,
    InverseVolatilityGapAudit,
    LeakageGapAudit,
    LeanG5AlphaAudit,
    LeanG5Geometry,
    PlusOnePValue,
)
from prpg.validation.qualification import G5_ESTIMATOR_CONTRACT_FINGERPRINT
from prpg.validation.records import (
    ValidationDecision,
    build_canonical_results,
    build_canonical_thresholds,
)
from prpg.validation.statistics import (
    HolmAudit,
    HolmDecision,
    empirical_higher_critical_value,
    pointwise_outer_null_audit,
)

_PROCESSED = "a" * 64
_DESIGN_ARTIFACT = "b" * 64
_PRODUCTION_ARTIFACT = "c" * 64


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _immutable(values: Any, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _continuation(segments: tuple[int, ...]) -> np.ndarray:
    result = np.ones(sum(segments), dtype=np.bool_)
    offset = 0
    for length in segments:
        result[offset] = False
        offset += length
    return result


def _slice(
    role: str,
    monthly_rows: int,
    daily_rows: int,
    *,
    monthly_continuation: np.ndarray | None = None,
    daily_continuation: np.ndarray | None = None,
) -> CalibrationSlice:
    monthly = (
        np.arange(monthly_rows * 3, dtype=np.float64).reshape(monthly_rows, 3) * 1e-4
    )
    daily = np.arange(daily_rows * 3, dtype=np.float64).reshape(daily_rows, 3) * 1e-6
    return CalibrationSlice(
        role=cast(Any, role),
        monthly_returns=_immutable(monthly, np.float64),
        daily_returns=_immutable(daily, np.float64),
        macro_features=_immutable(np.zeros((monthly_rows, 4)), np.float64),
        monthly_period_ordinals=_immutable(
            np.arange(1_000, 1_000 + monthly_rows), np.int64
        ),
        daily_session_ns=_immutable(np.arange(10_000, 10_000 + daily_rows), np.int64),
        monthly_continuation=_immutable(
            _continuation((monthly_rows,))
            if monthly_continuation is None
            else monthly_continuation,
            np.bool_,
        ),
        daily_continuation=_immutable(
            _continuation((daily_rows,))
            if daily_continuation is None
            else daily_continuation,
            np.bool_,
        ),
    )


def _calibration_input(config: PRPGConfig) -> CalibrationInput:
    geometry = SplitGeometry(
        holdout_months=24,
        full_monthly_rows=217,
        full_daily_rows=4_547,
        design_monthly_rows=193,
        design_daily_rows=4_049,
        holdout_monthly_rows=24,
        holdout_daily_rows=498,
        full_first_month_ordinal=1_000,
        full_last_month_ordinal=1_216,
        design_last_month_ordinal=1_192,
        holdout_first_month_ordinal=1_193,
        full_first_daily_session_ns=10_000,
        full_last_daily_session_ns=14_546,
        design_last_daily_session_ns=14_048,
        holdout_first_daily_session_ns=14_049,
    )
    full = _slice(
        "full",
        217,
        4_547,
        monthly_continuation=_continuation((210, 7)),
        daily_continuation=_continuation((4_404, 143)),
    )
    source_files = []
    for path, field_name in (
        ("arrays/monthly_returns.npy", "monthly_returns"),
        ("arrays/daily_returns.npy", "daily_returns"),
        ("arrays/macro_features.npy", "macro_features"),
        ("arrays/monthly_period_ordinals.npy", "monthly_period_ordinals"),
        ("arrays/daily_session_ns.npy", "daily_session_ns"),
        ("arrays/monthly_continuation.npy", "monthly_continuation"),
        ("arrays/daily_continuation.npy", "daily_continuation"),
    ):
        size, digest = _npy_hash(getattr(full, field_name))
        source_files.append(
            SourceFileProvenance(
                layer="processed",
                relative_path=path,
                bytes=size,
                sha256=digest,
            )
        )
    provenance = CalibrationProvenance(
        processed_data_fingerprint=_PROCESSED,
        raw_snapshot_fingerprint="d" * 64,
        processed_config_fingerprint="e" * 64,
        calibration_config_fingerprint=config.fingerprint(),
        raw_acquisition_contract_sha256="f" * 64,
        data_dictionary_sha256="1" * 64,
        asset_columns=("equity", "muni_bond", "taxable_bond"),
        macro_columns=(
            "industrial_production_growth",
            "inflation_yoy",
            "yield_curve_spread",
            "credit_spread",
        ),
        source_identities=(),
        source_files=tuple(source_files),
    )
    value = CalibrationInput(
        fingerprint="0" * 64,
        provenance=provenance,
        geometry=geometry,
        full=full,
        design=_slice("design", 193, 4_049),
        holdout=_slice("holdout", 24, 498),
    )
    fingerprint = hashlib.sha256(
        (
            json.dumps(
                value.identity_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return replace(value, fingerprint=fingerprint)


def _policy(alpha: str, selected_lambda: float | None) -> ScientificGenerationPolicy:
    return ScientificGenerationPolicy(
        schema_version=SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=4,
        daily_block_length=20,
        alpha_policy=cast(Any, alpha),
        selected_neutral_lambda=selected_lambda,
        neutral_lambda_grid=CANONICAL_NEUTRAL_LAMBDA_GRID,
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _test_adversary_fit() -> Any:
    monthly_x = np.linspace(-0.03, 0.04, 24)
    daily_x = np.linspace(-0.01, 0.012, 72)

    def paths(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = np.column_stack((x, np.sin(11.0 * x), np.cos(7.0 * x) * 0.01))
        second = np.column_stack(
            (x[::-1] * 0.8, np.cos(9.0 * x) * 0.01, np.sin(5.0 * x) * 0.02)
        )
        return first, second

    return fit_g5_adversary_pair(paths(monthly_x), paths(daily_x))


def _test_thresholds(policy: ScientificGenerationPolicy, adversary_fit: Any) -> Any:
    policy_fingerprint = _json_fingerprint(policy.as_dict())
    critical = empirical_higher_critical_value(np.zeros(50_000), 0.95)
    return build_canonical_thresholds(
        threshold_set_id=f"g5-test-v{g5_policy_scientific_version(policy)}",
        binding_fingerprints={
            "g5_estimator_contract": G5_ESTIMATOR_CONTRACT_FINGERPRINT,
            "g5_generation_policy": policy_fingerprint,
            "g5_policy_stream": g5_policy_stream_fingerprint(policy),
            "g5_adversary_specification": (G5_ADVERSARY_SPECIFICATION_FINGERPRINT),
            "g5_adversary_fit": adversary_fit.fingerprint,
        },
        sobol_direction_fingerprint=_fp("g5-sobol-directions"),
        critical_values={"g5_a.max": critical},
        audit_intervals={"g5_a.practical_cap": pointwise_outer_null_audit(0)},
    )


def _required_qualification_decisions(
    *, directional_passed: bool = True, g5_a_holm_passed: bool = True
) -> tuple[ValidationDecision, ...]:
    passed_by_name = {
        "candidate_counts_exact": True,
        "g5_a_direct_alpha_equivalence": directional_passed,
        "g5_d_complete_path_diversity": True,
        "g5_a_holm": g5_a_holm_passed,
        "g5_b_holm": True,
        "g5_c_holm": True,
        "g5_d_holm": True,
    }
    return tuple(
        ValidationDecision(
            name=name,
            required=True,
            passed=passed,
            observed=passed,
            criterion="test canonical required decision",
        )
        for name, passed in passed_by_name.items()
    )


def _candidate_policy() -> ScientificCandidatePolicySet:
    return ScientificCandidatePolicySet(
        schema_id=SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_ID,
        schema_version=SCIENTIFIC_CANDIDATE_POLICY_SET_SCHEMA_VERSION,
        generator_algorithm=SCIENTIFIC_GENERATOR_ALGORITHM,
        block_policy_version=BLOCK_POLICY_VERSION,
        monthly_block_length=4,
        daily_block_length=20,
        candidate_policies=("historical_vector", "kernel_mean_neutral"),
        neutral_lambda_candidates=CANONICAL_NEUTRAL_LAMBDA_GRID,
        selected_policy=None,
        selected_neutral_lambda=None,
        qualification_gate="G5",
        initial_state="stationary",
        synchronized_return_vectors=True,
    )


def _artifact_arrays(
    role: str,
    *,
    source_mode: str,
    neutral_offsets: bool,
    n_states: int = 2,
) -> dict[str, np.ndarray]:
    monthly_segments = (193,) if role == "design" else (210, 7)
    daily_segments = (4_049,) if role == "design" else (4_404, 143)

    def states(count: int) -> np.ndarray:
        if source_mode == "blocks":
            return np.minimum(
                np.arange(count, dtype=np.int64) * n_states // count,
                n_states - 1,
            )
        return np.arange(count, dtype=np.int64) % n_states

    monthly_states = states(sum(monthly_segments))
    daily_states = states(sum(daily_segments))
    lambdas = np.asarray(CANONICAL_NEUTRAL_LAMBDA_GRID, dtype=np.float64)
    monthly_offsets = np.zeros((4, n_states, 3), dtype=np.float64)
    daily_offsets = np.zeros((4, n_states, 3), dtype=np.float64)
    if neutral_offsets:
        monthly_offsets[1] = (
            np.arange(1, n_states * 3 + 1, dtype=np.float64).reshape(n_states, 3) * 0.01
        )
        daily_offsets[1] = (
            np.arange(1, n_states * 3 + 1, dtype=np.float64).reshape(n_states, 3)
            * 0.001
        )
    transition = (
        np.asarray([[0.85, 0.15], [0.20, 0.80]])
        if n_states == 2
        else np.full((n_states, n_states), 0.15 / (n_states - 1))
    )
    if n_states != 2:
        np.fill_diagonal(transition, 0.85)
    raw: dict[str, np.ndarray] = {
        "hmm_start_probabilities": np.full(n_states, 1.0 / n_states),
        "hmm_transition_matrix": transition,
        "hmm_state_means": np.zeros((n_states, 4)),
        "hmm_tied_covariance": np.eye(4),
        "hmm_scaler_means": np.zeros(4),
        "hmm_scaler_standard_deviations": np.ones(4),
        "canonical_to_original": np.arange(n_states, dtype=np.int64),
        "original_to_canonical": np.arange(n_states, dtype=np.int64),
        "monthly_source_states": monthly_states,
        "daily_source_states": daily_states,
        "monthly_continuation": _continuation(monthly_segments),
        "daily_continuation": _continuation(daily_segments),
        "monthly_state_counts": np.bincount(monthly_states, minlength=n_states),
        "daily_state_counts": np.bincount(daily_states, minlength=n_states),
        "monthly_pc1_loadings": np.asarray([1.0, 0.0, 0.0]),
        "daily_pc1_loadings": np.asarray([1.0, 0.0, 0.0]),
        "monthly_kernel_lambdas": lambdas,
        "daily_kernel_lambdas": lambdas,
        "monthly_kernel_means": np.zeros((n_states, 3)),
        "daily_kernel_means": np.zeros((n_states, 3)),
        "monthly_kernel_half_widths": np.full((n_states, 3), 0.0005),
        "daily_kernel_half_widths": np.full((n_states, 3), 0.0005),
        "monthly_kernel_state_counts": np.full(n_states, 10_000, dtype=np.int64),
        "daily_kernel_state_counts": np.full(n_states, 10_000, dtype=np.int64),
        "monthly_kernel_offsets": monthly_offsets,
        "daily_kernel_offsets": daily_offsets,
    }
    return {name: _immutable(value, value.dtype) for name, value in raw.items()}


def _npy_hash(array: np.ndarray) -> tuple[int, str]:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    content = output.getvalue()
    return len(content), hashlib.sha256(content).hexdigest()


def _artifact(
    role: str,
    binding: ScientificArtifactBinding,
    policy: ScientificCandidatePolicySet,
    *,
    source_mode: str,
    neutral_offsets: bool,
    n_states: int = 2,
) -> LoadedScientificModelArtifact:
    import prpg.model.scientific_artifact as artifact_module

    arrays = _artifact_arrays(
        role,
        source_mode=source_mode,
        neutral_offsets=neutral_offsets,
        n_states=n_states,
    )
    records = []
    for name, array in sorted(arrays.items()):
        size, digest = _npy_hash(array)
        records.append(
            {
                "name": name,
                "member_name": f"{name}.npy",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "order": "C",
                "bytes": size,
                "sha256": digest,
            }
        )
    fingerprint = _DESIGN_ARTIFACT if role == "design" else _PRODUCTION_ARTIFACT
    reference = ModelArtifactReference(
        fingerprint=fingerprint,
        path=Path("/verified") / fingerprint,
        manifest=MappingProxyType(
            {
                "identity": MappingProxyType(
                    {"parameters": MappingProxyType({"arrays": tuple(records)})}
                )
            }
        ),
        reused=True,
    )
    result = object.__new__(LoadedScientificModelArtifact)
    for name, value in (
        ("reference", reference),
        ("role", role),
        ("selected_k", n_states),
        ("geometry", ScientificArtifactGeometry.for_role(cast(Any, role))),
        ("binding", binding),
        (
            "provenance",
            ScientificArtifactProvenance(
                source_code_fingerprint="2" * 64,
                dependency_lock_fingerprint="3" * 64,
            ),
        ),
        ("generation_policy", policy),
        ("arrays", MappingProxyType(arrays)),
        ("reports", MappingProxyType({})),
        ("core_model_fingerprint", "4" * 64),
        ("parameter_set_fingerprint", "5" * 64),
        ("_schema_seal", artifact_module._LOADED_ARTIFACT_SEAL),
    ):
        object.__setattr__(result, name, value)
    return result


def _passed_bundle(
    config: PRPGConfig, calibration_input: CalibrationInput
) -> G3EvidenceBundle:
    records = tuple(
        gate_record(
            gate_id,
            passed=True,
            evidence={"task": gate_id},
            summary={"passed": True},
        )
        for gate_id in G3_GATE_ORDER
    )
    return build_g3_evidence_bundle(
        config_fingerprint=config.fingerprint(),
        processed_data_fingerprint=(
            calibration_input.provenance.processed_data_fingerprint
        ),
        calibration_input_fingerprint=calibration_input.fingerprint,
        design_artifact_fingerprint=_DESIGN_ARTIFACT,
        production_artifact_fingerprint=_PRODUCTION_ARTIFACT,
        records=records,
    )


def _authority(
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    policy: ScientificGenerationPolicy,
    *,
    source_mode: str,
    neutral_offsets: bool,
    n_states: int = 2,
) -> G5QualifiedProductionAuthority:
    import prpg.model.g3_evidence_store as evidence_module

    binding = ScientificArtifactBinding(
        config_fingerprint=config.fingerprint(),
        processed_data_fingerprint=(
            calibration_input.provenance.processed_data_fingerprint
        ),
        calibration_input_fingerprint=calibration_input.fingerprint,
        calibration_run_fingerprint="6" * 64,
        preflight_fingerprint="7" * 64,
    )
    design = _artifact(
        "design",
        binding,
        _candidate_policy(),
        source_mode=source_mode,
        neutral_offsets=neutral_offsets,
        n_states=n_states,
    )
    production = _artifact(
        "production",
        binding,
        _candidate_policy(),
        source_mode=source_mode,
        neutral_offsets=neutral_offsets,
        n_states=n_states,
    )
    bundle = _passed_bundle(config, calibration_input)
    task_fingerprints = {task_id: _fp(f"task:{task_id}") for task_id in G3_GATE_ORDER}
    checkpoint_fingerprints = {
        task_id: _fp(f"checkpoint:{task_id}") for task_id in G3_GATE_ORDER
    }
    look_fingerprints = {
        spec.family_id: [_fp(f"look:{spec.family_id}")]
        for spec in G3_PLANNED_LOOK_REGISTRY
    }
    pair_receipt_fingerprint = _fp("artifact-pair")
    evidence_identity = {
        "schema_id": G3_EVIDENCE_SCHEMA_ID,
        "schema_version": G3_EVIDENCE_SCHEMA_VERSION,
        "contract_version": bundle.contract_version,
        "task_registry_fingerprint": G3_TASK_REGISTRY_FINGERPRINT,
        "planned_look_registry_fingerprint": (G3_PLANNED_LOOK_REGISTRY_FINGERPRINT),
        "passed": True,
        "generation_authorized": False,
        "evidence_bundle": {
            "identity": bundle.identity_dict(),
            "fingerprint": bundle.fingerprint,
        },
        "run": {
            "run_id": "6" * 64,
            "completion_fingerprint": "9" * 64,
            "completion_sequence": 2,
            "preflight_fingerprint": "7" * 64,
            "workers": 1,
            "task_result_fingerprints": task_fingerprints,
            "checkpoint_fingerprints": checkpoint_fingerprints,
            "planned_look_fingerprints": look_fingerprints,
        },
        "artifacts": {
            "design_artifact_fingerprint": design.reference.fingerprint,
            "production_artifact_fingerprint": production.reference.fingerprint,
            "pair_receipt_fingerprint": pair_receipt_fingerprint,
        },
        "provenance": {
            "execution_closure_schema": G3_EXECUTION_CLOSURE_SCHEMA,
            "execution_closure_manifest": G3_EXECUTION_CLOSURE_MANIFEST.as_dict(),
            "execution_closure_manifest_fingerprint": (
                G3_EXECUTION_CLOSURE_MANIFEST_FINGERPRINT
            ),
            "source_code_fingerprint": "2" * 64,
            "dependency_lock_fingerprint": "3" * 64,
        },
    }
    evidence_fingerprint = hashlib.sha256(
        (
            json.dumps(
                evidence_identity,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    result = object.__new__(LoadedG3EvidenceAuthority)
    for name, value in (
        (
            "reference",
            G3EvidenceReference(
                fingerprint=evidence_fingerprint,
                path=Path("/verified/evidence"),
                manifest=MappingProxyType(
                    {
                        "schema_version": G3_EVIDENCE_SCHEMA_VERSION,
                        "evidence_fingerprint": evidence_fingerprint,
                        "identity": evidence_identity,
                    }
                ),
                reused=True,
            ),
        ),
        ("evidence_bundle", bundle),
        ("run_id", "6" * 64),
        ("completion_fingerprint", "9" * 64),
        ("task_result_fingerprints", MappingProxyType(task_fingerprints)),
        ("checkpoint_fingerprints", MappingProxyType(checkpoint_fingerprints)),
        (
            "planned_look_fingerprints",
            MappingProxyType(
                {key: tuple(values) for key, values in look_fingerprints.items()}
            ),
        ),
        ("design_artifact", design),
        ("production_artifact", production),
        ("pair_receipt_fingerprint", pair_receipt_fingerprint),
        ("g3_source_code_fingerprint", "2" * 64),
        ("g3_dependency_lock_fingerprint", "3" * 64),
        ("_load_seal", evidence_module._LOAD_SEAL),
    ):
        object.__setattr__(result, name, value)
    evidence_module._LOADED_G3_EVIDENCE[id(result)] = result
    evidence_module.verify_loaded_g3_evidence(result)
    adversary_fit = _test_adversary_fit()
    historical_policy = _policy("historical_vector", None)
    historical_threshold = _produce_g5_threshold_evidence_for_testing(
        g3_evidence=result,
        generation_policy=historical_policy,
        thresholds=_test_thresholds(historical_policy, adversary_fit),
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        adversary_fit=adversary_fit,
        seed_registry_fingerprint=_fp("g5-qualification-seeds"),
        g5_source_code_fingerprint=_fp("g5-source"),
        g5_dependency_lock_fingerprint=_fp("g5-lock"),
        _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
    )
    threshold = historical_threshold
    if policy.alpha_policy == "kernel_mean_neutral":
        baseline_results = build_canonical_results(
            result_id="g5-test-baseline-nonpass",
            threshold_fingerprint=historical_threshold.threshold_registry_fingerprint,
            subject_fingerprint=_fp("g5-baseline-qualification-subject"),
            metric_fingerprints={"adversary_fit": adversary_fit.fingerprint},
            decisions=_required_qualification_decisions(directional_passed=False),
        )
        rows: list[G5DevelopmentPolicyDecision] = []
        selected_seen = False
        for neutral_lambda in CANONICAL_NEUTRAL_LAMBDA_GRID:
            candidate = _policy("kernel_mean_neutral", neutral_lambda)
            evaluated = not selected_seen
            passed = evaluated and neutral_lambda == policy.selected_neutral_lambda
            rows.append(
                G5DevelopmentPolicyDecision(
                    generation_policy=candidate,
                    generation_policy_fingerprint=_json_fingerprint(
                        candidate.as_dict()
                    ),
                    neutral_lambda=neutral_lambda,
                    policy_scientific_version=g5_policy_scientific_version(candidate),
                    evaluated=evaluated,
                    development_result_fingerprint=(
                        _fp(f"g5-development-{neutral_lambda}") if evaluated else None
                    ),
                    passed=passed if evaluated else None,
                )
            )
            selected_seen = selected_seen or passed
        selection = produce_g5_development_policy_selection(
            baseline_threshold_evidence=historical_threshold,
            baseline_qualification_results=baseline_results,
            decisions=rows,
        )
        threshold = _produce_g5_threshold_evidence_for_testing(
            g3_evidence=result,
            generation_policy=policy,
            thresholds=_test_thresholds(policy, adversary_fit),
            estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
            adversary_fit=adversary_fit,
            seed_registry_fingerprint=_fp("g5-qualification-seeds"),
            g5_source_code_fingerprint=_fp("g5-source"),
            g5_dependency_lock_fingerprint=_fp("g5-lock"),
            development_selection=selection,
            _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
        )
    qualification_results = build_canonical_results(
        result_id=f"g5-test-qualification-v{g5_policy_scientific_version(policy)}",
        threshold_fingerprint=threshold.threshold_registry_fingerprint,
        subject_fingerprint=_fp(
            f"g5-qualification-subject-{g5_policy_scientific_version(policy)}"
        ),
        metric_fingerprints={"adversary_fit": adversary_fit.fingerprint},
        decisions=_required_qualification_decisions(),
    )
    qualification = _produce_g5_qualification_evidence_for_testing(
        threshold_evidence=threshold,
        generation_policy=policy,
        qualification_results=qualification_results,
        _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
    )
    meta_results = build_canonical_results(
        result_id=f"g5-test-meta-v{g5_policy_scientific_version(policy)}",
        threshold_fingerprint=threshold.threshold_registry_fingerprint,
        subject_fingerprint=qualification.qualification_results_fingerprint,
        metric_fingerprints={"meta": _fp("g5-meta-metrics")},
        decisions=(
            ValidationDecision(
                name="meta_validation",
                required=True,
                passed=True,
                observed=True,
                criterion="test canonical meta pass",
            ),
        ),
    )
    meta = _produce_g5_meta_validation_evidence_for_testing(
        threshold_evidence=threshold,
        qualification_evidence=qualification,
        meta_validation_results=meta_results,
        _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
    )
    return _produce_g5_qualified_production_authority_for_testing(
        g3_evidence=result,
        generation_policy=policy,
        threshold_evidence=threshold,
        qualification_evidence=qualification,
        meta_validation_evidence=meta,
        _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
    )


def _config() -> PRPGConfig:
    return load_config("configs/smoke.yaml")


def _fixture(
    *,
    neutral: bool = False,
    source_mode: str = "alternating",
) -> tuple[
    SerialGenerationInput,
    G5QualifiedProductionAuthority,
    PRPGConfig,
    CalibrationInput,
]:
    config = _config()
    calibration_input = _calibration_input(config)
    policy = _policy(
        "kernel_mean_neutral" if neutral else "historical_vector",
        0.5 if neutral else None,
    )
    authority = _authority(
        config,
        calibration_input,
        policy,
        source_mode=source_mode,
        neutral_offsets=True,
    )
    generation_input = _build_noncanonical_g5_serial_generation_input_for_testing(
        authority=authority,
        config=config,
        calibration_input=calibration_input,
    )
    return generation_input, authority, config, calibration_input


@pytest.mark.parametrize(
    ("neutral", "expected_policy", "expected_lambda"),
    [
        (False, "historical_vector", None),
        (True, "kernel_mean_neutral", 0.5),
    ],
)
def test_unselected_g3_config_defers_policy_choice_to_sealed_g5_authority(
    neutral: bool, expected_policy: str, expected_lambda: float | None
) -> None:
    generation_input, authority, config, _ = _fixture(neutral=neutral)

    assert config.model.selected_alpha_policy is None
    assert config.model.selected_neutral_lambda is None
    assert generation_input.generation_policy == authority.generation_policy
    assert generation_input.generation_policy.alpha_policy == expected_policy
    assert generation_input.generation_policy.selected_neutral_lambda == expected_lambda


def test_pre_g5_rehearsal_path_is_explicitly_noncanonical_and_unreleasable() -> None:
    _, authority, config, calibration_input = _fixture()
    rehearsal = _build_rehearsal_serial_generation_input_for_testing(
        g3_evidence=authority.g3_evidence,
        generation_policy=authority.generation_policy,
        config=config,
        calibration_input=calibration_input,
        rehearsal_scope_fingerprint=_fp("phase4-reference-engine-rehearsal"),
    )
    assert rehearsal.execution_mode == "rehearsal"
    assert rehearsal.generation_authorized is False
    assert rehearsal.g5_authority_fingerprint is None
    assert is_rehearsal_serial_generation_input(rehearsal)
    assert not is_sealed_serial_generation_input(rehearsal)
    with pytest.raises(GenerationError):
        generate_serial_base_path(rehearsal, family="LF", path_entity=1, years=1)

    path = _generate_rehearsal_serial_base_path_for_testing(
        rehearsal,
        family="LF",
        path_entity=1,
        years=1,
    )
    assert path.execution_mode == "rehearsal"
    assert path.generation_authorized is False
    assert is_rehearsal_serial_base_path(path)
    assert not is_sealed_serial_base_path(path)
    with pytest.raises(GenerationError):
        aggregate_lf_log_returns(path)  # type: ignore[arg-type]


def test_public_g4_reference_api_uses_development_and_stays_out_of_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prpg.simulation.serial as serial_module
    from prpg.simulation.rng import Domain
    from prpg.storage.csv_v1 import write_serial_csv_shard

    _, authority, config, calibration_input = _fixture()
    reference_input = build_g4_reference_generation_input(
        g3_evidence=authority.g3_evidence,
        generation_policy=authority.generation_policy,
        config=config,
        calibration_input=calibration_input,
        reference_scope_fingerprint=_fp("g4-public-reference"),
    )
    observed_domains: list[Domain] = []
    real_uniforms = serial_module.uniforms

    def capture_domain(key: Any, size: int) -> np.ndarray:
        observed_domains.append(key.domain)
        return real_uniforms(key, size)

    monkeypatch.setattr(serial_module, "uniforms", capture_domain)
    path = generate_g4_reference_base_path(
        reference_input,
        family="LF",
        path_entity=1,
        years=2,
    )

    assert is_g4_reference_generation_input(reference_input)
    assert is_g4_reference_base_path(path)
    assert reference_input.generation_authorized is False
    assert path.generation_authorized is False
    assert set(observed_domains) == {Domain.DEVELOPMENT}
    with pytest.raises(GenerationError):
        generate_serial_base_path(  # type: ignore[arg-type]
            reference_input, family="LF", path_entity=1, years=2
        )
    with pytest.raises(GenerationError):
        aggregate_lf_log_returns(path)  # type: ignore[arg-type]
    with pytest.raises(GenerationError):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=(path,),
        )


def test_fixed_g4_reference_fixture_writes_and_revalidates_all_frequencies(
    tmp_path: Path,
) -> None:
    from prpg.data.processed import ProcessedSnapshotStore
    from prpg.model.g3_preflight import CANONICAL_PROCESSED_DATA_FINGERPRINT
    from prpg.model.input import load_calibration_input
    from prpg.simulation.g4_reference import (
        run_g4_reference_fixture,
        validate_g4_reference_csvs,
    )
    from prpg.storage.csv_v1 import CSV_HEADER, format_csv_float64

    config = load_config("configs/canonical.yaml")
    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(config.data.artifact_root),
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=config.fingerprint(),
        holdout_months=config.model.design_holdout_months,
    )
    authority = _authority(
        config,
        calibration_input,
        _policy("historical_vector", None),
        source_mode="alternating",
        neutral_offsets=True,
        n_states=4,
    )
    output = (tmp_path / "g4-reference").resolve()

    report = run_g4_reference_fixture(
        g3_evidence=authority.g3_evidence,
        config=config,
        calibration_input=calibration_input,
        output_root=output,
    )

    assert report.g4_passed is True
    assert report.canonical_authority is False
    assert report.generation_authorized is False
    assert report.releasable is False
    assert report.structural_validation.passed is True
    assert report.structural_validation.files == 25
    assert report.structural_validation.rows == 6_760
    assert report.identity["fixture"] == {
        "policies": 5,
        "lf_paths_per_policy": 4,
        "hf_paths_per_policy": 2,
        "years": 2,
        "base_paths": 30,
        "files": 25,
        "rows": 6_760,
    }
    assert {item.policy_id for item in report.files} == {
        "historical_vector",
        "kernel_mean_neutral_lambda_0_25",
        "kernel_mean_neutral_lambda_0_50",
        "kernel_mean_neutral_lambda_0_75",
        "kernel_mean_neutral_lambda_1_00",
    }
    manifest = json.loads(report.path.read_bytes())
    assert manifest["report_fingerprint"] == report.fingerprint
    assert manifest["identity"]["g4_passed"] is True
    assert manifest["identity"]["generation_authorized"] is False
    assert validate_g4_reference_csvs(output, report.files).passed is True
    assert all(
        (output / item.relative_path).read_text(encoding="utf-8").splitlines()[0]
        == CSV_HEADER
        for item in report.files
    )

    # One re-signed derived-value corruption proves validation is semantic,
    # rather than merely trusting the recorded file hash.
    record = next(
        item
        for item in report.files
        if item.policy_id == "historical_vector" and item.frequency == "quarterly"
    )
    csv_path = output / record.relative_path
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[4] = format_csv_float64(float(fields[4]) + 0.01)
    lines[1] = ",".join(fields)
    changed_content = ("\n".join(lines) + "\n").encode("utf-8")
    csv_path.write_bytes(changed_content)
    changed_record = replace(
        record,
        bytes=len(changed_content),
        sha256=hashlib.sha256(changed_content).hexdigest(),
    )
    changed_files = tuple(
        changed_record if item is record else item for item in report.files
    )
    with pytest.raises(IntegrityError, match="log-sum identity failed"):
        validate_g4_reference_csvs(output, changed_files)


def test_g5_candidate_and_final_production_authorities_are_separate() -> None:
    _, final, config, calibration_input = _fixture()
    with pytest.raises(TypeError, match="public G5 producer"):
        G5QualifiedProductionAuthority()
    with pytest.raises(TypeError, match="G5 qualification coordinator"):
        G5QualificationGenerationAuthority()
    assert final.generation_authorized is True
    assert final.outer_null_replicates == G5_OUTER_NULL_REPLICATES
    assert final.g3_evidence.generation_authorized is False
    assert is_g5_qualified_production_authority(final)
    assert not is_g5_qualified_production_authority(copy.copy(final))
    for bypass in (config, final.production_artifact, final.generation_policy):
        with pytest.raises(GenerationError, match="qualified G5 production"):
            build_serial_generation_input(
                authority=bypass,  # type: ignore[arg-type]
                config=config,
                calibration_input=calibration_input,
            )
    with pytest.raises(ModelError, match="approved sealed literal fresh-history"):
        produce_g5_threshold_evidence(
            g3_evidence=final.g3_evidence,
            generation_policy=final.generation_policy,
            thresholds=final.threshold_evidence.thresholds,
            estimator_contract_fingerprint=_fp("wrong-estimator"),
            adversary_fit=final.threshold_evidence.adversary_fit,
            seed_registry_fingerprint=_fp("g5-qualification-seeds"),
            g5_source_code_fingerprint=_fp("g5-source"),
            g5_dependency_lock_fingerprint=_fp("g5-lock"),
        )

    with pytest.raises(ModelError, match="awaits sealed threshold"):
        produce_g5_qualification_generation_authority(
            threshold_evidence=final.threshold_evidence,
        )
    with pytest.raises(GenerationError, match="noncanonical G5 authority"):
        build_serial_generation_input(
            authority=final,
            config=config,
            calibration_input=calibration_input,
        )

    qualification = _produce_g5_qualification_generation_authority_for_testing(
        threshold_evidence=final.threshold_evidence,
        _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
    )
    assert qualification.generation_authorized is False
    assert qualification.qualification_generation_authorized is True
    assert qualification.development_lf_path_count == G5_DEVELOPMENT_LF_PATH_COUNT
    assert qualification.development_hf_path_count == G5_DEVELOPMENT_HF_PATH_COUNT
    assert qualification.qualification_lf_path_count == G5_QUALIFICATION_LF_PATH_COUNT
    assert qualification.qualification_hf_path_count == G5_QUALIFICATION_HF_PATH_COUNT
    assert not hasattr(qualification, "qualification_path_count")
    assert not hasattr(qualification, "outer_null_replicates")
    assert is_g5_qualification_generation_authority(qualification)
    assert not is_g5_qualified_production_authority(qualification)
    with pytest.raises(GenerationError, match="qualified G5 production"):
        build_serial_generation_input(
            authority=qualification,  # type: ignore[arg-type]
            config=config,
            calibration_input=calibration_input,
        )


def test_g5_candidate_paths_use_distinct_domains_exact_bounds_and_no_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import prpg.simulation.serial as serial_module
    from prpg.storage.csv_v1 import write_serial_csv_shard

    _, final, config, calibration_input = _fixture()
    qualification_authority = (
        _produce_g5_qualification_generation_authority_for_testing(
            threshold_evidence=final.threshold_evidence,
            _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
        )
    )
    development = build_g5_candidate_serial_generation_input(
        authority=final.g3_evidence,
        generation_policy=final.generation_policy,
        candidate_stage="development",
        config=config,
        calibration_input=calibration_input,
    )
    with pytest.raises(GenerationError, match="noncanonical test authority"):
        build_g5_candidate_serial_generation_input(
            authority=qualification_authority,
            generation_policy=final.generation_policy,
            candidate_stage="qualification",
            config=config,
            calibration_input=calibration_input,
        )
    qualification = _build_g5_candidate_serial_generation_input_for_testing(
        authority=qualification_authority,
        generation_policy=final.generation_policy,
        candidate_stage="qualification",
        config=config,
        calibration_input=calibration_input,
    )
    neutral = build_g5_candidate_serial_generation_input(
        authority=final.g3_evidence,
        generation_policy=_policy("kernel_mean_neutral", 0.5),
        candidate_stage="development",
        config=config,
        calibration_input=calibration_input,
    )

    assert is_g5_candidate_serial_generation_input(development)
    assert is_g5_candidate_serial_generation_input(qualification)
    assert development.generation_authorized is False
    assert development.rng_domain is Domain.DEVELOPMENT
    assert qualification.rng_domain is Domain.QUALIFICATION
    assert (development.maximum_lf_paths, development.maximum_hf_paths) == (100, 20)
    assert (qualification.maximum_lf_paths, qualification.maximum_hf_paths) == (
        500,
        200,
    )
    assert neutral.fingerprint != development.fingerprint
    assert neutral.generation_policy_fingerprint != (
        development.generation_policy_fingerprint
    )
    with pytest.raises(TypeError, match="public builder"):
        G5CandidateSerialGenerationInput()
    with pytest.raises(TypeError, match="public generator"):
        G5CandidateSerialBasePath()

    for candidate, family, entity in (
        (development, "LF", 101),
        (development, "HF", 21),
        (qualification, "LF", 501),
        (qualification, "HF", 201),
    ):
        with pytest.raises(GenerationError, match="registered range"):
            generate_g5_candidate_serial_base_path(
                candidate,
                family=cast(Any, family),
                path_entity=entity,
            )

    observed_domains: list[Domain] = []
    real_uniforms = serial_module.uniforms

    def capture_domain(key: Any, size: int) -> np.ndarray:
        observed_domains.append(key.domain)
        return real_uniforms(key, size)

    monkeypatch.setattr(serial_module, "uniforms", capture_domain)
    development_path = generate_g5_candidate_serial_base_path(
        development, family="LF", path_entity=1
    )
    assert observed_domains == [Domain.DEVELOPMENT] * 3
    observed_domains.clear()
    qualification_path = generate_g5_candidate_serial_base_path(
        qualification, family="LF", path_entity=1
    )
    assert observed_domains == [Domain.QUALIFICATION] * 3
    assert is_g5_candidate_serial_base_path(development_path)
    assert is_g5_candidate_serial_base_path(qualification_path)
    assert development_path.generation_authorized is False
    assert development_path.fingerprint != qualification_path.fingerprint
    with pytest.raises(GenerationError, match="serial base path"):
        aggregate_lf_log_returns(development_path)  # type: ignore[arg-type]
    with pytest.raises(GenerationError, match="serial base path"):
        aggregate_lf_log_returns(development_path.base_path)  # type: ignore[arg-type]
    with pytest.raises(GenerationError):
        write_serial_csv_shard(
            tmp_path,
            frequency="monthly",
            shard_index=0,
            first_path_entity=1,
            last_path_entity=1,
            values=(development_path,),  # type: ignore[arg-type]
        )


def test_qualification_evaluation_window_uses_selected_policy_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prpg.validation.qualification as qualification_module

    _, final, config, calibration_input = _fixture(neutral=True)
    qualification_authority = (
        _produce_g5_qualification_generation_authority_for_testing(
            threshold_evidence=final.threshold_evidence,
            _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
        )
    )
    generation_input = _build_g5_candidate_serial_generation_input_for_testing(
        authority=qualification_authority,
        generation_policy=final.generation_policy,
        candidate_stage="qualification",
        config=config,
        calibration_input=calibration_input,
    )
    path = generate_g5_candidate_serial_base_path(
        generation_input, family="LF", path_entity=1
    )
    real_key = qualification_module.calibration_rng_key
    observed_versions: list[int] = []

    def capture_version(**kwargs: Any) -> Any:
        observed_versions.append(kwargs["scientific_version"])
        return real_key(**kwargs)

    monkeypatch.setattr(qualification_module, "calibration_rng_key", capture_version)
    qualification_module._candidate_path_view(path)

    assert generation_input.policy_scientific_version != 11
    assert observed_versions == [generation_input.policy_scientific_version]


def test_g5_final_authority_requires_all_passed_typed_evidence() -> None:
    _, final, _, _ = _fixture()
    failed_results = build_canonical_results(
        result_id="g5-test-failed-qualification",
        threshold_fingerprint=final.threshold_registry_fingerprint,
        subject_fingerprint=_fp("failed-qualification-subject"),
        metric_fingerprints={"adversary_fit": final.adversary_fit_fingerprint},
        decisions=_required_qualification_decisions(directional_passed=False),
    )
    with pytest.raises(ModelError, match="passed canonical result"):
        _produce_g5_qualification_evidence_for_testing(
            threshold_evidence=final.threshold_evidence,
            generation_policy=final.generation_policy,
            qualification_results=failed_results,
            _test_capability=_G5_NONCANONICAL_TEST_CAPABILITY,
        )
    with pytest.raises(ModelError, match="approved literal-null"):
        produce_g5_qualified_production_authority(
            g3_evidence=final.g3_evidence,
            generation_policy=final.generation_policy,
            threshold_evidence=final.threshold_evidence,
            qualification_evidence=final.qualification_evidence,
            meta_validation_evidence=object(),  # type: ignore[arg-type]
        )


def test_g5_evidence_store_roundtrip_no_overwrite_tamper_and_serial_acceptance(
    tmp_path: Path,
) -> None:
    _, final, _, _ = _fixture()
    store = G5EvidenceStore(tmp_path)
    fit_record = store.publish_adversary_fit(final.threshold_evidence.adversary_fit)
    loaded_fit = store.load_adversary_fit(fit_record.fingerprint)
    assert loaded_fit.fingerprint == final.adversary_fit_fingerprint
    with pytest.raises(IntegrityError, match="overwrite refused"):
        store.publish_adversary_fit(final.threshold_evidence.adversary_fit)
    with pytest.raises(ModelError, match="noncanonical test threshold"):
        store.publish_threshold_evidence(final.threshold_evidence)
    with pytest.raises(ModelError, match="noncanonical test qualification"):
        store.publish_qualification_evidence(final.qualification_evidence)
    with pytest.raises(ModelError, match="noncanonical test meta"):
        store.publish_meta_validation_evidence(final.meta_validation_evidence)
    with pytest.raises(ModelError, match="noncanonical test final"):
        store.publish_final_authority(final)


def test_serial_replay_golden_draws_vector_synchrony_and_path_separation() -> None:
    generation_input, _, _, calibration_input = _fixture()

    first = generate_serial_base_path(
        generation_input, family="LF", path_entity=1, years=1
    )
    replay = generate_serial_base_path(
        generation_input, family="LF", path_entity=1, years=1
    )
    different = generate_serial_base_path(
        generation_input, family="LF", path_entity=2, years=1
    )

    assert is_sealed_serial_generation_input(generation_input)
    assert is_sealed_serial_base_path(first)
    assert first.fingerprint == replay.fingerprint
    np.testing.assert_array_equal(first.log_returns, replay.log_returns)
    assert first.fingerprint != different.fingerprint
    np.testing.assert_array_equal(
        first.log_returns,
        calibration_input.full.monthly_returns[first.source_indices],
    )
    assert first.target_monthly_states.tolist() == [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
    ]
    assert first.source_indices.tolist() == [
        99,
        69,
        74,
        16,
        106,
        178,
        104,
        30,
        200,
        186,
        115,
        61,
    ]
    assert first.state_uniforms_consumed == 12
    assert first.restart_audit.restart_uniforms_consumed == 12
    assert first.restart_audit.source_rank_uniforms_consumed == 12


def test_hf_expands_monthly_states_and_copies_synchronized_vectors() -> None:
    generation_input, _, _, calibration_input = _fixture()
    path = generate_serial_base_path(
        generation_input, family="HF", path_entity=1, years=2
    )
    assert path.log_returns.shape == (504, 3)
    np.testing.assert_array_equal(
        path.target_states, np.repeat(path.target_monthly_states, 21)
    )
    np.testing.assert_array_equal(
        path.log_returns,
        calibration_input.full.daily_returns[path.source_indices],
    )


def test_kernel_offset_is_exactly_one_shot_and_historical_is_unchanged() -> None:
    neutral, _, _, calibration_input = _fixture(neutral=True)
    path = generate_serial_base_path(neutral, family="LF", path_entity=1, years=1)
    offsets = neutral.monthly_kernel_offsets[1, path.target_states]
    np.testing.assert_array_equal(
        path.log_returns,
        calibration_input.full.monthly_returns[path.source_indices] - offsets,
    )

    historical, _, _, raw_input = _fixture()
    raw_path = generate_serial_base_path(
        historical, family="LF", path_entity=1, years=1
    )
    np.testing.assert_array_equal(
        raw_path.log_returns, raw_input.full.monthly_returns[raw_path.source_indices]
    )


@pytest.mark.parametrize(
    ("source_index", "state_draws", "restart_second", "expected"),
    [
        (216, np.zeros(12), 0.99, RestartReason.SOURCE_END),
        (209, np.full(12, 0.99), 0.99, RestartReason.SOURCE_GAP),
        (0, np.zeros(12), 0.99, RestartReason.SOURCE_STATE_MISMATCH),
    ],
)
def test_forced_source_restart_reasons(
    monkeypatch: pytest.MonkeyPatch,
    source_index: int,
    state_draws: np.ndarray,
    restart_second: float,
    expected: RestartReason,
) -> None:
    import prpg.simulation.serial as module

    generation_input, _, _, _ = _fixture()
    state = 0 if state_draws[0] == 0 else 1
    eligible = np.flatnonzero(generation_input.monthly_source_states == state)
    rank = int(np.flatnonzero(eligible == source_index)[0])
    source_draws = np.zeros(12)
    source_draws[0] = (rank + 0.1) / eligible.size
    restart_draws = np.full(12, 0.99)
    restart_draws[1] = restart_second

    def fake_uniforms(key: Any, count: int) -> np.ndarray:
        assert count == 12
        if int(key.stage) == int(Stage.STATE_CHAIN):
            return state_draws
        if int(key.stage) == int(Stage.RESTART_UNIFORMS):
            return restart_draws
        return source_draws

    monkeypatch.setattr(module, "uniforms", fake_uniforms)
    path = generate_serial_base_path(
        generation_input, family="LF", path_entity=1, years=1
    )
    assert path.source_indices[0] == source_index
    assert path.restart_audit.reasons[1] is expected


def test_random_restart_continuation_and_target_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prpg.simulation.serial as module

    generation_input, _, _, _ = _fixture(source_mode="blocks")

    def run(state: np.ndarray, second_restart: float) -> RestartReason:
        def fake_uniforms(key: Any, count: int) -> np.ndarray:
            if int(key.stage) == int(Stage.STATE_CHAIN):
                return state
            if int(key.stage) == int(Stage.RESTART_UNIFORMS):
                values = np.full(count, 0.99)
                values[1] = second_restart
                return values
            return np.zeros(count)

        monkeypatch.setattr(module, "uniforms", fake_uniforms)
        return generate_serial_base_path(
            generation_input, family="LF", path_entity=1, years=1
        ).restart_audit.reasons[1]

    assert run(np.zeros(12), 0.99) is RestartReason.CONTINUATION
    assert run(np.zeros(12), 0.0) is RestartReason.RANDOM_RESTART
    changed = np.zeros(12)
    changed[1] = 0.99
    assert run(changed, 0.99) is RestartReason.TARGET_STATE_TRANSITION


def test_configured_wrapper_and_invalid_ranges() -> None:
    generation_input, _, _, _ = _fixture()
    path = generate_configured_serial_base_path(
        generation_input, family="HF", path_entity=2
    )
    assert path.years == 2
    assert path.log_returns.shape == (504, 3)
    with pytest.raises(GenerationError):
        generate_configured_serial_base_path(
            generation_input, family="HF", path_entity=3
        )
    for entity in (0, -1, True, 2**32):
        with pytest.raises(GenerationError):
            generate_serial_base_path(
                generation_input,
                family="LF",
                path_entity=cast(Any, entity),
                years=1,
            )
    for years in (0, -1, True):
        with pytest.raises(GenerationError):
            generate_serial_base_path(
                generation_input,
                family="LF",
                path_entity=1,
                years=cast(Any, years),
            )
    with pytest.raises(GenerationError):
        generate_serial_base_path(
            generation_input, family=cast(Any, "lf"), path_entity=1, years=1
        )
    with pytest.raises(TypeError):
        generate_serial_base_path(
            generation_input,
            family="LF",
            path_entity=1,
            years=1,
            worker_id=2,  # type: ignore[call-arg]
        )


def test_copied_and_tampered_generation_capabilities_are_rejected() -> None:
    generation_input, _, _, _ = _fixture()
    copied = copy.copy(generation_input)
    assert not is_sealed_serial_generation_input(copied)
    with pytest.raises(IntegrityError):
        generate_serial_base_path(copied, family="LF", path_entity=1, years=1)

    path = generate_serial_base_path(
        generation_input, family="LF", path_entity=1, years=1
    )
    copied_path = copy.copy(path)
    assert not is_sealed_serial_base_path(copied_path)
    with pytest.raises(IntegrityError):
        aggregate_lf_log_returns(copied_path)
    with pytest.raises(ValueError):
        path.log_returns.setflags(write=True)
    with pytest.raises(ValueError):
        path.log_returns[0, 0] = 0.0


def test_builder_rejects_authority_config_input_and_artifact_mismatches() -> None:
    _, authority, config, calibration_input = _fixture()
    with pytest.raises(GenerationError, match="G3 evidence is evidence-only"):
        build_serial_generation_input(
            authority=authority.g3_evidence,  # type: ignore[arg-type]
            config=config,
            calibration_input=calibration_input,
        )
    unauthorized = copy.copy(authority)
    with pytest.raises(IntegrityError):
        _build_noncanonical_g5_serial_generation_input_for_testing(
            authority=unauthorized,
            config=config,
            calibration_input=calibration_input,
        )

    stale_reference = copy.copy(authority)
    object.__setattr__(
        stale_reference,
        "g3_evidence_fingerprint",
        "8" * 64,
    )
    with pytest.raises(IntegrityError):
        _build_noncanonical_g5_serial_generation_input_for_testing(
            authority=stale_reference,
            config=config,
            calibration_input=calibration_input,
        )

    with pytest.raises(IntegrityError):
        _build_noncanonical_g5_serial_generation_input_for_testing(
            authority=authority,
            config=load_config("configs/canonical.yaml"),
            calibration_input=calibration_input,
        )

    bad_full = replace(
        calibration_input.full,
        monthly_returns=np.array(calibration_input.full.monthly_returns, copy=True),
    )
    bad_input = replace(calibration_input, full=bad_full)
    with pytest.raises(IntegrityError):
        _build_noncanonical_g5_serial_generation_input_for_testing(
            authority=authority,
            config=config,
            calibration_input=bad_input,
        )

    tampered_artifact = copy.copy(authority.production_artifact)
    arrays = dict(tampered_artifact.arrays)
    arrays["monthly_source_states"] = _immutable(
        np.roll(arrays["monthly_source_states"], 1), np.int64
    )
    object.__setattr__(tampered_artifact, "arrays", MappingProxyType(arrays))
    bad_authority = copy.copy(authority)
    object.__setattr__(bad_authority, "production_artifact", tampered_artifact)
    with pytest.raises(IntegrityError):
        _build_noncanonical_g5_serial_generation_input_for_testing(
            authority=bad_authority,
            config=config,
            calibration_input=calibration_input,
        )


def test_lf_and_hf_aggregation_calendar_coverage_and_simple_identity() -> None:
    generation_input, _, _, _ = _fixture()
    lf = generate_serial_base_path(
        generation_input, family="LF", path_entity=1, years=3
    )
    lf_derived = aggregate_lf_log_returns(lf)
    assert is_sealed_lf_aggregated_returns(lf_derived)
    assert lf_derived.quarterly_log_returns.shape == (12, 3)
    assert lf_derived.annual_log_returns.shape == (3, 3)
    np.testing.assert_array_equal(
        lf_derived.quarterly_log_returns,
        lf.log_returns.reshape(3, 4, 3, 3).sum(axis=2).reshape(12, 3),
    )
    np.testing.assert_allclose(
        np.expm1(lf_derived.annual_log_returns),
        (1.0 + np.expm1(lf.log_returns.reshape(3, 12, 3))).prod(axis=1) - 1.0,
        rtol=0.0,
        atol=1e-10,
    )

    hf = generate_serial_base_path(
        generation_input, family="HF", path_entity=1, years=3
    )
    hf_derived = aggregate_hf_weekly_log_returns(hf)
    assert is_sealed_hf_weekly_returns(hf_derived)
    assert hf_derived.weekly_log_returns.shape == (156, 3)
    assert sum(np.diff(np.asarray([252 * week // 52 for week in range(53)]))) == 252
    daily_total = hf.log_returns.sum(axis=0)
    reassociation_tolerance = (
        16.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(daily_total))))
    )
    np.testing.assert_allclose(
        hf_derived.weekly_log_returns.sum(axis=0),
        daily_total,
        rtol=0.0,
        atol=reassociation_tolerance,
    )
    with pytest.raises(GenerationError):
        aggregate_lf_log_returns(hf)
    with pytest.raises(GenerationError):
        aggregate_hf_weekly_log_returns(lf)


def test_aggregate_outputs_are_immutable_and_copies_are_rejected() -> None:
    generation_input, _, _, _ = _fixture()
    lf = aggregate_lf_log_returns(
        generate_serial_base_path(generation_input, family="LF", path_entity=1, years=1)
    )
    hf = aggregate_hf_weekly_log_returns(
        generate_serial_base_path(generation_input, family="HF", path_entity=1, years=1)
    )
    with pytest.raises(ValueError):
        lf.annual_log_returns.setflags(write=True)
    with pytest.raises(ValueError):
        hf.weekly_log_returns[0, 0] = 1.0
    assert not is_sealed_lf_aggregated_returns(copy.copy(lf))
    assert not is_sealed_hf_weekly_returns(copy.copy(hf))


def test_nonfinite_offset_output_and_wrong_input_types_fail_closed() -> None:
    generation_input, _, _, _ = _fixture(neutral=True)
    copied = copy.copy(generation_input)
    huge = np.array(copied.monthly_kernel_offsets, copy=True)
    huge[1] = np.finfo(np.float64).max
    object.__setattr__(copied, "monthly_kernel_offsets", _immutable(huge, np.float64))
    object.__setattr__(copied, "_owner_id", id(copied))
    assert not is_sealed_serial_generation_input(copied)

    assert not is_sealed_serial_generation_input(object())
    assert not is_sealed_serial_base_path(object())
    assert not is_sealed_lf_aggregated_returns(object())
    assert not is_sealed_hf_weekly_returns(object())


def test_calibration_policy_and_artifact_fail_closed_guards() -> None:
    import prpg.simulation.serial as module

    _, authority, config, calibration_input = _fixture()
    with pytest.raises(GenerationError):
        module._validated_config(object())
    with pytest.raises(GenerationError):
        module._validate_calibration_input(object())
    with pytest.raises(IntegrityError):
        module._validate_calibration_input(
            replace(calibration_input, fingerprint="0" * 64)
        )
    with pytest.raises(IntegrityError):
        module._validate_calibration_source_array_hashes(
            replace(
                calibration_input,
                provenance=replace(calibration_input.provenance, source_files=()),
            )
        )
    changed_full = replace(
        calibration_input.full,
        monthly_returns=_immutable(
            np.roll(calibration_input.full.monthly_returns, 1, axis=0),
            np.float64,
        ),
    )
    with pytest.raises(IntegrityError):
        module._validate_calibration_source_array_hashes(
            replace(calibration_input, full=changed_full)
        )

    with pytest.raises(IntegrityError):
        module._validate_calibration_slice(
            replace(calibration_input.full, role="design"), "full", 217, 4_547
        )
    writable = np.array(calibration_input.full.monthly_returns, copy=True)
    with pytest.raises(IntegrityError):
        module._validate_calibration_slice(
            replace(calibration_input.full, monthly_returns=writable),
            "full",
            217,
            4_547,
        )
    nonfinite = np.array(calibration_input.full.monthly_returns, copy=True)
    nonfinite[0, 0] = np.nan
    with pytest.raises(IntegrityError):
        module._validate_calibration_slice(
            replace(
                calibration_input.full,
                monthly_returns=_immutable(nonfinite, np.float64),
            ),
            "full",
            217,
            4_547,
        )
    for field_name in ("monthly_continuation", "daily_continuation"):
        bad = np.array(getattr(calibration_input.full, field_name), copy=True)
        bad[0] = True
        with pytest.raises(IntegrityError):
            module._validate_calibration_slice(
                replace(
                    calibration_input.full,
                    **{field_name: _immutable(bad, np.bool_)},
                ),
                "full",
                217,
                4_547,
            )
    for field_name in ("monthly_period_ordinals", "daily_session_ns"):
        bad = np.array(getattr(calibration_input.full, field_name), copy=True)
        bad[1] = bad[0]
        with pytest.raises(IntegrityError):
            module._validate_calibration_slice(
                replace(
                    calibration_input.full,
                    **{field_name: _immutable(bad, np.int64)},
                ),
                "full",
                217,
                4_547,
            )

    with pytest.raises(IntegrityError):
        module._validated_policy(object())
    invalid_policy = replace(authority.generation_policy)
    object.__setattr__(invalid_policy, "monthly_block_length", 1)
    with pytest.raises(IntegrityError):
        module._validated_policy(invalid_policy)
    # G3 leaves both candidates unselected; the sealed G5 choice may be either
    # registered policy, but a neutral lambda outside the G3 grid fails closed.
    module._validate_policy_against_config(
        replace(
            authority.generation_policy,
            alpha_policy="kernel_mean_neutral",
            selected_neutral_lambda=0.5,
        ),
        config,
    )
    with pytest.raises(IntegrityError, match="outside the resolved G3 candidate"):
        module._validate_policy_against_config(
            replace(
                authority.generation_policy,
                alpha_policy="kernel_mean_neutral",
                selected_neutral_lambda=0.33,
            ),
            config,
        )

    artifact = authority.production_artifact
    missing_records = copy.copy(artifact)
    bad_reference = replace(
        missing_records.reference,
        manifest=MappingProxyType({"identity": MappingProxyType({})}),
    )
    object.__setattr__(
        missing_records,
        "reference",
        bad_reference,
    )
    with pytest.raises(IntegrityError):
        module._validate_live_artifact_arrays(missing_records)


def test_generation_input_and_path_integrity_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prpg.simulation.serial as module

    generation_input, _, _, _ = _fixture()

    def copied_input(**changes: object) -> SerialGenerationInput:
        result = copy.copy(generation_input)
        object.__setattr__(result, "_owner_id", id(result))
        for name, value in changes.items():
            object.__setattr__(result, name, value)
        return result

    with pytest.raises(IntegrityError):
        module._verify_generation_input(copied_input(config_fingerprint="bad"))
    with pytest.raises(IntegrityError):
        module._verify_generation_input(
            copied_input(array_fingerprints=MappingProxyType({}))
        )
    writable = np.array(generation_input.monthly_returns, copy=True)
    with pytest.raises(IntegrityError):
        module._verify_generation_input(copied_input(monthly_returns=writable))
    changed = _immutable(
        np.roll(generation_input.monthly_returns, 1, axis=0), np.float64
    )
    with pytest.raises(IntegrityError):
        module._verify_generation_input(copied_input(monthly_returns=changed))
    with pytest.raises(IntegrityError):
        module._verify_generation_input(copied_input(master_seed=99))

    monkeypatch.setattr(
        module, "_record_fingerprint", lambda *_: generation_input.fingerprint
    )
    with pytest.raises(IntegrityError):
        module._verify_generation_input(
            copied_input(
                transition_matrix=_immutable(np.eye(3), np.float64),
                array_fingerprints=MappingProxyType(
                    {
                        **generation_input.array_fingerprints,
                        "transition_matrix": module._array_fingerprint(
                            _immutable(np.eye(3), np.float64), "transition_matrix"
                        ),
                    }
                ),
            )
        )

    path = generate_serial_base_path(
        generation_input, family="LF", path_entity=1, years=1
    )

    def copied_path(**changes: object) -> Any:
        result = copy.copy(path)
        object.__setattr__(result, "_owner_id", id(result))
        for name, value in changes.items():
            object.__setattr__(result, name, value)
        return result

    with pytest.raises(IntegrityError):
        module._verify_base_path(copied_path(base_frequency="daily"))
    with pytest.raises(IntegrityError):
        module._verify_base_path(copied_path(restart_audit=object()))
    with pytest.raises(IntegrityError):
        module._verify_base_path(
            copied_path(log_returns=np.array(path.log_returns, copy=True))
        )
    with pytest.raises(IntegrityError):
        module._verify_base_path(
            copied_path(
                target_states=_immutable(np.roll(path.target_states, 1), np.int64)
            )
        )
    with pytest.raises(IntegrityError):
        module._verify_base_path(copied_path(state_uniforms_consumed=11))
    with pytest.raises(IntegrityError):
        module._verify_base_path(copied_path(fingerprint="0" * 64))


def test_low_level_draw_array_and_identity_guards() -> None:
    import prpg.simulation.serial as module

    for values in (object(), np.zeros(2), np.asarray([np.nan]), np.asarray([1.0])):
        with pytest.raises(GenerationError):
            module._uniform_array(values, 1, "draws")
    for values, dtype in (
        (np.asarray(1.0), np.float64),
        (np.asarray([], dtype=np.float64), np.float64),
        (np.asarray([1], dtype=np.int64), np.float64),
        (np.asarray([np.inf]), np.float64),
    ):
        with pytest.raises(IntegrityError):
            module._immutable_array(values, dtype=dtype, label="array")
    with pytest.raises(IntegrityError):
        module._canonical_json({"bad": float("nan")})


def test_aggregate_fingerprint_and_array_guards() -> None:
    import prpg.simulation.aggregation as module

    generation_input, _, _, _ = _fixture()
    lf = aggregate_lf_log_returns(
        generate_serial_base_path(generation_input, family="LF", path_entity=1, years=1)
    )
    hf = aggregate_hf_weekly_log_returns(
        generate_serial_base_path(generation_input, family="HF", path_entity=1, years=1)
    )
    bad_lf = copy.copy(lf)
    object.__setattr__(bad_lf, "_owner_id", id(bad_lf))
    object.__setattr__(bad_lf, "fingerprint", "0" * 64)
    with pytest.raises(IntegrityError):
        module._verify_lf_aggregate(bad_lf)
    bad_hf = copy.copy(hf)
    object.__setattr__(bad_hf, "_owner_id", id(bad_hf))
    object.__setattr__(bad_hf, "fingerprint", "0" * 64)
    with pytest.raises(IntegrityError):
        module._verify_hf_weekly(bad_hf)
    with pytest.raises(IntegrityError):
        module._verify_aggregate_arrays({"bad": np.asarray([np.nan])})


@pytest.fixture(scope="module")
def lean_qualification_fixture() -> tuple[
    LeanG5QualificationPermit,
    G5CandidateSerialGenerationInput,
    G5CandidateSerialGenerationInput,
    G5QualifiedProductionAuthority,
    PRPGConfig,
    CalibrationInput,
]:
    _, final, config, calibration_input = _fixture()
    permit = build_lean_g5_qualification_permit(
        g3_evidence=final.g3_evidence,
        generation_policy=final.generation_policy,
        development_decision_fingerprint=_fp("lean-g5-development-decision"),
    )
    qualification = build_lean_g5_qualification_serial_generation_input(
        permit=permit,
        config=config,
        calibration_input=calibration_input,
    )
    development = build_g5_candidate_serial_generation_input(
        authority=final.g3_evidence,
        generation_policy=final.generation_policy,
        candidate_stage="development",
        config=config,
        calibration_input=calibration_input,
    )
    return permit, qualification, development, final, config, calibration_input


def test_lean_g5_permit_binds_exact_g3_policy_contract_and_decision(
    lean_qualification_fixture: tuple[
        LeanG5QualificationPermit,
        G5CandidateSerialGenerationInput,
        G5CandidateSerialGenerationInput,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    permit, qualification, _, final, _, _ = lean_qualification_fixture

    assert is_lean_g5_qualification_permit(permit)
    assert verify_lean_g5_qualification_permit(permit) is permit
    assert permit.production_generation_authorized is False
    assert permit.contract_id == LEAN_G5_QUALIFICATION_CONTRACT_ID
    assert permit.g3_evidence is final.g3_evidence
    assert permit.g3_evidence_fingerprint == final.g3_evidence.reference.fingerprint
    assert permit.generation_policy == final.generation_policy
    assert permit.development_decision_fingerprint == _fp(
        "lean-g5-development-decision"
    )
    assert qualification.lean_qualification_permit is permit
    assert qualification.qualification_authority is None
    assert qualification.threshold_evidence_fingerprint is None
    assert qualification.threshold_registry_fingerprint is None


def test_lean_g5_qualification_uses_distinct_lf_and_hf_rng_domains(
    monkeypatch: pytest.MonkeyPatch,
    lean_qualification_fixture: tuple[
        LeanG5QualificationPermit,
        G5CandidateSerialGenerationInput,
        G5CandidateSerialGenerationInput,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    import prpg.simulation.serial as serial_module

    _, qualification, development, _, _, _ = lean_qualification_fixture
    real_uniforms = serial_module.uniforms
    observed: list[tuple[Domain, int]] = []

    def capture_key(key: Any, size: int) -> np.ndarray:
        observed.append((key.domain, key.family))
        return real_uniforms(key, size)

    monkeypatch.setattr(serial_module, "uniforms", capture_key)
    for source, expected_domain in (
        (development, Domain.DEVELOPMENT),
        (qualification, Domain.QUALIFICATION),
    ):
        for family, expected_family in (("LF", 1), ("HF", 2)):
            observed.clear()
            path = generate_g5_candidate_serial_base_path(
                source, family=cast(Any, family), path_entity=1
            )
            assert path.rng_domain is expected_domain
            assert observed == [(expected_domain, expected_family)] * 3


def test_lean_g5_qualification_is_deterministic_and_exactly_bounded(
    lean_qualification_fixture: tuple[
        LeanG5QualificationPermit,
        G5CandidateSerialGenerationInput,
        G5CandidateSerialGenerationInput,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    permit, qualification, _, final, config, calibration_input = (
        lean_qualification_fixture
    )
    replay_permit = build_lean_g5_qualification_permit(
        g3_evidence=final.g3_evidence,
        generation_policy=final.generation_policy,
        development_decision_fingerprint=_fp("lean-g5-development-decision"),
    )
    replay_input = build_lean_g5_qualification_serial_generation_input(
        permit=replay_permit,
        config=config,
        calibration_input=calibration_input,
    )

    assert replay_permit.fingerprint == permit.fingerprint
    assert replay_input.fingerprint == qualification.fingerprint
    assert qualification.rng_domain is Domain.QUALIFICATION
    assert qualification.policy_scientific_version == g5_policy_scientific_version(
        final.generation_policy
    )
    assert (qualification.maximum_lf_paths, qualification.maximum_hf_paths) == (
        500,
        200,
    )
    for family, maximum in (("LF", 500), ("HF", 200)):
        first = generate_g5_candidate_serial_base_path(
            qualification, family=cast(Any, family), path_entity=maximum
        )
        replay = generate_g5_candidate_serial_base_path(
            replay_input, family=cast(Any, family), path_entity=maximum
        )
        assert first.fingerprint == replay.fingerprint
        np.testing.assert_array_equal(first.log_returns, replay.log_returns)
        with pytest.raises(GenerationError, match="registered range"):
            generate_g5_candidate_serial_base_path(
                qualification,
                family=cast(Any, family),
                path_entity=maximum + 1,
            )


def test_lean_g5_qualification_fails_closed_on_policy_and_identity_tampering(
    lean_qualification_fixture: tuple[
        LeanG5QualificationPermit,
        G5CandidateSerialGenerationInput,
        G5CandidateSerialGenerationInput,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    permit, _, _, final, config, calibration_input = lean_qualification_fixture
    outside_g3_policy = replace(
        final.generation_policy,
        monthly_block_length=final.generation_policy.monthly_block_length + 1,
    )
    with pytest.raises(ModelError, match="outside the verified G3 candidate set"):
        build_lean_g5_qualification_permit(
            g3_evidence=final.g3_evidence,
            generation_policy=outside_g3_policy,
            development_decision_fingerprint=_fp("wrong-policy-decision"),
        )
    neutral_without_typed_selection = replace(
        final.generation_policy,
        alpha_policy="kernel_mean_neutral",
        selected_neutral_lambda=0.25,
    )
    with pytest.raises(ModelError, match="typed development selection"):
        build_lean_g5_qualification_permit(
            g3_evidence=final.g3_evidence,
            generation_policy=neutral_without_typed_selection,
            development_decision_fingerprint=_fp("untyped-neutral-decision"),
        )
    with pytest.raises(GenerationError, match="requires its public builder"):
        build_g5_candidate_serial_generation_input(
            authority=cast(Any, permit),
            generation_policy=permit.generation_policy,
            candidate_stage="qualification",
            config=config,
            calibration_input=calibration_input,
        )

    tampered_permit = build_lean_g5_qualification_permit(
        g3_evidence=final.g3_evidence,
        generation_policy=final.generation_policy,
        development_decision_fingerprint=_fp("tampered-permit-decision"),
    )
    object.__setattr__(tampered_permit, "g3_evidence_fingerprint", "0" * 64)
    with pytest.raises(IntegrityError, match="permit identity changed"):
        verify_lean_g5_qualification_permit(tampered_permit)

    intact_permit = build_lean_g5_qualification_permit(
        g3_evidence=final.g3_evidence,
        generation_policy=final.generation_policy,
        development_decision_fingerprint=_fp("tampered-input-decision"),
    )
    tampered_input = build_lean_g5_qualification_serial_generation_input(
        permit=intact_permit,
        config=config,
        calibration_input=calibration_input,
    )
    object.__setattr__(tampered_input, "maximum_lf_paths", 499)
    with pytest.raises(IntegrityError, match="candidate input identity changed"):
        verify_g5_candidate_serial_generation_input(tampered_input)


def _passing_lean_g5_report(
    policy: ScientificGenerationPolicy,
    geometry: LeanG5Geometry = LEAN_G5_PRODUCTION_GEOMETRY,
) -> LeanG5AssessmentReport:
    critical = HigherEmpiricalQuantile(
        0.95,
        geometry.resamples,
        "higher",
        geometry.resamples - 1,
        geometry.resamples,
        1.0,
    )
    zeros12 = (0.0,) * 12
    truths12 = (True,) * 12
    alpha = LeanG5AlphaAudit(
        geometry=geometry,
        equivalence=AlphaEquivalenceAudit(
            geometry,
            0.90,
            0.10,
            critical,
            LEAN_G5_COMPONENT_NAMES,
            zeros12,
            (0.01,) * 12,
            (-0.02,) * 12,
            (0.02,) * 12,
            truths12,
            True,
        ),
        leakage_gap=LeakageGapAudit(
            geometry,
            0.95,
            0.10,
            critical,
            LEAN_G5_COMPONENT_NAMES,
            zeros12,
            zeros12,
            zeros12,
            (0.01,) * 12,
            (0.02,) * 12,
            truths12,
            True,
        ),
        inverse_volatility=InverseVolatilityGapAudit(
            geometry,
            0.95,
            0.20,
            critical,
            ("monthly", "daily"),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.01, 0.01),
            (0.02, 0.02),
            (True, True),
            True,
        ),
        passed=True,
    )
    comparisons = tuple(
        LeanG5GroupedComparison(
            family=name,
            endpoint_count=3,
            status="pass",
            observed_statistic=0.0,
            critical_value=critical,
            p_value=PlusOnePValue(0.0, geometry.resamples, geometry.resamples, 1.0),
            reason="inside empirical null",
        )
        for name in ("G5-A", "G5-B", "G5-C", "G5-D")
    )
    holm = HolmAudit(
        familywise_alpha=0.05,
        decisions=tuple(
            HolmDecision(name, 1.0, 1.0, index, 0.05, False)
            for index, name in enumerate(("G5-A", "G5-B", "G5-C", "G5-D"), start=1)
        ),
        any_rejected=False,
    )
    transition = (
        (0.25, 0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.25),
    )
    state_behavior = LeanG5StateBehavior(
        schema_id=LEAN_G5_STATE_BEHAVIOR_SCHEMA_ID,
        path_count=geometry.monthly_paths,
        observations=2_000,
        state_count=4,
        supported_states=(True, True, True, True),
        occupancy_counts=(500, 500, 500, 500),
        occupancy=(0.25, 0.25, 0.25, 0.25),
        departure_counts=(499, 499, 499, 499),
        transition_probabilities=transition,
        episode_counts=(10, 10, 10, 10),
        mean_dwell=(5.0, 5.0, 5.0, 5.0),
        median_dwell=(5.0, 5.0, 5.0, 5.0),
        one_period_spell_fraction=(0.1, 0.1, 0.1, 0.1),
        state_conditional_log_mean=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )
    state_audit = LeanG5StateAudit(
        state_behavior, state_behavior, state_behavior, state_behavior, 88
    )

    def diversity(frequency: str, paths: int, periods: int) -> LeanG5ReturnDiversity:
        return LeanG5ReturnDiversity(
            schema_id=LEAN_G5_DIVERSITY_SCHEMA_ID,
            frequency=cast(Any, frequency),
            path_count=paths,
            periods_per_path=periods,
            unique_complete_paths=paths,
            complete_path_duplicate_copies=0,
            complete_path_duplicate_groups=0,
            maximum_complete_path_multiplicity=1,
            subsequence_window=6 if frequency == "monthly" else 63,
            subsequence_windows=paths * (periods - 5),
            unique_subsequences=paths * (periods - 5),
            repeated_subsequence_copies=0,
            repeated_subsequence_groups=0,
            maximum_subsequence_multiplicity=1,
        )

    monthly_diversity = diversity("monthly", geometry.monthly_paths, 217)
    daily_diversity = diversity("daily", geometry.daily_paths, 4_547)
    diversity_audit = LeanG5DiversityAudit(
        monthly_diversity,
        daily_diversity,
        monthly_diversity,
        daily_diversity,
        (("monthly", 0.0, 0.0, 1, 1), ("daily", 0.0, 0.0, 1, 1)),
        True,
        "pass",
        "no excess repetition",
    )
    report = LeanG5AssessmentReport(
        schema_id=LEAN_G5_ASSESSMENT_SCHEMA_ID,
        geometry=geometry,
        generation_policy_fingerprint=_json_fingerprint(policy.as_dict()),
        development_bundle_fingerprint=_fp("lean-development-bundle"),
        qualification_bundle_fingerprint=_fp("lean-qualification-bundle"),
        control_bank_fingerprints=tuple(
            (role, _fp(f"lean-control-{role}")) for role in ("A", "B", "C")
        ),
        control_mean_block_lengths=(
            ("monthly", float(policy.monthly_block_length)),
            ("daily", float(policy.daily_block_length)),
        ),
        model_fingerprints=(
            ("strategy.monthly", _fp("lean-strategy-monthly")),
            ("strategy.daily", _fp("lean-strategy-daily")),
            ("inverse_volatility.monthly", _fp("lean-inverse-monthly")),
            ("inverse_volatility.daily", _fp("lean-inverse-daily")),
        ),
        resample_indices_fingerprint=_fp("lean-resample-indices"),
        alpha=alpha,
        grouped_comparisons=comparisons,
        holm=holm,
        states=state_audit,
        diversity=diversity_audit,
        decision="pass",
        reasons=("PASS: all implemented lean G5 assessment controls passed",),
        generation_authorized=False,
        fingerprint="0" * 64,
    )
    return replace(
        report,
        fingerprint=_lean_authority_fingerprint(_assessment_identity(report)),
    )


def _passing_lean_g5_verification_receipt(
    policy: ScientificGenerationPolicy,
) -> Any:
    return build_lean_g5_verification_receipt(
        pilot_assessment=_passing_lean_g5_report(policy, LEAN_G5_PILOT_GEOMETRY),
        one_worker_fingerprint=_fp("lean-pilot-parity"),
        nine_worker_fingerprint=_fp("lean-pilot-parity"),
        worker_parity_passed=True,
        resource_projection_fingerprint=_fp("lean-resource-projection"),
        resource_projection_passed=True,
        pristine_assessment_fingerprint=_fp("lean-pristine-assessment"),
        pristine_passed=True,
        defect_results=tuple((name, True) for name in LEAN_G5_REQUIRED_DEFECTS),
    )


@pytest.fixture(scope="module")
def lean_production_fixture() -> tuple[
    LeanG5ProductionAuthority,
    LeanG5AssessmentReport,
    LeanG5QualificationPermit,
    G5QualifiedProductionAuthority,
    PRPGConfig,
    CalibrationInput,
]:
    _, legacy, config, calibration_input = _fixture()
    permit = build_lean_g5_qualification_permit(
        g3_evidence=legacy.g3_evidence,
        generation_policy=legacy.generation_policy,
        development_decision_fingerprint=_fp("lean-final-development-decision"),
    )
    report = _passing_lean_g5_report(legacy.generation_policy)
    verification = _passing_lean_g5_verification_receipt(legacy.generation_policy)
    authority = mint_lean_g5_production_authority(
        g3_evidence=legacy.g3_evidence,
        generation_policy=legacy.generation_policy,
        assessment=report,
        qualification_permit=permit,
        verification_receipt=verification,
        downstream_source_code_fingerprint=_fp("lean-downstream-source"),
        downstream_dependency_lock_fingerprint=_fp("lean-downstream-lock"),
    )
    return authority, report, permit, legacy, config, calibration_input


def test_lean_g5_final_authority_mints_only_from_canonical_complete_pass(
    lean_production_fixture: tuple[
        LeanG5ProductionAuthority,
        LeanG5AssessmentReport,
        LeanG5QualificationPermit,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    authority, report, permit, legacy, _, _ = lean_production_fixture
    assert authority.generation_authorized is True
    assert is_lean_g5_production_authority(authority)
    assert authority.g3_evidence is legacy.g3_evidence
    assert authority.assessment_fingerprint == report.fingerprint
    assert authority.qualification_permit_fingerprint == permit.fingerprint

    non_pass = replace(report, decision="non_pass", fingerprint="0" * 64)
    non_pass = replace(
        non_pass,
        fingerprint=_lean_authority_fingerprint(_assessment_identity(non_pass)),
    )
    with pytest.raises(ValidationError, match="did not pass every release control"):
        mint_lean_g5_production_authority(
            g3_evidence=legacy.g3_evidence,
            generation_policy=legacy.generation_policy,
            assessment=non_pass,
            qualification_permit=permit,
            verification_receipt=_passing_lean_g5_verification_receipt(
                legacy.generation_policy
            ),
            downstream_source_code_fingerprint=_fp("lean-downstream-source"),
            downstream_dependency_lock_fingerprint=_fp("lean-downstream-lock"),
        )

    with pytest.raises(ModelError, match="verification receipt"):
        mint_lean_g5_production_authority(
            g3_evidence=legacy.g3_evidence,
            generation_policy=legacy.generation_policy,
            assessment=report,
            qualification_permit=permit,
            verification_receipt=cast(Any, None),
            downstream_source_code_fingerprint=_fp("lean-downstream-source"),
            downstream_dependency_lock_fingerprint=_fp("lean-downstream-lock"),
        )

    mismatched = replace(
        report, generation_policy_fingerprint="0" * 64, fingerprint="0" * 64
    )
    mismatched = replace(
        mismatched,
        fingerprint=_lean_authority_fingerprint(_assessment_identity(mismatched)),
    )
    with pytest.raises(ValidationError, match="report/permit/verification policy"):
        mint_lean_g5_production_authority(
            g3_evidence=legacy.g3_evidence,
            generation_policy=legacy.generation_policy,
            assessment=mismatched,
            qualification_permit=permit,
            verification_receipt=_passing_lean_g5_verification_receipt(
                legacy.generation_policy
            ),
            downstream_source_code_fingerprint=_fp("lean-downstream-source"),
            downstream_dependency_lock_fingerprint=_fp("lean-downstream-lock"),
        )

    with pytest.raises(ValidationError, match="bounded verification"):
        build_lean_g5_verification_receipt(
            pilot_assessment=_passing_lean_g5_report(
                legacy.generation_policy, LEAN_G5_PILOT_GEOMETRY
            ),
            one_worker_fingerprint=_fp("lean-pilot-parity"),
            nine_worker_fingerprint=_fp("lean-pilot-parity"),
            worker_parity_passed=True,
            resource_projection_fingerprint=_fp("lean-resource-projection"),
            resource_projection_passed=True,
            pristine_assessment_fingerprint=_fp("lean-pristine-assessment"),
            pristine_passed=True,
            defect_results=tuple(
                (name, True) for name in LEAN_G5_REQUIRED_DEFECTS[:-1]
            ),
        )

    copied = copy.copy(authority)
    object.__setattr__(copied, "_owner_id", id(copied))
    with pytest.raises(ModelError, match="loader/mint provenance"):
        verify_lean_g5_production_authority(copied)


def test_lean_g5_final_authority_store_load_and_tamper_rejection(
    tmp_path: Path,
    lean_production_fixture: tuple[
        LeanG5ProductionAuthority,
        LeanG5AssessmentReport,
        LeanG5QualificationPermit,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    authority, _, _, legacy, _, _ = lean_production_fixture
    store = LeanG5ProductionAuthorityStore(tmp_path)
    path = store.store(authority)
    loaded = store.load(authority.fingerprint, g3_evidence=legacy.g3_evidence)
    assert loaded.fingerprint == authority.fingerprint
    assert loaded.g3_evidence is legacy.g3_evidence
    assert is_lean_g5_production_authority(loaded)

    manifest = json.loads(path.read_bytes())
    manifest["identity"]["assessment"]["decision"] = "non_pass"
    path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(IntegrityError, match="fingerprint"):
        store.load(authority.fingerprint, g3_evidence=legacy.g3_evidence)


def test_lean_g5_authority_mints_canonical_serial_input_and_path(
    lean_production_fixture: tuple[
        LeanG5ProductionAuthority,
        LeanG5AssessmentReport,
        LeanG5QualificationPermit,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    authority, report, permit, _, config, calibration_input = lean_production_fixture
    generation_input = build_serial_generation_input(
        authority=authority,
        config=config,
        calibration_input=calibration_input,
    )
    assert generation_input.g5_authority_schema_id == (
        LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID
    )
    assert generation_input.g5_threshold_registry_fingerprint == permit.fingerprint
    assert generation_input.g5_qualification_results_fingerprint == report.fingerprint
    assert generation_input.g5_meta_validation_fingerprint is None
    assert is_sealed_serial_generation_input(generation_input)
    path = generate_configured_serial_base_path(
        generation_input, family="LF", path_entity=1
    )
    assert path.generation_authorized is True
    assert is_sealed_serial_base_path(path)


def test_production_loader_prefers_durable_lean_g5_authority(
    monkeypatch: pytest.MonkeyPatch,
    lean_production_fixture: tuple[
        LeanG5ProductionAuthority,
        LeanG5AssessmentReport,
        LeanG5QualificationPermit,
        G5QualifiedProductionAuthority,
        PRPGConfig,
        CalibrationInput,
    ],
) -> None:
    import prpg.simulation.production as production_module

    authority, _, _, legacy, config, calibration_input = lean_production_fixture
    loader = production_module.ProductionLoaderSpec(
        project_root=str(Path.cwd()),
        config_path=str(Path.cwd() / "configs" / "smoke.yaml"),
        processed_data_fingerprint=(
            calibration_input.provenance.processed_data_fingerprint
        ),
        g3_evidence_fingerprint=legacy.g3_evidence.reference.fingerprint,
        g5_authority_fingerprint=authority.fingerprint,
        config_fingerprint=config.fingerprint(),
    )

    class _G3Store:
        def load(self, *args: object, **kwargs: object) -> LoadedG3EvidenceAuthority:
            return legacy.g3_evidence

    class _LeanStore:
        def contains(self, fingerprint: str) -> bool:
            return fingerprint == authority.fingerprint

        def load(self, *args: object, **kwargs: object) -> LeanG5ProductionAuthority:
            return authority

    monkeypatch.setattr(
        production_module, "ProcessedSnapshotStore", lambda root: object()
    )
    monkeypatch.setattr(
        production_module,
        "load_calibration_input",
        lambda *args, **kwargs: calibration_input,
    )
    monkeypatch.setattr(production_module, "G3EvidenceStore", lambda root: _G3Store())
    monkeypatch.setattr(production_module, "CalibrationRunStore", lambda root: object())
    monkeypatch.setattr(
        production_module, "CalibrationCheckpointStore", lambda root: object()
    )
    monkeypatch.setattr(production_module, "ModelArtifactStore", lambda root: object())
    monkeypatch.setattr(
        production_module,
        "LeanG5ProductionAuthorityStore",
        lambda root: _LeanStore(),
    )
    monkeypatch.setattr(
        production_module,
        "G5EvidenceStore",
        lambda root: pytest.fail("legacy G5 store should not be loaded"),
    )

    result = production_module.load_production_generation_input(loader)
    assert is_sealed_serial_generation_input(result)
    assert result.g5_authority_schema_id == LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID
