"""Pure, deterministic Phase-4 serial return-path generation.

This module is the first consumer of qualified G5 authority.  It deliberately
does not load stores, write files, choose workers, or understand shards.  A
caller must first mint :class:`SerialGenerationInput` from a freshly reverified
G5 authority, the exact resolved configuration, and the matching immutable
calibration input.  The minting boundary detaches every numerical dependency
onto a ``bytes`` owner and binds it into a process-local capability.

The path routine then implements Sections 8.3 and 8.5 literally: one keyed
state uniform per model month, one keyed restart uniform and source-rank
uniform per base observation, a stationary Markov chain, a non-circular
state/gap-conditioned source trace, synchronized three-vector copying, and at
most one declared deterministic kernel offset.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypeGuard

import numpy as np
import numpy.typing as npt

from prpg.config import PRPGConfig
from prpg.errors import GenerationError, IntegrityError, ModelError
from prpg.model.bootstrap import (
    BootstrapTraceReason,
    conditioned_source_index_trace,
    simulate_target_states_from_uniforms,
)
from prpg.model.g3_evidence_store import (
    LoadedG3EvidenceAuthority,
    verify_loaded_g3_evidence,
)
from prpg.model.g5_generation_authority import (
    G5_DEVELOPMENT_HF_PATH_COUNT,
    G5_DEVELOPMENT_LF_PATH_COUNT,
    G5_GENERATION_AUTHORITY_SCHEMA_ID,
    G5_QUALIFICATION_HF_PATH_COUNT,
    G5_QUALIFICATION_LF_PATH_COUNT,
    G5QualificationGenerationAuthority,
    G5QualifiedProductionAuthority,
    g5_policy_scientific_version,
    is_g5_noncanonical_test_fixture,
    verify_g5_qualification_generation_authority,
    verify_g5_qualified_production_authority,
)
from prpg.model.hmm import validate_markov_chain
from prpg.model.input import ASSET_COLUMNS, CalibrationInput, CalibrationSlice
from prpg.model.lean_g5_qualification import (
    LeanG5QualificationPermit,
    verify_lean_g5_qualification_permit,
)
from prpg.model.scientific_artifact import (
    LoadedScientificModelArtifact,
    is_verified_scientific_model_artifact,
)
from prpg.model.scientific_policy import (
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
    generation_policy_from_mapping,
)
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    UINT32_MAX,
    Domain,
    Family,
    RNGKey,
    Stage,
    uniforms,
)

if TYPE_CHECKING:
    from prpg.model.lean_g5_generation_authority import LeanG5ProductionAuthority

SERIAL_GENERATION_INPUT_SCHEMA_VERSION: Final = 3
SERIAL_BASE_PATH_SCHEMA_VERSION: Final = 2
SERIAL_GENERATOR_VERSION: Final = "prpg-serial-generator-v1"
ASSET_COUNT: Final = len(ASSET_COLUMNS)
MONTHS_PER_YEAR: Final = 12
SESSIONS_PER_MONTH: Final = 21
SESSIONS_PER_YEAR: Final = 252
_CALIBRATION_ARRAY_PATHS: Final = MappingProxyType(
    {
        "arrays/monthly_returns.npy": "monthly_returns",
        "arrays/daily_returns.npy": "daily_returns",
        "arrays/macro_features.npy": "macro_features",
        "arrays/monthly_period_ordinals.npy": "monthly_period_ordinals",
        "arrays/daily_session_ns.npy": "daily_session_ns",
        "arrays/monthly_continuation.npy": "monthly_continuation",
        "arrays/daily_continuation.npy": "daily_continuation",
    }
)

PathFamily: TypeAlias = Literal["LF", "HF"]
BaseFrequency: TypeAlias = Literal["monthly", "daily"]
G5CandidateStage: TypeAlias = Literal["development", "qualification"]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_INPUT_SEAL = object()
_REHEARSAL_INPUT_SEAL = object()
_BASE_PATH_SEAL = object()
_REHEARSAL_BASE_PATH_SEAL = object()
_REHEARSAL_FACTORY_CAPABILITY = object()
_G5_CANDIDATE_INPUT_SEAL = object()
_G5_CANDIDATE_PATH_SEAL = object()
_G5_NONCANONICAL_SERIAL_TEST_CAPABILITY = object()
_LEAN_G5_QUALIFICATION_FACTORY_CAPABILITY = object()

G5_CANDIDATE_INPUT_SCHEMA_ID: Final = "prpg-g5-candidate-serial-input-v2"
G5_CANDIDATE_INPUT_SCHEMA_VERSION: Final = 2
G5_CANDIDATE_PATH_SCHEMA_ID: Final = "prpg-g5-candidate-serial-path-v1"
G5_CANDIDATE_PATH_SCHEMA_VERSION: Final = 1
G5_META_SERIAL_INPUT_SCHEMA_ID: Final = "prpg-g5-meta-serial-input-v1"
G5_META_SERIAL_INPUT_SCHEMA_VERSION: Final = 1


class RestartReason(StrEnum):
    """Stable public restart-audit vocabulary for one generated path."""

    FIRST_OBSERVATION = "first_observation"
    TARGET_STATE_TRANSITION = "target_state_transition"
    SOURCE_END = "source_end"
    SOURCE_GAP = "source_gap"
    SOURCE_STATE_MISMATCH = "source_state_mismatch"
    RANDOM_RESTART = "random_restart"
    CONTINUATION = "continuation"


@dataclass(frozen=True, slots=True)
class RestartAudit:
    """Immutable branch audit with fixed stream-consumption counts."""

    restart_flags: BoolArray
    reasons: tuple[RestartReason, ...]
    restart_uniform_used: BoolArray
    source_rank_uniform_used: BoolArray
    restart_uniforms_consumed: int
    source_rank_uniforms_consumed: int


@dataclass(frozen=True, slots=True, init=False)
class SerialGenerationInput:
    """Process-bound, integrity-rechecking numerical input to serial generation.

    Ordinary construction is disabled.  ``_owner_id`` makes shallow copies
    unusable, while ``fingerprint`` detects field replacement and every array
    is backed by immutable bytes.  These are correctness boundaries rather
    than a hostile-code security sandbox.
    """

    schema_version: int
    generator_version: str
    execution_mode: Literal["canonical", "rehearsal"]
    generation_authorized: bool
    rehearsal_scope_fingerprint: str | None
    config_fingerprint: str
    processed_data_fingerprint: str
    calibration_input_fingerprint: str
    g3_bundle_fingerprint: str
    durable_evidence_fingerprint: str
    pair_receipt_fingerprint: str
    g5_authority_schema_id: str | None
    g5_authority_fingerprint: str | None
    g5_threshold_registry_fingerprint: str | None
    g5_qualification_results_fingerprint: str | None
    g5_meta_validation_fingerprint: str | None
    production_artifact_fingerprint: str
    core_model_fingerprint: str
    parameter_set_fingerprint: str
    master_seed: int
    scientific_version: int
    selected_k: int
    configured_lf_paths: int
    configured_hf_paths: int
    configured_years: int
    generation_policy: ScientificGenerationPolicy
    stationary_distribution: FloatArray
    transition_matrix: FloatArray
    monthly_returns: FloatArray
    daily_returns: FloatArray
    monthly_source_states: IntArray
    daily_source_states: IntArray
    monthly_continuation_links: BoolArray
    daily_continuation_links: BoolArray
    monthly_kernel_lambdas: FloatArray
    daily_kernel_lambdas: FloatArray
    monthly_kernel_offsets: FloatArray
    daily_kernel_offsets: FloatArray
    array_fingerprints: Mapping[str, str]
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("SerialGenerationInput is created only by its authority mint")


@dataclass(frozen=True, slots=True, init=False)
class RehearsalSerialGenerationInput(SerialGenerationInput):
    """Noncanonical serial-engine input that can never enter storage/release."""

    def __init__(self) -> None:
        raise TypeError(
            "RehearsalSerialGenerationInput is created only by the test factory"
        )


@dataclass(frozen=True, slots=True, init=False)
class SerialBasePath:
    """One immutable base-frequency path plus hidden generation audit."""

    schema_version: int
    generator_version: str
    execution_mode: Literal["canonical", "rehearsal"]
    generation_authorized: bool
    generation_input_fingerprint: str
    family: PathFamily
    base_frequency: BaseFrequency
    path_entity: int
    years: int
    log_returns: FloatArray
    target_monthly_states: IntArray
    target_states: IntArray
    source_indices: IntArray
    restart_audit: RestartAudit
    state_uniforms_consumed: int
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("SerialBasePath is created only by the serial generator")


@dataclass(frozen=True, slots=True, init=False)
class RehearsalSerialBasePath(SerialBasePath):
    """Noncanonical base path rejected by aggregation, storage, and release."""

    def __init__(self) -> None:
        raise TypeError(
            "RehearsalSerialBasePath is created only by the rehearsal generator"
        )


G4ReferenceGenerationInput: TypeAlias = RehearsalSerialGenerationInput
G4ReferenceBasePath: TypeAlias = RehearsalSerialBasePath


@dataclass(frozen=True, slots=True, init=False)
class G5CandidateSerialGenerationInput:
    """One policy/stage input that is permanently nonauthorizing."""

    schema_id: str
    schema_version: int
    generation_authorized: Literal[False]
    candidate_stage: G5CandidateStage
    rng_domain: Domain
    g3_evidence: LoadedG3EvidenceAuthority
    qualification_authority: G5QualificationGenerationAuthority | None
    qualification_authority_fingerprint: str | None
    lean_qualification_permit: LeanG5QualificationPermit | None
    lean_qualification_permit_fingerprint: str | None
    g3_evidence_fingerprint: str
    threshold_evidence_fingerprint: str | None
    threshold_registry_fingerprint: str | None
    generation_policy: ScientificGenerationPolicy
    generation_policy_fingerprint: str
    policy_scientific_version: int
    maximum_lf_paths: int
    maximum_hf_paths: int
    configured_years: int
    numerical_input: RehearsalSerialGenerationInput
    numerical_input_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5CandidateSerialGenerationInput is created only by its public builder"
        )


@dataclass(frozen=True, slots=True)
class G5MetaSerialNumericalInput:
    """Pickle-safe, explicitly nonauthorizing view of the serial numerics.

    Purpose-12 workers cannot use the process-bound rehearsal input under the
    spawn multiprocessing policy.  This compact view contains only the exact
    numerical fields consumed by the existing conditioned serial engine.  Its
    content fingerprint is rechecked in every worker; it grants no production
    or qualification authority and cannot enter a canonical storage surface.
    """

    schema_id: str
    schema_version: int
    generation_authorized: Literal[False]
    source_generation_input_fingerprint: str
    master_seed: int
    scientific_version: int
    selected_k: int
    configured_years: int
    generation_policy: ScientificGenerationPolicy
    stationary_distribution: FloatArray
    transition_matrix: FloatArray
    monthly_returns: FloatArray
    daily_returns: FloatArray
    monthly_source_states: IntArray
    daily_source_states: IntArray
    monthly_continuation_links: BoolArray
    daily_continuation_links: BoolArray
    monthly_kernel_lambdas: FloatArray
    daily_kernel_lambdas: FloatArray
    monthly_kernel_offsets: FloatArray
    daily_kernel_offsets: FloatArray
    array_fingerprints: tuple[tuple[str, str], ...]
    fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class G5CandidateSerialBasePath:
    """Bounded candidate path rejected by every production consumer."""

    schema_id: str
    schema_version: int
    generation_authorized: Literal[False]
    candidate_stage: G5CandidateStage
    rng_domain: Domain
    generation_input: G5CandidateSerialGenerationInput
    generation_input_fingerprint: str
    generation_policy_fingerprint: str
    family: PathFamily
    base_frequency: BaseFrequency
    path_entity: int
    years: int
    base_path: RehearsalSerialBasePath
    base_path_fingerprint: str
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError(
            "G5CandidateSerialBasePath is created only by its public generator"
        )

    @property
    def log_returns(self) -> FloatArray:
        """Expose immutable base-frequency values for G5 statistics."""

        return self.base_path.log_returns

    @property
    def target_monthly_states(self) -> IntArray:
        return self.base_path.target_monthly_states

    @property
    def target_states(self) -> IntArray:
        return self.base_path.target_states

    @property
    def source_indices(self) -> IntArray:
        return self.base_path.source_indices


_G5_CANDIDATE_INPUTS: dict[int, G5CandidateSerialGenerationInput] = {}
_G5_CANDIDATE_PATHS: dict[int, G5CandidateSerialBasePath] = {}


def build_serial_generation_input(
    *,
    authority: G5QualifiedProductionAuthority | LeanG5ProductionAuthority,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> SerialGenerationInput:
    """Mint the only accepted input to the serial generator.

    The function performs no random draws and no I/O.  The supplied authority
    must be an original live-valid legacy or lean G5 qualification; its exact
    G3 evidence, production artifact, selected policy, and calibration arrays
    are rechecked in memory before detached immutable copies are made.
    """

    result = _build_serial_generation_input_core(
        authority=authority,
        config=config,
        calibration_input=calibration_input,
        rehearsal_policy=None,
        rehearsal_scope_fingerprint=None,
        rehearsal_capability=None,
        noncanonical_g5_test_capability=None,
    )
    if type(result) is not SerialGenerationInput:  # pragma: no cover - invariant
        raise AssertionError("canonical serial mint returned a rehearsal input")
    return result


def build_practical_k4_serial_generation_input(
    *,
    authority: object,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> SerialGenerationInput:
    """Mint a canonical input from the compact owner-approved K=4 authority.

    This delivery path intentionally does not require a passed G3/G5 evidence
    graph.  It binds the exact reviewed design-period labels and transition
    matrix to the verified design return pools, then reuses the ordinary
    deterministic serial engine unchanged.
    """

    from prpg.model.practical_k4 import (
        OWNER_APPROVED_MASTER_SEED,
        OWNER_APPROVED_SCIENTIFIC_VERSION,
        PRACTICAL_K4_AUTHORITY_SCHEMA_ID,
        PracticalK4Artifact,
        validate_practical_k4_calibration_binding,
        verify_practical_k4_artifact,
    )

    if not isinstance(authority, PracticalK4Artifact):
        raise GenerationError("practical K=4 serial mint requires its typed authority")
    artifact = verify_practical_k4_artifact(authority)
    checked_config = _validated_config(config)
    _validate_calibration_input(calibration_input)
    design = validate_practical_k4_calibration_binding(artifact, calibration_input)
    if (
        checked_config.run.master_seed != OWNER_APPROVED_MASTER_SEED
        or checked_config.simulation.scientific_version
        != OWNER_APPROVED_SCIENTIFIC_VERSION
        or checked_config.model.fixed_regime_count != 4
        or checked_config.model.regime_selection_policy != "owner_fixed_k4_v3"
        or checked_config.model.initial_state != "stationary"
    ):
        raise IntegrityError(
            "configuration is incompatible with owner-approved practical K=4"
        )

    policy = artifact.generation_policy
    chain = validate_markov_chain(artifact.transition_matrix)
    zero_offsets = np.zeros(
        (len(policy.neutral_lambda_grid), 4, ASSET_COUNT), dtype=np.float64
    )
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "stationary_distribution": _immutable_array(
            chain.stationary_distribution,
            dtype=np.float64,
            label="practical stationary distribution",
        ),
        "transition_matrix": _immutable_array(
            artifact.transition_matrix,
            dtype=np.float64,
            label="practical transition matrix",
        ),
        "monthly_returns": _immutable_array(
            design.monthly_returns,
            dtype=np.float64,
            label="practical monthly design returns",
        ),
        "daily_returns": _immutable_array(
            design.daily_returns,
            dtype=np.float64,
            label="practical daily design returns",
        ),
        "monthly_source_states": _immutable_array(
            artifact.monthly_source_states,
            dtype=np.int64,
            label="practical monthly states",
        ),
        "daily_source_states": _immutable_array(
            artifact.daily_source_states,
            dtype=np.int64,
            label="practical daily states",
        ),
        "monthly_continuation_links": _immutable_array(
            design.monthly_continuation[1:],
            dtype=np.bool_,
            label="practical monthly continuation links",
        ),
        "daily_continuation_links": _immutable_array(
            design.daily_continuation[1:],
            dtype=np.bool_,
            label="practical daily continuation links",
        ),
        # The historical-vector policy does not consume offsets.  Retaining a
        # correctly shaped zero grid keeps the existing serial value object
        # compact and avoids a second generator representation.
        "monthly_kernel_lambdas": _immutable_array(
            policy.neutral_lambda_grid,
            dtype=np.float64,
            label="practical monthly unused lambdas",
        ),
        "daily_kernel_lambdas": _immutable_array(
            policy.neutral_lambda_grid,
            dtype=np.float64,
            label="practical daily unused lambdas",
        ),
        "monthly_kernel_offsets": _immutable_array(
            zero_offsets,
            dtype=np.float64,
            label="practical monthly unused offsets",
        ),
        "daily_kernel_offsets": _immutable_array(
            zero_offsets,
            dtype=np.float64,
            label="practical daily unused offsets",
        ),
    }
    array_fingerprints = MappingProxyType(
        {
            name: _array_fingerprint(value, name)
            for name, value in sorted(arrays.items())
        }
    )
    config_fingerprint = checked_config.fingerprint()
    metadata: dict[str, object] = {
        "schema_version": SERIAL_GENERATION_INPUT_SCHEMA_VERSION,
        "generator_version": SERIAL_GENERATOR_VERSION,
        "execution_mode": "canonical",
        "generation_authorized": True,
        "rehearsal_scope_fingerprint": None,
        "config_fingerprint": config_fingerprint,
        "processed_data_fingerprint": artifact.processed_data_fingerprint,
        "calibration_input_fingerprint": calibration_input.fingerprint,
        "g3_bundle_fingerprint": artifact.fingerprint,
        "durable_evidence_fingerprint": artifact.fingerprint,
        "pair_receipt_fingerprint": artifact.design_input_fingerprint,
        "g5_authority_schema_id": PRACTICAL_K4_AUTHORITY_SCHEMA_ID,
        "g5_authority_fingerprint": artifact.fingerprint,
        "g5_threshold_registry_fingerprint": None,
        "g5_qualification_results_fingerprint": None,
        "g5_meta_validation_fingerprint": None,
        "production_artifact_fingerprint": artifact.fingerprint,
        "core_model_fingerprint": artifact.model_fingerprint,
        "parameter_set_fingerprint": artifact.model_fingerprint,
        "master_seed": checked_config.run.master_seed,
        "scientific_version": checked_config.simulation.scientific_version,
        "selected_k": 4,
        "configured_lf_paths": checked_config.simulation.low_frequency.paths,
        "configured_hf_paths": checked_config.simulation.high_frequency.paths,
        "configured_years": checked_config.simulation.low_frequency.years,
        "generation_policy": policy.as_dict(),
    }
    result = object.__new__(SerialGenerationInput)
    fields: tuple[tuple[str, object], ...] = (
        *(
            tuple(
                (name, value)
                for name, value in metadata.items()
                if name != "generation_policy"
            )
        ),
        ("generation_policy", policy),
        *(tuple(arrays.items())),
        ("array_fingerprints", array_fingerprints),
        ("fingerprint", _record_fingerprint(metadata, array_fingerprints)),
        ("_seal", _GENERATION_INPUT_SEAL),
    )
    for name, value in fields:
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _verify_generation_input(result)
    return result


def build_g4_reference_generation_input(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    generation_policy: ScientificGenerationPolicy,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    reference_scope_fingerprint: str,
) -> G4ReferenceGenerationInput:
    """Mint a public, explicitly noncanonical G4 reference-engine input.

    The original live loaded G3 evidence and a policy from its registered
    candidate set are required.  This boundary never accepts or creates G5
    authority, always uses the development RNG domain, and cannot enter the
    canonical generator, aggregation, storage, or release paths.
    """

    result = _build_serial_generation_input_core(
        authority=g3_evidence,
        config=config,
        calibration_input=calibration_input,
        rehearsal_policy=generation_policy,
        rehearsal_scope_fingerprint=reference_scope_fingerprint,
        rehearsal_capability=_REHEARSAL_FACTORY_CAPABILITY,
        noncanonical_g5_test_capability=None,
    )
    if type(result) is not RehearsalSerialGenerationInput:  # pragma: no cover
        raise AssertionError("G4 reference mint returned a canonical input")
    return result


def build_g5_candidate_serial_generation_input(
    *,
    authority: LoadedG3EvidenceAuthority | G5QualificationGenerationAuthority,
    generation_policy: ScientificGenerationPolicy,
    candidate_stage: G5CandidateStage,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> G5CandidateSerialGenerationInput:
    """Build one public bounded G5 candidate input."""

    return _build_g5_candidate_serial_generation_input_core(
        authority=authority,
        generation_policy=generation_policy,
        candidate_stage=candidate_stage,
        config=config,
        calibration_input=calibration_input,
        noncanonical_test_capability=None,
        lean_qualification_capability=None,
    )


def build_lean_g5_qualification_serial_generation_input(
    *,
    permit: LeanG5QualificationPermit,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> G5CandidateSerialGenerationInput:
    """Build the exact untouched 500-LF/200-HF lean-G5 input.

    The permit already fixes G3, the selected policy, its registered stream
    version, the lean contract, and the development decision.  The returned
    input remains nonauthorizing for production and uses the existing bounded
    candidate generator unchanged.
    """

    checked = verify_lean_g5_qualification_permit(permit)
    return _build_g5_candidate_serial_generation_input_core(
        authority=checked,
        generation_policy=checked.generation_policy,
        candidate_stage="qualification",
        config=config,
        calibration_input=calibration_input,
        noncanonical_test_capability=None,
        lean_qualification_capability=_LEAN_G5_QUALIFICATION_FACTORY_CAPABILITY,
    )


def _build_g5_candidate_serial_generation_input_for_testing(
    *,
    authority: LoadedG3EvidenceAuthority | G5QualificationGenerationAuthority,
    generation_policy: ScientificGenerationPolicy,
    candidate_stage: G5CandidateStage,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> G5CandidateSerialGenerationInput:
    """Accept a registered noncanonical qualification fixture in unit tests."""

    return _build_g5_candidate_serial_generation_input_core(
        authority=authority,
        generation_policy=generation_policy,
        candidate_stage=candidate_stage,
        config=config,
        calibration_input=calibration_input,
        noncanonical_test_capability=_G5_NONCANONICAL_SERIAL_TEST_CAPABILITY,
        lean_qualification_capability=None,
    )


def _build_g5_candidate_serial_generation_input_core(
    *,
    authority: (
        LoadedG3EvidenceAuthority
        | G5QualificationGenerationAuthority
        | LeanG5QualificationPermit
    ),
    generation_policy: ScientificGenerationPolicy,
    candidate_stage: G5CandidateStage,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    noncanonical_test_capability: object | None,
    lean_qualification_capability: object | None,
) -> G5CandidateSerialGenerationInput:
    """Build one bounded, policy-specific G5 candidate input.

    Development is available directly from freshly verified passed G3 evidence
    before thresholds and uses common-random-number version 11 for every
    registered policy.  Qualification requires typed threshold authority,
    permits only its selected policy, and uses that policy's registered version
    11--15.  Both stages embed a rehearsal input and remain nonauthorizing.
    """

    stage = _g5_candidate_stage(candidate_stage)
    checked_authority: G5QualificationGenerationAuthority | None
    checked_permit: LeanG5QualificationPermit | None
    if stage == "development":
        if type(authority) is not LoadedG3EvidenceAuthority:
            raise GenerationError(
                "G5 development requires freshly verified passed G3 evidence"
            )
        evidence = verify_loaded_g3_evidence(authority)
        checked_authority = None
        checked_permit = None
    else:
        if type(authority) is LeanG5QualificationPermit:
            if (
                lean_qualification_capability
                is not _LEAN_G5_QUALIFICATION_FACTORY_CAPABILITY
                or noncanonical_test_capability is not None
            ):
                raise GenerationError(
                    "lean G5 qualification permit requires its public builder"
                )
            checked_permit = verify_lean_g5_qualification_permit(authority)
            checked_authority = None
            evidence = checked_permit.g3_evidence
        else:
            if lean_qualification_capability is not None:
                raise GenerationError(
                    "lean G5 qualification builder requires its typed permit"
                )
            checked_authority = verify_g5_qualification_generation_authority(authority)
            is_test_fixture = is_g5_noncanonical_test_fixture(checked_authority)
            if is_test_fixture != (
                noncanonical_test_capability is _G5_NONCANONICAL_SERIAL_TEST_CAPABILITY
            ):
                raise GenerationError(
                    "public G5 qualification rejects noncanonical test authority"
                )
            checked_permit = None
            evidence = checked_authority.g3_evidence
    policy = _validated_policy(generation_policy)
    _validate_rehearsal_policy_candidate(policy, evidence.production_artifact)
    if checked_authority is not None and policy != checked_authority.generation_policy:
        raise GenerationError(
            "G5 qualification permits only the threshold-selected policy"
        )
    if checked_permit is not None and policy != checked_permit.generation_policy:
        raise GenerationError("lean G5 permit policy changed")
    checked_config = _validated_config(config)
    if type(calibration_input) is not CalibrationInput:
        raise GenerationError("G5 candidate input requires a CalibrationInput")
    domain = Domain.DEVELOPMENT if stage == "development" else Domain.QUALIFICATION
    policy_scientific_version = (
        checked_config.simulation.scientific_version
        if stage == "development"
        else (
            checked_permit.policy_scientific_version
            if checked_permit is not None
            else g5_policy_scientific_version(policy)
        )
    )
    if stage == "development" and policy_scientific_version != 11:
        raise GenerationError("G5 development common-random stream must be version 11")
    if stage == "development":
        maximum_lf = G5_DEVELOPMENT_LF_PATH_COUNT
        maximum_hf = G5_DEVELOPMENT_HF_PATH_COUNT
    else:
        maximum_lf = G5_QUALIFICATION_LF_PATH_COUNT
        maximum_hf = G5_QUALIFICATION_HF_PATH_COUNT
    scope_identity: dict[str, object] = {
        "schema_id": G5_CANDIDATE_INPUT_SCHEMA_ID,
        "schema_version": G5_CANDIDATE_INPUT_SCHEMA_VERSION,
        "qualification_authority_fingerprint": (
            None if checked_authority is None else checked_authority.fingerprint
        ),
        "g3_evidence_fingerprint": evidence.reference.fingerprint,
        "candidate_stage": stage,
        "rng_domain": int(domain),
        "policy_scientific_version": policy_scientific_version,
        "generation_policy": policy.as_dict(),
        "config_fingerprint": checked_config.fingerprint(),
        "calibration_input_fingerprint": calibration_input.fingerprint,
    }
    if checked_permit is not None:
        scope_identity["lean_qualification_permit_fingerprint"] = (
            checked_permit.fingerprint
        )
    scope_fingerprint = _identity_fingerprint(scope_identity)
    numerical_input = build_g4_reference_generation_input(
        g3_evidence=evidence,
        generation_policy=policy,
        config=checked_config,
        calibration_input=calibration_input,
        reference_scope_fingerprint=scope_fingerprint,
    )
    identity = _g5_candidate_input_identity(
        authority=checked_authority,
        lean_permit=checked_permit,
        g3_evidence=evidence,
        policy=policy,
        candidate_stage=stage,
        rng_domain=domain,
        policy_scientific_version=policy_scientific_version,
        maximum_lf_paths=maximum_lf,
        maximum_hf_paths=maximum_hf,
        configured_years=numerical_input.configured_years,
        numerical_input_fingerprint=numerical_input.fingerprint,
    )
    result = object.__new__(G5CandidateSerialGenerationInput)
    for name, value in (
        ("schema_id", G5_CANDIDATE_INPUT_SCHEMA_ID),
        ("schema_version", G5_CANDIDATE_INPUT_SCHEMA_VERSION),
        ("generation_authorized", False),
        ("candidate_stage", stage),
        ("rng_domain", domain),
        ("g3_evidence", evidence),
        ("qualification_authority", checked_authority),
        (
            "qualification_authority_fingerprint",
            None if checked_authority is None else checked_authority.fingerprint,
        ),
        ("lean_qualification_permit", checked_permit),
        (
            "lean_qualification_permit_fingerprint",
            None if checked_permit is None else checked_permit.fingerprint,
        ),
        (
            "g3_evidence_fingerprint",
            evidence.reference.fingerprint,
        ),
        (
            "threshold_evidence_fingerprint",
            None
            if checked_authority is None
            else checked_authority.threshold_evidence_fingerprint,
        ),
        (
            "threshold_registry_fingerprint",
            None
            if checked_authority is None
            else checked_authority.threshold_registry_fingerprint,
        ),
        ("generation_policy", policy),
        ("generation_policy_fingerprint", _identity_fingerprint(policy.as_dict())),
        ("policy_scientific_version", policy_scientific_version),
        ("maximum_lf_paths", maximum_lf),
        ("maximum_hf_paths", maximum_hf),
        ("configured_years", numerical_input.configured_years),
        ("numerical_input", numerical_input),
        ("numerical_input_fingerprint", numerical_input.fingerprint),
        ("fingerprint", _identity_fingerprint(identity)),
        ("_seal", _G5_CANDIDATE_INPUT_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _G5_CANDIDATE_INPUTS[id(result)] = result
    return verify_g5_candidate_serial_generation_input(result)


def verify_g5_candidate_serial_generation_input(
    value: object,
) -> G5CandidateSerialGenerationInput:
    """Revalidate an original bounded G5 candidate input."""

    _verify_g5_candidate_generation_input(value)
    assert isinstance(value, G5CandidateSerialGenerationInput)
    return value


def is_g5_candidate_serial_generation_input(
    value: object,
) -> TypeGuard[G5CandidateSerialGenerationInput]:
    try:
        _verify_g5_candidate_generation_input(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def build_g5_meta_serial_numerical_input(
    generation_input: RehearsalSerialGenerationInput,
) -> G5MetaSerialNumericalInput:
    """Detach verified rehearsal numerics for spawned Purpose-12 workers.

    The returned object is data-only and permanently nonauthorizing.  It does
    not preserve the live rehearsal capability and therefore cannot be used by
    any production generator, aggregator, storage writer, or release surface.
    """

    _verify_rehearsal_generation_input(generation_input)
    return _mint_g5_meta_serial_numerical_input(
        source_generation_input_fingerprint=generation_input.fingerprint,
        master_seed=generation_input.master_seed,
        scientific_version=generation_input.scientific_version,
        selected_k=generation_input.selected_k,
        configured_years=generation_input.configured_years,
        generation_policy=generation_input.generation_policy,
        stationary_distribution=generation_input.stationary_distribution,
        transition_matrix=generation_input.transition_matrix,
        monthly_returns=generation_input.monthly_returns,
        daily_returns=generation_input.daily_returns,
        monthly_source_states=generation_input.monthly_source_states,
        daily_source_states=generation_input.daily_source_states,
        monthly_continuation_links=generation_input.monthly_continuation_links,
        daily_continuation_links=generation_input.daily_continuation_links,
        monthly_kernel_lambdas=generation_input.monthly_kernel_lambdas,
        daily_kernel_lambdas=generation_input.daily_kernel_lambdas,
        monthly_kernel_offsets=generation_input.monthly_kernel_offsets,
        daily_kernel_offsets=generation_input.daily_kernel_offsets,
    )


def verify_g5_meta_serial_numerical_input(
    value: object,
) -> G5MetaSerialNumericalInput:
    """Recheck one process-portable, nonauthorizing numerical view."""

    _verify_g5_meta_serial_numerical_input(value)
    assert isinstance(value, G5MetaSerialNumericalInput)
    return value


def _build_rehearsal_serial_generation_input_for_testing(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    generation_policy: ScientificGenerationPolicy,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    rehearsal_scope_fingerprint: str,
) -> RehearsalSerialGenerationInput:
    """Mint a visibly noncanonical input for pre-G5 engine tests only."""

    return build_g4_reference_generation_input(
        g3_evidence=g3_evidence,
        generation_policy=generation_policy,
        config=config,
        calibration_input=calibration_input,
        reference_scope_fingerprint=rehearsal_scope_fingerprint,
    )


def _build_noncanonical_g5_serial_generation_input_for_testing(
    *,
    authority: G5QualifiedProductionAuthority,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> SerialGenerationInput:
    """Exercise the canonical serial math from a registered test fixture."""

    result = _build_serial_generation_input_core(
        authority=authority,
        config=config,
        calibration_input=calibration_input,
        rehearsal_policy=None,
        rehearsal_scope_fingerprint=None,
        rehearsal_capability=None,
        noncanonical_g5_test_capability=_G5_NONCANONICAL_SERIAL_TEST_CAPABILITY,
    )
    if type(result) is not SerialGenerationInput:  # pragma: no cover - invariant
        raise AssertionError("noncanonical G5 test mint returned rehearsal input")
    return result


def _build_serial_generation_input_core(
    *,
    authority: object,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    rehearsal_policy: ScientificGenerationPolicy | None,
    rehearsal_scope_fingerprint: str | None,
    rehearsal_capability: object | None,
    noncanonical_g5_test_capability: object | None,
) -> SerialGenerationInput | RehearsalSerialGenerationInput:
    from prpg.model.lean_g5_generation_authority import (
        LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID,
        LeanG5ProductionAuthority,
        verify_lean_g5_production_authority,
    )

    rehearsal = rehearsal_capability is _REHEARSAL_FACTORY_CAPABILITY
    if rehearsal:
        evidence = verify_loaded_g3_evidence(authority)
        artifact = evidence.production_artifact
        policy = _validated_policy(rehearsal_policy)
        _validate_rehearsal_policy_candidate(policy, artifact)
        scope_fingerprint = _required_sha256(
            rehearsal_scope_fingerprint, "rehearsal scope"
        )
        g5_authority_schema_id = None
        g5_authority_fingerprint = None
        g5_threshold_registry_fingerprint = None
        g5_qualification_results_fingerprint = None
        g5_meta_validation_fingerprint = None
    else:
        if rehearsal_policy is not None or rehearsal_scope_fingerprint is not None:
            raise GenerationError("canonical serial mint rejects rehearsal inputs")
        _validate_authority(authority)
        if type(authority) is LeanG5ProductionAuthority:
            if noncanonical_g5_test_capability is not None:
                raise GenerationError(
                    "lean G5 production authority rejects test capabilities"
                )
            lean_authority = verify_lean_g5_production_authority(authority)
            evidence = lean_authority.g3_evidence
            artifact = lean_authority.production_artifact
            policy = _validated_policy(lean_authority.generation_policy)
            g5_authority_schema_id = LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID
            g5_authority_fingerprint = lean_authority.fingerprint
            # Versioned compatibility mapping for the old serial fields:
            # threshold_registry -> lean qualification permit;
            # qualification_results -> lean assessment; meta remains absent.
            g5_threshold_registry_fingerprint = (
                lean_authority.qualification_permit_fingerprint
            )
            g5_qualification_results_fingerprint = lean_authority.assessment_fingerprint
            g5_meta_validation_fingerprint = None
        else:
            assert isinstance(authority, G5QualifiedProductionAuthority)
            is_test_fixture = is_g5_noncanonical_test_fixture(authority)
            if is_test_fixture != (
                noncanonical_g5_test_capability
                is _G5_NONCANONICAL_SERIAL_TEST_CAPABILITY
            ):
                raise GenerationError(
                    "public canonical serial mint rejects noncanonical G5 authority"
                )
            evidence = authority.g3_evidence
            artifact = authority.production_artifact
            policy = _validated_policy(authority.generation_policy)
            g5_authority_schema_id = G5_GENERATION_AUTHORITY_SCHEMA_ID
            g5_authority_fingerprint = authority.fingerprint
            g5_threshold_registry_fingerprint = authority.threshold_registry_fingerprint
            g5_qualification_results_fingerprint = (
                authority.qualification_results_fingerprint
            )
            g5_meta_validation_fingerprint = authority.meta_validation_fingerprint
        scope_fingerprint = None
    checked_config = _validated_config(config)
    _validate_calibration_input(calibration_input)

    bundle = evidence.evidence_bundle
    config_fingerprint = checked_config.fingerprint()
    _require_equal(
        config_fingerprint,
        bundle.config_fingerprint,
        "configuration differs from durable G3 evidence",
    )
    _require_equal(
        config_fingerprint,
        artifact.binding.config_fingerprint,
        "configuration differs from the production artifact",
    )
    _require_equal(
        config_fingerprint,
        calibration_input.provenance.calibration_config_fingerprint,
        "configuration differs from the calibration input",
    )
    _require_equal(
        calibration_input.provenance.processed_data_fingerprint,
        bundle.processed_data_fingerprint,
        "calibration data differs from durable G3 evidence",
    )
    _require_equal(
        calibration_input.provenance.processed_data_fingerprint,
        artifact.binding.processed_data_fingerprint,
        "calibration data differs from the production artifact",
    )
    _require_equal(
        calibration_input.fingerprint,
        bundle.calibration_input_fingerprint,
        "calibration input differs from durable G3 evidence",
    )
    _require_equal(
        calibration_input.fingerprint,
        artifact.binding.calibration_input_fingerprint,
        "calibration input differs from the production artifact",
    )
    if (
        calibration_input.geometry.holdout_months
        != checked_config.model.design_holdout_months
    ):
        raise IntegrityError(
            "configuration holdout geometry differs from calibration input"
        )
    if artifact.selected_k not in checked_config.model.regime_candidates:
        raise IntegrityError("production K is outside the configured candidate set")
    if (
        artifact.binding.scientific_version
        != checked_config.simulation.scientific_version
    ):
        raise IntegrityError("scientific version differs across config and artifact")

    _validate_policy_against_config(policy, checked_config)
    _validate_live_artifact_arrays(artifact)
    _validate_artifact_input_geometry(artifact, calibration_input)

    arrays: dict[str, np.ndarray[Any, Any]] = {
        "stationary_distribution": _immutable_array(
            validate_markov_chain(
                artifact.arrays["hmm_transition_matrix"]
            ).stationary_distribution,
            dtype=np.float64,
            label="stationary distribution",
        ),
        "transition_matrix": _immutable_array(
            artifact.arrays["hmm_transition_matrix"],
            dtype=np.float64,
            label="transition matrix",
        ),
        "monthly_returns": _immutable_array(
            calibration_input.full.monthly_returns,
            dtype=np.float64,
            label="monthly returns",
        ),
        "daily_returns": _immutable_array(
            calibration_input.full.daily_returns,
            dtype=np.float64,
            label="daily returns",
        ),
        "monthly_source_states": _immutable_array(
            artifact.arrays["monthly_source_states"],
            dtype=np.int64,
            label="monthly source states",
        ),
        "daily_source_states": _immutable_array(
            artifact.arrays["daily_source_states"],
            dtype=np.int64,
            label="daily source states",
        ),
        # Artifact and calibration input use a false-at-row-start convention;
        # the bootstrap primitive consumes links for i -> i+1.
        "monthly_continuation_links": _immutable_array(
            artifact.arrays["monthly_continuation"][1:],
            dtype=np.bool_,
            label="monthly continuation links",
        ),
        "daily_continuation_links": _immutable_array(
            artifact.arrays["daily_continuation"][1:],
            dtype=np.bool_,
            label="daily continuation links",
        ),
        "monthly_kernel_lambdas": _immutable_array(
            artifact.arrays["monthly_kernel_lambdas"],
            dtype=np.float64,
            label="monthly kernel lambdas",
        ),
        "daily_kernel_lambdas": _immutable_array(
            artifact.arrays["daily_kernel_lambdas"],
            dtype=np.float64,
            label="daily kernel lambdas",
        ),
        "monthly_kernel_offsets": _immutable_array(
            artifact.arrays["monthly_kernel_offsets"],
            dtype=np.float64,
            label="monthly kernel offsets",
        ),
        "daily_kernel_offsets": _immutable_array(
            artifact.arrays["daily_kernel_offsets"],
            dtype=np.float64,
            label="daily kernel offsets",
        ),
    }
    array_fingerprints = MappingProxyType(
        {
            name: _array_fingerprint(value, name)
            for name, value in sorted(arrays.items())
        }
    )
    metadata = _generation_input_metadata(
        config=checked_config,
        evidence=evidence,
        artifact=artifact,
        calibration_input=calibration_input,
        policy=policy,
        execution_mode="rehearsal" if rehearsal else "canonical",
        generation_authorized=not rehearsal,
        rehearsal_scope_fingerprint=scope_fingerprint,
        g5_authority_schema_id=g5_authority_schema_id,
        g5_authority_fingerprint=g5_authority_fingerprint,
        g5_threshold_registry_fingerprint=g5_threshold_registry_fingerprint,
        g5_qualification_results_fingerprint=(g5_qualification_results_fingerprint),
        g5_meta_validation_fingerprint=g5_meta_validation_fingerprint,
    )
    fingerprint = _record_fingerprint(metadata, array_fingerprints)
    result_type = RehearsalSerialGenerationInput if rehearsal else SerialGenerationInput
    result = object.__new__(result_type)
    fields: tuple[tuple[str, object], ...] = (
        ("schema_version", SERIAL_GENERATION_INPUT_SCHEMA_VERSION),
        ("generator_version", SERIAL_GENERATOR_VERSION),
        ("execution_mode", "rehearsal" if rehearsal else "canonical"),
        ("generation_authorized", not rehearsal),
        ("rehearsal_scope_fingerprint", scope_fingerprint),
        ("config_fingerprint", config_fingerprint),
        ("processed_data_fingerprint", bundle.processed_data_fingerprint),
        ("calibration_input_fingerprint", calibration_input.fingerprint),
        ("g3_bundle_fingerprint", bundle.fingerprint),
        ("durable_evidence_fingerprint", evidence.reference.fingerprint),
        ("pair_receipt_fingerprint", evidence.pair_receipt_fingerprint),
        ("g5_authority_schema_id", g5_authority_schema_id),
        ("g5_authority_fingerprint", g5_authority_fingerprint),
        (
            "g5_threshold_registry_fingerprint",
            g5_threshold_registry_fingerprint,
        ),
        (
            "g5_qualification_results_fingerprint",
            g5_qualification_results_fingerprint,
        ),
        ("g5_meta_validation_fingerprint", g5_meta_validation_fingerprint),
        ("production_artifact_fingerprint", artifact.reference.fingerprint),
        ("core_model_fingerprint", artifact.core_model_fingerprint),
        ("parameter_set_fingerprint", artifact.parameter_set_fingerprint),
        ("master_seed", checked_config.run.master_seed),
        ("scientific_version", checked_config.simulation.scientific_version),
        ("selected_k", artifact.selected_k),
        ("configured_lf_paths", checked_config.simulation.low_frequency.paths),
        ("configured_hf_paths", checked_config.simulation.high_frequency.paths),
        ("configured_years", checked_config.simulation.low_frequency.years),
        ("generation_policy", policy),
        *(tuple(arrays.items())),
        ("array_fingerprints", array_fingerprints),
        ("fingerprint", fingerprint),
        ("_seal", _REHEARSAL_INPUT_SEAL if rehearsal else _GENERATION_INPUT_SEAL),
    )
    for name, value in fields:
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    if rehearsal:
        _verify_rehearsal_generation_input(result)
    else:
        _verify_generation_input(result)
    return result


def is_sealed_serial_generation_input(
    value: object,
) -> TypeGuard[SerialGenerationInput]:
    """Return whether ``value`` is the original, untampered minted instance."""

    try:
        _verify_generation_input(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_rehearsal_serial_generation_input(
    value: object,
) -> TypeGuard[RehearsalSerialGenerationInput]:
    """Return whether ``value`` is a noncanonical rehearsal-only input."""

    try:
        _verify_rehearsal_generation_input(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g4_reference_generation_input(
    value: object,
) -> TypeGuard[G4ReferenceGenerationInput]:
    """Return whether ``value`` is an original nonauthorizing G4 input."""

    return is_rehearsal_serial_generation_input(value)


def generate_serial_base_path(
    generation_input: SerialGenerationInput,
    *,
    family: PathFamily,
    path_entity: int,
    years: int,
) -> SerialBasePath:
    """Generate one pure serial LF-monthly or HF-daily base path.

    ``path_entity`` is the one-based numeric RNG entity.  This low-level pure
    API allows any registered uint32 entity and any positive integer horizon;
    use :func:`generate_configured_serial_base_path` for the configured path
    count and horizon boundary.
    """

    _verify_generation_input(generation_input)
    checked_family = _family(family)
    entity = _path_entity(path_entity, maximum=UINT32_MAX)
    horizon = _positive_int(years, "years")
    monthly_count = MONTHS_PER_YEAR * horizon
    base_count = (
        monthly_count if checked_family == "LF" else SESSIONS_PER_YEAR * horizon
    )

    rng_family = Family.LF if checked_family == "LF" else Family.HF
    state_uniforms = uniforms(
        _production_key(generation_input, rng_family, entity, Stage.STATE_CHAIN),
        monthly_count,
    )
    restart_uniforms = uniforms(
        _production_key(generation_input, rng_family, entity, Stage.RESTART_UNIFORMS),
        base_count,
    )
    source_rank_uniforms = uniforms(
        _production_key(
            generation_input, rng_family, entity, Stage.SOURCE_RANK_UNIFORMS
        ),
        base_count,
    )
    return _materialize_serial_base_path(
        generation_input,
        family=checked_family,
        path_entity=entity,
        years=horizon,
        state_uniforms=state_uniforms,
        restart_uniforms=restart_uniforms,
        source_rank_uniforms=source_rank_uniforms,
    )


def generate_g4_reference_base_path(
    generation_input: G4ReferenceGenerationInput,
    *,
    family: PathFamily,
    path_entity: int,
    years: int,
) -> G4ReferenceBasePath:
    """Generate one noncanonical reference path in ``Domain.DEVELOPMENT``."""

    _verify_rehearsal_generation_input(generation_input)
    checked_family = _family(family)
    entity = _path_entity(path_entity, maximum=UINT32_MAX)
    horizon = _positive_int(years, "years")
    monthly_count = MONTHS_PER_YEAR * horizon
    base_count = (
        monthly_count if checked_family == "LF" else SESSIONS_PER_YEAR * horizon
    )
    rng_family = Family.LF if checked_family == "LF" else Family.HF
    state_uniforms = uniforms(
        _production_key(generation_input, rng_family, entity, Stage.STATE_CHAIN),
        monthly_count,
    )
    restart_uniforms = uniforms(
        _production_key(generation_input, rng_family, entity, Stage.RESTART_UNIFORMS),
        base_count,
    )
    source_rank_uniforms = uniforms(
        _production_key(
            generation_input, rng_family, entity, Stage.SOURCE_RANK_UNIFORMS
        ),
        base_count,
    )
    result = _materialize_serial_base_path(
        generation_input,
        family=checked_family,
        path_entity=entity,
        years=horizon,
        state_uniforms=state_uniforms,
        restart_uniforms=restart_uniforms,
        source_rank_uniforms=source_rank_uniforms,
        rehearsal=True,
    )
    if type(result) is not RehearsalSerialBasePath:  # pragma: no cover
        raise AssertionError("G4 reference generator returned a canonical path")
    return result


def generate_g5_candidate_serial_base_path(
    generation_input: G5CandidateSerialGenerationInput,
    *,
    family: PathFamily,
    path_entity: int,
) -> G5CandidateSerialBasePath:
    """Generate one exact-horizon bounded development/qualification path."""

    checked_input = verify_g5_candidate_serial_generation_input(generation_input)
    checked_family = _family(family)
    maximum = (
        checked_input.maximum_lf_paths
        if checked_family == "LF"
        else checked_input.maximum_hf_paths
    )
    entity = _path_entity(path_entity, maximum=maximum)
    years = checked_input.configured_years
    monthly_count = MONTHS_PER_YEAR * years
    base_count = monthly_count if checked_family == "LF" else SESSIONS_PER_YEAR * years
    rng_family = Family.LF if checked_family == "LF" else Family.HF
    state_uniforms = uniforms(
        _g5_candidate_key(checked_input, rng_family, entity, Stage.STATE_CHAIN),
        monthly_count,
    )
    restart_uniforms = uniforms(
        _g5_candidate_key(checked_input, rng_family, entity, Stage.RESTART_UNIFORMS),
        base_count,
    )
    source_rank_uniforms = uniforms(
        _g5_candidate_key(
            checked_input, rng_family, entity, Stage.SOURCE_RANK_UNIFORMS
        ),
        base_count,
    )
    base_path = _materialize_serial_base_path(
        checked_input.numerical_input,
        family=checked_family,
        path_entity=entity,
        years=years,
        state_uniforms=state_uniforms,
        restart_uniforms=restart_uniforms,
        source_rank_uniforms=source_rank_uniforms,
        rehearsal=True,
    )
    if type(base_path) is not RehearsalSerialBasePath:  # pragma: no cover
        raise AssertionError("G5 candidate generator returned a canonical base path")
    identity = _g5_candidate_path_identity(
        generation_input=checked_input,
        family=checked_family,
        path_entity=entity,
        base_path=base_path,
    )
    result = object.__new__(G5CandidateSerialBasePath)
    for name, value in (
        ("schema_id", G5_CANDIDATE_PATH_SCHEMA_ID),
        ("schema_version", G5_CANDIDATE_PATH_SCHEMA_VERSION),
        ("generation_authorized", False),
        ("candidate_stage", checked_input.candidate_stage),
        ("rng_domain", checked_input.rng_domain),
        ("generation_input", checked_input),
        ("generation_input_fingerprint", checked_input.fingerprint),
        (
            "generation_policy_fingerprint",
            checked_input.generation_policy_fingerprint,
        ),
        ("family", checked_family),
        ("base_frequency", base_path.base_frequency),
        ("path_entity", entity),
        ("years", years),
        ("base_path", base_path),
        ("base_path_fingerprint", base_path.fingerprint),
        ("fingerprint", _identity_fingerprint(identity)),
        ("_seal", _G5_CANDIDATE_PATH_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _G5_CANDIDATE_PATHS[id(result)] = result
    return verify_g5_candidate_serial_base_path(result)


def materialize_g5_meta_rehearsal_base_path(
    generation_input: G5MetaSerialNumericalInput,
    *,
    family: PathFamily,
    path_entity: int,
    years: int,
    state_uniforms: npt.ArrayLike,
    restart_uniforms: npt.ArrayLike,
    source_rank_uniforms: npt.ArrayLike,
    transition_matrix: npt.ArrayLike | None = None,
) -> RehearsalSerialBasePath:
    """Run the exact conditioned engine from registered Purpose-12 draws.

    This is the sole process-portable numerical extraction for G5 meta work.
    It always returns a rehearsal-only path.  ``transition_matrix`` exists
    only for the registered Purpose-13/v5 P-prime regeneration; the fitted
    stationary law must remain stationary for that matrix.
    """

    checked = verify_g5_meta_serial_numerical_input(generation_input)
    checked_family = _family(family)
    entity = _path_entity(path_entity, maximum=UINT32_MAX)
    horizon = _positive_int(years, "meta path years")
    if horizon > checked.configured_years:
        raise GenerationError("meta path years exceed the configured horizon")
    result = _materialize_serial_base_path(
        checked,
        family=checked_family,
        path_entity=entity,
        years=horizon,
        state_uniforms=state_uniforms,
        restart_uniforms=restart_uniforms,
        source_rank_uniforms=source_rank_uniforms,
        rehearsal=True,
        meta_input=True,
        transition_matrix=transition_matrix,
    )
    if type(result) is not RehearsalSerialBasePath:  # pragma: no cover
        raise AssertionError("G5 meta generator returned a canonical path")
    return result


def verify_g5_candidate_serial_base_path(
    value: object,
) -> G5CandidateSerialBasePath:
    """Revalidate an original bounded G5 candidate path."""

    _verify_g5_candidate_base_path(value)
    assert isinstance(value, G5CandidateSerialBasePath)
    return value


def is_g5_candidate_serial_base_path(
    value: object,
) -> TypeGuard[G5CandidateSerialBasePath]:
    try:
        _verify_g5_candidate_base_path(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def _generate_rehearsal_serial_base_path_for_testing(
    generation_input: RehearsalSerialGenerationInput,
    *,
    family: PathFamily,
    path_entity: int,
    years: int,
) -> RehearsalSerialBasePath:
    """Exercise the reference engine without creating canonical output."""

    return generate_g4_reference_base_path(
        generation_input,
        family=family,
        path_entity=path_entity,
        years=years,
    )


def generate_configured_serial_base_path(
    generation_input: SerialGenerationInput,
    *,
    family: PathFamily,
    path_entity: int,
) -> SerialBasePath:
    """Generate one path using the configured family range and horizon."""

    _verify_generation_input(generation_input)
    checked_family = _family(family)
    maximum = (
        generation_input.configured_lf_paths
        if checked_family == "LF"
        else generation_input.configured_hf_paths
    )
    entity = _path_entity(path_entity, maximum=maximum)
    return generate_serial_base_path(
        generation_input,
        family=checked_family,
        path_entity=entity,
        years=generation_input.configured_years,
    )


def is_sealed_serial_base_path(value: object) -> TypeGuard[SerialBasePath]:
    """Return whether ``value`` is the original, untampered generated path."""

    try:
        _verify_base_path(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_rehearsal_serial_base_path(
    value: object,
) -> TypeGuard[RehearsalSerialBasePath]:
    """Return whether ``value`` is a noncanonical rehearsal-only path."""

    try:
        _verify_rehearsal_base_path(value)
    except (GenerationError, IntegrityError, ModelError, TypeError, ValueError):
        return False
    return True


def is_g4_reference_base_path(value: object) -> TypeGuard[G4ReferenceBasePath]:
    """Return whether ``value`` is an original nonauthorizing G4 path."""

    return is_rehearsal_serial_base_path(value)


def _materialize_serial_base_path(
    generation_input: (
        SerialGenerationInput
        | RehearsalSerialGenerationInput
        | G5MetaSerialNumericalInput
    ),
    *,
    family: PathFamily,
    path_entity: int,
    years: int,
    state_uniforms: npt.ArrayLike,
    restart_uniforms: npt.ArrayLike,
    source_rank_uniforms: npt.ArrayLike,
    rehearsal: bool = False,
    meta_input: bool = False,
    transition_matrix: npt.ArrayLike | None = None,
) -> SerialBasePath | RehearsalSerialBasePath:
    """Materialize a path from already allocated fixed-consumption draws."""

    if meta_input:
        if not rehearsal or type(generation_input) is not G5MetaSerialNumericalInput:
            raise GenerationError("meta serial numerics are rehearsal-only")
        checked_meta = verify_g5_meta_serial_numerical_input(generation_input)
    elif rehearsal:
        _verify_rehearsal_generation_input(generation_input)
    else:
        _verify_generation_input(generation_input)
    policy = generation_input.generation_policy
    monthly_count = MONTHS_PER_YEAR * years
    state_draws = _uniform_array(state_uniforms, monthly_count, "state uniforms")
    family_checked = _family(family)
    if family_checked == "LF":
        frequency: BaseFrequency = "monthly"
        base_count = monthly_count
        source_returns = generation_input.monthly_returns
        source_states = generation_input.monthly_source_states
        source_links = generation_input.monthly_continuation_links
        restart_probability = policy.monthly_restart_probability
        lambdas = generation_input.monthly_kernel_lambdas
        offsets = generation_input.monthly_kernel_offsets
    else:
        frequency = "daily"
        base_count = SESSIONS_PER_YEAR * years
        source_returns = generation_input.daily_returns
        source_states = generation_input.daily_source_states
        source_links = generation_input.daily_continuation_links
        restart_probability = policy.daily_restart_probability
        lambdas = generation_input.daily_kernel_lambdas
        offsets = generation_input.daily_kernel_offsets
    restart_draws = _uniform_array(restart_uniforms, base_count, "restart uniforms")
    source_draws = _uniform_array(
        source_rank_uniforms, base_count, "source-rank uniforms"
    )

    chain = generation_input.transition_matrix
    if transition_matrix is not None:
        if not meta_input:
            raise GenerationError("transition override is registered only for G5 meta")
        candidate = np.asarray(transition_matrix, dtype=np.float64)
        if candidate.shape != (checked_meta.selected_k, checked_meta.selected_k):
            raise GenerationError("meta transition override has invalid geometry")
        validate_markov_chain(candidate)
        if not np.allclose(
            checked_meta.stationary_distribution @ candidate,
            checked_meta.stationary_distribution,
            rtol=0.0,
            atol=1e-12,
        ):
            raise GenerationError(
                "meta transition override changed the registered stationary law"
            )
        chain = candidate
    monthly_trace = simulate_target_states_from_uniforms(
        generation_input.stationary_distribution,
        chain,
        state_draws,
    )
    monthly_states = np.asarray(monthly_trace.states, dtype=np.int64)
    target_states = (
        monthly_states
        if family_checked == "LF"
        else np.repeat(monthly_states, SESSIONS_PER_MONTH)
    )
    if target_states.size != base_count:  # pragma: no cover - arithmetic invariant
        raise AssertionError("base target-state geometry is inconsistent")
    source_trace = conditioned_source_index_trace(
        target_states,
        source_states,
        source_links,
        restart_draws,
        source_draws,
        restart_probability=restart_probability,
    )
    sampled = np.asarray(source_returns[source_trace.source_indices], dtype=np.float64)
    values = np.array(sampled, dtype=np.float64, copy=True, order="C")
    if policy.alpha_policy == "kernel_mean_neutral":
        selected = policy.selected_neutral_lambda
        if selected is None:  # pragma: no cover - policy validator prevents this
            raise IntegrityError("kernel-neutral policy has no selected lambda")
        matches = np.flatnonzero(lambdas == float(selected))
        if matches.size != 1:
            raise IntegrityError("selected neutral lambda has no unique offset row")
        offset_row = offsets[int(matches[0])]
        with np.errstate(over="ignore", invalid="ignore"):
            values = values - offset_row[target_states]
    if values.shape != (base_count, ASSET_COUNT) or not bool(np.isfinite(values).all()):
        raise GenerationError(
            "generated base path is non-finite or has invalid geometry"
        )

    reason_map = {
        BootstrapTraceReason.FIRST_OBSERVATION: RestartReason.FIRST_OBSERVATION,
        BootstrapTraceReason.TARGET_STATE_TRANSITION: (
            RestartReason.TARGET_STATE_TRANSITION
        ),
        BootstrapTraceReason.SOURCE_END: RestartReason.SOURCE_END,
        BootstrapTraceReason.SOURCE_GAP: RestartReason.SOURCE_GAP,
        BootstrapTraceReason.SOURCE_STATE_MISMATCH: (
            RestartReason.SOURCE_STATE_MISMATCH
        ),
        BootstrapTraceReason.RANDOM_RESTART: RestartReason.RANDOM_RESTART,
        BootstrapTraceReason.CONTINUATION: RestartReason.CONTINUATION,
    }
    try:
        reasons = tuple(reason_map[value] for value in source_trace.reasons)
    except KeyError as error:
        # Target segments are never present in one continuous generated path.
        raise GenerationError(
            "generator produced an unsupported restart reason"
        ) from error
    audit = RestartAudit(
        restart_flags=_immutable_array(
            source_trace.restart_flags, dtype=np.bool_, label="restart flags"
        ),
        reasons=reasons,
        restart_uniform_used=_immutable_array(
            source_trace.restart_uniform_used,
            dtype=np.bool_,
            label="restart-uniform use flags",
        ),
        source_rank_uniform_used=_immutable_array(
            source_trace.source_rank_uniform_used,
            dtype=np.bool_,
            label="source-rank-uniform use flags",
        ),
        restart_uniforms_consumed=source_trace.restart_uniforms_consumed,
        source_rank_uniforms_consumed=source_trace.source_rank_uniforms_consumed,
    )
    frozen_arrays: dict[str, np.ndarray[Any, Any]] = {
        "log_returns": _immutable_array(
            values, dtype=np.float64, label="generated log returns"
        ),
        "target_monthly_states": _immutable_array(
            monthly_states, dtype=np.int64, label="target monthly states"
        ),
        "target_states": _immutable_array(
            target_states, dtype=np.int64, label="target base states"
        ),
        "source_indices": _immutable_array(
            source_trace.source_indices, dtype=np.int64, label="source indices"
        ),
    }
    metadata = {
        "schema_version": SERIAL_BASE_PATH_SCHEMA_VERSION,
        "generator_version": SERIAL_GENERATOR_VERSION,
        "execution_mode": "rehearsal" if rehearsal else "canonical",
        "generation_authorized": not rehearsal,
        "generation_input_fingerprint": generation_input.fingerprint,
        "family": family_checked,
        "base_frequency": frequency,
        "path_entity": path_entity,
        "years": years,
        "state_uniforms_consumed": monthly_trace.uniforms_consumed,
        "restart_uniforms_consumed": audit.restart_uniforms_consumed,
        "source_rank_uniforms_consumed": audit.source_rank_uniforms_consumed,
        "restart_reasons": [value.value for value in audit.reasons],
    }
    fingerprint_arrays = {
        **frozen_arrays,
        "restart_flags": audit.restart_flags,
        "restart_uniform_used": audit.restart_uniform_used,
        "source_rank_uniform_used": audit.source_rank_uniform_used,
    }
    fingerprint = _record_fingerprint(
        metadata,
        {
            name: _array_fingerprint(value, name)
            for name, value in sorted(fingerprint_arrays.items())
        },
    )
    result_type = RehearsalSerialBasePath if rehearsal else SerialBasePath
    result = object.__new__(result_type)
    for name, value in (
        ("schema_version", SERIAL_BASE_PATH_SCHEMA_VERSION),
        ("generator_version", SERIAL_GENERATOR_VERSION),
        ("execution_mode", "rehearsal" if rehearsal else "canonical"),
        ("generation_authorized", not rehearsal),
        ("generation_input_fingerprint", generation_input.fingerprint),
        ("family", family_checked),
        ("base_frequency", frequency),
        ("path_entity", path_entity),
        ("years", years),
        *tuple(frozen_arrays.items()),
        ("restart_audit", audit),
        ("state_uniforms_consumed", monthly_trace.uniforms_consumed),
        ("fingerprint", fingerprint),
        ("_seal", _REHEARSAL_BASE_PATH_SEAL if rehearsal else _BASE_PATH_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    if rehearsal:
        _verify_rehearsal_base_path(result)
    else:
        _verify_base_path(result)
    return result


def _validate_authority(authority: object) -> None:
    from prpg.model.lean_g5_generation_authority import (
        LeanG5ProductionAuthority,
        verify_lean_g5_production_authority,
    )

    if type(authority) is LoadedG3EvidenceAuthority:
        raise GenerationError(
            "loaded G3 evidence is evidence-only; serial generation requires "
            "qualified G5 production authority"
        )
    if type(authority) is LeanG5ProductionAuthority:
        try:
            lean = verify_lean_g5_production_authority(authority)
        except (IntegrityError, ModelError) as error:
            raise IntegrityError(
                "lean G5 production authority failed live reverification"
            ) from error
        evidence = lean.g3_evidence
        design = evidence.design_artifact
        production = lean.production_artifact
    elif type(authority) is G5QualifiedProductionAuthority:
        try:
            legacy = verify_g5_qualified_production_authority(authority)
        except (IntegrityError, ModelError) as error:
            raise IntegrityError(
                "qualified G5 production authority failed live reverification"
            ) from error
        evidence = legacy.g3_evidence
        design = evidence.design_artifact
        production = legacy.production_artifact
    else:
        raise GenerationError(
            "serial generation requires qualified G5 production authority "
            "(legacy or lean)"
        )
    if not is_verified_scientific_model_artifact(
        design
    ) or not is_verified_scientific_model_artifact(production):
        raise IntegrityError("durable G3 evidence contains an unverified artifact")
    if design.role != "design" or production.role != "production":
        raise IntegrityError("durable G3 evidence artifact roles are invalid")
    bundle = evidence.evidence_bundle
    if (
        bundle.design_artifact_fingerprint != design.reference.fingerprint
        or bundle.production_artifact_fingerprint != production.reference.fingerprint
        or design.reference.fingerprint == production.reference.fingerprint
    ):
        raise IntegrityError("durable G3 evidence artifact fingerprints disagree")


def _validated_config(config: object) -> PRPGConfig:
    if type(config) is not PRPGConfig:
        raise GenerationError("serial generation requires a typed PRPGConfig")
    try:
        rebuilt = PRPGConfig.model_validate(config.model_dump(mode="python"))
    except (TypeError, ValueError) as error:
        raise IntegrityError(
            "serial generation configuration no longer validates"
        ) from error
    if rebuilt != config or rebuilt.canonical_json() != config.canonical_json():
        raise IntegrityError("serial generation configuration is not canonical")
    return rebuilt


def _validate_calibration_input(value: object) -> None:
    if type(value) is not CalibrationInput:
        raise GenerationError("serial generation requires a typed CalibrationInput")
    expected_fingerprint = hashlib.sha256(
        (_canonical_json(value.identity_dict()) + "\n").encode()
    ).hexdigest()
    if value.fingerprint != expected_fingerprint:
        raise IntegrityError("calibration-input fingerprint is invalid")
    geometry = value.geometry
    expected_rows = {
        "full": (geometry.full_monthly_rows, geometry.full_daily_rows),
        "design": (geometry.design_monthly_rows, geometry.design_daily_rows),
        "holdout": (geometry.holdout_monthly_rows, geometry.holdout_daily_rows),
    }
    for role, item in (
        ("full", value.full),
        ("design", value.design),
        ("holdout", value.holdout),
    ):
        _validate_calibration_slice(item, role, *expected_rows[role])
    _validate_calibration_source_array_hashes(value)


def _validate_calibration_source_array_hashes(value: CalibrationInput) -> None:
    records = {
        item.relative_path: item
        for item in value.provenance.source_files
        if item.layer == "processed" and item.relative_path in _CALIBRATION_ARRAY_PATHS
    }
    if set(records) != set(_CALIBRATION_ARRAY_PATHS):
        raise IntegrityError("calibration input lacks exact processed-array identities")
    for path, field_name in _CALIBRATION_ARRAY_PATHS.items():
        array = getattr(value.full, field_name)
        content = _npy_bytes(array)
        record = records[path]
        if (
            record.bytes != len(content)
            or record.sha256 != hashlib.sha256(content).hexdigest()
        ):
            raise IntegrityError(
                "calibration input array differs from processed snapshot identity",
                details={"relative_path": path},
            )


def _validate_calibration_slice(
    value: CalibrationSlice,
    role: str,
    monthly_rows: int,
    daily_rows: int,
) -> None:
    if not isinstance(value, CalibrationSlice) or value.role != role:
        raise IntegrityError("calibration-input slice role is invalid")
    specifications: tuple[
        tuple[str, np.ndarray[Any, Any], tuple[int, ...], np.dtype[Any]], ...
    ] = (
        (
            "monthly_returns",
            value.monthly_returns,
            (monthly_rows, ASSET_COUNT),
            np.dtype(np.float64),
        ),
        (
            "daily_returns",
            value.daily_returns,
            (daily_rows, ASSET_COUNT),
            np.dtype(np.float64),
        ),
        (
            "macro_features",
            value.macro_features,
            (monthly_rows, 4),
            np.dtype(np.float64),
        ),
        (
            "monthly_period_ordinals",
            value.monthly_period_ordinals,
            (monthly_rows,),
            np.dtype(np.int64),
        ),
        ("daily_session_ns", value.daily_session_ns, (daily_rows,), np.dtype(np.int64)),
        (
            "monthly_continuation",
            value.monthly_continuation,
            (monthly_rows,),
            np.dtype(np.bool_),
        ),
        (
            "daily_continuation",
            value.daily_continuation,
            (daily_rows,),
            np.dtype(np.bool_),
        ),
    )
    for name, array, shape, dtype in specifications:
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            or array.flags.writeable
            or not _has_bytes_owner(array)
        ):
            raise IntegrityError(
                "calibration-input array is not an immutable canonical materialization",
                details={"role": role, "array": name},
            )
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise IntegrityError("calibration-input array contains a non-finite value")
    if monthly_rows and bool(value.monthly_continuation[0]):
        raise IntegrityError(
            "monthly calibration continuation must restart at row zero"
        )
    if daily_rows and bool(value.daily_continuation[0]):
        raise IntegrityError("daily calibration continuation must restart at row zero")
    if monthly_rows > 1 and not bool(
        np.all(np.diff(value.monthly_period_ordinals) > 0)
    ):
        raise IntegrityError("calibration monthly ordinals are not strictly increasing")
    if daily_rows > 1 and not bool(np.all(np.diff(value.daily_session_ns) > 0)):
        raise IntegrityError("calibration daily sessions are not strictly increasing")


def _validated_policy(value: object) -> ScientificGenerationPolicy:
    if not isinstance(value, ScientificGenerationPolicy):
        raise IntegrityError("production artifact generation policy is not typed")
    try:
        rebuilt = generation_policy_from_mapping(value.as_dict())
    except (IntegrityError, ModelError) as error:
        raise IntegrityError(
            "production artifact generation policy is invalid"
        ) from error
    if rebuilt != value:
        raise IntegrityError("production artifact generation policy is noncanonical")
    return rebuilt


def _validate_policy_against_config(
    policy: ScientificGenerationPolicy, config: PRPGConfig
) -> None:
    if (
        policy.alpha_policy not in config.model.alpha_policy_candidates
        or tuple(policy.neutral_lambda_grid) != tuple(config.model.neutral_lambda_grid)
        or policy.initial_state != config.model.initial_state
        or policy.synchronized_return_vectors is not True
        or (
            policy.alpha_policy == "historical_vector"
            and policy.selected_neutral_lambda is not None
        )
        or (
            policy.alpha_policy == "kernel_mean_neutral"
            and policy.selected_neutral_lambda not in config.model.neutral_lambda_grid
        )
    ):
        raise IntegrityError(
            "generation policy is outside the resolved G3 candidate configuration"
        )


def _validate_rehearsal_policy_candidate(
    policy: ScientificGenerationPolicy,
    artifact: LoadedScientificModelArtifact,
) -> None:
    candidates = artifact.generation_policy
    if not isinstance(candidates, ScientificCandidatePolicySet):
        raise IntegrityError("rehearsal requires the exact G3 candidate-policy set")
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
        raise IntegrityError("rehearsal policy is outside the G3 candidate set")


def _validate_live_artifact_arrays(artifact: LoadedScientificModelArtifact) -> None:
    identity = artifact.reference.manifest.get("identity")
    parameters = identity.get("parameters") if isinstance(identity, Mapping) else None
    records = parameters.get("arrays") if isinstance(parameters, Mapping) else None
    if not isinstance(records, tuple | list):
        raise IntegrityError("production artifact array records are unavailable")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise IntegrityError("production artifact array record is invalid")
        indexed[record["name"]] = record
    if set(indexed) != set(artifact.arrays):
        raise IntegrityError(
            "production artifact live-array allowlist differs from manifest"
        )
    for name, array in artifact.arrays.items():
        record = indexed[name]
        if (
            record.get("dtype") != array.dtype.str
            or tuple(record.get("shape", ())) != array.shape
            or record.get("sha256") != _npy_sha256(array)
            or array.flags.writeable
            or not _has_bytes_owner(array)
        ):
            raise IntegrityError(
                "production artifact live array differs from verified bytes",
                details={"array": name},
            )


def _validate_artifact_input_geometry(
    artifact: LoadedScientificModelArtifact,
    calibration_input: CalibrationInput,
) -> None:
    geometry = calibration_input.geometry
    if (
        artifact.geometry.monthly_rows != geometry.full_monthly_rows
        or artifact.geometry.daily_rows != geometry.full_daily_rows
        or artifact.arrays["monthly_source_states"].shape
        != (geometry.full_monthly_rows,)
        or artifact.arrays["daily_source_states"].shape != (geometry.full_daily_rows,)
        or artifact.arrays["monthly_continuation"].shape
        != (geometry.full_monthly_rows,)
        or artifact.arrays["daily_continuation"].shape != (geometry.full_daily_rows,)
    ):
        raise IntegrityError(
            "production state/pool geometry differs from calibration input"
        )
    if not np.array_equal(
        artifact.arrays["monthly_continuation"],
        calibration_input.full.monthly_continuation,
    ) or not np.array_equal(
        artifact.arrays["daily_continuation"],
        calibration_input.full.daily_continuation,
    ):
        raise IntegrityError(
            "production continuation identity differs from calibration input"
        )
    for frequency in ("monthly", "daily"):
        states = artifact.arrays[f"{frequency}_source_states"]
        if bool((states < 0).any()) or bool((states >= artifact.selected_k).any()):
            raise IntegrityError("production source state is outside selected K")
        counts = np.bincount(states, minlength=artifact.selected_k)
        if not np.array_equal(counts, artifact.arrays[f"{frequency}_state_counts"]):
            raise IntegrityError(
                "production state counts differ from source-state identity"
            )


def _generation_input_metadata(
    *,
    config: PRPGConfig,
    evidence: LoadedG3EvidenceAuthority,
    artifact: LoadedScientificModelArtifact,
    calibration_input: CalibrationInput,
    policy: ScientificGenerationPolicy,
    execution_mode: Literal["canonical", "rehearsal"],
    generation_authorized: bool,
    rehearsal_scope_fingerprint: str | None,
    g5_authority_schema_id: str | None,
    g5_authority_fingerprint: str | None,
    g5_threshold_registry_fingerprint: str | None,
    g5_qualification_results_fingerprint: str | None,
    g5_meta_validation_fingerprint: str | None,
) -> dict[str, object]:
    return {
        "schema_version": SERIAL_GENERATION_INPUT_SCHEMA_VERSION,
        "generator_version": SERIAL_GENERATOR_VERSION,
        "execution_mode": execution_mode,
        "generation_authorized": generation_authorized,
        "rehearsal_scope_fingerprint": rehearsal_scope_fingerprint,
        "config_fingerprint": config.fingerprint(),
        "processed_data_fingerprint": (
            evidence.evidence_bundle.processed_data_fingerprint
        ),
        "calibration_input_fingerprint": calibration_input.fingerprint,
        "g3_bundle_fingerprint": evidence.evidence_bundle.fingerprint,
        "durable_evidence_fingerprint": evidence.reference.fingerprint,
        "pair_receipt_fingerprint": evidence.pair_receipt_fingerprint,
        "g5_authority_schema_id": g5_authority_schema_id,
        "g5_authority_fingerprint": g5_authority_fingerprint,
        "g5_threshold_registry_fingerprint": g5_threshold_registry_fingerprint,
        "g5_qualification_results_fingerprint": (g5_qualification_results_fingerprint),
        "g5_meta_validation_fingerprint": g5_meta_validation_fingerprint,
        "production_artifact_fingerprint": artifact.reference.fingerprint,
        "core_model_fingerprint": artifact.core_model_fingerprint,
        "parameter_set_fingerprint": artifact.parameter_set_fingerprint,
        "master_seed": config.run.master_seed,
        "scientific_version": config.simulation.scientific_version,
        "selected_k": artifact.selected_k,
        "configured_lf_paths": config.simulation.low_frequency.paths,
        "configured_hf_paths": config.simulation.high_frequency.paths,
        "configured_years": config.simulation.low_frequency.years,
        "generation_policy": policy.as_dict(),
    }


def _verify_generation_input(value: object) -> None:
    _verify_generation_input_common(value, rehearsal=False)


def _verify_rehearsal_generation_input(value: object) -> None:
    _verify_generation_input_common(value, rehearsal=True)


def _verify_generation_input_common(value: object, *, rehearsal: bool) -> None:
    from prpg.model.lean_g5_generation_authority import (
        LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID,
    )
    from prpg.model.practical_k4 import PRACTICAL_K4_AUTHORITY_SCHEMA_ID

    expected_type = (
        RehearsalSerialGenerationInput if rehearsal else SerialGenerationInput
    )
    expected_seal = _REHEARSAL_INPUT_SEAL if rehearsal else _GENERATION_INPUT_SEAL
    if type(value) is not expected_type:
        raise GenerationError(
            "serial generator input was not minted by the required boundary"
        )
    assert isinstance(value, SerialGenerationInput)
    if (
        value._seal is not expected_seal
        or value._owner_id != id(value)
        or value.schema_version != SERIAL_GENERATION_INPUT_SCHEMA_VERSION
        or value.generator_version != SERIAL_GENERATOR_VERSION
        or value.execution_mode != ("rehearsal" if rehearsal else "canonical")
        or value.generation_authorized is rehearsal
        or value.scientific_version <= 0
        or value.selected_k not in {2, 3, 4, 5}
    ):
        raise IntegrityError("serial generator input process seal is invalid")
    for fingerprint in (
        value.config_fingerprint,
        value.processed_data_fingerprint,
        value.calibration_input_fingerprint,
        value.g3_bundle_fingerprint,
        value.durable_evidence_fingerprint,
        value.pair_receipt_fingerprint,
        value.production_artifact_fingerprint,
        value.core_model_fingerprint,
        value.parameter_set_fingerprint,
        value.fingerprint,
    ):
        if (
            not isinstance(fingerprint, str)
            or _FINGERPRINT.fullmatch(fingerprint) is None
        ):
            raise IntegrityError(
                "serial generator input contains an invalid fingerprint"
            )
    g5_fingerprints = (
        value.g5_authority_fingerprint,
        value.g5_threshold_registry_fingerprint,
        value.g5_qualification_results_fingerprint,
        value.g5_meta_validation_fingerprint,
    )
    if rehearsal:
        if value.g5_authority_schema_id is not None or any(
            item is not None for item in g5_fingerprints
        ):
            raise IntegrityError("rehearsal input must not claim G5 qualification")
        _required_sha256(value.rehearsal_scope_fingerprint, "rehearsal scope")
    elif value.rehearsal_scope_fingerprint is not None:
        raise IntegrityError("canonical input claims rehearsal scope")
    elif value.g5_authority_schema_id == G5_GENERATION_AUTHORITY_SCHEMA_ID:
        if any(
            not isinstance(item, str) or _FINGERPRINT.fullmatch(item) is None
            for item in g5_fingerprints
        ):
            raise IntegrityError("canonical input lacks exact legacy G5 qualification")
    elif value.g5_authority_schema_id == LEAN_G5_PRODUCTION_AUTHORITY_SCHEMA_ID:
        if (
            any(
                not isinstance(item, str) or _FINGERPRINT.fullmatch(item) is None
                for item in g5_fingerprints[:3]
            )
            or value.g5_meta_validation_fingerprint is not None
        ):
            raise IntegrityError("canonical input lacks exact lean G5 qualification")
    elif value.g5_authority_schema_id == PRACTICAL_K4_AUTHORITY_SCHEMA_ID:
        if (
            not isinstance(value.g5_authority_fingerprint, str)
            or _FINGERPRINT.fullmatch(value.g5_authority_fingerprint) is None
            or any(item is not None for item in g5_fingerprints[1:])
            or value.g3_bundle_fingerprint != value.g5_authority_fingerprint
            or value.durable_evidence_fingerprint != value.g5_authority_fingerprint
            or value.production_artifact_fingerprint != value.g5_authority_fingerprint
            or value.core_model_fingerprint != value.parameter_set_fingerprint
        ):
            raise IntegrityError("canonical input lacks exact practical K=4 authority")
    else:
        raise IntegrityError("canonical input G5 authority schema is unsupported")
    policy = _validated_policy(value.generation_policy)
    arrays = _generation_input_arrays(value)
    if set(value.array_fingerprints) != set(arrays):
        raise IntegrityError(
            "serial generator input array-fingerprint map is incomplete"
        )
    for name, array in arrays.items():
        if array.flags.writeable or not _has_bytes_owner(array):
            raise IntegrityError("serial generator input array is not immutable")
        if value.array_fingerprints[name] != _array_fingerprint(array, name):
            raise IntegrityError("serial generator input array fingerprint changed")
    metadata = {
        "schema_version": value.schema_version,
        "generator_version": value.generator_version,
        "execution_mode": value.execution_mode,
        "generation_authorized": value.generation_authorized,
        "rehearsal_scope_fingerprint": value.rehearsal_scope_fingerprint,
        "config_fingerprint": value.config_fingerprint,
        "processed_data_fingerprint": value.processed_data_fingerprint,
        "calibration_input_fingerprint": value.calibration_input_fingerprint,
        "g3_bundle_fingerprint": value.g3_bundle_fingerprint,
        "durable_evidence_fingerprint": value.durable_evidence_fingerprint,
        "pair_receipt_fingerprint": value.pair_receipt_fingerprint,
        "g5_authority_schema_id": value.g5_authority_schema_id,
        "g5_authority_fingerprint": value.g5_authority_fingerprint,
        "g5_threshold_registry_fingerprint": (value.g5_threshold_registry_fingerprint),
        "g5_qualification_results_fingerprint": (
            value.g5_qualification_results_fingerprint
        ),
        "g5_meta_validation_fingerprint": value.g5_meta_validation_fingerprint,
        "production_artifact_fingerprint": value.production_artifact_fingerprint,
        "core_model_fingerprint": value.core_model_fingerprint,
        "parameter_set_fingerprint": value.parameter_set_fingerprint,
        "master_seed": value.master_seed,
        "scientific_version": value.scientific_version,
        "selected_k": value.selected_k,
        "configured_lf_paths": value.configured_lf_paths,
        "configured_hf_paths": value.configured_hf_paths,
        "configured_years": value.configured_years,
        "generation_policy": policy.as_dict(),
    }
    if value.fingerprint != _record_fingerprint(metadata, value.array_fingerprints):
        raise IntegrityError("serial generator input fingerprint changed")
    if value.transition_matrix.shape != (value.selected_k, value.selected_k):
        raise IntegrityError("serial generator transition geometry changed")
    chain = validate_markov_chain(value.transition_matrix)
    if not np.array_equal(value.stationary_distribution, chain.stationary_distribution):
        raise IntegrityError("serial generator stationary distribution changed")
    if (
        value.monthly_returns.shape != (value.monthly_source_states.size, ASSET_COUNT)
        or value.daily_returns.shape != (value.daily_source_states.size, ASSET_COUNT)
        or value.monthly_continuation_links.size != value.monthly_source_states.size - 1
        or value.daily_continuation_links.size != value.daily_source_states.size - 1
    ):
        raise IntegrityError("serial generator source-pool geometry changed")


def _generation_input_arrays(
    value: SerialGenerationInput,
) -> dict[str, np.ndarray[Any, Any]]:
    return {
        name: getattr(value, name)
        for name in (
            "stationary_distribution",
            "transition_matrix",
            "monthly_returns",
            "daily_returns",
            "monthly_source_states",
            "daily_source_states",
            "monthly_continuation_links",
            "daily_continuation_links",
            "monthly_kernel_lambdas",
            "daily_kernel_lambdas",
            "monthly_kernel_offsets",
            "daily_kernel_offsets",
        )
    }


def _mint_g5_meta_serial_numerical_input(
    *,
    source_generation_input_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    selected_k: int,
    configured_years: int,
    generation_policy: ScientificGenerationPolicy,
    stationary_distribution: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    monthly_returns: npt.ArrayLike,
    daily_returns: npt.ArrayLike,
    monthly_source_states: npt.ArrayLike,
    daily_source_states: npt.ArrayLike,
    monthly_continuation_links: npt.ArrayLike,
    daily_continuation_links: npt.ArrayLike,
    monthly_kernel_lambdas: npt.ArrayLike,
    daily_kernel_lambdas: npt.ArrayLike,
    monthly_kernel_offsets: npt.ArrayLike,
    daily_kernel_offsets: npt.ArrayLike,
) -> G5MetaSerialNumericalInput:
    """Mint the process-portable numerical view after boundary validation."""

    source = _required_sha256(
        source_generation_input_fingerprint, "meta source generation input"
    )
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or master_seed < 0
    ):
        raise GenerationError("meta master seed must be a non-negative integer")
    version = _positive_int(scientific_version, "meta scientific version")
    states = _positive_int(selected_k, "meta selected K")
    if states not in {2, 3, 4, 5}:
        raise GenerationError("meta selected K is outside the closed registry")
    years = _positive_int(configured_years, "meta configured years")
    policy = _validated_policy(generation_policy)
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "stationary_distribution": _immutable_array(
            stationary_distribution, dtype=np.float64, label="meta stationary law"
        ),
        "transition_matrix": _immutable_array(
            transition_matrix, dtype=np.float64, label="meta transition matrix"
        ),
        "monthly_returns": _immutable_array(
            monthly_returns, dtype=np.float64, label="meta monthly returns"
        ),
        "daily_returns": _immutable_array(
            daily_returns, dtype=np.float64, label="meta daily returns"
        ),
        "monthly_source_states": _immutable_array(
            monthly_source_states,
            dtype=np.int64,
            label="meta monthly source states",
        ),
        "daily_source_states": _immutable_array(
            daily_source_states, dtype=np.int64, label="meta daily source states"
        ),
        "monthly_continuation_links": _immutable_array(
            monthly_continuation_links,
            dtype=np.bool_,
            label="meta monthly continuation links",
        ),
        "daily_continuation_links": _immutable_array(
            daily_continuation_links,
            dtype=np.bool_,
            label="meta daily continuation links",
        ),
        "monthly_kernel_lambdas": _immutable_array(
            monthly_kernel_lambdas,
            dtype=np.float64,
            label="meta monthly kernel lambdas",
        ),
        "daily_kernel_lambdas": _immutable_array(
            daily_kernel_lambdas,
            dtype=np.float64,
            label="meta daily kernel lambdas",
        ),
        "monthly_kernel_offsets": _immutable_array(
            monthly_kernel_offsets,
            dtype=np.float64,
            label="meta monthly kernel offsets",
        ),
        "daily_kernel_offsets": _immutable_array(
            daily_kernel_offsets,
            dtype=np.float64,
            label="meta daily kernel offsets",
        ),
    }
    fingerprints = tuple(
        (name, _array_fingerprint(array, name))
        for name, array in sorted(arrays.items())
    )
    metadata = {
        "schema_id": G5_META_SERIAL_INPUT_SCHEMA_ID,
        "schema_version": G5_META_SERIAL_INPUT_SCHEMA_VERSION,
        "generation_authorized": False,
        "source_generation_input_fingerprint": source,
        "master_seed": master_seed,
        "scientific_version": version,
        "selected_k": states,
        "configured_years": years,
        "generation_policy": policy.as_dict(),
    }
    result = G5MetaSerialNumericalInput(
        schema_id=G5_META_SERIAL_INPUT_SCHEMA_ID,
        schema_version=G5_META_SERIAL_INPUT_SCHEMA_VERSION,
        generation_authorized=False,
        source_generation_input_fingerprint=source,
        master_seed=master_seed,
        scientific_version=version,
        selected_k=states,
        configured_years=years,
        generation_policy=policy,
        stationary_distribution=arrays["stationary_distribution"],
        transition_matrix=arrays["transition_matrix"],
        monthly_returns=arrays["monthly_returns"],
        daily_returns=arrays["daily_returns"],
        monthly_source_states=arrays["monthly_source_states"],
        daily_source_states=arrays["daily_source_states"],
        monthly_continuation_links=arrays["monthly_continuation_links"],
        daily_continuation_links=arrays["daily_continuation_links"],
        monthly_kernel_lambdas=arrays["monthly_kernel_lambdas"],
        daily_kernel_lambdas=arrays["daily_kernel_lambdas"],
        monthly_kernel_offsets=arrays["monthly_kernel_offsets"],
        daily_kernel_offsets=arrays["daily_kernel_offsets"],
        array_fingerprints=fingerprints,
        fingerprint=_record_fingerprint(metadata, dict(fingerprints)),
    )
    return verify_g5_meta_serial_numerical_input(result)


def _verify_g5_meta_serial_numerical_input(value: object) -> None:
    if type(value) is not G5MetaSerialNumericalInput:
        raise GenerationError("G5 meta serial input has the wrong type")
    assert isinstance(value, G5MetaSerialNumericalInput)
    if (
        value.schema_id != G5_META_SERIAL_INPUT_SCHEMA_ID
        or value.schema_version != G5_META_SERIAL_INPUT_SCHEMA_VERSION
        or value.generation_authorized is not False
    ):
        raise IntegrityError("G5 meta serial input schema changed")
    source = _required_sha256(
        value.source_generation_input_fingerprint, "meta source generation input"
    )
    if (
        isinstance(value.master_seed, bool)
        or not isinstance(value.master_seed, int)
        or value.master_seed < 0
    ):
        raise IntegrityError("G5 meta serial master seed changed")
    version = _positive_int(value.scientific_version, "meta scientific version")
    states = _positive_int(value.selected_k, "meta selected K")
    years = _positive_int(value.configured_years, "meta configured years")
    if states not in {2, 3, 4, 5}:
        raise IntegrityError("G5 meta serial K changed")
    policy = _validated_policy(value.generation_policy)
    arrays = {
        name: getattr(value, name)
        for name in (
            "stationary_distribution",
            "transition_matrix",
            "monthly_returns",
            "daily_returns",
            "monthly_source_states",
            "daily_source_states",
            "monthly_continuation_links",
            "daily_continuation_links",
            "monthly_kernel_lambdas",
            "daily_kernel_lambdas",
            "monthly_kernel_offsets",
            "daily_kernel_offsets",
        )
    }
    expected_dtypes = {
        "stationary_distribution": np.dtype(np.float64),
        "transition_matrix": np.dtype(np.float64),
        "monthly_returns": np.dtype(np.float64),
        "daily_returns": np.dtype(np.float64),
        "monthly_source_states": np.dtype(np.int64),
        "daily_source_states": np.dtype(np.int64),
        "monthly_continuation_links": np.dtype(np.bool_),
        "daily_continuation_links": np.dtype(np.bool_),
        "monthly_kernel_lambdas": np.dtype(np.float64),
        "daily_kernel_lambdas": np.dtype(np.float64),
        "monthly_kernel_offsets": np.dtype(np.float64),
        "daily_kernel_offsets": np.dtype(np.float64),
    }
    for name, array in arrays.items():
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != expected_dtypes[name]
            or not array.flags.c_contiguous
            or (array.dtype.kind == "f" and not bool(np.isfinite(array).all()))
        ):
            raise IntegrityError("G5 meta serial array changed")
    transition = arrays["transition_matrix"]
    stationary = arrays["stationary_distribution"]
    monthly = arrays["monthly_returns"]
    daily = arrays["daily_returns"]
    monthly_states = arrays["monthly_source_states"]
    daily_states = arrays["daily_source_states"]
    if (
        stationary.shape != (states,)
        or transition.shape != (states, states)
        or monthly.ndim != 2
        or monthly.shape[1] != ASSET_COUNT
        or daily.ndim != 2
        or daily.shape[1] != ASSET_COUNT
        or monthly_states.shape != (monthly.shape[0],)
        or daily_states.shape != (daily.shape[0],)
        or arrays["monthly_continuation_links"].shape != (monthly.shape[0] - 1,)
        or arrays["daily_continuation_links"].shape != (daily.shape[0] - 1,)
        or arrays["monthly_kernel_lambdas"].ndim != 1
        or arrays["daily_kernel_lambdas"].ndim != 1
        or arrays["monthly_kernel_offsets"].shape
        != (arrays["monthly_kernel_lambdas"].size, states, ASSET_COUNT)
        or arrays["daily_kernel_offsets"].shape
        != (arrays["daily_kernel_lambdas"].size, states, ASSET_COUNT)
        or bool((monthly_states < 0).any())
        or bool((monthly_states >= states).any())
        or bool((daily_states < 0).any())
        or bool((daily_states >= states).any())
    ):
        raise IntegrityError("G5 meta serial geometry changed")
    chain = validate_markov_chain(transition)
    if not np.array_equal(stationary, chain.stationary_distribution):
        raise IntegrityError("G5 meta serial stationary law changed")
    observed_fingerprints = tuple(
        (name, _array_fingerprint(array, name))
        for name, array in sorted(arrays.items())
    )
    if value.array_fingerprints != observed_fingerprints:
        raise IntegrityError("G5 meta serial array fingerprint changed")
    metadata = {
        "schema_id": value.schema_id,
        "schema_version": value.schema_version,
        "generation_authorized": value.generation_authorized,
        "source_generation_input_fingerprint": source,
        "master_seed": value.master_seed,
        "scientific_version": version,
        "selected_k": states,
        "configured_years": years,
        "generation_policy": policy.as_dict(),
    }
    if value.fingerprint != _record_fingerprint(metadata, dict(observed_fingerprints)):
        raise IntegrityError("G5 meta serial input fingerprint changed")


def _verify_base_path(value: object) -> None:
    _verify_base_path_common(value, rehearsal=False)


def _verify_rehearsal_base_path(value: object) -> None:
    _verify_base_path_common(value, rehearsal=True)


def _verify_base_path_common(value: object, *, rehearsal: bool) -> None:
    expected_type = RehearsalSerialBasePath if rehearsal else SerialBasePath
    expected_seal = _REHEARSAL_BASE_PATH_SEAL if rehearsal else _BASE_PATH_SEAL
    if type(value) is not expected_type:
        raise GenerationError("serial base path was not minted by the generator")
    assert isinstance(value, SerialBasePath)
    if (
        value._seal is not expected_seal
        or value._owner_id != id(value)
        or value.schema_version != SERIAL_BASE_PATH_SCHEMA_VERSION
        or value.generator_version != SERIAL_GENERATOR_VERSION
        or value.execution_mode != ("rehearsal" if rehearsal else "canonical")
        or value.generation_authorized is rehearsal
    ):
        raise IntegrityError("serial base-path process seal is invalid")
    family = _family(value.family)
    expected_frequency = "monthly" if family == "LF" else "daily"
    if value.base_frequency != expected_frequency:
        raise IntegrityError("serial base-path family/frequency binding changed")
    years = _positive_int(value.years, "path years")
    _path_entity(value.path_entity, maximum=UINT32_MAX)
    monthly_count = MONTHS_PER_YEAR * years
    base_count = monthly_count if family == "LF" else SESSIONS_PER_YEAR * years
    audit = value.restart_audit
    if not isinstance(audit, RestartAudit) or len(audit.reasons) != base_count:
        raise IntegrityError("serial base-path restart audit is invalid")
    arrays = {
        "log_returns": value.log_returns,
        "target_monthly_states": value.target_monthly_states,
        "target_states": value.target_states,
        "source_indices": value.source_indices,
        "restart_flags": audit.restart_flags,
        "restart_uniform_used": audit.restart_uniform_used,
        "source_rank_uniform_used": audit.source_rank_uniform_used,
    }
    expected_shapes = {
        "log_returns": (base_count, ASSET_COUNT),
        "target_monthly_states": (monthly_count,),
        "target_states": (base_count,),
        "source_indices": (base_count,),
        "restart_flags": (base_count,),
        "restart_uniform_used": (base_count,),
        "source_rank_uniform_used": (base_count,),
    }
    for name, array in arrays.items():
        if (
            not isinstance(array, np.ndarray)
            or array.shape != expected_shapes[name]
            or array.flags.writeable
            or not _has_bytes_owner(array)
        ):
            raise IntegrityError("serial base-path array is invalid")
    if not bool(np.isfinite(value.log_returns).all()):
        raise IntegrityError("serial base-path return is non-finite")
    if family == "HF" and not np.array_equal(
        value.target_states, np.repeat(value.target_monthly_states, SESSIONS_PER_MONTH)
    ):
        raise IntegrityError("HF target-state expansion changed")
    if family == "LF" and not np.array_equal(
        value.target_states, value.target_monthly_states
    ):
        raise IntegrityError("LF target-state identity changed")
    if (
        value.state_uniforms_consumed != monthly_count
        or audit.restart_uniforms_consumed != base_count
        or audit.source_rank_uniforms_consumed != base_count
    ):
        raise IntegrityError("serial base-path RNG consumption changed")
    metadata = {
        "schema_version": value.schema_version,
        "generator_version": value.generator_version,
        "execution_mode": value.execution_mode,
        "generation_authorized": value.generation_authorized,
        "generation_input_fingerprint": value.generation_input_fingerprint,
        "family": family,
        "base_frequency": value.base_frequency,
        "path_entity": value.path_entity,
        "years": years,
        "state_uniforms_consumed": value.state_uniforms_consumed,
        "restart_uniforms_consumed": audit.restart_uniforms_consumed,
        "source_rank_uniforms_consumed": audit.source_rank_uniforms_consumed,
        "restart_reasons": [item.value for item in audit.reasons],
    }
    fingerprints = {
        name: _array_fingerprint(array, name) for name, array in sorted(arrays.items())
    }
    if value.fingerprint != _record_fingerprint(metadata, fingerprints):
        raise IntegrityError("serial base-path fingerprint changed")


def _verify_g5_candidate_generation_input(value: object) -> None:
    if (
        type(value) is not G5CandidateSerialGenerationInput
        or value._seal is not _G5_CANDIDATE_INPUT_SEAL
        or value._owner_id != id(value)
        or _G5_CANDIDATE_INPUTS.get(id(value)) is not value
    ):
        raise GenerationError("G5 candidate input was not minted by its builder")
    evidence = verify_loaded_g3_evidence(value.g3_evidence)
    _verify_rehearsal_generation_input(value.numerical_input)
    policy = _validated_policy(value.generation_policy)
    stage = _g5_candidate_stage(value.candidate_stage)
    authority: G5QualificationGenerationAuthority | None
    permit: LeanG5QualificationPermit | None
    if stage == "development":
        if (
            value.qualification_authority is not None
            or value.lean_qualification_permit is not None
        ):
            raise IntegrityError("G5 development acquired qualification authority")
        authority = None
        permit = None
        expected_domain = Domain.DEVELOPMENT
        expected_lf = G5_DEVELOPMENT_LF_PATH_COUNT
        expected_hf = G5_DEVELOPMENT_HF_PATH_COUNT
        expected_version = value.numerical_input.scientific_version
        if expected_version != 11:
            raise IntegrityError("G5 development common-random version changed")
    else:
        if value.lean_qualification_permit is not None:
            if value.qualification_authority is not None:
                raise IntegrityError("G5 qualification has overlapping permits")
            permit = verify_lean_g5_qualification_permit(
                value.lean_qualification_permit
            )
            authority = None
            if permit.g3_evidence is not evidence or policy != permit.generation_policy:
                raise IntegrityError("lean G5 qualification policy/G3 binding changed")
            expected_version = permit.policy_scientific_version
        else:
            permit = None
            authority = verify_g5_qualification_generation_authority(
                value.qualification_authority
            )
            if (
                authority.g3_evidence is not evidence
                or policy != authority.generation_policy
            ):
                raise IntegrityError("G5 qualification policy/authority changed")
            expected_version = authority.policy_scientific_version
        expected_domain = Domain.QUALIFICATION
        expected_lf = G5_QUALIFICATION_LF_PATH_COUNT
        expected_hf = G5_QUALIFICATION_HF_PATH_COUNT
    _validate_rehearsal_policy_candidate(policy, evidence.production_artifact)
    for fingerprint in (
        value.g3_evidence_fingerprint,
        value.generation_policy_fingerprint,
        value.numerical_input_fingerprint,
        value.fingerprint,
    ):
        _required_sha256(fingerprint, "G5 candidate input")
    if authority is None and permit is None:
        if (
            value.qualification_authority_fingerprint is not None
            or value.lean_qualification_permit_fingerprint is not None
            or value.threshold_evidence_fingerprint is not None
            or value.threshold_registry_fingerprint is not None
        ):
            raise IntegrityError("G5 development claims threshold evidence")
    elif permit is not None:
        _required_sha256(
            value.lean_qualification_permit_fingerprint,
            "lean G5 qualification permit",
        )
        if (
            value.qualification_authority_fingerprint is not None
            or value.threshold_evidence_fingerprint is not None
            or value.threshold_registry_fingerprint is not None
        ):
            raise IntegrityError("lean G5 qualification claims legacy thresholds")
    else:
        if value.lean_qualification_permit_fingerprint is not None:
            raise IntegrityError("legacy G5 qualification claims a lean permit")
        for optional_fingerprint in (
            value.qualification_authority_fingerprint,
            value.threshold_evidence_fingerprint,
            value.threshold_registry_fingerprint,
        ):
            _required_sha256(optional_fingerprint, "G5 qualification input")
    expected = _g5_candidate_input_identity(
        authority=authority,
        lean_permit=permit,
        g3_evidence=evidence,
        policy=policy,
        candidate_stage=stage,
        rng_domain=expected_domain,
        policy_scientific_version=expected_version,
        maximum_lf_paths=expected_lf,
        maximum_hf_paths=expected_hf,
        configured_years=value.numerical_input.configured_years,
        numerical_input_fingerprint=value.numerical_input.fingerprint,
    )
    if (
        value.schema_id != G5_CANDIDATE_INPUT_SCHEMA_ID
        or value.schema_version != G5_CANDIDATE_INPUT_SCHEMA_VERSION
        or value.generation_authorized is not False
        or value.rng_domain != expected_domain
        or value.qualification_authority_fingerprint
        != (None if authority is None else authority.fingerprint)
        or value.lean_qualification_permit_fingerprint
        != (None if permit is None else permit.fingerprint)
        or value.g3_evidence_fingerprint != evidence.reference.fingerprint
        or value.threshold_evidence_fingerprint
        != (None if authority is None else authority.threshold_evidence_fingerprint)
        or value.threshold_registry_fingerprint
        != (None if authority is None else authority.threshold_registry_fingerprint)
        or value.generation_policy_fingerprint
        != _identity_fingerprint(policy.as_dict())
        or value.policy_scientific_version != expected_version
        or value.maximum_lf_paths != expected_lf
        or value.maximum_hf_paths != expected_hf
        or value.configured_years != value.numerical_input.configured_years
        or value.numerical_input_fingerprint != value.numerical_input.fingerprint
        or value.numerical_input.generation_policy != policy
        or value.numerical_input.durable_evidence_fingerprint
        != evidence.reference.fingerprint
        or value.fingerprint != _identity_fingerprint(expected)
    ):
        raise IntegrityError("G5 candidate input identity changed")


def _verify_g5_candidate_base_path(value: object) -> None:
    if (
        type(value) is not G5CandidateSerialBasePath
        or value._seal is not _G5_CANDIDATE_PATH_SEAL
        or value._owner_id != id(value)
        or _G5_CANDIDATE_PATHS.get(id(value)) is not value
    ):
        raise GenerationError("G5 candidate path was not minted by its generator")
    generation_input = verify_g5_candidate_serial_generation_input(
        value.generation_input
    )
    _verify_rehearsal_base_path(value.base_path)
    family = _family(value.family)
    maximum = (
        generation_input.maximum_lf_paths
        if family == "LF"
        else generation_input.maximum_hf_paths
    )
    _path_entity(value.path_entity, maximum=maximum)
    expected = _g5_candidate_path_identity(
        generation_input=generation_input,
        family=family,
        path_entity=value.path_entity,
        base_path=value.base_path,
    )
    if (
        value.schema_id != G5_CANDIDATE_PATH_SCHEMA_ID
        or value.schema_version != G5_CANDIDATE_PATH_SCHEMA_VERSION
        or value.generation_authorized is not False
        or value.candidate_stage != generation_input.candidate_stage
        or value.rng_domain != generation_input.rng_domain
        or value.generation_input_fingerprint != generation_input.fingerprint
        or value.generation_policy_fingerprint
        != generation_input.generation_policy_fingerprint
        or value.base_frequency != ("monthly" if family == "LF" else "daily")
        or value.years != generation_input.configured_years
        or value.base_path.family != family
        or value.base_path.path_entity != value.path_entity
        or value.base_path.years != value.years
        or value.base_path.generation_input_fingerprint
        != generation_input.numerical_input_fingerprint
        or value.base_path_fingerprint != value.base_path.fingerprint
        or value.fingerprint != _identity_fingerprint(expected)
    ):
        raise IntegrityError("G5 candidate base-path identity changed")


def _g5_candidate_input_identity(
    *,
    authority: G5QualificationGenerationAuthority | None,
    lean_permit: LeanG5QualificationPermit | None,
    g3_evidence: LoadedG3EvidenceAuthority,
    policy: ScientificGenerationPolicy,
    candidate_stage: G5CandidateStage,
    rng_domain: Domain,
    policy_scientific_version: int,
    maximum_lf_paths: int,
    maximum_hf_paths: int,
    configured_years: int,
    numerical_input_fingerprint: str,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_id": G5_CANDIDATE_INPUT_SCHEMA_ID,
        "schema_version": G5_CANDIDATE_INPUT_SCHEMA_VERSION,
        "generation_authorized": False,
        "candidate_stage": candidate_stage,
        "rng_domain": int(rng_domain),
        "qualification_authority_fingerprint": (
            None if authority is None else authority.fingerprint
        ),
        "g3_evidence_fingerprint": g3_evidence.reference.fingerprint,
        "threshold_evidence_fingerprint": (
            None if authority is None else authority.threshold_evidence_fingerprint
        ),
        "threshold_registry_fingerprint": (
            None if authority is None else authority.threshold_registry_fingerprint
        ),
        "generation_policy": policy.as_dict(),
        "generation_policy_fingerprint": _identity_fingerprint(policy.as_dict()),
        "policy_scientific_version": policy_scientific_version,
        "maximum_lf_paths": maximum_lf_paths,
        "maximum_hf_paths": maximum_hf_paths,
        "configured_years": configured_years,
        "numerical_input_fingerprint": numerical_input_fingerprint,
    }
    if lean_permit is not None:
        identity["lean_qualification_permit_fingerprint"] = lean_permit.fingerprint
        identity["lean_qualification_contract_id"] = lean_permit.contract_id
        identity["development_decision_fingerprint"] = (
            lean_permit.development_decision_fingerprint
        )
    return identity


def _g5_candidate_path_identity(
    *,
    generation_input: G5CandidateSerialGenerationInput,
    family: PathFamily,
    path_entity: int,
    base_path: RehearsalSerialBasePath,
) -> dict[str, object]:
    return {
        "schema_id": G5_CANDIDATE_PATH_SCHEMA_ID,
        "schema_version": G5_CANDIDATE_PATH_SCHEMA_VERSION,
        "generation_authorized": False,
        "candidate_stage": generation_input.candidate_stage,
        "rng_domain": int(generation_input.rng_domain),
        "generation_input_fingerprint": generation_input.fingerprint,
        "generation_policy_fingerprint": (
            generation_input.generation_policy_fingerprint
        ),
        "family": family,
        "base_frequency": base_path.base_frequency,
        "path_entity": path_entity,
        "years": base_path.years,
        "base_path_fingerprint": base_path.fingerprint,
    }


def _g5_candidate_key(
    generation_input: G5CandidateSerialGenerationInput,
    family: Family,
    entity: int,
    stage: Stage,
) -> RNGKey:
    numerical = generation_input.numerical_input
    return RNGKey(
        master_seed=numerical.master_seed,
        scientific_version=generation_input.policy_scientific_version,
        domain=generation_input.rng_domain,
        family=family,
        entity=entity,
        stage=stage,
        contract=RNG_CONTRACT_VERSION,
    )


def _g5_candidate_stage(value: object) -> G5CandidateStage:
    if value not in {"development", "qualification"}:
        raise GenerationError(
            "G5 candidate stage must be 'development' or 'qualification'"
        )
    return value  # type: ignore[return-value]


def _identity_fingerprint(value: object) -> str:
    return hashlib.sha256((_canonical_json(value) + "\n").encode()).hexdigest()


def _production_key(
    generation_input: SerialGenerationInput | RehearsalSerialGenerationInput,
    family: Family,
    entity: int,
    stage: Stage,
) -> RNGKey:
    return RNGKey(
        master_seed=generation_input.master_seed,
        scientific_version=(
            generation_input.scientific_version
            if type(generation_input) is RehearsalSerialGenerationInput
            else g5_policy_scientific_version(generation_input.generation_policy)
        ),
        domain=(
            Domain.DEVELOPMENT
            if generation_input.execution_mode == "rehearsal"
            else Domain.PRODUCTION
        ),
        family=family,
        entity=entity,
        stage=stage,
        contract=RNG_CONTRACT_VERSION,
    )


def _family(value: object) -> PathFamily:
    if value not in {"LF", "HF"}:
        raise GenerationError("family must be exactly 'LF' or 'HF'")
    return value  # type: ignore[return-value]


def _path_entity(value: object, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise GenerationError(
            "path entity is outside the one-based registered range",
            details={"minimum": 1, "maximum": maximum},
        )
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationError(f"{label} must be a positive integer")
    return value


def _uniform_array(values: npt.ArrayLike, size: int, label: str) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GenerationError(f"{label} must be numeric") from error
    if (
        result.shape != (size,)
        or not bool(np.isfinite(result).all())
        or bool(((result < 0.0) | (result >= 1.0)).any())
    ):
        raise GenerationError(
            f"{label} must contain one finite [0,1) draw per observation"
        )
    return result


def _immutable_array(
    values: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike,
    label: str,
) -> np.ndarray[Any, Any]:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise IntegrityError(f"{label} cannot be converted to an array") from error
    expected = np.dtype(dtype)
    if source.ndim == 0 or source.size == 0:
        raise IntegrityError(f"{label} must be nonempty and nonscalar")
    if source.dtype != expected:
        raise IntegrityError(
            f"{label} has an invalid dtype",
            details={"expected": expected.str, "actual": source.dtype.str},
        )
    contiguous = np.ascontiguousarray(source)
    if contiguous.dtype.kind == "f" and not bool(np.isfinite(contiguous).all()):
        raise IntegrityError(f"{label} contains a non-finite value")
    content = contiguous.tobytes(order="C")
    return np.frombuffer(content, dtype=expected).reshape(contiguous.shape)


def _has_bytes_owner(array: np.ndarray[Any, Any]) -> bool:
    owner: object = array
    visited: set[int] = set()
    while isinstance(owner, np.ndarray) and owner.base is not None:
        if id(owner) in visited:  # pragma: no cover - NumPy bases are acyclic
            return False
        visited.add(id(owner))
        owner = owner.base
    return isinstance(owner, bytes)


def _array_fingerprint(array: np.ndarray[Any, Any], role: str) -> str:
    content = np.ascontiguousarray(array).tobytes(order="C")
    identity = {
        "schema": "prpg-serial-array-v1",
        "role": role,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return hashlib.sha256((_canonical_json(identity) + "\n").encode()).hexdigest()


def _record_fingerprint(
    metadata: Mapping[str, object], array_fingerprints: Mapping[str, str]
) -> str:
    identity = {
        "metadata": dict(metadata),
        "array_fingerprints": dict(sorted(array_fingerprints.items())),
    }
    return hashlib.sha256((_canonical_json(identity) + "\n").encode()).hexdigest()


def _npy_sha256(array: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(_npy_bytes(array)).hexdigest()


def _npy_bytes(array: np.ndarray[Any, Any]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_thaw(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise IntegrityError(
            "serial generation identity is not canonical JSON"
        ) from error


def _json_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_thaw(item) for item in value]
    return value


def _require_equal(actual: str, expected: str, message: str) -> None:
    if actual != expected:
        raise IntegrityError(message)


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} fingerprint is invalid")
    return value
