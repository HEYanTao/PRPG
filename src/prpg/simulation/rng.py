"""Version-1 deterministic random-number stream contract.

Every scientific stream is reconstructed directly from a stable key. No
module-global generator or scheduling-dependent ``SeedSequence.spawn`` call is
used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

UINT32_MAX = (1 << 32) - 1
UINT128_MAX = (1 << 128) - 1
RNG_CONTRACT_VERSION = 1


class Domain(IntEnum):
    MODEL_FIT = 1
    BLOCK_ALPHA_CALIBRATION = 2
    THRESHOLD_CALIBRATION = 3
    DEVELOPMENT = 4
    QUALIFICATION = 5
    PRODUCTION = 6
    RELEASE_VALIDATION = 7


class Family(IntEnum):
    NOT_APPLICABLE = 0
    LF = 1
    HF = 2


class Stage(IntEnum):
    HMM_LIBRARY_SEED = 1
    STATE_CHAIN = 2
    RESTART_UNIFORMS = 3
    SOURCE_RANK_UNIFORMS = 4
    OPTIONAL_RESIDUAL_NOISE = 5
    REFERENCE_RESAMPLING = 6
    VALIDATION_SAMPLING_OR_STRATEGY = 7


class ModelFitScope(IntEnum):
    """Closed four-bit model-fit scope registry from Sections 4.5/4.6."""

    DESIGN_MAIN = 0
    DESIGN_MACRO_STABILITY = 1
    DESIGN_ROLLING_FOLD = 2
    DESIGN_PARAMETRIC_ADEQUACY = 3
    PRODUCTION_MAIN_AND_FOLDS = 4
    PRODUCTION_MACRO_STABILITY = 5
    PRODUCTION_PARAMETRIC_ADEQUACY = 6
    BRIDGE_MODEL_SELECTION_PREFIX = 7
    BRIDGE_MODEL_SELECTION_FULL = 8
    BRIDGE_NONPARAMETRIC_SELECTION_PREFIX = 9
    BRIDGE_NONPARAMETRIC_SELECTION_FULL = 10
    BRIDGE_MODEL_FIXED_K_PREFIX = 11
    BRIDGE_MODEL_FIXED_K_FULL = 12
    BRIDGE_NONPARAMETRIC_FIXED_K_PREFIX = 13
    BRIDGE_NONPARAMETRIC_FIXED_K_FULL = 14
    HMM_IDENTIFICATION_META = 15


class CalibrationPurpose(IntEnum):
    """Closed four-bit non-model purpose registry from Section 4.6 Point 10."""

    MACRO_BOOTSTRAP = 0
    ROLLING_SCORE_BOOTSTRAP = 1
    LATENT_PPC = 2
    PARAMETRIC_ADEQUACY = 3
    CONDITIONED_BLOCK = 4
    IDEAL_BLOCK = 5
    DEPENDENCE_AUDIT = 6
    KERNEL_ESTIMATION = 7
    KERNEL_VERIFICATION = 8
    BRIDGE_HISTORY = 9
    G5_OUTER_NULL = 10
    G5_QUALIFICATION = 11
    META_TYPE_I = 12
    META_POWER = 13
    SOBOL_OR_STRATEGY = 14
    RESERVED = 15


class CalibrationArtifact(IntEnum):
    """Two-bit purpose-scoped artifact/context field."""

    DESIGN_OR_CONTROL = 0
    PRODUCTION_OR_CANDIDATE = 1
    BRIDGE_MODEL = 2
    BRIDGE_NONPARAMETRIC = 3


@dataclass(frozen=True, slots=True)
class RNGKey:
    """Complete deterministic identity of one pseudorandom stream."""

    master_seed: int
    scientific_version: int
    domain: int | Domain
    family: int | Family
    entity: int
    stage: int | Stage
    contract: int = RNG_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_uint(self.master_seed, 128, "master_seed")
        for name in (
            "contract",
            "scientific_version",
            "domain",
            "family",
            "entity",
            "stage",
        ):
            _require_uint(getattr(self, name), 32, name)
        _require_registered(self.domain, Domain, "domain")
        _require_registered(self.family, Family, "family")
        _require_registered(self.stage, Stage, "stage")
        if self.contract != RNG_CONTRACT_VERSION:
            raise ValueError(f"unsupported RNG contract version: {self.contract}")

    @property
    def spawn_key(self) -> tuple[int, int, int, int, int, int]:
        """Return the exact version-1 six-component SeedSequence spawn key."""

        return (
            int(self.contract),
            int(self.scientific_version),
            int(self.domain),
            int(self.family),
            int(self.entity),
            int(self.stage),
        )

    def seed_sequence(self) -> np.random.SeedSequence:
        """Construct a fresh NumPy SeedSequence for this key."""

        return np.random.SeedSequence(
            entropy=self.master_seed,
            spawn_key=self.spawn_key,
        )

    def generator(self) -> np.random.Generator:
        """Construct a fresh PCG64DXSM generator for this key."""

        return np.random.Generator(np.random.PCG64DXSM(self.seed_sequence()))


def seed_sequence(
    *,
    master_seed: int,
    scientific_version: int,
    domain: int | Domain,
    family: int | Family,
    entity: int,
    stage: int | Stage,
    contract: int = RNG_CONTRACT_VERSION,
) -> np.random.SeedSequence:
    """Convenience constructor for the exact version-1 SeedSequence."""

    return RNGKey(
        master_seed=master_seed,
        scientific_version=scientific_version,
        domain=domain,
        family=family,
        entity=entity,
        stage=stage,
        contract=contract,
    ).seed_sequence()


def generator(
    *,
    master_seed: int,
    scientific_version: int,
    domain: int | Domain,
    family: int | Family,
    entity: int,
    stage: int | Stage,
    contract: int = RNG_CONTRACT_VERSION,
) -> np.random.Generator:
    """Convenience constructor for a fresh keyed PCG64DXSM generator."""

    return RNGKey(
        master_seed=master_seed,
        scientific_version=scientific_version,
        domain=domain,
        family=family,
        entity=entity,
        stage=stage,
        contract=contract,
    ).generator()


def model_fit_entity(
    scope: int | ModelFitScope,
    replicate: int,
    k: int,
    restart_index: int,
) -> int:
    """Pack the approved model-fit identity into one unsigned 32-bit entity.

    The layout is ``scope:4 | replicate:16 | K:4 | restart:8``.  In addition
    to field-width checks, this function enforces every scope-specific
    replicate range so an apparently valid bit pattern cannot silently enter
    an unregistered scientific namespace.
    """

    _require_registered(scope, ModelFitScope, "model-fit scope")
    _require_uint(replicate, 16, "model-fit replicate")
    _require_uint(k, 4, "K")
    _require_uint(restart_index, 8, "restart_index")
    checked_scope = ModelFitScope(int(scope))
    if k not in {2, 3, 4, 5}:
        raise ValueError("K must be one of 2, 3, 4, or 5")
    if restart_index >= 50:
        raise ValueError("restart_index must be in [0, 49]")
    maximum_replicate = _model_scope_maximum_replicate(checked_scope)
    if replicate > maximum_replicate:
        raise ValueError(
            f"replicate is outside the registered range for scope "
            f"{int(checked_scope)}: [0, {maximum_replicate}]"
        )
    if checked_scope is ModelFitScope.HMM_IDENTIFICATION_META and k != 3:
        raise ValueError(
            "HMM-identification meta-validation is registered only for K=3"
        )
    return (int(checked_scope) << 28) | (replicate << 12) | (k << 8) | restart_index


def hmm_library_seed(
    *,
    master_seed: int,
    scientific_version: int,
    scope: int | ModelFitScope,
    replicate: int,
    k: int,
    restart_index: int,
) -> int:
    """Return the exact uint32 seed for one registered HMM model start."""

    sequence = seed_sequence(
        master_seed=master_seed,
        scientific_version=scientific_version,
        domain=Domain.MODEL_FIT,
        family=Family.NOT_APPLICABLE,
        entity=model_fit_entity(scope, replicate, k, restart_index),
        stage=Stage.HMM_LIBRARY_SEED,
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def calibration_entity(
    purpose: int | CalibrationPurpose,
    artifact: int | CalibrationArtifact,
    variant: int,
    replicate: int,
) -> int:
    """Pack the approved non-model identity into one uint32 entity.

    The layout is ``purpose:4 | artifact:2 | variant:6 | replicate:20``.
    Variant allocation is closed even for deterministic, streamless semantic
    identifiers; :func:`calibration_rng_key` separately rejects every
    no-stream purpose/variant.
    """

    _require_registered(purpose, CalibrationPurpose, "calibration purpose")
    _require_registered(artifact, CalibrationArtifact, "calibration artifact")
    _require_uint(variant, 6, "calibration variant")
    _require_uint(replicate, 20, "calibration replicate")
    checked_purpose = CalibrationPurpose(int(purpose))
    _validate_purpose_variant(checked_purpose, variant)
    return (
        (int(checked_purpose) << 28)
        | (int(artifact) << 26)
        | (variant << 20)
        | replicate
    )


def calibration_rng_key(
    *,
    master_seed: int,
    scientific_version: int,
    purpose: int | CalibrationPurpose,
    artifact: int | CalibrationArtifact,
    variant: int,
    replicate: int,
    domain: int | Domain,
    family: int | Family,
    stage: int | Stage,
) -> RNGKey:
    """Construct a key only for a stream combination listed in Section 4.6.

    This is the authoritative closed-registry validator.  It prevents a new
    stream from being invented merely because its bit fields fit inside the
    generic :class:`RNGKey` representation.
    """

    _require_registered(purpose, CalibrationPurpose, "calibration purpose")
    _require_registered(artifact, CalibrationArtifact, "calibration artifact")
    _require_registered(domain, Domain, "domain")
    _require_registered(family, Family, "family")
    _require_registered(stage, Stage, "stage")
    _require_uint(variant, 6, "calibration variant")
    _require_uint(replicate, 20, "calibration replicate")
    checked_purpose = CalibrationPurpose(int(purpose))
    checked_artifact = CalibrationArtifact(int(artifact))
    checked_domain = Domain(int(domain))
    checked_family = Family(int(family))
    checked_stage = Stage(int(stage))
    _validate_calibration_stream(
        checked_purpose,
        checked_artifact,
        variant,
        replicate,
        checked_domain,
        checked_family,
        checked_stage,
    )
    return RNGKey(
        master_seed=master_seed,
        scientific_version=scientific_version,
        domain=checked_domain,
        family=checked_family,
        entity=calibration_entity(
            checked_purpose, checked_artifact, variant, replicate
        ),
        stage=checked_stage,
    )


def sobol_direction_seed(*, master_seed: int, scientific_version: int) -> int:
    """Derive the exact first-uint32 seed for the registered Sobol engine."""

    key = calibration_rng_key(
        master_seed=master_seed,
        scientific_version=scientific_version,
        purpose=CalibrationPurpose.SOBOL_OR_STRATEGY,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        variant=0,
        replicate=0,
        domain=Domain.THRESHOLD_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.VALIDATION_SAMPLING_OR_STRATEGY,
    )
    return int(key.seed_sequence().generate_state(1, dtype=np.uint32)[0])


def hmm_emission_standard_normals(
    key: RNGKey, n_macro_rows: int
) -> NDArray[np.float64]:
    """Consume exactly four standard normals per macro row in feature order."""

    if int(key.family) != int(Family.NOT_APPLICABLE) or int(key.stage) != int(
        Stage.OPTIONAL_RESIDUAL_NOISE
    ):
        raise ValueError("HMM emission normals require family 0 and stage 5")
    if isinstance(n_macro_rows, bool) or not isinstance(n_macro_rows, int):
        raise TypeError("n_macro_rows must be an integer")
    if n_macro_rows < 0:
        raise ValueError("n_macro_rows must be non-negative")
    values = key.generator().standard_normal(n_macro_rows * 4)
    return np.asarray(values, dtype=np.float64).reshape(n_macro_rows, 4)


def uniforms(key: RNGKey, count: int) -> NDArray[np.float64]:
    """Pre-draw an exact non-negative number of stream uniforms."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    return key.generator().random(count)


def categorical_from_uniform(probabilities: Sequence[float], uniform: float) -> int:
    """Select a categorical index by cumulative search and last-bin fallback.

    ``uniform`` must already have been drawn from ``[0, 1)``. The explicit
    implementation prevents changes to a third-party ``choice`` algorithm from
    changing the scientific stream contract.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if (
        isinstance(uniform, bool)
        or not np.isfinite(uniform)
        or not 0.0 <= uniform < 1.0
    ):
        raise ValueError("uniform must be finite and in [0, 1)")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one within absolute 1e-12")
    cumulative = 0.0
    for index, probability in enumerate(values[:-1]):
        cumulative += float(probability)
        if uniform < cumulative:
            return index
    return int(values.size - 1)


def _model_scope_maximum_replicate(scope: ModelFitScope) -> int:
    if scope is ModelFitScope.DESIGN_MAIN:
        return 0
    if scope is ModelFitScope.DESIGN_MACRO_STABILITY:
        return 99
    if scope is ModelFitScope.DESIGN_ROLLING_FOLD:
        return 5
    if scope is ModelFitScope.DESIGN_PARAMETRIC_ADEQUACY:
        return 999
    if scope is ModelFitScope.PRODUCTION_MAIN_AND_FOLDS:
        return 6
    if scope is ModelFitScope.PRODUCTION_MACRO_STABILITY:
        return 99
    if scope is ModelFitScope.PRODUCTION_PARAMETRIC_ADEQUACY:
        return 999
    if scope in {
        ModelFitScope.BRIDGE_MODEL_SELECTION_PREFIX,
        ModelFitScope.BRIDGE_MODEL_SELECTION_FULL,
        ModelFitScope.BRIDGE_NONPARAMETRIC_SELECTION_PREFIX,
        ModelFitScope.BRIDGE_NONPARAMETRIC_SELECTION_FULL,
    }:
        return 7 * 250 - 1
    if scope in {
        ModelFitScope.BRIDGE_MODEL_FIXED_K_PREFIX,
        ModelFitScope.BRIDGE_MODEL_FIXED_K_FULL,
        ModelFitScope.BRIDGE_NONPARAMETRIC_FIXED_K_PREFIX,
        ModelFitScope.BRIDGE_NONPARAMETRIC_FIXED_K_FULL,
    }:
        return 9_999
    if scope is ModelFitScope.HMM_IDENTIFICATION_META:
        return 999
    raise AssertionError(f"unhandled model-fit scope: {scope}")


def _validate_purpose_variant(purpose: CalibrationPurpose, variant: int) -> None:
    allowed: range | tuple[int, ...]
    if purpose in {
        CalibrationPurpose.MACRO_BOOTSTRAP,
        CalibrationPurpose.ROLLING_SCORE_BOOTSTRAP,
        CalibrationPurpose.LATENT_PPC,
        CalibrationPurpose.PARAMETRIC_ADEQUACY,
        CalibrationPurpose.CONDITIONED_BLOCK,
        CalibrationPurpose.IDEAL_BLOCK,
        CalibrationPurpose.DEPENDENCE_AUDIT,
        CalibrationPurpose.KERNEL_ESTIMATION,
        CalibrationPurpose.KERNEL_VERIFICATION,
        CalibrationPurpose.G5_OUTER_NULL,
        CalibrationPurpose.G5_QUALIFICATION,
        CalibrationPurpose.META_TYPE_I,
    }:
        allowed = (0,)
    elif purpose is CalibrationPurpose.BRIDGE_HISTORY:
        allowed = (0, 1)
    elif purpose is CalibrationPurpose.META_POWER:
        allowed = range(10)
    elif purpose is CalibrationPurpose.SOBOL_OR_STRATEGY:
        allowed = range(25)
    else:
        allowed = ()
    if variant not in allowed:
        raise ValueError(
            f"variant {variant} is not registered for purpose {int(purpose)}"
        )


def _validate_calibration_stream(
    purpose: CalibrationPurpose,
    artifact: CalibrationArtifact,
    variant: int,
    replicate: int,
    domain: Domain,
    family: Family,
    stage: Stage,
) -> None:
    _validate_purpose_variant(purpose, variant)

    def require(
        condition: bool,
        *,
        maximum_replicate: int,
        message: str,
    ) -> None:
        if not condition:
            raise ValueError(message)
        if replicate > maximum_replicate:
            raise ValueError(
                f"replicate must be in [0, {maximum_replicate}] for the "
                f"registered purpose/variant"
            )

    design_or_production = artifact in {
        CalibrationArtifact.DESIGN_OR_CONTROL,
        CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    }
    if purpose is CalibrationPurpose.MACRO_BOOTSTRAP:
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family is Family.NOT_APPLICABLE
            and design_or_production
            and stage is Stage.REFERENCE_RESAMPLING,
            maximum_replicate=99,
            message="unregistered macro-bootstrap stream combination",
        )
        return
    if purpose is CalibrationPurpose.ROLLING_SCORE_BOOTSTRAP:
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family is Family.NOT_APPLICABLE
            and stage is Stage.REFERENCE_RESAMPLING,
            maximum_replicate=9_999,
            message="unregistered rolling-score stream combination",
        )
        return
    if purpose is CalibrationPurpose.LATENT_PPC:
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family is Family.NOT_APPLICABLE
            and design_or_production
            and stage in {Stage.STATE_CHAIN, Stage.REFERENCE_RESAMPLING},
            maximum_replicate=9_999,
            message="unregistered latent-PPC stream combination",
        )
        return
    if purpose is CalibrationPurpose.PARAMETRIC_ADEQUACY:
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family is Family.NOT_APPLICABLE
            and design_or_production
            and stage in {Stage.STATE_CHAIN, Stage.OPTIONAL_RESIDUAL_NOISE},
            maximum_replicate=999,
            message="unregistered adequacy stream combination",
        )
        return
    if purpose is CalibrationPurpose.CONDITIONED_BLOCK:
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family in {Family.LF, Family.HF}
            and design_or_production
            and stage
            in {
                Stage.STATE_CHAIN,
                Stage.RESTART_UNIFORMS,
                Stage.SOURCE_RANK_UNIFORMS,
            },
            maximum_replicate=9_999,
            message="unregistered conditioned-block stream combination",
        )
        return
    if purpose in {
        CalibrationPurpose.IDEAL_BLOCK,
        CalibrationPurpose.DEPENDENCE_AUDIT,
        CalibrationPurpose.KERNEL_VERIFICATION,
        CalibrationPurpose.G5_QUALIFICATION,
    }:
        raise ValueError(f"purpose {int(purpose)} is streamless")
    if purpose is CalibrationPurpose.KERNEL_ESTIMATION:
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family in {Family.LF, Family.HF}
            and design_or_production
            and stage is Stage.REFERENCE_RESAMPLING,
            maximum_replicate=319_999,
            message="unregistered kernel-estimation stream combination",
        )
        return
    if purpose is CalibrationPurpose.BRIDGE_HISTORY:
        is_model = artifact is CalibrationArtifact.BRIDGE_MODEL
        is_nonparametric = artifact is CalibrationArtifact.BRIDGE_NONPARAMETRIC
        stage_valid = (
            is_model and stage in {Stage.STATE_CHAIN, Stage.OPTIONAL_RESIDUAL_NOISE}
        ) or (is_nonparametric and stage is Stage.REFERENCE_RESAMPLING)
        maximum = 249 if variant == 0 else 9_999
        require(
            domain is Domain.BLOCK_ALPHA_CALIBRATION
            and family is Family.NOT_APPLICABLE
            and (is_model or is_nonparametric)
            and stage_valid,
            maximum_replicate=maximum,
            message="unregistered bridge-history stream combination",
        )
        return
    if purpose is CalibrationPurpose.G5_OUTER_NULL:
        require(
            domain is Domain.THRESHOLD_CALIBRATION
            and family in {Family.LF, Family.HF}
            and design_or_production
            and stage is Stage.REFERENCE_RESAMPLING,
            maximum_replicate=49_999,
            message="unregistered G5 outer-null stream combination",
        )
        return
    if purpose is CalibrationPurpose.META_TYPE_I:
        is_hmm = family is Family.NOT_APPLICABLE and stage in {
            Stage.STATE_CHAIN,
            Stage.OPTIONAL_RESIDUAL_NOISE,
        }
        is_g5 = family in {Family.LF, Family.HF} and stage in {
            Stage.STATE_CHAIN,
            Stage.RESTART_UNIFORMS,
            Stage.SOURCE_RANK_UNIFORMS,
        }
        require(
            domain is Domain.THRESHOLD_CALIBRATION
            and artifact is CalibrationArtifact.DESIGN_OR_CONTROL
            and (is_hmm or is_g5),
            maximum_replicate=999,
            message="unregistered meta type-I stream combination",
        )
        return
    if purpose is CalibrationPurpose.META_POWER:
        if variant != 9:
            raise ValueError("meta-power variants 0..8 are streamless transforms")
        require(
            domain is Domain.THRESHOLD_CALIBRATION
            and family in {Family.LF, Family.HF}
            and artifact is CalibrationArtifact.PRODUCTION_OR_CANDIDATE
            and stage is Stage.VALIDATION_SAMPLING_OR_STRATEGY,
            maximum_replicate=999,
            message="unregistered subsequence-power stream combination",
        )
        return
    if purpose is CalibrationPurpose.SOBOL_OR_STRATEGY:
        if variant == 0:
            require(
                domain is Domain.THRESHOLD_CALIBRATION
                and family is Family.NOT_APPLICABLE
                and artifact is CalibrationArtifact.DESIGN_OR_CONTROL
                and stage is Stage.VALIDATION_SAMPLING_OR_STRATEGY
                and replicate == 0,
                maximum_replicate=0,
                message="unregistered Sobol-direction stream combination",
            )
            return
        if variant == 1:
            maximum = 499 if family is Family.LF else 199
            require(
                domain is Domain.QUALIFICATION
                and family in {Family.LF, Family.HF}
                and design_or_production
                and stage is Stage.VALIDATION_SAMPLING_OR_STRATEGY,
                maximum_replicate=maximum,
                message="unregistered evaluation-window stream combination",
            )
            return
        raise ValueError("strategy variants 2..24 are streamless identifiers")
    raise ValueError(f"purpose {int(purpose)} has no permitted stream")


def _require_uint(value: int | np.integer, bits: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an unsigned integer")
    maximum = (1 << bits) - 1
    if not 0 <= int(value) <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}]")


def _require_registered(value: int, registry: type[IntEnum], name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be a registered integer")
    try:
        registry(int(value))
    except ValueError as error:
        raise ValueError(f"unregistered {name}: {value}") from error
