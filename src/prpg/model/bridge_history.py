"""Registered nested pseudo-history generation for the Phase-3 bridge.

This module owns only the two frozen bridge reference laws.  It deliberately
does not fit an HMM or make a Tier-A/Tier-B decision.  Every returned full
history contains 217 macro rows with segment lengths ``(210, 7)``; its first
193 rows are the exact nested design prefix.  The production gap therefore
resets both the model state chain and the nonparametric source trace.

The model DGP consumes the registered purpose-9 state-chain and emission-noise
streams.  The nonparametric DGP consumes one registered purpose-9 reference-
resampling stream and always materializes both restart and source-rank draws,
regardless of which draws are operationally used.  No module-global RNG or
schedule-dependent spawning is permitted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ModelError
from prpg.model.hmm import HMMParameters
from prpg.model.refit_adequacy import simulate_artifact_macro_history
from prpg.simulation.rng import (
    RNG_CONTRACT_VERSION,
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    RNGKey,
    Stage,
    calibration_entity,
    calibration_rng_key,
    hmm_emission_standard_normals,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
BridgeDGP: TypeAlias = Literal["model", "nonparametric"]
BridgeTier: TypeAlias = Literal["selection", "fixed_k"]

BRIDGE_PREFIX_ROWS = 193
BRIDGE_FULL_LENGTHS = (210, 7)
BRIDGE_FULL_ROWS = sum(BRIDGE_FULL_LENGTHS)
BRIDGE_GAP_ROW = BRIDGE_FULL_LENGTHS[0]
BRIDGE_SELECTION_PAIRS = 250
BRIDGE_FIXED_K_MAXIMUM_PAIRS = 10_000


@dataclass(frozen=True, slots=True)
class BridgeStreamAudit:
    """Exact registered stream coordinates and fixed draw accounting."""

    artifact: CalibrationArtifact
    variant: int
    replicate: int
    master_seed: int
    scientific_version: int
    domain: Domain
    family: Family
    contract: int
    stages: tuple[Stage, ...]
    draws_by_stage: tuple[int, ...]
    entities_by_stage: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BridgePseudoHistory:
    """One nested full/prefix bridge history with immutable provenance."""

    dgp: BridgeDGP
    tier: BridgeTier
    pair: int
    design_n_states: int
    design_parameters_sha256: str
    design_history_sha256: str
    values: FloatArray
    continuation: BoolArray
    latent_states: IntArray | None
    source_indices: IntArray | None
    macro_block_length: int | None
    stream_audit: BridgeStreamAudit

    @property
    def prefix_values(self) -> FloatArray:
        """Return the immutable 193-row nested design prefix."""

        return self.values[:BRIDGE_PREFIX_ROWS]

    @property
    def prefix_continuation(self) -> BoolArray:
        """Return the immutable one-segment prefix continuation mask."""

        return self.continuation[:BRIDGE_PREFIX_ROWS]

    @property
    def full_lengths(self) -> tuple[int, int]:
        return BRIDGE_FULL_LENGTHS


def bridge_continuation_mask() -> BoolArray:
    """Return the exact false-at-start row mask for ``(210, 7)``."""

    continuation = np.ones(BRIDGE_FULL_ROWS, dtype=np.bool_)
    continuation[0] = False
    continuation[BRIDGE_GAP_ROW] = False
    continuation.setflags(write=False)
    return continuation


def validate_bridge_pseudo_history(
    history: BridgePseudoHistory,
    *,
    expected_tier: BridgeTier | None = None,
    expected_n_states: int | None = None,
    expected_master_seed: int | None = None,
    expected_scientific_version: int | None = None,
) -> None:
    """Validate exact geometry, closed-registry provenance, and immutability.

    ``BridgePseudoHistory`` is intentionally a plain dataclass so evidence can
    later be reconstructed from durable artifacts.  Consumers must therefore
    validate rather than trust a caller-constructed record.  This check is
    deterministic and instantiates no RNG stream.
    """

    if not isinstance(history, BridgePseudoHistory):
        raise ModelError("bridge history has the wrong record type")
    artifact = _artifact(history.dgp)
    variant, maximum = _variant_and_maximum(history.tier)
    pair = _pair(history.pair, maximum)
    if expected_tier is not None and history.tier != expected_tier:
        raise ModelError(f"bridge history must use the {expected_tier} tier")
    design_n_states = _expected_n_states(history.design_n_states)
    if not _is_sha256(history.design_parameters_sha256) or not _is_sha256(
        history.design_history_sha256
    ):
        raise ModelError("bridge history design-reference fingerprints are invalid")
    if expected_n_states is not None and (
        _expected_n_states(expected_n_states) != design_n_states
    ):
        raise ModelError(
            "bridge history selected K does not match its design reference"
        )

    values = np.asarray(history.values)
    continuation = np.asarray(history.continuation)
    if (
        values.shape != (BRIDGE_FULL_ROWS, 4)
        or values.dtype != np.dtype(np.float64)
        or not np.isfinite(values).all()
        or values.flags.writeable
    ):
        raise ModelError(
            "bridge history values must be an immutable finite float64 (217, 4) matrix"
        )
    expected_continuation = bridge_continuation_mask()
    if (
        continuation.dtype.kind != "b"
        or continuation.shape != expected_continuation.shape
        or not np.array_equal(continuation, expected_continuation)
        or continuation.flags.writeable
    ):
        raise ModelError(
            "bridge history continuation must have exact (210, 7) geometry"
        )

    audit = history.stream_audit
    if not isinstance(audit, BridgeStreamAudit):
        raise ModelError("bridge history stream audit has the wrong record type")
    if (
        audit.artifact is not artifact
        or audit.variant != variant
        or audit.replicate != pair
        or audit.domain is not Domain.BLOCK_ALPHA_CALIBRATION
        or audit.family is not Family.NOT_APPLICABLE
        or audit.contract != RNG_CONTRACT_VERSION
    ):
        raise ModelError("bridge history stream audit identity is inconsistent")
    if expected_master_seed is not None and audit.master_seed != expected_master_seed:
        raise ModelError("bridge history master seed does not match execution")
    if (
        expected_scientific_version is not None
        and audit.scientific_version != expected_scientific_version
    ):
        raise ModelError("bridge history scientific version does not match execution")
    entity = calibration_entity(
        CalibrationPurpose.BRIDGE_HISTORY,
        artifact,
        variant,
        pair,
    )
    expected_stages: tuple[Stage, ...]
    expected_draws: tuple[int, ...]
    if history.dgp == "model":
        expected_stages = (Stage.STATE_CHAIN, Stage.OPTIONAL_RESIDUAL_NOISE)
        expected_draws = (BRIDGE_FULL_ROWS, BRIDGE_FULL_ROWS * 4)
        latent = np.asarray(history.latent_states)
        if (
            latent.shape != (BRIDGE_FULL_ROWS,)
            or latent.dtype != np.dtype(np.int64)
            or bool((latent < 0).any())
            or latent.flags.writeable
            or history.source_indices is not None
            or history.macro_block_length is not None
        ):
            raise ModelError("model-DGP bridge history payload is inconsistent")
        if bool((latent >= design_n_states).any()):
            raise ModelError("model-DGP latent state is outside selected K")
    else:
        expected_stages = (Stage.REFERENCE_RESAMPLING,)
        expected_draws = (2 * BRIDGE_FULL_ROWS,)
        source = np.asarray(history.source_indices)
        if (
            history.latent_states is not None
            or source.shape != (BRIDGE_FULL_ROWS,)
            or source.dtype != np.dtype(np.int64)
            or bool(((source < 0) | (source >= BRIDGE_PREFIX_ROWS)).any())
            or source.flags.writeable
            or history.macro_block_length is None
        ):
            raise ModelError("nonparametric-DGP bridge history payload is inconsistent")
        _macro_block_length(history.macro_block_length, BRIDGE_PREFIX_ROWS)
    if (
        audit.stages != expected_stages
        or audit.draws_by_stage != expected_draws
        or audit.entities_by_stage != (entity,) * len(expected_stages)
    ):
        raise ModelError("bridge history stream accounting is inconsistent")


def bridge_fit_input_sha256(history: BridgePseudoHistory) -> str:
    """Hash the exact immutable fit inputs for accelerator pairing audits."""

    validate_bridge_pseudo_history(history)
    values = np.ascontiguousarray(np.asarray(history.values, dtype="<f8"))
    continuation = np.ascontiguousarray(
        np.asarray(history.continuation, dtype=np.uint8)
    )
    digest = hashlib.sha256()
    digest.update(b"PRPG-BRIDGE-FIT-INPUT-v1\0")
    digest.update(values.tobytes(order="C"))
    digest.update(continuation.tobytes(order="C"))
    return digest.hexdigest()


def bridge_parameters_sha256(parameters: HMMParameters) -> str:
    """Return a canonical exact-byte fingerprint of design HMM parameters."""

    start = np.asarray(parameters.start_probabilities, dtype=np.float64)
    transition = np.asarray(parameters.transition_matrix, dtype=np.float64)
    means = np.asarray(parameters.means, dtype=np.float64)
    covariance = np.asarray(parameters.tied_covariance, dtype=np.float64)
    if (
        start.ndim != 1
        or len(start) not in {2, 3, 4, 5}
        or transition.shape != (len(start), len(start))
        or means.shape != (len(start), 4)
        or covariance.shape != (4, 4)
        or not all(
            np.isfinite(array).all() for array in (start, transition, means, covariance)
        )
    ):
        raise ModelError("bridge design HMM parameters have inconsistent dimensions")
    digest = hashlib.sha256()
    digest.update(b"PRPG-BRIDGE-DESIGN-HMM-v1\0")
    for name, array in (
        (b"start", start),
        (b"transition", transition),
        (b"means", means),
        (b"tied_covariance", covariance),
    ):
        canonical = _canonical_float64_bytes(array)
        digest.update(name + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(canonical)
    return digest.hexdigest()


def nonparametric_bridge_indices_from_uniforms(
    source_continuation: ArrayLike,
    target_continuation: ArrayLike,
    restart_uniforms: ArrayLike,
    source_rank_uniforms: ArrayLike,
    *,
    block_length: int,
) -> IntArray:
    """Build one synchronous non-circular source trace of arbitrary length.

    ``False`` target links force a new source block.  A source block also ends
    at the source boundary or a source gap.  A continuation is always the
    literal next source row; wrapping from the final row to row zero is never
    allowed.
    """

    source_mask = _continuation(source_continuation, "source continuation")
    target_mask = _continuation(target_continuation, "target continuation")
    length = _macro_block_length(block_length, len(source_mask))
    rows = len(target_mask)
    restarts = _uniforms(restart_uniforms, rows, "bridge restart uniforms")
    ranks = _uniforms(source_rank_uniforms, rows, "bridge source-rank uniforms")
    restart_probability = 1.0 / length
    indices = np.empty(rows, dtype=np.int64)
    source_rows = len(source_mask)
    for row in range(rows):
        forced = (
            row == 0
            or not bool(target_mask[row])
            or int(indices[row - 1]) + 1 >= source_rows
            or not bool(source_mask[int(indices[row - 1]) + 1])
        )
        if forced or restarts[row] < restart_probability:
            indices[row] = min(int(ranks[row] * source_rows), source_rows - 1)
        else:
            indices[row] = indices[row - 1] + 1
    indices.setflags(write=False)
    return indices


def generate_registered_bridge_history(
    *,
    dgp: BridgeDGP,
    tier: BridgeTier,
    pair: int,
    master_seed: int,
    scientific_version: int,
    design_parameters: HMMParameters,
    design_standardized_history: ArrayLike,
    design_continuation: ArrayLike,
    macro_block_length: int,
) -> BridgePseudoHistory:
    """Generate one exact registered nested bridge pseudo-history."""

    artifact = _artifact(dgp)
    variant, maximum_pair = _variant_and_maximum(tier)
    checked_pair = _pair(pair, maximum_pair)
    design_values = _design_values(design_standardized_history)
    design_mask = _continuation(design_continuation, "design continuation")
    if design_mask.shape != (BRIDGE_PREFIX_ROWS,):
        raise ModelError("bridge design continuation must contain exactly 193 rows")
    if not bool(design_mask[1:].all()):
        raise ModelError(
            "bridge design history must be the exact gap-free (193) prefix"
        )
    length = _macro_block_length(macro_block_length, BRIDGE_PREFIX_ROWS)
    target_mask = bridge_continuation_mask()
    design_parameters_sha256 = bridge_parameters_sha256(design_parameters)
    design_n_states = int(np.asarray(design_parameters.means).shape[0])
    design_history_sha256 = _design_history_sha256(design_values, design_mask)

    if dgp == "model":
        state_key = _bridge_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            artifact=artifact,
            variant=variant,
            pair=checked_pair,
            stage=Stage.STATE_CHAIN,
        )
        emission_key = _bridge_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            artifact=artifact,
            variant=variant,
            pair=checked_pair,
            stage=Stage.OPTIONAL_RESIDUAL_NOISE,
        )
        state_uniforms = state_key.generator().random(BRIDGE_FULL_ROWS)
        normals = hmm_emission_standard_normals(emission_key, BRIDGE_FULL_ROWS)
        simulation = simulate_artifact_macro_history(
            design_parameters,
            BRIDGE_FULL_LENGTHS,
            state_uniforms,
            normals,
        )
        values = np.asarray(simulation.values, dtype=np.float64).copy()
        latent = np.asarray(simulation.latent_states, dtype=np.int64).copy()
        values.setflags(write=False)
        latent.setflags(write=False)
        audit = _stream_audit(
            artifact,
            variant,
            checked_pair,
            (state_key, emission_key),
            (BRIDGE_FULL_ROWS, BRIDGE_FULL_ROWS * 4),
        )
        return BridgePseudoHistory(
            "model",
            tier,
            checked_pair,
            design_n_states,
            design_parameters_sha256,
            design_history_sha256,
            values,
            target_mask,
            latent,
            None,
            None,
            audit,
        )

    resampling_key = _bridge_key(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        variant=variant,
        pair=checked_pair,
        stage=Stage.REFERENCE_RESAMPLING,
    )
    draws = resampling_key.generator().random(2 * BRIDGE_FULL_ROWS)
    indices = nonparametric_bridge_indices_from_uniforms(
        design_mask,
        target_mask,
        draws[:BRIDGE_FULL_ROWS],
        draws[BRIDGE_FULL_ROWS:],
        block_length=length,
    )
    values = design_values[indices].astype(np.float64, copy=True)
    values.setflags(write=False)
    audit = _stream_audit(
        artifact,
        variant,
        checked_pair,
        (resampling_key,),
        (2 * BRIDGE_FULL_ROWS,),
    )
    return BridgePseudoHistory(
        "nonparametric",
        tier,
        checked_pair,
        design_n_states,
        design_parameters_sha256,
        design_history_sha256,
        values,
        target_mask,
        None,
        indices,
        length,
        audit,
    )


def _bridge_key(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    variant: int,
    pair: int,
    stage: Stage,
) -> RNGKey:
    return calibration_rng_key(
        master_seed=master_seed,
        scientific_version=scientific_version,
        purpose=CalibrationPurpose.BRIDGE_HISTORY,
        artifact=artifact,
        variant=variant,
        replicate=pair,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=stage,
    )


def _stream_audit(
    artifact: CalibrationArtifact,
    variant: int,
    pair: int,
    keys: tuple[RNGKey, ...],
    draws: tuple[int, ...],
) -> BridgeStreamAudit:
    if not keys:  # pragma: no cover - fixed registered construction
        raise AssertionError("bridge stream audit requires at least one key")
    first = keys[0]
    if any(
        key.master_seed != first.master_seed
        or key.scientific_version != first.scientific_version
        or key.domain != first.domain
        or key.family != first.family
        or key.contract != first.contract
        for key in keys[1:]
    ):  # pragma: no cover - fixed registered construction
        raise AssertionError("bridge stream keys do not share one RNG namespace")
    return BridgeStreamAudit(
        artifact,
        variant,
        pair,
        first.master_seed,
        first.scientific_version,
        Domain(int(first.domain)),
        Family(int(first.family)),
        first.contract,
        tuple(Stage(int(key.stage)) for key in keys),
        draws,
        tuple(int(key.entity) for key in keys),
    )


def _artifact(dgp: BridgeDGP) -> CalibrationArtifact:
    if dgp == "model":
        return CalibrationArtifact.BRIDGE_MODEL
    if dgp == "nonparametric":
        return CalibrationArtifact.BRIDGE_NONPARAMETRIC
    raise ModelError("bridge DGP must be model or nonparametric")


def _variant_and_maximum(tier: BridgeTier) -> tuple[int, int]:
    if tier == "selection":
        return 0, BRIDGE_SELECTION_PAIRS
    if tier == "fixed_k":
        return 1, BRIDGE_FIXED_K_MAXIMUM_PAIRS
    raise ModelError("bridge tier must be selection or fixed_k")


def _pair(value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError("bridge pair must be an integer")
    if not 0 <= value < maximum:
        raise ModelError(
            "bridge pair is outside the registered tier range",
            details={"pair": value, "maximum_exclusive": maximum},
        )
    return value


def _design_values(values: ArrayLike) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (BRIDGE_PREFIX_ROWS, 4) or not np.isfinite(result).all():
        raise ModelError(
            "bridge design standardized history must have finite shape (193, 4)"
        )
    return result


def _continuation(values: ArrayLike, label: str) -> BoolArray:
    result = np.asarray(values)
    if (
        result.ndim != 1
        or result.size == 0
        or result.dtype.kind != "b"
        or bool(result[0])
    ):
        raise ModelError(f"{label} must be a false-at-start Boolean row mask")
    return result.astype(np.bool_, copy=False)


def _macro_block_length(value: int, source_rows: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError("bridge macro block length must be an integer")
    if not 2 <= value <= 24 or value > source_rows // 8:
        raise ModelError("bridge macro block length violates its binding hard caps")
    return value


def _expected_n_states(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value
        not in {
            2,
            3,
            4,
            5,
        }
    ):
        raise ModelError("bridge expected state count must be one of 2,3,4,5")
    return value


def _uniforms(values: ArrayLike, rows: int, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ModelError(f"{label} must contain exactly {rows} finite values")
    if bool(((result < 0.0) | (result >= 1.0)).any()):
        raise ModelError(f"{label} must lie in [0, 1)")
    return result


def _design_history_sha256(values: FloatArray, continuation: BoolArray) -> str:
    digest = hashlib.sha256()
    digest.update(b"PRPG-BRIDGE-DESIGN-HISTORY-v1\0")
    digest.update(_canonical_float64_bytes(values))
    digest.update(np.ascontiguousarray(continuation, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _canonical_float64_bytes(values: ArrayLike) -> bytes:
    result = np.asarray(values, dtype=np.float64).copy(order="C")
    result[result == 0.0] = 0.0
    return np.ascontiguousarray(result, dtype="<f8").tobytes(order="C")


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
