"""Registered-RNG orchestration for the two latent adequacy cells.

The statistical implementation accepts caller-supplied arrays or one generic
Generator for audit fixtures.  Canonical execution has a stricter requirement:
every chain and every whole-chain bootstrap replicate owns its separately
registered purpose-2 key.  This module materializes those common draws once,
then delegates the exact estimators to :mod:`prpg.model.adequacy`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypeGuard

import numpy as np

from prpg.errors import ModelError
from prpg.model.adequacy import (
    LATENT_BOOTSTRAP_REPLICATES,
    LATENT_BOOTSTRAP_RESAMPLE_SIZE,
    LATENT_CHAIN_COUNT,
    LATENT_EXTENSION_MONTHS,
    LATENT_MEASURED_MONTHS,
    LATENT_SURVIVAL_MONTHS,
    LatentChainSimulation,
    LatentClusterBootstrapEvidence,
    generate_latent_chains,
    latent_cluster_bootstrap,
)
from prpg.model.hmm import GaussianHMMFit
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    rng_execution_fingerprint,
    scientific_array_fingerprint,
    scientific_materialization_fingerprint,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    Stage,
    calibration_rng_key,
)

_REGISTERED_LATENT_CAPABILITY = object()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class RegisteredLatentCell:
    """Latent simulation/bootstrap evidence plus exact stream accounting."""

    artifact: CalibrationArtifact
    master_seed: int
    scientific_version: int
    parent_model_fingerprint: str
    transition_matrix_fingerprint: str
    simulation_rng_execution_fingerprint: str
    bootstrap_rng_execution_fingerprint: str
    rng_execution_fingerprint: str
    source_evidence_fingerprint: str
    simulation: LatentChainSimulation
    bootstrap: LatentClusterBootstrapEvidence
    chain_streams: int
    bootstrap_streams: int
    state_uniforms_per_chain: int
    resampled_ids_per_bootstrap: int
    _source_parent_fit: GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RegisteredLatentCellIdentity:
    """Reverified source, result, and split purpose-2 stream identities."""

    artifact: CalibrationArtifact
    master_seed: int
    scientific_version: int
    parent_model_fingerprint: str
    transition_matrix_fingerprint: str
    simulation_rng_execution_fingerprint: str
    bootstrap_rng_execution_fingerprint: str
    rng_execution_fingerprint: str
    source_evidence_fingerprint: str


_MINTED_REGISTERED_LATENT_CELLS: dict[int, RegisteredLatentCell] = {}


def registered_latent_cell_identity(
    value: RegisteredLatentCell,
) -> RegisteredLatentCellIdentity:
    """Revalidate one code-minted latent cell against its immutable content."""

    if (
        not isinstance(value, RegisteredLatentCell)
        or value._execution_seal is not _REGISTERED_LATENT_CAPABILITY
        or _MINTED_REGISTERED_LATENT_CELLS.get(id(value)) is not value
    ):
        raise ModelError("registered latent cell lacks executor authority")
    if _SHA256.fullmatch(value.parent_model_fingerprint) is None:
        raise ModelError("latent parent model fingerprint is invalid")
    if value.artifact not in {
        CalibrationArtifact.DESIGN_OR_CONTROL,
        CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    }:
        raise ModelError("registered latent cell artifact is invalid")
    source_fit = value._source_parent_fit
    if not isinstance(source_fit, GaussianHMMFit):
        raise ModelError("registered latent cell lacks its typed parent fit")
    expected_parent = gaussian_hmm_fit_fingerprint(source_fit)
    _verify_latent_result_geometry(value)
    expected_transition = scientific_array_fingerprint(
        source_fit.parameters.transition_matrix,
        role="latent_transition_matrix",
    )
    simulation_rng = _latent_simulation_rng_execution_fingerprint(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        artifact=value.artifact,
        chain_count=value.chain_streams,
        measured_months=value.simulation.measured_months,
        extension_months=value.simulation.extension_months,
    )
    bootstrap_rng = _latent_bootstrap_rng_execution_fingerprint(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        artifact=value.artifact,
        bootstrap_replicates=value.bootstrap_streams,
        chain_count=value.chain_streams,
        resample_size=value.resampled_ids_per_bootstrap,
        survival_months=value.bootstrap.survival_months,
    )
    combined_rng = _latent_rng_execution_fingerprint(
        master_seed=value.master_seed,
        scientific_version=value.scientific_version,
        simulation_rng_fingerprint=simulation_rng,
        bootstrap_rng_fingerprint=bootstrap_rng,
    )
    source_fingerprint = _latent_source_evidence_fingerprint(value)
    if (
        value.parent_model_fingerprint != expected_parent
        or value.transition_matrix_fingerprint != expected_transition
        or not np.array_equal(
            value.simulation.transition_matrix,
            source_fit.parameters.transition_matrix,
        )
        or value.simulation_rng_execution_fingerprint != simulation_rng
        or value.bootstrap_rng_execution_fingerprint != bootstrap_rng
        or value.rng_execution_fingerprint != combined_rng
        or value.source_evidence_fingerprint != source_fingerprint
    ):
        raise ModelError("registered latent cell identity changed")
    return RegisteredLatentCellIdentity(
        value.artifact,
        value.master_seed,
        value.scientific_version,
        expected_parent,
        expected_transition,
        simulation_rng,
        bootstrap_rng,
        combined_rng,
        source_fingerprint,
    )


def is_registered_latent_cell(value: object) -> TypeGuard[RegisteredLatentCell]:
    """Return whether the cell is the unchanged executor-minted instance."""

    try:
        registered_latent_cell_identity(value)  # type: ignore[arg-type]
    except (AttributeError, ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def registered_latent_cell_source_fit(
    value: RegisteredLatentCell,
) -> GaussianHMMFit:
    """Return the exact parent fit after full registered-cell reverification."""

    registered_latent_cell_identity(value)
    source_fit = value._source_parent_fit
    if not isinstance(source_fit, GaussianHMMFit):  # pragma: no cover - reverified.
        raise AssertionError("registered latent parent fit disappeared")
    return source_fit


def run_registered_latent_cell(
    original_fit: GaussianHMMFit,
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    chain_count: int = LATENT_CHAIN_COUNT,
    measured_months: int = LATENT_MEASURED_MONTHS,
    extension_months: int = LATENT_EXTENSION_MONTHS,
    bootstrap_replicates: int = LATENT_BOOTSTRAP_REPLICATES,
    resample_size: int = LATENT_BOOTSTRAP_RESAMPLE_SIZE,
    survival_months: int = LATENT_SURVIVAL_MONTHS,
    generation_chunk_size: int = 1_024,
    bootstrap_chunk_size: int = 32,
) -> RegisteredLatentCell:
    """Run one latent cell from an exact typed fit and registered key per replicate."""

    if not isinstance(original_fit, GaussianHMMFit):
        raise ModelError("latent adequacy requires a typed GaussianHMMFit parent")
    if artifact not in {
        CalibrationArtifact.DESIGN_OR_CONTROL,
        CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
    }:
        raise ModelError("latent adequacy artifact must be design or production")
    expected_scope = (
        "design_training"
        if artifact is CalibrationArtifact.DESIGN_OR_CONTROL
        else "production_full"
    )
    if original_fit.scaler_scope != expected_scope:
        raise ModelError("latent adequacy fit and artifact roles differ")
    parent_model_fingerprint = gaussian_hmm_fit_fingerprint(original_fit)
    transition_matrix = original_fit.parameters.transition_matrix
    for value, label in (
        (chain_count, "chain_count"),
        (measured_months, "measured_months"),
        (extension_months, "extension_months"),
        (bootstrap_replicates, "bootstrap_replicates"),
        (resample_size, "resample_size"),
    ):
        _positive(value, label)
    if chain_count > 10_000 or bootstrap_replicates > 10_000:
        raise ModelError("latent execution exceeds the registered replicate range")
    if resample_size != chain_count:
        raise ModelError("latent bootstrap resample size must equal chain count")
    draws_per_chain = 1 + measured_months + extension_months
    uniforms = np.empty((chain_count, draws_per_chain), dtype=np.float64)
    for replicate in range(chain_count):
        key = calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.LATENT_PPC,
            artifact=artifact,
            variant=0,
            replicate=replicate,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=Family.NOT_APPLICABLE,
            stage=Stage.STATE_CHAIN,
        )
        uniforms[replicate] = key.generator().random(draws_per_chain)
    simulation = generate_latent_chains(
        transition_matrix,
        chain_count=chain_count,
        measured_months=measured_months,
        extension_months=extension_months,
        uniforms=uniforms,
        chunk_size=generation_chunk_size,
    )
    del uniforms

    index_dtype: np.dtype[np.unsignedinteger]
    if chain_count <= np.iinfo(np.uint16).max:
        index_dtype = np.dtype(np.uint16)
    else:  # The registered maximum is 10,000; retained as a defensive branch.
        index_dtype = np.dtype(np.uint32)
    indices = np.empty(
        (bootstrap_replicates, resample_size),
        dtype=index_dtype,
    )
    for replicate in range(bootstrap_replicates):
        key = calibration_rng_key(
            master_seed=master_seed,
            scientific_version=scientific_version,
            purpose=CalibrationPurpose.LATENT_PPC,
            artifact=artifact,
            variant=0,
            replicate=replicate,
            domain=Domain.BLOCK_ALPHA_CALIBRATION,
            family=Family.NOT_APPLICABLE,
            stage=Stage.REFERENCE_RESAMPLING,
        )
        indices[replicate] = key.generator().integers(
            0,
            chain_count,
            size=resample_size,
            dtype=index_dtype,
        )
    bootstrap = latent_cluster_bootstrap(
        simulation,
        expected_chain_count=chain_count,
        bootstrap_replicates=bootstrap_replicates,
        resample_size=resample_size,
        survival_months=survival_months,
        resample_indices=indices,
        chunk_size=bootstrap_chunk_size,
    )
    transition_fingerprint = scientific_array_fingerprint(
        simulation.transition_matrix,
        role="latent_transition_matrix",
    )
    simulation_rng_fingerprint = _latent_simulation_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        chain_count=chain_count,
        measured_months=measured_months,
        extension_months=extension_months,
    )
    bootstrap_rng_fingerprint = _latent_bootstrap_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=artifact,
        bootstrap_replicates=bootstrap_replicates,
        chain_count=chain_count,
        resample_size=resample_size,
        survival_months=survival_months,
    )
    rng_fingerprint = _latent_rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        simulation_rng_fingerprint=simulation_rng_fingerprint,
        bootstrap_rng_fingerprint=bootstrap_rng_fingerprint,
    )
    result = RegisteredLatentCell(
        artifact,
        master_seed,
        scientific_version,
        parent_model_fingerprint,
        transition_fingerprint,
        simulation_rng_fingerprint,
        bootstrap_rng_fingerprint,
        rng_fingerprint,
        "",
        simulation,
        bootstrap,
        chain_count,
        bootstrap_replicates,
        draws_per_chain,
        resample_size,
    )
    object.__setattr__(result, "_source_parent_fit", original_fit)
    if gaussian_hmm_fit_fingerprint(original_fit) != parent_model_fingerprint:
        raise ModelError("latent adequacy parent fit changed during execution")
    object.__setattr__(
        result,
        "source_evidence_fingerprint",
        _latent_source_evidence_fingerprint(result),
    )
    object.__setattr__(result, "_execution_seal", _REGISTERED_LATENT_CAPABILITY)
    _MINTED_REGISTERED_LATENT_CELLS[id(result)] = result
    registered_latent_cell_identity(result)
    return result


def _latent_simulation_rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    chain_count: int,
    measured_months: int,
    extension_months: int,
) -> str:
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="latent_adequacy_simulation",
        contract={
            "artifact": int(artifact),
            "chain_count": chain_count,
            "extension_months": extension_months,
            "measured_months": measured_months,
            "purpose": int(CalibrationPurpose.LATENT_PPC),
            "state_stage": int(Stage.STATE_CHAIN),
        },
    )


def _latent_bootstrap_rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    artifact: CalibrationArtifact,
    bootstrap_replicates: int,
    chain_count: int,
    resample_size: int,
    survival_months: int,
) -> str:
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="latent_adequacy_bootstrap",
        contract={
            "artifact": int(artifact),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_stage": int(Stage.REFERENCE_RESAMPLING),
            "chain_count": chain_count,
            "purpose": int(CalibrationPurpose.LATENT_PPC),
            "resample_size": resample_size,
            "survival_months": survival_months,
        },
    )


def _latent_rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    simulation_rng_fingerprint: str,
    bootstrap_rng_fingerprint: str,
) -> str:
    if (
        _SHA256.fullmatch(simulation_rng_fingerprint) is None
        or _SHA256.fullmatch(bootstrap_rng_fingerprint) is None
    ):
        raise ModelError("latent split RNG identity is invalid")
    return rng_execution_fingerprint(
        master_seed=master_seed,
        scientific_version=scientific_version,
        namespace="latent_adequacy_composite",
        contract={
            "simulation_rng_fingerprint": simulation_rng_fingerprint,
            "bootstrap_rng_fingerprint": bootstrap_rng_fingerprint,
        },
    )


def _verify_latent_result_geometry(value: RegisteredLatentCell) -> None:
    simulation = value.simulation
    bootstrap = value.bootstrap
    endpoint_count = len(bootstrap.target.labels)
    if (
        endpoint_count == 0
        or value.chain_streams != simulation.chain_count
        or value.bootstrap_streams != bootstrap.bootstrap_replicates
        or value.state_uniforms_per_chain != simulation.uniforms_per_chain
        or value.resampled_ids_per_bootstrap != bootstrap.resample_size
        or simulation.chain_count != bootstrap.chain_count
        or simulation.measured_months != bootstrap.measured_months
        or simulation.extension_months != bootstrap.extension_months
        or bootstrap.resample_size != simulation.chain_count
        or simulation.chains.shape
        != (simulation.chain_count, simulation.uniforms_per_chain)
        or simulation.uniforms_per_chain
        != 1 + simulation.measured_months + simulation.extension_months
        or simulation.transition_matrix.ndim != 2
        or simulation.transition_matrix.shape[0]
        != simulation.transition_matrix.shape[1]
        or simulation.stationary_distribution.shape
        != (simulation.transition_matrix.shape[0],)
        or bool((simulation.chains >= simulation.transition_matrix.shape[0]).any())
        or bootstrap.target.labels != bootstrap.pooled_estimate.labels
        or bootstrap.bootstrap_estimates.shape
        != (bootstrap.bootstrap_replicates, endpoint_count)
        or bootstrap.standard_errors.shape != (endpoint_count,)
        or bootstrap.active_components.shape != (endpoint_count,)
        or bootstrap.bootstrap_max_statistics.shape != (bootstrap.bootstrap_replicates,)
        or not 0.0 <= bootstrap.p_value <= 1.0
    ):
        raise ModelError("registered latent cell geometry is inconsistent")
    for array, label in (
        (simulation.transition_matrix, "latent transition matrix"),
        (simulation.stationary_distribution, "latent stationary distribution"),
        (simulation.chains, "latent chains"),
        (bootstrap.target.values, "latent target"),
        (bootstrap.pooled_estimate.values, "latent pooled estimate"),
        (bootstrap.bootstrap_estimates, "latent bootstrap estimates"),
        (bootstrap.standard_errors, "latent standard errors"),
        (bootstrap.active_components, "latent active components"),
        (bootstrap.bootstrap_max_statistics, "latent bootstrap maxima"),
    ):
        if not isinstance(array, np.ndarray) or array.flags.writeable:
            raise ModelError(f"{label} must be an immutable array")


def _latent_source_evidence_fingerprint(value: RegisteredLatentCell) -> str:
    simulation = value.simulation
    bootstrap = value.bootstrap
    return scientific_materialization_fingerprint(
        schema_id="registered_latent_cell_source",
        metadata={
            "artifact": int(value.artifact),
            "master_seed": value.master_seed,
            "scientific_version": value.scientific_version,
            "parent_model_fingerprint": value.parent_model_fingerprint,
            "transition_matrix_fingerprint": value.transition_matrix_fingerprint,
            "simulation_rng_execution_fingerprint": (
                value.simulation_rng_execution_fingerprint
            ),
            "bootstrap_rng_execution_fingerprint": (
                value.bootstrap_rng_execution_fingerprint
            ),
            "rng_execution_fingerprint": value.rng_execution_fingerprint,
            "chain_streams": value.chain_streams,
            "bootstrap_streams": value.bootstrap_streams,
            "state_uniforms_per_chain": value.state_uniforms_per_chain,
            "resampled_ids_per_bootstrap": value.resampled_ids_per_bootstrap,
            "measured_months": simulation.measured_months,
            "extension_months": simulation.extension_months,
            "survival_months": bootstrap.survival_months,
            "bootstrap_chunk_size": bootstrap.chunk_size,
            "target_labels": bootstrap.target.labels,
            "pooled_labels": bootstrap.pooled_estimate.labels,
            "observed_max_statistic": bootstrap.observed_max_statistic,
            "p_value": bootstrap.p_value,
        },
        arrays={
            "transition_matrix": simulation.transition_matrix,
            "stationary_distribution": simulation.stationary_distribution,
            "chains": simulation.chains,
            "target_values": bootstrap.target.values,
            "pooled_values": bootstrap.pooled_estimate.values,
            "bootstrap_estimates": bootstrap.bootstrap_estimates,
            "standard_errors": bootstrap.standard_errors,
            "active_components": bootstrap.active_components,
            "bootstrap_max_statistics": bootstrap.bootstrap_max_statistics,
        },
    )


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
