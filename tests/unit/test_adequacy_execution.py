from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import tests.unit.test_g3_calibration_views as calibration_fixtures

from prpg.errors import ModelError
from prpg.model.adequacy_execution import (
    is_registered_latent_cell,
    registered_latent_cell_identity,
    registered_latent_cell_source_fit,
    run_registered_latent_cell,
)
from prpg.model.hmm import GaussianHMMFit
from prpg.model.materialization_identity import gaussian_hmm_fit_fingerprint
from prpg.simulation.rng import CalibrationArtifact


def _fit(artifact: CalibrationArtifact) -> GaussianHMMFit:
    if artifact is CalibrationArtifact.DESIGN_OR_CONTROL:
        source = calibration_fixtures._fit(193)
    else:
        source = calibration_fixtures._fit(
            217,
            scaler_scope="production_full",
            lengths=(210, 7),
        )
    return replace(
        source,
        parameters=replace(
            source.parameters,
            transition_matrix=np.asarray([[0.75, 0.25], [0.20, 0.80]]),
        ),
    )


def test_registered_latent_cell_is_deterministic_and_counts_every_stream() -> None:
    fit = _fit(CalibrationArtifact.DESIGN_OR_CONTROL)
    kwargs = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": CalibrationArtifact.DESIGN_OR_CONTROL,
        "chain_count": 20,
        "measured_months": 25,
        "extension_months": 100,
        "bootstrap_replicates": 30,
        "resample_size": 20,
        "survival_months": 5,
        "generation_chunk_size": 7,
        "bootstrap_chunk_size": 6,
    }
    first = run_registered_latent_cell(fit, **kwargs)  # type: ignore[arg-type]
    second = run_registered_latent_cell(fit, **kwargs)  # type: ignore[arg-type]

    np.testing.assert_array_equal(first.simulation.chains, second.simulation.chains)
    np.testing.assert_array_equal(
        first.bootstrap.bootstrap_estimates,
        second.bootstrap.bootstrap_estimates,
    )
    assert first.chain_streams == 20
    assert first.bootstrap_streams == 30
    assert first.state_uniforms_per_chain == 126
    assert first.resampled_ids_per_bootstrap == 20
    assert first.parent_model_fingerprint == gaussian_hmm_fit_fingerprint(fit)
    assert registered_latent_cell_source_fit(first) is fit
    assert len(first.transition_matrix_fingerprint) == 64
    assert len(first.simulation_rng_execution_fingerprint) == 64
    assert len(first.bootstrap_rng_execution_fingerprint) == 64
    assert len(first.rng_execution_fingerprint) == 64
    assert len(first.source_evidence_fingerprint) == 64
    assert (
        first.simulation_rng_execution_fingerprint
        != first.bootstrap_rng_execution_fingerprint
    )
    identity = registered_latent_cell_identity(first)
    assert identity.source_evidence_fingerprint == first.source_evidence_fingerprint
    assert is_registered_latent_cell(first)
    assert not is_registered_latent_cell(replace(first))
    assert not is_registered_latent_cell(copy.copy(first))

    changed_seed = run_registered_latent_cell(
        fit,
        **{**kwargs, "master_seed": 124},  # type: ignore[arg-type]
    )
    assert changed_seed.rng_execution_fingerprint != first.rng_execution_fingerprint


def test_registered_latent_cell_rejects_writable_or_tampered_result_arrays() -> None:
    kwargs = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": CalibrationArtifact.DESIGN_OR_CONTROL,
        "chain_count": 20,
        "measured_months": 25,
        "extension_months": 100,
        "bootstrap_replicates": 30,
        "resample_size": 20,
        "survival_months": 5,
    }
    writable = run_registered_latent_cell(
        _fit(CalibrationArtifact.DESIGN_OR_CONTROL),
        **kwargs,  # type: ignore[arg-type]
    )
    writable.simulation.chains.setflags(write=True)
    assert not is_registered_latent_cell(writable)

    tampered = run_registered_latent_cell(
        _fit(CalibrationArtifact.DESIGN_OR_CONTROL),
        **kwargs,  # type: ignore[arg-type]
    )
    tampered.bootstrap.bootstrap_estimates.setflags(write=True)
    tampered.bootstrap.bootstrap_estimates[0, 0] += 0.01
    tampered.bootstrap.bootstrap_estimates.setflags(write=False)
    assert not is_registered_latent_cell(tampered)


def test_artifact_changes_registered_trace() -> None:
    common = {
        "master_seed": 123,
        "scientific_version": 1,
        "chain_count": 10,
        "measured_months": 20,
        "extension_months": 100,
        "bootstrap_replicates": 10,
        "resample_size": 10,
        "survival_months": 4,
    }
    design = run_registered_latent_cell(
        _fit(CalibrationArtifact.DESIGN_OR_CONTROL),
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        **common,  # type: ignore[arg-type]
    )
    production = run_registered_latent_cell(
        _fit(CalibrationArtifact.PRODUCTION_OR_CANDIDATE),
        artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        **common,  # type: ignore[arg-type]
    )
    assert not np.array_equal(design.simulation.chains, production.simulation.chains)


@pytest.mark.parametrize(
    "updates",
    [
        {"artifact": CalibrationArtifact.BRIDGE_MODEL},
        {"chain_count": 10_001, "resample_size": 10_001},
        {"chain_count": 10, "resample_size": 9},
    ],
)
def test_registered_latent_inputs_fail_closed(updates: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": CalibrationArtifact.DESIGN_OR_CONTROL,
        "chain_count": 10,
        "measured_months": 20,
        "extension_months": 100,
        "bootstrap_replicates": 10,
        "resample_size": 10,
        "survival_months": 4,
    }
    kwargs.update(updates)
    with pytest.raises(ModelError):
        run_registered_latent_cell(
            _fit(CalibrationArtifact.DESIGN_OR_CONTROL),
            **kwargs,  # type: ignore[arg-type]
        )


def test_registered_latent_requires_and_retains_exact_typed_parent() -> None:
    common = {
        "master_seed": 123,
        "scientific_version": 1,
        "artifact": CalibrationArtifact.DESIGN_OR_CONTROL,
        "chain_count": 10,
        "measured_months": 20,
        "extension_months": 100,
        "bootstrap_replicates": 10,
        "resample_size": 10,
        "survival_months": 4,
    }
    with pytest.raises(ModelError, match="typed GaussianHMMFit"):
        run_registered_latent_cell(object(), **common)  # type: ignore[arg-type]

    fit = _fit(CalibrationArtifact.DESIGN_OR_CONTROL)
    result = run_registered_latent_cell(fit, **common)  # type: ignore[arg-type]
    fit.parameters.transition_matrix[0, 0] -= 0.01
    assert not is_registered_latent_cell(result)
