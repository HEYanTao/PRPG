from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, qmc

from prpg.simulation.rng import sobol_direction_seed
from prpg.validation.sobol import (
    SOBOL_CLIP_LOWER,
    SOBOL_CLIP_UPPER,
    registered_sobol_directions,
)


def test_registered_sobol_directions_match_frozen_scipy_recipe() -> None:
    master_seed = 202_607_14
    scientific_version = 3
    result = registered_sobol_directions(
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    seed = sobol_direction_seed(
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    uniforms = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(m=9)
    normals = norm.ppf(np.clip(uniforms, SOBOL_CLIP_LOWER, SOBOL_CLIP_UPPER))
    expected = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    assert result.directions.shape == (512, 3)
    assert np.array_equal(result.directions, expected)
    assert np.allclose(
        np.linalg.norm(result.directions, axis=1), 1.0, rtol=0.0, atol=1e-15
    )
    assert not result.directions.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.directions.setflags(write=True)


def test_registered_sobol_directions_are_byte_reproducible() -> None:
    left = registered_sobol_directions(master_seed=44, scientific_version=3)
    right = registered_sobol_directions(master_seed=44, scientific_version=3)

    assert left.fingerprint == right.fingerprint
    assert left.directions_sha256 == right.directions_sha256
    assert np.array_equal(left.directions, right.directions)
