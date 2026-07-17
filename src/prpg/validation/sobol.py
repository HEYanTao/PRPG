"""Registered scrambled-Sobol directions for G5 joint-distance metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, qmc

from prpg.errors import ValidationError
from prpg.simulation.rng import sobol_direction_seed

SOBOL_DIMENSION: Final = 3
SOBOL_BASE2_EXPONENT: Final = 9
SOBOL_DIRECTION_COUNT: Final = 1 << SOBOL_BASE2_EXPONENT
SOBOL_CLIP_LOWER: Final = 2.0**-53
SOBOL_CLIP_UPPER: Final = 1.0 - SOBOL_CLIP_LOWER
SOBOL_METHOD: Final = "scipy-scrambled-sobol-normalized-v1"

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SobolDirectionSet:
    """One immutable registered 512-by-3 unit-direction matrix."""

    scientific_version: int
    registered_seed: int
    method: str
    dimension: int
    count: int
    clip_lower: float
    clip_upper: float
    directions: FloatArray
    directions_sha256: str
    fingerprint: str

    def identity_dict(self) -> dict[str, str | float | int]:
        """Return compact identity metadata bound by ``fingerprint``."""

        return {
            "scientific_version": self.scientific_version,
            "registered_seed": self.registered_seed,
            "method": self.method,
            "dimension": self.dimension,
            "count": self.count,
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "directions_sha256": self.directions_sha256,
        }


def registered_sobol_directions(
    *,
    master_seed: int,
    scientific_version: int,
) -> SobolDirectionSet:
    """Generate the approved 512 clipped-normalized scrambled directions.

    The engine seed comes only from the closed PRPG RNG registry.  Uniform
    coordinates are clipped before the normal inverse CDF, and every resulting
    three-vector is normalized independently.
    """

    try:
        seed = sobol_direction_seed(
            master_seed=master_seed,
            scientific_version=scientific_version,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("Sobol RNG identity is invalid") from error
    engine = qmc.Sobol(d=SOBOL_DIMENSION, scramble=True, seed=seed)
    uniforms = np.asarray(
        engine.random_base2(m=SOBOL_BASE2_EXPONENT),
        dtype=np.float64,
    )
    if uniforms.shape != (SOBOL_DIRECTION_COUNT, SOBOL_DIMENSION):
        raise ValidationError("Sobol engine returned an unexpected geometry")
    clipped = np.clip(uniforms, SOBOL_CLIP_LOWER, SOBOL_CLIP_UPPER)
    normals = np.asarray(norm.ppf(clipped), dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if (
        not bool(np.isfinite(normals).all())
        or not bool(np.isfinite(lengths).all())
        or bool((lengths <= 0.0).any())
    ):
        raise ValidationError("Sobol normal transformation is not normalizable")
    directions = normals / lengths[:, np.newaxis]
    if not bool(np.isfinite(directions).all()):
        raise ValidationError("Sobol directions must all be finite")

    # A bytes owner makes the array genuinely read-only: callers cannot simply
    # flip WRITEABLE back to true on this evidence value.
    little_endian = np.asarray(directions, dtype="<f8", order="C")
    frozen = np.frombuffer(little_endian.tobytes(order="C"), dtype="<f8").reshape(
        SOBOL_DIRECTION_COUNT,
        SOBOL_DIMENSION,
    )
    direction_digest = hashlib.sha256(frozen.tobytes(order="C")).hexdigest()
    identity: dict[str, str | float | int] = {
        "scientific_version": scientific_version,
        "registered_seed": seed,
        "method": SOBOL_METHOD,
        "dimension": SOBOL_DIMENSION,
        "count": SOBOL_DIRECTION_COUNT,
        "clip_lower": SOBOL_CLIP_LOWER,
        "clip_upper": SOBOL_CLIP_UPPER,
        "directions_sha256": direction_digest,
    }
    fingerprint = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    result = SobolDirectionSet(
        scientific_version=scientific_version,
        registered_seed=seed,
        method=SOBOL_METHOD,
        dimension=SOBOL_DIMENSION,
        count=SOBOL_DIRECTION_COUNT,
        clip_lower=SOBOL_CLIP_LOWER,
        clip_upper=SOBOL_CLIP_UPPER,
        directions=frozen,
        directions_sha256=direction_digest,
        fingerprint=fingerprint,
    )
    if not np.allclose(
        np.linalg.norm(result.directions, axis=1),
        1.0,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    ):
        raise ValidationError("Sobol directions are not unit length")
    return result


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
