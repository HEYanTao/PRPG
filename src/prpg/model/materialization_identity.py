"""Deterministic fingerprints for in-memory scientific materializations.

The Phase-3 task graph moves typed numerical objects between independently
checkpointed tasks.  Python type identity and a frozen dataclass are not
enough to prevent a result from one model, input slice, or run being paired
with another.  This module gives the numerical parent objects a compact,
platform-independent SHA-256 identity that task bindings can retain.

The hashes are identities, not serialization formats and not authorization
capabilities.  Arrays are normalized to explicit little-endian dtypes and
hashed together with their names and shapes.  Text geometry is encoded as
canonical JSON.  No pickle bytes or object representations enter a digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from prpg.errors import ModelError
from prpg.model.hmm import GaussianHMMFit, HMMFeatureMatrix, HMMParameters
from prpg.model.regime_policy import validate_owner_fixed_k4_disposition

MATERIALIZATION_IDENTITY_SCHEMA_VERSION: Final = 1
_MODEL_SCHEMA_ID: Final = "prpg-hmm-model-materialization-v2"
_FEATURE_SCHEMA_ID: Final = "prpg-hmm-feature-materialization-v1"
_PARAMETER_SCHEMA_ID: Final = "prpg-hmm-parameter-materialization-v1"
_ARRAY_SCHEMA_ID: Final = "prpg-scientific-array-materialization-v1"


def scientific_array_fingerprint(
    values: npt.ArrayLike,
    *,
    role: str,
) -> str:
    """Fingerprint one finite numeric or Boolean array under a named role."""

    _validate_role(role)
    return _fingerprint(
        schema_id=_ARRAY_SCHEMA_ID,
        metadata={"role": role},
        arrays={role: values},
    )


def scientific_materialization_fingerprint(
    *,
    schema_id: str,
    metadata: Mapping[str, object],
    arrays: Mapping[str, npt.ArrayLike],
) -> str:
    """Fingerprint a closed named scientific record and its numeric arrays.

    Callers own the schema-specific validation.  This shared encoder still
    enforces a stable lowercase schema identifier, canonical JSON metadata,
    nonempty uniquely named arrays, supported dtypes, and finite floats.
    """

    _validate_role(schema_id)
    if not isinstance(metadata, Mapping):
        raise ModelError("scientific materialization metadata must be a mapping")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ModelError("scientific materialization arrays must be nonempty")
    return _fingerprint(
        schema_id=f"prpg-{schema_id}-v1",
        metadata=metadata,
        arrays=arrays,
    )


def hmm_parameter_fingerprint(parameters: HMMParameters) -> str:
    """Fingerprint the four canonical Gaussian-HMM parameter arrays."""

    if not isinstance(parameters, HMMParameters):
        raise ModelError("HMM parameter fingerprint requires HMMParameters")
    return _fingerprint(
        schema_id=_PARAMETER_SCHEMA_ID,
        metadata={
            "n_states": parameters.n_states,
            "n_features": parameters.n_features,
        },
        arrays={
            "start_probabilities": parameters.start_probabilities,
            "transition_matrix": parameters.transition_matrix,
            "means": parameters.means,
            "tied_covariance": parameters.tied_covariance,
        },
    )


def gaussian_hmm_fit_fingerprint(fit: GaussianHMMFit) -> str:
    """Fingerprint the fitted model/scaler/path materialization used downstream.

    Restart and diagnostic reports deliberately do not enter this identity:
    their task checkpoint fingerprints carry the scientific decision.  This
    digest identifies the exact numerical model, scaler, decoded path, source
    geometry, and chosen restart consumed by a downstream computation.
    """

    if not isinstance(fit, GaussianHMMFit):
        raise ModelError("HMM model fingerprint requires GaussianHMMFit")
    validate_owner_fixed_k4_disposition(fit)
    lengths = _positive_lengths(fit.lengths)
    n_observations = _positive_integer(fit.n_observations, "n_observations")
    metadata = {
        "best_restart_index": _integer(fit.best_restart_index, "best_restart_index"),
        "best_seed": _integer(fit.best_seed, "best_seed"),
        "feature_names": _strings(fit.feature_names, "feature_names"),
        "lengths": lengths,
        "n_features": _positive_integer(fit.n_features, "n_features"),
        "n_observations": n_observations,
        "n_states": _positive_integer(fit.n_states, "n_states"),
        "parameter_count": _positive_integer(fit.parameter_count, "parameter_count"),
        "scaler_scope": fit.scaler_scope,
        "owner_fixed_k4_disposition": (
            None
            if fit.owner_fixed_k4_disposition is None
            else fit.owner_fixed_k4_disposition.as_dict()
        ),
    }
    if fit.scaler_scope not in {"design_training", "production_full"}:
        raise ModelError("HMM model fingerprint has an invalid scaler scope")
    if sum(lengths) != n_observations:
        raise ModelError("HMM model fingerprint geometry is inconsistent")
    return _fingerprint(
        schema_id=_MODEL_SCHEMA_ID,
        metadata=metadata,
        arrays={
            "canonical_to_original": fit.canonical_to_original,
            "decoded_states": fit.decoded_states,
            "means": fit.parameters.means,
            "original_to_canonical": fit.original_to_canonical,
            "scaler_means": fit.scaler_means,
            "scaler_standard_deviations": fit.scaler_standard_deviations,
            "start_probabilities": fit.parameters.start_probabilities,
            "tied_covariance": fit.parameters.tied_covariance,
            "transition_matrix": fit.parameters.transition_matrix,
        },
    )


def hmm_feature_matrix_fingerprint(features: HMMFeatureMatrix) -> str:
    """Fingerprint a scaler-scoped feature matrix and its exact row geometry."""

    if not isinstance(features, HMMFeatureMatrix):
        raise ModelError("feature fingerprint requires HMMFeatureMatrix")
    lengths = _positive_lengths(features.lengths)
    feature_names = _strings(features.feature_names, "feature_names")
    row_labels = _strings(features.row_labels, "row_labels")
    values = np.asarray(features.values)
    if values.ndim != 2:
        raise ModelError("feature fingerprint values must be two-dimensional")
    if sum(lengths) != values.shape[0] or len(row_labels) != values.shape[0]:
        raise ModelError("feature fingerprint row geometry is inconsistent")
    if len(feature_names) != values.shape[1]:
        raise ModelError("feature fingerprint column geometry is inconsistent")
    if features.scaler_scope not in {"design_training", "production_full"}:
        raise ModelError("feature fingerprint has an invalid scaler scope")
    return _fingerprint(
        schema_id=_FEATURE_SCHEMA_ID,
        metadata={
            "feature_names": feature_names,
            "lengths": lengths,
            "row_labels": row_labels,
            "scaler_scope": features.scaler_scope,
        },
        arrays={
            "scaler_means": features.scaler_means,
            "scaler_standard_deviations": features.scaler_standard_deviations,
            "values": values,
        },
    )


def _fingerprint(
    *,
    schema_id: str,
    metadata: Mapping[str, object],
    arrays: Mapping[str, npt.ArrayLike],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "schema_id": schema_id,
                "schema_version": MATERIALIZATION_IDENTITY_SCHEMA_VERSION,
                "metadata": metadata,
                "array_names": sorted(arrays),
            }
        )
    )
    for name, value in sorted(arrays.items()):
        _validate_role(name)
        canonical = _canonical_array(value, name=name)
        header = _canonical_json_bytes(
            {
                "name": name,
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
                "nbytes": canonical.nbytes,
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        content = canonical.tobytes(order="C")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _canonical_array(value: npt.ArrayLike, *, name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ModelError(
            "scientific materialization array cannot be converted",
            details={"name": name},
        ) from error
    if source.ndim == 0 or source.size == 0:
        raise ModelError(
            "scientific materialization array must be nonempty and nonscalar",
            details={"name": name},
        )
    if source.dtype.fields is not None:
        raise ModelError(
            "scientific materialization array cannot be structured",
            details={"name": name},
        )
    kind = source.dtype.kind
    if kind == "f":
        if not bool(np.isfinite(source).all()):
            raise ModelError(
                "scientific materialization array must be finite",
                details={"name": name},
            )
        target: np.dtype[Any] = np.dtype("<f8")
    elif kind in {"i", "u"}:
        target = np.dtype("<i8") if kind == "i" else np.dtype("<u8")
    elif kind == "b":
        target = np.dtype("?")
    else:
        raise ModelError(
            "scientific materialization array has an unsupported dtype",
            details={"name": name, "dtype": source.dtype.str},
        )
    return np.ascontiguousarray(source.astype(target, copy=False))


def _canonical_json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError(
            "scientific materialization metadata is not canonical"
        ) from error


def _validate_role(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in value
        )
        or not value[0].isalpha()
    ):
        raise ModelError("scientific materialization array role is invalid")


def _strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ModelError(f"{label} must be a nonempty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ModelError(f"{label} contains an invalid value")
    return values


def _positive_lengths(values: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ModelError("HMM materialization lengths must be a nonempty tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ModelError("HMM materialization lengths must be positive integers")
    return values


def _integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: int, label: str) -> int:
    checked = _integer(value, label)
    if checked == 0:
        raise ModelError(f"{label} must be positive")
    return checked


def rng_execution_fingerprint(
    *,
    master_seed: int,
    scientific_version: int,
    namespace: str,
    contract: Mapping[str, object],
) -> str:
    """Fingerprint a closed registered-RNG namespace and its exact geometry."""

    _validate_role(namespace)
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or master_seed < 0
    ):
        raise ModelError("registered RNG master_seed must be a nonnegative integer")
    if (
        isinstance(scientific_version, bool)
        or not isinstance(scientific_version, int)
        or scientific_version <= 0
    ):
        raise ModelError("registered RNG scientific_version must be positive")
    if not isinstance(contract, Mapping):
        raise ModelError("registered RNG contract must be a mapping")
    for value in contract.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ModelError("registered RNG contract contains a non-finite value")
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_id": "prpg-registered-rng-execution-v1",
                "master_seed": master_seed,
                "scientific_version": scientific_version,
                "namespace": namespace,
                "contract": contract,
            }
        )
    ).hexdigest()
