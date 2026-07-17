"""Practical owner-approved K=4 model authority.

This module is the deliberately small replacement for the failed G3 research
gate path.  It reproduces exactly the already reviewed DESIGN_MAIN restart,
checks only that its complete month labels match the owner's four intuitive
investment environments, and persists the numerical material needed by the
existing conditioned return generator.

It does not run candidate selection, posterior-identification thresholds,
holdout campaigns, macro bootstraps, rolling stability, or a production
refit.  Those former research gates are not part of this delivery authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from prpg.errors import IntegrityError, ModelError
from prpg.model.block_length import BLOCK_POLICY_VERSION
from prpg.model.calibration import prepare_hmm_feature_matrix
from prpg.model.hmm import (
    AuditedTiedGaussianHMM,
    HMMParameters,
    canonicalize_hmm,
    decode_viterbi,
    validate_markov_chain,
)
from prpg.model.input import CalibrationInput, CalibrationSlice
from prpg.model.scientific_policy import (
    CANONICAL_NEUTRAL_LAMBDA_GRID,
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    SCIENTIFIC_GENERATOR_ALGORITHM,
    ScientificGenerationPolicy,
    generation_policy_from_mapping,
)
from prpg.model.stability import map_monthly_states_to_daily
from prpg.simulation.rng import ModelFitScope, hmm_library_seed

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PRACTICAL_K4_AUTHORITY_SCHEMA_ID: Final = "prpg-practical-k4-authority-v1"
PRACTICAL_K4_AUTHORITY_SCHEMA_VERSION: Final = 1
OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT: Final = (
    "84a9604691ae43f6e5c9325ac4001e61477931922816b50516f141d91b017fb5"
)
OWNER_APPROVED_SOURCE_CONFIG_FINGERPRINT: Final = (
    "572e0a843a540b0a6e307cfb72c3c9c43d2fb6cafc8c2c52f6691f408c38ae58"
)
OWNER_APPROVED_MASTER_SEED: Final = 20_260_714
OWNER_APPROVED_SCIENTIFIC_VERSION: Final = 11
OWNER_APPROVED_RESTART_INDEX: Final = 43
OWNER_APPROVED_RESTART_SEED: Final = 2_529_155_491
OWNER_APPROVED_LOG_LIKELIHOOD: Final = -752.7283424245143
OWNER_APPROVED_ITERATIONS: Final = 37
OWNER_APPROVED_STATE_COUNTS: Final = (11, 26, 95, 61)
OWNER_APPROVED_DESIGN_MONTHS: Final = 193
OWNER_APPROVED_DESIGN_DAILY_SESSIONS: Final = 4_049
OWNER_APPROVED_DESIGN_START: Final = "2008-04"
OWNER_APPROVED_DESIGN_END: Final = "2024-04"

OWNER_APPROVED_STATE_NAMES: Final = (
    "acute_contraction_credit_crisis",
    "high_inflation_rate_stress",
    "steep_curve_recovery_easy_policy_expansion",
    "flat_curve_tight_credit_risk_on",
)
OWNER_APPROVED_MONTH_RANGES: Final = (
    ("2008-09/2009-05", "2020-03/2020-04"),
    ("2008-04/2008-08", "2021-04/2022-12"),
    ("2009-06/2017-03", "2020-05/2020-05"),
    ("2017-04/2020-02", "2020-06/2021-03", "2023-01/2024-04"),
)
OWNER_APPROVED_RAW_CENTROIDS: Final = np.asarray(
    [
        [-2.945288077602296, 0.8378002484628133, 2.3571397948586847, 4.904187298878389],
        [
            -0.07070343360676756,
            6.367498479400343,
            1.281291949964363,
            2.2367255858202597,
        ],
        [0.1780041919752537, 1.4792461586896781, 2.325518463060201, 2.8146855772153514],
        [
            0.2626896034974619,
            2.4222356536901115,
            0.1978288346627557,
            2.0643220982999617,
        ],
    ],
    dtype=np.float64,
)
OWNER_APPROVED_TRANSITION_MATRIX: Final = np.asarray(
    [
        [0.81819068825022, 0.0, 0.18180931174978004, 0.0],
        [0.038584563175614914, 0.922820914714835, 0.0, 0.03859452210955017],
        [0.0, 0.0, 0.9789544864211729, 0.02104551357882709],
        [0.016652663249793698, 0.016657288494689147, 0.0, 0.9666900482555172],
    ],
    dtype=np.float64,
)

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_NAMES: Final = (
    "start_probabilities",
    "transition_matrix",
    "means_standardized",
    "tied_covariance",
    "scaler_means",
    "scaler_standard_deviations",
    "monthly_period_ordinals",
    "daily_session_ns",
    "monthly_source_states",
    "daily_source_states",
)


@dataclass(frozen=True, slots=True)
class PracticalK4ArtifactReference:
    """Location of one content-addressed practical K=4 artifact."""

    fingerprint: str
    directory: Path


@dataclass(frozen=True, slots=True)
class PracticalK4Artifact:
    """Compact authority for the exact owner-reviewed K=4 generation model."""

    schema_id: str
    schema_version: int
    fingerprint: str
    model_fingerprint: str
    processed_data_fingerprint: str
    design_input_fingerprint: str
    source_config_fingerprint: str
    master_seed: int
    scientific_version: int
    fit_scope: str
    fit_replicate: int
    best_restart_index: int
    best_seed: int
    converged: bool
    iterations: int
    log_likelihood: float
    state_names: tuple[str, str, str, str]
    month_ranges: tuple[tuple[str, ...], ...]
    generation_policy: ScientificGenerationPolicy
    start_probabilities: FloatArray
    transition_matrix: FloatArray
    means_standardized: FloatArray
    tied_covariance: FloatArray
    scaler_means: FloatArray
    scaler_standard_deviations: FloatArray
    monthly_period_ordinals: IntArray
    daily_session_ns: IntArray
    monthly_source_states: IntArray
    daily_source_states: IntArray


class PracticalK4ArtifactStore:
    """Minimal content-addressed persistence for practical K=4 authority."""

    def __init__(self, artifact_root: str | Path) -> None:
        root = Path(artifact_root).expanduser()
        self.root = root / "models" / "practical-k4" / "sha256"

    def contains(self, fingerprint: str) -> bool:
        return (
            _checked_fingerprint(fingerprint, "artifact")
            and (self.root / fingerprint / "manifest.json").is_file()
        )

    def persist(self, artifact: PracticalK4Artifact) -> PracticalK4ArtifactReference:
        checked = verify_practical_k4_artifact(artifact)
        destination = self.root / checked.fingerprint
        if destination.exists():
            self.load(checked.fingerprint)
            return PracticalK4ArtifactReference(checked.fingerprint, destination)

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".practical-k4-", dir=self.root))
        try:
            for name in _ARRAY_NAMES:
                np.save(
                    temporary / f"{name}.npy",
                    getattr(checked, name),
                    allow_pickle=False,
                )
            manifest = {
                "identity": _artifact_identity(checked),
                "fingerprint": checked.fingerprint,
            }
            (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        loaded = self.load(checked.fingerprint)
        if loaded.fingerprint != checked.fingerprint:  # pragma: no cover
            raise IntegrityError("persisted practical K=4 artifact changed")
        return PracticalK4ArtifactReference(checked.fingerprint, destination)

    def load(self, fingerprint: str) -> PracticalK4Artifact:
        _require_fingerprint(fingerprint, "artifact fingerprint")
        directory = self.root / fingerprint
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "practical K=4 artifact manifest is unreadable"
            ) from error
        if not isinstance(manifest, dict) or set(manifest) != {
            "identity",
            "fingerprint",
        }:
            raise IntegrityError("practical K=4 artifact manifest is invalid")
        if manifest["fingerprint"] != fingerprint or not isinstance(
            manifest["identity"], dict
        ):
            raise IntegrityError("practical K=4 artifact address is inconsistent")
        identity = manifest["identity"]
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        try:
            for name in _ARRAY_NAMES:
                arrays[name] = _immutable(
                    np.load(directory / f"{name}.npy", allow_pickle=False)
                )
        except (OSError, ValueError) as error:
            raise IntegrityError(
                "practical K=4 artifact array is unreadable"
            ) from error
        try:
            policy = generation_policy_from_mapping(identity["generation_policy"])
            artifact = PracticalK4Artifact(
                schema_id=identity["schema_id"],
                schema_version=identity["schema_version"],
                fingerprint=fingerprint,
                model_fingerprint=identity["model_fingerprint"],
                processed_data_fingerprint=identity["processed_data_fingerprint"],
                design_input_fingerprint=identity["design_input_fingerprint"],
                source_config_fingerprint=identity["source_config_fingerprint"],
                master_seed=identity["master_seed"],
                scientific_version=identity["scientific_version"],
                fit_scope=identity["fit_scope"],
                fit_replicate=identity["fit_replicate"],
                best_restart_index=identity["best_restart_index"],
                best_seed=identity["best_seed"],
                converged=identity["converged"],
                iterations=identity["iterations"],
                log_likelihood=identity["log_likelihood"],
                state_names=tuple(identity["state_names"]),
                month_ranges=tuple(tuple(item) for item in identity["month_ranges"]),
                generation_policy=policy,
                **arrays,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                "practical K=4 artifact identity is invalid"
            ) from error
        return verify_practical_k4_artifact(artifact)


def practical_historical_vector_policy() -> ScientificGenerationPolicy:
    """Return the fixed practical sampling policy approved for delivery."""

    return generation_policy_from_mapping(
        {
            "schema_version": SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
            "generator_algorithm": SCIENTIFIC_GENERATOR_ALGORITHM,
            "block_policy_version": BLOCK_POLICY_VERSION,
            "monthly_block_length": 2,
            "daily_block_length": 62,
            "alpha_policy": "historical_vector",
            "selected_neutral_lambda": None,
            "neutral_lambda_grid": list(CANONICAL_NEUTRAL_LAMBDA_GRID),
            "initial_state": "stationary",
            "synchronized_return_vectors": True,
        },
        error_type=ModelError,
    )


def practical_k4_design_input_fingerprint(
    calibration_input: CalibrationInput,
) -> str:
    """Bind the exact design arrays while allowing smoke/canonical config identities."""

    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("practical K=4 requires a calibration input")
    design = calibration_input.design
    records = {
        name: _array_record(getattr(design, name))
        for name in (
            "monthly_returns",
            "daily_returns",
            "macro_features",
            "monthly_period_ordinals",
            "daily_session_ns",
            "monthly_continuation",
            "daily_continuation",
        )
    }
    return _sha256_json(
        {
            "schema_id": "prpg-practical-k4-design-input-v1",
            "processed_data_fingerprint": (
                calibration_input.provenance.processed_data_fingerprint
            ),
            "arrays": records,
        }
    )


def expected_owner_approved_monthly_states(
    monthly_period_ordinals: NDArray[np.integer[Any]] | np.ndarray[Any, Any],
) -> IntArray:
    """Return the complete owner-reviewed state label for every design month."""

    ordinals = np.asarray(monthly_period_ordinals, dtype=np.int64)
    if ordinals.shape != (OWNER_APPROVED_DESIGN_MONTHS,):
        raise ModelError("owner-approved K=4 labels require exactly 193 months")
    months = pd.PeriodIndex.from_ordinals(ordinals, freq="M")
    if (
        str(months[0]) != OWNER_APPROVED_DESIGN_START
        or str(months[-1]) != OWNER_APPROVED_DESIGN_END
        or not months.is_monotonic_increasing
        or not months.is_unique
    ):
        raise ModelError("owner-approved K=4 design month range changed")
    labels = np.full(len(months), -1, dtype=np.int64)
    for state, ranges in enumerate(OWNER_APPROVED_MONTH_RANGES):
        for encoded in ranges:
            start_text, end_text = encoded.split("/", maxsplit=1)
            selected = (months >= pd.Period(start_text, freq="M")) & (
                months <= pd.Period(end_text, freq="M")
            )
            if bool((labels[selected] >= 0).any()):
                raise ModelError("owner-approved K=4 month ranges overlap")
            labels[selected] = state
    if bool((labels < 0).any()):
        missing = [str(value) for value in months[labels < 0]]
        raise ModelError(
            "owner-approved K=4 month ranges are incomplete",
            details={"missing_months": missing},
        )
    if tuple(np.bincount(labels, minlength=4).tolist()) != OWNER_APPROVED_STATE_COUNTS:
        raise ModelError("owner-approved K=4 state counts changed")
    return _immutable(labels, dtype=np.int64)


def reproduce_owner_approved_k4(
    calibration_input: CalibrationInput,
) -> PracticalK4Artifact:
    """Reproduce the one reviewed restart and seal its intuitive label assignment."""

    _validate_design_source(calibration_input)
    design = calibration_input.design
    features = prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="design",
        design_rows=OWNER_APPROVED_DESIGN_MONTHS,
    )
    derived_seed = hmm_library_seed(
        master_seed=OWNER_APPROVED_MASTER_SEED,
        scientific_version=OWNER_APPROVED_SCIENTIFIC_VERSION,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        k=4,
        restart_index=OWNER_APPROVED_RESTART_INDEX,
    )
    if derived_seed != OWNER_APPROVED_RESTART_SEED:
        raise IntegrityError("owner-approved K=4 RNG contract changed")

    model = AuditedTiedGaussianHMM(
        n_components=4,
        covariance_type="tied",
        min_covar=1e-6,
        startprob_prior=1.0,
        transmat_prior=1.0,
        means_weight=0.0,
        covars_prior=0.0,
        covars_weight=0.0,
        n_iter=1_000,
        tol=1e-6,
        random_state=OWNER_APPROVED_RESTART_SEED,
        algorithm="viterbi",
        implementation="log",
        params="stmc",
        init_params="stmc",
    )
    model.fit(features.values, lengths=features.lengths)
    converged = bool(model.monitor_.converged)
    iterations = int(model.monitor_.iter)
    log_likelihood = float(model.score(features.values, lengths=features.lengths))
    raw_covariance = np.asarray(model._covars_, dtype=np.float64)
    parameters = HMMParameters(
        start_probabilities=np.asarray(model.startprob_, dtype=np.float64),
        transition_matrix=np.asarray(model.transmat_, dtype=np.float64),
        means=np.asarray(model.means_, dtype=np.float64),
        tied_covariance=raw_covariance,
    )
    canonical = canonicalize_hmm(parameters)
    parameters = canonical.parameters
    decoded = decode_viterbi(
        features.values,
        parameters,
        lengths=features.lengths,
    )
    expected = expected_owner_approved_monthly_states(design.monthly_period_ordinals)
    raw_centroids = (
        parameters.means * features.scaler_standard_deviations + features.scaler_means
    )
    if (
        not converged
        or iterations != OWNER_APPROVED_ITERATIONS
        or not math.isclose(
            log_likelihood,
            OWNER_APPROVED_LOG_LIKELIHOOD,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not np.allclose(
            parameters.transition_matrix,
            OWNER_APPROVED_TRANSITION_MATRIX,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            raw_centroids,
            OWNER_APPROVED_RAW_CENTROIDS,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.array_equal(decoded, expected)
    ):
        raise ModelError(
            "reproduced K=4 fit differs from the owner-reviewed month assignment"
        )
    validate_markov_chain(parameters.transition_matrix)
    daily_states = map_monthly_states_to_daily(
        design.monthly_period_ordinals,
        expected,
        design.daily_session_ns,
    )
    artifact = _mint_artifact(
        calibration_input=calibration_input,
        converged=converged,
        iterations=iterations,
        log_likelihood=log_likelihood,
        parameters=parameters,
        scaler_means=features.scaler_means,
        scaler_standard_deviations=features.scaler_standard_deviations,
        monthly_source_states=expected,
        daily_source_states=daily_states,
    )
    return verify_practical_k4_artifact(artifact)


def verify_practical_k4_artifact(value: object) -> PracticalK4Artifact:
    """Recheck the compact authority without invoking any former research gate."""

    if not isinstance(value, PracticalK4Artifact):
        raise IntegrityError("practical K=4 authority has the wrong type")
    policy = practical_historical_vector_policy()
    scalars_match = (
        value.schema_id == PRACTICAL_K4_AUTHORITY_SCHEMA_ID
        and value.schema_version == PRACTICAL_K4_AUTHORITY_SCHEMA_VERSION
        and value.processed_data_fingerprint
        == OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT
        and value.source_config_fingerprint == OWNER_APPROVED_SOURCE_CONFIG_FINGERPRINT
        and value.master_seed == OWNER_APPROVED_MASTER_SEED
        and value.scientific_version == OWNER_APPROVED_SCIENTIFIC_VERSION
        and value.fit_scope == ModelFitScope.DESIGN_MAIN.name
        and value.fit_replicate == 0
        and value.best_restart_index == OWNER_APPROVED_RESTART_INDEX
        and value.best_seed == OWNER_APPROVED_RESTART_SEED
        and value.converged is True
        and value.iterations == OWNER_APPROVED_ITERATIONS
        and math.isclose(
            value.log_likelihood,
            OWNER_APPROVED_LOG_LIKELIHOOD,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and value.state_names == OWNER_APPROVED_STATE_NAMES
        and value.month_ranges == OWNER_APPROVED_MONTH_RANGES
        and value.generation_policy == policy
    )
    if not scalars_match:
        raise IntegrityError("practical K=4 authority metadata changed")
    _require_fingerprint(value.design_input_fingerprint, "design input fingerprint")
    _require_fingerprint(value.model_fingerprint, "model fingerprint")
    _require_fingerprint(value.fingerprint, "artifact fingerprint")
    expected_shapes = {
        "start_probabilities": ((4,), np.dtype(np.float64)),
        "transition_matrix": ((4, 4), np.dtype(np.float64)),
        "means_standardized": ((4, 4), np.dtype(np.float64)),
        "tied_covariance": ((4, 4), np.dtype(np.float64)),
        "scaler_means": ((4,), np.dtype(np.float64)),
        "scaler_standard_deviations": ((4,), np.dtype(np.float64)),
        "monthly_period_ordinals": (
            (OWNER_APPROVED_DESIGN_MONTHS,),
            np.dtype(np.int64),
        ),
        "daily_session_ns": (
            (OWNER_APPROVED_DESIGN_DAILY_SESSIONS,),
            np.dtype(np.int64),
        ),
        "monthly_source_states": (
            (OWNER_APPROVED_DESIGN_MONTHS,),
            np.dtype(np.int64),
        ),
        "daily_source_states": (
            (OWNER_APPROVED_DESIGN_DAILY_SESSIONS,),
            np.dtype(np.int64),
        ),
    }
    for name, (shape, dtype) in expected_shapes.items():
        array = getattr(value, name)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != dtype
            or array.flags.writeable
            or not array.flags.c_contiguous
            or (array.dtype.kind == "f" and not bool(np.isfinite(array).all()))
        ):
            raise IntegrityError(
                "practical K=4 authority array is invalid", details={"array": name}
            )
    expected_states = expected_owner_approved_monthly_states(
        value.monthly_period_ordinals
    )
    mapped_daily = map_monthly_states_to_daily(
        value.monthly_period_ordinals,
        expected_states,
        value.daily_session_ns,
    )
    raw_centroids = (
        value.means_standardized * value.scaler_standard_deviations + value.scaler_means
    )
    parameters = HMMParameters(
        value.start_probabilities,
        value.transition_matrix,
        value.means_standardized,
        value.tied_covariance,
    )
    canonical = canonicalize_hmm(parameters)
    if (
        not np.array_equal(value.monthly_source_states, expected_states)
        or not np.array_equal(value.daily_source_states, mapped_daily)
        or tuple(np.bincount(value.monthly_source_states, minlength=4).tolist())
        != OWNER_APPROVED_STATE_COUNTS
        or not np.array_equal(canonical.canonical_to_original, np.arange(4))
        or not np.allclose(
            raw_centroids,
            OWNER_APPROVED_RAW_CENTROIDS,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            value.transition_matrix,
            OWNER_APPROVED_TRANSITION_MATRIX,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise IntegrityError("practical K=4 authority labels or parameters changed")
    validate_markov_chain(value.transition_matrix)
    model_fingerprint = _model_fingerprint(value)
    if value.model_fingerprint != model_fingerprint:
        raise IntegrityError("practical K=4 model fingerprint changed")
    if value.fingerprint != _artifact_fingerprint(value):
        raise IntegrityError("practical K=4 artifact fingerprint changed")
    return value


def _mint_artifact(
    *,
    calibration_input: CalibrationInput,
    converged: bool,
    iterations: int,
    log_likelihood: float,
    parameters: HMMParameters,
    scaler_means: np.ndarray[Any, Any],
    scaler_standard_deviations: np.ndarray[Any, Any],
    monthly_source_states: np.ndarray[Any, Any],
    daily_source_states: np.ndarray[Any, Any],
) -> PracticalK4Artifact:
    design = calibration_input.design
    arrays = {
        "start_probabilities": _immutable(parameters.start_probabilities, np.float64),
        "transition_matrix": _immutable(parameters.transition_matrix, np.float64),
        "means_standardized": _immutable(parameters.means, np.float64),
        "tied_covariance": _immutable(parameters.tied_covariance, np.float64),
        "scaler_means": _immutable(scaler_means, np.float64),
        "scaler_standard_deviations": _immutable(
            scaler_standard_deviations, np.float64
        ),
        "monthly_period_ordinals": _immutable(design.monthly_period_ordinals, np.int64),
        "daily_session_ns": _immutable(design.daily_session_ns, np.int64),
        "monthly_source_states": _immutable(monthly_source_states, np.int64),
        "daily_source_states": _immutable(daily_source_states, np.int64),
    }
    provisional = PracticalK4Artifact(
        schema_id=PRACTICAL_K4_AUTHORITY_SCHEMA_ID,
        schema_version=PRACTICAL_K4_AUTHORITY_SCHEMA_VERSION,
        fingerprint="0" * 64,
        model_fingerprint="0" * 64,
        processed_data_fingerprint=OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT,
        design_input_fingerprint=practical_k4_design_input_fingerprint(
            calibration_input
        ),
        source_config_fingerprint=OWNER_APPROVED_SOURCE_CONFIG_FINGERPRINT,
        master_seed=OWNER_APPROVED_MASTER_SEED,
        scientific_version=OWNER_APPROVED_SCIENTIFIC_VERSION,
        fit_scope=ModelFitScope.DESIGN_MAIN.name,
        fit_replicate=0,
        best_restart_index=OWNER_APPROVED_RESTART_INDEX,
        best_seed=OWNER_APPROVED_RESTART_SEED,
        converged=converged,
        iterations=iterations,
        log_likelihood=log_likelihood,
        state_names=OWNER_APPROVED_STATE_NAMES,
        month_ranges=OWNER_APPROVED_MONTH_RANGES,
        generation_policy=practical_historical_vector_policy(),
        **arrays,
    )
    model_fingerprint = _model_fingerprint(provisional)
    with_model = _replace_fingerprints(
        provisional,
        model_fingerprint=model_fingerprint,
        fingerprint="0" * 64,
    )
    return _replace_fingerprints(
        with_model,
        model_fingerprint=model_fingerprint,
        fingerprint=_artifact_fingerprint(with_model),
    )


def _replace_fingerprints(
    value: PracticalK4Artifact,
    *,
    model_fingerprint: str,
    fingerprint: str,
) -> PracticalK4Artifact:
    fields = {
        name: getattr(value, name) for name in PracticalK4Artifact.__dataclass_fields__
    }
    fields["model_fingerprint"] = model_fingerprint
    fields["fingerprint"] = fingerprint
    return PracticalK4Artifact(**fields)


def _validate_design_source(calibration_input: CalibrationInput) -> None:
    if not isinstance(calibration_input, CalibrationInput):
        raise ModelError("owner-approved K=4 requires a calibration input")
    if (
        calibration_input.provenance.processed_data_fingerprint
        != OWNER_APPROVED_PROCESSED_DATA_FINGERPRINT
    ):
        raise IntegrityError("owner-approved K=4 processed data changed")
    design = calibration_input.design
    if (
        len(design.monthly_returns) != OWNER_APPROVED_DESIGN_MONTHS
        or len(design.daily_returns) != OWNER_APPROVED_DESIGN_DAILY_SESSIONS
    ):
        raise IntegrityError("owner-approved K=4 design geometry changed")
    expected_owner_approved_monthly_states(design.monthly_period_ordinals)


def validate_practical_k4_calibration_binding(
    artifact: PracticalK4Artifact,
    calibration_input: CalibrationInput,
) -> CalibrationSlice:
    """Return the exact design pool after binding it to loaded authority."""

    checked = verify_practical_k4_artifact(artifact)
    _validate_design_source(calibration_input)
    design = calibration_input.design
    if (
        practical_k4_design_input_fingerprint(calibration_input)
        != checked.design_input_fingerprint
        or not np.array_equal(
            design.monthly_period_ordinals, checked.monthly_period_ordinals
        )
        or not np.array_equal(design.daily_session_ns, checked.daily_session_ns)
    ):
        raise IntegrityError(
            "calibration design pool differs from practical K=4 authority"
        )
    return design


def _model_fingerprint(value: PracticalK4Artifact) -> str:
    return _sha256_json(
        {
            "schema_id": "prpg-practical-k4-model-v1",
            "best_seed": value.best_seed,
            "log_likelihood": value.log_likelihood,
            "arrays": {
                name: _array_record(getattr(value, name))
                for name in (
                    "start_probabilities",
                    "transition_matrix",
                    "means_standardized",
                    "tied_covariance",
                    "scaler_means",
                    "scaler_standard_deviations",
                    "monthly_source_states",
                    "daily_source_states",
                )
            },
        }
    )


def _artifact_identity(value: PracticalK4Artifact) -> dict[str, object]:
    return {
        "schema_id": value.schema_id,
        "schema_version": value.schema_version,
        "model_fingerprint": value.model_fingerprint,
        "processed_data_fingerprint": value.processed_data_fingerprint,
        "design_input_fingerprint": value.design_input_fingerprint,
        "source_config_fingerprint": value.source_config_fingerprint,
        "master_seed": value.master_seed,
        "scientific_version": value.scientific_version,
        "fit_scope": value.fit_scope,
        "fit_replicate": value.fit_replicate,
        "best_restart_index": value.best_restart_index,
        "best_seed": value.best_seed,
        "converged": value.converged,
        "iterations": value.iterations,
        "log_likelihood": value.log_likelihood,
        "state_names": list(value.state_names),
        "month_ranges": [list(item) for item in value.month_ranges],
        "generation_policy": value.generation_policy.as_dict(),
        "arrays": {name: _array_record(getattr(value, name)) for name in _ARRAY_NAMES},
    }


def _artifact_fingerprint(value: PracticalK4Artifact) -> str:
    return _sha256_json(_artifact_identity(value))


def _array_record(value: np.ndarray[Any, Any]) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _immutable(
    value: np.ndarray[Any, Any], dtype: np.dtype[Any] | type[np.generic] | None = None
) -> np.ndarray[Any, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def _canonical_bytes(value: object) -> bytes:
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


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_fingerprint(value: object, label: str) -> None:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise IntegrityError(f"{label} is invalid")


def _checked_fingerprint(value: object, _label: str) -> bool:
    return isinstance(value, str) and _FINGERPRINT.fullmatch(value) is not None
