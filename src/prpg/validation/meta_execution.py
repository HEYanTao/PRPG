"""Nonauthorizing Purpose-12/13 return-meta materialization infrastructure.

This module deliberately stops before scientific meta-validation.  It creates
the pristine registered return paths once, applies each of the ten non-HMM
fixtures independently, verifies their numerical identities, and records
deterministic fingerprints.  The approved adversary/score implementation is a
separate prerequisite; no object in this module can report a meta pass, a Holm
decision, or production authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from prpg.errors import IntegrityError, ValidationError
from prpg.execution import deterministic_process_map
from prpg.model.g5_generation_authority import (
    G5_DEVELOPMENT_HF_PATH_COUNT,
    G5_DEVELOPMENT_LF_PATH_COUNT,
    G5_QUALIFICATION_HF_PATH_COUNT,
    G5_QUALIFICATION_LF_PATH_COUNT,
)
from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    RNGKey,
    Stage,
    calibration_rng_key,
    uniforms,
)
from prpg.simulation.serial import (
    MONTHS_PER_YEAR,
    SESSIONS_PER_YEAR,
    G5MetaSerialNumericalInput,
    PathFamily,
    materialize_g5_meta_rehearsal_base_path,
    verify_g5_meta_serial_numerical_input,
)
from prpg.validation.meta_validation import (
    G5_DEFECT_REGISTRY,
    RegisteredDefect,
    apply_registered_return_defect,
    perturb_transition_matrix,
)
from prpg.validation.statistics import META_REPLICATES

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Split: TypeAlias = Literal["development", "qualification"]

G5_RETURN_META_INFRASTRUCTURE_SCHEMA_ID: Final = "prpg-g5-return-meta-infrastructure-v1"
G5_RETURN_META_CANONICAL_YEARS: Final = 50
G5_RETURN_META_BLOCKED_PREREQUISITE: Final = (
    "approved exact ridge/logistic/walk-forward adversary implementation"
)
G5_RETURN_META_DEFECT_CLAIMS: Final = tuple(
    item.claim_name
    for item in G5_DEFECT_REGISTRY
    if item.designated_family != "hmm_identification"
)
_EXPECTED_RETURN_CLAIMS: Final = (
    "linear_alpha_sharpe",
    "normalized_covariance",
    "maximum_correlation",
    "one_session_desynchronization",
    "tail_clipping",
    "transition_perturbation",
    "nonlinear_daily",
    "nonlinear_monthly",
    "subsequence_daily",
    "subsequence_monthly",
)
if G5_RETURN_META_DEFECT_CLAIMS != _EXPECTED_RETURN_CLAIMS:  # pragma: no cover
    raise AssertionError("return-meta defect registry changed")
_PURPOSE_13_VARIANT: Final = {
    "linear_alpha_sharpe": 0,
    "normalized_covariance": 1,
    "maximum_correlation": 2,
    "one_session_desynchronization": 3,
    "tail_clipping": 4,
    "transition_perturbation": 5,
    "nonlinear_daily": 8,
    "nonlinear_monthly": 8,
    "subsequence_daily": 9,
    "subsequence_monthly": 9,
}

_RUN_SEAL = object()
_MINTED_RUNS: dict[int, G5ReturnMetaInfrastructureRun] = {}


@dataclass(frozen=True, slots=True)
class G5ReturnMetaGeometry:
    """Path-cluster geometry; only one exact tuple is registered canonical."""

    development_lf_paths: int
    development_hf_paths: int
    qualification_lf_paths: int
    qualification_hf_paths: int
    years: int

    @property
    def canonical(self) -> bool:
        return self == G5_RETURN_META_CANONICAL_GEOMETRY

    @property
    def base_rows_per_replicate(self) -> int:
        lf = self.development_lf_paths + self.qualification_lf_paths
        hf = self.development_hf_paths + self.qualification_hf_paths
        return self.years * (lf * MONTHS_PER_YEAR + hf * SESSIONS_PER_YEAR)


G5_RETURN_META_CANONICAL_GEOMETRY: Final = G5ReturnMetaGeometry(
    development_lf_paths=G5_DEVELOPMENT_LF_PATH_COUNT,
    development_hf_paths=G5_DEVELOPMENT_HF_PATH_COUNT,
    qualification_lf_paths=G5_QUALIFICATION_LF_PATH_COUNT,
    qualification_hf_paths=G5_QUALIFICATION_HF_PATH_COUNT,
    years=G5_RETURN_META_CANONICAL_YEARS,
)


@dataclass(frozen=True, slots=True)
class G5ReturnMetaFixtureReceipt:
    """One applied-alone Purpose-13 fixture identity without a score."""

    claim_name: str
    purpose_13_variant: int
    transform_id: str
    designated_family: str
    oracle_values: tuple[tuple[str, float], ...]
    transformed_bytes_fingerprint: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5ReturnMetaReplicateReceipt:
    """One pristine Purpose-12 replicate and all ten fixture identities."""

    replicate: int
    purpose_12_key_fingerprints: tuple[tuple[str, str], ...]
    purpose_13_key_fingerprints: tuple[tuple[str, str], ...]
    pristine_bytes_fingerprint: str
    fixture_receipts: tuple[G5ReturnMetaFixtureReceipt, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True, init=False)
class G5ReturnMetaInfrastructureRun:
    """Sealed infrastructure evidence that is never meta authority."""

    schema_id: str
    generation_authorized: Literal[False]
    scientific_meta_ready: Literal[False]
    blocked_prerequisite: str
    numerical_input_fingerprint: str
    geometry: G5ReturnMetaGeometry
    replicate_count: int
    exact_registered_replicates: bool
    canonical_geometry: bool
    execution_workers: int
    elapsed_seconds: float
    replicate_receipts: tuple[G5ReturnMetaReplicateReceipt, ...]
    fingerprint: str
    _seal: object = field(repr=False, compare=False)
    _owner_id: int = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("return-meta infrastructure is created only by its runner")


@dataclass(frozen=True, slots=True)
class G5ReturnMetaResourceProjection:
    """Linear same-host projection for infrastructure work only."""

    pilot_run_fingerprint: str
    pilot_replicates: int
    pilot_workers: int
    target_replicates: int
    target_workers: int
    projected_seconds: float
    projected_hours: float
    estimated_peak_per_worker_bytes: int
    estimated_all_workers_bytes: int
    physical_memory_bytes: int
    estimated_memory_fraction: float
    memory_passed: bool


@dataclass(frozen=True, slots=True)
class _MetaPath:
    split: Split
    family: PathFamily
    path_rank: int
    path_id: str
    log_returns: FloatArray
    target_monthly_states: IntArray
    target_states: IntArray
    source_indices: IntArray
    state_uniforms: FloatArray
    restart_uniforms: FloatArray
    source_rank_uniforms: FloatArray
    path_fingerprint: str


@dataclass(frozen=True, slots=True)
class _MetaWork:
    numerical_input: G5MetaSerialNumericalInput
    geometry: G5ReturnMetaGeometry
    replicate: int


def build_g5_return_meta_pilot_geometry(
    *,
    development_lf_paths: int,
    development_hf_paths: int,
    qualification_lf_paths: int,
    qualification_hf_paths: int,
    years: int,
) -> G5ReturnMetaGeometry:
    """Build a visibly noncanonical reduced geometry for rehearsal/projection."""

    geometry = _validated_geometry(
        G5ReturnMetaGeometry(
            development_lf_paths=development_lf_paths,
            development_hf_paths=development_hf_paths,
            qualification_lf_paths=qualification_lf_paths,
            qualification_hf_paths=qualification_hf_paths,
            years=years,
        )
    )
    if geometry.canonical:
        raise ValidationError("pilot geometry must be explicitly noncanonical")
    return geometry


def run_g5_return_meta_infrastructure(
    numerical_input: G5MetaSerialNumericalInput,
    *,
    replicate_count: int,
    workers: int,
    geometry: G5ReturnMetaGeometry = G5_RETURN_META_CANONICAL_GEOMETRY,
) -> G5ReturnMetaInfrastructureRun:
    """Materialize Purpose-12/13 bytes without producing scientific decisions."""

    checked = verify_g5_meta_serial_numerical_input(numerical_input)
    checked_geometry = _validated_geometry(geometry)
    count = _bounded_replicates(replicate_count)
    worker_count = _workers(workers, count)
    if checked_geometry.years > checked.configured_years:
        raise ValidationError("meta geometry exceeds configured serial horizon")
    jobs = tuple(
        _MetaWork(checked, checked_geometry, replicate) for replicate in range(count)
    )
    started = time.monotonic()
    receipts = deterministic_process_map(
        _materialize_replicate, jobs, workers=worker_count, chunksize=1
    )
    elapsed = time.monotonic() - started
    if tuple(item.replicate for item in receipts) != tuple(range(count)):
        raise IntegrityError("return-meta replicate ordering changed")
    return _mint_run(
        numerical_input_fingerprint=checked.fingerprint,
        geometry=checked_geometry,
        replicate_count=count,
        execution_workers=worker_count,
        elapsed_seconds=elapsed,
        receipts=receipts,
    )


def verify_g5_return_meta_infrastructure_run(
    value: object,
) -> G5ReturnMetaInfrastructureRun:
    """Reverify an original process-sealed infrastructure run."""

    if (
        type(value) is not G5ReturnMetaInfrastructureRun
        or _MINTED_RUNS.get(id(value)) is not value
    ):
        raise IntegrityError("return-meta infrastructure was not minted by its runner")
    assert isinstance(value, G5ReturnMetaInfrastructureRun)
    if (
        value._seal is not _RUN_SEAL
        or value._owner_id != id(value)
        or value.schema_id != G5_RETURN_META_INFRASTRUCTURE_SCHEMA_ID
        or value.generation_authorized is not False
        or value.scientific_meta_ready is not False
        or value.blocked_prerequisite != G5_RETURN_META_BLOCKED_PREREQUISITE
        or value.replicate_count != len(value.replicate_receipts)
        or value.exact_registered_replicates
        is not (value.replicate_count == META_REPLICATES)
        or value.canonical_geometry is not value.geometry.canonical
        or not math.isfinite(value.elapsed_seconds)
        or value.elapsed_seconds < 0.0
    ):
        raise IntegrityError("return-meta infrastructure identity changed")
    _required_sha256(value.numerical_input_fingerprint, "meta numerical input")
    _workers(value.execution_workers, value.replicate_count)
    _validated_geometry(value.geometry)
    for expected, receipt in enumerate(value.replicate_receipts):
        if receipt.replicate != expected:
            raise IntegrityError("return-meta receipt sequence changed")
        _verify_replicate_receipt(receipt)
    expected_fingerprint = _run_fingerprint(
        numerical_input_fingerprint=value.numerical_input_fingerprint,
        geometry=value.geometry,
        replicate_count=value.replicate_count,
        receipts=value.replicate_receipts,
    )
    if value.fingerprint != expected_fingerprint:
        raise IntegrityError("return-meta infrastructure fingerprint changed")
    return value


def project_g5_return_meta_resources(
    pilot: G5ReturnMetaInfrastructureRun,
    *,
    target_workers: int,
    physical_memory_bytes: int,
) -> G5ReturnMetaResourceProjection:
    """Project exact infrastructure cost from a reduced same-host pilot."""

    checked = verify_g5_return_meta_infrastructure_run(pilot)
    if checked.geometry.canonical or checked.replicate_count >= META_REPLICATES:
        raise ValidationError("resource projection requires a reduced pilot")
    workers = _positive_int(target_workers, "target workers")
    memory = _positive_int(physical_memory_bytes, "physical memory bytes")
    if checked.elapsed_seconds <= 0.0:
        raise ValidationError("resource pilot must record positive elapsed work")
    row_ratio = (
        G5_RETURN_META_CANONICAL_GEOMETRY.base_rows_per_replicate
        / checked.geometry.base_rows_per_replicate
    )
    projected = (
        checked.elapsed_seconds
        * (META_REPLICATES / checked.replicate_count)
        * row_ratio
        * (checked.execution_workers / workers)
    )
    canonical_rows = G5_RETURN_META_CANONICAL_GEOMETRY.base_rows_per_replicate
    per_worker = canonical_rows * 224 + 256 * 1024**2
    all_workers = per_worker * workers
    fraction = all_workers / memory
    return G5ReturnMetaResourceProjection(
        pilot_run_fingerprint=checked.fingerprint,
        pilot_replicates=checked.replicate_count,
        pilot_workers=checked.execution_workers,
        target_replicates=META_REPLICATES,
        target_workers=workers,
        projected_seconds=projected,
        projected_hours=projected / 3600.0,
        estimated_peak_per_worker_bytes=per_worker,
        estimated_all_workers_bytes=all_workers,
        physical_memory_bytes=memory,
        estimated_memory_fraction=fraction,
        memory_passed=fraction <= 0.70,
    )


def _materialize_replicate(work: _MetaWork) -> G5ReturnMetaReplicateReceipt:
    checked = verify_g5_meta_serial_numerical_input(work.numerical_input)
    geometry = _validated_geometry(work.geometry)
    replicate = _replicate(work.replicate)
    paths: dict[PathFamily, tuple[_MetaPath, ...]] = {}
    key_fingerprints: dict[str, str] = {}
    for family in ("LF", "HF"):
        family_paths, family_keys = _materialize_family(
            checked, geometry, replicate, family
        )
        paths[family] = family_paths
        key_fingerprints.update(family_keys)

    lf = paths["LF"]
    hf = paths["HF"]
    pristine = _path_collection_fingerprint((*lf, *hf))
    quarter_indices, ranking_keys = _registered_quarter_indices(
        checked, geometry, replicate
    )
    receipts = tuple(
        _apply_fixture(
            defect,
            checked,
            geometry,
            lf,
            hf,
            quarter_indices=quarter_indices,
        )
        for defect in G5_DEFECT_REGISTRY
        if defect.designated_family != "hmm_identification"
    )
    identity = {
        "replicate": replicate,
        "purpose_12_key_fingerprints": key_fingerprints,
        "purpose_13_key_fingerprints": ranking_keys,
        "pristine_bytes_fingerprint": pristine,
        "fixture_fingerprints": [item.fingerprint for item in receipts],
    }
    result = G5ReturnMetaReplicateReceipt(
        replicate=replicate,
        purpose_12_key_fingerprints=tuple(sorted(key_fingerprints.items())),
        purpose_13_key_fingerprints=tuple(sorted(ranking_keys.items())),
        pristine_bytes_fingerprint=pristine,
        fixture_receipts=receipts,
        fingerprint=_fingerprint(identity),
    )
    _verify_replicate_receipt(result)
    return result


def _materialize_family(
    numerical_input: G5MetaSerialNumericalInput,
    geometry: G5ReturnMetaGeometry,
    replicate: int,
    family: PathFamily,
) -> tuple[tuple[_MetaPath, ...], dict[str, str]]:
    rng_family = Family.LF if family == "LF" else Family.HF
    development_count = (
        geometry.development_lf_paths
        if family == "LF"
        else geometry.development_hf_paths
    )
    qualification_count = (
        geometry.qualification_lf_paths
        if family == "LF"
        else geometry.qualification_hf_paths
    )
    total_paths = development_count + qualification_count
    monthly_rows = geometry.years * MONTHS_PER_YEAR
    base_rows = monthly_rows if family == "LF" else geometry.years * SESSIONS_PER_YEAR
    keys = {
        stage: _purpose_12_key(numerical_input, replicate, rng_family, stage)
        for stage in (
            Stage.STATE_CHAIN,
            Stage.RESTART_UNIFORMS,
            Stage.SOURCE_RANK_UNIFORMS,
        )
    }
    state_draws = uniforms(keys[Stage.STATE_CHAIN], total_paths * monthly_rows)
    restart_draws = uniforms(keys[Stage.RESTART_UNIFORMS], total_paths * base_rows)
    source_draws = uniforms(keys[Stage.SOURCE_RANK_UNIFORMS], total_paths * base_rows)
    paths: list[_MetaPath] = []
    for global_rank in range(total_paths):
        split: Split = (
            "development" if global_rank < development_count else "qualification"
        )
        split_rank = (
            global_rank + 1
            if split == "development"
            else global_rank - development_count + 1
        )
        state_start = global_rank * monthly_rows
        base_start = global_rank * base_rows
        state = state_draws[state_start : state_start + monthly_rows]
        restart = restart_draws[base_start : base_start + base_rows]
        source = source_draws[base_start : base_start + base_rows]
        materialized = materialize_g5_meta_rehearsal_base_path(
            numerical_input,
            family=family,
            path_entity=global_rank + 1,
            years=geometry.years,
            state_uniforms=state,
            restart_uniforms=restart,
            source_rank_uniforms=source,
        )
        prefix = "D" if split == "development" else "Q"
        paths.append(
            _MetaPath(
                split=split,
                family=family,
                path_rank=split_rank,
                path_id=f"{prefix}{family}{split_rank:06d}",
                log_returns=materialized.log_returns,
                target_monthly_states=materialized.target_monthly_states,
                target_states=materialized.target_states,
                source_indices=materialized.source_indices,
                state_uniforms=state,
                restart_uniforms=restart,
                source_rank_uniforms=source,
                path_fingerprint=materialized.fingerprint,
            )
        )
    key_fingerprints = {
        f"{family}.{stage.name.lower()}": _rng_key_fingerprint(key)
        for stage, key in keys.items()
    }
    return tuple(paths), key_fingerprints


def _apply_fixture(
    defect: RegisteredDefect,
    numerical_input: G5MetaSerialNumericalInput,
    geometry: G5ReturnMetaGeometry,
    lf_paths: tuple[_MetaPath, ...],
    hf_paths: tuple[_MetaPath, ...],
    *,
    quarter_indices: dict[tuple[PathFamily, Split], tuple[int, ...]],
) -> G5ReturnMetaFixtureReceipt:
    claim = defect.claim_name
    if claim == "transition_perturbation":
        return _transition_fixture(
            defect, numerical_input, geometry, lf_paths, hf_paths
        )

    transformed_fingerprints: dict[str, str] = {}
    oracles: list[tuple[str, float]] = []
    frequencies: tuple[PathFamily, ...]
    if claim in {
        "linear_alpha_sharpe",
        "normalized_covariance",
        "maximum_correlation",
        "tail_clipping",
    }:
        frequencies = ("LF", "HF")
    elif claim in {
        "one_session_desynchronization",
        "nonlinear_daily",
        "subsequence_daily",
    }:
        frequencies = ("HF",)
    elif claim in {"nonlinear_monthly", "subsequence_monthly"}:
        frequencies = ("LF",)
    else:  # pragma: no cover - registry assertion closes this branch
        raise AssertionError(f"unhandled return defect {claim}")

    for family in frequencies:
        paths = lf_paths if family == "LF" else hf_paths
        development, qualification = _split_return_arrays(paths)
        kwargs: dict[str, object] = {}
        if claim.startswith("subsequence"):
            kwargs = {
                "selected_development_paths": quarter_indices[(family, "development")],
                "selected_qualification_paths": quarter_indices[
                    (family, "qualification")
                ],
            }
        bundle = apply_registered_return_defect(
            claim,
            development,
            qualification,
            annualization=12 if family == "LF" else 252,
            **kwargs,  # type: ignore[arg-type]
        )
        transformed_fingerprints[family] = bundle.fingerprint
        oracles.append((f"{family}.{bundle.oracle_name}", bundle.oracle_value))
    unaffected = tuple(
        path.path_fingerprint
        for family, paths in (("LF", lf_paths), ("HF", hf_paths))
        if family not in frequencies
        for path in paths
    )
    transformed = _fingerprint(
        {
            "claim_name": claim,
            "transformed_family_fingerprints": transformed_fingerprints,
            "unaffected_path_fingerprints": unaffected,
        }
    )
    return _fixture_receipt(defect, tuple(oracles), transformed)


def _transition_fixture(
    defect: RegisteredDefect,
    numerical_input: G5MetaSerialNumericalInput,
    geometry: G5ReturnMetaGeometry,
    lf_paths: tuple[_MetaPath, ...],
    hf_paths: tuple[_MetaPath, ...],
) -> G5ReturnMetaFixtureReceipt:
    original = numerical_input.transition_matrix
    changed = perturb_transition_matrix(original)
    expected = original.copy()
    expected[np.diag_indices_from(expected)] += 0.10
    expected /= 1.10
    oracle_error = float(np.max(np.abs(changed - expected)))
    if oracle_error > 1e-12:
        raise ValidationError("transition P-prime numeric oracle failed")
    fingerprints: list[str] = []
    for path in (*lf_paths, *hf_paths):
        materialized = materialize_g5_meta_rehearsal_base_path(
            numerical_input,
            family=path.family,
            path_entity=(
                path.path_rank
                if path.split == "development"
                else (
                    geometry.development_lf_paths
                    if path.family == "LF"
                    else geometry.development_hf_paths
                )
                + path.path_rank
            ),
            years=geometry.years,
            state_uniforms=path.state_uniforms,
            restart_uniforms=path.restart_uniforms,
            source_rank_uniforms=path.source_rank_uniforms,
            transition_matrix=changed,
        )
        fingerprints.append(materialized.fingerprint)
    transformed = _fingerprint(
        {
            "transition_sha256": _array_sha256(changed),
            "regenerated_path_fingerprints": fingerprints,
        }
    )
    return _fixture_receipt(
        defect,
        (("P_prime_maximum_formula_error", oracle_error),),
        transformed,
    )


def _registered_quarter_indices(
    numerical_input: G5MetaSerialNumericalInput,
    geometry: G5ReturnMetaGeometry,
    replicate: int,
) -> tuple[
    dict[tuple[PathFamily, Split], tuple[int, ...]],
    dict[str, str],
]:
    selections: dict[tuple[PathFamily, Split], tuple[int, ...]] = {}
    fingerprints: dict[str, str] = {}
    groups: tuple[tuple[PathFamily, Family, int, int], ...] = (
        (
            "LF",
            Family.LF,
            geometry.development_lf_paths,
            geometry.qualification_lf_paths,
        ),
        (
            "HF",
            Family.HF,
            geometry.development_hf_paths,
            geometry.qualification_hf_paths,
        ),
    )
    for family, rng_family, development_count, qualification_count in groups:
        key = calibration_rng_key(
            master_seed=numerical_input.master_seed,
            scientific_version=numerical_input.scientific_version,
            purpose=CalibrationPurpose.META_POWER,
            artifact=CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
            variant=9,
            replicate=replicate,
            domain=Domain.THRESHOLD_CALIBRATION,
            family=rng_family,
            stage=Stage.VALIDATION_SAMPLING_OR_STRATEGY,
        )
        draws = uniforms(key, development_count + qualification_count)
        development_draws = draws[:development_count]
        qualification_draws = draws[development_count:]
        selections[(family, "development")] = _lowest_quarter(development_draws)
        selections[(family, "qualification")] = _lowest_quarter(qualification_draws)
        fingerprints[f"{family}.recipient_ranking"] = _rng_key_fingerprint(key)
    return selections, fingerprints


def _purpose_12_key(
    numerical_input: G5MetaSerialNumericalInput,
    replicate: int,
    family: Family,
    stage: Stage,
) -> RNGKey:
    return calibration_rng_key(
        master_seed=numerical_input.master_seed,
        scientific_version=numerical_input.scientific_version,
        purpose=CalibrationPurpose.META_TYPE_I,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        variant=0,
        replicate=replicate,
        domain=Domain.THRESHOLD_CALIBRATION,
        family=family,
        stage=stage,
    )


def _fixture_receipt(
    defect: RegisteredDefect,
    oracle_values: tuple[tuple[str, float], ...],
    transformed_bytes_fingerprint: str,
) -> G5ReturnMetaFixtureReceipt:
    transformed = _required_sha256(
        transformed_bytes_fingerprint, "transformed fixture bytes"
    )
    if not oracle_values or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for name, value in oracle_values
    ):
        raise ValidationError("return-meta fixture oracle is invalid")
    checked_oracles = tuple((name, float(value)) for name, value in oracle_values)
    identity = {
        "claim_name": defect.claim_name,
        "purpose_13_variant": _PURPOSE_13_VARIANT[defect.claim_name],
        "transform_id": defect.transform_id,
        "designated_family": defect.designated_family,
        "oracle_values": dict(checked_oracles),
        "transformed_bytes_fingerprint": transformed,
    }
    return G5ReturnMetaFixtureReceipt(
        claim_name=defect.claim_name,
        purpose_13_variant=_PURPOSE_13_VARIANT[defect.claim_name],
        transform_id=defect.transform_id,
        designated_family=defect.designated_family,
        oracle_values=checked_oracles,
        transformed_bytes_fingerprint=transformed,
        fingerprint=_fingerprint(identity),
    )


def _split_return_arrays(
    paths: tuple[_MetaPath, ...],
) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    development = tuple(
        path.log_returns for path in paths if path.split == "development"
    )
    qualification = tuple(
        path.log_returns for path in paths if path.split == "qualification"
    )
    if not development or not qualification:
        raise ValidationError("return-meta split is empty")
    return development, qualification


def _lowest_quarter(draws: FloatArray) -> tuple[int, ...]:
    count = int(draws.size)
    if count % 4 != 0 or count <= 0:
        raise ValidationError("recipient-ranking count must be positive/divisible by 4")
    if not bool(np.isfinite(draws).all()) or bool(
        ((draws < 0.0) | (draws >= 1.0)).any()
    ):
        raise ValidationError("recipient-ranking uniforms are invalid")
    # lexsort uses the last key first: uniform primary, zero-based path rank
    # secondary.  The returned indices are then canonicalized for the fixture.
    order = np.lexsort((np.arange(count, dtype=np.int64), draws))
    return tuple(sorted(int(value) for value in order[: count // 4]))


def _path_collection_fingerprint(paths: tuple[_MetaPath, ...]) -> str:
    return _fingerprint(
        {
            "path_ids": [path.path_id for path in paths],
            "path_fingerprints": [path.path_fingerprint for path in paths],
            "return_sha256": [_array_sha256(path.log_returns) for path in paths],
            "target_monthly_state_sha256": [
                _array_sha256(path.target_monthly_states) for path in paths
            ],
            "target_state_sha256": [
                _array_sha256(path.target_states) for path in paths
            ],
            "source_index_sha256": [
                _array_sha256(path.source_indices) for path in paths
            ],
        }
    )


def _rng_key_fingerprint(key: RNGKey) -> str:
    return _fingerprint(
        {
            "master_seed": key.master_seed,
            "spawn_key": list(key.spawn_key),
        }
    )


def _mint_run(
    *,
    numerical_input_fingerprint: str,
    geometry: G5ReturnMetaGeometry,
    replicate_count: int,
    execution_workers: int,
    elapsed_seconds: float,
    receipts: tuple[G5ReturnMetaReplicateReceipt, ...],
) -> G5ReturnMetaInfrastructureRun:
    fingerprint = _run_fingerprint(
        numerical_input_fingerprint=numerical_input_fingerprint,
        geometry=geometry,
        replicate_count=replicate_count,
        receipts=receipts,
    )
    result = object.__new__(G5ReturnMetaInfrastructureRun)
    for name, value in (
        ("schema_id", G5_RETURN_META_INFRASTRUCTURE_SCHEMA_ID),
        ("generation_authorized", False),
        ("scientific_meta_ready", False),
        ("blocked_prerequisite", G5_RETURN_META_BLOCKED_PREREQUISITE),
        ("numerical_input_fingerprint", numerical_input_fingerprint),
        ("geometry", geometry),
        ("replicate_count", replicate_count),
        ("exact_registered_replicates", replicate_count == META_REPLICATES),
        ("canonical_geometry", geometry.canonical),
        ("execution_workers", execution_workers),
        ("elapsed_seconds", elapsed_seconds),
        ("replicate_receipts", receipts),
        ("fingerprint", fingerprint),
        ("_seal", _RUN_SEAL),
    ):
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_owner_id", id(result))
    _MINTED_RUNS[id(result)] = result
    return verify_g5_return_meta_infrastructure_run(result)


def _run_fingerprint(
    *,
    numerical_input_fingerprint: str,
    geometry: G5ReturnMetaGeometry,
    replicate_count: int,
    receipts: tuple[G5ReturnMetaReplicateReceipt, ...],
) -> str:
    # Worker count and elapsed time are operational evidence and intentionally
    # excluded, so one-worker and nine-worker scientific bytes compare exactly.
    return _fingerprint(
        {
            "schema_id": G5_RETURN_META_INFRASTRUCTURE_SCHEMA_ID,
            "generation_authorized": False,
            "scientific_meta_ready": False,
            "blocked_prerequisite": G5_RETURN_META_BLOCKED_PREREQUISITE,
            "numerical_input_fingerprint": _required_sha256(
                numerical_input_fingerprint, "meta numerical input"
            ),
            "geometry": _geometry_identity(geometry),
            "replicate_count": replicate_count,
            "replicate_fingerprints": [item.fingerprint for item in receipts],
        }
    )


def _verify_replicate_receipt(value: G5ReturnMetaReplicateReceipt) -> None:
    if not isinstance(value, G5ReturnMetaReplicateReceipt):
        raise IntegrityError("return-meta replicate receipt is untyped")
    replicate = _replicate(value.replicate)
    if tuple(item.claim_name for item in value.fixture_receipts) != (
        G5_RETURN_META_DEFECT_CLAIMS
    ):
        raise IntegrityError("return-meta fixture registry changed")
    if {name for name, _ in value.purpose_12_key_fingerprints} != {
        "HF.restart_uniforms",
        "HF.source_rank_uniforms",
        "HF.state_chain",
        "LF.restart_uniforms",
        "LF.source_rank_uniforms",
        "LF.state_chain",
    } or {name for name, _ in value.purpose_13_key_fingerprints} != {
        "HF.recipient_ranking",
        "LF.recipient_ranking",
    }:
        raise IntegrityError("return-meta RNG key registry changed")
    for _name, fingerprint in (
        *value.purpose_12_key_fingerprints,
        *value.purpose_13_key_fingerprints,
    ):
        _required_sha256(fingerprint, "return-meta RNG key")
    pristine = _required_sha256(
        value.pristine_bytes_fingerprint, "pristine return-meta bytes"
    )
    for receipt, registered in zip(
        value.fixture_receipts,
        (
            item
            for item in G5_DEFECT_REGISTRY
            if item.designated_family != "hmm_identification"
        ),
        strict=True,
    ):
        _verify_fixture_receipt(receipt, registered)
    identity = {
        "replicate": replicate,
        "purpose_12_key_fingerprints": dict(value.purpose_12_key_fingerprints),
        "purpose_13_key_fingerprints": dict(value.purpose_13_key_fingerprints),
        "pristine_bytes_fingerprint": pristine,
        "fixture_fingerprints": [item.fingerprint for item in value.fixture_receipts],
    }
    if value.fingerprint != _fingerprint(identity):
        raise IntegrityError("return-meta replicate fingerprint changed")


def _verify_fixture_receipt(
    value: G5ReturnMetaFixtureReceipt, registered: RegisteredDefect
) -> None:
    if (
        not isinstance(value, G5ReturnMetaFixtureReceipt)
        or value.claim_name != registered.claim_name
        or value.purpose_13_variant != _PURPOSE_13_VARIANT[registered.claim_name]
        or value.transform_id != registered.transform_id
        or value.designated_family != registered.designated_family
        or not value.oracle_values
    ):
        raise IntegrityError("return-meta fixture receipt changed")
    transformed = _required_sha256(
        value.transformed_bytes_fingerprint, "transformed fixture bytes"
    )
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(observed, float)
        or not math.isfinite(observed)
        for name, observed in value.oracle_values
    ):
        raise IntegrityError("return-meta fixture oracle changed")
    identity = {
        "claim_name": value.claim_name,
        "purpose_13_variant": value.purpose_13_variant,
        "transform_id": value.transform_id,
        "designated_family": value.designated_family,
        "oracle_values": dict(value.oracle_values),
        "transformed_bytes_fingerprint": transformed,
    }
    if value.fingerprint != _fingerprint(identity):
        raise IntegrityError("return-meta fixture fingerprint changed")


def _validated_geometry(value: object) -> G5ReturnMetaGeometry:
    if not isinstance(value, G5ReturnMetaGeometry):
        raise ValidationError("return-meta geometry must be typed")
    counts = (
        value.development_lf_paths,
        value.development_hf_paths,
        value.qualification_lf_paths,
        value.qualification_hf_paths,
    )
    maxima = (
        G5_DEVELOPMENT_LF_PATH_COUNT,
        G5_DEVELOPMENT_HF_PATH_COUNT,
        G5_QUALIFICATION_LF_PATH_COUNT,
        G5_QUALIFICATION_HF_PATH_COUNT,
    )
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count > maximum
        or count % 4 != 0
        for count, maximum in zip(counts, maxima, strict=True)
    ):
        raise ValidationError(
            "return-meta path counts must be positive, bounded, and divisible by four"
        )
    years = _positive_int(value.years, "return-meta years")
    if years > G5_RETURN_META_CANONICAL_YEARS or years < 6:
        raise ValidationError("return-meta years must be in [6,50]")
    return value


def _geometry_identity(value: G5ReturnMetaGeometry) -> dict[str, int]:
    checked = _validated_geometry(value)
    return {
        "development_lf_paths": checked.development_lf_paths,
        "development_hf_paths": checked.development_hf_paths,
        "qualification_lf_paths": checked.qualification_lf_paths,
        "qualification_hf_paths": checked.qualification_hf_paths,
        "years": checked.years,
    }


def _bounded_replicates(value: object) -> int:
    count = _positive_int(value, "return-meta replicate count")
    if count > META_REPLICATES:
        raise ValidationError("return-meta replicate count exceeds 1,000")
    return count


def _workers(value: object, replicate_count: int) -> int:
    workers = _positive_int(value, "return-meta workers")
    if workers > replicate_count:
        raise ValidationError("return-meta workers exceed replicate count")
    return workers


def _replicate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1000:
        raise ValidationError("return-meta replicate must be in [0,999]")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} fingerprint is invalid")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValidationError(f"{label} fingerprint is invalid") from error
    return value


def _array_sha256(value: NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _fingerprint(value: object) -> str:
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
        ).encode("utf-8")
    ).hexdigest()
