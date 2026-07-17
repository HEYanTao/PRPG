"""Bounded, nonauthorizing G4 end-to-end reference fixture.

This module deliberately proves only the observable vertical slice needed
before G5: the registered G3 candidates can drive the production serial
numerical core, all five return frequencies can be serialized with the shared
CSV grammar, and the resulting files satisfy their structural compounding
identities.  It does not mint G5 authority, production paths, group commits,
release state, or crash-recovery machinery.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from prpg.config import PRPGConfig
from prpg.data.calendar import WEEK_BOUNDARIES, WEEKS_PER_YEAR
from prpg.errors import GenerationError, IntegrityError, ModelError
from prpg.model.g3_evidence_store import (
    LoadedG3EvidenceAuthority,
    verify_loaded_g3_evidence,
)
from prpg.model.g3_preflight import (
    CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
    CANONICAL_CONFIG_FINGERPRINT,
    CANONICAL_PROCESSED_DATA_FINGERPRINT,
)
from prpg.model.input import CalibrationInput
from prpg.model.scientific_policy import (
    SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
    ScientificCandidatePolicySet,
    ScientificGenerationPolicy,
    candidate_policy_set_from_mapping,
    generation_policy_from_mapping,
)
from prpg.simulation.aggregation import AGGREGATION_VERSION
from prpg.simulation.rng import RNG_CONTRACT_VERSION, Domain
from prpg.simulation.serial import (
    ASSET_COUNT,
    MONTHS_PER_YEAR,
    SERIAL_GENERATOR_VERSION,
    SESSIONS_PER_YEAR,
    G4ReferenceBasePath,
    build_g4_reference_generation_input,
    generate_g4_reference_base_path,
    is_g4_reference_base_path,
)
from prpg.storage.csv_v1 import (
    CSV_HEADER,
    CSV_HEADER_BYTES,
    CSV_SCHEMA_VERSION,
    CSV_SERIALIZER_VERSION,
    encode_csv_row,
    format_csv_float64,
)

G4_REFERENCE_SCHEMA_ID: Final = "prpg-g4-noncanonical-reference-v1"
G4_REFERENCE_SCHEMA_VERSION: Final = 1
G4_REFERENCE_CONTRACT_VERSION: Final = "g4-bounded-reference-fixture-v1"
G4_REFERENCE_REPORT_NAME: Final = "g4-reference-report.json"
G4_REFERENCE_LF_PATHS: Final = 4
G4_REFERENCE_HF_PATHS: Final = 2
G4_REFERENCE_YEARS: Final = 2
G4_REFERENCE_LOG_SUM_TOLERANCE: Final = 1e-12
G4_REFERENCE_SIMPLE_RETURN_TOLERANCE: Final = 1e-10

ReferenceFrequency: TypeAlias = Literal[
    "monthly", "quarterly", "annual", "daily", "weekly"
]
ReferenceFamily: TypeAlias = Literal["LF", "HF"]
FloatArray: TypeAlias = npt.NDArray[np.float64]

_FREQUENCIES: Final[Mapping[ReferenceFrequency, tuple[ReferenceFamily, int]]] = {
    "monthly": ("LF", 12),
    "quarterly": ("LF", 4),
    "annual": ("LF", 1),
    "daily": ("HF", 252),
    "weekly": ("HF", 52),
}
_POLICY_ID = re.compile(r"[a-z0-9_]+\Z")


@dataclass(frozen=True, slots=True)
class G4ReferenceFileRecord:
    """One reference CSV's exact content and source-path identity."""

    policy_id: str
    alpha_policy: str
    selected_neutral_lambda: float | None
    family: ReferenceFamily
    frequency: ReferenceFrequency
    relative_path: str
    path_count: int
    years: int
    rows: int
    bytes: int
    sha256: str
    source_path_fingerprints: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "alpha_policy": self.alpha_policy,
            "selected_neutral_lambda": self.selected_neutral_lambda,
            "family": self.family,
            "frequency": self.frequency,
            "relative_path": self.relative_path,
            "path_count": self.path_count,
            "years": self.years,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "source_path_fingerprints": list(self.source_path_fingerprints),
        }


@dataclass(frozen=True, slots=True)
class G4StructuralValidation:
    """Outcome-only structural validation of the emitted reference CSVs."""

    passed: Literal[True]
    files: int
    rows: int
    maximum_log_sum_abs_error: float
    maximum_simple_return_roundtrip_abs_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "files": self.files,
            "rows": self.rows,
            "maximum_log_sum_abs_error": self.maximum_log_sum_abs_error,
            "maximum_simple_return_roundtrip_abs_error": (
                self.maximum_simple_return_roundtrip_abs_error
            ),
            "log_sum_abs_tolerance": G4_REFERENCE_LOG_SUM_TOLERANCE,
            "simple_return_roundtrip_abs_tolerance": (
                G4_REFERENCE_SIMPLE_RETURN_TOLERANCE
            ),
        }


@dataclass(frozen=True, slots=True)
class G4ReferenceReport:
    """Simple final report that can never represent production authority."""

    g4_passed: ClassVar[Literal[True]] = True
    canonical_authority: ClassVar[Literal[False]] = False
    generation_authorized: ClassVar[Literal[False]] = False
    releasable: ClassVar[Literal[False]] = False

    fingerprint: str
    sha256: str
    path: Path
    identity: Mapping[str, Any]
    files: tuple[G4ReferenceFileRecord, ...]
    structural_validation: G4StructuralValidation


def run_g4_reference_fixture(
    *,
    g3_evidence: LoadedG3EvidenceAuthority,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
    output_root: str | Path,
) -> G4ReferenceReport:
    """Run the fixed 5-policy, 4-LF/2-HF, two-year offline fixture."""

    evidence = verify_loaded_g3_evidence(g3_evidence)
    _validate_canonical_inputs(evidence, config, calibration_input)
    root = _create_isolated_output_root(output_root)
    policies = _registered_reference_policies(evidence)

    file_records: list[G4ReferenceFileRecord] = []
    policy_records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    for policy_id, policy in policies:
        scope_fingerprint = _sha256_json(
            {
                "contract_version": G4_REFERENCE_CONTRACT_VERSION,
                "g3_evidence_fingerprint": evidence.reference.fingerprint,
                "config_fingerprint": config.fingerprint(),
                "calibration_input_fingerprint": calibration_input.fingerprint,
                "policy": policy.as_dict(),
            }
        )
        generation_input = build_g4_reference_generation_input(
            g3_evidence=evidence,
            generation_policy=policy,
            config=config,
            calibration_input=calibration_input,
            reference_scope_fingerprint=scope_fingerprint,
        )
        lf_paths = tuple(
            generate_g4_reference_base_path(
                generation_input,
                family="LF",
                path_entity=entity,
                years=G4_REFERENCE_YEARS,
            )
            for entity in range(1, G4_REFERENCE_LF_PATHS + 1)
        )
        hf_paths = tuple(
            generate_g4_reference_base_path(
                generation_input,
                family="HF",
                path_entity=entity,
                years=G4_REFERENCE_YEARS,
            )
            for entity in range(1, G4_REFERENCE_HF_PATHS + 1)
        )
        if not all(is_g4_reference_base_path(item) for item in (*lf_paths, *hf_paths)):
            raise IntegrityError("G4 reference runner received an invalid base path")
        quarterly = tuple(_quarterly_values(item) for item in lf_paths)
        annual = tuple(_annual_values(item) for item in lf_paths)
        weekly = tuple(_weekly_values(item) for item in hf_paths)
        policy_files = (
            _write_reference_csv(root, policy_id, policy, "monthly", lf_paths),
            _write_reference_csv(
                root, policy_id, policy, "quarterly", lf_paths, quarterly
            ),
            _write_reference_csv(root, policy_id, policy, "annual", lf_paths, annual),
            _write_reference_csv(root, policy_id, policy, "daily", hf_paths),
            _write_reference_csv(root, policy_id, policy, "weekly", hf_paths, weekly),
        )
        file_records.extend(policy_files)
        policy_records.append(
            {
                "policy_id": policy_id,
                "policy": policy.as_dict(),
                "policy_fingerprint": _sha256_json(policy.as_dict()),
                "reference_scope_fingerprint": scope_fingerprint,
                "reference_input_fingerprint": generation_input.fingerprint,
            }
        )
        path_records.extend(
            {
                "policy_id": policy_id,
                "family": item.family,
                "path_entity": item.path_entity,
                "years": item.years,
                "fingerprint": item.fingerprint,
                "generation_authorized": item.generation_authorized,
            }
            for item in (*lf_paths, *hf_paths)
        )

    files = tuple(file_records)
    structural = validate_g4_reference_csvs(root, files)
    identity: dict[str, object] = {
        "schema_id": G4_REFERENCE_SCHEMA_ID,
        "schema_version": G4_REFERENCE_SCHEMA_VERSION,
        "contract_version": G4_REFERENCE_CONTRACT_VERSION,
        "execution_mode": "g4_reference_noncanonical",
        "g4_passed": True,
        "canonical_authority": False,
        "generation_authorized": False,
        "releasable": False,
        "structural_validation_passed": structural.passed,
        "g3_evidence_fingerprint": evidence.reference.fingerprint,
        "g3_bundle_fingerprint": evidence.evidence_bundle.fingerprint,
        "g3_run_id": evidence.run_id,
        "g3_source_code_fingerprint": evidence.g3_source_code_fingerprint,
        "g3_dependency_lock_fingerprint": evidence.g3_dependency_lock_fingerprint,
        "production_artifact_fingerprint": (
            evidence.production_artifact.reference.fingerprint
        ),
        "core_model_fingerprint": evidence.production_artifact.core_model_fingerprint,
        "parameter_set_fingerprint": (
            evidence.production_artifact.parameter_set_fingerprint
        ),
        "candidate_policy_set_fingerprint": _sha256_json(
            cast(
                ScientificCandidatePolicySet,
                evidence.production_artifact.generation_policy,
            ).as_dict()
        ),
        "config_fingerprint": config.fingerprint(),
        "processed_data_fingerprint": (
            calibration_input.provenance.processed_data_fingerprint
        ),
        "calibration_input_fingerprint": calibration_input.fingerprint,
        "scientific_version": config.simulation.scientific_version,
        "selected_n_states": evidence.production_artifact.selected_k,
        "serial_generator_version": SERIAL_GENERATOR_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "csv_serializer_version": CSV_SERIALIZER_VERSION,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "rng_domain": {
            "name": Domain.DEVELOPMENT.name,
            "value": int(Domain.DEVELOPMENT),
        },
        "fixture": {
            "policies": len(policies),
            "lf_paths_per_policy": G4_REFERENCE_LF_PATHS,
            "hf_paths_per_policy": G4_REFERENCE_HF_PATHS,
            "years": G4_REFERENCE_YEARS,
            "base_paths": len(path_records),
            "files": len(files),
            "rows": structural.rows,
        },
        "policies": policy_records,
        "paths": path_records,
        "files": [item.as_dict() for item in files],
        "structural_validation": structural.as_dict(),
    }
    fingerprint = _sha256_json(identity)
    manifest = {
        "schema_id": G4_REFERENCE_SCHEMA_ID,
        "schema_version": G4_REFERENCE_SCHEMA_VERSION,
        "report_fingerprint": fingerprint,
        "identity": identity,
    }
    content = _canonical_json_bytes(manifest)
    report_path = root / G4_REFERENCE_REPORT_NAME
    try:
        with report_path.open("xb") as handle:
            handle.write(content)
    except OSError as error:
        raise IntegrityError("G4 reference report cannot be written") from error
    return G4ReferenceReport(
        fingerprint=fingerprint,
        sha256=hashlib.sha256(content).hexdigest(),
        path=report_path,
        identity=identity,
        files=files,
        structural_validation=structural,
    )


def validate_g4_reference_csvs(
    output_root: str | Path,
    files: Sequence[G4ReferenceFileRecord],
) -> G4StructuralValidation:
    """Reread all fixed-fixture CSVs and prove their structural identities."""

    root = Path(output_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise IntegrityError("G4 reference validation root is unsafe")
    if not isinstance(files, Sequence) or len(files) != 25:
        raise IntegrityError("G4 reference validation requires exactly 25 CSV files")
    parsed: dict[tuple[str, ReferenceFrequency], FloatArray] = {}
    total_rows = 0
    expected_keys = {
        (policy_id, frequency)
        for policy_id in _expected_policy_ids()
        for frequency in _FREQUENCIES
    }
    for record in files:
        values = _read_reference_csv(root, record)
        key = (record.policy_id, record.frequency)
        if key in parsed:
            raise IntegrityError("G4 reference CSV record is duplicated")
        parsed[key] = values
        total_rows += record.rows
    if set(parsed) != expected_keys:
        raise IntegrityError("G4 reference CSV policy/frequency matrix is incomplete")

    maximum_log_error = 0.0
    maximum_simple_error = 0.0
    for policy_id in _expected_policy_ids():
        monthly = parsed[(policy_id, "monthly")]
        quarterly = parsed[(policy_id, "quarterly")]
        annual = parsed[(policy_id, "annual")]
        daily = parsed[(policy_id, "daily")]
        weekly = parsed[(policy_id, "weekly")]
        expected_quarterly = (
            monthly.reshape(
                G4_REFERENCE_LF_PATHS, G4_REFERENCE_YEARS, 4, 3, ASSET_COUNT
            )
            .sum(axis=3)
            .reshape(G4_REFERENCE_LF_PATHS, -1, ASSET_COUNT)
        )
        expected_annual = monthly.reshape(
            G4_REFERENCE_LF_PATHS,
            G4_REFERENCE_YEARS,
            MONTHS_PER_YEAR,
            ASSET_COUNT,
        ).sum(axis=2)
        expected_weekly = _weekly_from_daily_matrix(daily)
        for actual, expected, grouped in (
            (
                quarterly,
                expected_quarterly,
                monthly.reshape(
                    G4_REFERENCE_LF_PATHS,
                    G4_REFERENCE_YEARS * 4,
                    3,
                    ASSET_COUNT,
                ),
            ),
            (
                annual,
                expected_annual,
                monthly.reshape(
                    G4_REFERENCE_LF_PATHS,
                    G4_REFERENCE_YEARS,
                    MONTHS_PER_YEAR,
                    ASSET_COUNT,
                ),
            ),
            (weekly, expected_weekly, _weekly_groups(daily)),
        ):
            maximum_log_error = max(
                maximum_log_error, float(np.max(np.abs(actual - expected)))
            )
            with np.errstate(over="ignore", invalid="ignore"):
                simple_expected = np.prod(1.0 + np.expm1(grouped), axis=2) - 1.0
                simple_actual = np.expm1(actual)
            if not bool(
                np.isfinite(simple_expected).all() and np.isfinite(simple_actual).all()
            ):
                raise IntegrityError(
                    "G4 reference simple-return identity is non-finite"
                )
            maximum_simple_error = max(
                maximum_simple_error,
                float(np.max(np.abs(simple_actual - simple_expected))),
            )
    if maximum_log_error > G4_REFERENCE_LOG_SUM_TOLERANCE:
        raise IntegrityError(
            "G4 reference cross-frequency log-sum identity failed",
            details={"maximum_abs_error": maximum_log_error},
        )
    if maximum_simple_error > G4_REFERENCE_SIMPLE_RETURN_TOLERANCE:
        raise IntegrityError(
            "G4 reference simple-return roundtrip identity failed",
            details={"maximum_abs_error": maximum_simple_error},
        )
    return G4StructuralValidation(
        passed=True,
        files=len(files),
        rows=total_rows,
        maximum_log_sum_abs_error=maximum_log_error,
        maximum_simple_return_roundtrip_abs_error=maximum_simple_error,
    )


def _validate_canonical_inputs(
    evidence: LoadedG3EvidenceAuthority,
    config: PRPGConfig,
    calibration_input: CalibrationInput,
) -> None:
    if (
        type(config) is not PRPGConfig
        or config.fingerprint() != CANONICAL_CONFIG_FINGERPRINT
    ):
        raise ModelError("G4 reference runner requires the canonical resolved config")
    if (
        type(calibration_input) is not CalibrationInput
        or calibration_input.fingerprint != CANONICAL_CALIBRATION_INPUT_FINGERPRINT
        or calibration_input.provenance.processed_data_fingerprint
        != CANONICAL_PROCESSED_DATA_FINGERPRINT
        or calibration_input.provenance.calibration_config_fingerprint
        != CANONICAL_CONFIG_FINGERPRINT
    ):
        raise ModelError("G4 reference runner requires the canonical calibration input")
    artifact = evidence.production_artifact
    if (
        not evidence.evidence_bundle.passed
        or evidence.evidence_bundle.generation_authorized is not False
        or evidence.evidence_bundle.config_fingerprint != CANONICAL_CONFIG_FINGERPRINT
        or evidence.evidence_bundle.processed_data_fingerprint
        != CANONICAL_PROCESSED_DATA_FINGERPRINT
        or evidence.evidence_bundle.calibration_input_fingerprint
        != CANONICAL_CALIBRATION_INPUT_FINGERPRINT
        or artifact.selected_k != 4
        or config.model.fixed_regime_count != 4
        or config.validation.csv_log_sum_abs_tolerance != G4_REFERENCE_LOG_SUM_TOLERANCE
        or config.validation.simple_return_identity_abs_tolerance
        != G4_REFERENCE_SIMPLE_RETURN_TOLERANCE
    ):
        raise ModelError("G4 reference runner G3/config bindings are not canonical")
    candidates = artifact.generation_policy
    if not isinstance(candidates, ScientificCandidatePolicySet):
        raise ModelError("G4 reference runner requires the G3 candidate-policy set")
    candidate_policy_set_from_mapping(candidates.as_dict(), error_type=ModelError)


def _registered_reference_policies(
    evidence: LoadedG3EvidenceAuthority,
) -> tuple[tuple[str, ScientificGenerationPolicy], ...]:
    candidates = cast(
        ScientificCandidatePolicySet, evidence.production_artifact.generation_policy
    )
    base = {
        "schema_version": SCIENTIFIC_GENERATION_POLICY_SCHEMA_VERSION,
        "generator_algorithm": candidates.generator_algorithm,
        "block_policy_version": candidates.block_policy_version,
        "monthly_block_length": candidates.monthly_block_length,
        "daily_block_length": candidates.daily_block_length,
        "neutral_lambda_grid": list(candidates.neutral_lambda_candidates),
        "initial_state": candidates.initial_state,
        "synchronized_return_vectors": candidates.synchronized_return_vectors,
    }
    values: list[tuple[str, ScientificGenerationPolicy]] = []
    values.append(
        (
            "historical_vector",
            generation_policy_from_mapping(
                {
                    **base,
                    "alpha_policy": "historical_vector",
                    "selected_neutral_lambda": None,
                },
                error_type=ModelError,
            ),
        )
    )
    for selected in candidates.neutral_lambda_candidates:
        policy_id = f"kernel_mean_neutral_lambda_{selected:.2f}".replace(".", "_")
        values.append(
            (
                policy_id,
                generation_policy_from_mapping(
                    {
                        **base,
                        "alpha_policy": "kernel_mean_neutral",
                        "selected_neutral_lambda": selected,
                    },
                    error_type=ModelError,
                ),
            )
        )
    result = tuple(values)
    if tuple(item[0] for item in result) != _expected_policy_ids():
        raise ModelError("G4 reference policy registry is not exact")
    return result


def _expected_policy_ids() -> tuple[str, ...]:
    return (
        "historical_vector",
        "kernel_mean_neutral_lambda_0_25",
        "kernel_mean_neutral_lambda_0_50",
        "kernel_mean_neutral_lambda_0_75",
        "kernel_mean_neutral_lambda_1_00",
    )


def _create_isolated_output_root(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise GenerationError("G4 reference output root must be a path")
    root = Path(value)
    if not root.is_absolute() or ".." in root.parts:
        raise GenerationError("G4 reference output root must be absolute")
    if root.exists() or root.is_symlink():
        raise IntegrityError("G4 reference output root must not already exist")
    try:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
    except OSError as error:
        raise IntegrityError("G4 reference output root cannot be created") from error
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("G4 reference output root is unsafe")
    return root


def _write_reference_csv(
    root: Path,
    policy_id: str,
    policy: ScientificGenerationPolicy,
    frequency: ReferenceFrequency,
    paths: Sequence[G4ReferenceBasePath],
    derived: Sequence[FloatArray] | None = None,
) -> G4ReferenceFileRecord:
    if _POLICY_ID.fullmatch(policy_id) is None:
        raise IntegrityError("G4 reference policy ID is unsafe")
    family, periods_per_year = _FREQUENCIES[frequency]
    expected_paths = G4_REFERENCE_LF_PATHS if family == "LF" else G4_REFERENCE_HF_PATHS
    if len(paths) != expected_paths or (
        derived is not None and len(derived) != len(paths)
    ):
        raise IntegrityError("G4 reference CSV source geometry is invalid")
    relative = PurePosixPath("reference", "returns", policy_id, f"{frequency}.csv")
    destination = root.joinpath(*relative.parts)
    try:
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(CSV_HEADER_BYTES)
            for path_index, path in enumerate(paths):
                if (
                    not is_g4_reference_base_path(path)
                    or path.family != family
                    or path.path_entity != path_index + 1
                    or path.years != G4_REFERENCE_YEARS
                ):
                    raise IntegrityError("G4 reference CSV path sequence is invalid")
                values = path.log_returns if derived is None else derived[path_index]
                expected_rows = G4_REFERENCE_YEARS * periods_per_year
                if values.shape != (expected_rows, ASSET_COUNT):
                    raise IntegrityError("G4 reference CSV value geometry is invalid")
                path_id = f"{family}{path.path_entity:06d}"
                for zero_index, vector in enumerate(values):
                    handle.write(
                        encode_csv_row(
                            path_id,
                            zero_index + 1,
                            zero_index // periods_per_year + 1,
                            zero_index % periods_per_year + 1,
                            vector,
                        )
                    )
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, UnicodeError) as error:
        raise IntegrityError("G4 reference CSV cannot be written") from error
    content = destination.read_bytes()
    return G4ReferenceFileRecord(
        policy_id=policy_id,
        alpha_policy=policy.alpha_policy,
        selected_neutral_lambda=policy.selected_neutral_lambda,
        family=family,
        frequency=frequency,
        relative_path=relative.as_posix(),
        path_count=len(paths),
        years=G4_REFERENCE_YEARS,
        rows=len(paths) * G4_REFERENCE_YEARS * periods_per_year,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        source_path_fingerprints=tuple(item.fingerprint for item in paths),
    )


def _read_reference_csv(root: Path, record: G4ReferenceFileRecord) -> FloatArray:
    if not isinstance(record, G4ReferenceFileRecord):
        raise IntegrityError("G4 reference file record is not typed")
    family, periods_per_year = _FREQUENCIES[record.frequency]
    expected_paths = G4_REFERENCE_LF_PATHS if family == "LF" else G4_REFERENCE_HF_PATHS
    expected_relative = PurePosixPath(
        "reference", "returns", record.policy_id, f"{record.frequency}.csv"
    )
    path = root.joinpath(*expected_relative.parts)
    if (
        record.relative_path != expected_relative.as_posix()
        or record.family != family
        or record.path_count != expected_paths
        or record.years != G4_REFERENCE_YEARS
        or record.rows != expected_paths * G4_REFERENCE_YEARS * periods_per_year
        or len(record.source_path_fingerprints) != expected_paths
        or not path.is_file()
        or path.is_symlink()
    ):
        raise IntegrityError("G4 reference file record geometry is invalid")
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise IntegrityError("G4 reference CSV cannot be read") from error
    if (
        len(content) != record.bytes
        or hashlib.sha256(content).hexdigest() != record.sha256
    ):
        raise IntegrityError("G4 reference CSV content hash changed")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as error:
        raise IntegrityError("G4 reference CSV header is missing") from error
    if header != CSV_HEADER.split(","):
        raise IntegrityError("G4 reference CSV schema or hidden columns changed")
    values = np.empty((expected_paths, G4_REFERENCE_YEARS * periods_per_year, 3))
    row_count = 0
    try:
        for row_count, row in enumerate(reader, start=1):
            if row_count > record.rows:
                raise IntegrityError("G4 reference CSV contains too many rows")
            if len(row) != 7:
                raise IntegrityError(
                    "G4 reference CSV row has hidden or missing columns"
                )
            zero_index = row_count - 1
            entity = zero_index // (G4_REFERENCE_YEARS * periods_per_year) + 1
            period_zero = zero_index % (G4_REFERENCE_YEARS * periods_per_year)
            expected_integers = (
                str(period_zero + 1),
                str(period_zero // periods_per_year + 1),
                str(period_zero % periods_per_year + 1),
            )
            if (
                row[0] != f"{family}{entity:06d}"
                or tuple(row[1:4]) != expected_integers
            ):
                raise IntegrityError("G4 reference CSV row ordering is invalid")
            for asset, token in enumerate(row[4:]):
                try:
                    parsed = float(token)
                except ValueError as error:
                    raise IntegrityError(
                        "G4 reference CSV return is not numeric"
                    ) from error
                if not math.isfinite(parsed) or format_csv_float64(parsed) != token:
                    raise IntegrityError(
                        "G4 reference CSV float grammar is noncanonical"
                    )
                values[entity - 1, period_zero, asset] = parsed
    except csv.Error as error:
        raise IntegrityError("G4 reference CSV syntax is invalid") from error
    if row_count != record.rows:
        raise IntegrityError("G4 reference CSV row count is invalid")
    return values


def _quarterly_values(path: G4ReferenceBasePath) -> FloatArray:
    values = path.log_returns.reshape(path.years, 4, 3, ASSET_COUNT).sum(axis=2)
    return np.asarray(values.reshape(path.years * 4, ASSET_COUNT), dtype=np.float64)


def _annual_values(path: G4ReferenceBasePath) -> FloatArray:
    return np.asarray(
        path.log_returns.reshape(path.years, MONTHS_PER_YEAR, ASSET_COUNT).sum(axis=1),
        dtype=np.float64,
    )


def _weekly_values(path: G4ReferenceBasePath) -> FloatArray:
    return np.asarray(
        _weekly_from_daily_matrix(path.log_returns.reshape(1, -1, ASSET_COUNT))[0],
        dtype=np.float64,
    )


def _weekly_from_daily_matrix(daily: FloatArray) -> FloatArray:
    result = np.empty(
        (daily.shape[0], G4_REFERENCE_YEARS * WEEKS_PER_YEAR, ASSET_COUNT),
        dtype=np.float64,
    )
    for path_index in range(daily.shape[0]):
        output = 0
        for year in range(G4_REFERENCE_YEARS):
            offset = year * SESSIONS_PER_YEAR
            for week in range(WEEKS_PER_YEAR):
                start = offset + WEEK_BOUNDARIES[week]
                stop = offset + WEEK_BOUNDARIES[week + 1]
                result[path_index, output] = daily[path_index, start:stop].sum(axis=0)
                output += 1
    return result


def _weekly_groups(daily: FloatArray) -> FloatArray:
    # SC252-52-v1 has 44 five-session and eight four-session weeks.  Padding
    # four-session weeks with exact log zero preserves both tested identities.
    result = np.zeros(
        (daily.shape[0], G4_REFERENCE_YEARS * WEEKS_PER_YEAR, 5, ASSET_COUNT),
        dtype=np.float64,
    )
    for path_index in range(daily.shape[0]):
        output = 0
        for year in range(G4_REFERENCE_YEARS):
            offset = year * SESSIONS_PER_YEAR
            for week in range(WEEKS_PER_YEAR):
                start = offset + WEEK_BOUNDARIES[week]
                stop = offset + WEEK_BOUNDARIES[week + 1]
                width = stop - start
                result[path_index, output, :width] = daily[path_index, start:stop]
                output += 1
    return result


def _canonical_json_bytes(value: object) -> bytes:
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
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
