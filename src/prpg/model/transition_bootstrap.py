"""Aggregate aligned transition evidence from macro-stability refits.

Every failed macro-bootstrap attempt remains in the fixed 100-attempt edge-
frequency denominator as an unsupported matrix.  Probability quantiles use
only successful refits, after their candidate labels are permuted onto the
main artifact's canonical state order.  Full per-attempt matrices are reduced
to counts, quantiles, and compact hashes rather than retained.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from typing import Literal, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray

from prpg.errors import ModelError
from prpg.model.calibration import (
    MacroBootstrapAttempt,
    MacroStabilityEvidence,
    registered_macro_stability_execution_identity,
)
from prpg.model.calibration_identity import macro_stability_evidence_fingerprint
from prpg.model.hmm import (
    GaussianHMMFit,
    HistoricalTransitionSupportReport,
    historical_transition_support_report,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    scientific_array_fingerprint,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MACRO_BOOTSTRAP_ATTEMPTS = 100
MINIMUM_SUCCESSFUL_REFITS = 90
TRANSITION_PROBABILITY_QUANTILES = (0.05, 0.50, 0.95)
TRANSITION_QUANTILE_METHOD: Literal["higher"] = "higher"
HISTORICAL_PROBABILITY_MINIMUM = 0.01
HISTORICAL_EXPECTED_COUNT_MINIMUM = 2.0
HISTORICAL_VITERBI_COUNT_MINIMUM = 1
_TRANSITION_AGGREGATE_CAPABILITY = object()
_MINTED_TRANSITION_AGGREGATES: dict[int, TransitionBootstrapAggregate] = {}
_TRANSITION_IDENTITY_FIELDS = frozenset(
    {
        "parent_model_fingerprint",
        "macro_input_fingerprint",
        "macro_evidence_fingerprint",
        "source_materialization_fingerprint",
        "rng_execution_fingerprint",
        "execution_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class TransitionBootstrapAttemptAudit:
    """Compact alignment/support evidence for one fixed attempt ID."""

    attempt: int
    successful: bool
    candidate_to_reference: tuple[int, ...] | None
    supported_cell_count: int
    aligned_input_sha256: str | None


@dataclass(frozen=True, slots=True)
class TransitionBootstrapAggregate:
    """Aligned aggregate and rebuilt main-fit historical support report."""

    n_states: int
    attempts: int
    successful_refits: int
    failed_refits: int
    edge_frequency_denominator: int
    quantile_probabilities: tuple[float, float, float]
    quantile_method: str
    bootstrap_edge_frequencies: FloatArray
    bootstrap_probability_quantiles: FloatArray
    historical_transition_support: HistoricalTransitionSupportReport
    attempt_audits: tuple[TransitionBootstrapAttemptAudit, ...]
    parent_model_fingerprint: str | None = None
    macro_input_fingerprint: str | None = None
    macro_evidence_fingerprint: str | None = None
    source_materialization_fingerprint: str | None = None
    rng_execution_fingerprint: str | None = None
    execution_fingerprint: str | None = None
    _main_fit: GaussianHMMFit | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _macro_stability: MacroStabilityEvidence | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _master_seed: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _scientific_version: int | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _execution_seal: object | None = field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class TransitionBootstrapAggregateIdentity:
    """Reverified transition result and exact main-fit/macro lineage."""

    role: Literal["design", "production"]
    n_states: int
    master_seed: int
    scientific_version: int
    parent_model_fingerprint: str
    macro_input_fingerprint: str
    macro_evidence_fingerprint: str
    source_materialization_fingerprint: str
    rng_execution_fingerprint: str
    execution_fingerprint: str


def transition_bootstrap_aggregate_identity(
    value: TransitionBootstrapAggregate,
) -> TransitionBootstrapAggregateIdentity:
    """Revalidate one producer-minted aggregate against both live parents."""

    if (
        not isinstance(value, TransitionBootstrapAggregate)
        or value._execution_seal is not _TRANSITION_AGGREGATE_CAPABILITY
        or _MINTED_TRANSITION_AGGREGATES.get(id(value)) is not value
    ):
        raise ModelError("transition-bootstrap aggregate lacks producer authority")
    main_fit = value._main_fit
    macro_stability = value._macro_stability
    master_seed = value._master_seed
    scientific_version = value._scientific_version
    if (
        main_fit is None
        or macro_stability is None
        or master_seed is None
        or scientific_version is None
    ):
        raise ModelError("transition-bootstrap retained source identity is incomplete")
    macro = registered_macro_stability_execution_identity(macro_stability)
    parent_fingerprint = gaussian_hmm_fit_fingerprint(main_fit)
    evidence_fingerprint = macro_stability_evidence_fingerprint(macro_stability)
    if (
        macro.parent_model_fingerprint != parent_fingerprint
        or macro.n_states != value.n_states
        or main_fit.n_states != value.n_states
        or macro.master_seed != master_seed
        or macro.scientific_version != scientific_version
    ):
        raise ModelError("transition-bootstrap parent lineage changed")
    source_fingerprint = _transition_source_materialization_fingerprint(
        role=macro.role,
        n_states=value.n_states,
        parent_model_fingerprint=parent_fingerprint,
        macro_input_fingerprint=macro.macro_input_fingerprint,
        macro_evidence_fingerprint=evidence_fingerprint,
        master_seed=master_seed,
        scientific_version=scientific_version,
        rng_execution_fingerprint=macro.rng_execution_fingerprint,
    )
    _require_immutable_public_arrays(value)
    execution_fingerprint = _transition_execution_fingerprint(value)
    if (
        value.parent_model_fingerprint != parent_fingerprint
        or value.macro_input_fingerprint != macro.macro_input_fingerprint
        or value.macro_evidence_fingerprint != evidence_fingerprint
        or value.source_materialization_fingerprint != source_fingerprint
        or value.rng_execution_fingerprint != macro.rng_execution_fingerprint
        or value.execution_fingerprint != execution_fingerprint
    ):
        raise ModelError("transition-bootstrap aggregate identity changed")
    return TransitionBootstrapAggregateIdentity(
        role=macro.role,
        n_states=value.n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
        parent_model_fingerprint=parent_fingerprint,
        macro_input_fingerprint=macro.macro_input_fingerprint,
        macro_evidence_fingerprint=evidence_fingerprint,
        source_materialization_fingerprint=source_fingerprint,
        rng_execution_fingerprint=macro.rng_execution_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def is_registered_transition_bootstrap_aggregate(
    value: object,
) -> TypeGuard[TransitionBootstrapAggregate]:
    """Return whether ``value`` is an unchanged producer-minted aggregate."""

    if not isinstance(value, TransitionBootstrapAggregate):
        return False
    try:
        transition_bootstrap_aggregate_identity(value)
    except (ModelError, TypeError, ValueError, OverflowError):
        return False
    return True


def aggregate_transition_bootstrap(
    main_fit: GaussianHMMFit,
    macro_stability: MacroStabilityEvidence,
) -> TransitionBootstrapAggregate:
    """Align and aggregate the exact 100 macro-stability transition refits."""

    if not isinstance(main_fit, GaussianHMMFit):
        raise ModelError("transition-bootstrap main fit must be GaussianHMMFit")
    if not isinstance(macro_stability, MacroStabilityEvidence):
        raise ModelError("transition-bootstrap evidence must be MacroStabilityEvidence")
    n_states = main_fit.n_states
    if n_states not in {2, 3, 4, 5}:
        raise ModelError("transition-bootstrap main fit has an unregistered K")
    if macro_stability.n_states != n_states:
        raise ModelError("transition-bootstrap evidence K does not match main fit")
    attempts = macro_stability.attempts
    if len(attempts) != MACRO_BOOTSTRAP_ATTEMPTS:
        raise ModelError("transition bootstrap requires exactly 100 attempts")
    attempt_ids = tuple(attempt.attempt for attempt in attempts)
    if attempt_ids != tuple(range(MACRO_BOOTSTRAP_ATTEMPTS)):
        raise ModelError("transition-bootstrap attempt IDs must be exactly 0..99")
    success_count = sum(attempt.successful for attempt in attempts)
    if macro_stability.summary.successful_attempts != success_count:
        raise ModelError("macro-stability success metadata is inconsistent")
    if success_count < MINIMUM_SUCCESSFUL_REFITS:
        raise ModelError(
            "transition bootstrap has fewer than 90 successful refits",
            details={
                "actual": success_count,
                "required": MINIMUM_SUCCESSFUL_REFITS,
            },
        )

    supported_counts = np.zeros((n_states, n_states), dtype=np.int64)
    successful_probabilities = np.empty(
        (success_count, n_states, n_states), dtype=np.float64
    )
    audits: list[TransitionBootstrapAttemptAudit] = []
    success_row = 0
    for attempt in attempts:
        if not attempt.successful:
            _validate_failed_attempt(attempt)
            audits.append(
                TransitionBootstrapAttemptAudit(
                    attempt.attempt,
                    False,
                    None,
                    0,
                    None,
                )
            )
            continue
        fitted = attempt.fit
        if fitted is None:  # pragma: no cover - narrowed by successful property.
            raise AssertionError("successful macro attempt lost its fit")
        if not isinstance(fitted, GaussianHMMFit):
            raise ModelError("successful macro attempt has an invalid fit record")
        if fitted.n_states != n_states:
            raise ModelError(
                "successful macro attempt K does not match the main fit",
                details={"attempt": attempt.attempt},
            )
        mapping, reference_to_candidate = _validated_permutation(
            attempt.candidate_to_reference,
            n_states=n_states,
            attempt=attempt.attempt,
        )
        probability = _probability_matrix(
            fitted.parameters.transition_matrix,
            n_states=n_states,
            attempt=attempt.attempt,
        )
        viterbi = _count_matrix(
            fitted.state_path_diagnostics.transition_counts,
            n_states=n_states,
            attempt=attempt.attempt,
        )
        expected = _expected_count_matrix(
            fitted.forward_backward.expected_transition_counts,
            n_states=n_states,
            attempt=attempt.attempt,
        )
        aligned_probability = probability[
            np.ix_(reference_to_candidate, reference_to_candidate)
        ]
        aligned_viterbi = viterbi[
            np.ix_(reference_to_candidate, reference_to_candidate)
        ]
        aligned_expected = expected[
            np.ix_(reference_to_candidate, reference_to_candidate)
        ]
        supported = (
            (aligned_probability >= HISTORICAL_PROBABILITY_MINIMUM)
            & (aligned_expected >= HISTORICAL_EXPECTED_COUNT_MINIMUM)
            & (aligned_viterbi >= HISTORICAL_VITERBI_COUNT_MINIMUM)
        )
        supported_counts += supported.astype(np.int64)
        successful_probabilities[success_row] = aligned_probability
        success_row += 1
        audits.append(
            TransitionBootstrapAttemptAudit(
                attempt.attempt,
                True,
                tuple(int(value) for value in mapping),
                int(np.count_nonzero(supported)),
                _aligned_input_sha256(
                    mapping,
                    aligned_probability,
                    aligned_viterbi,
                    aligned_expected,
                ),
            )
        )
    if success_row != success_count:
        raise AssertionError("transition-bootstrap success rows are inconsistent")

    frequencies = supported_counts.astype(np.float64) / MACRO_BOOTSTRAP_ATTEMPTS
    raw_quantiles = np.quantile(
        successful_probabilities,
        TRANSITION_PROBABILITY_QUANTILES,
        axis=0,
        method=TRANSITION_QUANTILE_METHOD,
    )
    quantiles = np.moveaxis(raw_quantiles, 0, 2).astype(np.float64, copy=True)
    report = historical_transition_support_report(
        main_fit.parameters.transition_matrix,
        main_fit.state_path_diagnostics.transition_counts,
        main_fit.forward_backward.expected_transition_counts,
        bootstrap_edge_frequencies=frequencies,
        bootstrap_probability_quantiles=quantiles,
    )
    frequencies = frequencies.astype(np.float64, copy=True)
    quantiles = quantiles.astype(np.float64, copy=True)
    _readonly(frequencies, quantiles)
    result = TransitionBootstrapAggregate(
        n_states=n_states,
        attempts=MACRO_BOOTSTRAP_ATTEMPTS,
        successful_refits=success_count,
        failed_refits=MACRO_BOOTSTRAP_ATTEMPTS - success_count,
        edge_frequency_denominator=MACRO_BOOTSTRAP_ATTEMPTS,
        quantile_probabilities=TRANSITION_PROBABILITY_QUANTILES,
        quantile_method=TRANSITION_QUANTILE_METHOD,
        bootstrap_edge_frequencies=frequencies,
        bootstrap_probability_quantiles=quantiles,
        historical_transition_support=report,
        attempt_audits=tuple(audits),
    )
    result = cast(TransitionBootstrapAggregate, _immutable_public_tree(result))
    try:
        macro = registered_macro_stability_execution_identity(macro_stability)
    except ModelError:
        # The aggregation estimator remains usable in isolated/report-only tests,
        # but unregistered macro evidence cannot mint a canonical G3 identity.
        return result
    parent_fingerprint = gaussian_hmm_fit_fingerprint(main_fit)
    if (
        macro.parent_model_fingerprint != parent_fingerprint
        or macro.n_states != n_states
    ):
        raise ModelError(
            "transition-bootstrap macro evidence belongs to another main fit"
        )
    evidence_fingerprint = macro_stability_evidence_fingerprint(macro_stability)
    source_fingerprint = _transition_source_materialization_fingerprint(
        role=macro.role,
        n_states=n_states,
        parent_model_fingerprint=parent_fingerprint,
        macro_input_fingerprint=macro.macro_input_fingerprint,
        macro_evidence_fingerprint=evidence_fingerprint,
        master_seed=macro.master_seed,
        scientific_version=macro.scientific_version,
        rng_execution_fingerprint=macro.rng_execution_fingerprint,
    )
    execution_fingerprint = _transition_execution_fingerprint(result)
    for name, value in (
        ("parent_model_fingerprint", parent_fingerprint),
        ("macro_input_fingerprint", macro.macro_input_fingerprint),
        ("macro_evidence_fingerprint", evidence_fingerprint),
        ("source_materialization_fingerprint", source_fingerprint),
        ("rng_execution_fingerprint", macro.rng_execution_fingerprint),
        ("execution_fingerprint", execution_fingerprint),
        ("_main_fit", main_fit),
        ("_macro_stability", macro_stability),
        ("_master_seed", macro.master_seed),
        ("_scientific_version", macro.scientific_version),
        ("_execution_seal", _TRANSITION_AGGREGATE_CAPABILITY),
    ):
        object.__setattr__(result, name, value)
    _MINTED_TRANSITION_AGGREGATES[id(result)] = result
    transition_bootstrap_aggregate_identity(result)
    return result


def _validate_failed_attempt(attempt: MacroBootstrapAttempt) -> None:
    if attempt.fit is not None or attempt.candidate_to_reference is not None:
        raise ModelError(
            "failed macro attempt retains fit or alignment evidence",
            details={"attempt": attempt.attempt},
        )


def _validated_permutation(
    values: NDArray[np.integer] | None,
    *,
    n_states: int,
    attempt: int,
) -> tuple[IntArray, IntArray]:
    if values is None:
        raise ModelError(
            "successful macro attempt is missing its label permutation",
            details={"attempt": attempt},
        )
    raw = np.asarray(values)
    if raw.shape != (n_states,) or raw.dtype.kind not in {"i", "u"}:
        raise ModelError(
            "macro-attempt label permutation has invalid shape or dtype",
            details={"attempt": attempt},
        )
    mapping = raw.astype(np.int64, copy=True)
    if tuple(sorted(int(value) for value in mapping)) != tuple(range(n_states)):
        raise ModelError(
            "macro-attempt label mapping is not an exact permutation",
            details={"attempt": attempt},
        )
    inverse = np.empty(n_states, dtype=np.int64)
    inverse[mapping] = np.arange(n_states, dtype=np.int64)
    return mapping, inverse


def _probability_matrix(
    values: NDArray[np.floating], *, n_states: int, attempt: int
) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (n_states, n_states) or not np.isfinite(matrix).all():
        raise ModelError(
            "macro-attempt transition probabilities have invalid shape or values",
            details={"attempt": attempt},
        )
    if (
        bool((matrix < 0.0).any())
        or float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) > 1e-12
    ):
        raise ModelError(
            "macro-attempt transition probabilities are invalid",
            details={"attempt": attempt},
        )
    return matrix


def _count_matrix(
    values: NDArray[np.integer], *, n_states: int, attempt: int
) -> IntArray:
    raw = np.asarray(values)
    if raw.shape != (n_states, n_states) or raw.dtype.kind not in {"i", "u"}:
        raise ModelError(
            "macro-attempt Viterbi counts have invalid shape or dtype",
            details={"attempt": attempt},
        )
    result = raw.astype(np.int64, copy=False)
    if bool((result < 0).any()):
        raise ModelError(
            "macro-attempt Viterbi counts must be non-negative",
            details={"attempt": attempt},
        )
    return result


def _expected_count_matrix(
    values: NDArray[np.floating], *, n_states: int, attempt: int
) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.shape != (n_states, n_states)
        or not np.isfinite(matrix).all()
        or bool((matrix < 0.0).any())
    ):
        raise ModelError(
            "macro-attempt expected counts are invalid",
            details={"attempt": attempt},
        )
    return matrix


def _aligned_input_sha256(
    mapping: IntArray,
    probability: FloatArray,
    viterbi: IntArray,
    expected: FloatArray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mapping, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(probability, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(viterbi, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(expected, dtype="<f8").tobytes())
    return digest.hexdigest()


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)


def _transition_source_materialization_fingerprint(
    *,
    role: Literal["design", "production"],
    n_states: int,
    parent_model_fingerprint: str,
    macro_input_fingerprint: str,
    macro_evidence_fingerprint: str,
    master_seed: int,
    scientific_version: int,
    rng_execution_fingerprint: str,
) -> str:
    payload = {
        "schema_id": "prpg-transition-bootstrap-source-materialization-v1",
        "role": role,
        "n_states": n_states,
        "parent_model_fingerprint": parent_model_fingerprint,
        "macro_input_fingerprint": macro_input_fingerprint,
        "macro_evidence_fingerprint": macro_evidence_fingerprint,
        "master_seed": master_seed,
        "scientific_version": scientific_version,
        "rng_execution_fingerprint": rng_execution_fingerprint,
    }
    return _json_fingerprint(payload)


def _transition_execution_fingerprint(value: TransitionBootstrapAggregate) -> str:
    return _json_fingerprint(
        {
            "schema_id": "prpg-transition-bootstrap-aggregate-v1",
            "result": _normalize_scientific_content(
                value,
                excluded_fields=_TRANSITION_IDENTITY_FIELDS,
            ),
        }
    )


def _normalize_scientific_content(
    value: object,
    *,
    excluded_fields: frozenset[str] | set[str] = frozenset(),
) -> object:
    if isinstance(value, np.ndarray):
        return {
            "array_fingerprint": scientific_array_fingerprint(
                value,
                role="transition_bootstrap_array",
            ),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _normalize_scientific_content(value.item())
    if isinstance(value, Enum):
        return _normalize_scientific_content(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _normalize_scientific_content(
                    getattr(value, item.name),
                    excluded_fields=excluded_fields,
                )
                for item in fields(value)
                if not item.name.startswith("_") and item.name not in excluded_fields
            },
        }
    if isinstance(value, tuple | list):
        return [
            _normalize_scientific_content(item, excluded_fields=excluded_fields)
            for item in value
        ]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("transition-bootstrap identity contains non-finite data")
        return value
    raise ModelError(
        "transition-bootstrap identity contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _json_fingerprint(value: object) -> str:
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
        ).encode()
    ).hexdigest()


def _immutable_public_tree(value: object) -> object:
    if isinstance(value, np.ndarray):
        canonical = np.ascontiguousarray(value)
        result = np.frombuffer(
            canonical.tobytes(order="C"),
            dtype=canonical.dtype,
        ).reshape(canonical.shape)
        _require_immutable_array(result)
        return result
    if isinstance(value, tuple):
        return tuple(_immutable_public_tree(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: _immutable_public_tree(getattr(value, item.name))
            for item in fields(value)
            if item.init and not item.name.startswith("_")
        }
        return replace(value, **updates)
    return value


def _require_immutable_public_arrays(value: object) -> None:
    if isinstance(value, np.ndarray):
        _require_immutable_array(value)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _require_immutable_public_arrays(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if not item.name.startswith("_"):
                _require_immutable_public_arrays(getattr(value, item.name))


def _require_immutable_array(value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.flags.writeable or not array.flags.c_contiguous:
        raise ModelError("transition-bootstrap outputs must be immutable arrays")
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, bytes):
        raise ModelError("transition-bootstrap outputs must be bytes-backed")
