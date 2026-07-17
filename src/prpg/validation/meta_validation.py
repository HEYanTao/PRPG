"""Lean deterministic G5 suite meta-validation and material defects.

The expensive suite is supplied as two ordinary callables: one clean-null
replicate and one registered-defect replicate.  This keeps the coordinator
small while making the authority boundary exact: canonical evidence requires
1,000 clean nulls, all twelve named 1,000-replicate claims, the approved 37..64
type-I count, and at least 925 detections for every claim.

Return defects below are pure, copy-on-write transforms.  They never retune a
threshold and each call starts from the pristine arrays passed by its caller.
Unexpected numerical or operational errors abort rather than becoming a pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError
from prpg.validation.records import (
    CanonicalValidationResults,
    ValidationDecision,
    build_canonical_results,
)
from prpg.validation.statistics import (
    META_MIN_DETECTIONS,
    META_REPLICATES,
    META_TYPE_I_MAX_FALSE_REJECTIONS,
    META_TYPE_I_MIN_FALSE_REJECTIONS,
    POWER_CLAIM_NAMES,
    MetaPowerAudit,
    MetaTypeIAudit,
    bonferroni_power_audit,
    holm_decisions,
    meta_type_i_audit,
)

FloatArray = NDArray[np.float64]
FamilyName = Literal["g5_a", "g5_b", "g5_c", "g5_d", "hmm_identification"]

G5_META_SCHEMA_ID: Final = "prpg-g5-meta-validation-v1"


@dataclass(frozen=True, slots=True)
class RegisteredDefect:
    """One frozen claim, transform identity, and designated detection family."""

    claim_name: str
    transform_id: str
    designated_family: FamilyName
    material_target: str


G5_DEFECT_REGISTRY: Final = (
    RegisteredDefect("linear_alpha_sharpe", "linear-lag-sign-v1", "g5_a", "0.10"),
    RegisteredDefect("normalized_covariance", "equity-scale-v1", "g5_b", "0.10"),
    RegisteredDefect("maximum_correlation", "correlation-cholesky-v1", "g5_b", "0.05"),
    RegisteredDefect(
        "one_session_desynchronization", "taxable-lag-v1", "g5_b", "one-session"
    ),
    RegisteredDefect(
        "tail_clipping", "development-quantile-clip-v1", "g5_d", "0.01/0.99"
    ),
    RegisteredDefect(
        "transition_perturbation", "diagonal-transition-v1", "g5_d", "+0.10/1.10"
    ),
    RegisteredDefect(
        "hmm_merged_states", "fixed-k3-merged-v1", "hmm_identification", "reject"
    ),
    RegisteredDefect(
        "hmm_overlapping_states", "fixed-k3-overlap-v1", "hmm_identification", "reject"
    ),
    RegisteredDefect("nonlinear_daily", "quadratic-sign-v1", "g5_a", "0.10"),
    RegisteredDefect("nonlinear_monthly", "quadratic-sign-v1", "g5_a", "0.10"),
    RegisteredDefect("subsequence_daily", "exact-subsequence-v1", "g5_a", "h=63"),
    RegisteredDefect("subsequence_monthly", "exact-subsequence-v1", "g5_a", "h=6"),
)


@dataclass(frozen=True, slots=True)
class ReturnDefectBundle:
    """New development/qualification arrays and a verified fixture oracle."""

    claim_name: str
    development_paths: tuple[FloatArray, ...]
    qualification_paths: tuple[FloatArray, ...]
    oracle_name: str
    oracle_value: float
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5MetaValidationAssessment:
    """Reduced diagnostic or exact canonical 13,000-suite-call result."""

    schema_id: str
    replicate_count: int
    false_rejections: int
    detection_counts: tuple[tuple[str, int], ...]
    elapsed_seconds: float
    exact_canonical_count: bool
    type_i_audit: MetaTypeIAudit | None
    power_audit: MetaPowerAudit | None
    decisions: tuple[ValidationDecision, ...]
    passed: bool
    canonical_results: CanonicalValidationResults | None
    fingerprint: str


@dataclass(frozen=True, slots=True)
class G5MetaResourceProjection:
    """Linear projection from a reduced same-host meta pilot."""

    pilot_replicates: int
    pilot_elapsed_seconds: float
    target_replicates: int
    suite_calls_per_replicate: int
    projected_seconds: float
    projected_hours: float


NullSuite = Callable[[int], Mapping[str, float]]
PowerSuite = Callable[[RegisteredDefect, int], bool]


def run_g5_meta_validation(
    null_suite: NullSuite,
    power_suite: PowerSuite,
    *,
    threshold_fingerprint: str,
    qualification_results_fingerprint: str,
    result_id: str,
    replicate_count: int = META_REPLICATES,
) -> G5MetaValidationAssessment:
    """Execute one clean and twelve defect suites per fixed replicate.

    A clean replicate is a false rejection when Holm rejects any of its four
    supplied G5 family p-values.  A power callback must report detection only
    for the defect's designated family after the complete suite correction.
    """

    count = _positive_int(replicate_count, "meta replicate count")
    if count > META_REPLICATES:
        raise ValidationError("meta replicate count exceeds the registered 1,000")
    threshold = _sha256(threshold_fingerprint, "threshold")
    qualification = _sha256(qualification_results_fingerprint, "qualification results")
    if tuple(item.claim_name for item in G5_DEFECT_REGISTRY) != POWER_CLAIM_NAMES:
        raise AssertionError("G5 defect registry diverges from power claims")

    started = time.monotonic()
    false_rejections = 0
    detections = dict.fromkeys(POWER_CLAIM_NAMES, 0)
    for replicate in range(count):
        clean_p_values = null_suite(replicate)
        if not isinstance(clean_p_values, Mapping) or set(clean_p_values) != {
            "g5_a",
            "g5_b",
            "g5_c",
            "g5_d",
        }:
            raise ValidationError("meta null callback must return exactly G5-A/B/C/D")
        if holm_decisions(clean_p_values).any_rejected:
            false_rejections += 1
        for defect in G5_DEFECT_REGISTRY:
            detected = power_suite(defect, replicate)
            if type(detected) is not bool:
                raise ValidationError("meta power callback must return a boolean")
            detections[defect.claim_name] += int(detected)
    elapsed = time.monotonic() - started

    exact = count == META_REPLICATES
    type_i: MetaTypeIAudit | None = None
    power: MetaPowerAudit | None = None
    decisions: list[ValidationDecision] = []
    if exact:
        type_i = meta_type_i_audit(false_rejections)
        power = bonferroni_power_audit(detections)
        decisions.append(
            ValidationDecision(
                name="meta_type_i",
                required=True,
                passed=type_i.passed,
                observed=false_rejections,
                criterion="1,000 clean-suite false rejections in [37,64]",
            )
        )
        for claim in power.claims:
            decisions.append(
                ValidationDecision(
                    name=f"power_{claim.name}",
                    required=True,
                    passed=claim.passed,
                    observed=claim.detections,
                    criterion="1,000 defect detections >= 925",
                )
            )
    else:
        lower_type_i = math.floor(
            META_TYPE_I_MIN_FALSE_REJECTIONS * count / META_REPLICATES
        )
        upper_type_i = math.ceil(
            META_TYPE_I_MAX_FALSE_REJECTIONS * count / META_REPLICATES
        )
        decisions.append(
            ValidationDecision(
                name="meta_type_i_reduced_diagnostic",
                required=False,
                passed=lower_type_i <= false_rejections <= upper_type_i,
                observed=false_rejections,
                criterion=f"diagnostic scaled interval [{lower_type_i},{upper_type_i}]",
            )
        )
        scaled_power = math.ceil(META_MIN_DETECTIONS * count / META_REPLICATES)
        for name in POWER_CLAIM_NAMES:
            decisions.append(
                ValidationDecision(
                    name=f"power_{name}_reduced_diagnostic",
                    required=False,
                    passed=detections[name] >= scaled_power,
                    observed=detections[name],
                    criterion=f"diagnostic scaled detections >= {scaled_power}",
                )
            )

    decisions_tuple = tuple(decisions)
    canonical_results: CanonicalValidationResults | None = None
    if exact:
        assert type_i is not None and power is not None
        metric_fingerprints = {
            "meta_type_i": _fingerprint(type_i.as_dict()),
            **{
                f"power_{claim.name}": _fingerprint(claim.as_dict())
                for claim in power.claims
            },
        }
        canonical_results = build_canonical_results(
            result_id=result_id,
            threshold_fingerprint=threshold,
            subject_fingerprint=qualification,
            metric_fingerprints=metric_fingerprints,
            decisions=decisions_tuple,
        )
    passed = canonical_results is not None and canonical_results.passed
    identity = {
        "schema_id": G5_META_SCHEMA_ID,
        "replicate_count": count,
        "false_rejections": false_rejections,
        "detection_counts": detections,
        "exact_canonical_count": exact,
        "decisions": [item.as_dict() for item in decisions_tuple],
        "passed": passed,
        "canonical_results_fingerprint": (
            None if canonical_results is None else canonical_results.fingerprint
        ),
    }
    return G5MetaValidationAssessment(
        schema_id=G5_META_SCHEMA_ID,
        replicate_count=count,
        false_rejections=false_rejections,
        detection_counts=tuple((name, detections[name]) for name in POWER_CLAIM_NAMES),
        elapsed_seconds=elapsed,
        exact_canonical_count=exact,
        type_i_audit=type_i,
        power_audit=power,
        decisions=decisions_tuple,
        passed=passed,
        canonical_results=canonical_results,
        fingerprint=_fingerprint(identity),
    )


def apply_registered_return_defect(
    claim_name: str,
    development_paths: Sequence[ArrayLike],
    qualification_paths: Sequence[ArrayLike],
    *,
    annualization: Literal[12, 252],
    selected_development_paths: Sequence[int] | None = None,
    selected_qualification_paths: Sequence[int] | None = None,
) -> ReturnDefectBundle:
    """Apply one registered return defect to pristine path arrays only."""

    if claim_name not in POWER_CLAIM_NAMES:
        raise ValidationError("return defect claim is not registered")
    development = _paths(development_paths, "development")
    qualification = _paths(qualification_paths, "qualification")
    if annualization not in {12, 252}:
        raise ValidationError("return defect annualization must be 12 or 252")

    if claim_name in {"linear_alpha_sharpe", "nonlinear_daily", "nonlinear_monthly"}:
        nonlinear = claim_name.startswith("nonlinear")
        if nonlinear and (
            (claim_name.endswith("daily") and annualization != 252)
            or (claim_name.endswith("monthly") and annualization != 12)
        ):
            raise ValidationError("nonlinear claim/frequency disagree")
        transformed, oracle = _predictable_drift(
            development, qualification, annualization, nonlinear=nonlinear
        )
        return _defect_bundle(
            claim_name, *transformed, "alpha_sharpe_increment", oracle
        )
    if claim_name == "normalized_covariance":
        transformed, oracle = _covariance_distortion(development, qualification)
        return _defect_bundle(claim_name, *transformed, "normalized_covariance", oracle)
    if claim_name == "maximum_correlation":
        transformed, oracle = _correlation_distortion(development, qualification)
        return _defect_bundle(claim_name, *transformed, "maximum_correlation", oracle)
    if claim_name == "one_session_desynchronization":
        if annualization != 252:
            raise ValidationError("desynchronization is an HF-only defect")
        changed_paths = tuple(
            _desynchronize(path) for path in (*development, *qualification)
        )
        split = len(development)
        return _defect_bundle(
            claim_name, changed_paths[:split], changed_paths[split:], "session_lag", 1.0
        )
    if claim_name == "tail_clipping":
        pooled = np.vstack(development)
        lower = np.quantile(pooled, 0.01, axis=0, method="higher")
        upper = np.quantile(pooled, 0.99, axis=0, method="higher")
        clipped_paths = tuple(
            _freeze(np.clip(path, lower, upper))
            for path in (*development, *qualification)
        )
        split = len(development)
        return _defect_bundle(
            claim_name,
            clipped_paths[:split],
            clipped_paths[split:],
            "clipped_tail_mass",
            0.02,
        )
    if claim_name in {"subsequence_daily", "subsequence_monthly"}:
        expected = 252 if claim_name.endswith("daily") else 12
        if annualization != expected:
            raise ValidationError("subsequence claim/frequency disagree")
        dev_selected = _quarter_indices(
            selected_development_paths, len(development), "development"
        )
        qual_selected = _quarter_indices(
            selected_qualification_paths, len(qualification), "qualification"
        )
        n0, horizon = (756, 63) if annualization == 252 else (60, 6)
        changed_development = _inject_subsequence(
            development, dev_selected, n0, horizon
        )
        changed_qualification = _inject_subsequence(
            qualification, qual_selected, n0, horizon
        )
        return _defect_bundle(
            claim_name,
            changed_development,
            changed_qualification,
            "selected_fraction",
            0.25,
        )
    raise ValidationError("claim is not a return-array defect")


def perturb_transition_matrix(transition: ArrayLike) -> FloatArray:
    """Apply the registered +0.10 diagonal then row-/1.10 transform."""

    matrix = np.asarray(transition, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not bool(np.isfinite(matrix).all())
        or bool((matrix < 0.0).any())
        or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValidationError("transition defect requires a finite Markov matrix")
    changed = matrix.copy()
    changed[np.diag_indices_from(changed)] += 0.10
    changed /= 1.10
    if not np.allclose(changed.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValidationError("transition defect lost row normalization")
    return _freeze(changed)


def fixed_hmm_meta_means(claim_name: str) -> FloatArray:
    """Return the approved K=3 clean/merged/overlap four-feature means."""

    first_coordinates = {
        "clean": (-2.0, 0.0, 2.0),
        "hmm_merged_states": (-1.0, -1.0, 1.0),
        "hmm_overlapping_states": (-0.25, 0.25, 2.0),
    }
    if claim_name not in first_coordinates:
        raise ValidationError("HMM meta-DGP claim is invalid")
    result = np.zeros((3, 4), dtype=np.float64)
    result[:, 0] = first_coordinates[claim_name]
    return _freeze(result)


def project_g5_meta_resources(
    pilot: G5MetaValidationAssessment,
) -> G5MetaResourceProjection:
    """Project the exact 1,000-replicate suite from a measured reduced pilot."""

    if pilot.replicate_count <= 0 or pilot.elapsed_seconds <= 0.0:
        raise ValidationError("meta resource pilot requires positive measured work")
    projected = pilot.elapsed_seconds * META_REPLICATES / pilot.replicate_count
    return G5MetaResourceProjection(
        pilot_replicates=pilot.replicate_count,
        pilot_elapsed_seconds=pilot.elapsed_seconds,
        target_replicates=META_REPLICATES,
        suite_calls_per_replicate=1 + len(G5_DEFECT_REGISTRY),
        projected_seconds=projected,
        projected_hours=projected / 3600.0,
    )


def _predictable_drift(
    development: tuple[FloatArray, ...],
    qualification: tuple[FloatArray, ...],
    annualization: int,
    *,
    nonlinear: bool,
) -> tuple[tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]], float]:
    pooled = np.vstack(development)
    center = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0, ddof=0)
    if bool((scale <= 1e-12).any()):
        raise ValidationError("predictable-drift standardizer is degenerate")
    raw_development = [
        _predictor(path, center, scale, nonlinear) for path in development
    ]
    pooled_q = np.concatenate([item[1:] for item in raw_development])
    q_mean = float(np.mean(pooled_q))
    q_rms = float(np.sqrt(np.mean(np.square(pooled_q - q_mean))))
    if not math.isfinite(q_rms) or q_rms <= 0.0:
        raise ValidationError("predictable-drift signal is degenerate")

    def signal(path: FloatArray) -> FloatArray:
        return (_predictor(path, center, scale, nonlinear) - q_mean) / q_rms

    dev_signals = tuple(signal(path) for path in development)
    u = np.concatenate(
        [g[1:] * path[1:, 0] for g, path in zip(dev_signals, development, strict=True)]
    )
    sigma0 = float(np.std(u, ddof=1))
    if not math.isfinite(sigma0) or sigma0 <= 0.0:
        raise ValidationError("predictable-drift payoff scale is degenerate")
    amount = 0.10 * sigma0 / math.sqrt(annualization)

    def change(path: FloatArray) -> FloatArray:
        result = path.copy()
        result[1:, 0] += amount * signal(path)[1:]
        return _freeze(result)

    changed_development = tuple(change(path) for path in development)
    changed_qualification = tuple(change(path) for path in qualification)
    oracle = math.sqrt(annualization) * amount / sigma0
    if abs(oracle - 0.10) > 1e-12:
        raise ValidationError("predictable-drift oracle is wrong")
    return (changed_development, changed_qualification), oracle


def _predictor(
    path: FloatArray, center: FloatArray, scale: FloatArray, nonlinear: bool
) -> FloatArray:
    z = (path - center) / scale
    result = np.zeros(path.shape[0], dtype=np.float64)
    equity = np.where(z[:-1, 0] >= 0.0, 1.0, -1.0)
    if nonlinear:
        muni = np.where(z[:-1, 1] >= 0.0, 1.0, -1.0)
        equity *= muni
    result[1:] = equity
    return result


def _covariance_distortion(
    development: tuple[FloatArray, ...], qualification: tuple[FloatArray, ...]
) -> tuple[tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]], float]:
    pooled = np.vstack(development)
    center = np.mean(pooled, axis=0)
    covariance = np.cov(pooled, rowvar=False, ddof=0)
    denominator = float(np.linalg.norm(covariance, ord="fro"))
    if denominator <= 0.0:
        raise ValidationError("covariance defect reference is degenerate")

    def error(c: float) -> float:
        transform = np.diag((1.0 + c, 1.0, 1.0))
        changed = transform @ covariance @ transform.T
        return float(np.linalg.norm(changed - covariance, ord="fro") / denominator)

    low, high = 0.0, 4.0
    if error(high) < 0.10:
        raise ValidationError("covariance defect target is not bracketed")
    for _ in range(80):
        middle = (low + high) / 2.0
        if error(middle) < 0.10:
            low = middle
        else:
            high = middle
    coefficient = (low + high) / 2.0
    oracle = error(coefficient)
    if abs(oracle - 0.10) > 1e-10:
        raise ValidationError("covariance defect target was not solved")
    transform = np.diag((1.0 + coefficient, 1.0, 1.0))
    changed = tuple(
        _freeze(center + (path - center) @ transform.T)
        for path in (*development, *qualification)
    )
    split = len(development)
    return (changed[:split], changed[split:]), oracle


def _correlation_distortion(
    development: tuple[FloatArray, ...], qualification: tuple[FloatArray, ...]
) -> tuple[tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]], float]:
    pooled = np.vstack(development)
    center = np.mean(pooled, axis=0)
    covariance = np.cov(pooled, rowvar=False, ddof=0)
    scales = np.sqrt(np.diag(covariance))
    if bool((scales <= 1e-12).any()):
        raise ValidationError("correlation defect reference is degenerate")
    correlation = covariance / np.outer(scales, scales)
    target: FloatArray | None = None
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for sign in (1.0, -1.0):
            candidate = correlation.copy()
            candidate[left, right] += sign * 0.05
            candidate[right, left] += sign * 0.05
            if float(np.min(np.linalg.eigvalsh(candidate))) > 1e-6:
                target = candidate
                break
        if target is not None:
            break
    if target is None:
        raise ValidationError("correlation defect has no positive-definite edit")
    target_covariance = target * np.outer(scales, scales)
    lower = np.linalg.cholesky(covariance)
    lower_target = np.linalg.cholesky(target_covariance)
    # A = L_target L^-1, expressed as a solve to avoid an explicit inverse.
    transform = np.linalg.solve(lower.T, lower_target.T).T
    changed = tuple(
        _freeze(center + (path - center) @ transform.T)
        for path in (*development, *qualification)
    )
    changed_correlation = np.corrcoef(
        np.vstack(changed[: len(development)]), rowvar=False
    )
    oracle = float(np.max(np.abs(changed_correlation - correlation)))
    if abs(oracle - 0.05) > 1e-10:
        raise ValidationError("correlation defect identity is wrong")
    split = len(development)
    return (changed[:split], changed[split:]), oracle


def _desynchronize(path: FloatArray) -> FloatArray:
    result = path.copy()
    result[1:, 2] = path[:-1, 2]
    return _freeze(result)


def _inject_subsequence(
    paths: tuple[FloatArray, ...], selected: tuple[int, ...], n0: int, horizon: int
) -> tuple[FloatArray, ...]:
    result = [path.copy() for path in paths]
    source_start = n0 - (horizon + 1)
    stop = n0 + horizon + 1
    if source_start < 0 or any(path.shape[0] < stop for path in result):
        raise ValidationError("subsequence defect path is too short")
    for index in selected:
        result[index][n0:stop] = result[index][source_start:n0]
    return tuple(_freeze(path) for path in result)


def _quarter_indices(
    value: Sequence[int] | None, count: int, label: str
) -> tuple[int, ...]:
    if count % 4 != 0:
        raise ValidationError("subsequence path count must be divisible by four")
    if value is None:
        raise ValidationError(f"{label} registered quarter selection is required")
    selected = tuple(value)
    if (
        len(selected) != count // 4
        or len(set(selected)) != len(selected)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in selected)
        or any(not 0 <= item < count for item in selected)
    ):
        raise ValidationError(f"{label} quarter selection is invalid")
    return tuple(sorted(selected))


def _paths(value: Sequence[ArrayLike], label: str) -> tuple[FloatArray, ...]:
    if not isinstance(value, Sequence) or not value:
        raise ValidationError(f"{label} defect paths are required")
    result: list[FloatArray] = []
    horizon: int | None = None
    for raw in value:
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 3:
            raise ValidationError(f"{label} defect path geometry is invalid")
        if not bool(np.isfinite(array).all()):
            raise ValidationError(f"{label} defect paths must be finite")
        if horizon is None:
            horizon = int(array.shape[0])
        elif array.shape[0] != horizon:
            raise ValidationError(f"{label} defect path horizons differ")
        result.append(_freeze(array))
    return tuple(result)


def _defect_bundle(
    claim: str,
    development: tuple[FloatArray, ...],
    qualification: tuple[FloatArray, ...],
    oracle_name: str,
    oracle_value: float,
) -> ReturnDefectBundle:
    identity = {
        "claim_name": claim,
        "development_sha256": [_array_sha256(item) for item in development],
        "qualification_sha256": [_array_sha256(item) for item in qualification],
        "oracle_name": oracle_name,
        "oracle_value": oracle_value,
    }
    return ReturnDefectBundle(
        claim_name=claim,
        development_paths=development,
        qualification_paths=qualification,
        oracle_name=oracle_name,
        oracle_value=oracle_value,
        fingerprint=_fingerprint(identity),
    )


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    if not bool(np.isfinite(array).all()):
        raise ValidationError("defect transform produced non-finite values")
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _array_sha256(value: FloatArray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} fingerprint is invalid")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValidationError(f"{label} fingerprint is invalid") from error
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
