"""Lean observable G7 hook for the production CSV structural scan.

The hook does three jobs and no more:

* stream exact complete-path return identities for every LF monthly and HF
  daily production path;
* retain the fixed first 500 LF and 200 HF equal-history windows for direct
  return-only diagnostics; and
* compare those diagnostics with untouched historical control C by the same
  A/B-centred max-statistic construction used by lean G5.

Quarterly, annual, and weekly paths are deliberately ignored here.  Their
identities, counts, and compounding relationships remain the structural
scanner's responsibility.  This module has no publisher, CLI, generator,
hidden-state input, or release authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import IntegrityError, ValidationError
from prpg.validation.lean_g5_assessment import LeanG5SuppliedControlBank
from prpg.validation.lean_g5_control_banks import (
    LEAN_G5_CONTROL_BANK_CONTRACT_ID,
    LEAN_G5_CONTROL_BANK_ROLES,
    LEAN_G5_HF_HISTORY_LENGTH,
    LEAN_G5_LF_HISTORY_LENGTH,
    BankRole,
    LeanG5HistoricalControlDraw,
)
from prpg.validation.lean_g5_diagnostics import (
    LeanG5PathDiagnostics,
    LeanG5ReturnDiversity,
    compute_lean_g5_path_diagnostics,
    compute_lean_g5_return_diversity,
)
from prpg.validation.lean_g5_statistics import LeanG5ResampleIndices
from prpg.validation.statistics import HolmAudit, holm_decisions
from prpg.validation.structural import JSONValue, StructuralPathView

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Frequency: TypeAlias = Literal["monthly", "daily"]
Family: TypeAlias = Literal["LF", "HF"]

LEAN_G7_SCHEMA_ID: Final = "prpg-g7-lean-structural-hook-v1"
LEAN_G7_HOOK_NAME: Final = "lean_g7"
LEAN_G7_GROUPS: Final = ("G5-B", "G5-C", "G5-D")
_FREQUENCIES: Final[tuple[Frequency, Frequency]] = ("monthly", "daily")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESAMPLE_BATCH: Final = 64
_MIN_MONTHLY_DIAGNOSTIC_ROWS: Final = 13
_MIN_DAILY_DIAGNOSTIC_ROWS: Final = 253
_PRODUCTION_GEOMETRY_VALUES: Final = (
    5_000,
    1_000,
    500,
    200,
    10_000,
    600,
    12_600,
    LEAN_G5_LF_HISTORY_LENGTH,
    LEAN_G5_HF_HISTORY_LENGTH,
)

_DIAGNOSTIC_PREFIXES: Final = {
    "G5-B": (
        "covariance.",
        "correlation.",
        "raw_acf.",
        "absolute_acf.",
        "squared_acf.",
        "cross_lag.",
    ),
    "G5-C": (
        "quantile.",
        "lower_tail_mean.",
        "upper_tail_mean.",
        "taxable_muni_spread.",
        "downside_coexceedance.",
    ),
    "G5-D": (
        "annualized_log_mean.",
        "annualized_volatility.",
        "drawdown_depth.",
        "drawdown_duration_fraction.",
    ),
}


@dataclass(frozen=True, slots=True)
class LeanG7Geometry:
    """Production totals, fixed scientific subset, and equal-history sizes."""

    lf_paths: int
    hf_paths: int
    scientific_lf_paths: int
    scientific_hf_paths: int
    resamples: int
    monthly_path_length: int
    daily_path_length: int
    monthly_history_length: int
    daily_history_length: int
    mode: Literal["production", "test"]

    def __post_init__(self) -> None:
        integer_fields = (
            self.lf_paths,
            self.hf_paths,
            self.scientific_lf_paths,
            self.scientific_hf_paths,
            self.resamples,
            self.monthly_path_length,
            self.daily_path_length,
            self.monthly_history_length,
            self.daily_history_length,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in integer_fields
        ):
            raise ValidationError("lean G7 geometry values must be integers >= 2")
        if (
            self.scientific_lf_paths > self.lf_paths
            or self.scientific_hf_paths > self.hf_paths
        ):
            raise ValidationError("lean G7 scientific subset exceeds production")
        if (
            self.monthly_history_length > self.monthly_path_length
            or self.daily_history_length > self.daily_path_length
        ):
            raise ValidationError("lean G7 equal-history window exceeds a path")
        if self.monthly_history_length < _MIN_MONTHLY_DIAGNOSTIC_ROWS:
            raise ValidationError("lean G7 monthly history is too short")
        if self.daily_history_length < _MIN_DAILY_DIAGNOSTIC_ROWS:
            raise ValidationError("lean G7 daily history is too short")
        if self.mode not in ("production", "test"):
            raise ValidationError("lean G7 geometry mode is invalid")
        if self.mode == "production" and integer_fields != _PRODUCTION_GEOMETRY_VALUES:
            raise ValidationError("lean G7 production geometry must be exact")

    @property
    def canonical(self) -> bool:
        return self.mode == "production"

    @classmethod
    def production(cls) -> LeanG7Geometry:
        return cls(
            lf_paths=5_000,
            hf_paths=1_000,
            scientific_lf_paths=500,
            scientific_hf_paths=200,
            resamples=10_000,
            monthly_path_length=600,
            daily_path_length=12_600,
            monthly_history_length=LEAN_G5_LF_HISTORY_LENGTH,
            daily_history_length=LEAN_G5_HF_HISTORY_LENGTH,
            mode="production",
        )

    @classmethod
    def test(
        cls,
        *,
        lf_paths: int,
        hf_paths: int,
        scientific_lf_paths: int,
        scientific_hf_paths: int,
        resamples: int,
        monthly_path_length: int,
        daily_path_length: int,
        monthly_history_length: int,
        daily_history_length: int,
    ) -> LeanG7Geometry:
        return cls(
            lf_paths,
            hf_paths,
            scientific_lf_paths,
            scientific_hf_paths,
            resamples,
            monthly_path_length,
            daily_path_length,
            monthly_history_length,
            daily_history_length,
            "test",
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "lf_paths": self.lf_paths,
            "hf_paths": self.hf_paths,
            "scientific_lf_paths": self.scientific_lf_paths,
            "scientific_hf_paths": self.scientific_hf_paths,
            "resamples": self.resamples,
            "monthly_path_length": self.monthly_path_length,
            "daily_path_length": self.daily_path_length,
            "monthly_history_length": self.monthly_history_length,
            "daily_history_length": self.daily_history_length,
            "mode": self.mode,
        }


LEAN_G7_PRODUCTION_GEOMETRY: Final = LeanG7Geometry(
    lf_paths=5_000,
    hf_paths=1_000,
    scientific_lf_paths=500,
    scientific_hf_paths=200,
    resamples=10_000,
    monthly_path_length=600,
    daily_path_length=12_600,
    monthly_history_length=LEAN_G5_LF_HISTORY_LENGTH,
    daily_history_length=LEAN_G5_HF_HISTORY_LENGTH,
    mode="production",
)


@dataclass(frozen=True, slots=True)
class _GroupResult:
    family: str
    endpoint_count: int
    observed_statistic: float | None
    p_value: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class _RepetitionResult:
    frequency: Frequency
    production_copy_rate: float
    control_rate_limit: float
    production_maximum_multiplicity: int
    control_multiplicity_limit: int
    passed: bool


class LeanG7StructuralHook:
    """Return-only metric hook for one complete production structural scan."""

    name: Final = LEAN_G7_HOOK_NAME

    def __init__(
        self,
        control_banks: Sequence[LeanG5SuppliedControlBank],
        registered_resample_indices: LeanG5ResampleIndices,
        *,
        g5_assessment_fingerprint: str,
        g5_authority_fingerprint: str,
        geometry: LeanG7Geometry = LEAN_G7_PRODUCTION_GEOMETRY,
    ) -> None:
        if not isinstance(geometry, LeanG7Geometry):
            raise ValidationError("lean G7 geometry has the wrong type")
        self.geometry = geometry
        self._g5_assessment_fingerprint = _required_fingerprint(
            g5_assessment_fingerprint, "G5 assessment"
        )
        self._g5_authority_fingerprint = _required_fingerprint(
            g5_authority_fingerprint, "G5 authority"
        )
        self._controls, self._controls_fingerprint = _control_returns(
            control_banks, geometry
        )
        self._indices, self._indices_fingerprint = _resample_indices(
            registered_resample_indices, geometry
        )
        self._seen: dict[Family, set[int]] = {"LF": set(), "HF": set()}
        self._complete_identities: list[tuple[Family, int, str]] = []
        self._complete_counts: Counter[tuple[Family, str]] = Counter()
        self._scientific: dict[Frequency, list[FloatArray]] = {
            "monthly": [],
            "daily": [],
        }
        self._report: dict[str, JSONValue] | None = None

    def observe_path(self, path: StructuralPathView) -> None:
        """Consume monthly/daily paths; deliberately ignore derived frequencies."""

        if self._report is not None:
            raise ValidationError("lean G7 hook was already finalized")
        if not isinstance(path, StructuralPathView):
            raise ValidationError("lean G7 hook received the wrong path type")
        if path.frequency not in ("monthly", "daily"):
            return
        frequency = cast(Frequency, path.frequency)
        family: Family = "LF" if frequency == "monthly" else "HF"
        if path.family != family:
            raise IntegrityError("lean G7 base path uses the wrong family")
        expected_paths = (
            self.geometry.lf_paths if family == "LF" else self.geometry.hf_paths
        )
        if not 1 <= path.path_entity <= expected_paths:
            raise IntegrityError("lean G7 base path is outside the fixed range")
        if path.path_entity in self._seen[family]:
            raise IntegrityError("lean G7 base path was observed more than once")
        if path.path_id != f"{family}{path.path_entity:06d}":
            raise IntegrityError("lean G7 base path ID is noncanonical")

        expected_rows = (
            self.geometry.monthly_path_length
            if frequency == "monthly"
            else self.geometry.daily_path_length
        )
        values = _returns(path.log_returns, expected_rows)
        digest = _array_sha256(values)
        self._seen[family].add(path.path_entity)
        self._complete_identities.append((family, path.path_entity, digest))
        self._complete_counts[(family, digest)] += 1

        scientific_limit = (
            self.geometry.scientific_lf_paths
            if family == "LF"
            else self.geometry.scientific_hf_paths
        )
        if path.path_entity <= scientific_limit:
            history = (
                self.geometry.monthly_history_length
                if frequency == "monthly"
                else self.geometry.daily_history_length
            )
            # A fixed leading window keeps G7 equal-information with A/B/C and
            # avoids adding another evaluation RNG or tuning surface.
            self._scientific[frequency].append(_freeze(values[:history]))

    def finalize(self) -> Mapping[str, JSONValue]:
        """Return one compact canonical-JSON pass/non-pass mapping."""

        if self._report is not None:
            return dict(self._report)
        self._require_complete()

        diagnostics: dict[str, dict[Frequency, tuple[LeanG5PathDiagnostics, ...]]] = {
            "production": _diagnostics(self._scientific),
            **{
                role: _diagnostics(self._controls[role])
                for role in LEAN_G5_CONTROL_BANK_ROLES
            },
        }
        groups = tuple(
            _compare_group(
                family,
                diagnostics,
                self._indices,
                self.geometry,
            )
            for family in LEAN_G7_GROUPS
        )
        valid_p_values = all(item.p_value is not None for item in groups)
        holm = (
            holm_decisions({item.family: cast(float, item.p_value) for item in groups})
            if valid_p_values
            else None
        )
        repetition = tuple(
            _repetition_result(
                frequency,
                self._scientific[frequency],
                {
                    role: self._controls[role][frequency]
                    for role in LEAN_G5_CONTROL_BANK_ROLES
                },
            )
            for frequency in _FREQUENCIES
        )

        duplicate_copies = sum(
            count - 1 for count in self._complete_counts.values() if count > 1
        )
        reasons: list[str] = []
        for group in groups:
            if group.p_value is None:
                reasons.append(f"NON_PASS: {group.family}: {group.reason}")
        if holm is not None:
            for holm_decision in holm.decisions:
                if holm_decision.rejected:
                    reasons.append(
                        f"NON_PASS: {holm_decision.name} production observables "
                        "reject the A/B-calibrated historical null"
                    )
        if duplicate_copies:
            reasons.append(
                "NON_PASS: complete production base-path duplicates were observed"
            )
        for repeated in repetition:
            if not repeated.passed:
                reasons.append(
                    f"NON_PASS: {repeated.frequency} repeated subsequences exceed "
                    "the two-times-A/B/C-control rule"
                )
        if not reasons:
            reasons.append("PASS: lean G7 observable production checks passed")

        overall_decision = (
            "non_pass"
            if any(reason.startswith("NON_PASS:") for reason in reasons)
            else "pass"
        )
        identity_fingerprint = _fingerprint(
            [
                [family, entity, digest]
                for family, entity, digest in sorted(self._complete_identities)
            ]
        )
        report_without_fingerprint: dict[str, JSONValue] = {
            "schema_id": LEAN_G7_SCHEMA_ID,
            "geometry": self.geometry.as_dict(),
            "g5_assessment_fingerprint": self._g5_assessment_fingerprint,
            "g5_authority_fingerprint": self._g5_authority_fingerprint,
            "control_banks_fingerprint": self._controls_fingerprint,
            "resample_indices_fingerprint": self._indices_fingerprint,
            "decision": overall_decision,
            "reasons": [cast(JSONValue, reason) for reason in reasons],
            "counts": {
                "observed_lf_paths": len(self._seen["LF"]),
                "observed_hf_paths": len(self._seen["HF"]),
                "scientific_lf_paths": len(self._scientific["monthly"]),
                "scientific_hf_paths": len(self._scientific["daily"]),
                "complete_unique_paths": len(self._complete_counts),
                "complete_duplicate_copies": duplicate_copies,
            },
            "complete_path_identity_fingerprint": identity_fingerprint,
            "grouped_p_values": _group_payload(groups, holm),
            "repeated_subsequences": [
                {
                    "frequency": item.frequency,
                    "production_copy_rate": item.production_copy_rate,
                    "control_rate_limit": item.control_rate_limit,
                    "production_maximum_multiplicity": (
                        item.production_maximum_multiplicity
                    ),
                    "control_multiplicity_limit": item.control_multiplicity_limit,
                    "passed": item.passed,
                }
                for item in repetition
            ],
        }
        self._report = {
            **report_without_fingerprint,
            "fingerprint": _fingerprint(report_without_fingerprint),
        }
        return dict(self._report)

    def _require_complete(self) -> None:
        expected = {
            "LF": set(range(1, self.geometry.lf_paths + 1)),
            "HF": set(range(1, self.geometry.hf_paths + 1)),
        }
        if self._seen != expected:
            raise IntegrityError("lean G7 base-frequency stream is incomplete")
        if (
            len(self._scientific["monthly"]) != self.geometry.scientific_lf_paths
            or len(self._scientific["daily"]) != self.geometry.scientific_hf_paths
        ):
            raise IntegrityError("lean G7 scientific subset is incomplete")


def _control_returns(
    values: Sequence[LeanG5SuppliedControlBank], geometry: LeanG7Geometry
) -> tuple[dict[BankRole, dict[Frequency, tuple[FloatArray, ...]]], str]:
    if not isinstance(values, Sequence) or len(values) != 3:
        raise ValidationError("lean G7 requires exact control banks A/B/C")
    output: dict[BankRole, dict[Frequency, tuple[FloatArray, ...]]] = {}
    identities: list[object] = []
    for bank in values:
        if (
            not isinstance(bank, LeanG5SuppliedControlBank)
            or bank.bank_role not in LEAN_G5_CONTROL_BANK_ROLES
            or bank.bank_role in output
            or bank.contract_id != LEAN_G5_CONTROL_BANK_CONTRACT_ID
        ):
            raise ValidationError("lean G7 control-bank identity is invalid")
        monthly, monthly_identity = _control_frequency(
            bank.monthly_draws,
            role=bank.bank_role,
            frequency="monthly",
            paths=geometry.scientific_lf_paths,
            history=geometry.monthly_history_length,
        )
        daily, daily_identity = _control_frequency(
            bank.daily_draws,
            role=bank.bank_role,
            frequency="daily",
            paths=geometry.scientific_hf_paths,
            history=geometry.daily_history_length,
        )
        output[bank.bank_role] = {"monthly": monthly, "daily": daily}
        identities.append(
            {
                "role": bank.bank_role,
                "contract_id": bank.contract_id,
                "monthly": monthly_identity,
                "daily": daily_identity,
            }
        )
    if set(output) != set(LEAN_G5_CONTROL_BANK_ROLES):
        raise ValidationError("lean G7 control-bank roles are incomplete")
    return output, _fingerprint(identities)


def _control_frequency(
    draws: Sequence[LeanG5HistoricalControlDraw],
    *,
    role: BankRole,
    frequency: Frequency,
    paths: int,
    history: int,
) -> tuple[tuple[FloatArray, ...], list[JSONValue]]:
    if not isinstance(draws, Sequence) or len(draws) != paths:
        raise ValidationError(f"lean G7 {frequency} control count is invalid")
    returns: list[FloatArray] = []
    identity: list[JSONValue] = []
    for index, draw in enumerate(draws):
        if (
            not isinstance(draw, LeanG5HistoricalControlDraw)
            or draw.bank_role != role
            or draw.frequency != frequency
            or draw.path_index != index
            or draw.history_length != history
        ):
            raise ValidationError(f"lean G7 {frequency} control draw is invalid")
        path = _returns(draw.log_returns, history)
        returns.append(path)
        identity.append(
            {
                "path_index": index,
                "draw_fingerprint": _required_fingerprint(
                    draw.fingerprint, "control draw"
                ),
                "returns_sha256": _array_sha256(path),
            }
        )
    return tuple(returns), identity


def _resample_indices(
    value: LeanG5ResampleIndices, geometry: LeanG7Geometry
) -> tuple[dict[str, IntArray], str]:
    if not isinstance(value, LeanG5ResampleIndices):
        raise ValidationError("lean G7 resample indices have the wrong type")
    specifications = (
        ("simulated_monthly", value.simulated_monthly, geometry.scientific_lf_paths),
        ("simulated_daily", value.simulated_daily, geometry.scientific_hf_paths),
        (
            "historical_monthly",
            value.historical_monthly,
            geometry.scientific_lf_paths,
        ),
        ("historical_daily", value.historical_daily, geometry.scientific_hf_paths),
    )
    output: dict[str, IntArray] = {}
    identities: list[list[JSONValue]] = []
    for name, raw, paths in specifications:
        array = np.asarray(raw)
        if array.dtype.kind not in "iu":
            raise ValidationError("lean G7 resample indices must be integers")
        checked = np.asarray(array, dtype="<i8", order="C")
        if (
            checked.shape != (geometry.resamples, paths)
            or bool((checked < 0).any())
            or bool((checked >= paths).any())
        ):
            raise ValidationError("lean G7 resample index geometry is invalid")
        output[name] = checked
        identities.append([name, _array_sha256(checked)])
    return output, _fingerprint(identities)


def _diagnostics(
    paths: Mapping[Frequency, Sequence[FloatArray]],
) -> dict[Frequency, tuple[LeanG5PathDiagnostics, ...]]:
    return {
        frequency: tuple(
            compute_lean_g5_path_diagnostics(path, frequency=frequency)
            for path in paths[frequency]
        )
        for frequency in _FREQUENCIES
    }


def _family_matrix(rows: Sequence[LeanG5PathDiagnostics], family: str) -> FloatArray:
    if not rows:
        raise ValidationError("lean G7 diagnostics are empty")
    names = rows[0].names
    if any(row.names != names for row in rows):
        raise ValidationError("lean G7 diagnostic contracts differ")
    prefixes = _DIAGNOSTIC_PREFIXES[family]
    selected = tuple(
        index for index, name in enumerate(names) if name.startswith(prefixes)
    )
    if not selected:
        raise ValidationError("lean G7 diagnostic family has no endpoints")
    return np.asarray(
        [[row.values[index] for index in selected] for row in rows],
        dtype=np.float64,
    )


def _compare_group(
    family: str,
    diagnostics: Mapping[str, Mapping[Frequency, Sequence[LeanG5PathDiagnostics]]],
    indices: Mapping[str, IntArray],
    geometry: LeanG7Geometry,
) -> _GroupResult:
    observed_parts: list[FloatArray] = []
    null_parts: list[FloatArray] = []
    center_parts: list[FloatArray] = []
    for frequency in _FREQUENCIES:
        production = _family_matrix(diagnostics["production"][frequency], family)
        control_a = _family_matrix(diagnostics["A"][frequency], family)
        control_b = _family_matrix(diagnostics["B"][frequency], family)
        control_c = _family_matrix(diagnostics["C"][frequency], family)
        endpoint_counts = {
            production.shape[1],
            control_a.shape[1],
            control_b.shape[1],
            control_c.shape[1],
        }
        if len(endpoint_counts) != 1:
            raise ValidationError("lean G7 endpoint counts differ")
        observed_parts.append(np.mean(production, axis=0) - np.mean(control_c, axis=0))
        null_parts.append(
            _resampled_mean(control_a, indices[f"simulated_{frequency}"])
            - _resampled_mean(control_b, indices[f"historical_{frequency}"])
        )
        center_parts.append(np.mean(control_a, axis=0) - np.mean(control_b, axis=0))

    observed = np.concatenate(observed_parts)
    null = np.concatenate(null_parts, axis=1)
    center = np.concatenate(center_parts)
    centered = null - center
    scale = np.std(centered, axis=0, ddof=1, dtype=np.float64)
    if (
        null.shape != (geometry.resamples, observed.size)
        or not bool(np.isfinite(centered).all())
        or not bool(np.isfinite(scale).all())
        or bool((scale <= 1.0e-12).any())
    ):
        return _GroupResult(
            family,
            int(observed.size),
            None,
            None,
            "A/B controls cannot calibrate every endpoint",
        )
    null_statistics = np.max(np.abs(centered / scale), axis=1)
    observed_statistic = float(np.max(np.abs(observed / scale)))
    exceedances = int(np.count_nonzero(null_statistics >= observed_statistic))
    p_value = (exceedances + 1) / (geometry.resamples + 1)
    return _GroupResult(
        family,
        int(observed.size),
        observed_statistic,
        p_value,
        "A/B-centred max-statistic calibrated successfully",
    )


def _resampled_mean(values: FloatArray, indices: IntArray) -> FloatArray:
    result = np.empty((indices.shape[0], values.shape[1]), dtype=np.float64)
    for start in range(0, indices.shape[0], _RESAMPLE_BATCH):
        stop = min(start + _RESAMPLE_BATCH, indices.shape[0])
        result[start:stop] = np.mean(values[indices[start:stop]], axis=1)
    return result


def _repetition_result(
    frequency: Frequency,
    production: Sequence[FloatArray],
    controls: Mapping[BankRole, Sequence[FloatArray]],
) -> _RepetitionResult:
    observed = compute_lean_g5_return_diversity(production, frequency=frequency)
    control_audits = tuple(
        compute_lean_g5_return_diversity(controls[role], frequency=frequency)
        for role in LEAN_G5_CONTROL_BANK_ROLES
    )
    observed_rate = observed.repeated_subsequence_copies / observed.subsequence_windows
    control_rate = max(_copy_rate(item) for item in control_audits)
    rate_limit = 2.0 * control_rate
    control_excess = max(
        item.maximum_subsequence_multiplicity - 1 for item in control_audits
    )
    multiplicity_limit = 1 + 2 * control_excess
    return _RepetitionResult(
        frequency,
        observed_rate,
        rate_limit,
        observed.maximum_subsequence_multiplicity,
        multiplicity_limit,
        (
            observed_rate <= rate_limit + 1.0e-15
            and observed.maximum_subsequence_multiplicity <= multiplicity_limit
        ),
    )


def _copy_rate(value: LeanG5ReturnDiversity) -> float:
    return value.repeated_subsequence_copies / value.subsequence_windows


def _group_payload(
    groups: Sequence[_GroupResult], holm: HolmAudit | None
) -> list[JSONValue]:
    decisions = {} if holm is None else {item.name: item for item in holm.decisions}
    result: list[JSONValue] = []
    for group in groups:
        decision = decisions.get(group.family)
        result.append(
            {
                "family": group.family,
                "endpoint_count": group.endpoint_count,
                "observed_statistic": group.observed_statistic,
                "raw_p_value": group.p_value,
                "adjusted_p_value": (
                    None if decision is None else decision.adjusted_p_value
                ),
                "rejected": None if decision is None else decision.rejected,
                "reason": group.reason,
            }
        )
    return result


def _returns(value: ArrayLike, expected_rows: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype="<f8", order="C")
    except (TypeError, ValueError) as error:
        raise ValidationError("lean G7 returns must be numeric") from error
    if result.shape != (expected_rows, 3) or not bool(np.isfinite(result).all()):
        raise ValidationError("lean G7 returns have invalid geometry")
    return result


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _array_sha256(value: FloatArray | IntArray) -> str:
    normalized = np.asarray(value, order="C").copy()
    if normalized.dtype.kind == "f":
        normalized[normalized == 0.0] = 0.0
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _required_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"lean G7 {label} fingerprint is invalid")
    return value


def _fingerprint(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
