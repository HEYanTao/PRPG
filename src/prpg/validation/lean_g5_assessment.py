"""Delivery-first coordinator for the lean G5 scientific assessment.

The coordinator consumes already-generated development, qualification, and
historical-control paths.  It fits on development only, evaluates the fixed
six price-only probes and inverse-volatility timer, and reduces return/state
paths to compact grouped statistics.  It does not generate paths, choose a
policy, write evidence, expose a CLI, or grant release authority.

The repeated-subsequence count uses one deliberately small, prospective rule:
for each frequency, qualification's repeated-copy rate and excess maximum
multiplicity may be no more than twice the largest corresponding A/B/C
historical-control value.  The rule uses direct scans only, adds no resampling
campaign, and reduces to a zero-tolerance rule when all controls have zero
repetition.  Complete qualification-path duplicates remain an exact failure.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpg.errors import ValidationError
from prpg.model.g5_generation_authority import (
    G5_DEVELOPMENT_HF_PATH_COUNT,
    G5_DEVELOPMENT_LF_PATH_COUNT,
)
from prpg.validation.lean_g5_control_banks import (
    LEAN_G5_CONTROL_BANK_CONTRACT_ID,
    LEAN_G5_CONTROL_BANK_ROLES,
    LEAN_G5_HF_HISTORY_LENGTH,
    LEAN_G5_LF_HISTORY_LENGTH,
    BankRole,
    LeanG5HistoricalControlDraw,
    lean_g5_control_rng_identity,
)
from prpg.validation.lean_g5_diagnostics import (
    LeanG5PathDiagnostics,
    LeanG5ReturnDiversity,
    LeanG5StateBehavior,
    compute_lean_g5_path_diagnostics,
    compute_lean_g5_return_diversity,
    compute_lean_g5_state_behavior,
)
from prpg.validation.lean_g5_inverse_volatility import (
    LeanG5InverseVolatilityModel,
    LeanG5InverseVolatilityRows,
    fit_lean_g5_inverse_volatility_model,
    score_lean_g5_inverse_volatility_paths,
)
from prpg.validation.lean_g5_statistics import (
    HigherEmpiricalQuantile,
    LeanG5AlphaAudit,
    LeanG5AlphaRows,
    LeanG5Geometry,
    LeanG5ResampleIndices,
    PlusOnePValue,
    evaluate_lean_g5_alpha,
    four_family_holm,
    higher_empirical_quantile,
    plus_one_upper_tail_p_value,
)
from prpg.validation.lean_g5_strategies import (
    LEAN_G5_STRATEGY_NAMES,
    LeanG5StrategyModel,
    fit_lean_g5_strategy_model,
    score_lean_g5_path,
)
from prpg.validation.qualification import G5CandidateBundle, G5CandidatePathView
from prpg.validation.statistics import HolmAudit

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Decision: TypeAlias = Literal["pass", "non_pass", "ambiguous"]
Frequency: TypeAlias = Literal["monthly", "daily"]

LEAN_G5_ASSESSMENT_SCHEMA_ID: Final = "prpg-g5-lean-assessment-v1"
LEAN_G5_GROUP_NAMES: Final = ("G5-A", "G5-B", "G5-C", "G5-D")
_RESAMPLE_BATCH: Final = 64
_STATE_COUNT: Final = 4
_STATE_SUFFICIENT_COLUMNS: Final = 44

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
class LeanG5SuppliedControlBank:
    """Materialized A/B/C draws plus K4 labels from each return source.

    ``LeanG5HistoricalControlDraw`` is emitted by the existing control-bank
    primitive.  The coordinator maps the supplied source-state vectors through
    each draw's ``source_indices``; historical controls never masquerade as
    candidate path views.
    """

    bank_role: BankRole
    monthly_draws: tuple[LeanG5HistoricalControlDraw, ...]
    daily_draws: tuple[LeanG5HistoricalControlDraw, ...]
    monthly_source_states: ArrayLike
    daily_source_states: ArrayLike
    contract_id: str = LEAN_G5_CONTROL_BANK_CONTRACT_ID


@dataclass(frozen=True, slots=True)
class LeanG5GroupedComparison:
    """One A/B-calibrated grouped max-statistic comparison."""

    family: str
    endpoint_count: int
    status: Decision
    observed_statistic: float | None
    critical_value: HigherEmpiricalQuantile | None
    p_value: PlusOnePValue | None
    reason: str


@dataclass(frozen=True, slots=True)
class LeanG5StateAudit:
    """Observable K4 behavior for qualification and untouched control C."""

    qualification_monthly: LeanG5StateBehavior
    qualification_daily: LeanG5StateBehavior
    historical_monthly: LeanG5StateBehavior
    historical_daily: LeanG5StateBehavior
    calibrated_endpoint_count: int


@dataclass(frozen=True, slots=True)
class LeanG5DiversityAudit:
    """Direct return-byte duplicate and repeated-subsequence scan."""

    qualification_monthly: LeanG5ReturnDiversity
    qualification_daily: LeanG5ReturnDiversity
    historical_monthly: LeanG5ReturnDiversity
    historical_daily: LeanG5ReturnDiversity
    control_relative_limits: tuple[tuple[str, float, float, int, int], ...]
    complete_duplicates_passed: bool
    repeated_subsequence_status: Decision
    repeated_subsequence_reason: str


@dataclass(frozen=True, slots=True)
class LeanG5AssessmentReport:
    """One typed scientific result; never a generation/release authority."""

    schema_id: str
    geometry: LeanG5Geometry
    generation_policy_fingerprint: str
    development_bundle_fingerprint: str
    qualification_bundle_fingerprint: str
    control_bank_fingerprints: tuple[tuple[str, str], ...]
    control_mean_block_lengths: tuple[tuple[str, float], ...]
    model_fingerprints: tuple[tuple[str, str], ...]
    resample_indices_fingerprint: str
    alpha: LeanG5AlphaAudit
    grouped_comparisons: tuple[LeanG5GroupedComparison, ...]
    holm: HolmAudit | None
    states: LeanG5StateAudit
    diversity: LeanG5DiversityAudit
    decision: Decision
    reasons: tuple[str, ...]
    generation_authorized: Literal[False]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _PathWindows:
    returns: tuple[FloatArray, ...]
    states: tuple[IntArray, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _ControlWindows:
    bank_role: BankRole
    monthly: _PathWindows
    daily: _PathWindows
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _ComparisonParts:
    observed: FloatArray
    null: FloatArray
    center: FloatArray


class _AmbiguousComparison(Exception):
    pass


def assess_lean_g5(
    development_bundle: G5CandidateBundle,
    qualification_bundle: G5CandidateBundle,
    control_banks: Sequence[LeanG5SuppliedControlBank],
    registered_resample_indices: LeanG5ResampleIndices,
    *,
    geometry: LeanG5Geometry,
) -> LeanG5AssessmentReport:
    """Evaluate supplied paths under the bounded lean-G5 contract.

    Production requires 100/20 development paths, 500/200 qualification paths,
    three 500/200 controls, and 10,000 registered resamples.  ``pilot`` and
    ``test`` geometry alter qualification/control/resample counts only; model
    fitting always uses exactly 100 LF and 20 HF development histories.
    """

    if not isinstance(geometry, LeanG5Geometry):
        raise ValidationError("lean G5 assessment geometry has the wrong type")
    development = _bundle_windows(
        development_bundle,
        stage="development",
        monthly_paths=G5_DEVELOPMENT_LF_PATH_COUNT,
        daily_paths=G5_DEVELOPMENT_HF_PATH_COUNT,
        require_exact_marker=True,
    )
    qualification = _bundle_windows(
        qualification_bundle,
        stage="qualification",
        monthly_paths=geometry.monthly_paths,
        daily_paths=geometry.daily_paths,
        require_exact_marker=geometry.canonical,
    )
    if (
        development_bundle.generation_policy_fingerprint
        != qualification_bundle.generation_policy_fingerprint
    ):
        raise ValidationError("lean G5 development and qualification policies differ")
    if development_bundle.policy_scientific_version != 11:
        raise ValidationError("lean G5 development bundle must use policy version 11")
    qualification_version = qualification_bundle.policy_scientific_version
    if isinstance(qualification_version, bool) or not isinstance(
        qualification_version, int
    ):
        raise ValidationError("lean G5 qualification policy version is missing")
    controls = _control_windows(
        control_banks,
        geometry,
        expected_scientific_version=qualification_version,
    )
    control_block_lengths = (
        ("monthly", control_banks[0].monthly_draws[0].mean_block_length),
        ("daily", control_banks[0].daily_draws[0].mean_block_length),
    )
    resample_fingerprint = _resample_indices_fingerprint(
        registered_resample_indices, geometry
    )

    strategy_models = {
        "monthly": fit_lean_g5_strategy_model(
            development.monthly.returns, frequency="monthly"
        ),
        "daily": fit_lean_g5_strategy_model(
            development.daily.returns, frequency="daily"
        ),
    }
    inverse_models = {
        "monthly": fit_lean_g5_inverse_volatility_model(
            development.monthly.returns, frequency="monthly"
        ),
        "daily": fit_lean_g5_inverse_volatility_model(
            development.daily.returns, frequency="daily"
        ),
    }

    qualification_alpha = _alpha_rows(qualification, strategy_models)
    qualification_inverse = _inverse_rows(qualification, inverse_models)
    control_alpha = {
        role: _alpha_rows(control, strategy_models)
        for role, control in controls.items()
    }
    control_inverse = {
        role: _inverse_rows(control, inverse_models)
        for role, control in controls.items()
    }
    alpha_audit = evaluate_lean_g5_alpha(
        qualification_alpha,
        control_alpha["C"],
        registered_resample_indices,
        qualification_inverse,
        control_inverse["C"],
        geometry=geometry,
    )

    diagnostics: dict[str, dict[Frequency, tuple[LeanG5PathDiagnostics, ...]]] = {
        "qualification": _diagnostic_rows(qualification),
        **{role: _diagnostic_rows(control) for role, control in controls.items()},
    }
    state_sufficient: dict[str, dict[Frequency, FloatArray]] = {
        "qualification": _state_sufficient_rows(qualification),
        **{role: _state_sufficient_rows(control) for role, control in controls.items()},
    }

    comparisons: list[LeanG5GroupedComparison] = []
    alpha_parts = _mean_parts(
        _alpha_inverse_matrices(qualification_alpha, qualification_inverse),
        _alpha_inverse_matrices(control_alpha["C"], control_inverse["C"]),
        _alpha_inverse_matrices(control_alpha["A"], control_inverse["A"]),
        _alpha_inverse_matrices(control_alpha["B"], control_inverse["B"]),
        registered_resample_indices,
        geometry,
    )
    comparisons.append(_finish_comparison("G5-A", (alpha_parts,), geometry))

    for family in ("G5-B", "G5-C"):
        parts = _mean_parts(
            _diagnostic_family_matrices(diagnostics["qualification"], family),
            _diagnostic_family_matrices(diagnostics["C"], family),
            _diagnostic_family_matrices(diagnostics["A"], family),
            _diagnostic_family_matrices(diagnostics["B"], family),
            registered_resample_indices,
            geometry,
        )
        comparisons.append(_finish_comparison(family, (parts,), geometry))

    d_diagnostics = _mean_parts(
        _diagnostic_family_matrices(diagnostics["qualification"], "G5-D"),
        _diagnostic_family_matrices(diagnostics["C"], "G5-D"),
        _diagnostic_family_matrices(diagnostics["A"], "G5-D"),
        _diagnostic_family_matrices(diagnostics["B"], "G5-D"),
        registered_resample_indices,
        geometry,
    )
    try:
        d_states = _state_parts(
            state_sufficient["qualification"],
            state_sufficient["C"],
            state_sufficient["A"],
            state_sufficient["B"],
            registered_resample_indices,
            geometry,
        )
        comparisons.append(
            _finish_comparison("G5-D", (d_diagnostics, d_states), geometry)
        )
    except _AmbiguousComparison as error:
        comparisons.append(_ambiguous_comparison("G5-D", str(error)))

    comparison_by_name = {item.family: item for item in comparisons}
    complete_groups = all(
        item.status != "ambiguous" and item.p_value is not None for item in comparisons
    )
    holm = (
        four_family_holm(
            {
                name: comparison_by_name[name].p_value.p_value  # type: ignore[union-attr]
                for name in LEAN_G5_GROUP_NAMES
            }
        )
        if complete_groups
        else None
    )

    state_audit = LeanG5StateAudit(
        qualification_monthly=compute_lean_g5_state_behavior(
            qualification.monthly.returns, qualification.monthly.states
        ),
        qualification_daily=compute_lean_g5_state_behavior(
            qualification.daily.returns, qualification.daily.states
        ),
        historical_monthly=compute_lean_g5_state_behavior(
            controls["C"].monthly.returns, controls["C"].monthly.states
        ),
        historical_daily=compute_lean_g5_state_behavior(
            controls["C"].daily.returns, controls["C"].daily.states
        ),
        calibrated_endpoint_count=(
            2 * len(_state_endpoint_names())
            if comparison_by_name["G5-D"].status != "ambiguous"
            else 0
        ),
    )
    diversity = _diversity_audit(qualification, controls)

    reasons = _assessment_reasons(
        alpha_audit, tuple(comparisons), holm, diagnostics["qualification"], diversity
    )
    known_non_pass = any(reason.startswith("NON_PASS:") for reason in reasons)
    has_ambiguity = any(reason.startswith("AMBIGUOUS:") for reason in reasons)
    decision: Decision = (
        "non_pass" if known_non_pass else "ambiguous" if has_ambiguity else "pass"
    )
    model_fingerprints = (
        ("strategy.monthly", strategy_models["monthly"].fingerprint),
        ("strategy.daily", strategy_models["daily"].fingerprint),
        (
            "inverse_volatility.monthly",
            _inverse_model_fingerprint(inverse_models["monthly"]),
        ),
        (
            "inverse_volatility.daily",
            _inverse_model_fingerprint(inverse_models["daily"]),
        ),
    )
    control_fingerprints = tuple(
        (role, controls[role].fingerprint) for role in LEAN_G5_CONTROL_BANK_ROLES
    )
    identity = {
        "schema_id": LEAN_G5_ASSESSMENT_SCHEMA_ID,
        "geometry": _jsonable(geometry),
        "generation_policy_fingerprint": (
            qualification_bundle.generation_policy_fingerprint
        ),
        "development_bundle_fingerprint": development.fingerprint,
        "qualification_bundle_fingerprint": qualification.fingerprint,
        "control_bank_fingerprints": control_fingerprints,
        "control_mean_block_lengths": control_block_lengths,
        "model_fingerprints": model_fingerprints,
        "resample_indices_fingerprint": resample_fingerprint,
        "alpha": _jsonable(alpha_audit),
        "grouped_comparisons": _jsonable(tuple(comparisons)),
        "holm": _jsonable(holm),
        "states": _jsonable(state_audit),
        "diversity": _jsonable(diversity),
        "decision": decision,
        "reasons": reasons,
        "generation_authorized": False,
    }
    return LeanG5AssessmentReport(
        schema_id=LEAN_G5_ASSESSMENT_SCHEMA_ID,
        geometry=geometry,
        generation_policy_fingerprint=(
            qualification_bundle.generation_policy_fingerprint
        ),
        development_bundle_fingerprint=development.fingerprint,
        qualification_bundle_fingerprint=qualification.fingerprint,
        control_bank_fingerprints=control_fingerprints,
        control_mean_block_lengths=control_block_lengths,
        model_fingerprints=model_fingerprints,
        resample_indices_fingerprint=resample_fingerprint,
        alpha=alpha_audit,
        grouped_comparisons=tuple(comparisons),
        holm=holm,
        states=state_audit,
        diversity=diversity,
        decision=decision,
        reasons=reasons,
        generation_authorized=False,
        fingerprint=_fingerprint(identity),
    )


def _bundle_windows(
    bundle: object,
    *,
    stage: Literal["development", "qualification"],
    monthly_paths: int,
    daily_paths: int,
    require_exact_marker: bool,
) -> _ControlWindows:
    if not isinstance(bundle, G5CandidateBundle):
        raise ValidationError(f"lean G5 {stage} bundle has the wrong type")
    if bundle.candidate_stage != stage or bundle.generation_authorized is not False:
        raise ValidationError(f"lean G5 {stage} bundle has invalid lifecycle state")
    if require_exact_marker and not bundle.exact_canonical_counts:
        raise ValidationError(f"lean G5 {stage} bundle is not exact-count eligible")
    monthly = _path_windows(
        bundle.lf_paths,
        family="LF",
        frequency="monthly",
        expected_paths=monthly_paths,
        history_length=LEAN_G5_LF_HISTORY_LENGTH,
    )
    daily = _path_windows(
        bundle.hf_paths,
        family="HF",
        frequency="daily",
        expected_paths=daily_paths,
        history_length=LEAN_G5_HF_HISTORY_LENGTH,
    )
    identity = {
        "stage": stage,
        "generation_policy_fingerprint": bundle.generation_policy_fingerprint,
        "monthly": monthly.fingerprint,
        "daily": daily.fingerprint,
    }
    return _ControlWindows(
        bank_role="A",  # lifecycle bundles are not control banks; role is unused
        monthly=monthly,
        daily=daily,
        fingerprint=_fingerprint(identity),
    )


def _control_windows(
    values: Sequence[LeanG5SuppliedControlBank],
    geometry: LeanG5Geometry,
    *,
    expected_scientific_version: int,
) -> dict[BankRole, _ControlWindows]:
    if not isinstance(values, Sequence) or len(values) != 3:
        raise ValidationError("lean G5 assessment requires exactly control banks A/B/C")
    output: dict[BankRole, _ControlWindows] = {}
    for value in values:
        if not isinstance(value, LeanG5SuppliedControlBank):
            raise ValidationError("lean G5 supplied control bank has the wrong type")
        if (
            value.bank_role not in LEAN_G5_CONTROL_BANK_ROLES
            or value.bank_role in output
            or value.contract_id != LEAN_G5_CONTROL_BANK_CONTRACT_ID
        ):
            raise ValidationError("lean G5 supplied control bank identity is invalid")
        if (
            len({draw.source_fingerprint for draw in value.monthly_draws}) != 1
            or len({draw.source_fingerprint for draw in value.daily_draws}) != 1
        ):
            raise ValidationError("lean G5 control bank mixes historical sources")
        draw_contracts = {
            (draw.rng_identity.master_seed, draw.rng_identity.scientific_version)
            for draw in (*value.monthly_draws, *value.daily_draws)
        }
        if len(draw_contracts) != 1:
            raise ValidationError("lean G5 control bank mixes RNG policy streams")
        if next(iter(draw_contracts))[1] != expected_scientific_version:
            raise ValidationError(
                "lean G5 control bank policy version differs from qualification"
            )
        monthly = _control_path_windows(
            value.monthly_draws,
            value.monthly_source_states,
            bank_role=value.bank_role,
            frequency="monthly",
            expected_paths=geometry.monthly_paths,
            history_length=LEAN_G5_LF_HISTORY_LENGTH,
        )
        daily = _control_path_windows(
            value.daily_draws,
            value.daily_source_states,
            bank_role=value.bank_role,
            frequency="daily",
            expected_paths=geometry.daily_paths,
            history_length=LEAN_G5_HF_HISTORY_LENGTH,
        )
        identity = {
            "contract_id": value.contract_id,
            "bank_role": value.bank_role,
            "monthly": monthly.fingerprint,
            "daily": daily.fingerprint,
        }
        output[value.bank_role] = _ControlWindows(
            bank_role=value.bank_role,
            monthly=monthly,
            daily=daily,
            fingerprint=_fingerprint(identity),
        )
    if set(output) != set(LEAN_G5_CONTROL_BANK_ROLES):
        raise ValidationError("lean G5 supplied control bank roles are incomplete")
    monthly_sources = {value.monthly_draws[0].source_fingerprint for value in values}
    daily_sources = {value.daily_draws[0].source_fingerprint for value in values}
    monthly_block_lengths = {
        draw.mean_block_length for value in values for draw in value.monthly_draws
    }
    daily_block_lengths = {
        draw.mean_block_length for value in values for draw in value.daily_draws
    }
    rng_contracts = {
        (
            value.monthly_draws[0].rng_identity.master_seed,
            value.monthly_draws[0].rng_identity.scientific_version,
            value.daily_draws[0].rng_identity.master_seed,
            value.daily_draws[0].rng_identity.scientific_version,
        )
        for value in values
    }
    monthly_state_sources = {
        _array_sha256(np.asarray(value.monthly_source_states, dtype="<i8"))
        for value in values
    }
    daily_state_sources = {
        _array_sha256(np.asarray(value.daily_source_states, dtype="<i8"))
        for value in values
    }
    if (
        len(monthly_sources) != 1
        or len(daily_sources) != 1
        or len(monthly_block_lengths) != 1
        or len(daily_block_lengths) != 1
        or len(rng_contracts) != 1
        or len(monthly_state_sources) != 1
        or len(daily_state_sources) != 1
    ):
        raise ValidationError(
            "lean G5 A/B/C controls do not share source, block, and RNG contracts"
        )
    return output


def _control_path_windows(
    draws: Sequence[LeanG5HistoricalControlDraw],
    source_states: ArrayLike,
    *,
    bank_role: BankRole,
    frequency: Frequency,
    expected_paths: int,
    history_length: int,
) -> _PathWindows:
    if not isinstance(draws, Sequence) or len(draws) != expected_paths:
        raise ValidationError(
            f"lean G5 {frequency} control count differs from registered geometry"
        )
    raw_states = np.asarray(source_states)
    if (
        raw_states.ndim != 1
        or raw_states.dtype.kind not in "iu"
        or raw_states.size < 2
        or bool((raw_states < 0).any())
        or bool((raw_states >= _STATE_COUNT).any())
    ):
        raise ValidationError(
            f"lean G5 {frequency} control source states are not K4 labels"
        )
    checked_states = np.asarray(raw_states, dtype=np.int64)
    returns: list[FloatArray] = []
    states: list[IntArray] = []
    identities: list[dict[str, object]] = []
    for expected_index, draw in enumerate(draws):
        if (
            not isinstance(draw, LeanG5HistoricalControlDraw)
            or draw.bank_role != bank_role
            or draw.frequency != frequency
            or draw.path_index != expected_index
            or draw.history_length != history_length
            or not math.isfinite(draw.mean_block_length)
            or draw.mean_block_length <= 0.0
        ):
            raise ValidationError(
                f"lean G5 {frequency} control draw identity is invalid"
            )
        expected_rng = lean_g5_control_rng_identity(
            bank_role=bank_role,
            frequency=frequency,
            path_index=expected_index,
            master_seed=draw.rng_identity.master_seed,
            scientific_version=draw.rng_identity.scientific_version,
        )
        if draw.rng_identity != expected_rng:
            raise ValidationError(
                f"lean G5 {frequency} control RNG identity is not registered"
            )
        path = _returns(draw.log_returns, history_length)
        if path.shape[0] != history_length:
            raise ValidationError(
                f"lean G5 {frequency} control draw has the wrong history"
            )
        source_indices = np.asarray(draw.source_indices)
        if (
            source_indices.shape != (history_length,)
            or source_indices.dtype.kind not in "iu"
            or bool((source_indices < 0).any())
            or bool((source_indices >= checked_states.size).any())
        ):
            raise ValidationError(
                f"lean G5 {frequency} control source indices are invalid"
            )
        return_window = _freeze(path)
        state_window = _freeze_int(checked_states[source_indices])
        returns.append(return_window)
        states.append(state_window)
        identities.append(
            {
                "path_index": expected_index,
                "draw_fingerprint": draw.fingerprint,
                "rng_identity_fingerprint": draw.rng_identity.fingerprint,
                "returns_sha256": _array_sha256(return_window),
                "states_sha256": _array_sha256(state_window),
            }
        )
    return _PathWindows(
        returns=tuple(returns),
        states=tuple(states),
        fingerprint=_fingerprint(
            {
                "bank_role": bank_role,
                "frequency": frequency,
                "history_length": history_length,
                "source_states_sha256": _array_sha256(checked_states),
                "draws": identities,
            }
        ),
    )


def _path_windows(
    values: Sequence[G5CandidatePathView],
    *,
    family: Literal["LF", "HF"],
    frequency: Frequency,
    expected_paths: int,
    history_length: int,
) -> _PathWindows:
    if not isinstance(values, Sequence) or len(values) != expected_paths:
        raise ValidationError(
            f"lean G5 {frequency} path count differs from the registered geometry"
        )
    returns: list[FloatArray] = []
    states: list[IntArray] = []
    identities: list[dict[str, object]] = []
    for ordinal, value in enumerate(values, start=1):
        if not isinstance(value, G5CandidatePathView) or value.family != family:
            raise ValidationError(f"lean G5 {frequency} path view is invalid")
        path = _returns(value.log_returns, history_length)
        state = _states(value.target_states, path.shape[0])
        choices = path.shape[0] - history_length + 1
        if choices > 1:
            if not value.registered_evaluation_window:
                raise ValidationError(
                    f"lean G5 {frequency} long path lacks a registered window"
                )
            uniform = _unit_interval(value.evaluation_uniform)
            offset = min(choices - 1, math.floor(uniform * choices))
        else:
            offset = 0
        return_window = _freeze(path[offset : offset + history_length])
        state_window = _freeze_int(state[offset : offset + history_length])
        returns.append(return_window)
        states.append(state_window)
        identities.append(
            {
                "ordinal": ordinal,
                "path_id": value.path_id,
                "offset": offset,
                "returns_sha256": _array_sha256(return_window),
                "states_sha256": _array_sha256(state_window),
            }
        )
    return _PathWindows(
        returns=tuple(returns),
        states=tuple(states),
        fingerprint=_fingerprint(
            {
                "frequency": frequency,
                "history_length": history_length,
                "paths": identities,
            }
        ),
    )


def _alpha_rows(
    windows: _ControlWindows,
    models: Mapping[str, LeanG5StrategyModel],
) -> LeanG5AlphaRows:
    return LeanG5AlphaRows(
        monthly=_score_alpha(models["monthly"], windows.monthly.returns),
        daily=_score_alpha(models["daily"], windows.daily.returns),
    )


def _score_alpha(model: LeanG5StrategyModel, paths: Sequence[ArrayLike]) -> FloatArray:
    rows = np.vstack(
        tuple(score_lean_g5_path(model, path).alpha_sharpes for path in paths)
    )
    expected = (len(paths), len(LEAN_G5_STRATEGY_NAMES))
    if rows.shape != expected or not bool(np.isfinite(rows).all()):
        raise ValidationError("lean G5 strategy scoring returned invalid rows")
    return _freeze(rows)


def _inverse_rows(
    windows: _ControlWindows,
    models: Mapping[str, LeanG5InverseVolatilityModel],
) -> dict[str, LeanG5InverseVolatilityRows]:
    return {
        "monthly": score_lean_g5_inverse_volatility_paths(
            models["monthly"], windows.monthly.returns
        ),
        "daily": score_lean_g5_inverse_volatility_paths(
            models["daily"], windows.daily.returns
        ),
    }


def _alpha_inverse_matrices(
    alpha: LeanG5AlphaRows,
    inverse: Mapping[str, LeanG5InverseVolatilityRows],
) -> dict[Frequency, FloatArray]:
    return {
        "monthly": np.column_stack(
            (
                np.asarray(alpha.monthly, dtype=np.float64),
                np.asarray(inverse["monthly"].improvements, dtype=np.float64),
            )
        ),
        "daily": np.column_stack(
            (
                np.asarray(alpha.daily, dtype=np.float64),
                np.asarray(inverse["daily"].improvements, dtype=np.float64),
            )
        ),
    }


def _diagnostic_rows(
    windows: _ControlWindows,
) -> dict[Frequency, tuple[LeanG5PathDiagnostics, ...]]:
    return {
        "monthly": tuple(
            compute_lean_g5_path_diagnostics(path, frequency="monthly")
            for path in windows.monthly.returns
        ),
        "daily": tuple(
            compute_lean_g5_path_diagnostics(path, frequency="daily")
            for path in windows.daily.returns
        ),
    }


def _diagnostic_family_matrices(
    diagnostics: Mapping[Frequency, tuple[LeanG5PathDiagnostics, ...]],
    family: str,
) -> dict[Frequency, FloatArray]:
    prefixes = _DIAGNOSTIC_PREFIXES.get(family)
    if prefixes is None:
        raise ValidationError("lean G5 diagnostic family is invalid")
    output: dict[Frequency, FloatArray] = {}
    for frequency in ("monthly", "daily"):
        rows = diagnostics[frequency]
        names = rows[0].names
        if any(row.names != names for row in rows):
            raise ValidationError("lean G5 diagnostic row contracts differ")
        selected = tuple(
            index for index, name in enumerate(names) if name.startswith(prefixes)
        )
        if not selected:
            raise ValidationError("lean G5 diagnostic family has no endpoints")
        output[frequency] = np.asarray(
            [[row.values[index] for index in selected] for row in rows],
            dtype=np.float64,
        )
    return output


def _mean_parts(
    qualification: Mapping[Frequency, FloatArray],
    historical: Mapping[Frequency, FloatArray],
    bank_a: Mapping[Frequency, FloatArray],
    bank_b: Mapping[Frequency, FloatArray],
    indices: LeanG5ResampleIndices,
    geometry: LeanG5Geometry,
) -> _ComparisonParts:
    observed: list[FloatArray] = []
    null: list[FloatArray] = []
    center: list[FloatArray] = []
    groups: tuple[tuple[Frequency, int, ArrayLike, ArrayLike], ...] = (
        (
            "monthly",
            geometry.monthly_paths,
            indices.simulated_monthly,
            indices.historical_monthly,
        ),
        (
            "daily",
            geometry.daily_paths,
            indices.simulated_daily,
            indices.historical_daily,
        ),
    )
    for frequency, paths, sim_indices, hist_indices in groups:
        q = _matrix(qualification[frequency], paths, "qualification")
        c = _matrix(historical[frequency], paths, "historical C")
        a = _matrix(bank_a[frequency], paths, "historical A")
        b = _matrix(bank_b[frequency], paths, "historical B")
        if len({q.shape[1], c.shape[1], a.shape[1], b.shape[1]}) != 1:
            raise ValidationError("lean G5 comparison endpoint counts differ")
        sim = _indices(sim_indices, geometry.resamples, paths)
        hist = _indices(hist_indices, geometry.resamples, paths)
        observed.append(np.mean(q, axis=0) - np.mean(c, axis=0))
        null.append(_resampled_mean(a, sim) - _resampled_mean(b, hist))
        center.append(np.mean(a, axis=0) - np.mean(b, axis=0))
    return _ComparisonParts(
        observed=np.concatenate(observed),
        null=np.concatenate(null, axis=1),
        center=np.concatenate(center),
    )


def _state_sufficient_rows(
    windows: _ControlWindows,
) -> dict[Frequency, FloatArray]:
    return {
        "monthly": _state_sufficient(windows.monthly.returns, windows.monthly.states),
        "daily": _state_sufficient(windows.daily.returns, windows.daily.states),
    }


def _state_sufficient(
    returns: Sequence[FloatArray], states: Sequence[IntArray]
) -> FloatArray:
    rows = np.zeros((len(returns), _STATE_SUFFICIENT_COLUMNS), dtype=np.float64)
    for row, (path, state) in enumerate(zip(returns, states, strict=True)):
        occupancy = np.bincount(state, minlength=_STATE_COUNT).astype(np.float64)
        transitions = np.zeros((_STATE_COUNT, _STATE_COUNT), dtype=np.float64)
        np.add.at(transitions, (state[:-1], state[1:]), 1.0)
        boundaries = np.flatnonzero(state[1:] != state[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), boundaries))
        stops = np.concatenate((boundaries, np.asarray([state.size])))
        episodes = np.zeros(_STATE_COUNT, dtype=np.float64)
        dwell = np.zeros(_STATE_COUNT, dtype=np.float64)
        one = np.zeros(_STATE_COUNT, dtype=np.float64)
        for start, stop in zip(starts, stops, strict=True):
            label = int(state[start])
            length = int(stop - start)
            episodes[label] += 1.0
            dwell[label] += length
            one[label] += float(length == 1)
        return_sum = np.zeros((_STATE_COUNT, 3), dtype=np.float64)
        for label in range(_STATE_COUNT):
            return_sum[label] = np.sum(path[state == label], axis=0, dtype=np.float64)
        rows[row] = np.concatenate(
            (
                occupancy,
                transitions.ravel(),
                episodes,
                dwell,
                one,
                return_sum.ravel(),
            )
        )
    return rows


def _state_parts(
    qualification: Mapping[Frequency, FloatArray],
    historical: Mapping[Frequency, FloatArray],
    bank_a: Mapping[Frequency, FloatArray],
    bank_b: Mapping[Frequency, FloatArray],
    indices: LeanG5ResampleIndices,
    geometry: LeanG5Geometry,
) -> _ComparisonParts:
    observed: list[FloatArray] = []
    null: list[FloatArray] = []
    center: list[FloatArray] = []
    groups: tuple[tuple[Frequency, int, ArrayLike, ArrayLike], ...] = (
        (
            "monthly",
            geometry.monthly_paths,
            indices.simulated_monthly,
            indices.historical_monthly,
        ),
        (
            "daily",
            geometry.daily_paths,
            indices.simulated_daily,
            indices.historical_daily,
        ),
    )
    for frequency, paths, sim_indices, hist_indices in groups:
        q = _matrix(qualification[frequency], paths, "qualification state")
        c = _matrix(historical[frequency], paths, "historical C state")
        a = _matrix(bank_a[frequency], paths, "historical A state")
        b = _matrix(bank_b[frequency], paths, "historical B state")
        sim = _indices(sim_indices, geometry.resamples, paths)
        hist = _indices(hist_indices, geometry.resamples, paths)
        try:
            q_value = _state_endpoints(np.sum(q, axis=0, keepdims=True))[0]
            c_value = _state_endpoints(np.sum(c, axis=0, keepdims=True))[0]
            a_value = _state_endpoints(np.sum(a, axis=0, keepdims=True))[0]
            b_value = _state_endpoints(np.sum(b, axis=0, keepdims=True))[0]
            null_value = _resampled_state_endpoints(
                a, sim
            ) - _resampled_state_endpoints(b, hist)
        except _AmbiguousComparison as error:
            raise _AmbiguousComparison(
                f"K4 {frequency} comparison is undefined: {error}"
            ) from error
        observed.append(q_value - c_value)
        null.append(null_value)
        center.append(a_value - b_value)
    return _ComparisonParts(
        observed=np.concatenate(observed),
        null=np.concatenate(null, axis=1),
        center=np.concatenate(center),
    )


def _state_endpoints(sums: FloatArray) -> FloatArray:
    occupancy = sums[:, 0:4]
    transitions = sums[:, 4:20].reshape(-1, 4, 4)
    episodes = sums[:, 20:24]
    dwell = sums[:, 24:28]
    one = sums[:, 28:32]
    return_sum = sums[:, 32:44].reshape(-1, 4, 3)
    total = np.sum(occupancy, axis=1, keepdims=True)
    departures = np.sum(transitions, axis=2)
    if (
        bool((total <= 0.0).any())
        or bool((occupancy <= 0.0).any())
        or bool((departures <= 0.0).any())
        or bool((episodes <= 0.0).any())
    ):
        raise _AmbiguousComparison(
            "at least one aggregate sample lacks support for a K4 state"
        )
    result = np.asarray(
        np.concatenate(
            (
                occupancy / total,
                (transitions / departures[:, :, np.newaxis]).reshape(-1, 16),
                dwell / episodes,
                one / episodes,
                (return_sum / occupancy[:, :, np.newaxis]).reshape(-1, 12),
            ),
            axis=1,
        ),
        dtype=np.float64,
    )
    if not bool(np.isfinite(result).all()):
        raise _AmbiguousComparison("K4 endpoint transform is non-finite")
    return result


def _state_endpoint_names() -> tuple[str, ...]:
    return (
        *(f"occupancy.{state}" for state in range(4)),
        *(f"transition.{left}.{right}" for left in range(4) for right in range(4)),
        *(f"mean_dwell.{state}" for state in range(4)),
        *(f"one_period_fraction.{state}" for state in range(4)),
        *(
            f"conditional_mean.{state}.{asset}"
            for state in range(4)
            for asset in range(3)
        ),
    )


def _resampled_state_endpoints(values: FloatArray, indices: IntArray) -> FloatArray:
    result = np.empty(
        (indices.shape[0], len(_state_endpoint_names())), dtype=np.float64
    )
    for start in range(0, indices.shape[0], _RESAMPLE_BATCH):
        stop = min(start + _RESAMPLE_BATCH, indices.shape[0])
        sums = np.asarray(
            np.sum(values[indices[start:stop]], axis=1, dtype=np.float64),
            dtype=np.float64,
        )
        result[start:stop] = _state_endpoints(sums)
    return result


def _resampled_mean(values: FloatArray, indices: IntArray) -> FloatArray:
    result = np.empty((indices.shape[0], values.shape[1]), dtype=np.float64)
    for start in range(0, indices.shape[0], _RESAMPLE_BATCH):
        stop = min(start + _RESAMPLE_BATCH, indices.shape[0])
        result[start:stop] = np.mean(values[indices[start:stop]], axis=1)
    return result


def _finish_comparison(
    family: str,
    parts: Sequence[_ComparisonParts],
    geometry: LeanG5Geometry,
) -> LeanG5GroupedComparison:
    observed = np.concatenate(tuple(part.observed for part in parts))
    null = np.concatenate(tuple(part.null for part in parts), axis=1)
    center = np.concatenate(tuple(part.center for part in parts))
    # A* - B* is a conditional bootstrap and is therefore centered by its
    # realized A-B difference.  Q-C is a new independent same-geometry
    # comparison whose null target is already zero; subtracting A-B from Q-C
    # would add an unrelated finite-bank realization rather than remove bias.
    centered = null - center
    scale = np.std(centered, axis=0, ddof=1, dtype=np.float64)
    if (
        observed.size == 0
        or null.shape != (geometry.resamples, observed.size)
        or scale.shape != observed.shape
    ):
        raise ValidationError("lean G5 grouped comparison geometry is invalid")
    if not bool(np.isfinite(centered).all()) or not bool(np.isfinite(scale).all()):
        return _ambiguous_comparison(family, "A/B null contains non-finite values")
    if bool((scale <= 1.0e-12).any()):
        count = int(np.count_nonzero(scale <= 1.0e-12))
        return _ambiguous_comparison(
            family,
            f"A/B null cannot calibrate {count} degenerate endpoint(s)",
        )
    null_statistic = np.max(np.abs(centered / scale), axis=1)
    observed_statistic = float(np.max(np.abs(observed / scale)))
    critical = higher_empirical_quantile(null_statistic, 0.95, geometry=geometry)
    p_value = plus_one_upper_tail_p_value(
        null_statistic, observed_statistic, geometry=geometry
    )
    status: Decision = "pass" if p_value.p_value > 0.05 else "non_pass"
    return LeanG5GroupedComparison(
        family=family,
        endpoint_count=int(observed.size),
        status=status,
        observed_statistic=observed_statistic,
        critical_value=critical,
        p_value=p_value,
        reason=(
            "grouped A/B-calibrated statistic is above the 5% empirical tail"
            if status == "non_pass"
            else "grouped A/B-calibrated statistic is inside the empirical null"
        ),
    )


def _ambiguous_comparison(family: str, reason: str) -> LeanG5GroupedComparison:
    return LeanG5GroupedComparison(
        family=family,
        endpoint_count=0,
        status="ambiguous",
        observed_statistic=None,
        critical_value=None,
        p_value=None,
        reason=reason,
    )


def _diversity_audit(
    qualification: _ControlWindows,
    controls: Mapping[BankRole, _ControlWindows],
) -> LeanG5DiversityAudit:
    q_monthly = compute_lean_g5_return_diversity(
        qualification.monthly.returns, frequency="monthly"
    )
    q_daily = compute_lean_g5_return_diversity(
        qualification.daily.returns, frequency="daily"
    )
    control_audits = {
        role: {
            "monthly": compute_lean_g5_return_diversity(
                control.monthly.returns, frequency="monthly"
            ),
            "daily": compute_lean_g5_return_diversity(
                control.daily.returns, frequency="daily"
            ),
        }
        for role, control in controls.items()
    }
    h_monthly = control_audits["C"]["monthly"]
    h_daily = control_audits["C"]["daily"]
    duplicate_copies = (
        q_monthly.complete_path_duplicate_copies
        + q_daily.complete_path_duplicate_copies
    )
    repeated_passed = True
    limits: list[tuple[str, float, float, int, int]] = []
    for frequency, qualification_audit in (
        ("monthly", q_monthly),
        ("daily", q_daily),
    ):
        historical_audits = tuple(
            control_audits[role][frequency] for role in LEAN_G5_CONTROL_BANK_ROLES
        )
        qualification_rate = (
            qualification_audit.repeated_subsequence_copies
            / qualification_audit.subsequence_windows
        )
        maximum_control_rate = max(
            item.repeated_subsequence_copies / item.subsequence_windows
            for item in historical_audits
        )
        rate_limit = 2.0 * maximum_control_rate
        maximum_control_excess = max(
            item.maximum_subsequence_multiplicity - 1 for item in historical_audits
        )
        multiplicity_limit = 1 + 2 * maximum_control_excess
        frequency_passed = (
            qualification_rate <= rate_limit + 1.0e-15
            and qualification_audit.maximum_subsequence_multiplicity
            <= multiplicity_limit
        )
        repeated_passed = repeated_passed and frequency_passed
        limits.append(
            (
                frequency,
                qualification_rate,
                rate_limit,
                qualification_audit.maximum_subsequence_multiplicity,
                multiplicity_limit,
            )
        )
    repeated_status: Decision = "pass" if repeated_passed else "non_pass"
    return LeanG5DiversityAudit(
        qualification_monthly=q_monthly,
        qualification_daily=q_daily,
        historical_monthly=h_monthly,
        historical_daily=h_daily,
        control_relative_limits=tuple(limits),
        complete_duplicates_passed=duplicate_copies == 0,
        repeated_subsequence_status=repeated_status,
        repeated_subsequence_reason=(
            "qualification repetition is within twice the largest A/B/C "
            "control excess at both frequencies"
            if repeated_passed
            else (
                "qualification repetition exceeds the prospective "
                "two-times-A/B/C-control rule"
            )
        ),
    )


def _assessment_reasons(
    alpha: LeanG5AlphaAudit,
    comparisons: tuple[LeanG5GroupedComparison, ...],
    holm: HolmAudit | None,
    qualification_diagnostics: Mapping[Frequency, tuple[LeanG5PathDiagnostics, ...]],
    diversity: LeanG5DiversityAudit,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not alpha.equivalence.passed:
        reasons.append("NON_PASS: G5-A direct alpha equivalence failed")
    if not alpha.leakage_gap.passed:
        reasons.append("NON_PASS: G5-A simulated-minus-historical alpha gap failed")
    if not alpha.inverse_volatility.passed:
        reasons.append("NON_PASS: G5-A inverse-volatility improvement gap failed")
    for comparison in comparisons:
        if comparison.status == "ambiguous":
            reasons.append(f"AMBIGUOUS: {comparison.family}: {comparison.reason}")
    if holm is not None:
        rejected = tuple(item.name for item in holm.decisions if item.rejected)
        if rejected:
            reasons.append(
                "NON_PASS: Holm rejected grouped family/families " + ", ".join(rejected)
            )
    constant_paths = sum(
        any(
            name.startswith("constant_asset.") and value != 0.0
            for name, value in zip(row.names, row.values, strict=True)
        )
        for rows in qualification_diagnostics.values()
        for row in rows
    )
    if constant_paths:
        reasons.append(
            f"NON_PASS: {constant_paths} qualification path(s) contain a constant asset"
        )
    if not diversity.complete_duplicates_passed:
        reasons.append("NON_PASS: exact duplicate qualification return paths observed")
    if diversity.repeated_subsequence_status == "non_pass":
        reasons.append("NON_PASS: " + diversity.repeated_subsequence_reason)
    if not reasons:
        reasons.append("PASS: all implemented lean G5 assessment controls passed")
    return tuple(reasons)


def _resample_indices_fingerprint(value: object, geometry: LeanG5Geometry) -> str:
    if not isinstance(value, LeanG5ResampleIndices):
        raise ValidationError("lean G5 registered resample indices have the wrong type")
    specifications = (
        ("simulated_monthly", value.simulated_monthly, geometry.monthly_paths),
        ("simulated_daily", value.simulated_daily, geometry.daily_paths),
        ("historical_monthly", value.historical_monthly, geometry.monthly_paths),
        ("historical_daily", value.historical_daily, geometry.daily_paths),
    )
    matrices = tuple(
        (
            name,
            _array_sha256(_indices(matrix, geometry.resamples, paths)),
        )
        for name, matrix, paths in specifications
    )
    return _fingerprint(
        {
            "geometry": (
                geometry.monthly_paths,
                geometry.daily_paths,
                geometry.resamples,
                geometry.mode,
            ),
            "matrices": matrices,
        }
    )


def _matrix(value: ArrayLike, rows: int, label: str) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"lean G5 {label} matrix must be numeric") from error
    if (
        result.ndim != 2
        or result.shape[0] != rows
        or result.shape[1] == 0
        or not bool(np.isfinite(result).all())
    ):
        raise ValidationError(f"lean G5 {label} matrix has invalid geometry")
    return result


def _indices(value: ArrayLike, resamples: int, paths: int) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValidationError("lean G5 registered resample indices are not integers")
    result = np.asarray(raw, dtype=np.int64)
    if (
        result.shape != (resamples, paths)
        or bool((result < 0).any())
        or bool((result >= paths).any())
    ):
        raise ValidationError("lean G5 registered resample indices are invalid")
    return result


def _returns(value: ArrayLike, minimum_rows: int) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValidationError("lean G5 path returns must be numeric") from error
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or result.shape[0] < minimum_rows
        or not bool(np.isfinite(result).all())
    ):
        raise ValidationError("lean G5 path returns have invalid geometry")
    return result


def _states(value: ArrayLike, observations: int) -> IntArray:
    raw = np.asarray(value)
    if (
        raw.shape != (observations,)
        or raw.dtype.kind not in "iu"
        or bool((raw < 0).any())
        or bool((raw >= _STATE_COUNT).any())
    ):
        raise ValidationError("lean G5 path states are not an aligned K4 sequence")
    return np.asarray(raw, dtype=np.int64)


def _unit_interval(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError("lean G5 evaluation-window uniform is invalid")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValidationError("lean G5 evaluation-window uniform is invalid")
    return result


def _inverse_model_fingerprint(model: LeanG5InverseVolatilityModel) -> str:
    return _fingerprint(
        {
            "frequency": model.frequency,
            "lookback_periods": model.lookback_periods,
            "rebalance_periods": model.rebalance_periods,
            "annualization": model.annualization,
            "comparator_weights_sha256": _array_sha256(model.comparator_weights),
            "development_paths": model.development_paths,
            "development_weight_rows": model.development_weight_rows,
        }
    )


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_jsonable(item) for item in value)
    raise TypeError(f"lean G5 report cannot serialize {type(value).__name__}")


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype="<f8", order="C").copy()
    array[array == 0.0] = 0.0
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _freeze_int(value: ArrayLike) -> IntArray:
    array = np.asarray(value, dtype="<i8", order="C").copy()
    return np.frombuffer(array.tobytes(order="C"), dtype="<i8").reshape(array.shape)


def _array_sha256(value: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


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
