"""Small, deterministic, report-only diagnostics for canonical v5 returns.

It adds no threshold or decision, and uses the deterministic materializer
already proven equal to canonical CSVs. Hidden data stay in an audit section.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from prpg.errors import IntegrityError, ValidationError
from prpg.v5.config import ResolvedV5Inputs
from prpg.v5.data import load_v5_month_library
from prpg.v5.generation import (
    load_v5_generation_input,
    materialize_v5_path,
)
from prpg.validation.lean_g5_diagnostics import (
    compute_lean_g5_path_diagnostics,
    compute_lean_g5_return_diversity,
    compute_lean_g5_state_behavior,
)
from prpg.validation.lean_g5_strategies import (
    LEAN_G5_STRATEGY_NAMES,
    LeanG5StrategyModel,
    fit_lean_g5_strategy_model,
    lean_g5_strategy_weights,
)

FloatArray: TypeAlias = NDArray[np.float64]
Frequency: TypeAlias = Literal["daily", "monthly"]
SCHEMA: Final = "prpg-v5-report-only-diagnostics-v1"
FILENAME: Final = "diagnostics-report.json"
DEVELOPMENT_IDS: Final = tuple(range(1, 13))
ASSESSMENT_IDS: Final = (
    *range(51, 63),
    *range(4_001, 4_007),
    *range(4_501, 4_507),
)
STATE_LABELS: Final = (
    "persistent high-volatility stress/rebound",
    "broad bond selloff/rate pressure",
    "steady positive-return/normal market",
    "strong bond rally/mixed-equity relief",
)


@dataclass(frozen=True, slots=True)
class V5DiagnosticsResult:
    report_path: Path
    status: Literal["diagnostic_report_only"]

    def as_dict(self) -> dict[str, object]:
        return {
            "report_path": str(self.report_path),
            "status": self.status,
            "release_gate": False,
            "development_paths": len(DEVELOPMENT_IDS),
            "assessment_paths": len(ASSESSMENT_IDS),
        }


def run_v5_report_diagnostics(
    inputs: ResolvedV5Inputs, run_root: str | Path
) -> V5DiagnosticsResult:
    """Write one bounded descriptive report beside the canonical manifest."""

    root = _root(run_root)
    manifest = _manifest(root)
    processed = _fingerprint(manifest.get("processed_data_fingerprint"))
    model_fingerprint = _fingerprint(manifest.get("model_fingerprint"))
    generation = load_v5_generation_input(inputs, processed, model_fingerprint)
    horizon = inputs.config.simulation.horizon_months
    paths = {
        path_id: materialize_v5_path(generation, path_id, horizon)
        for path_id in (*DEVELOPMENT_IDS, *ASSESSMENT_IDS)
    }
    development = tuple(
        paths[path_id].monthly_log_returns for path_id in DEVELOPMENT_IDS
    )
    assessed = tuple(paths[path_id] for path_id in ASSESSMENT_IDS)
    assessed_monthly = tuple(path.monthly_log_returns for path in assessed)
    assessed_daily = tuple(path.daily_log_returns for path in assessed)
    library = load_v5_month_library(inputs, processed)
    historical_monthly = np.asarray(library.monthly_returns[:, (0, 2, 1)])
    historical_daily = np.asarray(library.daily_returns[:, (0, 2, 1)])
    states = tuple(path.selection.states for path in assessed)
    sources = tuple(path.selection.source_month_indices for path in assessed)
    state_report = asdict(compute_lean_g5_state_behavior(assessed_monthly, states))
    state_report["labels"] = list(STATE_LABELS)

    report: dict[str, object] = {
        "schema_id": SCHEMA,
        "status": "diagnostic_report_only",
        "release_gate": False,
        "decision": None,
        "scope": {
            "canonical_paths": 5_000,
            "development_path_ids": list(DEVELOPMENT_IDS),
            "assessment_path_ids": list(ASSESSMENT_IDS),
            "assessment_roles": {
                "training": list(range(51, 63)),
                "validation": list(range(4_001, 4_007)),
                "test": list(range(4_501, 4_507)),
            },
            "sample_basis": (
                "fixed IDs; deterministic replay previously proven byte-equal "
                "to the fully validated canonical consumer CSVs"
            ),
            "statistical_claim": "bounded descriptive convenience sample only",
            "historical_control_period": {
                "first": _month(int(library.monthly_period_ordinals[0])),
                "last": _month(int(library.monthly_period_ordinals[-1])),
            },
        },
        "provenance": {
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "run_fingerprint": manifest["run_fingerprint"],
            "processed_data_fingerprint": processed,
            "model_fingerprint": model_fingerprint,
            "config_fingerprint": inputs.config_fingerprint,
        },
        "consumer_observable_diagnostics": {
            "monthly": _observables(assessed_monthly, historical_monthly, "monthly"),
            "daily": _observables(assessed_daily, historical_daily, "daily"),
        },
        "return_diversity": {
            **asdict(
                compute_lean_g5_return_diversity(
                    assessed_monthly, frequency="monthly", subsequence_window=6
                )
            ),
            "note": "exact six-month consumer-return windows; report-only",
        },
        "audit_only_state_and_source_use": {
            "hidden_fields_in_consumer_csv": False,
            "state_behavior": state_report,
            "source_use": _source_use(
                sources,
                states,
                generation.state_pools,
                library.monthly_period_ordinals,
            ),
        },
        "price_only_strategy_probe": _strategies(
            development, assessed_monthly, historical_monthly
        ),
        "limitations": [
            "No diagnostic threshold or release gate was added.",
            "Only 24 of 5,000 paths are described.",
            "Historical control has 219 months versus 600 simulated months.",
            "Six price-only probes are not an exhaustive predictability search.",
        ],
    }
    report_path = root / FILENAME
    _write(report_path, report)
    return V5DiagnosticsResult(report_path, "diagnostic_report_only")


def _observables(
    paths: Sequence[FloatArray], historical: FloatArray, frequency: Frequency
) -> dict[str, object]:
    rows = tuple(
        compute_lean_g5_path_diagnostics(path, frequency=frequency) for path in paths
    )
    control = compute_lean_g5_path_diagnostics(historical, frequency=frequency)
    names = tuple(name for name in rows[0].names if _keep(name, frequency))
    endpoints = {}
    for name in names:
        values = np.asarray([row.value(name) for row in rows])
        endpoints[name] = {
            "simulated_mean": float(np.mean(values)),
            "simulated_median": float(np.median(values)),
            "simulated_q05": float(np.quantile(values, 0.05)),
            "simulated_q95": float(np.quantile(values, 0.95)),
            "historical_control": control.value(name),
        }
    return {
        "simulated_paths": len(paths),
        "simulated_observations": [len(path) for path in paths],
        "historical_observations": len(historical),
        "endpoints": endpoints,
    }


def _keep(name: str, frequency: Frequency) -> bool:
    prefixes = (
        "annualized_volatility.",
        "covariance.",
        "correlation.",
        "quantile.0.01.",
        "quantile.0.05.",
        "quantile.0.95.",
        "quantile.0.99.",
        "lower_tail_mean.",
        "upper_tail_mean.",
        "drawdown_",
        "taxable_muni_spread.",
        "downside_coexceedance.",
    )
    lags = (1, 3, 12) if frequency == "monthly" else (1, 5, 20)
    return name.startswith(prefixes) or any(
        name.startswith(f"{kind}.{lag}.")
        for kind in ("raw_acf", "absolute_acf", "squared_acf")
        for lag in lags
    )


def _strategies(
    development: Sequence[FloatArray],
    assessed: Sequence[FloatArray],
    historical: FloatArray,
) -> dict[str, object]:
    model = fit_lean_g5_strategy_model(development, frequency="monthly")
    held = [_quarter_weights(model, path)[1] for path in development]
    comparator = np.mean(np.concatenate(held), axis=0)
    simulated = np.vstack(
        [_quarter_alpha(model, path, comparator)[1] for path in assessed]
    )
    control_observations, control = _quarter_alpha(model, historical, comparator)
    summary = {}
    for column, name in enumerate(LEAN_G5_STRATEGY_NAMES):
        values = simulated[:, column]
        median = float(np.median(values))
        summary[name] = {
            "simulated_mean": float(np.mean(values)),
            "simulated_median": median,
            "simulated_q05": float(np.quantile(values, 0.05)),
            "simulated_q95": float(np.quantile(values, 0.95)),
            "historical_control": float(control[column]),
            "historical_minus_simulated_median": float(control[column] - median),
        }
    return {
        "strategy_names": list(LEAN_G5_STRATEGY_NAMES),
        "available_inputs": "past returns for the three consumer assets only",
        "unavailable_inputs": ["HMM state", "source identity", "macro", "future"],
        "timing": (
            "weights decided after each completed quarter and held for the "
            "following three monthly returns; 12-month warmup"
        ),
        "development_path_ids": list(DEVELOPMENT_IDS),
        "assessment_path_ids": list(ASSESSMENT_IDS),
        "fitted_model_fingerprint": model.fingerprint,
        "historical_control": {
            "kind": "direct synchronized historical sequence",
            "score_observations": control_observations,
            "calculated_with_identical_scorer": True,
            "zero_filled": False,
        },
        "summary": summary,
        "release_gate": False,
    }


def _quarter_weights(
    model: LeanG5StrategyModel, path: FloatArray
) -> tuple[int, FloatArray]:
    monthly = lean_g5_strategy_weights(model, path)
    held = np.empty_like(monthly.values)
    for start in range(0, len(held), 3):
        held[start : start + 3] = monthly.values[start]
    return monthly.first_return_index, held


def _quarter_alpha(
    model: LeanG5StrategyModel, path: FloatArray, comparator: FloatArray
) -> tuple[int, FloatArray]:
    first, weights = _quarter_weights(model, path)
    if comparator.shape != (len(LEAN_G5_STRATEGY_NAMES), 3):
        raise ValidationError("quarter-end comparator geometry is invalid")
    simple = np.expm1(path[first:])
    design = np.column_stack((np.ones(len(simple)), simple))
    if np.linalg.matrix_rank(design) != 4:
        raise ValidationError("quarter-end alpha regression is rank-degenerate")
    active = np.einsum("tsa,ta->ts", weights - comparator, simple, optimize=True)
    coefficients, _, _, _ = np.linalg.lstsq(design, active, rcond=None)
    residual = active - design @ coefficients
    scale = np.sqrt(np.sum(residual * residual, axis=0) / (len(design) - 4))
    if not bool(np.isfinite(scale).all()):
        raise ValidationError("quarter-end alpha regression is non-finite")
    alphas = np.empty(len(LEAN_G5_STRATEGY_NAMES))
    for strategy in range(len(alphas)):
        if scale[strategy] > 1e-12:
            alphas[strategy] = (
                math.sqrt(12.0) * coefficients[0, strategy] / scale[strategy]
            )
            continue
        # A constant weight is a static exposure, so residual timing alpha is zero.
        if not np.all(weights[:, strategy] == weights[0, strategy]):
            raise ValidationError("dynamic quarter-end alpha is degenerate")
        tolerance = 256 * np.finfo(np.float64).eps * math.sqrt(len(design))
        if (
            abs(float(coefficients[0, strategy])) > tolerance
            or np.linalg.norm(residual[:, strategy]) > tolerance
        ):
            raise ValidationError("static quarter-end alpha is inconsistent")
        alphas[strategy] = 0.0
    return len(design), np.asarray(alphas)


def _source_use(
    sources: Sequence[NDArray[np.int64]],
    states: Sequence[NDArray[np.int64]],
    pools: Sequence[NDArray[np.int64]],
    ordinals: NDArray[np.int64],
) -> dict[str, object]:
    all_sources = np.concatenate(sources)
    counts = Counter(map(int, all_sources))
    by_state = []
    for state, pool in enumerate(pools):
        selected = np.concatenate(
            [x[s == state] for x, s in zip(sources, states, strict=True)]
        )
        state_counts = Counter(map(int, selected))
        by_state.append(
            {
                "state": state,
                "label": STATE_LABELS[state],
                "selections": len(selected),
                "pool_months": len(pool),
                "unique_source_months": len(state_counts),
                "maximum_source_month_use": max(state_counts.values()),
            }
        )
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "selections": len(all_sources),
        "available_source_months": len(ordinals),
        "unique_source_months": len(counts),
        "maximum_source_month_use": max(counts.values()),
        "top_source_months": [
            {"month": _month(int(ordinals[index])), "uses": uses} for index, uses in top
        ],
        "per_state": by_state,
    }


def _month(ordinal: int) -> str:
    return f"{1970 + ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def _manifest(root: Path) -> Mapping[str, object]:
    try:
        value = json.loads((root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError("diagnostic manifest could not be read") from error
    plan = value.get("plan") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_id") != "prpg-v5-generation-manifest-v1"
        or value.get("status") != "complete"
        or not isinstance(plan, Mapping)
        or plan.get("profile") != "canonical"
        or plan.get("path_count") != 5_000
    ):
        raise IntegrityError("diagnostics require the complete canonical v5 run")
    return value


def _root(value: str | Path) -> Path:
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise IntegrityError("diagnostic run root does not exist") from error
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("diagnostic run root is unsafe")
    return root


def _fingerprint(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError("diagnostic manifest fingerprint is invalid")
    return value


def _write(path: Path, report: Mapping[str, object]) -> None:
    content = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise IntegrityError("diagnostic report could not be written") from error
