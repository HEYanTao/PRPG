#!/usr/bin/env python3
"""Build a reproducible, report-only v5 historical comparison.

The canonical hard validator answers whether the complete delivery satisfies
its frozen release contract.  This script answers a different onboarding
question: what do the delivered monthly paths look like next to the finite
historical month library?  It adds no threshold or release decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from prpg.errors import IntegrityError
from prpg.v5.config import load_v5_inputs
from prpg.v5.data import load_v5_month_library

FloatArray = NDArray[np.float64]
RETURN_COLUMNS = (
    "equity_log_return",
    "muni_bond_log_return",
    "taxable_bond_log_return",
)
ASSET_ROLES = ("equity", "muni_bond", "taxable_bond")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: FloatArray) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _maximum_drawdown(paths: FloatArray) -> FloatArray:
    cumulative = np.cumsum(paths, axis=1)
    peaks = np.maximum.accumulate(
        np.concatenate((np.zeros((len(paths), 1, paths.shape[2])), cumulative), axis=1),
        axis=1,
    )[:, 1:, :]
    return np.asarray(
        np.max(1.0 - np.exp(cumulative - peaks), axis=1), dtype=np.float64
    )


def _load_simulated_months(
    run_root: Path, horizon_months: int, matched_months: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int, int]:
    files = sorted((run_root / "consumer" / "returns").glob("*/monthly/*.csv"))
    if not files:
        raise IntegrityError("no monthly consumer CSV files were found")
    pooled: list[FloatArray] = []
    matched_drawdowns: list[FloatArray] = []
    full_drawdowns: list[FloatArray] = []
    cagr: list[FloatArray] = []
    path_count = 0
    for path in files:
        frame = pd.read_csv(path, usecols=("path_id", *RETURN_COLUMNS))
        counts = frame.groupby("path_id", sort=False).size().to_numpy()
        if len(counts) == 0 or bool((counts != horizon_months).any()):
            raise IntegrityError(f"monthly path geometry is invalid in {path}")
        values = frame.loc[:, RETURN_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise IntegrityError(f"monthly returns are non-finite in {path}")
        paths = values.reshape(len(counts), horizon_months, 3)
        pooled.append(values)
        full_drawdowns.append(_maximum_drawdown(paths))
        matched_drawdowns.append(_maximum_drawdown(paths[:, :matched_months, :]))
        annualized_log = paths.sum(axis=1) * (12.0 / horizon_months)
        cagr.append(np.expm1(annualized_log))
        path_count += len(paths)
    return (
        np.concatenate(pooled, axis=0),
        np.concatenate(full_drawdowns, axis=0),
        np.concatenate(matched_drawdowns, axis=0),
        np.concatenate(cagr, axis=0),
        path_count,
        len(files),
    )


def _asset_summary(
    simulated: FloatArray,
    historical: FloatArray,
    cagr: FloatArray,
    full_drawdown: FloatArray,
    matched_drawdown: FloatArray,
    historical_drawdown: FloatArray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for column, role in enumerate(ASSET_ROLES):
        simulated_simple = np.expm1(simulated[:, column])
        historical_simple = np.expm1(historical[:, column])
        result[role] = {
            "annualized_geometric_return": {
                "historical": float(np.expm1(12.0 * historical[:, column].mean())),
                "simulated_pooled": float(np.expm1(12.0 * simulated[:, column].mean())),
                "simulated_path_distribution": _quantiles(cagr[:, column]),
            },
            "annualized_monthly_log_volatility": {
                "historical": float(
                    historical[:, column].std(ddof=1) * math.sqrt(12.0)
                ),
                "simulated_pooled": float(
                    simulated[:, column].std(ddof=1) * math.sqrt(12.0)
                ),
            },
            "monthly_simple_return_quantiles": {
                "historical": {
                    f"q{int(q * 100):02d}": float(np.quantile(historical_simple, q))
                    for q in (0.01, 0.05, 0.50, 0.95, 0.99)
                },
                "simulated_pooled": {
                    f"q{int(q * 100):02d}": float(np.quantile(simulated_simple, q))
                    for q in (0.01, 0.05, 0.50, 0.95, 0.99)
                },
            },
            "maximum_drawdown": {
                "historical_full_219_months": float(historical_drawdown[0, column]),
                "simulated_first_219_months": _quantiles(matched_drawdown[:, column]),
                "simulated_full_600_months": _quantiles(full_drawdown[:, column]),
            },
        }
    return result


def _correlation_rows(matrix: FloatArray) -> dict[str, float]:
    return {
        "equity__muni_bond": float(matrix[0, 1]),
        "equity__taxable_bond": float(matrix[0, 2]),
        "muni_bond__taxable_bond": float(matrix[1, 2]),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    assets = report["assets"]
    tickers = report["tickers"]
    lines = [
        "# PRPG v5 historical versus simulated return summary",
        "",
        "**Status:** descriptive report only; this is not a release gate or a "
        "hypothesis test.",
        "",
        "The simulated columns use every delivered monthly observation from all "
        "5,000 fifty-year paths. Historical values use the 219 complete synchronized "
        "months from April 2008 through June 2026.",
        "",
        "## Core return and volatility comparison",
        "",
        "| Role (ticker) | Historical geometric return | Simulated pooled "
        "geometric return | Historical volatility | Simulated volatility |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in ASSET_ROLES:
        item = assets[role]
        annual = item["annualized_geometric_return"]
        volatility = item["annualized_monthly_log_volatility"]
        lines.append(
            f"| {role.replace('_', ' ').title()} ({tickers[role]}) | "
            f"{annual['historical']:.2%} | {annual['simulated_pooled']:.2%} | "
            f"{volatility['historical']:.2%} | {volatility['simulated_pooled']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Distribution of 50-year path annualized returns",
            "",
            "| Role | 5th percentile | Median | 95th percentile |",
            "|---|---:|---:|---:|",
        ]
    )
    for role in ASSET_ROLES:
        distribution = assets[role]["annualized_geometric_return"][
            "simulated_path_distribution"
        ]
        lines.append(
            f"| {role.replace('_', ' ').title()} | {distribution['q05']:.2%} | "
            f"{distribution['median']:.2%} | {distribution['q95']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Monthly tail comparison",
            "",
            "| Role | History 1% | Simulated 1% | History 5% | Simulated 5% | "
            "History 95% | Simulated 95% | History 99% | Simulated 99% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for role in ASSET_ROLES:
        quantiles = assets[role]["monthly_simple_return_quantiles"]
        history = quantiles["historical"]
        simulated = quantiles["simulated_pooled"]
        lines.append(
            f"| {role.replace('_', ' ').title()} | {history['q01']:.2%} | "
            f"{simulated['q01']:.2%} | {history['q05']:.2%} | "
            f"{simulated['q05']:.2%} | {history['q95']:.2%} | "
            f"{simulated['q95']:.2%} | {history['q99']:.2%} | "
            f"{simulated['q99']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Correlation comparison",
            "",
            "These are pooled monthly log-return correlations; the canonical hard "
            "validator separately reports full-population daily correlations.",
            "",
            "| Pair | Historical | Simulated pooled |",
            "|---|---:|---:|",
        ]
    )
    history_correlation = report["correlation"]["historical"]
    simulated_correlation = report["correlation"]["simulated_pooled"]
    for pair in history_correlation:
        lines.append(
            f"| {pair.replace('__', ' / ').replace('_', ' ')} | "
            f"{history_correlation[pair]:.3f} | {simulated_correlation[pair]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Drawdown comparison",
            "",
            "| Role | Historical 219m | Simulated first 219m median (5%-95%) | "
            "Simulated full 600m median (5%-95%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for role in ASSET_ROLES:
        drawdown = assets[role]["maximum_drawdown"]
        matched = drawdown["simulated_first_219_months"]
        full = drawdown["simulated_full_600_months"]
        lines.append(
            f"| {role.replace('_', ' ').title()} | "
            f"{drawdown['historical_full_219_months']:.2%} | "
            f"{matched['median']:.2%} ({matched['q05']:.2%}-"
            f"{matched['q95']:.2%}) | "
            f"{full['median']:.2%} ({full['q05']:.2%}-"
            f"{full['q95']:.2%}) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The close return, volatility, and correlation values show what "
            "the synchronized empirical-month sampler preserves well at the "
            "ensemble level.",
            "- Tail values remain tied to observed source months; the HMM changes "
            "their "
            "ordering and persistence rather than inventing unseen shocks.",
            "- Matched-horizon drawdowns are the fairer historical comparison. "
            "Fifty-year drawdowns are naturally deeper because the simulated "
            "horizon is much longer.",
            "- The comparison is conditional on one 219-month ETF history and one "
            "fitted K4 model. It does not measure parameter uncertainty or prove "
            "future realism.",
            "- Cross-month dependence is approximated by the K4 transition chain. "
            "Within "
            "each sampled month, exact synchronized daily returns are retained.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *, config: Path, run_root: Path, output_json: Path, output_markdown: Path
) -> None:
    inputs = load_v5_inputs(config)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest["plan"]
    horizon = int(plan["horizon_months"])
    processed = str(manifest["processed_data_fingerprint"])
    library = load_v5_month_library(inputs, processed)
    # Library order is equity, taxable, muni; consumer order is equity, muni, taxable.
    historical = np.asarray(library.monthly_returns[:, (0, 2, 1)], dtype=np.float64)
    matched_months = len(historical)
    simulated, full_drawdown, matched_drawdown, cagr, path_count, file_count = (
        _load_simulated_months(run_root, horizon, matched_months)
    )
    historical_drawdown = _maximum_drawdown(historical[None, :, :])
    simulated_correlation = np.asarray(np.corrcoef(simulated, rowvar=False))
    historical_correlation = np.asarray(np.corrcoef(historical, rowvar=False))
    tickers = {
        "equity": manifest["assets"]["equity"],
        "muni_bond": manifest["assets"]["muni_bond"],
        "taxable_bond": manifest["assets"]["taxable_bond"],
    }
    report: dict[str, object] = {
        "schema_id": "prpg-v5-historical-simulated-comparison-v1",
        "status": "descriptive_report_only",
        "release_gate": False,
        "decision": None,
        "provenance": {
            "manifest_sha256": _sha256(manifest_path),
            "run_fingerprint": manifest["run_fingerprint"],
            "processed_data_fingerprint": processed,
            "model_fingerprint": manifest["model_fingerprint"],
        },
        "scope": {
            "simulated_paths": path_count,
            "simulated_months_per_path": horizon,
            "simulated_monthly_observations": len(simulated),
            "monthly_csv_files": file_count,
            "historical_monthly_observations": len(historical),
            "historical_first_month": "2008-04",
            "historical_last_month": "2026-06",
        },
        "tickers": tickers,
        "assets": _asset_summary(
            simulated,
            historical,
            cagr,
            full_drawdown,
            matched_drawdown,
            historical_drawdown,
        ),
        "correlation": {
            "historical": _correlation_rows(historical_correlation),
            "simulated_pooled": _correlation_rows(simulated_correlation),
        },
        "limitations": [
            "One finite 219-month historical ETF record is the only control.",
            "Five thousand paths reuse historical months with replacement.",
            "The paths are conditional on one fitted K4 model.",
            "The HMM approximates cross-month dependence; it is not a causal "
            "macro model.",
            "Full 600-month drawdowns are not directly comparable with 219-month "
            "history.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(_markdown(report), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/v5-canonical.yaml")
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/v5/v5-production-5000-20260716"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/v5-historical-vs-simulated-summary.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("doc/PRPG-v5-historical-vs-simulated-summary.md"),
    )
    arguments = parser.parse_args(argv)
    build_report(
        config=arguments.config,
        run_root=arguments.run_root,
        output_json=arguments.output_json,
        output_markdown=arguments.output_markdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
