from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import prpg.cli as cli
import prpg.v5.diagnostics as diagnostics
from prpg.validation.lean_g5_strategies import fit_lean_g5_strategy_model


def _paths(seed: int, *, count: int, observations: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        rng.normal((0.006, 0.002, 0.003), (0.04, 0.012, 0.018), (observations, 3))
        for _ in range(count)
    )


def test_quarter_end_weights_hold_three_months_without_lookahead() -> None:
    development = _paths(1, count=6, observations=72)
    model = fit_lean_g5_strategy_model(development, frequency="monthly")
    original = development[0].copy()
    changed = original.copy()
    changed[14] += (0.5, -0.3, 0.2)

    first, weights = diagnostics._quarter_weights(model, original)
    changed_first, changed_weights = diagnostics._quarter_weights(model, changed)

    assert first == changed_first == 12
    np.testing.assert_array_equal(weights[0], weights[1])
    np.testing.assert_array_equal(weights[0], weights[2])
    np.testing.assert_array_equal(weights[:3], changed_weights[:3])


def test_historical_control_alpha_is_calculated_not_zero_filled() -> None:
    development = _paths(2, count=6, observations=72)
    model = fit_lean_g5_strategy_model(development, frequency="monthly")
    held = [diagnostics._quarter_weights(model, path)[1] for path in development]
    comparator = np.mean(np.concatenate(held), axis=0)

    observations, alpha = diagnostics._quarter_alpha(
        model, _paths(3, count=1, observations=84)[0], comparator
    )

    assert observations == 72
    assert np.isfinite(alpha).all()
    assert np.any(np.abs(alpha) > 1e-8)


def test_v5_diagnose_cli_emits_report_only_status(monkeypatch: object) -> None:
    class _Result:
        def as_dict(self) -> dict[str, object]:
            return {
                "report_path": "/tmp/diagnostics-report.json",
                "status": "diagnostic_report_only",
                "release_gate": False,
            }

    monkeypatch.setattr(cli, "load_v5_inputs", lambda path: "inputs")  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli, "run_v5_report_diagnostics", lambda inputs, root: _Result()
    )
    result = CliRunner().invoke(
        cli.app,
        ["v5", "diagnose", str(Path("run")), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "diagnostic_report_only"
    assert payload["release_gate"] is False
