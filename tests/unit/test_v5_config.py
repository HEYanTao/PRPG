from __future__ import annotations

from pathlib import Path

import pytest

from prpg.errors import ConfigurationError
from prpg.v5.config import load_v5_inputs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "v5-canonical.yaml"


def test_canonical_v5_contract_is_frozen() -> None:
    resolved = load_v5_inputs(CONFIG)

    config = resolved.config
    assert config.data.assets.model_dump() == {
        "equity": "ACWI",
        "muni_bond": "MUB",
        "taxable_bond": "LQD",
    }
    assert config.data.cutoff.isoformat() == "2026-06-30"
    assert config.model.feature_order == ("equity", "taxable_bond", "muni_bond")
    assert config.model.states == 4
    assert config.model.covariance_type == "diag"
    assert config.model.deterministic_restarts == 50
    assert config.model.transition_probability_prior == 1.5
    assert config.model.minimum_state_months == 10
    assert config.simulation.paths == 5_000
    assert config.simulation.horizon_months == 600
    assert config.simulation.split.model_dump() == {
        "training": 4_000,
        "validation": 500,
        "test": 500,
    }
    assert config.simulation.output_frequencies == (
        "daily",
        "monthly",
        "quarterly",
        "annual",
    )
    assert "weekly" not in config.simulation.output_frequencies
    assert config.execution.workers == 9
    assert config.validation.aggregate_branch_coverage_minimum_percent == 80.0
    assert resolved.config_fingerprint == (
        "b903100a9e4c322b1ac20555272de32c4666d9625ef408ce459c5c8ec8081466"
    )
    assert resolved.seed_registry_fingerprint == (
        "2c58077db1196b97866d39c5445d093047cc84b0402e1e45ed6d0e596c34956c"
    )
    assert resolved.seed_registry.path_generation.training_path_ids == (1, 4_000)


def test_seed_registry_must_match_config(tmp_path: Path) -> None:
    config_text = (
        CONFIG.read_text(encoding="utf-8")
        .replace("master_seed: 20260716", "master_seed: 20260717", 1)
        .replace("seed_registry: v5-seed-registry.yaml", "seed_registry: seeds.yaml")
    )
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    (tmp_path / "seeds.yaml").write_text(
        (ROOT / "configs" / "v5-seed-registry.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="master seeds differ"):
        load_v5_inputs(tmp_path / "config.yaml")


def test_k4_numerical_hyperparameters_are_configurable(tmp_path: Path) -> None:
    config_text = (
        CONFIG.read_text(encoding="utf-8")
        .replace("seed_registry: v5-seed-registry.yaml", "seed_registry: seeds.yaml")
        .replace("deterministic_restarts: 50", "deterministic_restarts: 7")
        .replace("variance_floor: 1.0e-6", "variance_floor: 1.0e-5")
        .replace("start_probability_prior: 1.0", "start_probability_prior: 1.2")
        .replace(
            "transition_probability_prior: 1.5", "transition_probability_prior: 2.0"
        )
        .replace("means_prior: 0.0", "means_prior: 0.1")
        .replace("means_weight: 0.0", "means_weight: 0.2")
        .replace("covariance_prior: 0.0", "covariance_prior: 0.1")
        .replace("covariance_weight: 0.0", "covariance_weight: 0.2")
        .replace("maximum_iterations: 1000", "maximum_iterations: 500")
        .replace("tolerance: 1.0e-6", "tolerance: 1.0e-5")
        .replace("standard_deviation_ddof: 0", "standard_deviation_ddof: 1")
        .replace("minimum_state_months: 10", "minimum_state_months: 8")
    )
    seed_text = (
        (ROOT / "configs" / "v5-seed-registry.yaml")
        .read_text(encoding="utf-8")
        .replace("restart_last: 49", "restart_last: 6")
    )
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    (tmp_path / "seeds.yaml").write_text(seed_text, encoding="utf-8")

    model = load_v5_inputs(tmp_path / "config.yaml").config.model

    assert model.states == 4
    assert model.covariance_type == "diag"
    assert model.deterministic_restarts == 7
    assert model.variance_floor == 1.0e-5
    assert model.standard_deviation_ddof == 1
    assert model.minimum_state_months == 8


@pytest.mark.parametrize(
    ("original", "replacement"),
    [("states: 4", "states: 3"), ("covariance_type: diag", "covariance_type: full")],
)
def test_k4_architecture_remains_fixed(
    tmp_path: Path, original: str, replacement: str
) -> None:
    config_text = (
        CONFIG.read_text(encoding="utf-8")
        .replace("seed_registry: v5-seed-registry.yaml", "seed_registry: seeds.yaml")
        .replace(original, replacement, 1)
    )
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    (tmp_path / "seeds.yaml").write_text(
        (ROOT / "configs" / "v5-seed-registry.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="configuration is invalid"):
        load_v5_inputs(tmp_path / "config.yaml")
