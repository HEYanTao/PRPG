from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from prpg.config import (
    PRPGConfig,
    config_json_schema,
    load_config,
    parse_config,
    redact_mapping,
)
from prpg.errors import ConfigurationError, ExitCode

ROOT = Path(__file__).parents[2]
CANONICAL = ROOT / "configs" / "canonical.yaml"
SMOKE = ROOT / "configs" / "smoke.yaml"


def _canonical_mapping() -> dict[str, object]:
    value = yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _set_nested(
    mapping: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    target: Any = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_canonical_config_resolves_approved_contract() -> None:
    config = load_config(CANONICAL)

    assert config.schema_version == 1
    assert config.run.master_seed == 20_260_714
    assert config.data.assets.model_dump() == {
        "equity": "ACWI",
        "muni_bond": "MUB",
        "taxable_bond": "AGG",
    }
    assert str(config.data.cutoff) == "2026-06-30"
    assert str(config.data.yfinance.end_exclusive) == "2026-07-01"
    assert config.simulation.calendar == "SC252-52-v1"
    assert config.simulation.scientific_version == 11
    assert config.model.covariance_type == "tied"
    assert config.model.alpha_policy_candidates == (
        "historical_vector",
        "kernel_mean_neutral",
    )
    assert config.model.selected_alpha_policy is None
    assert config.model.selected_neutral_lambda is None
    assert config.simulation.low_frequency.paths == 5_000
    assert config.simulation.high_frequency.paths == 1_000
    assert config.simulation.low_frequency.years == 50
    assert config.export.low_paths_per_shard == 100
    assert config.export.high_paths_per_shard == 50


def test_approved_statistical_policy_is_typed_and_frozen() -> None:
    validation = load_config(CANONICAL).validation

    assert validation.kernel_mean_calibration.prefix_window_counts == (
        10_000,
        20_000,
        40_000,
        80_000,
        160_000,
        320_000,
    )
    assert (
        validation.kernel_mean_calibration.annualized_simultaneous_half_width_max
        == 0.001
    )
    assert validation.raw_distortion.daily_quantiles == (0.01, 0.05, 0.5, 0.95, 0.99)
    assert validation.raw_distortion.monthly_quantiles == (0.1, 0.5, 0.9)
    assert validation.raw_distortion.absolute_annualized_mean_shift == 0.01
    assert (
        validation.raw_distortion.taxable_minus_muni_absolute_annualized_mean_shift
        == 0.005
    )
    assert validation.volatility_timing.annualized_sharpe_upper_bound_exclusive == 0.20


def test_smoke_config_is_non_release_but_scientifically_compatible() -> None:
    config = load_config(SMOKE)

    assert config.validation.profile == "smoke"
    assert config.simulation.low_frequency.paths == 4
    assert config.simulation.high_frequency.paths == 2
    assert config.simulation.low_frequency.years == 2
    assert config.simulation.scientific_version == 11
    assert config.execution.workers == 1


def test_configuration_is_deeply_immutable() -> None:
    config = load_config(CANONICAL)

    with pytest.raises(ValidationError):
        config.run.master_seed = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.model.regime_candidates[0] = 3  # type: ignore[index]


def test_unknown_key_fails_closed_without_echoing_input() -> None:
    mapping = _canonical_mapping()
    mapping["credential_token"] = "DO-NOT-ECHO"

    with pytest.raises(ConfigurationError) as captured:
        parse_config(mapping)

    assert captured.value.exit_code == ExitCode.CONFIGURATION
    assert "DO-NOT-ECHO" not in str(captured.value.details)
    assert captured.value.details["errors"][0]["type"] == "extra_forbidden"


def test_scientific_scalar_string_is_not_coerced() -> None:
    mapping = _canonical_mapping()
    simulation = copy.deepcopy(mapping["simulation"])
    assert isinstance(simulation, dict)
    low = simulation["low_frequency"]
    assert isinstance(low, dict)
    low["paths"] = "5000"
    mapping["simulation"] = simulation

    with pytest.raises(ConfigurationError, match="validation failed"):
        parse_config(mapping)


def test_release_profile_rejects_reduced_counts() -> None:
    mapping = _canonical_mapping()
    simulation = copy.deepcopy(mapping["simulation"])
    assert isinstance(simulation, dict)
    low = simulation["low_frequency"]
    assert isinstance(low, dict)
    low["paths"] = 100
    mapping["simulation"] = simulation

    with pytest.raises(ConfigurationError) as captured:
        parse_config(mapping)

    assert "canonical 5000/1000/50" in str(captured.value.details)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_alpha_policy", "historical_vector"),
        ("selected_alpha_policy", "kernel_mean_neutral"),
        ("selected_neutral_lambda", 0.50),
    ],
)
def test_g3_configuration_rejects_generation_policy_preselection(
    field: str, value: object
) -> None:
    mapping = _canonical_mapping()
    model = copy.deepcopy(mapping["model"])
    assert isinstance(model, dict)
    model[field] = value
    mapping["model"] = model

    with pytest.raises(ConfigurationError) as captured:
        parse_config(mapping)

    assert captured.value.details["errors"][0]["type"] == "none_required"


def test_canonical_hash_is_mapping_order_invariant() -> None:
    mapping = _canonical_mapping()
    reverse_order = dict(reversed(tuple(mapping.items())))

    first = parse_config(mapping)
    second = parse_config(reverse_order)

    assert first.canonical_json() == second.canonical_json()
    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64


def test_redaction_is_recursive_and_non_mutating() -> None:
    original = {
        "FRED_API_KEY": "abc",
        "nested": [{"authorization": "bearer x", "safe": 4}],
        "monkey": "not-a-secret-marker",
    }

    redacted = redact_mapping(original)

    assert redacted == {
        "FRED_API_KEY": "<redacted>",
        "nested": [{"authorization": "<redacted>", "safe": 4}],
        "monkey": "not-a-secret-marker",
    }
    assert original["FRED_API_KEY"] == "abc"


@pytest.mark.parametrize("contents", ["- item\n", "[unterminated"])
def test_non_mapping_or_malformed_yaml_fails_safely(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_config(path)


def test_missing_file_has_stable_nonsecret_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "absent.yaml"

    with pytest.raises(ConfigurationError) as captured:
        load_config(path)

    assert captured.value.details == {
        "path": str(path),
        "reason": "FileNotFoundError",
    }


def test_generated_json_schema_closes_unknown_keys() -> None:
    schema = config_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == 1
    assert PRPGConfig.model_validate(_canonical_mapping()).schema_version == 1


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data", "assets", "muni_bond"), "ACWI", "distinct tickers"),
        (("data", "macro", "inflation"), "INDPRO", "distinct FRED series"),
        (("data", "yfinance", "start"), "2026-07-01", "must be after start"),
        (("data", "cutoff"), "2026-07-01", "must be later than data cutoff"),
        (("data", "cutoff"), "2026-06-29", "cutoff plus one day"),
        (("model", "regime_candidates"), [2, 3, 4], "must be [2, 3, 4, 5]"),
        (
            ("model", "alpha_policy_candidates"),
            ["kernel_mean_neutral", "historical_vector"],
            "candidates/order is frozen",
        ),
        (("model", "neutral_lambda_grid"), [0.25, 0.5, 1.0], "grid is frozen"),
        (
            ("model", "selected_neutral_lambda"),
            0.25,
            "Input should be None",
        ),
        (
            ("simulation", "high_frequency", "years"),
            49,
            "LF and HF horizons must match",
        ),
        (
            (
                "validation",
                "kernel_mean_calibration",
                "annualized_simultaneous_half_width_max",
            ),
            0.002,
            "half-width maximum is frozen",
        ),
        (
            ("validation", "kernel_mean_calibration", "confidence"),
            0.90,
            "kernel mean confidence is frozen",
        ),
        (
            ("validation", "raw_distortion", "mean_shift_over_raw_volatility"),
            0.051,
            "raw-distortion cap mean_shift_over_raw_volatility",
        ),
        (
            ("validation", "raw_distortion", "daily_quantiles"),
            [0.02, 0.05, 0.50, 0.95, 0.99],
            "daily distortion quantiles are frozen",
        ),
        (
            ("validation", "raw_distortion", "monthly_quantiles"),
            [0.10, 0.50, 0.95],
            "monthly distortion quantiles are frozen",
        ),
        (
            (
                "validation",
                "volatility_timing",
                "simultaneous_familywise_one_sided_confidence",
            ),
            0.90,
            "volatility-timing confidence is frozen",
        ),
        (
            (
                "validation",
                "volatility_timing",
                "annualized_sharpe_upper_bound_exclusive",
            ),
            0.21,
            "Sharpe upper bound is frozen",
        ),
        (("export", "low_paths_per_shard"), 5_001, "LF paths per shard"),
        (("export", "high_paths_per_shard"), 1_001, "HF paths per shard"),
    ],
)
def test_frozen_cross_field_contracts_fail_closed(
    path: tuple[str, ...], value: object, message: str
) -> None:
    mapping = _canonical_mapping()
    _set_nested(mapping, path, value)

    with pytest.raises(ConfigurationError) as captured:
        parse_config(mapping)

    assert message in str(captured.value.details)


def test_redaction_preserves_tuple_shape() -> None:
    original = ({"password": "hidden", "safe": 1}, "plain")

    assert redact_mapping(original) == (
        {"password": "<redacted>", "safe": 1},
        "plain",
    )
