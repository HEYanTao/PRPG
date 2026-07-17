"""Tests for registered nested bridge pseudo-history generation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.bridge_history import (
    BRIDGE_FULL_ROWS,
    BRIDGE_GAP_ROW,
    BRIDGE_PREFIX_ROWS,
    bridge_continuation_mask,
    bridge_fit_input_sha256,
    generate_registered_bridge_history,
    nonparametric_bridge_indices_from_uniforms,
    validate_bridge_pseudo_history,
)
from prpg.model.hmm import HMMParameters
from prpg.simulation.rng import CalibrationArtifact, Stage


def _parameters() -> HMMParameters:
    return HMMParameters(
        start_probabilities=np.array([0.5, 0.5]),
        transition_matrix=np.array([[0.90, 0.10], [0.15, 0.85]]),
        means=np.array([[-0.7, -0.3, 0.2, 0.5], [0.6, 0.4, -0.2, -0.4]]),
        tied_covariance=np.array(
            [
                [0.8, 0.1, 0.0, 0.0],
                [0.1, 0.7, 0.1, 0.0],
                [0.0, 0.1, 0.9, 0.05],
                [0.0, 0.0, 0.05, 0.6],
            ]
        ),
    )


def _design() -> np.ndarray:
    rng = np.random.default_rng(303)
    return rng.normal(size=(BRIDGE_PREFIX_ROWS, 4))


def _mask() -> np.ndarray:
    return np.r_[False, np.ones(BRIDGE_PREFIX_ROWS - 1, dtype=np.bool_)]


def test_registered_model_history_is_exact_nested_and_replayable() -> None:
    arguments = {
        "dgp": "model",
        "tier": "selection",
        "pair": 17,
        "master_seed": 90210,
        "scientific_version": 1,
        "design_parameters": _parameters(),
        "design_standardized_history": _design(),
        "design_continuation": _mask(),
        "macro_block_length": 6,
    }
    first = generate_registered_bridge_history(**arguments)  # type: ignore[arg-type]
    replay = generate_registered_bridge_history(**arguments)  # type: ignore[arg-type]

    assert first.values.shape == (217, 4)
    assert first.prefix_values.shape == (193, 4)
    assert first.full_lengths == (210, 7)
    assert np.array_equal(first.values, replay.values)
    assert np.array_equal(first.latent_states, replay.latent_states)
    assert first.source_indices is None
    assert first.macro_block_length is None
    assert not first.values.flags.writeable
    assert not first.prefix_values.flags.writeable
    assert not first.continuation.flags.writeable
    assert first.stream_audit.artifact is CalibrationArtifact.BRIDGE_MODEL
    assert first.stream_audit.variant == 0
    assert first.stream_audit.stages == (
        Stage.STATE_CHAIN,
        Stage.OPTIONAL_RESIDUAL_NOISE,
    )
    assert first.stream_audit.draws_by_stage == (217, 868)
    assert first.design_n_states == 2
    assert len(first.design_parameters_sha256) == 64
    assert len(first.design_history_sha256) == 64
    validate_bridge_pseudo_history(
        first,
        expected_tier="selection",
        expected_n_states=2,
    )
    assert bridge_fit_input_sha256(first) == bridge_fit_input_sha256(replay)


def test_registered_nonparametric_history_reuses_design_rows_and_changes_tier() -> None:
    design = _design()
    common = {
        "dgp": "nonparametric",
        "pair": 9,
        "master_seed": 888,
        "scientific_version": 1,
        "design_parameters": _parameters(),
        "design_standardized_history": design,
        "design_continuation": _mask(),
        "macro_block_length": 5,
    }
    selection = generate_registered_bridge_history(
        tier="selection",
        **common,  # type: ignore[arg-type]
    )
    fixed = generate_registered_bridge_history(
        tier="fixed_k",
        **common,  # type: ignore[arg-type]
    )

    assert selection.latent_states is None
    assert selection.source_indices is not None
    np.testing.assert_array_equal(selection.values, design[selection.source_indices])
    assert selection.stream_audit.artifact is CalibrationArtifact.BRIDGE_NONPARAMETRIC
    assert selection.stream_audit.stages == (Stage.REFERENCE_RESAMPLING,)
    assert selection.stream_audit.draws_by_stage == (434,)
    assert selection.stream_audit.variant == 0
    assert fixed.stream_audit.variant == 1
    assert not np.array_equal(selection.source_indices, fixed.source_indices)
    assert not selection.source_indices.flags.writeable


def test_nonparametric_trace_is_non_circular_and_resets_at_target_gap() -> None:
    source = np.r_[False, np.ones(16, dtype=np.bool_)]
    target = bridge_continuation_mask()
    restart = np.ones(BRIDGE_FULL_ROWS) * 0.99
    ranks = np.full(BRIDGE_FULL_ROWS, 0.1)
    ranks[0] = 0.999  # source row 16, forcing the next target row to restart.
    ranks[1] = 0.1
    ranks[BRIDGE_GAP_ROW] = 0.3

    indices = nonparametric_bridge_indices_from_uniforms(
        source,
        target,
        restart,
        ranks,
        block_length=2,
    )

    assert indices[0] == 16
    assert indices[1] == 1
    assert indices[2] == 2
    assert indices[BRIDGE_GAP_ROW] == 5
    assert not np.any((indices[:-1] == 16) & (indices[1:] == 0))
    assert not indices.flags.writeable


def test_exact_bridge_continuation_has_only_two_segment_starts() -> None:
    mask = bridge_continuation_mask()
    assert mask.shape == (217,)
    np.testing.assert_array_equal(np.flatnonzero(~mask), [0, 210])


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"dgp": "unknown"}, "DGP"),
        ({"tier": "unknown"}, "tier"),
        ({"pair": 250}, "registered tier range"),
        ({"macro_block_length": 1}, "hard caps"),
        ({"design_standardized_history": np.ones((192, 4))}, "shape"),
        ({"design_continuation": np.ones(193, dtype=np.bool_)}, "false-at-start"),
        (
            {
                "design_continuation": np.r_[
                    False,
                    np.ones(80, dtype=np.bool_),
                    False,
                    np.ones(111, dtype=np.bool_),
                ]
            },
            "gap-free",
        ),
    ],
)
def test_registered_history_rejects_contract_drift(
    replacement: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "dgp": "model",
        "tier": "selection",
        "pair": 0,
        "master_seed": 12,
        "scientific_version": 1,
        "design_parameters": _parameters(),
        "design_standardized_history": _design(),
        "design_continuation": _mask(),
        "macro_block_length": 4,
    }
    arguments.update(replacement)
    with pytest.raises(ModelError, match=message):
        generate_registered_bridge_history(**arguments)  # type: ignore[arg-type]


def test_history_validator_rejects_stream_accounting_tampering() -> None:
    history = generate_registered_bridge_history(
        dgp="model",
        tier="fixed_k",
        pair=3,
        master_seed=12,
        scientific_version=1,
        design_parameters=_parameters(),
        design_standardized_history=_design(),
        design_continuation=_mask(),
        macro_block_length=4,
    )
    tampered = replace(
        history,
        stream_audit=replace(history.stream_audit, draws_by_stage=(217, 867)),
    )
    with pytest.raises(ModelError, match="stream accounting"):
        validate_bridge_pseudo_history(tampered)

    with pytest.raises(ModelError, match="immutable finite"):
        validate_bridge_pseudo_history(replace(history, values=history.values.copy()))
    with pytest.raises(ModelError, match="continuation"):
        validate_bridge_pseudo_history(
            replace(history, continuation=history.continuation.copy())
        )
    with pytest.raises(ModelError, match="audit identity"):
        validate_bridge_pseudo_history(
            replace(
                history,
                stream_audit=replace(history.stream_audit, replicate=4),
            )
        )
    with pytest.raises(ModelError, match="fingerprints"):
        validate_bridge_pseudo_history(
            replace(history, design_history_sha256="not-a-sha256")
        )
    with pytest.raises(ModelError, match="model-DGP.*payload"):
        validate_bridge_pseudo_history(
            replace(history, source_indices=np.zeros(217, dtype=np.int64))
        )
    with pytest.raises(ModelError, match="expected state count"):
        validate_bridge_pseudo_history(history, expected_n_states=6)


@pytest.mark.parametrize(
    ("restarts", "ranks", "message"),
    [
        (np.zeros(216), np.zeros(217), "exactly 217"),
        (np.zeros(217), np.full(217, 1.0), r"\[0, 1\)"),
    ],
)
def test_nonparametric_trace_rejects_malformed_uniforms(
    restarts: np.ndarray,
    ranks: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        nonparametric_bridge_indices_from_uniforms(
            _mask(),
            bridge_continuation_mask(),
            restarts,
            ranks,
            block_length=4,
        )
