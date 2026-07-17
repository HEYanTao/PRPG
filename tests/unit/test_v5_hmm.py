from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.v5.config import load_v5_inputs
from prpg.v5.hmm import (
    RestartOutcome,
    _convergence_result,
    _correlation_matrix_or_none,
    _fit_one_restart,
    _format_optional,
    _proposed_state_labels,
    _report_only_stationary_distribution,
    _select_best_outcome,
)


def _outcome(
    index: int, likelihood: float | None, *, valid: bool = True
) -> RestartOutcome:
    return RestartOutcome(
        restart_index=index,
        seed=1_000 + index,
        valid=valid,
        iterations=10,
        log_likelihood=likelihood,
        final_likelihood_increment=1e-7 if valid else None,
        reason=None if valid else "non_convergence",
    )


def test_convergence_requires_small_final_absolute_increment() -> None:
    converged, increment, reason = _convergence_result(
        np.asarray([-10.0, -9.0]), tolerance=1e-6
    )
    assert not converged
    assert increment == 1.0
    assert reason == "final_absolute_likelihood_increment_not_below_tol"

    converged, increment, reason = _convergence_result(
        np.asarray([-10.0, -10.0 + 5e-7]), tolerance=1e-6
    )
    assert converged
    assert increment == pytest.approx(5e-7)
    assert reason is None


def test_highest_valid_likelihood_wins_without_pool_rescue() -> None:
    outcomes = [_outcome(index, -100.0 - index) for index in range(50)]
    outcomes[7] = _outcome(7, -80.0)
    outcomes[8] = _outcome(8, -80.0)
    outcomes[9] = _outcome(9, -70.0, valid=False)

    selected = _select_best_outcome(outcomes)

    assert selected.restart_index == 7
    assert selected.log_likelihood == -80.0


def test_exactly_fifty_restarts_are_required() -> None:
    with pytest.raises(ModelError, match="exactly 50"):
        _select_best_outcome([_outcome(index, -100.0) for index in range(49)])


def test_noncanonical_restart_count_is_explicit_and_deterministic() -> None:
    outcomes = [_outcome(index, -100.0 - index) for index in range(7)]
    outcomes[4] = _outcome(4, -50.0)

    selected = _select_best_outcome(outcomes, expected_restarts=7)

    assert selected.restart_index == 4


def test_k4_review_language_uses_configured_tickers() -> None:
    labels = _proposed_state_labels(
        np.asarray(
            [
                [-0.10, -0.02, -0.01],
                [0.00, -0.03, -0.02],
                [0.01, 0.01, 0.02],
                [0.10, 0.03, 0.02],
            ]
        ),
        feature_names=("EQX", "TAXX", "MUNIX"),
    )

    assert "EQX" in labels[0][1]
    assert any("TAXX/MUNIX" in interpretation for _, interpretation in labels.values())


def test_real_diagonal_restart_enforces_the_variance_floor() -> None:
    inputs = load_v5_inputs("configs/v5-canonical.yaml")
    rng = np.random.default_rng(44)
    centroids = np.asarray(
        [
            [-2.0, -1.0, -0.8],
            [-0.5, 0.7, 0.4],
            [0.5, -0.4, 0.8],
            [2.0, 1.0, 0.9],
        ]
    )
    source_states = np.tile(np.repeat(np.arange(4), 20), 4)
    values = centroids[source_states] + rng.normal(0.0, 0.15, (320, 3))

    outcome = _fit_one_restart(
        values,
        (len(values),),
        restart_index=0,
        seed=2,
        inputs=inputs,
    )

    assert outcome.valid
    assert outcome.variances_standardized is not None
    assert float(outcome.variances_standardized.min()) >= 1e-6
    assert outcome.transition_matrix is not None
    np.testing.assert_allclose(
        outcome.transition_matrix.sum(axis=1), np.ones(4), rtol=0.0, atol=1e-12
    )


def test_report_only_stationary_failure_does_not_become_a_fit_veto() -> None:
    stationary, warning = _report_only_stationary_distribution(
        np.eye(4),
        probability_tolerance=1e-12,
        residual_tolerance=1e-12,
    )

    assert stationary is None
    assert warning == "K4 stationary distribution is not unique"


def test_sparse_owner_metrics_render_as_not_estimable() -> None:
    assert _correlation_matrix_or_none(np.empty((0, 3))) is None
    assert _correlation_matrix_or_none(np.ones((1, 3))) is None
    assert _format_optional(None, ".2%") == "not estimable"
