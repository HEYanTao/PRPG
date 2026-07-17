from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.special import ndtr

from prpg.errors import ModelError
from prpg.model.adequacy import (
    LATENT_BOOTSTRAP_CHUNK_SIZE,
    LATENT_BOOTSTRAP_REPLICATES,
    LATENT_BOOTSTRAP_RESAMPLE_SIZE,
    LATENT_CHAIN_COUNT,
    LATENT_EXTENSION_MONTHS,
    LATENT_GENERATION_CHUNK_SIZE,
    LATENT_MEASURED_MONTHS,
    LATENT_SURVIVAL_MONTHS,
    EndpointDirection,
    LatentChainSimulation,
    four_cell_holm,
    generate_latent_chains,
    latent_analytic_target,
    latent_cluster_bootstrap,
    latent_pooled_estimate,
    marginal_moments,
    mixed_max_statistic_cell,
    predictive_mixture_diagnostics,
    residual_acf_diagnostics,
)


def _latent_bootstrap_fixture() -> LatentChainSimulation:
    transition = np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=np.float64)
    stationary = np.asarray([0.5, 0.5], dtype=np.float64)
    # Every chain independently has measured departures and complete episodes
    # in both states, while their sufficient statistics are not identical.
    chains = np.asarray(
        [
            [0, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    return LatentChainSimulation(
        transition_matrix=transition,
        stationary_distribution=stationary,
        chains=chains,
        chain_count=4,
        measured_months=4,
        extension_months=3,
        uniforms_per_chain=8,
    )


def test_latent_target_has_exact_order_and_geometric_formulas() -> None:
    transition = np.asarray([[0.8, 0.2], [0.3, 0.7]])

    target = latent_analytic_target(transition, survival_months=3)

    assert target.labels == (
        "occupancy/0",
        "occupancy/1",
        "transition/0/1",
        "transition/1/0",
        "episode_rate/0",
        "episode_rate/1",
        "one_period/0",
        "one_period/1",
        "median_dwell/0",
        "median_dwell/1",
        "dwell_survival/0/1",
        "dwell_survival/0/2",
        "dwell_survival/0/3",
        "dwell_survival/1/1",
        "dwell_survival/1/2",
        "dwell_survival/1/3",
    )
    np.testing.assert_allclose(target.values[:2], [0.6, 0.4])
    np.testing.assert_allclose(target.values[2:4], [0.2, 0.3])
    np.testing.assert_allclose(target.values[4:6], [0.12, 0.12])
    np.testing.assert_allclose(target.values[6:8], [0.2, 0.3])
    np.testing.assert_allclose(target.values[8:10], [4.0, 2.0])
    np.testing.assert_allclose(target.values[10:], [1.0, 0.8, 0.64, 1.0, 0.7, 0.49])


def test_latent_pooled_estimator_uses_sentinel_and_extension_exactly() -> None:
    chains = np.asarray(
        [
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )

    estimate = latent_pooled_estimate(
        chains,
        2,
        measured_months=3,
        extension_months=2,
        survival_months=3,
    )

    np.testing.assert_allclose(estimate.values[:2], [0.5, 0.5])
    np.testing.assert_allclose(estimate.values[2:4], [0.75, 1.0])
    np.testing.assert_allclose(estimate.values[4:6], [2 / 6, 3 / 6])
    np.testing.assert_allclose(estimate.values[6:8], [0.5, 2 / 3])
    np.testing.assert_allclose(estimate.values[8:10], [2.0, 1.0])
    np.testing.assert_allclose(estimate.values[10:], [1.0, 0.5, 0.0, 1.0, 1 / 3, 0.0])


def test_latent_unfinished_extension_and_zero_support_fail_closed() -> None:
    unfinished = np.asarray([[1, 0, 0, 0, 0, 0]], dtype=np.int64)
    with pytest.raises(ModelError, match="remains incomplete"):
        latent_pooled_estimate(
            unfinished,
            2,
            measured_months=3,
            extension_months=2,
            survival_months=2,
        )
    with pytest.raises(ModelError):
        latent_analytic_target([[1.0, 0.0], [0.0, 1.0]])


def test_latent_generation_and_bootstrap_defaults_freeze_approved_contract() -> None:
    generation = inspect.signature(generate_latent_chains).parameters
    bootstrap = inspect.signature(latent_cluster_bootstrap).parameters

    assert generation["chain_count"].default == LATENT_CHAIN_COUNT == 10_000
    assert generation["measured_months"].default == LATENT_MEASURED_MONTHS == 600
    assert generation["extension_months"].default == LATENT_EXTENSION_MONTHS == 2_000
    assert generation["chunk_size"].default == LATENT_GENERATION_CHUNK_SIZE
    assert bootstrap["expected_chain_count"].default == LATENT_CHAIN_COUNT == 10_000
    assert (
        bootstrap["bootstrap_replicates"].default
        == LATENT_BOOTSTRAP_REPLICATES
        == 10_000
    )
    assert (
        bootstrap["resample_size"].default == LATENT_BOOTSTRAP_RESAMPLE_SIZE == 10_000
    )
    assert bootstrap["survival_months"].default == LATENT_SURVIVAL_MONTHS == 12
    assert bootstrap["chunk_size"].default == LATENT_BOOTSTRAP_CHUNK_SIZE


def test_latent_chain_generation_uses_stationary_sentinel_and_exact_uniforms() -> None:
    transition = np.asarray([[0.75, 0.25], [0.5, 0.5]])
    uniforms = np.asarray(
        [
            [0.10, 0.80, 0.40, 0.74, 0.75],
            [0.90, 0.49, 0.74, 0.90, 0.10],
        ]
    )

    result = generate_latent_chains(
        transition,
        chain_count=2,
        measured_months=2,
        extension_months=2,
        uniforms=uniforms,
        chunk_size=1,
    )

    np.testing.assert_allclose(result.stationary_distribution, [2 / 3, 1 / 3])
    np.testing.assert_array_equal(
        result.chains,
        [[0, 1, 0, 0, 1], [1, 0, 0, 1, 0]],
    )
    assert result.chain_count == 2
    assert result.measured_months == 2
    assert result.extension_months == 2
    assert result.uniforms_per_chain == 5
    assert result.chains.dtype == np.uint8
    assert not result.chains.flags.writeable
    assert not result.transition_matrix.flags.writeable
    assert not result.stationary_distribution.flags.writeable


def test_latent_generation_rng_trace_is_chunk_invariant() -> None:
    transition = np.asarray([[0.8, 0.2], [0.4, 0.6]])
    seed = 1234
    uniforms = np.random.default_rng(seed).random((5, 8))
    from_uniforms = generate_latent_chains(
        transition,
        chain_count=5,
        measured_months=4,
        extension_months=3,
        uniforms=uniforms,
        chunk_size=2,
    )
    one_row_chunks = generate_latent_chains(
        transition,
        chain_count=5,
        measured_months=4,
        extension_months=3,
        rng=np.random.default_rng(seed),
        chunk_size=1,
    )
    four_row_chunks = generate_latent_chains(
        transition,
        chain_count=5,
        measured_months=4,
        extension_months=3,
        rng=np.random.default_rng(seed),
        chunk_size=4,
    )

    np.testing.assert_array_equal(from_uniforms.chains, one_row_chunks.chains)
    np.testing.assert_array_equal(one_row_chunks.chains, four_row_chunks.chains)


class _BadRandomGenerator(np.random.Generator):
    def random(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return np.zeros((1, 1), dtype=np.float64)


class _BadIntegerGenerator(np.random.Generator):
    def integers(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return np.zeros((1, 1), dtype=np.int64)


@pytest.mark.parametrize(
    ("uniforms", "message"),
    [
        (np.zeros((2, 4)), "wrong exact shape"),
        (np.zeros((2, 5), dtype=np.bool_), "non-boolean"),
        (np.full((2, 5), np.nan), "non-finite"),
        (np.full((2, 5), 1.0), "half-open"),
        (np.full((2, 5), "bad"), "numeric"),
    ],
)
def test_latent_generation_rejects_malformed_uniform_traces(
    uniforms: np.ndarray, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        generate_latent_chains(
            [[0.5, 0.5], [0.5, 0.5]],
            chain_count=2,
            measured_months=2,
            extension_months=2,
            uniforms=uniforms,
        )


def test_latent_generation_requires_one_valid_injected_randomness_source() -> None:
    transition = [[0.5, 0.5], [0.5, 0.5]]
    uniforms = np.zeros((2, 5))
    with pytest.raises(ModelError, match="exactly one"):
        generate_latent_chains(
            transition,
            chain_count=2,
            measured_months=2,
            extension_months=2,
        )
    with pytest.raises(ModelError, match="exactly one"):
        generate_latent_chains(
            transition,
            chain_count=2,
            measured_months=2,
            extension_months=2,
            uniforms=uniforms,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ModelError, match="numpy Generator"):
        generate_latent_chains(
            transition,
            chain_count=2,
            measured_months=2,
            extension_months=2,
            rng=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="wrong exact shape"):
        generate_latent_chains(
            transition,
            chain_count=2,
            measured_months=2,
            extension_months=2,
            rng=_BadRandomGenerator(np.random.PCG64(1)),
        )


def test_latent_cluster_bootstrap_matches_direct_whole_chain_reference() -> None:
    simulation = _latent_bootstrap_fixture()
    indices = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 0, 2, 3],
            [1, 1, 2, 3],
            [0, 1, 2, 2],
            [0, 1, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int64,
    )

    result = latent_cluster_bootstrap(
        simulation,
        expected_chain_count=4,
        bootstrap_replicates=6,
        resample_size=4,
        survival_months=2,
        resample_indices=indices,
        chunk_size=2,
    )

    direct_pooled = latent_pooled_estimate(
        simulation.chains,
        2,
        measured_months=4,
        extension_months=3,
        survival_months=2,
    )
    np.testing.assert_allclose(result.pooled_estimate.values, direct_pooled.values)
    for replicate, selected in enumerate(indices):
        direct = latent_pooled_estimate(
            simulation.chains[selected],
            2,
            measured_months=4,
            extension_months=3,
            survival_months=2,
        )
        np.testing.assert_allclose(result.bootstrap_estimates[replicate], direct.values)

    standard_errors = result.bootstrap_estimates.std(axis=0, ddof=1)
    active = standard_errors > 1e-12
    direct_bootstrap_max = np.max(
        np.abs(
            (result.bootstrap_estimates[:, active] - direct_pooled.values[None, active])
            / standard_errors[None, active]
        ),
        axis=1,
    )
    direct_observed_max = float(
        np.max(
            np.abs(
                (direct_pooled.values[active] - result.target.values[active])
                / standard_errors[active]
            )
        )
    )
    direct_p = (1 + np.count_nonzero(direct_bootstrap_max >= direct_observed_max)) / 7
    np.testing.assert_allclose(result.standard_errors, standard_errors)
    np.testing.assert_array_equal(result.active_components, active)
    np.testing.assert_allclose(result.bootstrap_max_statistics, direct_bootstrap_max)
    assert result.observed_max_statistic == pytest.approx(direct_observed_max)
    assert result.p_value == pytest.approx(direct_p)
    assert result.chain_count == result.resample_size == 4
    assert result.bootstrap_replicates == 6
    assert result.chunk_size == 2
    assert not result.bootstrap_estimates.flags.writeable
    assert not result.bootstrap_max_statistics.flags.writeable


def test_latent_cluster_bootstrap_rng_is_exactly_chunk_invariant() -> None:
    simulation = _latent_bootstrap_fixture()
    arguments = {
        "expected_chain_count": 4,
        "bootstrap_replicates": 20,
        "resample_size": 4,
        "survival_months": 2,
    }
    single = latent_cluster_bootstrap(
        simulation,
        **arguments,
        rng=np.random.default_rng(901),
        chunk_size=1,
    )
    chunked = latent_cluster_bootstrap(
        simulation,
        **arguments,
        rng=np.random.default_rng(901),
        chunk_size=7,
    )

    np.testing.assert_array_equal(
        single.bootstrap_estimates, chunked.bootstrap_estimates
    )
    np.testing.assert_array_equal(
        single.bootstrap_max_statistics, chunked.bootstrap_max_statistics
    )
    assert single.observed_max_statistic == chunked.observed_max_statistic
    assert single.p_value == chunked.p_value


def test_latent_cluster_bootstrap_all_constant_passing_case() -> None:
    transition = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    chains = np.asarray(
        [[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
        dtype=np.uint8,
    )
    simulation = LatentChainSimulation(
        transition,
        np.asarray([0.5, 0.5]),
        chains,
        chain_count=2,
        measured_months=4,
        extension_months=2,
        uniforms_per_chain=7,
    )
    result = latent_cluster_bootstrap(
        simulation,
        expected_chain_count=2,
        bootstrap_replicates=2,
        resample_size=2,
        survival_months=2,
        resample_indices=[[0, 1], [1, 0]],
    )

    assert not result.active_components.any()
    np.testing.assert_array_equal(result.bootstrap_max_statistics, [0.0, 0.0])
    assert result.observed_max_statistic == 0.0
    assert result.p_value == 1.0


def test_latent_cluster_bootstrap_constant_mismatch_fails_closed() -> None:
    transition = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    chains = np.asarray(
        [[0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0]],
        dtype=np.uint8,
    )
    simulation = LatentChainSimulation(
        transition,
        np.asarray([0.5, 0.5]),
        chains,
        chain_count=2,
        measured_months=4,
        extension_months=2,
        uniforms_per_chain=7,
    )
    with pytest.raises(ModelError, match="constant target"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=2,
            bootstrap_replicates=2,
            resample_size=2,
            survival_months=2,
            resample_indices=[[0, 1], [1, 0]],
        )


def test_latent_bootstrap_fails_on_unfinished_spell_or_zero_denominator() -> None:
    transition = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    unfinished = LatentChainSimulation(
        transition,
        np.asarray([0.5, 0.5]),
        np.asarray([[1, 0, 0, 0, 0, 0]], dtype=np.uint8),
        chain_count=1,
        measured_months=3,
        extension_months=2,
        uniforms_per_chain=6,
    )
    with pytest.raises(ModelError, match="remains incomplete"):
        latent_cluster_bootstrap(
            unfinished,
            expected_chain_count=1,
            bootstrap_replicates=2,
            resample_size=1,
            survival_months=2,
            resample_indices=[[0], [0]],
        )

    no_support = LatentChainSimulation(
        transition,
        np.asarray([0.5, 0.5]),
        np.zeros((1, 6), dtype=np.uint8),
        chain_count=1,
        measured_months=3,
        extension_months=2,
        uniforms_per_chain=6,
    )
    with pytest.raises(ModelError, match="transition denominator"):
        latent_cluster_bootstrap(
            no_support,
            expected_chain_count=1,
            bootstrap_replicates=2,
            resample_size=1,
            survival_months=2,
            resample_indices=[[0], [0]],
        )


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.zeros((5, 4), dtype=np.int64), "wrong exact shape"),
        (np.zeros((6, 4), dtype=np.float64), "integer dtype"),
        (np.full((6, 4), 4, dtype=np.int64), "out-of-range"),
    ],
)
def test_latent_bootstrap_rejects_malformed_index_traces(
    indices: np.ndarray, message: str
) -> None:
    with pytest.raises(ModelError, match=message):
        latent_cluster_bootstrap(
            _latent_bootstrap_fixture(),
            expected_chain_count=4,
            bootstrap_replicates=6,
            resample_size=4,
            survival_months=2,
            resample_indices=indices,
        )


def test_latent_bootstrap_rejects_wrong_exact_counts_and_rng_shape() -> None:
    simulation = _latent_bootstrap_fixture()
    with pytest.raises(ModelError, match="exact bootstrap contract"):
        latent_cluster_bootstrap(
            simulation,
            bootstrap_replicates=2,
            resample_size=10_000,
            resample_indices=np.zeros((2, 10_000), dtype=np.int64),
        )
    with pytest.raises(ModelError, match="full chain-count"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=4,
            bootstrap_replicates=2,
            resample_size=3,
            resample_indices=np.zeros((2, 3), dtype=np.int64),
        )
    with pytest.raises(ModelError, match="at least two"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=4,
            bootstrap_replicates=1,
            resample_size=4,
            resample_indices=np.zeros((1, 4), dtype=np.int64),
        )
    with pytest.raises(ModelError, match="wrong exact shape"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=4,
            bootstrap_replicates=2,
            resample_size=4,
            survival_months=2,
            rng=_BadIntegerGenerator(np.random.PCG64(1)),
        )


def test_latent_bootstrap_validates_evidence_and_randomness_contract() -> None:
    simulation = _latent_bootstrap_fixture()
    indices = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(ModelError, match="exactly one"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=4,
            bootstrap_replicates=2,
            resample_size=4,
            survival_months=2,
        )
    with pytest.raises(ModelError, match="exactly one"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=4,
            bootstrap_replicates=2,
            resample_size=4,
            survival_months=2,
            resample_indices=indices,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ModelError, match="numpy Generator"):
        latent_cluster_bootstrap(
            simulation,
            expected_chain_count=4,
            bootstrap_replicates=2,
            resample_size=4,
            survival_months=2,
            rng=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ModelError, match="chain_count metadata"):
        latent_cluster_bootstrap(
            LatentChainSimulation(
                simulation.transition_matrix,
                simulation.stationary_distribution,
                simulation.chains,
                chain_count=3,
                measured_months=4,
                extension_months=3,
                uniforms_per_chain=8,
            ),
            expected_chain_count=3,
            bootstrap_replicates=2,
            resample_size=3,
            survival_months=2,
            resample_indices=np.zeros((2, 3), dtype=np.int64),
        )
    with pytest.raises(ModelError, match="stationary distribution"):
        latent_cluster_bootstrap(
            LatentChainSimulation(
                simulation.transition_matrix,
                np.asarray([0.4, 0.6]),
                simulation.chains,
                chain_count=4,
                measured_months=4,
                extension_months=3,
                uniforms_per_chain=8,
            ),
            expected_chain_count=4,
            bootstrap_replicates=2,
            resample_size=4,
            survival_months=2,
            resample_indices=indices,
        )


def test_mixed_max_stat_respects_upper_tail_and_plus_one_ties() -> None:
    reference = np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    result = mixed_max_statistic_cell(
        [0.0, -100.0],
        reference,
        (EndpointDirection.TWO_SIDED, EndpointDirection.UPPER_TAIL),
        minimum_successes=3,
    )

    assert result.observed_statistic == 0.0
    assert result.p_value == 1.0
    np.testing.assert_allclose(result.replicate_statistics, [1.0, 0.0, 1.0])


def test_constant_component_semantics_are_exact() -> None:
    reference = np.ones((3, 2))
    passing = mixed_max_statistic_cell(
        [1.0, 0.0],
        reference,
        (EndpointDirection.TWO_SIDED, EndpointDirection.UPPER_TAIL),
        minimum_successes=3,
    )
    assert passing.p_value == 1.0
    assert not passing.active_components.any()
    with pytest.raises(ModelError, match="fails a constant"):
        mixed_max_statistic_cell(
            [1.1, 0.0],
            reference,
            (EndpointDirection.TWO_SIDED, EndpointDirection.UPPER_TAIL),
            minimum_successes=3,
        )
    with pytest.raises(ModelError, match="fails a constant"):
        mixed_max_statistic_cell(
            [1.0, 1.1],
            reference,
            (EndpointDirection.TWO_SIDED, EndpointDirection.UPPER_TAIL),
            minimum_successes=3,
        )


def test_four_cell_holm_requires_exactly_four_values() -> None:
    result = four_cell_holm([0.001, 0.01, 0.20, 0.90])
    np.testing.assert_array_equal(result.rejected, [True, True, False, False])
    with pytest.raises(ModelError, match="exactly four"):
        four_cell_holm([0.01, 0.02])


def test_predictive_mixture_residual_pit_cvm_and_covariance_order() -> None:
    values = np.asarray([[0.0, 0.0], [1.0, -1.0]])
    weights = np.ones((2, 1))
    means = np.zeros((1, 2))

    result = predictive_mixture_diagnostics(values, weights, means, np.eye(2))

    np.testing.assert_allclose(result.predictive_residuals, values)
    np.testing.assert_allclose(result.pits, ndtr(values))
    np.testing.assert_allclose(
        result.predictive_covariance_error_upper, [-0.5, -0.5, -0.5]
    )
    assert result.pit_cramer_von_mises.shape == (2,)
    assert not result.pits.flags.writeable


def test_residual_acf_never_crosses_gap_or_viterbi_run() -> None:
    values = np.arange(1.0, 9.0)[:, None]
    continuation = np.asarray([False, True, True, True, False, True, True, True])
    states = np.asarray([0, 0, 0, 1, 1, 1, 1, 1])

    ordinary = residual_acf_diagnostics(values, continuation, (1, 2), minimum_pairs=1)
    within = residual_acf_diagnostics(
        values, continuation, (1,), states=states, minimum_pairs=1
    )

    np.testing.assert_array_equal(ordinary.raw_pair_counts, [[6, 4]])
    np.testing.assert_array_equal(within.raw_pair_counts, [[5]])
    with pytest.raises(ModelError, match="insufficient"):
        residual_acf_diagnostics(values, continuation, (2,), minimum_pairs=5)


def test_marginal_moments_use_population_nonexcess_convention() -> None:
    values = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    result = marginal_moments(values)
    np.testing.assert_allclose(result[:, :2], [[1.5, 1.25]])
    assert result[0, 2] == pytest.approx(0.0)
    assert result[0, 3] == pytest.approx(1.64)
    with pytest.raises(ModelError, match="variance"):
        marginal_moments(np.ones((4, 1)))


def test_latent_contract_validation_fails_closed_on_malformed_evidence() -> None:
    simulation = _latent_bootstrap_fixture()
    common = {
        "expected_chain_count": 4,
        "bootstrap_replicates": 2,
        "resample_size": 4,
        "resample_indices": np.zeros((2, 4), dtype=np.int64),
    }
    with pytest.raises(ModelError, match="LatentChainSimulation"):
        latent_cluster_bootstrap(object(), **common)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="simulated chain horizon"):
        latent_cluster_bootstrap(simulation, survival_months=8, **common)
    with pytest.raises(ModelError, match="constant tolerance"):
        latent_cluster_bootstrap(
            simulation, survival_months=2, constant_tolerance=np.nan, **common
        )
    with pytest.raises(ModelError, match="uniform-consumption metadata"):
        latent_cluster_bootstrap(
            LatentChainSimulation(
                simulation.transition_matrix,
                simulation.stationary_distribution,
                simulation.chains,
                chain_count=4,
                measured_months=4,
                extension_months=3,
                uniforms_per_chain=7,
            ),
            survival_months=2,
            **common,
        )
    with pytest.raises(ModelError, match="stationary distribution is invalid"):
        latent_cluster_bootstrap(
            LatentChainSimulation(
                simulation.transition_matrix,
                np.asarray([np.nan, 0.5]),
                simulation.chains,
                chain_count=4,
                measured_months=4,
                extension_months=3,
                uniforms_per_chain=8,
            ),
            survival_months=2,
            **common,
        )
    with pytest.raises(ModelError, match="integer dtype"):
        latent_cluster_bootstrap(
            LatentChainSimulation(
                simulation.transition_matrix,
                simulation.stationary_distribution,
                simulation.chains.astype(np.float64),
                chain_count=4,
                measured_months=4,
                extension_months=3,
                uniforms_per_chain=8,
            ),
            survival_months=2,
            **common,
        )
    with pytest.raises(ModelError, match="out-of-range state"):
        latent_cluster_bootstrap(
            LatentChainSimulation(
                simulation.transition_matrix,
                simulation.stationary_distribution,
                np.full((4, 8), 2, dtype=np.int64),
                chain_count=4,
                measured_months=4,
                extension_months=3,
                uniforms_per_chain=8,
            ),
            survival_months=2,
            **common,
        )


def test_mixed_max_statistic_rejects_invalid_geometry_and_counts() -> None:
    directions = (EndpointDirection.TWO_SIDED,)
    with pytest.raises(ModelError, match="shape"):
        mixed_max_statistic_cell([0.0], np.zeros((2, 2)), directions)
    with pytest.raises(ModelError, match="non-finite"):
        mixed_max_statistic_cell([0.0], [[0.0], [np.nan]], directions)
    with pytest.raises(ModelError, match="insufficient"):
        mixed_max_statistic_cell([0.0], [[0.0], [1.0]], directions, minimum_successes=3)
    with pytest.raises(ModelError, match="at least two"):
        mixed_max_statistic_cell([0.0], [[0.0]], directions)
    with pytest.raises(ModelError, match="directions"):
        mixed_max_statistic_cell([0.0], [[0.0], [1.0]], ())
    with pytest.raises(ModelError, match="constant tolerance"):
        mixed_max_statistic_cell(
            [0.0], [[0.0], [1.0]], directions, constant_tolerance=-1.0
        )


def test_predictive_mixture_validation_rejects_invalid_inputs() -> None:
    values = np.zeros((2, 2))
    means = np.zeros((1, 2))
    covariance = np.eye(2)
    with pytest.raises(ModelError, match="weight shape"):
        predictive_mixture_diagnostics(values, np.ones((2, 2)), means, covariance)
    with pytest.raises(ModelError, match="non-negative"):
        predictive_mixture_diagnostics(values, -np.ones((2, 1)), means, covariance)
    with pytest.raises(ModelError, match="sum to one"):
        predictive_mixture_diagnostics(values, np.full((2, 1), 0.5), means, covariance)
    with pytest.raises(ModelError, match="dimensions differ"):
        predictive_mixture_diagnostics(
            np.zeros((2, 1)), np.ones((2, 1)), means, covariance
        )
    with pytest.raises(ModelError, match="shape or values"):
        predictive_mixture_diagnostics(values, np.ones((2, 1)), means, np.eye(1))
    with pytest.raises(ModelError, match="positive definite"):
        predictive_mixture_diagnostics(
            values, np.ones((2, 1)), means, np.asarray([[1.0, 0.0], [0.0, 0.0]])
        )


def test_residual_acf_validation_rejects_bad_link_and_run_inputs() -> None:
    values = np.arange(1.0, 7.0)[:, None]
    links = np.asarray([False, True, True, True, True, True])
    with pytest.raises(ModelError, match="non-empty"):
        residual_acf_diagnostics(values, links, ())
    with pytest.raises(ModelError, match="boolean vector"):
        residual_acf_diagnostics(values, np.ones(6, dtype=np.int64), (1,))
    with pytest.raises(ModelError, match="false at row zero"):
        residual_acf_diagnostics(values, np.ones(6, dtype=np.bool_), (1,))
    with pytest.raises(ModelError, match="integer vector"):
        residual_acf_diagnostics(values, links, (1,), states=np.zeros(6))
    with pytest.raises(ModelError, match="denominator"):
        residual_acf_diagnostics(np.ones((6, 1)), links, (1,), minimum_pairs=1)
