"""Tests for fixed-K bridge acceleration and Tier-B execution."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.bridge_fixed_k import (
    ActualBridgeExecution,
    BridgeFixedKPairEvaluation,
    actual_bridge_execution_identity,
    assemble_actual_bridge,
    assemble_bridge_tier_b,
    audit_bridge_accelerator,
    bridge_fixed_k_scope,
    fixed_k_pair_components,
    is_actual_bridge_execution,
    parent_parameters_in_fit_coordinates,
    retain_audited_bridge_pair,
    run_bridge_fixed_k_pair,
)
from prpg.model.bridge_history import generate_registered_bridge_history
from prpg.model.hmm import (
    BridgeAcceleratedHMMFit,
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    fit_gaussian_hmm_bridge_accelerated,
    summarize_state_path,
)
from prpg.model.input import MACRO_COLUMNS
from prpg.simulation.rng import ModelFitScope


def _parent() -> HMMParameters:
    return HMMParameters(
        np.array([0.5, 0.5]),
        np.array([[0.82, 0.18], [0.22, 0.78]]),
        np.array([[-0.5, -0.3, 0.2, 0.4], [0.6, 0.4, -0.2, -0.5]]),
        np.eye(4) * 0.5,
    )


def _design() -> np.ndarray:
    return np.random.default_rng(311).normal(size=(193, 4))


def _mask() -> np.ndarray:
    return np.r_[False, np.ones(192, dtype=np.bool_)]


def _fake_fit(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
) -> GaussianHMMFit:
    assert len(seeds) in {9, 50}
    rows = len(features.values)
    transition = np.full((n_states, n_states), 0.15 / (n_states - 1))
    np.fill_diagonal(transition, 0.85)
    means = np.linspace(-0.5, 0.5, n_states)[:, None] * np.array(
        [[1.0, 0.8, -0.6, -1.0]]
    )
    parameters = HMMParameters(
        np.full(n_states, 1.0 / n_states),
        transition,
        means,
        np.eye(4) * 0.7,
    )
    decoded = np.resize(np.repeat(np.arange(n_states), 5), rows).astype(np.int64)
    stationary = np.full(n_states, 1.0 / n_states)
    order = np.arange(n_states, dtype=np.int64)
    return GaussianHMMFit(
        parameters,
        decoded,
        order,
        order.copy(),
        -float(rows),
        200.0,
        20,
        rows,
        4,
        n_states,
        features.lengths,
        1,
        int(seeds[1]),
        (),
        cast(Any, SimpleNamespace()),
        summarize_state_path(decoded, n_states, lengths=features.lengths),
        cast(Any, SimpleNamespace(stationary_distribution=stationary)),
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        features.scaler_scope,
        np.asarray(features.scaler_means).copy(),
        np.asarray(features.scaler_standard_deviations).copy(),
        features.feature_names,
    )


def _fake_accelerated(
    features: HMMFeatureMatrix,
    n_states: int,
    seeds: Sequence[int],
    parent: HMMParameters,
) -> BridgeAcceleratedHMMFit:
    del parent
    fit = _fake_fit(features, n_states, seeds)
    return BridgeAcceleratedHMMFit(
        fit,
        0,
        tuple(range(9)),
        tuple(range(1, 10)),
        "registered_random_0",
        int(seeds[0]),
    )


def _history(pair: int = 0):
    return generate_registered_bridge_history(
        dgp="model",
        tier="fixed_k",
        pair=pair,
        master_seed=456,
        scientific_version=1,
        design_parameters=_parent(),
        design_standardized_history=_design(),
        design_continuation=_mask(),
        macro_block_length=4,
    )


def _clone_pair(
    result: BridgeFixedKPairEvaluation,
    pair: int,
) -> BridgeFixedKPairEvaluation:
    return replace(
        result,
        pair=pair,
        prefix=replace(result.prefix, pair=pair),
        full=replace(result.full, pair=pair),
    )


def test_fixed_k_scope_registry_is_exact() -> None:
    assert bridge_fixed_k_scope("model", "prefix") is (
        ModelFitScope.BRIDGE_MODEL_FIXED_K_PREFIX
    )
    assert bridge_fixed_k_scope("model", "full") is (
        ModelFitScope.BRIDGE_MODEL_FIXED_K_FULL
    )
    assert bridge_fixed_k_scope("nonparametric", "prefix") is (
        ModelFitScope.BRIDGE_NONPARAMETRIC_FIXED_K_PREFIX
    )
    assert bridge_fixed_k_scope("nonparametric", "full") is (
        ModelFitScope.BRIDGE_NONPARAMETRIC_FIXED_K_FULL
    )
    with pytest.raises(ModelError, match="outside the registry"):
        bridge_fixed_k_scope("bad", "prefix")  # type: ignore[arg-type]


def test_parent_parameters_are_mapped_into_recomputed_scaler_space() -> None:
    parent = _parent()
    means = np.array([1.0, 2.0, 3.0, 4.0])
    scales = np.array([2.0, 3.0, 4.0, 5.0])
    result = parent_parameters_in_fit_coordinates(parent, means, scales)
    np.testing.assert_allclose(result.means, (parent.means - means) / scales)
    np.testing.assert_allclose(
        result.tied_covariance,
        parent.tied_covariance / (scales[:, None] * scales[None, :]),
    )
    with pytest.raises(ModelError, match="coordinates"):
        parent_parameters_in_fit_coordinates(parent, means[:3], scales)


def test_fixed_k_pair_builds_aligned_coordinate_families() -> None:
    history = _history()
    result = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="ordinary_50",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    assert result.valid
    assert result.components is not None
    assert result.components.normalized_coordinates.shape == (34,)
    assert result.components.component_decisions.shape == (34,)
    assert len(result.components.coordinate_family) == 34
    assert len(result.components.coordinate_names) == 34
    assert result.components.coordinate_names[0] == "centroid/0"
    assert result.components.coordinate_names[-1] == "dwell_survival/1/12"
    assert not result.components.normalized_coordinates.flags.writeable
    assert len(result.history_sha256) == 64
    altered_parent = _parent()
    altered_parent = HMMParameters(
        altered_parent.start_probabilities,
        altered_parent.transition_matrix,
        altered_parent.means + 0.01,
        altered_parent.tied_covariance,
    )
    with pytest.raises(ModelError, match="parent parameters"):
        run_bridge_fixed_k_pair(
            history,
            n_states=2,
            parent_parameters=altered_parent,
            master_seed=456,
            scientific_version=1,
            mode="ordinary_50",
            ordinary_fit_function=_fake_fit,
            accelerated_fit_function=_fake_accelerated,
        )


def test_fixed_k_pair_retains_fit_and_component_failures() -> None:
    history = _history()

    def fit_failure(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: Sequence[int],
    ) -> GaussianHMMFit:
        del features, n_states, seeds
        raise ModelError("synthetic member fit failure")

    member_failure = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="ordinary_50",
        ordinary_fit_function=fit_failure,
        accelerated_fit_function=_fake_accelerated,
    )
    assert not member_failure.valid
    assert member_failure.failure_type == "MemberFitFailure"

    def no_completed_dwell(
        features: HMMFeatureMatrix,
        n_states: int,
        seeds: Sequence[int],
    ) -> GaussianHMMFit:
        fit = _fake_fit(features, n_states, seeds)
        decoded = np.zeros(len(features.values), dtype=np.int64)
        return replace(
            fit,
            decoded_states=decoded,
            state_path_diagnostics=summarize_state_path(
                decoded,
                n_states,
                lengths=features.lengths,
            ),
        )

    component_failure = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="ordinary_50",
        ordinary_fit_function=no_completed_dwell,
        accelerated_fit_function=_fake_accelerated,
    )
    assert not component_failure.valid
    assert component_failure.failure_type == "ModelError"
    assert "completed dwell support" in (component_failure.failure_message or "")

    with pytest.raises(ModelError, match="fit mode"):
        run_bridge_fixed_k_pair(
            history,
            n_states=2,
            parent_parameters=_parent(),
            master_seed=456,
            scientific_version=1,
            mode="bad",  # type: ignore[arg-type]
            ordinary_fit_function=_fake_fit,
            accelerated_fit_function=_fake_accelerated,
        )


def test_component_builder_accepts_explicit_parent_coordinate_scalers() -> None:
    def feature(rows: int, lengths: tuple[int, ...]) -> HMMFeatureMatrix:
        return HMMFeatureMatrix(
            values=np.random.default_rng(5 + rows).normal(size=(rows, 4)),
            lengths=lengths,
            feature_names=("a", "b", "c", "d"),
            row_labels=tuple(map(str, range(rows))),
            scaler_means=np.zeros(4),
            scaler_standard_deviations=np.ones(4),
            scaler_scope="production_full",
        )

    first = _fake_fit(feature(193, (193,)), 2, tuple(range(50)))
    second = _fake_fit(feature(217, (210, 7)), 2, tuple(range(50)))
    result = fixed_k_pair_components(
        _parent(),
        first,
        second,
        prefix_scaler_means_in_parent_coordinates=np.zeros(4),
        prefix_scaler_scales_in_parent_coordinates=np.ones(4),
        full_scaler_means_in_parent_coordinates=np.zeros(4),
        full_scaler_scales_in_parent_coordinates=np.ones(4),
    )
    assert result.statistic.raw_maxima[0] == pytest.approx(0.0)
    assert result.statistic.raw_maxima[1] == pytest.approx(0.0)
    assert result.statistic.raw_maxima[2] == pytest.approx(0.0)
    shifted_parameters = HMMParameters(
        second.parameters.start_probabilities,
        second.parameters.transition_matrix,
        second.parameters.means
        + np.array([[0.3, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        second.parameters.tied_covariance,
    )
    shifted_full = replace(second, parameters=shifted_parameters)
    euclidean = fixed_k_pair_components(
        _parent(),
        first,
        shifted_full,
        prefix_scaler_means_in_parent_coordinates=np.zeros(4),
        prefix_scaler_scales_in_parent_coordinates=np.ones(4),
        full_scaler_means_in_parent_coordinates=np.zeros(4),
        full_scaler_scales_in_parent_coordinates=np.ones(4),
    )
    assert euclidean.statistic.raw_maxima[0] == pytest.approx(0.5)
    with pytest.raises(ModelError, match="state counts"):
        fixed_k_pair_components(
            HMMParameters(
                np.full(3, 1 / 3),
                np.full((3, 3), 1 / 3),
                np.zeros((3, 4)),
                np.eye(4),
            ),
            first,
            second,
        )
    with pytest.raises(ModelError, match="prefix/full geometry"):
        fixed_k_pair_components(
            _parent(),
            replace(first, n_observations=192),
            second,
        )
    with pytest.raises(ModelError, match="overrides must be paired"):
        fixed_k_pair_components(
            _parent(),
            first,
            second,
            prefix_scaler_means_in_parent_coordinates=np.zeros(4),
        )


def test_actual_bridge_enforces_exact_geometry_counts_and_kernel_cubes() -> None:
    design_feature = HMMFeatureMatrix(
        values=np.random.default_rng(51).normal(size=(193, 4)),
        lengths=(193,),
        feature_names=tuple(MACRO_COLUMNS),
        row_labels=tuple(map(str, range(193))),
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        scaler_scope="design_training",
    )
    production_feature = HMMFeatureMatrix(
        values=np.random.default_rng(52).normal(size=(217, 4)),
        lengths=(210, 7),
        feature_names=tuple(MACRO_COLUMNS),
        row_labels=tuple(map(str, range(217))),
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        scaler_scope="production_full",
    )
    design = _fake_fit(design_feature, 2, tuple(range(50)))
    production = _fake_fit(production_feature, 2, tuple(range(50)))
    zero_kernel = np.zeros((4, 2, 3))
    result = assemble_actual_bridge(
        design_fit=design,
        production_fit=production,
        design_monthly_block_length=5,
        production_monthly_block_length=5,
        design_daily_block_length=126,
        production_daily_block_length=126,
        design_monthly_state_counts=design.state_path_diagnostics.state_counts,
        production_monthly_state_counts=(
            production.state_path_diagnostics.state_counts
        ),
        design_daily_state_counts=np.array([2_025, 2_024]),
        production_daily_state_counts=np.array([2_274, 2_273]),
        design_monthly_kernel_offsets=zero_kernel,
        production_monthly_kernel_offsets=zero_kernel,
        design_daily_kernel_offsets=zero_kernel,
        production_daily_kernel_offsets=zero_kernel,
    )
    assert result.gate.passed
    assert result.n_states == 2
    identity = actual_bridge_execution_identity(result)
    assert identity.execution_fingerprint == result.execution_fingerprint
    assert identity.source_materialization_fingerprint == (
        result.source_materialization_fingerprint
    )
    assert is_actual_bridge_execution(result)
    assert not is_actual_bridge_execution(copy.copy(result))
    with pytest.raises(TypeError, match="created only"):
        ActualBridgeExecution()
    np.testing.assert_array_equal(result.kernel_lambdas, [0.25, 0.50, 0.75, 1.00])
    assert not result.monthly_kernel_offset_differences.flags.writeable
    assert not result.daily_kernel_offset_differences.flags.writeable
    original_sources = result._sources
    substituted_kernel = original_sources.production_daily_kernel_offsets.copy()
    substituted_kernel[0, 0, 0] += 1e-6
    substituted_kernel.setflags(write=False)
    object.__setattr__(
        result,
        "_sources",
        replace(
            original_sources,
            production_daily_kernel_offsets=substituted_kernel,
        ),
    )
    with pytest.raises(ModelError, match="exact sources"):
        actual_bridge_execution_identity(result)
    object.__setattr__(result, "_sources", original_sources)
    changed = result.daily_kernel_offset_differences.copy()
    changed[0, 0, 0] += 1e-6
    changed.setflags(write=False)
    object.__setattr__(result, "daily_kernel_offset_differences", changed)
    with pytest.raises(ModelError, match="exact sources"):
        actual_bridge_execution_identity(result)
    with pytest.raises(ModelError, match="outside the approved bounds"):
        assemble_actual_bridge(
            design_fit=design,
            production_fit=production,
            design_monthly_block_length=5,
            production_monthly_block_length=5,
            design_daily_block_length=127,
            production_daily_block_length=126,
            design_monthly_state_counts=design.state_path_diagnostics.state_counts,
            production_monthly_state_counts=(
                production.state_path_diagnostics.state_counts
            ),
            design_daily_state_counts=np.array([2_025, 2_024]),
            production_daily_state_counts=np.array([2_274, 2_273]),
            design_monthly_kernel_offsets=zero_kernel,
            production_monthly_kernel_offsets=zero_kernel,
            design_daily_kernel_offsets=zero_kernel,
            production_daily_kernel_offsets=zero_kernel,
        )
    with pytest.raises(ModelError, match="shape"):
        assemble_actual_bridge(
            design_fit=design,
            production_fit=production,
            design_monthly_block_length=5,
            production_monthly_block_length=5,
            design_daily_block_length=20,
            production_daily_block_length=20,
            design_monthly_state_counts=(design.state_path_diagnostics.state_counts),
            production_monthly_state_counts=(
                production.state_path_diagnostics.state_counts
            ),
            design_daily_state_counts=np.array([2_025, 2_024]),
            production_daily_state_counts=np.array([2_274, 2_273]),
            design_monthly_kernel_offsets=np.zeros((2, 2, 3)),
            production_monthly_kernel_offsets=zero_kernel,
            design_daily_kernel_offsets=zero_kernel,
            production_daily_kernel_offsets=zero_kernel,
        )


def test_actual_bridge_aligns_production_counts_and_kernel_offsets() -> None:
    design_feature = HMMFeatureMatrix(
        values=np.random.default_rng(61).normal(size=(193, 4)),
        lengths=(193,),
        feature_names=tuple(MACRO_COLUMNS),
        row_labels=tuple(map(str, range(193))),
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        scaler_scope="design_training",
    )
    production_feature = HMMFeatureMatrix(
        values=np.random.default_rng(62).normal(size=(217, 4)),
        lengths=(210, 7),
        feature_names=tuple(MACRO_COLUMNS),
        row_labels=tuple(map(str, range(217))),
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        scaler_scope="production_full",
    )
    design = _fake_fit(design_feature, 2, tuple(range(50)))
    production = _fake_fit(production_feature, 2, tuple(range(50)))
    permutation = np.array([1, 0], dtype=np.int64)
    parameters = production.parameters
    swapped_parameters = HMMParameters(
        parameters.start_probabilities[permutation],
        parameters.transition_matrix[np.ix_(permutation, permutation)],
        parameters.means[permutation],
        parameters.tied_covariance,
    )
    swapped_states = permutation[production.decoded_states]
    swapped_production = replace(
        production,
        parameters=swapped_parameters,
        decoded_states=swapped_states,
        state_path_diagnostics=summarize_state_path(
            swapped_states,
            2,
            lengths=(210, 7),
        ),
    )
    monthly_kernel = np.zeros((4, 2, 3))
    monthly_kernel[:, 0, :] = 0.0001
    monthly_kernel[:, 1, :] = 0.00005
    daily_kernel = np.zeros((4, 2, 3))
    daily_kernel[:, 0, :] = 0.00001
    daily_kernel[:, 1, :] = 0.000005
    result = assemble_actual_bridge(
        design_fit=design,
        production_fit=swapped_production,
        design_monthly_block_length=5,
        production_monthly_block_length=5,
        design_daily_block_length=20,
        production_daily_block_length=20,
        design_monthly_state_counts=design.state_path_diagnostics.state_counts,
        production_monthly_state_counts=(
            swapped_production.state_path_diagnostics.state_counts
        ),
        design_daily_state_counts=np.array([2_000, 2_049]),
        production_daily_state_counts=np.array([2_347, 2_200]),
        design_monthly_kernel_offsets=np.zeros((4, 2, 3)),
        production_monthly_kernel_offsets=monthly_kernel,
        design_daily_kernel_offsets=np.zeros((4, 2, 3)),
        production_daily_kernel_offsets=daily_kernel,
    )
    assert not result.gate.same_order
    np.testing.assert_allclose(
        result.gate.daily_support_ratios,
        np.array([2_200 / 2_000, 2_347 / 2_049]),
    )
    assert result.gate.maximum_annualized_monthly_kernel_change == pytest.approx(0.0012)
    assert result.gate.maximum_annualized_daily_kernel_change == pytest.approx(0.00252)


def test_accelerator_audit_requires_99_and_selects_one_result_family() -> None:
    history = _history()
    accelerated_zero = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="accelerated_10",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    ordinary_zero = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="ordinary_50",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    accelerated = [_clone_pair(accelerated_zero, pair) for pair in range(100)]
    ordinary = [_clone_pair(ordinary_zero, pair) for pair in range(100)]
    assert accelerated[0].components is not None
    shifted = accelerated[0].components.normalized_coordinates.copy()
    shifted[0] += 0.002
    shifted.setflags(write=False)
    accelerated[0] = replace(
        accelerated[0],
        components=replace(
            accelerated[0].components,
            normalized_coordinates=shifted,
        ),
    )
    audit = audit_bridge_accelerator(
        accelerated,
        ordinary,
        dgp="model",
        n_states=2,
    )
    assert audit.result.agreeing_pairs == 99
    assert audit.eligible
    assert (
        retain_audited_bridge_pair(accelerated[0], ordinary[0], audit)
        is (accelerated[0])
    )
    with pytest.raises(ModelError, match="exactly 100"):
        audit_bridge_accelerator(
            accelerated[:-1],
            ordinary,
            dgp="model",
            n_states=2,
        )
    incomplete_history_audit = replace(
        audit,
        history_sha256=audit.history_sha256[:-1],
    )
    with pytest.raises(ModelError, match="history identities"):
        retain_audited_bridge_pair(
            accelerated[0],
            ordinary[0],
            incomplete_history_audit,
        )
    inconsistent_decision = replace(
        audit,
        result=replace(audit.result, eligible=False),
    )
    with pytest.raises(ModelError, match="decision is inconsistent"):
        retain_audited_bridge_pair(
            accelerated[0],
            ordinary[0],
            inconsistent_decision,
        )

    accelerated[1] = replace(
        accelerated[1],
        components=replace(
            accelerated[1].components,
            normalized_coordinates=shifted,
        ),
    )
    failed = audit_bridge_accelerator(
        accelerated,
        ordinary,
        dgp="model",
        n_states=2,
    )
    assert failed.result.agreeing_pairs == 98
    assert not failed.eligible
    assert (
        retain_audited_bridge_pair(accelerated[0], ordinary[0], failed) is (ordinary[0])
    )


def test_accelerator_audit_requires_identical_pseudo_history_bytes() -> None:
    first_history = _history()
    second_history = generate_registered_bridge_history(
        dgp="model",
        tier="fixed_k",
        pair=0,
        master_seed=999,
        scientific_version=1,
        design_parameters=_parent(),
        design_standardized_history=_design(),
        design_continuation=_mask(),
        macro_block_length=4,
    )
    accelerated_zero = run_bridge_fixed_k_pair(
        first_history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="accelerated_10",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    ordinary_zero = run_bridge_fixed_k_pair(
        second_history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=999,
        scientific_version=1,
        mode="ordinary_50",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    with pytest.raises(ModelError, match="identical history bytes"):
        audit_bridge_accelerator(
            [_clone_pair(accelerated_zero, pair) for pair in range(100)],
            [_clone_pair(ordinary_zero, pair) for pair in range(100)],
            dgp="model",
            n_states=2,
        )


def test_matching_member_failures_can_agree_in_accelerator_audit() -> None:
    history = _history()
    accelerated_zero = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="accelerated_10",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    ordinary_zero = run_bridge_fixed_k_pair(
        history,
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="ordinary_50",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )

    def invalidate_full(
        result: BridgeFixedKPairEvaluation,
    ) -> BridgeFixedKPairEvaluation:
        return replace(
            result,
            full=replace(
                result.full,
                fit=None,
                accelerated=None,
                failure_type="ModelError",
                failure_message="synthetic common failure",
            ),
            components=None,
            failure_type="MemberFitFailure",
            failure_message="synthetic common failure",
        )

    accelerated_zero = invalidate_full(accelerated_zero)
    ordinary_zero = invalidate_full(ordinary_zero)
    audit = audit_bridge_accelerator(
        [_clone_pair(accelerated_zero, pair) for pair in range(100)],
        [_clone_pair(ordinary_zero, pair) for pair in range(100)],
        dgp="model",
        n_states=2,
    )
    assert audit.result.agreeing_pairs == 100
    assert audit.eligible


def test_tier_b_accepts_only_first_binding_look() -> None:
    fast_zero = run_bridge_fixed_k_pair(
        _history(),
        n_states=2,
        parent_parameters=_parent(),
        master_seed=456,
        scientific_version=1,
        mode="accelerated_10",
        ordinary_fit_function=_fake_fit,
        accelerated_fit_function=_fake_accelerated,
    )
    ordinary_zero = replace(fast_zero, mode="ordinary_50")
    ordinary_zero = replace(
        ordinary_zero,
        prefix=replace(ordinary_zero.prefix, mode="ordinary_50"),
        full=replace(ordinary_zero.full, mode="ordinary_50"),
    )
    fast_audit = [_clone_pair(fast_zero, pair) for pair in range(100)]
    ordinary_audit = [_clone_pair(ordinary_zero, pair) for pair in range(100)]
    audit = audit_bridge_accelerator(
        fast_audit,
        ordinary_audit,
        dgp="model",
        n_states=2,
    )
    assert audit.eligible
    pairs = [_clone_pair(fast_zero, pair) for pair in range(1_000)]
    actual = max(float(result.statistic or 0.0) for result in pairs)
    execution = assemble_bridge_tier_b(
        pairs,
        dgp="model",
        n_states=2,
        actual_statistic=actual,
        accelerator=audit,
    )
    assert execution.gate.passed
    assert execution.gate.pairs_used == 1_000
    with pytest.raises(ModelError, match="continued beyond"):
        assemble_bridge_tier_b(
            [_clone_pair(fast_zero, pair) for pair in range(2_000)],
            dgp="model",
            n_states=2,
            actual_statistic=actual,
            accelerator=audit,
        )


class _Monitor:
    converged = True
    iter = 3
    history = (0.0, 1.0, 2.0)


class _WarmModel:
    def __init__(self, owner: _WarmFactory, kwargs: dict[str, Any]) -> None:
        self.owner = owner
        self.seed = int(kwargs["random_state"])
        self.monitor_ = _Monitor()
        self.startprob_ = np.array([0.5, 0.5])
        self.transmat_ = np.array([[0.8, 0.2], [0.2, 0.8]])
        self.means_ = np.array([[-2.0, -2.0], [2.0, 2.0]])
        self._covars_ = np.eye(2) * 0.25
        self.covars_ = np.repeat(self._covars_[None], 2, axis=0)
        self.raw_covariance_min_eigenvalues_ = [0.25] * 3
        self.covariance_floor_relative_changes_ = [0.0] * 3

    def fit(
        self,
        values: np.ndarray,
        lengths: Sequence[int] | None = None,
    ) -> _WarmModel:
        del values, lengths
        self.owner.fitted_parameters.append(
            (
                self.startprob_.copy(),
                self.transmat_.copy(),
                self.means_.copy(),
                self._covars_.copy(),
            )
        )
        return self

    def score(
        self,
        values: np.ndarray,
        lengths: Sequence[int] | None = None,
    ) -> float:
        del values, lengths
        return float(self.seed)


class _WarmFactory:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.fitted_parameters: list[tuple[np.ndarray, ...]] = []

    def __call__(self, **kwargs: Any) -> _WarmModel:
        self.kwargs.append(kwargs)
        return _WarmModel(self, kwargs)


def test_accelerated_hmm_has_one_warm_and_nine_registered_random_initializers() -> None:
    latent = np.resize(np.repeat([0, 1], 5), 100)
    values = np.column_stack(
        (
            np.where(latent == 0, -2.0, 2.0),
            np.where(latent == 0, -2.0, 2.0),
        )
    )
    features = HMMFeatureMatrix(
        values=values,
        lengths=(100,),
        feature_names=("a", "b"),
        row_labels=tuple(map(str, range(100))),
        scaler_means=np.zeros(2),
        scaler_standard_deviations=np.ones(2),
        scaler_scope="production_full",
    )
    parent = HMMParameters(
        np.array([0.6, 0.4]),
        np.array([[0.75, 0.25], [0.15, 0.85]]),
        np.array([[-3.0, -3.0], [3.0, 3.0]]),
        np.eye(2) * 0.4,
    )
    factory = _WarmFactory()
    result = fit_gaussian_hmm_bridge_accelerated(
        features,
        2,
        (100, 1, 2, 3, 4, 5, 6, 7, 8),
        parent,
        model_factory=factory,
    )
    assert len(factory.kwargs) == 10
    assert factory.kwargs[0]["init_params"] == ""
    assert all(kwargs["init_params"] == "stmc" for kwargs in factory.kwargs[1:])
    np.testing.assert_array_equal(
        factory.fitted_parameters[0][0], parent.start_probabilities
    )
    np.testing.assert_array_equal(
        factory.fitted_parameters[0][1], parent.transition_matrix
    )
    np.testing.assert_array_equal(factory.fitted_parameters[0][2], parent.means)
    np.testing.assert_array_equal(
        factory.fitted_parameters[0][3], parent.tied_covariance
    )
    assert result.best_initialization == "parent_warm"
    assert result.fit.best_restart_index == 0
    assert result.random_restart_ids == tuple(range(9))
