from __future__ import annotations

import copy
import inspect
from dataclasses import fields, replace
from functools import partial
from typing import Any

import pytest
from tests.unit.test_g3_calibration_views import (
    _MASTER_SEED,
    _SCIENTIFIC_VERSION,
    _candidate_views,
    _fake_production_fit,
    _production_features,
)

import prpg.model.g3_calibration_views as views_module
import prpg.model.production_fit_execution as execution_module
from prpg.errors import IntegrityError, ModelError
from prpg.model.g3_calibration_views import selected_design_fit_from_view
from prpg.model.hmm import (
    HMM_RESTARTS,
    GaussianHMMFit,
    HMMFeatureMatrix,
    NoNumericallyValidHMMRestarts,
    RestartDiagnostic,
    deterministic_hmm_restart_seeds,
)
from prpg.model.production_fit_execution import (
    RegisteredProductionFitNoValidRestarts,
    RegisteredProductionFitSameK,
    execute_registered_production_fit_same_k,
    is_registered_production_fit_no_valid_restarts,
    is_registered_production_fit_same_k,
    registered_production_fit_no_valid_restarts_identity,
    registered_production_fit_no_valid_restarts_sources,
    registered_production_fit_same_k_identity,
    registered_production_fit_same_k_sources,
)
from prpg.simulation.rng import ModelFitScope


def _successful_diagnostic(
    index: int, seed: int, likelihood: float
) -> RestartDiagnostic:
    return RestartDiagnostic(
        restart_index=index,
        seed=seed,
        converged=True,
        iterations=12,
        log_likelihood=likelihood,
        valid=True,
        raw_minimum_eigenvalue=0.25,
        maximum_floor_relative_change=0.0,
        covariance_condition_number=1.0,
        maximum_likelihood_decrease=0.0,
        preliminary_chain_passed=True,
        failure_type=None,
        failure_message=None,
    )


def _failed_diagnostic(index: int, seed: int) -> RestartDiagnostic:
    return RestartDiagnostic(
        restart_index=index,
        seed=seed,
        converged=False,
        iterations=0,
        log_likelihood=None,
        valid=False,
        raw_minimum_eigenvalue=None,
        maximum_floor_relative_change=None,
        covariance_condition_number=None,
        maximum_likelihood_decrease=None,
        preliminary_chain_passed=False,
        failure_type="NonConvergence",
        failure_message="bounded synthetic restart failure",
    )


def _registered_fake_fit(
    features: HMMFeatureMatrix,
    *,
    selected_n_states: int,
    master_seed: int,
    scientific_version: int,
) -> GaussianHMMFit:
    base = _fake_production_fit(
        features,
        selected_n_states=selected_n_states,
        master_seed=master_seed,
        scientific_version=scientific_version,
    )
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=selected_n_states,
    )
    diagnostics = (
        _successful_diagnostic(0, seeds[0], base.log_likelihood),
        *(
            _failed_diagnostic(index, seed)
            for index, seed in enumerate(seeds[1:], start=1)
        ),
    )
    return replace(
        base,
        best_restart_index=0,
        best_seed=seeds[0],
        restart_diagnostics=diagnostics,
    )


def _execute(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, HMMFeatureMatrix, RegisteredProductionFitSameK, list[object]]:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()
    calls: list[object] = []

    def spy(
        supplied: HMMFeatureMatrix,
        *,
        selected_n_states: int,
        master_seed: int,
        scientific_version: int,
    ) -> GaussianHMMFit:
        calls.append(
            (
                supplied,
                selected_n_states,
                master_seed,
                scientific_version,
            )
        )
        return _registered_fake_fit(
            supplied,
            selected_n_states=selected_n_states,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )

    monkeypatch.setattr(execution_module, "fit_production_selected_k", spy)
    result = execute_registered_production_fit_same_k(
        selection,
        features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    assert isinstance(result, RegisteredProductionFitSameK)
    return selection, features, result, calls


def _no_valid_restarts_signal(
    *,
    selected_n_states: int,
    master_seed: int,
    scientific_version: int,
) -> NoNumericallyValidHMMRestarts:
    seeds = deterministic_hmm_restart_seeds(
        master_seed=master_seed,
        scientific_version=scientific_version,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        n_states=selected_n_states,
    )
    diagnostics = tuple(
        _failed_diagnostic(index, seed) for index, seed in enumerate(seeds)
    )
    return NoNumericallyValidHMMRestarts(
        n_states=selected_n_states,
        restart_diagnostics=diagnostics,
    )


def _execute_no_valid_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Any,
    HMMFeatureMatrix,
    RegisteredProductionFitNoValidRestarts,
]:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()

    def fail(
        _supplied: HMMFeatureMatrix,
        *,
        selected_n_states: int,
        master_seed: int,
        scientific_version: int,
    ) -> GaussianHMMFit:
        raise _no_valid_restarts_signal(
            selected_n_states=selected_n_states,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )

    monkeypatch.setattr(execution_module, "fit_production_selected_k", fail)
    result = execute_registered_production_fit_same_k(
        selection,
        features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    assert isinstance(result, RegisteredProductionFitNoValidRestarts)
    return selection, features, result


def _authorize_iteration_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in only for the fixed-point wrapper in focused producer tests."""

    monkeypatch.setattr(
        views_module,
        "is_registered_candidate_selection_task_view",
        views_module.is_candidate_selection_task_view,
    )


def test_legacy_selection_is_rejected_before_the_fit_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, selection = _candidate_views()
    features = _production_features()
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> GaussianHMMFit:
        nonlocal called
        called = True
        raise AssertionError("fit must remain locked")

    monkeypatch.setattr(execution_module, "fit_production_selected_k", forbidden)
    with pytest.raises(ModelError, match="final fixed-point"):
        execute_registered_production_fit_same_k(
            selection,
            features,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )
    assert not called


def test_registered_executor_runs_once_and_retains_exact_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, features, result, calls = _execute(monkeypatch)

    assert len(calls) == 1
    assert calls[0][0] is features
    assert calls[0][1:] == (2, _MASTER_SEED, _SCIENTIFIC_VERSION)
    assert (
        "fit_function"
        not in inspect.signature(execute_registered_production_fit_same_k).parameters
    )
    assert is_registered_production_fit_same_k(result)
    identity = registered_production_fit_same_k_identity(result)
    assert identity.design_selected_k == 2
    assert result.restart_seeds == tuple(
        diagnostic.seed for diagnostic in result.production_fit.restart_diagnostics
    )
    assert len(result.restart_seeds) == HMM_RESTARTS
    retained = registered_production_fit_same_k_sources(result)
    assert retained[0] is selection
    assert retained[1] is selected_design_fit_from_view(selection)
    assert retained[2] is features
    assert retained[3] is result.production_fit
    assert not result.production_fit.parameters.transition_matrix.flags.writeable


def test_registered_executor_propagates_nine_restart_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()
    fit_functions: list[object] = []

    def spy(
        supplied: HMMFeatureMatrix,
        *,
        selected_n_states: int,
        master_seed: int,
        scientific_version: int,
        fit_function: object,
    ) -> GaussianHMMFit:
        fit_functions.append(fit_function)
        return _registered_fake_fit(
            supplied,
            selected_n_states=selected_n_states,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )

    monkeypatch.setattr(execution_module, "fit_production_selected_k", spy)
    result = execute_registered_production_fit_same_k(
        selection,
        features,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        workers=9,
    )

    assert isinstance(result, RegisteredProductionFitSameK)
    assert len(fit_functions) == 1
    assert isinstance(fit_functions[0], partial)
    assert fit_functions[0].keywords == {"workers": 9}


def test_exactly_fifty_invalid_numerical_restarts_mint_sealed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, features, result = _execute_no_valid_restarts(monkeypatch)

    assert result.outcome == "no_numerically_valid_restarts"
    assert len(result.restart_diagnostics) == HMM_RESTARTS
    assert result.restart_seeds == tuple(
        item.seed for item in result.restart_diagnostics
    )
    assert tuple(item.restart_index for item in result.restart_diagnostics) == tuple(
        range(HMM_RESTARTS)
    )
    assert all(not item.valid for item in result.restart_diagnostics)
    assert is_registered_production_fit_no_valid_restarts(result)
    identity = registered_production_fit_no_valid_restarts_identity(result)
    assert identity.design_selected_k == 2
    assert identity.restart_diagnostics_fingerprint == (
        result.restart_diagnostics_fingerprint
    )
    retained = registered_production_fit_no_valid_restarts_sources(result)
    assert retained[0] is selection
    assert retained[1] is selected_design_fit_from_view(selection)
    assert retained[2] is features
    assert result.generation_authorized is False


@pytest.mark.parametrize("defect", ["forty_nine", "wrong_seed", "wrong_index"])
def test_incomplete_or_misaligned_no_valid_signal_is_rejected_without_authority(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()
    signal = _no_valid_restarts_signal(
        selected_n_states=2,
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    diagnostics = signal.restart_diagnostics
    if defect == "forty_nine":
        object.__setattr__(signal, "restart_diagnostics", diagnostics[:-1])
    elif defect == "wrong_seed":
        changed = replace(diagnostics[0], seed=diagnostics[0].seed + 1)
        object.__setattr__(
            signal,
            "restart_diagnostics",
            (changed, *diagnostics[1:]),
        )
    else:
        changed = replace(diagnostics[0], restart_index=1)
        object.__setattr__(
            signal,
            "restart_diagnostics",
            (changed, *diagnostics[1:]),
        )

    def fail(*_args: object, **_kwargs: object) -> GaussianHMMFit:
        raise signal

    before = len(execution_module._MINTED_PRODUCTION_FIT_FAILURES)
    monkeypatch.setattr(execution_module, "fit_production_selected_k", fail)
    with pytest.raises(ModelError, match="exactly 50|registered seed schedule"):
        execute_registered_production_fit_same_k(
            selection,
            features,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )
    assert len(execution_module._MINTED_PRODUCTION_FIT_FAILURES) == before


def test_no_valid_restart_outcome_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result = _execute_no_valid_restarts(monkeypatch)

    changed = replace(
        result.restart_diagnostics[0],
        failure_message="tampered numerical diagnosis",
    )
    object.__setattr__(
        result,
        "restart_diagnostics",
        (changed, *result.restart_diagnostics[1:]),
    )
    assert not is_registered_production_fit_no_valid_restarts(result)
    with pytest.raises(ModelError, match="authority|changed"):
        registered_production_fit_no_valid_restarts_identity(result)


def test_no_valid_restart_constructor_copy_and_stolen_seal_have_no_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result = _execute_no_valid_restarts(monkeypatch)

    with pytest.raises(TypeError, match="created only"):
        RegisteredProductionFitNoValidRestarts()
    assert not is_registered_production_fit_no_valid_restarts(copy.copy(result))

    forged = object.__new__(RegisteredProductionFitNoValidRestarts)
    for item in fields(RegisteredProductionFitNoValidRestarts):
        object.__setattr__(forged, item.name, getattr(result, item.name))
    assert forged._execution_seal is result._execution_seal
    assert not is_registered_production_fit_no_valid_restarts(forged)


def test_constructor_copy_replace_and_stolen_seal_have_no_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, _ = _execute(monkeypatch)

    with pytest.raises(TypeError, match="created only"):
        RegisteredProductionFitSameK()
    assert not is_registered_production_fit_same_k(copy.copy(result))
    with pytest.raises(TypeError):
        replace(result, master_seed=result.master_seed + 1)

    forged = object.__new__(RegisteredProductionFitSameK)
    for item in fields(RegisteredProductionFitSameK):
        object.__setattr__(forged, item.name, getattr(result, item.name))
    assert forged._execution_seal is result._execution_seal
    assert not is_registered_production_fit_same_k(forged)


def test_source_and_result_mutation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, features, result, _ = _execute(monkeypatch)

    features.values[0, 0] += 0.25
    with pytest.raises(ModelError, match="sources or result changed"):
        registered_production_fit_same_k_identity(result)

    _, _, fresh, _ = _execute(monkeypatch)
    with pytest.raises(ValueError):
        fresh.production_fit.parameters.transition_matrix[0, 0] = 0.5
    object.__setattr__(
        fresh.production_fit,
        "bic",
        fresh.production_fit.bic + 1.0,
    )
    with pytest.raises(ModelError, match="sources or result changed"):
        registered_production_fit_same_k_identity(fresh)


def test_equivalent_parent_or_result_substitution_fails_object_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, features, result, _ = _execute(monkeypatch)
    _, _, equivalent_selection = _candidate_views()
    equivalent_features = _production_features()
    assert equivalent_selection is not selection
    assert equivalent_selection.view_fingerprint == selection.view_fingerprint
    assert equivalent_features is not features

    object.__setattr__(result, "_selection_source", equivalent_selection)
    object.__setattr__(
        result,
        "_design_fit_source",
        selected_design_fit_from_view(equivalent_selection),
    )
    assert not is_registered_production_fit_same_k(result)

    _, _, feature_result, _ = _execute(monkeypatch)
    object.__setattr__(
        feature_result, "_production_features_source", equivalent_features
    )
    assert not is_registered_production_fit_same_k(feature_result)

    _, _, fit_result, _ = _execute(monkeypatch)
    object.__setattr__(
        fit_result,
        "production_fit",
        copy.copy(fit_result.production_fit),
    )
    assert not is_registered_production_fit_same_k(fit_result)


@pytest.mark.parametrize("defect", ["scope", "feature_names", "lengths"])
def test_wrong_production_feature_identity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()
    if defect == "scope":
        features = replace(features, scaler_scope="design_training")
    elif defect == "feature_names":
        features = replace(
            features,
            feature_names=("wrong", *features.feature_names[1:]),
        )
    else:
        features = replace(features, lengths=(217,))
    monkeypatch.setattr(
        execution_module,
        "fit_production_selected_k",
        _registered_fake_fit,
    )

    with pytest.raises(ModelError, match="feature geometry"):
        execute_registered_production_fit_same_k(
            selection,
            features,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )


@pytest.mark.parametrize("defect", ["wrong_k", "wrong_seed", "wrong_scope"])
def test_wrong_fit_k_scope_or_restart_seed_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()

    def defective_fit(
        supplied: HMMFeatureMatrix,
        *,
        selected_n_states: int,
        master_seed: int,
        scientific_version: int,
    ) -> GaussianHMMFit:
        fit = _registered_fake_fit(
            supplied,
            selected_n_states=selected_n_states,
            master_seed=master_seed,
            scientific_version=scientific_version,
        )
        if defect == "wrong_k":
            return replace(fit, n_states=3)
        if defect == "wrong_scope":
            return replace(fit, scaler_scope="design_training")
        first = replace(fit.restart_diagnostics[0], seed=fit.best_seed + 1)
        return replace(
            fit,
            restart_diagnostics=(first, *fit.restart_diagnostics[1:]),
        )

    monkeypatch.setattr(
        execution_module,
        "fit_production_selected_k",
        defective_fit,
    )
    with pytest.raises(ModelError, match="inconsistent|registered seeds"):
        execute_registered_production_fit_same_k(
            selection,
            features,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )


@pytest.mark.parametrize(
    "failure",
    [
        ModelError("unexpected generic model defect"),
        IntegrityError("unexpected integrity defect"),
        RuntimeError("unexpected fitter defect"),
        OSError("unexpected filesystem defect"),
        MemoryError("unexpected memory defect"),
    ],
)
def test_generic_fit_faults_propagate_without_sealed_failure_conversion(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    _authorize_iteration_fixture(monkeypatch)
    _, _, selection = _candidate_views()
    features = _production_features()

    def fail(*_args: object, **_kwargs: object) -> GaussianHMMFit:
        raise failure

    monkeypatch.setattr(execution_module, "fit_production_selected_k", fail)
    before = len(execution_module._MINTED_PRODUCTION_FIT_FAILURES)
    with pytest.raises(type(failure)) as captured:
        execute_registered_production_fit_same_k(
            selection,
            features,
            master_seed=_MASTER_SEED,
            scientific_version=_SCIENTIFIC_VERSION,
        )
    assert captured.value is failure
    assert len(execution_module._MINTED_PRODUCTION_FIT_FAILURES) == before
