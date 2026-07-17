from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.hmm import (
    GaussianHMMFit,
    HMMFeatureMatrix,
    HMMParameters,
    decode_viterbi,
    forward_backward,
    gaussian_hmm_bic,
    gaussian_hmm_parameter_count,
    historical_transition_support_report,
    posterior_identification_report,
    summarize_state_path,
    tied_covariance_diagnostic,
    validate_numerical_chain,
)
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    hmm_parameter_fingerprint,
    rng_execution_fingerprint,
    scientific_array_fingerprint,
    scientific_materialization_fingerprint,
)

_FEATURE_NAMES = ("first", "second", "third", "fourth")


def _fit() -> GaussianHMMFit:
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        means=np.asarray([[-3.0, -0.2, 0.1, -0.1], [3.0, 0.2, -0.1, 0.1]]),
        tied_covariance=np.eye(4) * 0.25,
    )
    states = np.tile(np.repeat(np.asarray([0, 1]), 5), 20)
    values = parameters.means[states] + np.random.default_rng(55).normal(
        0.0, 0.1, size=(len(states), 4)
    )
    decoded = decode_viterbi(values, parameters, lengths=(len(values),))
    path = summarize_state_path(decoded, 2, lengths=(len(values),))
    posterior = forward_backward(values, parameters, lengths=(len(values),))
    identification = posterior_identification_report(
        posterior.posterior_probabilities,
        decoded,
        parameters,
    )
    support = historical_transition_support_report(
        parameters.transition_matrix,
        path.transition_counts,
        posterior.expected_transition_counts,
    )
    return GaussianHMMFit(
        parameters=parameters,
        decoded_states=decoded,
        canonical_to_original=np.asarray([0, 1]),
        original_to_canonical=np.asarray([0, 1]),
        log_likelihood=posterior.total_log_likelihood,
        bic=gaussian_hmm_bic(posterior.total_log_likelihood, len(values), 2, 4),
        parameter_count=gaussian_hmm_parameter_count(2, 4),
        n_observations=len(values),
        n_features=4,
        n_states=2,
        lengths=(len(values),),
        best_restart_index=0,
        best_seed=1,
        restart_diagnostics=(),
        covariance_diagnostic=tied_covariance_diagnostic(parameters.tied_covariance),
        state_path_diagnostics=path,
        chain_diagnostics=validate_numerical_chain(parameters.transition_matrix),
        forward_backward=posterior,
        identification=identification,
        historical_transition_support=support,
        scaler_scope="design_training",
        scaler_means=np.zeros(4),
        scaler_standard_deviations=np.ones(4),
        feature_names=_FEATURE_NAMES,
    )


def _features() -> HMMFeatureMatrix:
    return HMMFeatureMatrix(
        values=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=">f4"),
        lengths=(2,),
        feature_names=("first", "second"),
        row_labels=("2000-01", "2000-02"),
        scaler_means=np.asarray([0.0, 0.0]),
        scaler_standard_deviations=np.asarray([1.0, 1.0]),
        scaler_scope="design_training",
    )


def test_array_identity_is_platform_normalized_and_role_bound() -> None:
    big_endian = np.asarray([[1.0, 2.0]], dtype=">f4")
    little_endian = np.asarray([[1.0, 2.0]], dtype="<f8")

    assert scientific_array_fingerprint(
        big_endian, role="transition"
    ) == scientific_array_fingerprint(little_endian, role="transition")
    assert scientific_array_fingerprint(
        little_endian, role="transition"
    ) != scientific_array_fingerprint(little_endian, role="other_transition")
    assert scientific_array_fingerprint(
        little_endian, role="transition"
    ) != scientific_array_fingerprint(little_endian.reshape(2, 1), role="transition")


def test_hmm_parameter_and_feature_fingerprints_bind_exact_content() -> None:
    parameters = HMMParameters(
        start_probabilities=np.asarray([0.6, 0.4]),
        transition_matrix=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        means=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        tied_covariance=np.eye(2),
    )
    changed_parameters = replace(
        parameters,
        transition_matrix=np.asarray([[0.7, 0.3], [0.3, 0.7]]),
    )
    assert hmm_parameter_fingerprint(parameters) != hmm_parameter_fingerprint(
        changed_parameters
    )

    features = _features()
    same_values_different_labels = replace(
        features,
        row_labels=("2000-01", "2000-03"),
    )
    assert hmm_feature_matrix_fingerprint(features) != hmm_feature_matrix_fingerprint(
        same_values_different_labels
    )


def test_full_hmm_identity_binds_model_scaler_path_and_geometry() -> None:
    fit = _fit()
    fingerprint = gaussian_hmm_fit_fingerprint(fit)
    assert fingerprint == gaussian_hmm_fit_fingerprint(fit)
    assert fingerprint != gaussian_hmm_fit_fingerprint(
        replace(fit, scaler_means=np.asarray([0.0, 0.0, 0.0, 0.1]))
    )
    assert fingerprint != gaussian_hmm_fit_fingerprint(
        replace(
            fit,
            parameters=replace(
                fit.parameters,
                transition_matrix=np.asarray([[0.79, 0.21], [0.2, 0.8]]),
            ),
        )
    )

    with pytest.raises(ModelError, match="geometry"):
        gaussian_hmm_fit_fingerprint(
            replace(fit, lengths=(len(fit.decoded_states) - 1,))
        )
    with pytest.raises(ModelError, match="scaler scope"):
        gaussian_hmm_fit_fingerprint(
            replace(fit, scaler_scope="wrong")  # type: ignore[arg-type]
        )


def test_registered_rng_identity_binds_seed_version_namespace_and_contract() -> None:
    arguments = {
        "master_seed": 123,
        "scientific_version": 1,
        "namespace": "test_execution",
        "contract": {"attempts": 10, "stage": 2},
    }
    first = rng_execution_fingerprint(**arguments)  # type: ignore[arg-type]
    second = rng_execution_fingerprint(**arguments)  # type: ignore[arg-type]
    changed_seed = rng_execution_fingerprint(
        **{**arguments, "master_seed": 124},  # type: ignore[arg-type]
    )
    changed_contract = rng_execution_fingerprint(
        **{**arguments, "contract": {"attempts": 11, "stage": 2}},  # type: ignore[arg-type]
    )

    assert first == second
    assert first != changed_seed
    assert first != changed_contract


@pytest.mark.parametrize(
    ("value", "role"),
    [
        (np.asarray([np.nan]), "valid_role"),
        (np.asarray([1.0]), "Invalid-Role"),
        (np.asarray(1.0), "valid_role"),
        (np.asarray(["text"]), "valid_role"),
    ],
)
def test_invalid_array_materializations_fail_closed(
    value: np.ndarray, role: str
) -> None:
    with pytest.raises(ModelError):
        scientific_array_fingerprint(value, role=role)


def test_invalid_feature_geometry_and_rng_metadata_fail_closed() -> None:
    with pytest.raises(ModelError, match="row geometry"):
        hmm_feature_matrix_fingerprint(replace(_features(), lengths=(1,)))
    with pytest.raises(ModelError, match="master_seed"):
        rng_execution_fingerprint(
            master_seed=-1,
            scientific_version=1,
            namespace="test_execution",
            contract={"attempts": 1},
        )
    with pytest.raises(ModelError, match="non-finite"):
        rng_execution_fingerprint(
            master_seed=1,
            scientific_version=1,
            namespace="test_execution",
            contract={"threshold": float("inf")},
        )
    with pytest.raises(ModelError, match="scientific_version"):
        rng_execution_fingerprint(
            master_seed=1,
            scientific_version=0,
            namespace="test_execution",
            contract={"attempts": 1},
        )


def test_generic_materialization_and_type_contracts_fail_closed() -> None:
    with pytest.raises(ModelError, match="metadata"):
        scientific_materialization_fingerprint(
            schema_id="valid_schema",
            metadata=None,  # type: ignore[arg-type]
            arrays={"values": np.asarray([1.0])},
        )
    with pytest.raises(ModelError, match="nonempty"):
        scientific_materialization_fingerprint(
            schema_id="valid_schema",
            metadata={},
            arrays={},
        )
    with pytest.raises(ModelError, match="HMM parameter"):
        hmm_parameter_fingerprint(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="feature fingerprint"):
        hmm_feature_matrix_fingerprint(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="HMM model"):
        gaussian_hmm_fit_fingerprint(object())  # type: ignore[arg-type]

    features = _features()
    with pytest.raises(ModelError, match="column geometry"):
        hmm_feature_matrix_fingerprint(replace(features, feature_names=("only_one",)))
    with pytest.raises(ModelError, match="scaler scope"):
        hmm_feature_matrix_fingerprint(
            replace(features, scaler_scope="wrong")  # type: ignore[arg-type]
        )

    with pytest.raises(ModelError, match="two-dimensional"):
        hmm_feature_matrix_fingerprint(replace(features, values=np.asarray([1.0, 2.0])))

    fit = _fit()
    for malformed in (
        replace(fit, feature_names=()),
        replace(fit, feature_names=("first", "", "third", "fourth")),
        replace(fit, lengths=()),
        replace(fit, lengths=(-1,)),
        replace(fit, best_restart_index=True),  # type: ignore[arg-type]
        replace(fit, n_features=0),
    ):
        with pytest.raises(ModelError):
            gaussian_hmm_fit_fingerprint(malformed)


def test_low_level_array_and_json_failures_are_normalized() -> None:
    assert (
        len(
            scientific_array_fingerprint(
                np.asarray([True, False]), role="boolean_values"
            )
        )
        == 64
    )

    structured = np.asarray([(1.0,)], dtype=[("field", "<f8")])
    with pytest.raises(ModelError, match="structured"):
        scientific_array_fingerprint(structured, role="structured_values")

    class BadArray:
        def __array__(self) -> np.ndarray:
            raise ValueError("cannot convert")

    with pytest.raises(ModelError, match="cannot be converted"):
        scientific_array_fingerprint(BadArray(), role="bad_values")  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="not canonical"):
        scientific_materialization_fingerprint(
            schema_id="valid_schema",
            metadata={"bad": {1, 2}},
            arrays={"values": np.asarray([1.0])},
        )
