from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from prpg.simulation.rng import (
    CalibrationArtifact,
    CalibrationPurpose,
    Domain,
    Family,
    ModelFitScope,
    RNGKey,
    Stage,
    calibration_entity,
    calibration_rng_key,
    categorical_from_uniform,
    hmm_emission_standard_normals,
    hmm_library_seed,
    model_fit_entity,
    sobol_direction_seed,
    uniforms,
)


def _production_key(entity: int, stage: Stage = Stage.STATE_CHAIN) -> RNGKey:
    return RNGKey(
        master_seed=20_260_714,
        scientific_version=1,
        domain=Domain.PRODUCTION,
        family=Family.LF,
        entity=entity,
        stage=stage,
    )


def test_spawn_key_exact_contract_order() -> None:
    key = _production_key(42, Stage.SOURCE_RANK_UNIFORMS)

    assert key.spawn_key == (1, 1, 6, 1, 42, 4)
    assert key.seed_sequence().spawn_key == (1, 1, 6, 1, 42, 4)
    assert key.seed_sequence().entropy == 20_260_714


def test_generator_is_fresh_and_reproducible() -> None:
    key = _production_key(1)

    first = key.generator().random(8)
    second = key.generator().random(8)

    np.testing.assert_array_equal(first, second)


def test_frozen_reference_vector_guards_numpy_contract() -> None:
    actual = _production_key(1).generator().random(5)
    expected = np.array(
        [
            0.6396249948208429,
            0.38678227036821944,
            0.14734896911229223,
            0.83354355799587376,
            0.29601375576793598,
        ]
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "replacement",
    [
        {"scientific_version": 2},
        {"domain": Domain.QUALIFICATION},
        {"family": Family.HF},
        {"entity": 2},
        {"stage": Stage.RESTART_UNIFORMS},
    ],
)
def test_each_key_dimension_separates_streams(replacement: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "master_seed": 20_260_714,
        "scientific_version": 1,
        "domain": Domain.PRODUCTION,
        "family": Family.LF,
        "entity": 1,
        "stage": Stage.STATE_CHAIN,
    }
    arguments.update(replacement)

    changed = RNGKey(**arguments)  # type: ignore[arg-type]

    assert not np.array_equal(
        changed.generator().random(4),
        _production_key(1).generator().random(4),
    )


def test_entity_keys_are_worker_order_invariant() -> None:
    entities = list(range(1, 33))

    serial = {entity: uniforms(_production_key(entity), 12) for entity in entities}
    with ThreadPoolExecutor(max_workers=7) as executor:
        reverse_values = executor.map(
            lambda entity: (entity, uniforms(_production_key(entity), 12)),
            reversed(entities),
        )
    parallel = dict(reverse_values)

    for entity in entities:
        np.testing.assert_array_equal(serial[entity], parallel[entity])


def test_model_fit_entity_and_library_seed_are_stable() -> None:
    assert model_fit_entity(ModelFitScope.DESIGN_MAIN, 0, 2, 0) == 512
    assert model_fit_entity(ModelFitScope.DESIGN_ROLLING_FOLD, 5, 5, 49) == (
        (2 << 28) | (5 << 12) | (5 << 8) | 49
    )
    first = hmm_library_seed(
        master_seed=20_260_714,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        k=2,
        restart_index=0,
    )
    assert first == hmm_library_seed(
        master_seed=20_260_714,
        scientific_version=1,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        k=2,
        restart_index=0,
    )
    assert first != hmm_library_seed(
        master_seed=20_260_714,
        scientific_version=1,
        scope=ModelFitScope.PRODUCTION_MAIN_AND_FOLDS,
        replicate=0,
        k=2,
        restart_index=0,
    )


@pytest.mark.parametrize(
    ("scope", "replicate", "k", "restart"),
    [
        (ModelFitScope.DESIGN_MAIN, 1, 2, 0),
        (ModelFitScope.DESIGN_ROLLING_FOLD, 6, 2, 0),
        (ModelFitScope.PRODUCTION_MAIN_AND_FOLDS, 7, 2, 0),
        (ModelFitScope.BRIDGE_MODEL_SELECTION_PREFIX, 1_750, 2, 0),
        (ModelFitScope.BRIDGE_MODEL_FIXED_K_PREFIX, 10_000, 2, 0),
        (ModelFitScope.HMM_IDENTIFICATION_META, 0, 2, 0),
        (ModelFitScope.DESIGN_MAIN, 0, 1, 0),
        (ModelFitScope.DESIGN_MAIN, 0, 2, 50),
    ],
)
def test_model_fit_entity_rejects_unregistered_combinations(
    scope: ModelFitScope, replicate: int, k: int, restart: int
) -> None:
    with pytest.raises(ValueError):
        model_fit_entity(scope, replicate, k, restart)


def test_calibration_entity_uses_exact_layout_and_closed_variants() -> None:
    assert calibration_entity(
        CalibrationPurpose.G5_OUTER_NULL,
        CalibrationArtifact.PRODUCTION_OR_CANDIDATE,
        0,
        49_999,
    ) == ((10 << 28) | (1 << 26) | 49_999)
    with pytest.raises(ValueError, match="not registered"):
        calibration_entity(
            CalibrationPurpose.G5_OUTER_NULL,
            CalibrationArtifact.DESIGN_OR_CONTROL,
            1,
            0,
        )
    with pytest.raises(ValueError, match="not registered"):
        calibration_entity(
            CalibrationPurpose.RESERVED,
            CalibrationArtifact.DESIGN_OR_CONTROL,
            0,
            0,
        )


@pytest.mark.parametrize(
    (
        "purpose",
        "artifact",
        "variant",
        "replicate",
        "domain",
        "family",
        "stage",
    ),
    [
        (0, 0, 0, 99, 2, 0, 6),
        (1, 3, 0, 9_999, 2, 0, 6),
        (2, 1, 0, 9_999, 2, 0, 2),
        (2, 1, 0, 9_999, 2, 0, 6),
        (3, 0, 0, 999, 2, 0, 5),
        (4, 1, 0, 9_999, 2, 2, 4),
        (7, 0, 0, 319_999, 2, 1, 6),
        (9, 2, 0, 249, 2, 0, 2),
        (9, 2, 1, 9_999, 2, 0, 5),
        (9, 3, 1, 9_999, 2, 0, 6),
        (10, 1, 0, 49_999, 3, 2, 6),
        (12, 0, 0, 999, 3, 0, 5),
        (12, 0, 0, 999, 3, 1, 3),
        (13, 1, 9, 999, 3, 2, 7),
        (14, 0, 0, 0, 3, 0, 7),
        (14, 1, 1, 199, 5, 2, 7),
    ],
)
def test_closed_calibration_registry_accepts_every_stream_family(
    purpose: int,
    artifact: int,
    variant: int,
    replicate: int,
    domain: int,
    family: int,
    stage: int,
) -> None:
    key = calibration_rng_key(
        master_seed=20_260_714,
        scientific_version=1,
        purpose=purpose,
        artifact=artifact,
        variant=variant,
        replicate=replicate,
        domain=domain,
        family=family,
        stage=stage,
    )
    assert isinstance(key, RNGKey)


@pytest.mark.parametrize(
    (
        "purpose",
        "artifact",
        "variant",
        "replicate",
        "domain",
        "family",
        "stage",
    ),
    [
        (5, 0, 0, 0, 2, 1, 3),
        (6, 0, 0, 0, 2, 1, 2),
        (8, 0, 0, 0, 2, 1, 6),
        (11, 0, 0, 0, 3, 1, 7),
        (13, 1, 8, 0, 3, 1, 7),
        (14, 0, 2, 0, 3, 1, 7),
        (0, 0, 0, 100, 2, 0, 6),
        (4, 0, 0, 0, 2, 0, 2),
        (9, 2, 0, 250, 2, 0, 2),
        (9, 3, 0, 0, 2, 0, 2),
        (10, 0, 0, 50_000, 3, 1, 6),
        (12, 1, 0, 0, 3, 1, 2),
        (14, 0, 1, 500, 5, 1, 7),
    ],
)
def test_closed_calibration_registry_rejects_unlisted_or_streamless_combinations(
    purpose: int,
    artifact: int,
    variant: int,
    replicate: int,
    domain: int,
    family: int,
    stage: int,
) -> None:
    with pytest.raises(ValueError):
        calibration_rng_key(
            master_seed=20_260_714,
            scientific_version=1,
            purpose=purpose,
            artifact=artifact,
            variant=variant,
            replicate=replicate,
            domain=domain,
            family=family,
            stage=stage,
        )


def test_sobol_seed_and_four_normal_emission_consumption_are_exact() -> None:
    seed = sobol_direction_seed(master_seed=20_260_714, scientific_version=1)
    assert 0 <= seed <= np.iinfo(np.uint32).max
    assert seed == sobol_direction_seed(master_seed=20_260_714, scientific_version=1)
    key = calibration_rng_key(
        master_seed=20_260_714,
        scientific_version=1,
        purpose=CalibrationPurpose.PARAMETRIC_ADEQUACY,
        artifact=CalibrationArtifact.DESIGN_OR_CONTROL,
        variant=0,
        replicate=0,
        domain=Domain.BLOCK_ALPHA_CALIBRATION,
        family=Family.NOT_APPLICABLE,
        stage=Stage.OPTIONAL_RESIDUAL_NOISE,
    )
    values = hmm_emission_standard_normals(key, 3)
    assert values.shape == (3, 4)
    np.testing.assert_array_equal(values.ravel(), key.generator().standard_normal(12))
    with pytest.raises(ValueError, match="family 0 and stage 5"):
        hmm_emission_standard_normals(_production_key(1), 1)


@pytest.mark.parametrize(
    ("uniform", "expected"),
    [(0.0, 0), (0.199999, 0), (0.2, 1), (0.69999, 1), (0.7, 2), (0.999999, 2)],
)
def test_categorical_uses_cumulative_search_and_last_fallback(
    uniform: float, expected: int
) -> None:
    assert categorical_from_uniform([0.2, 0.5, 0.3], uniform) == expected


@pytest.mark.parametrize(
    ("probabilities", "uniform"),
    [([], 0.2), ([0.5, -0.1, 0.6], 0.2), ([0.2, 0.2], 0.2), ([1.0], 1.0)],
)
def test_categorical_rejects_invalid_inputs(
    probabilities: list[float], uniform: float
) -> None:
    with pytest.raises(ValueError):
        categorical_from_uniform(probabilities, uniform)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"master_seed": -1},
        {"master_seed": 1 << 128},
        {"scientific_version": 1 << 32},
        {"domain": 8},
        {"family": 3},
        {"entity": -1},
        {"stage": 0},
        {"contract": 2},
        {"domain": True},
    ],
)
def test_rng_key_rejects_out_of_contract_values(kwargs: dict[str, int]) -> None:
    values = {
        "master_seed": 1,
        "scientific_version": 1,
        "domain": 6,
        "family": 1,
        "entity": 1,
        "stage": 2,
        "contract": 1,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        RNGKey(**values)


def test_uniform_count_is_exact_and_validated() -> None:
    assert uniforms(_production_key(1), 0).shape == (0,)
    assert uniforms(_production_key(1), 7).shape == (7,)
    with pytest.raises(ValueError):
        uniforms(_production_key(1), -1)
    with pytest.raises(TypeError):
        uniforms(_production_key(1), True)
