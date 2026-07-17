"""Tests for the closed four-cell dependence-gate execution."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import prpg.model.g3_assembly as assembly_module
from prpg.errors import ModelError
from prpg.model import dependence_gate_execution as subject
from prpg.model.block_execution import (
    Artifact,
    BlockStateModel,
    CalibrationGeometry,
    RegisteredUniformMatrices,
    resolve_calibration_geometry,
)
from prpg.model.block_execution_registered import (
    RegisteredBlockCalibration,
    RegisteredBlockUniformMaterialization,
    materialize_registered_block_uniforms,
    registered_block_core_identity,
    registered_block_uniform_identity,
    run_registered_materialized_block_calibration,
)
from prpg.model.block_length import Frequency
from prpg.model.dependence import feasible_sensitivity_lengths
from prpg.model.dependence_gate_execution import (
    DEPENDENCE_CELL_ORDER,
    ArtifactDependenceExecution,
    DependenceCellEvidence,
    DependenceCellId,
    DependenceGateCellInput,
    DependenceTaskCellEvidence,
    DependenceTaskView,
    FourCellDependenceGateExecution,
    FourCellDependenceHolmView,
    FragmentationTaskCellEvidence,
    RegisteredArtifactDependenceExecution,
    assemble_artifact_dependence_execution,
    assemble_four_cell_dependence_gate,
    assemble_four_cell_dependence_holm_view,
    build_dependence_task_view,
    build_fragmentation_task_view,
    build_registered_dependence_task_view,
    deterministic_loyo_selector_robustness,
    execute_artifact_dependence_role,
    execute_four_cell_dependence_gate,
    execute_registered_artifact_dependence_role,
)
from prpg.model.g3_assembly import (
    derive_g3_gate_decision,
    task_view_cross_binding_error,
)
from prpg.model.g3_task_publication import (
    dependence_executor_source_slots,
    dependence_holm_executor_source_slots,
    fragmentation_executor_source_slots,
    scientific_task_source_slots,
)
from prpg.model.inference import holm_correction


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _geometry(cell_id: DependenceCellId) -> CalibrationGeometry:
    artifact, frequency = subject._CELL_ROLE[cell_id]
    if frequency == "monthly":
        return resolve_calibration_geometry(
            artifact=artifact,
            frequency=frequency,
            strict_canonical=False,
            monthly_segment_lengths=(80,) if artifact == "design" else (40, 40),
        )
    return resolve_calibration_geometry(
        artifact=artifact,
        frequency=frequency,
        strict_canonical=False,
        monthly_segment_lengths=(10,) if artifact == "design" else (6, 4),
        daily_segment_lengths=(210,) if artifact == "design" else (126, 84),
    )


def _materialization(
    cell_id: DependenceCellId, *, histories: int = 12
) -> RegisteredBlockUniformMaterialization:
    geometry = _geometry(cell_id)
    return materialize_registered_block_uniforms(
        master_seed=50_000 + DEPENDENCE_CELL_ORDER.index(cell_id),
        scientific_version=1,
        artifact=geometry.artifact,
        frequency=geometry.frequency,
        history_count=histories,
        chunk_size=5,
        strict_canonical=False,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if geometry.frequency == "daily" else None
        ),
    )


def _returns(rows: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(size=rows)
    return np.column_stack(
        (
            common + 0.40 * rng.normal(size=rows),
            0.50 * common + rng.normal(size=rows),
            -0.20 * common + rng.normal(size=rows),
        )
    )


def _input(cell_id: DependenceCellId) -> DependenceGateCellInput:
    geometry = _geometry(cell_id)
    rows = geometry.target_observations
    artifact, frequency = subject._CELL_ROLE[cell_id]
    states = (np.arange(rows, dtype=np.int64) // 5) % 2
    if frequency == "monthly":
        years = np.repeat(np.arange(2000, 2008, dtype=np.int64), 10)
    else:
        years = np.repeat(np.arange(2000, 2003, dtype=np.int64), 70)
    model = BlockStateModel(
        stationary_distribution=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.82, 0.18], [0.18, 0.82]]),
    )
    return DependenceGateCellInput(
        cell_id=cell_id,
        artifact=artifact,
        frequency=frequency,
        model=model,
        historical_returns=_readonly(
            _returns(rows, seed=71_000 + DEPENDENCE_CELL_ORDER.index(cell_id))
        ),
        historical_source_states=_readonly(states),
        source_continuation_links=_readonly(geometry.target_continuation_links.copy()),
        calendar_years=_readonly(years),
    )


def _inputs() -> tuple[
    DependenceGateCellInput,
    DependenceGateCellInput,
    DependenceGateCellInput,
    DependenceGateCellInput,
]:
    return cast(
        tuple[
            DependenceGateCellInput,
            DependenceGateCellInput,
            DependenceGateCellInput,
            DependenceGateCellInput,
        ],
        tuple(_input(cell_id) for cell_id in DEPENDENCE_CELL_ORDER),
    )


def _registered_parent(
    cell_id: DependenceCellId,
    *,
    master_seed: int,
    scientific_version: int,
) -> RegisteredBlockCalibration:
    cell = _input(cell_id)
    geometry = _geometry(cell_id)
    materialization = materialize_registered_block_uniforms(
        master_seed=master_seed,
        scientific_version=scientific_version,
        artifact=cell.artifact,
        frequency=cell.frequency,
        history_count=12,
        chunk_size=5,
        strict_canonical=False,
        monthly_segment_lengths=geometry.monthly_segment_lengths,
        daily_segment_lengths=(
            geometry.target_segment_lengths if geometry.frequency == "daily" else None
        ),
    )
    loyo = deterministic_loyo_selector_robustness(
        cell.historical_returns,
        cell.source_continuation_links,
        cell.calendar_years,
        frequency=cell.frequency,
    )
    return run_registered_materialized_block_calibration(
        materialization,
        model=cell.model,
        starting_length=loyo.full_sample_starting_length,
        historical_source_states=cell.historical_source_states,
        source_continuation_links=cell.source_continuation_links,
        chunk_size=5,
    )


def _evidence_tuple(
    values: list[DependenceCellEvidence] | tuple[DependenceCellEvidence, ...],
) -> tuple[
    DependenceCellEvidence,
    DependenceCellEvidence,
    DependenceCellEvidence,
    DependenceCellEvidence,
]:
    return cast(
        tuple[
            DependenceCellEvidence,
            DependenceCellEvidence,
            DependenceCellEvidence,
            DependenceCellEvidence,
        ],
        tuple(values),
    )


@pytest.fixture(scope="module")
def four_cell_result() -> tuple[
    FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
]:
    calls: list[DependenceCellId] = []

    def producer(
        cell: DependenceGateCellInput,
    ) -> RegisteredBlockUniformMaterialization:
        calls.append(cell.cell_id)
        return _materialization(cell.cell_id)

    result = execute_four_cell_dependence_gate(
        _inputs(),
        materialization_producer=producer,
        chunk_size=5,
        strict_canonical=False,
    )
    return result, tuple(calls)


@pytest.fixture(scope="module")
def artifact_executions(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> tuple[ArtifactDependenceExecution, ArtifactDependenceExecution]:
    result, _ = four_cell_result
    design = assemble_artifact_dependence_execution(
        (result.cells[0], result.cells[1]),
        role="design",
        strict_canonical=False,
    )
    production = assemble_artifact_dependence_execution(
        (result.cells[2], result.cells[3]),
        role="production",
        strict_canonical=False,
    )
    return design, production


@pytest.fixture(scope="module")
def block_parents() -> tuple[
    RegisteredBlockCalibration,
    RegisteredBlockCalibration,
    RegisteredBlockCalibration,
    RegisteredBlockCalibration,
]:
    parents = [
        _registered_parent(
            cell.cell_id,
            master_seed=60_000 + (0 if cell.artifact == "design" else 1),
            scientific_version=3,
        )
        for cell in _inputs()
    ]
    return cast(
        tuple[
            RegisteredBlockCalibration,
            RegisteredBlockCalibration,
            RegisteredBlockCalibration,
            RegisteredBlockCalibration,
        ],
        tuple(parents),
    )


@pytest.fixture(scope="module")
def staged_executions(
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
) -> tuple[
    DependenceTaskView,
    RegisteredArtifactDependenceExecution,
    DependenceTaskView,
    RegisteredArtifactDependenceExecution,
]:
    run_id = "1" * 64
    design_parents = (block_parents[0], block_parents[1])
    production_parents = (block_parents[2], block_parents[3])
    design_fragmentation = build_fragmentation_task_view(
        design_parents,
        task_id="design_fragmentation",
        run_id=run_id,
        binding_fingerprint="2" * 64,
    )
    production_fragmentation = build_fragmentation_task_view(
        production_parents,
        task_id="production_fragmentation",
        run_id=run_id,
        binding_fingerprint="4" * 64,
    )
    inputs = _inputs()
    design_execution = execute_registered_artifact_dependence_role(
        (inputs[0], inputs[1]),
        role="design",
        block_parents=design_parents,
        fragmentation_parent=design_fragmentation,
        chunk_size=5,
        strict_canonical=False,
    )
    production_execution = execute_registered_artifact_dependence_role(
        (inputs[2], inputs[3]),
        role="production",
        block_parents=production_parents,
        fragmentation_parent=production_fragmentation,
        chunk_size=5,
        strict_canonical=False,
    )
    return (
        design_fragmentation,
        design_execution,
        production_fragmentation,
        production_execution,
    )


@pytest.fixture(scope="module")
def task_views(
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
) -> tuple[
    DependenceTaskView,
    DependenceTaskView,
    DependenceTaskView,
    DependenceTaskView,
]:
    design_fragmentation, design, production_fragmentation, production = (
        staged_executions
    )
    run_id = "1" * 64
    return (
        design_fragmentation,
        build_registered_dependence_task_view(
            design,
            task_id="design_dependence",
            run_id=run_id,
            binding_fingerprint="3" * 64,
        ),
        production_fragmentation,
        build_registered_dependence_task_view(
            production,
            task_id="production_dependence",
            run_id=run_id,
            binding_fingerprint="5" * 64,
        ),
    )


@pytest.fixture(scope="module")
def holm_view(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> FourCellDependenceHolmView:
    return assemble_four_cell_dependence_holm_view(
        task_views[1],
        task_views[3],
        run_id="1" * 64,
        binding_fingerprint="6" * 64,
    )


def test_v2_publication_slots_match_prebinding_executors_and_exact_lineage(
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    holm_view: FourCellDependenceHolmView,
) -> None:
    design_parents = (block_parents[0], block_parents[1])
    production_parents = (block_parents[2], block_parents[3])
    design_execution = staged_executions[1]
    production_execution = staged_executions[3]

    expected = (
        fragmentation_executor_source_slots(
            design_parents,
            task_id="design_fragmentation",
        ),
        dependence_executor_source_slots(
            design_execution,
            task_id="design_dependence",
        ),
        fragmentation_executor_source_slots(
            production_parents,
            task_id="production_fragmentation",
        ),
        dependence_executor_source_slots(
            production_execution,
            task_id="production_dependence",
        ),
    )
    assert tuple(scientific_task_source_slots(item) for item in task_views) == expected
    assert scientific_task_source_slots(holm_view) == (
        dependence_holm_executor_source_slots(
            task_views[1],
            task_views[3],
            run_id="1" * 64,
        )
    )

    for fragmentation, dependence in (
        (expected[0], expected[1]),
        (expected[2], expected[3]),
    ):
        fragmentation_materializations = dict(
            fragmentation.materialization_fingerprints
        )
        dependence_materializations = dict(dependence.materialization_fingerprints)
        parent_slot = next(
            name
            for name in fragmentation_materializations
            if name.endswith("block_calibration_parent_materialization")
        )
        assert (
            fragmentation_materializations[parent_slot]
            == (dependence_materializations[parent_slot])
        )
        assert (
            dependence_materializations["task_source_materialization"]
            != (fragmentation_materializations["task_source_materialization"])
        )

    different_run_and_binding = build_fragmentation_task_view(
        design_parents,
        task_id="design_fragmentation",
        run_id="7" * 64,
        binding_fingerprint="8" * 64,
    )
    assert scientific_task_source_slots(different_run_and_binding) == expected[0]
    different_dependence_binding = build_registered_dependence_task_view(
        design_execution,
        task_id="design_dependence",
        run_id="1" * 64,
        binding_fingerprint="9" * 64,
    )
    assert scientific_task_source_slots(different_dependence_binding) == expected[1]
    different_holm_binding = assemble_four_cell_dependence_holm_view(
        task_views[1],
        task_views[3],
        run_id="1" * 64,
        binding_fingerprint="a" * 64,
    )
    assert scientific_task_source_slots(different_holm_binding) == (
        scientific_task_source_slots(holm_view)
    )

    with pytest.raises(ModelError, match="code-owned binding-slot derivation"):
        scientific_task_source_slots(
            replace(
                task_views[1],
                fragmentation_parent_view_fingerprint="f" * 64,
            )
        )
    with pytest.raises(ModelError, match="roles/order"):
        fragmentation_executor_source_slots(
            (design_parents[1], design_parents[0]),
            task_id="design_fragmentation",
        )
    with pytest.raises(ModelError, match="wrong role"):
        dependence_executor_source_slots(
            production_execution,
            task_id="design_dependence",
        )
    with pytest.raises(ModelError, match="registered staged evidence"):
        dependence_executor_source_slots(
            copy.copy(design_execution),
            task_id="design_dependence",
        )
    with pytest.raises(ModelError, match="another run"):
        dependence_holm_executor_source_slots(
            task_views[1],
            task_views[3],
            run_id="b" * 64,
        )


def test_g3_assembly_uses_disjoint_role_views_then_exact_dependence_holm(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    holm_view: FourCellDependenceHolmView,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical: list[DependenceTaskView] = []
    for view in task_views:
        cells: list[object] = []
        for cell in view.cells:
            if isinstance(cell, FragmentationTaskCellEvidence):
                cells.append(
                    replace(
                        cell,
                        histories=10_000,
                        common_draw_reused=True,
                        fragmentation=tuple(
                            replace(part, reasons=()) for part in cell.fragmentation
                        ),
                    )
                )
            else:
                cells.append(
                    replace(
                        cell,
                        histories=10_000,
                        common_draw_reused=True,
                        selected_result=replace(
                            cell.selected_result,
                            observed_statistic=0.0,
                            critical_value_95=1.0,
                            p_value=0.5,
                            repetitions=10_000,
                        ),
                    )
                )
        canonical.append(
            replace(
                view,
                cells=tuple(cells),
                passed=True,
                failure_reasons=(),
                strict_canonical=True,
            )
        )
    selected = (canonical[1], canonical[3])
    corrected = holm_correction(np.full(4, 0.5), alpha=0.05)
    canonical_holm = replace(
        holm_view,
        source_views=selected,
        source_view_fingerprints=tuple(view.view_fingerprint for view in selected),
        holm=replace(holm_view.holm, holm=corrected, passed=True),
        passed=True,
        failure_reasons=(),
        strict_canonical=True,
    )
    monkeypatch.setattr(
        assembly_module,
        "is_registered_dependence_task_view",
        lambda value: any(value is item for item in canonical),
    )
    monkeypatch.setattr(
        assembly_module,
        "is_registered_four_cell_dependence_holm_view",
        lambda value: value is canonical_holm,
    )

    sources = {view.task_id: view for view in canonical} | {
        "four_cell_dependence_holm": canonical_holm
    }
    decisions = {
        task_id: derive_g3_gate_decision(task_id, source)
        for task_id, source in sources.items()
    }
    assert all(decision.passed for decision in decisions.values())
    assert task_view_cross_binding_error(sources) is None
    assert "p_value" not in decisions["design_fragmentation"].evidence["cells"][0]
    assert (
        "fragmentation_passed"
        not in decisions["design_dependence"].evidence["cells"][0]
    )
    for task_id, decision in decisions.items():
        assert decision.evidence["run_id"] == sources[task_id].run_id
        assert decision.evidence["binding_fingerprint"] == (
            sources[task_id].binding_fingerprint
        )

    with pytest.raises(ModelError, match="sealed dependence task view"):
        derive_g3_gate_decision(
            "design_dependence",
            copy.copy(canonical[1]),
        )
    mixed = replace(
        canonical[1],
        parent_block_core_fingerprints=("9" * 64, "8" * 64),
    )
    assert "block-core lineage differs" in (
        task_view_cross_binding_error({**sources, "design_dependence": mixed}) or ""
    )
    wrong_parent = replace(canonical[1], fragmentation_parent_view_fingerprint="9" * 64)
    assert "does not descend from fragmentation" in (
        task_view_cross_binding_error({**sources, "design_dependence": wrong_parent})
        or ""
    )
    future_fragmentation = replace(
        canonical[0], fragmentation_parent_view_fingerprint="9" * 64
    )
    assert "contains future lineage" in (
        task_view_cross_binding_error(
            {**sources, "design_fragmentation": future_fragmentation}
        )
        or ""
    )


def test_end_to_end_producer_runs_once_per_cell_and_returns_compact_gate(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, calls = four_cell_result
    assert calls == DEPENDENCE_CELL_ORDER
    assert not subject.is_registered_four_cell_dependence_execution(result)
    assert subject.is_report_only_four_cell_dependence_execution(result)
    assert result.report_only
    assert all(
        subject.is_report_only_dependence_cell_evidence(cell)
        and not subject.is_registered_dependence_cell_evidence(cell)
        for cell in result.cells
    )
    assert not subject.is_registered_four_cell_dependence_execution(replace(result))
    assert result.cell_order == DEPENDENCE_CELL_ORDER
    assert result.neighbors_report_only
    assert result.loyo_report_only
    assert not result.strict_canonical
    assert result.passed == (not result.failure_reasons)
    np.testing.assert_array_equal(
        result.holm.holm.raw_p_values,
        [cell.selected_result.p_value for cell in result.cells],
    )

    for cell in result.cells:
        assert cell.histories == 12
        assert cell.common_draw_reused
        assert cell.requested_lengths == feasible_sensitivity_lengths(
            cell.selected_length, frequency=cell.frequency
        )
        assert tuple(item.intended_length for item in cell.length_results) == (
            cell.requested_lengths
        )
        assert (
            sum(item.decision_role == "selected" for item in cell.length_results) == 1
        )
        assert all(
            item.decision_role == "report_only_neighbor"
            for item in cell.length_results
            if item.intended_length != cell.selected_length
        )
        assert cell.endpoint_count == (87 if cell.frequency == "monthly" else 31)
        assert len(cell.endpoint_labels_fingerprint) == 64
        assert cell.loyo.report_only
        assert cell.loyo.full_sample_starting_length == cell.starting_length
        assert cell.loyo.observations == sum(cell.target_segment_lengths)
        assert tuple(record.omitted_year for record in cell.loyo.records) == tuple(
            sorted(record.omitted_year for record in cell.loyo.records)
        )


def test_copied_neighbor_and_loyo_mutation_cannot_reenter_authoritative_gate(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    cells = list(result.cells)
    first = cells[0]
    neighbor_index = next(
        index
        for index, item in enumerate(first.length_results)
        if item.decision_role == "report_only_neighbor"
    )
    lengths = list(first.length_results)
    lengths[neighbor_index] = replace(
        lengths[neighbor_index],
        observed_statistic=lengths[neighbor_index].critical_value_95 + 1_000.0,
        p_value=0.0,
    )

    records = list(first.loyo.records)
    records[0] = replace(
        records[0],
        starting_length=None,
        raw_ceiling_length=None,
        controlling_series=(),
        failure_type="ModelError",
        failure_message="deliberate report-only selector failure",
    )
    changed_loyo = subject._with_loyo_fingerprint(
        replace(first.loyo, records=tuple(records), fingerprint="")
    )
    cells[0] = replace(
        first,
        length_results=tuple(lengths),
        loyo=changed_loyo,
    )
    with pytest.raises(ModelError, match="registered execution authority"):
        assemble_four_cell_dependence_gate(
            _evidence_tuple(cells),
            strict_canonical=False,
        )
    assert result.cells[0].length_results[neighbor_index].decision_role == (
        "report_only_neighbor"
    )
    assert result.cells[0].loyo.records[0].succeeded


def test_loyo_is_deterministic_and_records_scientific_subset_failure() -> None:
    random = _returns(40, seed=7_901)
    constant = np.full((40, 3), (0.02, 0.01, 0.015))
    returns = np.vstack((random, constant))
    years = np.repeat(np.asarray([2000, 2001], dtype=np.int64), 40)
    links = np.ones(79, dtype=np.bool_)

    first = deterministic_loyo_selector_robustness(
        returns, links, years, frequency="monthly"
    )
    second = deterministic_loyo_selector_robustness(
        returns, links, years, frequency="monthly"
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.report_only
    assert not first.records[0].succeeded
    assert first.records[0].failure_type == "PreliminaryPPWNumericalFailure"
    assert first.records[1].succeeded


def test_deleted_year_never_splices_original_rows() -> None:
    retained = np.asarray([0, 1, 4, 5], dtype=np.int64)
    original = np.ones(5, dtype=np.bool_)
    np.testing.assert_array_equal(
        subject._retained_links(retained, original),
        [True, False, True],
    )
    np.testing.assert_array_equal(
        subject._retained_links(np.asarray([2]), original),
        np.zeros(0, dtype=np.bool_),
    )


def test_mapping_path_rejects_cross_cell_aliases() -> None:
    shared = _materialization("design_monthly")
    materializations = dict.fromkeys(DEPENDENCE_CELL_ORDER, shared)
    with pytest.raises(ModelError, match="cannot share"):
        execute_four_cell_dependence_gate(
            _inputs(),
            materializations=materializations,
            strict_canonical=False,
        )


def test_successful_mapping_path_matches_sequential_producer(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    expected, _ = four_cell_result
    actual = execute_four_cell_dependence_gate(
        _inputs(),
        materializations={
            cell_id: _materialization(cell_id) for cell_id in DEPENDENCE_CELL_ORDER
        },
        chunk_size=5,
        strict_canonical=False,
    )

    assert actual.failure_reasons == expected.failure_reasons
    assert actual.passed == expected.passed
    np.testing.assert_array_equal(
        actual.holm.holm.raw_p_values, expected.holm.holm.raw_p_values
    )
    assert tuple(cell.loyo.fingerprint for cell in actual.cells) == tuple(
        cell.loyo.fingerprint for cell in expected.cells
    )


def test_producer_geometry_identity_is_validated_before_execution() -> None:
    with pytest.raises(ModelError, match="wrong artifact/frequency"):
        execute_four_cell_dependence_gate(
            _inputs(),
            materialization_producer=lambda _: _materialization("design_monthly"),
            strict_canonical=False,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        (
            {
                "materializations": {
                    cell_id: _materialization(cell_id)
                    for cell_id in DEPENDENCE_CELL_ORDER
                },
                "materialization_producer": _materialization,
            },
            "exactly one",
        ),
        ({"materialization_producer": _materialization, "chunk_size": 0}, "positive"),
    ],
)
def test_execution_options_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        execute_four_cell_dependence_gate(
            _inputs(),
            strict_canonical=False,
            **kwargs,  # type: ignore[arg-type]
        )


def test_assembly_requires_fixed_order_and_canonical_counts(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    with pytest.raises(ModelError, match="fixed authoritative order"):
        assemble_four_cell_dependence_gate(
            (result.cells[1], result.cells[0], *result.cells[2:]),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="canonical mode"):
        assemble_four_cell_dependence_gate(result.cells)


def test_selected_result_requires_one_exact_match(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    cell = result.cells[0]
    without_selected = replace(
        cell,
        length_results=tuple(
            item
            for item in cell.length_results
            if item.intended_length != cell.selected_length
        ),
    )
    with pytest.raises(ModelError, match="no unique"):
        _ = without_selected.selected_result


def _wrong_role(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, artifact="production")


def _missing_neighbor(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, requested_lengths=(cell.selected_length,))


def _wrong_result_order(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, length_results=tuple(reversed(cell.length_results)))


def _wrong_result_role(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    values = list(cell.length_results)
    wrong_role: subject.DecisionRole = (
        "report_only_neighbor" if values[0].decision_role == "selected" else "selected"
    )
    values[0] = replace(values[0], decision_role=wrong_role)
    return replace(cell, length_results=tuple(values))


def _invalid_result(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    values = list(cell.length_results)
    values[0] = replace(values[0], p_value=math.nan)
    return replace(cell, length_results=tuple(values))


def _wrong_endpoints(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, endpoint_count=cell.endpoint_count + 1)


def _wrong_label_hash(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, endpoint_labels_fingerprint="bad")


def _wrong_namespace(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, stream_namespace="purpose4-wrong")


def _empty_fragmentation(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, fragmentation=())


def _wrong_fragmentation_length(
    cell: DependenceCellEvidence,
) -> DependenceCellEvidence:
    values = list(cell.fragmentation)
    values[0] = replace(values[0], intended_length=cell.selected_length + 1)
    return replace(cell, fragmentation=tuple(values))


def _wrong_loyo_identity(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    loyo = subject._with_loyo_fingerprint(
        replace(
            cell.loyo,
            full_sample_starting_length=cell.starting_length + 1,
            fingerprint="",
        )
    )
    return replace(cell, loyo=loyo)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_wrong_role, "compact role"),
        (_missing_neighbor, "omits"),
        (_wrong_result_order, "result order"),
        (_wrong_result_role, "result role"),
        (_invalid_result, "invalid dependence"),
        (_wrong_endpoints, "endpoint count"),
        (_wrong_label_hash, "endpoint-label"),
        (_wrong_namespace, "stream namespace"),
        (_empty_fragmentation, "state order"),
        (_wrong_fragmentation_length, "fragmentation length"),
        (_wrong_loyo_identity, "LOYO selector identity"),
    ],
)
def test_compact_cell_validation_fails_closed(
    mutation: Callable[[DependenceCellEvidence], DependenceCellEvidence],
    message: str,
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    cells = list(result.cells)
    cells[0] = mutation(cells[0])
    with pytest.raises(ModelError, match=message):
        assemble_four_cell_dependence_gate(
            _evidence_tuple(cells), strict_canonical=False
        )


def _loyo_report_role(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, loyo=replace(cell.loyo, report_only=False))


def _loyo_policy(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, loyo=replace(cell.loyo, policy_version="wrong"))


def _loyo_order(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(
        cell, loyo=replace(cell.loyo, records=tuple(reversed(cell.loyo.records)))
    )


def _loyo_duplicate(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    records = list(cell.loyo.records)
    records[1] = replace(records[1], omitted_year=records[0].omitted_year)
    return replace(cell, loyo=replace(cell.loyo, records=tuple(records)))


def _loyo_record(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    records = list(cell.loyo.records)
    records[0] = replace(records[0], omitted_observations=0)
    return replace(cell, loyo=replace(cell.loyo, records=tuple(records)))


def _loyo_fingerprint(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    return replace(cell, loyo=replace(cell.loyo, fingerprint="0" * 64))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_loyo_report_role, "report-only"),
        (_loyo_policy, "policy version"),
        (_loyo_order, "deterministic order"),
        (_loyo_duplicate, "duplicated"),
        (_loyo_record, "record is inconsistent"),
        (_loyo_fingerprint, "fingerprint"),
    ],
)
def test_loyo_evidence_verification_fails_closed(
    mutation: Callable[[DependenceCellEvidence], DependenceCellEvidence],
    message: str,
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    cells = list(result.cells)
    cells[0] = mutation(cells[0])
    with pytest.raises(ModelError, match=message):
        assemble_four_cell_dependence_gate(
            _evidence_tuple(cells), strict_canonical=False
        )


def test_copied_cell_cannot_manufacture_stream_or_holm_failure(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    cells = list(result.cells)
    first = cells[0]
    lengths = list(first.length_results)
    selected_index = next(
        index for index, item in enumerate(lengths) if item.decision_role == "selected"
    )
    lengths[selected_index] = replace(lengths[selected_index], p_value=0.0)
    cells[0] = replace(first, common_draw_reused=False, length_results=tuple(lengths))

    with pytest.raises(ModelError, match="registered execution authority"):
        assemble_four_cell_dependence_gate(
            _evidence_tuple(cells), strict_canonical=False
        )


def test_assembly_basic_type_guards(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    with pytest.raises(ModelError, match="strict_canonical"):
        assemble_four_cell_dependence_gate(result.cells, strict_canonical=cast(bool, 1))
    with pytest.raises(ModelError, match="exactly four"):
        assemble_four_cell_dependence_gate(
            result.cells[:3],  # type: ignore[arg-type]
            strict_canonical=False,
        )


def test_strict_assembly_reaches_canonical_geometry_check(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    result, _ = four_cell_result
    cells = list(result.cells)
    first = cells[0]
    cells[0] = replace(
        first,
        histories=10_000,
        stream_namespace=subject._stream_namespace(
            first.artifact, first.frequency, 10_000
        ),
        length_results=tuple(
            replace(item, repetitions=10_000) for item in first.length_results
        ),
    )
    with pytest.raises(ModelError, match="canonical mode"):
        assemble_four_cell_dependence_gate(_evidence_tuple(cells))


def test_option_validation_additional_guards() -> None:
    inputs = _inputs()
    valid_mapping = {
        cell_id: _materialization(cell_id) for cell_id in DEPENDENCE_CELL_ORDER
    }
    cases: tuple[tuple[object, dict[str, object], str], ...] = (
        (
            inputs,
            {"materialization_producer": _materialization, "strict_canonical": 1},
            "boolean",
        ),
        (inputs[:3], {"materialization_producer": _materialization}, "exactly four"),
        (
            (object(), *inputs[1:]),
            {"materialization_producer": _materialization},
            "invalid cell",
        ),
        (
            (inputs[1], inputs[0], *inputs[2:]),
            {"materialization_producer": _materialization},
            "fixed order",
        ),
        (
            (replace(inputs[0], artifact="production"), *inputs[1:]),
            {"materialization_producer": _materialization},
            "wrong artifact",
        ),
        (
            inputs,
            {"materializations": {"design_monthly": valid_mapping["design_monthly"]}},
            "exact four",
        ),
        (
            inputs,
            {"materializations": {**valid_mapping, "design_monthly": object()}},
            "invalid value",
        ),
        (inputs, {"materialization_producer": 1}, "callable"),
    )
    for raw_cells, kwargs, message in cases:
        with pytest.raises(ModelError, match=message):
            execute_four_cell_dependence_gate(
                raw_cells,  # type: ignore[arg-type]
                strict_canonical=cast(bool, kwargs.pop("strict_canonical", False)),
                **kwargs,  # type: ignore[arg-type]
            )

    with pytest.raises(ModelError, match="wrong evidence type"):
        execute_four_cell_dependence_gate(
            inputs,
            materialization_producer=cast(
                subject.MaterializationProducer, lambda _: object()
            ),
            strict_canonical=False,
        )


def test_materialization_validation_guards() -> None:
    cell = _input("design_monthly")
    materialization = _materialization("design_monthly")

    with pytest.raises(ModelError, match="registered materialization"):
        subject._validate_materialization(
            object(),  # type: ignore[arg-type]
            cell_input=cell,
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="canonical mode"):
        subject._validate_materialization(
            materialization,
            cell_input=cell,
            strict_canonical=True,
        )
    with pytest.raises(ModelError, match="history count is invalid"):
        subject._validate_materialization(
            replace(materialization, history_count=0),
            cell_input=cell,
            strict_canonical=False,
        )

    canonical_geometry = resolve_calibration_geometry(
        artifact="design", frequency="monthly", strict_canonical=True
    )
    with pytest.raises(ModelError, match="10,000 histories"):
        subject._validate_materialization(
            replace(materialization, geometry=canonical_geometry),
            cell_input=cell,
            strict_canonical=True,
        )

    state = np.asarray(materialization.matrices.monthly_state_uniforms)
    restart = np.asarray(materialization.matrices.restart_uniforms)
    rank = np.asarray(materialization.matrices.source_rank_uniforms)
    bad_shape = RegisteredUniformMatrices(
        _readonly(state[:, :-1].copy()), restart, rank
    )
    with pytest.raises(ModelError, match="wrong exact shape"):
        subject._validate_materialization(
            replace(materialization, matrices=bad_shape),
            cell_input=cell,
            strict_canonical=False,
        )

    writeable = RegisteredUniformMatrices(state.copy(), restart, rank)
    with pytest.raises(ModelError, match="immutable"):
        subject._validate_materialization(
            replace(materialization, matrices=writeable),
            cell_input=cell,
            strict_canonical=False,
        )

    aliased = RegisteredUniformMatrices(state, state, rank)
    with pytest.raises(ModelError, match="must not alias"):
        subject._validate_materialization(
            replace(materialization, matrices=aliased),
            cell_input=cell,
            strict_canonical=False,
        )

    with pytest.raises(ModelError, match="byte accounting"):
        subject._validate_materialization(
            replace(
                materialization,
                allocated_bytes=materialization.allocated_bytes + 8,
            ),
            cell_input=cell,
            strict_canonical=False,
        )


def test_cross_cell_stage_array_alias_is_rejected() -> None:
    materializations = {
        cell_id: _materialization(cell_id) for cell_id in DEPENDENCE_CELL_ORDER
    }
    design_daily = materializations["design_daily"]
    production_daily = materializations["production_daily"]
    aliased = RegisteredUniformMatrices(
        production_daily.matrices.monthly_state_uniforms,
        production_daily.matrices.restart_uniforms,
        design_daily.matrices.source_rank_uniforms,
    )
    materializations["production_daily"] = replace(production_daily, matrices=aliased)
    with pytest.raises(ModelError, match="namespaces cannot alias"):
        subject._validate_no_cross_cell_aliasing(materializations)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cell: replace(cell, historical_returns=[["not-numeric"]]),
            "must be numeric",
        ),
        (
            lambda cell: replace(cell, historical_returns=np.zeros((2, 3))),
            "returns have the wrong",
        ),
        (
            lambda cell: replace(
                cell,
                historical_source_states=np.zeros(
                    _geometry("design_monthly").target_observations
                ),
            ),
            "states have the wrong",
        ),
        (
            lambda cell: replace(cell, source_continuation_links=np.ones(2)),
            "links have the wrong",
        ),
        (
            lambda cell: replace(
                cell,
                source_continuation_links=np.logical_not(
                    np.asarray(cell.source_continuation_links)
                ),
            ),
            "links do not match",
        ),
        (
            lambda cell: replace(
                cell, calendar_years=np.asarray(cell.calendar_years, dtype=np.float64)
            ),
            "calendar years must be an integer",
        ),
        (
            lambda cell: replace(
                cell, calendar_years=np.asarray(cell.calendar_years)[::-1]
            ),
            "ordered values",
        ),
    ],
)
def test_prepared_cell_input_guards(
    mutation: Callable[[DependenceGateCellInput], DependenceGateCellInput],
    message: str,
) -> None:
    cell = mutation(_input("design_monthly"))
    with pytest.raises(ModelError, match=message):
        subject._prepare_cell(cell, _geometry("design_monthly"))


@pytest.mark.parametrize(
    ("returns", "links", "years", "frequency", "message"),
    [
        (
            np.zeros((20, 3)),
            np.ones(19, dtype=np.bool_),
            np.arange(20),
            "weekly",
            "frequency",
        ),
        (
            [["bad"]],
            np.zeros(0, dtype=np.bool_),
            np.asarray([2000]),
            "monthly",
            "numeric",
        ),
        (
            np.zeros((20, 2)),
            np.ones(19, dtype=np.bool_),
            np.full(20, 2000),
            "monthly",
            "three-column",
        ),
        (
            np.zeros((20, 3)),
            np.ones(18, dtype=np.bool_),
            np.full(20, 2000),
            "monthly",
            "forward-link",
        ),
        (
            np.zeros((20, 3)),
            np.ones(19, dtype=np.bool_),
            np.full(20, 2000.0),
            "monthly",
            "integer vector",
        ),
        (
            np.zeros((20, 3)),
            np.ones(19, dtype=np.bool_),
            np.arange(20)[::-1],
            "monthly",
            "ordered values",
        ),
    ],
)
def test_loyo_raw_input_guards(
    returns: object,
    links: object,
    years: object,
    frequency: object,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        deterministic_loyo_selector_robustness(
            cast(npt.ArrayLike, returns),
            cast(npt.ArrayLike, links),
            cast(npt.ArrayLike, years),
            frequency=cast(Frequency, frequency),
        )


def test_private_compact_and_fragmentation_type_guards() -> None:
    with pytest.raises(ModelError, match="invalid compact cell"):
        subject._validate_compact_cell(object(), strict_canonical=False)  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="invalid state order"):
        subject._fragmentation_evidence(())


def test_design_role_executes_before_any_production_input_exists() -> None:
    calls: list[DependenceCellId] = []

    def producer(
        cell: DependenceGateCellInput,
    ) -> RegisteredBlockUniformMaterialization:
        calls.append(cell.cell_id)
        return _materialization(cell.cell_id)

    inputs = _inputs()
    result = execute_artifact_dependence_role(
        (inputs[0], inputs[1]),
        role="design",
        materialization_producer=producer,
        chunk_size=5,
        strict_canonical=False,
    )

    assert calls == ["design_monthly", "design_daily"]
    assert result.role == "design"
    assert result.cell_order == ("design_monthly", "design_daily")
    assert tuple(cell.artifact for cell in result.cells) == ("design", "design")
    assert not subject.is_registered_artifact_dependence_execution(result)
    assert subject.is_report_only_artifact_dependence_execution(result)
    assert result.report_only


@pytest.mark.parametrize(
    ("cells", "role", "message"),
    [
        ((_input("production_monthly"), _input("production_daily")), "design", "order"),
        ((_input("design_daily"), _input("design_monthly")), "design", "order"),
        ((_input("design_monthly"),), "design", "exactly two"),
    ],
)
def test_role_execution_rejects_future_wrong_order_or_incomplete_inputs(
    cells: object,
    role: object,
    message: str,
) -> None:
    with pytest.raises(ModelError, match=message):
        execute_artifact_dependence_role(
            cells,  # type: ignore[arg-type]
            role=cast(Artifact, role),
            materialization_producer=_materialization,
            strict_canonical=False,
        )


def test_role_execution_requires_exact_role_materialization_mapping() -> None:
    inputs = _inputs()
    with pytest.raises(ModelError, match="exact two cells"):
        execute_artifact_dependence_role(
            (inputs[0], inputs[1]),
            role="design",
            materializations={
                "design_monthly": _materialization("design_monthly"),
                "production_monthly": _materialization("production_monthly"),
            },
            strict_canonical=False,
        )


def test_exact_input_and_model_fingerprints_change_source_identity(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
) -> None:
    original, _ = artifact_executions
    monthly = _input("design_monthly")
    daily = _input("design_daily")
    changed_returns = np.asarray(monthly.historical_returns).copy()
    changed_returns[0, 0] += 0.125
    changed_input = execute_artifact_dependence_role(
        (replace(monthly, historical_returns=_readonly(changed_returns)), daily),
        role="design",
        materialization_producer=lambda cell: _materialization(cell.cell_id),
        chunk_size=5,
        strict_canonical=False,
    )
    alternate_model = BlockStateModel(
        stationary_distribution=np.asarray([0.5, 0.5]),
        transition_matrix=np.asarray([[0.74, 0.26], [0.26, 0.74]]),
    )
    changed_model = execute_artifact_dependence_role(
        (
            replace(monthly, model=alternate_model),
            replace(daily, model=alternate_model),
        ),
        role="design",
        materialization_producer=lambda cell: _materialization(cell.cell_id),
        chunk_size=5,
        strict_canonical=False,
    )

    assert (
        changed_input.cells[0].input_fingerprint != original.cells[0].input_fingerprint
    )
    assert (
        changed_input.cells[0].model_fingerprint == original.cells[0].model_fingerprint
    )
    assert changed_input.source_execution_fingerprint != (
        original.source_execution_fingerprint
    )
    assert (
        changed_model.cells[0].model_fingerprint != original.cells[0].model_fingerprint
    )
    assert changed_model.source_execution_fingerprint != (
        original.source_execution_fingerprint
    )
    with pytest.raises(ModelError, match="different models"):
        assemble_artifact_dependence_execution(
            (original.cells[0], changed_model.cells[1]),
            role="design",
            strict_canonical=False,
        )


def test_task_views_are_exact_run_bound_and_task_scoped(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    design_fragmentation, design_dependence, production_fragmentation, _ = task_views

    assert subject.is_registered_dependence_task_view(design_fragmentation)
    assert subject.is_registered_dependence_task_view(design_dependence)
    assert design_fragmentation.run_id == "1" * 64
    assert design_fragmentation.binding_fingerprint == "2" * 64
    assert tuple(cell.cell_id for cell in design_fragmentation.cells) == (
        "design_monthly",
        "design_daily",
    )
    assert all(
        isinstance(cell, subject.FragmentationTaskCellEvidence)
        for cell in design_fragmentation.cells
    )
    assert all(
        not hasattr(cell, "selected_result") for cell in design_fragmentation.cells
    )
    assert all(
        isinstance(cell, DependenceTaskCellEvidence) for cell in design_dependence.cells
    )
    assert all(not hasattr(cell, "fragmentation") for cell in design_dependence.cells)
    assert tuple(cell.cell_id for cell in production_fragmentation.cells) == (
        "production_monthly",
        "production_daily",
    )
    assert design_fragmentation.source_execution_fingerprint != (
        design_dependence.source_execution_fingerprint
    )
    assert design_dependence.fragmentation_parent_view_fingerprint == (
        design_fragmentation.view_fingerprint
    )
    assert design_dependence.parent_block_core_fingerprints == (
        design_fragmentation.parent_block_core_fingerprints
    )


def test_staged_views_expose_separate_exact_parent_rng_and_result_identities(
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    fragmentation, execution, _, _ = staged_executions
    dependence = task_views[1]
    assert subject.is_registered_staged_dependence_execution(execution)
    assert fragmentation._source_staged_execution is None
    assert fragmentation._source_fragmentation_view is None
    assert fragmentation.fragmentation_parent_view_fingerprint is None
    assert all(
        not hasattr(cell, "dependence_result_fingerprint")
        for cell in fragmentation.cells
    )
    for cell, parent in zip(execution.cells, block_parents[:2], strict=True):
        core = registered_block_core_identity(parent)
        materialization = parent._source_materialization
        assert materialization is not None
        uniform = registered_block_uniform_identity(materialization)
        assert subject.is_registered_staged_dependence_cell(cell)
        assert cell.parent_block_core_fingerprint == core.core_fingerprint
        assert cell.purpose4_materialization_fingerprint == (
            uniform.materialization_fingerprint
        )
        assert cell.purpose4_stream_fingerprint == uniform.stream_fingerprint
        assert cell.rng_execution_fingerprint == uniform.rng_execution_fingerprint
        assert cell.dependence_result_fingerprint != cell.source_fingerprint
    assert dependence.fragmentation_parent_view_fingerprint == (
        fragmentation.view_fingerprint
    )
    assert not subject.is_registered_staged_dependence_execution(copy.copy(execution))
    assert not subject.is_registered_staged_dependence_cell(
        copy.copy(execution.cells[0])
    )


def test_legacy_all_at_once_projection_is_report_only_and_rejected_canonically(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
) -> None:
    legacy = build_dependence_task_view(
        artifact_executions[0],
        task_id="design_fragmentation",
        run_id="a" * 64,
        binding_fingerprint="b" * 64,
    )
    assert legacy.report_only
    assert not legacy.canonical_stage
    assert subject.is_report_only_dependence_task_view(legacy)
    assert not subject.is_registered_dependence_task_view(legacy)
    with pytest.raises(ModelError, match="registered authority"):
        assemble_four_cell_dependence_holm_view(
            legacy,
            legacy,
            run_id="a" * 64,
            binding_fingerprint="c" * 64,
        )


def test_staged_topology_rejects_cross_role_copies_rng_registry_and_future_source(
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
) -> None:
    design_parents = (block_parents[0], block_parents[1])
    with pytest.raises(ModelError, match="registered block parents"):
        build_fragmentation_task_view(
            cast(
                tuple[RegisteredBlockCalibration, RegisteredBlockCalibration],
                artifact_executions,
            ),
            task_id="design_fragmentation",
            run_id="1" * 64,
            binding_fingerprint="d" * 64,
        )
    with pytest.raises(ModelError, match="authority"):
        build_fragmentation_task_view(
            (copy.copy(block_parents[0]), block_parents[1]),
            task_id="design_fragmentation",
            run_id="1" * 64,
            binding_fingerprint="d" * 64,
        )
    with pytest.raises(ModelError, match="wrong role/frequency"):
        build_fragmentation_task_view(
            (block_parents[2], block_parents[3]),
            task_id="design_fragmentation",
            run_id="1" * 64,
            binding_fingerprint="d" * 64,
        )
    wrong_seed_daily = _registered_parent(
        "design_daily", master_seed=60_099, scientific_version=3
    )
    with pytest.raises(ModelError, match="different RNG registry"):
        build_fragmentation_task_view(
            (block_parents[0], wrong_seed_daily),
            task_id="design_fragmentation",
            run_id="1" * 64,
            binding_fingerprint="d" * 64,
        )
    wrong_version_daily = _registered_parent(
        "design_daily", master_seed=60_000, scientific_version=99
    )
    with pytest.raises(ModelError, match="different RNG registry"):
        build_fragmentation_task_view(
            (block_parents[0], wrong_version_daily),
            task_id="design_fragmentation",
            run_id="1" * 64,
            binding_fingerprint="d" * 64,
        )
    equivalent_daily = _registered_parent(
        "design_daily", master_seed=60_000, scientific_version=3
    )
    with pytest.raises(ModelError, match="exact fragmentation block parents"):
        execute_registered_artifact_dependence_role(
            (_input("design_monthly"), _input("design_daily")),
            role="design",
            block_parents=(block_parents[0], equivalent_daily),
            fragmentation_parent=task_views[0],
            chunk_size=5,
            strict_canonical=False,
        )
    assert task_views[0]._source_block_parents is not None
    assert all(
        left is right
        for left, right in zip(
            design_parents, task_views[0]._source_block_parents, strict=True
        )
    )


def test_task_view_rejects_wrong_role_task_or_fingerprint(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
) -> None:
    design, _ = artifact_executions
    with pytest.raises(ModelError, match="role does not match"):
        build_dependence_task_view(
            design,
            task_id="production_dependence",
            run_id="1" * 64,
            binding_fingerprint="2" * 64,
        )
    with pytest.raises(ModelError, match="SHA-256"):
        build_dependence_task_view(
            design,
            task_id="design_dependence",
            run_id="bad",
            binding_fingerprint="2" * 64,
        )


def test_all_dependence_authority_is_object_specific(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    holm_view: FourCellDependenceHolmView,
) -> None:
    gate, _ = four_cell_result
    cell = gate.cells[0]
    execution = artifact_executions[0]
    task_view = task_views[1]

    assert not subject.is_registered_dependence_cell_evidence(copy.copy(cell))
    assert not subject.is_registered_dependence_cell_evidence(replace(cell))
    assert not subject.is_registered_artifact_dependence_execution(copy.copy(execution))
    assert not subject.is_registered_artifact_dependence_execution(replace(execution))
    assert not subject.is_registered_dependence_task_view(copy.copy(task_view))
    assert not subject.is_registered_dependence_task_view(replace(task_view))
    assert not subject.is_registered_four_cell_dependence_holm_view(
        copy.copy(holm_view)
    )
    assert not subject.is_registered_four_cell_dependence_holm_view(replace(holm_view))
    assert not subject.is_registered_four_cell_dependence_execution(copy.copy(gate))
    assert not subject.is_registered_four_cell_dependence_execution(replace(gate))


def test_holm_view_recomputes_exact_selected_l_p_values(
    holm_view: FourCellDependenceHolmView,
) -> None:
    assert subject.is_registered_four_cell_dependence_holm_view(holm_view)
    p_values = np.asarray([cell.selected_result.p_value for cell in holm_view.cells])
    expected = subject.holm_correction(p_values, alpha=0.05)
    np.testing.assert_array_equal(holm_view.holm.holm.raw_p_values, p_values)
    np.testing.assert_array_equal(
        holm_view.holm.holm.adjusted_p_values, expected.adjusted_p_values
    )
    np.testing.assert_array_equal(holm_view.holm.holm.rejected, expected.rejected)
    assert holm_view.passed == (not holm_view.failure_reasons)


def test_holm_view_rejects_cross_run_role_binding_and_mutated_sources(
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    design_fragmentation, design_dependence, _, production_dependence = task_views
    production_parents = (block_parents[2], block_parents[3])
    other_fragmentation = build_fragmentation_task_view(
        production_parents,
        task_id="production_fragmentation",
        run_id="7" * 64,
        binding_fingerprint="7" * 64,
    )
    inputs = _inputs()
    other_execution = execute_registered_artifact_dependence_role(
        (inputs[2], inputs[3]),
        role="production",
        block_parents=production_parents,
        fragmentation_parent=other_fragmentation,
        chunk_size=5,
        strict_canonical=False,
    )
    other_run = build_registered_dependence_task_view(
        other_execution,
        task_id="production_dependence",
        run_id="7" * 64,
        binding_fingerprint="8" * 64,
    )
    with pytest.raises(ModelError, match="different runs"):
        assemble_four_cell_dependence_holm_view(
            design_dependence,
            other_run,
            run_id="1" * 64,
            binding_fingerprint="6" * 64,
        )
    with pytest.raises(ModelError, match="design source"):
        assemble_four_cell_dependence_holm_view(
            design_fragmentation,
            production_dependence,
            run_id="1" * 64,
            binding_fingerprint="6" * 64,
        )
    duplicate_binding = build_registered_dependence_task_view(
        staged_executions[3],
        task_id="production_dependence",
        run_id="1" * 64,
        binding_fingerprint=design_dependence.binding_fingerprint,
    )
    with pytest.raises(ModelError, match="distinct source task bindings"):
        assemble_four_cell_dependence_holm_view(
            design_dependence,
            duplicate_binding,
            run_id="1" * 64,
            binding_fingerprint="6" * 64,
        )
    with pytest.raises(ModelError, match="differ from source"):
        assemble_four_cell_dependence_holm_view(
            design_dependence,
            production_dependence,
            run_id="1" * 64,
            binding_fingerprint=design_dependence.binding_fingerprint,
        )
    with pytest.raises(ModelError, match="registered authority"):
        assemble_four_cell_dependence_holm_view(
            replace(design_dependence, passed=not design_dependence.passed),
            production_dependence,
            run_id="1" * 64,
            binding_fingerprint="6" * 64,
        )


def test_noncanonical_holm_mode_recomputes_requested_alpha(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    result = assemble_four_cell_dependence_holm_view(
        task_views[1],
        task_views[3],
        run_id="1" * 64,
        binding_fingerprint="9" * 64,
        alpha=0.10,
    )
    assert not result.strict_canonical
    assert result.holm.holm.alpha == 0.10
    assert subject.is_registered_four_cell_dependence_holm_view(result)


def test_public_topology_type_guards_reject_untyped_values() -> None:
    assert not subject.is_registered_four_cell_dependence_execution(object())
    assert not subject.is_registered_dependence_cell_evidence(object())
    assert not subject.is_registered_artifact_dependence_execution(object())
    assert not subject.is_registered_dependence_task_view(object())
    assert not subject.is_registered_four_cell_dependence_holm_view(object())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"role": "wrong"}, "design or production"),
        ({"strict_canonical": 1}, "boolean"),
        ({"chunk_size": 0}, "positive"),
        ({"cells": (object(), object())}, "invalid input"),
        (
            {
                "cells": (
                    replace(_input("design_monthly"), artifact="production"),
                    _input("design_daily"),
                )
            },
            "wrong artifact",
        ),
        ({"materialization_producer": None}, "exactly one"),
        ({"materialization_producer": 1}, "callable"),
    ],
)
def test_role_execution_option_guards_cover_closed_boundary(
    mutation: dict[str, object], message: str
) -> None:
    kwargs: dict[str, object] = {
        "cells": (_input("design_monthly"), _input("design_daily")),
        "role": "design",
        "materialization_producer": _materialization,
        "strict_canonical": False,
    }
    kwargs.update(mutation)
    with pytest.raises(ModelError, match=message):
        execute_artifact_dependence_role(**kwargs)  # type: ignore[arg-type]


def test_artifact_assembly_guards_reject_untyped_mode_count_and_order(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
) -> None:
    design, _ = artifact_executions
    with pytest.raises(ModelError, match="design or production"):
        assemble_artifact_dependence_execution(
            design.cells,
            role=cast(Artifact, "wrong"),
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="boolean"):
        assemble_artifact_dependence_execution(
            design.cells,
            role="design",
            strict_canonical=cast(bool, 1),
        )
    with pytest.raises(ModelError, match="exactly two"):
        assemble_artifact_dependence_execution(
            design.cells[:1],  # type: ignore[arg-type]
            role="design",
            strict_canonical=False,
        )
    with pytest.raises(ModelError, match="role-scoped order"):
        assemble_artifact_dependence_execution(
            (design.cells[1], design.cells[0]),
            role="design",
            strict_canonical=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda view: replace(view, cells=view.cells[:1]), "exactly two"),
        (lambda view: replace(view, cells=tuple(reversed(view.cells))), "cell order"),
        (
            lambda view: replace(
                view,
                cells=(
                    replace(view.cells[0], artifact="production"),
                    view.cells[1],
                ),
            ),
            "cell role",
        ),
        (
            lambda view: replace(
                view,
                cells=(replace(view.cells[0], histories=0), view.cells[1]),
            ),
            "geometry",
        ),
        (
            lambda view: replace(
                view,
                cells=(
                    replace(view.cells[0], input_fingerprint="bad"),
                    view.cells[1],
                ),
            ),
            "SHA-256",
        ),
        (
            lambda view: replace(
                view,
                cells=(
                    replace(view.cells[0], model_fingerprint="a" * 64),
                    view.cells[1],
                ),
            ),
            "mixes fitted models",
        ),
        (
            lambda view: replace(
                view,
                cells=(
                    replace(
                        view.cells[0],
                        materialization_fingerprint=(
                            view.cells[1].materialization_fingerprint
                        ),
                    ),
                    view.cells[1],
                ),
            ),
            "reuses one materialization",
        ),
    ],
)
def test_task_scoped_cell_validator_rejects_mutation_and_mixing(
    mutation: Callable[[DependenceTaskView], DependenceTaskView],
    message: str,
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    with pytest.raises(ModelError, match=message):
        subject._verify_task_scoped_cells(mutation(task_views[1]))


def test_task_specific_failure_reasons_and_holm_are_recomputed(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    holm_view: FourCellDependenceHolmView,
) -> None:
    fragmentation = task_views[0].cells
    first_fragmentation = cast(subject.FragmentationTaskCellEvidence, fragmentation[0])
    failed_fragmentation = replace(
        first_fragmentation,
        common_draw_reused=False,
        fragmentation=(
            replace(first_fragmentation.fragmentation[0], reasons=("failed",)),
            *first_fragmentation.fragmentation[1:],
        ),
    )
    fragmentation_reasons = subject._task_view_failure_reasons(
        (
            failed_fragmentation,
            cast(subject.FragmentationTaskCellEvidence, fragmentation[1]),
        ),
        kind="fragmentation",
    )
    assert "design_monthly:common_purpose4_draw_not_reused" in fragmentation_reasons
    assert "design_monthly:selected_L_fragmentation_failed" in fragmentation_reasons

    cells = holm_view.cells
    failed_selected = replace(
        cells[0].selected_result,
        observed_statistic=cells[0].selected_result.critical_value_95 + 1.0,
        p_value=0.0,
    )
    failed_first = replace(
        cells[0], common_draw_reused=False, selected_result=failed_selected
    )
    holm, reasons = subject._recomputed_task_view_holm(
        (failed_first, *cells[1:]),
        alpha=0.05,
        strict_canonical=False,
    )
    assert "design_monthly:common_purpose4_draw_not_reused" in reasons
    assert "design_monthly:selected_L_predictive_envelope_failed" in reasons
    assert "design_monthly:holm_rejected" in reasons
    assert not holm.passed


def test_private_topology_helpers_reject_invalid_shapes_modes_and_types(
    holm_view: FourCellDependenceHolmView,
) -> None:
    cells = holm_view.cells
    with pytest.raises(ModelError, match="boolean"):
        subject._recomputed_task_view_holm(
            cells,
            alpha=0.05,
            strict_canonical=cast(bool, 1),
        )
    with pytest.raises(ModelError, match="four selected-L"):
        subject._recomputed_task_view_holm(
            cells[:3], alpha=0.05, strict_canonical=False
        )
    with pytest.raises(ModelError, match="wrong exact order"):
        subject._recomputed_task_view_holm(
            (cells[1], cells[0], *cells[2:]),
            alpha=0.05,
            strict_canonical=False,
        )
    canonical_cells = tuple(
        replace(
            cell,
            histories=10_000,
            selected_result=replace(cell.selected_result, repetitions=10_000),
        )
        for cell in cells
    )
    with pytest.raises(ModelError, match="alpha"):
        subject._recomputed_task_view_holm(
            canonical_cells,
            alpha=0.10,
            strict_canonical=True,
        )
    with pytest.raises(ModelError, match="invalid fitted model"):
        subject._dependence_model_fingerprint(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="wrong type"):
        subject._dependence_materialization_fingerprint(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="canonical JSON"):
        subject._sha256_json({"bad": math.nan})


def _authorize_staged_cell_copy(
    original: subject.RegisteredDependenceCellExecution,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> subject.RegisteredDependenceCellExecution:
    """Mint a test-only copy so validation reaches scientific tamper checks."""

    changed = replace(original, **changes)
    object.__setattr__(
        changed,
        "_execution_seal",
        subject._REGISTERED_STAGED_DEPENDENCE_CELL_CAPABILITY,
    )
    monkeypatch.setitem(subject._MINTED_STAGED_DEPENDENCE_CELLS, id(changed), changed)
    return changed


def _authorize_staged_execution_copy(
    original: RegisteredArtifactDependenceExecution,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> RegisteredArtifactDependenceExecution:
    """Retain exact private parents while changing one public execution claim."""

    changed = replace(original, **changes)
    object.__setattr__(changed, "_block_parents", original._block_parents)
    object.__setattr__(changed, "_fragmentation_parent", original._fragmentation_parent)
    object.__setattr__(
        changed,
        "_execution_seal",
        subject._REGISTERED_STAGED_DEPENDENCE_EXECUTION_CAPABILITY,
    )
    monkeypatch.setitem(
        subject._MINTED_STAGED_DEPENDENCE_EXECUTIONS, id(changed), changed
    )
    return changed


def _authorize_artifact_execution_copy(
    original: ArtifactDependenceExecution,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> ArtifactDependenceExecution:
    changed = replace(original, **changes)
    object.__setattr__(
        changed,
        "_execution_seal",
        subject._REGISTERED_ARTIFACT_DEPENDENCE_CAPABILITY,
    )
    monkeypatch.setitem(
        subject._MINTED_ARTIFACT_DEPENDENCE_EXECUTIONS, id(changed), changed
    )
    return changed


def _authorize_task_view_copy(
    original: DependenceTaskView,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> DependenceTaskView:
    changed = replace(original, **changes)
    for name in (
        "_source_block_parents",
        "_source_fragmentation_view",
        "_source_staged_execution",
    ):
        object.__setattr__(changed, name, getattr(original, name))
    object.__setattr__(
        changed, "_view_seal", subject._REGISTERED_DEPENDENCE_TASK_VIEW_CAPABILITY
    )
    monkeypatch.setitem(subject._MINTED_DEPENDENCE_TASK_VIEWS, id(changed), changed)
    return changed


def _authorize_report_only_view_copy(
    original: DependenceTaskView,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> DependenceTaskView:
    changed = replace(original, **changes)
    object.__setattr__(
        changed, "_view_seal", subject._REPORT_ONLY_DEPENDENCE_TASK_VIEW_CAPABILITY
    )
    monkeypatch.setitem(
        subject._MINTED_REPORT_ONLY_DEPENDENCE_TASK_VIEWS, id(changed), changed
    )
    return changed


def _authorize_holm_view_copy(
    original: FourCellDependenceHolmView,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> FourCellDependenceHolmView:
    changed = replace(original, **changes)
    object.__setattr__(
        changed, "_view_seal", subject._REGISTERED_DEPENDENCE_HOLM_VIEW_CAPABILITY
    )
    monkeypatch.setitem(subject._MINTED_DEPENDENCE_HOLM_VIEWS, id(changed), changed)
    return changed


def _authorize_four_cell_gate_copy(
    original: FourCellDependenceGateExecution,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> FourCellDependenceGateExecution:
    changed = replace(original, **changes)
    object.__setattr__(
        changed, "_execution_seal", subject._REGISTERED_DEPENDENCE_CAPABILITY
    )
    monkeypatch.setitem(
        subject._MINTED_FOUR_CELL_DEPENDENCE_GATES, id(changed), changed
    )
    return changed


def _replace_first_length_result(
    cell: subject.RegisteredDependenceCellExecution, **changes: object
) -> tuple[subject.DependenceLengthEvidence, ...]:
    return (replace(cell.length_results[0], **changes), *cell.length_results[1:])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"artifact": "production"}, "role"),
        ({"strict_canonical": 1}, "canonical mode"),
        ({"neighbors_report_only": False}, "report-only"),
        ({"histories": True}, "geometry"),
        ({"histories": 0}, "geometry"),
        ({"starting_length": True}, "geometry"),
        ({"starting_length": 0}, "geometry"),
        ({"selected_length": True}, "geometry"),
        ({"selected_length": 0}, "geometry"),
        ({"common_draw_reused": False}, "geometry"),
        ({"strict_canonical": True}, "10,000"),
        ({"requested_lengths": (1,)}, "sensitivity lengths"),
        ({"endpoint_count": 1}, "endpoint count"),
        ({"endpoint_labels_fingerprint": "bad"}, "SHA-256"),
        ({"dependence_result_fingerprint": "a" * 64}, "result fingerprint"),
        ({"source_fingerprint": "a" * 64}, "source fingerprint"),
        ({"evidence_fingerprint": "a" * 64}, "evidence fingerprint"),
    ],
)
def test_staged_cell_validator_rejects_independent_claim_tampering(
    changes: dict[str, object],
    message: str,
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = staged_executions[1].cells[0]
    changed = _authorize_staged_cell_copy(original, monkeypatch, **changes)
    with pytest.raises(ModelError, match=message):
        subject._verify_staged_dependence_cell(changed)


@pytest.mark.parametrize(
    ("result_changes", "message"),
    [
        ({"decision_role": "report_only_neighbor"}, "result is invalid"),
        ({"repetitions": 1}, "result is invalid"),
        ({"observed_statistic": math.nan}, "result is invalid"),
        ({"critical_value_95": math.nan}, "result is invalid"),
        ({"p_value": math.nan}, "result is invalid"),
        ({"p_value": 2.0}, "result is invalid"),
        ({"active_components": -1}, "result is invalid"),
    ],
)
def test_staged_cell_validator_rejects_invalid_selected_and_neighbor_results(
    result_changes: dict[str, object],
    message: str,
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = staged_executions[1].cells[0]
    changed = _authorize_staged_cell_copy(
        original,
        monkeypatch,
        length_results=_replace_first_length_result(original, **result_changes),
    )
    with pytest.raises(ModelError, match=message):
        subject._verify_staged_dependence_cell(changed)


def test_staged_cell_validator_rejects_rebound_loyo_identity(
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = staged_executions[1].cells[0]
    changed_loyo = subject._with_loyo_fingerprint(
        replace(original.loyo, frequency="daily", fingerprint="")
    )
    changed = _authorize_staged_cell_copy(original, monkeypatch, loyo=changed_loyo)
    with pytest.raises(ModelError, match="LOYO identity"):
        subject._verify_staged_dependence_cell(changed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contract_version": "wrong"}, "contract"),
        ({"role": "wrong"}, "invalid role"),
        ({"strict_canonical": 1}, "canonical mode"),
        ({"neighbors_report_only": False}, "report-only"),
        ({"cell_order": ("design_daily", "design_monthly")}, "cell order"),
        ({"parent_block_core_fingerprints": ("a" * 64, "b" * 64)}, "lineage"),
        ({"fragmentation_parent_view_fingerprint": "a" * 64}, "lineage"),
        ({"strict_canonical": True}, "source modes"),
        ({"source_execution_fingerprint": "a" * 64}, "fingerprint"),
    ],
)
def test_staged_execution_validator_rejects_lineage_and_contract_tampering(
    changes: dict[str, object],
    message: str,
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = _authorize_staged_execution_copy(
        staged_executions[1], monkeypatch, **changes
    )
    with pytest.raises(ModelError, match=message):
        subject._verify_staged_dependence_execution(changed)


def test_staged_execution_requires_retained_exact_source_parents(
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = _authorize_staged_execution_copy(staged_executions[1], monkeypatch)
    object.__setattr__(changed, "_block_parents", None)
    with pytest.raises(ModelError, match="source parents"):
        subject._verify_staged_dependence_execution(changed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contract_version": "wrong"}, "contract"),
        ({"neighbors_report_only": False}, "report-only"),
        ({"report_only": False}, "report-only"),
        ({"role": "wrong"}, "invalid role"),
        ({"strict_canonical": 1}, "canonical mode"),
        ({"cell_order": ("design_daily", "design_monthly")}, "cell order"),
        ({"source_execution_fingerprint": "a" * 64}, "source fingerprint"),
    ],
)
def test_legacy_artifact_execution_validator_rejects_tampering(
    changes: dict[str, object],
    message: str,
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = _authorize_artifact_execution_copy(
        artifact_executions[0], monkeypatch, **changes
    )
    with pytest.raises(ModelError, match=message):
        subject._verify_artifact_execution(changed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contract_version": "wrong"}, "contract"),
        ({"canonical_stage": False}, "report-only views"),
        ({"neighbors_report_only": False}, "report-only"),
        ({"task_id": "production_dependence"}, "role, and kind"),
        ({"parent_block_core_fingerprints": ("a" * 64, "b" * 64)}, "lineage"),
        ({"view_fingerprint": "a" * 64}, "view fingerprint"),
    ],
)
def test_registered_task_view_validator_rejects_contract_and_lineage_tampering(
    changes: dict[str, object],
    message: str,
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = _authorize_task_view_copy(task_views[1], monkeypatch, **changes)
    with pytest.raises(ModelError, match=message):
        subject._verify_task_view(changed)


def test_registered_task_views_require_exact_private_parents_and_no_future_source(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependence = _authorize_task_view_copy(task_views[1], monkeypatch)
    object.__setattr__(dependence, "_source_staged_execution", None)
    with pytest.raises(ModelError, match="lacks its staged parents"):
        subject._verify_task_view(dependence)

    fragmentation = _authorize_task_view_copy(task_views[0], monkeypatch)
    object.__setattr__(fragmentation, "_source_fragmentation_view", task_views[0])
    with pytest.raises(ModelError, match="future evidence"):
        subject._verify_task_view(fragmentation)

    no_blocks = _authorize_task_view_copy(task_views[1], monkeypatch)
    object.__setattr__(no_blocks, "_source_block_parents", None)
    with pytest.raises(ModelError, match="no exact block parents"):
        subject._verify_task_view(no_blocks)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contract_version": "wrong"}, "contract"),
        ({"task_id": "production_dependence"}, "identity"),
        ({"view_fingerprint": "a" * 64}, "fingerprint"),
    ],
)
def test_report_only_task_view_validator_rejects_tampering(
    changes: dict[str, object],
    message: str,
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_dependence_task_view(
        artifact_executions[0],
        task_id="design_dependence",
        run_id="a" * 64,
        binding_fingerprint="b" * 64,
    )
    changed = _authorize_report_only_view_copy(original, monkeypatch, **changes)
    with pytest.raises(ModelError, match=message):
        subject._verify_report_only_task_view(changed)


def test_report_only_task_view_recomputes_its_decision(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_dependence_task_view(
        artifact_executions[0],
        task_id="design_dependence",
        run_id="a" * 64,
        binding_fingerprint="b" * 64,
    )
    changed = _authorize_report_only_view_copy(
        original, monkeypatch, passed=not original.passed
    )
    with pytest.raises(ModelError, match="decision"):
        subject._verify_report_only_task_view(changed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"contract_version": "wrong"}, "contract/order"),
        ({"neighbors_report_only": False}, "report-only"),
        ({"source_views": ()}, "exactly two"),
        ({"source_view_fingerprints": ("a" * 64, "b" * 64)}, "source fingerprints"),
        ({"view_fingerprint": "a" * 64}, "view fingerprint"),
    ],
)
def test_holm_view_validator_rejects_contract_result_and_lineage_tampering(
    changes: dict[str, object],
    message: str,
    holm_view: FourCellDependenceHolmView,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = _authorize_holm_view_copy(holm_view, monkeypatch, **changes)
    with pytest.raises(ModelError, match=message):
        subject._verify_holm_view(changed)


def test_holm_view_recomputes_its_decision(
    holm_view: FourCellDependenceHolmView,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = _authorize_holm_view_copy(
        holm_view, monkeypatch, passed=not holm_view.passed
    )
    with pytest.raises(ModelError, match="decision"):
        subject._verify_holm_view(changed)


def test_holm_view_validator_rejects_wrong_source_order_run_binding_and_mode(
    holm_view: FourCellDependenceHolmView,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (
            {"source_views": tuple(reversed(holm_view.source_views))},
            "source task roles",
        ),
        ({"run_id": "a" * 64}, "source runs"),
        (
            {"binding_fingerprint": holm_view.source_views[0].binding_fingerprint},
            "task bindings",
        ),
        ({"strict_canonical": True}, "canonical modes"),
    )
    for changes, message in cases:
        changed = _authorize_holm_view_copy(holm_view, monkeypatch, **changes)
        with pytest.raises(ModelError, match=message):
            subject._verify_holm_view(changed)


def test_holm_and_legacy_gate_validators_recompute_exact_results(
    holm_view: FourCellDependenceHolmView,
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_raw_p_values = holm_view.holm.holm.raw_p_values.copy()
    changed_raw_p_values[0] = 0.0 if changed_raw_p_values[0] != 0.0 else 1.0
    changed_holm = replace(
        holm_view.holm,
        holm=replace(
            holm_view.holm.holm,
            raw_p_values=changed_raw_p_values,
        ),
    )
    changed_view = _authorize_holm_view_copy(holm_view, monkeypatch, holm=changed_holm)
    with pytest.raises(ModelError, match="not recomputed exactly"):
        subject._verify_holm_view(changed_view)

    gate = four_cell_result[0]
    changed_gate = _authorize_four_cell_gate_copy(
        gate, monkeypatch, contract_version="wrong"
    )
    with pytest.raises(ModelError, match="contract"):
        subject._verify_four_cell_gate(changed_gate)
    changed_gate = _authorize_four_cell_gate_copy(
        gate, monkeypatch, holm=replace(gate.holm, passed=not gate.holm.passed)
    )
    with pytest.raises(ModelError, match="Holm evidence"):
        subject._verify_four_cell_gate(changed_gate)
    changed_gate = _authorize_four_cell_gate_copy(
        gate, monkeypatch, passed=not gate.passed
    )
    with pytest.raises(ModelError, match="decision"):
        subject._verify_four_cell_gate(changed_gate)


def test_type_guards_fail_closed_for_wrong_types_and_tampered_evidence(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
) -> None:
    gate = four_cell_result[0]
    assert not subject.is_report_only_four_cell_dependence_execution(object())
    assert not subject.is_report_only_four_cell_dependence_execution(
        replace(gate, contract_version="wrong")
    )
    assert not subject.is_report_only_dependence_cell_evidence(object())
    assert not subject.is_report_only_dependence_cell_evidence(
        replace(gate.cells[0], report_only=False)
    )
    assert not subject.is_report_only_artifact_dependence_execution(object())
    assert not subject.is_report_only_artifact_dependence_execution(
        replace(artifact_executions[0], report_only=False)
    )
    assert not subject.is_report_only_dependence_task_view(object())
    assert not subject.is_report_only_dependence_task_view(
        replace(task_views[1], report_only=True)
    )
    assert not subject.is_registered_staged_dependence_cell(object())
    assert not subject.is_registered_staged_dependence_execution(object())
    assert not subject.is_registered_staged_dependence_cell(
        replace(staged_executions[1].cells[0], strict_canonical=1)
    )
    assert not subject.is_registered_staged_dependence_execution(
        replace(staged_executions[1], contract_version="wrong")
    )


def test_public_boundaries_reject_unregistered_tasks_and_same_model_materialization(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelError, match="registered task ID"):
        build_fragmentation_task_view(
            (block_parents[0], block_parents[1]),
            task_id=cast(object, "design_dependence"),  # type: ignore[arg-type]
            run_id="a" * 64,
            binding_fingerprint="b" * 64,
        )
    with pytest.raises(ModelError, match="unregistered task ID"):
        build_dependence_task_view(
            artifact_executions[0],
            task_id=cast(object, "unknown"),  # type: ignore[arg-type]
            run_id="a" * 64,
            binding_fingerprint="b" * 64,
        )
    first, second = artifact_executions[0].cells
    same_materialization = replace(
        second,
        materialization_fingerprint=first.materialization_fingerprint,
        source_fingerprint=subject._dependence_cell_source_fingerprint(
            cell_id=second.cell_id,
            artifact=second.artifact,
            frequency=second.frequency,
            input_fingerprint=second.input_fingerprint,
            model_fingerprint=second.model_fingerprint,
            materialization_fingerprint=first.materialization_fingerprint,
            strict_canonical=second.strict_canonical,
        ),
        evidence_fingerprint="",
    )
    same_materialization = replace(
        same_materialization,
        evidence_fingerprint=subject._dependence_cell_evidence_fingerprint(
            same_materialization
        ),
    )
    object.__setattr__(
        same_materialization,
        "_execution_seal",
        subject._REGISTERED_DEPENDENCE_CELL_CAPABILITY,
    )
    monkeypatch.setitem(
        subject._MINTED_DEPENDENCE_CELLS,
        id(same_materialization),
        same_materialization,
    )
    with pytest.raises(ModelError, match="not distinct"):
        assemble_artifact_dependence_execution(
            (first, same_materialization), role="design", strict_canonical=False
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selected_length": True}, "geometry"),
        ({"selected_length": 0}, "geometry"),
        ({"common_draw_reused": 1}, "geometry"),
        ({"selected_k": True}, "geometry"),
        ({"selected_k": 0}, "geometry"),
        ({"starting_length": True}, "geometry"),
        ({"starting_length": 0}, "geometry"),
    ],
)
def test_task_scoped_geometry_rejects_boolean_and_nonpositive_claims(
    changes: dict[str, object],
    message: str,
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    view = task_views[1]
    changed = replace(view, cells=(replace(view.cells[0], **changes), view.cells[1]))
    with pytest.raises(ModelError, match=message):
        subject._verify_task_scoped_cells(changed)


def test_task_scoped_fragmentation_and_selected_l_validation_is_fail_closed(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    fragmentation = task_views[0]
    fragment_cell = cast(FragmentationTaskCellEvidence, fragmentation.cells[0])
    for changed_fragment, message in (
        ((), "state order"),
        (
            (
                replace(
                    fragment_cell.fragmentation[0],
                    intended_length=fragment_cell.selected_length + 1,
                ),
                *fragment_cell.fragmentation[1:],
            ),
            "selected length",
        ),
    ):
        changed = replace(
            fragmentation,
            cells=(
                replace(fragment_cell, fragmentation=changed_fragment),
                fragmentation.cells[1],
            ),
        )
        with pytest.raises(ModelError, match=message):
            subject._verify_task_scoped_cells(changed)

    dependence = task_views[1]
    dependence_cell = cast(DependenceTaskCellEvidence, dependence.cells[0])
    for selected_changes in (
        {"decision_role": "report_only_neighbor"},
        {"intended_length": dependence_cell.selected_length + 1},
        {"repetitions": 1},
        {"observed_statistic": math.nan},
        {"critical_value_95": math.nan},
        {"p_value": math.nan},
        {"p_value": 2.0},
    ):
        changed_cell = replace(
            dependence_cell,
            selected_result=replace(
                dependence_cell.selected_result, **selected_changes
            ),
        )
        with pytest.raises(ModelError, match="selected-L evidence"):
            subject._verify_task_scoped_cells(
                replace(dependence, cells=(changed_cell, dependence.cells[1]))
            )


def test_recomputed_legacy_holm_rejects_type_alpha_and_records_all_failures(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = four_cell_result[0].cells
    with pytest.raises(ModelError, match="invalid compact cell"):
        subject._recomputed_holm(
            (object(), *cells[1:]), alpha=0.05, strict_canonical=False
        )

    canonical = tuple(
        replace(
            cell,
            histories=10_000,
            stream_namespace=subject._stream_namespace(
                cell.artifact, cell.frequency, 10_000
            ),
            length_results=tuple(
                replace(item, repetitions=10_000) for item in cell.length_results
            ),
            strict_canonical=True,
        )
        for cell in cells
    )
    for cell in canonical:
        object.__setattr__(
            cell, "_execution_seal", subject._REGISTERED_DEPENDENCE_CELL_CAPABILITY
        )
        monkeypatch.setitem(subject._MINTED_DEPENDENCE_CELLS, id(cell), cell)
    monkeypatch.setattr(
        subject, "_verify_registered_cell", lambda *args, **kwargs: None
    )
    with pytest.raises(ModelError, match="alpha"):
        subject._recomputed_holm(canonical, alpha=0.10, strict_canonical=True)

    first = replace(
        cells[0],
        common_draw_reused=False,
        fragmentation=(
            replace(cells[0].fragmentation[0], reasons=("failed",)),
            *cells[0].fragmentation[1:],
        ),
        length_results=tuple(
            replace(
                item,
                observed_statistic=(item.critical_value_95 + 1.0),
                p_value=0.0,
            )
            if item.intended_length == cells[0].selected_length
            else item
            for item in cells[0].length_results
        ),
    )
    _, reasons = subject._recomputed_holm(
        (first, *cells[1:]), alpha=0.05, strict_canonical=False
    )
    assert "design_monthly:common_purpose4_draw_not_reused" in reasons
    assert "design_monthly:selected_L_fragmentation_failed" in reasons
    assert "design_monthly:selected_L_predictive_envelope_failed" in reasons
    assert "design_monthly:holm_rejected" in reasons


def _authorize_compact_cell_copy(
    original: DependenceCellEvidence,
    monkeypatch: pytest.MonkeyPatch,
    **changes: object,
) -> DependenceCellEvidence:
    changed = replace(original, **changes)
    object.__setattr__(
        changed, "_execution_seal", subject._REGISTERED_DEPENDENCE_CELL_CAPABILITY
    )
    monkeypatch.setitem(subject._MINTED_DEPENDENCE_CELLS, id(changed), changed)
    return changed


def _rehash_compact_cell(cell: DependenceCellEvidence) -> DependenceCellEvidence:
    source_fingerprint = subject._dependence_cell_source_fingerprint(
        cell_id=cell.cell_id,
        artifact=cell.artifact,
        frequency=cell.frequency,
        input_fingerprint=cell.input_fingerprint,
        model_fingerprint=cell.model_fingerprint,
        materialization_fingerprint=cell.materialization_fingerprint,
        strict_canonical=cell.strict_canonical,
    )
    provisional = replace(
        cell, source_fingerprint=source_fingerprint, evidence_fingerprint=""
    )
    return replace(
        provisional,
        evidence_fingerprint=subject._dependence_cell_evidence_fingerprint(provisional),
    )


def _rehash_staged_cell(
    cell: subject.RegisteredDependenceCellExecution,
) -> subject.RegisteredDependenceCellExecution:
    result_fingerprint = subject._staged_dependence_result_fingerprint(
        selected_length=cell.selected_length,
        requested_lengths=cell.requested_lengths,
        endpoint_count=cell.endpoint_count,
        endpoint_labels_fingerprint=cell.endpoint_labels_fingerprint,
        length_results=cell.length_results,
        loyo=cell.loyo,
    )
    source_fingerprint = subject._staged_dependence_source_fingerprint(
        cell_id=cell.cell_id,
        input_fingerprint=cell.input_fingerprint,
        parent_block_core_fingerprint=cell.parent_block_core_fingerprint,
        materialization_fingerprint=cell.purpose4_materialization_fingerprint,
        stream_fingerprint=cell.purpose4_stream_fingerprint,
        rng_fingerprint=cell.rng_execution_fingerprint,
        strict_canonical=cell.strict_canonical,
    )
    provisional = replace(
        cell,
        dependence_result_fingerprint=result_fingerprint,
        source_fingerprint=source_fingerprint,
        evidence_fingerprint="",
    )
    return replace(
        provisional,
        evidence_fingerprint=subject._staged_dependence_cell_fingerprint(provisional),
    )


def test_compact_cell_strict_mode_report_role_and_canonical_geometry_guards(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
) -> None:
    cell = four_cell_result[0].cells[0]
    with pytest.raises(ModelError, match="canonical mode must be boolean"):
        subject._validate_compact_cell(
            replace(cell, strict_canonical=cast(bool, 1)), strict_canonical=False
        )
    with pytest.raises(ModelError, match="remain report-only"):
        subject._validate_compact_cell(
            replace(cell, report_only=False), strict_canonical=False
        )
    with pytest.raises(ModelError, match="10,000 histories"):
        subject._validate_compact_cell(
            replace(cell, strict_canonical=True), strict_canonical=True
        )

    canonical_count = replace(
        cell,
        histories=10_000,
        length_results=tuple(
            replace(result, repetitions=10_000) for result in cell.length_results
        ),
        stream_namespace=subject._stream_namespace(
            cell.artifact, cell.frequency, 10_000
        ),
        strict_canonical=True,
    )
    with pytest.raises(ModelError, match="noncanonical geometry"):
        subject._validate_compact_cell(canonical_count, strict_canonical=True)


def test_registered_compact_cell_rejects_type_source_and_evidence_tampering(
    four_cell_result: tuple[
        FourCellDependenceGateExecution, tuple[DependenceCellId, ...]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelError, match="registered compact cells"):
        subject._verify_registered_cell(object(), strict_canonical=False)  # type: ignore[arg-type]

    cell = four_cell_result[0].cells[0]
    changed = _authorize_compact_cell_copy(
        cell, monkeypatch, source_fingerprint="a" * 64
    )
    with pytest.raises(ModelError, match="source fingerprint"):
        subject._verify_registered_cell(changed, strict_canonical=False)
    changed = _authorize_compact_cell_copy(
        cell, monkeypatch, evidence_fingerprint="a" * 64
    )
    with pytest.raises(ModelError, match="evidence fingerprint"):
        subject._verify_registered_cell(changed, strict_canonical=False)


def test_private_verifiers_reject_values_of_the_wrong_evidence_type() -> None:
    with pytest.raises(ModelError, match="staged dependence cell has the wrong type"):
        subject._verify_staged_dependence_cell(object())  # type: ignore[arg-type]
    with pytest.raises(
        ModelError, match="staged dependence execution has the wrong type"
    ):
        subject._verify_staged_dependence_execution(object())  # type: ignore[arg-type]
    with pytest.raises(
        ModelError, match="artifact dependence execution has the wrong type"
    ):
        subject._verify_artifact_execution(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="dependence task view has the wrong type"):
        subject._verify_task_view(object())  # type: ignore[arg-type]
    with pytest.raises(
        ModelError, match="report-only dependence task view has the wrong type"
    ):
        subject._verify_report_only_task_view(object())  # type: ignore[arg-type]
    with pytest.raises(ModelError, match="dependence Holm view has the wrong type"):
        subject._verify_holm_view(object())  # type: ignore[arg-type]


def test_staged_input_boundary_rejects_mode_shape_order_role_and_parent_mixing(
    block_parents: tuple[
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
        RegisteredBlockCalibration,
    ],
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    cells = (_input("design_monthly"), _input("design_daily"))
    common = {
        "role": "design",
        "block_parents": (block_parents[0], block_parents[1]),
        "fragmentation_parent": task_views[0],
        "chunk_size": 5,
        "strict_canonical": False,
    }
    cases: tuple[tuple[object, dict[str, object], str], ...] = (
        (cells, {"strict_canonical": 1}, "boolean"),
        (cells, {"chunk_size": 0}, "positive integer"),
        ((object(), object()), {}, "exactly two cell inputs"),
        (tuple(reversed(cells)), {}, "role-scoped order"),
        (
            (replace(cells[0], artifact="production"), cells[1]),
            {},
            "wrong artifact/frequency",
        ),
        (cells, {"fragmentation_parent": object()}, "fragmentation task view"),
        (
            cells,
            {"fragmentation_parent": task_views[2]},
            "wrong fragmentation parent",
        ),
        (cells, {"strict_canonical": True}, "canonical modes differ"),
    )
    for raw_cells, changes, message in cases:
        kwargs = {**common, **changes}
        with pytest.raises(ModelError, match=message):
            subject._validate_staged_role_inputs(
                raw_cells,
                role=cast(Artifact, kwargs["role"]),
                block_parents=kwargs["block_parents"],
                fragmentation_parent=kwargs["fragmentation_parent"],
                chunk_size=cast(int, kwargs["chunk_size"]),
                strict_canonical=cast(bool, kwargs["strict_canonical"]),
            )

    with pytest.raises(ModelError, match="design or production"):
        subject._validated_block_parent_pair(
            (block_parents[0], block_parents[1]), role=cast(Artifact, "wrong")
        )


def test_role_mapping_rejects_invalid_materialization_values() -> None:
    with pytest.raises(ModelError, match="invalid value"):
        execute_artifact_dependence_role(
            (_input("design_monthly"), _input("design_daily")),
            role="design",
            materializations={
                "design_monthly": _materialization("design_monthly"),
                "design_daily": object(),  # type: ignore[dict-item]
            },
            strict_canonical=False,
        )


def test_staged_execution_rejects_wrong_fragmentation_role_and_parent_identity(
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = staged_executions[1]
    wrong_role = _authorize_staged_execution_copy(execution, monkeypatch)
    object.__setattr__(wrong_role, "_fragmentation_parent", staged_executions[2])
    with pytest.raises(ModelError, match="fragmentation role"):
        subject._verify_staged_dependence_execution(wrong_role)

    alternative_parents = (
        _registered_parent("design_monthly", master_seed=60_000, scientific_version=3),
        _registered_parent("design_daily", master_seed=60_000, scientific_version=3),
    )
    alternative_fragmentation = build_fragmentation_task_view(
        alternative_parents,
        task_id="design_fragmentation",
        run_id="1" * 64,
        binding_fingerprint="7" * 64,
    )
    wrong_objects = _authorize_staged_execution_copy(execution, monkeypatch)
    object.__setattr__(
        wrong_objects, "_fragmentation_parent", alternative_fragmentation
    )
    with pytest.raises(ModelError, match="reuse fragmentation parents"):
        subject._verify_staged_dependence_execution(wrong_objects)


def test_staged_execution_rejects_cell_that_no_longer_matches_block_parent(
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = staged_executions[1]
    changed_cell = _rehash_staged_cell(
        replace(execution.cells[0], model_fingerprint="a" * 64)
    )
    changed_cell = _authorize_staged_cell_copy(changed_cell, monkeypatch)
    changed_execution = _authorize_staged_execution_copy(
        execution,
        monkeypatch,
        cells=(changed_cell, execution.cells[1]),
    )
    with pytest.raises(ModelError, match="does not match block parent"):
        subject._verify_staged_dependence_execution(changed_execution)


def test_artifact_execution_rejects_rehashed_model_and_materialization_mixing(
    artifact_executions: tuple[
        ArtifactDependenceExecution, ArtifactDependenceExecution
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = artifact_executions[0]
    first, second = execution.cells
    changed_model = _rehash_compact_cell(replace(second, model_fingerprint="a" * 64))
    changed_model = _authorize_compact_cell_copy(changed_model, monkeypatch)
    mixed_model = _authorize_artifact_execution_copy(
        execution, monkeypatch, cells=(first, changed_model)
    )
    with pytest.raises(ModelError, match="mixes fitted models"):
        subject._verify_artifact_execution(mixed_model)

    changed_materialization = _rehash_compact_cell(
        replace(
            second,
            materialization_fingerprint=first.materialization_fingerprint,
        )
    )
    changed_materialization = _authorize_compact_cell_copy(
        changed_materialization, monkeypatch
    )
    reused = _authorize_artifact_execution_copy(
        execution, monkeypatch, cells=(first, changed_materialization)
    )
    with pytest.raises(ModelError, match="reuses one materialization"):
        subject._verify_artifact_execution(reused)


def test_registered_task_view_rejects_deeper_topology_and_decision_tampering(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
    staged_executions: tuple[
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
        DependenceTaskView,
        RegisteredArtifactDependenceExecution,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependence = task_views[1]
    fragmentation = task_views[0]
    cases = (
        ({"run_id": "a" * 64}, "fragmentation topology"),
        (
            {"binding_fingerprint": fragmentation.binding_fingerprint},
            "bindings must differ",
        ),
        ({"fragmentation_parent_view_fingerprint": "a" * 64}, "fingerprint changed"),
        ({"source_execution_fingerprint": "a" * 64}, "source fingerprint"),
        ({"passed": not dependence.passed}, "decision"),
        (
            {
                "cells": (
                    replace(dependence.cells[0], source_evidence_fingerprint="a" * 64),
                    dependence.cells[1],
                )
            },
            "cells changed",
        ),
    )
    for changes, message in cases:
        changed = _authorize_task_view_copy(dependence, monkeypatch, **changes)
        with pytest.raises(ModelError, match=message):
            subject._verify_task_view(changed)

    wrong_source_objects = _authorize_task_view_copy(dependence, monkeypatch)
    object.__setattr__(
        wrong_source_objects, "_source_staged_execution", staged_executions[3]
    )
    with pytest.raises(ModelError, match="source objects changed"):
        subject._verify_task_view(wrong_source_objects)

    canonical_cells = tuple(
        replace(
            cell,
            histories=10_000,
            selected_result=replace(cell.selected_result, repetitions=10_000),
        )
        for cell in dependence.cells
    )
    wrong_parent_mode = _authorize_task_view_copy(
        dependence,
        monkeypatch,
        cells=canonical_cells,
        strict_canonical=True,
    )
    with pytest.raises(ModelError, match="parent mode"):
        subject._verify_task_view(wrong_parent_mode)


def test_task_scoped_kind_k_and_canonical_count_guards(
    task_views: tuple[
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
        DependenceTaskView,
    ],
) -> None:
    dependence = task_views[1]
    with pytest.raises(ModelError, match="wrong task"):
        subject._verify_task_scoped_cells(
            replace(dependence, task_kind="fragmentation")
        )
    with pytest.raises(ModelError, match="10,000 histories"):
        subject._verify_task_scoped_cells(replace(dependence, strict_canonical=True))
    with pytest.raises(ModelError, match="mixes selected K"):
        subject._verify_task_scoped_cells(
            replace(
                dependence,
                cells=(
                    replace(
                        dependence.cells[0],
                        selected_k=dependence.cells[0].selected_k + 1,
                    ),
                    dependence.cells[1],
                ),
            )
        )
    with pytest.raises(ModelError, match="not a dependence task view"):
        subject._dependence_task_cells(task_views[0])
