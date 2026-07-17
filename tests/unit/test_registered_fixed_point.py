"""Producer, projection, and publication tests for DEF-013."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import pytest
from test_g3_calibration_views import (
    _MASTER_SEED,
    _RUN,
    _SCIENTIFIC_VERSION,
    _SELECTION_BINDING,
    _VIABILITY_BINDING,
    _next_registered_iteration_suite,
    _registered_iteration_suite,
    _registered_selection_bootstrap,
)

import prpg.model.candidate_gate as gate_module
import prpg.model.g3_boundary_views as boundary_module
import prpg.model.g3_calibration_views as view_module
import prpg.model.g3_task_publication as publication_module
import prpg.model.registered_fixed_point as fixed_point_module
from prpg.errors import ModelError
from prpg.model.block_execution_registered import (
    registered_block_calibration_identity,
)
from prpg.model.candidate_gate import (
    CandidateGateResult,
    CandidateViabilityResult,
    execute_registered_candidate_viability,
    has_registered_final_fixed_point_candidate_authority,
    run_owner_fixed_k4_selection,
)
from prpg.model.candidate_suite import RegisteredDesignCandidateSuite
from prpg.model.g3_boundary_views import (
    BlockCalibrationGateEvidence,
    build_block_calibration_task_view,
    is_block_calibration_task_view,
)
from prpg.model.g3_calibration_views import (
    CandidateSelectionTaskView,
    CandidateViabilityTaskView,
    build_candidate_selection_task_view,
    build_candidate_viability_task_view,
    is_registered_candidate_selection_task_view,
    is_registered_candidate_viability_task_view,
)
from prpg.model.g3_task_publication import (
    block_calibration_executor_source_slots,
    candidate_selection_executor_source_slots,
    candidate_viability_executor_source_slots,
    scientific_task_source_slots,
)
from prpg.model.registered_fixed_point import (
    RegisteredDesignFixedPoint,
    RegisteredDesignFixedPointIteration,
    assemble_registered_design_fixed_point,
    is_registered_design_fixed_point,
    registered_design_fixed_point_identity,
    registered_design_fixed_point_sources,
    registered_final_block_projection_identity,
    registered_final_selected_blocks,
    registered_final_selection_projection_identity,
    registered_final_viability_projection_identity,
)


@dataclass(frozen=True, slots=True)
class _FixedPointBundle:
    suites: tuple[RegisteredDesignCandidateSuite, ...]
    viability: tuple[CandidateViabilityResult, ...]
    selections: tuple[CandidateGateResult, ...]
    fixed_point: RegisteredDesignFixedPoint


def _execute_stages(
    suite: RegisteredDesignCandidateSuite,
) -> tuple[CandidateViabilityResult, CandidateGateResult]:
    viability = execute_registered_candidate_viability(suite)
    selection = run_owner_fixed_k4_selection(
        viability,
        _registered_selection_bootstrap(),
    )
    return viability, selection


def _execute_chain(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> tuple[
    tuple[RegisteredDesignCandidateSuite, ...],
    tuple[CandidateViabilityResult, ...],
    tuple[CandidateGateResult, ...],
]:
    suites = [_registered_iteration_suite(monkeypatch, fixed_k4=True)]
    while len(suites) < count:
        suites.append(_next_registered_iteration_suite(suites[-1]))
    stages = tuple(_execute_stages(item) for item in suites)
    return (
        tuple(suites),
        tuple(item[0] for item in stages),
        tuple(item[1] for item in stages),
    )


def _install_controlled_identity_overrides(
    monkeypatch: pytest.MonkeyPatch,
    viability: tuple[CandidateViabilityResult, ...],
    selections: tuple[CandidateGateResult, ...],
    mutate_projection: Callable[[], None],
) -> None:
    """Keep real parents while controlling only terminal-state test rows.

    These tests target the fixed-point state machine and publication handling,
    not candidate-fit production (which has its own producer tests).  Capture
    each real closed-executor identity before changing the compact audit
    projection, then update only the two result fingerprints that the
    controlled projection necessarily changes.
    """

    original_viability_verifier = (
        gate_module.verified_candidate_viability_execution_identity
    )
    original_selection_verifier = gate_module.verified_candidate_gate_execution_identity
    original_viability = {
        id(item): original_viability_verifier(item) for item in viability
    }
    original_selection = {
        id(item): original_selection_verifier(item) for item in selections
    }
    mutate_projection()
    viability_identities = {
        id(item): replace(
            original_viability[id(item)],
            viability_result_fingerprint=(
                gate_module._candidate_viability_result_fingerprint(item)
            ),
        )
        for item in viability
    }
    selection_identities = {
        id(item): replace(
            original_selection[id(item)],
            viability_result_fingerprint=(
                viability_identities[
                    id(item.viability_result)
                ].viability_result_fingerprint
            ),
            selection_fingerprint=gate_module._candidate_selection_fingerprint(
                item.selection
            ),
        )
        for item in selections
    }

    def verify_viability(value: CandidateViabilityResult) -> object:
        return viability_identities.get(id(value)) or original_viability_verifier(value)

    def verify_selection(value: CandidateGateResult) -> object:
        return selection_identities.get(id(value)) or original_selection_verifier(value)

    monkeypatch.setattr(
        gate_module,
        "verified_candidate_viability_execution_identity",
        verify_viability,
    )
    monkeypatch.setattr(
        gate_module,
        "verified_candidate_gate_execution_identity",
        verify_selection,
    )
    monkeypatch.setattr(
        fixed_point_module,
        "verified_candidate_viability_execution_identity",
        verify_viability,
    )
    monkeypatch.setattr(
        fixed_point_module,
        "verified_candidate_gate_execution_identity",
        verify_selection,
    )


def _set_admissible_candidates(
    value: CandidateViabilityResult,
    candidates: tuple[int, ...],
) -> None:
    rows = tuple(
        replace(
            row,
            viability=replace(
                row.viability,
                rejection_reasons=(
                    ()
                    if row.n_states in candidates
                    else row.viability.rejection_reasons
                ),
            ),
        )
        for row in value.audit_rows
    )
    object.__setattr__(value, "audit_rows", rows)


@pytest.fixture(scope="module")
def converged_bundle() -> Iterator[_FixedPointBundle]:
    patch = pytest.MonkeyPatch()
    try:
        first = _registered_iteration_suite(patch, fixed_k4=True)
        second = _next_registered_iteration_suite(first)
        first_viability, first_selection = _execute_stages(first)
        second_viability, second_selection = _execute_stages(second)
        fixed_point = assemble_registered_design_fixed_point(
            (
                (first, first_viability, first_selection),
                (second, second_viability, second_selection),
            )
        )
        yield _FixedPointBundle(
            (first, second),
            (first_viability, second_viability),
            (first_selection, second_selection),
            fixed_point,
        )
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def no_admissible_bundle() -> Iterator[_FixedPointBundle]:
    patch = pytest.MonkeyPatch()
    try:
        suite = _registered_iteration_suite(patch, viable=False, fixed_k4=True)
        viability, selection = _execute_stages(suite)
        fixed_point = assemble_registered_design_fixed_point(
            ((suite, viability, selection),)
        )
        yield _FixedPointBundle((suite,), (viability,), (selection,), fixed_point)
    finally:
        patch.undo()


def _relax_geometry_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every authority check while reducing only the 5,000-history size."""

    viability_verifier = (
        gate_module.verified_registered_candidate_viability_execution_identity
    )
    selection_verifier = (
        gate_module.verified_registered_candidate_gate_execution_identity
    )

    def viability(value: CandidateViabilityResult, **_kwargs: object) -> object:
        return viability_verifier(value, require_strict_canonical_geometry=False)

    def selection(value: CandidateGateResult, **_kwargs: object) -> object:
        return selection_verifier(value, require_strict_canonical_geometry=False)

    monkeypatch.setattr(
        view_module,
        "verified_registered_candidate_viability_execution_identity",
        viability,
    )
    monkeypatch.setattr(
        view_module,
        "verified_registered_candidate_gate_execution_identity",
        selection,
    )
    monkeypatch.setattr(
        publication_module,
        "verified_registered_candidate_viability_execution_identity",
        viability,
    )
    monkeypatch.setattr(
        publication_module,
        "verified_registered_candidate_gate_execution_identity",
        selection,
    )


def _task_views(
    bundle: _FixedPointBundle,
) -> tuple[CandidateViabilityTaskView, CandidateSelectionTaskView]:
    viability = build_candidate_viability_task_view(
        bundle.viability[-1],
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
        run_id=_RUN,
        binding_fingerprint=_VIABILITY_BINDING,
        require_canonical_suite=True,
    )
    selection = build_candidate_selection_task_view(
        viability,
        bundle.selections[-1],
        run_id=_RUN,
        binding_fingerprint=_SELECTION_BINDING,
    )
    return viability, selection


def test_reduced_geometry_converges_only_on_adjacent_equality_and_reuses_blocks(
    converged_bundle: _FixedPointBundle,
) -> None:
    value = converged_bundle.fixed_point
    identity = registered_design_fixed_point_identity(value)

    assert is_registered_design_fixed_point(value)
    assert value.outcome == "converged"
    assert value.passed
    assert value.failure_reason is None
    assert value.converged_at_iteration == 2
    assert value.iterations[0].state == value.iterations[1].state == value.state
    assert identity.final_selection_projection.iteration_fingerprints == tuple(
        item.iteration_fingerprint for item in value.iterations
    )
    assert registered_final_selected_blocks(converged_bundle.selections[-1]) == (
        converged_bundle.suites[-1].rows[2].monthly_block,
        converged_bundle.suites[-1].rows[2].daily_block,
    )
    sources, monthly, daily = registered_design_fixed_point_sources(value)
    assert sources[-1] == (
        converged_bundle.suites[-1],
        converged_bundle.viability[-1],
        converged_bundle.selections[-1],
    )
    assert monthly is converged_bundle.suites[-1].rows[2].monthly_block
    assert daily is converged_bundle.suites[-1].rows[2].daily_block


def test_task8_binds_and_publishes_exact_converged_block_parents(
    converged_bundle: _FixedPointBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monthly, daily = registered_final_selected_blocks(converged_bundle.selections[-1])
    monthly_block = registered_block_calibration_identity(monthly)
    projection = registered_final_block_projection_identity(monthly, daily)
    kernel_core = "a" * 64

    def kernel(frequency: str, selected_length: int) -> SimpleNamespace:
        identity = SimpleNamespace(
            artifact="design",
            frequency=frequency,
            n_states=projection.selected_n_states,
            selected_length=selected_length,
            parent_model_fingerprint=monthly_block.parent_model_fingerprint,
            master_seed=projection.master_seed,
            scientific_version=projection.scientific_version,
            source_materialization_fingerprint="b" * 64,
            rng_execution_fingerprint="c" * 64,
            execution_fingerprint=("d" if frequency == "monthly" else "e") * 64,
        )
        return SimpleNamespace(
            model_artifact_fingerprint=kernel_core,
            identity=identity,
        )

    monthly_kernel = kernel(
        "monthly",
        monthly.execution.policy_selection.selected_length,
    )
    daily_kernel = kernel(
        "daily",
        daily.execution.policy_selection.selected_length,
    )
    transition_identity = SimpleNamespace(
        role="design",
        n_states=projection.selected_n_states,
        parent_model_fingerprint=monthly_block.parent_model_fingerprint,
        master_seed=projection.master_seed,
        scientific_version=projection.scientific_version,
        source_materialization_fingerprint="f" * 64,
        rng_execution_fingerprint="1" * 64,
        execution_fingerprint="2" * 64,
    )
    transition = SimpleNamespace(identity=transition_identity)
    monkeypatch.setattr(
        boundary_module,
        "kernel_calibration_execution_identity",
        lambda value: value.identity,
    )
    monkeypatch.setattr(
        boundary_module,
        "transition_bootstrap_aggregate_identity",
        lambda value: value.identity,
    )
    source = BlockCalibrationGateEvidence(
        role="design",
        selected_k=projection.selected_n_states,
        monthly=monthly,
        daily=daily,
        monthly_kernel=monthly_kernel,
        daily_kernel=daily_kernel,
        transition=transition,
    )
    view = build_block_calibration_task_view(
        source,
        task_id="design_block_calibration",
        run_id=_RUN,
        binding_fingerprint="f" * 64,
    )
    assert is_block_calibration_task_view(view)
    assert view.design_fixed_point_fingerprint == projection.fixed_point_fingerprint
    raw = block_calibration_executor_source_slots(
        source,
        task_id="design_block_calibration",
    )
    assert raw == scientific_task_source_slots(view)
    assert (
        dict(raw.materialization_fingerprints)[
            "design_fixed_point_final_block_parent_materialization"
        ]
        == view.design_fixed_point_final_block_parent_fingerprint
    )

    with pytest.raises(ModelError, match="exact final fixed-point"):
        registered_final_block_projection_identity(copy.copy(monthly), daily)


def test_task2_projection_is_final_only_while_task3_binds_full_history(
    converged_bundle: _FixedPointBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _relax_geometry_only(monkeypatch)
    viability, selection = _task_views(converged_bundle)
    assert is_registered_candidate_viability_task_view(viability)
    assert is_registered_candidate_selection_task_view(selection)

    task2_names = {item.name for item in fields(CandidateViabilityTaskView)}
    assert "fixed_point_viability_projection_fingerprint" in task2_names
    assert "fixed_point_final_iteration" in task2_names
    assert "selection" not in task2_names
    assert "selected_n_states" not in task2_names
    assert "selection_fingerprint" not in task2_names
    assert "fixed_point_iteration_history_fingerprint" not in task2_names
    assert "fixed_point_fingerprint" not in task2_names

    raw_task2 = candidate_viability_executor_source_slots(
        converged_bundle.viability[-1],
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    )
    final_task2 = scientific_task_source_slots(viability)
    assert raw_task2 == final_task2
    task2_slots = dict(raw_task2.materialization_fingerprints)
    assert "design_fixed_point_viability_projection" in task2_slots
    assert "design_fixed_point_history_materialization" not in task2_slots
    assert "design_fixed_point_final_selection_materialization" not in task2_slots
    assert all(
        item.iteration_fingerprint not in task2_slots.values()
        for item in converged_bundle.fixed_point.iterations[:-1]
    )

    raw_task3 = candidate_selection_executor_source_slots(
        viability,
        converged_bundle.selections[-1],
    )
    final_task3 = scientific_task_source_slots(selection)
    assert raw_task3 == final_task3
    task3_slots = dict(raw_task3.materialization_fingerprints)
    assert task3_slots["design_fixed_point_history_materialization"] == (
        converged_bundle.fixed_point.iteration_history_fingerprint
    )
    assert task3_slots["design_fixed_point_final_selection_materialization"] == (
        registered_final_selection_projection_identity(
            converged_bundle.selections[-1]
        ).final_selection_fingerprint
    )
    assert selection.fixed_point_outcome == "converged"
    assert selection.passed


def test_no_admissible_is_minted_and_publishes_typed_task_failures(
    no_admissible_bundle: _FixedPointBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = no_admissible_bundle.fixed_point
    assert value.outcome == "no_admissible_candidates"
    assert not value.passed
    assert value.state is None
    assert value.converged_at_iteration == 1
    assert has_registered_final_fixed_point_candidate_authority(
        no_admissible_bundle.viability[-1]
    )
    assert has_registered_final_fixed_point_candidate_authority(
        no_admissible_bundle.selections[-1]
    )
    with pytest.raises(ModelError, match="no selected block"):
        registered_final_selected_blocks(no_admissible_bundle.selections[-1])

    _relax_geometry_only(monkeypatch)
    viability, selection = _task_views(no_admissible_bundle)
    assert not viability.passed
    assert not selection.passed
    assert selection.fixed_point_outcome == "no_admissible_candidates"
    assert selection.fixed_point_failure_reason == "owner_fixed_k4_not_admissible"
    assert candidate_viability_executor_source_slots(
        no_admissible_bundle.viability[-1],
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    ) == scientific_task_source_slots(viability)
    assert candidate_selection_executor_source_slots(
        viability,
        no_admissible_bundle.selections[-1],
    ) == scientific_task_source_slots(selection)


def test_viable_but_empty_selection_is_typed_task3_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suites, viability, selections = _execute_chain(monkeypatch, 1)

    def fail_selection() -> None:
        object.__setattr__(
            selections[0],
            "selection",
            replace(
                selections[0].selection,
                selected_n_states=None,
                passed=False,
                failure_reason="synthetic_empty_intersection",
            ),
        )

    _install_controlled_identity_overrides(
        monkeypatch,
        viability,
        selections,
        fail_selection,
    )
    fixed_point = assemble_registered_design_fixed_point(
        ((suites[0], viability[0], selections[0]),)
    )
    bundle = _FixedPointBundle(suites, viability, selections, fixed_point)
    assert fixed_point.outcome == "candidate_selection_failed"
    assert fixed_point.failure_reason == "synthetic_empty_intersection"
    assert not fixed_point.passed

    _relax_geometry_only(monkeypatch)
    task2, task3 = _task_views(bundle)
    assert task2.passed
    assert not task3.passed
    assert task3.fixed_point_outcome == "candidate_selection_failed"
    assert candidate_viability_executor_source_slots(
        viability[-1],
        master_seed=_MASTER_SEED,
        scientific_version=_SCIENTIFIC_VERSION,
    ) == scientific_task_source_slots(task2)
    assert candidate_selection_executor_source_slots(
        task2,
        selections[-1],
    ) == scientific_task_source_slots(task3)


def test_report_only_non_k_changes_do_not_delay_registered_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suites, viability, selections = _execute_chain(monkeypatch, 2)

    def changing_report_projection() -> None:
        for result, candidates in zip(
            viability,
            ((4,), (3, 4)),
            strict=True,
        ):
            _set_admissible_candidates(result, candidates)

    _install_controlled_identity_overrides(
        monkeypatch,
        viability,
        selections,
        changing_report_projection,
    )
    fixed_point = assemble_registered_design_fixed_point(
        tuple(zip(suites, viability, selections, strict=True))
    )
    bundle = _FixedPointBundle(suites, viability, selections, fixed_point)
    assert fixed_point.outcome == "converged"
    assert fixed_point.failure_reason is None
    assert fixed_point.passed
    assert fixed_point.state is not None
    assert fixed_point.state.selected_n_states == 4
    assert not hasattr(fixed_point.state, "admissible_n_states")

    _relax_geometry_only(monkeypatch)
    task2, task3 = _task_views(bundle)
    assert task2.passed
    assert task3.passed
    assert task3.fixed_point_outcome == "converged"


def test_full_non_k_report_family_is_not_a_distinct_fixed_point_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suites, viability, selections = _execute_chain(monkeypatch, 2)

    def expanded_report_projection() -> None:
        for result, candidates in zip(
            viability,
            ((4,), (2, 3, 4, 5)),
            strict=True,
        ):
            _set_admissible_candidates(result, candidates)

    _install_controlled_identity_overrides(
        monkeypatch,
        viability,
        selections,
        expanded_report_projection,
    )
    fixed_point = assemble_registered_design_fixed_point(
        tuple(zip(suites, viability, selections, strict=True))
    )
    bundle = _FixedPointBundle(suites, viability, selections, fixed_point)
    assert fixed_point.outcome == "converged"
    assert fixed_point.converged_at_iteration == 2
    assert fixed_point.passed

    _relax_geometry_only(monkeypatch)
    task2, task3 = _task_views(bundle)
    assert task2.passed
    assert task3.passed
    assert task3.fixed_point_outcome == "converged"


def test_flags_seals_copies_equivalents_mutations_and_reordering_fail_closed(
    converged_bundle: _FixedPointBundle,
) -> None:
    value = converged_bundle.fixed_point
    final_viability = converged_bundle.viability[-1]
    final_selection = converged_bundle.selections[-1]
    earlier_viability = converged_bundle.viability[0]
    earlier_selection = converged_bundle.selections[0]

    assert not has_registered_final_fixed_point_candidate_authority(
        copy.copy(final_viability)
    )
    assert not has_registered_final_fixed_point_candidate_authority(
        copy.copy(final_selection)
    )
    assert not is_registered_design_fixed_point(copy.copy(value))

    object.__setattr__(earlier_viability, "_final_fixed_point_authority", True)
    object.__setattr__(
        earlier_viability,
        "_final_fixed_point_seal",
        final_viability._final_fixed_point_seal,
    )
    object.__setattr__(earlier_selection, "_final_fixed_point_authority", True)
    object.__setattr__(
        earlier_selection,
        "_final_fixed_point_seal",
        final_selection._final_fixed_point_seal,
    )
    try:
        assert not has_registered_final_fixed_point_candidate_authority(
            earlier_viability
        )
        assert not has_registered_final_fixed_point_candidate_authority(
            earlier_selection
        )
    finally:
        object.__setattr__(earlier_viability, "_final_fixed_point_authority", False)
        object.__setattr__(earlier_viability, "_final_fixed_point_seal", None)
        object.__setattr__(earlier_selection, "_final_fixed_point_authority", False)
        object.__setattr__(earlier_selection, "_final_fixed_point_seal", None)

    retained_iterations = value.iterations
    object.__setattr__(value, "iterations", tuple(reversed(value.iterations)))
    try:
        assert not is_registered_design_fixed_point(value)
        assert not has_registered_final_fixed_point_candidate_authority(final_selection)
    finally:
        object.__setattr__(value, "iterations", retained_iterations)
    assert is_registered_design_fixed_point(value)

    final_suite = converged_bundle.suites[-1]
    original = final_suite.suite_fingerprint
    object.__setattr__(final_suite, "suite_fingerprint", "0" * 64)
    try:
        assert not has_registered_final_fixed_point_candidate_authority(final_viability)
    finally:
        object.__setattr__(final_suite, "suite_fingerprint", original)
    assert is_registered_design_fixed_point(value)


def test_favorable_truncation_extra_iteration_reorder_and_skip_are_rejected(
    converged_bundle: _FixedPointBundle,
) -> None:
    first = (
        converged_bundle.suites[0],
        converged_bundle.viability[0],
        converged_bundle.selections[0],
    )
    second = (
        converged_bundle.suites[1],
        converged_bundle.viability[1],
        converged_bundle.selections[1],
    )
    with pytest.raises(ModelError, match="favorably truncated"):
        assemble_registered_design_fixed_point((first,))
    with pytest.raises(ModelError, match="object-unique"):
        assemble_registered_design_fixed_point((first, first))
    with pytest.raises(ModelError, match="ancestry differs"):
        assemble_registered_design_fixed_point((second, first))


def test_constructors_do_not_mint_authority() -> None:
    with pytest.raises(TypeError, match="created only"):
        RegisteredDesignFixedPoint()
    with pytest.raises(TypeError, match="created only"):
        RegisteredDesignFixedPointIteration()


def test_final_viability_projection_has_no_selection_or_history_fields(
    converged_bundle: _FixedPointBundle,
) -> None:
    projection = registered_final_viability_projection_identity(
        converged_bundle.viability[-1]
    )
    names = {item.name for item in fields(type(projection))}
    assert names == {
        "contract_version",
        "master_seed",
        "scientific_version",
        "final_iteration",
        "final_suite_fingerprint",
        "final_viability_result_fingerprint",
        "viability_projection_fingerprint",
    }
