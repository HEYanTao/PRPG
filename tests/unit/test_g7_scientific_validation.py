from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prpg.errors import IntegrityError, ValidationError
from prpg.validation import scientific as scientific_module
from prpg.validation.adversaries import (
    G5_ADVERSARY_SPECIFICATION_FINGERPRINT,
    fit_g5_adversary_pair,
)
from prpg.validation.qualification import (
    G5_ESTIMATOR_CONTRACT_FINGERPRINT,
    G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING,
    G5_THRESHOLD_ESTIMATOR_BINDING,
    build_g5_family_null_calibration,
    build_g5_reference_data,
)
from prpg.validation.records import ValidationDecision, build_canonical_results
from prpg.validation.scientific import (
    CANONICAL_G7_SCIENTIFIC_PLAN,
    G7_PRIMARY_HF_PATHS,
    G7_PRIMARY_LF_PATHS,
    G7_THRESHOLD_ADVERSARY_FIT_BINDING,
    G7ScientificMetricHook,
    build_g7_frozen_scientific_contract,
    build_g7_scientific_pass_report,
    publish_g7_scientific_pass_report,
    publish_g7_validated_pair,
    reduced_g7_scientific_plan,
)
from prpg.validation.sobol import registered_sobol_directions
from prpg.validation.structural import StructuralPathView


def _fp(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _de_bruijn(alphabet: int, order: int) -> list[int]:
    work = [0] * (alphabet * order)
    sequence: list[int] = []

    def visit(prefix: int, period: int) -> None:
        if prefix > order:
            if order % period == 0:
                sequence.extend(work[1 : period + 1])
            return
        work[prefix] = work[prefix - period]
        visit(prefix + 1, period)
        for item in range(work[prefix - period] + 1, alphabet):
            work[prefix] = item
            visit(prefix + 1, prefix)

    visit(1, 1)
    return sequence


_STATE_CYCLE = _de_bruijn(8, 2)


def _balanced_returns(observations: int, *, scale_offset: float = 0.0) -> np.ndarray:
    assert (observations - 1) % len(_STATE_CYCLE) == 0
    states = [_STATE_CYCLE[index % len(_STATE_CYCLE)] for index in range(observations)]
    signs = np.asarray(
        [
            tuple(1.0 if state & (1 << asset) else -1.0 for asset in range(3))
            for state in states
        ]
    )
    scales = np.asarray((0.012, 0.007, 0.009)) * (1.0 + scale_offset)
    return signs * scales


def _contract() -> object:
    directions = registered_sobol_directions(master_seed=909, scientific_version=11)
    monthly = build_g5_reference_data(
        _balanced_returns(193), np.arange(193) % 4, frequency="monthly"
    )
    daily = build_g5_reference_data(
        _balanced_returns(257), np.arange(257) % 4, frequency="daily"
    )
    adversary_fit = fit_g5_adversary_pair(
        tuple(
            _balanced_returns(193, scale_offset=index * 1e-4) for index in range(1, 5)
        ),
        tuple(
            _balanced_returns(257, scale_offset=index * 2e-4) for index in range(1, 5)
        ),
    )
    grid = np.linspace(0.0, 1_000_000.0, 50_000)
    null = build_g5_family_null_calibration(
        {
            "g5_a": grid,
            "g5_b": grid + 1.0,
            "g5_c": grid + 2.0,
            "g5_d": grid + 3.0,
        },
        threshold_set_id="g5.production.historical-vector.v1",
        binding_fingerprints={
            "g5_design": _fp("g5-design"),
            G5_THRESHOLD_ESTIMATOR_BINDING: G5_ESTIMATOR_CONTRACT_FINGERPRINT,
            G5_THRESHOLD_ADVERSARY_SPECIFICATION_BINDING: (
                G5_ADVERSARY_SPECIFICATION_FINGERPRINT
            ),
            G7_THRESHOLD_ADVERSARY_FIT_BINDING: adversary_fit.fingerprint,
        },
        sobol_direction_fingerprint=directions.fingerprint,
    )
    assert null.canonical_thresholds is not None
    decisions = (
        ValidationDecision("candidate_counts_exact", True, True, True, "exact"),
        ValidationDecision(
            "g5_a_direct_alpha_equivalence", True, True, 0.0, "inside margin"
        ),
        ValidationDecision(
            "generation_policy_and_stream_bound", True, True, 11, "bound"
        ),
        ValidationDecision(
            "primary_estimator_contract_bound", True, True, True, "bound"
        ),
        ValidationDecision(
            "registered_evaluation_windows", True, True, True, "registered"
        ),
        ValidationDecision("g5_b_practical_cap", True, True, 0.0, "inside cap"),
        ValidationDecision("g5_c_practical_cap", True, True, 0.0, "inside cap"),
        ValidationDecision("g5_d_practical_cap", True, True, 0.0, "inside cap"),
        ValidationDecision(
            "latent_chain_correctness", True, True, 0.0, "passed fitted-chain gate"
        ),
        ValidationDecision(
            "g5_d_complete_path_diversity", True, True, 0, "no duplicates"
        ),
        *(
            ValidationDecision(f"{name}_holm", True, True, 1.0, "Holm non-rejection")
            for name in ("g5_a", "g5_b", "g5_c", "g5_d")
        ),
    )
    qualification = build_canonical_results(
        result_id="g5.candidate.historical-vector",
        threshold_fingerprint=null.canonical_thresholds.fingerprint,
        subject_fingerprint=_fp("qualification-bundle"),
        metric_fingerprints={
            "adversary_fit": adversary_fit.fingerprint,
            "observed_metrics": _fp("g5-observed-metrics"),
        },
        decisions=decisions,
    )
    return build_g7_frozen_scientific_contract(
        production_authority_fingerprint=_fp("g5-production-authority"),
        thresholds=null.canonical_thresholds,
        qualification_results=qualification,
        null_calibration=null,
        monthly_reference=monthly,
        daily_reference=daily,
        sobol_directions=directions,
        adversary_fit=adversary_fit,
        estimator_contract_fingerprint=G5_ESTIMATOR_CONTRACT_FINGERPRINT,
        evaluation_master_seed=909,
        policy_scientific_version=11,
        g3_production_latent_task_fingerprint=_fp("g3-latent-task"),
        g3_production_latent_checkpoint_fingerprint=_fp("g3-latent-checkpoint"),
        g3_production_adequacy_task_fingerprint=_fp("g3-adequacy-task"),
        g3_production_adequacy_checkpoint_fingerprint=_fp("g3-adequacy-checkpoint"),
    )


def _views() -> tuple[StructuralPathView, ...]:
    monthly = tuple(
        StructuralPathView(
            family="LF",
            frequency="monthly",
            path_entity=entity,
            path_id=f"LF{entity:06d}",
            log_returns=_balanced_returns(257, scale_offset=entity * 1e-5),
        )
        for entity in range(1, 4)
    )
    daily = tuple(
        StructuralPathView(
            family="HF",
            frequency="daily",
            path_entity=entity,
            path_id=f"HF{entity:06d}",
            log_returns=_balanced_returns(321, scale_offset=entity * 2e-5),
        )
        for entity in range(1, 3)
    )
    return (*monthly, *daily)


def _reduced_hook() -> G7ScientificMetricHook:
    return G7ScientificMetricHook(
        _contract(),  # type: ignore[arg-type]
        plan=reduced_g7_scientific_plan(
            expected_lf_paths=3,
            expected_hf_paths=2,
            primary_lf_paths=2,
            primary_hf_paths=2,
        ),
    )


def test_reduced_stream_is_deterministic_and_records_secondary_only_as_diagnostic() -> (
    None
):
    first = _reduced_hook()
    second = _reduced_hook()
    views = _views()
    for view in views:
        first.observe_path(view)
    for view in reversed(views):
        second.observe_path(view)

    first_payload = first.finalize()
    second_payload = second.finalize()

    assert first_payload == second_payload
    assessment = first.assessment
    assert assessment.fingerprint == second.assessment.fingerprint
    assert assessment.primary.lf_paths == 2
    assert assessment.primary.hf_paths == 2
    assert assessment.all_path_secondary is not None
    assert assessment.all_path_secondary.lf_paths == 3
    assert assessment.all_path_secondary.hf_paths == 2
    assert assessment.zero_nonfinite_or_data_issues
    assert not assessment.canonical_profile
    assert not assessment.primary_passed
    assert dict(assessment.primary.family_statistics).keys() == {
        "g5_a",
        "g5_b",
        "g5_c",
        "g5_d",
    }
    assert not any(
        name.endswith(("source_concentration", "repeated_subsequence"))
        for name, _ in assessment.primary.endpoint_values
    )
    assert assessment.primary.csv_excluded_endpoints == (
        "source_concentration",
        "repeated_subsequence",
        "latent_chain_correctness",
    )
    decisions = {item.name: item for item in assessment.decisions}
    assert decisions["shared_section14_estimator_contract"].passed
    assert decisions["inherited_state_source_evidence_bound"].passed
    assert assessment.estimator_contract_ready
    assert not decisions["canonical_primary_subset"].passed
    assert all(
        decisions[f"{name}_fixed_threshold"].passed and decisions[f"{name}_holm"].passed
        for name in ("g5_a", "g5_b", "g5_c", "g5_d")
    )


def test_hook_ignores_derived_frequency_but_aborts_incomplete_or_duplicate_base() -> (
    None
):
    hook = _reduced_hook()
    hook.observe_path(
        StructuralPathView(
            family="LF",
            frequency="annual",
            path_entity=1,
            path_id="LF000001",
            log_returns=np.zeros((2, 3)),
        )
    )
    hook.observe_path(_views()[0])
    with pytest.raises(IntegrityError, match="more than once"):
        hook.observe_path(_views()[0])

    incomplete = _reduced_hook()
    incomplete.observe_path(_views()[0])
    with pytest.raises(IntegrityError, match="incomplete"):
        incomplete.finalize()


def test_hook_aborts_nonfinite_and_noncanonical_path_identity() -> None:
    hook = _reduced_hook()
    bad = _views()[0].log_returns.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValidationError, match="NaN or infinity"):
        hook.observe_path(StructuralPathView("LF", "monthly", 1, "LF000001", bad))

    wrong_id = _reduced_hook()
    with pytest.raises(IntegrityError, match="ID is noncanonical"):
        wrong_id.observe_path(
            StructuralPathView("LF", "monthly", 1, "LF000999", _views()[0].log_returns)
        )


def test_hook_uses_frozen_fit_and_registered_output_independent_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = _reduced_hook()
    original = scientific_module.score_g5_adversary_path
    seen: list[str] = []

    def score(model: object, path: object) -> np.ndarray:
        assert (
            model is hook.contract.adversary_fit.monthly
            or model is hook.contract.adversary_fit.daily
        )
        seen.append(model.fingerprint)  # type: ignore[attr-defined]
        return original(model, path)  # type: ignore[arg-type]

    monkeypatch.setattr(scientific_module, "score_g5_adversary_path", score)
    first_uniform = scientific_module._registered_evaluation_uniform(
        hook.contract, family="LF", path_entity=1
    )
    second_uniform = scientific_module._registered_evaluation_uniform(
        hook.contract, family="LF", path_entity=1
    )
    assert first_uniform == second_uniform

    for view in _views():
        hook.observe_path(view)
    hook.finalize()

    assert len(seen) == len(_views())
    assert set(seen) == {
        hook.contract.adversary_fit.monthly.fingerprint,
        hook.contract.adversary_fit.daily.fingerprint,
    }


def test_contract_requires_exact_shared_estimator_fingerprint() -> None:
    contract = _contract()
    with pytest.raises(IntegrityError, match="estimator differs"):
        build_g7_frozen_scientific_contract(
            production_authority_fingerprint=contract.production_authority_fingerprint,  # type: ignore[attr-defined]
            thresholds=contract.thresholds,  # type: ignore[attr-defined]
            qualification_results=contract.qualification_results,  # type: ignore[attr-defined]
            null_calibration=contract.null_calibration,  # type: ignore[attr-defined]
            monthly_reference=contract.monthly_reference,  # type: ignore[attr-defined]
            daily_reference=contract.daily_reference,  # type: ignore[attr-defined]
            sobol_directions=contract.sobol_directions,  # type: ignore[attr-defined]
            adversary_fit=contract.adversary_fit,  # type: ignore[attr-defined]
            estimator_contract_fingerprint=_fp("wrong-estimator"),
            evaluation_master_seed=contract.evaluation_master_seed,  # type: ignore[attr-defined]
            policy_scientific_version=contract.policy_scientific_version,  # type: ignore[attr-defined]
            g3_production_latent_task_fingerprint=(
                contract.g3_production_latent_task_fingerprint  # type: ignore[attr-defined]
            ),
            g3_production_latent_checkpoint_fingerprint=(
                contract.g3_production_latent_checkpoint_fingerprint  # type: ignore[attr-defined]
            ),
            g3_production_adequacy_task_fingerprint=(
                contract.g3_production_adequacy_task_fingerprint  # type: ignore[attr-defined]
            ),
            g3_production_adequacy_checkpoint_fingerprint=(
                contract.g3_production_adequacy_checkpoint_fingerprint  # type: ignore[attr-defined]
            ),
        )


def test_contract_rejects_a_passed_but_nonqualification_result() -> None:
    contract = _contract()
    thresholds = contract.thresholds  # type: ignore[attr-defined]
    result = build_canonical_results(
        result_id="not-g5-qualification",
        threshold_fingerprint=thresholds.fingerprint,
        subject_fingerprint=_fp("subject"),
        metric_fingerprints={"metric": _fp("metric")},
        decisions=(ValidationDecision("other", True, True, True, "pass"),),
    )
    with pytest.raises(ValidationError, match="exact passed G5 qualification"):
        build_g7_frozen_scientific_contract(
            production_authority_fingerprint=contract.production_authority_fingerprint,  # type: ignore[attr-defined]
            thresholds=thresholds,
            qualification_results=result,
            null_calibration=contract.null_calibration,  # type: ignore[attr-defined]
            monthly_reference=contract.monthly_reference,  # type: ignore[attr-defined]
            daily_reference=contract.daily_reference,  # type: ignore[attr-defined]
            sobol_directions=contract.sobol_directions,  # type: ignore[attr-defined]
            adversary_fit=contract.adversary_fit,  # type: ignore[attr-defined]
            estimator_contract_fingerprint=(
                contract.estimator_contract_fingerprint  # type: ignore[attr-defined]
            ),
            evaluation_master_seed=contract.evaluation_master_seed,  # type: ignore[attr-defined]
            policy_scientific_version=contract.policy_scientific_version,  # type: ignore[attr-defined]
            g3_production_latent_task_fingerprint=(
                contract.g3_production_latent_task_fingerprint  # type: ignore[attr-defined]
            ),
            g3_production_latent_checkpoint_fingerprint=(
                contract.g3_production_latent_checkpoint_fingerprint  # type: ignore[attr-defined]
            ),
            g3_production_adequacy_task_fingerprint=(
                contract.g3_production_adequacy_task_fingerprint  # type: ignore[attr-defined]
            ),
            g3_production_adequacy_checkpoint_fingerprint=(
                contract.g3_production_adequacy_checkpoint_fingerprint  # type: ignore[attr-defined]
            ),
        )


def test_reduced_fixture_cannot_claim_canonical_primary_or_publish(
    tmp_path: object,
) -> None:
    assert CANONICAL_G7_SCIENTIFIC_PLAN.primary_lf_paths == G7_PRIMARY_LF_PATHS
    assert CANONICAL_G7_SCIENTIFIC_PLAN.primary_hf_paths == G7_PRIMARY_HF_PATHS
    with pytest.raises(ValidationError, match="cannot use canonical primary"):
        reduced_g7_scientific_plan(
            expected_lf_paths=G7_PRIMARY_LF_PATHS,
            expected_hf_paths=G7_PRIMARY_HF_PATHS,
            primary_lf_paths=G7_PRIMARY_LF_PATHS,
            primary_hf_paths=G7_PRIMARY_HF_PATHS,
        )

    hook = _reduced_hook()
    for view in _views():
        hook.observe_path(view)
    hook.finalize()
    with pytest.raises(IntegrityError, match="sealed structural report"):
        build_g7_scientific_pass_report(
            hook,
            object(),
            result_id="g7.production.historical-vector",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="not built by its producer"):
        publish_g7_scientific_pass_report(
            tmp_path,
            object(),  # type: ignore[arg-type]
        )


def test_coordinated_publication_writes_shared_reports_before_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structural_fingerprint = _fp("structural-report")
    manifest_fingerprint = _fp("consumer-manifest")
    structural = SimpleNamespace(
        canonical_profile=True,
        plan_fingerprint=scientific_module.CANONICAL_STRUCTURAL_PLAN.fingerprint,
        fingerprint=structural_fingerprint,
        manifest_sha256=manifest_fingerprint,
        schema_version=1,
        identity=lambda: {"schema_id": "test-structural"},
    )
    science = SimpleNamespace(
        fingerprint=_fp("scientific-pass"),
        structural_report_fingerprint=structural_fingerprint,
        consumer_manifest_sha256=manifest_fingerprint,
        assessment=SimpleNamespace(
            plan_fingerprint=scientific_module.CANONICAL_G7_SCIENTIFIC_PLAN.fingerprint
        ),
        identity_dict=lambda: {"schema_id": "test-science"},
    )
    monkeypatch.setattr(scientific_module, "_verify_pass_report", lambda value: value)
    monkeypatch.setattr(
        scientific_module,
        "is_sealed_structural_validation_report",
        lambda value: True,
    )

    published = publish_g7_validated_pair(
        tmp_path,
        scientific_report=science,  # type: ignore[arg-type]
        structural_report=structural,  # type: ignore[arg-type]
    )

    assert published.structural_report_path.is_file()
    assert published.scientific_report_path.is_file()
    assert published.data_validated_path.is_file()
    marker = json.loads(published.data_validated_path.read_bytes())
    assert marker["scientific_pass_fingerprint"] == science.fingerprint
    assert marker["structural_report_fingerprint"] == structural.fingerprint
    assert not (tmp_path / "COMPLETE").exists()
    assert not (tmp_path / "RELEASED").exists()


def test_coordinated_publication_rejects_mixed_scans_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structural = SimpleNamespace(
        canonical_profile=True,
        plan_fingerprint=scientific_module.CANONICAL_STRUCTURAL_PLAN.fingerprint,
        fingerprint=_fp("structural-report"),
        manifest_sha256=_fp("consumer-manifest"),
    )
    science = SimpleNamespace(
        fingerprint=_fp("scientific-pass"),
        structural_report_fingerprint=_fp("another-structural-report"),
        consumer_manifest_sha256=structural.manifest_sha256,
        assessment=SimpleNamespace(
            plan_fingerprint=scientific_module.CANONICAL_G7_SCIENTIFIC_PLAN.fingerprint
        ),
    )
    monkeypatch.setattr(scientific_module, "_verify_pass_report", lambda value: value)
    monkeypatch.setattr(
        scientific_module,
        "is_sealed_structural_validation_report",
        lambda value: True,
    )

    with pytest.raises(IntegrityError, match="share one scan"):
        publish_g7_validated_pair(
            tmp_path,
            scientific_report=science,  # type: ignore[arg-type]
            structural_report=structural,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "validation").exists()
    assert not (tmp_path / "DATA_VALIDATED").exists()
