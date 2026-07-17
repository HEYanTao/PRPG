"""Read-only, noncanonical v11 K4 fit and source-support diagnostic.

This script does not create launch authority, a registered task result, a model
artifact, or a block simulation.  It reproduces the deterministic 50-restart
design K4 fit and reports raw scientific dispositions plus source-start support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prpg.config import load_config
from prpg.data.processed import ProcessedSnapshotStore
from prpg.model.block_length import state_support
from prpg.model.calibration import (
    decoded_return_pools,
    preliminary_block_calibrations,
    prepare_hmm_feature_matrix,
)
from prpg.model.g3_preflight import (
    CANONICAL_CALIBRATION_INPUT_FINGERPRINT,
    CANONICAL_CONFIG_FINGERPRINT,
    CANONICAL_PROCESSED_DATA_FINGERPRINT,
)
from prpg.model.hmm import (
    deterministic_hmm_restart_seeds,
    fit_gaussian_hmm_restarts,
)
from prpg.model.input import load_calibration_input
from prpg.model.materialization_identity import (
    gaussian_hmm_fit_fingerprint,
    hmm_feature_matrix_fingerprint,
    scientific_array_fingerprint,
)
from prpg.model.regime_policy import OWNER_FIXED_K4_PROSPECTIVE_POLICY
from prpg.simulation.rng import ModelFitScope

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "canonical.yaml"
WORKERS = 9
N_STATES = 4


def _first_source_supported_length(
    *,
    starting_length: int,
    frequency: str,
    states: np.ndarray,
    continuation: np.ndarray,
) -> dict[str, object] | None:
    for intended in range(starting_length, 1, -1):
        support = state_support(
            states,
            continuation,
            intended_length=intended,
            selected_states=tuple(range(N_STATES)),
        )
        required = max(8, intended) if frequency == "monthly" else max(50, 2 * intended)
        if all(item.eligible_start_count >= required for item in support):
            return {
                "intended_length": intended,
                "required_eligible_starts": required,
                "eligible_start_counts": [
                    item.eligible_start_count for item in support
                ],
                "completed_episode_counts": [
                    item.completed_episodes for item in support
                ],
            }
    return None


def main() -> None:
    config = load_config(CONFIG_PATH)
    assert config.fingerprint() == CANONICAL_CONFIG_FINGERPRINT
    assert config.model.regime_selection_policy == "owner_fixed_k4_v3"
    assert config.model.fixed_regime_count == N_STATES
    assert config.simulation.scientific_version == 11

    calibration_input = load_calibration_input(
        ProcessedSnapshotStore(config.data.artifact_root),
        CANONICAL_PROCESSED_DATA_FINGERPRINT,
        calibration_config_fingerprint=config.fingerprint(),
        holdout_months=config.model.design_holdout_months,
    )
    assert calibration_input.fingerprint == CANONICAL_CALIBRATION_INPUT_FINGERPRINT
    design = calibration_input.design
    features = prepare_hmm_feature_matrix(
        calibration_input.full.macro_features,
        calibration_input.full.monthly_period_ordinals,
        calibration_input.full.monthly_continuation,
        role="design",
        design_rows=len(design.macro_features),
    )
    seeds = deterministic_hmm_restart_seeds(
        master_seed=config.run.master_seed,
        scientific_version=config.simulation.scientific_version,
        scope=ModelFitScope.DESIGN_MAIN,
        replicate=0,
        n_states=N_STATES,
    )
    fit = fit_gaussian_hmm_restarts(
        features,
        N_STATES,
        seeds,
        workers=WORKERS,
        owner_fixed_k4_policy=OWNER_FIXED_K4_PROSPECTIVE_POLICY,
    )
    pools = decoded_return_pools(fit, design)
    preliminary = preliminary_block_calibrations(design, role="design")

    best = fit.restart_diagnostics[fit.best_restart_index]
    identification = fit.identification
    transition = fit.historical_transition_support
    monthly = pools.monthly_diagnostics
    daily = pools.daily_diagnostics
    result = {
        "diagnostic_scope": "read_only_noncanonical_source_support_only",
        "realized_effective_block_simulation": "not_run",
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_fingerprint": config.fingerprint(),
        "processed_data_fingerprint": CANONICAL_PROCESSED_DATA_FINGERPRINT,
        "calibration_input_fingerprint": calibration_input.fingerprint,
        "master_seed": config.run.master_seed,
        "scientific_version": config.simulation.scientific_version,
        "workers": WORKERS,
        "n_states": fit.n_states,
        "feature_fingerprint": hmm_feature_matrix_fingerprint(features),
        "restart_seed_fingerprint": scientific_array_fingerprint(
            np.asarray(seeds, dtype=np.uint32),
            role="v11_design_k4_restart_seeds",
        ),
        "restart_seeds": list(seeds),
        "valid_restart_count": sum(item.valid for item in fit.restart_diagnostics),
        "best_restart_index": fit.best_restart_index,
        "best_seed": fit.best_seed,
        "best_restart_diagnostic": {
            "restart_index": best.restart_index,
            "seed": best.seed,
            "converged": best.converged,
            "iterations": best.iterations,
            "log_likelihood": best.log_likelihood,
            "valid": best.valid,
            "covariance_condition_number": best.covariance_condition_number,
            "maximum_likelihood_decrease": best.maximum_likelihood_decrease,
            "preliminary_chain_passed": best.preliminary_chain_passed,
            "failure_type": best.failure_type,
            "failure_message": best.failure_message,
        },
        "fit_fingerprint": gaussian_hmm_fit_fingerprint(fit),
        "raw_identification": identification.as_dict(),
        "raw_historical_transition_support": {
            "passed": transition.passed,
            "rejection_reasons": list(transition.rejection_reasons),
            "viterbi_transition_counts": transition.viterbi_counts.tolist(),
            "supported_off_diagonal_edges": (
                transition.supported_off_diagonal_edges.tolist()
            ),
            "supported_self_loop_states": list(transition.supported_self_loop_states),
            "decoded_entries": transition.decoded_entries.tolist(),
            "decoded_exits": transition.decoded_exits.tolist(),
        },
        "owner_fixed_k4_disposition": (
            None
            if fit.owner_fixed_k4_disposition is None
            else fit.owner_fixed_k4_disposition.as_dict()
        ),
        "monthly": {
            "state_counts": monthly.state_counts.tolist(),
            "occupancy": monthly.occupancy.tolist(),
            "observed_episode_counts": monthly.episode_counts.tolist(),
            "completed_episode_counts": monthly.completed_episode_counts.tolist(),
            "starting_length": preliminary.monthly.combined.starting_length,
            "first_source_supported_length": _first_source_supported_length(
                starting_length=preliminary.monthly.combined.starting_length,
                frequency="monthly",
                states=pools.monthly_states,
                continuation=design.monthly_continuation[1:],
            ),
        },
        "daily": {
            "state_counts": daily.state_counts.tolist(),
            "occupancy": daily.occupancy.tolist(),
            "observed_run_counts": daily.episode_counts.tolist(),
            "completed_run_counts": daily.completed_episode_counts.tolist(),
            "starting_length": preliminary.daily.combined.starting_length,
            "first_source_supported_length": _first_source_supported_length(
                starting_length=preliminary.daily.combined.starting_length,
                frequency="daily",
                states=pools.daily_states,
                continuation=design.daily_continuation[1:],
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
