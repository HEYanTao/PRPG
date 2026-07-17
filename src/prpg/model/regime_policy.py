"""Versioned owner disposition for the prospective fixed-K=4 regime model.

The scientific reports remain raw evidence.  This module does not mutate their
``passed`` flags or rejection reasons; it records the narrowly authorized
interpretation separately and exposes fail-closed effective-decision helpers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from prpg.errors import ModelError

if TYPE_CHECKING:
    from prpg.model.hmm import (
        GaussianHMMFit,
        HistoricalTransitionSupportReport,
        PosteriorIdentificationReport,
    )


OWNER_FIXED_K4_DISPOSITION_CONTRACT: Final = "owner-fixed-k4-rarity-v1"
OWNER_FIXED_K4_RECURRING_SUPPORT_CONTRACT: Final = "owner-fixed-k4-recurring-history-v1"
OWNER_FIXED_K4_MINIMUM_OBSERVED_EPISODES: Final = 2
OWNER_FIXED_K4_PROSPECTIVE_POLICY: Final = "owner_fixed_k4_v3"
OWNER_FIXED_K4_PROSPECTIVE_SCIENTIFIC_VERSION: Final = 11
OwnerFixedK4Policy: TypeAlias = Literal["owner_fixed_k4_v3"]

_EFFECTIVE_MASS_REASON = re.compile(
    r"state_[0-9]+_posterior_effective_mass_below_minimum\Z"
)
_SCARCITY_TRANSITION_REASON = re.compile(
    r"state_[0-9]+_decoded_(?:entries|exits)_below_three\Z"
)
_SCARCITY_TRANSITION_LITERALS: Final = frozenset(
    {
        "supported_off_diagonal_graph_not_strongly_connected",
        "no_historically_supported_self_loop",
    }
)


class _RawScientificReport(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def rejection_reasons(self) -> tuple[str, ...]: ...


def owner_fixed_k4_policy_for_scientific_version(
    scientific_version: int,
) -> OwnerFixedK4Policy | None:
    """Return the prospective policy only for its preregistered version.

    This helper is intentionally exact rather than monotone: later scientific
    versions must opt in under their own prospective contract instead of
    silently inheriting version 3's owner disposition.
    """

    if isinstance(scientific_version, bool) or not isinstance(scientific_version, int):
        raise ModelError("scientific version must be an integer")
    if scientific_version == OWNER_FIXED_K4_PROSPECTIVE_SCIENTIFIC_VERSION:
        return "owner_fixed_k4_v3"
    return None


def validate_owner_fixed_k4_policy(
    policy: OwnerFixedK4Policy | None,
) -> OwnerFixedK4Policy | None:
    """Validate the closed explicit policy switch used by HMM fitters."""

    if policy is None:
        return None
    if policy != OWNER_FIXED_K4_PROSPECTIVE_POLICY:
        raise ModelError("unknown owner-fixed K=4 HMM policy")
    return policy


@dataclass(frozen=True, slots=True)
class OwnerFixedK4Disposition:
    """Immutable record of the owner's narrow K=4 rarity interpretation."""

    contract_version: Literal["owner-fixed-k4-rarity-v1"]
    n_states: Literal[4]
    identification_rejection_reasons: tuple[str, ...]
    transition_rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != OWNER_FIXED_K4_DISPOSITION_CONTRACT:
            raise ModelError("owner K=4 disposition contract is invalid")
        if self.n_states != 4:
            raise ModelError("owner K=4 disposition requires exactly four states")
        if not all(
            _is_effective_mass_reason(reason)
            for reason in self.identification_rejection_reasons
        ):
            raise ModelError(
                "owner K=4 disposition contains a binding identification failure"
            )
        if not all(
            _is_scarcity_transition_reason(reason)
            for reason in self.transition_rejection_reasons
        ):
            raise ModelError(
                "owner K=4 disposition contains a binding transition failure"
            )

    def as_dict(self) -> dict[str, object]:
        """Return canonical primitive metadata for model materialization identity."""

        return {
            "contract_version": self.contract_version,
            "n_states": self.n_states,
            "identification_rejection_reasons": list(
                self.identification_rejection_reasons
            ),
            "transition_rejection_reasons": list(self.transition_rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class OwnerFixedK4RecurringHistorySupport:
    """Raw recurring-history counts and their fixed-K4 decision."""

    contract_version: Literal["owner-fixed-k4-recurring-history-v1"]
    n_states: Literal[4]
    minimum_observed_episodes: Literal[2]
    monthly_episode_counts: tuple[int, int, int, int]
    daily_run_counts: tuple[int, int, int, int]
    passed: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return canonical audit metadata without hiding rare-state counts."""

        return {
            "contract_version": self.contract_version,
            "n_states": self.n_states,
            "minimum_observed_episodes": self.minimum_observed_episodes,
            "monthly_episode_counts": list(self.monthly_episode_counts),
            "daily_run_counts": list(self.daily_run_counts),
            "passed": self.passed,
            "rejection_reasons": list(self.rejection_reasons),
        }


class OwnerFixedK4RecurringHistoryFailure(ModelError):
    """Scientific non-pass for inadequate recurring K=4 history support."""

    report: OwnerFixedK4RecurringHistorySupport

    def __init__(self, report: OwnerFixedK4RecurringHistorySupport) -> None:
        self.report = report
        super().__init__(
            "owner-fixed K=4 recurring-history support failed",
            details=report.as_dict(),
        )


def owner_fixed_k4_recurring_history_support(
    *,
    monthly_episode_counts: Sequence[int],
    daily_run_counts: Sequence[int],
) -> OwnerFixedK4RecurringHistorySupport:
    """Require two observed monthly episodes and two daily runs per state."""

    monthly = _four_nonnegative_counts(
        monthly_episode_counts,
        label="monthly episode counts",
    )
    daily = _four_nonnegative_counts(
        daily_run_counts,
        label="daily run counts",
    )
    reasons = tuple(
        [
            f"state_{state}_monthly_observed_episodes_below_two"
            for state, count in enumerate(monthly)
            if count < OWNER_FIXED_K4_MINIMUM_OBSERVED_EPISODES
        ]
        + [
            f"state_{state}_daily_runs_below_two"
            for state, count in enumerate(daily)
            if count < OWNER_FIXED_K4_MINIMUM_OBSERVED_EPISODES
        ]
    )
    return OwnerFixedK4RecurringHistorySupport(
        contract_version=OWNER_FIXED_K4_RECURRING_SUPPORT_CONTRACT,
        n_states=4,
        minimum_observed_episodes=OWNER_FIXED_K4_MINIMUM_OBSERVED_EPISODES,
        monthly_episode_counts=monthly,
        daily_run_counts=daily,
        passed=not reasons,
        rejection_reasons=reasons,
    )


def owner_fixed_k4_disposition(
    *,
    n_states: int,
    identification: PosteriorIdentificationReport,
    transition: HistoricalTransitionSupportReport,
) -> OwnerFixedK4Disposition | None:
    """Return the disposition only for K=4 reports within its exact scope."""

    if n_states != 4:
        return None
    if not _report_is_disposition_eligible(
        identification,
        allowed_reason=_is_effective_mass_reason,
    ) or not _report_is_disposition_eligible(
        transition,
        allowed_reason=_is_scarcity_transition_reason,
    ):
        return None
    return OwnerFixedK4Disposition(
        contract_version=OWNER_FIXED_K4_DISPOSITION_CONTRACT,
        n_states=4,
        identification_rejection_reasons=identification.rejection_reasons,
        transition_rejection_reasons=transition.rejection_reasons,
    )


def validate_owner_fixed_k4_disposition(fit: GaussianHMMFit) -> None:
    """Reject a disposition that is detached from its fit's exact raw reports."""

    disposition = getattr(fit, "owner_fixed_k4_disposition", None)
    if disposition is None:
        return
    if not isinstance(disposition, OwnerFixedK4Disposition):
        raise ModelError("HMM owner K=4 disposition has an invalid type")
    if fit.n_states != disposition.n_states:
        raise ModelError("HMM owner K=4 disposition state count is inconsistent")
    if (
        disposition.identification_rejection_reasons
        != fit.identification.rejection_reasons
        or disposition.transition_rejection_reasons
        != fit.historical_transition_support.rejection_reasons
    ):
        raise ModelError("HMM owner K=4 disposition is detached from raw evidence")
    if not _report_is_disposition_eligible(
        fit.identification,
        allowed_reason=_is_effective_mass_reason,
    ) or not _report_is_disposition_eligible(
        fit.historical_transition_support,
        allowed_reason=_is_scarcity_transition_reason,
    ):
        raise ModelError("HMM owner K=4 disposition exceeds its authorized scope")


def effective_identification_passed(fit: GaussianHMMFit) -> bool:
    """Return raw identification unless a valid K=4 disposition applies."""

    validate_owner_fixed_k4_disposition(fit)
    return effective_identification_report_passed(
        fit.identification,
        n_states=fit.n_states,
        disposition=getattr(fit, "owner_fixed_k4_disposition", None),
    )


def effective_transition_support_passed(fit: GaussianHMMFit) -> bool:
    """Return raw transition support unless a valid K=4 disposition applies."""

    validate_owner_fixed_k4_disposition(fit)
    return effective_transition_report_passed(
        fit.historical_transition_support,
        n_states=fit.n_states,
        disposition=getattr(fit, "owner_fixed_k4_disposition", None),
    )


def effective_identification_report_passed(
    report: PosteriorIdentificationReport,
    *,
    n_states: int,
    disposition: OwnerFixedK4Disposition | None,
) -> bool:
    """Apply an existing K=4 authority to a derived identification report."""

    return _effective_report_passed(
        report,
        n_states=n_states,
        disposition=disposition,
        allowed_reason=_is_effective_mass_reason,
    )


def effective_transition_report_passed(
    report: HistoricalTransitionSupportReport,
    *,
    n_states: int,
    disposition: OwnerFixedK4Disposition | None,
) -> bool:
    """Apply an existing K=4 authority to a rebuilt transition report."""

    return _effective_report_passed(
        report,
        n_states=n_states,
        disposition=disposition,
        allowed_reason=_is_scarcity_transition_reason,
    )


def _effective_report_passed(
    report: _RawScientificReport,
    *,
    n_states: int,
    disposition: OwnerFixedK4Disposition | None,
    allowed_reason: Callable[[str], bool],
) -> bool:
    raw_passed = bool(getattr(report, "passed", False))
    if disposition is None:
        return raw_passed
    if not isinstance(disposition, OwnerFixedK4Disposition):
        raise ModelError("effective K=4 decision received an invalid disposition")
    reasons = getattr(report, "rejection_reasons", None)
    if not isinstance(reasons, tuple) or any(
        not isinstance(reason, str) for reason in reasons
    ):
        raise ModelError("effective K=4 decision requires typed raw rejection reasons")
    if raw_passed:
        if reasons:
            raise ModelError("scientific report pass flag contradicts its reasons")
        return True
    if n_states != 4 or disposition.n_states != 4:
        raise ModelError("effective K=4 decision has an inconsistent state count")
    return bool(reasons) and all(allowed_reason(reason) for reason in reasons)


def _report_is_disposition_eligible(
    report: _RawScientificReport,
    *,
    allowed_reason: Callable[[str], bool],
) -> bool:
    if report.passed:
        return not report.rejection_reasons
    return bool(report.rejection_reasons) and all(
        allowed_reason(reason) for reason in report.rejection_reasons
    )


def _is_effective_mass_reason(reason: str) -> bool:
    return (
        isinstance(reason, str) and _EFFECTIVE_MASS_REASON.fullmatch(reason) is not None
    )


def _is_scarcity_transition_reason(reason: str) -> bool:
    return isinstance(reason, str) and (
        reason in _SCARCITY_TRANSITION_LITERALS
        or _SCARCITY_TRANSITION_REASON.fullmatch(reason) is not None
    )


def _four_nonnegative_counts(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, int, int, int]:
    supplied = tuple(values)
    if len(supplied) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in supplied
    ):
        raise ModelError(f"owner-fixed K=4 {label} must contain four counts")
    return (supplied[0], supplied[1], supplied[2], supplied[3])
