"""Pure ordinal calendar contract for PRPG simulation outputs."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

CALENDAR_ID = "SC252-52-v1"
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4
WEEKS_PER_YEAR = 52
SESSIONS_PER_YEAR = 252
SESSIONS_PER_MONTH = 21

Frequency = Literal["daily", "weekly", "monthly", "quarterly", "annual"]


def week_boundaries() -> tuple[int, ...]:
    """Return the 53 frozen cumulative session boundaries for one model year."""

    return tuple((SESSIONS_PER_YEAR * week) // WEEKS_PER_YEAR for week in range(53))


WEEK_BOUNDARIES = week_boundaries()


@dataclass(frozen=True, slots=True)
class PeriodCounts:
    """Exact frequency counts for a positive integer model horizon."""

    years: int
    annual: int
    quarterly: int
    monthly: int
    weekly: int
    daily: int


@dataclass(frozen=True, slots=True)
class SC25252Calendar:
    """Versioned ordinal calendar with 252 sessions and 52 weeks per year."""

    years: int

    def __post_init__(self) -> None:
        if isinstance(self.years, bool) or not isinstance(self.years, int):
            raise TypeError("years must be an integer")
        if self.years <= 0:
            raise ValueError("years must be positive")

    @property
    def calendar_id(self) -> str:
        return CALENDAR_ID

    @property
    def counts(self) -> PeriodCounts:
        """Return exact period counts for this horizon."""

        return PeriodCounts(
            years=self.years,
            annual=self.years,
            quarterly=QUARTERS_PER_YEAR * self.years,
            monthly=MONTHS_PER_YEAR * self.years,
            weekly=WEEKS_PER_YEAR * self.years,
            daily=SESSIONS_PER_YEAR * self.years,
        )

    @staticmethod
    def month_for_session(session_in_year: int) -> int:
        """Map one-based model-year session to its one-based model month."""

        _require_index(session_in_year, SESSIONS_PER_YEAR, "session_in_year")
        return ((session_in_year - 1) // SESSIONS_PER_MONTH) + 1

    @staticmethod
    def quarter_for_month(month_in_year: int) -> int:
        """Map one-based model month to its one-based quarter."""

        _require_index(month_in_year, MONTHS_PER_YEAR, "month_in_year")
        return ((month_in_year - 1) // 3) + 1

    @staticmethod
    def week_for_session(session_in_year: int) -> int:
        """Map one-based session to its exact boundary-defined model week."""

        _require_index(session_in_year, SESSIONS_PER_YEAR, "session_in_year")
        return bisect_left(WEEK_BOUNDARIES[1:], session_in_year) + 1

    @staticmethod
    def week_ranges() -> tuple[tuple[int, int], ...]:
        """Return 52 inclusive one-based session ranges for one model year."""

        return tuple(
            (WEEK_BOUNDARIES[index] + 1, WEEK_BOUNDARIES[index + 1])
            for index in range(WEEKS_PER_YEAR)
        )

    @staticmethod
    def period_ranges(frequency: Frequency) -> tuple[tuple[int, int], ...]:
        """Return zero-based half-open base-session bins for one model year."""

        if frequency == "daily":
            return tuple((index, index + 1) for index in range(SESSIONS_PER_YEAR))
        if frequency == "weekly":
            return tuple(pairwise(WEEK_BOUNDARIES))
        if frequency == "monthly":
            return tuple(
                (index * SESSIONS_PER_MONTH, (index + 1) * SESSIONS_PER_MONTH)
                for index in range(MONTHS_PER_YEAR)
            )
        if frequency == "quarterly":
            sessions_per_quarter = SESSIONS_PER_YEAR // QUARTERS_PER_YEAR
            return tuple(
                (index * sessions_per_quarter, (index + 1) * sessions_per_quarter)
                for index in range(QUARTERS_PER_YEAR)
            )
        if frequency == "annual":
            return ((0, SESSIONS_PER_YEAR),)
        raise ValueError(f"unsupported frequency: {frequency!r}")


def _require_index(value: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
