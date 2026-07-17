from __future__ import annotations

import pytest

from prpg.data.calendar import (
    CALENDAR_ID,
    SESSIONS_PER_YEAR,
    WEEK_BOUNDARIES,
    SC25252Calendar,
    week_boundaries,
)


def test_calendar_identity_and_canonical_counts() -> None:
    calendar = SC25252Calendar(50)

    assert calendar.calendar_id == CALENDAR_ID == "SC252-52-v1"
    assert calendar.counts.years == 50
    assert calendar.counts.monthly == 600
    assert calendar.counts.quarterly == 200
    assert calendar.counts.annual == 50
    assert calendar.counts.daily == 12_600
    assert calendar.counts.weekly == 2_600


def test_week_boundaries_follow_exact_integer_formula() -> None:
    expected = tuple((252 * week) // 52 for week in range(53))

    assert week_boundaries() == expected
    assert expected == WEEK_BOUNDARIES
    assert WEEK_BOUNDARIES[0] == 0
    assert WEEK_BOUNDARIES[-1] == 252
    assert len(WEEK_BOUNDARIES) == 53


def test_weeks_cover_every_session_once_with_frozen_lengths() -> None:
    ranges = SC25252Calendar.week_ranges()
    lengths = tuple(end - start + 1 for start, end in ranges)
    covered = [session for start, end in ranges for session in range(start, end + 1)]

    assert len(ranges) == 52
    assert lengths.count(5) == 44
    assert lengths.count(4) == 8
    assert covered == list(range(1, 253))


@pytest.mark.parametrize(
    ("session", "month"),
    [(1, 1), (21, 1), (22, 2), (231, 11), (232, 12), (252, 12)],
)
def test_month_for_session_exact_boundaries(session: int, month: int) -> None:
    assert SC25252Calendar.month_for_session(session) == month


@pytest.mark.parametrize(
    ("month", "quarter"), [(1, 1), (3, 1), (4, 2), (9, 3), (10, 4), (12, 4)]
)
def test_quarter_for_month_exact_boundaries(month: int, quarter: int) -> None:
    assert SC25252Calendar.quarter_for_month(month) == quarter


def test_week_mapping_agrees_with_inclusive_ranges() -> None:
    for expected_week, (start, end) in enumerate(
        SC25252Calendar.week_ranges(), start=1
    ):
        for session in range(start, end + 1):
            assert SC25252Calendar.week_for_session(session) == expected_week


def test_period_ranges_are_half_open_partitions() -> None:
    expected_counts = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "annual": 1,
    }
    for frequency, expected_count in expected_counts.items():
        ranges = SC25252Calendar.period_ranges(frequency)  # type: ignore[arg-type]
        covered = [index for start, end in ranges for index in range(start, end)]
        assert len(ranges) == expected_count
        assert covered == list(range(SESSIONS_PER_YEAR))


@pytest.mark.parametrize("years", [0, -1])
def test_nonpositive_horizon_is_rejected(years: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        SC25252Calendar(years)


@pytest.mark.parametrize("years", [True, 2.5, "50"])
def test_noninteger_horizon_is_rejected(years: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        SC25252Calendar(years)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 253])
def test_invalid_session_is_rejected(value: int) -> None:
    with pytest.raises(ValueError):
        SC25252Calendar.month_for_session(value)


def test_unknown_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported frequency"):
        SC25252Calendar.period_ranges("hourly")  # type: ignore[arg-type]
