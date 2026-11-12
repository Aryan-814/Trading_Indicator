"""Tests for 2026 NYSE holiday and half-day session boundaries per Section 15.3."""

from datetime import date

from src.live.collector import get_nyse_session_info


def test_nyse_2026_calendar_fallbacks():
    # 1. Thanksgiving 2026-11-26: closed
    thx = get_nyse_session_info(date(2026, 11, 26))
    assert thx["closed"] is True
    assert thx["day_bars"] == 0

    # 2. Black Friday 2026-11-27: early close 13:00 ET (18:00 UTC), 42 bars
    bf = get_nyse_session_info(date(2026, 11, 27))
    assert bf["closed"] is False
    assert bf["is_half"] == 1
    assert bf["day_bars"] == 42
    assert bf["open_utc"] == "2026-11-27T14:30:00Z"
    assert bf["close_utc"] == "2026-11-27T18:00:00Z"

    # 3. Christmas Eve 2026-12-24: early close 13:00 ET (18:00 UTC), 42 bars
    xmas_eve = get_nyse_session_info(date(2026, 12, 24))
    assert xmas_eve["closed"] is False
    assert xmas_eve["is_half"] == 1
    assert xmas_eve["day_bars"] == 42
    assert xmas_eve["open_utc"] == "2026-12-24T14:30:00Z"
    assert xmas_eve["close_utc"] == "2026-12-24T18:00:00Z"

    # 4. Christmas Day 2026-12-25: closed
    xmas = get_nyse_session_info(date(2026, 12, 25))
    assert xmas["closed"] is True
    assert xmas["day_bars"] == 0

    # 5. Normal regular trading session (e.g. 2026-11-25 Wednesday): 78 bars
    norm = get_nyse_session_info(date(2026, 11, 25))
    assert norm["closed"] is False
    assert norm["is_half"] == 0
    assert norm["day_bars"] == 78
    assert norm["open_utc"] == "2026-11-25T14:30:00Z"
    assert norm["close_utc"] == "2026-11-25T21:00:00Z"

