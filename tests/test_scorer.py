"""Tests for LiveScorer outcome resolution, missing candle fallbacks, and baseline stats."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.live.db import (
    checkpoint_and_close,
    init_db,
    insert_baseline,
    insert_signal,
    query_recent_health_events,
    save_candle,
)
from src.live.scorer import LiveScorer, compute_running_stats


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_scorer_resolve_due_and_pnl(tmp_path: Path):
    conn = init_db(tmp_path / "test_scorer.db")
    try:
        t0 = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
        h = 12

        # Save candle t0 (ref), t1 (entry), and t+12 (exit)
        save_candle(conn, {"ts_start": _iso(t0), "symbol": "AAPL", "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000, "vwap": 100.0})
        save_candle(conn, {"ts_start": _iso(t0 + timedelta(minutes=5)), "symbol": "AAPL", "open": 100.0, "high": 101.0, "low": 99.8, "close": 100.5, "volume": 1000, "vwap": 100.4})
        save_candle(conn, {"ts_start": _iso(t0 + timedelta(minutes=h * 5)), "symbol": "AAPL", "open": 101.5, "high": 102.5, "low": 101.0, "close": 102.0, "volume": 1000, "vwap": 101.8})

        sig_id = insert_signal(
            conn,
            {
                "candle_ts": _iso(t0),
                "horizon_bars": h,
                "direction": "up",
                "confidence": 0.70,
                "p_up": 0.65,
                "meta_prob": 0.60,
                "reason": None,
            },
        )
        assert sig_id is not None
        insert_baseline(conn, _iso(t0), "always_up", "up")
        insert_baseline(conn, _iso(t0), "rsi_rule", "none")

        scorer = LiveScorer(conn=conn, horizon_bars=h, cost_bps_per_side=1.0)
        # Before horizon arrives, nothing resolves
        assert scorer.resolve_due(_iso(t0 + timedelta(minutes=(h - 1) * 5))) == []

        # At exact exit bucket, signal resolves
        resolved = scorer.resolve_due(_iso(t0 + timedelta(minutes=h * 5)))
        assert len(resolved) == 1
        out = resolved[0]
        assert out["entry_price"] == 100.0
        assert out["exit_price"] == 102.0
        assert abs(out["fwd_ret"] - 0.02) < 1e-6
        assert abs(out["gross_bps"] - 200.0) < 1e-4
        assert out["correct"] == 1
        assert abs(out["pnl_bps"] - 198.0) < 1e-4  # 200 - 2*1.0

        stats = compute_running_stats(conn, horizon_bars=h, cost_bps=1.0)
        assert stats["resolved_trades"] == 1
        assert stats["accuracy"] == 1.0
        assert abs(stats["cumulative_pnl_bps"] - 198.0) < 1e-4
        assert stats["baselines"]["always_up"]["same_candle"]["count"] == 1
    finally:
        checkpoint_and_close(conn)


def test_scorer_missing_entry_and_late_exit_fallbacks(tmp_path: Path):
    conn = init_db(tmp_path / "test_scorer_fb.db")
    try:
        t0 = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)
        h = 12
        # Ref candle at t0 exists, t+1 is MISSING, t+12 is MISSING, t+13 exists
        save_candle(conn, {"ts_start": _iso(t0), "symbol": "AAPL", "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.0, "volume": 1000, "vwap": 200.0})
        late_exit_dt = t0 + timedelta(minutes=(h + 1) * 5)
        save_candle(conn, {"ts_start": _iso(late_exit_dt), "symbol": "AAPL", "open": 198.0, "high": 199.0, "low": 197.0, "close": 198.0, "volume": 1000, "vwap": 198.0})

        insert_signal(
            conn,
            {
                "candle_ts": _iso(t0),
                "horizon_bars": h,
                "direction": "down",
                "confidence": 0.65,
                "p_up": 0.35,
                "meta_prob": 0.58,
                "reason": None,
            },
        )
        # Also test that horizon_past_close is excluded from baseline_results view and outcomes
        t_late = datetime(2026, 11, 12, 20, 30, tzinfo=timezone.utc)
        save_candle(conn, {"ts_start": _iso(t_late), "symbol": "AAPL", "open": 198.0, "high": 199.0, "low": 197.0, "close": 198.0, "volume": 500, "vwap": 198.0})
        insert_signal(
            conn,
            {
                "candle_ts": _iso(t_late),
                "horizon_bars": h,
                "direction": "none",
                "confidence": 0.0,
                "p_up": 0.5,
                "meta_prob": 0.0,
                "reason": "horizon_past_close",
            },
        )
        insert_baseline(conn, _iso(t_late), "always_up", "none")

        scorer = LiveScorer(conn=conn, horizon_bars=h, cost_bps_per_side=1.0)
        resolved = scorer.resolve_due(_iso(late_exit_dt))
        assert len(resolved) == 1
        out = resolved[0]
        assert out["entry_price"] == 200.0  # fell back to close of t0
        assert out["exit_price"] == 198.0   # fell back to late exit t+13
        assert out["exit_ts"] == _iso(late_exit_dt)
        assert out["correct"] == 1          # predicted down, price fell -100 bps

        events = query_recent_health_events(conn, limit=10)
        ev_types = {e["event_type"] for e in events}
        assert "missing_entry_candle" in ev_types
        assert "late_exit_candle" in ev_types

        # Confirm view excludes horizon_past_close
        view_rows = conn.execute("SELECT * FROM baseline_results").fetchall()
        assert all(r["candle_ts"] != _iso(t_late) for r in view_rows)
    finally:
        checkpoint_and_close(conn)

