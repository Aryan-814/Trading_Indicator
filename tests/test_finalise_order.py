"""Tests for deterministic finalisation order and missing-candle data quality rules."""

from pathlib import Path

from src.live.collector import CandleCollector
from src.live.db import checkpoint_and_close, get_signal_by_ts, init_db, query_recent_health_events
from src.live.predictor import LivePredictor


def dummy_model(bucket_ts, hist):
    return {
        "candle_ts": bucket_ts,
        "horizon_bars": 12,
        "direction": "up",
        "confidence": 0.65,
        "p_up": 0.62,
        "meta_prob": 0.60,
        "reason": None,
        "scenarios": {"bear_q10": -0.005, "base_q50": 0.002, "bull_q90": 0.010},
        "drivers": [{"feature": "ret_1", "contribution": 0.1, "value": 0.001}],
        "feature_hash": "dummy_hash",
    }


def test_deterministic_finalise_order_spy_qqq_aapl_predictor_scorer(tmp_path: Path):
    conn = init_db(tmp_path / "test_order.db")
    try:
        call_order = []

        predictor = LivePredictor(conn=conn, custom_predict_fn=dummy_model, symbol="AAPL")

        def on_saved(candle):
            call_order.append(f"saved:{candle['symbol']}")

        def on_finalised(bucket_ts, quality_ok):
            call_order.append(f"predictor:{quality_ok}")
            predictor.on_bucket_closed(bucket_ts, quality_ok=quality_ok)

        def on_scored(bucket_ts):
            call_order.append(f"scorer:{bucket_ts}")

        collector = CandleCollector(
            conn=conn,
            symbols=("SPY", "QQQ", "AAPL"),
            primary_symbol="AAPL",
            on_candle_saved=on_saved,
            on_bucket_finalised=on_finalised,
            on_bucket_scored=on_scored,
        )

        # Feed bars in reverse order (AAPL, QQQ, SPY)
        for sym in ("AAPL", "QQQ", "SPY"):
            collector.on_minute_bar(
                {"symbol": sym, "timestamp": "2026-11-12T15:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 100, "trade_count": 10}
            )

        collector.finalise_bucket("2026-11-12T15:00:00Z")

        assert call_order == [
            "saved:SPY",
            "saved:QQQ",
            "saved:AAPL",
            "predictor:True",
            "scorer:2026-11-12T15:00:00Z",
        ]

        sig = get_signal_by_ts(conn, "2026-11-12T15:00:00Z")
        assert sig is not None
        assert sig["direction"] == "up"
        assert sig["scenarios"]["horizon_bars"] == 12
        assert sig["scenarios"]["horizon_minutes"] == 60
    finally:
        checkpoint_and_close(conn)


def test_missing_aapl_candle_triggers_data_quality_signal(tmp_path: Path):
    conn = init_db(tmp_path / "test_missing_aapl.db")
    try:
        predictor = LivePredictor(conn=conn, custom_predict_fn=dummy_model, symbol="AAPL")
        collector = CandleCollector(
            conn=conn,
            symbols=("SPY", "QQQ", "AAPL"),
            primary_symbol="AAPL",
            on_bucket_finalised=lambda b, q: predictor.on_bucket_closed(b, quality_ok=q),
        )

        # Only SPY arrives, AAPL and QQQ missing
        collector.on_minute_bar(
            {"symbol": "SPY", "timestamp": "2026-11-12T15:05:00Z", "open": 500, "high": 501, "low": 499, "close": 500.5, "volume": 1000, "trade_count": 10}
        )
        collector.finalise_bucket("2026-11-12T15:05:00Z")

        events = query_recent_health_events(conn, limit=10)
        event_types = {e["event_type"] for e in events}
        assert "missing_qqq_candle" in event_types
        assert "missing_aapl_candle" in event_types

        sig = get_signal_by_ts(conn, "2026-11-12T15:05:00Z")
        assert sig is not None
        assert sig["direction"] == "none"
        assert sig["reason"] == "data_quality"
        assert sig["scenarios"] is None
        assert sig["baselines"] == {"always_up": "none", "rsi_rule": "none"}
    finally:
        checkpoint_and_close(conn)

