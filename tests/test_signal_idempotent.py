"""Tests ensuring repeated processing of the same bucket is strictly idempotent."""

from pathlib import Path

from src.live.db import checkpoint_and_close, init_db, save_candle
from src.live.predictor import LivePredictor


def dummy_model(bucket_ts, hist):
    return {
        "candle_ts": bucket_ts,
        "horizon_bars": 12,
        "direction": "down",
        "confidence": 0.61,
        "p_up": 0.35,
        "meta_prob": 0.59,
        "reason": None,
        "scenarios": {"bear_q10": -0.01, "base_q50": -0.002, "bull_q90": 0.005},
        "drivers": [],
        "feature_hash": "abc123",
    }


def test_signal_generation_idempotent(tmp_path: Path):
    conn = init_db(tmp_path / "test_idemp.db")
    try:
        bucket_ts = "2026-11-12T15:00:00Z"
        save_candle(
            conn,
            {
                "ts_start": bucket_ts,
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "vwap": 100.2,
                "trade_count": 20,
            },
        )

        predictor = LivePredictor(conn=conn, custom_predict_fn=dummy_model, symbol="AAPL")

        first = predictor.on_bucket_closed(bucket_ts, quality_ok=True)
        second = predictor.on_bucket_closed(bucket_ts, quality_ok=True)
        third = predictor.write_special_signal(bucket_ts, reason="missed")

        assert first is not None
        assert second is None
        assert third is None

        sig_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        base_count = conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0]
        assert sig_count == 1
        assert base_count == 2  # always_up + rsi_rule
    finally:
        checkpoint_and_close(conn)

