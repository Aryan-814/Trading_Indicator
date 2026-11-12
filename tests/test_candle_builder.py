"""Tests for 1-minute bar aggregation, idempotence, and late bar rejection."""

from pathlib import Path

from src.live.collector import CandleCollector, floor_to_5_minutes
from src.live.db import checkpoint_and_close, get_candle, init_db, query_recent_health_events


def test_floor_to_5_minutes():
    assert floor_to_5_minutes("2026-11-12T14:30:00Z") == "2026-11-12T14:30:00Z"
    assert floor_to_5_minutes("2026-11-12T14:34:59Z") == "2026-11-12T14:30:00Z"
    assert floor_to_5_minutes("2026-11-12T14:35:00Z") == "2026-11-12T14:35:00Z"


def test_candle_aggregation_and_overwrite_idempotence(tmp_path: Path):
    conn = init_db(tmp_path / "test.db")
    try:
        collector = CandleCollector(conn=conn, symbols=("AAPL",), primary_symbol="AAPL")

        bars = [
            {"symbol": "AAPL", "timestamp": "2026-11-12T14:30:00Z", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000, "vwap": 100.2, "trade_count": 10},
            {"symbol": "AAPL", "timestamp": "2026-11-12T14:31:00Z", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 2000, "vwap": 101.0, "trade_count": 20},
            {"symbol": "AAPL", "timestamp": "2026-11-12T14:32:00Z", "open": 101.5, "high": 101.8, "low": 99.0, "close": 101.0, "volume": 1000, "vwap": 100.5, "trade_count": 15},
        ]

        for b in bars:
            collector.on_minute_bar(b)

        # Duplicate bar arrives for 14:31 with corrected volume and high -> must overwrite, not add!
        dup_bar = {"symbol": "AAPL", "timestamp": "2026-11-12T14:31:00Z", "open": 100.5, "high": 103.0, "low": 100.0, "close": 101.5, "volume": 3000, "vwap": 101.5, "trade_count": 25}
        collector.on_minute_bar(dup_bar)

        res = collector.finalise_bucket("2026-11-12T14:30:00Z")
        candle = res["AAPL"]
        assert candle is not None
        assert candle["open"] == 100.0
        assert candle["high"] == 103.0
        assert candle["low"] == 99.0
        assert candle["close"] == 101.0
        assert candle["volume"] == 5000.0  # 1000 + 3000 + 1000 (not 7000)
        assert candle["trade_count"] == 50  # 10 + 25 + 15
        assert candle["bar_count"] == 3

        stored = get_candle(conn, "2026-11-12T14:30:00Z", symbol="AAPL")
        assert stored is not None
        assert stored["volume"] == 5000.0
    finally:
        checkpoint_and_close(conn)


def test_out_of_order_minutes_and_zero_volume(tmp_path: Path):
    conn = init_db(tmp_path / "test_ooo.db")
    try:
        collector = CandleCollector(conn=conn, symbols=("AAPL",), primary_symbol="AAPL")
        # Deliver minute 4 before minute 0, with zero volume on minute 4
        collector.on_minute_bar({"symbol": "AAPL", "timestamp": "2026-11-12T14:34:00Z", "open": 105.0, "high": 106.0, "low": 104.0, "close": 105.5, "volume": 0.0, "vwap": 105.5, "trade_count": 0})
        collector.on_minute_bar({"symbol": "AAPL", "timestamp": "2026-11-12T14:30:00Z", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 0.0, "vwap": 100.5, "trade_count": 0})

        res = collector.finalise_bucket("2026-11-12T14:30:00Z")
        c = res["AAPL"]
        assert c is not None
        assert c["open"] == 100.0  # sorted by timestamp: 14:30 open, not 14:34 open
        assert c["close"] == 105.5  # 14:34 close
        assert c["vwap"] == 105.5   # falls back cleanly to close when volume == 0
    finally:
        checkpoint_and_close(conn)


def test_late_minute_bar_rejected(tmp_path: Path):
    conn = init_db(tmp_path / "test_late.db")
    try:
        collector = CandleCollector(conn=conn, symbols=("AAPL",), primary_symbol="AAPL")
        collector.on_minute_bar({"symbol": "AAPL", "timestamp": "2026-11-12T14:30:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 500, "trade_count": 5})
        collector.finalise_bucket("2026-11-12T14:30:00Z")

        # Late minute bar arrives after finalisation
        res = collector.on_minute_bar({"symbol": "AAPL", "timestamp": "2026-11-12T14:32:00Z", "open": 100.5, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 9000, "trade_count": 90})
        assert res is None

        stored = get_candle(conn, "2026-11-12T14:30:00Z", symbol="AAPL")
        assert stored is not None
        assert stored["volume"] == 500.0  # unchanged

        events = query_recent_health_events(conn, limit=5)
        assert any(e["event_type"] == "late_minute_bar" for e in events)
    finally:
        checkpoint_and_close(conn)

