"""Tests for WAL-consistent SQLite backup snapshots."""

import asyncio
from pathlib import Path
import sqlite3

from src.live.db import (
    backup_db,
    checkpoint_and_close,
    init_db,
    insert_signal,
    run_backup_async,
    save_candle,
)


def test_sqlite_online_backup_consistency(tmp_path: Path):
    db_file = tmp_path / "indicator.db"
    backup_dir = tmp_path / "backups"

    conn = init_db(db_file)
    try:
        save_candle(
            conn,
            {
                "ts_start": "2026-11-12T15:00:00Z",
                "symbol": "AAPL",
                "open": 150.0,
                "high": 151.0,
                "low": 149.5,
                "close": 150.5,
                "volume": 2000.0,
                "vwap": 150.3,
                "trade_count": 50,
            },
        )
        insert_signal(
            conn,
            {
                "candle_ts": "2026-11-12T15:00:00Z",
                "horizon_bars": 12,
                "direction": "up",
                "confidence": 0.65,
                "p_up": 0.62,
                "meta_prob": 0.58,
                "reason": None,
            },
        )

        # Perform online backup while WAL connection is active
        snap_path = backup_db(db_file, backup_dir)
        assert snap_path.exists()

        async_snap = asyncio.run(run_backup_async(db_file, backup_dir))
        assert async_snap.exists()
        assert async_snap != snap_path

        # Inspect restored snapshot contents independently
        snap_conn = sqlite3.connect(str(async_snap))
        try:
            candle_count = snap_conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
            signal_count = snap_conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            assert candle_count == 1
            assert signal_count == 1
        finally:
            snap_conn.close()
    finally:
        checkpoint_and_close(conn)

