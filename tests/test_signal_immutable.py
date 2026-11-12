"""Tests enforcing SQLite immutability triggers on signals and baselines."""

from pathlib import Path
import sqlite3

import pytest

from src.live.db import checkpoint_and_close, init_db, insert_baseline, insert_signal


def test_signals_and_baselines_immutability_triggers(tmp_path: Path):
    conn = init_db(tmp_path / "test_immut.db")
    try:
        sig_id = insert_signal(
            conn,
            {
                "candle_ts": "2026-11-12T15:00:00Z",
                "horizon_bars": 12,
                "direction": "up",
                "confidence": 0.65,
                "p_up": 0.60,
                "meta_prob": 0.58,
                "reason": None,
            },
        )
        assert sig_id is not None
        assert insert_baseline(conn, "2026-11-12T15:00:00Z", "always_up", "up") is True

        # Trigger prevents UPDATE on signals
        with pytest.raises(sqlite3.DatabaseError, match="signals table is immutable"):
            conn.execute("UPDATE signals SET direction = 'down' WHERE id = ?", (sig_id,))
        conn.rollback()

        # Trigger prevents DELETE on signals
        with pytest.raises(sqlite3.DatabaseError, match="signals table is immutable"):
            conn.execute("DELETE FROM signals WHERE id = ?", (sig_id,))
        conn.rollback()

        # Trigger prevents UPDATE on baselines
        with pytest.raises(sqlite3.DatabaseError, match="baselines table is immutable"):
            conn.execute(
                "UPDATE baselines SET direction = 'none' WHERE candle_ts = ? AND name = 'always_up'",
                ("2026-11-12T15:00:00Z",),
            )
        conn.rollback()

        # Trigger prevents DELETE on baselines
        with pytest.raises(sqlite3.DatabaseError, match="baselines table is immutable"):
            conn.execute(
                "DELETE FROM baselines WHERE candle_ts = ?",
                ("2026-11-12T15:00:00Z",),
            )
        conn.rollback()
    finally:
        checkpoint_and_close(conn)
