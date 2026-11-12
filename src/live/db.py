"""SQLite persistence layer in WAL mode for the live trading indicator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


schema_sql = """
CREATE TABLE IF NOT EXISTS candles (
    ts_start TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    vwap REAL NOT NULL,
    trade_count INTEGER NOT NULL DEFAULT 0,
    bar_count INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'live',
    PRIMARY KEY (ts_start, symbol)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts
    ON candles (symbol, ts_start);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_ts TEXT NOT NULL UNIQUE,
    issued_at TEXT NOT NULL,
    horizon_bars INTEGER NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('up', 'down', 'none')),
    confidence REAL,
    p_up REAL NOT NULL,
    meta_prob REAL NOT NULL,
    reason TEXT,
    scenarios_json TEXT,
    drivers_json TEXT,
    quality_json TEXT,
    feature_hash TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS baselines (
    candle_ts TEXT NOT NULL,
    name TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('up', 'down', 'none')),
    PRIMARY KEY (candle_ts, name)
);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id INTEGER PRIMARY KEY REFERENCES signals(id),
    resolved_at TEXT NOT NULL,
    ref_price REAL NOT NULL,
    entry_price REAL,
    exit_price REAL,
    exit_ts TEXT,
    fwd_ret REAL,
    gross_bps REAL,
    correct INTEGER,
    pnl_bps REAL
);

CREATE TABLE IF NOT EXISTS health_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT
);

CREATE VIEW IF NOT EXISTS baseline_results AS
SELECT
    b.candle_ts,
    b.name,
    b.direction,
    o.entry_price,
    o.exit_price,
    o.fwd_ret,
    o.gross_bps
FROM baselines b
JOIN signals s ON s.candle_ts = b.candle_ts
JOIN outcomes o ON o.signal_id = s.id
WHERE coalesce(s.reason, '') != 'horizon_past_close';

CREATE TRIGGER IF NOT EXISTS signals_no_update
BEFORE UPDATE ON signals
BEGIN
    SELECT RAISE(ABORT, 'signals table is immutable');
END;

CREATE TRIGGER IF NOT EXISTS signals_no_delete
BEFORE DELETE ON signals
BEGIN
    SELECT RAISE(ABORT, 'signals table is immutable');
END;

CREATE TRIGGER IF NOT EXISTS baselines_no_update
BEFORE UPDATE ON baselines
BEGIN
    SELECT RAISE(ABORT, 'baselines table is immutable');
END;

CREATE TRIGGER IF NOT EXISTS baselines_no_delete
BEFORE DELETE ON baselines
BEGIN
    SELECT RAISE(ABORT, 'baselines table is immutable');
END;
"""


def utc_now_iso() -> str:
    """Return current UTC time formatted as an ISO-8601 Z string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for WAL mode and row dict access."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialise the database schema, views, and immutability triggers."""
    conn = connect_db(db_path)
    conn.executescript(schema_sql)
    conn.commit()
    return conn


def checkpoint_and_close(conn: sqlite3.Connection) -> None:
    """Flush the WAL into the main database file and close the connection."""
    try:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()


def backup_db(source_path: str | Path, backup_dir: str | Path) -> Path:
    """Create a WAL-consistent SQLite backup snapshot using sqlite3.backup."""
    src_path = Path(source_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Database not found for backup: {src_path}")

    dst_dir = Path(backup_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    dst_path = dst_dir / f"indicator_{stamp}.db"
    suffix = 1
    while dst_path.exists():
        dst_path = dst_dir / f"indicator_{stamp}_{suffix}.db"
        suffix += 1

    src_conn = sqlite3.connect(str(src_path))
    dst_conn = sqlite3.connect(str(dst_path))
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()
    return dst_path


async def run_backup_async(source_path: str | Path, backup_dir: str | Path) -> Path:
    """Run SQLite online backup in a worker thread to avoid blocking the event loop."""
    return await asyncio.to_thread(backup_db, source_path, backup_dir)


def save_candle(conn: sqlite3.Connection, candle: dict[str, Any], commit: bool = True) -> bool:
    """Insert a finalised 5-minute candle idempotently; returns True if inserted."""
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO candles (
            ts_start, symbol, open, high, low, close, volume, vwap,
            trade_count, bar_count, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(candle["ts_start"]),
            str(candle["symbol"]),
            float(candle["open"]),
            float(candle["high"]),
            float(candle["low"]),
            float(candle["close"]),
            float(candle["volume"]),
            float(candle["vwap"]),
            int(candle.get("trade_count", 0)),
            int(candle.get("bar_count", 1)),
            str(candle.get("source", "live")),
        ),
    )
    if commit:
        conn.commit()
    return cursor.rowcount > 0


def insert_signal(conn: sqlite3.Connection, signal: dict[str, Any], commit: bool = True) -> int | None:
    """Insert an immutable signal row once per candle_ts; returns id if inserted."""
    scenarios_json = signal.get("scenarios_json")
    if scenarios_json is None and "scenarios" in signal:
        scenarios_json = json.dumps(signal["scenarios"])

    drivers_json = signal.get("drivers_json")
    if drivers_json is None and "drivers" in signal:
        drivers_json = json.dumps(signal["drivers"])

    quality_json = signal.get("quality_json")
    if quality_json is None and "quality" in signal:
        quality_json = json.dumps(signal["quality"])

    conf = signal.get("confidence")
    conf_val = float(conf) if conf is not None else None

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO signals (
            candle_ts, issued_at, horizon_bars, direction, confidence,
            p_up, meta_prob, reason, scenarios_json, drivers_json,
            quality_json, feature_hash, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(signal["candle_ts"]),
            str(signal.get("issued_at") or utc_now_iso()),
            int(signal["horizon_bars"]),
            str(signal["direction"]),
            conf_val,
            float(signal["p_up"]),
            float(signal["meta_prob"]),
            signal.get("reason"),
            scenarios_json,
            drivers_json,
            quality_json,
            signal.get("feature_hash"),
            signal.get("model_version"),
        ),
    )
    if commit:
        conn.commit()
    if cursor.rowcount == 0:
        return None
    return int(cursor.lastrowid)


def insert_baseline(
    conn: sqlite3.Connection,
    candle_ts: str,
    name: str,
    direction: str,
    commit: bool = True,
) -> bool:
    """Insert an immutable baseline prediction idempotently."""
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO baselines (candle_ts, name, direction)
        VALUES (?, ?, ?)
        """,
        (str(candle_ts), str(name), str(direction)),
    )
    if commit:
        conn.commit()
    return cursor.rowcount > 0


def save_outcome(conn: sqlite3.Connection, outcome: dict[str, Any], commit: bool = True) -> bool:
    """Insert a resolved outcome for a signal idempotently."""
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO outcomes (
            signal_id, resolved_at, ref_price, entry_price, exit_price,
            exit_ts, fwd_ret, gross_bps, correct, pnl_bps
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(outcome["signal_id"]),
            str(outcome.get("resolved_at") or utc_now_iso()),
            float(outcome["ref_price"]),
            None if outcome.get("entry_price") is None else float(outcome["entry_price"]),
            None if outcome.get("exit_price") is None else float(outcome["exit_price"]),
            outcome.get("exit_ts"),
            None if outcome.get("fwd_ret") is None else float(outcome["fwd_ret"]),
            None if outcome.get("gross_bps") is None else float(outcome["gross_bps"]),
            None if outcome.get("correct") is None else int(outcome["correct"]),
            None if outcome.get("pnl_bps") is None else float(outcome["pnl_bps"]),
        ),
    )
    if commit:
        conn.commit()
    return cursor.rowcount > 0


def log_health_event(
    conn: sqlite3.Connection,
    event_type: str,
    detail: str = "",
    ts: str | None = None,
    commit: bool = True,
) -> int:
    """Record a system or data quality event into health_events."""
    cursor = conn.execute(
        "INSERT INTO health_events (ts, event_type, detail) VALUES (?, ?, ?)",
        (ts or utc_now_iso(), str(event_type), str(detail)),
    )
    if commit:
        conn.commit()
    return int(cursor.lastrowid)


def get_candle(conn: sqlite3.Connection, ts_start: str, symbol: str = "AAPL") -> dict[str, Any] | None:
    """Fetch a single finalised candle by timestamp and symbol."""
    row = conn.execute(
        "SELECT * FROM candles WHERE ts_start = ? AND symbol = ?",
        (str(ts_start), str(symbol)),
    ).fetchone()
    return dict(row) if row is not None else None


def load_candles(
    conn: sqlite3.Connection,
    symbol: str = "AAPL",
    limit: int | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> list[dict[str, Any]]:
    """Load finalised candles ordered ascending by ts_start (last `limit` if given)."""
    clauses = ["symbol = ?"]
    params: list[Any] = [str(symbol)]
    if start_ts is not None:
        clauses.append("ts_start >= ?")
        params.append(str(start_ts))
    if end_ts is not None:
        clauses.append("ts_start <= ?")
        params.append(str(end_ts))

    where_sql = " AND ".join(clauses)
    if limit is not None:
        rows = conn.execute(
            f"""
            SELECT * FROM (
                SELECT * FROM candles
                WHERE {where_sql}
                ORDER BY ts_start DESC
                LIMIT ?
            ) sub
            ORDER BY ts_start ASC
            """,
            (*params, int(limit)),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM candles WHERE {where_sql} ORDER BY ts_start ASC",
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def get_finalised_buckets(
    conn: sqlite3.Connection,
    symbol: str = "AAPL",
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> set[str]:
    """Return the set of candle ts_start strings persisted for a symbol."""
    clauses = ["symbol = ?"]
    params: list[Any] = [str(symbol)]
    if start_ts is not None:
        clauses.append("ts_start >= ?")
        params.append(str(start_ts))
    if end_ts is not None:
        clauses.append("ts_start <= ?")
        params.append(str(end_ts))
    rows = conn.execute(
        f"SELECT ts_start FROM candles WHERE {' AND '.join(clauses)}",
        tuple(params),
    ).fetchall()
    return {str(row["ts_start"]) for row in rows}


def get_signal_by_ts(conn: sqlite3.Connection, candle_ts: str) -> dict[str, Any] | None:
    """Fetch a signal joined with any resolved outcome and baselines for candle_ts."""
    row = conn.execute(
        """
        SELECT
            s.*,
            o.resolved_at,
            o.ref_price,
            o.entry_price,
            o.exit_price,
            o.exit_ts,
            o.fwd_ret,
            o.gross_bps,
            o.correct,
            o.pnl_bps
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        WHERE s.candle_ts = ?
        """,
        (str(candle_ts),),
    ).fetchone()
    if row is None:
        return None
    item = _decode_signal_row(dict(row))
    baseline_rows = conn.execute(
        "SELECT name, direction FROM baselines WHERE candle_ts = ? ORDER BY name ASC",
        (str(candle_ts),),
    ).fetchall()
    item["baselines"] = {str(b["name"]): str(b["direction"]) for b in baseline_rows}
    return item


def query_unresolved_signals(conn: sqlite3.Connection, cutoff_ts: str) -> list[dict[str, Any]]:
    """Return eligible unresolved signals at or before cutoff_ts."""
    rows = conn.execute(
        """
        SELECT s.id, s.candle_ts, s.horizon_bars, s.direction, s.reason
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        WHERE o.signal_id IS NULL
          AND coalesce(s.reason, '') NOT IN ('horizon_past_close', 'data_quality', 'missed')
          AND s.candle_ts <= ?
        ORDER BY s.candle_ts ASC
        """,
        (str(cutoff_ts),),
    ).fetchall()
    return [dict(row) for row in rows]


def query_signals_with_outcomes(
    conn: sqlite3.Connection,
    limit: int = 50,
    offset: int = 0,
    day: str | None = None,
    direction: str | None = None,
) -> list[dict[str, Any]]:
    """Query signals joined with outcomes ordered newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if day is not None:
        clauses.append("substr(s.candle_ts, 1, 10) = ?")
        params.append(str(day))
    if direction is not None:
        clauses.append("s.direction = ?")
        params.append(str(direction))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT
            s.*,
            o.resolved_at,
            o.ref_price,
            o.entry_price,
            o.exit_price,
            o.exit_ts,
            o.fwd_ret,
            o.gross_bps,
            o.correct,
            o.pnl_bps
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        {where_sql}
        ORDER BY s.candle_ts DESC
        LIMIT ? OFFSET ?
        """,
        (*params, int(limit), int(offset)),
    ).fetchall()
    return [_decode_signal_row(dict(row)) for row in rows]


def query_recent_health_events(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Return most recent health_events ordered newest first."""
    rows = conn.execute(
        "SELECT id, ts, event_type, detail FROM health_events ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    return [dict(row) for row in rows]


def _decode_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON columns on a signal record when present."""
    raw_scenarios = row.pop("scenarios_json", None)
    raw_drivers = row.pop("drivers_json", None)
    raw_quality = row.pop("quality_json", None)
    row["scenarios"] = json.loads(raw_scenarios) if raw_scenarios else None
    row["drivers"] = json.loads(raw_drivers) if raw_drivers else []
    row["quality"] = json.loads(raw_quality) if raw_quality else None
    return row
