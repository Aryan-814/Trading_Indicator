"""Seed data/live/indicator.db with warm start candles, signals, baselines, and outcomes."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

# Add repo root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.live.collector import (
    backfill_current_session,
    warm_start_from_parquet,
)
from src.live.db import (
    checkpoint_and_close,
    init_db,
)
from src.live.predictor import LivePredictor
from src.live.scorer import LiveScorer, compute_running_stats
from src.models import load_bundle


def seed_database(
    db_path: str = "data/live/indicator.db",
    parquet_path: str = "data/clean/aapl_5m_clean.parquet",
    bundle_path: str = "models/v1",
    days: int = 35,
) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        db_file.unlink()
    for shm_or_wal in (f"{db_path}-shm", f"{db_path}-wal"):
        p = Path(shm_or_wal)
        if p.exists():
            p.unlink()

    print(f"Initializing {db_path}...")
    conn = init_db(db_file)

    print(f"Warm starting {days} trading days for AAPL, SPY, and QQQ from {parquet_path}...")
    n_candles = warm_start_from_parquet(
        conn=conn,
        parquet_path=parquet_path,
        symbols=("AAPL", "SPY", "QQQ"),
        days=days,
    )
    print(f"Inserted {n_candles} historical candles.")

    bundle = load_bundle(bundle_path)
    scorer = LiveScorer(conn=conn, horizon_bars=int(bundle["thr"]["horizon_bars"]), cost_bps_per_side=1.0)
    predictor = LivePredictor(conn=conn, bundle=bundle, symbol="AAPL")

    print("Backfilling session signals and resolving due outcomes...")
    n_signals = backfill_current_session(
        conn=conn,
        predictor=predictor,
        scorer=scorer,
        symbol="AAPL",
    )
    print(f"Generated {n_signals} session signals.")

    stats = compute_running_stats(conn, horizon_bars=int(bundle["thr"]["horizon_bars"]))
    checkpoint_and_close(conn)

    print("=== Database Seed Complete ===")
    print(f"Total signals generated: {stats['total_signals']}")
    print(f"Trade signals emitted:   {stats['trade_signals']}")
    print(f"Resolved outcomes:       {stats['resolved_trades']}")


if __name__ == "__main__":
    seed_database()
