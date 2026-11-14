"""Seed data/live/indicator.db with live Alpaca REST warm start candles, signals, baselines, and outcomes."""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add repo root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

load_dotenv(dotenv_path=root_dir / ".env")

from src.live.collector import (
    backfill_current_session,
    warm_start,
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
    days: int = 30,
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

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")

    if key and secret:
        print(f"Pulling last {days} trading days for AAPL, SPY, and QQQ from Alpaca REST (feed=DataFeed.IEX)...")
    else:
        print(f"Alpaca credentials not found; falling back to {parquet_path}...")

    n_candles = warm_start(
        conn=conn,
        api_key=key,
        secret_key=secret,
        symbols=("AAPL", "SPY", "QQQ"),
        days=days,
        parquet_path=parquet_path,
    )
    print(f"Inserted {n_candles} historical candles.")

    bundle = load_bundle(bundle_path)
    scorer = LiveScorer(conn=conn, horizon_bars=int(bundle["thr"]["horizon_bars"]), cost_bps_per_side=1.0)
    predictor = LivePredictor(conn=conn, bundle=bundle, symbol="AAPL")

    print("Backfilling last 5 trading sessions (2026-11-09 to 2026-11-13)...")
    n_backfilled = backfill_current_session(conn, predictor=predictor, scorer=scorer, symbol="AAPL", num_sessions=5)
    print(f"Backfilled {n_backfilled} total signals across last 5 trading sessions.")

    stats = compute_running_stats(conn, horizon_bars=int(bundle["thr"]["horizon_bars"]))
    checkpoint_and_close(conn)

    print("=== Database Seed Complete ===")
    print(f"Total signals generated: {stats['total_signals']}")
    print(f"Trade signals emitted:   {stats['trade_signals']}")
    print(f"Resolved outcomes:       {stats['resolved_trades']}")


if __name__ == "__main__":
    seed_database()
