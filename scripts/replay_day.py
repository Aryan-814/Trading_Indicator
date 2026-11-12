"""Replay a historical trading session through the live pipeline and verify 0 train/serve skew."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Add repo root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.live.collector import (
    CandleCollector,
    format_utc_iso,
    get_nyse_session_info,
    parse_utc_dt,
)
from src.live.db import (
    get_signal_by_ts,
    init_db,
    load_candles,
    save_candle,
)
from src.live.predictor import LivePredictor, prepare_history_frame
from src.live.scorer import LiveScorer, compute_running_stats
from src.models import load_bundle, make_signal

tz_ny = ZoneInfo("America/New_York")
tz_utc = timezone.utc


def synthesize_minute_bars(candle_5m: dict) -> list[dict]:
    """Break a 5m candle into 5 realistic 1m bars that aggregate back to the exact 5m candle."""
    ts_start = parse_utc_dt(candle_5m["timestamp"] if "timestamp" in candle_5m else candle_5m["ts_start"])
    o = float(candle_5m["open"])
    h = float(candle_5m["high"])
    l = float(candle_5m["low"])
    c = float(candle_5m["close"])
    v = float(candle_5m.get("volume", 1000.0))
    vwap = float(candle_5m.get("vwap", c))
    tc = int(candle_5m.get("trade_count", 100))

    vol_per_bar = v / 5.0
    tc_per_bar = tc // 5

    # Interpolate prices across the 5 minutes
    m0_open, m0_close = o, (o + h) / 2.0
    m1_open, m1_close = m0_close, h
    m2_open, m2_close = h, l
    m3_open, m3_close = l, (l + c) / 2.0
    m4_open, m4_close = m3_close, c

    bars = [
        {"o": m0_open, "c": m0_close, "h": max(m0_open, m0_close), "l": min(m0_open, m0_close)},
        {"o": m1_open, "c": m1_close, "h": h, "l": min(m1_open, m1_close)},
        {"o": m2_open, "c": m2_close, "h": h, "l": l},
        {"o": m3_open, "c": m3_close, "h": max(m3_open, m3_close), "l": l},
        {"o": m4_open, "c": m4_close, "h": max(m4_open, m4_close), "l": min(m4_open, m4_close)},
    ]

    out_bars = []
    for idx, b in enumerate(bars):
        bar_ts = format_utc_iso(ts_start + timedelta(minutes=idx))
        out_bars.append(
            {
                "symbol": "AAPL",
                "timestamp": bar_ts,
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": vol_per_bar,
                "vwap": vwap,
                "trade_count": tc_per_bar,
            }
        )
    return out_bars


def run_replay(day: str, bundle_path: str = "models/v1", parquet_path: str = "data/clean/aapl_5m_clean.parquet") -> bool:
    """Run full session replay and verify 100% equivalence with offline make_signal."""
    print(f"=== Replaying Session: {day} ===")
    p_path = Path(parquet_path)
    if not p_path.exists():
        print(f"Error: dataset {p_path} not found")
        return False

    df = pd.read_parquet(p_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts_ny"] = df["timestamp"].dt.tz_convert(tz_ny)
    df["day_str"] = df["ts_ny"].dt.date.astype(str)

    days_avail = sorted(df["day_str"].unique())
    if day not in days_avail:
        print(f"Error: day {day} not found in {parquet_path}. Available range: {days_avail[0]} to {days_avail[-1]}")
        return False

    day_idx = days_avail.index(day)
    # Warm start using up to 25 prior days
    warm_start_days = days_avail[max(0, day_idx - 25) : day_idx]
    warm_df = df[df["day_str"].isin(warm_start_days)].copy()
    replay_df = df[df["day_str"] == day].copy()

    print(f"Warm start: {len(warm_start_days)} days ({len(warm_df)} 5m bars)")
    print(f"Replay day: {len(replay_df)} 5m bars")

    bundle = load_bundle(bundle_path)
    temp_dir = tempfile.mkdtemp(prefix="replay_indicator_")
    db_file = Path(temp_dir) / "replay.db"

    try:
        conn = init_db(db_file)

        # Seed warm start candles into SQLite
        for row in warm_df.itertuples(index=False):
            ts_str = format_utc_iso(parse_utc_dt(row.timestamp))
            c = {
                "ts_start": ts_str,
                "symbol": "AAPL",
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
                "vwap": float(getattr(row, "vwap", row.close)),
                "trade_count": int(getattr(row, "trade_count", 0)),
                "bar_count": 5,
                "source": "warm_start",
            }
            save_candle(conn, c, commit=False)
        conn.commit()

        scorer = LiveScorer(conn=conn, horizon_bars=int(bundle["thr"]["horizon_bars"]), cost_bps_per_side=1.0)
        predictor = LivePredictor(conn=conn, bundle=bundle, symbol="AAPL")

        collector = CandleCollector(
            conn=conn,
            symbols=("AAPL",),
            primary_symbol="AAPL",
            min_minutes=1,
            on_bucket_finalised=lambda b_ts, q_ok: predictor.on_bucket_closed(b_ts, quality_ok=q_ok),
            on_bucket_scored=lambda b_ts: scorer.resolve_due(b_ts),
        )

        mismatches = 0
        signals_compared = 0
        trade_signals_count = 0

        # Iterate bucket by bucket
        for row in replay_df.itertuples(index=False):
            ts_start_iso = format_utc_iso(parse_utc_dt(row.timestamp))
            candle_dict = {
                "ts_start": ts_start_iso,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "vwap": getattr(row, "vwap", row.close),
                "trade_count": getattr(row, "trade_count", 0),
            }

            # Synthesize 5 1-minute bars
            minute_bars = synthesize_minute_bars(candle_dict)
            for m_bar in minute_bars:
                collector.on_minute_bar(m_bar)

            # Finalise 5-minute bucket
            collector.finalise_bucket(ts_start_iso, source="live")

            # Load the live signal just produced
            live_sig = get_signal_by_ts(conn, ts_start_iso)
            if live_sig is None:
                print(f"Error: Live signal missing for bucket {ts_start_iso}")
                mismatches += 1
                continue

            # Compute offline make_signal
            current_candles = load_candles(conn, symbol="AAPL", limit=2400, end_ts=ts_start_iso)
            hist = prepare_history_frame(current_candles)
            offline_sig = make_signal(bundle, hist)

            # Compare fields
            signals_compared += 1
            if live_sig["direction"] in ("up", "down"):
                trade_signals_count += 1

            dir_match = live_sig["direction"] == offline_sig["direction"]
            reason_match = live_sig["reason"] == offline_sig["reason"]
            hash_match = live_sig["feature_hash"] == offline_sig["feature_hash"]

            prob_match = True
            for k in ("p_up", "meta_prob", "confidence"):
                l_val = live_sig.get(k)
                o_val = offline_sig.get(k)
                if o_val is None:
                    continue
                if abs(float(l_val) - float(o_val)) > 1e-5:
                    prob_match = False
                    break

            if not (dir_match and reason_match and hash_match and prob_match):
                print(f"MISMATCH at {ts_start_iso}:")
                print(f"  Live:    dir={live_sig['direction']}, reason={live_sig['reason']}, hash={live_sig['feature_hash']}")
                print(f"  Offline: dir={offline_sig['direction']}, reason={offline_sig['reason']}, hash={offline_sig['feature_hash']}")
                mismatches += 1

        # Flush end of session
        session_info = get_nyse_session_info(day)
        scorer.flush_session_end(session_info["open_utc"], session_info["close_utc"])

        stats = compute_running_stats(conn, horizon_bars=int(bundle["thr"]["horizon_bars"]))
        print(f"\n--- Replay Summary for {day} ---")
        print(f"Signals evaluated:       {signals_compared}")
        print(f"Trade signals emitted:   {trade_signals_count}")
        print(f"Signals resolved:        {stats['resolved_trades']}")
        print(f"Field-by-field matches:  {signals_compared - mismatches}/{signals_compared}")
        print(f"Total net PnL (bps):     {stats['cumulative_pnl_bps']:.2f}")
        if stats["accuracy"] is not None:
            print(f"Model Accuracy:          {stats['accuracy'] * 100:.1f}%")

        if mismatches == 0 and signals_compared > 0:
            print("\nSUCCESS: 100% equivalence between live pipeline and offline model bundle!")
            return True
        else:
            print(f"\nFAILURE: {mismatches} mismatches detected.")
            return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay past trading day through live backend")
    parser.add_argument("--day", type=str, default="2025-12-10", help="Trading day (YYYY-MM-DD)")
    parser.add_argument("--bundle", type=str, default="models/v1", help="Path to model bundle")
    parser.add_argument("--parquet", type=str, default="data/clean/aapl_5m_clean.parquet", help="Clean dataset")
    args = parser.parse_args()

    success = run_replay(day=args.day, bundle_path=args.bundle, parquet_path=args.parquet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

