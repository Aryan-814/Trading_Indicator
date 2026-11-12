"""Live signal generator wrapping the frozen Phase 2 model bundle."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
from pathlib import Path
import sqlite3
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from src.features import wilder_rsi
from src.live.collector import (
    format_utc_iso,
    get_nyse_session_info,
    parse_utc_dt,
    tz_ny,
)
from src.live.db import (
    get_candle,
    insert_baseline,
    insert_signal,
    load_candles,
    log_health_event,
    utc_now_iso,
)
from src.models import load_bundle, make_signal

logger = logging.getLogger(__name__)


def prepare_history_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw AAPL candle dicts from SQLite into a clean.py-compatible DataFrame."""
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles).copy()
    df["timestamp"] = pd.to_datetime(df["ts_start"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    df["ts"] = df["timestamp"].dt.tz_convert(tz_ny)
    df["day"] = df["ts"].dt.date.astype(str)

    # Map each day to its official NYSE session length (78 full, 42 half).
    day_meta: dict[str, dict[str, Any]] = {}
    for day_str in df["day"].unique():
        day_meta[day_str] = get_nyse_session_info(day_str)

    df["day_bars"] = [int(day_meta[d].get("day_bars") or 78) for d in df["day"]]
    df["is_half"] = [int(day_meta[d].get("is_half") or 0) for d in df["day"]]

    # Clock-based bar_idx stays accurate even if an earlier bucket in the session was missing.
    minutes_since_open = df["ts"].dt.hour * 60 + df["ts"].dt.minute - (9 * 60 + 30)
    df["bar_idx"] = (minutes_since_open // 5).astype(int)
    df["bars_left"] = df["day_bars"] - df["bar_idx"] - 1
    df["bar_end"] = df["timestamp"] + pd.Timedelta(minutes=5)

    first = df["bar_idx"] == 0
    prev_close = df["close"].shift(1)
    df["gap"] = np.where(first, np.log(df["open"] / prev_close), 0.0)
    df["gap"] = df["gap"].fillna(0.0)

    df["ret_1"] = np.where(
        first,
        np.log(df["close"] / df["open"]),
        np.log(df["close"] / prev_close),
    )

    def past_median(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(20, min_periods=10).median()

    for col in ("volume", "trade_count"):
        if col in df.columns:
            ref = df.groupby("bar_idx")[col].transform(past_median)
            df[f"rel_{col}"] = df[col] / ref
        else:
            df[f"rel_{col}"] = np.nan

    return df


def compute_rsi_direction(hist: pd.DataFrame) -> str:
    """Compute baseline RSI-14 directional rule matching offline evaluation."""
    if hist.empty or len(hist) < 15:
        return "none"
    rsi_val = wilder_rsi(hist["close"], n=14).iloc[-1]
    if pd.isna(rsi_val):
        return "none"
    if float(rsi_val) < 30.0:
        return "up"
    if float(rsi_val) > 70.0:
        return "down"
    return "none"


class LivePredictor:
    """Generates immutable signals and baseline predictions on each closed 5m bucket."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        bundle_path: str | Path | None = "models/v1",
        bundle: dict[str, Any] | None = None,
        custom_predict_fn: Callable[[str, pd.DataFrame], dict[str, Any]] | None = None,
        symbol: str = "AAPL",
        history_limit: int = 2400,
        on_signal_emitted: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> None:
        self.conn = conn
        self.symbol = symbol
        self.history_limit = history_limit
        self.custom_predict_fn = custom_predict_fn
        self.on_signal_emitted = on_signal_emitted

        if bundle is not None:
            self.bundle = bundle
        elif custom_predict_fn is None and bundle_path is not None and Path(bundle_path).exists():
            self.bundle = load_bundle(str(bundle_path))
        else:
            self.bundle = None

        if self.bundle is not None:
            thr = self.bundle.get("thr", {})
            info = self.bundle.get("info", {})
            self.horizon_bars = int(thr.get("horizon_bars", 12))
            self.model_version = str(info.get("bundle_version", info.get("version", "v1")))
        else:
            self.horizon_bars = 12
            self.model_version = "custom"
        self.horizon_minutes = self.horizon_bars * 5

    def _emit_ws(self, payload: dict[str, Any]) -> None:
        if self.on_signal_emitted is None:
            return
        res = self.on_signal_emitted(payload)
        if asyncio.iscoroutine(res):
            asyncio.create_task(res)

    def _persist_signal_and_baselines(
        self,
        signal_row: dict[str, Any],
        rsi_dir: str,
    ) -> dict[str, Any] | None:
        """Persist signal + baselines idempotently; returns full row if newly inserted."""
        signal_id = insert_signal(self.conn, signal_row, commit=False)
        if signal_id is None:
            self.conn.rollback()
            return None

        reason = signal_row.get("reason")
        excluded = reason in ("horizon_past_close", "data_quality", "missed")
        always_up_dir = "none" if excluded else "up"
        rsi_rule_dir = "none" if excluded else rsi_dir

        insert_baseline(self.conn, signal_row["candle_ts"], "always_up", always_up_dir, commit=False)
        insert_baseline(self.conn, signal_row["candle_ts"], "rsi_rule", rsi_rule_dir, commit=False)
        self.conn.commit()

        emitted = {**signal_row, "id": signal_id}
        self._emit_ws(emitted)
        return emitted

    def on_bucket_closed(
        self,
        bucket_ts: str,
        quality_ok: bool = True,
        issued_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Generate one immutable signal for bucket_ts if not already present."""
        if not quality_ok:
            return self.write_special_signal(bucket_ts, reason="data_quality", issued_at=issued_at)

        candle = get_candle(self.conn, bucket_ts, symbol=self.symbol)
        if candle is None:
            return self.write_special_signal(bucket_ts, reason="data_quality", issued_at=issued_at)

        candles = load_candles(
            self.conn,
            symbol=self.symbol,
            limit=self.history_limit,
            end_ts=bucket_ts,
        )
        hist = prepare_history_frame(candles)
        if hist.empty:
            return self.write_special_signal(bucket_ts, reason="data_quality", issued_at=issued_at)

        if self.custom_predict_fn is not None:
            raw_sig = dict(self.custom_predict_fn(bucket_ts, hist))
        else:
            if self.bundle is None:
                raise RuntimeError("No model bundle or custom_predict_fn configured")
            raw_sig = make_signal(self.bundle, hist)

        scenarios = raw_sig.get("scenarios")
        if isinstance(scenarios, dict):
            scenarios = {
                **scenarios,
                "horizon_bars": self.horizon_bars,
                "horizon_minutes": self.horizon_minutes,
            }

        signal_row = {
            "candle_ts": bucket_ts,
            "issued_at": issued_at or utc_now_iso(),
            "horizon_bars": int(raw_sig.get("horizon_bars", self.horizon_bars)),
            "direction": str(raw_sig.get("direction", "none")),
            "confidence": float(raw_sig["confidence"]) if raw_sig.get("confidence") is not None else 0.0,
            "p_up": float(raw_sig["p_up"]) if raw_sig.get("p_up") is not None else 0.5,
            "meta_prob": float(raw_sig["meta_prob"]) if raw_sig.get("meta_prob") is not None else 0.0,
            "reason": raw_sig.get("reason"),
            "scenarios": scenarios,
            "drivers": raw_sig.get("drivers") or [],
            "feature_hash": raw_sig.get("feature_hash"),
            "model_version": self.model_version,
        }
        rsi_dir = compute_rsi_direction(hist)
        return self._persist_signal_and_baselines(signal_row, rsi_dir=rsi_dir)

    def write_special_signal(
        self,
        bucket_ts: str,
        reason: str,
        issued_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Record an immutable non-trade signal (data_quality or missed)."""
        signal_row = {
            "candle_ts": bucket_ts,
            "issued_at": issued_at or utc_now_iso(),
            "horizon_bars": self.horizon_bars,
            "direction": "none",
            "confidence": 0.0,
            "p_up": 0.5,
            "meta_prob": 0.0,
            "reason": reason,
            "scenarios": None,
            "drivers": [],
            "feature_hash": None,
            "model_version": self.model_version,
        }
        return self._persist_signal_and_baselines(signal_row, rsi_dir="none")

    def mark_missed_buckets(self, bucket_list: list[str]) -> int:
        """Write 'missed' signal rows for buckets closed while collector was offline."""
        count = 0
        for bucket_ts in sorted(bucket_list):
            if self.write_special_signal(bucket_ts, reason="missed") is not None:
                count += 1
        if count > 0:
            log_health_event(
                self.conn,
                event_type="missed_signals_marked",
                detail=f"Marked {count} missed buckets as reason='missed'",
            )
        return count


def compute_session_buckets_up_to(session_open_dt: Any, cutoff_dt: Any) -> list[str]:
    """Return 5-minute UTC ISO bucket strings from session open up to cutoff_dt exclusive."""
    buckets: list[str] = []
    cur = parse_utc_dt(session_open_dt)
    end = parse_utc_dt(cutoff_dt)
    while cur < end:
        buckets.append(format_utc_iso(cur))
        cur += timedelta(minutes=5)
    return buckets

