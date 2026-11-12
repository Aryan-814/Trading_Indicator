"""Outcome resolution and running statistical performance evaluation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Awaitable, Callable

import numpy as np

from src.evaluate import expected_calibration_error, wilson_score_interval
from src.live.collector import (
    format_utc_iso,
    get_nyse_session_info,
    parse_utc_dt,
)
from src.live.db import (
    get_candle,
    load_candles,
    log_health_event,
    query_unresolved_signals,
    save_outcome,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


def compute_pnl_bps(direction: str, fwd_ret: float, cost_bps_per_side: float = 1.0) -> tuple[int | None, float]:
    """Compute correctness flag (1/0/None) and net P&L in basis points after round-trip costs."""
    gross_bps = fwd_ret * 10_000.0
    round_trip_cost = 2.0 * cost_bps_per_side

    if direction == "up":
        correct = 1 if fwd_ret > 0 else 0
        pnl = gross_bps - round_trip_cost
        return correct, pnl
    elif direction == "down":
        correct = 1 if fwd_ret < 0 else 0
        pnl = (-gross_bps) - round_trip_cost
        return correct, pnl
    else:
        return None, 0.0


class LiveScorer:
    """Resolves trade outcomes when horizon candles close and monitors live metrics."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        horizon_bars: int = 12,
        cost_bps_per_side: float = 1.0,
        symbol: str = "AAPL",
        on_outcome_resolved: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> None:
        self.conn = conn
        self.horizon_bars = int(horizon_bars)
        self.cost_bps_per_side = float(cost_bps_per_side)
        self.symbol = symbol
        self.on_outcome_resolved = on_outcome_resolved

    def _emit_ws(self, payload: dict[str, Any]) -> None:
        if self.on_outcome_resolved is None:
            return
        res = self.on_outcome_resolved(payload)
        if asyncio.iscoroutine(res):
            asyncio.create_task(res)

    def try_resolve_signal(
        self,
        signal_row: dict[str, Any],
        session_end_iso: str | None = None,
    ) -> dict[str, Any] | None:
        """Attempt to resolve entry, exit, return, and P&L for a single signal."""
        reason = signal_row.get("reason")
        if reason in ("horizon_past_close", "data_quality", "missed"):
            return None

        candle_ts_str = str(signal_row["candle_ts"])
        candle_dt = parse_utc_dt(candle_ts_str)
        h = int(signal_row.get("horizon_bars", self.horizon_bars))

        c_t = get_candle(self.conn, candle_ts_str, symbol=self.symbol)
        if c_t is None:
            return None
        ref_price = float(c_t["close"])

        # Entry candle is t + 1 (i.e. t + 5m)
        entry_ts = format_utc_iso(candle_dt + timedelta(minutes=5))
        c_entry = get_candle(self.conn, entry_ts, symbol=self.symbol)
        if c_entry is not None:
            entry_price = float(c_entry["open"])
        else:
            entry_price = ref_price
            log_health_event(
                self.conn,
                event_type="missing_entry_candle",
                detail=f"Signal {signal_row['id']} at {candle_ts_str}: missing entry candle {entry_ts}, using ref close",
            )

        # Exit candle is t + h (i.e. t + h*5m)
        target_exit_ts = format_utc_iso(candle_dt + timedelta(minutes=h * 5))
        c_exit = get_candle(self.conn, target_exit_ts, symbol=self.symbol)
        exit_ts = target_exit_ts
        exit_price: float | None = None

        if c_exit is not None:
            exit_price = float(c_exit["close"])
        else:
            # Look up subsequent candle in same session up to session_end_iso
            search_end = session_end_iso
            subsequent = load_candles(
                self.conn,
                symbol=self.symbol,
                start_ts=target_exit_ts,
                end_ts=search_end,
                limit=10,
            )
            if subsequent:
                c_exit = subsequent[0]
                exit_price = float(c_exit["close"])
                exit_ts = str(c_exit["ts_start"])
                log_health_event(
                    self.conn,
                    event_type="late_exit_candle",
                    detail=f"Signal {signal_row['id']}: missing target exit {target_exit_ts}, using late exit {exit_ts}",
                )
            else:
                # Still waiting for candle, or session ended without it
                if session_end_iso is not None and target_exit_ts >= session_end_iso:
                    log_health_event(
                        self.conn,
                        event_type="missing_exit_candle",
                        detail=f"Signal {signal_row['id']} target {target_exit_ts} reached session end without candle",
                    )
                return None

        if entry_price <= 0.0 or exit_price <= 0.0:
            return None

        fwd_ret = (exit_price - entry_price) / entry_price
        gross_bps = fwd_ret * 10_000.0
        correct, pnl_bps = compute_pnl_bps(
            direction=str(signal_row["direction"]),
            fwd_ret=fwd_ret,
            cost_bps_per_side=self.cost_bps_per_side,
        )

        outcome = {
            "signal_id": int(signal_row["id"]),
            "resolved_at": utc_now_iso(),
            "ref_price": ref_price,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_ts": exit_ts,
            "fwd_ret": fwd_ret,
            "gross_bps": gross_bps,
            "correct": correct,
            "pnl_bps": pnl_bps,
        }
        if save_outcome(self.conn, outcome, commit=True):
            self._emit_ws(outcome)
            return outcome
        return None

    def resolve_due(
        self,
        current_bucket_ts: str,
        session_start_ts: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve all unresolved signals whose horizon has elapsed by current_bucket_ts."""
        curr_dt = parse_utc_dt(current_bucket_ts)
        cutoff_dt = curr_dt - timedelta(minutes=self.horizon_bars * 5)
        cutoff_iso = format_utc_iso(cutoff_dt)

        unresolved = query_unresolved_signals(self.conn, cutoff_ts=cutoff_iso)
        resolved_list: list[dict[str, Any]] = []
        for sig in unresolved:
            if session_start_ts and sig["candle_ts"] < session_start_ts:
                continue
            out = self.try_resolve_signal(sig)
            if out is not None:
                resolved_list.append(out)
        return resolved_list

    def flush_session_end(
        self,
        session_start_ts: str,
        session_end_ts: str,
    ) -> list[dict[str, Any]]:
        """Flush remaining unresolved signals at end of trading session."""
        unresolved = query_unresolved_signals(self.conn, cutoff_ts=session_end_ts)
        resolved_list: list[dict[str, Any]] = []
        for sig in unresolved:
            if sig["candle_ts"] < session_start_ts:
                continue
            out = self.try_resolve_signal(sig, session_end_iso=session_end_ts)
            if out is not None:
                resolved_list.append(out)
        return resolved_list


def compute_running_stats(
    conn: sqlite3.Connection,
    horizon_bars: int = 12,
    cost_bps: float = 1.0,
) -> dict[str, Any]:
    """Compute comprehensive accuracy, calibration, and simulated P&L metrics."""
    rows = conn.execute(
        """
        SELECT
            s.id,
            s.candle_ts,
            s.direction,
            s.confidence,
            s.p_up,
            s.meta_prob,
            s.reason,
            o.entry_price,
            o.exit_price,
            o.fwd_ret,
            o.gross_bps,
            o.correct,
            o.pnl_bps
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        ORDER BY s.candle_ts ASC
        """
    ).fetchall()

    total_signals = len(rows)
    resolved_trades: list[sqlite3.Row] = []
    trade_count = 0
    none_count = 0
    reason_counts: dict[str, int] = {}

    for r in rows:
        d = r["direction"]
        if d in ("up", "down"):
            trade_count += 1
            if r["correct"] is not None:
                resolved_trades.append(r)
        else:
            none_count += 1
            reason = r["reason"] or "none_unspecified"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    n_resolved = len(resolved_trades)
    if n_resolved == 0:
        return {
            "total_signals": total_signals,
            "trade_signals": trade_count,
            "none_signals": none_count,
            "resolved_trades": 0,
            "accuracy": None,
            "accuracy_wilson_ci": None,
            "cumulative_pnl_bps": 0.0,
            "mean_pnl_bps": 0.0,
            "cost_sensitivity_bps": {"cost_0_5": 0.0, "cost_1_0": 0.0, "cost_3_0": 0.0},
            "max_drawdown_bps": 0.0,
            "baselines": {},
            "reliability_table": [],
            "ece": None,
            "reason_breakdown": reason_counts,
        }

    correct_vals = [int(r["correct"]) for r in resolved_trades]
    acc = float(np.mean(correct_vals))
    n_eff = max(int(n_resolved / max(horizon_bars, 1)), 1)
    ci = wilson_score_interval(acc, n_eff, z=1.96)

    # Simulated PnL with custom cost sensitivities (0.5 bps, 1.0 bps, 3.0 bps per side)
    def calc_pnl(side_cost: float) -> list[float]:
        res: list[float] = []
        for r in resolved_trades:
            gross = float(r["gross_bps"])
            d = str(r["direction"])
            round_cost = 2.0 * side_cost
            p = (gross - round_cost) if d == "up" else ((-gross) - round_cost)
            res.append(p)
        return res

    pnls_1_0 = calc_pnl(cost_bps)
    pnls_0_5 = calc_pnl(0.5)
    pnls_3_0 = calc_pnl(3.0)

    cum_pnl = float(np.sum(pnls_1_0))
    mean_pnl = float(np.mean(pnls_1_0))

    # Max drawdown
    equity = np.cumsum(pnls_1_0)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calibration & ECE
    import pandas as pd
    conf_series = pd.Series(float(r["confidence"]) for r in resolved_trades)
    ok_series = pd.Series(int(r["correct"]) for r in resolved_trades)
    n_bins = min(10, n_resolved)
    if n_bins >= 2:
        try:
            bin_ece, rel_table = expected_calibration_error(conf_series, ok_series, n_bins=n_bins)
        except Exception:
            bin_ece, rel_table = None, []
    else:
        bin_ece, rel_table = None, []

    # Baselines from baseline_results view
    b_rows = conn.execute(
        """
        SELECT name, direction, gross_bps, candle_ts
        FROM baseline_results
        WHERE gross_bps IS NOT NULL
        """
    ).fetchall()

    traded_timestamps = {str(r["candle_ts"]) for r in resolved_trades}
    baselines_stat: dict[str, Any] = {}
    for b_name in ("always_up", "rsi_rule"):
        b_all = [r for r in b_rows if r["name"] == b_name and r["direction"] in ("up", "down")]
        b_same = [r for r in b_all if str(r["candle_ts"]) in traded_timestamps]

        def summarise_baseline(sub: list[sqlite3.Row]) -> dict[str, Any]:
            if not sub:
                return {"count": 0, "accuracy": None, "mean_pnl_bps": 0.0, "total_pnl_bps": 0.0}
            pnls: list[float] = []
            corrects: list[int] = []
            for r in sub:
                g = float(r["gross_bps"])
                d = str(r["direction"])
                c = 1 if ((d == "up" and g > 0) or (d == "down" and g < 0)) else 0
                p = (g - 2.0 * cost_bps) if d == "up" else ((-g) - 2.0 * cost_bps)
                corrects.append(c)
                pnls.append(p)
            return {
                "count": len(sub),
                "accuracy": float(np.mean(corrects)),
                "mean_pnl_bps": float(np.mean(pnls)),
                "total_pnl_bps": float(np.sum(pnls)),
            }

        baselines_stat[b_name] = {
            "overall": summarise_baseline(b_all),
            "same_candle": summarise_baseline(b_same),
        }

    return {
        "total_signals": total_signals,
        "trade_signals": trade_count,
        "none_signals": none_count,
        "resolved_trades": n_resolved,
        "accuracy": acc,
        "accuracy_wilson_ci": ci,
        "cumulative_pnl_bps": cum_pnl,
        "mean_pnl_bps": mean_pnl,
        "cost_sensitivity_bps": {
            "cost_0_5": float(np.sum(pnls_0_5)),
            "cost_1_0": cum_pnl,
            "cost_3_0": float(np.sum(pnls_3_0)),
        },
        "max_drawdown_bps": max_dd,
        "baselines": baselines_stat,
        "reliability_table": rel_table,
        "ece": bin_ece,
        "reason_breakdown": reason_counts,
    }
