"""Real-time candle builder and market feed collector."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from src.live.db import (
    get_finalised_buckets,
    init_db,
    log_health_event,
    save_candle,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

tz_ny = ZoneInfo("America/New_York")
tz_utc = timezone.utc

nyse_fallback: dict[str, dict[str, Any]] = {
    "2026-11-26": {"closed": True},
    "2026-11-27": {"close_et": "13:00", "day_bars": 42, "is_half": 1},
    "2026-12-24": {"close_et": "13:00", "day_bars": 42, "is_half": 1},
    "2026-12-25": {"closed": True},
}


def parse_utc_dt(value: str | datetime) -> datetime:
    """Parse string or datetime into an aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=tz_utc)
        return value.astimezone(tz_utc)
    raw = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz_utc)
    return dt.astimezone(tz_utc)


def format_utc_iso(dt: datetime) -> str:
    """Format aware datetime as UTC ISO-8601 string."""
    return dt.astimezone(tz_utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_to_5_minutes(value: str | datetime) -> str:
    """Floor a UTC timestamp down to its 5-minute bucket start."""
    dt = parse_utc_dt(value)
    floored_minute = dt.minute - (dt.minute % 5)
    floored_dt = dt.replace(minute=floored_minute, second=0, microsecond=0)
    return format_utc_iso(floored_dt)


def get_nyse_session_info(day_obj: date | str) -> dict[str, Any]:
    """Return NYSE session boundaries (start_utc, end_utc, day_bars, is_half, closed)."""
    if isinstance(day_obj, str):
        day_str = day_obj[:10]
        parsed_date = date.fromisoformat(day_str)
    else:
        parsed_date = day_obj
        day_str = parsed_date.isoformat()

    if parsed_date.weekday() >= 5:
        return {"closed": True, "is_half": 0, "day_bars": 0}

    fallback = nyse_fallback.get(day_str)
    if fallback and fallback.get("closed"):
        return {"closed": True, "is_half": 0, "day_bars": 0}

    is_half = 1 if (fallback and fallback.get("is_half")) else 0
    close_hour = 13 if is_half else 16
    day_bars = 42 if is_half else 78

    open_dt_ny = datetime.combine(parsed_date, time(9, 30), tzinfo=tz_ny)
    close_dt_ny = datetime.combine(parsed_date, time(close_hour, 0), tzinfo=tz_ny)
    open_utc = open_dt_ny.astimezone(tz_utc)
    close_utc = close_dt_ny.astimezone(tz_utc)

    return {
        "closed": False,
        "is_half": is_half,
        "day_bars": day_bars,
        "open_utc": format_utc_iso(open_utc),
        "close_utc": format_utc_iso(close_utc),
        "open_dt": open_utc,
        "close_dt": close_utc,
    }


def is_nyse_market_open(now_utc: datetime | None = None) -> bool:
    """Check whether current UTC time falls within NYSE regular trading hours."""
    now = now_utc or datetime.now(tz_utc)
    ny_now = now.astimezone(tz_ny)
    session = get_nyse_session_info(ny_now.date())
    if session.get("closed"):
        return False
    return session["open_dt"] <= now < session["close_dt"]


def aggregate_minute_bars(
    bucket_ts: str,
    symbol: str,
    minutes_dict: dict[str, dict[str, Any]],
    source: str = "live",
) -> dict[str, Any] | None:
    """Aggregate collected 1-minute bars into a 5-minute candle."""
    if not minutes_dict:
        return None

    sorted_bars = [minutes_dict[ts] for ts in sorted(minutes_dict.keys())]
    open_price = float(sorted_bars[0]["open"])
    high_price = max(float(b["high"]) for b in sorted_bars)
    low_price = min(float(b["low"]) for b in sorted_bars)
    close_price = float(sorted_bars[-1]["close"])
    total_volume = sum(float(b.get("volume", 0.0)) for b in sorted_bars)
    total_trades = sum(int(b.get("trade_count", 0)) for b in sorted_bars)

    if total_volume > 0.0:
        vwap_numerator = sum(
            float(b.get("vwap", b["close"])) * float(b.get("volume", 0.0))
            for b in sorted_bars
        )
        candle_vwap = vwap_numerator / total_volume
    else:
        candle_vwap = close_price

    return {
        "ts_start": bucket_ts,
        "symbol": symbol,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": total_volume,
        "vwap": candle_vwap,
        "trade_count": total_trades,
        "bar_count": len(sorted_bars),
        "source": source,
    }


class CandleCollector:
    """Builds 5-minute candles from 1-minute live bars with ordered finalisation."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL"),
        primary_symbol: str = "AAPL",
        min_minutes: int = 1,
        on_candle_saved: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_forming_candle: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_bucket_finalised: Callable[[str, bool], Awaitable[None] | None] | None = None,
        on_bucket_scored: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> None:
        self.conn = conn
        self.symbols = tuple(symbols)
        self.primary_symbol = primary_symbol
        self.min_minutes = min_minutes

        self.on_candle_saved = on_candle_saved
        self.on_forming_candle = on_forming_candle
        self.on_bucket_finalised = on_bucket_finalised
        self.on_bucket_scored = on_bucket_scored

        self.open_buckets: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self.finalised_buckets: dict[str, set[str]] = {
            s: get_finalised_buckets(conn, symbol=s) for s in self.symbols
        }
        self.last_bar_time: datetime | None = None
        self.last_stale_alert: datetime | None = None

    def on_minute_bar(self, raw_bar: dict[str, Any]) -> dict[str, Any] | None:
        """Process an incoming 1-minute bar. Overwrite-not-add into open bucket."""
        symbol = str(raw_bar["symbol"]).upper()
        if symbol not in self.symbols:
            return None

        bar_ts = format_utc_iso(parse_utc_dt(raw_bar["timestamp"]))
        bucket_ts = floor_to_5_minutes(bar_ts)

        if bucket_ts in self.finalised_buckets.get(symbol, set()):
            log_health_event(
                self.conn,
                event_type="late_minute_bar",
                detail=f"Rejected {symbol} minute {bar_ts} for finalised bucket {bucket_ts}",
            )
            return None

        self.last_bar_time = datetime.now(tz_utc)
        key = (symbol, bucket_ts)
        if key not in self.open_buckets:
            self.open_buckets[key] = {}

        self.open_buckets[key][bar_ts] = {
            "symbol": symbol,
            "timestamp": bar_ts,
            "open": float(raw_bar["open"]),
            "high": float(raw_bar["high"]),
            "low": float(raw_bar["low"]),
            "close": float(raw_bar["close"]),
            "volume": float(raw_bar.get("volume", 0.0)),
            "vwap": float(raw_bar.get("vwap", raw_bar["close"])),
            "trade_count": int(raw_bar.get("trade_count", 0)),
        }

        forming = aggregate_minute_bars(
            bucket_ts=bucket_ts,
            symbol=symbol,
            minutes_dict=self.open_buckets[key],
            source="forming",
        )
        if forming and symbol == self.primary_symbol and self.on_forming_candle:
            res = self.on_forming_candle(forming)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)
        return forming

    def finalise_bucket(
        self,
        bucket_ts: str,
        source: str = "live",
    ) -> dict[str, dict[str, Any] | None]:
        """Finalise candles in deterministic order: SPY -> QQQ -> AAPL -> callbacks."""
        results: dict[str, dict[str, Any] | None] = {}
        for sym in self.symbols:
            key = (sym, bucket_ts)
            minutes = self.open_buckets.pop(key, {})
            if len(minutes) >= self.min_minutes:
                candle = aggregate_minute_bars(bucket_ts, sym, minutes, source=source)
                if candle:
                    save_candle(self.conn, candle, commit=True)
                    self.finalised_buckets.setdefault(sym, set()).add(bucket_ts)
                    results[sym] = candle
                    if self.on_candle_saved:
                        res = self.on_candle_saved(candle)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
                else:
                    results[sym] = None
            else:
                results[sym] = None
                log_health_event(
                    self.conn,
                    event_type=f"missing_{sym.lower()}_candle",
                    detail=f"Bucket {bucket_ts} had {len(minutes)} minute bars (< {self.min_minutes})",
                )

        aapl_candle = results.get(self.primary_symbol)
        quality_ok = aapl_candle is not None

        if self.on_bucket_finalised:
            res = self.on_bucket_finalised(bucket_ts, quality_ok)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)

        if self.on_bucket_scored:
            res = self.on_bucket_scored(bucket_ts)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)

        return results

    def get_forming_candle(self, bucket_ts: str, symbol: str | None = None) -> dict[str, Any] | None:
        """Return the current in-progress candle aggregation without finalising."""
        sym = symbol or self.primary_symbol
        key = (sym, bucket_ts)
        minutes = self.open_buckets.get(key)
        if not minutes:
            return None
        return aggregate_minute_bars(bucket_ts, sym, minutes, source="forming")

    def check_feed_stale(self, max_stale_seconds: float = 90.0) -> bool:
        """Check feed staleness against max_stale_seconds during NYSE market hours."""
        now = datetime.now(tz_utc)
        if not is_nyse_market_open(now):
            return False
        if self.last_bar_time is None:
            return False

        delta = (now - self.last_bar_time).total_seconds()
        if delta > max_stale_seconds:
            if self.last_stale_alert is None or (now - self.last_stale_alert).total_seconds() > 60:
                self.last_stale_alert = now
                log_health_event(
                    self.conn,
                    event_type="stale_feed",
                    detail=f"No bars received for {delta:.1f}s during market hours",
                )
            return True
        return False


def warm_start_from_parquet(
    conn: sqlite3.Connection,
    parquet_path: str | Path = "data/clean/aapl_5m_clean.parquet",
    symbol: str = "AAPL",
    days: int = 30,
) -> int:
    """Seed SQLite candles table from clean historical parquet if available."""
    import pandas as pd

    path = Path(parquet_path)
    if not path.exists():
        return 0

    df = pd.read_parquet(path)
    if df.empty:
        return 0

    df = df.sort_values("timestamp").copy()
    unique_days = df["day"].unique()
    if days > 0 and len(unique_days) > days:
        cutoff_day = unique_days[-days]
        df = df[df["day"] >= cutoff_day]

    inserted = 0
    for row in df.itertuples(index=False):
        ts_val = row.timestamp
        if hasattr(ts_val, "to_pydatetime"):
            ts_iso = format_utc_iso(parse_utc_dt(ts_val.to_pydatetime()))
        else:
            ts_iso = format_utc_iso(parse_utc_dt(str(ts_val)))
        candle = {
            "ts_start": ts_iso,
            "symbol": symbol,
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
        if save_candle(conn, candle, commit=False):
            inserted += 1
    conn.commit()
    return inserted


def fetch_bars_rest(
    api_key: str,
    secret_key: str,
    symbols: list[str] | tuple[str, ...],
    start_dt: datetime,
    end_dt: datetime,
    timeframe_minutes: int = 1,
    fallback_minutes: int = 16,
) -> list[dict[str, Any]]:
    """Fetch bars via Alpaca REST with DataFeed.IEX and now-16m fallback."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)

    req_end = end_dt
    try:
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=tf,
            start=start_dt,
            end=req_end,
            feed=DataFeed.IEX,
        )
        barset = client.get_stock_bars(request)
    except Exception:
        from datetime import timedelta

        capped_end = min(end_dt, datetime.now(tz_utc) - timedelta(minutes=fallback_minutes))
        if capped_end <= start_dt:
            return []
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=tf,
            start=start_dt,
            end=capped_end,
            feed=DataFeed.IEX,
        )
        barset = client.get_stock_bars(request)

    rows: list[dict[str, Any]] = []
    data_map = barset.data if hasattr(barset, "data") else {}
    for sym, bars in data_map.items():
        for bar in bars:
            rows.append(
                {
                    "symbol": str(sym),
                    "timestamp": format_utc_iso(parse_utc_dt(bar.timestamp)),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap": float(bar.vwap if bar.vwap is not None else bar.close),
                    "trade_count": int(bar.trade_count or 0),
                }
            )
    return rows


def backfill_buckets_rest(
    conn: sqlite3.Connection,
    api_key: str,
    secret_key: str,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL"),
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    fallback_minutes: int = 16,
    source: str = "backfill",
) -> int:
    """Backfill missed 5-minute candles via Alpaca IEX 1-minute REST bars."""
    from datetime import timedelta

    now = end_dt or datetime.now(tz_utc)
    start = start_dt or (now - timedelta(days=5))
    minute_bars = fetch_bars_rest(
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        start_dt=start,
        end_dt=now,
        timeframe_minutes=1,
        fallback_minutes=fallback_minutes,
    )
    if not minute_bars:
        return 0

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for bar in minute_bars:
        bucket = floor_to_5_minutes(bar["timestamp"])
        key = (bar["symbol"], bucket)
        grouped.setdefault(key, {})[bar["timestamp"]] = bar

    saved = 0
    for (sym, bucket), mins in sorted(grouped.items()):
        candle = aggregate_minute_bars(bucket, sym, mins, source=source)
        if candle and save_candle(conn, candle, commit=False):
            saved += 1
    conn.commit()
    return saved


def reconcile_day_rest(
    conn: sqlite3.Connection,
    api_key: str,
    secret_key: str,
    day_str: str,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL"),
    fallback_minutes: int = 16,
) -> int:
    """Reconcile missing candles for a NYSE session from REST 1-minute bars."""
    session = get_nyse_session_info(day_str)
    if session.get("closed"):
        return 0
    count = backfill_buckets_rest(
        conn=conn,
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        start_dt=session["open_dt"],
        end_dt=session["close_dt"],
        fallback_minutes=fallback_minutes,
        source="reconcile",
    )
    log_health_event(
        conn,
        event_type="reconcile_complete",
        detail=f"Reconciled {count} missing candles for {day_str}",
    )
    return count


class LiveStreamRunner:
    """Manages Alpaca StockDataStream, bucket finalisation timer, and watchdog."""

    def __init__(
        self,
        collector: CandleCollector,
        api_key: str,
        secret_key: str,
        grace_seconds: int = 8,
        stale_seconds: float = 90.0,
        backup_fn: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self.collector = collector
        self.api_key = api_key
        self.secret_key = secret_key
        self.grace_seconds = grace_seconds
        self.stale_seconds = stale_seconds
        self.backup_fn = backup_fn
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._stream: Any = None

    async def start(self) -> None:
        """Start background stream, bucket finaliser, and stale watchdog tasks."""
        self._stop_event.clear()
        self._tasks = [
            asyncio.create_task(self._run_stream_loop()),
            asyncio.create_task(self._run_finalise_loop()),
            asyncio.create_task(self._run_watchdog_loop()),
        ]

    async def stop(self) -> None:
        """Gracefully stop stream and background loops."""
        self._stop_event.set()
        if self._stream is not None:
            try:
                await self._stream.stop_ws()
            except Exception:
                pass
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _handle_bar(self, bar: Any) -> None:
        """Convert Alpaca bar object or dict into collector.on_minute_bar."""
        raw = {
            "symbol": getattr(bar, "symbol", None) or bar["symbol"],
            "timestamp": getattr(bar, "timestamp", None) or bar["timestamp"],
            "open": getattr(bar, "open", None) if hasattr(bar, "open") else bar["open"],
            "high": getattr(bar, "high", None) if hasattr(bar, "high") else bar["high"],
            "low": getattr(bar, "low", None) if hasattr(bar, "low") else bar["low"],
            "close": getattr(bar, "close", None) if hasattr(bar, "close") else bar["close"],
            "volume": getattr(bar, "volume", 0.0) if hasattr(bar, "volume") else bar.get("volume", 0.0),
            "vwap": getattr(bar, "vwap", None) if hasattr(bar, "vwap") else bar.get("vwap"),
            "trade_count": getattr(bar, "trade_count", 0) if hasattr(bar, "trade_count") else bar.get("trade_count", 0),
        }
        if raw["vwap"] is None:
            raw["vwap"] = raw["close"]
        self.collector.on_minute_bar(raw)

    async def _run_stream_loop(self) -> None:
        """Connect to Alpaca StockDataStream (IEX) with exponential backoff (1s-60s)."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream

        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                self._stream = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    feed=DataFeed.IEX,
                )
                self._stream.subscribe_bars(self._handle_bar, *self.collector.symbols)
                backoff = 1.0
                await self._stream._run_forever()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log_health_event(
                    self.collector.conn,
                    event_type="stream_reconnect",
                    detail=f"Stream error ({exc}); reconnecting in {backoff:.0f}s",
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, 60.0)

    async def _run_finalise_loop(self) -> None:
        """Sleep until each 5-minute close + grace_seconds, then finalise previous bucket."""
        from datetime import timedelta

        last_finalised_day: str | None = None
        while not self._stop_event.is_set():
            now = datetime.now(tz_utc)
            next_minute = ((now.minute // 5) + 1) * 5
            if next_minute >= 60:
                boundary = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            else:
                boundary = now.replace(minute=next_minute, second=0, microsecond=0)

            wake_dt = boundary + timedelta(seconds=self.grace_seconds)
            wait_secs = max((wake_dt - now).total_seconds(), 0.5)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_secs)
                break
            except asyncio.TimeoutError:
                pass

            bucket_dt = boundary - timedelta(minutes=5)
            bucket_ny = bucket_dt.astimezone(tz_ny)
            session = get_nyse_session_info(bucket_ny.date())
            if session.get("closed"):
                continue

            if session["open_dt"] <= bucket_dt < session["close_dt"]:
                bucket_ts = format_utc_iso(bucket_dt)
                self.collector.finalise_bucket(bucket_ts, source="live")

                last_bucket_dt = session["close_dt"] - timedelta(minutes=5)
                if bucket_dt == last_bucket_dt:
                    day_key = bucket_ny.date().isoformat()
                    if last_finalised_day != day_key:
                        last_finalised_day = day_key
                        if self.backup_fn is not None:
                            await self.backup_fn()

    async def _run_watchdog_loop(self) -> None:
        """Periodically inspect feed staleness."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=15.0)
                break
            except asyncio.TimeoutError:
                self.collector.check_feed_stale(max_stale_seconds=self.stale_seconds)

