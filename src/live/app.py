"""FastAPI REST and WebSocket application for the live indicator backend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yaml

from src.live.collector import (
    CandleCollector,
    LiveStreamRunner,
    backfill_current_session,
    is_nyse_market_open,
    nyse_fallback,
    utc_now_iso,
    warm_start,
    warm_start_from_parquet,
)
from src.live.db import (
    backup_db,
    checkpoint_and_close,
    connect_db,
    get_candle,
    get_signal_by_ts,
    init_db,
    load_candles,
    log_health_event,
    query_recent_health_events,
    query_signals_with_outcomes,
    run_backup_async,
)
from src.live.predictor import LivePredictor
from src.live.scorer import LiveScorer, compute_running_stats
from src.models import load_bundle

logger = logging.getLogger(__name__)


def load_live_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load live configuration block from config.yaml with sensible fallbacks."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}
    with open(cfg_file, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("live", {})


class ConnectionManager:
    """Manages active WebSocket client connections and JSON broadcasts."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        async with self._lock:
            conns = list(self.active_connections)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self.active_connections:
                        self.active_connections.remove(ws)


def create_app(
    db_path: str | Path = "data/live/indicator.db",
    backup_dir: str | Path = "backups",
    bundle_path: str | Path = "models/v1",
    enable_stream: bool = False,
    enable_lock: bool = False,
    lock_path: str | Path = "data/live/collector.lock",
    alpaca_api_key: str | None = None,
    alpaca_secret_key: str | None = None,
    enable_warm_start: bool | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    cfg = load_live_config()
    db_p = Path(db_path or cfg.get("db_path", "data/live/indicator.db"))
    backup_d = Path(backup_dir or cfg.get("db_backup_dir", "backups"))
    bundle_p = Path(bundle_path)
    lock_p = Path(lock_path or cfg.get("lock_path", "data/live/collector.lock"))

    ws_manager = ConnectionManager()
    lock_fd: int | None = None
    stream_runner: LiveStreamRunner | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal lock_fd, stream_runner
        # Process exclusivity via file lock if enabled
        if enable_lock:
            lock_p.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(str(lock_p), os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                os.close(lock_fd)
                lock_fd = None
                raise RuntimeError(f"Could not acquire exclusive lock on {lock_p}") from exc

        conn = init_db(db_p)
        app.state.conn = conn
        app.state.db_path = db_p
        app.state.backup_dir = backup_d
        app.state.ws_manager = ws_manager

        # Model bundle loading
        if bundle_p.exists():
            bundle = load_bundle(str(bundle_p))
            app.state.bundle = bundle
            horizon_bars = int(bundle.get("thr", {}).get("horizon_bars", 12))
        else:
            bundle = None
            app.state.bundle = None
            horizon_bars = 12

        cost_bps = float(cfg.get("cost_bps_per_side", 1.0))
        scorer = LiveScorer(
            conn=conn,
            horizon_bars=horizon_bars,
            cost_bps_per_side=cost_bps,
            symbol="AAPL",
            on_outcome_resolved=lambda out: ws_manager.broadcast({"type": "outcome", "data": out}),
        )
        app.state.scorer = scorer

        predictor = LivePredictor(
            conn=conn,
            bundle=bundle,
            bundle_path=bundle_p if bundle_p.exists() else None,
            symbol="AAPL",
            on_signal_emitted=lambda sig: ws_manager.broadcast({"type": "signal", "data": sig}),
        )
        app.state.predictor = predictor

        def handle_candle_saved(candle: dict[str, Any]) -> None:
            return ws_manager.broadcast({"type": "candle_closed", "data": candle})

        def handle_forming_candle(candle: dict[str, Any]) -> None:
            return ws_manager.broadcast({"type": "forming", "data": candle})

        def handle_bucket_finalised(bucket_ts: str, quality_ok: bool) -> None:
            predictor.on_bucket_closed(bucket_ts, quality_ok=quality_ok)

        def handle_bucket_scored(bucket_ts: str) -> None:
            scorer.resolve_due(bucket_ts)

        collector = CandleCollector(
            conn=conn,
            symbols=tuple(cfg.get("symbols", ["SPY", "QQQ", "AAPL"])),
            primary_symbol="AAPL",
            min_minutes=int(cfg.get("min_minutes", 1)),
            on_candle_saved=handle_candle_saved,
            on_forming_candle=handle_forming_candle,
            on_bucket_finalised=handle_bucket_finalised,
            on_bucket_scored=handle_bucket_scored,
        )
        app.state.collector = collector

        key = alpaca_api_key or os.getenv("ALPACA_API_KEY")
        secret = alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")

        # Warm start if enabled and SQLite candles is sparse (< 30 days)
        should_warm_start = enable_warm_start
        if should_warm_start is None:
            default_live_db = str(Path(cfg.get("db_path", "data/live/indicator.db")).resolve())
            should_warm_start = str(db_p.resolve()) == default_live_db

        if should_warm_start:
            cnt_row = conn.execute("SELECT COUNT(*) FROM candles WHERE symbol = 'AAPL'").fetchone()
            candle_count = cnt_row[0] if cnt_row else 0
            if candle_count < 156:
                warm_start(
                    conn=conn,
                    api_key=key,
                    secret_key=secret,
                    symbols=tuple(cfg.get("symbols", ["AAPL", "SPY", "QQQ"])),
                    days=int(cfg.get("warm_start_days", 30)),
                    parquet_path=cfg.get("data_path", "data/clean/aapl_5m_clean.parquet"),
                )

            # Backfill current session signals if enabled
            if cfg.get("backfill_session_on_start", True):
                backfill_current_session(conn, predictor=predictor, scorer=scorer, symbol="AAPL", num_sessions=int(cfg.get("backfill_sessions", 5)))
        if enable_stream and key and secret:
            async def run_backup() -> None:
                await run_backup_async(db_p, backup_d)

            stream_runner = LiveStreamRunner(
                collector=collector,
                api_key=key,
                secret_key=secret,
                grace_seconds=int(cfg.get("grace_seconds", 8)),
                stale_seconds=float(cfg.get("stale_seconds", 90.0)),
                backup_fn=run_backup,
            )
            app.state.stream_runner = stream_runner
            await stream_runner.start()

        try:
            yield
        finally:
            if stream_runner is not None:
                await stream_runner.stop()
            checkpoint_and_close(conn)
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError:
                    pass

    application = FastAPI(
        title="Live Trading Indicator Backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def get_health() -> dict[str, Any]:
        conn: sqlite3.Connection = application.state.conn
        collector: CandleCollector = application.state.collector
        try:
            candles = load_candles(conn, symbol="AAPL", limit=1)
            last_candle_ts = candles[-1]["ts_start"] if candles else None
            sig_row = conn.execute("SELECT candle_ts FROM signals ORDER BY id DESC LIMIT 1").fetchone()
            last_signal_ts = sig_row["candle_ts"] if sig_row else None
            recent_events = query_recent_health_events(conn, limit=10)
            db_connected = True
        except Exception as exc:
            db_connected = False
            last_candle_ts = None
            last_signal_ts = None
            recent_events = []
            logger.error("Health DB check failed: %s", exc)

        return {
            "status": "ok" if db_connected else "degraded",
            "timestamp": utc_now_iso(),
            "market_open": is_nyse_market_open(),
            "db_connected": db_connected,
            "last_candle_ts": last_candle_ts,
            "last_signal_ts": last_signal_ts,
            "recent_health_events": recent_events,
        }

    @application.get("/api/candles")
    def get_candles(
        symbol: str = Query(default="AAPL"),
        limit: int = Query(default=390, ge=1, le=1000),
        start_ts: str | None = Query(default=None),
        end_ts: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        conn: sqlite3.Connection = application.state.conn
        return load_candles(
            conn,
            symbol=symbol.upper(),
            limit=limit,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    @application.get("/api/signals")
    def get_signals(
        limit: int = Query(default=390, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        day: str | None = Query(default=None),
        direction: str | None = Query(default=None),
    ) -> list[dict[str, Any]]:
        conn: sqlite3.Connection = application.state.conn
        return query_signals_with_outcomes(
            conn,
            limit=limit,
            offset=offset,
            day=day,
            direction=direction,
        )

    @application.get("/api/signal/{candle_ts}")
    def get_single_signal(candle_ts: str) -> dict[str, Any]:
        conn: sqlite3.Connection = application.state.conn
        sig = get_signal_by_ts(conn, candle_ts)
        if sig is None:
            raise HTTPException(status_code=404, detail=f"Signal for candle_ts {candle_ts} not found")
        return sig

    @application.get("/api/stats")
    def get_stats(
        horizon_bars: int | None = Query(default=None),
        cost_bps: float = Query(default=1.0),
    ) -> dict[str, Any]:
        conn: sqlite3.Connection = application.state.conn
        h = horizon_bars or application.state.scorer.horizon_bars
        return compute_running_stats(conn, horizon_bars=h, cost_bps=cost_bps)

    @application.get("/api/model_info")
    def get_model_info() -> dict[str, Any]:
        bundle = application.state.bundle
        if not bundle:
            return {
                "bundle_version": "none",
                "horizon_bars": 12,
                "horizon_minutes": 60,
                "kdz": 0.3,
                "tau_m": 0.5218,
                "skip_first_bars": 3,
                "feature_count": 33,
                "feature_list": [],
                "label_type": "fixed_horizon (kdz=0.30)",
            }
        thr = bundle.get("thr", {})
        info = bundle.get("info", {})
        flist = bundle.get("feature_list", {})
        h_bars = int(thr.get("horizon_bars", 12))
        kdz_val = float(thr.get("kdz", 0.3))
        feat_cols = flist.get("feat_cols", [])
        meta_cols = flist.get("meta_cols", [])
        total_feat_count = len(feat_cols) + len(meta_cols) if (len(feat_cols) + len(meta_cols)) > 0 else 33
        return {
            "bundle_version": str(info.get("bundle_version", info.get("version", "v1"))),
            "horizon_bars": h_bars,
            "horizon_minutes": h_bars * 5,
            "kdz": kdz_val,
            "tau_m": float(thr.get("tau_m", 0.5218)),
            "skip_first_bars": int(thr.get("skip_first_bars", 3)),
            "target_coverage": float(thr.get("target_coverage", 0.2)),
            "feature_count": total_feat_count,
            "feature_list": feat_cols,
            "meta_feature_list": meta_cols,
            "label_type": f"fixed_horizon (kdz={kdz_val:.2f})",
        }

    @application.get("/api/export.csv")
    def export_csv(day: str | None = Query(default=None)) -> Response:
        conn: sqlite3.Connection = application.state.conn
        signals = query_signals_with_outcomes(conn, limit=5000, offset=0, day=day)
        fieldnames = [
            "id",
            "candle_ts",
            "issued_at",
            "horizon_bars",
            "direction",
            "confidence",
            "p_up",
            "meta_prob",
            "reason",
            "feature_hash",
            "model_version",
            "resolved_at",
            "ref_price",
            "entry_price",
            "exit_price",
            "exit_ts",
            "fwd_ret",
            "gross_bps",
            "correct",
            "pnl_bps",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in signals:
            writer.writerow(s)
        return Response(content=buf.getvalue(), media_type="text/csv")

    @application.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await ws_manager.connect(websocket)
        conn: sqlite3.Connection = application.state.conn
        collector: CandleCollector = application.state.collector
        try:
            recent_candles = load_candles(conn, symbol="AAPL", limit=390)
            recent_sigs = query_signals_with_outcomes(conn, limit=390)
            stats = compute_running_stats(conn, horizon_bars=application.state.scorer.horizon_bars)
            forming = None
            if recent_candles:
                forming = collector.get_forming_candle(recent_candles[-1]["ts_start"], symbol="AAPL")

            snapshot_msg = {
                "type": "snapshot",
                "timestamp": utc_now_iso(),
                "data": {
                    "recent_candles": recent_candles,
                    "recent_signals": recent_sigs,
                    "stats": stats,
                    "forming_candle": forming,
                },
            }
            await websocket.send_text(json.dumps(snapshot_msg))

            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": utc_now_iso()}))
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            await ws_manager.disconnect(websocket)

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.exists():
        from fastapi.staticfiles import StaticFiles

        application.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return application


app = create_app()

