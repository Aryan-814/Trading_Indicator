"""Tests for FastAPI REST endpoints and WebSocket /ws schema contracts."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.live.app import create_app
from src.live.db import insert_baseline, insert_signal, save_candle


def test_rest_endpoints_and_websocket_snapshot(tmp_path: Path):
    db_file = tmp_path / "api_test.db"
    bundle_dir = Path(__file__).resolve().parent.parent / "models" / "v1"

    app = create_app(
        db_path=db_file,
        backup_dir=tmp_path / "backups",
        bundle_path=bundle_dir,
        enable_stream=False,
        enable_lock=False,
    )

    with TestClient(app) as client:
        conn = app.state.conn
        ts = "2026-11-12T15:00:00Z"
        save_candle(
            conn,
            {
                "ts_start": ts,
                "symbol": "AAPL",
                "open": 180.0,
                "high": 181.0,
                "low": 179.5,
                "close": 180.5,
                "volume": 5000.0,
                "vwap": 180.2,
                "trade_count": 120,
            },
        )
        insert_signal(
            conn,
            {
                "candle_ts": ts,
                "horizon_bars": 12,
                "direction": "up",
                "confidence": 0.64,
                "p_up": 0.60,
                "meta_prob": 0.56,
                "reason": None,
                "scenarios": {"bear_q10": -0.005, "base_q50": 0.002, "bull_q90": 0.009, "horizon_bars": 12, "horizon_minutes": 60},
                "drivers": [{"feature": "ret_1", "contribution": 0.05, "value": 0.001}],
                "feature_hash": "hash123",
                "model_version": "v1",
            },
        )
        insert_baseline(conn, ts, "always_up", "up")

        # 1. GET /api/health
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        h_data = r_health.json()
        assert h_data["status"] == "ok"
        assert h_data["db_connected"] is True
        assert h_data["last_candle_ts"] == ts

        # 2. GET /api/candles
        r_candles = client.get("/api/candles", params={"symbol": "AAPL", "limit": 10})
        assert r_candles.status_code == 200
        candles = r_candles.json()
        assert len(candles) == 1
        assert candles[0]["ts_start"] == ts

        # 3. GET /api/signals
        r_signals = client.get("/api/signals", params={"limit": 10})
        assert r_signals.status_code == 200
        sigs = r_signals.json()
        assert len(sigs) == 1
        assert sigs[0]["direction"] == "up"

        # 4. GET /api/signal/{candle_ts}
        r_sig = client.get(f"/api/signal/{ts}")
        assert r_sig.status_code == 200
        sig_detail = r_sig.json()
        assert sig_detail["baselines"] == {"always_up": "up"}
        assert client.get("/api/signal/2099-01-01T00:00:00Z").status_code == 404

        # 5. GET /api/stats
        r_stats = client.get("/api/stats")
        assert r_stats.status_code == 200
        stats = r_stats.json()
        assert stats["total_signals"] == 1
        assert stats["trade_signals"] == 1

        # 6. GET /api/model_info
        r_info = client.get("/api/model_info")
        assert r_info.status_code == 200
        info = r_info.json()
        assert info["horizon_bars"] == 12
        assert info["horizon_minutes"] == 60
        assert info["feature_count"] == 24
        assert info["kdz"] == 0.3

        # 7. GET /api/export.csv
        r_csv = client.get("/api/export.csv")
        assert r_csv.status_code == 200
        assert "text/csv" in r_csv.headers["content-type"]
        assert ts in r_csv.text

        # 8. WebSocket /ws
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            assert len(msg["data"]["recent_candles"]) == 1
            assert len(msg["data"]["recent_signals"]) == 1
            ws.send_text("ping")
            pong = ws.receive_json()
            assert pong["type"] == "pong"

