# Production Operations Runbook

**Service:** AAPL AI Trading Indicator Live Backend & UI  
**Target Architecture:** Linux Cloud VM (1–2 vCPU, 2 GB RAM, Ubuntu 22.04/24.04 LTS)

---

## 1. System Architecture & Prerequisites

- **Python Virtualenv**: `/opt/aapl_indicator/venv` (Python 3.12)
- **App Directory**: `/opt/aapl_indicator`
- **Database**: `/opt/aapl_indicator/data/live/indicator.db` (SQLite WAL mode)
- **Backup Directory**: `/opt/aapl_indicator/backups`
- **Environment File**: `/opt/aapl_indicator/.env` containing:
  ```bash
  ALPACA_API_KEY="PK..."
  ALPACA_SECRET_KEY="..."
  ```

---

## 2. Service Management (systemd)

The backend is supervised by systemd with automatic restart on failure, 15-second graceful shutdown timeout (`TimeoutStopSec=15`), and `KillSignal=SIGINT` to allow FastAPI lifespan WAL truncation:

```bash
# Check service status
sudo systemctl status aapl_indicator.service

# View live streaming logs
sudo journalctl -u aapl_indicator.service -f -n 100

# Restart the service (graceful WAL checkpoint & restart)
sudo systemctl restart aapl_indicator.service

# Stop the service cleanly
sudo systemctl stop aapl_indicator.service
```

---

## 3. Health Checks & Verification

### 3.1 REST Health Endpoint
```bash
curl -s http://127.0.0.1:8000/api/health | jq .
```
Expected output:
```json
{
  "status": "ok",
  "market_open": true,
  "db_connected": true,
  "last_candle_ts": "2026-11-12T15:55:00Z",
  "last_signal_ts": "2026-11-12T15:55:00Z"
}
```

### 3.2 WebSocket Verification
Verify that the WebSocket snapshot is delivered:
```bash
wscat -c ws://127.0.0.1:8000/ws
```

---

## 4. SQLite Online Backups & Data Integrity

In WAL mode, running `cp indicator.db backup.db` while the process is active can copy a stale or torn database file. Always use the built-in SQLite backup API:

```bash
# Trigger an online non-blocking snapshot using python
/opt/aapl_indicator/venv/bin/python -c "
from src.live.db import backup_db
snap = backup_db('data/live/indicator.db', 'backups')
print('Snapshot created at:', snap)
"
```

Automatic daily snapshots are scheduled by the collector at each regular session market close.

---

## 5. Troubleshooting & Incident Response

### 5.1 Stale Feed Alert (`stale_feed`)
- **Symptom**: Health banner displays "Feed STALE" or `health_events` logs `stale_feed`.
- **Cause**: No bars received from Alpaca IEX websocket for >90 seconds during open NYSE hours.
- **Action**:
  1. Check Alpaca API status page: `https://status.alpaca.markets`.
  2. Test local network connectivity: `curl -I https://data.alpaca.markets`.
  3. Restart the service: `sudo systemctl restart aapl_indicator.service`.

### 5.2 Process Lock Exclusivity Failure
- **Symptom**: `RuntimeError: Could not acquire exclusive lock on data/live/collector.lock`.
- **Cause**: A concurrent process or zombie worker holds the file lock.
- **Action**:
  ```bash
  sudo fuser data/live/collector.lock
  sudo systemctl restart aapl_indicator.service
  ```

