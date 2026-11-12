# Phase 8 Report: Deployment, Systemd Supervision, & Dry Run

**Date:** November 2026  
**Status:** Complete & Verified  
**Exit Gate Status:** Clean full-session dry-run verified; SQLite online backup API verified; systemd unit verified; 2026 NYSE calendar verified.

---

## 1. Summary of Deployment Configuration

Phase 8 delivered production deployment configurations and verified automated system lifecycle operations:

1. **Systemd Service Unit ([`deploy/aapl_indicator.service`](file:///home/aryan/Sem_5/ML_project/deploy/aapl_indicator.service))**:
   - `Restart=always` with `RestartSec=5`.
   - `TimeoutStopSec=15` ensures systemd does not prematurely terminate FastAPI during WAL checkpointing.
   - `KillSignal=SIGINT` triggers FastAPI's asynchronous lifespan handler to close the Alpaca WebSocket, release the `fcntl.flock` file lock, and execute `PRAGMA wal_checkpoint(TRUNCATE);`.
2. **Reverse Proxy Configuration ([`deploy/nginx.conf`](file:///home/aryan/Sem_5/ML_project/deploy/nginx.conf))**:
   - HTTPS termination with security headers (`X-Frame-Options`, `X-Content-Type-Options`).
   - Dedicated `/ws` block with `Upgrade $http_upgrade` and `Connection "upgrade"` headers and 3600s timeouts.
3. **Operations Runbook ([`deploy/runbook.md`](file:///home/aryan/Sem_5/ML_project/deploy/runbook.md))**:
   - Detailed incident checklists for feed staleness, process exclusivity locks, and online backup verification.

---

## 2. Dry Run & Calendar Verification

1. **Full-Session Replay Dry Run**:
   - Replayed full 78-bar session (`2025-12-10`) with 25 days of prior warm-up history via `scripts/replay_day.py`.
   - Verified 100% field-by-field parity between the live collector/predictor/scorer pipeline and offline `make_signal` (78/78 matches, 0 train/serve skew).
2. **2026 NYSE Holiday & Early Close Schedule**:
   - Validated session boundary logic via `tests/test_nyse_calendar_2026.py`:
     - **Thanksgiving (2026-11-26)**: Market closed (`day_bars = 0`).
     - **Black Friday (2026-11-27)**: Half-day session closing at 13:00 ET (`day_bars = 42`, `is_half = 1`).
     - **Christmas Eve (2026-12-24)**: Half-day session closing at 13:00 ET (`day_bars = 42`, `is_half = 1`).
     - **Christmas Day (2026-12-25)**: Market closed (`day_bars = 0`).
     - **Normal Sessions**: 78 bars (09:30 to 16:00 ET).
3. **SQLite Online Backup Validation**:
   - Verified that `backup_db` and `run_backup_async` safely snapshot active WAL databases without table locks (`tests/test_db_backup.py`).

