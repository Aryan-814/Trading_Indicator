# Phase 6 Report: Live Backend System

**Date:** November 2026  
**Status:** Complete & Verified  
**Exit Gate Status:** All 21 automated tests passing; 100% field-by-field parity in session replay (0 train/serve skew).

---

## 1. Executive Summary

Phase 6 (Phase 3 of the Indicator Delivery Plan) implements the live market streaming, scheduling, signal inference, outcome resolution, and data delivery backend for the AAPL 5-minute directional indicator.

The live backend connects to the Alpaca Market Data stream (`DataFeed.IEX`), idempotently aggregates 1-minute bars into closed 5-minute candles, deterministically evaluates signals using the frozen Phase 2 LightGBM model bundle (`models/v1/`), resolves trade outcomes dynamically over the configured horizon ($h=12$ bars / 60 minutes), and exposes state via a high-performance, read-only FastAPI REST API and WebSocket (`/ws`) stream backed by SQLite in Write-Ahead Logging (WAL) mode.

---

## 2. Architecture & Modules Implemented

```
src/live/
├── __init__.py           ← Package initializer
├── db.py                 ← SQLite WAL persistence, schema, immutability triggers, online backups
├── collector.py          ← Minute bar aggregation, NYSE calendar + 2026 fallbacks, stream runner
├── predictor.py          ← Live history preparation, model inference, dynamic horizons, baselines
├── scorer.py             ← Outcome resolution, entry/exit fallbacks, running stats, Wilson CI, ECE
└── app.py                ← FastAPI application, lifespan locking, 7 REST endpoints, WebSocket broadcast
```

### 2.1 Database & Immutability (`src/live/db.py`)
- **Engine**: SQLite configured with `PRAGMA journal_mode=WAL;`, `PRAGMA synchronous=NORMAL;`, `PRAGMA busy_timeout=5000;`, and `foreign_keys=ON`.
- **Primary Schema**:
  - `candles`: Composite primary key `(ts_start, symbol)` with index `idx_candles_symbol_ts`.
  - `signals`: One row per closed candle (`candle_ts UNIQUE`), recording `direction`, `confidence`, `p_up`, `meta_prob`, `reason`, `scenarios_json`, `drivers_json`, and `feature_hash`.
  - `baselines`: Primary key `(candle_ts, name)`, storing `always_up` and `rsi_rule` baselines.
  - `outcomes`: Primary key `signal_id` referencing `signals(id)`, tracking `ref_price`, `entry_price`, `exit_price`, `exit_ts`, `fwd_ret`, `gross_bps`, `correct`, and `pnl_bps`.
  - `health_events`: System audit log capturing stream restarts, stale feeds, missing candles, and fallback events.
  - `baseline_results`: SQL view joining `baselines`, `signals`, and `outcomes`, excluding un-tradeable `horizon_past_close` rows.
- **Immutability Enforcement**: SQLite triggers (`signals_no_update`, `signals_no_delete`, `baselines_no_update`, `baselines_no_delete`) block any modification or deletion of emitted signals or baseline records.
- **WAL Backups**: `backup_db` and `run_backup_async` use `sqlite3.Connection.backup` running inside `asyncio.to_thread` to produce point-in-time snapshots without locking the live writer.

### 2.2 Candle Collector (`src/live/collector.py`)
- **Idempotent Aggregation**: `open_buckets[(symbol, bucket)][minute_ts]` stores incoming minute bars by exact minute timestamp. Duplicate bar arrivals overwrite earlier values rather than double-counting volume or trades.
- **Zero-Volume Protection**: If aggregated volume is 0, candle VWAP cleanly falls back to close price without raising `ZeroDivisionError`.
- **Late Bar Rejection**: Minute bars arriving after a bucket has been finalised are rejected and logged as `late_minute_bar` health events.
- **Deterministic Finalisation Order**: Every 5 minutes at `t + grace_seconds`, buckets finalise in strict sequence:
  ```
  SPY ──► QQQ ──► AAPL ──► predictor.on_bucket_closed ──► scorer.resolve_due
  ```
- **Data Quality Handling**: If the primary symbol (AAPL) bucket has no valid minute bars, collector logs `missing_aapl_candle` and notifies predictor with `quality_ok=False`, writing a `direction='none'`, `reason='data_quality'` signal row.
- **NYSE Calendar**: Full session boundaries (09:30–16:00 ET, 78 bars) and half-day boundaries (09:30–13:00 ET, 42 bars) with hardcoded 2026 Thanksgiving/Christmas early close and holiday fallbacks.
- **Watchdog & REST Backfill**: Automatic `stale_feed` detection if no bar arrives for >90s during market hours, with REST backfill supporting `now - 16 min` fallbacks on Alpaca IEX data.

### 2.3 Live Predictor (`src/live/predictor.py`)
- **Clock-Based Bar Indexing**: In live operation, `bar_idx` is computed directly from clock time:
  $$\text{bar index} = \lfloor(\text{hour} \times 60 + \text{minute} - 570) / 5\rfloor$$
  This ensures that missing candles in an earlier bucket do not cause an artificial index drift for subsequent bars.
- **Zero Train/Serve Skew**: Directly invokes `src.models.make_signal()` on the history frame reconstructed from SQLite candles, producing byte-identical feature hashes and identical decision paths.
- **Dynamic Horizon**: Extracts `horizon_bars` directly from the bundle configuration ($h=12$), calculating `horizon_minutes = 60` dynamically without hardcoding.
- **Synchronous Baselines**: Emits corresponding `always_up` and `rsi_rule` baseline records for each bucket. Excluded signals (`horizon_past_close`, `data_quality`, `missed`) cleanly set baseline directions to `'none'`.

### 2.4 Live Scorer (`src/live/scorer.py`)
- **Range Query Resolution**: At each closed bucket, queries candidate unresolved signals where:
  $$\text{candle timestamp} \le \text{current bucket timestamp} - h \times 5\text{m}$$
- **Entry & Exit Fallbacks**:
  - Entry price is the open of $t + 1$ (5m after signal). If missing, logs `missing_entry_candle` and falls back to close of $t$.
  - Exit price is the close of $t + h$. If missing, searches for the next available candle in the same session, logging `late_exit_candle`.
- **Baseline Scoring**: Signals with `direction='none'` still record forward return and gross bps if valid, allowing `baseline_results` to accurately evaluate `always_up` and `rsi_rule` across all non-excluded bars.
- **Statistical Metrics (`compute_running_stats`)**:
  - Model accuracy with Wilson 95% confidence intervals based on effective sample size $n_{\text{eff}} = n / h$.
  - Realised net P&L with round-trip transaction costs ($2 \times 1.0\text{ bps}$ per side), plus sensitivity analysis at $0.5\text{ bps}$ and $3.0\text{ bps}$.
  - Maximum drawdown in basis points.
  - Overall vs same-candle baseline comparisons.
  - 10-bin Expected Calibration Error (ECE) and reliability table.

### 2.5 REST API & WebSocket Server (`src/live/app.py`)
- **Process Exclusivity**: File locking (`fcntl.flock`) on `collector.lock` prevents concurrent backend instances.
- **Endpoints**:
  - `GET /api/health`: System status, market open state, DB connectivity, latest candle/signal timestamps, recent health events.
  - `GET /api/candles`: Paginated finalised candles by symbol with optional `start_ts` and `end_ts` filters.
  - `GET /api/signals`: Paginated signals with joined outcomes and baseline predictions.
  - `GET /api/signal/{candle_ts}`: Detailed signal view with scenarios, feature drivers, and baselines.
  - `GET /api/stats`: Running accuracy, Wilson CI, drawdown, cost sensitivity, and baseline benchmarks.
  - `GET /api/model_info`: Model version, horizon, dead-zone threshold ($k_{\text{dz}}=0.3$), meta gate ($\tau_m=0.5218$), and 24 feature names.
  - `GET /api/export.csv`: CSV export of signals joined with resolved outcomes.
  - `WebSocket /ws`: Emits an immediate `"snapshot"` upon connection and broadcasts live `"forming"`, `"candle_closed"`, `"signal"`, `"outcome"`, and `"health"` events.

---

## 3. Verification & Quality Gates

### 3.1 Automated Test Suite
All 21 unit, integration, and contract tests pass cleanly when run from the repository root and from the `tests/` directory:

```bash
$ venv/bin/pytest -v
======================== 21 passed, 1 warning in 14.52s ========================
```

| Test File | Tests | Focus Area | Result |
| :--- | :---: | :--- | :---: |
| `tests/test_api_schema.py` | 1 | All 7 REST endpoints and WebSocket `/ws` snapshot contract | **PASSED** |
| `tests/test_candle_builder.py` | 4 | Minute aggregation, overwrite idempotence, 0 volume, late bars | **PASSED** |
| `tests/test_finalise_order.py` | 2 | SPY $\to$ QQQ $\to$ AAPL order, missing candle `data_quality` | **PASSED** |
| `tests/test_signal_immutable.py` | 1 | SQLite triggers preventing `UPDATE` and `DELETE` on signals/baselines | **PASSED** |
| `tests/test_signal_idempotent.py` | 1 | Idempotent duplicate bucket handling | **PASSED** |
| `tests/test_scorer.py` | 2 | Outcome resolution, missing entry / late exit fallbacks, baselines view | **PASSED** |
| `tests/test_db_backup.py` | 1 | Synchronous and asynchronous online SQLite WAL backups | **PASSED** |
| *Prior Phase 1–2 Tests* | 9 | Causal features, label alignment, walk-forward splits, policy, holdout | **PASSED** |

### 3.2 Full-Session Replay (`scripts/replay_day.py`)
To verify zero train/serve skew, `scripts/replay_day.py` synthesized 1-minute bars from historical 5-minute session data (`2025-12-10`, 78 bars), seeded warm-up history into an isolated SQLite database, and stepped through the live pipeline bucket-by-bucket:

```bash
$ venv/bin/python scripts/replay_day.py --day 2025-12-10
=== Replaying Session: 2025-12-10 ===
Warm start: 25 days (1914 5m bars)
Replay day: 78 5m bars

--- Replay Summary for 2025-12-10 ---
Signals evaluated:       78
Trade signals emitted:   3
Signals resolved:        3
Field-by-field matches:  78/78
Total net PnL (bps):     -41.63
Model Accuracy:          0.0%

SUCCESS: 100% equivalence between live pipeline and offline model bundle!
```
- **Parity Result**: 78 out of 78 signals matched offline `src.models.make_signal()` identically across `direction`, `confidence`, `p_up`, `meta_prob`, `reason`, and `feature_hash`.

---

## 4. Deliverables Summary

1. `src/live/db.py`: Complete SQLite WAL schema with immutability triggers and async backup helper.
2. `src/live/collector.py`: Robust minute bar collector with deterministic SPY/QQQ/AAPL finalisation order.
3. `src/live/predictor.py`: Live predictor guaranteeing 0 train/serve skew and clock-based indexing.
4. `src/live/scorer.py`: Range-based trade outcome resolver with late candle fallbacks and baseline scoring.
5. `src/live/app.py`: FastAPI server with 7 read-only REST endpoints and WebSocket live broadcast.
6. `scripts/replay_day.py`: Automated CLI verifying full-day replay equivalence.
7. `tests/`: 7 new test files bringing the total automated test count to 21 passing tests.

