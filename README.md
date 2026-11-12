# AAPL 5-Minute Live Trading Indicator

A machine-learning trading indicator for Apple (AAPL) 5-minute candles. It predicts the **direction** (up / down) and **calibrated confidence** of price movement over the next $h$ bars (selected $h = 12$, i.e. 60 minutes), gating signals through a stacked meta-labeler so that only high-confidence predictions are emitted.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Methodology](#2-methodology)
   - [2.1 Data Pipeline](#21-data-pipeline)
   - [2.2 Feature Engineering](#22-feature-engineering)
   - [2.3 Labels](#23-labels)
   - [2.4 Model Architecture — Stacked Classifier](#24-model-architecture--stacked-classifier)
   - [2.5 Cross-Validation Strategy](#25-cross-validation-strategy)
   - [2.6 Calibration](#26-calibration)
   - [2.7 Quantile Price Forecasts](#27-quantile-price-forecasts)
   - [2.8 Signal Policy](#28-signal-policy)
   - [2.9 Evaluation & Baselines](#29-evaluation--baselines)
3. [Project Structure](#3-project-structure)
4. [Setup & Data](#4-setup--data)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Running Tests](#6-running-tests)
7. [Configuration Reference](#7-configuration-reference)
8. [Model Bundle](#8-model-bundle)
9. [Performance](#9-performance)
10. [Live System (Backend Implemented)](#10-live-system-backend-implemented)
11. [Limitations](#11-limitations)
12. [Design Decisions](#12-design-decisions)
13. [References](#13-references)

---

## 1. Project Overview

| Property | Value |
|---|---|
| Asset | AAPL (Apple Inc.) |
| Timeframe | 5-minute candles, regular hours only (09:30–15:55 ET) |
| Data feed | Alpaca IEX (split- and dividend-adjusted) |
| Prediction horizon | 12 bars = 60 minutes |
| Primary model | LightGBM binary classifier (up vs. down) |
| Meta-layer | LightGBM meta-labeler predicting whether the primary is correct |
| Calibration | Isotonic regression mapping raw meta-probability $m \to P(\text{correct})$ for display |
| Scenario output | Bear ($q_{10}$) / Base ($q_{50}$) / Bull ($q_{90}$) quantile price forecasts |
| Signal gate | Signal emitted only when raw meta-probability $m \ge \tau_m$ |
| Feature drivers | Native LightGBM SHAP contributions (`pred_contrib=True`, top 5 per bar) |

---

## 2. Methodology

### 2.1 Data Pipeline

```
Alpaca API (IEX feed, AAPL)
     │
     ▼ python -m src.data_pull
data/raw/aapl_5m_raw.parquet
     │
     ▼ python -m src.clean
data/clean/aapl_5m_clean.parquet
     │
     ▼ src/features.py + src/labels.py
train.py (check | val | final | refit)
     │
     ▼
models/v1/   (frozen bundle)
     │
     ▼ python predict.py
Live signal output
```

**Cleaning rules** (`src/clean.py` — canonical output `data/clean/aapl_5m_clean.parquet`):

| Step | Rule |
|---|---|
| Dedup | Drop exact UTC timestamp duplicates |
| Hours | Keep regular market hours 09:30–15:55 ET only |
| Day filter | Keep only full days (78 bars) or early-close half-days (42 bars; 11 sessions, 462 rows) |
| `bar_idx` | 0-indexed position within the day |
| `bars_left` | `day_bars - bar_idx - 1` (bars remaining **after** the current bar) |
| `gap` | Overnight log-return `log(open / prev_close)` on first bar of each day; 0 otherwise |
| `ret_1` | First bar of day: `log(close / open)`; all other bars: `log(close / prev_close)` |
| `rel_volume` | Volume $\div$ past-20-day median (`min_periods=10`) at same `bar_idx` via causal `shift(1).rolling(20, min_periods=10).median()`. First 10 trading days (before 2020-08-17) are dropped as warm-up. |
| `rel_trade_count` | Same causal calculation applied to trade count |

---

### 2.2 Feature Engineering

All **24 features** are computed in `src/features.py`. Every feature at bar $t$ uses only bars $\le t$ (verified by `tests/test_features_no_lookahead.py`, max diff `0.00e+00`).

#### Price / Momentum Features (9)

| Feature | Definition | Purpose |
|---|---|---|
| `ret_1` | First bar of day: `log(close / open)`; otherwise `log(close_t / close_{t-1})` | Most recent 5-min return |
| `ret_3` | `log(close_t / close_{t-3})` (within day, falls back to `day_open`) | 15-min momentum |
| `ret_6` | `log(close_t / close_{t-6})` (within day, falls back to `day_open`) | 30-min momentum |
| `ret_12` | `log(close_t / close_{t-12})` (within day, falls back to `day_open`) | 60-min momentum |
| `ret_24` | `log(close_t / close_{t-24})` (within day, falls back to `day_open`) | 2-hour momentum |
| `gap` | Overnight log-return on first bar, 0 otherwise | Gap-open regime |
| `day_ret` | `log(close_t / day_open)` | Intraday cumulative return |
| `prev_day_ret` | Prior day's close-to-close log-return | Day-over-day context |
| `prev_day_range` | Prior day's `(high - low) / close` | Day-over-day volatility |

#### Volatility Features (4)

| Feature | Definition |
|---|---|
| `vol_12` | Rolling 12-bar std of `ret_1` ($\approx$ 1-hour realised vol) |
| `vol_78` | Rolling 78-bar std of `ret_1` ($\approx$ 1-day realised vol) |
| `range_1` | `(high - low) / close` for the current bar |
| `range_12` | Rolling 12-bar mean of `range_1` |

#### Microstructure & Oscillator Features (5)

| Feature | Definition |
|---|---|
| `vwap_dist` | `log(close / bar_vwap)` — distance from bar VWAP |
| `sess_vwap_dist` | `log(close / session_vwap)` — distance from cumulative intraday VWAP |
| `rel_volume_log` | `log(clip(rel_volume, 0.05, 20))` |
| `rel_tc_log` | `log(clip(rel_trade_count, 0.05, 20))` |
| `rsi_14` | Wilder RSI over 14 bars (causal EWM) |

#### Calendar / Session Features (6)

| Feature | Definition |
|---|---|
| `tod` | `bar_idx / day_bars` (fractional time-of-day in $[0, 1)$) |
| `tod_sin` | `sin(2 * pi * tod)` |
| `tod_cos` | `cos(2 * pi * tod)` |
| `dow` | Day-of-week (`0 = Mon` … `4 = Fri`) |
| `is_half` | `1` if early-close half-day (42 bars), `0` otherwise |
| `bars_left` | Bars remaining after the current bar (`day_bars - bar_idx - 1`) |

---

### 2.3 Labels

Labels are computed in `src/labels.py`.

**Forward return** (strictly within-day):

$$\text{fwd}_t = \log\!\left(\frac{\text{close}_{t+h}}{\text{close}_t}\right)$$

Computed via `groupby("day")["close"].shift(-h)` so it **never crosses the overnight boundary** (`NaN` when `bars_left < h`).

**Volatility scale**:

$$s_t = \sigma_t \cdot \sqrt{h}$$

where $\sigma_t$ is the causal EWM std of `ret_1` (`span=78`, `min_periods=30`).

**Dead-zone primary label**:

$$y_t = \begin{cases} +1 & \text{if } \text{fwd}_t > k_{dz} \cdot s_t \\ -1 & \text{if } \text{fwd}_t < -k_{dz} \cdot s_t \\ 0  & \text{otherwise (dead zone)} \\ \text{NaN} & \text{if } \text{bars\_left}_t < h \text{ or warm-up} \end{cases}$$

At the production setting ($h = 12$, $k_{dz} = 0.30$), 32.6% of valid rows fall in the dead zone and 51.8% of non-dead-zone rows are UP. Only $y \in \{-1, +1\}$ rows enter primary classifier training.

**Scoring label**:

$$y^{\text{sign}}_t = \text{sign}(\text{fwd}_t), \quad \text{NaN if } \text{fwd}_t = 0 \text{ or } \text{bars\_left}_t < h$$

---

### 2.4 Model Architecture — Stacked Classifier

```
Bar features (24 dims)
        │
        ▼
  Primary LightGBM        → p_up ∈ (0, 1)   "probability this bar goes up"
  (trained on non-dead-zone bars, y ∈ {-1, +1})
        │
        ├─→ side = sign(p_up − 0.5)          "+1 (up) or −1 (down)"
        │
        ▼
  Meta-labeler LightGBM   → m ∈ (0, 1)      "raw probability primary is CORRECT"
  (inputs: p_up, p_dist, vol_12, vol_78, tod,
   rel_volume_log, range_12, is_half, bars_left)
        │
        ├─→ Gate: m ≥ tau_m ?  emit signal : none
        │
        ▼ (display only)
  Isotonic Calibrator     → conf ∈ [0, 1]   "calibrated confidence shown to user"
```

---

### 2.5 Cross-Validation Strategy

This project uses an **expanding-window purged walk-forward split** (`walk_forward_blocks` in `src/splits.py`):

```
 Q1..Q4   │  Q5  │  Q6  │  Q7  │ ...
 ═════════╪══════╪══════╪══════╪
 TRAIN    │ TEST │      │      │  fold 1 (min_q = 4)
 TRAIN───────────│ TEST │      │  fold 2
 TRAIN──────────────────│ TEST │  fold 3
```

**Purge**: Labels never cross a day boundary (`bars_left < h` is `NaN`), so dropping all training days on or after the first day of the test quarter is a complete purge — no extra embargo is needed.

- Out-of-fold primary predictions (`oof_primary`) start at the 5th quarter (`min_q = 4`).
- Out-of-fold meta predictions (`oof_meta`) start later (`min_q = 2` on the quarters that have OOF primary predictions).
- The `IsotonicRegression` calibrator is fit on out-of-fold meta predictions (`m_oof`) only.

---

### 2.6 Calibration

Raw meta-probabilities `m` are mapped to display confidence `conf` via `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` fit on `m_oof`. Platt (logistic) scaling was not compared empirically. The signal gate uses raw `m >= tau_m`; `conf` is used for display and calibration diagnostics (ECE and Brier score).

---

### 2.7 Quantile Price Forecasts

Three LightGBM quantile regressors (`objective="quantile"`, $\alpha \in \{0.10, 0.50, 0.90\}$) predict the forward log-return distribution over $h$ bars. Outputs are sorted to guarantee monotonicity (`Bear <= Base <= Bull`) and converted to price targets:

$$P^{\text{Bear}} = \text{close} \cdot e^{q_{10}}, \quad P^{\text{Base}} = \text{close} \cdot e^{q_{50}}, \quad P^{\text{Bull}} = \text{close} \cdot e^{q_{90}}$$

---

### 2.8 Signal Policy

| Condition | Rule | Reason when blocked |
|---|---|---|
| Enough time left | `bars_left >= h` | `horizon_past_close` |
| Opening settle | `bar_idx >= skip_first_bars` (3) | `opening_settle` |
| Confidence threshold | `meta_prob (m) >= tau_m` | `low_confidence` |

`tau_m` is set to the 80th percentile (`1 - target_coverage`) of `m` on the validation year (2025), targeting **20% of eligible candles**. Because eligible candles (`bar_idx >= 3` and `bars_left >= 12`) are 80.7% of all candles, this corresponds to **~16% of all candles** — observed **13.0 signals/day** on validation and **12.6 signals/day** on holdout (`reports/generated_tables.md`).

---

### 2.9 Evaluation & Baselines

All metrics are computed in `train.py:evaluate()` and `src/evaluate.py` and exported by `scripts/make_report_tables.py` to `reports/generated_tables.md`.

**Baselines**:

| Baseline | Definition | Notes |
|---|---|---|
| **B1** Always-Up | Always predict UP | Evaluated both on all candles and on the exact policy-selected candles |
| **B2** Wilder RSI rule | `rsi_14 < 30` $\to$ UP, `rsi_14 > 70` $\to$ DOWN | Triggers on ~9.4%–10.6% of candles (different coverage than the 20% policy) |
| **B3** Momentum | Continuation of `ret_6` | All eligible candles |
| **B4** Logistic Regression | L2-regularized `LogisticRegression` on standardized 24 features, fit on train | Equal 20% coverage by `|p - 0.5|` |
| **B5** LightGBM Direct | Primary LightGBM alone, no meta-layer | Equal 20% coverage by `|p_up - 0.5|` |

**Statistical tests**:
- **Wilson 95% confidence intervals** use effective sample size $n_{\text{eff}} = n / h$ to adjust for $h$-bar overlapping return autocorrelation.
- **Block permutation test** permutes contiguous blocks of length $h$ ($2{,}000$ permutations).
- **Multiple testing**: Total logged trials across all validation sweeps is **16** (`6` hist_gb + `4` aborted LightGBM + `6` final LightGBM). We report the Bonferroni-adjusted p-value $\min(1, p \times 16)$.

---

## 3. Project Structure

```
ML_project/
├── config.yaml                  ← Single source of truth for parameters
├── train.py                     ← Training pipeline (check | val | final | refit)
├── predict.py                   ← Thin CLI calling src/models.py
├── requirements.txt             ← Exact pinned dependencies
├── .env.example                 ← Alpaca credential template
├── .gitignore
│
├── scripts/
│   ├── make_report_tables.py    ← Generates reports/generated_tables.md
│   └── replay_day.py            ← Full-day live pipeline replay & parity check
│
├── src/
│   ├── features.py              ← 24 causal features; shared by train and predict
│   ├── labels.py                ← Dead-zone and scoring labels
│   ├── clean.py                 ← Cleaning pipeline writing canonical parquet
│   ├── data_pull.py             ← Alpaca IEX AAPL download
│   ├── splits.py                ← walk_forward_blocks()
│   ├── evaluate.py              ← Wilson CI, ECE, block permutation test
│   ├── models.py                ← load_bundle(), make_signal()
│   ├── calibrate.py             ← Minimal stub (calibration inline in train.py)
│   └── live/                    ← FastAPI, SQLite WAL, Alpaca IEX stream, scorer
│       ├── db.py                ← Schema, immutability triggers, online backups
│       ├── collector.py         ← Minute bar builder, NYSE schedule, REST backfill
│       ├── predictor.py         ← Live inference wrapping models/v1
│       ├── scorer.py            ← Outcome resolution, Wilson CI, ECE, baseline views
│       └── app.py               ← FastAPI REST + WebSocket (/ws) server
│
├── data/
│   ├── raw/                     ← Raw parquet (gitignored)
│   ├── clean/                   ← Canonical data/clean/aapl_5m_clean.parquet (gitignored)
│   └── live/                    ← Live SQLite WAL DB and process lock (gitignored)
│
├── models/v1/                   ← Frozen LightGBM bundle (.txt + calibrator.joblib + metadata)
├── reports/
│   ├── generated_tables.md      ← Auto-generated tables (source of all doc numbers)
│   ├── experiments.csv          ← All 16 validation trials (labelled by model & run_id)
│   ├── best_cfg.json            ← Selected h, kdz, total n_trials
│   ├── holdout_log.json         ← Holdout evaluation audit log (2 entries)
│   ├── shuffled_label_check.txt ← Output of python train.py check
│   ├── decisions.md             ← Decision register
│   ├── phase_{0..6}.md          ← Phase reports
│   └── archive/                 ← Archived scikit-learn HistGradientBoostingClassifier run
├── tests/                       ← 21 automated correctness & live backend tests
├── notebooks/01_clean_data.ipynb
└── docs/                        ← references.md, user_guide.md
```

---

## 4. Setup & Data

```bash
cd ML_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with ALPACA_API_KEY and ALPACA_SECRET_KEY
```

**Data handling**: Raw and clean datasets (`*.csv`, `*.parquet` under `data/`) are excluded from git via `.gitignore`, as re-pulling from Alpaca can shift historical adjustments. The canonical clean file `data/clean/aapl_5m_clean.parquet` (120,504 rows, SHA-256 recorded in `models/v1/bundle_info.json`) ships with the submission archive. To re-pull and clean from scratch:

```bash
python -m src.data_pull --symbols AAPL --start 2020-08-03
python -m src.clean
```

---

## 5. Running the Pipeline

All commands run from the repository root:

```bash
# 1. Shuffled-label leak check (saved to reports/shuffled_label_check.txt)
python train.py check

# 2. Horizon and dead-zone sweep on 2025 validation (6 LightGBM trials)
python train.py val

# 3. Holdout evaluation (blocked by holdout_log.json once evaluated for lightgbm) + final refit
python train.py final

# 4. Safe bundle rebuild on full data (never touches holdout)
python train.py refit

# 5. Regenerate markdown tables from metrics.json and experiments.csv
python scripts/make_report_tables.py

# 6. Run live CLI smoke test on trailing bars
python predict.py
```

---

## 6. Running Tests

```bash
python -m pytest tests/ -v
```

Results: **21 passed in ~14.5 seconds** (verified from both repo root and `tests/`).

Tests cover:
- `test_features_no_lookahead.py`: full history vs. truncated history (`0.00e+00` max diff across 200 bars)
- `test_labels_alignment.py`: `bars_left < h` masked `NaN`, 500-row same-day raw forward return verification, class balance
- `test_splits_purge.py`: `walk_forward_blocks` fold ordering, `min_q=4`, strict day-boundary separation
- `test_policy_rules.py`: `opening_settle`, `horizon_past_close`, `low_confidence`, active signal bounds, required keys
- `test_bundle_consistency.py`: `feature_list.json` vs `feat_cols` vs booster `feature_name()`, zero all-NaN columns, SHA-256 hashes
- `test_holdout_gate.py`: `holdout_log.json` blocks repeated evaluation of same model family
- `test_candle_builder.py`: 1-minute bar aggregation, overwrite idempotence, zero-volume VWAP fallback, late bar rejection
- `test_finalise_order.py`: deterministic `SPY -> QQQ -> AAPL -> predictor -> scorer` finalisation sequence and missing AAPL `data_quality` rule
- `test_signal_immutable.py`: SQLite `UPDATE`/`DELETE` abort triggers on `signals` and `baselines`
- `test_signal_idempotent.py`: duplicate closed-bucket notification writes strictly one signal row
- `test_scorer.py`: horizon outcome resolution, missing entry/late exit fallbacks, `baseline_results` view exclusions
- `test_db_backup.py`: WAL-safe synchronous and async SQLite online snapshot consistency
- `test_api_schema.py`: all 7 FastAPI REST endpoints and WebSocket `/ws` snapshot handshake

---

## 7. Configuration Reference

Parameters live in [`config.yaml`](file:///home/aryan/Sem_5/ML_project/config.yaml). All training keys are wired into `train.py` (`live.*` is reserved for the future live backend).

---

## 8. Model Bundle

`models/v1/` contains:
- `primary.txt`, `meta.txt`, `quantile_{10,50,90}.txt` (native LightGBM text format only — duplicate `.joblib` wrappers removed)
- `calibrator.joblib` (`sklearn` isotonic regression)
- `feature_list.json` (24 primary features, 9 meta features)
- `thresholds.json` (`horizon_bars=12`, `kdz=0.3`, refit `tau_m=0.5218`, `skip_first_bars=3`, `target_coverage=0.2`)
- `config_snapshot.yaml`, `bundle_info.json` (with `lightgbm`, `pandas`, `numpy`, `sklearn`, `python`, `git_commit`, `data_sha256`, and per-file SHA-256 hashes), and `metrics.json`.

---

## 9. Performance

*Source: [`models/v1/metrics.json`](file:///home/aryan/Sem_5/ML_project/models/v1/metrics.json) and [`reports/generated_tables.md`](file:///home/aryan/Sem_5/ML_project/reports/generated_tables.md).*

On the validation year (2025) the selected configuration reached 51.76% accuracy on the 20% most confident candles, versus 51.79% for always-Up on the same candles. This is the best of 16 logged trials, so it is subject to selection bias (permutation p = 0.1209, Bonferroni-adjusted 1.0000).

On the 2026 holdout the policy accuracy was 50.80% (95% CI 44.14% to 57.42%, effective sample size n/h) versus 50.76% for always-Up on the same candles (permutation p = 0.3803). Net P&L at 1 bp per side was -2.00 bps per signal. **No directional edge was demonstrated.** The calibrated confidence Brier score (0.25005) was not better than a constant predictor (0.24990). Holdout ECE was 0.0288 (validation ECE: 0.0168). The 10 to 90 percent scenario band contained 79.8% of outcomes (nominal 80%), which is the one component that held up out of sample.

The holdout was evaluated twice (scikit-learn, then LightGBM) with no tuning in between.

| Metric | Validation (2025) | Holdout (2026) |
|---|---|---|
| Policy accuracy | 51.76% | 50.80% |
| Always-Up on same candles | 51.79% | 50.76% |
| Signals / day (248 / 203 days) | 13.0 | 12.6 |
| Net P&L @ 1.0 bp/side | -0.48 bps | -2.00 bps |
| ECE | 0.0168 | 0.0288 |
| 10–90 band coverage | 80.5% | 79.8% |

> [!WARNING]
> Do not widen the validation sweep grid (`grid` in `train.py`) in search of higher validation scores. More trials on the same 2025 validation year increase selection bias, and the 2026 holdout has already been evaluated.

---

## 10. Live System (Backend Implemented)

The live streaming and inference backend (`src/live/`) is implemented (Phase 4 frontend UI remains out of scope for this milestone):
- Subscribes to Alpaca `DataFeed.IEX` 1-minute bar streams via `StockDataStream` and deterministically aggregates 5-minute candles (`grace_seconds=8`, `stale_seconds=90`).
- Persists candles, immutable signals, baseline predictions, resolved outcomes, and health events in SQLite WAL mode (`data/live/indicator.db`) with `UPDATE`/`DELETE` immutability triggers.
- Exposes 7 read-only REST endpoints (`/api/health`, `/api/candles`, `/api/signals`, `/api/signal/{candle_ts}`, `/api/stats`, `/api/model_info`, `/api/export.csv`) and a `/ws` WebSocket feed.
- Run the backend server:
  ```bash
  uvicorn src.live.app:app --host 0.0.0.0 --port 8000
  ```
- Verify zero train/serve skew against historical data via full-day replay:
  ```bash
  python scripts/replay_day.py --day 2025-12-10
  ```

---

## 11. Limitations

- **No demonstrated directional edge**: Out-of-sample holdout performance (`50.80%`) is statistically indistinguishable from the `50.76%` always-Up base rate and yields `-2.00 bps` net per signal at 1 bp/side transaction costs.
- **Day-level feature memorization**: Daily/slow features (`prev_day_ret`, `vol_78`, `prev_day_range`, `dow`) frequently top tree splits across only ~1,000 training days, allowing trees to memorize specific days and contributing to validation-to-holdout decay.
- **Paper trading / research only**: Do not use for live capital allocation; this is not financial advice.

---

## 12. Design Decisions

See [`reports/decisions.md`](file:///home/aryan/Sem_5/ML_project/reports/decisions.md) for the full register (`D1`–`D9`, `D15`–`D18`).

---

## 13. References

- Mesfin (2026). *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ*. arXiv:2605.17724.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94–107.
- Joubert, J. F. (2022). "Meta-Labeling: Theory and Framework." *Journal of Financial Data Science*, 4(3).
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*.
- Zadrozny, B. & Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *KDD*.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.
- Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA*, 22(158), 209–212.
- Scikit-learn (`IsotonicRegression`, `LogisticRegression`, metrics) and LightGBM documentation.
