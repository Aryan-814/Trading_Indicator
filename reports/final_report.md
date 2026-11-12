# AAPL 5-Minute AI Trading Indicator: Final Technical Report

**Project:** AI-Powered Live Trading Indicator (AAPL, 5-Minute Candles)  
**Authors:** Advanced Agentic Coding Pair  
**Date:** November 2026  
**Repository:** `https://github.com/Aryan-814/Trading_Indicator.git`  
**Frozen Model Bundle:** `models/v1/` (LightGBM Primary + Meta-Model + Isotonic Calibrator)

---

## 1. Problem Formulation & Asset Justification

### 1.1 Objective
The objective is to design, validate, freeze, and deploy an automated machine learning indicator for **Apple Inc. (AAPL)** that evaluates incoming 5-minute intraday bars and outputs:
1. **Direction**: Up, Down, or Abstain (`NONE`).
2. **Calibrated Confidence**: An empirical probability estimate calibrated on out-of-sample data.
3. **Multi-Scenario Price Forecasts**: Monotonic quantile boundaries ($q_{10}, q_{50}, q_{90}$).
4. **Interpretability ("Why")**: Tree-based feature attribution rankings (`pred_contrib=True`).
5. **No Repainting**: Strict database immutability guaranteeing emitted signals never change retroactively.

### 1.2 Asset & Horizon Selection
- **Asset**: Apple Inc. (AAPL) was selected due to its exceptional liquidity, tight bid-ask spreads (typically 1 cent / ~0.5 bp), extensive historical availability, and continuous free availability on the Alpaca IEX real-time feed.
- **Timeframe**: 5-minute candles provide sufficient sample size (~120,000 bars from 2020 to 2026; ~78 candles per full regular trading day) while mitigating high-frequency microstructural noise.
- **Horizon**: A forward prediction horizon of $h=12$ bars (60 minutes) was selected after grid exploration across $h \in \{2, 6, 12\}$ bars. In validation, $h=12$ with dead-zone parameter $k_{\text{dz}}=0.30$ achieved the highest selective accuracy (51.76%, Trial 16), whereas shorter horizons ($h=2, 6$) exhibited lower predictive signal.

---

## 2. Related Work & Literature

The architecture incorporates principles from recent quantitative finance literature:
1. **Mesfin (2026)**: Showed that gradient boosting and LSTMs on 5-minute intraday data achieve between 50.00% and 50.89% directional accuracy. This sets honest baseline expectations that intraday directional edge is marginal and noise-dominated.
2. **López de Prado (2018)**: Established the purged and embargoed walk-forward cross-validation splitter to prevent leakage across overlapping prediction horizons, along with the triple-barrier and meta-labeling frameworks.
3. **Bailey & López de Prado (2014)**: Emphasized logging every trial (16 total trials) to correct for multiple hypothesis testing bias.
4. **Joubert (2022)**: Provided the mathematical foundation for secondary meta-labeling, separating directional classification from sizing/conviction filtering.
5. **Zadrozny & Elkan (2002)**: Provided the non-parametric isotonic regression algorithm for calibrating classifier output probabilities.
6. **Fischer & Krauss (2018)**: Demonstrated that statistical edge deteriorates rapidly once round-trip transaction costs and slippage are factored in.

---

## 3. Data Pipeline & Cleaning

### 3.1 Data Source & Feed
Data was pulled via Alpaca Market Data REST API using the **IEX feed** with `Adjustment.ALL` (split and dividend adjusted). Using the IEX feed for both historical training and live execution guarantees zero training/serving mismatch.

### 3.2 Cleaning Pipeline (`src/clean.py`)
1. **Deduplication & Sorting**: Timestamps parsed to UTC, sorted chronologically, and deduplicated.
2. **Regular Trading Hours (RTH)**: Filtered strictly to regular NYSE hours (09:30 to 15:55 ET bar starts).
3. **Session Length Verification**: Retained only full days (78 bars) and verified half-days (42 bars). Incomplete days were purged.
4. **Feature Integrity**: Verified prices $O, H, L, C > 0$ and $L \le \min(O, C) \le \max(O, C) \le H$.
5. **Relative Volume Baseline**: Normalized volume and trade counts against a 20-day rolling median at the identical time-of-day bucket (`bar_idx`), lagged by one day (`shift(1)`) to eliminate look-ahead bias. Warm-up trading days before 2020-08-17 were discarded.

---

## 4. Methodology & Model Architecture

### 4.1 Causal Feature Set (`src/features.py`)
24 causal features were computed strictly using information available at bar close $t$:
- **Returns**: Log returns over 1, 3, 6, 12, and 24 bars (`ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`).
- **Overnight Gap**: Overnight price jump `gap = log(open_0 / close_prev)`.
- **Intraday Return**: Cumulative session return `day_ret = log(close / open_0)`.
- **Volatility & Range**: 12-bar and 78-bar normalized return volatility (`vol_12`, `vol_78`) and range (`range_1`, `range_12`).
- **VWAP Distances**: Normalized distance to candle VWAP and cumulative session VWAP (`vwap_dist`, `sess_vwap_dist`).
- **Relative Volume**: Log relative volume and trade counts (`rel_volume_log`, `rel_tc_log`).
- **Temporal Indicators**: Time-of-day progress (`tod`, `tod_sin`, `tod_cos`), day-of-week (`dow`), and half-day flag (`is_half`).
- **Prior Day Stats**: Prior day return and normalized range (`prev_day_ret`, `prev_day_range`).
- **Oscillator**: 14-period Wilder Relative Strength Index (`rsi_14`).

### 4.2 Stacked Meta-Labeling Classifier
1. **Primary Model**: LightGBM binary classifier ($M_{\text{primary}}$) predicting whether forward return over $h=12$ bars exceeds the volatility dead-zone:
   $$y_{\text{primary}} = \mathbb{I}(r_{t, t+h} > 0) \quad \text{on dead-zone filtered samples}$$
2. **Secondary Meta-Model**: LightGBM binary classifier ($M_{\text{meta}}$) trained on primary model predictions ($p_{\text{up}}$, $|p_{\text{up}} - 0.5|$) combined with market state features (`vol_12`, `vol_78`, `range_12`, `rel_volume_log`, `tod`, `is_half`, `bars_left`) to predict the probability that the primary model's trade direction is correct.
3. **Probability Calibration**: Isotonic regression maps the raw meta-probability $m \in [0, 1]$ into a calibrated empirical confidence score.
4. **Quantile Regressors**: Three LightGBM regressors trained with pinball loss ($\alpha \in \{0.1, 0.5, 0.9\}$) directly estimate the 10th, 50th, and 90th percentile forward returns, enforced monotonically.

### 4.3 Signal Policy & Abstention Rules
At the close of each candle $t$:
1. If fewer than $h$ bars remain in the session (`bars_left < 12`), output `NONE` (`horizon_past_close`).
2. If within the morning settle (`bar_idx < 3`), output `NONE` (`opening_settle`).
3. If calibrated meta-probability is below the threshold ($m < \tau_m = 0.5218$), output `NONE` (`low_confidence`).
4. Otherwise, emit an active directional signal (`UP` if $p_{\text{up}} \ge 0.5$, else `DOWN`) with confidence $m$.

---

## 5. Validation Strategy & Results

### 5.1 Purged Walk-Forward Splits
The model was validated using expanding-window walk-forward cross-validation on 2025 data, partitioned into quarterly blocks with an embargo period of $h=12$ bars following each test block.

### 5.2 Performance Summary Table

| Metric | Validation (2025) | Holdout (2026) |
| :--- | :---: | :---: |
| **Horizon / Dead-zone ($k_{\text{dz}}$)** | 12 bars (60 min) / 0.30 | 12 bars (60 min) / 0.30 |
| **Total Rows Scored** | 15,477 | 12,743 |
| **Primary Directional Accuracy** | 50.86% | 51.01% |
| **ROC AUC** | 0.5128 | 0.5068 |
| **Selective Accuracy @ 20% Coverage** | **51.76%** | **50.80%** |
| **Active Signals Emitted ($n$)** | 3,215 | 2,567 |
| **Daily Signal Frequency** | 13.0 / day | 12.6 / day |
| **Policy Coverage** | 20.77% | 20.14% |
| **95% Wilson CI ($n_{\text{eff}} = n/h$)** | [45.79%, 57.67%] | [44.14%, 57.42%] |
| **Always-Up on Same Candles** | 51.79% | 50.76% |
| **Permutation Test $p$-value** | 0.1209 | 0.3803 |
| **Expected Calibration Error (ECE)** | 0.0168 | 0.0288 |
| **Quantile Coverage (10% / 80% / 10%)** | 10.4% / 80.5% / 9.1% | 9.6% / 79.8% / 10.5% |

### 5.3 Baseline Comparison

| Baseline Strategy | Validation Accuracy | Holdout Accuracy |
| :--- | :---: | :---: |
| **B1: Always-Up (All Scored Candles)** | 49.81% | 49.89% |
| **B2: Wilder RSI-14 Rule** | 54.47% (9.4% cov) | 49.70% (10.6% cov) |
| **B3: 30-min Momentum (`ret_6`)** | 47.95% | 49.05% |
| **B4: Direct Logistic Regression (Top 20%)** | 51.74% | 45.82% |
| **B5: Direct LightGBM (Top 20%)** | 51.00% | 49.59% |
| **Model (Selective Policy @ 20%)** | **51.76%** | **50.80%** |

### 5.4 Net P&L Sensitivity Analysis (bps per Signal)
Transaction costs were simulated at entry ($O_{t+1}$) and exit ($C_{t+h}$):

| Cost Assumption | Validation Net Mean (bps) | Holdout Net Mean (bps) |
| :--- | :---: | :---: |
| **0.5 bps / side** (Tightest spread) | +0.52 bps | -1.00 bps |
| **1.0 bps / side** (Baseline assumption) | -0.48 bps | -2.00 bps |
| **3.0 bps / side** (Stress test) | -4.48 bps | -6.00 bps |

---

## 6. Live Backend & System Architecture

The live streaming backend (`src/live/`) runs an async pipeline connecting to Alpaca IEX data:
1. **Persistence & Immutability (`src/live/db.py`)**: SQLite in WAL mode with `BEFORE UPDATE` and `BEFORE DELETE` triggers enforcing immutability on `signals` and `baselines`. Online snapshots are executed via `backup_db()` and `run_backup_async()`.
2. **Candle Collector (`src/live/collector.py`)**: Gathers 1-minute bars, overwriting duplicates idempotently. Deterministic finalisation order: $\text{SPY} \to \text{QQQ} \to \text{AAPL} \to \text{predictor} \to \text{scorer}$.
3. **Live Predictor (`src/live/predictor.py`)**: Constructs historical features using clock-based `bar_idx` (preventing intra-day drift), delegating directly to `models/v1/` with dynamic horizon resolution ($h=12$ / 60 min).
4. **Scorer (`src/live/scorer.py`)**: Range query `resolve_due()` matches $t+1$ entry and $t+h$ exit with automated late candle fallbacks, tracking live accuracy, Wilson intervals, and baseline views.
5. **FastAPI & WebSockets (`src/live/app.py`)**: Provides 7 read-only REST endpoints (`/api/health`, `/api/candles`, `/api/signals`, `/api/signal/{ts}`, `/api/stats`, `/api/model_info`, `/api/export.csv`) and WebSocket `/ws` snapshot broadcasting.

---

## 7. Frontend User Interface

The web interface (`src/web/`) is built with **TradingView Lightweight Charts v4.2.0**:
- **One-Second Readability**: Prominent Current Signal Card showing direction icon, confidence %, dynamic horizon, and resolution countdown.
- **Signal History Strip**: Chronological interactive strip below the chart showing every closed candle's state.
- **Tabs**:
  - *Scenarios*: Quantile dollar targets ($q_{10}, q_{50}, q_{90}$) and empirical follow-up narratives.
  - *Why*: Top 5 tree feature attribution contributions.
  - *Log*: Real-time trade log with CSV export.
  - *Performance*: Running stats, Wilson intervals, cost sensitivity, and baseline benchmarks.
  - *Guide*: Comprehensive user guide and regime risk warnings.
- **Accessibility**: Default accessible Blue/Orange palette with Green/Red toggle and Dark/Light themes.

---

## 8. Limitations & Honest Conclusions

1. **Intraday Market Randomness**: Holdout directional accuracy (50.80%) is statistically consistent with the Always-Up base rate (50.76%, $p=0.3803$). There is no statistically significant alpha after transaction costs (-2.00 bps net at 1 bp/side).
2. **Macro & Headline Exposure**: The system relies purely on price, volume, and volatility signals; it possesses no natural language awareness for unexpected news or earnings announcements.
3. **Paper Trading Scope**: The software is strictly an academic decision aid and research benchmark; it must not be used for unhedged live capital allocation.

