# AAPL 5-Minute Live Trading Indicator

A machine-learning trading signal for Apple (AAPL) 5-minute candles. The model predicts the **direction** (up / down) and **calibrated confidence** of price movement over the next *h* bars (default h = 12, i.e. 60 minutes), and gates live signals through a stacked meta-labeler so that only high-confidence predictions are shown.

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
   - [2.7 Quantile Forecasts and Scenario Engine](#27-quantile-forecasts-and-scenario-engine)
   - [2.8 Signal Policy](#28-signal-policy)
   - [2.9 Evaluation](#29-evaluation)
3. [Project Structure](#3-project-structure)
4. [Setup](#4-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Running Tests](#6-running-tests)
7. [Configuration Reference](#7-configuration-reference)
8. [Model Bundle](#8-model-bundle)
9. [Performance](#9-performance)
10. [Design Decisions](#10-design-decisions)
11. [References](#11-references)

---

## 1. Project Overview

| Property | Value |
|---|---|
| Asset | AAPL (Apple Inc.) |
| Timeframe | 5-minute candles, regular hours only (09:30–15:55 ET) |
| Data feed | Alpaca IEX (split- and dividend-adjusted) |
| Prediction horizon | 12 bars = 60 minutes (tunable) |
| Primary model | LightGBM classifier (up vs. down) |
| Meta-layer | LightGBM meta-labeler predicting whether the primary is correct |
| Calibration | Isotonic regression mapping meta-probability → P(correct) |
| Scenario output | Bear / Base / Bull price targets via quantile GBM regressors |
| Signal gate | Signal emitted only when calibrated confidence ≥ τ_m |
| Feature drivers | SHAP contributions from LightGBM (top 5 per bar) |

---

## 2. Methodology

### 2.1 Data Pipeline

```
Alpaca API (IEX feed)
     │
     ▼ src/data_pull.py
data/raw/aapl_5m_raw.parquet      (also spy, qqq)
     │
     ▼ src/clean.py
data/clean/aapl_5m_clean.parquet
     │
     ▼ src/features.py + src/labels.py
data/features/features_h12.parquet, labels_h12.parquet
     │
     ▼ train.py
models/v1/   (bundle)
     │
     ▼ predict.py
Live signal JSON
```

**Cleaning rules** (`src/clean.py` — spec §5.3):

| Step | Rule |
|---|---|
| Dedup | Drop exact timestamp duplicates |
| Hours | Keep 09:30–15:55 ET only |
| Day filter | Keep only days with exactly 78 bars (full) or 42 bars (early-close half-day) |
| `bar_idx` | 0-indexed position within the day |
| `bars_left` | `day_bars − bar_idx − 1` (bars remaining including current) |
| `gap` | Overnight log-return on first bar of each day; 0 otherwise |
| `ret_1` | Within-bar log-return: `log(close/open)` on first bar, `log(close/prev_close)` otherwise |
| `rel_volume` | Volume ÷ past-20-day median at same `bar_idx` (causal shift-then-roll) |
| `rel_trade_count` | Same as `rel_volume` but for trade count |

> [!NOTE]
> The past-20-day median for `rel_volume` is computed **per `bar_idx`** using `groupby("bar_idx").transform(shift(1).rolling(20))`. This means bar_idx = 0 always compares against the last 20 opening bars — no look-ahead.

---

### 2.2 Feature Engineering

All 33 features are computed in `src/features.py`. Every feature at bar *t* uses only bars ≤ *t* (verified by `tests/test_features_no_lookahead.py`).

#### Price / Momentum Features

| Feature | Definition | Purpose |
|---|---|---|
| `ret_1` | `log(close_t / close_{t-1})` | Most recent 5-min return |
| `ret_3` | `log(close_t / close_{t-3})` | 15-min momentum |
| `ret_6` | `log(close_t / close_{t-6})` | 30-min momentum |
| `ret_12` | `log(close_t / close_{t-12})` | 60-min momentum |
| `ret_24` | `log(close_t / close_{t-24})` | 2-hour momentum |
| `ret_78` | `log(close_t / close_{t-78})` | Full-day (prev close proxy) momentum |
| `gap` | Overnight log-return | Capture gap-open regime |
| `day_ret` | `log(close_t / day_open)` | Intraday cumulative return |
| `prev_day_ret` | Prior day's log-return | Day-over-day regime |
| `prev_day_range` | Prior day's `(high − low) / close` | Day-over-day volatility |

> [!NOTE]
> `ret_k` lookbacks use `groupby("day")["close"].shift(k)`, filling with `day_open` when fewer than *k* bars have elapsed. This ensures lookback returns never cross the overnight boundary.

#### Volatility Features

| Feature | Definition |
|---|---|
| `vol_12` | Rolling 12-bar std of `ret_1` ≈ 1-hour realised vol |
| `vol_78` | Rolling 78-bar std of `ret_1` ≈ 1-day realised vol |
| `range_1` | `(high − low) / close` for the current bar |
| `range_12` | Rolling 12-bar mean of `range_1` |

#### Microstructure Features

| Feature | Definition |
|---|---|
| `vwap_dist` | `log(close / bar_vwap)` — distance from bar VWAP |
| `sess_vwap_dist` | `log(close / session_vwap)` — distance from session VWAP (cumulative within day) |
| `rel_volume_log` | `log(rel_volume)` clipped to [−3, 3] |
| `rel_tc_log` | `log(rel_trade_count)` clipped to [−3, 3] |
| `rsi_14` | Wilder RSI over 14 bars (EWM-based) |

#### Calendar / Session Features

| Feature | Definition |
|---|---|
| `tod` | `bar_idx / day_bars` (fractional time-of-day in [0, 1]) |
| `tod_sin` | `sin(2π · tod)` |
| `tod_cos` | `cos(2π · tod)` |
| `dow` | Day-of-week (0 = Mon … 4 = Fri) |
| `is_half` | 1 if early-close half-day, 0 otherwise |
| `bars_left` | Bars remaining to close |

#### Market Context Features (SPY / QQQ)

| Feature | Definition |
|---|---|
| `spy_ret_1/6/12` | SPY log-return over 1 / 6 / 12 bars |
| `qqq_ret_1/6/12` | QQQ log-return over 1 / 6 / 12 bars |
| `spy_vol_12` | Rolling 12-bar std of SPY `ret_1` |
| `aapl_vs_spy_6` | `ret_6 − spy_ret_6` (alpha vs. market) |

> [!IMPORTANT]
> SPY and QQQ data must be downloaded alongside AAPL (`python -m src.data_pull --symbols AAPL SPY QQQ`). If the files are absent, all market-context features default to `NaN` and LightGBM handles them natively — the model still runs but loses market-context signal.

---

### 2.3 Labels

Labels are computed in `src/labels.py` (spec §7).

**Forward return** (within-day only):

$$\text{fwd}_t = \log\!\left(\frac{\text{close}_{t+h}}{\text{close}_t}\right)$$

Computed as `groupby("day")["close"].shift(-h)` so it **never crosses the overnight boundary** (NaN for the last *h* bars of each day).

**Volatility scale**:

$$s_t = \sigma_t \cdot \sqrt{h}$$

where σ_t is the EWM std of `ret_1` with span 78 (≈ 1-day lookback), `min_periods=30`.

**Dead-zone label** (primary training target):

$$y_t = \begin{cases} +1 & \text{if } \text{fwd}_t > k_{dz} \cdot s_t \\ -1 & \text{if } \text{fwd}_t < -k_{dz} \cdot s_t \\ 0  & \text{otherwise (dead zone)} \\ \text{NaN} & \text{if } \text{bars\_left}_t < h \text{ or warm-up} \end{cases}$$

The dead-zone multiplier $k_{dz}$ (default 0.30) prevents the model from learning tiny noise moves. Only $y = ±1$ rows are used in primary classifier training.

**Scoring label** (for evaluation accuracy):

$$y^{\text{sign}}_t = \text{sign}(\text{fwd}_t), \quad \text{NaN if } |\text{fwd}| = 0 \text{ or } \text{bars\_left} < h$$

---

### 2.4 Model Architecture — Stacked Classifier

The model uses a **two-stage stack** (meta-labeling) with isotonic calibration.

```
Bar features (33 dims)
        │
        ▼
  Primary LightGBM        → p_up ∈ (0, 1)   "probability this bar goes up"
        │
        ├─→ side = sign(p_up − 0.5)          "+1 (long) or −1 (short)"
        │
        ▼
  Meta-labeler LightGBM   → m ∈ (0, 1)      "probability primary is CORRECT"
  (inputs: p_up, p_dist, vol_12, vol_78, tod,
   rel_volume_log, range_12, is_half, bars_left)
        │
        ▼
  Isotonic Calibrator     → conf ∈ (0, 1)   "calibrated P(correct)"
        │
        ▼
  Gate: m ≥ τ_m  ?  emit signal  :  no signal
```

**Why meta-labeling?**
The primary classifier is trained on all bars and learns the direction. The meta-labeler is then trained on the primary's out-of-fold predictions to learn *when* the primary is likely to be right. This separates direction prediction from regime filtering, and naturally produces coverage control (we only act when the meta says we're confident).

**Why LightGBM?**
- Handles NaN natively (critical for SPY/QQQ fallback and warm-up periods)
- Fast to train and retrain on quarterly data updates
- Supports `pred_contrib=True` for SHAP-style feature attribution (spec §10.4)
- Outperforms linear models on tabular financial features with interaction structure

---

### 2.5 Cross-Validation Strategy

Standard k-fold CV **leaks** future information in time-series data. This project uses **Purged Walk-Forward CV** (López de Prado, 2018):

```
 Q1 2021  │  Q2 2021  │  Q3 2021  │  Q4 2021  │  Q1 2022  │ ...
 ══════════╪═══════════╪═══════════╪═══════════╪═══════════╪
 TRAIN     │   TEST    │           │           │           │  fold 1
 TRAIN     │  purge    │   TEST    │           │           │  fold 2
 TRAIN     │  purge    │  purge    │   TEST    │           │  fold 3
```

**Purge**: Since labels are strictly intraday (never crossing midnight), dropping all days `>= test_start_day` from training is sufficient — no training bar's label window `[t, t+h]` can overlap the test block.

**Key properties**:
- Test blocks are quarterly (∼78 × 65 ≈ 5,000 bars each)
- Minimum of 4 quarters of history required before the first fold
- OOF (out-of-fold) predictions cover the entire training period without leakage, used to train the meta-labeler

The splitter is implemented in `src/splits.py` as `PurgedWalkForwardSplitter` and independently tested in `tests/test_splits_purge.py`.

---

### 2.6 Calibration

After meta-labeling, the raw meta-probability `m` is mapped to a calibrated confidence score via **isotonic regression** (`src/calibrate.py`):

```
m  →  IsotonicRegression(out_of_bounds="clip")  →  conf
```

Isotonic regression fits a monotone non-decreasing step function that minimises squared error between `m` and the binary correct/incorrect labels. It is the least restrictive calibration method: it makes no parametric assumption about the shape.

**Why not Platt scaling?** Platt (logistic calibration) forces a sigmoid shape, which is too restrictive when meta-probabilities are already roughly monotone but have complex non-linearities from the GBM structure.

The calibrated confidence is used in:
- The signal gate (`conf ≥ τ_m`)
- The UI "confidence bar" and narrative text
- The Expected Calibration Error (ECE) metric

---

### 2.7 Quantile Forecasts and Scenario Engine

Three separate **LightGBM quantile regressors** (10th, 50th, 90th percentile) are trained on the same feature set to predict the *forward log-return* distribution:

```python
LGBMRegressor(objective="quantile", alpha=0.10)   →  q10 (Bear)
LGBMRegressor(objective="quantile", alpha=0.50)   →  q50 (Base)
LGBMRegressor(objective="quantile", alpha=0.90)   →  q90 (Bull)
```

Price targets are:

$$P^{\text{Bear}} = \text{close} \cdot e^{q_{10}}, \quad P^{\text{Base}} = \text{close} \cdot e^{q_{50}}, \quad P^{\text{Bull}} = \text{close} \cdot e^{q_{90}}$$

Monotonicity is enforced by sorting the three quantiles after prediction.

**Validation**: The 10–90 band should contain approximately 80% of actual outcomes. The `band_validation` and `band_holdout` entries in `models/v1/metrics.json` report this.

**Analog narrative** (`src/scenarios.py`):
The offline analog table (`models/v1/analog_table.parquet`) stores empirical statistics from the training period, binned by:
- `vol_bin` (low / mid / high) based on `vol_78` terciles
- `time_bin` (open / mid / afternoon / close) based on `bar_idx`
- `move_bin` (strong_down / mild_down / flat / mild_up / strong_up) based on z-scored `fwd`

For each bin, the table records the count, median forward return, 25th/75th percentiles, share of moves that continued vs. reversed. The narrative text (`lookup_analog_narrative`) is **100% number-traced**: every figure comes directly from the table row — no LLM-generated text.

---

### 2.8 Signal Policy

A signal is **not** emitted unless all of the following conditions hold:

| Condition | Rule | Reason |
|---|---|---|
| Opening settle | `bar_idx >= skip_first_bars` (default: 3) | First 3 bars have unstable spreads and order-book imbalance |
| Enough time left | `bars_left >= h` | Cannot close the position within regular hours otherwise |
| Confidence threshold | `meta_prob >= τ_m` | Only trade when meta-labeler is confident |

`τ_m` is set at the `1 − target_coverage` quantile of meta-probs on the **validation set** at training time. With `target_coverage = 0.20`, roughly 20% of eligible bars will receive a signal.

When a signal fires, the output includes:
- `direction`: "up" or "down"
- `confidence`: calibrated P(correct)
- `p_up`: primary classifier probability
- `meta_prob`: raw meta-labeler probability
- `scenarios`: Bear / Base / Bull price targets
- `drivers`: top-5 features by SHAP contribution magnitude

---

### 2.9 Evaluation

All evaluation is in `train.py:evaluate()` and `src/evaluate.py`.

| Metric | Description |
|---|---|
| `auc_primary` | ROC-AUC of primary classifier (all bars, `y_sign` as truth) |
| `acc_primary` | Raw accuracy of direction call (all bars) |
| `selective_acc_{cov}` | Accuracy at top `cov` fraction by meta-prob confidence |
| `brier_conf` | Brier score of calibrated confidence vs. correct/incorrect |
| `ece` | Expected Calibration Error (10 equal-frequency bins) |
| `perm_pvalue` | Block-permutation test p-value (blocks of size *h*) |
| `pnl_bps` | Simulated P&L in basis points net of transaction cost |
| `band_{val,holdout}` | Coverage of 10–90 quantile band |

**Baselines** computed on the same evaluation rows:

| Baseline | Description |
|---|---|
| B1 `b1_always_up` | Always predict up — fraction of bars that did go up |
| B2 `b2_rsi_rule` | RSI < 30 → long, RSI > 70 → short |
| B3 `b3_momentum_ret6` | Predict continuation of 30-min momentum |

The model must beat all three baselines on selective accuracy and pass the permutation test (p < 0.05) to proceed to holdout.

**Deflated Sharpe Ratio** (Bailey & López de Prado, 2014): Accounts for the fact that we tried multiple configurations (6 trials). It downward-adjusts the observed Sharpe by the expected maximum Sharpe under the null of no skill across *n* trials. Available in `src/evaluate.py:deflated_sharpe_ratio()`.

**Wilson confidence intervals**: All accuracy estimates use Wilson score intervals with effective sample size `n / h` (not raw bar count), which accounts for autocorrelation across *h*-bar blocks.

---

## 3. Project Structure

```
ML_project/
├── config.yaml                  ← Single source of truth for all parameters
├── train.py                     ← Full training pipeline (check | val | final | refit)
├── predict.py                   ← Live signal generation
├── requirements.txt
├── .env.example                 ← Template — copy to .env and fill Alpaca keys
├── .gitignore
│
├── src/                         ← All importable modules
│   ├── features.py              ← 33 features; shared by train and predict
│   ├── labels.py                ← Dead-zone labels and scoring labels
│   ├── clean.py                 ← Data cleaning pipeline
│   ├── data_pull.py             ← Alpaca IEX data download
│   ├── splits.py                ← PurgedWalkForwardSplitter
│   ├── evaluate.py              ← Metrics, baselines, permutation test, DSR
│   ├── calibrate.py             ← ConfidenceCalibrator, AdaptiveConformalInference
│   ├── models.py                ← load_bundle(), make_signal() (shared logic)
│   └── scenarios.py             ← predict_scenarios(), lookup_analog_narrative()
│
├── data/
│   ├── raw/                     ← aapl_5m_raw.parquet, spy_5m_raw.parquet, qqq_5m_raw.parquet
│   ├── clean/                   ← aapl_5m_clean.parquet
│   └── features/                ← features_h12.parquet, labels_h12.parquet
│
├── models/
│   └── v1/
│       ├── primary.txt          ← LightGBM native primary classifier
│       ├── meta.txt             ← LightGBM native meta-labeler
│       ├── calibrator.joblib    ← Isotonic regression calibrator
│       ├── quantile_10.txt      ← Bear quantile regressor
│       ├── quantile_50.txt      ← Base quantile regressor
│       ├── quantile_90.txt      ← Bull quantile regressor
│       ├── analog_table.parquet ← Offline scenario lookup table
│       ├── feature_list.json    ← Canonical feature / meta-col names
│       ├── thresholds.json      ← τ_m, h, skip_first_bars, vol_cutoffs
│       ├── config_snapshot.yaml ← Training config frozen at refit time
│       ├── bundle_info.json     ← Library versions + SHA-256 of all bundle files
│       └── metrics.json        ← Validation & holdout results (written once)
│
├── reports/
│   ├── experiments.csv          ← One row per trial from train.py val
│   ├── best_cfg.json            ← Best h and kdz from val run
│   ├── decisions.md             ← Annotated design decision log
│   └── phase_{0..5}.md         ← Phase exit-gate reports
│
├── tests/                       ← Mandatory correctness tests
│   ├── test_features_no_lookahead.py
│   ├── test_labels_alignment.py
│   ├── test_policy_rules.py
│   ├── test_scenarios_trace.py
│   └── test_splits_purge.py
│
├── notebooks/
│   └── 01_clean_data.ipynb
└── docs/
    ├── references.md
    └── user_guide.md
```

---

## 4. Setup

### Prerequisites

- Python 3.10 or 3.12
- Alpaca API key and secret (free tier works for IEX data)

### Installation

```bash
# 1. Clone / navigate to project
cd ML_project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in your Alpaca keys:
#   ALPACA_API_KEY=your_key
#   ALPACA_SECRET_KEY=your_secret
```

### Download Historical Data

```bash
# Download AAPL, SPY, QQQ from 2020-08-03 to today
python -m src.data_pull --symbols AAPL SPY QQQ --start 2020-08-03
```

### Clean the Data

```bash
python src/clean.py
# Saves data/clean/aapl_5m_clean.parquet
```

---

## 5. Running the Pipeline

All commands run from the **project root**.

### Step 1 — Sanity check (shuffled-label null test)

```bash
python train.py check
```

Runs the primary model on **shuffled** labels. Expected output:
- Accuracy ≈ 0.50 (confirms no data leak)
- AUC ≈ 0.50

### Step 2 — Validation sweep (hyperparameter search)

```bash
python train.py val
```

Trains 6 configurations (3 horizons × 2 dead-zones) on training data (≤ 2024-12-31), evaluates on validation (2025). Results written to `reports/experiments.csv`. Best config (by selective accuracy at 20% coverage, filtered by ECE ≤ 0.03) written to `reports/best_cfg.json`.

### Step 3 — Holdout evaluation + final refit

```bash
# Run holdout (once only) then refit on all data
python train.py final

# If metrics.json already exists and you want to skip holdout:
python train.py refit
```

`final` runs the holdout **exactly once** (gated by existence of `models/v1/metrics.json`) then calls `refit`. `refit` trains on all available data and saves the production bundle to `models/v1/`.

### Step 4 — Live inference

```bash
python predict.py
```

Loads `models/v1/`, reads the last 3,000 bars of `aapl_5m_clean.csv`, and prints signals on the last 200 candles along with the signal for the most recent bar.

---

## 6. Running Tests

```bash
# Run all 5 mandatory correctness tests
python -m pytest tests/ -v

# Or individually:
python tests/test_features_no_lookahead.py    # no look-ahead verification
python tests/test_labels_alignment.py         # label correctness
python tests/test_policy_rules.py             # signal gate rules
python tests/test_scenarios_trace.py          # narrative number tracing
python tests/test_splits_purge.py             # CV purge correctness
```

Expected output: **5 passed** (≈ 14 seconds).

---

## 7. Configuration Reference

All parameters live in `config.yaml`. `train.py` reads them at startup; hardcoded fallbacks are used only if the file is absent.

```yaml
asset: AAPL
timeframe_minutes: 5
feed: iex

horizon_bars: 12         # h: prediction horizon in 5-min bars

label:
  kdz: 0.30              # Dead-zone multiplier (k_dz × σ × √h)
  vol_span: 78           # EWM span for σ_t

periods:
  train_end: "2024-12-31"
  val_start: "2025-01-01"
  val_end:   "2025-12-31"
  holdout_start: "2026-01-01"

cv:
  block: quarter          # Cross-validation block size
  embargo_bars: 12        # Must equal horizon_bars

primary:
  num_leaves: 15
  max_depth: 5
  learning_rate: 0.03
  n_estimators: 200
  min_child_samples: 300
  subsample: 0.7
  colsample_bytree: 0.7
  reg_lambda: 5
  reg_alpha: 0.1
  seed: 42

meta:
  num_leaves: 7
  min_child_samples: 500

policy:
  target_coverage: 0.20   # Fraction of eligible bars to signal
  skip_first_bars: 3      # Skip opening bars (0-indexed)
```

---

## 8. Model Bundle

The production bundle at `models/v1/` is self-contained. Every consumer (predict.py, any downstream service) only needs this directory plus `src/features.py`.

| File | Description |
|---|---|
| `primary.txt` | LightGBM native-format primary classifier |
| `meta.txt` | LightGBM native-format meta-labeler |
| `calibrator.joblib` | Isotonic regression calibrator |
| `quantile_{10,50,90}.txt` | Bear / Base / Bull quantile regressors |
| `analog_table.parquet` | Offline scenario lookup table (empirical bin statistics) |
| `feature_list.json` | Canonical `feat_cols` and `meta_cols` lists |
| `thresholds.json` | τ_m, h, skip_first_bars, vol_cutoffs |
| `config_snapshot.yaml` | Training config frozen at refit time |
| `bundle_info.json` | Library versions + SHA-256 of all bundle files |
| `metrics.json` | Validation and holdout evaluation results (written once) |

The `bundle_info.json` SHA-256 hashes allow integrity verification of every model file and can be used to detect accidental bundle corruption.

---

## 9. Performance

Performance is stored in `models/v1/metrics.json` after `train.py final`. Typical results on this dataset:

| Metric | Validation (2025) | Holdout (2026) |
|---|---|---|
| Selective accuracy @ 20% coverage | ≈ 53% | ≈ 50% |
| Permutation p-value | < 0.05 | > 0.05 |
| ECE | < 0.03 | — |
| 10–90 band coverage | ≈ 80% | ≈ 79% |

> [!NOTE]
> Holdout accuracy near 50% is expected and consistent with the published intraday direction-prediction literature. The 5-minute horizon is very noisy. The validation edge (≈ 53%) reflects genuine signal but is likely attenuated in holdout due to regime shift. The model's value is in **uncertainty quantification** (knowing *when* not to trade) and **scenario ranges**, not in raw directional accuracy.

> [!TIP]
> Run `train.py val` with more aggressive hyperparameter grids (expand `grid` in `train.py`) and filter by ECE ≤ 0.03 to find configurations with better calibration, which matters more than raw accuracy for sizing decisions.

---

## 10. Design Decisions

See `reports/decisions.md` for the full annotated log. Key decisions:

| Decision | Choice | Rationale |
|---|---|---|
| D1 Asset | AAPL only | Highest 5-min liquidity, representative of large-cap tech |
| D2 Feed | Alpaca IEX | Free tier, split-adjusted, consistent with live trading |
| D3 Label | Fixed-horizon dead-zone | Simpler than triple-barrier; dead-zone prevents learning noise |
| D4 Primary model | LightGBM | Handles NaN, fast refit, native SHAP contributions |
| D5 Meta-labeling | Two-stage stack | Separates direction from regime; enables coverage control |
| D6 Calibration | Isotonic regression | Non-parametric, monotone — no shape assumption |
| D7 CV | Purged walk-forward quarterly | Prevents leakage; quarterly mirrors realistic refit cadence |
| D8 Embargo | Same as horizon (h bars) | Labels are intraday-only → day boundary is sufficient purge |
| D9 Scenarios | LightGBM quantile regressors | Coherent distributional forecasts; avoids generative hallucination |
| D10 Market context | SPY + QQQ 5-min bars | Key macro-regime signal; relative strength vs. index |

---

## 11. References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Purged walk-forward CV, meta-labeling, deflated Sharpe ratio
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems.* — RSI definition
- Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA*. — Wilson score intervals

See also `docs/references.md` for DOIs and links.
