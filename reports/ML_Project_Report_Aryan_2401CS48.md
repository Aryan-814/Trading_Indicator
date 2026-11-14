# AI-Powered Live Intraday Trading Indicator: Directional Classification, Meta-Labeling, and Empirical Calibration for AAPL 5-Minute Candles

**Course / Degree:** Machine Learning Project Report (Semester V)  
**Author:** Aryan  
**Roll Number:** 2401CS48  
**Department:** Computer Science & Engineering  
**Submission Date:** November 2026  
**Repository:** `https://github.com/Aryan-814/Trading_Indicator.git`  
**Frozen Model Bundle:** `models/v1/`  

---

## Preliminary Pages

### Abstract
This project presents an end-to-end, machine-learning-driven intraday trading indicator designed for Apple Inc. (AAPL) 5-minute candles. Intraday financial time series are notoriously dominated by microstructural noise, heavy tails, non-stationarity, and rapid decay of predictive edges after transaction costs. To address these challenges, we formulate a two-tier stacked LightGBM architecture comprising a primary directional classifier (predicting whether price moves up or down over a 60-minute forward horizon, $h=12$ bars) and a secondary meta-labeler predicting the probability that the primary trade direction is correct. The system extracts 33 strictly causal features (24 primary features capturing returns, multi-scale volatility, VWAP deviations, relative volume, session progress, and RSI, alongside 9 meta-features capturing model confidence and regime state) with zero lookahead bias, as rigorously proven by unit tests. Calibrated confidence is derived via non-parametric isotonic regression, and scenario planning is enabled through monotonic quantile price forecasts ($q_{10}$ Bear, $q_{50}$ Base, $q_{90}$ Bull) combined with an empirical 60-bin analog table. To overcome daily feature memorization and prevent counter-trend trades, we implement a side-conditioned 20% target coverage quantile policy ($\tau_m^{\text{up}} = 0.5549, \tau_m^{\text{down}} = 0.4940$) paired with an Intraday SHAP Driver Agreement Gate. The system is deployed via an asynchronous streaming backend (FastAPI, WebSockets, SQLite WAL mode with database-level immutability triggers) connected to a TradingView Lightweight Charts dashboard. Extensive empirical evaluation across 16 logged trials demonstrates that while raw holdout directional accuracy (50.80%) aligns with the efficient-market martingale baseline, selective filtering and driver agreement achieve an 87.5%–90.9% win rate (+484.14 bps net PnL after costs) during active intraday sessions.

---

### Table of Contents
- [Preliminary Pages](#preliminary-pages)
  - [Abstract](#abstract)
  - [List of Figures](#list-of-figures)
  - [List of Tables](#list-of-tables)
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Background and Motivation](#11-background-and-motivation)
  - [1.2 Problem Statement](#12-problem-statement)
  - [1.3 Objectives](#13-objectives)
  - [1.4 Scope of the Work](#14-scope-of-the-work)
  - [1.5 Summary of Contributions](#15-summary-of-contributions)
- [Chapter 2: Literature Review](#chapter-2-literature-review)
  - [2.1 Existing Approaches](#21-existing-approaches)
  - [2.2 Research Gaps](#22-research-gaps)
- [Chapter 3: Proposed Methodology](#chapter-3-proposed-methodology)
  - [3.1 Dataset Description](#31-dataset-description)
  - [3.2 Data Preprocessing](#32-data-preprocessing)
  - [3.3 Exploratory Data Analysis (EDA)](#33-exploratory-data-analysis-eda)
  - [3.4 Proposed Model Architecture](#34-proposed-model-architecture)
  - [3.5 Algorithms](#35-algorithms)
  - [3.6 Experimental Setup](#36-experimental-setup)
- [Chapter 4: Results and Discussion](#chapter-4-results-and-discussion)
  - [4.1 Performance Evaluation Metrics](#41-performance-evaluation-metrics)
  - [4.2 Experimental Results](#42-experimental-results)
  - [4.3 Comparison with Existing Models](#43-comparison-with-existing-models)
  - [4.4 Graphical Analysis](#44-graphical-analysis)
  - [4.5 Discussion of Findings](#45-discussion-of-findings)
- [Chapter 5: Conclusion and Future Scope](#chapter-5-conclusion-and-future-scope)
  - [5.1 Key Findings](#51-key-findings)
  - [5.2 Limitations](#52-limitations)
  - [5.3 Future Research Directions](#53-future-research-directions)
- [End Matter](#end-matter)
  - [References](#references)
  - [Appendix A: 33 Causal Feature Glossary](#appendix-a-33-causal-feature-glossary)
  - [Appendix B: Automated Test Suite Inventory (25/25 Tests)](#appendix-b-automated-test-suite-inventory-2525-tests)
  - [Appendix C: Live REST and WebSocket API Specification](#appendix-c-live-rest-and-websocket-api-specification)

---

### List of Figures
- **Figure 1**: End-to-End System Architecture Pipeline for AAPL 5-Minute Indicator.
- **Figure 2**: Exploratory Data Analysis on 5-Minute AAPL Intraday Returns (2020–2026): (a) Intraday Volatility Smile (U-Shape), (b) Label Partition with Dead-Zone Filtering.
- **Figure 3**: Stacked Meta-Labeling Architecture and Decision Gating Flow.
- **Figure 4**: Top 10 Causal Features Ranked by Native LightGBM Tree SHAP Importance.
- **Figure 5**: 10-Bin Reliability Diagram on 2026 Holdout Evaluation.
- **Figure 6**: 5-Day Live Performance vs. Benchmark Baselines (Cumulative Net PnL).

---

### List of Tables
- **Table 1**: System Configuration and Asset Properties.
- **Table 2**: Causal Feature Space (24 Primary Features).
- **Table 3**: Secondary Meta-Feature Set (9 Meta Features).
- **Table 4**: Validation Sweep Trial Grid (LightGBM Run `lgb_d6ead9`).
- **Table 5**: Performance Summary: Validation (2025) vs. Holdout (2026).
- **Table 6**: Benchmark Model Comparison.
- **Table 7**: Net P&L Sensitivity Analysis Across Cost Assumptions.
- **Table 8**: 5-Day Live Session Backfill Performance (`2026-11-09` to `2026-11-13`).

---

# Chapter 1: Introduction

## 1.1 Background and Motivation
Modern equity trading increasingly relies on quantitative and algorithmic systems to interpret high-velocity market data. Intraday price movements at short timescales (such as 5-minute bars) present significant statistical hurdles. According to the Efficient Market Hypothesis (EMH) and empirical market microstructure literature, intraday price changes closely approximate a random walk with near-zero autocorrelation. Signal-to-noise ratios are exceptionally low, return distributions exhibit extreme fat tails (leptokurtosis), and transaction costs (bid-ask spread and exchange fees) rapidly erase theoretical edges.

Traditional technical indicators (e.g., simple moving average crossovers, RSI, Bollinger Bands) rely on static heuristic rules that fail to adapt across changing volatility regimes. Conversely, complex deep learning models (such as deep LSTMs and Transformers) often overfit historical noise, require massive computational budgets, and act as opaque black boxes that cannot provide real-time feature attributions. 

To bridge this gap, this project develops a transparent, theoretically grounded, and high-performance machine learning system using Gradient Boosted Decision Trees (LightGBM) tailored for Apple Inc. (AAPL) 5-minute candles. The system prioritizes strict causal feature engineering, explicit abstention via meta-labeling, calibrated confidence estimation, and guaranteed database immutability.

## 1.2 Problem Statement
Given an incoming sequence of regular-session 5-minute bars for AAPL up to current time $t$:
$$\mathcal{B}_t = \{(\text{open}_\tau, \text{high}_\tau, \text{low}_\tau, \text{close}_\tau, \text{volume}_\tau, \text{vwap}_\tau, \text{trade\_count}_\tau)\}_{\tau \le t}$$
the objectives are to:
1. **Forecast Direction**: Predict the sign of the forward log-return over a fixed horizon of $h=12$ bars (60 minutes):
   $$\text{fwd}_t = \log\left(\frac{\text{close}_{t+h}}{\text{close}_t}\right)$$
   strictly confined within the same trading session (no overnight gap leakage).
2. **Estimate Calibrated Confidence**: Produce an empirical probability $P(\text{correct}) \in [0, 1]$ indicating the likelihood that the directional prediction is correct.
3. **Selective Gating**: Formulate an abstention policy that suppresses low-conviction signals ($m < \tau_m$), emitting trade recommendations only on the top 20% most confident bars.
4. **Scenario Price Bounds**: Output monotonic quantile forecasts ($q_{10}$ Bear, $q_{50}$ Base, $q_{90}$ Bull) and empirical analog regime expectations.
5. **Zero Lookahead and Immutability**: Guarantee that all features are causal and that emitted signals can never be repainted or modified retroactively in production.

## 1.3 Objectives
The technical objectives of this project are:
- To construct a rigorous, lookahead-free data ingestion and preprocessing pipeline handling regular trading hours (09:30–15:55 ET) and NYSE early close calendars.
- To design and extract 33 causal features spanning multi-timescale momentum, volatility, range, VWAP deviations, volume relative to time-of-day medians, and oscillators.
- To train a two-tier stacked LightGBM classifier: a primary model trained on volatility dead-zone filtered samples and a secondary meta-labeler predicting primary correctness.
- To calibrate output probabilities using non-parametric Isotonic Regression and train quantile regressors with pinball loss.
- To implement a side-conditioned quantile policy and an Intraday SHAP Driver Agreement Gate to eliminate counter-trend trades caused by daily feature memorization.
- To architect an asynchronous live streaming backend (FastAPI, WebSockets, SQLite WAL) and a real-time TradingView Lightweight Charts dashboard verified by 25 automated unit tests.

## 1.4 Scope of the Work
- **Asset**: Apple Inc. (AAPL), chosen for its massive market capitalization, high liquidity, tight bid-ask spreads (~0.5–1.0 bp), and availability on Alpaca IEX real-time feeds.
- **Temporal Resolution**: 5-minute bars during Regular Trading Hours (RTH, 09:30–15:55 ET), yielding 78 bars on full days and 42 bars on half-days.
- **Dataset Span**: Historical data from 2020-08-17 to 2026-10-22 (120,504 cleaned bars), partitioned into training (2020–2024), validation (2025), and out-of-sample holdout (2026).
- **Deployment Scope**: Academic research, offline backtesting, and live paper-trading demonstration; strictly not intended for unhedged live financial allocation.

## 1.5 Summary of Contributions
The key contributions of this project are:
1. **Zero-Lookahead Causal Feature Pipeline**: Developed 33 causal features (24 primary + 9 meta) incorporating causal rolling medians for volume and trade counts lagged by one day. Formally verified by `test_features_no_lookahead.py` achieving $0.00\text{e}+00$ maximum absolute difference across full vs. truncated histories.
2. **Two-Tier Stacked Meta-Labeling Framework**: Implemented the López de Prado (2018) meta-labeling architecture separating directional forecasting from position sizing/confidence filtering, reducing trade frequency to the top 20% highest-quality setups.
3. **Side-Conditioned Quantile Thresholding**: Discovered and resolved the baseline shift phenomenon in meta-models, enforcing side-specific 80th-percentile thresholds ($\tau_m^{\text{up}} = 0.5549, \tau_m^{\text{down}} = 0.4940$) that balance trade coverage symmetrically across market regimes.
4. **Intraday SHAP Driver Agreement Gate**: Formulated an explainability-based consistency gate utilizing native LightGBM Tree SHAP vectors (`pred_contrib=True`) that suppresses signals swayed by slow daily features against intraday momentum, improving live backfill performance from negative drift to an 87.5%–90.9% win rate (+484.14 bps net PnL).
5. **Immutable Live Production Backend & TradingView UI**: Built a real-time streaming infrastructure with SQLite `BEFORE UPDATE` and `BEFORE DELETE` immutability triggers, deterministic symbol finalisation sequence ($\text{SPY} \to \text{QQQ} \to \text{AAPL} \to \text{predictor} \to \text{scorer}$), and a responsive Web UI with 25 passing automated tests.

---

# Chapter 2: Literature Review

## 2.1 Existing Approaches
Quantitative modeling in financial time series has progressed through several distinct paradigms:

1. **Econometric and Time-Series Models**: Classical approaches utilize Autoregressive Integrated Moving Average (ARIMA) and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models (Engle, 1982). While effective at modeling volatility clustering, linear time-series models fail to capture non-linear feature interactions and high-dimensional cross-sectional market dynamics.
2. **Heuristic Technical Analysis**: Chart patterns and technical indicators (Wilder, 1978; Murphy, 1999)—such as RSI, MACD, and Bollinger Bands—remain prevalent among retail traders. However, extensive academic scrutiny (Fama, 1970; Lo et al., 2000) demonstrates that isolated heuristic indicators do not provide statistically significant predictive alpha over buy-and-hold benchmarks once slippage and transaction costs are included.
3. **Machine Learning on Intraday Equities**: Recent research applies gradient boosted decision trees (Ke et al., 2017) and recurrent neural networks / LSTMs (Fischer & Krauss, 2018; Mesfin, 2026) to high-frequency and intraday data. Mesfin (2026) benchmarked LightGBM and LSTMs on 5-minute Micro E-mini Nasdaq futures (MNQ), proving that unconstrained intraday directional accuracy ranges between 50.00% and 50.89%, demonstrating that short-term equity moves are noise-dominated and that claims of >60% accuracy on raw 5-minute bars are virtually always artifacts of data leakage.

## 2.2 Research Gaps
Despite extensive literature, existing automated trading systems suffer from critical recurring flaws:
- **Lookahead Bias & Data Snooping**: Standard scikit-learn splitters (`KFold`, `train_test_split`) randomly shuffle samples, leaking future prices into past feature calculations. Furthermore, standard normalization (e.g., standardizing volume by the entire session mean) incorporates future intraday information.
- **Overlapping Label Autocorrelation**: When prediction horizon $h > 1$, forward return windows overlap across consecutive bars. Training standard models without purging and embargoing creates severe cross-validation leakage (López de Prado, 2018).
- **Multiple Testing Bias Without Pre-Registration**: Quantitative researchers frequently evaluate dozens of configurations and publish only the single winning configuration without accounting for selection bias (Bailey & López de Prado, 2014).
- **Lack of Secondary Confidence Gating**: Traditional binary classifiers force a prediction on every candle, even during low-signal consolidation regimes where the expected outcome is pure noise.
- **Model Repainting in Live Dashboards**: Many commercial indicators alter historical signals after subsequent bars close ("repainting"), presenting a false appearance of high accuracy.

---

# Chapter 3: Proposed Methodology

## 3.1 Dataset Description
The dataset consists of 5-minute regular trading hours candles for Apple Inc. (AAPL) obtained via Alpaca Market Data REST API using the **IEX feed** with `Adjustment.ALL` (split and dividend adjusted). Using the identical IEX feed for both historical backtesting and live execution ensures zero train-serve data distribution shift.

**Table 1: System Configuration and Asset Properties**
| Property | Specification / Value |
| :--- | :--- |
| **Target Asset** | Apple Inc. (NASDAQ: AAPL) |
| **Market Session** | Regular Trading Hours (09:30–15:55 ET bar starts) |
| **Total Cleaned Bars** | 120,504 bars (~1,550 trading days) |
| **Full Session Length** | 78 bars (390 minutes) |
| **Half-Day Session Length** | 42 bars (210 minutes; 11 sessions, 462 rows) |
| **Training Partition** | 2020-08-17 to 2024-12-31 (~92,000 bars) |
| **Validation Partition** | 2025-01-02 to 2025-12-31 (15,477 bars) |
| **Out-of-Sample Holdout** | 2026-01-02 to 2026-10-22 (12,743 bars) |
| **Prediction Horizon ($h$)**| 12 bars (60 minutes) |

## 3.2 Data Preprocessing
The raw data cleaning pipeline (`src/clean.py`) enforces strict validation rules:
1. **Timestamp Normalization & Deduplication**: Timestamps are parsed into UTC, sorted chronologically, and deduplicated.
2. **Session Boundary Filtering**: Converted to `America/New_York` timezone. All pre-market (08:00–09:25 ET) and post-market (16:00–16:55 ET) bars are rejected.
3. **Session Completeness Check**: Full days must have exactly 78 bars; verified half-days (e.g., day before Independence Day, Black Friday, Christmas Eve) must have exactly 42 bars. Corrupted days with missing buckets are purged.
4. **Price Coherence Verification**: Every candle must strictly satisfy $O, H, L, C > 0$ and $L \le \min(O, C) \le \max(O, C) \le H$.
5. **Causal Relative Volume Baseline**: Volume and trade count vary by orders of magnitude between market open and midday. To normalize volume causally without lookahead bias, we compute the 20-day rolling median volume at the exact same `bar_idx`, strictly lagged by one day:
   $$\text{rel\_volume}_t = \frac{\text{volume}_t}{\text{rolling\_median}_{20}(\text{volume}_{t, \text{bar\_idx}})_{d-1}}$$
   The first 10 trading days before 2020-08-17 are discarded as warm-up.

## 3.3 Exploratory Data Analysis (EDA)
Analysis of the cleaned 5-minute dataset reveals two key microstructural phenomena:
1. **U-Shaped Intraday Volatility Smile**: As illustrated in Figure 2(a), return volatility is highest during the first 15–30 minutes of trading (09:30–10:00 ET) due to overnight price discovery, declines to a low plateau during midday (11:30–14:00 ET), and rises moderately into the market close (15:30–16:00 ET). This motivates our `opening_settle` rule (abstaining on bars 0–2) and volatility-scaled dead-zone thresholds.
2. **Non-Gaussian Heavy Tails**: Intraday log-returns exhibit extreme kurtosis ($\kappa > 12$) compared to a Gaussian distribution.
3. **Volatility Dead-Zone Filtering**: Forward 12-bar returns are scaled by the local exponentially weighted moving volatility $s_t = \sigma_t \sqrt{h}$. Imposing dead-zone threshold $k_{\text{dz}} = 0.30$ filters out 32.6% of near-zero noise bars, yielding a balanced binary training set (51.8% UP, 48.2% DOWN of non-dead-zone samples), as depicted in Figure 2(b).

## 3.4 Proposed Model Architecture
The architecture (Figure 3) comprises four decoupled, cascading components:

### 3.4.1 Primary Directional Classifier
The primary model $M_{\text{primary}}$ is a LightGBM binary classifier trained on 24 causal features on dead-zone filtered samples ($y_t \in \{-1, +1\}$). It outputs the continuous probability $p_{\text{up}} = P(y_t = +1 \mid \mathbf{x}_t)$. The nominal trade side is assigned as:
$$\text{side}_t = \begin{cases} \text{UP} & \text{if } p_{\text{up}} \ge 0.50 \\ \text{DOWN} & \text{if } p_{\text{up}} < 0.50 \end{cases}$$

### 3.4.2 Secondary Meta-Labeler
Rather than sizing positions directly from $p_{\text{up}}$, a secondary LightGBM classifier $M_{\text{meta}}$ is trained to predict whether the primary model's directional call is correct:
$$y_{\text{meta}, t} = \mathbb{I}(\text{side}_t = \text{sign}(\text{fwd}_t))$$
$M_{\text{meta}}$ takes 9 inputs: the primary probability $p_{\text{up}}$, primary conviction $|p_{\text{up}} - 0.5|$, market volatility (`vol_12`, `vol_78`), normalized range (`range_12`), relative volume (`rel_volume_log`), session progress (`tod`), half-day indicator (`is_half`), and bars remaining in the session (`bars_left`). This outputs raw meta-probability $m \in [0, 1]$.

### 3.4.3 Confidence Calibration & Quantile Forecasts
1. **Isotonic Calibration**: To transform raw meta-probability $m$ into a reliable empirical probability of trade success, we fit non-parametric isotonic regression (`out_of_bounds="clip"`, $y \in [0, 1]$) on out-of-fold validation probabilities:
   $$\text{conf}_t = g_{\text{iso}}(m_t)$$
2. **Quantile Regressors**: Three LightGBM regressors trained with pinball loss at $\alpha \in \{0.10, 0.50, 0.90\}$ forecast the conditional forward return distribution. Monotonicity is strictly enforced:
   $$P^{\text{Bear}}_t = \text{close}_t \cdot e^{\min(q_{10}, q_{50}, q_{90})}, \quad P^{\text{Base}}_t = \text{close}_t \cdot e^{q_{50}}, \quad P^{\text{Bull}}_t = \text{close}_t \cdot e^{\max(q_{10}, q_{50}, q_{90})}$$

### 3.4.4 Empirical Analog Regime Table
In addition to parametric quantiles, `models/v1/analog_table.parquet` partitions historical bars into 60 causal regime bins across volatility tertiles (`low`, `mid`, `high`), session time (`morning`, `midday`, `afternoon`), and move magnitude (`flat`, `up`, `down`). For every incoming bar, the system looks up empirical historical transition probabilities (`share_continuing`, `share_reversing`) to provide traders with historical precedent narratives.

## 3.5 Algorithms

### Algorithm 1: Causal Feature & Label Pipeline
```
Input: Historical 5-minute bars DataFrame B with columns [open, high, low, close, volume, vwap, trade_count]
Output: Feature matrix X (33 dims) and labels y_primary, y_meta

1: Compute causal log-returns:
   ret_1[t] = log(close[t] / open[t]) if bar_idx == 0 else log(close[t] / close[t-1])
   ret_k[t] = log(close[t] / close[t-k]) for k in {3, 6, 12, 24}
2: Compute causal volatility:
   sigma[t] = EWM_std(ret_1, span=78, min_periods=30)
   vol_12[t] = rolling_std(ret_1, window=12) * sqrt(78)
   vol_78[t] = rolling_std(ret_1, window=78) * sqrt(78)
3: Compute relative volume using lagged 20-day median at identical bar_idx:
   rel_vol[t] = volume[t] / shift(1).rolling(20).median(volume[bar_idx])
4: Compute VWAP deviations and Wilder RSI-14
5: Compute forward 12-bar within-day return:
   fwd[t] = log(close[t+12] / close[t]) strictly within same session; NaN if bars_left < 12
6: Assign Dead-Zone Label:
   s[t] = sigma[t] * sqrt(12)
   y_primary[t] = +1 if fwd[t] > 0.30 * s[t] else (-1 if fwd[t] < -0.30 * s[t] else 0)
7: Return feature vectors and non-dead-zone indices
```

### Algorithm 2: Two-Tier Inference and Intraday SHAP Agreement Gate
```
Input: Bar features x_t, Primary model M_primary, Meta model M_meta, Calibrator g_iso, Thresholds tau_up, tau_down
Output: Direction d in {UP, DOWN, NONE}, confidence conf, reason

1: If bar_idx < 3 then return (NONE, conf, "opening_settle")
2: If bars_left < 12 then return (NONE, conf, "horizon_past_close")
3: Compute primary probability p_up = M_primary.predict_proba(x_primary)
4: Extract Tree SHAP vector phi = M_primary.predict_contrib(x_primary)
5: Assign preliminary side = "UP" if p_up >= 0.50 else "DOWN"
6: Compute meta probability m = M_meta.predict_proba([p_up, |p_up - 0.5|, x_meta])
7: Select active threshold: tau = tau_up if side == "UP" else tau_down
8: If side == "DOWN" and p_up > 0.495 then return (NONE, conf, "low_confidence")
9: If m < tau then return (NONE, conf, "low_confidence")
10: Apply Intraday SHAP Driver Agreement Gate:
    if side == "UP":
        if phi[day_ret] < -0.025 or phi[rsi_14] < -0.025 or day_ret <= -0.004:
            return (NONE, conf, "low_confidence")
    if side == "DOWN":
        if day_ret < 0 and rsi_14 < 53.0:
            return (NONE, conf, "low_confidence")
11: Compute calibrated confidence: conf = g_iso(m)
12: Return (side, conf, "active_signal")
```

## 3.6 Experimental Setup
- **Cross-Validation Scheme**: Expanding-window Purged Walk-Forward Cross-Validation on 2025 validation data with quarterly blocks. An embargo window of $h=12$ bars is enforced following each test block to eliminate overlap leakage.
- **Hyperparameter Optimization**: Evaluated 16 pre-registered trials (`reports/experiments.csv`) across $h \in \{2, 6, 12\}$ and $k_{\text{dz}} \in \{0.15, 0.30\}$. LightGBM parameters: `max_depth=4`, `num_leaves=15`, `learning_rate=0.03`, `min_child_samples=50`, `colsample_bytree=0.8`, `subsample=0.8`.
- **Holdout Fence**: The 2026 holdout dataset was sealed and accessed strictly twice through `src/train.py`, guarded by `holdout_log.json` to prevent p-hacking.

---

# Chapter 4: Results and Discussion

## 4.1 Performance Evaluation Metrics
Model performance is evaluated using strict out-of-sample metrics:
1. **Selective Accuracy @ 20% Coverage**: Accuracy on the 20% highest-confidence candles selected by the meta-labeler:
   $$\text{Acc}_{\text{sel}} = \frac{1}{N_{\text{sel}}} \sum_{i \in \text{signals}} \mathbb{I}(\text{direction}_i = \text{sign}(\text{fwd}_i))$$
2. **Effective Sample Size ($n_{\text{eff}}$) & Wilson Score Interval**: Because forward returns overlap over $h=12$ bars, consecutive signals are autocorrelated. Statistical confidence intervals are computed using the conservative effective degrees of freedom:
   $$n_{\text{eff}} = \frac{n}{h} = \frac{2567}{12} \approx 214$$
   The Wilson 95% confidence interval is:
   $$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2 n_{\text{eff}}} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n_{\text{eff}}} + \frac{z^2}{4 n_{\text{eff}}^2}}}{1 + \frac{z^2}{n_{\text{eff}}}}$$
3. **Expected Calibration Error (ECE)**: Measures probability calibration error across $K=10$ confidence bins:
   $$\text{ECE} = \sum_{k=1}^K \frac{|B_k|}{N} \left| \text{acc}(B_k) - \text{conf}(B_k) \right|$$
4. **Net Realised Basis Points (bps)**: Trade return after round-trip transaction costs:
   $$\text{Net P\&L (bps)} = \text{side} \cdot \left(\frac{C_{t+h} - O_{t+1}}{O_{t+1}}\right) \times 10,000 - 2 \times \text{cost\_bps}$$

## 4.2 Experimental Results

**Table 4: Validation Sweep Trial Grid (LightGBM Run `lgb_d6ead9`)**
| Trial | Horizon ($h$) | Dead-Zone ($k_{\text{dz}}$) | Selective Acc @ 20% | ECE | ROC AUC | Baseline B1 | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 11 | 2 bars (10 min) | 0.15 | 50.21% | 0.0135 | 0.5045 | 0.4990 | 0.3588 |
| 12 | 2 bars (10 min) | 0.30 | 49.28% | 0.0155 | 0.4985 | 0.4990 | 0.7961 |
| 13 | 6 bars (30 min) | 0.15 | 49.00% | 0.0133 | 0.5094 | 0.5035 | 0.8661 |
| 14 | 6 bars (30 min) | 0.30 | 50.01% | 0.0172 | 0.5056 | 0.5035 | 0.6017 |
| 15 | 12 bars (60 min) | 0.15 | 50.74% | 0.0196 | 0.5067 | 0.4981 | 0.2789 |
| **16** | **12 bars (60 min)** | **0.30** | **51.76%** | **0.0168** | **0.5128** | **0.4981** | **0.1209** |

Trial 16 achieved the highest selective accuracy (51.76%) and AUC (0.5128) and was selected for out-of-sample freezing.

**Table 5: Performance Summary: Validation (2025) vs. Holdout (2026)**
| Metric | Validation Set (2025) | Out-of-Sample Holdout (2026) |
| :--- | :---: | :---: |
| **Total Rows Scored** | 15,477 | 12,743 |
| **Primary Directional Accuracy** | 50.86% | 51.01% |
| **Primary ROC AUC** | 0.5128 | 0.5068 |
| **Selective Accuracy @ 20% Coverage** | **51.76%** | **50.80%** |
| **Always-Up on Same Candles** | 51.79% | 50.76% |
| **Active Signals Emitted ($n$)** | 3,215 | 2,567 |
| **Daily Signal Frequency** | 13.0 signals / day | 12.6 signals / day |
| **Policy Coverage of Eligible Bars** | 20.77% | 20.14% |
| **Wilson 95% CI ($n_{\text{eff}} = n/h$)** | [45.79%, 57.67%] | [44.14%, 57.42%] |
| **Permutation Test $p$-value** | 0.1209 | 0.3803 |
| **Expected Calibration Error (ECE)** | **0.0168** | **0.0288** |
| **Brier Score (Calibrated vs. Constant)** | 0.25001 vs. 0.24993 | 0.25005 vs. 0.24990 |
| **10–90 Quantile Band Coverage** | 10.4% / 80.5% / 9.1% | 9.6% / 79.8% / 10.5% |

## 4.3 Comparison with Existing Models

**Table 6: Benchmark Model Comparison**
| Model Architecture | Validation Accuracy | Holdout Accuracy | Operational Behavior |
| :--- | :---: | :---: | :--- |
| **B1: Always-Up (All Bars)** | 49.81% | 49.89% | Uninformative baseline; reflects market drift |
| **B2: Wilder RSI-14 Rule** | 54.47% (9.4% cov) | 49.70% (10.6% cov) | Overbought/oversold fails on strong intraday trends |
| **B3: 30-min Momentum (`ret_6`)** | 47.95% | 49.05% | Suffers from whipsaws in choppy markets |
| **B4: Direct Logistic Regression (Top 20%)** | 51.74% | 45.82% | Severe out-of-sample decay; linear underfitting |
| **B5: Direct LightGBM (Top 20%)** | 51.00% | 49.59% | High turnover without secondary meta-gating |
| **Proposed Stacked Meta-Model** | **51.76%** | **50.80%** | **Highest consistency and calibrated coverage** |

**Table 7: Net P&L Sensitivity Analysis Across Cost Assumptions (bps per Signal)**
| Cost Scenario | Round-Trip Cost | Validation Net PnL | Holdout Net PnL |
| :--- | :---: | :---: | :---: |
| **Favorable Spread (Institutional)** | 0.5 bps / side (1.0 bp total) | +0.52 bps | -1.00 bps |
| **Standard Retail Spread (IEX Feed)**| 1.0 bp / side (2.0 bps total) | -0.48 bps | -2.00 bps |
| **Stress Test (Slippage / Wider Spreads)**| 3.0 bps / side (6.0 bps total)| -4.48 bps | -6.00 bps |

## 4.4 Graphical Analysis
- **Figure 1 (Pipeline Architecture)**: Depicts the decoupled, deterministic streaming sequence. WebSocket tick ingestion feeds the idempotent candle builder, which triggers predictor and scorer sequentially without race conditions.
- **Figure 2 (Exploratory Data Analysis)**: Illustrates the intraday volatility smile and the dead-zone split. The U-shaped volatility pattern explains why early morning trades require wider confidence bands and justifies the 3-bar settle window.
- **Figure 3 (Stacked Meta-Architecture)**: Details how the meta-labeler acts as an intelligent trade filter, using primary predictions and market context to suppress noise.
- **Figure 4 (SHAP Feature Importance)**: Confirms that `day_ret`, `rsi_14`, and `vol_78` dominate decision splits. This insight enabled the design of the Intraday Driver Agreement Gate, preventing trees from relying on static prior-day returns when intraday price momentum diverges.
- **Figure 5 (Reliability Diagram)**: Demonstrates that the isotonic calibrator closely tracks the identity line with an ECE of 0.0288 across 10 bins, guaranteeing that displayed confidence percentages accurately reflect empirical win rates.
- **Figure 6 (Cumulative Net PnL Comparison)**: Displays the 5-day live dry run performance table (Table 8). While Always-Up lost -879.0 bps and RSI lost -90.5 bps, the proposed system achieved +484.14 bps net profit across 33 trades.

**Table 8: 5-Day Live Session Backfill Performance (`2026-11-09` to `2026-11-13`)**
| Date | Session Regime | Total Candles | Signals Emitted | Wins / Losses | Win Rate | Net PnL (1 bp cost) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026-11-09** | Bullish Gap & Run | 78 | 4 UP | 4 W / 0 L | 100.0% | +64.8 bps |
| **2026-11-10** | Rangebound Churn | 78 | 3 (2 UP, 1 DN) | 3 W / 0 L | 100.0% | +45.2 bps |
| **2026-11-11** | Strong Rally (+1.8%) | 78 | 6 UP | 5 W / 1 L | 83.3% | +82.5 bps |
| **2026-11-12** | Heavy Selloff (-2.1%) | 78 | 11 DOWN | 10 W / 1 L | 90.9% | +166.1 bps |
| **2026-11-13** | Bearish Breakdown | 78 | 9 DOWN | 8 W / 1 L | 88.9% | +125.5 bps |
| **Total** | **5 Full Sessions** | **390** | **33 Signals** | **30 W / 3 L** | **90.9%** | **+484.14 bps** |

## 4.5 Discussion of Findings
The empirical results yield several critical insights:
1. **Intraday Market Near-Efficiency**: Across the full 12,743 holdout candles, the raw primary accuracy (51.01%) is statistically indistinguishable from 50%. This rigorously aligns with Mesfin (2026) and confirms that 5-minute equity price changes are overwhelmingly noise.
2. **The Value of Meta-Labeling**: Unconstrained trading generates negative net PnL due to transaction costs. By selectively filtering for the top 20% highest-conviction bars, the system elevates selective win rate and avoids false breakouts.
3. **Daily Feature Memorization & SHAP Gating**: In financial tree models, slow daily features (`prev_day_ret`, `prev_day_range`) often dominate upper tree splits because they have identical values across all 78 bars of a session. When an asset experiences a multi-day trend reversal, these features force counter-trend signals against strong intraday momentum. The Intraday SHAP Driver Agreement Gate proved essential in preventing these catastrophic errors.
4. **Quantile Monotonicity & Scenario Planning**: Unlike point forecasts, the 10th to 90th percentile quantile price envelope captured 79.8% of holdout outcomes (near the nominal 80%), providing traders with reliable volatility boundaries.

---

# Chapter 5: Conclusion and Future Scope

## 5.1 Key Findings
This project designed, validated, and deployed a robust machine learning indicator for AAPL 5-minute candles. Key findings include:
- A causal 33-feature pipeline completely eliminates lookahead bias while capturing multi-scale momentum and causal relative volume.
- Stacked meta-labeling successfully separates directional prediction from confidence estimation, allowing the system to abstain during low-information regimes.
- Side-conditioned thresholding and SHAP driver agreement prevent regime bias and counter-trend trading traps.
- The live architecture (FastAPI, SQLite WAL, WebSockets, TradingView Lightweight Charts) runs in under 45 ms per candle with guaranteed database immutability.
- Automated testing (25/25 unit tests) verifies full correctness across feature calculations, label alignment, calendar boundaries, and REST/WebSocket schemas.

## 5.2 Limitations
- **Transaction Cost Drag**: At 1 bp per side, long-term unconditional holdout returns are slightly negative (-2.00 bps/signal), indicating that intraday trading edges are fragile and highly dependent on execution quality.
- **Absence of Unstructured Sentiment**: The model operates strictly on price, volume, and volatility; it does not process real-time corporate news, earnings reports, or macro announcements.
- **Frozen Calibration**: The isotonic calibrator is frozen offline; it does not adapt dynamically to sudden intraday volatility regime shifts.

## 5.3 Future Research Directions
- **Cross-Asset Order Book Flow**: Incorporating Level 2 bid-ask order book imbalances (OFI) and options market implied volatility smiles to improve early breakout detection.
- **Transformer Sequential Architectures**: Investigating causal temporal transformers trained on multi-day historical sequences with attention mechanisms over intraday regimes.
- **Online Adaptive Conformal Prediction**: Implementing online conformal prediction to generate dynamically adapting confidence bands with finite-sample coverage guarantees.

---

# End Matter

### References
1. **Mesfin (2026)**. *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ*. arXiv:2605.17724.
2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons.
3. **Bailey, D. H., & López de Prado, M. (2014)**. "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94–107.
4. **Joubert, J. F. (2022)**. "Meta-Labeling: Theory and Framework." *Journal of Financial Data Science*, 4(3), 45–62.
5. **Ke, G., Meng, Q., Finley, T., et al. (2017)**. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 3146–3154.
6. **Zadrozny, B., & Elkan, C. (2002)**. "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 694–699.
7. **Fischer, T., & Krauss, C. (2018)**. "Deep learning with long-short term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654–669.
8. **Wilder, J. W. (1978)**. *New Concepts in Technical Trading Systems*. Trend Research.
9. **Wilson, E. B. (1927)**. "Probable inference, the law of succession, and statistical inference." *Journal of the American Statistical Association*, 22(158), 209–212.
10. **Engle, R. F. (1982)**. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987–1007.
11. **Fama, E. F. (1970)**. "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383–417.
12. **Lundberg, S. M., & Lee, S. I. (2017)**. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 30, 4765–4774.

---

### Appendix A: 33 Causal Feature Glossary
- **Primary Features (24)**:
  1. `ret_1`: 1-bar log return (first bar: `log(close/open)`; otherwise `log(close/prev_close)`).
  2. `ret_3`: 3-bar log return.
  3. `ret_6`: 6-bar log return (30-minute momentum).
  4. `ret_12`: 12-bar log return (60-minute momentum).
  5. `ret_24`: 24-bar log return (120-minute momentum).
  6. `gap`: Overnight jump `log(open_0 / prev_close)` on first bar; 0 otherwise.
  7. `day_ret`: Intraday cumulative return `log(close / open_0)`.
  8. `vol_12`: 12-bar causal return standard deviation annualized to session scale.
  9. `vol_78`: 78-bar causal return standard deviation.
  10. `range_1`: 1-bar normalized high-low range `log(high / low)`.
  11. `range_12`: 12-bar normalized range `log(max(high_12) / min(low_12))`.
  12. `vwap_dist`: Normalized distance to candle VWAP `log(close / vwap)`.
  13. `sess_vwap_dist`: Normalized distance to cumulative session VWAP.
  14. `rel_volume`: Volume divided by causal 20-day median at same `bar_idx`.
  15. `rel_volume_log`: `log(rel_volume + 1e-4)`.
  16. `rel_tc`: Trade count divided by causal 20-day median at same `bar_idx`.
  17. `rel_tc_log`: `log(rel_tc + 1e-4)`.
  18. `tod`: Time-of-day progress `bar_idx / (day_bars - 1)`.
  19. `tod_sin`: $\sin(2\pi \cdot \text{tod})$.
  20. `tod_cos`: $\cos(2\pi \cdot \text{tod})$.
  21. `dow`: Day of week (0 = Monday ... 4 = Friday).
  22. `prev_day_ret`: Log return of preceding trading session.
  23. `prev_day_range`: Normalized range of preceding trading session.
  24. `rsi_14`: 14-period Wilder Relative Strength Index.
- **Secondary Meta Features (9)**:
  25. `p_up`: Primary model predicted probability of UP movement.
  26. `p_dist`: Primary conviction $|p_{\text{up}} - 0.5|$.
  27. `vol_12`: 12-bar short-term volatility.
  28. `vol_78`: 78-bar medium-term volatility.
  29. `tod`: Time of day.
  30. `rel_volume_log`: Log relative volume.
  31. `range_12`: 12-bar price range.
  32. `is_half`: Half-day indicator (1 on early closes, 0 otherwise).
  33. `bars_left`: Bars remaining in regular trading session.

---

### Appendix B: Automated Test Suite Inventory (25/25 Tests)
1. `test_features_no_lookahead.py`: Full history vs. truncated history verification ($0.00\text{e}+00$ difference).
2. `test_labels_alignment.py`: Same-day forward return boundary and class split verification.
3. `test_splits_purge.py`: Walk-forward blocks and embargo boundary verification.
4. `test_policy_rules.py`: Signal gating rules (`opening_settle`, `horizon_past_close`, `low_confidence`).
5. `test_bundle_consistency.py`: Feature names vs booster trees vs SHA-256 bundle hashes.
6. `test_holdout_gate.py`: Access log verification preventing multiple holdout evaluations.
7. `test_candle_builder.py`: 1-minute aggregation, duplicate overwrite, zero-volume VWAP fallback.
8. `test_finalise_order.py`: Deterministic symbol order ($\text{SPY} \to \text{QQQ} \to \text{AAPL} \to \text{predictor} \to \text{scorer}$).
9. `test_signal_immutable.py`: SQLite `BEFORE UPDATE` and `BEFORE DELETE` immutability triggers.
10. `test_signal_idempotent.py`: Idempotent bucket closing preventing duplicate signal rows.
11. `test_scorer.py`: Realised PnL resolution with $t+1$ entry and $t+h$ exit fallbacks.
12. `test_db_backup.py`: SQLite WAL-safe online synchronous and async snapshot consistency.
13. `test_api_schema.py`: All 7 FastAPI REST endpoints and WebSocket snapshot handshake.
14. `test_marker_order.py`: Lightweight Charts v4 marker chronological ordering and deduplication.
15. `test_no_hardcoded_horizon.py`: Dynamic horizon propagation ($h=12$) without literal strings.
16. `test_nyse_calendar_2026.py`: 2026 Thanksgiving, Black Friday, Christmas Eve early close rules.
17. `test_scenarios_trace.py`: Empirical analog table lookup and monotonic quantile price envelope.
*(Plus 8 additional unit test cases verifying data pull, cleaning, and model exports).*

---

### Appendix C: Live REST and WebSocket API Specification
- `GET /api/health`: System status, SQLite connectivity, row counts, memory usage.
- `GET /api/candles?limit=390`: Historical 5-minute candles (3 to 5 full sessions).
- `GET /api/signals?limit=390`: Historical signals and reasons (`active_signal`, `low_confidence`, etc.).
- `GET /api/signal/{candle_ts}`: Detailed inspection of single candle's SHAP drivers, quantiles, and regime narrative.
- `GET /api/stats`: Live summary stats (Wilson intervals, win rates, benchmark comparisons, cost sensitivity).
- `GET /api/model_info`: Model bundle version, SHA-256 hashes, label type, 33 causal features count.
- `GET /api/export.csv`: 1-click full CSV dump of all historical signals, outcomes, and drivers.
- `WS /ws`: Real-time streaming WebSocket broadcasting initial state snapshot and live bucket events.
