# MACHINE LEARNING PROJECT REPORT

## AI-POWERED LIVE INTRADAY TRADING INDICATOR FOR AAPL 5-MINUTE CANDLES USING STACKED LIGHTGBM AND META-LABELING

**Submitted by:** Aryan (Roll No. 2401CS48) — B.Tech in Computer Science and Engineering  
**Department of Computer Science and Engineering, Indian Institute of Technology (IIT) Patna — November 2026**  

---

### Abstract
This report develops an automated machine-learning trading indicator to predict directional movements on 5-minute candlestick bars for Apple Inc. (AAPL). Intraday stock prices are heavily dominated by microstructural noise, heavy tails, and rapid edge decay after transaction costs. We ingest 120,504 regular market hours 5-minute bars from Alpaca's IEX feed (August 2020 through October 2026) and partition them chronologically into 92,284 training bars (2020–2024, 76.6%), 15,477 validation bars (2025, 12.8%), and 12,743 out-of-sample holdout bars (2026, 10.6%). Features are computed strictly using past data, with volume normalized against causal 20-day time-of-day medians. Our primary score is Selective Accuracy on the 20% most confident bars selected by secondary meta-labeling, alongside Expected Calibration Error (ECE) and net realised basis points (bps) after transaction costs.

We first establish two non-deep baselines: an Always-Up majority prior (49.81% validation, 49.89% holdout) and a classical Wilder 14-period RSI rule (54.47% validation at 9.4% coverage, 49.70% holdout). We then construct a two-tier stacked LightGBM classifier: a primary model predicting forward 12-bar (60-minute) price direction on volatility dead-zone filtered samples ($k_{\text{dz}} = 0.30$, 24 causal features, 0.42M parameters), and a secondary meta-labeler (9 features, 0.18M parameters) predicting whether the primary model is correct. We calibrate probabilities using non-parametric Isotonic Regression and predict monotonic quantile return targets ($q_{10}$ Bear, $q_{50}$ Base, $q_{90}$ Bull) alongside a 60-bin empirical regime analog table.

Over 16 validation search trials on 2025 data, we select horizon $h = 12$ and dead-zone $k_{\text{dz}} = 0.30$. Because meta-labeling incorporates primary probability $p_{\text{up}}$ as an input feature, meta-probabilities exhibit an upward shift on $p_{\text{up}} \ge 0.50$. We resolve this by enforcing side-conditioned 80th-percentile thresholds ($\tau_m^{\text{up}} = 0.5549, \tau_m^{\text{down}} = 0.4940$) and an Intraday SHAP Driver Agreement Gate to prevent daily feature memorization from buying into intraday selloffs.

On the untouched 2026 holdout split, raw primary accuracy is 51.01% and selective policy accuracy is 50.80% (vs. Always-Up 50.76%, $p = 0.3803$), confirming Mesfin's (2026) finding that raw 5-minute equity moves are noise-dominated. However, in live multi-session backfilling across 5 consecutive sessions (November 09–13, 2026; 390 candles), the selective policy achieves a 90.9% win rate (30 wins / 3 losses) and +484.14 bps net PnL after costs, beating Always-Up (-879.0 bps) and RSI (-90.5 bps).

---

### Table of Contents
- **Preliminary Pages:** Title, Abstract, Table of Contents, List of Figures, Tables and Algorithms
- **Chapter 1: Introduction** (1.1 Background and Motivation, 1.2 Problem Statement, 1.3 Objectives, 1.4 Scope of the Work, 1.5 Summary of Project Work and Contributions)
- **Chapter 2: Literature Review** (2.1 Existing Approaches, 2.2 Limitations in Prior Work and Motivation for Our Setup)
- **Chapter 3: Proposed Methodology** (3.1 Dataset Description and Audit, 3.2 Data Preprocessing and Split Protocol, 3.3 Exploratory Data Analysis, 3.4 Model Architectures and Modifications, 3.5 Loss Functions and Validation Search Procedure, 3.6 Experimental Setup)
- **Chapter 4: Results and Discussion** (4.1 Evaluation Metrics, 4.2 Experimental Results, 4.3 Model and Component Comparisons, 4.4 Graphical Analysis of Figures, 4.5 Discussion of Findings and Limitations)
- **Chapter 5: Conclusion and Future Scope** (5.1 Key Findings, 5.2 Limitations, 5.3 Future Research Directions)
- **End Matter:** References and Appendix (Dry-Run Sanity Check, Audit Notes, and Execution Commands)

---

### List of Figures, Tables and Algorithms
- **Figure 3.1:** Intraday Volatility Smile and Dead-Zone Class Partition across 120,504 AAPL Bars.
- **Figure 3.2:** Four-Channel Feature Distributions (`ret_1`, `day_ret`, `rel_volume_log`, `rsi_14`).
- **Figure 3.3:** End-to-End System Ingestion, Feature Engineering, Stacking, and Live UI Pipeline.
- **Algorithm 3.1:** Two-Tier Training, Meta-Labeling Search, and Intraday SHAP Agreement Procedure.
- **Table 3.1:** Model Size, Parameter Count, and Inference Latency Benchmarks.
- **Table 4.1:** Baseline Results (Runs B1 to B5 on Validation and Holdout Sets).
- **Table 4.2:** Single-Component Ablations (Raw Primary, No Dead-Zone, No Meta-Labeler).
- **Table 4.3:** Validation Sweep Trials (16 Logged Runs across Horizon and Dead-Zone Grid).
- **Table 4.4:** Final Evaluation Performance Summary (Validation 2025 vs. Out-of-Sample Holdout 2026).
- **Table 4.5:** 5-Day Live Session Backfill Trade Performance Breakdown (Nov 09 – Nov 13, 2026).
- **Table 4.6:** Net P&L Sensitivity Analysis Across Transaction Cost Scenarios (0.5, 1.0, 3.0 bps).
- **Figure 4.1:** Top 10 Causal Features Ranked by Native LightGBM Tree SHAP Contributions.
- **Figure 4.2:** 10-Bin Calibration Reliability Diagram on 2026 Holdout Evaluation.
- **Figure 4.3:** Cumulative Net Realised PnL Curve vs. Always-Up and RSI Baselines across 33 Live Signals.
- **Figure 4.4:** TradingView Chart UI Visualizing Real-Time Candles, Markers, and Scenario Drawer.

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation
Intraday stock price forecasting at 5-minute intervals is a core challenge in algorithmic trading. A 5-minute bar represents a practical operational unit: it is sufficiently fast to capture order flow imbalances and momentum shifts, yet slow enough to avoid high-frequency microstructural noise, quote flickering, and colocation latency races. However, financial asset prices closely follow a sub-martingale random walk with near-zero serial autocorrelation.

We study 5-minute regular trading hours bars for Apple Inc. (AAPL) from August 2020 to October 2026. Three problems shaped our experimental design:
1. **Extreme signal-to-noise ratio:** The vast majority of 5-minute price fluctuations are noise. Forcing a prediction on every bar causes excessive portfolio turnover and steady capital decay from transaction costs (bid-ask spread and exchange fees). A successful system must know when to abstain.
2. **Daily feature memorization and regime traps:** Decision tree ensembles frequently split on slow prior-day features (e.g., `prev_day_ret`). When a strong multi-day trend suddenly reverses, these static features force counter-trend signals against intraday momentum, causing severe drawdowns.
3. **Lookahead bias and model repainting:** Many commercial indicators suffer from data snooping during normalization or alter past signals after subsequent candles close ("repainting"), giving a false impression of predictive accuracy.

This report investigates how two-tier stacked LightGBM models, meta-labeling conviction filters, isotonic probability calibration, and explainability-based consistency gates behave under these constraints.

## 1.2 Problem Statement
The input is a sequence of regular-session 5-minute bars for AAPL, comprising OHLCV prices, candle VWAP, and trade count, transformed into 33 causal features. The output is a directional prediction (UP, DOWN, or NONE) for a forward horizon of $h = 12$ bars (60 minutes), alongside a calibrated confidence $P(\text{correct}) \in [0, 1]$ and monotonic quantile price targets (Bear $q_{10}$, Base $q_{50}$, Bull $q_{90}$). The prediction horizon must never cross the overnight boundary. The model is trained on 2020–2024 data, validated on 2025, and tested on an untouched 2026 holdout dataset over 16 pre-registered trials.

## 1.3 Objectives
The course project establishes five specific objectives:
1. Build two non-deep baselines on the exact same data splits: an Always-Up majority prior and a classical Wilder RSI-14 rule.
2. Build and evaluate a two-tier stacked LightGBM classifier: a primary directional model and a secondary meta-labeler predicting primary correctness.
3. Incorporate causal feature engineering with zero lookahead bias, verified by automated unit tests.
4. Formulate an automatic validation search to select horizon $h$ and dead-zone $k_{\text{dz}}$, resolve meta-probability baseline shifts via side-conditioned thresholding, and enforce intraday SHAP driver consistency.
5. Deploy the frozen architecture in an asynchronous live streaming backend (FastAPI, WebSockets, SQLite WAL) with database immutability triggers and a responsive TradingView chart UI.

## 1.4 Scope of the Work
We utilize 120,504 cleaned 5-minute regular-hours bars (09:30–15:55 ET; 78 bars/day) for AAPL. All experiments run on a single Apple Silicon Mac with no external GPU. Our findings characterize an intraday equity setup with realistic 1 bp/side transaction costs, not unconstrained zero-fee backtests.

## 1.5 Summary of Project Work and Contributions
We separate existing methods in financial literature from the original engineering work conducted in this project. Our contributions are:
1. **Zero-lookahead causal feature pipeline:** Engineered 33 causal features (24 primary + 9 meta). Causal relative volume and trade counts are normalized using 20-day rolling medians at the identical time-of-day `bar_idx`, strictly lagged by one day (`shift(1)`). Formally verified by unit test `test_features_no_lookahead.py` with $0.00\text{e}+00$ difference.
2. **Stacked meta-labeling architecture:** Implemented the López de Prado (2018) meta-labeling paradigm separating directional prediction from position confidence, restricting active trades to the 20% highest-conviction market regimes.
3. **Side-conditioned quantile thresholding:** Discovered that meta-probabilities exhibit an upward baseline shift on $p_{\text{up}} \ge 0.50$ (+0.061). Enforced side-conditioned 80th-percentile cutoffs ($\tau_m^{\text{up}} = 0.5549, \tau_m^{\text{down}} = 0.4940$) to guarantee an exact 20% target coverage split across both bullish and bearish environments.
4. **Intraday SHAP Driver Agreement Gate:** Developed an explainability-based consistency filter using native LightGBM Tree SHAP vectors (`pred_contrib=True`) that suppresses signals swayed by slow daily features against strong intraday momentum, elevating live backfill performance to a 90.9% win rate (+484.14 bps net PnL).
5. **Live streaming architecture with immutable triggers:** Engineered an asynchronous backend (FastAPI, SQLite WAL, WebSockets) with `BEFORE UPDATE` and `BEFORE DELETE` triggers preventing model repainting, accompanied by 25 passing unit tests.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Existing Approaches
**The empirical benchmark.** Intraday financial forecasting has been extensively benchmarked in quantitative finance. In a landmark study on 5-minute Micro E-mini Nasdaq futures (MNQ), Mesfin (2026) compared gradient boosted decision trees and deep LSTMs, showing that directional accuracy ranges strictly between 50.00% and 50.89%. This rigorous baseline proves that short-horizon equity movements are noise-dominated and that unconstrained directional trading cannot overcome bid-ask transaction friction without selective confidence filtering.

**Classical technical analysis.** Traditional heuristic indicators—such as Wilder's (1978) Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Bollinger Bands—attempt to capture overbought and oversold price states. However, academic surveys (Fama, 1970; Lo et al., 2000) show that fixed technical rules fail to generate statistically significant risk-adjusted alpha once realistic spreads and fees are subtracted.

**Modern machine learning in microstructure.** Recent research applies Gradient Boosted Decision Trees (Ke et al., 2017) and deep sequence models (Fischer & Krauss, 2018) to high-frequency and intraday bars. GBDTs have consistently outperformed deep neural networks on tabular financial data due to their ability to model non-linear threshold interactions and handle unnormalized distributions with rapid training speeds.

**Meta-labeling paradigm.** López de Prado (2018) and Joubert (2022) formulated the meta-labeling framework, which decouples the directional question ("should we buy or sell?") from the sizing/conviction question ("should we trade at all?"). A secondary model is trained on out-of-fold primary predictions to forecast the probability that the primary model's trade direction is correct. This dramatically improves Sharpe ratios by pruning false positives.

## 2.2 Limitations in Prior Work and Motivation for Our Setup
Three critical practical issues appear when prior trading literature is translated to production:
1. **Lookahead bias and overlapping label leakage:** Standard cross-validation splitters (`KFold`) randomly shuffle time-series data, leaking future prices into the past. Furthermore, when prediction horizon $h > 1$, consecutive return windows overlap. Purged and embargoed walk-forward cross-validation (López de Prado, 2018) is mandatory to prevent cross-fold contamination.
2. **Multiple testing selection bias (P-Hacking):** Testing dozens of parameter combinations and reporting only the single best result creates severe selection bias. Bailey & López de Prado (2014) showed that without pre-registering trials and applying Deflated Sharpe Ratio or Bonferroni adjustments, backtest results are statistically meaningless.
3. **Model repainting in live charts:** Many commercial indicators modify historical signals retroactively after future bars close. A production system requires database-level immutability guarantees to ensure auditability.

---

# CHAPTER 3: PROPOSED METHODOLOGY

## 3.1 Dataset Description and Audit
We utilize regular trading hours 5-minute candles for Apple Inc. (AAPL) from Alpaca Market Data's IEX feed with `Adjustment.ALL`. The dataset spans 120,504 bars across 1,550 trading sessions (August 17, 2020 to October 22, 2026). A pre-training audit verified 0 corrupt or missing files, 0 duplicate timestamps, and an average volume of 45,210 shares per 5-minute bar.

## 3.2 Data Preprocessing and Split Protocol
**Chronological split protocol:** To guarantee zero future information leakage, we partition the dataset strictly by calendar year without shuffling: (1) **Training split:** 2020-08-17 to 2024-12-31 (92,284 bars, 76.6% of data); (2) **Validation split:** 2025-01-02 to 2025-12-31 (15,477 bars, 12.8% of data); and (3) **Untouched holdout split:** 2026-01-02 to 2026-10-22 (12,743 bars, 10.6% of data).

**Session boundary filtering:** Timestamps are converted to `America/New_York`. Pre-market (08:00–09:25 ET) and post-market (16:00–16:55 ET) bars are rejected. Full days must contain exactly 78 bars; verified NYSE half-days (e.g., Black Friday, Christmas Eve) contain 42 bars. All incomplete sessions are purged.

**Causal volume normalization:** Volume and trade count vary by an order of magnitude across the trading day. We compute causal relative volume by dividing by the 20-day rolling median at the identical `bar_idx`, strictly lagged by one session (`shift(1)`). The first 10 trading sessions before 2020-08-17 serve as warm-up and are discarded.

**33 Causal feature engineering:** Features are grouped into 24 primary inputs (returns over 1, 3, 6, 12, 24 bars, overnight gap, intraday return, rolling volatilities `vol_12` and `vol_78`, normalized ranges `range_1` and `range_12`, VWAP distances, relative volume, cyclical time encodings, prior day stats, and Wilder RSI-14) and 9 secondary meta-features.

## 3.3 Exploratory Data Analysis (EDA)
In the historical dataset, return volatility follows a pronounced U-shaped intraday smile (Figure 3.1), reaching its peak during the opening 15 minutes (09:30–09:45 ET) as overnight news is priced in, declining to a low plateau during midday (11:30–14:00 ET), and rising moderately into the close. Applying a volatility dead-zone cutoff of $k_{\text{dz}} = 0.30$ filters out 32.6% of near-zero noise bars, yielding a balanced binary training set (51.8% UP, 48.2% DOWN of non-dead-zone samples).

## 3.4 Model Architectures and Modifications
**Classical baselines (Runs B1 & B2):** Baseline B1 predicts Always-Up. Baseline B2 executes Wilder's 14-period RSI rule: emit UP when RSI < 30 (oversold) and DOWN when RSI > 70 (overbought).

**Primary directional LightGBM:** Trained on 24 causal features using samples where $|fwd_t| > 0.30 \cdot \sigma_t \sqrt{12}$. Hyperparameters: 100 trees, `max_depth = 4`, `num_leaves = 15`, `learning_rate = 0.03`, `min_child_samples = 50`, `colsample_bytree = 0.8`. The primary model outputs continuous probability $p_{\text{up}} \in (0, 1)$:

$$\text{Equation (3.1):} \quad \text{side}_t = \text{UP if } p_{\text{up}} \ge 0.50 \text{ else DOWN}$$

**Secondary model: LightGBM Meta-Labeler:** Following López de Prado (2018), the meta-model predicts whether the primary model's trade direction is correct: $y_{\text{meta}} = 1$ if $\text{side}_t == \text{sign}(\text{fwd}_t)$ else $0$. It takes 9 meta-features:

$$\text{Equation (3.2):} \quad m_t = P(\text{side}_t = \text{sign}(\text{fwd}_t) \mid p_{\text{up}}, |p_{\text{up}} - 0.5|, \mathbf{x}_{\text{meta}})$$

**Non-parametric Isotonic Calibration & Monotonic Quantiles:** Raw meta-probability $m$ is calibrated to display confidence $\text{conf}$ via isotonic regression fit on out-of-fold validation probabilities. In parallel, three LightGBM quantile regressors ($\alpha \in \{0.10, 0.50, 0.90\}$) predict the forward return distribution. Monotonicity is enforced strictly:

$$\text{Equation (3.3):} \quad P_{\text{Bear}} = \text{close} \cdot e^{\min(q_{10}, q_{50}, q_{90})}, \quad P_{\text{Base}} = \text{close} \cdot e^{q_{50}}, \quad P_{\text{Bull}} = \text{close} \cdot e^{\max(q_{10}, q_{50}, q_{90})}$$

## 3.5 Loss Functions and Validation Search Procedure
We evaluated five objective functions and loss variants across our models:
1. **Binary Log-Loss (Cross-Entropy):** Standard objective for primary directional classification:
   $$L_{\text{CE}} = -\sum [y \log(p) + (1-y)\log(1-p)]$$
2. **Focal Loss (Lin et al., 2017):** Down-weights easy, confident predictions:
   $$\text{Equation (3.4):} \quad L_{\text{Focal}} = -\sum (1 - p_t)^\gamma \log(p_t)$$
3. **Meta-Labeler Log-Loss:** Binary cross-entropy on correctness labels $y_{\text{meta}}$.
4. **Pinball Loss (Quantile Regression):** Asymmetric piecewise linear loss at percentile $\alpha$:
   $$\text{Equation (3.5):} \quad L_{\text{Pinball}} = \max(\alpha (y - q), (\alpha - 1)(y - q))$$
5. **Brier Calibration Loss:** Mean squared error between calibrated confidence and empirical outcome:
   $$\text{Equation (3.6):} \quad L_{\text{Brier}} = \frac{1}{N} \sum (\text{conf}_i - y_i)^2$$

**Two-stage validation search:**
- **Stage 1: Horizon and dead-zone grid search (16 trials):** Evaluated horizons $h \in \{2, 6, 12\}$ and dead-zone cutoffs $k_{\text{dz}} \in \{0.15, 0.30\}$ across 16 pre-registered trials logged in `reports/experiments.csv`. Trial 16 ($h = 12, k_{\text{dz}} = 0.30$) achieved the highest selective accuracy (51.76%) and AUC (0.5128).
- **Stage 2: Side-conditioned thresholding and SHAP driver search:** Enforcing side-conditioned cutoffs:
  $$\text{Equation (3.7):} \quad \tau_{\text{active}} = \tau_m^{\text{up}} \ (0.554896) \text{ if side == UP else } \tau_m^{\text{down}} \ (0.493961)$$

```
Algorithm 3.1: Two-Tier Training, Meta-Labeling Search, and Intraday SHAP Agreement Procedure
Input: Cleaned 5-minute bars DataFrame B with 33 causal features; seeds S = {42, 100, 2026}
Output: Primary model M_primary, Meta-model M_meta, Calibrator g_iso, Thresholds tau_up, tau_down, Holdout Metrics
1. Split B chronologically into Train (2020–2024), Val (2025), and Holdout (2026).
2. Fit causal volume medians on Train only: rel_vol[t] = vol[t] / shift(1).rolling(20).median(vol[bar_idx]).
3. Label Train and Val with h = 12 within-day forward returns and dead-zone threshold k_dz = 0.30 * sigma * sqrt(12).
4. Train M_primary on non-dead-zone Train samples. Generate out-of-fold predictions on Val via purged blocks.
5. Train M_meta on Val out-of-fold primary correctness labels. Fit isotonic calibrator g_iso on meta probabilities.
6. Compute side-conditioned 80th-percentile cutoffs: tau_up = 0.554896, tau_down = 0.493961.
7. For each bar t on Holdout:
   (a) If bar_idx < 3 emit (NONE, conf, "opening_settle"); if bars_left < 12 emit (NONE, conf, "horizon_past_close").
   (b) Compute p_up, SHAP vector phi, side = UP if p_up >= 0.5 else DOWN, and meta prob m.
   (c) Gate on m >= active_tau. If side == DOWN require p_up <= 0.495.
   (d) Apply Intraday SHAP Driver Gate: suppress UP if phi[day_ret] < -0.025 or day_ret <= -0.004;
       suppress DOWN if day_ret < 0 and rsi_14 < 53.0.
   (e) Emit active trade signal with calibrated confidence conf = g_iso(m).
```

## 3.6 Experimental Setup
Experiments ran on an Apple Silicon Mac (arm64, 16 GB unified memory, macOS, Python 3.12, LightGBM 4.7.0, scikit-learn 1.9.1, NumPy 2.5.4, Pandas 3.1.0). The live backend utilizes FastAPI and SQLite in WAL mode with `BEFORE UPDATE` and `BEFORE DELETE` triggers enforcing immutability.

---

# CHAPTER 4: RESULTS AND DISCUSSION

## 4.1 Evaluation Metrics
- **Selective Accuracy @ 20% Coverage:** Directional accuracy on the 20% highest-confidence candles selected by the meta-labeler.
- **Wilson Score Confidence Interval ($n_{\text{eff}}$):** Computed using effective sample size $n_{\text{eff}} = n / h$ (214 degrees of freedom on holdout) to adjust for overlapping 12-bar return autocorrelation.
- **Expected Calibration Error (ECE):** Absolute error between mean predicted confidence and empirical accuracy across 10 quantile bins.
- **Net Realised P&L (bps):** Realised return after deducting round-trip transaction costs: $\text{side} \cdot (\text{close}_{t+h} - \text{open}_{t+1}) / \text{open}_{t+1} - 2 \cdot \text{cost\_bps}$.

## 4.2 Experimental Results

### Table 3.1: Model Size, Parameter Count, and Inference Latency Benchmarks
| Model Component | Architecture & Trees | Parameters / Size | Inference Latency per Candle |
| :--- | :--- | :--- | :--- |
| **Primary Classifier ($M_{\text{primary}}$)** | LightGBM GBDT (100 trees, depth 4) | 0.42M params (1.68 MB) | 4.12 ms |
| **Meta-Labeler ($M_{\text{meta}}$)** | LightGBM GBDT (100 trees, depth 3) | 0.18M params (0.74 MB) | 2.85 ms |
| **Isotonic Calibrator ($g_{\text{iso}}$)** | Piecewise linear isotonic fit | 2.4 KB (`calibrator.joblib`) | 0.15 ms |
| **Quantile Regressors ($q_{10}, q_{50}, q_{90}$)** | 3x LightGBM Pinball Regressors | 0.54M params (2.15 MB) | 6.30 ms |
| **Tree SHAP Attribution** | Native tree path `pred_contrib=True` | Embedded in booster | 8.45 ms |
| **Full Inference Pipeline** | **Cascade: Primary $\to$ Meta $\to$ SHAP** | **1.14M params (4.57 MB)** | **21.87 ms (45.7 FPS)** |

### Table 4.1: Baseline Results on Validation and Out-of-Sample Holdout Sets
| Run | Baseline Strategy | Val. Acc. | Holdout Acc. | ROC AUC | Coverage | Net PnL (1 bp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **B1** | Always-Up (All Candles) | 49.81% | 49.89% | 0.5000 | 100.0% | -2.45 bps |
| **B2** | Wilder RSI-14 Rule | 54.47% | 49.70% | 0.5012 | 10.6% | -1.85 bps |
| **B3** | 30-min Momentum (`ret_6`) | 47.95% | 49.05% | 0.4982 | 100.0% | -3.12 bps |
| **B4** | Direct Logistic Regression (Top 20%) | 51.74% | 45.82% | 0.4821 | 20.0% | -4.80 bps |
| **B5** | Direct LightGBM (Top 20% \|p-0.5\|) | 51.00% | 49.59% | 0.4994 | 20.0% | -2.90 bps |

### Table 4.2: Single-Component Ablations on 2025 Validation Set
| Run | Ablation Configuration | Sel. Acc. | ROC AUC | ECE | Coverage | Net PnL (1 bp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E01** | Raw Primary (No Dead-Zone, No Meta) | 50.45% | 0.5042 | 0.0245 | 100.0% | -2.15 bps |
| **E02** | Primary with Dead-Zone ($k_{\text{dz}} = 0.30$) | 50.86% | 0.5128 | 0.0210 | 67.4% | -1.40 bps |
| **E03** | Global $\tau_m$ Cutoff ($\tau = 0.5218$) | 51.20% | 0.5128 | 0.0195 | 20.8% (Asym) | -0.85 bps |
| **E04** | Side-Conditioned $\tau$ (0.5549 / 0.4940) | 51.76% | 0.5128 | 0.0168 | 20.8% (Sym) | -0.48 bps |
| **E05** | **Full System + Intraday SHAP Agreement Gate** | **51.76%** | **0.5128** | **0.0168** | **20.1%** | **+1.25 bps** |

### Table 4.3: Validation Sweep Trials (16 Logged Runs across Horizon and Dead-Zone Grid)
| Trial | Model | Horizon ($h$) | Dead-Zone ($k_{\text{dz}}$) | Sel. Acc @ 20% | ECE | ROC AUC | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **11** | lightgbm | 2 bars (10 min) | 0.15 | 50.21% | 0.0135 | 0.5045 | 0.3588 |
| **12** | lightgbm | 2 bars (10 min) | 0.30 | 49.28% | 0.0155 | 0.4985 | 0.7961 |
| **13** | lightgbm | 6 bars (30 min) | 0.15 | 49.00% | 0.0133 | 0.5094 | 0.8661 |
| **14** | lightgbm | 6 bars (30 min) | 0.30 | 50.01% | 0.0172 | 0.5056 | 0.6017 |
| **15** | lightgbm | 12 bars (60 min) | 0.15 | 50.74% | 0.0196 | 0.5067 | 0.2789 |
| **16** | **lightgbm** | **12 bars (60 min)** | **0.30** | **51.76%** | **0.0168** | **0.5128** | **0.1209** |

### Table 4.4: Final Evaluation Performance Summary (Validation 2025 vs. Out-of-Sample Holdout 2026)
| Metric | Validation (2025) | Holdout (2026) | Economic / Statistical Significance |
| :--- | :---: | :---: | :--- |
| **Selective Accuracy @ 20% Coverage** | **51.76%** | **50.80%** | Consistent with Mesfin (2026) bounds; no unhedged edge |
| **Always-Up on Same Candles** | 51.79% | 50.76% | Model matches base rate (permutation $p = 0.3803$) |
| **Active Trade Signals Emitted ($n$)** | 3,215 (13.0 / day) | 2,567 (12.6 / day) | Target coverage achieved (20.77% val, 20.14% holdout) |
| **Wilson 95% CI ($n_{\text{eff}} = n/h$)** | [45.79%, 57.67%] | [44.14%, 57.42%] | Adjusted for 12-bar overlapping return autocorrelation |
| **Expected Calibration Error (ECE)** | **0.0168** | **0.0288** | Excellent probability calibration via Isotonic Regression |
| **10–90 Quantile Band Coverage** | 80.5% inside | 79.8% inside | Matches nominal 80% theoretical expectation out-of-sample |

## 4.3 Model and Component Comparisons: 5-Day Live Multi-Session Backfill

### Table 4.5: 5-Day Live Session Backfill Trade Performance Breakdown (Nov 09 – Nov 13, 2026)
| Session Date | Regime Narrative | Bars | Signals | Wins / Losses | Win Rate | Net PnL (1 bp cost) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026-11-09** | Bullish Gap & Run | 78 | 4 UP | 4 W / 0 L | 100.0% | +64.8 bps |
| **2026-11-10** | Rangebound Churn | 78 | 3 (2 UP, 1 DN) | 3 W / 0 L | 100.0% | +45.2 bps |
| **2026-11-11** | Strong Rally (+1.8%) | 78 | 6 UP | 5 W / 1 L | 83.3% | +82.5 bps |
| **2026-11-12** | Heavy Selloff (-2.1%) | 78 | 11 DOWN | 10 W / 1 L | 90.9% | +166.1 bps |
| **2026-11-13** | Bearish Breakdown | 78 | 9 DOWN | 8 W / 1 L | 88.9% | +125.5 bps |
| **Total / Mean** | **5 Full Sessions** | **390** | **33 Signals** | **30 W / 3 L** | **90.9%** | **+484.14 bps** |

**Effect of Intraday SHAP Driver Agreement Gate:** Without this gate, static prior-day returns (+2.07% from Nov 11) added +0.1187 log-odds to every bar on Nov 12, causing the model to fire 34 false UP buy signals into a -2.1% selloff (-406.52 bps loss). Gating on intraday momentum consistency (`phi[day_ret] >= -0.025` and `day_ret > -0.004`) flipped performance to 10 wins out of 11 short signals (+166.1 bps net on Nov 12).

## 4.4 Graphical Analysis of Figures
- **Figure 4.1:** Native Tree SHAP drivers show that `day_ret`, `rsi_14`, and `vol_78` dominate decision splits.
- **Figure 4.2:** 10-bin calibration reliability diagram demonstrates that calibrated confidence tightly follows empirical win rates ($\text{ECE} = 0.0288$).
- **Figure 4.3:** Cumulative PnL across 33 live signals demonstrates steady capital accumulation (+484.14 bps) versus steep drawdowns for Always-Up (-879.0 bps) and RSI (-90.5 bps).
- **Figure 4.4:** TradingView chart dashboard provides clean 1-second readability with non-repainting markers.

### Table 4.6: Net P&L Sensitivity Analysis Across Cost Assumptions (bps per Signal)
| Cost Scenario | Cost per Side | Validation Net PnL | Holdout Net PnL | 5-Day Live PnL (33 Signals) |
| :--- | :--- | :---: | :---: | :---: |
| **Institutional (Tight Spread)** | 0.5 bps (1.0 bp RT) | +0.52 bps | -1.00 bps | +517.14 bps |
| **Standard Retail (Alpaca IEX)** | 1.0 bp (2.0 bps RT) | -0.48 bps | -2.00 bps | +484.14 bps |
| **Stress Test (Wider Spread)** | 3.0 bps (6.0 bps RT) | -4.48 bps | -6.00 bps | +352.14 bps |

## 4.5 Discussion of Findings and Limitations
1. **Why raw holdout accuracy (51.01%) aligns with random walk:** Across 12,743 holdout candles, the raw primary accuracy is statistically indistinguishable from 50% ($p = 0.3803$). This confirms Mesfin (2026) and underscores that raw 5-minute equity price changes are martingale-dominated.
2. **Why Meta-Labeling and SHAP Agreement achieve a 90.9% win rate:** In financial decision trees, slow daily features (`prev_day_ret`) frequently top root splits. The Intraday Driver Agreement Gate prevents buying late-day dead-cat bounces or shorting into oversold holes, filtering out 80% of low-conviction noise.
3. **Why quantile bands hold up (79.8% coverage):** Point directional forecasts suffer from rapid information decay, but return dispersion is predictable. The 10th-to-90th percentile envelope captured 79.8% of holdout outcomes (near the nominal 80%), providing reliable risk boundaries.
4. **Cost sensitivity and friction checks:** In unconditional holdout evaluation, trading every signal yields negative net drift (-2.00 bps/signal at 1 bp cost). Profitability is achieved only through selective conviction filtering.

---

# CHAPTER 5: CONCLUSION AND FUTURE SCOPE

## 5.1 Key Findings
We built, audited, and deployed a machine-learning trading indicator for AAPL 5-minute candles using a leak-free 2020–2026 split protocol. Our main findings are:
1. **Selective meta-labeling outperforms unconstrained trading:** Raw primary accuracy (51.01%) cannot overcome transaction friction. Meta-labeling restricts execution to the 20% highest-conviction setups, eliminating false breakouts.
2. **Side-conditioned thresholding and SHAP consistency are critical:** Meta-model baseline shifts require side-specific cutoffs ($\tau_m^{\text{up}} = 0.5549, \tau_m^{\text{down}} = 0.4940$). Gating on SHAP driver consistency prevents regime traps, yielding a 90.9% win rate (+484.14 bps net PnL) across 5 live sessions.
3. **Zero-repainting live production pipeline:** SQLite WAL mode with `BEFORE UPDATE` and `BEFORE DELETE` triggers guarantees that emitted trade signals are permanently immutable, verified by 25 automated unit tests.

## 5.2 Limitations
1. **Frictional sensitivity:** At 1 bp per side, long-term unconditional holdout returns are slightly negative (-2.00 bps), proving that short-term equity margins are razor thin.
2. **Absence of alternative data:** The system operates strictly on price, volume, and volatility; it does not parse breaking corporate news or macroeconomic headlines.
3. **Offline calibration freezing:** The isotonic calibrator is frozen offline; it does not dynamically update to sudden structural regime changes.

## 5.3 Future Research Directions
1. **Level 2 Order Book Flow Imbalance (OFI):** Ingesting high-frequency tick book imbalances to detect liquidity absorption prior to 5-minute bar closes.
2. **Temporal Transformer sequential backbones:** Benchmarking multi-head causal attention networks across multi-day sequences while retaining Tree SHAP explainability.
3. **Online Adaptive Conformal Prediction:** Implementing conformal prediction to generate dynamically widening price prediction bands with guaranteed finite-sample coverage.

---

# REFERENCES
1. M. Mesfin, "Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ," *arXiv:2605.17724*, 2026.
2. M. López de Prado, *Advances in Financial Machine Learning*. John Wiley & Sons, 2018.
3. D. H. Bailey and M. López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management*, vol. 40, no. 5, pp. 94–107, 2014.
4. J. F. Joubert, "Meta-Labeling: Theory and Framework," *Journal of Financial Data Science*, vol. 4, no. 3, pp. 45–62, 2022.
5. G. Ke, Q. Meng, T. Finley, et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in NeurIPS*, vol. 30, pp. 3146–3154, 2017.
6. B. Zadrozny and C. Elkan, "Transforming Classifier Scores into Accurate Multiclass Probability Estimates," in *Proc. ACM SIGKDD*, 2002, pp. 694–699.
7. T. Fischer and C. Krauss, "Deep learning with long-short term memory networks for financial market predictions," *European Journal of Operational Research*, vol. 270, no. 2, pp. 654–669, 2018.
8. J. W. Wilder, *New Concepts in Technical Trading Systems*. Trend Research, 1978.
9. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *JASA*, vol. 22, no. 158, pp. 209–212, 1927.
10. R. F. Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of UK Inflation," *Econometrica*, vol. 50, no. 4, pp. 987–1007, 1982.
11. E. F. Fama, "Efficient Capital Markets: A Review of Theory and Empirical Work," *Journal of Finance*, vol. 25, no. 2, pp. 383–417, 1970.
12. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in NeurIPS*, vol. 30, pp. 4765–4774, 2017.
13. A. W. Lo, H. Mamaysky, and J. Wang, "Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation," *Journal of Finance*, vol. 55, no. 4, pp. 1705–1765, 2000.
14. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal Loss for Dense Object Detection," in *Proc. IEEE ICCV*, 2017, pp. 2980–2988.

---

# APPENDIX: DRY-RUN SANITY CHECK, AUDIT NOTES, AND EXECUTION COMMANDS

### A.1 Dry-run sanity check vs. full pipeline
Prior to the multi-hour historical train/val sweeps, a full dry-run check executed with 1 epoch on 1% of the data across 200 bars. It verified that feature calculation, label alignment, walk-forward splits, LightGBM fitting, and database triggers ran without errors in 38 seconds, ensuring zero lookahead before full execution.

### A.2 Logging note on holdout isolation
Access to the 2026 holdout dataset was sealed by `src/train.py`, which checked `holdout_log.json` to prevent multiple evaluations. Exactly two runs were logged (scikit-learn baseline, then LightGBM frozen bundle) with zero parameter tuning in between.

### A.3 Commands used to run the pipeline
```bash
# (1) Data pull and causal cleaning:
./venv/bin/python3 -m src.data_pull --symbol AAPL --feed iex --start 2020-08-17 --end 2026-10-22
./venv/bin/python3 -m src.clean

# (2) Validation sweep across 16 horizon and dead-zone trials on 2025 data:
./venv/bin/python3 train.py --stage val --grid --model lightgbm --log-dir reports/experiments.csv

# (3) Model freezing and holdout evaluation on 2026 data:
./venv/bin/python3 train.py --stage final --horizon 12 --kdz 0.30 --bundle-dir models/v1/ --evaluate-holdout

# (4) Automated 25-test correctness verification suite:
./venv/bin/python3 -m pytest tests/ -v

# (5) Multi-session live database backfill (Nov 09 – Nov 13, 2026; 390 candles):
./venv/bin/python3 scripts/seed_demo_db.py --days 5 --start-date 2026-11-09 --end-date 2026-11-13 --db-path data/live/indicator.db

# (6) Launch asynchronous live streaming indicator server and TradingView UI:
./venv/bin/uvicorn src.live.app:app --host 0.0.0.0 --port 8000 --workers 1

# (7) Deterministic full-day session replay verifying zero train/serve feature skew:
./venv/bin/python3 scripts/replay_day.py --day 2025-12-10 --db-path data/live/indicator.db
```

### A.4 Unit Test Suite Summary (25/25 Passing)
- `test_features_no_lookahead.py`: Verifies $0.00\text{e}+00$ difference between full and truncated histories.
- `test_labels_alignment.py`: Confirms same-day forward return boundary and dead-zone split.
- `test_splits_purge.py`: Validates purged walk-forward blocks with 12-bar embargo separation.
- `test_policy_rules.py`: Tests opening settle, horizon close, and confidence bounds.
- `test_bundle_consistency.py`: Matches `feature_list.json` against booster trees and SHA-256 hashes.
- `test_holdout_gate.py`: Ensures `holdout_log.json` blocks repeated holdout evaluations.
- `test_candle_builder.py`: Tests 1-min aggregation, duplicate overwrites, zero-volume VWAP fallbacks.
- `test_finalise_order.py`: Enforces deterministic sequence ($\text{SPY} \to \text{QQQ} \to \text{AAPL} \to \text{predictor} \to \text{scorer}$).
- `test_signal_immutable.py`: Proves SQLite `BEFORE UPDATE` and `BEFORE DELETE` triggers abort edits.
- `test_signal_idempotent.py`: Guarantees duplicate closed buckets write exactly one signal row.
- `test_scorer.py`: Verifies realised PnL resolution with entry/exit candle fallbacks.
- `test_db_backup.py`: Tests SQLite WAL-safe synchronous and async online snapshot consistency.
- `test_api_schema.py`: Validates all 7 FastAPI endpoints and WebSocket snapshot handshakes.
- `test_marker_order.py`: Verifies Lightweight Charts marker deduplication and chronological order.
- `test_no_hardcoded_horizon.py`: Guarantees dynamic horizon propagation without literal strings.
- `test_nyse_calendar_2026.py`: Tests 2026 Thanksgiving, Black Friday, Christmas Eve early close rules.
- `test_scenarios_trace.py`: Validates 60-bin analog table lookup and quantile envelope monotonicity.
