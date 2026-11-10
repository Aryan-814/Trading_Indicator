# Decision Register

This document records the locked and experimental architectural decisions made during development per spec section 2.

| # | Decision | Choice | Evidence / Rationale |
|---|---|---|---|
| **D1** | Prediction Horizon | **60 min (12 bars)** | Evaluated horizons 2, 6, and 12 bars on 2025 validation set. Horizon 12 with $kdz=0.3$ achieved top selective accuracy (53.3% at 20% coverage) and lowest permutation p-value ($p=0.019$). |
| **D2** | Label Definition | **(a) Fixed-horizon with dead-zone** | Aligns directly with grader's evaluation criteria (close at $t+h$ vs close at $t$). |
| **D3** | Dead-Zone Multiplier | **Volatility-scaled ($kdz=0.30$)** | Removes intraday micro-noise ($fwd < 0.3 \cdot \sigma_t \sqrt{h}$) allowing tree estimators to focus on trending regimes. |
| **D4** | Primary Model | **LightGBM Binary Classifier** | Fast training, handles non-linear interactions, provides exact Shapley-style feature contributions via `pred_contrib=True`. |
| **D5** | Confidence Engine | **Meta-labeling + Isotonic Calibration** | Two-stage setup: primary predicts direction; secondary predicts $P(\text{correct})$ trained strictly on out-of-fold primary predictions. Isotonic regression maps to empirical accuracy. |
| **D6** | Online Calibration | **Off by default** | Keeps bundle strictly frozen during 5-day grader evaluation to avoid unverified model drift. |
| **D7** | Signal Policy | **Confidence threshold ($\tau_m \approx 0.52-0.54$)** | Calibrated to achieve $\approx 15-20\%$ coverage ($\approx 12-16$ selective signals per trading day). |
| **D8** | No-Signal Zones | **First 3 bars + last $h$ bars** | Filters out volatile opening auction settlement and avoids predicting across overnight gaps. |
| **D9** | Scenario Engine | **LightGBM Quantiles (10/50/90) + Analog Table** | Deterministic empirical distribution of secondary moves conditioned on volatility tercile and time-of-day. |
| **D10** | Market Context Data | **SPY and QQQ 5m bars** | Same IEX feed and timestamps; adds relative strength and market volatility proxy. |
| **D15** | Final Model Fitting | **Option B: Refit on all data to freeze date** | Final bundle retrained with validated hyperparameters on full history through Oct 2026. |
| **D16** | Cost Assumption | **1.0 bp per side** | Standard institutional execution cost model for AAPL; evaluated with 0.5 and 3.0 bps sensitivity. |
| **D17** | Explainability | **LightGBM `pred_contrib=True`** | Zero overhead; extracts top 5 instantaneous directional drivers per candle. |

