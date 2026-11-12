# AAPL Live Trading Indicator — User Guide

## 1. What the Indicator Predicts

The indicator predicts the directional price movement of **Apple Inc. (AAPL)** over a fixed forward horizon following the close of each regular-session 5-minute candle.

- **Horizon**: The prediction applies across a horizon of **{horizon_bars} candles ({horizon_minutes} minutes)**, dynamically loaded from the active model bundle (`models/v1/thresholds.json`). For example, in the production configuration where $h=12$ bars, each signal at candle $t$ forecasts price direction at $t+12$ bars (60 minutes forward).
- **Directional Target**:
  - **UP (▲)**: Model forecasts that AAPL close price at the horizon will be higher than the reference entry price.
  - **DOWN (▼)**: Model forecasts that AAPL close price at the horizon will be lower than the reference entry price.
  - **NO SIGNAL (⚪)**: The model chooses to abstain from taking a directional position.

---

## 2. How to Read Signals & the User Interface

### 2.1 The Current Signal Card (One-Second Readability)
A trader or viewer should be able to glance at the signal card and immediately answer:
1. **What is the stance?** Large bold icon and text: `UP`, `DOWN`, or `NO SIGNAL`.
2. **How sure is it?** Displayed confidence percentage (e.g., `58%`).
3. **When does it resolve?** Explicit clock time in New York ET (or local time) and remaining countdown.

### 2.2 What Calibrated Confidence Means (and Does NOT Mean)
- **What it means**: Confidence is derived through isotonic regression fitted on out-of-sample walk-forward validation folds. A signal with confidence 60% means that historically, on candles where the meta-model produced a similar score, the primary directional prediction was empirically correct approximately 60% of the time.
- **What it does NOT mean**: It is **not a guarantee** of a profitable trade. A 60% confidence signal still carries an estimated 40% probability of an adverse move.

### 2.3 The Signal History Strip
Directly beneath the chart is a thin chronological strip where each cell represents one closed 5-minute candle:
- **Blue**: Confident Up signal.
- **Orange**: Confident Down signal.
- **Gray / Muted**: Abstention (`NONE`).
- **Cell Opacity**: Scales directly with calibrated confidence (darker / more opaque cells reflect higher conviction).

### 2.4 Quantile Price Scenarios
The *Scenarios* tab displays three monotonic price targets for the horizon:
- **Bear Case ($q_{10}$)**: 10th percentile outcome (tail risk).
- **Base Case ($q_{50}$)**: Median expected outcome.
- **Bull Case ($q_{90}$)**: 90th percentile outcome.

The accompanying follow-up analog narrative projects historical continuation versus mean-reversion frequencies based on similar market volatility regimes and time-of-day bins.

---

## 3. When NOT to Trust the Indicator

> [!WARNING]
> In live trading conditions, the indicator should be treated with extreme caution or ignored during the following regimes:

1. **Earnings Announcements & Breaking Macro News**:
   The model is trained on technical price action, multi-scale return momentum, volatility, and volume profiles. It does not parse live natural language news feeds, earnings releases, CPI reports, or FOMC interest rate statements.
2. **Opening Settle (First 15 Minutes)**:
   Between 09:30 and 09:45 ET (`bar_idx < 3`), morning price discovery exhibits wide spreads and auction imbalances. The model strictly abstains during this window (`opening_settle`).
3. **Approaching Session Close**:
   When fewer than $h$ bars remain before 16:00 ET (`bars_left < h`), the model outputs `NO SIGNAL` (`horizon_past_close`) to prevent predicting across overnight holding risk.
4. **Stale or Disconnected Market Feeds**:
   If the WebSocket connection drops or Alpaca IEX bars stop arriving for >90 seconds, the health banner turns red and signals are suspended (`data_quality`).
5. **Low-Liquidity Half-Days**:
   Sessions with early 13:00 ET closes (e.g., the day after Thanksgiving, Christmas Eve) contain only 42 bars and exhibit compressed trading volume.
6. **Recent Model Refits or Parameter Changes**:
   Directly following model updates, allow several sessions to observe live calibration stability before acting on predictions.

---

## 4. Honest Expectations & Abstention Philosophy

- **Intraday Market Reality**: Peer-reviewed financial literature (e.g., Mesfin 2026, Fischer & Krauss 2018) demonstrates that 5-minute equity and futures directional returns sit near a random 50.0% base rate. After accounting for bid-ask spread crossing and broker commission costs (1.0 to 3.0 bps per side), an unselective trading strategy yields negative expected return.
- **Abstention is an Essential Strategy**: The indicator actively filters out low-conviction bars using a meta-labeling gating mechanism ($\tau_m$). Abstaining on 70% to 80% of candles is the intended, mathematically sound behavior—not a software fault.
- **Decision Aid Only**: This software is designed strictly as an academic research and paper-trading decision aid. It does not constitute financial advice.
