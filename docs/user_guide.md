# AAPL Live Trading Indicator — User Guide

## Overview

The indicator runs on 5-minute candles for **AAPL** during regular NYSE trading hours (09:30 to 16:00 ET). At the close of each 5-minute candle, it outputs:

1. **Direction**: `UP`, `DOWN`, or `NONE` (no clear signal).
2. **Confidence**: A calibrated probability estimate (e.g., `Up, 68%`).
3. **Scenarios (Multi-Horizon Out-of-Sample Price Forecast)**:
   - **Bear case (q10)**: 10th percentile forward price target.
   - **Base case (q50)**: Median forward price target.
   - **Bull case (q90)**: 90th percentile forward price target.
4. **Drivers ("Why")**: Top 5 feature contributions pushing the prediction higher or lower.
5. **Historical Analog**: Empirical continuation vs reversal statistics under matching market regime conditions.

---

## Signal Policy Rules

Signals are strictly deterministic and follow this sequence:

1. **Data Quality Filter**: If market data is stale or corrupted, signal is `NONE` (reason: `data_quality`).
2. **Opening Settle**: First 3 bars of the day (09:30 to 09:45 ET) output `NONE` (reason: `opening_settle`).
3. **End-of-Day Cutoff**: When fewer than $h$ bars remain before 16:00 ET, signal is `NONE` (reason: `horizon_past_close`). The model never predicts across overnight gaps.
4. **Selective Confidence Threshold**: A directional signal (`UP` or `DOWN`) is ONLY produced if calibrated meta-confidence $m \ge \tau_m$. Otherwise, signal is `NONE` (reason: `low_confidence`).
5. **No Repainting**: Once a candle closes and its signal record is generated, it is immutable.

---

## When NOT to Trust the Indicator

> [!WARNING]
> Do not execute signals blindly during the following regimes:

- **Breaking News & Earnings Releases**: The model evaluates technical price action, relative volume, and market correlation. It does not parse live breaking headlines or quarterly earnings surprise calls.
- **FOMC Rate Decision & CPI Announcements**: 14:00 ET Federal Reserve announcements introduce extreme spread widening and regime discontinuity where historical analogies break down.
- **Half-Day Sessions**: The day before July 4th, Black Friday, and Christmas Eve feature thin volume (42 bars) where tree estimators experience lower reliability.
- **Low Confidence Regimes**: When meta-probability $m < \tau_m$, price movement is statistically indistinguishable from a random walk intraday. Respect the `NONE` signal.

