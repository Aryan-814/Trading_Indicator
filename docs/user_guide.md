# AAPL Live Trading Indicator — User Guide

## Overview

The indicator runs on 5-minute candles for **AAPL** during regular NYSE trading hours (09:30 to 16:00 ET). At the close of each 5-minute candle, it outputs:

1. **Direction**: `UP`, `DOWN`, or `NONE` (no clear signal).
2. **Confidence**: A calibrated probability estimate for display (e.g., `Up, 54%`).
3. **Scenarios (Out-of-Sample Price Quantiles)**:
   - **Bear case (q10)**: 10th percentile forward price target.
   - **Base case (q50)**: Median forward price target.
   - **Bull case (q90)**: 90th percentile forward price target.
4. **Drivers ("Why")**: Top 5 feature contributions pushing the primary prediction higher or lower (`pred_contrib=True`).

> [!IMPORTANT]
> Out-of-sample holdout testing showed no directional edge over always-Up. Signals are for research and paper trading only and should not be traded blindly.

---

## Signal Policy Rules

Signals are strictly deterministic and follow this sequence:

1. **End-of-Day Cutoff**: When fewer than $h$ bars remain before 16:00 ET (`bars_left < h`), signal is `NONE` (reason: `horizon_past_close`). The model never predicts across overnight gaps.
2. **Opening Settle**: First 3 bars of the day (`bar_idx < 3`, 09:30 to 09:45 ET) output `NONE` (reason: `opening_settle`).
3. **Selective Confidence Threshold**: A directional signal (`UP` or `DOWN`) is emitted only if the raw meta-probability $m \ge \tau_m$. Otherwise, signal is `NONE` (reason: `low_confidence`).
4. **No Repainting**: Features use only closed bars $\le t$, so historical signals do not repaint.

---

## When NOT to Trust the Indicator

> [!WARNING]
> Do not rely on signals during the following regimes:

- **Breaking News & Earnings Releases**: The model evaluates intraday price action, volatility, and relative volume. It does not parse live headlines or quarterly earnings surprises.
- **FOMC Rate Decision & CPI Announcements**: Macroeconomic releases introduce extreme spread widening and regime discontinuity.
- **Half-Day Sessions**: Only 11 half-days (42 bars each, e.g., before July 4th, Black Friday, Christmas Eve) appear in the training data; treat signals on early-close sessions with caution.
- **Low Confidence Regimes**: When meta-probability $m < \tau_m$, movement is statistically indistinguishable from random noise intraday. Respect the `NONE` signal.
