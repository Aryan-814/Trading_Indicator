# Demonstration Script & Evaluator Q&A Guide

**Duration:** 5–7 Minutes  
**Target Audience:** Instructor / Technical Evaluator  
**Purpose:** Walk through the live dashboard, signal lifecycle, statistical validation, and system fault tolerance.

---

## 1. Demo Walkthrough Steps

### Minute 0:00–1:00: Overview & Live Interface
1. **Open the Web Application**: Show `http://localhost:8000/` running in the browser.
2. **One-Second Readability**:
   - Point out the header ticker: AAPL live price, daily change, and badges (`Feed: LIVE`, `Market: OPEN`, `Model: v1`).
   - Highlight the **Current Signal Card**: large arrow, directional verdict (`UP`, `DOWN`, or `NO SIGNAL`), calibrated confidence percentage, and dynamic horizon countdown ("Next 60 min, resolves at 15:55 ET").
3. **Chart & History Strip**:
   - Show the 5-minute candlestick chart powered by Lightweight Charts v4.2.0.
   - Show the chronologically sorted directional markers (blue arrow below candle for Up, orange arrow above for Down with confidence tag).
   - Point to the **Signal History Strip** beneath the chart showing the continuous state of every closed candle.

### Minute 1:00–2:30: Signal Scenarios & Interpretability ("Why")
1. **Select an Active Signal**: Click a marker on the chart or strip.
2. **Scenarios Tab**:
   - Point to the Bear ($q_{10}$), Base ($q_{50}$), and Bull ($q_{90}$) price targets.
   - Explain the follow-up analog narrative: "In similar market regimes (normal volatility, midday session), historical continuation occurred in ~52% of sessions."
   - Emphasize the disclaimer: Quantile price estimates, not guarantees.
3. **Why Tab (Feature Drivers)**:
   - Show the top 5 tree feature contributions (`pred_contrib=True` from LightGBM), explaining how momentum (`ret_1`, `ret_12`), volatility (`vol_12`), and relative volume influenced the classification.

### Minute 2:30–4:00: Signal Log, Immutability & Database Audit
1. **Log Tab**:
   - Show the live table joining signals with resolved forward outcomes, entry prices, exit prices, and net P&L after transaction costs (1.0 bp/side).
   - Click **Export CSV** to demonstrate direct CSV download from `/api/export.csv`.
2. **Database Immutability**:
   - Open a terminal and demonstrate that SQLite triggers strictly block retroactive tampering:
     ```sql
     UPDATE signals SET direction = 'up' WHERE id = 1;
     -- Error: signals table is immutable
     ```
   - Explain that signals on closed candles never repaint and the database guarantees audit integrity.

### Minute 4:00–5:30: Performance & Honest Validation
1. **Performance Tab**:
   - Show running accuracy alongside its 95% Wilson confidence interval based on effective sample size $n_{\text{eff}} = n / h$.
   - Compare active model accuracy against the baseline strategies (**Always-Up** and **RSI Rule**) evaluated on the exact same candles.
   - Show the cost sensitivity breakdown ($0.5$, $1.0$, and $3.0$ bps per side).
2. **Honest Validation Context**:
   - Explain that intraday direction is close to random (~50.8% base rate).
   - Emphasize why the indicator uses meta-labeling to abstain on low-conviction candles: abstention is an intentional, disciplined design choice.

### Minute 5:30–6:30: Fault Tolerance & Failure Handling
1. **Feed Watchdog & Error Handling**:
   - Show the health banner behavior when a feed is stale or disconnected.
   - Explain that missing minute bars are handled deterministically without crashing: missing AAPL candles trigger a `data_quality` non-trade signal, while missing outcome candles use adjacent session bars.

---

## 2. Anticipated Evaluator Questions & Technical Answers

### Q1: Why is overall directional accuracy near 50%?
> **Answer**: High-frequency 5-minute equity returns are dominated by intraday noise and order flow microstructure. Academic literature (e.g., Mesfin 2026 on futures data) consistently finds that even deep models achieve ~50.0% to 50.8% accuracy. We deliberately avoided overfitting or peeking at the holdout dataset to report an artificial number. Instead, we use meta-labeling to filter trades and abstain when confidence is low.

### Q2: How is confidence calibrated?
> **Answer**: Raw meta-model probabilities are calibrated using non-parametric isotonic regression fitted strictly on out-of-sample walk-forward validation folds. A 60% confidence signal indicates that in historical validation, predictions with similar confidence were correct 60% of the time.

### Q3: Why abstain rather than emit a signal on every candle?
> **Answer**: In financial markets, taking a position when edge is negative or zero guarantees capital erosion through the bid-ask spread and transaction costs. By establishing a meta-probability gate ($\tau_m = 0.5218$), the indicator selects only the top ~20% of high-conviction opportunities.

### Q4: How is data leakage prevented?
> **Answer**:
> 1. All features at bar $t$ use only data available at or before $t$ close (proven by `tests/test_features_no_lookahead.py` with zero difference).
> 2. Relative volume and trade count use a rolling 20-day median with a 1-day lag (`shift(1)`).
> 3. Cross-validation uses purged walk-forward splits with an embargo of length $h$.
> 4. The 2026 holdout dataset was evaluated exactly once after all models and thresholds were frozen.

### Q5: How do you guarantee the indicator does not repaint?
> **Answer**: Once a 5-minute candle closes, the signal is inserted into SQLite. Database-level `BEFORE UPDATE` and `BEFORE DELETE` triggers reject any modification. Replay scripts verify that the live pipeline produces byte-for-byte identical signals to the offline model bundle.

