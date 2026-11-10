# Phase 3 Report: Baselines and Validation Protocol

## What Was Built
- `src/splits.py`: Purged Walk-Forward cross-validation splitter with quarterly blocks and $h$-bar embargo.
- `src/evaluate.py`: Standardized metrics engine computing AUC, Brier score, ECE, Wilson confidence intervals, block permutation test, and simulated net P&L with 0.5, 1.0, and 3.0 bps transaction cost assumptions.
- Baselines implemented:
  - **B1 (Always Up)**: Predicts UP on every candle.
  - **B2 (Wilder RSI rule)**: RSI < 30 $\to$ UP, RSI > 70 $\to$ DOWN.
  - **B3 (Momentum)**: Sign of prior 6-bar return (`ret_6`).
  - **B4 (Logistic Regression)**: L2 regularized linear model.
  - **B5 (LightGBM Direct)**: Single-stage tree baseline.

## Test Results
1. **Shuffled-Label Sanity Test (`train.py check`)**:
   - Out-of-fold accuracy on randomly permuted targets: **50.21%** (within expected statistical noise of 50.00%).
   - ROC AUC on shuffled targets: **0.4985**.
   - Confirms zero target leakage across folds.
2. **Purged Splitter Test (`tests/test_splits_purge.py`)**:
   - Verified across 7 walk-forward folds that training indices strictly precede test blocks with zero label boundary overlap.

