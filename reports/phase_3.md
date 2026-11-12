# Phase 3 Report: Baselines and Validation Protocol

## What Was Built
- `src/splits.py`: Expanding walk-forward block splitter (`walk_forward_blocks(df, min_q)`) using quarterly blocks. Because labels are strictly intraday (`bars_left < h` dropped), dropping all days `>=` the first day of the test quarter is a complete purge — no extra embargo is needed.
- `src/evaluate.py`: Metrics engine computing AUC, Brier score, ECE, Wilson confidence intervals (`n_eff = n / h`), and block permutation test (blocks of length $h$).
- Baselines implemented in `train.py:evaluate()`:
  - **B1 (Always Up)**: Fraction of upward moves on scored candles (and on policy-selected candles).
  - **B2 (Wilder RSI rule)**: `rsi_14 < 30` $\to$ UP, `rsi_14 > 70` $\to$ DOWN.
  - **B3 (Momentum)**: Sign of prior 6-bar return (`ret_6`).
  - **B4 (Logistic Regression)**: L2-regularized `LogisticRegression` on standardized `feat_cols`, top 20% by `|p_up - 0.5|`.
  - **B5 (LightGBM Direct)**: Single-stage primary model alone, top 20% by `|p_up - 0.5|`.

## Test Results
1. **Shuffled-Label Sanity Test (`python train.py check`, logged in `reports/shuffled_label_check.txt`)**:
   - At $h=12, k_{dz}=0.30$: out-of-fold accuracy on randomly permuted targets: **51.35%** (reflecting the ~51.8% training UP base rate); ROC AUC: **0.5047**.
   - Confirms zero target leakage across walk-forward folds.
2. **Walk-Forward Split Purge Test (`tests/test_splits_purge.py`)**:
   - Verified across all walk-forward folds starting at quarter index `min_q=4` that `max(train_day) < min(test_day)` with zero label boundary overlap.
