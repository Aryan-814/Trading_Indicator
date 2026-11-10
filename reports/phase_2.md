# Phase 2 Report: Features and Labels

## What Was Built
- `src/features.py`: Causal feature engineering engine producing 33 features strictly using historical information up to bar $t$:
  - Short- and multi-horizon intraday returns: `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`, `ret_78`, `day_ret`, `gap`.
  - Volatility and range: `vol_12`, `vol_78`, `range_1`, `range_12`.
  - Order flow / VWAP: `vwap_dist`, `sess_vwap_dist`, `rel_volume_log`, `rel_tc_log`.
  - Temporal and seasonality: `tod`, `tod_sin`, `tod_cos`, `dow`, `is_half`, `bars_left`.
  - Prior day context: `prev_day_ret`, `prev_day_range`.
  - Technical oscillators: `rsi_14` (Wilder continuous smoothing).
  - Market context: `spy_ret_k`, `qqq_ret_k`, `spy_vol_12`, `aapl_vs_spy_6`.
- `src/labels.py`: Volatility-scaled fixed-horizon directional labeling ($h \in \{2, 6, 12\}$):
  - $s_t = \sigma_t \sqrt{h}$, where $\sigma_t$ is causal 78-span EWM standard deviation.
  - Primary label $y \in \{+1, -1, 0\}$, with dead-zone $kdz \in \{0.15, 0.30\}$.
  - Scoring label $y_{\text{sign}} = \text{sign}(fwd)$.
  - Zero label crossing: bars with `bars_left < h` receive `NaN` and are omitted from training/scoring.

## Test Results
1. **No-look-ahead Test (`tests/test_features_no_lookahead.py`)**:
   - 200 random intraday evaluation points tested.
   - Max absolute difference between full-sample feature calculation and truncated real-time buffer: `0.00e+00`. Passed.
2. **Label Alignment Test (`tests/test_labels_alignment.py`)**:
   - Exactly 9,300 bars with `bars_left < 6` confirmed `NaN`.
   - Forward returns strictly bounded within the session day.
   - Class balance: 51.3% UP vs 48.7% DOWN on non-dead-zone samples. Passed.

