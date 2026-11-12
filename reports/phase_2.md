# Phase 2 Report: Features and Labels

## What Was Built
- `src/features.py`: Causal feature engineering engine producing **24 features** strictly using historical information up to bar $t$:
  - Short- and multi-horizon intraday returns: `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`, `day_ret`, `gap`.
  - Volatility and range: `vol_12`, `vol_78`, `range_1`, `range_12`.
  - Order flow / VWAP: `vwap_dist`, `sess_vwap_dist`, `rel_volume_log`, `rel_tc_log`.
  - Temporal and seasonality: `tod`, `tod_sin`, `tod_cos`, `dow`, `is_half`, `bars_left`.
  - Prior day context: `prev_day_ret`, `prev_day_range`.
  - Technical oscillators: `rsi_14` (Wilder continuous smoothing).
  - *(Note: SPY/QQQ market-context features and `ret_78` — which duplicated `day_ret` — were removed because SPY/QQQ bars were not pulled.)*
- `src/labels.py`: Volatility-scaled fixed-horizon directional labeling ($h \in \{2, 6, 12\}$):
  - $s_t = \sigma_t \sqrt{h}$, where $\sigma_t$ is causal 78-span EWM standard deviation (`min_periods=30`).
  - Primary label $y \in \{+1, -1, 0\}$, with dead-zone $k_{dz} \in \{0.15, 0.30\}$.
  - Scoring label $y_{\text{sign}} = \text{sign}(fwd)$.
  - Zero cross-day leakage: bars with `bars_left < h` receive `NaN` and are omitted from training and scoring.

## Label Statistics & Tests
- **Production setting ($h = 12$, $k_{dz} = 0.30$)**: 101,875 valid labelled rows (18,629 dropped at end-of-day / warm-up); 35,571 UP (51.8% of non-dead-zone), 33,143 DOWN (48.2%), and 33,161 dead-zone (32.6% of valid rows).
- **No-look-ahead Test (`tests/test_features_no_lookahead.py`)**: Tested 200 random intraday evaluation points against truncated history; max diff `0.00e+00`. Passed.
- **Label Alignment Test (`tests/test_labels_alignment.py`)**: Verified `bars_left < h` rows are `NaN`, sampled 500 eligible rows against raw `log(close[t+h]/close[t])`, and checked label values/balance. Passed.
