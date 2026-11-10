# Phase 4 Report: Modeling, Meta-Labeling, and Calibration

## What Was Built
- **Primary Model**: LightGBM binary classifier (`num_leaves=15`, `max_depth=5`, `min_child_samples=300`, `subsample=0.7`, `colsample_bytree=0.7`, `reg_lambda=5.0`, `reg_alpha=0.1`).
- **Meta-Model (Confidence Engine)**: Secondary LightGBM trained strictly on out-of-fold primary predictions. Features: $P(\text{up})$, $|P(\text{up}) - 0.5|$, intraday volatility, range, relative volume, time-of-day. Predicts $P(\text{primary is correct})$.
- **Calibration Engine**: Non-parametric isotonic regression mapping raw meta probabilities to empirical accuracy.
- **Selective Policy Selection**: Signal cutoff $\tau_m$ chosen dynamically on the 2025 validation period to achieve $20\%$ coverage ($\approx 12-16$ selective signals per day).

## Hyperparameter Grid Search (2025 Validation)
Recorded in `reports/experiments.csv`:

| Trial | Horizon $h$ | Dead-Zone $kdz$ | Selective Acc (20% cov) | ECE | ROC AUC | Baseline B1 |
|---|---|---|---|---|---|---|
| 1 | 2 (10 min) | 0.15 | 49.59% | 0.0125 | 0.5024 | 49.90% |
| 2 | 2 (10 min) | 0.30 | 50.49% | 0.0117 | 0.5019 | 49.90% |
| 3 | 6 (30 min) | 0.15 | 51.31% | 0.0135 | 0.5033 | 50.35% |
| 4 | 6 (30 min) | 0.30 | 49.19% | 0.0152 | 0.5102 | 50.35% |
| 5 | 12 (60 min) | 0.15 | 50.26% | 0.0207 | 0.5068 | 49.81% |
| 6 | 12 (60 min) | 0.30 | **53.34%** | **0.0205** | **0.5145** | 49.81% |

**Winning Configuration**: Horizon 12 bars (60 min), $kdz=0.30$, Trial ID 6. Met ECE criteria ($\le 0.03$), demonstrated statistically significant edge over random walk ($p=0.019$ in block permutation test).

