# Phase 4 Report: Modeling, Meta-Labeling, and Calibration

## What Was Built
- **Primary Model**: LightGBM binary classifier (`num_leaves=15`, `max_depth=5`, `min_child_samples=300`, `subsample=0.7`, `colsample_bytree=0.7`, `reg_lambda=5.0`, `reg_alpha=0.1`) trained on the 24 causal features over non-dead-zone bars ($y \in \{-1, +1\}$).
- **Meta-Model (Confidence Engine)**: Secondary LightGBM trained on out-of-fold primary predictions (`p_up`, `p_dist`, `vol_12`, `vol_78`, `tod`, `rel_volume_log`, `range_12`, `is_half`, `bars_left`) to predict $P(\text{primary is correct})$.
- **Calibration Engine**: Non-parametric `IsotonicRegression` fit on out-of-fold meta predictions (`m_oof`) to produce display confidence `conf`.
- **Selective Policy Gate**: Signals are emitted only when raw meta-probability $m \ge \tau_m$ (threshold chosen at the 80th percentile of validation eligible bars, targeting 20% coverage).

## LightGBM Sweep (2025 Validation)
Recorded in `reports/experiments.csv` (run `lgb_d6ead9`) and summarized in `reports/generated_tables.md`:

| Trial | Horizon $h$ | Dead-Zone $k_{dz}$ | Selective Acc (20% cov) | ECE | ROC AUC | B1 Always-Up | perm_p |
|---|---|---|---|---|---|---|---|
| 11 | 2 (10 min) | 0.15 | 50.21% | 0.0135 | 0.5045 | 0.4990 | 0.3588 |
| 12 | 2 (10 min) | 0.30 | 49.28% | 0.0155 | 0.4985 | 0.4990 | 0.7961 |
| 13 | 6 (30 min) | 0.15 | 49.00% | 0.0133 | 0.5094 | 0.5035 | 0.8661 |
| 14 | 6 (30 min) | 0.30 | 50.01% | 0.0172 | 0.5056 | 0.5035 | 0.6017 |
| 15 | 12 (60 min) | 0.15 | 50.74% | 0.0196 | 0.5067 | 0.4981 | 0.2789 |
| 16 | 12 (60 min) | 0.30 | **51.76%** | **0.0168** | **0.5128** | 0.4981 | 0.1209 |

**Selected Configuration**: Horizon 12 bars (60 min), $k_{dz}=0.30$ (Trial 16). Passed the calibration filter ($\text{ECE} = 0.0168 \le 0.03$), but **was not statistically significant** even before multiple-testing adjustment (`perm_p = 0.1209`, Bonferroni-adjusted $p = 1.0000$ across all 16 logged validation trials).

---

## Scikit-learn Run (Archived)
An earlier validation sweep (`hist_gb_run1`, trials 1–6 in `reports/experiments.csv`) used scikit-learn `HistGradientBoostingClassifier` and is preserved in `reports/archive/` for transparency:

| Trial | Horizon $h$ | Dead-Zone $k_{dz}$ | Selective Acc (20% cov) | ECE | ROC AUC | Baseline B1 |
|---|---|---|---|---|---|---|
| 1 | 2 (10 min) | 0.15 | 49.59% | 0.0125 | 0.5024 | 49.90% |
| 2 | 2 (10 min) | 0.30 | 50.49% | 0.0117 | 0.5019 | 49.90% |
| 3 | 6 (30 min) | 0.15 | 51.31% | 0.0135 | 0.5033 | 50.35% |
| 4 | 6 (30 min) | 0.30 | 49.19% | 0.0152 | 0.5102 | 50.35% |
| 5 | 12 (60 min) | 0.15 | 50.26% | 0.0207 | 0.5068 | 49.81% |
| 6 | 12 (60 min) | 0.30 | 53.34% | 0.0205 | 0.5145 | 49.81% |

*(Note: Trial 6's raw `p = 0.0195` is not significant after multiple-testing correction across the logged trials.)*
