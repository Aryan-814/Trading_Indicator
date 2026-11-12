# Phase 5 Report: Scenarios, Holdout Gate, and Frozen Bundle

*All metrics below are generated from `models/v1/metrics.json` via `scripts/make_report_tables.py` (`reports/generated_tables.md`).*

## What Was Built
- **Scenario Regressors**: LightGBM quantile regression models fitted for $\alpha \in \{0.10, 0.50, 0.90\}$ generating Bear, Base, and Bull price targets with strict monotonicity enforced ($q_{10} \le q_{50} \le q_{90}$).
- **Explainability Drivers**: Native LightGBM `pred_contrib=True` extracting top 5 directional feature contributions for each candle.
- **Holdout Gate**: Tracked in `reports/holdout_log.json`, blocking repeated evaluation for the same model family unless explicitly forced.

## Holdout Gate (2026-01-01 to 2026-10-22)
**Disclosure**: The 2026 holdout was evaluated twice: first by the archived scikit-learn `HistGradientBoostingClassifier` run (`reports/archive/metrics_hist_gb_holdout1.json`), and once more by the current LightGBM pipeline with no tuning in between.

LightGBM holdout results (`models/v1/metrics.json`):
- **Sample size**: 12,743 scored bars (203 trading days).
- **Primary accuracy / AUC**: 51.01% / 0.5068.
- **Policy accuracy (`tau_m = 0.5312` from validation)**: **50.80%** across 2,567 signals (20.14% coverage; 12.6 signals/day; 95% Wilson CI `[44.14%, 57.42%]` with effective sample size `n/h`).
- **Always-Up on same candles**: 50.76% (permutation $p = 0.3803$).
- **Calibration ECE / Brier**: ECE `0.0288`; Brier `0.25005` vs constant predictor `0.24990`.
- **Net P&L after costs**: `+0.52` / `-0.48` / `-4.48` bps per signal (validation) and `-1.00` / `-2.00` / `-6.00` bps per signal (holdout) at `0.5` / `1.0` / `3.0` bps per side.
- **Quantile Band Coverage**: 9.6% below $q_{10}$, 79.8% inside band, 10.5% above $q_{90}$ (nominal 80%).
- **Observation & Limitations**: No directional edge was demonstrated out of sample (policy accuracy `50.80%` sits right on top of the `50.76%` always-Up rate on the same candles, consistent with Mesfin 2026). Day-level features (`prev_day_ret`, `prev_day_range`, `dow`) and slowly changing `vol_78` can also memorize individual training days across ~1,000 training days, contributing to validation-to-holdout decay. Only the 10–90 quantile range held up at ~80% empirical coverage.

## Frozen Bundle (`models/v1/`)
Final LightGBM bundle refit on all data through `2026-10-22` (Decision D15 Option B, `tau_m = 0.5218`):
- `primary.txt`, `meta.txt` (native LightGBM booster format)
- `quantile_10.txt`, `quantile_50.txt`, `quantile_90.txt`
- `calibrator.joblib` (fitted isotonic regression calibrator)
- `feature_list.json` (24 primary features, 9 meta features)
- `thresholds.json` (`tau_m = 0.5218`, `horizon_bars = 12`, `kdz = 0.3`, `skip_first_bars = 3`)
- `config_snapshot.yaml` (frozen parameters snapshot)
- `bundle_info.json` (library versions, Python version, git commit, clean data SHA-256, and file SHA-256 hashes)
- `metrics.json` (LightGBM validation and holdout metrics plus `holdout_evaluations` disclosure)
