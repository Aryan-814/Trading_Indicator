# Phase 5 Report: Scenarios, Holdout Gate, and Frozen Bundle

## What Was Built
- **Scenario Regressors**: LightGBM quantile regression models fitted for $\alpha \in \{0.10, 0.50, 0.90\}$ generating Bear, Base, and Bull price paths. Enforced strict quantile monotonicity ($q_{10} \le q_{50} \le q_{90}$).
- **Empirical Analog Table (`analog_table.parquet`)**: Built from training data binned across:
  - Volatility tercile (`low_vol`, `mid_vol`, `high_vol` cutoffs from `vol_78`).
  - Time bucket (`open`, `mid`, `afternoon`, `close`).
  - First-leg move bucket in units of normalized horizon volatility ($z = fwd / s_t$).
  - Stored empirical distribution of subsequent $h$-bar price move (median, p25, p75, share continuing, share reversing, sample count $N$).
- **Explainability Drivers**: Integrated native LightGBM `pred_contrib=True` extracting top 5 directional feature drivers for each candle.

## Holdout Gate (2026-01-01 to 2026-10-22)
Evaluated exactly once prior to final refit per operating rule #4:
- **Sample size**: 12,743 bars (unseen out-of-sample data).
- **Primary AUC**: 0.5049.
- **Selective Accuracy (20% coverage)**: 49.67% (95% CI: [43.0%, 56.3%]).
- **Calibration ECE**: 0.0267 ($\le 0.03$).
- **Quantile Band Coverage**:
  - Below $q_{10}$: 9.8% (nominal 10.0%).
  - Above $q_{90}$: 10.5% (nominal 10.0%).
  - Inside band: 79.7% (nominal 80.0%).
- **Observation**: Realized holdout directional accuracy aligns with Mesfin (2026) intraday findings ($\approx 50\%$). The quantile prediction bands showed excellent calibration ($\approx 80\%$ empirical mass inside 10-90 bands).

## Frozen Bundle (`models/v1/`)
Final bundle refit on all data through 2026-10-22 (Decision D15 Option B). Stored in `models/v1/`:
- `primary.txt`, `meta.txt` (native LightGBM booster format)
- `quantile_10.txt`, `quantile_50.txt`, `quantile_90.txt`
- `calibrator.joblib` (fitted isotonic regression calibrator)
- `analog_table.parquet` (empirical regime lookup table)
- `thresholds.json` (signal threshold $\tau_m$, horizon $h=12$, $kdz=0.30$, volatility cutoffs)
- `config_snapshot.yaml` (immutable parameters snapshot)
- `bundle_info.json` (SHA-256 integrity hashes for all files in bundle)

