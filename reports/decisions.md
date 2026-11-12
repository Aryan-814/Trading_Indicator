# Decision Register

This document records the architectural decisions made during development.

| # | Decision | Choice | Evidence / Rationale |
|---|---|---|---|
| **D1** | Prediction Horizon | **60 min (12 bars)** | Evaluated horizons 2, 6, and 12 bars on 2025 validation set across 6 LightGBM trials (`lgb_d6ead9`). Horizon 12 with `kdz=0.3` achieved the highest selective accuracy at 20% coverage (`51.76%`, `ece=0.0168`) in the validation sweep (`reports/generated_tables.md`). |
| **D2** | Label Definition | **(a) Fixed-horizon with dead-zone** | Aligns directly with evaluation criteria (`close` at $t+h$ vs `close` at $t$, within-day only). |
| **D3** | Dead-Zone Multiplier | **Volatility-scaled (`kdz=0.30`)** | Filters intraday micro-noise ($|fwd| < 0.30 \cdot \sigma_t \sqrt{h}$, 32.6% of labelled rows) so primary trees focus on larger moves. |
| **D4** | Primary Model | **LightGBM Binary Classifier** | Fast training, handles tabular interactions, and provides native feature contributions via `pred_contrib=True`. |
| **D5** | Confidence Engine | **Meta-labeling + Isotonic Calibration** | Primary predicts direction; meta-model predicts $P(\text{correct})$ trained on out-of-fold primary predictions (`walk_forward_blocks`). Isotonic regression calibrates `m` to display confidence `conf`. |
| **D6** | Online Calibration | **Not implemented / offline only** | Keeps bundle frozen during evaluation. |
| **D7** | Signal Policy | **Confidence threshold (`m >= tau_m`)** | Validation cutoff $\tau_m = 0.5312$ (refit $\tau_m = 0.5218$) targeting 20% of eligible candles (13.0 signals/day on validation, 12.6 signals/day on holdout; `reports/generated_tables.md`). |
| **D8** | No-Signal Zones | **First 3 bars + `bars_left < h`** | Skips opening settle (`bar_idx < 3`) and avoids predicting across overnight gaps (`bars_left < h`). |
| **D9** | Scenario Engine | **LightGBM Quantiles (10/50/90) only** | Predicts Bear ($q_{10}$), Base ($q_{50}$), and Bull ($q_{90}$) forward price targets with monotonicity enforced. |
| **D15** | Final Model Fitting | **Option B: Refit on all data to freeze date** | Final bundle retrained with validated `h=12, kdz=0.30` on full history through `2026-10-22` (no out-of-sample score for the refit bundle). |
| **D16** | Cost Assumption | **1.0 bp per side** | Evaluated at 0.5, 1.0, and 3.0 bps per side (`config.yaml:cost_bps`). |
| **D17** | Explainability | **LightGBM `pred_contrib=True`** | Extracts top 5 directional feature drivers per candle. |
| **D18** | Analog Table Removed | **Deleted** | The original analog-table conditioning used future forward returns (`fwd`) as lookup keys, which is unavailable at live time; deleted to keep outputs strictly causal and verified. |
