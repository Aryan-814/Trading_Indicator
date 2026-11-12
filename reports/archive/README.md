# Archive Notice

The files here come from the first full run of the pipeline, which used scikit-learn
`HistGradientBoostingClassifier` instead of LightGBM. That run read the 2026 holdout
once (evaluation 1). The submitted LightGBM bundle was evaluated on the same holdout
once more (evaluation 2) with the same horizon, dead-zone and thresholds rule; no tuning
happened between the two. These files are kept for transparency only.

| File | Contents |
|---|---|
| `metrics_hist_gb_holdout1.json` | Validation and holdout metrics from the scikit-learn run |
| `val_run_hist_gb.log` | Console log from the scikit-learn validation sweep |

