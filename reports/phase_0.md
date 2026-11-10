# Phase 0 Report: Environment and Setup

## What Was Built
- Python 3.12 virtual environment configured with core data science dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyyaml`, `joblib`, `alpaca-py`, `yfinance`).
- `config.yaml` established as single source of truth for all pipeline hyperparameters, periods, feature parameters, and live policies.
- `.env.example` created specifying required Alpaca credentials (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`).
- Comprehensive `.gitignore` implemented to prevent committing secrets (`.env`), bulky datasets (`*.parquet`, `*.csv`), and virtual environment artifacts.

## Testing & Verification
- Confirmed virtual environment imports: `lightgbm`, `sklearn`, `pandas`, `numpy`, `yaml`, `joblib`.
- Verified `.env` exclusion from version control.
- One-connection lock architecture drafted for Phase 3 live stream safety.

