# Phase 0 Report: Environment and Setup

## What Was Built
- Python 3.12 virtual environment with exact pinned dependencies in `requirements.txt` (`lightgbm==4.7.0`, `pandas==3.1.0`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `pyarrow==26.0.0`, `joblib==1.6.0`, `PyYAML==6.0.3`, `alpaca-py==0.44.0`, `python-dotenv==1.2.4`, `pytest==9.2.0`).
- `config.yaml` established as single source of truth for pipeline parameters, periods, CV settings, and live-backend reserved settings.
- `.env.example` created specifying required Alpaca credentials (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`).
- `.gitignore` implemented to exclude `.env`, virtual environments, logs, and raw/cleaned datasets.

## Testing & Verification
- Confirmed virtual environment imports and exact pinned versions.
- Verified `.env` exclusion from version control.
