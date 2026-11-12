# Phase 1 Report: Data Ingestion and Cleaning

## What Was Built
- `src/data_pull.py`: Extraction script querying Alpaca Market Data API (IEX feed, 5-minute AAPL bars, split/dividend adjusted `Adjustment.ALL`). Covers 2020-08-03 to 2026-10-22.
- `src/clean.py` and `notebooks/01_clean_data.ipynb`: Cleaning pipeline outputting the canonical file `data/clean/aapl_5m_clean.parquet`:
  1. Deduplication on UTC timestamp.
  2. `America/New_York` timezone conversion and filtering to regular trading hours (09:30 to 15:55 ET).
  3. Day bar integrity check: Kept full days (78 bars) and half days (42 bars, 11 sessions, 462 rows); dropped malformed sessions.
  4. Added intra-day positioning: `bar_idx`, `bars_left` (`day_bars - bar_idx - 1`), `bar_end`, `is_half`.
  5. Decoupled overnight gap (`gap = log(open_1 / prev_close)`) from first intraday bar return (`ret_1 = log(close_1 / open_1)`).
  6. Derived `rel_volume` and `rel_trade_count` relative to the past 20-day rolling median (`min_periods=10`) at the same `bar_idx` using strictly causal `shift(1)`. The first 10 trading days (before 2020-08-17) are dropped as warm-up.

## Quality Assertions Passed
- Zero NaNs in OHLCV dataset.
- All prices strictly positive.
- Bar high $\ge \max(\text{open}, \text{close})$ and bar low $\le \min(\text{open}, \text{close})$ with zero violations.
- Clean dataset produced: 120,504 bars spanning 1,550 trading days (2020-08-17 to 2026-10-22).
