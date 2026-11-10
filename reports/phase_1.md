# Phase 1 Report: Data Ingestion and Cleaning

## What Was Built
- `src/data_pull.py`: Extraction script querying Alpaca Market Data API (IEX feed, 5-minute bars, split/dividend adjusted `Adjustment.ALL`). Covers 2020-08-03 to 2026-10-22.
- `src/clean.py` and `clean_aapl_5m.ipynb`: Rigorous cleaning pipeline adhering to spec section 5.3:
  1. Deduplication on UTC timestamp.
  2. America/New_York timezone conversion and filtering to regular trading hours (09:30 to 15:55 ET).
  3. Day bar integrity check: Kept full days (78 bars) and half days (42 bars: Christmas Eve, Black Friday, July 3rd); dropped malformed sessions (e.g. 13 Apr 2021 with 72 bars).
  4. Added intra-day positioning: `bar_idx`, `bars_left`, `bar_end`, `is_half`.
  5. Decoupled overnight gap (`gap = log(open_1 / prev_close)`) from first intraday bar return (`ret_1 = log(close_1 / open_1)`).
  6. Derived relative volume and trade count relative to the 20-day past rolling median at the same intraday `bar_idx` using strictly causal `shift(1)`.

## Quality Assertions Passed
- Zero NaNs in OHLCV dataset.
- All prices strictly positive.
- Bar high $\ge \max(\text{open}, \text{close})$ with zero violations.
- Bar low $\le \min(\text{open}, \text{close})$ with zero violations.
- Clean dataset produced: 120,504 bars spanning August 2020 through October 2026.

