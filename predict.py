"""Live signal generation CLI — entry point only.

All logic is in src/models.py. This script loads the bundle and prints signals.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from models import load_bundle, make_signal
from features import load_bars


if __name__ == "__main__":
    bundle = load_bundle()
    print(f"Loaded bundle format: {bundle['format']}")
    bars = load_bars("data/clean/aapl_5m_clean.parquet")
    tail = bars.iloc[-3000:].reset_index(drop=True)
    n_sig = 0
    n_all = 0
    for i in range(len(tail) - 200, len(tail)):
        hist = tail.iloc[: i + 1].reset_index(drop=True)
        s = make_signal(bundle, hist)
        n_all += 1
        if s["direction"] != "none":
            n_sig += 1
    print("last candle:", make_signal(bundle, tail))
    print("signals on last 200 candles:", n_sig, "of", n_all)
