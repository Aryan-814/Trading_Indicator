"""Mandatory correctness test 1 (section 8.4):
Features from full history equal features from history truncated at the bar.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("src"))
from features import load_bars, make_features


def test_features_no_lookahead():
    d = load_bars("aapl_5m_clean.csv")
    full = make_features(d)

    rng = np.random.default_rng(42)
    # Pick 200 random bars beyond the warm-up window
    sample_bars = rng.choice(np.arange(1000, len(d)), size=200, replace=False)

    worst_diff = 0.0
    for idx in sample_bars:
        truncated = d.iloc[: idx + 1].reset_index(drop=True)
        feat_trunc = make_features(truncated)

        row_full = full.iloc[idx].values.astype(float)
        row_trunc = feat_trunc.iloc[-1].values.astype(float)

        both_nan = np.isnan(row_full) & np.isnan(row_trunc)
        diff = np.abs(row_full - row_trunc)
        diff[both_nan] = 0.0

        max_d = np.nanmax(diff)
        if max_d > worst_diff:
            worst_diff = max_d

    print(f"Max feature difference across 200 bars: {worst_diff:.2e}")
    assert worst_diff < 1e-9, f"Feature look-ahead detected! Worst diff: {worst_diff}"


if __name__ == "__main__":
    test_features_no_lookahead()
    print("test_features_no_lookahead PASSED!")
