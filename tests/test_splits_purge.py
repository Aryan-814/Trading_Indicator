"""Mandatory correctness test (section 16):
Purged walk-forward ensures no training row's label window overlaps a test block.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))
from features import load_bars
from splits import PurgedWalkForwardSplitter


def test_splits_purge():
    d = load_bars("aapl_5m_clean.csv")
    # Take a representative slice of 2 years
    subset = d[(d["day"] >= "2021-01-01") & (d["day"] <= "2022-12-31")].copy().reset_index(drop=True)

    h = 6
    splitter = PurgedWalkForwardSplitter(block="quarter", embargo_bars=h, horizon_bars=h)

    fold_count = 0
    for train_idx, test_idx in splitter.split(subset):
        train_days = subset.loc[train_idx, "day"]
        test_days = subset.loc[test_idx, "day"]

        max_train_day = train_days.max()
        min_test_day = test_days.min()

        # Training must strictly precede test start day
        assert max_train_day < min_test_day, f"Purge violation: max train {max_train_day} >= min test {min_test_day}"
        fold_count += 1

    assert fold_count > 0, "No splits generated!"
    print(f"Purge and embargo test passed across {fold_count} folds.")


if __name__ == "__main__":
    test_splits_purge()
    print("test_splits_purge PASSED!")
