"""Mandatory correctness test:
Label alignment, no day boundary crossing, and valid label values.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("src"))
from features import load_bars
from labels import make_labels


def test_labels_alignment():
    d = load_bars("aapl_5m_clean.csv")
    h = 6
    lab = make_labels(d, h=h, kdz=0.15)

    # 1. bars_left < h must have NaN labels
    end_of_day = d[d["bars_left"] < h]
    assert lab.loc[end_of_day.index, "y"].isna().all(), "bars_left < h received non-NaN label!"

    # 2. Forward return must NOT cross day boundary
    g = d.groupby("day")
    fwd = np.log(g["close"].shift(-h) / d["close"])
    assert fwd.loc[end_of_day.index].isna().all(), "fwd_ret crossed day boundary!"

    # 3. Label values must be in {-1, 0, 1}
    valid_vals = {-1.0, 0.0, 1.0}
    actual_vals = set(lab["y"].dropna().unique())
    assert actual_vals.issubset(valid_vals), f"Invalid labels found: {actual_vals}"

    # 4. Class balance
    up = (lab["y"] == 1.0).sum()
    down = (lab["y"] == -1.0).sum()
    pct_up = up / (up + down)
    assert 0.45 <= pct_up <= 0.55, f"Heavily unbalanced labels: {pct_up:.1%}"


if __name__ == "__main__":
    test_labels_alignment()
    print("test_labels_alignment PASSED!")
