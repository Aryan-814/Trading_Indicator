"""Mandatory correctness test: label alignment, no day-boundary crossing, valid values."""
import json
import numpy as np
import pytest
from pathlib import Path
from features import load_bars
from labels import make_labels

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "data" / "clean" / "aapl_5m_clean.parquet")
BEST_CFG = str(ROOT / "reports" / "best_cfg.json")


def _get_hkdz():
    cfg_path = Path(BEST_CFG)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            best = json.load(fh)
        return best["horizon_bars"], best["kdz"]
    return 12, 0.30


def test_labels_alignment():
    h, kdz = _get_hkdz()
    d = load_bars(DATA)
    lab = make_labels(d, h=h, kdz=kdz)

    # 1. bars_left < h must have NaN labels
    end_of_day = d[d["bars_left"] < h]
    assert lab.loc[end_of_day.index, "y"].isna().all(), \
        "bars_left < h received non-NaN label!"

    # 2. Forward return must equal log(close[t+h]/close[t]) for same-day rows
    rng = np.random.default_rng(42)
    eligible = d[d["bars_left"] >= h].index.values
    sample = rng.choice(eligible, size=min(500, len(eligible)), replace=False)
    for idx in sample:
        row = d.iloc[idx]
        future_idx = idx + h
        if future_idx >= len(d):
            continue
        future_row = d.iloc[future_idx]
        # Must be same day
        if future_row["day"] != row["day"]:
            continue
        expected_fwd = np.log(future_row["close"] / row["close"])
        actual_fwd = lab.iloc[idx]["fwd"]
        assert not np.isnan(actual_fwd), f"fwd is NaN at eligible idx {idx}"
        assert abs(actual_fwd - expected_fwd) < 1e-9, \
            f"fwd mismatch at idx {idx}: {actual_fwd} vs {expected_fwd}"

    # 3. Label values must be in {-1, 0, 1}
    valid_vals = {-1.0, 0.0, 1.0}
    actual_vals = set(lab["y"].dropna().unique())
    assert actual_vals.issubset(valid_vals), f"Invalid labels found: {actual_vals}"

    # 4. Class balance (non-dead-zone rows)
    up = (lab["y"] == 1.0).sum()
    down = (lab["y"] == -1.0).sum()
    pct_up = up / (up + down)
    assert 0.40 <= pct_up <= 0.60, f"Heavily unbalanced labels: {pct_up:.1%}"


if __name__ == "__main__":
    test_labels_alignment()
    print("test_labels_alignment PASSED!")
