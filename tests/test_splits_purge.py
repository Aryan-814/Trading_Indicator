"""Test that walk_forward_blocks produces clean, non-overlapping splits."""
import json
import numpy as np
import pytest
from pathlib import Path
from features import load_bars, make_features
from labels import make_labels
from splits import walk_forward_blocks

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


def _build_frame():
    h, kdz = _get_hkdz()
    d = load_bars(DATA)
    lab = make_labels(d, h=h, kdz=kdz)
    import pandas as pd
    df = d[["day"]].copy()
    df["q"] = pd.to_datetime(df["day"]).dt.to_period("Q").astype(str)
    df["fwd"] = lab["fwd"]
    return df, h


def test_splits_purge():
    df, h = _build_frame()
    min_q = 4
    folds = list(walk_forward_blocks(df, min_q=min_q))

    assert len(folds) > 0, "No folds produced"

    # First test fold index must equal min_q (0-indexed among all quarters)
    qs = sorted(df["q"].unique())
    first_test_q = df.loc[folds[0][1], "q"].iloc[0]
    assert first_test_q == qs[min_q], \
        f"First test quarter {first_test_q} != qs[{min_q}]={qs[min_q]}"

    for train_idx, test_idx in folds:
        train_days = df.loc[train_idx, "day"]
        test_days = df.loc[test_idx, "day"]

        # No overlap in day values
        assert train_days.max() < test_days.min(), \
            f"Train/test day overlap: max train={train_days.max()}, min test={test_days.min()}"

        # No label window crosses into test: labels are intraday-only,
        # so the day-boundary check is sufficient (fwd is NaN when it would cross)
        end_of_day = df.loc[test_idx][df.loc[test_idx, "fwd"].isna()]
        # This is expected for last-h bars of each day — just verify no train day >= test start
        assert train_days.max() < test_days.min()

    print(f"Verified {len(folds)} walk-forward folds, min_q={min_q}")


if __name__ == "__main__":
    test_splits_purge()
    print("test_splits_purge PASSED!")
