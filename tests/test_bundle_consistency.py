"""Test bundle internal consistency: feature names, no all-NaN cols, hashes."""
import hashlib
import json
import os
import pytest
from pathlib import Path
import lightgbm as lgb
from features import load_bars, make_features, feat_cols

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = str(ROOT / "models" / "v1")
DATA = str(ROOT / "data" / "clean" / "aapl_5m_clean.parquet")


def test_feature_names_match_booster():
    """feature_list.json must match feat_cols and the booster's own feature names."""
    with open(os.path.join(BUNDLE, "feature_list.json")) as fh:
        fl = json.load(fh)
    assert fl["feat_cols"] == feat_cols, (
        f"Bundle vs src mismatch:\n  json: {fl['feat_cols']}\n  src: {feat_cols}"
    )
    booster = lgb.Booster(model_file=os.path.join(BUNDLE, "primary.txt"))
    assert fl["feat_cols"] == booster.feature_name(), (
        f"Mismatch:\n  json: {fl['feat_cols']}\n  booster: {booster.feature_name()}"
    )


def test_no_all_nan_feature():
    """No feature column should be entirely NaN on the full training frame."""
    bars = load_bars(DATA)
    f = make_features(bars)
    all_nan = [c for c in feat_cols if f[c].isna().all()]
    assert all_nan == [], f"All-NaN columns found: {all_nan}"


def test_bundle_hashes():
    """SHA-256 hashes in bundle_info.json must match the actual files."""
    with open(os.path.join(BUNDLE, "bundle_info.json")) as fh:
        info = json.load(fh)
    for fname, expected in info.get("sha256", {}).items():
        fpath = os.path.join(BUNDLE, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        assert actual == expected, f"Hash mismatch for {fname}"


if __name__ == "__main__":
    test_feature_names_match_booster()
    test_no_all_nan_feature()
    test_bundle_hashes()
    print("test_bundle_consistency PASSED!")

