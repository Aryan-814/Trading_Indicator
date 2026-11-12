"""Mandatory correctness test: signal policy rules."""
import copy
import json
import pytest
from pathlib import Path
from features import load_bars
from models import load_bundle, make_signal

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "data" / "clean" / "aapl_5m_clean.parquet")
BUNDLE = str(ROOT / "models" / "v1")


def test_policy_rules():
    bundle = load_bundle(BUNDLE)
    d = load_bars(DATA)

    h = bundle["thr"]["horizon_bars"]
    skip_bars = bundle["thr"]["skip_first_bars"]
    tau_m = bundle["thr"]["tau_m"]

    # 1. Opening bars → opening_settle
    open_bars = d[d["bar_idx"] < skip_bars]
    assert len(open_bars) > 0
    test_hist_open = d.iloc[: open_bars.index[0] + 1].copy()
    sig_open = make_signal(bundle, test_hist_open)
    assert sig_open["direction"] == "none"
    assert sig_open["reason"] == "opening_settle", sig_open

    # 2. End-of-day bars → horizon_past_close
    close_bars = d[d["bars_left"] < h]
    assert len(close_bars) > 0
    test_hist_close = d.iloc[: close_bars.index[0] + 1].copy()
    sig_close = make_signal(bundle, test_hist_close)
    assert sig_close["direction"] == "none"
    assert sig_close["reason"] == "horizon_past_close", sig_close

    # 3. Force tau_m = 1.0 → low_confidence on a mid-day bar
    b_high_tau = copy.deepcopy(bundle)
    b_high_tau["thr"] = dict(bundle["thr"])
    b_high_tau["thr"]["tau_m"] = 1.0
    eligible = d[(d["bar_idx"] >= skip_bars) & (d["bars_left"] >= h)]
    test_hist_mid = d.iloc[: eligible.index[0] + 1].copy()
    sig_low = make_signal(b_high_tau, test_hist_mid)
    assert sig_low["direction"] == "none"
    assert sig_low["reason"] == "low_confidence", sig_low

    # 4. Force tau_m = 0.0 → signal fires, direction in {up, down}, confidence in [0, 1]
    b_zero_tau = copy.deepcopy(bundle)
    b_zero_tau["thr"] = dict(bundle["thr"])
    b_zero_tau["thr"]["tau_m"] = 0.0
    sig_up = make_signal(b_zero_tau, test_hist_mid)
    assert sig_up["direction"] in {"up", "down"}, sig_up
    assert sig_up["confidence"] is not None
    assert 0.0 <= sig_up["confidence"] <= 1.0, sig_up

    # 5. All documented keys are present
    required_keys = {
        "candle_ts", "horizon_bars", "direction", "confidence",
        "p_up", "meta_prob", "reason", "scenarios", "drivers", "feature_hash",
    }
    assert required_keys.issubset(set(sig_up.keys())), \
        f"Missing keys: {required_keys - set(sig_up.keys())}"


if __name__ == "__main__":
    test_policy_rules()
    print("test_policy_rules PASSED!")
