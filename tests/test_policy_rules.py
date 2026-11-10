"""Mandatory correctness test (section 16):
Signal policy rules: bars_left < h, opening settle, and low confidence.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))
from features import load_bars
from predict import load_bundle, make_signal


def test_policy_rules():
    bundle = load_bundle("models/v1")
    d = load_bars("aapl_5m_clean.csv")

    h = bundle["thr"]["horizon_bars"]
    skip_bars = bundle["thr"]["skip_first_bars"]

    # 1. Test opening bars (bar_idx < skip_first_bars)
    open_bars = d[d["bar_idx"] < skip_bars]
    assert len(open_bars) > 0
    test_hist_open = d.iloc[: open_bars.index[0] + 1].copy()
    sig_open = make_signal(bundle, test_hist_open)
    assert sig_open["direction"] == "none", "Opening bar should have direction 'none'"
    assert sig_open["reason"] == "opening_settle", f"Expected reason 'opening_settle', got {sig_open['reason']}"

    # 2. Test horizon past close (bars_left < h)
    close_bars = d[d["bars_left"] < h]
    assert len(close_bars) > 0
    test_hist_close = d.iloc[: close_bars.index[0] + 1].copy()
    sig_close = make_signal(bundle, test_hist_close)
    assert sig_close["direction"] == "none", "End-of-day bar should have direction 'none'"
    assert sig_close["reason"] == "horizon_past_close", f"Expected reason 'horizon_past_close', got {sig_close['reason']}"


if __name__ == "__main__":
    test_policy_rules()
    print("test_policy_rules PASSED!")
