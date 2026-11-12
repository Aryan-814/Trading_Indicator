"""Feature calculation per spec section 6.

Shared between training and live serving to prevent mismatch.
24 features total; no SPY/QQQ market context (data not available).
"""
import os
import numpy as np
import pandas as pd

ret_ks = [3, 6, 12, 24]

feat_cols = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "gap", "day_ret", "vol_12", "vol_78",
    "range_1", "range_12", "vwap_dist", "sess_vwap_dist",
    "rel_volume_log", "rel_tc_log",
    "tod", "tod_sin", "tod_cos", "dow", "is_half", "bars_left",
    "prev_day_ret", "prev_day_range", "rsi_14",
]

meta_cols = [
    "p_up", "p_dist", "vol_12", "vol_78", "tod",
    "rel_volume_log", "range_12", "is_half", "bars_left",
]


def load_bars(path):
    """Load a bar file; resolves relative paths from the project root."""
    if not os.path.exists(path):
        for candidate in [os.path.join("..", path), os.path.join("data", "clean", path)]:
            if os.path.exists(candidate):
                path = candidate
                break
    if path.endswith(".csv"):
        d = pd.read_csv(path)
    else:
        d = pd.read_parquet(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").reset_index(drop=True)
    return d


def wilder_rsi(close, n=14):
    """Wilder RSI using EWM (causal, no look-ahead)."""
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def make_features(d):
    """Build the 24-column feature frame for all bars in d.

    Guaranteed: feature at bar t uses only bars <= t (no look-ahead).

    Parameters
    ----------
    d : clean DataFrame from load_bars / clean.py

    Returns
    -------
    DataFrame with columns == feat_cols, same index as d.
    """
    g = d.groupby("day")
    f = pd.DataFrame(index=d.index)
    day_open = g["open"].transform("first")

    f["ret_1"] = d["ret_1"]
    for k in ret_ks:
        ref = g["close"].shift(k).fillna(day_open)
        f["ret_" + str(k)] = np.log(d["close"] / ref)

    f["gap"] = d["gap"]
    f["day_ret"] = np.log(d["close"] / day_open)
    f["vol_12"] = d["ret_1"].rolling(12).std()
    f["vol_78"] = d["ret_1"].rolling(78).std()

    f["range_1"] = (d["high"] - d["low"]) / d["close"]
    f["range_12"] = f["range_1"].rolling(12).mean()
    f["vwap_dist"] = np.log(d["close"] / d["vwap"])

    pv = d["vwap"] * d["volume"]
    cum_pv = pv.groupby(d["day"]).cumsum()
    cum_v = d["volume"].groupby(d["day"]).cumsum()
    sess_vwap = cum_pv / cum_v.replace(0, np.nan)
    f["sess_vwap_dist"] = np.log(d["close"] / sess_vwap)

    f["rel_volume_log"] = np.log(d["rel_volume"].clip(0.05, 20))
    f["rel_tc_log"] = np.log(d["rel_trade_count"].clip(0.05, 20))

    tod = d["bar_idx"] / d["day_bars"]
    f["tod"] = tod
    f["tod_sin"] = np.sin(2 * np.pi * tod)
    f["tod_cos"] = np.cos(2 * np.pi * tod)
    f["dow"] = pd.to_datetime(d["day"]).dt.day_of_week
    f["is_half"] = d["is_half"]
    f["bars_left"] = d["bars_left"]

    daily = d.groupby("day").agg(
        c=("close", "last"), h=("high", "max"), l=("low", "min")
    )
    daily["ret"] = np.log(daily["c"] / daily["c"].shift(1))
    daily["rng"] = (daily["h"] - daily["l"]) / daily["c"]
    f["prev_day_ret"] = d["day"].map(daily["ret"].shift(1))
    f["prev_day_range"] = d["day"].map(daily["rng"].shift(1))

    f["rsi_14"] = wilder_rsi(d["close"], 14)

    return f[feat_cols]
