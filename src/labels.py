"""Label definition and checks per spec section 7."""
import numpy as np
import pandas as pd


def make_labels(d, h=6, kdz=0.15):
    """Compute primary training labels and scoring labels for bar t.

    Parameters
    ----------
    d : DataFrame with bars (must contain 'day', 'close', 'ret_1', 'bars_left')
    h : prediction horizon in bars
    kdz : dead-zone multiplier in units of horizon volatility

    Returns
    -------
    DataFrame with columns:
      - fwd: forward return over h bars
      - s: horizon volatility scale (sigma_t * sqrt(h))
      - y: primary training label (+1, -1, 0, or NaN)
      - y_sign: scoring label (sign of fwd_ret, NaN if fwd==0 or bars_left < h)
    """
    g = d.groupby("day")
    # Forward return within the day
    fwd = np.log(g["close"].shift(-h) / d["close"])

    # Exponentially weighted standard deviation of ret_1 (span 78, causal)
    sigma = d["ret_1"].ewm(span=78, min_periods=30).std()
    s = sigma * np.sqrt(h)

    # Primary label with volatility-scaled dead-zone
    y = pd.Series(0.0, index=d.index)
    y[fwd > kdz * s] = 1.0
    y[fwd < -kdz * s] = -1.0

    # Ensure bars_left < h or warm-up NaN get NaN label
    mask_nan = fwd.isna() | s.isna()
    if "bars_left" in d.columns:
        mask_nan = mask_nan | (d["bars_left"] < h)
    y[mask_nan] = np.nan

    # Scoring label: sign of fwd_ret (excluding 0)
    y_sign = np.sign(fwd)
    y_sign[fwd == 0] = np.nan
    y_sign[mask_nan] = np.nan

    return pd.DataFrame({
        "fwd": fwd,
        "s": s,
        "y": y,
        "y_sign": y_sign,
    })


def label_statistics(d, lab):
    """Compute label statistics per spec section 7.3."""
    out = {}
    valid = lab["y"].notna()
    y_val = lab.loc[valid, "y"]
    out["total_valid"] = int(valid.sum())
    out["total_dropped"] = int((~valid).sum())
    out["up_count"] = int((y_val == 1.0).sum())
    out["down_count"] = int((y_val == -1.0).sum())
    out["dead_zone_count"] = int((y_val == 0.0).sum())

    non_zero = out["up_count"] + out["down_count"]
    if non_zero > 0:
        out["up_pct"] = float(out["up_count"] / non_zero)
        out["down_pct"] = float(out["down_count"] / non_zero)
    else:
        out["up_pct"] = 0.0
        out["down_pct"] = 0.0

    return out

