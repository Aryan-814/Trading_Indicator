"""Model loading helpers per spec section 9.7.

This module exposes load_bundle() and make_signal() that are shared between
predict.py and any downstream consumer.  predict.py is the canonical entry
point; this module houses the reusable logic so tests can import it without
running __main__.
"""
# NOTE: predict.py duplicates load_bundle / make_signal for standalone use.
# If you change the bundle layout here, update predict.py as well.
import hashlib
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from features import feat_cols, meta_cols, make_features  # noqa: F401
except ImportError:  # allow import without src/ on path (tests add it)
    feat_cols = []
    meta_cols = []


def load_bundle(path="models/v1"):
    """Load a trained model bundle from *path*.

    Supports both LightGBM native format (.txt) and sklearn joblib format
    for backwards compatibility.

    Parameters
    ----------
    path : str
        Directory containing the bundle files produced by train.py refit.

    Returns
    -------
    dict with keys: format, primary, meta, q, cal, thr, info, feature_list
    """
    if not os.path.exists(path) and os.path.exists(os.path.join("..", path)):
        path = os.path.join("..", path)

    b: dict = {}
    txt_primary = os.path.join(path, "primary.txt")
    if os.path.exists(txt_primary):
        b["format"] = "lightgbm"
        b["primary"] = lgb.Booster(model_file=txt_primary)
        b["meta"] = lgb.Booster(model_file=os.path.join(path, "meta.txt"))
        b["q"] = {
            a: lgb.Booster(model_file=os.path.join(path, f"quantile_{a}.txt"))
            for a in (10, 50, 90)
        }
    else:
        b["format"] = "sklearn"
        b["primary"] = joblib.load(os.path.join(path, "primary.joblib"))
        b["meta"] = joblib.load(os.path.join(path, "meta.joblib"))
        b["q"] = {
            a: joblib.load(os.path.join(path, f"quantile_{a}.joblib"))
            for a in (10, 50, 90)
        }

    b["cal"] = joblib.load(os.path.join(path, "calibrator.joblib"))

    import json

    with open(os.path.join(path, "thresholds.json")) as fh:
        b["thr"] = json.load(fh)
    with open(os.path.join(path, "bundle_info.json")) as fh:
        b["info"] = json.load(fh)

    fl_path = os.path.join(path, "feature_list.json")
    if os.path.exists(fl_path):
        with open(fl_path) as fh:
            b["feature_list"] = json.load(fh)
    else:
        b["feature_list"] = {"feat_cols": feat_cols, "meta_cols": meta_cols}

    return b


def _predict_primary(b, row, used_feat):
    if b["format"] == "lightgbm":
        return float(b["primary"].predict(row[used_feat])[0])
    return float(b["primary"].predict_proba(row[used_feat])[0, 1])


def _predict_meta(b, mrow):
    if b["format"] == "lightgbm":
        return float(b["meta"].predict(mrow)[0])
    return float(b["meta"].predict_proba(mrow)[0, 1])


def make_signal(b, hist, spy=None, qqq=None):
    """Generate a live signal for the last bar in *hist*.

    Parameters
    ----------
    b    : bundle dict from load_bundle()
    hist : DataFrame of clean 5-min bars up to and including the current bar
    spy  : optional SPY bar DataFrame aligned to AAPL timestamps
    qqq  : optional QQQ bar DataFrame aligned to AAPL timestamps

    Returns
    -------
    dict with keys:
      candle_ts, horizon_bars, direction, confidence, p_up, meta_prob,
      reason, scenarios, drivers, feature_hash
    """
    thr = b["thr"]
    h = thr["horizon_bars"]
    last = hist.iloc[-1]

    out = {
        "candle_ts": str(last["timestamp"]),
        "horizon_bars": h,
        "direction": "none",
        "confidence": None,
        "p_up": None,
        "meta_prob": None,
        "reason": None,
        "scenarios": None,
        "drivers": [],
    }

    f = make_features(hist, spy=spy, qqq=qqq)
    row = f.iloc[[-1]].copy()

    used_feat = b.get("feature_list", {}).get("feat_cols", feat_cols)
    used_meta = b.get("feature_list", {}).get("meta_cols", meta_cols)

    out["feature_hash"] = hashlib.sha256(
        row[used_feat].to_numpy(dtype="float64").tobytes()
    ).hexdigest()

    if last["bars_left"] < h:
        out["reason"] = "horizon_past_close"
        return out
    if last["bar_idx"] < thr["skip_first_bars"]:
        out["reason"] = "opening_settle"
        return out

    p_up = _predict_primary(b, row, used_feat)
    side = 1 if p_up >= 0.5 else -1

    mrow = pd.DataFrame(
        {
            "p_up": [p_up],
            "p_dist": [abs(p_up - 0.5)],
            "vol_12": row["vol_12"].values,
            "vol_78": row["vol_78"].values,
            "tod": row["tod"].values,
            "rel_volume_log": row["rel_volume_log"].values,
            "range_12": row["range_12"].values,
            "is_half": row["is_half"].values,
            "bars_left": row["bars_left"].values,
        }
    )[used_meta]

    m = _predict_meta(b, mrow)
    conf = float(b["cal"].predict([m])[0])

    out["p_up"] = p_up
    out["meta_prob"] = m

    # Quantile price targets
    if b["format"] == "lightgbm":
        raw_qs = [float(b["q"][a].predict(row[used_feat])[0]) for a in (10, 50, 90)]
    else:
        raw_qs = [float(b["q"][a].predict(row[used_feat])[0]) for a in (10, 50, 90)]
    qs = np.sort(raw_qs)
    close = float(last["close"])
    out["scenarios"] = {
        "bear_q10": close * float(np.exp(qs[0])),
        "base_q50": close * float(np.exp(qs[1])),
        "bull_q90": close * float(np.exp(qs[2])),
    }

    # Spec 10.4: top-5 feature drivers by SHAP contribution
    if b["format"] == "lightgbm":
        try:
            contribs = b["primary"].predict(row[used_feat], pred_contrib=True)[0]
            feat_contribs = contribs[:-1]  # last element is bias
            top_idx = np.argsort(np.abs(feat_contribs))[-5:][::-1]
            out["drivers"] = [
                {
                    "feature": used_feat[i],
                    "contribution": float(feat_contribs[i]),
                    "value": (
                        float(row[used_feat[i]].values[0])
                        if not pd.isna(row[used_feat[i]].values[0])
                        else None
                    ),
                }
                for i in top_idx
            ]
        except Exception:
            out["drivers"] = []

    if m >= thr["tau_m"]:
        out["direction"] = "up" if side == 1 else "down"
        out["confidence"] = conf
    else:
        out["reason"] = "low_confidence"

    return out

