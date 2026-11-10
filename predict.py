import json
import os
import sys
import hashlib

sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from features import feat_cols, meta_cols, load_bars, make_features


def load_bundle(path="models/v1"):
    if not os.path.exists(path) and os.path.exists(os.path.join("..", path)):
        path = os.path.join("..", path)

    b = {}
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


def make_signal(b, hist, spy=None, qqq=None):
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

    # Determine which feature columns to feed into the primary model
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

    if b["format"] == "lightgbm":
        p_up = float(b["primary"].predict(row[used_feat])[0])
    else:
        p_up = float(b["primary"].predict_proba(row[used_feat])[0, 1])

    side = 1 if p_up >= 0.5 else -1
    mrow = pd.DataFrame({
        "p_up": [p_up],
        "p_dist": [abs(p_up - 0.5)],
        "vol_12": row["vol_12"].values,
        "vol_78": row["vol_78"].values,
        "tod": row["tod"].values,
        "rel_volume_log": row["rel_volume_log"].values,
        "range_12": row["range_12"].values,
        "is_half": row["is_half"].values,
        "bars_left": row["bars_left"].values,
    })[used_meta]

    if b["format"] == "lightgbm":
        m = float(b["meta"].predict(mrow)[0])
    else:
        m = float(b["meta"].predict_proba(mrow)[0, 1])

    conf = float(b["cal"].predict([m])[0])

    out["p_up"] = p_up
    out["meta_prob"] = m

    if b["format"] == "lightgbm":
        qs = np.sort([float(b["q"][a].predict(row[used_feat])[0]) for a in (10, 50, 90)])
    else:
        qs = np.sort([float(b["q"][a].predict(row[used_feat])[0]) for a in (10, 50, 90)])

    close = float(last["close"])
    out["scenarios"] = {
        "bear_q10": close * float(np.exp(qs[0])),
        "base_q50": close * float(np.exp(qs[1])),
        "bull_q90": close * float(np.exp(qs[2])),
    }

    # Spec 10.4: feature drivers (top 5 by contribution)
    if b["format"] == "lightgbm":
        try:
            contribs = b["primary"].predict(row[used_feat], pred_contrib=True)[0]
            feat_contribs = contribs[:-1]
            top_idx = np.argsort(np.abs(feat_contribs))[-5:][::-1]
            out["drivers"] = [
                {
                    "feature": used_feat[i],
                    "contribution": float(feat_contribs[i]),
                    "value": float(row[used_feat[i]].values[0]) if not pd.isna(row[used_feat[i]].values[0]) else None,
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


if __name__ == "__main__":
    bundle = load_bundle()
    print(f"Loaded bundle format: {bundle['format']}")
    bars = load_bars("aapl_5m_clean.csv")
    tail = bars.iloc[-3000:].reset_index(drop=True)
    n_sig = 0
    n_all = 0
    for i in range(len(tail) - 200, len(tail)):
        hist = tail.iloc[: i + 1].reset_index(drop=True)
        s = make_signal(bundle, hist)
        n_all += 1
        if s["direction"] != "none":
            n_sig += 1
    print("last candle:", make_signal(bundle, tail))
    print("signals on last 200 candles:", n_sig, "of", n_all)
