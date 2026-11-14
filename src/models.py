"""Model loading and live signal generation per spec section 9.7.

This module is the single source of truth for load_bundle() and make_signal().
predict.py is a thin CLI that imports from here.
"""
import hashlib
import json
import os
import sys
import warnings

# Ensure src/ is importable when loaded from any working directory.
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import feat_cols, meta_cols, make_features


def load_bundle(path="models/v1"):
    """Load a trained model bundle from *path*.

    Supports only LightGBM native format (.txt). The sklearn fallback
    for primary/meta was removed; only calibrator.joblib remains as joblib.

    Parameters
    ----------
    path : str
        Directory produced by train.py refit.

    Returns
    -------
    dict with keys: format, primary, meta, q, cal, thr, info, feature_list
    """
    if not os.path.exists(path) and os.path.exists(os.path.join("..", path)):
        path = os.path.join("..", path)

    b = {}
    txt_primary = os.path.join(path, "primary.txt")
    if not os.path.exists(txt_primary):
        raise FileNotFoundError(
            f"primary.txt not found in {path}. Run 'python train.py refit' first."
        )

    b["format"] = "lightgbm"
    b["primary"] = lgb.Booster(model_file=txt_primary)
    b["meta"] = lgb.Booster(model_file=os.path.join(path, "meta.txt"))
    b["q"] = {
        a: lgb.Booster(model_file=os.path.join(path, f"quantile_{a}.txt"))
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

    analog_path = os.path.join(path, "analog_table.parquet")
    if os.path.exists(analog_path):
        b["analog_table"] = pd.read_parquet(analog_path)
    else:
        b["analog_table"] = None

    return b


def make_signal(b, hist):
    """Generate a live signal for the last bar in *hist*.

    Parameters
    ----------
    b    : bundle dict from load_bundle()
    hist : DataFrame of clean 5-min bars up to and including the current bar

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

    f = make_features(hist)
    row = f.iloc[[-1]].copy()

    used_feat = b.get("feature_list", {}).get("feat_cols", feat_cols)
    used_meta = b.get("feature_list", {}).get("meta_cols", meta_cols)

    out["feature_hash"] = hashlib.sha256(
        row[used_feat].to_numpy(dtype="float64").tobytes()
    ).hexdigest()

    p_up = float(b["primary"].predict(row[used_feat])[0])
    side_dir = "up" if p_up >= 0.5 else "down"
    side = 1 if side_dir == "up" else -1

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

    m = float(b["meta"].predict(mrow)[0])

    # Side-conditioned thresholds and calibration (Section 1 & User spec)
    base_tau = float(thr.get("tau_m", 0.521784452254343))
    tau_m_up = float(thr.get("tau_m_up", base_tau))
    tau_m_down = float(thr.get("tau_m_down", base_tau))

    if base_tau <= 0.0:
        active_tau = 0.0
    elif base_tau >= 1.0:
        active_tau = 1.0
    else:
        active_tau = tau_m_up if side_dir == "up" else tau_m_down

    # Map calibrated confidence so both UP and DOWN signals display in matching 0.51..0.58 range
    if side_dir == "down" and tau_m_down < base_tau:
        cal_input = m + (base_tau - tau_m_down)
    else:
        cal_input = m
    conf = float(b["cal"].predict([cal_input])[0])

    out["p_up"] = p_up
    out["meta_prob"] = m

    # Quantile price targets — Bear / Base / Bull + Regime & Diagnostics
    raw_qs = [float(b["q"][a].predict(row[used_feat])[0]) for a in (10, 50, 90)]
    qs = np.sort(raw_qs)
    close = float(last["close"])

    v78 = float(row["vol_78"].values[0]) if "vol_78" in row and not pd.isna(row["vol_78"].values[0]) else 0.004
    vol_cutoffs = thr.get("vol_cutoffs", {})
    v_q33 = vol_cutoffs.get("v_q33", 0.001165)
    v_q66 = vol_cutoffs.get("v_q66", 0.001649)
    if v78 <= v_q33:
        vol_bin = "low"
    elif v78 <= v_q66:
        vol_bin = "mid"
    else:
        vol_bin = "high"

    b_idx = int(last["bar_idx"])
    if b_idx <= 11:
        time_bin = "open"
    elif b_idx <= 38:
        time_bin = "mid"
    elif b_idx <= 65:
        time_bin = "afternoon"
    else:
        time_bin = "close"

    # Compute move_bin based on base quantile expected return
    s_t = v78 * np.sqrt(h / 78) if v78 > 0 else 0.001
    fwd_est = float(qs[1])
    z_fwd = fwd_est / s_t
    if z_fwd <= -1.0:
        move_bin = "strong_down"
    elif z_fwd <= -0.3:
        move_bin = "mild_down"
    elif z_fwd <= 0.3:
        move_bin = "flat"
    elif z_fwd <= 1.0:
        move_bin = "mild_up"
    else:
        move_bin = "strong_up"

    side_dir = "up" if p_up >= 0.5 else "down"
    conf_pct = conf * 100.0
    horizon_minutes = h * 5

    narrative = None
    if b.get("analog_table") is not None:
        try:
            from scenarios import lookup_analog_narrative
            narrative = lookup_analog_narrative(
                analog_table=b["analog_table"],
                vol_bin=vol_bin,
                time_bin=time_bin,
                move_bin=move_bin,
                side_direction=side_dir,
                conf_pct=conf_pct,
                horizon_min=horizon_minutes,
            )
        except Exception as exc:
            warnings.warn(f"lookup_analog_narrative failed: {exc}")

    out["scenarios"] = {
        "bear_q10": close * float(np.exp(qs[0])),
        "base_q50": close * float(np.exp(qs[1])),
        "bull_q90": close * float(np.exp(qs[2])),
        "regime": {"vol_bin": vol_bin, "time_bin": time_bin},
        "diagnostic_confidence": conf,
        "lean": side_dir,
        "narrative": narrative,
    }

    # Spec 10.4: top-5 feature drivers by SHAP contribution
    feat_contrib_map = {}
    try:
        contribs = b["primary"].predict(row[used_feat], pred_contrib=True)[0]
        feat_contribs = contribs[:-1]  # last element is bias
        feat_contrib_map = dict(zip(used_feat, feat_contribs))
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
    except (ValueError, KeyError, AttributeError) as exc:
        warnings.warn(f"pred_contrib failed: {exc}")
        out["drivers"] = []

    # Down side also requires p_up <= 0.495 so flat 0.499 bars do not trigger a short
    is_side_eligible = (side_dir == "up") or (side_dir == "down" and p_up <= 0.495)

    # Intraday SHAP Driver Agreement Gate:
    # Prevents static prior-day features from forcing counter-trend signals against intraday momentum
    day_ret_val = (
        float(row["day_ret"].values[0])
        if "day_ret" in row and not pd.isna(row["day_ret"].values[0])
        else 0.0
    )
    rsi_val = (
        float(row["rsi_14"].values[0])
        if "rsi_14" in row and not pd.isna(row["rsi_14"].values[0])
        else 50.0
    )
    day_ret_contrib = feat_contrib_map.get("day_ret", 0.0)
    rsi_contrib = feat_contrib_map.get("rsi_14", 0.0)

    suppress_driver_conflict = False
    if side_dir == "up":
        if day_ret_contrib < -0.025 or rsi_contrib < -0.025 or day_ret_val < -0.005:
            suppress_driver_conflict = True
    elif side_dir == "down":
        if day_ret_val < 0.0 and rsi_val < 53.0:
            suppress_driver_conflict = True

    # Policy Gates: suppress trade signal when conditions are met
    if last["bars_left"] < h:
        out["direction"] = "none"
        out["confidence"] = None
        out["reason"] = "horizon_past_close"
    elif last["bar_idx"] < thr["skip_first_bars"]:
        out["direction"] = "none"
        out["confidence"] = None
        out["reason"] = "opening_settle"
    elif m >= active_tau and is_side_eligible and not (suppress_driver_conflict and base_tau > 0.0):
        out["direction"] = side_dir
        out["confidence"] = conf
        out["reason"] = None
    else:
        out["direction"] = "none"
        out["confidence"] = None
        out["reason"] = "low_confidence"

    return out
