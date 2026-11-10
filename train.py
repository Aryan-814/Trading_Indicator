import sys
import os

sys.path.insert(0, os.path.abspath("src"))
import json
import time
import hashlib

import numpy as np
import pandas as pd
import yaml
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from features import feat_cols, meta_cols, load_bars, make_features
from labels import make_labels

# Load config.yaml if present
config_file = "config.yaml"
if not os.path.exists(config_file) and os.path.exists(os.path.join("..", config_file)):
    config_file = os.path.join("..", config_file)

yaml_cfg = {}
if os.path.exists(config_file):
    with open(config_file) as f:
        yaml_cfg = yaml.safe_load(f) or {}

cfg = {
    "data_path": "aapl_5m_clean.csv",
    "out_dir": "models/v1",
    "log_path": "reports/experiments.csv",
    "best_path": "reports/best_cfg.json",
    "train_end": yaml_cfg.get("periods", {}).get("train_end", "2024-12-31"),
    "val_start": yaml_cfg.get("periods", {}).get("val_start", "2025-01-01"),
    "val_end": yaml_cfg.get("periods", {}).get("val_end", "2025-12-31"),
    "holdout_start": yaml_cfg.get("periods", {}).get("holdout_start", "2026-01-01"),
    "skip_first_bars": yaml_cfg.get("policy", {}).get("skip_first_bars", 3),
    "target_coverage": yaml_cfg.get("policy", {}).get("target_coverage", 0.20),
    "ece_max": 0.03,
    "cost_bps": [0.5, 1.0, 3.0],
    "n_perm": 2000,
    "seed": yaml_cfg.get("primary", {}).get("seed", 42),
}

# Resolve data_path if not in current directory
if not os.path.exists(cfg["data_path"]) and os.path.exists(os.path.join("..", cfg["data_path"])):
    cfg["data_path"] = os.path.join("..", cfg["data_path"])

pr_params = dict(
    num_leaves=yaml_cfg.get("primary", {}).get("num_leaves", 15),
    max_depth=yaml_cfg.get("primary", {}).get("max_depth", 5),
    learning_rate=yaml_cfg.get("primary", {}).get("learning_rate", 0.03),
    n_estimators=yaml_cfg.get("primary", {}).get("n_estimators", 200),
    min_child_samples=yaml_cfg.get("primary", {}).get("min_child_samples", 300),
    subsample=yaml_cfg.get("primary", {}).get("subsample", 0.7),
    subsample_freq=1,
    colsample_bytree=yaml_cfg.get("primary", {}).get("colsample_bytree", 0.7),
    reg_lambda=yaml_cfg.get("primary", {}).get("reg_lambda", 5.0),
    reg_alpha=yaml_cfg.get("primary", {}).get("reg_alpha", 0.1),
    random_state=cfg["seed"],
    verbose=-1,
)
meta_params = dict(
    num_leaves=yaml_cfg.get("meta", {}).get("num_leaves", 7),
    max_depth=3,
    learning_rate=0.03,
    n_estimators=100,
    min_child_samples=yaml_cfg.get("meta", {}).get("min_child_samples", 500),
    reg_lambda=10.0,
    random_state=cfg["seed"],
    verbose=-1,
)
q_params = dict(
    num_leaves=15,
    max_depth=5,
    learning_rate=0.03,
    n_estimators=150,
    min_child_samples=300,
    reg_lambda=5.0,
    random_state=cfg["seed"],
    verbose=-1,
)

grid = [(h, kdz) for h in (2, 6, 12) for kdz in (0.15, 0.3)]


def build_frame(h, kdz):
    d = load_bars(cfg["data_path"])
    f = make_features(d)
    lab = make_labels(d, h, kdz)
    base = d[["day", "open", "close", "bar_idx"]].copy()
    base["next_open"] = d.groupby("day")["open"].shift(-1)
    df = pd.concat([base, f, lab], axis=1)
    df["q"] = pd.to_datetime(df["day"]).dt.to_period("Q").astype(str)
    df = df[df["vol_78"].notna()].reset_index(drop=True)
    return df


def elig(df):
    return df["fwd"].notna() & (df["bar_idx"] >= cfg["skip_first_bars"])


def fit_primary(train, shuffle=False, rng=None):
    rows = train[train["y"].isin([1, -1])]
    target = (rows["y"] == 1).astype(int).values
    if shuffle:
        target = rng.permutation(target)
    m = lgb.LGBMClassifier(**pr_params)
    m.fit(rows[feat_cols], target)
    return m


def oof_primary(df, h=6, min_q=4, shuffle=False):
    """Purged walk-forward primary predictions with block embargo."""
    rng = np.random.default_rng(cfg["seed"])
    qs = sorted(df["q"].unique())
    p = pd.Series(np.nan, index=df.index)
    for i in range(min_q, len(qs)):
        in_q = df["q"] == qs[i]
        test_start_day = df.loc[in_q, "day"].min()
        # purge: drop training bars whose label window overlaps test
        train = df[df["day"] < test_start_day]
        if len(train) == 0:
            continue
        test = df[in_q & elig(df)]
        if len(test) == 0:
            continue
        m = fit_primary(train, shuffle, rng)
        p[test.index] = m.predict_proba(test[feat_cols])[:, 1]
    return p


def add_side(df):
    df["side"] = np.where(df["p_up"] >= 0.5, 1, -1)
    df["p_dist"] = (df["p_up"] - 0.5).abs()
    ok = (np.sign(df["fwd"]) == df["side"]).astype(float)
    bad = (df["fwd"] == 0) | df["fwd"].isna() | df["p_up"].isna()
    ok[bad] = np.nan
    df["ok"] = ok


def oof_meta(df, h=6, min_q=2):
    """Purged walk-forward meta predictions."""
    rows = df[df["p_up"].notna() & df["ok"].notna()]
    qs = sorted(rows["q"].unique())
    m = pd.Series(np.nan, index=df.index)
    for i in range(min_q, len(qs)):
        test = rows[rows["q"] == qs[i]]
        test_start_day = test["day"].min()
        train = rows[rows["day"] < test_start_day]
        if len(train) == 0:
            continue
        mm = lgb.LGBMClassifier(**meta_params)
        mm.fit(train[meta_cols], train["ok"].astype(int))
        m[test.index] = mm.predict_proba(test[meta_cols])[:, 1]
    return m


def fit_stack(df):
    df["m_oof"] = oof_meta(df)
    rows = df[df["p_up"].notna() & df["ok"].notna()]
    meta = lgb.LGBMClassifier(**meta_params)
    meta.fit(rows[meta_cols], rows["ok"].astype(int))
    cal_rows = df[df["m_oof"].notna() & df["ok"].notna()]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(cal_rows["m_oof"], cal_rows["ok"])
    return meta, cal


def build_stack_on_train(df):
    tr = df[df["day"] <= cfg["train_end"]].copy()
    tr["p_up"] = oof_primary(tr)
    add_side(tr)
    meta, cal = fit_stack(tr)
    primary = fit_primary(tr)
    return primary, meta, cal, tr


def score_rows(df, primary, meta, cal):
    out = df[elig(df)].copy()
    out["p_up"] = primary.predict_proba(out[feat_cols])[:, 1]
    add_side(out)
    out["m"] = meta.predict_proba(out[meta_cols])[:, 1]
    out["conf"] = cal.predict(out["m"])
    return out


def fit_quant(df):
    rows = df[elig(df)]
    out = {}
    for a in (0.1, 0.5, 0.9):
        mm = lgb.LGBMRegressor(objective="quantile", alpha=a, **q_params)
        mm.fit(rows[feat_cols], rows["fwd"])
        out[a] = mm
    return out


def band_cov(qm, sc):
    q = np.column_stack([qm[a].predict(sc[feat_cols]) for a in (0.1, 0.5, 0.9)])
    q = np.sort(q, axis=1)
    below = float((sc["fwd"].values < q[:, 0]).mean())
    above = float((sc["fwd"].values > q[:, 2]).mean())
    return {"below_q10": below, "above_q90": above, "inside": 1 - below - above}


def wilson(p, n, z=1.96):
    if n <= 0:
        return [float("nan"), float("nan")]
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - w) / den), float((c + w) / den)]


def ece_score(conf, ok, nb=10):
    bins = pd.qcut(conf.rank(method="first"), nb, labels=False, duplicates="drop")
    tab = pd.DataFrame({"conf": conf.values, "ok": ok.values, "b": bins})
    g = tab.groupby("b").agg(
        conf=("conf", "mean"), ok=("ok", "mean"), n=("ok", "size")
    )
    ece = (g["n"] * (g["conf"] - g["ok"]).abs()).sum() / g["n"].sum()
    return float(ece), g.round(4).reset_index().to_dict("records")


def perm_test(ev, sel, h, n_perm):
    rng = np.random.default_rng(cfg["seed"])
    sgn = np.sign(ev["fwd"].values)
    side = ev["side"].values
    nb = len(sgn) // h
    cut = nb * h
    blocks = sgn[:cut].reshape(nb, h)
    side_c = side[:cut]
    sel_c = sel[:cut]
    obs = float((sgn[:cut][sel_c] == side_c[sel_c]).mean())
    count = 0
    for _ in range(n_perm):
        perm = blocks[rng.permutation(nb)].reshape(-1)
        acc = (perm[sel_c] == side_c[sel_c]).mean()
        if acc >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def evaluate(sc, h, tau_m):
    ev = sc[sc["ok"].notna()].sort_index().copy()
    res = {"n_rows": int(len(ev))}
    res["auc_primary"] = float(roc_auc_score(ev["fwd"] > 0, ev["p_up"]))
    res["acc_primary"] = float(ev["ok"].mean())
    res["brier_conf"] = float(brier_score_loss(ev["ok"], ev["conf"]))
    const = np.full(len(ev), ev["ok"].mean())
    res["brier_const"] = float(brier_score_loss(ev["ok"], const))
    res["ece"], res["reliability"] = ece_score(ev["conf"], ev["ok"])

    sel_tab = {}
    for cv in (0.05, 0.10, 0.15, 0.20, 0.25, 0.50):
        cut = ev["m"].quantile(1 - cv)
        s = ev[ev["m"] >= cut]
        acc = float(s["ok"].mean())
        sel_tab[str(cv)] = {
            "n": int(len(s)), "acc": acc,
            "ci95_neff": wilson(acc, len(s) / h),
        }
    res["selective"] = sel_tab

    up = ev["fwd"] > 0
    res["b1_always_up"] = float(up.mean())
    res["b3_momentum_ret6"] = float((np.sign(ev["ret_6"]) == np.sign(ev["fwd"])).mean())
    rsi_sig = ev[(ev["rsi_14"] < 30) | (ev["rsi_14"] > 70)]
    rsi_pred = np.where(rsi_sig["rsi_14"] < 30, 1, -1)
    res["b2_rsi_rule"] = {
        "coverage": float(len(rsi_sig) / len(ev)),
        "acc": float((rsi_pred == np.sign(rsi_sig["fwd"])).mean()) if len(rsi_sig) > 0 else float("nan"),
    }

    sel = (ev["m"] >= tau_m).values
    s = ev[sel]
    acc = float(s["ok"].mean()) if len(s) > 0 else float("nan")
    res["policy"] = {
        "tau_m": float(tau_m),
        "coverage": float(sel.mean()),
        "n_signals": int(sel.sum()),
        "acc": acc,
        "ci95_neff": wilson(acc, sel.sum() / h),
        "b1_on_same_candles": float((s["fwd"] > 0).mean()) if len(s) > 0 else float("nan"),
        "edge_vs_b1": float(acc - (s["fwd"] > 0).mean()) if len(s) > 0 else float("nan"),
    }

    pnl_mask = s["next_open"].notna()
    s_pnl = s[pnl_mask]
    n_pnl_excluded = int((~pnl_mask).sum())
    gross = 1e4 * (s_pnl["fwd"] + np.log(s_pnl["close"] / s_pnl["next_open"]))
    pnl = s_pnl["side"] * gross
    res["policy"]["pnl_bps"] = {}
    res["policy"]["pnl_n_excluded"] = n_pnl_excluded
    for c in cfg["cost_bps"]:
        net = pnl - 2 * c
        res["policy"]["pnl_bps"][str(c)] = {
            "mean_per_signal": float(net.mean()) if len(net) > 0 else 0.0,
            "sum": float(net.sum()),
            "n_signals_with_pnl": int(len(net)),
        }
    obs, pval = perm_test(ev, sel, h, cfg["n_perm"])
    res["policy"]["perm_acc"] = obs
    res["policy"]["perm_pvalue"] = float(pval)
    return res


def build_analog_table(df, h):
    """Build offline analog table from training data per spec section 10.3 step 2."""
    tr = df[df["day"] <= cfg["train_end"]].copy()
    v = tr["vol_78"].dropna()
    v_q33, v_q66 = float(v.quantile(1 / 3)), float(v.quantile(2 / 3))

    def get_vol_bin(val):
        if val <= v_q33:
            return "low_vol"
        elif val <= v_q66:
            return "mid_vol"
        return "high_vol"

    tr["vol_bin"] = tr["vol_78"].apply(get_vol_bin)

    def get_tod_bin(idx):
        if idx < 18:
            return "open"
        elif idx < 48:
            return "mid"
        elif idx < 72:
            return "afternoon"
        return "close"

    tr["time_bin"] = tr["bar_idx"].apply(get_tod_bin)

    tr["z_fwd"] = tr["fwd"] / tr["s"].replace(0, np.nan)
    move_bins = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]
    move_labels = ["strong_down", "mild_down", "flat", "mild_up", "strong_up"]
    tr["move_bin"] = pd.cut(tr["z_fwd"], bins=move_bins, labels=move_labels).astype(str)

    g = tr.groupby("day")
    tr["fwd_next"] = np.log(g["close"].shift(-2 * h) / g["close"].shift(-h))

    valid = tr[tr["fwd"].notna() & tr["fwd_next"].notna() & tr["s"].notna()]
    grouped = valid.groupby(["vol_bin", "time_bin", "move_bin"], observed=False)
    records = []
    for (vb, tb, mb), grp in grouped:
        n = len(grp)
        if n == 0:
            continue
        f2 = grp["fwd_next"].values
        f1 = grp["fwd"].values
        cont = float((np.sign(f2) == np.sign(f1)).mean())
        rev = float((np.sign(f2) == -np.sign(f1)).mean())
        records.append({
            "vol_bin": vb,
            "time_bin": tb,
            "move_bin": mb,
            "count": n,
            "median": float(np.median(f2)),
            "p25": float(np.percentile(f2, 25)),
            "p75": float(np.percentile(f2, 75)),
            "share_continuing": cont,
            "share_reversing": rev,
        })
    analog_df = pd.DataFrame(records)
    return analog_df, {"v_q33": v_q33, "v_q66": v_q66}


def log_trial(row):
    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(cfg["log_path"]):
        old = pd.read_csv(cfg["log_path"])
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(cfg["log_path"], index=False)


def run_val():
    rows = []
    for i, (h, kdz) in enumerate(grid):
        t0 = time.time()
        df = build_frame(h, kdz)
        primary, meta, cal, tr = build_stack_on_train(df)
        va = df[(df["day"] >= cfg["val_start"]) & (df["day"] <= cfg["val_end"])]
        sc = score_rows(va, primary, meta, cal)
        tau_m = float(sc.loc[sc["ok"].notna(), "m"].quantile(1 - cfg["target_coverage"]))
        res = evaluate(sc, h, tau_m)
        row = {
            "trial_id": i + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "validation",
            "model": "lightgbm",
            "horizon_bars": h,
            "label_type": "fixed_horizon",
            "kdz": kdz,
            "params_json": json.dumps(pr_params),
            "selective_acc_20": res["selective"]["0.2"]["acc"],
            "selective_acc_10": res["selective"]["0.1"]["acc"],
            "brier": res["brier_conf"],
            "ece": res["ece"],
            "auc": res["auc_primary"],
            "notes": "b1=%.4f" % res["b1_always_up"],
        }
        log_trial(row)
        rows.append(row)
        print("trial", i + 1, "h", h, "kdz", kdz,
              "sel20 %.4f" % row["selective_acc_20"],
              "ece %.4f" % row["ece"], "auc %.4f" % row["auc"],
              "b1 %.4f" % res["b1_always_up"],
              "secs %d" % (time.time() - t0), flush=True)
    ok_rows = [r for r in rows if r["ece"] <= cfg["ece_max"]]
    pool = ok_rows if ok_rows else rows
    best = max(pool, key=lambda r: r["selective_acc_20"])
    out = {"horizon_bars": best["horizon_bars"], "kdz": best["kdz"],
           "trial_id": best["trial_id"], "n_trials": len(rows)}
    os.makedirs(os.path.dirname(cfg["best_path"]), exist_ok=True)
    with open(cfg["best_path"], "w") as fh:
        json.dump(out, fh, indent=2)
    print("best", out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def run_holdout(df, h, kdz, best, force):
    mpath = os.path.join(cfg["out_dir"], "metrics.json")
    if os.path.exists(mpath) and not force:
        print("holdout already used once; metrics.json exists. stop.")
        return False
    primary, meta, cal, tr = build_stack_on_train(df)
    va = df[(df["day"] >= cfg["val_start"]) & (df["day"] <= cfg["val_end"])]
    sc_va = score_rows(va, primary, meta, cal)
    tau_m = float(sc_va.loc[sc_va["ok"].notna(), "m"].quantile(1 - cfg["target_coverage"]))
    ho = df[df["day"] >= cfg["holdout_start"]]
    sc_ho = score_rows(ho, primary, meta, cal)
    qm = fit_quant(tr)
    metrics = {
        "note": "validation and holdout scores come from the model trained "
                "through 2024 (before the final refit). the refit model has "
                "no out-of-sample score.",
        "horizon_bars": h, "kdz": kdz, "n_trials": best["n_trials"],
        "validation": evaluate(sc_va, h, tau_m),
        "holdout": evaluate(sc_ho, h, tau_m),
        "band_validation": band_cov(qm, sc_va),
        "band_holdout": band_cov(qm, sc_ho),
    }
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("holdout written once")
    return True


def run_refit(df, h, kdz):
    al = df.copy()
    al["p_up"] = oof_primary(al)
    add_side(al)
    meta_f, cal_f = fit_stack(al)
    primary_f = fit_primary(al)
    qm_f = fit_quant(al)

    vmask = (al["day"] >= cfg["val_start"]) & (al["day"] <= cfg["val_end"])
    vrows = al[vmask & al["ok"].notna()]
    m_val = meta_f.predict_proba(vrows[meta_cols])[:, 1]
    tau_f = float(np.quantile(m_val, 1 - cfg["target_coverage"]))

    od = cfg["out_dir"]
    os.makedirs(od, exist_ok=True)

    # 1. Models in LightGBM native format + calibrator
    primary_f.booster_.save_model(os.path.join(od, "primary.txt"))
    meta_f.booster_.save_model(os.path.join(od, "meta.txt"))
    joblib.dump(cal_f, os.path.join(od, "calibrator.joblib"))
    for a in (0.1, 0.5, 0.9):
        name = "quantile_%d.txt" % int(a * 100)
        qm_f[a].booster_.save_model(os.path.join(od, name))

    # Also keep .joblib wrappers for any legacy consumers
    joblib.dump(primary_f, os.path.join(od, "primary.joblib"))
    joblib.dump(meta_f, os.path.join(od, "meta.joblib"))
    for a in (0.1, 0.5, 0.9):
        name = "quantile_%d.joblib" % int(a * 100)
        joblib.dump(qm_f[a], os.path.join(od, name))

    # 2. Analog table (spec requirement D9 & 10.3)
    analog_table, vol_cutoffs = build_analog_table(df, h)
    analog_table.to_parquet(os.path.join(od, "analog_table.parquet"), index=False)

    # 3. Feature list
    with open(os.path.join(od, "feature_list.json"), "w") as fh:
        json.dump({"feat_cols": feat_cols, "meta_cols": meta_cols}, fh, indent=2)

    # 4. Thresholds
    thr = {
        "horizon_bars": h,
        "kdz": kdz,
        "tau_m": tau_f,
        "skip_first_bars": cfg["skip_first_bars"],
        "target_coverage": cfg["target_coverage"],
        "vol_cutoffs": vol_cutoffs,
    }
    with open(os.path.join(od, "thresholds.json"), "w") as fh:
        json.dump(thr, fh, indent=2)

    # 5. Config snapshot
    config_snap = {
        "asset": "AAPL",
        "timeframe_minutes": 5,
        "feed": "iex",
        "horizon_bars": h,
        "kdz": kdz,
        "train_end": cfg["train_end"],
        "val_start": cfg["val_start"],
        "val_end": cfg["val_end"],
        "holdout_start": cfg["holdout_start"],
        "primary_params": pr_params,
        "meta_params": meta_params,
        "quantile_params": q_params,
        "skip_first_bars": cfg["skip_first_bars"],
        "target_coverage": cfg["target_coverage"],
    }
    with open(os.path.join(od, "config_snapshot.yaml"), "w") as fh:
        yaml.dump(config_snap, fh, default_flow_style=False)

    # 6. Bundle info with SHA-256 hashes
    skip = ("bundle_info.json", "metrics.json")
    files = sorted(f for f in os.listdir(od) if f not in skip and not f.endswith(".tmp"))
    info = {
        "training_end_date": str(df["day"].max()),
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sha256": {f: sha256_file(os.path.join(od, f)) for f in files},
    }
    with open(os.path.join(od, "bundle_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print("bundle saved to", od, "tau_m", tau_f)


def run_final(force=False, refit_only=False):
    with open(cfg["best_path"]) as fh:
        best = json.load(fh)
    h, kdz = best["horizon_bars"], best["kdz"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    df = build_frame(h, kdz)
    if not refit_only:
        if not run_holdout(df, h, kdz, best, force):
            return
    run_refit(df, h, kdz)


def run_check():
    df = build_frame(6, 0.15)
    tr = df[df["day"] <= cfg["train_end"]].copy()
    tr["p_up"] = oof_primary(tr, shuffle=True)
    add_side(tr)
    ev = tr[tr["ok"].notna()]
    print("shuffled-label oof accuracy %.4f (should be about 0.50)" % ev["ok"].mean())
    print("auc %.4f" % roc_auc_score(ev["fwd"] > 0, ev["p_up"]))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "val"
    if stage == "check":
        run_check()
    elif stage == "val":
        run_val()
    elif stage == "final":
        run_final(force=("force" in sys.argv))
    elif stage == "refit":
        run_final(refit_only=True)
    else:
        print("use: python train.py check | val | final | refit")
