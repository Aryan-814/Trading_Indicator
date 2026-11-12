import sys
import os

sys.path.insert(0, os.path.abspath("src"))

import hashlib
import json
import platform
import subprocess
import time
import uuid

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (
    block_permutation_test,
    expected_calibration_error,
    wilson_score_interval,
)
from features import feat_cols, load_bars, make_features, meta_cols
from labels import make_labels
from splits import walk_forward_blocks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config_file = "config.yaml"
if not os.path.exists(config_file) and os.path.exists(os.path.join("..", config_file)):
    config_file = os.path.join("..", config_file)

yaml_cfg = {}
if os.path.exists(config_file):
    with open(config_file) as fh:
        yaml_cfg = yaml.safe_load(fh) or {}

cfg = {
    "data_path": yaml_cfg.get("data_path", "data/clean/aapl_5m_clean.parquet"),
    "out_dir": "models/v1",
    "log_path": "reports/experiments.csv",
    "best_path": "reports/best_cfg.json",
    "holdout_log_path": "reports/holdout_log.json",
    "default_h": yaml_cfg.get("horizon_bars", 12),
    "default_kdz": yaml_cfg.get("label", {}).get("kdz", 0.30),
    "vol_span": yaml_cfg.get("label", {}).get("vol_span", 78),
    "min_q": yaml_cfg.get("cv", {}).get("min_q", 4),
    "train_end": yaml_cfg.get("periods", {}).get("train_end", "2024-12-31"),
    "val_start": yaml_cfg.get("periods", {}).get("val_start", "2025-01-01"),
    "val_end": yaml_cfg.get("periods", {}).get("val_end", "2025-12-31"),
    "holdout_start": yaml_cfg.get("periods", {}).get("holdout_start", "2026-01-01"),
    "skip_first_bars": yaml_cfg.get("policy", {}).get("skip_first_bars", 3),
    "target_coverage": yaml_cfg.get("policy", {}).get("target_coverage", 0.20),
    "ece_max": 0.03,
    "cost_bps": yaml_cfg.get("cost_bps", [0.5, 1.0, 3.0]),
    "n_perm": 2000,
    "seed": yaml_cfg.get("primary", {}).get("seed", 42),
}

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
    max_depth=yaml_cfg.get("meta", {}).get("max_depth", 3),
    learning_rate=yaml_cfg.get("meta", {}).get("learning_rate", 0.03),
    n_estimators=yaml_cfg.get("meta", {}).get("n_estimators", 100),
    min_child_samples=yaml_cfg.get("meta", {}).get("min_child_samples", 500),
    reg_lambda=yaml_cfg.get("meta", {}).get("reg_lambda", 10.0),
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

# Horizon x dead-zone sweep grid (locked; do not widen after holdout use)
grid = [(h, kdz) for h in (2, 6, 12) for kdz in (0.15, 0.3)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data / frames
# ---------------------------------------------------------------------------

def build_frame(h, kdz):
    d = load_bars(cfg["data_path"])
    f = make_features(d)
    lab = make_labels(d, h=h, kdz=kdz, vol_span=cfg["vol_span"])
    base = d[["day", "open", "close", "bar_idx"]].copy()
    base["next_open"] = d.groupby("day")["open"].shift(-1)
    df = pd.concat([base, f, lab], axis=1)
    df["q"] = pd.to_datetime(df["day"]).dt.to_period("Q").astype(str)
    df = df[df["vol_78"].notna()].reset_index(drop=True)
    return df


def elig(df):
    return df["fwd"].notna() & (df["bar_idx"] >= cfg["skip_first_bars"])


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_primary(train, shuffle=False, rng=None):
    rows = train[train["y"].isin([1, -1])]
    target = (rows["y"] == 1).astype(int).values
    if shuffle:
        target = rng.permutation(target)
    m = lgb.LGBMClassifier(**pr_params)
    m.fit(rows[feat_cols], target)
    return m


def fit_b4_logistic(train):
    """Fit B4 L2-regularized logistic regression baseline on standardized feat_cols."""
    rows = train[train["y"].isin([1, -1])]
    target = (rows["y"] == 1).astype(int).values
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=cfg["seed"])),
    ])
    pipe.fit(rows[feat_cols], target)
    return pipe


def oof_primary(df, min_q=None, shuffle=False):
    """Purged walk-forward primary predictions."""
    if min_q is None:
        min_q = cfg["min_q"]
    rng = np.random.default_rng(cfg["seed"])
    p = pd.Series(np.nan, index=df.index)
    for train_idx, test_idx in walk_forward_blocks(df, min_q=min_q):
        train = df.loc[train_idx]
        test_block = df.loc[test_idx]
        test = test_block[elig(test_block)]
        if len(train) == 0 or len(test) == 0:
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


def oof_meta(df, min_q=2):
    """Purged walk-forward meta predictions."""
    rows = df[df["p_up"].notna() & df["ok"].notna()]
    m_series = pd.Series(np.nan, index=df.index)
    for train_idx, test_idx in walk_forward_blocks(rows, min_q=min_q):
        train = rows.loc[train_idx]
        test = rows.loc[test_idx]
        if len(train) == 0 or len(test) == 0:
            continue
        mm = lgb.LGBMClassifier(**meta_params)
        mm.fit(train[meta_cols], train["ok"].astype(int))
        m_series[test.index] = mm.predict_proba(test[meta_cols])[:, 1]
    return m_series


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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(sc, h, tau_m, b4_model=None):
    ev = sc[sc["ok"].notna()].sort_index().copy()
    res = {"n_rows": int(len(ev))}
    res["auc_primary"] = float(roc_auc_score(ev["fwd"] > 0, ev["p_up"]))
    res["acc_primary"] = float(ev["ok"].mean())
    res["brier_conf"] = float(brier_score_loss(ev["ok"], ev["conf"]))
    const = np.full(len(ev), ev["ok"].mean())
    res["brier_const"] = float(brier_score_loss(ev["ok"], const))
    res["ece"], res["reliability"] = expected_calibration_error(ev["conf"], ev["ok"])

    sel_tab = {}
    for cv in (0.05, 0.10, 0.15, 0.20, 0.25, 0.50):
        cut = ev["m"].quantile(1 - cv)
        s = ev[ev["m"] >= cut]
        acc = float(s["ok"].mean())
        sel_tab[str(cv)] = {
            "n": int(len(s)),
            "acc": acc,
            "ci95_neff": wilson_score_interval(acc, len(s) / h),
        }
    res["selective"] = sel_tab

    up = ev["fwd"] > 0
    res["b1_always_up"] = float(up.mean())
    res["b3_momentum_ret6"] = float(
        (np.sign(ev["ret_6"]) == np.sign(ev["fwd"])).mean()
    )
    rsi_sig = ev[(ev["rsi_14"] < 30) | (ev["rsi_14"] > 70)]
    rsi_pred = np.where(rsi_sig["rsi_14"] < 30, 1, -1)
    res["b2_rsi_rule"] = {
        "coverage": float(len(rsi_sig) / len(ev)),
        "acc": float((rsi_pred == np.sign(rsi_sig["fwd"])).mean())
        if len(rsi_sig) > 0
        else float("nan"),
    }

    # B4: logistic regression on standardized feat_cols, top 20% by |p_up - 0.5|
    if b4_model is not None:
        p_b4 = b4_model.predict_proba(ev[feat_cols])[:, 1]
        side_b4 = np.where(p_b4 >= 0.5, 1, -1)
        dist_b4 = np.abs(p_b4 - 0.5)
        cut_b4 = np.quantile(dist_b4, 1 - cfg["target_coverage"])
        sel_b4 = dist_b4 >= cut_b4
        acc_b4 = float((np.sign(ev["fwd"].values[sel_b4]) == side_b4[sel_b4]).mean())
        res["b4_logistic"] = {
            "coverage": float(sel_b4.mean()),
            "acc": acc_b4,
            "n": int(sel_b4.sum()),
            "auc": float(roc_auc_score(ev["fwd"] > 0, p_b4)),
        }

    # B5: primary-only top-20% by |p_up - 0.5| (shows whether meta-model adds value)
    top20_cut = ev["p_dist"].quantile(1 - cfg["target_coverage"])
    b5_sel = ev[ev["p_dist"] >= top20_cut]
    res["b5_primary_direct"] = {
        "coverage": float(len(b5_sel) / len(ev)),
        "acc": float(b5_sel["ok"].mean()) if len(b5_sel) > 0 else float("nan"),
        "n": int(len(b5_sel)),
    }

    sel = (ev["m"] >= tau_m).values
    s = ev[sel]
    acc = float(s["ok"].mean()) if len(s) > 0 else float("nan")
    res["policy"] = {
        "tau_m": float(tau_m),
        "coverage": float(sel.mean()),
        "n_signals": int(sel.sum()),
        "acc": acc,
        "ci95_neff": wilson_score_interval(acc, sel.sum() / h),
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

    # Block permutation test (blocks of size h; actual signs in {-1, +1})
    obs, pval = block_permutation_test(
        np.sign(ev["fwd"].values),
        ev["side"].values,
        sel,
        h=h,
        n_perm=cfg["n_perm"],
        seed=cfg["seed"],
    )
    res["policy"]["perm_acc"] = obs
    res["policy"]["perm_pvalue"] = float(pval)
    return res


# ---------------------------------------------------------------------------
# Trial logging
# ---------------------------------------------------------------------------

def log_trial(row):
    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(cfg["log_path"]):
        old = pd.read_csv(cfg["log_path"])
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(cfg["log_path"], index=False)


def _total_trial_count():
    if os.path.exists(cfg["log_path"]):
        return len(pd.read_csv(cfg["log_path"]))
    return 0


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_val():
    run_id = "lgb_" + str(uuid.uuid4())[:6]
    rows = []
    for h, kdz in grid:
        t0 = time.time()
        df = build_frame(h, kdz)
        primary, meta, cal, tr = build_stack_on_train(df)
        b4 = fit_b4_logistic(tr)
        va = df[(df["day"] >= cfg["val_start"]) & (df["day"] <= cfg["val_end"])]
        sc = score_rows(va, primary, meta, cal)
        tau_m = float(sc.loc[sc["ok"].notna(), "m"].quantile(1 - cfg["target_coverage"]))
        res = evaluate(sc, h, tau_m, b4_model=b4)
        perm_p = res["policy"]["perm_pvalue"]
        row = {
            "run_id": run_id,
            "trial_id": _total_trial_count() + 1,
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
            "perm_p": perm_p,
            "notes": "b1=%.4f" % res["b1_always_up"],
        }
        log_trial(row)
        rows.append(row)
        print(
            "trial", row["trial_id"], "h", h, "kdz", kdz,
            "sel20 %.4f" % row["selective_acc_20"],
            "ece %.4f" % row["ece"],
            "auc %.4f" % row["auc"],
            "perm_p %.4f" % perm_p,
            "b1 %.4f" % res["b1_always_up"],
            "secs %d" % (time.time() - t0),
            flush=True,
        )

    ok_rows = [r for r in rows if r["ece"] <= cfg["ece_max"]]
    pool = ok_rows if ok_rows else rows
    best = max(pool, key=lambda r: r["selective_acc_20"])
    n_total = _total_trial_count()
    out = {
        "horizon_bars": best["horizon_bars"],
        "kdz": best["kdz"],
        "trial_id": best["trial_id"],
        "run_id": best["run_id"],
        "n_trials": n_total,
    }
    os.makedirs(os.path.dirname(cfg["best_path"]), exist_ok=True)
    with open(cfg["best_path"], "w") as fh:
        json.dump(out, fh, indent=2)
    print("best", out)


def run_holdout(df, h, kdz, best, force):
    log_path = cfg["holdout_log_path"]
    model_name = "lightgbm"

    if os.path.exists(log_path):
        with open(log_path) as fh:
            ho_log = json.load(fh)
    else:
        ho_log = []

    already_run = any(e.get("model") == model_name for e in ho_log)
    if already_run and not force:
        print(
            f"holdout already evaluated for model '{model_name}'. "
            "Use 'force' to re-run (counts as a second evaluation)."
        )
        return False

    primary, meta, cal, tr = build_stack_on_train(df)
    b4 = fit_b4_logistic(tr)
    va = df[(df["day"] >= cfg["val_start"]) & (df["day"] <= cfg["val_end"])]
    sc_va = score_rows(va, primary, meta, cal)
    tau_m = float(sc_va.loc[sc_va["ok"].notna(), "m"].quantile(1 - cfg["target_coverage"]))
    ho = df[df["day"] >= cfg["holdout_start"]]
    sc_ho = score_rows(ho, primary, meta, cal)
    qm = fit_quant(tr)

    n_total = best.get("n_trials", _total_trial_count())
    metrics = {
        "note": (
            "Validation and holdout scores come from the model trained through "
            "train_end (before the final refit). The refit model has no "
            "out-of-sample score. Holdout evaluated twice (scikit-learn, then "
            "LightGBM) with no tuning in between."
        ),
        "horizon_bars": h,
        "kdz": kdz,
        "n_trials": n_total,
        "trials_total": n_total,
        "validation": evaluate(sc_va, h, tau_m, b4_model=b4),
        "holdout": evaluate(sc_ho, h, tau_m, b4_model=b4),
        "band_validation": band_cov(qm, sc_va),
        "band_holdout": band_cov(qm, sc_ho),
        "holdout_evaluations": [
            {
                "model": "hist_gb",
                "date": "2026-11-09",
                "note": "Evaluation 1 — scikit-learn HistGradientBoostingClassifier",
            },
            {
                "model": "lightgbm",
                "date": time.strftime("%Y-%m-%d"),
                "note": "Evaluation 2 — LightGBM",
            },
        ],
    }

    mpath = os.path.join(cfg["out_dir"], "metrics.json")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)

    entry = {
        "model": model_name,
        "date": time.strftime("%Y-%m-%d"),
        "git_commit": _get_git_commit(),
        "horizon_bars": h,
        "kdz": kdz,
        "note": "Holdout evaluation 2 — LightGBM run.",
    }
    ho_log.append(entry)
    with open(log_path, "w") as fh:
        json.dump(ho_log, fh, indent=2)

    print("holdout written")
    return True


def run_refit(df, h, kdz):
    al = df.copy()
    al["p_up"] = oof_primary(al)
    add_side(al)
    meta_f, cal_f = fit_stack(al)
    primary_f = fit_primary(al)
    qm_f = fit_quant(al)

    # tau_m from the final meta-model's own scores on the validation year
    vmask = (al["day"] >= cfg["val_start"]) & (al["day"] <= cfg["val_end"])
    vrows = al[vmask & al["ok"].notna()]
    m_val = meta_f.predict_proba(vrows[meta_cols])[:, 1]
    tau_f = float(np.quantile(m_val, 1 - cfg["target_coverage"]))

    od = cfg["out_dir"]
    os.makedirs(od, exist_ok=True)

    # 1. Models in LightGBM native format only (.txt) + calibrator.joblib
    primary_f.booster_.save_model(os.path.join(od, "primary.txt"))
    meta_f.booster_.save_model(os.path.join(od, "meta.txt"))
    joblib.dump(cal_f, os.path.join(od, "calibrator.joblib"))
    for a in (0.1, 0.5, 0.9):
        qm_f[a].booster_.save_model(os.path.join(od, "quantile_%d.txt" % int(a * 100)))

    # 2. Feature list
    with open(os.path.join(od, "feature_list.json"), "w") as fh:
        json.dump({"feat_cols": feat_cols, "meta_cols": meta_cols}, fh, indent=2)

    # 3. Thresholds
    thr = {
        "horizon_bars": h,
        "kdz": kdz,
        "tau_m": tau_f,
        "skip_first_bars": cfg["skip_first_bars"],
        "target_coverage": cfg["target_coverage"],
    }
    with open(os.path.join(od, "thresholds.json"), "w") as fh:
        json.dump(thr, fh, indent=2)

    # 4. Config snapshot
    config_snap = {
        "asset": "AAPL",
        "timeframe_minutes": 5,
        "feed": "iex",
        "data_path": cfg["data_path"],
        "horizon_bars": h,
        "kdz": kdz,
        "vol_span": cfg["vol_span"],
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

    # 5. Remove stale .joblib model copies and analog_table if present
    for stale in [
        "primary.joblib", "meta.joblib", "analog_table.parquet",
        "quantile_10.joblib", "quantile_50.joblib", "quantile_90.joblib",
    ]:
        stale_path = os.path.join(od, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    # 6. Bundle info with SHA-256 hashes and environment versions
    skip = {"bundle_info.json", "metrics.json"}
    files = sorted(f for f in os.listdir(od) if f not in skip and not f.endswith(".tmp"))
    info = {
        "training_end_date": str(df["day"].max()),
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "git_commit": _get_git_commit(),
        "data_path": cfg["data_path"],
        "data_sha256": sha256_file(cfg["data_path"]),
        "sha256": {f: sha256_file(os.path.join(od, f)) for f in files},
    }
    with open(os.path.join(od, "bundle_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print("bundle saved to", od, "tau_m", tau_f)


def run_final(force=False, refit_only=False):
    with open(cfg["best_path"]) as fh:
        best = json.load(fh)
    h, kdz = best["horizon_bars"], best["kdz"]
    df = build_frame(h, kdz)
    if not refit_only:
        if not run_holdout(df, h, kdz, best, force):
            return
    run_refit(df, h, kdz)


def run_check():
    """Shuffled-label sanity test: accuracy should be near 50%."""
    if os.path.exists(cfg["best_path"]):
        with open(cfg["best_path"]) as fh:
            best = json.load(fh)
        h, kdz = best["horizon_bars"], best["kdz"]
    else:
        h, kdz = cfg["default_h"], cfg["default_kdz"]
    df = build_frame(h, kdz)
    tr = df[df["day"] <= cfg["train_end"]].copy()
    tr["p_up"] = oof_primary(tr, shuffle=True)
    add_side(tr)
    ev = tr[tr["ok"].notna()]
    print("h=%d kdz=%.2f shuffled-label oof accuracy %.4f (expect ~0.50)" % (h, kdz, ev["ok"].mean()))
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
        print("use: python train.py check | val | final [force] | refit")
