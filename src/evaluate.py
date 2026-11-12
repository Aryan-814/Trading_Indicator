"""Evaluation metrics, baselines, and statistical tests per spec section 8.5."""
import numpy as np
import pandas as pd


def wilson_score_interval(p, n, z=1.96):
    """Wilson score confidence interval for binomial proportion."""
    if n <= 0 or np.isnan(p):
        return [float("nan"), float("nan")]
    den = 1.0 + (z * z) / n
    center = p + (z * z) / (2.0 * n)
    margin = z * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return [float((center - margin) / den), float((center + margin) / den)]


def expected_calibration_error(conf, ok, n_bins=10):
    """Compute ECE and 10-bin reliability table."""
    bins = pd.qcut(conf.rank(method="first"), n_bins, labels=False, duplicates="drop")
    tab = pd.DataFrame({"conf": conf.values, "ok": ok.values, "b": bins})
    g = tab.groupby("b").agg(
        conf=("conf", "mean"), ok=("ok", "mean"), n=("ok", "size")
    )
    ece = (g["n"] * (g["conf"] - g["ok"]).abs()).sum() / g["n"].sum()
    return float(ece), g.round(4).reset_index().to_dict("records")


def block_permutation_test(actual_signs, predicted_sides, selected_mask, h, n_perm=2000, seed=42):
    """Block permutation test preserving return autocorrelation of length h."""
    rng = np.random.default_rng(seed)
    sgn = np.asarray(actual_signs, dtype=float)
    side = np.asarray(predicted_sides, dtype=float)
    sel = np.asarray(selected_mask, dtype=bool)

    nb = len(sgn) // h
    cut = nb * h
    if cut == 0 or sel[:cut].sum() == 0:
        return float("nan"), 1.0

    blocks = sgn[:cut].reshape(nb, h)
    side_c = side[:cut]
    sel_c = sel[:cut]
    obs_acc = float((sgn[:cut][sel_c] == side_c[sel_c]).mean())

    exceed_count = 0
    for _ in range(n_perm):
        perm_blocks = blocks[rng.permutation(nb)].reshape(-1)
        perm_acc = (perm_blocks[sel_c] == side_c[sel_c]).mean()
        if perm_acc >= obs_acc:
            exceed_count += 1

    p_value = (exceed_count + 1) / (n_perm + 1)
    return obs_acc, float(p_value)
