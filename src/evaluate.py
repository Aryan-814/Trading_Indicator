"""Evaluation metrics, baselines, and statistical tests per spec section 8.5."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


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


def block_permutation_test(actual_signs, predicted_sides, selected_mask, h=6, n_perm=2000, seed=42):
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


def deflated_sharpe_ratio(observed_sr, n_trials, var_sr=1.0, skew=0.0, kurt=3.0, sample_len=1000):
    """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""
    from scipy.stats import norm
    euler_mascheroni = 0.5772156649
    # Expected maximum Sharpe under the null of no skill across n_trials
    if n_trials <= 1:
        max_sr_expected = 0.0
    else:
        z = np.sqrt(2 * np.log(n_trials))
        max_sr_expected = np.sqrt(var_sr) * (z + euler_mascheroni / z)

    # Standard error of Sharpe ratio
    sr_std = np.sqrt((1 - skew * observed_sr + ((kurt - 1) / 4) * (observed_sr ** 2)) / sample_len)
    if sr_std <= 0:
        return 0.5
    dsr_stat = (observed_sr - max_sr_expected) / sr_std
    return float(norm.cdf(dsr_stat))

