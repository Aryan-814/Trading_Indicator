"""Confidence calibration helpers per spec section 9.6.

Two complementary approaches:
  1. ConfidenceCalibrator — offline isotonic regression fit at training time.
  2. AdaptiveConformalInference — online conformal wrapper for live use (optional).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ConfidenceCalibrator:
    """Thin wrapper around sklearn IsotonicRegression for meta-probability calibration.

    Fits a monotone mapping  raw_meta_prob → P(correct side | signal).
    Usage mirrors sklearn API: fit(X, y) then predict(X).
    """

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, meta_probs, correct_flags):
        """Fit the isotonic calibrator.

        Parameters
        ----------
        meta_probs    : array-like, shape (n,) — raw meta-label probabilities
        correct_flags : array-like, shape (n,) — 1 if primary was correct, 0 otherwise
        """
        self._iso.fit(np.asarray(meta_probs), np.asarray(correct_flags, dtype=float))
        self.fitted = True
        return self

    def predict(self, meta_probs):
        """Return calibrated confidence scores in [0, 1]."""
        return self._iso.predict(np.asarray(meta_probs))

    def predict_one(self, meta_prob):
        """Convenience wrapper for a single scalar."""
        return float(self.predict([meta_prob])[0])


class AdaptiveConformalInference:
    """Online conformal prediction set size tracker (optional live use).

    Maintains an exponentially decaying quantile of non-conformity scores
    so that the confidence threshold adapts to recent calibration drift.

    Parameters
    ----------
    alpha   : target miscoverage rate (default 0.20 → 80% coverage)
    gamma   : forgetting factor per bar (default 0.005 ≈ 1/200 bars)
    """

    def __init__(self, alpha=0.20, gamma=0.005):
        self.alpha = alpha
        self.gamma = gamma
        self._scores: list = []
        self._weights: list = []

    def observe(self, score):
        """Record a new non-conformity score (e.g. 1 - calibrated_conf when wrong).

        Parameters
        ----------
        score : float — non-conformity score for the most recent completed trade
        """
        n = len(self._scores)
        # Decay existing weights
        self._weights = [w * (1 - self.gamma) for w in self._weights]
        self._scores.append(score)
        self._weights.append(1.0)

    def threshold(self):
        """Return the current adaptive tau (minimum meta_prob to signal).

        Returns nan if fewer than 10 observations have been recorded.
        """
        if len(self._scores) < 10:
            return float("nan")
        s = np.asarray(self._scores)
        w = np.asarray(self._weights)
        w = w / w.sum()
        # Weighted quantile
        sorted_idx = np.argsort(s)
        cum_w = np.cumsum(w[sorted_idx])
        q_idx = np.searchsorted(cum_w, 1.0 - self.alpha)
        return float(s[sorted_idx[min(q_idx, len(s) - 1)]])

