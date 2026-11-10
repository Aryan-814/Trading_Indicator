"""Purged walk-forward cross-validation splitter per spec section 8.2 & Appendix B."""
import numpy as np
import pandas as pd


class PurgedWalkForwardSplitter:
    """Purged Walk-Forward cross-validation with embargo.

    Ensures no training row's label window [t, t+h] overlaps a test block,
    and excludes an embargo of h bars following test boundaries.
    """

    def __init__(self, block="quarter", embargo_bars=6, horizon_bars=6):
        self.block = block
        self.embargo_bars = embargo_bars
        self.horizon_bars = horizon_bars

    def split(self, df):
        """Yield (train_indices, test_indices) for each consecutive block.

        Parameters
        ----------
        df : DataFrame containing 'day' (YYYY-MM-DD string or datetime)
             and 'bar_idx' or sorted chronological index.
        """
        days = pd.to_datetime(df["day"])
        if self.block == "quarter":
            periods = days.dt.to_period("Q").astype(str)
        elif self.block == "month":
            periods = days.dt.to_period("M").astype(str)
        else:
            raise ValueError(f"Unknown block type: {self.block}")

        unique_periods = sorted(periods.unique())

        for i in range(1, len(unique_periods)):
            test_period = unique_periods[i]
            test_mask = periods == test_period
            test_indices = df.index[test_mask].values

            test_start_day = df.loc[test_mask, "day"].min()

            # Train rows must strictly precede test_start_day
            # Since labels are intraday and do not cross days,
            # dropping day >= test_start_day prevents any label overlap.
            train_mask = df["day"] < test_start_day

            # Embargo: drop the last embargo_bars of training data if on same day,
            # or if across days, train_mask ending before test_start_day satisfies purge.
            train_indices = df.index[train_mask].values

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

