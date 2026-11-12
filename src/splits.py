"""Walk-forward block splitting per spec section 8.2.

Labels never cross a day boundary, so dropping all training days on or
after the first day of the test quarter is a complete purge — no extra
embargo is needed.
"""
import pandas as pd


def walk_forward_blocks(df, min_q=4):
    """Yield (train_idx, test_idx) for each quarter index >= min_q.

    Parameters
    ----------
    df : DataFrame with 'day' (YYYY-MM-DD string) and 'q' (quarter string).
    min_q : int — minimum number of prior quarters before first test block.

    Yields
    ------
    train_idx, test_idx : numpy arrays of integer positions into df.index
    """
    qs = sorted(df["q"].unique())
    for i in range(min_q, len(qs)):
        test_mask = df["q"] == qs[i]
        test_start_day = df.loc[test_mask, "day"].min()
        train_mask = df["day"] < test_start_day
        train_idx = df.index[train_mask].values
        test_idx = df.index[test_mask].values
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx
