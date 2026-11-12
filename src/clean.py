"""Data cleaning rules per spec section 5.3."""
import os
import yaml
import numpy as np
import pandas as pd
from datetime import time as dtime


def _canonical_out_path():
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            y = yaml.safe_load(fh) or {}
        if "data_path" in y:
            return y["data_path"]
    return "data/clean/aapl_5m_clean.parquet"


def clean_bars(raw_path_or_df):
    """Clean raw 5-minute stock bars according to section 5.3."""
    if isinstance(raw_path_or_df, str):
        if raw_path_or_df.endswith(".csv"):
            df = pd.read_csv(raw_path_or_df)
        else:
            df = pd.read_parquet(raw_path_or_df)
    else:
        df = raw_path_or_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    drop_cols = [c for c in ["symbol", "ts_ny"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["ts"] = df["timestamp"].dt.tz_convert("America/New_York")
    df["day"] = df["ts"].dt.date.astype(str)

    # 3. Regular market hours only (09:30 to 15:55 ET)
    t = df["ts"].dt.time
    in_hours = (t >= dtime(9, 30)) & (t <= dtime(15, 55))
    df = df[in_hours].copy()

    # 4. Keep full days (78 bars) and half days (42 bars)
    df["day_bars"] = df.groupby("day")["close"].transform("size")
    df = df[df["day_bars"].isin([78, 42])].copy()
    df["is_half"] = (df["day_bars"] == 42).astype(int)

    # 5. Position in day and end timestamps
    df["bar_idx"] = df.groupby("day").cumcount()
    df["bars_left"] = df["day_bars"] - df["bar_idx"] - 1
    df["bar_end"] = df["timestamp"] + pd.Timedelta(minutes=5)

    # 6. Gap and ret_1 (ret_1 on first bar uses close/open; gap handles overnight move)
    first = df["bar_idx"] == 0
    prev_close = df["close"].shift(1)
    df["gap"] = np.where(first, np.log(df["open"] / prev_close), 0.0)
    df["gap"] = df["gap"].fillna(0.0)

    df["ret_1"] = np.where(
        first,
        np.log(df["close"] / df["open"]),
        np.log(df["close"] / prev_close),
    )

    # 7. Relative volume and trade count vs past 20-day median (min_periods=10) at same bar_idx
    def past_median(s):
        return s.shift(1).rolling(20, min_periods=10).median()

    for col in ["volume", "trade_count"]:
        if col in df.columns:
            ref = df.groupby("bar_idx")[col].transform(past_median)
            df["rel_" + col] = df[col] / ref

    # 8. Assertions
    price_cols = ["open", "high", "low", "close"]
    assert df[price_cols + ["volume"]].notna().all().all(), "NaN in OHLCV"
    assert (df[price_cols] > 0).all().all(), "Non-positive price"
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-6).all(), "High < max(open, close)"
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-6).all(), "Low > min(open, close)"

    # Drop warm-up rows (first 10 trading days before 2020-08-17)
    req_cols = [c for c in ["rel_volume", "rel_trade_count"] if c in df.columns]
    df = df.dropna(subset=req_cols).reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw_path = "data/raw/aapl_5m_raw.parquet"
    if not os.path.exists(raw_path):
        for candidate in ["aapl_5m_raw.parquet", "aapl_5m_raw.csv"]:
            if os.path.exists(candidate):
                raw_path = candidate
                break

    if os.path.exists(raw_path):
        cleaned = clean_bars(raw_path)
        print("Cleaned shape:", cleaned.shape)
        out_path = _canonical_out_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if out_path.endswith(".csv"):
            cleaned.to_csv(out_path, index=False)
        else:
            cleaned.to_parquet(out_path, index=False)
        print("Saved to", out_path)
    else:
        print(f"Raw file {raw_path} not found.")
