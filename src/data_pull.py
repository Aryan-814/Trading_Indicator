"""Historical 5-minute data extraction per spec section 5.2."""
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment


def pull_symbol_bars(client, symbol, start="2020-08-03", end=None, out_dir="data/raw"):
    """Download 5-minute IEX bars with split/dividend adjustment."""
    os.makedirs(out_dir, exist_ok=True)
    if end is None:
        end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    print(f"Fetching {symbol} from {start} to {end}...")
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to regular hours
    df["ts_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df[df["ts_ny"].dt.time.between(
        pd.Timestamp("09:30").time(),
        pd.Timestamp("15:55").time(),
    )]
    df = df.drop(columns=[c for c in ["symbol", "ts_ny"] if c in df.columns]).reset_index(drop=True)

    out_file = os.path.join(out_dir, f"{symbol.lower()}_5m_raw.parquet")
    df.to_parquet(out_file, index=False)
    print(f"Saved {symbol}: {df.shape[0]} rows to {out_file}")
    return df


def pull_all_symbols(symbols=("AAPL",), start="2020-08-03", end=None, out_dir="data/raw"):
    load_dotenv()
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

    client = StockHistoricalDataClient(key, secret)
    results = {}
    for sym in symbols:
        results[sym] = pull_symbol_bars(client, sym, start=start, end=end, out_dir=out_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull historical 5-minute bars from Alpaca")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"], help="Symbols to pull")
    parser.add_argument("--start", default="2020-08-03", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory")
    args = parser.parse_args()

    pull_all_symbols(symbols=args.symbols, start=args.start, end=args.end, out_dir=args.out_dir)

