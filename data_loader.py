# data_loader.py
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from joblib import Parallel, delayed


def download_intraday(symbol: str = "SPY",
                      period: str = "60d",
                      interval: str = "5m") -> pd.DataFrame:
    """
    ORIGINAL SINGLE-SYMBOL FUNCTION (unchanged).
    Download intraday OHLCV data with yfinance and normalize columns.
    """
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    df.dropna(how="all", inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=1)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype(float)
    df.dropna(inplace=True)
    return df


# === NEW MULTI-ASSET LOADER ===================================================

def _download_symbol(symbol: str,
                     period: str,
                     interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        df.dropna(how="all", inplace=True)
        if df.empty:
            print(f"[{symbol}] No data returned.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1, level=1)
            except Exception:
                df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.astype(float)
        df.dropna(inplace=True)
        df["symbol"] = symbol
        return df
    except Exception as e:
        print(f"[{symbol}] Download failed: {e}")
        return None


def download_multi(
    symbols: List[str],
    period: str = "10y",
    interval: str = "1d",
    n_jobs: int = -1
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for many symbols concurrently.
    Returns dict: symbol -> df
    """
    print(f"Downloading data for {len(symbols)} symbols...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_download_symbol)(sym, period, interval) for sym in symbols
    )
    data_by_symbol = {}
    for sym, df in zip(symbols, results):
        if df is not None and len(df) > 0:
            data_by_symbol[sym] = df
    print(f"Downloaded {len(data_by_symbol)} / {len(symbols)} symbols successfully.")
    return data_by_symbol


def to_panel(data_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine dict of symbol->df into a single MultiIndex DataFrame:
    index = DatetimeIndex
    columns = MultiIndex (field, symbol)
    """
    if not data_by_symbol:
        raise ValueError("No data provided to to_panel().")

    aligned = []
    for sym, df in data_by_symbol.items():
        temp = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        temp.columns = pd.MultiIndex.from_product(
            [temp.columns, [sym]], names=["field", "symbol"]
        )
        aligned.append(temp)

    panel = pd.concat(aligned, axis=1).sort_index()
    return panel


if __name__ == "__main__":
    # Quick sanity check for the new multi-symbol loader
    syms = ["GOOGL", "AAPL", "MSFT"]
    data = download_multi(syms, period="1y", interval="1d")
    panel = to_panel(data)
    print(panel.head())

