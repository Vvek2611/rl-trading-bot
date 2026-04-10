import yfinance as yf
import pandas as pd
import os

def get_stock_data(ticker="AAPL", period="3y", interval="1d", save=True):
    print(f"Fetching {ticker} data...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    # Fix multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only numeric columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    if save:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/{ticker}.csv", index=False)
        print(f"Saved {len(df)} rows")

    return df

def load_stock_data(ticker="AAPL"):
    path = f"data/raw/{ticker}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    return get_stock_data(ticker)