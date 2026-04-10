import yfinance as yf
import os

def get_stock_data(ticker="AAPL", period="3y", interval="1d", save=True):
    print(f"Fetching {ticker} data...")
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    if save:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/{ticker}.csv", index=False)
    return df

def load_stock_data(ticker="AAPL"):
    path = f"data/raw/{ticker}.csv"
    if os.path.exists(path):
        import pandas as pd
        return pd.read_csv(path)
    return get_stock_data(ticker)