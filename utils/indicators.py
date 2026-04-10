from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def add_indicators(df):
    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    df['EMA_20']      = EMAIndicator(close, window=20).ema_indicator()
    df['EMA_50']      = EMAIndicator(close, window=50).ema_indicator()
    df['RSI']         = RSIIndicator(close, window=14).rsi()
    macd              = MACD(close)
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist']   = macd.macd_diff()
    bb                = BollingerBands(close)
    df['BB_Width']    = bb.bollinger_wband()
    df['ATR']         = AverageTrueRange(high, low, close).average_true_range()
    df['Volume_MA']   = volume.rolling(20).mean()
    df['Volume_Ratio']= volume / df['Volume_MA']

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df