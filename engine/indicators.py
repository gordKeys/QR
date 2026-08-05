import pandas as pd


def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()


def atr(data: pd.DataFrame, period: int = 14):
    high = data["high"]
    low = data["low"]
    close = data["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()