import pandas as pd
import numpy as np

class FeatureEngine:

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # returns
        df["returns"] = df["close"].pct_change()

        # EMA
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        # ATR (true range)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        df["atr"] = tr.rolling(14).mean()

        # volatility (IMPORTANT for regime + allocator)
        df["volatility"] = df["returns"].rolling(20).std()

        return df.dropna()