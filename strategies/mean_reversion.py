import pandas as pd
from strategies.base_strategy import BaseStrategy

class MeanReversion(BaseStrategy):

    def __init__(self, lookback=20, entry_z=2.0):
        self.lookback = lookback
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame):

        df = data.copy()
        signals = pd.Series(0, index=df.index)

        ma = df["close"].rolling(self.lookback).mean()
        std = df["close"].rolling(self.lookback).std()

        z = (df["close"] - ma) / std

        for i in range(self.lookback, len(df)):

            if z.iloc[i] > self.entry_z:
                signals.iloc[i] = -1

            elif z.iloc[i] < -self.entry_z:
                signals.iloc[i] = 1

        return signals
