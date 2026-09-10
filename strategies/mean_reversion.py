import pandas as pd
from strategies.base_strategy import BaseStrategy

class MeanReversion(BaseStrategy):

    def __init__(self, lookback=20, entry_z=2.0):
        self.lookback = lookback
        self.entry_z = entry_z

    @staticmethod
    def _bullish_reversal(current, previous):
        return current["close"] > current["open"] and current["close"] > previous["close"]

    @staticmethod
    def _bearish_reversal(current, previous):
        return current["close"] < current["open"] and current["close"] < previous["close"]

    def generate_signals(self, data: pd.DataFrame):

        df = data.copy()
        signals = pd.Series(0, index=df.index)

        ma = df["close"].rolling(self.lookback).mean()
        std = df["close"].rolling(self.lookback).std()

        z = (df["close"] - ma) / std

        for i in range(self.lookback + 1, len(df)):
            previous = df.iloc[i - 1]
            current = df.iloc[i]
            previous_z = z.iloc[i - 1]
            current_z = z.iloc[i]

            if pd.isna(previous_z) or pd.isna(current_z):
                continue

            if (
                previous_z >= self.entry_z
                and current_z < previous_z
                and self._bearish_reversal(current, previous)
            ):
                signals.iloc[i] = -1
            elif (
                previous_z <= -self.entry_z
                and current_z > previous_z
                and self._bullish_reversal(current, previous)
            ):
                signals.iloc[i] = 1

        return signals
