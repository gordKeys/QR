import pandas as pd
from strategies.base_strategy import BaseStrategy

class TrendFollowing(BaseStrategy):

    def generate_signals(self, data: pd.DataFrame):

        df = data.copy()
        signals = pd.Series(0, index=df.index)

        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        ema200 = df["close"].ewm(span=200, adjust=False).mean()

        for i in range(200, len(df)):

            if ema50.iloc[i] > ema200.iloc[i]:
                signals.iloc[i] = 1

            elif ema50.iloc[i] < ema200.iloc[i]:
                signals.iloc[i] = -1

        return signals
