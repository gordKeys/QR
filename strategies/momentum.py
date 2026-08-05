import pandas as pd
from strategies.base_strategy import BaseStrategy

class Momentum(BaseStrategy):

    def generate_signals(self, data: pd.DataFrame):

        df = data.copy()
        signals = pd.Series(0, index=df.index)

        ret = df["close"].pct_change(3)

        for i in range(10, len(df)):

            if ret.iloc[i] > 0.001:
                signals.iloc[i] = 1

            elif ret.iloc[i] < -0.001:
                signals.iloc[i] = -1

        return signals