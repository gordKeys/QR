import pandas as pd
from strategies.base_strategy import BaseStrategy


class VolatilityBreakout(BaseStrategy):

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        df = data.copy()
        signals = pd.Series(0, index=df.index)

        # Rolling volatility (ATR proxy already exists but we reinforce structure)
        range_high = df["high"].rolling(20).max()
        range_low = df["low"].rolling(20).min()

        atr = df["atr"] if "atr" in df.columns else (df["high"] - df["low"]).rolling(14).mean()

        for i in range(25, len(df)):

            price = df["close"].iloc[i]

            # breakout range
            resistance = range_high.iloc[i]
            support = range_low.iloc[i]

            # volatility filter (important prop-firm filter)
            if atr.iloc[i] < atr.rolling(100).mean().iloc[i]:
                continue

            # breakout logic
            if price > resistance:
                signals.iloc[i] = 1

            elif price < support:
                signals.iloc[i] = -1

        return signals