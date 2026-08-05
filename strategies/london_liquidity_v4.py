import pandas as pd
from strategies.base_strategy import BaseStrategy
from engine.sessions import SessionEngine


class LondonLiquidityV4(BaseStrategy):

    def __init__(self):
        self.session = SessionEngine()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        df = self.session.add_sessions(df)

        signals = pd.Series(0, index=df.index)

        df["hour"] = df.index.hour

        # WIDER LONDON WINDOW
        df["window"] = (df["hour"] >= 7) & (df["hour"] <= 12)

        for i in range(200, len(df)):

            if not df["window"].iloc[i]:
                continue

            asian_high = df["asian_high"].iloc[i]
            asian_low = df["asian_low"].iloc[i]

            if pd.isna(asian_high) or pd.isna(asian_low):
                continue

            price = df["close"].iloc[i]

            # -----------------------------
            # CONTINUATION + BREAKOUT LOGIC
            # -----------------------------

            # breakout continuation above Asian high
            if price > asian_high and df["close"].iloc[i-1] <= asian_high:
                signals.iloc[i] = 1

            # breakout continuation below Asian low
            elif price < asian_low and df["close"].iloc[i-1] >= asian_low:
                signals.iloc[i] = -1

        return signals