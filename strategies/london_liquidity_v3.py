import pandas as pd
from strategies.base_strategy import BaseStrategy
from engine.sessions import SessionEngine


class LondonLiquidityV3(BaseStrategy):

    def __init__(self):
        self.session = SessionEngine()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        df = self.session.add_sessions(df)

        signals = pd.Series(0, index=df.index)

        df["hour"] = df.index.hour

        # STRICT LONDON KILL ZONE
        df["killzone"] = (df["hour"] >= 7) & (df["hour"] <= 10)

        for i in range(200, len(df)):

            if not df["killzone"].iloc[i]:
                continue

            asian_high = df["asian_high"].iloc[i]
            asian_low = df["asian_low"].iloc[i]

            if pd.isna(asian_high) or pd.isna(asian_low):
                continue

            price = df["close"].iloc[i]

            # -----------------------------
            # LIQUIDITY SWEEP LOGIC
            # -----------------------------

            # sweep above Asian high then reject → SELL
            if (
                df["high"].iloc[i] > asian_high and
                price < asian_high
            ):
                signals.iloc[i] = -1

            # sweep below Asian low then reject → BUY
            elif (
                df["low"].iloc[i] < asian_low and
                price > asian_low
            ):
                signals.iloc[i] = 1

        return signals