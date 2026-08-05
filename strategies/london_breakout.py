import pandas as pd
from strategies.base_strategy import BaseStrategy
from engine.indicators import ema, atr
from engine.sessions import SessionEngine


class LondonBreakoutStrategyV2(BaseStrategy):

    def __init__(self):
        self.session = SessionEngine()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()

        # -----------------------
        # SESSIONS
        # -----------------------
        df = self.session.add_sessions(df)

        # -----------------------
        # INDICATORS
        # -----------------------
        df["ema50"] = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
        df["atr"] = atr(df)

        signals = pd.Series(0, index=df.index)

        # -----------------------
        # FILTERS PRECOMPUTATION
        # -----------------------

        # ATR regime filter (volatility expansion only)
        df["atr_filter"] = df["atr"] > df["atr"].rolling(100).median()

        # Trend strength (not just direction)
        df["trend_strength"] = abs(df["ema50"] - df["ema200"]) > (df["atr"] * 0.5)

        # Breakout confirmation candle
        df["bull_confirm"] = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
        df["bear_confirm"] = (df["close"] < df["open"]) & (df["close"] < df["low"].shift(1))

        # -----------------------
        # STRICT LONDON WINDOW (KILL ZONE)
        # -----------------------
        df["hour"] = df.index.hour
        df["london_killzone"] = (df["hour"] >= 7) & (df["hour"] <= 10)

        # -----------------------
        # MAIN LOOP
        # -----------------------
        for i in range(200, len(df)):

            # must be London kill zone only
            if not df["london_killzone"].iloc[i]:
                continue

            # must have valid asian range
            asian_high = df["asian_high"].iloc[i]
            asian_low = df["asian_low"].iloc[i]

            if pd.isna(asian_high) or pd.isna(asian_low):
                continue

            price = df["close"].iloc[i]

            trend_up = df["ema50"].iloc[i] > df["ema200"].iloc[i]
            trend_down = df["ema50"].iloc[i] < df["ema200"].iloc[i]

            # -----------------------
            # QUALITY FILTER STACK
            # -----------------------
            if not df["atr_filter"].iloc[i]:
                continue

            if not df["trend_strength"].iloc[i]:
                continue

            # -----------------------
            # BREAKOUT LOGIC (CONFIRMED ONLY)
            # -----------------------

            # BUY
            if (
                trend_up and
                price > asian_high and
                df["bull_confirm"].iloc[i]
            ):
                signals.iloc[i] = 1

            # SELL
            elif (
                trend_down and
                price < asian_low and
                df["bear_confirm"].iloc[i]
            ):
                signals.iloc[i] = -1

        return signals