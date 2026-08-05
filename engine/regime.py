import pandas as pd
import numpy as np


class RegimeDetector:

    def detect(self, df: pd.DataFrame):

        df = df.copy()

        # -------------------------
        # VOLATILITY SCORE
        # -------------------------
        ret = df["close"].pct_change()

        vol = ret.rolling(20).std()
        vol_norm = (vol - vol.rolling(100).mean()) / (vol.rolling(100).std() + 1e-9)

        df["vol_score"] = np.tanh(vol_norm)  # -1 to 1

        # -------------------------
        # TREND STRENGTH SCORE
        # -------------------------
        ema50 = df["close"].ewm(span=50).mean()
        ema200 = df["close"].ewm(span=200).mean()

        trend_strength = (ema50 - ema200) / df["close"]

        trend_norm = (trend_strength - trend_strength.rolling(100).mean()) / (
            trend_strength.rolling(100).std() + 1e-9
        )

        df["trend_score"] = np.tanh(trend_norm)

        # -------------------------
        # FINAL REGIME SCORE (0–1)
        # -------------------------
        df["regime_score"] = (
            0.6 * df["vol_score"].abs() +
            0.4 * df["trend_score"].abs()
        )

        df["regime"] = 0

        df.loc[
            (df["trend_score"] > 0.4) & (df["vol_score"].abs() < 0.5),
            "regime"
        ] = 1  # TREND

        df.loc[
            (df["vol_score"].abs() > 0.5) & (df["trend_score"].abs() < 0.4),
            "regime"
        ] = -1  # MEAN REVERSION

        df.loc[
            (df["vol_score"].abs() > 0.7) & (df["trend_score"].abs() > 0.5),
            "regime"
        ] = 2  # MOMENTUM BREAKOUT

        return df