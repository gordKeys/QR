import pandas as pd


class SignalFilter:

    def compute_score(self, df, i, base_signal):

        score = 50

        regime = df["regime"].iloc[i]

        # ONLY reward alignment, not magnitude
        if regime == 1 and abs(base_signal) == 1:
            score += 20

        elif regime == -1 and base_signal == -1:
            score += 20

        elif regime == 2 and base_signal == 1:
            score += 15

        else:
            score -= 15

        # ATR filter (fix rolling instability)
        if "atr" in df.columns:
            atr_now = df["atr"].iloc[i]
            atr_mean = df["atr"].iloc[max(0, i - 50):i].mean()

            if atr_now < atr_mean * 0.7:
                score -= 20

        # EMA trend sanity
        if base_signal == 1 and df["ema50"].iloc[i] < df["ema200"].iloc[i]:
            score -= 10

        if base_signal == -1 and df["ema50"].iloc[i] > df["ema200"].iloc[i]:
            score -= 10

        return max(0, min(100, score))