import pandas as pd
from strategies.base_strategy import BaseStrategy

class MeanReversion(BaseStrategy):

    def __init__(
        self,
        lookback=20,
        entry_z=2.0,
        min_score=3,
        max_spread_points=30,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.min_score = min_score
        self.max_spread_points = max_spread_points

    @staticmethod
    def _bullish_reversal(current, previous):
        return current["close"] > current["open"] and current["close"] > previous["close"]

    @staticmethod
    def _bearish_reversal(current, previous):
        return current["close"] < current["open"] and current["close"] < previous["close"]

    @staticmethod
    def _m15_ranging_series(data):
        frame = data[["open", "high", "low", "close"]].resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(frame) < 30:
            return pd.Series(False, index=data.index)

        previous_close = frame["close"].shift()
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14).mean()
        ema = frame["close"].ewm(span=20, adjust=False).mean()
        atr_value = atr.iloc[-1]
        slope = abs(ema.iloc[-1] - ema.iloc[-2])
        slope_series = ema.diff().abs()
        median_atr = atr.rolling(20).median()
        ranging = (slope_series <= atr * 0.35) & (atr <= median_atr * 1.5)
        return ranging.reindex(data.index, method="ffill").fillna(False)

    def _spread_volatility_series(self, data):
        atr = data.get("atr", pd.Series(index=data.index, dtype=float))
        close = data["close"]
        spread = data.get("spread", pd.Series(0.0, index=data.index)).fillna(0.0)
        median_atr = atr.rolling(100, min_periods=20).median()
        atr_ratio = atr / close
        return (
            (spread <= self.max_spread_points)
            & atr_ratio.between(0.00001, 0.0015)
            & (atr <= median_atr * 2.0)
        ).fillna(False)

    def _z_series(self, data):
        close = data["close"]
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        return (close - mean) / std

    def _spread_and_volatility_ok(self, data, spread_points=None):
        series = self._spread_volatility_series(data)
        if spread_points is None:
            return bool(series.iloc[-1])
        current = data.iloc[-1]
        atr = float(current.get("atr", 0.0) or 0.0)
        close = float(current["close"])
        if atr <= 0 or close <= 0 or spread_points > self.max_spread_points:
            return False
        atr_history = data["atr"].dropna().tail(100)
        return (
            len(atr_history) >= 20
            and 0.00001 <= atr / close <= 0.0015
            and atr <= float(atr_history.median()) * 2.0
        )

    def _score_at(self, data, index, z=None, spread_points=None, ranging=None, spread_volatility=None):
        if index < self.lookback + 1:
            return 0, 0, {}

        if z is None:
            z = self._z_series(data)
        previous = data.iloc[index - 1]
        current = data.iloc[index]
        previous_z = z.iloc[index - 1]
        current_z = z.iloc[index]
        if pd.isna(previous_z) or pd.isna(current_z):
            return 0, 0, {}

        direction = 0
        if previous_z >= self.entry_z:
            direction = -1
        elif previous_z <= -self.entry_z:
            direction = 1
        if direction == 0:
            return 0, 0, {}

        z_extreme = True
        z_turning = current_z < previous_z if direction == -1 else current_z > previous_z
        reversal = self._bearish_reversal(current, previous) if direction == -1 else self._bullish_reversal(current, previous)
        if ranging is None:
            ranging = bool(self._m15_ranging_series(data).iloc[-1])
        else:
            ranging = bool(ranging.iloc[index])
        if spread_points is None and spread_volatility is not None:
            spread_volatility = bool(spread_volatility.iloc[index])
        else:
            spread_volatility = self._spread_and_volatility_ok(data.iloc[: index + 1], spread_points=spread_points)
        components = {
            "z_extreme": bool(z_extreme),
            "z_turning": bool(z_turning),
            "reversal_candle": bool(reversal),
            "m15_ranging": bool(ranging),
            "spread_volatility": bool(spread_volatility),
            "previous_z": float(previous_z),
            "current_z": float(current_z),
        }
        score = sum(
            int(components[name])
            for name in ("z_extreme", "z_turning", "reversal_candle", "m15_ranging", "spread_volatility")
        )
        return direction, score, components

    def evaluate_latest(self, data, spread_points=None):
        ranging = self._m15_ranging_series(data)
        spread_volatility = self._spread_volatility_series(data)
        direction, score, components = self._score_at(
            data,
            len(data) - 1,
            z=self._z_series(data),
            spread_points=spread_points,
            ranging=ranging,
            spread_volatility=spread_volatility,
        )
        signal = direction if score >= self.min_score else 0
        return signal, score, components

    @staticmethod
    def risk_fraction(score):
        if score >= 4:
            return 0.025
        if score >= 3:
            return 0.02
        return 0.0

    def generate_signals(self, data: pd.DataFrame):

        df = data.copy()
        signals = pd.Series(0, index=df.index)
        z = self._z_series(df)
        ranging = self._m15_ranging_series(df)
        spread_volatility = self._spread_volatility_series(df)

        for i in range(self.lookback + 1, len(df)):
            signal, _, _ = self._score_at(
                df,
                i,
                z=z,
                ranging=ranging,
                spread_volatility=spread_volatility,
            )
            signals.iloc[i] = signal

        return signals
