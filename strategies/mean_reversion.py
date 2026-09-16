import pandas as pd
from strategies.base_strategy import BaseStrategy

class MeanReversion(BaseStrategy):

    def __init__(self, lookback=20, entry_z=2.0, min_score=4, max_spread_points=30):
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
            [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14).mean()
        ema = frame["close"].ewm(span=20, adjust=False).mean()
        ranging = (ema.diff().abs() <= atr * 0.35) & (atr <= atr.rolling(20).median() * 1.5)
        return ranging.reindex(data.index, method="ffill").fillna(False)

    def _spread_volatility_series(self, data):
        atr = data.get("atr", pd.Series(index=data.index, dtype=float))
        close = data["close"]
        spread = data.get("spread", pd.Series(0.0, index=data.index)).fillna(0.0)
        median_atr = atr.rolling(100, min_periods=20).median()
        return ((spread <= self.max_spread_points) & (atr / close).between(0.00001, 0.0015) & (atr <= median_atr * 2.0)).fillna(False)

    def _z_series(self, data):
        close = data["close"]
        return (close - close.rolling(self.lookback).mean()) / close.rolling(self.lookback).std()

    def _spread_and_volatility_ok(self, data, spread_points=None):
        current = data.iloc[-1]
        atr = float(current.get("atr", 0.0) or 0.0)
        close = float(current["close"])
        if atr <= 0 or close <= 0:
            return False
        if spread_points is not None and spread_points > self.max_spread_points:
            return False
        history = data["atr"].dropna().tail(100)
        return len(history) >= 20 and 0.00001 <= atr / close <= 0.0015 and atr <= float(history.median()) * 2.0

    def _score_at(self, data, index, z=None, spread_points=None, ranging=None, spread_volatility=None):
        if index < self.lookback + 1:
            return 0, 0, {}
        z = z if z is not None else self._z_series(data)
        previous_z, current_z = z.iloc[index - 1], z.iloc[index]
        if pd.isna(previous_z) or pd.isna(current_z):
            return 0, 0, {}
        direction = -1 if previous_z >= self.entry_z else 1 if previous_z <= -self.entry_z else 0
        if direction == 0:
            return 0, 0, {}
        previous, current = data.iloc[index - 1], data.iloc[index]
        ranging = bool(ranging.iloc[index]) if ranging is not None else bool(self._m15_ranging_series(data).iloc[index])
        spread_ok = bool(spread_volatility.iloc[index]) if spread_points is None and spread_volatility is not None else self._spread_and_volatility_ok(data.iloc[: index + 1], spread_points=spread_points)
        components = {
            "z_extreme": True,
            "z_turning": current_z < previous_z if direction == -1 else current_z > previous_z,
            "reversal_candle": self._bearish_reversal(current, previous) if direction == -1 else self._bullish_reversal(current, previous),
            "m15_ranging": ranging,
            "spread_volatility": spread_ok,
            "previous_z": float(previous_z),
            "current_z": float(current_z),
        }
        score = sum(int(components[name]) for name in ("z_extreme", "z_turning", "reversal_candle", "m15_ranging", "spread_volatility"))
        return direction, score, components

    def evaluate_latest(self, data, spread_points=None):
        direction, score, components = self._score_at(
            data, len(data) - 1, z=self._z_series(data), spread_points=spread_points,
            ranging=self._m15_ranging_series(data), spread_volatility=self._spread_volatility_series(data),
        )
        core_confirmed = all(components.get(name, False) for name in ("z_extreme", "z_turning", "reversal_candle", "m15_ranging"))
        return (direction if score >= self.min_score and core_confirmed else 0), score, components

    def generate_signals(self, data: pd.DataFrame):
        signals = pd.Series(0, index=data.index)
        z = self._z_series(data)
        ranging = self._m15_ranging_series(data)
        spread_volatility = self._spread_volatility_series(data)
        for index in range(self.lookback + 1, len(data)):
            direction, score, components = self._score_at(data, index, z=z, ranging=ranging, spread_volatility=spread_volatility)
            if direction and score >= self.min_score and all(components.get(name, False) for name in ("z_extreme", "z_turning", "reversal_candle", "m15_ranging")):
                signals.iloc[index] = direction
        return signals
