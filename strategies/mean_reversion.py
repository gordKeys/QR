import pandas as pd

from strategies.base_strategy import BaseStrategy


class MeanReversion(BaseStrategy):

    def __init__(
        self,
        lookback=20,
        entry_z=2.0,
        adx_period=14,
        max_adx=25.0,
        atr_baseline_period=50,
        max_atr_ratio=1.5,
        slope_window=5,
        max_slope_atr=0.75,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.adx_period = adx_period
        self.max_adx = max_adx
        self.atr_baseline_period = atr_baseline_period
        self.max_atr_ratio = max_atr_ratio
        self.slope_window = slope_window
        self.max_slope_atr = max_slope_atr

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]

        frame["mean"] = close.rolling(self.lookback).mean()
        frame["std"] = close.rolling(self.lookback).std()
        frame["z_score"] = (close - frame["mean"]) / frame["std"]

        true_range = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = true_range.ewm(
            alpha=1 / self.adx_period,
            min_periods=self.adx_period,
            adjust=False,
        ).mean()
        frame["atr_baseline"] = frame["atr"].rolling(self.atr_baseline_period).mean()

        upward_move = high.diff()
        downward_move = -low.diff()
        plus_dm = upward_move.where(
            (upward_move > downward_move) & (upward_move > 0),
            0.0,
        )
        minus_dm = downward_move.where(
            (downward_move > upward_move) & (downward_move > 0),
            0.0,
        )
        atr = frame["atr"].replace(0, pd.NA)
        plus_di = 100 * plus_dm.ewm(
            alpha=1 / self.adx_period,
            min_periods=self.adx_period,
            adjust=False,
        ).mean() / atr
        minus_di = 100 * minus_dm.ewm(
            alpha=1 / self.adx_period,
            min_periods=self.adx_period,
            adjust=False,
        ).mean() / atr
        directional_index = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
        frame["adx"] = directional_index.ewm(
            alpha=1 / self.adx_period,
            min_periods=self.adx_period,
            adjust=False,
        ).mean()
        frame["mean_slope"] = frame["mean"].diff(self.slope_window)
        return frame

    @staticmethod
    def _bullish_reversal(current, previous):
        return current["close"] > current["open"] and current["close"] > previous["close"]

    @staticmethod
    def _bearish_reversal(current, previous):
        return current["close"] < current["open"] and current["close"] < previous["close"]

    def _is_range_regime(self, row):
        if pd.isna(row["adx"]) or pd.isna(row["atr_baseline"]) or pd.isna(row["mean_slope"]):
            return False
        if row["adx"] > self.max_adx:
            return False
        if row["atr"] > row["atr_baseline"] * self.max_atr_ratio:
            return False
        return abs(row["mean_slope"]) <= row["atr"] * self.max_slope_atr

    def generate_signals(self, data: pd.DataFrame):
        frame = self._add_indicators(data)
        signals = pd.Series(0, index=frame.index)
        start = max(self.lookback, self.atr_baseline_period, self.slope_window) + 1

        for index in range(start, len(frame)):
            current = frame.iloc[index]
            previous = frame.iloc[index - 1]
            if pd.isna(current["z_score"]) or pd.isna(previous["z_score"]):
                continue
            if not self._is_range_regime(current):
                continue

            bullish_confirmation = current["z_score"] > previous["z_score"] or self._bullish_reversal(current, previous)
            bearish_confirmation = current["z_score"] < previous["z_score"] or self._bearish_reversal(current, previous)

            if previous["z_score"] <= -self.entry_z and bullish_confirmation:
                signals.iloc[index] = 1
            elif previous["z_score"] >= self.entry_z and bearish_confirmation:
                signals.iloc[index] = -1

        return signals
