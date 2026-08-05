import pandas as pd

from strategies.base_strategy import BaseStrategy


class MultiTimeframePriceActionStrategy(BaseStrategy):

    def __init__(
        self,
        *,
        htf="H1",
        setup_mode="all",
        trend_fast=20,
        trend_slow=50,
        pullback_ema=20,
        swing_lookback=20,
        fib_tolerance=0.15,
        rr=2.0,
        atr_mult=1.5,
        min_atr_points=5,
        volume_factor=1.05,
        strategy_name=None,
    ):
        self.htf = htf
        self.setup_mode = setup_mode
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.pullback_ema = pullback_ema
        self.swing_lookback = swing_lookback
        self.fib_tolerance = fib_tolerance
        self.rr = rr
        self.atr_mult = atr_mult
        self.min_atr_points = min_atr_points
        self.volume_factor = volume_factor
        self.strategy_name = strategy_name or f"mtf_price_action_{setup_mode}"

    def _htf_rule(self):
        mapping = {
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
        }
        return mapping.get(str(self.htf).upper(), "1h")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        close = frame["close"]
        volume = frame["tick_volume"]

        frame["ema_fast"] = close.ewm(span=self.pullback_ema, adjust=False).mean()
        frame["ema_slow"] = close.ewm(span=self.trend_slow, adjust=False).mean()
        frame["atr"] = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - close.shift()).abs(),
                (frame["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        frame["vol_avg"] = volume.rolling(20).mean()
        frame["vol"] = volume
        frame["swing_high"] = frame["high"].rolling(self.swing_lookback).max().shift(1)
        frame["swing_low"] = frame["low"].rolling(self.swing_lookback).min().shift(1)
        frame["range_high"] = frame["high"].rolling(self.swing_lookback).max().shift(1)
        frame["range_low"] = frame["low"].rolling(self.swing_lookback).min().shift(1)
        return frame

    def _htf_bias(self, frame: pd.DataFrame):
        htf = frame[["open", "high", "low", "close", "tick_volume"]].resample(self._htf_rule()).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        if len(htf) < self.trend_slow:
            return pd.Series(index=frame.index, dtype="object")

        htf["ema_fast"] = htf["close"].ewm(span=self.trend_fast, adjust=False).mean()
        htf["ema_slow"] = htf["close"].ewm(span=self.trend_slow, adjust=False).mean()
        bias = pd.Series(index=frame.index, dtype="object")
        htf_bias = pd.Series(index=htf.index, dtype="object")
        htf_bias.loc[:] = "neutral"
        htf_bias.loc[htf["ema_fast"] > htf["ema_slow"]] = "up"
        htf_bias.loc[htf["ema_fast"] < htf["ema_slow"]] = "down"
        bias = htf_bias.reindex(frame.index, method="ffill")
        return bias

    @staticmethod
    def _bullish_candle(current, previous):
        engulf = (
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["open"] <= previous["close"]
            and current["close"] >= previous["open"]
        )
        body = abs(current["close"] - current["open"])
        pin = body > 0 and (min(current["close"], current["open"]) - current["low"]) >= 2 * body
        return engulf or pin

    @staticmethod
    def _bearish_candle(current, previous):
        engulf = (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] >= previous["close"]
            and current["close"] <= previous["open"]
        )
        body = abs(current["close"] - current["open"])
        pin = body > 0 and (current["high"] - max(current["close"], current["open"])) >= 2 * body
        return engulf or pin

    def _fib_zone(self, current, swing_low, swing_high, direction):
        if pd.isna(swing_low) or pd.isna(swing_high) or swing_high <= swing_low:
            return False

        retrace_382 = swing_high - (swing_high - swing_low) * 0.382
        retrace_500 = swing_high - (swing_high - swing_low) * 0.500
        retrace_618 = swing_high - (swing_high - swing_low) * 0.618
        if direction == 1:
            return current["low"] <= retrace_500 * (1 + self.fib_tolerance / 100) and current["close"] >= retrace_618
        return current["high"] >= retrace_500 * (1 - self.fib_tolerance / 100) and current["close"] <= retrace_382

    @staticmethod
    def _strong_breakout_candle(current, direction):
        body = abs(current["close"] - current["open"])
        candle_range = max(current["high"] - current["low"], 1e-9)
        body_ratio = body / candle_range
        if direction == 1:
            return current["close"] >= current["high"] - 0.2 * candle_range and body_ratio >= 0.45
        return current["close"] <= current["low"] + 0.2 * candle_range and body_ratio >= 0.45

    def generate_signals(self, data: pd.DataFrame):
        frame = self._add_indicators(data)
        bias = self._htf_bias(frame)
        signals = pd.Series(0, index=frame.index)

        for index in range(max(self.swing_lookback, self.trend_slow) + 2, len(frame)):
            current = frame.iloc[index]
            previous = frame.iloc[index - 1]
            htf_bias = bias.iloc[index]
            if pd.isna(current["atr"]) or pd.isna(current["swing_high"]) or pd.isna(current["swing_low"]):
                continue

            volume_ok = current["vol_avg"] > 0 and current["vol"] >= current["vol_avg"] * self.volume_factor
            trend_up = htf_bias == "up"
            trend_down = htf_bias == "down"

            bullish_pullback = trend_up and current["close"] >= current["ema_fast"] and self._bullish_candle(current, previous)
            bearish_pullback = trend_down and current["close"] <= current["ema_fast"] and self._bearish_candle(current, previous)
            bullish_breakout = trend_up and current["close"] > current["swing_high"] and volume_ok and self._strong_breakout_candle(current, 1)
            bearish_breakout = trend_down and current["close"] < current["swing_low"] and volume_ok and self._strong_breakout_candle(current, -1)
            bullish_fib = trend_up and self._fib_zone(current, current["swing_low"], current["swing_high"], 1) and self._bullish_candle(current, previous)
            bearish_fib = trend_down and self._fib_zone(current, current["swing_low"], current["swing_high"], -1) and self._bearish_candle(current, previous)

            if self.setup_mode in ("all", "trend_pullback") and bullish_pullback:
                signals.iloc[index] = 1
            elif self.setup_mode in ("all", "trend_pullback") and bearish_pullback:
                signals.iloc[index] = -1
            elif self.setup_mode in ("all", "breakout") and bullish_breakout:
                signals.iloc[index] = 1
            elif self.setup_mode in ("all", "breakout") and bearish_breakout:
                signals.iloc[index] = -1
            elif self.setup_mode in ("all", "fib_retracement") and bullish_fib:
                signals.iloc[index] = 1
            elif self.setup_mode in ("all", "fib_retracement") and bearish_fib:
                signals.iloc[index] = -1

        return signals

    def build_trade_plan(self, data: pd.DataFrame, index: int, symbol: str, cost_profile, equity: float):
        frame = self._add_indicators(data)
        current = frame.iloc[index]
        signal = int(self.generate_signals(data).iloc[index])
        if signal == 0 or pd.isna(current["atr"]):
            return None

        if signal == 1:
            stop_anchor = current["swing_low"] if not pd.isna(current["swing_low"]) else current["low"]
            stop = min(stop_anchor, current["low"]) - current["atr"] * self.atr_mult
            target = current["close"] + abs(current["close"] - stop) * self.rr
        else:
            stop_anchor = current["swing_high"] if not pd.isna(current["swing_high"]) else current["high"]
            stop = max(stop_anchor, current["high"]) + current["atr"] * self.atr_mult
            target = current["close"] - abs(stop - current["close"]) * self.rr

        if pd.isna(stop) or pd.isna(target):
            return None

        return {
            "signal": signal,
            "price": float(current["close"]),
            "stop": float(stop),
            "target": float(target),
            "size": 0.0,
            "size_reason": "price_action",
            "setup": self.setup_mode,
        }
