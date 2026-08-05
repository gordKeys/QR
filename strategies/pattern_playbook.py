import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class PatternPlaybookStrategy(BaseStrategy):

    def __init__(
        self,
        *,
        setup_mode="all",
        entry_style="breakout",
        higher_timeframe="H4",
        trend_fast=20,
        trend_slow=50,
        lookback=60,
        pivot_window=3,
        swing_tolerance=0.004,
        rr=2.0,
        atr_mult=1.4,
        min_atr_points=8,
        volume_factor=1.05,
        retest_window=3,
        strategy_name=None,
    ):
        self.setup_mode = setup_mode
        self.entry_style = entry_style
        self.higher_timeframe = higher_timeframe
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.lookback = lookback
        self.pivot_window = pivot_window
        self.swing_tolerance = swing_tolerance
        self.rr = rr
        self.atr_mult = atr_mult
        self.min_atr_points = min_atr_points
        self.volume_factor = volume_factor
        self.retest_window = retest_window
        self.strategy_name = strategy_name or f"pattern_playbook_{setup_mode}_{entry_style}"

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        close = frame["close"]
        volume = frame["tick_volume"]

        frame["ema_fast"] = close.ewm(span=self.trend_fast, adjust=False).mean()
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
        frame["swing_high"] = frame["high"].rolling(self.lookback).max().shift(1)
        frame["swing_low"] = frame["low"].rolling(self.lookback).min().shift(1)
        return frame

    def _higher_tf_rule(self):
        mapping = {
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
        }
        return mapping.get(str(self.higher_timeframe).upper(), "4h")

    def _higher_tf_bias(self, frame: pd.DataFrame):
        higher = frame[["open", "high", "low", "close", "tick_volume"]].resample(self._higher_tf_rule()).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        if len(higher) < self.trend_slow:
            return pd.Series(index=frame.index, dtype="object")

        higher["ema_fast"] = higher["close"].ewm(span=self.trend_fast, adjust=False).mean()
        higher["ema_slow"] = higher["close"].ewm(span=self.trend_slow, adjust=False).mean()
        bias = pd.Series(index=higher.index, dtype="object")
        bias.loc[:] = "neutral"
        bias.loc[higher["ema_fast"] > higher["ema_slow"]] = "up"
        bias.loc[higher["ema_fast"] < higher["ema_slow"]] = "down"
        return bias.reindex(frame.index, method="ffill")

    @staticmethod
    def _bullish_candle(current, previous):
        engulf = (
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["open"] <= previous["close"]
            and current["close"] >= previous["open"]
        )
        body = abs(current["close"] - current["open"])
        hammer = body > 0 and (min(current["close"], current["open"]) - current["low"]) >= 2 * body
        return engulf or hammer

    @staticmethod
    def _bearish_candle(current, previous):
        engulf = (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] >= previous["close"]
            and current["close"] <= previous["open"]
        )
        body = abs(current["close"] - current["open"])
        shooting_star = body > 0 and (current["high"] - max(current["close"], current["open"])) >= 2 * body
        return engulf or shooting_star

    def _pivot_points(self, frame: pd.DataFrame, end_index: int):
        start = max(1, end_index - self.lookback)
        end = max(start + 2, end_index)
        pivots = []

        for index in range(start, end - 1):
            current = frame.iloc[index]
            left = frame.iloc[max(0, index - self.pivot_window):index]
            right = frame.iloc[index + 1:min(len(frame), index + 1 + self.pivot_window)]
            if len(left) < self.pivot_window or len(right) < self.pivot_window:
                continue

            if current["high"] >= left["high"].max() and current["high"] >= right["high"].max():
                pivots.append(("high", index, float(current["high"])))
            if current["low"] <= left["low"].min() and current["low"] <= right["low"].min():
                pivots.append(("low", index, float(current["low"])))

        pivots.sort(key=lambda item: item[1])
        deduped = []
        for pivot in pivots:
            if not deduped or deduped[-1][1] != pivot[1] or deduped[-1][0] != pivot[0]:
                deduped.append(pivot)
        return deduped

    @staticmethod
    def _line_value(start_index, start_price, end_index, end_price, current_index):
        if end_index == start_index:
            return float(end_price)
        slope = (end_price - start_price) / (end_index - start_index)
        return float(start_price + slope * (current_index - start_index))

    @staticmethod
    def _fit_line(points):
        if len(points) < 2:
            return None
        x = np.array([point[1] for point in points], dtype=float)
        y = np.array([point[2] for point in points], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

    def _triangle_plan(self, frame: pd.DataFrame, index: int, current, previous, bias):
        window = frame.iloc[max(0, index - self.lookback): index + 1]
        if len(window) < max(self.lookback // 2, 20):
            return None

        x = np.arange(len(window), dtype=float)
        upper_slope, upper_intercept = np.polyfit(x, window["high"].values, 1)
        lower_slope, lower_intercept = np.polyfit(x, window["low"].values, 1)
        upper_now = upper_slope * (len(window) - 1) + upper_intercept
        lower_now = lower_slope * (len(window) - 1) + lower_intercept
        upper_start = upper_intercept
        lower_start = lower_intercept
        gap_start = upper_start - lower_start
        gap_now = upper_now - lower_now
        converging = gap_start > 0 and gap_now < gap_start * 0.85
        volume_ok = current["vol_avg"] > 0 and current["vol"] >= current["vol_avg"] * self.volume_factor
        bullish = self._bullish_candle(current, previous)
        bearish = self._bearish_candle(current, previous)
        height = float(window["high"].max() - window["low"].min())
        atr = float(current["atr"])
        if atr <= 0:
            return None

        upper_flat = abs(upper_slope) <= abs(lower_slope) * 0.25
        lower_flat = abs(lower_slope) <= abs(upper_slope) * 0.25
        symmetrical = upper_slope < 0 and lower_slope > 0
        ascending = upper_flat and lower_slope > 0
        descending = lower_flat and upper_slope < 0

        pattern_name = None
        if symmetrical:
            pattern_name = "sym_triangle"
        elif ascending:
            pattern_name = "asc_triangle"
        elif descending:
            pattern_name = "desc_triangle"

        if pattern_name is None or not converging:
            return None

        breakout_up = current["close"] > upper_now and volume_ok and bullish
        breakout_down = current["close"] < lower_now and volume_ok and bearish

        if self.entry_style == "retest":
            recent_breakout_up = any(bar["close"] > upper_now for _, bar in window.iloc[-self.retest_window:].iterrows())
            recent_breakout_down = any(bar["close"] < lower_now for _, bar in window.iloc[-self.retest_window:].iterrows())
            breakout_up = recent_breakout_up and current["low"] <= upper_now and current["close"] > upper_now and bullish
            breakout_down = recent_breakout_down and current["high"] >= lower_now and current["close"] < lower_now and bearish

        if bias == "down":
            breakout_up = False
        if bias == "up":
            breakout_down = False

        if not breakout_up and not breakout_down:
            return None

        direction = 1 if breakout_up else -1
        if direction == 1:
            stop = min(lower_now, current["low"]) - max(atr * self.atr_mult, 0.25 * height)
            target = current["close"] + max(height, abs(current["close"] - stop) * self.rr)
        else:
            stop = max(upper_now, current["high"]) + max(atr * self.atr_mult, 0.25 * height)
            target = current["close"] - max(height, abs(stop - current["close"]) * self.rr)

        return {
            "signal": direction,
            "price": float(current["close"]),
            "stop": float(stop),
            "target": float(target),
            "size": 0.0,
            "size_reason": "triangle_breakout",
            "pattern": pattern_name,
        }

    def _head_shoulders_plan(self, frame: pd.DataFrame, index: int, current, previous, bias):
        pivots = self._pivot_points(frame, index)
        if len(pivots) < 5:
            return None

        recent = pivots[-7:]
        highs = [pivot for pivot in recent if pivot[0] == "high"]
        lows = [pivot for pivot in recent if pivot[0] == "low"]
        if len(highs) < 3 or len(lows) < 2:
            return None

        last_highs = highs[-3:]
        last_lows = lows[-2:]
        shoulder_tolerance = self.swing_tolerance
        neckline_tolerance = self.swing_tolerance * 1.5
        atr = float(current["atr"])
        if atr <= 0:
            return None

        bearish_hs = (
            last_highs[0][1] < last_lows[0][1] < last_highs[1][1] < last_lows[1][1] < last_highs[2][1]
            and last_highs[1][2] > last_highs[0][2] * (1 + shoulder_tolerance)
            and last_highs[1][2] > last_highs[2][2] * (1 + shoulder_tolerance)
            and abs(last_highs[0][2] - last_highs[2][2]) / max(last_highs[0][2], last_highs[2][2]) <= shoulder_tolerance
            and abs(last_lows[0][2] - last_lows[1][2]) / max(last_lows[0][2], last_lows[1][2]) <= neckline_tolerance
        )

        if bearish_hs:
            neckline = min(last_lows[0][2], last_lows[1][2])
            breakout = current["close"] < neckline and self._bearish_candle(current, previous)
            if self.entry_style == "retest":
                breakout = current["high"] >= neckline and current["close"] < neckline and self._bearish_candle(current, previous)
            if bias == "up":
                breakout = False
            if breakout:
                pattern_height = last_highs[1][2] - neckline
                stop = last_highs[1][2] + max(atr * self.atr_mult, pattern_height * 0.25)
                target = current["close"] - max(pattern_height, abs(stop - current["close"]) * self.rr)
                return {
                    "signal": -1,
                    "price": float(current["close"]),
                    "stop": float(stop),
                    "target": float(target),
                    "size": 0.0,
                    "size_reason": "head_shoulders",
                    "pattern": "head_shoulders",
                }

        if len(lows) >= 3 and len(highs) >= 2:
            left_shoulder = lows[-3]
            head = lows[-2]
            right_shoulder = lows[-1]
            if (
                left_shoulder[1] < highs[-2][1] < head[1] < highs[-1][1] < right_shoulder[1]
                and head[2] < min(left_shoulder[2], right_shoulder[2]) * (1 - shoulder_tolerance)
                and abs(left_shoulder[2] - right_shoulder[2]) / max(left_shoulder[2], right_shoulder[2]) <= shoulder_tolerance
            ):
                neckline = max(highs[-2][2], highs[-1][2])
                breakout = current["close"] > neckline and self._bullish_candle(current, previous)
                if self.entry_style == "retest":
                    breakout = current["low"] <= neckline and current["close"] > neckline and self._bullish_candle(current, previous)
                if bias == "down":
                    breakout = False
                if breakout:
                    pattern_height = neckline - head[2]
                    stop = head[2] - max(atr * self.atr_mult, pattern_height * 0.25)
                    target = current["close"] + max(pattern_height, abs(current["close"] - stop) * self.rr)
                    return {
                        "signal": 1,
                        "price": float(current["close"]),
                        "stop": float(stop),
                        "target": float(target),
                        "size": 0.0,
                        "size_reason": "inverse_head_shoulders",
                        "pattern": "inverse_head_shoulders",
                    }

        return None

    def _double_triple_plan(self, frame: pd.DataFrame, index: int, current, previous, bias):
        pivots = self._pivot_points(frame, index)
        if len(pivots) < 4:
            return None

        recent_highs = [pivot for pivot in pivots if pivot[0] == "high"][-4:]
        recent_lows = [pivot for pivot in pivots if pivot[0] == "low"][-4:]
        atr = float(current["atr"])
        if atr <= 0:
            return None

        def equal_prices(a, b, tolerance):
            return abs(a - b) / max(a, b) <= tolerance

        if len(recent_highs) >= 2:
            top1, top2 = recent_highs[-2:]
            neckline_points = [pivot for pivot in pivots if pivot[0] == "low" and top1[1] < pivot[1] < top2[1]]
            if equal_prices(top1[2], top2[2], self.swing_tolerance) and neckline_points:
                neckline = min(point[2] for point in neckline_points)
                breakout = current["close"] < neckline and self._bearish_candle(current, previous)
                if self.entry_style == "retest":
                    breakout = current["high"] >= neckline and current["close"] < neckline and self._bearish_candle(current, previous)
                if bias == "up":
                    breakout = False
                if breakout:
                    pattern_height = max(top1[2], top2[2]) - neckline
                    stop = max(top1[2], top2[2]) + max(atr * self.atr_mult, pattern_height * 0.25)
                    target = current["close"] - max(pattern_height, abs(stop - current["close"]) * self.rr)
                    return {
                        "signal": -1,
                        "price": float(current["close"]),
                        "stop": float(stop),
                        "target": float(target),
                        "size": 0.0,
                        "size_reason": "double_top",
                        "pattern": "double_top",
                    }

        if len(recent_lows) >= 2:
            bot1, bot2 = recent_lows[-2:]
            neckline_points = [pivot for pivot in pivots if pivot[0] == "high" and bot1[1] < pivot[1] < bot2[1]]
            if equal_prices(bot1[2], bot2[2], self.swing_tolerance) and neckline_points:
                neckline = max(point[2] for point in neckline_points)
                breakout = current["close"] > neckline and self._bullish_candle(current, previous)
                if self.entry_style == "retest":
                    breakout = current["low"] <= neckline and current["close"] > neckline and self._bullish_candle(current, previous)
                if bias == "down":
                    breakout = False
                if breakout:
                    pattern_height = neckline - min(bot1[2], bot2[2])
                    stop = min(bot1[2], bot2[2]) - max(atr * self.atr_mult, pattern_height * 0.25)
                    target = current["close"] + max(pattern_height, abs(current["close"] - stop) * self.rr)
                    return {
                        "signal": 1,
                        "price": float(current["close"]),
                        "stop": float(stop),
                        "target": float(target),
                        "size": 0.0,
                        "size_reason": "double_bottom",
                        "pattern": "double_bottom",
                    }

        if len(recent_highs) >= 3:
            top1, top2, top3 = recent_highs[-3:]
            neckline_points = [pivot for pivot in pivots if pivot[0] == "low" and top1[1] < pivot[1] < top3[1]]
            if equal_prices(top1[2], top2[2], self.swing_tolerance) and equal_prices(top2[2], top3[2], self.swing_tolerance) and neckline_points:
                neckline = min(point[2] for point in neckline_points)
                breakout = current["close"] < neckline and self._bearish_candle(current, previous)
                if bias == "up":
                    breakout = False
                if breakout:
                    pattern_height = max(top1[2], top2[2], top3[2]) - neckline
                    stop = max(top1[2], top2[2], top3[2]) + max(atr * self.atr_mult, pattern_height * 0.25)
                    target = current["close"] - max(pattern_height, abs(stop - current["close"]) * self.rr)
                    return {
                        "signal": -1,
                        "price": float(current["close"]),
                        "stop": float(stop),
                        "target": float(target),
                        "size": 0.0,
                        "size_reason": "triple_top",
                        "pattern": "triple_top",
                    }

        if len(recent_lows) >= 3:
            bot1, bot2, bot3 = recent_lows[-3:]
            neckline_points = [pivot for pivot in pivots if pivot[0] == "high" and bot1[1] < pivot[1] < bot3[1]]
            if equal_prices(bot1[2], bot2[2], self.swing_tolerance) and equal_prices(bot2[2], bot3[2], self.swing_tolerance) and neckline_points:
                neckline = max(point[2] for point in neckline_points)
                breakout = current["close"] > neckline and self._bullish_candle(current, previous)
                if bias == "down":
                    breakout = False
                if breakout:
                    pattern_height = neckline - min(bot1[2], bot2[2], bot3[2])
                    stop = min(bot1[2], bot2[2], bot3[2]) - max(atr * self.atr_mult, pattern_height * 0.25)
                    target = current["close"] + max(pattern_height, abs(current["close"] - stop) * self.rr)
                    return {
                        "signal": 1,
                        "price": float(current["close"]),
                        "stop": float(stop),
                        "target": float(target),
                        "size": 0.0,
                        "size_reason": "triple_bottom",
                        "pattern": "triple_bottom",
                    }

        return None

    def _mtf_reversal_plan(self, frame: pd.DataFrame, index: int, current, previous, bias):
        if pd.isna(current["swing_high"]) or pd.isna(current["swing_low"]) or pd.isna(current["atr"]):
            return None

        volume_ok = current["vol_avg"] > 0 and current["vol"] >= current["vol_avg"] * self.volume_factor
        bullish = self._bullish_candle(current, previous)
        bearish = self._bearish_candle(current, previous)
        atr = float(current["atr"])
        if atr <= 0:
            return None

        if bias == "up":
            support_touch = current["low"] <= current["ema_fast"] or current["low"] <= current["swing_low"] + 0.25 * atr
            if support_touch and bullish and volume_ok:
                stop = min(current["swing_low"], current["low"]) - atr * self.atr_mult
                target = current["close"] + abs(current["close"] - stop) * self.rr
                return {
                    "signal": 1,
                    "price": float(current["close"]),
                    "stop": float(stop),
                    "target": float(target),
                    "size": 0.0,
                    "size_reason": "mtf_reversal",
                    "pattern": "mtf_reversal",
                }

        if bias == "down":
            resistance_touch = current["high"] >= current["ema_fast"] or current["high"] >= current["swing_high"] - 0.25 * atr
            if resistance_touch and bearish and volume_ok:
                stop = max(current["swing_high"], current["high"]) + atr * self.atr_mult
                target = current["close"] - abs(stop - current["close"]) * self.rr
                return {
                    "signal": -1,
                    "price": float(current["close"]),
                    "stop": float(stop),
                    "target": float(target),
                    "size": 0.0,
                    "size_reason": "mtf_reversal",
                    "pattern": "mtf_reversal",
                }

        return None

    def _pattern_plan(self, frame: pd.DataFrame, index: int):
        current = frame.iloc[index]
        previous = frame.iloc[index - 1]
        if pd.isna(current["atr"]):
            return None

        bias = self._higher_tf_bias(frame).iloc[index]
        candidates = []

        if self.setup_mode in ("all", "triangles"):
            candidates.append(self._triangle_plan(frame, index, current, previous, bias))
        if self.setup_mode in ("all", "head_shoulders"):
            candidates.append(self._head_shoulders_plan(frame, index, current, previous, bias))
        if self.setup_mode in ("all", "double_triple"):
            candidates.append(self._double_triple_plan(frame, index, current, previous, bias))
        if self.setup_mode in ("all", "mtf_reversal"):
            candidates.append(self._mtf_reversal_plan(frame, index, current, previous, bias))

        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    def generate_signals(self, data: pd.DataFrame):
        frame = self._add_indicators(data)
        signals = pd.Series(0, index=frame.index)

        start = max(self.lookback, self.trend_slow, self.pivot_window * 2) + 2
        for index in range(start, len(frame)):
            plan = self._pattern_plan(frame, index)
            if plan is not None:
                signals.iloc[index] = int(plan["signal"])

        return signals

    def build_trade_plan(self, data: pd.DataFrame, index: int, symbol: str, cost_profile, equity: float):
        frame = self._add_indicators(data)
        if index < 2:
            return None

        plan = self._pattern_plan(frame, index)
        if plan is None:
            return None

        plan["setup"] = self.setup_mode
        return plan
