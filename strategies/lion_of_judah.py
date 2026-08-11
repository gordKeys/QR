import pandas as pd

from strategies.base_strategy import BaseStrategy


class LionOfJudahFiveSignalStrategy(BaseStrategy):

    def __init__(
        self,
        *,
        use_trend=True,
        use_candle=True,
        require_trend_alignment=True,
        min_score=3,
        rr=1.5,
        atr=1.5,
        fast=5,
        slow=13,
        rsi_p=9,
        ob=68,
        os=32,
        mirror=False,
        trend_timeframe="15min",
        strategy_name="lion_of_judah",
    ):
        self.use_trend = use_trend
        self.use_candle = use_candle
        self.require_trend_alignment = require_trend_alignment
        self.min_score = min_score
        self.rr = rr
        self.atr_mult = atr
        self.fast = fast
        self.slow = slow
        self.rsi_p = rsi_p
        self.ob = ob
        self.os = os
        self.mirror = mirror
        self.trend_timeframe = trend_timeframe
        self.strategy_name = strategy_name

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        close = df["close"]
        volume = df["tick_volume"]

        df["ema_fast"] = close.ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.slow, adjust=False).mean()

        macd_fast = close.ewm(span=12, adjust=False).mean()
        macd_slow = close.ewm(span=26, adjust=False).mean()
        df["macd"] = macd_fast - macd_slow
        df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_h"] = df["macd"] - df["macd_sig"]

        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_p, min_periods=self.rsi_p, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_p, min_periods=self.rsi_p, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

        df["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std

        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - close.shift()).abs(),
                (df["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        df["vol_avg"] = volume.rolling(20).mean()
        df["vol"] = volume
        df["sr_sup"] = df["low"].rolling(20).min().shift(1)
        df["sr_res"] = df["high"].rolling(20).max().shift(1)
        return df

    @staticmethod
    def _bull_engulf(current, previous):
        return (
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["open"] < previous["close"]
            and current["close"] > previous["open"]
        )

    @staticmethod
    def _bear_engulf(current, previous):
        return (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] > previous["close"]
            and current["close"] < previous["open"]
        )

    @staticmethod
    def _bull_pin(row):
        body = abs(row["close"] - row["open"])
        return body > 0 and (min(row["close"], row["open"]) - row["low"]) >= 2 * body and (row["high"] - max(row["close"], row["open"])) <= body

    @staticmethod
    def _bear_pin(row):
        body = abs(row["close"] - row["open"])
        return body > 0 and (row["high"] - max(row["close"], row["open"])) >= 2 * body and (min(row["close"], row["open"]) - row["low"]) <= body

    def _trend_direction(self, data: pd.DataFrame):
        if len(data) < 70:
            return None

        trend_frame = data[["open", "high", "low", "close", "tick_volume"]].resample(self.trend_timeframe).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        if len(trend_frame) < 20:
            return None

        ema = trend_frame["close"].ewm(span=20, adjust=False).mean()
        return "UP" if trend_frame["close"].iloc[-1] > ema.iloc[-1] else "DOWN"

    def _score_signals(self, current, previous, trend):
        volume_ok = current["vol_avg"] > 0 and current["vol"] >= current["vol_avg"] * 1.05
        buy = {}
        sell = {}

        if self.use_trend:
            buy["ema"] = previous["ema_fast"] <= previous["ema_slow"] and current["ema_fast"] > current["ema_slow"]
            sell["ema"] = previous["ema_fast"] >= previous["ema_slow"] and current["ema_fast"] < current["ema_slow"]

        buy["bb"] = current["close"] <= current["bb_lower"] * 1.001
        sell["bb"] = current["close"] >= current["bb_upper"] * 0.999

        buy["rsi"] = current["rsi"] <= self.os
        sell["rsi"] = current["rsi"] >= self.ob

        if self.use_candle:
            buy["candle"] = self._bull_engulf(current, previous) or self._bull_pin(current)
            sell["candle"] = self._bear_engulf(current, previous) or self._bear_pin(current)

        buy["volume"] = volume_ok
        sell["volume"] = volume_ok

        support = current.get("sr_sup")
        resistance = current.get("sr_res")
        atr = current["atr"]
        buy["sr"] = support is not None and not pd.isna(support) and current["low"] <= support + 0.5 * atr
        sell["sr"] = resistance is not None and not pd.isna(resistance) and current["high"] >= resistance - 0.5 * atr

        buy_score = sum(bool(value) for value in buy.values())
        sell_score = sum(bool(value) for value in sell.values())

        if self.require_trend_alignment and trend:
            if trend == "DOWN":
                buy_score = 0
            if trend == "UP":
                sell_score = 0

        return buy_score, sell_score

    def generate_signals(self, data: pd.DataFrame):
        frame = self._add_indicators(data)
        signals = pd.Series(0, index=frame.index)

        for index in range(25, len(frame)):
            current = frame.iloc[index]
            previous = frame.iloc[index - 1]
            if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
                continue

            trend = self._trend_direction(frame.iloc[: index + 1])
            buy_score, sell_score = self._score_signals(current, previous, trend)

            if buy_score >= self.min_score and buy_score >= sell_score:
                signals.iloc[index] = 1
            elif sell_score >= self.min_score and sell_score > buy_score:
                signals.iloc[index] = -1

        return signals

    def _build_trade_plan(self, broker, symbol, equity, base_signal, price, current):
        info = broker.symbol_info(symbol)
        if info is None:
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        atr = float(current["atr"])
        if point <= 0 or atr / point < 5:
            return None

        atr_stop = self.atr_mult * atr
        min_points = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        tick = broker.symbol_tick(symbol)
        if tick is None:
            return None
        spread_points = 0.0
        if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
            spread_points = float(tick.ask - tick.bid) / point
        floor_price = max(min_points, spread_points) * 1.5 * point
        stop_distance = max(atr_stop, floor_price)

        order_signal = -base_signal if self.mirror else base_signal
        sl_distance = stop_distance * self.rr if self.mirror else stop_distance
        tp_distance = stop_distance if self.mirror else stop_distance * self.rr

        if price is None:
            price = float(tick.ask if order_signal == 1 else tick.bid)

        if order_signal == 1:
            price = float(tick.ask if tick is not None else price)
            stop = price - sl_distance
            target = price + tp_distance
        else:
            price = float(tick.bid if tick is not None else price)
            stop = price + sl_distance
            target = price - tp_distance

        digits = int(getattr(info, "digits", 2) or 2)
        stop = round(stop, digits)
        target = round(target, digits)

        risk_amount = equity * 0.004
        stop_distance_abs = abs(price - stop)
        if stop_distance_abs <= 0:
            return None

        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)

        if tick_value > 0 and tick_size > 0:
            value_per_price_unit = tick_value / tick_size
            raw_volume = risk_amount / (stop_distance_abs * value_per_price_unit)
        elif contract_size > 0:
            raw_volume = risk_amount / (stop_distance_abs * contract_size)
        else:
            return None

        volume = broker.normalize_volume(symbol, raw_volume)
        min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
        volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)

        while volume >= min_volume:
            margin = broker.order_calc_margin(order_signal, symbol, volume, price)
            if margin is not None and margin <= equity * 0.85:
                return {
                    "signal": order_signal,
                    "price": price,
                    "stop": stop,
                    "target": target,
                    "size": volume,
                    "size_reason": "lion",
                }
            volume = broker.normalize_volume(symbol, volume - volume_step)

        return None

    def analyze_trade(self, data: pd.DataFrame, *, broker=None, symbol=None, equity=0.0, signal=None, price=None):
        if signal == 0:
            return None

        frame = self._add_indicators(data)
        current = frame.iloc[-1]
        if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
            return None

        trend = self._trend_direction(frame)
        if self.require_trend_alignment and trend:
            if signal == 1 and trend == "DOWN":
                return None
            if signal == -1 and trend == "UP":
                return None

        if broker is not None and symbol is not None:
            return self._build_trade_plan(broker, symbol, equity, signal, price, current)

        atr_stop = self.atr_mult * float(current["atr"])
        if atr_stop <= 0:
            return None

        order_signal = -signal if self.mirror else signal
        if price is None:
            price = float(current["close"])

        if order_signal == 1:
            stop = price - (atr_stop * self.rr if self.mirror else atr_stop)
            target = price + (atr_stop if self.mirror else atr_stop * self.rr)
        else:
            stop = price + (atr_stop * self.rr if self.mirror else atr_stop)
            target = price - (atr_stop if self.mirror else atr_stop * self.rr)

        risk_amount = equity * 0.004
        stop_distance_abs = abs(price - stop)
        if stop_distance_abs <= 0:
            return None

        volume = risk_amount / stop_distance_abs
        return {
            "signal": order_signal,
            "price": price,
            "stop": stop,
            "target": target,
            "size": volume,
            "size_reason": "lion_dry_run",
        }
