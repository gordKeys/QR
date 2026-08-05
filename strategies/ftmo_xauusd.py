import pandas as pd

from strategies.base_strategy import BaseStrategy


class FTMOXAUUSD(BaseStrategy):

    def __init__(
        self,
        rr=1.5,
        atr_mult=1.5,
        min_score=3,
        fast=5,
        slow=13,
        rsi_period=9,
        overbought=68,
        oversold=32,
        trend_ema=20,
        risk_pct=0.4,
        stop_buffer_mult=2.0,
        min_atr_points=5,
        forex_hours=range(8, 22),
        volume_factor=1.05,
    ):
        self.rr = rr
        self.atr_mult = atr_mult
        self.min_score = min_score
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.trend_ema = trend_ema
        self.risk_pct = risk_pct
        self.stop_buffer_mult = stop_buffer_mult
        self.min_atr_points = min_atr_points
        self.forex_hours = set(forex_hours)
        self.volume_factor = volume_factor

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss))

        df["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - close.shift()).abs(),
                (df["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        df["vol_avg"] = volume.rolling(20).mean()
        df["vol"] = volume
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

    def _trend_direction(self, broker, symbol):
        if broker is None or symbol is None:
            return None

        rates = broker.rates_copy(symbol, broker.mt5.TIMEFRAME_H1, 70)
        if rates is None or len(rates) == 0:
            return None

        closes = pd.Series([rate["close"] for rate in rates])
        ema = closes.ewm(span=self.trend_ema, adjust=False).mean()
        return "UP" if rates[-1]["close"] > ema.iloc[-1] else "DOWN"

    def generate_signals(self, data: pd.DataFrame):
        df = self._add_indicators(data)
        signals = pd.Series(0, index=df.index)

        for index in range(25, len(df)):
            current = df.iloc[index]
            previous = df.iloc[index - 1]
            if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
                continue

            volume_ok = current["vol_avg"] > 0 and current["vol"] >= current["vol_avg"] * self.volume_factor

            buy = {
                "ema": previous["ema_fast"] <= previous["ema_slow"] and current["ema_fast"] > current["ema_slow"],
                "macd": current["macd_h"] > 0 and current["macd_h"] > previous["macd_h"],
                "bb": current["close"] <= current["bb_lower"] * 1.001 and current["rsi"] < self.overbought,
                "candle": self._bull_engulf(current, previous) or self._bull_pin(current),
                "volume": volume_ok,
            }
            sell = {
                "ema": previous["ema_fast"] >= previous["ema_slow"] and current["ema_fast"] < current["ema_slow"],
                "macd": current["macd_h"] < 0 and current["macd_h"] < previous["macd_h"],
                "bb": current["close"] >= current["bb_upper"] * 0.999 and current["rsi"] > self.oversold,
                "candle": self._bear_engulf(current, previous) or self._bear_pin(current),
                "volume": volume_ok,
            }

            buy_score = sum(bool(value) for value in buy.values())
            sell_score = sum(bool(value) for value in sell.values())

            if buy_score >= self.min_score and buy_score >= sell_score:
                signals.iloc[index] = 1
            elif sell_score >= self.min_score and sell_score > buy_score:
                signals.iloc[index] = -1

        return signals

    def analyze_trade(self, data: pd.DataFrame, *, broker=None, symbol=None, equity=0.0, signal=None, price=None):
        if signal == 0:
            return None

        now = data.index[-1].to_pydatetime()
        if symbol is not None and now.hour not in self.forex_hours:
            return None

        df = self._add_indicators(data)
        current = df.iloc[-1]
        previous = df.iloc[-2]

        if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
            return None

        trend = self._trend_direction(broker, symbol)
        if signal == 1 and trend == "DOWN":
            return None
        if signal == -1 and trend == "UP":
            return None

        atr = float(current["atr"])
        if price is None:
            price = float(current["close"])

        if broker is not None and symbol is not None:
            info = broker.symbol_info(symbol)
            if info is None:
                return None

            point = float(getattr(info, "point", 0.0) or 0.0)
            if point > 0 and atr / point < self.min_atr_points:
                return None

            spread = 0.0
            tick = broker.symbol_tick(symbol)
            if tick is not None and getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
                spread = float(tick.ask - tick.bid)

            min_points = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
            floor_price = max(min_points * point, spread) * self.stop_buffer_mult
            stop_distance = max(self.atr_mult * atr, floor_price)

            if stop_distance <= 0:
                return None

            if signal == 1:
                stop = price - stop_distance
                target = price + stop_distance * self.rr
            else:
                stop = price + stop_distance
                target = price - stop_distance * self.rr

            stop = round(stop, int(getattr(info, "digits", 2)))
            target = round(target, int(getattr(info, "digits", 2)))

            risk_amount = equity * (self.risk_pct / 100.0)
            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)

            stop_distance_abs = abs(price - stop)
            if stop_distance_abs <= 0:
                return None

            if tick_value > 0 and tick_size > 0:
                value_per_price_unit = tick_value / tick_size
                raw_volume = risk_amount / (stop_distance_abs * value_per_price_unit)
            elif contract_size > 0:
                raw_volume = risk_amount / (stop_distance_abs * contract_size)
            else:
                return None

            volume = broker.normalize_volume(symbol, raw_volume)

            while volume >= float(getattr(info, "volume_min", 0.01) or 0.01):
                margin = broker.order_calc_margin(signal, symbol, volume, price)
                if margin is not None and margin <= equity * 0.85:
                    return {
                        "signal": signal,
                        "price": price,
                        "stop": stop,
                        "target": target,
                        "size": volume,
                        "size_reason": "ftmo",
                    }
                volume = broker.normalize_volume(symbol, volume - float(getattr(info, "volume_step", 0.01) or 0.01))

            return None

        stop_distance = self.atr_mult * atr
        if stop_distance <= 0:
            return None

        if signal == 1:
            stop = price - stop_distance
            target = price + stop_distance * self.rr
        else:
            stop = price + stop_distance
            target = price - stop_distance * self.rr

        risk_amount = equity * (self.risk_pct / 100.0)
        stop_distance_abs = abs(price - stop)
        if stop_distance_abs <= 0:
            return None

        volume = risk_amount / stop_distance_abs
        return {
            "signal": signal,
            "price": price,
            "stop": stop,
            "target": target,
            "size": volume,
            "size_reason": "ftmo_dry_run",
        }
