"""
Standalone Exness MT5 bot for the current QuantFX strategy set.

Usage example:
  python exness_single_bot.py ^
    --login 12345678 --password YOUR_PASSWORD --server "Exness-MT5" ^
    --terminal-path "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
"""

from __future__ import annotations

# Fill these in once if you want to run the bot without command-line args.
# Leave them as None / placeholders if you prefer env vars or CLI overrides.
EXNESS_LOGIN = None  # e.g. 12345678
EXNESS_PASSWORD = None  # e.g. "your_password"
EXNESS_SERVER = None  # e.g. "Exness-MT5"

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception as exc:  # pragma: no cover - import guard for local syntax checks
    mt5 = None
    MT5_IMPORT_ERROR = exc
else:
    MT5_IMPORT_ERROR = None


@dataclass
class TradePlan:
    signal: int
    price: float
    stop: float
    target: float
    size: float
    reason: str


class BaseStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, entry_z: float = 2.0):
        self.lookback = lookback
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = data.copy()
        signals = pd.Series(0, index=frame.index)
        ma = frame["close"].rolling(self.lookback).mean()
        std = frame["close"].rolling(self.lookback).std()
        z = (frame["close"] - ma) / std
        for index in range(self.lookback, len(frame)):
            if z.iloc[index] > self.entry_z:
                signals.iloc[index] = -1
            elif z.iloc[index] < -self.entry_z:
                signals.iloc[index] = 1
        return signals


class MomentumStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = data.copy()
        signals = pd.Series(0, index=frame.index)
        returns = frame["close"].pct_change(3)
        for index in range(10, len(frame)):
            if returns.iloc[index] > 0.001:
                signals.iloc[index] = 1
            elif returns.iloc[index] < -0.001:
                signals.iloc[index] = -1
        return signals


class TrendFollowingStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = data.copy()
        signals = pd.Series(0, index=frame.index)
        ema_fast = frame["close"].ewm(span=50, adjust=False).mean()
        ema_slow = frame["close"].ewm(span=200, adjust=False).mean()
        for index in range(200, len(frame)):
            if ema_fast.iloc[index] > ema_slow.iloc[index]:
                signals.iloc[index] = 1
            elif ema_fast.iloc[index] < ema_slow.iloc[index]:
                signals.iloc[index] = -1
        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = data.copy()
        signals = pd.Series(0, index=frame.index)
        range_high = frame["high"].rolling(20).max()
        range_low = frame["low"].rolling(20).min()
        atr = frame["atr"] if "atr" in frame.columns else (frame["high"] - frame["low"]).rolling(14).mean()
        atr_mean = atr.rolling(100).mean()
        for index in range(25, len(frame)):
            if pd.isna(range_high.iloc[index]) or pd.isna(range_low.iloc[index]) or pd.isna(atr.iloc[index]):
                continue
            if atr.iloc[index] < atr_mean.iloc[index]:
                continue
            price = frame["close"].iloc[index]
            if price > range_high.iloc[index]:
                signals.iloc[index] = 1
            elif price < range_low.iloc[index]:
                signals.iloc[index] = -1
        return signals


class LionOfJudahFiveSignalStrategy(BaseStrategy):
    def __init__(
        self,
        *,
        use_trend: bool = True,
        use_candle: bool = True,
        require_trend_alignment: bool = True,
        min_score: int = 3,
        rr: float = 1.5,
        atr: float = 1.5,
        fast: int = 5,
        slow: int = 13,
        rsi_p: int = 9,
        ob: int = 68,
        os: int = 32,
        mirror: bool = False,
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

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        close = frame["close"]
        volume = frame["tick_volume"]

        frame["ema_fast"] = close.ewm(span=self.fast, adjust=False).mean()
        frame["ema_slow"] = close.ewm(span=self.slow, adjust=False).mean()

        macd_fast = close.ewm(span=12, adjust=False).mean()
        macd_slow = close.ewm(span=26, adjust=False).mean()
        frame["macd"] = macd_fast - macd_slow
        frame["macd_sig"] = frame["macd"].ewm(span=9, adjust=False).mean()
        frame["macd_h"] = frame["macd"] - frame["macd_sig"]

        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_p, min_periods=self.rsi_p, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_p, min_periods=self.rsi_p, adjust=False).mean()
        frame["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

        frame["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        frame["bb_upper"] = frame["bb_mid"] + 2 * bb_std
        frame["bb_lower"] = frame["bb_mid"] - 2 * bb_std

        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - close.shift()).abs(),
                (frame["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        frame["vol_avg"] = volume.rolling(20).mean()
        frame["vol"] = volume
        frame["sr_sup"] = frame["low"].rolling(20).min().shift(1)
        frame["sr_res"] = frame["high"].rolling(20).max().shift(1)
        return frame

    @staticmethod
    def bull_engulf(current, previous) -> bool:
        return (
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["open"] < previous["close"]
            and current["close"] > previous["open"]
        )

    @staticmethod
    def bear_engulf(current, previous) -> bool:
        return (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] > previous["close"]
            and current["close"] < previous["open"]
        )

    @staticmethod
    def bull_pin(row) -> bool:
        body = abs(row["close"] - row["open"])
        return body > 0 and (min(row["close"], row["open"]) - row["low"]) >= 2 * body and (row["high"] - max(row["close"], row["open"])) <= body

    @staticmethod
    def bear_pin(row) -> bool:
        body = abs(row["close"] - row["open"])
        return body > 0 and (row["high"] - max(row["close"], row["open"])) >= 2 * body and (min(row["close"], row["open"]) - row["low"]) <= body

    def trend_direction(self, data: pd.DataFrame) -> Optional[str]:
        if len(data) < 70:
            return None
        hourly = data[["open", "high", "low", "close", "tick_volume"]].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        if len(hourly) < 20:
            return None
        ema = hourly["close"].ewm(span=20, adjust=False).mean()
        return "UP" if hourly["close"].iloc[-1] > ema.iloc[-1] else "DOWN"

    def score_signals(self, current, previous, trend: Optional[str]):
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
            buy["candle"] = self.bull_engulf(current, previous) or self.bull_pin(current)
            sell["candle"] = self.bear_engulf(current, previous) or self.bear_pin(current)

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

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = self.add_indicators(data)
        signals = pd.Series(0, index=frame.index)
        for index in range(25, len(frame)):
            current = frame.iloc[index]
            previous = frame.iloc[index - 1]
            if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
                continue
            trend = self.trend_direction(frame.iloc[: index + 1])
            buy_score, sell_score = self.score_signals(current, previous, trend)
            if buy_score >= self.min_score and buy_score >= sell_score:
                signals.iloc[index] = 1
            elif sell_score >= self.min_score and sell_score > buy_score:
                signals.iloc[index] = -1
        return signals

    def build_trade_plan(self, broker, symbol: str, equity: float, signal: int, price: float, current) -> Optional[TradePlan]:
        info = broker.symbol_info(symbol)
        if info is None:
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        atr = float(current["atr"])
        if point <= 0 or atr / point < 5:
            return None

        tick = broker.symbol_tick(symbol)
        if tick is None:
            return None

        min_points = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        spread_points = 0.0
        if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
            spread_points = float(tick.ask - tick.bid) / point
        floor_price = max(min_points, spread_points) * 1.5 * point
        stop_distance = max(self.atr_mult * atr, floor_price)

        order_signal = -signal if self.mirror else signal
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

        stop = round(stop, int(getattr(info, "digits", 2) or 2))
        target = round(target, int(getattr(info, "digits", 2) or 2))

        return TradePlan(signal=order_signal, price=price, stop=stop, target=target, size=0.0, reason="lion")


class FTMOXAUUSDStrategy(BaseStrategy):
    def __init__(
        self,
        rr: float = 1.5,
        atr_mult: float = 1.5,
        min_score: int = 3,
        fast: int = 5,
        slow: int = 13,
        rsi_period: int = 9,
        overbought: int = 68,
        oversold: int = 32,
        trend_ema: int = 20,
        risk_pct: float = 0.4,
        stop_buffer_mult: float = 2.0,
        min_atr_points: float = 5.0,
        forex_hours=range(8, 22),
        volume_factor: float = 1.05,
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

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        close = frame["close"]
        volume = frame["tick_volume"]

        frame["ema_fast"] = close.ewm(span=self.fast, adjust=False).mean()
        frame["ema_slow"] = close.ewm(span=self.slow, adjust=False).mean()

        macd_fast = close.ewm(span=12, adjust=False).mean()
        macd_slow = close.ewm(span=26, adjust=False).mean()
        frame["macd"] = macd_fast - macd_slow
        frame["macd_sig"] = frame["macd"].ewm(span=9, adjust=False).mean()
        frame["macd_h"] = frame["macd"] - frame["macd_sig"]

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        frame["rsi"] = 100 - (100 / (1 + gain / loss))

        frame["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        frame["bb_upper"] = frame["bb_mid"] + 2 * bb_std
        frame["bb_lower"] = frame["bb_mid"] - 2 * bb_std

        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - close.shift()).abs(),
                (frame["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        frame["vol_avg"] = volume.rolling(20).mean()
        frame["vol"] = volume
        return frame

    @staticmethod
    def bull_engulf(current, previous) -> bool:
        return (
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["open"] < previous["close"]
            and current["close"] > previous["open"]
        )

    @staticmethod
    def bear_engulf(current, previous) -> bool:
        return (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] > previous["close"]
            and current["close"] < previous["open"]
        )

    @staticmethod
    def bull_pin(row) -> bool:
        body = abs(row["close"] - row["open"])
        return body > 0 and (min(row["close"], row["open"]) - row["low"]) >= 2 * body and (row["high"] - max(row["close"], row["open"])) <= body

    @staticmethod
    def bear_pin(row) -> bool:
        body = abs(row["close"] - row["open"])
        return body > 0 and (row["high"] - max(row["close"], row["open"])) >= 2 * body and (min(row["close"], row["open"]) - row["low"]) <= body

    def trend_direction(self, broker, symbol: str) -> Optional[str]:
        if broker is None or symbol is None:
            return None
        rates = broker.rates_copy(symbol, broker.mt5.TIMEFRAME_H1, 70)
        if rates is None or len(rates) == 0:
            return None
        closes = pd.Series([rate["close"] for rate in rates])
        ema = closes.ewm(span=self.trend_ema, adjust=False).mean()
        return "UP" if rates[-1]["close"] > ema.iloc[-1] else "DOWN"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = self.add_indicators(data)
        signals = pd.Series(0, index=frame.index)
        for index in range(25, len(frame)):
            current = frame.iloc[index]
            previous = frame.iloc[index - 1]
            if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
                continue
            volume_ok = current["vol_avg"] > 0 and current["vol"] >= current["vol_avg"] * self.volume_factor
            buy = {
                "ema": previous["ema_fast"] <= previous["ema_slow"] and current["ema_fast"] > current["ema_slow"],
                "macd": current["macd_h"] > 0 and current["macd_h"] > previous["macd_h"],
                "bb": current["close"] <= current["bb_lower"] * 1.001 and current["rsi"] < self.overbought,
                "candle": self.bull_engulf(current, previous) or self.bull_pin(current),
                "volume": volume_ok,
            }
            sell = {
                "ema": previous["ema_fast"] >= previous["ema_slow"] and current["ema_fast"] < current["ema_slow"],
                "macd": current["macd_h"] < 0 and current["macd_h"] < previous["macd_h"],
                "bb": current["close"] >= current["bb_upper"] * 0.999 and current["rsi"] > self.oversold,
                "candle": self.bear_engulf(current, previous) or self.bear_pin(current),
                "volume": volume_ok,
            }
            buy_score = sum(bool(value) for value in buy.values())
            sell_score = sum(bool(value) for value in sell.values())
            if buy_score >= self.min_score and buy_score >= sell_score:
                signals.iloc[index] = 1
            elif sell_score >= self.min_score and sell_score > buy_score:
                signals.iloc[index] = -1
        return signals

    def build_trade_plan(self, broker, symbol: str, equity: float, signal: int, price: float, current) -> Optional[TradePlan]:
        now = current.name.to_pydatetime() if hasattr(current.name, "to_pydatetime") else datetime.now(timezone.utc)
        if now.hour not in self.forex_hours:
            return None

        info = broker.symbol_info(symbol)
        if info is None:
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        atr = float(current["atr"])
        if point > 0 and atr / point < self.min_atr_points:
            return None

        spread = 0.0
        tick = broker.symbol_tick(symbol)
        if tick is not None and getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
            spread = float(tick.ask - tick.bid)

        min_points = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        floor_price = max(min_points * point, spread) * self.stop_buffer_mult
        stop_distance = max(self.atr_mult * atr, floor_price)

        if signal == 1:
            stop = price - stop_distance
            target = price + stop_distance * self.rr
        else:
            stop = price + stop_distance
            target = price - stop_distance * self.rr

        digits = int(getattr(info, "digits", 2) or 2)
        stop = round(stop, digits)
        target = round(target, digits)
        return TradePlan(signal=signal, price=price, stop=stop, target=target, size=0.0, reason="ftmo")


class StrategyRouter:
    def __init__(self):
        self.registry = {
            "mean_reversion": MeanReversionStrategy(lookback=20, entry_z=1.5),
            "mean_reversion_strict": MeanReversionStrategy(lookback=30, entry_z=2.0),
            "momentum": MomentumStrategy(),
            "trend": TrendFollowingStrategy(),
            "volatility_breakout": VolatilityBreakoutStrategy(),
            "ftmo_xauusd": FTMOXAUUSDStrategy(min_score=4),
            "lion_usdjpy_mirror": LionOfJudahFiveSignalStrategy(
                use_trend=True,
                use_candle=True,
                require_trend_alignment=True,
                min_score=5,
                rr=1.5,
                atr=0.8,
                fast=5,
                slow=13,
                rsi_p=9,
                ob=68,
                os=32,
                mirror=True,
            ),
            "lion_usdchf_actual": LionOfJudahFiveSignalStrategy(
                use_trend=True,
                use_candle=True,
                require_trend_alignment=True,
                min_score=3,
                rr=1.5,
                atr=0.8,
                fast=5,
                slow=13,
                rsi_p=9,
                ob=68,
                os=32,
                mirror=False,
            ),
        }
        self.symbol_map = {
            "EURUSD": "mean_reversion_strict",
            "GBPUSD": "mean_reversion_strict",
            "USDJPY": "lion_usdjpy_mirror",
            "USDCHF": "lion_usdchf_actual",
            "XAUUSD": "ftmo_xauusd",
        }
        self.default_strategy = "mean_reversion"

    def get_strategy_name(self, symbol: str) -> str:
        return self.symbol_map.get(symbol.upper(), self.default_strategy)

    def get_strategy(self, symbol: str) -> BaseStrategy:
        return self.registry[self.get_strategy_name(symbol)]


class ExnessMT5Bot:
    def __init__(self, args):
        self.args = args
        self.router = StrategyRouter()
        self.magic = args.magic

    def initialize(self):
        if mt5 is None:
            raise RuntimeError(f"MetaTrader5 import failed: {MT5_IMPORT_ERROR}")

        init_kwargs = {"path": self.args.terminal_path}
        if self.args.login:
            init_kwargs["login"] = int(self.args.login)
        if self.args.password:
            init_kwargs["password"] = self.args.password
        if self.args.server:
            init_kwargs["server"] = self.args.server

        if not mt5.initialize(**init_kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        account = mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account_info() returned None")

        print(
            f"Connected | balance={account.balance:.2f} | equity={account.equity:.2f} | "
            f"leverage=1:{getattr(account, 'leverage', 'n/a')}"
        )

    def shutdown(self):
        if mt5 is not None:
            mt5.shutdown()

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        wanted = symbol.upper()
        symbols = mt5.symbols_get() or []
        exact = {getattr(item, "name", "").upper(): getattr(item, "name", "") for item in symbols}
        if wanted in exact:
            return exact[wanted]
        prefix_matches = []
        for item in symbols:
            name = getattr(item, "name", "")
            upper_name = name.upper()
            if upper_name.startswith(wanted) and len(upper_name) - len(wanted) <= 8:
                prefix_matches.append(name)
        if not prefix_matches:
            return None
        return sorted(prefix_matches)[0]

    def ensure_symbol(self, symbol: str) -> Optional[str]:
        resolved = self.resolve_symbol(symbol)
        if resolved is None:
            print(f"{symbol}: not found in broker symbol list")
            return None
        if not mt5.symbol_select(resolved, True):
            print(f"{resolved}: symbol_select failed")
            return None
        return resolved

    def fetch_data(self, symbol: str, bars: int = 2500) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("time")
        expected = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        for column in expected:
            if column not in frame.columns:
                frame[column] = 0
        frame = frame[expected].sort_index()
        frame["returns"] = frame["close"].pct_change()
        frame["ema50"] = frame["close"].ewm(span=50, adjust=False).mean()
        frame["ema200"] = frame["close"].ewm(span=200, adjust=False).mean()
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - frame["close"].shift()).abs(),
                (frame["low"] - frame["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        frame["volatility"] = frame["returns"].rolling(20).std()
        return frame.dropna()

    def has_open_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        return bool(positions)

    def spread_points(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return 0.0
        point = float(getattr(info, "point", 0.0) or 0.0)
        if point <= 0 or getattr(tick, "ask", None) is None or getattr(tick, "bid", None) is None:
            return 0.0
        return float(tick.ask - tick.bid) / point

    def volume_for_plan(self, symbol: str, direction: int, entry: float, stop: float, equity: float) -> tuple[float, str]:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0, "no_symbol_info"

        min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
        max_volume = float(getattr(info, "volume_max", min_volume) or min_volume)
        step = float(getattr(info, "volume_step", 0.01) or 0.01)
        volume = float(self.args.fixed_lot if self.args.fixed_lot is not None else min_volume)
        volume = max(min_volume, min(volume, max_volume))
        if self.args.lot_mode == "risk":
            risk_amount = equity * self.args.risk_pct
            stop_distance = abs(entry - stop)
            if stop_distance <= 0:
                return 0.0, "invalid_stop"
            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
            if tick_value > 0 and tick_size > 0:
                value_per_price_unit = tick_value / tick_size
                raw_volume = risk_amount / (stop_distance * value_per_price_unit)
            elif contract_size > 0:
                raw_volume = risk_amount / (stop_distance * contract_size)
            else:
                raw_volume = min_volume
            volume = max(min_volume, min(float(raw_volume), max_volume))

        normalized = self.normalize_volume(symbol, volume)
        while normalized >= min_volume:
            order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
            margin = mt5.order_calc_margin(order_type, symbol, normalized, entry)
            if margin is not None and margin <= equity * 0.9:
                return normalized, "ok"
            if self.args.lot_mode == "fixed":
                break
            normalized = self.normalize_volume(symbol, normalized - step)
        return 0.0, "insufficient_margin"

    def normalize_volume(self, symbol: str, volume: float) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            return volume
        min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
        max_volume = float(getattr(info, "volume_max", volume) or volume)
        step = float(getattr(info, "volume_step", 0.01) or 0.01)
        clipped = max(min_volume, min(volume, max_volume))
        steps = round(clipped / step)
        normalized = steps * step
        return max(min_volume, round(normalized, 8))

    def min_stop_distance(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0
        stops_level = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        point = float(getattr(info, "point", 0.0) or 0.0)
        return stops_level * point

    def conform_stops(self, symbol: str, direction: int, entry: float, stop: float, target: float):
        info = mt5.symbol_info(symbol)
        if info is None:
            return None, None
        min_distance = self.min_stop_distance(symbol)
        if min_distance > 0:
            if direction == 1:
                stop = min(stop, entry - min_distance)
                target = max(target, entry + min_distance)
            else:
                stop = max(stop, entry + min_distance)
                target = min(target, entry - min_distance)
        digits = int(getattr(info, "digits", 2) or 2)
        stop = round(stop, digits)
        target = round(target, digits)
        entry = round(entry, digits)
        if direction == 1 and not (stop < entry < target):
            return None, None
        if direction == -1 and not (target < entry < stop):
            return None, None
        return stop, target

    def place_order(self, symbol: str, direction: int, volume: float, stop: float, target: float, price: float):
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop,
            "tp": target,
            "deviation": self.args.deviation,
            "magic": int(self.magic),
            "comment": self.args.comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        info = mt5.symbol_info(symbol)
        fill_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        if info is not None:
            supported = getattr(info, "filling_mode", 0) or 0
            preferred = []
            if supported & 2:
                preferred.append(mt5.ORDER_FILLING_IOC)
            if supported & 1:
                preferred.append(mt5.ORDER_FILLING_FOK)
            for mode in fill_modes:
                if mode not in preferred:
                    preferred.append(mode)
            fill_modes = preferred

        last_result = None
        for fill_mode in fill_modes:
            request["type_filling"] = fill_mode
            last_result = mt5.order_send(request)
            if last_result is not None and getattr(last_result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                return last_result
        return last_result

    def build_trade_plan(self, symbol: str, data: pd.DataFrame, signal: int, price: float, strategy: BaseStrategy) -> Optional[TradePlan]:
        if hasattr(strategy, "build_trade_plan"):
            current = strategy.add_indicators(data).iloc[-1]
            return strategy.build_trade_plan(self, symbol, self.equity, signal, price, current)

        current_atr = float(data["atr"].iloc[-1])
        if current_atr <= 0:
            return None
        rr_map = {
            "mean_reversion": 1.5,
            "mean_reversion_strict": 1.5,
            "momentum": 1.2,
            "trend": 2.0,
            "volatility_breakout": 2.0,
        }
        rr = rr_map.get(self.router.get_strategy_name(symbol), 1.5)
        stop_distance = max(current_atr * 1.2, self.min_stop_distance(symbol))
        if signal == 1:
            stop = price - stop_distance
            target = price + stop_distance * rr
        else:
            stop = price + stop_distance
            target = price - stop_distance * rr
        stop, target = self.conform_stops(symbol, signal, price, stop, target)
        if stop is None or target is None:
            return None
        return TradePlan(signal=signal, price=price, stop=stop, target=target, size=0.0, reason="generic")

    def run_once(self):
        account = mt5.account_info()
        if account is None:
            print("No account info available.")
            return

        self.equity = float(getattr(account, "equity", getattr(account, "balance", 0.0)) or 0.0)

        for requested_symbol in self.args.symbols:
            symbol = self.ensure_symbol(requested_symbol)
            if symbol is None:
                continue

            if self.has_open_position(symbol):
                print(f"{symbol}: skipped because position already open")
                continue

            data = self.fetch_data(symbol)
            if data is None or data.empty:
                print(f"{symbol}: no data available")
                continue

            spread = self.spread_points(symbol)
            if self.args.max_spread_points > 0 and spread > self.args.max_spread_points:
                print(f"{symbol}: skipped because spread is too wide ({spread:.1f} points)")
                continue

            strategy = self.router.get_strategy(requested_symbol)
            signal = int(strategy.generate_signals(data).iloc[-1])
            if signal == 0:
                print(f"{symbol}: no trade")
                continue

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"{symbol}: no live tick")
                continue

            entry_price = float(tick.ask if signal == 1 else tick.bid)
            plan = self.build_trade_plan(symbol, data, signal, entry_price, strategy)
            if plan is None:
                print(f"{symbol}: skipped because trade plan could not be built")
                continue

            volume, size_reason = self.volume_for_plan(symbol, plan.signal, plan.price, plan.stop, self.equity)
            if volume <= 0:
                print(f"{symbol}: skipped due to sizing ({size_reason})")
                continue

            plan.size = volume
            stop, target = self.conform_stops(symbol, plan.signal, plan.price, plan.stop, plan.target)
            if stop is None or target is None:
                print(f"{symbol}: skipped because broker stop levels are invalid")
                continue

            if self.args.dry_run:
                print(
                    f"{symbol}: DRY signal={plan.signal} size={plan.size:.2f} "
                    f"entry={plan.price:.5f} sl={stop:.5f} tp={target:.5f} strategy={strategy.__class__.__name__}"
                )
                continue

            result = self.place_order(symbol, plan.signal, plan.size, stop, target, plan.price)
            print(
                f"{symbol}: signal={plan.signal} size={plan.size:.2f} "
                f"entry={plan.price:.5f} sl={stop:.5f} tp={target:.5f} "
                f"result={result}"
            )

    def run(self):
        try:
            while True:
                self.run_once()
                if self.args.loop_once:
                    break
                time.sleep(self.args.poll_seconds)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.shutdown()


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone Exness MT5 bot for the QuantFX strategy set.")
    parser.add_argument("--terminal-path", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    parser.add_argument("--login", type=str, default=os.getenv("EXNESS_LOGIN") or EXNESS_LOGIN)
    parser.add_argument("--password", type=str, default=os.getenv("EXNESS_PASSWORD") or EXNESS_PASSWORD)
    parser.add_argument("--server", type=str, default=os.getenv("EXNESS_SERVER") or EXNESS_SERVER)
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--magic", type=int, default=26072026)
    parser.add_argument("--comment", type=str, default="Exness-QuantFX")
    parser.add_argument("--deviation", type=int, default=20)
    parser.add_argument("--max-spread-points", type=float, default=80.0)
    parser.add_argument("--lot-mode", choices=["fixed", "risk"], default="fixed")
    parser.add_argument("--fixed-lot", type=float, default=0.01)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.login or not args.password or not args.server:
        print(
            "Missing Exness credentials. Fill in EXNESS_LOGIN, EXNESS_PASSWORD, and EXNESS_SERVER "
            "at the top of exness_single_bot.py, or pass them as CLI args."
        )
        sys.exit(1)

    bot = ExnessMT5Bot(args)
    bot.initialize()
    bot.run()


if __name__ == "__main__":
    main()
