"""
XAUUSD scalp machine for Exness MT5.

Strategies used:
- Trend breakout
- Trend pullback
- Mean-reversion rejection

Run backtest:
  python xauusd_scalp_machine.py --mode backtest

Run live:
  python xauusd_scalp_machine.py --mode live --login ... --password ... --server ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.mtf_backtester import MTFBacktester
from engine.symbol_universe import default_cost_profile

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception as exc:  # pragma: no cover
    mt5 = None
    MT5_IMPORT_ERROR = exc
else:
    MT5_IMPORT_ERROR = None


EXNESS_LOGIN = 173186259  # e.g. 12345678
EXNESS_PASSWORD = "Gordonpap@2023"  # e.g. "your_password"
EXNESS_SERVER = "Exness-MT5Real"  # e.g. "Exness-MT5"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"


@dataclass
class TradePlan:
    signal: int
    price: float
    stop: float
    target: float
    size: float = 0.0
    size_reason: str = "xau_scalp"


class XAUUSDScalpMachine:
    def __init__(
        self,
        *,
        mode: str = "all",
        trend_fast: int = 20,
        trend_slow: int = 50,
        trend_ema: int = 50,
        rsi_period: int = 7,
        atr_period: int = 14,
        atr_mult: float = 1.15,
        rr_breakout: float = 1.35,
        rr_pullback: float = 1.25,
        rr_reversion: float = 1.0,
        volume_factor: float = 1.05,
        session_start_hour: int = 12,
        session_end_hour: int = 21,
    ):
        self.mode = mode
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.trend_ema = trend_ema
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.rr_breakout = rr_breakout
        self.rr_pullback = rr_pullback
        self.rr_reversion = rr_reversion
        self.volume_factor = volume_factor
        self.session_start_hour = session_start_hour
        self.session_end_hour = session_end_hour
        self.strategy_name = f"xauusd_scalp_{mode}"
        self._cached_data_id = None
        self._cached_signals = None

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        close = frame["close"]
        volume = frame["tick_volume"]

        frame["ema_fast"] = close.ewm(span=self.trend_fast, adjust=False).mean()
        frame["ema_slow"] = close.ewm(span=self.trend_slow, adjust=False).mean()
        frame["ema_trend"] = close.ewm(span=self.trend_ema, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        frame["rsi"] = 100 - (100 / (1 + gain / loss))

        frame["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        frame["bb_upper"] = frame["bb_mid"] + 2 * bb_std
        frame["bb_lower"] = frame["bb_mid"] - 2 * bb_std

        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - close.shift()).abs(),
                (frame["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = tr.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        frame["vol_avg"] = volume.rolling(20).mean()
        frame["vol"] = volume
        frame["range_high"] = frame["high"].rolling(20).max().shift(1)
        frame["range_low"] = frame["low"].rolling(20).min().shift(1)
        return frame

    def _session_ok(self, timestamp) -> bool:
        hour = timestamp.hour
        if self.session_start_hour <= self.session_end_hour:
            return self.session_start_hour <= hour < self.session_end_hour
        return hour >= self.session_start_hour or hour < self.session_end_hour

    def _higher_timeframe_bias(self, frame: pd.DataFrame):
        higher = frame[["open", "high", "low", "close", "tick_volume"]].resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        if len(higher) < max(self.trend_fast, self.trend_slow):
            return pd.Series(index=frame.index, dtype="object")
        fast = higher["close"].ewm(span=self.trend_fast, adjust=False).mean()
        slow = higher["close"].ewm(span=self.trend_slow, adjust=False).mean()
        bias = pd.Series(index=higher.index, dtype="object")
        bias.loc[:] = "neutral"
        bias.loc[fast > slow] = "up"
        bias.loc[fast < slow] = "down"
        return bias.reindex(frame.index, method="ffill")

    @staticmethod
    def _bullish_reversal(current, previous) -> bool:
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
    def _bearish_reversal(current, previous) -> bool:
        engulf = (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] >= previous["close"]
            and current["close"] <= previous["open"]
        )
        body = abs(current["close"] - current["open"])
        pin = body > 0 and (current["high"] - max(current["close"], current["open"])) >= 2 * body
        return engulf or pin

    def _ensure_signals(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        data_id = id(data)
        if self._cached_data_id != data_id or self._cached_signals is None:
            frame = self._add_indicators(data)
            bias = self._higher_timeframe_bias(frame)
            signals = pd.Series(0, index=frame.index)
            start = max(self.trend_slow, 25)
            for index in range(start, len(frame)):
                current = frame.iloc[index]
                previous = frame.iloc[index - 1]
                if pd.isna(current["atr"]) or pd.isna(current["bb_upper"]):
                    continue
                timestamp = frame.index[index]
                if not self._session_ok(timestamp.to_pydatetime()):
                    continue
                if current["vol_avg"] > 0 and current["vol"] < current["vol_avg"] * self.volume_factor:
                    continue
                trend = bias.iloc[index]
                close = current["close"]
                ema_fast = current["ema_fast"]
                ema_slow = current["ema_slow"]

                breakout_up = trend == "up" and close > current["range_high"] and self._strong_breakout(current, 1)
                breakout_down = trend == "down" and close < current["range_low"] and self._strong_breakout(current, -1)

                pullback_up = trend == "up" and close >= ema_fast and current["low"] <= ema_fast and self._bullish_reversal(current, previous)
                pullback_down = trend == "down" and close <= ema_fast and current["high"] >= ema_fast and self._bearish_reversal(current, previous)

                reversion_up = close <= current["bb_lower"] * 1.001 and current["rsi"] <= 30 and self._bullish_reversal(current, previous)
                reversion_down = close >= current["bb_upper"] * 0.999 and current["rsi"] >= 70 and self._bearish_reversal(current, previous)

                if self.mode in ("all", "breakout") and breakout_up:
                    signals.iloc[index] = 1
                elif self.mode in ("all", "breakout") and breakout_down:
                    signals.iloc[index] = -1
                elif self.mode in ("all", "pullback") and pullback_up:
                    signals.iloc[index] = 1
                elif self.mode in ("all", "pullback") and pullback_down:
                    signals.iloc[index] = -1
                elif self.mode in ("all", "reversion") and reversion_up:
                    signals.iloc[index] = 1
                elif self.mode in ("all", "reversion") and reversion_down:
                    signals.iloc[index] = -1

            self._cached_data_id = data_id
            self._cached_signals = signals
            self._cached_frame = frame
        return self._cached_frame, self._cached_signals

    @staticmethod
    def _strong_breakout(current, direction: int) -> bool:
        body = abs(current["close"] - current["open"])
        candle_range = max(current["high"] - current["low"], 1e-9)
        body_ratio = body / candle_range
        if direction == 1:
            return current["close"] >= current["high"] - 0.2 * candle_range and body_ratio >= 0.45
        return current["close"] <= current["low"] + 0.2 * candle_range and body_ratio >= 0.45

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        _, signals = self._ensure_signals(data)
        return signals

    def build_trade_plan(self, data: pd.DataFrame, index: int, symbol: str, cost_profile, equity: float, signal=None):
        frame, signals = self._ensure_signals(data)
        if signal is None:
            signal = int(signals.iloc[index])
        if signal == 0:
            return None

        current = frame.iloc[index]
        atr = float(current["atr"])
        if atr <= 0:
            return None

        recent_high = float(current["range_high"]) if not pd.isna(current["range_high"]) else float(current["high"])
        recent_low = float(current["range_low"]) if not pd.isna(current["range_low"]) else float(current["low"])

        if signal == 1:
            stop = min(current["low"], recent_low) - atr * self.atr_mult
            rr = self.rr_breakout if current["close"] > current["ema_fast"] else self.rr_pullback
            target = current["close"] + abs(current["close"] - stop) * rr
        else:
            stop = max(current["high"], recent_high) + atr * self.atr_mult
            rr = self.rr_breakout if current["close"] < current["ema_fast"] else self.rr_pullback
            target = current["close"] - abs(stop - current["close"]) * rr

        return {
            "signal": signal,
            "price": float(current["close"]),
            "stop": float(stop),
            "target": float(target),
            "size": 0.0,
            "size_reason": "xau_scalp",
        }


class MT5Connector:
    def __init__(self, terminal_path: str, login: Optional[str], password: Optional[str], server: Optional[str]):
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server

    def initialize(self):
        if mt5 is None:
            raise RuntimeError(f"MetaTrader5 package import failed: {MT5_IMPORT_ERROR}")
        kwargs = {"path": self.terminal_path}
        if self.login:
            kwargs["login"] = int(self.login)
        if self.password:
            kwargs["password"] = self.password
        if self.server:
            kwargs["server"] = self.server
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def shutdown(self):
        if mt5 is not None:
            mt5.shutdown()


def load_data(symbol: str, data_dir: str):
    loader = DataLoader(symbol=symbol, data_dir=data_dir)
    frame = loader.load()
    if frame is None or frame.empty:
        return frame
    return FeatureEngine().add_features(frame)


def build_cost_profile(commission_round_turn: float):
    base = default_cost_profile("XAUUSD")
    return base.__class__(
        point=base.point,
        contract_size=base.contract_size,
        commission_round_turn=commission_round_turn,
        swap_long_per_lot_day=base.swap_long_per_lot_day,
        swap_short_per_lot_day=base.swap_short_per_lot_day,
        spread_multiplier=base.spread_multiplier,
    )


def run_backtest(args):
    data = load_data("XAUUSD", args.data_dir)
    if data is None or data.empty:
        print("No XAUUSD data available.")
        return 1

    strategy = XAUUSDScalpMachine(mode=args.strategy_mode)
    cost_profile = build_cost_profile(args.commission_round_turn)
    result = MTFBacktester(
        data=data,
        strategy=strategy,
        symbol="XAUUSD",
        initial_balance=args.initial_balance,
        risk_per_trade=args.risk_pct,
        cost_profile=cost_profile,
    ).run()

    print("\n=== XAUUSD SCALP BACKTEST ===")
    print(f"Mode: {args.strategy_mode}")
    print(f"Final balance: {result['final_balance']:.2f}")
    print(f"Net profit: {result['net_profit']:.2f}")
    print(f"Trades: {result['total_trades']}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Profit factor: {result['profit_factor']:.2f}")
    print(f"Max drawdown: {result['max_drawdown']:.2%}")
    return 0


def normalize_volume(info, requested_volume: float) -> float:
    min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
    max_volume = float(getattr(info, "volume_max", requested_volume) or requested_volume)
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    clipped = max(min_volume, min(requested_volume, max_volume))
    steps = round(clipped / step)
    normalized = steps * step
    return max(min_volume, round(normalized, 8))


def run_live(args):
    connector = MT5Connector(args.terminal_path, args.login or EXNESS_LOGIN, args.password or EXNESS_PASSWORD, args.server or EXNESS_SERVER)
    connector.initialize()

    try:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("No account info available.")
        print(
            f"Connected | balance={account.balance:.2f} | equity={account.equity:.2f} | "
            f"leverage=1:{getattr(account, 'leverage', 'n/a')}"
        )

        strategy = XAUUSDScalpMachine(mode=args.strategy_mode)
        symbol = "XAUUSD"
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError("XAUUSD not selectable on this account.")

        while True:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, args.bars)
            if rates is None or len(rates) == 0:
                print("No live rates.")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            frame = pd.DataFrame(rates)
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame = frame.set_index("time")
            frame = FeatureEngine().add_features(frame[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]])

            signals = strategy.generate_signals(frame)
            signal = int(signals.iloc[-1])
            if signal == 0:
                print("XAUUSD: no signal")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if tick is None or info is None:
                print("XAUUSD: no tick/info")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            if mt5.positions_get(symbol=symbol):
                print("XAUUSD: position already open")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            plan = strategy.build_trade_plan(frame, len(frame) - 1, symbol, None, float(account.equity), signal=signal)
            if plan is None:
                print("XAUUSD: no trade plan")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            price = float(tick.ask if plan["signal"] == 1 else tick.bid)
            stop = float(plan["stop"])
            target = float(plan["target"])
            stop, target = round(stop, int(getattr(info, "digits", 2) or 2)), round(target, int(getattr(info, "digits", 2) or 2))

            spread = 0.0
            if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
                spread = float(tick.ask - tick.bid)
            if args.max_spread_points > 0 and spread > args.max_spread_points * float(getattr(info, "point", 0.01) or 0.01):
                print("XAUUSD: spread too wide")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            risk_amount = float(account.equity) * args.risk_pct
            stop_distance = abs(price - stop)
            if stop_distance <= 0:
                print("XAUUSD: invalid stop")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
            if tick_value > 0 and tick_size > 0:
                value_per_price_unit = tick_value / tick_size
                raw_volume = risk_amount / (stop_distance * value_per_price_unit)
            elif contract_size > 0:
                raw_volume = risk_amount / (stop_distance * contract_size)
            else:
                raw_volume = args.fixed_lot

            volume = normalize_volume(info, min(args.fixed_lot, raw_volume) if args.lot_mode == "risk" else args.fixed_lot)
            if args.lot_mode == "risk":
                volume = normalize_volume(info, max(args.fixed_lot, raw_volume))

            margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY if plan["signal"] == 1 else mt5.ORDER_TYPE_SELL, symbol, volume, price)
            if margin is not None and margin > float(account.equity) * 0.9:
                print("XAUUSD: insufficient margin")
                if args.loop_once:
                    break
                time.sleep(args.poll_seconds)
                continue

            if args.dry_run:
                print(
                    f"XAUUSD DRY signal={plan['signal']} volume={volume:.2f} "
                    f"price={price:.2f} sl={stop:.2f} tp={target:.2f}"
                )
            else:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if plan["signal"] == 1 else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": stop,
                    "tp": target,
                    "deviation": args.deviation,
                    "magic": args.magic,
                    "comment": "XAUUSD-Scalp",
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                result = mt5.order_send(request)
                print(f"XAUUSD order result={result}")

            if args.loop_once:
                break
            time.sleep(args.poll_seconds)
    finally:
        connector.shutdown()


def build_parser():
    parser = argparse.ArgumentParser(description="XAUUSD scalp machine for Exness MT5.")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--strategy-mode", choices=["all", "breakout", "pullback", "reversion"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=0.004)
    parser.add_argument("--commission-round-turn", type=float, default=7.0)
    parser.add_argument("--terminal-path", default=TERMINAL_PATH)
    parser.add_argument("--login", default=os.getenv("EXNESS_LOGIN") or EXNESS_LOGIN)
    parser.add_argument("--password", default=os.getenv("EXNESS_PASSWORD") or EXNESS_PASSWORD)
    parser.add_argument("--server", default=os.getenv("EXNESS_SERVER") or EXNESS_SERVER)
    parser.add_argument("--magic", type=int, default=26072028)
    parser.add_argument("--fixed-lot", type=float, default=0.01)
    parser.add_argument("--lot-mode", choices=["fixed", "risk"], default="fixed")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bars", type=int, default=2500)
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--deviation", type=int, default=25)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "backtest":
        return run_backtest(args)

    if not args.login or not args.password or not args.server:
        print("Missing Exness credentials. Fill EXNESS_LOGIN, EXNESS_PASSWORD, EXNESS_SERVER or pass CLI args.")
        return 1
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
