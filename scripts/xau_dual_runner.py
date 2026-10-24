"""Run the current XAUUSD strategy and trailing breakout side-by-side.

This is intentionally separate from the multi-symbol runner.  The two strategy
sleeves use different MT5 magic numbers, share a global position cap, and only
manage positions carrying their own magic number.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap import add_project_root

add_project_root()

from engine.features import FeatureEngine
from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError
from strategies.ftmo_xauusd import FTMOXAUUSD


# These IDs are unique to this dual runner and deliberately do not reuse the
# standard live runner's magic number.
CURRENT_MAGIC = 26072601
TRAILING_MAGIC = 26072602
SYMBOL = "XAUUSD"


def true_range(frame):
    previous = frame["close"].shift(1)
    return pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)


def adx(frame, period=14):
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = true_range(frame)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_breakout_features(frame):
    frame = frame.copy()
    frame["atr20"] = true_range(frame).rolling(20, min_periods=20).mean()
    frame["prior_high"] = frame["high"].rolling(20, min_periods=20).max().shift(1)
    frame["prior_low"] = frame["low"].rolling(20, min_periods=20).min().shift(1)
    frame["prior_low10"] = frame["low"].rolling(10, min_periods=10).min().shift(1)
    frame["prior_high10"] = frame["high"].rolling(10, min_periods=10).max().shift(1)
    frame["body_ratio"] = (frame["close"] - frame["open"]).abs() / (frame["high"] - frame["low"]).replace(0, np.nan)
    for label, rule in (("15min", "15min"), ("30min", "30min")):
        higher = frame[["open", "high", "low", "close", "tick_volume"]].resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        higher["atr"] = true_range(higher).rolling(14, min_periods=14).mean()
        higher["atr_median"] = higher["atr"].rolling(50, min_periods=20).median()
        higher["atr_ratio"] = higher["atr"] / higher["atr_median"]
        higher["ema20"] = higher["close"].ewm(span=20, adjust=False).mean()
        higher["ema50"] = higher["close"].ewm(span=50, adjust=False).mean()
        higher["ema_slope"] = higher["ema20"].diff(3)
        higher["adx"] = adx(higher)
        for column in ("close", "atr_ratio", "ema20", "ema50", "ema_slope", "adx"):
            frame[f"{label}_{column}"] = higher[column].reindex(frame.index, method="ffill")
    return frame


def load_data(broker, broker_symbol):
    rates = broker.rates_copy(broker_symbol, broker.mt5.TIMEFRAME_M5, 3000)
    if rates is None or len(rates) == 0:
        return None
    frame = pd.DataFrame(rates)
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()
    columns = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    frame = frame[[column for column in columns if column in frame.columns]]
    frame = FeatureEngine().add_features(frame)
    return add_breakout_features(frame)


def trend_aligned(row, direction, adx_min=25.0):
    for prefix in ("15min_", "30min_"):
        values = [row.get(f"{prefix}{name}") for name in ("close", "ema20", "ema50", "ema_slope", "adx", "atr_ratio")]
        if any(pd.isna(value) for value in values):
            return False
        close, ema20, ema50, slope, adx_value, atr_ratio = values
        directional = close > ema20 > ema50 and slope > 0 if direction == 1 else close < ema20 < ema50 and slope < 0
        if not directional or adx_value < adx_min or not 0.75 <= atr_ratio <= 2.5:
            return False
    return True


def trailing_plan(data):
    if len(data) < 2:
        return 0, None
    row = data.iloc[-1]
    if row.name.hour < 7 or row.name.hour >= 21:
        return 0, None
    required = ("atr20", "prior_high", "prior_low", "prior_low10", "prior_high10")
    if any(pd.isna(row.get(name)) for name in required) or float(row["body_ratio"]) < 0.35:
        return 0, None
    direction = 1 if row["close"] > row["prior_high"] and row["close"] > row["open"] and trend_aligned(row, 1) else 0
    if not direction and row["close"] < row["prior_low"] and row["close"] < row["open"] and trend_aligned(row, -1):
        direction = -1
    if not direction:
        return 0, None
    atr = float(row["atr20"])
    entry = float(row["close"])
    structure = float(row["prior_low10"] if direction == 1 else row["prior_high10"])
    stop = structure - 0.15 * atr if direction == 1 else structure + 0.15 * atr
    if direction == 1 and stop >= entry - 0.5 * atr:
        stop = entry - 1.5 * atr
    if direction == -1 and stop <= entry + 0.5 * atr:
        stop = entry + 1.5 * atr
    target = entry + direction * abs(entry - stop) * 2.0
    return direction, {"stop": stop, "target": target, "atr": atr, "signal_time": row.name}


def owned_positions(broker, broker_symbol):
    return [
        position
        for position in broker.positions_get(symbol=broker_symbol)
        if getattr(position, "magic", None) in {CURRENT_MAGIC, TRAILING_MAGIC}
    ]


def strategy_positions(broker, broker_symbol, magic):
    return [position for position in broker.positions_get(symbol=broker_symbol) if getattr(position, "magic", None) == magic]


def calculate_volume(broker, broker_symbol, direction, price, stop, equity, risk_pct):
    info = broker.symbol_info(broker_symbol)
    if info is None:
        return 0.0
    distance = abs(price - stop)
    tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
    if distance <= 0 or tick_value <= 0 or tick_size <= 0:
        return 0.0
    volume = broker.normalize_volume(broker_symbol, equity * risk_pct / (distance * tick_value / tick_size))
    minimum = float(getattr(info, "volume_min", 0.01) or 0.01)
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    while volume >= minimum:
        margin = broker.order_calc_margin(direction, broker_symbol, volume, price)
        if margin is not None and margin <= equity * 0.85:
            return volume
        volume = broker.normalize_volume(broker_symbol, volume - step)
    return 0.0


def manage_trailing(broker, broker_symbol, data, position, state):
    ticket = getattr(position, "ticket", None)
    if ticket is None or len(data) < 2:
        return
    direction = 1 if getattr(position, "type", None) == getattr(broker.mt5, "POSITION_TYPE_BUY", 0) else -1
    entry = float(getattr(position, "price_open", 0.0) or 0.0)
    current_sl = float(getattr(position, "sl", 0.0) or 0.0)
    volume = float(getattr(position, "volume", 0.0) or 0.0)
    info = broker.symbol_info(broker_symbol)
    tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0) if info else 0.0
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0) if info else 0.0
    risk = abs(entry - current_sl) / tick_size * tick_value * volume if tick_size and tick_value else 0.0
    profit = float(getattr(position, "profit", 0.0) or 0.0)
    bar = data.index[-1]
    if risk <= 0 or profit < risk or state.get(ticket) == bar:
        return
    previous = data.iloc[-2]
    atr = float(previous.get("atr20", 0.0) or 0.0)
    if atr <= 0:
        return
    candidate = float(previous["low"] - 0.25 * atr) if direction == 1 else float(previous["high"] + 0.25 * atr)
    if direction == 1 and candidate <= current_sl:
        return
    if direction == -1 and candidate >= current_sl:
        return
    result = broker.modify_position_stops(
        position_ticket=ticket,
        symbol=broker_symbol,
        stop_loss=candidate,
        take_profit=getattr(position, "tp", None),
        comment="QuantFX-XAU-trailing",
        magic=TRAILING_MAGIC,
    )
    if result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE:
        state[ticket] = bar
        print(f"XAUUSD trailing stop moved | ticket={ticket} | stop={candidate:.2f} | profit=${profit:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Concurrent XAUUSD current and trailing breakout runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-open-positions", type=int, default=2)
    parser.add_argument("--risk-current", type=float, default=0.0075)
    parser.add_argument("--risk-trailing", type=float, default=0.0075)
    parser.add_argument("--current-min-score", type=int, default=4)
    parser.add_argument("--mt5-path")
    args = parser.parse_args()
    broker = None
    if not args.dry_run:
        try:
            broker = MT5BrokerAdapter()
            broker.initialize(path=args.mt5_path)
        except MT5UnavailableError as exc:
            raise SystemExit(f"MT5 unavailable: {exc}") from exc
    if broker is None:
        raise SystemExit("--dry-run is reserved for future signal-only mode; use a connected MT5 terminal")

    broker_symbol = broker.resolve_symbol(SYMBOL) or SYMBOL
    current_strategy = FTMOXAUUSD(min_score=args.current_min_score)
    last_signal_bar = {"current": None, "trailing": None}
    trailing_state = {}
    print("XAUUSD dual runner")
    print(f"current magic={CURRENT_MAGIC} risk={args.risk_current:.2%} min_score={args.current_min_score}")
    print(f"trailing magic={TRAILING_MAGIC} risk={args.risk_trailing:.2%} max_positions={args.max_open_positions}")
    while True:
        started = datetime.now(timezone.utc)
        data = load_data(broker, broker_symbol)
        if data is None or data.empty:
            print("XAUUSD: no data")
        else:
            for position in strategy_positions(broker, broker_symbol, TRAILING_MAGIC):
                manage_trailing(broker, broker_symbol, data, position, trailing_state)
            positions = owned_positions(broker, broker_symbol)
            tick = broker.symbol_tick(broker_symbol)
            equity = broker.account_equity() or 0.0
            if tick is not None and equity > 0:
                current_signal = int(current_strategy.generate_signals(data).iloc[-1])
                trailing_signal, trailing = trailing_plan(data)
                candidates = [("current", current_signal, current_strategy, CURRENT_MAGIC, args.risk_current), ("trailing", trailing_signal, None, TRAILING_MAGIC, args.risk_trailing)]
                for name, signal, strategy, magic, risk_pct in candidates:
                    if not signal or len(positions) >= args.max_open_positions or last_signal_bar[name] == data.index[-1]:
                        continue
                    if strategy is not None:
                        price = float(tick.ask if signal == 1 else tick.bid)
                        plan = strategy.analyze_trade(data, broker=broker, symbol=broker_symbol, equity=equity, signal=signal, price=price)
                        if not plan:
                            continue
                        stop, target = float(plan["stop"]), float(plan["target"])
                    else:
                        price = float(tick.ask if signal == 1 else tick.bid)
                        stop, target = float(trailing["stop"]), float(trailing["target"])
                    stop, target = broker.conform_stop_levels(broker_symbol, signal, price, stop, target)
                    if stop is None or target is None:
                        print(f"XAUUSD {name}: invalid broker stop levels")
                        continue
                    volume = calculate_volume(broker, broker_symbol, signal, price, stop, equity, risk_pct)
                    if volume <= 0:
                        print(f"XAUUSD {name}: insufficient margin or invalid volume")
                        continue
                    result = broker.place_order(
                        symbol=broker_symbol,
                        direction=signal,
                        volume=volume,
                        stop_loss=stop,
                        take_profit=target,
                        price=price,
                        comment=f"QuantFX-XAU-{name}",
                        magic=magic,
                    )
                    print(f"XAUUSD {name}: signal={signal} size={volume:.2f} sl={stop:.2f} tp={target:.2f} result={result}")
                    if result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE:
                        last_signal_bar[name] = data.index[-1]
                        positions = owned_positions(broker, broker_symbol)
        if args.loop_once:
            break
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        time.sleep(max(1, args.poll_seconds - int(elapsed)))
    broker.shutdown()


if __name__ == "__main__":
    main()
