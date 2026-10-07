"""Live-equivalent backtest for the strategies routed by run_project.py live.

Run on the Windows MT5 machine. It downloads M5 history, reuses the project
strategy router, applies live risk overrides and position limits, models
commission, and reports FTMO daily/total equity-limit results.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bootstrap import add_project_root
add_project_root()

from engine.features import FeatureEngine
from engine.risk_manager import RiskManager
from strategy_router import StrategyRouter


SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "XAUUSD"]
RISK_BY_SYMBOL = {"EURUSD": 0.01, "GBPUSD": 0.01, "USDJPY": 0.005, "AUDUSD": 0.005, "USDCHF": 0.005, "XAUUSD": 0.005}
TP_ATR_OVERRIDES = {"EURUSD": 4.0, "GBPUSD": 4.5}
MAX_OPEN_POSITIONS = 3
MAX_CONSECUTIVE_LOSSES = 2
LOSS_PAUSE_HOURS = 3
DAILY_LOSS_PCT = 0.05
TOTAL_LOSS_PCT = 0.10
COMMISSION_PER_LOT = 5.0


@dataclass
class Position:
    symbol: str
    broker_symbol: str
    direction: int
    opened_at: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    volume: float
    risk_usd: float
    signal_score: object
    score_components: dict
    strategy: str
    entry_index: int
    mfe_usd: float = 0.0


class HistoricalBroker:
    def __init__(self, frames, specs):
        self.frames = frames
        self.specs = specs
        self.current_symbol = None
        self.current_price = 0.0
        self.mt5 = SimpleNamespace(TIMEFRAME_M15="M15")

    def set_context(self, symbol, price):
        self.current_symbol = symbol
        self.current_price = float(price)

    def symbol_info(self, symbol):
        return self.specs[symbol]

    def symbol_tick(self, symbol):
        info = self.specs[symbol]
        spread = float(info.backtest_spread_price)
        return SimpleNamespace(ask=self.current_price + spread / 2, bid=self.current_price - spread / 2)

    def rates_copy(self, symbol, timeframe, count):
        frame = self.frames[symbol][["open", "high", "low", "close", "tick_volume"]].resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna().tail(count)
        return frame.to_dict("records")

    def normalize_volume(self, symbol, volume):
        info = self.specs[symbol]
        step = float(info.volume_step)
        volume = min(float(info.volume_max), max(0.0, float(volume)))
        return round(np.floor(volume / step) * step, 8)

    def order_calc_margin(self, direction, symbol, volume, price):
        return 0.0


def download_history(symbols, output_dir, days):
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("Install MetaTrader5 on the Windows VPS before running this script") from exc
    terminal_path = os.getenv("VX_MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if not mt5.initialize(path=terminal_path):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        login = int(os.getenv("VX_MT5_LOGIN", "0"))
        password = os.getenv("VX_MT5_PASSWORD", "")
        server = os.getenv("VX_MT5_SERVER", "")
        if login and password and server and not mt5.login(login, password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        output_dir.mkdir(parents=True, exist_ok=True)
        frames, specs, broker_symbols = {}, {}, {}
        bars = days * 24 * 12 + 300
        for canonical in symbols:
            matches = mt5.symbols_get(group=f"*{canonical}*") or []
            exact = next((item for item in matches if item.name.upper() == canonical), None)
            selected = exact or next((item for item in matches if item.name.upper().startswith(canonical)), None)
            if selected is None or not mt5.symbol_select(selected.name, True):
                raise RuntimeError(f"Could not resolve/select broker symbol for {canonical}")
            broker_symbols[canonical] = selected.name
            chunks = []
            chunk_size = 5000
            start_pos = 1
            while sum(len(chunk) for chunk in chunks) < bars:
                chunk = mt5.copy_rates_from_pos(
                    selected.name, mt5.TIMEFRAME_M5, start_pos, chunk_size
                )
                if chunk is None or len(chunk) == 0:
                    break
                chunks.append(chunk)
                start_pos += len(chunk)
                if len(chunk) < chunk_size:
                    break
            rates = np.concatenate(chunks) if chunks else None
            if rates is None or len(rates) < 500:
                raise RuntimeError(f"Insufficient M5 history for {selected.name}: {mt5.last_error()}")
            frame = pd.DataFrame(rates)
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame = frame.set_index("time").sort_index()
            frame = frame.tail(bars)
            frame["spread"] = frame.get("spread", 0.0).fillna(0.0)
            frame["real_volume"] = frame.get("real_volume", 0.0).fillna(0.0)
            frames[canonical] = FeatureEngine().add_features(frame)
            info = mt5.symbol_info(selected.name)
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.00001))
            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            value_per_price = tick_value / tick_size if tick_value and tick_size else float(getattr(info, "trade_contract_size", 100000.0) or 100000.0)
            specs[canonical] = SimpleNamespace(
                point=float(getattr(info, "point", tick_size) or tick_size),
                digits=int(getattr(info, "digits", 5) or 5),
                trade_tick_size=tick_size,
                trade_tick_value=tick_value or value_per_price * tick_size,
                trade_contract_size=float(getattr(info, "trade_contract_size", 100000.0) or 100000.0),
                trade_stops_level=float(getattr(info, "trade_stops_level", 0.0) or 0.0),
                volume_min=float(getattr(info, "volume_min", 0.01) or 0.01),
                volume_max=float(getattr(info, "volume_max", 100.0) or 100.0),
                volume_step=float(getattr(info, "volume_step", 0.01) or 0.01),
                value_per_price=value_per_price,
                backtest_spread_price=float(frame["spread"].median()) * float(getattr(info, "point", tick_size) or tick_size),
            )
            frame.to_csv(output_dir / f"{selected.name}_M5.csv")
            print(f"Downloaded {len(frame):,} M5 bars: {selected.name}")
        return frames, specs, broker_symbols
    finally:
        mt5.shutdown()


def load_cached_history(symbols, data_dir):
    frames, specs, broker_symbols = {}, {}, {}
    for symbol in symbols:
        candidates = list(data_dir.glob(f"*{symbol}*_M5.csv"))
        if not candidates:
            raise FileNotFoundError(f"No cached M5 data for {symbol} in {data_dir}")
        path = candidates[0]
        frame = pd.read_csv(path)
        time_col = "time" if "time" in frame else "datetime"
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
        frame = frame.set_index(time_col).sort_index()
        if "tick_volume" not in frame:
            frame["tick_volume"] = frame.get("volume", 0.0)
        if "spread" not in frame:
            frame["spread"] = 0.0
        if "real_volume" not in frame:
            frame["real_volume"] = 0.0
        frames[symbol] = FeatureEngine().add_features(frame)
        point = 0.01 if symbol == "XAUUSD" else 0.00001
        value_per_price = 100.0 if symbol == "XAUUSD" else 100000.0
        specs[symbol] = SimpleNamespace(
            point=point, digits=2 if symbol == "XAUUSD" else 5, trade_tick_size=point,
            trade_tick_value=value_per_price * point, trade_contract_size=value_per_price,
            trade_stops_level=0.0, volume_min=0.01, volume_max=100.0, volume_step=0.01,
            value_per_price=value_per_price, backtest_spread_price=float(frame["spread"].median()) * point,
        )
        broker_symbols[symbol] = symbol
    return frames, specs, broker_symbols


def spread_points(frame, index, info):
    value = float(frame["spread"].iloc[index]) if "spread" in frame else 0.0
    if value <= 0:
        return 0.0
    return value


def calculate_volume(symbol, equity, entry, stop, specs):
    distance = abs(entry - stop)
    if distance <= 0:
        return 0.0
    risk_usd = equity * RISK_BY_SYMBOL[symbol]
    raw = risk_usd / (distance * specs[symbol].value_per_price)
    return max(0.0, np.floor(raw / specs[symbol].volume_step) * specs[symbol].volume_step)


def pnl_for_move(position, exit_price, specs):
    sign = 1 if position.direction == 1 else -1
    gross = (exit_price - position.entry_price) * sign * position.volume * specs[position.symbol].value_per_price
    spread_cost = specs[position.symbol].backtest_spread_price * position.volume * specs[position.symbol].value_per_price
    return gross - COMMISSION_PER_LOT * position.volume - spread_cost


def close_position(position, exit_time, exit_price, reason, balance, specs):
    pnl = pnl_for_move(position, exit_price, specs)
    return {
        **asdict(position), "exit_time": exit_time, "exit_price": exit_price,
        "reason": reason, "pnl": pnl, "balance_before": balance, "balance_after": balance + pnl,
    }


def m5_candles_held(position, timestamp):
    return max(0, int((timestamp - position.opened_at).total_seconds() // 300))


def build_plan(symbol, frame, index, signal, score, components, equity, broker, strategy, risk):
    window = frame.iloc[max(0, index - 600): index + 1]
    broker.set_context(symbol, float(frame["close"].iloc[index]))
    price = float(frame["close"].iloc[index])
    analysis = getattr(strategy, "analyze_trade", None)
    if callable(analysis):
        plan = analysis(window, broker=broker, symbol=symbol, equity=equity, signal=signal, price=price)
    else:
        atr = float(frame["atr"].iloc[index])
        if not np.isfinite(atr) or atr <= 0:
            return None
        stop, target = RiskManager().calculate_sl_tp(signal, price, atr, tp_atr=TP_ATR_OVERRIDES.get(symbol, 5.0))
        plan = {"signal": signal, "price": price, "stop": stop, "target": target}
    if not plan:
        return None
    order_signal = int(plan["signal"])
    entry = float(plan["price"])
    stop = float(plan["stop"])
    target = float(plan["target"])
    volume = calculate_volume(symbol, equity, entry, stop, broker.specs)
    if volume <= 0:
        return None
    return Position(symbol, symbol, order_signal, frame.index[index], entry, stop, target, volume, equity * risk, score, components, strategy.__class__.__name__, index)


def run_simulation(frames, specs, initial_balance, start=None, end=None):
    router = StrategyRouter()
    broker = HistoricalBroker(frames, specs)
    frames = {symbol: frame.loc[start:end] for symbol, frame in frames.items()}
    timeline = sorted(set().union(*(frame.index for frame in frames.values())))
    signals = {}
    for symbol, frame in frames.items():
        strategy = router.get_strategy(symbol)
        signals[symbol] = strategy.generate_signals(frame)

    positions = {}
    pending = {}
    balance = float(initial_balance)
    peak_equity = balance
    daily_base = balance
    current_prague_day = None
    consecutive_losses = 0
    pause_until = None
    trades = []
    equity_rows = []

    for timestamp in timeline:
        prague_day = timestamp.tz_convert("Europe/Prague").date()
        if prague_day != current_prague_day:
            current_prague_day = prague_day
            daily_base = balance

        for symbol, position in list(positions.items()):
            if timestamp not in frames[symbol].index:
                continue
            bar = frames[symbol].loc[timestamp]
            floating = pnl_for_move(position, float(bar.close), specs) + COMMISSION_PER_LOT * position.volume
            position.mfe_usd = max(position.mfe_usd, floating)
            reason = None
            exit_price = None
            if position.direction == 1:
                if bar.low <= position.stop:
                    reason, exit_price = "SL", position.stop
                elif bar.high >= position.target:
                    reason, exit_price = "TP", position.target
            else:
                if bar.high >= position.stop:
                    reason, exit_price = "SL", position.stop
                elif bar.low <= position.target:
                    reason, exit_price = "TP", position.target
            candidate_signal = int(signals[symbol].iloc[frames[symbol].index.get_loc(timestamp)])
            if reason is None and candidate_signal == -position.direction:
                reason, exit_price = "setup_invalidation", float(bar.close)
            initial_risk = abs(position.entry_price - position.stop) * position.volume * specs[symbol].value_per_price
            if reason is None and m5_candles_held(position, timestamp) >= 12 and (floating <= 0 or position.mfe_usd < initial_risk * 0.25):
                reason, exit_price = "time_stop", float(bar.close)
            if reason is not None:
                result = close_position(position, timestamp, exit_price, reason, balance, specs)
                balance = result["balance_after"]
                trades.append(result)
                del positions[symbol]
                if result["pnl"] < 0:
                    consecutive_losses += 1
                    pause_until = timestamp + pd.Timedelta(hours=LOSS_PAUSE_HOURS) if consecutive_losses >= MAX_CONSECUTIVE_LOSSES else pause_until
                else:
                    consecutive_losses = 0
                    pause_until = None

        floating_total = sum(pnl_for_move(position, float(frames[symbol].loc[timestamp].close), specs) + COMMISSION_PER_LOT * position.volume for symbol, position in positions.items() if timestamp in frames[symbol].index)
        equity = balance + floating_total
        peak_equity = max(peak_equity, equity)
        daily_limit = daily_base - initial_balance * DAILY_LOSS_PCT
        total_limit = initial_balance * (1 - TOTAL_LOSS_PCT)
        if equity <= daily_limit or equity <= total_limit:
            for symbol, position in list(positions.items()):
                if timestamp in frames[symbol].index:
                    result = close_position(position, timestamp, float(frames[symbol].loc[timestamp].close), "FTMO_limit", balance, specs)
                    balance = result["balance_after"]
                    trades.append(result)
                    del positions[symbol]
            equity = balance
        equity_rows.append({"time": timestamp, "balance": balance, "equity": equity, "daily_limit": daily_limit, "total_limit": total_limit, "peak_equity": peak_equity})

        for symbol, frame in frames.items():
            index = frame.index.get_loc(timestamp) if timestamp in frame.index else None
            if index is None or index + 1 >= len(frame) or symbol in positions or timestamp in pending:
                continue
            if len(positions) >= MAX_OPEN_POSITIONS or (pause_until is not None and timestamp < pause_until):
                continue
            if timestamp.tz_convert("Africa/Accra").weekday() in (0, 1, 2, 3, 4) and 21 <= timestamp.tz_convert("Africa/Accra").hour < 23:
                continue
            if equity <= daily_limit or equity <= total_limit:
                continue
            signal = int(signals[symbol].iloc[index])
            if signal == 0:
                continue
            strategy = router.get_strategy(symbol)
            score = None
            components = {}
            evaluator = getattr(strategy, "evaluate_latest", None)
            if callable(evaluator):
                _, score, components = evaluator(frame.iloc[max(0, index - 600): index + 1], spread_points=spread_points(frame, index, specs[symbol]))
            plan = build_plan(symbol, frame, index, signal, score, components, equity, broker, strategy, RISK_BY_SYMBOL[symbol])
            if plan is not None:
                pending[symbol] = (index + 1, plan)

        for symbol, (entry_index, plan) in list(pending.items()):
            if entry_index == (frames[symbol].index.get_loc(timestamp) if timestamp in frames[symbol].index else -1):
                if symbol not in positions and len(positions) < MAX_OPEN_POSITIONS:
                    bar = frames[symbol].iloc[entry_index]
                    plan.entry_price = float(bar.open)
                    positions[symbol] = plan
                del pending[symbol]

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)


def metrics(trades, equity, initial_balance):
    if equity.empty:
        return {"trades": 0}
    curve = equity.set_index("time")["equity"]
    dd = curve - curve.cummax()
    day_key = equity["time"].dt.normalize()
    daily_min = equity.groupby(day_key)["equity"].min()
    daily_start = equity.groupby(day_key)["equity"].first()
    daily = daily_min - daily_start
    wins = trades[trades.pnl > 0] if not trades.empty else trades
    losses = trades[trades.pnl <= 0] if not trades.empty else trades
    gross_loss = float(-losses.pnl.sum()) if not losses.empty else 0.0
    gross_profit = float(wins.pnl.sum()) if not wins.empty else 0.0
    max_initial_loss = float(curve.min() - initial_balance)
    return {
        "start_balance": initial_balance,
        "end_balance": float(curve.iloc[-1]),
        "net_pnl": float(curve.iloc[-1] - initial_balance),
        "return_pct": float((curve.iloc[-1] / initial_balance - 1) * 100),
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": float(len(wins) / len(trades) * 100) if len(trades) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "max_peak_to_trough_usd": float(dd.min()),
        "max_peak_to_trough_pct": float((dd / curve.cummax() * 100).min()),
        "max_initial_loss_usd": max_initial_loss,
        "max_initial_loss_pct": float(max_initial_loss / initial_balance * 100),
        "max_daily_loss_usd": float(daily.min()),
        "max_daily_loss_pct": float(daily.min() / initial_balance * 100),
        "ftmo_daily_limit_usd": initial_balance * DAILY_LOSS_PCT,
        "ftmo_total_limit_usd": initial_balance * TOTAL_LOSS_PCT,
        "ftmo_daily_pass": bool(daily.min() >= -initial_balance * DAILY_LOSS_PCT),
        "ftmo_total_pass": bool(max_initial_loss >= -initial_balance * TOTAL_LOSS_PCT),
        "ftmo_overall_pass": bool(daily.min() >= -initial_balance * DAILY_LOSS_PCT and max_initial_loss >= -initial_balance * TOTAL_LOSS_PCT),
        "by_symbol": {symbol: {"trades": int(len(group)), "pnl": float(group.pnl.sum()), "win_rate_pct": float((group.pnl > 0).mean() * 100)} for symbol, group in trades.groupby("symbol")} if not trades.empty else {},
    }


def walkforward(frames, specs, initial_balance, window_days):
    """Evaluate consecutive historical windows with a fresh account per window."""
    timeline = sorted(set().union(*(frame.index for frame in frames.values())))
    if not timeline:
        return []
    start = timeline[0]
    end = timeline[-1]
    windows = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + pd.Timedelta(days=window_days), end)
        window_frames = {
            symbol: frame.loc[cursor:window_end]
            for symbol, frame in frames.items()
            if not frame.loc[cursor:window_end].empty
        }
        if window_frames:
            trades, equity = run_simulation(window_frames, specs, initial_balance=initial_balance)
            window_metrics = metrics(trades, equity, initial_balance)
            windows.append({
                "start": cursor,
                "end": window_end,
                **window_metrics,
            })
        cursor = window_end + pd.Timedelta(minutes=5)
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--data-dir", default="results/live_project_backtest/data")
    parser.add_argument("--output", default="results/live_project_backtest")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--walkforward-days", type=int, default=30)
    args = parser.parse_args()
    output = Path(args.output)
    data_dir = Path(args.data_dir)
    cached_history = all(any(data_dir.glob(f"*{symbol}*_M5.csv")) for symbol in SYMBOLS)
    if args.force_download or not cached_history:
        frames, specs, broker_symbols = download_history(SYMBOLS, data_dir, args.history_days)
    else:
        frames, specs, broker_symbols = load_cached_history(SYMBOLS, data_dir)
    trades, equity = run_simulation(frames, specs, args.initial_balance)
    walkforward_windows = walkforward(frames, specs, args.initial_balance, args.walkforward_days)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trades.csv", index=False)
    equity.to_csv(output / "equity.csv", index=False)
    report = {
        "settings": {
            **vars(args),
            "symbols": SYMBOLS,
            "risk_by_symbol": RISK_BY_SYMBOL,
            "tp_atr_overrides": TP_ATR_OVERRIDES,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "max_consecutive_losses": MAX_CONSECUTIVE_LOSSES,
            "loss_pause_hours": LOSS_PAUSE_HOURS,
            "daily_loss_pct": DAILY_LOSS_PCT,
            "total_loss_pct": TOTAL_LOSS_PCT,
            "commission_per_lot_round_turn": COMMISSION_PER_LOT,
        },
        "broker_symbols": broker_symbols,
        "metrics": metrics(trades, equity, args.initial_balance),
        "walkforward": {
            "window_days": args.walkforward_days,
            "windows": walkforward_windows,
        },
        "limitations": [
            "Historical news blackout is not modeled without a timestamped economic-calendar archive.",
            "Spread uses MT5 historical spread fields; slippage is not modeled separately.",
            "Walk-forward windows reset the simulated account at each window; they are diagnostic, not a continuous account curve.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, default=str))


if __name__ == "__main__":
    main()
