"""Standalone backtest and walk-forward evaluator for the rules in 2.py.

The simulator uses M15 CSV data, derives an H1 EMA trend filter, enters on the
next bar, models spread and round-turn commission, and reports trade, risk,
drawdown, daily, weekly, and walk-forward statistics.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SYMBOL_DEFAULTS = {
    "EURUSDm": {"min": 4, "recover": 5, "atr_mult": 1.2, "risk": 2.0, "value": 100000.0},
    "GBPUSDm": {"min": 4, "recover": 5, "atr_mult": 1.3, "risk": 2.0, "value": 100000.0},
    "XAUUSDm": {"min": 4, "recover": 5, "atr_mult": 1.5, "risk": 3.0, "value": 100.0},
}
DEFAULT_SYMBOLS = tuple(SYMBOL_DEFAULTS)
RR_RATIO = 2.0
SESSION_HOURS = set(range(9, 22))
H1_EMA = 50
WIN_STREAK_BOOST = 1.5
MAX_WIN_BOOST = 4.0
PAUSE_LOSSES = 2
PAUSE_MINUTES = 15


@dataclass
class Position:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    volume: float
    risk_usd: float
    score: int
    atr: float
    entry_index: int


def parse_mapping(values: list[str], defaults: dict[str, float]) -> dict[str, float]:
    result = dict(defaults)
    for item in values:
        key, value = item.split("=", 1)
        matching_key = next((candidate for candidate in defaults if candidate.upper() == key.upper()), key)
        result[matching_key] = float(value)
    return result


def load_m15(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_column = next((name for name in ("time", "datetime", "date", "timestamp") if name in frame), None)
    if time_column is None:
        raise ValueError(f"{path}: expected time, datetime, date, or timestamp column")
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    frame = frame.set_index(time_column).sort_index()
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "tick_volume" not in frame:
        frame["tick_volume"] = frame.get("volume", 0.0)
    return frame[["open", "high", "low", "close", "tick_volume"]].dropna()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"]
    volume = data["tick_volume"]
    data["fast_ema"] = close.ewm(span=9, adjust=False).mean()
    data["slow_ema"] = close.ewm(span=21, adjust=False).mean()
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    data["macd_hist"] = macd_fast - macd_slow
    data["macd_hist"] -= data["macd_hist"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    data["rsi"] = 100 - (100 / (1 + gain / loss))
    data["bb_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    data["bb_upper"] = data["bb_mid"] + 2 * std
    data["bb_lower"] = data["bb_mid"] - 2 * std
    true_range = pd.concat(
        [data["high"] - data["low"], (data["high"] - close.shift()).abs(), (data["low"] - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    data["vol_avg"] = volume.rolling(20).mean()

    h1 = data.resample("1h").agg({"close": "last"}).dropna()
    h1["ema"] = h1["close"].ewm(span=H1_EMA, adjust=False).mean()
    h1["trend"] = np.where(h1["close"] > h1["ema"], "UP", "DOWN")
    data["h1_trend"] = h1["trend"].shift(1).reindex(data.index, method="ffill")
    return data


def pattern(row: pd.Series, previous: pd.Series, bullish: bool) -> bool:
    body = abs(row.close - row.open)
    if body == 0:
        pin = False
    elif bullish:
        pin = min(row.close, row.open) - row.low >= 2 * body and row.high - max(row.close, row.open) <= body
    else:
        pin = row.high - max(row.close, row.open) >= 2 * body and min(row.close, row.open) - row.low <= body
    if bullish:
        engulf = previous.close < previous.open and row.close > row.open and row.open < previous.close and row.close > previous.open
    else:
        engulf = previous.close > previous.open and row.close < row.open and row.open > previous.close and row.close < previous.open
    return bool(pin or engulf)


def signal_at(data: pd.DataFrame, index: int, symbol: str, state: dict) -> tuple[Optional[str], int, float]:
    if index < 30:
        return None, 0, np.nan
    row, previous = data.iloc[index], data.iloc[index - 1]
    if pd.isna(row.atr) or pd.isna(row.bb_upper) or row.name.hour not in SESSION_HOURS:
        return None, 0, row.atr
    cfg = SYMBOL_DEFAULTS[symbol]
    threshold = cfg["recover"] if state["losses"] > 0 else cfg["min"]
    volume_ok = row.tick_volume >= row.vol_avg * 1.1 if row.vol_avg > 0 else False
    buy = {
        "ema_cross": previous.fast_ema <= previous.slow_ema and row.fast_ema > row.slow_ema,
        "macd": row.macd_hist > 0 and row.macd_hist > previous.macd_hist,
        "bb": row.close <= row.bb_lower * 1.001 and row.rsi < 65,
        "candle": pattern(row, previous, True),
        "volume": volume_ok,
    }
    sell = {
        "ema_cross": previous.fast_ema >= previous.slow_ema and row.fast_ema < row.slow_ema,
        "macd": row.macd_hist < 0 and row.macd_hist < previous.macd_hist,
        "bb": row.close >= row.bb_upper * 0.999 and row.rsi > 35,
        "candle": pattern(row, previous, False),
        "volume": volume_ok,
    }
    buy_score = sum(buy.values())
    sell_score = sum(sell.values())
    if row.h1_trend == "DOWN":
        buy_score = 0
    elif row.h1_trend == "UP":
        sell_score = 0
    if buy_score >= threshold and buy_score >= sell_score:
        return "BUY", buy_score, float(row.atr)
    if sell_score >= threshold and sell_score > buy_score:
        return "SELL", sell_score, float(row.atr)
    return None, max(buy_score, sell_score), float(row.atr)


def close_position(position: Position, bar: pd.Series, spread: float, commission_per_lot: dict[str, float]) -> Optional[dict]:
    if position.direction == "BUY":
        hit_sl = bar.low <= position.stop
        hit_tp = bar.high >= position.target
        if not hit_sl and not hit_tp:
            return None
        exit_price = position.stop if hit_sl else position.target
        reason = "SL" if hit_sl else "TP"
        gross = (exit_price - position.entry_price) * position.volume * SYMBOL_DEFAULTS[position.symbol]["value"]
    else:
        hit_sl = bar.high >= position.stop
        hit_tp = bar.low <= position.target
        if not hit_sl and not hit_tp:
            return None
        exit_price = position.stop if hit_sl else position.target
        reason = "SL" if hit_sl else "TP"
        gross = (position.entry_price - exit_price) * position.volume * SYMBOL_DEFAULTS[position.symbol]["value"]
    costs = commission_per_lot.get(position.symbol, 0.0) * position.volume + spread * position.volume * SYMBOL_DEFAULTS[position.symbol]["value"]
    return {"exit_time": bar.name, "exit_price": exit_price, "reason": reason, "gross": gross, "costs": costs, "pnl": gross - costs}


def run_backtest(frames: dict[str, pd.DataFrame], initial_balance: float, spread_points: dict[str, float], commission: dict[str, float], point: dict[str, float], start=None, end=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {symbol: data.loc[start:end] for symbol, data in frames.items()}
    timeline = sorted(set().union(*(data.index for data in frames.values())))
    states = {symbol: {"losses": 0, "wins": 0, "win_streak": 0, "last_loss": None} for symbol in frames}
    positions: dict[str, Position] = {}
    pending: list[tuple[str, str, int, float]] = []
    balance = initial_balance
    equity_curve = []
    trades = []

    for timestamp in timeline:
        for symbol, direction, score, atr in pending:
            if symbol in positions or timestamp not in frames[symbol].index:
                continue
            bar = frames[symbol].loc[timestamp]
            cfg = SYMBOL_DEFAULTS[symbol]
            spread = spread_points.get(symbol, 0.0) * point.get(symbol, 0.00001)
            entry = float(bar.open) + spread / 2 if direction == "BUY" else float(bar.open) - spread / 2
            distance = cfg["atr_mult"] * atr
            stop = entry - distance if direction == "BUY" else entry + distance
            target = entry + distance * RR_RATIO if direction == "BUY" else entry - distance * RR_RATIO
            state = states[symbol]
            win_mult = min(WIN_STREAK_BOOST ** max(0, state["win_streak"] - 1), MAX_WIN_BOOST)
            effective_risk = min(cfg["risk"] * win_mult, cfg["risk"] * 5)
            risk_usd = balance * effective_risk / 100
            value_per_price = cfg["value"]
            volume = risk_usd / max(distance * value_per_price, 1e-12)
            positions[symbol] = Position(symbol, direction, timestamp, entry, stop, target, volume, risk_usd, score, atr, 0)
        pending = []

        for symbol, position in list(positions.items()):
            if timestamp not in frames[symbol].index:
                continue
            outcome = close_position(position, frames[symbol].loc[timestamp], spread_points.get(symbol, 0.0) * point.get(symbol, 0.00001), commission)
            if outcome is None:
                continue
            balance += outcome["pnl"]
            state = states[symbol]
            if outcome["pnl"] > 0:
                state["wins"] += 1
                state["win_streak"] += 1
                state["losses"] = 0
            else:
                state["losses"] += 1
                state["win_streak"] = 0
                state["last_loss"] = timestamp
            trades.append({**asdict(position), **outcome, "balance_after": balance, "effective_risk_usd": position.risk_usd})
            del positions[symbol]

        floating = 0.0
        for symbol, position in positions.items():
            bar = frames[symbol].loc[timestamp]
            mark = float(bar.close)
            direction = 1 if position.direction == "BUY" else -1
            floating += (mark - position.entry_price) * direction * position.volume * SYMBOL_DEFAULTS[symbol]["value"]
        equity_curve.append({"time": timestamp, "balance": balance, "equity": balance + floating})

        for symbol, data in frames.items():
            if timestamp not in data.index or symbol in positions:
                continue
            state = states[symbol]
            if state["losses"] >= PAUSE_LOSSES and state["last_loss"] is not None:
                minutes = (timestamp - state["last_loss"]).total_seconds() / 60
                if minutes < PAUSE_MINUTES:
                    continue
            direction, score, atr = signal_at(data, data.index.get_loc(timestamp), symbol, state)
            if direction is not None:
                pending.append((symbol, direction, score, atr))

        if any(symbol == "XAUUSDm" for symbol, *_ in pending):
            pending = [item for item in pending if item[0] == "XAUUSDm" or item[0] not in {"EURUSDm", "GBPUSDm"}]

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_balance: float) -> dict:
    if equity.empty:
        return {"trades": 0}
    curve = equity.set_index("time")["equity"]
    drawdown = curve - curve.cummax()
    daily_losses = []
    for _, day in curve.groupby(curve.index.normalize()):
        day_start = initial_balance if not daily_losses else previous_day_end
        daily_losses.append(float(day.min() - day_start))
        previous_day_end = float(day.iloc[-1])
    weekly = curve.resample("1W").last().diff().dropna()
    wins = trades.loc[trades.pnl > 0, "pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades.pnl <= 0, "pnl"] if not trades.empty else pd.Series(dtype=float)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    max_daily_loss = min(daily_losses) if daily_losses else 0.0
    max_loss_from_initial = float(curve.min() - initial_balance)
    daily_limit = initial_balance * 0.05
    total_limit = initial_balance * 0.10
    return {
        "start_balance": float(equity.iloc[0].equity),
        "end_balance": float(equity.iloc[-1].equity),
        "net_pnl": float(equity.iloc[-1].equity - equity.iloc[0].equity),
        "return_pct": float((equity.iloc[-1].equity / equity.iloc[0].equity - 1) * 100),
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": float(len(wins) / len(trades) * 100) if len(trades) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "avg_trade": float(trades.pnl.mean()) if len(trades) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_pct": float((drawdown / curve.cummax() * 100).min()),
        "max_loss_from_initial_usd": max_loss_from_initial,
        "max_loss_from_initial_pct": float(max_loss_from_initial / initial_balance * 100),
        "max_daily_loss_usd": float(max_daily_loss),
        "max_daily_loss_pct": float(max_daily_loss / initial_balance * 100),
        "max_weekly_loss_usd": float(weekly.min()) if len(weekly) else 0.0,
        "prop_limits": {
            "daily_limit_pct": 5.0,
            "daily_limit_usd": daily_limit,
            "total_limit_pct": 10.0,
            "total_limit_usd": total_limit,
            "daily_limit_pass": max_daily_loss >= -daily_limit,
            "total_limit_pass": max_loss_from_initial >= -total_limit,
            "overall_pass": max_daily_loss >= -daily_limit and max_loss_from_initial >= -total_limit,
        },
        "weekly_pnl": {str(index.date()): float(value) for index, value in weekly.items()},
        "by_symbol": {
            symbol: {
                "trades": int(len(group)),
                "pnl": float(group.pnl.sum()),
                "win_rate_pct": float((group.pnl > 0).mean() * 100),
            }
            for symbol, group in trades.groupby("symbol")
        } if len(trades) else {},
    }


def download_m15_history(symbols: list[str], output_dir: Path, days: int) -> dict[str, Path]:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 is required for automatic downloads; run this on the Windows VPS") from exc

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
        paths = {}
        bars = days * 24 * 4 + 200
        for symbol in symbols:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"MT5 could not select {symbol}")
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, bars)
            if rates is None or len(rates) < 100:
                raise RuntimeError(f"MT5 returned insufficient M15 history for {symbol}: {mt5.last_error()}")
            frame = pd.DataFrame(rates)
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            path = output_dir / f"{symbol}_M15.csv"
            frame.to_csv(path, index=False)
            paths[symbol] = path
            print(f"Downloaded {len(frame):,} M15 bars: {symbol} -> {path}")
        return paths
    finally:
        mt5.shutdown()


def resolve_data(args, symbols: list[str], output: Path) -> dict[str, Path]:
    if args.data:
        resolved = {}
        for item in args.data:
            raw_symbol, path = item.split("=", 1)
            symbol = next((candidate for candidate in SYMBOL_DEFAULTS if candidate.upper() == raw_symbol.upper()), raw_symbol)
            resolved[symbol] = Path(path)
        return resolved
    cache_dir = output / "data"
    cached = {symbol: cache_dir / f"{symbol}_M15.csv" for symbol in symbols}
    if not args.force_download and all(path.exists() for path in cached.values()):
        print(f"Using cached M15 history in {cache_dir}")
        return cached
    return download_m15_history(symbols, cache_dir, args.history_days)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    symbols = list(SYMBOL_DEFAULTS) if not args.data else [item.split("=", 1)[0] for item in args.data]
    symbols = [next((candidate for candidate in SYMBOL_DEFAULTS if candidate.upper() == symbol.upper()), symbol) for symbol in symbols]
    input_map = resolve_data(args, symbols, output)
    symbols = list(input_map)
    unknown = set(symbols) - set(SYMBOL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported symbols: {sorted(unknown)}")
    frames = {symbol: add_indicators(load_m15(path)) for symbol, path in input_map.items()}
    spreads = parse_mapping(args.spread_points, {symbol: 0.0 for symbol in symbols})
    points = parse_mapping(args.point, {symbol: (0.01 if symbol.startswith("XAU") else 0.00001) for symbol in symbols})
    commissions = parse_mapping(args.commission_per_lot, {"EURUSDm": 2.5, "GBPUSDm": 2.5, "XAUUSDm": 0.0})
    trades, equity = run_backtest(frames, args.initial_balance, spreads, commissions, points)
    report = {"mode": "backtest", "settings": vars(args), "metrics": metrics(trades, equity, args.initial_balance)}
    trades.to_csv(output / "vx2_trades.csv", index=False)
    equity.to_csv(output / "vx2_equity.csv", index=False)
    (output / "vx2_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, default=str))

    if args.walkforward_days:
        all_times = sorted(set().union(*(data.index for data in frames.values())))
        start, end = all_times[0], all_times[-1]
        windows = []
        cursor = start + pd.Timedelta(days=args.train_days)
        while cursor < end:
            test_end = min(cursor + pd.Timedelta(days=args.walkforward_days), end)
            test_trades, test_equity = run_backtest(frames, args.initial_balance, spreads, commissions, points, cursor, test_end)
            windows.append({"test_start": str(cursor), "test_end": str(test_end), "metrics": metrics(test_trades, test_equity, args.initial_balance)})
            cursor += pd.Timedelta(days=args.walkforward_days)
        (output / "vx2_walkforward.json").write_text(json.dumps(windows, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"walkforward_windows": windows}, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest/walk-forward evaluator for 2.py")
    parser.add_argument("--data", action="append", default=[], metavar="SYMBOL=CSV", help="optional M15 CSV override")
    parser.add_argument("--initial-balance", type=float, default=2000.0)
    parser.add_argument("--spread-points", action="append", default=[], metavar="SYMBOL=POINTS")
    parser.add_argument("--point", action="append", default=[], metavar="SYMBOL=PRICE_POINT")
    parser.add_argument("--commission-per-lot", action="append", default=[], metavar="SYMBOL=USD", help="round-turn commission override")
    parser.add_argument("--output", default="results/vx2_auto")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--walkforward-days", type=int, default=30, help="walk-forward test-window length")
    run(parser.parse_args())
