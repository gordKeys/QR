"""Download MT5 history and run a cost-aware research backtest.

The script intentionally reports research limitations instead of pretending that
bar data can reproduce every tick-level execution detail. Run it on the MT5
VPS, where broker history and contract specifications are available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bootstrap import add_project_root

add_project_root()

from scripts.strategy_router import StrategyRouter
from engine.features import FeatureEngine
from engine.risk_manager import RiskManager


SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "XAUUSD"]
RISK_BY_SYMBOL = {
    "EURUSD": 0.01,
    "GBPUSD": 0.01,
    "USDJPY": 0.005,
    "AUDUSD": 0.005,
    "USDCHF": 0.005,
    "XAUUSD": 0.005,
}
COMMISSION_PER_LOT_ROUND_TURN = 5.0
DAILY_LOSS_PCT = 0.05
TOTAL_LOSS_PCT = 0.10
MAX_OPEN_POSITIONS = 3
SLIPPAGE_POINTS = 1.0


@dataclass
class Position:
    symbol: str
    direction: int
    opened_at: pd.Timestamp
    entry: float
    stop: float
    target: float
    volume: float
    risk_usd: float
    score: object
    components: dict
    strategy: str
    regime: str
    news_context: str
    entry_index: int
    entry_commission: float
    entry_spread: float
    mfe_usd: float = 0.0
    mae_usd: float = 0.0


class ResearchBroker:
    def __init__(self, frames, specs):
        self.frames = frames
        self.specs = specs
        self.symbol = None
        self.price = 0.0
        self.mt5 = SimpleNamespace(TIMEFRAME_M15="M15")

    def set_context(self, symbol, price):
        self.symbol, self.price = symbol, float(price)

    def symbol_info(self, symbol):
        return self.specs[symbol]

    def symbol_tick(self, symbol):
        info = self.specs[symbol]
        spread = info.spread_price
        return SimpleNamespace(ask=self.price + spread / 2, bid=self.price - spread / 2)

    def rates_copy(self, symbol, timeframe, count):
        frame = self.frames[symbol][["open", "high", "low", "close", "tick_volume"]]
        frame = frame.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}).dropna()
        return frame.tail(count).to_dict("records")

    def normalize_volume(self, symbol, volume):
        info = self.specs[symbol]
        value = min(info.volume_max, max(0.0, float(volume)))
        return round(np.floor(value / info.volume_step) * info.volume_step, 8)

    def order_calc_margin(self, direction, symbol, volume, price):
        info = self.specs[symbol]
        return volume * info.contract_size * price / max(info.leverage, 1.0)


def mt5_symbol(canonical, mt5):
    matches = mt5.symbols_get(group=f"*{canonical}*") or []
    exact = next((item for item in matches if item.name.upper() == canonical), None)
    selected = exact or next((item for item in matches if item.name.upper().startswith(canonical)), None)
    if selected is None or not mt5.symbol_select(selected.name, True):
        raise RuntimeError(f"Could not resolve/select {canonical}")
    return selected


def download_history(symbols, output_dir, days, terminal_path):
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("Install MetaTrader5 on the Windows VPS") from exc
    if not mt5.initialize(path=terminal_path):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    frames, specs, names = {}, {}, {}
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        wanted = days * 24 * 12 + 300
        for canonical in symbols:
            selected = mt5_symbol(canonical, mt5)
            names[canonical] = selected.name
            chunks, position = [], 1
            while sum(len(chunk) for chunk in chunks) < wanted:
                chunk = mt5.copy_rates_from_pos(selected.name, mt5.TIMEFRAME_M5, position, 5000)
                if chunk is None or len(chunk) == 0:
                    break
                chunks.append(chunk)
                position += len(chunk)
                if len(chunk) < 5000:
                    break
            rates = np.concatenate(chunks) if chunks else None
            if rates is None or len(rates) < 500:
                raise RuntimeError(f"Insufficient history for {selected.name}: {mt5.last_error()}")
            frame = pd.DataFrame(rates).tail(wanted)
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame = frame.set_index("time").sort_index()
            frame["spread"] = frame.get("spread", 0).fillna(0)
            frame["tick_volume"] = frame.get("tick_volume", 0).fillna(0)
            frames[canonical] = FeatureEngine().add_features(frame)
            info = mt5.symbol_info(selected.name)
            point = float(getattr(info, "point", 0.00001) or 0.00001)
            tick_size = float(getattr(info, "trade_tick_size", point) or point)
            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            contract = float(getattr(info, "trade_contract_size", 100000.0) or 100000.0)
            specs[canonical] = {
                "point": point,
                "digits": int(getattr(info, "digits", 5) or 5),
                "tick_size": tick_size,
                "tick_value": tick_value or contract * tick_size,
                "contract_size": contract,
                "stops_level": float(getattr(info, "trade_stops_level", 0) or 0),
                "volume_min": float(getattr(info, "volume_min", 0.01) or 0.01),
                "volume_max": float(getattr(info, "volume_max", 100.0) or 100.0),
                "volume_step": float(getattr(info, "volume_step", 0.01) or 0.01),
                "leverage": 100.0,
                "spread_price": float(frame["spread"].median()) * point,
                "swap_long": float(getattr(info, "swap_long", 0.0) or 0.0),
                "swap_short": float(getattr(info, "swap_short", 0.0) or 0.0),
                "swap_3day": int(getattr(info, "swap_rollover3days", 3) or 3),
            }
            frame.to_csv(output_dir / f"{selected.name}_M5.csv")
            print(f"Downloaded {len(frame):,} M5 bars: {selected.name}")
        (output_dir / "contract_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
        (output_dir / "symbol_map.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
        return frames, {symbol: SimpleNamespace(**spec) for symbol, spec in specs.items()}, names
    finally:
        mt5.shutdown()


def load_cached(symbols, data_dir):
    specs_path = data_dir / "contract_specs.json"
    specs = json.loads(specs_path.read_text(encoding="utf-8")) if specs_path.exists() else {}
    frames, names = {}, {}
    for symbol in symbols:
        paths = list(data_dir.glob(f"*{symbol}*_M5.csv"))
        if not paths:
            raise FileNotFoundError(f"No cached data for {symbol} in {data_dir}")
        frame = pd.read_csv(paths[0])
        time_col = "time" if "time" in frame else "datetime"
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
        frame = frame.set_index(time_col).sort_index()
        for column in ("spread", "tick_volume"):
            if column not in frame:
                frame[column] = 0.0
        frames[symbol] = frame
        names[symbol] = paths[0].stem.replace("_M5", "")
        if symbol not in specs:
            point = 0.01 if symbol == "XAUUSD" else 0.00001
            specs[symbol] = {"point": point, "digits": 2 if symbol == "XAUUSD" else 5, "tick_size": point, "tick_value": 100 * point if symbol == "XAUUSD" else 100000 * point, "contract_size": 100 if symbol == "XAUUSD" else 100000, "stops_level": 0, "volume_min": 0.01, "volume_max": 100, "volume_step": 0.01, "leverage": 100, "spread_price": float(frame.spread.median()) * point, "swap_long": 0, "swap_short": 0, "swap_3day": 3}
        frames[symbol] = FeatureEngine().add_features(frame)
        specs[symbol] = SimpleNamespace(**specs[symbol])
    return frames, specs, names


def load_news(path):
    if not path:
        return pd.DataFrame(columns=["time", "currency", "title"])
    frame = pd.read_csv(path)
    required = {"time", "currency"}
    if not required.issubset(frame.columns):
        raise ValueError("News CSV must contain time,currency columns")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["currency"] = frame.currency.astype(str).str.upper()
    return frame


def news_blocked(news, symbol, timestamp, before, after):
    if news.empty:
        return False
    currencies = [symbol[:3], symbol[3:]] if len(symbol) == 6 else ["USD"]
    if "XAU" in symbol:
        currencies = ["USD"]
    relevant = news[news.currency.isin(currencies)]
    delta = (relevant.time - timestamp).dt.total_seconds() / 60
    return bool(((delta >= -after) & (delta <= before)).any())


def price_value(position, price, specs):
    sign = 1 if position.direction == 1 else -1
    return (price - position.entry) * sign * position.volume * specs[position.symbol].tick_value / specs[position.symbol].tick_size


def regime_label(frame, index):
    row = frame.iloc[index]
    atr = float(row.get("atr", 0.0) or 0.0)
    median_atr = float(frame["atr"].iloc[max(0, index - 100): index + 1].median())
    close = float(row["close"])
    ema50 = float(row.get("ema50", close))
    if atr <= 0 or median_atr <= 0:
        return "unknown"
    if atr > median_atr * 2.0:
        return "high_volatility"
    if abs(close - ema50) > atr * 2.0:
        return "trending"
    return "ranging"


def commission(volume):
    return volume * COMMISSION_PER_LOT_ROUND_TURN / 2


def exit_trade(position, timestamp, price, reason, balance, specs, spread_price, slippage_price, swap):
    gross = price_value(position, price, specs)
    exit_spread = spread_price * position.volume * specs[position.symbol].tick_value / specs[position.symbol].tick_size
    slip = slippage_price * position.volume * specs[position.symbol].tick_value / specs[position.symbol].tick_size
    pnl = gross - commission(position.volume) - exit_spread - slip + swap
    return {**asdict(position), "exit_time": timestamp, "exit": price, "reason": reason, "pnl": pnl, "gross_pnl": gross, "exit_commission": commission(position.volume), "exit_spread_cost": exit_spread, "slippage_cost": slip, "swap": swap, "balance_before": balance, "balance_after": balance + pnl, "stopped_before_later_tp": False}


def later_target_reached(result, frames, lookahead_bars=288):
    """Check whether an SL/time-stop would have reached its original target later."""
    if result["reason"] not in {"SL", "SL_and_TP_same_bar_conservative_SL", "weekend_gap"}:
        return False
    frame = frames[result["symbol"]]
    start = frame.index.get_indexer([pd.Timestamp(result["exit_time"])], method="nearest")[0] + 1
    future = frame.iloc[start: start + lookahead_bars]
    if result["direction"] == 1:
        return bool((future["high"] >= result["target"]).any())
    return bool((future["low"] <= result["target"]).any())


def build_plan(symbol, frame, index, signal, score, components, equity, broker, strategy, news_context):
    data = frame.iloc[max(0, index - 700): index + 1]
    broker.set_context(symbol, float(frame.close.iloc[index]))
    price = float(frame.close.iloc[index])
    if hasattr(strategy, "analyze_trade"):
        plan = strategy.analyze_trade(data, broker=broker, symbol=symbol, equity=equity, signal=signal, price=price)
    else:
        atr = float(frame.atr.iloc[index])
        if not np.isfinite(atr) or atr <= 0:
            return None
        stop, target = RiskManager().calculate_sl_tp(
            signal,
            price,
            atr,
            tp_atr=5.0,
        )
        plan = {"signal": signal, "price": price, "stop": stop, "target": target}
    if not plan:
        return None
    info = broker.specs[symbol]
    stop_distance = abs(float(plan["price"]) - float(plan["stop"]))
    risk = equity * RISK_BY_SYMBOL.get(symbol, 0.005)
    raw = risk / max(stop_distance * info.tick_value / info.tick_size, 1e-12)
    volume = broker.normalize_volume(symbol, raw)
    if volume < info.volume_min or broker.order_calc_margin(signal, symbol, volume, float(plan["price"])) > equity * 0.85:
        return None
    return Position(symbol, signal, frame.index[index], float(plan["price"]), float(plan["stop"]), float(plan["target"]), volume, risk, score, components, strategy.__class__.__name__, regime_label(frame, index), news_context, index, commission(volume), info.spread_price * volume * info.tick_value / info.tick_size)


def simulate(frames, specs, news, initial_balance, slippage_points):
    router = StrategyRouter()
    broker = ResearchBroker(frames, specs)
    signals = {symbol: router.get_strategy(symbol).generate_signals(frame) for symbol, frame in frames.items()}
    timeline = sorted(set().union(*(frame.index for frame in frames.values())))
    positions, pending, trades, equity_rows = {}, {}, [], []
    balance, peak, consecutive_losses, pause_until = initial_balance, initial_balance, 0, None
    daily_base, current_day = initial_balance, None
    daily_blocked = False
    total_blocked = False
    for timestamp in timeline:
        prague_day = timestamp.tz_convert("Europe/Prague").date()
        if prague_day != current_day:
            current_day, daily_base = prague_day, balance
            daily_blocked = False
        for symbol, position in list(positions.items()):
            if timestamp not in frames[symbol].index:
                continue
            bar = frames[symbol].loc[timestamp]
            floating = price_value(position, float(bar.close), specs)
            position.mfe_usd = max(position.mfe_usd, floating)
            position.mae_usd = min(position.mae_usd, floating)
            reason, exit_price = None, None
            if position.direction == 1:
                stop_hit, target_hit = bar.low <= position.stop, bar.high >= position.target
            else:
                stop_hit, target_hit = bar.high >= position.stop, bar.low <= position.target
            if stop_hit and target_hit:
                reason, exit_price = "SL_and_TP_same_bar_conservative_SL", position.stop
            elif stop_hit:
                reason, exit_price = "SL", position.stop
            elif target_hit:
                reason, exit_price = "TP", position.target
            if reason is None and timestamp.weekday() >= 5:
                reason, exit_price = "weekend_gap", float(bar.open)
            if reason is not None:
                held_days = max(0, (timestamp - position.opened_at).total_seconds() / 86400)
                spec = specs[symbol]
                swap_rate = spec.swap_long if position.direction == 1 else spec.swap_short
                swap = -abs(swap_rate) * position.volume * max(0, int(held_days))
                result = exit_trade(position, timestamp, exit_price, reason, balance, specs, spec.spread_price, slippage_points * spec.point, swap)
                balance = result["balance_after"]
                trades.append(result)
                del positions[symbol]
                if result["pnl"] < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= 2:
                        pause_until = timestamp + pd.Timedelta(hours=3)
                else:
                    consecutive_losses, pause_until = 0, None
        floating_total = sum(price_value(pos, float(frames[symbol].loc[timestamp].close), specs) for symbol, pos in positions.items() if timestamp in frames[symbol].index)
        equity = balance + floating_total
        peak = max(peak, equity)
        daily_limit = daily_base - initial_balance * DAILY_LOSS_PCT
        total_limit = initial_balance * (1 - TOTAL_LOSS_PCT)
        if equity <= daily_limit or equity <= total_limit:
            for symbol, pos in list(positions.items()):
                if timestamp in frames[symbol].index:
                    result = exit_trade(pos, timestamp, float(frames[symbol].loc[timestamp].close), "FTMO_limit", balance, specs, specs[symbol].spread_price, slippage_points * specs[symbol].point, 0)
                    balance = result["balance_after"]
                    trades.append(result)
                    del positions[symbol]
            equity = balance
            if equity <= total_limit:
                total_blocked = True
            if equity <= daily_limit:
                daily_blocked = True
        equity_rows.append({"time": timestamp, "balance": balance, "equity": equity, "daily_limit": daily_limit, "total_limit": total_limit, "peak_equity": peak})
        for symbol, frame in frames.items():
            if timestamp not in frame.index or symbol in positions or len(positions) >= MAX_OPEN_POSITIONS or (pause_until and timestamp < pause_until) or daily_blocked or total_blocked:
                continue
            local = timestamp.tz_convert("Africa/Accra")
            if local.weekday() < 5 and 21 <= local.hour < 23:
                continue
            index = frame.index.get_loc(timestamp)
            signal = int(signals[symbol].iloc[index])
            if not signal or news_blocked(news, symbol, timestamp, 5, 10):
                continue
            strategy = router.get_strategy(symbol)
            score, components = None, {}
            if hasattr(strategy, "evaluate_latest"):
                _, score, components = strategy.evaluate_latest(frame.iloc[max(0, index - 700): index + 1], spread_points=float(frame.spread.iloc[index]))
            news_context = "near_news" if news_blocked(news, symbol, timestamp, 60, 60) else "no_news"
            plan = build_plan(symbol, frame, index, signal, score, components, equity, broker, strategy, news_context)
            if plan is not None:
                pending[symbol] = (index + 1, plan)
        for symbol, (entry_index, plan) in list(pending.items()):
            frame = frames[symbol]
            if entry_index < len(frame) and frame.index[entry_index] == timestamp:
                if symbol not in positions and len(positions) < MAX_OPEN_POSITIONS:
                    bar = frame.iloc[entry_index]
                    spec = specs[symbol]
                    entry = float(bar.open) + (spec.spread_price / 2 if plan.direction == 1 else -spec.spread_price / 2)
                    plan.entry = entry
                    plan.entry_spread = spec.spread_price * plan.volume * spec.tick_value / spec.tick_size
                    positions[symbol] = plan
                del pending[symbol]
    trades_frame = pd.DataFrame(trades)
    if not trades_frame.empty:
        trades_frame["stopped_before_later_tp"] = trades_frame.apply(lambda row: later_target_reached(row.to_dict(), frames), axis=1)
    return trades_frame, pd.DataFrame(equity_rows)


def report_metrics(trades, equity, initial):
    if equity.empty or trades.empty:
        return {"trades": 0}
    wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    curve = equity.set_index("time").equity
    dd = curve - curve.cummax()
    day = equity.time.dt.tz_convert("Europe/Prague").dt.date
    daily = equity.groupby(day).equity.min() - equity.groupby(day).equity.first()
    streak, longest = 0, 0
    for pnl in trades.pnl if not trades.empty else []:
        streak = streak + 1 if pnl <= 0 else 0
        longest = max(longest, streak)
    gross_profit, gross_loss = wins.pnl.sum(), -losses.pnl.sum()
    commission = trades[["entry_commission", "exit_commission"]].sum().sum() if not trades.empty else 0
    spread = trades[["entry_spread", "exit_spread_cost"]].sum().sum() if not trades.empty else 0
    summary = {"start_balance": initial, "end_balance": float(curve.iloc[-1]), "net_pnl": float(curve.iloc[-1] - initial), "trades": int(len(trades)), "win_rate_pct": float(len(wins) / len(trades) * 100) if len(trades) else 0, "profit_factor": float(gross_profit / gross_loss) if gross_loss else None, "expectancy_per_trade": float(trades.pnl.mean()) if not trades.empty else 0, "average_win": float(wins.pnl.mean()) if not wins.empty else 0, "average_loss": float(losses.pnl.mean()) if not losses.empty else 0, "max_drawdown_usd": float(dd.min()), "max_drawdown_pct": float((dd / curve.cummax() * 100).min()), "max_daily_drawdown_usd": float(daily.min()), "max_daily_drawdown_pct": float(daily.min() / initial * 100), "longest_losing_streak": longest, "average_holding_hours": float((pd.to_datetime(trades.exit_time, utc=True) - pd.to_datetime(trades.opened_at, utc=True)).dt.total_seconds().mean() / 3600) if not trades.empty else 0, "commission_as_pct_gross_profit": float(commission / gross_profit * 100) if gross_profit > 0 else None, "spread_as_pct_gross_profit": float(spread / gross_profit * 100) if gross_profit > 0 else None, "mae_average": float(trades.mae_usd.mean()) if not trades.empty else 0, "mfe_average": float(trades.mfe_usd.mean()) if not trades.empty else 0, "stopped_before_later_tp_pct": None, "ftmo_daily_pass": bool(daily.min() >= -initial * DAILY_LOSS_PCT), "ftmo_total_pass": bool(curve.min() >= initial * (1 - TOTAL_LOSS_PCT))}
    summary["by_symbol"] = {}
    if not trades.empty:
        for symbol, group in trades.groupby("symbol"):
            summary["by_symbol"][symbol] = {"trades": int(len(group)), "pnl": float(group.pnl.sum()), "win_rate_pct": float((group.pnl > 0).mean() * 100), "expectancy": float(group.pnl.mean()), "average_mae": float(group.mae_usd.mean()), "average_mfe": float(group.mfe_usd.mean())}
        for name, group in trades.groupby(trades.strategy):
            summary.setdefault("by_strategy", {})[name] = {"trades": int(len(group)), "pnl": float(group.pnl.sum()), "win_rate_pct": float((group.pnl > 0).mean() * 100), "expectancy": float(group.pnl.mean())}
        trades["session"] = pd.to_datetime(trades.opened_at, utc=True).dt.tz_convert("Europe/London").dt.hour.map(lambda h: "Asia" if h < 8 else "London" if h < 13 else "NewYork" if h < 21 else "Late")
        summary["by_session"] = trades.groupby("session").pnl.agg(["count", "sum", "mean"]).to_dict("index")
        trades["score_group"] = trades.score.fillna("n/a").astype(str)
        summary["by_score"] = trades.groupby("score_group").pnl.agg(["count", "sum", "mean"]).to_dict("index")
        summary["by_regime"] = trades.groupby("regime").pnl.agg(["count", "sum", "mean"]).to_dict("index")
        summary["by_news_context"] = trades.groupby("news_context").pnl.agg(["count", "sum", "mean"]).to_dict("index")
        summary["stopped_before_later_tp_pct"] = float(trades.stopped_before_later_tp.mean() * 100) if "stopped_before_later_tp" in trades else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-balance", type=float, default=10000)
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--data-dir", default="results/research_backtest/data")
    parser.add_argument("--output", default="results/research_backtest")
    parser.add_argument("--news-csv", help="CSV with UTC time,currency,title columns")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--slippage-points", type=float, default=SLIPPAGE_POINTS)
    args = parser.parse_args()
    data_dir, output = Path(args.data_dir), Path(args.output)
    cached = all(list(data_dir.glob(f"*{symbol}*_M5.csv")) for symbol in SYMBOLS)
    terminal = os.getenv("VX_MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if args.force_download or not cached:
        frames, specs, names = download_history(SYMBOLS, data_dir, args.history_days, terminal)
    else:
        frames, specs, names = load_cached(SYMBOLS, data_dir)
    news = load_news(args.news_csv)
    trades, equity = simulate(frames, specs, news, args.initial_balance, args.slippage_points)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trades.csv", index=False)
    equity.to_csv(output / "equity.csv", index=False)
    report = {"settings": {**vars(args), "symbols": SYMBOLS, "risk_by_symbol": RISK_BY_SYMBOL, "commission_round_turn_per_lot": COMMISSION_PER_LOT_ROUND_TURN, "daily_loss_pct": DAILY_LOSS_PCT, "total_loss_pct": TOTAL_LOSS_PCT, "max_open_positions": MAX_OPEN_POSITIONS}, "broker_symbols": names, "metrics": report_metrics(trades, equity, args.initial_balance), "limitations": ["News blackout is applied only when --news-csv is supplied; live calendar data is not a historical archive.", "Same-bar SL and TP is resolved conservatively as SL.", "MT5 bar data cannot reproduce tick ordering; slippage is modeled as a fixed point cost.", "Swap values depend on broker symbol settings and are captured at download time."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, default=str))


if __name__ == "__main__":
    main()
