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


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    regime_gate: bool = False
    m30_gate: bool = False
    session_guard: bool = False
    cost_reward_gate: bool = False
    no_chase_gate: bool = False
    exit_management: bool = False
    max_open_positions: int = MAX_OPEN_POSITIONS
    shared_fx_risk: float | None = None
    daily_soft_stop_pct: float | None = None
    block_new_york_symbols: tuple[str, ...] = ()
    new_york_tp_atr: float | None = None
    continue_after_ftmo_breach: bool = False
    regime_symbols: tuple[str, ...] = ()
    m30_symbols: tuple[str, ...] = ()
    spread_guard_symbols: tuple[str, ...] = ()
    max_spread_atr_ratio: float | None = None
    no_trade_accra_start_hour: int = 20
    no_trade_accra_end_hour: int = 23


EXPERIMENTS = {
    "baseline": ExperimentConfig("baseline"),
    "entry_quality": ExperimentConfig(
        "entry_quality", regime_gate=True, m30_gate=True, session_guard=True,
        cost_reward_gate=True, no_chase_gate=True,
    ),
    "entry_quality_exits": ExperimentConfig(
        "entry_quality_exits", regime_gate=True, m30_gate=True, session_guard=True,
        cost_reward_gate=True, no_chase_gate=True, exit_management=True,
    ),
    "full_controls": ExperimentConfig(
        "full_controls", regime_gate=True, m30_gate=True, session_guard=True,
        cost_reward_gate=True, no_chase_gate=True, exit_management=True,
        max_open_positions=2, shared_fx_risk=0.01, daily_soft_stop_pct=0.02,
    ),
}


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
        self.last_plan_rejection = None
        self.mt5 = SimpleNamespace(TIMEFRAME_M15="M15")

    def set_context(self, symbol, price):
        self.symbol, self.price = symbol, float(price)

    def symbol_info(self, symbol):
        info = self.specs[symbol]
        aliases = {
            "trade_tick_size": "tick_size",
            "trade_tick_value": "tick_value",
            "trade_contract_size": "contract_size",
            "trade_stops_level": "stops_level",
        }
        for mt5_name, research_name in aliases.items():
            if not hasattr(info, mt5_name):
                setattr(info, mt5_name, getattr(info, research_name, 0.0))
        return info

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
            frames[canonical] = add_research_features(FeatureEngine().add_features(frame))
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
        frames[symbol] = add_research_features(FeatureEngine().add_features(frame))
        specs[symbol] = SimpleNamespace(**specs[symbol])
    return frames, specs, names


def add_research_features(frame):
    """Add completed M15/M30 regime features without using future bars."""
    frame = frame.copy()
    for minutes, prefix in ((15, "m15"), (30, "m30")):
        higher = frame[["open", "high", "low", "close", "tick_volume"]].resample(f"{minutes}min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna()
        previous_close = higher.close.shift(1)
        tr = pd.concat(
            [higher.high - higher.low, (higher.high - previous_close).abs(), (higher.low - previous_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        up = higher.high.diff()
        down = -higher.low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        plus_di = 100 * plus_dm.rolling(14).sum() / tr.rolling(14).sum()
        minus_di = 100 * minus_dm.rolling(14).sum() / tr.rolling(14).sum()
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(14).mean()
        ema = higher.close.ewm(span=20, adjust=False).mean()
        bb_mid = higher.close.rolling(20).mean()
        bb_width = (4 * higher.close.rolling(20).std() / bb_mid).replace([np.inf, -np.inf], np.nan)
        chop = 100 * np.log10(tr.rolling(14).sum() / (higher.high.rolling(14).max() - higher.low.rolling(14).min())) / np.log10(14)
        vwap = (higher.close * higher.tick_volume).rolling(20).sum() / higher.tick_volume.rolling(20).sum().replace(0, np.nan)
        persistence = higher.close.diff().gt(0).rolling(8).mean().sub(0.5).abs() * 2
        higher[f"{prefix}_adx"] = adx
        higher[f"{prefix}_atr"] = atr
        higher[f"{prefix}_atr_ratio"] = atr / atr.rolling(50).median()
        higher[f"{prefix}_ema_slope"] = ema.diff(3)
        higher[f"{prefix}_bb_width"] = bb_width
        higher[f"{prefix}_chop"] = chop
        higher[f"{prefix}_vwap_distance"] = (higher.close - vwap).abs()
        higher[f"{prefix}_persistence"] = persistence
        higher[f"{prefix}_ema"] = ema
        # Shift higher-timeframe values so a 05-minute bar only sees a completed
        # higher-timeframe candle.
        columns = [column for column in higher.columns if column.startswith(prefix)]
        mapped = higher[columns].shift(1).reindex(frame.index, method="ffill")
        frame = frame.join(mapped)
    return frame


def regime_snapshot(frame, index):
    row = frame.iloc[index]
    atr = float(row.get("atr", 0.0) or 0.0)
    checks = {
        "adx_low_moderate": float(row.get("m15_adx", np.nan)) <= 22.0,
        "atr_stable": 0.70 <= float(row.get("m15_atr_ratio", np.nan)) <= 1.50,
        "ema_slope_flat": abs(float(row.get("m15_ema_slope", np.nan))) <= max(atr * 0.20, 1e-12),
        "bb_width_stable": float(row.get("m15_bb_width", np.nan)) <= float(frame["m15_bb_width"].iloc[max(0, index - 100): index + 1].median()) * 1.50,
        "choppy": float(row.get("m15_chop", np.nan)) >= 45.0,
        "near_vwap": float(row.get("m15_vwap_distance", np.nan)) <= max(atr * 1.50, 1e-12),
        "not_persistent": float(row.get("m15_persistence", np.nan)) <= 0.65,
    }
    valid = {key: bool(value) for key, value in checks.items() if not pd.isna(value)}
    score = sum(valid.values())
    adx = float(row.get("m15_adx", np.nan))
    slope = abs(float(row.get("m15_ema_slope", np.nan)))
    trending = bool(adx >= 25 and slope > max(atr * 0.25, 1e-12))
    label = "ranging" if score >= 5 else "trending" if trending else "neutral"
    return {"score": score, "label": label, "components": valid}


def m30_allows_mean_reversion(frame, index):
    row = frame.iloc[index]
    adx = float(row.get("m30_adx", np.nan))
    atr = float(row.get("atr", 0.0) or 0.0)
    slope = abs(float(row.get("m30_ema_slope", np.nan)))
    if pd.isna(adx) or pd.isna(slope) or atr <= 0:
        return False
    return adx <= 25 and slope <= atr * 0.45


def session_name(timestamp):
    hour = timestamp.tz_convert("Europe/London").hour
    return "Asia" if hour < 8 else "London" if hour < 13 else "NewYork" if hour < 21 else "Late"


def session_allows(symbol, score, timestamp):
    session = session_name(timestamp)
    if session == "Late":
        return False
    if symbol == "GBPUSD" and session in {"London", "NewYork"}:
        return score >= 5
    return True


def session_is_new_york(timestamp):
    return session_name(timestamp) == "NewYork"


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
    return regime_snapshot(frame, index)["label"]


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
    broker.last_plan_rejection = None
    data = frame.iloc[max(0, index - 700): index + 1]
    broker.set_context(symbol, float(frame.close.iloc[index]))
    price = float(frame.close.iloc[index])
    if hasattr(strategy, "analyze_trade"):
        plan = strategy.analyze_trade(data, broker=broker, symbol=symbol, equity=equity, signal=signal, price=price)
    else:
        atr = float(frame.atr.iloc[index])
        if not np.isfinite(atr) or atr <= 0:
            broker.last_plan_rejection = "missing_atr"
            return None
        stop, target = RiskManager().calculate_sl_tp(
            signal,
            price,
            atr,
            tp_atr=5.0,
        )
        plan = {"signal": signal, "price": price, "stop": stop, "target": target}
    if not plan:
        broker.last_plan_rejection = "strategy_plan_rejected"
        return None
    try:
        info = broker.specs[symbol]
    except KeyError:
        broker.last_plan_rejection = "contract_spec_missing"
        return None
    stop_distance = abs(float(plan["price"]) - float(plan["stop"]))
    if not np.isfinite(stop_distance) or stop_distance <= 0:
        broker.last_plan_rejection = "invalid_stop_distance"
        return None
    risk = equity * RISK_BY_SYMBOL.get(symbol, 0.005)
    raw = risk / max(stop_distance * info.tick_value / info.tick_size, 1e-12)
    volume = broker.normalize_volume(symbol, raw)
    if volume < info.volume_min:
        broker.last_plan_rejection = "volume_below_minimum"
        return None
    if broker.order_calc_margin(signal, symbol, volume, float(plan["price"])) > equity * 0.85:
        broker.last_plan_rejection = "margin_exceeded"
        return None
    return Position(symbol, signal, frame.index[index], float(plan["price"]), float(plan["stop"]), float(plan["target"]), volume, risk, score, components, strategy.__class__.__name__, regime_label(frame, index), news_context, index, commission(volume), info.spread_price * volume * info.tick_value / info.tick_size)


def execution_exit(position, m5_bar, m1_bars):
    """Use sequential M1 candles to resolve exits inside an M5 bar."""
    if m1_bars is None:
        bars = [m5_bar]
    elif isinstance(m1_bars, list):
        bars = m1_bars or [m5_bar]
    elif getattr(m1_bars, "empty", False):
        bars = [m5_bar]
    else:
        bars = m1_bars
    for bar in bars:
        if position.direction == 1:
            stop_hit, target_hit = bar.low <= position.stop, bar.high >= position.target
        else:
            stop_hit, target_hit = bar.high >= position.stop, bar.low <= position.target
        if stop_hit and target_hit:
            return "SL_and_TP_same_bar_conservative_SL", position.stop
        if stop_hit:
            return "SL", position.stop
        if target_hit:
            return "TP", position.target
    return None, None


def simulate(frames, specs, news, initial_balance, slippage_points, config=None, candidate_rows=None, execution_frames=None):
    config = config or EXPERIMENTS["baseline"]
    candidate_rows = candidate_rows if candidate_rows is not None else []
    router = StrategyRouter()
    broker = ResearchBroker(frames, specs)
    signals = {symbol: router.get_strategy(symbol).generate_signals(frame) for symbol, frame in frames.items()}
    timeline = sorted(set().union(*(frame.index for frame in frames.values())))
    positions, pending, trades, equity_rows = {}, {}, [], []
    balance, peak, consecutive_losses, pause_until = initial_balance, initial_balance, 0, None
    daily_base, current_day = initial_balance, None
    daily_blocked = False
    total_blocked = False
    previous_timestamp = None
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
            m1_bars = None
            if execution_frames is not None and symbol in execution_frames:
                start = previous_timestamp or timestamp - pd.Timedelta(minutes=5)
                m1_bars = execution_frames[symbol].loc[
                    (execution_frames[symbol].index >= start)
                    & (execution_frames[symbol].index < timestamp)
                ].itertuples(index=False)
                m1_bars = list(m1_bars)
            reason, exit_price = execution_exit(position, bar, m1_bars)
            current_index = frames[symbol].index.get_loc(timestamp)
            if config.exit_management and reason is None:
                current_signal = int(signals[symbol].iloc[current_index])
                held_bars = max(0, current_index - position.entry_index)
                initial_risk = max(position.risk_usd, 1e-12)
                if current_signal == -position.direction and held_bars >= 3:
                    reason, exit_price = "setup_invalidation", float(bar.close)
                elif held_bars >= 12 and (floating <= 0 or position.mfe_usd < initial_risk * 0.25):
                    reason, exit_price = "time_stop", float(bar.close)
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
            if config.continue_after_ftmo_breach:
                # Diagnostic mode records the breach through the equity curve but
                # deliberately keeps positions and entries running for comparison.
                pass
            else:
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
            if timestamp not in frame.index or symbol in positions or len(positions) >= config.max_open_positions or (pause_until and timestamp < pause_until) or daily_blocked or total_blocked:
                continue
            local = timestamp.tz_convert("Africa/Accra")
            if local.weekday() < 5 and config.no_trade_accra_start_hour <= local.hour < config.no_trade_accra_end_hour:
                continue
            index = frame.index.get_loc(timestamp)
            signal = int(signals[symbol].iloc[index])
            if not signal:
                continue
            strategy = router.get_strategy(symbol)
            score, components = None, {}
            if hasattr(strategy, "evaluate_latest"):
                _, score, components = strategy.evaluate_latest(frame.iloc[max(0, index - 700): index + 1], spread_points=float(frame.spread.iloc[index]))
            regime = regime_snapshot(frame, index)
            news_near = news_blocked(news, symbol, timestamp, 60, 60)
            news_block = news_blocked(news, symbol, timestamp, 5, 10)
            news_context = "near_news" if news_near else "no_news"
            atr_value = float(frame.atr.iloc[index])
            spread_atr_ratio = None
            if np.isfinite(atr_value) and atr_value > 0:
                spread_atr_ratio = float(frame.spread.iloc[index]) * specs[symbol].point / atr_value
            rejection = None
            if news_block:
                rejection = "news_blackout"
            elif symbol in config.block_new_york_symbols and session_is_new_york(timestamp):
                rejection = "new_york_blocked"
            elif config.session_guard and not session_allows(symbol, int(score or 0), timestamp):
                rejection = "session_filter"
            elif config.regime_gate and symbol in (set(config.regime_symbols) or {"EURUSD", "GBPUSD"}) and regime["label"] != "ranging":
                rejection = "regime_filter"
            elif config.m30_gate and symbol in (set(config.m30_symbols) or {"EURUSD", "GBPUSD"}) and not m30_allows_mean_reversion(frame, index):
                rejection = "m30_regime_filter"
            elif config.no_chase_gate and components.get("current_z") is not None and abs(float(components["current_z"])) > 2.75:
                rejection = "no_chase"
            elif config.max_spread_atr_ratio is not None and symbol in config.spread_guard_symbols and (spread_atr_ratio is None or spread_atr_ratio > config.max_spread_atr_ratio):
                rejection = "spread_too_wide"
            plan = None if rejection else build_plan(symbol, frame, index, signal, score, components, equity, broker, strategy, news_context)
            if plan is not None and config.new_york_tp_atr is not None and symbol in {"EURUSD", "GBPUSD"} and session_is_new_york(timestamp):
                atr = float(frame.atr.iloc[index])
                if np.isfinite(atr) and atr > 0:
                    plan.target = plan.entry + signal * config.new_york_tp_atr * atr
            if rejection is None and plan is None:
                rejection = broker.last_plan_rejection or "invalid_plan_or_margin"
            if rejection is None and config.cost_reward_gate:
                info = specs[symbol]
                gross_target = abs(plan.target - plan.entry) * plan.volume * info.tick_value / info.tick_size
                round_trip_cost = plan.entry_commission * 2 + plan.entry_spread * 2 + slippage_points * info.point * plan.volume * info.tick_value / info.tick_size * 2
                if gross_target - round_trip_cost < plan.risk_usd:
                    rejection = "insufficient_net_reward"
            if rejection is None and config.shared_fx_risk is not None and symbol in {"EURUSD", "GBPUSD"}:
                current_fx_risk = sum(pos.risk_usd for pos in positions.values() if pos.symbol in {"EURUSD", "GBPUSD"})
                if current_fx_risk + plan.risk_usd > equity * config.shared_fx_risk:
                    rejection = "shared_fx_risk_cap"
            if rejection is None and config.daily_soft_stop_pct is not None and equity <= daily_base * (1 - config.daily_soft_stop_pct):
                rejection = "daily_soft_stop"
            candidate_rows.append({
                "experiment": config.name,
                "symbol": symbol,
                "time": timestamp,
                "signal": signal,
                "score": score,
                "score_components": json.dumps(components, default=str),
                "regime_score": regime["score"],
                "regime": regime["label"],
                "regime_components": json.dumps(regime["components"], default=str),
                "m30_allows_mean_reversion": m30_allows_mean_reversion(frame, index),
                "session": session_name(timestamp),
                "news_context": news_context,
                "spread_points": float(frame.spread.iloc[index]),
                "atr": float(frame.atr.iloc[index]),
                "spread_atr_ratio": spread_atr_ratio,
                "entry_price": plan.entry if plan else None,
                "stop": plan.stop if plan else None,
                "target": plan.target if plan else None,
                "accepted": rejection is None,
                "rejection_reason": rejection or "accepted",
            })
            if plan is not None and rejection is None:
                pending[symbol] = (index + 1, plan)
        for symbol, (entry_index, plan) in list(pending.items()):
            frame = frames[symbol]
            if entry_index < len(frame) and frame.index[entry_index] == timestamp:
                if symbol not in positions and len(positions) < config.max_open_positions:
                    bar = frame.iloc[entry_index]
                    spec = specs[symbol]
                    entry = float(bar.open) + (spec.spread_price / 2 if plan.direction == 1 else -spec.spread_price / 2)
                    plan.entry = entry
                    plan.entry_spread = spec.spread_price * plan.volume * spec.tick_value / spec.tick_size
                    positions[symbol] = plan
                del pending[symbol]
        previous_timestamp = timestamp
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
    daily_breaches = equity[equity.equity <= equity.daily_limit]
    total_breaches = equity[equity.equity <= equity.total_limit]
    daily_breach_times = pd.to_datetime(daily_breaches["time"], utc=True) if not daily_breaches.empty else pd.Series(dtype="datetime64[ns, UTC]")
    summary = {"start_balance": initial, "end_balance": float(curve.iloc[-1]), "net_pnl": float(curve.iloc[-1] - initial), "trades": int(len(trades)), "winning_trades": int(len(wins)), "losing_trades": int(len(losses)), "win_rate_pct": float(len(wins) / len(trades) * 100) if len(trades) else 0, "profit_factor": float(gross_profit / gross_loss) if gross_loss else None, "expectancy_per_trade": float(trades.pnl.mean()) if not trades.empty else 0, "average_win": float(wins.pnl.mean()) if not wins.empty else 0, "average_loss": float(losses.pnl.mean()) if not losses.empty else 0, "max_drawdown_usd": float(dd.min()), "max_drawdown_pct": float((dd / curve.cummax() * 100).min()), "max_daily_drawdown_usd": float(daily.min()), "max_daily_drawdown_pct": float(daily.min() / initial * 100), "longest_losing_streak": longest, "average_holding_hours": float((pd.to_datetime(trades.exit_time, utc=True) - pd.to_datetime(trades.opened_at, utc=True)).dt.total_seconds().mean() / 3600) if not trades.empty else 0, "commission_as_pct_gross_profit": float(commission / gross_profit * 100) if gross_profit > 0 else None, "spread_as_pct_gross_profit": float(spread / gross_profit * 100) if gross_profit > 0 else None, "mae_average": float(trades.mae_usd.mean()) if not trades.empty else 0, "mfe_average": float(trades.mfe_usd.mean()) if not trades.empty else 0, "stopped_before_later_tp_pct": None, "daily_breach_count": int(daily_breach_times.dt.tz_convert("Europe/Prague").dt.date.nunique()) if not daily_breaches.empty else 0, "total_breach": bool(not total_breaches.empty), "first_daily_breach": str(daily_breaches.time.iloc[0]) if not daily_breaches.empty else None, "first_total_breach": str(total_breaches.time.iloc[0]) if not total_breaches.empty else None, "ftmo_daily_pass": bool(daily.min() >= -initial * DAILY_LOSS_PCT), "ftmo_total_pass": bool(curve.min() >= initial * (1 - TOTAL_LOSS_PCT))}
    summary["by_symbol"] = {}
    if not trades.empty:
        for symbol, group in trades.groupby("symbol"):
            group_wins = group[group.pnl > 0]
            group_losses = group[group.pnl <= 0]
            group_gross_loss = -group_losses.pnl.sum()
            summary["by_symbol"][symbol] = {"trades": int(len(group)), "wins": int(len(group_wins)), "losses": int(len(group_losses)), "pnl": float(group.pnl.sum()), "win_rate_pct": float((group.pnl > 0).mean() * 100), "profit_factor": float(group_wins.pnl.sum() / group_gross_loss) if group_gross_loss else None, "expectancy": float(group.pnl.mean()), "average_win": float(group_wins.pnl.mean()) if not group_wins.empty else 0, "average_loss": float(group_losses.pnl.mean()) if not group_losses.empty else 0, "average_mae": float(group.mae_usd.mean()), "average_mfe": float(group.mfe_usd.mean())}
        for name, group in trades.groupby(trades.strategy):
            summary.setdefault("by_strategy", {})[name] = {"trades": int(len(group)), "pnl": float(group.pnl.sum()), "win_rate_pct": float((group.pnl > 0).mean() * 100), "expectancy": float(group.pnl.mean())}
        trades["session"] = pd.to_datetime(trades.opened_at, utc=True).dt.tz_convert("Europe/London").dt.hour.map(lambda h: "Asia" if h < 8 else "London" if h < 13 else "NewYork" if h < 21 else "Late")
        summary["by_session"] = trades.groupby("session").pnl.agg(["count", "sum", "mean"]).to_dict("index")
        summary["by_symbol_session"] = {}
        for (symbol, session), group in trades.groupby(["symbol", "session"]):
            group_wins = group[group.pnl > 0]
            group_losses = group[group.pnl <= 0]
            group_gross_loss = -group_losses.pnl.sum()
            summary["by_symbol_session"].setdefault(symbol, {})[session] = {"trades": int(len(group)), "wins": int(len(group_wins)), "losses": int(len(group_losses)), "win_rate_pct": float((group.pnl > 0).mean() * 100), "net_pnl": float(group.pnl.sum()), "average_pnl": float(group.pnl.mean()), "profit_factor": float(group_wins.pnl.sum() / group_gross_loss) if group_gross_loss else None, "average_mae": float(group.mae_usd.mean()), "average_mfe": float(group.mfe_usd.mean())}
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
    parser.add_argument("--profile", choices=["baseline", "entry_quality", "entry_quality_exits", "full_controls", "suite"], default="suite")
    args = parser.parse_args()
    data_dir, output = Path(args.data_dir), Path(args.output)
    cached = all(list(data_dir.glob(f"*{symbol}*_M5.csv")) for symbol in SYMBOLS)
    terminal = os.getenv("VX_MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if args.force_download or not cached:
        frames, specs, names = download_history(SYMBOLS, data_dir, args.history_days, terminal)
    else:
        frames, specs, names = load_cached(SYMBOLS, data_dir)
    news = load_news(args.news_csv)
    output.mkdir(parents=True, exist_ok=True)
    profile_names = list(EXPERIMENTS) if args.profile == "suite" else [args.profile]
    reports = {}
    all_candidates = []
    for profile_name in profile_names:
        candidates = []
        trades, equity = simulate(frames, specs, news, args.initial_balance, args.slippage_points, EXPERIMENTS[profile_name], candidates)
        trades.to_csv(output / f"trades_{profile_name}.csv", index=False)
        equity.to_csv(output / f"equity_{profile_name}.csv", index=False)
        metrics = report_metrics(trades, equity, args.initial_balance)
        reports[profile_name] = {"config": asdict(EXPERIMENTS[profile_name]), "metrics": metrics}
        all_candidates.extend(candidates)
        print(json.dumps({"profile": profile_name, **metrics}, indent=2, default=str))
    pd.DataFrame(all_candidates).to_csv(output / "candidates.csv", index=False)
    report = {"settings": {**vars(args), "symbols": SYMBOLS, "risk_by_symbol": RISK_BY_SYMBOL, "commission_round_turn_per_lot": COMMISSION_PER_LOT_ROUND_TURN, "daily_loss_pct": DAILY_LOSS_PCT, "total_loss_pct": TOTAL_LOSS_PCT}, "broker_symbols": names, "experiments": reports, "limitations": ["News blackout is applied only when --news-csv is supplied; live calendar data is not a historical archive.", "Same-bar SL and TP is resolved conservatively as SL.", "MT5 bar data cannot reproduce tick ordering; slippage is modeled as a fixed point cost.", "Swap values depend on broker symbol settings and are captured at download time.", "The experiment suite changes one group of controls at a time; it is for research and does not alter live execution."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
