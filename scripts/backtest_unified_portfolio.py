"""Unified research backtest for EURUSD, GBPUSD, and dual XAUUSD.

This script is deliberately research-only. It uses one timeline, one equity
account, one global position cap, broker-style bid/ask costs, and Prague-time
FTMO diagnostics. Historical news is required for a fully validated run; when
omitted, the report is explicitly marked incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bootstrap import add_project_root

add_project_root()

from engine.features import FeatureEngine
from strategies.ftmo_xauusd import FTMOXAUUSD
from strategies.mean_reversion import MeanReversion


PRAGUE = ZoneInfo("Europe/Prague")
FTMO_DAILY_PCT = 0.05
FTMO_TOTAL_PCT = 0.10
COMMISSION_PER_LOT_ROUND_TURN = 5.0
SLIPPAGE_POINTS = 1.0


@dataclass(frozen=True)
class Profile:
    name: str
    eur_mode: str
    gbp_mode: str
    eur_risk: float = 0.015
    gbp_risk: float = 0.015
    xau_current_risk: float = 0.005
    xau_trailing_risk: float = 0.005
    max_positions: int = 2


PROFILES = (
    Profile("profile_a_relaxed_fx_dual_xau", "relaxed", "relaxed"),
    Profile("profile_b_current_fx_dual_xau", "current", "current"),
    Profile("profile_c_relaxed_eur_current_gbp_dual_xau", "relaxed", "current"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_column = "time" if "time" in frame.columns else frame.columns[0]
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    frame = frame.rename(columns={time_column: "time"}).set_index("time").sort_index()
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "spread" not in frame.columns:
        frame["spread"] = 0.0
    if "tick_volume" not in frame.columns:
        frame["tick_volume"] = 0.0
    return FeatureEngine().add_features(frame)


def load_news(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["time", "currency", "title"])
    news = pd.read_csv(path)
    required = {"time", "currency"}
    if not required.issubset(news.columns):
        raise ValueError("news CSV must contain time,currency columns")
    news["time"] = pd.to_datetime(news["time"], utc=True)
    return news


def news_blocked(news: pd.DataFrame, symbol: str, timestamp: pd.Timestamp) -> bool:
    if news.empty:
        return False
    currencies = {symbol[:3], symbol[3:]} if len(symbol) == 6 else {"USD"}
    if symbol == "XAUUSD":
        currencies = {"USD"}
    relevant = news[news.currency.str.upper().isin(currencies)]
    delta = (relevant.time - timestamp).dt.total_seconds() / 60.0
    return bool(((delta >= -15) & (delta <= 30)).any())


def session_allowed(timestamp: pd.Timestamp) -> bool:
    hour = timestamp.tz_convert("Africa/Accra").hour
    return not (21 <= hour < 23)


def spread_price(frame: pd.DataFrame, index: int, point: float) -> float:
    value = float(frame.iloc[index].get("spread", 0.0) or 0.0)
    return max(value, 0.0) * point


def atr(frame: pd.DataFrame, index: int, period: int = 20) -> float:
    previous = frame.close.shift(1)
    true_range = pd.concat(
        [(frame.high - frame.low), (frame.high - previous).abs(), (frame.low - previous).abs()],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(period, min_periods=period).mean().iloc[index]
    return float(value) if np.isfinite(value) else 0.0


def xau_trailing_signal(frame: pd.DataFrame, index: int):
    if index < 25:
        return 0, None
    row = frame.iloc[index]
    current_atr = atr(frame, index)
    prior_high = frame.high.iloc[index - 20:index].max()
    prior_low = frame.low.iloc[index - 20:index].min()
    if current_atr <= 0:
        return 0, None
    if row.close > prior_high and row.close > row.open:
        stop = frame.low.iloc[index - 10:index].min() - 0.15 * current_atr
        if stop >= row.close - 0.5 * current_atr:
            stop = row.close - 1.5 * current_atr
        return 1, (float(stop), float(row.close + 2.0 * (row.close - stop)), current_atr)
    if row.close < prior_low and row.close < row.open:
        stop = frame.high.iloc[index - 10:index].max() + 0.15 * current_atr
        if stop <= row.close + 0.5 * current_atr:
            stop = row.close + 1.5 * current_atr
        return -1, (float(stop), float(row.close - 2.0 * (stop - row.close)), current_atr)
    return 0, None


def signal_and_plan(symbol, frame, index, mode, precomputed_signal=None):
    row = frame.iloc[index]
    current_atr = atr(frame, index)
    if current_atr <= 0:
        return 0, None, "missing_atr"
    if symbol == "EURUSD":
        signal = int(precomputed_signal if precomputed_signal is not None else MeanReversion(lookback=20, entry_z=2.0 if mode == "current" else 1.5).generate_signals(frame).iloc[index])
        sl_atr, tp_atr = 2.5, 5.0
    elif symbol == "GBPUSD":
        signal = int(precomputed_signal if precomputed_signal is not None else MeanReversion(lookback=20, entry_z=2.0 if mode == "current" else 1.5).generate_signals(frame).iloc[index])
        sl_atr, tp_atr = 2.5, 4.0
    else:
        signal = int(precomputed_signal if precomputed_signal is not None else FTMOXAUUSD(min_score=4).generate_signals(frame).iloc[index])
        sl_atr, tp_atr = 1.5, 2.25
    if not signal:
        return 0, None, "no_signal"
    entry = float(row.close)
    stop = entry - signal * sl_atr * current_atr
    target = entry + signal * tp_atr * current_atr
    return signal, (entry, stop, target, current_atr), "ok"


def run_profile(profile: Profile, frames, news, initial_balance: float):
    point = {"EURUSD": 0.00001, "GBPUSD": 0.00001, "XAUUSD": 0.01}
    contract = {"EURUSD": 100000.0, "GBPUSD": 100000.0, "XAUUSD": 100.0}
    risk = {"EURUSD": profile.eur_risk, "GBPUSD": profile.gbp_risk}
    balance = initial_balance
    peak = initial_balance
    daily_base = initial_balance
    current_day = None
    positions = []
    trades = []
    equity_rows = []
    timeline = sorted(set().union(*(frame.index for frame in frames.values())))
    signal_cache = {}
    for symbol, mode in (("EURUSD", profile.eur_mode), ("GBPUSD", profile.gbp_mode)):
        signal_cache[symbol] = MeanReversion(
            lookback=20,
            entry_z=2.0 if mode == "current" else 1.5,
        ).generate_signals(frames[symbol])
    signal_cache["XAUUSD"] = FTMOXAUUSD(min_score=4).generate_signals(frames["XAUUSD"])

    for timestamp in timeline:
        day = timestamp.tz_convert(PRAGUE).date()
        if day != current_day:
            current_day, daily_base = day, balance
        for position in list(positions):
            frame = frames[position["symbol"]]
            if timestamp not in frame.index:
                continue
            row = frame.loc[timestamp]
            direction = position["direction"]
            stop_hit = row.low <= position["stop"] if direction == 1 else row.high >= position["stop"]
            target_hit = row.high >= position["target"] if direction == 1 else row.low <= position["target"]
            if not stop_hit and not target_hit:
                continue
            reason = "SL" if stop_hit else "TP"
            exit_price = position["stop"] if stop_hit else position["target"]
            gross = (exit_price - position["entry"]) * direction * position["volume"] * contract[position["symbol"]]
            spread_cost = spread_price(frame, frame.index.get_loc(timestamp), point[position["symbol"]]) * position["volume"] * contract[position["symbol"]]
            commission = COMMISSION_PER_LOT_ROUND_TURN * position["volume"]
            slippage = SLIPPAGE_POINTS * point[position["symbol"]] * position["volume"] * contract[position["symbol"]]
            pnl = gross - spread_cost - commission - slippage
            balance += pnl
            trades.append({**position, "exit_time": timestamp, "exit": exit_price, "reason": reason, "pnl": pnl})
            positions.remove(position)

        floating = 0.0
        for position in positions:
            frame = frames[position["symbol"]]
            if timestamp in frame.index:
                price = float(frame.loc[timestamp].close)
                floating += (price - position["entry"]) * position["direction"] * position["volume"] * contract[position["symbol"]]
        equity = balance + floating
        peak = max(peak, equity)
        equity_rows.append({"time": timestamp, "balance": balance, "equity": equity, "daily_base": daily_base, "peak": peak})

        for symbol, frame in frames.items():
            if timestamp not in frame.index or len(positions) >= profile.max_positions:
                continue
            if any(position["symbol"] == symbol for position in positions):
                continue
            if not session_allowed(timestamp) or news_blocked(news, symbol, timestamp):
                continue
            index = frame.index.get_loc(timestamp)
            if symbol in {"EURUSD", "GBPUSD"}:
                mode = profile.eur_mode if symbol == "EURUSD" else profile.gbp_mode
                signal, plan, reason = signal_and_plan(symbol, frame, index, mode, signal_cache[symbol].iloc[index])
                if not signal:
                    continue
                entry, stop, target, entry_atr = plan
                strategy_name = f"mean_reversion_{mode}"
                position_risk = risk[symbol]
            elif symbol == "XAUUSD":
                signal, plan, reason = signal_and_plan(symbol, frame, index, "current", signal_cache[symbol].iloc[index])
                candidates = []
                if signal and plan:
                    candidates.append(("xau_current", signal, plan, profile.xau_current_risk))
                trail_signal, trail_plan = xau_trailing_signal(frame, index)
                if trail_signal and trail_plan:
                    candidates.append(("xau_trailing", trail_signal, (float(frame.iloc[index].close), trail_plan[0], trail_plan[1], trail_plan[2]), profile.xau_trailing_risk))
                for strategy_name, direction, strategy_plan, position_risk in candidates:
                    if len(positions) >= profile.max_positions:
                        break
                    entry, stop, target, entry_atr = strategy_plan
                    volume = (equity * position_risk) / (abs(entry - stop) * contract[symbol])
                    positions.append({"symbol": symbol, "strategy": strategy_name, "direction": direction, "entry_time": timestamp, "entry": entry, "stop": stop, "target": target, "volume": volume, "risk_pct": position_risk, "entry_atr": entry_atr})
                continue
            volume = (equity * position_risk) / (abs(entry - stop) * contract[symbol])
            positions.append({"symbol": symbol, "strategy": strategy_name, "direction": signal, "entry_time": timestamp, "entry": entry, "stop": stop, "target": target, "volume": volume, "risk_pct": position_risk, "entry_atr": entry_atr})

    for position in sorted(positions, key=lambda value: value["entry_time"]):
        trades.append({**position, "exit_time": timeline[-1], "exit": float(frames[position["symbol"]].iloc[-1].close), "reason": "end_of_data", "pnl": 0.0})
    equity = pd.DataFrame(equity_rows)
    equity["day"] = equity.time.dt.tz_convert(PRAGUE).dt.date
    daily_min = equity.groupby("day").equity.min() - equity.groupby("day").equity.first()
    daily_pct = daily_min / equity.groupby("day").equity.first() * 100
    drawdown = equity.equity / equity.peak * 100 - 100
    return trades, equity, {
        "start_balance": initial_balance,
        "end_balance": float(balance),
        "net_pnl": float(balance - initial_balance),
        "return_pct": float((balance / initial_balance - 1) * 100),
        "trades": len(trades),
        "winning_trades": sum(trade["pnl"] > 0 for trade in trades),
        "losing_trades": sum(trade["pnl"] < 0 for trade in trades),
        "max_equity_drawdown_usd": float((equity.equity - equity.peak).min()),
        "max_equity_drawdown_pct": float(drawdown.min()),
        "max_daily_drawdown_usd": float(daily_min.min()),
        "max_daily_drawdown_pct": float(daily_pct.min()),
        "daily_max_loss_usd": initial_balance * FTMO_DAILY_PCT,
        "total_max_loss_usd": initial_balance * FTMO_TOTAL_PCT,
        "daily_ftmo_breached": bool((daily_min <= -initial_balance * FTMO_DAILY_PCT).any()),
        "total_ftmo_breached": bool((equity.equity <= initial_balance * (1 - FTMO_TOTAL_PCT)).any()),
        "ftmo_daily_pass": bool((daily_min > -initial_balance * FTMO_DAILY_PCT).all()),
        "ftmo_total_pass": bool((equity.equity > initial_balance * (1 - FTMO_TOTAL_PCT)).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--news-csv")
    args = parser.parse_args()
    data_dir, output = Path(args.data_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {symbol: data_dir / f"{symbol}_M5.csv" for symbol in ("EURUSD", "GBPUSD", "XAUUSD")}
    frames = {symbol: load_frame(path) for symbol, path in paths.items()}
    news = load_news(Path(args.news_csv) if args.news_csv else None)
    reports = {}
    for profile in PROFILES:
        trades, equity, report = run_profile(profile, frames, news, args.initial_balance)
        prefix = profile.name
        pd.DataFrame(trades).to_csv(output / f"trades_{prefix}.csv", index=False)
        equity.to_csv(output / f"equity_{prefix}.csv", index=False)
        reports[prefix] = {"settings": profile.__dict__, "metrics": report, "news_rows": len(news)}
    commit = "unknown"
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    manifest = {
        "code_commit": commit,
        "data_files": {symbol: {"path": str(path), "sha256": sha256(path), "rows": len(frames[symbol])} for symbol, path in paths.items()},
        "initial_balance": args.initial_balance,
        "ftmo_daily_loss_pct": FTMO_DAILY_PCT * 100,
        "ftmo_total_loss_pct": FTMO_TOTAL_PCT * 100,
        "commission_round_turn_per_lot": COMMISSION_PER_LOT_ROUND_TURN,
        "slippage_points": SLIPPAGE_POINTS,
        "news_file": str(args.news_csv) if args.news_csv else None,
        "news_rows": len(news),
        "validation_status": "incomplete_without_historical_news" if news.empty else "cost_and_news_inputs_present",
        "limitations": ["M5 bars cannot reconstruct tick ordering.", "Broker contract specs are explicit research defaults; verify against the downloaded broker specs before live use."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "profiles": reports}, indent=2))


if __name__ == "__main__":
    main()
