"""Research-only concurrent XAUUSD current-versus-trailing portfolio test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_trades(current_report: Path, trailing_csv: Path):
    current_report_data = json.loads(current_report.read_text(encoding="utf-8"))
    current = pd.DataFrame(current_report_data["symbols"]["XAUUSD"]["trades"])
    current["strategy"] = "current"
    trailing = pd.read_csv(trailing_csv)
    trailing["strategy"] = "trailing_breakout"
    columns = ["strategy", "entry_time", "exit_time", "pnl"]
    trades = pd.concat([current[columns], trailing[columns]], ignore_index=True)
    for column in ("entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True, format="mixed")
    trades["pnl"] = trades["pnl"].astype(float)
    return trades.sort_values(["entry_time", "exit_time", "strategy"]).reset_index(drop=True)


def replay(trades: pd.DataFrame, initial_balance: float, max_positions: int):
    standalone = {"current": initial_balance, "trailing_breakout": initial_balance}
    active = []
    balance = initial_balance
    rows = []
    skipped = 0
    for trade in trades.itertuples(index=False):
        # Close any positions that have ended by this entry time.
        still_active = []
        for position in active:
            if position["exit_time"] <= trade.entry_time:
                balance += position["pnl"] * balance / position["standalone_balance"]
                rows.append({"time": position["exit_time"], "balance": balance, "event": "exit", "strategy": position["strategy"]})
                standalone[position["strategy"]] += position["pnl"]
            else:
                still_active.append(position)
        active = still_active
        if len(active) >= max_positions:
            skipped += 1
            continue
        strategy_balance = standalone[trade.strategy]
        active.append({"strategy": trade.strategy, "entry_time": trade.entry_time, "exit_time": trade.exit_time, "pnl": trade.pnl, "standalone_balance": strategy_balance})
        rows.append({"time": trade.entry_time, "balance": balance, "event": "entry", "strategy": trade.strategy})
    for position in sorted(active, key=lambda item: item["exit_time"]):
        balance += position["pnl"] * balance / position["standalone_balance"]
        rows.append({"time": position["exit_time"], "balance": balance, "event": "exit", "strategy": position["strategy"]})
    curve = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    if curve.empty:
        return {"end_balance": initial_balance, "net_pnl": 0.0, "return_pct": 0.0, "trades": 0, "skipped_entries": skipped, "max_drawdown_pct": 0.0, "daily_loss_pct": 0.0, "daily_breached": False, "total_breached": False}, curve
    exit_curve = curve[curve["event"] == "exit"].copy()
    equity = exit_curve["balance"]
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    daily_start = exit_curve.groupby(exit_curve["time"].dt.date)["balance"].transform("first")
    daily_loss = exit_curve["balance"] / daily_start - 1.0
    return {
        "end_balance": float(balance),
        "net_pnl": float(balance - initial_balance),
        "return_pct": float((balance / initial_balance - 1.0) * 100.0),
        "trades": int(len(exit_curve)),
        "skipped_entries": int(skipped),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "max_daily_loss_pct": float(daily_loss.min() * 100.0),
        "daily_breached": bool((daily_loss <= -5.0 / 100.0).any()),
        "total_breached": bool((drawdown <= -10.0 / 100.0).any()),
    }, curve


def walkforward(trades, initial_balance, max_positions, days=30, step=15):
    start = trades["entry_time"].min().floor("D")
    end = trades["exit_time"].max()
    windows = []
    while start < end:
        window_end = start + pd.Timedelta(days=days)
        subset = trades[(trades["entry_time"] >= start) & (trades["exit_time"] < window_end)]
        if len(subset) >= 2:
            result, _ = replay(subset, initial_balance, max_positions)
            windows.append({"start": start.isoformat(), "end": window_end.isoformat(), **{key: result[key] for key in ("return_pct", "trades", "max_drawdown_pct", "daily_breached", "total_breached")}})
        start += pd.Timedelta(days=step)
    returns = [window["return_pct"] for window in windows]
    return {
        "windows": windows,
        "mean_return_pct": float(np.mean(returns)) if returns else 0.0,
        "median_return_pct": float(np.median(returns)) if returns else 0.0,
        "profitable_windows": int(sum(value > 0 for value in returns)),
        "daily_pass_windows": int(sum(not window["daily_breached"] for window in windows)),
        "total_pass_windows": int(sum(not window["total_breached"] for window in windows)),
    }


def random_walkforward(trades, initial_balance, max_positions, windows=10, days=30, step=15, seed=26072601):
    starts = pd.date_range(
        trades["entry_time"].min().floor("D"),
        trades["entry_time"].max().floor("D") - pd.Timedelta(days=days),
        freq=f"{step}D",
        tz="UTC",
    )
    if len(starts) == 0:
        return {"seed": seed, "window_days": days, "step_days": step, "windows": [], "mean_return_pct": 0.0, "median_return_pct": 0.0, "profitable_windows": 0, "daily_pass_windows": 0, "total_pass_windows": 0}
    rng = np.random.default_rng(seed)
    selected = rng.choice(starts, size=min(windows, len(starts)), replace=False)
    results = []
    for start in sorted(selected):
        end = start + pd.Timedelta(days=days)
        subset = trades[(trades["entry_time"] >= start) & (trades["exit_time"] < end)]
        if len(subset) < 2:
            continue
        result, _ = replay(subset, initial_balance, max_positions)
        results.append({"start": start.isoformat(), "end": end.isoformat(), **{key: result[key] for key in ("return_pct", "trades", "max_drawdown_pct", "daily_breached", "total_breached")}})
    returns = [item["return_pct"] for item in results]
    return {
        "seed": seed,
        "window_days": days,
        "step_days": step,
        "windows": results,
        "mean_return_pct": float(np.mean(returns)) if returns else 0.0,
        "median_return_pct": float(np.median(returns)) if returns else 0.0,
        "profitable_windows": int(sum(value > 0 for value in returns)),
        "daily_pass_windows": int(sum(not item["daily_breached"] for item in results)),
        "total_pass_windows": int(sum(not item["total_breached"] for item in results)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-report", default="results/xau_current_compare/report.json")
    parser.add_argument("--trailing-trades", default="results/xau_trailing_compare/trades_breakout_trailing_london_ny.csv")
    parser.add_argument("--output", default="results/xau_dual_live_compare")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--random-windows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=26072601)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trades = load_trades(Path(args.current_report), Path(args.trailing_trades))
    result, curve = replay(trades, args.initial_balance, args.max_positions)
    result["walkforward"] = walkforward(trades, args.initial_balance, args.max_positions)
    result["random_walkforward"] = random_walkforward(
        trades,
        args.initial_balance,
        args.max_positions,
        windows=args.random_windows,
        days=30,
        step=15,
        seed=args.seed,
    )
    result["method"] = "two concurrent strategy streams; close-to-close combined equity"
    result["risk_note"] = "Each stream was calibrated at 1% standalone risk; concurrent operation can expose up to approximately 2% before overlap/cost effects."
    (output / "trades_combined.csv").write_text(trades.to_csv(index=False), encoding="utf-8")
    (output / "equity_events.csv").write_text(curve.to_csv(index=False), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
