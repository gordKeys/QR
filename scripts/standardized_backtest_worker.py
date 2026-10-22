"""Worker for standardized_backtest.py.

Runs inside a detached worktree so each commit supplies its own strategy code,
while the orchestrator supplies one frozen data snapshot and one configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import research_backtest as backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-balance", type=float, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--risk-json", required=True)
    parser.add_argument("--daily-loss-pct", type=float, required=True)
    parser.add_argument("--total-loss-pct", type=float, required=True)
    parser.add_argument("--max-open-positions", type=int, required=True)
    parser.add_argument("--walkforward-window-days", type=int, required=True)
    parser.add_argument("--walkforward-step-days", type=int, required=True)
    parser.add_argument("--walkforward-windows", type=int, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    return parser.parse_args()


def run(frames, specs, news, initial, config, execution_frames, start=None, end=None):
    if start is not None:
        frames = {
            symbol: frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
            for symbol, frame in frames.items()
        }
    candidates = []
    selected_execution = execution_frames
    if start is not None:
        selected_execution = {
            symbol: frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
            for symbol, frame in execution_frames.items()
        }
    trades, equity = backtest.simulate(
        frames,
        specs,
        news,
        initial,
        backtest.SLIPPAGE_POINTS,
        config,
        candidates,
        execution_frames=selected_execution,
    )
    return trades, equity


def main() -> None:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    risk = json.loads(args.risk_json)
    backtest.SYMBOLS = symbols
    backtest.RISK_BY_SYMBOL = risk
    backtest.DAILY_LOSS_PCT = args.daily_loss_pct
    backtest.TOTAL_LOSS_PCT = args.total_loss_pct
    backtest.MAX_OPEN_POSITIONS = args.max_open_positions

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames, specs, names = backtest.load_cached(symbols, data_dir)
    execution_frames = {}
    for symbol in symbols:
        paths = sorted(data_dir.glob(f"*{symbol}*_M1.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing frozen M1 execution data for {symbol}: {data_dir}")
        frame = pd.read_csv(paths[0])
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        execution_frames[symbol] = frame.set_index("time").sort_index()
    news = pd.DataFrame(columns=["time", "currency", "title"])
    config = backtest.ExperimentConfig(
        "standardized_live_profile",
        regime_gate=False,
        m30_gate=False,
        session_guard=True,
        cost_reward_gate=False,
        no_chase_gate=False,
        exit_management=False,
        max_open_positions=args.max_open_positions,
        daily_soft_stop_pct=0.03,
    )

    trades, equity = run(frames, specs, news, args.initial_balance, config, execution_frames)
    trades.to_csv(output / "trades_full.csv", index=False)
    equity.to_csv(output / "equity_full.csv", index=False)
    report = {
        "symbols": symbols,
        "broker_symbols": names,
        "risk_by_symbol": risk,
        "config": backtest.asdict(config),
        "full_sample": backtest.report_metrics(trades, equity, args.initial_balance),
        "walkforward": [],
        "limitations": [
            "Research engine uses M5 bars, not tick data.",
            "Historical news is not included unless the engine is extended with a frozen news file.",
            "Live GBPUSD break-even and dual-XAU execution are not represented by older research engines.",
        ],
    }

    start = max(frame.index.min() for frame in frames.values())
    end = min(frame.index.max() for frame in frames.values())
    cursor = start
    for window_number in range(args.walkforward_windows):
        window_end = cursor + pd.Timedelta(days=args.walkforward_window_days)
        if window_end > end:
            break
        window_trades, window_equity = run(
            frames,
            specs,
            news,
            args.initial_balance,
            config,
            execution_frames,
            cursor,
            window_end,
        )
        window_output = output / f"walkforward_{window_number + 1:02d}"
        window_output.mkdir(exist_ok=True)
        window_trades.to_csv(window_output / "trades.csv", index=False)
        window_equity.to_csv(window_output / "equity.csv", index=False)
        report["walkforward"].append(
            {
                "window": window_number + 1,
                "start": cursor,
                "end": window_end,
                "metrics": backtest.report_metrics(
                    window_trades, window_equity, args.initial_balance
                ),
            }
        )
        cursor += pd.Timedelta(days=args.walkforward_step_days)

    (output / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
