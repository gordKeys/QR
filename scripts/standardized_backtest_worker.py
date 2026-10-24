"""Worker for standardized_backtest.py.

Runs inside a detached worktree so each commit supplies its own strategy code,
while the orchestrator supplies one frozen data snapshot and one configuration.
"""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--soft-stop-pct", type=float, required=True)
    parser.add_argument("--max-open-positions", type=int, required=True)
    parser.add_argument("--walkforward-window-days", type=int, required=True)
    parser.add_argument("--walkforward-step-days", type=int, required=True)
    parser.add_argument("--walkforward-windows", type=int, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--profile", choices=("live", "ny_blocked", "gbp_london_quality", "ny_tp_1atr", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause", "diagnostic"), default="live")
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
    return trades, equity, pd.DataFrame(candidates)


def candidate_diagnostics(candidates: pd.DataFrame) -> dict:
    if candidates.empty:
        return {"candidates": 0, "by_symbol": {}, "by_symbol_session_reason": []}
    by_symbol = {}
    for symbol, group in candidates.groupby("symbol"):
        by_symbol[symbol] = {
            "candidates": int(len(group)),
            "accepted": int(group["accepted"].sum()),
            "rejected": int((~group["accepted"]).sum()),
            "rejection_reasons": group.loc[~group["accepted"], "rejection_reason"].value_counts().to_dict(),
        }
    grouped = (
        candidates.groupby(["symbol", "session", "rejection_reason"])
        .size()
        .reset_index(name="count")
    )
    return {
        "candidates": int(len(candidates)),
        "by_symbol": by_symbol,
        "by_symbol_session_reason": grouped.to_dict("records"),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
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
    print(
        f"starting worker; symbols={','.join(symbols)}; "
        f"walk-forward windows requested={args.walkforward_windows}",
        flush=True,
    )
    frames, specs, names = backtest.load_cached(symbols, data_dir)
    router = backtest.StrategyRouter()
    signal_counts = {
        symbol: int((router.get_strategy(symbol).generate_signals(frames[symbol]) != 0).sum())
        for symbol in symbols
    }
    print(
        "loaded M5 data: "
        + ", ".join(f"{symbol}={len(frames[symbol])} bars" for symbol in symbols),
        flush=True,
    )
    print(f"raw strategy signal counts: {signal_counts}", flush=True)
    execution_frames = {}
    for symbol in symbols:
        paths = sorted(data_dir.glob(f"*{symbol}*_M1.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing frozen M1 execution data for {symbol}: {data_dir}")
        frame = pd.read_csv(paths[0])
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        execution_frames[symbol] = frame.set_index("time").sort_index()
    print(
        "loaded M1 execution data: "
        + ", ".join(f"{symbol}={len(execution_frames[symbol])} bars" for symbol in symbols),
        flush=True,
    )
    news = pd.DataFrame(columns=["time", "currency", "title"])
    config = backtest.ExperimentConfig(
        f"standardized_{args.profile}_profile",
        session_guard=True,
        cost_reward_gate=False,
        no_chase_gate=False,
        exit_management=False,
        max_open_positions=args.max_open_positions,
        daily_soft_stop_pct=args.soft_stop_pct,
        new_york_tp_atr=1.0 if args.profile == "ny_tp_1atr" else None,
        allowed_sessions_by_symbol=(
            (("EURUSD", ("Asia", "London")), ("GBPUSD", ("London",)))
            if args.profile == "target_session_mix" else
            (("EURUSD", ("Asia",)), ("GBPUSD", ("London",)), ("XAUUSD", ("NewYork",)))
            if args.profile in {"target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"} else ()
        ),
        session_tp_atr_overrides=(
            (("EURUSD", "London", 1.5),)
            if args.profile == "target_session_mix" else ()
        ),
        continue_after_ftmo_breach=args.diagnostic or args.profile == "diagnostic",
        loss_pause_after=3 if args.profile == "target_session_mix_xau_ny_pause" else 2,
        loss_pause_hours=3,
        regime_gate=args.profile in {"gbp_london_quality", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"},
        m30_gate=args.profile in {"gbp_london_quality", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"},
        block_new_york_symbols=("GBPUSD",) if args.profile == "gbp_london_quality" else (("EURUSD", "GBPUSD") if args.profile in {"ny_blocked", "target_session_mix"} else ()),
        regime_symbols=("GBPUSD",) if args.profile in {"gbp_london_quality", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"} else (),
        m30_symbols=("GBPUSD",) if args.profile in {"gbp_london_quality", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"} else (),
        spread_guard_symbols=("GBPUSD",) if args.profile in {"gbp_london_quality", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"} else (),
        max_spread_atr_ratio=0.20 if args.profile in {"gbp_london_quality", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause"} else None,
    )

    print("full-sample backtest started", flush=True)
    full_started = time.perf_counter()
    trades, equity, candidates = run(frames, specs, news, args.initial_balance, config, execution_frames)
    print(
        f"full-sample backtest complete: trades={len(trades)} "
        f"elapsed={time.perf_counter() - full_started:.1f}s",
        flush=True,
    )
    trades.to_csv(output / "trades_full.csv", index=False)
    equity.to_csv(output / "equity_full.csv", index=False)
    if not candidates.empty:
        candidates.to_csv(output / "candidates_full.csv", index=False)
    report = {
        "symbols": symbols,
        "broker_symbols": names,
        "risk_by_symbol": risk,
        "config": backtest.asdict(config),
        "raw_strategy_signal_counts": signal_counts,
        "full_sample": backtest.report_metrics(trades, equity, args.initial_balance),
        "candidate_diagnostics": candidate_diagnostics(candidates),
        "walkforward": [],
        "limitations": [
            "Research engine uses M5 bars, not tick data.",
            "Historical news is not included unless the engine is extended with a frozen news file.",
            "Live GBPUSD break-even and dual-XAU execution are not represented by older research engines.",
            "XAUUSD trades require the branch XAU strategy's signal and plan guards; raw signal counts and candidate rejections are recorded for diagnosis.",
        ],
        "diagnostic_mode": bool(config.continue_after_ftmo_breach),
    }

    start = max(frame.index.min() for frame in frames.values())
    end = min(frame.index.max() for frame in frames.values())
    cursor = start
    for window_number in range(args.walkforward_windows):
        window_end = cursor + pd.Timedelta(days=args.walkforward_window_days)
        if window_end > end:
            print(
                f"walk-forward window {window_number + 1}: skipped; "
                f"{cursor} to {window_end} exceeds data end {end}",
                flush=True,
            )
            break
        window_started = time.perf_counter()
        print(
            f"walk-forward window {window_number + 1}/{args.walkforward_windows} "
            f"started: {cursor} to {window_end}",
            flush=True,
        )
        window_trades, window_equity, window_candidates = run(
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
        if not window_candidates.empty:
            window_candidates.to_csv(window_output / "candidates.csv", index=False)
        report["walkforward"].append(
            {
                "window": window_number + 1,
                "start": cursor,
                "end": window_end,
                "metrics": backtest.report_metrics(
                    window_trades, window_equity, args.initial_balance
                ),
                "candidate_diagnostics": candidate_diagnostics(window_candidates),
            }
        )
        print(
            f"walk-forward window {window_number + 1}/{args.walkforward_windows} "
            f"complete: trades={len(window_trades)} "
            f"elapsed={time.perf_counter() - window_started:.1f}s",
            flush=True,
        )
        cursor += pd.Timedelta(days=args.walkforward_step_days)

    (output / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"worker complete: report={output / 'report.json'} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
