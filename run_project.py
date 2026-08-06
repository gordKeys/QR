import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(script, extra_args=None):
    cmd = [PYTHON, str(ROOT / "scripts" / script)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=str(ROOT))


def print_status(mode, args):
    print("\n=== QuantFX Launcher ===")
    print(f"Mode: {mode}")
    print(f"Project root: {ROOT}")
    if args.symbol:
        print(f"Symbols: {', '.join(args.symbol)}")
    elif getattr(args, "symbols", None):
        print(f"Symbols: {', '.join(args.symbols)}")
    if args.data:
        print(f"Data files: {', '.join(args.data)}")
    if mode == "live":
        print(f"Dry run: {args.dry_run}")
        print(f"Loop once: {args.loop_once}")
        print(f"Max consecutive losses: {args.max_consecutive_losses or 3}")
    print("======================\n")


def main():
    parser = argparse.ArgumentParser(description="QuantFX project launcher")
    parser.add_argument(
        "mode",
        choices=["test", "walkforward", "sweep", "combo", "mtf", "patterns", "download", "live"],
        help="Choose what to run",
    )
    parser.add_argument("--symbol", action="append", help="Repeatable symbol filter")
    parser.add_argument("--symbols", nargs="+", action="extend", help="Symbol list for mtf/patterns")
    parser.add_argument("--data", action="append", help="Repeatable CSV path filter")
    parser.add_argument("--dry-run", action="store_true", help="Live mode only")
    parser.add_argument("--loop-once", action="store_true", help="Live mode only")
    parser.add_argument("--max-consecutive-losses", type=int, help="Live mode only")
    parser.add_argument("--revenge-mode", action="store_true", help="Live mode only")
    parser.add_argument("--revenge-multiplier", type=float, help="Live mode only")
    parser.add_argument("--revenge-gap-trades", type=int, help="Live mode only")
    parser.add_argument("--revenge-boosts", type=int, help="Live mode only")
    parser.add_argument("--hard-drawdown-switch", action="store_true", help="Live mode only")
    parser.add_argument("--hard-drawdown-usd", type=float, help="Live mode only")
    parser.add_argument("--break-even-trigger-usd", type=float, help="Live mode only")
    parser.add_argument("--break-even-commission-round-turn", type=float, help="Live mode only")
    parser.add_argument("--use-mt5-costs", action="store_true", help="MTF/patterns mode only")
    parser.add_argument("--entry-style", choices=["breakout", "retest"], help="Patterns mode only")
    parser.add_argument("--strategies", nargs="+", help="Patterns mode only")
    parser.add_argument("--pattern-timeframe", choices=["15min", "30min", "1h", "4h"], help="Patterns mode only")
    parser.add_argument("--top-n", type=int, help="Patterns mode only")
    args = parser.parse_args()

    forwarded = []
    for item in args.symbol or []:
        forwarded.extend(["--symbol", item])
    for item in args.data or []:
        forwarded.extend(["--data", item])
    for item in args.symbols or []:
        forwarded.extend(["--symbols", item])

    if args.mode == "test":
        print_status(args.mode, args)
        return run_script("evaluate_multi_symbol.py", forwarded)

    if args.mode == "walkforward":
        print_status(args.mode, args)
        return run_script("walkforward_multi_symbol.py", forwarded)

    if args.mode == "sweep":
        print_status(args.mode, args)
        return run_script("sweep_mean_reversion.py", forwarded)

    if args.mode == "combo":
        print_status(args.mode, args)
        return run_script("run_symbol_combo.py", forwarded)

    if args.mode == "mtf":
        print_status(args.mode, args)
        if args.use_mt5_costs:
            forwarded.append("--use-mt5-costs")
        return run_script("backtest_multi_timeframe.py", forwarded)

    if args.mode == "patterns":
        print_status(args.mode, args)
        if args.use_mt5_costs:
            forwarded.append("--use-mt5-costs")
        if args.entry_style is not None:
            forwarded.extend(["--entry-style", args.entry_style])
        if args.strategies is not None:
            forwarded.extend(["--strategies", *args.strategies])
        if args.pattern_timeframe is not None:
            forwarded.extend(["--pattern-timeframe", args.pattern_timeframe])
        if args.top_n is not None:
            forwarded.extend(["--top-n", str(args.top_n)])
        return run_script("backtest_pattern_playbook.py", forwarded)

    if args.mode == "download":
        print_status(args.mode, args)
        return run_script("download_mt5_history.py", forwarded)

    if args.mode == "live":
        print_status(args.mode, args)
        live_args = forwarded[:]
        if args.dry_run:
            live_args.append("--dry-run")
        if args.loop_once:
            live_args.append("--loop-once")
        if args.max_consecutive_losses is not None:
            live_args.extend(["--max-consecutive-losses", str(args.max_consecutive_losses)])
        if args.revenge_mode:
            live_args.append("--revenge-mode")
        if args.revenge_multiplier is not None:
            live_args.extend(["--revenge-multiplier", str(args.revenge_multiplier)])
        if args.revenge_gap_trades is not None:
            live_args.extend(["--revenge-gap-trades", str(args.revenge_gap_trades)])
        if args.revenge_boosts is not None:
            live_args.extend(["--revenge-boosts", str(args.revenge_boosts)])
        if args.hard_drawdown_switch:
            live_args.append("--hard-drawdown-switch")
        if args.hard_drawdown_usd is not None:
            live_args.extend(["--hard-drawdown-usd", str(args.hard_drawdown_usd)])
        if args.break_even_trigger_usd is not None:
            live_args.extend(["--break-even-trigger-usd", str(args.break_even_trigger_usd)])
        if args.break_even_commission_round_turn is not None:
            live_args.extend(["--break-even-commission-round-turn", str(args.break_even_commission_round_turn)])
        if not live_args:
            live_args = ["--symbols", "EURUSD", "GBPUSD", "USDJPY"]
        return run_script("live_runner.py", live_args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
