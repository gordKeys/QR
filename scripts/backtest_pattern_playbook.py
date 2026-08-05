from bootstrap import add_project_root
add_project_root()

import argparse
from pathlib import Path

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.pattern_backtester import PatternBacktester
from engine.symbol_universe import (
    PATTERN_RECOMMENDED_SYMBOLS,
    default_cost_profile,
    load_pattern_symbol_candidates,
)
from strategies.pattern_playbook import PatternPlaybookStrategy


def available_csv_symbols(data_dir):
    directory = Path(data_dir)
    if not directory.exists():
        return []
    return sorted({path.name.split("_")[0].upper() for path in directory.glob("*_M5.csv")})


def load_data_for_symbol(symbol, data_dir):
    loader = DataLoader(symbol=symbol, data_dir=data_dir)
    return FeatureEngine().add_features(loader.load())


def build_strategies(entry_style="breakout"):
    return {
        "triangle_breakout": PatternPlaybookStrategy(setup_mode="triangles", entry_style=entry_style, higher_timeframe="H4"),
        "head_shoulders": PatternPlaybookStrategy(setup_mode="head_shoulders", entry_style=entry_style, higher_timeframe="H4"),
        "double_triple": PatternPlaybookStrategy(setup_mode="double_triple", entry_style=entry_style, higher_timeframe="H4"),
        "mtf_reversal": PatternPlaybookStrategy(setup_mode="mtf_reversal", entry_style="breakout", higher_timeframe="H4"),
    }


def summarize_result(result):
    return result["net_profit"] * (1 + result["win_rate"]) / (1 + result["max_drawdown"] * 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", help="Optional symbol list. Defaults to the curated pattern universe.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=0.004)
    parser.add_argument("--commission-round-turn", type=float, default=7.0)
    parser.add_argument("--swap-long", type=float, default=0.0)
    parser.add_argument("--swap-short", type=float, default=0.0)
    parser.add_argument("--use-mt5-costs", action="store_true")
    parser.add_argument("--entry-style", choices=["breakout", "retest"], default="breakout")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    available = available_csv_symbols(args.data_dir)
    candidate_symbols = load_pattern_symbol_candidates(args.symbols, available_symbols=available)

    broker = None
    if args.use_mt5_costs:
        try:
            from mt5_broker_adapter import MT5BrokerAdapter

            broker = MT5BrokerAdapter()
            broker.initialize()
        except Exception as exc:
            print(f"MT5 cost lookup unavailable, using configured defaults: {exc}")
            broker = None

    strategies = build_strategies(entry_style=args.entry_style)
    rows = []

    print("\n=== PATTERN PLAYBOOK BACKTEST ===")
    print(f"Curated universe: {', '.join(PATTERN_RECOMMENDED_SYMBOLS)}")
    print(f"Available CSV symbols: {', '.join(available) if available else 'none'}")
    print(f"Testing symbols: {', '.join(candidate_symbols)}")
    print(f"Entry style: {args.entry_style}")
    print("-" * 108)

    for symbol in candidate_symbols:
        if symbol not in available:
            print(f"{symbol:>8} | skipped (no CSV data)")
            continue

        data = load_data_for_symbol(symbol, args.data_dir)

        if broker is not None:
            resolved_symbol = broker.resolve_symbol(symbol)
            info = broker.symbol_info(resolved_symbol) if resolved_symbol is not None else None
            if info is not None:
                cost_profile = default_cost_profile(symbol)
                cost_profile = cost_profile.__class__(
                    point=float(getattr(info, "point", cost_profile.point) or cost_profile.point),
                    contract_size=float(getattr(info, "trade_contract_size", cost_profile.contract_size) or cost_profile.contract_size),
                    commission_round_turn=args.commission_round_turn,
                    swap_long_per_lot_day=float(getattr(info, "swap_long", args.swap_long) or args.swap_long),
                    swap_short_per_lot_day=float(getattr(info, "swap_short", args.swap_short) or args.swap_short),
                )
            else:
                cost_profile = default_cost_profile(symbol)
        else:
            cost_profile = default_cost_profile(symbol)
            cost_profile = cost_profile.__class__(
                point=cost_profile.point,
                contract_size=cost_profile.contract_size,
                commission_round_turn=args.commission_round_turn,
                swap_long_per_lot_day=args.swap_long,
                swap_short_per_lot_day=args.swap_short,
            )

        for strategy_name, strategy in strategies.items():
            result = PatternBacktester(
                data=data,
                strategy=strategy,
                symbol=symbol,
                initial_balance=args.initial_balance,
                risk_per_trade=args.risk_pct,
                cost_profile=cost_profile,
            ).run()

            score = summarize_result(result)
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "final_balance": result["final_balance"],
                    "net_profit": result["net_profit"],
                    "trades": result["total_trades"],
                    "win_rate": result["win_rate"],
                    "profit_factor": result["profit_factor"],
                    "max_drawdown": result["max_drawdown"],
                    "avg_r": result["avg_r"],
                    "score": score,
                }
            )

            print(
                f"{symbol:>8} | {strategy_name:>18} | "
                f"bal={result['final_balance']:10.2f} | "
                f"net={result['net_profit']:9.2f} | "
                f"trades={result['total_trades']:4d} | "
                f"wr={result['win_rate']:.2%} | "
                f"pf={result['profit_factor']:.2f} | "
                f"dd={result['max_drawdown']:.2%}"
            )

    if broker is not None:
        try:
            broker.shutdown()
        except Exception:
            pass

    if not rows:
        print("\nNo results generated.")
        return

    print("\n=== BEST BY SYMBOL ===")
    by_symbol = {}
    for row in rows:
        current = by_symbol.get(row["symbol"])
        if current is None or row["score"] > current["score"]:
            by_symbol[row["symbol"]] = row
    for row in sorted(by_symbol.values(), key=lambda item: item["score"], reverse=True):
        print(
            f"{row['symbol']:>8} | best={row['strategy']:<18} | "
            f"score={row['score']:10.2f} | net={row['net_profit']:9.2f} | "
            f"dd={row['max_drawdown']:.2%} | wr={row['win_rate']:.2%}"
        )

    print("\n=== TOP SETUPS ===")
    for row in sorted(rows, key=lambda item: item["score"], reverse=True)[: args.top_n]:
        print(
            f"{row['symbol']:>8} | {row['strategy']:<18} | "
            f"score={row['score']:10.2f} | net={row['net_profit']:9.2f} | "
            f"trades={row['trades']:4d} | wr={row['win_rate']:.2%} | dd={row['max_drawdown']:.2%}"
        )

    print("\n=== SYMBOL RECOMMENDATION ===")
    symbol_rank = sorted(by_symbol.values(), key=lambda item: item["score"], reverse=True)
    print(", ".join(row["symbol"] for row in symbol_rank[: args.top_n]))


if __name__ == "__main__":
    main()
