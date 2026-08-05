from bootstrap import add_project_root
add_project_root()

import argparse

from engine.backtester import Backtester
from strategy_batch_tools import load_symbol_data, resolve_symbol_inputs, infer_symbol_from_path
from strategy_router import StrategyRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append", help="Symbol name like EURUSD. Repeatable.")
    parser.add_argument("--data", action="append", help="CSV path like data/EURUSD_M5.csv. Repeatable.")
    args = parser.parse_args()

    symbol_inputs = []
    if args.data:
        symbol_inputs.extend(args.data)
    if args.symbol:
        symbol_inputs.extend(args.symbol)

    inputs = resolve_symbol_inputs(symbol_inputs or None)
    router = StrategyRouter()

    rows = []
    for item in inputs:
        if item.endswith(".csv"):
            symbol_name = infer_symbol_from_path(item)
            data = load_symbol_data(data_path=item)
        else:
            symbol_name = item
            data = load_symbol_data(symbol=item)

        strategy = router.get_strategy(symbol_name)
        strategy_name = router.get_strategy_name(symbol_name)

        result = Backtester(data, strategy).run()
        rows.append(
            {
                "symbol": symbol_name,
                "strategy": strategy_name,
                "final_balance": result["final_balance"],
                "total_trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "avg_r": result["avg_r"],
            }
        )

    total_balance = sum(row["final_balance"] for row in rows)

    print("\n=== SYMBOL COMBO RESULTS ===")
    print(f"{'symbol':>10} | {'strategy':>18} | {'balance':>12} | {'trades':>6} | {'win_rate':>8} | {'avg_r':>8}")
    print("-" * 82)
    for row in rows:
        print(
            f"{row['symbol']:>10} | "
            f"{row['strategy']:>18} | "
            f"{row['final_balance']:12.2f} | "
            f"{row['total_trades']:6d} | "
            f"{row['win_rate']:8.2%} | "
            f"{row['avg_r']:8.4f}"
        )

    print("\n=== PORTFOLIO SUMMARY ===")
    print(f"Combined balance: {total_balance:.2f}")
    print(f"Average balance per symbol: {total_balance / len(rows):.2f}")


if __name__ == "__main__":
    main()
