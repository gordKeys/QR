from bootstrap import add_project_root
add_project_root()

import argparse

from strategy_batch_tools import (
    default_strategy_grid,
    load_symbol_data,
    resolve_symbol_inputs,
    infer_symbol_from_path,
    run_strategy_on_data,
)
from timing_utils import timed


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
    strategies = default_strategy_grid()

    rows = []
    with timed("multi_symbol_evaluation", style="days"):
        for item in inputs:
            if item.endswith(".csv"):
                symbol_name = infer_symbol_from_path(item)
                data = load_symbol_data(data_path=item)
            else:
                symbol_name = item
                data = load_symbol_data(symbol=item)

            for strategy_name, strategy in strategies.items():
                rows.append(run_strategy_on_data(data, strategy, symbol_name, strategy_name))

    rows = sorted(rows, key=lambda row: (row.symbol, -row.balance))

    print("\n=== MULTI-SYMBOL STRATEGY RESULTS ===")
    print(f"{'symbol':>10} | {'strategy':>18} | {'balance':>12} | {'trades':>6} | {'win_rate':>8} | {'avg_r':>8}")
    print("-" * 82)
    for row in rows:
        print(
            f"{row.symbol:>10} | "
            f"{row.strategy:>18} | "
            f"{row.balance:12.2f} | "
            f"{row.trades:6d} | "
            f"{row.win_rate:8.2%} | "
            f"{row.avg_r:8.4f}"
        )


if __name__ == "__main__":
    main()
