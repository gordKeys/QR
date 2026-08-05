from bootstrap import add_project_root
add_project_root()

import argparse

from strategy_batch_tools import (
    default_strategy_grid,
    load_symbol_data,
    resolve_symbol_inputs,
    infer_symbol_from_path,
)
from engine.backtester import Backtester
from timing_utils import timed


def walkforward_for_data(data, strategy, train_bars=2000, test_bars=500, step_bars=500):
    start = 0
    balances = []
    trades = []
    win_rates = []
    avg_rs = []

    while start + train_bars + test_bars <= len(data):
        test = data.iloc[start + train_bars : start + train_bars + test_bars].copy()
        result = Backtester(test, strategy).run()
        balances.append(result["final_balance"])
        trades.append(result["total_trades"])
        win_rates.append(result["win_rate"])
        avg_rs.append(result["avg_r"])
        start += step_bars

    folds = max(1, len(balances))
    return {
        "avg_balance": sum(balances) / folds,
        "avg_trades": sum(trades) / folds,
        "avg_win_rate": sum(win_rates) / folds,
        "avg_avg_r": sum(avg_rs) / folds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append", help="Symbol name like EURUSD. Repeatable.")
    parser.add_argument("--data", action="append", help="CSV path like data/EURUSD_M5.csv. Repeatable.")
    parser.add_argument("--train-bars", type=int, default=2000)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--step-bars", type=int, default=500)
    args = parser.parse_args()

    symbol_inputs = []
    if args.data:
        symbol_inputs.extend(args.data)
    if args.symbol:
        symbol_inputs.extend(args.symbol)

    inputs = resolve_symbol_inputs(symbol_inputs or None)
    strategies = default_strategy_grid()

    with timed("multi_symbol_walkforward", style="days"):
        print("\n=== MULTI-SYMBOL WALK-FORWARD ===")
        print(f"{'symbol':>10} | {'strategy':>18} | {'avg_balance':>12} | {'avg_trades':>10} | {'avg_win':>8} | {'avg_r':>8}")
        print("-" * 82)

        for item in inputs:
            if item.endswith(".csv"):
                symbol_name = infer_symbol_from_path(item)
                data = load_symbol_data(data_path=item)
            else:
                symbol_name = item
                data = load_symbol_data(symbol=item)

            for strategy_name, strategy in strategies.items():
                result = walkforward_for_data(
                    data=data,
                    strategy=strategy,
                    train_bars=args.train_bars,
                    test_bars=args.test_bars,
                    step_bars=args.step_bars,
                )

                print(
                    f"{symbol_name:>10} | "
                    f"{strategy_name:>18} | "
                    f"{result['avg_balance']:12.2f} | "
                    f"{result['avg_trades']:10.2f} | "
                    f"{result['avg_win_rate']:8.2%} | "
                    f"{result['avg_avg_r']:8.4f}"
                )


if __name__ == "__main__":
    main()
