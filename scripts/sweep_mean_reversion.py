from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass
from itertools import product

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester
from strategies.mean_reversion import MeanReversion
from timing_utils import timed


@dataclass
class SweepRow:
    lookback: int
    entry_z: float
    avg_test_balance: float
    avg_test_trades: float
    avg_test_win_rate: float
    avg_test_avg_r: float


def run_walkforward(data, lookback, entry_z, train_bars, test_bars, step_bars):
    start = 0
    test_balances = []
    test_trades = []
    test_win_rates = []
    test_avg_rs = []

    while start + train_bars + test_bars <= len(data):
        test = data.iloc[start + train_bars : start + train_bars + test_bars].copy()
        strategy = MeanReversion(lookback=lookback, entry_z=entry_z)
        result = Backtester(test, strategy).run()

        test_balances.append(result["final_balance"])
        test_trades.append(result["total_trades"])
        test_win_rates.append(result["win_rate"])
        test_avg_rs.append(result["avg_r"])

        start += step_bars

    folds = max(1, len(test_balances))
    return SweepRow(
        lookback=lookback,
        entry_z=entry_z,
        avg_test_balance=sum(test_balances) / folds,
        avg_test_trades=sum(test_trades) / folds,
        avg_test_win_rate=sum(test_win_rates) / folds,
        avg_test_avg_r=sum(test_avg_rs) / folds,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--data", default=None)
    parser.add_argument("--train-bars", type=int, default=2000)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--step-bars", type=int, default=500)
    args = parser.parse_args()

    loader = DataLoader(path=args.data, symbol=args.symbol)
    data = FeatureEngine().add_features(loader.load())

    lookbacks = [10, 20, 30, 40]
    entry_z_values = [1.5, 2.0, 2.5]

    rows = []
    with timed("mean_reversion_sweep", style="days"):
        for lookback, entry_z in product(lookbacks, entry_z_values):
            rows.append(
                run_walkforward(
                    data=data,
                    lookback=lookback,
                    entry_z=entry_z,
                    train_bars=args.train_bars,
                    test_bars=args.test_bars,
                    step_bars=args.step_bars,
                )
            )

    rows = sorted(rows, key=lambda row: row.avg_test_balance, reverse=True)

    print(f"Total bars: {len(data)}")
    print("\n=== MEAN REVERSION SWEEP ===")
    print(f"{'lookback':>8} | {'entry_z':>7} | {'avg_balance':>12} | {'avg_trades':>10} | {'avg_win':>8} | {'avg_r':>8}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row.lookback:8d} | "
            f"{row.entry_z:7.2f} | "
            f"{row.avg_test_balance:12.2f} | "
            f"{row.avg_test_trades:10.2f} | "
            f"{row.avg_test_win_rate:8.2%} | "
            f"{row.avg_test_avg_r:8.4f}"
        )

    best = rows[0]
    print("\n=== BEST CONFIG ===")
    print(
        f"lookback={best.lookback}, entry_z={best.entry_z}, "
        f"avg_test_balance={best.avg_test_balance:.2f}, "
        f"avg_trades={best.avg_test_trades:.2f}, "
        f"avg_win={best.avg_test_win_rate:.2%}, "
        f"avg_r={best.avg_test_avg_r:.4f}"
    )


if __name__ == "__main__":
    main()
