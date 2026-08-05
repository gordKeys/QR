from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester

from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


@dataclass
class StrategyResult:
    name: str
    split: str
    final_balance: float
    total_trades: int
    win_rate: float
    avg_r: float


def run_strategy(data, strategy, name, split_name):
    result = Backtester(data, strategy).run()
    return StrategyResult(
        name=name,
        split=split_name,
        final_balance=result["final_balance"],
        total_trades=result["total_trades"],
        win_rate=result["win_rate"],
        avg_r=result["avg_r"],
    )


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'strategy':>18} | {'balance':>12} | {'trades':>6} | {'win_rate':>8} | {'avg_r':>8}")
    print("-" * 68)
    for row in rows:
        print(
            f"{row.name:>18} | "
            f"{row.final_balance:12.2f} | "
            f"{row.total_trades:6d} | "
            f"{row.win_rate:8.2%} | "
            f"{row.avg_r:8.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    loader = DataLoader(path=args.data, symbol=args.symbol)
    data = FeatureEngine().add_features(loader.load())

    split_index = int(len(data) * 0.7)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    strategies = {
        "mean_reversion": MeanReversion(),
        "momentum": Momentum(),
        "trend": TrendFollowing(),
        "volatility_breakout": VolatilityBreakout(),
    }

    train_rows = []
    test_rows = []

    for name, strategy in strategies.items():
        train_rows.append(run_strategy(train, strategy, name, "train"))
        test_rows.append(run_strategy(test, strategy, name, "test"))

    train_rows = sorted(train_rows, key=lambda row: row.final_balance, reverse=True)
    test_rows = sorted(test_rows, key=lambda row: row.final_balance, reverse=True)

    print(f"Total bars: {len(data)}")
    print(f"Train bars: {len(train)}")
    print(f"Test bars: {len(test)}")

    print_table("TRAIN RESULTS", train_rows)
    print_table("TEST RESULTS", test_rows)

    best_test = test_rows[0]
    print("\n=== BEST OUT-OF-SAMPLE ===")
    print(
        f"{best_test.name} | balance={best_test.final_balance:.2f} | "
        f"trades={best_test.total_trades} | win_rate={best_test.win_rate:.2%} | "
        f"avg_r={best_test.avg_r:.4f}"
    )


if __name__ == "__main__":
    main()
