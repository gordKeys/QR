from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass, asdict

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester

from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


@dataclass
class FoldResult:
    fold: int
    strategy: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_balance: float
    test_balance: float
    train_trades: int
    test_trades: int
    train_win_rate: float
    test_win_rate: float
    train_avg_r: float
    test_avg_r: float


def evaluate_split(train, test, strategy, strategy_name, fold):
    train_result = Backtester(train, strategy).run()
    test_result = Backtester(test, strategy).run()

    return FoldResult(
        fold=fold,
        strategy=strategy_name,
        train_start=str(train.index.min()),
        train_end=str(train.index.max()),
        test_start=str(test.index.min()),
        test_end=str(test.index.max()),
        train_balance=train_result["final_balance"],
        test_balance=test_result["final_balance"],
        train_trades=train_result["total_trades"],
        test_trades=test_result["total_trades"],
        train_win_rate=train_result["win_rate"],
        test_win_rate=test_result["win_rate"],
        train_avg_r=train_result["avg_r"],
        test_avg_r=test_result["avg_r"],
    )


def print_fold(row: FoldResult):
    print(
        f"fold {row.fold:>2} | {row.strategy:>18} | "
        f"train={row.train_balance:10.2f} ({row.train_trades:3d} trades, {row.train_win_rate:.2%}) | "
        f"test={row.test_balance:10.2f} ({row.test_trades:3d} trades, {row.test_win_rate:.2%})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    loader = DataLoader(path=args.data, symbol=args.symbol)
    data = FeatureEngine().add_features(loader.load())

    strategies = {
        "mean_reversion": MeanReversion(),
        "momentum": Momentum(),
        "trend": TrendFollowing(),
        "volatility_breakout": VolatilityBreakout(),
    }

    train_bars = 2000
    test_bars = 500
    step_bars = 500

    rows = []
    fold = 1

    start = 0
    while start + train_bars + test_bars <= len(data):
        train = data.iloc[start : start + train_bars].copy()
        test = data.iloc[start + train_bars : start + train_bars + test_bars].copy()

        for name, strategy in strategies.items():
            rows.append(evaluate_split(train, test, strategy, name, fold))

        fold += 1
        start += step_bars

    print(f"Total bars: {len(data)}")
    print(f"Train window: {train_bars}")
    print(f"Test window: {test_bars}")
    print(f"Step: {step_bars}")
    print(f"Folds: {fold - 1}")

    print("\n=== WALK-FORWARD RESULTS ===")
    for row in rows:
        print_fold(row)

    summary = {}
    for row in rows:
        item = summary.setdefault(
            row.strategy,
            {"test_balance": 0.0, "test_trades": 0, "test_win_rate": 0.0, "test_avg_r": 0.0, "folds": 0}
        )
        item["test_balance"] += row.test_balance
        item["test_trades"] += row.test_trades
        item["test_win_rate"] += row.test_win_rate
        item["test_avg_r"] += row.test_avg_r
        item["folds"] += 1

    ranked = []
    for strategy, item in summary.items():
        folds = item["folds"]
        ranked.append(
            {
                "strategy": strategy,
                "avg_test_balance": item["test_balance"] / folds,
                "avg_test_trades": item["test_trades"] / folds,
                "avg_test_win_rate": item["test_win_rate"] / folds,
                "avg_test_avg_r": item["test_avg_r"] / folds,
            }
        )

    ranked = sorted(ranked, key=lambda row: row["avg_test_balance"], reverse=True)

    print("\n=== WALK-FORWARD SUMMARY ===")
    print(f"{'strategy':>18} | {'avg_test_balance':>15} | {'avg_trades':>10} | {'avg_win':>8} | {'avg_r':>8}")
    print("-" * 76)
    for row in ranked:
        print(
            f"{row['strategy']:>18} | "
            f"{row['avg_test_balance']:15.2f} | "
            f"{row['avg_test_trades']:10.2f} | "
            f"{row['avg_test_win_rate']:8.2%} | "
            f"{row['avg_test_avg_r']:8.4f}"
        )

    best = ranked[0]
    print("\n=== BEST WALK-FORWARD STRATEGY ===")
    print(
        f"{best['strategy']} | avg_test_balance={best['avg_test_balance']:.2f} | "
        f"avg_trades={best['avg_test_trades']:.2f} | avg_win={best['avg_test_win_rate']:.2%} | "
        f"avg_r={best['avg_test_avg_r']:.4f}"
    )


if __name__ == "__main__":
    main()
