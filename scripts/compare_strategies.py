from bootstrap import add_project_root
add_project_root()

import argparse

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester

from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


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

    rows = []

    for name, strategy in strategies.items():
        result = Backtester(data, strategy).run()

        rows.append(
            {
                "strategy": name,
                "final_balance": result["final_balance"],
                "total_trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "avg_r": result["avg_r"],
            }
        )

    rows = sorted(rows, key=lambda row: row["final_balance"], reverse=True)

    print("\n=== STRATEGY RANKING ===")
    for row in rows:
        print(
            f"{row['strategy']:>18} | "
            f"balance={row['final_balance']:.2f} | "
            f"trades={row['total_trades']:>4} | "
            f"win_rate={row['win_rate']:.2%} | "
            f"avg_r={row['avg_r']:.4f}"
        )


if __name__ == "__main__":
    main()
