from bootstrap import add_project_root
add_project_root()

import os
from dataclasses import dataclass
from typing import Iterable

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester

from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


def default_strategy_grid():
    return {
        "mean_reversion": MeanReversion(),
        "momentum": Momentum(),
        "trend": TrendFollowing(),
        "volatility_breakout": VolatilityBreakout(),
    }


def load_symbol_data(symbol=None, data_path=None):
    loader = DataLoader(path=data_path, symbol=symbol)
    return FeatureEngine().add_features(loader.load())


@dataclass
class StrategySummary:
    symbol: str
    strategy: str
    balance: float
    trades: int
    win_rate: float
    avg_r: float


def run_strategy_on_data(data, strategy, symbol_name, strategy_name):
    result = Backtester(data, strategy).run()
    return StrategySummary(
        symbol=symbol_name,
        strategy=strategy_name,
        balance=result["final_balance"],
        trades=result["total_trades"],
        win_rate=result["win_rate"],
        avg_r=result["avg_r"],
    )


def infer_symbol_from_path(path):
    base = os.path.basename(path)
    if "_" in base:
        return base.split("_")[0]
    return os.path.splitext(base)[0]


def resolve_symbol_inputs(symbols: Iterable[str] | None, data_dir="data"):
    if symbols:
        return list(symbols)
    return [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.endswith("_M5.csv")
    ]
