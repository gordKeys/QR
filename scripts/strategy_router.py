from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.ftmo_xauusd import FTMOXAUUSD
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


class StrategyRouter:

    def __init__(self, score_mode="current", score_mode_by_symbol=None):
        self.score_mode = score_mode
        self.score_mode_by_symbol = {
            key.upper(): value for key, value in (score_mode_by_symbol or {}).items()
        }
        modes = {score_mode, *self.score_mode_by_symbol.values()}
        self.registries = {mode: self._build_registry(mode) for mode in modes}
        self.registry = self.registries[score_mode]

        self.symbol_map = {
            "EURUSD": "mean_reversion_strict",
            "GBPUSD": "mean_reversion_strict",
            "XAUUSD": "ftmo_xauusd",
        }

        self.default_strategy = "mean_reversion"

    @staticmethod
    def _build_registry(score_mode):
        relaxed = score_mode in {"relaxed", "very_relaxed"}
        very_relaxed = score_mode == "very_relaxed"
        return {
            "mean_reversion": MeanReversion(lookback=20, entry_z=1.5),
            "mean_reversion_strict": MeanReversion(lookback=30, entry_z=2.0, min_score=3 if relaxed else 4),
            "momentum": Momentum(),
            "trend": TrendFollowing(),
            "volatility_breakout": VolatilityBreakout(),
            "ftmo_xauusd": FTMOXAUUSD(
                min_score=1 if very_relaxed else (3 if relaxed else 4),
                min_atr_points=0 if very_relaxed else 5,
                forex_hours=range(24) if very_relaxed else range(8, 22),
                require_trend_alignment=not very_relaxed,
            ),
        }

    def get_strategy_name(self, symbol: str) -> str:
        return self.symbol_map.get(symbol.upper(), self.default_strategy)

    def get_strategy(self, symbol: str):
        mode = self.score_mode_by_symbol.get(symbol.upper(), self.score_mode)
        return self.registries[mode][self.get_strategy_name(symbol)]

    def get_registry(self):
        return self.registry

    def update_mapping(self, symbol: str, strategy_name: str):
        if strategy_name not in self.registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.symbol_map[symbol.upper()] = strategy_name
