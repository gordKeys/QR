from strategies.mean_reversion import MeanReversion
from strategies.lion_of_judah import LionOfJudahFiveSignalStrategy
from strategies.momentum import Momentum
from strategies.audusd_confluence import AUDUSDConfluenceStrategy
from strategies.ftmo_xauusd import FTMOXAUUSD
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


class StrategyRouter:

    def __init__(self):
        self.registry = {
            "mean_reversion": MeanReversion(lookback=20, entry_z=1.5),
            "mean_reversion_strict": MeanReversion(lookback=30, entry_z=2.0),
            "momentum": Momentum(),
            "trend": TrendFollowing(),
            "volatility_breakout": VolatilityBreakout(),
            "ftmo_xauusd": FTMOXAUUSD(min_score=4),
            "audusd_confluence": AUDUSDConfluenceStrategy(min_score=3),
            "lion_usdjpy_mirror": LionOfJudahFiveSignalStrategy(
                use_trend=True,
                use_candle=True,
                require_trend_alignment=True,
                min_score=5,
                rr=1.5,
                atr=0.8,
                fast=5,
                slow=13,
                rsi_p=9,
                ob=68,
                os=32,
                mirror=True,
                strategy_name="five_signal_confluence_scalper_strict_jpy_trend",
            ),
            "lion_usdchf_actual": LionOfJudahFiveSignalStrategy(
                use_trend=True,
                use_candle=True,
                require_trend_alignment=True,
                min_score=3,
                rr=1.5,
                atr=0.8,
                fast=5,
                slow=13,
                rsi_p=9,
                ob=68,
                os=32,
                mirror=False,
                strategy_name="five_signal_confluence_scalper_strict",
            ),
        }

        self.symbol_map = {
            "EURUSD": "mean_reversion_strict",
            "GBPUSD": "mean_reversion_strict",
            "AUDUSD": "audusd_confluence",
            "USDJPY": "lion_usdjpy_mirror",
            "USDCHF": "lion_usdchf_actual",
            "XAUUSD": "ftmo_xauusd",
        }

        self.default_strategy = "mean_reversion"

    def get_strategy_name(self, symbol: str) -> str:
        return self.symbol_map.get(symbol.upper(), self.default_strategy)

    def get_strategy(self, symbol: str):
        return self.registry[self.get_strategy_name(symbol)]

    def get_registry(self):
        return self.registry

    def update_mapping(self, symbol: str, strategy_name: str):
        if strategy_name not in self.registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.symbol_map[symbol.upper()] = strategy_name
