from dataclasses import dataclass
from typing import Iterable


RECOMMENDED_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "EURGBP",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "CADJPY",
    "EURAUD",
    "USDCHF",
    "XAUUSD",
]

PATTERN_RECOMMENDED_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "AUDUSD",
    "USDCAD",
    "XAUUSD",
]


@dataclass(frozen=True)
class CostProfile:
    point: float
    contract_size: float
    commission_round_turn: float = 0.0
    swap_long_per_lot_day: float = 0.0
    swap_short_per_lot_day: float = 0.0
    spread_multiplier: float = 1.0


def default_cost_profile(symbol: str) -> CostProfile:
    symbol = symbol.upper()
    if symbol.startswith("XAU"):
        return CostProfile(point=0.01, contract_size=100.0, commission_round_turn=7.0)

    if symbol.endswith("JPY"):
        return CostProfile(point=0.01, contract_size=100000.0, commission_round_turn=7.0)

    return CostProfile(point=0.0001, contract_size=100000.0, commission_round_turn=7.0)


def load_symbol_candidates(symbols: Iterable[str] | None, available_symbols: Iterable[str] | None = None):
    if symbols:
        return [item.upper() for item in symbols]

    available = {item.upper() for item in available_symbols or []}
    if available:
        return [symbol for symbol in RECOMMENDED_SYMBOLS if symbol in available]

    return list(RECOMMENDED_SYMBOLS)


def load_pattern_symbol_candidates(symbols: Iterable[str] | None, available_symbols: Iterable[str] | None = None):
    if symbols:
        return [item.upper() for item in symbols]

    available = {item.upper() for item in available_symbols or []}
    if available:
        return [symbol for symbol in PATTERN_RECOMMENDED_SYMBOLS if symbol in available]

    return list(PATTERN_RECOMMENDED_SYMBOLS)
