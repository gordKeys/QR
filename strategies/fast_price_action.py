from __future__ import annotations

from typing import Optional

import pandas as pd

from strategies.pattern_playbook import PatternPlaybookStrategy
from strategies.price_action_mtf import MultiTimeframePriceActionStrategy


class CachedMultiTimeframePriceActionStrategy:
    def __init__(self, strategy: MultiTimeframePriceActionStrategy):
        self.strategy = strategy
        self.strategy_name = getattr(strategy, "strategy_name", f"mtf_price_action_{strategy.setup_mode}")
        self._cached_data_id = None
        self._cached_signals = None

    def _ensure_cache(self, data: pd.DataFrame):
        data_id = id(data)
        if self._cached_data_id != data_id or self._cached_signals is None:
            self._cached_signals = self.strategy.generate_signals(data)
            self._cached_data_id = data_id
        return self._cached_signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._ensure_cache(data)

    def build_trade_plan(
        self,
        data: pd.DataFrame,
        index: int,
        symbol: str,
        cost_profile,
        equity: float,
        signal: Optional[int] = None,
    ):
        signals = self._ensure_cache(data)
        if signal is None:
            signal = int(signals.iloc[index])
        if signal == 0:
            return None

        frame = self.strategy._add_indicators(data)
        current = frame.iloc[index]
        if pd.isna(current["atr"]):
            return None

        if signal == 1:
            stop_anchor = current["swing_low"] if not pd.isna(current["swing_low"]) else current["low"]
            stop = min(stop_anchor, current["low"]) - current["atr"] * self.strategy.atr_mult
            target = current["close"] + abs(current["close"] - stop) * self.strategy.rr
        else:
            stop_anchor = current["swing_high"] if not pd.isna(current["swing_high"]) else current["high"]
            stop = max(stop_anchor, current["high"]) + current["atr"] * self.strategy.atr_mult
            target = current["close"] - abs(stop - current["close"]) * self.strategy.rr

        if pd.isna(stop) or pd.isna(target):
            return None

        return {
            "signal": signal,
            "price": float(current["close"]),
            "stop": float(stop),
            "target": float(target),
            "size": 0.0,
            "size_reason": "price_action",
            "setup": self.strategy.setup_mode,
        }


class CachedPatternPlaybookStrategy:
    def __init__(self, strategy: PatternPlaybookStrategy):
        self.strategy = strategy
        self.strategy_name = getattr(strategy, "strategy_name", strategy.__class__.__name__)
        self._cached_data_id = None
        self._cached_frame = None
        self._cached_bias = None
        self._cached_signals = None

    def _ensure_cache(self, data: pd.DataFrame):
        data_id = id(data)
        if self._cached_data_id != data_id or self._cached_frame is None:
            self._cached_frame = self.strategy._add_indicators(data)
            self._cached_bias = self.strategy._higher_tf_bias(self._cached_frame)
            self._cached_signals = None
            self._cached_data_id = data_id
        return self._cached_frame, self._cached_bias

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame, bias = self._ensure_cache(data)
        if self._cached_signals is None:
            signals = pd.Series(0, index=frame.index)
            start = max(self.strategy.lookback, self.strategy.trend_slow, self.strategy.pivot_window * 2) + 2
            for index in range(start, len(frame)):
                plan = self.strategy._pattern_plan(frame, index, bias=bias)
                if plan is not None:
                    signals.iloc[index] = int(plan["signal"])
            self._cached_signals = signals
        return self._cached_signals

    def build_trade_plan(
        self,
        data: pd.DataFrame,
        index: int,
        symbol: str,
        cost_profile,
        equity: float,
        signal: Optional[int] = None,
        bias=None,
    ):
        frame, cached_bias = self._ensure_cache(data)
        if signal is not None and int(signal) == 0:
            return None
        plan = self.strategy._pattern_plan(frame, index, bias=bias or cached_bias)
        if plan is None:
            return None
        if signal is not None and int(plan["signal"]) != int(signal):
            return None
        plan["setup"] = self.strategy.setup_mode
        return plan


def build_fast_price_action_registry():
    return {
        "mtf_pa_breakout": CachedMultiTimeframePriceActionStrategy(
            MultiTimeframePriceActionStrategy(setup_mode="breakout")
        ),
        "pattern_playbook_double_triple": CachedPatternPlaybookStrategy(
            PatternPlaybookStrategy(setup_mode="double_triple", entry_style="breakout", higher_timeframe="H4")
        ),
    }
