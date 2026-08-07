from bootstrap import add_project_root
add_project_root()

import argparse
from pathlib import Path

import pandas as pd

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.mtf_backtester import MTFBacktester
from engine.pattern_backtester import PatternBacktester
from engine.symbol_universe import RECOMMENDED_SYMBOLS, default_cost_profile, load_symbol_candidates
from strategies.price_action_mtf import MultiTimeframePriceActionStrategy
from strategies.pattern_playbook import PatternPlaybookStrategy


class CachedPriceActionMTFStrategy:
    def __init__(self, strategy: MultiTimeframePriceActionStrategy):
        self.strategy = strategy
        self.strategy_name = getattr(strategy, "strategy_name", strategy.__class__.__name__)
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

    def build_trade_plan(self, data: pd.DataFrame, index: int, symbol: str, cost_profile, equity: float, signal=None):
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

    def build_trade_plan(self, data: pd.DataFrame, index: int, symbol: str, cost_profile, equity: float, signal=None, bias=None):
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


def available_csv_symbols(data_dir):
    directory = Path(data_dir)
    if not directory.exists():
        return []
    return sorted({path.name.split("_")[0].upper() for path in directory.glob("*_M5.csv")})


def load_mtf_data(symbol, data_dir):
    loader = DataLoader(symbol=symbol, data_dir=data_dir)
    frame = loader.load()
    if frame is None or frame.empty:
        return frame
    return FeatureEngine().add_features(frame)


def load_pattern_data(symbol, data_dir, timeframe="1h"):
    loader = DataLoader(symbol=symbol, data_dir=data_dir)
    frame = loader.load()
    if frame is None or frame.empty:
        return frame

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)

    frame = frame.sort_index()
    frame = frame.resample(timeframe).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "mean",
            "real_volume": "sum",
        }
    ).dropna()
    return FeatureEngine().add_features(frame)


def build_cost_profile(symbol, commission_round_turn, swap_long, swap_short):
    base = default_cost_profile(symbol)
    return base.__class__(
        point=base.point,
        contract_size=base.contract_size,
        commission_round_turn=commission_round_turn,
        swap_long_per_lot_day=swap_long,
        swap_short_per_lot_day=swap_short,
    )


def summarize(result):
    return result["net_profit"] * (1 + result["win_rate"]) / (1 + result["max_drawdown"] * 10)


def main():
    parser = argparse.ArgumentParser(description="Fast backtest for price_action_mtf and pattern_playbook.")
    parser.add_argument("--symbols", nargs="+", help="Optional symbol list. Defaults to the CSV symbols in data/.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=0.004)
    parser.add_argument("--commission-round-turn", type=float, default=7.0)
    parser.add_argument("--swap-long", type=float, default=0.0)
    parser.add_argument("--swap-short", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--mtf-setups",
        nargs="+",
        default=["all", "trend_pullback", "breakout", "fib_retracement"],
    )
    parser.add_argument(
        "--pattern-setups",
        nargs="+",
        default=["triangle_breakout", "head_shoulders", "double_triple", "mtf_reversal"],
    )
    parser.add_argument("--pattern-timeframe", choices=["15min", "30min", "1h", "4h"], default="1h")
    args = parser.parse_args()

    available = available_csv_symbols(args.data_dir)
    candidate_symbols = load_symbol_candidates(args.symbols, available_symbols=available)

    mtf_strategies = {
        f"mtf_pa_{setup}": CachedPriceActionMTFStrategy(MultiTimeframePriceActionStrategy(setup_mode=setup))
        for setup in args.mtf_setups
    }
    pattern_strategies = {
        "triangle_breakout": CachedPatternPlaybookStrategy(
            PatternPlaybookStrategy(setup_mode="triangles", entry_style="breakout", higher_timeframe="H4")
        ),
        "head_shoulders": CachedPatternPlaybookStrategy(
            PatternPlaybookStrategy(setup_mode="head_shoulders", entry_style="breakout", higher_timeframe="H4")
        ),
        "double_triple": CachedPatternPlaybookStrategy(
            PatternPlaybookStrategy(setup_mode="double_triple", entry_style="breakout", higher_timeframe="H4")
        ),
        "mtf_reversal": CachedPatternPlaybookStrategy(
            PatternPlaybookStrategy(setup_mode="mtf_reversal", entry_style="breakout", higher_timeframe="H4")
        ),
    }
    pattern_strategies = {name: strategy for name, strategy in pattern_strategies.items() if name in set(args.pattern_setups)}

    rows = []

    print("\n=== FAST PRICE ACTION BACKTEST ===")
    print(f"Recommended universe: {', '.join(RECOMMENDED_SYMBOLS)}")
    print(f"Available CSV symbols: {', '.join(available) if available else 'none'}")
    print(f"Testing symbols: {', '.join(candidate_symbols)}")
    print(f"Pattern timeframe: {args.pattern_timeframe}")
    print(f"MTF setups: {', '.join(mtf_strategies.keys())}")
    print(f"Pattern setups: {', '.join(pattern_strategies.keys())}")
    print("-" * 120)

    for symbol in candidate_symbols:
        if symbol not in available:
            print(f"{symbol:>8} | skipped (no CSV data)")
            continue

        mtf_data = load_mtf_data(symbol, args.data_dir)
        pattern_data = load_pattern_data(symbol, args.data_dir, timeframe=args.pattern_timeframe)

        mtf_cost_profile = build_cost_profile(symbol, args.commission_round_turn, args.swap_long, args.swap_short)
        pattern_cost_profile = build_cost_profile(symbol, args.commission_round_turn, args.swap_long, args.swap_short)

        for strategy_name, strategy in mtf_strategies.items():
            if mtf_data is None or mtf_data.empty:
                continue
            result = MTFBacktester(
                data=mtf_data,
                strategy=strategy,
                symbol=symbol,
                initial_balance=args.initial_balance,
                risk_per_trade=args.risk_pct,
                cost_profile=mtf_cost_profile,
            ).run()
            rows.append({"symbol": symbol, "strategy": strategy_name, **result, "score": summarize(result)})
            print(
                f"{symbol:>8} | {strategy_name:>16} | "
                f"bal={result['final_balance']:10.2f} | "
                f"net={result['net_profit']:9.2f} | "
                f"trades={result['total_trades']:4d} | "
                f"wr={result['win_rate']:.2%} | pf={result['profit_factor']:.2f} | dd={result['max_drawdown']:.2%}"
            )

        for strategy_name, strategy in pattern_strategies.items():
            if pattern_data is None or pattern_data.empty:
                continue
            result = PatternBacktester(
                data=pattern_data,
                strategy=strategy,
                symbol=symbol,
                initial_balance=args.initial_balance,
                risk_per_trade=args.risk_pct,
                cost_profile=pattern_cost_profile,
            ).run()
            rows.append({"symbol": symbol, "strategy": strategy_name, **result, "score": summarize(result)})
            print(
                f"{symbol:>8} | {strategy_name:>16} | "
                f"bal={result['final_balance']:10.2f} | "
                f"net={result['net_profit']:9.2f} | "
                f"trades={result['total_trades']:4d} | "
                f"wr={result['win_rate']:.2%} | pf={result['profit_factor']:.2f} | dd={result['max_drawdown']:.2%}"
            )

    if not rows:
        print("\nNo results generated.")
        return

    print("\n=== BEST BY SYMBOL ===")
    best_by_symbol = {}
    for row in rows:
        current = best_by_symbol.get(row["symbol"])
        if current is None or row["score"] > current["score"]:
            best_by_symbol[row["symbol"]] = row
    for row in sorted(best_by_symbol.values(), key=lambda item: item["score"], reverse=True):
        print(
            f"{row['symbol']:>8} | best={row['strategy']:<16} | "
            f"score={row['score']:10.2f} | net={row['net_profit']:9.2f} | "
            f"dd={row['max_drawdown']:.2%} | wr={row['win_rate']:.2%}"
        )

    print("\n=== TOP SETUPS ===")
    for row in sorted(rows, key=lambda item: item["score"], reverse=True)[: args.top_n]:
        print(
            f"{row['symbol']:>8} | {row['strategy']:<16} | "
            f"score={row['score']:10.2f} | net={row['net_profit']:9.2f} | "
            f"trades={row['total_trades']:4d} | wr={row['win_rate']:.2%} | dd={row['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
