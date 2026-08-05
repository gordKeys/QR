import pandas as pd

from engine.capital_allocator import CapitalAllocator
from engine.strategy_performance import StrategyPerformanceTracker
from engine.signal_filter import SignalFilter

from engine.execution_engine import ExecutionEngine
from engine.risk_manager import RiskManager


class MultiStrategyBacktester:

    def __init__(self, data, strategies, selector,
                 spread=0.0002,
                 commission_per_trade=0.0,
                 initial_balance=10000):

        self.filter = SignalFilter()
        self.performance = StrategyPerformanceTracker()
        self.allocator = CapitalAllocator(strategies)

        self.execution = ExecutionEngine()
        self.risk = RiskManager()

        self.data = data.copy()
        self.strategies = strategies
        self.selector = selector

        self.spread = spread
        self.commission_per_trade = commission_per_trade
        self.balance = initial_balance

        self.current_strategy = None
        self.position_size = 0.0

        self.trades = []
        self.equity_curve = []

    def run(self):

        strategy_signals = {
            name: strat.generate_signals(self.data)
            for name, strat in self.strategies.items()
        }

        for i in range(len(self.data)):

            price = self.data["close"].iloc[i]

            # -----------------------------
            # SELECT STRATEGY (REGIME)
            # -----------------------------
            selected = self.selector.select(self.data, i)

            base_signal = strategy_signals[selected].iloc[i]

            # -----------------------------
            # SIGNAL QUALITY FILTER
            # -----------------------------
            quality = self.filter.compute_score(self.data, i, base_signal)

            if quality < 60:
                signal = 0
            else:
                signal = base_signal

            # ------------------------------------
            # UPDATE EXISTING TRADE
            # ------------------------------------

            closed_trade = self.execution.update(
                high=self.data["high"].iloc[i],
                low=self.data["low"].iloc[i],
                current_time=self.data.index[i]
            )

            if closed_trade:
                self.balance += closed_trade.pnl
                self.balance -= self.commission_per_trade

                trade = {
                    "strategy": closed_trade.strategy,
                    "regime": closed_trade.regime,
                    "type": "long" if closed_trade.direction == 1 else "short",
                    "entry": closed_trade.entry_price,
                    "exit": closed_trade.exit_price,
                    "pnl": closed_trade.pnl,
                    "R": closed_trade.r_multiple,
                    "size": closed_trade.position_size,
                    "commission": self.commission_per_trade,
                    "quality": quality,
                    "time": closed_trade.exit_time,
                }

                self.trades.append(trade)

                self.performance.update(trade)

                self.allocator.update_weights(self.trades)

            # ------------------------------------
            # OPEN NEW POSITION
            # ------------------------------------

            if signal != 0 and not self.execution.has_position():

                atr = self.safe("atr", i)

                if pd.notna(atr):
                    stop, target = self.risk.calculate_sl_tp(
                        signal,
                        price,
                        atr
                    )

                    allocation = self.allocator.get_allocation(selected)

                    size = self.risk.calculate_position_size(
                        self.balance,
                        price,
                        stop
                    )

                    size *= allocation

                    self.execution.open_position(
                        direction=signal,
                        entry=price,
                        stop=stop,
                        target=target,
                        size=size,
                        strategy=selected,
                        regime=self.safe("regime", i),
                        time=self.data.index[i]
                    )

            self.equity_curve.append(self.balance)

        return self.results()

    def _pnl(self, exit_price, entry_price, pos):

        if pos == 1:
            return (exit_price - entry_price) - self.spread
        else:
            return (entry_price - exit_price) - self.spread

    def results(self):

        wins = len([t for t in self.trades if t["pnl"] > 0])

        return {
            "final_balance": self.balance,
            "total_trades": len(self.trades),
            "win_rate": wins / len(self.trades) if self.trades else 0,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }

    def safe(self, col, i, default=0):
        if col in self.data.columns:
            return self.data[col].iloc[i]
        return default
