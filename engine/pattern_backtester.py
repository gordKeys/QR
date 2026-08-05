from dataclasses import dataclass
from datetime import datetime
from math import inf

import pandas as pd

from engine.risk_manager import RiskManager
from engine.symbol_universe import CostProfile, default_cost_profile


@dataclass
class PatternTrade:
    symbol: str
    strategy: str
    direction: int
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pattern: str = ""
    swap_cost: float = 0.0
    last_swap_date: object = None
    exit_time: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl: float = 0.0
    r_multiple: float = 0.0


class PatternBacktester:

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        symbol: str,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.004,
        cost_profile: CostProfile | None = None,
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.symbol = symbol.upper()
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.risk_per_trade = float(risk_per_trade)
        self.cost_profile = cost_profile or default_cost_profile(self.symbol)
        self.risk = RiskManager(risk_per_trade=self.risk_per_trade)

        self.position = None
        self.trades = []
        self.equity_curve = []

    def _spread_cost(self, spread_points):
        return float(spread_points or 0.0) * self.cost_profile.point * self.cost_profile.spread_multiplier

    def _entry_price(self, mid_price, direction, spread_points):
        half_spread = self._spread_cost(spread_points) / 2.0
        return mid_price + half_spread if direction == 1 else mid_price - half_spread

    def _exit_price(self, trigger_price, direction, spread_points):
        half_spread = self._spread_cost(spread_points) / 2.0
        return trigger_price - half_spread if direction == 1 else trigger_price + half_spread

    def _position_size(self, equity, entry_price, stop_price):
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0.0
        risk_amount = equity * self.risk_per_trade
        raw_lots = risk_amount / (stop_distance * self.cost_profile.contract_size)
        return max(0.0, raw_lots)

    def _finalize_trade(self, trade, exit_time, exit_price, reason):
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason

        gross_pnl = (
            (exit_price - trade.entry_price)
            if trade.direction == 1
            else (trade.entry_price - exit_price)
        ) * trade.position_size * self.cost_profile.contract_size

        commission = self.cost_profile.commission_round_turn * trade.position_size
        trade.pnl = gross_pnl - commission - trade.swap_cost
        trade.r_multiple = trade.pnl / max(1e-9, abs(trade.entry_price - trade.stop_loss) * trade.position_size * self.cost_profile.contract_size)
        self.balance += trade.pnl
        self.trades.append(trade)
        self.position = None

    def run(self):
        try:
            signals = self.strategy.generate_signals(self.data)
        except Exception:
            signals = pd.Series(0, index=self.data.index)

        for index, (timestamp, row) in enumerate(self.data.iterrows()):
            mid_price = float(row["close"])
            spread_points = float(row["spread"]) if "spread" in row and pd.notna(row["spread"]) else 0.0

            if self.position is not None and self.position.last_swap_date is not None:
                if timestamp.date() > self.position.last_swap_date:
                    days_held = (timestamp.date() - self.position.last_swap_date).days
                    swap_rate = self.cost_profile.swap_long_per_lot_day if self.position.direction == 1 else self.cost_profile.swap_short_per_lot_day
                    self.position.swap_cost += swap_rate * self.position.position_size * days_held
                    self.position.last_swap_date = timestamp.date()

            if self.position is not None:
                if self.position.direction == 1:
                    if row["low"] <= self.position.stop_loss:
                        exit_price = self._exit_price(self.position.stop_loss, 1, spread_points)
                        self._finalize_trade(self.position, timestamp, exit_price, "stop")
                    elif row["high"] >= self.position.take_profit:
                        exit_price = self._exit_price(self.position.take_profit, 1, spread_points)
                        self._finalize_trade(self.position, timestamp, exit_price, "target")
                else:
                    if row["high"] >= self.position.stop_loss:
                        exit_price = self._exit_price(self.position.stop_loss, -1, spread_points)
                        self._finalize_trade(self.position, timestamp, exit_price, "stop")
                    elif row["low"] <= self.position.take_profit:
                        exit_price = self._exit_price(self.position.take_profit, -1, spread_points)
                        self._finalize_trade(self.position, timestamp, exit_price, "target")

            signal = int(signals.iloc[index]) if index < len(signals) else 0
            plan = None
            if self.position is None:
                if signal != 0 and hasattr(self.strategy, "build_trade_plan"):
                    try:
                        plan = self.strategy.build_trade_plan(
                            self.data,
                            index,
                            self.symbol,
                            self.cost_profile,
                            self.balance,
                            signal=signal,
                        )
                    except TypeError:
                        plan = None

                if plan is None and signal != 0 and hasattr(self.strategy, "analyze_trade"):
                    try:
                        plan = self.strategy.analyze_trade(
                            self.data.iloc[: index + 1],
                            symbol=self.symbol,
                            equity=self.balance,
                            signal=signal,
                            price=mid_price,
                        )
                    except TypeError:
                        plan = None

                if plan is None and signal != 0:
                    stop, target = self.risk.calculate_sl_tp(signal, mid_price, float(row["atr"]))
                    plan = {
                        "signal": signal,
                        "price": mid_price,
                        "stop": stop,
                        "target": target,
                        "size": 0.0,
                        "size_reason": "fallback",
                    }

                if plan is not None:
                    direction = int(plan.get("signal", signal))
                    stop = float(plan["stop"])
                    target = float(plan["target"])
                    mid_price = float(plan.get("price", mid_price))
                    entry_price = self._entry_price(mid_price, direction, spread_points)
                    lots = self._position_size(self.balance, entry_price, stop)
                    if lots > 0:
                        self.position = PatternTrade(
                            symbol=self.symbol,
                            strategy=getattr(self.strategy, "strategy_name", self.strategy.__class__.__name__),
                            direction=direction,
                            entry_time=timestamp.to_pydatetime(),
                            entry_price=entry_price,
                            stop_loss=stop,
                            take_profit=target,
                            position_size=lots,
                            pattern=str(plan.get("pattern", "")),
                            last_swap_date=timestamp.date(),
                        )

            self.equity_curve.append(self.balance)

        return self.results()

    def results(self):
        total_trades = len(self.trades)
        wins = len([trade for trade in self.trades if trade.pnl > 0])
        gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else inf

        peak = -inf
        max_drawdown = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - value) / peak)

        avg_r = sum(trade.r_multiple for trade in self.trades) / total_trades if total_trades else 0.0

        return {
            "symbol": self.symbol,
            "strategy": getattr(self.strategy, "strategy_name", self.strategy.__class__.__name__),
            "final_balance": self.balance,
            "net_profit": self.balance - self.initial_balance,
            "total_trades": total_trades,
            "win_rate": wins / total_trades if total_trades else 0.0,
            "avg_r": avg_r,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }
