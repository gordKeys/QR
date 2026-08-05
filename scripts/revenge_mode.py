from dataclasses import dataclass
from datetime import datetime


@dataclass
class RevengePlan:
    symbol: str
    sl_time: datetime
    boost_multiplier: float = 2.0
    boosts_total: int = 3
    normal_gap: int = 2
    trade_index: int = 0

    @property
    def total_trades(self):
        return self.boosts_total + self.normal_gap * (self.boosts_total - 1)

    def is_boost_trade(self):
        return self.trade_index % (self.normal_gap + 1) == 0

    def current_multiplier(self):
        return self.boost_multiplier if self.is_boost_trade() else 1.0

    def advance(self):
        self.trade_index += 1

    def is_complete(self):
        return self.trade_index >= self.total_trades


class RevengeTradeManager:

    def __init__(self, *, enabled=False, boost_multiplier=2.0, boosts_total=3, normal_gap=2):
        self.enabled = enabled
        self.boost_multiplier = boost_multiplier
        self.boosts_total = boosts_total
        self.normal_gap = normal_gap
        self.plans = {}
        self.pending = []
        self.active_symbol = None

    def register_stop_loss(self, symbol, sl_time):
        if not self.enabled:
            return None

        existing = self.plans.get(symbol)
        if existing is not None and not existing.is_complete():
            return None

        plan = RevengePlan(
            symbol=symbol,
            sl_time=sl_time,
            boost_multiplier=self.boost_multiplier,
            boosts_total=self.boosts_total,
            normal_gap=self.normal_gap,
        )
        self.plans[symbol] = plan
        self.pending.append(symbol)
        self.pending.sort(key=lambda candidate: self.plans[candidate].sl_time)
        self._activate_next_if_needed()
        return plan

    def is_active(self, symbol):
        return self.enabled and self.active_symbol == symbol and symbol in self.plans

    def has_plan(self, symbol):
        return symbol in self.plans

    def cancel_plan(self, symbol):
        plan = self.plans.pop(symbol, None)
        self.pending = [pending_symbol for pending_symbol in self.pending if pending_symbol != symbol]
        if self.active_symbol == symbol:
            self.active_symbol = None
            self._activate_next_if_needed()
        return plan

    def _activate_next_if_needed(self):
        if self.active_symbol is not None:
            return

        while self.pending:
            candidate = self.pending.pop(0)
            plan = self.plans.get(candidate)
            if plan is None or plan.is_complete():
                continue
            self.active_symbol = candidate
            return

    def get_trade_context(self, symbol):
        if not self.enabled:
            return {
                "enabled": False,
                "blocked": False,
                "active": False,
                "boosted": False,
                "multiplier": 1.0,
                "stage": None,
                "total": None,
            }

        plan = self.plans.get(symbol)
        if plan is None:
            return {
                "enabled": True,
                "blocked": False,
                "active": False,
                "boosted": False,
                "multiplier": 1.0,
                "stage": None,
                "total": None,
            }

        if symbol != self.active_symbol:
            return {
                "enabled": True,
                "blocked": True,
                "active": False,
                "boosted": False,
                "multiplier": 0.0,
                "stage": plan.trade_index + 1,
                "total": plan.total_trades,
            }

        return {
            "enabled": True,
            "blocked": False,
            "active": True,
            "boosted": plan.is_boost_trade(),
            "multiplier": plan.current_multiplier(),
            "stage": plan.trade_index + 1,
            "total": plan.total_trades,
        }

    def on_trade_filled(self, symbol):
        if not self.enabled:
            return None

        plan = self.plans.get(symbol)
        if plan is None or symbol != self.active_symbol:
            return None

        boosted = plan.is_boost_trade()
        stage = plan.trade_index + 1
        plan.advance()
        completed = plan.is_complete()

        if completed:
            del self.plans[symbol]
            self.active_symbol = None
            self._activate_next_if_needed()

        return {
            "symbol": symbol,
            "boosted": boosted,
            "stage": stage,
            "completed": completed,
        }
