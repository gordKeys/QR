from collections import defaultdict


class HardDrawdownGuard:

    def __init__(self, enabled=False, drawdown_usd=3000.0):
        self.enabled = enabled
        self.drawdown_usd = float(drawdown_usd)
        self.start_equity = None
        self.day_start_equity = None
        self.triggered = False
        self.reason = None

    def reset_session(self, equity):
        self.start_equity = float(equity)
        self.day_start_equity = float(equity)
        self.triggered = False
        self.reason = None

    def reset_day(self, equity):
        self.day_start_equity = float(equity)

    def check(self, equity):
        if not self.enabled:
            return False

        equity = float(equity)
        if self.start_equity is None:
            self.reset_session(equity)

        if self.day_start_equity is None:
            self.day_start_equity = equity

        total_drawdown = self.start_equity - equity
        daily_drawdown = self.day_start_equity - equity

        if total_drawdown >= self.drawdown_usd:
            self.triggered = True
            self.reason = f"total_drawdown_{total_drawdown:.2f}"
            return True

        if daily_drawdown >= self.drawdown_usd:
            self.triggered = True
            self.reason = f"daily_drawdown_{daily_drawdown:.2f}"
            return True

        return False


class SymbolCircuitBreaker:

    def __init__(self, max_stop_losses=2):
        self.max_stop_losses = int(max_stop_losses)
        self.state = defaultdict(
            lambda: {
                "consecutive_stop_losses": 0,
                "paused": False,
                "reason": None,
            }
        )

    def is_paused(self, symbol):
        return self.state[symbol]["paused"]

    def pause(self, symbol, reason):
        self.state[symbol]["paused"] = True
        self.state[symbol]["reason"] = reason

    def cancel(self, symbol):
        if symbol in self.state:
            self.state[symbol]["paused"] = True
            self.state[symbol]["reason"] = "cancelled"

    def register_close(self, symbol, stop_loss_hit, revenge_failed=False):
        state = self.state[symbol]
        if stop_loss_hit:
            state["consecutive_stop_losses"] += 1
            if revenge_failed:
                state["paused"] = True
                state["reason"] = "revenge_cycle_failed"
                return state
            if state["consecutive_stop_losses"] >= self.max_stop_losses:
                state["paused"] = True
                state["reason"] = "stop_loss_limit"
                return state
        else:
            state["consecutive_stop_losses"] = 0

        return state
