class FtmoRules:

    def __init__(
        self,
        initial_balance=10000,
        max_daily_loss_pct=0.05,
        max_total_loss_pct=0.10,
        max_risk_per_trade_pct=0.005,
        max_open_positions=1,
        max_consecutive_losses=3,
    ):
        self.initial_balance = initial_balance
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_loss_pct = max_total_loss_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_open_positions = max_open_positions
        self.max_consecutive_losses = max_consecutive_losses

    @property
    def daily_loss_limit(self):
        return self.initial_balance * self.max_daily_loss_pct

    @property
    def total_loss_limit(self):
        return self.initial_balance * self.max_total_loss_pct

    @property
    def per_trade_risk_limit(self):
        return self.initial_balance * self.max_risk_per_trade_pct


class FtmoRiskGuard:

    def __init__(self, rules: FtmoRules):
        self.rules = rules
        self.day_start_equity = None
        self.equity_peak = rules.initial_balance
        self.consecutive_losses = 0
        self.current_day = None

    def reset_day(self, equity, day=None):
        self.day_start_equity = equity
        self.current_day = day

    def update_equity(self, equity):
        self.equity_peak = max(self.equity_peak, equity)

    def can_trade(self, equity, open_positions=0, day=None):
        if self.day_start_equity is None or (day is not None and day != self.current_day):
            self.reset_day(equity, day=day)

        daily_loss = self.day_start_equity - equity
        total_loss = self.rules.initial_balance - equity

        if daily_loss >= self.rules.daily_loss_limit:
            return False, "daily_loss_limit"
        if total_loss >= self.rules.total_loss_limit:
            return False, "total_loss_limit"
        if open_positions >= self.rules.max_open_positions:
            return False, "max_open_positions"
        if self.consecutive_losses >= self.rules.max_consecutive_losses:
            return False, "max_consecutive_losses"

        return True, "ok"

    def register_closed_trade(self, pnl):
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
