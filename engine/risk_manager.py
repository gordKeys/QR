class RiskManager:

    def __init__(
        self,
        risk_per_trade=0.2,       # 5%
        max_daily_loss=0.3,
        max_drawdown=0.20
    ):

        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

    def calculate_position_size(self, balance, entry, stop, atr=None):
        risk_amount = balance * self.risk_per_trade

        stop_distance = abs(entry - stop)

        if stop_distance == 0:
            return 0

        # volatility scaling (IMPORTANT FOR PROP FIRMS)
        if atr is not None:
            volatility_factor = max(0.5, min(1.5, atr / entry))
            risk_amount *= 1 / volatility_factor

        return risk_amount / stop_distance

    def calculate_sl_tp(
        self,
        direction,
        entry,
        atr,
        sl_atr=2.5,
        tp_atr=5.0
    ):

        if direction == 1:

            stop = entry - atr * sl_atr
            target = entry + atr * tp_atr

        else:

            stop = entry + atr * sl_atr
            target = entry - atr * tp_atr

        return stop, target
