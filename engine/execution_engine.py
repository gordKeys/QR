from engine.trade import Trade


class ExecutionEngine:

    def __init__(self):

        self.open_trade = None

    def has_position(self):

        return self.open_trade is not None

    def open_position(self, direction, entry, stop, target, size, strategy, regime, time):

        # simulate spread
        spread = 0.0001 if direction == 1 else -0.0001
        entry = entry + spread

        self.open_trade = Trade(
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            position_size=size,
            strategy=strategy,
            regime=regime,
            entry_time=time
        )

    def update(

        self,

        high,

        low,

        current_time

    ):

        if self.open_trade is None:
            return None

        t = self.open_trade

        if t.direction == 1:

            if low <= t.stop_loss:

                t.exit_price = t.stop_loss
                t.status = "STOP"

            elif high >= t.take_profit:

                t.exit_price = t.take_profit
                t.status = "TARGET"

            else:
                return None

        else:

            if high >= t.stop_loss:

                t.exit_price = t.stop_loss
                t.status = "STOP"

            elif low <= t.take_profit:

                t.exit_price = t.take_profit
                t.status = "TARGET"

            else:
                return None

        t.exit_time = current_time

        risk = abs(t.entry_price - t.stop_loss)

        reward = abs(t.exit_price - t.entry_price)

        if t.direction == 1:
            pnl = (t.exit_price - t.entry_price)
        else:
            pnl = (t.entry_price - t.exit_price)

        t.pnl = pnl * t.position_size

        t.r_multiple = reward / risk

        finished = self.open_trade

        self.open_trade = None

        return finished