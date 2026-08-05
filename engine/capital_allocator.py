import numpy as np

class CapitalAllocator:

    def __init__(self, strategies, decay=0.97):
        self.strategies = list(strategies.keys())
        self.decay = decay

        # track performance
        self.profits = {s: 0.0 for s in self.strategies}
        self.trades = {s: 0 for s in self.strategies}

        # initial equal weights
        self.weights = {s: 1 / len(self.strategies) for s in self.strategies}
        self._last_trade_count = 0

    def update_weights(self, trades):

        recent = trades[self._last_trade_count:]
        self._last_trade_count = len(trades)

        for t in recent:
            s = t["strategy"]

            r = t.get("r_multiple", t.get("R", 0))

            self.profits[s] = self.decay * self.profits[s] + r
            self.trades[s] += 1

        total = sum(max(v, 0.0001) for v in self.profits.values())

        for s in self.strategies:
            self.weights[s] = max(self.profits[s], 0.0001) / total

    def get_allocation(self, strategy_name):

        """
        Returns position size multiplier
        """

        return self.weights.get(strategy_name, 0.1)


    def get_position_size(self, strategy):

        return self.weights.get(strategy, 1.0)
