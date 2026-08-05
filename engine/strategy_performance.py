import pandas as pd


class StrategyPerformanceTracker:

    def __init__(self):
        self.history = []

    def update(self, trade):
        self.history.append(trade)

    def summary(self):
        df = pd.DataFrame(self.history)

        if df.empty:
            return {}

        grouped = df.groupby("strategy")["pnl"]

        return grouped.agg([
            "count",
            "sum",
            "mean"
        ])