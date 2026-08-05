import pandas as pd


class PerformanceReporter:

    def __init__(self, trades):
        self.df = pd.DataFrame(trades)

    def strategy_summary(self):

        if self.df.empty:
            return "No trades"

        grouped = self.df.groupby("strategy")

        summary = grouped["pnl"].agg([
            "count",
            "sum",
            "mean"
        ])

        summary["win_rate"] = grouped.apply(
            lambda x: (x["pnl"] > 0).mean()
        )

        return summary.sort_values("sum", ascending=False)

    def regime_summary(self):

        if "regime" not in self.df.columns:
            return "No regime data"

        grouped = self.df.groupby("regime")

        return grouped["pnl"].agg(["count", "sum", "mean"])

    def full_trades(self, n=20):
        return self.df.tail(n)