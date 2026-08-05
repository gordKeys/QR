import pandas as pd
import numpy as np


class PerformanceReport:

    def __init__(self, results):
        self.results = results

    def generate(self):

        trades = pd.DataFrame(self.results["trades"])

        if trades.empty:
            print("No trades found")
            return

        pnl = trades["pnl"]

        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        total_profit = wins.sum()
        total_loss = abs(losses.sum())

        profit_factor = (
            total_profit / total_loss
            if total_loss > 0
            else np.inf
        )

        expectancy = pnl.mean()

        win_rate = len(wins) / len(trades)

        avg_win = wins.mean() if len(wins) else 0
        avg_loss = losses.mean() if len(losses) else 0

        max_win = pnl.max()
        max_loss = pnl.min()

        print("\n==============================")
        print("ACCOUNT PERFORMANCE")
        print("==============================")

        print(f"Final Balance: {self.results['final_balance']:.2f}")
        print(f"Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")

        print(f"\nProfit Factor: {profit_factor:.2f}")
        print(f"Expectancy: {expectancy:.6f}")

        print(f"\nAverage Win: {avg_win:.6f}")
        print(f"Average Loss: {avg_loss:.6f}")

        print(f"\nLargest Win: {max_win:.6f}")
        print(f"Largest Loss: {max_loss:.6f}")


    def drawdown_stats(self):

        equity = pd.Series(self.results["equity_curve"])

        rolling_max = equity.cummax()

        drawdown = equity - rolling_max

        max_drawdown = drawdown.min()

        print(f"\nMax Drawdown: {max_drawdown:.2f}")

    def sharpe_ratio(self):

        trades = pd.DataFrame(self.results["trades"])

        if trades.empty:
            return

        returns = trades["pnl"]

        sharpe = (
            returns.mean() /
            returns.std()
            if returns.std() > 0
            else 0
        )

        print(f"Sharpe Ratio: {sharpe:.2f}")


    def strategy_breakdown(self):

        trades = pd.DataFrame(self.results["trades"])

        if trades.empty:
            return

        print("\n==============================")
        print("STRATEGY BREAKDOWN")
        print("==============================")

        summary = trades.groupby("strategy").agg(
            trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean")
        )

        print(summary)

    def regime_breakdown(self):

        trades = pd.DataFrame(self.results["trades"])

        if trades.empty:
            return

        print("\n==============================")
        print("REGIME BREAKDOWN")
        print("==============================")

        print(
            trades.groupby("regime")
            .agg(
                trades=("pnl", "count"),
                pnl=("pnl", "sum")
            )
        )

    def monthly_returns(self):

        trades = pd.DataFrame(self.results["trades"])

        if trades.empty:
            return

        trades["month"] = pd.to_datetime(
            trades["time"]
        ).dt.to_period("M")

        monthly = trades.groupby("month")["pnl"].sum()

        print("\n==============================")
        print("MONTHLY RETURNS")
        print("==============================")

        print(monthly)

    def full_report(self):

        self.generate()

        self.drawdown_stats()

        self.sharpe_ratio()

        self.strategy_breakdown()

        self.regime_breakdown()

        self.monthly_returns()