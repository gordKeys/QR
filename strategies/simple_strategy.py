import pandas as pd
from strategies.base_strategy import BaseStrategy


class SimpleStrategy(BaseStrategy):
    """
    Clean test strategy (still naive, but structured properly)
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)

        signals[data["close"] > data["open"]] = 1
        signals[data["close"] < data["open"]] = -1

        return signals