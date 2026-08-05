from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    All strategies must inherit from this class.
    This enforces clean separation of logic.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Must return a Series:
        1  -> buy
        -1 -> sell
        0  -> no trade
        """
        pass