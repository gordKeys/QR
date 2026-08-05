import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, path=None, symbol=None, timeframe="M5", data_dir="data"):
        if path is None:
            if symbol is None:
                symbol = "EURUSD"
            path = Path(data_dir) / f"{symbol}_{timeframe}.csv"

        self.path = Path(path)

    def load(self):

        df = pd.read_csv(self.path)

        # Parse MT5 datetime
        df["time"] = pd.to_datetime(df["time"])

        df = df.set_index("time")

        # Keep only the columns we need
        df = df[
            [
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            ]
        ]

        df = df.sort_index()

        return df
