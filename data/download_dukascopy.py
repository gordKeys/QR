import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import os


class DukascopyDownloader:
    """
    Downloads FX historical data from Dukascopy
    """

    BASE_URL = "https://datafeed.dukascopy.com/datafeed"

    def __init__(self, symbol="EURUSD", timeframe="M5"):
        self.symbol = symbol
        self.timeframe = timeframe

    def _get_url(self, year, month, day, hour):
        return f"{self.BASE_URL}/{self.symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    def download(self, start, end):
        """
        Downloads tick data (we will convert to M5 later)
        """
        data = []

        current = start

        print("Downloading data...")

        while current < end:
            for hour in range(24):
                url = self._get_url(current.year, current.month, current.day, hour)

                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        data.append((current, r.content))
                except:
                    pass

            current += timedelta(days=1)

        print("Download complete")
        return data

def save_dummy_mt5_style_data():
    """
    SESSION-BASED FX SIMULATOR
    - Asian session: low volatility range
    - London session: breakout expansion
    - NY session: noise + continuation
    """

    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=3000, freq="5min")

    price = 1.1000
    rows = []

    for i, dt in enumerate(dates):

        hour = dt.hour

        # -------------------------
        # SESSION REGIMES
        # -------------------------

        # ASIAN SESSION (00:00 - 07:00)
        if 0 <= hour < 7:
            drift = np.random.normal(0, 0.00002)
            volatility = 0.00005

        # LONDON SESSION (07:00 - 14:00)
        elif 7 <= hour < 14:
            drift = np.random.normal(0, 0.00015)
            volatility = 0.00020

        # NY SESSION (14:00 - 22:00)
        else:
            drift = np.random.normal(0, 0.00010)
            volatility = 0.00015

        # -------------------------
        # PRICE GENERATION
        # -------------------------
        change = drift + np.random.normal(0, volatility)
        price += change

        high = price + abs(np.random.normal(0, volatility))
        low = price - abs(np.random.normal(0, volatility))
        open_price = price - change
        close = price

        rows.append([dt, open_price, high, low, close])

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    df.to_csv("data/EURUSD_M5.csv")

    print("Saved realistic session-based EURUSD_M5.csv")


if __name__ == "__main__":
    # For now we use synthetic data (so we can continue building engine)
    save_dummy_mt5_style_data()