import pandas as pd


def ticks_to_m5(df_ticks):
    df = df_ticks.copy()

    # EXPECTED COLUMNS:
    # timestamp (ms), price

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")

    ohlc = df["price"].resample("5min").agg([
        "first", "max", "min", "last"
    ])

    ohlc.columns = ["open", "high", "low", "close"]

    return ohlc.dropna()