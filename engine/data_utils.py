def ticks_to_m5(df_ticks):
    df = df_ticks.copy()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")

    ohlc = df["price"].resample("5min").agg({
        "first": "first",
        "max": "max",
        "min": "min",
        "last": "last"
    })

    ohlc.columns = ["open", "high", "low", "close"]

    return ohlc.dropna()