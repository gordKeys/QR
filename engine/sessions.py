import pandas as pd


class SessionEngine:

    def add_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        hour = df.index.hour

        df["is_asian"] = (hour >= 0) & (hour < 7)
        df["is_london"] = (hour >= 7) & (hour < 16)

        df["date"] = df.index.date

        # STEP 1: compute full asian range per day
        asian_range = df[df["is_asian"]].groupby("date").agg({
            "high": "max",
            "low": "min"
        })

        asian_range.columns = ["asian_high", "asian_low"]

        # STEP 2: merge back to full dataset (CRITICAL FIX)
        df = df.join(asian_range, on="date")

        return df