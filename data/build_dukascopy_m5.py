import pandas as pd
import requests
import io
import gzip
from datetime import datetime, timedelta


BASE_URL = "https://datafeed.dukascopy.com/datafeed"


def get_url(symbol, year, month, day, hour):
    return f"{BASE_URL}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"


def download_ticks(symbol, start, end):
    all_ticks = []

    current = start

    while current <= end:

        for hour in range(24):

            url = get_url(symbol, current.year, current.month, current.day, hour)

            r = requests.get(url, timeout=10)

            if r.status_code != 200:
                continue

            try:
                data = gzip.decompress(r.content)
                all_ticks.append(data)

            except:
                continue

        current += timedelta(days=1)



    return all_ticks