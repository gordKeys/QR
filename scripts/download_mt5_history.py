from bootstrap import add_project_root
add_project_root()

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.symbol_universe import RECOMMENDED_SYMBOLS, load_symbol_candidates
from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError


TIMEFRAME_MAP = {
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
}


def normalize_rates(rates):
    if rates is None or len(rates) == 0:
        return None

    frame = pd.DataFrame(rates)
    if frame.empty:
        return None

    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.rename(columns={"time": "time"})

    for column in ("open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
        if column not in frame.columns:
            frame[column] = 0

    frame = frame[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    frame = frame.sort_values("time").drop_duplicates(subset=["time"])
    return frame


def export_symbol_history(broker, symbol, timeframe_name, start, end, output_dir):
    tf_attr = TIMEFRAME_MAP[timeframe_name]
    timeframe = getattr(broker.mt5, tf_attr)

    rates = broker.rates_range(symbol, timeframe, start, end)
    frame = normalize_rates(rates)
    if frame is None or frame.empty:
        chunks = []
        cursor = start
        chunk_size = pd.Timedelta(days=30)
        while cursor < end:
            chunk_end = min(cursor + chunk_size, end)
            chunk_rates = broker.rates_range(symbol, timeframe, cursor, chunk_end)
            chunk_frame = normalize_rates(chunk_rates)
            if chunk_frame is not None and not chunk_frame.empty:
                chunks.append(chunk_frame)
            cursor = chunk_end

        if chunks:
            frame = pd.concat(chunks, ignore_index=True)
            frame = frame.sort_values("time").drop_duplicates(subset=["time"])

    if frame is None or frame.empty:
        return None

    output_path = Path(output_dir) / f"{symbol}_{timeframe_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path, len(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", help="Symbols to download; defaults to recommended universe.")
    parser.add_argument("--timeframe", choices=list(TIMEFRAME_MAP.keys()), default="M5")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--from-date", type=str, help="Start date YYYY-MM-DD. Overrides --days.")
    parser.add_argument("--to-date", type=str, help="End date YYYY-MM-DD. Defaults to now UTC.")
    args = parser.parse_args()

    if args.from_date:
        start = datetime.fromisoformat(args.from_date).replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc) - pd.Timedelta(days=args.days)

    if args.to_date:
        end = datetime.fromisoformat(args.to_date).replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)

    symbols = load_symbol_candidates(args.symbols, available_symbols=RECOMMENDED_SYMBOLS)

    try:
        broker = MT5BrokerAdapter()
        broker.initialize()
    except MT5UnavailableError as exc:
        print(f"MT5 unavailable: {exc}")
        return 1

    print("\n=== MT5 HISTORY DOWNLOAD ===")
    print(f"Timeframe: {args.timeframe}")
    print(f"Range: {start.isoformat()} -> {end.isoformat()}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print(f"Symbols: {', '.join(symbols)}")
    print("-" * 80)

    outputs = []
    try:
        for symbol in symbols:
            resolved_symbol = broker.resolve_symbol(symbol)
            if resolved_symbol is None:
                print(f"{symbol:>8} | skipped (symbol not available)")
                continue

            if resolved_symbol != symbol:
                print(f"{symbol:>8} -> {resolved_symbol}")

            info = broker.symbol_info(resolved_symbol)
            if info is None:
                print(f"{symbol:>8} | skipped (symbol not available)")
                continue

            if not broker.mt5.symbol_select(resolved_symbol, True):
                print(f"{resolved_symbol:>8} | skipped (could not select in Market Watch)")
                continue

            result = export_symbol_history(
                broker=broker,
                symbol=resolved_symbol,
                timeframe_name=args.timeframe,
                start=start,
                end=end,
                output_dir=args.output_dir,
            )

            if result is None:
                print(f"{resolved_symbol:>8} | no data returned")
                continue

            output_path, row_count = result
            outputs.append(output_path)
            print(f"{resolved_symbol:>8} | saved {row_count:>8} rows -> {output_path}")
    finally:
        try:
            broker.shutdown()
        except Exception:
            pass

    print("\nDownloaded files:")
    for path in outputs:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
