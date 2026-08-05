# QuantFX-Research-Lab
This is to help me build a very good Quant trading system

## Running evaluations

- One launcher for everything: `./venv/bin/python run_project.py <mode>`
- Modes: `test`, `walkforward`, `sweep`, `combo`, `live`
- Full combined system: `./venv/bin/python run_project.py combo`
- Multi-symbol evaluation: `./venv/bin/python run_project.py test`
- Multi-symbol walk-forward: `./venv/bin/python run_project.py walkforward`
- Mean reversion sweep: `./venv/bin/python run_project.py sweep`
- Live VPS runner: `./venv/bin/python run_project.py live`

## Symbol inputs

- Pass a symbol: `./venv/bin/python run_project.py test --symbol GBPUSD`
- Pass a custom file: `./venv/bin/python run_project.py test --data data/GBPUSD_M5.csv`
- Run one symbol combo: `./venv/bin/python run_project.py combo --symbol EURUSD --symbol GBPUSD --symbol XAUUSD`

## Live notes

- `scripts/live_runner.py` is MT5-ready, but it needs the `MetaTrader5` package on the VPS.
- The live runner now halts only after 3 consecutive losses, then pauses for a cooldown window.
- For a quick dry run on VPS: `./venv/bin/python run_project.py live --dry-run --loop-once`
- Live logs are written to `logs/live_run.jsonl`.
- Logs rotate by broker date: `logs/live_run_YYYY-MM-DD.jsonl`
- Daily summaries are written to `logs/daily_summary_YYYY-MM-DD.json`
- The live runner skips only on no-signal or cooldown state.
- Leverage is controlled by your FTMO/broker account, not by the script; the bot now sizes to symbol limits and free margin before sending orders.

## Project layout

- `run_project.py` is the only top-level runnable file.
- Research and live tools live in `scripts/`.
- Core logic stays in `engine/` and `strategies/`.

## VPS setup

1. Clone the repo on the VPS.
2. Create and activate a virtual environment:
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3. Install dependencies:
   - `pip install --upgrade pip`
   - `pip install pandas numpy MetaTrader5`
4. Confirm the data files exist in `data/`:
   - `EURUSD_M5.csv`
   - `GBPUSD_M5.csv`
   - `XAUUSD_M5.csv`
5. Verify the launcher works:
   - `./venv/bin/python run_project.py test`
6. Run a walk-forward check:
   - `./venv/bin/python run_project.py walkforward`
7. Run demo live in dry-run mode first:
   - `./venv/bin/python run_project.py live --dry-run`
8. If the logs look good, run MT5 demo:
   - `./venv/bin/python run_project.py live`

## VPS tips

- Keep the MT5 terminal open and logged in.
- Use demo first; do not jump straight to FTMO live.
- Check `logs/` daily for errors, skips, and summary counts.
- If you restart the VPS, re-activate the venv before running the launcher.

## Local Mac testing

- Run the exact strategy set on your CSVs:
  - `./venv/bin/python run_project.py test`
- Run walk-forward on the same CSVs:
  - `./venv/bin/python run_project.py walkforward`
- Run the routed combo on the same CSVs:
  - `./venv/bin/python run_project.py combo`
- Run the mean reversion sweep on the same CSVs:
  - `./venv/bin/python run_project.py sweep`
- By default, these use:
  - `data/EURUSD_M5.csv`
  - `data/GBPUSD_M5.csv`
  - `data/XAUUSD_M5.csv`
