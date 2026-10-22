"""Run identical research tests across a branch and its recent commits.

The runner freezes the selected Git commits, data hashes, date range, costs,
risk, position limits, and walk-forward schedule in manifest.json. It never
downloads data and never changes the caller's checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("standardized_backtest_worker.py")


def command(*parts: str, cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(parts, cwd=cwd, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(parts)}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_list(branch: str, count: int) -> list[str]:
    return command("git", "rev-list", "--first-parent", f"--max-count={count}", branch).splitlines()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="rollback-06fcc6d")
    parser.add_argument("--commits", nargs="+", help="Exact commits; overrides --branch")
    parser.add_argument("--previous-count", type=int, default=10)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "XAUUSD"])
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--risk", default='{"EURUSD":0.01,"GBPUSD":0.01,"XAUUSD":0.01}')
    parser.add_argument("--daily-loss-pct", type=float, default=0.05)
    parser.add_argument("--total-loss-pct", type=float, default=0.10)
    parser.add_argument("--max-open-positions", type=int, default=2)
    parser.add_argument("--walkforward-window-days", type=int, default=30)
    parser.add_argument("--walkforward-step-days", type=int, default=15)
    parser.add_argument("--walkforward-windows", type=int, default=10)
    parser.add_argument("--diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    symbols = [symbol.upper() for symbol in args.symbols]
    risk = json.loads(args.risk)
    data_files = {}
    for symbol in symbols:
        matches = sorted(data_dir.glob(f"*{symbol}*_M5.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing frozen M5 data for {symbol}: {data_dir}")
        data_files[symbol] = {"path": str(matches[0]), "sha256": sha256(matches[0]), "bytes": matches[0].stat().st_size}
    commits = args.commits or commit_list(args.branch, args.previous_count + 1)
    manifest = {
        "runner": "standardized_backtest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "branch_requested": args.branch,
        "commits_requested": commits,
        "data_dir": str(data_dir),
        "data_files": data_files,
        "symbols": symbols,
        "initial_balance": args.initial_balance,
        "risk_by_symbol": risk,
        "commission_round_turn_per_lot": 5.0,
        "slippage_points": 1.0,
        "daily_loss_pct": args.daily_loss_pct,
        "total_loss_pct": args.total_loss_pct,
        "max_open_positions": args.max_open_positions,
        "walkforward": {
            "window_days": args.walkforward_window_days,
            "step_days": args.walkforward_step_days,
            "windows_requested": args.walkforward_windows,
        },
        "diagnostic_mode": args.diagnostic,
        "notes": [
            "Data is never downloaded by this runner.",
            "Every commit receives the same snapshot and command parameters.",
            "A failed or incompatible historical commit is recorded, not silently substituted.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    for commit in commits:
        short = commit[:12]
        commit_output = output / short
        worktree = Path(tempfile.mkdtemp(prefix=f"standardized-{short}-"))
        try:
            command("git", "worktree", "add", "--detach", str(worktree), commit)
            worker_copy = worktree / "scripts" / WORKER.name
            shutil.copy2(WORKER, worker_copy)
            worker_args = [
                sys.executable,
                str(worker_copy),
                "--data-dir", str(data_dir),
                "--output", str(commit_output),
                "--initial-balance", str(args.initial_balance),
                "--symbols", *symbols,
                "--risk-json", json.dumps(risk, separators=(",", ":")),
                "--daily-loss-pct", str(args.daily_loss_pct),
                "--total-loss-pct", str(args.total_loss_pct),
                "--max-open-positions", str(args.max_open_positions),
                "--walkforward-window-days", str(args.walkforward_window_days),
                "--walkforward-step-days", str(args.walkforward_step_days),
                "--walkforward-windows", str(args.walkforward_windows),
            ]
            if args.diagnostic:
                worker_args.append("--diagnostic")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(worktree) + os.pathsep + str(worktree / "scripts")
            result = subprocess.run(worker_args, cwd=worktree, env=env, text=True, capture_output=True)
            item = {"commit": commit, "status": "passed" if result.returncode == 0 else "failed", "output": str(commit_output)}
            if result.returncode:
                item["error"] = result.stderr[-4000:]
            else:
                item["report"] = json.loads((commit_output / "report.json").read_text(encoding="utf-8"))
            results.append(item)
        finally:
            command("git", "worktree", "remove", "--force", str(worktree), check=False)
            shutil.rmtree(worktree, ignore_errors=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"manifest": str(output / "manifest.json"), "results": str(output / "results.json"), "commits": len(results)}, indent=2))


if __name__ == "__main__":
    main()
