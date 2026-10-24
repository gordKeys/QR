"""Run identical research tests across a branch and its recent commits.

The runner freezes the selected Git commits, data hashes, date range, costs,
risk, position limits, and walk-forward schedule in manifest.json. It can
download one frozen M1/M5 snapshot from MT5 and never changes the caller's
checkout.
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
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("standardized_backtest_worker.py")
RISK_MATRIX = {
    "all_0.75pct": {"EURUSD": 0.0075, "GBPUSD": 0.0075, "XAUUSD": 0.0075},
    "eur1_gbp0.75_xau0.5": {"EURUSD": 0.01, "GBPUSD": 0.0075, "XAUUSD": 0.005},
    "eur1.5_gbp1_xau1": {"EURUSD": 0.015, "GBPUSD": 0.01, "XAUUSD": 0.01},
    "all_1pct": {"EURUSD": 0.01, "GBPUSD": 0.01, "XAUUSD": 0.01},
    "eur2_gbp1_xau1": {"EURUSD": 0.02, "GBPUSD": 0.01, "XAUUSD": 0.01},
}
ATR_MATRIX = {
    "baseline": {
        "EURUSD": {"sl": 2.5, "tp": 5.0}, "GBPUSD": {"sl": 2.5, "tp": 5.0}, "XAUUSD": {"sl": 1.5, "tp": 2.25},
    },
    "balanced": {
        "EURUSD": {"sl": 2.5, "tp": 4.0}, "GBPUSD": {"sl": 2.5, "tp": 4.0}, "XAUUSD": {"sl": 1.5, "tp": 2.0},
    },
    "tighter": {
        "EURUSD": {"sl": 2.0, "tp": 3.5}, "GBPUSD": {"sl": 2.0, "tp": 3.5}, "XAUUSD": {"sl": 1.25, "tp": 1.75},
    },
    "wider": {
        "EURUSD": {"sl": 3.0, "tp": 5.0}, "GBPUSD": {"sl": 3.0, "tp": 4.5}, "XAUUSD": {"sl": 1.75, "tp": 2.5},
    },
    "eur_tight": {
        "EURUSD": {"sl": 2.0, "tp": 4.0}, "GBPUSD": {"sl": 2.5, "tp": 5.0}, "XAUUSD": {"sl": 1.5, "tp": 2.25},
    },
    "eur_wide": {
        "EURUSD": {"sl": 3.0, "tp": 5.5}, "GBPUSD": {"sl": 2.5, "tp": 5.0}, "XAUUSD": {"sl": 1.5, "tp": 2.25},
    },
    "gbp_tight": {
        "EURUSD": {"sl": 2.5, "tp": 5.0}, "GBPUSD": {"sl": 2.0, "tp": 3.5}, "XAUUSD": {"sl": 1.5, "tp": 2.25},
    },
    "gbp_wide": {
        "EURUSD": {"sl": 2.5, "tp": 5.0}, "GBPUSD": {"sl": 3.0, "tp": 4.5}, "XAUUSD": {"sl": 1.5, "tp": 2.25},
    },
    "xau_tight": {
        "EURUSD": {"sl": 2.5, "tp": 5.0}, "GBPUSD": {"sl": 2.5, "tp": 5.0}, "XAUUSD": {"sl": 1.25, "tp": 1.75},
    },
    "xau_wide": {
        "EURUSD": {"sl": 2.5, "tp": 5.0}, "GBPUSD": {"sl": 2.5, "tp": 5.0}, "XAUUSD": {"sl": 1.75, "tp": 2.5},
    },
}


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
    parser.add_argument("--download-mt5", action="store_true")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--from-date")
    parser.add_argument("--to-date")
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "XAUUSD"])
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--risk", default='{"EURUSD":0.01,"GBPUSD":0.01,"XAUUSD":0.01}')
    parser.add_argument("--risk-matrix", action="store_true",
                        help="Run the built-in five-profile risk matrix into separate folders.")
    parser.add_argument("--atr", default=json.dumps(ATR_MATRIX["baseline"]),
                        help="JSON mapping symbol to SL/TP ATR multipliers.")
    parser.add_argument("--atr-matrix", action="store_true",
                        help="Run the built-in per-symbol ATR matrix into separate folders.")
    parser.add_argument("--daily-loss-pct", type=float, default=0.05)
    parser.add_argument("--total-loss-pct", type=float, default=0.10)
    parser.add_argument("--soft-stop-pct", type=float, default=0.03,
                        help="Internal daily entry stop; keep below --daily-loss-pct.")
    parser.add_argument("--max-open-positions", type=int, default=2)
    parser.add_argument("--walkforward-window-days", type=int, default=30)
    parser.add_argument("--walkforward-step-days", type=int, default=15)
    parser.add_argument("--walkforward-windows", type=int, default=10)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("live", "ny_blocked", "gbp_london_quality", "ny_tp_1atr", "target_session_mix", "target_session_mix_xau_ny", "target_session_mix_xau_ny_pause", "diagnostic", "suite"),
        default="live",
        help="Run one profile or all four profiles with suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.soft_stop_pct >= args.daily_loss_pct:
        raise ValueError("--soft-stop-pct must be below the FTMO --daily-loss-pct")
    started = time.perf_counter()
    profiles = ["live", "ny_blocked", "gbp_london_quality", "ny_tp_1atr", "diagnostic"] if args.profile == "suite" else [args.profile]
    print(
        f"[standardized] starting branch={args.branch} "
        f"commits={len(args.commits) if args.commits else args.previous_count + 1} "
        f"profiles={','.join(profiles)} symbols={','.join(args.symbols)}",
        flush=True,
    )
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    symbols = [symbol.upper() for symbol in args.symbols]
    if args.risk_matrix and args.atr_matrix:
        raise ValueError("--risk-matrix and --atr-matrix cannot be combined")
    risk_sets = list(RISK_MATRIX.items()) if args.risk_matrix else [("single", json.loads(args.risk))]
    atr_sets = list(ATR_MATRIX.items()) if args.atr_matrix else [("single", json.loads(args.atr))]
    if args.download_mt5:
        downloader = ROOT / "scripts" / "download_mt5_history.py"
        for timeframe in ("M1", "M5"):
            print(f"[standardized] downloading {timeframe} history...", flush=True)
            download_args = [
                sys.executable, str(downloader), "--symbols", *symbols,
                "--timeframe", timeframe, "--output-dir", str(data_dir),
            ]
            if args.from_date:
                download_args += ["--from-date", args.from_date]
            else:
                download_args += ["--days", str(args.days)]
            if args.to_date:
                download_args += ["--to-date", args.to_date]
            subprocess.run(download_args, cwd=ROOT, check=True)
            print(f"[standardized] {timeframe} download complete", flush=True)
    data_files = {}
    for symbol in symbols:
        data_files[symbol] = {}
        for timeframe in ("M1", "M5"):
            matches = sorted(data_dir.glob(f"*{symbol}*_{timeframe}.csv"))
            if not matches:
                raise FileNotFoundError(f"Missing frozen {timeframe} data for {symbol}: {data_dir}")
            data_files[symbol][timeframe] = {"path": str(matches[0]), "sha256": sha256(matches[0]), "bytes": matches[0].stat().st_size}
    commits = args.commits or commit_list(args.branch, args.previous_count + 1)
    print(
        f"[standardized] frozen snapshot ready; testing {len(commits)} commit(s)",
        flush=True,
    )
    manifest = {
        "runner": "standardized_backtest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "branch_requested": args.branch,
        "commits_requested": commits,
        "data_dir": str(data_dir),
        "data_files": data_files,
        "symbols": symbols,
        "profiles": profiles,
        "initial_balance": args.initial_balance,
        "risk_matrix": dict(risk_sets) if args.risk_matrix else None,
        "risk_by_symbol": None if args.risk_matrix else risk_sets[0][1],
        "atr_matrix": dict(atr_sets) if args.atr_matrix else None,
        "atr_by_symbol": None if args.atr_matrix else atr_sets[0][1],
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
            "M5 generates signals; sequential M1 candles resolve intrabar exits.",
            "Data is downloaded only once when --download-mt5 is supplied.",
            "Every commit receives the same snapshot and command parameters.",
            "A failed or incompatible historical commit is recorded, not silently substituted.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    for risk_name, risk in risk_sets:
        for atr_name, atr in atr_sets:
            variant_name = atr_name if args.atr_matrix else risk_name
            variant_output = output / variant_name if (args.atr_matrix or args.risk_matrix) else output
            for commit_number, commit in enumerate(commits, start=1):
                short = commit[:12]
                commit_output = variant_output / short
                worktree = Path(tempfile.mkdtemp(prefix=f"standardized-{short}-"))
                try:
                    command("git", "worktree", "add", "--detach", str(worktree), commit)
                    for profile in profiles:
                        profile_started = time.perf_counter()
                        profile_output = commit_output / profile if len(profiles) > 1 else commit_output
                        print(
                            f"[standardized] risk={risk_name} atr={atr_name} commit {commit_number}/{len(commits)} {short} "
                            f"profile={profile}: starting worker",
                            flush=True,
                        )
                        worker_copy = worktree / "scripts" / WORKER.name
                        shutil.copy2(WORKER, worker_copy)
                        shutil.copy2(ROOT / "scripts" / "research_backtest.py", worktree / "scripts" / "research_backtest.py")
                        worker_args = [
                            sys.executable, str(worker_copy),
                            "--data-dir", str(data_dir), "--output", str(profile_output),
                            "--initial-balance", str(args.initial_balance), "--symbols", *symbols,
                            "--risk-json", json.dumps(risk, separators=(",", ":")),
                            "--atr-json", json.dumps(atr, separators=(",", ":")),
                            "--daily-loss-pct", str(args.daily_loss_pct), "--total-loss-pct", str(args.total_loss_pct),
                            "--soft-stop-pct", str(args.soft_stop_pct),
                            "--max-open-positions", str(args.max_open_positions),
                            "--walkforward-window-days", str(args.walkforward_window_days),
                            "--walkforward-step-days", str(args.walkforward_step_days),
                            "--walkforward-windows", str(args.walkforward_windows), "--profile", profile,
                        ]
                        if args.diagnostic or profile == "diagnostic":
                            worker_args.append("--diagnostic")
                        env = os.environ.copy()
                        env["PYTHONPATH"] = str(worktree) + os.pathsep + str(worktree / "scripts")
                        process = subprocess.Popen(worker_args, cwd=worktree, env=env, text=True,
                                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                        worker_output = []
                        assert process.stdout is not None
                        for line in process.stdout:
                            line = line.rstrip()
                            worker_output.append(line)
                            print(f"[worker {risk_name}/{atr_name}/{short}/{profile}] {line}", flush=True)
                        result_code = process.wait()
                        item = {"risk_profile": risk_name, "risk_by_symbol": risk, "atr_profile": atr_name, "atr_by_symbol": atr, "commit": commit, "profile": profile,
                                "status": "passed" if result_code == 0 else "failed",
                                "output": str(profile_output)}
                        if result_code:
                            item["error"] = "\n".join(worker_output)[-4000:]
                        else:
                            item["report"] = json.loads((profile_output / "report.json").read_text(encoding="utf-8"))
                        results.append(item)
                        print(f"[standardized] {risk_name}/{atr_name}/{short}/{profile}: {item['status']} "
                              f"({time.perf_counter() - profile_started:.1f}s)", flush=True)
                finally:
                    command("git", "worktree", "remove", "--force", str(worktree), check=False)
                    shutil.rmtree(worktree, ignore_errors=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "results": str(output / "results.json"),
                "runs": len(results),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
