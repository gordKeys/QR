from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ACCOUNT_TYPE_STANDARD = "STANDARD"
ACCOUNT_TYPE_SWING = "SWING"
PRAGUE_TZ = ZoneInfo("Europe/Prague")


@dataclass
class FTMOComplianceConfig:
    account_type: str = ACCOUNT_TYPE_STANDARD
    initial_balance: float = 0.0
    max_daily_loss_pct: float = 5.0
    max_total_loss_pct: float = 10.0
    state_path: Path = Path("logs/ftmo_compliance_state.json")
    market_close_buffer_minutes: int = 5


class FTMOComplianceEngine:
    """Execution guard; it does not alter strategy signals or position sizing."""

    def __init__(self, config: FTMOComplianceConfig):
        self.config = config
        self.state_path = Path(config.state_path)
        self.state = self._load_state()
        self.balance = self.equity = self.prague_now = None
        self.daily_base = self.total_base = None
        self.daily_limit = self.total_limit = None
        self.seconds_to_reset = None

    def _load_state(self):
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def refresh_account(self, *, balance, equity, now_utc=None):
        now_utc = now_utc or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        self.balance = float(balance)
        self.equity = float(equity)
        self.prague_now = now_utc.astimezone(PRAGUE_TZ)
        current_day = self.prague_now.date().isoformat()
        initial = float(self.state.get("initial_balance") or self.config.initial_balance or balance)
        if self.state.get("prague_day") != current_day:
            self.state.update({
                "initial_balance": initial,
                "prague_day": current_day,
                "daily_base": self.balance,
                "highest_daily_base": max(self.balance, float(self.state.get("highest_daily_base") or self.balance)),
            })
            self._save_state()
        self.daily_base = float(self.state.get("daily_base") or self.balance)
        highest_base = float(self.state.get("highest_daily_base") or self.daily_base)
        self.total_base = max(initial, highest_base)
        daily_loss = initial * self.config.max_daily_loss_pct / 100.0
        total_loss = initial * self.config.max_total_loss_pct / 100.0
        self.daily_limit = self.daily_base - daily_loss
        self.total_limit = self.total_base - total_loss
        next_midnight = datetime.combine(self.prague_now.date() + timedelta(days=1), time.min, tzinfo=PRAGUE_TZ)
        self.seconds_to_reset = max(0.0, (next_midnight - self.prague_now).total_seconds())
        return self.current_report()

    def current_report(self):
        if self.equity is None:
            return None
        initial = float(self.state.get("initial_balance") or self.config.initial_balance or self.balance or 0.0)
        return {
            "prague_now": self.prague_now,
            "seconds_to_reset": self.seconds_to_reset,
            "balance": self.balance,
            "equity": self.equity,
            "daily_base": self.daily_base,
            "daily_loss_amount": initial * self.config.max_daily_loss_pct / 100.0,
            "daily_limit": self.daily_limit,
            "daily_buffer": self.equity - self.daily_limit,
            "total_base": self.total_base,
            "total_loss_amount": initial * self.config.max_total_loss_pct / 100.0,
            "total_limit": self.total_limit,
            "total_buffer": self.equity - self.total_limit,
        }

    def is_loss_limit_breached(self):
        return self.equity is not None and (self.equity <= self.daily_limit or self.equity <= self.total_limit)

    def platform_market_closed(self, now_utc=None):
        if self.config.account_type.upper() != ACCOUNT_TYPE_STANDARD:
            return False
        now = (now_utc or datetime.now(timezone.utc)).astimezone(PRAGUE_TZ)
        if now.weekday() in (5, 6):
            return True
        buffer = self.config.market_close_buffer_minutes
        if now.weekday() == 4 and now.hour == 23 and now.minute >= 60 - buffer:
            return True
        return now.weekday() == 0 and now.hour == 0 and now.minute < buffer

    def should_block_new_entry(self, symbol, now_utc=None):
        if self.is_loss_limit_breached():
            return {"reason": "loss_limit_breached"}
        if self.platform_market_closed(now_utc):
            return {"reason": "market_closed"}
        return None

    def should_flatten_position(self, symbol, now_utc=None):
        if self.is_loss_limit_breached():
            return {"reason": "loss_limit_breached"}
        if self.platform_market_closed(now_utc):
            return {"reason": "market_closed"}
        return None
