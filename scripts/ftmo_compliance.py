from __future__ import annotations

import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ACCOUNT_TYPE_STANDARD = "STANDARD"
ACCOUNT_TYPE_SWING = "SWING"
PRAGUE_TZ = ZoneInfo("Europe/Prague")
GHANA_TZ = ZoneInfo("Africa/Accra")
LONDON_TZ = ZoneInfo("Europe/London")
DEFAULT_CALENDAR_URL = "https://www.forexfactory.com/calendar?week=this&export=xml"


@dataclass
class FTMOComplianceConfig:
    account_type: str = ACCOUNT_TYPE_STANDARD
    initial_balance: float = 0.0
    max_daily_loss_pct: float = 5.0
    max_total_loss_pct: float = 10.0
    state_path: Path = Path("logs/ftmo_compliance_state.json")
    market_close_buffer_minutes: int = 5
    ghana_no_entry_start_hour: int = 19
    ghana_no_entry_end_hour: int = 1
    news_calendar_url: str = DEFAULT_CALENDAR_URL
    news_pre_minutes: int = 5
    news_post_minutes: int = 10
    max_spread_points: float = 30.0
    max_candle_atr: float = 2.5
    max_atr_ratio: float = 3.0


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
        self._news_cache = []
        self._news_cache_time = None

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

    def ghana_no_entry_window(self, now_utc=None):
        now = (now_utc or datetime.now(timezone.utc)).astimezone(GHANA_TZ)
        evening_window = now.weekday() in (0, 1, 2, 3, 4) and now.hour >= self.config.ghana_no_entry_start_hour
        overnight_window = now.weekday() in (1, 2, 3, 4, 5) and now.hour < self.config.ghana_no_entry_end_hour
        return evening_window or overnight_window

    def _news_events(self):
        now = datetime.now(timezone.utc)
        if self._news_cache_time and now - self._news_cache_time < timedelta(minutes=15):
            return self._news_cache
        try:
            with urllib.request.urlopen(self.config.news_calendar_url, timeout=8) as response:
                root = ET.fromstring(response.read())
        except Exception:
            self._news_cache = []
            self._news_cache_time = now
            return []
        events = []
        for item in root.findall('.//event'):
            impact = (item.findtext('impact') or '').strip().upper()
            currency = (item.findtext('currency') or item.findtext('country') or '').strip().upper()
            title = (item.findtext('title') or item.findtext('description') or '').strip()
            date_text = (item.findtext('date') or '').strip()
            time_text = (item.findtext('time') or '').strip().lower().replace('.', '')
            if 'HIGH' not in impact or not currency or not title or not date_text or not time_text:
                continue
            try:
                event_date = datetime.strptime(date_text, '%m/%d/%Y').date()
                event_time = datetime.strptime(time_text, '%I:%M%p').time()
            except ValueError:
                continue
            event_dt = datetime.combine(event_date, event_time, tzinfo=LONDON_TZ).astimezone(timezone.utc)
            events.append({'time': event_dt, 'currency': currency, 'title': title})
        self._news_cache = events
        self._news_cache_time = now
        return events

    @staticmethod
    def _news_affects_symbol(symbol, currency):
        symbol = symbol.upper()
        return currency in symbol or (currency == 'USD' and 'XAU' in symbol)

    def news_blackout(self, symbol, now_utc=None):
        if self.config.account_type.upper() != ACCOUNT_TYPE_STANDARD:
            return None
        now_utc = now_utc or datetime.now(timezone.utc)
        before = timedelta(minutes=self.config.news_pre_minutes)
        after = timedelta(minutes=self.config.news_post_minutes)
        for event in self._news_events():
            if not self._news_affects_symbol(symbol, event['currency']):
                continue
            if event['time'] - before <= now_utc <= event['time'] + after:
                return {'reason': 'news_blackout', 'event': event}
        return None

    def execution_market_guard(self, symbol, *, spread_points=None, candle_range=None, atr=None, typical_atr=None):
        if spread_points is not None and spread_points > self.config.max_spread_points:
            return {'reason': 'spread_too_wide', 'spread_points': spread_points}
        if atr and candle_range is not None and candle_range > atr * self.config.max_candle_atr:
            return {'reason': 'abnormal_candle_range', 'candle_range': candle_range, 'atr': atr}
        if atr and typical_atr and atr > typical_atr * self.config.max_atr_ratio:
            return {'reason': 'volatility_spike', 'atr': atr, 'typical_atr': typical_atr}
        return None

    def should_block_new_entry(self, symbol, now_utc=None):
        if self.is_loss_limit_breached():
            return {"reason": "loss_limit_breached"}
        if self.platform_market_closed(now_utc):
            return {"reason": "market_closed"}
        if self.ghana_no_entry_window(now_utc):
            return {"reason": "ghana_no_entry_window"}
        news_state = self.news_blackout(symbol, now_utc)
        if news_state:
            return news_state
        return None

    def should_flatten_position(self, symbol, now_utc=None):
        if self.is_loss_limit_breached():
            return {"reason": "loss_limit_breached"}
        if self.platform_market_closed(now_utc):
            return {"reason": "market_closed"}
        return None
