from __future__ import annotations

import csv
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ACCOUNT_TYPE_STANDARD = "STANDARD"
ACCOUNT_TYPE_SWING = "SWING"

PRAGUE_TZ = ZoneInfo("Europe/Prague")
LONDON_TZ = ZoneInfo("Europe/London")
FOREX_FACTORY_XML_URL = "https://www.forexfactory.com/calendar?week=this&export=xml"

FTMO_RESTRICTED_EVENTS = {
    "USD": [
        "FEDERAL FUNDS RATE",
        "NON-FARM EMPLOYMENT CHANGE",
        "UNEMPLOYMENT RATE",
        "ADVANCE GDP",
        "FOMC MEETING MINUTES",
        "CPI Y/Y",
    ],
    "EUR": ["MAIN REFINANCING RATE"],
    "GBP": ["OFFICIAL BANK RATE", "MPC VOTES", "CPI Y/Y"],
    "CAD": ["OVERNIGHT RATE", "BOC RATE STATEMENT", "CPI M/M", "EMPLOYMENT CHANGE", "UNEMPLOYMENT RATE"],
    "AUD": ["CASH RATE", "RBA STATEMENT", "EMPLOYMENT CHANGE", "UNEMPLOYMENT RATE", "CPI M/M", "CPI Y/Y", "GDP Q/Q"],
    "NZD": [
        "OFFICIAL CASH RATE",
        "RBNZ RATE STATEMENT",
        "EMPLOYMENT CHANGE",
        "UNEMPLOYMENT RATE",
        "CPI Q/Q",
        "GDP Q/Q",
    ],
    "CHF": ["SNB POLICY RATE"],
    "OIL": ["CRUDE OIL INVENTORIES"],
}


@dataclass(frozen=True)
class NewsEvent:
    event_time_utc: datetime
    currency: str
    title: str
    impact: str


@dataclass
class FTMOComplianceConfig:
    account_type: str = ACCOUNT_TYPE_STANDARD
    initial_balance: float = 0.0
    max_daily_loss_pct: float = 5.0
    max_total_loss_pct: float = 10.0
    news_pre_minutes: int = 2
    news_post_minutes: int = 2
    news_gap_minutes: int = 5
    flatten_before_news_minutes: int = 5
    market_close_buffer_minutes: int = 5
    calendar_url: str = FOREX_FACTORY_XML_URL
    state_path: Path = Path("logs/ftmo_compliance_state.json")


class FTMOComplianceEngine:
    def __init__(self, config: FTMOComplianceConfig):
        self.config = config
        self.state_path = Path(config.state_path)
        self._state = self._load_state()
        self._events_cache: list[NewsEvent] = []
        self._events_cache_time: datetime | None = None
        self._cached_prague_day: date | None = None
        self._cached_midnight_balance: float | None = None
        self._cached_highest_midnight_balance: float | None = None
        self._cached_initial_balance: float = float(config.initial_balance or 0.0)
        self._current_equity: float | None = None
        self._current_balance: float | None = None
        self._current_prague_now: datetime | None = None
        self._current_limit_state: dict[str, float] = {}

    def _load_state(self):
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prague_day": self._cached_prague_day.isoformat() if self._cached_prague_day else None,
            "midnight_balance": self._cached_midnight_balance,
            "highest_midnight_balance": self._cached_highest_midnight_balance,
            "initial_balance": self._cached_initial_balance,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _parse_number(value):
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _platform_now(self, now_utc: datetime) -> datetime:
        return now_utc.astimezone(timezone(timedelta(hours=3)))

    def _prague_now(self, now_utc: datetime) -> datetime:
        return now_utc.astimezone(PRAGUE_TZ)

    def refresh_account(self, *, balance: float, equity: float, now_utc: datetime | None = None):
        now_utc = now_utc or datetime.now(timezone.utc)
        prague_now = self._prague_now(now_utc)
        prague_day = prague_now.date()

        self._current_balance = float(balance)
        self._current_equity = float(equity)
        self._current_prague_now = prague_now

        stored_day = self._state.get("prague_day")
        stored_midnight_balance = self._parse_number(self._state.get("midnight_balance"))
        stored_highest_midnight_balance = self._parse_number(self._state.get("highest_midnight_balance"))
        stored_initial_balance = self._parse_number(self._state.get("initial_balance"))

        if stored_initial_balance is not None and stored_initial_balance > 0:
            self._cached_initial_balance = stored_initial_balance
        elif self._cached_initial_balance <= 0:
            self._cached_initial_balance = float(balance)

        if stored_day == prague_day.isoformat() and stored_midnight_balance is not None:
            self._cached_prague_day = prague_day
            self._cached_midnight_balance = stored_midnight_balance
            self._cached_highest_midnight_balance = max(
                float(stored_highest_midnight_balance or stored_midnight_balance or balance),
                float(stored_midnight_balance or balance),
                float(self._cached_initial_balance or balance),
            )
        else:
            self._cached_prague_day = prague_day
            self._cached_midnight_balance = float(balance)
            previous_highest = float(stored_highest_midnight_balance or stored_midnight_balance or balance)
            self._cached_highest_midnight_balance = max(previous_highest, self._cached_midnight_balance, float(self._cached_initial_balance or balance))
            self._state = {
                "prague_day": prague_day.isoformat(),
                "midnight_balance": self._cached_midnight_balance,
                "highest_midnight_balance": self._cached_highest_midnight_balance,
                "initial_balance": self._cached_initial_balance,
            }
            self._save_state()

        daily_loss_amount = self._cached_initial_balance * (self.config.max_daily_loss_pct / 100.0)
        total_loss_amount = self._cached_initial_balance * (self.config.max_total_loss_pct / 100.0)
        next_prague_midnight = datetime.combine(prague_day + timedelta(days=1), time.min, tzinfo=PRAGUE_TZ)
        seconds_to_reset = max(0.0, (next_prague_midnight - prague_now).total_seconds())
        daily_basis_balance = self._cached_midnight_balance
        total_basis_balance = max(float(self._cached_initial_balance), float(self._cached_highest_midnight_balance or self._cached_initial_balance))

        daily_limit = daily_basis_balance - daily_loss_amount
        total_limit = total_basis_balance - total_loss_amount

        self._current_limit_state = {
            "daily_basis_balance": daily_basis_balance,
            "total_basis_balance": total_basis_balance,
            "daily_limit": daily_limit,
            "total_limit": total_limit,
            "daily_buffer": self._current_equity - daily_limit,
            "total_buffer": self._current_equity - total_limit,
            "daily_loss_amount": daily_loss_amount,
            "total_loss_amount": total_loss_amount,
            "prague_midnight_next": next_prague_midnight,
            "seconds_to_reset": seconds_to_reset,
        }
        return self._current_limit_state

    def current_report(self):
        if self._current_equity is None or self._current_balance is None:
            return None

        prague_now = self._current_prague_now or self._prague_now(datetime.now(timezone.utc))
        current_pnl = self._current_equity - self._cached_initial_balance
        current_pnl_pct = (current_pnl / self._cached_initial_balance * 100.0) if self._cached_initial_balance else 0.0
        return {
            "prague_now": prague_now,
            "prague_midnight_next": self._current_limit_state.get("prague_midnight_next"),
            "seconds_to_reset": self._current_limit_state.get("seconds_to_reset"),
            "balance": self._current_balance,
            "equity": self._current_equity,
            "pnl": current_pnl,
            "pnl_pct": current_pnl_pct,
            "midnight_balance": self._cached_midnight_balance,
            "highest_midnight_balance": self._cached_highest_midnight_balance,
            "daily_basis_balance": self._current_limit_state.get("daily_basis_balance"),
            "total_basis_balance": self._current_limit_state.get("total_basis_balance"),
            "daily_loss_amount": self._current_limit_state.get("daily_loss_amount"),
            "total_loss_amount": self._current_limit_state.get("total_loss_amount"),
            "daily_limit": self._current_limit_state.get("daily_limit"),
            "daily_buffer": self._current_limit_state.get("daily_buffer"),
            "total_limit": self._current_limit_state.get("total_limit"),
            "total_buffer": self._current_limit_state.get("total_buffer"),
        }

    def is_loss_limit_breached(self):
        if self._current_equity is None:
            return False
        daily_limit = self._current_limit_state.get("daily_limit")
        total_limit = self._current_limit_state.get("total_limit")
        if daily_limit is not None and self._current_equity <= daily_limit:
            return True
        if total_limit is not None and self._current_equity <= total_limit:
            return True
        return False

    def _calendar_payload(self) -> list[NewsEvent]:
        if self._events_cache and self._events_cache_time:
            if datetime.now(timezone.utc) - self._events_cache_time < timedelta(hours=6):
                return self._events_cache

        try:
            with urllib.request.urlopen(self.config.calendar_url, timeout=20) as handle:
                payload = handle.read()
        except Exception:
            return []

        events: list[NewsEvent] = []
        try:
            root = ET.fromstring(payload)
            for element in root.findall(".//event"):
                title = (element.findtext("title") or element.findtext("descr") or "").strip()
                currency = (element.findtext("currency") or element.findtext("country") or "").strip().upper()
                date_text = (element.findtext("date") or "").strip()
                time_text = (element.findtext("time") or "").strip()
                impact = (element.findtext("impact") or "").strip().upper()
                event_time_utc = self._parse_event_time(date_text, time_text)
                if title and currency and event_time_utc is not None:
                    events.append(NewsEvent(event_time_utc=event_time_utc, currency=currency, title=title, impact=impact))
        except Exception:
            try:
                text = payload.decode("utf-8", errors="ignore")
                reader = csv.DictReader(text.splitlines())
                for row in reader:
                    title = (row.get("Title") or row.get("title") or "").strip()
                    currency = (row.get("Currency") or row.get("currency") or "").strip().upper()
                    date_text = (row.get("Date") or row.get("date") or "").strip()
                    time_text = (row.get("Time") or row.get("time") or "").strip()
                    impact = (row.get("Impact") or row.get("impact") or "").strip().upper()
                    event_time_utc = self._parse_event_time(date_text, time_text)
                    if title and currency and event_time_utc is not None:
                        events.append(NewsEvent(event_time_utc=event_time_utc, currency=currency, title=title, impact=impact))
            except Exception:
                return []

        self._events_cache = events
        self._events_cache_time = datetime.now(timezone.utc)
        return events

    @staticmethod
    def _parse_event_time(date_text: str, time_text: str) -> datetime | None:
        if not date_text:
            return None

        for date_format in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(date_text, date_format).date()
                break
            except ValueError:
                parsed_date = None
        if parsed_date is None:
            return None

        if not time_text or time_text.lower() in {"all day", "tentative", "all-day"}:
            return None

        normalized_time = time_text.strip().lower().replace(".", "")
        try:
            parsed_time = datetime.strptime(normalized_time, "%I:%M%p").time()
        except ValueError:
            try:
                parsed_time = datetime.strptime(normalized_time, "%H:%M").time()
            except ValueError:
                return None

        london_dt = datetime.combine(parsed_date, parsed_time, tzinfo=LONDON_TZ)
        return london_dt.astimezone(timezone.utc)

    @staticmethod
    def _symbol_matches_currency(symbol: str, currency: str) -> bool:
        upper_symbol = symbol.upper()
        currency = currency.upper()
        if currency == "OIL":
            return any(token in upper_symbol for token in ("USOIL", "UKOIL", "OIL"))
        return currency in upper_symbol

    @staticmethod
    def _restricted_event_matches(event: NewsEvent) -> bool:
        currency_rules = FTMO_RESTRICTED_EVENTS.get(event.currency.upper())
        if not currency_rules:
            return False

        title = event.title.upper()
        return any(keyword in title for keyword in currency_rules)

    def _symbol_event_windows(self, symbol: str):
        events = self._calendar_payload()
        for event in events:
            if not self._restricted_event_matches(event):
                continue
            if not self._symbol_matches_currency(symbol, event.currency):
                continue
            yield event

    def news_blackout(self, symbol: str, now_utc: datetime | None = None) -> dict | None:
        if self.config.account_type.upper() != ACCOUNT_TYPE_STANDARD:
            return None

        now_utc = now_utc or datetime.now(timezone.utc)
        pre = timedelta(minutes=self.config.news_pre_minutes)
        post = timedelta(minutes=self.config.news_post_minutes)
        anti_gap = timedelta(minutes=self.config.news_gap_minutes)
        flatten_before = timedelta(minutes=self.config.flatten_before_news_minutes)

        for event in self._symbol_event_windows(symbol):
            window_start = event.event_time_utc - pre
            window_end = event.event_time_utc + max(post, anti_gap)
            safe_entry_end = window_end
            flatten_at = event.event_time_utc - flatten_before
            if window_start <= now_utc <= window_end:
                return {
                    "active": True,
                    "reason": "news_blackout",
                    "event": event,
                    "window_start": window_start,
                    "window_end": window_end,
                    "allow_entries_after": safe_entry_end,
                }
            if now_utc >= flatten_at and now_utc < window_start:
                return {
                    "active": False,
                    "reason": "news_flatten",
                    "event": event,
                    "flatten_at": flatten_at,
                    "window_start": window_start,
                    "window_end": window_end,
                }

        return None

    def platform_market_closed(self, now_utc: datetime | None = None) -> bool:
        if self.config.account_type.upper() != ACCOUNT_TYPE_STANDARD:
            return False

        now_utc = now_utc or datetime.now(timezone.utc)
        platform_now = self._platform_now(now_utc)
        close_buffer = int(self.config.market_close_buffer_minutes or 5)
        if platform_now.weekday() in (5, 6):
            return True
        if platform_now.weekday() == 4 and platform_now.hour >= 23 and platform_now.minute >= max(0, 55 - close_buffer):
            return True
        if platform_now.weekday() == 0 and platform_now.hour == 0 and platform_now.minute < 5:
            return True
        return False

    def should_flatten_position_for_market_hours(self, symbol: str, now_utc: datetime | None = None) -> bool:
        if self.config.account_type.upper() != ACCOUNT_TYPE_STANDARD:
            return False
        return self.platform_market_closed(now_utc=now_utc)

    def should_block_new_entry(self, symbol: str, now_utc: datetime | None = None) -> dict | None:
        if self.is_loss_limit_breached():
            return {"reason": "loss_limit_breached"}

        if self.should_flatten_position_for_market_hours(symbol, now_utc=now_utc):
            return {"reason": "market_closed"}

        news_state = self.news_blackout(symbol, now_utc=now_utc)
        if news_state is not None:
            if news_state.get("active"):
                return {"reason": "news_blackout", **news_state}
            return {"reason": "news_flatten", **news_state}

        return None

    def should_flatten_position(self, symbol: str, now_utc: datetime | None = None) -> dict | None:
        if self.is_loss_limit_breached():
            return {"reason": "loss_limit_breached"}

        if self.should_flatten_position_for_market_hours(symbol, now_utc=now_utc):
            return {"reason": "market_closed"}

        news_state = self.news_blackout(symbol, now_utc=now_utc)
        if news_state is not None and news_state.get("reason") == "news_flatten":
            return news_state

        return None
