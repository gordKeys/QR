from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import re
from typing import Iterable

import requests
from bs4 import BeautifulSoup


HIGH_IMPACT_KEYWORDS = (
    "interest rate",
    "rate statement",
    "press conference",
    "monetary policy",
    "central bank",
    "fomc",
    "fed chair",
    "powell",
    "cpi",
    "consumer price",
    "ppi",
    "pce",
    "nfp",
    "non-farm",
    "employment change",
    "unemployment rate",
    "average hourly earnings",
    "gdp",
    "retail sales",
    "core retail sales",
    "ism",
    "manufacturing pmi",
    "services pmi",
    "flash manufacturing pmi",
    "flash services pmi",
    "bank holiday",
    "holiday",
    "speaker",
)

TIME_RE = re.compile(r"^(?:\d{1,2}:\d{2}(?:am|pm)|All Day)$", re.IGNORECASE)
DATE_RE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Z][a-z]{2}\s+\d{1,2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class NewsEvent:
    event_time: datetime
    currency: str
    title: str
    impact: str = "unknown"
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None

    def is_high_impact(self) -> bool:
        impact_text = (self.impact or "").lower()
        title_text = self.title.lower()
        if "high" in impact_text:
            return True
        return any(keyword in title_text for keyword in HIGH_IMPACT_KEYWORDS)


class ForexFactoryNewsClient:
    def __init__(self, calendar_url="https://www.forexfactory.com/calendar?day=today", timeout_seconds=20):
        self.calendar_url = calendar_url
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    @staticmethod
    def _parse_event_time(date_label: str, time_label: str, calendar_timezone: ZoneInfo, reference_year: int):
        if not date_label or not time_label:
            return None

        normalized_date = date_label.strip()
        if not DATE_RE.match(normalized_date):
            return None

        normalized_time = time_label.strip()
        if normalized_time.lower() == "all day":
            normalized_time = "12:00am"

        try:
            parsed_date = datetime.strptime(f"{normalized_date} {reference_year}", "%a %b %d %Y")
            parsed_time = datetime.strptime(normalized_time.lower(), "%I:%M%p")
        except ValueError:
            return None

        merged = parsed_date.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
            tzinfo=calendar_timezone,
        )
        return merged

    @staticmethod
    def _row_cells(row):
        cells = []
        for cell in row.find_all(["td", "th"]):
            text = cell.get_text(" ", strip=True)
            cells.append((text, cell))
        return cells

    @staticmethod
    def _row_impact(row_cell_nodes) -> str:
        for _, cell in row_cell_nodes:
            for image in cell.find_all("img"):
                candidate = " ".join(
                    part
                    for part in (
                        image.get("alt"),
                        image.get("title"),
                        " ".join(image.get("class", [])) if image.get("class") else None,
                        image.get("src"),
                    )
                    if part
                ).lower()
                if "high" in candidate:
                    return "high"
                if "med" in candidate or "medium" in candidate:
                    return "medium"
                if "low" in candidate:
                    return "low"
        return "unknown"

    def fetch_events(self) -> list[NewsEvent]:
        response = self.session.get(self.calendar_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        page_text = soup.get_text(" ", strip=True)

        timezone_match = re.search(r"Calendar Time Zone:\s*([A-Za-z0-9_/\-+ ]+)", page_text)
        calendar_timezone_name = timezone_match.group(1).strip() if timezone_match else "UTC"
        if " " in calendar_timezone_name and "/" not in calendar_timezone_name:
            calendar_timezone_name = calendar_timezone_name.split(" ", 1)[0]
        try:
            calendar_timezone = ZoneInfo(calendar_timezone_name)
        except Exception:
            calendar_timezone = timezone.utc

        reference_year = datetime.now(calendar_timezone).year
        event_tables = []
        for table in soup.find_all("table"):
            table_text = table.get_text(" ", strip=True)
            if "Currency" in table_text and "Detail" in table_text:
                event_tables.append(table)

        events: list[NewsEvent] = []
        for table in event_tables:
            current_date_label = None
            for row in table.find_all("tr"):
                row_cells = self._row_cells(row)
                if not row_cells:
                    continue

                row_text = " ".join(text for text, _ in row_cells if text)
                if DATE_RE.match(row_text):
                    current_date_label = row_text
                    continue

                if current_date_label is None:
                    continue

                currency_index = None
                time_index = None
                for index, (text, _) in enumerate(row_cells):
                    if currency_index is None and CURRENCY_RE.match(text):
                        currency_index = index
                    elif time_index is None and TIME_RE.match(text):
                        time_index = index

                if currency_index is None or time_index is None:
                    continue

                currency = row_cells[currency_index][0]
                time_label = row_cells[time_index][0]
                event_time = self._parse_event_time(current_date_label, time_label, calendar_timezone, reference_year)
                if event_time is None:
                    continue

                title = None
                for index in range(min(len(row_cells), currency_index + 1), len(row_cells)):
                    candidate = row_cells[index][0].strip()
                    if candidate and not CURRENCY_RE.match(candidate) and not TIME_RE.match(candidate):
                        if any(character.isalpha() for character in candidate):
                            title = candidate
                            break
                if title is None:
                    continue

                impact = self._row_impact(row_cells)
                actual = forecast = previous = None
                numeric_cells = [text for text, _ in row_cells if text and text not in {current_date_label, time_label, currency, title}]
                if len(numeric_cells) >= 2:
                    actual = numeric_cells[-3] if len(numeric_cells) >= 3 else None
                    forecast = numeric_cells[-2] if len(numeric_cells) >= 2 else None
                    previous = numeric_cells[-1]

                events.append(
                    NewsEvent(
                        event_time=event_time,
                        currency=currency,
                        title=title,
                        impact=impact,
                        actual=actual,
                        forecast=forecast,
                        previous=previous,
                    )
                )

        return events


class NewsBlackoutGuard:
    def __init__(
        self,
        enabled=False,
        before_minutes=30,
        after_minutes=15,
        impact_level="high",
        calendar_url="https://www.forexfactory.com/calendar?day=today",
        cache_minutes=1,
        cache_path="logs/news_calendar_cache.json",
    ):
        self.enabled = enabled
        self.before_minutes = int(before_minutes)
        self.after_minutes = int(after_minutes)
        self.impact_level = impact_level
        self.cache_minutes = int(cache_minutes)
        self.cache_path = Path(cache_path)
        self.client = ForexFactoryNewsClient(calendar_url=calendar_url)
        self._cached_at: datetime | None = None
        self._events: list[NewsEvent] = []

    @staticmethod
    def symbol_currencies(symbol: str) -> set[str]:
        symbol = symbol.upper()
        if symbol == "EURUSD":
            return {"EUR", "USD"}
        if symbol == "GBPUSD":
            return {"GBP", "USD"}
        if symbol == "USDJPY":
            return {"USD", "JPY"}
        if symbol == "USDCHF":
            return {"USD", "CHF"}
        if symbol == "XAUUSD":
            return {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
        return {"USD"}

    def _cache_is_fresh(self, now: datetime) -> bool:
        if self._cached_at is None:
            return False
        return (now - self._cached_at) < timedelta(minutes=self.cache_minutes)

    def _load_cache(self) -> list[NewsEvent]:
        if not self.cache_path.exists():
            return []
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        events = []
        for item in payload.get("events", []):
            try:
                events.append(
                    NewsEvent(
                        event_time=datetime.fromisoformat(item["event_time"]),
                        currency=item["currency"],
                        title=item["title"],
                        impact=item.get("impact", "unknown"),
                        actual=item.get("actual"),
                        forecast=item.get("forecast"),
                        previous=item.get("previous"),
                    )
                )
            except Exception:
                continue
        cached_at_text = payload.get("cached_at")
        if cached_at_text:
            try:
                self._cached_at = datetime.fromisoformat(cached_at_text)
            except Exception:
                self._cached_at = None
        return events

    def _save_cache(self, events: Iterable[NewsEvent], cached_at: datetime):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cached_at": cached_at.isoformat(),
            "events": [
                {
                    "event_time": event.event_time.isoformat(),
                    "currency": event.currency,
                    "title": event.title,
                    "impact": event.impact,
                    "actual": event.actual,
                    "forecast": event.forecast,
                    "previous": event.previous,
                }
                for event in events
            ],
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def refresh(self, now: datetime):
        if not self.enabled:
            return self._events

        if self._cache_is_fresh(now):
            return self._events

        try:
            events = self.client.fetch_events()
            self._events = events
            self._cached_at = now
            self._save_cache(events, now)
        except Exception:
            cached_events = self._load_cache()
            if cached_events:
                self._events = cached_events
            else:
                self._events = []
        return self._events

    def events(self):
        return list(self._events)

    def should_block(self, symbol: str, now: datetime):
        if not self.enabled:
            return False, None

        if not self._events:
            self.refresh(now)

        relevant_currencies = self.symbol_currencies(symbol)
        now_utc = now.astimezone(timezone.utc)
        for event in self._events:
            if event.currency not in relevant_currencies:
                continue
            if self.impact_level == "high" and not event.is_high_impact():
                continue
            if self.impact_level == "medium":
                title_text = event.title.lower()
                if not (event.is_high_impact() or any(keyword in title_text for keyword in ("pmi", "cpi", "ppi", "gdp", "sales", "employment", "rate", "speech", "confidence"))):
                    continue

            event_time_utc = event.event_time.astimezone(timezone.utc)
            window_start = event_time_utc - timedelta(minutes=self.before_minutes)
            window_end = event_time_utc + timedelta(minutes=self.after_minutes)
            if window_start <= now_utc <= window_end:
                return True, event

        return False, None
