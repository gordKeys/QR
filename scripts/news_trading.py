from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from news_calendar import NewsEvent


@dataclass
class NewsTradePlan:
    symbol: str
    signal: int
    entry: float
    stop: float
    target: float
    size: float
    event: NewsEvent
    reason: str


class NewsSniperModule:
    def __init__(
        self,
        *,
        enabled=False,
        symbols=None,
        magic=26072027,
        risk_pct=0.25,
        rr=1.4,
        post_delay_minutes=1,
        post_window_minutes=10,
        lookback_bars=12,
        breakout_buffer_points=12.0,
        spread_multiplier=1.25,
        min_atr_points=8.0,
        event_cooldown_minutes=30,
    ):
        self.enabled = enabled
        self.symbols = [symbol.upper() for symbol in (symbols or ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])]
        self.magic = int(magic)
        self.risk_pct = float(risk_pct)
        self.rr = float(rr)
        self.post_delay_minutes = int(post_delay_minutes)
        self.post_window_minutes = int(post_window_minutes)
        self.lookback_bars = int(lookback_bars)
        self.breakout_buffer_points = float(breakout_buffer_points)
        self.spread_multiplier = float(spread_multiplier)
        self.min_atr_points = float(min_atr_points)
        self.event_cooldown_minutes = int(event_cooldown_minutes)
        self.fired_events: set[str] = set()

    @staticmethod
    def _event_key(event: NewsEvent, symbol: str) -> str:
        return f"{symbol.upper()}|{event.currency}|{event.title}|{event.event_time.isoformat()}"

    @staticmethod
    def _symbol_currencies(symbol: str) -> set[str]:
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

    def _relevant_events(self, symbol: str, events: Iterable[NewsEvent], now: datetime) -> list[NewsEvent]:
        now_utc = now.astimezone(timezone.utc)
        currencies = self._symbol_currencies(symbol)
        relevant = []
        for event in events:
            if event.currency not in currencies:
                continue
            if not event.is_high_impact():
                continue
            event_time_utc = event.event_time.astimezone(timezone.utc)
            start = event_time_utc + timedelta(minutes=self.post_delay_minutes)
            end = event_time_utc + timedelta(minutes=self.post_window_minutes)
            if start <= now_utc <= end:
                relevant.append(event)
        return relevant

    @staticmethod
    def _rates_to_frame(rates):
        if rates is None or len(rates) == 0:
            return None
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("time")
        frame = frame[["open", "high", "low", "close", "tick_volume"]]
        return frame

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14):
        close = frame["close"]
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - close.shift()).abs(),
                (frame["low"] - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    def _build_plan(self, symbol, broker, event, now, equity):
        timeframe = getattr(broker.mt5, "TIMEFRAME_M1", getattr(broker.mt5, "TIMEFRAME_M5", None))
        if timeframe is None:
            return None

        rates = broker.rates_copy(symbol, timeframe, 120)
        frame = self._rates_to_frame(rates)
        if frame is None or len(frame) < max(self.lookback_bars + 5, 20):
            return None

        atr_series = self._atr(frame)
        current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
        info = broker.symbol_info(symbol)
        tick = broker.symbol_tick(symbol)
        if info is None or tick is None:
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        if point <= 0 or current_atr / point < self.min_atr_points:
            return None

        spread = 0.0
        if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
            spread = float(tick.ask - tick.bid)
        spread_limit = max(point * self.breakout_buffer_points, current_atr * self.spread_multiplier)
        if spread_limit > 0 and spread > spread_limit:
            return None

        event_time = event.event_time.astimezone(timezone.utc)
        window = frame.loc[frame.index <= now.astimezone(timezone.utc)]
        if window.empty:
            return None

        pre_event = window.loc[window.index < event_time]
        post_event = window.loc[window.index >= event_time]
        if len(pre_event) < self.lookback_bars or len(post_event) < 2:
            return None

        anchor_slice = pre_event.tail(self.lookback_bars)
        reaction_bars = post_event.iloc[:-1]
        if reaction_bars.empty:
            return None

        anchor_high = float(anchor_slice["high"].max())
        anchor_low = float(anchor_slice["low"].min())
        current_bar = post_event.iloc[-1]
        current_close = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        breakout_buffer = max(point * self.breakout_buffer_points, current_atr * 0.20, spread * 1.2)
        range_high = max(anchor_high, float(reaction_bars["high"].max()))
        range_low = min(anchor_low, float(reaction_bars["low"].min()))

        signal = 0
        entry = 0.0
        stop = 0.0
        target = 0.0

        if current_close >= range_high + breakout_buffer or current_high >= range_high + breakout_buffer:
            signal = 1
            entry = float(tick.ask)
            stop_anchor = min(range_low, current_low)
            stop = stop_anchor - max(breakout_buffer, current_atr * 0.75)
            target = entry + abs(entry - stop) * self.rr
        elif current_close <= range_low - breakout_buffer or current_low <= range_low - breakout_buffer:
            signal = -1
            entry = float(tick.bid)
            stop_anchor = max(range_high, current_high)
            stop = stop_anchor + max(breakout_buffer, current_atr * 0.75)
            target = entry - abs(entry - stop) * self.rr
        else:
            return None

        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return None

        risk_amount = equity * (self.risk_pct / 100.0)
        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)

        if tick_value > 0 and tick_size > 0:
            value_per_price_unit = tick_value / tick_size
            raw_volume = risk_amount / (stop_distance * value_per_price_unit)
        elif contract_size > 0:
            raw_volume = risk_amount / (stop_distance * contract_size)
        else:
            return None

        volume = broker.normalize_volume(symbol, raw_volume)
        min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
        volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)

        while volume >= min_volume:
            margin = broker.order_calc_margin(signal, symbol, volume, entry)
            if margin is not None and margin <= equity * 0.75:
                return NewsTradePlan(
                    symbol=symbol,
                    signal=signal,
                    entry=entry,
                    stop=broker.normalize_price(symbol, stop),
                    target=broker.normalize_price(symbol, target),
                    size=volume,
                    event=event,
                    reason="news_breakout",
                )
            volume = broker.normalize_volume(symbol, volume - volume_step)

        return None

    def maybe_trade(self, symbol, broker, events, now, equity):
        if not self.enabled:
            return None

        symbol = symbol.upper()
        if symbol not in self.symbols:
            return None

        relevant = self._relevant_events(symbol, events, now)
        if not relevant:
            return None

        if broker.positions_get(symbol=symbol):
            return None

        for event in sorted(relevant, key=lambda item: item.event_time):
            key = self._event_key(event, symbol)
            if key in self.fired_events:
                continue

            plan = self._build_plan(symbol, broker, event, now, equity)
            if plan is None:
                continue

            self.fired_events.add(key)
            return plan

        return None
