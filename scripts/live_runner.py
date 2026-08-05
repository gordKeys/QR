from bootstrap import add_project_root
add_project_root()

import argparse
from datetime import datetime, timezone
import pandas as pd
import json
from pathlib import Path
from collections import Counter
from datetime import timedelta

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.risk_manager import RiskManager
from ftmo_rules import FtmoRules, FtmoRiskGuard
from strategy_router import StrategyRouter
from revenge_mode import RevengeTradeManager
from live_protection import HardDrawdownGuard, SymbolCircuitBreaker
from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError
from timing_utils import timed


def build_data_for_symbol(symbol, broker=None):
    if broker is None:
        return FeatureEngine().add_features(DataLoader(symbol=symbol).load())

    rates = broker.rates_copy(symbol, broker.mt5.TIMEFRAME_M5, 2000)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "tick_volume"})
    df = df.set_index("time")
    df = df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    return FeatureEngine().add_features(df)


def latest_signal(symbol, data, router):
    strategy = router.get_strategy(symbol)
    signal_series = strategy.generate_signals(data)
    return int(signal_series.iloc[-1]), strategy


def ensure_log_dir():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def date_log_paths(log_dir, day):
    return (
        log_dir / f"live_run_{day.isoformat()}.jsonl",
        log_dir / f"daily_summary_{day.isoformat()}.json",
    )


def format_status(symbol, consecutive_losses, cooldown_until, last_closed_pnl):
    cooldown_text = "off"
    if cooldown_until is not None:
        remaining = cooldown_until - datetime.now(timezone.utc)
        if remaining.total_seconds() > 0:
            cooldown_text = f"{remaining}"
        else:
            cooldown_text = "expired"

    pnl_text = "n/a" if last_closed_pnl is None else f"{last_closed_pnl:.2f}"
    return (
        f"STATUS | symbol={symbol} | "
        f"consecutive_losses={consecutive_losses} | "
        f"cooldown_remaining={cooldown_text} | "
        f"last_closed_pnl={pnl_text}"
    )


def calculate_trade_volume(
    broker,
    symbol,
    direction,
    entry_price,
    stop_price,
    account_equity,
    risk_per_trade,
    size_multiplier=1.0,
):
    risk_amount = account_equity * risk_per_trade * size_multiplier
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0, "invalid_stop"

    info = broker.symbol_info(symbol)
    if info is None:
        return 0.0, "no_symbol_info"

    tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
    contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
    ask_price = entry_price

    if tick_value > 0 and tick_size > 0:
        value_per_price_unit = tick_value / tick_size
        raw_volume = risk_amount / (stop_distance * value_per_price_unit)
    elif contract_size > 0:
        raw_volume = risk_amount / (stop_distance * contract_size)
    else:
        return 0.0, "no_symbol_pricing"

    volume = broker.normalize_volume(symbol, raw_volume)
    min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
    volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)

    while volume >= min_volume:
        margin = broker.order_calc_margin(direction, symbol, volume, ask_price)
        if margin is not None and margin <= account_equity * 0.85:
            return volume, "ok"
        volume = broker.normalize_volume(symbol, volume - volume_step)

    return 0.0, "insufficient_margin"


def deal_is_stop_loss(deal, broker):
    reason = getattr(deal, "reason", None)
    entry = getattr(deal, "entry", None)
    deal_profit = float(getattr(deal, "profit", 0.0) or 0.0)

    stop_loss_reason = getattr(broker.mt5, "DEAL_REASON_SL", None)
    close_entry = getattr(broker.mt5, "DEAL_ENTRY_OUT", None)

    if stop_loss_reason is not None and reason == stop_loss_reason:
        return True

    if close_entry is not None and entry == close_entry and deal_profit < 0:
        return True

    return deal_profit < 0 and reason is None


def position_side(position, broker):
    buy_type = getattr(broker.mt5, "POSITION_TYPE_BUY", 0)
    sell_type = getattr(broker.mt5, "POSITION_TYPE_SELL", 1)
    pos_type = getattr(position, "type", None)
    if pos_type == buy_type:
        return 1
    if pos_type == sell_type:
        return -1
    return 0


def estimate_break_even_stop(position, broker, commission_round_turn_per_lot):
    info = broker.symbol_info(position.symbol)
    tick = broker.symbol_tick(position.symbol)
    if info is None or tick is None:
        return None

    point = float(getattr(info, "point", 0.0) or 0.0)
    digits = int(getattr(info, "digits", 2) or 2)
    contract_size = float(getattr(info, "trade_contract_size", 100000.0) or 100000.0)
    spread = 0.0
    if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
        spread = float(tick.ask - tick.bid)

    commission_price = float(commission_round_turn_per_lot) / max(contract_size, 1.0)
    safety_buffer = max(point * 2, spread * 0.1)
    buffer = spread + commission_price + safety_buffer
    direction = position_side(position, broker)
    if direction == 1:
        stop = float(getattr(position, "price_open", 0.0) or 0.0) + buffer
    elif direction == -1:
        stop = float(getattr(position, "price_open", 0.0) or 0.0) - buffer
    else:
        return None

    return round(stop, digits)


def ensure_trade_state(trade_states, position):
    state = trade_states.get(position.symbol)
    if state is None or state.get("ticket") != getattr(position, "ticket", None):
        state = {
            "ticket": getattr(position, "ticket", None),
            "mfe_usd": float(getattr(position, "profit", 0.0) or 0.0),
            "break_even_moved": False,
            "break_even_sl": None,
        }
        trade_states[position.symbol] = state
    return state


def update_trade_mfe(trade_states, position, new_profit):
    state = ensure_trade_state(trade_states, position)
    previous = float(state.get("mfe_usd", 0.0) or 0.0)
    current = float(new_profit or 0.0)
    if current > previous:
        state["mfe_usd"] = current
        return True, state
    return False, state


def higher_timeframe_direction(data, timeframe="1h", ema_span=20):
    if data is None or data.empty or len(data) < ema_span + 5:
        return None

    higher = data[["open", "high", "low", "close", "tick_volume"]].resample(timeframe).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
    ).dropna()
    if len(higher) < ema_span + 3:
        return None

    ema = higher["close"].ewm(span=ema_span, adjust=False).mean()
    if higher["close"].iloc[-1] > ema.iloc[-1]:
        return "UP"
    if higher["close"].iloc[-1] < ema.iloc[-1]:
        return "DOWN"
    return None


def spread_filter(symbol, broker, price, atr, max_spread_points=40, max_spread_atr_ratio=0.12):
    info = broker.symbol_info(symbol)
    tick = broker.symbol_tick(symbol)
    if info is None or tick is None:
        return False, None, None, None

    point = float(getattr(info, "point", 0.0) or 0.0)
    symbol_name = str(getattr(info, "name", symbol) or symbol).upper()
    rule_name = "forex"
    if symbol_name.startswith("XAU"):
        rule_name = "xauusd"
        max_spread_points = 220
        max_spread_atr_ratio = 0.28

    spread = None
    if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
        spread = float(tick.ask - tick.bid)
    if spread is None:
        return False, None, None, rule_name

    spread_points = spread / point if point > 0 else None
    point_threshold = max_spread_points * point if point > 0 else None
    atr_threshold = atr * max_spread_atr_ratio if atr and atr > 0 else None
    allowed = None
    for threshold in (point_threshold, atr_threshold):
        if threshold is not None and threshold > 0:
            allowed = threshold if allowed is None else max(allowed, threshold)

    if allowed is None:
        return False, spread, None, rule_name

    return spread > allowed, spread, allowed, rule_name


def structure_stop_from_data(data, symbol, broker, direction, price, atr, lookback=12, buffer_atr=0.35):
    if broker is None:
        return None

    info = broker.symbol_info(symbol)
    if info is None or data is None or data.empty:
        return None

    point = float(getattr(info, "point", 0.0) or 0.0)
    digits = int(getattr(info, "digits", 2) or 2)
    min_stop_distance = float(getattr(info, "trade_stops_level", 0.0) or 0.0) * point
    if len(data) < lookback + 2:
        return None

    window = data.iloc[-(lookback + 1):-1]
    atr_buffer = max((atr or 0.0) * buffer_atr, min_stop_distance, point * 2)

    if direction == 1:
        swing_low = float(window["low"].min())
        candidate = min(float(price) - max(min_stop_distance, atr_buffer), swing_low - atr_buffer)
        return round(candidate, digits)

    if direction == -1:
        swing_high = float(window["high"].max())
        candidate = max(float(price) + max(min_stop_distance, atr_buffer), swing_high + atr_buffer)
        return round(candidate, digits)

    return None


def partial_close_volume(position, broker):
    info = broker.symbol_info(position.symbol)
    if info is None:
        return 0.0

    min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
    total_volume = float(getattr(position, "volume", 0.0) or 0.0)
    if total_volume <= 0:
        return 0.0

    close_volume = broker.normalize_volume(position.symbol, total_volume / 2.0)
    if close_volume < min_volume:
        return 0.0
    if total_volume - close_volume < min_volume:
        return 0.0
    return close_volume


def manage_open_position(symbol, data, broker, position, state, args, started, run_log):
    info = broker.symbol_info(symbol)
    tick = broker.symbol_tick(symbol)
    if info is None or tick is None:
        return state

    direction = position_side(position, broker)
    current_price = float(tick.bid if direction == 1 else tick.ask)
    current_profit = float(getattr(position, "profit", 0.0) or 0.0)
    updated, state = update_trade_mfe({symbol: state}, position, current_profit)
    if updated:
        append_jsonl(
            run_log,
            {
                "event": "position_mfe_update",
                "symbol": symbol,
                "ticket": getattr(position, "ticket", None),
                "mfe_usd": state["mfe_usd"],
                "current_profit": current_profit,
                "time": started,
            },
        )

    contract_size = float(state.get("contract_size") or getattr(info, "trade_contract_size", 100000.0) or 100000.0)
    entry_price = float(state.get("entry_price") or getattr(position, "price_open", 0.0) or 0.0)
    initial_stop = float(state.get("initial_stop") or getattr(position, "sl", 0.0) or 0.0)
    if state.get("risk_usd") is None and entry_price and initial_stop and contract_size > 0:
        state["risk_usd"] = abs(entry_price - initial_stop) * float(getattr(position, "volume", 0.0) or 0.0) * contract_size

    current_stop = float(getattr(position, "sl", 0.0) or 0.0)
    structure_stop = structure_stop_from_data(data, symbol, broker, direction, current_price, float(data["atr"].iloc[-1]), lookback=12)

    if not state.get("break_even_moved") and current_profit >= args.break_even_trigger_usd:
        break_even_sl = estimate_break_even_stop(
            position,
            broker,
            args.break_even_commission_round_turn,
        )
        if break_even_sl is not None:
            candidate_stop = break_even_sl
            if structure_stop is not None:
                candidate_stop = max(candidate_stop, structure_stop) if direction == 1 else min(candidate_stop, structure_stop)
            if direction == 1 and candidate_stop < current_price:
                result = broker.modify_position_stops(
                    position_ticket=getattr(position, "ticket", None),
                    symbol=symbol,
                    stop_loss=candidate_stop,
                    take_profit=getattr(position, "tp", None),
                )
                accepted = result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                if accepted:
                    state["break_even_moved"] = True
                    state["break_even_sl"] = candidate_stop
                    current_stop = candidate_stop
                    append_jsonl(
                        run_log,
                        {
                            "event": "break_even_moved",
                            "symbol": symbol,
                            "ticket": getattr(position, "ticket", None),
                            "profit_usd": current_profit,
                            "new_stop": candidate_stop,
                            "mfe_usd": state["mfe_usd"],
                            "time": started,
                        },
                    )
            elif direction == -1 and candidate_stop > current_price:
                result = broker.modify_position_stops(
                    position_ticket=getattr(position, "ticket", None),
                    symbol=symbol,
                    stop_loss=candidate_stop,
                    take_profit=getattr(position, "tp", None),
                )
                accepted = result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                if accepted:
                    state["break_even_moved"] = True
                    state["break_even_sl"] = candidate_stop
                    current_stop = candidate_stop
                    append_jsonl(
                        run_log,
                        {
                            "event": "break_even_moved",
                            "symbol": symbol,
                            "ticket": getattr(position, "ticket", None),
                            "profit_usd": current_profit,
                            "new_stop": candidate_stop,
                            "mfe_usd": state["mfe_usd"],
                            "time": started,
                        },
                    )

    risk_usd = float(state.get("risk_usd") or 0.0)
    if not state.get("partial_taken") and risk_usd > 0 and current_profit >= risk_usd:
        close_volume = partial_close_volume(position, broker)
        if close_volume > 0:
            close_result = broker.close_position_partial(
                position_ticket=getattr(position, "ticket", None),
                symbol=symbol,
                direction=direction,
                volume=close_volume,
            )
            accepted = close_result is not None and getattr(close_result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
            if accepted:
                state["partial_taken"] = True
                state["partial_closed_volume"] = close_volume
                append_jsonl(
                    run_log,
                    {
                        "event": "partial_profit_taken",
                        "symbol": symbol,
                        "ticket": getattr(position, "ticket", None),
                        "close_volume": close_volume,
                        "profit_usd": current_profit,
                        "risk_usd": risk_usd,
                        "time": started,
                    },
                )

    trail_allowed = state.get("partial_taken") or state.get("break_even_moved")
    if trail_allowed and structure_stop is not None:
        if direction == 1:
            candidate_stop = structure_stop if current_stop <= 0 else max(current_stop, structure_stop)
            if candidate_stop > 0 and candidate_stop < current_price and candidate_stop > current_stop:
                result = broker.modify_position_stops(
                    position_ticket=getattr(position, "ticket", None),
                    symbol=symbol,
                    stop_loss=candidate_stop,
                    take_profit=getattr(position, "tp", None),
                )
                accepted = result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                if accepted:
                    state["trail_sl"] = candidate_stop
                    append_jsonl(
                        run_log,
                        {
                            "event": "trail_stop_moved",
                            "symbol": symbol,
                            "ticket": getattr(position, "ticket", None),
                            "new_stop": candidate_stop,
                            "current_profit": current_profit,
                            "time": started,
                        },
                    )
        elif direction == -1:
            candidate_stop = structure_stop if current_stop <= 0 else min(current_stop, structure_stop)
            if candidate_stop > current_price and (current_stop <= 0 or candidate_stop < current_stop):
                result = broker.modify_position_stops(
                    position_ticket=getattr(position, "ticket", None),
                    symbol=symbol,
                    stop_loss=candidate_stop,
                    take_profit=getattr(position, "tp", None),
                )
                accepted = result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                if accepted:
                    state["trail_sl"] = candidate_stop
                    append_jsonl(
                        run_log,
                        {
                            "event": "trail_stop_moved",
                            "symbol": symbol,
                            "ticket": getattr(position, "ticket", None),
                            "new_stop": candidate_stop,
                            "current_profit": current_profit,
                            "time": started,
                        },
                    )

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument("--cooldown-hours", type=int, default=3)
    parser.add_argument("--revenge-mode", action="store_true")
    parser.add_argument("--revenge-multiplier", type=float, default=2.0)
    parser.add_argument("--revenge-gap-trades", type=int, default=2)
    parser.add_argument("--revenge-boosts", type=int, default=3)
    parser.add_argument("--hard-drawdown-switch", action="store_true")
    parser.add_argument("--hard-drawdown-usd", type=float, default=3000.0)
    parser.add_argument("--break-even-trigger-usd", type=float, default=190.0)
    parser.add_argument("--break-even-commission-round-turn", type=float, default=7.0)
    args = parser.parse_args()

    router = StrategyRouter()
    rules = FtmoRules(initial_balance=10000, max_consecutive_losses=args.max_consecutive_losses)
    guard = FtmoRiskGuard(rules)
    risk = RiskManager(risk_per_trade=rules.max_risk_per_trade_pct)
    revenge = RevengeTradeManager(
        enabled=args.revenge_mode,
        boost_multiplier=args.revenge_multiplier,
        boosts_total=args.revenge_boosts,
        normal_gap=args.revenge_gap_trades,
    )
    hard_drawdown = HardDrawdownGuard(enabled=args.hard_drawdown_switch, drawdown_usd=args.hard_drawdown_usd)
    symbol_breaker = SymbolCircuitBreaker(max_stop_losses=2)
    trade_states = {}
    log_dir = ensure_log_dir()
    cooldown_until = None
    last_deal_check = None
    last_closed_pnl = None
    active_positions = 0
    hard_drawdown_day = None

    print(f"Revenge mode: {'ON' if args.revenge_mode else 'OFF'}")
    if args.revenge_mode:
        print(
            "Revenge settings: "
            f"multiplier={args.revenge_multiplier}, "
            f"boosts={args.revenge_boosts}, "
            f"gap_trades={args.revenge_gap_trades}"
        )
    print(f"Hard drawdown switch: {'ON' if args.hard_drawdown_switch else 'OFF'} | threshold=${args.hard_drawdown_usd:.2f}")
    print(f"Break-even trigger: ${args.break_even_trigger_usd:.2f}")

    broker = None
    if not args.dry_run:
        try:
            broker = MT5BrokerAdapter()
            broker.initialize()
            last_deal_check = datetime.now(timezone.utc) - timedelta(minutes=5)
            live_equity = broker.account_equity()
            if live_equity is not None:
                hard_drawdown.reset_session(live_equity)
        except MT5UnavailableError as exc:
            print(f"MT5 unavailable, falling back to dry-run: {exc}")
            args.dry_run = True

    while True:
        started = datetime.now(timezone.utc)
        cycle_counts = Counter()
        current_day = started.date()
        run_log, summary_file = date_log_paths(log_dir, current_day)
        print(f"\n=== LIVE CYCLE {started.isoformat()} ===")
        append_jsonl(run_log, {"event": "cycle_start", "time": started})

        if cooldown_until and started < cooldown_until:
            remaining = cooldown_until - started
            print(f"Cooldown active for {remaining}")
            append_jsonl(
                run_log,
                {
                    "event": "cooldown_active",
                    "time": started,
                    "cooldown_until": cooldown_until,
                    "remaining_seconds": remaining.total_seconds(),
                },
            )
            if args.loop_once:
                break
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            sleep_for = max(1, args.poll_seconds - int(elapsed))
            print(f"Sleeping {sleep_for}s")
            import time
            time.sleep(sleep_for)
            continue

        if cooldown_until and started >= cooldown_until:
            cooldown_until = None
            guard.consecutive_losses = 0
            append_jsonl(run_log, {"event": "cooldown_lifted", "time": started})

        if broker and not args.dry_run:
            current_equity = broker.account_equity()
            if current_equity is not None:
                if hard_drawdown_day != current_day:
                    hard_drawdown.reset_day(current_equity)
                    hard_drawdown_day = current_day
                hard_drawdown.check(current_equity)
                append_jsonl(
                    run_log,
                    {
                        "event": "equity_snapshot",
                        "time": started,
                        "equity": current_equity,
                        "hard_drawdown_triggered": hard_drawdown.triggered,
                        "hard_drawdown_reason": hard_drawdown.reason,
                    },
                )

        if broker and not args.dry_run and last_deal_check is not None:
            closed_deals = broker.history_deals_since(last_deal_check, magic=26072026)
            last_deal_check = started
            if closed_deals:
                for deal in sorted(closed_deals, key=lambda item: getattr(item, "time", started)):
                    profit = float(getattr(deal, "profit", 0.0) or 0.0)
                    closed_time = getattr(deal, "time", started)
                    symbol = getattr(deal, "symbol", "")
                    stop_loss_hit = deal_is_stop_loss(deal, broker)
                    revenge_active = revenge.is_active(symbol)
                    open_positions_after = broker.positions_get(symbol=symbol)
                    is_final_close = not open_positions_after

                    trade_state = trade_states.get(symbol, {})
                    append_jsonl(
                        run_log,
                        {
                            "event": "closed_deal",
                            "symbol": symbol,
                            "profit": profit,
                            "time": closed_time,
                            "consecutive_losses": guard.consecutive_losses,
                            "mfe_usd": trade_state.get("mfe_usd"),
                            "break_even_moved": trade_state.get("break_even_moved", False),
                            "break_even_sl": trade_state.get("break_even_sl"),
                            "is_final_close": is_final_close,
                        },
                    )

                    if not is_final_close:
                        continue

                    last_closed_pnl = profit
                    guard.register_closed_trade(profit)
                    trade_state = trade_states.pop(symbol, {})

                    if args.revenge_mode and stop_loss_hit:
                        breaker_state = symbol_breaker.register_close(symbol, stop_loss_hit, revenge_failed=revenge_active)
                        if breaker_state["paused"]:
                            revenge.cancel_plan(symbol)
                            print(f"{symbol}: circuit breaker tripped ({breaker_state['reason']})")
                            append_jsonl(
                                run_log,
                                {
                                    "event": "symbol_paused",
                                    "symbol": symbol,
                                    "reason": breaker_state["reason"],
                                    "time": closed_time,
                                },
                            )
                        elif not revenge_active:
                            plan = revenge.register_stop_loss(symbol, closed_time)
                            if plan is not None:
                                queue_depth = len(revenge.pending) + (1 if revenge.active_symbol else 0)
                                print(f"{symbol}: revenge queued (queue_depth={queue_depth})")
                                append_jsonl(
                                    run_log,
                                    {
                                        "event": "revenge_queued",
                                        "symbol": symbol,
                                        "time": closed_time,
                                        "boost_multiplier": plan.boost_multiplier,
                                        "boosts_total": plan.boosts_total,
                                        "normal_gap": plan.normal_gap,
                                    },
                                )
                    else:
                        symbol_breaker.register_close(symbol, stop_loss_hit, revenge_failed=False)

                    if hard_drawdown.triggered:
                        print(f"Hard drawdown guard active: {hard_drawdown.reason}")
                        append_jsonl(
                            run_log,
                            {
                                "event": "hard_drawdown_triggered",
                                "reason": hard_drawdown.reason,
                                "time": closed_time,
                            },
                        )

                if guard.consecutive_losses >= rules.max_consecutive_losses:
                    cooldown_until = started + timedelta(hours=args.cooldown_hours)
                    print(f"3 consecutive losses reached; pausing until {cooldown_until.isoformat()}")
                    append_jsonl(
                        run_log,
                        {
                            "event": "cooldown_started",
                            "time": started,
                            "cooldown_until": cooldown_until,
                            "consecutive_losses": guard.consecutive_losses,
                        },
                    )
                    if args.loop_once:
                        break
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    sleep_for = max(1, args.poll_seconds - int(elapsed))
                    print(f"Sleeping {sleep_for}s")
                    import time
                    time.sleep(sleep_for)
                    continue

        for symbol in args.symbols:
            print(format_status(symbol, guard.consecutive_losses, cooldown_until, last_closed_pnl))
            with timed(f"{symbol} evaluation"):
                data = build_data_for_symbol(symbol, broker=broker if not args.dry_run else None)
                if data is None or data.empty:
                    print(f"{symbol}: no data available")
                    append_jsonl(run_log, {"event": "no_data", "symbol": symbol, "time": datetime.now(timezone.utc)})
                    cycle_counts["no_data"] += 1
                    continue

                if broker and not args.dry_run:
                    positions = broker.positions_get(symbol=symbol)
                    if positions:
                        current_position = positions[0]
                        state = ensure_trade_state(trade_states, current_position)
                        state = manage_open_position(symbol, data, broker, current_position, state, args, started, run_log)
                        trade_states[symbol] = state
                        cycle_counts["managed_position"] += 1
                        continue

                if hard_drawdown.triggered:
                    print(f"{symbol}: skipped because hard drawdown switch is active")
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip",
                            "symbol": symbol,
                            "reason": "hard_drawdown",
                            "broker_time": data.index[-1].to_pydatetime(),
                        },
                    )
                    cycle_counts["hard_drawdown_skip"] += 1
                    continue

                if symbol_breaker.is_paused(symbol):
                    breaker_reason = symbol_breaker.state[symbol]["reason"]
                    print(f"{symbol}: paused by circuit breaker ({breaker_reason})")
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip",
                            "symbol": symbol,
                            "reason": breaker_reason,
                            "broker_time": data.index[-1].to_pydatetime(),
                        },
                    )
                    cycle_counts["symbol_paused"] += 1
                    continue

                revenge_context = revenge.get_trade_context(symbol)
                if revenge_context["blocked"]:
                    print(f"{symbol}: revenge mode queued, waiting for its turn")
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip",
                            "symbol": symbol,
                            "reason": "revenge_waiting",
                            "broker_time": data.index[-1].to_pydatetime(),
                        },
                    )
                    cycle_counts["revenge_waiting"] += 1
                    continue

                signal, strategy = latest_signal(symbol, data, router)
                atr = float(data["atr"].iloc[-1])
                broker_time = data.index[-1].to_pydatetime()

                if broker and not args.dry_run:
                    spread_blocked, spread_value, spread_limit, spread_rule = spread_filter(symbol, broker, None, atr)
                    if spread_blocked:
                        print(
                            f"{symbol}: skipped because spread is too wide "
                            f"(spread={spread_value:.5f}, limit={spread_limit:.5f}, rule={spread_rule})"
                        )
                        append_jsonl(
                            run_log,
                            {
                                "event": "skip",
                                "symbol": symbol,
                                "reason": "spread_too_wide",
                                "spread": spread_value,
                                "spread_limit": spread_limit,
                                "spread_rule": spread_rule,
                                "broker_time": broker_time,
                            },
                        )
                        cycle_counts["skip_spread"] += 1
                        continue

                htf_bias = higher_timeframe_direction(data)
                if htf_bias is not None:
                    if signal == 1 and htf_bias == "DOWN":
                        print(f"{symbol}: skipped because higher timeframe is bearish")
                        append_jsonl(
                            run_log,
                            {
                                "event": "skip",
                                "symbol": symbol,
                                "reason": "htf_bias_down",
                                "signal": signal,
                                "htf_bias": htf_bias,
                                "broker_time": broker_time,
                            },
                        )
                        cycle_counts["skip_htf_bias"] += 1
                        continue
                    if signal == -1 and htf_bias == "UP":
                        print(f"{symbol}: skipped because higher timeframe is bullish")
                        append_jsonl(
                            run_log,
                            {
                                "event": "skip",
                                "symbol": symbol,
                                "reason": "htf_bias_up",
                                "signal": signal,
                                "htf_bias": htf_bias,
                                "broker_time": broker_time,
                            },
                        )
                        cycle_counts["skip_htf_bias"] += 1
                        continue

                if broker and not args.dry_run:
                    tick = broker.symbol_tick(symbol)
                    if tick is None:
                        print(f"{symbol}: no live tick available")
                        append_jsonl(run_log, {"event": "skip", "symbol": symbol, "reason": "no_tick", "broker_time": broker_time})
                        cycle_counts["skip_no_tick"] += 1
                        continue
                    price = float(tick.ask if signal == 1 else tick.bid)
                    equity = broker.account_equity() or rules.initial_balance
                else:
                    price = float(data["close"].iloc[-1])
                    equity = rules.initial_balance

                if signal == 0:
                    print(f"{symbol}: no trade (no_signal)")
                    cycle_counts["skip_no_signal"] += 1
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip",
                            "symbol": symbol,
                            "reason": "no_signal",
                            "signal": signal,
                            "equity": equity,
                            "broker_time": broker_time,
                        },
                    )
                    continue

                strategy_plan = None
                analyze_trade = getattr(strategy, "analyze_trade", None)
                if callable(analyze_trade):
                    try:
                        strategy_plan = analyze_trade(
                            data,
                            broker=broker if not args.dry_run else None,
                            symbol=symbol,
                            equity=equity,
                            signal=signal,
                            price=price,
                        )
                    except TypeError:
                        strategy_plan = analyze_trade(data)

                if strategy_plan:
                    signal = int(strategy_plan.get("signal", signal))
                    price = float(strategy_plan.get("price", price))
                    stop = float(strategy_plan.get("stop"))
                    target = float(strategy_plan.get("target"))
                    size_reason = strategy_plan.get("size_reason", "strategy_plan")
                else:
                    stop, target = risk.calculate_sl_tp(signal, price, atr)

                structure_stop = structure_stop_from_data(data, symbol, broker if broker is not None else broker, signal, price, atr)
                if structure_stop is not None:
                    if signal == 1:
                        stop = min(stop, structure_stop)
                    else:
                        stop = max(stop, structure_stop)

                if broker and not args.dry_run:
                    stop, target = broker.conform_stop_levels(symbol, signal, price, stop, target)
                    if stop is None or target is None:
                        print(f"{symbol}: skipped because broker stop levels are invalid")
                        cycle_counts["skip_invalid_stops"] += 1
                        append_jsonl(
                            run_log,
                            {
                                "event": "skip_invalid_stops",
                                "symbol": symbol,
                                "signal": signal,
                                "price": price,
                                "broker_time": broker_time,
                            },
                        )
                        continue

                    size, size_reason = calculate_trade_volume(
                        broker=broker,
                        symbol=symbol,
                        direction=signal,
                        entry_price=price,
                        stop_price=stop,
                        account_equity=equity,
                        risk_per_trade=rules.max_risk_per_trade_pct,
                        size_multiplier=revenge_context["multiplier"],
                    )
                else:
                    size = risk.calculate_position_size(equity, price, stop, atr=atr) * revenge_context["multiplier"]
                    size_reason = "dry_run"

                if size <= 0:
                    print(f"{symbol}: skipped due to sizing ({size_reason})")
                    cycle_counts["skip_zero_size"] += 1
                    append_jsonl(run_log, {"event": "skip_zero_size", "symbol": symbol, "reason": size_reason, "broker_time": broker_time})
                    continue

                print(
                    f"{symbol}: strategy={strategy.__class__.__name__} signal={signal} "
                    f"price={price:.5f} size={size:.2f} sl={stop:.5f} tp={target:.5f}"
                )
                append_jsonl(
                    run_log,
                    {
                        "event": "signal",
                        "symbol": symbol,
                        "strategy": strategy.__class__.__name__,
                        "signal": signal,
                        "price": price,
                        "size": size,
                        "stop": stop,
                        "target": target,
                        "equity": equity,
                        "size_reason": size_reason,
                        "revenge_mode": revenge_context["enabled"],
                        "revenge_stage": revenge_context["stage"],
                        "revenge_boosted": revenge_context["boosted"],
                        "broker_time": broker_time,
                    },
                )
                cycle_counts["signals"] += 1

                if broker and not args.dry_run:
                    active_positions = broker.positions_total(symbol)
                    if active_positions >= 1:
                        print(f"{symbol}: skipped because position already open")
                        cycle_counts["skip_open_position"] += 1
                        continue
                    result = broker.place_order(
                        symbol=symbol,
                        direction=signal,
                        volume=size,
                        stop_loss=stop,
                        take_profit=target,
                        price=price,
                    )
                    print(f"{symbol}: order result={result}")
                    accepted = getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                    cycle_counts["orders_sent"] += int(bool(accepted))
                    if accepted:
                        positions = broker.positions_get(symbol=symbol)
                        if positions:
                            current_position = positions[0]
                            contract_size = float(getattr(broker.symbol_info(symbol), "trade_contract_size", 100000.0) or 100000.0)
                            trade_states[symbol] = {
                                "ticket": getattr(current_position, "ticket", None),
                                "entry_price": float(getattr(current_position, "price_open", price) or price),
                                "initial_stop": float(getattr(current_position, "sl", stop) or stop),
                                "initial_target": float(getattr(current_position, "tp", target) or target),
                                "volume": float(getattr(current_position, "volume", size) or size),
                                "contract_size": contract_size,
                                "risk_usd": abs(float(price) - float(stop)) * float(getattr(current_position, "volume", size) or size) * contract_size,
                                "mfe_usd": float(getattr(current_position, "profit", 0.0) or 0.0),
                                "break_even_moved": False,
                                "break_even_sl": None,
                                "partial_taken": False,
                                "partial_closed_volume": 0.0,
                                "trail_sl": None,
                            }
                        revenge_update = revenge.on_trade_filled(symbol)
                        if revenge_update is not None:
                            append_jsonl(
                                run_log,
                                {
                                    "event": "revenge_trade_filled",
                                    "symbol": symbol,
                                    "stage": revenge_update["stage"],
                                    "boosted": revenge_update["boosted"],
                                    "completed": revenge_update["completed"],
                                    "broker_time": broker_time,
                                },
                            )
                    try:
                        retcode = getattr(result, "retcode", None)
                        if retcode is not None and retcode != broker.mt5.TRADE_RETCODE_DONE:
                            append_jsonl(
                                run_log,
                                {
                                    "event": "order_rejected",
                                    "symbol": symbol,
                                    "retcode": retcode,
                                    "comment": getattr(result, "comment", ""),
                                    "broker_time": broker_time,
                                },
                            )
                        else:
                            append_jsonl(
                                run_log,
                                {
                                    "event": "order_accepted",
                                    "symbol": symbol,
                                    "result": str(result),
                                    "broker_time": broker_time,
                                },
                            )
                    except Exception:
                        pass

        append_jsonl(
            run_log,
            {
                "event": "cycle_summary",
                "time": datetime.now(timezone.utc),
                "summary": dict(cycle_counts),
            },
        )
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "day": current_day.isoformat(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "summary": dict(cycle_counts),
                },
                handle,
                indent=2,
            )

        if args.loop_once:
            break

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        sleep_for = max(1, args.poll_seconds - int(elapsed))
        print(f"Sleeping {sleep_for}s")

        import time
        time.sleep(sleep_for)

    if broker and not args.dry_run:
        broker.shutdown()

    append_jsonl(run_log, {"event": "runner_stop", "time": datetime.now(timezone.utc)})


if __name__ == "__main__":
    main()
