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
from news_calendar import NewsBlackoutGuard
from news_trading import NewsSniperModule
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


def spread_policy(symbol: str):
    symbol = symbol.upper()
    if symbol in {"EURUSD", "GBPUSD"}:
        return {"tier": "core", "max_points": 80.0, "atr_ratio": 0.20}
    if symbol == "USDJPY":
        return {"tier": "moderate", "max_points": 100.0, "atr_ratio": 0.18}
    if symbol == "XAUUSD":
        return {"tier": "aggressive", "max_points": 120.0, "atr_ratio": 0.15}
    return {"tier": "balanced", "max_points": 90.0, "atr_ratio": 0.18}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"])
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
    parser.add_argument("--break-even-trigger-usd", type=float, default=0.0)
    parser.add_argument("--break-even-commission-round-turn", type=float, default=7.0)
    parser.add_argument("--news-filter", action="store_true", default=True)
    parser.add_argument("--no-news-filter", action="store_false", dest="news_filter")
    parser.add_argument("--news-before-minutes", type=int, default=30)
    parser.add_argument("--news-after-minutes", type=int, default=15)
    parser.add_argument("--news-impact-level", choices=["high", "medium"], default="high")
    parser.add_argument("--news-cache-minutes", type=int, default=1)
    parser.add_argument(
        "--news-calendar-url",
        type=str,
        default="https://www.forexfactory.com/calendar?day=today",
    )
    parser.add_argument("--news-sniper", action="store_true", default=True)
    parser.add_argument("--no-news-sniper", action="store_false", dest="news_sniper")
    parser.add_argument("--news-sniper-symbols", nargs="+")
    parser.add_argument("--news-sniper-magic", type=int, default=26072027)
    parser.add_argument("--news-sniper-risk-pct", type=float, default=0.25)
    parser.add_argument("--news-sniper-rr", type=float, default=1.4)
    parser.add_argument("--news-sniper-post-delay-minutes", type=int, default=1)
    parser.add_argument("--news-sniper-post-window-minutes", type=int, default=10)
    parser.add_argument("--news-sniper-lookback-bars", type=int, default=12)
    parser.add_argument("--news-sniper-buffer-points", type=float, default=12.0)
    parser.add_argument("--news-sniper-spread-multiplier", type=float, default=1.25)
    parser.add_argument("--news-sniper-min-atr-points", type=float, default=8.0)
    parser.add_argument("--use-fast-price-action", action="store_true")
    args = parser.parse_args()

    router = StrategyRouter()
    if args.use_fast_price_action:
        router.update_mapping("XAUUSD", "mtf_pa_breakout")
        router.update_mapping("GBPUSD", "pattern_playbook_double_triple")
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
    news_guard = NewsBlackoutGuard(
        enabled=args.news_filter or args.news_sniper,
        before_minutes=args.news_before_minutes,
        after_minutes=args.news_after_minutes,
        impact_level=args.news_impact_level,
        calendar_url=args.news_calendar_url,
        cache_minutes=args.news_cache_minutes,
    )
    news_sniper = NewsSniperModule(
        enabled=args.news_sniper,
        symbols=args.news_sniper_symbols or ["XAUUSD", "EURUSD", "GBPUSD"],
        magic=args.news_sniper_magic,
        risk_pct=args.news_sniper_risk_pct,
        rr=args.news_sniper_rr,
        post_delay_minutes=args.news_sniper_post_delay_minutes,
        post_window_minutes=args.news_sniper_post_window_minutes,
        lookback_bars=args.news_sniper_lookback_bars,
        breakout_buffer_points=args.news_sniper_buffer_points,
        spread_multiplier=args.news_sniper_spread_multiplier,
        min_atr_points=args.news_sniper_min_atr_points,
    )
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
    print(
        "News filter: "
        f"{'ON' if args.news_filter else 'OFF'} | "
        f"before={args.news_before_minutes}m | after={args.news_after_minutes}m | "
        f"impact={args.news_impact_level}"
    )
    print(
        "News sniper: "
        f"{'ON' if args.news_sniper else 'OFF'} | "
        f"symbols={', '.join(news_sniper.symbols)} | "
        f"magic={news_sniper.magic}"
    )

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
        news_guard.refresh(started)

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

                    last_closed_pnl = profit
                    guard.register_closed_trade(profit)
                    trade_state = trade_states.pop(symbol, {})
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
                        },
                    )

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
                        current_profit = float(getattr(current_position, "profit", 0.0) or 0.0)
                        updated, state = update_trade_mfe(trade_states, current_position, current_profit)
                        if updated:
                            append_jsonl(
                                run_log,
                                {
                                    "event": "position_mfe_update",
                                    "symbol": symbol,
                                    "ticket": getattr(current_position, "ticket", None),
                                    "mfe_usd": state["mfe_usd"],
                                    "current_profit": current_profit,
                                    "time": started,
                                },
                            )

                        if not state.get("break_even_moved") and current_profit >= args.break_even_trigger_usd:
                            break_even_sl = estimate_break_even_stop(
                                current_position,
                                broker,
                                args.break_even_commission_round_turn,
                            )
                            if break_even_sl is not None:
                                result = broker.modify_position_stops(
                                    position_ticket=getattr(current_position, "ticket", None),
                                    symbol=symbol,
                                    stop_loss=break_even_sl,
                                    take_profit=getattr(current_position, "tp", None),
                                )
                                accepted = result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                                if accepted:
                                    state["break_even_moved"] = True
                                    state["break_even_sl"] = break_even_sl
                                    append_jsonl(
                                        run_log,
                                        {
                                            "event": "break_even_moved",
                                            "symbol": symbol,
                                            "ticket": getattr(current_position, "ticket", None),
                                            "profit_usd": current_profit,
                                            "new_stop": break_even_sl,
                                            "mfe_usd": state["mfe_usd"],
                                            "time": started,
                                        },
                                    )

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
                    tick = broker.symbol_tick(symbol)
                    if tick is None:
                        print(f"{symbol}: no live tick available")
                        append_jsonl(run_log, {"event": "skip", "symbol": symbol, "reason": "no_tick", "broker_time": broker_time})
                        cycle_counts["skip_no_tick"] += 1
                        continue
                    policy = spread_policy(symbol)
                    info = broker.symbol_info(symbol)
                    point = float(getattr(info, "point", 0.0) or 0.0) if info is not None else 0.0
                    live_spread = 0.0
                    if getattr(tick, "ask", None) is not None and getattr(tick, "bid", None) is not None:
                        live_spread = float(tick.ask - tick.bid)
                    spread_limit = float(policy["max_points"]) * point if point > 0 else 0.0
                    atr_limit = float(data["atr"].iloc[-1]) * float(policy["atr_ratio"]) if "atr" in data.columns else 0.0
                    allowed_spread = max(spread_limit, atr_limit)
                    if allowed_spread > 0 and live_spread > allowed_spread:
                        print(
                            f"{symbol}: skipped because spread is too wide "
                            f"(spread={live_spread:.5f}, limit={allowed_spread:.5f}, tier={policy['tier']})"
                        )
                        append_jsonl(
                            run_log,
                            {
                                "event": "skip",
                                "symbol": symbol,
                                "reason": "spread_too_wide",
                                "tier": policy["tier"],
                                "spread": live_spread,
                                "spread_limit": allowed_spread,
                                "broker_time": broker_time,
                            },
                        )
                        cycle_counts["skip_wide_spread"] += 1
                        continue
                    price = float(tick.ask if signal == 1 else tick.bid)
                    equity = broker.account_equity() or rules.initial_balance
                else:
                    price = float(data["close"].iloc[-1])
                    equity = rules.initial_balance

                news_sniper_plan = news_sniper.maybe_trade(
                    symbol=symbol,
                    broker=broker if not args.dry_run else None,
                    events=news_guard.events(),
                    now=broker_time,
                    equity=equity,
                ) if args.news_sniper and broker and not args.dry_run else None
                if news_sniper_plan is not None:
                    result = broker.place_order(
                        symbol=news_sniper_plan.symbol,
                        direction=news_sniper_plan.signal,
                        volume=news_sniper_plan.size,
                        stop_loss=news_sniper_plan.stop,
                        take_profit=news_sniper_plan.target,
                        price=news_sniper_plan.entry,
                        comment="QuantFX-News",
                        magic=news_sniper.magic,
                    )
                    accepted = result is not None and getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                    event_time = news_sniper_plan.event.event_time
                    if accepted:
                        print(
                            f"{symbol}: NEWS trade placed "
                            f"event={news_sniper_plan.event.currency} {news_sniper_plan.event.title} "
                            f"signal={news_sniper_plan.signal} price={news_sniper_plan.entry:.5f} "
                            f"sl={news_sniper_plan.stop:.5f} tp={news_sniper_plan.target:.5f}"
                        )
                        append_jsonl(
                            run_log,
                            {
                                "event": "news_trade_accepted",
                                "symbol": symbol,
                                "magic": news_sniper.magic,
                                "signal": news_sniper_plan.signal,
                                "price": news_sniper_plan.entry,
                                "stop": news_sniper_plan.stop,
                                "target": news_sniper_plan.target,
                                "size": news_sniper_plan.size,
                                "currency": news_sniper_plan.event.currency,
                                "title": news_sniper_plan.event.title,
                                "event_time": event_time,
                                "broker_time": broker_time,
                            },
                        )
                        cycle_counts["news_trade_accepted"] += 1
                        continue
                    print(
                        f"{symbol}: NEWS order rejected "
                        f"event={news_sniper_plan.event.currency} {news_sniper_plan.event.title}"
                    )
                    append_jsonl(
                        run_log,
                        {
                            "event": "news_trade_rejected",
                            "symbol": symbol,
                            "magic": news_sniper.magic,
                            "currency": news_sniper_plan.event.currency,
                            "title": news_sniper_plan.event.title,
                            "result": str(result),
                            "broker_time": broker_time,
                        },
                    )

                news_blocked, blocked_event = news_guard.should_block(symbol, broker_time)
                if news_blocked and blocked_event is not None:
                    print(
                        f"{symbol}: skipped because of news blackout "
                        f"({blocked_event.currency} {blocked_event.title} at {blocked_event.event_time.isoformat()})"
                    )
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip",
                            "symbol": symbol,
                            "reason": "news_blackout",
                            "currency": blocked_event.currency,
                            "title": blocked_event.title,
                            "event_time": blocked_event.event_time,
                            "impact": blocked_event.impact,
                            "broker_time": broker_time,
                        },
                    )
                    cycle_counts["skip_news_blackout"] += 1
                    continue

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
                build_trade_plan = getattr(strategy, "build_trade_plan", None)
                if callable(build_trade_plan):
                    try:
                        strategy_plan = build_trade_plan(
                            data,
                            len(data) - 1,
                            symbol,
                            None,
                            equity,
                            signal=signal,
                        )
                    except TypeError:
                        try:
                            strategy_plan = build_trade_plan(
                                data,
                                len(data) - 1,
                                symbol,
                                None,
                                equity,
                            )
                        except TypeError:
                            strategy_plan = None

                analyze_trade = getattr(strategy, "analyze_trade", None)
                if strategy_plan is None and callable(analyze_trade):
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
                    size = float(strategy_plan.get("size")) * revenge_context["multiplier"]
                    size_reason = strategy_plan.get("size_reason", "strategy_plan")
                else:
                    stop, target = risk.calculate_sl_tp(signal, price, atr)

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
                            trade_states[symbol] = {
                                "ticket": getattr(current_position, "ticket", None),
                                "mfe_usd": float(getattr(current_position, "profit", 0.0) or 0.0),
                                "break_even_moved": False,
                                "break_even_sl": None,
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
