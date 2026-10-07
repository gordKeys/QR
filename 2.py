"""
================================================================================

  ██╗   ██╗██╗  ██╗
  ██║   ██║╚██╗██╔╝
  ██║   ██║ ╚███╔╝
  ╚██╗ ██╔╝ ██╔██╗
   ╚████╔╝ ██╔╝ ██╗
    ╚═══╝  ╚═╝  ╚═╝  v2 - works good made two profits

  -  Jesus will do it for me Amen  -

  VX TRADING BOT — LIVE DEPLOYMENT v2
  Broker    : Exness (MT5)
  Symbols   : EURUSDm, GBPUSDm, XAUUSDm
  Timeframe : M15 + H1 trend filter

  v2 IMPROVEMENTS OVER v1:
  1. datetime.utcnow() deprecation fixed → datetime.now(UTC)
  2. Session start buffer — no trades in first 2hrs (7-9am UTC)
  3. Stricter entry after losses — requires 5/5 confirmations
     on first trade after a losing streak (instead of 4/5)
  4. Martingale pause — after 2 consecutive losses, wait 1 candle
     (15 mins) before re-entering to avoid chasing bad markets
  5. Gold-first priority — when Gold has a valid signal, skip
     EUR/GBP that loop to avoid correlated exposure

  BACKTEST: 82.8% return / 90 days | 27.6% monthly | PF: 1.79
  Mode     : UNRESTRICTED — same risk as v1
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime, timezone, timedelta

UTC = timezone.utc

# ─────────────────────────────────────────────
#  ACCOUNT
# ─────────────────────────────────────────────

ACCOUNT_LOGIN    = int(os.getenv("VX_MT5_LOGIN", "0"))
ACCOUNT_SERVER   = os.getenv("VX_MT5_SERVER", "Exness-MT5Real")
ACCOUNT_PASSWORD = os.getenv("VX_MT5_PASSWORD", "")
TERMINAL_PATH   = os.getenv("VX_MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")

# ─────────────────────────────────────────────
#  VX v2 SETTINGS
# ─────────────────────────────────────────────

SYMBOLS   = ["EURUSDm", "GBPUSDm", "XAUUSDm"]
TIMEFRAME = mt5.TIMEFRAME_M15
RR_RATIO  = 2.0

SYMBOL_CONFIG = {
    "EURUSDm": {
        "min_confirmations"        : 4,   # Normal entry threshold
        "min_confirmations_recover": 5,   # After a loss — stricter
        "atr_multiplier"           : 1.2,
        "base_risk"                : 2.0,
        "max_risk"                 : 10.0,
    },
    "GBPUSDm": {
        "min_confirmations"        : 4,
        "min_confirmations_recover": 5,
        "atr_multiplier"           : 1.3,
        "base_risk"                : 2.0,
        "max_risk"                 : 10.0,
    },
    "XAUUSDm": {
        "min_confirmations"        : 4,
        "min_confirmations_recover": 5,
        "atr_multiplier"           : 1.5,
        "base_risk"                : 3.0,
        "max_risk"                 : 15.0,
    },
}

# Sizing
MARTINGALE_FACTOR = 2.0
MAX_MARTINGALE    = 4.0
WIN_STREAK_BOOST  = 1.5
MAX_WIN_BOOST     = 4.0

# v2: Martingale pause after N consecutive losses
MARTINGALE_PAUSE_LOSSES  = 2       # Pause after this many losses in a row
MARTINGALE_PAUSE_MINUTES = 15      # Wait this many minutes before next entry

# Filters
USE_SESSION_FILTER  = True
SESSION_START_UTC   = 9            # v2: No trades before 9am UTC (was 7am)
SESSION_END_UTC     = 22
ACTIVE_HOURS_UTC    = list(range(SESSION_START_UTC, SESSION_END_UTC))
USE_TREND_FILTER    = True
H1_TREND_EMA        = 50
USE_REENTRY         = True
USE_GOLD_PRIORITY   = True         # v2: When Gold signals, skip EUR/GBP

# Indicators
FAST_EMA      = 9
SLOW_EMA      = 21
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
RSI_PERIOD    = 14
RSI_OB        = 65
RSI_OS        = 35
BB_PERIOD     = 20
BB_STD        = 2.0
ATR_PERIOD    = 14
VOLUME_FACTOR = 1.1
CANDLE_COUNT  = 150
LOOP_INTERVAL = 60


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler("VX_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


# ─────────────────────────────────────────────
#  PER SYMBOL STATE
# ─────────────────────────────────────────────

state = {
    sym: {
        "consec_wins"   : 0,
        "consec_losses" : 0,
        "martingale"    : 1.0,
        "last_signal"   : None,
        "last_sl_time"  : None,   # datetime of last SL hit
        "total_trades"  : 0,
        "total_wins"    : 0,
        "total_pnl"     : 0.0,
    }
    for sym in SYMBOLS
}


# ─────────────────────────────────────────────
#  CONNECTION
# ─────────────────────────────────────────────

def connect():
    if not ACCOUNT_LOGIN or not ACCOUNT_PASSWORD:
        log.error("Set VX_MT5_LOGIN and VX_MT5_PASSWORD before starting the bot")
        return False
    if not mt5.initialize(path=TERMINAL_PATH):
        log.error("MT5 init failed. Is MT5 running?")
        return False
    authorized = mt5.login(
        login=ACCOUNT_LOGIN,
        password=ACCOUNT_PASSWORD,
        server=ACCOUNT_SERVER
    )
    if not authorized:
        log.error(f"Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    info = mt5.account_info()
    log.info(f"VX v2 CONNECTED  |  Account: {info.login}  |  Balance: ${info.balance:.2f}  |  Equity: ${info.equity:.2f}")
    for sym in SYMBOLS:
        mt5.symbol_select(sym, True)
    return True


# ─────────────────────────────────────────────
#  MARKET DATA
# ─────────────────────────────────────────────

def get_candles(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def get_h1_trend(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
    if rates is None or len(rates) == 0:
        return None
    closes = pd.Series([r["close"] for r in rates])
    ema    = closes.ewm(span=H1_TREND_EMA, adjust=False).mean()
    return "UP" if rates[-1]["close"] > ema.iloc[-1] else "DOWN"


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────

def add_indicators(df):
    close  = df["close"]
    volume = df["tick_volume"]

    df["fast_ema"]  = close.ewm(span=FAST_EMA, adjust=False).mean()
    df["slow_ema"]  = close.ewm(span=SLOW_EMA, adjust=False).mean()

    macd_fast       = close.ewm(span=MACD_FAST, adjust=False).mean()
    macd_slow       = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"]      = macd_fast - macd_slow
    df["macd_sig"]  = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs       = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"]   = close.rolling(BB_PERIOD).mean()
    bb_std         = close.rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift()).abs(),
        (df["low"]  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(alpha=1/ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
    df["vol_avg"] = volume.rolling(20).mean()

    return df


# ─────────────────────────────────────────────
#  CANDLESTICK PATTERNS
# ─────────────────────────────────────────────

def bullish_engulfing(curr, prev):
    return (prev["close"] < prev["open"] and curr["close"] > curr["open"] and
            curr["open"] < prev["close"] and curr["close"] > prev["open"])

def bearish_engulfing(curr, prev):
    return (prev["close"] > prev["open"] and curr["close"] < curr["open"] and
            curr["open"] > prev["close"] and curr["close"] < prev["open"])

def bullish_pin_bar(row):
    body = abs(row["close"] - row["open"])
    if body == 0: return False
    lower = min(row["close"], row["open"]) - row["low"]
    upper = row["high"] - max(row["close"], row["open"])
    return lower >= 2 * body and upper <= body

def bearish_pin_bar(row):
    body = abs(row["close"] - row["open"])
    if body == 0: return False
    upper = row["high"] - max(row["close"], row["open"])
    lower = min(row["close"], row["open"]) - row["low"]
    return upper >= 2 * body and lower <= body


# ─────────────────────────────────────────────
#  v2: MARTINGALE PAUSE CHECK
# ─────────────────────────────────────────────

def is_paused(sym_state, symbol):
    """
    Returns True if this symbol should be skipped due to
    consecutive losses + pause timer.
    """
    if sym_state["consec_losses"] < MARTINGALE_PAUSE_LOSSES:
        return False

    last_sl = sym_state.get("last_sl_time")
    if last_sl is None:
        return False

    # Use timezone-aware UTC now
    now_utc   = datetime.now(UTC)
    # last_sl_time stored as naive local — convert for comparison
    elapsed   = (now_utc - last_sl.replace(tzinfo=UTC)).total_seconds() / 60

    if elapsed < MARTINGALE_PAUSE_MINUTES:
        remaining = int(MARTINGALE_PAUSE_MINUTES - elapsed)
        log.info(
            f"[{symbol}]  PAUSED after {sym_state['consec_losses']} losses  |  "
            f"Resuming in {remaining} min"
        )
        return True

    return False


# ─────────────────────────────────────────────
#  SIGNAL ENGINE
# ─────────────────────────────────────────────

def get_signal(symbol, df, h1_trend, sym_state):
    cfg = SYMBOL_CONFIG[symbol]

    # v2: Use stricter threshold if recovering from a loss
    if sym_state["consec_losses"] > 0:
        min_con = cfg["min_confirmations_recover"]
    else:
        min_con = cfg["min_confirmations"]

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    if pd.isna(curr["atr"]) or pd.isna(curr["bb_upper"]):
        return None, 0, 0

    # v2: Fixed datetime — no deprecation warning
    now_utc = datetime.now(UTC)
    if USE_SESSION_FILTER and now_utc.hour not in ACTIVE_HOURS_UTC:
        log.info(f"[{symbol}]  Outside session hours ({now_utc.hour}:00 UTC) — skipping")
        return None, 0, 0

    vol = curr["tick_volume"]

    buy = {
        "ema_cross": prev["fast_ema"] <= prev["slow_ema"] and curr["fast_ema"] > curr["slow_ema"],
        "macd"     : curr["macd_hist"] > 0 and curr["macd_hist"] > prev["macd_hist"],
        "bb"       : curr["close"] <= curr["bb_lower"] * 1.001 and curr["rsi"] < RSI_OB,
        "candle"   : bullish_engulfing(curr, prev) or bullish_pin_bar(curr),
        "volume"   : vol >= curr["vol_avg"] * VOLUME_FACTOR if curr["vol_avg"] > 0 else False,
    }

    sell = {
        "ema_cross": prev["fast_ema"] >= prev["slow_ema"] and curr["fast_ema"] < curr["slow_ema"],
        "macd"     : curr["macd_hist"] < 0 and curr["macd_hist"] < prev["macd_hist"],
        "bb"       : curr["close"] >= curr["bb_upper"] * 0.999 and curr["rsi"] > RSI_OS,
        "candle"   : bearish_engulfing(curr, prev) or bearish_pin_bar(curr),
        "volume"   : vol >= curr["vol_avg"] * VOLUME_FACTOR if curr["vol_avg"] > 0 else False,
    }

    buy_score  = sum(buy.values())
    sell_score = sum(sell.values())

    log.info(f"[{symbol}]  BUY  conditions: " + " | ".join([f"{k}={'Y' if v else 'N'}" for k,v in buy.items()]))
    log.info(f"[{symbol}]  SELL conditions: " + " | ".join([f"{k}={'Y' if v else 'N'}" for k,v in sell.items()]))

    if USE_TREND_FILTER and h1_trend is not None:
        if h1_trend == "DOWN":
            buy_score = 0
            log.info(f"[{symbol}]  H1 trend DOWN — BUY blocked")
        if h1_trend == "UP":
            sell_score = 0
            log.info(f"[{symbol}]  H1 trend UP — SELL blocked")

    atr = curr["atr"]

    # v2: Log which threshold is active
    threshold_label = f"(recover mode — {min_con}/5)" if sym_state["consec_losses"] > 0 else f"({min_con}/5)"
    log.info(
        f"[{symbol}]  Score: BUY {buy_score}/5  SELL {sell_score}/5  |  "
        f"Threshold: {threshold_label}  |  RSI: {curr['rsi']:.1f}  ATR: {atr:.5f}  H1: {h1_trend}"
    )

    if buy_score >= min_con and buy_score >= sell_score:
        log.info(f"[{symbol}]  BUY SIGNAL CONFIRMED  {buy_score}/5")
        return "BUY", atr, buy_score

    if sell_score >= min_con and sell_score > buy_score:
        log.info(f"[{symbol}]  SELL SIGNAL CONFIRMED  {sell_score}/5")
        return "SELL", atr, sell_score

    log.info(f"[{symbol}]  No signal")
    return None, atr, max(buy_score, sell_score)


# ─────────────────────────────────────────────
#  POSITION SIZING — UNRESTRICTED VX
# ─────────────────────────────────────────────

def get_volume(symbol, balance, atr, sym_state):
    cfg         = SYMBOL_CONFIG[symbol]
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0

    pip         = symbol_info.point
    pip_val     = symbol_info.trade_tick_value
    sl_distance = cfg["atr_multiplier"] * atr
    sl_pips     = sl_distance / pip

    if sl_pips <= 0 or pip_val <= 0:
        return symbol_info.volume_min

    wins     = sym_state["consec_wins"]
    win_mult = min(WIN_STREAK_BOOST ** max(0, wins - 1), MAX_WIN_BOOST)
    mart_mult = sym_state["martingale"]

    eff_risk    = cfg["base_risk"] * win_mult * mart_mult
    eff_risk    = min(eff_risk, cfg["max_risk"])

    risk_amount = balance * (eff_risk / 100)
    volume      = risk_amount / (sl_pips * pip_val)

    volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
    step   = symbol_info.volume_step
    volume = round(round(volume / step) * step, 2)

    log.info(
        f"[{symbol}]  VX Sizing  |  "
        f"Base: {cfg['base_risk']}%  "
        f"WinMult: {win_mult:.2f}x  "
        f"Martingale: {mart_mult:.2f}x  "
        f"EffRisk: {eff_risk:.1f}%  "
        f"Vol: {volume}  "
        f"Risk$: ${risk_amount:.2f}"
    )
    return volume


# ─────────────────────────────────────────────
#  TRADE EXECUTION
# ─────────────────────────────────────────────

def place_order(symbol, signal, atr, volume, score):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False

    tick        = mt5.symbol_info_tick(symbol)
    sl_distance = SYMBOL_CONFIG[symbol]["atr_multiplier"] * atr
    tp_distance = sl_distance * RR_RATIO

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
        sl         = price - sl_distance
        tp         = price + tp_distance
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
        sl         = price + sl_distance
        tp         = price - tp_distance

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : volume,
        "type"        : order_type,
        "price"       : price,
        "sl"          : round(sl, symbol_info.digits),
        "tp"          : round(tp, symbol_info.digits),
        "deviation"   : 20,
        "magic"       : 999999,
        "comment"     : f"VX2 {signal} {score}/5",
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"[{symbol}]  Order FAILED  |  Code: {result.retcode}  Msg: {result.comment}")
        return False

    log.info(
        f"[{symbol}]  {signal} ORDER PLACED  |  "
        f"Price: {price:.5f}  "
        f"SL: {round(sl, symbol_info.digits)}  "
        f"TP: {round(tp, symbol_info.digits)}  "
        f"Vol: {volume}  "
        f"Score: {score}/5"
    )
    return True


def get_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return positions[0] if positions and len(positions) > 0 else None


# ─────────────────────────────────────────────
#  TRACK CLOSED TRADES
# ─────────────────────────────────────────────

def update_state_from_history(symbol, sym_state):
    now   = datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    deals = mt5.history_deals_get(start, now)
    if deals is None or len(deals) == 0:
        return

    sym_deals = [
        d for d in deals
        if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT
    ]
    if not sym_deals:
        return

    last   = sorted(sym_deals, key=lambda d: d.time)[-1]
    profit = last.profit

    sym_state["total_trades"] += 1
    sym_state["total_pnl"]    += profit

    if profit > 0:
        sym_state["total_wins"]    += 1
        sym_state["consec_wins"]   += 1
        sym_state["consec_losses"]  = 0
        sym_state["martingale"]     = 1.0
        sym_state["last_signal"]    = None
        log.info(
            f"[{symbol}]  CLOSED WIN  +${profit:.2f}  |  "
            f"Win streak: {sym_state['consec_wins']}  |  "
            f"Total P&L: ${sym_state['total_pnl']:.2f}"
        )
    else:
        sym_state["consec_losses"] += 1
        sym_state["consec_wins"]    = 0
        sym_state["last_sl_time"]   = datetime.now()
        sym_state["martingale"]     = min(
            sym_state["martingale"] * MARTINGALE_FACTOR,
            MAX_MARTINGALE
        )
        log.info(
            f"[{symbol}]  CLOSED LOSS  ${profit:.2f}  |  "
            f"Loss streak: {sym_state['consec_losses']}  |  "
            f"Martingale: {sym_state['martingale']:.2f}x  |  "
            f"Total P&L: ${sym_state['total_pnl']:.2f}"
        )


# ─────────────────────────────────────────────
#  PERFORMANCE SUMMARY
# ─────────────────────────────────────────────

def print_summary(start_balance):
    account = mt5.account_info()
    if account is None:
        return
    pnl     = account.balance - start_balance
    pnl_pct = (pnl / start_balance) * 100

    log.info("\n" + "=" * 55)
    log.info("  VX v2 LIVE PERFORMANCE SUMMARY")
    log.info("=" * 55)
    log.info(f"  Starting balance : ${start_balance:.2f}")
    log.info(f"  Current balance  : ${account.balance:.2f}")
    log.info(f"  Current equity   : ${account.equity:.2f}")
    log.info(f"  Total P&L        : ${pnl:.2f} ({pnl_pct:+.1f}%)")
    log.info("  --- Per Symbol ---")
    for sym in SYMBOLS:
        s  = state[sym]
        wr = (s["total_wins"] / s["total_trades"] * 100) if s["total_trades"] > 0 else 0
        log.info(
            f"  {sym:<12}  "
            f"Trades: {s['total_trades']}  "
            f"WR: {wr:.0f}%  "
            f"P&L: ${s['total_pnl']:.2f}  "
            f"Streak: {s['consec_wins']}W/{s['consec_losses']}L  "
            f"Mart: {s['martingale']:.1f}x"
        )
    log.info("=" * 55)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def run_vx():
    if not connect():
        return

    account       = mt5.account_info()
    start_balance = account.balance

    log.info("=" * 65)
    log.info("  VX TRADING BOT v2 — LIVE")
    log.info(f"  Balance     : ${account.balance:.2f}")
    log.info(f"  Session     : {SESSION_START_UTC}am-{SESSION_END_UTC}pm UTC (v2: no early trades)")
    log.info(f"  Recover mode: 5/5 confirmations after any loss")
    log.info(f"  Mart pause  : {MARTINGALE_PAUSE_MINUTES}min after {MARTINGALE_PAUSE_LOSSES} consecutive losses")
    log.info(f"  Gold first  : {'ON' if USE_GOLD_PRIORITY else 'OFF'}")
    log.info(f"  Mode        : UNRESTRICTED")
    log.info("=" * 65)

    prev_positions = {sym: None for sym in SYMBOLS}
    loop_count     = 0

    try:
        while True:
            now        = datetime.now()
            now_utc    = datetime.now(UTC)
            loop_count += 1
            log.info(f"\n-- VX v2 tick #{loop_count}: {now.strftime('%Y-%m-%d %H:%M:%S')} --")

            account = mt5.account_info()
            if account:
                pnl = account.equity - start_balance
                log.info(
                    f"Balance: ${account.balance:.2f}  |  "
                    f"Equity: ${account.equity:.2f}  |  "
                    f"Session P&L: ${pnl:+.2f}"
                )

            # ── v2: Gold priority check ──────────────────
            # Scan Gold first. If it has a valid signal,
            # set flag to skip EUR/GBP this loop.
            gold_trading_this_loop = False

            # Pre-scan Gold for signal (won't place — just peek)
            if USE_GOLD_PRIORITY:
                gold_state = state["XAUUSDm"]
                gold_pos   = get_open_position("XAUUSDm")
                if not gold_pos and not is_paused(gold_state, "XAUUSDm"):
                    gdf = get_candles("XAUUSDm", TIMEFRAME, CANDLE_COUNT)
                    if gdf is not None:
                        gdf      = add_indicators(gdf)
                        g_trend  = get_h1_trend("XAUUSDm") if USE_TREND_FILTER else None
                        g_signal, g_atr, g_score = get_signal("XAUUSDm", gdf, g_trend, gold_state)
                        if g_signal:
                            gold_trading_this_loop = True
                            log.info("  [GOLD PRIORITY] Gold has a signal — skipping EUR/GBP this loop")

            # ── Main symbol loop ─────────────────────────
            for symbol in SYMBOLS:
                sym_state = state[symbol]
                log.info(f"\n-- {symbol} --")

                # v2: Gold priority — skip minor pairs if Gold trading
                if USE_GOLD_PRIORITY and gold_trading_this_loop and symbol != "XAUUSDm":
                    log.info(f"[{symbol}]  Skipped — Gold priority active this loop")
                    continue

                curr_pos = get_open_position(symbol)

                # Detect closed position and update state
                if prev_positions[symbol] is not None and curr_pos is None:
                    update_state_from_history(symbol, sym_state)

                prev_positions[symbol] = curr_pos

                # Position open — let SL/TP manage
                if curr_pos:
                    log.info(
                        f"[{symbol}]  OPEN  |  "
                        f"Ticket: {curr_pos.ticket}  "
                        f"Vol: {curr_pos.volume}  "
                        f"P&L: ${curr_pos.profit:.2f}"
                    )
                    continue

                # v2: Martingale pause check
                if is_paused(sym_state, symbol):
                    continue

                # Get candles + indicators
                df = get_candles(symbol, TIMEFRAME, CANDLE_COUNT)
                if df is None:
                    log.error(f"[{symbol}]  Could not fetch candles")
                    continue
                df = add_indicators(df)

                h1_trend = get_h1_trend(symbol) if USE_TREND_FILTER else None

                # Get signal (v2: passes sym_state for recover-mode threshold)
                signal, atr, score = get_signal(symbol, df, h1_trend, sym_state)

                # Re-entry check
                if signal is None and USE_REENTRY:
                    last_sl  = sym_state.get("last_sl_time")
                    last_sig = sym_state.get("last_signal")
                    if (last_sl is not None and last_sig is not None and
                            (now - last_sl).seconds < 300):
                        curr = df.iloc[-1]
                        if last_sig == "BUY" and curr["fast_ema"] > curr["slow_ema"] and curr["rsi"] < RSI_OB:
                            signal = "BUY"
                            log.info(f"[{symbol}]  RE-ENTRY BUY")
                        elif last_sig == "SELL" and curr["fast_ema"] < curr["slow_ema"] and curr["rsi"] > RSI_OS:
                            signal = "SELL"
                            log.info(f"[{symbol}]  RE-ENTRY SELL")

                # Place order
                if signal:
                    balance = mt5.account_info().balance
                    volume  = get_volume(symbol, balance, atr, sym_state)
                    if volume > 0:
                        success = place_order(symbol, signal, atr, volume, score)
                        if success:
                            sym_state["last_signal"] = signal

            # Summary every 10 loops
            if loop_count % 10 == 0:
                print_summary(start_balance)

            log.info(f"\nVX v2 waiting {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log.info("\nVX v2 stopped manually.")
        print_summary(start_balance)
    finally:
        mt5.shutdown()
        log.info("VX v2 disconnected.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_vx()
