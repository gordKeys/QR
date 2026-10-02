"""
Sell Gun -- MT5 (XM)
Opens as many SELL positions on gold (XAUUSD) as the account allows, each with
its own SL and TP. Works on any account type (demo or real) -- no gate.

Count = min(margin cap, risk cap, MAX_TRADES).
Deps: pip install MetaTrader5 pandas numpy
"""
import os, sys, time
from datetime import datetime, timezone
import pandas as pd
import MetaTrader5 as mt5

# ============================ LOGIN ==========================================
MT5_LOGIN = 362294406                 # your XM account number
MT5_PASSWORD = "Gordonpap@2023"          # your XM password (blank = attach to open, logged-in terminal)
MT5_SERVER = "XMGlobal-MT5 12"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# ============================ SETTINGS =======================================
SYMBOL_OVERRIDE = ""                       # exact XM name if you want to force one, e.g. "GOLD"
SYMBOL_CANDIDATES = ["XAUUSD", "GOLD"]     # tried in order, EXACT names only (XM often calls it GOLD)

DIRECTION = mt5.ORDER_TYPE_SELL
MAGIC = 130014
MAX_TRADES = 100
LOT = 0.01
MARGIN_USE_TARGET = 0.70    # fraction of equity usable as margin for the whole batch
BATCH_RISK_CAP = 1.00       # batch worst-case loss at SL as fraction of equity

SL_ATR_MULT = 1.5
TP_ATR_MULT = 12.0
ATR_PERIOD = 14
ATR_TF = "H1"

TF_MAP = {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}


def log(*a):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z]", *a, flush=True)


def connect():
    kw = {"path": TERMINAL_PATH}
    if MT5_PASSWORD:
        kw.update(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
    if not mt5.initialize(**kw):
        log(f"initialize failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        log(f"no account_info: {mt5.last_error()}")
        return False
    log(f"Connected {info.login} | {info.server} | bal {info.balance} {info.currency}")
    return True


def resolve_symbol():
    names = {s.name.upper(): s.name for s in (mt5.symbols_get() or [])}
    tries = [SYMBOL_OVERRIDE] if SYMBOL_OVERRIDE else SYMBOL_CANDIDATES
    chosen = None
    for t in tries:
        if t.upper() in names:
            chosen = names[t.upper()]
            break
    if chosen is None and not SYMBOL_OVERRIDE:
        # suffixed variants like XAUUSD.m / XAUUSD# (starts-with only, never substring)
        pre = sorted(n for u, n in names.items() if u.startswith("XAUUSD"))
        chosen = pre[0] if pre else None
    if chosen is None:
        log(f"no gold symbol found. Tried {tries}. Gold-like names: "
            f"{[n for u, n in names.items() if 'XAU' in u or u.startswith('GOLD')]}")
        return None
    if not mt5.symbol_select(chosen, True):
        log(f"could not select {chosen}")
        return None
    log(f"using symbol '{chosen}'")
    return chosen


def current_atr(symbol, tf_key, period):
    tf = TF_MAP.get(tf_key, mt5.TIMEFRAME_H1)
    for _ in range(20):                      # wait for a live tick / subscription
        t = mt5.symbol_info_tick(symbol)
        if t and t.time:
            break
        time.sleep(0.5)
    r = None
    for attempt in range(5):                 # wait for history to sync
        r = mt5.copy_rates_from_pos(symbol, tf, 0, period + 50)
        if r is not None and len(r) >= period + 2:
            break
        log(f"history not ready (attempt {attempt + 1}/5)...")
        time.sleep(2)
    if r is None or len(r) < period + 2:
        return None
    d = pd.DataFrame(r)
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def filling_mode(si):
    fm = si.filling_mode                      # bit 1 = FOK, bit 2 = IOC
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def compute_batch_size(symbol, si, sl_dist, lot):
    equity = mt5.account_info().equity
    tick = mt5.symbol_info_tick(symbol)
    px = tick.bid
    margin_each = mt5.order_calc_margin(DIRECTION, symbol, lot, px)
    if not margin_each or margin_each <= 0:
        log("could not compute margin; aborting")
        return 0
    risk_each = sl_dist * si.trade_contract_size * lot
    n_margin = int((equity * MARGIN_USE_TARGET) // margin_each)
    n_risk = int((equity * BATCH_RISK_CAP) // risk_each) if risk_each > 0 else 0
    n = max(0, min(n_margin, n_risk, MAX_TRADES))
    log(f"per position: margin ${margin_each:.2f}, SL loss ${risk_each:.2f} | equity ${equity:.2f}")
    log(f"margin cap {n_margin} | risk cap {n_risk} | opening {n}")
    return n


def open_positions(symbol):
    si = mt5.symbol_info(symbol)
    atr = current_atr(symbol, ATR_TF, ATR_PERIOD)
    if not atr or atr <= 0:
        log("could not measure ATR; aborting")
        return
    sl_dist, tp_dist = SL_ATR_MULT * atr, TP_ATR_MULT * atr
    lot = round(min(max(LOT, si.volume_min), si.volume_max), 4)
    log(f"{symbol}: ATR~{atr:.5g} | SL dist {sl_dist:.5g} | TP dist {tp_dist:.5g} | lot {lot}")

    n = compute_batch_size(symbol, si, sl_dist, lot)
    if not n:
        log("account can't fit a position under these limits")
        return

    fill = filling_mode(si)
    ok = 0
    for i in range(n):
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lot),
            "type": DIRECTION, "price": price,
            "sl": round(price + sl_dist, si.digits),
            "tp": round(price - tp_dist, si.digits),
            "deviation": 30, "magic": MAGIC, "comment": "sell_gun",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": fill,
        }
        res = mt5.order_send(req)
        if res is None:
            log(f"  {i + 1}/{n} -> order_send returned None: {mt5.last_error()}")
            continue
        log(f"  {i + 1}/{n} -> retcode={res.retcode} {res.comment}")
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            ok += 1
        elif res.retcode in (mt5.TRADE_RETCODE_NO_MONEY, mt5.TRADE_RETCODE_MARKET_CLOSED,
                             mt5.TRADE_RETCODE_TRADE_DISABLED, mt5.TRADE_RETCODE_LIMIT_POSITIONS):
            log("  stopping: broker refuses further orders")
            break
    log(f"done: {ok}/{n} positions opened")


def main():
    if not connect():
        sys.exit(1)
    try:
        symbol = resolve_symbol()
        if symbol:
            open_positions(symbol)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
