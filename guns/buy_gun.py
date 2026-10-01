"""
Buy Gun -- MT5 (HFM)
=====================

Opens as many BUY positions on GOLD as the account can support, sized so
that after opening them, the account has enough FREE margin/equity to
survive an adverse swing until they hit their SL or TP (not a margin call
mid-trade). Position count is derived from lot size + available equity, not
picked by hand.

This is the same sizing logic as your reference script (buy_bot.py), pulled
out into a broker-agnostic HFM version with clearer knobs. Two independent
caps decide the count, and the smaller one wins:

  1. MARGIN CAP  -- don't use more than MARGIN_USE_TARGET of equity as margin
                    for the whole batch, so free margin stays available.
  2. RISK CAP    -- the batch's total worst-case loss (if every position hits
                    its stop) can't exceed BATCH_RISK_CAP of equity.

Honest note, same as the reference: buying with no signal has no expected
edge -- direction is a coin flip, so over time this costs you spread +
commission. All positions are correlated (same direction), so if price
moves against you, they lose together. TEST ON DEMO FIRST.

Deps (Windows MT5 machine):  pip install MetaTrader5 pandas numpy
"""

import sys
from datetime import datetime, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


# ===================== YOUR HFM LOGIN (fill these in) ========================
MT5_LOGIN = 1302211830                # your HFM account number
MT5_PASSWORD = "Gordonpap@2023"           # <-- your HFM password
MT5_SERVER = "XMGlobal-MT5 6" 
# Leave MT5_PASSWORD blank to just attach to an MT5 terminal that's already
# open and logged in, instead of logging in from the script.

TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# ============================== SETTINGS =====================================
BASE_SYMBOL = "GOLD"        # base instrument name -- actual broker symbol is auto-resolved
DIRECTION = mt5.ORDER_TYPE_BUY if mt5 else 0

MAX_TRADES = 50               # hard ceiling regardless of what the math allows
LOT = 0.01                    # fixed lot per position (check the symbol's min lot)
MARGIN_USE_TARGET = 0.70      # use up to this fraction of equity as MARGIN across
                              # all positions opened; the rest stays FREE
BATCH_RISK_CAP = 1.00         # batch's worst-case loss at SL must stay under this
                              # fraction of equity (1.00 = never risk more than the account)

SL_ATR_MULT = 1.5             # stop distance  = this * ATR
TP_ATR_MULT = 12.0            # target distance = this * ATR
ATR_PERIOD = 14
ATR_TF = "H1"

FORCE_LIVE_ON_REAL = True    # safety: must be flipped to True to trade on a REAL account

TF_MAP = {
    "M15": getattr(mt5, "TIMEFRAME_M15", 15) if mt5 else 15,
    "H1": getattr(mt5, "TIMEFRAME_H1", 16385) if mt5 else 16385,
    "H4": getattr(mt5, "TIMEFRAME_H4", 16388) if mt5 else 16388,
}


def log(*a):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z]", *a)


def connect() -> bool:
    if mt5 is None:
        print("[FATAL] MetaTrader5 not importable. On the Windows MT5 machine: "
              "pip install MetaTrader5")
        return False
    kw = {"path": TERMINAL_PATH}
    if MT5_PASSWORD:
        kw.update(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER)
    if not mt5.initialize(**kw):
        print(f"[FATAL] initialize failed: {mt5.last_error()} -- check login/password/server "
              f"or open MT5 and log in first.")
        return False
    info = mt5.account_info()
    if info is None:
        print(f"[FATAL] no account_info: {mt5.last_error()}")
        return False
    log(f"Connected {info.login} | {info.server} | bal {info.balance} {info.currency} | "
        f"{'DEMO' if info.trade_mode == 0 else 'REAL/other'}")
    return True


def resolve_symbol(base: str):
    """
    Find the broker's actual symbol name for `base`, whatever prefix/suffix it
    uses (e.g. GOLD -> GOLDm, GOLD.raw, m.GOLD, GOLDm, ...).
    Preference order: exact match > shortest name containing base as a
    substring (fewest extra prefix/suffix chars) > first visible match.
    """
    all_syms = mt5.symbols_get()
    if not all_syms:
        log("could not fetch symbol list from broker")
        return None

    base_u = base.upper()
    candidates = [s.name for s in all_syms if base_u in s.name.upper()]
    if not candidates:
        log(f"no symbol containing '{base}' found on this account")
        return None

    exact = [n for n in candidates if n.upper() == base_u]
    if exact:
        chosen = exact[0]
    else:
        chosen = sorted(candidates, key=len)[0]  # shortest = least extra prefix/suffix

    if not mt5.symbol_select(chosen, True):
        log(f"found symbol '{chosen}' but could not select it")
        return None

    if len(candidates) > 1:
        log(f"resolved '{base}' -> '{chosen}' (other matches on this account: "
            f"{[c for c in candidates if c != chosen]})")
    else:
        log(f"resolved '{base}' -> '{chosen}'")
    return chosen


def current_atr(symbol, tf_key, period):
    tf = TF_MAP.get(tf_key, TF_MAP["H1"])
    r = mt5.copy_rates_from_pos(symbol, tf, 0, period + 50)
    if r is None or len(r) < period + 2:
        return None
    d = pd.DataFrame(r)
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def compute_batch_size(symbol, sl_dist, lot):
    """Shared sizing math: how many positions fit under the margin + risk caps."""
    si = mt5.symbol_info(symbol)
    acc = mt5.account_info()
    equity = acc.equity
    tick = mt5.symbol_info_tick(symbol)

    margin_each = mt5.order_calc_margin(DIRECTION, symbol, lot, tick.ask if DIRECTION == mt5.ORDER_TYPE_BUY else tick.bid)
    if not margin_each or margin_each <= 0:
        log("could not compute margin for this symbol; aborting")
        return None

    risk_each = sl_dist * si.trade_contract_size * lot  # worst-case loss if the stop hits

    n_margin = int((equity * MARGIN_USE_TARGET) // margin_each)
    n_risk = int((equity * BATCH_RISK_CAP) // risk_each) if risk_each > 0 else 0
    n = max(0, min(n_margin, n_risk, MAX_TRADES))

    used_margin = margin_each * n
    batch_loss = risk_each * n
    log(f"per position: margin ${margin_each:.2f}, worst-case loss ${risk_each:.2f} | equity ${equity:.2f}")
    log(f"margin cap -> {n_margin} | risk cap -> {n_risk} | opening {n} "
        f"(margin used ${used_margin:.2f} = {used_margin / equity * 100:.0f}%, "
        f"free ${equity - used_margin:.2f}; worst-case loss ${batch_loss:.2f} = "
        f"{batch_loss / equity * 100:.0f}%)")
    return n


def open_positions(symbol):
    if not mt5.symbol_select(symbol, True) or mt5.symbol_info(symbol) is None:
        log(f"symbol not available: {symbol}")
        return
    si = mt5.symbol_info(symbol)
    atr = current_atr(symbol, ATR_TF, ATR_PERIOD)
    if not atr or atr <= 0:
        log("could not measure ATR; aborting")
        return

    sl_dist, tp_dist = SL_ATR_MULT * atr, TP_ATR_MULT * atr
    lot = round(min(max(LOT, si.volume_min), si.volume_max), 4)

    log(f"{symbol}: ATR~{atr:.5g} | SL~{sl_dist:.5g} | TP~{tp_dist:.5g} | lot={lot}")

    n = compute_batch_size(symbol, sl_dist, lot)
    if not n:
        log("Can't fit even one position under these limits at this balance. Deposit more, or "
            "on DEMO raise MARGIN_USE_TARGET / BATCH_RISK_CAP to push it.")
        return

    for i in range(n):
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lot),
            "type": mt5.ORDER_TYPE_BUY, "price": price,
            "sl": round(price - sl_dist, si.digits),
            "tp": round(price + tp_dist, si.digits),
            "deviation": 30, "magic": 130013, "comment": "buy_gun",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        log(f"  buy {i + 1}/{n} -> retcode={res.retcode} {res.comment}")


def main():
    if not connect():
        sys.exit(1)
    info = mt5.account_info()
    if info.trade_mode != 0 and not FORCE_LIVE_ON_REAL:
        log("REAL account -- refusing to place live trades. Test on DEMO, or set "
            "FORCE_LIVE_ON_REAL = True to allow real orders.")
        mt5.shutdown()
        return
    try:
        symbol = resolve_symbol(BASE_SYMBOL)
        if not symbol:
            log(f"could not resolve a broker symbol for '{BASE_SYMBOL}'; aborting")
            return
        open_positions(symbol)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
