from __future__ import annotations

import numpy as np
import pandas as pd


def _adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    previous_close = close.shift()
    tr = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _snapshot(frame: pd.DataFrame) -> dict:
    if len(frame) < 80:
        return {"score": 0, "label": "insufficient_data", "components": {}}
    frame = frame.copy()
    previous_close = frame["close"].shift()
    tr = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    atr_median = atr.rolling(50, min_periods=30).median()
    ema = frame["close"].ewm(span=20, adjust=False).mean()
    slope = ema.diff(5) / 5
    bb_mid = frame["close"].rolling(20, min_periods=20).mean()
    bb_std = frame["close"].rolling(20, min_periods=20).std()
    bb_width = (4 * bb_std / bb_mid.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    bb_median = bb_width.rolling(50, min_periods=30).median()
    direction = np.sign(ema.diff()).replace(0, np.nan).ffill().rolling(5, min_periods=5).mean().abs()
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    volume = frame.get("tick_volume", pd.Series(1.0, index=frame.index)).fillna(0).replace(0, 1.0)
    vwap = (typical * volume).rolling(96, min_periods=30).sum() / volume.rolling(96, min_periods=30).sum()
    chop_tr = tr.rolling(14, min_periods=14).sum()
    chop_range = frame["high"].rolling(14, min_periods=14).max() - frame["low"].rolling(14, min_periods=14).min()
    choppiness = 100 * np.log10(chop_tr / chop_range.replace(0, np.nan)) / np.log10(14)
    same_direction = frame["close"].diff().gt(0).astype(int).rolling(8, min_periods=8).mean()
    persistence = same_direction.where(same_direction <= 0.5, 1 - same_direction).mul(2)
    row = pd.DataFrame({
        "adx": _adx(frame), "atr": atr, "atr_median": atr_median, "slope": slope,
        "bb_width": bb_width, "bb_median": bb_median, "direction": direction,
        "vwap": vwap, "choppiness": choppiness, "persistence": persistence,
    }).iloc[-1]
    values = row.fillna(np.nan)
    atr_value = float(values["atr"] or 0.0) if pd.notna(values["atr"]) else 0.0
    atr_med = float(values["atr_median"] or 0.0) if pd.notna(values["atr_median"]) else 0.0
    close = float(frame["close"].iloc[-1])
    adx = float(values["adx"]) if pd.notna(values["adx"]) else 999.0
    atr_ratio = atr_value / atr_med if atr_med > 0 else 999.0
    slope_ratio = abs(float(values["slope"])) / atr_value if atr_value > 0 and pd.notna(values["slope"]) else 999.0
    bb_ratio = float(values["bb_width"]) / float(values["bb_median"]) if pd.notna(values["bb_width"]) and pd.notna(values["bb_median"]) and values["bb_median"] > 0 else 999.0
    vwap_distance = abs(close - float(values["vwap"])) / atr_value if atr_value > 0 and pd.notna(values["vwap"]) else 999.0
    chop = float(values["choppiness"]) if pd.notna(values["choppiness"]) else 0.0
    persistence_value = float(values["persistence"]) if pd.notna(values["persistence"]) else 0.0
    components = {
        "adx_low": adx <= 22.0,
        "atr_stable": 0.7 <= atr_ratio <= 1.5,
        "ema_flat": slope_ratio <= 0.20,
        "bb_stable": bb_ratio <= 1.5,
        "choppy": chop >= 45.0,
        "near_vwap": vwap_distance <= 1.5,
        "mixed_candles": persistence_value >= 0.35,
    }
    score = sum(int(value) for value in components.values())
    label = "ranging" if score >= 5 and adx <= 25.0 else "trending" if adx >= 25.0 and slope_ratio > 0.20 else "neutral"
    return {
        "score": score,
        "label": label,
        "components": components,
        "adx": adx,
        "atr": atr_value,
        "atr_median": atr_med,
        "atr_ratio": atr_ratio,
        "ema_slope": float(values["slope"]) if pd.notna(values["slope"]) else None,
        "ema_slope_atr_ratio": slope_ratio,
        "bb_width": float(values["bb_width"]) if pd.notna(values["bb_width"]) else None,
        "bb_width_ratio": bb_ratio,
        "choppiness": chop,
        "vwap_distance_atr": vwap_distance,
        "trend_persistence": persistence_value,
        "close": close,
    }


def _resample_completed(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = data[["open", "high", "low", "close", "tick_volume"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
    ).dropna()
    return frame.iloc[:-1]


def classify(data: pd.DataFrame) -> dict:
    m15 = _snapshot(_resample_completed(data, "15min"))
    m30 = _snapshot(_resample_completed(data, "30min"))
    return {"m15": m15, "m30": m30}
