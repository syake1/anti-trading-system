from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    return result.where(loss.ne(0), 100.0).where(gain.ne(0), 0.0)


def bollinger(close: pd.Series, period: int = 20, stds: float = 2) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + stds * sd,
                         "bb_lower": mid - stds * sd, "bb_sigma": (close - mid) / sd.replace(0, np.nan)})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = df["Close"].shift()
    true_range = pd.concat([(df["High"] - df["Low"]),
                            (df["High"] - previous).abs(),
                            (df["Low"] - previous).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def enrich(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    for n in (25, 75, 200):
        out[f"MA{n}"] = out["Close"].rolling(n).mean()
    out["RSI14"] = rsi(out["Close"], config["indicators"]["rsi_period"])
    out = out.join(bollinger(out["Close"], config["indicators"]["bb_period"], config["indicators"]["bb_std"]))
    out["ATR14"] = atr(out, config["indicators"]["atr_period"])
    out["volume_avg20"] = out["Volume"].shift(1).rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_avg20"]
    out["change_1d"] = out["Close"].pct_change() * 100
    out["change_3d"] = out["Close"].pct_change(3) * 100
    out["change_5d"] = out["Close"].pct_change(5) * 100
    out["ma25_deviation"] = (out["Close"] / out["MA25"] - 1) * 100
    out["range_atr"] = (out["High"] - out["Low"]) / out["ATR14"]
    return out
