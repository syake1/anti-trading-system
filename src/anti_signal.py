from __future__ import annotations

import pandas as pd


def detect(df: pd.DataFrame, slope_days: int = 3) -> dict | None:
    if len(df) < slope_days + 3 or df[["K", "D"]].iloc[-slope_days-2:].isna().any().any():
        return None
    k, d = df.K, df.D
    d_up, d_down = d.iloc[-1] > d.iloc[-1-slope_days], d.iloc[-1] < d.iloc[-1-slope_days]
    buy_pullback = k.iloc[-3] > k.iloc[-2] or k.iloc[-2] <= d.iloc[-2]
    sell_bounce = k.iloc[-3] < k.iloc[-2] or k.iloc[-2] >= d.iloc[-2]
    if d_up and buy_pullback and k.iloc[-1] > k.iloc[-2]:
        return {"side": "buy", "cross": k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2], "d_trend": True}
    if d_down and sell_bounce and k.iloc[-1] < k.iloc[-2]:
        return {"side": "sell", "cross": k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2], "d_trend": True}
    return None

