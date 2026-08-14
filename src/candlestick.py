from __future__ import annotations

import pandas as pd


def patterns(df: pd.DataFrame, side: str) -> list[str]:
    a, b = df.iloc[-2], df.iloc[-1]
    body = abs(b.Close - b.Open)
    span = max(b.High - b.Low, 1e-9)
    result = []
    if side == "buy":
        if a.Close < a.Open and b.Close > b.Open: result.append("陰線→陽線")
        if b.Close > b.Open and b.Open <= a.Close and b.Close >= a.Open: result.append("強気包み足")
        if b.High <= a.High and b.Low >= a.Low: result.append("はらみ足")
        if b.Close > b.Open and min(b.Open, b.Close) - b.Low >= max(body * 2, span * .4): result.append("長い下ヒゲ（ハンマー）")
        if b.Low < a.Low and b.Close > a.Close: result.append("前日安値割れ反転")
        # 2～5日すべてが連続安でなくても、直近まで売られていた初反転を拾う。
        prior = df.Close.iloc[-6:-1].pct_change().dropna()
        if b.Close > b.Open and len(prior) >= 2 and (prior < 0).sum() >= 2 and prior.sum() < 0:
            result.append("2～5日下落後の陽線")
    else:
        if a.Close > a.Open and b.Close < b.Open: result.append("陽線→陰線")
        if b.Close < b.Open and b.Open >= a.Close and b.Close <= a.Open: result.append("弱気包み足")
        if b.High <= a.High and b.Low >= a.Low: result.append("はらみ足")
        if b.Close < b.Open and b.High - max(b.Open, b.Close) >= max(body * 2, span * .4): result.append("長い上ヒゲ")
        if b.High > a.High and b.Close < a.Close: result.append("前日高値超え反転")
    return result
