from __future__ import annotations


def score(df, signal: dict, pats: list[str], config: dict) -> tuple[int, list[str]]:
    w, side = config["weights"], signal["side"]
    now, prev = df.iloc[-1], df.iloc[-2]
    points, reasons = w["d_trend"] + w["k_reversal"], ["%D方向一致", "%K押し・戻りから反転"]
    def add(condition, key, label):
        nonlocal points
        if condition: points += w[key]; reasons.append(label)
    add(signal["cross"], "k_d_cross", "%Kが%Dをクロス")
    bullish = side == "buy"
    add(now.Close > now.MA200 if bullish else now.Close < now.MA200, "above_ma200", "200日線方向一致")
    add(now.MA25 > prev.MA25 if bullish else now.MA25 < prev.MA25, "ma25_slope", "MA25傾き一致")
    add(now.MA75 > prev.MA75 if bullish else now.MA75 < prev.MA75, "ma75_slope", "MA75傾き一致")
    add((prev.RSI14 < 40 and now.RSI14 > prev.RSI14) if bullish else (prev.RSI14 > 60 and now.RSI14 < prev.RSI14), "rsi40_reversal", "RSI強反転")
    add((30 <= now.RSI14 <= 55 and now.RSI14 > prev.RSI14) if bullish else (45 <= now.RSI14 <= 70 and now.RSI14 < prev.RSI14), "rsi_zone", "RSIゾーン反転")
    add((prev.bb_sigma <= -2 and now.Close > prev.Close) if bullish else (prev.bb_sigma >= 2 and now.Close < prev.Close), "bb_minus2_rebound", "BB±2σ反転")
    add((prev.bb_sigma <= -1 and now.Close > prev.Close) if bullish else (prev.bb_sigma >= 1 and now.Close < prev.Close), "bb_minus1_rebound", "BB±1σ反転")
    add(any("包み" in p for p in pats), "engulfing", "包み足")
    add(any("→" in p for p in pats), "color_reversal", "ローソク色反転")
    add(any("ヒゲ" in p for p in pats), "long_wick", "長いヒゲ")
    add(1.3 <= now.volume_ratio < 1.5, "volume_1_3", "出来高1.3倍")
    add(now.volume_ratio >= 1.5, "volume_1_5", "出来高1.5倍以上")
    return int(points), reasons


def rank(value: int, config: dict) -> str:
    t = config["rank_thresholds"]
    return "S" if value >= t["S"] else "A" if value >= t["A"] else "B" if value >= t["B"] else "C"
