from __future__ import annotations


def score(df, signal: dict, pats: list[str], config: dict) -> tuple[int, list[str]]:
    w, side = config["weights"], signal["side"]
    now, prev = df.iloc[-1], df.iloc[-2]
    points, reasons = 0, []
    def add(condition, key, label):
        nonlocal points
        if condition: points += w[key]; reasons.append(label)
    bullish = side == "buy"
    if bullish:
        # 買いは「上昇中」ではなく、直近の下落からの反転だけを中心に採点する。
        prior = df.Close.iloc[-6:-1].pct_change().dropna()
        recently_falling = len(prior) >= 2 and (prior < 0).sum() >= 2 and prior.sum() < 0
        add(recently_falling, "recent_decline", "直近2～5日まで下落")
        add(prev.RSI14 <= 40 and now.RSI14 > prev.RSI14, "rsi40_reversal", "RSI40以下から反転上昇")
        add(prev.bb_sigma <= -2 and now.Close > prev.Close, "bb_minus2_rebound", "BB -2σ付近から反発")
        add(-2 < prev.bb_sigma <= -1 and now.Close > prev.Close, "bb_minus1_rebound", "BB -1σ以下から反発")
        add(min(df.K.iloc[-3], prev.K) <= 25 and now.K > prev.K, "stoch_oversold_reversal", "ストキャス売られ過ぎから反転")
        add(now.K > prev.K, "k_reversal", "%K反転")
        add(now.D > prev.D, "d_improvement", "%D方向改善")
        add(any("強気包み" in p for p in pats), "engulfing", "陽の包み足")
        add(any("下ヒゲ" in p or "前日安値割れ" in p for p in pats), "long_wick", "長い下ヒゲ反転")
        add(any("下落後" in p for p in pats), "decline_bullish", "2～5日下落後の陽線")
        reversal = now.Close > now.Open and (recently_falling or bool(pats))
        add(reversal and now.volume_ratio >= 1.3, "volume_reversal", "出来高増加を伴う反転")
        if not recently_falling:
            # 単なる当日上昇を通常通知へ上げない。足型等を記録してもB以下に留める。
            points = min(points, config["rank_thresholds"]["A"] - 1)
            reasons.append("直近下落なし（B以下に制限）")
        return int(points), reasons

    points, reasons = w["d_trend"] + w["k_reversal"], ["%D方向一致", "%K押し・戻りから反転"]
    add(signal["cross"], "k_d_cross", "%Kが%Dをクロス")
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
