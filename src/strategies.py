"""Independent, explainable strategy detectors for the daily scanner."""
from __future__ import annotations

import pandas as pd


def _rising_lows(df: pd.DataFrame, window: int = 10) -> bool:
    lows = df.Low.iloc[-window:]
    half = max(2, len(lows) // 2)
    return lows.iloc[-half:].min() > lows.iloc[:-half].min()


def accumulation(df: pd.DataFrame, config: dict) -> tuple[bool, list[str]]:
    """Detect objective base-building features; never claims actual accumulation."""
    c = config["accumulation"]
    now = df.iloc[-1]
    lookback = int(c["lookback_days"])
    recent, prior = df.iloc[-lookback:], df.iloc[-lookback * 2:-lookback]
    if len(prior) < lookback:
        return False, []
    reasons = []
    if _rising_lows(df, lookback): reasons.append("直近安値切り上げ")
    if recent.ATR14.mean() < prior.ATR14.mean() * c["atr_contraction_ratio"]: reasons.append("ATR低下")
    recent_width = ((recent.bb_upper - recent.bb_lower) / recent.bb_mid).mean()
    prior_width = ((prior.bb_upper - prior.bb_lower) / prior.bb_mid).mean()
    if recent_width < prior_width * c["bb_width_ratio"]: reasons.append("BB幅収縮")
    quiet = df.volume_ratio.iloc[-6:-1].mean() < c["quiet_volume_ratio"]
    if quiet and now.volume_ratio >= c["breakout_volume_ratio"]: reasons.append("出来高減少後の増加")
    if df.RSI14.iloc[-5:].min() > df.RSI14.iloc[-10:-5].min(): reasons.append("RSI安値切り上げ")
    range_high = df.High.iloc[-lookback-1:-1].max()
    if now.Close > range_high: reasons.append("レンジ上限突破")
    confirmed = len(reasons) >= c["minimum_features"] and (
        "レンジ上限突破" in reasons or "直近安値切り上げ" in reasons)
    return confirmed, reasons


def evaluate(df: pd.DataFrame, anti: dict | None, pats: list[str], config: dict) -> dict:
    """Return independent A-D strategy flags and a reversal gate."""
    now, prev = df.iloc[-1], df.iloc[-2]
    bullish_candle = now.Close > now.Open and now.Close > prev.Close
    candle_reversal = bullish_candle or any(
        key in p for p in pats for key in ("包み", "下ヒゲ", "前日安値割れ", "下落後")
    )
    bb_rebound = prev.bb_sigma <= -1 and now.Close > prev.Close
    rsi_reversal = 25 <= now.RSI14 <= 40 and now.RSI14 > prev.RSI14
    recently_oversold = df.RSI14.iloc[-4:-1].min() <= 30 and now.RSI14 > prev.RSI14
    stoch_reversal = df.K.iloc[-3:-1].min() <= 25 and now.K > prev.K and now.K < 50
    stoch_cross = stoch_reversal and now.K > now.D and prev.K <= prev.D
    base, base_reasons = accumulation(df, config)
    flags = {
        "BB逆張り": bb_rebound and candle_reversal,
        "BB＋RSI＋ストキャス": bb_rebound and candle_reversal and (rsi_reversal or recently_oversold) and stoch_reversal,
        "底固め": base,
        "アンチ": bool(anti and anti.get("side") == "buy"),
    }
    return {"flags": flags, "reversal": candle_reversal, "bb_rebound": bb_rebound,
            "rsi_reversal": rsi_reversal or recently_oversold, "stoch_reversal": stoch_reversal,
            "stoch_cross": stoch_cross, "base_reasons": base_reasons}
