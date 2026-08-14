"""Deterministic order construction (no broker connectivity).

The module deliberately contains no LLM call: persisted market data and config are
the complete inputs, which makes an order proposal reproducible and auditable.
"""
from __future__ import annotations

import math
import pandas as pd


ORDER_LABELS = {"market": "成り行き", "limit": "指値", "stop": "逆指値買い", "skip": "見送り"}


def number(row: dict, key: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def order_prices(row: dict, config: dict) -> dict:
    """Calculate trigger, zone, chase limit, stop and RR target."""
    rules = config["order_rules"]
    close = number(row, "現在値")
    atr = max(number(row, "ATR14", number(row, "ATR", close * .02)), close * .001)
    previous_high = number(row, "前日高値", close)
    reversal_high = number(row, "反転足高値", previous_high)
    high2 = number(row, "直近2日高値", max(previous_high, reversal_high))
    confirmation = max(previous_high, reversal_high, high2)
    trigger = confirmation + atr * rules["stop_entry_atr_buffer"]

    bb1 = number(row, "BB-1σ", close)
    bb2 = number(row, "BB-2σ", close)
    ma25 = number(row, "MA25", close)
    support = number(row, "直近支持線", min(close, bb1, ma25))
    anchors = [x for x in (close, bb1, bb2, ma25, support) if x > 0]
    zone_mid = min(close, max(bb2, support, min(bb1, ma25))) if anchors else close
    zone_low = max(0.01, zone_mid - atr * rules["limit_zone_atr_width"])
    zone_high = min(close, zone_mid + atr * rules["limit_zone_atr_width"])
    if zone_high < zone_low:
        zone_low, zone_high = zone_high, zone_low

    chase = max(trigger + atr * rules["chase_atr_multiplier"], trigger * (1 + rules["chase_pct"] / 100))
    recent_low = number(row, "直近安値", number(row, "損切り候補", close - atr))
    reversal_low = number(row, "反転足安値", recent_low)
    bb_lower = number(row, "BB下限", bb2)
    valid_stops = [x for x in (recent_low, reversal_low, bb_lower) if 0 < x < trigger]
    structural_stop = min(valid_stops) if valid_stops else close - atr * rules["stop_atr_multiplier"]
    entry_for_risk = trigger
    atr_stop = entry_for_risk - atr * rules["stop_atr_multiplier"]
    stop = min(structural_stop, atr_stop)
    if stop >= entry_for_risk:
        stop = entry_for_risk - atr * rules["stop_atr_multiplier"]
    risk = entry_for_risk - stop
    target = entry_for_risk + risk * rules["reward_risk"]
    return {"反転確認高値": confirmation, "逆指値発動価格": trigger,
            "買いゾーン下限": zone_low, "買いゾーン上限": zone_high,
            "追いかけ禁止価格": chase, "損切り価格": stop, "利確目標": target,
            "ATR14": atr, "基準RR": rules["reward_risk"]}


def select_order(row: dict, prices: dict, config: dict, method_stats: dict | None = None) -> tuple[str, str]:
    """Select A/B/C/D using ordered, deterministic safety rules."""
    rules = config["order_rules"]
    method_stats = method_stats or {}
    gap = number(row, "gap_pct", number(row, "ギャップ率", 0))
    volume = number(row, "出来高倍率")
    rsi = number(row, "RSI14", 50)
    surged = bool(str(row.get("除外理由", "")).strip()) or str(row.get("ランク")) == "除外"
    recent_rise = max(number(row, "前日比"), number(row, "直近3日騰落率"), number(row, "直近5日騰落率"))
    if surged or gap >= rules["gap_skip_pct"] or recent_rise >= rules["recent_rise_skip_pct"]:
        return "skip", "急騰・大幅ギャップを追わない"
    if volume < rules["minimum_order_volume_ratio"]:
        return "skip", "注文に必要な出来高不足"
    reversal = str(row.get("ローソク足パターン", "なし")) != "なし"
    confirmed = bool(row.get("反転確認済み", False)) or number(row, "現在値") > prices["反転確認高値"]
    market_pf = float(method_stats.get("market", {}).get("PF") or 0)
    if (confirmed and reversal and gap <= rules["market_max_gap_pct"] and rsi < rules["market_max_rsi"]
            and volume >= rules["market_min_volume_ratio"] and market_pf >= rules["market_min_pf"]):
        return "market", "反転・出来高・小幅ギャップ・寄り成り実績を確認"
    # Countertrend setups default to confirmation above the reversal high.
    if reversal and ("BB" in str(row.get("シグナル種別", "")) or rsi <= rules["countertrend_max_rsi"]):
        return "stop", "安値ではなく反転確認高値の上抜けを待つ"
    if number(row, "現在値") >= prices["買いゾーン下限"]:
        return "limit", "BB・MA25・支持線の買いゾーンまで押しを待つ"
    return "skip", "設定済み注文条件を満たさない"


def size_order(entry: float, stop: float, classification: str, config: dict, *,
               cash: float, positions: int, crash: bool = False) -> dict:
    p = config["portfolio"]
    capital, lot = float(p["initial_capital"]), int(p["lot_size"])
    scale = float(p["core_size_pct"] if classification == "主力" else p["small_size_pct"]) / 100
    if crash:
        scale = min(scale, float(p["crash_rebound_size_pct"]) / 100)
    risk = max(0, entry - stop)
    risk_budget = capital * float(p["max_risk_per_trade_pct"]) / 100 * scale
    reserve = capital * float(p["minimum_cash_ratio"]) / 100
    cash_budget = min(float(p["max_position_amount"]) * scale, max(0, cash - reserve))
    shares = min(math.floor(risk_budget / risk) if risk else 0,
                 math.floor(cash_budget / entry) if entry else 0)
    shares = math.floor(shares / lot) * lot if positions < int(p["max_positions"]) else 0
    return {"推奨株数": shares, "必要資金": round(shares * entry),
            "最大想定損失": round(shares * risk), "1株リスク": round(risk, 2),
            "購入後現金比率": round((cash - shares * entry) / capital * 100, 1)}
