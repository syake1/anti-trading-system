"""Deterministic, bank-specific sector and entry timing evaluation.

The evaluator deliberately keeps the yield environment (stage 1) separate from
the individual share timing (stage 2).  A favourable yield curve is therefore
never sufficient, by itself, to turn a bank share into a buy candidate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping

from src.jgb_yields import JGBAnalysis


REVERSAL_PATTERNS = ("強気包み足", "陽線包み", "ハンマー", "長い下ヒゲ", "下ヒゲ", "前日安値割れ反転", "下落後の陽線", "陰線→陽線")


@dataclass(frozen=True)
class BankSectorEvaluation:
    """Auditable result of both evaluation stages."""

    rate_regime: str
    rate_wind: str
    rate_confidence: str
    rate_metrics: dict[str, Any]
    classification: str
    principle: str
    morning_recheck: bool
    reasons: tuple[str, ...]
    missing_data: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float | None:
    """Read scanner numbers, including strings such as ``+1.25σ`` and ``3%``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _yield_rows(jgb: JGBAnalysis | Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = jgb.tenors if isinstance(jgb, JGBAnalysis) else jgb.get("tenors", ())
    return {str(row.get("tenor")): row for row in raw}


def _rate_stage(jgb: JGBAnalysis | Mapping[str, Any] | None, rules: Mapping[str, Any]):
    if jgb is None:
        return "評価不能", "評価不能", "なし", {}, ["金利データ"]
    rows = _yield_rows(jgb)
    metrics: dict[str, Any] = {}
    missing: list[str] = []
    for tenor in ("2Y", "10Y", "30Y"):
        row = rows.get(tenor, {})
        for field in ("yield_pct", "change_1d_bp", "change_5d_bp"):
            value = _number(row.get(field))
            metrics[f"{tenor}_{field}"] = value
            if value is None:
                missing.append(f"{tenor}_{field}")

    spread = _number(jgb.spread_10y_2y_bp if isinstance(jgb, JGBAnalysis) else jgb.get("spread_10y_2y_bp"))
    metrics["spread_10y_2y_bp"] = spread
    if spread is None:
        missing.append("spread_10y_2y_bp")
    for horizon in ("1d", "5d"):
        long_move, short_move = metrics.get(f"10Y_change_{horizon}_bp"), metrics.get(f"2Y_change_{horizon}_bp")
        metrics[f"spread_change_{horizon}_bp"] = None if long_move is None or short_move is None else round(long_move - short_move, 4)
    if missing:
        return "評価不能", "評価不能", "なし", metrics, missing

    move = metrics["10Y_change_5d_bp"]
    curve = metrics["spread_change_5d_bp"]
    move_min = float(rules["yield_change_5d_bp"])
    curve_min = float(rules["spread_change_5d_bp"])
    if move >= move_min and curve >= curve_min:
        regime, wind, confidence = "上昇・スティープ化", "追い風候補", "高"
    elif metrics["2Y_change_5d_bp"] >= move_min and curve <= -curve_min:
        regime, wind, confidence = "上昇・フラット化", "追い風候補", "中"
    elif move <= -move_min and curve >= curve_min:
        regime, wind, confidence = "低下・スティープ化", "逆風", "中"
    elif move <= -move_min and curve <= -curve_min:
        regime, wind, confidence = "低下・フラット化", "逆風", "高"
    else:
        regime, wind, confidence = "方向不明", "中立", "低"

    rapid_drop = move <= -float(rules["rapid_yield_drop_5d_bp"])
    rapid_inversion = (curve <= -float(rules["rapid_spread_contraction_5d_bp"]) or
                       spread <= float(rules["inverted_spread_bp"]))
    metrics.update({"rapid_yield_drop": rapid_drop, "rapid_inversion": rapid_inversion})
    if rapid_drop or rapid_inversion:
        wind, confidence = "新規買い抑制", "高"
    return regime, wind, confidence, metrics, []


def evaluate_bank_sector(
    jgb: JGBAnalysis | Mapping[str, Any] | None,
    bank: Mapping[str, Any],
    config: Mapping[str, Any],
) -> BankSectorEvaluation:
    """Evaluate a bank share from observed JGB data and one scanner result row.

    All policy thresholds come from ``config['sector_strategy']['bank']``.
    Missing rates go to morning review rather than being silently treated as
    neutral; missing required technical values likewise produces 評価不能.
    """
    rules = config["sector_strategy"]["bank"]
    regime, wind, confidence, metrics, missing = _rate_stage(jgb, rules)
    principle = "金利上昇基調でも高値追いせず、反転確認後の押し目を拾う"
    reasons = [f"金利局面={regime}（{wind}、確信度={confidence}）"]
    if missing:
        return BankSectorEvaluation(regime, wind, confidence, metrics, "評価不能", principle, True,
                                    tuple(reasons + ["必須金利データ欠損のため朝会で再確認"]), tuple(missing))

    values = {
        "終値": _number(bank.get("現在値", bank.get("終値"))),
        "MA25": _number(bank.get("MA25")),
        "25日線乖離率": _number(bank.get("25日線乖離率")),
        "RSI14": _number(bank.get("RSI14")),
        "BB位置": _number(bank.get("BB位置")),
        "出来高倍率": _number(bank.get("出来高倍率")),
        "当日騰落率": _number(bank.get("当日騰落率", bank.get("前日比"))),
        "3日騰落率": _number(bank.get("直近3日騰落率", bank.get("3日騰落率"))),
        "5日騰落率": _number(bank.get("直近5日騰落率", bank.get("5日騰落率"))),
    }
    # Deviation can be calculated, but is never guessed, when price and MA25 exist.
    if values["25日線乖離率"] is None and values["終値"] is not None and values["MA25"] not in (None, 0):
        values["25日線乖離率"] = (values["終値"] / values["MA25"] - 1) * 100
    required = ("終値", "MA25", "25日線乖離率", "RSI14", "BB位置", "出来高倍率", "当日騰落率", "3日騰落率", "5日騰落率")
    technical_missing = [name for name in required if values[name] is None]
    if technical_missing:
        return BankSectorEvaluation(regime, wind, confidence, metrics, "評価不能", principle, True,
                                    tuple(reasons + ["必須テクニカル値が不足"]), tuple(technical_missing))

    deviation, rsi, bb, volume = (values["25日線乖離率"], values["RSI14"], values["BB位置"], values["出来高倍率"])
    pattern = str(bank.get("ローソク足パターン", ""))
    reversal = any(label in pattern for label in REVERSAL_PATTERNS)
    excluded = bool(str(bank.get("除外理由", "")).strip()) or str(bank.get("ランク", "")) == "除外"
    overheated = (excluded or deviation >= float(rules["chase_ma25_deviation_pct"]) or
                  rsi >= float(rules["rsi_overheated"]) or bb >= float(rules["bb_overextended_sigma"]))
    pullback = float(rules["pullback_ma25_deviation_min_pct"]) <= deviation <= float(rules["pullback_ma25_deviation_max_pct"])
    trend_maintained = deviation >= float(rules["trend_ma25_deviation_min_pct"])
    rate_tailwind = wind == "追い風候補"
    volume_confirmed = volume >= float(rules["volume_confirmation_ratio"])
    volume_surge = volume >= float(rules["movement_volume_ratio"])

    if overheated:
        classification = "追いかけ禁止"
        reasons.append("急騰除外またはテクニカル過熱")
    elif wind in ("逆風", "新規買い抑制") and not trend_maintained and not reversal:
        classification = "回避"
        reasons.append("金利逆風に加えて25日線割れ・反転未確認")
    elif volume_surge and reversal:
        classification = "動意候補"
        reasons.append("出来高増加を伴う初動反転（急騰除外未到達）")
    elif (rate_tailwind and trend_maintained and pullback and
          rsi < float(rules["rsi_overheated"]) and reversal and volume_confirmed):
        classification = "押し目候補"
        reasons.append("金利追い風、銀行株トレンド、押し目、非過熱、反転足を確認")
    elif rate_tailwind:
        classification = "押し目監視"
        reasons.append("金利追い風だけでは昇格せず、反転足または出来高の完成待ち")
    else:
        classification = "回避"
        reasons.append("金利追い風とエントリー条件が揃っていない")
    morning_recheck = classification in ("押し目監視", "評価不能") or (rate_tailwind and not reversal)
    metrics["technical"] = {**values, "reversal_pattern": reversal, "volume_confirmed": volume_confirmed,
                            "trend_maintained": trend_maintained, "surge_excluded": excluded}
    return BankSectorEvaluation(regime, wind, confidence, metrics, classification, principle,
                                morning_recheck, tuple(reasons), ())
