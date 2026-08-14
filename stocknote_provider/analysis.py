"""Real stocknote analysis for one Japanese equity.

Only market observations and explicitly supplied fundamentals are used.  In
particular, absent fundamental values are not estimated from prices or copied
between official and reference source sections.
"""
from __future__ import annotations

import math
import re
from typing import Callable

import pandas as pd

from src.indicators import bollinger, rsi


CODE = re.compile(r"^[0-9]{4}$")
_FUNDAMENTALS = {
    "per": ("per", "PER"),
    "pbr": ("pbr", "PBR"),
    "financial_health": ("financial_health", "財務健全性"),
    "revenue_growth": ("revenue_growth", "売上成長", "売上高成長率"),
    "profit_growth": ("profit_growth", "利益成長", "利益成長率"),
}


def _load_history(code: str) -> pd.DataFrame:
    """Download daily observations for exactly one TSE security."""
    import yfinance as yf

    return yf.download(f"{code}.T", period="1y", interval="1d", auto_adjust=False,
                       progress=False, threads=False)


def _close_series(history: pd.DataFrame) -> pd.Series:
    if not isinstance(history, pd.DataFrame) or "Close" not in history:
        raise ValueError("株価履歴にCloseがありません")
    close = history["Close"]
    # yfinance can return a one-symbol MultiIndex frame depending on version.
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("銘柄を一意に特定できません")
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close.map(lambda value: math.isfinite(float(value)) and value > 0)]
    if len(close) < 75:
        raise ValueError("指標計算に必要な75営業日分の株価がありません")
    return close


def _available(source: object) -> dict:
    """Select known, actually present fundamentals without inventing defaults."""
    if not isinstance(source, dict):
        return {}
    selected = {}
    for output, aliases in _FUNDAMENTALS.items():
        value = next((source[key] for key in aliases if key in source and source[key] is not None), None)
        if value is None or value == "":
            continue
        if output in {"per", "pbr"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            value = float(value)
        elif not isinstance(value, str):
            continue
        selected[output] = value
    return selected


def _assessment(score: float) -> str:
    if score >= 65:
        return "positive"
    if score >= 40:
        return "neutral"
    return "negative"


def analyze_candidate(*, code: str, official_information: dict | None = None,
                      reference_information: dict | None = None,
                      history_loader: Callable[[str], pd.DataFrame] | None = None) -> dict:
    """Analyze one four-digit code and return a Phase 2 compatible result.

    Prices are observational levels: lower Bollinger band (buy), 20-day mean
    (expected sell), and upper band (final target).  They are not forecasts.
    """
    if not isinstance(code, str) or not CODE.fullmatch(code):
        raise ValueError("code must be a four-digit string")
    close = _close_series((history_loader or _load_history)(code))
    rsi_value = float(rsi(close, 14).iloc[-1])
    bands = bollinger(close, 20, 2).iloc[-1]
    if not all(math.isfinite(float(bands[key])) for key in ("bb_mid", "bb_lower", "bb_upper", "bb_sigma")):
        raise ValueError("ボリンジャーバンドを計算できません")

    sigma = float(bands["bb_sigma"])
    # Equal-weight oversold and lower-band proximity components, each 0..50.
    rsi_component = max(0.0, min(50.0, (50.0 - rsi_value) * 2.5))
    bb_component = max(0.0, min(50.0, -sigma * 25.0))
    score = round(rsi_component + bb_component, 1)
    ma25, ma75 = float(close.tail(25).mean()), float(close.tail(75).mean())
    trend = "上昇" if close.iloc[-1] > ma25 > ma75 else "下降" if close.iloc[-1] < ma25 < ma75 else "横ばい・混在"
    assessment = _assessment(score)
    cautions = "終値ベースのテクニカル分析。価格水準は予測ではなく、出来高・決算・窓開けを未考慮。"
    result = {
        "assessment": assessment,
        "confidence": 0.7,
        "summary": f"{code}: 逆張りスコア{score:.1f}、RSI {rsi_value:.1f}、BB {sigma:+.2f}σ、トレンドは{trend}。",
        "contrarian_score": score,
        "rsi": round(rsi_value, 2),
        "bb_position": f"{sigma:+.2f}σ",
        "trend": trend,
        "recommended_buy_price": round(float(bands["bb_lower"]), 2),
        "expected_sell_price": round(float(bands["bb_mid"]), 2),
        "final_target_price": round(float(bands["bb_upper"]), 2),
        "cautions": cautions,
    }
    official = _available(official_information)
    reference = _available(reference_information)
    if official:
        result["official"] = official
    if reference:
        result["reference"] = reference
    return result
