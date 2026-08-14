"""Stocknote analysis with explicit, auditable observation precedence."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Callable

import pandas as pd

from src.indicators import bollinger, rsi


CODE = re.compile(r"^[0-9]{4}$")
ROOT = Path(__file__).resolve().parents[1]
LOCAL_HISTORY_PATH = ROOT / "data" / "signal_history.csv"
BB_VALUE = re.compile(r"^([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))σ$")
NUMERIC_TECHNICAL = {"現在値", "RSI14", "MA25", "MA75", "MA200", "出来高倍率", "ATR14",
                     "25日線乖離率", "損切り候補", "利確候補", "RR"}
TEXT_TECHNICAL = {"BB位置", "反転足", "ローソク足パターン", "シグナル種別"}
_FUNDAMENTALS = {
    "per": ("per", "PER"), "pbr": ("pbr", "PBR"),
    "financial_health": ("financial_health", "財務健全性"),
    "revenue_growth": ("revenue_growth", "売上成長", "売上高成長率"),
    "profit_growth": ("profit_growth", "利益成長", "利益成長率"),
}


def _load_history(code: str) -> pd.DataFrame:
    """Download daily observations for exactly one TSE security."""
    import yfinance as yf
    return yf.download(f"{code}.T", period="1y", interval="1d", auto_adjust=False,
                       progress=False, threads=False)


def load_local_observations(code: str, path: str | Path = LOCAL_HISTORY_PATH) -> dict:
    """Return the latest dated, valid saved signal row for ``code``."""
    try:
        frame = pd.read_csv(path, dtype={"コード": str}, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError):
        return {}
    if not {"コード", "シグナル日"} <= set(frame):
        return {}
    rows = frame[frame["コード"].str.zfill(4) == code].copy()
    rows["_date"] = pd.to_datetime(rows["シグナル日"], errors="coerce")
    rows = rows.dropna(subset=["_date"]).sort_values("_date", ascending=False)
    for _, row in rows.iterrows():
        values = _clean_technical(row.to_dict())
        if values:
            return values
    return {}


def _clean_technical(source: object) -> dict:
    if not isinstance(source, dict):
        return {}
    result = {}
    for key in NUMERIC_TECHNICAL:
        value = source.get(key)
        if value is None or pd.isna(value) or isinstance(value, bool):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[key] = value
    for key in TEXT_TECHNICAL:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    if "BB位置" in result and not BB_VALUE.fullmatch(result["BB位置"]):
        result.pop("BB位置")
    return result


def _close_series(history: pd.DataFrame) -> pd.Series:
    if not isinstance(history, pd.DataFrame) or "Close" not in history:
        raise ValueError("株価履歴にCloseがありません")
    close = history["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("銘柄を一意に特定できません")
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[close.map(lambda value: math.isfinite(float(value)) and value > 0)]
    if len(close) < 75:
        raise ValueError("指標計算に必要な75営業日分の株価がありません")
    return close


def _from_price_history(history: pd.DataFrame) -> dict:
    close = _close_series(history)
    bands = bollinger(close, 20, 2).iloc[-1]
    rsi_value = float(rsi(close, 14).iloc[-1])
    sigma = float(bands["bb_sigma"])
    if not math.isfinite(rsi_value) or not math.isfinite(sigma):
        return {}
    return {"現在値": float(close.iloc[-1]), "RSI14": rsi_value, "BB位置": f"{sigma:+.2f}σ",
            "MA25": float(close.tail(25).mean()), "MA75": float(close.tail(75).mean())}


def _available(source: object) -> dict:
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
                      technical_information: dict | None = None,
                      history_loader: Callable[[str], pd.DataFrame] | None = None,
                      local_history_loader: Callable[[str], dict] | None = None) -> dict:
    """Resolve request, saved observations, then Yahoo; never synthesize gaps."""
    if not isinstance(code, str) or not CODE.fullmatch(code):
        raise ValueError("code must be a four-digit string")
    observations = _clean_technical(technical_information)
    sources = {key: "request" for key in observations}

    def analyzable(values: dict) -> int:
        return sum(key in values for key in ("RSI14", "BB位置", "25日線乖離率", "出来高倍率"))

    # Two observed signals are sufficient. Lower-priority sources only fill a
    # genuinely insufficient request, so an external service is never mandatory.
    if analyzable(observations) < 2:
        local = _clean_technical((local_history_loader or load_local_observations)(code))
        for key, value in local.items():
            if key not in observations:
                observations[key], sources[key] = value, "local_history"
    if analyzable(observations) < 2:
        try:
            external = _from_price_history((history_loader or _load_history)(code))
        except Exception:
            external = {}
        for key, value in external.items():
            if key not in observations:
                observations[key], sources[key] = value, "yahoo_history"

    components = []
    if "RSI14" in observations:
        components.append(max(0.0, min(50.0, (50.0 - observations["RSI14"]) * 2.5)))
    bb_sigma = None
    if "BB位置" in observations:
        bb_sigma = float(BB_VALUE.fullmatch(observations["BB位置"]).group(1))
        components.append(max(0.0, min(50.0, -bb_sigma * 25.0)))
    if "25日線乖離率" in observations:
        components.append(max(0.0, min(25.0, -observations["25日線乖離率"] * 2.5)))
    if "出来高倍率" in observations:
        components.append(max(0.0, min(25.0, (observations["出来高倍率"] - 1.0) * 25.0)))
    if not components:
        result = {"assessment": "insufficient", "confidence": 0.0,
                  "summary": f"{code}: 分析可能なRSI・BB位置の実データがありません。"}
    else:
        score = round(min(100.0, sum(components)), 1)
        current, ma25, ma75 = (observations.get(key) for key in ("現在値", "MA25", "MA75"))
        trend = None
        if all(value is not None for value in (current, ma25, ma75)):
            trend = "上昇" if current > ma25 > ma75 else "下降" if current < ma25 < ma75 else "横ばい・混在"
        used = [key for key in ("RSI14", "BB位置", "25日線乖離率", "出来高倍率") if key in observations]
        source_text = "、".join(f"{key}={sources[key]}" for key in used)
        details = []
        if "損切り候補" in observations:
            details.append(f"観測済み損切り候補 {observations['損切り候補']:g}")
        if "利確候補" in observations:
            details.append(f"観測済み利確候補 {observations['利確候補']:g}")
        if "RR" in observations:
            details.append(f"観測済みRR {observations['RR']:g}")
        summary = f"{code}: 逆張りスコア{score:.1f}（{source_text}）。"
        if details:
            summary += " " + "、".join(details) + "。"
        result = {"assessment": _assessment(score), "confidence": min(1.0, round(0.25 * len(components), 2)),
                  "summary": summary, "contrarian_score": score,
                  "cautions": "明示された観測値のみを採点。候補価格は予測値・注文条件ではありません。"}
        if "RSI14" in observations:
            result["rsi"] = observations["RSI14"]
        if bb_sigma is not None:
            result["bb_position"] = observations["BB位置"]
        if trend is not None:
            result["trend"] = trend

    official, reference = _available(official_information), _available(reference_information)
    if official:
        result["official"] = official
    if reference:
        result["reference"] = reference
    return result
