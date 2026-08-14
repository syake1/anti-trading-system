"""Japanese government-bond curve analysis.

Rates are optional observations: no interpolation, proxy, or invented value is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.utils import ROOT, now_tokyo


@dataclass(frozen=True)
class JapanRates:
    timestamp: str
    jp_2y_yield: float | None = None
    jp_10y_yield: float | None = None
    jp_30y_yield: float | None = None
    jp_10y_previous: float | None = None
    jp_10y_5d: float | None = None
    jp_10y_20d: float | None = None
    jp_10y_change_bp: float | None = None
    jp_10y_change_5d_bp: float | None = None
    jp_10y_change_20d_bp: float | None = None
    jp_2y_change_bp: float | None = None
    jp_2y_change_5d_bp: float | None = None
    jp_10y_2y_spread: float | None = None
    jp_10y_2y_spread_change_bp: float | None = None
    jp_10y_2y_spread_change_5d_bp: float | None = None
    jp_rate_regime: str = "取得不可"
    boj_tightening_risk: bool = False
    boj_news_observed: bool = False
    source: str = ""
    source_timestamp: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _value(series: pd.Series, offset: int) -> float | None:
    return float(series.iloc[-offset]) if len(series) >= offset else None


def _bp(current: float | None, previous: float | None) -> float | None:
    return round((current - previous) * 100, 4) if current is not None and previous is not None else None


def classify_rate(change_bp: float | None, change_5d_bp: float | None,
                  change_20d_bp: float | None, config: dict) -> str:
    if change_bp is None:
        return "取得不可"
    r = config["japan_rates"]["regime_thresholds_bp"]
    # Multi-day moves are confirmation only; they cannot reverse today's direction.
    momentum = (change_5d_bp or 0) * float(r.get("five_day_weight", .15)) + (change_20d_bp or 0) * float(r.get("twenty_day_weight", .05))
    effective = change_bp + momentum
    if change_bp >= r["surge"] or effective >= r["surge"]: return "急上昇"
    if change_bp <= r["plunge"] or effective <= r["plunge"]: return "急低下"
    if effective >= r["rise"]: return "上昇"
    if effective <= r["fall"]: return "低下"
    return "横ばい"


def analyze_japan_rates(histories: Mapping[str, pd.Series] | None, config: dict, *,
                        observed_at: str | None = None, boj_news=False,
                        usd_jpy_change_pct: float | None = None) -> JapanRates:
    observed_at = observed_at or now_tokyo().isoformat(timespec="seconds")
    clean = {k: pd.to_numeric(v, errors="coerce").dropna() for k, v in (histories or {}).items()}
    y10, y2, y30 = clean.get("JP10Y", pd.Series(dtype=float)), clean.get("JP2Y", pd.Series(dtype=float)), clean.get("JP30Y", pd.Series(dtype=float))
    c10, p10, d5, d20 = _value(y10, 1), _value(y10, 2), _value(y10, 6), _value(y10, 21)
    c2, p2, p2_5 = _value(y2, 1), _value(y2, 2), _value(y2, 6)
    ch, ch5, ch20 = _bp(c10, p10), _bp(c10, d5), _bp(c10, d20)
    ch2, ch2_5 = _bp(c2, p2), _bp(c2, p2_5)
    spread = round(c10 - c2, 4) if c10 is not None and c2 is not None else None
    spread_prev = (p10 - p2) if p10 is not None and p2 is not None else None
    spread_5 = (d5 - p2_5) if d5 is not None and p2_5 is not None else None
    regime = classify_rate(ch, ch5, ch20, config)
    risk = bool(boj_news and ch2 is not None and ch2 >= config["japan_rates"]["boj_risk_2y_bp"] and
                regime in ("上昇", "急上昇") and usd_jpy_change_pct is not None and
                usd_jpy_change_pct <= config["japan_rates"]["boj_risk_usdjpy_pct"])
    return JapanRates(observed_at, c2, c10, _value(y30, 1), p10, d5, d20, ch, ch5, ch20,
                      ch2, ch2_5, spread, _bp(spread, spread_prev), _bp(spread, spread_5),
                      regime, risk, bool(boj_news))


def fetch_japan_rates(config: dict, *, boj_news=False, usd_jpy_change_pct=None) -> JapanRates:
    """Best-effort Yahoo Finance fetch; empty result keeps the meeting operational."""
    observed = now_tokyo().isoformat(timespec="seconds")
    histories, succeeded = {}, []
    try:
        import yfinance as yf
        for tenor, ticker in config["japan_rates"]["tickers"].items():
            try:
                frame = yf.download(ticker, period="3mo", interval="1d", progress=False,
                                    auto_adjust=False, timeout=10)
                close = frame.get("Close", pd.Series(dtype=float))
                if hasattr(close, "columns"): close = close.iloc[:, 0]
                close = pd.to_numeric(close, errors="coerce").dropna()
                if not close.empty: histories[tenor] = close; succeeded.append(ticker)
            except Exception:
                continue
    except ImportError:
        pass
    result = analyze_japan_rates(histories, config, observed_at=observed, boj_news=boj_news,
                                 usd_jpy_change_pct=usd_jpy_change_pct)
    if not succeeded: return result
    values = result.as_dict(); values["source"] = "Yahoo Finance (JGB yield symbols: " + ", ".join(succeeded) + ")"
    values["source_timestamp"] = observed
    return JapanRates(**values)


def save_japan_rates(rates: JapanRates, path: Path | None = None) -> None:
    path = path or ROOT / "data/japan_rates.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    row = rates.as_dict()
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def rate_sector_impacts(rates: JapanRates, *, risk_off=False, credit_risk=False) -> dict[str, int]:
    level = {"急上昇": 2, "上昇": 1, "横ばい": 0, "低下": -1, "急低下": -2}.get(rates.jp_rate_regime, 0)
    financial = 0 if (level > 0 and (risk_off or credit_risk)) else level
    return {"銀行": financial, "地方銀行": financial, "メガバンク": financial,
            "保険": financial, "生命保険": financial, "損害保険": financial,
            "不動産": -level, "REIT": -level, "J-REIT": -level, "住宅": -level,
            "高レバレッジ": -level, "高PERグロース": -level, "グロース": -level,
            "高配当": 0}


def stock_rate_impact(row: dict, rates: JapanRates, us10y_change_bp=None,
                      usd_jpy_change_pct=None, *, risk_off=False, credit_risk=False) -> tuple[int, str]:
    impacts = rate_sector_impacts(rates, risk_off=risk_off, credit_risk=credit_risk)
    sector = str(row.get("業種", "")); tags = str(row.get("属性", ""))
    keys = [k for k in impacts if k in sector or k in tags]
    score = impacts.get(sector, 0) if sector in impacts else (min((impacts[k] for k in keys), default=0) if keys else 0)
    reasons = [f"日本10年金利{rates.jp_rate_regime}のため{sector or '銘柄属性'}を{'加点' if score > 0 else '減点' if score < 0 else '中立'}"]
    if score < 0 and ("グロース" in sector + tags) and us10y_change_bp is not None and us10y_change_bp > 0:
        score -= 1; reasons.append("日米金利が同時上昇")
    if rates.jp_10y_change_bp is not None and rates.jp_10y_change_bp > 0 and usd_jpy_change_pct is not None and usd_jpy_change_pct < 0 and ("輸出" in sector + tags or "自動車" in sector):
        score -= 1; reasons.append("日本金利上昇と円高で輸出採算を警戒")
    # High dividend is not mechanically penalized: only weak quality flags permit -1.
    if "高配当" in tags and sector not in ("銀行", "保険"):
        healthy = all(float(row.get(k, 0) or 0) > 0 for k in ("利益成長率", "配当利回り")) and float(row.get("配当性向", 101) or 101) <= 70
        if rates.jp_10y_change_bp and rates.jp_10y_change_bp > 0 and not healthy:
            score -= 1; reasons.append("配当品質が未確認で国債との相対魅力低下を補助評価")
    return max(-2, min(2, score)), "、".join(reasons)
