"""Deterministic market-regime analysis using only observed market data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.utils import ROOT, now_tokyo


@dataclass(frozen=True)
class MarketEnvironment:
    observed_at: str
    indicators: tuple[dict, ...]
    total_score: float
    regime: str
    capital_ratio: float

    def as_dict(self):
        return {"observed_at": self.observed_at, "indicators": list(self.indicators),
                "total_score": self.total_score, "regime": self.regime,
                "capital_ratio": self.capital_ratio}


def _band_score(value: float, thresholds: list[float], reverse=False) -> int:
    """Map a change to -2..2. thresholds are [strong down, down, up, strong up]."""
    a, b, c, d = thresholds
    score = -2 if value <= a else -1 if value < b else 0 if value <= c else 1 if value < d else 2
    return -score if reverse else score


def analyze_market(rows: list[dict] | pd.DataFrame | None, config: dict, observed_at: str | None = None) -> MarketEnvironment:
    observed_at = observed_at or now_tokyo().isoformat(timespec="seconds")
    data = [] if rows is None else (rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows)
    rules = config["market_environment"]
    indicators = []
    for raw in data:
        name = str(raw.get("indicator", "")).strip()
        current, previous = raw.get("current"), raw.get("previous_close")
        if not name or pd.isna(current) or pd.isna(previous) or float(previous) == 0:
            continue
        change = float(current) - float(previous)
        pct = float(raw.get("change_pct", change / float(previous) * 100))
        rule = rules["indicator_rules"].get(name)
        if not rule:  # Unknown observed indicators are retained but never guessed/scored.
            score = 0
        else:
            measure = change * 100 if rule.get("unit") == "bp" else pct
            score = _band_score(measure, rule["thresholds"], rule.get("reverse", False))
        indicators.append({"indicator": name, "current": float(current), "previous_close": float(previous),
                           "change": change, "change_pct": pct, "short_change_pct": raw.get("short_change_pct"),
                           "change_bp": change * 100 if name == "US10Y" else None, "score": score})
    total = sum(x["score"] * rules["weights"].get(x["indicator"], 1) for x in indicators)
    cut = rules["regime_thresholds"]
    regime = ("強いリスクオン" if total >= cut["strong_risk_on"] else "リスクオン" if total >= cut["risk_on"]
              else "強いリスクオフ" if total <= cut["strong_risk_off"] else "警戒" if total <= cut["caution"] else "中立")
    ratio = float(rules["capital_ratios"][regime])
    return MarketEnvironment(observed_at, tuple(indicators), total, regime, ratio)


def fetch_market_data(config: dict) -> list[dict]:
    """Best-effort daily download. Failures/missing quotes are omitted, never imputed."""
    try:
        import yfinance as yf
    except ImportError:
        return []
    result = []
    for name, ticker in config["market_environment"]["tickers"].items():
        try:
            frame = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False, timeout=10)
            close = frame["Close"].dropna()
            if hasattr(close, "columns"): close = close.iloc[:, 0]
            if len(close) < 2: continue
            previous, current = float(close.iloc[-2]), float(close.iloc[-1])
            short = (current / float(close.iloc[-4]) - 1) * 100 if len(close) >= 4 else None
            result.append({"indicator": name, "current": current, "previous_close": previous,
                           "change_pct": (current / previous - 1) * 100, "short_change_pct": short})
        except Exception:
            continue
    return result


def save_market_environment(env: MarketEnvironment, path: Path | None = None) -> None:
    path = path or ROOT / "data/market_environment.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"date": env.observed_at[:10], "time": env.observed_at[11:19], **x,
             "market_regime": env.regime, "total_score": env.total_score} for x in env.indicators]
    columns = ["date", "time", "indicator", "current", "previous_close", "change", "change_pct",
               "short_change_pct", "change_bp", "score", "market_regime", "total_score"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
