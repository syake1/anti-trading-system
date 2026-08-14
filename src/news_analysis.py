"""Source-gated, rule-based news classification (works without an LLM)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
from src.utils import ROOT


def _matches(text, words): return any(w.lower() in text.lower() for w in words)


def analyze_news(source: str | Path | pd.DataFrame | None, config: dict) -> pd.DataFrame:
    cols = ["event_id", "published_at", "fetched_at", "source", "title", "url", "category", "severity",
            "direction", "scope", "sectors", "symbols", "persistence", "news_impact_score", "trusted", "emergency_risk"]
    if source is None: return pd.DataFrame(columns=cols)
    if isinstance(source, pd.DataFrame): raw = source.copy()
    else:
        path = Path(source)
        if not path.exists() or path.stat().st_size == 0: return pd.DataFrame(columns=cols)
        raw = pd.read_csv(path, dtype=str).fillna("")
    rules, trusted_sources = config["news_analysis"]["categories"], set(config["news_analysis"]["trusted_sources"])
    out, seen = [], set()
    for row in raw.to_dict("records"):
        title, origin, url = str(row.get("title", "")).strip(), str(row.get("source", "")).strip(), str(row.get("url", "")).strip()
        key = (title.casefold(), url)
        if not title or key in seen: continue
        seen.add(key)
        trusted = bool(origin and origin in trusted_sources and (url or row.get("identifier")))
        category, rule = "other", {"severity": 1, "direction": 0, "sectors": {}}
        for name, candidate in rules.items():
            if _matches(title, candidate["keywords"]): category, rule = name, candidate; break
        severity = int(row.get("severity") or rule.get("severity", 1)); direction = int(row.get("direction") or rule.get("direction", 0))
        # Untrusted events are auditable but their effective trading impact is always zero.
        impact = severity * direction if trusted else 0
        emergency = trusted and severity >= config["news_analysis"]["emergency_min_severity"] and category in config["news_analysis"]["emergency_categories"]
        event_id = hashlib.sha256(f"{title}|{url}".encode()).hexdigest()[:16]
        out.append({"event_id": event_id, "published_at": row.get("published_at", ""), "fetched_at": row.get("fetched_at", ""),
                    "source": origin, "title": title, "url": url or row.get("identifier", ""), "category": category,
                    "severity": severity, "direction": direction, "scope": row.get("scope", "market"),
                    "sectors": row.get("sectors", ",".join(rule.get("sectors", {}))), "symbols": row.get("symbols", ""),
                    "persistence": row.get("persistence", rule.get("persistence", "不明")),
                    "news_impact_score": impact, "trusted": trusted, "emergency_risk": emergency})
    return pd.DataFrame(out, columns=cols)


def save_news(events: pd.DataFrame, path: Path | None = None):
    path = path or ROOT / "data/news_events.csv"; path.parent.mkdir(parents=True, exist_ok=True); events.to_csv(path, index=False)


def news_impacts(events: pd.DataFrame, sector: str, symbol: str, config: dict) -> float:
    total = 0.0
    for event in events.to_dict("records"):
        if not event["trusted"]: continue
        rule = config["news_analysis"]["categories"].get(event["category"], {})
        if symbol and symbol in str(event["symbols"]).split(","): total += event["news_impact_score"]
        elif sector in rule.get("sectors", {}): total += float(rule["sectors"][sector]) * event["severity"]
        elif event["scope"] == "market": total += event["news_impact_score"] * .25
    return total
