"""Conservative fundamental assessment from auditable official disclosures.

No value is inferred or carried forward.  The input is deliberately a normalized
CSV so EDINET/TDnet/company IR (and licensed J-Quants exports) can be reviewed
before it affects an order proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import math
import pandas as pd

from src.utils import ROOT

FIELDS = ("revenue_yoy", "operating_profit_yoy", "ordinary_or_net_profit_yoy",
          "eps", "per", "pbr", "roe", "equity_ratio", "dividend_yield",
          "company_forecast", "latest_earnings_date", "revision", "important_disclosure")
REQUIRED = ("revenue_yoy", "operating_profit_yoy", "ordinary_or_net_profit_yoy",
            "eps", "per", "pbr", "roe", "equity_ratio", "dividend_yield",
            "company_forecast", "latest_earnings_date")
OFFICIAL_SOURCES = {"EDINET", "TDnet", "企業IR", "J-Quants", "JPX"}
AUDIT_COLUMNS = ["acquired_at", "code", "source", "source_reference", "status", "reason"]


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() != ""


def _num(value):
    try:
        return float(value) if _present(value) else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class FundamentalAssessment:
    data: dict
    score: int | None
    label: str
    sufficient: bool
    reason: str


def assess(row: dict, config: dict) -> FundamentalAssessment:
    """Score only explicitly supplied values; return insufficient rather than guessing."""
    fc = config.get("fundamentals", {})
    minimum = int(fc.get("minimum_required_fields", len(REQUIRED)))
    present = sum(_present(row.get(k)) for k in REQUIRED)
    source = str(row.get("fundamental_source", row.get("source", ""))).strip()
    reference = str(row.get("fundamental_source_reference", row.get("source_reference", ""))).strip()
    official = source in set(fc.get("official_sources", OFFICIAL_SOURCES)) and bool(reference)
    sufficient = official and present >= minimum
    if not sufficient:
        why = "公式取得元または参照先なし" if not official else f"必須項目 {present}/{minimum}"
        return FundamentalAssessment(row, None, "データ不足", False, why)

    revenue, operating = _num(row.get("revenue_yoy")), _num(row.get("operating_profit_yoy"))
    profit = _num(row.get("ordinary_or_net_profit_yoy"))
    roe, equity = _num(row.get("roe")), _num(row.get("equity_ratio"))
    per, pbr, dividend = _num(row.get("per")), _num(row.get("pbr")), _num(row.get("dividend_yield"))
    score = 0
    score += 1 if revenue is not None and revenue > 0 else 0
    score += 2 if operating is not None and operating > 0 else 0
    score += 1 if profit is not None and profit > 0 else 0
    score += 1 if _num(row.get("eps")) is not None and _num(row.get("eps")) > 0 else 0
    score += 1 if per is not None and 0 < per <= 25 and pbr is not None and 0 < pbr <= 3 else 0
    score += 1 if roe is not None and roe >= 8 else 0
    score += 1 if equity is not None and equity >= 30 else 0
    score += 1 if dividend is not None and dividend > 0 else 0
    forecast = str(row.get("company_forecast", ""))
    score += 1 if "増収増益" in forecast else 0
    revision = str(row.get("revision", "")); material = str(row.get("important_disclosure", ""))
    if "上方修正" in revision: score += 1
    if "下方修正" in revision: score -= 3
    if any(x in forecast + material for x in ("赤字転落", "債務超過", "不祥事", "重大事故")): score -= 4
    if "減収減益" in forecast: score -= 2
    score = max(0, min(10, score))
    label = "良好" if score >= 8 else "普通" if score >= 6 else "注意" if score >= 4 else "弱い"
    return FundamentalAssessment(row, score, label, True, "評価完了")


def load_fundamentals(path: Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data/fundamentals_input.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"code": str})
    if "code" in frame: frame["code"] = frame["code"].str.replace(r"\.0$", "", regex=True).str.zfill(4)
    return frame


def enrich_candidates(candidates: pd.DataFrame, config: dict, path: Path | None = None,
                      audit_path: Path | None = None) -> pd.DataFrame:
    """Left join reviewed fundamentals and audit every success and failure."""
    if candidates.empty: return candidates.copy()
    data = load_fundamentals(path)
    indexed = data.drop_duplicates("code", keep="last").set_index("code") if "code" in data else pd.DataFrame()
    output, audits = [], []
    acquired = datetime.now(timezone.utc).isoformat()
    for record in candidates.to_dict("records"):
        code = str(record.get("コード", "")).replace(".0", "").zfill(4)
        supplied = indexed.loc[code].to_dict() if not indexed.empty and code in indexed.index else {}
        record.update(supplied)
        record["fundamental_source"] = supplied.get("source", "")
        record["fundamental_source_reference"] = supplied.get("source_reference", "")
        result = assess(record, config)
        record["ファンダメンタルスコア"] = result.score
        record["ファンダメンタル評価"] = result.label
        record["ファンダメンタル十分"] = result.sufficient
        record["ファンダメンタル不足理由"] = result.reason
        output.append(record)
        audits.append({"acquired_at": supplied.get("acquired_at", acquired), "code": code,
                       "source": supplied.get("source", "未取得"),
                       "source_reference": supplied.get("source_reference", ""),
                       "status": "success" if result.sufficient else "failure", "reason": result.reason})
    audit_path = audit_path or ROOT / config.get("fundamentals", {}).get("audit_path", "data/fundamentals_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    exists = audit_path.exists() and audit_path.stat().st_size > 0
    with audit_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS); writer.writeheader() if not exists else None; writer.writerows(audits)
    return pd.DataFrame(output)
