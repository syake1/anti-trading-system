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
AUDIT_COLUMNS = ["acquired_at", "code", "source", "document_id", "source_reference", "status", "reason"]


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() != ""


def _num(value):
    try:
        return float(value) if _present(value) else None
    except (TypeError, ValueError):
        return None


def _verified(value) -> bool:
    """Accept only an explicit verification flag, never infer comparability."""
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


@dataclass(frozen=True)
class FundamentalAssessment:
    data: dict
    score: int | None
    label: str
    sufficient: bool
    reason: str
    score_reasons: tuple[str, ...] = ()


def derive_official_metrics(row: dict) -> dict:
    """Derive comparisons only from pairs of explicitly supplied official facts."""
    result = dict(row)
    comparable = _verified(row.get("comparison_basis_verified"))
    current, prior = _num(row.get("net_profit")), _num(row.get("net_profit_prior"))
    if not comparable or current is None or prior is None:
        result["profit_transition"] = "評価不能"
    elif prior < 0 <= current:
        result["profit_transition"] = "黒字転換"
    elif prior >= 0 > current:
        result["profit_transition"] = "赤字転落"
    elif prior < 0 and current < 0:
        result["profit_transition"] = "赤字継続"
    else:
        result["profit_transition"] = "黒字継続"

    revenue, revenue_prior = _num(row.get("revenue")), _num(row.get("revenue_prior"))
    operating, operating_prior = _num(row.get("operating_profit")), _num(row.get("operating_profit_prior"))
    if not comparable or None in (revenue, revenue_prior, operating, operating_prior):
        result["growth_quadrant"] = "評価不能"
    else:
        result["growth_quadrant"] = ("増収" if revenue >= revenue_prior else "減収") + ("増益" if operating >= operating_prior else "減益")

    eps, dividend = _num(row.get("eps")), _num(row.get("dividend"))
    payout_comparable = _verified(row.get("payout_basis_verified"))
    result["payout_ratio"] = dividend / eps * 100 if payout_comparable and eps not in (None, 0) and dividend is not None else None
    dividend_prior = _num(row.get("dividend_prior"))
    dividend_comparable = _verified(row.get("dividend_comparison_verified"))
    if not dividend_comparable or dividend is None or dividend_prior is None:
        result["dividend_change"] = "評価不能"
    elif dividend == 0 and dividend_prior > 0:
        result["dividend_change"] = "無配転落"
    elif dividend > dividend_prior:
        result["dividend_change"] = "増配"
    elif dividend < dividend_prior:
        result["dividend_change"] = "減配"
    else:
        result["dividend_change"] = "維持"
    return result


def assess(row: dict, config: dict) -> FundamentalAssessment:
    """Score only explicitly supplied values; return insufficient rather than guessing."""
    original = row
    row = derive_official_metrics(row)
    fc = config.get("fundamentals", {})
    minimum = int(fc.get("minimum_required_fields", len(REQUIRED)))
    present = sum(_present(row.get(k)) for k in REQUIRED)
    source = str(row.get("fundamental_source", row.get("source", ""))).strip()
    reference = str(row.get("fundamental_source_reference", row.get("source_reference", ""))).strip()
    official = source in set(fc.get("official_sources", OFFICIAL_SOURCES)) and bool(reference)
    if not official:
        # Never present classifications calculated from an unaudited input as
        # official analysis, even if the caller supplied plausible numbers.
        row = {**original, "profit_transition": "評価不能", "growth_quadrant": "評価不能",
               "payout_ratio": None, "dividend_change": "評価不能"}
    sufficient = official and present >= minimum
    if not sufficient:
        why = "公式取得元または参照先なし" if not official else f"必須項目 {present}/{minimum}"
        return FundamentalAssessment(row, None, "データ不足", False, why)

    revenue, operating = _num(row.get("revenue_yoy")), _num(row.get("operating_profit_yoy"))
    profit = _num(row.get("ordinary_or_net_profit_yoy"))
    roe, equity = _num(row.get("roe")), _num(row.get("equity_ratio"))
    per, pbr, dividend = _num(row.get("per")), _num(row.get("pbr")), _num(row.get("dividend_yield"))
    score, reasons = 0, []
    def apply(points: int, reason: str):
        nonlocal score
        score += points; reasons.append(f"{points:+d} {reason}")
    if revenue is not None and revenue > 0: apply(1, "売上高前年比プラス")
    if operating is not None and operating > 0: apply(2, "営業利益前年比プラス")
    transition = row.get("profit_transition")
    if transition == "黒字転換": apply(1, "黒字転換（前年比率より優先）")
    elif transition == "赤字転落": apply(-4, "赤字転落（前年比率より優先）")
    elif transition in ("評価不能", "黒字継続") and profit is not None and profit > 0: apply(1, "利益前年比プラス")
    if _num(row.get("eps")) is not None and _num(row.get("eps")) > 0: apply(1, "EPSプラス")
    if per is not None and 0 < per <= 25 and pbr is not None and 0 < pbr <= 3: apply(1, "PER・PBR基準内")
    if roe is not None and roe >= 8: apply(1, "ROE 8%以上")
    if equity is not None and equity >= 30: apply(1, "自己資本比率30%以上")
    if dividend is not None and dividend > 0: apply(1, "配当利回りプラス")
    forecast = str(row.get("company_forecast", ""))
    if "増収増益" in forecast: apply(1, "会社予想が増収増益")
    revision = str(row.get("revision", "")); material = str(row.get("important_disclosure", ""))
    if "上方修正" in revision: apply(1, "上方修正")
    if "下方修正" in revision: apply(-3, "下方修正")
    if any(x in forecast + material for x in ("赤字転落", "債務超過", "不祥事", "重大事故")): apply(-4, "重大な悪材料")
    if "減収減益" in forecast: apply(-2, "会社予想が減収減益")
    raw_score = score
    score = max(0, min(10, score))
    if score != raw_score: reasons.append(f"得点範囲補正 {raw_score}→{score}")
    label = "良好" if score >= 8 else "普通" if score >= 6 else "注意" if score >= 4 else "弱い"
    return FundamentalAssessment(row, score, label, True, "評価完了", tuple(reasons))


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
    # Automated official data takes precedence when the EDINET secret is present;
    # failures are isolated inside the adapter and the reviewed CSV remains usable.
    if path is None:
        from src.fundamental_sources import acquire
        automated = acquire(candidates, config, audit_path=audit_path)
        if not automated.empty:
            data = pd.concat([data, automated], ignore_index=True)
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
        record.update(result.data)
        record["ファンダメンタルスコア"] = result.score
        record["ファンダメンタル評価"] = result.label
        record["ファンダメンタル十分"] = result.sufficient
        record["ファンダメンタル不足理由"] = result.reason
        record["ファンダメンタル加減点理由"] = " / ".join(result.score_reasons)
        output.append(record)
        audits.append({"acquired_at": supplied.get("acquired_at", acquired), "code": code,
                       "source": supplied.get("source", "未取得"),
                       "document_id": supplied.get("document_id", ""),
                       "source_reference": supplied.get("source_reference", ""),
                       "status": "success" if result.sufficient else "failure", "reason": result.reason})
    audit_path = audit_path or ROOT / config.get("fundamentals", {}).get("audit_path", "data/fundamentals_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    exists = audit_path.exists() and audit_path.stat().st_size > 0
    with audit_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS); writer.writeheader() if not exists else None; writer.writerows(audits)
    return pd.DataFrame(output)
