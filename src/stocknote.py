"""Phase 0/1 file contract for the optional stocknote shadow analyst.

Stocknote is deliberately advisory: this module only adds namespaced columns and
writes a comparison report.  It never changes fundamentals, decisions or orders.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import uuid
import math

import pandas as pd

SCHEMA_VERSION = "1.0"
CODE = re.compile(r"^[0-9]{4}$")
STOCKNOTE_FIELDS = {
    "assessment": "stocknote_評価", "confidence": "stocknote_信頼度", "summary": "stocknote_要約",
    "contrarian_score": "stocknote_逆張りスコア", "rsi": "stocknote_RSI",
    "bb_position": "stocknote_BB位置", "trend": "stocknote_トレンド", "per": "stocknote_PER",
    "pbr": "stocknote_PBR", "financial_health": "stocknote_財務健全性",
    "revenue_growth": "stocknote_売上成長", "profit_growth": "stocknote_利益成長",
    "recommended_buy_price": "stocknote_推奨買い価格", "expected_sell_price": "stocknote_予想売り価格",
    "final_target_price": "stocknote_最終目標価格", "cautions": "stocknote_注意点",
}
STOCKNOTE_COLUMNS = list(STOCKNOTE_FIELDS.values())
BASE_FIELDS = {"code", "assessment", "confidence", "summary"}
TEXT_DETAIL_FIELDS = {"bb_position", "trend", "financial_health", "revenue_growth", "profit_growth", "cautions"}
NUMBER_DETAIL_FIELDS = {"contrarian_score", "rsi", "per", "pbr", "recommended_buy_price",
                        "expected_sell_price", "final_target_price"}


class StocknoteContractError(ValueError):
    """A stocknote file is unsafe to consume."""


def _finite_json(path: Path) -> dict:
    def reject(value: str):
        raise StocknoteContractError(f"non-finite number: {value}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except StocknoteContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StocknoteContractError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StocknoteContractError("root must be an object")
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise StocknoteContractError("generated_at must be an RFC 3339 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StocknoteContractError("invalid generated_at") from exc
    if parsed.tzinfo is None:
        raise StocknoteContractError("generated_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_response(payload: dict, run_id: str, candidate_codes: set[str], *,
                      now: datetime | None = None, max_age_hours: float = 24) -> list[dict]:
    if set(payload) != {"schema_version", "run_id", "generated_at", "analyses"}:
        raise StocknoteContractError("invalid response fields")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise StocknoteContractError("unsupported schema_version")
    if payload.get("run_id") != run_id:
        raise StocknoteContractError("run_id mismatch")
    generated = _timestamp(payload.get("generated_at"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if generated < current - timedelta(hours=max_age_hours) or generated > current + timedelta(minutes=5):
        raise StocknoteContractError("response is expired or generated in the future")
    analyses = payload.get("analyses")
    if not isinstance(analyses, list):
        raise StocknoteContractError("analyses must be an array")
    seen: set[str] = set()
    for item in analyses:
        if not isinstance(item, dict) or not BASE_FIELDS.issubset(item) or not set(item) <= set(STOCKNOTE_FIELDS) | {"code"}:
            raise StocknoteContractError("invalid analysis fields")
        code = item["code"]
        if not isinstance(code, str) or not CODE.fullmatch(code) or code not in candidate_codes:
            raise StocknoteContractError("analysis code must be a requested four-digit code")
        if code in seen:
            raise StocknoteContractError("duplicate analysis code")
        seen.add(code)
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise StocknoteContractError("confidence must be between 0 and 1")
        if item["assessment"] not in {"positive", "neutral", "negative", "insufficient"}:
            raise StocknoteContractError("invalid assessment")
        if not isinstance(item["summary"], str) or len(item["summary"]) > 2000:
            raise StocknoteContractError("invalid summary")
        if any(field in item and not isinstance(item[field], str) for field in TEXT_DETAIL_FIELDS):
            raise StocknoteContractError("invalid detail text")
        for field in NUMBER_DETAIL_FIELDS:
            value = item.get(field)
            if field in item and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise StocknoteContractError("invalid detail number")
    return analyses


def export_request(result: pd.DataFrame, directory: Path, *, run_id: str | None = None,
                   generated_at: datetime | None = None) -> tuple[str, Path]:
    run_id = run_id or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}", run_id):
        raise StocknoteContractError("invalid run_id")
    codes = result.get("コード", pd.Series(dtype=str)).astype(str).tolist()
    if any(not CODE.fullmatch(code) for code in codes) or len(codes) != len(set(codes)):
        raise StocknoteContractError("candidate codes must be unique four-digit strings")
    def safe(value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise StocknoteContractError("request contains a non-finite number")
        return value

    candidates = []
    for _, row in result.iterrows():
        technical = {}
        for key in ("現在値", "RSI14", "BB位置", "MA25", "MA75", "MA200", "出来高倍率", "ATR14",
                    "ローソク足パターン", "シグナル種別", "RR"):
            technical[key] = safe(row.get(key))
        technical["損切り候補"] = safe(row.get("損切り候補", row.get("損切り価格")))
        technical["利確候補"] = safe(row.get("利確候補", row.get("利確目標")))
        candidates.append({"code": str(row["コード"]), "name": str(row.get("銘柄名", "")),
            "meeting_decision": str(row.get("最終判断", "")),
            "official_fundamentals": {k: safe(row.get(k, "")) for k in ("ファンダメンタル評価", "ファンダメンタルスコア", "ファンダメンタル取得元")},
            "technical_values": technical,
            "order_plan": {k: safe(row.get(k, "")) for k in ("注文方式", "買いゾーン下限", "買いゾーン上限", "損切り価格", "利確目標", "RR")}})
    created = generated_at or datetime.now(timezone.utc)
    if created.tzinfo is None:
        raise StocknoteContractError("generated_at must include a timezone")
    payload = {"schema_version": SCHEMA_VERSION, "run_id": run_id,
               "generated_at": created.isoformat(), "candidates": candidates}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"stocknote_request_{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return run_id, path


def consume_shadow(result: pd.DataFrame, directory: Path, run_id: str, *, max_age_hours: float = 24,
                   now: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """Return an annotated copy and status; every file failure is fail-open."""
    annotated = result.copy()
    for column in STOCKNOTE_COLUMNS:
        annotated[column] = ""
    path = directory / f"stocknote_response_{run_id}.json"
    if not path.exists():
        annotated.attrs["stocknote_status"] = "response_missing"
        return annotated, "response_missing"
    try:
        payload = _finite_json(path)
        analyses = validate_response(payload, run_id, set(annotated["コード"].astype(str)), now=now,
                                     max_age_hours=max_age_hours)
    except StocknoteContractError as exc:
        status = f"response_rejected: {exc}"
        annotated.attrs["stocknote_status"] = status
        return annotated, status
    indexed = {item["code"]: item for item in analyses}
    for index, row in annotated.iterrows():
        item = indexed.get(str(row["コード"]))
        if item:
            for source, column in STOCKNOTE_FIELDS.items():
                annotated.at[index, column] = item.get(source, "")
    annotated.attrs["stocknote_status"] = "accepted"
    return annotated, "accepted"


def write_shadow_report(result: pd.DataFrame, path: Path, run_id: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = result[result["stocknote_評価"] != ""] if "stocknote_評価" in result else result.iloc[0:0]
    lines = ["# stocknote 分析監査レポート", "", f"- run_id: `{run_id}`", f"- response: `{status}`",
             "- 参考情報のみ", "- 注文・最終判断・公式ファンダメンタルへの反映: なし",
             "- スコア・分類・価格・株数への反映: なし"]
    if status == "response_missing":
        lines += ["", "stocknote未取得"]
    elif status.startswith("response_rejected"):
        lines += ["", f"stocknote応答拒否: {status.partition(':')[2].strip()}"]
    for _, row in rows.iterrows():
        lines += ["", f"## {row['コード']} {row.get('銘柄名', '')}", "", f"- 既存判断: {row['最終判断']}"]
        for label, column in (("評価", "stocknote_評価"), ("信頼度", "stocknote_信頼度"),
                              ("要約", "stocknote_要約"), ("逆張りスコア", "stocknote_逆張りスコア"),
                              ("RSI", "stocknote_RSI"), ("BB位置", "stocknote_BB位置"),
                              ("トレンド", "stocknote_トレンド"),
                              ("推奨買い価格", "stocknote_推奨買い価格"), ("予想売り価格", "stocknote_予想売り価格"),
                              ("最終目標価格", "stocknote_最終目標価格"), ("注意点", "stocknote_注意点")):
            lines.append(f"- {label}: {row.get(column, '') or '未提示'}")
        lines += [f"- PER: {row.get('stocknote_PER', '') or '未提示'}（参考値・公式未確認）",
                  f"- PBR: {row.get('stocknote_PBR', '') or '未提示'}（参考値・公式未確認）",
                  f"- 財務健全性: {row.get('stocknote_財務健全性', '') or '未提示'}（参考値・公式未確認）",
                  f"- 売上成長: {row.get('stocknote_売上成長', '') or '未提示'}（参考値・公式未確認）",
                  f"- 利益成長: {row.get('stocknote_利益成長', '') or '未提示'}（参考値・公式未確認）",
                  f"- 公式PER（別枠）: {row.get('PER', '') or '未取得'}",
                  f"- 公式PBR（別枠）: {row.get('PBR', '') or '未取得'}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
