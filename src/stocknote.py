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
STOCKNOTE_COLUMNS = ["stocknote_評価", "stocknote_信頼度", "stocknote_要約"]


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
        if not isinstance(item, dict) or set(item) != {"code", "assessment", "confidence", "summary"}:
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
        candidates.append({"code": str(row["コード"]), "name": str(row.get("銘柄名", "")),
            "meeting_decision": str(row.get("最終判断", "")),
            "official_fundamentals": {k: safe(row.get(k, "")) for k in ("ファンダメンタル評価", "ファンダメンタルスコア", "ファンダメンタル取得元")},
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
        return annotated, "response_missing"
    try:
        payload = _finite_json(path)
        analyses = validate_response(payload, run_id, set(annotated["コード"].astype(str)), now=now,
                                     max_age_hours=max_age_hours)
    except StocknoteContractError as exc:
        return annotated, f"response_rejected: {exc}"
    indexed = {item["code"]: item for item in analyses}
    for index, row in annotated.iterrows():
        item = indexed.get(str(row["コード"]))
        if item:
            annotated.at[index, "stocknote_評価"] = item["assessment"]
            annotated.at[index, "stocknote_信頼度"] = item["confidence"]
            annotated.at[index, "stocknote_要約"] = item["summary"]
    return annotated, "accepted"


def write_shadow_report(result: pd.DataFrame, path: Path, run_id: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = result[result["stocknote_評価"] != ""] if "stocknote_評価" in result else result.iloc[0:0]
    lines = ["# stocknote シャドーモード比較", "", f"- run_id: `{run_id}`", f"- response: `{status}`",
             "- 注文・最終判断・公式ファンダメンタルへの反映: なし", "", "|コード|既存判断|stocknote評価|信頼度|", "|---|---|---|---|"]
    lines += [f"|{r['コード']}|{r['最終判断']}|{r['stocknote_評価']}|{r['stocknote_信頼度']}|" for _, r in rows.iterrows()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
