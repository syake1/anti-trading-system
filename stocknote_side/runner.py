"""Validate one stocknote request and atomically produce its response."""
from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

SCHEMA_VERSION = "1.0"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")
CODE = re.compile(r"^[0-9]{4}$")
ASSESSMENTS = {"positive", "neutral", "negative", "insufficient"}
RESPONSE_FIELDS = {
    "code", "assessment", "confidence", "summary", "contrarian_score", "rsi",
    "bb_position", "trend", "per", "pbr", "financial_health", "revenue_growth",
    "profit_growth", "recommended_buy_price", "expected_sell_price",
    "final_target_price", "cautions",
}
NUMBER_FIELDS = {
    "contrarian_score", "rsi", "per", "pbr", "recommended_buy_price",
    "expected_sell_price", "final_target_price",
}
TEXT_FIELDS = {
    "bb_position", "trend", "financial_health", "revenue_growth", "profit_growth", "cautions",
}
REQUIRED_RESULT_FIELDS = {"assessment", "confidence", "summary"}
OFFICIAL_SOURCES = {"official", "edinet", "tdnet", "jpx", "company_ir", "company-ir"}
REFERENCE_SOURCES = {"reference", "kabutan", "minkabu", "株探", "みんかぶ"}


class ContractError(ValueError):
    """The input or generated output violates the Phase 2 contract."""


def _reject_constant(value: str):
    raise ContractError(f"non-finite number is not allowed: {value}")


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("request root must be an object")
    return payload


def _date_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ContractError("generated_at must be an RFC 3339 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("generated_at must be a valid RFC 3339 date-time") from exc
    if parsed.tzinfo is None:
        raise ContractError("generated_at must include a timezone")
    return parsed


def validate_request(payload: dict, path: Path) -> list[dict]:
    if set(payload) != {"schema_version", "run_id", "generated_at", "candidates"}:
        raise ContractError("invalid request fields")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("unsupported schema_version")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ContractError("invalid run_id")
    if path.name != f"stocknote_request_{run_id}.json":
        raise ContractError("request filename and run_id do not match")
    _date_time(payload["generated_at"])
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ContractError("candidates must contain between 1 and 3 stocks")
    seen = set()
    required = {"code", "name", "meeting_decision", "official_fundamentals", "order_plan"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ContractError("invalid candidate fields")
        code = candidate["code"]
        if not isinstance(code, str) or not CODE.fullmatch(code):
            raise ContractError("candidate code must be a four-digit string")
        if code in seen:
            raise ContractError("duplicate candidate code")
        seen.add(code)
        if not isinstance(candidate["name"], str) or not isinstance(candidate["meeting_decision"], str):
            raise ContractError("candidate name and meeting_decision must be strings")
        if not isinstance(candidate["official_fundamentals"], dict) or not isinstance(candidate["order_plan"], dict):
            raise ContractError("candidate information must be objects")
    return candidates


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _source_sections(raw: dict) -> tuple[dict, dict, dict]:
    """Keep trusted and reference values separate; reference never wins precedence."""
    official, reference, direct = {}, {}, {}
    for key, value in raw.items():
        normalized = str(key).lower()
        if normalized in OFFICIAL_SOURCES:
            if isinstance(value, dict):
                official.update(value)
        elif normalized in REFERENCE_SOURCES:
            if isinstance(value, dict):
                reference.update(value)
        elif key in RESPONSE_FIELDS:
            direct[key] = value
    # Existing flat stocknote results are accepted as direct results. Source-tagged
    # values use official > direct > reference precedence.
    merged = {**reference, **direct, **official}
    return official, reference, merged


def map_analysis(code: str, raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ContractError("analysis result must be an object")
    _official, _reference, merged = _source_sections(raw)
    if not REQUIRED_RESULT_FIELDS.issubset(merged):
        raise ContractError("analysis result is missing required fields")
    result = {"code": code}
    for field in RESPONSE_FIELDS - {"code"}:
        if field in merged and merged[field] is not None:
            result[field] = merged[field]
    if result["assessment"] not in ASSESSMENTS:
        raise ContractError("invalid assessment")
    if not _finite(result["confidence"]) or not 0 <= result["confidence"] <= 1:
        raise ContractError("confidence must be between 0 and 1")
    if not isinstance(result["summary"], str) or len(result["summary"]) > 2000:
        raise ContractError("summary must be a string of at most 2000 characters")
    for field in NUMBER_FIELDS:
        if field in result and not _finite(result[field]):
            raise ContractError(f"{field} must be a finite number")
    for field in TEXT_FIELDS:
        if field in result and not isinstance(result[field], str):
            raise ContractError(f"{field} must be a string")
    # Insufficient means absent observations remain absent: nothing is synthesized.
    return result


def _call_analyzer(analyzer: Callable, candidate: dict) -> object:
    kwargs = {
        "code": candidate["code"],
        "official_information": candidate["official_fundamentals"],
        "reference_information": {},
    }
    parameters = inspect.signature(analyzer).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()) or set(kwargs) <= set(parameters):
        return analyzer(**kwargs)
    return analyzer(candidate["code"])


def validate_response(payload: dict, requested_codes: set[str]) -> None:
    if set(payload) != {"schema_version", "run_id", "generated_at", "analyses"}:
        raise ContractError("invalid response fields")
    _date_time(payload["generated_at"])
    if payload["schema_version"] != SCHEMA_VERSION or not RUN_ID.fullmatch(payload["run_id"]):
        raise ContractError("invalid response identity")
    analyses = payload["analyses"]
    if not isinstance(analyses, list):
        raise ContractError("analyses must be an array")
    seen = set()
    for item in analyses:
        if not isinstance(item, dict) or not {"code", *REQUIRED_RESULT_FIELDS} <= set(item) <= RESPONSE_FIELDS:
            raise ContractError("invalid response analysis fields")
        if item["code"] not in requested_codes or item["code"] in seen:
            raise ContractError("response contains an unrequested or duplicate code")
        seen.add(item["code"])
        map_analysis(item["code"], item)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Validate bytes actually written before publishing them.
        check = _read_json(Path(temporary))
        validate_response(check, {item["code"] for item in payload["analyses"]})
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def process_request(request_path: str | Path, analyzer: Callable, *, force: bool = False,
                    now: datetime | None = None) -> Path:
    path = Path(request_path)
    payload = _read_json(path)
    candidates = validate_request(payload, path)
    response_path = path.with_name(f"stocknote_response_{payload['run_id']}.json")
    if response_path.exists() and not force:
        raise FileExistsError(f"response already exists: {response_path}")
    analyses = []
    for candidate in candidates:
        try:
            analyses.append(map_analysis(candidate["code"], _call_analyzer(analyzer, candidate)))
        except Exception as exc:  # One stock must never prevent analysis of the others.
            analyses.append({
                "code": candidate["code"], "assessment": "insufficient", "confidence": 0.0,
                "summary": f"分析できませんでした: {type(exc).__name__}",
            })
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    response = {"schema_version": SCHEMA_VERSION, "run_id": payload["run_id"],
                "generated_at": created, "analyses": analyses}
    validate_response(response, {candidate["code"] for candidate in candidates})
    _atomic_write(response_path, response)
    return response_path
