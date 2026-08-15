"""Fail-open Google Sheets audit logging for existing meeting results."""
from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

SHEET_NAME = "会議記録"
HEADERS = ["日付", "会議種別", "銘柄コード", "銘柄名", "セクター", "会議判定", "選定理由", "テクニカル理由",
           "RSI14", "BB位置", "MA25", "MA25乖離率", "出来高倍率", "ローソク足パターン", "Stocknote評価",
           "Stocknote confidence", "Stocknote contrarian score", "セクター評価", "候補時株価", "エントリー条件",
           "損切り候補", "利確候補", "RR", "備考"]


def _value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value) != "":
            return value
    return ""


def _source_by_code(candidates: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if candidates is None or candidates.empty:
        return {}
    return {str(row.get("コード", "")).strip(): row for row in candidates.fillna("").to_dict("records")}


def build_meeting_records(meeting_type: str, result: Any, candidates: pd.DataFrame | None = None,
                          observed_at: datetime | str | None = None) -> list[dict[str, Any]]:
    """Map existing decisions to the sheet schema without making new decisions."""
    sources = _source_by_code(candidates)
    stocknote: dict[str, Mapping[str, Any]] = {}
    if isinstance(result, Mapping):
        observed_at = observed_at or result.get("observed_at")
        stocknote = {str(row.get("code", "")): row for row in result.get("stocknote_analyses", [])}
        rows = result.get("bank_evaluations", []) if meeting_type == "夜会" else result.get("candidates", [])
    elif isinstance(result, pd.DataFrame):
        rows = result.fillna("").to_dict("records")
    else:
        rows = []
    if isinstance(observed_at, datetime):
        day = observed_at.astimezone(ZoneInfo("Asia/Tokyo")).date().isoformat()
    elif observed_at:
        day = str(observed_at)[:10]
    else:
        day = datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat()

    records = []
    for decision in rows:
        code = str(_value(decision, "コード", "code")).strip()
        source, note = sources.get(code, {}), stocknote.get(code, {})
        reasons = _value(decision, "分析コメント", "reasons", "summary")
        if isinstance(reasons, (list, tuple)):
            reasons = " / ".join(str(item) for item in reasons)
        entry = _value(decision, "注文理由")
        if not entry and meeting_type == "夜会":
            entry = "翌朝会で再確認" if decision.get("morning_recheck") else decision.get("principle", "")
        if not entry and meeting_type == "週末会議":
            entry = "月曜朝会で再確認"
        values = [day, meeting_type, code,
                  _value(decision, "銘柄名", "name") or _value(source, "会社名", "銘柄名"),
                  _value(source, "業種") or _value(decision, "sector"),
                  _value(decision, "最終分類", "最終判断", "night_category", "category"),
                  reasons or _value(source, "判定理由"), _value(source, "判定理由") or reasons,
                  _value(decision, "RSI", "RSI14") or _value(source, "RSI14", "RSI"),
                  _value(decision, "BB位置") or _value(source, "BB位置"), _value(source, "MA25"),
                  _value(source, "25日線乖離率"), _value(decision, "出来高倍率") or _value(source, "出来高倍率"),
                  _value(source, "ローソク足パターン"), _value(decision, "stocknote_評価") or note.get("assessment", ""),
                  _value(decision, "stocknote_信頼度") or note.get("confidence", ""),
                  _value(decision, "stocknote_逆張りスコア") or note.get("contrarian_score", ""),
                  _value(decision, "業種環境スコア", "rate_regime") or _value(source, "業種環境スコア"),
                  _value(source, "現在値"), entry, _value(decision, "損切り価格") or _value(source, "損切り候補"),
                  _value(decision, "利確目標") or _value(source, "利確候補"), _value(decision, "RR") or _value(source, "RR"),
                  _value(decision, "運用コメント") or note.get("summary", "")]
        records.append(dict(zip(HEADERS, values)))
    return records


def append_records(records: Iterable[Mapping[str, Any]], spreadsheet_id: str, credentials_json: str) -> int:
    """Append records that do not share the date/type/code uniqueness key."""
    import gspread

    client = gspread.service_account_from_dict(json.loads(credentials_json))
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.get_worksheet(0)
    if worksheet.title != SHEET_NAME:
        worksheet.update_title(SHEET_NAME)
    values = worksheet.get_all_values()
    if not values:
        worksheet.append_row(HEADERS, value_input_option="RAW")
        values = [HEADERS]
    existing = {(row[0], row[1], row[2]) for row in values[1:] if len(row) >= 3}
    pending = []
    for record in records:
        key = tuple(str(record.get(name, "")) for name in HEADERS[:3])
        if key not in existing:
            pending.append([record.get(name, "") for name in HEADERS])
            existing.add(key)
    if pending:
        worksheet.append_rows(pending, value_input_option="RAW")
    return len(pending)


def record_meeting_safely(meeting_type: str, result: Any, candidates: pd.DataFrame | None = None,
                          observed_at: datetime | str | None = None) -> int:
    """Write an audit; absent configuration and every external error fail open."""
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID", "").strip()
    credentials = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON", "").strip()
    if not spreadsheet_id or not credentials:
        print("警告: Google Sheets Secrets未設定のため会議記録をスキップします。")
        return 0
    try:
        count = append_records(build_meeting_records(meeting_type, result, candidates, observed_at),
                               spreadsheet_id, credentials)
        print(f"Google Sheetsへ会議記録を{count}件追記しました。")
        return count
    except Exception as exc:  # External audit logging must never stop a meeting or Discord.
        print(f"警告: Google Sheetsへの会議記録に失敗しました。処理を継続します: {exc}")
        return 0
