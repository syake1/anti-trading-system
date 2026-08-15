import pandas as pd
import sys
from types import SimpleNamespace

from src.google_sheets import HEADERS, append_records, build_meeting_records, record_meeting_safely


class Worksheet:
    title = "Sheet1"

    def __init__(self, values=None):
        self.values = values or []
        self.appended = []

    def update_title(self, title): self.title = title
    def get_all_values(self): return self.values
    def append_row(self, row, **_): self.values.append(row)
    def append_rows(self, rows, **_): self.appended.extend(rows)


def test_build_records_uses_existing_morning_decision_and_candidate_data():
    result = pd.DataFrame([{"コード": "1234", "銘柄名": "テスト", "最終分類": "主力候補",
                            "分析コメント": "既存理由", "注文理由": "反転後", "RR": 2.0}])
    candidates = pd.DataFrame([{"コード": "1234", "業種": "銀行業", "RSI14": 39,
                                "MA25": 1000, "25日線乖離率": -3, "現在値": 970,
                                "判定理由": "テクニカル既存判定"}])
    record = build_meeting_records("朝会", result, candidates, "2026-08-15T06:00:00+09:00")[0]
    assert record["会議判定"] == "主力候補"
    assert record["テクニカル理由"] == "テクニカル既存判定"
    assert record["RSI14"] == 39
    assert record["候補時株価"] == 970


def test_append_records_deduplicates(monkeypatch):
    worksheet = Worksheet([HEADERS, ["2026-08-15", "朝会", "1234"]])
    spreadsheet = type("Spreadsheet", (), {"get_worksheet": lambda self, index: worksheet})()
    client = type("Client", (), {"open_by_key": lambda self, key: spreadsheet})()
    monkeypatch.setitem(sys.modules, "gspread", SimpleNamespace(service_account_from_dict=lambda data: client))
    records = [dict(zip(HEADERS, ["2026-08-15", "朝会", "1234"] + [""] * (len(HEADERS) - 3))),
               dict(zip(HEADERS, ["2026-08-15", "朝会", "5678"] + [""] * (len(HEADERS) - 3)))]
    assert append_records(records, "sheet", "{}") == 1
    assert worksheet.title == "会議記録"
    assert worksheet.appended[0][:3] == ["2026-08-15", "朝会", "5678"]


def test_missing_secrets_and_write_error_are_fail_open(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_SPREADSHEET_ID", raising=False)
    monkeypatch.delenv("GOOGLE_SHEETS_CREDENTIALS_JSON", raising=False)
    assert record_meeting_safely("朝会", pd.DataFrame()) == 0
    assert "Secrets未設定" in capsys.readouterr().out
    monkeypatch.setenv("GOOGLE_SPREADSHEET_ID", "id")
    monkeypatch.setenv("GOOGLE_SHEETS_CREDENTIALS_JSON", "not-json")
    assert record_meeting_safely("朝会", pd.DataFrame()) == 0
    assert "処理を継続" in capsys.readouterr().out
