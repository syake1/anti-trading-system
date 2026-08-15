from datetime import date

import pandas as pd

from src.dashboard_exports import csv_download_data, dated_csv_filename, filter_meeting_history


def test_csv_download_data_uses_utf8_bom_and_preserves_japanese():
    payload = csv_download_data(pd.DataFrame([{"銘柄名": "トヨタ自動車"}]))

    assert payload.startswith(b"\xef\xbb\xbf")
    assert "トヨタ自動車" in payload.decode("utf-8-sig")


def test_dated_csv_filename_contains_requested_date():
    assert dated_csv_filename("meeting_history", date(2026, 8, 15)) == "meeting_history_20260815.csv"


def test_filter_meeting_history_matches_the_visible_filters():
    history = pd.DataFrame([
        {"シグナル日": "2026-08-15", "コード": "7203", "会社名": "トヨタ自動車", "ランク": "S"},
        {"シグナル日": "2026-08-14", "コード": "6758", "会社名": "ソニーグループ", "ランク": "A"},
    ])

    result = filter_meeting_history(history, ["2026-08-15"], ["S"], "トヨタ")

    assert result.to_dict("records") == [history.iloc[0].to_dict()]
