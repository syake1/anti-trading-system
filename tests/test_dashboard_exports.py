from datetime import date

import pandas as pd

from src.dashboard_exports import (csv_download_data, dated_csv_filename, filter_meeting_history,
                                   performance_for_code, ranked_buy_candidates, read_candidate_csv,
                                   strategy_performance)


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


def test_ranked_buy_candidates_deduplicates_codes_and_limits_to_ten():
    candidates = pd.DataFrame([
        {"コード": str(1000 + index % 11), "買い・売り": "買い", "スコア": index, "シグナル日": "2026-08-15"}
        for index in range(15)
    ] + [{"コード": "9999", "買い・売り": "売り", "スコア": 999}])

    result = ranked_buy_candidates(candidates)

    assert len(result) == 10
    assert result["コード"].is_unique
    assert "9999" not in result["コード"].tolist()
    assert result["順位"].tolist() == list(range(1, 11))


def test_read_candidate_csv_fails_open_for_empty_and_missing_code(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    frame, error = read_candidate_csv(empty)
    assert frame.empty and error

    missing = tmp_path / "missing.csv"
    missing.write_text("会社名,スコア\n銘柄,10\n", encoding="utf-8")
    frame, error = read_candidate_csv(missing)
    assert frame.empty and "コード" in error


def test_performance_views_support_missing_values_and_strategy_summary():
    performance = pd.DataFrame([
        {"コード": "7203", "シグナル種別": "BB逆張り", "1日後騰落率": 1, "3日後騰落率": 3, "5日後騰落率": 5},
        {"コード": "7203", "シグナル種別": "BB逆張り", "1日後騰落率": -1, "3日後騰落率": 1, "5日後騰落率": -1},
        {"コード": "6758", "シグナル種別": None, "1日後騰落率": None},
    ])

    detail = performance_for_code(performance, "7203")
    comparison = strategy_performance(performance)

    assert len(detail) == 2
    bb = comparison.set_index("戦略").loc["BB逆張り"]
    assert bb["件数"] == 2
    assert bb["5日平均"] == 2
    assert bb["5日勝率"] == .5
