from datetime import date

import pandas as pd

from src.dashboard_exports import (csv_download_data, dated_csv_filename, filter_meeting_history,
                                   candidate_detail_rows, candidate_selector_labels, latest_meeting_view,
                                   load_meeting_reports, meeting_history_summary,
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


def test_saved_meeting_report_drives_latest_view_and_history(tmp_path):
    report = tmp_path / "morning" / "morning_meeting_20260815.md"
    report.parent.mkdir()
    report.write_text("""saved text
```csv
コード,銘柄名,最終判断,注文方式,分析コメント,注文理由,運用コメント
1234,実銘柄,小口,逆指値買い,反転を確認,高値抜けを待つ,損失上限を適用
```
""", encoding="utf-8")

    history = load_meeting_reports(tmp_path)
    view = latest_meeting_view(history, pd.DataFrame([{"code": "1234", "assessment": "慎重", "summary": "材料待ち"}]))
    summary = meeting_history_summary(history, pd.DataFrame([{
        "シグナル日": "2026-08-15", "コード": "1234", "1日後騰落率": 2.0,
        "3日後騰落率": -1.0, "5日後騰落率": 3.0,
    }]))

    assert view["date"] == "2026-08-15"
    assert "反転を確認" in view["system_opinion"]
    assert "材料待ち" in view["stocknote_opinion"]
    assert summary.iloc[0][["候補数", "採用数", "見送り数"]].tolist() == [1, 1, 0]
    assert summary.iloc[0]["翌日勝率"] == 1.0
    assert view["fundamentals"].iloc[0]["ファンダメンタル評価"] == "データ不足"
    assert "保存記録" in view["fundamentals"].iloc[0]["ファンダメンタル不足理由"]


def test_latest_meeting_view_exposes_persisted_fundamental_assessment():
    history = pd.DataFrame([{
        "会議日": "2026-08-16", "記録元": "morning_meeting_20260816.md",
        "コード": "7203", "銘柄名": "トヨタ自動車", "最終判断": "主力",
        "ファンダメンタル評価": "良好", "ファンダメンタルスコア": 9,
        "ファンダメンタル十分": True, "PER": 12.3, "ROE": 14.2,
        "ファンダメンタル加減点理由": "+1 ROE 8%以上",
        "ファンダメンタル取得元": "EDINET",
        "ファンダメンタル参照先": "https://disclosure2.edinet-fsa.go.jp/",
        "売上高前年同期比": 4.2, "営業利益前年同期比": 5.1, "時価総額": 1000000000,
        "直近決算発表日": "2026-06-30", "次回決算予定日": "データなし",
        "総合判定": "良好", "判定理由": "評価完了", "ファンダメンタル取得日時": "2026-08-16T00:00:00Z",
    }])

    fundamentals = latest_meeting_view(history)["fundamentals"]

    assert fundamentals.iloc[0]["ファンダメンタル評価"] == "良好"
    assert fundamentals.iloc[0]["ファンダメンタルスコア"] == 9
    assert fundamentals.iloc[0]["ファンダメンタル取得元"] == "EDINET"
    assert fundamentals.iloc[0]["ファンダメンタル参照先"].startswith("https://")
    assert fundamentals.iloc[0]["次回決算予定日"] == "データなし"
    assert fundamentals.iloc[0]["時価総額"] == 1000000000


def test_broken_meeting_reports_fail_open(tmp_path):
    (tmp_path / "morning_meeting_20260815.md").write_text("```csv\nnot,a,valid,row\n\"", encoding="utf-8")
    assert load_meeting_reports(tmp_path).empty


def test_saved_machine_readable_csv_drives_streamlit_view(tmp_path):
    folder = tmp_path / "morning"
    folder.mkdir()
    report = folder / "morning_meeting_20260816.md"
    report.write_text("human-readable report without an embedded table", encoding="utf-8")
    pd.DataFrame([{
        "コード": "7203", "銘柄名": "トヨタ自動車", "最終判断": "監視",
        "ファンダメンタル評価": "普通", "ファンダメンタルスコア": 7,
        "PER": 10.1, "売上高前年同期比": 6.2,
        "ファンダメンタル取得元": "EDINET",
    }]).to_csv(report.with_suffix(".csv"), index=False)

    view = latest_meeting_view(load_meeting_reports(tmp_path))

    assert view["fundamentals"].iloc[0]["PER"] == 10.1
    assert view["fundamentals"].iloc[0]["売上高前年同期比"] == 6.2


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


def test_candidate_selector_labels_show_rank_code_and_name():
    ranked = pd.DataFrame([
        {"順位": 1, "コード": "1111", "会社名": "一位社"},
        {"順位": 2, "コード": "2222", "会社名": "二位社"},
    ])

    assert candidate_selector_labels(ranked) == {
        "1111": "1位｜1111｜一位社",
        "2222": "2位｜2222｜二位社",
    }


def test_candidate_detail_rows_never_leaves_first_rank_data_after_switch():
    ranked = pd.DataFrame([
        {"順位": 1, "コード": "1111", "会社名": "一位社", "判定理由": "一位の理由"},
        {"順位": 2, "コード": "2222", "会社名": "二位社", "判定理由": "二位の理由"},
    ])
    meeting = pd.DataFrame([
        {"コード": "1111", "注文理由": "一位の注文", "損切り価格": 100},
        {"コード": "2222", "注文理由": "二位の注文", "損切り価格": 200},
    ])
    stocknote = pd.DataFrame([
        {"code": "1111", "summary": "一位の評価"},
        {"code": "2222", "summary": "二位の評価"},
    ])
    performance = pd.DataFrame([
        {"コード": "1111", "シグナル日": "2026-08-15", "1日後騰落率": 1},
        {"コード": "2222", "シグナル日": "2026-08-15", "1日後騰落率": 2},
    ])

    candidate, plan, notes, outcomes = candidate_detail_rows(
        ranked, meeting, stocknote, performance, "2222",
    )

    assert candidate["判定理由"] == "二位の理由"
    assert plan["注文理由"] == "二位の注文"
    assert notes["summary"].tolist() == ["二位の評価"]
    assert outcomes["コード"].tolist() == ["2222"]
    combined = " ".join(map(str, [candidate.to_dict(), plan.to_dict(), notes.to_dict(), outcomes.to_dict()]))
    assert "一位" not in combined and "1111" not in combined


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
