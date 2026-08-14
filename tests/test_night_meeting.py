from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pandas as pd

from src.investment_meeting import evaluate_candidates, join_candidate_sectors, run_stocknote_cli
from src.stocknote import consume_shadow, export_request
from src.jgb_yields import analyze_jgb
from src.night_meeting import (generate_night_result, generate_weekend_result, night_message,
                               save_night_result, save_weekend_result, load_latest_night_result,
                               weekend_message)
from src.meeting_schedule import scheduled_meeting


def config():
    return json.load(open("config.json", encoding="utf-8"))


def rates(two_move=2, ten_move=8, thirty_move=7):
    return analyze_jgb({"2Y": pd.Series([0.5] * 5 + [0.5 + two_move / 100]),
                        "10Y": pd.Series([1.0] * 5 + [1.0 + ten_move / 100]),
                        "30Y": pd.Series([1.5] * 5 + [1.5 + thirty_move / 100])}, "x")


def bank(**updates):
    row = {"コード": "8300", "会社名": "テスト銀行", "業種": "銀行", "現在値": 990, "MA25": 1000,
           "25日線乖離率": -1, "RSI14": 48, "BB位置": "-0.5σ", "出来高倍率": 1.2,
           "当日騰落率": 1, "直近3日騰落率": -1, "直近5日騰落率": -2,
           "ローソク足パターン": "強気包み足", "除外理由": ""}
    row.update(updates)
    return row


def make(rows, jgb=None):
    observed = datetime(2026, 8, 14, 19, tzinfo=ZoneInfo("Asia/Tokyo"))
    return generate_night_result(pd.DataFrame(rows), jgb or rates(), config(), observed)


def test_bank_pullback_is_provisional_night_candidate():
    result = make([bank()])
    assert result["status"] == "provisional" and result["final_decision"] is False
    assert result["valid_until"] == "2026-08-17T05:45+09:00"
    assert result["categories"]["押し目候補"][0]["bank_classification"] == "押し目候補"
    assert "主力" not in night_message(result) or "主力・小口" in night_message(result)


def test_bank_overheating_is_never_chased():
    result = make([bank(RSI14=74)])
    row = result["categories"]["避ける"][0]
    assert row["bank_classification"] == "追いかけ禁止"
    assert "追いかけ禁止：8300 テスト銀行" in night_message(result)


def test_rate_headwind_is_avoided():
    result = make([bank(**{"25日線乖離率": -6, "現在値": 940, "ローソク足パターン": "なし"})],
                  rates(two_move=-2, ten_move=-8))
    assert result["categories"]["避ける"][0]["rate_wind"] == "逆風"


def test_missing_rates_go_to_morning_recheck():
    result = make([bank()], analyze_jgb({"10Y": pd.Series([1.0] * 6)}, "x"))
    assert result["categories"]["翌朝再確認"][0]["bank_classification"] == "評価不能"
    assert "日本2年金利：欠損" in night_message(result)


def test_night_candidate_cannot_become_primary_and_does_not_change_orders(tmp_path):
    # Night artifacts are reference-only and evaluate_candidates has no night input.
    candidate = {**bank(), "ランク": "A", "損切り候補": 900, "利確候補": 1200, "RR": 2,
                 "シグナル種別": "BB逆張り"}
    frame = pd.DataFrame([candidate])
    before = evaluate_candidates(frame, config())
    night = make([candidate])
    save_night_result(night, tmp_path)
    loaded = load_latest_night_result(tmp_path)
    after = evaluate_candidates(frame, config())
    assert loaded["categories"]["押し目候補"]
    assert "最終判断" not in loaded["categories"]["押し目候補"][0]
    pd.testing.assert_frame_equal(before, after)
    for column in ("注文方式", "推奨株数", "逆指値発動価格", "損切り価格", "利確目標", "RR"):
        assert before[column].equals(after[column])


def test_jst_schedule_selects_weekdays_saturday_and_never_sunday():
    utc = ZoneInfo("UTC")
    # UTC dates deliberately differ from the operational JST dates.
    assert scheduled_meeting(datetime(2026, 8, 14, 12, 0, tzinfo=utc)) == "night"  # Fri 21:00 JST
    assert scheduled_meeting(datetime(2026, 8, 14, 21, 0, tzinfo=utc)) == "weekend"  # Sat 06:00 JST
    assert scheduled_meeting(datetime(2026, 8, 15, 12, 0, tzinfo=utc)) is None  # Sat 21:00 JST
    assert scheduled_meeting(datetime(2026, 8, 16, 6, 0, tzinfo=ZoneInfo("Asia/Tokyo"))) is None


def test_every_weekday_and_only_weekdays_have_night_meeting():
    for day in range(10, 17):  # Monday through Sunday in August 2026
        at_nine = datetime(2026, 8, day, 21, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert (scheduled_meeting(at_nine) == "night") is (at_nine.weekday() < 5)


def test_weekend_meeting_is_broad_and_provisional():
    observed = datetime(2026, 8, 15, 6, tzinfo=ZoneInfo("Asia/Tokyo"))
    result = generate_weekend_result(pd.DataFrame([
        bank(), {**bank(コード="7203", 会社名="テスト自動車", RSI14=55, 当日騰落率=2), "業種": "自動車"},
    ]), observed)
    assert result["status"] == "provisional" and result["final_decision"] is False
    assert result["basis"] == "Friday close"
    assert {row["sector"] for row in result["candidates"]} == {"銀行", "自動車"}
    assert result["stocknote_employee"] == "enabled"
    message = weekend_message(result)
    assert "月曜朝の再確認条件" in message and "注文は確定せず" in message
    assert all("最終判断" not in row and "推奨株数" not in row for row in result["candidates"])


def test_weekend_integrates_bank_strategy_and_has_dedicated_section():
    observed = datetime(2026, 8, 15, 6, tzinfo=ZoneInfo("Asia/Tokyo"))
    result = generate_weekend_result(pd.DataFrame([bank()]), rates(), config(), observed)
    message = weekend_message(result)
    assert result["focus_sectors"] == ["銀行"]
    assert result["bank_evaluations"][0]["bank_classification"] == "押し目候補"
    for label in ("金利局面", "2年・10年・30年金利", "10年-2年スプレッド", "基本戦略",
                  "銀行押し目候補", "押し目監視", "追いかけ禁止", "回避", "月曜朝再確認"):
        assert label in message
    assert "注目セクター：不明" not in message


def test_weekend_carries_observed_jgb_even_without_bank_candidate():
    observed = datetime(2026, 8, 15, 6, tzinfo=ZoneInfo("Asia/Tokyo"))
    frame = pd.DataFrame([{**bank(コード="7203", 会社名="テスト自動車"), "業種": "自動車"}])
    result = generate_weekend_result(frame, rates(), config(), observed)
    assert result["jgb_rate_metrics"] == {
        "2Y_yield_pct": .52, "10Y_yield_pct": 1.08, "30Y_yield_pct": 1.57,
        "spread_10y_2y_bp": 56.0,
    }
    message = weekend_message(result)
    assert "金利局面：実データ取得済み（銀行候補なし）" in message
    assert "2年・10年・30年金利：+0.520% / +1.080% / +1.570%" in message
    assert "10年-2年スプレッド：+56.000bp" in message


def test_sector_master_fills_only_missing_sectors_and_focus_is_not_empty(tmp_path):
    master = tmp_path / "stocks.csv"
    pd.DataFrame([
        {"code": "1860", "name": "戸田建設", "market": "プライム", "industry": "建設"},
        {"code": "6758", "name": "ソニー", "market": "プライム", "industry": "電気機器"},
    ]).to_csv(master, index=False)
    candidates = pd.DataFrame([
        {**bank(コード="1860", 会社名="戸田建設"), "業種": ""},
        {**bank(コード="6758", 会社名="ソニー"), "業種": "電気機器"},
        {**bank(コード="9999", 会社名="不明"), "業種": ""},
    ])
    joined = join_candidate_sectors(candidates, master)
    result = generate_weekend_result(joined)
    assert result["focus_sectors"] == ["建設", "電気機器"]
    assert len(result["candidates"]) == 3 and result["candidates"][2]["sector"] == ""


def test_weekend_stocknote_cli_success_and_missing_response_fail_open(tmp_path):
    candidates = pd.DataFrame([bank()])
    run_id, request = export_request(candidates, tmp_path, run_id="weekend_123456")
    assert run_stocknote_cli(request, 30)
    annotated, status = consume_shadow(candidates, tmp_path, run_id)
    assert status == "accepted"
    analysis = annotated.iloc[0]
    result = generate_weekend_result(candidates)
    result["stocknote_employee"] = status
    result["stocknote_analyses"] = [{
        "code": "8300", "name": "テスト銀行",
        "assessment": analysis["stocknote_評価"], "confidence": analysis["stocknote_信頼度"],
        "contrarian_score": analysis["stocknote_逆張りスコア"], "summary": analysis["stocknote_要約"],
    }]
    assert "stocknote上位候補" in weekend_message(result)

    missing, missing_status = consume_shadow(candidates, tmp_path, "missing_123456")
    assert missing_status == "response_missing"
    assert missing["コード"].tolist() == candidates["コード"].tolist()


def test_weekend_discord_limits_and_overflow_audit_csv(tmp_path):
    observed = datetime(2026, 8, 15, 6, tzinfo=ZoneInfo("Asia/Tokyo"))
    rows = [bank(コード=f"{8300 + i}", RSI14=40) for i in range(12)]
    result = generate_weekend_result(pd.DataFrame(rows), rates(), config(), observed)
    message = weekend_message(result)
    assert "押し目候補 TOP10" in message and "ほか2件は監査CSV" in message
    save_weekend_result(result, tmp_path)
    audit = pd.read_csv(tmp_path / "weekend_meeting_20260815_audit.csv", dtype={"code": str})
    assert audit["code"].tolist() == ["8310", "8311"]


def test_weekend_stocknote_compact_fields_are_advisory():
    result = generate_weekend_result(pd.DataFrame([bank()]))
    result["stocknote_employee"] = "accepted"
    result["stocknote_analyses"] = [{"code": "8300", "name": "テスト銀行", "assessment": "positive",
                                      "confidence": .8, "contrarian_score": 7.5, "summary": "押し目を確認"}]
    message = weekend_message(result)
    assert "positive / confidence 80% / contrarian 7.5 / 押し目を確認" in message
    assert "判断には未反映" in message


def test_provisional_workflow_uses_required_utc_crons_and_is_separate():
    workflow = open(".github/workflows/provisional_meetings.yml", encoding="utf-8").read()
    assert "0 12 * * 1-5" in workflow  # weekdays 21:00 JST
    assert "0 21 * * 5" in workflow   # Friday 21:00 UTC = Saturday 06:00 JST
    assert "src.meeting_schedule" in workflow
    assert "morning" not in workflow
