from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pandas as pd

from src.investment_meeting import evaluate_candidates
from src.jgb_yields import analyze_jgb
from src.night_meeting import generate_night_result, night_message, save_night_result, load_latest_night_result


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
