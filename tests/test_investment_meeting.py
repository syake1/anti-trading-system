import copy
import json
import pandas as pd

from src.investment_meeting import evaluate_candidates, morning_message
from src.risk_manager import position_size


def config():
    return json.load(open("config.json", encoding="utf-8"))


def candidate(**changes):
    row = {"コード": "9065", "会社名": "山九", "ランク": "A", "現在値": 1000,
           "損切り候補": 900, "利確候補": 1200, "RR": 2, "RSI14": 39,
           "BB位置": "-1.13σ", "出来高倍率": 1.1, "直近3日騰落率": -2,
           "直近5日騰落率": -3, "25日線乖離率": -4,
           "ローソク足パターン": "強気包み足", "シグナル種別": "BB逆張り", "除外理由": ""}
    row.update(changes)
    return row


def test_moderate_pullback_can_be_core_and_is_deterministic():
    frame = pd.DataFrame([candidate()])
    first = evaluate_candidates(frame, config())
    second = evaluate_candidates(frame, config())
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["最終分類"] == "主力候補"


def test_surge_is_rejected_low_volume_not_core_and_crash_is_capped():
    rows = pd.DataFrame([
        candidate(コード="7936", 会社名="アシックス", 除外理由="急騰済み（当日急騰）"),
        candidate(コード="3911", 会社名="Aiming", 出来高倍率=.5),
        candidate(コード="9999", 直近3日騰落率=-11),
    ])
    result = evaluate_candidates(rows, config()).set_index("コード")
    assert result.loc["7936", "最終分類"] == "見送り"
    assert result.loc["3911", "最終分類"] != "主力候補"
    assert result.loc["9999", "最終分類"] in ("小口候補", "監視", "見送り")


def test_position_size_respects_all_portfolio_limits():
    c = config()
    sized = position_size(1000, 900, "主力候補", c)
    assert sized["必要資金"] <= 500_000
    assert sized["推奨株数"] * 100 <= 21_000
    assert sized["現金比率（購入後）"] >= 25
    capped = copy.deepcopy(c); capped["portfolio"]["current_positions"] = 5
    assert position_size(1000, 900, "主力候補", capped)["推奨株数"] == 0
    cash = copy.deepcopy(c); cash["portfolio"]["current_cash"] = 750_000
    assert position_size(1000, 900, "主力候補", cash)["推奨株数"] == 0


def test_zero_candidates_reports_no_new_purchase():
    result = evaluate_candidates(pd.DataFrame(), config())
    assert "本日は新規買いなし" in morning_message(result, config())


def test_workflow_utc_conversion_and_manual_dispatch():
    workflow = open(".github/workflows/morning_investment_meeting.yml", encoding="utf-8").read()
    assert "45 20 * * 0-4" in workflow  # 05:45 JST, leaving 45 minutes before 06:30
    assert "30 23 * * 0-4" in workflow  # 08:30 JST
    assert "workflow_dispatch" in workflow


def test_order_levels_primary_cap_and_determinism():
    rows = pd.DataFrame([candidate(コード=f"{i:04d}", スコア=20-i, ATR14=20,
        前日高値=1010, 反転足高値=1015, 直近2日高値=1020, 直近安値=900,
        反転足安値=920, BB下限=880, **{"BB-1σ": 970, "BB-2σ": 940, "直近支持線": 950}) for i in range(5)])
    result = evaluate_candidates(rows, config())
    assert (result["最終判断"] == "主力").sum() <= 3
    active = result[result["最終判断"].isin(["主力", "小口"])]
    assert (active["逆指値発動価格"] > active["反転確認高値"]).all()
    assert (active["追いかけ禁止価格"] > active["逆指値発動価格"]).all()
    assert (active["損切り価格"] < active["逆指値発動価格"]).all()
    assert (active["RR"] == 2).all()
    assert active["必要資金"].sum() <= 2_250_000
    pd.testing.assert_frame_equal(result, evaluate_candidates(rows, config()))


def test_large_gap_is_skipped():
    result = evaluate_candidates(pd.DataFrame([candidate(gap_pct=5.0)]), config())
    assert result.iloc[0]["最終判断"] == "見送り"
