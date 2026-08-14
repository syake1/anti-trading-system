import json

import pandas as pd

from src.jgb_yields import analyze_jgb
from src.sector_strategy import evaluate_bank_sector


def config():
    with open("config.json", encoding="utf-8") as source:
        return json.load(source)


def bank(**updates):
    row = {"現在値": 990, "MA25": 1000, "25日線乖離率": -1, "RSI14": 48,
           "BB位置": "-0.5σ", "出来高倍率": 1.2, "当日騰落率": 1,
           "直近3日騰落率": -1, "直近5日騰落率": -2,
           "ローソク足パターン": "強気包み足", "除外理由": ""}
    row.update(updates)
    return row


def rates(two_move=2, ten_move=8, thirty_move=7):
    return analyze_jgb({"2Y": pd.Series([0.5] * 5 + [0.5 + two_move / 100]),
                        "10Y": pd.Series([1.0] * 5 + [1.0 + ten_move / 100]),
                        "30Y": pd.Series([1.5] * 5 + [1.5 + thirty_move / 100])}, "x")


def test_tailwind_does_not_promote_without_reversal():
    result = evaluate_bank_sector(rates(), bank(ローソク足パターン="なし"), config())
    assert result.rate_regime == "上昇・スティープ化"
    assert result.classification == "押し目監視"
    assert result.morning_recheck is True
    assert result.rate_metrics["spread_change_5d_bp"] == 6


def test_reversal_without_volume_confirmation_remains_watch():
    result = evaluate_bank_sector(rates(), bank(出来高倍率=0.8), config())
    assert result.classification == "押し目監視"


def test_pullback_requires_tailwind_trend_and_reversal():
    result = evaluate_bank_sector(rates(), bank(), config())
    assert result.classification == "押し目候補"
    assert result.morning_recheck is False


def test_flattening_lowers_confidence_and_overheating_forbids_chasing():
    result = evaluate_bank_sector(rates(two_move=10, ten_move=5), bank(RSI14=74), config())
    assert result.rate_regime == "上昇・フラット化"
    assert result.rate_confidence == "中"
    assert result.classification == "追いかけ禁止"


def test_missing_rate_or_technical_data_is_not_neutral():
    no_rate = evaluate_bank_sector(analyze_jgb({"10Y": pd.Series([1.0] * 6)}, "x"), bank(), config())
    no_technical = evaluate_bank_sector(rates(), bank(**{"BB位置": None}), config())
    assert no_rate.classification == no_rate.rate_regime == "評価不能"
    assert no_rate.morning_recheck is True
    assert no_technical.classification == "評価不能"
    assert "BB位置" in no_technical.missing_data


def test_rate_headwind_and_broken_trend_is_avoid():
    result = evaluate_bank_sector(rates(two_move=-2, ten_move=-8),
                                  bank(**{"25日線乖離率": -6, "現在値": 940, "ローソク足パターン": "なし"}), config())
    assert result.rate_regime == "低下・フラット化"
    assert result.classification == "回避"
