import copy
import json
import pandas as pd
from src.market_environment import analyze_market
from src.news_analysis import analyze_news, news_impacts
from src.sector_impact import sector_impacts
from src.investment_meeting import evaluate_candidates


def cfg(): return json.load(open("config.json", encoding="utf-8"))


def candidate():
    return pd.DataFrame([{"コード":"1","会社名":"半導体社","業種":"半導体","ランク":"A","現在値":1000,"RSI14":39,"BB位置":"-1σ","出来高倍率":1.1,"直近3日騰落率":-2,"直近5日騰落率":-3,"25日線乖離率":-4,"ローソク足パターン":"強気包み足","除外理由":""}])


def test_missing_market_and_zero_news_are_safe_and_deterministic():
    env = analyze_market([], cfg(), "2026-01-01T05:45:00+09:00")
    assert env.regime == "中立" and env.indicators == ()
    assert analyze_news(None, cfg()).empty
    assert env == analyze_market([], cfg(), "2026-01-01T05:45:00+09:00")


def test_sox_and_oil_are_sector_specific():
    env = analyze_market([{"indicator":"SOX","current":97,"previous_close":100}, {"indicator":"WTI","current":104,"previous_close":100}], cfg(), "x")
    impacts = sector_impacts(env, cfg())
    assert impacts["半導体"] < 0 and "銀行" not in impacts
    assert impacts["石油・資源"] > 0 and impacts["航空"] < 0


def test_japanese_bond_yields_bp_curve_and_sector_effects():
    rows = [{"indicator":"JP2Y", "current":0.8, "previous_close":0.75},
            {"indicator":"JP10Y", "current":1.7, "previous_close":1.6},
            {"indicator":"JP30Y", "current":3.0, "previous_close":2.8}]
    env = analyze_market(rows, cfg(), "x")
    values = {row["indicator"]: row for row in env.indicators}
    assert round(values["JP2Y"]["change_bp"], 6) == 5
    assert round(values["JP10Y"]["change_bp"], 6) == 10
    assert round(values["JP30Y"]["change_bp"], 6) == 20
    assert round(values["JP10Y_2Y_SPREAD"]["current"] * 100) == 90
    impacts = sector_impacts(env, cfg())
    assert impacts["銀行"] > 0 and impacts["不動産"] < 0


def test_emergency_stops_new_buy_and_untrusted_news_has_no_impact():
    raw = pd.DataFrame([{"source":"気象庁","title":"大規模地震と津波","url":"https://example.test/id","severity":5},
                        {"source":"","title":"半導体輸出規制","url":"","severity":5}])
    news = analyze_news(raw, cfg())
    assert news.iloc[0].emergency_risk
    assert news.iloc[1].news_impact_score == 0 and not news.iloc[1].trusted
    env = analyze_market([], cfg(), "x")
    assert evaluate_candidates(candidate(), cfg(), env, news).iloc[0]["最終判断"] == "見送り"


def test_strong_risk_off_shrinks_or_stops_core_size():
    rows = [{"indicator":x,"current":97,"previous_close":100} for x in ["NIKKEI_FUTURES","TOPIX","SP500","NASDAQ","SOX"]]
    rows += [{"indicator":"VIX","current":120,"previous_close":100}]
    env = analyze_market(rows, cfg(), "x")
    assert env.regime == "強いリスクオフ" and env.capital_ratio <= .25
    result = evaluate_candidates(candidate(), cfg(), env, analyze_news(None, cfg()))
    assert result.iloc[0]["推奨株数"] == 0
