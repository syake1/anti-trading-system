import pandas as pd

from src.japan_rates import (JapanRates, analyze_japan_rates, classify_rate,
                              rate_sector_impacts, stock_rate_impact)
from src.utils import load_config


def histories():
    return {"JP10Y": pd.Series([1.50] * 15 + [1.60, 1.65, 1.68, 1.70, 1.74, 1.82]),
            "JP2Y": pd.Series([.50] * 20 + [.57]), "JP30Y": pd.Series([2.5, 2.6])}


def test_bp_curve_and_determinism():
    config = load_config()
    first = analyze_japan_rates(histories(), config, observed_at="2026-08-14T06:00:00+09:00")
    second = analyze_japan_rates(histories(), config, observed_at="2026-08-14T06:00:00+09:00")
    assert first == second
    assert first.jp_10y_change_bp == 8
    assert first.jp_10y_change_5d_bp == 22
    assert first.jp_10y_change_20d_bp == 32
    assert first.jp_10y_2y_spread == 1.25
    assert first.jp_rate_regime == "急上昇"  # multi-day confirmation


def test_missing_is_not_guessed_and_does_not_crash():
    rates = analyze_japan_rates({}, load_config())
    assert rates.jp_10y_yield is None
    assert rates.jp_rate_regime == "取得不可"


def test_configurable_regimes_and_sector_directions():
    config = load_config()
    assert classify_rate(10, 0, 0, config) == "急上昇"
    rising = JapanRates("x", jp_10y_change_bp=10, jp_rate_regime="急上昇")
    impacts = rate_sector_impacts(rising)
    assert impacts["銀行"] == 2 and impacts["保険"] == 2
    assert impacts["不動産"] == -2 and impacts["REIT"] == -2
    assert rate_sector_impacts(rising, risk_off=True)["銀行"] == 0


def test_growth_us_rates_and_export_yen_combinations_are_separate():
    rising = JapanRates("x", jp_10y_change_bp=8, jp_rate_regime="上昇")
    growth, reason = stock_rate_impact({"業種":"高PERグロース"}, rising, us10y_change_bp=4)
    export, export_reason = stock_rate_impact({"業種":"輸出"}, rising, us10y_change_bp=-4, usd_jpy_change_pct=-1)
    bank, _ = stock_rate_impact({"業種":"銀行"}, rising, us10y_change_bp=-4, usd_jpy_change_pct=-1)
    assert growth == -2 and "日米金利" in reason
    assert export == -1 and "円高" in export_reason
    assert bank == 1


def test_boj_tightening_requires_news_rates_and_yen():
    rates = analyze_japan_rates(histories(), load_config(), boj_news=True, usd_jpy_change_pct=-.5)
    assert rates.boj_news_observed is True and rates.boj_tightening_risk is True
