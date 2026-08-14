import json
import pandas as pd

from src.jgb_yields import analyze_jgb, format_jgb_message, jgb_sector_impacts


def config():
    with open("config.json", encoding="utf-8") as source:
        return json.load(source)


def test_yields_changes_and_curve_are_observed_values():
    history = {"2Y": pd.Series([0.5] * 20 + [0.55]),
               "10Y": pd.Series([1.0] * 20 + [1.15]),
               "30Y": pd.Series([2.0] * 20 + [2.1])}
    result = analyze_jgb(history, "2026-08-14T06:00:00+09:00")
    rows = {row["tenor"]: row for row in result.tenors}
    assert rows["10Y"]["change_1d_bp"] == 15
    assert rows["10Y"]["change_20d_bp"] == 15
    assert result.spread_10y_2y_bp == 60
    assert jgb_sector_impacts(result, config())["銀行"] > 0


def test_missing_data_is_not_guessed():
    result = analyze_jgb({"10Y": pd.Series([1.1])}, "x")
    rows = {row["tenor"]: row for row in result.tenors}
    assert rows["2Y"]["yield_pct"] is None
    assert rows["10Y"]["change_1d_bp"] is None
    assert result.spread_10y_2y_bp is None
    assert jgb_sector_impacts(result, config()) == {}
    assert any("欠損" in line for line in format_jgb_message(result))
