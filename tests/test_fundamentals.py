import json
import pandas as pd

from src.fundamentals import assess, enrich_candidates
from src.investment_meeting import evaluate_candidates, fundamental_message
from tests.test_investment_meeting import candidate


def config():
    return json.load(open("config.json", encoding="utf-8"))


def test_missing_fundamentals_can_never_be_primary():
    raw = candidate()
    for key in list(raw):
        if key.startswith("fundamental_") or key in ("revenue_yoy", "operating_profit_yoy", "ordinary_or_net_profit_yoy",
                "eps", "per", "pbr", "roe", "equity_ratio", "dividend_yield", "company_forecast", "latest_earnings_date", "revision", "important_disclosure"):
            raw.pop(key)
    result = evaluate_candidates(pd.DataFrame([raw]), config()).iloc[0]
    assert result["最終判断"] != "主力"
    assert result["ファンダメンタル評価"] == "データ不足"
    assert fundamental_message(result) == ["ファンダメンタル：データ不足", "主力判定不可 → 小口または見送り"]


def test_strong_official_data_scores_good_and_formats_discord():
    result = evaluate_candidates(pd.DataFrame([candidate()]), config()).iloc[0]
    assert result["ファンダメンタルスコア"] >= 8
    assert result["ファンダメンタル評価"] == "良好"
    assert "PER 12.6倍" in "\n".join(fundamental_message(result))


def test_downgrade_and_major_bad_news_are_strong_penalties():
    row = candidate(revision="下方修正", important_disclosure="不祥事", company_forecast="赤字転落・減収減益")
    evaluation = assess(row, config())
    assert evaluation.score <= 4
    assert evaluate_candidates(pd.DataFrame([row]), config()).iloc[0]["最終判断"] != "主力"


def test_enrichment_audits_success_and_missing_without_imputation(tmp_path):
    source = tmp_path / "fundamentals.csv"
    pd.DataFrame([{**{k: v for k, v in candidate().items() if k not in ("コード", "会社名")},
                   "code": "9065", "source": "TDnet", "source_reference": "TD20260807", "acquired_at": "2026-08-08T00:00:00Z"}]).to_csv(source, index=False)
    audit = tmp_path / "audit.csv"
    enriched = enrich_candidates(pd.DataFrame([candidate(コード="9065"), candidate(コード="9999")]), config(), source, audit)
    assert bool(enriched.iloc[0]["ファンダメンタル十分"])
    assert not bool(enriched.iloc[1]["ファンダメンタル十分"])
    logged = pd.read_csv(audit)
    assert set(logged.status) == {"success", "failure"}
    assert logged.loc[logged.code == 9999, "reason"].iloc[0].startswith("公式取得元")
