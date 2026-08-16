import json
import pandas as pd

from src.fundamentals import assess, derive_official_metrics, enrich_candidates
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


def test_explicit_profit_transitions_and_growth_quadrants():
    basis = {"comparison_basis_verified": True}
    assert derive_official_metrics({**basis, "net_profit_prior": -10, "net_profit": 5})["profit_transition"] == "黒字転換"
    assert derive_official_metrics({**basis, "net_profit_prior": 10, "net_profit": -5})["profit_transition"] == "赤字転落"
    assert derive_official_metrics({**basis, "net_profit_prior": -10, "net_profit": -5})["profit_transition"] == "赤字継続"
    assert derive_official_metrics({**basis, "net_profit_prior": 10, "net_profit": 5})["profit_transition"] == "黒字継続"
    cases = [((110, 100, 12, 10), "増収増益"), ((110, 100, 8, 10), "増収減益"),
             ((90, 100, 12, 10), "減収増益"), ((90, 100, 8, 10), "減収減益")]
    for values, expected in cases:
        revenue, prior_revenue, operating, prior_operating = values
        result = derive_official_metrics({**basis, "revenue": revenue, "revenue_prior": prior_revenue,
                                          "operating_profit": operating, "operating_profit_prior": prior_operating})
        assert result["growth_quadrant"] == expected


def test_payout_and_dividend_change_require_both_explicit_values():
    verified = {"payout_basis_verified": True, "dividend_comparison_verified": True}
    result = derive_official_metrics({**verified, "eps": 50, "dividend": 20, "dividend_prior": 10})
    assert result["payout_ratio"] == 40
    assert result["dividend_change"] == "増配"
    assert derive_official_metrics({"eps": 50})["payout_ratio"] is None
    assert derive_official_metrics({**verified, "dividend": 0, "dividend_prior": 10})["dividend_change"] == "無配転落"
    assert derive_official_metrics({"dividend": 10})["dividend_change"] == "評価不能"


def test_score_reasons_are_saved_and_abbreviated_for_discord():
    result = evaluate_candidates(pd.DataFrame([candidate(comparison_basis_verified=True,
        payout_basis_verified=True, dividend_comparison_verified=True, net_profit_prior=-10, net_profit=5,
        revenue=110, revenue_prior=100, operating_profit=12, operating_profit_prior=10,
        dividend=20, dividend_prior=10)]), config()).iloc[0]
    assert "+1 売上高前年比プラス" in result["ファンダメンタル加減点理由"]
    message = "\n".join(fundamental_message(result))
    assert "黒字転換 / 業績 増収増益 / 配当 増配" in message
    assert "加減点：" in message
    assert "他" in message


def test_unverified_period_or_scope_never_produces_derived_assessment():
    result = derive_official_metrics({"net_profit_prior": -10, "net_profit": 5,
        "revenue": 110, "revenue_prior": 100, "operating_profit": 12,
        "operating_profit_prior": 10, "eps": 50, "dividend": 20, "dividend_prior": 10})
    assert result["profit_transition"] == "評価不能"
    assert result["growth_quadrant"] == "評価不能"
    assert result["payout_ratio"] is None
    assert result["dividend_change"] == "評価不能"


def test_partial_official_data_is_scored_but_remains_insufficient():
    evaluation = assess({"source": "EDINET", "source_reference": "https://example.invalid/doc",
                         "roe": 12, "equity_ratio": 45, "per": 10, "pbr": 1.2}, config())
    assert evaluation.score == 3
    assert evaluation.label == "データ不足"
    assert not evaluation.sufficient
    assert "必須項目" in evaluation.reason
