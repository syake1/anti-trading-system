from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from src.stocknote import (StocknoteContractError, consume_shadow, export_request,
                           validate_response, write_shadow_report)


NOW = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)


def meeting_result():
    return pd.DataFrame([{
        "コード": "9065", "銘柄名": "山九", "最終判断": "小口",
        "ファンダメンタル評価": "良好", "ファンダメンタルスコア": 8,
        "ファンダメンタル取得元": "EDINET", "注文方式": "指値",
        "買いゾーン下限": 980, "買いゾーン上限": 1000,
        "損切り価格": 900, "利確目標": 1200, "RR": 2.0,
    }])


def response(run_id="run_123456", **changes):
    value = {"schema_version": "1.0", "run_id": run_id, "generated_at": NOW.isoformat(),
             "analyses": [{"code": "9065", "assessment": "positive",
                            "confidence": .8, "summary": "堅調"}]}
    value.update(changes)
    return value


def detailed_response():
    item = response()["analyses"][0]
    item.update({"contrarian_score": 72, "rsi": 38.5, "bb_position": "-1.2σ", "trend": "反転待ち",
                 "per": 13.2, "pbr": 1.3, "financial_health": "健全", "revenue_growth": "+6%",
                 "profit_growth": "+10%", "recommended_buy_price": 990,
                 "expected_sell_price": 1150, "final_target_price": 1200, "cautions": "出来高に注意"})
    return response(analyses=[item])


def test_request_contract_and_filename(tmp_path):
    run_id, path = export_request(meeting_result(), tmp_path, run_id="run_123456", generated_at=NOW)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "stocknote_request_run_123456.json"
    assert payload["schema_version"] == "1.0"
    assert payload["candidates"][0]["code"] == "9065"
    assert run_id == payload["run_id"]


@pytest.mark.parametrize("payload", [
    response(run_id="wrong_run"),
    response(schema_version="2.0"),
    response(analyses=[{"code": "123", "assessment": "neutral", "confidence": .5, "summary": "x"}]),
    response(analyses=[{"code": "9065", "assessment": "neutral", "confidence": .5, "summary": "x"},
                       {"code": "9065", "assessment": "neutral", "confidence": .5, "summary": "x"}]),
])
def test_response_rejects_identity_version_code_and_duplicates(payload):
    with pytest.raises(StocknoteContractError):
        validate_response(payload, "run_123456", {"9065"}, now=NOW)


def test_expired_response_and_non_finite_json_fail_open(tmp_path):
    run_id = "run_123456"
    old = response(generated_at=(NOW - timedelta(hours=25)).isoformat())
    (tmp_path / f"stocknote_response_{run_id}.json").write_text(json.dumps(old), encoding="utf-8")
    annotated, status = consume_shadow(meeting_result(), tmp_path, run_id, now=NOW)
    assert status.startswith("response_rejected")
    assert annotated.loc[0, "stocknote_評価"] == ""
    (tmp_path / f"stocknote_response_{run_id}.json").write_text(
        json.dumps(response()).replace("0.8", "NaN"), encoding="utf-8")
    _, status = consume_shadow(meeting_result(), tmp_path, run_id, now=NOW)
    assert "non-finite" in status


def test_missing_and_broken_response_never_stop_meeting(tmp_path):
    result, status = consume_shadow(meeting_result(), tmp_path, "run_123456", now=NOW)
    assert status == "response_missing"
    assert list(result["最終判断"]) == ["小口"]
    (tmp_path / "stocknote_response_run_123456.json").write_text("{broken", encoding="utf-8")
    result, status = consume_shadow(meeting_result(), tmp_path, "run_123456", now=NOW)
    assert status.startswith("response_rejected")
    assert result.loc[0, "損切り価格"] == 900


def test_accepted_response_only_adds_dedicated_columns_and_shadow_report(tmp_path):
    baseline = meeting_result()
    (tmp_path / "stocknote_response_run_123456.json").write_text(
        json.dumps(response(), ensure_ascii=False), encoding="utf-8")
    result, status = consume_shadow(baseline, tmp_path, "run_123456", now=NOW)
    assert status == "accepted"
    assert result.loc[0, "stocknote_評価"] == "positive"
    for protected in ("最終判断", "ファンダメンタル評価", "ファンダメンタルスコア",
                      "損切り価格", "利確目標", "RR", "買いゾーン下限", "買いゾーン上限"):
        assert result.loc[0, protected] == baseline.loc[0, protected]
    report = tmp_path / "report.md"
    write_shadow_report(result, report, "run_123456", status)
    assert "注文・最終判断・公式ファンダメンタルへの反映: なし" in report.read_text(encoding="utf-8")


def test_phase2_details_are_advisory_and_official_values_remain_separate(tmp_path):
    baseline = meeting_result()
    baseline["PER"], baseline["PBR"], baseline["推奨株数"] = 12.6, 1.1, 100
    (tmp_path / "stocknote_response_run_123456.json").write_text(
        json.dumps(detailed_response(), ensure_ascii=False), encoding="utf-8")
    result, status = consume_shadow(baseline, tmp_path, "run_123456", now=NOW)
    assert result.loc[0, "stocknote_PER"] == 13.2
    assert result.loc[0, "PER"] == 12.6
    for protected in ("最終判断", "注文方式", "買いゾーン下限", "買いゾーン上限", "推奨株数"):
        assert result.loc[0, protected] == baseline.loc[0, protected]
    report = tmp_path / "audit.md"
    write_shadow_report(result, report, "run_123456", status)
    audit = report.read_text(encoding="utf-8")
    assert "13.2（参考値・公式未確認）" in audit
    assert "公式PER（別枠）: 12.6" in audit
    assert "最終目標価格: 1200" in audit


def test_stocknote_message_is_compact():
    result = meeting_result()
    result["推奨株数"] = 100
    result["stocknote_評価"], result["stocknote_信頼度"], result["stocknote_要約"] = "positive", .8, "堅調"
    result.attrs["stocknote_status"] = "accepted"
    # Avoid constructing unrelated market fixtures: this test checks the dedicated formatter through its public helper.
    from src.investment_meeting import stocknote_message
    assert "参考情報・売買判断には未反映" in "\n".join(stocknote_message(result.iloc[0], "accepted"))
    assert stocknote_message(result.iloc[0], "response_missing") == ["stocknote分析社員：stocknote未取得"]
    rejected = stocknote_message(result.iloc[0], "response_rejected: invalid JSON")
    assert rejected == ["stocknote分析社員：response_rejected（invalid JSON）"]
