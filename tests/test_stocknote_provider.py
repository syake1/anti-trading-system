from datetime import datetime, timezone
import json

import pandas as pd

from src.investment_meeting import stocknote_message
from src.stocknote import consume_shadow
from stocknote_provider.analysis import analyze_candidate
from stocknote_side.runner import process_request, validate_response


def history(_code):
    # A real calculation over observations, with a final pullback.
    return pd.DataFrame({"Close": [100 + i * .4 for i in range(90)] + [132, 128, 124, 120, 118]})


def test_real_provider_response_is_accepted_and_shown_at_meeting(tmp_path, monkeypatch):
    monkeypatch.setattr("stocknote_provider.analysis._load_history", history)
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    request = {
        "schema_version": "1.0", "run_id": "real_123456", "generated_at": now.isoformat(),
        "candidates": [{"code": "7203", "name": "test", "meeting_decision": "watch",
                        "official_fundamentals": {"per": 10.5},
                        "technical_values": {key: None for key in ("現在値", "RSI14", "BB位置", "MA25", "MA75", "MA200", "出来高倍率", "ATR14", "ローソク足パターン", "シグナル種別", "損切り候補", "利確候補", "RR")},
                        "order_plan": {}}],
    }
    path = tmp_path / "stocknote_request_real_123456.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    response_path = process_request(path, analyze_candidate, now=now)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    validate_response(response, {"7203"})
    analysis = response["analyses"][0]
    assert analysis["assessment"] != "insufficient"
    assert analysis["confidence"] > 0
    assert analysis["per"] == 10.5
    assert {"contrarian_score", "rsi", "bb_position", "trend"} <= analysis.keys()

    meeting = pd.DataFrame([{"コード": "7203", "銘柄名": "test", "最終判断": "監視"}])
    accepted, status = consume_shadow(meeting, tmp_path, "real_123456", now=now)
    assert status == "accepted"
    assert "逆張りスコア" in analysis["summary"]
    assert "stocknote分析社員" in "\n".join(stocknote_message(accepted.iloc[0], status))


def test_missing_and_reference_fundamentals_remain_separate():
    result = analyze_candidate(code="7203", official_information={},
                               reference_information={"PER": 99.0}, history_loader=history)
    assert "official" not in result
    assert result["reference"] == {"per": 99.0}
    assert "pbr" not in result["reference"]


def technical_values(**changes):
    values = {key: None for key in ("現在値", "RSI14", "BB位置", "MA25", "MA75", "MA200", "出来高倍率",
                                               "ATR14", "ローソク足パターン", "シグナル種別", "損切り候補", "利確候補", "RR")}
    values.update(changes)
    return values


def test_kumagai_request_is_sufficient_without_external_history(tmp_path, monkeypatch):
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    technical = technical_values(**{
        "現在値": 1303.0, "RSI14": 30.63, "BB位置": "-1.69σ", "MA25": 1423.24,
        "MA75": 1427.57, "MA200": 1553.21, "出来高倍率": 2.11, "ATR14": 38.7,
        "ローソク足パターン": "陰線→陽線 / 2～5日下落後の陽線",
        "シグナル種別": "BB逆張り / BB＋RSI＋ストキャス", "損切り候補": 1244.96,
        "利確候補": 1419.09, "RR": 2.0,
    })
    calls = []
    def forbidden(_code):
        calls.append(_code)
        raise AssertionError("Yahoo loader must not be called")
    monkeypatch.setattr("stocknote_provider.analysis._load_history", forbidden)
    request = {"schema_version": "1.0", "run_id": "kumagai_1861", "generated_at": now.isoformat(),
               "candidates": [{"code": "1861", "name": "熊谷組", "meeting_decision": "watch",
                               "official_fundamentals": {}, "technical_values": technical, "order_plan": {}}]}
    path = tmp_path / "stocknote_request_kumagai_1861.json"
    path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    item = json.loads(process_request(path, analyze_candidate, now=now).read_text(encoding="utf-8"))["analyses"][0]
    assert item["assessment"] != "insufficient" and item["confidence"] > 0
    assert item["rsi"] == technical["RSI14"] and item["bb_position"] == technical["BB位置"]
    assert "1244.96" in item["summary"] and "1419.09" in item["summary"] and "RR 2" in item["summary"]
    assert calls == []
    assert request["candidates"][0]["technical_values"] == technical
    assert not {"recommended_buy_price", "expected_sell_price", "final_target_price"} & item.keys()


def test_observation_precedence_request_then_local_then_external():
    external_calls = []
    def external(code):
        external_calls.append(code)
        return history(code)
    local = lambda _code: {"BB位置": "-1.25σ", "RSI14": 40.0}
    result = analyze_candidate(code="7203", technical_information={"RSI14": 31.0},
                               local_history_loader=local, history_loader=external)
    assert result["rsi"] == 31.0 and result["bb_position"] == "-1.25σ"
    assert external_calls == []

    local_calls = []
    result = analyze_candidate(code="7203", technical_information={},
                               local_history_loader=lambda code: local_calls.append(code) or {},
                               history_loader=external)
    assert local_calls == ["7203"] and external_calls == ["7203"]
    assert result["assessment"] != "insufficient"


def test_only_all_sources_without_analyzable_values_is_insufficient():
    result = analyze_candidate(code="7203", technical_information={"現在値": 100.0},
                               local_history_loader=lambda _code: {"MA25": 101.0},
                               history_loader=lambda _code: (_ for _ in ()).throw(RuntimeError("offline")))
    assert result["assessment"] == "insufficient" and result["confidence"] == 0
