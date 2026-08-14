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
                        "official_fundamentals": {"per": 10.5}, "order_plan": {}}],
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
