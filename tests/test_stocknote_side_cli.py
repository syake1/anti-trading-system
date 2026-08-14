from datetime import datetime, timezone
import json

import pytest

from stocknote_side.runner import ContractError, process_request, validate_response


NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


def write_request(tmp_path, codes=("1111",), run_id="run_123456", **changes):
    candidates = [{"code": code, "name": f"name-{code}", "meeting_decision": "watch",
                   "official_fundamentals": {"per": 10.0}, "order_plan": {}}
                  for code in codes]
    payload = {"schema_version": "1.0", "run_id": run_id, "generated_at": NOW.isoformat(),
               "candidates": candidates}
    payload.update(changes)
    path = tmp_path / f"stocknote_request_{run_id}.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return path


def good_analyzer(*, code, official_information, reference_information):
    return {"assessment": "positive", "confidence": 0.8, "summary": code,
            "rsi": 31.5, "per": official_information["per"]}


@pytest.mark.parametrize("codes", [("1111",), ("1111", "2222", "3333")])
def test_one_and_three_stocks_succeed(tmp_path, codes):
    output = process_request(write_request(tmp_path, codes), good_analyzer, now=NOW)
    response = json.loads(output.read_text(encoding="utf-8"))
    assert [item["code"] for item in response["analyses"]] == list(codes)
    validate_response(response, set(codes))


def test_one_failure_does_not_stop_remaining_stocks(tmp_path):
    def analyzer(*, code, **_):
        if code == "2222":
            raise RuntimeError("provider unavailable")
        return {"assessment": "neutral", "confidence": 0.5, "summary": "ok"}

    output = process_request(write_request(tmp_path, ("1111", "2222", "3333")), analyzer, now=NOW)
    analyses = json.loads(output.read_text(encoding="utf-8"))["analyses"]
    assert [item["assessment"] for item in analyses] == ["neutral", "insufficient", "neutral"]
    assert "RuntimeError" in analyses[1]["summary"]


def test_more_than_three_and_duplicate_codes_are_rejected(tmp_path):
    with pytest.raises(ContractError, match="between 1 and 3"):
        process_request(write_request(tmp_path, ("1111", "2222", "3333", "4444")), good_analyzer)
    with pytest.raises(ContractError, match="duplicate"):
        process_request(write_request(tmp_path, ("1111", "1111"), run_id="run_654321"), good_analyzer)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_request_values_are_rejected(tmp_path, constant):
    path = write_request(tmp_path)
    path.write_text(path.read_text().replace("10.0", constant), encoding="utf-8")
    with pytest.raises(ContractError, match="non-finite"):
        process_request(path, good_analyzer)


def test_official_values_are_not_overwritten_by_reference_values(tmp_path):
    def analyzer(**_):
        return {"assessment": "positive", "confidence": 0.7, "summary": "separated",
                "official": {"per": 11.0, "financial_health": "official"},
                "reference": {"per": 99.0, "financial_health": "kabutan"}}

    output = process_request(write_request(tmp_path), analyzer, now=NOW)
    item = json.loads(output.read_text(encoding="utf-8"))["analyses"][0]
    assert item["per"] == 11.0
    assert item["financial_health"] == "official"


def test_insufficient_does_not_fill_missing_optional_values(tmp_path):
    def insufficient(**_):
        return {"assessment": "insufficient", "confidence": 0.0, "summary": "missing"}

    item = json.loads(process_request(write_request(tmp_path), insufficient, now=NOW).read_text())["analyses"][0]
    assert item == {"code": "1111", "assessment": "insufficient", "confidence": 0.0,
                    "summary": "missing"}


def test_existing_response_requires_force_and_publish_is_atomic(tmp_path):
    request = write_request(tmp_path)
    output = process_request(request, good_analyzer, now=NOW)
    original = output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        process_request(request, lambda **_: {}, now=NOW)
    assert output.read_text(encoding="utf-8") == original
    process_request(request, good_analyzer, force=True, now=NOW)
    assert not list(tmp_path.glob("*.tmp"))
