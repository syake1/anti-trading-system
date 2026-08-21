import pytest

from src.jquants_source import (
    JQUANTS_SUMMARY,
    JQuantsError,
    acquire_forecast,
    derive_forecast,
    fetch_statements,
    get_api_key,
)


class _Response:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
    with pytest.raises(JQuantsError, match="JQUANTS_API_KEY未設定"):
        get_api_key()


def test_v2_summary_uses_api_key_and_data_response():
    session = _Session(_Response({"data": [
        {"DiscDate": "2026-01-20", "Code": "86970"},
        {"DiscDate": "2026-04-20", "Code": "86970"},
    ]}))
    result = fetch_statements("8697", "secret-value", session)
    assert [row["DiscDate"] for row in result] == ["2026-04-20", "2026-01-20"]
    url, kwargs = session.calls[0]
    assert url == JQUANTS_SUMMARY
    assert kwargs["params"] == {"code": "8697"}
    assert kwargs["headers"] == {"x-api-key": "secret-value"}


def test_v2_forecast_and_revision_are_derived_from_official_values():
    result = derive_forecast([
        {"DiscDate": "2026-04-20", "CurFYEn": "2027-03-31", "Sales": "100",
         "OP": "10", "FSales": "120", "FOP": "15", "FNP": "8"},
        {"DiscDate": "2026-01-20", "CurFYEn": "2027-03-31",
         "FSales": "115", "FOP": "12", "FNP": "7"},
    ])
    assert result["company_forecast"] == "増収増益"
    assert result["revision"] == "上方修正"
    assert result["important_disclosure"] == ""


def test_acquire_forecast_uses_environment_api_key(monkeypatch):
    monkeypatch.setenv("JQUANTS_API_KEY", "configured-key")
    session = _Session(_Response({"data": [
        {"DiscDate": "2026-04-20", "CurFYEn": "2027-03-31", "Sales": "100",
         "OP": "10", "FSales": "120", "FOP": "15", "FNP": "8"},
    ]}))
    result = acquire_forecast("8697", session)
    assert result["source"] == "J-Quants"
    assert session.calls[0][1]["headers"] == {"x-api-key": "configured-key"}
