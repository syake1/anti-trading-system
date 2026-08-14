import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pytest

from src.data_sources import (FetchFailure, Observation, fetch_eia, fetch_instrument,
                              fetch_mof_jgb, normalize_japanese_date)
from src.market_environment import analyze_market


def config(*sources, compare=False):
    return {"data_sources": {"compare_sources": compare, "instruments": {"X": {
        "instrument_type": "yield", "sources": list(sources)}}}}


def source(name, priority, instrument_type="yield"):
    return {"name": name, "enabled": True, "priority": priority,
            "allowed_for_automation": True, "timeout": 1,
            "instrument_type": instrument_type, "adapter": name}


def good(value=1.5):
    def adapter(_source, _session):
        return Observation(value, "2026-08-14", "percent", pd.Series([1.4, value])), 200
    return adapter


@pytest.mark.parametrize("maturity,expected", [("2年", 0.812), ("10年", 1.635), ("30年", 3.148)])
def test_mof_jgb_cp932_title_unit_and_maturity_header_fixture(maturity, expected):
    content = (Path(__file__).parent / "fixtures/mof_jgbcm_cp932.csv").read_bytes()

    class FixtureSession:
        @staticmethod
        def get(_url, timeout):
            return type("Response", (), {"content": content, "status_code": 200, "ok": True})()

    observation, status = fetch_mof_jgb(
        {"url": "https://example.test/jgbcm.csv", "timeout": 1,
         "unit": "percent", "value_columns": [maturity]}, FixtureSession)
    assert status == 200
    assert observation.value == pytest.approx(expected)
    assert observation.observation_time == "2026-08-13"


@pytest.mark.parametrize("raw", [
    "2026-08-14",
    "2026/8/14",
    "令和8年8月14日",
    "R8.8.14",
])
def test_japanese_observation_date_is_normalized_to_iso(raw):
    assert normalize_japanese_date(raw) == "2026-08-14"


def test_reiwa_first_year_is_2019():
    assert normalize_japanese_date("令和元年5月1日") == "2019-05-01"


def test_mof_jgb_finds_shift_jis_header_when_skiprows_one_is_wrong():
    content = ("財務省公表資料,,,,\n単位：％,,,,\n"
               "基準日,2年,10年,30年\n2026/8/13,0.812,1.635,3.148\n").encode("shift_jis")

    class FixtureSession:
        @staticmethod
        def get(_url, timeout):
            return type("Response", (), {"content": content, "status_code": 200,
                                          "ok": True})()

    observation, _ = fetch_mof_jgb(
        {"url": "https://example.test/jgbcm.csv", "timeout": 1,
         "unit": "percent", "value_columns": ["10年"]}, FixtureSession)
    assert observation.value == pytest.approx(1.635)


def test_mof_jgb_does_not_guess_when_requested_maturity_is_absent():
    content = "資料,,\n基準日,10年\n2026/8/13,1.635\n".encode("cp932")

    class FixtureSession:
        @staticmethod
        def get(_url, timeout):
            return type("Response", (), {"content": content, "status_code": 200,
                                          "ok": True})()

    with pytest.raises(FetchFailure, match="header not found") as caught:
        fetch_mof_jgb(
            {"url": "https://example.test/jgbcm.csv", "timeout": 1,
             "unit": "percent", "value_columns": ["30年"]}, FixtureSession)
    assert caught.value.error_type == "schema_change"


def test_topix_does_not_use_unverified_jpx_download_url():
    configured = json.loads(Path("config.json").read_text(encoding="utf-8"))
    sources = configured["data_sources"]["instruments"]["TOPIX"]["sources"]
    assert [item["name"] for item in sources] == ["Yahoo Finance"]


@pytest.mark.parametrize("kind,status", [("http_429", 429), ("http_403", 403)])
def test_http_failure_falls_back_and_is_audited(tmp_path, kind, status):
    def fail(_source, _session): raise FetchFailure(kind, kind, status)
    audit = tmp_path / "audit.csv"
    result = fetch_instrument("X", config(source("first", 1), source("second", 2)),
                              adapters={"first": fail, "second": good()}, audit_path=audit)
    rows = list(csv.DictReader(audit.open()))
    assert result.value == 1.5
    assert [row["error_type"] for row in rows] == [kind, ""]
    assert rows[0]["http_status"] == str(status) and rows[1]["selected"] == "True"


def test_first_source_success_does_not_contact_fallback(tmp_path):
    def forbidden(*_): raise AssertionError("fallback must not run")
    result = fetch_instrument("X", config(source("first", 1), source("second", 2)),
                              adapters={"first": good(), "second": forbidden}, audit_path=tmp_path / "a.csv")
    assert result.value == 1.5


def test_timeout_falls_back(tmp_path):
    def timeout(*_): raise FetchFailure("timeout", "request timed out")
    result = fetch_instrument("X", config(source("first", 1), source("second", 2)),
                              adapters={"first": timeout, "second": good(2)}, audit_path=tmp_path / "a.csv")
    assert result.value == 2


def test_all_sources_fail_is_missing_and_meeting_analysis_continues(tmp_path):
    def empty(*_): raise FetchFailure("empty_data", "no observations")
    assert fetch_instrument("X", config(source("a", 1), source("b", 2)),
                            adapters={"a": empty, "b": empty}, audit_path=tmp_path / "a.csv") is None
    market_config = {"market_environment": {"indicator_rules": {}, "weights": {},
        "regime_thresholds": {"strong_risk_on": 6, "risk_on": 2, "caution": -2, "strong_risk_off": -6},
        "capital_ratios": {"強いリスクオン": 1, "リスクオン": 1, "中立": .75, "警戒": .5, "強いリスクオフ": .25}}}
    assert analyze_market([], market_config, "fixed").regime == "中立"


def test_different_instrument_definition_is_never_mixed(tmp_path):
    result = fetch_instrument("X", config(source("future", 1, "futures"), source("spot", 2)),
                              adapters={"future": good(99), "spot": good(1)}, audit_path=tmp_path / "a.csv")
    assert result.value == 1
    assert "instrument_mismatch" in (tmp_path / "a.csv").read_text()


def test_comparison_difference_and_deterministic_selection(tmp_path):
    cfg = config(source("official", 1), source("auxiliary", 2), compare=True)
    result = fetch_instrument("X", cfg, adapters={"official": good(1.0), "auxiliary": good(1.2)},
                              audit_path=tmp_path / "a.csv")
    rows = list(csv.DictReader((tmp_path / "a.csv").open()))
    assert result.value == 1.0
    assert float(rows[1]["difference"]) == pytest.approx(.2)


def test_stale_official_source_falls_back_to_yahoo(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
    fresh = datetime.now(timezone.utc).date().isoformat()

    def observation(value, when):
        return lambda *_: (Observation(value, when, "percent", pd.Series([value])), 200)

    cfg = config(source("official", 1), source("yahoo", 2))
    cfg["data_sources"]["max_age_business_days"] = 3
    audit = tmp_path / "audit.csv"
    result = fetch_instrument("X", cfg,
        adapters={"official": observation(1, old), "yahoo": observation(2, fresh)}, audit_path=audit)
    rows = list(csv.DictReader(audit.open()))
    assert result.value == 2
    assert rows[0]["error_type"] == "stale_data"
    assert rows[1]["source"] == "yahoo" and rows[1]["selected"] == "True"


def test_eia_key_is_redacted_from_exception_and_audit(monkeypatch, tmp_path):
    key = "highly-sensitive/key+value"
    monkeypatch.setenv("EIA_API_KEY", key)
    eia = source("eia_api", 1)
    eia.update({"url": "https://example.test/?api_key={api_key}", "unit": "usd",
                "api_key_env": "EIA_API_KEY"})

    class LeakingSession:
        @staticmethod
        def get(url, timeout):
            raise RuntimeError(f"failed request: {url}")

    with pytest.raises(FetchFailure) as caught:
        fetch_eia(eia, LeakingSession)
    assert key not in str(caught.value)
    assert quote(key, safe="") not in str(caught.value)

    audit = tmp_path / "audit.csv"
    assert fetch_instrument("X", config(eia), session=LeakingSession, audit_path=audit) is None
    assert key not in audit.read_text()


def test_eia_key_encoded_with_lowercase_escapes_is_redacted(monkeypatch):
    key = "Secret/With Spaces+"
    monkeypatch.setenv("EIA_API_KEY", key)
    eia = source("eia_api", 1)
    eia.update({"url": "https://example.test/?api_key={api_key}", "unit": "usd"})

    class EncodedLeakSession:
        @staticmethod
        def get(_url, timeout):
            encoded = quote(key, safe="").lower()
            raise RuntimeError(f"request failed for credential={encoded}")

    with pytest.raises(FetchFailure) as caught:
        fetch_eia(eia, EncodedLeakSession)
    assert key.lower() not in str(caught.value).lower()
    assert quote(key, safe="").lower() not in str(caught.value).lower()


def test_audit_redacts_key_from_unexpected_custom_adapter(monkeypatch, tmp_path):
    key = "audit-secret/value"
    monkeypatch.setenv("EIA_API_KEY", key)
    eia = source("eia_api", 1)
    eia.update({"url": "https://example.test/?api_key={api_key}", "unit": "usd"})

    def broken_adapter(*_):
        raise RuntimeError(f"unexpected failure at {quote(key, safe='')}")

    audit = tmp_path / "audit.csv"
    assert fetch_instrument("X", config(eia), adapters={"eia_api": broken_adapter},
                            audit_path=audit) is None
    text = audit.read_text()
    assert key not in text
    assert quote(key, safe="") not in text


def test_missing_eia_secret_is_a_normal_source_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    eia = source("eia_api", 1)
    eia.update({"url": "https://example.test/?api_key={api_key}", "unit": "usd"})
    audit = tmp_path / "audit.csv"
    assert fetch_instrument("X", config(eia), audit_path=audit) is None
    row = next(csv.DictReader(audit.open()))
    assert row["error_type"] == "missing_secret"
    assert "EIA_API_KEY" not in row["error_message"]
