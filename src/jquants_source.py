"""J-Quants adapter: fills the timely-disclosure fields EDINET cannot provide.

J-Quants is already listed in config.json's official_sources. This module only
reads officially disclosed figures (決算短信 / 業績予想) and derives
comparisons between two explicitly reported values. No scraping, no guessing.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
import requests

JQUANTS_SUMMARY = "https://api.jquants.com/v2/fins/summary"

_FORECAST_KEYS = ("FSales", "FOP", "FOdP", "FNP", "FEPS")


class JQuantsError(RuntimeError):
    """An expected, auditable J-Quants acquisition failure."""


def _num(value):
    try:
        return float(str(value).replace(",", "")) if str(value).strip() not in ("", "None") else None
    except (TypeError, ValueError):
        return None


def get_api_key() -> str:
    """Read the J-Quants V2 API key without ever logging its value."""
    api_key = os.getenv("JQUANTS_API_KEY", "").strip()
    if not api_key:
        raise JQuantsError("JQUANTS_API_KEY未設定")
    return api_key


def fetch_statements(code: str, api_key: str, session=requests) -> list[dict]:
    """Return every officially disclosed statement for this security, newest first."""
    jquants_code = str(code).strip()
    response = session.get(JQUANTS_SUMMARY, params={"code": jquants_code},
                            headers={"x-api-key": api_key}, timeout=30)
    if not response.ok:
        raise JQuantsError(f"J-Quants財務情報取得失敗 HTTP {response.status_code}")
    statements = response.json().get("data", [])
    return sorted(statements, key=lambda s: str(s.get("DiscDate", "")), reverse=True)


def derive_forecast(statements: list[dict]) -> dict:
    """Derive company_forecast/revision/important_disclosure from two explicit filings only."""
    if not statements:
        return {}
    latest = statements[0]
    fiscal_end = latest.get("CurFYEn")
    prior = next((s for s in statements[1:]
                  if s.get("CurFYEn") == fiscal_end
                  and s.get("DiscDate") != latest.get("DiscDate")), None)

    latest_forecast = {k: _num(latest.get(k)) for k in _FORECAST_KEYS}
    revenue_f, op_f, profit_f = latest_forecast["FSales"], latest_forecast["FOP"], latest_forecast["FNP"]
    revenue_a, op_a = _num(latest.get("Sales")), _num(latest.get("OP"))

    forecast_label = ""
    if revenue_f is not None and op_f is not None and revenue_a is not None and op_a is not None:
        forecast_label = ("増収" if revenue_f >= revenue_a else "減収") + ("増益" if op_f >= op_a else "減益")
    if profit_f is not None and profit_f < 0:
        forecast_label = (forecast_label + "・赤字転落予想").strip("・")

    revision_label = important_disclosure = ""
    if prior is not None:
        prior_forecast = {k: _num(prior.get(k)) for k in _FORECAST_KEYS}
        prior_op, prior_profit = prior_forecast["FOP"], prior_forecast["FNP"]
        if op_f is not None and prior_op is not None and op_f != prior_op:
            revision_label = "上方修正" if op_f > prior_op else "下方修正"
        if (prior_op is not None and op_f is not None and prior_op >= 0 > op_f) or \
           (prior_profit is not None and profit_f is not None and prior_profit >= 0 > profit_f):
            important_disclosure = "赤字転落"

    return {
        "company_forecast": forecast_label,
        "revision": revision_label,
        "important_disclosure": important_disclosure,
        "next_earnings_date": "",
    }


def acquire_forecast(code: str, session=requests) -> dict:
    """Single-code convenience wrapper; raises JQuantsError on any failure so the
    caller can audit and fall back to the existing 'insufficient' behavior."""
    api_key = get_api_key()
    statements = fetch_statements(code, api_key, session)
    if not statements:
        raise JQuantsError("J-Quants財務情報0件")
    data = derive_forecast(statements)
    data.update({"source": "J-Quants", "source_reference": "https://jpx-jquants.com/",
                 "acquired_at": datetime.now(timezone.utc).isoformat()})
    return data
