"""J-Quants adapter: fills the timely-disclosure fields EDINET cannot provide.

J-Quants is already listed in config.json's official_sources. This module only
reads officially disclosed figures (決算短信 / 業績予想) and derives
comparisons between two explicitly reported values. No scraping, no guessing.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
import requests

JQUANTS_AUTH_REFRESH = "https://api.jquants.com/v1/token/auth_refresh"
JQUANTS_STATEMENTS = "https://api.jquants.com/v1/fins/statements"

_FORECAST_KEYS = ("ForecastNetSales", "ForecastOperatingProfit", "ForecastOrdinaryProfit",
                  "ForecastProfit", "ForecastEarningsPerShare")


class JQuantsError(RuntimeError):
    """An expected, auditable J-Quants acquisition failure."""


def _num(value):
    try:
        return float(str(value).replace(",", "")) if str(value).strip() not in ("", "None") else None
    except (TypeError, ValueError):
        return None


def get_id_token(session=requests) -> str:
    """Exchange the long-lived refresh token (env var) for a short-lived ID token."""
    refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN", "")
    if not refresh_token:
        raise JQuantsError("JQUANTS_REFRESH_TOKEN未設定")
    response = session.post(JQUANTS_AUTH_REFRESH, params={"refreshtoken": refresh_token}, timeout=30)
    if not response.ok:
        raise JQuantsError(f"J-Quants認証失敗 HTTP {response.status_code}")
    token = response.json().get("idToken", "")
    if not token:
        raise JQuantsError("J-QuantsIDトークン取得失敗")
    return token


def fetch_statements(code: str, id_token: str, session=requests) -> list[dict]:
    """Return every officially disclosed statement for this security, newest first."""
    jquants_code = f"{code}0"  # J-Quants uses the 5-digit padded code.
    response = session.get(JQUANTS_STATEMENTS, params={"code": jquants_code},
                            headers={"Authorization": f"Bearer {id_token}"}, timeout=30)
    if not response.ok:
        raise JQuantsError(f"J-Quants財務情報取得失敗 HTTP {response.status_code}")
    statements = response.json().get("statements", [])
    return sorted(statements, key=lambda s: str(s.get("DisclosedDate", "")), reverse=True)


def derive_forecast(statements: list[dict]) -> dict:
    """Derive company_forecast/revision/important_disclosure from two explicit filings only."""
    if not statements:
        return {}
    latest = statements[0]
    fiscal_end = latest.get("CurrentFiscalYearEndDate")
    prior = next((s for s in statements[1:]
                  if s.get("CurrentFiscalYearEndDate") == fiscal_end
                  and s.get("DisclosedDate") != latest.get("DisclosedDate")), None)

    latest_forecast = {k: _num(latest.get(k)) for k in _FORECAST_KEYS}
    revenue_f, op_f, profit_f = latest_forecast["ForecastNetSales"], latest_forecast["ForecastOperatingProfit"], latest_forecast["ForecastProfit"]
    revenue_a, op_a = _num(latest.get("NetSales")), _num(latest.get("OperatingProfit"))

    forecast_label = ""
    if revenue_f is not None and op_f is not None and revenue_a is not None and op_a is not None:
        forecast_label = ("増収" if revenue_f >= revenue_a else "減収") + ("増益" if op_f >= op_a else "減益")
    if profit_f is not None and profit_f < 0:
        forecast_label = (forecast_label + "・赤字転落予想").strip("・")

    revision_label = important_disclosure = ""
    if prior is not None:
        prior_forecast = {k: _num(prior.get(k)) for k in _FORECAST_KEYS}
        prior_op, prior_profit = prior_forecast["ForecastOperatingProfit"], prior_forecast["ForecastProfit"]
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
    id_token = get_id_token(session)
    statements = fetch_statements(code, id_token, session)
    if not statements:
        raise JQuantsError("J-Quants財務情報0件")
    data = derive_forecast(statements)
    data.update({"source": "J-Quants", "source_reference": "https://jpx-jquants.com/",
                 "acquired_at": datetime.now(timezone.utc).isoformat()})
    return data
