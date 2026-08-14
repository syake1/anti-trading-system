"""Audited, definition-safe market data fallback.

Only sources explicitly enabled and allowed for automation in ``config.json`` are
contacted.  A quote can fall back only within the same instrument definition.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable
from urllib.parse import quote, quote_plus

import pandas as pd
import requests

from src.utils import ROOT


AUDIT_COLUMNS = ["instrument", "source", "timestamp", "success", "http_status",
                 "error_type", "error_message", "value", "unit", "observation_time",
                 "instrument_type", "selected", "comparison_source", "difference"]


class FetchFailure(Exception):
    def __init__(self, error_type: str, message: str, status: int | None = None):
        super().__init__(message); self.error_type, self.status = error_type, status


def _redact(message: object, secrets: tuple[str, ...]) -> str:
    """Remove raw and URL-encoded secret values from externally supplied text."""
    result = str(message)
    for secret in secrets:
        if not secret:
            continue
        variants = {secret, quote(secret, safe=""), quote_plus(secret)}
        # Percent escapes are case-insensitive and libraries do not agree on case.
        variants.update({value.lower() for value in variants})
        variants.update({value.upper() for value in variants})
        for value in sorted(variants, key=len, reverse=True):
            result = result.replace(value, "***")
    return result


@dataclass
class Observation:
    value: float
    observation_time: str
    unit: str
    history: pd.Series | None = None


@dataclass
class Attempt:
    instrument: str
    source: str
    timestamp: str
    success: bool
    http_status: int | None = None
    error_type: str = ""
    error_message: str = ""
    value: float | None = None
    unit: str = ""
    observation_time: str = ""
    instrument_type: str = ""
    selected: bool = False
    comparison_source: str = ""
    difference: float | None = None


def _failure(response: requests.Response) -> None:
    if response.status_code == 403: raise FetchFailure("http_403", "HTTP 403 Forbidden", 403)
    if response.status_code == 429: raise FetchFailure("http_429", "HTTP 429 Too Many Requests", 429)
    if not response.ok: raise FetchFailure("http_error", f"HTTP {response.status_code}", response.status_code)
    if not response.content: raise FetchFailure("empty_data", "response body is empty", response.status_code)


def _csv_series(text: str, date_names: tuple[str, ...], value_names: tuple[str, ...]) -> pd.Series:
    try: frame = pd.read_csv(StringIO(text))
    except Exception as exc: raise FetchFailure("parse_error", str(exc)) from exc
    if frame.empty: raise FetchFailure("empty_data", "CSV contains no rows")
    date = next((c for c in frame if c.strip().lower() in date_names), None)
    value = next((c for c in frame if c.strip().lower() in value_names), None)
    if date is None or value is None:
        raise FetchFailure("schema_change", f"expected date/value columns; got {list(frame.columns)}")
    values = pd.to_numeric(frame[value].replace({"ND": None, "N/A": None, "-": None}), errors="coerce")
    series = pd.Series(values.values, index=frame[date].astype(str)).dropna()
    if series.empty: raise FetchFailure("empty_data", "no numeric observations")
    return series


def _http_csv(source: dict, session=requests) -> tuple[str, int]:
    try: response = session.get(source["url"], timeout=float(source.get("timeout", 10)))
    except requests.Timeout as exc: raise FetchFailure("timeout", str(exc) or "request timed out") from exc
    except requests.RequestException as exc: raise FetchFailure("network_error", str(exc)) from exc
    _failure(response)
    return response.text, response.status_code


def fetch_fred(source: dict, session=requests) -> tuple[Observation, int]:
    text, status = _http_csv(source, session)
    series = _csv_series(text, ("date", "observation_date"), ("dgs10", "value"))
    return Observation(float(series.iloc[-1]), str(series.index[-1]), "percent", series), status


def fetch_simple_csv(source: dict, session=requests) -> tuple[Observation, int]:
    text, status = _http_csv(source, session)
    series = _csv_series(text, ("date", "observation_date", "年月日", "基準日"),
                         tuple(x.lower() for x in source.get("value_columns", ["close", "value"])))
    return Observation(float(series.iloc[-1]), str(series.index[-1]), source["unit"], series), status


def fetch_eia(source: dict, session=requests) -> tuple[Observation, int]:
    import os
    key = os.getenv(source.get("api_key_env", "EIA_API_KEY"))
    if not key: raise FetchFailure("missing_secret", "missing EIA API credential")
    configured = dict(source); configured["url"] = source["url"].format(api_key=key)
    secrets = (key,)
    try:
        response = session.get(configured["url"], timeout=float(source.get("timeout", 10))); _failure(response)
        payload = response.json(); rows = payload["response"]["data"]
        if not rows: raise FetchFailure("empty_data", "EIA returned no observations", response.status_code)
        values = pd.Series({str(x["period"]): float(x["value"]) for x in rows}).sort_index()
        return Observation(float(values.iloc[-1]), str(values.index[-1]), source["unit"], values), response.status_code
    except FetchFailure as exc:
        # Never propagate an exception object whose text can contain the credential.
        raise FetchFailure(exc.error_type, _redact(exc, secrets), exc.status) from None
    except (KeyError, TypeError) as exc: raise FetchFailure("schema_change", _redact(exc, secrets)) from None
    except (ValueError, requests.JSONDecodeError) as exc: raise FetchFailure("parse_error", _redact(exc, secrets)) from None
    except requests.Timeout as exc: raise FetchFailure("timeout", _redact(exc, secrets) or "request timed out") from None
    except requests.RequestException as exc: raise FetchFailure("network_error", _redact(exc, secrets)) from None
    except Exception as exc: raise FetchFailure("unexpected_error", _redact(exc, secrets)) from None


def fetch_yahoo(source: dict, session=None) -> tuple[Observation, int | None]:
    try:
        import yfinance as yf
        frame = yf.download(source["ticker"], period=source.get("period", "3mo"), interval="1d",
                            progress=False, auto_adjust=False, timeout=float(source.get("timeout", 10)))
        close = frame["Close"]
        if hasattr(close, "columns"): close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        if close.empty: raise FetchFailure("empty_data", "Yahoo returned no closes")
        return Observation(float(close.iloc[-1]), str(close.index[-1]), source["unit"], close), None
    except FetchFailure: raise
    except (KeyError, TypeError) as exc: raise FetchFailure("schema_change", str(exc)) from exc
    except Exception as exc:
        message = str(exc); lowered = message.lower()
        if "429" in message: raise FetchFailure("http_429", message, 429) from exc
        if "403" in message: raise FetchFailure("http_403", message, 403) from exc
        if "timed out" in lowered or "timeout" in lowered: raise FetchFailure("timeout", message) from exc
        raise FetchFailure("parse_error", message) from exc


ADAPTERS: dict[str, Callable] = {"fred_csv": fetch_fred, "official_csv": fetch_simple_csv,
                                 "eia_api": fetch_eia, "yahoo": fetch_yahoo}


def _business_day_age(observation_time: str, today: date | None = None) -> int:
    """Return weekdays elapsed after an observation date (holidays are not inferred)."""
    try:
        observed = pd.to_datetime(observation_time, utc=True).date()
    except (TypeError, ValueError, OverflowError) as exc:
        raise FetchFailure("invalid_observation_time", "observation_time is not a valid date") from exc
    today = today or datetime.now(timezone.utc).date()
    if observed >= today:
        return 0
    return len(pd.bdate_range(observed, today, inclusive="right"))


def save_audit(attempts: list[Attempt], path: Path | None = None) -> None:
    if not attempts: return
    path = path or ROOT / "data/source_audit.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=AUDIT_COLUMNS)
        if not exists: writer.writeheader()
        for attempt in attempts: writer.writerow(asdict(attempt))


def fetch_instrument(instrument: str, config: dict, *, session=requests,
                     adapters: dict[str, Callable] | None = None,
                     audit_path: Path | None = None) -> Observation | None:
    """Try allowed sources by priority and return only an exactly matching definition."""
    source_config = config.get("data_sources", {})
    definitions = source_config.get("instruments", {})
    definition = definitions.get(instrument)
    if not definition: return None
    adapters = adapters or ADAPTERS; attempts: list[Attempt] = []; successful: list[tuple[Attempt, Observation]] = []
    # Defense in depth: even a custom/broken EIA adapter must not write its key to
    # source_audit.csv.  The adapter itself performs the same redaction before an
    # exception can escape to a caller.
    import os
    eia_secrets = tuple(filter(None, (
        os.getenv(source.get("api_key_env", "EIA_API_KEY"))
        for source in definition.get("sources", []) if source.get("adapter") == "eia_api"
    )))
    sources = sorted(definition.get("sources", []), key=lambda x: (int(x["priority"]), x["name"]))
    expected_type = definition["instrument_type"]
    for source in sources:
        if not source.get("enabled") or not source.get("allowed_for_automation"): continue
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        attempt = Attempt(instrument, source["name"], stamp, False, instrument_type=source.get("instrument_type", ""))
        if source.get("instrument_type") != expected_type:
            attempt.error_type, attempt.error_message = "instrument_mismatch", "source definition differs from instrument"
            attempts.append(attempt); continue
        try:
            observation, status = adapters[source["adapter"]](source, session)
            max_age = int(source.get("max_age_business_days",
                                     source_config.get("max_age_business_days", 3)))
            age = _business_day_age(observation.observation_time)
            if age > max_age:
                raise FetchFailure(
                    "stale_data",
                    f"observation is {age} business days old (maximum {max_age})",
                    status,
                )
            attempt.success, attempt.http_status = True, status
            attempt.value, attempt.unit, attempt.observation_time = observation.value, observation.unit, observation.observation_time
            successful.append((attempt, observation))
            if not config.get("data_sources", {}).get("compare_sources", False): break
        except FetchFailure as exc:
            attempt.http_status, attempt.error_type = exc.status, exc.error_type
            attempt.error_message = _redact(exc, eia_secrets)
        except Exception as exc:
            attempt.error_type, attempt.error_message = "unexpected_error", _redact(exc, eia_secrets)
        attempts.append(attempt)
    if successful:
        selected_attempt, selected = successful[0]; selected_attempt.selected = True
        for attempt, other in successful[1:]:
            attempt.comparison_source = selected_attempt.source
            attempt.difference = other.value - selected.value
        # Successful attempts were not appended until now, preserving source order.
        attempts.extend(a for a, _ in successful if a not in attempts)
        attempts.sort(key=lambda a: next((i for i, s in enumerate(sources) if s["name"] == a.source), 999))
        configured_path = source_config.get("audit_path")
        save_audit(attempts, audit_path or (ROOT / configured_path if configured_path else None)); return selected
    configured_path = source_config.get("audit_path")
    save_audit(attempts, audit_path or (ROOT / configured_path if configured_path else None)); return None


def fetch_configured(config: dict, instruments: list[str] | None = None, **kwargs) -> dict[str, Observation]:
    names = instruments or list(config.get("data_sources", {}).get("instruments", {}))
    return {name: value for name in names if (value := fetch_instrument(name, config, **kwargs)) is not None}
