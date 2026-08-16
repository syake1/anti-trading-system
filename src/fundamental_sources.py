"""Minimal, official-only fundamental data acquisition.

The adapters deliberately use only EDINET's API and JPX's distributed workbook.
Missing facts stay missing: no HTML scraping, estimation, or previous-value carry
forward is performed.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import os
from pathlib import Path
import re
from urllib.parse import quote, quote_plus
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.utils import ROOT

EDINET_CODELIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
EDINET_API = "https://api.edinet-fsa.go.jp/api/v2"
AUDIT_COLUMNS = ["acquired_at", "code", "source", "document_id", "source_reference", "status", "reason"]


class AcquisitionError(RuntimeError):
    """An expected, auditable acquisition failure."""


def _safe_error(exc: object, api_key: str = "") -> str:
    text = str(exc).replace("\n", " ")[:500]
    for value in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
        if value:
            text = re.sub(re.escape(value), "***", text, flags=re.IGNORECASE)
    return text


def _response(response):
    if not response.ok:
        raise AcquisitionError(f"HTTP {response.status_code}")
    if not response.content:
        raise AcquisitionError("empty response")
    return response


def cache_official_lists(cache_dir: Path, session=requests, max_age_days: int = 7) -> tuple[Path, Path]:
    """Cache the EDINET code list and JPX listed-company workbook atomically."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = (cache_dir / "Edinetcode.zip", cache_dir / "data_j.xls")
    for path, url in zip(outputs, (EDINET_CODELIST_URL, JPX_LIST_URL)):
        fresh = path.exists() and datetime.now().timestamp() - path.stat().st_mtime < max_age_days * 86400
        if fresh:
            continue
        response = _response(session.get(url, timeout=30))
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return outputs


def load_edinet_code_map(archive: Path) -> dict[str, str]:
    """Return securities-code -> EDINET-code from the official CSV archive."""
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise AcquisitionError("EDINET code-list CSV missing")
        raw = zf.read(names[0])
    frame = None
    for encoding in ("cp932", "utf-8-sig"):
        try:
            candidate = pd.read_csv(BytesIO(raw), encoding=encoding, dtype=str)
            frame = candidate
            break
        except UnicodeDecodeError:
            pass
    if frame is None:
        raise AcquisitionError("EDINET code-list encoding unsupported")
    edinet_col = next((c for c in frame if "ＥＤＩＮＥＴコード" in c or "EDINETコード" in c.upper()), None)
    security_col = next((c for c in frame if "証券コード" in c), None)
    if not edinet_col or not security_col:
        raise AcquisitionError("EDINET code-list schema changed")
    result = {}
    for security, edinet in zip(frame[security_col], frame[edinet_col]):
        digits = re.sub(r"\D", "", str(security))
        if len(digits) >= 4 and str(edinet).strip() not in ("", "nan"):
            result[digits[:4]] = str(edinet).strip()
    return result


def load_jpx_codes(workbook: Path) -> set[str]:
    """Validate codes against JPX's official distributed listed-company file."""
    frame = pd.read_excel(workbook, dtype=str)
    column = next((c for c in frame if "コード" in str(c)), None)
    if column is None:
        raise AcquisitionError("JPX workbook schema changed")
    return {m.group(1) for value in frame[column] if (m := re.match(r"(\d{4})", str(value).strip()))}


def _api_get(path: str, api_key: str, session=requests, **params):
    # EDINET specifies Subscription-Key as a query parameter.  Exceptions are
    # always redacted before they cross this adapter's boundary.
    params["Subscription-Key"] = api_key
    try:
        response = session.get(f"{EDINET_API}/{path}", params=params, timeout=30)
    except requests.RequestException as exc:
        raise AcquisitionError(_safe_error(exc, api_key)) from exc
    return _response(response)


def find_latest_document(edinet_code: str, api_key: str, session=requests,
                         today: date | None = None, lookback_days: int = 400) -> dict:
    """Find the latest annual/semiannual securities report in EDINET results."""
    today = today or datetime.now(timezone.utc).date()
    matches = []
    for offset in range(lookback_days + 1):
        day = today - timedelta(days=offset)
        payload = _api_get("documents.json", api_key, session, date=day.isoformat(), type=2).json()
        for item in payload.get("results", []):
            description = str(item.get("docDescription", ""))
            if (item.get("edinetCode") == edinet_code and item.get("docTypeCode") in {"120", "140", "160"}
                    and any(label in description for label in ("有価証券報告書", "半期報告書", "四半期報告書"))
                    and not item.get("withdrawalStatus") == "1"):
                matches.append(item)
        if matches:  # API is searched newest first.
            return max(matches, key=lambda x: str(x.get("submitDateTime", "")))
    raise AcquisitionError("対象期間に有価証券報告書・半期報告書等なし")


def download_xbrl(document_id: str, api_key: str, session=requests) -> tuple[bytes, str]:
    response = _api_get(f"documents/{document_id}", api_key, session, type=1)
    reference = f"https://disclosure2.edinet-fsa.go.jp/WEEE0030.aspx?bXbrl={document_id}"
    return response.content, reference


CONCEPTS = {
    "revenue": ("Revenue", "Sales", "NetSales", "OperatingRevenue"),
    "operating_profit": ("OperatingIncome",),
    "net_profit": ("ProfitLossAttributableToOwnersOfParent", "ProfitLoss"),
    "eps": ("BasicEarningsLossPerShare",),
    "equity": ("EquityAttributableToOwnersOfParent", "NetAssets"),
    "equity_ratio": ("EquityToAssetRatio", "CapitalAdequacyRatio"),
    "roe": ("RateOfReturnOnEquity", "ReturnOnEquity"),
    "dividend": ("DividendPaidPerShare", "AnnualDividendPerShare"),
    "bps": ("NetAssetsPerShare", "EquityAttributableToOwnersOfParentPerShare"),
    "shares_outstanding": ("NumberOfIssuedSharesAsOfFiscalYearEnd", "TotalNumberOfIssuedShares"),
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_xbrl(archive: bytes) -> dict:
    """Normalize explicitly reported consolidated current/prior facts."""
    with zipfile.ZipFile(BytesIO(archive)) as zf:
        xbrls = [n for n in zf.namelist() if n.lower().endswith(".xbrl") and "audit" not in n.lower()]
        if not xbrls:
            raise AcquisitionError("XBRL instance missing")
        root = ET.fromstring(zf.read(sorted(xbrls, key=len)[0]))
    contexts = {}
    for node in root.iter():
        if _local(node.tag) != "context": continue
        ident = node.attrib.get("id", "")
        text = { _local(child.tag): (child.text or "").strip() for child in node.iter() }
        contexts[ident] = {"start": text.get("startDate"), "end": text.get("endDate"), "instant": text.get("instant"),
                           "consolidated": not any("NonConsolidated" in (member.text or "") for member in node.iter() if _local(member.tag) == "explicitMember")}
    facts = []
    wanted = {name for aliases in CONCEPTS.values() for name in aliases}
    for node in root.iter():
        name, context_id = _local(node.tag), node.attrib.get("contextRef")
        if name not in wanted or not context_id or not contexts.get(context_id, {}).get("consolidated") or node.attrib.get("nil", "false") == "true": continue
        try: value = float((node.text or "").replace(",", ""))
        except ValueError: continue
        facts.append((name, value, contexts[context_id]))
    # Duration facts determine the reporting periods.  Instant-only balance
    # sheet contexts must not accidentally become a "prior period" for flows.
    durations = sorted({(f[2]["start"], f[2]["end"]) for f in facts
                        if f[2].get("start") and f[2].get("end")}, key=lambda period: period[1], reverse=True)
    def period_kind(period):
        days = (date.fromisoformat(period[1]) - date.fromisoformat(period[0])).days
        return "annual" if days >= 300 else "semiannual" if days >= 150 else "quarterly"
    current_period = durations[0] if durations else None
    prior_period = next((period for period in durations[1:]
                         if period_kind(period) == period_kind(current_period)), None) if current_period else None
    current = current_period[1] if current_period else None
    prior = prior_period[1] if prior_period else None
    output = {}
    for key, aliases in CONCEPTS.items():
        for period, suffix in ((current, ""), (prior, "_prior")):
            match = next((value for name, value, ctx in facts if name in aliases and (ctx.get("end") or ctx.get("instant")) == period), None)
            if match is not None: output[key + suffix] = match
    # The parser selected consolidated facts from explicit XBRL contexts and
    # matching current/prior duration endpoints.  These flags are the audit
    # evidence required by the downstream conservative derivations.
    output["comparison_basis_verified"] = bool(current and prior)
    output["payout_basis_verified"] = bool(current and output.get("eps") is not None and output.get("dividend") is not None)
    output["dividend_comparison_verified"] = bool(
        current and prior and output.get("dividend") is not None and output.get("dividend_prior") is not None
    )
    # YoY is valid only when both explicitly reported facts have distinct periods.
    # A negative base is not expressed as an ordinary percentage: the sign
    # transition derived downstream is the meaningful official-fact comparison.
    for source, target in (("revenue", "revenue_yoy"), ("operating_profit", "operating_profit_yoy"), ("net_profit", "ordinary_or_net_profit_yoy")):
        old, new = output.get(source + "_prior"), output.get(source)
        if current and prior and old is not None and old > 0 and new is not None:
            output[target] = (new / old - 1) * 100
    output["latest_earnings_date"] = current or ""
    return output


def acquire(candidates: pd.DataFrame, config: dict, session=requests,
            cache_dir: Path | None = None, audit_path: Path | None = None) -> pd.DataFrame:
    """Acquire candidate fundamentals; isolate and audit every per-code failure."""
    if candidates.empty: return pd.DataFrame()
    key = os.getenv("EDINET_API_KEY", "")
    cache_dir = cache_dir or ROOT / "data/cache/fundamentals"
    audit_path = audit_path or ROOT / config.get("fundamentals", {}).get("audit_path", "data/fundamentals_audit.csv")
    acquired_at, rows, audits = datetime.now(timezone.utc).isoformat(), [], []
    if not key:
        reason, mapping, jpx = "EDINET_API_KEY未設定", {}, set()
    else:
        try:
            code_file, jpx_file = cache_official_lists(cache_dir, session)
            mapping, jpx, reason = load_edinet_code_map(code_file), load_jpx_codes(jpx_file), ""
        except Exception as exc:
            reason, mapping, jpx = _safe_error(exc, key), {}, set()
    for item in candidates.to_dict("records"):
        code = re.sub(r"\D", "", str(item.get("コード", "")))[:4].zfill(4)
        document_id = reference = ""
        try:
            if reason: raise AcquisitionError(reason)
            if code not in jpx: raise AcquisitionError("JPX上場銘柄一覧に証券コードなし")
            if code not in mapping: raise AcquisitionError("EDINETコード対応なし")
            document = find_latest_document(mapping[code], key, session,
                                            lookback_days=int(config.get("fundamentals", {}).get("edinet_lookback_days", 400)))
            document_id = str(document["docID"])
            payload, reference = download_xbrl(document_id, key, session)
            row = parse_xbrl(payload)
            price = pd.to_numeric(item.get("現在値"), errors="coerce")
            if pd.notna(price):
                if row.get("eps") not in (None, 0): row["per"] = float(price) / row["eps"]
                if row.get("bps") not in (None, 0): row["pbr"] = float(price) / row["bps"]
                if row.get("dividend") is not None: row["dividend_yield"] = row["dividend"] / float(price) * 100
                if row.get("shares_outstanding") is not None:
                    row["market_cap"] = float(price) * row["shares_outstanding"]
            row.update({"code": code, "source": "EDINET", "source_reference": reference,
                        "document_id": document_id, "acquired_at": acquired_at,
                        # EDINET does not provide these required timely disclosures.
                        "company_forecast": "", "revision": "", "important_disclosure": "",
                        "next_earnings_date": ""})
            rows.append(row); status, failure = "success", ""
        except Exception as exc:
            status, failure = "failure", _safe_error(exc, key)
        audits.append({"acquired_at": acquired_at, "code": code, "source": "EDINET", "document_id": document_id,
                       "source_reference": reference, "status": status, "reason": failure})
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    exists = audit_path.exists() and audit_path.stat().st_size > 0
    with audit_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS); writer.writeheader() if not exists else None; writer.writerows(audits)
    return pd.DataFrame(rows)
