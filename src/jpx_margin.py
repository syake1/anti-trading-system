"""JPX「銘柄別信用取引週末残高」の取得と候補への付与。"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests


DEFAULT_PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/margin/05.html"
DATA_COLUMNS = [
    "売残", "買残", "売残前週比", "買残前週比", "信用倍率", "基準日", "取得元URL", "取得日時"
]
NO_DATA = "データなし"


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def _links(html: str, page_url: str) -> list[str]:
    """Return spreadsheet links in page order, without depending on JPX CSS."""
    found = re.findall(r'''href=["']([^"']+\.(?:xlsx?|csv)(?:\?[^"']*)?)["']''', html, re.I)
    return list(dict.fromkeys(urljoin(page_url, link) for link in found))


def _clean(value: object) -> str:
    return re.sub(r"[\s　]+", "", str(value)).replace("（", "(").replace("）", ")")


def _column_index(row: pd.Series, aliases: tuple[str, ...]) -> int | None:
    for index, value in enumerate(row):
        text = _clean(value)
        if any(alias in text for alias in aliases):
            return index
    return None


def _header_columns(raw: pd.DataFrame, end: int) -> list[str]:
    start = max(0, end - 2)
    block = raw.iloc[start:end + 1].ffill(axis=1)
    return [
        _clean(" ".join(str(value) for value in block.iloc[:, column] if pd.notna(value)))
        for column in range(raw.shape[1])
    ]


def _balance_columns(headers: list[str]) -> tuple[int | None, int | None]:
    sells = [i for i, text in enumerate(headers) if "売残高" in text or text.endswith("売残")]
    buys = [i for i, text in enumerate(headers) if "買残高" in text or text.endswith("買残")]
    total_sells = [i for i in sells if "合計" in headers[i]]
    total_buys = [i for i in buys if "合計" in headers[i]]
    # JPXの多段見出しでは制度・一般・合計の順。合計と明記された列だけを優先し、
    # 単一ペアの場合だけ単純形式として受け入れる（複数ペアからの推測はしない）。
    if total_sells and total_buys:
        return total_sells[-1], total_buys[-1]
    if len(sells) == len(buys) == 1:
        return sells[0], buys[0]
    return None, None


def parse_workbook(content: bytes, source_url: str) -> tuple[str, pd.DataFrame]:
    """Parse a JPX workbook defensively; schema changes raise instead of guessing."""
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, dtype=object)
    for raw in sheets.values():
        for header_pos in range(min(30, len(raw))):
            headers = _header_columns(raw, header_pos)
            code_i = next((i for i, text in enumerate(headers) if "銘柄コード" in text or text == "コード"), None)
            sell_i, buy_i = _balance_columns(headers)
            if None in (code_i, sell_i, buy_i) or len({code_i, sell_i, buy_i}) != 3:
                continue
            date = _reference_date(raw.iloc[:header_pos + 1], source_url)
            rows = []
            for _, values in raw.iloc[header_pos + 1:].iterrows():
                code_match = re.fullmatch(r"(\d{4})0?", _clean(values.iloc[code_i]).split(".")[0])
                if not code_match:
                    continue
                sell = pd.to_numeric(values.iloc[sell_i], errors="coerce")
                buy = pd.to_numeric(values.iloc[buy_i], errors="coerce")
                if pd.isna(sell) or pd.isna(buy):
                    continue
                rows.append({"コード": code_match.group(1), "売残": int(sell), "買残": int(buy)})
            if rows and date:
                return date, pd.DataFrame(rows).drop_duplicates("コード", keep="last")
    raise ValueError("JPXファイル形式変更: 必須列または基準日を認識できません")


def _reference_date(header: pd.DataFrame, source_url: str) -> str | None:
    text = " ".join(str(value) for value in header.to_numpy().ravel() if pd.notna(value))
    candidates = re.findall(r"(20\d{2})[年/.-]\s*(\d{1,2})[月/.-]\s*(\d{1,2})日?", text)
    if not candidates:
        candidates = re.findall(r"(20\d{2})(\d{2})(\d{2})", source_url)
    for year, month, day in candidates:
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except ValueError:
            pass
    return None


def fetch_weekly_margin(page_url: str = DEFAULT_PAGE_URL, timeout: float = 20) -> tuple[pd.DataFrame, str | None]:
    """Fetch latest and previous official files and calculate week-over-week values."""
    retrieved_at = _now()
    session = requests.Session()
    response = session.get(page_url, timeout=timeout)
    response.raise_for_status()
    links = _links(response.text, page_url)
    parsed: list[tuple[str, pd.DataFrame, str]] = []
    errors = []
    for link in links:
        try:
            file_response = session.get(link, timeout=timeout)
            file_response.raise_for_status()
            date, frame = parse_workbook(file_response.content, link)
            parsed.append((date, frame, link))
        except Exception as exc:
            errors.append(f"{link}: {exc}")
        if len({date for date, _, _ in parsed}) >= 2:
            break
    by_date = {date: (frame, link) for date, frame, link in parsed}
    dates = sorted(by_date, reverse=True)
    if len(dates) < 2:
        detail = "; ".join(errors[:3]) or "対象ファイルが2週分ありません"
        raise ValueError(f"JPX最新週・前週を取得できません: {detail}")
    latest_date, previous_date = dates[:2]
    latest, source_url = by_date[latest_date]
    previous, _ = by_date[previous_date]
    result = latest.merge(previous, on="コード", how="left", suffixes=("", "_前週"))
    result["売残前週比"] = result["売残"] - result["売残_前週"]
    result["買残前週比"] = result["買残"] - result["買残_前週"]
    result["信用倍率"] = result.apply(
        lambda row: "算出不能・売残0" if row["売残"] == 0 else round(row["買残"] / row["売残"], 2), axis=1
    )
    result["基準日"] = latest_date
    result["取得元URL"] = source_url
    result["取得日時"] = retrieved_at
    return result[["コード", *DATA_COLUMNS]], None


def enrich_candidates(candidates: pd.DataFrame, config: dict | None = None, root: Path | None = None) -> pd.DataFrame:
    """Attach official data; every failure/missing code becomes explicit dataなし."""
    result = candidates.copy()
    if result.empty:
        for column in DATA_COLUMNS:
            result[column] = pd.Series(dtype=object)
        return result
    settings = (config or {}).get("jpx_margin", {})
    try:
        margin, _ = fetch_weekly_margin(
            settings.get("page_url", DEFAULT_PAGE_URL), float(settings.get("timeout", 20))
        )
        result["コード"] = result["コード"].astype(str).str.zfill(4)
        result = result.drop(columns=[c for c in DATA_COLUMNS if c in result], errors="ignore").merge(
            margin, on="コード", how="left"
        )
    except Exception as exc:
        print(f"JPX信用取引データ取得失敗 → データなしで続行: {exc}")
        for column in DATA_COLUMNS:
            result[column] = NO_DATA
    for column in DATA_COLUMNS:
        result[column] = result[column].where(result[column].notna(), NO_DATA)
    if root is not None:
        audit = root / "data" / "jpx_margin_balances.csv"
        audit.parent.mkdir(parents=True, exist_ok=True)
        result[["コード", *DATA_COLUMNS]].to_csv(audit, index=False, encoding="utf-8-sig")
    return result
