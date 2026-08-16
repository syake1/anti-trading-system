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
SCORE_COLUMNS = [
    "テクニカルスコア", "信用需給による減点", "信用需給スコア", "総合調整後スコア",
    "テクニカル順位", "調整後順位", "信用需給判定", "信用需給判定理由",
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


def _number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _fundamentals_not_worsening(row: pd.Series) -> bool:
    """Require an affirmative fundamental result; missing data is never approval."""
    label = str(row.get("ファンダメンタル評価", row.get("総合判定", ""))).strip()
    score = _number(row.get("ファンダメンタルスコア"))
    return label in {"良好", "普通"} or (score is not None and score >= 6)


def apply_margin_scoring(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add a conservative, auditable margin-balance adjustment and both rankings."""
    result = candidates.copy()
    if result.empty:
        for column in SCORE_COLUMNS:
            result[column] = pd.Series(dtype=object)
        return result

    records = []
    for _, row in result.iterrows():
        technical = _number(row.get("スコア")) or 0.0
        ratio = _number(row.get("信用倍率"))
        buy = _number(row.get("買残"))
        buy_change = _number(row.get("買残前週比"))
        missing = str(row.get("信用倍率", "")).strip() in {NO_DATA, "算出不能・売残0", ""} or ratio is None
        penalty = 0
        reasons: list[str] = []
        if missing:
            judgment = "信用需給は判定不能"
            reasons.append("信用データがデータなし、または売残0で信用倍率を算出不能")
        else:
            if ratio >= 30:
                penalty -= 12
                reasons.append("信用倍率30倍以上: -12")
            elif ratio >= 20:
                penalty -= 8
                reasons.append("信用倍率20倍以上30倍未満: -8")
            elif ratio >= 10:
                penalty -= 5
                reasons.append("信用倍率10倍以上20倍未満: -5")
            elif ratio >= 5:
                penalty -= 2
                reasons.append("信用倍率5倍以上10倍未満: -2")
            else:
                reasons.append("信用倍率5倍未満: 減点なし")

            falling = any((_number(row.get(column)) or 0) < 0 for column in ("直近3日騰落率", "直近5日騰落率"))
            if falling and buy_change is not None and buy_change > 0:
                penalty -= 5
                reasons.append("直近3～5日で株価下落かつ信用買残増加: -5")
            previous_buy = buy - buy_change if buy is not None and buy_change is not None else None
            increase_rate = buy_change / previous_buy if previous_buy is not None and previous_buy > 0 else None
            if increase_rate is not None and increase_rate >= .20:
                penalty -= 3
                reasons.append(f"信用買残増加率{increase_rate:.1%}: -3")

            if ratio >= 30:
                pattern = str(row.get("ローソク足パターン", ""))
                clear_reversal = pattern not in {"", "なし", NO_DATA} and any(
                    word in pattern for word in ("包み", "下ヒゲ", "陽線", "反転")
                )
                volume_confirmed = (_number(row.get("出来高倍率")) or 0) >= 1.3
                fundamentals_ok = _fundamentals_not_worsening(row)
                if clear_reversal and volume_confirmed and fundamentals_ok:
                    judgment = "小口・反転確認後"
                    reasons.append("明確な反転足・出来高増加・ファンダメンタル非悪化を確認")
                else:
                    judgment = "見送り"
                    missing_checks = [name for ok, name in (
                        (clear_reversal, "明確な反転足"), (volume_confirmed, "出来高増加"),
                        (fundamentals_ok, "ファンダメンタル非悪化"),
                    ) if not ok]
                    reasons.append("原則反転確認待ち（未確認: " + "・".join(missing_checks) + "）")
            elif penalty < 0:
                judgment = "注意"
            else:
                judgment = "良好"
        records.append((technical, penalty, technical + penalty, judgment, " / ".join(reasons)))

    result["テクニカルスコア"] = [record[0] for record in records]
    result["信用需給による減点"] = [record[1] for record in records]
    result["信用需給スコア"] = result["信用需給による減点"]
    result["総合調整後スコア"] = [record[2] for record in records]
    result["信用需給判定"] = [record[3] for record in records]
    result["信用需給判定理由"] = [record[4] for record in records]
    if "判定理由" in result:
        original = result["判定理由"].fillna("").astype(str).str.strip()
        margin_reason = result["信用需給判定理由"].map(lambda value: f"信用需給: {value}")
        result["判定理由"] = original.where(original.eq(""), original + " / ") + margin_reason
    result["テクニカル順位"] = result["テクニカルスコア"].rank(method="min", ascending=False).astype(int)
    result["調整後順位"] = result["総合調整後スコア"].rank(method="min", ascending=False).astype(int)
    return result
