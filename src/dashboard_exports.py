"""Defensive data adapters used only by the Streamlit review dashboard."""
from datetime import date
import json
from pathlib import Path
import re

import pandas as pd

NO_RECORD = "まだ記録がありません"


def _read_report_table(path: Path) -> pd.DataFrame:
    """Read the audit CSV embedded in a meeting markdown report, fail-open."""
    try:
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"```csv\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if not blocks:
            return pd.DataFrame()
        from io import StringIO
        return pd.read_csv(StringIO(blocks[-1]), dtype={"コード": "string"})
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


def load_meeting_reports(directory: Path) -> pd.DataFrame:
    """Load persisted morning/recheck audit reports; broken reports are skipped."""
    rows = []
    for path in sorted(directory.glob("**/*.md")):
        if not path.stem.startswith(("morning_meeting_", "morning_recheck_")):
            continue
        match = re.search(r"(\d{8})", path.stem)
        table = _read_report_table(path)
        if not match or table.empty or "コード" not in table:
            continue
        table = table.copy()
        table["会議日"] = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce").date().isoformat()
        table["会議種別"] = "再確認" if "recheck" in path.stem else "朝会"
        table["記録元"] = str(path)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def latest_meeting_view(history: pd.DataFrame, stocknote: pd.DataFrame | None = None) -> dict:
    """Summarise the latest saved decisions without inventing meeting prose."""
    if history.empty or "会議日" not in history:
        return {}
    latest = history[history["会議日"] == history["会議日"].max()].copy()
    if latest.empty:
        return {}
    adopted = latest[latest.get("最終判断", pd.Series("", index=latest.index)).isin(["主力", "小口"])]
    focus = adopted if not adopted.empty else latest.head(10)
    def joined(column):
        if column not in focus:
            return NO_RECORD
        values = [str(value).strip() for value in focus[column].dropna() if str(value).strip()]
        return "\n\n".join(dict.fromkeys(values)) or NO_RECORD
    notes = []
    if stocknote is not None and not stocknote.empty and "code" in stocknote:
        codes = set(focus["コード"].astype(str))
        for _, note in stocknote[stocknote["code"].astype(str).isin(codes)].iterrows():
            parts = [str(note.get(key, "")).strip() for key in ("assessment", "summary", "cautions")]
            text = " / ".join(part for part in parts if part and part.lower() != "nan")
            if text:
                notes.append(f'{note.get("code")}: {text}')
    decisions = focus.get("最終判断", pd.Series(dtype=str)).fillna("").astype(str)
    # Neither persistence schema has explicit agreement/conflict fields. Do not
    # infer them from independent comments and present that inference as fact.
    common = NO_RECORD
    conflict = NO_RECORD
    order_columns = [name for name in ("コード", "銘柄名", "最終判断", "注文方式", "買いゾーン下限", "買いゾーン上限",
                                      "逆指値発動価格", "損切り価格", "利確目標", "推奨株数", "RR") if name in focus]
    return {"date": latest["会議日"].iloc[0], "source": latest["記録元"].iloc[0],
            "system_opinion": joined("分析コメント"), "stocknote_opinion": "\n\n".join(notes) or NO_RECORD,
            "agreement": common, "conflict": conflict,
            "decision": " / ".join(f"{key}: {value}件" for key, value in decisions.value_counts().items()) or NO_RECORD,
            "reason": joined("注文理由"), "orders": focus[order_columns],
            "risk": joined("運用コメント")}


def meeting_history_summary(history: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate saved meetings and only outcomes with matching date/code keys."""
    columns = ["日付", "候補数", "採用数", "見送り数", "翌日平均", "翌日勝率", "3日後平均", "3日後勝率", "5日後平均", "5日後勝率", "最終判断"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    data = history.copy()
    # Prefer the final recheck when both report types exist for one day/code.
    data["_order"] = data.get("会議種別", "朝会").map({"朝会": 0, "再確認": 1}).fillna(0)
    data = data.sort_values("_order").drop_duplicates(["会議日", "コード"], keep="last")
    perf = performance.copy()
    if not perf.empty and {"シグナル日", "コード"}.issubset(perf):
        perf["シグナル日"] = perf["シグナル日"].astype(str)
        perf["コード"] = perf["コード"].astype(str).str.replace(r"\.0$", "", regex=True)
        data = data.merge(perf, left_on=["会議日", "コード"], right_on=["シグナル日", "コード"], how="left", suffixes=("", "_実績"))
    rows = []
    for day, group in data.groupby("会議日", sort=True):
        decisions = group.get("最終判断", pd.Series("", index=group.index)).fillna("").astype(str)
        row = {"日付": day, "候補数": group["コード"].nunique(), "採用数": decisions.isin(["主力", "小口"]).sum(),
               "見送り数": (decisions == "見送り").sum(),
               "最終判断": " / ".join(f"{key} {value}件" for key, value in decisions.value_counts().items()) or NO_RECORD}
        for label, column in (("翌日", "1日後騰落率"), ("3日後", "3日後騰落率"), ("5日後", "5日後騰落率")):
            values = pd.to_numeric(group.get(column), errors="coerce").dropna() if column in group else pd.Series(dtype=float)
            row[f"{label}平均"] = values.mean() if not values.empty else pd.NA
            row[f"{label}勝率"] = (values > 0).mean() if not values.empty else pd.NA
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("日付", ascending=False)


def read_candidate_csv(source) -> tuple[pd.DataFrame, str | None]:
    """Read an uploaded/path CSV without letting an empty or broken file stop the UI."""
    try:
        frame = pd.read_csv(source, dtype={"コード": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, OSError, ValueError) as exc:
        return pd.DataFrame(), f"候補CSVを読み込めませんでした: {exc}"
    if "コード" not in frame:
        return pd.DataFrame(), "候補CSVに必須列「コード」がありません。"
    frame["コード"] = (frame["コード"].astype("string").str.strip()
                         .str.replace(r"\.0$", "", regex=True).fillna(""))
    frame = frame[frame["コード"] != ""].copy()
    return frame, None


def ranked_buy_candidates(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return the best buy row per code; scanner score determines ranking."""
    if frame.empty or "コード" not in frame:
        return pd.DataFrame(columns=frame.columns)
    result = frame.copy()
    if "買い・売り" in result:
        result = result[result["買い・売り"].fillna("").astype(str).eq("買い")]
    result["コード"] = result["コード"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    result["_score"] = pd.to_numeric(result.get("スコア", 0), errors="coerce").fillna(0)
    result["_date"] = pd.to_datetime(result.get("シグナル日", pd.NaT), errors="coerce")
    result = (result.sort_values(["_score", "_date", "コード"], ascending=[False, False, True], kind="stable")
              .drop_duplicates("コード", keep="first").head(limit).reset_index(drop=True))
    result.insert(0, "順位", range(1, len(result) + 1))
    return result.drop(columns=["_score", "_date"])


def candidate_selector_labels(ranked: pd.DataFrame) -> dict[str, str]:
    """Build unambiguous rank/code/name labels for the candidate selector."""
    labels = {}
    for _, row in ranked.iterrows():
        code = re.sub(r"\.0$", "", str(row.get("コード", "")))
        name = row.get("会社名", row.get("銘柄名", ""))
        name = "" if pd.isna(name) else str(name).strip()
        rank = row.get("順位", "—")
        labels[code] = f"{rank}位｜{code}｜{name or '銘柄名なし'}"
    return labels


def candidate_detail_rows(
    ranked: pd.DataFrame,
    meeting: pd.DataFrame,
    stocknote: pd.DataFrame,
    performance: pd.DataFrame,
    code: str,
) -> tuple[pd.Series, pd.Series | None, pd.DataFrame, pd.DataFrame]:
    """Return every code-scoped detail source for one selected candidate."""
    normalized = re.sub(r"\.0$", "", str(code))

    def matching(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        if frame.empty or column not in frame:
            return pd.DataFrame()
        values = frame[column].astype(str).str.replace(r"\.0$", "", regex=True)
        return frame[values.eq(normalized)]

    candidates = matching(ranked, "コード")
    if candidates.empty:
        raise KeyError(f"ランキングに存在しない銘柄コードです: {normalized}")
    plans = matching(meeting, "コード")
    notes = matching(stocknote, "code")
    outcomes = performance_for_code(performance, normalized)
    return candidates.iloc[0], (plans.iloc[0] if not plans.empty else None), notes, outcomes


def performance_for_code(performance: pd.DataFrame, code: str) -> pd.DataFrame:
    """Select the requested outcome horizons while tolerating old CSV schemas."""
    wanted = ["シグナル日", "コード", "ランク", "シグナル種別", "1日後騰落率", "3日後騰落率", "5日後騰落率",
              "利確到達", "損切り到達"]
    if performance.empty or "コード" not in performance:
        return pd.DataFrame(columns=wanted)
    selected = performance[performance["コード"].astype(str).str.replace(r"\.0$", "", regex=True) == str(code)]
    return selected.reindex(columns=wanted).sort_values("シグナル日", ascending=False, na_position="last")


def strategy_performance(performance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1/3/5-day results by strategy for an apples-to-apples display."""
    if performance.empty or "シグナル種別" not in performance:
        return pd.DataFrame(columns=["戦略", "件数", "1日平均", "3日平均", "5日平均", "5日勝率"])
    data = performance.copy()
    # Old files left strategy blank but retained direction. Calling these rows
    # "未分類" concealed that the signed returns are short-strategy results.
    fallback = data.get("買い・売り", pd.Series("", index=data.index)).fillna("").astype(str)
    data["シグナル種別"] = data["シグナル種別"].fillna("").astype(str).str.strip()
    data.loc[data["シグナル種別"].eq(""), "シグナル種別"] = fallback.map({"買い": "買い（旧形式）", "売り": "売り（旧形式）"}).fillna("未分類")
    for column in ("1日後騰落率", "3日後騰落率", "5日後騰落率"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    rows = []
    for name, group in data.groupby("シグナル種別", dropna=False):
        five = group["5日後騰落率"].dropna()
        rows.append({"戦略": name, "件数": len(group), "1日平均": group["1日後騰落率"].mean(),
                     "3日平均": group["3日後騰落率"].mean(), "5日平均": five.mean(),
                     "5日勝率": (five > 0).mean() if not five.empty else float("nan")})
    return pd.DataFrame(rows).sort_values(["5日平均", "件数"], ascending=[False, False], na_position="last")


def latest_stocknote_analysis(directory: Path) -> tuple[pd.DataFrame, str]:
    """Load namespaced advisory values from the newest response, fail-open."""
    files = sorted(directory.glob("stocknote_response_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        return pd.DataFrame(), "stocknote分析は未取得です。"
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        analyses = payload.get("analyses", [])
        if not isinstance(analyses, list):
            raise ValueError("analysesが配列ではありません")
        frame = pd.DataFrame(analyses)
        if "code" not in frame:
            return pd.DataFrame(), "最新のstocknote応答に分析結果がありません。"
        frame["code"] = frame["code"].astype(str)
        return frame, f"参考データ: {files[0].name}"
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return pd.DataFrame(), f"stocknote参考データを表示できません: {exc}"


def csv_download_data(frame: pd.DataFrame) -> bytes:
    """Return an Excel-friendly UTF-8 CSV, including its BOM."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def dated_csv_filename(stem: str, day: date | None = None) -> str:
    """Build a stable, filesystem-safe CSV filename containing a date."""
    return f"{stem}_{(day or date.today()):%Y%m%d}.csv"


def filter_meeting_history(
    history: pd.DataFrame,
    dates: list[str] | None = None,
    ranks: list[str] | None = None,
    query: str = "",
) -> pd.DataFrame:
    """Apply the dashboard's history filters without mutating source data."""
    filtered = history.copy()
    if dates and "シグナル日" in filtered:
        filtered = filtered[filtered["シグナル日"].astype(str).isin(dates)]
    if ranks and "ランク" in filtered:
        filtered = filtered[filtered["ランク"].astype(str).isin(ranks)]
    query = query.strip()
    if query:
        code = filtered.get("コード", pd.Series("", index=filtered.index)).astype(str)
        name = filtered.get("会社名", pd.Series("", index=filtered.index)).astype(str)
        filtered = filtered[code.str.contains(query, case=False, regex=False) |
                            name.str.contains(query, case=False, regex=False)]
    return filtered
