"""Defensive data adapters used only by the Streamlit review dashboard."""
from datetime import date
import json
from pathlib import Path

import pandas as pd


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
    data["シグナル種別"] = data["シグナル種別"].fillna("未分類").replace("", "未分類")
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
