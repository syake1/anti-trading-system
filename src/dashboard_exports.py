"""CSV export helpers used only by the Streamlit review dashboard."""
from datetime import date

import pandas as pd


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
