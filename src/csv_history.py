from __future__ import annotations

import csv
from pathlib import Path
import warnings

import pandas as pd

LEGACY_SIGNAL_COLUMNS = [
    ["シグナル日", "コード", "会社名", "市場", "現在値", "前日比", "スコア", "ランク", "%K", "%D",
     "%D傾き", "RSI14", "MA25", "MA75", "MA200", "BB位置", "出来高倍率", "ローソク足パターン",
     "アンチ判定", "買い・売り", "損切り候補", "利確候補", "RR", "判定理由", "Yahoo Financeリンク"],
    ["シグナル日", "コード", "会社名", "市場", "現在値", "前日比", "直近3日騰落率", "直近5日騰落率",
     "25日線乖離率", "ATR当日値幅", "スコア", "ランク", "%K", "%D", "%D傾き", "RSI14", "MA25",
     "MA75", "MA200", "BB位置", "出来高倍率", "ローソク足パターン", "アンチ判定", "買い・売り",
     "損切り候補", "利確候補", "RR", "判定理由", "除外理由", "Yahoo Financeリンク"],
]


def read_mixed_csv(
    path: str | Path, schemas: list[list[str]], **kwargs
) -> pd.DataFrame:
    """Read a CSV that may contain headerless rows from several schema versions.

    A normal ``pandas.read_csv`` uses the first record as the header and therefore
    cannot recover when an append-only file later gains columns.  Parse records
    independently instead: explicit headers select their schema, while legacy
    headerless records are identified by their field count.  Unidentifiable
    records are skipped without discarding the rest of the history.
    """
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.stat().st_size:
        return pd.DataFrame()

    known = {len(schema): schema for schema in schemas}
    active_schema: list[str] | None = None
    records: list[dict] = []
    bad_lines: list[int] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for line_number, values in enumerate(csv.reader(source), 1):
            if not values or not any(value.strip() for value in values):
                continue
            if values[0].strip() == "シグナル日":
                active_schema = values
                continue
            schema = active_schema if active_schema and len(active_schema) == len(values) else known.get(len(values))
            if schema is None:
                bad_lines.append(line_number)
                continue
            records.append(dict(zip(schema, values)))

    if bad_lines:
        warnings.warn(
            f"{csv_path}: 列構成を判別できない {len(bad_lines)} 行をスキップしました "
            f"(行番号: {', '.join(map(str, bad_lines[:10]))})",
            RuntimeWarning,
            stacklevel=2,
        )
    frame = pd.DataFrame.from_records(records)
    dtype = kwargs.pop("dtype", None)
    if dtype:
        for column, value_type in dtype.items():
            if column in frame:
                frame[column] = frame[column].astype(value_type)
    if kwargs:
        raise TypeError(f"未対応の読み込みオプション: {', '.join(kwargs)}")
    return frame


def write_merged_csv(path: str | Path, old: pd.DataFrame, new: pd.DataFrame) -> None:
    """Atomically rewrite history using the union of old and new columns."""
    csv_path = Path(path)
    columns = list(new.columns) + [column for column in old.columns if column not in new.columns]
    merged = pd.concat([old, new], ignore_index=True).reindex(columns=columns)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    merged.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(csv_path)
