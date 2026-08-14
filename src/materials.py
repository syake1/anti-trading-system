"""Validated CSV-backed corporate-event models (no inferred event data)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

BUYBACK_COLUMNS = ["発表日", "銘柄コード", "取得上限株数", "取得上限金額", "取得期間開始日", "取得期間終了日",
                   "発行済株式数", "時価総額", "発表時株価", "1日平均売買代金"]
EVENT_COLUMNS = ["発表日", "銘柄コード", "種別", "概要", "情報源URL"]


def ensure_templates(root: Path) -> None:
    data = root / "data"; data.mkdir(parents=True, exist_ok=True)
    for name, columns in (("buybacks.csv", BUYBACK_COLUMNS), ("events.csv", EVENT_COLUMNS)):
        path = data / name
        if not path.exists(): pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def load_buybacks(root: Path) -> pd.DataFrame:
    ensure_templates(root)
    try: df = pd.read_csv(root / "data/buybacks.csv", dtype={"銘柄コード": str})
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=BUYBACK_COLUMNS)
    if df.empty: return df
    for col in BUYBACK_COLUMNS[2:]:
        if "日" not in col and "期間" not in col: df[col] = pd.to_numeric(df[col], errors="coerce")
    start, end = pd.to_datetime(df["取得期間開始日"], errors="coerce"), pd.to_datetime(df["取得期間終了日"], errors="coerce")
    df["時価総額比"] = df["取得上限金額"] / df["時価総額"] * 100
    df["発行済株式比"] = df["取得上限株数"] / df["発行済株式数"] * 100
    df["取得期間日数"] = (end - start).dt.days.add(1).clip(lower=1)
    df["1日想定取得額"] = df["取得上限金額"] / df["取得期間日数"]
    df["売買代金比"] = df["1日想定取得額"] / df["1日平均売買代金"] * 100
    return df


def signals_for(code: str, root: Path, config: dict) -> tuple[dict, list[str]]:
    buybacks = load_buybacks(root)
    bb = buybacks[buybacks["銘柄コード"].astype(str).str.zfill(4) == str(code).zfill(4)]
    flags, reasons = {"自社株買い": False, "決算・上方修正": False}, []
    impact = 0.0
    if not bb.empty:
        row = bb.sort_values("発表日").iloc[-1]
        impact = float(row.get("時価総額比", 0) or 0)
        flags["自社株買い"] = impact >= config["materials"]["buyback_market_cap_pct"] or float(row.get("発行済株式比", 0) or 0) >= config["materials"]["buyback_shares_pct"]
        if flags["自社株買い"]: reasons.append("自社株買いインパクト大")
    events_path = root / "data/events.csv"
    events = pd.read_csv(events_path, dtype={"銘柄コード": str}) if events_path.exists() and events_path.stat().st_size else pd.DataFrame()
    if not events.empty:
        match = events[events["銘柄コード"].astype(str).str.zfill(4) == str(code).zfill(4)]
        flags["決算・上方修正"] = match["種別"].isin(["好決算", "上方修正", "増配"]).any()
        if flags["決算・上方修正"]: reasons.append("好決算・上方修正")
    return {"flags": flags, "buyback_ratio": impact}, reasons
