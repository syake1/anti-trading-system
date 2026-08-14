from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from src.csv_history import LEGACY_SIGNAL_COLUMNS, read_mixed_csv
from src.utils import ROOT

PERFORMANCE_COLUMNS = [
    "シグナル日", "コード", "ランク", "買い・売り", "RSI14", "出来高倍率", "BB位置",
    "ローソク足パターン", "シグナル種別", *(f"{n}日後終値" for n in (1, 2, 3, 5, 10, 20)),
    *(f"{n}日後騰落率" for n in (1, 2, 3, 5, 10, 20)), "期間中最高値",
    "期間中最安値", "最大上昇率", "最大下落率", "利確到達", "損切り到達", "5日損益率",
]

SIGNAL_COLUMNS = {
    "シグナル日", "コード", "ランク", "買い・売り", "RSI14", "出来高倍率", "BB位置",
    "ローソク足パターン", "現在値", "利確候補", "損切り候補",
}

PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def _to_numeric(values: pd.Series) -> tuple[pd.Series, int]:
    """CSVで使われる装飾を除き、変換不能な（欠損ではない）値も数える。"""
    text = values.astype("string").str.strip()
    missing = text.isna() | text.eq("") | text.str.lower().isin(("nan", "none", "null"))
    cleaned = text.str.replace(",", "", regex=False).str.replace("円", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned.mask(missing), errors="coerce")
    return numeric, int((~missing & numeric.isna()).sum())


def _numeric_scalar(value: object) -> tuple[float, int]:
    numeric, failures = _to_numeric(pd.Series([value], dtype="object"))
    return float(numeric.iloc[0]) if pd.notna(numeric.iloc[0]) else np.nan, failures


def read_csv_if_populated(path: str | Path, **kwargs) -> pd.DataFrame:
    """Return an empty frame when a CSV is absent or contains no parseable rows."""
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.read_text(encoding="utf-8-sig").strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def update() -> Path:
    source, output = ROOT / "data/signal_history.csv", ROOT / "data/performance.csv"
    # 履歴は追加時点のスキャナー版により列数が異なるため、行単位で版を判別する。
    from src.scanner import RESULT_COLUMNS
    signals = read_mixed_csv(source, [*LEGACY_SIGNAL_COLUMNS, RESULT_COLUMNS], dtype={"コード": str})
    if signals.empty or not SIGNAL_COLUMNS.issubset(signals.columns):
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=PERFORMANCE_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
        print("バックテスト: スキップ件数=0, 数値変換失敗件数=0, 正常評価件数=0")
        return output
    signals = signals.drop_duplicates(["シグナル日", "コード", "買い・売り"])
    records = []
    skipped = conversion_failures = evaluated = 0
    for row in signals.to_dict("records"):
        current, failed = _numeric_scalar(row.get("現在値"))
        take_profit, take_failed = _numeric_scalar(row.get("利確候補"))
        stop_loss, stop_failed = _numeric_scalar(row.get("損切り候補"))
        conversion_failures += failed + take_failed + stop_failed
        if not np.isfinite(current) or current == 0 or not np.isfinite(take_profit) or not np.isfinite(stop_loss):
            skipped += 1
            continue
        signal_date = pd.to_datetime(row.get("シグナル日"), errors="coerce")
        if pd.isna(signal_date):
            skipped += 1
            continue
        prices = yf.download(f'{row["コード"]}.T', start=row["シグナル日"],
                             end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
        if isinstance(prices.columns, pd.MultiIndex): prices.columns = prices.columns.get_level_values(0)
        available_price_columns = [column for column in PRICE_COLUMNS if column in prices]
        if not {"High", "Low", "Close"}.issubset(available_price_columns):
            skipped += 1
            continue
        for column in available_price_columns:
            prices[column], failures = _to_numeric(prices[column])
            conversion_failures += failures
        # 壊れた価格行だけを除外し、後続の有効な取引日は引き続き評価する。
        prices = prices.dropna(subset=["High", "Low", "Close"])
        future = prices.loc[prices.index > signal_date].head(20)
        if future.empty:
            skipped += 1
            continue
        direction = 1 if row["買い・売り"] == "買い" else -1
        rec = {"シグナル日": row["シグナル日"], "コード": row["コード"], "ランク": row["ランク"], "買い・売り": row["買い・売り"],
               "RSI14": row["RSI14"], "出来高倍率": row["出来高倍率"], "BB位置": row["BB位置"], "ローソク足パターン": row["ローソク足パターン"],
               "シグナル種別": row.get("シグナル種別", "アンチ")}
        for n in (1, 2, 3, 5, 10, 20):
            close = future.Close.iloc[n-1] if len(future) >= n else np.nan
            rec[f"{n}日後終値"] = close
            rec[f"{n}日後騰落率"] = direction * (close / current - 1) * 100 if pd.notna(close) else np.nan
        rec.update({"期間中最高値": future.High.max(), "期間中最安値": future.Low.min(),
                    "最大上昇率": (future.High.max()/current-1)*100, "最大下落率": (future.Low.min()/current-1)*100,
                    "利確到達": bool((future.High >= take_profit).any() if direction == 1 else (future.Low <= take_profit).any()),
                    "損切り到達": bool((future.Low <= stop_loss).any() if direction == 1 else (future.High >= stop_loss).any()),
                    "5日損益率": rec["5日後騰落率"]})
        records.append(rec)
        evaluated += 1
    pd.DataFrame(records, columns=PERFORMANCE_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"バックテスト: スキップ件数={skipped}, 数値変換失敗件数={conversion_failures}, 正常評価件数={evaluated}")
    return output


def summary(path=ROOT / "data/performance.csv") -> dict:
    df = read_csv_if_populated(path)
    if df.empty: return {"全シグナル数": 0}
    arithmetic_columns = [
        "RSI14", "出来高倍率", "5日損益率", "最大上昇率", "最大下落率",
        *(f"{n}日後終値" for n in (1, 2, 3, 5, 10, 20)),
        *(f"{n}日後騰落率" for n in (1, 2, 3, 5, 10, 20)),
        "期間中最高値", "期間中最安値",
    ]
    for column in arithmetic_columns:
        if column in df:
            df[column], _ = _to_numeric(df[column])
    pnl = df["5日損益率"]
    equity = (1 + pnl / 100).cumprod()
    win_rate = lambda x: float((x > 0).mean())
    volume_band = pd.cut(df["出来高倍率"], [0, 1, 1.3, 1.5, 2, np.inf])
    strategy_labels = df["シグナル種別"].fillna("アンチ") if "シグナル種別" in df else pd.Series("アンチ", index=df.index)
    strategy = df.assign(戦略=strategy_labels.str.split(" / ")).explode("戦略")
    strategy_summary = {}
    for name, group in strategy.groupby("戦略"):
        p = group["5日損益率"].dropna(); eq = (1 + p / 100).cumprod()
        strategy_summary[name] = {"シグナル数": len(p), "勝率": float((p > 0).mean()) if len(p) else 0,
            "平均利益率": float(p[p > 0].mean()), "平均損失率": float(p[p <= 0].mean()),
            "PF": float(p[p > 0].sum() / abs(p[p < 0].sum())) if (p < 0).any() else None,
            "最大ドローダウン": float((eq / eq.cummax() - 1).min()) if len(eq) else 0,
            "平均最大上昇率": float(group["最大上昇率"].mean()), "平均最大下落率": float(group["最大下落率"].mean())}
    return {"全シグナル数": len(df), "勝率": float((pnl > 0).mean()), "平均利益率": float(pnl[pnl > 0].mean()),
            "平均損失率": float(pnl[pnl <= 0].mean()), "PF": float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else None,
            "最大ドローダウン": float(((equity/equity.cummax())-1).min()),
            "ランク別勝率": df.groupby("ランク", observed=True)["5日損益率"].apply(win_rate).to_dict(),
            "RSI別勝率": df.groupby(pd.cut(df.RSI14, [0,30,40,55,70,100]), observed=True)["5日損益率"].apply(win_rate).to_dict(),
            "出来高倍率別勝率": df.groupby(volume_band, observed=True)["5日損益率"].apply(win_rate).to_dict(),
            "BB位置別勝率": df.groupby("BB位置")["5日損益率"].apply(win_rate).to_dict(),
            "ローソク足パターン別勝率": df.assign(pattern=df["ローソク足パターン"].str.split(" / ")).explode("pattern").groupby("pattern")["5日損益率"].apply(win_rate).to_dict(),
            "戦略別成績": strategy_summary}


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); print(update()); print(summary())
