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
        return output
    signals = signals.drop_duplicates(["シグナル日", "コード", "買い・売り"])
    records = []
    for row in signals.to_dict("records"):
        prices = yf.download(f'{row["コード"]}.T', start=row["シグナル日"],
                             end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
        if isinstance(prices.columns, pd.MultiIndex): prices.columns = prices.columns.get_level_values(0)
        future = prices.loc[prices.index > pd.Timestamp(row["シグナル日"])].head(20)
        if future.empty: continue
        direction = 1 if row["買い・売り"] == "買い" else -1
        rec = {"シグナル日": row["シグナル日"], "コード": row["コード"], "ランク": row["ランク"], "買い・売り": row["買い・売り"],
               "RSI14": row["RSI14"], "出来高倍率": row["出来高倍率"], "BB位置": row["BB位置"], "ローソク足パターン": row["ローソク足パターン"],
               "シグナル種別": row.get("シグナル種別", "アンチ")}
        for n in (1, 2, 3, 5, 10, 20):
            close = future.Close.iloc[n-1] if len(future) >= n else np.nan
            rec[f"{n}日後終値"] = close
            rec[f"{n}日後騰落率"] = direction * (close / row["現在値"] - 1) * 100 if pd.notna(close) else np.nan
        rec.update({"期間中最高値": future.High.max(), "期間中最安値": future.Low.min(),
                    "最大上昇率": (future.High.max()/row["現在値"]-1)*100, "最大下落率": (future.Low.min()/row["現在値"]-1)*100,
                    "利確到達": bool((future.High >= row["利確候補"]).any() if direction == 1 else (future.Low <= row["利確候補"]).any()),
                    "損切り到達": bool((future.Low <= row["損切り候補"]).any() if direction == 1 else (future.High >= row["損切り候補"]).any()),
                    "5日損益率": rec["5日後騰落率"]})
        records.append(rec)
    pd.DataFrame(records, columns=PERFORMANCE_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def summary(path=ROOT / "data/performance.csv") -> dict:
    df = read_csv_if_populated(path)
    if df.empty: return {"全シグナル数": 0}
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
