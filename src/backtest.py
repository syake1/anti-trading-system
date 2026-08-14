from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from src.csv_history import LEGACY_SIGNAL_COLUMNS, read_mixed_csv
from src.utils import ROOT
from src.utils import load_config
from src.order_planner import order_prices

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
ORDER_PERFORMANCE_COLUMNS = ["シグナル日", "コード", "注文方式", "約定", "約定価格",
    "1日後損益", "3日後損益", "5日後損益", "10日後損益", "MFE", "MAE",
    "未約定後最大上昇率", "未約定後最大下落率"]


def simulate_order_methods(signal: dict, future: pd.DataFrame, config: dict) -> list[dict]:
    """Paper-trade market/limit/stop concurrently; unfilled orders have no P&L."""
    prices = order_prices(signal, config)
    if future.empty:
        return []
    base = {"シグナル日": signal.get("シグナル日", ""), "コード": str(signal.get("コード", ""))}
    records = []
    for method in ("market", "limit", "stop"):
        fill_index = None; fill = np.nan
        for idx, bar in future.head(int(config["order_backtest"]["valid_days"])).iterrows():
            if method == "market" and pd.notna(bar.get("Open")):
                fill_index, fill = idx, float(bar.Open)
            elif method == "limit" and bar.Low <= prices["買いゾーン上限"]:
                fill_index, fill = idx, min(float(bar.Open), prices["買いゾーン上限"])
            elif method == "stop" and bar.High >= prices["逆指値発動価格"] and bar.Open <= prices["追いかけ禁止価格"]:
                fill_index, fill = idx, max(float(bar.Open), prices["逆指値発動価格"])
            if fill_index is not None: break
        rec = {**base, "注文方式": method, "約定": fill_index is not None, "約定価格": fill}
        if fill_index is None:
            close = number_series = float(signal.get("現在値", 0) or 0)
            rec.update({f"{n}日後損益": np.nan for n in (1,3,5,10)})
            rec.update({"MFE": np.nan, "MAE": np.nan,
                        "未約定後最大上昇率": (future.High.max()/close-1)*100 if close else np.nan,
                        "未約定後最大下落率": (future.Low.min()/close-1)*100 if close else np.nan})
        else:
            after = future.loc[fill_index:].head(10)
            for n in (1,3,5,10):
                rec[f"{n}日後損益"] = (after.Close.iloc[n-1]/fill-1)*100 if len(after) >= n else np.nan
            rec.update({"MFE": (after.High.max()/fill-1)*100, "MAE": (after.Low.min()/fill-1)*100,
                        "未約定後最大上昇率": np.nan, "未約定後最大下落率": np.nan})
        records.append(rec)
    return records


def order_method_summary(path=ROOT / "data/order_method_performance.csv") -> dict:
    df = read_csv_if_populated(path)
    if df.empty: return {}
    result = {}
    for method, group in df.groupby("注文方式"):
        filled = group[group["約定"].astype(str).str.lower().isin(("true", "1"))].copy()
        pnl = pd.to_numeric(filled.get("5日後損益"), errors="coerce").dropna()
        mfe = pd.to_numeric(filled.get("MFE"), errors="coerce")
        mae = pd.to_numeric(filled.get("MAE"), errors="coerce")
        result[method] = {"サンプル数": len(group), "約定件数": len(filled), "約定率": len(filled)/len(group),
          "平均約定価格": _safe_float(pd.to_numeric(filled.get("約定価格"), errors="coerce").mean()),
          "勝率": _win_rate(pnl), "平均損益": _safe_float(pnl.mean()), "平均利益": _safe_float(pnl[pnl>0].mean()),
          "平均損失": _safe_float(pnl[pnl<0].mean()), "PF": _profit_factor(pnl),
          "MFE": _safe_float(mfe.mean()), "MAE": _safe_float(mae.mean()), "最大DD": _max_drawdown(pnl)}
    return result


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


def _safe_float(value: object, default: float = 0.0) -> float:
    """Nullable dtype の集計値を通常の float へ安全に変換する。"""
    return float(value) if pd.notna(value) else default


def _win_rate(values: pd.Series) -> float:
    valid = values.dropna()
    return _safe_float((valid > 0).mean()) if not valid.empty else 0.0


def _profit_factor(values: pd.Series) -> float | None:
    """損失がない場合は、定義できない指標として欠損表示を返す。"""
    valid = values.dropna()
    losses = valid[valid < 0]
    if losses.empty:
        return None
    return _safe_float(valid[valid > 0].sum() / abs(losses.sum()))


def _max_drawdown(values: pd.Series) -> float:
    valid = values.dropna()
    if valid.empty:
        return 0.0
    equity = (1 + valid / 100).cumprod()
    return _safe_float((equity / equity.cummax() - 1).min())


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
    records, order_records = [], []
    config = load_config()
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
        # yfinance の end は排他的。pandas の日時演算に依存せず翌日を指定する。
        end_date = (date.today() + timedelta(days=1)).isoformat()
        prices = yf.download(f'{row["コード"]}.T', start=row["シグナル日"],
                             end=end_date, progress=False, auto_adjust=False)
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
        if "Open" in future:
            normalized_signal = {**row, "現在値": current, "損切り候補": stop_loss, "利確候補": take_profit}
            order_records.extend(simulate_order_methods(normalized_signal, future, config))
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
    order_output = ROOT / "data/order_method_performance.csv"
    pd.DataFrame(order_records, columns=ORDER_PERFORMANCE_COLUMNS).to_csv(order_output, index=False, encoding="utf-8-sig")
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
        if column not in df:
            df[column] = pd.Series(np.nan, index=df.index, dtype="float64")
        else:
            df[column], _ = _to_numeric(df[column])
    pnl = df["5日損益率"]
    volume_band = pd.cut(df["出来高倍率"], [0, 1, 1.3, 1.5, 2, np.inf])
    strategy_labels = (df["シグナル種別"].astype("string").fillna("アンチ")
                       if "シグナル種別" in df else pd.Series("アンチ", index=df.index, dtype="string"))
    strategy = df.assign(戦略=strategy_labels.str.split(" / ")).explode("戦略")
    strategy_summary = {}
    for name, group in strategy.dropna(subset=["戦略"]).groupby("戦略"):
        p = group["5日損益率"].dropna()
        strategy_summary[name] = {"シグナル数": len(p), "勝率": _win_rate(p),
            "平均利益率": _safe_float(p[p > 0].mean()), "平均損失率": _safe_float(p[p <= 0].mean()),
            "PF": _profit_factor(p), "最大ドローダウン": _max_drawdown(p),
            "平均最大上昇率": _safe_float(group["最大上昇率"].mean()),
            "平均最大下落率": _safe_float(group["最大下落率"].mean())}
    rank = df["ランク"] if "ランク" in df else pd.Series(pd.NA, index=df.index)
    bb_position = df["BB位置"] if "BB位置" in df else pd.Series(pd.NA, index=df.index)
    patterns = (df["ローソク足パターン"].astype("string")
                if "ローソク足パターン" in df else pd.Series(pd.NA, index=df.index, dtype="string"))
    return {"全シグナル数": len(df), "勝率": _win_rate(pnl),
            "平均利益率": _safe_float(pnl[pnl > 0].mean()),
            "平均損失率": _safe_float(pnl[pnl <= 0].mean()), "PF": _profit_factor(pnl),
            "最大ドローダウン": _max_drawdown(pnl),
            "ランク別勝率": df.assign(_group=rank).groupby("_group", observed=True)["5日損益率"].apply(_win_rate).to_dict(),
            "RSI別勝率": df.groupby(pd.cut(df["RSI14"], [0,30,40,55,70,100]), observed=True)["5日損益率"].apply(_win_rate).to_dict(),
            "出来高倍率別勝率": df.groupby(volume_band, observed=True)["5日損益率"].apply(_win_rate).to_dict(),
            "BB位置別勝率": df.assign(_group=bb_position).groupby("_group")["5日損益率"].apply(_win_rate).to_dict(),
            "ローソク足パターン別勝率": df.assign(pattern=patterns.str.split(" / ")).explode("pattern").groupby("pattern")["5日損益率"].apply(_win_rate).to_dict(),
            "戦略別成績": strategy_summary}


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); print(update()); print(summary())
