"""Post-trade grouping for market/news regime validation; never edits production rules."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def grouped_performance(trades: pd.DataFrame, group: str, minimum_samples=20) -> pd.DataFrame:
    rows = []
    if trades.empty or group not in trades: return pd.DataFrame()
    for name, frame in trades.groupby(group, dropna=False):
        pnl = pd.to_numeric(frame["pnl"], errors="coerce").dropna()
        wins, losses = pnl[pnl > 0].sum(), abs(pnl[pnl < 0].sum())
        rows.append({group: name, "samples": len(pnl), "status": "検証可能" if len(pnl) >= minimum_samples else "継続検証",
                     "win_rate": float((pnl > 0).mean()) if len(pnl) else None, "average_pnl": pnl.mean() if len(pnl) else None,
                     "PF": wins / losses if losses else None,
                     "MFE": pd.to_numeric(frame["MFE"], errors="coerce").mean() if "MFE" in frame else None,
                     "MAE": pd.to_numeric(frame["MAE"], errors="coerce").mean() if "MAE" in frame else None})
    return pd.DataFrame(rows)


def write_research_report(trades: pd.DataFrame, output: Path, minimum_samples=20):
    groups = ["market_regime", "market_condition", "news_condition", "jgb_10y_condition", "jgb_curve_condition"]
    text = ["# AI研究員・市場環境／ニュース条件別検証", "", "本番ロジックは自動変更しません。"]
    for group in groups:
        text += [f"\n## {group}\n", grouped_performance(trades, group, minimum_samples).to_markdown(index=False) if group in trades else "継続検証（列またはサンプルなし）"]
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(text), encoding="utf-8")
