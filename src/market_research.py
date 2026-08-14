"""Post-trade grouping for market/news regime validation; never edits production rules."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def japan_rate_performance(samples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rate bands, sectors, and JP/US combinations for the AI researcher."""
    columns = ["検証軸", "環境", "業種", "サンプル数", "勝率", "平均損益", "PF", "MFE", "MAE"]
    required = {"jp_10y_change_bp", "5日損益率"}
    if samples.empty or not required.issubset(samples.columns): return pd.DataFrame(columns=columns)
    data = samples.copy()
    data["金利変化幅"] = pd.cut(pd.to_numeric(data["jp_10y_change_bp"], errors="coerce"),
        [-float("inf"), -10, -5, 0, 5, 10, float("inf")],
        labels=["-10bp以下", "-10～-5bp", "-5～0bp", "0～+5bp", "+5～10bp", "+10bp以上"], include_lowest=True)
    jp = pd.to_numeric(data["jp_10y_change_bp"], errors="coerce")
    us = pd.to_numeric(data.get("us_10y_change_bp"), errors="coerce")
    data["日米金利"] = (jp.ge(0).map({True:"日本↑", False:"日本↓"}) + " " + us.ge(0).map({True:"米国↑", False:"米国↓"}))
    result = []
    for axis, key in (("変化幅", "金利変化幅"), ("日米組合せ", "日米金利")):
        groups = data.groupby([key, data.get("業種", pd.Series("全体", index=data.index))], observed=True)
        for (environment, sector), group in groups:
            pnl = pd.to_numeric(group["5日損益率"], errors="coerce").dropna()
            losses = abs(pnl[pnl < 0].sum())
            result.append({"検証軸": axis, "環境": environment, "業種": sector, "サンプル数": len(pnl),
                "勝率": (pnl > 0).mean() if len(pnl) else 0, "平均損益": pnl.mean(),
                "PF": pnl[pnl > 0].sum()/losses if losses else None,
                "MFE": pd.to_numeric(group.get("MFE"), errors="coerce").mean(),
                "MAE": pd.to_numeric(group.get("MAE"), errors="coerce").mean()})
    return pd.DataFrame(result, columns=columns)


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
    groups = ["market_regime", "market_condition", "news_condition"]
    text = ["# AI研究員・市場環境／ニュース条件別検証", "", "本番ロジックは自動変更しません。"]
    for group in groups:
        text += [f"\n## {group}\n", grouped_performance(trades, group, minimum_samples).to_markdown(index=False) if group in trades else "継続検証（列またはサンプルなし）"]
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(text), encoding="utf-8")
