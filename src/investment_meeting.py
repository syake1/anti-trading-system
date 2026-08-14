"""Rule-based meetings by the analysis and portfolio AI employees.

The employees describe the outcome; only this version-controlled rule set decides it.
They never rewrite production files. Improvement ideas belong in reports/proposals/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.backtest import read_csv_if_populated
from src.discord_notify import post
from src.risk_manager import position_size
from src.utils import ROOT, load_config, now_tokyo

ORDER = {"主力候補": 0, "小口候補": 1, "監視": 2, "見送り": 3}
MEETING_COLUMNS = ["コード", "銘柄名", "最終分類", "分析評価", "運用評価", "株価", "推奨株数", "必要資金",
                   "損切り", "利確", "RR", "RSI", "BB位置", "出来高倍率", "3日騰落率", "5日騰落率",
                   "25日線乖離", "シグナル", "暴落リバウンド型", "分析コメント", "運用コメント"]


def _num(row, key, default=0.0):
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def analysis_employee(row: dict, config: dict) -> dict:
    m = config["meeting"]
    crash = (_num(row, "直近3日騰落率") <= m["crash_3d_pct"] or
             _num(row, "直近5日騰落率") <= m["crash_5d_pct"] or
             _num(row, "25日線乖離率") <= m["crash_ma25_deviation_pct"])
    surged = bool(str(row.get("除外理由", "")).strip()) or str(row.get("ランク")) == "除外"
    volume = _num(row, "出来高倍率")
    reversal = str(row.get("ローソク足パターン", "なし")) != "なし"
    rsi = _num(row, "RSI14", 50)
    bb = str(row.get("BB位置", ""))
    if surged:
        classification, reason = "見送り", "急騰済み判定を最優先"
    elif volume < m["core_minimum_volume_ratio"]:
        classification, reason = "監視", "反転形状より出来高不足を優先"
    elif crash:
        classification, reason = ("小口候補" if reversal and volume >= m["minimum_volume_ratio"] else "監視"), "暴落リバウンド型"
    elif str(row.get("ランク")) in ("S", "A") and reversal and rsi <= 45 and (bb.startswith("-") or "-" in bb):
        classification, reason = "主力候補", "適度な押しからの反転条件が揃う"
    elif volume < m["minimum_volume_ratio"]:
        classification, reason = "監視", "出来高増加待ち"
    else:
        classification, reason = "監視", "反転・指標条件の完成待ち"
    grade = "A+" if classification == "主力候補" else "A" if classification == "小口候補" else "B" if classification == "監視" else "C"
    return {"classification": classification, "grade": grade, "crash": crash,
            "comment": f"{reason}（RSI {rsi:.1f}、出来高 {volume:.2f}倍、BB {bb or '不明'}）"}


def operations_employee(row: dict, analysis: dict, config: dict) -> dict:
    classification = analysis["classification"]
    rr = _num(row, "RR")
    if rr < config["meeting"]["minimum_rr"]:
        classification, reason = "見送り", "RR不足"
    elif classification == "主力候補" and _num(row, "出来高倍率") < config["meeting"]["minimum_volume_ratio"]:
        classification, reason = "監視", "主力に必要な出来高不足"
    else:
        reason = "通常サイズで検討可能" if classification == "主力候補" else "リスクを抑えて判断"
    size = position_size(_num(row, "現在値"), _num(row, "損切り候補"), classification, config,
                         crash_rebound=analysis["crash"])
    if classification in ("主力候補", "小口候補") and size["推奨株数"] == 0:
        classification, reason = "見送り", "資金・最低売買単位・保有上限の制約"
    return {"classification": classification, "comment": reason, **size}


def evaluate_candidates(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for row in candidates.fillna("").to_dict("records"):
        a = analysis_employee(row, config)
        o = operations_employee(row, a, config)
        final = max((a["classification"], o["classification"]), key=ORDER.get)
        rows.append({"コード": str(row.get("コード", "")), "銘柄名": row.get("会社名", ""), "最終分類": final,
                     "分析評価": a["grade"], "運用評価": o["classification"], "株価": _num(row, "現在値"),
                     "推奨株数": o["推奨株数"], "必要資金": o["必要資金"], "損切り": o["損切り"],
                     "利確": _num(row, "利確候補"), "RR": _num(row, "RR"), "RSI": _num(row, "RSI14"),
                     "BB位置": row.get("BB位置", ""), "出来高倍率": _num(row, "出来高倍率"),
                     "3日騰落率": _num(row, "直近3日騰落率"), "5日騰落率": _num(row, "直近5日騰落率"),
                     "25日線乖離": _num(row, "25日線乖離率"), "シグナル": row.get("シグナル種別", ""),
                     "暴落リバウンド型": a["crash"], "分析コメント": a["comment"], "運用コメント": o["comment"]})
    return pd.DataFrame(rows, columns=MEETING_COLUMNS)


def morning_message(result: pd.DataFrame, config: dict, *, recheck=False) -> str:
    p = config["portfolio"]
    counts = result["最終分類"].value_counts() if not result.empty else {}
    title = "📊 AI投資会議・8:30再確認" if recheck else "📊 AI投資会議・朝会"
    lines = [title, "", f'運用資産：{p["initial_capital"]:,.0f}円',
             f'現金比率：{p.get("current_cash", p["initial_capital"])/p["initial_capital"]:.0%}',
             *[f"{name}：{counts.get(name, 0)}銘柄" for name in ORDER]]
    actionable = result[result["最終分類"].isin(["主力候補", "小口候補"])] if not result.empty else result
    if actionable.empty:
        lines += ["", "本日は新規買いなし"]
    for _, r in actionable.head(5).iterrows():
        lines += ["", f'🔥 {r["最終分類"]}', f'{r["コード"]} {r["銘柄名"]}',
                  f'買値候補：{r["株価"]:,.0f}円 / 損切り：{r["損切り"]:,.0f}円',
                  f'推奨株数：{r["推奨株数"]}株 / 必要資金：{r["必要資金"]:,.0f}円',
                  f'分析担当：{r["分析コメント"]}', f'運用担当：{r["運用コメント"]}', f'最終判断：{r["最終分類"]}']
    if recheck:
        lines += ["", "前日終値データのみのため6:30朝会から更新なし"]
    return "\n".join(lines)


def run(kind="morning", notify=True, candidates_path: Path | None = None) -> Path:
    config, today = load_config(), now_tokyo().strftime("%Y%m%d")
    if candidates_path is None:
        files = sorted(ROOT.glob("anti_candidates_*.csv"), reverse=True)
        candidates_path = files[0] if files else None
    candidates = read_csv_if_populated(candidates_path, dtype={"コード": str}) if candidates_path else pd.DataFrame()
    result = evaluate_candidates(candidates, config)
    folder = "morning" if kind == "recheck" else kind
    prefix = {"morning": "morning_meeting", "recheck": "morning_recheck", "close": "close_meeting", "weekly": "weekly_meeting"}[kind]
    output = ROOT / "reports/meeting" / folder / f"{prefix}_{today}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    if kind in ("morning", "recheck"):
        message = morning_message(result, config, recheck=kind == "recheck")
    elif kind == "close":
        message = f"📊 AI投資会議・大引け後\n\n本日の候補：{len(result)}\nルール違反売買：0\n翌日監視：{(result['最終分類'] == '監視').sum() if not result.empty else 0}\n\n終値確定後の実績は backtest で追跡します。"
    else:
        message = "📊 AI投資会議・週次\n\n勝率・PF・MFE・MAE・最大DDは data/performance.csv の確定実績を検証します。\n見送り銘柄も履歴に残し、フィルターの厳しさを人間が確認します。"
    table = "```csv\n" + result.to_csv(index=False) + "```" if not result.empty else "候補なし"
    output.write_text(message + "\n\n## 最終候補\n\n" + table + "\n", encoding="utf-8")
    if notify:
        post(message)
    print(message)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["morning", "recheck", "close", "weekly"], default="morning")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()
    run(args.kind, not args.no_notify)
