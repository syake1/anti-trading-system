"""Reproducible two-employee investment meeting and executable order proposals.

This creates proposals for human review only; it never sends an order to a broker.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.backtest import order_method_summary, read_csv_if_populated
from src.discord_notify import post
from src.order_planner import ORDER_LABELS, number, order_prices, select_order, size_order
from src.utils import ROOT, load_config, now_tokyo
from src.market_environment import analyze_market, fetch_market_data, save_market_environment
from src.japan_rates import (JapanRates, fetch_japan_rates, rate_sector_impacts,
                             save_japan_rates, stock_rate_impact)
from src.news_analysis import analyze_news, news_impacts, save_news
from src.sector_impact import sector_impacts, save_sector_impacts

CAUTION = {"主力": 0, "小口": 1, "監視": 2, "見送り": 3}
MEETING_COLUMNS = ["コード", "銘柄名", "最終判断", "最終分類", "注文方式", "買いゾーン下限", "買いゾーン上限",
 "反転確認高値", "逆指値発動価格", "追いかけ禁止価格", "損切り価格", "利確目標", "推奨株数",
 "必要資金", "最大想定損失", "RR", "分析評価", "運用評価", "総合順位点", "RSI", "BB位置",
 "出来高倍率", "暴落リバウンド型", "ファンダメンタルスコア", "テクニカルスコア", "反転確認スコア",
 "市場環境スコア", "業種環境スコア", "日本金利影響", "日本金利影響理由", "ニュース影響スコア", "資金・リスク評価", "分析コメント", "運用コメント", "注文理由"]


def analysis_employee(row: dict, config: dict) -> dict:
    m = config["meeting"]
    crash = (number(row, "直近3日騰落率") <= m["crash_3d_pct"] or
             number(row, "直近5日騰落率") <= m["crash_5d_pct"] or
             number(row, "25日線乖離率") <= m["crash_ma25_deviation_pct"])
    surged = bool(str(row.get("除外理由", "")).strip()) or str(row.get("ランク")) == "除外"
    volume, rsi = number(row, "出来高倍率"), number(row, "RSI14", 50)
    reversal = str(row.get("ローソク足パターン", "なし")) != "なし"
    if surged:
        classification, reason = "見送り", "急騰済みを最優先して除外"
    elif volume < m["core_minimum_volume_ratio"]:
        classification, reason = "監視", "出来高0.6倍未満のため主力不可"
    elif crash:
        classification, reason = ("小口" if reversal and volume >= m["minimum_volume_ratio"] else "監視"), "暴落リバウンド型のため主力不可"
    elif str(row.get("ランク")) in ("S", "A") and reversal and rsi <= 45 and "-" in str(row.get("BB位置", "")):
        classification, reason = "主力", "押し目と反転足を確認"
    else:
        classification, reason = "監視", "反転条件の完成待ち"
    score = number(row, "スコア") + min(volume, 3) * 2 + (3 if reversal else 0) - (4 if crash else 0)
    fundamentals = "入力あり" if any(row.get(k) not in (None, "") for k in ("EPS成長率", "ROE", "配当利回り", "上方修正")) else "入力なし（推測しない）"
    return {"classification": classification, "crash": crash, "score": score,
            "grade": "A+" if classification == "主力" else "A" if classification == "小口" else "B" if classification == "監視" else "C",
            "comment": f"{reason}。ファンダメンタル={fundamentals}、RSI {rsi:.1f}、出来高 {volume:.2f}倍、BB {row.get('BB位置', '不明')}"}


def _method_stats() -> dict:
    return order_method_summary(ROOT / "data/order_method_performance.csv")


def evaluate_candidates(candidates: pd.DataFrame, config: dict, environment=None, news=None, japan_rates=None) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=MEETING_COLUMNS)
    environment = environment or analyze_market([], config)
    news = news if news is not None else analyze_news(None, config)
    japan_rates = japan_rates or JapanRates(timestamp=environment.observed_at)
    sector_scores = sector_impacts(environment, config)
    emergency = bool(news["emergency_risk"].any()) if not news.empty else False
    stats, staged = _method_stats(), []
    for row in candidates.fillna("").to_dict("records"):
        a = analysis_employee(row, config)
        sector = str(row.get("業種", "")).strip()
        news_score = news_impacts(news, sector, str(row.get("コード", "")), config)
        sector_score = sector_scores.get(sector, 0)
        values = {x["indicator"]: x for x in environment.indicators}
        jp_score, jp_reason = stock_rate_impact(row, japan_rates,
            values.get("US10Y", {}).get("change_bp"), values.get("USDJPY", {}).get("change_pct"),
            risk_off=environment.regime == "強いリスクオフ", credit_risk=emergency)
        a["score"] += environment.total_score + sector_score + news_score + jp_score
        prices = order_prices(row, config)
        method, order_reason = select_order(row, prices, config, stats)
        operations_class = a["classification"]
        if emergency: operations_class, method, order_reason = "見送り", "skip", "重大ニュースによる緊急リスク：新規主力買い停止"
        elif environment.regime == "強いリスクオフ" and sector_score < 3:
            operations_class, method, order_reason = "監視", "skip", "強いリスクオフのため新規主力買い停止"
        elif environment.regime == "警戒" and operations_class == "主力": operations_class = "小口"
        if japan_rates.jp_rate_regime in ("急上昇", "急低下") and jp_score < 0 and operations_class == "主力":
            operations_class, order_reason = "小口", "日本金利急変が逆風のため主力から小口へ降格"
        if japan_rates.jp_rate_regime in ("急上昇", "急低下") and japan_rates.boj_news_observed and method == "market":
            method, order_reason = "skip", "日本金利急変＋日銀ニュースのため寄り成りを避け監視"
        if method == "skip": operations_class = "見送り" if a["classification"] != "見送り" else a["classification"]
        elif a["classification"] not in ("主力", "小口"): operations_class = "監視"
        final = max((a["classification"], operations_class), key=CAUTION.get)
        entry = prices["逆指値発動価格"] if method == "stop" else prices["買いゾーン上限"] if method == "limit" else number(row, "現在値")
        staged.append((row, a, prices, method, order_reason, final, entry, sector_score, news_score, jp_score, jp_reason))
    # Operational priority, not just scanner score. Stable code tie-break ensures reproducibility.
    staged.sort(key=lambda x: (-x[1]["score"], str(x[0].get("コード", ""))))
    primary_seen = 0
    cash = float(config["portfolio"].get("current_cash", config["portfolio"]["initial_capital"]))
    positions = int(config["portfolio"].get("current_positions", 0))
    sector_positions: dict[str, int] = {}
    result = []
    for row, a, prices, method, order_reason, final, entry, sector_score, news_score, jp_score, jp_reason in staged:
        sector = str(row.get("業種", "")).strip()
        if sector and sector_positions.get(sector, 0) >= int(config["portfolio"].get("max_positions_per_sector", 2)):
            final, method, order_reason = "監視", "skip", "業種偏り上限のため監視"
        if final == "主力":
            primary_seen += 1
            if primary_seen > int(config["meeting"]["max_primary_candidates"]):
                final, method, order_reason = "監視", "skip", "主力上限超過のため4位以下は監視"
        sized = size_order(entry, prices["損切り価格"], final, config, cash=cash, positions=positions, crash=a["crash"])
        sized["推奨株数"] = int(sized["推奨株数"] * environment.capital_ratio // config["portfolio"]["lot_size"] * config["portfolio"]["lot_size"])
        sized["必要資金"] = sized["推奨株数"] * entry
        sized["最大想定損失"] = sized["推奨株数"] * max(entry - prices["損切り価格"], 0)
        if final in ("主力", "小口") and sized["推奨株数"] == 0:
            final, method, order_reason = "見送り", "skip", "資金・リスク・売買単位・保有上限の制約"
        if final in ("主力", "小口"):
            cash -= sized["必要資金"]; positions += 1
            if sector: sector_positions[sector] = sector_positions.get(sector, 0) + 1
        risk = entry - prices["損切り価格"]
        rr = (prices["利確目標"] - entry) / risk if risk > 0 else 0
        method_pf = stats.get(method, {}).get("PF")
        perf = f"方式別PF {method_pf:.2f}" if isinstance(method_pf, (float, int)) else "方式別実績はサンプル蓄積中"
        legacy_final = {"主力": "主力候補", "小口": "小口候補"}.get(final, final)
        result.append({"コード": str(row.get("コード", "")), "銘柄名": row.get("会社名", ""), "最終判断": final, "最終分類": legacy_final,
          "注文方式": ORDER_LABELS[method], **{k: round(v, 2) for k, v in prices.items() if k != "基準RR"}, **sized,
          "RR": round(rr, 2), "分析評価": a["grade"], "運用評価": operations_class, "総合順位点": round(a["score"], 2),
          "RSI": number(row, "RSI14"), "BB位置": row.get("BB位置", ""), "出来高倍率": number(row, "出来高倍率"),
          "暴落リバウンド型": a["crash"], "分析コメント": a["comment"],
          "ファンダメンタルスコア": number(row, "ファンダメンタルスコア"), "テクニカルスコア": number(row, "スコア"),
          "反転確認スコア": 3 if str(row.get("ローソク足パターン", "なし")) != "なし" else 0,
          "市場環境スコア": environment.total_score, "業種環境スコア": sector_score, "日本金利影響": jp_score,
          "日本金利影響理由": jp_reason, "ニュース影響スコア": news_score,
          "資金・リスク評価": f"{environment.regime}・通常の{environment.capital_ratio:.0%}",
          "運用コメント": f"{perf}。市場={environment.regime}、ニュース重大度と300万円の損失・投入・現金・保有数制約を適用", "注文理由": order_reason})
    return pd.DataFrame(result, columns=MEETING_COLUMNS)


def morning_message(result: pd.DataFrame, config: dict, *, recheck=False, environment=None, news=None, japan_rates=None) -> str:
    p = config["portfolio"]
    title = "📊 AI投資会議・8:30再確認" if recheck else "📊 AI投資会議・朝会\n最終注文案"
    actionable = result[result["最終判断"].isin(["主力", "小口"])] if not result.empty else result
    environment = environment or analyze_market([], config); news = news if news is not None else analyze_news(None, config)
    japan_rates = japan_rates or JapanRates(timestamp=environment.observed_at)
    values = {x['indicator']: x for x in environment.indicators}
    lines = ["🌏 市場環境"] + [f"{name}: {values[name]['current']:,.2f} ({values[name]['change_pct']:+.2f}%) score {values[name]['score']:+d}" for name in config['market_environment']['tickers'] if name in values]
    impacts = rate_sector_impacts(japan_rates, risk_off=environment.regime == "強いリスクオフ")
    rate = (f"{japan_rates.jp_10y_yield:.3f}%" if japan_rates.jp_10y_yield is not None else "取得不可")
    bp = lambda v: f"{v:+.1f}bp" if v is not None else "NA"
    lines += ["", "🇯🇵 日本金利環境", f"日本10年国債利回り：{rate}",
              f"前日差：{bp(japan_rates.jp_10y_change_bp)}", f"5日変化：{bp(japan_rates.jp_10y_change_5d_bp)}",
              f"20日変化：{bp(japan_rates.jp_10y_change_20d_bp)}", f"判定：{japan_rates.jp_rate_regime}",
              "日本2年国債利回り：" + (f"{japan_rates.jp_2y_yield:.3f}%" if japan_rates.jp_2y_yield is not None else "取得不可"),
              f"金融政策警戒：{'あり' if japan_rates.boj_tightening_risk else 'なし'}", "日本金利による業種影響",
              " / ".join(f"{k}：{impacts[k]:+d}" for k in ("銀行", "保険", "不動産", "REIT", "高PERグロース"))]
    headlines = news.loc[news['trusted'], 'title'].head(3).tolist() if not news.empty else []
    lines += [f"市場判定：{environment.regime} ({environment.total_score:+.1f})", "重大ニュース：" + (" / ".join(headlines) if headlines else "取得・入力された信頼済みニュースなし"), f"本日の資金方針：通常の{environment.capital_ratio:.0%}", "", title, "", f'運用資産：{p["initial_capital"]:,.0f}円',
             f'現金比率：{p.get("current_cash", p["initial_capital"])/p["initial_capital"]:.0%}',
             f'本日の主力候補：{(result["最終判断"] == "主力").sum() if not result.empty else 0}',
             f'小口候補：{(result["最終判断"] == "小口").sum() if not result.empty else 0}']
    if actionable.empty:
        lines += ["", "本日は新規注文なし（本日は新規買いなし）。", "理由：主力・小口条件を満たす銘柄なし。", "現金維持。"]
    for _, r in actionable.iterrows():
        lines += ["", f'🔥 {r["最終判断"]}候補', f'{r["コード"]} {r["銘柄名"]}',
          f'最終判断：{r["最終判断"]}', f'注文方式：{r["注文方式"]}',
          f'買いゾーン：{r["買いゾーン下限"]:,.0f}～{r["買いゾーン上限"]:,.0f}円',
          f'逆指値：{r["逆指値発動価格"]:,.0f}円', f'追いかけ禁止：{r["追いかけ禁止価格"]:,.0f}円',
          f'損切り：{r["損切り価格"]:,.0f}円', f'利確目標：{r["利確目標"]:,.0f}円',
          f'推奨株数：{r["推奨株数"]}株', f'必要資金：{r["必要資金"]:,.0f}円',
          f'最大想定損失：{r["最大想定損失"]:,.0f}円 / RR {r["RR"]:.2f}',
          f'日本金利影響：{r["日本金利影響"]:+.0f}（{r["日本金利影響理由"]}）',
          f'分析担当：{r["分析コメント"]}', f'運用担当：{r["運用コメント"]}', f'実行条件：{r["注文理由"]}']
    if recheck: lines += ["", "気配・寄値の更新データなし：朝会から判断変更なし"]
    lines += ["", "※証券会社への自動発注は行いません。注文前に人間が価格・気配・材料を確認してください。"]
    return "\n".join(lines)


def run(kind="morning", notify=True, candidates_path: Path | None = None) -> Path:
    config, today = load_config(), now_tokyo().strftime("%Y%m%d")
    if candidates_path is None:
        files = sorted(ROOT.glob("anti_candidates_*.csv"), reverse=True); candidates_path = files[0] if files else None
    candidates = read_csv_if_populated(candidates_path, dtype={"コード": str}) if candidates_path else pd.DataFrame()
    market_input = ROOT / "data/market_input.csv"
    market_rows = read_csv_if_populated(market_input) if market_input.exists() and market_input.stat().st_size > 60 else fetch_market_data(config)
    environment = analyze_market(market_rows, config); news = analyze_news(ROOT / "data/news_input.csv", config)
    boj_news = bool((news.get("category") == "monetary_policy").any()) if not news.empty else False
    usd = next((x.get("change_pct") for x in environment.indicators if x["indicator"] == "USDJPY"), None)
    japan_rates = fetch_japan_rates(config, boj_news=boj_news, usd_jpy_change_pct=usd)
    impacts = sector_impacts(environment, config); save_market_environment(environment); save_japan_rates(japan_rates); save_news(news); save_sector_impacts(impacts, environment.observed_at)
    result = evaluate_candidates(candidates, config, environment, news, japan_rates)
    folder = "morning" if kind == "recheck" else kind
    prefix = {"morning":"morning_meeting", "recheck":"morning_recheck", "close":"close_meeting", "weekly":"weekly_meeting"}[kind]
    output = ROOT / "reports/meeting" / folder / f"{prefix}_{today}.md"; output.parent.mkdir(parents=True, exist_ok=True)
    if kind in ("morning", "recheck"): message = morning_message(result, config, recheck=kind == "recheck", environment=environment, news=news, japan_rates=japan_rates)
    elif kind == "close":
        message = "📊 AI投資会議・大引け後\n\n朝の注文案ごとに、約定／未約定／追いかけ禁止見送り／損切り／利確／保有継続を data/order_method_performance.csv へ記録します。価格データ未確定時は推測しません。"
    else:
        s = order_method_summary(ROOT / "data/order_method_performance.csv")
        message = "📊 AI投資会議・週次\n\n" + ("\n".join(f"{ORDER_LABELS.get(k,k)}: 約定率 {v['約定率']:.1%}, 勝率 {v['勝率']:.1%}, PF {v['PF'] if v['PF'] is not None else 'N/A'}" for k,v in s.items()) if s else "方式別サンプル蓄積中") + "\n\n改善案のみ reports/proposals/ に保存し、本番ルールは自動変更しません。"
        proposal = ROOT / "reports/proposals" / f"order_research_{today}.md"; proposal.parent.mkdir(parents=True, exist_ok=True)
        proposal.write_text("# AI研究員・注文方式改善提案\n\n" + ("\n".join(
            f"- {ORDER_LABELS.get(k,k)}: PF={v['PF']}, 約定率={v['約定率']:.1%}, MFE={v['MFE']:.2f}, MAE={v['MAE']:.2f}"
            for k,v in s.items()) if s else "十分なサンプルがないためルール変更を提案しません。") +
            "\n\nRR 1.5/2.0/2.5、逆指値ATR幅、追いかけ禁止幅は追加サンプルで比較してください。設定は自動変更していません。\n", encoding="utf-8")
    output.write_text(message + "\n\n## 全候補監査表\n\n```csv\n" + result.to_csv(index=False) + "```\n", encoding="utf-8")
    if notify: post(message)
    print(message); return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--kind", choices=["morning","recheck","close","weekly"], default="morning"); parser.add_argument("--no-notify", action="store_true")
    args=parser.parse_args(); run(args.kind, not args.no_notify)
