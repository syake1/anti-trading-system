"""Reproducible two-employee investment meeting and executable order proposals.

This creates proposals for human review only; it never sends an order to a broker.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd

from src.backtest import order_method_summary, read_csv_if_populated
from src.discord_notify import post
from src.order_planner import ORDER_LABELS, number, order_prices, select_order, size_order
from src.utils import ROOT, load_config, now_tokyo
from src.market_environment import analyze_market, fetch_market_data, save_market_environment
from src.news_analysis import analyze_news, news_impacts, save_news
from src.sector_impact import sector_impacts, save_sector_impacts
from src.jgb_yields import (analyze_jgb, fetch_jgb_data, format_jgb_message,
                            jgb_sector_impacts, save_jgb_analysis)
from src.market_research import write_research_report
from src.fundamentals import assess, enrich_candidates
from src.google_sheets import record_meeting_safely
from src.stocknote import consume_shadow, export_request, write_shadow_report
from src.night_meeting import (generate_night_result, generate_weekend_result, load_latest_night_result,
                               night_message, save_night_result, save_weekend_result, weekend_message)

CAUTION = {"主力": 0, "小口": 1, "監視": 2, "見送り": 3}
MEETING_COLUMNS = ["コード", "銘柄名", "最終判断", "最終分類", "注文方式", "買いゾーン下限", "買いゾーン上限",
 "反転確認高値", "逆指値発動価格", "追いかけ禁止価格", "損切り価格", "利確目標", "推奨株数",
 "必要資金", "最大想定損失", "RR", "分析評価", "運用評価", "総合順位点", "RSI", "BB位置",
 "出来高倍率", "暴落リバウンド型", "ファンダメンタルスコア", "テクニカルスコア", "反転確認スコア",
 "市場環境スコア", "業種環境スコア", "ニュース影響スコア", "資金・リスク評価", "分析コメント", "運用コメント", "注文理由"]
MEETING_COLUMNS += ["ファンダメンタル評価", "ファンダメンタル十分", "ファンダメンタル不足理由",
 "売上前年比", "営業利益前年比", "経常・純利益前年比", "EPS", "PER", "PBR", "ROE", "自己資本比率",
 "配当利回り", "利益状態", "業績4象限", "配当性向", "配当判定", "ファンダメンタル加減点理由",
 "今期会社予想", "直近決算発表日", "次回決算予定日", "時価総額", "総合判定", "判定理由",
 "業績修正", "重要適時開示", "ファンダメンタル取得元", "ファンダメンタル参照先", "ファンダメンタル取得日時"]


def _display(value):
    """Persist an explicit missing marker without feeding it back into calculations."""
    return "データなし" if value is None or str(value).strip() in ("", "nan") else value


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


def evaluate_candidates(candidates: pd.DataFrame, config: dict, environment=None, news=None, jgb=None) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=MEETING_COLUMNS)
    environment = environment or analyze_market([], config)
    news = news if news is not None else analyze_news(None, config)
    sector_scores = sector_impacts(environment, config)
    if jgb is not None:
        for sector, score in jgb_sector_impacts(jgb, config).items():
            sector_scores[sector] = sector_scores.get(sector, 0) + score
    emergency = bool(news["emergency_risk"].any()) if not news.empty else False
    stats, staged = _method_stats(), []
    for row in candidates.fillna("").to_dict("records"):
        a = analysis_employee(row, config)
        fundamental = assess(row, config)
        row = fundamental.data
        a["comment"] = a["comment"].replace("ファンダメンタル=入力なし（推測しない）", f"ファンダメンタル={fundamental.label}")
        sector = str(row.get("業種", "")).strip()
        news_score = news_impacts(news, sector, str(row.get("コード", "")), config)
        sector_score = sector_scores.get(sector, 0)
        a["score"] += environment.total_score + sector_score + news_score
        prices = order_prices(row, config)
        method, order_reason = select_order(row, prices, config, stats)
        operations_class = a["classification"]
        if emergency: operations_class, method, order_reason = "見送り", "skip", "重大ニュースによる緊急リスク：新規主力買い停止"
        elif environment.regime == "強いリスクオフ" and sector_score < 3:
            operations_class, method, order_reason = "監視", "skip", "強いリスクオフのため新規主力買い停止"
        elif environment.regime == "警戒" and operations_class == "主力": operations_class = "小口"
        if method == "skip": operations_class = "見送り" if a["classification"] != "見送り" else a["classification"]
        elif a["classification"] not in ("主力", "小口"): operations_class = "監視"
        final = max((a["classification"], operations_class), key=CAUTION.get)
        if not fundamental.sufficient and final == "主力":
            final, order_reason = "小口", f"ファンダメンタルデータ不足（{fundamental.reason}）：主力判定不可"
        elif fundamental.sufficient and fundamental.score <= 4 and final == "主力":
            final, order_reason = "小口", f"ファンダメンタル{fundamental.score}点のため主力から降格"
        entry = prices["逆指値発動価格"] if method == "stop" else prices["買いゾーン上限"] if method == "limit" else number(row, "現在値")
        staged.append((row, a, prices, method, order_reason, final, entry, sector_score, news_score, fundamental))
    # Operational priority, not just scanner score. Stable code tie-break ensures reproducibility.
    staged.sort(key=lambda x: (-x[1]["score"], str(x[0].get("コード", ""))))
    primary_seen = 0
    cash = float(config["portfolio"].get("current_cash", config["portfolio"]["initial_capital"]))
    positions = int(config["portfolio"].get("current_positions", 0))
    sector_positions: dict[str, int] = {}
    result = []
    for row, a, prices, method, order_reason, final, entry, sector_score, news_score, fundamental in staged:
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
          "ファンダメンタルスコア": _display(fundamental.score), "テクニカルスコア": number(row, "スコア"),
          "反転確認スコア": 3 if str(row.get("ローソク足パターン", "なし")) != "なし" else 0,
          "市場環境スコア": environment.total_score, "業種環境スコア": sector_score, "ニュース影響スコア": news_score,
          "資金・リスク評価": f"{environment.regime}・通常の{environment.capital_ratio:.0%}",
          "運用コメント": f"{perf}。市場={environment.regime}、ニュース重大度と300万円の損失・投入・現金・保有数制約を適用", "注文理由": order_reason,
          "ファンダメンタル評価": fundamental.label, "ファンダメンタル十分": fundamental.sufficient,
          "ファンダメンタル不足理由": fundamental.reason, "売上前年比": _display(row.get("revenue_yoy")),
          "営業利益前年比": _display(row.get("operating_profit_yoy")), "経常・純利益前年比": _display(row.get("ordinary_or_net_profit_yoy")),
          "EPS": _display(row.get("eps")), "PER": _display(row.get("per")), "PBR": _display(row.get("pbr")), "ROE": _display(row.get("roe")),
          "自己資本比率": _display(row.get("equity_ratio")), "配当利回り": _display(row.get("dividend_yield")),
          "利益状態": row.get("profit_transition", "評価不能"), "業績4象限": row.get("growth_quadrant", "評価不能"),
          "配当性向": row.get("payout_ratio", ""), "配当判定": row.get("dividend_change", "評価不能"),
          "ファンダメンタル加減点理由": " / ".join(fundamental.score_reasons),
          "今期会社予想": _display(row.get("company_forecast")), "直近決算発表日": _display(row.get("latest_earnings_date")),
          "次回決算予定日": _display(row.get("next_earnings_date")), "時価総額": _display(row.get("market_cap")),
          "総合判定": fundamental.label, "判定理由": fundamental.reason + (" / " + " / ".join(fundamental.score_reasons) if fundamental.score_reasons else ""),
          "業績修正": row.get("revision", ""), "重要適時開示": row.get("important_disclosure", ""),
          "ファンダメンタル取得元": _display(row.get("fundamental_source", row.get("source", ""))),
          "ファンダメンタル参照先": _display(row.get("fundamental_source_reference", row.get("source_reference", ""))),
          "ファンダメンタル取得日時": _display(row.get("acquired_at"))})
    return pd.DataFrame(result, columns=MEETING_COLUMNS)


def morning_message(result: pd.DataFrame, config: dict, *, recheck=False, environment=None, news=None, jgb=None,
                    night_reference=None) -> str:
    p = config["portfolio"]
    title = "📊 AI投資会議・8:30再確認" if recheck else "📊 AI投資会議・朝会\n最終注文案"
    actionable = result[result["最終判断"].isin(["主力", "小口"])] if not result.empty else result
    environment = environment or analyze_market([], config); news = news if news is not None else analyze_news(None, config)
    values = {x['indicator']: x for x in environment.indicators}
    lines = ["🌏 市場環境"] + [f"{name}: {values[name]['current']:,.2f} ({values[name]['change_pct']:+.2f}%) score {values[name]['score']:+d}" for name in config['market_environment']['tickers'] if name in values]
    if jgb is not None:
        lines += format_jgb_message(jgb)
    headlines = news.loc[news['trusted'], 'title'].head(3).tolist() if not news.empty else []
    lines += [f"市場判定：{environment.regime} ({environment.total_score:+.1f})", "重大ニュース：" + (" / ".join(headlines) if headlines else "取得・入力された信頼済みニュースなし"), f"本日の資金方針：通常の{environment.capital_ratio:.0%}", "", title, "", f'運用資産：{p["initial_capital"]:,.0f}円',
             f'現金比率：{p.get("current_cash", p["initial_capital"])/p["initial_capital"]:.0%}',
             f'本日の主力候補：{(result["最終判断"] == "主力").sum() if not result.empty else 0}',
             f'小口候補：{(result["最終判断"] == "小口").sum() if not result.empty else 0}']
    if night_reference:
        counts = {name: len(rows) for name, rows in night_reference.get("categories", {}).items()}
        lines += ["夜会（参考情報・自動昇格なし）：" + " / ".join(f"{name} {count}件" for name, count in counts.items()),
                  f"夜会有効期限：{night_reference.get('valid_until', '不明')}（朝会条件で必ず再確認）"]
    lines += ["⚠️ EDINETだけでは会社予想・上方下方修正・重要適時開示が不足します。",
              "不足項目を推測せず、現行の必須11項目判定は緩和しません。"]
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
          *(fundamental_message(r)),
          *(stocknote_message(r, result.attrs.get("stocknote_status")) if config.get("stocknote", {}).get("enabled", False) else []),
          f'分析担当：{r["分析コメント"]}', f'運用担当：{r["運用コメント"]}', f'実行条件：{r["注文理由"]}']
    if recheck: lines += ["", "気配・寄値の更新データなし：朝会から判断変更なし"]
    lines += ["", "※証券会社への自動発注は行いません。注文前に人間が価格・気配・材料を確認してください。"]
    return "\n".join(lines)


def stocknote_message(row, status: str | None) -> list[str]:
    """Compact advisory section; the complete payload is kept in the audit report."""
    if status == "response_missing" or not status:
        return ["stocknote分析社員：stocknote未取得"]
    if status.startswith("response_rejected"):
        reason = status.partition(":")[2].strip()
        return [f"stocknote分析社員：response_rejected（{reason[:80]}）"]
    if not str(row.get("stocknote_評価", "")).strip():
        return ["stocknote分析社員：stocknote未取得"]
    confidence = row.get("stocknote_信頼度", "")
    try: confidence = f"{float(confidence):.0%}"
    except (TypeError, ValueError): confidence = str(confidence)
    summary = str(row.get("stocknote_要約", "")).replace("\n", " ")[:160]
    return ["stocknote分析社員（参考情報・売買判断には未反映）：",
            f"評価 {row.get('stocknote_評価')} / 信頼度 {confidence} / {summary}",
            "PER・PBR等は参考値・公式未確認。詳細はstocknote分析監査レポート参照。"]


def fundamental_message(row) -> list[str]:
    if not bool(row.get("ファンダメンタル十分", False)):
        return ["ファンダメンタル：データ不足", "主力判定不可 → 小口または見送り"]
    def pct(name):
        try: return f'{float(row.get(name)):+.1f}%'
        except (TypeError, ValueError): return "欠損"
    def val(name, suffix=""):
        try: return f'{float(row.get(name)):.1f}{suffix}'
        except (TypeError, ValueError): return "欠損"
    material = str(row.get("重要適時開示", "")).strip() or "特になし"
    revision = str(row.get("業績修正", "")).strip()
    if revision: material = f"{revision} / {material}"
    reasons = [reason.strip() for reason in str(row.get("ファンダメンタル加減点理由", "")).split("/") if reason.strip()]
    reason_summary = " / ".join(reasons[:3]) + (f" / 他{len(reasons) - 3}件" if len(reasons) > 3 else "")
    return [f'ファンダメンタル：{int(row["ファンダメンタルスコア"])}/10 {row["ファンダメンタル評価"]}',
            f'利益 {row.get("利益状態") or "評価不能"} / 業績 {row.get("業績4象限") or "評価不能"} / 配当 {row.get("配当判定") or "評価不能"}',
            f'加減点：{reason_summary or "なし"}',
            f'売上 {pct("売上前年比")} / 営業利益 {pct("営業利益前年比")} / EPS {val("EPS")}',
            f'PER {val("PER", "倍")} / PBR {val("PBR", "倍")} / ROE {val("ROE", "%")}',
            f'自己資本比率 {val("自己資本比率", "%")} / 配当利回り {val("配当利回り", "%")}',
            f'今期予想：{row.get("今期会社予想") or "欠損"}', f'直近決算：{row.get("直近決算発表日") or "欠損"}',
            f'重要材料：{material}']


def load_market_input(config: dict, path: Path | None = None) -> list[dict] | pd.DataFrame:
    """Use manual observations when present, otherwise obtain external data."""
    market_input = path or ROOT / "data/market_input.csv"
    market_rows = read_csv_if_populated(market_input) if market_input.exists() else pd.DataFrame()
    return fetch_market_data(config) if market_rows.empty else market_rows


def join_candidate_sectors(candidates: pd.DataFrame, master_path: Path | None = None) -> pd.DataFrame:
    """Fill only missing candidate sectors from the existing JPX security master."""
    if candidates.empty:
        return candidates.copy()
    result = candidates.copy()
    if "業種" not in result:
        result["業種"] = ""
    path = master_path or ROOT / "stocks.csv"
    master = read_csv_if_populated(path, dtype={"code": str}) if path.exists() else pd.DataFrame()
    if not {"code", "industry"}.issubset(master.columns):
        return result
    sectors = (master.assign(code=master["code"].astype(str).str.strip())
               .drop_duplicates("code").set_index("code")["industry"])
    missing = result["業種"].fillna("").astype(str).str.strip().eq("")
    codes = result["コード"].astype(str).str.strip()
    result.loc[missing, "業種"] = codes[missing].map(sectors).fillna("")
    return result


def run_stocknote_cli(request_path: Path, timeout_seconds: float) -> bool:
    """Run the existing side CLI; optional analysis always fails open."""
    try:
        subprocess.run([sys.executable, "-m", "stocknote_side.cli", str(request_path)],
                       cwd=ROOT, check=True, timeout=timeout_seconds,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def run(kind="morning", notify=True, candidates_path: Path | None = None) -> Path:
    config, today = load_config(), now_tokyo().strftime("%Y%m%d")
    if candidates_path is None:
        files = sorted(ROOT.glob("anti_candidates_*.csv"), reverse=True); candidates_path = files[0] if files else None
    candidates = read_csv_if_populated(candidates_path, dtype={"コード": str}) if candidates_path else pd.DataFrame()
    if kind == "weekend":
        candidates = join_candidate_sectors(candidates)
    # A header-only CSV is a valid frame but contains no observations, so fallback is
    # deliberately decided after parsing rather than from an arbitrary byte size.
    market_rows = load_market_input(config)
    environment = analyze_market(market_rows, config); news = analyze_news(ROOT / "data/news_input.csv", config)
    jgb = analyze_jgb(fetch_jgb_data(config), environment.observed_at)
    impacts = sector_impacts(environment, config)
    for sector, score in jgb_sector_impacts(jgb, config).items(): impacts[sector] = impacts.get(sector, 0) + score
    save_market_environment(environment); save_jgb_analysis(jgb); save_news(news); save_sector_impacts(impacts, environment.observed_at)
    if kind == "night":
        night_result = generate_night_result(candidates, jgb, config)
        _, output = save_night_result(night_result)
        message = night_message(night_result)
        record_meeting_safely("夜会", night_result, candidates)
        if notify: post(message)
        print(message)
        return output
    if kind == "weekend":
        weekend_result = generate_weekend_result(candidates, jgb, config)
        stocknote = config.get("stocknote", {})
        if stocknote.get("enabled", False):
            exchange = ROOT / stocknote.get("exchange_directory", "data/stocknote")
            # The established stocknote contract accepts at most three candidates.
            # Discord TOP limits and the complete weekend audit remain unchanged.
            run_id, request_path = export_request(candidates.head(3), exchange)
            weekend_result["stocknote_employee"] = "request_exported"
            weekend_result["stocknote_run_id"] = run_id
            weekend_result["stocknote_request"] = str(request_path.relative_to(ROOT))
            run_stocknote_cli(request_path, float(stocknote.get("timeout_seconds", 300)))
            annotated, status = consume_shadow(
                candidates, exchange, run_id,
                max_age_hours=float(stocknote.get("max_response_age_hours", 24)))
            weekend_result["stocknote_employee"] = status
            if status == "accepted":
                by_code = {str(row["コード"]): row for _, row in annotated.iterrows()
                           if str(row.get("stocknote_評価", "")).strip()}
                weekend_result["stocknote_analyses"] = [{
                    "code": row["code"], "name": row["name"],
                    "assessment": by_code[row["code"]].get("stocknote_評価", ""),
                    "confidence": by_code[row["code"]].get("stocknote_信頼度", ""),
                    "contrarian_score": by_code[row["code"]].get("stocknote_逆張りスコア", ""),
                    "summary": by_code[row["code"]].get("stocknote_要約", ""),
                } for row in weekend_result["candidates"] if row["code"] in by_code]
        _, output = save_weekend_result(weekend_result)
        message = weekend_message(weekend_result)
        record_meeting_safely("週末会議", weekend_result, candidates)
        if notify: post(message)
        print(message)
        return output
    candidates = enrich_candidates(candidates, config)
    result = evaluate_candidates(candidates, config, environment, news, jgb)
    stocknote = config.get("stocknote", {})
    if stocknote.get("enabled", False):
        exchange = ROOT / stocknote.get("exchange_directory", "data/stocknote")
        run_id, _ = export_request(result, exchange)
        result, stocknote_status = consume_shadow(
            result, exchange, run_id, max_age_hours=float(stocknote.get("max_response_age_hours", 24)))
        write_shadow_report(result, ROOT / "reports/stocknote" / f"stocknote_shadow_{run_id}.md",
                            run_id, stocknote_status)
    folder = "morning" if kind == "recheck" else kind
    prefix = {"morning":"morning_meeting", "recheck":"morning_recheck", "close":"close_meeting", "weekly":"weekly_meeting"}[kind]
    output = ROOT / "reports/meeting" / folder / f"{prefix}_{today}.md"; output.parent.mkdir(parents=True, exist_ok=True)
    if kind in ("morning", "recheck"): message = morning_message(result, config, recheck=kind == "recheck", environment=environment, news=news, jgb=jgb, night_reference=load_latest_night_result())
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
        performance = read_csv_if_populated(ROOT / "data/performance.csv")
        write_research_report(performance, ROOT / "reports/proposals" / f"market_research_{today}.md",
                              config["meeting"]["minimum_backtest_samples"])
    output.write_text(message + "\n\n## 全候補監査表\n\n```csv\n" + result.to_csv(index=False) + "```\n", encoding="utf-8")
    # Machine-readable siblings make the exact meeting input/evaluation durable;
    # dashboard rendering continues to use the same persisted report table.
    result.to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(
        json.dumps(result.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if kind in ("morning", "recheck"):
        record_meeting_safely("朝会", result, candidates)
    if notify: post(message)
    print(message); return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--kind", choices=["night","weekend","morning","recheck","close","weekly"], default="morning"); parser.add_argument("--no-notify", action="store_true")
    args=parser.parse_args(); run(args.kind, not args.no_notify)
