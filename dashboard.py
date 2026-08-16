"""One-page Streamlit review dashboard. Run: streamlit run dashboard.py."""
from functools import partial
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from src.backtest import order_method_summary, read_csv_if_populated
from src.dashboard_exports import (csv_download_data, dated_csv_filename, latest_stocknote_analysis,
                                   candidate_detail_rows, candidate_selector_labels, latest_meeting_view,
                                   load_meeting_reports, meeting_history_summary,
                                   ranked_buy_candidates, read_candidate_csv,
                                   strategy_performance)
from src.indicators import enrich
from src.investment_meeting import evaluate_candidates
from src.utils import ROOT, load_config

st.set_page_config(page_title="日本株 反転初動ダッシュボード", page_icon="📊", layout="wide")
st.title("📊 anti-trading-system 意思決定ダッシュボード")
st.caption("候補選定から検証までを一画面で確認します。表示内容は人間の確認用で、自動発注は行いません。")


def value(row: pd.Series, key: str, default="—"):
    item = row.get(key, default)
    return default if pd.isna(item) or item == "" else item


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(code: str) -> pd.DataFrame:
    try:
        raw = yf.download(f"{code}.T", period="2y", progress=False, auto_adjust=False, timeout=10)
    except Exception:  # Network/provider failures must not take down the review screen.
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    try:
        return enrich(raw, load_config()).tail(220)
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()


def price_chart(frame: pd.DataFrame, plan: pd.Series | None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.03,
                        row_heights=[.65, .18, .17], subplot_titles=("価格・テクニカル", "出来高", "RSI (14)"))
    fig.add_trace(go.Candlestick(x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"],
                                 close=frame["Close"], name="ローソク足"), row=1, col=1)
    for label, column, color in (("BB +2σ", "bb_upper", "#8b949e"), ("BB -2σ", "bb_lower", "#8b949e"),
                                  ("MA25", "MA25", "#f59e0b"), ("MA75", "MA75", "#3b82f6"),
                                  ("MA200", "MA200", "#a855f7")):
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=label,
                                     line={"color": color, "width": 1.2}), row=1, col=1)
    fig.add_trace(go.Bar(x=frame.index, y=frame["Volume"], name="出来高", marker_color="#64748b"), row=2, col=1)
    if "RSI14" in frame:
        fig.add_trace(go.Scatter(x=frame.index, y=frame["RSI14"], name="RSI14", line={"color": "#22c55e"}), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#3b82f6", row=3, col=1)
    if plan is not None:
        for label, key, color in (("エントリー下限", "買いゾーン下限", "#2563eb"), ("エントリー上限", "買いゾーン上限", "#2563eb"),
                                  ("逆指値発動", "逆指値発動価格", "#16a34a"), ("損切り", "損切り価格", "#dc2626"),
                                  ("利確", "利確目標", "#9333ea")):
            number = pd.to_numeric(pd.Series([plan.get(key)]), errors="coerce").iloc[0]
            if pd.notna(number):
                fig.add_hline(y=float(number), annotation_text=label, line_color=color, line_dash="dash", row=1, col=1)
    fig.update_layout(height=720, margin={"l": 20, "r": 20, "t": 45, "b": 20}, xaxis_rangeslider_visible=False,
                      legend={"orientation": "h", "y": 1.04})
    return fig


def select_ranked_row(ranked: pd.DataFrame) -> None:
    """Synchronize a ranking-table click to the candidate detail selector."""
    rows = st.session_state["buy-candidate-ranking"].selection.rows
    if rows:
        st.session_state["selected_candidate_code"] = str(ranked.iloc[rows[0]]["コード"])


# Candidate CSV input and normalization
files = sorted(ROOT.glob("anti_candidates_*.csv"), reverse=True)
upload = st.file_uploader("候補CSVを読み込む", type="csv", help="未指定時はリポジトリ内の最新候補CSVを使用します。")
source = upload if upload is not None else (files[0] if files else None)
if source is None:
    candidates, csv_error = pd.DataFrame(), "候補CSVがありません。"
else:
    candidates, csv_error = read_candidate_csv(source)
if csv_error:
    st.warning(csv_error)
ranked = ranked_buy_candidates(candidates, 10)
config = load_config()
try:
    meeting = evaluate_candidates(ranked.drop(columns=["順位"], errors="ignore"), config)
except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
    meeting = pd.DataFrame()
    st.warning(f"候補データの欠損により注文案を計算できませんでした: {exc}")

# Persisted reports are the source of truth for the meeting sections.  The
# currently uploaded candidates are intentionally not presented as a saved meeting.
saved_meetings = load_meeting_reports(ROOT / "reports/meeting")
performance = read_csv_if_populated(ROOT / "data/performance.csv", dtype={"コード": str})
stocknote_all, stocknote_status = latest_stocknote_analysis(ROOT / "data/stocknote")
today_meeting = latest_meeting_view(saved_meetings, stocknote_all)

st.header("本日のAI投資会議")
if not today_meeting:
    st.info("まだ記録がありません")
else:
    st.caption(f'会議日時: {today_meeting["date"]}（保存記録: {Path(today_meeting["source"]).name}）')
    system_col, note_col = st.columns(2)
    with system_col:
        st.subheader("システム評価社員の意見")
        st.write(today_meeting["system_opinion"])
    with note_col:
        st.subheader("stocknote分析社員の意見")
        st.caption(stocknote_status)
        st.write(today_meeting["stocknote_opinion"])
    agreement_col, conflict_col = st.columns(2)
    with agreement_col:
        st.subheader("一致した点")
        st.write(today_meeting["agreement"])
    with conflict_col:
        st.subheader("対立した点")
        st.write(today_meeting["conflict"])
    st.subheader("最終判断")
    st.write(today_meeting["decision"])
    reason_col, risk_col = st.columns(2)
    with reason_col:
        st.markdown("**採用・見送り理由**")
        st.write(today_meeting["reason"])
    with risk_col:
        st.markdown("**リスク上の注意点**")
        st.write(today_meeting["risk"])
    st.subheader("ファンダメンタル評価")
    st.caption("保存された会議記録の公式取得データのみを表示します。欠損値は推測せず、データ不足として扱います。")
    st.dataframe(today_meeting["fundamentals"], hide_index=True, use_container_width=True,
                 column_config={"ファンダメンタル参照先": st.column_config.LinkColumn(display_text="公式資料")})
    st.markdown("**注文条件**")
    st.dataframe(today_meeting["orders"], hide_index=True, use_container_width=True)

st.header("会議・検証の履歴")
meeting_stats = meeting_history_summary(saved_meetings, performance)
if meeting_stats.empty:
    st.info("まだ記録がありません")
else:
    st.dataframe(meeting_stats, hide_index=True, use_container_width=True,
                 column_config={"翌日平均": st.column_config.NumberColumn(format="%.2f%%"),
                                "3日後平均": st.column_config.NumberColumn(format="%.2f%%"),
                                "5日後平均": st.column_config.NumberColumn(format="%.2f%%"),
                                "翌日勝率": st.column_config.NumberColumn(format="%.1%%"),
                                "3日後勝率": st.column_config.NumberColumn(format="%.1%%"),
                                "5日後勝率": st.column_config.NumberColumn(format="%.1%%")})
    if meeting_stats[["翌日平均", "3日後平均", "5日後平均"]].isna().all().all():
        st.info("翌日・3日後・5日後の会議対象と一致する検証実績は、まだ記録がありません")

proposal_files = sorted((ROOT / "reports/proposals").glob("*.md"), reverse=True)
with st.expander("条件・システム変更履歴 / 過去バージョンとの成績比較", expanded=False):
    if proposal_files:
        for proposal in proposal_files:
            st.markdown(f"**{proposal.name}**")
            try:
                st.markdown(proposal.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                st.warning(f"{proposal.name} を読み込めませんでした。")
    else:
        st.info("条件・システム変更履歴: まだ記録がありません")
    st.info("過去バージョンとの成績比較: まだ記録がありません")

st.header("1. 買い候補ランキング TOP 10")
if ranked.empty:
    st.info("表示できる買い候補はありません。空CSV・必須列不足の場合も他の集計は引き続き確認できます。")
else:
    table_columns = [column for column in ("順位", "テクニカル順位", "調整後順位", "コード", "会社名", "ランク", "テクニカルスコア", "信用需給による減点", "信用需給スコア", "総合調整後スコア", "信用需給判定", "信用需給判定理由", "現在値", "RSI14", "BB位置", "出来高倍率", "シグナル種別") if column in ranked]
    st.dataframe(
        ranked[table_columns], hide_index=True, use_container_width=True,
        key="buy-candidate-ranking", on_select=partial(select_ranked_row, ranked),
        selection_mode="single-row",
    )
    labels = candidate_selector_labels(ranked)
    codes = list(labels)
    if st.session_state.get("selected_candidate_code") not in codes:
        st.session_state["selected_candidate_code"] = codes[0]
    st.selectbox(
        "チャートと詳細を確認する銘柄（順位｜コード｜銘柄名）",
        codes,
        format_func=labels.get,
        key="selected_candidate_code",
        help="選択ボックス、または上のランキング表の行をクリックして切り替えられます。",
    )
    st.download_button("候補TOP10をダウンロード", csv_download_data(ranked), dated_csv_filename("buy_candidates_top10"),
                       "text/csv", key="top10-download")

if not ranked.empty:
    code = st.session_state["selected_candidate_code"]
    candidate, plan, note, selected_performance = candidate_detail_rows(
        ranked, meeting, stocknote_all, performance, code,
    )

    st.header(f"2. 銘柄チャート — {code} {value(candidate, '会社名', '')}")
    prices = load_prices(code)
    if prices.empty:
        st.warning("価格データを取得できませんでした。ネットワークまたは銘柄データを確認してください。")
    else:
        st.plotly_chart(price_chart(prices, plan), use_container_width=True)

    reason_col, stocknote_col = st.columns(2)
    with reason_col:
        st.subheader("3. 選定理由")
        st.write(value(candidate, "判定理由", "選定理由は未記録です。"))
        st.dataframe(pd.DataFrame({"項目": ["ランク", "スコア", "RSI14", "BB位置", "出来高倍率", "ローソク足"],
                                   "値": [str(value(candidate, key)) for key in ("ランク", "スコア", "RSI14", "BB位置", "出来高倍率", "ローソク足パターン")]}),
                     hide_index=True, use_container_width=True)
    with stocknote_col:
        st.subheader("4. stocknote分析社員の評価（参考）")
        st.caption(stocknote_status)
        if note.empty:
            st.info("この銘柄のstocknote評価はありません。")
        else:
            note = note.iloc[0]
            st.metric("評価", value(note, "assessment"), value(note, "confidence", "信頼度未提示"))
            st.write(value(note, "summary", "要約なし"))
            st.dataframe(pd.DataFrame({"参考項目": ["推奨買い価格", "予想売り価格", "最終目標価格", "注意点"],
                                       "参考値": [value(note, key) for key in ("recommended_buy_price", "expected_sell_price", "final_target_price", "cautions")]}),
                         hide_index=True, use_container_width=True)
        st.warning("stocknoteの価格・RRは参考表示のみです。既存の注文価格、損切り、利確、逆指値、株数計算には反映しません。")

    st.header("5. 既存ロジックによる注文・リスク管理")
    if plan is None:
        st.info("注文案は計算できませんでした。候補CSVの価格・ATR等の列を確認してください。")
    else:
        decision, sizing = st.columns([2, 3])
        with decision:
            st.metric("最終判断", value(plan, "最終判断"))
            st.metric("市場・資金評価", value(plan, "資金・リスク評価"))
            st.write(value(plan, "注文理由", "注文理由なし"))
        with sizing:
            fields = ("注文方式", "買いゾーン下限", "買いゾーン上限", "逆指値発動価格", "損切り価格", "利確目標", "RR", "推奨株数", "必要資金", "最大想定損失")
            st.dataframe(pd.DataFrame({"項目": fields, "既存ロジックの計算値": [str(value(plan, key)) for key in fields]}),
                         hide_index=True, use_container_width=True)
        st.info("強いリスクオフ時の新規主力停止、警戒相場での小口化、資金・損失・保有上限は既存の投資会議ロジックをそのまま適用しています。")

    st.header("6. 翌日・3日後・5日後の成績")
    if selected_performance.empty:
        st.info("この銘柄の評価可能な過去成績はまだありません。")
    else:
        st.dataframe(selected_performance, hide_index=True, use_container_width=True)
st.header("7. 過去成績・戦略別比較")
if performance.empty:
    st.info("過去成績はまだありません。空の成績CSVでも画面は継続します。")
else:
    horizons = [column for column in ("シグナル日", "コード", "ランク", "シグナル種別", "1日後騰落率", "3日後騰落率", "5日後騰落率", "利確到達", "損切り到達") if column in performance]
    with st.expander("過去成績の明細", expanded=False):
        st.dataframe(performance[horizons].sort_values("シグナル日", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("過去成績CSVをダウンロード", csv_download_data(performance), dated_csv_filename("performance"), "text/csv")
    comparison = strategy_performance(performance)
    if comparison.empty:
        st.info("戦略別に比較できるデータがありません。")
    else:
        st.caption("旧CSVでシグナル種別が空欄の行は「買い・売り」列から復元しています。騰落率は売買方向を反映した戦略損益率です。")
        st.dataframe(comparison, hide_index=True, use_container_width=True,
                     column_config={"1日平均": st.column_config.NumberColumn(format="%.2f%%"),
                                    "3日平均": st.column_config.NumberColumn(format="%.2f%%"),
                                    "5日平均": st.column_config.NumberColumn(format="%.2f%%"),
                                    "5日勝率": st.column_config.NumberColumn(format="%.1%%")})

st.subheader("注文方式別比較")
method_stats = order_method_summary()
if method_stats:
    st.dataframe(pd.DataFrame.from_dict(method_stats, orient="index"), use_container_width=True)
else:
    st.info("注文方式別の仮想売買サンプルを蓄積中です。")
