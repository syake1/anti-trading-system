"""Streamlit review dashboard. Run: streamlit run dashboard.py."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from src.backtest import read_csv_if_populated, summary
from src.indicators import enrich
from src.materials import BUYBACK_COLUMNS, ensure_templates
from src.stochastic import stochastic
from src.utils import ROOT, load_config

st.set_page_config(page_title="日本株 反転初動スクリーナー", layout="wide")
st.title("📊 AI投資会議")
ensure_templates(ROOT)
files = sorted(ROOT.glob("anti_candidates_*.csv"), reverse=True)
candidates = read_csv_if_populated(files[0], dtype={"コード": str}) if files else pd.DataFrame()
stats = summary(); config = load_config()
from src.investment_meeting import evaluate_candidates
meeting = evaluate_candidates(candidates, config)
portfolio = config["portfolio"]
cols = st.columns(6)
values = [("運用資産", f'{portfolio["initial_capital"]:,.0f}円'),
          ("現金比率", f'{portfolio.get("current_cash", portfolio["initial_capital"])/portfolio["initial_capital"]:.0%}'),
          ("保有銘柄数", portfolio.get("current_positions", 0)),
          ("主力候補数", (meeting.get("最終分類") == "主力候補").sum() if not meeting.empty else 0),
          ("小口候補数", (meeting.get("最終分類") == "小口候補").sum() if not meeting.empty else 0),
          ("最大許容損失", f'{portfolio["initial_capital"]*portfolio["max_risk_per_trade_pct"]/100:,.0f}円')]
for col, (label, value) in zip(cols, values): col.metric(label, value)

upload = st.file_uploader("候補CSVをアップロード", type="csv")
if upload is not None: candidates = pd.read_csv(upload, dtype={"コード": str})
meeting = evaluate_candidates(candidates, config)
if candidates.empty:
    st.info("候補CSVは空です。スキャン後、またはCSVアップロード後に候補が表示されます。")
else:
    left, right = st.columns(2)
    with left:
        st.subheader("AI社員① 投資分析担当")
        st.dataframe(meeting[["コード", "銘柄名", "分析評価", "分析コメント"]], hide_index=True, use_container_width=True)
    with right:
        st.subheader("AI社員② 運用・検証担当")
        st.dataframe(meeting[["コード", "運用評価", "推奨株数", "必要資金", "運用コメント"]], hide_index=True, use_container_width=True)
    st.subheader("最終候補")
    st.dataframe(meeting, use_container_width=True, hide_index=True)
    shown = [c for c in ["コード", "会社名", "現在値", "ランク", "スコア", "シグナル種別", "RSI14", "BB位置",
             "直近3日騰落率", "直近5日騰落率", "25日線乖離率", "出来高倍率", "自社株買い比率", "除外理由"] if c in candidates]
    st.dataframe(candidates[shown], use_container_width=True, hide_index=True)
    st.download_button("候補一覧をダウンロード", candidates.to_csv(index=False).encode("utf-8-sig"), "candidates.csv", "text/csv")
    selected = st.selectbox("チャート銘柄", candidates["コード"].astype(str) + " " + candidates["会社名"].astype(str))
    code = selected.split()[0]
    raw = yf.download(f"{code}.T", period="2y", progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    if not raw.empty:
        config = load_config(); df = enrich(raw, config).join(stochastic(raw, 7, 10)).tail(150)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.55, .15, .15, .15])
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="価格"), row=1, col=1)
        for name, col, color in [("BB +2σ", "bb_upper", "gray"), ("BB -2σ", "bb_lower", "gray"), ("MA25", "MA25", "orange"), ("MA75", "MA75", "blue"), ("MA200", "MA200", "purple")]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, line={"color": color, "width": 1}), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="出来高"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.RSI14, name="RSI"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.K, name="%K"), row=4, col=1); fig.add_trace(go.Scatter(x=df.index, y=df.D, name="%D"), row=4, col=1)
        dates = pd.to_datetime(candidates.loc[candidates["コード"].astype(str) == code, "シグナル日"])
        buys = df.loc[df.index.normalize().isin(dates.dt.normalize())]
        fig.add_trace(go.Scatter(x=buys.index, y=buys.Low, mode="markers+text", text="BUY", textposition="bottom center", marker={"symbol": "triangle-up", "size": 14, "color": "red"}, name="BUY"), row=1, col=1)
        fig.update_layout(height=850, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)

st.subheader("手動材料CSV")
for path in (ROOT / "data/buybacks.csv", ROOT / "data/events.csv"):
    data = path.read_bytes()
    st.download_button(f"{path.name} をダウンロード", data, path.name, "text/csv", key=path.name)
