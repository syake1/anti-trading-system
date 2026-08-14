"""Streamlit review dashboard. Run: streamlit run dashboard.py."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from src.backtest import order_method_summary, read_csv_if_populated, summary
from src.indicators import enrich
from src.materials import BUYBACK_COLUMNS, ensure_templates
from src.stochastic import stochastic
from src.utils import ROOT, load_config
from src.japan_rates import JapanRates, rate_sector_impacts

st.set_page_config(page_title="日本株 反転初動スクリーナー", layout="wide")
st.title("📊 AI投資会議")
ensure_templates(ROOT)
st.header("🌏 市場環境")
market_history = read_csv_if_populated(ROOT / "data/market_environment.csv")
news_history = read_csv_if_populated(ROOT / "data/news_events.csv")
sector_history = read_csv_if_populated(ROOT / "data/sector_impact.csv")
japan_rate_history = read_csv_if_populated(ROOT / "data/japan_rates.csv")
if market_history.empty:
    st.info("市場データ未取得です。取得不能な値は推測せず、朝会実行後に表示します。")
else:
    latest = market_history[market_history["date"] == market_history["date"].iloc[-1]]
    a, b = st.columns([2, 1])
    a.dataframe(latest[[c for c in ["indicator", "current", "previous_close", "change", "change_pct", "short_change_pct", "change_bp", "score"] if c in latest]], hide_index=True, use_container_width=True)
    b.metric("市場判定", latest["market_regime"].iloc[-1]); b.metric("合計スコア", latest["total_score"].iloc[-1])
if not news_history.empty:
    st.subheader("重大ニュース（取得元・時刻・URLを監査）")
    st.dataframe(news_history, hide_index=True, use_container_width=True)
else: st.caption("信頼済みニュースは0件です。未設定時は data/news_input.csv を利用できます。")
if not sector_history.empty:
    st.subheader("業種別影響"); st.dataframe(sector_history.tail(30), hide_index=True, use_container_width=True)
st.header("🇯🇵 日本金利")
if japan_rate_history.empty:
    current_japan_rates = JapanRates(timestamp="")
    st.info("日本金利データ：取得不可（欠損のまま他の市場分析を継続します）")
else:
    raw_rate = japan_rate_history.iloc[-1].where(pd.notna(japan_rate_history.iloc[-1]), None).to_dict()
    allowed = JapanRates.__dataclass_fields__.keys()
    current_japan_rates = JapanRates(**{k: raw_rate.get(k) for k in allowed if k in raw_rate})
    rate_cols = st.columns(4)
    rate_cols[0].metric("日本2年", f"{current_japan_rates.jp_2y_yield:.3f}%" if current_japan_rates.jp_2y_yield is not None else "NA")
    rate_cols[1].metric("日本10年", f"{current_japan_rates.jp_10y_yield:.3f}%" if current_japan_rates.jp_10y_yield is not None else "NA", f"{current_japan_rates.jp_10y_change_bp:+.1f}bp" if current_japan_rates.jp_10y_change_bp is not None else None)
    rate_cols[2].metric("日本30年", f"{current_japan_rates.jp_30y_yield:.3f}%" if current_japan_rates.jp_30y_yield is not None else "NA")
    rate_cols[3].metric("10年－2年", f"{current_japan_rates.jp_10y_2y_spread:.3f}%" if current_japan_rates.jp_10y_2y_spread is not None else "NA")
    st.write(f"5日変化 **{current_japan_rates.jp_10y_change_5d_bp:+.1f}bp** / 20日変化 **{current_japan_rates.jp_10y_change_20d_bp:+.1f}bp**" if current_japan_rates.jp_10y_change_5d_bp is not None and current_japan_rates.jp_10y_change_20d_bp is not None else "5日・20日変化: NA")
    st.write(f"金利環境判定: **{current_japan_rates.jp_rate_regime}** / 日銀金融政策警戒: **{'あり' if current_japan_rates.boj_tightening_risk else 'なし'}**")
    st.dataframe(pd.DataFrame([rate_sector_impacts(current_japan_rates)]).T.rename(columns={0:"影響スコア"}), use_container_width=True)
files = sorted(ROOT.glob("anti_candidates_*.csv"), reverse=True)
candidates = read_csv_if_populated(files[0], dtype={"コード": str}) if files else pd.DataFrame()
stats = summary(); config = load_config()
from src.investment_meeting import evaluate_candidates
meeting = evaluate_candidates(candidates, config, japan_rates=current_japan_rates)
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
meeting = evaluate_candidates(candidates, config, japan_rates=current_japan_rates)
if candidates.empty:
    st.info("候補CSVは空です。スキャン後、またはCSVアップロード後に候補が表示されます。")
else:
    left, right = st.columns(2)
    with left:
        st.subheader("AI社員① 投資分析担当")
        st.dataframe(meeting[["コード", "銘柄名", "分析評価", "日本金利影響", "日本金利影響理由", "分析コメント"]], hide_index=True, use_container_width=True)
    with right:
        st.subheader("AI社員② 運用・検証担当")
        st.dataframe(meeting[["コード", "運用評価", "推奨株数", "必要資金", "運用コメント"]], hide_index=True, use_container_width=True)
    st.subheader("最終候補")
    st.dataframe(meeting, use_container_width=True, hide_index=True)
    st.caption("最終注文案は人間による確認用です。証券会社へ自動発注しません。")
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
        proposal = meeting.loc[meeting["コード"].astype(str) == code]
        if not proposal.empty:
            proposal = proposal.iloc[0]
            for label, key, color, dash in [("買いゾーン下限", "買いゾーン下限", "royalblue", "dot"),
                    ("買いゾーン上限", "買いゾーン上限", "royalblue", "dot"),
                    ("逆指値", "逆指値発動価格", "green", "dash"), ("追いかけ禁止", "追いかけ禁止価格", "orange", "dash"),
                    ("損切り", "損切り価格", "red", "dash"), ("利確", "利確目標", "purple", "dash")]:
                fig.add_hline(y=float(proposal[key]), line_color=color, line_dash=dash,
                              annotation_text=label, row=1, col=1)
        fig.update_layout(height=850, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)

st.subheader("注文方式比較")
method_stats = order_method_summary()
if method_stats:
    comparison = pd.DataFrame.from_dict(method_stats, orient="index").rename(index={"market":"寄り成り", "limit":"指値", "stop":"逆指値"})
    st.dataframe(comparison[["サンプル数", "約定率", "勝率", "平均損益", "PF", "MFE", "MAE", "最大DD"]], use_container_width=True)
else:
    st.info("注文方式別の仮想売買サンプルを蓄積中です。未約定は損失に含めません。")

st.subheader("手動材料CSV")
for path in (ROOT / "data/buybacks.csv", ROOT / "data/events.csv"):
    data = path.read_bytes()
    st.download_button(f"{path.name} をダウンロード", data, path.name, "text/csv", key=path.name)
