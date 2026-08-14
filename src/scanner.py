from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.anti_signal import detect
from src.candlestick import patterns
from src.discord_notify import candidate_message, post
from src.indicators import enrich
from src.scoring import rank, score
from src.stochastic import stochastic
from src.utils import ROOT, load_config, now_tokyo, save_json


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()


def analyze(data: pd.DataFrame, stock: dict, config: dict) -> dict | None:
    df = normalize(data)
    if len(df) < 205: return None
    df = enrich(df, config).join(
        stochastic(df, config["stochastic"]["k_period"], config["stochastic"]["d_period"])
    )
    now, prev = df.iloc[-1], df.iloc[-2]
    liquid = now.Close >= config["liquidity"]["min_price"] and now.volume_avg20 >= config["liquidity"]["min_avg_volume_20"]
    signal = detect(df, config["stochastic"]["d_slope_days"])
    if not liquid or not signal: return None
    pats = patterns(df, signal["side"])
    value, reasons = score(df, signal, pats, config)
    side = signal["side"]
    lookback = config["risk"]["lookback_days"]
    structural = df.Low.iloc[-lookback:].min() if side == "buy" else df.High.iloc[-lookback:].max()
    atr_stop = now.Close - config["risk"]["atr_multiplier"] * now.ATR14 if side == "buy" else now.Close + config["risk"]["atr_multiplier"] * now.ATR14
    stop = min(structural, atr_stop) if side == "buy" else max(structural, atr_stop)
    risk = abs(now.Close - stop)
    target = now.Close + risk * config["risk"]["reward_risk"] * (1 if side == "buy" else -1)
    date = pd.Timestamp(df.index[-1]).date().isoformat()
    return {"シグナル日": date, "コード": str(stock["code"]), "会社名": stock["name"], "市場": stock["market"],
            "現在値": round(now.Close, 2), "前日比": round((now.Close / prev.Close - 1) * 100, 2), "スコア": value,
            "ランク": rank(value, config), "%K": round(now.K, 2), "%D": round(now.D, 2),
            "%D傾き": round(now.D - df.D.iloc[-1-config["stochastic"]["d_slope_days"]], 2), "RSI14": round(now.RSI14, 2),
            "MA25": round(now.MA25, 2), "MA75": round(now.MA75, 2), "MA200": round(now.MA200, 2),
            "BB位置": f'{now.bb_sigma:+.2f}σ', "出来高倍率": round(now.volume_ratio, 2),
            "ローソク足パターン": " / ".join(pats) or "なし", "アンチ判定": "強" if signal["cross"] else "通常",
            "買い・売り": "買い" if side == "buy" else "売り", "損切り候補": round(stop, 2),
            "利確候補": round(target, 2), "RR": config["risk"]["reward_risk"], "判定理由": "、".join(reasons),
            "Yahoo Financeリンク": f'https://finance.yahoo.co.jp/quote/{stock["code"]}.T'}


def run(notify: bool = True) -> Path:
    config = load_config()
    stocks = pd.read_csv(ROOT / "stocks.csv", dtype={"code": str}).to_dict("records")
    rows = []
    for stock in stocks:
        try:
            data = yf.download(f'{stock["code"]}.T', period=config["scan"]["history_period"], progress=False, auto_adjust=False)
            candidate = analyze(data, stock, config)
            if candidate: rows.append(candidate)
        except Exception as exc:
            print(f'{stock["code"]}: {exc}')
    result = pd.DataFrame(rows)
    output = ROOT / f'anti_candidates_{now_tokyo():%Y%m%d}.csv'
    result.to_csv(output, index=False, encoding="utf-8-sig")
    if rows:
        history = ROOT / "data/signal_history.csv"
        result.to_csv(history, mode="a", header=not history.exists() or not history.stat().st_size, index=False, encoding="utf-8-sig")
        selected = result[result["ランク"].isin(config["scan"]["notify_ranks"])]
        save_json(ROOT / "data/watchlist.json", selected[["コード", "会社名", "買い・売り", "シグナル日"]].to_dict("records"))
        if notify:
            for row in selected.to_dict("records"): post(candidate_message(row))
    else: save_json(ROOT / "data/watchlist.json", [])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()
    print(run(notify=not args.no_notify))
