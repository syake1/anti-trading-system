from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

from src.anti_signal import detect
from src.candlestick import patterns
from src.discord_notify import candidate_message, post
from src.indicators import enrich
from src.scoring import rank, score
from src.stochastic import stochastic
from src.utils import ROOT, load_config, now_tokyo, save_json


RESULT_COLUMNS = ["シグナル日", "コード", "会社名", "市場", "現在値", "前日比", "スコア", "ランク", "%K", "%D",
                  "%D傾き", "RSI14", "MA25", "MA75", "MA200", "BB位置", "出来高倍率", "ローソク足パターン",
                  "アンチ判定", "買い・売り", "損切り候補", "利確候補", "RR", "判定理由", "Yahoo Financeリンク"]


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()


def analyze(data: pd.DataFrame, stock: dict, config: dict) -> dict | None:
    df = normalize(data)
    if len(df) < 205: return None
    recent = df.iloc[-20:]
    liquidity = config["liquidity"]
    avg_volume = recent.Volume.mean()
    avg_value = (recent.Close * recent.Volume).mean()
    if (df.Close.iloc[-1] < liquidity["min_price"]
            or avg_volume < liquidity["min_avg_volume_20"]
            or avg_value < liquidity.get("min_avg_trading_value_20", 0)):
        return None
    df = enrich(df, config).join(
        stochastic(df, config["stochastic"]["k_period"], config["stochastic"]["d_period"])
    )
    now, prev = df.iloc[-1], df.iloc[-2]
    signal = detect(df, config["stochastic"]["d_slope_days"])
    if not signal: return None
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


def _ticker_frame(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if not isinstance(download.columns, pd.MultiIndex):
        return download
    for level in range(download.columns.nlevels):
        if ticker in download.columns.get_level_values(level):
            return download.xs(ticker, axis=1, level=level, drop_level=True)
    return pd.DataFrame()


def _download_batch(tickers: list[str], config: dict) -> pd.DataFrame:
    return yf.download(
        tickers, period=config["scan"]["history_period"], group_by="column",
        # yfinance の既定値に任せるとバッチ内の全銘柄へ同時に接続し得るため、
        # 同時接続数にも上限を設ける。
        threads=max(1, int(config["scan"].get("download_threads", 5))),
        progress=False, auto_adjust=False,
        timeout=config["scan"].get("download_timeout", 30),
    )


def _download_with_retry(tickers: list[str], config: dict) -> dict[str, pd.DataFrame]:
    """Download a batch, retrying only missing tickers with exponential backoff."""
    frames = {ticker: pd.DataFrame() for ticker in tickers}
    pending = list(tickers)
    attempts = max(1, int(config["scan"].get("download_max_attempts", 4)))
    backoff = max(0, float(config["scan"].get("retry_backoff_seconds", 5)))

    for attempt in range(1, attempts + 1):
        try:
            downloaded = _download_batch(pending, config)
            for ticker in pending:
                frame = _ticker_frame(downloaded, ticker)
                required = {"Open", "High", "Low", "Close", "Volume"}
                if required.issubset(frame.columns) and not frame.dropna(subset=list(required)).empty:
                    frames[ticker] = frame
        except Exception as exc:
            # 429に限らず一時的な通信障害も同じ方法で回復させる。
            print(f"一括取得失敗 (試行 {attempt}/{attempts}, {len(pending)}銘柄): {exc}")

        pending = [ticker for ticker in pending if frames[ticker].empty]
        if not pending:
            break
        if attempt < attempts:
            delay = backoff * (2 ** (attempt - 1))
            print(f"未取得 {len(pending)}銘柄を {delay:g}秒後に再取得します")
            time.sleep(delay)
    return frames


def run(notify: bool = True) -> Path:
    started_at = time.monotonic()
    config = load_config()
    stocks = pd.read_csv(ROOT / "stocks.csv", dtype={"code": str}).to_dict("records")
    output = ROOT / f'anti_candidates_{now_tokyo():%Y%m%d}.csv'
    if not stocks:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
        save_json(ROOT / "data/watchlist.json", [])
        print("対象銘柄数: 0\n取得成功数: 0\n取得失敗数: 0\n判定完了数: 0\nSランク件数: 0\nAランク件数: 0")
        print(f"処理時間: {time.monotonic() - started_at:.1f}秒")
        return output
    limit = int(config["scan"].get("scan_limit", 0))
    if limit > 0:
        stocks = stocks[:limit]
    total = len(stocks)
    print(f"対象銘柄数: {total}")
    print("処理開始")
    rows = []
    success = failed = completed = liquid_count = 0
    failed_stocks = []
    batch_size = max(1, int(config["scan"].get("batch_size", 50)))
    for start in range(0, total, batch_size):
        batch = stocks[start:start + batch_size]
        tickers = [f'{stock["code"]}.T' for stock in batch]
        frames = _download_with_retry(tickers, config)
        for stock, ticker in zip(batch, tickers):
            data = frames[ticker]
            try:
                normalized = normalize(data)
                if normalized.empty:
                    raise ValueError("空データ")
            except Exception as exc:
                failed += 1
                failed_stocks.append({"コード": stock["code"], "会社名": stock["name"], "理由": str(exc)})
                print(f"{stock['code']} 取得失敗 → スキップ: {exc}")
                continue
            success += 1
            try:
                recent = normalized.iloc[-20:]
                liquidity = config["liquidity"]
                if (len(normalized) >= 205 and normalized.Close.iloc[-1] >= liquidity["min_price"]
                        and recent.Volume.mean() >= liquidity["min_avg_volume_20"]
                        and (recent.Close * recent.Volume).mean() >= liquidity.get("min_avg_trading_value_20", 0)):
                    liquid_count += 1
                candidate = analyze(normalized, stock, config)
                completed += 1
                if candidate:
                    rows.append(candidate)
            except Exception as exc:
                print(f"{stock['code']} 判定失敗 → スキップ: {exc}")
        done = min(start + len(batch), total)
        print(f"{done} / {total}")
        if done < total:
            time.sleep(max(0, float(config["scan"].get("batch_pause_seconds", 0.5))))
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values("スコア", ascending=False, ignore_index=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    failure_log = ROOT / f'data/download_failures_{now_tokyo():%Y%m%d}.csv'
    pd.DataFrame(failed_stocks, columns=["コード", "会社名", "理由"]).to_csv(
        failure_log, index=False, encoding="utf-8-sig"
    )
    if rows:
        history = ROOT / "data/signal_history.csv"
        result.to_csv(history, mode="a", header=not history.exists() or not history.stat().st_size, index=False, encoding="utf-8-sig")
        selected = result[result["ランク"].isin(config["scan"].get("watchlist_ranks", ["S", "A"]))]
        selected = selected.head(int(config["scan"].get("watchlist_max_stocks", 50)))
        save_json(ROOT / "data/watchlist.json", selected[["コード", "会社名", "買い・売り", "シグナル日"]].to_dict("records"))
        if notify:
            alerts = result[result["ランク"].isin(config["scan"]["notify_ranks"])]
            for row in alerts.head(int(config["scan"].get("discord_max_alerts", 20))).to_dict("records"):
                post(candidate_message(row))
    else: save_json(ROOT / "data/watchlist.json", [])
    print(f"対象銘柄数: {total}")
    print(f"取得成功数: {success}")
    print(f"取得失敗数: {failed}")
    print(f"判定完了数: {completed}")
    print(f"一次フィルター通過: {liquid_count}")
    print(f"アンチ候補件数: {len(result)}")
    print(f"Sランク件数: {(result['ランク'] == 'S').sum() if not result.empty else 0}")
    print(f"Aランク件数: {(result['ランク'] == 'A').sum() if not result.empty else 0}")
    print(f"処理時間: {time.monotonic() - started_at:.1f}秒")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()
    print(run(notify=not args.no_notify))
