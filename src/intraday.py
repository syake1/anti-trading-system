from __future__ import annotations

import pandas as pd
import yfinance as yf
from src.discord_notify import post
from src.stochastic import stochastic
from src.utils import ROOT, load_config, load_json, now_tokyo, save_json


def psar(df: pd.DataFrame, step=.02, maximum=.2) -> pd.Series:
    values, bull, af, extreme = [df.Low.iloc[0]], True, step, df.High.iloc[0]
    for i in range(1, len(df)):
        value = values[-1] + af * (extreme - values[-1])
        if bull:
            value = min(value, df.Low.iloc[i-1], df.Low.iloc[max(0, i-2)])
            if df.Low.iloc[i] < value: bull, value, af, extreme = False, extreme, step, df.Low.iloc[i]
            elif df.High.iloc[i] > extreme: extreme, af = df.High.iloc[i], min(maximum, af + step)
        else:
            value = max(value, df.High.iloc[i-1], df.High.iloc[max(0, i-2)])
            if df.High.iloc[i] > value: bull, value, af, extreme = True, extreme, step, df.High.iloc[i]
            elif df.Low.iloc[i] < extreme: extreme, af = df.Low.iloc[i], min(maximum, af + step)
        values.append(value)
    return pd.Series(values, index=df.index)


def market_open() -> bool:
    now = now_tokyo()
    minute = now.hour * 60 + now.minute
    return now.weekday() < 5 and ((9*60 <= minute <= 11*60+30) or (12*60+30 <= minute <= 15*60+30))


def run() -> None:
    if not market_open(): print("東京市場時間外"); return
    watch = load_json(ROOT / "data/watchlist.json", [])
    state = load_json(ROOT / "data/alert_state.json", {})
    config = load_config()["stochastic"]
    for item in watch:
        code = str(item["コード"])
        df = yf.download(f"{code}.T", period="5d", interval="15m", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 25: continue
        # yfinanceの最後の足は形成中の可能性があるため、一つ前の確定足だけを評価する。
        df = df.iloc[:-1].copy().join(stochastic(df.iloc[:-1], config["k_period"], config["d_period"]))
        df["PSAR"] = psar(df)
        a, b = df.iloc[-2], df.iloc[-1]
        side = item["買い・売り"]
        reversal = b.K > a.K if side == "買い" else b.K < a.K
        cross = (b.K > b.D and a.K <= a.D) if side == "買い" else (b.K < b.D and a.K >= a.D)
        sar = (b.Close > b.PSAR and a.Close <= a.PSAR) if side == "買い" else (b.Close < b.PSAR and a.Close >= a.PSAR)
        candle = b.Close > b.Open if side == "買い" else b.Close < b.Open
        volume = b.Volume > df.Volume.iloc[-21:-1].mean() * 1.3
        triggers = [name for name, ok in {"%K反転": reversal, "%K%Dクロス": cross, "SAR転換": sar, "出来高増加": volume, "ローソク転換": candle}.items() if ok]
        if reversal and (cross or sar) and candle:
            stamp = pd.Timestamp(df.index[-1]).isoformat()
            key = f'{code}:{stamp}:{side}:entry'
            if key not in state:
                post(f'🚨 エントリー候補\n{code} {item["会社名"]}（{side}）\n確定足：{stamp}\n確認：{"、".join(triggers)}')
                state[key] = now_tokyo().isoformat()
    save_json(ROOT / "data/alert_state.json", state)


if __name__ == "__main__": run()

