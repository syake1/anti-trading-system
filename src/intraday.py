from __future__ import annotations

import pandas as pd
import yfinance as yf
from src.discord_notify import post
from src.stochastic import stochastic
from src.utils import ROOT, load_config, load_json, now_tokyo, save_json


BAR_MINUTES = 15


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


def entry_confirmed(*, reversal: bool, sar: bool, candle: bool) -> bool:
    """Notify only when momentum, Parabolic SAR and candle direction all agree."""
    return reversal and sar and candle


def finalized_bars(df: pd.DataFrame, now=None) -> pd.DataFrame:
    """Keep every completed 15-minute bar and exclude only a forming bar."""
    if df.empty:
        return df.copy()
    now = pd.Timestamp(now or now_tokyo())
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("Asia/Tokyo")
    elif now.tzinfo is None:
        now = now.tz_localize(index.tz)
    else:
        now = now.tz_convert(index.tz)
    completed = index + pd.Timedelta(minutes=BAR_MINUTES) <= now
    return df.loc[completed].copy()


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
        # 最後の足を無条件に削除すると、すでに確定済みの足まで捨てて15分遅れる。
        # 足の開始時刻+15分で判定し、形成中の足だけを除外する。
        df = finalized_bars(df)
        if len(df) < 25: continue
        df = df.join(stochastic(df, config["k_period"], config["d_period"]))
        df["PSAR"] = psar(df)
        a, b = df.iloc[-2], df.iloc[-1]
        side = item["買い・売り"]
        reversal = b.K > a.K if side == "買い" else b.K < a.K
        cross = (b.K > b.D and a.K <= a.D) if side == "買い" else (b.K < b.D and a.K >= a.D)
        sar = (b.Close > b.PSAR and a.Close <= a.PSAR) if side == "買い" else (b.Close < b.PSAR and a.Close >= a.PSAR)
        candle = b.Close > b.Open if side == "買い" else b.Close < b.Open
        volume = b.Volume > df.Volume.iloc[-21:-1].mean() * 1.3
        triggers = [name for name, ok in {"%K反転": reversal, "%K%Dクロス": cross, "SAR転換": sar, "出来高増加": volume, "ローソク転換": candle}.items() if ok]
        # %K/%Dクロスは補足表示にだけ使い、クロス単独では通知しない。
        if entry_confirmed(reversal=reversal, sar=sar, candle=candle):
            stamp = pd.Timestamp(df.index[-1]).isoformat()
            key = f'{code}:{stamp}:{side}:entry'
            if key not in state:
                post(f'🚨 エントリー候補\n{code} {item["会社名"]}（{side}）\n確定足：{stamp}\n確認：{"、".join(triggers)}')
                state[key] = now_tokyo().isoformat()
    save_json(ROOT / "data/alert_state.json", state)


if __name__ == "__main__": run()
