import json
import numpy as np
import pandas as pd
from src.anti_signal import detect
from src.indicators import enrich
from src.stochastic import stochastic


def test_indicators_and_anti_buy():
    n = 240
    close = np.linspace(100, 200, n) + np.sin(np.arange(n)/3) * 3
    frame = pd.DataFrame({"Open": close-.2, "High": close+2, "Low": close-2, "Close": close, "Volume": 200_000})
    config = json.load(open("config.json", encoding="utf-8"))
    result = enrich(frame, config).join(stochastic(frame, 7, 10))
    assert result.iloc[-1].MA200 > 0
    # 判定器を制御値で確認する。
    result.loc[result.index[-4:], "D"] = [30, 32, 34, 36]
    result.loc[result.index[-3:], "K"] = [40, 25, 38]
    signal = detect(result, 3)
    assert signal and signal["side"] == "buy" and signal["cross"]
