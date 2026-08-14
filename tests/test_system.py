import json
import numpy as np
import pandas as pd
from src import backtest, scanner
from src.anti_signal import detect
from src.indicators import enrich
from src.stochastic import stochastic
from src.utils import load_json


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


def test_backtest_accepts_missing_empty_and_header_only_history(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "ROOT", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    history = data / "signal_history.csv"

    for content in (None, "", "シグナル日,コード,買い・売り\n"):
        history.unlink(missing_ok=True)
        if content is not None:
            history.write_text(content, encoding="utf-8")
        output = backtest.update()
        assert output.exists()
        assert backtest.summary(output) == {"全シグナル数": 0}
        assert list(pd.read_csv(output).columns) == backtest.PERFORMANCE_COLUMNS


def test_first_run_continues_through_scanner(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "ROOT", tmp_path)
    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stocks.csv").write_text("code,name,market\n", encoding="utf-8")

    backtest.update()
    candidate_path = scanner.run(notify=False)

    assert candidate_path.exists()
    assert load_json(tmp_path / "data/watchlist.json", None) == []


def test_empty_or_missing_json_state_uses_default(tmp_path):
    path = tmp_path / "state.json"
    for content in (None, "", "  \n", "{"):
        path.unlink(missing_ok=True)
        if content is not None:
            path.write_text(content, encoding="utf-8")
        assert load_json(path, {"fresh": True}) == {"fresh": True}


def test_daily_workflow_runs_backtest_before_scanner():
    workflow = open(".github/workflows/anti_daily_scan.yml", encoding="utf-8").read()
    assert workflow.index("python -m src.backtest") < workflow.index("python -m src.scanner")


def test_update_stocks_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    from src import update_stocks
    output = tmp_path / "stocks.csv"
    output.write_text("code,name,market\n7203,Toyota,プライム\n", encoding="utf-8")
    monkeypatch.setattr(update_stocks, "fetch", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    update_stocks.update(output)
    assert "7203" in output.read_text(encoding="utf-8")


def test_batch_ticker_frame_accepts_yfinance_multiindex():
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["7203.T", "6758.T"]])
    downloaded = pd.DataFrame([[100, 200, 1_000, 2_000]], columns=columns)
    frame = scanner._ticker_frame(downloaded, "7203.T")
    assert list(frame.columns) == ["Close", "Volume"]
    assert frame.iloc[0].Close == 100


def test_daily_scan_defaults_to_entire_stock_list():
    config = json.load(open("config.json", encoding="utf-8"))
    assert config["scan"]["scan_limit"] == 0


def test_empty_scan_prints_action_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stocks.csv").write_text("code,name,market\n", encoding="utf-8")

    scanner.run(notify=False)

    output = capsys.readouterr().out
    for label in ("対象銘柄数", "取得成功数", "取得失敗数", "判定完了数",
                  "Sランク件数", "Aランク件数", "処理時間"):
        assert f"{label}:" in output
