import json
import numpy as np
import pandas as pd
from src import backtest, scanner
from src.anti_signal import detect
from src.indicators import enrich
from src.discord_notify import candidate_message
from src.scoring import score
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


def test_backtest_accepts_missing_empty_or_incomplete_history(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "ROOT", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    history = data / "signal_history.csv"

    for content in (
        None,
        "",
        "シグナル日,コード,買い・売り\n",
        "コード,買い・売り\n7203,買い\n",
        "シグナル日,買い・売り\n2026-08-14,買い\n",
        "シグナル日,コード\n2026-08-14,7203\n",
    ):
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


def test_batch_download_retries_only_missing_tickers_with_exponential_backoff(monkeypatch):
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["7203.T", "6758.T"]])
    first = pd.DataFrame([[100, np.nan] * len(fields)], columns=columns)
    second = pd.DataFrame({field: [200] for field in fields})
    downloads = iter([first, second])
    calls = []
    sleeps = []
    monkeypatch.setattr(scanner, "_download_batch", lambda tickers, config: (calls.append(tickers.copy()), next(downloads))[1])
    monkeypatch.setattr(scanner.time, "sleep", sleeps.append)
    config = {"scan": {"download_max_attempts": 4, "retry_backoff_seconds": 3}}

    frames = scanner._download_with_retry(["7203.T", "6758.T"], config)

    assert calls == [["7203.T", "6758.T"], ["6758.T"]]
    assert sleeps == [3]
    assert all(not frame.empty for frame in frames.values())


def test_batch_download_continues_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(scanner, "_download_batch", lambda tickers, config: (_ for _ in ()).throw(RuntimeError("429")))
    sleeps = []
    monkeypatch.setattr(scanner.time, "sleep", sleeps.append)

    frames = scanner._download_with_retry(
        ["7203.T", "6758.T"],
        {"scan": {"download_max_attempts": 3, "retry_backoff_seconds": 2}},
    )

    assert all(frame.empty for frame in frames.values())
    assert sleeps == [2, 4]


def test_batch_download_counts_yfinance_internally_caught_429(monkeypatch):
    """yf.download reports per-ticker 429s via shared errors instead of raising."""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    downloads = iter([pd.DataFrame(), pd.DataFrame({field: [100] for field in fields})])

    def fake_download(tickers, config):
        frame = next(downloads)
        scanner._LAST_DOWNLOAD_ERRORS = (
            {"7203.T": "YFRateLimitError('Too Many Requests. Rate limited. 429')"}
            if frame.empty else {}
        )
        return frame

    monkeypatch.setattr(scanner, "_download_batch", fake_download)
    monkeypatch.setattr(scanner.time, "sleep", lambda _: None)
    limited = set()

    frames = scanner._download_with_retry(
        ["7203.T"],
        {"scan": {"download_max_attempts": 2, "retry_backoff_seconds": 0}},
        limited,
    )

    assert not frames["7203.T"].empty
    assert limited == {"7203.T"}


def test_thrown_429_is_exhausted_without_escaping(monkeypatch):
    monkeypatch.setattr(
        scanner, "_download_batch",
        lambda tickers, config: (_ for _ in ()).throw(RuntimeError("429 Too Many Requests")),
    )
    monkeypatch.setattr(scanner.time, "sleep", lambda _: None)
    limited = set()

    frames = scanner._download_with_retry(
        ["7203.T", "6758.T"],
        {"scan": {"download_max_attempts": 2, "retry_backoff_seconds": 0}},
        limited,
    )

    assert all(frame.empty for frame in frames.values())
    assert limited == {"7203.T", "6758.T"}


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


def _config():
    return json.load(open("config.json", encoding="utf-8"))


def test_surge_metrics_and_configurable_exclusion_for_3660_and_8995(monkeypatch):
    config = _config()
    monkeypatch.setattr(scanner, "detect", lambda df, slope: {"side": "buy", "cross": True, "d_trend": True})
    monkeypatch.setattr(scanner, "stochastic", lambda df, k, d: pd.DataFrame(
        {"K": np.linspace(20, 40, len(df)), "D": np.linspace(18, 38, len(df))}, index=df.index
    ))
    n = 240
    base = np.linspace(100, 110, n)
    for code, final_gain in (("3660", 9), ("8995", 13)):
        close = base.copy()
        close[-1] = close[-2] * (1 + final_gain / 100)
        frame = pd.DataFrame({
            "Open": close * .995, "High": close * 1.02, "Low": close * .98,
            "Close": close, "Volume": 1_500_000,
        })
        row = enrich(frame, config).iloc[-1]
        reasons = scanner.surge_exclusion(row, config)
        candidate = scanner.analyze(frame, {"code": code, "name": code, "market": "東証"}, config)
        assert row.change_1d >= 8, code
        assert "当日急騰" in reasons, code
        assert candidate["ランク"] == "除外", code
        assert candidate["除外理由"].startswith("急騰済み"), code


def test_countertrend_reversal_scores_above_s_rank_threshold():
    config = _config()
    # 数日下落 → 売られ過ぎ → 下限反発、陽の包み、下ヒゲ、出来高増を再現。
    frame = pd.DataFrame({
        "Open": [104, 103, 102, 101, 98, 94],
        "High": [105, 104, 103, 102, 100, 101],
        "Low": [102, 101, 100, 96, 93, 89],
        "Close": [103, 102, 101, 98, 94, 100],
        "Volume": [100, 100, 100, 100, 100, 180],
        "K": [35, 30, 25, 20, 15, 28], "D": [34, 31, 27, 23, 20, 21],
        "RSI14": [42, 39, 35, 31, 28, 33],
        "bb_sigma": [-.5, -1, -1.4, -1.8, -2.2, -1.5],
        "volume_ratio": [1, 1, 1, 1, 1, 1.8],
    })
    pats = ["陰線→陽線", "強気包み足", "長い下ヒゲ（ハンマー）", "2～5日下落後の陽線"]
    value, reasons = score(frame, {"side": "buy", "cross": True}, pats, config)
    assert value >= config["rank_thresholds"]["S"]
    assert "直近2～5日まで下落" in reasons
    assert "ストキャス売られ過ぎから反転" in reasons


def test_one_day_rise_without_recent_decline_cannot_be_sa_rank():
    config = _config()
    frame = pd.DataFrame({
        "Open": [99, 100, 101, 102, 103, 104], "Close": [100, 101, 102, 103, 104, 108],
        "K": [10, 12, 14, 16, 18, 30], "D": [10, 11, 12, 13, 14, 20],
        "RSI14": [30, 31, 32, 33, 34, 38], "bb_sigma": [-2.2, -2, -1.8, -1.5, -1.2, -.5],
        "volume_ratio": [1, 1, 1, 1, 1, 2],
    })
    value, reasons = score(frame, {"side": "buy", "cross": True}, ["強気包み足"], config)
    assert value < config["rank_thresholds"]["A"]
    assert "直近下落なし（B以下に制限）" in reasons


def test_discord_buy_message_shows_not_already_surged_fields():
    row = {"買い・売り": "買い", "ランク": "S", "コード": "0001", "会社名": "反転株",
           "現在値": 100, "スコア": 15, "%K": 25, "%D": 20, "直近3日騰落率": -4.2,
           "直近5日騰落率": -7.1, "25日線乖離率": -5.5, "RSI14": 34,
           "出来高倍率": 1.6, "BB位置": "-1.80σ", "ローソク足パターン": "強気包み足",
           "損切り候補": 92, "利確候補": 116, "RR": 2, "判定理由": "反転"}
    message = candidate_message(row)
    for label in ("3日騰落率", "5日騰落率", "25日線乖離", "RSI", "BB位置", "反転パターン"):
        assert label in message
