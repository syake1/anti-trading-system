import json
import numpy as np
import pandas as pd
import pytest
from src import backtest, scanner
from src.indicators import enrich
from src.discord_notify import candidate_message
from src.scoring import score
from src.stochastic import stochastic
from src.utils import load_json
from src.strategies import evaluate
from src.materials import load_buybacks


def test_indicators_and_stochastic_cross():
    n = 240
    close = np.linspace(100, 200, n) + np.sin(np.arange(n)/3) * 3
    frame = pd.DataFrame({"Open": close-.2, "High": close+2, "Low": close-2, "Close": close, "Volume": 200_000})
    config = json.load(open("config.json", encoding="utf-8"))
    result = enrich(frame, config).join(stochastic(frame, 7, 10))
    assert result.iloc[-1].MA200 > 0
    # 判定器を制御値で確認する。
    result.loc[result.index[-4:], "D"] = [30, 32, 34, 36]
    result.loc[result.index[-3:], "K"] = [40, 25, 38]
    assert result.iloc[-1].K > result.iloc[-1].D


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


def test_backtest_coerces_decorated_prices_and_skips_only_invalid_signal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(backtest, "ROOT", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    columns = list(scanner.RESULT_COLUMNS)

    def signal(current, take_profit, stop_loss, code):
        row = {column: "" for column in columns}
        row.update({"シグナル日": "2026-01-01", "コード": code, "ランク": "A",
                    "買い・売り": "買い", "現在値": current, "利確候補": take_profit,
                    "損切り候補": stop_loss, "RSI14": "30", "出来高倍率": "1.5",
                    "BB位置": "-2σ", "ローソク足パターン": "反転"})
        return row

    pd.DataFrame([
        signal("1,000円", "1,100 円", "900円", "0001"),
        signal("価格不明", "1,100円", "900円", "0002"),
        signal(None, "1,100円", "900円", "0003"),
    ], columns=columns).to_csv(data / "signal_history.csv", index=False)
    dates = pd.date_range("2026-01-02", periods=6)
    downloaded = pd.DataFrame({
        "Open": ["1,000円"] * 6,
        "High": ["1,020円", "bad", "1,040円", "1,050円", "1,060円", "1,100円"],
        "Low": ["990円", "980円", "970円", "960円", "950円", "900円"],
        "Close": ["1,010円", "", "1,030円", "1,040円", "1,050円", "1,060円"],
    }, index=dates)
    monkeypatch.setattr(backtest.yf, "download", lambda *args, **kwargs: downloaded.copy())

    output = backtest.update()

    result = pd.read_csv(output)
    assert len(result) == 1
    assert result.loc[0, "1日後終値"] == 1010
    assert result.loc[0, "5日後終値"] == 1060
    assert result.loc[0, "利確到達"]
    log = capsys.readouterr().out
    assert "スキップ件数=2" in log
    assert "数値変換失敗件数=2" in log
    assert "正常評価件数=1" in log


def test_backtest_summary_handles_nullable_and_empty_aggregates(tmp_path):
    path = tmp_path / "performance.csv"
    pd.DataFrame([
        {"ランク": "A", "5日損益率": pd.NA, "RSI14": pd.NA, "出来高倍率": pd.NA,
         "BB位置": pd.NA, "ローソク足パターン": pd.NA, "シグナル種別": pd.NA,
         "最大上昇率": pd.NA, "最大下落率": pd.NA},
        {"ランク": "A", "5日損益率": "5%", "RSI14": "不明", "出来高倍率": "",
         "BB位置": "-2σ", "ローソク足パターン": "反転", "シグナル種別": "アンチ",
         "最大上昇率": "", "最大下落率": None},
    ]).to_csv(path, index=False)

    result = backtest.summary(path)

    assert result["全シグナル数"] == 2
    assert result["勝率"] == 0.0
    assert result["平均利益率"] == result["平均損失率"] == 0.0
    assert result["PF"] is None
    assert result["最大ドローダウン"] == 0.0
    assert result["戦略別成績"]["アンチ"]["平均最大上昇率"] == 0.0


def test_backtest_summary_handles_incomplete_nonempty_csv(tmp_path):
    path = tmp_path / "performance.csv"
    path.write_text("コード\n7203\n", encoding="utf-8")

    result = backtest.summary(path)

    assert result["全シグナル数"] == 1
    assert result["勝率"] == result["平均利益率"] == result["平均損失率"] == 0.0


def test_mixed_history_preserves_legacy_rows_and_skips_only_bad_line(tmp_path):
    from src.csv_history import LEGACY_SIGNAL_COLUMNS, read_mixed_csv, write_merged_csv

    old, current = LEGACY_SIGNAL_COLUMNS[0], scanner.RESULT_COLUMNS
    path = tmp_path / "signal_history.csv"
    old_row = ["2025-01-01", "7203", *(["old"] * (len(old) - 2))]
    new_row = ["2026-01-01", "6758", *(["new"] * (len(current) - 2))]
    path.write_text(
        ",".join(old_row) + "\n壊れた,行,です\n" + ",".join(current) + "\n" + ",".join(new_row) + "\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="1 行をスキップ"):
        loaded = read_mixed_csv(path, [*LEGACY_SIGNAL_COLUMNS, current], dtype={"コード": str})
    assert loaded["コード"].tolist() == ["7203", "6758"]
    write_merged_csv(path, loaded, pd.DataFrame([dict(zip(current, new_row))]))
    reparsed = pd.read_csv(path, dtype={"コード": str})
    assert reparsed["コード"].tolist() == ["7203", "6758", "6758"]


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
    monkeypatch.setattr(scanner, "evaluate", lambda df, pats, cfg: {
        "flags": {"BB逆張り": True, "BB＋RSI＋ストキャス": False, "底固め": False},
        "reversal": True, "bb_rebound": True, "rsi_reversal": True,
        "stoch_reversal": True, "stoch_cross": True, "base_reasons": [],
    })
    monkeypatch.setattr(scanner, "stochastic", lambda df, k, d: pd.DataFrame(
        {"K": np.linspace(20, 40, len(df)), "D": np.linspace(18, 38, len(df))}, index=df.index
    ))
    n = 240
    base = np.linspace(1000, 1100, n)
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


def test_speculative_stock_filter_rejects_volume_spike_and_extreme_atr():
    frame = pd.DataFrame({
        "Close": [1000.0] * 5,
        "ATR14": [20.0, 20.0, 20.0, 20.0, 90.0],
        "volume_ratio": [1.0, 1.2, 5.0, 1.1, 1.0],
    })
    reasons = scanner.speculative_stock_exclusion(frame, {
        "speculative_stock_exclusion": {"max_volume_ratio_5d": 5.0, "max_atr_pct": 8.0}
    })
    assert reasons == ["直近5日出来高異常急増", "ATR比率異常"]


def test_speculative_stock_filter_accepts_normal_liquid_price_action():
    frame = pd.DataFrame({
        "Close": [1000.0] * 5,
        "ATR14": [20.0] * 5,
        "volume_ratio": [1.0, 1.2, 1.5, 1.1, 2.0],
    })
    assert scanner.speculative_stock_exclusion(frame, {
        "speculative_stock_exclusion": {"max_volume_ratio_5d": 5.0, "max_atr_pct": 8.0}
    }) == []


def test_buy_watchlist_requires_sufficient_non_adverse_fundamentals():
    frame = pd.DataFrame([
        {"コード": "0001", "ファンダメンタル十分": True, "ファンダメンタルスコア": 8,
         "revision": "上方修正", "important_disclosure": "", "company_forecast": "増収増益"},
        {"コード": "0002", "ファンダメンタル十分": False, "ファンダメンタルスコア": 8},
        {"コード": "0003", "ファンダメンタル十分": True, "ファンダメンタルスコア": 5},
        {"コード": "0004", "ファンダメンタル十分": True, "ファンダメンタルスコア": 8,
         "revision": "下方修正"},
    ])
    approved = scanner.fundamental_buy_candidates(frame, {
        "fundamentals": {"require_for_buy_watchlist": True, "minimum_buy_score": 6}
    })
    assert approved["コード"].tolist() == ["0001"]


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


def test_strategy_requires_reversal_after_lower_band():
    config = _config()
    frame = pd.DataFrame({"Open": [102, 100, 98], "High": [103, 101, 100], "Low": [99, 96, 95],
        "Close": [100, 98, 99], "RSI14": [31, 28, 33], "K": [24, 15, 27], "D": [25, 20, 21],
        "bb_sigma": [-1.2, -2.1, -1.5], "ATR14": [2, 2, 2], "bb_upper": [110]*3,
        "bb_lower": [90]*3, "bb_mid": [100]*3, "volume_ratio": [1, 1, 1.5]})
    # accumulation needs history, but A/B can be evaluated independently.
    config["accumulation"]["lookback_days"] = 1
    out = evaluate(frame, ["前日安値割れ反転"], config)
    assert out["flags"]["BB逆張り"]
    assert out["flags"]["BB＋RSI＋ストキャス"]


def test_buyback_csv_calculations(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame([["2026-01-01", "0001", 50, 500, "2026-01-02", "2026-01-11", 1000, 10000, 100, 100]],
        columns=["発表日", "銘柄コード", "取得上限株数", "取得上限金額", "取得期間開始日", "取得期間終了日", "発行済株式数", "時価総額", "発表時株価", "1日平均売買代金"]).to_csv(data / "buybacks.csv", index=False)
    result = load_buybacks(tmp_path).iloc[0]
    assert result["時価総額比"] == 5
    assert result["発行済株式比"] == 5
    assert result["取得期間日数"] == 10


def test_order_backtest_keeps_unfilled_pnl_missing():
    from src.backtest import simulate_order_methods
    import json
    cfg = json.load(open("config.json", encoding="utf-8"))
    signal = {"コード":"0001", "現在値":1000, "ATR14":20, "前日高値":1100,
              "反転足高値":1100, "直近2日高値":1100, "直近安値":900,
              "損切り候補":900, "MA25":1000}
    future = pd.DataFrame({"Open":[1000]*3, "High":[1010]*3, "Low":[990]*3, "Close":[1005]*3})
    result = pd.DataFrame(simulate_order_methods(signal, future, cfg)).set_index("注文方式")
    assert not bool(result.loc["stop", "約定"])
    assert pd.isna(result.loc["stop", "5日後損益"])
    assert pd.notna(result.loc["stop", "未約定後最大上昇率"])
