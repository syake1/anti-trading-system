import io

import pandas as pd

from src import jpx_margin


def workbook(date, rows):
    output = io.BytesIO()
    frame = pd.DataFrame([[f"基準日：{date}", None, None, None], ["銘柄コード", "銘柄名", "売残高", "買残高"], *rows])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, header=False)
    return output.getvalue()


class Response:
    def __init__(self, text="", content=b""):
        self.text, self.content = text, content

    def raise_for_status(self):
        pass


def test_fetch_4751_matches_published_workbooks(monkeypatch):
    latest = workbook("2026年8月7日", [[4751, "サイバーエージェント", 100, 350]])
    previous = workbook("2026年7月31日", [[4751, "サイバーエージェント", 80, 300]])
    responses = {
        jpx_margin.DEFAULT_PAGE_URL: Response('<a href="/latest.xlsx">最新</a><a href="/previous.xlsx">前週</a>'),
        "https://www.jpx.co.jp/latest.xlsx": Response(content=latest),
        "https://www.jpx.co.jp/previous.xlsx": Response(content=previous),
    }
    monkeypatch.setattr(jpx_margin.requests.Session, "get", lambda self, url, timeout: responses[url])
    result, error = jpx_margin.fetch_weekly_margin()
    row = result.set_index("コード").loc["4751"]
    assert error is None
    assert (row["売残"], row["買残"]) == (100, 350)
    assert (row["売残前週比"], row["買残前週比"]) == (20, 50)
    assert row["信用倍率"] == 3.5
    assert row["基準日"] == "2026-08-07"
    assert row["取得元URL"] == "https://www.jpx.co.jp/latest.xlsx"
    assert row["取得日時"]


def test_zero_sell_balance_is_not_infinity(monkeypatch):
    latest = workbook("2026/8/7", [[4751, "サイバーエージェント", 0, 350]])
    previous = workbook("2026/7/31", [[4751, "サイバーエージェント", 0, 300]])
    monkeypatch.setattr(jpx_margin, "_links", lambda html, url: ["latest.xlsx", "previous.xlsx"])
    payloads = iter([Response(text="page"), Response(content=latest), Response(content=previous)])
    monkeypatch.setattr(jpx_margin.requests.Session, "get", lambda self, url, timeout: next(payloads))
    result, _ = jpx_margin.fetch_weekly_margin()
    assert result.iloc[0]["信用倍率"] == "算出不能・売残0"


def test_failure_is_data_none_and_continues(monkeypatch, tmp_path):
    monkeypatch.setattr(jpx_margin, "fetch_weekly_margin", lambda *args: (_ for _ in ()).throw(ValueError("schema")))
    result = jpx_margin.enrich_candidates(pd.DataFrame([{"コード": "4751"}]), {}, tmp_path)
    assert (result[jpx_margin.DATA_COLUMNS] == "データなし").all().all()
    assert (tmp_path / "data/jpx_margin_balances.csv").exists()


def test_missing_code_is_data_none(monkeypatch):
    margin = pd.DataFrame([{"コード": "7203", **{column: "value" for column in jpx_margin.DATA_COLUMNS}}])
    monkeypatch.setattr(jpx_margin, "fetch_weekly_margin", lambda *args: (margin, None))
    result = jpx_margin.enrich_candidates(pd.DataFrame([{"コード": "4751"}]))
    assert (result[jpx_margin.DATA_COLUMNS] == "データなし").all().all()


def test_multilevel_header_uses_total_not_account_subtotals():
    output = io.BytesIO()
    frame = pd.DataFrame([
        ["基準日 2026年8月7日", None, "制度信用", None, "一般信用", None, "合計", None],
        ["銘柄コード", "銘柄名", "売残高", "買残高", "売残高", "買残高", "売残高", "買残高"],
        [4751, "サイバーエージェント", 10, 20, 30, 40, 40, 60],
    ])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, header=False)
    _, result = jpx_margin.parse_workbook(output.getvalue(), "https://www.jpx.co.jp/test.xlsx")
    assert result.iloc[0][["売残", "買残"]].tolist() == [40, 60]


def test_4751_high_ratio_and_rising_buy_balance_lowers_adjusted_rank():
    candidates = pd.DataFrame([
        {
            "コード": "4751", "スコア": 20, "判定理由": "テクニカル反転", "信用倍率": 30.83,
            "買残": 3083, "買残前週比": 600, "直近3日騰落率": -2.5, "直近5日騰落率": -4,
            "ローソク足パターン": "なし", "出来高倍率": 1.1,
        },
        {
            "コード": "7203", "スコア": 18, "判定理由": "テクニカル反転", "信用倍率": 4.0,
            "買残": 1000, "買残前週比": 0, "直近3日騰落率": 1, "直近5日騰落率": 2,
        },
    ])

    result = jpx_margin.apply_margin_scoring(candidates).set_index("コード")
    cyberagent = result.loc["4751"]

    assert cyberagent["テクニカルスコア"] == 20
    assert cyberagent["信用需給による減点"] == -20
    assert cyberagent["信用需給スコア"] == -20
    assert cyberagent["総合調整後スコア"] == 0
    assert cyberagent["テクニカル順位"] == 1
    assert cyberagent["調整後順位"] == 2
    assert cyberagent["信用需給判定"] == "見送り"
    assert "原則反転確認待ち" in cyberagent["判定理由"]


def test_missing_or_zero_sell_margin_data_is_unscored_and_not_favorable():
    candidates = pd.DataFrame([
        {"コード": "1111", "スコア": 10, "信用倍率": "データなし"},
        {"コード": "2222", "スコア": 9, "信用倍率": "算出不能・売残0"},
    ])

    result = jpx_margin.apply_margin_scoring(candidates)

    assert result["信用需給による減点"].tolist() == [0, 0]
    assert result["総合調整後スコア"].tolist() == [10, 9]
    assert result["信用需給判定"].tolist() == ["信用需給は判定不能"] * 2
