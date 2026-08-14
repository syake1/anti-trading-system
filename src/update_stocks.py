"""JPXの公開一覧から東証の国内普通株を更新する。"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from src.utils import ROOT


JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
MARKETS = {
    "プライム（内国株式）": "プライム",
    "スタンダード（内国株式）": "スタンダード",
    "グロース（内国株式）": "グロース",
}


def fetch(url: str = JPX_LIST_URL, timeout: int = 30) -> pd.DataFrame:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "anti-trading-system/1.0 (+https://github.com/)"},
    )
    response.raise_for_status()
    source = pd.read_excel(BytesIO(response.content), dtype=str)
    source.columns = [str(column).strip() for column in source.columns]
    required = {"コード", "銘柄名", "市場・商品区分", "33業種区分"}
    if not required.issubset(source.columns):
        raise ValueError(f"JPX一覧の列が想定外です: {list(source.columns)}")
    stocks = source[source["市場・商品区分"].isin(MARKETS)].copy()
    stocks["code"] = stocks["コード"].str.strip()
    stocks = stocks[stocks["code"].str.fullmatch(r"\d{4}", na=False)]
    stocks["name"] = stocks["銘柄名"].str.strip()
    stocks["market"] = stocks["市場・商品区分"].map(MARKETS)
    # JPX is the existing security master.  Keep its official industry alongside
    # the listing fields so downstream meetings do not have to infer a sector.
    stocks["industry"] = stocks["33業種区分"].fillna("").str.strip()
    stocks = stocks[["code", "name", "market", "industry"]].drop_duplicates("code").sort_values("code")
    if len(stocks) < 1000:
        raise ValueError(f"取得銘柄数が不自然です: {len(stocks)}")
    return stocks.reset_index(drop=True)


def update(output: Path | None = None) -> Path:
    output = output or ROOT / "stocks.csv"
    try:
        stocks = fetch()
        temporary = output.with_suffix(".csv.tmp")
        stocks.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(output)
        print(f"JPX銘柄一覧を更新しました: {len(stocks)}銘柄 ({output})")
    except Exception as exc:
        # 成功するまで一時ファイルだけを使い、利用中の一覧は絶対に壊さない。
        print(f"銘柄一覧の更新に失敗しました。既存ファイルを維持します: {exc}")
        if not output.exists():
            raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    update(args.output)
