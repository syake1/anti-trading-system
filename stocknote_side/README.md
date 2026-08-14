# stocknote Phase 2 手動 CLI

このディレクトリは stocknote 側だけで動く、1リクエスト単位の最小アダプターです。
常駐監視、HTTP API、自動スケジュールは含みません。

```bash
python -m stocknote_side.cli /exchange/stocknote_request_run_123456.json
```

既存の分析関数が別モジュールにある場合は `module:function` 形式で指定します。

```bash
python -m stocknote_side.cli REQUEST.json --analyzer my_stocknote.analysis:analyze_candidate
```

同梱の最小実分析プロバイダー（東証の1銘柄だけを取得）は次のように指定します。

```bash
python -m stocknote_side.cli REQUEST.json \
  --analyzer stocknote_provider.analysis:analyze_candidate
```

株価履歴からRSI、ボリンジャーバンド位置、トレンド、逆張りスコアと3つの価格水準を
計算します。ファンダメンタルはリクエストで明示された利用可能な値だけを返し、欠損値を
補完しません。株探・みんかぶ等の値を渡す場合は `reference_information` として公式値と
分離してください。

分析関数にはキーワード引数 `code`、`official_information`、
`reference_information` が渡ります。戻り値は必須4項目のうち `code` を除く
`assessment`、`confidence`、`summary` と、response schema の任意項目を持つ辞書です。
既存の1引数関数には、互換用に銘柄コードだけを渡します。

戻り値で情報源を区別する場合は `official` と `reference` を別辞書にします。
`kabutan` / `minkabu`（または `株探` / `みんかぶ`）も reference として扱われ、
同じ項目の公式値を上書きしません。

```python
{
    "assessment": "positive",
    "confidence": 0.8,
    "summary": "反転を確認",
    "official": {"per": 12.3},
    "kabutan": {"per": 12.8},
}
```

データがない任意項目は省略してください。既定の統合関数は外部データを推測せず、
`assessment=insufficient` を返します。既存responseがある場合は終了し、明示的な
再生成だけ `--force` を使います。出力は同じディレクトリの
`stocknote_response_<run_id>.json` へ一時ファイルから原子的に公開されます。
