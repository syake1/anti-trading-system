# アンチ売買システム

日本株の日足から「大きな流れに逆行した短期の押し／戻りが、元の方向へ反転した」局面を抽出し、15分足でエントリーを絞る独立した Python プロジェクトです。これは売買助言ではなく、検証用ツールです。空売り可否、手数料、スリッページは利用者が確認してください。

## アンチと %K / %D

Linda Bradford Raschke / Laurence A. Connors のアンチ・パターンの考え方を参考に、ストキャスティクスの **7期間 %K** を短期トレンド、%K の **10期間平均 %D** を遅い方向として扱います（いずれも `config.json` で変更可能）。買いは %D 上昇中に %K が押してから上向く局面、売りはその逆です。クロスは強いシグナルですが必須ではありません。

## 日足条件とスコア

基本条件は「%D の3日前比の方向」「%K の直前の押し／戻り」「最新足での %K 反転」です。その上で %K/%D クロス、MA25/75/200、RSI14、20日ボリンジャーバンド、ATR14、ローソク足、出来高を加点します。買い・売りは対称に評価します。既定配点は `config.json` の `weights` が唯一の設定元です。

ランクは S=12点以上、A=9～11、B=7～8、C=6以下。既定の Discord 通知と15分足 watchlist は S/A のみです。流動性は株価100円以上かつ直前20日平均出来高10万株以上です。損切りは直近5日安値/高値と1.5 ATRを比較して値幅を確保し、利確は2Rです。

検出する足型は、色反転、包み足、はらみ足、長いヒゲ、前日高安値を抜けて戻る形、買い側の続落後陽線です。足型だけで候補にはしません。

## セットアップと銘柄一覧

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`stocks.csv` は `code,name,market` 形式です。リポジトリには動作確認用の少数例だけを同梱しています。本番では、JPX等から正当に入手した最新一覧で置換し、プライム・スタンダード・グロースを収録してください。コードに `.T` は不要です（取得時に自動付与）。

## 実行

```bash
# 日足抽出（Webhookへ送らない試運転）
python -m src.scanner --no-notify
# 日足抽出・S/A通知
DISCORD_WEBHOOK='https://discord.com/api/webhooks/...' python -m src.scanner
# 過去候補の5営業日追跡と集計
python -m src.backtest
# 15分足監視（東京市場時間外は何もしない）
DISCORD_WEBHOOK='...' python -m src.intraday
```

日足結果は `anti_candidates_YYYYMMDD.csv`、候補履歴は `data/signal_history.csv`、追跡結果は `data/performance.csv` です。出力には指定された価格・指標・足型・売買方向・ストップ・目標・RR・理由・Yahoo Financeリンクを含みます。

## 自動検証

`src.backtest` は各候補の翌日～5日後終値、期間最高/最安、最大上昇/下落率、利確/損切り到達、方向調整済み5日損益率を再計算します。`summary()` は件数、勝率、平均利益/損失、PF、複利系列の最大ドローダウン、ランク別・RSI帯別勝率を返します。CSVを pandas でグループ化すれば、出来高帯、BB位置、足型別も分析できます。日足ワークフローでは既存候補を先に追跡し、その後に当日候補を作ります。

## 15分足と重複防止

日足S/Aだけを `data/watchlist.json` に保存します。15分足は %K反転、%K/%Dクロス、パラボリックSAR転換、20本比の出来高増、ローソク色を確認し、反転＋（クロスまたはSAR）＋ローソク一致で通知します。形成中の最後の足を除外して **最新確定足だけ** を評価します。`data/alert_state.json` の「コード・確定足時刻・方向・シグナル」キーで同一通知を一度に限定するため、過去足を遡って通知しません。

## Discord / GitHub Actions

GitHub リポジトリの **Settings → Secrets and variables → Actions** に次を登録します。

* `DISCORD_WEBHOOK`（必須：通知先Webhook URL）

`.github/workflows/anti_daily_scan.yml` は平日09:00 UTC（18:00 JST）、`anti_intraday_alert.yml` は平日00:00～06:59 UTCに5分間隔で起動します。Pythonが `Asia/Tokyo` で曜日、前場 09:00–11:30、後場 12:30–15:30を再確認します。Actions遅延はあり得ます。結果と通知状態をコミットするため workflow の `contents: write` を使用します。ブランチ保護時は、永続化方法をartifactや外部ストレージへ変更してください。

## Version 1 / Version 2 構造

Version 1 の中核（ストキャス、方向・押し戻り・反転、RSI、MA200、出来高、足型、採点、CSV、Discord、成績追跡）を実装済みです。また拡張可能な独立モジュールとして、Version 2項目の BB、ATR、15分足、PSARも先行実装しています。地合い判定と業種相対強度は未実装で、将来 `anti_signal.py` / `scoring.py` に特徴量と設定配点を加えられます。

## ファイル案内

* `src/indicators.py`, `stochastic.py`, `candlestick.py` — 指標と足型
* `src/anti_signal.py`, `scoring.py`, `scanner.py` — 判定、採点、全銘柄処理
* `src/discord_notify.py`, `intraday.py` — 通知と15分足
* `src/backtest.py` — 自動追跡・集計
* `config.json`, `stocks.csv` — 変更可能な設定とユニバース

Yahoo Financeの欠損・レート制限・上場廃止等は銘柄単位でエラー表示して継続します。無料データの時刻・補正仕様を理解し、実運用前に十分なフォワード検証を行ってください。
