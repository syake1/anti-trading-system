# アンチ売買システム

日本株の日足から「大きな流れに逆行した短期の押し／戻りが、元の方向へ反転した」局面を抽出し、15分足でエントリーを絞る独立した Python プロジェクトです。これは売買助言ではなく、検証用ツールです。空売り可否、手数料、スリッページは利用者が確認してください。

## アンチと %K / %D

Linda Bradford Raschke / Laurence A. Connors のアンチ・パターンの考え方を参考に、ストキャスティクスの **7期間 %K** を短期トレンド、%K の **10期間平均 %D** を遅い方向として扱います（いずれも `config.json` で変更可能）。買いは %D 上昇中に %K が押してから上向く局面、売りはその逆です。クロスは強いシグナルですが必須ではありません。

## 日足条件とスコア

基本条件は「%D の3日前比の方向」「%K の直前の押し／戻り」「最新足での %K 反転」です。買い側はさらに、直近2～5日の下落、RSI40以下・ストキャス売られ過ぎ・BB下限からの反転、陽の包み足、下ヒゲ、出来高増を中心に加点します。既定配点は `config.json` の `weights` が唯一の設定元です。

すでに急騰した買いシグナルは、当日/3日/5日騰落率、25日線乖離、RSI14、BB位置、ATR比の当日値幅で判定します。既定では当日+8%、3日+12%、5日+18%、25日線乖離+10%、RSI 72、BB +2.2σ、またはストップ高に近い値動きのいずれかで通常S/Aから強制除外します。全閾値は `config.json` の `surge_exclusion` で変更できます。除外行も分析用CSVに `除外理由=急騰済み（…）` として残しますが、Discordと15分足watchlistには入りません。

ランクは S=12点以上、A=9～11、B=7～8、C=6以下。既定の Discord 通知と15分足 watchlist は S/A のみです。流動性は株価100円以上かつ直前20日平均出来高10万株以上です。損切りは直近5日安値/高値と1.5 ATRを比較して値幅を確保し、利確は2Rです。

検出する足型は、色反転、包み足、はらみ足、長いヒゲ、前日高安値を抜けて戻る形、買い側の続落後陽線です。足型だけで候補にはしません。

## セットアップと銘柄一覧

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`stocks.csv` は `code,name,market` 形式です。次のコマンドは **日本取引所グループ（JPX）の「東証上場銘柄一覧」** を取得し、プライム・スタンダード・グロースの「内国株式」だけを原子的に保存します。ETF、ETN、REIT、インフラファンド、外国株等は市場・商品区分で除外します。取得失敗や異常に少ない一覧の場合は既存CSVを残します。コードは4桁で保存し、Yahoo Financeへの要求時だけ `.T` を付けます。

```bash
python -m src.update_stocks
```

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

日足はyfinanceへ `scan.batch_size`（既定100）銘柄ずつ並列一括要求し、欠損銘柄も一銘柄ずつではなく欠損分を一括で1回再試行します。バッチ間には既定0.5秒の短い待機を入れ、空データでも後続を継続します。既定の `scan.scan_limit` は `0`（`stocks.csv` の全銘柄）です。必要な場合だけ正数にすると先頭N銘柄に制限できます。最後に対象銘柄数、取得成功・失敗数、判定完了数、一次フィルター通過数、候補数、S/Aランク件数、処理時間を表示します。一次フィルターの既定値は株価100円、20日平均出来高10万株、20日平均売買代金1億円です。

日足結果は `anti_candidates_YYYYMMDD.csv`、候補履歴は `data/signal_history.csv`、追跡結果は `data/performance.csv` です。出力には価格、1/3/5日騰落率、25日線乖離、ATR比の当日値幅、指標、足型、売買方向、ストップ、目標、RR、理由、急騰除外理由、Yahoo Financeリンクを含みます。

## 自動検証

`src.backtest` は各候補の翌日～5日後終値、期間最高/最安、最大上昇/下落率、利確/損切り到達、方向調整済み5日損益率を再計算します。`summary()` は件数、勝率、平均利益/損失、PF、複利系列の最大ドローダウン、ランク別・RSI帯別勝率を返します。CSVを pandas でグループ化すれば、出来高帯、BB位置、足型別も分析できます。日足ワークフローでは既存候補を先に追跡し、その後に当日候補を作ります。

## 15分足と重複防止

日足S/Aだけをスコア順に最大50銘柄まで `data/watchlist.json` に保存します（`scan.watchlist_ranks` / `scan.watchlist_max_stocks` で変更可能）。15分足は全上場株ではなく、このリストだけを監視します。15分足は %K反転、%K/%Dクロス、パラボリックSAR転換、20本比の出来高増、ローソク色を確認し、反転＋（クロスまたはSAR）＋ローソク一致で通知します。形成中の最後の足を除外して **最新確定足だけ** を評価します。`data/alert_state.json` の「コード・確定足時刻・方向・シグナル」キーで同一通知を一度に限定するため、過去足を遡って通知しません。

## Discord / GitHub Actions

GitHub リポジトリの **Settings → Secrets and variables → Actions** に次を登録します。

* `DISCORD_WEBHOOK`（任意：未設定なら通知をスキップ）

`.github/workflows/update_stocks.yml` は毎週日曜12:00 UTC（JST 21:00）に一覧を更新し、変更時だけcommit/pushします。`anti_daily_scan.yml` は平日09:00 UTC（18:00 JST、30分timeout）に backtest → scanner の順、`anti_intraday_alert.yml` は平日00:00～06:59 UTCに5分間隔で起動します。各ワークフローはGitHubの **Actions** タブで選択し **Run workflow** から手動実行できます。日足Discord通知はS/Aをスコア順に `scan.discord_max_alerts`（既定20）件まで送ります。全銘柄スキャンは通常数分～30分程度を想定しますが、無料APIの応答・レート制限に依存します。結果をcommitするため `contents: write` を使用し、ブランチ保護時はartifact等への変更が必要です。

## Version 1 / Version 2 構造

Version 1 の中核（ストキャス、方向・押し戻り・反転、RSI、MA200、出来高、足型、採点、CSV、Discord、成績追跡）を実装済みです。また拡張可能な独立モジュールとして、Version 2項目の BB、ATR、15分足、PSARも先行実装しています。地合い判定と業種相対強度は未実装で、将来 `anti_signal.py` / `scoring.py` に特徴量と設定配点を加えられます。

## ファイル案内

* `src/indicators.py`, `stochastic.py`, `candlestick.py` — 指標と足型
* `src/anti_signal.py`, `scoring.py`, `scanner.py` — 判定、採点、全銘柄処理
* `src/discord_notify.py`, `intraday.py` — 通知と15分足
* `src/backtest.py` — 自動追跡・集計
* `config.json`, `stocks.csv` — 変更可能な設定とユニバース

Yahoo Financeの欠損・レート制限・上場廃止等は銘柄単位でエラー表示して継続します。無料データの時刻・補正仕様を理解し、実運用前に十分なフォワード検証を行ってください。
