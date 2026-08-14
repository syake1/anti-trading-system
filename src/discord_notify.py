from __future__ import annotations

import os
import requests


def post(content: str) -> bool:
    webhook = os.getenv("DISCORD_WEBHOOK")
    if not webhook:
        return False
    response = requests.post(webhook, json={"content": content[:2000]}, timeout=20)
    response.raise_for_status()
    return True


def candidate_message(row: dict) -> str:
    icon = "🔥" if row["買い・売り"] == "買い" else "📉"
    return (f'{icon} アンチ{row["買い・売り"]} {row["ランク"]}ランク\n\n'
            f'銘柄：{row["コード"]} {row["会社名"]}\n株価：{row["現在値"]:,.1f}円\n'
            f'スコア：{row["スコア"]}\n%K：{row["%K"]:.1f} / %D：{row["%D"]:.1f}\n'
            f'RSI：{row["RSI14"]:.1f}　出来高：{row["出来高倍率"]:.2f}倍　BB：{row["BB位置"]}\n'
            f'ローソク足：{row["ローソク足パターン"]}\n損切り：{row["損切り候補"]:,.1f}円\n'
            f'利確目標：{row["利確候補"]:,.1f}円　RR：{row["RR"]:.1f}\n理由：{row["判定理由"]}')

