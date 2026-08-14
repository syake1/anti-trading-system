from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TOKYO = ZoneInfo("Asia/Tokyo")


def load_config(path: str | Path = ROOT / "config.json") -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_json(path: str | Path, default):
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return default
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def now_tokyo() -> datetime:
    return datetime.now(TOKYO)

