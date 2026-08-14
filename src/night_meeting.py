"""Phase 1 provisional night meeting (bank sector only).

The night meeting narrows the next morning's review set.  It deliberately has
no order planning or position sizing code: all executable decisions remain in
the existing morning meeting.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from src.jgb_yields import JGBAnalysis
from src.sector_strategy import BankSectorEvaluation, evaluate_bank_sector
from src.utils import ROOT, now_tokyo


NIGHT_CATEGORIES = ("押し目候補", "動意候補", "翌朝再確認", "避ける")
DETAIL_CLASSES = ("押し目候補", "押し目監視", "動意候補", "追いかけ禁止", "回避", "評価不能")
_CATEGORY = {
    "押し目候補": "押し目候補", "動意候補": "動意候補",
    "押し目監視": "翌朝再確認", "評価不能": "翌朝再確認",
    "追いかけ禁止": "避ける", "回避": "避ける",
}


def _next_morning_start(observed_at: datetime, config: Mapping[str, Any]) -> datetime:
    value = config.get("night_meeting", {}).get("morning_start_jst", "05:45")
    hour, minute = (int(part) for part in value.split(":"))
    candidate = datetime.combine(observed_at.date() + timedelta(days=1), time(hour, minute),
                                 tzinfo=ZoneInfo("Asia/Tokyo"))
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def generate_night_result(candidates: pd.DataFrame, jgb: JGBAnalysis | Mapping[str, Any] | None,
                          config: Mapping[str, Any], observed_at: datetime | None = None) -> dict[str, Any]:
    """Evaluate bank rows and return an auditable, non-final night result."""
    observed_at = observed_at or now_tokyo()
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    rows = []
    if not candidates.empty:
        for bank in candidates.fillna("").to_dict("records"):
            if str(bank.get("業種", "")).strip() != "銀行":
                continue
            evaluation = evaluate_bank_sector(jgb, bank, config)
            rows.append(_candidate_result(bank, evaluation))
    return {
        "status": "provisional",
        "final_decision": False,
        "observed_at": observed_at.isoformat(timespec="seconds"),
        "valid_until": _next_morning_start(observed_at, config).isoformat(timespec="minutes"),
        "scope": {"implemented": ["銀行"],
                  "design_only": ["半導体", "建設", "資源", "電力", "防衛", "不動産", "商社",
                                  "自社株買い・IR・監理整理・上場廃止等の本格自動取得"]},
        "categories": {name: [row for row in rows if row["night_category"] == name]
                       for name in NIGHT_CATEGORIES},
        "bank_evaluations": rows,
    }


def _candidate_result(bank: Mapping[str, Any], evaluation: BankSectorEvaluation) -> dict[str, Any]:
    return {"code": str(bank.get("コード", "")), "name": str(bank.get("会社名", bank.get("銘柄名", ""))),
            "night_category": _CATEGORY[evaluation.classification],
            "bank_classification": evaluation.classification,
            "morning_recheck": evaluation.morning_recheck,
            "rate_regime": evaluation.rate_regime, "rate_wind": evaluation.rate_wind,
            "rate_confidence": evaluation.rate_confidence, "rate_metrics": evaluation.rate_metrics,
            "principle": evaluation.principle, "reasons": list(evaluation.reasons),
            "missing_data": list(evaluation.missing_data)}


def night_message(result: Mapping[str, Any]) -> str:
    """Render every required bank list and rate metric without buy sizing labels."""
    evaluations = result["bank_evaluations"]
    first = evaluations[0] if evaluations else None
    lines = ["🌙 AI投資会議・夜会（銀行セクター Phase 1）",
             f"status={result['status']}", f"final_decision={str(result['final_decision']).lower()}",
             f"valid_until={result['valid_until']}", "※主力・小口および注文判断は翌朝会だけで決定します。", ""]
    if first:
        metrics = first["rate_metrics"]
        lines += [f"金利局面：{first['rate_regime']}（{first['rate_wind']} / 確信度 {first['rate_confidence']}）"]
        for tenor in ("2Y", "10Y", "30Y"):
            level = metrics.get(f"{tenor}_yield_pct")
            one = metrics.get(f"{tenor}_change_1d_bp")
            five = metrics.get(f"{tenor}_change_5d_bp")
            lines.append(f"日本{tenor.replace('Y', '年')}金利：{_fmt(level, '%')} / 前日変化 {_fmt(one, 'bp')} / 5日変化 {_fmt(five, 'bp')}")
        lines += [f"10年-2年スプレッド：{_fmt(metrics.get('spread_10y_2y_bp'), 'bp')}",
                  f"基本戦略：{first['principle']}", ""]
    else:
        lines += ["金利局面：評価不能（銀行候補なし）", "基本戦略：金利上昇基調でも高値追いせず、反転確認後の押し目を拾う", ""]
    for classification in DETAIL_CLASSES:
        label = "翌朝再確認銘柄" if classification == "評価不能" else classification
        selected = [f"{row['code']} {row['name']}".strip() for row in evaluations
                    if row["bank_classification"] == classification]
        lines.append(f"{label}：" + (" / ".join(selected) if selected else "なし"))
    explicit_recheck = [f"{row['code']} {row['name']}".strip() for row in evaluations if row["morning_recheck"]]
    lines.append("翌朝再確認銘柄：" + (" / ".join(explicit_recheck) if explicit_recheck else "なし"))
    lines += ["", "夜会候補分類（押し目候補 / 動意候補 / 翌朝再確認 / 避ける）"]
    for category in NIGHT_CATEGORIES:
        selected = [f"{row['code']} {row['name']}".strip() for row in result["categories"][category]]
        lines.append(f"{category}：" + (" / ".join(selected) if selected else "なし"))
    return "\n".join(lines)


def _fmt(value: Any, suffix: str) -> str:
    return "欠損" if value is None else f"{float(value):+.3f}{suffix}"


def save_night_result(result: Mapping[str, Any], folder: Path | None = None) -> tuple[Path, Path]:
    folder = folder or ROOT / "reports/meeting/night"
    folder.mkdir(parents=True, exist_ok=True)
    day = str(result["observed_at"])[:10].replace("-", "")
    json_path, markdown_path = folder / f"night_meeting_{day}.json", folder / f"night_meeting_{day}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(night_message(result) + "\n", encoding="utf-8")
    return json_path, markdown_path


def load_latest_night_result(folder: Path | None = None) -> dict[str, Any] | None:
    """Load reference conditions only; callers must not feed this into order evaluation."""
    files = sorted((folder or ROOT / "reports/meeting/night").glob("night_meeting_*.json"), reverse=True)
    if not files:
        return None
    try:
        result = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return result if result.get("status") == "provisional" and result.get("final_decision") is False else None
