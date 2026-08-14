"""Japanese government bond yield analysis based exclusively on observed closes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.utils import ROOT, now_tokyo


TENORS = ("2Y", "10Y", "30Y")
HORIZONS = {"1d": 1, "5d": 5, "20d": 20}


@dataclass(frozen=True)
class JGBAnalysis:
    observed_at: str
    tenors: tuple[dict, ...]
    spread_10y_2y_bp: float | None

    def as_dict(self) -> dict:
        return {"observed_at": self.observed_at, "tenors": list(self.tenors),
                "spread_10y_2y_bp": self.spread_10y_2y_bp}


def analyze_jgb(history: dict[str, pd.Series] | None, observed_at: str | None = None) -> JGBAnalysis:
    """Calculate yield changes; insufficient observations remain ``None``."""
    observed_at = observed_at or now_tokyo().isoformat(timespec="seconds")
    rows, current = [], {}
    for tenor in TENORS:
        raw = (history or {}).get(tenor)
        series = pd.to_numeric(raw, errors="coerce").dropna() if raw is not None else pd.Series(dtype=float)
        value = float(series.iloc[-1]) if len(series) else None
        current[tenor] = value
        row = {"tenor": tenor, "yield_pct": value}
        for label, days in HORIZONS.items():
            row[f"change_{label}_bp"] = round((value - float(series.iloc[-days - 1])) * 100, 4) if value is not None and len(series) > days else None
        rows.append(row)
    spread = None if current["10Y"] is None or current["2Y"] is None else round((current["10Y"] - current["2Y"]) * 100, 4)
    return JGBAnalysis(observed_at, tuple(rows), spread)


def fetch_jgb_data(config: dict) -> dict[str, pd.Series]:
    """Fetch configured yield tickers. Failed/empty tenors are omitted, never filled."""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    result = {}
    for tenor, ticker in config.get("jgb_yields", {}).get("tickers", {}).items():
        try:
            frame = yf.download(ticker, period="3mo", interval="1d", progress=False,
                                auto_adjust=False, timeout=10)
            close = frame["Close"]
            if hasattr(close, "columns"): close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            if not close.empty:
                result[tenor] = close
        except Exception:
            continue
    return result


def jgb_sector_impacts(analysis: JGBAnalysis, config: dict) -> dict[str, float]:
    """Apply configured sensitivities to the observed 20-day 10Y move and curve."""
    tenors = {row["tenor"]: row for row in analysis.tenors}
    move = tenors.get("10Y", {}).get("change_20d_bp")
    if move is None:
        return {}
    scale = float(config.get("jgb_yields", {}).get("impact_bp_scale", 10))
    return {sector: round(move / scale * float(weight), 4)
            for sector, weight in config.get("jgb_yields", {}).get("sector_impacts", {}).items()}


def save_jgb_analysis(analysis: JGBAnalysis, path: Path | None = None) -> None:
    path = path or ROOT / "data/jgb_yields.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"date": analysis.observed_at[:10], "time": analysis.observed_at[11:19], **row,
             "spread_10y_2y_bp": analysis.spread_10y_2y_bp} for row in analysis.tenors]
    pd.DataFrame(rows).to_csv(path, index=False)


def format_jgb_message(analysis: JGBAnalysis) -> list[str]:
    def number(value, suffix=""):
        return "欠損" if value is None else f"{value:+.2f}{suffix}"
    lines = ["🇯🇵 日本国債金利"]
    for row in analysis.tenors:
        level = "欠損" if row["yield_pct"] is None else f'{row["yield_pct"]:.3f}%'
        lines.append(f'{row["tenor"]}: {level}（前日 {number(row["change_1d_bp"], "bp")} / 5日 {number(row["change_5d_bp"], "bp")} / 20日 {number(row["change_20d_bp"], "bp")}）')
    lines.append(f'10年-2年スプレッド: {number(analysis.spread_10y_2y_bp, "bp")}')
    return lines
