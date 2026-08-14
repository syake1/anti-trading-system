"""Translate market indicators into sector-specific scores."""
from __future__ import annotations
import pandas as pd
from src.utils import ROOT


def sector_impacts(environment, config: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    mappings = config["sector_impacts"]
    for indicator in environment.indicators:
        for sector, multiplier in mappings.get(indicator["indicator"], {}).items():
            totals[sector] = totals.get(sector, 0) + indicator["score"] * float(multiplier)
    return totals


def save_sector_impacts(impacts: dict, observed_at: str, path=None):
    path = path or ROOT / "data/sector_impact.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": observed_at[:10], "time": observed_at[11:19], "sector": k, "impact_score": v}
                  for k, v in sorted(impacts.items())], columns=["date", "time", "sector", "impact_score"]).to_csv(path, index=False)
