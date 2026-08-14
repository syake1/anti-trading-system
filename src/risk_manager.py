"""Deterministic position sizing for the configured portfolio."""
from __future__ import annotations

import math


def position_size(entry: float, stop: float, classification: str, config: dict,
                  *, crash_rebound: bool = False) -> dict:
    """Return an order which satisfies risk, cash, lot and position limits."""
    p = config["portfolio"]
    capital = float(p["initial_capital"])
    cash = float(p.get("current_cash", capital))
    risk_per_share = max(0.0, float(entry) - float(stop))
    max_loss = capital * float(p["max_risk_per_trade_pct"]) / 100
    reserved_cash = capital * float(p["minimum_cash_ratio"]) / 100
    available = max(0.0, cash - reserved_cash)
    position_room = int(p.get("current_positions", 0)) < int(p["max_positions"])
    pct = float(p["core_size_pct"] if classification == "主力候補" else p["small_size_pct"])
    if crash_rebound:
        pct = min(pct, float(p["crash_rebound_size_pct"]))
    budget = min(float(p["max_position_amount"]), available) * pct / 100
    lot = max(1, int(p.get("lot_size", 100)))
    # Small/crash candidates scale both the capital budget and risk budget, so a
    # cheap volatile stock cannot accidentally receive the full core size.
    risk_shares = math.floor((max_loss * pct / 100) / risk_per_share) if risk_per_share else 0
    cash_shares = math.floor(budget / entry) if entry > 0 else 0
    shares = min(risk_shares, cash_shares)
    shares = math.floor(shares / lot) * lot if position_room else 0
    return {
        "買値候補": round(entry, 2), "損切り": round(stop, 2),
        "1株リスク": round(risk_per_share, 2), "最大許容損失": round(max_loss),
        "推奨株数": shares, "必要資金": round(shares * entry),
        "現金比率（購入後）": round((cash - shares * entry) / capital * 100, 1),
    }
