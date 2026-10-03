"""Dynamic pricing service.

Mirrors the verified Part-4 engine for a single product/week: baseline demand at
the current (observed) price from the Production forecast, a constant-elasticity
demand response over the blueprint scenario grid, profit/revenue optimum, and the
business guardrails. Cost is optional; without it profit is not computed and the
revenue-optimal scenario is returned flagged ``PROFIT_NOT_COMPUTABLE``.
"""

from __future__ import annotations

import math

import pandas as pd

from src.optimization.guardrails import GuardrailConfig, apply_guardrails
from src.optimization.optimizer import choose_optimal
from src.optimization.simulation import simulate_scenarios

from app.errors import NO_ELASTICITY, NO_PRICE_OBSERVATION, AppError
from app.services.prediction_service import model_summary, predict_row
from app.services.state import AppState

REASON_NO_PRICE = "NO_PRICE_OBSERVATION"
REASON_NO_COST = "PROFIT_NOT_COMPUTABLE (no cost supplied)"
DEFAULT_GUARDRAILS = GuardrailConfig()


def pricing_row(
    state: AppState,
    product_id: str,
    week_label: str,
    cost: float | None,
) -> dict:
    pid = state.validate_product(product_id)
    prediction = predict_row(state, pid, week_label)
    baseline = float(prediction["predicted_demand"])
    category = state.category_of.get(pid)
    current_price = state.price_at(week_label, pid)

    base = {
        "product_id": pid,
        "week": week_label,
        "baseline_demand": baseline,
        "category": category,
        "current_price": current_price,
        "model": model_summary(state),
    }

    if current_price is None or not math.isfinite(current_price):
        return {
            **base,
            "elasticity_status": None,
            "elasticity_source": None,
            "elasticity": None,
            "reason": NO_PRICE_OBSERVATION,
            "profit_computable": False,
            "scenarios": [],
        }

    elasticity = state.elasticity(pid)
    status = state.elasticity_status_of(pid)
    source = state.elasticity_source_of(pid)
    usable = (
        elasticity is not None
        and math.isfinite(elasticity)
        and source not in (None, "none")
    )
    if not usable:
        return {
            **base,
            "elasticity_status": status,
            "elasticity_source": source,
            "elasticity": elasticity,
            "reason": NO_ELASTICITY,
            "profit_computable": False,
            "scenarios": [],
        }

    sim = simulate_scenarios(
        product_id=pid, q0=baseline, p0=current_price, elasticity=float(elasticity), cost=cost
    )
    rec = choose_optimal(sim, cost=cost)
    out = apply_guardrails(
        float(rec.optimal_price), float(current_price), cost=cost, cfg=DEFAULT_GUARDRAILS
    )
    profit_ok = cost is not None and rec.profit_computable

    rec_dict = {
        "product_id": pid,
        "week": week_label,
        "baseline_demand": baseline,
        "category": category,
        "current_price": current_price,
        "elasticity_status": status,
        "elasticity_source": source,
        "elasticity": float(elasticity),
        "reason": "optimized" if profit_ok else REASON_NO_COST,
        "profit_computable": profit_ok,
        "objective": rec.objective,
        "optimal_price": float(rec.optimal_price),
        "optimal_multiplier": float(rec.optimal_multiplier),
        "optimal_revenue": float(rec.optimal_revenue),
        "optimal_profit": rec.optimal_profit,
        "recommended_price": float(out.recommended_price),
        "extrapolation_required": bool(rec.extrapolation_required),
        "guardrails": {
            "margin_floor_applied": bool(out.margin_floor_applied),
            "margin_floor_inactive_no_cost": bool(out.margin_floor_inactive_no_cost),
            "ceiling_applied": bool(out.ceiling_applied),
            "floor_applied": bool(out.floor_applied),
            "smoothed": bool(out.smoothed),
            "notes": list(out.notes),
        },
        "scenarios": [asdict_scenario(r) for r in sim.to_dict(orient="records")],
        "model": model_summary(state),
    }
    return rec_dict


def asdict_scenario(row: dict) -> dict:
    return {
        "multiplier": float(row["multiplier"]),
        "price": float(row["price"]),
        "pred_demand": float(row["pred_demand"]),
        "revenue": float(row["revenue"]),
        "profit": None if row.get("profit") is None or pd.isna(row.get("profit")) else float(row["profit"]),
    }