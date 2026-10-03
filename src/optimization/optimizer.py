"""Part-4 profit/revenue optimizer (Phase 4.4).

Selects ``P* = argmax Profit(P)`` over the simulation grid from Phase 4.3, mirroring
the Blueprint objective ``Profit = (Price - Cost) * Predicted Demand``.

Behavior when cost is absent (see Phase-4.0 feasibility): profit is **not** computable,
so the optimizer returns the revenue-optimal scenario and records
``PROFIT_NOT_COMPUTABLE (no cost)`` instead of inventing a margin.

Every optimization flags whether the optimum sits at the tested-range boundary
(``±20%``) — those cases need extrapolation/interpolation and are marked separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

OBJECTIVE_PROFIT = "profit"
OBJECTIVE_REVENUE = "revenue"


@dataclass(frozen=True)
class PriceRecommendation:
    product_id: str
    current_price: float
    optimal_price: float
    optimal_multiplier: float
    objective: str
    extrapolation_required: bool
    reason: str
    profit_computable: bool
    baseline_demand: float
    optimal_demand: float
    optimal_revenue: float
    optimal_profit: float | None = None


def _objective_series(sim: pd.DataFrame, cost: float | None) -> tuple[pd.Series, str, bool]:
    if cost is not None:
        return sim["profit"], OBJECTIVE_PROFIT, True
    return sim["revenue"], OBJECTIVE_REVENUE, False


def choose_optimal(sim: pd.DataFrame, *, cost: float | None = None) -> PriceRecommendation:
    """Choose the grid optimum for one product's scenario table."""
    if sim.empty:
        raise ValueError("optimizer: empty scenario table")
    series, objective, profit_ok = _objective_series(sim, cost)
    best_idx = int(np.argmax(series.to_numpy()))
    best = sim.iloc[best_idx]
    is_boundary = best_idx in (0, len(sim) - 1)
    reason = (
        "PROFIT_NOT_COMPUTABLE (no cost): revenue-optimal scenario returned as a "
        "proxy; supply a cost source to enable profit optimization"
        if not profit_ok
        else ""
    )
    return PriceRecommendation(
        product_id=str(best["product_id"]),
        current_price=float(sim.iloc[0]["price"] / sim.iloc[0]["multiplier"]),
        optimal_price=float(best["price"]),
        optimal_multiplier=float(best["multiplier"]),
        objective=objective,
        extrapolation_required=bool(is_boundary),
        reason=reason,
        profit_computable=profit_ok,
        baseline_demand=float(sim.iloc[(sim["multiplier"] == 1.0).to_numpy().argmax()]["pred_demand"]),
        optimal_demand=float(best["pred_demand"]),
        optimal_revenue=float(best["revenue"]),
        optimal_profit=float(best["profit"]) if profit_ok and np.isfinite(best["profit"]) else None,
    )


def optimal_price_from_panel(
    sim: pd.DataFrame,
    *,
    cost_by_product: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Optimize every product in a simulated panel; returns a recommendation frame."""
    cost_by_product = cost_by_product or {}
    records = []
    for pid, grp in sim.groupby("product_id", sort=False):
        rec = choose_optimal(grp, cost=cost_by_product.get(str(pid)))
        records.append(rec.__dict__)
    if not records:
        return pd.DataFrame(columns=list(PriceRecommendation.__dataclass_fields__))
    return pd.DataFrame(records)


# Constant-elasticity analytic interior solution (informational, cross-check only).
def analytic_optimal_price(elasticity: float, cost: float) -> float | None:
    """``P* = b*cost/(b+1)`` for ``b < -1``; ``None`` otherwise (no finite optimum).

    Derivation: maximize ``(P - C) * a * P**b`` over ``P > 0``.
    Informational cross-check; the discrete grid argmax is the operational optimizer.
    """
    if cost is None or cost <= 0 or not (elasticity < -1):
        return None
    return float(elasticity * cost / (elasticity + 1.0))