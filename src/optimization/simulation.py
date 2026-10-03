"""Part-4 price-scenario simulation engine (Phase 4.3).

Given a baseline demand ``Q0`` at a current price ``P0`` and a price elasticity
``b`` (constant-elasticity demand response), simulate demand and revenue over the
blueprint scenario grid ``P ± {5,10,15,20}%`` plus the current price.

Architecture separation (Part-4 contract):

* the Part-3 Production forecast supplies ``Q0`` (the demand baseline at current price);
* this module applies only the price demand-response ``Q(P) = Q0 * (P/P0) ** b``;
* profit is computed **only** when a real cost value is supplied (never invented).

No module here reads ``data/splits/test.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

# Blueprint scenario grid: P ± {20, 15, 10, 5}% around the current price (incl. P).
SCENARIO_MULTIPLIERS: tuple[float, ...] = (
    0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
)

BASELINE_SCENARIO_LABEL = "current"


def demand_response(q0: float, p0: float, b: float, price: float) -> float:
    """Constant-elasticity demand: ``Q(P) = Q0 * (P/P0) ** b``."""
    if not (p0 > 0):
        raise ValueError("simulation: baseline price p0 must be > 0")
    if price <= 0:
        raise ValueError("simulation: price must be > 0")
    return float(q0 * (price / p0) ** b)


def simulate_scenarios(
    *,
    product_id: str,
    q0: float,
    p0: float,
    elasticity: float,
    multipliers: tuple[float, ...] = SCENARIO_MULTIPLIERS,
    cost: float | None = None,
) -> pd.DataFrame:
    """Simulate demand / revenue (+ profit when ``cost`` is given) for one product."""
    rows = []
    for m in multipliers:
        price = p0 * m
        demand = demand_response(q0, p0, elasticity, price)
        revenue = price * demand
        profit = (price - cost) * demand if cost is not None else np.nan
        rows.append(
            {
                "product_id": product_id,
                "multiplier": m,
                "price": price,
                "pred_demand": demand,
                "revenue": revenue,
                "profit": profit,
            }
        )
    return pd.DataFrame(rows)


def simulate_panel(
    baselines: pd.DataFrame,
    elasticity_by_product: dict[str, float],
    *,
    multipliers: tuple[float, ...] = SCENARIO_MULTIPLIERS,
    cost_by_product: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate the scenario grid across a baseline frame.

    ``baselines`` needs ``product_id``, ``baseline_demand`` and ``current_price``.
    Products with no elasticity are skipped (they get no re-price recommendation).
    """
    cost_by_product = cost_by_product or {}
    frames = []
    for row in baselines.itertuples(index=False):
        b = elasticity_by_product.get(str(row.product_id))
        if b is None or not np.isfinite(b):
            continue
        frames.append(
            simulate_scenarios(
                product_id=str(row.product_id),
                q0=float(row.baseline_demand),
                p0=float(row.current_price),
                elasticity=float(b),
                multipliers=multipliers,
                cost=cost_by_product.get(str(row.product_id)),
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["product_id", "multiplier", "price", "pred_demand", "revenue", "profit"]
        )
    return pd.concat(frames, ignore_index=True)