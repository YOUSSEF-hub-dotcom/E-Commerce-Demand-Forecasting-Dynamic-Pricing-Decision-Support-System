"""Part-4 optimizer tests: P* selection, boundary flags, profit gating."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.optimizer import (
    analytic_optimal_price,
    choose_optimal,
    optimal_price_from_panel,
)
from src.optimization.simulation import simulate_scenarios


def _sim(q0=100.0, p0=10.0, b=-1.5, cost=None):
    return simulate_scenarios(product_id="p1", q0=q0, p0=p0, elasticity=b, cost=cost)


def test_choose_optimal_revenue_when_no_cost():
    rec = choose_optimal(_sim(cost=None))
    assert rec.profit_computable is False
    assert rec.objective == "revenue"
    assert rec.optimal_profit is None
    assert "PROFIT_NOT_COMPUTABLE" in rec.reason


def test_choose_optimal_profit_when_cost_given():
    rec = choose_optimal(_sim(cost=5.0), cost=5.0)
    assert rec.profit_computable is True
    assert rec.objective == "profit"
    assert rec.optimal_profit is not None and rec.optimal_profit > 0
    # manual argmax over the grid
    sim = _sim(cost=5.0)
    assert rec.optimal_price == pytest.approx(
        sim.loc[sim["profit"].idxmax(), "price"]
    )


def test_choose_optimal_boundary_flag():
    # a strongly elastic product: optimum will sit at the lowest tested price (boundary)
    rec = choose_optimal(_sim(q0=100.0, p0=10.0, b=-2.5, cost=8.0))
    assert rec.extrapolation_required is True
    assert rec.optimal_multiplier <= 0.85


def test_optimal_price_from_panel_aggregates():
    sim = pd.concat(
        [
            simulate_scenarios(product_id="p1", q0=100.0, p0=10.0, elasticity=-1.5, cost=5.0),
            simulate_scenarios(product_id="p2", q0=50.0, p0=20.0, elasticity=-0.5, cost=5.0),
        ],
        ignore_index=True,
    )
    recs = optimal_price_from_panel(sim, cost_by_product={"p1": 5.0, "p2": 5.0})
    assert set(recs["product_id"]) == {"p1", "p2"}
    assert (recs["objective"] == "profit").all()


def test_analytic_optimal_price():
    assert abs(analytic_optimal_price(-2.0, 10.0) - (-2.0 * 10.0 / -1.0)) < 1e-9
    assert analytic_optimal_price(-0.5, 10.0) is None   # inelastic -> no finite optimum
    assert analytic_optimal_price(-2.0, 0.0) is None    # no valid cost