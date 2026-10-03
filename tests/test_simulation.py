"""Part-4 simulation tests: constant-elasticity demand response, revenue, profit gating."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.simulation import (
    demand_response,
    simulate_panel,
    simulate_scenarios,
)


def test_demand_response_math():
    # Q0=100 at P0=10, b=-1: P=11 -> 100 * (1.1)**-1
    assert abs(demand_response(100.0, 10.0, -1.0, 11.0) - 100.0 / 1.1) < 1e-9


def test_demand_response_requires_positive_prices():
    with pytest.raises(ValueError, match="p0 must be > 0"):
        demand_response(10.0, 0.0, -1.0, 5.0)
    with pytest.raises(ValueError, match="price must be > 0"):
        demand_response(10.0, 5.0, -1.0, 0.0)


def test_simulate_scenarios_grid_and_revenue():
    sim = simulate_scenarios(product_id="p1", q0=100.0, p0=10.0, elasticity=-1.5)
    assert len(sim) == 9
    assert list(sim["multiplier"]) == [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    assert sim["multiplier"].is_monotonic_increasing
    base = sim[sim["multiplier"] == 1.0].iloc[0]
    assert abs(base["pred_demand"] - 100.0) < 1e-9
    assert abs(base["revenue"] - 1000.0) < 1e-9
    assert np.isnan(base["profit"])  # no cost -> profit not computable


def test_simulate_scenarios_with_cost_profit():
    sim = simulate_scenarios(product_id="p1", q0=100.0, p0=10.0, elasticity=-1.5, cost=5.0)
    row = sim[sim["multiplier"] == 1.0].iloc[0]
    assert abs(row["profit"] - (10.0 - 5.0) * 100.0) < 1e-9


def test_simulate_panel_skips_without_elasticity():
    baselines = pd.DataFrame(
        {"product_id": ["p1", "p2", "p3"], "baseline_demand": [100.0, 50.0, 20.0], "current_price": [10.0, 8.0, 6.0]}
    )
    elast = {"p1": -1.5, "p3": -0.8}
    sim = simulate_panel(baselines, elast)
    assert set(sim["product_id"]) == {"p1", "p3"}  # p2 has no elasticity
    assert len(sim) == 2 * 9


def test_simulate_panel_empty_when_no_elasticity():
    baselines = pd.DataFrame(
        {"product_id": ["p1"], "baseline_demand": [10.0], "current_price": [5.0]}
    )
    sim = simulate_panel(baselines, {})
    assert sim.empty