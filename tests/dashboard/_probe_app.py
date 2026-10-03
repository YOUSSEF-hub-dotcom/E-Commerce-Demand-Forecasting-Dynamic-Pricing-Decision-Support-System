"""AppTest fixture that mirrors dashboard/app.py section wiring.

Each section renders the shared ``ProductSelector`` + ``week_selector`` and, on
Run, records the exact (product, week[, cost]) that the section would send to
the API. Used by the dashboard selector integration tests to prove the
Forecast / Pricing / Explainability sections all consume shared selector state
and forward the selected real product unchanged.

The Pricing section mirrors the dedicated business flow in ``app.py``: it emits
a machine-readable ``PRICING_OPPORTUNITIES:<tier1>:<tier2>:<tier3>`` markdown
line (so tests can assert the dynamic opportunities count) and, when a product
is selected, the real ``PRICING:<product>:<week>:<cost>`` forwarding line. The
opportunities summary itself is the app's own
``render_pricing_opportunities`` renderer, so the probe cannot drift from it.

Setting ``probe_missing_readiness`` in the session state repoints the catalog at
an absent Part-4 artifact and clears its caches, which exercises the real
unavailable-data path end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

import dashboard.product_catalog as product_catalog
from dashboard.pricing_presenter import (
    render_pricing_opportunities,
    render_pricing_readiness,
)
from dashboard.product_catalog import (
    ProductSelector,
    week_selector,
)

st.set_page_config(page_title="probe", layout="wide")

if st.session_state.get("probe_missing_readiness"):
    product_catalog.PRICING_READY_REPORT = (
        ROOT / "reports" / "forecasting" / "__absent_for_probe__.json"
    )
    product_catalog._readiness_report.cache_clear()
    product_catalog._elasticity_details.cache_clear()
    product_catalog.load_pricing_ready_sources.clear()
    product_catalog.pricing_options.clear()

section = st.sidebar.radio(
    "Section",
    ["Forecast", "Pricing", "Explainability"],
    key="section",
)


def run_controls(selector: ProductSelector, allow_cost: bool) -> tuple:
    c1, c2 = st.columns(2)
    with c2:
        week = week_selector(key=selector.key)
    with c1:
        product_id = selector.render(week=week)
    cost = None
    if allow_cost:
        cost = c1.number_input(
            "Unit cost (optional)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"{selector.key}:cost",
        )
    submitted = st.button("Run", key=f"{selector.key}:run")
    return product_id, week, (cost if cost and cost > 0 else None), submitted


def render_probe_pricing_section() -> None:
    week = week_selector(key="pricing")
    tier1, tier2, tier3 = render_pricing_opportunities(week)
    st.markdown(f"PRICING_OPPORTUNITIES:{tier1}:{tier2}:{tier3}")
    selector = ProductSelector("pricing", pricing=True)
    product_id = selector.render(week=week)
    cost = None
    if product_id is not None:
        render_pricing_readiness(product_id, week)
        cost = st.number_input(
            "Unit cost (optional)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="pricing:cost",
        )
    submitted = st.button("Run", key="pricing:run")
    if submitted and product_id:
        st.write(f"PRICING:{product_id}:{week}:{cost if cost and cost > 0 else None}")


if section == "Forecast":
    selector = ProductSelector("forecast")
    pid, week, _, submitted = run_controls(selector, allow_cost=False)
    if submitted and pid:
        st.write(f"FORECAST:{pid}:{week}:n/a")
elif section == "Pricing":
    render_probe_pricing_section()
else:
    selector = ProductSelector("explain")
    pid, week, _, submitted = run_controls(selector, allow_cost=False)
    if submitted and pid:
        st.write(f"EXPLAIN:{pid}:{week}:n/a")