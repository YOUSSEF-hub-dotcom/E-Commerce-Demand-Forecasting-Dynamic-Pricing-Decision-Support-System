"""Streamlit manager UI for the deployment API.

Only the business-level contract is used: product ID, ISO week, optional cost.
Technical feature names and the 93-feature schema are never rendered; SHAP
contributions are shown with business labels.

Product selection is shared across Forecast / Pricing / Explainability via a
reusable, searchable selector backed by the authoritative product catalog (the
same weekly demand matrix the API consumes). Managers never type a Product ID.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import ApiClient, ApiError
from dashboard.config import PAGE_ICON, PAGE_TITLE
from dashboard.pricing_presenter import (
    render_pricing_opportunities,
    render_pricing_readiness,
    render_pricing_response,
)
from dashboard.product_catalog import (
    ProductSelector,
    week_selector,
)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

CLIENT = ApiClient()

st.title("E-Commerce Demand & Pricing Manager")
st.caption(
    "Production demand model `demand_forecasting_final` v2 - demand forecast, "
    "price recommendation and SHAP attribution."
)


def render_error(exc: ApiError) -> None:
    st.error(str(exc))


def render_pricing_section(client: ApiClient) -> None:
    """Business-oriented pricing flow: week first, then pricing-ready products.

    After the week is selected a dynamic count of price-available, elastic
    products is shown; the product selector leads with those products, and the
    selected product's real metadata is summarised before cost entry. No product
    is hidden - the full catalog stays reachable through the search box.
    """
    week = week_selector(key="pricing", label="Week (ISO Monday)")
    render_pricing_opportunities(week)
    selector = ProductSelector("pricing", pricing=True)
    product_id = selector.render(week=week)
    if product_id is not None:
        render_pricing_readiness(product_id, week)
        cost = st.number_input(
            "Unit cost (optional)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key="pricing:cost",
            help="When omitted, revenue-optimal is returned and profit is "
            "flagged PROFIT_NOT_COMPUTABLE.",
        )
    else:
        cost = None
    submitted = st.button("Get price recommendation", type="primary", key="pricing:run")
    if submitted and product_id is None:
        st.info("Search and select a product to continue.")
    elif submitted and product_id:
        try:
            res = client.pricing(
                product_id, week or None, cost if cost and cost > 0 else None
            )
            render_pricing_response(res)
        except ApiError as exc:
            render_error(exc)


def run_controls(
    *, selector: ProductSelector, allow_cost: bool, button_label: str
):
    """Shared run controls: searchable product + week (+ optional cost) + run.

    Returns ``(product_id, week, cost, submitted)``. ``submit`` is True only
    when the Run button was pressed with a product selected.
    """

    c1, c2 = st.columns(2)
    with c2:
        entered_week = week_selector(key=selector.key)
    with c1:
        product_id = selector.render(week=entered_week)
    cost = None
    if allow_cost:
        cost = c1.number_input(
            "Unit cost (optional)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"{selector.key}:cost",
            help="When omitted, revenue-optimal is returned and profit is "
            "flagged PROFIT_NOT_COMPUTABLE.",
        )
    submitted = st.button(button_label, type="primary", key=f"{selector.key}:run")
    if product_id is None:
        if submitted:
            st.info("Search and select a product to continue.")
        return None, entered_week, (cost if cost and cost > 0 else None), False
    effective_cost = cost if allow_cost and cost and cost > 0 else None
    return product_id, entered_week, effective_cost, submitted


st.divider()
section = st.sidebar.radio(
    "Section",
    ["Forecast", "Pricing", "Explainability", "System status"],
)

if section == "Forecast":
    st.subheader("Demand forecast")
    selector = ProductSelector("forecast")
    pid, week, _, submitted = run_controls(
        selector=selector, allow_cost=False, button_label="Run forecast"
    )
    if submitted and pid:
        try:
            res = CLIENT.predict(pid, week or None)
            st.success(f"Predicted demand: **{res['predicted_demand']:.4f} units**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Product", res["product_id"])
            col2.metric("Week", res["week"])
            col3.metric("Predicted demand", f"{res['predicted_demand']:.4f}")
            st.caption(
                f"Model v{res['model']['version']} · "
                f"inference_logged={res['inference']['inference_logged']}"
            )
        except ApiError as exc:
            render_error(exc)

elif section == "Pricing":
    st.subheader("Price recommendation")
    render_pricing_section(CLIENT)

elif section == "Explainability":
    st.subheader("SHAP attribution")
    st.caption(
        "Attribution, not causality. Contributions are in the model's log1p domain; "
        "the served unit prediction is `expm1(base_log1p + sum(shap))`."
    )
    selector = ProductSelector("explain")
    pid, week, _, submitted = run_controls(
        selector=selector, allow_cost=False, button_label="Run attribution"
    )
    if submitted and pid:
        try:
            res = CLIENT.explain(pid, week or None)
            st.success(f"Predicted demand: **{res['predicted_demand_units']:.4f} units**")
            st.info(res["reconstruction_block"])
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Top positive contributors")
                if res["top_positive"]:
                    st.dataframe(
                        pd.DataFrame(res["top_positive"]), use_container_width=True
                    )
                else:
                    st.write("None")
            with col2:
                st.markdown("#### Top negative contributors")
                if res["top_negative"]:
                    st.dataframe(
                        pd.DataFrame(res["top_negative"]), use_container_width=True
                    )
                else:
                    st.write("None")
            st.caption(
                f"Model v{res['model']['version']} · "
                f"inference_logged={res['inference']['inference_logged']}"
            )
        except ApiError as exc:
            render_error(exc)

else:
    st.subheader("System status")
    if st.button("Refresh", key="refresh_status"):
        try:
            ready = CLIENT.ready()
            health = CLIENT.health()
            if ready["status"] == "ready":
                st.success("Status: **ready**")
            else:
                st.warning(f"Status: **{ready['status']}**")
            st.markdown(f"API health: `{health['status']}` · env `{health['app_env']}`")
            st.table(pd.DataFrame(ready.get("components", {})).T)
        except Exception as exc:
            st.error(str(exc))