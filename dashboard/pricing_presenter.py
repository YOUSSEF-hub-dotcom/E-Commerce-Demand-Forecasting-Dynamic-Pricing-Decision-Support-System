"""Business-facing presenter for the ``/pricing`` response (UI layer only).

Maps the API's technical ``reason`` to clear copy for a non-technical manager.
Pure presentation: it never computes, invents, or fabricates a price. The
recommended price is always the value returned by the API, and a
``NO_ELASTICITY`` response renders an explicit "recommendation unavailable"
block instead of a recommendation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.product_catalog import (
    READINESS_AVAILABLE,
    STATUS_NO_ELASTICITY,
    STATUS_NO_PRICE,
    STATUS_READINESS_UNAVAILABLE,
    STATUS_READY,
    elasticity_source_label,
    pricing_readiness_state,
    pricing_tier_counts,
    product_pricing_status,
)

REASON_NO_ELASTICITY = "NO_ELASTICITY"

NO_ELASTICITY_TITLE = "Price recommendation unavailable"
NO_ELASTICITY_EXPLANATION = (
    "This product does not have sufficient historical price-demand variation "
    "to estimate price elasticity reliably."
)
NO_ELASTICITY_PROFIT = (
    "Profit optimization was not performed because elasticity is unavailable."
)

NO_PRICE_TITLE = "Price recommendation unavailable"
NO_PRICE_EXPLANATION = (
    "This product does not have a valid current price for the selected week."
)
NO_PRICE_PROFIT = (
    "Profit optimization was not performed because no current price observation "
    "is available for the selected week."
)

READINESS_UNAVAILABLE_TITLE = "Pricing readiness data is unavailable"
READINESS_UNAVAILABLE_EXPLANATION = (
    "The persisted pricing readiness state could not be loaded, so products "
    "cannot be classified as ready or not ready for this week. Products stay "
    "selectable, but no elasticity-based counts or prioritisation are shown "
    "until the data is available."
)


def render_pricing_opportunities(week: str) -> tuple[int, int, int]:
    """Render the week's dynamic pricing-opportunity summary.

    Shows the real ``(tier1, tier2, tier3)`` headcounts derived from the
    persisted Part-4 state and the weekly matrix. When that state cannot be
    loaded an explicit unavailable message is shown instead of headcounts,
    because an empty elasticity map is not evidence that no product is ready.

    Returns the rendered headcounts, all zero when the state is unavailable.
    """
    st.markdown("#### Pricing opportunities for this week")
    state = pricing_readiness_state()
    if state != READINESS_AVAILABLE:
        st.error(READINESS_UNAVAILABLE_TITLE)
        st.markdown(READINESS_UNAVAILABLE_EXPLANATION)
        st.caption(f"Pricing readiness state: {state}.")
        return (0, 0, 0)
    tier1, tier2, tier3 = pricing_tier_counts(week)
    if tier1:
        st.markdown(
            f"**{tier1} products have both a valid current price and usable "
            f"elasticity for this week.**"
        )
    else:
        st.info(
            "No products currently have both usable elasticity and a valid "
            "current price for this week."
        )
    st.caption(
        f"{tier2} more products have usable elasticity but no current price this "
        f"week · {tier3} remaining products are selectable via search."
    )
    return (tier1, tier2, tier3)


def render_pricing_response(res: dict) -> None:
    """Render a ``/pricing`` response for a business user."""
    if res.get("reason") == REASON_NO_ELASTICITY:
        _render_no_elasticity(res)
    else:
        _render_with_recommendation(res)


def _render_no_elasticity(res: dict) -> None:
    st.warning(NO_ELASTICITY_TITLE)
    st.markdown(NO_ELASTICITY_EXPLANATION)
    st.markdown(f"**{NO_ELASTICITY_PROFIT}**")
    with st.expander("Technical diagnostic details"):
        cols = st.columns(4)
        cols[0].metric("Current price", f"{res.get('current_price') or 'n/a'}")
        cols[1].metric("Baseline demand", f"{res.get('baseline_demand') or 'n/a'}")
        cols[2].metric("Elasticity", f"{res.get('elasticity') or 'n/a'}")
        cols[3].metric("Status", res.get("elasticity_status") or "n/a")
        st.info(
            f"Reason: **{res['reason']}** · objective=n/a · optimal price=n/a · "
            f"profit_computable={res['profit_computable']}"
        )
        st.caption(
            f"Model v{res['model']['version']} · "
            f"inference_logged={res['inference']['inference_logged']}"
        )


def _render_with_recommendation(res: dict) -> None:
    st.success(f"Recommended price: **{res.get('recommended_price', 'n/a')}**")
    cols = st.columns(4)
    cols[0].metric("Current price", f"{res.get('current_price') or 'n/a'}")
    cols[1].metric("Baseline demand", f"{res.get('baseline_demand') or 'n/a'}")
    cols[2].metric("Elasticity", f"{res.get('elasticity') or 'n/a'}")
    cols[3].metric("Status", res.get("elasticity_status") or "n/a")
    st.info(
        f"Reason: **{res['reason']}** · objective={res.get('objective') or 'n/a'} · "
        f"optimal price={res.get('optimal_price') or 'n/a'} · "
        f"profit_computable={res['profit_computable']}"
    )
    g = res.get("guardrails")
    if g:
        st.caption(
            f"Guardrails: margin_floor={g['margin_floor_applied']} · "
            f"ceiling={g['ceiling_applied']} · floor={g['floor_applied']} · "
            f"smoothed={g['smoothed']}"
        )
    if res.get("scenarios"):
        st.dataframe(pd.DataFrame(res["scenarios"]), use_container_width=True)
    st.caption(
        f"Model v{res['model']['version']} · "
        f"inference_logged={res['inference']['inference_logged']}"
    )


def render_pricing_readiness(product_id: str, week: str) -> None:
    """Business summary for a product before the recommendation is requested.

    Shows only values that actually exist in the persisted Part-4 state and the
    weekly matrix. A clear reason is given when the product is not ready for the
    selected week; nothing is fabricated.
    """
    status = product_pricing_status(product_id, week)
    st.markdown("**Selected product**")
    st.markdown(f"Product ID: `{product_id}`")
    if status["status"] == STATUS_READINESS_UNAVAILABLE:
        st.error(READINESS_UNAVAILABLE_TITLE)
        st.markdown(READINESS_UNAVAILABLE_EXPLANATION)
        _readiness_metrics(status, show_price=status["current_price"] is not None)
        return
    if status["status"] == STATUS_READY:
        cols = st.columns(4)
        cols[0].metric("Current price", f"${status['current_price']:.2f}")
        cols[1].metric("Elasticity", f"{status['elasticity']:.2f}")
        cols[2].metric(
            "Elasticity source", elasticity_source_label(status["elasticity_source"])
        )
        cols[3].metric("Status", "Ready for pricing")
        return
    if status["status"] == STATUS_NO_PRICE:
        st.warning(NO_PRICE_TITLE)
        st.markdown(NO_PRICE_EXPLANATION)
        st.markdown(f"**{NO_PRICE_PROFIT}**")
        _readiness_metrics(status, show_price=False)
        return
    if status["status"] == STATUS_NO_ELASTICITY:
        st.warning(NO_ELASTICITY_TITLE)
        st.markdown(NO_ELASTICITY_EXPLANATION)
        st.markdown(f"**{NO_ELASTICITY_PROFIT}**")
        _readiness_metrics(status, show_price=status["current_price"] is not None)


def _readiness_metrics(status: dict, *, show_price: bool) -> None:
    """Three-column metric row for a non-ready product (only real values)."""
    cols = st.columns(3)
    cols[0].metric(
        "Current price",
        f"${status['current_price']:.2f}" if show_price else "n/a",
    )
    elasticity = status.get("elasticity")
    cols[1].metric(
        "Elasticity",
        f"{elasticity:.2f}" if elasticity is not None else "n/a",
    )
    source = status.get("elasticity_source")
    cols[2].metric(
        "Elasticity source",
        elasticity_source_label(source) if source else "n/a",
    )