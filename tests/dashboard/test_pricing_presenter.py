"""UI tests for the pricing presenter (NO_ELASTICITY UX).

The presenter is the exact function the deployed Pricing tab renders. We feed
it raw /pricing API responses (contract fixtures) through an AppTest probe and
assert only on the rendered UI: business copy for NO_ELASTICITY, unchanged
rendering for every other reason, and no fabricated price.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROBE = Path(__file__).with_name("_pricing_presenter_probe.py")

MODEL = {"name": "demand_forecasting_final", "version": 2}
INFERENCE = {"request_id": "abc", "inference_logged": False}

NO_ELASTICITY = {
    "product_id": "00066f42aeeb9f3007548bb9d3f33c38",
    "week": "2018-05-14",
    "baseline_demand": 1.1569295662414527,
    "category": "nan",
    "current_price": 101.65,
    "elasticity_status": None,
    "elasticity_source": None,
    "elasticity": None,
    "reason": "NO_ELASTICITY",
    "profit_computable": False,
    "scenarios": [],
    "model": MODEL,
    "inference": INFERENCE,
}

OPTIMIZED = {
    "product_id": "005030ef108f58b46b78116f754d8d38",
    "week": "2018-05-14",
    "baseline_demand": 1.2028221225976239,
    "category": "perfumaria",
    "current_price": 13.99,
    "elasticity_status": "Unsupported",
    "elasticity_source": "category_pooled",
    "elasticity": -0.9674207023796844,
    "reason": "optimized",
    "profit_computable": True,
    "objective": "profit",
    "optimal_price": 16.788,
    "optimal_multiplier": 1.2,
    "optimal_revenue": 16.927732638028896,
    "optimal_profit": 6.844498996124622,
    "recommended_price": 16.79,
    "extrapolation_required": True,
    "guardrails": {
        "margin_floor_applied": True,
        "margin_floor_inactive_no_cost": False,
        "ceiling_applied": False,
        "floor_applied": False,
        "smoothed": False,
        "notes": [],
    },
    "scenarios": [
        {"multiplier": 1.0, "price": 13.99, "pred_demand": 1.2, "revenue": 16.8, "profit": 4.8}
    ],
    "model": MODEL,
    "inference": INFERENCE,
}

NO_PRICE_OBSERVATION = {
    **NO_ELASTICITY,
    "reason": "NO_PRICE_OBSERVATION",
    "current_price": None,
    "scenarios": [],
}

PROFIT_NOT_COMPUTABLE = {
    **OPTIMIZED,
    "reason": "PROFIT_NOT_COMPUTABLE (no cost supplied)",
    "profit_computable": False,
    "optimal_profit": None,
}


def _render(res: dict) -> AppTest:
    at = AppTest.from_file(str(PROBE))
    at.session_state["pricing_response"] = res
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_business_message_fields() -> None:
    at = _render(NO_ELASTICITY)
    warnings = [w.value for w in at.warning]
    assert warnings, "NO_ELASTICITY must render a warning, not a success/fake price"
    assert any("Price recommendation unavailable" in w for w in warnings)
    md = "\n".join(m.value for m in at.markdown)
    assert "This product does not have sufficient historical price-demand" in md
    assert "to estimate price elasticity reliably." in md
    assert "Profit optimization" in md
    assert "not performed because elasticity is unavailable." in md


def test_no_price_recommendation_shown_for_no_elasticity() -> None:
    at = _render(NO_ELASTICITY)
    successes = [s.value for s in at.success]
    assert not any("Recommended price" in s for s in successes), (
        "NO_ELASTICITY must not present a recommended (or fake) price"
    )
    md = "\n".join(m.value for m in at.markdown)
    assert "recommended_price" not in md.replace("Price recommendation unavailable", "")
    assert "16.79" not in md and "520" not in md


def test_technical_details_retained_in_expander() -> None:
    at = _render(NO_ELASTICITY)
    expander_titles = [e.label for e in at.expander]
    assert any("Technical diagnostic details" in t for t in expander_titles)
    infos = "\n".join(i.value for i in at.info)
    assert "NO_ELASTICITY" in infos
    assert "profit_computable=False" in infos
    assert "objective=n/a" in infos and "optimal price=n/a" in infos


def test_optimized_response_unchanged() -> None:
    at = _render(OPTIMIZED)
    successes = [s.value for s in at.success]
    assert any("Recommended price: **16.79**" in s for s in successes)
    infos = "\n".join(i.value for i in at.info)
    assert "objective=profit" in infos
    assert "profit_computable=True" in infos
    assert "optimal price=16.788" in infos
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Current price"] == "13.99"
    assert metrics["Baseline demand"] == "1.2028221225976239"
    assert metrics["Elasticity"] == "-0.9674207023796844"
    assert metrics["Status"] == "Unsupported"
    guardrail_captions = "\n".join(c.value for c in at.caption)
    assert "margin_floor=True" in guardrail_captions


def test_no_price_observation_reason_behaves_as_before() -> None:
    at = _render(NO_PRICE_OBSERVATION)
    assert not at.exception
    successes = [s.value for s in at.success]
    assert any("Recommended price: **n/a**" in s for s in successes)
    infos = "\n".join(i.value for i in at.info)
    assert "NO_PRICE_OBSERVATION" in infos
    assert "profit_computable=False" in infos


def test_profit_not_computable_reason_behaves_as_before() -> None:
    at = _render(PROFIT_NOT_COMPUTABLE)
    assert not at.exception
    successes = [s.value for s in at.success]
    assert any("Recommended price: **16.79**" in s for s in successes)
    infos = "\n".join(i.value for i in at.info)
    assert "PROFIT_NOT_COMPUTABLE (no cost supplied)" in infos
    assert "profit_computable=False" in infos


def test_no_response_injected_renders_placeholder() -> None:
    at = AppTest.from_file(str(PROBE))
    at.run(timeout=120)
    assert not at.exception
    assert any("NO_RESPONSE" in m.value for m in at.markdown)