"""Integration tests: the shared product selector is used by all sections.

These use ``streamlit.testing.v1.AppTest`` against ``_probe_app.py``, which
mirrors dashboard/app.py wiring exactly: each section renders the same
``ProductSelector`` and ``week_selector`` and, on Run, records the selected
product / week / cost the section would forward to the API.

The probe never hits the network: it only proves the selected value travelling
through the shared selector is a real catalog product, forwarded unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.product_catalog import is_real_product

PROBE = Path(__file__).with_name("_probe_app.py")
SAMPLE_REAL = "00066f42aeeb9f3007548bb9d3f33c38"

TAG = "\u00b7 Pricing-ready"
PRICE_AVAILABLE = "Price available"
NO_PRICE_WEEK = "No price this week"


def _raw_id(label: str) -> str:
    """Strip the pricing-ready tag from a selectbox option label."""
    return label.split(f" {TAG}")[0]


def _new_app() -> AppTest:
    at = AppTest.from_file(str(PROBE))
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture()
def app():
    return _new_app()


def _select_product(at: AppTest, section: str, query: str) -> str:
    at.sidebar.radio[0].set_value(section)
    at.run(timeout=120)
    key_prefix = {"Forecast": "forecast", "Pricing": "pricing", "Explainability": "explain"}[section]
    at.text_input(key=f"{key_prefix}:search").set_value(query)
    at.run(timeout=120)
    options = at.selectbox(key=f"{key_prefix}:select").options
    assert options, f"no filter matches for {query}"
    real = options[0]
    at.selectbox(key=f"{key_prefix}:select").select(real)
    at.run(timeout=120)
    return real


def test_selector_renders_search_for_each_section(app: AppTest):
    for section, prefix in (
        ("Forecast", "forecast"),
        ("Pricing", "pricing"),
        ("Explainability", "explain"),
    ):
        app.sidebar.radio[0].set_value(section)
        app.run(timeout=120)
        assert not app.exception, [e.value for e in app.exception]
        assert app.text_input(key=f"{prefix}:search").value is not None
        assert app.selectbox(key=f"{prefix}:select").options
        assert len(app.sidebar.radio[0].options) == 3


def test_products_shown_are_real(app: AppTest):
    app.sidebar.radio[0].set_value("Forecast")
    app.run(timeout=120)
    opts = app.selectbox(key="forecast:select").options
    assert opts, "selectbox must offer products"
    for pid in opts:
        assert is_real_product(pid)


def test_filtering_narrows_products(app: AppTest):
    app.sidebar.radio[0].set_value("Forecast")
    app.run(timeout=120)
    before = app.selectbox(key="forecast:select").options
    app.text_input(key="forecast:search").set_value("0006")
    app.run(timeout=120)
    after = app.selectbox(key="forecast:select").options
    assert len(after) <= len(before)
    assert all("0006" in pid for pid in after)


def test_forecast_selects_real_product_and_forwards_it(app: AppTest):
    real = _select_product(app, "Forecast", "00066f42")
    app.button(key="forecast:run").click()
    app.run(timeout=120)
    assert not app.exception, [e.value for e in app.exception]
    (line,) = [m.value for m in app.markdown if m.value.startswith("FORECAST:")]
    _, pid, week, _ = line.split(":")
    assert pid == real
    assert is_real_product(pid)
    assert week == "2018-05-14"


def test_pricing_selects_real_product_and_forwards_cost(app: AppTest):
    real = _select_product(app, "Pricing", "00066f42")
    app.number_input(key="pricing:cost").set_value(20.0)
    app.button(key="pricing:run").click()
    app.run(timeout=120)
    assert not app.exception, [e.value for e in app.exception]
    (line,) = [m.value for m in app.markdown if m.value.startswith("PRICING:")]
    _, pid, week, cost = line.split(":")
    assert pid == real
    assert is_real_product(pid)
    assert week == "2018-05-14"
    assert cost == "20.0"


def test_explain_selects_real_product_and_forwards_it(app: AppTest):
    real = _select_product(app, "Explainability", "00066f42")
    app.button(key="explain:run").click()
    app.run(timeout=120)
    assert not app.exception, [e.value for e in app.exception]
    (line,) = [m.value for m in app.markdown if m.value.startswith("EXPLAIN:")]
    _, pid, week, _ = line.split(":")
    assert pid == real
    assert is_real_product(pid)
    assert week == "2018-05-14"


def test_no_matches_shows_guidance(app: AppTest):
    app.sidebar.radio[0].set_value("Forecast")
    app.run(timeout=120)
    app.text_input(key="forecast:search").set_value("zzzz-nope")
    app.run(timeout=120)
    captions = [c.value for c in app.caption]
    assert any("No product matches" in c for c in captions)
    assert all(sb.key != "forecast:select" for sb in app.selectbox)


def test_pricing_tab_leads_with_pricing_ready_products():
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    opts = at.selectbox(key="pricing:select").options
    assert opts, "pricing selector must offer products"
    assert "Pricing-ready" in opts[0], (
        "un-narrowed pricing selector must list usable-elasticity products first"
    )
    first_untagged = next(
        (i for i, label in enumerate(opts) if "Pricing-ready" not in label),
        len(opts),
    )
    assert first_untagged >= 1
    assert all("Pricing-ready" in label for label in opts[:first_untagged])
    assert is_real_product(_raw_id(opts[0])), "tagged label must wrap a real product id"


def test_pricing_tab_no_elasticity_product_reachable_and_forwarded_unchanged():
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    at.text_input(key="pricing:search").set_value("00066f42")
    at.run(timeout=120)
    opts = at.selectbox(key="pricing:select").options
    assert opts
    pid = _raw_id(opts[0])
    assert is_real_product(pid)
    assert "Pricing-ready" not in opts[0], "the NO_ELASTICITY product must stay untagged"
    at.selectbox(key="pricing:select").select(pid)
    at.run(timeout=120)
    at.number_input(key="pricing:cost").set_value(10.0)
    at.button(key="pricing:run").click()
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    (line,) = [m.value for m in at.markdown if m.value.startswith("PRICING:")]
    _, forwarded, week, _ = line.split(":")
    assert forwarded == pid, "raw product_id must be forwarded unchanged"
    assert is_real_product(forwarded)


def test_pricing_tab_price_available_products_lead_by_default():
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    opts = at.selectbox(key="pricing:select").options
    assert opts, "pricing selector must offer products"
    assert PRICE_AVAILABLE in opts[0], (
        "at the default reference week the first options must have a current price"
    )
    assert TAG in opts[0]
    assert is_real_product(_raw_id(opts[0]))
    assert any(NO_PRICE_WEEK in label for label in opts[:1500]), (
        "elasticity-only products must still be shown (tagged, not hidden)"
    )


def _week_with_different_price_availability() -> str:
    from dashboard.product_catalog import (
        DEFAULT_WEEK,
        load_available_weeks,
        load_price_available_for_week,
        load_pricing_ready_sources,
    )

    ready = load_pricing_ready_sources()
    priced_default = load_price_available_for_week(DEFAULT_WEEK) & ready
    return next(
        w
        for w in load_available_weeks()
        if w != DEFAULT_WEEK and (load_price_available_for_week(w) & ready) != priced_default
    )


def test_pricing_week_change_reorders_and_forwards_week():
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    default_first = at.selectbox(key="pricing:select").options[0]
    other = _week_with_different_price_availability()
    at.selectbox(key="pricing:week").select(other)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    other_first = at.selectbox(key="pricing:select").options[0]
    assert other_first != default_first, (
        "changing the selected week must re-order the pricing options"
    )
    pid = _raw_id(other_first)
    at.selectbox(key="pricing:select").select(pid)
    at.button(key="pricing:run").click()
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    (line,) = [m.value for m in at.markdown if m.value.startswith("PRICING:")]
    _, forwarded, week, _ = line.split(":")
    assert forwarded == pid, "raw product_id must be forwarded unchanged"
    assert week == other, "the changed week must be forwarded unchanged"


def test_pricing_no_price_observation_product_can_still_run_and_forward():
    # 0009406f... is pricing-ready (elasticity) but has no valid price at the
    # default week -> the backend returns NO_PRICE_OBSERVATION there. The
    # dashboard must still let the manager select and forward it unchanged.
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    at.text_input(key="pricing:search").set_value("0009406fd7479715e4bef61dd91f2462")
    at.run(timeout=120)
    opts = at.selectbox(key="pricing:select").options
    assert opts
    pid = _raw_id(opts[0])
    assert pid == "0009406fd7479715e4bef61dd91f2462"
    assert NO_PRICE_WEEK in opts[0], "elasticity-only product must be tagged as such"
    at.selectbox(key="pricing:select").select(pid)
    at.run(timeout=120)
    at.number_input(key="pricing:cost").set_value(10.0)
    at.button(key="pricing:run").click()
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    (line,) = [m.value for m in at.markdown if m.value.startswith("PRICING:")]
    _, forwarded, week, _ = line.split(":")
    assert forwarded == pid
    assert is_real_product(forwarded)
    assert week == "2018-05-14"


def test_never_hardcodes_fake_ids_in_markdown(app: AppTest):
    real = _select_product(app, "Forecast", "00066f42")
    app.button(key="forecast:run").click()
    app.run(timeout=120)
    values = " ".join(m.value for m in app.markdown)
    for fake in ("1002", "12345", "P1"):
        assert f":{fake}:" not in values


def _opportunities(at: AppTest) -> tuple[int, int, int]:
    """Parse the probe's machine-readable opportunities summary line."""
    lines = [m.value for m in at.markdown if m.value.startswith("PRICING_OPPORTUNITIES:")]
    assert len(lines) == 1, f"expected exactly one opportunities line, got {lines}"
    _, t1, t2, t3 = lines[0].split(":")
    return int(t1), int(t2), int(t3)


def test_pricing_opportunities_count_is_dynamic_and_matches_tier_counts():
    week = "2018-05-14"
    from dashboard.product_catalog import pricing_tier_counts

    expected = pricing_tier_counts(week)
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert _opportunities(at) == expected, (
        "summary must use the real data-driven tier counts (no hard-coded numbers)"
    )


def test_pricing_opportunities_count_changes_with_selected_week():
    from dashboard.product_catalog import (
        DEFAULT_WEEK,
        load_available_weeks,
        load_price_available_for_week,
        load_pricing_ready_sources,
        pricing_tier_counts,
    )

    ready = load_pricing_ready_sources()
    priced_default = load_price_available_for_week(DEFAULT_WEEK) & ready
    other = next(
        w
        for w in load_available_weeks()
        if w != DEFAULT_WEEK and (load_price_available_for_week(w) & ready) != priced_default
    )
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    default_t1 = _opportunities(at)[0]
    at.selectbox(key="pricing:week").select(other)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    other_counts = _opportunities(at)
    assert other_counts == pricing_tier_counts(other), (
        "summary must follow the newly selected week"
    )
    assert other_counts[0] != default_t1, "price-eligible headcount must differ across weeks"


def test_pricing_week_with_no_price_eligible_products_shows_business_message():
    from dashboard.product_catalog import load_available_weeks, pricing_tier_counts

    empty = next((w for w in load_available_weeks() if pricing_tier_counts(w)[0] == 0), None)
    if empty is None:
        pytest.skip("no servable week has zero price-eligible products in this data")
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    at.selectbox(key="pricing:week").select(empty)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert not any("no products" in str(w.value).lower() for w in at.warning), (
        "an empty-product day is informational, not an error"
    )
    infos = [i.value for i in at.info]
    assert any("no products" in i.lower() for i in infos), (
        "a clear business message must explain the empty tier-1 (no hard-coded count)"
    )
    assert _opportunities(at)[0] == 0


def _summary_metric(at: AppTest, label: str, nth: int = 0) -> str | None:
    hits = [m for m in at.metric if m.label == label]
    if len(hits) <= nth:
        return None
    return hits[nth].value


def _raw_product_id(label: str) -> str:
    """Leading 32-hex real product id of a (possibly tagged) option label."""
    match = re.match(r"^([0-9a-f]{32})", label)
    assert match, f"option label must start with a real product id: {label!r}"
    return match.group(1)


def test_pricing_selected_ready_product_shows_real_summary():
    from dashboard.product_catalog import (
        load_elasticity_details,
        load_week_prices,
        pricing_tier_counts,
    )

    week = "2018-05-14"
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    opts = at.selectbox(key="pricing:select").options
    pid = _raw_product_id(opts[0])
    at.selectbox(key="pricing:select").select(pid)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    at.number_input(key="pricing:cost").set_value(20.0)
    at.button(key="pricing:run").click()
    at.run(timeout=120)
    (line,) = [m.value for m in at.markdown if m.value.startswith("PRICING:")]
    assert line.split(":")[1] == pid, "the selected tier-1 product must be forwarded"
    assert _summary_metric(at, "Status") == "Ready for pricing"
    assert _summary_metric(at, "Current price") == (
        f"${load_week_prices(week)[pid]:.2f}"
    )
    elasticity, source = load_elasticity_details()[pid]
    assert _summary_metric(at, "Elasticity") == f"{elasticity:.2f}"
    from dashboard.product_catalog import elasticity_source_label

    assert _summary_metric(at, "Elasticity source") == elasticity_source_label(source)
    assert pricing_tier_counts(week)[0] >= 1


def test_pricing_selected_no_price_product_shows_persisted_values_and_reason():
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    at.text_input(key="pricing:search").set_value("0009406fd7479715e4bef61dd91f2462")
    at.run(timeout=120)
    opts = at.selectbox(key="pricing:select").options
    pid = _raw_product_id(opts[0])
    assert pid == "0009406fd7479715e4bef61dd91f2462"
    at.selectbox(key="pricing:select").select(pid)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    from dashboard.product_catalog import load_elasticity_details

    elasticity, source = load_elasticity_details()[pid]
    assert any("Price recommendation unavailable" in str(w.value) for w in at.warning)
    markdown = "\n".join(str(m.value) for m in at.markdown).lower()
    assert "does not have a valid current price for the selected week" in markdown
    assert _summary_metric(at, "Current price") in (None, "n/a")
    assert _summary_metric(at, "Elasticity") == f"{elasticity:.2f}"
    from dashboard.product_catalog import elasticity_source_label

    assert _summary_metric(at, "Elasticity source") == elasticity_source_label(source)


def test_pricing_selected_no_elasticity_product_shows_price_and_reason():
    at = _new_app()
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    at.text_input(key="pricing:search").set_value("00066f42")
    at.run(timeout=120)
    opts = at.selectbox(key="pricing:select").options
    pid = _raw_product_id(opts[0])
    assert is_real_product(pid)
    at.selectbox(key="pricing:select").select(pid)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Price recommendation unavailable" in str(w.value) for w in at.warning)
    markdown = "\n".join(str(m.value) for m in at.markdown).lower()
    assert "does not have sufficient historical price-demand variation" in markdown
    assert _summary_metric(at, "Elasticity") == "n/a"
    assert _summary_metric(at, "Elasticity source") == "n/a"
    assert _summary_metric(at, "Current price") is not None, (
        "the real existing price must be shown, never fabricated"
    )


def test_pricing_missing_readiness_artifact_shows_unavailable_not_zero_counts():
    from dashboard.product_catalog import DEFAULT_WEEK, total_products

    at = _new_app()
    at.session_state["probe_missing_readiness"] = True
    at.sidebar.radio[0].set_value("Pricing")
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]

    errors = " ".join(str(e.value) for e in at.error)
    assert "Pricing readiness data is unavailable" in errors
    markdown = "\n".join(str(m.value) for m in at.markdown)
    assert "usable elasticity for this week" not in markdown, (
        "headcounts must not be presented when readiness data is unavailable"
    )
    assert _opportunities(at) == (0, 0, 0)

    at.text_input(key="pricing:search").set_value("00066f42")
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    opts = at.selectbox(key="pricing:select").options
    assert opts, "the catalog must stay searchable when readiness data is missing"
    pid = _raw_product_id(opts[0])
    assert is_real_product(pid)
    at.selectbox(key="pricing:select").select(pid)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert "Pricing readiness data is unavailable" in " ".join(
        str(e.value) for e in at.error
    )
    assert _summary_metric(at, "Elasticity") == "n/a"
    assert _summary_metric(at, "Current price") is not None, (
        "matrix prices stay real even when readiness data is unavailable"
    )
    assert total_products() == 32729
    assert DEFAULT_WEEK
