"""Catalog logic tests for the dashboard product selector.

The authoritative source is the dense weekly demand matrix (the same file the
deployment API consumes). The protected held-out split is never opened from the
dashboard package (verified statically in test_deployment_safety.py and
behaviourally here: the catalog is derived solely from the matrix).
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from dashboard import product_catalog as pc
from dashboard.product_catalog import PRICING_READY_REPORT
from src.optimization import pricing_data

SAMPLE_REAL = "00066f42aeeb9f3007548bb9d3f33c38"
OIDST_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(scope="module")
def catalog():
    return pc.load_catalog()


def test_catalog_contains_only_real_product_ids(catalog):
    ids = catalog[pc.PRODUCT_ID].astype(str).tolist()
    assert len(ids) == len(set(ids)), "catalog product ids must be unique"
    assert ids, "catalog must not be empty"
    assert all(OIDST_ID_RE.fullmatch(pid) for pid in ids), (
        "every catalog id must be a real 32-hex product id"
    )


def test_catalog_never_synthesises_ids(catalog):
    for fake in ("12345", "P1", "product-1", "00000000000000000000000000000000"):
        assert fake not in catalog[pc.PRODUCT_ID].astype(str).tolist()


def test_catalog_matches_matrix_product_set():
    matrix = pricing_data.load_matrix()
    matrix_ids = set(matrix[pricing_data.PRODUCT_COL].astype(str))
    catalog_ids = set(pc.load_catalog()[pc.PRODUCT_ID].astype(str))
    assert catalog_ids <= matrix_ids, "catalog ids must come from the matrix"
    assert catalog_ids == matrix_ids, (
        "every servable matrix product must remain selectable"
    )


def test_catalog_serves_only_servable_weeks():
    frame = pricing_data.load_matrix()
    max_servable = frame[frame[pricing_data.WEEK_COL].astype(str) <= pc.DEFAULT_WEEK]
    expected = set(max_servable[pricing_data.PRODUCT_COL].astype(str))
    got = set(pc.load_catalog()[pc.PRODUCT_ID].astype(str))
    assert got == expected


def test_sample_real_product_is_present_and_labeled(catalog):
    mask = catalog[pc.PRODUCT_ID].astype(str) == SAMPLE_REAL
    assert mask.any(), "known real product must be in the catalog"
    label = catalog.loc[mask, pc.DISPLAY].iloc[0]
    assert label.startswith(SAMPLE_REAL)


def test_product_options_empty_returns_capped_real_options():
    ids, total = pc.product_options("")
    assert total == pc.total_products() == len(pc.load_catalog())
    assert all(OIDST_ID_RE.fullmatch(pid) for pid in ids)


def test_product_options_filters_by_substring():
    ids, total = pc.product_options("0006")
    assert total >= 1
    assert all("0006" in pid for pid in ids)
    assert all(OIDST_ID_RE.fullmatch(pid) for pid in ids)


def test_product_options_case_insensitive():
    ids_lower, total_lower = pc.product_options("00066f42")
    ids_upper, total_upper = pc.product_options("00066F42")
    assert total_lower == total_upper
    assert set(ids_lower) == set(ids_upper)


def test_product_options_no_match_returns_empty():
    ids, total = pc.product_options("zzzz-nope")
    assert ids == [] and total == 0


def test_product_label_real(capfd):
    label = pc.product_label(SAMPLE_REAL)
    assert label.startswith(SAMPLE_REAL)


def test_product_label_unknown_falls_back(capfd):
    assert pc.product_label("deadbeef000000000000000000000001") == "deadbeef000000000000000000000001"


def test_is_real_product():
    assert pc.is_real_product(SAMPLE_REAL)
    assert not pc.is_real_product("12345")
    assert not pc.is_real_product("")
    assert not pc.is_real_product("deadbeef000000000000000000000001")


def test_available_weeks_are_mondays_before_test_boundary():
    weeks = pc.load_available_weeks()
    assert weeks, "servable weeks must not be empty"
    for w in weeks:
        d = dt.date.fromisoformat(w)
        assert d.weekday() == 0, f"{w} is not a Monday"
        assert w < pricing_data.TEST_START_WEEK, f"{w} is not servable"
        assert w <= pc.DEFAULT_WEEK


def test_default_week_is_servable_and_listed_first():
    options = pc.week_options()
    assert options[0] == pc.DEFAULT_WEEK
    assert pc.DEFAULT_WEEK in options


PRICING_READY_TAG = "\u00b7 Pricing-ready"
PRICE_AVAILABLE_TAG = "Price available"
NO_PRICE_WEEK_TAG = "No price this week"


def _expected_pricing_tiers(week: str) -> tuple[list[str], list[str], list[str]]:
    """Data-driven expectation for ``pricing_options`` on ``week``: (price-ready,
    elasticity-only, rest) all in catalog order."""
    catalog_ids = pc.load_catalog()[pc.PRODUCT_ID].astype(str).tolist()
    ready = pc.load_pricing_ready_sources()
    priced = pc.load_price_available_for_week(week)
    return (
        [pid for pid in catalog_ids if pid in ready and pid in priced],
        [pid for pid in catalog_ids if pid in ready and pid not in priced],
        [pid for pid in catalog_ids if pid not in ready],
    )


def test_pricing_ready_sources_come_from_persisted_part4_state():
    ready = pc.load_pricing_ready_sources()
    assert isinstance(ready, frozenset)
    assert ready, "persisted Part-4 state must expose usable-elasticity products"
    for pid in ready:
        assert pc.is_real_product(pid), "pricing-ready ids must be real catalog products"


def test_pricing_ready_html_labels_are_business_facing():
    ready = pc.load_pricing_ready_sources()
    pid = next(iter(ready))
    label = pc.pricing_product_label(pid)
    assert pid in label
    assert "elasticity" not in label.lower(), "label must not leak technical detail"
    assert PRICING_READY_TAG in label


def test_pricing_ready_label_absent_for_no_elasticity_product():
    label = pc.pricing_product_label(SAMPLE_REAL)
    assert SAMPLE_REAL in label
    assert PRICING_READY_TAG not in label, "no-elasticity product must not be tagged"


def test_product_label_never_tagged_by_default():
    ready = pc.load_pricing_ready_sources()
    pid = next(iter(ready))
    assert PRICING_READY_TAG not in pc.product_label(pid), (
        "default (Forecast/Explain) label must not change"
    )


def test_pricing_options_prioritise_pricing_ready_products():
    ids, total = pc.pricing_options("")
    assert total == pc.total_products(), "pricing must keep the full catalog"
    ready = pc.load_pricing_ready_sources()
    first_non_ready = next(
        (i for i, pid in enumerate(ids) if pid not in ready), len(ids)
    )
    assert first_non_ready >= 1, "at least one pricing-ready product must lead"
    assert all(pid in ready for pid in ids[:first_non_ready])


def test_pricing_options_keep_no_elasticity_products_reachable():
    ids, _ = pc.pricing_options(SAMPLE_REAL)
    assert ids == [SAMPLE_REAL], (
        "typing the product id must still surface the NO_ELASTICITY product"
    )


def test_pricing_options_search_cap_and_filter_match_product_options():
    for q in ("0006", "00066f42", "zzzz-nope"):
        p_ids, p_total = pc.pricing_options(q, week=pc.DEFAULT_WEEK)
        o_ids, o_total = pc.product_options(q)
        assert len(p_ids) <= pc._MAX_MATCHES
        assert p_total == o_total
        assert p_ids == o_ids, (
            "when narrowed by search, pricing options must match normal options "
            "(and must not depend on the selected week)"
        )


def test_pricing_ordering_is_data_driven_deterministic():
    week = pc.DEFAULT_WEEK
    a, _ = pc.pricing_options("", week=week)
    b, _ = pc.pricing_options("", week=week)
    assert a == b, "ordering must be deterministic across calls"
    assert len(a) == len(set(a)) == pc._MAX_MATCHES
    tiers = _expected_pricing_tiers(week)
    expected = (tiers[0] + tiers[1] + tiers[2])[: pc._MAX_MATCHES]
    assert a == expected, (
        "order must be price-available, then elasticity-only, then the rest "
        "(each in catalog order)"
    )


def test_pricing_price_available_products_lead_over_elasticity_only():
    week = pc.DEFAULT_WEEK
    price_now, elasticity_only, _ = _expected_pricing_tiers(week)
    assert price_now, "sanity: at least one product is demo-ready at the default week"
    ids, total = pc.pricing_options("", week=week)
    assert total == pc.total_products(), "pricing must keep the full catalog"
    assert ids[: len(price_now)] == price_now
    ready = pc.load_pricing_ready_sources()
    priced = pc.load_price_available_for_week(week)
    for pid in price_now[:50]:
        assert pid in ready and pid in priced


def test_elasticity_without_current_price_remains_selectable():
    week = pc.DEFAULT_WEEK
    _, elasticity_only, _ = _expected_pricing_tiers(week)
    assert elasticity_only, "sanity: some ready products lack a current price"
    pid = elasticity_only[0]
    assert pc.load_pricing_ready_sources() >= {pid}
    assert pid not in pc.load_price_available_for_week(week)
    hit, _ = pc.pricing_options(pid, week=week)
    assert hit == [pid], "elasticity-only products stay reachable by typing"
    full, _ = pc.pricing_options("", week=week)
    assert pid in full, "elasticity-only products stay in the un-narrowed catalogue"


def test_without_elasticity_remains_selectable_via_search():
    week = pc.DEFAULT_WEEK
    hit, total = pc.pricing_options(SAMPLE_REAL, week=week)
    assert hit == [SAMPLE_REAL]
    assert total == 1


def test_pricing_week_change_reorders_price_availability():
    default = pc.DEFAULT_WEEK
    ready = pc.load_pricing_ready_sources()
    priced_default = pc.load_price_available_for_week(default) & ready
    other = next(
        w
        for w in pc.load_available_weeks()
        if w != default and (pc.load_price_available_for_week(w) & ready) != priced_default
    )
    ids_default, _ = pc.pricing_options("", week=default)
    ids_other, _ = pc.pricing_options("", week=other)
    t_default = _expected_pricing_tiers(default)
    t_other = _expected_pricing_tiers(other)
    assert ids_default[: len(t_default[0])] == t_default[0]
    assert ids_other[: len(t_other[0])] == t_other[0]
    assert t_default[0] != t_other[0], "tier-1 must differ between weeks"
    assert ids_default != ids_other, "the dropdown ordering must follow the selected week"


def test_pricing_label_price_available_for_ready_and_priced():
    week = pc.DEFAULT_WEEK
    price_now, _, _ = _expected_pricing_tiers(week)
    assert price_now
    label = pc.pricing_product_label(price_now[0], week=week)
    assert PRICING_READY_TAG in label
    assert PRICE_AVAILABLE_TAG in label


def test_pricing_label_no_price_this_week_for_elasticity_without_price():
    week = pc.DEFAULT_WEEK
    _, elasticity_only, _ = _expected_pricing_tiers(week)
    assert elasticity_only
    label = pc.pricing_product_label(elasticity_only[0], week=week)
    assert PRICING_READY_TAG in label
    assert NO_PRICE_WEEK_TAG in label


def test_price_labels_never_leak_ml_internals():
    week = pc.DEFAULT_WEEK
    for pid in pc.pricing_options("", week=week)[0][:20]:
        label = pc.pricing_product_label(pid, week=week).lower()
        for term in ("lightgbm", "one-hot", "lags", "regime", "93 feature"):
            assert term not in label


def test_forecast_options_unrestricted_by_elasticity():
    forecast_ids, _ = pc.product_options("")
    ready = pc.load_pricing_ready_sources()
    first_block = forecast_ids[: min(len(forecast_ids), 50)]
    assert any(pid not in ready for pid in first_block), (
        "Forecast must not be reordered by elasticity"
    )
    assert SAMPLE_REAL in forecast_ids, "NO_ELASTICITY product stays in Forecast catalog"


def test_no_product_ids_hardcoded_in_ordering_logic():
    src = pc.__file__
    with open(src, encoding="utf-8") as fh:
        module_source = fh.read()
    for pid in list(pc.load_pricing_ready_sources())[:200]:
        assert pid not in module_source, (
            "a real product id must never be hard-coded in the catalog module"
        )
    assert SAMPLE_REAL not in module_source or "SAMPLE_REAL" in module_source


def test_elasticity_source_labels_are_business_facing():
    assert pc.elasticity_source_label("own") == "Own"
    assert pc.elasticity_source_label("category_pooled") == "Category"
    assert pc.elasticity_source_label(None) == "n/a"


def test_tier1_label_embeds_real_metadata():
    week = pc.DEFAULT_WEEK
    price_now, _, _ = _expected_pricing_tiers(week)
    assert price_now
    pid = price_now[0]
    details = pc.load_elasticity_details()[pid]
    elasticity, source = details
    label = pc.pricing_product_label(pid, week=week)
    assert f"${pc.load_week_prices(week)[pid]:.2f}" in label, "real price must be shown"
    assert f"{elasticity:.2f}" in label, "real elasticity must be shown"
    assert pc.elasticity_source_label(source) in label, "real source label must be shown"
    assert PRICING_READY_TAG in label
    assert PRICE_AVAILABLE_TAG in label


def test_tier2_label_shows_real_elasticity_and_no_price():
    week = pc.DEFAULT_WEEK
    _, elasticity_only, _ = _expected_pricing_tiers(week)
    assert elasticity_only
    pid = elasticity_only[0]
    elasticity, source = pc.load_elasticity_details()[pid]
    label = pc.pricing_product_label(pid, week=week)
    assert NO_PRICE_WEEK_TAG in label
    assert "$" not in label, "no fabricated price for an unpriced week"
    assert f"{elasticity:.2f}" in label
    assert pc.elasticity_source_label(source) in label


def test_elasticity_details_consume_only_persisted_part4_state():
    details = pc.load_elasticity_details()
    assert details, "usable-elasticity details must be available"
    assert set(details) == set(pc.load_pricing_ready_sources()), (
        "ready set must exactly match the persisted elasticity details keys"
    )
    report_text = PRICING_READY_REPORT.read_text(encoding="utf-8")
    for pid, (elasticity, source) in list(details.items())[:20]:
        assert source in ("own", "category_pooled")
        assert isinstance(elasticity, float)
        assert pid in report_text, "every surfaced detail must exist in the persistence"


def test_product_pricing_status_ready():
    week = pc.DEFAULT_WEEK
    price_now, _, _ = _expected_pricing_tiers(week)
    pid = price_now[0]
    status = pc.product_pricing_status(pid, week)
    assert status["status"] == pc.STATUS_READY
    assert status["current_price"] == pc.load_week_prices(week)[pid]
    assert status["elasticity"] is not None
    assert status["elasticity_source"] in ("own", "category_pooled")


def test_product_pricing_status_no_price_carries_elasticity_when_available():
    week = pc.DEFAULT_WEEK
    _, elasticity_only, _ = _expected_pricing_tiers(week)
    assert elasticity_only
    pid = elasticity_only[0]
    status = pc.product_pricing_status(pid, week)
    assert status["status"] == pc.STATUS_NO_PRICE
    assert status["current_price"] is None
    assert status["elasticity"] is not None, "real elasticity must still be surfaced"
    assert status["elasticity_source"] in ("own", "category_pooled")


def test_product_pricing_status_no_elasticity_carries_price_when_available():
    week = pc.DEFAULT_WEEK
    status = pc.product_pricing_status(SAMPLE_REAL, week)
    assert status["status"] == pc.STATUS_NO_ELASTICITY
    assert status["current_price"] is not None, "real price must still be surfaced"
    assert status["elasticity"] is None
    assert status["elasticity_source"] is None


def test_pricing_tier_counts_match_expected_tiers_per_week():
    week = pc.DEFAULT_WEEK
    ready = pc.load_pricing_ready_sources()
    priced = pc.load_price_available_for_week(week)
    t1s, t2s, t3s = _expected_pricing_tiers(week)
    assert pc.pricing_tier_counts(week) == (len(t1s), len(t2s), len(t3s))
    assert len(t1s) == len(ready & priced)
    assert len(t2s) == len(ready - priced)
    assert len(t3s) == pc.total_products() - len(ready)


def test_pricing_ready_for_week_is_tier1_set():
    week = pc.DEFAULT_WEEK
    t1s, _, _ = _expected_pricing_tiers(week)
    assert pc.pricing_ready_for_week(week) == set(t1s)


def test_week_prices_exclude_zero_prices_exactly_like_backend():
    matrix = pricing_data.load_matrix()
    week = pc.DEFAULT_WEEK
    sub = matrix[matrix[pricing_data.WEEK_COL].astype(str) == week]
    expected = {
        str(pid): float(price)
        for pid, price in zip(
            sub[pricing_data.PRODUCT_COL].astype(str),
            sub[pricing_data.PRICE_COL],
        )
        if float(price) > 0
    }
    assert pc.load_week_prices(week) == expected


def test_readiness_state_is_available_when_artifact_is_present():
    assert pc.pricing_readiness_state() == pc.READINESS_AVAILABLE
    assert len(pc.load_elasticity_details()) == 17012


@pytest.fixture
def readiness_artifact(tmp_path, monkeypatch):
    """Repoint the catalog at a controllable Part-4 artifact path."""

    def _point_at(path):
        monkeypatch.setattr(pc, "PRICING_READY_REPORT", path)
        pc._readiness_report.cache_clear()
        pc._elasticity_details.cache_clear()
        pc.load_pricing_ready_sources.clear()
        pc.pricing_options.clear()
        return path

    yield _point_at
    pc._readiness_report.cache_clear()
    pc._elasticity_details.cache_clear()
    pc.load_pricing_ready_sources.clear()
    pc.pricing_options.clear()


def test_missing_artifact_is_observable_and_never_means_zero_elasticity(
    readiness_artifact, tmp_path
):
    readiness_artifact(tmp_path / "absent.json")

    assert pc.pricing_readiness_state() == pc.READINESS_MISSING
    assert pc.load_elasticity_details() == {}

    priced = sorted(pc.load_week_prices(pc.DEFAULT_WEEK))
    assert priced, "matrix prices are independent of the readiness artifact"
    status = pc.product_pricing_status(priced[0], pc.DEFAULT_WEEK)
    assert status["status"] == pc.STATUS_READINESS_UNAVAILABLE
    assert status["status"] != pc.STATUS_NO_ELASTICITY
    assert status["elasticity"] is None
    assert status["current_price"] is not None


def test_unreadable_artifact_is_observable(readiness_artifact, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    readiness_artifact(broken)

    assert pc.pricing_readiness_state() == pc.READINESS_UNREADABLE
    assert pc.load_elasticity_details() == {}


def test_catalog_search_stays_usable_when_artifact_is_missing(
    readiness_artifact, tmp_path
):
    readiness_artifact(tmp_path / "absent.json")

    ids, total = pc.pricing_options("", week=pc.DEFAULT_WEEK)
    assert total == pc.total_products() == 32729
    assert len(ids) == pc._MAX_MATCHES

    needle = SAMPLE_REAL[:12]
    found, matched = pc.pricing_options(needle, week=pc.DEFAULT_WEEK)
    assert matched == 1
    assert found == [SAMPLE_REAL]
