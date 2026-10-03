"""Authoritative product catalog for the manager UI.

The catalog is loaded from the same dense weekly demand matrix that the
deployment API consumes (``data/processed/demand_matrix_weekly.csv.gz``). This
is the exact source the API uses to build its own product catalog
(``app.services.state`` -> ``pricing_data.load_matrix``), so the selector can
only ever expose products that the inference pipeline can actually serve.

Product IDs are full 32-hex Olist ids; no IDs are ever synthesised, and the
protected held-out split file is never opened from this package.

Only products that appear in servable weeks (on or before the reference week,
strictly before the protected test boundary) are exposed, matching the API's
week rules.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache, partial
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.config import DEFAULT_WEEK
from src.optimization import pricing_data

PRODUCT_ID = pricing_data.PRODUCT_COL  # product_id
WEEK = pricing_data.WEEK_COL  # week (normalised to Monday start)
CATEGORY = pricing_data.CATEGORY_COL  # product_category_name
PRICE = pricing_data.PRICE_COL  # avg_price (current price observation)
DISPLAY = "label"

_MAX_MATCHES = 5000

PRICING_READY_TAG = "\u00b7 Pricing-ready"
PRICE_AVAILABLE_TAG = " · Price available"
NO_PRICE_TAG = " · No price this week"

STATUS_READY = "READY"
STATUS_NO_PRICE = "NO_PRICE_OBSERVATION"
STATUS_NO_ELASTICITY = "NO_ELASTICITY"
STATUS_READINESS_UNAVAILABLE = "READINESS_UNAVAILABLE"

READINESS_AVAILABLE = "available"
READINESS_MISSING = "missing"
READINESS_UNREADABLE = "unreadable"

_SOURCE_LABELS = {"own": "Own", "category_pooled": "Category"}


def elasticity_source_label(source: str | None) -> str:
    """Business label for a persisted elasticity source."""
    if not source:
        return "n/a"
    return _SOURCE_LABELS.get(source, source)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICING_READY_REPORT = (
    PROJECT_ROOT / "reports" / "forecasting" / "part4_pricing_engine.json"
)


@lru_cache(maxsize=1)
def _load_matrix_cached() -> pd.DataFrame:
    """The dense weekly demand matrix (exists once, reused by catalog + weeks)."""
    return pricing_data.load_matrix()


def _servable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows on or before the reference week (the usable inference horizon)."""
    return frame[frame[WEEK].astype(str) <= DEFAULT_WEEK]


@st.cache_data(show_spinner="Loading product catalog…")
def load_catalog() -> pd.DataFrame:
    """Distinct servable products with a display label.

    Returns a DataFrame with ``product_id`` (raw, authoritative) and ``label``
    (product id, optionally augmented with the product category). Only products
    present on or before the last servable week (the reference week) are kept,
    so the selector never offers products the inference pipeline cannot serve.
    """
    frame = _load_matrix_cached()
    servable = _servable_rows(frame)
    catalog = (
        servable.assign(_pid=servable[PRODUCT_ID].astype(str))
        .assign(_cat=servable[CATEGORY].fillna("").astype(str))
        .drop_duplicates("_pid")[["_pid", "_cat"]]
        .rename(columns={"_pid": PRODUCT_ID, "_cat": CATEGORY})
        .reset_index(drop=True)
    )
    catalog[DISPLAY] = catalog.apply(
        lambda row: _product_label(row[PRODUCT_ID], row[CATEGORY]), axis=1
    )
    return catalog


@st.cache_data(show_spinner="Loading available weeks…")
def load_available_weeks() -> tuple[str, ...]:
    """Servable week-start labels (ISO Mondays, on or before the reference week)."""
    frame = _load_matrix_cached()
    weeks = sorted({str(w) for w in _servable_rows(frame)[WEEK].astype(str)})
    return tuple(weeks)


def _product_label(product_id: str, category: str) -> str:
    """Display label: the real product id, with category when it is usable."""
    cleaned = category.strip()
    if cleaned and cleaned not in ("nan", "None"):
        return f"{product_id} — {cleaned}"
    return product_id


@lru_cache(maxsize=1)
def _catalog_index() -> dict[str, str]:
    """product_id -> display label lookup."""
    catalog = load_catalog()
    return dict(zip(catalog[PRODUCT_ID].astype(str), catalog[DISPLAY]))


@lru_cache(maxsize=1)
def _product_id_set() -> frozenset[str]:
    """Real product ids in the catalog (authoritative membership check)."""
    return frozenset(load_catalog()[PRODUCT_ID].astype(str))


def product_options(query: str = "") -> tuple[list[str], int]:
    """Product ids matching ``query`` plus the total number of matches.

    Matching is a case-insensitive substring search over the real product ids.
    Returns ``(ids, total)`` where ``ids`` is capped at ``_MAX_MATCHES`` for the
    dropdown and ``total`` is the full match count. With an empty query the full
    catalog is returned (capped for rendering).
    """
    catalog = load_catalog()
    ids = catalog[PRODUCT_ID].astype(str)
    q = (query or "").strip().lower()
    if q:
        ids = ids[ids.str.lower().str.contains(q, regex=False)]
    return ids.head(_MAX_MATCHES).tolist(), int(len(ids))


@lru_cache(maxsize=1)
def _readiness_report() -> tuple[str, dict]:
    """``(state, report)`` for the persisted Part-4 pricing artifact.

    ``state`` is ``READINESS_AVAILABLE`` when the artifact was read and parsed,
    ``READINESS_MISSING`` when the file is absent and ``READINESS_UNREADABLE``
    when it cannot be parsed. An unavailable artifact is never reported as an
    empty-but-valid elasticity population, so the UI can state that pricing
    readiness data is unavailable instead of claiming zero products are ready.
    """
    if not PRICING_READY_REPORT.is_file():
        return READINESS_MISSING, {}
    try:
        report = json.loads(PRICING_READY_REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return READINESS_UNREADABLE, {}
    if not isinstance(report, dict):
        return READINESS_UNREADABLE, {}
    return READINESS_AVAILABLE, report


def pricing_readiness_state() -> str:
    """Availability of the persisted Part-4 pricing readiness artifact."""
    return _readiness_report()[0]


@lru_cache(maxsize=1)
def _elasticity_details() -> dict[str, tuple[float, str]]:
    """persisted Part-4 ``elasticity.effective``: product_id -> (elasticity, source).

    Read-only consumption of the existing Part-4 pricing engine state
    (``reports/forecasting/part4_pricing_engine.json``). No elasticity is
    recomputed here; the same authoritative table the deployment API builds at
    startup. An unavailable artifact yields an empty map, and
    :func:`pricing_readiness_state` keeps that case distinguishable from a
    genuinely empty population.
    """
    state, report = _readiness_report()
    if state != READINESS_AVAILABLE:
        return {}
    eff = report.get("elasticity", {}).get("effective") or []
    details: dict[str, tuple[float, str]] = {}
    for row in eff:
        source = row.get("elasticity_source")
        elasticity = row.get("elasticity")
        if (
            source in ("own", "category_pooled")
            and isinstance(elasticity, (int, float))
            and math.isfinite(elasticity)
        ):
            details[str(row.get(PRODUCT_ID))] = (float(elasticity), str(source))
    return details


def load_elasticity_details() -> dict[str, tuple[float, str]]:
    """Products with usable elasticity -> ``(elasticity, source)`` from the
    persisted Part-4 state (dashboard consumption only, never recomputed)."""
    return dict(_elasticity_details())


@st.cache_data(show_spinner="Loading pricing-ready state…")
def load_pricing_ready_sources() -> frozenset[str]:
    """Product ids with a usable elasticity, from the persisted Part-4 state.

    Consumes the artifact only: no elasticity is recomputed, no split is
    opened, and no estimation logic is duplicated here. Absent/unreadable
    artifact degrades to an empty set (no prioritisation, labels off) so the
    catalog stays fully usable.
    """
    return frozenset(_elasticity_details())


@st.cache_data(show_spinner="Loading current prices…")
def load_week_prices(week: str) -> dict[str, float]:
    """Valid current price per product for ``week`` (``avg_price > 0``), keyed by
    raw product id.

    Mirrors the API's exact valid-price rule (``AppState.price_at``): a product
    has a current price iff the dense weekly demand matrix - the same source the
    deployed API consumes - has a row for that exact week with ``avg_price > 0``.

    Used for ordering, labels and the readiness summary only. The backend remains
    the authority on whether a price observation is usable; the UI never changes
    what the API considers a valid price.
    """
    frame = _load_matrix_cached()
    sub = frame[frame[WEEK].astype(str) == week]
    sub = sub[sub[PRICE].fillna(0) > 0]
    return dict(zip(sub[PRODUCT_ID].astype(str), sub[PRICE].astype(float)))


def load_price_available_for_week(week: str) -> frozenset[str]:
    """Product ids with a valid current price observation in ``week``."""
    return frozenset(load_week_prices(week))


def pricing_ready_for_week(week: str) -> frozenset[str]:
    """Products pricing-ready for ``week``: usable elasticity **and** a valid
    current price for that exact week."""
    return load_pricing_ready_sources() & load_price_available_for_week(week)


def pricing_tier_counts(week: str) -> tuple[int, int, int]:
    """``(tier1, tier2, tier3)`` headcounts for ``week``:
    ready+priced, ready only, and everything else."""
    ready = load_pricing_ready_sources()
    priced = load_price_available_for_week(week)
    return len(ready & priced), len(ready - priced), total_products() - len(ready)


def product_pricing_status(product_id: str, week: str) -> dict:
    """Dashboard-side pricing-readiness status for a product/week.

    Mirrors the deployed service decision order (``app.services.pricing_service``):
    a missing/invalid current price wins over a missing elasticity. Only real
    persisted values are reported; nothing is fabricated. When the readiness
    artifact itself is unavailable the status is
    ``STATUS_READINESS_UNAVAILABLE``, which is never confused with a product
    that genuinely has no elasticity.
    """
    prices = load_week_prices(week)
    price = prices.get(product_id)
    if pricing_readiness_state() != READINESS_AVAILABLE:
        return {
            "current_price": price,
            "elasticity": None,
            "elasticity_source": None,
            "status": STATUS_READINESS_UNAVAILABLE,
        }
    details = _elasticity_details().get(product_id)
    if price is None:
        elasticity, source = details if details else (None, None)
        return {
            "current_price": None,
            "elasticity": elasticity,
            "elasticity_source": source,
            "status": STATUS_NO_PRICE,
        }
    if details is None:
        return {
            "current_price": price,
            "elasticity": None,
            "elasticity_source": None,
            "status": STATUS_NO_ELASTICITY,
        }
    elasticity, source = details
    return {
        "current_price": price,
        "elasticity": elasticity,
        "elasticity_source": source,
        "status": STATUS_READY,
    }


@st.cache_data(show_spinner="Loading pricing catalogue…")
def pricing_options(query: str = "", week: str | None = None) -> tuple[list[str], int]:
    """Product ids for the Pricing tab: price-available products first.

    Identical matching rules to ``product_options`` (case-insensitive substring
    over the real product ids, ``_MAX_MATCHES`` cap, full total). When the
    selector is not narrowed by search, products are grouped by demo-actionability
    for the selected ``week`` (defaulting to the reference week):

    1. usable elasticity **and** a valid current price for the selected week,
    2. usable elasticity but no valid current price for the selected week,
    3. everything else.

    Ordering is presentation-only; no product is filtered, the full catalog stays
    reachable by typing, and the backend remains the final gate.
    """
    catalog = load_catalog()
    ids = catalog[PRODUCT_ID].astype(str)
    q = (query or "").strip().lower()
    if q:
        matched = ids[ids.str.lower().str.contains(q, regex=False)]
        return matched.head(_MAX_MATCHES).tolist(), int(len(matched))
    ready = load_pricing_ready_sources()
    priced = load_price_available_for_week(week or DEFAULT_WEEK)
    price_now = [pid for pid in ids if pid in ready and pid in priced]
    ready_no_price = [pid for pid in ids if pid in ready and pid not in priced]
    rest = [pid for pid in ids if pid not in ready]
    return (price_now + ready_no_price + rest)[:_MAX_MATCHES], int(len(ids))


def product_label(product_id: str) -> str:
    """Display label for a single product id (falls back to the raw id)."""
    return _catalog_index().get(product_id, product_id)


def pricing_product_label(product_id: str, week: str | None = None) -> str:
    """Pricing tab label: catalog label plus lightweight business metadata.

    Products with a usable elasticity are tagged ``Pricing-ready`` and, when a
    selected ``week`` is provided, their real persisted metadata is appended:
    ``Price available · $<price>`` (or ``No price this week``) plus
    ``Elasticity <value> · <source>``. The raw product id is always preserved
    first; only real values are shown.
    """
    base = product_label(product_id)
    details = _elasticity_details().get(product_id)
    if details is None:
        return base
    elasticity, source = details
    if week is None:
        return f"{base} {PRICING_READY_TAG}"
    prices = load_week_prices(week)
    if product_id in prices:
        extra = f"{PRICE_AVAILABLE_TAG} · ${prices[product_id]:.2f}"
    else:
        extra = NO_PRICE_TAG
    return (
        f"{base} {PRICING_READY_TAG}{extra} · "
        f"Elasticity {elasticity:.2f} · {elasticity_source_label(source)}"
    )


def is_real_product(product_id: str) -> bool:
    """True when ``product_id`` belongs to the authoritative catalog."""
    return product_id in _product_id_set()


def total_products() -> int:
    """Number of distinct servable products in the authoritative catalog."""
    return len(_product_id_set())


def week_options() -> list[str]:
    """Servable week-start labels, reference week listed first."""

    weeks = list(load_available_weeks())
    if DEFAULT_WEEK not in weeks:
        return weeks
    return [DEFAULT_WEEK] + [w for w in weeks if w != DEFAULT_WEEK]


class ProductSelector:
    """Reusable searchable product selector used by all three sections.

    Renders a search box plus a selectbox of real catalog products and returns
    the selected product id (or None when nothing is selected). The widgets are
    keyed by ``key`` so Forecast / Pricing / Explainability each keep their own
    independent widget state.

    ``pricing=True`` targets the Pricing tab: the un-narrowed list is ordered by
    demo-actionability for the currently selected week - price-available products
    first, then usable-elasticity products without a current price, then the rest
    - and products get a lightweight status tag. No product is excluded; the full
    catalog remains selectable by typing.
    """

    def __init__(
        self,
        key: str,
        *,
        label: str = "Product",
        pricing: bool = False,
    ) -> None:
        self.key = key
        self.label = label
        self.pricing = pricing

    def render(self, week: str | None = None) -> str | None:
        search = st.text_input(
            "Search product",
            key=f"{self.key}:search",
            placeholder="Type part of a Product ID to filter…",
            help="Filters the selectable products below by product id.",
        )
        if self.pricing:
            ids, total = pricing_options(search, week=week)
            fmt = partial(pricing_product_label, week=week)
        else:
            ids, total = product_options(search)
            fmt = product_label
        if not ids:
            st.caption("No product matches your search.")
            return None
        if total > _MAX_MATCHES:
            st.caption(
                f"{total} products match - showing the first {_MAX_MATCHES}; "
                "keep typing to narrow down."
            )
        elif not (search or "").strip():
            st.caption(f"All {total_products()} products are selectable - type to search.")
        selected = st.selectbox(
            self.label,
            options=ids,
            index=None,
            key=f"{self.key}:select",
            format_func=fmt,
            placeholder="Search / select a product",
            help="A real product from the catalog; sent to the API unchanged.",
        )
        if selected is not None and not is_real_product(selected):
            return None
        return selected


def week_selector(*, key: str, label: str = "Week (ISO Monday)") -> str:
    """Servable week dropdown. Every option is a real, supported Monday on or
    before the reference week, so selection always satisfies the API's week
    validation; the backend remains the final validation gate.
    """

    options = week_options()
    index = options.index(DEFAULT_WEEK) if DEFAULT_WEEK in options else 0
    return st.selectbox(
        label,
        options=options,
        index=index,
        key=f"{key}:week",
        format_func=lambda w: f"{w} (reference)" if w == DEFAULT_WEEK else w,
        help="A supported Monday before the protected test boundary.",
    )