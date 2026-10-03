# Pricing Product Selector Demo Usability — Final Report

UX-only change: the Pricing tab now surfaces pricing-ready products first while
leaving the full catalog, Forecast, and Explainability untouched. No business,
pricing, or elasticity logic changed.

## Files changed

- `dashboard/product_catalog.py` — added:
  - `load_pricing_ready_sources()`: reads the **existing persisted Part-4
    elasticity state** (`reports/forecasting/part4_pricing_engine.json` →
    `elasticity.effective`). Consumes the artifact only — no recomputation, no
    estimation algorithm, no split opened. Products with a finite elasticity
    and source in (`own`, `category_pooled`) are pricing-ready. Missing /
    unreadable artifact degrades to an empty set (no reordering, no tags).
  - `pricing_options(query)` — Pricing-tab option source: same substring match,
    `_MAX_MATCHES` cap, and full total as `product_options`, but when the
    selector is not narrowed by search the list is ordered **pricing-ready
    products first, then the rest** (deterministic catalog order within each
    group). Search results are returned exactly as normal.
  - `pricing_product_label(pid)` — catalog label with a business-facing
    `· Pricing-ready` tag for pricing-ready products (no internal detail).
  - `ProductSelector(..., pricing=False)` — shared component flag; Pricing
    passes `pricing=True`, Forecast/Explain default unchanged.
- `dashboard/app.py` — Pricing tab only: `ProductSelector("pricing",
  pricing=True)`. Forecast and Explainability selectors untouched.
- `tests/dashboard/test_product_catalog.py` — 9 new tests.
- `tests/dashboard/test_product_selector_app.py` — 2 new AppTest UI tests;
  `_probe_app.py` mirrors the Pricing `pricing=True` wiring.

## How pricing-ready products are identified

From the authoritative, already-persisted Part-4 elasticity state
`elasticity.effective` in `reports/forecasting/part4_pricing_engine.json` —
the same table the deployment API loads at startup. A product is **pricing-ready**
when it has a finite elasticity with source `own` or `category_pooled`
(1,538 + 15,474 = 17,012 products at the reference week). This is pure
consumption: no elasticity is recomputed in the dashboard, no test split is
opened, and `AppState` / `pricing_service` / `elasticity.py` are untouched.

## Tests passed

- `tests/dashboard` — **41 passed** (was 29: +9 catalog, +2 AppTest; probe
  updated).
- Full suite `pytest tests -q` — **374 passed, 1 failed, 9 skipped**. The single
  failure is the same pre-existing environmental
  `tests/test_posthoc_holdout.py::test_marker_guard_prevents_second_execution`
  (missing once-guard marker under the temp experiments dir), unrelated to this
  change.

Each requested scenario is covered by a test:

1. **Pricing-ready prioritized** — `test_pricing_options_prioritise_pricing_ready_products`,
   `test_pricing_ordering_is_data_driven_deterministic`,
   `test_pricing_tab_leads_with_pricing_ready_products` (Pricing dropdown
   opens with `· Pricing-ready` options before any untagged product).
2. **Without-elasticity products remain accessible** — `test_pricing_options_keep_no_elasticity_products_reachable`,
   `test_pricing_tab_no_elasticity_product_reachable_and_forwarded_unchanged`
   (the NO_ELASTICITY product is reachable by search and can still be run).
3. **No product IDs hard-coded** — `test_no_product_ids_hardcoded_in_ordering_logic`
   (real ids must not appear in module source).
4. **Search still works** — `test_pricing_options_search_cap_and_filter_match_product_options`
   (pricing search results identical to normal search, cap preserved).
5. **Raw product_id unchanged** — `test_pricing_tab_no_elasticity_product_reachable_and_forwarded_unchanged`
   (forwarded `product_id` equals the selected real id, unchanged).
6. **Forecast unrestricted by elasticity** — `test_forecast_options_unrestricted_by_elasticity`,
   `test_product_label_never_tagged_by_default`.
7. **NO_ELASTICITY UX unchanged** — existing presenter tests
   (`test_pricing_presenter.py`) still pass untouched.
8. **Successful pricing UX unchanged** — existing presenter tests
   (`test_optimized_response_unchanged`, etc.) still pass untouched.

## Confirmations

- **No hard-coded product IDs added.** Ordering is computed from the persisted
  Part-4 elasticity state; the catalog remains the dense weekly matrix.
- **All products remain accessible.** The full catalog (`total_products()` =
  32,729) is still returned, just ordered (pricing-ready first under the
  always-applied 5,000 dropdown cap). The example NO_ELASTICITY product
  `00066f42aeeb9f3007548bb9d3f33c38` is reachable by typing and still returns
  `NO_ELASTICITY` with no recommendation (presenter unchanged).
- **NO_ELASTICITY behavior unchanged.** No elasticity was invented, no category
  fallback added, and the pricing/optimization path is untouched.
- **Nothing outside the dashboard presentation layer was modified.** No changes
  to Part 3, Part 4 elasticity methodology, `pricing_service`, optimizer,
  guardrails, API contract, MLflow, model, SHAP, data, test split, or
  deployment architecture.