# Pricing Selector Demo-Ready (Week-Aware) Prioritisation — Implementation Report

Presentation-only change: the Pricing product selector now surfaces, at the top,
products that will run a full recommendation for the currently selected week
(usable elasticity **and** a valid current price for that week), then products
with usable elasticity but no current price that week, then the rest. No product
is hidden; the full 32,729-product catalog remains reachable via search.

## Files changed (Dashboard layer only)

- `dashboard/product_catalog.py` — new `load_price_available_for_week(week)`
  (mirrors the backend's exact valid-price rule); `pricing_options(query, week)`
  now builds the three tiers for the selected week; `pricing_product_label`
  gained week-aware tags; `ProductSelector.render(week)` threads the selected
  week into ordering and labels.
- `dashboard/app.py` — `run_controls` reads the week widget before rendering the
  selector and passes `week=` to it. Visual layout unchanged (product left, week
  right).
- `tests/dashboard/_probe_app.py` — mirrors the same wiring for AppTest.
- `tests/dashboard/test_product_catalog.py` — updated 2 tests, added 8.
- `tests/dashboard/test_product_selector_app.py` — added 3 AppTest scenarios.

## UX before vs after

| | Before | After |
|---|---|---|
| Default ordering | products with persisted elasticity first, then everyone else (catalog order) | 1) elasticity + valid current price for the selected week → 2) elasticity, no current price that week → 3) everyone else |
| Week dependence | none (ordering fixed) | ordering and tags follow the selected week (flip weeks → flip order) |
| Labels | `· Pricing-ready` only | `· Pricing-ready · Price available` (tier 1), `· Pricing-ready · No price this week` (tier 2), untagged (tier 3) |
| Search | substring filter, unchanged | unchanged (search results ignore the week ordering; identical to Forecast search) |
| Catalog reach | all products by search | all 32,729 products still reachable by search |

`2018-05-14` remains the default reference week (unchanged, `dashboard/config.py`).

## How price-availability is determined

`load_price_available_for_week(week)` reads only the existing dense weekly demand
matrix (the same file the deployment API consumes) and a product is
price-available when the matrix has a row for that exact week with
`avg_price > 0` — byte-for-byte the rule in `app/services/state.py::price_at`.
No elasticity is recomputed; the persisted Part-4 `elasticity.effective` set is
consumed unchanged via `load_pricing_ready_sources()`.

## Tests added / updated

`tests/dashboard` (`test_product_catalog.py`):
1. `test_pricing_price_available_products_lead_over_elasticity_only` — tier-1
   (elasticity + price) is ordered before tier-2 (elasticity, no price).
2. `test_elasticity_without_current_price_remains_selectable` — tier-2 products
   stay in the un-narrowed list and are reachable by typing.
3. `test_without_elasticity_remains_selectable_via_search` — no-elasticity
   product stays searchable.
4. `test_pricing_week_change_reorders_price_availability` — ordering equals the
   data-driven tiers for two different weeks and differs between weeks.
5. `test_no_product_ids_hardcoded_in_ordering_logic` — pre-existing, still passes
   (ordering is fully data-driven).
6. `test_forecast_options_unrestricted_by_elasticity` / `test_product_label_never_tagged_by_default` —
   pre-existing, Forecast label/list unchanged.
7. Explainability uses the same shared selector with `pricing=False`; unchanged.
8. `test_pricing_options_search_cap_and_filter_match_product_options` — search
   results identical to Forecast search and independent of the week (payload as
   sent unchanged).
9. `test_pricing_label_*` — tier-1/tier-2 tags are business-facing, no internal
   ML terms (`lightgbm`, `one-hot`, `lags`, `regime`, `93 feature`,
   `elasticity` never appear in labels).
10. `test_pricing_ready_label_absent_for_no_elasticity_product` — NO_ELASTICITY
    product stays untagged.

`tests/dashboard` (`test_product_selector_app.py`, AppTest against the probe):
- `test_pricing_tab_price_available_products_lead_by_default`
- `test_pricing_week_change_reorders_and_forwards_week`
- `test_pricing_no_price_observation_product_can_still_run_and_forward`

## Test results

- `tests/dashboard`: **51 passed** (was 41).
- Full suite `pytest tests -q`: **384 passed, 1 failed, 9 skipped**. The single
  failure is the pre-existing environmental
  `tests/test_posthoc_holdout.py::test_marker_guard_prevents_second_execution`
  (missing once-guard marker under the temp experiments dir), unrelated to this
  change.

## Validation performed

- First default Pricing options are all tier-1: verified the first 3 are
  `· Pricing-ready · Price available`; tier-1 size at `2018-05-14` = 468.
- `00066f42aeeb9f3007548bb9d3f33c38` (NO_ELASTICITY): still reachable by search,
  untagged, not classified pricing-ready (persisted-state says absent).
- `0009406fd7479715e4bef61dd91f2462`: still selectable; replayed the exact
  backend rule read-only → no price at `2018-05-14` (backend returns
  `NO_PRICE_OBSERVATION`), labelled `· Pricing-ready · No price this week`.
- `NO_PRICE_OBSERVATION` / `NO_ELASTICITY` semantics are untouched: the dashboard
  only reorders/labels; validity is decided by the backend as before.

## Confirmations

- **No model, Part-3/Part-4 artifact, pricing engine, optimizer, guardrail,
  FastAPI endpoint/schema, feature code, data file, MLflow registry entry, DB
  schema, or deployment file was modified.** Only the files listed above
  (Dashboard + dashboard tests) were touched.
- **Full catalog remains selectable.** Ordering is prioritisation only; search
  still reaches any of the 32,729 products and the raw `product_id` is forwarded
  unchanged.

## Remaining UX caveat

The tier-1 list is bounded by real data: at the default reference week only 468
of 17,012 elastic products also have a current price, so the "Price available"
block is short and the dropdown quickly moves into `· No price this week`.
Some earlier servable weeks (e.g. 2016) have no price-available products at all,
so there the selector degrades to the previous elasticity-first ordering. This is
an inherent property of the data, not a selector defect; no week default or data
was changed to mask it.