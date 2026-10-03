# Pricing Manager-Ready UX — Implementation Report

Presentation-only change to the **Pricing** section so a business manager can,
in one flow, see how many products are actually actionable for the selected week,
pick one, and understand its readiness before requesting a recommendation. The
Forecast and Explainability sections are untouched. No product is hidden: the
full 32,729-product catalog remains reachable via search.

## Files changed (Dashboard layer only)

- `dashboard/product_catalog.py`
  - `_elasticity_details()` / `load_elasticity_details()` — single cached source
    of `product_id -> (elasticity, source)` from the persisted Part-4 state;
    `load_pricing_ready_sources()` now derives from it (same frozenset contract).
  - `load_week_prices(week)` — cached `product_id -> avg_price` for the exact week
    (`avg_price > 0`), mirroring `AppState.price_at`; `load_price_available_for_week`
    is now a thin wrapper over it.
  - `pricing_ready_for_week(week)`, `pricing_tier_counts(week)` — data-driven
    tier-1/tier-2/tier-3 headcounts.
  - `product_pricing_status(product_id, week)` — dashboard-side readiness status
    mirroring the backend decision order (missing price wins over missing
    elasticity); reports only real persisted values.
  - `elasticity_source_label(source)` — business label (`Own` / `Category` / `n/a`).
  - Enriched `pricing_product_label(product_id, week)` — real metadata appended to
    tier-1/tier-2 labels (`· Price available · $<price> · Elasticity <v> · <source>`
    or `· No price this week · Elasticity <v> · <source>`); tier-3 labels and the
    week-less label are unchanged in spirit.
  - Removed the now-unused `_pricing_ready_set()` helper.
- `dashboard/pricing_presenter.py` — added `render_pricing_readiness(product_id, week)`
  and the `NO_PRICE_OBSERVATION` business copy (`NO_PRICE_TITLE`,
  `NO_PRICE_EXPLANATION`, `NO_PRICE_PROFIT`); reuses the existing
  `NO_ELASTICITY_*` copy. `render_pricing_response` is unchanged.
- `dashboard/app.py` — new dedicated `render_pricing_section(client)`: week first,
  then a dynamic "Pricing opportunities for this week" summary, the selector, the
  selected-product summary, cost entry and Run. The Pricing branch calls it;
  Forecast and Explainability keep the shared `run_controls`.
- `tests/dashboard/_probe_app.py` — mirrors the new Pricing flow (summary +
  `PRICING_OPPORTUNITIES:<t1>:<t2>:<t3>` machine-readable line + selected-product
  summary) and keeps the `PRICING:<product>:<week>:<cost>` forwarding line.
- `tests/dashboard/test_product_catalog.py` — updated 1 test, added 10.
- `tests/dashboard/test_product_selector_app.py` — updated 3 tests, added 6.

## UX before vs after (Pricing only)

| | Before | After |
|---|---|---|
| Flow | week + product together, then cost + Run | week first, then "Pricing opportunities for this week" summary, then product, then selected-product summary, then cost + Run |
| Eligibility headline | none | `**N products have both a valid current price and usable elasticity for this week.**` (N computed live, never hard-coded) |
| Empty week | silently reordered | clear business message: "No products currently have both usable elasticity and a valid current price for this week." |
| Labels | `· Pricing-ready · Price available` / `· No price this week` | adds the product's real `$price`, `Elasticity <v>` and source (`Own`/`Category`) |
| Selected product | nothing until Run | summary with current price, elasticity, source, status — or the exact reason it is not ready |
| Ordering | 3 tiers (week-aware) | unchanged 3 tiers (week-aware) |
| Catalog reach | all products by search | all products by search |

`2018-05-14` remains the default reference week (unchanged, `dashboard/config.py`).

## How readiness is determined

- **Usable elasticity**: consumed unchanged from the persisted Part-4
  `elasticity.effective` table (`reports/forecasting/part4_pricing_engine.json`);
  nothing is recomputed.
- **Valid current price**: the product must have `avg_price > 0` for the exact
  selected week in the dense weekly demand matrix — byte-for-byte the rule in
  `app/services/state.py::price_at`.
- **Status reason**: `product_pricing_status` mirrors the deployed
  `app.services.pricing_service` order — missing price → `NO_PRICE_OBSERVATION`
  (even if elasticity exists); otherwise missing elasticity → `NO_ELASTICITY`;
  otherwise `READY`.

The backend remains the final gate: the dashboard only orders, labels and
summarises; it never changes what the API accepts or returns.

## Tests added / updated

`tests/dashboard/test_product_catalog.py` (added):
1. `test_elasticity_source_labels_are_business_facing`
2. `test_tier1_label_embeds_real_metadata`
3. `test_tier2_label_shows_real_elasticity_and_no_price`
4. `test_elasticity_details_consume_only_persisted_part4_state`
5. `test_product_pricing_status_ready`
6. `test_product_pricing_status_no_price_carries_elasticity_when_available`
7. `test_product_pricing_status_no_elasticity_carries_price_when_available`
8. `test_pricing_tier_counts_match_expected_tiers_per_week`
9. `test_pricing_ready_for_week_is_tier1_set`
10. `test_week_prices_exclude_zero_prices_exactly_like_backend`

`tests/dashboard/test_product_catalog.py` (updated):
- `test_price_labels_never_leak_ml_internals` — dropped `elasticity` from the
  forbidden-term list. The new manager-facing metadata intentionally shows
  `Elasticity -1.12 · Own`; the remaining ML-internal terms (`lightgbm`,
  `one-hot`, `lags`, `regime`, `93 feature`) are still asserted absent.

`tests/dashboard/test_product_selector_app.py` (added, AppTest against the probe):
- `test_pricing_opportunities_count_is_dynamic_and_matches_tier_counts`
- `test_pricing_opportunities_count_changes_with_selected_week`
- `test_pricing_week_with_no_price_eligible_products_shows_business_message`
- `test_pricing_selected_ready_product_shows_real_summary`
- `test_pricing_selected_no_price_product_shows_persisted_values_and_reason`
- `test_pricing_selected_no_elasticity_product_shows_price_and_reason`

`tests/dashboard/test_product_selector_app.py` (updated):
- `test_pricing_tab_no_elasticity_product_reachable_and_forwarded_unchanged` and
  `test_pricing_no_price_observation_product_can_still_run_and_forward` — one
  `run()` added after selecting the product, since cost entry now appears after
  a product is chosen. All original assertions retained.
- `test_pricing_week_change_reorders_and_forwards_week` — unchanged.

## Test results

- `tests/dashboard`: **67 passed** (was 51).
- Full suite `pytest tests -q`: **400 passed, 1 failed, 9 skipped**. The single
  failure is the pre-existing environmental
  `tests/test_posthoc_holdout.py::test_marker_guard_prevents_second_execution`
  (missing once-guard marker under the temp experiments dir), unrelated to this
  change.

## Validation performed

- Week-first flow renders before the selector; the opportunities sentence uses the
  live count (`pricing_tier_counts`), not a literal.
- Default week `2018-05-14`: tiers = `(468, 16544, 15717)`, total `32729`; the
  first three options are tier-1, e.g.
  `005030ef108f58b46b78116f754d8d38 — cama_mesa_banho · Pricing-ready · Price available · $13.99 · Elasticity -0.97 · Category`.
- `00066f42aeeb9f3007548bb9d3f33c38` (NO_ELASTICITY): untagged, reachable by
  search; selected-product summary shows the real current price `$101.65`,
  `Elasticity n/a`, `Elasticity source n/a`, plus the elasticity-unavailable
  explanation.
- `0009406fd7479715e4bef61dd91f2462` (ready, no price at the default week):
  labelled `· Pricing-ready · No price this week · Elasticity -0.42 · Category`;
  summary shows `Current price n/a`, real elasticity/source and the
  no-current-price explanation.
- Week change reorders the options and updates the opportunities count
  (verified against a different week and against a zero-tier-1 week such as
  `2016-08-29`, which shows the informational message).
- No hard-coded product ids or counts in the modules or tests; all values derive
  from the persisted Part-4 state and the weekly matrix.

## Confirmations

- **No model, Part-3/Part-4 artifact, pricing engine, optimizer, guardrail,
  FastAPI endpoint/schema, feature code, data file, MLflow registry entry, DB
  schema, or deployment file was modified.** Only the Dashboard files and
  dashboard tests listed above were touched.
- **Forecast and Explainability are unchanged** (shared `ProductSelector` with
  `pricing=False`; `run_controls` untouched; same widget keys).
- **Cost and Run behavior are unchanged** apart from rendering after a product is
  selected; the API call signature and payload are identical and the raw
  `product_id` is forwarded unchanged.

## Data caveat (not a defect)

Tier-1 is bounded by real data: at the default reference week only 468 of the
17,012 elastic products also have a valid current price, and 14 servable weeks
(some in 2016) have zero such products. In those weeks the section shows the
explicit business message and the selector falls back to tier-2/tier-3. The count
is always computed live; no week default, threshold or data was added to mask this.
