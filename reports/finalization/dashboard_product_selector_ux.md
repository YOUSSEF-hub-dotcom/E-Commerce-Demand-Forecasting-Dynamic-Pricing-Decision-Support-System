# Dashboard UX: Reusable Product Selector — Final Report

**Task.** Upgrade the Streamlit manager dashboard so managers **never manually
type a Product ID**. Product selection across Forecast, Pricing, and
Explainability now goes through one reusable, searchable Product Selector
backed by the authoritative product catalog. The FastAPI contracts, model,
registry, feature engineering, pricing/SHAP logic, validation rules, error
codes, week rules, DB schema, and Docker setup are unchanged.

## Files changed

| File | Change |
| --- | --- |
| `dashboard/product_catalog.py` | **New.** Authoritative catalog + reusable `ProductSelector` + `week_selector`. |
| `dashboard/app.py` | Replace the manual `product_id`/`week` text inputs (old `product_week_form`) with the shared selector via `run_controls`. Forecast / Pricing / Explainability each instantiate `ProductSelector` with an independent key. |
| `tests/dashboard/test_product_catalog.py` | **New.** Catalog-source and filtering unit tests. |
| `tests/dashboard/test_product_selector_app.py` | **New.** `AppTest` integration tests that the section wiring forwards the real selected product. |
| `tests/dashboard/_probe_app.py` | **New.** Minimal AppTest fixture mirroring the dashboard section wiring (used by the integration tests). |
| `tests/dashboard/__init__.py` | **New.** Test package marker. |

No FastAPI, model, registry, feature-engineering, pricing/SHAP, validation,
error-code, week-rule, DB-schema, or Docker files were touched.

## Product source (authoritative)

The catalog is loaded from the same dense weekly demand matrix the deployment
API consumes: `data/processed/demand_matrix_weekly.csv.gz`, via the existing
`src/optimization/pricing_data.load_matrix()`. This is the identical source the
API uses to build its own product set, so the selector can never offer a
product the inference pipeline cannot serve.

- **32,729 distinct product IDs**, all full 32-hex Olist ids. No synthetic,
  numeric, or fake IDs anywhere.
- Products are restricted to **servable weeks** (weeks on/before the reference
  week `2018-05-14`, i.e. strictly before the protected test boundary) — every
  product a manager can pick is one the API can actually serve.
- **90 servable week options**, all ISO Mondays on/before the reference week.
  The reference week is presented first by default.
- The catalog is loaded **once** via `st.cache_data` (plus a module-level
  matrix cache); no Postgres dependency, no per-interaction reloads, no test
  split access.

## How the selector is shared

`dashboard/product_catalog.py::ProductSelector` renders:
1. a **search text box** (substring, case-insensitive, over real product IDs),
2. a **selectbox** of matching products (display label = real ID, augmented
   with the product category when it exists), capped at 5,000 entries with a
   "keep typing to narrow" caption when the match set is larger,
3. a **week selector** dropdown of servable Mondays defaulting to
   `2018-05-14`.

All three sections (`Forecast`, `Pricing`, `Explainability`) instantiate the
same component with independent widget keys (`forecast:*`, `pricing:*`,
`explain:*`) so each section keeps its own selection. The section wiring lives
in `dashboard/app.py::run_controls`.

A widget returns `None` until the manager picks a real product, and the value
returned is always a raw authoritative product ID from the catalog — forwarded
to the API exactly as stored. There is no free-text product input left
anywhere in the UI.

## Contracts unchanged

- `ApiClient` requests are byte-for-byte the same: `predict(product_id, week)`,
  `pricing(product_id, week, cost)`, `explain(product_id, week)` where
  `product_id` is the real catalog ID and `week` a servable Monday (or None is
  no longer reachable — the week selector always supplies a value).
- Backend validation remains the final gate; the week dropdown only offers
  valid supported Mondays, so no invalid week can be submitted from the UI.
- `PRODUCT_NOT_FOUND` / other API errors are still surfaced through the
  existing `render_error` path (backend error codes untouched).
- Optional unit cost in Pricing is preserved (cost > 0 is forwarded, otherwise
  None = revenue-optimal / `PROFIT_NOT_COMPUTABLE`).

## Widget rules enforced

- Only real, authoritative product IDs appear; never hardcoded, never numeric,
  never test-split products.
- Selected value is sent unchanged; `format_func` only changes the **display**
  label (adds the category), never the payload value.
- `st.cache_data` used for catalog + week loading; catalog loads once.

## Testing

**New dashboard tests (22 test functions):**

- `tests/dashboard/test_product_catalog.py` (14 tests): catalog contains only
  unique 32-hex real IDs; never contains fake/numeric IDs; catalog equals the
  matrix product set and is restricted to servable weeks; search filtering
  (empty / substring / case-insensitive / no-match); label fallback for unknown
  IDs; `is_real_product` membership; available weeks are all ISO Mondays before
  the protected boundary and the default week is served first.
- `tests/dashboard/test_product_selector_app.py` (AppTest, 8 tests): every
  section renders the shared search box and a product list of **real** catalog
  products; typing narrows; Forecast, Pricing (with optional cost), and
  Explainability all forward the exact real selected product ID + default week;
  a no-match search shows guidance and removes the selectbox; no fake ID can
  appear in the forwarded line.

**Executed:**

| Command | Result |
| --- | --- |
| `pytest tests/dashboard -q` | **22 passed** |
| `pytest tests/dashboard tests/deployment/test_deployment_safety.py -q` | **28 passed** |
| `pytest tests -q` (full suite) | **355 passed, 1 failed, 1 skipped** |

The single failure is `tests/test_posthoc_holdout.py::test_marker_guard_prevents_second_execution`, a **pre-existing environment dependency**: the exactly-once guard marker at `C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments\part3_v2_posthoc_holdout_marker.json` was absent in this environment (temp dir cleaned since the one-time holdout execution). It has no relation to the dashboard work; it imports only `src.models.posthoc_holdout`. The marker is intentionally not regenerated (doing so would fabricate the exactly-once evidence).

Also verified separately: the real `dashboard/app.py` renders all three sections plus `System status` without exceptions under `streamlit.testing.v1.AppTest`.

## Limitations

- Display of product categories is sparse because the authoritative matrix has
  a category value for only 2 servable products; the label correctly falls back
  to just the product ID elsewhere. Production inference relies on the same
  matrix, so behavior matches the served model.
- The selectbox caps at 5,000 matches for dropdown responsiveness; the full
  catalog remains reachable by typing more characters (caption guides this).
- Minor cosmetic Streamlit quirk observed only in bare-mode console printing of
  the placeholder ellipsis; the rendered UI is unaffected.