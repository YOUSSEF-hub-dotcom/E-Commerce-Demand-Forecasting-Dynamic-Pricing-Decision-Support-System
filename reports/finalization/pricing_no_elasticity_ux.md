# UX Improvement for NO_ELASTICITY Pricing Response — Final Report

## Verdict context (unchanged)

Diagnostic audit concluded **A. EXPECTED_NO_ELASTICITY**: product
`00066f42aeeb9f3007548bb9d3f33c38` has zero TRAIN sales and therefore no usable
elasticity. No business logic was touched; this task was strictly a
presentation/UX change in the Streamlit Pricing tab.

## Objective

When the `/pricing` API returns `reason == "NO_ELASTICITY"`, the dashboard now
renders a clear business-facing explanation instead of the bare `Recommended
price: None` + `Elasticity: n/a` + `Status: n/a` block.

## Files changed

- `dashboard/pricing_presenter.py` — **new** presenter. Maps the API `reason`
  to UI copy. `NO_ELASTICITY` renders the business message; every other reason
  (`optimized`, `PROFIT_NOT_COMPUTABLE (no cost supplied)`,
  `NO_PRICE_OBSERVATION`) flows through the pre-existing render block,
  byte-for-byte unchanged.
- `dashboard/app.py` — Pricing tab now delegates rendering to
  `render_pricing_response(res)` (single call replaces the previous inline
  block). API request payload and response consumption are untouched.
- `tests/dashboard/test_pricing_presenter.py` — **new** UI tests.
- `tests/dashboard/_pricing_presenter_probe.py` — **new** AppTest probe that
  renders a raw `/pricing` response fixture through the exact deployed
  presenter.

## Exact UI behavior

### Before (NO_ELASTICITY)
- Green success banner: `Recommended price: **None**`
- Metrics: `Current price 101.65`, `Baseline demand 1.156929...`,
  `Elasticity n/a`, `Status n/a`
- Info line: `Reason: NO_ELASTICITY · objective=n/a · optimal price=n/a ·
  profit_computable=False`

### After (NO_ELASTICITY)
- Amber warning banner: **"Price recommendation unavailable"**
- Body: "This product does not have sufficient historical price-demand
  variation to estimate price elasticity reliably."
- Body: "Profit optimization was not performed because elasticity is
  unavailable."
- Collapsed expander "Technical diagnostic details" retains the exact
  diagnostic values (Current price, Baseline demand, Elasticity n/a,
  Status n/a, technical Reason/objective/optimal/profit_computable info line,
  model + inference caption).
- **No price is shown, no success banner, no fabricated/fake recommendation.**

### Successful / other reasons (unchanged)
- `optimized`: green banner `Recommended price: **<price>**`, 4 metrics,
  guardrails caption, scenarios table — unchanged.
- `PROFIT_NOT_COMPUTABLE (no cost supplied)` and `NO_PRICE_OBSERVATION`:
  identical rendering to before.

## Tests passed

- `tests/dashboard` — **29 passed** (7 new presenter/UI tests + 22 existing
  selector/catalog tests) in 74s.
- Full suite `pytest tests -q` — **362 passed, 1 failed, 9 skipped**. The single
  failure is the pre-existing, environmental
  `tests/test_posthoc_holdout.py::test_marker_guard_prevents_second_execution`
  (missing once-guard marker file under the temp experiments dir), identical to
  the state before this change and unrelated to dashboard code.

## Confirmations

- **Pricing/elasticity/model logic untouched**: only `dashboard/` UI files
  changed. No changes to Part 3 model, MLflow registry, elasticity methodology,
  `_per_product_elasticities`, category-pooled fallback, `effective_elasticity`,
  optimizer, `Profit(P) = (P - Cost) * PredictedDemand(P)`, ±20% simulation
  range, cost floor, smoothing guardrails, API request/response schema,
  FastAPI pricing service, `AppState`, SHAP, or the product selector.
- **Selected product still returns NO_ELASTICITY**: the dashboard reads the
  same `/pricing` response; nothing changed its inputs, so it still returns
  `reason=NO_ELASTICITY`, `recommended_price=None`, `profit_computable=False`.
  A full `AppState` reproduction from the audit is unchanged
  (elasticity/status/source all `None`).
- **No price is fabricated**: the presenter renders only what the API returns
  and has no pricing computation. The NO_ELASTICITY branch emits no
  recommendation at all.
- **No fallback added** for zero-TRAIN-sales products; no invented elasticity.