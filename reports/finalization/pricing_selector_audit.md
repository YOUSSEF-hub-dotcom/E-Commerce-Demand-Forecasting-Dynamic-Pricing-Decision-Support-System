# Pricing Product Selector — Read-Only Audit

## VERDICT: PASS (with a documented UX GAP)

The selector **correctly prioritizes the 17,012 pricing-ready products** exactly as
implemented and specified. The observed `NO_PRICE_OBSERVATION` behavior is an
expected consequence of **pricing-ready ≠ current-price-ready**, a distinction the
design treats as two separate concepts. No ordering bug was found. One demo-usability
gap is noted (see Recommended next step). No code, data, model, artifact, or
configuration was changed.

## 1. How `pricing_options()` determines order

`dashboard/product_catalog.py:155` — with an empty query the list is built as
`premium (ready, catalog order) + rest (not ready, catalog order)`, capped at 5,000.
With a search query it returns matches in plain catalog order (deliberately unchanged).

## 2. How the pricing-ready set is determined

`dashboard/product_catalog.py:124` `load_pricing_ready_sources()` reads the persisted
Part-4 state `reports/forecasting/part4_pricing_engine.json` →
`elasticity.effective` and keeps products with `elasticity_source` in
(`own`, `category_pooled`) **and** finite elasticity.

## 3. Verified: first options come from the persisted set

- Persisted `effective` rows: 19,099; non-null elasticity: 17,012.
- Selector pricing-ready set: **17,012** — exactly the startup-log/effective count.
- `ready ⊆ effective product ids` — **True**.
- First 10 displayed options: **10/10 are priced-ready** (`ready=True`, tagged
  `· Pricing-ready`).

## 4. Product `00066f42aeeb9f3007548bb9d3f33c38`

- **NOT pricing-ready**: absent from `elasticity.effective` entirely
  (zero TRAIN sales → no own estimate, no category-pooled eligibility under Part-4).
- It **should NOT appear among the first options**. It lives in the `rest` group and
  is only reachable by typing. Its `NO_ELASTICITY` at 2018-05-14 is correct (it does
  have a price that week, but no elasticity).

## 5. Product `0009406fd7479715e4bef61dd91f2462`

- **Has effective elasticity**: `status=Unsupported`, `source=category_pooled`,
  `elasticity=-0.4186`, category `cama_mesa_banho` — so it **is** pricing-ready
  (2nd option displayed).
- **Lacks a price at 2018-05-14**: 1 matrix row at that week but `avg_price` ≤ 0 →
  `NO_PRICE_OBSERVATION`. It has prices in *other* weeks. This confirms the two
  concepts are independent.

## 6. First-10 comparison

| # | product_id | pricing-ready | price at 2018-05-14 |
|---|---|---|---|
| 1 | `00088930…` | ✓ | ✗ |
| 2 | `0009406f…` | ✓ | ✗ |
| 3 | `0011c512…` | ✓ | ✗ |
| … | (… all ready …) | ✓ | ✗ |
| 10 | `0021a87d…` | ✓ | ✗ |

## 7. Can a non-pricing-ready product appear before pricing-ready ones?

- **Default (empty search) view: No.** `premium` holds all 17,012 ready products, and
  the 5,000-row cap is less than that, so every displayed option is pricing-ready;
  `rest` never surfaces. Non-ready products appear only after the ready block.
- **Search: Yes, by design.** A search returns matches in catalog order (not
  re-ordered), so a ready and a non-ready match can interleave. This is the specified
  "search returns normally" behavior, not a bug.

## Root cause

No ordering defect. The gap is conceptual: of the **17,012 pricing-ready**
products, only **468 also have a price at the default week 2018-05-14**
(2.75%). Of the 5,000 displayed, only **133** are price-ready that week, and the
first ones shown (catalog order within `premium`) mostly lack that week's price —
hence the immediate `NO_PRICE_OBSERVATION` on the first products a demo user clicks.

## Recommended next step (no implementation)

- Keep elasticity-based prioritization as-is (PASS).
- Optionally, add a secondary demo-aware **sort within pricing-ready products** by
  "has a price at the currently selected week" (falling back to catalog order), or a
  small **second tag** ("no price at selected week") — both purely presentational,
  keeping elasticity logic, artifacts, and the API untouched. Do not filter products
  out; keep the full catalog selectable.