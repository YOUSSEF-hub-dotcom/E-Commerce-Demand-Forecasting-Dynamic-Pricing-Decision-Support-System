# Elasticity Diagnostic Audit — Product `00066f42…` NO_ELASTICITY

**Status: DIAGNOSTIC ONLY — no code, model, or config changed.**

## Verdict

**A. EXPECTED_NO_ELASTICITY** — the selected product genuinely has **no usable
elasticity** under the existing Part-4 methodology. The deployed pricing
service retrieves the (absent) elasticity correctly and correctly refuses to
invent or optimize. This is **expected, consistent behavior**, matching the
offline Part-4 artifact.

## 1. Request trace (deployed code path, reproduced via full AppState load)

`POST /pricing {product_id=00066f42aeeb9f3007548bb9d3f33c38, week=2018-05-14, cost=10.0}`

Path: `app/api/pricing.py` → `pricing_row()` (`app/services/pricing_service.py:29`) →
`state.elasticity(pid)` / `elasticity_status_of` / `elasticity_source_of`
(`app/services/state.py:305`).

Reproduced with a full `AppState.load_all()` run (same model, same matrix, same
`elasticity_mod.estimate`):

```
selected product present in catalog:     True
state.category_of[pid]:                  'nan'     (not 'perfumaria')
state.elasticity(pid):                   None
state.elasticity_status_of(pid):         None
state.elasticity_source_of(pid):         None
price_at(2018-05-14):                    101.65
-> reason:            NO_ELASTICITY
-> profit_computable: False
-> objective / optimal / recommended:    None
```

## 2. Exact elasticity lookup in the deployed service

- Startup builds `_elasticity_map`, `_elasticity_status`, `_elasticity_source`
  from `elasticity_mod.estimate(train, val)["effective"]`
  (`app/services/state.py:166`, log line `elasticity ready: 17012 products`).
- `pricing_service.pricing_row` calls `state.elasticity(pid)` = dict `.get(pid)`
  → returns `None` when the product is not a key in `_elasticity_map`.
- `usable = elasticity is not None and finite and source not in (None,"none")`
  → `False` → returns `reason=NO_ELASTICITY` **before** any simulation/optimization.

## 3. Does the product exist in startup elasticity state?

**No.** It is absent from all three maps (`_elasticity_map`,
`_elasticity_status`, `_elasticity_source`).

## 4. Why it is absent — expected under Part-4 methodology

- In the demand matrix this product has **exactly one sale row in the entire
  history**: week `2018-05-14`, price 101.65, demand 1.
- **TRAIN period (weeks ≤ 2018-01-22): 74 rows, 0 sales** (price NaN, demand 0).
- Part-4 elasticity estimation (`src/optimization/elasticity.py`:
  `_per_product_elasticities`) **only iterates products with ≥1 sale row in
  TRAIN** (`sale = train[price > 0]`). Products with zero TRAIN sales never get
  a row at all — not even `Unsupported`.
- Therefore `per_product` → `effective_elasticity` → state maps never contain
  this product. Its `elasticity_source` is effectively **`none`** (absent), not
  `own`, not `category_pooled`, not `Unsupported`.

### Category-pooled fallback also cannot apply here

- `effective_elasticity` uses `category_of` built **inside `estimate()` from the
  TRAIN period** (`elasticity.py:273`). For this product TRAIN category is NaN →
  maps to the string `'nan'` → `per_category.get('nan')` → not in the 72 real
  categories → `cat_usable=False` → no `category_pooled` fallback.
- Note the **deployed `state.category_of` is also `'nan'`** — it comes from the
  first matrix occurrence of the product (`state.py:142`, per-product
  first-occurrence), not from the single sale row. This is why the API reports
  category `nan` while the offline artifact CSV (`recommended_prices.csv`)
  shows `perfumaria` (that CSV was built by merging the **reference-week** row,
  `pricing_engine.py:167`). That category display difference is cosmetic and
  does not affect elasticity.
- Even if the product had been classified `Unsupported`, the code path still
  requires `cat_usable` to be true; with TRAIN category `'nan'` it isn't.

## 5. Does it exist? — N/A (absent), so no lookup/key/serialization bug

The `None` result is a correct missing-key lookup. No product_id normalization,
index/key, week handling, or serialization mismatch is involved: the product is
simply not a key in any elasticity map.

## 6. Comparison: the 17,012 vs the selected product

- `estimate()` produces 17,012 products with a usable elasticity
  (15,474 `category_pooled`, 1,538 `own`). The selected product is not among
  them — it has **zero TRAIN sales**, the one hard requirement for inclusion.
- The offline artifact `artifacts/optimization/recommended_prices.csv` also
  contains this product with `reason=NO_ELASTICITY`, `profit_computable=False`
  — identical to the deployed response. Deployed behavior == Part-4 output.

## 7. Elasticity source for the selected product

**`none` (unavailable / absent).** Not `own`, not `category_pooled`, not even
`Unsupported` (it never reaches classification because it has no TRAIN sales).

## 8. Pricing engine correctly does NOT invent elasticity

`pricing_service.pricing_row` short-circuits with `NO_ELASTICITY` before
`simulate_scenarios` / `choose_optimal` when `usable` is false. No elasticity is
fabricated, no price optimized. Correct per methodology: "products with no
exploitable variation … no re-pricing recommendation rather than assumed
elastic."

## 9–10. Profit path & guardrails verified on known-good products

Because the selected product has no elasticity, the profit path and guardrails
are not reached for it — by design. Verified they **work** on products with
elasticity (cost 10.0 supplied, real profit optimization `(P-C)*D(P)` runs):

## 11. Diagnostic comparison (full AppState reproduction, week 2018-05-14, cost 10.00)

| Field | Selected product | Known-good (own) | Known-good (category_pooled) |
| --- | --- | --- | --- |
| product_id | `00066f42a…` | `0296dd002a…` | `005030ef108f…` (perfumaria) |
| elasticity | **None** | -1.1460 | -0.9674 |
| elasticity_source | none | own | category_pooled |
| elasticity_status | None | Weak | Unsupported |
| current_price | 101.65 | 650.00 | 13.99 |
| baseline_demand | 1.15692 | 0.77033 | 1.20282 |
| recommended_price | **None** | 520.00 | 16.79 |
| objective | None (not optimized) | profit | profit |
| profit_computable | **False** | True | True |
| reason | **NO_ELASTICITY** | optimized | optimized |

Guardrails applied on the two optimized rows: `margin_floor_applied=True`
(cost floor @ 10.0, policy min_margin 0% → floor=10; both well above), no
ceiling/floor/smoothing engaged (defaults: ±20% cap 1.2×, alpha=1.0). The
`category_pooled` example also confirms that **perfumaria's pooled elasticity
is usable** (-0.9674, t≈-4.47) — so the absence for the selected product is
specifically because that product has zero TRAIN sales, not because its
category is unusable.

## 12. Verdict & conclusion

- **A. EXPECTED_NO_ELASTICITY.**
- The selected product has no usable elasticity under the existing Part-4
  methodology (zero TRAIN sales → excluded from per-product estimation → no
  `own`, no `category_pooled`).
- The deployed pricing service is **behaving correctly** according to the Part-4
  design: it refuses to invent elasticity, optimizes only when a real
  elasticity is present, and its result exactly matches the verified offline
  `recommended_prices.csv` artifact.
- No code, model, registry, guardrail, contract, or dashboard change was made.