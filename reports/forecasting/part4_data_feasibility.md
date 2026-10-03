# Part 4 — Phase 4.0 Data Feasibility Audit (Dynamic Pricing & Profit Optimization)

**Scope:** Audit the data actually present in the repository before any elasticity/profit code is written. Read-only; no model, split, or artifact was modified. **No access to the protected `data/splits/test.csv`.**
**Date:** 2026-09-24

## 1. Required-input availability

| Blueprint input | Available? | Evidence (raw sources) |
|---|---|---|
| **Price** | ✅ YES — unit transaction price | `olist_order_items_dataset.csv.price` — 112,650 rows, min 0.85, mean 120.65, p99 890, max 6,735. Weekly average computable from item revenue/units. |
| **Cost** | ❌ **NO** — no cost/margin/COGS field anywhere | No `cost`/`margin`/`cogs`/`unit_cost` column in `order_items`, `orders`, `products`, `order_payments`. `order_payments` only records `payment_type` / `payment_installments` / `payment_value`. |
| **Demand** | ✅ YES — weekly units | `data/processed/demand_matrix_weekly.csv.gz` — 3,469,274 product×week rows; `order_demand` dense; also present in `data/splits/{train,val,test}.csv` (not used for Part 4 fitting). |
| **Promotions** | ❌ **NO** — no promo/discount/campaign/coupon flags | No such columns in any table. Price history therefore mixes any unobserved promotional discounts into the recorded transaction price (measurement caveat below). |
| **Time** | ✅ YES — purchase timestamps | `orders.order_purchase_timestamp` 2016-09-04 → 2018-10-17; weekly grain normalized into the 106-week dense matrix (2016-08-29 → 2018-09-03, week-start labels). |

## 2. Weekly price-variation check (pre-condition for elasticity)

Computed from item-level transactions aggregated to product-week (`total_revenue / units`):

| Statistic | Value |
|---|---|
| Products in item sales | 32,951 |
| Product-week rows with a sale | 79,335 (over 95 distinct weeks) |
| Per-product distinct weekly avg-prices — median / mean | **1 / 1.30** |
| Products with ≥2 distinct weekly avg-prices | **5,962 / 32,951 (18.1%)** |
| Products with >5 distinct weekly avg-prices | 240 (0.7%) |
| Products whose weekly price spans ≥5% (max/min) | 4,922 (14.9%) |
| Products whose weekly price spans ≥20% (max/min) | 2,156 (6.5%) |
| Within-product CV of weekly avg-price — median / p90 | 0.000 / 0.129 |

**Reading:** price is real but **mostly flat over time per product** — 82% of products show zero price variation week-over-week, so per-product elasticity is **identifiable only for a minority** (~0.7–18% depending on the bar). Any elasticity product must therefore:
- estimate per-product elasticity **only where within-product price variation exists** (with a determinism bar, e.g. ≥2 distinct prices **and** a min-span);
- pool at **category level** for products with too little signal — used as a *fallback*, always flagged;
- mark products with **no exploitable variation as `Unsupported`** (no re-pricing recommendation), not silently assumed elastic;
- never claim causal identification from cross-sectional/correlational variation (single-equation log-log is a *descriptive* demand response; no experimental or instrumental price variation exists).

## 3. Demand — granularity & separation

- Dense weekly matrix: **32,729 products × 106 weeks**. Every product-week has `order_demand` (zeros included).
- `avg_price` is non-null only on weeks with sales (78,940 / 3,469,274 rows) — price observations live on the demand surface, not in zero-demand weeks.
- Part 3 frozen 93-col forecasting schema **does include lagged price features** (`price_prev`, `price_chg_1w`, `freight_prev`). The forecasting model is therefore **not price-neutral**. Part 4 will still use the Production forecast as the **demand baseline at current price** and apply price-elasticity demand response on top — the documented Part 4 architecture separation (forecast → baseline; elasticity → response). No forecast model is modified.

## 4. Period boundaries for Part 4 (no Test-split use)

From the matrix week labels and the fixed Part 3 split:

| Period | Week-start range | Use in Part 4 |
|---|---|---|
| TRAIN-period (fitting) | 2016-08-29 → 2018-01-22 | elasticity fitting, base rates |
| VAL-period (validation) | 2018-01-29 → 2018-05-14 | elasticity validation / model selection |
| TEST-period | 2018-05-21 → 2018-09-03 | **NOT used for tuning or repeated inspection** |

The Part 4 modules never read `data/splits/test.csv`; they derive periods from the matrix week labels and assert the test-start boundary.

## 5. Profit-executability statement

`Profit = (Price − Cost) × Demand` **cannot be fully executed** because **Cost is not present in any repository data**. The engine is implemented with an **optional external cost input** (a documented schema: product_id → cost). Behavior without cost:
- price-simulation grid and demand-response/revenue projections: **fully executed**;
- profit columns and an absolute profit-optimal price: **NOT computed** — flagged `PROFIT_NOT_COMPUTABLE (no cost)`; the engine reports the revenue-optimal price scenario and guardrail bounds only;
- with a cost source supplied at run time, the profit optimizer and margin guardrails activate for the products that have cost.

## 6. Verdict

**FEASIBLE WITH LIMITATIONS.** Price, demand, and time are present and adequate. **Cost and promotions are absent** — profit optimization needs an explicit, user-supplied cost source, and elasticity is a descriptive response estimate for a minority of products (with category pooling and `Unsupported` marking for the rest). No business data is invented; no guardrail margin is fabricated; nothing is tuned on the Test split.