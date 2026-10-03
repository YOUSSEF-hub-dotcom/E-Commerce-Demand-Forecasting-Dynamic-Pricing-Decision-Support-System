# External API Input Contract Audit — Prediction, Pricing, Explanation

Date (UTC): 2026-09-25 · Scope: discovery/audit only — no implementation. Verifies every claim against source code.

**Legal boundary of this audit**
```
External contract (business)  ≠  Feature-engineering schema  ≠  93-column model schema
```
The manager provides **business concepts**. Feature engineering is owned by the backend. The Production model (`models:/demand_forecasting_final/2`) remains untouched.

---

## 1. Executive Summary

- The Production PyFunc model consumes a frozen **93-column** internal matrix: `13 base + 73 one-hot cat_* + 7 regime`.
- **Every one of the 93 features is strictly past-only** and can be reconstructed from just two external values — `product_id` + `week` — using the existing frozen pipeline (`final_forecast.build_frozen_features` → `demand_baseline.frozen_features_for_week` → `prepare_model_matrix`). No 93-field request is ever needed.
- The **minimal `PredictionRequest` = `{product_id, week?}`**. `week` is optional and defaults to the last validation week (2018-05-14). Pricing additionally accepts an optional `cost`.
- **Critical separation:** the model's price features are `price_prev` / `price_chg_1w` — *previous-week* prices, **never the current week's price**. The Part-4 pricing flow uses `current_price` (a separate, matrix-retrieved value) only as the simulation anchor `P0`, applied through elasticity — the 93-feature vector is not re-derived when simulating price changes.
- A genuine **forward-looking (future-week) forecast is not supported** by the frozen pipeline (no recursive/roll-forward loop). The service is scoped to weeks within the observed matrix timeline, ≤ the reference week. This is a boundary, not a bug; it must be surfaced to the manager.
- **Implementation readiness: `READY WITH BUSINESS DECISIONS`** — the external contracts are fully determined by existing code; a small set of owner decisions (§10) remains (cost source, week exposure, product-catalog scope).

---

## 2. Current 93-Feature Schema (complete classification)

Source of truth for column order: the Production model signature `models:/demand_forecasting_final/2` (verified identical to `est.feature_names_in_` and the wrapper `feature_columns`). Feature semantics from `src/features/feature_engineering.py`, `src/models/final_forecast.py`, `src/models/data_prep.py`.

Classification legend:
- **A = Direct Business Input** · **B = Historical/Retrieved** · **C = Derived** · **D = Encoding/Representation** · **E = Model/Inference internal** · **F = Forbidden/leakage-prone**

| # | Feature | Class | Prediction-time | Source / how generated | Notes |
|---|---------|:-----:|-----------------|------------------------|-------|
| 1 | `week_of_year` | C | Known | `add_calendar_features` ← week start (ISO week) | derived from requested week |
| 2 | `month` | C | Known | `add_calendar_features` ← week start | derived from requested week |
| 3 | `quarter` | C | Known | `add_calendar_features` ← week start | derived from requested week |
| 4 | `year` | C | Known | `add_calendar_features` ← week start | derived from requested week |
| 5 | `demand_lag_1` | B | Known (history) | `add_lag_features(shift(1))`, fillna 0 | realized demand ≤ t-1 |
| 6 | `demand_lag_2` | B | Known (history) | `shift(2)`, fillna 0 | realized demand ≤ t-2 |
| 7 | `demand_lag_4` | B | Known (history) | `shift(4)`, fillna 0 | realized demand ≤ t-4 |
| 8 | `rolling_demand_mean_4w` | C | Known (history) | `add_rolling_features`: `shift(1)` then mean(4) | covers [t-4, t-1] |
| 9 | `rolling_demand_std_4w` | C | Known (history) | `shift(1)` then std(4) | covers [t-4, t-1] |
| 10 | `price_prev` | B | Known (history) | `add_demand_drivers`: `avg_price.shift(1)` | avg price of **prior** week |
| 11 | `price_chg_1w` | C | Derived | `(price_prev − price_2w_prev)/price_2w_prev`, NaN→impute | prior-week *change*, both terms lagged |
| 12 | `freight_prev` | B | Known (history) | `avg_freight.shift(1)` | avg freight of prior week |
| 13 | `category_share_prev` | C | Derived | product demand(t-1) / category total(t-1), NaN→0 | prior-week share |
| 14–86 | `cat_*` (73) | D | Known (identity) | `product_category_name` → `pd.get_dummies(prefix="cat", float32)`; unseen→`unknown` | manager supplies semantic category (here: retrieved for the product), never the OHE columns |
| 87 | `rolling_demand_mean_8w` | C | Known (history) | `add_regime_features` (`shift(1)`+mean) | regime |
| 88 | `rolling_demand_mean_12w` | C | Known (history) | `add_regime_features` | regime |
| 89 | `rolling_demand_mean_16w` | C | Known (history) | `add_regime_features` | regime |
| 90 | `rolling_demand_std_8w` | C | Known (history) | `add_regime_features` | regime |
| 91 | `regime_ratio_4_12` | C | Derived | `mean4w/(mean12w+1e-6)` | derived |
| 92 | `regime_ratio_8_16` | C | Derived | `mean8w/(mean16w+1e-6)` | derived |
| 93 | `positive_rate_8w` | C | Derived | rolling mean of `(shift(1) demand > 0)` over 8w | derived |

**The 73 `cat_*` columns:** `agro_industria_e_comercio, alimentos, alimentos_bebidas, artes, artes_e_artesanato, artigos_de_festas, artigos_de_natal, audio, automotivo, bebes, bebidas, beleza_saude, brinquedos, cama_mesa_banho, casa_conforto, casa_conforto_2, casa_construcao, cds_dvds_musicais, cine_foto, climatizacao, consoles_games, construcao_ferramentas_construcao, construcao_ferramentas_ferramentas, construcao_ferramentas_iluminacao, construcao_ferramentas_jardim, construcao_ferramentas_seguranca, cool_stuff, dvds_blu_ray, eletrodomesticos, eletrodomesticos_2, eletronicos, eletroportateis, esporte_lazer, fashion_bolsas_e_acessorios, fashion_calcados, fashion_esporte, fashion_roupa_feminina, fashion_roupa_infanto_juvenil, fashion_roupa_masculina, fashion_underwear_e_moda_praia, ferramentas_jardim, flores, fraldas_higiene, industria_comercio_e_negocios, informatica_acessorios, instrumentos_musicais, la_cuisine, livros_importados, livros_interesse_geral, livros_tecnicos, malas_acessorios, market_place, moveis_colchao_e_estofado, moveis_cozinha_area_de_servico_jantar_e_jardim, moveis_decoracao, moveis_escritorio, moveis_quarto, moveis_sala, musica, papelaria, pc_gamer, pcs, perfumaria, pet_shop, portateis_casa_forno_e_cafe, relogios_presentes, seguros_e_servicos, sinalizacao_e_seguranca, tablets_impressao_imagem, telefonia, telefonia_fixa, unknown, utilidades_domesticas` (73 total; `unknown` is the reserved unseen-category bucket).

**Class totals:** A = 0 (no raw model feature is a direct manager input) · B = 6 (`demand_lag_1/2/4`, `price_prev`, `freight_prev`) · C = 14 (calendar 4 + rolling 4w 2 + change 1 + share 1 + regime 6) · D = 73 (`cat_*`) · E = 0 dedicated (the representation is internal to `prepare_model_matrix`) · F = 0 (all 93 are past-only by construction, verified in `feature_engineering.py` docstring + shift/roll implementation).

**Excluded columns (never model inputs, must never be accepted):** `product_id`, `week` (identity — carried for lookups only), `order_demand` (current-week target!), `demand_log1p` (target transform!), `product_category_name` (raw semantic, replaced by OHE), `avg_price`, `avg_freight` (current-week monetary cols, `CURRENT_WEEK_COLS`), `sample_weight`, plus intermediate `price_2w_prev`, `cat_total`.

---

## 3. Feature Dependency Graph (external info → 93 features)

```
product_id ──┐
             ├──► data/processed/demand_matrix_weekly.csv.gz  (dense product×week, 3,469,274 rows)
week ────────┘     (avg_price, avg_freight, order_demand, product_category_name per product-week)

  │  src/models/final_forecast.build_frozen_features(matrix)
  │    add_calendar_features      ──► week_of_year, month, quarter, year          (← week)
  │    add_demand_drivers         ──► price_prev, freight_prev, price_chg_1w,
  │                                   category_share_prev                          (← history)
  │    add_lag_features(1,2,4)    ──► demand_lag_1/2/4                             (← history)
  │    add_rolling_features(4w)   ──► rolling_demand_mean_4w, rolling_demand_std_4w (← history)
  │    add_regime_features        ──► rolling_demand_mean_8/12/16w, std_8w,
  │                                   regime_ratio_4_12, regime_ratio_8_16,
  │                                   positive_rate_8w                              (← history)
  │  row.nan fill = train-region column means (strictly-pre-VAL weeks, train-only)
  │  product_category_name mapped: model-signature vocabulary ∪ {"unknown"}
  │  src/models/data_prep.prepare_model_matrix(rows, categories=signature_vocab)
  │    pd.get_dummies(prefix="cat") → 73 cat_* (unseen → cat_unknown=1)
  │    reindex to the frozen 93 → schema order preserved
  │
  ▼
93-column design row (float64) ──► model.predict ──► expm1(log) ──► predicted_demand_units
```

**Single entry point used by Part 4/5 (and the future API):** `demand_baseline.frozen_features_for_week(matrix, reference_week, model=model)` → returns `[product_id, week, *93]`. Category vocabulary comes from the **Production signature** (`cat_*` inputs), not from re-computed stats — so unseen categories always collapse to `cat_unknown`.

---

## 4. `PredictionRequest` Proposal

Manager request: **`{ product_id, week? }`**. That is the complete business-level contract; everything else is backend-owned.

| Field | Type | Required | User input | Retrieved | Derived | Validation | Reason |
|-------|------|:--------:|:----------:|:---------:|:-------:|-----------|--------|
| `product_id` | str (≤64 chars) | yes | yes | no | no | non-empty; must exist in the product catalog (matrix). Unknown product → 422 (`PRODUCT_NOT_FOUND`). | identifies the product and its entire historical context |
| `week` | str `YYYY-MM-DD` (ISO Monday) | no (default = `reference_week`) | yes* | no | no | format `^\d{4}-\d{2}-\d{2}$`; must be an existing matrix week-start; `matrix_min_week ≤ week ≤ LAST_VAL_WEEK(2018-05-14)`; **`week < TEST_START_WEEK(2018-05-21)` enforced**; not a Monday → 422 (`INVALID_WEEK`). | the demand week being forecast; features use info ≤ week−1 |

\* `week` is manager-provided only when the manager wants a week other than the default reference week. Default keeps the UI to a single required field.

**Why no other fields:** category (§11), demand lags/rolling/regime (§3), previous prices, and train-only imputation are all computed by the pipeline from `product_id + week`. Current-week price is not a model feature (§10). No current/weekly demand is accepted (leakage, §16).

---

## 5. `PricingRequest` Proposal

Manager request: **`{ product_id, week?, cost? }`**. Computed server-side, in order: demand baseline Q0 (Production model on the frozen 93 features of `week`) → elasticity lookup → `simulate_scenarios` (grid P±{5,10,15,20}%) → `choose_optimal` → `apply_guardrails` → recommendation.

| Field | Type | Required | User input | Retrieved | Derived | Validation | Reason |
|-------|------|:--------:|:----------:|:---------:|:-------:|-----------|--------|
| `product_id` | str | yes | yes | no | no | as in `PredictionRequest` | product identity + history |
| `week` | str | no (default `reference_week`) | yes* | no | no | as in `PredictionRequest`; must be a Part-4 train/val week (`assign_period` ∈ {train,val}) | reference week: P0 + feature anchor |
| `cost` | float | no | yes | no | no | if present: `0 < cost < Inf`, finite; else ignored | profit optimization input — **see §10 (cost source decision)** |

**Behavior preserved from Part-4 (`pricing_engine.run_engine`):**
- `current_price` is **retrieved** from the matrix for the reference week (`avg_price > 0`); products without one → reason `NO_PRICE_OBSERVATION`.
- No elasticity for the product → reason `NO_ELASTICITY` (no re-price recommendation).
- No `cost` → reason `PROFIT_NOT_COMPUTABLE (no cost supplied)`, `profit_computable=false`, revenue-optimal scenario returned **as a flagged proxy**, guardrail margin-floor **inactive** (`guardrail_margin_floor_inactive_no_cost=true`) — **profit is never invented**.
- With `cost` → profit objective, margin floor `cost*(1+margin%)` active, ±20% ceiling, floor/smoothing per `GuardrailConfig()` defaults.
- Candidate-price range is **fixed** (`SCENARIO_MULTIPLIERS = 0.80…1.20`) — not user-configurable in the existing engine; exposing it would be a new business rule (§10).

---

## 6. `ExplanationRequest`

**Reuse `PredictionRequest` verbatim — no separate contract.** Explanation targets the exact same design row (same `product_id + week`) that produced the prediction: the Part-5 stack (`resolve_estimator` → `define_sample` → `TreeExplainer` → per-row additivity-corrected base) requires no additional business inputs. Response adds only attribution fields (top positive/negative contributors, base_log1p, additivity disclosure). Manager-facing semantics remain **feature contribution, not causality**.

---

## 7. Manager UI Input Specification (Streamlit)

**Manager-entered (2 + 1 optional):** `product_id` (search/select from catalog) · `week` (optional, date picker defaulted to reference week) · `cost` (optional, pricing section only, pre-filled "unknown").

**System-retrieved (displayed, not editable):** product category · current price (reference week) · elasticity status/source · historical demand summary (last weeks) · product’s presence in the catalog.

**System-generated (displayed):** predicted demand (units) · forecast week · top +/− SHAP contributors · recommended price · scenario table (P±grid → demand/profit) · guardrail flags · reason code.

**Advanced optional controls — none exposed by default.** Candidate-price grid, guardrail bounds, and elasticity are engine-owned. Exposing guardrail overrides is a `REQUIRES BUSINESS DECISION` (§10).

The manager never sees: LightGBM, one-hot columns, lag/rolling/regime features, the 93-column schema, SHAP internals.

---

## 8. Historical Data Dependencies

| Required information | User provides? | Backend retrieves? | Derived? | Historical? | Source |
|----------------------|:--------------:|:------------------:|:--------:|:-----------:|--------|
| Product identity | ✅ `product_id` | – | – | – | manager entry → catalog lookup |
| Forecast/reference week | ✅ (optional, defaulted) | default `2018-05-14` | – | – | manager entry or `LAST_VAL_WEEK` |
| Category (semantic) | ❌ | ✅ (product→matrix) | – | ✅ | `demand_matrix_weekly.csv.gz` / fact table |
| Historical demand (per product) | ❌ | ✅ | – | ✅ | matrix `order_demand` (dense, 0-filled) |
| Previous-week avg price | ❌ | ✅ | – | ✅ | matrix `avg_price` → `shift(1)` |
| Previous-week avg freight | ❌ | ✅ | – | ✅ | matrix `avg_freight` → `shift(1)` |
| Two-week-ago price basis | ❌ | self | ✅ | ✅ | `price_prev.shift(1)` |
| Category share | ❌ | self | ✅ | ✅ | demand / category-total (t−1) |
| Calendar fields | ❌ | self | ✅ | ✅ | week start (ISO) |
| Rolling stats 4/8/12/16w | ❌ | self | ✅ | ✅ | `shift(1)`+rolling over history |
| Regime ratios / positive rate | ❌ | self | ✅ | ✅ | rolling means, `shift(1) demand>0` |
| Train-region imputation stats | ❌ | ✅ (frozen) | – | ✅ | weeks < `VAL_START_WEEK` (fitted once) |
| One-hot category vocabulary | ❌ | ✅ (model signature) | – | – | Production model signature (73 `cat_*`) |
| Elasticity (status/source/value) | ❌ | ✅ (startup estimate) | ✅ | ✅ | Part-4 elasticity map (Part-4 train/val periods) |
| Current price P0 (pricing only) | ❌ | ✅ (matrix, ref week) | – | ✅ | matrix `avg_price > 0` for reference week |
| Cost (pricing only) | ✅ optional | future cost source (§10) | – | – | manager entry or external cost feed |
| Pricing bounds / candidate range | ❌ | engine defaults (`SCENARIO_MULTIPLIERS`, `GuardrailConfig`) | – | – | engine constants — not user-controlled |

---

## 9. Leakage Review (prediction-time safety per input)

| Proposed input | Is it known before the forecast week? | Verdict |
|----------------|--------------------------------------|---------|
| `product_id` | Yes (identity) | Safe |
| `week` | Yes (request context) | Safe — **but** must be **≤ reference week and < test boundary**; a `week` ≥ `TEST_START_WEEK` is rejected to protect the reserved split |
| `cost` | Yes (business field) | Safe; never used in model features (pricing-only) |
| — (nothing else is accepted) | — | — |

Every one of the 93 features is derived from information at weeks **≤ t−1** (`groupby shift`, `shift(1)` then rolling — verified in `feature_engineering.py`), so no feature leaks the target. **Rejected-explicitly inputs** (must never be accepted): `order_demand`/current or future demand, `demand_log1p`, realized sales of the forecast week, future prices/freight, any `avg_price`-of-forecast-week-as-a-feature, or any post-period outcome.

**BLOCKER scan:** none found — no existing inference path depends on future/target information (the on-the-fly builder slices weeks ≤ reference and fills with train-region stats only).

---

## 10. Ambiguities / Business Decisions

1. **`cost` source** — `REQUIRES BUSINESS DECISION`: Part-4 reads cost from an *external CSV path* (`load_costs`). For the API, is `cost` (a) a manager-entered optional per-request field, (b) loaded from a future business cost data source, or (c) both (override by source)? Recommended default per existing code: **optional per-request `cost`**; absent → `PROFIT_NOT_COMPUTABLE` (behavior preserved).
2. **`week` exposure** — `REQUIRES BUSINESS DECISION`: expose week as an optional manager field (default `reference_week`) vs. hide it and always use the reference week. Recommended: expose optional (existing pipeline is week-parameterized).
3. **Product catalog scope for prediction** — `REQUIRES BUSINESS DECISION`: prediction/pricing are supported only for products present in the matrix (that is what the frozen pipeline can reconstruct). New products require history; a cold-start route is a new feature, not inferable from code.
4. **Current-price override** — `REQUIRES BUSINESS DECISION`: Part-4 anchors P0 on the observed reference-week `avg_price` and drops products without one. A manager-entered `current_price` override is **not** part of the existing engine; exposing it would be a new rule.
5. **Candidate price range / guardrail overrides** — `REQUIRES BUSINESS DECISION`: grid ±20% and guardrail defaults are engine constants; user-configurable bounds are a new capability.
6. **Exact `product_id` max length / charset** — `REQUIRES BUSINESS DECISION` (a sane ≤64 UTF-8 default is suggested, without inventing an alphanumeric rule).
7. **Future-week forecasting** — `REQUIRES BUSINESS DECISION` (and `BLOCKED` at pipeline level): the frozen model cannot forecast weeks beyond the matrix timeline (no roll-forward). This is a documented boundary for the manager UI.

No item blocks the contracts as designed; each is flagged so the owner decides before implementation.

---

## 11. Recommended API Boundary

```
External Request                  PricingRequest (product_id, week?, cost?)
   {product_id, week?}        ──►  ▼
        ▼                        demand_baseline.frozen_features_for_week  (93-float64 row)
  Validation (Pydantic)          ▼
        ▼                     predict_baseline / wrapper.predict  →  Q0 baseline
  Feature Preparation            ▼
  (build_frozen_features +       elasticity lookup (Part-4 map)
   prepare_model_matrix)         ▼
        ▼                     simulate_scenarios (P±grid) → choose_optimal → apply_guardrails
  93-Feature Model Schema        ▼
        ▼                     recommendation {reason, recommended_price, guardrails}
  Production Model (expm1)
        ▼
  predicted_demand_units
```

`ExplanationRequest == PredictionRequest` → same row → Part-5 TreeExplainer → top +/− contributors.

---

## 12. Implementation Readiness

**`READY WITH BUSINESS DECISIONS`.**

The external contracts are fully derivable from existing code: `PredictionRequest = {product_id, week?}`, `PricingRequest = {product_id, week?, cost?}`, `ExplanationRequest = PredictionRequest`. No 93-field contract, no manual feature reconstruction, no leakage, no pipeline change. Implementation can start once the §10 owner decisions are confirmed (defaults are recommended for each).