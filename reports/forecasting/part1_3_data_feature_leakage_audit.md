# Part 1–3 Data & Feature Leakage Audit

**Date:** 2026-09-23
**Scope:** Evidence-based reconciliation of *documented vs actual* pipeline behavior. Audit only — no code, data, model, threshold, or registry changes were made.
**Method:** Read of `PROJECT_BLUEPRINT.md`, `PROJECT_MAP.md`, all `src/data/*`, `src/features/*`, `src/models/*`, `src/utils/*`, notebooks 01–06, `tests/*`, `artifacts/phase3/*`, and `reports/forecasting/*`; plus live verification of raw headers, split furniture, schema artifacts, category vocabularies, and the v2 audit trail.

---

## 1. Raw Data Feature Inventory (Q1)

Microsoft: only **4 of 9** raw tables are ingested by the forecasting pipeline (`orders`, `order_items`, `products`, `translation`). Customers, geolocation, payments, reviews, and sellers live in `data/raw` but are **never read** by `src/`.

| Feature | Original Dataset | Data Type | Business Meaning | Used Later? |
|---|---|---|---|---|
| `order_id` | orders | object (PK) | Order identifier | Yes — join key / aggregation group |
| `customer_id` | orders | object (FK) | Ordering customer | Yes (merge) but **not a model feature** |
| `order_status` | orders | object | Fulfilment state | Yes — status policy filter (`EXCLUDED_ORDER_STATUSES`) |
| `order_purchase_timestamp` | orders | datetime64 | Purchase time | Yes — week grain + calendar + target |
| `order_approved_at` | orders | datetime64 | Approval time | Parsed; **not a model feature** |
| `order_delivered_carrier_date` | orders | datetime64 | Carrier delivery | Parsed; **not a model feature** |
| `order_delivered_customer_date` | orders | datetime64 | Customer delivery | Parsed; **not a model feature** |
| `order_estimated_delivery_date` | orders | datetime64 | Estimated delivery | Parsed; **not a model feature** |
| `order_id` | order_items | object (FK) | Order | Yes — merge key |
| `order_item_id` | order_items | int64 | Item position (natural key half) | Yes — demand = `count(order_item_id)` per product-week |
| `product_id` | order_items | object (FK) | Product | Yes — aggregation + identity (dropped from design matrix) |
| `seller_id` | order_items | object (FK) | Seller | **Not used** (no seller feature enters model; not ingested) |
| `shipping_limit_date` | order_items | datetime64 | Ship-by deadline | Parsed; **not a model feature** |
| `price` | order_items | float64 | Unit price (BRL) | Yes — weekly `avg_price` → `price_prev` / `price_chg_1w` |
| `freight_value` | order_items | float64 | Freight charge | Yes — weekly `avg_freight` → `freight_prev` |
| `product_id` | products | object (PK) | Product | Yes — join key |
| `product_category_name` | products | object | Portuguese category | **Yes — THE categorical feature (OHE, 73 levels incl. `unknown`)** |
| `product_name_lenght` | products | int64 | Name length | Parsed; **not a model feature** |
| `product_description_lenght` | products | int64 | Description length | **Not used** |
| `product_photos_qty` | products | int64 | Photo count | **Not used** |
| `product_weight_g` | products | int64 | Weight | **Not used** |
| `product_length_cm` | products | int64 | Length | **Not used** |
| `product_height_cm` | products | int64 | Height | **Not used** |
| `product_width_cm` | products | int64 | Width | **Not used** |
| `product_category_name` | translation | object | Portuguese category | Yes — merge key |
| `product_category_name_english` | translation | object | English category | Merged; **not a model feature** |
| `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state` | customers | mixed | Customer profile | **Not ingested** |
| `geolocation_*` (5 cols) | geolocation | mixed | Zip→geo | **Not ingested** |
| `payment_*` (5 cols) | order_payments | mixed | Payment | **Not ingested** |
| `review_*` (7 cols) | order_reviews | mixed | Reviews | **Not ingested** |
| `seller_*` (5 cols) | sellers | mixed | Seller profile | **Not ingested** |

**Target:** `order_demand` = weekly `count(order_item_id)` per product (business units), modeled on `demand_log1p = log1p(order_demand)` scale.

**Leakage-prone-by-nature raw columns:** `shipping_limit_date` and e-commerce delivery dates extend past the purchase stream (`shipping_limit_date` to 2020-04-09). They are **delivery deadlines, not demand**, and never enter the model — so they cannot leak. Order *approval/delivery* timestamps likewise carry post-purchase information and are correctly excluded from all features.

---

## 2. Feature Engineering Inventory (Q2)

All engineering lives in `src/features/feature_engineering.py` (Phase-2 stack, Phase-3 stack) and `src/models/final_forecast.py` (`add_regime_features`, frozen EXP2). **Every feature is past-only by construction:** lags via `shift(k)`; rolling via `shift(1)` then rolling (window `[t-w, t-1]`); all drivers shifted ≥ 1 week before use; calendar features stateless.

| Feature | Source Columns | Transformation | Time Dependency | Leakage Risk | Actually Used? |
|---|---|---|---|---|---|
| `week_of_year` | week | `isocalendar().week` | stateless (week label) | none | **Yes (13-base)** |
| `month` | week | `dt.month` | stateless | none | **Yes** |
| `quarter` | week | `dt.quarter` | stateless | none | **Yes** |
| `year` | week | `dt.year` | stateless | none | **Yes** |
| `demand_lag_1` | order_demand | `shift(1)` per product | t−1 | none | **Yes (Phase-3 lags {1,2,4})** |
| `demand_lag_2` | order_demand | `shift(2)` | t−2 | none | **Yes** |
| `demand_lag_3` | order_demand | `shift(3)` | t−3 | none | Phase-2 split only — **excluded from model** |
| `demand_lag_4` | order_demand | `shift(4)` | t−4 | none | **Yes** |
| `rolling_demand_mean_4w` | order_demand | `shift(1)`+rolling(4).mean | [t−4, t−1] | none | **Yes** |
| `rolling_demand_max_4w` | order_demand | `shift(1)`+rolling(4).max | [t−4, t−1] | none | Phase-2 split only — **excluded from model** |
| `rolling_demand_std_4w` | order_demand | `shift(1)`+rolling(4).std | [t−4, t−1] | none | **Yes** |
| `rolling_demand_mean_8w/12w/16w` | order_demand | `shift(1)`+rolling(w).mean | past windows | none | **Yes (7 regime)** |
| `rolling_demand_std_8w` | order_demand | `shift(1)`+rolling(8).std | past window | none | **Yes (7 regime)** |
| `regime_ratio_4_12` | mean_4w, mean_12w | mean4/(mean12+ε) | past windows | none | **Yes (7 regime)** |
| `regime_ratio_8_16` | mean_8w, mean_16w | mean8/(mean16+ε) | past windows | none | **Yes (7 regime)** |
| `positive_rate_8w` | order_demand | `shift(1).gt(0)` rolling(8).mean | past 8 w | none | **Yes (7 regime)** |
| `price_prev` | avg_price (weekly mean) | `shift(1)` per product | t−1 | none | **Yes** |
| `price_2w_prev` | avg_price | `shift(2)` | t−2 | none | intermediate — **dropped** |
| `price_chg_1w` | price_prev, price_2w_prev | (p1−p2)/p2, inf→NaN | t−1 vs t−2 | none | **Yes** |
| `freight_prev` | avg_freight | `shift(1)` | t−1 | none | **Yes** |
| `category_share_prev` | order_demand, category | product_demand_prev / category_total_prev | t−1 | none | **Yes** |
| `cat_total` | order_demand | in-week category sum (transform) | t (current) | **current-week → immediately shifted** | intermediate — **dropped** |
| `avg_price` | price | mean per product-week | t (current) | **current-week** | excluded (`CURRENT_WEEK_COLS`) |
| `avg_freight` | freight_value | mean per product-week | t (current) | **current-week** | excluded (`CURRENT_WEEK_COLS`) |
| `demand_log1p` | order_demand | `np.log1p` | t (row-local) | none (stateless per row) | **target**, not feature |
| `sample_weight` | order_demand | `1 + log1p(demand)` | t (row-local) | none | used only as **weight** |

**Clean temporal evidence:** `cat_total`, `avg_price`, `avg_freight` are the only in-week statistics, and all three are either dropped or only consumed *after* an explicit 1-week shift. This is enforced by construction and verified behaviorally (`tests/test_data_processing.py::TestAntiLeakageProbe`, `tests/test_features.py::TestPhase3NoLookahead`, and the notebook-03 fold-boundary equivalence proof).

---

## 3. Final Model Feature Inventory (Q3)

Verified from code + live artifacts, not assumption:

- **A. Original features entering any engineering:** `order_purchase_timestamp` (→ week), `price`, `freight_value`, `product_category_name`, `product_id`, `order_item_id` (→ demand count).
- **B. Engineered features (all past-only):** 4 calendar + lags {1,2,4} (Phase-3) + rolling mean/std 4w + drivers {price_prev, price_chg_1w, freight_prev, category_share_prev} + 7 EXP2 regime + 73 one-hot category indicators.
- **C/D. Features actually passed into v2 — 93 columns** (confirmed in `reports/forecasting/part3_v2_audit.json` `schema_columns` = 93):

  | Group | Count | Columns |
  |---|---|---|
  | Base (FEATURE_COLS, 13) | 13 | `week_of_year, month, quarter, year, demand_lag_1, demand_lag_2, demand_lag_4, rolling_demand_mean_4w, rolling_demand_std_4w, price_prev, price_chg_1w, freight_prev, category_share_prev` |
  | One-hot category | 73 | `cat_*` (incl. `cat_unknown`; float32, 0/1) |
  | EXP2 regime | 7 | `rolling_demand_mean_8w/12w/16w, rolling_demand_std_8w, regime_ratio_4_12, regime_ratio_8_16, positive_rate_8w` |
  | **Total** | **93** | fingerprint `phase3-exp2-v1` |

- **Explicitly excluded** (`src/models/data_prep.py`): identity (`product_id`, `week`, `product_category_name`), current-week monetary (`avg_price`, `avg_freight`), and target/weight columns. **Not in `FEATURE_COLS`:** `demand_lag_3`, `rolling_demand_max_4w` (present in the persisted Phase-2 split CSVs but never consumed by the model).

> **Answer:** The final model used a **curated subset**, not "every available feature": 93 of the engineered columns were consumed; `demand_lag_3`, `rolling_demand_max_4w`, `avg_price`, `avg_freight`, and all raw product-spec/seller/customer/payment/review/geolocation columns were excluded. Exclusions are domain-driven (current-week leakage avoidance + identity) or by the Phase-3 hyperparameter-set reduction (lags {1,2,4}, rolling mean/std).

---

## 4. Feature Selection Audit (Q4/Q5/Q15)

| Method | Status | Evidence |
|---|---|---|
| Manual / domain-driven exclusion (current-week + identity + Phase-3 subset) | **Implemented** | `data_prep.py` `ROW_IDENTITY_COLS`, `CURRENT_WEEK_COLS`, `FEATURE_COLS`; `final_forecast.py` `REGIME_COLS`, `freeze_schema` (93) |
| Correlation filtering | **Not Implemented** (diagnostic only) | Spearman corr in notebooks 04/05 — exploratory, no column removed by it |
| Statistical / hypothesis-test-driven selection | **Not Implemented** (intentional per blueprint §Part1) | No t-test/ANOVA/chi² in code |
| Permutation importance | **Not Implemented** | Not present |
| SHAP-based selection | **Not Implemented** | Blueprint §Part5 phase (future), not Part 3 |
| Recursive feature elimination / model-based selection | **Not Implemented** | Not present |
| Constant / near-zero variance filtering | **Not Implemented** | Not present |
| Duplicate / redundant feature removal | **Partially Implemented** (structural only) | Drop of `price_2w_prev`, `cat_total` intermediates; not a systematic redundancy filter |
| Tree-gain importance | **Implemented (diagnostic only)** | Notebook 05 top-5, notebook 04 top-12 by GAIN on the winner — **never used to remove a feature** |

**Conclusion:** Feature *selection* as an automated elimination step was **never executed**. The feature stack was fixed a priori (Phase-2), reduced to a Phase-3 subset (`PHASE3_LAGS (1,2,4)`, rolling `mean/std`), and extended by the EXP2 controlled experiment (7 regime features), where selection was validated by **temporal-window metrics** (WAPE / cohort bias on validation) — consistent with the blueprint's "temporal-validation as primary relevance criterion" approach. Gain importance is present only as post-hoc explainability.

---

## 5. Encoding Audit (Q6/Q7)

| Item | Finding |
|---|---|
| Categorical features | Exactly **1**: `product_category_name` (73 raw levels incl. `unknown`; 71 translation rows; 72 real + unknown = 73 in matrix) |
| Encoding used | **One-Hot Encoding** via `pd.get_dummies(prefix="cat", dtype=float32)` in `prepare_model_matrix` |
| Resulting columns | 73 float32 `cat_*` columns (incl. `cat_unknown`) |
| Where encoder fitted | `get_dummies` is **stateless** (label→columns, no target). Column set aligned by `categories=` (union across splits) with `reindex`/`fill 0` |
| Fit on train only? | Value behavior: **effectively train-determined** — provenance check: `levels in val NOT in train: ∅` and `levels in test NOT in train: ∅` (imputation already remaps out-of-train labels to `unknown`, which exists in train) |
| Unseen categories handled? | Yes — reserved `unknown` label assigned by `impute_splits` (train-vocabulary based) before encoding |
| Native categorical (LightGBM `categorical_feature`) | **Not used** — LightGBM receives OHE columns as ordinary numeric features (no special handling) |
| OHE chosen deliberately? | Yes, documented (`data_prep.py` docstring: "a single code path works for both LightGBM and XGBoost") |

### 5.1 Evaluation of the claim: *"One-Hot Encoding is the safest engineering choice for high-cardinality categoricals in Time-Series because it prevents target leakage."*

1. **Leakage-safe in this implementation?** Yes. The encoder is stateless; column alignment uses only category labels; and the effective vocabulary is provably train-derived (verified: val/test contain no levels outside the train vocabulary after imputation). No target statistics are used anywhere in encoding.
2. **Fit without target information?** Yes — `pd.get_dummies` computes from labels only.
3. **Train/val/test handled correctly?** Yes — same `categories=` list (reindex, zero-fill) keeps the 73-column schema identical across splits; schema drift is guarded by the 93-column freeze and `freeze_schema` runtime assert.
4. **High cardinality / dimensionality concerns?** None material: 73 levels is moderate; 73 float32 columns on a 3.47M-row matrix is memory-safe and verified.
5. **Is LightGBM actually using sparse OHE?** No — the OHE block is dense float32 numeric input; no native categorical feature parameter (`categorical_feature`) is passed. "Sparse" is not an accurate description of the LightGBM consumption path.
6. **Native categorical handling available/used?** Available (LightGBM supports it) but **not used** — deliberate for cross-framework (LightGBM+XGBoost) code-path unification.
7. **Deliberate + documented?** Yes (module docstring).
8. **Did OHE introduce actual leakage?** No — confirmed by the zero-lookahead probe, vocabulary provenance equality check, and the fold-boundary equivalence proof.
9. **Is "zero leakage" supported by code?** Yes **for the categorical path**. *Nuance:* `prepare_frozen` computes the OHE column vocabulary by reading `train ∪ val ∪ test` category lists via `pd.concat`. Because imputation already constrained all three splits to the train vocabulary-and-`unknown`, the resulting column set is identical to the train-derived set (proven). This is a schema-alignment convenience, not a target/statistical leak, but it **does read split data during encoder-layout setup** — worth flagging as "minor, empirically zero-leak, but not a strict train-only layout."

**Project-specific conclusion:** The claim as stated is *too absolute* ("safest ... always"). In *this* project OHE is leakage-safe because the encoding is stateless and train-vocabulary-anchored; but the safety comes from those properties, not from OHE *per se*. At genuinely high cardinality (thousands of levels) native LightGBM categorical handling would be equally leakage-safe, more compact, and faster; the "safest because it prevents target leakage" rationale should be re-stated as "target-independent and vocabulary-controlled here — a good fit given 73 levels and a dual LightGBM/XGBoost code path," not as a universal rule.

---

## 6. Temporal Split Audit (Q8)

Verified (blob audit + split files + `temporal_split.py`):

| Split | Earliest week | Latest week | Distinct week labels | Rows |
|---|---|---|---|---|
| Train | 2016-08-29/09-04 | 2018-01-29/02-04 | 75 | 2,428,492 |
| Validation | 2018-01-29/02-04 | 2018-05-21/05-27 | 17 | 520,391 |
| Test | 2018-05-21/05-27 | 2018-09-03/09-09 | 16 | 520,391 |

- Ordering **Train ≤ Val ≤ Test** with `assert_chronological_order` (uses `<=`): boundary weeks are shared between train/val (`2018-01-29/02-04`) and val/test (`2018-05-21/05-27`). The row partitions are disjoint (overall coverage and order verified), and because all features are past-only, the shared week-label does not allow current-row facts to leak across splits.
- **Feature generation before split?** Engineering runs on the full demand matrix before splitting, but every engineered value is a function of strictly prior weeks (probes prove invariance to current-week facts). No future information reaches train/val rows.
- **Global statistical operations before split?** None that feed the model: the only in-week statistics (`avg_price`, `avg_freight`, `cat_total`) are either excluded from the design matrix or used only after a lag. Median imputation is fitted **after** the split on **train only**.

---

## 7. Statistical Operation Leakage Table (Q9–Q11)

| Operation | File | Dataset Used | Before/After Split | Train-only? | Leakage Risk | Status |
|---|---|---|---|---|---|---|
| `median` imputation (numeric) | `src/data/imputation.py` | train for fit; all splits for transform | After | **Yes** | none | **PASS** |
| `unknown` structural fill (categorical) | `imputation.py` | train vocabulary for mapping | After | **Yes** (train vocab) | none | **PASS** |
| `mean` per product-week (`avg_price`/`avg_freight`) | `target_engineering.py` | same-week rows | Before (matrix) | n/a — current-week, excluded from model | none (excluded or lagged) | **PASS** |
| `sum` per category-week (`cat_total`) | `feature_engineering.py` | same-week rows | Before | n/a — immediately shifted to prev week then dropped | none | **PASS** |
| `mean/std` rolling 8/12/16w | `feature_engineering.py` / `final_forecast.py` | shifted past window | Before | n/a — past-only by `shift(1)` | none | **PASS** |
| `log1p` target transform | `feature_engineering.py` / `data_prep.py` | per-row target | Before | n/a — stateless per row | none | **PASS** |
| `1+log1p` sample weights | `data_prep.py` | per-row target | n/a | n/a — row-local | none | **PASS** |
| MASE denominator (naive MAE) | `final_forecast.py` | **train only** | After | **Yes** | none | **PASS** |
| Cohort flags (B/B2/C) | `final_forecast.py` | **train only** | After | **Yes** | none | **PASS** |
| `expm1` inverse + floor | `final_forecast.py` / `pyfunc_wrapper.py` | predictions only | n/a | n/a | none (inference-time) | **PASS** |
| EDA `describe`, `corr`, `skew` | notebooks 01/04/05 | full tables | n/a (diagnostic) | n/a — **not** used to transform model inputs | none | **PASS (diagnostic only)** |

**Summary:** No global statistic computed before the temporal split is consumed by model inputs; every statistical transform that feeds the model is either past-only by construction, train-fitted (imputation, MASE denom, cohort flags), or row-local/stateless (log1p, weights). **No confirmed leakage from statistical operations.**

---

## 8. Skewness, Outliers, Missing Values (Q9–Q11)

- **Skewness:** right-skewed count target handled via **`log1p` (target transform)** and inverse `expm1` at prediction — both stateless/row-local; no feature-level power/Box-Cox transforms (tree models are monotonic-invariant). No leakage.
- **Outliers:** `price` (max 6,735 BRL) and freight extremes are **retained as business signals** — no winsorization, capping, or removal in the pipeline (only documented in EDA). No leakage; business-critical outliers preserved per blueprint.
- **Missing values:**
  - **Structural replacement:** `product_category_name` missing → reserved `unknown` (train-vocabulary check) — coded, tested.
  - **Statistical imputation:** numeric (`avg_price`, `avg_freight`, `price_prev`, `price_chg_1w`, `freight_prev`) → `SimpleImputer(strategy="median")` **fit on train only**, transform applied to all splits; tested (`test_data_processing.py::TestPipelineEndToEnd`).
  - Distinction is clear in code and docstrings (`imputation.py`); **leakage-safe**.

---

## 9. Hypothesis Testing Audit

| Test | Status |
|---|---|
| t-test / ANOVA / Mann-Whitney / Kruskal-Wallis / chi-square | **Not Implemented** in code |
| Spearman rank correlations | **Implemented (diagnostic only, notebooks 04/05)** |
| Lag-1 autocorrelation | **Implemented (diagnostic, notebook 05)** |
| Lag cross-correlations / ACF/PACF | **Partial** — lag-1 autocorr + Spearman present; full ACF/PACF plots not found |

**Assessment:** The blueprint **explicitly directs** avoiding standard cross-sectional hypothesis tests on raw temporal demand (autocorrelation + zero-inflation violate independence) in favor of non-parametric rank metrics, lag correlations, and temporal-CV metrics. Absence of formal tests is therefore **appropriate and spec-compliant**. The predictive-relevance strategy relies on (a) expanding-window CV/WAPE/Asymmetric loss (implemented), (b) rank correlations (implemented diagnostically), (c) model-based gain importance (implemented diagnostically). **PASS for appropriateness.**

---

## 10. Temporal Cross-Validation & Optuna Audit (Q12–Q14)

- `expanding_window_splits` (forecasting.py): chronological folds, validation horizon 4 weeks, expanding training window; notebook 03 used `n_splits=5, min_train_weeks=24` on the 120 mature-product tuning subset (8,899 rows); default module signature also supports `min_train_weeks=20`.
- Each fold: train weeks strictly before validation weeks; **future observations cannot enter training**.
- Feature matrices for folds are built from the same past-only stack; the fold-boundary equivalence proof (notebook 03, 8,100 overlapping product-week rows, 16 engineered columns identical) confirms recomputing features per window equals the global past-only pass.
- **Optuna** (`optuna_tuning.py`): TPE, seed 42, minimize **mean-fold WAPE**, 10 trials/model over the train-only folds; `sub_X`/`sub_weeks`/`sub_demand` are exclusively **train-split** rows. **No validation/test exposure during tuning.**
- Winner model fully refit; final validation WAPE used once for the notebook-03 winner decision (0.4001 LGB / 0.4349 XGB counter-check); final test split was **not** used during this phase.

**Result: PASS** — expanding-window temporal CV + Optuna both isolated from future data; deterministic and reproducible.

---

## 11. Feature Importance vs Feature Selection (Q15)

- **Feature Selection** (explicit removal before training): did **not** occur algorithmically. The feature set is a fixed curated stack + EXP2 addition, selected via domain reasoning and temporal-window experiment metrics.
- **Feature Importance** (post-hoc model insight): implemented — LightGBM **gain** importance (notebooks 04/05); Spearman correlations (diagnostic). Not used to prune features.
- These two concepts are handled distinctly in the codebase; the practical feature-set decision was driven by **temporal validation evidence** (WAPE/cohort-bias deltas in notebook 06), which matches the blueprint's stated criterion.

---

## 12–14. Part-by-Part Specification Audit (Q16)

### Part 1
| Requirement | Status | Evidence |
|---|---|---|
| Data types & temporal parsing | **IMPLEMENTED** | `load_core_tables` `parse_dates`, dtype asserts; tests |
| Duplicates & natural-key integrity | **IMPLEMENTED** | `validate="many_to_one"` on all hops; tests |
| Feature format & domain constraints | **PARTIAL** | EDA flags extremes (price 6,735; shipping to 2020); **no hard range validation in code** (0.85 < price OK); shipping-limit drift is a delivery deadline, not demand |
| Leakage-free missing values | **IMPLEMENTED** | structural `unknown` + train-only median |
| Distribution diagnostics | **IMPLEMENTED** | skew/sparsity in EDA; log1p target |
| Time-series statistical EDA | **PARTIAL** | Spearman + lag-1 autocorr (diagnostic); no formal ACF/PACF artifact |
| Practical effect size & feature impact | **PARTIAL** | gain importance diagnostic; no permutation/SHAP in Part 3 |

### Part 2
| Requirement | Status |
|---|---|
| Time-series splitting | **IMPLEMENTED** |
| Conditional imputation (train-only) | **IMPLEMENTED** |
| Feature engineering (lags/rolling/calendar/elasticity/drivers) | **IMPLEMENTED** |
| Outlier & skewness handling | **PARTIAL** — outliers retained (per spec); feature-level power transforms not applied (tree-invariant); log1p applied to **target** only |
| Scaling (`RobustScaler` or skip for trees) | **NOT APPLIED** — trees only; documented skip in `data_prep.py` / PROJECT_MAP ("skipping scaling if relying on tree-based models") |
| Categorical encoding | **IMPLEMENTED** — One-Hot (73 float32 cols), schema-aligned across splits |

### Part 3
| Requirement | Status |
|---|---|
| Model selection (baseline → XGB/LGBM) | **IMPLEMENTED** (naive WAPE 1.4923; LGBM winner WAPE 0.4001 val) |
| Temporal CV | **IMPLEMENTED** (expanding window, 5 folds) |
| Optuna (train folds only) | **IMPLEMENTED** (10 trials/model, seed 42, mean-fold WAPE) |
| Target transform (log1p) | **IMPLEMENTED** |
| Sample weighting (1+log1p) | **IMPLEMENTED** |
| Early stopping | **IMPLEMENTED** in notebook-03 final fits (`fit_full_with_early_stopping`, train-tail eval set). *Nuance:* frozen v1/v2 refits with `LGBMRegressor` full 600 estimators without early stopping — reproducible to the same 0.3964 val WAPE. |
| MLflow tracking | **IMPLEMENTED** (SQLite backend, params/metrics/artifacts) |
| Model packaging (PyFunc) | **IMPLEMENTED** (`DemandForecaster`) |
| Model signature | **IMPLEMENTED** (`infer_signature`, frozen 93 cols → `pred_demand`) |
| Model registry + auto stage | **IMPLEMENTED** | 
| Quality gate | **IMPLEMENTED** (12 numeric + 7 technical; PASS→Production / FAIL→Staging) |
| Lifecycle automation | **IMPLEMENTED** (v1 scored test once → FAIL → Staging; v2 validation-only gate → PASS → Production ⚠️ provisional; once-guard enforced) |

---

## 15. Specification vs Implementation — Confirmed Mismatches (Q17)

| # | Documented | Actual (evidence) | Severity |
|---|---|---|---|
| M1 | `PROJECT_MAP` Phase-3 deliverable: "`feature_columns.json` — exact 86-column feature schema" for the winner | `artifacts/phase3/feature_columns.json` = **86** cols (13 base + 73 OHE, no regime) — this is the notebook-03 staged winner. **Final production v1/v2 schema is 93** (86 + 7 regime). The on-disk phase-3 artifact does **not** reflect the deployed model's schema. | **Medium (artifact/documentation sync)** |
| M2 | Blueprint Part 1: "lag cross-correlations … ACF/PACF" as engineered deliverables | Only lag-1 autocorr + Spearman (diagnostic); no formal ACF/PACF artifact | Minor (diagnostic surface) |
| M3 | Blueprint Part 2: "For right-skewed features carrying rich information, apply log/power transformations" | Feature-level log/power transforms **not applied**; only the **target** uses `log1p`; features remain raw (tree-invariant) | Minor/partial |
| M4 | "Safest engineering choice … one-hot prevents target leakage" (discussed claim) | OHE is leakage-safe **here**, but not because OHE is universally safest; native LightGBM categorical would be equally safe at high cardinality. Documentation should be scoped to this project | Documentation precision |
| M5 | Blueprint Part 3 "Early stopping" as a training-stage element | Notebook uses early stopping (train-tail eval); frozen lifecycle refits with full 600 trees (no ES). Same WAPE (0.3964) — reproducibility observed | Minor (training detail) |

**No mismatch found** in: temporal split ordering, train-only imputation, past-only feature construction, model packaging, signature, registry transitions, quality-gate thresholds, or test-once protection.

---

## 16. Final Answers (Q18/Q19)

### Q18 — Confirmed leakage issues
**None confirmed.** Every confirmed check is `PASS`: features are past-only by construction and behavioral probes; imputation/MASE-denominator/cohort-flags are train-fitted; OHE vocabulary is provably train-equivalent (val/test contain no out-of-train level); CV/Optuna run within train only; the protected test was scored exactly once (v1) and v2 never re-scored it.

### Q19 — Suspicious / unverifiable areas requiring manual review
1. **Artifact-schema drift (M1):** `artifacts/phase3/feature_columns.json` (86) vs deployed v2 (93). The MLflow-run `feature_columns.json` inside each run is the correct 93-col artifact; the loose phase-3 file is stale. **Recommend:** regenerate/replace the phase-3 artifact to the 93-col schema or relabel it as "staged notebook-03 winner."
2. **OHE layout reads all splits** (`prepare_frozen` `pd.concat` over train/val/test for the dummy column set). Empirically zero-leak (proven equal to train vocabulary), but the layout is not "strictly train-only." **Recommend:** restrict the category-layout source to imputed-train only to make the invariant structural, and add an assertion `set(val∪test) ⊆ set(train) ∪ {unknown}`.
3. **Shared boundary week-labels** across train/val and val/test (documented `<=` guard). Features are past-only so no current-row leakage; but a reviewer who expects a hard week-boundary split will see overlapping *labels*. **Recommend:** document the rationale explicitly in `PROJECT_MAP.md`.
4. **Win rate / early-stopping difference** between notebook-03 (early stop on train tail) and frozen v1/v2 (fixed 600 trees). Reproducibility WAPE match (0.3964) confirms equivalence in practice; a manual double-check of `feature_importance`-based claims against the frozen refit is advised before Part 5 explainability.
5. **Hyperparameters tuned on a 120-product mature subset** (8,899 rows) rather than the full train matrix — documented as a speed/representativeness trade-off; winners were then refit on the full train. Low risk, but the tuning scope is narrower than the deployment population.

---

## 17. Status Index (Q16–Q19 rollup)

| Area | Status |
|---|---|
| RAW → ENGINEERING → PREPROCESSING → SELECTION → FINAL MODEL → TEMPORAL VALIDATION | **PASS** (selection = fixed curated stack, evidenced) |
| Feature-selection automation | **PARTIAL / NOT IMPLEMENTED** (manual + experiment-driven; no algorithmic filter) |
| Categorical encoding | **PASS** (OHE, leakage-safe, vocab train-equiv.) |
| Statistical-operation leakage | **PASS** |
| Skewness/outlier/imputation leakage | **PASS** |
| Temporal CV + Optuna isolation | **PASS** |
| Hypothesis-testing appropriateness | **PASS** (spec-compliant absence) |
| Spec↔impl mismatches | **PARTIAL** (5 flagged; 1 medium artifact-schema, 4 minor) |
| Unresolved/manual review | **5 items (see §16/Q19)** |
| Final-lifecycle status | v1 Staging (FAIL, test scored once); v2 Production ⚠️ **provisional** (validation-gated); FINAL independent holdout **BLOCKED** (dataset ends at 2018-09-09) |