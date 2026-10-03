# Part-3 V2 Post-Hoc Holdout Evaluation

* Generated: 2026-09-23T17:33:26.627890+00:00
* Evaluation type: `post_hoc_holdout_evaluation` (group `post_hoc_holdout`)
* Model: `models:/demand_forecasting_final/2` (version 2, run `5d729a8594dc488bbd18ba0c4e216f5a`)

## Disclosure (required classification)

The V2 holdout evaluation is classified as post-hoc because the designated Test split had previously been consumed during the V1 evaluation and diagnosis lifecycle. Therefore, this result is not claimed as an independent untouched final-test estimate.

This artifact never claims an untouched test, unseen test, independent final test, or never-before-seen test result.

## 1. Protocol

- Authorized post-hoc holdout evaluation of the frozen V2 candidate; the designated Test split was previously consumed by the V1 evaluation and diagnosis lifecycle.
- Executed exactly once (dedicated marker + `posthoc_holdout_evaluated` MLflow tag; this module exposes **no** `--force` override).
- Frozen V2 loaded from the Model Registry: no retraining, no tuning, no threshold or parameter change before or after evaluation.
- Existing Quality Gate implementation used unchanged (12 numeric bounds + 7 technical booleans), labeled `group="post_hoc_holdout"`.
- No further holdout evaluation is available or claimed (raw data ends at the original test boundary 2018-09-09; historical blocker report preserved).

## 2. Test split metadata (designated split)

| Field | Value |
|---|---|
| Start week | `2018-05-21/2018-05-27` |
| End week | `2018-09-03/2018-09-09` |
| Start date | `2018-05-21` |
| End date | `2018-09-09` |
| Rows | 520,391 |
| Products | 32,729 |
| Week labels | 16 |
| Total actual demand | 22,949.00 |
| Total predicted demand | 22,674.31 |
| Mean actual | 0.044100 |
| Mean predicted | 0.043572 |
| Cohort rows (B/B2/C) | {'C-long-tail': 518466, 'B2-mid': 1337, 'B-sustained': 588} |
| MASE denominator (train naive) | 0.038043 |
| `test.csv` md5 (pinned) | `d35bfcd546df74b63c9bd25151fde264` |
| Split reproduction | `{'csv_rows': 520391, 'prepared_rows': 520391, 'start_week': '2018-05-21/2018-05-27', 'end_week': '2018-09-03/2018-09-09'}` |

## 3. Prediction integrity

| Check | Result |
|---|---|
| `pred_integrity` (gate) | `True` |
| Rows scored | 520,391 |
| Negative predictions | 0 (physical floor active) |
| NaN / Inf predictions | 0 |
| Length match | `True` |
| Row/product/week alignment | `True` |
| Schema (93 cols, exact order) | `True` (93 cols) |
| MLflow signature == frozen schema | `True` |
| In-run `feature_columns.json` == schema | `True` |
| Inverse transform verified vs raw estimator | `True` (max abs diff 0.0) |

## 4. Post-hoc holdout metrics (unit space)

| Metric | Value |
|---|---|
| `wape` | 0.375373 |
| `mae` | 0.016554 |
| `rmse` | 0.185602 |
| `mase` | 0.435128 |
| `bias_pct` | -1.196961 |
| `asym21` | 0.025095 |
| `pos_wape` | 0.336382 |
| `B_wape` | 0.628239 |
| `B_bias_pct` | 9.851522 |
| `B_units` | -53.690795 |
| `B2_wape` | 0.556648 |
| `B2_bias_pct` | 0.861589 |
| `C_wape` | 0.365135 |
| `C_bias_pct` | -1.516463 |

Historical V1 context (already public in `part3_final_audit.json`, not re-scored here):

- V1 test WAPE `0.375395`, B WAPE `0.628350`, gate `False` -> `Staging`.

## 5. Quality gate (implementation unchanged)

- Gate: **FAIL** (group `post_hoc_holdout`)
- Resulting stage for version 2: **Staging**
- Failed criteria: `B_wape=0.628239 (bound 0.6)`

# Quality Gate Thresholds (Phase D)

Tolerance bands are set relative to the frozen EXP2 champion (val) and the persisted Part-3 winner reference. Global error bounds allow test-split variance without tolerating regression; cohort-bias bounds encode the business stockout-risk requirement (>= -20% on B/B2). Technical integrity is enforced as hard booleans.

| Criterion | Description | op | Bound | Reference (winner→EXP2) | Justification |
|---|---|---|---|---|---|
| wape | Global WAPE (primary quality metric, unit-independent) | le | 0.41 | 0.4001 (persisted winner) / 0.3964 (EXP2) | Frozen champion EXP2 = 0.3964 on val; winner reference 0.4001. A -2.5% band above reference (=0.41) rejects any regression worse than ~2.4% vs. the persisted production quality bar while absorbing test-split variance. |
| mae | Global MAE (unit sales) - interpretable absolute error | le | 0.025 | 0.0235 (winner) / 0.0233 (EXP2) | Reference MAE 0.0233-0.0235 units; 0.025 allows ~7% headroom for test-period noise yet rejects any accuracy collapse. |
| rmse | Global RMSE (heavy-tail penalty) | le | 0.3 | 0.2725 (winner) / 0.2592 (EXP2) | Reference RMSE 0.26-0.27; budget of 0.30 is ~+10% to keep outliers in check without over-fitting the threshold to one val snapshot. |
| mase | Global MASE (vs train 1-step naive denominator 0.03804) | le | 0.64 | 0.6179 (winner) / 0.6122 (EXP2) | Reference MASE ~0.61-0.62; 0.64 leaves ~3-4% band while preserving the 'well below naive (=1)' quality demonstration. |
| bias_pct | Global bias (forecast/actual - 1) in % | within | (-12.0, 12.0) | -9.6% (winner) / -6.15% (EXP2) | Reference bias -6..-9.6% (slight under-forecast). A symmetrical ±12% band bounds systematic under/over-forecasting - the business impact asymmetry is then handled by the 2:1 asym loss below. |
| asym21 | Asymmetric loss 2:1 (under-forecast penalised double) | le | 0.04 | 0.0381 (winner) / 0.0367 (EXP2) | Reference 0.037-0.038 cheap units; 0.04 is +~8% headroom; stricter than the raw pre-experiment score would already fail, guarding the stockout-cost asymmetry the metric encodes. |
| pos_wape | WAPE restricted to positive-demand rows (operational sellable demand) | le | 0.42 | 0.3721 (winner) / 0.3655 (EXP2) | Positive-demand error budget 0.42 vs reference 0.365-0.372; +13% band keeps sellable-demand accuracy near the champion. |
| B_wape | B cohort WAPE (sustained high-volume, the business-critical segment) | le | 0.6 | 0.5485 (winner) / 0.5102 (EXP2) | B cohort is the monetarily dominant segment. Reference 0.51-0.55; 0.60 bound rejects >~10% regression on high-volume products. |
| B_bias_pct | B cohort bias (%) - MUST NOT systematically under-forecast | ge | -20.0 | -30.0% (winner defect) / -18.2% (EXP2 fix) | Key business requirement distilled from the defect analysis: the production winner under-forecast B by -30%; EXP2 repaired it to -18.2%. The gate mandates >= -20% - i.e. the candidate MUST hold the fix and may not slip back towards the -30% stockout regime. |
| B2_wape | B2 cohort WAPE (upper-mid demand) | le | 0.65 | 0.5624 (winner) / 0.5557 (EXP2) | B2 reference 0.56; 0.65 band = +16% regression guard for this secondary near-mature cohort. |
| B2_bias_pct | B2 cohort bias (%) | ge | -20.0 | -28.0% (winner defect) / -18.1% (EXP2 fix) | Same stockout-risk requirement as B: the -28% production defect must stay fixed (>= -20%), matching the EXP2 result of -18.1%. |
| C_wape | C long-tail cohort WAPE (sparse, many zero-weeks) | le | 0.43 | 0.3807 (winner) / 0.3796 (EXP2) | Long tail (majority of 32.7k SKUs) must not be sacrificed for segment gains: 0.43 = +13% band over reference 0.38. |

Technical-integrity booleans (all must be True):
* `pred_integrity` - Predictions finite, non-negative, correct length.
* `schema_ok` - Model input schema == frozen 93-column signature (exact order).
* `model_load_ok` - Model artifact loads from MLflow.
* `pyfunc_predict_ok` - PyFunc predict returns pred_demand on unit scale.
* `signature_ok` - MLflow model signature recorded and valid.
* `artifacts_ok` - Model + feature_columns.json + config.json artifacts present in run.
* `tests_ok` - Regression suite green (Phase-G gate).

## 6. MLflow lifecycle update

- Registered model: **`demand_forecasting_final`**
- Model version: **2** (no new version created; no v3)
- Stage: **Production -> Staging** (transitioned: `True`)
- Gate decision tag: **FAIL**
- Tags applied to run `5d729a8594dc488bbd18ba0c4e216f5a` + version 2: `evaluation_type=post_hoc_holdout_evaluation`, `evaluation_dataset=test`, `lifecycle_stage`, `model_version`, `production_gate`, `test_start_date/test_end_date/test_rows`, `final_*` metrics, `posthoc_holdout_evaluated=true`.
- Historical records preserved: run/version tags `test_scored_once=false`, `evaluation_group=v2_validation`, `wape=0.3964` left untouched; V1 version 1 Staging + run `7fc87198...` + `part3_final_*` files untouched.

## 7. Statement

- No tuning, retraining, re-thresholding, or model changes were made before or after this evaluation.
- Result classification: **post-hoc holdout evaluation** - not an independent final-test estimate (see Disclosure).
- This is the final holdout evaluation step for Part 3; no further test evaluation is performed.
