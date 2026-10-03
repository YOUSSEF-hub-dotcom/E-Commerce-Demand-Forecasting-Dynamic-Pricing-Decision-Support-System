# Part-3 Final Lifecycle Report (frozen candidate: log-LGBM + EXP2)

* Generated: 2026-09-22T17:24:20.668625+00:00

## 1. Frozen candidate

- Feature version: `phase3-exp2-v1` (13 base + 73 one-hot + 7 regime = `93` cols)
- MASE denominator (train 1-step naive MAE): `0.038043`
- Schema frozen: 93 columns, order recorded in `feature_columns.json`.

## 2. Validation reproducibility check

- Validation WAPE: `0.3964` (EXP2 champion reference `0.3964`)
- MAE `0.0233` | bias `-6.15%` | MASE `0.6122`

## 3. FINAL TEST results (protected split, scored exactly once)

| Metric | value |
|---|---|
| `wape` | 0.375395 |
| `mae` | 0.016555 |
| `rmse` | 0.185603 |
| `mase` | 0.435154 |
| `bias_pct` | -1.199194 |
| `asym21` | 0.025097 |
| `pos_wape` | 0.336382 |
| `B_wape` | 0.628350 |
| `B_bias_pct` | 9.840414 |
| `B_units` | -53.630258 |
| `B2_wape` | 0.556710 |
| `B2_bias_pct` | 0.855382 |
| `C_wape` | 0.365155 |
| `C_bias_pct` | -1.518389 |

## 4. Automated quality gate (Phase D)

- Gate: **FAIL**
- Promotion target: **Staging**

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

## 5. Promotion + registry

Result versioned in the MLflow Model Registry (`demand_forecasting_final`).
## 6. Promotion result (real MLflow Model Registry)

- Registered model: **`demand_forecasting_final`**
- Model version: **1**
- Promotion stage: **Staging** (Production on gate PASS, Staging on FAIL)
- Source run id: `7fc871986e9240b1a4d98f7a9c1f893f`
