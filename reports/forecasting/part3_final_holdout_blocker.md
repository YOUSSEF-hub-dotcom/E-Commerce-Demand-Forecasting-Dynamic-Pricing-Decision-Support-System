# Part-3 Final Holdout Audit — BLOCKER REPORT

* Generated: 2026-09-23T00:42:00+00:00
* Status: **STOP — no genuinely untouched final holdout exists and none can be created without violating data-integrity rules.**

---

## 1. What was audited (exactly)

| # | Audit item | Source of truth |
|---|---|---|
| 1 | Raw demand-bearing data extent | `data/raw/olist_orders_dataset.csv` + `data/raw/olist_order_items_dataset.csv` (joined, status policy applied) |
| 2 | Dense demand matrix extent | `data/processed/demand_matrix_weekly.csv.gz` |
| 3 | Split definitions | `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv` |
| 4 | Split logic / no-leakage contract | `src/data/temporal_split.py`, `src/models/final_forecast.py:run_frozen_split` |
| 5 | v1 lifecycle | `reports/forecasting/part3_final_*.json`, MLflow run `7fc87198...` |
| 6 | v2 lifecycle | `reports/forecasting/part3_v2_*.json`, MLflow run `5d729a85...` |
| 7 | Quality gate | `src/models/quality_gate.py` (unchanged thresholds) |
| 8 | MLflow registry | `artifacts/mlflow_tracking.db` (versions/stages/tags) |

## 2. Answering the ten audit questions

1. **Rows used for v1 training:** `data/splits/train.csv` — weeks `2016-08-29/2016-09-04` → `2018-01-29/2018-02-04` (75 distinct week labels, 2,428,492 rows).
2. **Rows used for v1 validation:** `data/splits/val.csv` — weeks `2018-01-29/2018-02-04` → `2018-05-21/2018-05-27` (17 week labels incl. boundary overlap with train at `2018-01-29/2018-02-04`; 520,391 rows).
3. **Rows used for the original v1 test:** `data/splits/test.csv` — weeks `2018-05-21/2018-05-27` → `2018-09-03/2018-09-09` (16 week labels incl. boundary overlap with val at `2018-05-21/2018-05-27`; 520,391 rows). **This IS the final 16 weeks of the entire dataset.**
4. **Rows used for v2 training:** identical TRAIN split (v2 = frozen candidate re-trained on TRAIN only).
5. **Rows used for v2 validation:** identical VALIDATION split (v2 scored VALIDATION only; test untouched).
6. **Genuinely untouched future temporal data beyond the original test period:** **NONE.**
   * Raw orders span `2016-09-04` → `2018-10-17`, but the ONLY orders after the test end (`2018-09-10` → `2018-10-17`) are **13 orders, all with `order_status = canceled` and ZERO `order_items`** → zero demand.
   * Last economically-valid order timestamp: `2018-09-03 09:06:57` → falls in week `2018-09-03/2018-09-09` = the LAST week of the original test.
   * The dense demand matrix max week = `2018-09-03/2018-09-09` exactly matches the last valid order's week (matrix not truncated).
7. **Can a new independent temporal holdout be created without modifying raw data?** **NO.**
   * Carving from `train`/`val` would reuse data already exposed to the entire development cycle (feature engineering, Optuna tuning, Error Analysis, cohort definitions, v2 development and its validation gate) → NOT untouched, and would silently change the original split definitions (explicitly prohibited).
   * Reusing the original `test` as an "independent" holdout is explicitly prohibited (already scored by v1).
   * No additional demand-bearing rows exist anywhere in `data/raw` or `data/processed`.
   * a) train 75 + b) val 17 + c) test 16 = 108 week-labels but 106 distinct weeks (two boundary weeks are shared labels, not overlap), and the union of split weeks == matrix weeks exactly (106/106). Row counts: 2,428,492 + 520,391 + 520,391 = 3,469,274 == matrix rows. **The existing splits partition 100% of the dataset.**

## 3. Why this is a blocking condition (per the task rules)

* "If the repository does not contain a genuinely untouched final holdout that can be used
  without compromising data integrity, **STOP and report the blocker** instead of inventing or
  corrupting data."
* "Do NOT silently change the original train/validation/test definitions."
* "Do NOT pretend that this same test is an untouched test for v2."

Creating the "new" final holdout would require either (a) reusing the scored test,
(b) reusing dev/val data that was already used for model selection and Error Analysis
(violating "The final holdout must be unseen during model development"), or (c) inventing
data. All three violate the mandatory data-integrity rules. Therefore the correct action is
to STOP and report this blocker.

## 4. Verified evidence (machine-readable)

See `part3_final_holdout_blocker.json` for:

* raw order extent (min/max timestamps, valid-order max week, post-test order counts/status),
* matrix extent (106 weeks; min/max week),
* exact split boundaries + distinct-week coverage + row-count identity with the matrix,
* boundary-week sharing explanation,
* MD5/SHA256 hashes of raw CSV files, split CSVs, and the demand matrix (raw provenance),
* MLflow registry snapshot (v1 Staging FAIL | v2 Production PASS-on-validation).

## 5. What was NOT done (honesty statement)

* The original v1 test split was NOT re-scored, modified, or reinterpreted.
* v2 was NOT re-trained or tuned; no new FINAL TEST metrics were produced (there is no valid
  holdout to produce them on).
* The final production gate was NOT run against anything (it requires final-test metrics,
  which do not legitimately exist).
* The MLflow registry was NOT changed: v2's `Production` status remains **provisional** —
  granted solely by the earlier validation-based gate and is explicitly flagged as such. It
  is NOT a claim of final test-validated production.
* No raw data, split file, or matrix was modified.

## 6. Recommended path forward (blocked on data)

The only scientifically valid way to complete the final lifecycle is to obtain genuinely
new, future demand-bearing data (i.e., a later temporal window from the business) that was
never part of any development decision. Until such data exists, the final production gate
cannot be executed, and v2 must remain flagged as validation-gated (provisional Production
per earlier decision, or Staging per strictest reading of the lifecycle rules — see audit
notes). If new data becomes available, the frozen v2 artifact can be scored on it exactly
once and the gate run with unchanged thresholds.