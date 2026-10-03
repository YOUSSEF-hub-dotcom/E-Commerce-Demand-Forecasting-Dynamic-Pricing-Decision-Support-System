# Part 3 — Pre-Final-Test Cleanup & Hardening

- **Status:** COMPLETE — cleanup delivered in full; FINAL Test / Final Evaluation / Final Production Quality Gate remain PENDING and were explicitly NOT touched (dataset ends `2018-09-09`; no untouched final holdout exists).
- **Date (UTC):** 2026-09-23
- **Base commit of analysis:** the preserved frozen v2 candidate is byte-identical before/after this cleanup (see §3).

---

## 1. Scope

Cleanup/hardening pass resolving the five audit findings (M1–M5) from
`reports/forecasting/part1_3_data_feature_leakage_audit.md`, executed **before**
the separate Final Test lifecycle. It is a hardening pass only — no retraining,
no Optuna, no feature selection, no recalibration, no threshold changes, and no
use of the protected Test split.

### 1.1 Changes made (4 code/artifact files + 1 new doc + 1 map update)

| # | File | Change |
|---|---|---|
| 1 | `artifacts/phase3/feature_columns.json` | Regenerated from the authoritative frozen v2 schema: **86 → 93 cols**. Names + order now byte-identical to the MLflow in-run artifact, the registered model's MLflow signature, and a train-split-reconstructed layout. |
| 2 | `src/models/final_forecast.py` | `prepare_frozen` one-hot vocabulary now derived from **train only**; raises `RuntimeError` if val/test expand the train vocabulary. Frozen schema still 93 cols; **no behavioral change** (train vocab == former concatenated vocab). |
| 3 | `src/models/data_prep.py` | `prepare_model_matrix` hard guard: when a frozen schema is supplied, the encoded block can never gain columns beyond it (unseen labels must route to `unknown` before OHE). Docstring updated to train-only semantics. |
| 4 | `src/data/temporal_split.py` | Boundary-week convention documented (shared week labels train/val & val/test are intentional, not leaks); `assert_chronological_order` doc updated to the inclusive `<=` rationale. |
| 5 | `src/models/finalize_lifecycle.py` | Early-stopping-vs-frozen-refit and Optuna-subset differences documented. |
| 6 | `src/models/optuna_tuning.py` | Optuna tuning-subset design documented (120 products / 8,899 rows / 5 folds / horizon 4 / TPE seed 42 / 10 trials). |
| 7 | `tests/test_prefinal_cleanup.py` | **New** — 17 guards (schema 4-way equivalence, OHE train-only, boundary weeks, frozen-v2 integrity). |
| 8 | `PROJECT_MAP.md` | Status block updated (cleanup COMPLETE, 125 tests, M1 **Resolved**, unresolved list cleared). |

### 1.2 No-change confirmations (strictly untouched)

- Raw data, demand values, timestamps, split boundaries: **untouched**.
- Target & `log1p`/`expm1` transform: **unchanged**.
- Model architecture (LightGBM) and frozen v2 hyperparameters (`n_estimators=600`): **unchanged** (verified equal to run `5d729a85...` freeze config).
- No Optuna run, no feature added/removed, no recalibration, no v3.
- Quality Gate thresholds: **unchanged**.
- Protected Test split: **not read, not scored, not consumed** by this cleanup.
- v1 artifacts / Staging FAIL result: **unchanged**; no promotion/demotion performed.
- Existing validation results: reproduced / NOT recomputed into new numbers (module constants still `EXP2_VALIDATION_WAPE = 0.3964`).

---

## 2. What the audit found → what this cleanup did

| Audit finding | Severity | Cleanup action | Verification |
|---|---|---|---|
| M1 `feature_columns.json` = 86 (minor schema drift) | Medium | Regenerated to 93 | `==` MLflow in-run artifact (93), signature (93), train-reconstructed layout (93), audit `schema_columns` (93) |
| OHE layout read all splits (union) | — | Train-only vocabulary + no-expansion assertions | `prepare_frozen` learns vocab from train; rejects val/test extras; block cannot escape frozen schema |
| Shared boundary week-labels | — | Documented + tested | train/val `2018-01-29/2018-02-04`; val/test `2018-05-21/2018-05-27` |
| M5 Early stopping vs frozen refit | Minor | Documented | notebook ES (train-tail) vs frozen full-600 refit — same WAPE 0.3964 |
| Optuna 120-product subset | — | Documented + tested | subset = train-only; winner frozen, refit on full train |

---

## 3. Verification summary

### 3.1 Frozen schema — 4-way equality (all 93, exact order)

| Source | Cols | Equality |
|---|---|---|
| `artifacts/phase3/feature_columns.json` (updated) | 93 | — |
| MLflow in-run `mlruns/2/5d729a.../artifacts/feature_columns.json` | 93 | **PASS** |
| Registered model `m-e93cadaf...` MLflow signature inputs | 93 | **PASS** |
| Train-split reconstructed layout (`FEATURE_COLS` + train-only `cat_*` + `REGIME_COLS`) | 93 | **PASS** |
| `reports/forecasting/part3_v2_audit.json` `schema_columns` | 93 | **PASS** |
| `freeze_schema(prepared["train"])` on rebuilt cached preparation | 93 | **PASS** |

Layout: first 13 = `FEATURE_COLS`, next 73 = `cat_*` (incl. `cat_unknown`), last 7 = `REGIME_COLS`.

### 3.2 OHE train-only equivalence proof (why this changes nothing measurable)

| Check | Result |
|---|---|
| train-only vocab size == prior concatenated-train+val+test vocab size | 73 == 73 |
| set equality (train vocab vs global vocab) | `True` |
| val categories not in train | `[]` |
| test categories not in train | `[]` |
| `unknown` present in train vocab | `True` |
| column order identical | `True` |
| val aligned to train columns | `True` |
| base design cols (13 + 73) | 86 → final frozen incl. regime = **93** |

### 3.3 Frozen v2 candidate — untouched

- `n_estimators` = 600, feature_version `phase3-exp2-v1`, params equal to v2 audit `freeze_config` — **PASS**.
- Registered model `MLmodel` artifact hash unchanged across cleanup — **PASS**.
- No new MLflow run, registration, stage transition, or artifact overwrite created.

### 3.4 Tests

| | Count |
|---|---|
| Baseline before cleanup | 108 collected |
| New cleanup guards (`test_prefinal_cleanup.py`) | 17 |
| Final full suite | **125 passed**, 0 failed |

Sections covered: schema equivalence (4), OHE train-only (5), boundary-week convention (3), frozen-v2 integrity (5).

---

## 4. Remaining known limitation (unchanged — not created by this cleanup)

**FINAL independent holdout remains BLOCKED.** The demand-bearing raw data ends at
`2018-09-09` — exactly the original test boundary — so no genuinely untouched
future segment exists from which an independent holdout could be built without
re-using development or test data. Full rationale: `reports/forecasting/part3_final_holdout_blocker.md`
and `tests/test_final_holdout_integrity.py`. This cleanup does **not** attempt the
Final Test: that lifecycle remains a manual, separately-gated step once genuine
future data (or an owner-approved protocol) is available.

---

## 5. Next steps (outside this cleanup's scope)

1. **Final Test** — run `finalize_lifecycle` once a genuine untouched holdout is available (currently BLOCKED).
2. **Final Evaluation** — score the frozen v2 candidate on that holdout using the existing Quality Gate (thresholds unchanged).
3. **Final Production Quality Gate** — promotion decision per thresholds.