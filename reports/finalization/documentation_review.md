# FINAL DOCUMENTATION REVIEW

Status: **REVIEW COMPLETE** · Date: 2026-09-25 · Reviewer scope: read-only audit of the documentation layer against the actual repo (source, reports, tests, MLflow records, Part 3/4/5/6 reports, `requirements*.txt`, deployment files). **Only documentation was changed; no ML/model/deployment logic was touched.**

## Reviewed documents

| Document | Verdict |
|---|---|
| `README.md` | **PASS with corrections** (2 fixes) |
| `PROJECT_MAP.md` | **PASS with corrections** (4 fixes) |
| `docs/PROJECT_WALKTHROUGH.md` | **PASS with corrections** (4 fixes) |
| `docs/PROJECT_STORY.md` | **PASS with corrections** (2 fixes) |
| `docs/CV_DESCRIPTION.md` | **PASS with corrections** (2 fixes) |
| `docs/INTERVIEW_PREPARATION.md` | **PASS with corrections** (4 fixes) |
| `reports/finalization/finalization_report.md` | **PASS** (no changes) |
| `reports/forecasting/part1_3_data_feature_leakage_audit.md` | **PASS with correction** (1 fix) |
| `reports/deployment/part6_deployment_implementation_report.md` | **PASS with correction** (1 fix) |
| `PROJECT_BLUEPRINT.md` | **ABSENT** (referenced, not present; documented as a gap, not invented) |

---

## Audit items

### 1. Factual accuracy — number/window errors
1. **Dataset window misstated as "7 months" (WRONG, fixed).** The Olist window is 2016-09-04 → 2018-10-17 = **772 days ≈ 25.4 months**, not 7 months. Defect appeared in `docs/CV_DESCRIPTION.md` (one-line summary: "7 months of e-commerce data"), `docs/PROJECT_STORY.md` (30-second version), `docs/PROJECT_WALKTHROUGH.md` (§19 five-minute version). Corrected to "25 months (2016-2018)" in all three. `PROJECT_MAP` and `INTERVIEW` already stated 772 days — confirmed correct, unchanged.
2. **XGBoost "WAPE 0.4299" is not a WAPE (WRONG, fixed).** 0.4299 is actually the **OOS Spearman pred_resid vs resid correlation** (`pred_resid vs resid : +0.4299`), not an XGBoost WAPE. The true XGBoost numbers verified in `src/notebooks/03_demand_forecasting_engine.ipynb` (expanding temporal CV, train-only, 5 folds, 120 mature products, 8,899 rows) and `src/notebooks/05_forecasting_underforecast_deepdive.ipynb` (nb05 STEP 7 XGBoost counter-check): **CV mean-fold WAPE LGBM 0.5802 / XGB 0.5492; winner-decision XGB counter-check 0.4349**. All occurrences of "XGB 0.4299" (README, PROJECT_MAP, INTERVIEW, STORY, WALKTHROUGH, and `part1_3` audit report) corrected to the verified figures.
3. **Winner WAPE context fixed.** "LightGBM 0.4001 beat XGB on expanding-window CV" conflated the CV metric with the winner-decision metric. 0.4001 is the **full-train refit validation** WAPE used for the Phase-3 winner decision (reproduced in nb04/nb05/EXP0 control); 0.3964 is the frozen v2 final validation WAPE. Docs now state: CV (train-only): LGBM 0.5802 / XGB 0.5492; winner on refit validation: LGBM 0.4001 (final 0.3964) vs XGB counter-check 0.4349 vs naive persistence 1.4923.
4. **CV test-count math (WRONG, fixed).** `docs/CV_DESCRIPTION.md` claimed "45 fast + 8 integration + 282 broader pipeline tests". The correct, traceable decomposition is **290 pipeline + 45 deployment-contract fast = 335 passed (default)**, plus **8 real-model integration tests verified under `RUN_DEPLOYMENT_INTEGRATION=1`**. The "282" figure was not traceable; bullet rewritten with the verifiable 290/45/8 split.
5. **Suite runtime normalized (~7 min → ~6 min).** The finalization re-run in `.venv` took ~6 min (354 s). Earlier Part-6 runs were logged as ~7 min. All user-facing docs (`README`, `PROJECT_MAP` x2, `WALKTHROUGH`, `INTERVIEW`) plus `part6_deployment_implementation_report.md` normalized to "~6 min" to remove the conflicting figure.
6. **Baseline label precision (`INTERVIEW` memory sheet).** `naive 1.3783` → **`train-mean baseline WAPE 1.3783`** (verified in `02_temporal_preprocessing_and_target_engineering.ipynb`: "Baseline (train-mean): wape 1.3783"), distinct from the persistence baseline 1.4923.
7. **TECH_STACK visualization row.** `matplotlib, seaborn, plotly` → bordered: only `matplotlib` is imported in committed code (`src/explainability/shap_analysis.py`); `seaborn`/`plotly` are not used in any committed `.py` or notebook. Row updated + parenthetical note.
8. **TECH_STACK environment row.** Pinned `nbformat`/`nbconvert` versions were unverifiable and unpinned in `requirements*` (notebook deps gap, already a finalization limitation). Row now says `src/notebooks/` executed during development; `nbformat`/`nbconvert` not pinned.

### 2. Incorrect metrics
- All metric claims cross-checked against Part 3/4/5 reports + tracked `v2_validation` metrics + registry: **WAPE 0.3964, MAE 0.02329, RMSE 0.25917, MASE 0.6122, B WAPE 0.5101, B bias −18.2%, post-hoc global 0.3754, B 0.6282, elasticity −0.3952 (t −10.878, n 41,452), SHAP cat_unknown 0.0304, drift max 0.02697/mean 0.00263** — all **CONFIRMED**.
- The only metric error in the layer was item 1.2 above (0.4299) — now corrected.

### 3. Unsupported claims
- "282 broader pipeline tests" — unsupported, fixed (see 1.4).
- "7 months" — unsupported vs exploit 772-day window, fixed (see 1.1).
- XGB "WAPE 0.4299" — unsupported (it was a Spearman correlation), fixed (see 1.2).
- All other claims (gate policy history, registry versions/run ids, safety, test counts, elasticity/pricing statuses, SHAP budgets `bg 400 / explain 1,500` — verified in `part5_shap_explainability.md`) are traceable.

### 4. Duplicated / conflicting information
- Suite runtime conflict resolved (~6 min everywhere).
- `0.4001` vs `0.3964` now clearly distinguished (winner refit validation vs frozen v2 final) and consistent across README/MAP/INTERVIEW/STORY/WALKTHROUGH.
- CV context (expanding, 5 folds, min_train_weeks 24, horizon 4, 120 mature products, 8,899 rows, Optuna 10 trials/model, TPE, seed 42) consistent everywhere (nb03 confirms n_splits=5, horizon=4, min_train=24, 8,899 rows, 10 trials).
- Hyperparameters 600 trees / num_leaves 223 / lr 0.008468 consistent (final_execution_report + registry).
- Elasticity status distribution (Reliable 119 / Weak 1,589 / Insufficient 1,130 / Unsupported 16,261) — verified present in `part4_pricing_engine.json` summary block.

### 5. Architecture
- Flow diagram consistent: raw CSVs → fact table 112,650×23 → chronological split (train 2,428,492 / val 520,391 / test 520,391 = 3.47M matrix rows) → weekly matrix 32,729×106 → 93-col past-only schema (13 base + 73 OHE `cat_*` + 7 regime) → LightGBM → MLflow lifecycle → pricing → SHAP → deployment. All row counts verified.
- 71 native categories (nb02: `products=32951 translation=71`) vs 73 OHE columns consistent.

### 6. Production model / version
- Docs consistently state `demand_forecasting_final` **v2 Production, run 5d729a85, READY**, pyfunc-loaded, source `models:/m-e93cadaf963a4aa1b44b3e501379da92`, 93 inputs / 1 output. Verified against the live registry during finalization; unchanged.

### 7. Quality-gate interpretation
- Consistent: legacy policy 19 outcomes (12 numeric + 7 technical) vs revised 18 blocking + diagnostics; `B_wape` diagnostic, `B_bias_pct ≥ −20` hard; 19 → revised replay deterministic; v2 promoted via `REVISED_GATE_APPLIED`; guarded decision record; no V3; history preserved verbatim. No interpretive liberties found.

### 8. Accidental causal-elasticity / profit-uplift claims
- **None.** Docs consistently contain explicit anti-claims: no "increased profit/revenue", no "reduced cost"; the ~73% figure is always labeled **WAPE reduction vs the persistence baseline (1.4923 → 0.4001)**, never a business-dollar claim (CV_DESCRIPTION footer; WALKTHROUGH "Don't fabricate" section; INTERVIEW).
- Elasticity is consistently presented as an estimated statistical relationship (pooled product-FE), not a causal guarantee; pricing guardrails describe decision-support, not profit guarantees.

### 9. Protected Test exposure
- Docs correctly restrict claims: Test scored exactly twice historically (v1 + authorized v2 post-hoc), each once and guarded; `PART-6`/README report `test_used`/`test_scored_once`/`posthoc_holdout_evaluated` tags; no native access to Test in evaluation story; `test.csv` absent from Docker image (verified).

### 10. API contracts
- OpenAPI surface = exactly 5 routes (`/health /ready /predict /pricing /explain`); appended to `app/schemas.py` contract: `ProductParams(product_id 1–64, week optional date, extra="forbid")`, `PricingRequest(cost optional gt=0)`. All 17 canonical error codes in `app/errors.py` match the documented list (PRODUCT_NOT_FOUND, INVALID_WEEK, WEEK_OUT_OF_SUPPORTED_RANGE, TEST_BOUNDARY_VIOLATION, INVALID_COST, MODEL_NOT_READY, FEATURE_BUILD_FAILED, ELASTICITY_UNAVAILABLE, NO_ELASTICITY, NO_PRICE_OBSERVATION, PROFIT_NOT_COMPUTABLE, MODEL_LOAD_FAILED, DATABASE_UNAVAILABLE, VALIDATION_ERROR, INTERNAL_ERROR, PAYLOAD_TOO_LARGE, RATE_LIMITED) — **no corrections needed**.

### 11. Deployment instructions
- README/WALKTHROUGH/MAP: `uvicorn app.main:app`, `streamlit run dashboard/app.py`, `deployment/docker-compose.*`, `.env.example`, `docker compose up --build -d`. Cross-checked against repo files — all paths valid.

### 12. Outdated structure
- Repo structure in MAP/README matches current tree (no old `streamlit/` path remnants; `dashboard/` is correct).

### 13. Missing limitations
- Limitation sets (WALKTHROUGH §16, README, INTERVIEW, finalization report) unified: no untouched final holdout; dataset ends 2018; 106-week window; B-cohort weakness; elasticity statuses; pricing needs cost; e-commerce platform bias; **dev-layer reproducibility** (notebook deps unpinned) — all documented. PROJECT_MAP now also flags `PROJECT_BLUEPRINT.md` absence.

### 14. Missing technical decisions
- Key decisions documented: OHE over native categoricals (leakage-safety), log1p + expm1, `1+log1p` weights, train-only median imputation, expanding-window CV + Optuna confined to train folds, deterministic SHAP background, strategy of honesty for profit-not-computable, guarded exactly-once post-hoc. **No gaps found.**

### 15. Unclear Part explanation
- Part 1→6 explanation in WALKTHROUGH (§1–§19) consistent with the actual timeline and reports; each Part's deliverables, test counts, and progression (246 → 276 → 290 → 335+8) traceable.

### 16. Traceability of number sheet (INTERVIEW memory sheet)
- Every figure cross-checked (window, splits, features, hyperparams, validation/post-hoc metrics, elasticity, SHAP budgets, statuses, tests). Only two defects found (baseline label, CV/XGB cell) — both fixed.

---

## Corrections made (summary)

| Document | Change |
|---|---|
| `docs/CV_DESCRIPTION.md` | "7 months"→"25 months (2016-2018)"; test-count bullet → 290 pipeline + 45 fast + 8 gated integration |
| `docs/PROJECT_STORY.md` | "7 months"→"25 months"; Models bullet + 30-second version → XGB counter-check 0.4349 / winner refit validation 0.4001→0.3964 (blockquote indent fixed) |
| `docs/PROJECT_WALKTHROUGH.md` | "seven months"→"25 months (2016-2018)"; §3 candidate/CV block split into CV (0.5802/0.5492) vs refit-validation winner (0.4001/0.4349/1.4923); ~7→~6 min; §19/5-min narrative corrected |
| `README.md` | flow chip 0.4001 (CV-mislabeled) → refit-validation winner vs XGB 0.4349; ~7→~6 min |
| `PROJECT_MAP.md` | TECH_STACK Visualization + Environment rows; Temporal CV line (0.5802/0.5492 then winner 0.4001→0.3964/XGB 0.4349/naive 1.4923); XGB 0.4299 → 0.4349; ~7→~6 min (2 spots) |
| `docs/INTERVIEW_PREPARATION.md` | "0.4299"→counter-check 0.4349 (2 spots incl. memory sheet); "naive 1.3783"→"train-mean baseline 1.3783"; CV-picked-LightGBM nuance → CV narrowed family, winner on refit validation; ~7→~6 min |
| `reports/forecasting/part1_3_data_feature_leakage_audit.md` | "(0.4001 LGB / 0.4299 XGB)" → "(0.4001 LGB / 0.4349 XGB counter-check)" |
| `reports/deployment/part6_deployment_implementation_report.md` | full-suite runtime ~7→~6 min |

## Remaining issues

1. `PROJECT_BLUEPRINT.md` is referenced but **absent** from the repo. Kept as a documented gap (not fabricated). If a blueprint is wanted, it would be authored from `PROJECT_MAP.md` + finalization report, but it was intentionally not created as part of this review.
2. Notebook reproducibility: `nbformat`/`nbconvert`/plotly/matplotlib are not pinned in `requirements*`; committed notebooks contain no plotting imports (plots were produced during development). Documented as a limitation — no fix applied (would touch requirements).
3. `seaborn`/`plotly` appear only transitively under `.venv` (optuna/streamlit), not as direct deps — already reflected in corrected TECH_STACK row; no action needed.
4. Runtime: suite time varies by machine (~6 min in `.venv`; ~7 min recorded in earlier Part-6 runs). Docs normalized to ~6 min; remaining variance is environmental, not a doc error.

## Final documentation quality status

**PASS — documentation is internally consistent, traceable to the repo, and free of the previously identified factual errors.** All user-facing documents (README, PROJECT_MAP, docs/*) now match the verified source of truth (reports, notebooks, registry, error codes, tests). The three substantive defects found — a misstated dataset window, an incorrect test-count decomposition, and a misattributed XGBoost metric (0.4299, actually a Spearman correlation) — were corrected in documentation only. No ML/model/deployment logic was modified.