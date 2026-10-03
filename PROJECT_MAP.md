# PROJECT_MAP — E-Commerce Dynamic Pricing & Demand Forecasting Engine

Canonical tracking document for architecture, system flow, and directory/phase progress.
Updated after every completed pipeline stage.

> **SOURCE OF TRUTH:** `PROJECT_BLUEPRINT.md` (root) governs all technical, statistical,
> MLOps, dynamic pricing, and explainability specifications. This map mirrors live state.
> **Finalization audit note:** `PROJECT_BLUEPRINT.md` is referenced but **does not exist
> in the repository** (absent since before the finalization audit; it never materialized).
> The authoritative status/map is this file; factual specifications and results live in
> `reports/` and the documentation layer under `docs/`.
> **Current state:** this map describes the repository as it exists now. Parts 1–6 are
> complete; the per-phase narrative below is retained as provenance, not as pending work.
> **Approval gate (historical):** no phase advanced before its code + tests were executed,
> verified, and approved. Phases 1–6 have all cleared it.

---

## TECH_STACK

| Layer | Tools |
|---|---|
| Data & computation | `pandas`, `numpy` |
| Visualization | `matplotlib` (SHAP/report figures; `seaborn`/`plotly` not imported in committed code) |
| ML & pipeline | `scikit-learn`, `lightgbm` 4.7.0, `xgboost` 3.2.0, `optuna` 5.0.0 |
| Model explainability | `shap` |
| Model lifecycle / deployment | `mlflow` 3.9.0 (SQLite tracking backend), `FastAPI` 0.128 + `uvicorn` (single worker), `SQLAlchemy` + `PostgreSQL 16` (inference log), `streamlit` 1.54 UI, `Docker` + `docker compose` |
| Environment | Python 3.11.8 (Windows), Jupyter (`src/notebooks/`, executed during development; `nbformat`/`nbconvert` not pinned in requirements) |

## SYSTEM_FLOW

```
Raw CSVs (data/raw/)
   │  read-only ingestion
   ▼
Stage 1: Data Collection, Merging & Structural EDA        ◄── complete
Stage 2: Temporal Preprocessing & Target Engineering      ◄── complete
   │  → item-level fact table (112,650 x 23)
   │  → temporal split train/val/test (no shuffling — anti-leakage)
   ▼
Stage 3: Demand Forecasting Engine (supervised time-series regression) ◄── complete + lifecycle
   │  target: order_demand (weekly aggregated quantity sold, log1p model scale)
   │  models: LightGBM (winner) & XGBoost, Optuna tuned, MLflow PyFunc contract
   │  metrics: WAPE (primary), MAE, RMSE, Asymmetric Loss (2:1)
   │  official workflow: Data → Train/Validation → expanding-window Cross Validation → Model Selection → Optuna Tuning → Final Test Evaluation → Quality Gate → Production
   │  current state: v2 (inverse_log1p floor contract) = **Production** (run 5d729a85) under quality_gate_policy_v2_revised (REVISED_GATE_APPLIED, same run, no V3); v1 = **Staging** with its FAIL preserved verbatim.
   │  completed validation-only experiments and the Final Test Evaluation record are retained as provenance only — see HISTORICAL PROVENANCE. None of them is a current workflow stage.
   ▼
Stage 4: Dynamic Pricing Optimization Engine (PED decision simulation) ◄── implemented
   │  output: revenue-optimal scenario per product; an optimal_price for maximum net
   │  profit is returned only when a cost source is supplied (none exists in this data)
   │  profit maximization needs a COST source — none exists in raw data (Phase-4.0 audit):
   │  without costs the engine returns revenue-optimal scenario flagged PROFIT_NOT_COMPUTABLE.
   │  elasticity = descriptive demand response (log-log, product+week FE); no causal claim.
   ▼
Stage 5: SHAP Explainability & Model Interpretation ◄── implemented (Part 5)
   │  explains Production model models:/demand_forecasting_final/2 (read-only, no retrain/V3,
   │  Test split never accessed) — log1p contributions, per-row additivity-corrected base
   │  (LightGBM bin-quantization drift measured + documented), global + local artifacts.
   ▼
Stage 6: Production Deployment (MLflow / FastAPI / Docker)     ◄── implemented (Part 6)
   │  FastAPI business-level API (product_id / ISO week / optional cost) serving the
   │  immutable Production model m-e93cadaf963a4aa1b44b3e501379da92 (v2, run 5d729a85,
   │  READY) via a registry-verified portable artifact-dir load; endpoints /health /ready
   │  /predict /pricing /explain; PostgreSQL inference log; streamlit dashboard; single
   │  Python image + docker compose (api 8000 / dashboard 8501); protected splits never shipped.
```

## DIRECTORY STRUCTURE (finalized — production & MLOps layout)

```
E-Commerce/
├── data/
│   ├── raw/                   # UNMODIFIED Olist CSVs (filesystem READ-ONLY, ingestion only)
│   ├── processed/             # Filtered, cleaned & aggregated datasets
│   └── splits/                # Train / Validation / Test temporal splits (protected)
├── app/                       # FastAPI deployment service (business-level API, inference-log DB)
├── dashboard/                 # Streamlit manager UI (app + product_catalog + pricing_presenter)
├── deployment/                # Dockerfile, docker-compose.yml, healthcheck.py (deployed source)
├── src/
│   ├── notebooks/             # Exploratory EDA & research (6 notebooks + 3 helper .py) — never a runtime surface
│   ├── data/                  # Ingestion, cleaning, aggregation, temporal split scripts (.py)
│   ├── features/              # Lag features, rolling statistics, encoding pipelines (.py)
│   ├── models/                # Training pipelines, Optuna tuning, lifecycle, evaluation scripts (.py)
│   ├── explainability/        # SHAP interpretation: global/local attributions, artifacts, report (.py)
│   ├── optimization/          # Dynamic pricing, elasticity simulation, profit guardrails (.py)
│   └── utils/                 # Logger setups, custom losses (WAPE/Asymmetric), helpers (.py)
├── artifacts/                 # Saved models, MLflow artifacts, SHAP diagnostic outputs, tracking DB
├── mlruns/                    # MLflow run/artifact store (12 experiment dirs + 2 model dirs)
├── mlflow.db                  # MLflow tracking DB (root copy; artifacts/mlflow_tracking.db is the pipeline copy)
├── scripts/                   # Dev-only utilities (verify_imports.py) — never imported by the app
├── docs/                      # Public technical documentation
│   └── private/               # Private study / interview material (git-ignored, not published)
├── reports/
│   ├── forecasting/           # Per-part evidence: leakage audit, model selection, gate, B-cohort, pricing, SHAP
│   ├── finalization/          # Finalization + pricing-manager-ready audits (12 md)
│   ├── deployment/            # Part-6 deployment architecture audit + implementation report
│   └── experiments/           # Controlled-experiment reports (4)
├── tests/                     # Automated unit & integration regression tests (pipeline, 275)
│   ├── deployment/            # Part-6 deployment test suite (47 fast + 8 gated integration)
│   └── dashboard/             # Dashboard catalog / presenter / selector UI tests (72)
├── .env.example               # Documented settings template (copy to .env)
├── .dockerignore              # Build-context boundary for the image
├── .gitignore                 # Version-control exclusions
├── requirements.txt           # Runtime dependency pins (Docker + venv contract)
├── requirements-dev.txt       # Dev layer: pytest (+ runtime pins) for a clean install
├── README.md                  # Public technical overview: architecture, install/run/test, limitations
└── PROJECT_MAP.md             # Live architecture tracking & system flow synchronization
```

### Folder responsibilities & boundaries
| Folder | Boundary / Responsibility |
|---|---|
| `data/raw/` | Canonical source of truth. **Strictly read-only** (NTFS ReadOnly enforced); never mutated by pipelines. |
| `data/processed/` | Intermediate cleansed & aggregated tables — produced by `src/data/`, consumed by features. |
| `data/splits/` | Chronological train/val/test splits — the anti-leakage contract; split artifacts persisted here. |
| `src/notebooks/` | Exploration & research only. Never a deployment surface; findings are promoted into `src/*` modules. |
| `app/` | Part-6 FastAPI service: business-level endpoints, AppState model/feature/elasticity/SHAP loading, PostgreSQL inference log, structured error contract, middleware (request-id/logging/payload-limit/rate-limit). |
| `dashboard/` | Part-6 Streamlit manager UI. Named `dashboard/` (not `streamlit/`) so it never shadows the framework. Panels consume the business contract (product ID / week / cost) via the API, and the catalog additionally reads the weekly demand matrix + `reports/forecasting/part4_pricing_engine.json` from disk for product browsing, labels, ordering and the pricing-readiness tiering. Those reads mirror the API's valid-price rule and duplicate no estimation logic; the API stays the authority. Never the 93-feature schema. |
| `deployment/` | Part-6 container packaging, and the source of the deployed image: `Dockerfile` (single image for api + dashboard), `docker-compose.yml` (db/api/dashboard), `healthcheck.py` (copied into the image at `deployment/healthcheck.py`; backs the `/ready` and `/_stcore/health` probes). |
| `src/data/` | Production ingestion/merge/clean/temporal-split modules (`.py`). |
| `src/features/` | Feature engineering pipelines: lags, rolling stats, encoders (fit on train only). |
| `src/models/` | Training, Optuna hyperparameter tuning, model evaluation. |
| `src/optimization/` | Stage-4 dynamic pricing: elasticity simulation, optimizer, price guardrails. |
| `src/utils/` | Cross-cutting: logging config, WAPE/Asymmetric custom losses, shared helpers. |
| `artifacts/` | Trained models, MLflow run artifacts, SHAP outputs — reproducible, shipped to registry. |
| `tests/` | Unit & integration regression tests guarding pipeline contracts. |

## TEST SUITES

| Suite | Scope | Status |
|---|---|---|
| `test_metrics.py` | WAPE / MAE / RMSE / Asymmetric loss; zero-sales & zero-division guards | ✅ 14 passing |
| `test_target_engineering.py` | Demand target aggregation, week grain, zero-sales, missing-category edges | ✅ 12 passing |
| `test_temporal_split.py` | Chronological order, no overlap, full coverage, determinism | ✅ 8 passing |
| `test_data_processing.py` | Raw→fact-table flow, schema contract, train-only imputation, **zero-leakage probe** | ✅ 12 passing |
| `test_features.py` | Phase-3 feature stack: lags 1/2/4, rolling mean/std, drivers, calendar, boundary-window | ✅ 9 passing |
| `test_forecasting_models.py` | Model factories, naive baseline, expanding-window CV, Optuna objective, sample weights, schema alignment, MLflow PyFunc roundtrip | ✅ 17 passing |
| `test_quality_gate.py` | Multi-condition gate (+ policy versioning `quality_gate_policy_v1_legacy` default / `quality_gate_policy_v2_revised`): 12 numeric bounds + 7 technical booleans, PASS→Production / FAIL→Staging, `B_wape` diagnostic under revised policy, markdown audit for both policies | ✅ 10 passing |
| `test_registry.py` | MLflow tracking + Model Registry lifecycle: run logging, PyFunc reload, version registration, stage transition, audit JSON, v2 tag-override & version-preservation | ✅ 9 passing |
| `test_prediction_integrity.py` | `inverse_log1p` inference contract: non-negative, finite, consistent preds across model/wrapper/lifecycle | ✅ 10 passing |
| `test_final_holdout_integrity.py` | Final-Test data-boundary guards: raw data ends at the test boundary, splits partition the matrix exactly, `test.csv` provenance hash pinned, v1 FAIL preserved, v2 not claimed test-validated, dataset-end blocker report present | ✅ 7 passing |
| `test_prefinal_cleanup.py` | Pre-final-test cleanup & hardening: 4-way frozen-schema equivalence (artifact == MLflow in-run == registered signature == reconstructed layout), OHE train-only vocabulary + no-expansion guards, boundary-week convention, frozen-v2 integrity vs audit/MLflow | ✅ 17 passing |

| `test_b_cohort_diagnostics.py` | B-cohort root-cause diagnosis contract: verbatim Test Protection Statement embedded in JSON + MD, exact reproduction of V2 validation references (global 0.3964, B 0.5101460932701322, bias −18.227), B profile pins (37 product / 589 rows), module NEVER reads `splits/test.csv` (path only inside disclosure), frozen V2 loaded via `models:/demand_forecasting_final/2`, official `cohort_flags` + gate constants unchanged, all report sections present | ✅ 10 passing |
| `test_b_cohort_remediation.py` | Part-3 controlled B-cohort experiment contract (validation-only): module never reads the test split and has no CLI/`--force`; control arm = frozen 93-col `1+log1p` V2 replica reproducing global WAPE 0.3964 / B WAPE 0.5101460932701322 / bias −18.227% (+MASE denom pin); Arm 3 adds exactly the 4 pre-registered spike features (schema 93→97) whose toy-frame features depend only on strictly-prior rows (mutation-invariance, no future→train or future→val leakage, row-order preserved); Arm 2 weight formula pre-registered, deterministic, bounded [1, cap], zeros=1, train-only signature; official `cohort_flags` + B/B2 gate constants pinned; every pre-registered decision criterion checked and the delivered `NO_VALID_IMPROVEMENT` verdict matches observed metrics (Arm 3 fails only the C WAPE +0.01 tolerance); all arms pass prediction integrity with the full required metric suite; registry v1/v2 stages + run_ids untouched live, module has no `register_model`/`transition_model_version_stage`/`log_model` path, the recorded Final Test Evaluation result (B WAPE=0.628239) preserved verbatim; dedicated additive MLflow experiment `part3_b_cohort_remediation_exp` (3 FINISHED tagged runs, `test_used=false`, `validation_only`), no remediation noise in the lifecycle experiment | ✅ 29 passing |
| `test_b_cohort_asymmetric_objective.py` | Part-3 controlled asymmetric 2:1 training-objective experiment contract (validation-only): module never reads the test split, no CLI/`--force`, no registration/stage-transition/log-model path; control arm = exact frozen V2 replica reproducing global WAPE 0.39637084 / B WAPE 0.5101460932701322 / bias −18.227% (+MASE denom pin); objective correctness chain — `QUANTILE_ALPHA == 2/(2+1)`, under/over ratio exactly 2, LGBM built-in `quantile` accepted by installed LightGBM (objective_ pin), higher-alpha monotone upper shift, and the elementwise identity `asymmetric_loss(2,1) == 3 * pinball_(2/3)` (+ `asym21` == recorded `asymmetric_loss_2_1_direct`); treatment changes ONLY the objective (identical param dict otherwise, 93 cols, 2,428,492/520,391 rows); every pre-registered C1–C8 criterion checked and the delivered `NO_VALID_IMPROVEMENT` verdict matches observed metrics (global WAPE improves to 0.3665, B bias −18.2%→+1.5%, asym21/C improve, but B WAPE 0.5564 and B2 0.5972 FAIL C2/C5); all arms pass prediction integrity (n_neg 0, n_nan_inf 0); registry v1/v2 stages + run_ids untouched live (`7fc87198`/`5d729a85`), the recorded Final Test Evaluation result (B WAPE=0.628239) preserved verbatim, dedicated additive MLflow experiment `part3_b_cohort_asymmetric_exp` (2 FINISHED tagged runs `arm1_v2_control`/`arm2_asym_quantile_alpha067`, `test_used=false`, `validation_only`) with run_ids matching the JSON report | ✅ 30 passing |
| `test_gate_policy_revision.py` | Formal V2 quality-gate policy revision contract: policy versioning (legacy default 19 outcomes / revised 18 blocking + diagnostics), revised gate keeps `B_bias_pct >= -20` + all criteria hard and only moves `B_wape` to diagnostic, gate defaults to legacy (backward compatible), unknown-policy rejected; persisted-source replay only (module has no Test/split/scoring import path), deterministic legacy replay byte-identical to the persisted Final Test Evaluation FAIL (single `B_wape` failure), decision `REVISED_GATE_APPLIED`/`PRODUCTION`; no `--force` (no argparse/CLI), additive-only revision tags (never touch `production_gate`/`gate_decision`/`lifecycle_stage`/`evaluation_group` etc.), exactly-once marker guard, revision audit JSON content, live registry v2 → **Production** (same run `5d729a85`, v1 Staging untouched, no v3), required revision + preserved historical tags on run/version | ✅ 21 passing |
| `test_lifecycle_router.py` | Part-3 automation-audit gap closure: lifecycle operations route through one decision module (gate → stage → registry → report), no Test/split access, no `--force`, HTML artifact routing, idempotent report write | ✅ 16 passing |
| `test_pricing_data.py` | Part-4.0 data surface: matrix/week accessors, reference-week invariant (< test-period guard, anti-leakage), ceiling/floor price bounds, cost-source presence probe | ✅ 5 passing |
| `test_elasticity.py` | Part-4.1 elasticity: `integrate_elasticity`, log-log OLS via `np.linalg.pinv`, pooled product/product+week FE (within-demeaned, `pinv(design)`), per-product confidence + segmentation status (`Reliable`/`Weak`/`Insufficient`/`Unsupported`), validation sign/magnitude check, `nice_elasticity` | ✅ 3 passing |
| `test_simulation.py` | Part-4.3 PED-doubling scenario simulation over reference-week scope, extrapolation guard outside observed price range | ✅ 6 passing |
| `test_optimizer.py` | Part-4.4 profit optimizer incl. `analytic_optimal_price` closed-form sanity, `None`-or-NaN profit guards, threshold coverage, cost-gated profit computation | ✅ 5 passing |
| `test_guardrails.py` | Part-4.5 configurable guardrails (margin floor, increase cap, absolute floor/ceiling, smoothing) — applied-flag correctness, transparent no-guardrail default, invalid-config rejection | ✅ 7 passing |
| `test_pricing_engine.py` | Part-4.6 engine contract: end-to-end orchestration on synthetic matrix, 93-col frozen schema, `MODEL_URI`/cost/pricing-data wiring, markdown + JSON report schema, test-period assertion, `test_split_accessed=False`, AST anti-leakage (no `test.csv` read), JSON-safe serialization | ✅ 4 passing |
| `test_shap_explainability.py` | Part-5 SHAP contract: Production model/registry identity, AST anti-leakage (no `test.csv`), reference-week guard, schema order + no identity cols, SHAP dims/finite/exact per-row reconstruction, deterministic sampling, local examples, reproducibility, artifact existence/readability + importance-CSV parity, price features present | ✅ 14 passing |
| `tests/deployment/` | Part-6 production-deployment contract (55 tests = 47 fast + 8 real-model integration, see `PART-6 DELIVERABLES`): API/OpenAPI surface, validation & error codes, persistence into the inference log, readiness gating, registry read-only + artifact-portable loader + path-normalized fallback parity, docker hygiene (no protected splits in image), streamlit UI never references feature names | ✅ 47 passing + 8 gated integration |
| `tests/dashboard/` | Dashboard presentation-layer contract (72 tests): `test_product_catalog.py` (catalog loaders, week scopes, pricing-readiness states, tier counts, product statuses), `test_pricing_presenter.py` (pricing-opportunities rendering incl. unavailable state), `test_product_selector_app.py` (searchable selector + week selector) | ✅ 72 passing |

> **Current suite:** **402 collected** = 275 pipeline + 55 deployment (47 fast + 8 real-model integration, gated) + 72 dashboard. Last full run: **393 passed, 0 failed, 9 skipped** in ~10.6 min. Skips = 8 integration-gated (`RUN_DEPLOYMENT_INTEGRATION=1`) + `test_marker_records_policy_and_version` (skips when the local gate-revision marker file is absent). Suite growth history: Phase 3 → 230 tests · Part 4 → 276 · Part 5 → 290 · Part 6 → 335 · dashboard + safety work → 417 · obsolete-audit test removal → 402.

## PHASE PROGRESS

| Phase | Status | Artifacts |
|---|---|---|
| 1. Data Ingestion, Merging & Structural EDA | ✅ Complete | `01_initial_data_merging_and_eda.ipynb` |
| 2. Temporal Preprocessing & Target Engineering | ✅ Complete | `02_temporal_preprocessing_and_target_engineering.ipynb` · `src/data/*` · `src/features/feature_engineering.py` · `src/utils/{logging_setup,metrics}.py` · `tests/test_*.py` · `data/splits/*` |
| 3. Demand Forecasting Engine | ✅ **COMPLETE** — official workflow executed: Data → Train/Validation → expanding-window Cross Validation → Model Selection → Optuna Tuning → **Final Test Evaluation** (frozen v2 on Test, 520,391 rows, md5 pinned; global WAPE 0.3754, B-cohort 0.6282; prediction integrity PASS) → **Quality Gate** (quality_gate_policy_v2_revised: B_wape reclassified diagnostic, 18/18 blocking PASS; legacy FAIL preserved) → **Production** (demand_forecasting_final v2, run 5d729a85, no V3). v1 (version 1, run 7fc87198, FAIL) preserved as **Staging**. Pre-final-test cleanup complete (schema 86→93, OHE train-only). Three pre-registered **validation-only** experiments returned NO_VALID_IMPROVEMENT and are retained as provenance, not as workflow stages (see HISTORICAL PROVENANCE). | `03_demand_forecasting_engine.ipynb` · `src/models/*` (`optuna_tuning.py`, `final_forecast.py`, `finalize_lifecycle.py`, `finalize_v2.py`, `quality_gate.py`, `registry.py`, `gate_policy_revision.py`, `b_cohort_error_analysis.py`, `b_cohort_remediation.py`, `b_cohort_asymmetric_objective.py`) · `src/features/feature_engineering.py` · `src/data/data_prep.py` · `src/models/pyfunc_wrapper.py` · `tests/test_*.py` · `artifacts/phase3/*` · `artifacts/mlflow_tracking.db` · Model Registry `demand_forecasting_final` v1 **Staging** (FAIL, preserved) + v2 **Production** · `reports/forecasting/*` |
| 4. Dynamic Pricing Optimization Engine | ✅ **IMPLEMENTED** — code + 30 tests green on the real data (reference week 2018-05-14 < Test split 2018-05-21). Elasticity estimated (descriptive, log-log, product+week fixed effects): pooled product-FE b=−0.395 (t=−10.88, n=41.5k), val sign+mag match true; per-product status Reliable 119 / Weak 1,589 / Insufficient 1,130 / Unsupported 16,261; effective elasticity sources own 1,538 / category-pooled 15,474 / none 2,087. **Profit maximization blocked by design**: no cost source exists (Phase-4.0 feasibility audit) → 1,385 scored products → 0 optimized, every row `PROFIT_NOT_COMPUTABLE (no cost supplied)`, 16 guardrail ceiling applications, 468 extrapolation warnings. Reports `part4_pricing_engine.{md,json}`, `part4_data_feasibility.{md,json}`; anti-leakage enforced (worst-case week = last Val week, test split never read). | 
| 5. SHAP Explainability & Model Interpretation | ✅ **COMPLETE** — code + 14 tests + artifacts green on the Production model (`models:/demand_forecasting_final/2`, run `5d729a85`, 600-tree LightGBM): SHAP TreeExplainer over 1,500 reference-week (2018-05-14, Train/Val only) rows vs 400-row deterministic background (seed 20260924); 93-col schema match verified; per-row additivity correction applied because LightGBM bin-quantization drifts `base+sum(phi)` from `predict` (max 0.02697 / mean 0.00263 log1p) → served units reproduced exactly as `expm1(base_row+Σφ)`; global top feature = `cat_unknown` (mean\|SHAP\| 0.0304, long-tail products), `price_prev` #5 but minor (0.0016) and direction-ambiguous; group totals categorical 0.0417 / regime 0.0076 / driver 0.0074 / price 0.0016; price features reported as attribution-only (causality → Part-4 elasticity); beeswarm/importance/parquet/CSV/metadata + 3 local waterfalls (`typical/high/low`; zero-bucket empty in sample); `part5_shap_explainability.{md,json}`; Test split never accessed. |
| 6. Deployment (MLflow / FastAPI / Docker) | ✅ **COMPLETE** — FastAPI business-level API (product_id / ISO week / optional cost — never the 93-feature schema, never `data/splits`), served from the immutable Production model `m-e93cadaf963a4aa1b44b3e501379da92` (v2, run `5d729a85`, READY) via a **registry-verified portable artifact-dir** load (MLmodel yaml `artifacts\model.joblib` resolved through a path-normalized copy inside POSIX containers; identical served surface verified); endpoints `/health /ready /predict /pricing /explain`; PostgreSQL inference log (`inference_records`), `inference_logged` heartbeat; streamlit dashboard (`dashboard/app.py`); single Python image + `docker compose` (db 5432 / api 8000 / dashboard 8501, all healthchecks green); heavy real-model integration suite + native parity tests; real Docker smoke test passed — protected splits/test.csv provably absent from the image; registry untouched (read-only snapshot test). Full report `reports/deployment/part6_deployment_implementation_report.md`. |

## PART-4 DELIVERABLES (produced)

| Module | Responsibility | Certified |
|---|---|---|
| `src/optimization/pricing_data.py` | Reference-week scope, price ceiling/floor bounds, cost-source probe, matrix accessors | ✅ |
| `src/optimization/elasticity.py` | Descriptive price-elasticity: log-log OLS, pooled product / product+week within-demeaned FE, per-product statuses, validation sign+mag check | ✅ |
| `src/optimization/simulation.py` | PED-scenario price simulation (e.g. doubling) over reference-week scope, extrapolation guard | ✅ |
| `src/optimization/optimizer.py` | `analytic_optimal_price` + grid profit optimizer, cost-gated profit, NaN/`None` guards | ✅ |
| `src/optimization/guardrails.py` | Configurable guardrails (`min_margin_pct`, `max_increase_pct` as an increase cap, `price_floor_abs`, `price_ceiling_abs`, `smoothing_alpha`) with per-rule applied-flag integrity | ✅ |
| `src/optimization/demand_baseline.py` | Production-model demand prediction as baseline + on-the-fly frozen 93-col reference-week features (`frozen_features_for_week`, train-only encoding, unseen→`unknown`) | ✅ |
| `src/optimization/pricing_engine.py` | Orchestration: baseline → elasticity → effective-volume → optimizer → guardrails → reports (JSON-safe, markdown top-10) | ✅ |
| `reports/forecasting/part4_data_feasibility.{md,json}` | Phase-4.0 audit: no cost source, no `price*cost` profit, elasticity only | ✅ |
| `reports/forecasting/part4_pricing_engine.json` + `part4_pricing_engine_report.md` | Real-data engine run (reference week 2018-05-14, `feature_source=on-the-fly-frozen-val`). The `.json` is the only report shipped into the image (read at runtime by `dashboard/product_catalog.py`). | ✅ |
| `artifacts/optimization/recommended_prices.csv` | 1,385 rows × full reason taxonomy (468 `PROFIT_NOT_COMPUTABLE (no cost supplied)` / 917 `NO_ELASTICITY`) | ✅ |

## PART-5 DELIVERABLES (produced)

| Module | Responsibility | Certified |
|---|---|---|
| `src/explainability/shap_analysis.py` (+ `__init__.py`) | Explains the **registered Production model only** (`models:/demand_forecasting_final/2`, read-only; estimator via `unwrap_python_model()._model`, no surrogate). Deterministic sampling (`SEED=20260924`, numpy legacy `RandomState`), TreeExplainer (background 400 / explain 1,500), per-row additivity-corrected base (`base_row = pred_log1p − Σφ`, served units = `expm1(base+Σφ)` exact), beeswarm + global importance + 3 local waterfalls, parquet/CSV/metadata, markdown + JSON report. **Never reads `data/splits/test.csv`.** | ✅ |
| `tests/test_shap_explainability.py` | 14 guards: registry Production identity (v2/run `5d729a85`, READY), AST no-`test.csv`, reference-week < `TEST_START_WEEK`, schema = 93 in registry order, no identity cols, SHAP dims/finite + **exact** per-row `expm1(base+Σφ)` reconstruction on a tiny LGBM, deterministic sample (same rows twice), stable artifacts, importance CSV ↔ computed parity, price features present | ✅ |
| `artifacts/explainability/` | `shap_summary_beeswarm.png` · `shap_feature_importance.png` · `shap_global_values.parquet` (1,500×93) · `shap_feature_importance.csv` · `metadata.json` · `local/example_{typical,high,low}_demand.png` | ✅ |
| `reports/forecasting/part5_shap_explainability.{md,json}` | Real-data run (reference week 2018-05-14, bg 400 / explain 1,500, seed 20260924); faithfulness/additivity disclosure; top-12 global table; group totals; price-feature + regime tables; local summaries; limitations (SHAP ≠ causal; elasticity = price-response framework) | ✅ |

## PART-6 DELIVERABLES (produced)

| Module | Responsibility | Certified |
|---|---|---|
| `app/` (FastAPI package) | `config` (env/pydantic-settings) · `schemas` (business payloads) · `errors` + `exception_handlers` (structured codes, no stack leaks) · `dependencies` · `db` (SQLAlchemy models, engine/session, repository `build_success/error_record`+`persist`) · `services` (`model_service` registry-verified portable loader + MLmodel path-normalization fallback, `state` AppState with 5-component readiness gate, `prediction/pricing/explainability`) · `api` (`health`, `ready`, `predict`, `pricing`, `explain`) · `middleware` (request-id, structured logging, payload-limit 413, optional rate-limit) · `main` (`create_app` factory, lifespan load) | ✅ |
| `dashboard/` | Streamlit manager UI: forecast / pricing / explainability / system panels; business contract via the API (product ID, ISO week, optional cost); labels SHAP features with business names, never feature schema. Adds a searchable product selector, a week selector, and pricing-readiness tiering from the on-disk demand matrix + `part4_pricing_engine.json` (`product_catalog.py`, `pricing_presenter.py`, `api_client.py`, `config.py`) | ✅ |
| `deployment/Dockerfile` | `python:3.11-slim` + `libgomp1`, `PYTHONPATH=/app`, non-root uid 10001, single image for api + dashboard | ✅ |
| `deployment/docker-compose.yml` | `db` postgres:16-alpine (no published host port; 5432 internal only) → `api` (depends_on healthy; healthcheck `/ready`, start_period 600s) → `dashboard` (healthcheck `/_stcore/health`). No custom `networks:` — all three services share the implicit default Compose network by service name. Named volume `pgdata` persists the inference log across `down`; `down -v` deletes it | ✅ |
| `deployment/healthcheck.py` | HTTP 2xx gating used by both compose healthchecks; copied into the image at `deployment/healthcheck.py` | ✅ |
| `.dockerignore` | Excludes `data/raw`, `data/splits`, `tests`, notebooks, root `mlflow.db`, `.trash`, `.env`, `.venv`. Reports excluded via `reports/**` with one deliberate re-inclusion, `!reports/forecasting/part4_pricing_engine.json` (read at runtime by `dashboard/product_catalog.py`). Verified by image inspection: no `data/splits`/`test.csv`, exactly one report | ✅ |
| `requirements.txt` | Runtime pins for the image (numpy/pandas/sklearn/lgbm/shap/mlflow/fastapi/uvicorn/streamlit/SQLAlchemy/psycopg2 …) — duplicate-free, no training libs (optuna/xgboost excluded from served app) | ✅ |
| `tests/deployment/` | 47 fast tests + 8 real-model integration tests: OpenAPI surface = exactly 5 paths; VALIDATION/domain/codes (PRODUCT_NOT_FOUND, INVALID_WEEK, WEEK_OUT_OF_SUPPORTED_RANGE, TEST_BOUNDARY_VIOLATION, MODEL_LOAD_FAILED …); persistence rows; readiness gating; registry read-only snapshot; artifact-dir resolution + path-normalized-load parity; safety scans (no `test.csv`/`data/splits`, no registry writes, no `.fit()`, no optuna/xgboost imports, dashboard never references feature names); real-model end-to-end (predict/pricing/explain + DB writes) | ✅ |
| `reports/deployment/inference_request_schema_audit.md` · `reports/deployment/part6_deployment_architecture_audit.md` · `reports/deployment/part6_deployment_implementation_report.md` | Part-6 audits + implementation report | ✅ |

### Part-6 reference points (cross-checks)
- **Model identity served:** `demand_forecasting_final` v2, stage **Production**, status **READY**, run `5d729a8594dc488bbd18ba0c4e216f5a`, source `models:/m-e93cadaf963a4aa1b44b3e501379da92` (93 signature inputs, LGBMRegressor).
- **Week contract:** reference week `2018-05-14` (last Val week) — all weeks are Mondays in `[matrix start, 2018-05-14]` (inclusive) supported; `>= 2018-05-21` → `TEST_BOUNDARY_VIOLATION`; non-Monday → `INVALID_WEEK`.
- **Pricing never invents cost/price:** no cost supplied → 200 with `PROFIT_NOT_COMPUTABLE (no cost supplied)`; no reference-week price observation → `NO_PRICE_OBSERVATION`; guardrails applied only when profit is computable.
- **Explainability:** deterministic 400-row background (seed 20260924), `TreeExplainer(estimator, feature_names=...)`, per-row reconstruction `expm1(base_log1p + Σφ)` with residual asserted < 1e-6 in tests; business labels (`Category: unknown`, `Last observed price`, …) + technical `feature_code`.
- **Deployment verification:** heavy real-model integration suite (`RUN_DEPLOYMENT_INTEGRATION=1`) 8/8; full repo suite 402 collected → 393 passed / 9 skipped / 0 failed; Docker smoke: api+dashboard+db healthy, `/ready` 200, real predict/pricing/explain 200, inference rows in PostgreSQL, OpenAPI = 5 paths, image free of `test.csv` with exactly one report (`part4_pricing_engine.json`).
- **Pricing readiness tiers (dashboard):** products split per week into priced & ready / ready-unpriced / not ready. Reference week 2018-05-14: 468 / 16,544 / 15,717. Artifact availability is tracked as `available`/`missing`/`unreadable`; when unavailable the UI reports `READINESS_UNAVAILABLE` rather than zero ready products.
- **Compose networking & persistence:** implicit default network (no custom `networks:`); `db` publishes no host port; named volume `pgdata` (typically `deployment_pgdata`) survives `down`, deleted by `down -v`.

### Post-Part-6 work (dashboard UX + image packaging)
- `dashboard/product_catalog.py` / `pricing_presenter.py` / `app.py` (2026-09-30): searchable product selector, week-scoped pricing readiness + tiering, explicit readiness-degradation states.
- `.dockerignore` + `deployment/Dockerfile` (2026-09-30): the selective `reports/**` carve-out for the single pricing artifact; all other reports stay out of the image.
- `tests/dashboard/` (72 tests) and `tests/deployment/test_deployment_safety.py`: coverage for the presentation layer and image-boundary safety.
- `reports/finalization/pricing_manager_ready_*.md`: runtime audit, UX audit, and the Docker packaging fix with image verification.

## FINALIZATION STATUS (authoritative status & map — NOT a tutorial)

**Status: COMPLETE WITH LIMITATIONS** — see `reports/finalization/finalization_report.md` (Phase-1..10 audit, environment, reproducibility, final verification).

### Documentation layer

**Public technical documentation** (published):

| Doc | Purpose |
|---|---|
| `README.md` | Public overview: architecture (Mermaid), capabilities, install/run/test, limitations |
| `PROJECT_MAP.md` | This map: architecture, responsibilities, test inventory, artifacts, known issues |
| `reports/forecasting/` | Per-part evidence (32 files): leakage audit, model selection, gate policy, B-cohort, pricing, SHAP |
| `reports/deployment/` | Deployment architecture audit + implementation report |
| `reports/finalization/` | Finalization and pricing-manager-ready audits |

**Private material** (not published; excluded via `.gitignore`):

| Doc | Purpose |
|---|---|
| `docs/private/PROJECT_WALKTHROUGH.md` | Study guide: WHAT/WHY/HOW/RESULT per Part + summaries |
| `docs/private/PROJECT_STORY.md` | Narrative + elevator-pitch versions |
| `docs/private/CV_DESCRIPTION.md` | CV/ATS resume bullets |
| `docs/private/INTERVIEW_PREPARATION.md` | Interview Q&A + memory number sheet |
| `docs/Photos/` | Unreferenced screen-capture PNGs — pending manual review, not published |

Private docs carry the same verified metrics as the public record; they are separated for
publication hygiene, not because they are inaccurate.

### Final completion checklist
- [x] Parts 1–6 implemented, tested, and user-approved (see PHASE PROGRESS).
- [x] Production model `demand_forecasting_final` v2 (run `5d729a85`) stage **Production**, status **READY**, immutable, served read-only.
- [x] MLflow history preserved (v1 Staging FAIL + v2 validation + revision tags all present, read-only snapshot test).
- [x] Protected test split never extended/leaked (AST + split guards; image free of `data/splits`/`test.csv`).
- [x] Test suite: **402 collected** → 393 passed / 9 skipped / 0 failed (`.venv` + `requirements-dev.txt`).
- [x] Docker files validated (`docker compose config` valid; stack live & healthy in Part 6 smoke; `.dockerignore` scope verified by image inspection).
- [x] Documentation references real files/artifacts; metrics match reports; no unsupported claims. **Re-verified 2026-09-30** — this map had drifted (test counts, guardrail taxonomy, 86-vs-93 schema, dashboard file reads, `.dockerignore` scope) and is now synchronized with the repository.
- [x] Environment reproducible: `.venv` (Python 3.11.8) + pinned `requirements.txt` + dev layer (pytest 9.1.1).
- [x] Finalization report written at `reports/finalization/finalization_report.md`.

### Known limitations (explicit, documented)
- B-cohort underforecast on burst weeks (validation B WAPE 0.5101, B bias −18.2%); spike-week weakness documented.
- No cost source → profit optimization delivered but inert until a legitimate cost source is supplied.
- Elasticity descriptive (not causal); only a minority of products have identifiable own-price response.
- Dataset ends 2018-09-09 → no untouched future validation period can be built without reusing development data; the Test split is pinned by content hash.
- SHAP is attribution, not causality; price features are minor attribution contributors.
- No CI/CD or scheduler: lifecycle *triggers* are manual; gate evaluation/promotion routing is automated and guarded.
- `PROJECT_BLUEPRINT.md` referenced but absent (see header note).
- The dashboard requires its data on disk (weekly demand matrix + `part4_pricing_engine.json`) independent of the API; a missing pricing artifact is reported as `READINESS_UNAVAILABLE`, never as zero ready products.
- Version control is **not initialized** in this tree. `.gitignore` now excludes `docs/private/`, `data/`, `mlruns/`, `artifacts/` and root `mlflow.db`, so those exclusions take effect on `git init` — but until the tree is initialized, no rule is actually enforced.

## DATASET INSIGHTS (post Phase-1)

### Merge graph
`order_items` ⟕ `orders` (order_id) ⟕ `products` (product_id) ⟕ `category_translation` (product_category_name)

### Fact table (merged)
- **112,650 rows × 23 columns**, item-level granularity (`order_id` + `order_item_id` natural key).
- Zero exact duplicates; zero natural-key collisions.
- **4,029 rows (3.6%)** contain ≥1 missing value.

### Data types
- 6 `datetime64` : all timestamps parsed on ingestion. Purchase window **2016-09-04 → 2018-10-17 (772 days)**.
- 9 `float64`, 1 `int64`, 7 `object`.

### Missing-value notes (structural, not corruption)
- `order_delivered_customer_date` 2.2% / `order_delivered_carrier_date` 1.1% / `order_approved_at` 0.01% — absent for undelivered / not-yet-delivered orders.
- Product specs + category absent for **610 products (1,603 item rows, 1.42%)** — no category metadata in source.

### Statistical anomalies to manage downstream
- `price`: min 0.85 → max **6,735**; mean 120.65, p99 = 890 → extreme right tail.
- `freight_value`: mean 19.99, max 409.68.
- **`shipping_limit_date` extends to 2020-04-09** — far beyond purchase window → long-dated shipping limits; verify/filter in preprocessing.
- `order_status`: 7 values; 110,197/112,650 rows `delivered`.
- Basket size: mean 1.14, max **21** items/order.
- 32,951 distinct products, 71 categories, 3,095 sellers.

## EVALUATION FRAMEWORK (in production use)

Inputs: right-skewed, over-dispersed `order_demand` (count-like demand with heavy tails, zeros for long-tail idle weeks).

- **Primary:** `WAPE` (scale-relative accuracy unaffected by outlier magnitude).
- **Secondary:** `MAE` (interpretable absolute error in units sold) and `RMSE` (heavy-tailed penalty).
- **Cost-of-error framework:** asymmetric — under-forecasting = stockout/lost-sales (higher penalty); over-forecasting = holding/inventory cost (lower penalty). Ratio default 2:1 in `src/utils/metrics.asymmetric_loss`.

## PHASE-2 DELIVERABLES (produced)

| Module | Responsibility | Certified |
|---|---|---|
| `src/data/data_processing.py` | Raw→valid-order fact table (status policy, schema contract) | ✅ |
| `src/data/target_engineering.py` | Weekly `order_demand` target + dense zero-filled matrix | ✅ |
| `src/data/temporal_split.py` | Chronological 70/15/15 split (no shuffle, no-leakage guard) | ✅ |
| `src/data/imputation.py` | Train-only conditional imputation (median / `unknown`) | ✅ |
| `src/features/feature_engineering.py` | Calendar, lags 1-4, rolling 4w, price/freight/category-share drivers, log1p target | ✅ |
| `src/utils/metrics.py` | WAPE (primary), MAE, RMSE, Asymmetric Loss, `describe_metrics` | ✅ |
| `data/splits/{train,val,test}.csv` | Imputed, feature-engineered split artifacts (2.43M / 520K / 520K rows) | ✅ |
| `data/processed/demand_matrix_weekly.csv.gz` | Dense product×week matrix (3.47M rows, 13 MB) | ✅ |

### Phase-2 pipeline summary
- Valid-order fact table: **112,101 rows** (excludes 549 canceled/unavailable *items*; 1,234 orders).
- Dense demand matrix: **32,729 products × 106 weeks = 3,469,274 rows**.
- Split: train 2,428,492 / val 520,391 / test 520,391 (70/15/15). Boundary week-split allowed (`<=` guard); train strictly before test.
- **Zero-lookahead probe: PASSED** (features provably independent of current-week facts).
- Baseline sanity: no-skill train-mean model → WAPE 1.3783; oracle → 0.0.

## PHASE-3 DELIVERABLES (produced)

| Module | Responsibility | Certified |
|---|---|---|
| `src/features/feature_engineering.py` | Phase-3 stack: `PHASE3_LAGS (1,2,4)`, `PHASE3_ROLLING_AGGS (mean,std)` w/ 4-week window, `restrict_to_window`, `build_phase3_feature_matrix` | ✅ |
| `src/models/data_prep.py` | `prepare_model_matrix` (shared design, one-hot float32 + global schema alignment), `make_sample_weights` (1+log1p), `design_matrix_rows` | ✅ |
| `src/models/forecasting.py` | `make_model`, `naive_combined_predictions`, `expanding_window_splits`, `run_expanding_cv`, `fit_full_with_early_stopping`, `score_on` | ✅ |
| `src/models/optuna_tuning.py` | TPE tuning (10 trials/model) minimizing mean-fold WAPE; LightGBM & XGBoost suggestors | ✅ |
| `src/models/pyfunc_wrapper.py` | `DemandForecaster` PyFunc with strict schema validation; `save_winner` MLflow logging | ✅ |
| `artifacts/phase3/model.joblib` + `feature_columns.json` | Winner LightGBM artifact + exact **93**-column frozen feature schema (staged) | ✅ |
| `artifacts/mlflow_tracking.db` | SQLite MLflow backend: naive / final_lgbm / final_xgb / winner runs | ✅ |

### Phase-3 results (validation split, one-shot — test never touched)
- **Fold-boundary equivalence proof: PASSED** (8,100 overlapping product-week rows, 16 engineered columns identical to global pass).
- Naive persistence baseline: WAPE **1.4923**. LightGBM: WAPE **0.4001** (‑73% vs naive). XGBoost counter-check (full-train refit validation, same sample): WAPE **0.4349**.
- Winner = **LightGBM** (MAE 0.0235, RMSE 0.2725, Asymmetric 2:1 = 0.0381) — tuned on 120 mature products, 5 expanding-window folds, early stopping on the last 8 training weeks.
- PyFunc smoke test: logged model predicts on `pred_demand` units and rejects malformed schemas (`ValueError`).

## PART 1–3 DATA & FEATURE LEAKAGE AUDIT (evidence-based, audit-only)

Products: `reports/forecasting/part1_3_data_feature_leakage_audit.md` + `.json` — 19 audit questions answered, raw→engineering→selection→final-model→temporal-validation trace.

### Feature pipeline facts
- **Original data:** 38 raw columns across 9 tables; only **4** ingested (orders, order_items, products, translation → 23-col fact table). Delivery/deadline timestamps pulled as columns but **never become model features** (would be post-purchase).
- **Engineered:** 4 calendar + lags {1,2,4} + rolling mean/std 4w + 4 demand drivers + 7 EXP2 regime features + **log1p target with 1+log1p weights**. All features **past-only** (shifts / shift(1)+rolling / groupby-shift) — no torrent forward.
- **Final model (frozen v1/v2): 93-column schema** = 13 base + 73 one-hot `cat_*` (incl. `cat_unknown`, float32) + 7 regime. Excluded: `demand_lag_3`, `rolling_demand_max_4w`, current-week `avg_price`/`avg_freight`, identities, raw category.
- **Encoding:** one-hot (stateless `pd.get_dummies`), vocabulary **train-equivalent** (verified: no val/test category outside train vocab). Leakage-safe (target-free). Native LightGBM categorical not used (single LGBM/XGB code path).
- **Leakage result:** **PASS** — probes (zero-lookahead, phase-3 no-lookahead, imputation scope), fold-boundary equivalence proof, train-only median imputation, train-only MASE/cohort stats, Optuna confined to train folds, test scored once (v1). **No confirmed leakage.**
- **Feature selection:** NOT an automated algorithm (no RFE/permutation/shap/variance filters). **Manual curated stack** + Phase-3 subset + EXP2 experiment, decided via temporal-window WAPE/cohort metrics; gain importance diagnostic only.
- **Temporal CV:** expanding window (5 folds, horizon 4, min_train_weeks 24), 120 mature products (8,899 rows, train only): best mean-fold WAPE LGBM **0.5802** vs XGB **0.5492** (Optuna). Winner on full-train refit validation: LGBM **0.4001** → frozen v2 **0.3964**; XGB counter-check **0.4349**; naive persistence **1.4923** (−73%).
- **Preprocessing:** median imputer train-only + structural `unknown`; no scaler (trees); outliers retained; log1p target only; no feature-level skew transforms.

### Status per Part
- **Part 1:** data types/temporal parsing `IMPLEMENTED` · duplicates/natural-key `IMPLEMENTED` · domain constraints `PARTIAL` (EDA-only) · leakage-free missing `IMPLEMENTED` · distribution diagnostics `IMPLEMENTED` · time-series statistical EDA `PARTIAL` (Spearman + lag-1 autocorr, no ACF/PACF) · effect-size/feature-impact `PARTIAL` (gain diagnostic).
- **Part 2:** temporal split `IMPLEMENTED` (`<=` guard, shared boundary weeks) · imputation `IMPLEMENTED` · feature engineering `IMPLEMENTED` · outlier/skew `PARTIAL` · scaling `SKIPPED-BY-DESIGN (trees)` · encoding `IMPLEMENTED (OHE 73)`.
- **Part 3:** all 15 items `IMPLEMENTED`; early stopping has notebook-vs-frozen nuance (reproduced WAPE 0.3964). Official workflow completed: Train/Validation → expanding-window CV → Model Selection → Optuna Tuning → **Final Test Evaluation** (frozen v2 scored once on Test, 520,391 rows, md5 pinned; prediction integrity PASS with n_neg 0 and verified inverse transform; global WAPE 0.3754, B-cohort 0.6282) → **Quality Gate** (FAIL on `B_wape=0.6282` under `quality_gate_policy_v1_legacy`; `REVISED_GATE_APPLIED` under `quality_gate_policy_v2_revised` where `B_wape` is diagnostic and 18/18 blocking criteria PASS) → **Production** (v2 Staging → Production, run 5d729a85, `archive_existing_versions=False`, no V3, V1 untouched); **Part 3 COMPLETE** (275 pipeline guards, all green). The dataset ends 2018-09-09, so no future validation period can be built without reusing development data — preserved as a record in `part3_final_holdout_blocker.{json,md}`. Three pre-registered **validation-only** experiments (B-cohort diagnosis, B-cohort remediation, asymmetric 2:1 quantile objective) each returned `NO_VALID_IMPROVEMENT`; they changed no registry state and no production behaviour, and are documented in HISTORICAL PROVENANCE as provenance rather than workflow stages.

### Confirmed issues (mismatches, no leakage)
| # | Severity | Finding |
|---|---|---|
| M1 | ~~Medium~~ **Resolved** | ~~`artifacts/phase3/feature_columns.json` = 86 cols (staged notebook-03 winner, no regime) vs deployed v1/v2 = 93 cols~~ — regenerated from the authoritative frozen v2 schema; now byte-identical to MLflow in-run artifact, registered-model signature, and train-reconstructed layout (93 cols; `test_prefinal_cleanup` guards). |
| M2 | Minor | Blueprint ACF/PACF → only lag-1 autocorr + Spearman implemented (diagnostic). |
| M3 | Minor | Blueprint feature power transforms → only **target** log1p (trees invariant). |
| M4 | Minor | "OHE safest vs leakage" claim over-stated globally; project-safe here. |
| M5 | Minor | Early stopping in notebook; frozen refit trains full 600 trees (same WAPE) — **documented** in `finalize_lifecycle.py` (cleanup). |

### Unresolved / manual-review
1. ~~M1 artifact-schema sync~~ **resolved**. 2. ~~OHE layout reads all splits~~ **resolved** (train-only + assertion). 3. ~~Shared boundary week-labels train/val & val/test~~ **documented + tested**. 4. ~~Early-stopping nuance~~ **documented** for Part 5. 5. ~~Tuning on 120-product subset vs full train~~ **documented + tested** (`optuna_tuning.py`). — Full details in audit report §16/Q19 and cleanup report `part3_pre_final_test_cleanup.md`.

## HISTORICAL PROVENANCE (completed experiments — not current workflow stages)

This section preserves the factual record of experiments that were executed and completed.
None of them is a stage of the official workflow shown in `SYSTEM_FLOW`. They are retained
because their artifacts are evidence and their registry tags are asserted by active tests.
Full per-experiment detail lives in `reports/forecasting/` and in `mlruns/`.

### Final Test Evaluation record (frozen v2, scored once)

- One-off module with its own exactly-once marker and no `--force`. The module has been
  archived out of the importable tree to
  `reports/forecasting/_historical_audit/posthoc_holdout.py` (no longer part of `src/`, and
  not collected by pytest); its former test suite is archived beside it.
- Test reproduced (520,391 rows, md5 pinned), prediction integrity PASS (n_neg 0,
  inverse transform verified). Global Test WAPE **0.3754**, B-cohort **0.6282**.
- Quality Gate consumed unchanged → **FAIL** on `B_wape = 0.6282 > 0.60` → v2
  Production→Staging. This FAIL is preserved verbatim.
- Evidence: `reports/forecasting/part3_v2_posthoc_holdout_evaluation.{json,md}` — still read
  by `test_b_cohort_remediation`, `test_b_cohort_asymmetric_objective` and
  `test_gate_policy_revision` — plus the tagged copies under
  `mlruns/2/5d729a85.../artifacts/`.

### B-cohort root-cause diagnosis (validation-only)

- `src/models/b_cohort_error_analysis.py`; Train+Val only, no Test access, no model change.
- Validation B WAPE 0.5101460932701322 reproduced exactly; error concentrates on spike weeks
  (10% of rows → 27% of B error, WAPE 0.612, bias ≈ −58.9%). Primary mechanism = scale
  shrinkage on high-demand/spike weeks.
- Reports `part3_b_cohort_error_analysis.{json,md}`; guards `tests/test_b_cohort_diagnostics.py` (10).

### Controlled B-cohort remediation experiment (validation-only, no new version)

- `src/models/b_cohort_remediation.py`; 3 pre-registered arms on the same Train/Val; dedicated
  additive MLflow experiment `part3_b_cohort_remediation_exp` (`test_used=false`).

| Arm | Intervention | Result |
|---|---|---|
| `v2_control` | exact V2 replica | reproduced global 0.3964 / B 0.5101460932701322 / bias −18.227% |
| `wape_weights` | train-only demand-level alignment, `min(max(1,d/p50+),p99+/p50+)` | no improvement (B 0.531, global 0.458) |
| `spike_features` | 4 strictly-past features (+peak_8w, +lag1_ratio_to_8w_peak, +weeks_since_peak_8w, +burst_count_16w); schema 93→97 | B WAPE 0.4327 (−15.2%), B bias −0.94%, global 0.3960; B2 0.584 (+0.028, within +0.03) but C WAPE 0.3953 (+0.0158 > +0.01 tolerance) |

- Pre-registered decision **NO_VALID_IMPROVEMENT**. Registry v1/v2 stages and run_ids
  untouched; no V3. Guards: `tests/test_b_cohort_remediation.py` (29).

### Controlled asymmetric 2:1 objective experiment (validation-only, no new version)

- `src/models/b_cohort_asymmetric_objective.py`; pre-registered 8-criterion design; dedicated
  additive MLflow experiment `part3_b_cohort_asymmetric_exp` (2 FINISHED runs
  `arm1_v2_control` / `arm2_asym_quantile_alpha067`, `test_used=false`).
- Treatment = LightGBM built-in `quantile` objective at alpha = 2/3 (= 2/(2+1), the exact
  2:1 asymmetric-absolute business loss: 3·pinball_(2/3) ≡ asymmetric_loss(2,1), identity
  validated).
- Result: global WAPE 0.3665 (−7.5%); MAE/MASE/asym21/C improved; B bias −18.2% → +1.5%; but
  B WAPE worsened 0.5101 → 0.5564 (+0.046) and B2 0.5557 → 0.5972 (+0.042), violating
  pre-registered C2/C5. Decision **NO_VALID_IMPROVEMENT**; no combined or retuned arm was
  pre-registered, so none was selected. Registry untouched; no V3.
- Guards: `tests/test_b_cohort_asymmetric_objective.py` (30).

### V2 production-policy audit (audit-only)

- No code, model, gate or registry change; Test untouched; no V3.
- Provenance of `B_wape <= 0.60` = **engineering heuristic** (a reference-anchored regression
  guard against validation references 0.51–0.55, pre-registered before Test). The
  “monetarily dominant” label was asserted without independent in-repo evidence. Only the
  bias bounds carry a documented business requirement, and those pass (B +9.85%, B2 +0.86%).
- Decision **B_FORMAL_GATE_REVISION_REQUIRED**: V2 stays Staging, gate unchanged, an owner
  policy decision required before any re-baselining or re-formulation. Option C (repo-native
  alternative gate formulation) rejected. Robust Loss deferred as a possible future
  pre-registered validation-only experiment.
- Report `part3_v2_gate_production_policy.{json,md}`.

### Formal gate-policy revision → Production promotion

- `src/models/gate_policy_revision.py`, owner-authorized; policies versioned
  `quality_gate_policy_v1_legacy` (`B_wape` hard blocker, default) and
  `quality_gate_policy_v2_revised` (`B_wape` diagnostic/non-blocking; `B_bias_pct >= -20` and
  all other criteria unchanged and hard).
- Metrics replayed only from the persisted Final Test Evaluation report — no Test access, no
  estimator load, no retraining. Legacy replay deterministic and identical to the persisted
  FAIL (19 outcomes, single failure `B_wape`); revised gate **PASS** (18/18 blocking criteria;
  observed `B_wape` 0.628239 reported as diagnostic).
- v2 **Staging → Production**, same run 5d729a85. Exactly-once revision marker plus
  `gate_policy_revision_applied` tag; no `--force`; no V3; no re-registration; historical tags
  (`production_gate=FAIL`, `gate_decision=FAIL`, `promoted_stage=Staging`,
  `lifecycle_stage=Staging`) preserved verbatim; new tags `gate_policy_version` /
  `legacy_gate_decision=FAIL` / `revised_gate_decision=PASS` / `b_wape_role=diagnostic` /
  `evaluation_type` / `test_used_for_policy_revision=false`.
- Report `part3_v2_gate_policy_revision.{json,md}`; guards `tests/test_gate_policy_revision.py` (21).
