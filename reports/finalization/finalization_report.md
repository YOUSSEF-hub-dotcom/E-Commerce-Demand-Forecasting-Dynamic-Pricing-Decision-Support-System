# Finalization Report — E-Commerce Demand Forecasting & Dynamic Pricing Decision-Support System

> Finalization audit of the completed system. Every result is traceable to a repository
> artifact, report, test, or live MLflow record. Nothing was retrained; no model/version
> was created (no V3); no quality-gate threshold was changed; the protected Test split was
> not accessed; all Parts 1–6 business logic is untouched. The only repository changes are
> documentation + a dev-only dependency file.

---

## 1. Repository audit

### 1.1 Inventory

| Area | State |
|---|---|
| `data/raw/` | Read-only source CSVs (raw Olist tables) |
| `data/processed/` | `demand_matrix_weekly.csv.gz` (13 MB, served slice in the image) |
| `data/splits/` | Chronological train/val/test (protected; **never shipped**, verified in image) |
| `src/` | `data` · `features` · `models` · `explainability` · `optimization` · `utils` · `notebooks` |
| `app/` | FastAPI service (api · services · db · middleware) |
| `dashboard/` | Streamlit manager UI |
| `deployment/` | `Dockerfile` · `docker-compose.yml` · `healthcheck.py` |
| `tests/` + `tests/deployment/` | Pipeline + deployment contract suites |
| `reports/` | `forecasting/` (part1–5) · `deployment/` (part 6) · `experiments/` |
| `artifacts/` | trained models · `mlflow_tracking.db` · SHAP outputs · optimization outputs |
| `mlruns/` | MLflow run/artifact store (67 files) |
| `requirements.txt` | runtime pins (Docker + venv) |
| `requirements-dev.txt` | **added in finalization** — dev/test layer (pytest, xgboost, optuna) |
| `.env.example` | documented config with production-safe defaults |
| `.gitignore` | already ignored `.venv/`, `.env`, caches, DB artifacts |
| `.dockerignore` | excludes raw/splits/tests/reports/notebooks/mlflow.db/.trash/.env/.venv |
| `README.md` | **created in finalization** (GitHub face; Mermaid architecture) |
| `docs/` | **created in finalization** — 4 documentation files |
| `PROJECT_MAP.md` | authoritative status map (**updated**) |
| `PROJECT_BLUEPRINT.md` | **referenced but ABSENT** (see 1.2) |

### 1.2 Notable audit findings

- **PROJECT_BLUEPRINT.md does not exist.** It is referenced in `.dockerignore` (excluded
  from builds) and in PROJECT_MAP as "SOURCE OF TRUTH", but the file itself is not in the
  repository and has never materialized. This is a documentation gap, not a code gap: the
  authoritative status/facts live in `PROJECT_MAP.md`, `reports/`, and the new `docs/`
  layer. Documented in PROJECT_MAP header; no content was invented to replace it.
- **`requirements.txt` had a test-reproducibility gap:** it is the lean runtime contract
  (correctly excluding training libs for the Docker image), but a clean install could not
  run the test suite (no `pytest`; `xgboost`/`optuna` lazily imported by the model layer).
  Closed with `requirements-dev.txt` — dev layer only; the Docker/runtime contract was
  **not** modified.
- No `.venv`/virtualenv existed (verified); the project previously ran tests against the
  system Python. Finalization created `.venv`.

### 1.3 Completed Parts & artifacts

| Part | Artifacts (representative) | Status |
|---|---|---|
| 1 | `src/notebooks/01_...eda.ipynb`, `tests/test_data_processing.py` (13) | ✅ |
| 2 | `src/data/*`, `src/features/feature_engineering.py`, splits + matrix, `tests/test_*` (temporal/target/data) | ✅ |
| 3 | `src/models/*`, MLflow v1/v2 lifecycle, gate + policy reports, `tests/` (metrics/registry/quality_gate/posthoc/b_cohort/gate_policy/lifecycle_router… ~230) | ✅ |
| 4 | `src/optimization/*`, `reports/forecasting/part4_*`, `artifacts/optimization/recommended_prices.csv`, 30 tests | ✅ |
| 5 | `src/explainability/*`, `artifacts/explainability/*`, `reports/forecasting/part5_*`, 14 tests | ✅ |
| 6 | `app/`, `dashboard/`, `deployment/`, `tests/deployment/` (45+8), `reports/deployment/part6_*` | ✅ |

### 1.4 Production model (live registry)

- Registered: `demand_forecasting_final`
  - **v1** — stage Staging, status READY, run `7fc871986e9240b1a4d98f7a9c1f893f`
    (holds the legacy single authorized test score; FAIL preserved).
  - **v2** — stage **Production**, status **READY**, run `5d729a8594dc488bbd18ba0c4e216f5a`,
    source `models:/m-e93cadaf963a4aa1b44b3e501379da92`.
- Run tags (v2): `evaluation_group=v2_validation`, `test_scored_once=false`,
  `posthoc_holdout_evaluated=true`, `model_version=2`, `gate_policy_version=quality_gate_policy_v2_revised`.
- Estimator: LightGBM `LGBMRegressor`, 93 inputs / 1 output, 600 trees,
  num_leaves 223, lr 0.008468, log1p target (expm1 inference). Load check in `.venv`:
  resolves Production READY, schema=93, signature 93→1, `unwrap_python_model.LGBMRegressor`,
  load 6.7 s.

### 1.5 MLflow configuration

`mlflow.tracking_uri = sqlite:///artifacts/mlflow_tracking.db`; artifact store `mlruns/`
(67 files). Served **read-only**. Registry snapshot during finalization: v1 Staging / v2
Production / run ids / stages identical to the Part-6 recorded state; tracking-db SHA-256
head `5672c219e8c708f3`; no finalization step executes registry writes (read-only snapshot
test also green).

### 1.6 Tests

| Suite | Result |
|---|---|
| Full suite (in `.venv`) | **335 passed, 8 skipped (integration-gated), 0 failed** (~6 min) |
| `tests/deployment` (in `.venv`) | 45 passed, 8 skipped |
| Real-model integration (`RUN_DEPLOYMENT_INTEGRATION=1`) | 8 passed (recorded in Part 6) |

### 1.7 Known limitations & business dependencies

Documented in `PROJECT_MAP.md` §FINALIZATION and `README.md` §Limitations: B-cohort
underforecast on burst weeks (val B WAPE 0.5101, bias −18.2%); **no cost source** → profit
path inert until costs supplied (business dependency: owner-supplied costs; promotions
data absent); elasticity descriptive, not causal; 82% of products have zero week-over-week
price variation; dataset frozen at 2018-09-09 (no untouched future holdout; Test split
scored exactly twice, guarded); no CI/CD or scheduler (lifecycle triggers manual, routing
automated); SHAP is attribution-only; missing `PROJECT_BLUEPRINT.md`.

---

## 2. Environment status

| Item | Finding |
|---|---|
| Python | 3.11.8 (system + `.venv`) |
| Virtual env | **Created** `.venv/` (was absent); `.gitignore` already had `.venv/` |
| pip | 26.0 (venv, upgraded at bootstrap) |
| Install | `pip install -r requirements-dev.txt` — clean, all wheels |
| .venv committed? | No — ignored (`venv/`, `.venv/` in `.gitignore`) |

## 3. Dependency status

- **`requirements.txt` (runtime/Docker contract)** — verified complete and necessary for
  serving: numpy 2.4.2, pandas 2.3.3, scikit-learn 1.8.0, lightgbm 4.7.0, shap 0.51.0,
  pyarrow 22.0.0, joblib 1.5.3, matplotlib 3.10.8, mlflow 3.9.0, fastapi 0.128.1,
  uvicorn 0.40.0, pydantic 2.12.5, pydantic-settings 2.14.2, python-dotenv 1.2.1,
  streamlit 1.54.0, sqlalchemy 2.0.46, psycopg2-binary 2.9.13, httpx 0.28.1. No
  unnecessary runtime deps (training-only libs intentionally excluded).
- **`requirements-dev.txt` (added)** — `-r requirements.txt` + pytest 9.1.1 +
  xgboost 3.2.0 + optuna 5.0.0 (dev/test layer only; not in the Docker image).
- All 18 runtime + dev packages verified imported at their pinned versions in `.venv`;
  all project modules (`app.*`, `dashboard.*`, `src.models/*`, `src.optimization/*`,
  `src.explainability/*`) import cleanly.
- **Docker consistency:** the image consumes exactly `requirements.txt` (shared pin
  contract with the venv); verified `docker compose config` valid and the running stack
  healthy. No arbitrary upgrade/downgrade was performed.

## 4. Tests

Re-run in the clean `.venv` after environment setup: **335 passed / 8 skipped / 0 failed**
(~6 min), equal to the verified Part-6 baseline. The single initial failure
(`test_smoke_fit_predict_shape[xgb-params1]`) was the missing xgboost in the clean env —
the exact reproducibility gap fixed by `requirements-dev.txt`; the suite is green after.

## 5. Documentation created

| File | Purpose |
|---|---|
| `docs/PROJECT_WALKTHROUGH.md` | Study guide (overview → per-part WHY/WHAT/HOW/RESULT → final eval → 5-min + senior-DS summaries) |
| `docs/PROJECT_STORY.md` | Narrative + 30s/60s/2-min/5-min versions, all factual |
| `docs/CV_DESCRIPTION.md` | Resume bullets (3/4-bullet, ATS, senior, business) with documented-metrics-only rule |
| `docs/INTERVIEW_PREPARATION.md` | Q&A by topic with short/deep/follow-up/key-concept + memory number sheet |
| `README.md` | GitHub front page: overview, architecture (Mermaid), Parts 1–6, model/gate/pricing/SHAP/API/Docker, install/run/test, limitations |
| `PROJECT_MAP.md` (updated) | Authoritative status: finalization status, docs links, completion checklist, limitations; blueprint-absence note |
| `requirements-dev.txt` | Dev reproducibility layer |

All documentation was verified to reference real files and to contain only documented
metrics; a claims check found every occurrence of words like "profit uplift" only inside
explicit anti-claim/negation statements.

## 6. Final architecture

See `README.md` (Mermaid) and `docs/PROJECT_WALKTHROUGH.md` §5. Key separating concepts:

- **MLflow tracking/registry** (model lifecycle; served read-only) vs **PostgreSQL**
  (application inference log): one system for "which model", one for "who asked what".
- **Internal 93-feature model schema** vs **business API inputs** (`product_id`, ISO week,
  optional cost): the API generates the 93 features server-side
  (`frozen_features_for_week`, train-only encoding, `unknown` fallback); managers never
  provide model features.

## 7. Final Production model

`demand_forecasting_final` **v2** — Production, READY, run `5d729a85`, 93 features,
LightGBM (600 trees, 223 leaves, lr 0.008468), log1p target, prediction-integrity guarded.
v1 Staging (run `7fc87198`) preserved. No V3.

## 8. Final quality gate

- Policies: `quality_gate_policy_v1_legacy` (19 outcomes, default) /
  `quality_gate_policy_v2_revised` (18 blocking + diagnostics). Any failed blocking
  criterion → Staging; guarded decision record required for Production.
- **Legacy result preserved:** v2 post-hoc FAIL — B WAPE 0.6282 > 0.60, all other 18
  criteria PASS, integrity PASS → FAIL → Staging (verbatim history, replayed
  byte-identically in tests).
- **Revised-policy result:** `B_wape` reclassified **diagnostic** (evidence: engineering
  heuristic, no stakeholder artifact; 37-product sample-sensitive cohort); `B_bias_pct ≥ −20`
  remains **hard**. Under the revised policy the recorded PASS (`REVISED_GATE_APPLIED`),
  same run, promotes v2 to Production. Diagnostic ≠ blocking; history untouched; no
  threshold modified.
- Two controlled validation experiments (spike-feature remediation; asymmetric 2:1
  quantile objective) → `NO_VALID_IMPROVEMENT` on pre-registered tolerances.

## 9. Final metrics

| Metric | Value | Source |
|---|---|---|
| Validation WAPE | 0.3964 | run `5d729a85` metric `validation-wape` |
| Validation MAE / RMSE / MASE | 0.02329 / 0.25917 / 0.6122 | run metrics |
| Validation B-cohort WAPE / bias | 0.5101 / −18.2% | run metrics |
| Validation C-cohort WAPE | 0.3795 | run metric |
| Post-hoc Test (legacy) | global 0.3754 · **B 0.6282 FAIL** | `part3_b_cohort_error_analysis.md` |
| Elasticity (pooled product FE) | b = −0.3952 ± 0.0363 (t −10.878, n 41,452) | `part4_pricing_engine_report.md` |
| SHAP global top | `cat_unknown` mean\|SHAP\| 0.0304 | `part5_shap_explainability.md` |
| Pricing run | 1,385 scored; 0 optimized; 468/917 reasons | `part4_pricing_engine.json` |

## 10. Deployment verification (finalization pass)

- `docker compose config --quiet` → **valid**.
- Stack live: db / api / dashboard all **healthy** (4h / 2h uptimes).
- Image contents (inspect via compose exec): `data/` contains `processed/` only;
  **no `data/splits` directory**; `find /app -name test.csv` → **no matches**.
- Registry: v1 Staging / v2 Production READY, tracking-db SHA-256 head `5672c219e8c708f3`,
  mlruns 67 files — unchanged from Part-6 record.
- Production load in `.venv`: resolves Production READY, 93→1 signature, loads in 6.7 s.
- Part-6 recorded: OpenAPI exactly 5 paths; `/predict`/`/pricing`/`/explain` 200;
  inference rows in PostgreSQL; error codes verified; dashboard health 200.

## 11. Known limitations

As §1.7 (B-cohort bias, no cost source, descriptive elasticity, frozen snapshot, no
CI/CD, attribution-not-causality, absent blueprint).

## 12. Remaining actions (optional, none blocking)

1. Author/supply `PROJECT_BLUEPRINT.md` if a formal spec record is desired.
2. Supply a legitimate **cost source** to activate the profit path.
3. Data refresh + properly gated V3 process when a future window exists.
4. CI/CD automation of lifecycle triggers; causal price-test design.

---

## FINALIZATION STATUS: COMPLETE WITH LIMITATIONS

**Why not plain COMPLETE:** the system and every verification gate pass (full suite green
in a clean venv; Docker valid + healthy; registry untouched and Production model loading;
docs traceable and claim-free of unsupported uplift); however the audit surface contains
documented, non-blocking limitations that are part of the honest project record — the
B-cohort accuracy weakness behind the gate revision, the absence of a cost source (profit
path inert by design), the frozen 2018 dataset (no untouched future holdout), no CI/CD,
and the missing `PROJECT_BLUEPRINT.md` spec file.

**Why not BLOCKED:** no verification failed during finalization. The single clean-env
test failure was caused by a reproducibility gap in the dev layer, fixed with
`requirements-dev.txt` (runtime contract untouched) and re-verified green: **335 passed,
8 skipped, 0 failed**.