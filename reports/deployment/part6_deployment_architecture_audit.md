# PART 6 — DEPLOYMENT ARCHITECTURE AUDIT

Date (UTC): 2026-09-25 · Scope: audit-only deliverable for the Deployment phase (FastAPI + Docker + Streamlit). No deployment code is changed by this document.

**VERDICT — APPROVED TO IMPLEMENT**, subject to the guardrails in §I and §K. The existing system is a clean, layered, fully-tested engine that Deployment can *consume*; the required changes are strictly additive and aligned with the non-negotiable safety rules.

---

## A. Current Architecture (as-found)

### A.1 Repo layout (verified on disk)
```
E-Commerce/
├── PROJECT_BLUEPRINT.md / PROJECT_MAP.md        # canonical specs + live tracking
├── data/
│   ├── raw/        (Olist CSVs, read-only source)         # NOT used at runtime
│   ├── processed/  demand_matrix_weekly.csv.gz (3,469,274 rows, 32,729 products × 106 weeks)
│   └── splits/     train.csv / val.csv / test.csv         # test.csv PROTECTED — never used
├── src/
│   ├── data/         data_processing, target_engineering, temporal_split, imputation
│   ├── features/     feature_engineering (lag/rolling/regime/calendar/drivers)
│   ├── models/       forecasting, data_prep, final_forecast, pyfunc_wrapper, registry,
│   │                 quality_gate, lifecycle_router, finalize_lifecycle/v2, etc.
│   ├── optimization/ pricing_data, elasticity, simulation, optimizer, guardrails,
│   │                 demand_baseline, pricing_engine
│   ├── explainability/ shap_analysis (+ __init__)
│   ├── notebooks/    (research only)
│   └── utils/        logging_setup, metrics
├── artifacts/
│   ├── mlflow_tracking.db      SQLite tracking + model registry (752 KB)
│   ├── phase3/                model.joblib + feature_columns.json (staged duplicate, non-serving)
│   ├── optimization/          recommended_prices.csv (Part-4 batch output)
│   └── explainability/        Part-5 artifacts (PNGs/parquet/csv/metadata)
├── mlruns/                    artifact store backing the registry (root-level; MUST ship)
│   └── 2/models/m-{8e79f...,e93cad...}/artifacts/{MLmodel, artifacts/model.joblib, code/}
├── mlflow.db                  stray root-level file — NOT part of serving (housekeeping)
├── reports/forecasting/       Part 3/4/5 reports (md+json) — history, must be preserved
├── tests/                     24 suites == 290 tests (all green, 0 skipped)
└── (no requirements.txt, no .env, no .gitignore, no Dockerfile, no README, no pytest.ini)
```

### A.2 MLflow / model lifecycle (verified)
- Tracking + Model Registry in **one SQLite file**: `artifacts/mlflow_tracking.db`, URI `sqlite:///artifacts/mlflow_tracking.db` (`src/models/registry.py:44-47`).
- Registered model **`demand_forecasting_final`**, **version 2 = Production** (`run 5d729a85…`, status READY); v1 Staging — promoted by the gate (`transition_model_version_stage`) in `registry.register_and_promote`, now **guarded** by `lifecycle_router.guarded_promote` (decision-record contract; Production requires PASS; run_id provenance; policy tag). Deployment must **consume** this lifecycle and never reimplement/override it.
- Serving contract: MLflow **PyFunc** `DemandForecaster` (`pyfunc_wrapper.py`) — validates the frozen **93-column schema** on every call, serves business units `expm1(log output)` floored at 0. Underlying estimator captured via `unwrap_python_model()._model` (lightgbm.sklearn.LGBMRegressor, 600 trees).
- Model artifacts (incl. bundled `code/` copy of `src/{data,features,models,utils}`) live under `mlruns/2/models/*/artifacts/` — **required in the image** so `models:/demand_forecasting_final/2` resolves offline.

### A.3 What already exists for Deployment to consume (no rewrites needed)
| Capability | Module(s) | Interface |
|---|---|---|
| Production model load | `optimization/demand_baseline.load_production_model(uri)` | `mlflow.pyfunc.load_model` (read-only) |
| Frozen 93-col feature build (on-the-fly) | `demand_baseline.frozen_features_for_week(matrix, week, model)`; `shap_analysis.reference_frame` | matrix → design rows, train-only imputation, unseen→`unknown` |
| Demand baseline Q0 | `predict_baseline(model, features)` / wrapper `predict` | units |
| Elasticity | `optimization/elasticity.estimate(train, val)` | descriptive log-log; per-product status/source |
| Simulation grid | `simulation.simulate_scenarios` / `simulate_panel` | P±{5..20}% grid |
| Optimizer | `optimizer.choose_optimal` / `optimal_price_from_panel` | profit (needs cost) else revenue-proxy |
| Guardrails | `guardrails.apply_guardrails` | margin floor (cost-gated), ±20% ceiling, floor, smoothing |
| SHAP | `explainability.shap_analysis` (`resolve_estimator`, `define_sample`, `compute_shap`) | log1p contributions, per-row base additivity-corrected |
| Logging | `utils/logging_setup.get_logger` | single stdout handler |

### A.4 Config / packaging gaps (verified)
- **No** `requirements.txt`, `.env`, `.env.example`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `README.md`, or `pytest.ini`. `conftest.py` only bootstraps `sys.path`.
- Runtime data at serving time = `data/processed/demand_matrix_weekly.csv.gz` + `artifacts/mlflow_tracking.db` + `mlruns/`.
- Environment: Python 3.11.8; fastapi 0.128.1, uvicorn 0.40.0, streamlit 1.54.0, pydantic 2.12.5 (pydantic-settings 2.14.2, python-dotenv 1.2.1), mlflow 3.9.0, lightgbm 4.7.0, xgboost 3.2.0, optuna 5.0.0, pandas 2.3.3, numpy 2.4.2, scikit-learn 1.8.0, pyarrow 22.0.0, shap 0.51.0, matplotlib 3.10.8, joblib 1.5.3, httpx 0.28.1, docker SDK 7.1.0. **Docker daemon reachable (engine 29.6.2) — real `docker build` verification is practical.**

### A.5 Is the current structure appropriate? 
**Yes, with additive change only.** Existing structure is clean and layered. No existing folder is displaced; Deployment adds a thin new consumer layer (`app/`, `streamlit/`, `deployment/`, root config files).

---

## B. Proposed Architecture

```
                          ┌────────────────────────────┐
                          │  Streamlit Manager Dash.  │  (streamlit/ — thin UI, no business logic)
                          └────────────┬───────────────┘
                                       │  HTTPS/HTTP (httpx)
                         ┌─────────────▼──────────────┐
                         │  FastAPI REST backend      │  (app/ — validation, orchestration, metadata, errors)
                         └─────────────┬──────────────┘
                    ┌──────────────────┼────────────────────┐
                    ▼                  ▼                    ▼
            app/services/predict    app/services/explain  app/services/pricing
                    │                  │                    │
        ┌───────────┤                  │              ┌─────┴────────────────────┐
        ▼           ▼                  ▼              ▼            ▼              ▼
  MLflow PyFunc src/optimization  Part-5 SHAP layer  elasticity   simulation   guardrails
  (Production,   demand_baseline  (reuse: estimator  (lookup)     (grid)      (business
  models:/…/2,   frozen features  resolution +       └─────▲───────────────────────┘
  read-only)     93-col schema          compute_shap)      │
        │                              method)             │
        └───────────────┬───────────────────────────────────┘
                        ▼
     AppState caches (built once at startup, no per-request heavy work):
        model + estimator + schema · reference-week frame (93 cols) · TreeExplainer +
        deterministic background · elasticity state (per-product status/source/elasticity)
```

Principles:
1. **Consume, never reimplement** — endpoints call the existing Part-3/4/5 modules verbatim. No duplicated feature/SHAP/pricing logic in the API.
2. **Online vs offline (§10)** — request-path work is only: predict a row, explain a row, simulate+optimize+guardrail a row. Heavy work (global SHAP, batch pricing, elasticity *estimation*) is offline/startup-cached.
3. **Test protection** — the service reads only `data/processed/*` + MLflow registry; `data/splits/test.csv` is unreachable by construction (no import path, no `test.csv` token, `.dockerignore` excludes it).
4. **Lifecycle respect** — the app resolves the model from the registry (name + alias/stage/version) **read-only**; nothing promotes, registers, retrains, or bypasses the gate.
5. **Honest cost handling** — pricing never invents cost; without valid cost the response is `PROFIT_NOT_COMPUTABLE` (revenue-optimal scenario flagged), exactly as in Part 4.

---

## C. Folder Structure Changes

Legend: `REQ` = required, `OPT` = optional / deferred. "Affects existing?" = whether existing code/tests change.

### C.1 `app/` — NEW (REQ)
- **Proposed**
  ```
  app/
    __init__.py
    main.py        # FastAPI app factory + lifespan wiring + routers
    config.py      # pydantic-settings Settings, reads .env (app-level config; MLflow keys etc.)
    schemas.py     # Pydantic v2 request/response models (no internal DataFrames leak)
    api.py         # one router: /health /ready /model/info /predict /explain /pricing/recommendation
    services/
      __init__.py
      state.py     # AppState: model/estimator/schema/reference frame/explainer/elasticity caches (lazy, thread-safe)
      predict.py   # single-product-week predict adapter -> wrapper.predict (exact served contract)
      explain.py   # single-row SHAP adapter -> Part-5 resolve_estimator/define_sample/compute_shap method
      pricing.py   # single-product pricing adapter -> simulation/optimizer/guardrails shortcuts
  ```
- **Why needed / responsibility / insufficiency of current structure:** there is no serving entry point today; business logic was built and tested as library code only. `app/` owns *coordination* (validation → orchestration → structured responses), never *business logic*.
- **Affects existing code?** No. It imports from `src/*` only.
- **Required?** Yes — this is the FastAPI layer.

### C.2 `streamlit/` — NEW (REQ)
- **Proposed**
  ```
  streamlit/
    __init__.py
    app.py        # manager dashboard (4 sections, thin)
    api_client.py # typed httpx client for the FastAPI backend
    config.py     # dashboard settings (API_BASE_URL etc.)
  ```
- **Why:** business-facing UI; must not carry business logic or import `models:/` directly — it talks to FastAPI only.
- **Affects existing code?** No. **Required?** Yes.

### C.3 `deployment/` — NEW (REQ)
- **Proposed:** `Dockerfile`, `docker-compose.yml`, `.dockerignore` (placed here so root stays clean; build context = repo root).
- **Why:** single home for image/compose definitions; keeps Docker concerns out of the app code.
- **Affects existing code?** No. **Required?** Yes.

### C.4 Root config files — NEW (REQ)
- `requirements.txt` (§E), `.env` + `.env.example` (§D), `.gitignore` (ignore `.env`, caches, temp), `README.md` (deployment runbook).
- **Why:** no packaging/configuration exists today (A.4). **Affects existing code?** No (adds only).
- **Required?** Yes. `README.md` is `REQ` for deployment runbook (project convention documents everything).

### C.5 Explicitly NOT added (proliferation control)
- **`config/` package** — `.env` + `app/config.py` (pydantic-settings) suffice; a separate config *directory* adds no ownership.
- **`app/schemas/` directory** — a single `app/schemas.py` owns the ~8 models.
- **`app/api/` directory** — one router module `app/api.py` is enough for 6 endpoints.
- **`artifacts/mlflow/`, `artifacts/models/` move** — the existing artifact layout is the *live registry*; moving it would rewrite history (forbidden).
- **Notebooks/dev-only folders into the image** — excluded by `.dockerignore`.

---

## D. Configuration Plan

### D.1 `.env` (environment-specific, git-ignored) + `.env.example` (safe placeholders)
```
ENVIRONMENT=production
MLFLOW_TRACKING_URI=sqlite:////app/artifacts/mlflow_tracking.db
MLFLOW_MODEL_NAME=demand_forecasting_final
MLFLOW_MODEL_VERSION=2            # optional: explicit pin
MLFLOW_MODEL_ALIAS=Production      # optional: resolve by stage
REFERENCE_WEEK=2018-05-14          # last Part-4 validation week (< test boundary)
SHAP_BACKGROUND_SIZE=400
SHAP_EXPLAIN_SIZE=1500
SHAP_SEED=20260924
API_PORT=8000
STREAMLIT_PORT=8501
STREAMLIT_API_BASE_URL=http://api:8000
PYTHONPATH=/app                    # container
```
- `.env.example` holds the same keys with empty/placeholder values; never secrets.
- **Verdict on the prompt's suggested keys:** `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_NAME`, `MLFLOW_MODEL_VERSION/ALIAS`, `ENVIRONMENT` are correct and match the real registry; the service resolves the model **from the registry by name + (alias→version resolution, read-only)**, so a future re-promotion is picked up without code changes. `MLFLOW_MODEL_STAGE` is not needed — alias/stage resolution covers it.
- **Secret management note:** `.env` is *configuration management*, not secret management. This project contains **no real secrets** (SQLite registry, local model). A production cloud deployment would need a dedicated secret manager (e.g., cloud KMS/SSM) for credentials — documented, not invented here.

### D.2 MLflow configuration
- `app/config.py` calls `mlflow.set_tracking_uri(settings.mlflow_tracking_uri)` at startup **before** any loader runs, and the app passes the configured URI/name/version to the existing `load_production_model` / `model_info_from_registry` call sites (both already accept a URI). In-container the repo root is `/app`, so `src/models/registry.TRACKING_URI` (computed from `parents[2]`) coincides with the configured URI — verified at image-build time.
- The model-dir bundled `code/` (identical to repo `src/`) is what pyfunc loading uses; repo `src/` is on `PYTHONPATH` for the service modules. Duplicate but byte-identical modules — no conflict (validated in integration tests).

### D.3 Model configuration
- Resolution order: `MLFLOW_MODEL_VERSION` (pin) → `MLFLOW_MODEL_ALIAS` (find version with `current_stage==alias`) → error listing registry versions. The app **never** calls `transition_model_version_stage` / `register_model` / `log_model` (static-checked).

---

## E. Dependency Plan (`requirements.txt`)

Pinned to the verified working environment; runtime-only.

```
mlflow==3.9.0
numpy==2.4.2
pandas==2.3.3
scikit-learn==1.8.0
lightgbm==4.7.0
shap==0.51.0
pyarrow==22.0.0
joblib==1.5.3
matplotlib==3.10.8
fastapi==0.128.1
uvicorn==0.40.0
pydantic==2.12.5
pydantic-settings==2.14.2
python-dotenv==1.2.1
httpx==0.28.1
streamlit==1.54.0
docker==7.1.0        # deployment/verification tooling (pip/monitor scripts; optional runtime)
```
- **Dropped (deliberately):** `xgboost`, `optuna` — used only by the *offline training/tuning/experiment* suite, not by the deployed application (the service loads an already-frozen LightGBM PyFunc). Kept in the dev environment; documented in README.
- **Notes:** lightgbm's sklearn API requires scikit-learn; shap requires matplotlib; `docker` SDK used by the Docker verification test (kept, since the deployment test plan runs `docker build`).
- `psutil`/`watchdog`/`altair` are pulled transitively by streamlit — not pinned explicitly.

---

## F. API Design

All endpoints under `/api/v1` (versioned); structured JSON errors (RFC-style): `{detail, code, request_id}`; HTTP 400/422 for validation, 503 for readiness/model-unavailable, 500 wrapped. No internal DataFrames leak — Pydantic models serialize only safe fields.

| Endpoint | Purpose (verdict) | Responsibility |
|---|---|---|
| `GET /health` (needed) | Liveness | App process alive + DB/registry path configured. Always 200 when the process runs. |
| `GET /ready` (needed) | Readiness | Verifies: tracking URI set, Production model resolvable + loaded, 93-col schema, reference frame built, background/explainer built; returns component status map; 503 w/ detail when incomplete. Elasticity cache reported as a readiness component (built once, lazily). |
| `GET /model/info` (needed) | Safe metadata | Registry record (name, version, stage/status, run_id, source) + estimator type + schema count + environment. Read-only registry query. |
| `POST /predict` (needed) | Demand forecast | `{product_id, week}` → wrapper predict (exact served contract, float64) → `{predicted_demand_units, model_metadata, scale}`. Week must be a matrix week-start ≤ reference week and Part-4 train/val (never test); unknown product/week → structured 422. |
| `POST /explain` (needed) | Single-row SHAP | `{product_id, week}` → prediction + base_log1p + top positive/negative contributors (name, contribution, direction_label) + additivity note + model metadata. Reuses Part-5 estimator resolution, deterministic background sampling, TreeExplainer, per-row base correction. Response labels clearly: **model feature contribution — not causal**. |
| `POST /pricing/recommendation` (needed) | Per-product pricing | `{product_id, week(=reference, default), current_price, cost?}` → baseline Q0 (model), elasticity (status/source), scenario grid (P±5..20%), optimal (profit if cost>0 else revenue-proxy), guardrail outcome. Without valid cost: `reason=PROFIT_NOT_COMPUTABLE`, `profit_computable=false`, **no invented recommendation**; guardrail margin-floor inactive+flagged. `current_price<=0` or `cost<=0` → 422. |

- SHAP vs elasticity disambiguation is explicit in every response schema (`attribution` vs `elasticity`, both non-causal).

---

## G. Streamlit Manager Dashboard Design

Thin UI over FastAPI (`streamlit/api_client.py` → httpx); no business logic, no MLflow imports.

- **Model Overview:** calls `/model/info` + `/ready`; shows active model name/version/status/run_id, environment, schema count, readiness component badges.
- **Demand Forecast:** `product_id + week` form → `/predict`; shows predicted weekly demand (units) + model metadata + scale note.
- **Explainability:** for the same product/week → `/explain`; renders "Prediction → Top Positive Contributors / Top Negative Contributors" with contributions; banner: **"Model explanation / feature contribution — not causal."**
- **Pricing:** `product_id + week + current_price (+ optional cost)` → `/pricing/recommendation`; shows current price, elasticity status/source, scenario demand/price table, recommended price, guardrail statuses; when cost missing shows **"Profit optimization unavailable: valid product cost was not supplied."** and no fabricated price.

---

## H. Docker Design

- **Image:** `python:3.11-slim`; `apt-get install -y libgomp1` (LightGBM); `pip install -r requirements.txt`; **non-root user** (`appuser`, `USER appuser`); `WORKDIR /app`; `ENV PYTHONPATH=/app`.
- **Copied into image (minimal context, `.dockerignore`):** `src/` (services + identical bundled model code), `app/`, `streamlit/`, `requirements.txt`, `data/processed/demand_matrix_weekly.csv.gz` (only the matrix — not `data/raw` or `data/splits`), `artifacts/mlflow_tracking.db`, `mlruns/` (minus `.trash`) — the registry artifact store the DB references.
- **Never copied:** `data/splits/**` (incl. `test.csv`), `data/raw/**`, `tests/`, `reports/`, `src/notebooks/`, root `mlflow.db`, `.env`, `**/__pycache__`, `.pytest_cache`, temp experiment caches.
- **Runtime/env:** `.env` passed via compose `env_file` (never baked into the image); `MLFLOW_TRACKING_URI` resolves to the in-container path.
- **Entrypoints / ports:**
  - `api`: `uvicorn app.main:app --host 0.0.0.0 --port 8000` → port **8000**.
  - `dashboard`: `streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501` → port **8501**.
- **Health checks:** api → `GET /health` via `python -c "urllib.request.urlopen(...)"` (curl not in slim); dashboard → TCP probe on 8501.
- **Compose:** two services, `api` (healthcheck) and `dashboard` (`depends_on: api`); shared `env_file: .env`; ports mapped.
- **Security:** no secrets in Dockerfile; non-root; bind 0.0.0.0 (internal network; auth/rate-limiting documented as out-of-scope gateway concerns in README).

---

## I. Security / Reliability

| Risk | Mitigation |
|---|---|
| Test-split exposure | No `data/splits` copied into image; AST static guard denies `test.csv`/`splits` tokens in `app/`+`streamlit/`; week validation blocks `assign_period=="test"`; integration tests assert `test_split_accessed` semantics. |
| Model tampering / wrong model | Read-only registry resolution (alias/version), readiness validates model identity + 93-col signature; static guard forbids `transition_model_version_stage`/`register_model`/`log_model`/retraining imports in app code. |
| Secrets | None exist; `.env` git-ignored; .dockerignore excludes `.env`; README documents secret-manager requirement for real cloud. |
| Heavy work in request path | Elasticity estimation + global SHAP + batch pricing stay offline/startup-cached; request path is single-row ops. |
| Model drift / additivity caveat | Explain responses carry the documented additivity correction + residual disclosure from Part 5. |
| Availability | Health/readiness separation; graceful 503 on incomplete state; mlflow artifact store + DB bundled (offline-capable). |
| Container isolation | Non-root user; shutdown of interpreter on unhandled errors (uvicorn singles worker decision documented). |

---

## J. Testing Plan (before Deployment is complete)

New suites (additive; the existing **290-test suite must stay green**):

1. **Config/safety tests** — `.env.example` ↔ settings parity; defaults; invalid env rejected; AST scan of `app/`+`streamlit/`: no `test.csv`, no `splits`, no `transition_model_version_stage`, no `register_model`, no `log_model`, no training/tuning imports (`optuna`, `xgboost`, `.fit(`); no `data/raw` read.
2. **Schema tests** — Pydantic: required fields, numeric ranges, invalid/missing week (bad format, test-period week), invalid price/cost (≤0, NaN), product id format; structured error shape.
3. **API unit tests (FastAPI TestClient, isolated state)** — `/health`; `/ready` (full, degraded→503); `/model/info` identity; `/predict` valid + missing product/week + week ≥ test boundary + product with no history; `/explain` valid + disclaimers present + top contributors ordering/sign; `/pricing/recommendation` with/without cost (without → `PROFIT_NOT_COMPUTABLE`, guardrail margin-floor inactive flag, no fabricated price; with cost → profit objective + margin floor active); model-unavailable simulation → 503; structured error shape everywhere.
4. **Integration tests (real Production model, real registry, real matrix)** — FastAPI→MLflow PyFunc e2e; FastAPI→Part-5 SHAP layer; FastAPI→Part-4 pricing→guardrails; correctness: prediction equals direct wrapper predict; schema = 93 in registry order; additivity holds (per-row correction); pricing no-cost path identical to Part-4 engine semantics; elasticity source/status present.
5. **Streamlit tests** — module importability; api_client request/response mapping against mocked FastAPI (httpx MockTransport); dashboard render functions build without a server; AST no-test guard.
6. **Docker tests (gated on daemon availability; engine reachable here)** — `.dockerignore` correctness (no `splits`/`.env` context entries); `docker build` succeeds; container starts; `/health` returns 200 from inside; `/ready` OK; `.env` not present in image (`docker run --rm <img> ls` guard) via a light probe.
7. **Full-suite regression** — `python -m pytest tests -q` → green (baseline 290 + new suites).

---

## K. Approval Conditions Being Satisfied (non-negotiables → audit mapping)

| Rule | Status / proof |
|---|---|
| No retrain / no V3 / no Production-model change | Deployment loads `models:/demand_forecasting_final/2` read-only; static guard. |
| No MLflow history / version changes | Registry queries only; no transition/register/log paths. |
| No Part-3 gate/lifecycle-policy change | Gate + `lifecycle_router` untouched; app consumes stage resolution only. |
| No `data/splits/test.csv` access / no Test use | Not read any path; `.dockerignore`; AST; week-check ≤ reference week. |
| No new experiments, no feature/schema change | 93-col schema verified against signature; no schema edits. |
| Part-4 elasticity methodology unchanged | Service reuses `elasticity.estimate` inputs exactly (same periods/constants). |
| No invented cost / no fake profit | Same `PROFIT_NOT_COMPUTABLE` semantics; margin floor inactive w/o cost. |
| No replacement of tested functionality | Only new consumer modules; existing 290 tests must stay green. |

---

## Verdict

- **APPROVED TO IMPLEMENT.** Proceed in order: configuration (`requirements.txt`, `.env`, `.env.example`, `.gitignore`) → `app/` (FastAPI) → Streamlit dashboard → `deployment/` (Dockerfile + compose + .dockerignore) → integration + safety tests → real `docker build`/run verification → update `PROJECT_MAP.md` + `final_execution_report.md` (§15) with results and known limitations (incl. missing-cost dependency, offline/online split, no-auth-internal-network caveat).
- Single non-trivivial design decision to confirm with the owner: **elasticity estimation runs once at service startup (cached in `AppState`) rather than per-request** — this reuses the exact Part-4 methodology and keeps the request path single-row; alternative is a pre-built `artifacts/optimization/elasticity_state.json` produced by the existing Part-4 batch run. Default: compute-at-startup.