# Part 6 — Production Deployment Implementation Report

**Verdict: PART 6 COMPLETE**

Served model: `demand_forecasting_final` **v2** — stage `Production`, status `READY`,
run `5d729a8594dc488bbd18ba0c4e216f5a`, source `models:/m-e93cadaf963a4aa1b44b3e501379da92`
(93 signature inputs, LightGBM `LGBMRegressor`, `n_features=93`).

---

## 1. Scope and non-negotiables

- **Business-level API only**: `product_id` + optional ISO week + optional `cost`.
  The 93-feature schema and `data/splits/test.csv` are never exposed, read, shipped, or logged.
- **Immutable model**: no retrain, no V3, no Part 3–5 changes, no registry writes.
- **Frozen pipeline reused verbatim**: `frozen_features_for_week` →
  `prepare_model_matrix` for served features; `demand_engine/simulation/optimizer/
  guardrails` reused as-is for pricing; `shap_analysis`-equivalent TreeExplainer path.
- **Synchronous deployment**: FastAPI, single uvicorn worker (in-memory `AppState`).

## 2. What was built

| Area | Deliverable |
|---|---|
| FastAPI service | `app/` — config, schemas, errors, dependencies, exception handlers, SQLAlchemy inference-log (`db/`), services (model / state / prediction / pricing / explainability), 5 endpoints, middleware (request-id, structured logging, payload limit, optional rate limit) |
| UI | `dashboard/` — Streamlit manager (forecast / pricing / explainability / system status). Named `dashboard/`, **not** `streamlit/`, so it never shadows the installed framework |
| Packaging | `deployment/Dockerfile` (python:3.11-slim, libgomp1, non-root uid 10001, `PYTHONPATH=/app`), `deployment/docker-compose.yml` (db → api → dashboard, healthchecks gated by `/ready` and `/_stcore/health`), `deployment/healthcheck.py`, `.dockerignore` |
| Tests | `tests/deployment/` — 45 fast + 8 real-model integration tests (total 53) |

### Endpoints
- `GET /health` — liveness (`app_env`, version, UTC time).
- `GET /ready` — readiness gate: 2xx only when `model_loaded`, `feature_state`,
  `elasticity_state`, `shap_runtime`, `warmed_up` are all ready (DB is informational).
- `POST /predict` — demand forecast for a product/week; default week
  `2018-05-14` (last Val week, `LAST_VAL_WEEK`).
- `POST /pricing` — scenario simulation + optimal price; price/cost **never**
  invented (business reasons `NO_PRICE_OBSERVATION`, `NO_ELASTICITY`,
  `PROFIT_NOT_COMPUTABLE (no cost supplied)`).
- `POST /explain` — SHAP attribution (business labels + technical `feature_code`),
  exact per-row reconstruction `served_units = expm1(base_log1p + Σφ)`.

### Error contract
Structured JSON always: `{detail, code, request_id, extra}`. Codes: `PRODUCT_NOT_FOUND`,
`INVALID_WEEK`, `WEEK_OUT_OF_SUPPORTED_RANGE`, `TEST_BOUNDARY_VIOLATION`, `INVALID_COST`,
`MODEL_NOT_READY`, `FEATURE_BUILD_FAILED`, `ELASTICITY_UNAVAILABLE`, `NO_ELASTICITY`,
`PROFIT_NOT_COMPUTABLE`, `MODEL_LOAD_FAILED`, `INTERNAL_ERROR`, `VALIDATION_ERROR`,
`PAYLOAD_TOO_LARGE`, `RATE_LIMITED`. 422 for domain/schema, 503 for
readiness/infrastructure. Stack traces never leak (`INTERNAL_ERROR` handler).

### Week contract
All weeks are ISO Mondays. `week >= 2018-05-21` → `TEST_BOUNDARY_VIOLATION`; non-Monday →
`INVALID_WEEK`; week absent from the served matrix → `WEEK_OUT_OF_SUPPORTED_RANGE`;
supported historical Mondays (e.g. `2018-01-01`) → response 200.

## 3. Model loading (registry-verified + portable)

The registry v2 entry’s `source = models:/m-e93cadaf963a4aa1b44b3e501379da92`; its
artifact directory is resolved **by model id** from `mlruns/*/models/*/` (a registry-read
+ directory scan — no direct `models:/` URI, which is not guaranteed portable).

Discovered container defect, and deployment-layer workaround:

- The immutable `MLmodel` (MLflow 3.x layout) stores
  `flavors.python_function.artifacts.model.path = "artifacts\\model.joblib"` — a
  Windows backslash written at logging time. On case-sensitive POSIX inside the
  container, `mlflow.pyfunc.load_model` joins it with `os.path.join` and fails
  (`.../artifacts/artifacts\model.joblib` not found).
- Deployment fix (`app/services/model_service.py::_load_normalized_pyfunc`): when the
  primary load raises, copy the (4.5 MB) artifact dir to a temp dir, normalize
  backslashes in `flavors.python_function.artifacts` (and legacy top-level `data`),
  then let mlflow load that copy — the **exact same load path** as Windows.
- Verified parity: `tests/deployment/test_deployment_model_service.py::test_path_normalized_load_matches_primary`
  asserts the normalized model’s signature == primary load’s signature (93 inputs,
  same names) and identical predictions. The container logs the fallback warning and
  serves v2 run `5d729a85` (confirmed in every response `model` block).

This is a **justified portability deviation**, documented in
`reports/deployment/part6_deployment_architecture_audit.md` and tested.

## 4. Validation evidence

| Check | Result |
|---|---|
| Full repo suite | **335 passed, 8 skipped (deployment integration), 0 failed** (~6 min) |
| Deployment fast suite | **45 passed** (API/OpenAPI, errors, persistence, model-service, safety) |
| Real-model integration (`RUN_DEPLOYMENT_INTEGRATION=1`) | **8 passed** (285 s real startup: matrix, frozen features, elasticity, SHAP, predict/pricing/explain + DB writes) |
| Docker compose smoke | all containers **healthy**; `/health` 200, `/ready` 200; real `/predict` 200 (`predicted_demand` served), `/pricing` 200 (`NO_PRICE_OBSERVATION` — no price invented), `/explain` 200 (`additivity_residual_demand_units = 0.0`); inference rows present in PostgreSQL (`inference_records`: predict/pricing/explain × 1) |
| OpenAPI surface | exactly 5 paths: `/health`, `/ready`, `/predict`, `/pricing`, `/explain` — no feature-schema endpoint |
| Image hygiene | `data/` contains only `processed/demand_matrix_weekly.csv.gz`; **no** `data/splits`, **no** `test.csv`; registry present read-only |
| Registry integrity | read-only resolution snapshot test; no `register_model`/`transition_model_version_stage`/`log_model`/`set_alias` anywhere in served code (AST scan) |
| Error codes (container) | unknown product → 422 `PRODUCT_NOT_FOUND`; non-Monday → 422 `INVALID_WEEK`; `2018-05-28` → 422 `TEST_BOUNDARY_VIOLATION`; supported week `2018-01-01` → 200 |
| Safety scans | no `test.csv`/`data/splits` literals; no `.fit()`; no `optuna`/`xgboost` imports in served code; dashboard never renders feature names |

## 5. Operational notes

- **Start containers**: `cd deployment; docker compose up --build -d`
  (api `http://localhost:8000`, dashboard `http://localhost:8501`).
  `/ready` gates traffic; healthcheck has `start_period: 600s` (measured warm-up
  ~2–4 min in-container).
- **Config**: `.env.example` → every setting has a production-safe default; compose
  overrides set registry paths, `REFERENCE_WEEK`, `DATABASE_URL`, CORS, log level.
- **Infrastructure**: PostgreSQL 16 (`inference`/`inference`, dev-only bootstrap
  creds). DB unreachable at startup → service still serves, `inference_logged=false`.
- **Dev credentials / secrets**: none shipped; `.env` excluded by `.dockerignore`.
- **Known operational caveat**: matrix/feature warm-up takes minutes; a second load of
  the demand matrix is intentional (demand-baseline reference week + AppState feature
  state) — double-load accepted in design.

## 6. Deliverables

- `app/` (service) · `dashboard/` (UI) · `deployment/{Dockerfile,docker-compose.yml,healthcheck.py}`
- `.dockerignore` · `requirements.txt` (runtime pins) · `tests/deployment/` (53 tests)
- `reports/deployment/inference_request_schema_audit.md`
- `reports/deployment/part6_deployment_architecture_audit.md`
- `reports/deployment/part6_deployment_implementation_report.md` (this file)
- `PROJECT_MAP.md` — Part 6 marked COMPLETE, deliverables + reference points added

**Verdict: PART 6 COMPLETE** — the Production model is deployed behind a
business-level FastAPI service with a live PostgreSQL inference log and a Streamlit
manager UI, fully containerized and verified end-to-end.