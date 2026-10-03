# Final Reproducibility Report

| | |
|---|---|
| **Project** | E-Commerce Demand Forecasting & Dynamic Pricing |
| **Check date (UTC)** | 2026-09-25 |
| **Nature** | Cold verification of the project from its lock-files, pinned runtime, registry, tests and deployment artifacts. No project logic was changed. No retrain. No new model version. No quality-gate change. The protected Test split was not accessed. |
| **Verdict** | **REPRODUCIBLE WITH LIMITATIONS** |

---

## 1. Python version

| Layer | Version | Notes |
|---|---|---|
| System Python | 3.11.8 | `C:\Users\Hedaya_city\AppData\Local\Programs\Python\Python311\python.exe` |
| Project-local `.venv` | 3.11.8 | `D:\ALL Projects\E-Commerce\.venv\Scripts\python.exe` |
| Container base image | 3.11.16 | `python:3.11-slim` (latest 3.11 patch built into the image) |

Local `.venv` and the pinned Docker base match the 3.11 line used throughout the project.

## 2. Environment status

- `.venv` exists and is the project-local environment (recreated/verified under the finalization audit; confirmed again here).
- `pip 26.2.1` on `python 3.11.8`; 113 installed distributions (runtime + dev layer + transitive).
- `.venv` is NOT a git-tracked resource: the workspace is not yet an initialized git repository, and `.gitignore` already contains `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`.
- No `.env` file is present and none is required: `Settings` carries full defaults (`app/config.py`) identical to `.env.example`.

## 3. Dependency installation status

- `python -m pip install -r requirements.txt` → **clean resolution, zero upgrades**. Every top-level pin resolved to exactly the declared version.
- Runtime pins satisfied (18/18 verified): numpy 2.4.2, pandas 2.3.3, scikit-learn 1.8.0, lightgbm 4.7.0, shap 0.51.0, pyarrow 22.0.0, joblib 1.5.3, matplotlib 3.10.8, mlflow 3.9.0, fastapi 0.128.1, uvicorn 0.40.0, pydantic 2.12.5, pydantic-settings 2.14.2, python-dotenv 1.2.1, streamlit 1.54.0, sqlalchemy 2.0.46, psycopg2-binary 2.9.13, httpx 0.28.1.
- Dev layer (`requirements-dev.txt` = `-r requirements.txt` + pytest 9.1.1 + xgboost 3.2.0 + optuna 5.0.0) resolved cleanly. This layer is documented as test/training-only and is deliberately NOT in the Docker image.
- Docker image build consumed `requirements.txt` only and produced **identical versions** (17/17 compared inside the container matched the pins, including lightgbm 4.7.0, shap 0.51.0, mlflow 3.9.0, streamlit 1.54.0, psycopg2-binary 2.9.13).

## 4. Required imports

- Automated `scripts/verify_imports.py` imported **24 runtime modules + 3 dev modules + 16 application modules** (all `app.*` routers/services/db/middleware) under `PYTHONPATH=D:\ALL Projects\E-Commerce`.
- Result: **ALL IMPORTS OK** (python 3.11.8, `.venv`).

## 5. Test result

- Full suite: `python -m pytest tests -q`
- **335 passed, 8 skipped (integration-gated), 0 failed** in 348.9 s (~5 min 49 s), 94 warnings.
- The 8 skips are the real-model integration tests gated behind `RUN_DEPLOYMENT_INTEGRATION=1`.
- Integration suite (real Production model; predict/pricing/explain + persistence): `python -m pytest tests/deployment/test_deployment_integration.py -q` with `RUN_DEPLOYMENT_INTEGRATION=1` → **8 passed** in 241.7 s (~4 min).
- Numbers match the recorded baselines exactly (335+8 observed in the finalization audit).

## 6. Production model loading

- Verifiend read-only via the integration suite and live local FastAPI startup:
  - Registry (bundled `artifacts/mlflow_tracking.db`): **v2 `Production`, READY**, source `models:/m-e93cadaf963a4aa1b44b3e501379da92`; v1 `Staging`, READY.
  - `DemandForecaster.load_context` loaded estimator from `mlruns/2/models/m-e93cadaf963a4aa1b44b3e501379da92/artifacts/artifacts/model.joblib`; demand matrix `3,469,274 rows`; design matrix `32,729 rows x 86 cols` (incl. 73 one-hot categories).
  - Startup warm-up completed all components (model, feature state, elasticity state, SHAP runtime) → `/ready` `warmed_up = ok`.
- No training code executed; no registry mutation performed.

## 7. API result

- Local run (`uvicorn app.main:app`) started successfully on 127.0.0.1:8001.
- `/health` 200 `{"status":"ok","app_env":"production","version":"0.6.0"}`.
- `/ready` 200 with all six components `"ok"` (model_loaded, feature_state, elasticity_state, shap_runtime, warmed_up, database).
- OpenAPI (`/openapi.json`) exposes exactly the documented 5 routes: `/health`, `/ready`, `/predict`, `/pricing`, `/explain`.

## 8. Dashboard result

- `streamlit run dashboard/app.py` started headless on 127.0.0.1:8502; `/_stcore/health` → `ok`.
- Same image/command runs inside Docker (`dashboard` service, `--server.port=8501`), health-checks green.

## 9. Docker result

- `docker compose -f deployment\docker-compose.yml config` → **VALID**.
- Images built from the repo-root context, Dockerfile `python:3.11-slim`, `requirements.txt` only; `libgomp1` installed for lightgbm.
- Full stack `up -d`: **db healthy, api healthy, dashboard healthy**. Verified compose ordering (db → api → dashboard `depends_on` with `service_healthy`).
- Container-level package versions matched the pins (see §3).

## 10. Database result

- PostgreSQL 16.15 (`postgres:16-alpine`) reachable:
  - `pg_isready -U inference -d inference` → accepting connections.
  - psycopg2 connect from inside the api container → `PostgreSQL 16.15 on x86_64-pc-linux-musl`.
  - `inference_records` count after operations → **10 rows**, confirming the API wrote inference records to PostgreSQL through the deployment stack.
- Local/SQLite fallback (`artifacts/inference_log.db`) works and is git-ignored; production path in compose is PostgreSQL.

## 11. Health / readiness

- All healthchecks green in compose (`docker compose ps` → all `healthy`).
- `/health` and `/ready` return 200 through the deployed API (host → 127.0.0.1:8000).
- Dashboard `/_stcore/health` → ok through host port 8501.

## 12. predict / pricing / explain endpoints

Real payloads against the real Production model (product `00066f42aeeb9f3007548bb9d3f33c38`, week `2018-05-14`, cost 20):

| Endpoint | Local (8001) | Docker (8000) | Notes |
|---|---|---|---|
| `/predict` | 200 | 200 | `predicted_demand` correct; model v2 Production READY; `inference_logged=true` |
| `/pricing` | 200 | 200 | `reason=NO_ELASTICITY` (product has no reliable elasticity); `inference_logged=true` |
| `/explain` | 200 | 200 | SHAP reconstruction block matches `model.predict` output exactly; `inference_logged=true` |

Error contract also verified: unknown product → 422 `PRODUCT_NOT_FOUND` (matches `app/errors.py`).

## 13. Test split in image

- `.dockerignore` excludes `tests`, `reports`, `notebooks`, `**/*.ipynb`, `data/raw`, `data/splits`, root `mlflow.db`, `.env`, `mlruns/.trash`.
- Live image scan (`docker run --rm ... find`): only `/app/data/processed/demand_matrix_weekly.csv.gz` ships as data; no `tests/`, no `reports/`, no `.ipynb`, no `.env`, no root `mlflow.db`, and no `*test*.csv`/`*test*split*` anywhere in the image.
- **Protected training/test splits and raw analytics are not in the image.**

## 14. MLflow registry unchanged

Fingerprints taken before and after all operations (tests, integration suite, local API startup, Docker build + run):

| Artifact | Baseline | After | Status |
|---|---|---|---|
| `mlruns/` file count | 67 | 67 | unchanged |
| `mlruns/` total bytes | 26,324,776 | 26,324,776 | unchanged |
| `mlruns/` content SHA-256 head | 75b174021e16c9df | 75b174021e16c9df | unchanged |
| root `mlflow.db` SHA-256 head | a7f33e5b03eb953b | a7f33e5b03eb953b | unchanged |
| `artifacts/mlflow_tracking.db` (bundled registry) | SHA-256 head 5672c219e8c708f3, 770,048 bytes | 5672c219e8c708f3, 770,048 bytes | unchanged |

Model versions after all runs: **v2 Production READY / v1 Staging READY** — identical to baseline. No retrain, no new version, no stage change.

## 15. Documentation against real paths/commands

- All documented artifact paths exist: README.md, PROJECT_MAP.md, `deployment/docker-compose.yml`, `deployment/Dockerfile`, `dashboard/app.py`, `app/main.py`, `tests/`, `.env.example`, `data/processed/demand_matrix_weekly.csv.gz`, `artifacts/mlflow_tracking.db`, `src/models/forecasting.py`, `requirements*.txt`.
- Documented commands are real and were executed successfully: `python -m pytest tests`, `uvicorn app.main:app --host 0.0.0.0 --port 8000`, `streamlit run dashboard/app.py --server.port=8501 --server.headless=true`, `docker compose -f deployment\docker-compose.yml up`, `python -m pip install -r requirements[-dev].txt`.

## 16. `.env` not required to be committed

- No `.env` exists in the workspace; nothing depends on one.
- `Settings` resolves env vars → `.env` (optional) → defaults; all defaults are production-safe and match `.env.example`.
- `.gitignore` ignores `.env`, `*.env.local`, `*.pem`, and `deployment/.env`.

## 17. `.env.example` sufficiency

- Verified programmatically: **all 23 `Settings` fields are covered by `.env.example`** (23 keys, 1:1 mapping, no missing, no orphan keys).
- Values in the example equal the recommended defaults and mirror what Docker Compose passes to the containers.

## 18. `.venv` ignored by git

- `.venv/` and `venv/` are listed in `.gitignore` (Python section).
- The workspace is not yet an initialized git repository, so `git` ignore behavior will apply on first `git init`. No repository state was modified during this check.

## 19. Deployment artifacts reproducible

- Docker image produced from pinned inputs: base `python:3.11-slim`, pinned `requirements.txt` (no version drift observed), exact `COPY` list of runtime sources + served matrix + read-only registry.
- Container package versions matched the pins (17/17); `libgomp1` present (lightgbm works, WAPE-identical predictions).
- Compose config is deterministic: static image refs, fixed ports, healthcheck-based `depends_on` ordering, named volume `pgdata`.
- Integration predictions through the container matched predictions through the local installed runtime (same model/feature stack).

---

## Remaining limitations

1. **Patch-level Python drift for the container image**: local `.venv` runs 3.11.8; the Docker base `python:3.11-slim` pulled 3.11.16 at build time. Both are the 3.11 minor line, but patch levels are not pinned in the Dockerfile (`python:3.11-slim` is a moving tag). For bit-exact reproducibility, pin to `python:3.11.8-slim` (or a SHA) in `deployment/Dockerfile`.
2. **Notebook toolchain not pinned**: `nbformat`/`nbconvert` are unpinned and notebooks carry no plotting imports (documented in the finalization review); notebook byte-level reruns may drift. Tests and notebooks are excluded from the image, so this does not affect service reproducibility.
3. **Runtime variance**: full-suite wall time and model warm-up vary with machine/CPU/disk (observed ~5:49 for tests, ~4 min integration, ~3–4 min API warm-up). These are environmental, not functional.
4. **Repository not yet git-initialized**: `.gitignore` rules (incl. `.venv/`, `.env`) are correct and ready, but there is no `.git` yet to enforce them; this is a repo-hygiene step, not a reproducibility defect.

---

## Final verdict

**REPRODUCIBLE WITH LIMITATIONS**

All 19 checks pass: pinned runtime installs cleanly, all imports resolve, 335/335 default tests + 8/8 real-model integration tests pass, the Production model loads read-only with an unchanged registry (67 mlruns files, identical SHA-256) and unchanged v2 Production / v1 Staging versions, FastAPI and Streamlit start in both local and Docker environments, PostgreSQL connectivity and inference persistence work through the compose stack, predict/pricing/explain return correct 200s with the frozen model, the image contains no protected/test data, `.env` is optional and `.env.example` fully documents configuration, and deployment artifacts rebuild reproducibly.

The only limitations (documented above) are advisory: Docker base-image patch pinning (`3.11` vs exact patch), unpinned notebook toolchain, and expected machine-dependent timing.