# Pricing Manager-Ready UX — Docker Packaging Fix & Verification

**Verdict: `FIXED_AND_VERIFIED`.** The audit's `DOCKER_RUNTIME_BUG` is fixed: the
`deployment-dashboard` image now ships exactly the persisted Part-4 elasticity
artifact, the Docker runtime reproduces the local tier classification
bit-for-bit, and a missing artifact is now observable in the UI instead of
degrading silently to "0 products".

Scope respected: no change to the model, Part-3/Part-4 logic, the artifact
contents, API schemas, Docker architecture/Compose topology, data, or business
thresholds. The API still builds its elasticity state at startup via
`elasticity_mod.estimate` (`app/services/state.py`) — that was not replaced.

---

## 1. Root cause confirmed

The audit finding was reproduced and its cause isolated:

| Layer | Finding |
|---|---|
| `deployment/Dockerfile` | Copied `app`, `dashboard`, `src`, `healthcheck.py`, the weekly matrix, `artifacts/mlflow_tracking.db` and `mlruns` — but **no** `reports/` artifact. |
| `.dockerignore` | Contained a bare `reports` entry, so the whole directory was excluded from the build context. |
| `deployment/docker-compose.yml` | The `dashboard` service has **no** volume mount, so nothing supplied the artifact at runtime. |
| Image contents (before) | `/app/reports` did not exist; `/app/data/processed/demand_matrix_weekly.csv.gz` did. |
| Consequence | `_elasticity_details()` hit `except OSError -> {}`, so `pricing_tier_counts()` returned `(0, 0, 32729)` for every week while valid-price counts stayed correct — i.e. the matrix was fine, only the elasticity set was empty. |

## 2. Changes made

### 2.1 `.dockerignore` — exclude reports by content, re-include one file

```diff
 tests
-reports
+
+# Reports stay excluded from the image, except the single persisted Part-4
+# pricing artifact the dashboard reads at runtime (dashboard/product_catalog.py).
+# `reports/**` (not bare `reports`) is required: re-including a directory
+# un-excludes its whole subtree, which would ship every other report.
+reports/**
+!reports/forecasting/part4_pricing_engine.json
+
 notebooks
```

The pattern shape was **determined empirically, not assumed**, with throwaway
build contexts (`reports/forecasting/part4_pricing_engine.json`,
`reports/forecasting/part3_*.json`, `reports/forecasting/some_report.md`,
`reports/finalization/other_report.json`, `reports/top_level.md`,
`data/splits/test.csv`, `keep.txt`):

| Variant | Files shipped | Verdict |
|---|---|---|
| A: `reports` + `!reports/` + `!reports/forecasting/` | every report | **rejected** — re-including a directory un-excludes its subtree |
| B: `reports/**` + `!reports/forecasting/` + `!reports/…/part4…` | all files in `forecasting/` | **rejected** — siblings leak with the directory re-included |
| C: `reports/**` + `!reports/forecasting/part4_pricing_engine.json` | only `part4_pricing_engine.json` | **accepted** |

This is locked in by two new regression tests (see §4).

### 2.2 `deployment/Dockerfile` — bake in the single artifact

```diff
 COPY data/processed/demand_matrix_weekly.csv.gz ./data/processed/demand_matrix_weekly.csv.gz
 COPY artifacts/mlflow_tracking.db ./artifacts/mlflow_tracking.db
 COPY mlruns ./mlruns
+
+# Persisted Part-4 pricing state consumed read-only by the dashboard
+# (dashboard/product_catalog.py). Only this one artifact is shipped; every
+# other report stays excluded via .dockerignore.
+COPY reports/forecasting/part4_pricing_engine.json ./reports/forecasting/part4_pricing_engine.json
```

`docker-compose.yml` was intentionally **not** modified: a volume mount would
have masked the packaging bug instead of fixing it.

### 2.3 Missing artifact is now observable (smallest useful change)

Previously a missing artifact was indistinguishable from "no product has
elasticity". The dashboard now has an explicit state.

`dashboard/product_catalog.py`
- new `READINESS_AVAILABLE` / `READINESS_MISSING` / `READINESS_UNREADABLE` and
  `STATUS_READINESS_UNAVAILABLE`;
- new cached `_readiness_report() -> (state, report)` and public
  `pricing_readiness_state()`; an absent file reports `missing`, an unparsable
  file `unreadable`, and neither is ever presented as an empty-but-valid
  population;
- `_elasticity_details()` now consumes that state (same single artifact read);
- `product_pricing_status()` short-circuits to `STATUS_READINESS_UNAVAILABLE`,
  still reporting the **real** current price from the matrix.

`dashboard/pricing_presenter.py`
- new `render_pricing_opportunities(week)` — the single opportunities renderer
  used by both `app.py` and the AppTest probe (so the probe cannot drift). When
  readiness is unavailable it renders `st.error("Pricing readiness data is
  unavailable")` plus an explanation and the concrete state, and **no
  headcounts**; otherwise it renders the unchanged `(tier1, tier2, tier3)`
  summary.
- `render_pricing_readiness()` renders the same unavailable block for the
  selected product.

`dashboard/app.py` — `render_pricing_section()` now delegates to
`render_pricing_opportunities()`; the rendered content for a healthy deployment
is identical to before.

Behaviour deliberately unchanged: the catalog stays fully searchable, the
backend remains the final gate, no product is hidden, and no elasticity is ever
recomputed in the dashboard.

## 3. Runtime verification (Docker)

Images rebuilt with `--no-cache` (`docker compose -f deployment/docker-compose.yml
build --no-cache`), stack started; `db`, `api` and `dashboard` all report
**healthy**.

### 3.1 Artifact present, readable and byte-identical

| Check | Result |
|---|---|
| `/app/reports/forecasting/part4_pricing_engine.json` | present, 14,565,915 bytes |
| SHA256 host vs container | `2c696256dbd5f46dd875ff4e3edc78c6ec513d13319e80e490b540ddd1c03bb7` — identical |
| `pricing_readiness_state()` | `available` |
| Files under `/app/reports` (both `api` and `dashboard` images) | exactly one |
| `/app/data` | only `processed` — no `data/splits`, no `test.csv`, no `.trash` |
| Streamlit `/_stcore/health`, `/` | `ok`, HTTP 200; no errors in container logs |

### 3.2 Docker runtime now matches the local numbers exactly

| Metric | Local (expected) | Docker runtime | Match |
|---|---|---|---|
| Usable elasticity products | 17,012 | 17,012 | yes |
| Total products | 32,729 | 32,729 | yes |
| `2018-05-14` valid prices | 1,385 | 1,385 | yes |
| `2018-05-14` tiers | (468, 16544, 15717) | (468, 16544, 15717) | yes |
| `2017-12-18` valid prices | 782 | 782 | yes |
| `2017-12-18` tiers | (683, 16329, 15717) | (683, 16329, 15717) | yes |
| `tier1 + tier2 + tier3` | 32,729 both weeks | 32,729 both weeks | yes |

### 3.3 API contract unchanged and still correct

`/health` → `ok` (production); `/ready` → `ready` with every component `ok`
(`model_loaded`, `feature_state`, `elasticity_state`, `shap_runtime`,
`warmed_up`, `database`).

Tier-1 product `005030ef108f58b46b78116f754d8d38`, week `2018-05-14`:

| Call | Result |
|---|---|
| `POST /pricing` cost `12.5` | `reason=optimized`, `objective=profit`, `profit_computable=true`, `optimal_profit=4.32`, `recommended_price=16.79`, `current_price=13.99`, `elasticity=-0.967` (`category_pooled`), `inference_logged=true` |
| `POST /pricing` no cost | `reason=PROFIT_NOT_COMPUTABLE (no cost supplied)`, `objective=revenue`, `optimal_profit=null` — optional cost preserved |
| `POST /predict` | `predicted_demand=1.2028` |
| `POST /explain` | full SHAP block, additivity residual `0.0`, 93 features, business-labelled only |

### 3.4 End-to-end UI against the live container API

30/30 AppTest checks passed driving the real `dashboard/app.py` against the
running Docker API (`http://localhost:8000`):

- Reference week `2018-05-14` → `468` / `16544` / `15717`; week `2017-12-18` →
  `683` / `16329` / `15717`; counts follow the selected week; no error elements.
- A Tier-1 product is found by searching a real id prefix, shows the raw product
  id, `Ready for pricing`, `Current price $13.99`, `Elasticity -0.97`,
  `Elasticity source Category`.
- Submitting with cost `12.5` renders the live recommendation
  `Recommended price: 16.79`, `Reason: optimized`, `objective=profit`,
  `profit_computable=True`, `inference_logged=True`.
- With the artifact absent, the UI shows "Pricing readiness data is
  unavailable", reports the concrete state (`missing`), shows **no** headcounts,
  keeps the catalog searchable, and the selected product reports the same
  unavailability while still showing its real current price.

Note on the harness: `streamlit.testing.v1.AppTest` deadlocks inside this
container for any script that imports pandas/numpy (the script thread never
signals completion; a trivial app runs in 1.3 s). This is a test-harness/thread
environment issue, not an app defect — the containerized app itself was
verified directly (§3.1, §3.2) and the end-to-end UI was driven from the host
against the live container API (§3.3, §3.4).

## 4. Tests

| Suite | Before | After |
|---|---|---|
| `pytest tests/dashboard -q` | 67 passed | **72 passed** |
| `pytest tests -q` | 400 passed, 1 failed, 9 skipped | **407 passed, 1 failed, 9 skipped** |
| `pytest tests/deployment -q` | 45 passed, 8 skipped | **47 passed, 8 skipped** |

The single failure is **pre-existing and unrelated**:
`tests/test_posthoc_holdout.py::test_marker_guard_prevents_second_execution`
(absent holdout marker). It was not "fixed" and no marker was regenerated.

New tests added:

- `tests/dashboard/test_product_catalog.py`
  - `test_readiness_state_is_available_when_artifact_is_present` — pins the
    17,012 population;
  - `test_missing_artifact_is_observable_and_never_means_zero_elasticity` —
    missing artifact → `READINESS_MISSING`, empty details, status
    `READINESS_UNAVAILABLE` (≠ `NO_ELASTICITY`), real price preserved;
  - `test_unreadable_artifact_is_observable` — unparsable artifact →
    `READINESS_UNREADABLE`;
  - `test_catalog_search_stays_usable_when_artifact_is_missing` — all 32,729
    products still selectable via search;
  - new `readiness_artifact` fixture repoints the artifact path and clears the
    catalog caches.
- `tests/dashboard/test_product_selector_app.py`
  - `test_pricing_missing_readiness_artifact_shows_unavailable_not_zero_counts` —
    the unavailable path end to end through the probe.
- `tests/dashboard/_probe_app.py` — the probe now calls the app's own
  `render_pricing_opportunities()` and supports `probe_missing_readiness`.
- `tests/deployment/test_deployment_safety.py`
  - `test_dockerignore_ships_only_the_part4_pricing_artifact` — rejects the bare
    `reports` entry and any directory-level re-include;
  - `test_dockerfile_copies_the_part4_pricing_artifact` — pins the `COPY`.

## 5. Files changed (repository is not a git working tree)

Verified by modification-time audit of the project tree:

| File | Change |
|---|---|
| `.dockerignore` | `reports` → `reports/**` + single-file re-include |
| `deployment/Dockerfile` | `COPY` of the one artifact |
| `dashboard/product_catalog.py` | readiness state, `STATUS_READINESS_UNAVAILABLE` |
| `dashboard/pricing_presenter.py` | `render_pricing_opportunities()`, unavailable copy |
| `dashboard/app.py` | delegate to the shared renderer |
| `tests/dashboard/_probe_app.py` | shared renderer + missing-artifact simulation |
| `tests/dashboard/test_product_catalog.py` | 4 tests + fixture |
| `tests/dashboard/test_product_selector_app.py` | 1 test |
| `tests/deployment/test_deployment_safety.py` | 2 packaging regression tests |
| `reports/finalization/pricing_manager_ready_docker_fix.md` | this report |

Explicitly **unchanged**: `reports/forecasting/part4_pricing_engine.json` (SHA256
identical), `deployment/docker-compose.yml`, `app/**`, `src/**`, `data/**`,
`artifacts/**`, `mlruns/**`, `PROJECT_MAP.md`, and the protected held-out split
(which is neither present in the image nor referenced by served code).
