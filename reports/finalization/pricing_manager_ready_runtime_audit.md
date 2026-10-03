# Pricing Manager-Ready UX — Runtime / Data-Classification Audit

**Audit only. No code, model, pricing logic, artifact, API, Docker or data was
modified.**

## Verdict

**DOCKER_RUNTIME_BUG** (root cause: the persisted Part-4 elasticity artifact is
not shipped into the `deployment-dashboard` image, so the dashboard's
elasticity set is empty at runtime). The classification **logic is correct**; the
**deployment packaging** is not. Secondary contributor: the dashboard degrades
silently (`except OSError -> {}`), which hides the missing artifact.

---

## 1. Expected tier definitions (from the implementation)

`dashboard/product_catalog.py`:

```python
def pricing_tier_counts(week):
    ready = load_pricing_ready_sources()          # week-independent
    priced = load_price_available_for_week(week)  # week-dependent
    return len(ready & priced), len(ready - priced), total_products() - len(ready)
```

- **Tier 1** = `elasticity ∩ valid_price(week)`
- **Tier 2** = `elasticity - valid_price(week)`
- **Tier 3** = `all_products - elasticity`

- **Usable elasticity** = persisted Part-4 `elasticity.effective` rows with
  `elasticity_source in {"own","category_pooled"}` and finite `elasticity`.
  This is **product-level and week-independent**.
- **Valid current price** = the dense weekly demand matrix has a row for that
  exact week with `avg_price > 0` (mirrors `AppState.price_at`).

Identity check: `Tier 1 + Tier 2 + Tier 3 = |elasticity| + (total - |elasticity|) = total`.

## 2. Part 4 elasticity population (artifact, read-only)

`reports/forecasting/part4_pricing_engine.json` -> `elasticity.effective`:

| source | rows |
|---|---|
| `category_pooled` | 15,474 |
| `own` | 1,538 |
| `none` | 2,087 |
| **total effective rows** | **19,099** |
| **usable (own + category_pooled, finite)** | **17,012** |

`dashboard.load_pricing_ready_sources()` returns exactly **17,012** when the
artifact is present — confirming the set is derived correctly and is
week-independent.

## 3. Exact counts — `2018-05-14` (local, artifact present)

| quantity | value |
|---|---|
| total products | 32,729 |
| usable elasticity | 17,012 |
| valid price in week | 1,385 |
| elasticity ∩ valid_price (Tier 1) | **468** |
| elasticity − valid_price (Tier 2) | **16,544** |
| all − elasticity (Tier 3) | **15,717** |
| Tier1 + Tier2 + Tier3 | **32,729** |

## 4. Exact counts — `2017-12-18` (local, artifact present)

| quantity | value |
|---|---|
| total products | 32,729 |
| usable elasticity | 17,012 |
| valid price in week | 782 |
| elasticity ∩ valid_price (Tier 1) | **683** |
| elasticity − valid_price (Tier 2) | **16,329** |
| all − elasticity (Tier 3) | **15,717** |
| Tier1 + Tier2 + Tier3 | **32,729** |

The identities in §1 hold exactly for both weeks. Note Tier 3 is identical
(15,717) across weeks, as expected (it does not depend on the week or on prices).

## 5. Local vs Docker comparison

Same image `deployment-dashboard:latest`, run as
`docker run --rm deployment-dashboard:latest python -c "..."`.

| | Local (host) | Docker (image) |
|---|---|---|
| `PRICING_READY_REPORT` exists | yes | **no** |
| `|usable elasticity|` | 17,012 | **0** |
| valid price `2018-05-14` | 1,385 | 1,385 |
| valid price `2017-12-18` | 782 | 782 |
| **`2018-05-14` tiers (t1,t2,t3)** | **(468, 16544, 15717)** | **(0, 0, 32729)** |
| **`2017-12-18` tiers (t1,t2,t3)** | **(683, 16329, 15717)** | **(0, 0, 32729)** |

The **price side is identical** local vs Docker (1,385 / 782); only the
**elasticity side collapses to 0**. The screenshotted Docker result
`0 / 0 / 32,729` is reproduced exactly, and it is produced for **every** week
(not just `2017-12-18`), because `ready` is empty.

## 6. Runtime / artifact paths

Inside the image (`ls /app`, `find /app -name part4_pricing_engine.json`):

- `/app` contains `app`, `dashboard`, `src`, `data`, `deployment`, `artifacts`,
  `mlruns`, `requirements.txt`.
- **`/app/reports` does not exist**; the artifact is **not found** anywhere.
- `/app/data/processed/demand_matrix_weekly.csv.gz` **exists** (13.7 MB).
- `docker inspect`/compose: the `dashboard` service has **no volumes**, so the
  host `reports/` is never mounted into the container.

Evidence — Dockerfile (`deployment/Dockerfile`) copies only:
`app`, `dashboard`, `src`, `deployment/healthcheck.py`,
`data/processed/demand_matrix_weekly.csv.gz`, `artifacts/mlflow_tracking.db`,
`mlruns`. It **never copies `reports/`**.

Evidence — `.dockerignore` at the repository root contains a bare line
`reports`, which excludes the entire directory from the build context.

Because `dashboard/product_catalog.py` resolves the artifact as
`PROJECT_ROOT / "reports" / "forecasting" / "part4_pricing_engine.json"` =
`/app/reports/forecasting/part4_pricing_engine.json` (which is absent),
`_elasticity_details()` hits `except (OSError, ValueError): return {}` and returns
an empty map. `load_elasticity_details()` and `load_pricing_ready_sources()` are
therefore empty inside the container.

Contrast: the **API** (`app/services/state.py::_load_elasticity_state`) does
**not** read the report; it re-estimates elasticity at startup from the served
matrix (`src.optimization.elasticity.estimate(train, val)`), and the matrix **is**
shipped. So the API still returns correct pricing recommendations in Docker; only
the dashboard's elasticity-based **classification/labels** are inert.

## 7. Cache / type investigation

- **Week type/value**: the Streamlit `week_selector` returns `str` week labels
  (e.g. `"2017-12-18"`); `load_week_prices` compares with
  `frame[WEEK].astype(str) == week`. No datetime/date object crosses the
  boundary. Proof: the same week yields identical valid-price counts locally and
  in Docker (782), and local Tier 1 = 683 (non-zero) — so the week value and
  filtering are correct.
- **Week applied to elasticity?** No. `load_pricing_ready_sources()` takes no
  `week`; elasticity is week-independent by construction. The only week-dependent
  term is `load_price_available_for_week(week)`.
- **Accidental intersection removing all elasticity?** No. The intersection is
  used only for Tier 1; Tier 2 is `ready - priced`. With `ready = ∅`, both are 0
  and Tier 3 = total — which is exactly the observed output.
- **Stale cached empty result?** `load_pricing_ready_sources` is `@st.cache_data`;
  in Docker it caches the empty frozenset computed from the **missing file**. The
  cache is not the origin of the emptiness (the source is absent); it would only
  delay recovery if the file appeared later without a restart.
- **ISO parsing differences / normalization**: none applicable — the failure is a
  missing file, not a parse of a present file.
- **Artifact path differences inside Docker**: **yes** — this is the defect.

## 8. Root cause

`deployment/Dockerfile` does not `COPY reports/`, and the root `.dockerignore`
excludes `reports`, so
`/app/reports/forecasting/part4_pricing_engine.json` is absent in the dashboard
container. The dashboard's `_elasticity_details()` silently degrades to `{}`, so:

```
usable elasticity = 0
Tier 1 = |∅ ∩ priced| = 0
Tier 2 = |∅ − priced| = 0
Tier 3 = 32729 − 0 = 32729
```

for **every** week. This matches the screenshot (`0 / 0 / 32,729`) exactly.

## 9. Is the current implementation correct?

- **Classification logic: correct.** With the artifact present it produces the
  mathematically required tiers (verified: `(468,16544,15717)` and
  `(683,16329,15717)`, each summing to 32,729).
- **Deployment wiring: incorrect.** The dashboard depends on a persisted
  artifact that is not part of the runtime image, so the feature is inert in
  Docker while working locally.
- This is **not a regression introduced solely by the new UX**: the earlier
  selector prioritisation (prior task) consumed the same
  `load_pricing_ready_sources()` and was equally inert in Docker; the new UX
  merely made the emptiness **visible** via explicit `0 / 0 / 32,729` counts.
- The `except (OSError, ValueError): return {}` fallback is defensible
  (graceful degradation) but **masked** the packaging gap — no log/warning is
  emitted when the artifact is missing.

## 10. Minimal recommended fix (NOT implemented)

**Option A (minimal, recommended) — ship the single persisted artifact into the
image:**
1. In `deployment/Dockerfile`, add:
   `COPY reports/forecasting/part4_pricing_engine.json ./reports/forecasting/part4_pricing_engine.json`
2. In `.dockerignore`, after the `reports` line add a re-include (order matters):
   `!reports/forecasting/part4_pricing_engine.json`
   (optionally also `!reports/` before it if the negation requires the parent).
This keeps all other reports out of the build context, preserving the current
"never ship protected analytics" policy while shipping only the model/pricing
artifact the dashboard consumes read-only.

**Option B (larger) — align the dashboard's elasticity source with the API's
runtime source** (e.g. derive the dashboard set from the served matrix or from an
API endpoint) so both services share a single shipped source. This duplicates or
re-exposes logic and is a broader change than the minimal fix.

**Follow-up (recommended, independent of A/B):** make the missing-artifact case
observable (log a warning / surface status) instead of returning `{}` silently,
so this class of environment mismatch cannot hide again. Per the audit scope,
this is a recommendation only and was **not** implemented.

---

### Reproduction commands used (read-only)

```
# local
.\.venv\Scripts\python.exe -c "from dashboard import product_catalog as pc; print(pc.pricing_tier_counts('2017-12-18'))"
# -> (683, 16329, 15717)

# docker image contents
docker run --rm --entrypoint sh deployment-dashboard:latest -c "ls /app/reports; find /app -name part4_pricing_engine.json"

# docker classification
docker run --rm deployment-dashboard:latest python -c "from dashboard import product_catalog as pc; print(pc.load_pricing_ready_sources().__len__(), pc.pricing_tier_counts('2017-12-18'))"
# -> 0 (0, 0, 32729)
```
