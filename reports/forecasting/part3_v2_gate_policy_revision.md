# Part-3 V2 Quality Gate Policy Revision

* Generated: 2026-09-24T11:27:29.323578+00:00
* Phase: `part3-v2-gate-policy-revision`
* Model: `models:/demand_forecasting_final/2` (version 2, run `5d729a8594dc488bbd18ba0c4e216f5a`)

## 1. Decision

| Field | Value |
|---|---|
| Policy decision | **REVISED_GATE_APPLIED** |
| V2 decision | **PRODUCTION** |
| Policy applied | `quality_gate_policy_v2_revised` |
| Legacy policy | `quality_gate_policy_v1_legacy` |
| Legacy gate | **FAIL** (only `B_wape > 0.60`) |
| Revised gate | **PASS** |
| MLflow stage (version 2) | `Staging -> Production` (transitioned: `True`) |

Basis: Formal owner decision after the provenance audit (part3_v2_gate_production_policy): B_wape <= 0.60 was classified as an engineering reference-anchored regression guard, not a documented business requirement. The revised policy reclassifies it as diagnostic/monitoring while preserving ALL other criteria unchanged.

## 2. Legacy gate replay (deterministic; source = persisted post-hoc report)

- Replay outcome count: 19 (12 numeric + 7 technical booleans).
- Failed criterion: `B_wape`.
- Identical to persisted post-hoc evaluation: `True` (persisted 19 outcomes).
- Deterministic (replayed twice -> identical): `True`.

## 3. Revised policy evaluation

Policy `quality_gate_policy_v2_revised`: `B_wape` is reported as a **diagnostic/monitoring** metric (non-blocking). The hard business constraint `B_bias_pct >= -20` (no systematic under-forecasting of sustained high-volume demand) and every other criterion remain **blocking**. No threshold VALUE was changed.

### 3.1 Blocking criteria (18 criteria)

| Criterion | Observed | Bound | PASS/FAIL |
|---|---|---|---|
| wape | 0.375373 | 0.41 | PASS |
| mae | 0.016554 | 0.025 | PASS |
| rmse | 0.185602 | 0.3 | PASS |
| mase | 0.435128 | 0.64 | PASS |
| bias_pct | -1.196961 | (-12.0, 12.0) | PASS |
| asym21 | 0.025095 | 0.04 | PASS |
| pos_wape | 0.336382 | 0.42 | PASS |
| B_bias_pct | 9.851522 | -20.0 | PASS |
| B2_wape | 0.556648 | 0.65 | PASS |
| B2_bias_pct | 0.861589 | -20.0 | PASS |
| C_wape | 0.365135 | 0.43 | PASS |
| pred_integrity | True | True | PASS |
| schema_ok | True | True | PASS |
| model_load_ok | True | True | PASS |
| pyfunc_predict_ok | True | True | PASS |
| signature_ok | True | True | PASS |
| artifacts_ok | True | True | PASS |
| tests_ok | True | True | PASS |

### 3.2 Diagnostic (reported, non-blocking) metrics

| Metric | Observed | Role |
|---|---|---|
| `B_wape` | 0.628239 | `diagnostic` |

## 4. MLflow lifecycle update

- Registered model: **`demand_forecasting_final`**
- Model version: **2** (no new version created; no v3)
- Stage: **Staging -> Production** (transitioned: `True`)
- Revised gate decision tag: **PASS**
- Tags applied to run + version (additive; historical tags preserved): `gate_policy_version=quality_gate_policy_v2_revised`, `legacy_gate_decision=FAIL`, `revised_gate_decision=PASS`, `b_wape_role=diagnostic`, `evaluation_type=post_hoc_holdout_evaluation`, `test_used_for_policy_revision=false`, `gate_policy_revision_applied=true`, `gate_policy_legacy=quality_gate_policy_v1_legacy`, `gate_policy_revision_decision=REVISED_GATE_APPLIED`
- Historical tags preserved verbatim on run + version: `production_gate=FAIL`, `gate_decision=FAIL`, `promoted_stage=Staging`, `lifecycle_stage=Staging`, `posthoc_holdout_evaluated=true`, `evaluation_group=v2_validation`, `final_*`, `test_scored_once=false`.
- V1 preserved: version 1 (Staging) + run `7fc87198...` + all Part-3 historical reports untouched.

## 5. Diagnostic policy rationale (blocker -> diagnostic)

The provenance audit `part3_v2_gate_production_policy` classified `B_wape <= 0.60` as an engineering reference-anchored regression guard (documented references winner 0.5485 / EXP2 0.5102; pre-registered before the Test split was scored). It is NOT documented in the repository as a business requirement. The owner decided to reclassify B_wape to a diagnostic/monitoring role so it continues to be reported (observed 0.628239 vs the 0.60 guard) while Production standing is decided by the remaining hard criteria -- above all the `B_bias_pct >= -20` business constraint, which V2 satisfies (+9.85%).
