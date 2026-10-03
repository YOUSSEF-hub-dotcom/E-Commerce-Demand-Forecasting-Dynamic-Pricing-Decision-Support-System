# Reports index

These files are the **evidence and provenance record** of the project. They are dated
artefacts of work that actually happened. They are **not** a description of the current
pipeline, and several of them document one-off experiments that are *not* stages of the
official workflow.

Nothing in this directory has been rewritten to make the history look tidier. The metrics,
decisions and FAIL results below are the real recorded values.

> **Point-in-time snapshots.** Every number in these files — test counts, pass/fail
> tallies, durations — was true of the cycle that produced that report and has been left
> exactly as recorded. It is *not* the current state. The authoritative current test result
> and repository layout are in the top-level `README.md`, not here.

## Official workflow (the current pipeline)

```
Data → Train/Validation → Cross Validation → Model Selection → Optuna Tuning
     → Final Test Evaluation → Quality Gate → Production
     → Dynamic Pricing → SHAP Explainability → API → Dashboard → Docker
```

Reports that evidence a stage of that workflow:

| Report | Stage evidenced |
|---|---|
| `forecasting/part1_3_data_feature_leakage_audit.{json,md}` | Data / features — leakage probes and split partition proof |
| `forecasting/part3_pre_final_test_cleanup.md` | Features — schema 86→93, OHE train-only |
| `forecasting/part3_automation_audit.{json,md}` | Lifecycle — gate auto-invocation, promotion blocking |
| `forecasting/part3_final_lifecycle_report.md`, `part3_v2_lifecycle_report.md` | Quality Gate / Production — v1 and v2 lifecycle records |
| `forecasting/part4_data_feasibility.{json,md}` | Dynamic Pricing — cost-source feasibility |
| `forecasting/part4_pricing_engine.json`, `part4_pricing_engine_report.md` | Dynamic Pricing — elasticity, simulation, optimizer, guardrails |
| `forecasting/part5_shap_explainability.{json,md}` | SHAP Explainability |
| `deployment/part6_deployment_architecture_audit.md`, `part6_deployment_implementation_report.md`, `inference_request_schema_audit.md` | API / Dashboard / Docker |
| `experiments/06_forecasting_controlled_experiments_report.md` | Model selection + Optuna tuning experiments |
| `finalization/finalization_report.md`, `reproducibility_report.md`, `documentation_review.md` | Cross-project verification |

## Historical / provenance — completed one-off experiments

These document experiments that were executed and concluded. **None is a workflow stage.**
Several produced a recorded FAIL, and that FAIL is deliberately preserved.

| Report / artefact | What actually happened |
|---|---|
| `forecasting/part3_v2_posthoc_holdout_evaluation.{json,md}` | One-off Final Test Evaluation of frozen v2 (scored once). Global WAPE 0.3754, B-cohort 0.6282 → **FAIL** under `quality_gate_policy_v1_legacy`. Still read by `test_b_cohort_remediation`, `test_b_cohort_asymmetric_objective` and `test_gate_policy_revision` as a provenance fixture. |
| `forecasting/_historical_audit/posthoc_holdout.py` | The module that produced the above. Archived out of `src/`; not importable from the package and not collected by pytest. Kept so the report above has a reproducible source. |
| `forecasting/_historical_audit/test_posthoc_holdout.py` | Its former test suite, archived beside it. The one test that hard-coded an absolute temp path outside the repository was the reason this suite was removed from `tests/`. |
| `forecasting/part3_final_holdout_blocker.{json,md}` | `status: BLOCKER`. Records that the demand matrix physically ends at the week of `2018-09-03/2018-09-09` (`test_is_last_16_weeks_of_dataset: true`), so no further validation period can be built without reusing development data. The 13 orders after `2018-09-09` are cancelled only, with zero matching order items. Read by `test_final_holdout_integrity.py`. |
| `forecasting/part3_b_cohort_error_analysis.{json,md}` | B-cohort root-cause diagnosis, **Train+Val only**. Reproduced validation B WAPE exactly at 0.5101460932701322 and located the error concentration in spike weeks (B WAPE 0.612, bias −58.86%) versus non-spike weeks (0.474, bias −7.25%). |
| `forecasting/part3_b_cohort_remediation_experiments.{json,md}` | Two pre-registered validation-only arms (`wape_weights`, `spike_features` 93→97 features) in experiment `part3_b_cohort_remediation_exp` → **NO_VALID_IMPROVEMENT**, `selected_arm: null`. Arm 3 (spike features) passed six of seven criteria and failed only `C_wape`: C WAPE rose 0.37954 → 0.39529, **+0.0158** against a +0.01 tolerance. |
| `forecasting/part3_b_cohort_asymmetric_objective.{json,md}` | Pre-registered validation-only trial retargeting the 2:1 asymmetric loss with LightGBM's built-in quantile (pinball) objective, `alpha = 2/3` → **NO_VALID_IMPROVEMENT**. It did repair the under-forecast (B bias −18.23% → +1.50%) and improved global WAPE (0.39637 → 0.36651), but failed two pre-registered criteria: `C2` B WAPE *worsened* 0.51015 → 0.55642, and `C5` B2 WAPE 0.55569 → 0.59719 broke its +0.03 tolerance. Experiment `part3_b_cohort_asymmetric_exp`. |
| `forecasting/part3_v2_gate_production_policy.{json,md}` | Audit finding: `B_wape <= 0.60` is an engineering heuristic, not a documented business requirement → decision `B_FORMAL_GATE_REVISION_REQUIRED`. |
| `forecasting/part3_v2_gate_policy_revision.{json,md}` | The owner-authorized revision that reclassified `B_wape` to diagnostic and promoted v2 Staging → Production. **This is why the current gate is `quality_gate_policy_v2_revised`.** |
| `forecasting/part3_v2_audit.json`, `part3_v2_result.json`, `part3_final_audit.json`, `part3_final_result.json` | Machine-readable snapshots of the v2 and final lifecycle states. |
| `forecasting/final_execution_report.md` | Consolidated execution record for Part 3. |
| `experiments/07_b_cohort_rootcause.json`, `07b_b_cohort_deep.json`, `07c_v2_experiments_val.json` | Raw outputs of the validation-only exploration scripts in `src/notebooks/`. |

## UX / presentation audits (Part 6+)

`finalization/dashboard_product_selector_ux.md`,
`finalization/pricing_manager_ready_docker_fix.md`,
`finalization/pricing_manager_ready_runtime_audit.md`,
`finalization/pricing_manager_ready_ux.md`,
`finalization/pricing_no_elasticity_ux.md`,
`finalization/pricing_selector_audit.md`,
`finalization/pricing_selector_demo_ready_ux.md`,
`finalization/pricing_selector_demo_usability.md`,
`finalization/elasticity_diagnostic_audit.md`

These document dashboard/pricing-readiness UI work and image packaging verification. They
are current-system documentation, not experiment provenance.

## Naming note

Some filenames contain older terminology (`holdout`, `posthoc`). Those names were fixed at
the time each artefact was produced and are referenced by active tests and by the MLflow run
record, so they are kept as-is: renaming them would break provenance and could not be done
without altering the historical record. The *contents* are accurate and unaltered; only the
framing in the current documentation was changed to stop presenting them as workflow stages.