# Part-3 Controlled Validation-only Experiment: Asymmetric 2:1 Training Objective

* Generated: 2026-09-23T22:12:48.299092+00:00

## Hypothesis

> Training directly with the 2:1 asymmetric absolute business loss (as a quantile objective at alpha=2/3) shifts predictions toward the upper conditional quantile, reducing the diagnosed underforecasting on high-demand cohorts without materially degrading global/cohort accuracy.

## Control definition

> Exact frozen V2 replica: same Train/Val, same 93-col schema, same log1p target, same WINNER_PARAMS (n_estimators=600, seed=1, no early stopping), same 1+log1p weights, same validation.

- Control reproduced: True (global WAPE 0.3964, B WAPE 0.5101, B bias -18.2274%).

## Treatment definition

> Identical pipeline except training objective -> LightGBM built-in quantile (pinball), alpha=2/3; target/prediction transforms and matrix weights unchanged.

## Exact objective used

- LightGBM built-in `objective='quantile'`, `alpha=0.6667` (= tau = under/(under+over) = 2/(2+1) = 2/3).

### Why this objective represents the 2:1 business asymmetry

> project asymmetric_loss(2,1) is mean(w*|error|) absolute with w=2 under / 1 over; pinball(tau) weights under by tau and over by 1-tau; with tau=2/3: 3*pinball = asymmetric_loss. Exact match (L1 family), validated against installed LightGBM 4.7.0 built-in objective.

- Custom objective (Option B): not used. Option B skipped: pinball 2nd derivative is a Dirac at 0; the built-in quantile objective handles the Hessian robustly and the absolute-loss family exactly matches the evaluation loss; no experimental reason to implement both.
- expm1 is strictly increasing, so alpha-quantile of log1p(demand) maps to alpha-quantile of demand; prediction contract preserved.

## Pre-registered criteria (recorded before treatment results)

| id | criterion | threshold (relative to control) |
|---|---|---|
| C1_global_wape_abs_tol | C1 global wape abs tol | 0.01 |
| C2_B_wape_rel_to_control | C2 B wape rel to control | 0.95 |
| C3_B_bias_abs_reduction | C3 B bias abs reduction | 0.75 |
| C4_asym21_rel_tol | C4 asym21 rel tol | 1.1 |
| C5_B2_wape_abs_tol | C5 B2 wape abs tol | 0.03 |
| C6_C_wape_abs_tol | C6 C wape abs tol | 0.01 |
| C7_MAE_rel_tol | C7 MAE rel tol | 1.05 |
| C7_RMSE_rel_tol | C7 RMSE rel tol | 1.1 |
| C7_MASE_rel_tol | C7 MASE rel tol | 1.1 |
| C7_POS_WAPE_rel_tol | C7 POS WAPE rel tol | 1.1 |
| C8 | prediction integrity | finite, non-negative, length-aligned |

## Control metrics

- wape: 0.3964
- mae: 0.0233
- rmse: 0.2592
- mase: 0.6122
- bias_pct: -6.1492
- asym21: 0.0367
- pos_wape: 0.3655
- B_wape: 0.5101
- B_bias_pct: -18.2274
- B_units: 330.4624
- B2_wape: 0.5557
- B2_bias_pct: -18.1191
- C_wape: 0.3795
- C_bias_pct: -4.6490

## Treatment metrics

- wape: 0.3665
- mae: 0.0215
- rmse: 0.2607
- mase: 0.5661
- bias_pct: -9.6175
- asym21: 0.0351
- pos_wape: 0.3511
- B_wape: 0.5564
- B_bias_pct: 1.4999
- B_units: -27.1924
- B2_wape: 0.5972
- B2_bias_pct: 0.4195
- C_wape: 0.3405
- C_bias_pct: -10.9415

## Metric deltas (treatment - control)

| metric | control | treatment | delta |
|---|---|---|---|
| wape | 0.3964 | 0.3665 | -0.0299 |
| mae | 0.0233 | 0.0215 | -0.0018 |
| rmse | 0.2592 | 0.2607 | 0.0016 |
| mase | 0.6122 | 0.5661 | -0.0461 |
| bias_pct | -6.1492 | -9.6175 | -3.4683 |
| asym21 | 0.0367 | 0.0351 | -0.0016 |
| pos_wape | 0.3655 | 0.3511 | -0.0145 |
| B_wape | 0.5101 | 0.5564 | 0.0463 |
| B_bias_pct | -18.2274 | 1.4999 | 19.7272 |
| B2_wape | 0.5557 | 0.5972 | 0.0415 |
| B2_bias_pct | -18.1191 | 0.4195 | 18.5387 |
| C_wape | 0.3795 | 0.3405 | -0.0391 |
| C_bias_pct | -4.6490 | -10.9415 | -6.2925 |
| asymmetric_loss_2_1_direct | 0.0367 | 0.0351 | -0.0016 |

## Prediction quality

| view | V2 Control | Asym Quantile (2/3) |
|---|---|---|
| underforecast_rate | 0.0107 | 0.0139 |
| overforecast_rate | 0.9889 | 0.0642 |
| exact_match_rate | 0.0004 | 0.9219 |
| residual_mean | -0.0036 | -0.0057 |
| residual_median | 0.0011 | 0.0000 |
| pred_mean | 0.0551 | 0.0531 |
| pred_p50 | 0.0011 | 0.0000 |
| pred_p95 | 0.0138 | 0.0016 |
| pred_p99 | 1.3027 | 1.2780 |
| pred_max | 13.9632 | 13.0453 |
| n_zero_predictions | 214 | 479934 |
| n_negative_predictions | 0 | 0 |

| breakdown | V2 Control | Asym Quantile (2/3) |
|---|---|---|
| spike weeks WAPE | 0.3280 | 0.2881 |
| spike weeks bias% | -15.0785 | -23.9957 |
| spike weeks under-rate | 0.2696 | 0.3885 |
| high-CV products (train CV>2) WAPE | 0.4386 | 0.4199 |
| high-CV products (train CV>2) bias% | -5.3375 | -4.6665 |
| high-CV products (train CV>2) under-rate | 0.0094 | 0.0111 |
| zero-demand rows WAPE | — | — |
| zero-demand rows bias% | — | — |
| zero-demand rows under-rate | 0.0000 | 0.0000 |

## Pre-registered criterion results (pass/fail)

- C1_global_wape_not_materially_worse: PASS
- C2_B_wape_improved_ge_5pct: FAIL
- C3_B_bias_moves_toward_zero: PASS
- C4_asym21_not_materially_worse: PASS
- C5_B2_within_tolerance: FAIL
- C6_C_within_tolerance: PASS
- C7_MAE_not_degraded: PASS
- C7_RMSE_not_degraded: PASS
- C7_MASE_not_degraded: PASS
- C7_POS_WAPE_not_degraded: PASS
- C8_prediction_integrity: PASS

## Final classification

**NO_VALID_IMPROVEMENT**
- Rationale: Treatment improves the targeted underforecasting dimension but violates pre-registered criterion/criteria: C2_B_wape_improved_ge_5pct, C5_B2_within_tolerance.

## Should a candidate be considered for further testing?

- **NO in current form** - the treatment improves the targeted dimension but violates the pre-registered constraint(s) above; the trade-off is documented. Any future variant must be re-evaluated as a new pre-registered arm.

## Governance

- V2 remains at **Staging**; Part-3 history is unchanged; no V3; no stage transition; Test untouched.
- MLflow experiment `part3_b_cohort_asymmetric_exp` (additive; runs tagged `test_used=false`, `validation_only`).

> The protected Test split was never opened by this experiment; no objective choice, hyperparameter, feature, or decision used Test evidence. The historical post-hoc Test result (B WAPE=0.628239, gate FAIL) is preserved unchanged and quoted only as history.
