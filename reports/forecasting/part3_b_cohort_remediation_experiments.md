# Part-3 Controlled B-Cohort Remediation Experiments (validation-only)

* Generated: 2026-09-23T20:34:46.612239+00:00

## 0. Protocol (pre-registered)

- Three arms, same Train/Validation, same frozen log-LGBM architecture, same hyperparameters (`WINNER_PARAMS`, `n_estimators=600`, `random_state=1`, no early stopping).
- Validation-only model selection; Test never accessed; Quality Gate thresholds unchanged; no V3; no stage transition; V1/V2 provenance untouched.

Pre-registered decision criteria (relative to `v2_control`):

| criterion | threshold |
|---|---|
| B WAPE | `<= control x 0.95` |
| B bias | `|..| <= control x 0.75` and `>= control` |
| global WAPE | `<= control + 0.01` |
| B2 WAPE | `<= control + 0.03` |
| C WAPE | `<= control + 0.01` |
| MAE | `<= control x 1.05` |
| RMSE / Asym21 / PosWAPE | `<= control x 1.10` (gross guards) |

## A. Baseline (V2 control reproduction)

- Control reproduces V2 validation references: global WAPE 0.3964 (ref 0.3964), B WAPE 0.5101 (ref 0.5101460932701322), B bias -18.2274% (ref -18.227).
- Control validation reproduction OK: True

## B. Arm 2 - WAPE-aligned sample weights

- Formula: `min(max(1, d/p50+), p99+/p50+)` with `p50+`, `p99+` from positive TRAIN demand only.
- Stats: {"formula": "min(max(1, d/p50+), p99+/p50+)", "p50_positive_train": 1.0, "p99_positive_train": 6.0, "cap": 6.0, "n_rows": 2428492, "n_zero_demand_rows": 2386799, "weight_min": 1.0, "weight_p50": 1.0, "weight_p95": 1.0, "weight_max": 6.0, "weight_mean": 1.0061692605946406, "source": "train-labels + train statistics only"}
- Rationale: `1+log1p` weights are sub-linear; demand-level alignment gives every unit of demand equal training focus (the WAPE denominator perspective), directly countering the diagnosed scale shrinkage on high-demand weeks; `d=0` rows stay at weight 1 to preserve the well-learned zero region.
- Leakage: weights use training labels + train-set statistics only; applied to train rows only; nothing Test.
- Metrics: {
 "wape": "0.4580",
 "mae": "0.0269",
 "rmse": "0.2580",
 "mase": "0.7075",
 "bias_pct": "7.6634",
 "asym21": "0.0381",
 "pos_wape": "0.4317",
 "B_wape": "0.5308",
 "B_bias_pct": "-6.9375",
 "B_units": "125.7772",
 "B2_wape": "0.5726",
 "B2_bias_pct": "-4.1034",
 "C_wape": "0.4465",
 "C_bias_pct": "9.3200"
}

## C. Arm 3 - Spike-context features

- Added features (4): `peak_8w, lag1_ratio_to_8w_peak, weeks_since_peak_8w, burst_count_16w` (schema 93 -> 97).
  - `peak_8w`: max demand over previous 8 weeks
  - `lag1_ratio_to_8w_peak`: demand_lag_1/(peak_8w+1)
  - `weeks_since_peak_8w`: weeks back to latest 8w peak (8 sentinel)
  - `burst_count_16w`: count weeks (16w) with demand>2*rolling_mean_4w
- Causality: every feature references only earlier rows in the combined (train+val) per-product timeline (shifted before windows); no future information. Seam note: Train/Validation share one boundary week (`2018-01-29/2018-02-04`) whose Validation rows may see the identical-week Train copy as preceding row (contemporaneous duplicate, never future).
- Weights unchanged (1+log1p) - the ONLY intervention is the feature set.
- Metrics: {
 "wape": "0.3960",
 "mae": "0.0233",
 "rmse": "0.2582",
 "mase": "0.6116",
 "bias_pct": "-5.8826",
 "asym21": "0.0366",
 "pos_wape": "0.3658",
 "B_wape": "0.4327",
 "B_bias_pct": "-0.9380",
 "B_units": "0.2908",
 "B2_wape": "0.5841",
 "B2_bias_pct": "-42.7431",
 "C_wape": "0.3953",
 "C_bias_pct": "-5.7544"
}

## D. Comparison

| Metric | V2 Control | Arm 2: Weights | Arm 3: Spike Features |
|---|---|---|---|
| wape | 0.3964 | 0.4580 | 0.3960 |
| B_wape | 0.5101 | 0.5308 | 0.4327 |
| B_bias_pct | -18.2274 | -6.9375 | -0.9380 |
| B2_wape | 0.5557 | 0.5726 | 0.5841 |
| B2_bias_pct | -18.1191 | -4.1034 | -42.7431 |
| C_wape | 0.3795 | 0.4465 | 0.3953 |
| C_bias_pct | -4.6490 | 9.3200 | -5.7544 |
| mae | 0.0233 | 0.0269 | 0.0233 |
| rmse | 0.2592 | 0.2580 | 0.2582 |
| mase | 0.6122 | 0.7075 | 0.6116 |
| asym21 | 0.0367 | 0.0381 | 0.0366 |
| pos_wape | 0.3655 | 0.4317 | 0.3658 |

- Feature count: Control 93 / Arm2 93 (weights only) / Arm3 97 (+4).
- Validation rows: 520,391 | B rows: 589 | B products: 37
- Training config: log-LGBM, WINNER_PARAMS, n_estimators=600, seed=1, train rows 2,428,492.

## E. Error analysis (spike / high-volatility / zero rows)

| breakdown | V2 Control | Arm 2 | Arm 3 |
|---|---|---|---|
| spike weeks WAPE | 0.3280 | 0.3828 | 0.3279 |
| spike weeks bias% | -15.0785 | -3.4152 | -14.7990 |
| spike weeks under-rate | 0.2696 | 0.2330 | 0.2645 |
| spike weeks err-share | 0.5491 | 0.5545 | 0.5495 |
| high-CV products (train CV>2) WAPE | 0.4386 | 0.4976 | 0.4384 |
| high-CV products (train CV>2) bias% | -5.3375 | 9.4542 | -5.1758 |
| high-CV products (train CV>2) under-rate | 0.0094 | 0.0079 | 0.0093 |
| high-CV products (train CV>2) err-share | 0.5995 | 0.5886 | 0.5998 |
| zero-demand rows WAPE | — | — | — |
| zero-demand rows bias% | — | — | — |
| zero-demand rows under-rate | 0.0000 | 0.0000 | 0.0000 |
| zero-demand rows err-share | 0.0778 | 0.0575 | 0.0763 |

## F. Decision

**NO_VALID_IMPROVEMENT**
- Rationale: No non-control arm satisfies every pre-registered criterion on Validation.

- V2 remains the current candidate (Staging). Next diagnostic question: whether B-cohort error is reducible without degrading the long tail, or whether the B gate bound is achievable by cohort-specific post-processing (e.g., bounded bias correction) evaluated on Validation.

## G. Next step

- V2 remains at **Staging**; Part-3 history (V2 Quality Gate FAIL) is unchanged. This experiment created no V3 and made no Production decision. A separate, explicitly authorized lifecycle evaluation is required before any promotion.
- MLflow experiment `part3_b_cohort_remediation_exp` contains the three runs (additive; no registry interaction).

## H. Test protection

> The protected Test split was never opened by this experiment; no experiment selection, feature choice, or weighting decision used Test evidence. The historical post-hoc Test result (B WAPE=0.628239, gate FAIL) is preserved unchanged and quoted only as history.
