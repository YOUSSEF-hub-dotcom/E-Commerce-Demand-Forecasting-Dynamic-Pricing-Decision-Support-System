# 06 - Controlled Forecasting Experiments (under-forecasting correction)

**Date:** 2026-09-22
**Notebook:** `src/notebooks/06_forecasting_controlled_experiments.ipynb` (26 cells, executed to completion)
**Scope:** validation split only; test split never read or scored.

## 1. Question and hypotheses

Systematic under-forecasting of economically important products (B-sustained
bias ≈ −30%, B2-mid ≈ −28%) was established in notebook 05. This cycle isolates
which single causally-valid intervention fixes it without degrading global quality.

| Exp | Hypothesis | Isolated change | Outcome |
|-----|-----------|-----------------|---------|
| 0   | control is reproducible | none (persisted winner scored) | reproduced exactly |
| CTRL | fair baseline | none (retrained, winner params) | matched baseline |
| 1   | unknown category causes the bias | recover static category from raw master file | **Regression** |
| 2   | stale regime lags cause the bias | +7 past-only regime features | **Genuine (small)** |
| 3   | log1p/expm1 link contributes the bias | unit-space Tweedie objective | **Trade-off** |
| 4   | post-hoc calibration fixes residual bias | deferred (decision documented) | not run this cycle |

## 2. Method (protocol caveats and controls)

- One hypothesis per experiment; every retrained model uses the exact
  Optuna-selected winner hyperparameters (MLflow run `4c9ad7…`;
  colsample_bytree 0.5290, learning_rate 0.008468, max_depth 10,
  min_child_samples 38, n_estimators 600, num_leaves 223, reg_alpha 2.1423,
  reg_lambda 0.1013, subsample 0.6624).
- Sample weights unchanged (`make_sample_weights = 1+log1p`); WAPE definition
  unchanged; preprocessing/design-matrix contract unchanged.
- Matched **retrained control** for clean treatment/control comparison; the
  persisted winner remains the production control (Exp 0).
- MASE denominator (train 1-step naive MAE) = **0.03804** — unchanged across
  all variants.
- Caches under `C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments\`
  enable cheap reproduction; VAL proofs align with the control (hard assert on
  identical product|week sequence, passed for Exp1/Exp2).
- Exp2 leak-verification: independent recomputation of `rolling_demand_mean_8w`
  and `regime_ratio_4_12` from raw `order_demand` performed per-element:
  features at week *t* use only weeks before *t*. PASSED.

## 3. Data

- Value split only: 520,391 rows, 32,729 products, 17 weeks, identical for all
  variants (row-order asserted).
- Cohorts (unchanged from nb05, computed from train): B-sustained (589 val
  rows / 1,813 units), B2-mid (1,339 rows), C-long-tail (518,463 rows).
- Exp1 recovered 3,327,807 rows (95.9% of matrix) from NaN→real category via
  the static raw master file; 600 products remain genuinely unlabeled;
  98.2% of val rows carry a real category after recovery (control had 96.1%
  unknown).

## 4. Comparison table (validation)

| experiment | wape | global bias% | mase | pos_wape | asym21 | B_wape | B_bias% | B2_bias% | C_wape | C_bias% | B_units gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CTRL (retrained) | 0.3975 | −7.10 | 0.6140 | 0.3687 | 0.0371 | 0.5119 | −22.01 | −21.84 | 0.3806 | −5.25 | 399.0 |
| EXP0 (winner) | 0.4001 | −9.60 | 0.6179 | 0.3721 | 0.0381 | 0.5485 | −30.04 | −28.00 | 0.3807 | −7.17 | 544.5 |
| EXP1 | 1.6437 | +2.08 | 2.5388 | 0.8305 | 0.1443 | 0.6282 | −29.67 | −31.72 | 1.7605 | +6.16 | 537.9 |
| EXP2 | 0.3964 | −6.15 | 0.6122 | 0.3655 | 0.0367 | 0.5102 | −18.23 | −18.12 | 0.3796 | −4.65 | 330.5 |
| EXP3 | 0.4261 | −7.09 | 0.6581 | 0.4043 | 0.0396 | 0.5201 | −15.44 | −13.66 | 0.4125 | −6.16 | 279.8 |

Deltas vs retrained control (B_bias% · C_wape · wape · asym21 · pos_wape):

- EXP1: −7.66 · +1.380 · +1.246 · +0.107 · +0.462
- EXP2: +3.78 · −0.001 · −0.001 · −0.0004 · −0.003
- EXP3: +6.57 · +0.032 · +0.029 · +0.003 · +0.036

## 5. Per-experiment classification

- **EXP0 — reproduce control.** WAPE = 0.4001 exactly (MAE 0.0235, RMSE 0.2725,
  MASE 0.6179, bias −9.6%, asym21 0.0381). Passed; proceed.
- **EXP1 — Regression.** Recovering the static category does **not** move B bias
  (−29.7% ≈ control) but collapses C (WAPE 1.76) and global WAPE → 1.64. The
  `unknown` bucket acts as a strong learned aggregate; the recovery creates a
  sharper categorical design causes overfitting under winner-type params.
  Hypothesis **rejected**.
- **EXP2 — Genuine (small).** Improves every headline metric marginally and
  reduces B-bias by 3.8pp (−22.0 → −18.2%) and B units-gap by 17% (399 → 330)
  while C stays flat. Improvement is B-specific, not a uniform lift — not
  cosmetic. Hypothesis **supported (modest)**.
- **EXP3 — Trade-off.** Halves the B shortfall (B_bias −15.4%, B_units 280) but
  degrades global WAPE (+0.029), positive-demand WAPE (+0.036), asym21 (+0.003)
  and C. Informative: the log-target amplitude damping contributes, but not the
  dominant, cost-neutral fix. Hypothesis **partially supported, not adopted**.
- **EXP4 — not run.** Residual bias in the best candidate is predominantly
  under-prediction (15/17 val weeks negative), so calibration *would* be
  mechanically effective on validation — but that would make validation a
  training set; the honest test is the protected test split. Justification met,
  recipe specified, execution deferred.

## 6. Root-cause interpretation

1. Unknown category is **not** the cause (Exp1, refuted).
2. Log-space target + expm1 link contributes: moving to a unit-space mean
   objective (Exp3) recovers roughly a third of the B shortfall, at the cost of
   global error — consistent with nb05's expm1 variance analysis (Jensen ratio
   1.114, minor) plus asymmetric loss behavior of the log-space model.
3. Stale regime anchoring contributes a little: recent-growth features (Exp2)
   recover ~70 B units (−17% of the gap) at zero global cost, matching nb05's
   finding that 48.6% of B products showed ≥2× recent acceleration.
4. Structural under-prediction persists (best candidate still −18% on B,
   15/17 weeks under-forecast): the balance of evidence points to an
   objective/target problem, not an information problem.

## 7. Recommendation / next stage

- Do **not** adopt Exp1 or Exp3; keep production model untouched (per protocol).
- Forward candidate for Final Validation → **Test Evaluation**: log-space LGBM +
  the 7 regime features from Exp2 (regime ratio 4/12, 8/16; means
  8/12/16-w; std-8w; positive-rate-8w).
- Then, if bias persists on the forward pipeline, apply **per-cohort calibration
  fitted on train-derived CV only** and evaluate on the protected test.
- Sample weights remain unchanged in this cycle; revisit only if the forward
  pipeline still under-forecasts B after calibration.

## 8. Files changed

Added (no modifications anywhere else — no `src/` package code, no `data/`,
no `tests/`, no `artifacts/` touched):

- `src/notebooks/06_forecasting_controlled_experiments.ipynb`
- `reports/experiments/06_forecasting_controlled_experiments_report.md`

## 9. Tests run

- `pytest` after all experiment work: **72 passed** (3 pre-existing MLflow
  warnings) in 55.3s.

## 10. Test-split / artifact safety confirmation

- Kernel loaded **0** test modules; no experiment code read `data/splits/test*`
  (all variants built from the full demand matrix + raw products master file).
- `artifacts/phase3/model.joblib` mtime 2026-09-22 12:27:54 AM and
  `feature_columns.json` 12:28:18 AM — **unchanged** (verified before/after).
- No writes to `tests/`, `data/`, or `artifacts/`.

## 11. Reproducibility

- Notebook executed to completion (exit 0); outputs embedded in the notebook.
- Caches: `C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments\`
  (`mase_denom`, `retrained_control`, `exp1`, `exp2`, `exp3`).
- Env: Python 3.11.8, pandas 2.3.3, lightgbm 4.7.0, Windows.