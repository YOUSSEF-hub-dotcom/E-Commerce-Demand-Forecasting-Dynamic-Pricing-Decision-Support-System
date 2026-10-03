# Part 3 — B-Cohort Error Analysis and Root-Cause Investigation

**Model**: `demand_forecasting_final` version 2 (frozen, MLflow `models:/demand_forecasting_final/2`)
**Scope**: Diagnosis only — no model modification, no retraining, no tuning, no `--force`, no lifecycle change.
**Split usage**: `data/splits/train.csv` (75 weeks) for cohort, history and volatility statistics; `data/splits/val.csv` (17 weeks) as the development/evaluation set. **The Test split is never read by this module.**
**Data status**: the post-hoc Test evaluation (B WAPE `0.628239`, gate FAIL) is fixed historical evidence and is preserved unchanged.

---

## 1. Purpose, sequence and provenance

The Part-3 quality gate failed on **one and only one** criterion in the post-hoc Test evaluation:

| criterion | bound | result |
|---|---|---|
| `B_wape` | `<= 0.60` | **0.628239 → FAIL** |
| all other 18 criteria (12 numeric + 7 technical) | — | PASS |

This document answers two questions with **Train + Validation evidence only**:

1. **Where** does V2 fail inside the B-sustained cohort? (What rows, products, weeks, categories, patterns?)
2. **Why** does it fail there? (Purely diagnostic root-cause hypotheses with measured evidence — none assumed.)

It follows the mandated sequence: **Quality-Gate FAIL → B-cohort error analysis → root cause → controlled validation experiment → compare with V2 → V3 only if justified → then lifecycle.** This report covers the first four steps (the controlled experiment is *recommended and specified* in §7, not executed here — Part 4).

**Provenance / reproducibility check** (performed by `python -m src.models.b_cohort_error_analysis`): the frozen V2 model was re-scored on the **Validation** split and the official V2 validation references were reproduced exactly:

| metric | recomputed | reference |
|---|---|---|
| global val WAPE | `0.39637084` | `0.3964` |
| B-cohort val WAPE | `0.5101460932701322` | `0.5101460932701322` (VAL_B_WAPE_REF) |
| B-cohort val Bias | `−18.227%` | `−18.227` (VAL_B_bias) |

Input contract honored: frozen schema 93 columns, exact column order, all inputs cast to `float64` before `predict` (MLflow signature requires doubles).

**Method note on metrics below.** `wape` = Σ|error| / Σactual; `bias_pct` = Σerror / Σactual × 100; a *spike week* is defined here (new diagnostic construct, documented for reproducibility) as `actual > 2 × max(rolling_demand_mean_4w, demand_lag_1)`; a *collapse week* as `actual == 0 and demand_lag_1 > 0`. WAPE and bias are only defined on buckets with positive Σactual; buckets whose Σactual is 0 are marked `—`. A stray per-product WAPE of `1.516` and an undefined case come from products whose val Σactual ≤ 1 ([see appendix](#8-appendix-notes-and-caveats)).

---

## 2. Output A — B-cohort profile (Validation)

| attribute | B-sustained | B2-mid | C-long-tail | overall |
|---|---|---|---|---|
| rows | **589** | 1,339 | 518,463 | 520,391 |
| products | **37** | 84 | 32,608 | 32,729 |
| share of dev (train+val) rows | 0.020% | 0.045% | 17.58% | 17.65% |
| week labels covered | 17 | 17 | 17 | 17 |
| zero-demand rows | 204 (34.6%) | 874 (65.3%) | 498,980 (96.2%) | 499,980 (96.1%) |
| mean actual / median | 3.078 / 1.0 | 1.178 / 0.0 | 0.052 / 0.0 | 0.059 / 0.0 |
| p75 / p95 / max actual | 4 / 13 / 31 | 1 / 6 / 112 | 0 / 0 / 210 | 0 / 0 / 210 |
| Σactual | 1,813 | 1,578 | 27,042 | 30,433 |
| Σ\|error\| | 924.9 | 876.8 | 10,262 | 12,064 |
| **WAPE** | **0.5101** | 0.5557 | 0.3795 | 0.3964 |
| Bias % | **−18.23** | −18.12 | −4.65 | −6.15 |
| CV (row-level actual) | 1.50 | 2.91 | 6.65 | 7.41 |

**B profile in words.** B is a tiny, high-visibility cohort: **37 products, 589 product-weeks (0.11% of all rows) but 6.0% of all Σactual in the validation window**. Its demand is intermittent *even inside a "sustained" cohort*: 34.6% of B rows are zero and the median weekly demand is 1 unit. Per-product val CV has mean ≈ 1.10 (median ≈ 0.91, p95 ≈ 2.26). Every B product spans close to the whole window (15–16 of 17 weeks), i.e. the products are continuously present; the intermittency is *within-window*, not a coverage artefact of a specific subset. Because the cohort has only 37 independent product series, its WAPE is inherently **sample-sensitive** (a point returned to in §6).

**Validation vs Test here is crucial**: on Validation, B WAPE `0.5101` is *below* the 0.60 gate bound; on the frozen post-hoc **Test** run it is `0.6282`. So the gate failure is not "B is structurally the worst cohort everywhere" (on val, B2 at `0.5557` is worse) — it is a B-specific degradation that appears on the Test window.

---

## 3. Output B — Error analysis (where does the B error come from?)

### 3.1 Product-level
- Error concentration **mirrors demand concentration** almost exactly: the top-1 product contributes 11.5% of B error (14.3% of B actual), top-5 → 44.4% (45.7%), top-10 → 67.9% (68.7%), top-20 → 89.4% (89.7%). 15 products (40.5%) concentrate 50% of B error — but also ~82% of B actual. **There is no single outlier product**; the error is broadly spread over the same products that carry the demand.
- Consistently: **leave-one-out removal of the top-abs-error product raises cohort WAPE** (0.5101 → 0.5266), i.e. that product was actually *well* forecast relative to its demand (WAPE 0.411, bias −32%) — its large error is just a large denominator in disguise.
- **10 of 37 products have per-product WAPE > 0.60**. The gate failure is an emergent property of ~¼ of the cohort; two of them (`5a848e4a` WAPE 0.640; `aca2eb7d` WAPE 0.411) dominate in absolute terms.

### 3.2 Demand-pattern decomposition (the single most important figure)
The B error is **not** spread evenly across row types:

| row type | rows | Σactual | Σ\|error\| | WAPE | bias% | error share |
|---|---|---|---|---|---|---|
| full B | 589 | 1,813 | 924.9 | **0.510** | −18.2 | 100% |
| **spike weeks** (`act > 2·max(mean4w, lag1)`) | 59 (10.0%) | 405 (22.4%) | 247.9 | **0.612** | **−58.9** | **26.8%** |
| non-spike non-zero weeks | 326 (55.4%) | 1,408 (77.6%) | 667.1 | 0.474 | −7.2 | 72.1% |
| zero weeks | 204 (34.6%) | 0 | 10.0 | — | — | 1.1% |
| collapse weeks (`act=0`, `lag1>0`) | 74 | 0 | 6.1 | — | — | 0.7% |

Read literally:

- **The model guesses zeros well.** On its 204 zero weeks the total absolute error is ~10 units (1.1% of B error). Intermittency itself is *not* the problem — the model has learned periodic zeros for these products.
- **The model systematically misses the demand bursts.** 10% of B rows (spikes) carry 22% of the demand but 27% of the error, at WAPE `0.612` and bias `−58.9%` — i.e. when demand jumps, V2 jumps **far** too weakly (see §6, H1).
- Even on the "normal" non-spike weeks the bias is still slightly negative (−7.2%), so the under-forecast is a *level* tendency, sharpened enormously on bursts.

### 3.3 Error direction
- 410 over-forecast rows (70%) but only **32.1% of B error**; the **178 under-forecast rows (30%) carry 67.9%** of the error (their WAPE 0.456, bias −45.6% on Σactual 1,375).
- Over-forecast rows have tiny bases (Σactual just 438) — WAPE 0.679, bias +67.9%, but σ small in absolute terms.
- Mean error `−0.56`, **median +0.02**: the distribution is heavily skewed — V2 slightly over-shoots the many small/zero weeks (positively skewed median) while massively under-shooting the few large weeks (negative heavy tail). This is exactly the signature of a model whose predictions are **compressed toward the conditional mean/median of demand**.

### 3.4 Temporal
- Weekly B WAPE cycles between 0.362 and 0.774 across the window; no monotone trend (early-half WAPE 0.487, bias −19.8% vs recent-half 0.538, bias −16.3%).
- Worst measured weeks: `2018-04-09` WAPE 0.774, bias +58.0 (Σactual 57 — a low-demand week the model over-shot); `2018-02-19` 0.667 (+17.1); `2018-04-16` 0.605 (−27.3). Best weeks: `2018-03-12` 0.362 (+2.7).
- **Edge-coverage artefact**: in the final val week `2018-05-21/05-27` only 4 B rows exist (all zero; abs err 0.13) and the first week has 30 rows — B products are present 15–16 of 17 weeks, so the window edges are noisy denominators, not demand collapses. (Train window: `2016-08-29 … 2018-02-04`; val first week `2018-01-29/02-04` coincides with the train's last week — a 1-week overlap in the pipeline's own rolling split, noted as a caveat, not as a cause.)

### 3.5 Category (within B)
| category | rows | Σactual | WAPE | error share | actual share |
|---|---|---|---|---|---|
| ferramentas_jardim | 99 | 571 | 0.539 | 33.3% | 31.5% |
| moveis_decoracao | 26 | 311 | 0.432 | 14.5% | 17.2% |
| relogios_presentes | 44 | 197 | 0.571 | 12.2% | 10.9% |
| **unknown** (unmapped label) | 217 | 116 | **0.716** | 9.0% | 6.4% |
| beleza_saude | 46 | 139 | 0.467 | 7.0% | 7.7% |
| informatica_acessorios | 18 | 164 | 0.358 | 6.4% | 9.0% |
| cama_mesa_banho | 28 | 100 | 0.586 | 6.3% | 5.5% |
| others (8 cats, each ≤ 3%) | 111 | 215 | ~0.41–0.64 | 11.3% | 11.8% |

Category error share broadly tracks demand share; the strongest offenders are **ferramentas_jardim** (biggest single pool of B demand, WAPE 0.539) and the **`unknown` label** (12 B products, WAPE 0.716) — i.e. B products whose category maps to the reserved `unknown` bucket. `informatica_acessorios` is the best-served major category (WAPE 0.358). Category is a *stratifier*, not an independent driver: it describes where, not why.

### 3.6 History
Every B product has 60–76 weeks of training history (bucket `(60, 90]` contains 100% of B rows) and 11–48 positive training weeks. Within B there is **no short-history population** — history length is not a root cause (it is a *defining* property of the cohort). Training non-zero-week count also splits B only into `(10,20]` (96 rows, bias −32.1) vs `(20,∞]` (493 rows, bias −13.1): more sustained training products are under-forecast less.

### 3.7 Feature-pattern buckets (descriptive only)
| split | bucket | rows | Σactual | WAPE | bias% |
|---|---|---|---|---|---|
| train CV | (0.5, 1] | 16 | 29 | 0.525 | +0.4 |
| | (1, 2] | 349 | 911 | 0.485 | −10.8 |
| | **(2, ∞)** | 224 | 873 | 0.535 | **−26.6** |
| positive_rate_8w | (0, 0.25] | 80 | 31 | 0.497 | −11.4 |
| | (0.25, 0.75] | 219 | 358 | 0.464 | −11.0 |
| | **(0.75, 1]** | 290 | 1,424 | 0.522 | **−20.2** |
| rolling mean level | (0, 1] | 164 | 136 | 0.554 | −12.7 |
| | (1, 5] | 259 | 656 | 0.545 | −8.7 |
| | (5, ∞) | 107 | 1,001 | 0.482 | −25.0 |
| lag-1 | =0 | 191 | 147 | 0.571 | −22.8 |
| | >0 | 398 | 1,666 | 0.505 | −17.8 |
| lag-4 | =0 | 174 | 207 | 0.520 | −19.8 |
| | >0 | 415 | 1,606 | 0.509 | −18.0 |
| \|price_chg_1w\| | 0 | 522 | 1,437 | 0.515 | −16.3 |
| | (0, 0.05] | 46 | 261 | 0.513 | −25.5 |
| | (0.05, 0.15] | 21 | 115 | 0.441 | −25.5 |

Take-away: **no single feature bucket explodes WAPE**, but the bias is persistently negative across nearly every bucket and grows with (a) training-time volatility (`train_cv > 2`), (b) high recent positive rate, and (c) lag-1 = 0 situations. Price movements are rare and mild within B during validation, so price is not a candidate.

---

## 4. Output C — B vs B2 vs C vs overall (validation)

| metric | B-sustained | B2-mid | C-long-tail | overall |
|---|---|---|---|---|
| WAPE | **0.5101** | 0.5557 | 0.3795 | 0.3964 |
| bias % | −18.23 | −18.12 | −4.65 | −6.15 |
| zero rows % | 34.6 | 65.3 | 96.2 | 96.1 |
| mean / median actual | 3.08 / 1.0 | 1.18 / 0.0 | 0.05 / 0.0 | 0.06 / 0.0 |

On the **validation** split B is *middle* in WAPE — B2 is numerically worse. Both B and B2 share a strong **negative bias (−18%)**, while the long tail has only −4.6%. The long tail's lower WAPE is largely a definitional effect: its actuals are dominated by zeros, which V2 predicts well; B and B2 carry the *residual demand* the model can misjudge. On the **frozen Test** result only B crossed the 0.60 bound (B 0.628, B2 0.5566, C 0.3651) while the global Test WAPE improved (0.3754 vs 0.3964 val) — i.e. the failure is B-specific *and* split-specific.

---

## 5. Output D — Root-cause hypotheses (with measured evidence)

| # | Hypothesis | Evidence (measured, Train+Val only) | Verdict |
|---|---|---|---|
| H1 | **Scale/level shrinkage on high-demand weeks.** The learner compresses predictions toward the conditional mean — driven by the MAE/relative-invariant objective and/or the log-scaled target — so V2 systematically under-forecasts exactly when B demand is high. | Spike weeks: WAPE 0.612, bias −58.9%, 27% of B error from 10% of rows; under-forecast rows carry 68% of B error; bias −18.2% overall with positive median error (+0.02) — a heavy-left-tail / compressed-prediction signature; bias negative in nearly every bucket incl. high-positive-rate rows (−20.2%). | **Primary.** Matches all directional and distributional evidence. |
| H2 | **Structural intermittency inside B (periodic zeros) confounds level learning.** Even sustained products have 35% zero weeks; the model's learned "sleepy" regime bleeds into the following week's prediction. | Zero weeks: 204 rows but only 1.1% of error (model *does* predict zeros). lag-1 = 0 rows: WAPE 0.571 (vs 0.505). Bias on lag-1=0: −22.8%. | **Contributory but not primary** (zero-predictions are accurate; the cost is the residual low-bias regime). |
| H3 | **Volatile histories are intrinsically harder and under-forecast more.** | train_cv > 2 bucket: WAPE 0.535, bias −26.6% (vs −10.8% at cv 1–2). Products with fewer training positive weeks bias more negative (−32.1%). | **Contributory** — sharpens the same shrinkage mechanism. |
| H4 | **Small-cohort sampling variation.** B is only 37 products; per-split WAPE is noisy, and the val B WAPE (0.510) already sits just below the 0.60 bound while the Test B WAPE (0.628) crosses it, despite a *better* global Test WAPE. | Cohort size 37; val B 0.5101 vs Test B 0.6282; global val 0.3964 vs Test 0.3754. B2 (harder on val, 0.5557) does **not** fail on Test (0.5566). | **Contributory to explain the gate outcome**, not the error *mechanism*. |
| H5 | Category / unknown-label routing distorts features. | `unknown` category: 12/37 products, WAPE 0.716, but only 9% of B error. | **Minor stratifier, not a driver.** |
| H6 | Price changes / history length / temporal drift. | price changes rare & mild (effect ≤ ~0.08 WAPE); all B products full-history; weekly pattern flat (early 0.487 / recent 0.538). | **Rejected** — no supporting evidence. |

**Conclusion.** The dominant mechanism is **H1 (scale shrinkage on high-demand weeks)**, amplified by **H3 (volatile histories)** and conditioned by **H2 (intermittent regimes)**; **H4** explains why the *gate* flipped on Test without global degradation. The evidence is consistent and cross-validated across §3.2, §3.3, §3.7, and the profile — no hypothesis is assumed; H6 was explicitly tested and rejected.

---

## 6. Output E — Recommended next experiment (justified)

The gate failure is real (Test B WAPE 0.628 > 0.60) and the mechanism is actionable, with measurable headroom on Validation:

- B val WAPE is 0.5101; if spike weeks alone were forecast at the *non-spike* WAPE (0.474), B val WAPE would drop to **0.479**; if spike weeks were perfectly forecast, to **0.373**.
- This makes a controlled Validation experiment **justified**. It must **not** be executed here (Part 4) and must follow the mandated sequence.

### Recommended controlled experiment — "WAPE-aligned training vs spike-context features"

**Objective.** Reduce the B-cohort relative error concentrated on high-demand weeks (H1/H3) with identical architecture and training data, choosing the winning arm **only on the Validation split**.

**Intervention (3 arms, all identical to V2's algorithm & hyperparameters; the *only* difference is the listed knob):**
- **Arm 0 (control):** exact V2 reproduction (same `final_forecast` pipeline, same seed, same 93-col schema) — re-scored against the same val.
- **Arm 1 (primary):** WAPE-aligned row weights — train with sample weight `w = 1 / (order_demand + 1)` (weight normalized to median 1 at decision time). Rationale: WAPE is the gating metric and weights every row by `1/actual`, directly countering H1's mean-shrinkage of high weeks.
- **Arm 2 (secondary):** spike/burst-context features added on the Train side (e.g. `weeks_since_last_spike`, `burst_count_12w`, `max4w_ratio_8w`) appended **before** `freeze_schema`, extending the schema (schema length deliberately tracked as a new frozen contract, version) — targeted at the residual H2/H3 conditioning.

**Train data:** the same frozen 75-week Train split used by V2 (same `build_prepared` cache → `freeze_schema`); design matrices cast to `float64` before predict. **The Test split is not used in any arm.**

**Validation protocol:** all three arms scored on the same 17-week Validation split with the same design; cohort definitions exactly as `cohort_flags` (B = `min_pos_weeks>=6 and avg_weekly>=1.0`; B2 = `avg_weekly>=0.5`; C = rest). Reported per arm: B / B2 / C / global WAPE, bias, MAE, RMSE, MASE, plus the §3.2 spike-week breakdown.

**Decision metric & success/failure criterion (Validation, pre-registered):**
- **Success** (a winning arm exists) if, on Validation, **B-cohort WAPE ≤ 0.485** (≥5% improvement over 0.5101) **and** B |bias%| ≤ 18.2 **and** global WAPE ≤ 0.4164 (0.3964 + 0.02) **and** C WAPE ≤ 0.3995 (0.3795 + 0.02).
- **Failure** otherwise → **keep V2**; do not build a V3 on an intervention that only helps B at the expense of the long tail.
- If an arm succeeds, lock the validation decision first, then touch the Test split **exactly once** (pre-registered) to confirm the lifecycle gate (all 19 criteria incl. B WAPE < 0.60), then decide the lifecycle. **No Test-driven tuning, feature choice, threshold selection, or model selection at any earlier step.**

---

## 7. Output F — Explicit Test Protection Statement

> No new experimentation, tuning, feature selection, or model decision was based on the Test split. The Test split (data/splits/test.csv) is never read by this analysis; the B WAPE=0.628239 result from the post-hoc evaluation is preserved as fixed historical evidence, and Validation is used as the development/selection evaluation set.

*(This statement is emitted verbatim from `b_cohort_error_analysis.py` (machine-readable field `test_protection_statement` in the companion `part3_b_cohort_error_analysis.json`) and is asserted by the guard tests.)*

---

## 8. Appendix — notes and caveats

- **Artifact direction is safe.** The whole investigation §2–§6 was produced by scoring V2 once, on Validation only. No prediction was ever made on the Test split by this module; the fixed Test B WAPE 0.628239 is quoted solely as frozen history.
- **Infinite/singular buckets** (`wape = 127795117802` on week `2018-05-21`; per-product WAPE 1.516 for `5411e926…`, undefined for `f1c7f353…`) arise when bucket Σactual ≈ 0 (val edge coverage and tiny-actual products). They carry <0.02% of B error and are excluded from every interpretation above.
- **Spike definition** is a diagnostic construct of this analysis (see §1); it is needed to make the concentration finding (§3.2) reproducible.
- **1-week train/val overlap** at week `2018-01-29/02-04` is a property of the pipeline's rolling split (train's last week == val's first); flagged as a caveat, believed immaterial (denominator 17 weeks).
- **No files were overwritten**: this analysis writes only `reports/forecasting/part3_b_cohort_error_analysis.json` (module) and `reports/forecasting/part3_b_cohort_error_analysis.md` (this document). Existing post-hoc & validation artifacts are untouched.