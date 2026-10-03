# Part-3 V2 Production Policy Audit — Quality Gate `B_wape <= 0.60` Provenance and Production Decision

* Generated: 2026-09-24T00:00:00+00:00
* Scope: audit only. **No model artifact modified, no Test split accessed, no MLflow version promoted, no V3, no gate threshold changed.**
* Subject: frozen `demand_forecasting_final` version 2 (MLflow Staging).

---

## 0. Outcome summary

| Item | Finding |
|---|---|
| Gate criterion audited | `B_wape <= 0.60` (one of 12 numeric + 7 technical criteria) |
| Provenance classification | **Engineering heuristic / empirically derived threshold** (documented, reference-anchored regression guard). NOT an explicit or documented business requirement. |
| "Monetarily dominant segment" premise | **Asserted in the gate justification text only; no independent repository evidence** (no revenue/price analysis; as units-only data, B carries ~6% of validation Σactual). |
| B-cohort business significance | High-visibility, economically-motivated stratum in the delivery narrative; analytically a "sustained high-demand" cohort (37 products). Business-critical label is asserted, not independently evidenced. |
| Threshold assessment | Single-split WAPE on a 37-product cohort; val 0.5101 (PASS) vs post-hoc 0.6282 (FAIL) with NO global degradation and the **documented business requirement (B/B2 bias ≥ −20%, stockout guard) PASSING** on the post-hoc holdout. |
| **Production decision** | **B — FORMAL_GATE_REVISION_REQUIRED.** |
| Operational effect | **V2 remains Staging. No automatic gate change. No promotion.** A production-owner policy decision is required before any re-baselining/re-formulation of the B WAPE guard. |
| Criterion C (GATE_REVISION_JUSTIFIED) | **Not selected** — the repository contains no documented alternative gate formulation already grounded in business requirements; proposing one IS the policy decision that belongs under B. |
| Robust Loss experiment | **Not started** (per mandate). Blocker is primarily gate-policy/provenance, not demonstrated model performance; a future Robust Loss arm, if pursued, must be a separate validation-only, pre-registered experiment. |

---

## 1. Phase 1 — Quality Gate provenance audit

### 1.1 Where the threshold lives

- Canonical source: `src/models/quality_gate.py`, numeric criterion `B_wape`, `op="le"`, `bound=0.6000`, documented under "Quality Gate Thresholds (Phase D)".
- Also mirrored (verbatim copies of the same table): `reports/forecasting/part3_v2_lifecycle_report.md`, `part3_v2_posthoc_holdout_evaluation.md`, `part3_final_lifecycle_report.md`, and the MLflow run artifacts `final_report.md` / `posthoc_holdout_report.md`.
- Consumed unchanged by `src/models/posthoc_holdout.py` (`gate_decision`, `production_gate` tags) and by the validation lifecycle report.

### 1.2 The documented justification (verbatim, from `quality_gate.py`)

> `B_wape` — "B cohort WAPE (sustained high-volume, the business-critical segment)"
> bound `0.6000`; reference "0.5485 (winner) / 0.5102 (EXP2)"
> justification: "B cohort is the monetarily dominant segment. Reference 0.51-0.55; 0.60 bound rejects >~10% regression on high-volume products."

Gate policy statement (top of module):

> "Tolerance bands are set relative to the frozen EXP2 champion (val) and the persisted Part-3 winner reference. Global error bounds allow test-split variance without tolerating regression; **cohort-bias bounds encode the business requirement** that emerged from the defect analysis: sustained high-volume demand must not be systematically under-forecast by more than -20%."

### 1.3 What the source claim actually encodes

1. The reference values are **validation** metrics of the pre-V2 candidates: the persisted "winner"/EXP0 (val B WAPE 0.5485) and the EXP2 candidate which became V2 (val B WAPE 0.5102), both in `reports/experiments/06_forecasting_controlled_experiments_report.md` §4.
2. The bound `0.60` is therefore a **regression guard above a measured reference band** (0.51–0.55). Arithmetic nuance: `0.60/0.55 = +9.4%` and `0.60/0.5102 = +17.6%`, so the stated "rejects >~10% regression" reads naturally off the reference high-end; the band arithmetic is approximate, but the intent (reject material regression on the highest-demand cohort) is unambiguous.
3. The label "monetarily dominant segment" and "business-critical segment" appear **only in the gate's own justification text** (and its verbatim copies). The repository contains **no** revenue-weighted analysis supporting that claim. In the only in-repo unit-based cohort profile (`part3_b_cohort_error_analysis.md` §2), B is 37 products / 589 val rows (0.11% of rows, ~0.020% of dev rows) holding **6.0% of validation Σactual** — a small, high-visibility cohort, **not** the unit-dominant one (C-long-tail holds ~89% of val Σactual); "monetarily dominant" would further require per-unit revenue weighting, which no repository artifact computes.
4. Contrast with the criteria that the module itself labels "business requirement": `B_bias_pct >= -20` and `B2_bias_pct >= -20` ("Key business requirement distilled from the defect analysis … the gate mandates >= -20% … may not slip back towards the -30% stockout regime"). Only the **bias** bounds carry the business-requirement designation; **all WAPE bounds (including B_wape 0.60) are framed as reference-anchored tolerance bands.**

### 1.4 Pre-registration status (not reverse-engineered)

- The Phase-D threshold table was defined from **validation** references of the winner and EXP2 (experiment cycle 2026-09-22) and shipped in `quality_gate.py` before any Test evaluation.
- `part3_v2_lifecycle_report.md`: validation gate evaluated with `group="v2_validation"`, "thresholds UNCHANGED" → PASS → v2 (provisionally) Production.
- `part3_v2_posthoc_holdout_evaluation.md`: "no … re-thresholding … before or after this evaluation"; gate consumed unchanged → FAIL → v2 Production→Staging.
- Therefore `0.60` was **pre-registered relative to the Test split**; it was not tuned against the post-hoc Test result and it is not a threshold manufactured to fail V2.

### 1.5 Provenance classification (single label)

**Empirically derived threshold (engineering heuristic)** — documented, reference-anchored, pre-registered; no explicit stakeholder/business requirement artifact exists anywhere in the repository (the only `requirements.txt` files are MLflow pip dependencies; the only repo-wide docs are `PROJECT_BLUEPRINT.md` and `PROJECT_MAP.md`, neither of which specifies numeric gate bounds). Secondary label from the gate text itself: **domain/business guardrail (asserted, unevidenced)** — the "monetarily dominant / business-critical" premise is asserted but not independently substantiated.

---

## 2. Phase 2 — B cohort business significance

### 2.1 Definition (source: `src/models/final_forecast.py`, `cohort_flags`)

- Features computed from **Train** only: per-product `pos` = # weeks with demand > 0; `avg_wkly` = Σdemand / #weeks.
- Constants (pinned in guard tests): `B_MIN_POS_WEEKS = 6`, `B_MIN_AVG_WEEKLY = 1.0`, `B2_MIN_AVG_WEEKLY = 0.5`.
- `B-sustained` = `pos >= 6 and avg_wkly >= 1.0`; `B2-mid` = `pos >= 6 and avg_wkly >= 0.5` and not B; `C-long-tail` = rest.

### 2.2 Cohort profile (validation, source: `part3_b_cohort_error_analysis.md` §2)

| attribute | B-sustained | B2-mid | C-long-tail |
|---|---|---|---|
| products | **37** (of 32,729; 0.11%) | 84 | 32,608 |
| val rows | 589 (0.11%) | 1,339 | 518,463 |
| Σactual (val units) | 1,813 (**6.0%**) | 1,578 | 27,042 |
| zero-demand rows | 34.6% | 65.3% | 96.2% |
| mean / median actual | 3.08 / 1.0 | 1.18 / 0.0 | 0.05 / 0.0 |
| WAPE / bias | 0.5101 / −18.23% | 0.5557 / −18.12% | 0.3795 / −4.65% |

### 2.3 Business significance

- **What B is**: the cohort of products with sustained (≥6 positive weeks) and modest-to-high (≥1 unit/week average) demand — the "best sellers" stratum. Every B product is continuously present (15–16 of 17 val weeks); intermittency is within-window (34.6% zero weeks), not coverage.
- **Why a separate threshold**: cohort-specific gating keeps segment identity; the gate explicitly guards the long tail "must not be sacrificed for segment gains" (`C_wape` justification) and enforces "MUST NOT systematically under-forecast" on B/B2 (`B_bias_pct` justification) — B is the visibility stratum where the delivery/promise of the whole experiment cycle is judged.
- **Consequences of high WAPE in B**: within the project's cost-of-error framework (stockout/lost-sales = 2× the inventory-overage penalty, `src/utils/metrics.py`), a high B WAPE dominated by **under-forecasts** = lost sales on the best sellers (val evidence: 30% of B rows carry 68% of B error; spike weeks bias −58.9%). This is the *business* adverse outcome the gate was built to prevent.
- **Critical nuance (post-hoc holdout)**: on the frozen Test run, V2's B **bias is +9.85%** (over-forecast) and B2 **+0.86%** — the stockout mechanism **does not manifest on Test**; the failing B `WAPE=0.628` reflects absolute error magnitude (and its drivers in that window), not the documented under-forecast risk. The documented business requirement (bias ≥ −20%) PASSES on Test.
- **Is B "business-critical" or merely analytical?** The repo's **narrative** treats B as business-critical ("economically important products", "monetarily dominant segment"). The **evidence** substantiates B as: (a) the analytically-defined sustained-demand stratum, (b) the cohort where a model shortfall previously concentrated (−30% bias defect), and (c) a small (37-product) high-visibility unit bundle. Independent **economic** significance (revenue/lifetime-value weighting) is asserted but **not computed anywhere in the repository**.

---

## 3. Phase 3 — Threshold assessment

### 3.1 The evidence to weigh

| quantity | value | meaning |
|---|---|---|
| B WAPE, validation | 0.5101 | PASS vs 0.60 → was `<= 0.60` on the validation gate |
| B WAPE, post-hoc Test | 0.6282 | FAIL vs 0.60; the single failing criterion of 19 |
| Val→Test global WAPE | 0.3964 → 0.3754 | improved (−5.3%) while B degraded |
| B2 (harder on val: 0.5557) | 0.5566 on Test | does **not** fail; bound 0.65 |
| B bias / B2 bias on Test | +9.85% / +0.86% | pass the documented ≥ −20% business requirement |
| asym21 on Test | 0.025095 | improved (val 0.0367/PASS ≤ 0.04) |
| B cohort size | 37 products, 588–589 rows/split | small; per-split WAPE has high sampling variability |

### 3.2 Analysis

1. **The 0.60 bound has no independently justified business interpretation in-repo.** It is a regression guard above val references (0.51–0.55). No artifact ties B WAPE to a revenue-loss figure, a service-level target, or an owner-specified tolerance. (Its "monetarily dominant" premise is asserted, not evidenced.)
2. **Small-cohort sampling variability is real and documented.** The in-repo error analysis (H4) already classified B as "only 37 products; per-split WAPE is noisy", citing val 0.510 vs Test 0.628 while global improved and B2 "harder on val" did not fail. With a single Test realization, the crossing is consistent with sampling variation but cannot be *proven* to be just noise from one observation. This is exactly what makes an automatic engineer-side change indefensible **and** what makes formalizing the guard as a hard long-run requirement require owner input.
3. **The failing criterion is not the one the repo labels as the business requirement.** The only B-related "business requirement" in the gate is the stockout/bias rule (≥ −20%), which V2 **passes** on the post-hoc holdout. The failure is an **engineering tolerance** (WAPE regression guard) on a small-cohort, single-split estimator.
4. **No better model currently exists.** All three validation-only interventions (WAPE-aligned weights; spike features; asymmetric 2:1 quantile) returned `NO_VALID_IMPROVEMENT`. Nothing in hand would pass the full gate with a defensible margin that simply exchanging the threshold would "unlock".
5. **Symmetry caution.** Under single-split, small-cohort WAPE gating, PASS/FAIL boundaries near the bound (0.51 val vs 0.60) flip on plausible sampling noise; formally, this means the gate is *statistically fragile as written* for this cohort — a policy observation, not an argument for a specific new number. Any new number chosen from V2's results would violate the "no threshold tuned to the model" rule.

### 3.3 Conclusion of the assessment

- Keeping `0.60` is **defensible as the conservative status quo** (pre-registered, reference-anchored, not reverse-engineered).
- The existing evidence does **not** justify changing the threshold by an engineer; it also does **not** establish that `0.60` is a hard business requirement.
- The responsible posture is therefore: **enforce the current gate now; require an explicit production-policy decision before any re-baselining or re-formulation.**

---

## 4. Phase 4 — Production decision

**Decision: B — FORMAL_GATE_REVISION_REQUIRED**

Exactly one of the three allowed decisions is returned: **B**. Rationale against A and C:

- **Not A (KEEP_GATE_FAIL)** — `B_wape <= 0.60` is demonstrably not a *genuine, documented business requirement* (its own module text attaches the "business requirement" label to the bias bounds, not to the WAPE bounds); the "monetarily dominant" premise is unevidenced; and single-split gating of a 37-product cohort is statistically fragile as a hard rule. Treating it as an authoritative business requirement would overstate the evidence.
- **Not C (GATE_REVISION_JUSTIFIED)** — C requires "sufficient independent evidence … that **a different gate formulation is already supported by documented business requirements**." The repository contains **no such alternative formulation**: the documented business requirements (bias ≥ −20%, asym 2:1) are *separate criteria that already coexist with* `B_wape`; no repo artifact proposes dropping/replacing the B WAPE guard with a variance-aware or bias-only rule. Crafting that now WOULD be the business-policy decision — which belongs under B.

### 4.1 Required actions (all executed as part of this deliverable)

1. **Do NOT change the gate.** `src/models/quality_gate.py` remains untouched; `B_wape <= 0.60` remains enforced.
2. **Keep V2 in Staging.** No promotion, no V3, no automatic transition.
3. **Preserve history.** The old phase-D gate definition, the PASS validation result, and the FAIL post-hoc holdout result are retained verbatim; the FAIL is never relabeled PASS.
4. **Document the evidence** (this report) and **record the production policy** (below).

### 4.2 The production policy now in force

- **Historical gate definition:** Phase-D threshold table, 12 numeric + 7 technical criteria, `B_wape <= 0.60` justified as a reference-anchored regression guard; pre-registered before any Test evaluation.
- **Validation gate result:** **PASS** — B WAPE 0.5101 ≤ 0.60 (all 19 criteria green) → v2 (provisionally) Production.
- **Post-hoc holdout result:** **FAIL** — B WAPE 0.6282 > 0.60, the sole failing criterion of 19; → v2 Production→Staging. Classified `post_hoc_holdout_evaluation` (not an independent untouched final-test estimate), executed exactly once.
- **Business requirement:** B/B2 must not systematically under-forecast (bias ≥ −20%, stockout guard); asym 2:1 cost-of-error framework. **On the post-hoc holdout these PASS** (B +9.85%, B2 +0.86%, asym21 0.025095).
- **Technical heuristic:** all WAPE tolerance bands, including `B_wape <= 0.60`, are reference-anchored regression guards (engineering tolerances), not owner-specified requirements.
- **Production policy (binding):** a candidate is Production only when all 19 criteria pass on the held-out evaluation. V2 does not currently satisfy this policy → **V2 remains Staging**, and this is a FAIL policy outcome for the current gate.
- **Final production decision:** **B — FORMAL_GATE_REVISION_REQUIRED.** Blocker is treated as a gate-policy/provenance issue (single-split small-cohort WAPE guard whose business grounding is asserted, not documented), not as demonstrated model-performance failure. V2 stays Staging until an explicit owner decision.

### 4.3 The required business-policy decision (proposal, not adoption)

A production owner must decide, on the record:

1. **Confirm or reject the "B = monetarily dominant segment" premise** (with revenue evidence or explicit sign-off). Until then, `0.60` rests on an unevidenced label.
2. **Decide how a 37-product cohort's single-split WAPE may be gated:** (i) keep the current regression-guard form (≈10% above val reference); (ii) treat the bias/stockout rule as the sole B business requirement and route WAPE regression into a diagnostic (non-blocking) track; or (iii) adopt a variance-aware/formulation change (e.g., multi-week or multi-split robustness). Any chosen formulation must be **pre-registered and applied prospectively to all future candidates**, including a fresh evaluation of V2 under the new policy ONLY as a re-scored, clearly-labeled policy read — never as a rewrite of the historical FAIL.
3. **If (i) is reaffirmed**, `0.60` stays, V2 stays Staging, and a future **Robust Loss (or other) intervention** becomes relevant as a separate validation-only, pre-registered experiment — none started now.

### 4.4 Governance checklist (this deliverable)

| Guard | Status |
|---|---|
| No model artifact modified | ✅ |
| No Test split accessed | ✅ (audit reads reports/source only) |
| No MLflow version promoted / no V3 | ✅ |
| No automatic threshold change | ✅ |
| Historical V1/V2 records intact | ✅ |
| Post-hoc nature of Test evaluation preserved | ✅ |
| Tests added only for gate-policy change | ✅ no change → no new gate tests |

---

## 5. Files

- This report: `reports/forecasting/part3_v2_gate_production_policy.md`
- Machine-readable: `reports/forecasting/part3_v2_gate_production_policy.json`
- Read-only sources consulted: `src/models/quality_gate.py`, `src/models/final_forecast.py`, `reports/forecasting/part3_v2_posthoc_holdout_evaluation.md`, `part3_v2_lifecycle_report.md`, `part3_b_cohort_error_analysis.md`, `part3_final_holdout_blocker.md`, `reports/experiments/06_forecasting_controlled_experiments_report.md`, `PROJECT_BLUEPRINT.md`, `PROJECT_MAP.md`.