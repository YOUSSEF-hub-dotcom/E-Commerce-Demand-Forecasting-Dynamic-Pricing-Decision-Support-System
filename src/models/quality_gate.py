"""Part-3 automated quality gate for the frozen demand-forecasting candidate.

The gate is a *multi-condition* decision rule: the frozen candidate is promoted
to ``Production`` only if **every** criterion passes. It is deliberately NOT a
single ``WAPE < 0.4`` check; it combines:

* **global accuracy** (WAPE / MAE / RMSE / MASE),
* **business risk** (cohort-level bias on B/B2 sustained demand — the driver
  for the whole experiment cycle),
* **long-tail safety** (C cohort and positive-demand error),
* **technical integrity** (schema, serialization, PyFunc predict, signatures,
  artifacts, and the regression suite).

Every threshold below is **documented and justified** against actual measured
references from the persisted Part-3 winner / controlled experiments:

===========  Metric   ===========         reference (winner, val)   EXP2 (val)
==========   ========= ==========         ========================   =================
WAPE                                     0.4001                      0.3964
MAE                                      0.0235                      0.0233
RMSE                                     0.2725                      0.2592
MASE                                     0.6179                      0.6122
global bias (%)                          -9.6                        -6.15
asym 2:1                                 0.0381                      0.0367
pos-demand WAPE                          0.3721                      0.3655
B cohort WAPE                            0.5485                      0.5102
B cohort bias (%)                        -30.0 (defect reported)     -18.2
B2 cohort WAPE                           0.5624                      0.5557
B2 cohort bias (%)                       -28.0 (defect reported)     -18.1
C long-tail WAPE                         0.3807                      0.3796

Tolerance policy: global metrics allow a small band above the *strictly better*
EXP2 candidate (the frozen champion) to absorb test-split variance while still
rejecting any material regression vs. the persisted production winner. Cohort
bias thresholds encode the *business requirement* that emerged from the defect
analysis: sustained high-volume demand must not be systematically under-
forecast by more than -20%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Policy versioning
# ---------------------------------------------------------------------------
# Two formally versioned gate policies share the same documented threshold
# VALUES (nothing is re-tuned); they differ only in the ROLE of the B_wape
# metric:
#
# * ``quality_gate_policy_v1_legacy`` - B_wape is a hard Production blocker
#   (bound 0.60). This is the historical default used by the V1/V2 lifecycle
#   and the post-hoc holdout evaluation. ``evaluate_gate`` defaults to it so
#   every existing caller keeps the exact legacy behavior (19 outcomes).
# * ``quality_gate_policy_v2_revised`` - B_wape is reclassified to a
#   diagnostic / monitoring metric (non-blocking; still reported). This is the
#   decision taken after the provenance audit
#   ``reports/forecasting/part3_v2_gate_production_policy.*`` (B_wape <= 0.60
#   is an engineering reference-anchored regression guard, not a documented
#   business requirement). The hard B-business constraint (B_bias_pct >= -20,
#   MUST NOT systematically under-forecast) and all other criteria are
#   unchanged and remain blocking.
#
POLICY_LEGACY = "quality_gate_policy_v1_legacy"
POLICY_REVISED = "quality_gate_policy_v2_revised"

# Metrics that are REPORTED but NON-BLOCKING under the revised policy.
# ``key -> role`` mapping (currently a single reclassification).
DIAGNOSTIC_METRICS: dict[str, str] = {
    "B_wape": "diagnostic",
}

# ---------------------------------------------------------------------------
# Threshold table (machine-readable, self-documenting)
# ---------------------------------------------------------------------------

# Numeric criteria: key -> (unit, op, bound, label, reference, justification)
_NUMERIC_CRITERIA: list[dict[str, Any]] = [
    dict(
        key="wape",
        description="Global WAPE (primary quality metric, unit-independent)",
        op="le",
        bound=0.4100,
        reference="0.4001 (persisted winner) / 0.3964 (EXP2)",
        justification=(
            "Frozen champion EXP2 = 0.3964 on val; winner reference 0.4001. "
            "A -2.5% band above reference (=0.41) rejects any regression worse "
            "than ~2.4% vs. the persisted production quality bar while absorbing "
            "test-split variance."
        ),
    ),
    dict(
        key="mae",
        description="Global MAE (unit sales) - interpretable absolute error",
        op="le",
        bound=0.0250,
        reference="0.0235 (winner) / 0.0233 (EXP2)",
        justification=(
            "Reference MAE 0.0233-0.0235 units; 0.025 allows ~7% headroom for "
            "test-period noise yet rejects any accuracy collapse."
        ),
    ),
    dict(
        key="rmse",
        description="Global RMSE (heavy-tail penalty)",
        op="le",
        bound=0.3000,
        reference="0.2725 (winner) / 0.2592 (EXP2)",
        justification=(
            "Reference RMSE 0.26-0.27; budget of 0.30 is ~+10% to keep outliers "
            "in check without over-fitting the threshold to one val snapshot."
        ),
    ),
    dict(
        key="mase",
        description="Global MASE (vs train 1-step naive denominator 0.03804)",
        op="le",
        bound=0.6400,
        reference="0.6179 (winner) / 0.6122 (EXP2)",
        justification=(
            "Reference MASE ~0.61-0.62; 0.64 leaves ~3-4% band while preserving "
            "the 'well below naive (=1)' quality demonstration."
        ),
    ),
    dict(
        key="bias_pct",
        description="Global bias (forecast/actual - 1) in %",
        op="within",
        bound=(-12.0, 12.0),
        reference="-9.6% (winner) / -6.15% (EXP2)",
        justification=(
            "Reference bias -6..-9.6% (slight under-forecast). A symmetrical "
            "±12% band bounds systematic under/over-forecasting - the business "
            "impact asymmetry is then handled by the 2:1 asym loss below."
        ),
    ),
    dict(
        key="asym21",
        description="Asymmetric loss 2:1 (under-forecast penalised double)",
        op="le",
        bound=0.0400,
        reference="0.0381 (winner) / 0.0367 (EXP2)",
        justification=(
            "Reference 0.037-0.038 cheap units; 0.04 is +~8% headroom; stricter "
            "than the raw pre-experiment score would already fail, guarding the "
            "stockout-cost asymmetry the metric encodes."
        ),
    ),
    dict(
        key="pos_wape",
        description="WAPE restricted to positive-demand rows (operational sellable demand)",
        op="le",
        bound=0.4200,
        reference="0.3721 (winner) / 0.3655 (EXP2)",
        justification=(
            "Positive-demand error budget 0.42 vs reference 0.365-0.372; +13% "
            "band keeps sellable-demand accuracy near the champion."
        ),
    ),
    dict(
        key="B_wape",
        description="B cohort WAPE (sustained high-volume, the business-critical segment)",
        op="le",
        bound=0.6000,
        reference="0.5485 (winner) / 0.5102 (EXP2)",
        justification=(
            "B cohort is the monetarily dominant segment. Reference 0.51-0.55; "
            "0.60 bound rejects >~10% regression on high-volume products."
        ),
    ),
    dict(
        key="B_bias_pct",
        description="B cohort bias (%) - MUST NOT systematically under-forecast",
        op="ge",
        bound=-20.0,
        reference="-30.0% (winner defect) / -18.2% (EXP2 fix)",
        justification=(
            "Key business requirement distilled from the defect analysis: the "
            "production winner under-forecast B by -30%; EXP2 repaired it to "
            "-18.2%. The gate mandates >= -20% - i.e. the candidate MUST hold "
            "the fix and may not slip back towards the -30% stockout regime."
        ),
    ),
    dict(
        key="B2_wape",
        description="B2 cohort WAPE (upper-mid demand)",
        op="le",
        bound=0.6500,
        reference="0.5624 (winner) / 0.5557 (EXP2)",
        justification=(
            "B2 reference 0.56; 0.65 band = +16% regression guard for this "
            "secondary near-mature cohort."
        ),
    ),
    dict(
        key="B2_bias_pct",
        description="B2 cohort bias (%)",
        op="ge",
        bound=-20.0,
        reference="-28.0% (winner defect) / -18.1% (EXP2 fix)",
        justification=(
            "Same stockout-risk requirement as B: the -28% production defect "
            "must stay fixed (>= -20%), matching the EXP2 result of -18.1%."
        ),
    ),
    dict(
        key="C_wape",
        description="C long-tail cohort WAPE (sparse, many zero-weeks)",
        op="le",
        bound=0.4300,
        reference="0.3807 (winner) / 0.3796 (EXP2)",
        justification=(
            "Long tail (majority of 32.7k SKUs) must not be sacrificed for "
            "segment gains: 0.43 = +13% band over reference 0.38."
        ),
    ),
]

# Technical-integrity (boolean) criteria.
_TECHNICAL_CRITERIA: list[dict[str, str]] = [
    dict(
        key="pred_integrity",
        description="Predictions finite, non-negative, correct length",
        justification="Rejects NaN/Inf/negative/shape-broken outputs before any scoring is trusted.",
    ),
    dict(
        key="schema_ok",
        description="Model input schema == frozen 93-column signature (exact order)",
        justification="Schema drift silently corrupts inference; the gate enforces the frozen contract.",
    ),
    dict(
        key="model_load_ok",
        description="Model artifact loads from MLflow",
        justification="A logged model that cannot reload is not deployable regardless of metrics.",
    ),
    dict(
        key="pyfunc_predict_ok",
        description="PyFunc predict returns pred_demand on unit scale",
        justification="End-to-end MLflow PyFunc contract must hold for the pipeline to be served.",
    ),
    dict(
        key="signature_ok",
        description="MLflow model signature recorded and valid",
        justification="Signature drives downstream serving; its absence breaks the registry contract.",
    ),
    dict(
        key="artifacts_ok",
        description="Model + feature_columns.json + config.json artifacts present in run",
        justification="Reproducibility requires the schema and freeze config shipped with the model.",
    ),
    dict(
        key="tests_ok",
        description="Regression suite green (Phase-G gate)",
        justification="The candidate must not corrupt the project's existing contracts.",
    ),
]


@dataclass(frozen=True)
class CriterionOutcome:
    """Outcome of a single gate criterion."""

    key: str
    description: str
    op: str
    bound: Any
    observed: Any
    passed: bool
    reference: str = ""
    justification: str = ""
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityGateReport:
    """Machine + human readable gate outcome."""

    passed: bool
    outcomes: list[CriterionOutcome] = field(default_factory=list)
    promoted_stage: str = ""
    policy: str = POLICY_LEGACY
    diagnostics: list[CriterionOutcome] = field(default_factory=list)

    def failed_criteria(self) -> list[CriterionOutcome]:
        return [o for o in self.outcomes if not o.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "passed": self.passed,
            "promoted_stage": self.promoted_stage,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "diagnostics": [o.to_dict() for o in self.diagnostics],
        }

    def to_markdown(self) -> str:
        lines = ["| Criterion | Observed | Bound | PASS/FAIL |", "|---|---|---|---|"]
        for o in self.outcomes:
            raw = o.observed if not isinstance(o.observed, str) else f"`{o.observed}`"
            lines.append(f"| {o.key} | {raw} | {o.bound} | {'PASS' if o.passed else 'FAIL'} |")
        lines.append("")
        lines.append(f"**Gate: {'PASS' if self.passed else 'FAIL'} → promote to {self.promoted_stage}**")
        return "\n".join(lines)


def _apply(op: str, bound, observed) -> bool:
    if observed is None or (isinstance(observed, float) and np.isnan(observed)):
        return False
    if op == "le":
        return bool(observed <= bound)
    if op == "ge":
        return bool(observed >= bound)
    if op == "within":
        lo, hi = bound
        return bool(lo <= observed <= hi)
    if op == "bool":
        return bool(observed)
    raise ValueError(f"quality_gate: unknown op '{op}'")


def evaluate_gate(
    metrics: dict[str, Any],
    technical: dict[str, bool],
    group: str = "final_test",
    policy: str = POLICY_LEGACY,
) -> QualityGateReport:
    """Evaluate the full multi-condition gate on the *final test* metrics.

    Parameters
    ----------
    metrics : dict
        ``full_metric_suite`` output for the scored split.
    technical : dict
        Boolean integrity flags keyed by the technical criterion names.
    group : str
        Label recorded in outcomes (e.g. ``final_test``).
    policy : str
        Which documented policy to apply:

        * ``POLICY_LEGACY`` (default) - all 12 numeric criteria are hard
          Production blockers (B_wape included; 19 outcomes).
        * ``POLICY_REVISED`` - the diagnostic metrics reported by
          ``DIAGNOSTIC_METRICS`` (B_wape) are non-blocking and are returned in
          ``report.diagnostics`` instead of ``report.outcomes``; every other
          criterion (including the hard B_bias_pct >= -20 business constraint)
          remains blocking (18 outcomes).

    Returns
    -------
    QualityGateReport
        Decision structure; ``passed`` is True only if *all* hard criteria
        pass under the selected policy.
    """
    if policy not in (POLICY_LEGACY, POLICY_REVISED):
        raise ValueError(f"quality_gate: unknown policy '{policy}'")

    outcomes: list[CriterionOutcome] = []
    diagnostics: list[CriterionOutcome] = []

    for crit in _NUMERIC_CRITERIA:
        key = crit["key"]
        observed = metrics.get(key)
        if policy == POLICY_REVISED and key in DIAGNOSTIC_METRICS:
            diagnostics.append(
                CriterionOutcome(
                    key=key,
                    description=crit["description"],
                    op="diagnostic",
                    bound="diagnostic",
                    observed=round(float(observed), 6) if isinstance(observed, (int, float)) else observed,
                    passed=True,
                    reference=crit["reference"],
                    justification=(
                        f"Diagnostic (non-blocking) under `{POLICY_REVISED}`: "
                        f"{crit['justification']} Role `{DIAGNOSTIC_METRICS[key]}` - "
                        "monitoring only; no longer a hard Production blocker."
                    ),
                    group=group,
                )
            )
            continue
        okay = _apply(crit["op"], crit["bound"], observed)
        outcomes.append(
            CriterionOutcome(
                key=key,
                description=crit["description"],
                op=crit["op"],
                bound=crit["bound"],
                observed=round(float(observed), 6) if isinstance(observed, (int, float)) else observed,
                passed=okay,
                reference=crit["reference"],
                justification=crit["justification"],
                group=group,
            )
        )

    for crit in _TECHNICAL_CRITERIA:
        observed = technical.get(crit["key"], False)
        outcomes.append(
            CriterionOutcome(
                key=crit["key"],
                description=crit["description"],
                op="bool",
                bound=True,
                observed=bool(observed),
                passed=bool(observed),
                reference="required",
                justification=crit["justification"],
                group=group,
            )
        )

    passed = all(o.passed for o in outcomes)
    report = QualityGateReport(
        passed=passed,
        outcomes=outcomes,
        policy=policy,
        diagnostics=diagnostics,
    )
    report.promoted_stage = "Production" if passed else "Staging"
    return report


GATE_DOCUMENTATION = {
    "policy": (
        "Tolerance bands are set relative to the frozen EXP2 champion (val) and "
        "the persisted Part-3 winner reference. Global error bounds allow "
        "test-split variance without tolerating regression; cohort-bias bounds "
        "encode the business stockout-risk requirement (>= -20% on B/B2). "
        "Technical integrity is enforced as hard booleans. Threshold VALUES "
        "are versioned policy inputs: the revised policy only reclassifies "
        "B_wape to a diagnostic/monitoring role; no bound is re-tuned."
    ),
    "policies": {
        POLICY_LEGACY: {
            "description": "Historical default. All 12 numeric criteria (incl. "
            "B_wape <= 0.60) are hard Production blockers; 19 outcomes.",
            "blocking_numeric": [c["key"] for c in _NUMERIC_CRITERIA],
            "diagnostic": {},
        },
        POLICY_REVISED: {
            "description": "Formal revision after the provenance audit. B_wape "
            "is reported but non-blocking; the hard business constraint "
            "B_bias_pct >= -20 and every other criterion remain blockers.",
            "blocking_numeric": [
                c["key"] for c in _NUMERIC_CRITERIA if c["key"] not in DIAGNOSTIC_METRICS
            ],
            "diagnostic": dict(DIAGNOSTIC_METRICS),
        },
    },
    "thresholds": [
        {k: c[k] for k in ("key", "op", "bound", "reference", "justification")}
        for c in _NUMERIC_CRITERIA
    ],
}


def gate_documentation_markdown() -> str:
    """Human-readable threshold documentation (machine line + table)."""
    lines = ["# Quality Gate Thresholds (Phase D)", ""]
    lines.append(GATE_DOCUMENTATION["policy"])
    lines.append("")
    lines.append("| Criterion | Description | op | Bound | Reference (winner→EXP2) | Justification |")
    lines.append("|---|---|---|---|---|---|")
    for c in _NUMERIC_CRITERIA:
        lines.append(
            f"| {c['key']} | {c['description']} | {c['op']} | {c['bound']} "
            f"| {c['reference']} | {c['justification']} |"
        )
    lines.append("")
    lines.append("Technical-integrity booleans (all must be True):")
    for c in _TECHNICAL_CRITERIA:
        lines.append(f"* `{c['key']}` - {c['description']}.")
    lines.append("")
    lines.append("## Policy versioning")
    lines.append("")
    for policy, meta in GATE_DOCUMENTATION["policies"].items():
        lines.append(f"* **`{policy}`** - {meta['description']}")
        lines.append(
            f"  * Blocking numeric criteria ({len(meta['blocking_numeric'])}): "
            f"{', '.join(f'`{k}`' for k in meta['blocking_numeric'])}."
        )
    lines.append(
        f"  * Diagnostic (reported, non-blocking): "
        f"{', '.join(f'`{k}` ({r})' for k, r in DIAGNOSTIC_METRICS.items()) or 'none'}."
    )
    return "\n".join(lines)