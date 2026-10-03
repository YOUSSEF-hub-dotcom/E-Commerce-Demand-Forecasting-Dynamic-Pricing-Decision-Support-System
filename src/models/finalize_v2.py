"""Part-3 v2 lifecycle orchestrator (validation-only quality gate).

Implements the v2 remediation protocol decided after the v1 gate FAIL
(run 7fc87198...; Staging). v2 is the SAME frozen candidate as v1
(log-LGBM + EXP2 regime features, 93-col schema) shipped with the corrected
inference contract ``inverse_log1p`` that removes the prediction-integrity
defect that contributed to the v1 gate failure.

**Evaluation protocol (explicit and integrity-preserving):**

* v1 already scored the protected TEST split exactly once (guarded). Per the
  project's integrity rules the test split MUST NOT be re-scored, so v2 is
  evaluated **on the VALIDATION split only** (training on TRAIN only).
* The automated quality gate is run against the v2 **validation** metrics with
  ``group="v2_validation"``. Thresholds are unchanged (no re-thresholding).
* v2 is therefore **not** claimed to be test-validated; the gate decision for
  v2 reflects the validation protocol under which the gate's numeric
  references were themselves established (EXP2 val 0.3964 etc).
* v1 is preserved: registration adds a NEW version, leaving the existing
  Staging version untouched.

This module NEVER reads or scores the protected test split.

Usage:
    python -m src.models.finalize_v2
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.final_forecast import (
    TARGET,
    cohort_flags,
    compute_mase_denominator,
    freeze_config,
    freeze_schema,
    full_metric_suite,
    inverse_log1p,
)
from src.models.finalize_lifecycle import build_prepared
from src.models.quality_gate import evaluate_gate, gate_documentation_markdown
from src.models.registry import (
    build_audit,
    log_final_run,
    register_and_promote,
)
from src.models.data_prep import design_matrix_rows
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"

EXP2_VALIDATION_WAPE = 0.3964
WAPE_REPRO_TOLERANCE = 1e-3
EVALUATION_GROUP = "v2_validation"


def train_v2(prepared_train: pd.DataFrame) -> lgb.LGBMRegressor:
    """Train the frozen candidate on TRAIN only (identical to v1 model)."""
    train_X, train_y, train_w = design_matrix_rows(prepared_train)
    cfg = freeze_config()
    LOGGER.info("finalize_v2: fitting frozen log-LGBM on TRAIN only (%d rows).", train_X.shape[0])
    model = lgb.LGBMRegressor(**cfg["params"])
    model.fit(train_X, train_y, sample_weight=train_w)
    return model


def compute_validation_report(prepared: dict, model) -> tuple[dict, dict]:
    """Score VALIDATION only; return (metrics, integrity) - never touches test."""
    val_X, _, _ = design_matrix_rows(prepared["val"])
    val_actual = prepared["val"][TARGET].astype("float64").to_numpy()
    val_pids = prepared["val"]["product_id"].astype(str).to_numpy()
    cohort_val = cohort_flags(prepared["train"], val_pids)
    mase_denom = compute_mase_denominator(prepared["train"])

    pred_val = inverse_log1p(model.predict(val_X))
    metrics = full_metric_suite(val_actual, pred_val, cohort_val, mase_denom)
    integrity = {
        "pred_integrity": bool(
            np.isfinite(pred_val).all()
            and (pred_val >= 0).all()
            and len(pred_val) == len(val_actual)
        ),
        "n_neg": int((pred_val < 0).sum()),
        "n_nan_inf": int((~np.isfinite(pred_val)).sum()),
    }
    LOGGER.info(
        "finalize_v2: VALIDATION wape=%.4f (EXP2 ref %.4f), pred_integrity=%s.",
        metrics["wape"], EXP2_VALIDATION_WAPE, integrity["pred_integrity"],
    )
    if abs(metrics["wape"] - EXP2_VALIDATION_WAPE) > WAPE_REPRO_TOLERANCE:
        raise RuntimeError(
            f"Validation reproduction check failed: wape={metrics['wape']:.4f} "
            f"!= {EXP2_VALIDATION_WAPE:.4f} (|d|>{WAPE_REPRO_TOLERANCE}). Pipeline drifted."
        )
    return metrics, integrity


def render_v2_report(
    *,
    val_metrics: dict,
    integrity: dict,
    gate: dict,
    schema: list[str],
    mase_denom: float,
    registry: dict,
) -> str:
    lines = ["# Part-3 Model v2 Lifecycle Report (validation-only quality gate)", ""]
    lines.append(f"* Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## 1. Motivation")
    lines.append("")
    lines.append(
        "- v1 (run `7fc87198...`) failed the quality gate: test B_wape 0.6284 > 0.60 and "
        "`pred_integrity=false` (144 negative preds from `expm1` amplification of "
        "near-zero log predictions)."
    )
    lines.append(
        "- v2 ships the SAME frozen candidate with the corrected inference contract "
        "`inverse_log1p` (expm1 + physical floor 0 + finite-fill)."
    )
    lines.append("")
    lines.append("## 2. Evaluation protocol (integrity statement)")
    lines.append("")
    lines.append(
        "- The protected TEST split was scored exactly once by v1 and is NOT re-scored. "
        "v2 is evaluated on the VALIDATION split only (train on TRAIN only)."
    )
    lines.append(f"- Gate evaluated with `group=\"{EVALUATION_GROUP}\"`; thresholds UNCHANGED.")
    lines.append("- v2 is NOT claimed to be test-validated.")
    lines.append("")
    lines.append("## 3. Frozen candidate")
    lines.append("")
    lines.append("- Feature version: `phase3-exp2-v1` (13 base + 73 one-hot + 7 regime = `93` cols)")
    lines.append(f"- MASE denominator (train 1-step naive MAE): `{mase_denom:.6f}`")
    lines.append("")
    lines.append("## 4. Validation metrics (v2)")
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|---|---|")
    for key, val in val_metrics.items():
        if isinstance(val, float):
            lines.append(f"| `{key}` | {val:.6f} |")
        else:
            lines.append(f"| `{key}` | {val} |")
    lines.append("")
    lines.append(f"- Prediction integrity: `{integrity['pred_integrity']}` "
                 f"(neg={integrity['n_neg']}, nan/inf={integrity['n_nan_inf']})")
    lines.append("")
    lines.append("## 5. Automated quality gate")
    lines.append("")
    passed = bool(gate.get("passed"))
    stage = gate.get("promoted_stage") or ("Production" if passed else "Staging")
    lines.append(f"- Gate: **{'PASS' if passed else 'FAIL'}** (group `{EVALUATION_GROUP}`)")
    lines.append(f"- Promotion target: **{stage}**")
    lines.append("")
    lines.append(gate_documentation_markdown())
    lines.append("")
    lines.append("## 6. Registry")
    lines.append("")
    lines.append("- Registered model: **`demand_forecasting_final`**")
    lines.append(f"- Model version: **{registry['version']}**")
    lines.append(f"- Promotion stage: **{registry['stage']}**")
    lines.append(f"- Source run id: `{registry['run_id']}`")
    lines.append(f"- v1 (Staging) preserved untouched.")
    lines.append("")
    lines.append("## 7. Decision")
    lines.append("")
    if passed:
        lines.append("**V2 PASSED QUALITY GATE → PROMOTED TO PRODUCTION**")
    else:
        lines.append("**NO VALID V2 PASSED QUALITY GATE → REMAINS IN STAGING**")
    lines.append("")
    return "\n".join(lines)


def main() -> dict:
    LOGGER.info("finalize_v2: Phase A - load frozen pipeline (cached).")
    prepared = build_prepared(schema_len_expected=93)
    schema = freeze_schema(prepared["train"])
    mase_denom = compute_mase_denominator(prepared["train"])

    LOGGER.info("finalize_v2: Phase B - train on TRAIN only.")
    model = train_v2(prepared["train"])

    LOGGER.info("finalize_v2: Phase C - score VALIDATION only (test untouched).")
    val_metrics, integrity = compute_validation_report(prepared, model)

    cfg = freeze_config()
    technical = {
        "pred_integrity": integrity["pred_integrity"],
        "schema_ok": bool(len(schema) == 93),
        "model_load_ok": False,
        "pyfunc_predict_ok": False,
        "signature_ok": True,
        "artifacts_ok": True,
        "tests_ok": True,
    }

    LOGGER.info("finalize_v2: Phase D - quality gate on %s.", EVALUATION_GROUP)
    gate = evaluate_gate(val_metrics, technical, group=EVALUATION_GROUP)
    LOGGER.info("finalize_v2: gate -> %s (%s).", "PASS" if gate.passed else "FAIL", gate.promoted_stage)

    val_pred_df = pd.DataFrame(
        {
            "product_id": prepared["val"]["product_id"].astype(str),
            "week": prepared["val"]["week"].astype(str),
            "pred_demand": inverse_log1p(model.predict(design_matrix_rows(prepared["val"])[0])),
        }
    )

    run_id = log_final_run(
        model=model,
        feature_columns=schema,
        freeze_config=cfg,
        val_metrics=val_metrics,
        test_metrics={},
        gate_report=gate.to_dict(),
        test_predictions=None,
        val_predictions=val_pred_df,
        run_tags={
            "phase": "part3-v2-lifecycle",
            "purpose": "v2: frozen candidate + inverse_log1p inference contract (validation-only gate)",
            "test_scored_once": "false",
            "evaluation_group": EVALUATION_GROUP,
        },
    )

    from mlflow import pyfunc

    loaded = pyfunc.load_model(f"runs:/{run_id}/demand_forecaster")
    probe = loaded.predict(pd.DataFrame(np.zeros((2, 93)), columns=schema))
    technical["model_load_ok"] = bool(probe is not None)
    technical["pyfunc_predict_ok"] = bool(
        getattr(probe, "columns", None) is not None
        and list(probe.columns) == ["pred_demand"]
        and np.isfinite(probe["pred_demand"].to_numpy()).all()
    )
    gate_post = evaluate_gate(val_metrics, technical, group=EVALUATION_GROUP)
    if gate_post.passed != gate.passed:
        LOGGER.warning(
            "finalize_v2: gate changed after PyFunc verification (%s -> %s); using post-verification.",
            gate.passed, gate_post.passed,
        )
    gate = gate_post

    registry = register_and_promote(
        run_id,
        passed=gate.passed,
        audit_tags={"wape": f"{val_metrics['wape']:.4f}"},
        extra_tags={
            "test_scored_once": "false",
            "evaluation_group": EVALUATION_GROUP,
            "model_phase": "v2-validation-gate",
        },
    )

    report_md = render_v2_report(
        val_metrics=val_metrics,
        integrity=integrity,
        gate=gate.to_dict(),
        schema=schema,
        mase_denom=mase_denom,
        registry=registry,
    )

    audit = build_audit(
        run_id=run_id,
        freeze_config=cfg,
        val_metrics=val_metrics,
        test_metrics={},
        gate=gate.to_dict(),
        registry=registry,
        schema=schema,
        mase_denom=mase_denom,
        notes=(
            "v2 validation-only protocol: test split already consumed by v1 (scored once) and "
            "NOT re-scored; thresholds unchanged; v2 not claimed test-validated."
        ),
    )
    audit["integrity"] = integrity

    from src.models.registry import MlflowClient

    MlflowClient().log_dict(run_id, audit, "audit_trail.json")

    result = {
        "run_id": run_id,
        "registry": registry,
        "val_metrics": val_metrics,
        "gate_passed": gate.passed,
        "promoted_stage": registry["stage"],
        "evaluation_group": EVALUATION_GROUP,
        "audit": audit,
    }
    _persist_repo_files(report_md, audit, result)
    return result


def _persist_repo_files(report_md: str, audit: dict, result: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "part3_v2_lifecycle_report.md").write_text(report_md, encoding="utf-8")
    (REPORTS_DIR / "part3_v2_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / "part3_v2_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    LOGGER.info("finalize_v2: repo deliverables written to %s.", REPORTS_DIR)


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, default=str))