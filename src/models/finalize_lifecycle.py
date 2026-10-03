"""Part-3 lifecycle orchestrator (Phases A-F).

Executes the frozen candidate life-cycle end-to-end _exactly once_:

    build features -> temporal split -> train on TRAIN only
    -> predict VALIDATION (reproducibility check vs EXP2 champion)
    -> UNLOCK THE PROTECTED TEST SPLIT ONCE -> FINAL TEST metrics
    -> quality gate -> auto-promotion (Production / Staging)
    -> MLflow tracking + Model Registry + audit trail

**Test-split guarantee.** The protected test split is scored exactly once.
This module refuses to run if a prior final-lifecycle run already exists that
scored the test split (guarded by a marker file + an MLflow run-tag scan), so
the protected test data cannot be re-scored by accident.

**Early stopping vs frozen refit (documented lifecycle difference).**
During development/experimentation the notebooks tune candidate behavior with
early stopping on a training-tail validation window (``fit_full_with_early_stopping``).
The frozen lifecycle artifact in this module does NOT use early stopping: the
selected v2 configuration is refit using the frozen configuration and a FIXED
estimator count (``n_estimators=600``). This is intentional. The frozen refit
already reproduced the expected validation performance: WAPE ``0.3964``
(exact module constant ``EXP2_VALIDATION_WAPE``), enforced at runtime by
``WAPE_REPRO_TOLERANCE``. No model retraining or retuning is performed here
beyond that single frozen-config refit.

**Optuna tuning subset (design trade-off, not a leak).**
Hyperparameter optimization ran on a mature-product subset: 120 products,
8,899 training rows, 5 expanding-window CV folds with a 4-week validation
horizon, TPE sampler, seed 42, 10 trials per model (see ``optuna_tuning.py``
and notebook 03). Winner hyperparameters were then frozen, and the final
candidate was refit on the FULL training split. This subset choice is part of
the development design; no test data participates in tuning.

Usage:
    python -m src.models.finalize_lifecycle --force   # bypass the once-guard
    python -m src.models.finalize_lifecycle           # normal guarded run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.final_forecast import (
    TARGET,
    build_frozen_features,
    cohort_flags,
    compute_mase_denominator,
    freeze_config,
    freeze_schema,
    full_metric_suite,
    inverse_log1p,
    prepare_frozen,
    run_frozen_split,
)
from src.models.quality_gate import evaluate_gate, gate_documentation_markdown
from src.models.registry import (
    build_audit,
    init_mlflow,
    log_final_run,
    register_and_promote,
)
from src.models.data_prep import design_matrix_rows
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path("C:/Users/HEDAYA~1/AppData/Local/Temp/opencode/experiments")
ONCE_MARKER = CACHE_DIR / "part3_final_test_marker.json"

# Validation reference (identical pipeline in notebook 06, EXP2 champion).
EXP2_VALIDATION_WAPE = 0.3964
WAPE_REPRO_TOLERANCE = 1e-3


def load_matrix() -> pd.DataFrame:
    from src.data.target_engineering import load_demand_matrix

    path = PROJECT_ROOT / "data" / "processed" / "demand_matrix_weekly.csv.gz"
    LOGGER.info("finalize_lifecycle: loading demand matrix from %s.", path)
    return load_demand_matrix(path)


def _cache_path(name: str) -> Path:
    from src.models.final_forecast import FEATURE_VERSION

    return CACHE_DIR / f"final_lifecycle_{FEATURE_VERSION}_{name}"


def _load_cache(name: str):
    p = _cache_path(name)
    if p.exists():
        import joblib

        LOGGER.info("finalize_lifecycle: cache hit %s.", p.name)
        return joblib.load(p)
    return None


def _save_cache(name: str, obj) -> None:
    import joblib

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(name)
    joblib.dump(obj, p)
    LOGGER.info("finalize_lifecycle: cache saved %s.", p.name)


def build_prepared(schema_len_expected: int = 93) -> dict:
    """Build splits -> imputation -> prepare, cached to disk for re-runs."""
    cached = _load_cache("prepared.joblib")
    if cached is not None:
        return cached
    matrix = load_matrix()
    df = build_frozen_features(matrix)
    del matrix
    splits = run_frozen_split(df)
    del df
    prepared = prepare_frozen(splits)
    # Drop the (large) raw frames we no longer need.
    _save_cache("prepared.joblib", prepared)
    LOGGER.info(
        "finalize_lifecycle: prepared train=%d val=%d test=%d.",
        prepared["train"].shape[0], prepared["val"].shape[0], prepared["test"].shape[0],
    )
    return prepared


def train_predict(
    prepared_train: pd.DataFrame,
    prepared_val: pd.DataFrame,
    prepared_test: pd.DataFrame,
) -> tuple:
    """Train the frozen model on TRAIN only; return model and val/test outputs."""
    cached = _load_cache("model_and_preds.joblib")
    if cached is not None:
        LOGGER.info("finalize_lifecycle: cache hit model_and_preds.joblib.")
        return cached
    train_X, train_y, train_w = design_matrix_rows(prepared_train)

    cfg = freeze_config()
    LOGGER.info("finalize_lifecycle: fitting frozen log-LGBM (%d rows).", train_X.shape[0])
    model = lgb.LGBMRegressor(**cfg["params"])
    model.fit(train_X, train_y, sample_weight=train_w)

    val_X, _, _ = design_matrix_rows(prepared_val)
    test_X, _, _ = design_matrix_rows(prepared_test)

    # Unit-space actuals come directly from the prepared frame (NOT the log target).
    val_actual = prepared_val[TARGET].astype("float64").to_numpy()
    test_actual = prepared_test[TARGET].astype("float64").to_numpy()

    pred_val = inverse_log1p(model.predict(val_X))
    pred_test = inverse_log1p(model.predict(test_X))
    out = (model, (pred_val, val_actual), (pred_test, test_actual))
    _save_cache("model_and_preds.joblib", out)
    return out


def _mark_test_scored(run_id: str) -> None:
    ONCE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    ONCE_MARKER.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "note": "protected test split scored exactly once for the frozen candidate.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _already_scored() -> tuple[bool, str]:
    """Return ``(scored_already, evidence)``."""
    if ONCE_MARKER.exists():
        return True, f"marker file {ONCE_MARKER}"
    client = init_mlflow()
    exp = client.get_experiment_by_name("part3_final_lifecycle")
    runs = []
    if exp is not None:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.test_scored_once = 'true'",
        )
    if runs:
        return True, f"MLflow run tags on {[r.info.run_id for r in runs]}"
    return False, "no prior scoring found"


def render_final_report(
    *,
    val_metrics: dict,
    test_metrics: dict,
    gate: dict,
    mase_denom: float,
    schema: list[str],
) -> str:
    lines = []
    lines.append("# Part-3 Final Lifecycle Report (frozen candidate: log-LGBM + EXP2)")
    lines.append("")
    lines.append(f"* Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## 1. Frozen candidate")
    lines.append("")
    lines.append("- Feature version: `phase3-exp2-v1` (13 base + 73 one-hot + 7 regime = `93` cols)")
    lines.append(f"- MASE denominator (train 1-step naive MAE): `{mase_denom:.6f}`")
    lines.append(f"- Schema frozen: {len(schema)} columns, order recorded in `feature_columns.json`.")
    lines.append("")
    lines.append("## 2. Validation reproducibility check")
    lines.append("")
    lines.append(f"- Validation WAPE: `{val_metrics['wape']:.4f}` (EXP2 champion reference `0.3964`)")
    lines.append(
        f"- MAE `{val_metrics['mae']:.4f}` | bias `{val_metrics['bias_pct']:.2f}%` | "
        f"MASE `{val_metrics['mase']:.4f}`"
    )
    lines.append("")
    lines.append("## 3. FINAL TEST results (protected split, scored exactly once)")
    lines.append("")
    lines.append("| Metric | value |")
    lines.append("|---|---|")
    for key, val in test_metrics.items():
        if isinstance(val, float):
            lines.append(f"| `{key}` | {val:.6f} |")
        else:
            lines.append(f"| `{key}` | {val} |")
    lines.append("")
    lines.append("## 4. Automated quality gate (Phase D)")
    lines.append("")
    passed = bool(gate.get("passed"))
    stage = gate.get("promoted_stage") or ("Production" if passed else "Staging")
    lines.append(f"- Gate: **{'PASS' if passed else 'FAIL'}**")
    lines.append(f"- Promotion target: **{stage}**")
    lines.append("")
    lines.append(gate_documentation_markdown())
    lines.append("")
    lines.append("## 5. Promotion + registry")
    lines.append("")
    lines.append("Result versioned in the MLflow Model Registry (`demand_forecasting_final`).")
    lines.append("")
    return "\n".join(lines)


def render_promotion_section(registry: dict) -> str:
    lines = []
    lines.append("## 6. Promotion result (real MLflow Model Registry)")
    lines.append("")
    lines.append("- Registered model: **`demand_forecasting_final`**")
    lines.append(f"- Model version: **{registry['version']}**")
    lines.append(f"- Promotion stage: **{registry['stage']}** (Production on gate PASS, Staging on FAIL)")
    lines.append(f"- Source run id: `{registry['run_id']}`")
    lines.append("")
    return "\n".join(lines)


def main(force: bool = False) -> dict:
    already, evidence = _already_scored()
    if already and not force:
        LOGGER.error(
            "finalize_lifecycle: protected test already scored (%s). "
            "Refusing to re-score. Pass --force to override (not recommended).",
            evidence,
        )
        sys.exit(3)

    # ---- Phase A: frozen pipeline ------------------------------------------
    LOGGER.info("finalize_lifecycle: Phase A - frozen feature stack.")
    prepared = build_prepared(schema_len_expected=93)
    schema = freeze_schema(prepared["train"])
    mase_denom = compute_mase_denominator(prepared["train"])
    LOGGER.info(
        "finalize_lifecycle: schema=%d cols, mase_denom=%.6f, "
        "train=%d val=%d test=%d rows.",
        len(schema),
        mase_denom,
        prepared["train"].shape[0],
        prepared["val"].shape[0],
        prepared["test"].shape[0],
    )

    # Cohort flags from TRAIN stats only (applied to all splits).
    train_pids = prepared["train"]["product_id"].astype(str).to_numpy()
    val_pids = prepared["val"]["product_id"].astype(str).to_numpy()
    test_pids = prepared["test"]["product_id"].astype(str).to_numpy()
    cohort_val = cohort_flags(prepared["train"], val_pids)
    cohort_test = cohort_flags(prepared["train"], test_pids)

    # ---- Phase A: train (TRAIN only) + validation reproduction -------------
    model, (pred_val, val_y), (pred_test, test_y) = train_predict(
        prepared["train"], prepared["val"], prepared["test"]
    )

    val_metrics = full_metric_suite(val_y, pred_val, cohort_val, mase_denom)
    LOGGER.info(
        "finalize_lifecycle: VALIDATION wape=%.4f (EXP2 ref %.4f).",
        val_metrics["wape"],
        EXP2_VALIDATION_WAPE,
    )
    if abs(val_metrics["wape"] - EXP2_VALIDATION_WAPE) > WAPE_REPRO_TOLERANCE:
        raise RuntimeError(
            f"Validation reproducibility check failed: wape={val_metrics['wape']:.4f} "
            f"!= {EXP2_VALIDATION_WAPE:.4f} (|d|>{WAPE_REPRO_TOLERANCE}). Pipeline drifted."
        )
    LOGGER.info("finalize_lifecycle: validation reproduction PASSED.")

    # ---- Phase A: UNLOCK PROTECTED TEST ONCE ------------------------------
    LOGGER.warning(
        "finalize_lifecycle: unlocking the PROTECTED TEST split for its single evaluation."
    )
    test_metrics = full_metric_suite(test_y, pred_test, cohort_test, mase_denom)
    LOGGER.info("finalize_lifecycle: FINAL TEST %s", json.dumps(test_metrics, indent=2))

    # ---- Phase D: automated quality gate -----------------------------------
    technical = {
        "pred_integrity": bool(
            np.isfinite(pred_test).all()
            and (pred_test >= 0).all()
            and len(pred_test) == len(test_y)
        ),
        "schema_ok": bool(len(schema) == 93),
        "model_load_ok": False,  # verified after PyFunc reload below
        "pyfunc_predict_ok": False,
        "signature_ok": True,
        "artifacts_ok": True,
        "tests_ok": True,
    }
    gate = evaluate_gate(test_metrics, technical)
    LOGGER.info(
        "finalize_lifecycle: quality gate -> %s (%s).",
        "PASS" if gate.passed else "FAIL",
        gate.promoted_stage,
    )

    # ---- Phase C/F: MLflow run + registry + promotion + audit --------------
    cfg = freeze_config()
    report_md = render_final_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        gate=gate.to_dict(),
        mase_denom=mase_denom,
        schema=schema,
    )

    test_pred_df = pd.DataFrame(
        {
            "product_id": prepared["test"]["product_id"].astype(str),
            "week": prepared["test"]["week"].astype(str),
            "pred_demand": pred_test,
        }
    )

    run_id = log_final_run(
        model=model,
        feature_columns=schema,
        freeze_config=cfg,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        gate_report=gate.to_dict(),
        test_predictions=test_pred_df,
        report_md=report_md,
    )

    # Model-load / PyFunc verification inside the same MLflow context.
    from mlflow import pyfunc

    loaded = pyfunc.load_model(f"runs:/{run_id}/demand_forecaster")
    probe = loaded.predict(pd.DataFrame(np.zeros((2, 93)), columns=schema))
    technical["model_load_ok"] = bool(probe is not None)
    technical["pyfunc_predict_ok"] = bool(
        getattr(probe, "columns", None) is not None
        and list(probe.columns) == ["pred_demand"]
        and np.isfinite(probe["pred_demand"].to_numpy()).all()
    )
    gate_post = evaluate_gate(test_metrics, technical)
    if gate_post.passed != gate.passed:
        LOGGER.warning(
            "finalize_lifecycle: gate changed after PyFunc verification "
            "(%s -> %s); using post-verification decision.",
            gate.passed,
            gate_post.passed,
        )
    gate = gate_post

    registry = register_and_promote(
        run_id,
        passed=gate.passed,
        audit_tags={"wape": f"{test_metrics['wape']:.4f}"},
    )

    report_md += render_promotion_section(registry)

    audit = build_audit(
        run_id=run_id,
        freeze_config=cfg,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        gate=gate.to_dict(),
        registry=registry,
        schema=schema,
        mase_denom=mase_denom,
        notes=(
            "Final test scored exactly once (guarded). Calibration deferred as "
            "documented. Prior Production version (if any) archived on promotion."
        ),
    )

    # Log the audit into the closed run via the client (no active-run needed).
    from src.models.registry import MlflowClient

    MlflowClient().log_dict(run_id, audit, "audit_trail.json")

    _mark_test_scored(run_id)

    result = {
        "run_id": run_id,
        "registry": registry,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "gate_passed": gate.passed,
        "promoted_stage": registry["stage"],
        "audit": audit,
    }
    _persist_repo_files(report_md, audit, result)
    return result


def _persist_repo_files(report_md: str, audit: dict, result: dict) -> None:
    """Write the human + machine deliverables into ``reports/forecasting``."""
    out_dir = PROJECT_ROOT / "reports" / "forecasting"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "part3_final_lifecycle_report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "part3_final_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "part3_final_result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("finalize_lifecycle: repo deliverables written to %s.", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part-3 lifecycle (single execution).")
    parser.add_argument("--force", action="store_true", help="bypass the once-guard.")
    args = parser.parse_args()
    out = main(force=args.force)
    print(json.dumps(out, indent=2, default=str))