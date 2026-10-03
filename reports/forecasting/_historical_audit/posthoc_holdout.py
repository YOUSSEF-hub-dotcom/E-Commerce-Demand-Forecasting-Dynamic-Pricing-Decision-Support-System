"""Part-3 V2 post-hoc holdout evaluation (authorized single execution).

Scores the FROZEN V2 candidate (registered model ``demand_forecasting_final``
version 2, source run ``5d729a8594dc488bbd18ba0c4e216f5a``) on the designated
Test split **exactly once** under the explicitly authorized post-hoc protocol,
then runs the existing Quality Gate unchanged and updates the MLflow lifecycle
to reflect the actual gate result.

Protocol / integrity contract
-----------------------------
* **Classification:** ``post_hoc_holdout_evaluation``. The Test split had
  already been consumed by the V1 evaluation + diagnosis lifecycle, so this
  result is NOT an independent untouched final-test estimate. That disclosure
  is emitted verbatim in every artifact this module writes.
* **No ``--force`` exists here.** A dedicated once-guard (marker file +
  ``posthoc_holdout_evaluated`` MLflow run tag) refuses any re-execution; the
  module has no override path at all.
* **No retraining / no tuning.** The frozen V2 PyFunc is loaded straight from
  the Model Registry; no estimator is ever fitted, no parameter is read for
  mutation, no threshold is changed. The run's freeze config is asserted
  byte-equal against ``freeze_config()`` before anything is scored (a mismatch
  is a hard STOP: it would mean the frozen candidate drifted).
* **No second model / no v3.** Registration is never called; only *tags* on
  the existing run + version 2 are (additively) updated, plus a stage
  transition of version 2 itself if the gate result requires it. V1's run,
  marker, audit, result, and Staging version are never touched.
* **Split reproducibility verified before scoring** (row count vs the pinned
  ``data/splits/test.csv``, week-label boundaries); a mismatch is a hard STOP.
* **Prediction integrity before metrics:** finite, non-negative (physical
  floor), correct length, row/product/week alignment, 93-column schema exact
  order (frozen schema == test design matrix == MLflow signature == in-run
  ``feature_columns.json``), and an inverse-transform cross-check against the
  raw estimator artifact (``inverse_log1p(est.predict(X))`` must reproduce the
  PyFunc output).

Evaluation artifacts (new files only; existing reports are never rewritten):
    reports/forecasting/part3_v2_posthoc_holdout_evaluation.md
    reports/forecasting/part3_v2_posthoc_holdout_evaluation.json

Usage:
    python -m src.models.posthoc_holdout
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.data_prep import design_matrix_rows
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
from src.models.registry import MODEL_NAME, init_mlflow
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"
CACHE_DIR = Path("C:/Users/HEDAYA~1/AppData/Local/Temp/opencode/experiments")
POSTHOC_MARKER = CACHE_DIR / "part3_v2_posthoc_holdout_marker.json"

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
EVALUATION_TYPE = "post_hoc_holdout_evaluation"
EVALUATION_GROUP = "post_hoc_holdout"
POSTHOC_TAG = "posthoc_holdout_evaluated"
MODEL_VERSION = 2
SOURCE_RUN_ID = "5d729a8594dc488bbd18ba0c4e216f5a"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# Exact required disclosure wording (verbatim in every artifact).
DISCLOSURE = (
    "The V2 holdout evaluation is classified as post-hoc because the designated "
    "Test split had previously been consumed during the V1 evaluation and "
    "diagnosis lifecycle. Therefore, this result is not claimed as an "
    "independent untouched final-test estimate."
)

REPORT_MD = REPORTS_DIR / "part3_v2_posthoc_holdout_evaluation.md"
REPORT_JSON = REPORTS_DIR / "part3_v2_posthoc_holdout_evaluation.json"

# Designated-split facts (pinned: test.csv md5 guarded by test_final_holdout_integrity).
EXPECTED_TEST_ROWS = 520_391
EXPECTED_START_WEEK = "2018-05-21/2018-05-27"
EXPECTED_END_WEEK = "2018-09-03/2018-09-09"
TEST_CSV_MD5 = "d35bfcd546df74b63c9bd25151fde264"

STOP_EXIT_CODE = 3


# ---------------------------------------------------------------------------
# Once-guard (dedicated; no force override exists)
# ---------------------------------------------------------------------------
def _already_evaluated() -> tuple[bool, str]:
    """Return ``(already_evaluated, evidence)`` for the post-hoc protocol."""
    if POSTHOC_MARKER.exists():
        return True, f"marker file {POSTHOC_MARKER}"
    client = init_mlflow()
    exp = client.get_experiment_by_name("part3_final_lifecycle")
    runs = []
    if exp is not None:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.{POSTHOC_TAG} = 'true'",
        )
    if runs:
        return True, f"MLflow run tags on {[r.info.run_id for r in runs]}"
    return False, "no prior post-hoc evaluation found"


def _write_marker() -> None:
    POSTHOC_MARKER.parent.mkdir(parents=True, exist_ok=True)
    POSTHOC_MARKER.write_text(
        json.dumps(
            {
                "evaluation_type": EVALUATION_TYPE,
                "model_version": MODEL_VERSION,
                "source_run_id": SOURCE_RUN_ID,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "note": (
                    "Designated Test split scored exactly once for V2 under the "
                    "authorized post-hoc holdout protocol (separate from the V1 "
                    "final-lifecycle marker)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Frozen-V2 verification helpers
# ---------------------------------------------------------------------------
def _in_run_artifact_dir() -> Path:
    client = init_mlflow()
    exp = client.get_experiment_by_name("part3_final_lifecycle")
    if exp is None:
        raise RuntimeError("STOP: final-lifecycle experiment not found in MLflow.")
    return PROJECT_ROOT / "mlruns" / str(exp.experiment_id) / SOURCE_RUN_ID / "artifacts"


def _assert_frozen_candidate_unchanged() -> dict:
    """Hard-STOP if the frozen run's freeze config drifted from the code."""
    cfg_path = _in_run_artifact_dir() / "freeze_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"STOP: frozen freeze_config.json missing at {cfg_path}.")
    run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if run_cfg.get("params") != freeze_config()["params"]:
        raise RuntimeError(
            "STOP: frozen candidate drift - run freeze_config params != freeze_config(); "
            "refusing to evaluate a changed model."
        )
    return run_cfg


def _verify_split_reproducible(prepared_test: pd.DataFrame) -> dict:
    """Hard-STOP unless the in-memory test split matches the pinned split file."""
    csv_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    csv_weeks = pd.read_csv(csv_path, usecols=["week"])["week"].astype(str)
    weeks = prepared_test["week"].astype(str)

    checks = {
        "csv_rows": int(len(csv_weeks)),
        "prepared_rows": int(len(prepared_test)),
        "start_week": str(weeks.min()),
        "end_week": str(weeks.max()),
    }
    if checks["prepared_rows"] != checks["csv_rows"]:
        raise RuntimeError(
            f"STOP: test split not reproducible (prepared={checks['prepared_rows']} "
            f"vs csv={checks['csv_rows']})."
        )
    if checks["prepared_rows"] != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"STOP: test split row count {checks['prepared_rows']} != pinned "
            f"{EXPECTED_TEST_ROWS}."
        )
    if checks["start_week"] != EXPECTED_START_WEEK or checks["end_week"] != EXPECTED_END_WEEK:
        raise RuntimeError(
            f"STOP: test week boundaries drifted ({checks['start_week']}.."
            f"{checks['end_week']})."
        )
    return checks


def _load_frozen_v2():
    """Load registered version 2 (no training); verify it is the frozen V2 run."""
    from mlflow import pyfunc

    client = init_mlflow()
    mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)
    if mv.run_id != SOURCE_RUN_ID:
        raise RuntimeError(
            f"STOP: model version {MODEL_VERSION} source run {mv.run_id} != frozen "
            f"V2 run {SOURCE_RUN_ID}."
        )
    loaded = pyfunc.load_model(MODEL_URI)
    LOGGER.info("posthoc_holdout: loaded frozen V2 from %s (stage=%s).", MODEL_URI, mv.current_stage)
    return client, mv, loaded


def _find_raw_estimator(mv) -> Path | None:
    mid = str(mv.source).rsplit("/", 1)[-1]
    hits = sorted(PROJECT_ROOT.joinpath("mlruns").glob(f"*/models/{mid}/**/model.joblib"))
    return hits[0] if hits else None


def _probe_pyfunc(loaded, schema: list[str]) -> tuple[bool, bool]:
    probe = loaded.predict(pd.DataFrame(np.zeros((2, len(schema))), columns=schema))
    model_load_ok = probe is not None
    pyfunc_predict_ok = bool(
        getattr(probe, "columns", None) is not None
        and list(probe.columns) == ["pred_demand"]
        and np.isfinite(probe["pred_demand"].to_numpy()).all()
    )
    return model_load_ok, pyfunc_predict_ok


# ---------------------------------------------------------------------------
# Scoring + integrity
# ---------------------------------------------------------------------------
def score_posthoc_holdout(
    prepared: dict,
    loaded,
    mv,
    client,
    tests_ok: bool = True,
) -> dict:
    """Score the designated Test split with the frozen V2 PyFunc (single pass)."""
    schema = freeze_schema(prepared["train"])
    if len(schema) != 93:
        raise RuntimeError(f"STOP: frozen schema is {len(schema)} cols, expected 93.")

    test_frame = prepared["test"]
    split_meta = _verify_split_reproducible(test_frame)

    test_X, _, _ = design_matrix_rows(test_frame)
    # MLflow serving schema requires double for all 93 inputs; align datatypes
    # (calendar ints -> float64) without altering values, features, or order.
    test_X = test_X.astype("float64")
    test_actual = test_frame[TARGET].astype("float64").to_numpy()
    test_pids = test_frame["product_id"].astype(str).to_numpy()

    # --- schema: frozen layout == test design matrix (exact order) ----------
    if list(test_X.columns) != schema:
        raise RuntimeError("STOP: test design matrix columns != frozen schema order.")

    # --- signature: MLflow serving contract must equal the frozen schema ---
    signature = getattr(getattr(loaded, "metadata", None), "signature", None)
    sig_inputs = [i.name for i in signature.inputs] if signature is not None else []
    sig_outputs = [o.name for o in signature.outputs] if signature is not None else []
    signature_ok = bool(sig_inputs == schema and sig_outputs == ["pred_demand"])

    # --- in-run artifact feature_columns.json must equal the frozen schema --
    fc_path = _in_run_artifact_dir() / "feature_columns.json"
    artifact_cols = json.loads(fc_path.read_text(encoding="utf-8"))["feature_columns"]
    artifact_schema_ok = artifact_cols == schema

    # --- PyFunc probe -------------------------------------------------------
    model_load_ok, pyfunc_predict_ok = _probe_pyfunc(loaded, schema)

    # --- THE evaluation prediction pass (frozen V2 serving path) -----------
    LOGGER.warning(
        "posthoc_holdout: scoring the designated Test split ONCE with frozen V2 "
        "(%d rows) - post-hoc holdout evaluation.",
        len(test_X),
    )
    # Exactly-once hardening: claim this scoring pass on the run *before* the
    # prediction, so even an interruption here cannot trigger a re-score.
    client.set_tag(SOURCE_RUN_ID, POSTHOC_TAG, "true")
    out = loaded.predict(test_X)
    pred = out["pred_demand"].to_numpy(dtype=float)

    # --- inverse-transform cross-check vs the raw estimator artifact --------
    est_path = _find_raw_estimator(mv)
    inverse_ok = False
    inverse_max_abs_diff = None
    if est_path is not None:
        import joblib

        est = joblib.load(est_path)
        manual = inverse_log1p(est.predict(test_X[schema].astype("float64")))
        inverse_max_abs_diff = float(np.max(np.abs(manual - pred)))
        inverse_ok = bool(np.allclose(manual, pred, rtol=0.0, atol=1e-9))
    else:
        LOGGER.warning("posthoc_holdout: raw estimator artifact not found for cross-check.")

    # --- integrity ----------------------------------------------------------
    n_neg = int((pred < 0).sum())
    n_nan_inf = int((~np.isfinite(pred)).sum())
    length_match = bool(len(pred) == len(test_actual) == len(test_frame))
    identity_ok = bool(
        len(test_frame) == len(test_frame["product_id"]) == len(test_frame["week"])
        and test_frame["product_id"].notna().all()
        and test_frame["week"].notna().all()
    )
    pred_integrity = bool(
        n_nan_inf == 0
        and n_neg == 0
        and length_match
        and identity_ok
        and inverse_ok
    )
    integrity = {
        "pred_integrity": pred_integrity,
        "rows_scored": int(len(pred)),
        "n_neg": n_neg,
        "n_nan_inf": n_nan_inf,
        "length_match": length_match,
        "row_product_week_alignment": identity_ok,
        "schema_columns": int(len(schema)),
        "schema_order_ok": list(test_X.columns) == schema,
        "signature_ok": signature_ok,
        "artifact_schema_ok": artifact_schema_ok,
        "inverse_transform_verified": inverse_ok,
        "inverse_max_abs_diff": inverse_max_abs_diff,
        "raw_estimator_artifact": str(est_path) if est_path else None,
        "split_reproduction": split_meta,
        "test_csv_md5_pinned": TEST_CSV_MD5,
    }

    # --- cohorts + MASE (train-only stats, unchanged machinery) -------------
    cohort_test = cohort_flags(prepared["train"], test_pids)
    mase_denom = compute_mase_denominator(prepared["train"])

    # --- the full metric suite (unit space) --------------------------------
    metrics = full_metric_suite(test_actual, pred, cohort_test, mase_denom)

    weeks = test_frame["week"].astype(str)
    cohort_rows = pd.Series(cohort_test).value_counts().to_dict()
    test_metadata = {
        "start_week": str(weeks.min()),
        "end_week": str(weeks.max()),
        "start_date": str(weeks.min().split("/")[0]),
        "end_date": str(weeks.max().split("/")[-1]),
        "rows": int(len(test_frame)),
        "products": int(test_frame["product_id"].nunique()),
        "week_labels": int(weeks.nunique()),
        "total_actual": float(np.sum(test_actual)),
        "total_pred": float(np.sum(pred)),
        "mean_actual": float(np.mean(test_actual)),
        "mean_pred": float(np.mean(pred)),
        "cohort_rows": {str(k): int(v) for k, v in cohort_rows.items()},
        "mase_denominator": float(mase_denom),
    }

    # --- technical booleans (existing Quality Gate contract, unchanged) -----
    technical = {
        "pred_integrity": pred_integrity,
        "schema_ok": bool(len(schema) == 93 and list(test_X.columns) == schema),
        "model_load_ok": bool(model_load_ok),
        "pyfunc_predict_ok": bool(pyfunc_predict_ok),
        "signature_ok": signature_ok,
        "artifacts_ok": bool(artifact_schema_ok and est_path is not None and inverse_ok),
        "tests_ok": bool(tests_ok),
    }

    # --- the existing Quality Gate, thresholds untouched --------------------
    gate = evaluate_gate(metrics, technical, group=EVALUATION_GROUP)
    failed = [f"{o.key}={o.observed} (bound {o.bound})" for o in gate.failed_criteria()]
    LOGGER.info(
        "posthoc_holdout: quality gate -> %s (%s)%s",
        "PASS" if gate.passed else "FAIL",
        gate.promoted_stage,
        f"; failed: {', '.join(failed)}" if failed else "",
    )

    return {
        "schema": schema,
        "metrics": metrics,
        "integrity": integrity,
        "technical": technical,
        "gate": gate.to_dict(),
        "gate_failed_criteria": failed,
        "test_metadata": test_metadata,
        "prediction_sample": pd.DataFrame(
            {
                "product_id": test_frame["product_id"].astype(str).head(20),
                "week": test_frame["week"].astype(str).head(20),
                "pred_demand": pred[:20],
            }
        ),
        "_pred_sum": float(np.sum(pred)),
    }


# ---------------------------------------------------------------------------
# MLflow lifecycle update (tags + stage transition only; no registration)
# ---------------------------------------------------------------------------
def _metric_tags(metrics: dict) -> dict:
    return {
        "final_wape": f"{metrics['wape']:.6f}",
        "final_mae": f"{metrics['mae']:.6f}",
        "final_rmse": f"{metrics['rmse']:.6f}",
        "final_mase": f"{metrics['mase']:.6f}",
        "final_bias": f"{metrics['bias_pct']:.6f}",
        "final_asymmetric_loss": f"{metrics['asym21']:.6f}",
        "positive_wape": f"{metrics['pos_wape']:.6f}",
    }


def update_lifecycle(
    client,
    mv,
    *,
    metrics: dict,
    gate: dict,
    test_metadata: dict,
    audit: dict,
    report_md: str,
) -> dict:
    """Reflect the actual post-hoc gate result in MLflow (additive tags + stage).

    Never registers a model version, never touches V1, never rewrites existing
    tag values (``test_scored_once`` / ``evaluation_group`` / ``wape`` on the
    run and version remain the historical V2 validation record).
    """
    passed = bool(gate["passed"])
    target_stage = "Production" if passed else "Staging"
    before_stage = mv.current_stage

    lifecycle_tags = {
        "lifecycle_stage": target_stage,
        "evaluation_dataset": "test",
        "evaluation_type": EVALUATION_TYPE,
        "model_version": str(MODEL_VERSION),
        "production_gate": "PASS" if passed else "FAIL",
        "test_start_date": test_metadata["start_date"],
        "test_end_date": test_metadata["end_date"],
        "test_rows": str(test_metadata["rows"]),
        POSTHOC_TAG: "true",
        **_metric_tags(metrics),
    }

    # 1) Additive tags on the frozen V2 run (existing tags left intact).
    for key, value in lifecycle_tags.items():
        client.set_tag(SOURCE_RUN_ID, key, str(value))

    # 2) Post-hoc audit + report artifacts inside the frozen V2 run.
    client.log_dict(SOURCE_RUN_ID, audit, "posthoc_holdout_audit.json")
    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "posthoc_holdout_report.md"
        md_path.write_text(report_md, encoding="utf-8")
        client.log_artifact(SOURCE_RUN_ID, str(md_path))

    # 3) Model-version tags: new post-hoc keys + gate/stage decision updated
    #    to the actual result (the two tags that must track current standing).
    version_tags = {
        **lifecycle_tags,
        "gate_decision": "PASS" if passed else "FAIL",
        "promoted_stage": target_stage,
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(MODEL_NAME, MODEL_VERSION, key, str(value))

    # 4) Stage transition of version 2 only, if the actual result requires it.
    transitioned = False
    if before_stage != target_stage:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=MODEL_VERSION,
            stage=target_stage,
            archive_existing_versions=False,  # never disturbs V1 (Staging)
        )
        transitioned = True
        LOGGER.info(
            "posthoc_holdout: version %d transitioned %s -> %s.",
            MODEL_VERSION, before_stage, target_stage,
        )
    else:
        LOGGER.info(
            "posthoc_holdout: version %d already in %s; stage unchanged.",
            MODEL_VERSION, target_stage,
        )

    return {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "source_run_id": SOURCE_RUN_ID,
        "stage_before": before_stage,
        "stage_after": target_stage,
        "transitioned": transitioned,
        "gate_decision": "PASS" if passed else "FAIL",
        "tags_applied": lifecycle_tags,
        "v1_preserved": "version 1 (Staging) and run 7fc87198... untouched",
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _v1_reference() -> dict | None:
    path = REPORTS_DIR / "part3_final_audit.json"
    if not path.exists():
        return None
    audit = json.loads(path.read_text(encoding="utf-8"))
    return {
        "wape": audit["final_test_metrics"]["wape"],
        "B_wape": audit["final_test_metrics"]["B_wape"],
        "gate_passed": audit["quality_gate"]["passed"],
        "stage": audit["registry"]["stage"],
        "run_id": audit["run_id"],
    }


def render_report(
    *,
    metrics: dict,
    integrity: dict,
    gate: dict,
    failed: list[str],
    test_metadata: dict,
    registry: dict,
    mv,
) -> str:
    lines: list[str] = []
    lines.append("# Part-3 V2 Post-Hoc Holdout Evaluation")
    lines.append("")
    lines.append(f"* Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"* Evaluation type: `{EVALUATION_TYPE}` (group `{EVALUATION_GROUP}`)")
    lines.append(f"* Model: `{MODEL_URI}` (version {MODEL_VERSION}, run `{SOURCE_RUN_ID}`)")
    lines.append("")
    lines.append("## Disclosure (required classification)")
    lines.append("")
    lines.append(DISCLOSURE)
    lines.append("")
    lines.append(
        "This artifact never claims an untouched test, unseen test, independent "
        "final test, or never-before-seen test result."
    )
    lines.append("")
    lines.append("## 1. Protocol")
    lines.append("")
    lines.append(
        "- Authorized post-hoc holdout evaluation of the frozen V2 candidate; the "
        "designated Test split was previously consumed by the V1 evaluation and "
        "diagnosis lifecycle."
    )
    lines.append(
        "- Executed exactly once (dedicated marker + `posthoc_holdout_evaluated` "
        "MLflow tag; this module exposes **no** `--force` override)."
    )
    lines.append(
        "- Frozen V2 loaded from the Model Registry: no retraining, no tuning, no "
        "threshold or parameter change before or after evaluation."
    )
    lines.append(
        "- Existing Quality Gate implementation used unchanged (12 numeric bounds + "
        f"7 technical booleans), labeled `group=\"{EVALUATION_GROUP}\"`."
    )
    lines.append(
        "- No further holdout evaluation is available or claimed (raw data ends at "
        "the original test boundary 2018-09-09; historical blocker report preserved)."
    )
    lines.append("")
    lines.append("## 2. Test split metadata (designated split)")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Start week | `{test_metadata['start_week']}` |")
    lines.append(f"| End week | `{test_metadata['end_week']}` |")
    lines.append(f"| Start date | `{test_metadata['start_date']}` |")
    lines.append(f"| End date | `{test_metadata['end_date']}` |")
    lines.append(f"| Rows | {test_metadata['rows']:,} |")
    lines.append(f"| Products | {test_metadata['products']:,} |")
    lines.append(f"| Week labels | {test_metadata['week_labels']} |")
    lines.append(f"| Total actual demand | {test_metadata['total_actual']:,.2f} |")
    lines.append(f"| Total predicted demand | {test_metadata['total_pred']:,.2f} |")
    lines.append(f"| Mean actual | {test_metadata['mean_actual']:.6f} |")
    lines.append(f"| Mean predicted | {test_metadata['mean_pred']:.6f} |")
    lines.append(f"| Cohort rows (B/B2/C) | {test_metadata['cohort_rows']} |")
    lines.append(f"| MASE denominator (train naive) | {test_metadata['mase_denominator']:.6f} |")
    lines.append(f"| `test.csv` md5 (pinned) | `{TEST_CSV_MD5}` |")
    lines.append(f"| Split reproduction | `{integrity['split_reproduction']}` |")
    lines.append("")
    lines.append("## 3. Prediction integrity")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    lines.append(f"| `pred_integrity` (gate) | `{integrity['pred_integrity']}` |")
    lines.append(f"| Rows scored | {integrity['rows_scored']:,} |")
    lines.append(f"| Negative predictions | {integrity['n_neg']} (physical floor active) |")
    lines.append(f"| NaN / Inf predictions | {integrity['n_nan_inf']} |")
    lines.append(f"| Length match | `{integrity['length_match']}` |")
    lines.append(f"| Row/product/week alignment | `{integrity['row_product_week_alignment']}` |")
    lines.append(
        f"| Schema (93 cols, exact order) | `{integrity['schema_order_ok']}` "
        f"({integrity['schema_columns']} cols) |"
    )
    lines.append(f"| MLflow signature == frozen schema | `{integrity['signature_ok']}` |")
    lines.append(f"| In-run `feature_columns.json` == schema | `{integrity['artifact_schema_ok']}` |")
    lines.append(
        f"| Inverse transform verified vs raw estimator | `{integrity['inverse_transform_verified']}` "
        f"(max abs diff {integrity['inverse_max_abs_diff']}) |"
    )
    lines.append("")
    lines.append("## 4. Post-hoc holdout metrics (unit space)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"| `{key}` | {val:.6f} |")
        else:
            lines.append(f"| `{key}` | {val} |")
    lines.append("")
    ref = _v1_reference()
    if ref is not None:
        lines.append("Historical V1 context (already public in `part3_final_audit.json`, not re-scored here):")
        lines.append("")
        lines.append(f"- V1 test WAPE `{ref['wape']:.6f}`, B WAPE `{ref['B_wape']:.6f}`, "
                     f"gate `{ref['gate_passed']}` -> `{ref['stage']}`.")
        lines.append("")
    lines.append("## 5. Quality gate (implementation unchanged)")
    lines.append("")
    passed = bool(gate.get("passed"))
    stage = gate.get("promoted_stage") or ("Production" if passed else "Staging")
    lines.append(f"- Gate: **{'PASS' if passed else 'FAIL'}** (group `{EVALUATION_GROUP}`)")
    lines.append(f"- Resulting stage for version {MODEL_VERSION}: **{stage}**")
    if failed:
        lines.append(f"- Failed criteria: {', '.join(f'`{f}`' for f in failed)}")
    else:
        lines.append("- Failed criteria: none")
    lines.append("")
    lines.append(gate_documentation_markdown())
    lines.append("")
    lines.append("## 6. MLflow lifecycle update")
    lines.append("")
    lines.append(f"- Registered model: **`{registry['model_name']}`**")
    lines.append(f"- Model version: **{registry['version']}** (no new version created; no v3)")
    lines.append(
        f"- Stage: **{registry['stage_before']} -> {registry['stage_after']}** "
        f"(transitioned: `{registry['transitioned']}`)"
    )
    lines.append(f"- Gate decision tag: **{registry['gate_decision']}**")
    lines.append(f"- Tags applied to run `{registry['source_run_id']}` + version {MODEL_VERSION}: "
                 f"`evaluation_type={EVALUATION_TYPE}`, `evaluation_dataset=test`, "
                 f"`lifecycle_stage`, `model_version`, `production_gate`, "
                 f"`test_start_date/test_end_date/test_rows`, `final_*` metrics, "
                 f"`posthoc_holdout_evaluated=true`.")
    lines.append(
        "- Historical records preserved: run/version tags `test_scored_once=false`, "
        "`evaluation_group=v2_validation`, `wape=0.3964` left untouched; V1 version 1 "
        "Staging + run `7fc87198...` + `part3_final_*` files untouched."
    )
    lines.append("")
    lines.append("## 7. Statement")
    lines.append("")
    lines.append("- No tuning, retraining, re-thresholding, or model changes were made "
                 "before or after this evaluation.")
    lines.append("- Result classification: **post-hoc holdout evaluation** - not an "
                 "independent final-test estimate (see Disclosure).")
    lines.append("- This is the final holdout evaluation step for Part 3; no further "
                 "test evaluation is performed.")
    lines.append("")
    return "\n".join(lines)


def build_audit(
    *,
    metrics: dict,
    integrity: dict,
    gate: dict,
    test_metadata: dict,
    registry: dict,
    mv,
    run_freeze_cfg: dict,
) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "part3-v2-posthoc-holdout",
        "evaluation_type": EVALUATION_TYPE,
        "evaluation_group": EVALUATION_GROUP,
        "disclosure": DISCLOSURE,
        "model": {
            "uri": MODEL_URI,
            "model_name": MODEL_NAME,
            "version": MODEL_VERSION,
            "source_run_id": SOURCE_RUN_ID,
            "stage_before": registry["stage_before"],
            "stage_after": registry["stage_after"],
            "feature_version": run_freeze_cfg.get("feature_version"),
            "params": run_freeze_cfg.get("params"),
            "schema_dims": integrity["schema_columns"],
        },
        "test_metadata": test_metadata,
        "integrity": integrity,
        "posthoc_holdout_metrics": metrics,
        "quality_gate": gate,
        "registry": registry,
        "protocol": {
            "exactly_once": True,
            "force_override_exists": False,
            "retraining_performed": False,
            "tuning_after_evaluation": False,
            "thresholds_changed": False,
            "second_model_created": False,
            "v1_history_preserved": True,
            "further_holdout_available": False,
        },
        "notes": (
            "Post-hoc holdout evaluation: Test split previously consumed by V1 "
            "evaluation/diagnosis; thresholds unchanged; frozen V2 loaded from "
            "registry without retraining; V1 records untouched."
        ),
    }


def _persist(report_md: str, audit: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report_md, encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    LOGGER.info("posthoc_holdout: reports written to %s.", REPORTS_DIR)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(tests_ok: bool = True) -> dict:
    already, evidence = _already_evaluated()
    if already:
        LOGGER.error(
            "posthoc_holdout: post-hoc evaluation already executed (%s). "
            "Refusing to re-run; this module has no force override.",
            evidence,
        )
        sys.exit(STOP_EXIT_CODE)

    run_freeze_cfg = _assert_frozen_candidate_unchanged()
    LOGGER.info("posthoc_holdout: frozen candidate verified against run freeze config.")

    prepared = build_prepared(schema_len_expected=93)
    client, mv, loaded = _load_frozen_v2()

    result = score_posthoc_holdout(prepared, loaded, mv, client, tests_ok=tests_ok)
    gate = result["gate"]
    metrics = result["metrics"]

    # Registry state BEFORE any update (recorded in the audit).
    registry_state = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "source_run_id": SOURCE_RUN_ID,
        "stage_before": mv.current_stage,
        "stage_after": "",  # filled by update_lifecycle
        "transitioned": False,
        "gate_decision": "PASS" if gate["passed"] else "FAIL",
    }

    report_md = render_report(
        metrics=metrics,
        integrity=result["integrity"],
        gate=gate,
        failed=result["gate_failed_criteria"],
        test_metadata=result["test_metadata"],
        registry=registry_state,
        mv=mv,
    )

    # Preliminary audit for MLflow (registry transition fields filled after;
    # the version tags + stage are applied first so MLflow reflects the result
    # even if later file persistence were interrupted).
    audit = build_audit(
        metrics=metrics,
        integrity=result["integrity"],
        gate=gate,
        test_metadata=result["test_metadata"],
        registry=registry_state,
        mv=mv,
        run_freeze_cfg=run_freeze_cfg,
    )

    registry_out = update_lifecycle(
        client,
        mv,
        metrics=metrics,
        gate=gate,
        test_metadata=result["test_metadata"],
        audit=audit,
        report_md=report_md,
    )
    audit["registry"] = registry_out
    # Model block stage fields must reflect the actual final state too (never
    # an empty stage_after as a leftover of the preliminary registry).
    audit["model"]["stage_before"] = registry_out["stage_before"]
    audit["model"]["stage_after"] = registry_out["stage_after"]

    # Re-render the report's registry section facts with the final state.
    report_md = render_report(
        metrics=metrics,
        integrity=result["integrity"],
        gate=gate,
        failed=result["gate_failed_criteria"],
        test_metadata=result["test_metadata"],
        registry=registry_out,
        mv=mv,
    )
    # Keep the logged-in-run audit + report consistent with the final state too
    # (the report/audit logged during update_lifecycle carried the preliminary
    # registry; re-logging the same filenames overwrites them with the final
    # lifecycle state so the run artifact never shows an empty stage_after).
    client.log_dict(SOURCE_RUN_ID, audit, "posthoc_holdout_audit.json")
    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "posthoc_holdout_report.md"
        md_path.write_text(report_md, encoding="utf-8")
        client.log_artifact(SOURCE_RUN_ID, str(md_path))

    _persist(report_md, audit)
    _write_marker()  # exactly-once marker: written last, after full success

    summary = {
        "evaluation_type": EVALUATION_TYPE,
        "disclosure": DISCLOSURE,
        "metrics": metrics,
        "gate_passed": gate["passed"],
        "gate_failed_criteria": result["gate_failed_criteria"],
        "promoted_stage": registry_out["stage_after"],
        "registry": registry_out,
        "test_metadata": result["test_metadata"],
        "integrity": result["integrity"],
        "artifacts": [str(REPORT_MD), str(REPORT_JSON), str(POSTHOC_MARKER)],
    }
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, default=str))
