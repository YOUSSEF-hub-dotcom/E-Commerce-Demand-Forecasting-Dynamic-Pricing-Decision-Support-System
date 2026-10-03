"""Part-3 MLflow lifecycle driver (Phase C).

Real, non-simulated MLflow lifecycle:

* **Tracking backend** - the project's SQLite store
  (``artifacts/mlflow_tracking.db``) shared with the Part-3 notebooks so the
  final-lifecycle run sits in the same lineage.
* **Run logging** - params (freeze config + hyperparameters), validation and
  final-test metrics, and artifacts (PyFunc model, 93-column schema, freeze
  config, gate report, pred-only predictions, final report).
* **Model Registry** - the candidate is registered as a real *registered
  model* version (``MlflowClient.create_registered_model`` /
  ``register_model``) and then **auto-promoted** to ``Production`` (gate PASS)
  or ``Staging`` (gate FAIL) via ``transition_model_version_stage`` with the
  previous Production version archived (preserved).
* **Audit trail** - tags on the run and on the model version, plus a JSON
  audit artifact inside the run.

Nothing here reads or writes the protected test split inside ``data/splits``;
the lifecycle orchestrator computes test features/predictions in-memory from
the frozen pipeline and logs only predictions (never the test targets) as a
``pred_only`` parquet artifact.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.pyfunc_wrapper import DemandForecaster, save_winner
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRACKING_DB = ARTIFACTS_DIR / "mlflow_tracking.db"

# Use the existing Part-3 SQLite tracking backend (same lineage as notebooks).
TRACKING_URI = "sqlite:///" + TRACKING_DB.as_posix().replace("\\", "/")

FINAL_EXPERIMENT = "part3_final_lifecycle"

# Registered-model identity for the frozen champion.
MODEL_NAME = "demand_forecasting_final"
MODEL_ARTIFACT_PATH = "demand_forecaster"


def init_mlflow() -> MlflowClient:
    """Point MLflow at the project tracking DB and create the experiment."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        client.create_experiment(FINAL_EXPERIMENT)
        LOGGER.info("init_mlflow: created experiment '%s'.", FINAL_EXPERIMENT)
    except Exception:
        LOGGER.info("init_mlflow: experiment '%s' already exists.", FINAL_EXPERIMENT)
    mlflow.set_experiment(FINAL_EXPERIMENT)
    return client


def _flatten_params(freeze_config: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in freeze_config.items():
        fq = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=f"{fq}."))
        elif isinstance(v, (list, tuple)):
            out[fq] = json.dumps(list(v))
        else:
            out[fq] = str(v)
    return out


def _metric_name(kind: str, key: str) -> str:
    """Prefix metric keys by split kind: ``validation_wape`` / ``test_wape``."""
    return f"{kind}-{key}" if not key.startswith(kind) else key


def log_final_run(
    *,
    model,
    feature_columns: list[str],
    freeze_config: dict,
    val_metrics: dict,
    test_metrics: dict,
    gate_report: dict | None = None,
    test_predictions: pd.DataFrame | None = None,
    val_predictions: pd.DataFrame | None = None,
    report_md: str = "",
    audit: dict | None = None,
    run_tags: dict | None = None,
) -> str:
    """Log the final-lifecycle run and return the run_id.

    The PyFunc model is packaged with an explicit signature inferred from a
    single design row so the serving contract (93 inputs -> ``pred_demand``)
    is recorded in the registry version.
    """
    import joblib
    import tempfile

    from mlflow.models.signature import infer_signature

    init_mlflow()
    with mlflow.start_run(run_name="final_lifecycle_execution") as run:
        run_id = run.info.run_id
        LOGGER.info("log_final_run: started run_id=%s", run_id)

        for k, v in _flatten_params(freeze_config).items():
            mlflow.log_param(k, v)

        # Per-hyperparameter params (flattened top-level for easy registry views).
        for pname, pvalue in freeze_config["params"].items():
            mlflow.log_param(f"hp.{pname}", str(pvalue))

        # Validation metrics (reproduction check) and FINAL TEST metrics.
        for key in ("wape", "mae", "rmse", "mase", "bias_pct", "asym21", "pos_wape",
                    "B_wape", "B_bias_pct", "B_units", "B2_wape", "B2_bias_pct",
                    "C_wape", "C_bias_pct"):
            if key in val_metrics:
                mlflow.log_metric(_metric_name("validation", key), float(val_metrics[key]))
        for key in test_metrics:
            mlflow.log_metric(_metric_name("test", key), float(test_metrics[key]))

        # Package the PyFunc model with an explicit signature (93 -> pred_demand).
        sample_row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        signature = infer_signature(sample_row, pd.DataFrame({"pred_demand": [0.0]}))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            joblib.dump(model, tmp_dir / "model.joblib")
            project_root = PROJECT_ROOT
            code_paths = [
                str(project_root / "src" / p)
                for p in ("utils", "models", "data", "features")
            ]
            wrapper = DemandForecaster(feature_columns)
            mlflow.pyfunc.log_model(
                artifact_path=MODEL_ARTIFACT_PATH,
                python_model=wrapper,
                artifacts={"model": str(tmp_dir / "model.joblib")},
                code_paths=code_paths,
                signature=signature,
            )
            LOGGER.info("log_final_run: logged PyFunc model with signature.")

        # Reproducibility artifacts.
        mlflow.log_dict(freeze_config, "freeze_config.json")
        mlflow.log_dict({"feature_columns": feature_columns}, "feature_columns.json")

        if gate_report is not None:
            mlflow.log_dict(gate_report, "quality_gate_report.json")

        if test_predictions is not None:
            with tempfile.TemporaryDirectory() as tmp:
                pred_path = os.path.join(tmp, "test_predictions.parquet")
                test_predictions.to_parquet(pred_path, index=False)
                mlflow.log_artifact(pred_path)
        if val_predictions is not None:
            mlflow.log_dict(val_predictions.head(5).to_dict(), "val_predictions_sample.json")

        if report_md:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "final_report.md"
                p.write_text(report_md, encoding="utf-8")
                mlflow.log_artifact(str(p))

        if audit is not None:
            mlflow.log_dict(audit, "audit_trail.json")

        mlflow.set_tags(
            {
                "phase": "part3-final-lifecycle",
                "purpose": "frozen candidate: log-LGBM + EXP2 regime features",
                "test_scored_once": "true",
                **(run_tags or {}),
            }
        )
        LOGGER.info("log_final_run: run_id=%s complete.", run_id)
        return run_id


def register_and_promote(
    run_id: str,
    *,
    passed: bool,
    stage: str | None = None,
    audit_tags: dict | None = None,
    extra_tags: dict | None = None,
    model_name: str | None = None,
) -> dict[str, str]:
    """Register the run's model and auto-promote it in the real MLflow Model Registry.

    Parameters
    ----------
    run_id : str
        The final-lifecycle run containing ``demand_forecaster``.
    passed : bool
        Quality-gate decision: ``True`` -> ``Production``, ``False`` -> ``Staging``.
    stage : str | None
        Override stage. If ``None`` it is derived from ``passed``.
    audit_tags : dict | None
        Extra tags applied to the model version.
    extra_tags : dict | None
        Additional tags merged (and overriding) the default version-tag set;
        e.g. a v2 run records ``test_scored_once=False`` and the evaluation
        group it was gated on.
    model_name : str | None
        Registered-model name override (defaults to ``MODEL_NAME``).

    Returns
    -------
    dict
        ``{"model_name": ..., "version": ..., "stage": ..., "run_id": ...}``
    """
    client = init_mlflow()
    target = stage or ("Production" if passed else "Staging")
    reg_name = model_name or MODEL_NAME

    # Register as a real registered model (create once, register the version).
    try:
        client.create_registered_model(
            reg_name,
            tags={
                "model_type": "lightgbm",
                "feature_version": "phase3-exp2-v1",
                "owner": "forecasting-part3",
            },
        )
        LOGGER.info("register_and_promote: created registered model '%s'.", reg_name)
    except Exception:
        LOGGER.info("register_and_promote: registered model '%s' exists.", reg_name)

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
    version = mv.version
    LOGGER.info("register_and_promote: registered version %s (run %s).", version, run_id)

    # Auto-promotion: transition the new version to the target stage, archiving
    # (preserving) the previously-Production version when promoting to Production.
    if target == "Production":
        client.transition_model_version_stage(
            name=reg_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
    else:
        client.transition_model_version_stage(
            name=reg_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )

    version_tags = {
        "gate_decision": "PASS" if passed else "FAIL",
        "promoted_stage": target,
        "run_id": run_id,
        "test_scored_once": "true",
        **(audit_tags or {}),
        **(extra_tags or {}),
    }
    for k, v in version_tags.items():
        client.set_model_version_tag(reg_name, version, k, str(v))

    LOGGER.info(
        "register_and_promote: version %s of '%s' -> stage '%s'.",
        version,
        reg_name,
        target,
    )
    return {"model_name": reg_name, "version": version, "stage": target, "run_id": run_id}


def current_stage_versions() -> list[dict[str, str]]:
    """Snapshot of the registered model's versions + stages (audit view)."""
    client = init_mlflow()
    out = []
    try:
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            out.append(
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                    "tags": dict(v.tags or {}),
                }
            )
    except Exception as exc:  # registry may be empty on first call
        LOGGER.warning("current_stage_versions: no versions yet (%s).", exc)
    return out


def build_audit(
    *,
    run_id: str,
    freeze_config: dict,
    val_metrics: dict,
    test_metrics: dict,
    gate: dict,
    registry: dict[str, str],
    schema: list[str],
    mase_denom: float,
    notes: str = "",
) -> dict:
    """Assemble the machine-readable audit trail record."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "part3-final-lifecycle",
        "run_id": run_id,
        "freeze_config": freeze_config,
        "schema_columns": schema,
        "schema_dims": len(schema),
        "mase_denominator": mase_denom,
        "validation_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "quality_gate": gate,
        "registry": registry,
        "test_predictions_artifact": {
            "note": "pred-only parquet inside run; test targets are NOT copied into "
            "any artifact or report (protected split).",
        },
        "notes": notes,
    }