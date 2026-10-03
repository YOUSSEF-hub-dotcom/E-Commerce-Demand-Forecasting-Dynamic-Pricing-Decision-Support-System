"""Part-3 MLflow lifecycle tests (Phase G).

Exercises the real MLflow tracking + Model Registry machinery used by the
finalize_lifecycle orchestrator on an isolated local file store:

* run logging (params/metrics/artifacts incl. pred-only parquet),
* PyFunc reload + predict on unit scale,
* registration as a real registered-model version,
* auto-promotion: PASS -> Production (previous preserved/archived),
* FAIL candidate stays in Staging,
* audit trail artifact + version tags.
"""

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient

import src.models.registry as registry
from src.models.pyfunc_wrapper import DemandForecaster


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the registry module at a throwaway local MLflow file store."""
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setattr(registry, "TRACKING_URI", uri)
    monkeypatch.setattr(registry, "FINAL_EXPERIMENT", "test_lifecycle")
    monkeypatch.setattr(registry, "MODEL_NAME", "test_demand_forecaster")
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    mlflow.set_tracking_uri(uri)
    client = registry.init_mlflow()
    yield uri, client


def _dummy_model_and_schema() -> tuple:
    cols = [f"f{i:02d}" for i in range(93)]
    return cols, cols  # model is a trivial constant estimator below


class _ConstModel:
    """Minimal estimator returning log1p(0) -> pred_demand ~ 0 units."""

    def predict(self, X):
        return np.zeros(X.shape[0])


def test_run_logging_with_pred_only_parquet(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    preds = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "week": ["2018-01-01", "2018-01-08"],
            "pred_demand": [1.5, 0.0],
        }
    )
    run_id = registry.log_final_run(
        model=_ConstModel(),
        feature_columns=cols,
        freeze_config={
            "feature_version": "phase3-exp2-v1",
            "params": {"learning_rate": 0.008468},
        },
        val_metrics={"wape": 0.3964},
        test_metrics={"wape": 0.4000},
        test_predictions=preds,
        report_md="# report",
    )
    run = client.get_run(run_id)
    assert run.data.tags.get("test_scored_once") == "true"
    assert run.data.params.get("feature_version") == "phase3-exp2-v1"
    assert run.data.metrics.get("test-wape") is not None
    assert run.data.metrics.get("validation-wape") == 0.3964
    artifact_names = [a.path for a in mlflow.tracking.MlflowClient().list_artifacts(run_id)]
    assert "freeze_config.json" in artifact_names
    assert "feature_columns.json" in artifact_names


def test_pyfunc_reload_and_predict_unit_scale(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    run_id = registry.log_final_run(
        model=_ConstModel(),
        feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={},
        test_metrics={"wape": 0.4},
    )
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/demand_forecaster")
    request = pd.DataFrame(np.zeros((3, 93)), columns=cols)
    out = loaded.predict(request)
    assert list(out.columns) == ["pred_demand"]
    assert out.shape[0] == 3
    assert out["pred_demand"].ge(0).all()
    assert np.isfinite(out["pred_demand"]).all()


def test_pyfunc_rejects_broken_schema(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    run_id = registry.log_final_run(
        model=_ConstModel(),
        feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={},
        test_metrics={"wape": 0.4},
    )
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/demand_forecaster")
    bad = pd.DataFrame(np.zeros((2, 5)), columns=["x"] * 5)
    with pytest.raises((ValueError, mlflow.exceptions.MlflowException)):
        loaded.predict(bad)


def test_register_and_promote_production(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    run_id = registry.log_final_run(
        model=_ConstModel(),
        feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={},
        test_metrics={"wape": 0.4},
    )
    reg = registry.register_and_promote(run_id, passed=True)
    assert reg["stage"] == "Production"
    versions = client.search_model_versions(f"name='{registry.MODEL_NAME}'")
    assert len(versions) == 1
    assert versions[0].current_stage == "Production"
    assert versions[0].run_id == run_id


def test_promote_production_archives_previous(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    cfg = {"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}}
    run1 = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols, freeze_config=cfg,
        val_metrics={}, test_metrics={"wape": 0.4},
    )
    registry.register_and_promote(run1, passed=True)
    run2 = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols, freeze_config=cfg,
        val_metrics={}, test_metrics={"wape": 0.39},
    )
    registry.register_and_promote(run2, passed=True)
    versions = client.search_model_versions(f"name='{registry.MODEL_NAME}'")
    stages = {str(v.version): v.current_stage for v in versions}
    assert stages[str("2")] == "Production"
    assert stages[str("1")] == "Archived"  # previous production preserved, not deleted


def test_failed_candidate_stays_in_staging(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    run_id = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={}, test_metrics={"wape": 0.42},
    )
    reg = registry.register_and_promote(run_id, passed=False)
    assert reg["stage"] == "Staging"
    versions = client.search_model_versions(f"name='{registry.MODEL_NAME}'")
    assert versions[0].current_stage == "Staging"


def test_audit_trail_artifact_and_version_tags(isolated_store):
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    run_id = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={}, test_metrics={"wape": 0.4},
    )
    reg = registry.register_and_promote(run_id, passed=True)
    audit = registry.build_audit(
        run_id=run_id,
        freeze_config={"feature_version": "phase3-exp2-v1"},
        val_metrics={"wape": 0.3964},
        test_metrics={"wape": 0.4},
        gate={"passed": True, "promoted_stage": "Production"},
        registry=reg,
        schema=cols,
        mase_denom=0.03804,
    )
    client.log_dict(run_id, audit, "audit_trail.json")
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert "audit_trail.json" in artifacts
    versions = client.search_model_versions(f"name='{registry.MODEL_NAME}'")
    assert versions[0].tags.get("gate_decision") == "PASS"
    assert versions[0].tags.get("promoted_stage") == "Production"
    assert audit["schema_dims"] == 93
    assert audit["phase"] == "part3-final-lifecycle"


def test_v2_register_tag_overrides(isolated_store):
    """v2 path: extra_tags override the default test_scored_once='true' tag."""
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    run_id = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={"wape": 0.3964}, test_metrics={},
        run_tags={"test_scored_once": "false", "evaluation_group": "v2_validation"},
    )
    assert client.get_run(run_id).data.tags.get("test_scored_once") == "false"
    reg = registry.register_and_promote(
        run_id, passed=True,
        extra_tags={"test_scored_once": "false", "evaluation_group": "v2_validation"},
    )
    assert reg["stage"] == "Production"
    versions = client.search_model_versions(f"name='{registry.MODEL_NAME}'")
    assert versions[0].tags.get("test_scored_once") == "false"
    assert versions[0].tags.get("evaluation_group") == "v2_validation"


def test_v2_register_preserves_prior_staging_version(isolated_store):
    """Promoting v2 credits a new version without disturbing an existing Staging v1."""
    uri, client = isolated_store
    cols, _ = _dummy_model_and_schema()
    cfg = {"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}}
    run1 = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols, freeze_config=cfg,
        val_metrics={}, test_metrics={"wape": 0.42},
    )
    registry.register_and_promote(run1, passed=False)  # v1 -> Staging
    run2 = registry.log_final_run(
        model=_ConstModel(), feature_columns=cols, freeze_config=cfg,
        val_metrics={"wape": 0.3964}, test_metrics={},
        run_tags={"test_scored_once": "false", "evaluation_group": "v2_validation"},
    )
    registry.register_and_promote(
        run2, passed=True,
        extra_tags={"test_scored_once": "false", "evaluation_group": "v2_validation"},
    )
    versions = client.search_model_versions(f"name='{registry.MODEL_NAME}'")
    stages = {str(v.version): v.current_stage for v in versions}
    assert stages["2"] == "Production"
    assert stages["1"] == "Staging"  # v1 preserved untouched