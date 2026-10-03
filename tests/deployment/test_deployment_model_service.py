"""Registry-resolution tests against the real MLflow tracking DB (read-only)."""

from __future__ import annotations

import sqlite3

import pytest

from app.config import Settings
from app.errors import AppError
from app.services import model_service

REF_DB = "artifacts/mlflow_tracking.db"


@pytest.fixture(scope="module")
def settings():
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{REF_DB}",
        mlflow_model_name="demand_forecasting_final",
        mlflow_model_version="2",
        mlflow_model_alias="Production",
        mlflow_mlruns_dir="mlruns",
    )


def test_registry_db_exists():
    import os

    assert os.path.exists(REF_DB)


def test_production_version_is_pinned_and_ready(settings):
    reg = model_service.resolve_registered_version(settings)
    assert reg["name"] == "demand_forecasting_final"
    assert reg["version"] == 2
    assert reg["stage"] == "Production"
    assert reg["status"] == "READY"
    assert reg["run_id"]
    assert reg["source"].startswith("models:/")
    assert reg["model_id"] == reg["source"].removeprefix("models:/")


def test_artifact_dir_resolves_in_repo_layout(settings):
    reg = model_service.resolve_registered_version(settings)
    artifact_dir = model_service.resolve_artifact_dir(
        settings.mlflow_mlruns_dir, reg["model_id"]
    )
    assert artifact_dir is not None
    assert (artifact_dir / "MLmodel").is_file()
    assert (artifact_dir / "artifacts" / "model.joblib").is_file()


def test_pinned_version_missing_raises():
    s = Settings(
        mlflow_tracking_uri=f"sqlite:///{REF_DB}",
        mlflow_model_name="demand_forecasting_final",
        mlflow_model_version="999",
    )
    with pytest.raises(AppError) as ei:
        model_service.resolve_registered_version(s)
    assert ei.value.code == "MODEL_LOAD_FAILED"


def test_unknown_model_raises():
    s = Settings(
        mlflow_tracking_uri=f"sqlite:///{REF_DB}",
        mlflow_model_name="does_not_exist_model",
        mlflow_model_version="1",
    )
    with pytest.raises(AppError) as ei:
        model_service.resolve_registered_version(s)
    assert ei.value.code == "MODEL_LOAD_FAILED"


def test_artifact_dir_missing_returns_none():
    assert model_service.resolve_artifact_dir("mlruns", "m-does-not-exist") is None


def test_registry_readonly_is_not_modified(settings):
    """Resolution must not alter the registry (compare a registry snapshot)."""
    import sqlite3

    before = _snapshot()
    model_service.resolve_registered_version(settings)
    after = _snapshot()
    assert before == after


def test_path_normalized_load_matches_primary(settings):
    """Container parity: the packaged MLmodel stores its data path with a Windows
    separator, which mlflow cannot resolve on POSIX. Loading through a
    path-normalized copy must produce an identical served model (93-feature
    signature, same predictor) as the primary load."""
    import numpy as np
    import pandas as pd

    from app.services import model_service as ms_mod

    reg = ms_mod.resolve_registered_version(settings)
    artifact_dir = ms_mod.resolve_artifact_dir(settings.mlflow_mlruns_dir, reg["model_id"])
    import mlflow

    primary = mlflow.pyfunc.load_model(str(artifact_dir))
    normalized = ms_mod._load_normalized_pyfunc(artifact_dir)
    primary_inputs = [c.name for c in primary.metadata.signature.inputs.inputs]
    packaged_inputs = [c.name for c in normalized.metadata.signature.inputs.inputs]
    assert len(packaged_inputs) == 93
    assert packaged_inputs == primary_inputs

    feat = pd.DataFrame(
        np.full((1, 93), 0.5, dtype="float64"), columns=packaged_inputs
    )
    res = normalized.predict(feat)
    pred = res["pred_demand"].iloc[0] if hasattr(res, "iloc") else res[0]
    res_p = primary.predict(feat)
    pred_p = res_p["pred_demand"].iloc[0] if hasattr(res_p, "iloc") else res_p[0]
    assert pred == pred_p


def _snapshot() -> list:
    conn = sqlite3.connect(REF_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, version, current_stage, status, run_id, source"
        " FROM model_versions ORDER BY 1, 2"
    ).fetchall()
    return [tuple(r) for r in rows]