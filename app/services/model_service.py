"""Production model resolution and loading (read-only).

The registered Production model (`models:/demand_forecasting_final/2`) is served
without any modification. Because the registry's ``storage_location`` records the
*authoring machine's* absolute path (a Windows path), resolving the ``models:/``
URI directly is not portable into a Linux container. We therefore resolve the
registered version **through the registry** (name -> version -> stage/status ->
run id -> ``source`` -> model id) and then load the model's artifact directory
from the filesystem layout present inside the image (``mlruns/*/models/<id>/``).

This keeps the deployed weights byte-for-byte identical to the registered
Production model while making the load fully self-contained.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.errors import MODEL_LOAD_FAILED, AppError

logger = logging.getLogger("deploy.model")

EXPECTED_FEATURE_COUNT = 93
ARTIFACTS_MARKER = "MLmodel"


def resolve_registered_version(settings: Settings) -> dict:
    """Return the registry record for the pinned Production model version."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    try:
        registered = client.get_registered_model(settings.mlflow_model_name)
    except Exception as exc:
        raise AppError(
            503,
            MODEL_LOAD_FAILED,
            f"registered model {settings.mlflow_model_name!r} not found: {exc}",
        ) from exc
    versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")

    chosen = None
    if settings.mlflow_model_version:
        chosen = next(
            (v for v in versions if str(v.version) == str(settings.mlflow_model_version)),
            None,
        )
        if chosen is None:
            raise AppError(
                503,
                MODEL_LOAD_FAILED,
                f"model version {settings.mlflow_model_version} not found for "
                f"{settings.mlflow_model_name}",
            )
    elif settings.mlflow_model_alias:
        chosen = next(
            (v for v in versions if v.current_stage.lower() == settings.mlflow_model_alias.lower()),
            None,
        )
        if chosen is None:
            raise AppError(
                503,
                MODEL_LOAD_FAILED,
                f"no version of {settings.mlflow_model_name} is staged "
                f"{settings.mlflow_model_alias!r}",
            )
    else:
        raise AppError(
            503,
            MODEL_LOAD_FAILED,
            "no model version or alias configured for selection",
        )

    if chosen.status != "READY":
        raise AppError(
            503,
            MODEL_LOAD_FAILED,
            f"model version {chosen.version} is not READY (status={chosen.status})",
        )

    source = getattr(chosen, "source", "") or ""
    model_id = source.removeprefix("models:/").removesuffix("/")
    if not model_id:
        raise AppError(
            503,
            MODEL_LOAD_FAILED,
            f"model version {chosen.version} has no usable models:/ source ({source!r})",
        )
    return {
        "name": settings.mlflow_model_name,
        "version": int(chosen.version),
        "stage": chosen.current_stage,
        "status": chosen.status,
        "run_id": chosen.run_id,
        "source": source,
        "model_id": model_id,
    }


def resolve_artifact_dir(mlruns_dir: str | Path, model_id: str) -> Path | None:
    """Locate the model artifact directory containing ``MLmodel`` in the image layout."""
    root = Path(mlruns_dir)
    if not root.is_dir():
        return None
    candidates = sorted(root.glob(f"*/models/{model_id}/*"))
    for candidate in candidates:
        if (candidate / ARTIFACTS_MARKER).is_file():
            return candidate
    return None


def feature_schema(model) -> list[str]:
    sig = getattr(model, "metadata", None)
    if sig is None or getattr(sig, "signature", None) is None:
        raise AppError(503, MODEL_LOAD_FAILED, "model carries no metadata signature")
    inputs = sig.signature.inputs
    if inputs is None:
        raise AppError(503, MODEL_LOAD_FAILED, "model signature has no inputs")
    schema = [t.name for t in inputs]
    if len(schema) != EXPECTED_FEATURE_COUNT:
        raise AppError(
            503,
            MODEL_LOAD_FAILED,
            f"model signature has {len(schema)} features, expected {EXPECTED_FEATURE_COUNT}",
        )
    return schema


def unwrap_estimator(model, artifact_dir: Path) -> tuple:
    """Unwrap the LGBM regressor behind the registered PyFunc wrapper (read-only).

    Prefers the public ``unwrap_python_model`` API; falls back to deserialising
    the ``model.joblib`` artifact shipped next to ``MLmodel``.
    """
    import joblib

    est = None
    source = None
    if hasattr(model, "unwrap_python_model"):
        try:
            pm = model.unwrap_python_model()
            cand = getattr(pm, "_model", pm)
            if callable(getattr(cand, "predict", None)):
                est = cand
                source = f"unwrap_python_model.{type(cand).__name__}"
        except Exception as exc:  # private layout drift across mlflow versions
            logger.warning("unwrap_python_model failed (%s); falling back to joblib", exc)
    if est is None:
        candidates = [
            artifact_dir / "artifacts" / "model.joblib",
            artifact_dir / "model.joblib",
        ]
        joblib_path = next((p for p in candidates if p.is_file()), None)
        if joblib_path is None:
            raise AppError(
                503, MODEL_LOAD_FAILED, f"model artifact model.joblib is missing under {artifact_dir}"
            )
        loaded = joblib.load(joblib_path)
        est = getattr(loaded, "_model", loaded)
        source = f"model.joblib artifact ({joblib_path.name})"
    if not callable(getattr(est, "predict", None)):
        raise AppError(503, MODEL_LOAD_FAILED, "resolved estimator has no predict method")
    return est, source


def _load_normalized_pyfunc(artifact_dir: Path):
    """Load the packaged PyFunc model from a path-normalized copy.

    The immutable MLmodel was written on Windows and stores its data path with a
    backslash separator (``artifacts\\model.joblib``), which ``mlflow.pyfunc``
    cannot resolve on a case-sensitive/POSIX container. Copying to a temp dir and
    normalizing the separator lets mlflow take its exact normal load path, so the
    served surface (metadata, signature, context, predictions) is identical to a
    Windows load.
    """
    import shutil
    import tempfile

    import yaml

    tmp = Path(tempfile.mkdtemp(prefix="demand_model_normalized_"))
    target = tmp / "artifacts"
    shutil.copytree(artifact_dir, target)
    mlmodel = target / "MLmodel"
    conf = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))

    def _norm(value):
        if isinstance(value, str):
            return value.replace("\\", "/")
        if isinstance(value, dict):
            return {k: _norm(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_norm(v) for v in value]
        return value

    # MLflow 3.x stores artifact paths in flavors.python_function.artifacts;
    # legacy models keep a top-level `data` field. Normalize both so the
    # os.path.join inside mlflow resolves on POSIX.
    legacy_data = conf.get("data")
    if isinstance(legacy_data, str) and "\\" in legacy_data:
        conf["data"] = legacy_data.replace("\\", "/")
    pf = conf.get("flavors", {}).get("python_function")
    if isinstance(pf, dict) and isinstance(pf.get("artifacts"), dict):
        pf["artifacts"] = _norm(pf["artifacts"])
    mlmodel.write_text(
        yaml.safe_dump(conf, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    import mlflow

    return mlflow.pyfunc.load_model(str(target))


def load_production(settings: Settings) -> dict:
    """Registry-verified, filesystem-portable load of the Production model."""
    import mlflow

    registry = resolve_registered_version(settings)
    artifact_dir = resolve_artifact_dir(settings.mlflow_mlruns_dir, registry["model_id"])
    if artifact_dir is None:
        raise AppError(
            503,
            MODEL_LOAD_FAILED,
            f"model artifact directory for {registry['model_id']} not found under "
            f"{settings.mlflow_mlruns_dir!r}",
        )
    model, schema, estimator, estimator_source = None, None, None, None
    try:
        model = mlflow.pyfunc.load_model(str(artifact_dir))
    except Exception as exc:
        # The baked MLmodel references its data file with a Windows separator,
        # which mlflow cannot resolve on a case-sensitive/POSIX container. Retry
        # through a path-normalized copy so the exact normal load path is used.
        logger.warning(
            "mlflow.pyfunc.load_model failed (%s: %s); retrying via path-normalized copy",
            type(exc).__name__,
            exc,
        )
        try:
            model = _load_normalized_pyfunc(artifact_dir)
        except Exception as exc2:
            raise AppError(
                503,
                MODEL_LOAD_FAILED,
                f"model could not be loaded from the packaged MLmodel on this platform: {exc2}",
            ) from exc2
        estimator_source_hint = " (path-normalized copy)"
    else:
        estimator_source_hint = ""
    schema = feature_schema(model)
    estimator, estimator_source = unwrap_estimator(model, artifact_dir)
    estimator_source = estimator_source + estimator_source_hint
    model_info = {
        "name": registry["name"],
        "version": registry["version"],
        "stage": registry["stage"],
        "status": registry["status"],
        "run_id": registry["run_id"],
        "source": registry["source"],
    }
    logger.info(
        "production model loaded: %s v%d stage=%s run=%s from %s",
        registry["name"],
        registry["version"],
        registry["stage"],
        registry["run_id"],
        artifact_dir,
    )
    return {
        "model": model if model is not None else estimator,
        "estimator": estimator,
        "estimator_source": estimator_source,
        "schema": schema,
        "info": model_info,
        "artifact_dir": str(artifact_dir),
    }