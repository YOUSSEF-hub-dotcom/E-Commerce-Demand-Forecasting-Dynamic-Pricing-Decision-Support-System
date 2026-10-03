"""MLflow PyFunc wrapper for the winning demand forecaster.

Packages the trained model behind a strict input **signature contract**:

* ``DemandForecaster`` - an ``mlflow.pyfunc.PythonModel`` subclass that loads
  the underlying joblib estimator in ``load_context`` and validates every
  prediction request against the exact feature schema seen at training time.
* Predictions are returned on the business scale as ``pred_demand`` units
  (``expm1`` of the log-domain output), shaped as a DataFrame.
"""

import joblib
import numpy as np
import pandas as pd
import mlflow

from src.models.final_forecast import inverse_log1p
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def validate_input(frame, feature_columns: list[str]) -> None:
    """Validate that a prediction request matches the training schema.

    Parameters
    ----------
    frame : pd.DataFrame
        Incoming inference request.
    feature_columns : list[str]
        Exact model input columns from training.

    Raises
    ------
    ValueError
        If ``frame`` is not a DataFrame, misses required columns, or carries
        non-numeric values in a feature column.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"predict: expected pandas DataFrame, got {type(frame).__name__}")

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"predict: missing required columns {missing}")

    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f"predict: column '{col}' must be numeric, got {frame[col].dtype}")


class DemandForecaster(mlflow.pyfunc.PythonModel):
    """PyFunc-inference wrapper enforcing the feature schema on every request.

    Parameters
    ----------
    feature_columns : list[str]
        Exact design-matrix columns the underlying model was trained on.
    """

    def __init__(self, feature_columns: list[str]):
        self.feature_columns = list(feature_columns)

    def load_context(self, context) -> None:
        """Load the serialized estimator referenced by the MLflow artifacts."""
        if not hasattr(context, "artifacts") or "model" not in getattr(context, "artifacts", {}):
            raise ValueError("DemandForecaster: missing 'model' artifact in load_context.")
        model_path = context.artifacts["model"]
        self._model = joblib.load(model_path)
        LOGGER.info("DemandForecaster.load_context: estimator loaded from %s", model_path)

    def predict(self, context, model_input) -> pd.DataFrame:
        """Predict weekly demand (units) with strict schema validation.

        Parameters
        ----------
        context : model_context
            MLflow model context (unused beyond artifact loading).
        model_input : pd.DataFrame
            Inference rows with the exact training feature schema.

        Returns
        -------
        pd.DataFrame
            Single ``pred_demand`` column on the business (unit) scale.
        """
        validate_input(model_input, self.feature_columns)
        X = model_input[self.feature_columns].astype("float64")
        pred_log = self._model.predict(X)
        # Inference contract: expm1 + physical floor (demand >= 0, finite).
        pred_demand = inverse_log1p(pred_log)
        return pd.DataFrame({"pred_demand": np.asarray(pred_demand, dtype=float)})


def save_winner(
    model,
    feature_columns: list[str],
    artifact_dir: str,
    model_name: str = "demand_forecaster",
) -> mlflow.pyfunc.PyFuncModel:
    """Package a winner estimator as an MLflow PyFunc model artifact.

    Parameters
    ----------
    model : estimator
        Trained LightGBM / XGBoost estimator with a ``predict`` method.
    feature_columns : list[str]
        Design-matrix column order recorded at training time.
    artifact_dir : str
        Directory where the joblib artifact is staged before logging.
    model_name : str, optional
        Model name for MLflow logging (default ``demand_forecaster``).

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        The logged model handle for downstream loading.
    """
    import os
    import tempfile
    from pathlib import Path

    artifact_dir = os.path.abspath(artifact_dir)
    joblib.dump(model, os.path.join(artifact_dir, "model.joblib"))

    project_root = Path(__file__).resolve().parents[2]
    code_paths = [str(project_root / "src" / p) for p in ("utils", "models", "data", "features")]

    wrapper = DemandForecaster(feature_columns)
    LOGGER.info("save_winner: logging PyFunc model '%s'.", model_name)
    return mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=wrapper,
        artifacts={"model": os.path.join(artifact_dir, "model.joblib")},
        code_paths=code_paths,
    )