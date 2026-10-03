"""Part-4 demand baseline (consumes the Part-3 Production forecast).

Architecture separation (Blueprint + project contract):

* the Part-3 Production model (``models:/demand_forecasting_final/2``) supplies the
  **demand baseline** ``Q0`` at the current price — the forecasting model is never
  modified or retrained here;
* the Part-4 modules apply only the price elasticity demand response on top.

The baseline is computed for a reference week from the **frozen 93-column design**:
the caller supplies a feature frame (e.g. the frozen lifecycle ``prepared`` cache or a
rebuilt frozen feature matrix). This module never reads ``data/splits/test.csv`` and
never touches the Test period: the default reference week is the last full Part-4
validation week, strictly before the test boundary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.final_forecast import (
    IDENTITY_COLS,
    freeze_schema,
)
from src.optimization.pricing_data import LAST_VAL_WEEK, PRODUCT_COL, WEEK_COL
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PRODUCTION_MODEL_URI = "models:/demand_forecasting_final/2"


def schema_from_frame(frame: pd.DataFrame) -> list[str]:
    """Ordered 93-column frozen schema present in a prepared frame."""
    schema = [c for c in frame.columns if c not in IDENTITY_COLS]
    if len(schema) != 93:
        raise ValueError(f"demand_baseline: expected 93 feature columns, got {len(schema)}")
    return schema


def load_production_model(uri: str = PRODUCTION_MODEL_URI):
    """Load the Part-3 Production PyFunc model (same tracking backend as Part 3)."""
    import mlflow

    from src.models.registry import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    LOGGER.info("demand_baseline: loading production model %s.", uri)
    return mlflow.pyfunc.load_model(uri)


def baseline_frame_for_week(
    prepared: dict[str, pd.DataFrame] | Path,
    reference_week: str = LAST_VAL_WEEK,
) -> pd.DataFrame:
    """Extract the frozen design rows for a reference week from prepared splits.

    ``prepared`` may be the frozen-lifecycle prepared dict (``{train, val, test}``)
    or a path to the cached joblib. Only ``val`` is used (the reference week is a
    Part-4 validation week before the test boundary).
    """
    if isinstance(prepared, (str, Path)):
        import joblib

        prepared = joblib.load(prepared)
    val = prepared["val"]
    schema = freeze_schema(val)
    rows = val[val[WEEK_COL].astype(str).str.startswith(reference_week)].copy()
    if rows.empty:
        raise ValueError(
            f"demand_baseline: no rows for reference week {reference_week} in the "
            f"prepared val frame (validate the week label)."
        )
    rows = rows[[PRODUCT_COL, WEEK_COL, *schema]].copy()
    rows[WEEK_COL] = rows[WEEK_COL].astype(str)
    return rows


def frozen_features_for_week(
    matrix: pd.DataFrame,
    reference_week: str,
    model=None,
) -> pd.DataFrame:
    """Build the frozen 93-column design rows for a reference week from the matrix.

    Fallback used when the frozen-lifecycle ``prepared`` cache is unavailable:
    runs the *exact* Phase-3 feature stack (``build_frozen_features``) over the
    weekly matrix and slices the reference week. Feeds are strictly past-only
    (lags / rolling / regime are computed on history only), no Test split is
    read, and any remaining NaN in the model's input columns is filled with
    **train-only** column statistics (weeks before ``VAL_START_WEEK``) per the
    imputation contract. The one-hot vocabulary is the Production model's own
    signature when available (source of truth); otherwise the train-region
    categories.
    """
    from src.models.final_forecast import REGIME_COLS, build_frozen_features
    from src.models.data_prep import FEATURE_COLS, prepare_model_matrix
    from src.data.target_engineering import load_demand_matrix
    from src.optimization import pricing_data
    from src.optimization.pricing_data import VAL_START_WEEK


    m = load_demand_matrix(pricing_data.MATRIX_PATH)
    built = build_frozen_features(m)
    wk = built[WEEK_COL].astype(str).str.replace(
        r"\s*/.*", "", regex=True
    )
    rows = built[wk.str.startswith(reference_week)].copy()
    if rows.empty:
        raise ValueError(
            f"demand_baseline: no matrix rows for reference week {reference_week}."
        )
    train_region = built[wk < VAL_START_WEEK]
    for col in FEATURE_COLS:
        if col in rows.columns and rows[col].isna().any():
            fill = float(train_region[col].mean()) if col in train_region.columns else 0.0
            rows[col] = rows[col].fillna(fill)

    categories = None
    if model is not None:
        sig = getattr(getattr(model, "metadata", None), "signature", None)
        if sig is not None and sig.inputs is not None:
            ohe = [c.name for c in sig.inputs if c.name.startswith("cat_")]
            if ohe:
                categories = sorted(c.replace("cat_", "", 1) for c in ohe)
    if categories is None:
        categories = sorted(
            set(train_region["product_category_name"].astype(str).unique()) | {"unknown"}
        )
    approved = set(categories) | {"unknown"}
    rows["product_category_name"] = rows["product_category_name"].astype(str).where(
        rows["product_category_name"].astype(str).isin(approved), "unknown"
    )
    result = prepare_model_matrix(rows, categories=categories)
    result[REGIME_COLS] = rows[REGIME_COLS].reset_index(drop=True).astype("float64")
    return result[[PRODUCT_COL, WEEK_COL, *schema_from_frame(result)]]


def predict_baseline(model, features: pd.DataFrame) -> pd.DataFrame:
    """Predict unit-scale baseline demand for the frozen feature rows."""
    import numpy as np

    schema = [c for c in features.columns if c not in IDENTITY_COLS]
    # The Production model's schema enforces float64; prepared frames carry
    # int64 calendar features (week_of_year, month, ...) that must be cast.
    pred = model.predict(features[schema].astype("float64"))
    out = features[[PRODUCT_COL, WEEK_COL]].copy().reset_index(drop=True)
    pred_values = (
        pred["pred_demand"].to_numpy(dtype=float)
        if isinstance(pred, pd.DataFrame)
        else np.asarray(pred, dtype=float)
    )
    out["baseline_demand"] = pred_values
    return out


def attach_current_price(baseline: pd.DataFrame, matrix_ref: pd.DataFrame) -> pd.DataFrame:
    """Merge observed weekly price (and category) for the reference week onto baselines.

    ``matrix_ref`` = matrix rows for the reference week with a positive observed price.
    Products without a price observation for that week are left with ``current_price=NaN``
    and are handled by the engine (dropped with an explicit reason).
    """
    key = [PRODUCT_COL]
    m = matrix_ref[[PRODUCT_COL, "avg_price", "product_category_name"]].rename(
        columns={"avg_price": "current_price", "product_category_name": "category"}
    )
    return baseline.merge(m, on=key, how="left")