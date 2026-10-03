"""Per-product SHAP explainability for the Production model's prediction.

Attribution is computed on the model's log1p domain (the stored LightGBM
estimator target); the served unit-scale prediction is reconstructed exactly as
``units = expm1(base_log1p + sum(shap))`` with a per-row, additivity-corrected
base — the same convention verified in Part 5 (``check_additivity=False``).
Technical feature codes are mapped to business labels so that the API never
exposes the 93-feature engineering schema.
"""

from __future__ import annotations

import numpy as np

from app.errors import AppError
from app.services.prediction_service import model_summary
from app.services.state import PRODUCT_COL, AppState

REGIME_PREFIX = "regime_"
CAT_PREFIX = "cat_"

FEATURE_LABELS = {
    "price_prev": "Last observed price",
    "price_chg_1w": "Week-over-week price change",
    "freight_prev": "Last observed freight cost",
    "category_share_prev": "Category share of weekly orders",
    "week_of_year": "Week of year",
    "month": "Calendar month",
    "quarter": "Quarter",
    "year": "Calendar year",
}


def _label(feature_code: str) -> str:
    if feature_code in FEATURE_LABELS:
        return FEATURE_LABELS[feature_code]
    if feature_code.startswith("demand_lag_"):
        try:
            lag = int(feature_code.removeprefix("demand_lag_"))
            return f"Orders {lag} week(s) in the past"
        except ValueError:
            pass
    if feature_code.startswith("rolling_demand_mean_4w"):
        return "4-week average orders"
    if feature_code.startswith("rolling_demand_std_4w"):
        return "4-week order volatility"
    if feature_code.startswith(REGIME_PREFIX):
        window = feature_code.removeprefix(REGIME_PREFIX)
        return f"Long-run average order regime ({window})"
    if feature_code.startswith(CAT_PREFIX):
        cat = feature_code.removeprefix(CAT_PREFIX)
        return f"Category: {cat}"
    return feature_code.replace("_", " ").title()


def _contributor_rows(entries: list) -> list[dict]:
    out = []
    for code, value in entries:
        out.append(
            {
                "feature": _label(code),
                "feature_code": code,
                "contribution_log1p": float(value),
                "contribution_demand_units": float(np.expm1(value) - np.expm1(0)),
            }
        )
    return out


def explain_row(state: AppState, product_id: str, week_label: str, top_n: int) -> dict:
    from app.services.prediction_service import _product_row

    pid = state.validate_product(product_id)
    row = _product_row(state, pid, week_label)
    if state.explainer is None:
        raise AppError(
            503,
            "MODEL_NOT_READY",
            "SHAP explainer is not ready (see /ready)",
        )

    log_pred = float(np.asarray(state.estimator.predict(row), dtype=float).ravel()[0])
    if not np.isfinite(log_pred):
        raise AppError(503, "FEATURE_BUILD_FAILED", "non-finite model log prediction")
    units = float(np.clip(np.expm1(log_pred), 0.0, None))

    explanation = state.explainer(row, check_additivity=False)
    values = np.asarray(explanation.values, dtype=float).ravel()
    if values.shape != (len(state.schema),):
        raise AppError(
            503,
            "FEATURE_BUILD_FAILED",
            f"SHAP values shape {values.shape} != feature schema {len(state.schema)}",
        )
    if not np.isfinite(values).all():
        raise AppError(503, "FEATURE_BUILD_FAILED", "non-finite SHAP values produced")

    phi_sum = float(values.sum())
    base_log1p = log_pred - phi_sum
    served_units = float(np.clip(np.expm1(base_log1p + phi_sum), 0.0, None))
    residual_units = float(served_units - units)

    order = np.argsort(np.abs(values))[::-1]
    pos = sorted(
        ((state.schema[j], values[j]) for j in order if values[j] > 0),
        key=lambda t: t[1],
        reverse=True,
    )
    neg = sorted(
        ((state.schema[j], values[j]) for j in order if values[j] < 0),
        key=lambda t: t[1],
    )

    reconstruction_block = (
        "served_units = expm1(base_log1p + sum(shap_log1p))\n"
        f"  = expm1({base_log1p:.6f} + {phi_sum:.6f})\n"
        f"  = {served_units:.6f}\n"
        f"model.predict units = {units:.6f}\n"
        f"additivity residual (units) = {residual_units:.6f}"
    )

    return {
        "product_id": pid,
        "week": week_label,
        "predicted_demand_units": units,
        "base_log1p": base_log1p,
        "expected_value_log1p": state.explainer_expected_log1p,
        "sum_shap_log1p": phi_sum,
        "additivity_residual_demand_units": residual_units,
        "attribution_of": (
            f"predicted demand for product {pid} in week {week_label} at the observed price"
        ),
        "disclaimer": (
            "SHAP attributions describe feature contributions to the forecast; they are "
            "attribution, not evidence of causality."
        ),
        "reconstruction_block": reconstruction_block,
        "top_positive": _contributor_rows(pos[:top_n]),
        "top_negative": _contributor_rows(neg[:top_n]),
        "feature_count": len(state.schema),
        "model": model_summary(state),
    }