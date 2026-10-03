"""Demand prediction service (business-scale demand forecast)."""

from __future__ import annotations

import math
import numpy as np

from app.errors import FEATURE_BUILD_FAILED, AppError
from app.services.state import (
    PRODUCT_COL,
    AppState,
    predict_units,
)


def model_summary(state: AppState) -> dict:
    return {
        "name": state.model_info["name"],
        "version": state.model_info["version"],
        "stage": state.model_info["stage"],
        "status": state.model_info["status"],
        "run_id": state.model_info["run_id"],
        "source": state.model_info["source"],
    }


def _product_row(state: AppState, product_id: str, week_label: str):
    frame = state.frame_for_week(week_label)
    match = frame.loc[frame[PRODUCT_COL].astype(str) == product_id]
    if match.empty:
        raise AppError(
            422,
            "PRODUCT_NOT_FOUND",
            f"product {product_id!r} has no feature row in week {week_label}",
        )
    row = match.iloc[[0]][state.schema].astype("float64")
    return row


def predict_row(state: AppState, product_id: str, week_label: str) -> dict:
    row = _product_row(state, product_id, week_label)
    try:
        value = predict_units(state.model, row)
    except Exception as exc:
        raise AppError(
            503, FEATURE_BUILD_FAILED, f"prediction failed for {product_id}: {exc}"
        ) from exc
    if not math.isfinite(value):
        raise AppError(
            503,
            FEATURE_BUILD_FAILED,
            f"non-finite prediction for product {product_id}",
        )
    return {
        "product_id": product_id,
        "week": week_label,
        "predicted_demand": float(value),
        "model": model_summary(state),
    }