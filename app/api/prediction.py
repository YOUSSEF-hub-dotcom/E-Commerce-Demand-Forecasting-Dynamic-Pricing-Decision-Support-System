"""Prediction endpoint: demand forecast for a product/week."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from app.db.repository import build_error_record, build_success_record, persist
from app.dependencies import get_state, get_store, request_id
from app.errors import AppError
from app.schemas import InferenceMeta, PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_row
from app.services.state import AppState

logger = logging.getLogger("deploy.api.predict")
router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    state: AppState = get_state(request)
    state.require_ready()
    rid = request_id(request)
    start = time.perf_counter()
    try:
        week_label = state.validate_week(payload.week)
        outcome = predict_row(state, payload.product_id, week_label)
    except AppError as exc:
        persist(
            get_store(request),
            build_error_record(
                request_id=rid,
                endpoint="/predict",
                status_code=exc.status_code,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type=exc.code,
                error_message=exc.message,
                product_id=payload.product_id,
            ),
        )
        raise
    except Exception as exc:
        logger.exception("predict unexpected error")
        persist(
            get_store(request),
            build_error_record(
                request_id=rid,
                endpoint="/predict",
                status_code=500,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type="INTERNAL_ERROR",
                error_message=str(exc),
                product_id=payload.product_id,
            ),
        )
        raise

    logged = persist(
        get_store(request),
        build_success_record(
            request_id=rid,
            endpoint="/predict",
            status_code=200,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_name=outcome["model"]["name"],
            model_version=outcome["model"]["version"],
            product_id=outcome["product_id"],
            week=outcome["week"],
            prediction=outcome["predicted_demand"],
        ),
    )
    return PredictionResponse(
        product_id=outcome["product_id"],
        week=outcome["week"],
        predicted_demand=outcome["predicted_demand"],
        model=outcome["model"],
        inference=InferenceMeta(request_id=rid, inference_logged=logged),
    )