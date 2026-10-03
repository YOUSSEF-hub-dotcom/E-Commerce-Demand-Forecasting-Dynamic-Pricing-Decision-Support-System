"""Explanation endpoint: SHAP attribution for a product/week prediction."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from app.db.repository import build_error_record, build_success_record, persist
from app.dependencies import get_state, get_store, request_id
from app.errors import AppError
from app.schemas import (
    ExplanationAttribution,
    ExplainResponse,
    ExplanationRequest,
    InferenceMeta,
)
from app.services.explainability_service import explain_row
from app.services.state import AppState

logger = logging.getLogger("deploy.api.explain")
router = APIRouter(tags=["explanation"])


def _attrs(rows: list[dict]) -> list[ExplanationAttribution]:
    return [ExplanationAttribution(**r) for r in rows]


@router.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplanationRequest, request: Request) -> ExplainResponse:
    state: AppState = get_state(request)
    state.require_ready()
    rid = request_id(request)
    start = time.perf_counter()
    top_n = min(state.settings.shap_top_n, 40)
    try:
        week_label = state.validate_week(payload.week)
        outcome = explain_row(state, payload.product_id, week_label, top_n=top_n)
    except AppError as exc:
        persist(
            get_store(request),
            build_error_record(
                request_id=rid,
                endpoint="/explain",
                status_code=exc.status_code,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type=exc.code,
                error_message=exc.message,
                product_id=payload.product_id,
            ),
        )
        raise
    except Exception as exc:
        logger.exception("explain unexpected error")
        persist(
            get_store(request),
            build_error_record(
                request_id=rid,
                endpoint="/explain",
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
            endpoint="/explain",
            status_code=200,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_name=outcome["model"]["name"],
            model_version=outcome["model"]["version"],
            product_id=outcome["product_id"],
            week=outcome["week"],
            prediction=outcome["predicted_demand_units"],
        ),
    )
    return ExplainResponse(
        product_id=outcome["product_id"],
        week=outcome["week"],
        predicted_demand_units=outcome["predicted_demand_units"],
        base_log1p=outcome["base_log1p"],
        expected_value_log1p=outcome["expected_value_log1p"],
        sum_shap_log1p=outcome["sum_shap_log1p"],
        additivity_residual_demand_units=outcome["additivity_residual_demand_units"],
        attribution_of=outcome["attribution_of"],
        disclaimer=outcome["disclaimer"],
        reconstruction_block=outcome["reconstruction_block"],
        top_positive=_attrs(outcome["top_positive"]),
        top_negative=_attrs(outcome["top_negative"]),
        feature_count=outcome["feature_count"],
        model=outcome["model"],
        inference=InferenceMeta(request_id=rid, inference_logged=logged),
    )