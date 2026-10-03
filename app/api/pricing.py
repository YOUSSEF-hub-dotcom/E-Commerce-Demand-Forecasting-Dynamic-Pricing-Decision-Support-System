"""Pricing endpoint: recommended price for a product/week (cost optional)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from app.db.repository import build_error_record, build_success_record, persist
from app.dependencies import get_state, get_store, request_id
from app.errors import AppError
from app.schemas import (
    GuardrailReport,
    InferenceMeta,
    PricingRequest,
    PricingResponse,
    ScenarioRow,
)
from app.services.pricing_service import pricing_row
from app.services.state import AppState

logger = logging.getLogger("deploy.api.pricing")
router = APIRouter(tags=["pricing"])


@router.post("/pricing", response_model=PricingResponse)
def pricing(payload: PricingRequest, request: Request) -> PricingResponse:
    state: AppState = get_state(request)
    state.require_ready()
    rid = request_id(request)
    start = time.perf_counter()
    try:
        week_label = state.validate_week(payload.week)
        outcome = pricing_row(state, payload.product_id, week_label, cost=payload.cost)
    except AppError as exc:
        persist(
            get_store(request),
            build_error_record(
                request_id=rid,
                endpoint="/pricing",
                status_code=exc.status_code,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type=exc.code,
                error_message=exc.message,
                product_id=payload.product_id,
            ),
        )
        raise
    except Exception as exc:
        logger.exception("pricing unexpected error")
        persist(
            get_store(request),
            build_error_record(
                request_id=rid,
                endpoint="/pricing",
                status_code=500,
                latency_ms=(time.perf_counter() - start) * 1000,
                error_type="INTERNAL_ERROR",
                error_message=str(exc),
                product_id=payload.product_id,
            ),
        )
        raise

    guardrails = GuardrailReport(
        margin_floor_applied=outcome["guardrails"]["margin_floor_applied"]
        if outcome.get("guardrails")
        else False,
        margin_floor_inactive_no_cost=outcome["guardrails"]["margin_floor_inactive_no_cost"]
        if outcome.get("guardrails")
        else False,
        ceiling_applied=outcome["guardrails"]["ceiling_applied"]
        if outcome.get("guardrails")
        else False,
        floor_applied=outcome["guardrails"]["floor_applied"]
        if outcome.get("guardrails")
        else False,
        smoothed=outcome["guardrails"]["smoothed"]
        if outcome.get("guardrails")
        else False,
        notes=list(outcome["guardrails"]["notes"])
        if outcome.get("guardrails")
        else [],
    )

    response = PricingResponse(
        product_id=outcome["product_id"],
        week=outcome["week"],
        current_price=outcome.get("current_price"),
        baseline_demand=outcome.get("baseline_demand"),
        category=outcome.get("category"),
        elasticity_status=outcome.get("elasticity_status"),
        elasticity_source=outcome.get("elasticity_source"),
        elasticity=outcome.get("elasticity"),
        reason=outcome["reason"],
        profit_computable=outcome["profit_computable"],
        objective=outcome.get("objective"),
        optimal_price=outcome.get("optimal_price"),
        optimal_multiplier=outcome.get("optimal_multiplier"),
        optimal_revenue=outcome.get("optimal_revenue"),
        optimal_profit=outcome.get("optimal_profit"),
        recommended_price=outcome.get("recommended_price"),
        extrapolation_required=outcome.get("extrapolation_required"),
        guardrails=guardrails if outcome.get("guardrails") else None,
        scenarios=[ScenarioRow(**s) for s in outcome.get("scenarios", [])],
        model=outcome["model"],
        inference=InferenceMeta(request_id=rid, inference_logged=False),
    )

    logged = persist(
        get_store(request),
        build_success_record(
            request_id=rid,
            endpoint="/pricing",
            status_code=200,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_name=outcome["model"]["name"],
            model_version=outcome["model"]["version"],
            product_id=outcome["product_id"],
            week=outcome["week"],
            prediction=outcome.get("baseline_demand"),
            cost_supplied=payload.cost is not None,
            pricing_result={
                "reason": outcome["reason"],
                "objective": outcome.get("objective"),
                "elasticity_status": outcome.get("elasticity_status"),
                "elasticity_source": outcome.get("elasticity_source"),
                "optimal_price": outcome.get("optimal_price"),
                "recommended_price": outcome.get("recommended_price"),
                "profit_computable": outcome["profit_computable"],
                "extrapolation_required": outcome.get("extrapolation_required"),
            },
            elasticity_status=outcome.get("elasticity_status"),
            elasticity_source=outcome.get("elasticity_source"),
            recommended_price=outcome.get("recommended_price"),
            profit_computable=outcome["profit_computable"],
        ),
    )
    response.inference = InferenceMeta(request_id=rid, inference_logged=logged)
    return response