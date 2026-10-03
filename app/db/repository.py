"""Repository helpers: turn endpoint outcomes into persisted records.

This module is the single place that maps API results onto the inference log
schema. Persistence failures are logged and reported back as
``inference_logged: false`` — never as a silent drop.
"""

from __future__ import annotations

import logging

from app.db.database import InferenceStore
from app.db.models import InferenceRecord
from app.errors import PersistenceError

logger = logging.getLogger("deploy.repository")


def build_success_record(
    *,
    request_id: str,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    model_name: str | None = None,
    model_version: int | None = None,
    product_id: str | None = None,
    week: str | None = None,
    prediction: float | None = None,
    cost_supplied: bool | None = None,
    pricing_result: dict | None = None,
    elasticity_status: str | None = None,
    elasticity_source: str | None = None,
    recommended_price: float | None = None,
    profit_computable: bool | None = None,
) -> InferenceRecord:
    return InferenceRecord(
        request_id=request_id,
        endpoint=endpoint,
        status_code=status_code,
        latency_ms=round(latency_ms, 3),
        model_name=model_name,
        model_version=model_version,
        product_id=product_id,
        week=week,
        prediction=prediction,
        cost_supplied=cost_supplied,
        pricing_result=pricing_result,
        elasticity_status=elasticity_status,
        elasticity_source=elasticity_source,
        recommended_price=recommended_price,
        profit_computable=profit_computable,
    )


def build_error_record(
    *,
    request_id: str,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    error_type: str | None = None,
    error_message: str | None = None,
    product_id: str | None = None,
    week: str | None = None,
) -> InferenceRecord:
    return InferenceRecord(
        request_id=request_id,
        endpoint=endpoint,
        status_code=status_code,
        latency_ms=round(latency_ms, 3),
        product_id=product_id,
        week=week,
        error_type=error_type,
        error_message=error_message,
    )


def persist(store: InferenceStore, record: InferenceRecord) -> bool:
    """Persist the record. Return False and log on failure (never raise)."""
    try:
        store.insert(record)
        return True
    except PersistenceError:
        logger.exception(
            "could not persist inference record request_id=%s endpoint=%s",
            record.request_id,
            record.endpoint,
        )
        return False